"""Orchestrator for coordinating comparison evaluation."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import structlog
from mcp_optimizer_agent import McpOptimizerAgentRunner
from metrics import MetricsComputer
from models import ComparisonReport, ComparisonResult, TestCase, TestDataset
from native_approach import NativeApproachRunner
from results_exporter import ResultsExporter
from tool_converter import ToolConverter

logger = structlog.get_logger(__name__)


class ComparisonOrchestrator:
    """Orchestrates comparison between Native and MCP Optimizer approaches."""

    def __init__(
        self,
        dataset_path: Path,
        anthropic_api_key: str,
        openrouter_api_key: str,
        limit: int | None = None,
        max_concurrency: int = 10,
        llm_model: str = "anthropic/claude-sonnet-4",
        resume_from: Path | None = None,
        output_file: Path | None = None,
    ):
        """Initialize orchestrator.

        Args:
            dataset_path: Path to test dataset JSON
            anthropic_api_key: Anthropic API key
            openrouter_api_key: OpenRouter API key
            limit: Optional limit on test cases
            max_concurrency: Max concurrent test executions
            llm_model: LLM model for MCP Optimizer agent
            resume_from: Optional path to existing results JSON to resume from
            output_file: Optional path to output file for saving partial results
        """
        self.dataset_path = dataset_path
        self.anthropic_api_key = anthropic_api_key
        self.openrouter_api_key = openrouter_api_key
        self.limit = limit
        self.max_concurrency = max_concurrency
        self.llm_model = llm_model
        self.resume_from = resume_from
        self.output_file = output_file
        self.completed_count = 0
        self.total_count = 0
        self.baseline_all_tools_tokens: int | None = None

    def dry_run(self) -> bool:
        """Validate setup without running comparison.

        Returns:
            True if all validations pass

        Raises:
            ValueError: If any validation fails
        """
        logger.info("Running dry-run validation")

        # 1. Check dataset exists and can be loaded
        try:
            logger.info("Validating dataset", path=str(self.dataset_path))
            if not self.dataset_path.exists():
                raise ValueError(f"Dataset not found: {self.dataset_path}")

            dataset = TestDataset.load_from_json(self.dataset_path, self.limit)
            logger.info(
                "Dataset valid",
                total_cases=len(dataset.cases),
                limited_to=self.limit or "all",
            )
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}") from e

        # 2. Check MCP tools file exists
        mcp_tools_path = self.dataset_path.parent / "mcp_tools_cleaned.json"
        logger.info("Validating MCP tools file", path=str(mcp_tools_path))
        if not mcp_tools_path.exists():
            raise ValueError(f"MCP tools file not found: {mcp_tools_path}")

        # 3. Try to load and convert tools
        try:
            all_tools = self._load_and_convert_tools()
            logger.info("MCP tools loaded and converted", count=len(all_tools))
        except Exception as e:
            raise ValueError(f"Failed to load/convert MCP tools: {e}") from e

        # 4. Check database path is accessible
        test_db_path = Path(__file__).parent / "mcp_optimizer_test.db"
        logger.info("Validating database path", path=str(test_db_path))
        if test_db_path.exists():
            logger.info("Database file exists")
        else:
            logger.info("Database will be created at", path=str(test_db_path))

        # 5. Check output file path is writable
        if self.output_file:
            logger.info("Validating output path", path=str(self.output_file))
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Output directory is writable")

        # 6. Check resume file if specified
        if self.resume_from:
            logger.info("Validating resume file", path=str(self.resume_from))
            if not self.resume_from.exists():
                raise ValueError(f"Resume file not found: {self.resume_from}")
            try:
                existing_results = self._load_existing_results()
                logger.info("Resume file valid", count=len(existing_results))
            except Exception as e:
                raise ValueError(f"Failed to load resume file: {e}") from e

        logger.info("Dry-run validation passed")
        return True

    async def run(self) -> ComparisonReport:
        """Execute full comparison workflow.

        Returns:
            ComparisonReport with results
        """
        logger.info("Starting comparison", dataset=str(self.dataset_path))

        # 1. Load test dataset
        dataset = TestDataset.load_from_json(self.dataset_path, self.limit)
        self.total_count = len(dataset.cases)
        logger.info(
            "Loaded test dataset",
            total_cases=self.total_count,
        )

        # 2. Load existing results if resuming
        existing_results = {}
        if self.resume_from:
            existing_results = self._load_existing_results()
            logger.info(
                "Loaded existing results",
                count=len(existing_results),
                successful=sum(
                    1 for r in existing_results.values() if self._result_fully_succeeded(r)
                ),
                errored=sum(1 for r in existing_results.values() if self._result_has_errors(r)),
            )
            # Check if baseline was already measured
            try:
                with open(self.resume_from) as f:
                    data = json.load(f)
                self.baseline_all_tools_tokens = data.get("baseline_all_tools_tokens")
                if self.baseline_all_tools_tokens:
                    logger.info(
                        "Using existing baseline measurement",
                        tokens=self.baseline_all_tools_tokens,
                    )
            except Exception:
                pass

        # 3. Load and convert MCP tools for native approach
        logger.info("Loading and converting MCP tools")
        all_tools = self._load_and_convert_tools()
        logger.info("Converted tools", count=len(all_tools))

        # 4. Initialize runners
        logger.info("Initializing runners")
        native_runner = NativeApproachRunner(api_key=self.anthropic_api_key, all_tools=all_tools)

        # 5. Measure baseline token usage (only if not already measured)
        if self.baseline_all_tools_tokens is None:
            logger.info("Measuring baseline token usage with all tools loaded")
            self.baseline_all_tools_tokens = await native_runner.measure_baseline_all_tools_tokens()

        # Use test database in examples/anthropic_comparison directory
        test_db_path = Path(__file__).parent / "mcp_optimizer_test.db"
        mcp_agent_runner = McpOptimizerAgentRunner(
            llm_model=self.llm_model, test_db_path=test_db_path
        )

        # 6. Run comparison on all test cases
        logger.info("Running comparison on test cases")
        results = await self._run_comparison(
            dataset.cases, native_runner, mcp_agent_runner, existing_results
        )

        # 7. Compute aggregate metrics
        logger.info("Computing aggregate metrics")
        aggregate_metrics = MetricsComputer.compute_aggregate_metrics(results)

        # 8. Return report
        report = ComparisonReport(
            aggregate_metrics=aggregate_metrics,
            individual_results=results,
            dataset_path=str(self.dataset_path),
            num_test_cases=len(dataset.cases),
            timestamp=datetime.now().isoformat(),
            mcp_optimizer_model=self.llm_model,
            baseline_all_tools_tokens=self.baseline_all_tools_tokens,
        )

        logger.info("Comparison complete")
        return report

    def _load_existing_results(self) -> dict[str, "ComparisonResult"]:
        """Load existing results from JSON file.

        Returns:
            Dictionary mapping query to ComparisonResult
        """
        if not self.resume_from or not self.resume_from.exists():
            return {}

        try:
            with open(self.resume_from) as f:
                data = json.load(f)

            from models import ComparisonReport

            report = ComparisonReport.model_validate(data)

            # Create query -> result mapping
            return {result.test_case.query: result for result in report.individual_results}

        except Exception as e:
            logger.error(
                "Failed to load existing results", error=str(e), path=str(self.resume_from)
            )
            return {}

    def _save_partial_results(self, results: list[ComparisonResult]) -> None:
        """Save partial results to output file.

        Args:
            results: List of comparison results to save
        """
        if not self.output_file or not results:
            return

        try:
            # Compute aggregate metrics for current results
            aggregate_metrics = MetricsComputer.compute_aggregate_metrics(results)

            # Create partial report
            report = ComparisonReport(
                aggregate_metrics=aggregate_metrics,
                individual_results=results,
                dataset_path=str(self.dataset_path),
                num_test_cases=self.total_count,
                timestamp=datetime.now().isoformat(),
                mcp_optimizer_model=self.llm_model,
                baseline_all_tools_tokens=self.baseline_all_tools_tokens,
            )

            # Save to file
            ResultsExporter.save_json_report(report, self.output_file)
            logger.info(
                "Saved partial results",
                completed=len(results),
                total=self.total_count,
                path=str(self.output_file),
            )

        except Exception as e:
            logger.error("Failed to save partial results", error=str(e))

    def _get_progress_string(self) -> str:
        """Get formatted progress string.

        Returns:
            Progress string in format "completed/total (percentage%)"
        """
        progress_pct = (self.completed_count / self.total_count) * 100
        return f"{self.completed_count}/{self.total_count} ({progress_pct:.1f}%)"

    @staticmethod
    def _result_fully_succeeded(result: ComparisonResult) -> bool:
        """Check if all approaches in a comparison result succeeded.

        Args:
            result: Comparison result to check

        Returns:
            True if all native and MCP approaches succeeded (no errors)
        """
        all_native_succeeded = all(nr.error is None for nr in result.native_results.values())
        mcp_succeeded = result.mcp_optimizer_result.error is None
        return all_native_succeeded and mcp_succeeded

    @staticmethod
    def _result_has_errors(result: ComparisonResult) -> bool:
        """Check if any approach in a comparison result failed.

        Args:
            result: Comparison result to check

        Returns:
            True if any native or MCP approach has an error
        """
        any_native_failed = any(nr.error is not None for nr in result.native_results.values())
        mcp_failed = result.mcp_optimizer_result.error is not None
        return any_native_failed or mcp_failed

    def _load_and_convert_tools(self) -> list[dict]:
        """Load MCP tools and convert to Anthropic format.

        Returns:
            List of tools in Anthropic format
        """
        # Load from same directory as dataset
        mcp_tools_path = self.dataset_path.parent / "mcp_tools_cleaned.json"

        with open(mcp_tools_path) as f:
            mcp_servers = json.load(f)

        converter = ToolConverter()
        all_tools = []

        for server in mcp_servers:
            for tool in server["tools"]:
                anthropic_tool = converter.convert_tool(server, tool)
                all_tools.append(anthropic_tool.model_dump())

        return all_tools

    def _handle_native_exceptions(self, native_results: dict, test_case: TestCase) -> dict:
        """Handle exceptions in native search results.

        Args:
            native_results: Dictionary of native approach results
            test_case: Test case being run

        Returns:
            Dictionary with exceptions converted to error results
        """
        from models import NativeSearchResult

        handled_results = {}
        for approach, result in native_results.items():
            if isinstance(result, Exception):
                logger.error(
                    "Native search failed",
                    error=str(result),
                    query=test_case.query,
                    approach=approach,
                )
                handled_results[approach] = NativeSearchResult(
                    test_case=test_case,
                    approach=approach,
                    selected_tool_name=None,
                    retrieved_tools=[],
                    num_tools_returned=0,
                    search_time_s=0.0,
                    request_tokens=0,
                    response_tokens=0,
                    total_tokens=0,
                    error=str(result),
                )
            else:
                handled_results[approach] = result
        return handled_results

    async def _retry_failed_approaches(
        self,
        test_case: TestCase,
        existing: ComparisonResult,
        native_runner: NativeApproachRunner,
        mcp_runner: McpOptimizerAgentRunner,
    ) -> tuple[dict, any]:
        """Retry only the failed approaches from an existing result.

        Args:
            test_case: Test case to run
            existing: Existing comparison result with some failures
            native_runner: Native approach runner
            mcp_runner: MCP Optimizer agent runner

        Returns:
            Tuple of (native_results dict, mcp_result)
        """
        # Determine which runners to execute
        tasks = []
        native_approaches_to_run = []

        for approach in native_runner.APPROACHES:
            if (
                approach not in existing.native_results
                or existing.native_results[approach].error is not None
            ):
                native_approaches_to_run.append(approach)
                tasks.append(native_runner.search_tool(test_case, approach))

        if existing.mcp_optimizer_result.error is not None:
            tasks.append(mcp_runner.search_tool(test_case))
            need_mcp = True
        else:
            need_mcp = False

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Parse results
        native_results = {}
        approach_idx = 0
        for approach in native_runner.APPROACHES:
            if approach in native_approaches_to_run:
                native_results[approach] = results[approach_idx]
                approach_idx += 1
            else:
                native_results[approach] = existing.native_results[approach]

        if need_mcp:
            mcp_result = results[approach_idx]
        else:
            mcp_result = existing.mcp_optimizer_result

        return native_results, mcp_result

    def _handle_mcp_exception(self, mcp_result, test_case: TestCase):
        """Handle exception in MCP search result.

        Args:
            mcp_result: MCP search result or exception
            test_case: Test case being run

        Returns:
            Valid MCP result with error if exception occurred
        """
        if isinstance(mcp_result, Exception):
            logger.error("MCP search failed", error=str(mcp_result), query=test_case.query)
            from models import McpOptimizerSearchResult

            return McpOptimizerSearchResult(
                test_case=test_case,
                selected_server_name=None,
                selected_tool_name=None,
                retrieved_tools=[],
                search_time_s=0.0,
                request_tokens=0,
                response_tokens=0,
                total_tokens=0,
                num_tools_returned=0,
                error=str(mcp_result),
            )
        return mcp_result

    async def _run_single_test_case(
        self,
        test_case: TestCase,
        native_runner: NativeApproachRunner,
        mcp_runner: McpOptimizerAgentRunner,
        existing_results: dict[str, ComparisonResult],
        semaphore: asyncio.Semaphore,
    ) -> ComparisonResult:
        """Run all approaches on a single test case.

        Args:
            test_case: Test case to run
            native_runner: Native approach runner
            mcp_runner: MCP Optimizer agent runner
            existing_results: Dictionary of existing results to skip/retry
            semaphore: Semaphore for concurrency control

        Returns:
            Comparison result for the test case
        """
        async with semaphore:
            # Check if we already have successful results for this test case
            if test_case.query in existing_results:
                existing = existing_results[test_case.query]

                # Skip if all approaches succeeded
                if self._result_fully_succeeded(existing):
                    self.completed_count += 1
                    logger.info(
                        "Skipping completed test case",
                        query=test_case.query[:50] + "...",
                        progress=self._get_progress_string(),
                    )
                    return existing

                # Retry only the approach(es) that failed
                self.completed_count += 1
                logger.info(
                    "Retrying errored test case",
                    query=test_case.query[:50] + "...",
                    progress=self._get_progress_string(),
                )

                native_results, mcp_result = await self._retry_failed_approaches(
                    test_case, existing, native_runner, mcp_runner
                )

            else:
                self.completed_count += 1
                logger.info(
                    "Processing test case",
                    query=test_case.query[:50] + "...",
                    progress=self._get_progress_string(),
                )
                # Run all approaches in parallel
                tasks = []
                for approach in native_runner.APPROACHES:
                    tasks.append(native_runner.search_tool(test_case, approach))
                tasks.append(mcp_runner.search_tool(test_case))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Parse results - first ones are native (bm25, regex), last is mcp
                native_results = {}
                for i, approach in enumerate(native_runner.APPROACHES):
                    native_results[approach] = results[i]
                mcp_result = results[-1]

            # Handle exceptions
            native_results = self._handle_native_exceptions(native_results, test_case)
            mcp_result = self._handle_mcp_exception(mcp_result, test_case)

            # Compute comparison result
            return MetricsComputer.compute_single_result(test_case, native_results, mcp_result)

    async def _run_comparison(
        self,
        test_cases: list[TestCase],
        native_runner: NativeApproachRunner,
        mcp_runner: McpOptimizerAgentRunner,
        existing_results: dict[str, "ComparisonResult"],
    ) -> list[ComparisonResult]:
        """Run comparison on all test cases with concurrency control.

        Args:
            test_cases: List of test cases
            native_runner: Native approach runner
            mcp_runner: MCP Optimizer agent runner
            existing_results: Dictionary of existing results to skip/retry

        Returns:
            List of comparison results
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results_list: list[ComparisonResult] = []
        save_interval = 10  # Save partial results every N completions

        # Run all test cases with concurrent execution and periodic saves
        async def process_and_track():
            """Process tasks and track results."""
            tasks = [
                self._run_single_test_case(
                    test_case, native_runner, mcp_runner, existing_results, semaphore
                )
                for test_case in test_cases
            ]

            # Use asyncio.as_completed to process results as they finish
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results_list.append(result)

                # Save partial results every save_interval completions
                if len(results_list) % save_interval == 0:
                    self._save_partial_results(results_list)

            return results_list

        results = await process_and_track()

        # Final save to ensure all results are saved
        self._save_partial_results(results)

        logger.info("Completed all test cases", count=len(results))
        return results
