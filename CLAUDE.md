## Project

The general purpose of the ToolHive MCP Optimizer is to develop a MCP server that acts as an intelligent intermediary between AI clients and multiple MCP servers. The MCP Optimizer server addresses the challenge of managing large numbers of MCP tools by providing semantic tool discovery, caching, and unified access through a single endpoint.

## Technical considerations
- Use uv as package manager. `uv add <package>` for adding a package. `uv add <package> --dev` for development packages for linting and testing
- Use the taskfile for running linting and formatting
    - `task format` for running formatters
    - `task lint` for running linters
    - `task typecheck` for running typecheckers
    - `task test` for running tests
- Use `pydantic` for validating structured data
- pyproject.toml should be the central place for configuring the project, i.e. linters, typecheckers, testing, etc
- Always prefer to use native Python types over custom types, e.g. use `list` instead of `List`, `dict` instead of `Dict`, etc.
- Prefer using `uv run python -c "import this"` instead of `python -c "import this"`. This ensures that the correct python version and environment is used.

## Code Structure
- The main server code is located in `src/mcp_optimizer/server.py`
- The database configuration and CRUD operations are in `src/mcp_optimizer/db/`
- The mcp-optimizer CLI implementation is located in `src/mcp_optimizer/cli.py`

## Development Workflow
- After adding or modifying code, use `task format` to automatically format the code
- Then run `task lint` and `task typecheck` to identify and fix any remaining errors
- Some formatting and linting errors can be automatically resolved by running `task format`
- Use `task test` to run the test suite and ensure all tests pass

## CLI Implementation
- Follow the pattern that every new command should group big functionality and add as parameters the inputs for that big functionality
- Prefer using logger for printing in the CLI instead of using `click.echo`


## Database Considerations
- The database configuration is in `src/mcp_optimizer/db/config.py`
- Models are defined in `src/mcp_optimizer/db/models.py`
- CRUD operations are implemented in `src/mcp_optimizer/db/crud.py`
- Use SQLAlchemy for database interactions. I would like to be as database agnostic as possible, so try not to use database-specific features like specific SQL dialects or functions.
- Use Pydantic models for data validation and serialization
- I like to use UUIDs for primary keys, so use `uuid.uuid4()` for generating new IDs
- Use `text()` from SQLAlchemy to safely execute raw SQL queries
- I do not want to use ORM features, so prefer executing raw SQL queries using `text()` and `Connection.execute()`

## Embeddings Considerations
- The embedding happens in `src/mcp_optimizer/embeddings.py`
- I am using `fastembed` for embedding text
- Wherever possible try to embed several text entries at once. The function takes a list of texts and returns a list of embeddings.


## Commit messages
- I like to make a single commit, hence my commit messages normally end up being the PR description. When I ask you to write a commit message take into account that eventually it will be a PR description. Use simple language without a lot of self-promoting, be direct on the changes that were made in the code. Try to keep the message in less than 100 words (max 200). If possible keep all in a list. The purpose is that the PR description can be read and understood in less than 1 minute
- use markdown when doing commit messages but not a lot. Keep it simple and avoid excessive formatting. Specially do not use markdown in the title of the commit message
