{{/*
Expand the name of the chart.
*/}}
{{- define "mcp-optimizer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mcp-optimizer.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mcp-optimizer.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mcp-optimizer.labels" -}}
helm.sh/chart: {{ include "mcp-optimizer.chart" . }}
{{ include "mcp-optimizer.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mcp-optimizer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mcp-optimizer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "mcp-optimizer.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "mcp-optimizer.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get the image tag to use
*/}}
{{- define "mcp-optimizer.imageTag" -}}
{{- .Values.mcpserver.image.tag | default .Chart.AppVersion }}
{{- end }}

{{/*
Get the full image name
*/}}
{{- define "mcp-optimizer.image" -}}
{{- printf "%s:%s" .Values.mcpserver.image.repository (include "mcp-optimizer.imageTag" .) }}
{{- end }}

{{/*
Database URL for async operations
*/}}
{{- define "mcp-optimizer.asyncDbUrl" -}}
{{- if eq .Values.database.type "sqlite" }}
{{- printf "sqlite+aiosqlite://%s" .Values.database.sqlite.path }}
{{- else if eq .Values.database.type "postgresql" }}
{{- printf "postgresql+asyncpg://%s:PASSWORD@%s:%s/%s" .Values.database.postgresql.username .Values.database.postgresql.host (.Values.database.postgresql.port | toString) .Values.database.postgresql.database }}
{{- end }}
{{- end }}

{{/*
Database URL for sync operations
*/}}
{{- define "mcp-optimizer.dbUrl" -}}
{{- if eq .Values.database.type "sqlite" }}
{{- printf "sqlite://%s" .Values.database.sqlite.path }}
{{- else if eq .Values.database.type "postgresql" }}
{{- printf "postgresql://%s:PASSWORD@%s:%s/%s" .Values.database.postgresql.username .Values.database.postgresql.host (.Values.database.postgresql.port | toString) .Values.database.postgresql.database }}
{{- end }}
{{- end }}

