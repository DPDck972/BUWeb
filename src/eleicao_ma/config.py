"""Configuração central — eleicao_ma."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | None) -> Path:
    if path_value is None:
        return ROOT / "data" / "bweb_2t_RN_261020260742.csv"
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    project_candidate = ROOT / candidate
    return project_candidate if project_candidate.exists() else candidate


# ── LLM ──────────────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("BU_MODEL", "qwen3:8b")

# ── Dados ─────────────────────────────────────────────────────────────────────
DATA_PATH = resolve_path(os.getenv("BU_DATA_PATH"))

# ── Servidor MCP ──────────────────────────────────────────────────────────────
# As bibliotecas do servidor ficam em servidor_libs/ (fastmcp 4.x ≠ mcp 1.x
# usado pelo LangGraph). Mesmo padrão da Lesson 12.
SERVER_LIBS = str(ROOT / "servidor_libs")
SERVER_PY = str(Path(__file__).with_name("mcp_server.py"))


def get_servidor_config(
    data_path: str | Path | None = None,
    server_libs: str | Path | None = None,
    python_executable: str | None = None,
) -> dict:
    """Monta a configuração stdio do servidor MCP unificado."""
    resolved_data = str(resolve_path(str(data_path))) if data_path else str(DATA_PATH)
    resolved_libs = str(server_libs) if server_libs else SERVER_LIBS
    return {
        "eleicao": {
            "transport": "stdio",
            "command": python_executable or sys.executable,
            "args": [SERVER_PY],
            "env": {
                "PYTHONPATH": resolved_libs,
                "BU_DATA_PATH": resolved_data,
                "FASTMCP_LOG_LEVEL": "ERROR",
                "FASTMCP_SHOW_SERVER_BANNER": "false",
            },
        }
    }


SERVIDOR = get_servidor_config()

# ── Agente ────────────────────────────────────────────────────────────────────
MAX_STEPS = int(os.getenv("BU_MAX_STEPS", "6"))
