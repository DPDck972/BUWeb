"""Configuração central — eleicao_ma."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # pj_eleicao_multiagente/


def resolve_path(path_value: str | None) -> Path:
    if path_value is None:
        return ROOT / "data" / "bweb_2t_RN_261020260742.csv"
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    project_candidate = ROOT / candidate
    return project_candidate if project_candidate.exists() else candidate


# ── LLM ──────────────────────────────────────────────────────────────────────
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL       = os.getenv("BU_MODEL",   "qwen2.5:7b")

# ── Dados ─────────────────────────────────────────────────────────────────────
DATA_PATH   = resolve_path(os.getenv("BU_DATA_PATH"))

# ── Servidor MCP ──────────────────────────────────────────────────────────────
# As bibliotecas do servidor ficam em servidor_libs/ (fastmcp 4.x  ≠  mcp 1.x
# usado pelo LangGraph). Mesmo padrão da Lesson 12.
SERVER_LIBS = str(ROOT / "servidor_libs")
SERVER_PY   = str(Path(__file__).with_name("mcp_server.py"))

SERVIDOR = {
    "eleicao": {
        "transport": "stdio",
        "command":   "python",          # sobrescrito em runtime pelo sys.executable
        "args":      [SERVER_PY],
        "env": {
            "PYTHONPATH":                SERVER_LIBS,
            "FASTMCP_LOG_LEVEL":         "ERROR",
            "FASTMCP_SHOW_SERVER_BANNER": "false",
        },
    }
}

# ── Agente ────────────────────────────────────────────────────────────────────
MAX_STEPS = int(os.getenv("BU_MAX_STEPS", "6"))
