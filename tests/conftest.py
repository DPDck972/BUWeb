"""conftest — adiciona src/ ao PYTHONPATH e configura diretório temporário para testes."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("XDG_DATA_HOME", str(Path(tempfile.gettempdir()) / "eleicao_ma_xdg"))
