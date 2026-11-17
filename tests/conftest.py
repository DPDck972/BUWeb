"""conftest — adiciona src/ ao PYTHONPATH para todos os testes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
