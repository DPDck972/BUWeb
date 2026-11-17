"""Testes das duas arquiteturas multiagente."""
import pytest


@pytest.fixture
def csv_path():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data" / "bweb_2t_RN_261020260742.csv"
    if not p.exists():
        pytest.skip("CSV de dados não encontrado")
    return p


def test_tools_import():
    """As ferramentas pandas funcionam sem depender do LangGraph."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from eleicao_ma.data import load_bu
    bu = load_bu()
    assert len(bu) > 0


def test_total_por_tipo(csv_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from eleicao_ma import tools   # noqa: F401 — dispara load_bu
    from eleicao_ma.tools import total_por_tipo
    r = total_por_tipo(tipo_voto="Nulo", cargo="Presidente")
    assert "total_votos" in r
    assert r["total_votos"] > 0


def test_search_secoes(csv_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from eleicao_ma.tools import search_secoes
    r = search_secoes(tipo_voto="Nulo", cargo="Presidente", top_n=3)
    assert "secoes" in r
    assert len(r["secoes"]) <= 3


def test_total_sem_biometria(csv_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from eleicao_ma.tools import total_sem_biometria
    r = total_sem_biometria()
    assert "total_sem_biometria" in r
    assert r["total_sem_biometria"] >= 0


def test_tipo_invalido(csv_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from eleicao_ma.tools import total_por_tipo
    r = total_por_tipo(tipo_voto="Invalido")
    assert "error" in r
