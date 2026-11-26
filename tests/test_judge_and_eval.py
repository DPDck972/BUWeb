"""Testes unitários da Etapa 2: perguntas.json, trajetória, juiz HF+MCP e pipeline CSV."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from openai import RateLimitError

from eleicao_ma import evaluate, judge
from eleicao_ma.systems import SistemaAgente

ROOT = Path(__file__).resolve().parents[1]
PERGUNTAS_JSON = ROOT / "data" / "perguntas.json"


# ── 1. Schema e cobertura de data/perguntas.json ──────────────────────────────


def test_perguntas_json_schema_and_coverage():
    perguntas = evaluate.carregar_perguntas(PERGUNTAS_JSON)
    assert len(perguntas) == 20

    ids = [p["id"] for p in perguntas]
    assert ids == [f"Q{i:02d}" for i in range(1, 21)]

    obrigatorios = {
        "id",
        "pergunta",
        "categoria",
        "ferramenta_esperada",
        "esperado",
        "checar",
        "proibido",
    }
    categorias = set()
    ferramentas = set()
    for p in perguntas:
        assert obrigatorios.issubset(p.keys()), f"Campos ausentes em {p.get('id')}"
        assert isinstance(p["pergunta"], str) and p["pergunta"].strip()
        assert p["categoria"] in evaluate.CATEGORIAS_VALIDAS
        assert isinstance(p["checar"], list)
        assert isinstance(p["proibido"], list)
        categorias.add(p["categoria"])
        ferramentas.add(p["ferramenta_esperada"])

    # Categorias obrigatórias
    assert evaluate.CATEGORIAS_VALIDAS.issubset(categorias)
    assert sum(1 for p in perguntas if p["categoria"] == "capciosa") >= 1
    assert sum(1 for p in perguntas if p["categoria"] == "fora_escopo") >= 1

    # Todas as ferramentas MCP principais + null (fora de escopo)
    for ferramenta in (
        "secao_details",
        "search_secoes",
        "compare_secoes",
        "total_por_tipo",
        "total_sem_biometria",
        "soma_secoes",
        None,
    ):
        assert ferramenta in ferramentas, f"Ferramenta não coberta: {ferramenta}"


# ── 2. Cálculo automático de trajetória (trajetoria_ok) ───────────────────────


def test_avaliar_trajetoria_com_ferramenta_esperada():
    p_det = {
        "id": "Q01",
        "categoria": "item_especifico",
        "ferramenta_esperada": "secao_details",
        "checar": ["505"],
        "proibido": [],
    }
    assert evaluate.avaliar_trajetoria(p_det, ["secao_details"], "Seção 505...") is True
    assert evaluate.avaliar_trajetoria(p_det, ["search_secoes"], "Seção 505...") is False
    assert evaluate.avaliar_trajetoria(p_det, [], "Seção 505...") is False


def test_avaliar_trajetoria_fora_de_escopo():
    p_fora = {
        "id": "Q20",
        "categoria": "fora_escopo",
        "ferramenta_esperada": None,
        "checar": ["não está", "fora do escopo"],
        "proibido": ["a capital é natal"],
    }
    assert (
        evaluate.avaliar_trajetoria(
            p_fora, [], "Esse dado não está no arquivo do Boletim de Urna."
        )
        is True
    )
    assert (
        evaluate.avaliar_trajetoria(p_fora, [], "A capital é Natal.") is False
    )
    assert (
        evaluate.avaliar_trajetoria(
            p_fora, [], "Fora do escopo, mas a capital é Natal."
        )
        is False
    )


# ── 3. Juiz LLM via Hugging Face + MCP (mockado) ──────────────────────────────


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def create(self, **kwargs):
        idx = self.calls
        self.calls += 1
        item = self._responses[min(idx, len(self._responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


class FakeHFClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _make_completion(content="", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _make_tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def test_judge_consults_mcp_and_returns_structured_dict():
    fake_client = FakeHFClient(
        [
            _make_completion(
                content="",
                tool_calls=[
                    _make_tool_call(
                        "tc_1", "secao_details", {"zona": 3, "secao": 505}
                    )
                ],
            ),
            _make_completion(
                content=json.dumps(
                    {
                        "veredito_juiz": True,
                        "justificativa_juiz": "Dados conferidos no MCP batem com a resposta.",
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )

    executed = []

    def fake_mcp_executor(name: str, args: dict):
        executed.append((name, args))
        return {"zona": 3, "secao": 505, "aptos": 460, "comparecimento": 432}

    fake_schema = [
        {
            "type": "function",
            "function": {
                "name": "secao_details",
                "description": "Detalhes da seção",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    async def run():
        async with judge.MCPJudge(
            client=fake_client,
            tools_schema=fake_schema,
            tool_executor=fake_mcp_executor,
        ) as j:
            return await j.avaliar(
                pergunta="Me dá os detalhes da seção 505 da zona 3.",
                resposta="Seção 505 da zona 3 tem 460 aptos e 432 comparecimento.",
                categoria="item_especifico",
            )

    res = asyncio.run(run())

    assert set(res.keys()) == {
        "veredito_juiz",
        "justificativa_juiz",
        "consultas_juiz_mcp",
    }
    assert res["veredito_juiz"] is True
    assert "MCP" in res["justificativa_juiz"]
    assert res["consultas_juiz_mcp"] == [
        {"ferramenta": "secao_details", "argumentos": {"zona": 3, "secao": 505}}
    ]
    assert executed == [("secao_details", {"zona": 3, "secao": 505})]


def test_judge_retries_on_rate_limit():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, request=req, headers={"Retry-After": "0.001"})
    rl_error = RateLimitError("429 rate limited", response=resp, body=None)

    fake_client = FakeHFClient(
        [
            rl_error,
            _make_completion(
                content='{"veredito_juiz": false, "justificativa_juiz": "Número divergente."}'
            ),
        ]
    )

    async def run():
        async with judge.MCPJudge(
            client=fake_client,
            tools_schema=[],
            tool_executor=lambda n, a: {},
            max_retries=2,
            retry_delay_s=0.001,
        ) as j:
            return await j.avaliar("Pergunta", "Resposta incorreta")

    res = asyncio.run(run())
    assert res["veredito_juiz"] is False
    assert fake_client.completions.calls == 2


def test_judge_groq_default_and_hf_provider(monkeypatch):
    for var in ("JUDGE_PROVIDER", "JUDGE_MODEL", "GROQ_JUDGE_MODEL", "HF_JUDGE_MODEL", "GROQ_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    # Padrão: Groq com openai/gpt-oss-120b
    assert judge.get_judge_provider() == "groq"
    assert judge.get_judge_model() == "openai/gpt-oss-120b"

    j_groq = judge.MCPJudge()
    assert j_groq.provider == "groq"
    assert j_groq.model == "openai/gpt-oss-120b"
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        _ = j_groq.client

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    client_groq = judge.MCPJudge().client
    assert "api.groq.com/openai/v1" in str(client_groq.base_url)

    # Alternativa: Hugging Face com Qwen/Qwen2.5-72B-Instruct
    assert judge.get_judge_model(provider="huggingface") == "Qwen/Qwen2.5-72B-Instruct"
    monkeypatch.setenv("HF_JUDGE_MODEL", judge.HF_JUDGE_FALLBACK_MODEL)
    assert judge.get_judge_model(provider="hf") == "meta-llama/Llama-3.3-70B-Instruct"

    j_hf = judge.MCPJudge(provider="huggingface")
    with pytest.raises(EnvironmentError, match="HF_TOKEN"):
        _ = j_hf.client


# ── 4. Pipeline completo: ambiente.json, CSVs e verificação manual ────────────


def test_evaluate_pipeline_generates_ambiente_and_csvs(tmp_path: Path):
    perguntas = evaluate.carregar_perguntas(PERGUNTAS_JSON)
    # Seleciona 4 perguntas representativas: item_especifico, capciosa, total_sem_biometria, fora_escopo
    amostra = [p for p in perguntas if p["id"] in {"Q01", "Q10", "Q14", "Q20"}]
    assert len(amostra) == 4

    async def fake_lg_runner(question: str) -> dict:
        if "biometria" in question.lower():
            tools_used = ["total_sem_biometria"]
            resp = "180.838 eleitores sem biometria válida (6,8%)."
        elif "capital" in question.lower():
            tools_used = []
            resp = "Esse dado não está no arquivo."
        elif "menos" in question.lower():
            tools_used = ["compare_secoes"]
            resp = "A seção 505 tem menos votos nulos: 14 vs 17."
        else:
            tools_used = ["secao_details"]
            resp = "Seção 505, NATAL: 460 aptos, 432 comparecimento."
        return {
            "resposta": resp,
            "ferramentas_chamadas": tools_used,
            "chamadas_modelo": 2,
            "chamadas_ferramentas": len(tools_used),
            "latencia_s": 1.25,
            "tokens_entrada": 200,
            "tokens_saida": 50,
            "tokens_total": 250,
            "cpu_percent_max": 12.5,
            "ram_mb_max": 180.0,
            "gpu_mem_mb_max": None,
        }

    sistemas = {
        "langgraph_unico": SistemaAgente(
            sistema="langgraph_unico",
            framework="LangGraph",
            arquitetura="agente único",
            runner=fake_lg_runner,
        ),
        "crewai_unico": SistemaAgente(
            sistema="crewai_unico",
            framework="CrewAI",
            arquitetura="agente único",
            runner=fake_lg_runner,
        ),
    }

    fake_client = FakeHFClient(
        [
            _make_completion(
                content='{"veredito_juiz": true, "justificativa_juiz": "Resposta validada."}'
            )
        ]
    )

    async def run():
        async with judge.MCPJudge(
            client=fake_client,
            tools_schema=[],
            tool_executor=lambda n, a: {},
        ) as j:
            return await evaluate.executar_benchmark(
                sistemas,
                perguntas=amostra,
                judge=j,
                output_dir=tmp_path,
                salvar_copia_timestamp=True,
                verbose=False,
            )

    df_res, df_manual, arquivos = asyncio.run(run())

    # Verifica existência de todos os arquivos gerados
    assert arquivos["ambiente"].is_file()
    assert arquivos["resultados"].is_file()
    assert arquivos["verificacao_manual"].is_file()
    assert arquivos["resultados_timestamp"].is_file()

    # Verifica ambiente.json
    amb = json.loads(arquivos["ambiente"].read_text(encoding="utf-8"))
    for chave in (
        "cpu",
        "ram_total_gb",
        "gpu",
        "modelo_local",
        "quantizacao_modelo_local",
        "provedor_juiz",
        "modelo_juiz",
        "versoes",
    ):
        assert chave in amb
    assert amb["provedor_juiz"] == "groq"
    assert amb["modelo_juiz"] == "openai/gpt-oss-120b"
    for lib in ("python", "langgraph", "crewai", "mcp", "fastmcp", "pandas"):
        assert lib in amb["versoes"]

    # Verifica CSV de resultados
    csv_res = pd.read_csv(arquivos["resultados"])
    assert list(csv_res.columns) == list(evaluate.COLUNAS_RESULTADOS)
    assert len(csv_res) == 8  # 2 sistemas x 4 perguntas
    assert csv_res["veredito_juiz"].all()
    assert csv_res["trajetoria_ok"].all()

    # Verifica CSV de verificação manual
    csv_man = pd.read_csv(arquivos["verificacao_manual"], keep_default_na=False)
    assert list(csv_man.columns) == list(evaluate.COLUNAS_VERIFICACAO_MANUAL)
    assert len(csv_man) == 8
    assert (csv_man["veredito_manual"] == "").all()

    # Verifica agregação de latência (média, mediana, p95)
    resumo = evaluate.resumo_metricas_por_sistema(df_res)
    assert len(resumo) == 2
    assert set(
        ["latencia_media_s", "latencia_mediana_s", "latencia_p95_s", "taxa_trajetoria_ok"]
    ).issubset(resumo.columns)
