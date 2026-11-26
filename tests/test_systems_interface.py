"""Testes unitários da interface comum answer(question) e coleta de métricas sem Ollama."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from eleicao_ma import arq_a_gerador_verificador as arq_a
from eleicao_ma import config, utils
from eleicao_ma.arq_a_gerador_verificador import Revisao
from eleicao_ma.arq_b_supervisor import Rota
from eleicao_ma.systems import (
    RESULT_KEYS,
    SYSTEM_SPECS,
    answer,
    criar_sistema_crewai,
    criar_sistemas_langgraph,
)


class ScriptedLangGraphLLM:
    """LLM falso que dispara uma chamada de ferramenta e devolve tokens em usage_metadata."""

    model = "qwen3:8b"

    def __init__(self):
        self._tool_called = False

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, schema, include_raw=False):
        if schema is Rota:
            return FakeStructuredRunner(
                Rota(proximo="comparador", tarefa="Consultar seção 505 da zona 3"),
                Rota(proximo="answer", tarefa=""),
                include_raw=include_raw,
            )
        if schema is Revisao:
            return FakeStructuredRunner(
                Revisao(problemas=[]),
                include_raw=include_raw,
            )
        raise AssertionError(f"Schema inesperado: {schema}")

    async def ainvoke(self, messages):
        # Se houver uma mensagem de ferramenta já no histórico, responde em texto
        has_tool_msg = any(getattr(m, "type", None) == "tool" for m in messages)
        # Se for redator (sem bind_tools ou com relatórios no texto), responde direto
        is_redator = any("RELATÓRIOS DOS ESPECIALISTAS" in str(m.content) for m in messages)
        if not has_tool_msg and not is_redator and not self._tool_called:
            self._tool_called = True
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "secao_details",
                        "args": {"zona": 3, "secao": 505},
                        "id": "call_1",
                    }
                ],
                usage_metadata={"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
            )
        return AIMessage(
            content="Zona 3, Seção 505, município NATAL, com 460 aptos e 432 comparecimento.",
            usage_metadata={"input_tokens": 150, "output_tokens": 40, "total_tokens": 190},
        )


class FakeStructuredRunner:
    def __init__(self, *responses, include_raw=False):
        self._responses = list(responses)
        self._idx = 0
        self._include_raw = include_raw

    async def ainvoke(self, _messages):
        item = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        raw = AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
        )
        if self._include_raw:
            return {"raw": raw, "parsed": item, "parsing_error": None}
        return item


class FakeChatOllama:
    def __init__(self, **_kwargs):
        pass

    def with_structured_output(self, _schema, include_raw=False):
        return FakeStructuredRunner(Revisao(problemas=[]), include_raw=include_raw)


def _fake_tools():
    def _secao_details(zona: int = 3, secao: int = 505) -> str:
        return (
            '{"zona": 3, "secao": 505, "municipio": "NATAL", '
            '"aptos": 460, "comparecimento": 432, "abstencoes": 28, "sem_biometria": 39}'
        )

    nomes = (
        "search_secoes",
        "secao_details",
        "compare_secoes",
        "total_por_tipo",
        "total_sem_biometria",
        "soma_secoes",
    )
    return [
        StructuredTool.from_function(
            func=_secao_details if nome == "secao_details" else (lambda: "{}"),
            name=nome,
            description=f"Ferramenta mock {nome}.",
        )
        for nome in nomes
    ]


class FakeCrew:
    def __init__(self, texto_resp: str, chamadas: int, in_tok: int, out_tok: int, tools_used: list[str]):
        self._texto = texto_resp
        self._chamadas = chamadas
        self._in_tok = in_tok
        self._out_tok = out_tok
        self._tools_used = tools_used

    async def kickoff_async(self, inputs=None):
        return SimpleNamespace(
            raw=self._texto,
            token_usage={
                "prompt_tokens": self._in_tok,
                "completion_tokens": self._out_tok,
                "total_tokens": self._in_tok + self._out_tok,
                "successful_requests": self._chamadas,
            },
            ferramentas_chamadas=self._tools_used,
        )


def _assert_contract(res: dict, nome: str):
    assert set(res.keys()) == set(RESULT_KEYS)
    fw, arq = SYSTEM_SPECS[nome]
    assert res["sistema"] == nome
    assert res["framework"] == fw
    assert res["arquitetura"] == arq
    assert isinstance(res["resposta"], str) and res["resposta"]
    assert isinstance(res["ferramentas_chamadas"], list)
    assert res["chamadas_ferramentas"] == len(res["ferramentas_chamadas"])
    assert isinstance(res["chamadas_modelo"], int) and res["chamadas_modelo"] >= 1
    assert isinstance(res["latencia_s"], float) and res["latencia_s"] >= 0.0
    assert isinstance(res["tokens_entrada"], int) and res["tokens_entrada"] > 0
    assert isinstance(res["tokens_saida"], int) and res["tokens_saida"] > 0
    assert res["tokens_total"] == res["tokens_entrada"] + res["tokens_saida"]
    assert isinstance(res["cpu_percent_max"], float) and res["cpu_percent_max"] >= 0.0
    assert isinstance(res["ram_mb_max"], float) and res["ram_mb_max"] >= 0.0
    assert res["gpu_mem_mb_max"] is None or isinstance(res["gpu_mem_mb_max"], float)


def test_config_default_model_aligned():
    assert config.MODEL == "qwen3:8b"


def test_all_three_langgraph_systems_common_interface(monkeypatch):
    monkeypatch.setattr(arq_a, "ChatOllama", FakeChatOllama)
    ferramentas = _fake_tools()

    # Cada sistema recebe sua própria instância do LLM mockado para exercitar tool_calls
    sistemas = {
        "langgraph_unico": criar_sistemas_langgraph(ferramentas, ScriptedLangGraphLLM())["langgraph_unico"],
        "langgraph_gerador_verificador": criar_sistemas_langgraph(ferramentas, ScriptedLangGraphLLM())[
            "langgraph_gerador_verificador"
        ],
        "langgraph_supervisor": criar_sistemas_langgraph(ferramentas, ScriptedLangGraphLLM())[
            "langgraph_supervisor"
        ],
    }

    async def run_all():
        out = {}
        for nome, sis in sistemas.items():
            out[nome] = await answer(sis, "Me dá os detalhes da seção 505 da zona 3.")
        return out

    resultados = asyncio.run(run_all())

    # 1) langgraph_unico: 2 chamadas ao modelo (100+150 in, 25+40 out), 1 ferramenta
    r_unico = resultados["langgraph_unico"]
    _assert_contract(r_unico, "langgraph_unico")
    assert r_unico["chamadas_modelo"] == 2
    assert r_unico["ferramentas_chamadas"] == ["secao_details"]
    assert r_unico["tokens_entrada"] == 250
    assert r_unico["tokens_saida"] == 65

    # 2) langgraph_gerador_verificador: 2 gerador + 1 verificador = 3 chamadas; tokens somados
    r_gv = resultados["langgraph_gerador_verificador"]
    _assert_contract(r_gv, "langgraph_gerador_verificador")
    assert r_gv["chamadas_modelo"] == 3
    assert r_gv["ferramentas_chamadas"] == ["secao_details"]
    assert r_gv["tokens_entrada"] == 330
    assert r_gv["tokens_saida"] == 85

    # 3) langgraph_supervisor: supervisor(1) -> comparador(2) -> supervisor(1) -> redator(1) = 5 chamadas
    r_sup = resultados["langgraph_supervisor"]
    _assert_contract(r_sup, "langgraph_supervisor")
    assert r_sup["chamadas_modelo"] == 5
    assert r_sup["ferramentas_chamadas"] == ["secao_details"]
    assert r_sup["tokens_entrada"] == 80 + 100 + 150 + 80 + 150
    assert r_sup["tokens_saida"] == 20 + 25 + 40 + 20 + 40


def test_all_three_crewai_systems_common_interface():
    sistemas_crew = {
        "crewai_unico": criar_sistema_crewai(
            "crewai_unico",
            lambda: FakeCrew("Seção 505: 460 aptos.", 2, 210, 55, ["secao_details"]),
        ),
        "crewai_gerador_verificador": criar_sistema_crewai(
            "crewai_gerador_verificador",
            lambda: FakeCrew("Seção 505 validada.", 4, 420, 110, ["secao_details", "secao_details"]),
        ),
        "crewai_supervisor": criar_sistema_crewai(
            "crewai_supervisor",
            lambda: FakeCrew("Supervisor: seção 505.", 5, 530, 140, ["secao_details"]),
        ),
    }

    async def run_all():
        return {
            nome: await sis.answer("Me dá os detalhes da seção 505 da zona 3.")
            for nome, sis in sistemas_crew.items()
        }

    resultados = asyncio.run(run_all())

    for nome, res in resultados.items():
        _assert_contract(res, nome)

    assert resultados["crewai_unico"]["chamadas_modelo"] == 2
    assert resultados["crewai_gerador_verificador"]["chamadas_ferramentas"] == 2
    assert resultados["crewai_supervisor"]["tokens_total"] == 670


def test_resource_monitor_gpu_fallback(monkeypatch):
    monkeypatch.setattr(utils, "_ler_gpu_mem_mb", lambda: None)
    with utils.ResourceMonitor(interval=0.02) as mon:
        _ = sum(range(1000))
    snap = mon.snapshot()
    assert snap["cpu_percent_max"] >= 0.0
    assert snap["ram_mb_max"] >= 0.0
    assert snap["gpu_mem_mb_max"] is None


def test_supervisor_has_only_two_specialists():
    assert set(Rota.model_fields["proximo"].annotation.__args__) == {
        "buscador",
        "comparador",
        "answer",
    }
