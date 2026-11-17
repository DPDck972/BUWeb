"""Verifica a contagem de chamadas sem iniciar Ollama ou o servidor MCP."""
import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from eleicao_ma import arq_a_gerador_verificador as arq_a
from eleicao_ma.arq_a_gerador_verificador import Revisao, gerador_verificador
from eleicao_ma.arq_b_supervisor import Rota, supervisor_especialistas
from eleicao_ma.utils import rodar


class FakeLLM:
    model = "test-model"

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema):
        return FakeRouter()

    async def ainvoke(self, _messages):
        return AIMessage(content="Resposta de teste.")


class FakeRouter:
    async def ainvoke(self, _messages):
        return Rota(proximo="answer", tarefa="")


class FakeVerifier:
    async def ainvoke(self, _messages):
        return Revisao(problemas=[])


class FakeChatOllama:
    def __init__(self, **_kwargs):
        pass

    def with_structured_output(self, _schema):
        return FakeVerifier()


def test_gerador_verificador_conta_chamadas(monkeypatch):
    monkeypatch.setattr(arq_a, "ChatOllama", FakeChatOllama)
    app = gerador_verificador([], FakeLLM())

    resultado = asyncio.run(rodar(app, "Pergunta de teste", ver=False))

    assert resultado["chamadas"] == 2


def test_execucoes_concorrentes_mantem_contadores_separados(monkeypatch):
    monkeypatch.setattr(arq_a, "ChatOllama", FakeChatOllama)
    app = gerador_verificador([], FakeLLM())

    async def executar():
        return await asyncio.gather(
            rodar(app, "Pergunta A", ver=False),
            rodar(app, "Pergunta B", ver=False),
        )

    resultados = asyncio.run(executar())

    assert [resultado["chamadas"] for resultado in resultados] == [2, 2]


def test_supervisor_e_redator_contam_chamadas():
    nomes = (
        "search_secoes",
        "secao_details",
        "compare_secoes",
        "total_por_tipo",
        "total_sem_biometria",
    )
    ferramentas = [
        StructuredTool.from_function(
            func=lambda: {},
            name=nome,
            description="Ferramenta de teste.",
        )
        for nome in nomes
    ]
    app = supervisor_especialistas(ferramentas, FakeLLM())

    resultado = asyncio.run(rodar(app, "Pergunta de teste", ver=False))

    assert resultado["chamadas"] == 2
