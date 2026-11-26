"""Interface comum padronizada answer(question) para os 6 sistemas (LangGraph e CrewAI)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .arq_a_gerador_verificador import agente_simples, gerador_verificador
from .arq_b_supervisor import supervisor_especialistas
from .crew_arqs import (
    crew_gerador_verificador,
    crew_supervisor,
    crew_unico,
    rodar_crew,
)
from .utils import rodar

SYSTEM_SPECS: dict[str, tuple[str, str]] = {
    "langgraph_unico": ("LangGraph", "agente único"),
    "langgraph_gerador_verificador": ("LangGraph", "gerador + verificador"),
    "langgraph_supervisor": ("LangGraph", "supervisor"),
    "crewai_unico": ("CrewAI", "agente único"),
    "crewai_gerador_verificador": ("CrewAI", "gerador + verificador"),
    "crewai_supervisor": ("CrewAI", "supervisor"),
}

RESULT_KEYS: tuple[str, ...] = (
    "sistema",
    "framework",
    "arquitetura",
    "resposta",
    "ferramentas_chamadas",
    "chamadas_modelo",
    "chamadas_ferramentas",
    "latencia_s",
    "tokens_entrada",
    "tokens_saida",
    "tokens_total",
    "cpu_percent_max",
    "ram_mb_max",
    "gpu_mem_mb_max",
)


def _formatar_resultado(sistema: str, framework: str, arquitetura: str, bruto: dict) -> dict:
    ferramentas = list(bruto.get("ferramentas_chamadas", bruto.get("ferramentas", [])))
    tokens_in = int(bruto.get("tokens_entrada", 0))
    tokens_out = int(bruto.get("tokens_saida", 0))
    gpu = bruto.get("gpu_mem_mb_max")
    return {
        "sistema": sistema,
        "framework": framework,
        "arquitetura": arquitetura,
        "resposta": str(bruto.get("resposta", "")),
        "ferramentas_chamadas": ferramentas,
        "chamadas_modelo": int(bruto.get("chamadas_modelo", bruto.get("chamadas", 0))),
        "chamadas_ferramentas": int(bruto.get("chamadas_ferramentas", len(ferramentas))),
        "latencia_s": float(bruto.get("latencia_s", bruto.get("segundos", 0.0))),
        "tokens_entrada": tokens_in,
        "tokens_saida": tokens_out,
        "tokens_total": int(bruto.get("tokens_total", tokens_in + tokens_out)),
        "cpu_percent_max": float(bruto.get("cpu_percent_max", 0.0)),
        "ram_mb_max": float(bruto.get("ram_mb_max", 0.0)),
        "gpu_mem_mb_max": float(gpu) if gpu is not None else None,
    }


@dataclass
class SistemaAgente:
    """Wrapper uniforme que expõe `await sistema.answer(question)` para qualquer arquitetura."""

    sistema: str
    framework: str
    arquitetura: str
    runner: Callable[[str], Any]

    async def answer(self, question: str) -> dict:
        bruto = await self.runner(question)
        return _formatar_resultado(
            sistema=self.sistema,
            framework=self.framework,
            arquitetura=self.arquitetura,
            bruto=bruto,
        )


def criar_sistema_langgraph(nome: str, app) -> SistemaAgente:
    """Envolve um grafo compilado do LangGraph na interface comum `SistemaAgente`."""
    if nome not in SYSTEM_SPECS or SYSTEM_SPECS[nome][0] != "LangGraph":
        raise ValueError(f"Sistema LangGraph desconhecido: '{nome}'")
    framework, arquitetura = SYSTEM_SPECS[nome]
    return SistemaAgente(
        sistema=nome,
        framework=framework,
        arquitetura=arquitetura,
        runner=lambda q: rodar(app, q, ver=False),
    )


def criar_sistema_crewai(
    nome: str,
    crew_factory: Callable[[], Any],
    tools_names: set[str] | None = None,
) -> SistemaAgente:
    """Envolve uma factory CrewAI na interface comum `SistemaAgente`."""
    if nome not in SYSTEM_SPECS or SYSTEM_SPECS[nome][0] != "CrewAI":
        raise ValueError(f"Sistema CrewAI desconhecido: '{nome}'")
    framework, arquitetura = SYSTEM_SPECS[nome]
    return SistemaAgente(
        sistema=nome,
        framework=framework,
        arquitetura=arquitetura,
        runner=lambda q: rodar_crew(crew_factory, q, tools_names=tools_names, ver=False),
    )


def criar_sistemas_langgraph(
    ferramentas,
    llm,
    max_revisoes: int = 2,
    max_passos: int = 4,
    llm_verificador=None,
) -> dict[str, SistemaAgente]:
    """Cria os 3 sistemas LangGraph com a interface comum `answer(question)`."""
    apps = {
        "langgraph_unico": agente_simples(ferramentas, llm),
        "langgraph_gerador_verificador": gerador_verificador(
            ferramentas,
            llm,
            max_revisoes=max_revisoes,
            llm_verificador=llm_verificador,
        ),
        "langgraph_supervisor": supervisor_especialistas(
            ferramentas, llm, max_passos=max_passos
        ),
    }
    return {
        nome: criar_sistema_langgraph(nome, app)
        for nome, app in apps.items()
    }


def criar_sistemas_crewai(tools_map: dict, llm) -> dict[str, SistemaAgente]:
    """Cria os 3 sistemas CrewAI com a interface comum `answer(question)`."""
    tools_names = set(tools_map)
    factories = {
        "crewai_unico": lambda: crew_unico(tools_map, llm),
        "crewai_gerador_verificador": lambda: crew_gerador_verificador(tools_map, llm),
        "crewai_supervisor": lambda: crew_supervisor(tools_map, llm),
    }
    return {
        nome: criar_sistema_crewai(nome, factory, tools_names=tools_names)
        for nome, factory in factories.items()
    }


def criar_todos_sistemas(
    lg_ferramentas,
    lg_llm,
    crew_tools_map: dict,
    crew_llm,
    max_revisoes: int = 2,
    max_passos: int = 4,
    llm_verificador=None,
) -> dict[str, SistemaAgente]:
    """Cria todos os 6 sistemas (3 LangGraph + 3 CrewAI) sob a mesma interface."""
    sistemas = {}
    sistemas.update(
        criar_sistemas_langgraph(
            lg_ferramentas,
            lg_llm,
            max_revisoes=max_revisoes,
            max_passos=max_passos,
            llm_verificador=llm_verificador,
        )
    )
    sistemas.update(criar_sistemas_crewai(crew_tools_map, crew_llm))
    return sistemas


async def answer(sistema: SistemaAgente, question: str) -> dict:
    """Função de conveniência para invocar qualquer um dos 6 sistemas."""
    return await sistema.answer(question)
