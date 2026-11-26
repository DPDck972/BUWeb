"""Servidor MCP unificado das ferramentas de auditoria eleitoral."""

import sys
from pathlib import Path

from fastmcp import FastMCP

if __package__:
    from . import tools
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eleicao_ma import tools

mcp = FastMCP("eleicao-RN")


@mcp.tool
def total_por_tipo(
    tipo_voto: str | None = None,
    cargo: str | None = None,
) -> dict:
    """Total de votos por tipo ('Nulo', 'Branco' ou 'Nominal'), opcionalmente filtrado por cargo ('Governador' ou 'Presidente')."""
    return tools.total_por_tipo(tipo_voto=tipo_voto, cargo=cargo)


@mcp.tool
def total_sem_biometria(
    zona: str | int | None = None,
    municipio: str | None = None,
) -> dict:
    """Total de eleitores sem biometria válida no arquivo inteiro.

    Para o total geral, chame sem argumentos.
    """
    return tools.total_sem_biometria(zona=zona, municipio=municipio)


@mcp.tool
def secao_details(
    zona: str | int | None = None,
    secao: str | int | None = None,
) -> dict:
    """Todos os dados de uma seção eleitoral (informe zona e secao)."""
    return tools.secao_details(zona=zona, secao=secao)


@mcp.tool
def compare_secoes(
    zona_a: str | int | None = None,
    secao_a: str | int | None = None,
    zona_b: str | int | None = None,
    secao_b: str | int | None = None,
    tipo_voto: str | None = None,
) -> dict:
    """Compara duas seções lado a lado, agregando por cargo. tipo_voto aceita 'Nulo', 'Branco' ou 'Nominal'."""
    return tools.compare_secoes(
        zona_a=zona_a,
        secao_a=secao_a,
        zona_b=zona_b,
        secao_b=secao_b,
        tipo_voto=tipo_voto,
    )


@mcp.tool
def search_secoes(
    tipo_voto: str | None = None,
    zona: str | int | None = None,
    cargo: str | None = None,
    min_votos: str | int | None = None,
    top_n: str | int | None = 5,
) -> dict:
    """Busca seções no BU por tipo de voto ('Nulo', 'Branco', 'Nominal'), zona, cargo ('Governador', 'Presidente') e mínimo de votos."""
    return tools.search_secoes(
        tipo_voto=tipo_voto,
        zona=zona,
        cargo=cargo,
        min_votos=min_votos,
        top_n=top_n,
    )


@mcp.tool
def soma_secoes(
    zona_a: str | int | None = None,
    secao_a: str | int | None = None,
    zona_b: str | int | None = None,
    secao_b: str | int | None = None,
    tipo_voto: str | None = None,
) -> dict:
    """Soma os votos de duas seções, opcionalmente filtrado por tipo_voto ('Nulo', 'Branco' ou 'Nominal')."""
    return tools.soma_secoes(
        zona_a=zona_a,
        secao_a=secao_a,
        zona_b=zona_b,
        secao_b=secao_b,
        tipo_voto=tipo_voto,
    )


@mcp.resource("eleicao-RN://schema")
def schema() -> str:
    """As colunas do Boletim de Urna."""
    return ", ".join(tools.bu.columns)


@mcp.prompt
def recomendar_analise(
    cargo: str,
    tipo_voto: str,
    top_n: int = 5,
) -> str:
    """Pede uma análise de seções com cargo e tipo de voto preenchidos."""
    return (
        f"Analise as seções com mais votos {tipo_voto} para {cargo}. "
        f"Use a ferramenta search_secoes com cargo='{cargo}', "
        f"tipo_voto='{tipo_voto}' e top_n={top_n}. "
        f"Liste as {top_n} primeiras citando zona, seção, município e o total "
        "de votos. Depois aponte a seção com o maior valor e o quanto ela "
        "supera a segunda colocada."
    )


if __name__ == "__main__":
    mcp.run()
