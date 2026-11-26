"""As ferramentas de auditoria eleitoral do agente."""
import pandas as pd

from .data import _clean_opt, _num, _parse_int, load_bu

bu = load_bu()

TIPOS = ("Nulo", "Branco", "Nominal")
CARGOS = ("Governador", "Presidente")


def _validar_tipo_voto(tipo_voto):
    tipo_voto = _clean_opt(tipo_voto)
    if tipo_voto is not None and tipo_voto not in TIPOS:
        return None, {
            "error": (
                f"tipo_voto '{tipo_voto}' inválido. "
                "Use 'Nulo', 'Branco' ou 'Nominal'."
            )
        }
    return tipo_voto, None


def _validar_cargo(cargo):
    cargo = _clean_opt(cargo)
    if cargo is not None and cargo not in CARGOS:
        return None, {
            "error": f"cargo '{cargo}' inválido. Use 'Governador' ou 'Presidente'."
        }
    return cargo, None


def total_por_tipo(tipo_voto=None, cargo=None):
    """Total de votos por tipo, opcionalmente filtrado por cargo."""
    tipo_voto, err = _validar_tipo_voto(tipo_voto)
    if err:
        return err
    cargo, err = _validar_cargo(cargo)
    if err:
        return err

    r = bu.copy()
    if tipo_voto is not None:
        r = r[r.DS_TIPO_VOTAVEL == tipo_voto]
    if cargo is not None:
        r = r[r.DS_CARGO_PERGUNTA == cargo]

    if r.empty:
        return {"error": "nenhum voto encontrado com esses filtros"}

    return {
        "total_votos": int(r["QT_VOTOS_N"].sum()),
        "tipo_voto": tipo_voto,
        "cargo": cargo,
        "linhas": len(r),
    }


def total_sem_biometria(zona=None, municipio=None, **kwargs):
    """Total de eleitores sem biometria válida, opcionalmente por zona ou município."""
    zona = _clean_opt(zona)
    municipio = _clean_opt(municipio)

    r = bu.copy()
    if zona is not None:
        z, err = _parse_int(zona, "zona")
        if err:
            return err
        if z <= 0:
            return {
                "error": (
                    f"zona {z} não existe. "
                    "Omita o parâmetro zona para consultar todas as zonas."
                )
            }
        r = r[r["ZONA_N"] == z]
    if municipio is not None:
        r = r[r.NM_MUNICIPIO.str.upper() == str(municipio).strip().upper()]

    if r.empty:
        return {"error": "nenhuma seção encontrada com esses filtros"}

    # biometria é por seção — deduplica para não contar uma vez por linha de voto
    por_secao = r.drop_duplicates(subset=["ZONA_N", "SECAO_N"])
    biom = pd.to_numeric(
        por_secao["QT_ELEI_BIOM_SEM_HABILITACAO"], errors="coerce"
    ).fillna(0)
    aptos_s = pd.to_numeric(por_secao["QT_APTOS"], errors="coerce").fillna(0)

    total = int(biom.sum())
    aptos = int(aptos_s.sum())

    return {
        "total_sem_biometria": total,
        "total_aptos": aptos,
        "percentual": round(100 * total / aptos, 2) if aptos else None,
        "secoes": len(por_secao),
        "filtros": {"zona": zona, "municipio": municipio},
    }


def secao_details(zona=None, secao=None):
    """Todos os dados de uma seção."""
    z, err = _parse_int(zona, "zona")
    if err:
        return err
    s, err = _parse_int(secao, "secao")
    if err:
        return err

    r = bu[(bu["ZONA_N"] == z) & (bu["SECAO_N"] == s)]
    if r.empty:
        return {"error": f"seção {z}/{s} não encontrada"}

    primeira = r.iloc[0]
    votos = (
        r.groupby(["DS_CARGO_PERGUNTA", "DS_TIPO_VOTAVEL"], as_index=False)["QT_VOTOS_N"]
        .sum()
        .assign(QT_VOTOS=lambda d: d["QT_VOTOS_N"].astype(int))
        .drop(columns="QT_VOTOS_N")
        .to_dict(orient="records")
    )

    return {
        "zona": int(z),
        "secao": int(s),
        "municipio": primeira["NM_MUNICIPIO"],
        "aptos": int(_num(primeira["QT_APTOS"])),
        "comparecimento": int(_num(primeira["QT_COMPARECIMENTO"])),
        "abstencoes": int(_num(primeira["QT_ABSTENCOES"])),
        "sem_biometria": int(_num(primeira["QT_ELEI_BIOM_SEM_HABILITACAO"])),
        "votos": votos,
    }


def compare_secoes(zona_a=None, secao_a=None, zona_b=None, secao_b=None, tipo_voto=None):
    """Compara duas seções lado a lado, agregando por cargo."""
    tipo_voto, err = _validar_tipo_voto(tipo_voto)
    if err:
        return err

    pares = [
        ("zona_a", zona_a),
        ("secao_a", secao_a),
        ("zona_b", zona_b),
        ("secao_b", secao_b),
    ]
    ids = []
    for nome, valor in pares:
        v, err = _parse_int(valor, nome)
        if err:
            return err
        ids.append(v)

    secoes = []
    for i in range(0, 4, 2):
        z, s = ids[i], ids[i + 1]
        r_full = bu[(bu["ZONA_N"] == z) & (bu["SECAO_N"] == s)]
        if r_full.empty:
            return {"error": f"seção {z}/{s} não encontrada"}

        r = r_full if tipo_voto is None else r_full[r_full.DS_TIPO_VOTAVEL == tipo_voto]
        primeira = r_full.iloc[0]
        votos_cargo = r.groupby("DS_CARGO_PERGUNTA")["QT_VOTOS_N"].sum().to_dict()

        secoes.append(
            {
                "chave": f"{z}/{s}",
                "NR_ZONA": int(primeira["ZONA_N"]),
                "NR_SECAO": int(primeira["SECAO_N"]),
                "NM_MUNICIPIO": primeira["NM_MUNICIPIO"],
                "QT_APTOS": int(_num(primeira["QT_APTOS"])),
                "QT_COMPARECIMENTO": int(_num(primeira["QT_COMPARECIMENTO"])),
                "QT_ABSTENCOES": int(_num(primeira["QT_ABSTENCOES"])),
                "votos_por_cargo": {k: int(v) for k, v in votos_cargo.items()},
                "total_votos": int(r["QT_VOTOS_N"].sum()),
                "AGREGADA": bool(primeira["AGREGADA"]),
                "DS_SECOES_AGREGADAS": primeira["DS_SECOES_AGREGADAS"],
            }
        )

    a, b = secoes[0], secoes[1]
    best = {
        "mais_aptos": [a["chave"] if a["QT_APTOS"] >= b["QT_APTOS"] else b["chave"]],
        "mais_comparecimento": [
            a["chave"] if a["QT_COMPARECIMENTO"] >= b["QT_COMPARECIMENTO"] else b["chave"]
        ],
        "mais_abstencoes": [
            a["chave"] if a["QT_ABSTENCOES"] >= b["QT_ABSTENCOES"] else b["chave"]
        ],
        "mais_votos": [
            a["chave"] if a["total_votos"] >= b["total_votos"] else b["chave"]
        ],
    }

    vencedor = best["mais_votos"][0]
    resposta = (
        f"{a['chave']} tem mais votos: {a['total_votos']} vs {b['total_votos']}"
        if vencedor == a["chave"]
        else f"{b['chave']} tem mais votos: {b['total_votos']} vs {a['total_votos']}"
    )

    resultado = {
        "resposta": resposta,
        "secoes": secoes,
        "best": best,
        "tipo_voto_aplicado": tipo_voto,
    }

    agregadas = [s["chave"] for s in secoes if s["AGREGADA"]]
    if agregadas:
        resultado["aviso"] = (
            f"As seções {', '.join(agregadas)} são CONSOLIDADAS. "
            "QT_APTOS, QT_COMPARECIMENTO e QT_ABSTENCOES não são "
            "comparáveis entre elas e uma seção individual."
        )

    return resultado


def search_secoes(tipo_voto=None, zona=None, cargo=None, min_votos=None, top_n=5):
    """Busca seções no BU por tipo de voto, zona, cargo e mínimo de votos."""
    tipo_voto = _clean_opt(tipo_voto)
    zona = _clean_opt(zona)
    cargo = _clean_opt(cargo)
    min_votos = _clean_opt(min_votos)
    top_n = _clean_opt(top_n)

    # tolerância: se cargo recebeu um tipo de voto, move para tipo_voto
    if cargo in TIPOS:
        if tipo_voto is None:
            tipo_voto = cargo
        cargo = None

    cargo, err = _validar_cargo(cargo)
    if err:
        return err
    tipo_voto, err = _validar_tipo_voto(tipo_voto)
    if err:
        return err

    if top_n is None:
        top_n_int = 5
    else:
        top_n_int, err = _parse_int(top_n, "top_n")
        if err:
            return err
        if top_n_int <= 0:
            return {
                "error": (
                    f"top_n '{top_n}' inválido. "
                    "Use um número inteiro maior que zero."
                )
            }

    r = bu.copy()
    if tipo_voto is not None:
        r = r[r.DS_TIPO_VOTAVEL == tipo_voto]
    if zona is not None:
        zona_n, err = _parse_int(zona, "zona")
        if err:
            return err
        if zona_n <= 0:
            return {
                "error": (
                    f"zona {zona_n} não existe. "
                    "Omita o parâmetro zona para buscar em todas as zonas."
                )
            }
        r = r[r["ZONA_N"] == zona_n]
    if cargo is not None:
        r = r[r.DS_CARGO_PERGUNTA == cargo]
    if min_votos is not None:
        mv, err = _parse_int(min_votos, "min_votos")
        if err:
            return err
        if mv < 0:
            return {
                "error": (
                    f"min_votos '{min_votos}' inválido. "
                    "Use um número inteiro maior ou igual a zero."
                )
            }
        r = r[r["QT_VOTOS_N"] >= mv]

    if r.empty:
        return {"error": "nenhuma seção encontrada com esses filtros"}

    agg = (
        r.groupby(["ZONA_N", "SECAO_N", "NM_MUNICIPIO"], as_index=False)["QT_VOTOS_N"]
        .sum()
        .sort_values("QT_VOTOS_N", ascending=False)
    )
    top = agg.head(top_n_int)

    return {
        "total_found": len(agg),
        "resumo": f"{len(agg)} seções atendem ao filtro. Abaixo, as {len(top)} com mais votos.",
        "filtros_aplicados": {
            "tipo_voto": tipo_voto,
            "zona": zona,
            "cargo": cargo,
            "min_votos": min_votos,
        },
        "secoes": top.to_dict(orient="records"),
    }


def soma_secoes(zona_a=None, secao_a=None, zona_b=None, secao_b=None, tipo_voto=None):
    """Soma os votos de duas seções, opcionalmente filtrado por tipo de voto."""
    tipo_voto, err = _validar_tipo_voto(tipo_voto)
    if err:
        return err

    ids = []
    for nome, valor in (
        ("zona_a", zona_a),
        ("secao_a", secao_a),
        ("zona_b", zona_b),
        ("secao_b", secao_b),
    ):
        v, err = _parse_int(valor, nome)
        if err:
            return err
        ids.append(v)

    total = 0
    detalhe = []
    for i in range(0, 4, 2):
        z, s = ids[i], ids[i + 1]
        r_full = bu[(bu["ZONA_N"] == z) & (bu["SECAO_N"] == s)]
        if r_full.empty:
            return {"error": f"seção {z}/{s} não encontrada"}
        r = r_full if tipo_voto is None else r_full[r_full.DS_TIPO_VOTAVEL == tipo_voto]
        subtotal = int(r["QT_VOTOS_N"].sum())
        total += subtotal
        detalhe.append({"chave": f"{z}/{s}", "votos": subtotal})

    return {
        "total": total,
        "detalhe": detalhe,
        "tipo_voto": tipo_voto,
        "resposta": f"Total: {total} ({' + '.join(str(d['votos']) for d in detalhe)})",
    }


FUNCTIONS = {
    "search_secoes": search_secoes,
    "secao_details": secao_details,
    "compare_secoes": compare_secoes,
    "total_por_tipo": total_por_tipo,
    "total_sem_biometria": total_sem_biometria,
    "soma_secoes": soma_secoes,
}

__all__ = [
    "FUNCTIONS",
    "search_secoes",
    "secao_details",
    "compare_secoes",
    "total_por_tipo",
    "total_sem_biometria",
    "soma_secoes",
]
