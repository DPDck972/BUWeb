"""Carregamento e normalização do BU — reutilizado do pj_eleicao_MCP."""
from pathlib import Path

import pandas as pd

from .config import DATA_PATH, resolve_path


def load_bu(path: str | Path | None = None) -> pd.DataFrame:
    """Carrega o BU e prepara as colunas derivadas."""
    data_path = resolve_path(str(path)) if path is not None else DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")

    last_error = None
    for encoding in ("utf-8-sig", "latin1", "cp1252"):
        try:
            df = pd.read_csv(
                data_path,
                sep=";",
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise last_error

    # Compatibilidade de layout: alguns arquivos usam DS_CARGO em vez de DS_CARGO_PERGUNTA.
    if "DS_CARGO_PERGUNTA" not in df.columns:
        if "DS_CARGO" in df.columns:
            df["DS_CARGO_PERGUNTA"] = df["DS_CARGO"]
        else:
            raise ValueError(
                "Coluna de cargo não encontrada: esperado DS_CARGO_PERGUNTA ou DS_CARGO."
            )

    df = df.rename(columns={"NR_ZONA": "ZONA_N", "NR_SECAO": "SECAO_N"})
    df["QT_VOTOS_N"] = pd.to_numeric(df["QT_VOTOS"], errors="coerce").fillna(0)
    df["ZONA_N"] = pd.to_numeric(df["ZONA_N"], errors="coerce")
    df["SECAO_N"] = pd.to_numeric(df["SECAO_N"], errors="coerce")
    df["AGREGADA"] = df["DS_SECOES_AGREGADAS"] != "#NULO#"
    return df


def _clean_opt(valor):
    """Trata strings vazias ou só com espaços como filtro omitido (None)."""
    if valor is None:
        return None
    if isinstance(valor, str):
        limpo = valor.strip()
        return limpo if limpo else None
    return valor


def _parse_int(valor, nome):
    """Converte um valor para inteiro ou devolve erro claro e acionável."""
    if isinstance(valor, bool) or valor is None:
        return None, {"error": f"{nome} '{valor}' inválido. Use um número inteiro."}
    if isinstance(valor, str):
        texto = valor.strip()
        if not texto:
            return None, {"error": f"{nome} '' inválido. Use um número inteiro."}
        try:
            return int(texto), None
        except ValueError:
            try:
                f = float(texto)
                if f.is_integer():
                    return int(f), None
            except ValueError:
                pass
            return None, {"error": f"{nome} '{valor}' inválido. Use um número inteiro."}
    if isinstance(valor, float):
        if valor.is_integer():
            return int(valor), None
        return None, {"error": f"{nome} '{valor}' inválido. Use um número inteiro."}
    try:
        return int(valor), None
    except (TypeError, ValueError):
        return None, {"error": f"{nome} '{valor}' inválido. Use um número inteiro."}


def _num(v, default=0):
    n = pd.to_numeric(v, errors="coerce")
    return default if pd.isna(n) else n
