"""Cálculo e agregação de frequência discente a partir dos dados da DAE.

Implementa as decisões de arquitetura:
- D5: Frequência ponderada agregada por bimestre (Σ presenciadas / Σ ofertadas).
  O mês é a granularidade de entrada da DAE, nada sai por mês no relatório.
  Limiar legal de presença: 75%.
- D6: Mapeamento mês -> bimestre como parâmetro rastreável (período de apuração),
  com marcação de bimestre parcial.
- D11: Mês de referência definido pelo último mês lançado na base.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

# Calendário padrão de referência (ordem cronológica dos meses da DAE)
MESES_CALENDARIO: list[str] = [
    "fevereiro",
    "marco",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
]

# Ordem canônica dos meses do ano para ordenação robusta de meses lançados
_ORDEM_MESES_ANO: list[str] = [
    "janeiro",
    "fevereiro",
    "marco",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
]

# D6: Mapeamento default provisório mês -> bimestre (CEFET-MG)
MESES_POR_BIMESTRE: dict[int, list[str]] = {
    1: ["fevereiro", "marco", "abril"],
    2: ["maio", "junho", "julho"],
    3: ["agosto", "setembro"],
    4: ["outubro", "novembro"],
}


def meses_lancados(df: pd.DataFrame) -> list[str]:
    """Retorna a lista de meses que possuem algum 'ha_ofertadas' > 0, na ordem do calendário.

    Examina as colunas do tipo 'ha_ofertadas_<mes>' no DataFrame.
    """
    if df.empty:
        return []

    # Identifica todas as colunas de horas ofertadas
    colunas_ofertadas = [
        c for c in df.columns if c.startswith("ha_ofertadas_")
    ]
    if not colunas_ofertadas:
        return []

    meses_presentes: set[str] = set()
    for col in colunas_ofertadas:
        mes = col.removeprefix("ha_ofertadas_")
        valores = pd.to_numeric(df[col], errors="coerce")
        if (valores > 0).any():
            meses_presentes.add(mes)

    # Ordena os meses presentes de acordo com a ordem do calendário
    resultado: list[str] = []
    for m in _ORDEM_MESES_ANO:
        if m in meses_presentes:
            resultado.append(m)
            meses_presentes.remove(m)

    # Caso haja algum mês não mapeado no calendário padrão, inclui ao final mantendo estabilidade
    for m in sorted(meses_presentes):
        resultado.append(m)

    return resultado


def mes_referencia(df: pd.DataFrame) -> str | None:
    """Retorna o último mês lançado no DataFrame (D11).

    Retorna None se nenhum mês tiver horas ofertadas lançadas ou se a base for vazia.
    """
    lancados = meses_lancados(df)
    if not lancados:
        return None
    return lancados[-1]


def periodo_apuracao(
    df: pd.DataFrame,
    meses_por_bimestre: dict[int, list[str]] | None = None,
    bimestres: Sequence[int] | int | None = None,
) -> dict[int, dict[str, object]]:
    """Determina o período de apuração por bimestre (D6).

    Retorna um dicionário mapeando cada bimestre pedido para:
        - meses_calendario: lista de meses previstos no calendário do bimestre.
        - meses_lancados: lista de meses efetivamente lançados (com ha_ofertadas > 0).
        - parcial: bool indicando se faltam meses do calendário a serem lançados.
    """
    mapa_bimestres = (
        dict(meses_por_bimestre)
        if meses_por_bimestre is not None
        else MESES_POR_BIMESTRE
    )
    if bimestres is None:
        lista_bimestres = sorted(mapa_bimestres.keys())
    elif isinstance(bimestres, int):
        lista_bimestres = [bimestres]
    else:
        lista_bimestres = [int(b) for b in bimestres]

    todos_lancados = set(meses_lancados(df))

    resultado: dict[int, dict[str, object]] = {}
    for b in lista_bimestres:
        meses_cal = list(mapa_bimestres.get(b, []))
        meses_efetivos = [m for m in meses_cal if m in todos_lancados]
        parcial = len(meses_efetivos) < len(meses_cal)
        resultado[b] = {
            "meses_calendario": meses_cal,
            "meses_lancados": meses_efetivos,
            "parcial": parcial,
        }

    return resultado


def frequencia_ponderada(
    df: pd.DataFrame | pd.Series,
    meses: Sequence[str],
) -> pd.Series | float:
    """Calcula a frequência ponderada (D5): Σ presenciadas / Σ ofertadas.

    Retorna NaN se Σ ofertadas for 0 ou se todos os meses estiverem vazios/nulos.
    Se a entrada for uma pd.Series (um aluno), retorna float (ou np.nan).
    Se for pd.DataFrame, retorna pd.Series com o índice correspondente.
    """
    if isinstance(df, pd.Series):
        df_df = pd.DataFrame([df])
        serie = _calcular_frequencia_ponderada_df(df_df, meses)
        val = serie.iloc[0]
        return float(val) if pd.notna(val) else np.nan

    return _calcular_frequencia_ponderada_df(df, meses)


def _calcular_frequencia_ponderada_df(
    df: pd.DataFrame,
    meses: Sequence[str],
) -> pd.Series:
    """Implementação vetorizada da frequência ponderada para pd.DataFrame."""
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    cols_ofertadas = [
        f"ha_ofertadas_{m}" for m in meses if f"ha_ofertadas_{m}" in df.columns
    ]
    cols_presenciadas = [
        f"ha_presenciadas_{m}"
        for m in meses
        if f"ha_presenciadas_{m}" in df.columns
    ]

    if not cols_ofertadas:
        return pd.Series(np.nan, index=df.index, dtype=float)

    df_ofer = df[cols_ofertadas].apply(pd.to_numeric, errors="coerce")
    df_pres = (
        df[cols_presenciadas].apply(pd.to_numeric, errors="coerce")
        if cols_presenciadas
        else pd.DataFrame(0.0, index=df.index, columns=[])
    )

    # Soma as horas ofertadas e presenciadas
    soma_ofertadas = df_ofer.sum(axis=1, min_count=1)
    soma_presenciadas = df_pres.sum(axis=1, min_count=1)

    # Onde ofertadas for <= 0 ou nulo, frequência é NaN
    mascara_invalido = soma_ofertadas.isna() | (soma_ofertadas <= 0)

    # Onde ofertadas for válido mas presenciadas for nulo, assume 0 presenças
    soma_presenciadas_ajustada = soma_presenciadas.fillna(0.0)

    resultado = soma_presenciadas_ajustada / soma_ofertadas
    resultado = resultado.mask(mascara_invalido, np.nan)
    return resultado.astype(float)


def tabela_frequencia(
    df: pd.DataFrame,
    meses_por_bimestre: dict[int, list[str]] | None = None,
    bimestres: Sequence[int] | int | None = None,
) -> pd.DataFrame:
    """Gera tabela consolidada de frequência por bimestre e acumulada (D5).

    Colunas de saída adicionadas:
        - freq_bim_<n>: frequência ponderada para cada bimestre pedido.
        - freq_acumulada: frequência ponderada acumulada sobre os meses dos bimestres pedidos.
        - abaixo_75: bool indicando se a freq_acumulada é estritamente < 0.75 (False se NaN).
        - diff_vs_dae_pp: diferença entre freq_acumulada e acumulado_dae em pontos percentuais (p.p.).

    Nenhuma coluna mensal ('ha_ofertadas_*' ou 'ha_presenciadas_*') é mantida na saída.
    """
    mapa_bimestres = (
        dict(meses_por_bimestre)
        if meses_por_bimestre is not None
        else MESES_POR_BIMESTRE
    )
    if bimestres is None:
        lista_bimestres = sorted(mapa_bimestres.keys())
    elif isinstance(bimestres, int):
        lista_bimestres = [bimestres]
    else:
        lista_bimestres = [int(b) for b in bimestres]

    # Descarta colunas mensais de entrada (nenhuma coluna por mês na saída)
    cols_base = [
        c
        for c in df.columns
        if not (c.startswith("ha_ofertadas_") or c.startswith("ha_presenciadas_"))
    ]
    res = df[cols_base].copy()

    # Frequência por bimestre
    meses_acumulados: list[str] = []
    for b in lista_bimestres:
        meses_bim = mapa_bimestres.get(b, [])
        for m in meses_bim:
            if m not in meses_acumulados:
                meses_acumulados.append(m)
        res[f"freq_bim_{b}"] = frequencia_ponderada(df, meses_bim)

    # Frequência acumulada apenas sobre os meses dos bimestres pedidos
    freq_acum = frequencia_ponderada(df, meses_acumulados)
    res["freq_acumulada"] = freq_acum

    # Alerta abaixo de 75%: estritamente < 0.75 e não nulo
    res["abaixo_75"] = (freq_acum < 0.75) & freq_acum.notna()

    # Diferença vs acumulado DAE em pontos percentuais (p.p.)
    if "acumulado_dae" in df.columns:
        acumulado_num = pd.to_numeric(df["acumulado_dae"], errors="coerce")
        res["diff_vs_dae_pp"] = (freq_acum - acumulado_num) * 100.0
    else:
        res["diff_vs_dae_pp"] = pd.Series(np.nan, index=df.index, dtype=float)

    return res
