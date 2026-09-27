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
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from .calendario import (
        Calendario,
        carregar_calendario,
        ch_lecionada,
        dias_por_mes_bimestre,
        faixa_ch,
        sabados_do_responsavel,
    )
except ImportError:
    from calendario import (
        Calendario,
        carregar_calendario,
        ch_lecionada,
        dias_por_mes_bimestre,
        faixa_ch,
        sabados_do_responsavel,
    )


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

# D6: Mapeamento default confirmado pelo calendário oficial 2026 (dezembro, 4 dias, fora da planilha da DAE)
MESES_POR_BIMESTRE: dict[int, list[str]] = {
    1: ["fevereiro", "marco", "abril"],
    2: ["maio", "junho", "julho"],
    3: ["agosto", "setembro"],
    4: ["outubro", "novembro"],
}

_MAPA_NUMERO_MES: dict[int, str] = {
    1: "janeiro",
    2: "fevereiro",
    3: "marco",
    4: "abril",
    5: "maio",
    6: "junho",
    7: "julho",
    8: "agosto",
    9: "setembro",
    10: "outubro",
    11: "novembro",
    12: "dezembro",
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
    calendario: Calendario | None = None,
    cenario: str = "A",
    curso: str | None = None,
) -> dict[int, dict[str, object]]:
    """Determina o período de apuração por bimestre (D6, C6).

    Retorna um dicionário mapeando cada bimestre pedido para:
        - meses_calendario: lista de meses previstos no calendário do bimestre.
        - meses_lancados: lista de meses efetivamente lançados (com ha_ofertadas > 0).
        - parcial: bool indicando se faltam meses do calendário a serem lançados.

    Se 'calendario' for fornecido:
        - dias_letivos: dias letivos apurados no cenário ('A' ou 'REAL').
        - dias_letivos_total: total oficial de dias letivos do bimestre (com sábados).
        - dias_por_mes: dicionário {nome_do_mes: qtd_dias} daquele bimestre.
        - limite_diarios: data-limite de fechamento de diários ('DD/MM').
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

    dias_mes_map: dict[int, dict[int, int]] | None = None
    if calendario is not None:
        dias_mes_map = dias_por_mes_bimestre(calendario)

    resultado: dict[int, dict[str, object]] = {}
    for b in lista_bimestres:
        meses_cal = list(mapa_bimestres.get(b, []))
        meses_efetivos = [m for m in meses_cal if m in todos_lancados]
        parcial = len(meses_efetivos) < len(meses_cal)
        info_b: dict[str, object] = {
            "meses_calendario": meses_cal,
            "meses_lancados": meses_efetivos,
            "parcial": parcial,
        }

        if calendario is not None and b in calendario.bimestres:
            bim_obj = calendario.bimestres[b]
            dias_uteis_b = sum(
                bim_obj.dias_semana.get(d, 0)
                for d in ("SEG", "TER", "QUA", "QUI", "SEX")
            )
            cenario_norm = (cenario or "A").upper()
            if cenario_norm == "REAL" and curso:
                sabs_curso = sabados_do_responsavel(
                    calendario, responsavel=curso, bimestres=[b]
                )
                dias_let_cenario = dias_uteis_b + len(sabs_curso)
            else:
                dias_let_cenario = dias_uteis_b

            dias_let_total = bim_obj.dias_letivos
            dias_m_b = dias_mes_map.get(b, {}) if dias_mes_map else {}
            dias_por_mes_nome = {
                _MAPA_NUMERO_MES.get(m_num, str(m_num)): qtd
                for m_num, qtd in dias_m_b.items()
            }

            info_b["dias_letivos"] = dias_let_cenario
            info_b["dias_letivos_total"] = dias_let_total
            info_b["dias_por_mes"] = dias_por_mes_nome
            info_b["limite_diarios"] = bim_obj.limite_diarios.strftime("%d/%m")

        resultado[b] = info_b

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



def meses_por_bimestre_do_calendario(cal: Calendario) -> dict[int, list[str]]:
    """Mapeia cada mês para o bimestre com mais dias letivos dele (C6).

    Derivado do próprio calendário via dias_por_mes_bimestre; os nomes dos meses
    seguem _ORDEM_MESES_ANO (ex.: "marco", sem acento). Equivale a
    MESES_POR_BIMESTRE com "dezembro" a mais no 4º bimestre para o calendário
    2026 (4 dias letivos em dezembro, fora da planilha da DAE, que vai até
    novembro).
    """
    contagem: dict[int, dict[int, int]] = {}
    for b_num, meses in dias_por_mes_bimestre(cal).items():
        for m_num, dias in meses.items():
            contagem.setdefault(m_num, {})[b_num] = dias

    mapa: dict[int, list[str]] = {b: [] for b in cal.bimestres}
    for m_num in sorted(contagem):
        b_maioria = max(contagem[m_num], key=lambda b: contagem[m_num][b])
        mapa[b_maioria].append(_MAPA_NUMERO_MES[m_num])
    return mapa


def limite_faltas(ch: int | float, limiar: float = 0.75) -> int:
    """Máximo de faltas que mantém frequência >= limiar: floor(ch x (1 - limiar)) (C6)."""
    if ch is None or pd.isna(ch) or ch < 0:
        return 0
    return int(np.floor(float(ch) * (1.0 - limiar)))


def _razao_frequencia(faltas: int | float | pd.Series, ch: int | float) -> float | pd.Series:
    """1 - faltas/ch com guarda de CH <= 0 (NaN); preserva Series e índice (C6)."""
    if isinstance(faltas, pd.Series):
        f = pd.to_numeric(faltas, errors="coerce")
        if ch is None or pd.isna(ch) or ch <= 0:
            return pd.Series(np.nan, index=f.index)
        return 1.0 - (f / ch)
    if ch is None or pd.isna(ch) or ch <= 0:
        return np.nan
    if faltas is None or pd.isna(faltas):
        return np.nan
    return 1.0 - (float(faltas) / float(ch))


def frequencia_por_faltas(
    faltas: int | float | pd.Series,
    arranjo: dict[str, int],
    cal: Calendario,
    cenario: str = "A",
    bimestres: int | Iterable[int] | None = None,
    curso: str | None = None,
    sabado_reproduz: str | None = None,
) -> float | pd.Series:
    """Frequência a partir de faltas: 1 - faltas / ch_lecionada (C6).

    A carga horária lecionada é calculada pelo calendário a partir do arranjo de
    aulas por dia útil. Com ``faltas`` como ``pd.Series``, devolve ``pd.Series``
    com o mesmo índice. CH lecionada <= 0 (ex.: arranjo vazio) devolve NaN.
    """
    ch = ch_lecionada(
        cal,
        arranjo,
        cenario=cenario,
        bimestres=bimestres,
        curso=curso,
        sabado_reproduz=sabado_reproduz,
    )
    return _razao_frequencia(faltas, ch)


def faixa_frequencia_por_faltas(
    faltas: int | float,
    ch_semanal: int,
    cal: Calendario,
    cenario: str = "A",
    bimestres: int | Iterable[int] | None = None,
    curso: str | None = None,
) -> tuple[float, float]:
    """Faixa (pior, melhor) de frequência via faixa_ch do calendário (C6)."""
    min_ch, _, max_ch, _ = faixa_ch(
        cal,
        ch_semanal,
        cenario=cenario,
        bimestres=bimestres,
        curso=curso,
    )
    return (
        float(_razao_frequencia(faltas, min_ch)),
        float(_razao_frequencia(faltas, max_ch)),
    )


def frequencia_por_disciplina(
    df_faltas: pd.DataFrame,
    legenda: dict[str, str],
    ch_casada: dict[str, dict] | None,
    bimestre: int = 1,
) -> pd.DataFrame:
    """Frequência por aluno x código: 1 - faltas / ch_bim_<bimestre> (C17).

    A CH real da planilha substitui a estimativa por dia da semana: o
    denominador de cada disciplina é o ``ch_bim_<bimestre>`` casado da planilha
    de CH efetiva. Disciplina sem CH casada -> coluna NaN.
    """
    res = pd.DataFrame(index=df_faltas.index)
    for cod in legenda:
        casada = (ch_casada or {}).get(cod)
        ch_bim = casada.get(f"ch_bim_{bimestre}") if casada else None
        if not ch_bim or ch_bim <= 0:
            res[cod] = np.nan
        else:
            res[cod] = _razao_frequencia(
                pd.to_numeric(df_faltas[cod], errors="coerce"), ch_bim
            )
    return res


def resumo_frequencia_por_disciplina(
    df_faltas: pd.DataFrame,
    legenda: dict[str, str],
    ch_casada: dict[str, dict] | None,
    bimestre: int = 1,
    cal: Calendario | None = None,
    aulas_sem_estimadas: dict[str, int] | None = None,
) -> list[dict]:
    """Resumo agregado por disciplina com a precedência de C17 (C17).

    Precedência do denominador: CH da planilha (``fonte`` = "planilha") >
    faixa do calendário quando ``aulas_sem_estimadas`` traz o ``aulas_sem`` do
    código (``fonte`` = "estimada"; ``ch_bim`` vira o texto "mín-máx" de
    faixa_ch no bimestre e ``n_abaixo_75`` conta pelo mínimo, conservador) >
    sem horário (``fonte`` = "sem horário"; ``ch_bim``, ``limite_faltas_bim``
    e ``n_abaixo_75`` = None).

    Só agregados (LGPD): nenhum nome ou matrícula — chaves de C17 apenas.
    """
    if cal is None:
        cal = carregar_calendario()
    n_alunos = len(df_faltas)
    resumo: list[dict] = []

    for cod, nome in legenda.items():
        casada = (ch_casada or {}).get(cod)
        if casada:
            ch_bim = casada.get(f"ch_bim_{bimestre}")
            item = {
                "disciplina": casada["disciplina"],
                "aulas_sem": casada["aulas_sem"],
                "ch_bim": ch_bim,
                "ch_efetiva_ano": casada["ch_efetiva"],
                "ch_nominal": casada["ch_nominal"],
                "%_nominal": (
                    casada["ch_efetiva"] / casada["ch_nominal"]
                    if casada["ch_nominal"]
                    else None
                ),
                "limite_faltas_bim": limite_faltas(ch_bim),
                "n_alunos": n_alunos,
                "fonte": "planilha",
            }
            s_freq = _razao_frequencia(
                pd.to_numeric(df_faltas[cod], errors="coerce"), ch_bim
            )
            item["n_abaixo_75"] = (
                int(((s_freq < 0.75) & s_freq.notna()).sum())
                if isinstance(s_freq, pd.Series)
                else None
            )
        elif aulas_sem_estimadas and cod in aulas_sem_estimadas:
            aulas_sem = int(aulas_sem_estimadas[cod])
            min_ch, _, max_ch, _ = faixa_ch(cal, aulas_sem, "A", bimestres=bimestre)
            item = {
                "disciplina": nome,
                "aulas_sem": aulas_sem,
                "ch_bim": f"{min_ch}–{max_ch}",
                "ch_efetiva_ano": None,
                "ch_nominal": None,
                "%_nominal": None,
                "limite_faltas_bim": limite_faltas(min_ch),
                "n_alunos": n_alunos,
                "fonte": "estimada",
            }
            s_freq = _razao_frequencia(
                pd.to_numeric(df_faltas[cod], errors="coerce"), min_ch
            )
            item["n_abaixo_75"] = (
                int(((s_freq < 0.75) & s_freq.notna()).sum())
                if isinstance(s_freq, pd.Series)
                else None
            )
        else:
            item = {
                "disciplina": nome,
                "aulas_sem": None,
                "ch_bim": None,
                "ch_efetiva_ano": None,
                "ch_nominal": None,
                "%_nominal": None,
                "limite_faltas_bim": None,
                "n_alunos": n_alunos,
                "n_abaixo_75": None,
                "fonte": "sem horário",
            }
        resumo.append(item)

    return resumo
