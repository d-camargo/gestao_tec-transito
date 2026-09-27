"""Testes de validação com o mapa real de Estradas (2025-2026).

Executa verificações estruturais e estatísticas (C13) sobre o arquivo
oficial 'Estradas_2025-2026.xls' em sandbox/dae/dados/.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from calendario import carregar_calendario, sabados_do_responsavel
from cruzamento import _eh_matricula_valida, cruzar, resumo_mapa
from core.manipulacao import processar_multiplos_bimestres

CAMINHO_MAPA_REAL: Path = Path(__file__).resolve().parent.parent / "dados" / "Estradas_2025-2026.xls"
if not CAMINHO_MAPA_REAL.exists():
    CAMINHO_MAPA_REAL = Path("sandbox/dae/dados/Estradas_2025-2026.xls")

pytestmark = pytest.mark.skipif(
    not CAMINHO_MAPA_REAL.exists(),
    reason="Mapa real Estradas_2025-2026.xls não encontrado em sandbox/dae/dados/.",
)


@pytest.fixture(scope="module")
def cal():
    """Carrega o calendário escolar oficial de 2026."""
    return carregar_calendario()


@pytest.fixture(scope="module")
def conjuntos_real():
    """Processa o mapa real de Estradas (1º bimestre de 2026)."""
    return processar_multiplos_bimestres([str(CAMINHO_MAPA_REAL)])


def test_estrutura_e_metadados_mapa_real(conjuntos_real) -> None:
    """Verifica 1 conjunto e metadados oficiais do mapa real de Estradas."""
    assert len(conjuntos_real) == 1, f"Esperado 1 conjunto, obtido {len(conjuntos_real)}"
    _, _, _, meta = conjuntos_real[0]
    assert meta.get("bimestre_num") == 1, f"Esperado bimestre_num=1, obtido {meta.get('bimestre_num')}"
    assert meta.get("periodo_letivo") == "2026", f"Esperado periodo_letivo='2026', obtido {meta.get('periodo_letivo')}"
    assert meta.get("curso_amigavel") == "Estradas", f"Esperado curso_amigavel='Estradas', obtido {meta.get('curso_amigavel')}"
    assert meta.get("serie") == 2, f"Esperado serie=2, obtido {meta.get('serie')}"


def test_resumo_mapa_real(conjuntos_real) -> None:
    """Verifica resumo estatístico e de validação do mapa real: 45 alunos, 17 disciplinas, faltas 1.887 e mediana 28."""
    df_faltas = conjuntos_real[0][1]
    res = resumo_mapa(conjuntos_real)[0]

    assert res["n_alunos"] == 45, f"Esperado 45 alunos, obtido {res['n_alunos']}"
    assert res["n_matriculas_duplicadas"] == 0, (
        f"Esperado 0 duplicadas, obtido {res['n_matriculas_duplicadas']}"
    )

    matriculas = df_faltas["matricula"] if "matricula" in df_faltas.columns else pd.Series(dtype=object)
    n_validas = int(matriculas.apply(_eh_matricula_valida).sum())
    assert n_validas == 45, f"Esperado 45 matrículas válidas, obtido {n_validas}"
    assert res["n_matriculas_validas"] == 45, (
        f"Esperado 45 matrículas válidas, obtido {res['n_matriculas_validas']}"
    )

    assert res["n_disciplinas"] == 17, f"Esperado 17 disciplinas, obtido {res['n_disciplinas']}"
    assert res["faltas_total"] == 1887, f"Esperado total de faltas 1887, obtido {res['faltas_total']}"

    cols_disc = [c for c in df_faltas.columns if c not in ("matricula", "nome")]
    somas_aluno = df_faltas[cols_disc].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    mediana = float(somas_aluno.median())
    assert mediana == 28.0, f"Esperado mediana de faltas 28, obtido {mediana}"
    assert int(round(somas_aluno.sum())) == 1887, f"Esperado soma de faltas 1887, obtido {int(round(somas_aluno.sum()))}"
    assert res["faltas_mediana_aluno"] == 28.0, (
        f"Esperado mediana de faltas por aluno 28, obtido {res['faltas_mediana_aluno']}"
    )

    assert res["bimestre_num"] == 1, f"Esperado bimestre_num=1, obtido {res['bimestre_num']}"
    assert res["periodo_letivo"] == "2026", (
        f"Esperado periodo_letivo='2026', obtido {res['periodo_letivo']}"
    )
    assert res["curso_amigavel"] == "Estradas", (
        f"Esperado curso_amigavel='Estradas', obtido {res['curso_amigavel']}"
    )


def test_cruzamento_dae_em_memoria(conjuntos_real) -> None:
    """Verifica cruzamento contra df_dae com 30 primeiras matrículas + 5 fictícias de fev a abr."""
    df_faltas = conjuntos_real[0][1]
    mats_30 = list(df_faltas["matricula"].iloc[:30])
    mats_5 = [f"2026999900{i}" for i in range(1, 6)]

    meses = ["fevereiro", "marco", "abril"]
    linhas_dae = []
    for m in mats_30 + mats_5:
        row = {"matricula": m}
        for mes in meses:
            row[f"ha_ofertadas_{mes}"] = 100.0
            row[f"ha_presenciadas_{mes}"] = 90.0
        linhas_dae.append(row)
    df_dae = pd.DataFrame(linhas_dae)

    df_unido, resumo = cruzar(df_dae, conjuntos_real, bimestres=[1])

    assert resumo["nos_dois"] == 30, f"Esperado 30 nos_dois, obtido {resumo['nos_dois']}"
    assert resumo["so_app"] == 15, f"Esperado 15 so_app, obtido {resumo['so_app']}"
    assert resumo["so_dae"] == 5, f"Esperado 5 so_dae, obtido {resumo['so_dae']}"
    assert resumo["total"] == 50, f"Esperado total 50, obtido {resumo['total']}"

    casados_mask = df_unido["_merge"] == "both"
    n_casados_nao_nan = int(df_unido.loc[casados_mask, "diff_faltas_bim_1"].notna().sum())
    assert n_casados_nao_nan == 30, f"Esperado 30 diffs não-NaN nos casados, obtido {n_casados_nao_nan}"

    n_sobras_nan = int(df_unido.loc[~casados_mask, "diff_faltas_bim_1"].isna().sum())
    assert n_sobras_nan == 20, f"Esperado 20 diffs NaN nas sobras, obtido {n_sobras_nan}"

    n_total_nao_nan = int(df_unido["diff_faltas_bim_1"].notna().sum())
    assert n_total_nao_nan == 30, f"Esperado 30 diffs não-NaN no total, obtido {n_total_nao_nan}"
    assert (df_unido["diff_faltas_bim_1"].notna() == casados_mask).all(), (
        "Esperado diff_faltas_bim_1 não-NaN exatamente nos casados"
    )


def test_cruzamento_dae_vazio(conjuntos_real) -> None:
    """Verifica cruzamento contra df_dae vazio (só colunas) resultando em so_app 45."""
    cols_dae = [
        "matricula",
        "ha_ofertadas_fevereiro",
        "ha_presenciadas_fevereiro",
        "ha_ofertadas_marco",
        "ha_presenciadas_marco",
        "ha_ofertadas_abril",
        "ha_presenciadas_abril",
    ]
    df_dae_vazio = pd.DataFrame(columns=cols_dae)
    df_unido_vazio, resumo_vazio = cruzar(df_dae_vazio, conjuntos_real, bimestres=[1])

    assert resumo_vazio["so_app"] == 45, f"Esperado 45 so_app com DAE vazio, obtido {resumo_vazio['so_app']}"
    assert resumo_vazio["nos_dois"] == 0, f"Esperado 0 nos_dois com DAE vazio, obtido {resumo_vazio['nos_dois']}"
    assert resumo_vazio["so_dae"] == 0, f"Esperado 0 so_dae com DAE vazio, obtido {resumo_vazio['so_dae']}"
    assert resumo_vazio["total"] == 45, f"Esperado 45 total com DAE vazio, obtido {resumo_vazio['total']}"


def test_sabados_responsavel_estradas_b1(cal) -> None:
    """Verifica que sabados_do_responsavel para Estradas no 1º BI é vazio."""
    sabs = sabados_do_responsavel(cal, "Estradas", bimestres=[1])
    assert sabs == [], f"Esperado 0 sábados para Estradas no 1º BI, obtido {len(sabs)}"
