"""Testes unitários para o módulo de integração DET com dados sintéticos."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from tests.conftest import mapa_sintetico
from det import (
    CURSOS_DET,
    ROTULO_DET,
    carregar_det,
    classificar_mapas,
    conjuntos_det,
    eh_det,
    resumo_det,
    resumo_frequencia_det,
)


@pytest.fixture(autouse=True)
def _ler_xls_sintetico(monkeypatch: pytest.MonkeyPatch):
    """Permite que core.manipulacao._ler_xls_bruto receba DataFrames sintéticos diretamente."""
    import core.manipulacao as manipulacao

    original = manipulacao._ler_xls_bruto

    def _ler(arquivo_xls):
        if isinstance(arquivo_xls, pd.DataFrame):
            return arquivo_xls
        return original(arquivo_xls)

    monkeypatch.setattr(manipulacao, "_ler_xls_bruto", _ler)


@pytest.fixture
def mapas_sinteticos_det():
    """Gera mapa EST com 4 alunos (2 também no TT) e mapa TT com esses 2 alunos."""
    df_est = mapa_sintetico(
        curso="TÉCNICO EM ESTRADAS",
        bimestre=1,
        turma="EST.2A",
        disciplinas={
            "ING": "LÍNGUA ESTRANGEIRA: INGLÊS - 2ª SÉRIE",
            "TOP": "TOPOGRAFIA",
            "SOL": "LABORATÓRIO DE SOLOS",
        },
        alunos=[
            ("20260000001", "Aluno Compartilhado 1", {"ING": 15.0, "TOP": 12.0, "SOL": 12.0}, {"ING": 0, "TOP": 0, "SOL": 0}),
            ("20260000002", "Aluno Compartilhado 2", {"ING": 14.0, "TOP": 13.0, "SOL": 13.0}, {"ING": 0, "TOP": 0, "SOL": 0}),
            ("20260000003", "Aluno Estradas 3", {"ING": 16.0, "TOP": 14.0, "SOL": 14.0}, {"ING": 0, "TOP": 0, "SOL": 0}),
            ("20260000004", "Aluno Estradas 4", {"ING": 17.0, "TOP": 15.0, "SOL": 15.0}, {"ING": 0, "TOP": 0, "SOL": 0}),
        ],
    )

    df_tt = mapa_sintetico(
        curso="TÉCNICO EM TRÂNSITO",
        bimestre=1,
        turma="TRA.2A",
        disciplinas={
            "OPT": "OPERAÇÃO DE TRANSPORTES",
            "TRA": "PLANEJAMENTO DE TRANSPORTES",
        },
        alunos=[
            ("20260000001", "Aluno Compartilhado 1", {"OPT": 18.0, "TRA": 18.0}, {"OPT": 0, "TRA": 0}),
            ("20260000002", "Aluno Compartilhado 2", {"OPT": 19.0, "TRA": 19.0}, {"OPT": 0, "TRA": 0}),
        ],
    )
    return df_est, df_tt


def test_constantes_det():
    """Valida as constantes identificadoras do DET."""
    assert set(CURSOS_DET) == {"Estradas", "Trânsito"}
    assert ROTULO_DET == "Estradas + Trânsito (DET)"


def test_classificar_mapas(mapas_sinteticos_det):
    """Testa se classificar_mapas agrupa corretamente por curso_amigavel."""
    df_est, df_tt = mapas_sinteticos_det
    classificados = classificar_mapas([df_est, df_tt])

    assert "Estradas" in classificados
    assert "Trânsito" in classificados
    assert len(classificados["Estradas"]) == 1
    assert len(classificados["Trânsito"]) == 1
    assert classificados["Estradas"][0] is df_est
    assert classificados["Trânsito"][0] is df_tt


def test_eh_det(mapas_sinteticos_det):
    """Testa que eh_det retorna True apenas quando ambos os cursos do DET estão presentes."""
    df_est, df_tt = mapas_sinteticos_det
    classificados = classificar_mapas([df_est, df_tt])

    # Verdadeiro com ambos os cursos
    assert eh_det([df_est, df_tt]) is True
    assert eh_det(classificados) is True
    assert eh_det(["Estradas", "Trânsito"]) is True
    assert eh_det("Estradas", "Trânsito") is True

    # Falso com apenas um curso ou cursos incorretos
    assert eh_det([df_est]) is False
    assert eh_det([df_tt]) is False
    assert eh_det({"Estradas": [df_est]}) is False
    assert eh_det(["Estradas"]) is False
    assert eh_det(["Trânsito"]) is False
    assert eh_det(["Estradas", "Edificações"]) is False
    assert eh_det(["Estradas", "Trânsito", "Informática"]) is False
    assert eh_det([]) is False


def test_carregar_det_lado_vazio_gera_value_error(mapas_sinteticos_det):
    """Testa que carregar_det com um lado vazio levanta ValueError."""
    df_est, df_tt = mapas_sinteticos_det

    with pytest.raises(ValueError):
        carregar_det([df_est])

    with pytest.raises(ValueError):
        carregar_det([df_tt])

    with pytest.raises(ValueError):
        carregar_det([df_est], [])

    with pytest.raises(ValueError):
        carregar_det([], [df_tt])

    with pytest.raises(ValueError):
        carregar_det({"Estradas": [df_est], "Trânsito": []})


def test_carregar_det_e_resumo_det_sintetico(mapas_sinteticos_det):
    """Testa fluxo completo de carregamento e resumo DET com dados sintéticos."""
    df_est, df_tt = mapas_sinteticos_det

    # Carregamento
    cd = carregar_det([df_est, df_tt])
    assert isinstance(cd, conjuntos_det)
    conj_tt, conj_est = cd
    assert len(conj_tt) == 1
    assert len(conj_est) == 1

    # Resumo
    res = resumo_det(cd)
    assert res.alunos_total == 4
    assert res.alunos_por_curso == {"Estradas": 2, "Trânsito": 2}
    assert res.intersecao == 0

    # Acesso como dicionário
    assert res["alunos_total"] == 4
    assert res["alunos_por_curso"] == {"Estradas": 2, "Trânsito": 2}
    assert res["intersecao"] == 0

    # Resumo gerado diretamente dos mapas
    res_direto = resumo_det([df_est, df_tt])
    assert res_direto.alunos_total == 4
    assert res_direto.alunos_por_curso == {"Estradas": 2, "Trânsito": 2}
    assert res_direto.intersecao == 0

    # Faltas somadas dos dois lados (mapas sintéticos com faltas zeradas)
    assert res_direto.faltas_por_curso == {"Estradas": 0, "Trânsito": 0}
    assert res_direto.faltas_total == 0


def test_resumo_det_sem_matricula_nem_nome_no_str(mapas_sinteticos_det):
    """Garante que str(resumo_det) não expõe nenhuma matrícula nem nome de aluno (LGPD)."""
    df_est, df_tt = mapas_sinteticos_det
    res = resumo_det([df_est, df_tt])
    texto = str(res)

    # Nenhuma matrícula deve constar
    for mat in ("20260000001", "20260000002", "20260000003", "20260000004"):
        assert mat not in texto, f"Matrícula {mat} encontrada em str(resumo_det)"

    # Nenhum nome deve constar
    for nome in ("Aluno", "Compartilhado", "Estradas 3", "Estradas 4"):
        assert nome not in texto, f"Nome {nome} encontrado em str(resumo_det)"


def test_resumo_frequencia_det_sintetico(
    mapas_sinteticos_det: tuple[pd.DataFrame, pd.DataFrame],
    planilha_ch_sintetica: Path,
) -> None:
    """Verifica resumo_frequencia_det com dados sintéticos e planilha de CH efetiva.

    - Disciplina de núcleo comum (INGLÊS) aparece uma vez com n_alunos = soma dos lados (2 + 2 = 4).
    - Disciplina técnica de Estradas (TOPOGRAFIA) aparece com escopo 'Estradas'.
    - Disciplina técnica de Trânsito (OPERAÇÃO DE TRANSPORTES) aparece com escopo 'Trânsito'.
    - Sem horário identificado por curso.
    - Minimização de dados (sem matrícula/nome).
    """
    df_est, df_tt = mapas_sinteticos_det
    df_res, sem_horario = resumo_frequencia_det([df_est, df_tt], planilha_ch_sintetica, bimestre=1)

    # Núcleo comum
    df_nc = df_res[df_res["escopo"] == "Núcleo comum (EST/TT)"]
    assert len(df_nc) == 1
    assert df_nc.iloc[0]["disciplina"] == "INGLÊS"
    assert df_nc.iloc[0]["n_alunos"] == 4  # soma dos lados (2 em EST + 2 em TT)

    # Técnicas
    df_est_tec = df_res[df_res["escopo"] == "Estradas"]
    assert len(df_est_tec) == 1
    assert df_est_tec.iloc[0]["disciplina"] == "TOPOGRAFIA"
    assert df_est_tec.iloc[0]["n_alunos"] == 2

    df_tt_tec = df_res[df_res["escopo"] == "Trânsito"]
    assert len(df_tt_tec) == 1
    assert df_tt_tec.iloc[0]["disciplina"] == "OPERAÇÃO DE TRANSPORTES"
    assert df_tt_tec.iloc[0]["n_alunos"] == 2

    # Sem horário por curso
    assert "LABORATÓRIO DE SOLOS" in sem_horario["Estradas"]
    assert "PLANEJAMENTO DE TRANSPORTES" in sem_horario["Trânsito"]

    # LGPD: sem dados discentes
    assert "matricula" not in df_res.columns
    assert "nome" not in df_res.columns


def test_resumo_frequencia_det_ch_divergente_forcada(
    mapas_sinteticos_det: tuple[pd.DataFrame, pd.DataFrame],
    planilha_ch_sintetica: Path,
) -> None:
    """Verifica que CH do bimestre divergente no núcleo comum levanta ValueError."""
    from ch_efetiva import carregar_ch_efetiva

    df_est, df_tt = mapas_sinteticos_det
    df_ch = carregar_ch_efetiva(planilha_ch_sintetica)

    # Força divergência na disciplina de núcleo comum (INGLÊS) entre Estradas e Trânsito
    mask_ing = df_ch["disciplina_norm"] == "INGLES"
    row_est = df_ch[mask_ing].iloc[0:1].copy()
    row_est["curso"] = "Estradas"
    row_est["ch_bim_1"] = 18

    row_tt = df_ch[mask_ing].iloc[0:1].copy()
    row_tt["curso"] = "Trânsito"
    row_tt["ch_bim_1"] = 22

    df_ch_divergente = pd.concat([df_ch[~mask_ing], row_est, row_tt], ignore_index=True)

    with pytest.raises(ValueError, match="diverge"):
        resumo_frequencia_det([df_est, df_tt], df_ch_divergente, bimestre=1)

