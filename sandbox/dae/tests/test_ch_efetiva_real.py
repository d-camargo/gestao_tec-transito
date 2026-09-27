"""Testes de validação da planilha real de Carga Horária Efetiva (2026).

Executa verificações estruturais e estatísticas (C13) sobre o arquivo
oficial 'CH_Efetiva_Disciplinas_Integrado_2026.xlsx' em sandbox/dae/dados/.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from ch_efetiva import (
    CAMINHO_CH_EFETIVA_PADRAO,
    CAMINHO_ESTRADAS_PADRAO,
    PADRAO_TURMA,
    carregar_ch_efetiva,
    casar_disciplinas,
    ch_da_turma,
    divergencias_calendario,
    resumo_por_carga,
)
from calendario import carregar_calendario, faixa_ch
from core.manipulacao import processar_multiplos_bimestres
from frequencia import frequencia_por_disciplina, resumo_frequencia_por_disciplina

pytestmark = pytest.mark.skipif(
    not CAMINHO_CH_EFETIVA_PADRAO.exists(),
    reason="Planilha real CH_Efetiva_Disciplinas_Integrado_2026.xlsx não encontrada.",
)


@pytest.fixture(scope="module")
def cal():
    """Carrega o calendário oficial uma única vez para o módulo de testes."""
    return carregar_calendario()


@pytest.fixture(scope="module")
def df_real():
    """Carrega a planilha oficial real uma única vez para o módulo de testes."""
    return carregar_ch_efetiva(CAMINHO_CH_EFETIVA_PADRAO)


def test_linhas_e_cursos(df_real) -> None:
    """Verifica 309 linhas e 14 cursos na planilha oficial."""
    assert len(df_real) == 309
    assert df_real["curso"].nunique() == 14


def test_turmas_e_padrao(df_real) -> None:
    """Verifica 45 turmas distintas e aderência de todas ao padrão <CURSO>-<SERIE><LETRA>."""
    assert df_real["turma"].nunique() == 45
    for t in df_real["turma"]:
        assert PADRAO_TURMA.match(t) is not None, f"Turma fora do padrão: {t}"


def test_subgrupos(df_real) -> None:
    """Verifica distribuição de subgrupos: 270 sem divisão (""), 20 T1 e 19 T2."""
    assert (df_real["subgrupo"] == "").sum() == 270
    assert (df_real["subgrupo"] == "T1").sum() == 20
    assert (df_real["subgrupo"] == "T2").sum() == 19
    conts = df_real["subgrupo"].value_counts().to_dict()
    assert conts == {"": 270, "T1": 20, "T2": 19}


def test_aulas_semanais_distribuicao(df_real) -> None:
    """Verifica frequência das aulas semanais: {2: 240, 3: 32, 4: 29, 1: 8}."""
    conts_aulas = df_real["aulas_sem"].value_counts().to_dict()
    assert conts_aulas == {2: 240, 3: 32, 4: 29, 1: 8}


def test_ch_origem_planilha(df_real) -> None:
    """Verifica se ch_origem é 'planilha' no carregamento do arquivo real com cache."""
    assert df_real.attrs.get("ch_origem") == "planilha"


def test_resumo_por_carga_real(df_real) -> None:
    """Verifica resumo por carga oficial: 94 / 94 / 121 / 309."""
    resumo = resumo_por_carga(df_real)
    assert resumo == {"< 90%": 94, "90–95%": 94, "≥ 95%": 121, "total": 309}


def test_sem_coluna_professor(df_real) -> None:
    """Verifica que nenhuma coluna de professor está presente no DataFrame."""
    for col in df_real.columns:
        assert "professor" not in col.lower()
    assert "Professor(a)" not in df_real.columns


def test_divergencias_calendario_real(df_real, cal) -> None:
    """Verifica que a planilha real usa exatamente a tabela oficial do calendário."""
    divs = divergencias_calendario(df_real, cal)
    assert divs == [], f"Divergências encontradas: {len(divs)}"


def test_linhas_dentro_da_faixa_oficial(df_real, cal) -> None:
    """Verifica que toda linha da planilha real está dentro dos extremos oficiais de faixa_ch."""
    for _, row in df_real.iterrows():
        aulas_sem = int(row["aulas_sem"])
        min_ch, _, max_ch, _ = faixa_ch(cal, aulas_sem, "A")
        ch_efetiva = int(row["ch_efetiva"])
        assert min_ch <= ch_efetiva <= max_ch, (
            f"Linha fora dos extremos oficiais: turma={row['turma']}, disc={row['disciplina']}, "
            f"CH={ch_efetiva} fora de [{min_ch}, {max_ch}]"
        )


def test_ch_efetiva_minima_e_maxima(df_real) -> None:
    """Verifica que a CH efetiva mínima é 35 e a máxima é 152."""
    assert df_real["ch_efetiva"].min() == 35
    assert df_real["ch_efetiva"].max() == 152


@pytest.mark.skipif(
    not CAMINHO_ESTRADAS_PADRAO.exists(),
    reason="Mapa real Estradas_2025-2026.xls não encontrado em sandbox/dae/dados/.",
)
def test_casamento_mapa_real_estradas(df_real) -> None:
    """Verifica casamento do mapa real de Estradas com a planilha de CH efetiva (C17)."""
    conjuntos = processar_multiplos_bimestres([CAMINHO_ESTRADAS_PADRAO])
    meta = conjuntos[0][3]
    legenda_real = conjuntos[0][2]

    # ch_da_turma = 14 linhas
    df_turma = ch_da_turma(df_real, meta["curso_amigavel"], 2, meta["turma"])
    assert len(df_turma) == 14

    # 13 casadas e as 4 sem linha
    casadas, sem_linha = casar_disciplinas(legenda_real, df_turma)
    assert len(casadas) == 13
    assert len(sem_linha) == 4

    esperado_sem_linha = {
        "EDUCAÇÃO FÍSICA - 2ª SÉRIE",
        "LABORATÓRIO DE SOLOS",
        "LABORATÓRIO DE DESENHO TOPOGRÁFICO",
        "LABORATÓRIO DE TOPOGRAFIA",
    }
    assert set(sem_linha) == esperado_sem_linha

    # ch_bim_1 das casadas = GEOGRAFIA 31, FÍSICA 30, MATEMÁTICA 30, TOPOGRAFIA 22,
    # LÍNGUA PORTUGUESA 22, QUÍMICA 22, SOLOS 20, FILOSOFIA 20, HISTÓRIA 20, REDAÇÃO 20,
    # MÁQUINAS E EQUIPAMENTOS 18, BIOLOGIA 18, INGLÊS 18
    esperado_ch_b1 = {
        "GEOGRAFIA": 31,
        "FÍSICA": 30,
        "MATEMÁTICA": 30,
        "TOPOGRAFIA": 22,
        "LÍNGUA PORTUGUESA": 22,
        "QUÍMICA": 22,
        "SOLOS": 20,
        "FILOSOFIA": 20,
        "HISTÓRIA": 20,
        "REDAÇÃO": 20,
        "MÁQUINAS E EQUIPAMENTOS": 18,
        "BIOLOGIA": 18,
        "INGLÊS": 18,
    }

    # ch_bim_1 das casadas por nome da disciplina na planilha (chaves de casadas
    # são os códigos da legenda)
    ch_bim_1_por_disc = {c["disciplina"]: c["ch_bim_1"] for c in casadas.values()}
    assert ch_bim_1_por_disc == esperado_ch_b1


@pytest.mark.skipif(
    not CAMINHO_ESTRADAS_PADRAO.exists(),
    reason="Mapa real Estradas_2025-2026.xls não encontrado em sandbox/dae/dados/.",
)
def test_frequencia_por_disciplina_mapa_real(df_real) -> None:
    """Verifica frequencia_por_disciplina e resumo com o mapa real no 1º bimestre (C17)."""
    conjuntos = processar_multiplos_bimestres([CAMINHO_ESTRADAS_PADRAO])
    df_notas, df_faltas, legenda_real, meta = conjuntos[0]

    df_turma = ch_da_turma(df_real, meta["curso_amigavel"], 2, meta["turma"])
    casadas, sem_linha = casar_disciplinas(legenda_real, df_turma)

    df_freq = frequencia_por_disciplina(df_faltas, legenda_real, casadas, bimestre=1)
    resumo = resumo_frequencia_por_disciplina(df_faltas, legenda_real, casadas, bimestre=1)

    # 13 disciplinas com fonte="planilha", 4 com fonte="sem horário"
    cont_planilha = sum(1 for r in resumo if r["fonte"] == "planilha")
    cont_sem_horario = sum(1 for r in resumo if r["fonte"] == "sem horário")
    assert cont_planilha == 13, f"Esperado 13 com fonte='planilha', obtido {cont_planilha}"
    assert cont_sem_horario == 4, f"Esperado 4 com fonte='sem horário', obtido {cont_sem_horario}"

    # n_alunos 45 em todas as disciplinas
    n_alunos_total = len(df_faltas)
    assert n_alunos_total == 45, f"Esperado 45 alunos, obtido {n_alunos_total}"
    assert all(r["n_alunos"] == 45 for r in resumo), (
        f"Esperado n_alunos=45 em todas as {len(resumo)} disciplinas"
    )

    # Resumo não contém matrícula nem nome em nenhum dict (só chaves de C17)
    assert all("matricula" not in r for r in resumo), "Chave 'matricula' não deve constar no resumo"
    assert all("nome" not in r for r in resumo), "Chave 'nome' não deve constar no resumo"
    assert len(resumo) == 17, f"Esperado 17 disciplinas no resumo, obtido {len(resumo)}"

    # Frequência não-NaN em 45 × 13 = 585 pares (aluno, disciplina)
    cols_disc = [c for c in df_faltas.columns if c not in ("matricula", "nome")]
    pares_nao_nan = int(df_freq[cols_disc].notna().sum().sum())
    assert pares_nao_nan == 585, f"Esperado 585 pares não-NaN, obtido {pares_nao_nan}"

    # Frequência NaN nas 4 disciplinas sem horário: 45 × 4 = 180 pares
    pares_nan = int(df_freq[cols_disc].isna().sum().sum())
    assert pares_nan == 180, f"Esperado 180 pares NaN, obtido {pares_nan}"


