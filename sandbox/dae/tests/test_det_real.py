"""Testes de validação do módulo DET com os mapas reais de Estradas e Trânsito.

Executa verificações estruturais e estatísticas sobre os arquivos oficiais
'Estradas_2025-2026.xls' e 'Transito_2025-2026.xls' em sandbox/dae/dados/.
Mensagens de asserção operam exclusivamente sobre contagens agregadas (LGPD).
"""

from __future__ import annotations

from pathlib import Path
import pytest

from det import (
    CAMINHO_CH_EFETIVA_PADRAO,
    CAMINHO_ESTRADAS_PADRAO,
    CAMINHO_TRANSITO_PADRAO,
    carregar_det,
    classificar_mapas,
    eh_det,
    resumo_det,
    resumo_frequencia_det,
)

pytestmark = pytest.mark.skipif(
    not (
        CAMINHO_ESTRADAS_PADRAO.exists()
        and CAMINHO_TRANSITO_PADRAO.exists()
        and CAMINHO_CH_EFETIVA_PADRAO.exists()
    ),
    reason="Mapas reais Estradas/Trânsito ou CH efetiva não encontrados em sandbox/dae/dados/.",
)


def test_classificar_mapas_reais() -> None:
    """Verifica que classificar_mapas dos 2 arquivos de dados/ resulta em um de cada curso."""
    arquivos = [CAMINHO_ESTRADAS_PADRAO, CAMINHO_TRANSITO_PADRAO]
    grupos = classificar_mapas(arquivos)

    assert len(grupos.get("Estradas", [])) == 1, (
        f"Esperado 1 mapa de Estradas, obtido {len(grupos.get('Estradas', []))}"
    )
    assert len(grupos.get("Trânsito", [])) == 1, (
        f"Esperado 1 mapa de Trânsito, obtido {len(grupos.get('Trânsito', []))}"
    )
    assert len(grupos) == 2, f"Esperado 2 grupos, obtido {len(grupos)}"


def test_eh_det_reais() -> None:
    """Verifica que os dois mapas reais juntos caracterizam estritamente o DET."""
    arquivos = [CAMINHO_ESTRADAS_PADRAO, CAMINHO_TRANSITO_PADRAO]
    assert eh_det(arquivos) is True, "Esperado eh_det True para a lista dos 2 mapas reais"

    grupos = classificar_mapas(arquivos)
    assert eh_det(grupos) is True, "Esperado eh_det True para o dicionário dos 2 mapas reais"


def test_resumo_det_real() -> None:
    """Verifica resumo estatístico oficial do DET com dados reais (45 total, 24 EST, 21 TT, interseção 0)."""
    arquivos = [CAMINHO_ESTRADAS_PADRAO, CAMINHO_TRANSITO_PADRAO]
    res = resumo_det(arquivos)

    assert res["alunos_total"] == 45, (
        f"Esperado total de 45 alunos, obtido {res['alunos_total']}"
    )
    assert res["alunos_por_curso"] == {"Estradas": 24, "Trânsito": 21}, (
        f"Esperado 24 alunos em Estradas e 21 em Trânsito, obtido {res['alunos_por_curso']}"
    )
    assert res["intersecao"] == 0, (
        f"Esperada interseção 0, obtido {res['intersecao']}"
    )
    assert res["bimestres"] == [1], (
        f"Esperado 1º bimestre [1], obtido {res['bimestres']}"
    )
    assert res["serie"] == 2, (
        f"Esperada série 2, obtido {res['serie']}"
    )
    assert res["disciplinas_por_curso"] == {"Estradas": 17, "Trânsito": 16}, (
        f"Esperado 17 disciplinas em Estradas e 16 em Trânsito, obtido {res['disciplinas_por_curso']}"
    )
    assert res["faltas_por_curso"] == {"Estradas": 1045, "Trânsito": 990}, (
        f"Esperadas 1045 faltas em Estradas e 990 em Trânsito, obtido {res['faltas_por_curso']}"
    )
    assert res["faltas_total"] == 2035, (
        f"Esperado total de 2035 faltas, obtido {res['faltas_total']}"
    )


def test_resumo_frequencia_det_real() -> None:
    """Verifica resumo de frequência integrado DET com mapas reais e CH efetiva (D1).

    - Núcleo comum: 10 disciplinas casadas, com n_alunos=45 cada (soma 24 EST + 21 TT).
    - Técnicas casadas: Estradas 3, Trânsito 3.
    - Sem horário: Estradas 4 / Trânsito 3.
    - Nenhuma CH NaN nas casadas.
    """
    df, sem_horario = resumo_frequencia_det(
        [CAMINHO_TRANSITO_PADRAO, CAMINHO_ESTRADAS_PADRAO],
        CAMINHO_CH_EFETIVA_PADRAO,
        bimestre=1,
    )

    df_nc = df[df["escopo"] == "Núcleo comum (EST/TT)"]
    assert len(df_nc) == 10, f"Esperado 10 disciplinas no núcleo comum, obtido {len(df_nc)}"
    assert (df_nc["n_alunos"] == 45).all(), "Esperado n_alunos=45 em todas as disciplinas de núcleo comum"

    df_est = df[df["escopo"] == "Estradas"]
    assert len(df_est) == 3, f"Esperado 3 disciplinas técnicas casadas em Estradas, obtido {len(df_est)}"

    df_tt = df[df["escopo"] == "Trânsito"]
    assert len(df_tt) == 3, f"Esperado 3 disciplinas técnicas casadas em Trânsito, obtido {len(df_tt)}"

    assert len(sem_horario.get("Estradas", [])) == 4, (
        f"Esperado 4 disciplinas sem horário em Estradas, obtido {len(sem_horario.get('Estradas', []))}"
    )
    assert len(sem_horario.get("Trânsito", [])) == 3, (
        f"Esperado 3 disciplinas sem horário em Trânsito, obtido {len(sem_horario.get('Trânsito', []))}"
    )

    assert df["ch_bim"].notna().all(), "Nenhuma CH do bimestre deve ser NaN nas casadas"

    # Verificação de minimização de dados / LGPD
    assert "matricula" not in df.columns
    assert "nome" not in df.columns

