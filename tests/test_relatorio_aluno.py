"""Testes unitários para o módulo core/relatorio_aluno (Passos 1 e 2 do PLAN).

Cobre:
- listar_alunos: deduplica por matrícula entre bimestres/grupos e ordena por nome.
- comparar_aluno_turma:
  - média, mediana e diferença conferidas à mão numa turma sintética de 5 alunos;
  - posição na turma com rank(method='min', ascending=False) e tratamento de empates;
  - exclusão de disciplinas sem dados;
  - marcação de incompletas com ' *';
  - ausência de colunas de faltas quando df_faltas=None;
  - ValueError quando a matrícula solicitada não está na turma.
- evolucao_aluno_turma:
  - retorna None com menos de 2 bimestres;
  - retorna 3 linhas com porcentagens do MAX_PONTOS_BIMESTRE com 3 bimestres;
  - atribui NaN para pct_aluno quando o aluno estiver ausente num bimestre.
- criar_relatorio_aluno_pdf:
  - buffer não vazio e iniciando com %PDF em 1 e 3 bimestres (com e sem faltas);
  - LGPD (D4): atrás de pytest.importorskip('fitz'), apenas o aluno escolhido é nomeado.
"""

import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from core.relatorio_aluno import (
    QUADRANTES,
    _quadrante,
    comparar_aluno_turma,
    comparar_stem,
    consolidar_bimestres,
    criar_relatorio_aluno_pdf,
    evolucao_aluno_turma,
    grafico_dispersao_stem_aluno,
    listar_alunos,
)
from core.relatorios import calcular_estatisticas


def _turma_sintetica(bim_num=1):
    """Gera uma turma sintética de 5 alunos com dados e contas conhecidas à mão.

    Bimestre 1: max 20 pontos, limiar = 12.0.
    Notas:
      Ana   (2026001): MAT=18.0, FIS=16.0 -> Média = 17.0
      Beto  (2026002): MAT=14.0, FIS=14.0 -> Média = 14.0
      Caio  (2026003): MAT=12.0, FIS=10.0 -> Média = 11.0
      Duda  (2026004): MAT=10.0, FIS=8.0  -> Média = 9.0
      Elis  (2026005): MAT=6.0,  FIS=2.0  -> Média = 4.0

    Médias da turma:
      MAT: (18 + 14 + 12 + 10 + 6) / 5 = 60 / 5 = 12.0. Mediana MAT = 12.0.
      FIS: (16 + 14 + 10 + 8 + 2) / 5 = 50 / 5 = 10.0.  Mediana FIS = 10.0.
      Média Geral da Turma = (17 + 14 + 11 + 9 + 4) / 5 = 11.0.

    Faltas:
      Ana:  MAT=0, FIS=1 -> Total = 1
      Beto: MAT=1, FIS=0 -> Total = 1
      Caio: MAT=2, FIS=2 -> Total = 4
      Duda: MAT=3, FIS=4 -> Total = 7
      Elis: MAT=4, FIS=5 -> Total = 9
      Média faltas MAT = 10 / 5 = 2.0
      Média faltas FIS = 12 / 5 = 2.4
      Média faltas Total = 22 / 5 = 4.4
    """
    matriculas = ['2026001', '2026002', '2026003', '2026004', '2026005']
    nomes = ['Ana', 'Beto', 'Caio', 'Duda', 'Elis']
    df_notas = pd.DataFrame({
        'matricula': matriculas,
        'nome': nomes,
        'MAT': [18.0, 14.0, 12.0, 10.0, 6.0],
        'FIS': [18.0, 14.0, 10.0, 6.0, 2.0],
    })
    df_faltas = pd.DataFrame({
        'matricula': matriculas,
        'nome': nomes,
        'MAT': [0, 1, 2, 3, 4],
        'FIS': [1, 0, 2, 4, 5],
    })
    disc_dict = {
        'MAT': 'MATEMÁTICA - 2ª SÉRIE',
        'FIS': 'FÍSICA - 2ª SÉRIE',
    }
    metadados = {'bimestre_num': bim_num, 'turma': 'TRA.2A', 'periodo_letivo': '2026/1'}
    return df_notas, df_faltas, disc_dict, metadados


class TestCompararAlunoTurmaConferido:
    """Verifica média, mediana e diferença calculadas e conferidas à mão numa turma de 5."""

    def test_media_mediana_diferenca_conferidas_a_mao(self):
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)

        # 1. Aluna Ana (maior desempenho)
        res_ana = comparar_aluno_turma(df_notas, df_faltas, disc_dict, estat, '2026001')
        assert res_ana['aluno']['matricula'] == '2026001'
        assert res_ana['aluno']['nome'] == 'Ana'

        resumo_ana = res_ana['resumo']
        assert resumo_ana['media_aluno'] == pytest.approx(18.0)
        assert resumo_ana['media_turma'] == pytest.approx(11.0)
        assert resumo_ana['posicao'] == 1
        assert resumo_ana['total_alunos'] == 5
        assert resumo_ana['disciplinas_abaixo_limiar'] == 0
        assert resumo_ana['faltas_aluno'] == 1
        assert resumo_ana['media_faltas_turma'] == pytest.approx(4.4)

        df_ana = res_ana['por_disciplina']
        assert list(df_ana['codigo']) == ['MAT', 'FIS']

        # MAT: nota 18, média 12, mediana 12, dif 6
        mat_ana = df_ana[df_ana['codigo'] == 'MAT'].iloc[0]
        assert mat_ana['disciplina'] == 'Matemática - 2ª Série'
        assert mat_ana['nota_aluno'] == pytest.approx(18.0)
        assert mat_ana['media_turma'] == pytest.approx(12.0)
        assert mat_ana['mediana_turma'] == pytest.approx(12.0)
        assert mat_ana['diferenca'] == pytest.approx(6.0)
        assert mat_ana['faltas_aluno'] == 0
        assert mat_ana['media_faltas_turma'] == pytest.approx(2.0)

        # FIS: nota 18, média 10, mediana 10, dif 8
        fis_ana = df_ana[df_ana['codigo'] == 'FIS'].iloc[0]
        assert fis_ana['disciplina'] == 'Física - 2ª Série'
        assert fis_ana['nota_aluno'] == pytest.approx(18.0)
        assert fis_ana['media_turma'] == pytest.approx(10.0)
        assert fis_ana['mediana_turma'] == pytest.approx(10.0)
        assert fis_ana['diferenca'] == pytest.approx(8.0)
        assert fis_ana['faltas_aluno'] == 1
        assert fis_ana['media_faltas_turma'] == pytest.approx(2.4)

        # 2. Aluna Elis (menor desempenho)
        res_elis = comparar_aluno_turma(df_notas, df_faltas, disc_dict, estat, '2026005')
        resumo_elis = res_elis['resumo']
        assert resumo_elis['media_aluno'] == pytest.approx(4.0)
        assert resumo_elis['posicao'] == 5
        assert resumo_elis['disciplinas_abaixo_limiar'] == 2  # 6.0 e 2.0 < 12.0
        assert resumo_elis['faltas_aluno'] == 9

        df_elis = res_elis['por_disciplina']
        mat_elis = df_elis[df_elis['codigo'] == 'MAT'].iloc[0]
        assert mat_elis['nota_aluno'] == pytest.approx(6.0)
        assert mat_elis['diferenca'] == pytest.approx(-6.0)  # 6 - 12 = -6

        fis_elis = df_elis[df_elis['codigo'] == 'FIS'].iloc[0]
        assert fis_elis['nota_aluno'] == pytest.approx(2.0)
        assert fis_elis['diferenca'] == pytest.approx(-8.0)  # 2 - 10 = -8


class TestPosicaoComEmpate:
    """Verifica rank(method='min', ascending=False) com empates."""

    def test_posicao_com_empates(self):
        # 5 alunos com médias: 16.0, 16.0, 12.0, 8.0, 8.0
        # Posições esperadas: 1, 1, 3, 4, 4
        df_notas = pd.DataFrame({
            'matricula': ['M1', 'M2', 'M3', 'M4', 'M5'],
            'nome': ['Aluno 1', 'Aluno 2', 'Aluno 3', 'Aluno 4', 'Aluno 5'],
            'MAT': [16.0, 16.0, 12.0, 8.0, 8.0],
        })
        disc_dict = {'MAT': 'MATEMÁTICA'}
        meta = {'bimestre_num': 1}
        estat = calcular_estatisticas(df_notas, disc_dict, metadados=meta)

        r1 = comparar_aluno_turma(df_notas, None, disc_dict, estat, 'M1')
        r2 = comparar_aluno_turma(df_notas, None, disc_dict, estat, 'M2')
        r3 = comparar_aluno_turma(df_notas, None, disc_dict, estat, 'M3')
        r4 = comparar_aluno_turma(df_notas, None, disc_dict, estat, 'M4')
        r5 = comparar_aluno_turma(df_notas, None, disc_dict, estat, 'M5')

        assert r1['resumo']['posicao'] == 1
        assert r2['resumo']['posicao'] == 1
        assert r3['resumo']['posicao'] == 3
        assert r4['resumo']['posicao'] == 4
        assert r5['resumo']['posicao'] == 4


class TestDisciplinasSemDadosEIncompletas:
    """Verifica exclusão de disciplinas sem dados e marcação ' *' das incompletas."""

    def test_disciplina_sem_dados_excluida(self):
        # MAT tem notas; ART tem todas as notas zeradas (sem dados)
        df_notas = pd.DataFrame({
            'matricula': ['M1', 'M2'],
            'nome': ['A1', 'A2'],
            'MAT': [15.0, 17.0],
            'ART': [0.0, 0.0],
        })
        disc_dict = {'MAT': 'MATEMÁTICA', 'ART': 'ARTES'}
        meta = {'bimestre_num': 1}
        estat = calcular_estatisticas(df_notas, disc_dict, metadados=meta)

        assert 'ART' not in estat['disciplinas_com_notas']
        res = comparar_aluno_turma(df_notas, None, disc_dict, estat, 'M1')
        codigos = list(res['por_disciplina']['codigo'])
        assert 'ART' not in codigos
        assert 'MAT' in codigos

    def test_disciplina_incompleta_marcada_com_asterisco(self):
        # 10 alunos: MAT tem notas altas (max 18 de 20); BIO tem nota máxima 8 (<= 0.5 * 20 -> incompleta)
        df_notas = pd.DataFrame({
            'matricula': [f'M{i}' for i in range(10)],
            'nome': [f'Aluno {i}' for i in range(10)],
            'MAT': [18.0] * 10,
            'BIO': [8.0] * 10,  # Max observado = 8 <= 10 -> dispara sinal A
        })
        disc_dict = {'MAT': 'MATEMÁTICA', 'BIO': 'BIOLOGIA'}
        meta = {'bimestre_num': 1}
        estat = calcular_estatisticas(df_notas, disc_dict, metadados=meta)

        codigos_incompletas = [d['codigo'] for d in estat['disciplinas_incompletas']]
        assert 'BIO' in codigos_incompletas

        res = comparar_aluno_turma(df_notas, None, disc_dict, estat, 'M0')
        df_disc = res['por_disciplina']
        bio_row = df_disc[df_disc['codigo'] == 'BIO'].iloc[0]
        mat_row = df_disc[df_disc['codigo'] == 'MAT'].iloc[0]

        assert bio_row['disciplina'].endswith(' *')
        assert not mat_row['disciplina'].endswith(' *')


class TestFaltasOpcionaisEMatriculaAusente:
    """Verifica omissão de colunas sem faltas e lançamento de ValueError para matrícula ausente."""

    def test_sem_df_faltas_sem_colunas_de_faltas(self):
        df_notas, _, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=None, metadados=meta)

        res = comparar_aluno_turma(df_notas, None, disc_dict, estat, '2026001')
        assert 'faltas_aluno' not in res['por_disciplina'].columns
        assert 'media_faltas_turma' not in res['por_disciplina'].columns
        assert 'faltas_aluno' not in res['resumo']
        assert 'media_faltas_turma' not in res['resumo']

    def test_matricula_ausente_lanca_value_error(self):
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)

        with pytest.raises(ValueError, match="não encontrada"):
            comparar_aluno_turma(df_notas, df_faltas, disc_dict, estat, '9999999')


class TestEvolucaoAlunoTurma:
    """Verifica evolucao_aluno_turma com 1 bimestre e 3 bimestres."""

    def test_um_bimestre_retorna_none(self):
        conjunto1 = _turma_sintetica(1)
        res = evolucao_aluno_turma([conjunto1], '2026001')
        assert res is None

    def test_tres_bimestres_em_porcentagem_com_aluno_ausente(self):
        # Bimestre 1 (max 20): Ana tira MAT=16.0, FIS=16.0 -> média 16 / 20 = 80.0%
        # Bimestre 2 (max 30): Ana NÃO aparece no mapa deste bimestre -> pct_aluno deve ser NaN!
        # Bimestre 3 (max 20): Ana tira MAT=15.0, FIS=15.0 -> média 15 / 20 = 75.0%
        disc_dict = {'MAT': 'MATEMÁTICA', 'FIS': 'FÍSICA'}

        c1_notas = pd.DataFrame({
            'matricula': ['2026001', '2026002'],
            'nome': ['Ana', 'Beto'],
            'MAT': [16.0, 10.0],
            'FIS': [16.0, 10.0],
        })
        c1 = (c1_notas, None, disc_dict, {'bimestre_num': 1})

        # No bimestre 2, Ana está ausente (só Beto e Caio)
        c2_notas = pd.DataFrame({
            'matricula': ['2026002', '2026003'],
            'nome': ['Beto', 'Caio'],
            'MAT': [24.0, 18.0],
            'FIS': [24.0, 18.0],
        })
        c2 = (c2_notas, None, disc_dict, {'bimestre_num': 2})

        # No bimestre 3, Ana retorna
        c3_notas = pd.DataFrame({
            'matricula': ['2026001', '2026002'],
            'nome': ['Ana', 'Beto'],
            'MAT': [15.0, 12.0],
            'FIS': [15.0, 12.0],
        })
        c3 = (c3_notas, None, disc_dict, {'bimestre_num': 3})

        grupo = [c1, c2, c3]
        df_evolucao = evolucao_aluno_turma(grupo, '2026001')

        assert isinstance(df_evolucao, pd.DataFrame)
        assert len(df_evolucao) == 3
        assert list(df_evolucao.columns) == ['bimestre', 'pct_aluno', 'pct_turma']
        assert df_evolucao['bimestre'].tolist() == [1, 2, 3]

        # Linha 1 (Bimestre 1): 16/20 = 80%
        l1 = df_evolucao.iloc[0]
        assert l1['pct_aluno'] == pytest.approx(80.0)
        assert l1['pct_turma'] == pytest.approx((13.0 / 20.0) * 100)  # média turma = 13.0

        # Linha 2 (Bimestre 2): ausente -> NaN
        l2 = df_evolucao.iloc[1]
        assert np.isnan(l2['pct_aluno'])
        assert l2['pct_turma'] == pytest.approx((21.0 / 30.0) * 100)  # média turma = 21.0

        # Linha 3 (Bimestre 3): 15/20 = 75%
        l3 = df_evolucao.iloc[2]
        assert l3['pct_aluno'] == pytest.approx(75.0)
        assert l3['pct_turma'] == pytest.approx((13.5 / 20.0) * 100)


class TestListarAlunos:
    """Verifica deduplicação por matrícula entre bimestres e grupos e ordenação por nome."""

    def test_deduplicacao_e_ordenacao_por_nome(self):
        disc_dict = {'MAT': 'MATEMÁTICA'}
        # Grupo 1 (Bimestre 1 e 2)
        g1_b1 = (
            pd.DataFrame({'matricula': ['M02', 'M01'], 'nome': ['Beto', 'Ana'], 'MAT': [10.0, 15.0]}),
            None, disc_dict, {'bimestre_num': 1}
        )
        g1_b2 = (
            pd.DataFrame({'matricula': ['M01', 'M03'], 'nome': ['Ana', 'Carlos'], 'MAT': [12.0, 14.0]}),
            None, disc_dict, {'bimestre_num': 2}
        )

        # Grupo 2 (Bimestre 1 com homônimo de matrícula e novo aluno)
        g2_b1 = (
            pd.DataFrame({'matricula': ['M02', 'M04'], 'nome': ['Beto', 'Bruna'], 'MAT': [11.0, 16.0]}),
            None, disc_dict, {'bimestre_num': 1}
        )

        grupos = [[g1_b1, g1_b2], [g2_b1]]
        lista = listar_alunos(grupos)

        # Deve conter 4 alunos únicos: M01, M02, M03, M04
        # Ordenados por nome casefold: Ana, Beto, Bruna, Carlos
        assert lista == [
            ('M01', 'Ana'),
            ('M02', 'Beto'),
            ('M04', 'Bruna'),
            ('M03', 'Carlos'),
        ]

    def test_grupos_vazio(self):
        assert listar_alunos([]) == []
        assert listar_alunos(None) == []


def _turma_sintetica_tres_bimestres(com_faltas=True):
    """Gera 3 bimestres sintéticos para testes de evolução e relatórios multibimestrais."""
    conjuntos = []
    for bim in [1, 2, 3]:
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(bim)
        if not com_faltas:
            df_faltas = None
        conjuntos.append((df_notas, df_faltas, disc_dict, meta))
    return conjuntos


class TestCriarRelatorioAlunoPdf:
    """Verifica geração do relatório individual em PDF (Passo 2 do PLAN)."""

    def test_buffer_um_bimestre_com_faltas(self):
        """1 bimestre com faltas: buffer não vazio e começando com %PDF."""
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)
        comp = comparar_aluno_turma(df_notas, df_faltas, disc_dict, estat, '2026001')
        buf = criar_relatorio_aluno_pdf('Trânsito', comp, estat)
        val = buf.getvalue()
        assert val is not None
        assert len(val) > 0
        assert val.startswith(b'%PDF')

    def test_buffer_um_bimestre_sem_faltas(self):
        """1 bimestre sem faltas: buffer não vazio e começando com %PDF."""
        df_notas, _, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=None, metadados=meta)
        comp = comparar_aluno_turma(df_notas, None, disc_dict, estat, '2026001')
        buf = criar_relatorio_aluno_pdf('Trânsito', comp, estat)
        val = buf.getvalue()
        assert val is not None
        assert len(val) > 0
        assert val.startswith(b'%PDF')

    def test_buffer_tres_bimestres_com_faltas(self):
        """3 bimestres com faltas: buffer não vazio e começando com %PDF."""
        grupo = _turma_sintetica_tres_bimestres(com_faltas=True)
        df_notas, df_faltas, disc_dict, meta = grupo[-1]
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)
        comp = comparar_aluno_turma(df_notas, df_faltas, disc_dict, estat, '2026001')
        evo = evolucao_aluno_turma(grupo, '2026001')
        buf = criar_relatorio_aluno_pdf('Trânsito', comp, estat, evolucao=evo)
        val = buf.getvalue()
        assert val is not None
        assert len(val) > 0
        assert val.startswith(b'%PDF')

    def test_buffer_tres_bimestres_sem_faltas(self):
        """3 bimestres sem faltas: buffer não vazio e começando com %PDF."""
        grupo = _turma_sintetica_tres_bimestres(com_faltas=False)
        df_notas, df_faltas, disc_dict, meta = grupo[-1]
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=None, metadados=meta)
        comp = comparar_aluno_turma(df_notas, None, disc_dict, estat, '2026001')
        evo = evolucao_aluno_turma(grupo, '2026001')
        buf = criar_relatorio_aluno_pdf('Trânsito', comp, estat, evolucao=evo)
        val = buf.getvalue()
        assert val is not None
        assert len(val) > 0
        assert val.startswith(b'%PDF')

    def test_lgpd_apenas_aluno_escolhido_nomeado_um_bimestre(self):
        """1 bimestre: apenas o aluno escolhido é nomeado; nenhum outro aluno da turma (D4)."""
        fitz = pytest.importorskip('fitz')
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)
        comp = comparar_aluno_turma(df_notas, df_faltas, disc_dict, estat, '2026001')
        buf = criar_relatorio_aluno_pdf('Trânsito', comp, estat)

        doc = fitz.open(stream=buf.getvalue(), filetype='pdf')
        try:
            texto = '\n'.join(p.get_text() for p in doc)
        finally:
            doc.close()

        # O aluno escolhido deve constar no relatório
        assert 'Ana' in texto
        assert '2026001' in texto

        # Nenhum outro aluno da turma pode aparecer no PDF individual (D4)
        outros_alunos = ['Beto', 'Caio', 'Duda', 'Elis']
        for outro in outros_alunos:
            assert outro not in texto

        outras_matriculas = ['2026002', '2026003', '2026004', '2026005']
        for outra_mat in outras_matriculas:
            assert outra_mat not in texto

        # Nota de privacidade obrigatória
        assert "Os dados da turma são agregados; nenhum outro aluno é identificado neste relatório." in texto

    def test_lgpd_apenas_aluno_escolhido_nomeado_tres_bimestres(self):
        """3 bimestres: apenas o aluno escolhido é nomeado; nenhum outro aluno da turma (D4)."""
        fitz = pytest.importorskip('fitz')
        grupo = _turma_sintetica_tres_bimestres(com_faltas=True)
        df_notas, df_faltas, disc_dict, meta = grupo[-1]
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)
        comp = comparar_aluno_turma(df_notas, df_faltas, disc_dict, estat, '2026001')
        evo = evolucao_aluno_turma(grupo, '2026001')
        buf = criar_relatorio_aluno_pdf('Trânsito', comp, estat, evolucao=evo)

        doc = fitz.open(stream=buf.getvalue(), filetype='pdf')
        try:
            texto = '\n'.join(p.get_text() for p in doc)
        finally:
            doc.close()

        # O aluno escolhido deve constar no relatório
        assert 'Ana' in texto
        assert '2026001' in texto

        # Nenhum outro aluno da turma pode aparecer no PDF individual (D4)
        outros_alunos = ['Beto', 'Caio', 'Duda', 'Elis']
        for outro in outros_alunos:
            assert outro not in texto

        outras_matriculas = ['2026002', '2026003', '2026004', '2026005']
        for outra_mat in outras_matriculas:
            assert outra_mat not in texto

        # Nota de privacidade obrigatória
        assert "Os dados da turma são agregados; nenhum outro aluno é identificado neste relatório." in texto

    def test_buffer_pdf_stem_um_bimestre_com_e_sem_faltas(self):
        """PDF %PDF com STEM em 1 bimestre com e sem faltas."""
        matriculas = ['2026001', '2026002', '2026003']
        nomes = ['Ana', 'Beto', 'Caio']
        df_notas = pd.DataFrame({
            'matricula': matriculas,
            'nome': nomes,
            'MAT': [18.0, 14.0, 10.0],
            'FIS': [16.0, 12.0, 8.0],
            'QUI': [15.0, 11.0, 7.0],
        })
        df_faltas = pd.DataFrame({
            'matricula': matriculas,
            'nome': nomes,
            'MAT': [1, 2, 3],
            'FIS': [0, 1, 2],
            'QUI': [2, 0, 1],
        })
        disc_dict = {
            'MAT': 'MATEMÁTICA',
            'FIS': 'FÍSICA',
            'QUI': 'QUÍMICA',
        }
        meta = {'bimestre_num': 1, 'turma': 'TRA.2A'}

        # 1. Com faltas
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)
        comp = comparar_aluno_turma(df_notas, df_faltas, disc_dict, estat, '2026001')
        buf = criar_relatorio_aluno_pdf('Trânsito', comp, estat)
        val = buf.getvalue()
        assert val is not None and val.startswith(b'%PDF')

        # 2. Sem faltas
        estat_sf = calcular_estatisticas(df_notas, disc_dict, df_faltas=None, metadados=meta)
        comp_sf = comparar_aluno_turma(df_notas, None, disc_dict, estat_sf, '2026001')
        buf_sf = criar_relatorio_aluno_pdf('Trânsito', comp_sf, estat_sf)
        val_sf = buf_sf.getvalue()
        assert val_sf is not None and val_sf.startswith(b'%PDF')

    def test_buffer_pdf_stem_acumulado_tres_bimestres(self):
        """PDF %PDF com STEM acumulado de 3 bimestres."""
        c_test = TestConsolidarBimestres()
        c1, c2, c3 = c_test._dados_tres_bimestres()
        df_notas_pct, df_faltas_acum, disc_dict, estat_acum = consolidar_bimestres([c1, c2, c3])
        comp = comparar_aluno_turma(df_notas_pct, df_faltas_acum, disc_dict, estat_acum, '2026001')
        evo = evolucao_aluno_turma([c1, c2, c3], '2026001')
        buf = criar_relatorio_aluno_pdf('Trânsito', comp, estat_acum, evolucao=evo)
        val = buf.getvalue()
        assert val is not None and val.startswith(b'%PDF')

    def test_fitz_acumulado_rotulos_stem_e_privacidade(self):
        """No acumulado: texto contém Matemática, Física, Química, rótulos e não contém outro aluno (D4)."""
        fitz = pytest.importorskip('fitz')
        c_test = TestConsolidarBimestres()
        c1, c2, c3 = c_test._dados_tres_bimestres()
        df_notas_pct, df_faltas_acum, disc_dict, estat_acum = consolidar_bimestres([c1, c2, c3])
        comp = comparar_aluno_turma(df_notas_pct, df_faltas_acum, disc_dict, estat_acum, '2026001')
        evo = evolucao_aluno_turma([c1, c2, c3], '2026001')
        buf = criar_relatorio_aluno_pdf('Trânsito', comp, estat_acum, evolucao=evo)

        doc = fitz.open(stream=buf.getvalue(), filetype='pdf')
        try:
            texto = '\n'.join(p.get_text() for p in doc)
        finally:
            doc.close()

        texto_norm = re.sub(r'\s+', ' ', texto)

        # Rótulos exigidos
        assert "Matemática" in texto
        assert "Física" in texto
        assert "Química" in texto
        assert "Acumulado do 1º ao 3º Bimestre" in texto
        assert "Aproveitamento acumulado (%)" in texto_norm
        assert "Faltas acumuladas" in texto_norm
        assert "soma dos pontos obtidos ÷ soma dos pontos distribuídos nos bimestres enviados; limiar de 60%" in texto_norm

        # Aluno escolhido está no relatório
        assert "Ana" in texto
        assert "2026001" in texto

        # D4: Não contém nenhum outro aluno
        assert "Beto" not in texto
        assert "2026002" not in texto
        assert "Caio" not in texto
        assert "2026003" not in texto

    def test_fitz_um_bimestre_stem_e_privacidade(self):
        """1 bimestre com STEM: texto contém disciplinas e não contém outro aluno (D4); sem faltas traz aviso."""
        fitz = pytest.importorskip('fitz')
        matriculas = ['2026001', '2026002', '2026003']
        nomes = ['Ana', 'Beto', 'Caio']
        df_notas = pd.DataFrame({
            'matricula': matriculas,
            'nome': nomes,
            'MAT': [18.0, 14.0, 10.0],
            'FIS': [16.0, 12.0, 8.0],
            'QUI': [15.0, 11.0, 7.0],
        })
        df_faltas = pd.DataFrame({
            'matricula': matriculas,
            'nome': nomes,
            'MAT': [1, 2, 3],
            'FIS': [0, 1, 2],
            'QUI': [2, 0, 1],
        })
        disc_dict = {
            'MAT': 'MATEMÁTICA',
            'FIS': 'FÍSICA',
            'QUI': 'QUÍMICA',
        }
        meta = {'bimestre_num': 1, 'turma': 'TRA.2A'}

        # 1. Com faltas
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)
        comp = comparar_aluno_turma(df_notas, df_faltas, disc_dict, estat, '2026001')
        buf = criar_relatorio_aluno_pdf('Trânsito', comp, estat)

        doc = fitz.open(stream=buf.getvalue(), filetype='pdf')
        try:
            texto = '\n'.join(p.get_text() for p in doc)
        finally:
            doc.close()

        assert "Matemática" in texto
        assert "Física" in texto
        assert "Química" in texto
        assert "Ana" in texto
        assert "2026001" in texto

        # D4: Nenhum outro aluno
        assert "Beto" not in texto
        assert "2026002" not in texto
        assert "Caio" not in texto
        assert "2026003" not in texto

        # 2. Sem faltas: traz a nota sem faltas e nenhum outro aluno
        estat_sf = calcular_estatisticas(df_notas, disc_dict, df_faltas=None, metadados=meta)
        comp_sf = comparar_aluno_turma(df_notas, None, disc_dict, estat_sf, '2026001')
        buf_sf = criar_relatorio_aluno_pdf('Trânsito', comp_sf, estat_sf)

        doc_sf = fitz.open(stream=buf_sf.getvalue(), filetype='pdf')
        try:
            texto_sf = '\n'.join(p.get_text() for p in doc_sf)
        finally:
            doc_sf.close()

        assert "Sem faltas lançadas: os gráficos de dispersão notas × faltas não são gerados." in texto_sf
        assert "Ana" in texto_sf
        assert "2026001" in texto_sf
        assert "Beto" not in texto_sf
        assert "Caio" not in texto_sf

    def test_turma_sem_stem_omite_secao(self):
        """Turma sem disciplinas STEM gera PDF normalmente e omite a seção."""
        fitz = pytest.importorskip('fitz')
        disc_tec = {'TOP': 'TOPOGRAFIA', 'EST': 'ESTRADAS'}
        df_notas = pd.DataFrame({
            'matricula': ['2026001', '2026002'],
            'nome': ['Ana', 'Beto'],
            'TOP': [18.0, 14.0],
            'EST': [16.0, 12.0],
        })
        df_faltas = pd.DataFrame({
            'matricula': ['2026001', '2026002'],
            'nome': ['Ana', 'Beto'],
            'TOP': [1, 2],
            'EST': [0, 1],
        })
        meta = {'bimestre_num': 1, 'turma': 'TRA.2A'}
        estat = calcular_estatisticas(df_notas, disc_tec, df_faltas=df_faltas, metadados=meta)
        comp = comparar_aluno_turma(df_notas, df_faltas, disc_tec, estat, '2026001')
        buf = criar_relatorio_aluno_pdf('Trânsito', comp, estat)

        val = buf.getvalue()
        assert val is not None and val.startswith(b'%PDF')

        doc = fitz.open(stream=val, filetype='pdf')
        try:
            texto = '\n'.join(p.get_text() for p in doc)
        finally:
            doc.close()

        assert "Matemática, Física e Química" not in texto
        assert "Matemática" not in texto
        assert "Física" not in texto
        assert "Química" not in texto
        assert "Ana" in texto
        assert "2026001" in texto
        assert "Beto" not in texto


class TestConsolidarBimestres:
    """Testes para consolidar_bimestres (E2–E4)."""

    def _dados_tres_bimestres(self):
        # Bimestre 1 (max 20 pts)
        df_n1 = pd.DataFrame({
            'matricula': ['2026001', '2026002', '2026003'],
            'nome': ['Ana', 'Beto', 'Caio'],
            'MAT': [15.0, 10.0, 8.0],
            'FIS': [15.0, 12.0, 10.0],
            'BIO': [8.0, 6.0, 5.0],  # Max turma = 8.0 <= 10.0 -> incompleta (sinal A)
        })
        df_f1 = pd.DataFrame({
            'matricula': ['2026001', '2026002', '2026003'],
            'nome': ['Ana', 'Beto', 'Caio'],
            'MAT': [1, 0, 3],
            'FIS': [2, 1, 3],
            'BIO': [np.nan, 1, 2],  # Ana tem NaN, Beto/Caio têm faltas
        })
        disc1 = {'MAT': 'MATEMÁTICA', 'FIS': 'FÍSICA', 'BIO': 'BIOLOGIA'}
        meta1 = {'bimestre_num': 1, 'turma': 'TRA.2A'}
        c1 = (df_n1, df_f1, disc1, meta1)

        # Bimestre 2 (max 30 pts) — Caio ausente
        df_n2 = pd.DataFrame({
            'matricula': ['2026001', '2026002'],
            'nome': ['Ana', 'Beto'],
            'MAT': [21.0, 18.0],
            'FIS': [np.nan, 24.0],  # Ana sem nota lançada no 2º
            'BIO': [26.0, 20.0],
        })
        df_f2 = pd.DataFrame({
            'matricula': ['2026001', '2026002'],
            'nome': ['Ana', 'Beto'],
            'MAT': [3, 1],
            'FIS': [4, 2],
            'BIO': [np.nan, 2],  # Ana continua com NaN, Beto tem falta
        })
        disc2 = {'MAT': 'MATEMÁTICA', 'FIS': 'FÍSICA', 'BIO': 'BIOLOGIA'}
        meta2 = {'bimestre_num': 2, 'turma': 'TRA.2A'}
        c2 = (df_n2, df_f2, disc2, meta2)

        # Bimestre 3 (max 20 pts) — QUI só existe aqui
        df_n3 = pd.DataFrame({
            'matricula': ['2026001', '2026002'],
            'nome': ['Ana', 'Beto'],
            'MAT': [18.0, 14.0],
            'FIS': [16.0, 12.0],
            'BIO': [17.0, 15.0],
            'QUI': [14.0, 16.0],  # Disciplina que só existe no 3º
        })
        df_f3 = pd.DataFrame({
            'matricula': ['2026001', '2026002'],
            'nome': ['Ana', 'Beto'],
            'MAT': [2, 0],
            'FIS': [1, 0],
            'BIO': [np.nan, np.nan],
            'QUI': [0, 1],
        })
        disc3 = {'MAT': 'MATEMÁTICA', 'FIS': 'FÍSICA', 'BIO': 'BIOLOGIA', 'QUI': 'QUÍMICA TÉCNICA'}
        meta3 = {'bimestre_num': 3, 'turma': 'TRA.2A', 'periodo_letivo': '2026/1'}
        c3 = (df_n3, df_f3, disc3, meta3)

        return c1, c2, c3

    def test_grupo_menos_de_dois_bimestres_lanca_value_error(self):
        c1, _, _ = self._dados_tres_bimestres()
        with pytest.raises(ValueError):
            consolidar_bimestres([c1])
        with pytest.raises(ValueError):
            consolidar_bimestres([])
        with pytest.raises(ValueError):
            consolidar_bimestres(None)

    def test_consolidar_dois_bimestres_aproveitamento_e_regras(self):
        c1, c2, _ = self._dados_tres_bimestres()
        df_notas_pct, df_faltas_acum, disc_dict, estat_acum = consolidar_bimestres([c1, c2])

        # 1. limiar_aprovacao == 60.0 e metadados
        assert estat_acum['limiar_aprovacao'] == 60.0
        assert estat_acum['max_pontos_bimestre'] == 100
        assert estat_acum['acumulado'] is True
        assert estat_acum['bimestres_acumulados'] == [1, 2]
        assert estat_acum['bimestre_num'] == 2

        # 2. Aluno que só está no 1º bimestre (Caio '2026003') fica fora da turma
        assert len(df_notas_pct) == 2
        assert '2026003' not in df_notas_pct['matricula'].values
        assert list(df_notas_pct['matricula']) == ['2026001', '2026002']

        # 3. Aluno com 15/20 (1º) e 21/30 (2º) → 72,0%
        # Ana ('2026001') em MAT: (15 + 21) / (20 + 30) * 100 = 36 / 50 * 100 = 72.0%
        row_ana_n = df_notas_pct[df_notas_pct['matricula'] == '2026001'].iloc[0]
        assert row_ana_n['MAT'] == pytest.approx(72.0)

        # 4. Aluno sem nota no 2º numa disciplina → % só do 1º
        # Ana em FIS: 15/20 no 1º, sem nota no 2º → 15 / 20 * 100 = 75.0%
        assert row_ana_n['FIS'] == pytest.approx(75.0)

        # 5. Incompleta no 1º marca a disciplina
        codigos_inc = [d['codigo'] for d in estat_acum['disciplinas_incompletas']]
        assert 'BIO' in codigos_inc

        # 6. Faltas somadas com min_count=1
        assert df_faltas_acum is not None
        row_ana_f = df_faltas_acum[df_faltas_acum['matricula'] == '2026001'].iloc[0]
        # MAT: 1 + 3 = 4
        assert row_ana_f['MAT'] == pytest.approx(4.0)
        # FIS: 2 + 4 = 6
        assert row_ana_f['FIS'] == pytest.approx(6.0)
        # BIO: NaN no 1º e NaN no 2º → continua NaN
        assert pd.isna(row_ana_f['BIO'])

        # 7. comparar_aluno_turma roda sobre a saída sem erro
        comp = comparar_aluno_turma(df_notas_pct, df_faltas_acum, disc_dict, estat_acum, '2026001')
        assert comp['aluno']['matricula'] == '2026001'
        assert comp['aluno']['nome'] == 'Ana'
        assert comp['resumo']['posicao'] == 1
        assert comp['resumo']['total_alunos'] == 2
        assert comp['resumo']['faltas_aluno'] == 10  # 4 + 6 + 0
        df_disc = comp['por_disciplina']
        # BIO marcada com ' *'
        bio_row = df_disc[df_disc['codigo'] == 'BIO'].iloc[0]
        assert bio_row['disciplina'].endswith(' *')

    def test_consolidar_tres_bimestres_disciplina_que_so_existe_no_terceiro(self):
        c1, c2, c3 = self._dados_tres_bimestres()
        df_notas_pct, df_faltas_acum, disc_dict, estat_acum = consolidar_bimestres([c1, c2, c3])

        assert estat_acum['bimestres_acumulados'] == [1, 2, 3]
        assert estat_acum['bimestre_num'] == 3

        # Disciplina que só existe no 3º (QUI) → % só sobre 20 pts
        # Ana tirou 14.0 em QUI no 3º bimestre (valendo 20): 14 / 20 * 100 = 70.0%
        row_ana_n = df_notas_pct[df_notas_pct['matricula'] == '2026001'].iloc[0]
        assert row_ana_n['QUI'] == pytest.approx(70.0)

        # Faltas acumuladas nos 3 bimestres para Ana em MAT: 1 + 3 + 2 = 6
        row_ana_f = df_faltas_acum[df_faltas_acum['matricula'] == '2026001'].iloc[0]
        assert row_ana_f['MAT'] == pytest.approx(6.0)

        # comparar_aluno_turma roda sobre a saída sem erro
        comp = comparar_aluno_turma(df_notas_pct, df_faltas_acum, disc_dict, estat_acum, '2026001')
        assert comp['aluno']['matricula'] == '2026001'
        assert 'QUI' in list(comp['por_disciplina']['codigo'])

    def test_consolidar_bimestres_sem_faltas(self):
        c1, c2, _ = self._dados_tres_bimestres()
        c1_sem_f = (c1[0], None, c1[2], c1[3])
        c2_sem_f = (c2[0], None, c2[2], c2[3])
        df_notas_pct, df_faltas_acum, disc_dict, estat_acum = consolidar_bimestres([c1_sem_f, c2_sem_f])
        assert df_faltas_acum is None
        assert estat_acum['faltas_disponiveis'] is False
        assert estat_acum['_faltas_cols'] == []
        assert estat_acum['media_faltas_total'] == 0.0

        comp = comparar_aluno_turma(df_notas_pct, df_faltas_acum, disc_dict, estat_acum, '2026001')
        assert 'faltas_aluno' not in comp['resumo']
        assert 'faltas_aluno' not in comp['por_disciplina'].columns


class TestCompararStem:
    """Testes para comparar_stem, _quadrante e QUADRANTES (E5, E6, E8)."""

    def test_quadrantes_constante(self):
        """QUADRANTES contém os 4 rótulos e as cores padronizadas."""
        assert QUADRANTES == {
            'Em dia': 'seagreen',
            'Risco de frequência': '#e6a817',
            'Risco de desempenho': '#ff7f0e',
            'Risco duplo': '#d62728',
        }

    def test_ordem_stem_mat_fis_qui_sem_educacao_fisica(self):
        """Turma sintética com QUI, MAT, FIS e EDF fora de ordem → 3 linhas na ordem MAT/FIS/QUI, sem Ed. Física."""
        df_notas = pd.DataFrame({
            'matricula': ['2026001', '2026002'],
            'nome': ['Ana', 'Beto'],
            'QUI': [18.0, 12.0],
            'MAT': [18.0, 14.0],
            'FIS': [18.0, 10.0],
            'EDF': [20.0, 19.0],
        })
        df_faltas = pd.DataFrame({
            'matricula': ['2026001', '2026002'],
            'nome': ['Ana', 'Beto'],
            'QUI': [1, 2],
            'MAT': [0, 1],
            'FIS': [2, 0],
            'EDF': [0, 0],
        })
        disc_dict = {
            'QUI': 'QUÍMICA - 2ª SÉRIE',
            'MAT': 'MATEMÁTICA - 2ª SÉRIE',
            'FIS': 'FÍSICA - 2ª SÉRIE',
            'EDF': 'EDUCAÇÃO FÍSICA - 2ª SÉRIE',
        }
        meta = {'bimestre_num': 1}
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)

        df_stem = comparar_stem(df_notas, df_faltas, disc_dict, estat, '2026001')

        # 3 linhas na ordem MAT, FIS, QUI
        assert len(df_stem) == 3
        assert list(df_stem['codigo']) == ['MAT', 'FIS', 'QUI']
        assert list(df_stem['disciplina']) == [
            'Matemática - 2ª Série',
            'Física - 2ª Série',
            'Química - 2ª Série',
        ]
        # Educação Física não entra no grupo STEM
        assert 'EDF' not in list(df_stem['codigo'])
        assert not any('Educação Física' in d for d in df_stem['disciplina'])

        # Colunas completas quando há faltas
        assert list(df_stem.columns) == [
            'codigo', 'disciplina', 'nota_aluno', 'media_turma',
            'posicao', 'total_com_nota', 'faltas_aluno', 'media_faltas_turma', 'quadrante'
        ]

        # Dados da aluna Ana (2026001)
        row_mat = df_stem[df_stem['codigo'] == 'MAT'].iloc[0]
        assert row_mat['nota_aluno'] == pytest.approx(18.0)
        assert row_mat['media_turma'] == pytest.approx(16.0)
        assert row_mat['posicao'] == 1
        assert row_mat['total_com_nota'] == 2
        assert row_mat['faltas_aluno'] == 0
        assert row_mat['media_faltas_turma'] == pytest.approx(0.5)
        assert row_mat['quadrante'] == 'Em dia'

        row_fis = df_stem[df_stem['codigo'] == 'FIS'].iloc[0]
        assert row_fis['nota_aluno'] == pytest.approx(18.0)
        assert row_fis['media_turma'] == pytest.approx(14.0)
        assert row_fis['posicao'] == 1
        assert row_fis['total_com_nota'] == 2
        assert row_fis['faltas_aluno'] == 2
        assert row_fis['media_faltas_turma'] == pytest.approx(1.0)
        # Nota 18 >= limiar 12 e faltas 2 > media 1.0 -> Risco de frequência
        assert row_fis['quadrante'] == 'Risco de frequência'

    def test_bordas_do_quadrante(self):
        """Bordas: nota == limiar conta como acima; faltas == média conta como abaixo."""
        # Teste direto da função _quadrante
        limiar = 12.0
        media_f = 6.0

        # 1. nota == limiar, faltas == média -> 'Em dia'
        assert _quadrante(12.0, 6.0, limiar, media_f) == 'Em dia'
        # 2. nota == limiar, faltas > média -> 'Risco de frequência'
        assert _quadrante(12.0, 6.1, limiar, media_f) == 'Risco de frequência'
        # 3. nota < limiar, faltas == média -> 'Risco de desempenho'
        assert _quadrante(11.9, 6.0, limiar, media_f) == 'Risco de desempenho'
        # 4. nota < limiar, faltas > média -> 'Risco duplo'
        assert _quadrante(11.9, 6.1, limiar, media_f) == 'Risco duplo'
        # Nulos retornam None
        assert _quadrante(None, 6.0, limiar, media_f) is None
        assert _quadrante(12.0, None, limiar, media_f) is None
        assert _quadrante(np.nan, 6.0, limiar, media_f) is None

        # Teste na turma via comparar_stem
        # 6 alunos em MAT; faltas = [6, 6, 8, 4, 8, 4] -> soma 36 / 6 = 6.0 média de faltas exata!
        df_notas = pd.DataFrame({
            'matricula': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
            'nome': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
            'MAT': [12.0, 10.0, 12.0, 12.0, 10.0, 12.0],
        })
        df_faltas = pd.DataFrame({
            'matricula': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
            'nome': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
            'MAT': [6, 6, 8, 4, 8, 4],
        })
        disc_dict = {'MAT': 'MATEMÁTICA'}
        meta = {'bimestre_num': 1}  # limiar 12.0
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)

        # M1: nota 12.0 (== limiar), faltas 6 (== média 6.0) -> 'Em dia'
        res_m1 = comparar_stem(df_notas, df_faltas, disc_dict, estat, 'M1')
        assert res_m1.iloc[0]['quadrante'] == 'Em dia'

        # M2: nota 10.0 (< limiar), faltas 6 (== média 6.0) -> 'Risco de desempenho'
        res_m2 = comparar_stem(df_notas, df_faltas, disc_dict, estat, 'M2')
        assert res_m2.iloc[0]['quadrante'] == 'Risco de desempenho'

        # M3: nota 12.0 (== limiar), faltas 8 (> média 6.0) -> 'Risco de frequência'
        res_m3 = comparar_stem(df_notas, df_faltas, disc_dict, estat, 'M3')
        assert res_m3.iloc[0]['quadrante'] == 'Risco de frequência'

        # M5: nota 10.0 (< limiar), faltas 8 (> média 6.0) -> 'Risco duplo'
        res_m5 = comparar_stem(df_notas, df_faltas, disc_dict, estat, 'M5')
        assert res_m5.iloc[0]['quadrante'] == 'Risco duplo'

    def test_posicao_com_empate(self):
        """Posição por disciplina calculada com rank(method='min', ascending=False) e empates."""
        # 5 alunos: 18.0, 18.0, 15.0, 10.0, 10.0
        # Posições esperadas: 1, 1, 3, 4, 4
        df_notas = pd.DataFrame({
            'matricula': ['M1', 'M2', 'M3', 'M4', 'M5'],
            'nome': ['A1', 'A2', 'A3', 'A4', 'A5'],
            'FIS': [18.0, 18.0, 15.0, 10.0, 10.0],
        })
        disc_dict = {'FIS': 'FÍSICA'}
        meta = {'bimestre_num': 1}
        estat = calcular_estatisticas(df_notas, disc_dict, metadados=meta)

        p1 = comparar_stem(df_notas, None, disc_dict, estat, 'M1').iloc[0]
        p2 = comparar_stem(df_notas, None, disc_dict, estat, 'M2').iloc[0]
        p3 = comparar_stem(df_notas, None, disc_dict, estat, 'M3').iloc[0]
        p4 = comparar_stem(df_notas, None, disc_dict, estat, 'M4').iloc[0]
        p5 = comparar_stem(df_notas, None, disc_dict, estat, 'M5').iloc[0]

        assert p1['posicao'] == 1
        assert p2['posicao'] == 1
        assert p3['posicao'] == 3
        assert p4['posicao'] == 4
        assert p5['posicao'] == 4

        assert p1['total_com_nota'] == 5
        assert p3['total_com_nota'] == 5

    def test_sem_df_faltas_sem_colunas_faltas_e_quadrante(self):
        """Sem df_faltas → sem colunas faltas_aluno, media_faltas_turma e quadrante."""
        df_notas = pd.DataFrame({
            'matricula': ['M1', 'M2'],
            'nome': ['A1', 'A2'],
            'QUI': [15.0, 12.0],
        })
        disc_dict = {'QUI': 'QUÍMICA'}
        meta = {'bimestre_num': 1}
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=None, metadados=meta)

        df_stem = comparar_stem(df_notas, None, disc_dict, estat, 'M1')
        assert 'faltas_aluno' not in df_stem.columns
        assert 'media_faltas_turma' not in df_stem.columns
        assert 'quadrante' not in df_stem.columns
        assert list(df_stem.columns) == [
            'codigo', 'disciplina', 'nota_aluno', 'media_turma', 'posicao', 'total_com_nota'
        ]

    def test_mapa_so_com_tecnicas_retorna_vazio(self):
        """Mapa só com disciplinas técnicas → DataFrame vazio sem STEM."""
        df_notas = pd.DataFrame({
            'matricula': ['M1', 'M2'],
            'nome': ['A1', 'A2'],
            'TT1': [18.0, 14.0],
            'TT2': [16.0, 12.0],
        })
        df_faltas = pd.DataFrame({
            'matricula': ['M1', 'M2'],
            'nome': ['A1', 'A2'],
            'TT1': [0, 2],
            'TT2': [1, 0],
        })
        disc_dict = {
            'TT1': 'LABORATÓRIO DE SOLOS',
            'TT2': 'TOPOGRAFIA',
        }
        meta = {'bimestre_num': 1}
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)

        df_stem = comparar_stem(df_notas, df_faltas, disc_dict, estat, 'M1')
        assert df_stem.empty
        assert len(df_stem) == 0

        # Em comparar_aluno_turma, a chave 'stem' também deve estar vazia
        comp = comparar_aluno_turma(df_notas, df_faltas, disc_dict, estat, 'M1')
        assert comp['stem'].empty

    def test_matricula_ausente_lanca_value_error(self):
        """Matrícula ausente lança ValueError em comparar_stem."""
        df_notas = pd.DataFrame({
            'matricula': ['M1', 'M2'],
            'nome': ['A1', 'A2'],
            'MAT': [15.0, 12.0],
        })
        disc_dict = {'MAT': 'MATEMÁTICA'}
        meta = {'bimestre_num': 1}
        estat = calcular_estatisticas(df_notas, disc_dict, metadados=meta)

        with pytest.raises(ValueError, match="não encontrada"):
            comparar_stem(df_notas, None, disc_dict, estat, '9999999')

    def test_funciona_sobre_saida_de_consolidar_bimestres(self):
        """Funciona sobre a saída em % de consolidar_bimestres."""
        c_test = TestConsolidarBimestres()
        c1, c2, c3 = c_test._dados_tres_bimestres()
        df_notas_pct, df_faltas_acum, disc_dict, estat_acum = consolidar_bimestres([c1, c2, c3])

        df_stem = comparar_stem(df_notas_pct, df_faltas_acum, disc_dict, estat_acum, '2026001')

        # Contém MAT, FIS, QUI na ordem correta
        assert len(df_stem) == 3
        assert list(df_stem['codigo']) == ['MAT', 'FIS', 'QUI']

        # Limiar de aprovação acumulado é 60.0%
        assert estat_acum['limiar_aprovacao'] == 60.0

        # Ana ('2026001') em MAT: (15 + 21 + 18) / (20 + 30 + 20) * 100 = 54 / 70 * 100 = 77.14%
        row_mat = df_stem[df_stem['codigo'] == 'MAT'].iloc[0]
        assert row_mat['nota_aluno'] == pytest.approx(77.142857, rel=1e-3)
        assert row_mat['posicao'] == 1
        assert row_mat['total_com_nota'] == 2
        # Faltas de Ana em MAT: 1 + 3 + 2 = 6. Média turma = (6 + 1)/2 = 3.5.
        assert row_mat['faltas_aluno'] == 6
        assert row_mat['media_faltas_turma'] == pytest.approx(3.5)
        # Nota 77.14 >= 60.0 e faltas 6 > 3.5 -> Risco de frequência
        assert row_mat['quadrante'] == 'Risco de frequência'

    def test_comparar_aluno_turma_chaves_stem_e_dados(self):
        """comparar_aluno_turma devolve as chaves 'stem' e '_dados' sem alterar chaves existentes."""
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)

        res = comparar_aluno_turma(df_notas, df_faltas, disc_dict, estat, '2026001')

        # Chaves existentes intactas
        assert 'aluno' in res
        assert 'resumo' in res
        assert 'por_disciplina' in res

        # Novas chaves E8
        assert 'stem' in res
        assert '_dados' in res

        assert isinstance(res['stem'], pd.DataFrame)
        assert len(res['stem']) == 2  # MAT e FIS
        assert list(res['stem']['codigo']) == ['MAT', 'FIS']

        # _dados referencia df_notas e df_faltas
        dados_notas, dados_faltas = res['_dados']
        assert dados_notas is df_notas
        assert dados_faltas is df_faltas


class TestGraficoDispersaoStemAluno:
    """Testes para grafico_dispersao_stem_aluno (Passo 3 do PLAN / E7)."""

    def test_devolve_figure_com_faltas(self):
        """Devolve Figure quando há faltas disponíveis e aluno tem nota e faltas."""
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)
        fig = grafico_dispersao_stem_aluno(df_notas, df_faltas, estat, 'MAT', 'Matemática', '2026001')
        try:
            assert fig is not None
            assert isinstance(fig, matplotlib.figure.Figure)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_devolve_none_sem_faltas(self):
        """Devolve None quando faltas não estão disponíveis, df_faltas é None/vazio ou aluno sem faltas."""
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)

        # 1. df_faltas=None
        fig1 = grafico_dispersao_stem_aluno(df_notas, None, estat, 'MAT', 'Matemática', '2026001')
        assert fig1 is None

        # 2. df_faltas vazio
        fig2 = grafico_dispersao_stem_aluno(df_notas, pd.DataFrame(), estat, 'MAT', 'Matemática', '2026001')
        assert fig2 is None

        # 3. estat com faltas_disponiveis=False
        estat_sem_f = dict(estat, faltas_disponiveis=False)
        fig3 = grafico_dispersao_stem_aluno(df_notas, df_faltas, estat_sem_f, 'MAT', 'Matemática', '2026001')
        assert fig3 is None

        # 4. Código ausente em df_faltas
        df_f_sem_mat = df_faltas.drop(columns=['MAT'])
        fig4 = grafico_dispersao_stem_aluno(df_notas, df_f_sem_mat, estat, 'MAT', 'Matemática', '2026001')
        assert fig4 is None

        # 5. Aluno com falta NaN na disciplina
        df_f_nan = df_faltas.copy()
        df_f_nan.loc[df_f_nan['matricula'] == '2026001', 'MAT'] = np.nan
        fig5 = grafico_dispersao_stem_aluno(df_notas, df_f_nan, estat, 'MAT', 'Matemática', '2026001')
        assert fig5 is None

    def test_devolve_none_sem_nota_do_aluno(self):
        """Devolve None quando o aluno não tem nota na disciplina ou matrícula não existe."""
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)

        # 1. Aluno com nota NaN na disciplina
        df_n_nan = df_notas.copy()
        df_n_nan.loc[df_n_nan['matricula'] == '2026001', 'MAT'] = np.nan
        fig1 = grafico_dispersao_stem_aluno(df_n_nan, df_faltas, estat, 'MAT', 'Matemática', '2026001')
        assert fig1 is None

        # 2. Matrícula inexistente
        fig2 = grafico_dispersao_stem_aluno(df_notas, df_faltas, estat, 'MAT', 'Matemática', '9999999')
        assert fig2 is None

        # 3. Disciplina ausente em df_notas
        df_n_sem_mat = df_notas.drop(columns=['MAT'])
        fig3 = grafico_dispersao_stem_aluno(df_n_sem_mat, df_faltas, estat, 'MAT', 'Matemática', '2026001')
        assert fig3 is None

        # 4. df_notas vazio ou None
        assert grafico_dispersao_stem_aluno(None, df_faltas, estat, 'MAT', 'Matemática', '2026001') is None
        assert grafico_dispersao_stem_aluno(pd.DataFrame(), df_faltas, estat, 'MAT', 'Matemática', '2026001') is None

    def test_nenhum_nome_de_outro_aluno_e_nome_do_aluno_na_legenda(self):
        """Nenhum nome de outro aluno aparece na figura (fig.texts, eixos, legenda) e aluno está na legenda."""
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)

        # Aluna escolhida: Ana ('2026001')
        fig = grafico_dispersao_stem_aluno(df_notas, df_faltas, estat, 'MAT', 'Matemática', '2026001')
        assert fig is not None
        try:
            # Varrer todos os objetos Text da figura (D4 / LGPD)
            todos_textos_objs = fig.findobj(matplotlib.text.Text)
            todos_textos = [t.get_text() for t in todos_textos_objs]

            # Nenhum outro aluno pode aparecer em NENHUM texto da figura
            outros_alunos = ['Beto', 'Caio', 'Duda', 'Elis']
            for outro in outros_alunos:
                for t in todos_textos:
                    assert outro not in t, f"Nome do colega '{outro}' encontrado no texto: '{t}'"

            # Também verificar que fig.texts especificamente não contém nomes de outros
            for t in fig.texts:
                for outro in outros_alunos:
                    assert outro not in t.get_text()

            # O nome da aluna escolhida DEVE aparecer na legenda
            legenda = fig.axes[0].get_legend()
            assert legenda is not None
            textos_legenda = [t.get_text() for t in legenda.get_texts()]
            assert any('Ana' in t for t in textos_legenda), f"'Ana' não encontrada na legenda: {textos_legenda}"
        finally:
            plt.close(fig)

    def test_rotulo_eixo_y_conforme_acumulado(self):
        """Rótulo do eixo Y é 'Nota (0 a {max_pts})' no bimestre e 'Aproveitamento acumulado (%)' no acumulado."""
        # 1. Modo bimestre normal (1º bimestre: 20 pts)
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)
        fig1 = grafico_dispersao_stem_aluno(df_notas, df_faltas, estat, 'MAT', 'Matemática', '2026001')
        assert fig1 is not None
        try:
            assert fig1.axes[0].get_ylabel() == 'Nota (0 a 20)'
        finally:
            plt.close(fig1)

        # 2. Modo acumulado
        c_test = TestConsolidarBimestres()
        c1, c2, c3 = c_test._dados_tres_bimestres()
        df_notas_pct, df_faltas_acum, disc_dict_acum, estat_acum = consolidar_bimestres([c1, c2, c3])
        fig2 = grafico_dispersao_stem_aluno(df_notas_pct, df_faltas_acum, estat_acum, 'MAT', 'Matemática', '2026001')
        assert fig2 is not None
        try:
            assert fig2.axes[0].get_ylabel() == 'Aproveitamento acumulado (%)'
        finally:
            plt.close(fig2)

    def test_aluno_em_estrela_e_colegas_em_cinza(self):
        """Aluno é desenhado com marcador estrela e cor do quadrante; colegas em cinza sem anotações."""
        df_notas, df_faltas, disc_dict, meta = _turma_sintetica(1)
        estat = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)
        # Ana: nota 18.0 >= 12.0 e faltas 0 <= 2.0 -> quadrante 'Em dia' ('seagreen')
        fig = grafico_dispersao_stem_aluno(df_notas, df_faltas, estat, 'MAT', 'Matemática', '2026001')
        assert fig is not None
        try:
            ax = fig.axes[0]
            # O gráfico deve ter coleções de scatter (colegas e aluno)
            scatters = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
            assert len(scatters) >= 2  # colegas e aluno
            # Não pode haver nenhuma chamada a ax.annotate para outros alunos (apenas colegas anônimos)
            assert len(ax.texts) == 4  # apenas os 4 rótulos dos quadrantes nos cantos
        finally:
            plt.close(fig)




