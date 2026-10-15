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

import numpy as np
import pandas as pd
import pytest

from core.relatorio_aluno import (
    comparar_aluno_turma,
    criar_relatorio_aluno_pdf,
    evolucao_aluno_turma,
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

