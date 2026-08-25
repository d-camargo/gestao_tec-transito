"""Testes do IDA — Índice de Desempenho do Aluno (escala -1 a +1).

Casos numéricos fechados, definidos nas Decisões de arquitetura do plano:
zero exatamente no limiar de aprovação parcial; combinação convexa em
[-1, +1] por construção; e a armadilha do sinal — média abaixo do limiar
nunca gera componente de notas (nem IDA de corte) positivo.

D4 (2026-08-24): o IDA passou a ser **exclusivo da análise multibimestral**
— `calcular_estatisticas` (bimestre único) não chama mais `calcular_ida` e
não publica `ida_df`/`ida_medio_turma`. `calcular_ida` continua existindo e
testada (é a base da fórmula), mas quem o consome de fato é
`calcular_ida_multibimestre`, aplicando a mesma forma ao acumulado dos
bimestres enviados.
"""

import unittest

import pandas as pd

from core.relatorios import (
    calcular_estatisticas,
    calcular_ida,
    calcular_ida_multibimestre,
    grafico_ida_alunos,
)

LIMIAR = 12.0
MAX_PTS = 20


def _ida_direto(df_notas, df_faltas):
    """Chama calcular_ida com o contrato interno do módulo (limiar 12, 20 pts)."""
    estatisticas = {
        'disciplinas_com_notas': ['MAT'],
        'faltas_disponiveis': df_faltas is not None,
        '_faltas_cols': ['MAT'],
    }
    return calcular_ida(df_notas, df_faltas, estatisticas, LIMIAR, MAX_PTS)


class TestCalcularIda(unittest.TestCase):
    """Casos numéricos fechados da fórmula do IDA."""

    def test_nota_maxima_e_zero_falta_da_mais_um(self):
        """Aluno com nota máxima e zero falta → IDA = +1,0."""
        df_notas = pd.DataFrame({
            'nome': ['Ana', 'Beto', 'Caio'],
            'MAT': [20.0, 15.0, 14.0],
        })
        df_faltas = pd.DataFrame({
            'nome': ['Ana', 'Beto', 'Caio'],
            'MAT': [0, 5, 10],
        })
        ida = _ida_direto(df_notas, df_faltas)
        ana = ida[ida['nome'] == 'Ana'].iloc[0]
        self.assertAlmostEqual(ana['ida'], 1.0, places=6)

    def test_nota_zero_e_faltas_dobro_do_p90_da_menos_um(self):
        """Aluno com nota 0 e faltas ≥ 2×P90 → IDA = -1,0."""
        # 9 colegas sem falta mantêm o P90 baixo (4), então 40 faltas = 10×P90.
        nomes = ['Ana'] + [f'Aluno {i}' for i in range(2, 11)]
        df_notas = pd.DataFrame({
            'nome': nomes,
            'MAT': [0.0] + [10.0] * 9,
        })
        df_faltas = pd.DataFrame({
            'nome': nomes,
            'MAT': [40] + [0] * 9,
        })
        ida = _ida_direto(df_notas, df_faltas)
        ana = ida[ida['nome'] == 'Ana'].iloc[0]
        self.assertAlmostEqual(ana['ida'], -1.0, places=6)

    def test_exatamente_no_limiar_e_faltas_igual_p90_da_zero(self):
        """Aluno exatamente no limiar e com faltas iguais ao P90 → IDA = 0,0."""
        df_notas = pd.DataFrame({
            'nome': ['Ana', 'Beto', 'Caio'],
            'MAT': [12.0, 18.0, 16.0],
        })
        df_faltas = pd.DataFrame({
            'nome': ['Ana', 'Beto', 'Caio'],
            'MAT': [6, 6, 6],
        })
        ida = _ida_direto(df_notas, df_faltas)
        ana = ida[ida['nome'] == 'Ana'].iloc[0]
        self.assertAlmostEqual(ana['ida'], 0.0, places=6)

    def test_sem_faltas_ida_igual_a_c_nota(self):
        """Turma sem faltas lançadas → c_falta é NaN e o IDA vale c_nota."""
        df_notas = pd.DataFrame({
            'nome': ['Ana', 'Beto'],
            'MAT': [18.0, 10.0],
        })
        ida = _ida_direto(df_notas, None)
        self.assertTrue(ida['c_falta'].isna().all())
        self.assertTrue((ida['ida'] == ida['c_nota']).all())

    def test_armadilha_do_sinal_media_abaixo_do_limiar_e_negativo(self):
        """Média um pouco abaixo do limiar → c_nota e IDA negativos.

        Com faltas iguais ao P90 (c_falta = 0), o sinal final vem só da
        componente de notas — que nunca mente com a quebra no limiar.
        """
        df_notas = pd.DataFrame({
            'nome': ['Ana', 'Beto', 'Caio'],
            'MAT': [11.0, 18.0, 16.0],
        })
        df_faltas = pd.DataFrame({
            'nome': ['Ana', 'Beto', 'Caio'],
            'MAT': [6, 6, 6],
        })
        ida = _ida_direto(df_notas, df_faltas)
        ana = ida[ida['nome'] == 'Ana'].iloc[0]
        self.assertLess(ana['c_nota'], 0.0)
        self.assertLess(ana['ida'], 0.0)

    def test_ida_sempre_dentro_da_faixa_e_ordenado(self):
        """A combinação convexa fica em [-1, +1]; o DataFrame vem ordenado."""
        nomes = ['Ana', 'Beto', 'Caio', 'Duda', 'Elis', 'Felipe']
        df_notas = pd.DataFrame({
            'nome': nomes,
            'MAT': [20.0, 18.0, 16.0, 11.0, 10.0, 9.0],
        })
        ida = _ida_direto(df_notas, None)
        self.assertTrue((ida['ida'] >= -1.0).all() and (ida['ida'] <= 1.0).all())
        self.assertEqual(ida['ida'].tolist(), sorted(ida['ida'].tolist()))
        self.assertIn('faixa', ida.columns)

    def test_calcular_estatisticas_bimestre_unico_nao_publica_ida(self):
        """D4: bimestre único não calcula mais IDA — nem ida_df, nem ida_medio_turma."""
        df_notas = pd.DataFrame({
            'nome': ['Ana', 'Beto'],
            'MAT': [18.0, 10.0],
        })
        stats = calcular_estatisticas(
            df_notas, {'MAT': 'MATEMÁTICA - 2ª SÉRIE'}, metadados={'bimestre_num': 1})
        self.assertNotIn('ida_df', stats)
        self.assertNotIn('ida_medio_turma', stats)


class TestCalcularIdaMultibimestre(unittest.TestCase):
    """Passo 7: IDA aplicado ao acumulado dos bimestres enviados (S, P e L
    acumulados) — mesma forma da fórmula do bimestre único."""

    def _ida_de(self, medias, faltas=None):
        """Calcula o IDA de um único aluno 'Ana' com as `medias` dadas
        (dict bimestre->nota). Usa MAX_PONTOS_BIMESTRE = {1: 20, ...} do
        próprio módulo."""
        student_data = {1: {'nome': 'Ana', 'medias': medias, 'faltas': faltas or {}}}
        ida_df = calcular_ida_multibimestre(student_data)
        return ida_df.iloc[0]

    def test_s_igual_p_da_mais_um(self):
        """S = P (nota máxima em todos os bimestres enviados) → IDA = +1."""
        ana = self._ida_de({1: 20.0})
        self.assertAlmostEqual(ana['ida'], 1.0, places=6)

    def test_s_igual_zero_da_menos_um(self):
        """S = 0 → IDA = -1."""
        ana = self._ida_de({1: 0.0})
        self.assertAlmostEqual(ana['ida'], -1.0, places=6)

    def test_s_igual_l_da_zero(self):
        """S = L (exatamente no limiar acumulado, 60% de P) → IDA = 0."""
        ana = self._ida_de({1: 12.0})  # L = 0,60 * 20 = 12
        self.assertAlmostEqual(ana['ida'], 0.0, places=6)

    def test_s_igual_l_acumulado_em_dois_bimestres(self):
        """L acumulado em 2 bimestres (20+30=50 pts; L=30) → S=L também dá zero."""
        ana = self._ida_de({1: 15.0, 2: 15.0})  # S = 30 = 0,60 * 50
        self.assertAlmostEqual(ana['ida'], 0.0, places=6)

    def test_aluno_sem_medias_da_nan(self):
        """Aluno sem nenhuma média lançada em nenhum bimestre enviado → IDA = NaN."""
        ana = self._ida_de({})
        self.assertTrue(pd.isna(ana['ida']))


class TestGraficoIdaAlunos(unittest.TestCase):
    """Testes do gráfico de barras do IDA (só alunos abaixo da média)."""

    def _turma(self):
        """Turma com IDA multibimestral calculado a partir de um único
        bimestre enviado — mesma forma numérica do antigo teste de
        bimestre único, agora via calcular_ida_multibimestre (D4)."""
        valores = {
            'Ana': 20.0, 'Beto': 18.0, 'Caio': 16.0,
            'Duda': 11.0, 'Elis': 10.0, 'Felipe': 9.0,
        }
        student_data = {
            i: {'nome': nome, 'medias': {1: valor}, 'faltas': {}}
            for i, (nome, valor) in enumerate(valores.items())
        }
        ida_df = calcular_ida_multibimestre(student_data)
        ida_medio_turma = ida_df['ida'].mean()
        return ida_df, ida_medio_turma

    def test_figura_gerada_com_eixo_e_corte_corretos(self):
        """Figura gerada, eixo x de -1 a +1 e barras só dos abaixo da média."""
        ida_df, ida_medio_turma = self._turma()
        fig = grafico_ida_alunos(ida_df, ida_medio_turma, 'Trânsito')
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlim(), (-1.0, 1.0))
        n_abaixo = int((ida_df['ida'] < ida_medio_turma).sum())
        self.assertEqual(len(ax.patches), n_abaixo)
        self.assertLess(n_abaixo, len(ida_df))  # não é a turma inteira

    def test_devolve_none_sem_dados_ou_sem_aluno_abaixo(self):
        """None com DataFrame vazio ou quando ninguém fica abaixo da média."""
        self.assertIsNone(grafico_ida_alunos(pd.DataFrame(), 0.0, 'Trânsito'))
        ida_df = pd.DataFrame({'nome': ['Ana', 'Beto'], 'ida': [0.5, 0.5]})
        self.assertIsNone(grafico_ida_alunos(ida_df, 0.5, 'Trânsito'))


if __name__ == "__main__":
    unittest.main()
