"""Testes do gráfico de dispersão Notas × Faltas com quadrantes (Passo 6-bis).

`grafico_dispersao_notas_faltas` classifica cada aluno num de quatro
quadrantes a partir de dois cortes: o limiar de aprovação (nota, horizontal)
e a média de faltas da turma (faltas, vertical). Só o quadrante "Risco
duplo" (nota < limiar E faltas > média) recebe numeração no gráfico, com a
lista "número → nome" desenhada como texto da própria figura — é por essa
lista que os testes abaixo verificam a classificação de borda, já que o
quadrante de cada ponto não é devolvido como dado, só desenhado.
"""

import unittest

import pandas as pd

from core.relatorios import calcular_estatisticas, grafico_dispersao_notas_faltas

DISCIPLINAS = {'MAT': 'MATEMÁTICA - 2ª SÉRIE'}
METADADOS = {'bimestre_num': 1}  # 20 pts, limiar 12.0


def _texto_lista_risco_duplo(fig):
    """Concatena todo o texto desenhado diretamente na figura (fig.text) —
    é onde a lista "número → nome" do quadrante Risco duplo é escrita."""
    return ' '.join(t.get_text() for t in fig.texts)


class TestClassificacaoPorQuadrante(unittest.TestCase):
    """Os cortes que definem os quadrantes e os casos de borda.

    Turma de 5 alunos com faltas desenhadas para que a média da turma dê
    exatamente 6,0 faltas:
    - Ana (nota 18, falta 2) e Beto (nota 16, falta 2): "Em dia".
    - Zeca (nota EXATAMENTE no limiar = 12, falta 10 > média): se o corte
      de nota fosse ">" em vez de ">=", cairia (erradamente) em "Risco
      duplo"; com ">=", fica em "Risco de frequência" — não aparece na
      lista numerada.
    - Yara (nota 8 < limiar, falta EXATAMENTE na média = 6): se o corte de
      faltas contasse "= média" como acima, cairia (erradamente) em "Risco
      duplo"; com "> média" estrito, fica em "Risco de desempenho" — não
      aparece na lista numerada.
    - Wesley (nota 6 < limiar, falta 10 > média): controle positivo,
      genuinamente "Risco duplo" — tem que aparecer na lista numerada.
    """

    def _turma(self):
        df_notas = pd.DataFrame({
            'nome': ['Ana', 'Beto', 'Zeca', 'Yara', 'Wesley'],
            'MAT': [18.0, 16.0, 12.0, 8.0, 6.0],
        })
        df_faltas = pd.DataFrame({
            'nome': ['Ana', 'Beto', 'Zeca', 'Yara', 'Wesley'],
            'MAT': [2, 2, 10, 6, 10],
        })
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        self.assertAlmostEqual(stats['media_faltas_total'], 6.0)
        return df_notas, df_faltas, stats

    def test_aluno_exatamente_no_limiar_conta_como_maior_ou_igual(self):
        df_notas, df_faltas, stats = self._turma()
        limiar = stats['limiar_aprovacao']
        fig = grafico_dispersao_notas_faltas(df_notas, df_faltas, 'Trânsito', stats, limiar)
        self.assertIsNotNone(fig)
        texto = _texto_lista_risco_duplo(fig)
        self.assertNotIn('Zeca', texto)

    def test_aluno_exatamente_na_media_de_faltas_conta_como_menor_ou_igual(self):
        df_notas, df_faltas, stats = self._turma()
        limiar = stats['limiar_aprovacao']
        fig = grafico_dispersao_notas_faltas(df_notas, df_faltas, 'Trânsito', stats, limiar)
        self.assertIsNotNone(fig)
        texto = _texto_lista_risco_duplo(fig)
        self.assertNotIn('Yara', texto)

    def test_controle_positivo_risco_duplo_aparece_na_lista(self):
        df_notas, df_faltas, stats = self._turma()
        limiar = stats['limiar_aprovacao']
        fig = grafico_dispersao_notas_faltas(df_notas, df_faltas, 'Trânsito', stats, limiar)
        texto = _texto_lista_risco_duplo(fig)
        self.assertIn('Wesley', texto)

    def test_todos_os_alunos_com_faltas_lancadas_viram_pontos(self):
        df_notas, df_faltas, stats = self._turma()
        limiar = stats['limiar_aprovacao']
        fig = grafico_dispersao_notas_faltas(df_notas, df_faltas, 'Trânsito', stats, limiar)
        ax = fig.axes[0]
        n_pontos = len(ax.collections[0].get_offsets())
        self.assertEqual(n_pontos, 5)


class TestSemFaltasDevolveNone(unittest.TestCase):
    """Mesmo contrato dos demais gráficos de faltas: sem faltas disponíveis
    na turma, devolve None (e a chave simplesmente não aparece no PDF)."""

    def test_sem_df_faltas(self):
        df_notas = pd.DataFrame({'nome': ['Ana', 'Beto'], 'MAT': [18.0, 10.0]})
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        fig = grafico_dispersao_notas_faltas(
            df_notas, None, 'Trânsito', stats, stats['limiar_aprovacao'])
        self.assertIsNone(fig)

    def test_df_faltas_vazio(self):
        df_notas = pd.DataFrame({'nome': ['Ana', 'Beto'], 'MAT': [18.0, 10.0]})
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        fig = grafico_dispersao_notas_faltas(
            df_notas, pd.DataFrame(), 'Trânsito', stats, stats['limiar_aprovacao'])
        self.assertIsNone(fig)


if __name__ == "__main__":
    unittest.main()
