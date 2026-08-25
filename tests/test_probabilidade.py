"""Testes da probabilidade de reprovação anual (Passo 8, `core.relatorios`).

`calcular_probabilidade_reprovacao(student_data)` — só notas (a exigência de
75% de frequência e a recuperação/exame final ficam de fora, ver Passo 14).

`student_data` é o dicionário interno de `calcular_estatisticas_multibimestre`:
``{matricula: {'nome': str, 'medias': {bimestre: media}, 'faltas': {...}}}``.
"""

import unittest

import pandas as pd

from core.relatorios import MAX_PONTOS_BIMESTRE, calcular_probabilidade_reprovacao


class TestRamosDeterministicos(unittest.TestCase):
    """Os três ramos deterministicos: N<=0, N>R e o resto ("Depende")."""

    def test_n_menor_ou_igual_a_zero_e_aprovado_por_nota_com_0_por_cento(self):
        """3 bimestres cheios (20+30+20=70 >= 60) → aprovado, 0%, sem NaN."""
        student_data = {
            'a': {'nome': 'Ana', 'medias': {1: 20.0, 2: 30.0, 3: 20.0}, 'faltas': {}},
        }
        df = calcular_probabilidade_reprovacao(student_data)
        linha = df.iloc[0]
        self.assertEqual(linha['Situacao'], 'Aprovado por nota')
        self.assertEqual(linha['Prob_Reprovacao'], 0.0)
        self.assertLessEqual(linha['N'], 0)

    def test_n_maior_que_r_nao_alcanca_60_pontos_com_probabilidade_nan(self):
        """3 bimestres com notas muito baixas: R sobra pouco (30) e N (53)
        ultrapassa. O rótulo NÃO é mais "reprovação matemática" (Passo 14) e
        a probabilidade fica NaN — o EPTNM tem recuperação/exame final, via
        de aprovação que este modelo (só notas regulares) não enxerga."""
        student_data = {
            'b': {'nome': 'Beto', 'medias': {1: 2.0, 2: 3.0, 3: 2.0}, 'faltas': {}},
        }
        df = calcular_probabilidade_reprovacao(student_data)
        linha = df.iloc[0]
        self.assertEqual(linha['Situacao'], 'Não alcança 60 pontos nas notas regulares')
        self.assertNotEqual(linha['Situacao'], 'Reprovação matemática por nota')
        self.assertTrue(pd.isna(linha['Prob_Reprovacao']))
        self.assertGreater(linha['N'], linha['R'])

    def test_quatro_bimestres_lancados_nunca_cai_em_depende(self):
        """Com os 4 bimestres, R = 0 para todo mundo — só existe o
        determinístico: 'Aprovado por nota' ou 'Não alcança...', nunca
        'Depende'."""
        student_data = {
            'alta': {
                'nome': 'Alta', 'faltas': {},
                'medias': {1: 20.0, 2: 30.0, 3: 20.0, 4: 30.0},  # S=100
            },
            'baixa': {
                'nome': 'Baixa', 'faltas': {},
                'medias': {1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0},  # S=20
            },
            'limite': {
                'nome': 'Limite', 'faltas': {},
                'medias': {1: 12.0, 2: 18.0, 3: 12.0, 4: 18.0},  # S=60
            },
        }
        df = calcular_probabilidade_reprovacao(student_data)
        self.assertNotIn('Depende', df['Situacao'].tolist())
        self.assertTrue((df['R'] == 0).all())

    def test_aluno_com_lancamento_pendente_nao_cai_em_nao_alcanca(self):
        """A turma enviou os bimestres 1, 2 e 3, mas um aluno só tem nota no
        1º (lançamento pendente dos outros dois). R é calculado POR ALUNO
        (correção de 2026-08-24) — esse aluno não pode ser punido com "não
        alcança" só por faltar lançamento de colegas de turma."""
        student_data = {
            'pendente': {
                'nome': 'Pendente', 'faltas': {},
                'medias': {1: 15.0},  # só o 1º bimestre, nota razoável (75%)
            },
            # Os demais confirmam que a turma de fato já enviou 1, 2 e 3.
            'completo1': {
                'nome': 'Completo Um', 'faltas': {},
                'medias': {1: 16.0, 2: 24.0, 3: 16.0},
            },
            'completo2': {
                'nome': 'Completo Dois', 'faltas': {},
                'medias': {1: 14.0, 2: 22.0, 3: 14.0},
            },
        }
        df = calcular_probabilidade_reprovacao(student_data)
        linha = df[df['Aluno'] == 'Pendente'].iloc[0]
        self.assertEqual(linha['Situacao'], 'Depende')
        self.assertEqual(linha['Pontos_Lancados'], MAX_PONTOS_BIMESTRE[1])
        self.assertEqual(linha['R'], 100 - MAX_PONTOS_BIMESTRE[1])


class TestInvariante(unittest.TestCase):
    """Pontos_Lancados + R == 100 sempre, por aluno — não mais um R global."""

    def test_soma_pontos_lancados_e_r_fecha_100_em_varios_cenarios(self):
        student_data = {
            'um_bim': {'nome': 'A', 'medias': {1: 10.0}, 'faltas': {}},
            'dois_bim': {'nome': 'B', 'medias': {1: 10.0, 2: 20.0}, 'faltas': {}},
            'tres_bim': {'nome': 'C', 'medias': {1: 10.0, 2: 20.0, 3: 10.0}, 'faltas': {}},
            'quatro_bim': {
                'nome': 'D', 'medias': {1: 10.0, 2: 20.0, 3: 10.0, 4: 20.0}, 'faltas': {}},
            'sem_nota': {'nome': 'E', 'medias': {}, 'faltas': {}},
        }
        df = calcular_probabilidade_reprovacao(student_data)
        for _, linha in df.iterrows():
            self.assertEqual(linha['Pontos_Lancados'] + linha['R'], 100)


class TestRamoDependeMonotoniaELimites(unittest.TestCase):
    """A parte probabilística: monotonia (S maior => probabilidade menor) e
    o corte em [1%, 99%] (inclusive o piso de sigma quando o pool de
    encolhimento tem variância zero)."""

    def _turma_pool_variancia_zero(self, s_baixo, s_alto):
        """Dois alunos "pool" com variância amostral zero (mesmo
        aproveitamento nos dois bimestres) fixam sigma_pool² = 0 — e como
        os alunos de teste têm só 1 bimestre (sem s_i² próprio), o
        encolhimento cai inteiro no piso de sigma (0,05). Isso torna o
        resultado do "Depende" determinístico o bastante para comparar com
        limiar exato de clipping."""
        return {
            'pool1': {
                'nome': 'Pool Um', 'faltas': {},
                'medias': {1: 10.0, 2: 15.0},  # 50% em ambos os bimestres
            },
            'pool2': {
                'nome': 'Pool Dois', 'faltas': {},
                'medias': {1: 10.0, 2: 15.0},  # idêntico ao pool1 → variância 0
            },
            'baixo': {'nome': 'Baixo', 'faltas': {}, 'medias': {1: s_baixo}},
            'alto': {'nome': 'Alto', 'faltas': {}, 'medias': {1: s_alto}},
        }

    def test_monotonia_s_maior_gera_probabilidade_menor(self):
        student_data = self._turma_pool_variancia_zero(s_baixo=8.0, s_alto=16.0)
        df = calcular_probabilidade_reprovacao(student_data)
        prob_baixo = df[df['Aluno'] == 'Baixo'].iloc[0]['Prob_Reprovacao']
        prob_alto = df[df['Aluno'] == 'Alto'].iloc[0]['Prob_Reprovacao']
        self.assertEqual(df[df['Aluno'] == 'Baixo'].iloc[0]['Situacao'], 'Depende')
        self.assertEqual(df[df['Aluno'] == 'Alto'].iloc[0]['Situacao'], 'Depende')
        self.assertGreater(prob_baixo, prob_alto)

    def test_limites_em_1_e_99_por_cento_com_piso_de_sigma(self):
        """Com sigma no piso (0,05) e S bem distante do que falta, o z
        estoura e a probabilidade satura exatamente no clip: 99% para quem
        tem pouquíssimo aproveitamento, 1% para quem tem muito."""
        student_data = self._turma_pool_variancia_zero(s_baixo=8.0, s_alto=16.0)
        df = calcular_probabilidade_reprovacao(student_data)
        prob_baixo = df[df['Aluno'] == 'Baixo'].iloc[0]['Prob_Reprovacao']
        prob_alto = df[df['Aluno'] == 'Alto'].iloc[0]['Prob_Reprovacao']
        self.assertAlmostEqual(prob_baixo, 0.99, places=6)
        self.assertAlmostEqual(prob_alto, 0.01, places=6)

    def test_probabilidade_sempre_entre_1_e_99_por_cento(self):
        """Em qualquer cenário "Depende", a probabilidade nunca sai de [1%,99%]."""
        student_data = {
            'p1': {'nome': 'P1', 'faltas': {}, 'medias': {1: 12.0, 2: 18.0}},
            'p2': {'nome': 'P2', 'faltas': {}, 'medias': {1: 8.0, 2: 22.0}},
            'x1': {'nome': 'X1', 'faltas': {}, 'medias': {1: 19.9}},
            'x2': {'nome': 'X2', 'faltas': {}, 'medias': {1: 0.1}},
        }
        df = calcular_probabilidade_reprovacao(student_data)
        depende = df[df['Situacao'] == 'Depende']
        self.assertFalse(depende.empty)
        self.assertTrue((depende['Prob_Reprovacao'] >= 0.01).all())
        self.assertTrue((depende['Prob_Reprovacao'] <= 0.99).all())


class TestOrdenacaoPorGravidade(unittest.TestCase):
    """Passo 14: ordem por gravidade — "Não alcança..." (por N decrescente),
    depois "Depende" (por probabilidade decrescente, desempate por N
    decrescente — item 0 desta etapa), por último "Aprovado por nota"."""

    def test_ordem_da_tabela(self):
        student_data = {
            # "Não alcança..." — N=60 > N=53, então nao_alto vem antes de nao_baixo.
            'nao_alto': {'nome': 'Nao Alto', 'faltas': {},
                         'medias': {1: 0.0, 2: 0.0, 3: 0.0}},  # S=0, N=60
            'nao_baixo': {'nome': 'Nao Baixo', 'faltas': {},
                          'medias': {1: 2.0, 2: 3.0, 3: 2.0}},  # S=7, N=53
            # Pool de variância zero, reaproveitado das probabilidades exatas
            # (0,99 e 0,01) do teste de limites acima.
            'pool1': {'nome': 'Pool Um', 'faltas': {}, 'medias': {1: 10.0, 2: 15.0}},
            'pool2': {'nome': 'Pool Dois', 'faltas': {}, 'medias': {1: 10.0, 2: 15.0}},
            # Dois alunos empatam em 99% (item 0): dep_n_maior tem N maior
            # (precisa de mais pontos) e deve vir ANTES de dep_n_menor.
            'dep_n_menor': {'nome': 'Depende N Menor', 'faltas': {}, 'medias': {1: 8.0}},   # N=52
            'dep_n_maior': {'nome': 'Depende N Maior', 'faltas': {}, 'medias': {1: 5.0}},   # N=55
            'dep_baixa_prob': {'nome': 'Depende Prob Baixa', 'faltas': {}, 'medias': {1: 16.0}},  # ~1%
            # "Aprovado por nota" — sempre por último.
            'aprovado': {'nome': 'Aprovado', 'faltas': {},
                         'medias': {1: 20.0, 2: 30.0, 3: 20.0}},  # S=70, N=-10
        }
        df = calcular_probabilidade_reprovacao(student_data)
        # O pool (Pool Um/Dois) só existe para fixar sigma_pool² = 0 (ver
        # teste de limites acima) — também aparece na tabela como "Depende"
        # com seu próprio N/probabilidade, mas não é o que este teste audita.
        ordem = [a for a in df['Aluno'].tolist() if not a.startswith('Pool')]

        self.assertEqual(
            ordem,
            [
                'Nao Alto', 'Nao Baixo',
                'Depende N Maior', 'Depende N Menor', 'Depende Prob Baixa',
                'Aprovado',
            ],
        )
        # Confere que os dois empatados em 99% de fato empatam (prova que o
        # desempate por N decrescente é o que decidiu a ordem entre eles).
        prob_n_maior = df[df['Aluno'] == 'Depende N Maior'].iloc[0]['Prob_Reprovacao']
        prob_n_menor = df[df['Aluno'] == 'Depende N Menor'].iloc[0]['Prob_Reprovacao']
        self.assertAlmostEqual(prob_n_maior, prob_n_menor, places=6)


if __name__ == "__main__":
    unittest.main()
