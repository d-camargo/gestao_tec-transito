"""Carga, validação e exploração da planilha de Carga Horária (CH) Efetiva.

Schema da Planilha (C15):
    A planilha de carga horária efetiva das disciplinas do Ensino Técnico Integrado
    (2026) consolida a matriz curricular lecionada em sala de aula cruzada com o
    calendário escolar oficial do CEFET-MG. O documento possui quatro abas:

    1. 'Leia-me':
       Metadados da planilha, fontes de dados (horários de turmas e deliberações do
       CEPE), metodologia de apuração da CH e explicitação das limitações operacionais.

    2. 'Calendário':
       Distribuição dos dias letivos úteis (segunda a sexta-feira) por bimestre letivo
       e acumulado anual (1º BI: 50 dias; 2º BI: 48 dias; 3º BI: 44 dias; 4º BI: 41 dias;
       Ano: 183 dias úteis), além do registro de sábados letivos (17 dias no total de 200).

    3. 'CH por disciplina':
       Tabela detalhada de 20 colunas com cada oferta de disciplina por turma e subgrupo.
       - Colunas de entrada (dados brutos):
         * Curso: Nome por extenso do curso técnico.
         * Turma: Código identificador da turma no formato estrito <CURSO>-<SERIE><LETRA>
           (ex.: EST-2A, EST/TT-2A, EDI-1B).
         * Subgrupo: Divisão de turma prática ('T1' ou 'T2') ou '—' (travessão) para
           turma inteira / sem subgrupos (normalizado para "" no DataFrame).
         * Sigla: Sigla de identificação da disciplina na grade ou horários.
         * Disciplina: Nome descritivo da disciplina no PPC / horários.
         * Professor(a): Nome do(a) docente ministrante — DESCARTADA NA CARGA por
           diretriz de minimização de dados e LGPD, nunca sendo carregada no DataFrame.
         * SEG, TER, QUA, QUI, SEX: Quantidade inteira de aulas semanais alocadas
           em cada dia útil (valores inteiros >= 0).
       - Colunas de fórmula (calculadas ou derivadas de Calendário):
         * Aulas/sem: Total semanal de aulas (=SUM(SEG:SEX)).
         * CH nominal: Carga horária nominal de referência anual (=Aulas/sem * 40 semanas).
         * 1º BI: Horas-aula lecionadas no 1º bimestre (=SUMPRODUCT(SEG:SEX, Calendário!1ºBI)).
         * 2º BI: Horas-aula lecionadas no 2º bimestre (=SUMPRODUCT(SEG:SEX, Calendário!2ºBI)).
         * 3º BI: Horas-aula lecionadas no 3º bimestre (=SUMPRODUCT(SEG:SEX, Calendário!3ºBI)).
         * 4º BI: Horas-aula lecionadas no 4º bimestre (=SUMPRODUCT(SEG:SEX, Calendário!4ºBI)).
         * CH efetiva: Horas-aula lecionadas no ano (=SUM(1ºBI:4ºBI)).
         * Diferença: Diferença em horas-aula em relação ao nominal (=CH efetiva - CH nominal).
         * % do nominal: Razão da carga lecionada sobre o nominal (=CH efetiva / CH nominal).

    4. 'Resumo por carga':
       Tabela de referência da CH efetiva por aula semanal em cada dia e agregação da
       distribuição das disciplinas segundo o percentual do nominal atingido:
       - Abaixo de 90%: < 90% da carga horária nominal.
       - 90% a 95%: entre 90% (inclusive) e 95% (exclusive).
       - 95% ou mais: >= 95% da carga horária nominal.

Regras e Premissas de Domínio:
    - Regra "sábados não entram" = Cenário A:
      Os 17 sábados letivos previstos no calendário anual são alocados tematicamente por
      curso ou área acadêmica, sem reproduzir um dia útil regular da grade semanal.
      Portanto, a apuração da CH efetiva por disciplina na planilha reflete o Cenário A
      (apenas aulas ministradas de segunda a sexta-feira).
    - Faixas de CH efetiva (Cenário A):
      * 1 aula/sem:  35 a 38 h/a  (nominal 40 h/a)
      * 2 aulas/sem: 70 a 76 h/a  (nominal 80 h/a)
      * 3 aulas/sem: 105 a 114 h/a (nominal 120 h/a)
      * 4 aulas/sem: 140 a 152 h/a (nominal 160 h/a)
    - Limitação de salas 305–437:
      O mapeamento de horários original cobre apenas as 36 salas listadas (305 a 437).
      Disciplinas com aulas em laboratórios especializados, oficinas, quadras esportivas
      ou outras dependências podem não constar no mapa de horários ou apresentar menos
      aulas semanais registradas do que sua carga curricular oficial do PPC.
    - Minimização de dados / LGPD:
      A coluna 'Professor(a)' é descartada no ato da carga. O DataFrame resultante contém
      apenas atributos institucionais e operacionais das turmas e disciplinas.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import unicodedata

import openpyxl
import pandas as pd

try:
    from .calendario import (
        Calendario,
        carregar_calendario,
        ch_lecionada,
        faixa_ch,
        sabados_do_responsavel,
    )
except ImportError:
    from calendario import (
        Calendario,
        carregar_calendario,
        ch_lecionada,
        faixa_ch,
        sabados_do_responsavel,
    )

# Padrão de nome de arquivo para descoberta e isolamento (C10, C11)
PADRAO_CH_EFETIVA: re.Pattern[str] = re.compile(r"^CH_Efetiva.*\.xlsx$", re.IGNORECASE)

# Nomes de abas oficiais da planilha
ABA_LEIA_ME: str = "Leia-me"
ABA_CALENDARIO: str = "Calendário"
ABA_CH_DISCIPLINA: str = "CH por disciplina"
ABA_RESUMO_CARGA: str = "Resumo por carga"
ABAS_OBRIGATORIAS: tuple[str, ...] = (
    ABA_LEIA_ME,
    ABA_CALENDARIO,
    ABA_CH_DISCIPLINA,
    ABA_RESUMO_CARGA,
)

# Cabeçalho original de 20 colunas na aba 'CH por disciplina'
CABECALHO_CH_DISCIPLINA: tuple[str, ...] = (
    "Curso",
    "Turma",
    "Subgrupo",
    "Sigla",
    "Disciplina",
    "Professor(a)",
    "SEG",
    "TER",
    "QUA",
    "QUI",
    "SEX",
    "Aulas/sem",
    "CH nominal",
    "1º BI",
    "2º BI",
    "3º BI",
    "4º BI",
    "CH efetiva",
    "Diferença",
    "% do nominal",
)

# Colunas normalizadas devolvidas no DataFrame (C15: sem Professor(a) — descartada
# na carga; sem Diferença e % do nominal — derivadas, recalculáveis a partir das
# colunas mantidas; com disciplina_norm e serie/letra extraídas de turma)
COLUNAS_CH_EFETIVA: tuple[str, ...] = (
    "curso",
    "turma",
    "serie",
    "letra",
    "subgrupo",
    "sigla",
    "disciplina",
    "disciplina_norm",
    "seg",
    "ter",
    "qua",
    "qui",
    "sex",
    "aulas_sem",
    "ch_nominal",
    "ch_bim_1",
    "ch_bim_2",
    "ch_bim_3",
    "ch_bim_4",
    "ch_efetiva",
)

# Expressão regular para validação do padrão de turma (<CURSO>-<SERIE><LETRA>)
PADRAO_TURMA: re.Pattern[str] = re.compile(r"^([A-Z/]+)-(\d)([A-Z])$")

# Dias úteis da semana considerados na grade
DIAS_UTEIS: tuple[str, ...] = ("SEG", "TER", "QUA", "QUI", "SEX")

# Caminho padrão para o arquivo oficial versionado no repositório
PASTA_DADOS: Path = Path(__file__).resolve().parent / "dados"
CAMINHO_CH_EFETIVA_PADRAO: Path = PASTA_DADOS / "CH_Efetiva_Disciplinas_Integrado_2026.xlsx"
CAMINHO_ESTRADAS_PADRAO: Path = PASTA_DADOS / "Estradas_2025-2026.xls"
CAMINHO_TRANSITO_PADRAO: Path = PASTA_DADOS / "Transito_2025-2026.xls"


def remover_acentos(txt: str) -> str:
    """Remove caracteres diacríticos (acentos, cedilha) de uma string."""
    if not isinstance(txt, str):
        txt = str(txt) if txt is not None else ""
    return "".join(
        c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn"
    )


def normalizar_disciplina(nome: str) -> str:
    """Normaliza o nome de uma disciplina para casamento determinístico.

    Regras:
        1. Remove acentuação gráfica e caracteres diacríticos.
        2. Converte para maiúsculas e compacta múltiplos espaços em branco.
        3. Remove prefixos de língua estrangeira (ex.: 'LÍNGUA ESTRANGEIRA: INGLÊS' -> 'INGLÊS').
        4. Remove sufixos de série curricular (ex.: 'FÍSICA - 2ª SÉRIE' -> 'FÍSICA').
        5. Preserva distinções de laboratório (ex.: 'LABORATÓRIO DE SOLOS' ≠ 'SOLOS').

    Exemplos:
        >>> normalizar_disciplina("LÍNGUA ESTRANGEIRA: INGLÊS - 2ª SÉRIE")
        'INGLES'
        >>> normalizar_disciplina("FÍSICA - 2ª SÉRIE")
        'FISICA'
        >>> normalizar_disciplina("Máquinas  e Equipamentos")
        'MAQUINAS E EQUIPAMENTOS'
        >>> normalizar_disciplina("LABORATÓRIO DE SOLOS") != "SOLOS"
        True
    """
    if not isinstance(nome, str):
        nome = str(nome) if nome is not None else ""
    s = remover_acentos(nome).upper().strip()
    s = re.sub(r"\s+", " ", s)
    # Remove prefixo 'LINGUA ESTRANGEIRA:' ou 'LINGUA ESTRANGEIRA MODERNA:'
    s = re.sub(r"^LINGUA\s+ESTRANGEIRA(?:\s+MODERNA)?\s*:\s*", "", s)
    # Remove sufixo '- Xª SERIE' ou '- Xª ANO'
    s = re.sub(r"\s*-\s*\d+[ªºaAoO]?\s+SERIE.*$", "", s)
    s = re.sub(r"\s*-\s*\d+[ªºaAoO]?\s+ANO.*$", "", s)
    return s.strip()


def _extrair_dias_calendario(wb: openpyxl.Workbook) -> dict[int, dict[str, int]] | None:
    """Extrai os dias úteis por dia da semana de cada bimestre da aba 'Calendário'.

    Retorna None se a aba estiver ausente ou não permitir ler a matriz completa
    dos 4 bimestres (a recuperação por constantes embutidas seria uma segunda
    fonte do dado oficial, proibida por C1).
    """
    if ABA_CALENDARIO not in wb.sheetnames:
        return None

    ws = wb[ABA_CALENDARIO]

    # Localiza o cabeçalho dos bimestres (normalmente linha 3)
    linha_cabecalho = None
    for r in range(1, min(10, ws.max_row + 1)):
        val = str(ws.cell(row=r, column=1).value or "").strip().lower()
        if "bimestre" in val:
            linha_cabecalho = r
            break

    if linha_cabecalho is None:
        return None

    # Identifica colunas dos dias úteis
    col_dias: dict[str, int] = {}
    for c in range(2, min(12, ws.max_column + 1)):
        v = str(ws.cell(row=linha_cabecalho, column=c).value or "").strip().upper()
        if v in DIAS_UTEIS:
            col_dias[v] = c

    if len(col_dias) != 5:
        return None

    # Lê os 4 bimestres
    dias_bi: dict[int, dict[str, int]] = {}
    for r in range(linha_cabecalho + 1, min(linha_cabecalho + 6, ws.max_row + 1)):
        rotulo = str(ws.cell(row=r, column=1).value or "").strip().upper()
        num_bi = None
        for b in (1, 2, 3, 4):
            if f"{b}º" in rotulo or f"{b}O" in rotulo or f"{b} BI" in rotulo:
                num_bi = b
                break
        if num_bi is not None:
            dias_bi[num_bi] = {}
            for d in DIAS_UTEIS:
                c_val = ws.cell(row=r, column=col_dias[d]).value
                if isinstance(c_val, (int, float)):
                    dias_bi[num_bi][d] = int(c_val)
                else:
                    return None

    if len(dias_bi) == 4:
        return dias_bi
    return None


def carregar_ch_efetiva(
    caminho: str | Path | None = None,
    forcar_recalculo: bool = False,
) -> pd.DataFrame:
    """Carrega, valida e normaliza a planilha de Carga Horária Efetiva (C15).

    Args:
        caminho: Caminho para o arquivo .xlsx da planilha. Se None, utiliza o caminho
                 padrão 'sandbox/dae/dados/CH_Efetiva_Disciplinas_Integrado_2026.xlsx'.
        forcar_recalculo: Se True, recalcula todas as cargas horárias pelas aulas
                          semanais e calendário, independentemente de cache na planilha.

    Returns:
        pd.DataFrame contendo as 21 colunas de C15 (sem coluna de professor(a)),
        com attrs["ch_origem"] = "planilha" ou "recalculada".

    Raises:
        FileNotFoundError: Se o arquivo especificado não existir.
        ValueError: Se a extensão não for .xlsx, se o cabeçalho estiver trocado,
                    se turmas estiverem fora do padrão, se aulas forem inválidas/negativas,
                    se Aulas/sem for inconsistente, se CH efetiva ≠ soma dos bimestres,
                    ou se houver chaves duplicadas. Toda mensagem informa a aba e a linha.
    """
    if caminho is None:
        caminho = CAMINHO_CH_EFETIVA_PADRAO
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de CH efetiva não encontrado: '{caminho}'")

    if caminho.suffix.lower() != ".xlsx":
        raise ValueError(
            f"Extensão de arquivo inválida: '{caminho.suffix}'. A planilha deve ser .xlsx."
        )

    wb = openpyxl.load_workbook(caminho, data_only=True)

    if ABA_CH_DISCIPLINA not in wb.sheetnames:
        raise ValueError(
            f"Aba obrigatória '{ABA_CH_DISCIPLINA}' não encontrada no arquivo '{caminho.name}'."
        )

    ws = wb[ABA_CH_DISCIPLINA]

    # 1. Validação estrita do cabeçalho da linha 1
    header_lido = [
        ws.cell(row=1, column=c).value
        for c in range(1, len(CABECALHO_CH_DISCIPLINA) + 1)
    ]
    if header_lido != list(CABECALHO_CH_DISCIPLINA):
        raise ValueError(
            f"Erro na aba '{ABA_CH_DISCIPLINA}', linha 1: cabeçalho trocado ou inválido. "
            f"Esperado: {list(CABECALHO_CH_DISCIPLINA)}. Obtido: {header_lido}."
        )

    dias_calendario = _extrair_dias_calendario(wb)

    linhas_df: list[dict[str, Any]] = []
    chaves_vistas: dict[tuple[str, str, str], int] = {}
    todos_com_cache: bool = True

    for r in range(2, ws.max_row + 1):
        curso_raw = ws.cell(row=r, column=1).value
        turma_raw = ws.cell(row=r, column=2).value
        subgrupo_raw = ws.cell(row=r, column=3).value
        sigla_raw = ws.cell(row=r, column=4).value
        disc_raw = ws.cell(row=r, column=5).value
        # Coluna 6: Professor(a) é descartada na carga (minimização LGPD)

        # Linha em branco no final da planilha
        if curso_raw is None and turma_raw is None and disc_raw is None:
            continue

        turma = str(turma_raw or "").strip()
        m_turma = PADRAO_TURMA.match(turma)
        if not m_turma:
            raise ValueError(
                f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: turma '{turma}' fora do "
                "padrão esperado <CURSO>-<SERIE><LETRA> (ex.: EST-2A, EST/TT-2A)."
            )
        serie = int(m_turma.group(2))
        letra = m_turma.group(3)

        # Normalização do subgrupo: '—' -> ""
        subgrupo_str = str(subgrupo_raw or "").strip()
        if subgrupo_str in ("—", "-", ""):
            subgrupo = ""
        else:
            subgrupo = subgrupo_str

        sigla = str(sigla_raw or "").strip() if sigla_raw is not None else ""
        disciplina = str(disc_raw or "").strip()

        # Validação de chave única (turma, subgrupo, disciplina)
        chave_linha = (turma, subgrupo, disciplina)
        if chave_linha in chaves_vistas:
            linha_anterior = chaves_vistas[chave_linha]
            raise ValueError(
                f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: chave duplicada "
                f"(turma='{turma}', subgrupo='{subgrupo}', disciplina='{disciplina}'), "
                f"já encontrada na linha {linha_anterior}."
            )
        chaves_vistas[chave_linha] = r

        # Leitura e validação das aulas por dia útil (SEG a SEX, colunas 7 a 11)
        aulas_dia: dict[str, int] = {}
        for c_offset, dia in enumerate(DIAS_UTEIS):
            col_num = 7 + c_offset
            cel_val = ws.cell(row=r, column=col_num).value
            if cel_val is None or cel_val == "":
                cel_val = 0

            if isinstance(cel_val, (int, float)):
                if cel_val < 0:
                    raise ValueError(
                        f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: aula negativa {dia}={cel_val}."
                    )
                if int(cel_val) != cel_val:
                    raise ValueError(
                        f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: aula não inteira {dia}={cel_val}."
                    )
                aulas_dia[dia] = int(cel_val)
            else:
                # String que pode ser "-1", "1,5", etc.
                s_val = str(cel_val).strip()
                if "," in s_val or "." in s_val:
                    raise ValueError(
                        f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: aula não inteira {dia}={cel_val}."
                    )
                try:
                    num_val = int(s_val)
                except ValueError:
                    raise ValueError(
                        f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: quantidade de aulas inválida {dia}={cel_val}."
                    )
                if num_val < 0:
                    raise ValueError(
                        f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: aula negativa {dia}={cel_val}."
                    )
                aulas_dia[dia] = num_val

        soma_aulas_semana = sum(aulas_dia.values())

        # Leitura das colunas calculadas na planilha (Diferença e % do nominal,
        # colunas 19 e 20, são derivadas e descartadas na carga junto com Professor(a))
        v_aulas_sem = ws.cell(row=r, column=12).value
        v_ch_nom = ws.cell(row=r, column=13).value
        v_b1 = ws.cell(row=r, column=14).value
        v_b2 = ws.cell(row=r, column=15).value
        v_b3 = ws.cell(row=r, column=16).value
        v_b4 = ws.cell(row=r, column=17).value
        v_ch_efetiva = ws.cell(row=r, column=18).value

        # Detecção de cache numérico vs. fórmulas não avaliadas
        valores_calc = [v_aulas_sem, v_ch_nom, v_b1, v_b2, v_b3, v_b4, v_ch_efetiva]
        tem_valores_numericos = all(
            v is not None and not (isinstance(v, str) and v.startswith("="))
            for v in valores_calc
        )

        if not tem_valores_numericos or forcar_recalculo:
            todos_com_cache = False
            if dias_calendario is None:
                raise ValueError(
                    f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: colunas de CH sem valor "
                    f"em cache e aba '{ABA_CALENDARIO}' ilegível — impossível recalcular."
                )
            aulas_sem = soma_aulas_semana
            ch_nominal_calc = aulas_sem * 40
            ch_b1 = sum(aulas_dia[d] * dias_calendario[1][d] for d in DIAS_UTEIS)
            ch_b2 = sum(aulas_dia[d] * dias_calendario[2][d] for d in DIAS_UTEIS)
            ch_b3 = sum(aulas_dia[d] * dias_calendario[3][d] for d in DIAS_UTEIS)
            ch_b4 = sum(aulas_dia[d] * dias_calendario[4][d] for d in DIAS_UTEIS)
            ch_efetiva = ch_b1 + ch_b2 + ch_b3 + ch_b4
            if tem_valores_numericos and forcar_recalculo:
                # Em recálculo forçado, valida o que a planilha declara
                if int(v_aulas_sem) != aulas_sem:
                    raise ValueError(
                        f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: 'Aulas/sem' inconsistente "
                        f"({int(v_aulas_sem)} != soma dos dias {soma_aulas_semana})."
                    )
            ch_nominal = ch_nominal_calc
        else:
            try:
                aulas_sem = int(v_aulas_sem)
            except (ValueError, TypeError):
                aulas_sem = soma_aulas_semana
            if aulas_sem != soma_aulas_semana:
                raise ValueError(
                    f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: 'Aulas/sem' inconsistente "
                    f"({aulas_sem} != soma dos dias {soma_aulas_semana})."
                )

            ch_nominal = int(v_ch_nom)
            if ch_nominal != aulas_sem * 40:
                raise ValueError(
                    f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: 'CH nominal' ({ch_nominal}) "
                    f"≠ aulas/sem × 40 ({aulas_sem * 40})."
                )
            ch_b1 = int(v_b1)
            ch_b2 = int(v_b2)
            ch_b3 = int(v_b3)
            ch_b4 = int(v_b4)
            ch_efetiva = int(v_ch_efetiva)

            soma_bimestres = ch_b1 + ch_b2 + ch_b3 + ch_b4
            if ch_efetiva != soma_bimestres:
                raise ValueError(
                    f"Erro na aba '{ABA_CH_DISCIPLINA}', linha {r}: 'CH efetiva' ({ch_efetiva}) "
                    f"≠ soma dos bimestres ({soma_bimestres})."
                )

        linhas_df.append(
            {
                "curso": str(curso_raw or "").strip(),
                "turma": turma,
                "serie": serie,
                "letra": letra,
                "subgrupo": subgrupo,
                "sigla": sigla,
                "disciplina": disciplina,
                "disciplina_norm": normalizar_disciplina(disciplina),
                "seg": aulas_dia["SEG"],
                "ter": aulas_dia["TER"],
                "qua": aulas_dia["QUA"],
                "qui": aulas_dia["QUI"],
                "sex": aulas_dia["SEX"],
                "aulas_sem": aulas_sem,
                "ch_nominal": ch_nominal,
                "ch_bim_1": ch_b1,
                "ch_bim_2": ch_b2,
                "ch_bim_3": ch_b3,
                "ch_bim_4": ch_b4,
                "ch_efetiva": ch_efetiva,
            }
        )

    df = pd.DataFrame(linhas_df, columns=list(COLUNAS_CH_EFETIVA))
    df.attrs["ch_origem"] = (
        "recalculada" if (not todos_com_cache or forcar_recalculo) else "planilha"
    )
    df.attrs["dias_calendario"] = dias_calendario
    df.attrs["caminho"] = caminho
    return df


def resumo_por_carga(dados: pd.DataFrame | str | Path) -> dict[str, int]:
    """Calcula a distribuição agregada das disciplinas por percentual da CH nominal (C15).

    A razão é calculada de ``ch_efetiva / ch_nominal`` (não lida de coluna
    derivada da planilha), nas mesmas faixas da aba 'Resumo por carga':

        - "< 90%":  razão < 0,90
        - "90–95%": 0,90 <= razão < 0,95
        - "≥ 95%":  razão >= 0,95
        - "total":  total de disciplinas computadas

    Args:
        dados: DataFrame retornado por carregar_ch_efetiva ou caminho para a planilha.

    Returns:
        Dicionário com as quatro contagens (zeros incluídos).
    """
    if isinstance(dados, pd.DataFrame):
        df = dados
    else:
        df = carregar_ch_efetiva(dados)

    ch_nominal = pd.to_numeric(df["ch_nominal"], errors="coerce")
    ch_efetiva = pd.to_numeric(df["ch_efetiva"], errors="coerce")
    razao = (ch_efetiva / ch_nominal).where(ch_nominal > 0).dropna().round(6)

    return {
        "< 90%": int((razao < 0.90).sum()),
        "90–95%": int(((razao >= 0.90) & (razao < 0.95)).sum()),
        "≥ 95%": int((razao >= 0.95).sum()),
        "total": int(len(df)),
    }


def divergencias_calendario(
    planilha: pd.DataFrame | str | Path,
    cal: Calendario | None = None,
) -> list[str]:
    """Cruza a planilha de CH efetiva com o calendário escolar oficial e aponta divergências (C16).

    Compara:
        1. A matriz de dias letivos úteis da aba 'Calendário' com as contagens
           oficiais homologadas em cal.bimestres[b].dias_semana.
        2. A carga horária de cada disciplina por bimestre (1º BI a 4º BI) com o
           valor oficial esperado por ch_lecionada(cal, ..., cenario="A", bimestres=b).
        3. A CH efetiva de cada disciplina em relação aos extremos oficiais
           determinados por faixa_ch(cal, aulas_sem, "A").

    Args:
        planilha: DataFrame retornado por carregar_ch_efetiva ou caminho para a planilha .xlsx.
        cal: Instância de Calendario oficial. Se None, carrega via carregar_calendario().

    Returns:
        Lista de descrições textuais das divergências encontradas.
    """
    if cal is None:
        cal = carregar_calendario()

    divergencias: list[str] = []
    dias_planilha: dict[int, dict[str, int]] | None = None

    if isinstance(planilha, (str, Path)):
        caminho = Path(planilha)
        wb = openpyxl.load_workbook(caminho, data_only=True)
        dias_planilha = _extrair_dias_calendario(wb)
        wb.close()
        df = carregar_ch_efetiva(caminho)
    elif isinstance(planilha, pd.DataFrame):
        df = planilha
        dias_planilha = df.attrs.get("dias_calendario")
        if dias_planilha is None and "caminho" in df.attrs:
            caminho_attr = df.attrs["caminho"]
            if caminho_attr and Path(caminho_attr).exists():
                wb = openpyxl.load_workbook(caminho_attr, data_only=True)
                dias_planilha = _extrair_dias_calendario(wb)
                wb.close()
    else:
        raise TypeError(f"Tipo não suportado para planilha: {type(planilha)}")

    # 1. Comparação da aba 'Calendário' com cal oficial
    if dias_planilha is not None:
        for b in (1, 2, 3, 4):
            if b in cal.bimestres and b in dias_planilha:
                b_oficial = cal.bimestres[b].dias_semana
                for d in DIAS_UTEIS:
                    val_planilha = dias_planilha[b].get(d)
                    val_oficial = b_oficial.get(d)
                    if (
                        val_planilha is not None
                        and val_oficial is not None
                        and val_planilha != val_oficial
                    ):
                        divergencias.append(
                            f"Aba 'Calendário', {b}º BI, {d}: {val_planilha} × {val_oficial}"
                        )

    # 2. Comparação das disciplinas na grade com cal oficial
    for _, row in df.iterrows():
        turma = str(row.get("turma", "")).strip()
        disciplina = str(row.get("disciplina", "")).strip()
        aulas_sem = row.get("aulas_sem")
        ch_efetiva = row.get("ch_efetiva")

        dist = {d: int(row.get(d.lower(), 0)) for d in DIAS_UTEIS}

        # Verificação bimestral (1º a 4º BI)
        for b in (1, 2, 3, 4):
            col_b = f"ch_bim_{b}"
            if col_b in row and pd.notna(row[col_b]):
                ch_row = int(row[col_b])
                ch_esperada = ch_lecionada(cal, dist, cenario="A", bimestres=b)
                if ch_row != ch_esperada:
                    divergencias.append(
                        f"Turma {turma}, {disciplina}: {b}º BI ({ch_row} × {ch_esperada})"
                    )

        # Verificação da faixa de CH efetiva anual
        if (
            aulas_sem is not None
            and pd.notna(aulas_sem)
            and ch_efetiva is not None
            and pd.notna(ch_efetiva)
        ):
            aulas_sem_int = int(aulas_sem)
            ch_efetiva_int = int(ch_efetiva)
            if aulas_sem_int in (1, 2, 3, 4):
                min_ch, _, max_ch, _ = faixa_ch(cal, aulas_sem_int, cenario="A")
                if ch_efetiva_int < min_ch or ch_efetiva_int > max_ch:
                    divergencias.append(
                        f"Turma {turma}, {disciplina}: CH efetiva fora da faixa "
                        f"({ch_efetiva_int} fora de {min_ch}–{max_ch})"
                    )

    return divergencias


def nota_sabados(cal: Calendario, curso: str) -> str:
    """Gera nota explicativa sobre sábados letivos do curso e a CH efetiva (C16).

    Informa as datas e bimestres dos sábados letivos sob responsabilidade da coordenação
    do curso e esclarece que a planilha de CH efetiva é, por construção, o cenário A
    (os sábados letivos não entram no cômputo das horas-aula).

    Args:
        cal: Instância de Calendario oficial.
        curso: Nome do curso (ex.: "Estradas").

    Returns:
        Texto explicativo formatado.
    """
    nome_curso = curso
    sabs = sabados_do_responsavel(cal, responsavel=nome_curso)
    if not sabs:
        return (
            f"Não há sábados letivos atribuídos ao curso '{nome_curso}' no calendário oficial. "
            f"Os sábados letivos não entram no cômputo da carga horária efetiva das disciplinas (cenário A)."
        )

    detalhes: list[str] = []
    for dt in sabs:
        b_num = None
        for num, b in cal.bimestres.items():
            if b.inicio <= dt <= b.fim:
                b_num = num
                break
        fmt_dt = dt.strftime("%d/%m")
        if b_num is not None:
            detalhes.append(f"{fmt_dt} ({b_num}º bimestre)")
        else:
            detalhes.append(fmt_dt)

    sabs_str = ", ".join(detalhes)
    return (
        f"Sábado(s) letivo(s) atribuído(s) ao curso '{nome_curso}': {sabs_str}. "
        f"Os sábados letivos não entram no cômputo da carga horária efetiva das disciplinas (cenário A)."
    )



# ==============================================================================
# Casamento Mapa ↔ Planilha de CH Efetiva (C17)
# ==============================================================================


def _extrair_letra(identificador: str) -> str:
    """Extrai a letra da turma do identificador do mapa (ex.: 'BH-1EST - A (2025)' -> 'A')."""
    s = str(identificador).strip()
    if len(s) == 1 and s.isalpha():
        return s.upper()
    m = re.search(r"-\s*([A-Z])\s*\(", s) or re.search(r"(\d)([A-Z])$", s)
    if m:
        grupos = m.groups()
        return (grupos[-1]).upper()
    raise ValueError(f"Não foi possível extrair a letra da turma de '{identificador}'")


def ch_da_turma(
    df: pd.DataFrame,
    curso_amigavel: str,
    serie: int,
    turma: str,
) -> pd.DataFrame:
    """Filtra as linhas de CH efetiva correspondentes à turma especificada (C17).

    Linhas cujo ``curso`` normalizado contém ``curso_amigavel`` normalizado (pega
    'Estradas' e 'Estradas / Trânsito (núcleo comum)') e cuja turma termina em
    ``-<serie><letra>``, com ``letra`` extraída da turma do mapa.

    Args:
        df: DataFrame retornado por carregar_ch_efetiva.
        curso_amigavel: Nome amigável do curso (ex.: 'Estradas').
        serie: Série curricular (1, 2 ou 3).
        turma: Identificador da turma no mapa (ex.: 'TÉCNICO EM ESTRADAS - BH-1EST - A (2025)').

    Returns:
        pd.DataFrame com apenas as ofertas da turma filtrada.
    """
    letra = _extrair_letra(turma)
    alvo = normalizar_disciplina(str(curso_amigavel)).lower().strip()
    mask = (
        df["curso"].apply(lambda c: alvo in normalizar_disciplina(c).lower())
        & (df["serie"] == int(serie))
        & (df["letra"] == letra)
    )
    return df[mask].reset_index(drop=True)


ALIASES_DISCIPLINA: dict[str, str] = {
    "LABORATORIO DE DE PESQUISA DE TRANSPORTES E TRANSITO": "L. DE PESQUISA DE TRANSPORTE E TRANSITO",
    "LABORATORIO DE TOPOGRAFIA URBANA": "L. DE TOPOGRAFIA URBANA",
}
"""Mapeamento explícito de aliases de disciplinas entre o mapa de turma e a planilha de CH efetiva (D3).

Cada entrada é chave normalizada do mapa → nome normalizado da planilha, sem casamento aproximado.
"""


def casar_disciplinas(
    legenda: dict[str, str],
    df_turma: pd.DataFrame,
) -> tuple[dict[str, dict], list[str]]:
    """Casa as disciplinas da legenda do mapa com a grade de CH efetiva da turma (C17).

    Casamento determinístico por ``disciplina_norm`` exata. Para cada código da
    legenda que casa:

        - ``disciplina``: nome da disciplina na planilha;
        - ``aulas_sem``, ``ch_nominal``, ``ch_bim_1``..``ch_bim_4``, ``ch_efetiva``;
        - ``arranjo``: aulas por dia útil com aulas (> 0);
        - ``subgrupos_divergentes``: True quando a disciplina tem subgrupos
          (T1/T2) com CH diferente — adota-se a MENOR CH (conservador, C5).

    O segundo elemento da tupla é a lista dos nomes da legenda **sem** linha na
    planilha (típico de disciplinas em laboratório/quadra, fora das salas 305-437).

    Args:
        legenda: Dicionário {código: nome} da legenda do mapa de turma.
        df_turma: DataFrame filtrado por ch_da_turma.

    Returns:
        Tupla (casadas, sem_linha).
    """
    casadas: dict[str, dict] = {}
    sem_linha: list[str] = []

    df_norm = df_turma.copy()
    if "disciplina_norm" in df_norm.columns:
        normas = df_norm["disciplina_norm"]
    else:
        normas = df_norm["disciplina"].apply(normalizar_disciplina)

    for cod, nome_mapa in legenda.items():
        norm_mapa = normalizar_disciplina(nome_mapa)
        norm_mapa = ALIASES_DISCIPLINA.get(norm_mapa, norm_mapa)
        matches = df_norm[normas == norm_mapa]
        if matches.empty:
            sem_linha.append(nome_mapa)
            continue

        chs = {
            tuple(int(r[f"ch_bim_{b}"]) for b in (1, 2, 3, 4)) + (int(r["ch_efetiva"]),)
            for _, r in matches.iterrows()
        }
        subgrupos_divergentes = len(chs) > 1
        if subgrupos_divergentes:
            # conservador: menor CH efetiva (desempate pelos bimestres)
            escolhida = matches.sort_values(
                by=["ch_efetiva", "ch_bim_1", "ch_bim_2", "ch_bim_3", "ch_bim_4"]
            ).iloc[0]
        else:
            escolhida = matches.iloc[0]

        casadas[str(cod)] = {
            "disciplina": str(escolhida["disciplina"]),
            "aulas_sem": int(escolhida["aulas_sem"]),
            "ch_nominal": int(escolhida["ch_nominal"]),
            "ch_bim_1": int(escolhida["ch_bim_1"]),
            "ch_bim_2": int(escolhida["ch_bim_2"]),
            "ch_bim_3": int(escolhida["ch_bim_3"]),
            "ch_bim_4": int(escolhida["ch_bim_4"]),
            "ch_efetiva": int(escolhida["ch_efetiva"]),
            "arranjo": {
                d: int(escolhida[d.lower()])
                for d in DIAS_UTEIS
                if int(escolhida[d.lower()]) > 0
            },
            "subgrupos_divergentes": subgrupos_divergentes,
        }

    return casadas, sem_linha
