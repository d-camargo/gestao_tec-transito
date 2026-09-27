"""Carga e validação do Calendário Escolar da EPTNM (Integrado) do CEFET-MG.

Fonte:
    Arquivo Markdown oficial versionado em
    ``sandbox/dae/calendario/Calendario_Escolar_2026_EPTNM_Integrado_BH.md``,
    lastreado na Deliberação CEPE/CEFET-MG nº 17, de 30/09/2025, alterada pela
    Deliberação CEPE/CEFET-MG nº 1, de 27/02/2026 (versão de maio/2026).

Autoridade das tabelas-resumo (C2):
    As tabelas-resumo da seção "Visão geral" (Visão geral de bimestres, "Dias letivos
    por dia da semana" e "Dias letivos por mês") constituem a autoridade primária de
    dados sobre o calendário oficial. Eventuais resíduos ou divergências pontuais nas
    listagens descritivas do detalhamento linha a linha (como uma quinta-feira de abril
    não explicitada nos dias sem aula) não sobrepõem os totais oficiais homologados
    nas tabelas-resumo (total de 200 dias letivos anuais).

Cenários de apuração e carga horária (C5):
    - Cenário A (Piso / Padrão): Considera apenas as aulas ministradas em dias úteis
      regulares (segunda a sexta-feira), desconsiderando sábados letivos. É o cenário
      conservador oficial do anexo para monitoramento de frequência discente (denominador
      menor, gerando alertas precoces quando < 75%).
    - Cenário REAL: Cenário A somado aos sábados letivos efetivamente atribuídos à
      coordenação do curso do discente (ex.: sábado 23/05 para os Cursos Técnicos em
      Estradas e Trânsito). Sábados de áreas acadêmicas específicas não entram no REAL
      por ausência de mapeamento determinístico disciplina → área.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import itertools
from pathlib import Path
import re
from typing import Any, Iterable
import unicodedata

# Constantes de domínio (C1, C4)
BLOCOS_POR_CH: dict[int, tuple[int, ...]] = {
    1: (1,),
    2: (2,),
    3: (2, 1),
    4: (2, 2),
}
DIAS_UTEIS: tuple[str, ...] = ("SEG", "TER", "QUA", "QUI", "SEX")
DIAS_SEMANA: tuple[str, ...] = ("SEG", "TER", "QUA", "QUI", "SEX", "SAB")
CENARIOS: tuple[str, ...] = ("A", "REAL")
ANO_PADRAO: int = 2026
SEMANAS_NOMINAIS: int = 40

PASTA_CALENDARIO: Path = Path(__file__).resolve().parent / "calendario"
CAMINHO_MD_PADRAO: Path = PASTA_CALENDARIO / "Calendario_Escolar_2026_EPTNM_Integrado_BH.md"


@dataclass(frozen=True)
class SabadoLetivo:
    """Sábado letivo programado no calendário com data e responsável."""

    data: date
    responsavel: str

    def __iter__(self):
        return iter((self.data, self.responsavel))


@dataclass(frozen=True)
class Bimestre:
    """Dados oficiais de um bimestre letivo."""

    numero: int
    inicio: date
    fim: date
    dias_letivos: int
    acumulado: int
    limite_diarios: date
    dias_semana: dict[str, int]
    sabados: tuple[SabadoLetivo, ...] = ()
    sem_aula: dict[date, str] = field(default_factory=dict)


class TabelaDiasSemana(dict):
    """Representação da tabela de dias letivos por dia da semana (linhas por bimestre e linha Soma)."""

    def __init__(self, linhas: dict[Any, dict[str, int]], soma: dict[str, int]) -> None:
        super().__init__(linhas)
        self["Soma"] = soma
        self["Total"] = soma
        self._soma = soma

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            k_upper = key.upper()
            if k_upper in ("SEG", "TER", "QUA", "QUI", "SEX", "SAB", "SÁB"):
                k_norm = "SAB" if k_upper in ("SAB", "SÁB") else k_upper
                return self._soma.get(k_norm, self._soma.get(key, 0))
        return super().__getitem__(key)


@dataclass(frozen=True)
class Calendario:
    """Estrutura imutável contendo os dados oficiais do calendário escolar."""

    ano: int
    bimestres: dict[int, Bimestre]
    dias_semana: TabelaDiasSemana
    dias_mes: dict[int, int]
    deliberacao: str = ""
    titulo: str = ""

    @property
    def sabados(self) -> tuple[SabadoLetivo, ...]:
        """Todos os sábados letivos do calendário ordenados."""
        return tuple(s for b in self.bimestres.values() for s in b.sabados)

    @property
    def limite_diarios(self) -> dict[int, date]:
        """Data-limite para preenchimento de diários por bimestre."""
        return {b.numero: b.limite_diarios for b in self.bimestres.values()}


def _parse_data(texto: str, ano: int) -> date:
    """Extrai dia e mês no formato DD/MM (com ou sem anotações adicionais) e retorna date."""
    m = re.search(r"(\d{1,2})/(\d{1,2})", texto)
    if not m:
        raise ValueError(f"Formato de data inválido: '{texto}'")
    dia = int(m.group(1))
    mes = int(m.group(2))
    return date(ano, mes, dia)


def _extrair_tabela(linhas_secao: list[str]) -> tuple[list[str], list[list[str]]]:
    """Parseia a primeira tabela Markdown presente na lista de linhas fornecida."""
    cabecalho: list[str] = []
    linhas_dados: list[list[str]] = []
    em_tabela = False

    for l in linhas_secao:
        l_str = l.strip()
        if l_str.startswith("|") and l_str.endswith("|"):
            celulas = [c.strip().replace("**", "") for c in l_str.split("|")[1:-1]]
            if all(set(c) <= {"-", ":", " "} for c in celulas):
                continue
            if not cabecalho:
                cabecalho = celulas
                em_tabela = True
            else:
                linhas_dados.append(celulas)
        elif em_tabela:
            break

    return cabecalho, linhas_dados


def _get_secao_linhas(linhas: list[str], pattern_inicio: str, pattern_fim: str) -> list[str] | None:
    """Retorna as linhas pertencentes a uma seção delimitada por padrões de início e fim."""
    inicio = None
    for i, l in enumerate(linhas):
        if re.search(pattern_inicio, l, re.IGNORECASE):
            inicio = i + 1
            break

    if inicio is None:
        return None

    fim = len(linhas)
    for j in range(inicio, len(linhas)):
        if re.search(pattern_fim, linhas[j], re.IGNORECASE):
            fim = j
            break

    return linhas[inicio:fim]


def carregar_calendario(ano: int = ANO_PADRAO, caminho: Path | str | None = None) -> Calendario:
    """Carrega e valida o Calendário Escolar a partir do arquivo Markdown oficial.

    Args:
        ano: Ano de referência do calendário (padrão: 2026).
        caminho: Caminho opcional para o arquivo Markdown. Se omitido, usa o caminho padrão.

    Returns:
        Instância de Calendario devidamente validada.

    Raises:
        FileNotFoundError: Caso o arquivo não seja encontrado.
        ValueError: Caso alguma seção obrigatória esteja ausente ou haja inconsistência nos dados (C3).
    """
    if caminho is not None:
        caminho_arq = Path(caminho)
    elif ano == ANO_PADRAO:
        caminho_arq = CAMINHO_MD_PADRAO
    else:
        # C1: nome do arquivo por ano (Calendario_Escolar_<ano>_...)
        caminho_arq = PASTA_CALENDARIO / f"Calendario_Escolar_{ano}_EPTNM_Integrado_BH.md"
    if not caminho_arq.exists():
        raise FileNotFoundError(f"Arquivo de calendário não encontrado: {caminho_arq}")

    texto = caminho_arq.read_text(encoding="utf-8")
    linhas = texto.splitlines()

    m_titulo = re.search(r"^#\s*([^\n]+)", texto, re.MULTILINE)
    titulo = m_titulo.group(1).strip() if m_titulo else ""
    m_delib = re.search(r"(Deliberação[^\n]+)", texto)
    deliberacao = m_delib.group(1).strip() if m_delib else ""

    if ano == ANO_PADRAO and m_titulo:
        m_ano = re.search(r"20\d\d", m_titulo.group(1))
        if m_ano:
            ano = int(m_ano.group(0))

    # 1. Seção "Visão geral"
    linhas_vg = _get_secao_linhas(linhas, r"^##\s*Visão geral", r"^###\s|^##\s|^---\s*$")
    if linhas_vg is None:
        raise ValueError("Seção obrigatória ausente: 'Visão geral'")
    cab_vg, dados_vg = _extrair_tabela(linhas_vg)
    if not dados_vg:
        raise ValueError("Tabela ausente na seção 'Visão geral'")

    bimestres_info: dict[int, dict[str, Any]] = {}
    acumulado_calc = 0
    for linha in dados_vg:
        num_m = re.search(r"(\d+)", linha[0])
        if not num_m:
            continue
        num_bim = int(num_m.group(1))
        d_inicio = _parse_data(linha[1], ano)
        d_fim = _parse_data(linha[2], ano)
        dias_let = int(linha[3])
        acumulado = int(linha[4])
        d_limite = _parse_data(linha[5], ano)

        if dias_let < 0 or acumulado < 0:
            raise ValueError("Valor negativo encontrado na seção 'Visão geral'")

        acumulado_calc += dias_let
        if acumulado != acumulado_calc:
            raise ValueError(
                f"Na seção 'Visão geral', o Acumulado do {num_bim}º bimestre ({acumulado}) "
                f"não confere com o acumulado esperado ({acumulado_calc})"
            )

        bimestres_info[num_bim] = {
            "inicio": d_inicio,
            "fim": d_fim,
            "dias_letivos": dias_let,
            "acumulado": acumulado,
            "limite_diarios": d_limite,
        }

    # 2. Subseção "Dias letivos por dia da semana"
    linhas_sem = _get_secao_linhas(linhas, r"^###\s*Dias letivos por dia da semana", r"^###\s|^##\s|^---\s*$")
    if linhas_sem is None:
        raise ValueError("Seção obrigatória ausente: 'Dias letivos por dia da semana'")
    cab_sem, dados_sem = _extrair_tabela(linhas_sem)
    if not dados_sem:
        raise ValueError("Tabela ausente na seção 'Dias letivos por dia da semana'")

    dias_semana_linhas: dict[Any, dict[str, int]] = {}
    soma_esperada_colunas = {d: 0 for d in ("SEG", "TER", "QUA", "QUI", "SEX", "SAB", "Total")}
    linha_soma: dict[str, int] | None = None

    for linha in dados_sem:
        rotulo = linha[0].strip()
        valores_dias: dict[str, int] = {}
        dias_cols = ["SEG", "TER", "QUA", "QUI", "SEX", "SAB"]
        soma_linha = 0

        for idx_c, col_nome in enumerate(dias_cols, start=1):
            val = int(linha[idx_c])
            if val < 0:
                raise ValueError("Valor negativo encontrado na seção 'Dias letivos por dia da semana'")
            valores_dias[col_nome] = val
            if col_nome == "SAB":
                valores_dias["SÁB"] = val
            soma_linha += val

        tot_linha = int(linha[7])
        if tot_linha < 0:
            raise ValueError("Valor negativo encontrado na seção 'Dias letivos por dia da semana'")
        if soma_linha != tot_linha:
            raise ValueError(
                f"Na seção 'Dias letivos por dia da semana', a soma dos dias ({soma_linha}) "
                f"não confere com o Total ({tot_linha}) na linha '{rotulo}'"
            )
        valores_dias["Total"] = tot_linha

        if "soma" in rotulo.lower():
            linha_soma = valores_dias
        else:
            num_m = re.search(r"(\d+)", rotulo)
            if num_m:
                b_num = int(num_m.group(1))
                if b_num in bimestres_info and tot_linha != bimestres_info[b_num]["dias_letivos"]:
                    raise ValueError(
                        f"Na seção 'Dias letivos por dia da semana', o Total do {b_num}º bimestre "
                        f"({tot_linha}) não confere com os Dias letivos da 'Visão geral' "
                        f"({bimestres_info[b_num]['dias_letivos']})"
                    )
                dias_semana_linhas[b_num] = valores_dias
                dias_semana_linhas[rotulo] = valores_dias
                for k in ("SEG", "TER", "QUA", "QUI", "SEX", "SAB", "Total"):
                    soma_esperada_colunas[k] += valores_dias[k]

    if linha_soma is None:
        raise ValueError("Linha 'Soma' ausente na seção 'Dias letivos por dia da semana'")

    for k in ("SEG", "TER", "QUA", "QUI", "SEX", "SAB", "Total"):
        if linha_soma[k] != soma_esperada_colunas[k]:
            raise ValueError(
                f"Na seção 'Dias letivos por dia da semana', a coluna {k} da Soma ({linha_soma[k]}) "
                f"não confere com a soma das linhas ({soma_esperada_colunas[k]})"
            )

    tabela_dias_semana = TabelaDiasSemana(dias_semana_linhas, linha_soma)

    # 3. Subseção "Dias letivos por mês"
    linhas_mes = _get_secao_linhas(linhas, r"^###\s*Dias letivos por mês", r"^###\s|^##\s|^---\s*$")
    if linhas_mes is None:
        raise ValueError("Seção obrigatória ausente: 'Dias letivos por mês'")
    cab_mes, dados_mes = _extrair_tabela(linhas_mes)
    if not dados_mes:
        raise ValueError("Tabela ausente na seção 'Dias letivos por mês'")

    mapa_meses = {
        "fev": 2,
        "mar": 3,
        "abr": 4,
        "mai": 5,
        "jun": 6,
        "jul": 7,
        "ago": 8,
        "set": 9,
        "out": 10,
        "nov": 11,
        "dez": 12,
    }
    dias_mes: dict[int, int] = {}
    linha_m = dados_mes[0]
    total_meses_tabela: int | None = None
    soma_meses = 0

    for col_hdr, val_str in zip(cab_mes, linha_m):
        col_norm = col_hdr.strip().lower()
        val = int(val_str)
        if val < 0:
            raise ValueError("Valor negativo encontrado na seção 'Dias letivos por mês'")
        if col_norm in mapa_meses:
            m_num = mapa_meses[col_norm]
            dias_mes[m_num] = val
            soma_meses += val
        elif col_norm == "total":
            total_meses_tabela = val

    if total_meses_tabela is not None and soma_meses != total_meses_tabela:
        raise ValueError(
            f"Na seção 'Dias letivos por mês', a soma dos meses ({soma_meses}) "
            f"não confere com a coluna Total ({total_meses_tabela})"
        )
    if soma_meses != 200:
        raise ValueError(
            f"Na seção 'Dias letivos por mês', a soma dos meses ({soma_meses}) "
            f"não confere com o total anual de 200"
        )

    # 4. Bimestres 1 a 4
    bimestres_obj: dict[int, Bimestre] = {}
    for i in (1, 2, 3, 4):
        linhas_bim = _get_secao_linhas(linhas, rf"^##\s*{i}[ºo]\s*Bimestre", r"^##\s")
        if linhas_bim is None:
            raise ValueError(f"Seção obrigatória ausente: '{i}º Bimestre'")

        # Dias sem aula (dias úteis)
        sem_aula: dict[date, str] = {}
        linhas_sem_aula = _get_secao_linhas(linhas_bim, r"^\*\*Dias sem aula", r"^\*\*|^##\s|^---\s*$")
        if linhas_sem_aula is not None:
            _, dados_sa = _extrair_tabela(linhas_sem_aula)
            for r in dados_sa:
                dt_sa = _parse_data(r[0], ano)
                motivo = r[1].strip()
                sem_aula[dt_sa] = motivo

        # Sábados letivos
        sabados: list[SabadoLetivo] = []
        linhas_sab = _get_secao_linhas(linhas_bim, r"^\*\*Sábados letivos", r"^\*\*|^##\s|^---\s*$")
        if linhas_sab is None:
            raise ValueError(f"Na seção '{i}º Bimestre', subseção 'Sábados letivos' ausente")
        _, dados_sab = _extrair_tabela(linhas_sab)
        for r in dados_sab:
            dt_sab = _parse_data(r[0], ano)
            if dt_sab.weekday() != 5:
                dias_semana_nomes = [
                    "segunda-feira",
                    "terça-feira",
                    "quarta-feira",
                    "quinta-feira",
                    "sexta-feira",
                    "sábado",
                    "domingo",
                ]
                nome_dia = dias_semana_nomes[dt_sab.weekday()]
                raise ValueError(
                    f"Na seção '{i}º Bimestre' (Sábados letivos), a data {dt_sab.strftime('%d/%m/%Y')} "
                    f"não cai em um sábado (cai em {nome_dia})"
                )
            inicio_bim = bimestres_info[i]["inicio"]
            fim_bim = bimestres_info[i]["fim"]
            if not (inicio_bim <= dt_sab <= fim_bim):
                raise ValueError(
                    f"Na seção '{i}º Bimestre' (Sábados letivos), a data "
                    f"{dt_sab.strftime('%d/%m/%Y')} está fora do período do bimestre "
                    f"({inicio_bim.strftime('%d/%m')} a {fim_bim.strftime('%d/%m')})"
                )
            resp = r[1].strip()
            sabados.append(SabadoLetivo(data=dt_sab, responsavel=resp))

        esperado_sab = dias_semana_linhas[i]["SAB"]
        if len(sabados) != esperado_sab:
            raise ValueError(
                f"Na seção '{i}º Bimestre' (Sábados letivos), foram encontrados {len(sabados)} sábados, "
                f"mas eram esperados {esperado_sab}"
            )

        info = bimestres_info[i]
        bimestres_obj[i] = Bimestre(
            numero=i,
            inicio=info["inicio"],
            fim=info["fim"],
            dias_letivos=info["dias_letivos"],
            acumulado=info["acumulado"],
            limite_diarios=info["limite_diarios"],
            dias_semana=dias_semana_linhas[i],
            sabados=tuple(sabados),
            sem_aula=sem_aula,
        )

    return Calendario(
        ano=ano,
        bimestres=bimestres_obj,
        dias_semana=tabela_dias_semana,
        dias_mes=dias_mes,
        deliberacao=deliberacao,
        titulo=titulo,
    )


def _remover_acentos(txt: str) -> str:
    """Remove caracteres diacríticos (acentos, cedilha, etc.) de uma string."""
    if not isinstance(txt, str):
        txt = str(txt)
    return "".join(
        c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn"
    )


_STOPWORDS_RESPONSAVEL: frozenset[str] = frozenset({
    "de", "do", "da", "dos", "das", "e", "em", "para", "com", "a", "o", "as", "os", "um", "uma"
})


def _stem_palavra(w: str) -> str:
    """Redução simples de plural para singular em termos significativos."""
    w = w.lower()
    if len(w) > 3 and w.endswith("s"):
        w = w[:-1]
    return w


def _extrair_palavras_significativas(txt: str) -> set[str]:
    """Extrai conjunto de palavras significativas (sem acentos, em minúsculas e sem stopwords)."""
    txt_sem_acento = _remover_acentos(txt)
    tokens = re.findall(r"[a-zA-Z0-9]+", txt_sem_acento)
    return {_stem_palavra(t) for t in tokens if t.lower() not in _STOPWORDS_RESPONSAVEL}


def _corresponde_responsavel(consulta: str, alvo: str) -> bool:
    """Verifica se todas as palavras significativas da consulta estão presentes no alvo."""
    palavras_q = _extrair_palavras_significativas(consulta)
    if not palavras_q:
        return False
    palavras_alvo = _extrair_palavras_significativas(alvo)
    return palavras_q.issubset(palavras_alvo)


def sabados_do_responsavel(
    cal: Calendario,
    responsavel: str,
    bimestres: int | Iterable[int] | None = None,
) -> list[date]:
    """Retorna lista de datas de sábados letivos sob responsabilidade de um curso ou área.

    Args:
        cal: Instância de Calendario.
        responsavel: Nome do curso ou área responsável (ex.: "TÉCNICO EM ESTRADAS", "Trânsito").
        bimestres: Bimestre único (int) ou conjunto de bimestres. Se None, considera o ano todo.

    Returns:
        Lista de datas ordenadas dos sábados letivos correspondentes.
    """
    if bimestres is None:
        b_nums = sorted(cal.bimestres.keys())
    elif isinstance(bimestres, int):
        b_nums = [bimestres]
    else:
        b_nums = list(bimestres)

    datas: list[date] = []
    for b in b_nums:
        if b in cal.bimestres:
            for s in cal.bimestres[b].sabados:
                if _corresponde_responsavel(responsavel, s.responsavel):
                    datas.append(s.data)
    return datas


def dias_letivos(
    cal: Calendario,
    cenario: str = "A",
    bimestres: int | Iterable[int] | None = None,
    curso: str | None = None,
    sabado_reproduz: str | None = None,
) -> dict[str, int]:
    """Calcula os dias letivos por dia útil nos cenários A ou REAL.

    Args:
        cal: Instância de Calendario.
        cenario: "A" (padrão/piso, apenas dias úteis regulares) ou "REAL" (A + sábados do curso).
        bimestres: Bimestre(s) a apurar (None = todos os bimestres).
        curso: Nome do curso (obrigatório se cenario="REAL").
        sabado_reproduz: Dia útil ("SEG" a "SEX") que o sábado letivo reproduz (obrigatório se REAL).

    Returns:
        Dicionário com contagem de dias letivos por dia útil {"SEG": ..., "SEX": ...}.

    Raises:
        ValueError: Caso o cenário seja inválido ou faltem parâmetros obrigatórios no REAL.
    """
    if not isinstance(cenario, str) or cenario.upper() not in CENARIOS:
        raise ValueError(f"Cenário inválido: '{cenario}'. Use um dos cenários válidos: {CENARIOS}")
    cenario_norm = cenario.upper()

    if cenario_norm == "REAL":
        if not curso:
            raise ValueError("Para o cenário 'REAL', o parâmetro 'curso' é obrigatório.")
        if not sabado_reproduz:
            raise ValueError("Para o cenário 'REAL', o parâmetro 'sabado_reproduz' é obrigatório.")
        sab_rep = sabado_reproduz.upper().strip()
        if sab_rep not in DIAS_UTEIS:
            raise ValueError(f"sabado_reproduz inválido: '{sabado_reproduz}'. Deve ser um dos dias úteis: {DIAS_UTEIS}")
    else:
        sab_rep = None

    if bimestres is None:
        b_nums = sorted(cal.bimestres.keys())
    elif isinstance(bimestres, int):
        b_nums = [bimestres]
    else:
        b_nums = list(bimestres)

    totais = {d: 0 for d in DIAS_UTEIS}
    for b in b_nums:
        if b in cal.bimestres:
            b_dias = cal.bimestres[b].dias_semana
            for d in DIAS_UTEIS:
                totais[d] += b_dias[d]

    if cenario_norm == "REAL" and sab_rep is not None:
        sabs = sabados_do_responsavel(cal, responsavel=curso, bimestres=b_nums)
        totais[sab_rep] += len(sabs)

    return totais


def _contar_letivos_no_intervalo(bim: Bimestre, dt_ini: date, dt_fim: date) -> int:
    """Conta dias letivos (úteis sem aula excluídos + sábados letivos) em um intervalo."""
    sab_datas = {s.data for s in bim.sabados}
    cnt = 0
    cur = dt_ini
    while cur <= dt_fim:
        if cur.weekday() < 5 and cur not in bim.sem_aula:
            cnt += 1
        elif cur in sab_datas:
            cnt += 1
        cur += timedelta(days=1)
    return cnt


def dias_por_mes_bimestre(cal: Calendario) -> dict[int, dict[int, int]]:
    """Distribui os dias letivos oficiais de cada mês pelos 4 bimestres.

    Mês fora de fronteira adota o total homologado na tabela oficial de meses.
    Mês de fronteira divide os dias entre o bimestre anterior (contagem reconstruída
    dentro do seu período) e o posterior (total oficial do mês subtraído dessa contagem).

    Returns:
        Dicionário {bimestre: {mes: dias_letivos}} com cada bimestre totalizando seus dias oficiais.
    """
    resultado: dict[int, dict[int, int]] = {}
    residuos_fronteira: dict[int, int] = {}

    for b_num in (1, 2, 3, 4):
        bim = cal.bimestres[b_num]
        m_ini = bim.inicio.month
        m_fim = bim.fim.month
        res_b: dict[int, int] = {}
        for m in range(m_ini, m_fim + 1):
            if m in residuos_fronteira:
                res_b[m] = cal.dias_mes[m] - residuos_fronteira[m]
            elif b_num < 4 and m == m_fim and cal.bimestres[b_num + 1].inicio.month == m:
                cnt = _contar_letivos_no_intervalo(bim, date(bim.fim.year, m, 1), bim.fim)
                res_b[m] = cnt
                residuos_fronteira[m] = cnt
            else:
                res_b[m] = cal.dias_mes[m]
        resultado[b_num] = res_b

    return resultado


_NOMES_MESES: dict[int, str] = {
    2: "fevereiro",
    3: "março",
    4: "abril",
    5: "maio",
    6: "junho",
    7: "julho",
    8: "agosto",
    9: "setembro",
    10: "outubro",
    11: "novembro",
    12: "dezembro",
}


def divergencias(cal: Calendario) -> list[str]:
    """Compara a reconstrução dia a dia do calendário com as tabelas-resumo oficiais (C4, C8(i)).

    A reconstrução conta, dentro do período de cada bimestre, os dias úteis que
    não constam em "Dias sem aula", mais os sábados letivos listados, e repete
    a contagem por dia da semana e por mês. Cada contagem reconstruída que não
    confere com a tabela-resumo correspondente vira uma linha de divergência.
    As tabelas-resumo permanecem a autoridade (C2): a divergência é relatada,
    nunca aplicada.

    Pendência C8(i):
        Hoje a lista traz exatamente as duas linhas da quinta-feira de abril
        não listada em "Dias sem aula" do 1º BI (QUI 11 × 10 e abril 21 × 20).
        Quando essa quinta-feira for especificada no .md, a lista passa a ser
        vazia.
    """
    divs: list[str] = []
    letivos_por_mes: dict[int, int] = {}

    for b_num in sorted(cal.bimestres):
        bim = cal.bimestres[b_num]
        rec = dict.fromkeys(DIAS_SEMANA, 0)
        sab_datas = {s.data for s in bim.sabados}
        cur = bim.inicio
        while cur <= bim.fim:
            if cur in sab_datas:
                rec["SAB"] += 1
                letivos_por_mes[cur.month] = letivos_por_mes.get(cur.month, 0) + 1
            elif cur.weekday() < 5 and cur not in bim.sem_aula:
                rec[DIAS_SEMANA[cur.weekday()]] += 1
                letivos_por_mes[cur.month] = letivos_por_mes.get(cur.month, 0) + 1
            cur += timedelta(days=1)

        for d in DIAS_SEMANA:
            oficial = bim.dias_semana.get(d, rec[d])
            if rec[d] != oficial:
                divs.append(f"{b_num}º bimestre: {d} ({rec[d]} × {oficial})")

    for m_num in sorted(cal.dias_mes):
        rec_m = letivos_por_mes.get(m_num, 0)
        if rec_m != cal.dias_mes[m_num]:
            divs.append(
                f"{_NOMES_MESES[m_num]}: dias letivos ({rec_m} × {cal.dias_mes[m_num]})"
            )

    return divs


def ch_nominal(ch_semanal: int, semanas: int = SEMANAS_NOMINAIS) -> int:
    """Calcula a carga horária nominal anual a partir das aulas semanais (C4).

    Args:
        ch_semanal: Quantidade de aulas semanais (ex.: 1, 2, 3, 4).
        semanas: Número de semanas letivas nominais no ano (padrão: SEMANAS_NOMINAIS = 40).

    Returns:
        Carga horária nominal em horas-aula (ex.: 2 aulas * 40 semanas = 80).
    """
    if ch_semanal < 0:
        raise ValueError(f"Carga horária semanal não pode ser negativa: {ch_semanal}")
    return ch_semanal * semanas


def ch_lecionada(
    cal: Calendario,
    distribuicao: dict[str, int],
    cenario: str = "A",
    bimestres: int | Iterable[int] | None = None,
    curso: str | None = None,
    sabado_reproduz: str | None = None,
) -> int:
    """Calcula a carga horária efetivamente lecionada a partir da distribuição semanal (C4).

    Args:
        cal: Instância de Calendario.
        distribuicao: Dicionário mapeando dia útil para quantidade de aulas (ex.: {"SEG": 2}).
        cenario: Cenário de apuração ("A" ou "REAL").
        bimestres: Bimestre(s) a apurar (None para o ano letivo completo).
        curso: Nome do curso (obrigatório para cenario="REAL").
        sabado_reproduz: Dia útil que o sábado letivo reproduz (obrigatório para cenario="REAL").

    Returns:
        Total de horas-aula lecionadas no período.
    """
    dias = dias_letivos(
        cal,
        cenario=cenario,
        bimestres=bimestres,
        curso=curso,
        sabado_reproduz=sabado_reproduz,
    )
    total_ch = 0
    for dia_raw, aulas in distribuicao.items():
        if aulas < 0:
            raise ValueError(f"Quantidade de aulas não pode ser negativa: {dia_raw}={aulas}")
        if aulas == 0:
            continue
        dia = dia_raw.upper().strip()
        if dia not in dias:
            raise ValueError(
                f"Dia da semana inválido na distribuição: '{dia_raw}'. Use um dos dias úteis: {DIAS_UTEIS}"
            )
        total_ch += aulas * dias[dia]
    return total_ch


def _gerar_arranjos(ch_semanal: int) -> list[dict[str, int]]:
    """Gera arranjos semanais de aulas segundo BLOCOS_POR_CH em ordem canônica (C4)."""
    if ch_semanal not in BLOCOS_POR_CH:
        raise ValueError(
            f"Carga horária semanal não suportada: {ch_semanal}. Valores suportados: {sorted(BLOCOS_POR_CH.keys())}"
        )
    blocos = BLOCOS_POR_CH[ch_semanal]
    if blocos == (1,):
        return [{d: 1} for d in DIAS_UTEIS]
    elif blocos == (2,):
        return [{d: 2} for d in DIAS_UTEIS]
    elif blocos == (2, 1):
        arranjos: list[dict[str, int]] = []
        for d1 in DIAS_UTEIS:
            for d2 in DIAS_UTEIS:
                if d1 != d2:
                    arranjos.append({d1: 2, d2: 1})
        return arranjos
    elif blocos == (2, 2):
        return [{d1: 2, d2: 2} for d1, d2 in itertools.combinations(DIAS_UTEIS, 2)]
    raise ValueError(f"Configuração de blocos não implementada para CH {ch_semanal}")


def faixa_ch(
    cal: Calendario,
    ch_semanal: int,
    cenario: str = "A",
    bimestres: int | Iterable[int] | None = None,
    curso: str | None = None,
) -> tuple[int, dict[str, int], int, dict[str, int]]:
    """Determina a faixa de variação (mínimo e máximo) de CH efetiva para uma carga semanal (C4).

    Avalia as combinações possíveis de distribuição de aulas na semana segundo
    BLOCOS_POR_CH e identifica os arranjos que minimizam e maximizam as horas-aula.

    Args:
        cal: Instância de Calendario.
        ch_semanal: Carga horária semanal (1, 2, 3 ou 4).
        cenario: Cenário de apuração ("A" ou "REAL").
        bimestres: Bimestre(s) a apurar (None para o ano todo).
        curso: Nome do curso (obrigatório se cenario="REAL").

    Returns:
        Tupla (min_ch, min_distribuicao, max_ch, max_distribuicao).

    Raises:
        ValueError: Caso ch_semanal não seja suportada ou parâmetros do cenário sejam inválidos.
    """
    if not isinstance(cenario, str) or cenario.upper() not in CENARIOS:
        raise ValueError(f"Cenário inválido: '{cenario}'. Use um dos cenários válidos: {CENARIOS}")
    cenario_norm = cenario.upper()

    if cenario_norm == "REAL" and not curso:
        raise ValueError("Para o cenário 'REAL', o parâmetro 'curso' é obrigatório.")

    arranjos = _gerar_arranjos(ch_semanal)

    min_ch: int | None = None
    min_dist: dict[str, int] | None = None
    max_ch: int | None = None
    max_dist: dict[str, int] | None = None

    if cenario_norm == "REAL":
        sabs = sabados_do_responsavel(cal, responsavel=curso, bimestres=bimestres) if curso else []
        reproducoes = DIAS_UTEIS if sabs else (None,)
    else:
        reproducoes = (None,)

    for dist in arranjos:
        for sab_rep in reproducoes:
            if cenario_norm == "REAL" and sab_rep is not None:
                val = ch_lecionada(
                    cal,
                    dist,
                    cenario="REAL",
                    bimestres=bimestres,
                    curso=curso,
                    sabado_reproduz=sab_rep,
                )
            else:
                val = ch_lecionada(
                    cal,
                    dist,
                    cenario="A",
                    bimestres=bimestres,
                )

            if min_ch is None or val < min_ch:
                min_ch = val
                min_dist = dist
            if max_ch is None or val > max_ch:
                max_ch = val
                max_dist = dist

    assert min_ch is not None and min_dist is not None and max_ch is not None and max_dist is not None
    return min_ch, min_dist, max_ch, max_dist


