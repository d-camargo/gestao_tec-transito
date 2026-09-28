"""Módulo de integração e consolidação dos cursos do DET (Estradas e Trânsito).

Decisão D1 — Princípios de integração dos cursos do DET:
    - União, não soma: O total de estudantes do Departamento de Engenharia de Transportes
      (DET) é obtido pela união dos estudantes únicos de Estradas e Trânsito, e não pela
      simples soma aritmética dos mapas brutos (evitando dupla contagem de estudantes
      compartilhados entre os cursos).
    - Fonte autoritativa do app: O processamento utiliza diretamente a função autoritativa
      do app (core.manipulacao.processar_multiplos_bimestres_transito_estradas), garantindo
      que todas as regras de negócio de herança de disciplinas e saneamento sejam respeitadas.
    - Interseção 0: A função do app transfere as notas de Ensino Médio do mapa de Estradas
      para Trânsito e expurga do conjunto de Estradas as matrículas já constantes em
      Trânsito, garantindo que os conjuntos finais tenham interseção nula (intersecao = 0).

Decisão D2 — Minimização de dados e conformidade LGPD:
    - O módulo expõe resumos estatísticos agregados (totais, contagens por curso e série)
      sem incluir nomes, matrículas ou outros dados pessoais discentes em representações
      textuais (str() ou repr() de ResumoDET).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence

# Assegura que o diretório sandbox/dae e a raiz do repositório estejam no sys.path
_DIR_DAE = Path(__file__).resolve().parent
_DIR_RAIZ = _DIR_DAE.parent.parent
for _p in (_DIR_DAE, _DIR_RAIZ):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd

try:
    from .calendario import carregar_calendario
    from .carregar import _normalizar_rotulo
    from .ch_efetiva import (
        CAMINHO_CH_EFETIVA_PADRAO,
        carregar_ch_efetiva,
        casar_disciplinas,
        ch_da_turma,
        normalizar_disciplina,
    )
    from .frequencia import resumo_frequencia_por_disciplina
except ImportError:
    from calendario import carregar_calendario
    from carregar import _normalizar_rotulo
    from ch_efetiva import (
        CAMINHO_CH_EFETIVA_PADRAO,
        carregar_ch_efetiva,
        casar_disciplinas,
        ch_da_turma,
        normalizar_disciplina,
    )
    from frequencia import resumo_frequencia_por_disciplina

from core.disciplinas import detectar_serie
import core.manipulacao as manipulacao
from core.manipulacao import (
    ArquivoInvalidoError,
    _curso_amigavel,
    extrair_metadados,
    processar_multiplos_bimestres_transito_estradas,
)

CURSOS_DET: tuple[str, ...] = ("Estradas", "Trânsito")
ROTULO_DET: str = "Estradas + Trânsito (DET)"

PASTA_DADOS: Path = _DIR_DAE / "dados"
CAMINHO_ESTRADAS_PADRAO: Path = PASTA_DADOS / "Estradas_2025-2026.xls"
CAMINHO_TRANSITO_PADRAO: Path = PASTA_DADOS / "Transito_2025-2026.xls"


def classificar_mapas(arquivos: Any, *outros: Any) -> dict[str, list[Any]]:
    """Agrupa arquivos de mapa de turma pelo nome amigável do curso (curso_amigavel).

    Lê os metadados do cabeçalho de cada mapa (.xls ou DataFrame sintético)
    e organiza em um dicionário mapeando cada curso_amigavel para a lista
    de arquivos correspondentes.

    Args:
        arquivos: Lista/iterável de caminhos ou DataFrames, ou primeiro arquivo se passado posicionalmente.
        *outros: Arquivos adicionais passados como argumentos posicionais.

    Returns:
        dict[str, list]: Dicionário mapeando curso_amigavel (ex.: 'Estradas', 'Trânsito')
                         para a lista de mapas correspondentes.
    """
    if outros:
        lista_arquivos = [arquivos, *outros]
    elif isinstance(arquivos, (list, tuple, set)):
        lista_arquivos = list(arquivos)
    else:
        lista_arquivos = [arquivos]

    resultado: dict[str, list[Any]] = {}
    for arq in lista_arquivos:
        if isinstance(arq, pd.DataFrame):
            df_bruto = arq
        else:
            if hasattr(arq, "seek"):
                arq.seek(0)
            df_bruto = manipulacao._ler_xls_bruto(arq)
            if hasattr(arq, "seek"):
                arq.seek(0)

        meta = extrair_metadados(df_bruto)
        curso_amigavel = (
            meta.get("curso_amigavel")
            or _curso_amigavel(meta.get("curso"))
            or meta.get("curso")
            or ""
        )
        if curso_amigavel:
            resultado.setdefault(curso_amigavel, []).append(arq)
    return resultado


def eh_det(alvo: Any, *outros: Any) -> bool:
    """Verifica se o conjunto de mapas ou cursos corresponde estritamente ao DET.

    Retorna True APENAS quando os dois cursos do departamento (Estradas e Trânsito)
    estão presentes simultaneamente, sem outros cursos e sem ausência de qualquer um deles.

    Args:
        alvo: Dicionário classificado, lista de arquivos/DataFrames, lista de nomes de cursos,
              ou primeiro curso.
        *outros: Nomes de cursos ou mapas adicionais se passados posicionalmente.

    Returns:
        bool: True se contém exatamente Estradas e Trânsito, False caso contrário.
    """
    if outros:
        itens = [alvo, *outros]
    elif isinstance(alvo, (list, tuple, set)):
        itens = list(alvo)
    elif isinstance(alvo, dict):
        cursos = {k for k, v in alvo.items() if v}
        return cursos == set(CURSOS_DET)
    else:
        return False

    if not itens:
        return False

    # Se todos os itens forem strings (nomes de cursos)
    if all(isinstance(x, str) for x in itens):
        cursos_detectados = set()
        for s in itens:
            c = _curso_amigavel(s) or s.strip()
            if c:
                cursos_detectados.add(c)
        return cursos_detectados == set(CURSOS_DET)

    # Caso contrário, trata como mapas e classifica
    try:
        classificados = classificar_mapas(itens)
        cursos_com_mapas = {k for k, v in classificados.items() if v}
        return cursos_com_mapas == set(CURSOS_DET)
    except Exception:
        return False


class conjuntos_det(tuple):
    """Encapsula os conjuntos de bimestres de Trânsito e Estradas do DET.

    Permite desempacotamento como tupla de 2 elementos (conjuntos_tt, conjuntos_est),
    acesso por propriedades (.transito, .estradas) e indexação por nome de curso.
    """

    def __new__(cls, transito: Any, estradas: Any = None):
        if isinstance(transito, conjuntos_det):
            return transito
        if estradas is None:
            raise TypeError("conjuntos_det exige os conjuntos de Trânsito e de Estradas.")
        return super().__new__(cls, (list(transito), list(estradas)))

    @property
    def transito(self) -> list:
        return self[0]

    @property
    def estradas(self) -> list:
        return self[1]

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            chave = item.strip().lower()
            if "transito" in chave or "trânsito" in chave:
                return self[0]
            if "estradas" in chave:
                return self[1]
            raise KeyError(f"Curso desconhecido no DET: {item}")
        return super().__getitem__(item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except (KeyError, IndexError):
            return default


def carregar_det(
    mapas_ou_lado1: Any,
    lado2: Any = None,
) -> conjuntos_det:
    """Carrega e processa conjuntamente os mapas de turma do DET (Trânsito e Estradas).

    Utiliza a função autoritativa do app (processar_multiplos_bimestres_transito_estradas).
    Exige obrigatoriamente que ambos os cursos tenham mapas fornecidos.

    Args:
        mapas_ou_lado1: Lista de arquivos de ambos os cursos, dicionário classificado,
                        ou lista de arquivos do primeiro curso.
        lado2: Opcional. Lista de arquivos do segundo curso (caso mapas_ou_lado1 seja apenas um dos lados).

    Returns:
        conjuntos_det: Estrutura contendo (conjuntos_tt, conjuntos_est).

    Raises:
        ValueError: Se algum dos dois lados (Estradas ou Trânsito) estiver vazio.
    """
    if lado2 is not None:
        l1 = list(mapas_ou_lado1) if isinstance(mapas_ou_lado1, (list, tuple, set)) else ([mapas_ou_lado1] if mapas_ou_lado1 else [])
        l2 = list(lado2) if isinstance(lado2, (list, tuple, set)) else ([lado2] if lado2 else [])

        if not l1 or not l2:
            raise ValueError(
                "O carregamento do DET requer mapas de ambos os cursos (Estradas e Trânsito). "
                f"Lado 1 possui {len(l1)} arquivos e Lado 2 possui {len(l2)} arquivos."
            )

        classif1 = classificar_mapas(l1)
        classif2 = classificar_mapas(l2)
        if "Trânsito" in classif1 and "Estradas" in classif2:
            mapas_tt, mapas_est = l1, l2
        elif "Estradas" in classif1 and "Trânsito" in classif2:
            mapas_est, mapas_tt = l1, l2
        elif "Trânsito" in classif1 and "Trânsito" in classif2:
            raise ValueError("Ambos os lados fornecidos correspondem a Trânsito. Falta Estradas.")
        elif "Estradas" in classif1 and "Estradas" in classif2:
            raise ValueError("Ambos os lados fornecidos correspondem a Estradas. Falta Trânsito.")
        else:
            mapas_tt, mapas_est = l1, l2
    elif isinstance(mapas_ou_lado1, dict):
        mapas_tt = list(mapas_ou_lado1.get("Trânsito", []))
        mapas_est = list(mapas_ou_lado1.get("Estradas", []))
    elif isinstance(mapas_ou_lado1, (list, tuple, set)):
        classif = classificar_mapas(mapas_ou_lado1)
        mapas_tt = list(classif.get("Trânsito", []))
        mapas_est = list(classif.get("Estradas", []))
    else:
        raise ValueError("Argumento inválido fornecido para carregar_det.")

    if not mapas_tt or not mapas_est:
        raise ValueError(
            "O carregamento do DET requer mapas de ambos os cursos (Estradas e Trânsito). "
            f"Recebido: Trânsito={len(mapas_tt)}, Estradas={len(mapas_est)}."
        )

    conjuntos_tt, conjuntos_est = processar_multiplos_bimestres_transito_estradas(
        mapas_tt, mapas_est
    )
    return conjuntos_det(conjuntos_tt, conjuntos_est)


class ResumoDET(dict):
    """Resumo estatístico agregado do DET (Estradas + Trânsito).

    Minimização de dados / LGPD: contém apenas métricas agregadas (totais, contagens),
    sem nenhuma matrícula ou nome de estudante.
    """

    def __init__(
        self,
        alunos_total: int,
        alunos_por_curso: dict[str, int],
        intersecao: int,
        bimestres: list[int],
        serie: int | None,
        disciplinas_por_curso: dict[str, int],
        faltas_por_curso: dict[str, int],
        faltas_total: int,
    ) -> None:
        super().__init__(
            alunos_total=alunos_total,
            alunos_por_curso=alunos_por_curso,
            intersecao=intersecao,
            bimestres=bimestres,
            serie=serie,
            disciplinas_por_curso=disciplinas_por_curso,
            faltas_por_curso=faltas_por_curso,
            faltas_total=faltas_total,
        )
        self.alunos_total = alunos_total
        self.alunos_por_curso = alunos_por_curso
        self.intersecao = intersecao
        self.bimestres = bimestres
        self.serie = serie
        self.disciplinas_por_curso = disciplinas_por_curso
        self.faltas_por_curso = faltas_por_curso
        self.faltas_total = faltas_total

    def __str__(self) -> str:
        est_count = self.alunos_por_curso.get("Estradas", 0)
        tt_count = self.alunos_por_curso.get("Trânsito", 0)
        return (
            f"Resumo DET: {self.alunos_total} alunos "
            f"(Estradas: {est_count}, Trânsito: {tt_count}), "
            f"interseção: {self.intersecao}, bimestres: {self.bimestres}, "
            f"série: {self.serie}, disciplinas: {self.disciplinas_por_curso}, "
            f"faltas: {self.faltas_por_curso} (total {self.faltas_total})"
        )

    def __repr__(self) -> str:
        return (
            f"ResumoDET(alunos_total={self.alunos_total}, "
            f"alunos_por_curso={self.alunos_por_curso}, "
            f"intersecao={self.intersecao}, bimestres={self.bimestres}, "
            f"serie={self.serie}, disciplinas_por_curso={self.disciplinas_por_curso}, "
            f"faltas_por_curso={self.faltas_por_curso}, faltas_total={self.faltas_total})"
        )


def resumo_det(dados_ou_tt: Any, est: Any = None) -> ResumoDET:
    """Gera o resumo estatístico consolidado do DET (D1/D2).

    Calcula métricas agregadas preservando a privacidade (LGPD):
        - alunos_total: união de estudantes únicos de Trânsito e Estradas (D1)
        - alunos_por_curso: contagem de estudantes por curso
        - intersecao: alunos em comum entre os conjuntos finais (garantida 0 por D1)
        - bimestres: lista de bimestres cobertos
        - serie: série detectada
        - disciplinas_por_curso: contagem de disciplinas distintas de cada curso
        - faltas_por_curso / faltas_total: soma das faltas lançadas nos mapas de cada lado
    """
    if est is not None:
        conj_tt, conj_est = dados_ou_tt, est
    elif isinstance(dados_ou_tt, (conjuntos_det, tuple)) and len(dados_ou_tt) == 2 and isinstance(dados_ou_tt[0], list):
        conj_tt, conj_est = dados_ou_tt[0], dados_ou_tt[1]
    elif isinstance(dados_ou_tt, dict) and ("Trânsito" in dados_ou_tt or "Estradas" in dados_ou_tt):
        conj_tt = dados_ou_tt.get("Trânsito", [])
        conj_est = dados_ou_tt.get("Estradas", [])
    else:
        conj_tt, conj_est = carregar_det(dados_ou_tt)

    matriculas_tt = set()
    for df_notas, _, _, _ in conj_tt:
        if "matricula" in df_notas.columns:
            matriculas_tt.update(df_notas["matricula"].dropna().astype(str))

    matriculas_est = set()
    for df_notas, _, _, _ in conj_est:
        if "matricula" in df_notas.columns:
            matriculas_est.update(df_notas["matricula"].dropna().astype(str))

    intersecao = len(matriculas_tt & matriculas_est)
    alunos_total = len(matriculas_tt | matriculas_est)
    alunos_por_curso = {
        "Estradas": len(matriculas_est),
        "Trânsito": len(matriculas_tt),
    }

    bimestres_set = set()
    for _, _, _, meta in conj_tt:
        if meta and meta.get("bimestre_num"):
            bimestres_set.add(int(meta["bimestre_num"]))
    for _, _, _, meta in conj_est:
        if meta and meta.get("bimestre_num"):
            bimestres_set.add(int(meta["bimestre_num"]))
    bimestres = sorted(list(bimestres_set))

    serie = None
    for _, _, _, meta in conj_tt + conj_est:
        if meta and meta.get("serie"):
            serie = int(meta["serie"])
            break

    disc_est_set = set()
    for _, _, disc, _ in conj_est:
        if isinstance(disc, dict):
            disc_est_set.update(disc.keys())
    disc_tt_set = set()
    for _, _, disc, _ in conj_tt:
        if isinstance(disc, dict):
            disc_tt_set.update(disc.keys())

    disciplinas = {
        "Estradas": len(disc_est_set),
        "Trânsito": len(disc_tt_set),
    }

    def _total_faltas(conjuntos: list) -> int:
        total = 0
        for _, df_faltas, _, _ in conjuntos:
            for col in df_faltas.columns:
                if col in ("matricula", "nome"):
                    continue
                total += int(pd.to_numeric(df_faltas[col], errors="coerce").fillna(0).sum())
        return total

    faltas_por_curso = {
        "Estradas": _total_faltas(conj_est),
        "Trânsito": _total_faltas(conj_tt),
    }

    return ResumoDET(
        alunos_total=alunos_total,
        alunos_por_curso=alunos_por_curso,
        intersecao=intersecao,
        bimestres=bimestres,
        serie=serie,
        disciplinas_por_curso=disciplinas,
        faltas_por_curso=faltas_por_curso,
        faltas_total=sum(faltas_por_curso.values()),
    )


class ResultadoFrequenciaDET(tuple):
    """Resultado consolidado da apuração de frequência do DET (D1).

    Tupla contendo:
        - [0] / .df: pd.DataFrame consolidado com as disciplinas casadas (núcleo comum e técnicas).
        - [1] / .sem_horario: dict[str, list[str]] com as disciplinas sem horário por curso.
    """

    def __new__(cls, df: pd.DataFrame, sem_horario: dict[str, list[str]]):
        return super().__new__(cls, (df, sem_horario))

    @property
    def df(self) -> pd.DataFrame:
        return self[0]

    @property
    def sem_horario(self) -> dict[str, list[str]]:
        return self[1]


def resumo_frequencia_det(
    det: Any,
    df_ch: pd.DataFrame | str | Path,
    bimestre: int = 1,
    calendario: Any = None,
) -> ResultadoFrequenciaDET:
    """Gera o resumo de frequência integrado dos cursos do DET (D1).

    Por lado (Estradas e Trânsito):
        - Filtra a CH efetiva da turma via ch_da_turma;
        - Casa disciplinas da legenda com a grade horária via casar_disciplinas;
        - Apura o resumo de frequência por disciplina via resumo_frequencia_por_disciplina.

    Consolidação:
        - Junta em um único DataFrame com a coluna ``escopo`` ∈ {"Núcleo comum (EST/TT)", "Estradas", "Trânsito"};
        - Núcleo comum (disciplinas casadas em linhas 'EST/TT-*'):
            - Aparece uma única vez;
            - Soma n_alunos e n_abaixo_75 dos dois lados;
            - Levanta ValueError se a CH do bimestre divergir entre os lados;
        - Técnicas casadas: escopo recebe o nome do curso ('Estradas' ou 'Trânsito');
        - Sem horário: devolve lista de nomes de disciplinas sem horário agrupadas por curso;
        - Minimização de dados (LGPD): apenas métricas e chaves agregadas (sem matrículas ou nomes).

    Args:
        det: Estrutura conjuntos_det, tupla (conjuntos_tt, conjuntos_est), dicionário ou arquivos.
        df_ch: DataFrame de CH efetiva lecionada ou caminho do arquivo .xlsx.
        bimestre: Número do bimestre (padrão: 1).
        calendario: Opcional. Instância de Calendario para cálculo de limites de faltas.

    Returns:
        ResultadoFrequenciaDET: Tupla (df, sem_horario) com atributos .df e .sem_horario.

    Raises:
        ValueError: Se a CH do bimestre de disciplina do núcleo comum divergir entre Estradas e Trânsito.
    """
    if not isinstance(df_ch, pd.DataFrame):
        df_ch = carregar_ch_efetiva(df_ch)

    if calendario is None:
        calendario = carregar_calendario()

    if isinstance(det, (conjuntos_det, tuple)) and len(det) == 2 and isinstance(det[0], list):
        conj_tt, conj_est = det[0], det[1]
    elif isinstance(det, dict) and ("Trânsito" in det or "Estradas" in det):
        conj_tt = det.get("Trânsito", [])
        conj_est = det.get("Estradas", [])
    else:
        cd = carregar_det(det)
        conj_tt, conj_est = cd[0], cd[1]

    # Garante correspondência exata caso os lados venham em ordem invertida
    if conj_tt and conj_tt[0][3] and conj_tt[0][3].get("curso_amigavel") == "Estradas":
        conj_tt, conj_est = conj_est, conj_tt

    def _obter_bimestre(conjuntos: list, curso_nome: str):
        for tpl in conjuntos:
            meta = tpl[3] if len(tpl) > 3 else {}
            if meta and meta.get("bimestre_num") == int(bimestre):
                return tpl
        if len(conjuntos) == 1:
            meta = conjuntos[0][3] if len(conjuntos[0]) > 3 else {}
            if meta.get("bimestre_num") in (None, int(bimestre)):
                return conjuntos[0]
        raise ValueError(
            f"Bimestre {bimestre} não encontrado no conjunto de {curso_nome}."
        )

    df_notas_est, df_faltas_est, leg_est, meta_est = _obter_bimestre(conj_est, "Estradas")
    df_notas_tt, df_faltas_tt, leg_tt, meta_tt = _obter_bimestre(conj_tt, "Trânsito")

    serie_est = meta_est.get("serie") or detectar_serie(leg_est) or 2
    serie_tt = meta_tt.get("serie") or detectar_serie(leg_tt) or 2
    turma_est = meta_est.get("turma") or "A"
    turma_tt = meta_tt.get("turma") or "A"

    df_turma_est = ch_da_turma(df_ch, "Estradas", serie_est, turma_est)
    df_turma_tt = ch_da_turma(df_ch, "Trânsito", serie_tt, turma_tt)

    casadas_est, sem_linha_est = casar_disciplinas(leg_est, df_turma_est)
    casadas_tt, sem_linha_tt = casar_disciplinas(leg_tt, df_turma_tt)

    res_est = resumo_frequencia_por_disciplina(
        df_faltas_est, leg_est, casadas_est, bimestre=bimestre, cal=calendario
    )
    res_tt = resumo_frequencia_por_disciplina(
        df_faltas_tt, leg_tt, casadas_tt, bimestre=bimestre, cal=calendario
    )

    mask_nc_est = (
        df_turma_est["turma"].astype(str).str.startswith("EST/TT")
        if "turma" in df_turma_est.columns
        else pd.Series(False, index=df_turma_est.index)
    )
    nc_norm_est = (
        set(df_turma_est.loc[mask_nc_est, "disciplina_norm"])
        if "disciplina_norm" in df_turma_est.columns
        else set(df_turma_est.loc[mask_nc_est, "disciplina"].apply(normalizar_disciplina))
    )

    mask_nc_tt = (
        df_turma_tt["turma"].astype(str).str.startswith("EST/TT")
        if "turma" in df_turma_tt.columns
        else pd.Series(False, index=df_turma_tt.index)
    )
    nc_norm_tt = (
        set(df_turma_tt.loc[mask_nc_tt, "disciplina_norm"])
        if "disciplina_norm" in df_turma_tt.columns
        else set(df_turma_tt.loc[mask_nc_tt, "disciplina"].apply(normalizar_disciplina))
    )
    nc_norm_todos = nc_norm_est | nc_norm_tt

    itens_est = {
        normalizar_disciplina(r["disciplina"]): r
        for r in res_est
        if r.get("fonte") == "planilha"
    }
    itens_tt = {
        normalizar_disciplina(r["disciplina"]): r
        for r in res_tt
        if r.get("fonte") == "planilha"
    }

    chaves_nc: list[str] = []
    for r in res_est:
        norm = normalizar_disciplina(r["disciplina"])
        if r.get("fonte") == "planilha" and norm in nc_norm_todos and norm not in chaves_nc:
            chaves_nc.append(norm)
    for r in res_tt:
        norm = normalizar_disciplina(r["disciplina"])
        if r.get("fonte") == "planilha" and norm in nc_norm_todos and norm not in chaves_nc:
            chaves_nc.append(norm)

    linhas_nc = []
    for norm in chaves_nc:
        r_est = itens_est.get(norm)
        r_tt = itens_tt.get(norm)
        base = r_est if r_est is not None else r_tt

        if r_est is not None and r_tt is not None:
            ch_est = r_est.get("ch_bim")
            ch_tt = r_tt.get("ch_bim")
            if ch_est != ch_tt:
                raise ValueError(
                    f"Carga horária do {bimestre}º bimestre diverge para disciplina de núcleo comum '{base['disciplina']}': "
                    f"Estradas={ch_est} vs Trânsito={ch_tt}."
                )

        n_alunos_total = (r_est["n_alunos"] if r_est is not None else 0) + (r_tt["n_alunos"] if r_tt is not None else 0)

        tem_abaixo = False
        abaixo_total = 0
        if r_est is not None and r_est.get("n_abaixo_75") is not None:
            tem_abaixo = True
            abaixo_total += int(r_est["n_abaixo_75"])
        if r_tt is not None and r_tt.get("n_abaixo_75") is not None:
            tem_abaixo = True
            abaixo_total += int(r_tt["n_abaixo_75"])

        linhas_nc.append({
            "disciplina": base["disciplina"],
            "escopo": "Núcleo comum (EST/TT)",
            "aulas_sem": base.get("aulas_sem"),
            "ch_bim": base.get("ch_bim"),
            "ch_efetiva_ano": base.get("ch_efetiva_ano"),
            "ch_nominal": base.get("ch_nominal"),
            "%_nominal": base.get("%_nominal"),
            "limite_faltas_bim": base.get("limite_faltas_bim"),
            "n_alunos": n_alunos_total,
            "fonte": "planilha",
            "n_abaixo_75": abaixo_total if tem_abaixo else None,
        })

    linhas_est = []
    for r in res_est:
        if r.get("fonte") == "planilha":
            norm = normalizar_disciplina(r["disciplina"])
            if norm not in nc_norm_todos:
                linhas_est.append({
                    "disciplina": r["disciplina"],
                    "escopo": "Estradas",
                    "aulas_sem": r.get("aulas_sem"),
                    "ch_bim": r.get("ch_bim"),
                    "ch_efetiva_ano": r.get("ch_efetiva_ano"),
                    "ch_nominal": r.get("ch_nominal"),
                    "%_nominal": r.get("%_nominal"),
                    "limite_faltas_bim": r.get("limite_faltas_bim"),
                    "n_alunos": r.get("n_alunos"),
                    "fonte": "planilha",
                    "n_abaixo_75": r.get("n_abaixo_75"),
                })

    linhas_tt = []
    for r in res_tt:
        if r.get("fonte") == "planilha":
            norm = normalizar_disciplina(r["disciplina"])
            if norm not in nc_norm_todos:
                linhas_tt.append({
                    "disciplina": r["disciplina"],
                    "escopo": "Trânsito",
                    "aulas_sem": r.get("aulas_sem"),
                    "ch_bim": r.get("ch_bim"),
                    "ch_efetiva_ano": r.get("ch_efetiva_ano"),
                    "ch_nominal": r.get("ch_nominal"),
                    "%_nominal": r.get("%_nominal"),
                    "limite_faltas_bim": r.get("limite_faltas_bim"),
                    "n_alunos": r.get("n_alunos"),
                    "fonte": "planilha",
                    "n_abaixo_75": r.get("n_abaixo_75"),
                })

    colunas = [
        "disciplina",
        "escopo",
        "aulas_sem",
        "ch_bim",
        "ch_efetiva_ano",
        "ch_nominal",
        "%_nominal",
        "limite_faltas_bim",
        "n_alunos",
        "fonte",
        "n_abaixo_75",
    ]
    df_resultado = pd.DataFrame(linhas_nc + linhas_est + linhas_tt, columns=colunas)
    sem_horario = {
        "Estradas": list(sem_linha_est),
        "Trânsito": list(sem_linha_tt),
    }
    df_resultado.attrs["sem_horario"] = sem_horario
    return ResultadoFrequenciaDET(df_resultado, sem_horario)
