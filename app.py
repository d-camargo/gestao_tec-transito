"""Gestão Acadêmica EPTNM — CEFET-MG.

App web (Streamlit) para coordenadores dos cursos técnicos de nível médio (EPTNM).
O coordenador informa seu e-mail institucional, faz o upload do Mapa de Turma
(.xls) e recebe o relatório de acompanhamento acadêmico (PDF) por e-mail.

Caso especial: o Curso integrado **Trânsito + Estradas (1ª série)** exige 2
arquivos (mapa de Trânsito + mapa de Estradas) e produz **2 PDFs**, um para
cada curso.
"""
import os
import re
import subprocess

import streamlit as st

# Versão "humana" do app. Incremente ao publicar mudanças relevantes; o commit
# do Git (mostrado ao lado) é o identificador exato do que está no ar.
APP_VERSION = "1.1.0"

from core import relatorios
from core.email_sender import DOMINIO_INSTITUCIONAL, email_valido, enviar_relatorio
from core.manipulacao import (
    ArquivoInvalidoError,
    processar_multiplos_bimestres,
    processar_multiplos_bimestres_transito_estradas,
)
from core.usage_tracker import registrar_uso

st.set_page_config(
    page_title="Gestão Acadêmica EPTNM — CEFET-MG",
    page_icon="🎓",
    layout="centered",
)

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo_cefet.png")


def _secret(nome, default=""):
    """Lê um segredo de st.secrets e, em fallback, da variável de ambiente."""
    try:
        if nome in st.secrets:
            return st.secrets[nome]
    except Exception:
        pass
    return os.environ.get(nome, default)


def _slug(texto):
    return re.sub(r'\W+', '_', (texto or 'curso').strip().lower()).strip('_') or 'curso'


@st.cache_data(show_spinner=False)
def _build_info():
    """Identifica o build em execução (commit + data) para conferir o deploy.

    Lê do Git no diretório do app — funciona no Streamlit Cloud, que publica a
    partir de um clone do repositório. Retorna ('', '') se o Git não estiver
    disponível (ex.: execução fora de um repositório)."""
    base = os.path.dirname(os.path.abspath(__file__))

    def _git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=base,
                stderr=subprocess.DEVNULL, text=True, timeout=3,
            ).strip()
        except Exception:
            return ""

    commit = _git("rev-parse", "--short", "HEAD")
    data_commit = _git("show", "-s", "--format=%cd", "--date=short", "HEAD")
    return commit, data_commit


def _versao_label():
    """Monta o rótulo de versão exibido na página."""
    commit, data_commit = _build_info()
    partes = [f"Versão {APP_VERSION}"]
    if commit:
        partes.append(f"commit `{commit}`")
    if data_commit:
        partes.append(data_commit)
    return " · ".join(partes)


# --------------------------------
# Cabeçalho
# --------------------------------
st.title("🎓 Gestão Acadêmica EPTNM — CEFET-MG")
st.caption("Educação Profissional Técnica de Nível Médio · Centro Federal de "
           "Educação Tecnológica de Minas Gerais")
st.caption(f"🏷️ {_versao_label()}")
st.markdown(
    "Informe seu e-mail institucional e envie o **Mapa de Turma** (`.xls`). "
    "O relatório de acompanhamento acadêmico será gerado e **enviado para o seu "
    f"e-mail `@{DOMINIO_INSTITUCIONAL}`**."
)

# --------------------------------
# Barra lateral — configurações do curso
# --------------------------------
with st.sidebar:
    st.header("⚙️ Configurações")
    eh_transito_estradas = st.checkbox(
        "Curso integrado Trânsito + Estradas (1ª série)",
        value=False,
        help="Marque se você é coordenador(a) da 1ª série de Trânsito ou "
             "Estradas. Esse modo requer os dois mapas e gera dois relatórios "
             "(um para cada curso) no mesmo e-mail.",
    )
    usar_ia = st.toggle(
        "Incluir análise por IA (OpenAI)",
        value=False,
        help="Adiciona um comentário analítico ao relatório. Requer a chave "
             "OPENAI_API_KEY configurada.",
    )
    if usar_ia:
        st.caption("🔒 Os **nomes e as notas individuais dos alunos não são "
                   "enviados à IA** — apenas estatísticas agregadas por disciplina.")
    st.divider()
    if eh_transito_estradas:
        st.info("Modo **Trânsito + Estradas**: envie o mapa de Trânsito **e** o "
                "mapa de Estradas (o de Estradas traz as notas do ensino médio). "
                "Você receberá dois relatórios.")
    else:
        st.info("Envie **1 arquivo**: o mapa de turma completo do seu curso. "
                "Funciona para qualquer curso técnico do EPTNM.")

    st.divider()
    st.subheader("🔒 Privacidade e LGPD")
    st.warning(
        "Os **nomes dos alunos não são entregues à IA**. Quando a análise por IA "
        "está ativa, apenas **estatísticas agregadas por disciplina** (médias, "
        "medianas, desvios, mín./máx.) são enviadas à OpenAI — **sem nomes e sem "
        "notas individuais**."
    )
    with st.expander("Saiba mais sobre o tratamento dos dados"):
        st.markdown(
            "- **Nomes e notas** são *dados pessoais* protegidos pela **LGPD**. "
            "Notas escolares **não** são *dados sensíveis* no sentido legal "
            "(art. 5º, II), mas seguem protegidas como dados pessoais.\n"
            "- Esses dados ficam apenas no **processamento interno** e no **PDF** "
            "enviado à sua caixa institucional `@cefetmg.br`.\n"
            "- O mapa de turma e o PDF são processados **em memória** e **não são "
            "armazenados** no servidor.\n"
            "- Pela política da **API da OpenAI**, os dados enviados **não são "
            "usados para treinar** os modelos — ainda assim, por isso, **nenhum "
            "nome ou nota individual** é compartilhado."
        )

    st.divider()
    st.caption(f"🏷️ {_versao_label()}")

# --------------------------------
# Formulário
# --------------------------------
email = st.text_input(
    f"📧 Seu e-mail institucional (@{DOMINIO_INSTITUCIONAL})",
    placeholder=f"seu.nome@{DOMINIO_INSTITUCIONAL}",
)

if eh_transito_estradas:
    c1, c2 = st.columns(2)
    with c1:
        arquivo_transito = st.file_uploader(
            "📄 Mapa — Trânsito (.xls)", type=["xls"], key="up_transito", accept_multiple_files=True)
    with c2:
        arquivo_estradas = st.file_uploader(
            "📄 Mapa — Estradas (.xls)", type=["xls"], key="up_estradas", accept_multiple_files=True)
    arquivos_ok = bool(arquivo_transito and arquivo_estradas)
else:
    arquivo_unico = st.file_uploader(
        "📄 Mapa de Turma (.xls)", type=["xls"], key="up_unico", accept_multiple_files=True)
    arquivos_ok = bool(arquivo_unico)

st.caption("ℹ️ O nome do curso e o bimestre são lidos automaticamente do cabeçalho "
           "do arquivo. **Envie o mapa de um único bimestre por vez.**")

enviar = st.button("📨 Gerar e enviar relatório por e-mail", type="primary")


# --------------------------------
# Processamento
# --------------------------------
def _fmt_bimestres(bimestres):
    """Formata a lista de bimestres enviados para nome de arquivo/uso (ex.: '1-3')."""
    validos = sorted({b for b in bimestres if b is not None})
    if not validos:
        return "X"
    if len(validos) == 1:
        return str(validos[0])
    if validos == list(range(validos[0], validos[-1] + 1)):
        return f"{validos[0]}-{validos[-1]}"
    return ",".join(str(b) for b in validos)


def _gerar_pdf_para_conjunto(conjunto, usar_ia, api_key, estatisticas_multibimestre=None, bimestres_grupo=None):
    """Gera (nome_arquivo, pdf_buffer, nome_curso) para um conjunto (df, df, disc, meta)."""
    df_notas, df_faltas, disciplinas_dict, metadados = conjunto
    nome_curso = metadados.get('curso_amigavel') or metadados.get('curso') or 'Curso'

    estat = relatorios.calcular_estatisticas(
        df_notas, disciplinas_dict, df_faltas=df_faltas, metadados=metadados)
    if usar_ia:
        estat['comentario_ia'] = relatorios.gerar_comentario_ia(
            estat, nome_curso, api_key)
    figuras = relatorios.gerar_todos_graficos(
        df_notas, nome_curso, disciplinas_dict, estat, df_faltas=df_faltas)
    logo = LOGO_PATH if os.path.exists(LOGO_PATH) else None
    pdf_buffer = relatorios.criar_relatorio_pdf(
        nome_curso, estat, figuras, logo_path=logo, estatisticas_multibimestre=estatisticas_multibimestre)

    bim = _fmt_bimestres(bimestres_grupo) if bimestres_grupo else (metadados.get('bimestre_num') or 'X')
    serie = metadados.get('serie')
    serie_tag = f"_{serie}aserie" if serie else ""
    nome_arquivo = f"relatorio_{_slug(nome_curso)}{serie_tag}_bim{bim}.pdf"
    return nome_arquivo, pdf_buffer, nome_curso


def processar_e_enviar():
    if not email_valido(email):
        st.error(f"Informe um e-mail válido terminado em `@{DOMINIO_INSTITUCIONAL}`. "
                 "Só professores do CEFET-MG podem usar este serviço.")
        return
    if not arquivos_ok:
        st.error("Envie o(s) arquivo(s) `.xls` necessário(s).")
        return

    remetente = _secret("GMAIL_USER")
    senha_app = _secret("GMAIL_APP_PASSWORD")
    if not remetente or not senha_app:
        st.error("O envio de e-mail ainda não foi configurado pelo administrador "
                 "(GMAIL_USER / GMAIL_APP_PASSWORD).")
        return

    api_key = _secret("OPENAI_API_KEY") if usar_ia else ""

    # 1. Processa dados
    try:
        with st.spinner("Processando o(s) mapa(s) de turma..."):
            if eh_transito_estradas:
                if len(arquivo_transito) > 4 or len(arquivo_estradas) > 4:
                    raise ArquivoInvalidoError("Envie no máximo 4 arquivos de cada curso (um por bimestre).")
                conjuntos_tt, conjuntos_est = processar_multiplos_bimestres_transito_estradas(
                    arquivo_transito, arquivo_estradas
                )
                conjuntos_tt_validos = [c for c in conjuntos_tt if not c[0].empty]
                conjuntos_est_validos = [c for c in conjuntos_est if not c[0].empty]

                grupos_processar = []
                if conjuntos_tt_validos:
                    grupos_processar.append(conjuntos_tt_validos)
                if conjuntos_est_validos:
                    grupos_processar.append(conjuntos_est_validos)
            else:
                if len(arquivo_unico) > 4:
                    raise ArquivoInvalidoError("Envie no máximo 4 arquivos por vez (um por bimestre).")
                conjuntos = processar_multiplos_bimestres(arquivo_unico)
                conjuntos_validos = [c for c in conjuntos if not c[0].empty]

                grupos_processar = []
                if conjuntos_validos:
                    grupos_processar.append(conjuntos_validos)
    except ArquivoInvalidoError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
        return

    if not grupos_processar:
        st.error("Nenhum aluno válido foi encontrado no arquivo. Verifique o mapa de turma.")
        return

    # 2. Gera PDFs
    try:
        with st.spinner("Gerando o(s) relatório(s)..."):
            anexos = []
            cursos = []
            bimestres_enviados = []
            for grupo in grupos_processar:
                # O relatório principal é gerado para o bimestre mais recente (o último do grupo ordenado)
                conjunto_recente = grupo[-1]
                bimestres_grupo = [c[3].get('bimestre_num') for c in grupo]
                bimestres_enviados.extend(bimestres_grupo)

                # Estatísticas multibimestres (calculadas usando todos os bimestres do grupo)
                estatisticas_multibimestre = relatorios.calcular_estatisticas_multibimestre(grupo)

                nome_arquivo, pdf_buffer, nome_curso = _gerar_pdf_para_conjunto(
                    conjunto_recente, usar_ia, api_key,
                    estatisticas_multibimestre=estatisticas_multibimestre,
                    bimestres_grupo=bimestres_grupo,
                )
                anexos.append((nome_arquivo, pdf_buffer))
                cursos.append(nome_curso)
    except Exception as e:
        st.error(f"Erro ao gerar o(s) relatório(s): {e}")
        return

    # 3. Envia por e-mail
    try:
        with st.spinner(f"Enviando para {email}..."):
            enviar_relatorio(
                destinatario=email.strip(),
                remetente=remetente,
                senha_app=senha_app,
                anexos=anexos,
                cursos=cursos,
            )
    except Exception as e:
        st.error(f"Não foi possível enviar o e-mail: {e}")
        return

    bim = _fmt_bimestres(bimestres_enviados) if bimestres_enviados else None
    registrar_uso(cursos, bim, email.strip(), st.secrets)

    cursos_fmt = " e ".join(f"**{c}**" for c in cursos)
    st.success(f"✅ Relatório(s) — {cursos_fmt} — enviado(s) para **{email.strip()}**.")
    st.info("Verifique sua caixa de entrada (e a pasta de spam). "
            "Nenhum arquivo fica armazenado nesta página.")


if enviar:
    processar_e_enviar()
