"""Cálculo de estatísticas, geração de gráficos, comentário por IA e relatório PDF.

Versão desacoplada do Google Colab:
- Os gráficos são devolvidos como figuras matplotlib (exibíveis no Streamlit) e
  convertidos para PNG em memória quando montados no PDF.
- O PDF é gerado em um buffer (BytesIO), pronto para download.
- A chave da OpenAI é recebida por parâmetro; se ausente, o comentário por IA
  é simplesmente pulado.

Particularidades do CEFET-MG:
- Cada bimestre tem pontuação máxima diferente (1º e 3º: 20 pts; 2º e 4º: 30
  pts), o que muda o limiar de aprovação parcial (60% dos pontos). O limiar é
  calculado dinamicamente a partir dos metadados do arquivo.

Faltas:
- Análise por **sinal estatístico** (P90 e média+2σ por disciplina; top-10
  alunos por faltas totais). Sem dependência de carga horária ou calendário.
"""
import io
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # backend sem display, adequado para servidor
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from babel.dates import format_date
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


# --------------------------------
# Pontuação por bimestre (CEFET-MG)
# --------------------------------
MAX_PONTOS_BIMESTRE = {1: 20, 2: 30, 3: 20, 4: 30}
PERCENTUAL_APROVACAO = 0.60  # 60% dos pontos do bimestre


def _limiar_aprovacao(metadados):
    """Calcula o limiar de aprovação parcial a partir dos metadados.

    Devolve (limiar_float, max_pts_int, bim_num_int). Se metadados estiverem
    ausentes/desconhecidos, usa o padrão de 12.0 (60% de 20) para não quebrar
    fluxos legados.
    """
    bim = (metadados or {}).get('bimestre_num')
    if bim not in MAX_PONTOS_BIMESTRE:
        return 12.0, 20, None
    max_pts = MAX_PONTOS_BIMESTRE[bim]
    return max_pts * PERCENTUAL_APROVACAO, max_pts, bim


# --------------------------------
# Utilidades de nome de disciplina
# --------------------------------
def get_simplified_name(code, disciplinas_dict):
    """Devolve o nome legível da disciplina, usando o catálogo/legenda
    quando disponível. CAIXA ALTA do SIGAA é convertida para Title Case para
    ficar mais legível em tabelas e gráficos.
    """
    full_name = disciplinas_dict.get(str(code), str(code))
    return full_name.strip().title()


# --------------------------------
# Estatísticas
# --------------------------------
def calcular_estatisticas(df_notas, disciplinas_dict, df_faltas=None, metadados=None):
    """Calcula as estatísticas básicas (notas + faltas) para um curso/bimestre."""
    df_notas = df_notas.copy()
    estatisticas = {}
    limiar, max_pts, bim_num = _limiar_aprovacao(metadados)
    estatisticas['limiar_aprovacao'] = limiar
    estatisticas['max_pontos_bimestre'] = max_pts
    estatisticas['bimestre_num'] = bim_num
    estatisticas['metadados'] = metadados or {}

    # Classifica as disciplinas presentes em três grupos:
    # - sem_dados: nenhuma nota lançada OU todas as notas iguais a zero;
    # - incompletas: têm notas, mas a nota máxima observada é baixa demais
    #   (≤ metade da pontuação do bimestre), sugerindo lançamento incompleto;
    # - com_notas: as demais (entram nas estatísticas e gráficos).
    disciplinas_presentes = [c for c in disciplinas_dict if c in df_notas.columns]
    disciplinas_com_notas = []
    disciplinas_sem_dados = []
    incompletas = []
    for c in disciplinas_presentes:
        serie_col = pd.to_numeric(df_notas[c], errors='coerce').dropna()
        if serie_col.empty or serie_col.max() == 0:
            disciplinas_sem_dados.append(c)
            continue
        disciplinas_com_notas.append(c)
        if serie_col.max() <= max_pts / 2:
            incompletas.append((c, get_simplified_name(c, disciplinas_dict),
                                float(serie_col.max())))

    estatisticas['disciplinas_sem_dados'] = [
        (c, get_simplified_name(c, disciplinas_dict)) for c in disciplinas_sem_dados]
    estatisticas['disciplinas_incompletas'] = incompletas
    codigos_incompletas = {c for c, _, _ in incompletas}

    df_apenas_notas = df_notas[disciplinas_com_notas]

    media_por_aluno = df_apenas_notas.mean(axis=1)
    estatisticas['total_alunos'] = len(df_notas)
    estatisticas['media_geral_turma'] = media_por_aluno.mean()
    estatisticas['desvio_padrao_medias'] = media_por_aluno.std()

    taxa_aprovacao = (df_apenas_notas >= limiar).all(axis=1).mean() * 100
    estatisticas['taxa_aprovacao_geral'] = f"{taxa_aprovacao:.2f}%"

    media_por_disciplina = df_apenas_notas.mean().sort_values()
    if not media_por_disciplina.empty:
        disciplina_menor_code = media_por_disciplina.index[0]
        estatisticas['disciplina_menor_media_code'] = disciplina_menor_code
        estatisticas['disciplina_menor_media_nome'] = get_simplified_name(
            disciplina_menor_code, disciplinas_dict)
        estatisticas['menor_media'] = media_por_disciplina.iloc[0]
        estatisticas['desvio_padrao_disciplina_critica'] = df_apenas_notas[disciplina_menor_code].std()
        estatisticas['alunos_abaixo_limiar_disciplina_critica'] = int(
            (df_apenas_notas[disciplina_menor_code] < limiar).sum())

        disciplina_maior_code = media_por_disciplina.index[-1]
        estatisticas['disciplina_maior_media_nome'] = get_simplified_name(
            disciplina_maior_code, disciplinas_dict)
        estatisticas['maior_media'] = media_por_disciplina.iloc[-1]
    else:
        for key in ['disciplina_menor_media_nome', 'disciplina_maior_media_nome', 'disciplina_menor_media_code']:
            estatisticas[key] = "N/A"
        for key in ['menor_media', 'maior_media', 'desvio_padrao_disciplina_critica', 'alunos_abaixo_limiar_disciplina_critica']:
            estatisticas[key] = 0

    df_notas['disciplinas_abaixo_limiar'] = (df_apenas_notas < limiar).sum(axis=1)
    top_10 = df_notas.sort_values(by='disciplinas_abaixo_limiar', ascending=False).head(10)
    estatisticas['top_10_alunos_criticos'] = top_10[['nome', 'disciplinas_abaixo_limiar']]

    summary_list = []
    for col in disciplinas_com_notas:
        stats = df_apenas_notas[col].describe()
        nome_disc = get_simplified_name(col, disciplinas_dict)
        if col in codigos_incompletas:
            nome_disc += ' *'
        summary_list.append({
            'Disciplina': nome_disc,
            'Média': f"{stats.get('mean', 0):.2f}",
            'Mediana': f"{stats.get('50%', 0):.2f}",
            'Desv. Padrão': f"{stats.get('std', 0):.2f}",
            'Mínimo': f"{stats.get('min', 0):.2f}",
            'Máximo': f"{stats.get('max', 0):.2f}",
        })
    estatisticas['boxplot_summary_df'] = pd.DataFrame(summary_list)
    estatisticas['disciplinas_com_notas'] = disciplinas_com_notas

    # ----- Faltas (sinal estatístico) -----
    if df_faltas is not None and not df_faltas.empty:
        estatisticas.update(_calcular_estatisticas_faltas(df_faltas, disciplinas_dict))

    return estatisticas


def _calcular_estatisticas_faltas(df_faltas, disciplinas_dict):
    """Resumo estatístico das faltas: por disciplina (média, mediana, P90, σ,
    quantidade de alunos acima de P90 e acima de média+2σ) + top-10 alunos
    com mais faltas totais.
    """
    cols = [c for c in disciplinas_dict if c in df_faltas.columns]
    cols = [c for c in cols if df_faltas[c].notna().any()]
    if not cols:
        return {
            'faltas_disponiveis': False,
            'faltas_summary_df': pd.DataFrame(),
            'top_10_faltosos': pd.DataFrame(),
        }

    df_f = df_faltas[cols].apply(pd.to_numeric, errors='coerce')
    summary = []
    for col in cols:
        s = df_f[col].dropna()
        if s.empty:
            continue
        media = s.mean()
        mediana = s.median()
        p90 = s.quantile(0.90)
        std = s.std()
        cutoff_sigma = media + 2 * std
        n_acima_p90 = int((s > p90).sum())
        n_acima_sigma = int((s > cutoff_sigma).sum())
        summary.append({
            'Disciplina': get_simplified_name(col, disciplinas_dict),
            'Média': f"{media:.1f}",
            'Mediana': f"{mediana:.1f}",
            'P90': f"{p90:.1f}",
            'Desv. Padrão': f"{std:.1f}",
            'Alunos > P90': n_acima_p90,
            'Alunos > μ+2σ': n_acima_sigma,
        })
    summary_df = pd.DataFrame(summary)

    df_faltas_aluno = df_faltas[['nome'] + cols].copy()
    df_faltas_aluno['Total Faltas'] = df_f.sum(axis=1)
    top10 = df_faltas_aluno[['nome', 'Total Faltas']].sort_values(
        by='Total Faltas', ascending=False).head(10)
    top10['Total Faltas'] = top10['Total Faltas'].astype(int)

    return {
        'faltas_disponiveis': True,
        'faltas_summary_df': summary_df,
        'top_10_faltosos': top10,
        '_faltas_cols': cols,
    }


# --------------------------------
# Comentário por IA (opcional)
# --------------------------------
def gerar_comentario_ia(estatisticas, nome_curso, api_key, modelo="gpt-4o-mini"):
    """Gera um comentário analítico via API da OpenAI. Se `api_key` for vazio,
    devolve uma mensagem informando que a análise foi pulada."""
    if not api_key:
        return ("A análise por IA não foi gerada (chave da OpenAI não configurada). "
                "Configure-a para habilitar este comentário.")

    summary_markdown = estatisticas['boxplot_summary_df'].to_markdown(index=False)
    limiar = estatisticas.get('limiar_aprovacao', 12.0)
    max_pts = estatisticas.get('max_pontos_bimestre', 20)
    bim = estatisticas.get('bimestre_num')
    prompt = f"""
    Você é um especialista em análise de dados educacionais. Com base nos dados a seguir, gere uma análise em português. Não faça em formato MarkDown, ou seja, não use * ou #.

    **Contexto:**
    - **Curso:** {nome_curso}
    - **Bimestre:** {bim if bim else 'n/d'} (pontuação máxima {max_pts}, aprovação parcial ≥ {limiar:.1f})
    - **Total de Alunos:** {estatisticas['total_alunos']}

    **Análise Geral da Turma:**
    - **Média Geral (0-{max_pts}):** {estatisticas['media_geral_turma']:.2f}
    - **Dispersão das Médias (Desvio Padrão):** {estatisticas['desvio_padrao_medias']:.2f}
    - **Taxa de Aprovação Geral (Nota >= {limiar:.1f} em tudo):** {estatisticas['taxa_aprovacao_geral']}

    **Resumo Estatístico por Disciplina:**
    {summary_markdown}

    **Instruções:**
    1. Primeiro Parágrafo: desempenho geral da turma (média satisfatória? turma homogênea ou heterogênea? taxa de aprovação preocupante?).
    2. Segundo Parágrafo: analise a disciplina com menor média ({estatisticas['disciplina_menor_media_nome']}) e o número de alunos com nota baixa ({estatisticas.get('alunos_abaixo_limiar_disciplina_critica', 0)}).
    3. Terceiro Parágrafo: com base na tabela, aponte disciplinas com desempenho ruim, compare indicadores e sugira melhorias.

    O tom deve ser profissional e objetivo.
    """

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": modelo, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content')
        if content:
            return content.replace('\n', '<br/>')
        return "Análise da IA indisponível (resposta inesperada da API)."
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            return "A análise por IA não pôde ser gerada: a chave da OpenAI é inválida ou expirou."
        return f"Erro HTTP ao chamar a API da IA: {e}"
    except requests.exceptions.RequestException as e:
        return f"Erro de comunicação com a IA: {e}"
    except Exception as e:
        return f"Erro ao processar o comentário da IA: {e}"


# --------------------------------
# Gráficos (retornam figuras matplotlib)
# --------------------------------
def _colunas_com_notas(df_notas, disciplinas_dict):
    """Disciplinas com notas reais (descarta as sem dados ou totalmente zeradas),
    mantendo a coerência com as tabelas do relatório."""
    cols = []
    for c in disciplinas_dict:
        if c in df_notas.columns:
            s = pd.to_numeric(df_notas[c], errors='coerce')
            if s.notna().any() and s.max() > 0:
                cols.append(c)
    return cols


def grafico_distribuicao_notas(df_notas, nome_curso, disciplinas_dict, max_pts=20):
    presentes = _colunas_com_notas(df_notas, disciplinas_dict)
    if not presentes:
        return None
    media_aluno = df_notas[presentes].mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(media_aluno, kde=True, bins=15, ax=ax)
    ax.set_title(f'Distribuição das Médias Finais - {nome_curso}', fontsize=16)
    ax.set_xlabel(f'Média Final do Aluno (0 a {max_pts})', fontsize=12)
    ax.set_ylabel('Número de Alunos', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig


def grafico_media_por_disciplina(df_notas, nome_curso, disciplinas_dict, max_pts=20):
    presentes = _colunas_com_notas(df_notas, disciplinas_dict)
    if not presentes:
        return None
    media = df_notas[presentes].mean().sort_values(ascending=False)
    labels = [get_simplified_name(c, disciplinas_dict) for c in media.index]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=media.values, y=labels, ax=ax)
    ax.set_title(f'Média por Disciplina - {nome_curso}', fontsize=16)
    ax.set_xlabel(f'Média da Turma (0 a {max_pts})', fontsize=12)
    ax.set_ylabel('Disciplina', fontsize=12)
    ax.set_xlim(0, max_pts)
    fig.tight_layout()
    return fig


def grafico_boxplot_disciplinas(df_notas, nome_curso, disciplinas_dict, max_pts=20):
    presentes = _colunas_com_notas(df_notas, disciplinas_dict)
    if not presentes:
        return None
    df_melted = df_notas.melt(value_vars=presentes, var_name='disciplina_code', value_name='nota')
    df_melted['disciplina_nome'] = df_melted['disciplina_code'].apply(
        lambda c: get_simplified_name(c, disciplinas_dict))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='nota', y='disciplina_nome', data=df_melted, orient='h', ax=ax)
    ax.set_title(f'Dispersão de Notas por Disciplina - {nome_curso}', fontsize=16)
    ax.set_xlabel(f'Nota (0 a {max_pts})', fontsize=12)
    ax.set_ylabel('Disciplina', fontsize=12)
    ax.set_xlim(0, max_pts)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    fig.tight_layout()
    return fig


def grafico_disciplina_critica(df_notas, disciplina_code, disciplina_nome, nome_curso, max_pts=20):
    if disciplina_code == "N/A" or disciplina_code not in df_notas.columns:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_notas[disciplina_code].dropna(), kde=True, bins=10, color='indianred', ax=ax)
    ax.set_title(f'Dispersão de Notas: {disciplina_nome} ({nome_curso})', fontsize=16)
    ax.set_xlabel(f'Nota na Disciplina (0 a {max_pts})', fontsize=12)
    ax.set_ylabel('Número de Alunos', fontsize=12)
    ax.set_xlim(0, max_pts)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig


def grafico_faltas_total_por_aluno(df_faltas, nome_curso, cols_disciplinas):
    cols = [c for c in cols_disciplinas if c in df_faltas.columns]
    if not cols:
        return None
    totais = df_faltas[cols].apply(pd.to_numeric, errors='coerce').sum(axis=1).dropna()
    if totais.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(totais, kde=True, bins=15, ax=ax, color='steelblue')
    ax.set_title(f'Distribuição de Faltas Totais por Aluno - {nome_curso}', fontsize=16)
    ax.set_xlabel('Total de Faltas no Bimestre', fontsize=12)
    ax.set_ylabel('Número de Alunos', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig


def grafico_faltas_boxplot_disciplina(df_faltas, nome_curso, disciplinas_dict, cols_disciplinas):
    cols = [c for c in cols_disciplinas if c in df_faltas.columns]
    if not cols:
        return None
    df_melted = df_faltas.melt(value_vars=cols, var_name='disciplina_code', value_name='faltas')
    df_melted['faltas'] = pd.to_numeric(df_melted['faltas'], errors='coerce')
    df_melted = df_melted.dropna(subset=['faltas'])
    if df_melted.empty:
        return None
    df_melted['disciplina_nome'] = df_melted['disciplina_code'].apply(
        lambda c: get_simplified_name(c, disciplinas_dict))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='faltas', y='disciplina_nome', data=df_melted, orient='h', ax=ax, color='lightcoral')
    ax.set_title(f'Dispersão de Faltas por Disciplina - {nome_curso}', fontsize=16)
    ax.set_xlabel('Faltas no Bimestre', fontsize=12)
    ax.set_ylabel('Disciplina', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    fig.tight_layout()
    return fig


def gerar_todos_graficos(df_notas, nome_curso, disciplinas_dict, estatisticas, df_faltas=None):
    """Gera todas as figuras e devolve um dicionário {chave: Figure|None}."""
    max_pts = estatisticas.get('max_pontos_bimestre', 20)
    figuras = {
        'distribuicao_geral': grafico_distribuicao_notas(df_notas, nome_curso, disciplinas_dict, max_pts),
        'media_disciplina': grafico_media_por_disciplina(df_notas, nome_curso, disciplinas_dict, max_pts),
        'boxplot_disciplinas': grafico_boxplot_disciplinas(df_notas, nome_curso, disciplinas_dict, max_pts),
        'disciplina_critica': grafico_disciplina_critica(
            df_notas,
            estatisticas.get('disciplina_menor_media_code', 'N/A'),
            estatisticas.get('disciplina_menor_media_nome', 'N/A'),
            nome_curso,
            max_pts,
        ),
    }
    if df_faltas is not None and estatisticas.get('faltas_disponiveis'):
        cols = estatisticas.get('_faltas_cols', [])
        figuras['faltas_total_aluno'] = grafico_faltas_total_por_aluno(df_faltas, nome_curso, cols)
        figuras['faltas_boxplot_disciplina'] = grafico_faltas_boxplot_disciplina(
            df_faltas, nome_curso, disciplinas_dict, cols)
    return figuras


def _fig_para_imagem(fig):
    """Converte uma figura matplotlib em BytesIO PNG para uso no reportlab."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf


# --------------------------------
# Relatório PDF (em memória) com sumário (TOC)
# --------------------------------
def _cabecalho_factory(logo_path=None):
    def adicionar_cabecalho(canvas, doc):
        canvas.saveState()
        largura, altura = doc.pagesize

        # O logo institucional aparece apenas na primeira página (capa).
        if logo_path and doc.page == 1:
            try:
                logo = ImageReader(logo_path)
                logo_w, logo_h = logo.getSize()
                aspect = logo_h / float(logo_w)
                display_h = 2.0 * cm
                display_w = display_h / aspect
                canvas.drawImage(logo, 1.5 * cm, altura - 3 * cm,
                                 width=display_w, height=display_h,
                                 preserveAspectRatio=True, mask='auto')
            except Exception:
                pass

        canvas.setFont('Times-Bold', 10)
        canvas.setFillColor(colors.HexColor('#002060'))
        y = altura - 2 * cm
        canvas.drawCentredString(largura / 2.0, y, "Serviço Público Federal")
        canvas.drawCentredString(largura / 2.0, y - 0.5 * cm, "Ministério da Educação")
        canvas.drawCentredString(largura / 2.0, y - 1.0 * cm,
                                 "Centro Federal de Educação Tecnológica de Minas Gerais")

        canvas.setFont('Times-Italic', 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(
            largura / 2.0, 1.5 * cm,
            "Desenvolvido pelo Professor Diego Camargo (diegocamargo@cefetmg.br).")
        canvas.restoreState()
    return adicionar_cabecalho


class _DocComSumario(BaseDocTemplate):
    """BaseDocTemplate que captura os títulos H1/H2 do story e popula o TOC.

    O ReportLab dispara `afterFlowable` após cada flowable; identificamos os
    Paragraphs estilizados como ``H1Sumario`` ou ``H2Sumario`` e emitimos um
    ``TOCEntry`` com o nível, texto e número de página correspondente.
    """

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        text = flowable.getPlainText()
        if style_name == 'H1Sumario':
            self.notify('TOCEntry', (0, text, self.page))
        elif style_name == 'H2Sumario':
            self.notify('TOCEntry', (1, text, self.page))


def criar_relatorio_pdf(nome_curso, estatisticas, figuras, logo_path=None):
    """Cria o relatório em PDF e devolve um BytesIO pronto para download."""
    buffer = io.BytesIO()
    doc = _DocComSumario(buffer, pagesize=A4)
    largura, altura = A4
    frame = Frame(2.5 * cm, 2.5 * cm, largura - 5 * cm, altura - 6 * cm, id='normal')
    doc.addPageTemplates([
        PageTemplate(id='principal', frames=[frame],
                     onPage=_cabecalho_factory(logo_path))
    ])

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', parent=styles['BodyText'],
                              alignment=TA_JUSTIFY, fontName='Times-Roman'))
    style_titulo = ParagraphStyle(name='TituloCapa', fontSize=22, alignment=TA_CENTER,
                                  leading=26, spaceAfter=1.5 * cm,
                                  textColor=colors.HexColor('#002060'), fontName='Times-Bold')
    style_subtitulo = ParagraphStyle(name='SubTituloCapa', fontSize=18, alignment=TA_CENTER,
                                     spaceAfter=2 * cm, textColor=colors.HexColor('#002060'),
                                     fontName='Times-Roman')
    style_capa_info = ParagraphStyle(name='CapaInfo', fontSize=13, alignment=TA_CENTER,
                                     spaceBefore=0.3 * cm, fontName='Times-Bold',
                                     textColor=colors.HexColor('#002060'))
    style_data = ParagraphStyle(name='DataCapa', fontSize=12, alignment=TA_CENTER,
                                spaceBefore=9 * cm, fontName='Times-Roman')
    # Estilos H1/H2 que o `_DocComSumario` registra no TOC.
    style_h1 = ParagraphStyle(name='H1Sumario', parent=styles['h1'],
                              fontName='Times-Bold', textColor=colors.HexColor('#002060'),
                              spaceBefore=12, spaceAfter=8)
    style_h2 = ParagraphStyle(name='H2Sumario', parent=styles['h2'],
                              fontName='Times-Bold')
    style_corpo = styles['Justify']
    style_caption = ParagraphStyle(name='Caption', parent=styles['BodyText'],
                                   alignment=TA_LEFT, fontName='Times-Italic',
                                   fontSize=9, textColor=colors.grey)
    style_toc_titulo = ParagraphStyle(name='TocTitulo', fontSize=16, alignment=TA_CENTER,
                                      fontName='Times-Bold',
                                      textColor=colors.HexColor('#002060'),
                                      spaceAfter=0.8 * cm)
    style_nota = ParagraphStyle(name='NotaRodape', parent=styles['BodyText'],
                                alignment=TA_LEFT, fontName='Times-Italic',
                                fontSize=8, textColor=colors.HexColor('#7a3030'),
                                spaceBefore=4)
    style_celula = ParagraphStyle(name='Celula', parent=styles['BodyText'],
                                  fontName='Times-Roman', fontSize=9, leading=11)
    style_celula_b = ParagraphStyle(name='CelulaB', parent=styles['BodyText'],
                                    fontName='Times-Bold', fontSize=9, leading=11)

    story = []

    # Numeração automática de capítulos/seções (item: sumário efetivo) e quebras
    # de página que não deixam "fantasmas" — qualquer Spacer pendurado antes da
    # quebra é descartado para não gerar páginas em branco.
    contadores = {'cap': 0, 'sub': 0}

    def h1(texto):
        contadores['cap'] += 1
        contadores['sub'] = 0
        story.append(Paragraph(f"{contadores['cap']}. {texto}", style_h1))

    def h2(texto):
        contadores['sub'] += 1
        story.append(Paragraph(
            f"{contadores['cap']}.{contadores['sub']} {texto}", style_h2))

    def quebra_pagina():
        while story and isinstance(story[-1], Spacer):
            story.pop()
        story.append(PageBreak())

    # --- Metadados usados em todo o relatório ---
    limiar = estatisticas.get('limiar_aprovacao', 12.0)
    max_pts = estatisticas.get('max_pontos_bimestre', 20)
    bim = estatisticas.get('bimestre_num')
    meta = estatisticas.get('metadados', {}) or {}
    serie = meta.get('serie')
    serie_txt = {1: '1ª Série', 2: '2ª Série', 3: '3ª Série'}.get(serie)

    # --- Capa ---
    story.append(Spacer(1, 5 * cm))
    story.append(Paragraph("RELATÓRIO DE ACOMPANHAMENTO ACADÊMICO", style_titulo))
    subtitulo = f"Curso Técnico em {nome_curso.title()}"
    if serie_txt:
        subtitulo += f" — {serie_txt}"
    story.append(Paragraph(subtitulo, style_subtitulo))
    partes_capa = []
    if bim:
        partes_capa.append(f"{bim}º Bimestre")
    if meta.get('periodo_letivo'):
        partes_capa.append(f"Período Letivo {meta['periodo_letivo']}")
    if partes_capa:
        story.append(Paragraph(" · ".join(partes_capa), style_capa_info))
    data_pt = format_date(datetime.now(), format="d 'de' MMMM 'de' y", locale='pt_BR')
    story.append(Paragraph(data_pt, style_data))
    quebra_pagina()

    # --- Sumário ---
    story.append(Paragraph("Sumário", style_toc_titulo))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(fontName='Times-Bold', fontSize=12, name='TOCH1',
                       leftIndent=0, firstLineIndent=-20, spaceBefore=10, leading=16),
        ParagraphStyle(fontName='Times-Roman', fontSize=10, name='TOCH2',
                       leftIndent=20, firstLineIndent=-20, spaceBefore=4, leading=14),
    ]
    story.append(toc)
    quebra_pagina()

    # --- Glossário (termos usados no relatório) ---
    h1("Glossário")
    story.append(Paragraph(
        "Os termos abaixo aparecem ao longo deste relatório.", style_caption))
    story.append(Spacer(1, 0.3 * cm))
    termos = [
        ("Média", "Soma das notas dividida pela quantidade (de alunos ou de "
                  "disciplinas). Indica o desempenho típico."),
        ("Mediana", "Valor central quando as notas são ordenadas; metade da turma "
                    "fica acima e metade abaixo. É menos sensível a casos extremos "
                    "que a média."),
        ("Desvio Padrão (σ)", "Mede o quanto as notas se afastam da média. Quanto "
                              "maior, mais heterogênea é a turma."),
        ("Pontuação do Bimestre", f"Total de pontos distribuíveis no bimestre "
                                  f"(20 no 1º e 3º; 30 no 2º e 4º). Neste relatório: {max_pts}."),
        ("Limiar de Aprovação Parcial", "60% da pontuação do bimestre — referência "
                                        f"de acompanhamento (aqui, ≥ {limiar:.1f})."),
        ("Taxa de Aprovação", "Percentual de alunos com nota igual ou acima do "
                              "limiar em todas as disciplinas."),
        ("P90 (Percentil 90)", "Valor abaixo do qual estão 90% dos alunos. Em "
                               "faltas, quem ultrapassa o P90 destoa do grupo."),
        ("μ + 2σ", "Média mais dois desvios padrão. Limite estatístico que sinaliza "
                   "valores atípicos (usado na análise de faltas)."),
        ("Disciplina sem dados", "Disciplina sem nenhuma nota lançada ou com todas "
                                 "as notas zeradas. Não entra nas estatísticas."),
        ("Asterisco (*)", "Marca disciplinas cuja nota máxima observada é baixa "
                          "(≤ metade da pontuação do bimestre), sugerindo lançamento "
                          "possivelmente incompleto — convém confirmar com o professor."),
    ]
    linhas_gloss = [[Paragraph(f"<b>{t}</b>", style_celula_b), Paragraph(d, style_celula)]
                    for t, d in termos]
    tabela_gloss = Table(linhas_gloss, colWidths=[4.5 * cm, 11.5 * cm])
    tabela_gloss.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef1f7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(tabela_gloss)
    quebra_pagina()

    # --- Estatísticas gerais ---
    h1("Estatísticas Gerais da Turma")
    if bim:
        story.append(Paragraph(
            f"Bimestre: <b>{bim}º</b> · Pontuação máxima: <b>{max_pts}</b> · "
            f"Aprovação parcial (≥ 60%): <b>{limiar:.1f}</b>",
            style_caption,
        ))
    story.append(Spacer(1, 0.4 * cm))
    dados_gerais = [
        ['Total de Alunos:', estatisticas['total_alunos']],
        ['Média Geral da Turma:', f"{estatisticas['media_geral_turma']:.2f}"],
        ['Desvio Padrão das Médias:', f"{estatisticas['desvio_padrao_medias']:.2f}"],
        [f'Taxa de Aprovação (≥ {limiar:.1f} em tudo):', estatisticas['taxa_aprovacao_geral']],
        ['Disciplina com Maior Média:', f"{estatisticas['disciplina_maior_media_nome']} ({estatisticas['maior_media']:.2f})"],
        ['Disciplina com Menor Média:', f"{estatisticas['disciplina_menor_media_nome']} ({estatisticas['menor_media']:.2f})"],
    ]
    if meta.get('turma'):
        dados_gerais.insert(0, ['Turma:', meta['turma']])
    tabela_geral = Table(dados_gerais, colWidths=[7 * cm, 9 * cm])
    tabela_geral.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
    ]))
    story.append(tabela_geral)
    story.append(Spacer(1, 1 * cm))

    # --- Alunos críticos (notas) ---
    h2(f"Alunos com Maior Número de Disciplinas Abaixo do Limiar (&lt;{limiar:.1f})")
    story.append(Spacer(1, 0.4 * cm))
    dados_criticos = [['Aluno', f'Disciplinas < {limiar:.1f}']] + \
        estatisticas['top_10_alunos_criticos'].values.tolist()
    tabela_criticos = Table(dados_criticos, colWidths=[12 * cm, 4 * cm])
    tabela_criticos.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
    ]))
    story.append(tabela_criticos)
    story.append(Spacer(1, 1 * cm))

    # --- Gráficos gerais ---
    h1("Visualizações Gráficas Gerais")
    story.append(Spacer(1, 0.4 * cm))
    for chave in ['distribuicao_geral', 'media_disciplina', 'boxplot_disciplinas']:
        fig = figuras.get(chave)
        if fig is not None:
            story.append(Image(_fig_para_imagem(fig), width=16 * cm, height=11 * cm, kind='proportional'))
            story.append(Spacer(1, 1 * cm))

    # --- Resumo estatístico por disciplina ---
    h1("Resumo Estatístico por Disciplina")
    story.append(Spacer(1, 0.4 * cm))
    df_summary = estatisticas['boxplot_summary_df']
    table_data = [df_summary.columns.tolist()] + df_summary.values.tolist()
    tabela_summary = Table(table_data, colWidths=[4.5 * cm, 2 * cm, 2 * cm, 2.5 * cm, 2 * cm, 2 * cm])
    tabela_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.cadetblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
    ]))
    story.append(tabela_summary)

    # Nota de rodapé (item: disciplinas possivelmente incompletas) — asterisco.
    incompletas = estatisticas.get('disciplinas_incompletas') or []
    if incompletas:
        nomes_inc = '; '.join(f"{nome} (máx. {mx:.0f})" for _, nome, mx in incompletas)
        story.append(Paragraph(
            f"<b>*</b> Disciplina(s) com nota máxima observada igual ou inferior à "
            f"metade da pontuação do bimestre ({max_pts / 2:.0f}), o que sugere "
            f"lançamento possivelmente incompleto. Recomenda-se confirmar com o(a) "
            f"professor(a) se as notas estão corretas: {nomes_inc}.",
            style_nota,
        ))
    story.append(Spacer(1, 0.8 * cm))

    # --- Disciplinas sem dados (item: identificar disciplinas "sem nada") ---
    sem_dados = estatisticas.get('disciplinas_sem_dados') or []
    if sem_dados:
        h2("Disciplinas sem Notas Lançadas")
        story.append(Paragraph(
            "As disciplinas abaixo não tinham notas lançadas (ou estavam todas "
            "zeradas) no momento do processamento e, por isso, <b>não foram "
            "consideradas</b> nas estatísticas e nos gráficos. Recomenda-se "
            "verificar o lançamento com o(a) professor(a) responsável.",
            style_corpo,
        ))
        story.append(Spacer(1, 0.3 * cm))
        dados_sem = [['Disciplina', 'Observação']]
        for _cod, nome in sem_dados:
            dados_sem.append([
                Paragraph(nome, style_celula),
                Paragraph("Não havia dados lançados para esta disciplina no período.",
                          style_celula),
            ])
        tabela_sem = Table(dados_sem, colWidths=[6 * cm, 10 * cm])
        tabela_sem.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8a6d00')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(tabela_sem)
        story.append(Spacer(1, 0.8 * cm))

    # --- Disciplina crítica ---
    fig_critica = figuras.get('disciplina_critica')
    if fig_critica is not None:
        quebra_pagina()
        h1("Análise da Disciplina com Menor Desempenho")
        story.append(Spacer(1, 0.4 * cm))
        dados_critica = [
            ["Disciplina:", estatisticas['disciplina_menor_media_nome']],
            ["Média da Turma:", f"{estatisticas['menor_media']:.2f}"],
            ["Desvio Padrão:", f"{estatisticas['desvio_padrao_disciplina_critica']:.2f}"],
            [f"Alunos com Nota < {limiar:.1f}:",
             f"{estatisticas.get('alunos_abaixo_limiar_disciplina_critica', 0)}"],
        ]
        tabela_critica = Table(dados_critica, colWidths=[7 * cm, 9 * cm])
        tabela_critica.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightpink),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
        ]))
        story.append(tabela_critica)
        story.append(Spacer(1, 0.5 * cm))
        story.append(Image(_fig_para_imagem(fig_critica), width=16 * cm, height=11 * cm, kind='proportional'))

    # --- Frequência (Faltas) ---
    if estatisticas.get('faltas_disponiveis'):
        quebra_pagina()
        h1("Análise de Frequência (sinal estatístico)")
        story.append(Paragraph(
            "A análise abaixo destaca alunos cuja quantidade de faltas se "
            "afasta do comportamento típico da turma. Os limites estatísticos "
            "(P90 e μ+2σ) <b>não substituem</b> o limite legal de 25% da carga "
            "horária — eles apenas sinalizam, sem depender do calendário, quem "
            "merece um olhar atento.",
            style_corpo,
        ))
        story.append(Spacer(1, 0.5 * cm))

        h2("Top 10 Alunos com Mais Faltas no Bimestre")
        story.append(Spacer(1, 0.3 * cm))
        top10 = estatisticas.get('top_10_faltosos', pd.DataFrame())
        if not top10.empty:
            dados_falt = [['Aluno', 'Total de Faltas']] + top10.values.tolist()
            tabela_falt = Table(dados_falt, colWidths=[12 * cm, 4 * cm])
            tabela_falt.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7a3030')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ]))
            story.append(tabela_falt)
            story.append(Spacer(1, 0.8 * cm))

        h2("Resumo de Faltas por Disciplina")
        story.append(Spacer(1, 0.3 * cm))
        df_faltas_summary = estatisticas.get('faltas_summary_df', pd.DataFrame())
        if not df_faltas_summary.empty:
            cols_widths = [4.5 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 2 * cm, 2 * cm, 2.4 * cm]
            table_data = [df_faltas_summary.columns.tolist()] + df_faltas_summary.values.tolist()
            tabela_falt_disc = Table(table_data, colWidths=cols_widths)
            tabela_falt_disc.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7a3030')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
            ]))
            story.append(tabela_falt_disc)
            story.append(Spacer(1, 0.8 * cm))

        for chave in ['faltas_total_aluno', 'faltas_boxplot_disciplina']:
            fig = figuras.get(chave)
            if fig is not None:
                story.append(Image(_fig_para_imagem(fig), width=16 * cm, height=11 * cm, kind='proportional'))
                story.append(Spacer(1, 0.8 * cm))

    # --- Comentário da IA ---
    if estatisticas.get('comentario_ia'):
        quebra_pagina()
        h1("Análise e Comentários (Gerado por Inteligência Artificial)")
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(estatisticas['comentario_ia'], style_corpo))

    # Remove qualquer Spacer pendurado no fim do documento — evita uma página em
    # branco extra quando o último conteúdo termina perto do rodapé (item 6).
    while story and isinstance(story[-1], Spacer):
        story.pop()

    # `multiBuild` faz duas passadas: a primeira coleta as entradas do TOC, a
    # segunda monta o documento final já com os números de página corretos.
    doc.multiBuild(story)
    # Libera as figuras matplotlib para não acumular memória entre relatórios.
    for fig in figuras.values():
        if fig is not None:
            plt.close(fig)
    buffer.seek(0)
    return buffer
