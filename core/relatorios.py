"""Cálculo de estatísticas, geração de gráficos, comentário por IA e relatório PDF.

Versão desacoplada do Google Colab:
- Os gráficos são devolvidos como figuras matplotlib (exibíveis no Streamlit) e
  convertidos para PNG em memória quando montados no PDF.
- O PDF é gerado em um buffer (BytesIO), pronto para download.
- A chave da OpenAI é recebida por parâmetro; se ausente, o comentário por IA
  é simplesmente pulado.
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
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)


# --------------------------------
# Estatísticas
# --------------------------------
def get_simplified_name(code, disciplinas_dict):
    """Extrai o nome simplificado de uma disciplina a partir do seu código."""
    full_name = disciplinas_dict.get(str(code), str(code))
    return re.split(r' - |:', full_name)[0].strip().title()


def calcular_estatisticas(df_notas, disciplinas_dict):
    """Calcula as estatísticas básicas para um curso."""
    df_notas = df_notas.copy()
    estatisticas = {}

    disciplinas_presentes = [c for c in disciplinas_dict if c in df_notas.columns]
    disciplinas_com_notas = [c for c in disciplinas_presentes if df_notas[c].notna().any()]
    df_apenas_notas = df_notas[disciplinas_com_notas]

    media_por_aluno = df_apenas_notas.mean(axis=1)
    estatisticas['total_alunos'] = len(df_notas)
    estatisticas['media_geral_turma'] = media_por_aluno.mean()
    estatisticas['desvio_padrao_medias'] = media_por_aluno.std()

    taxa_aprovacao = (df_apenas_notas >= 12.0).all(axis=1).mean() * 100
    estatisticas['taxa_aprovacao_geral'] = f"{taxa_aprovacao:.2f}%"

    media_por_disciplina = df_apenas_notas.mean().sort_values()
    if not media_por_disciplina.empty:
        disciplina_menor_code = media_por_disciplina.index[0]
        estatisticas['disciplina_menor_media_code'] = disciplina_menor_code
        estatisticas['disciplina_menor_media_nome'] = get_simplified_name(
            disciplina_menor_code, disciplinas_dict)
        estatisticas['menor_media'] = media_por_disciplina.iloc[0]
        estatisticas['desvio_padrao_disciplina_critica'] = df_apenas_notas[disciplina_menor_code].std()
        estatisticas['alunos_abaixo_12_disciplina_critica'] = int(
            (df_apenas_notas[disciplina_menor_code] < 12.0).sum())

        disciplina_maior_code = media_por_disciplina.index[-1]
        estatisticas['disciplina_maior_media_nome'] = get_simplified_name(
            disciplina_maior_code, disciplinas_dict)
        estatisticas['maior_media'] = media_por_disciplina.iloc[-1]
    else:
        for key in ['disciplina_menor_media_nome', 'disciplina_maior_media_nome', 'disciplina_menor_media_code']:
            estatisticas[key] = "N/A"
        for key in ['menor_media', 'maior_media', 'desvio_padrao_disciplina_critica', 'alunos_abaixo_12_disciplina_critica']:
            estatisticas[key] = 0

    df_notas['disciplinas_abaixo_12'] = (df_apenas_notas < 12.0).sum(axis=1)
    top_10 = df_notas.sort_values(by='disciplinas_abaixo_12', ascending=False).head(10)
    estatisticas['top_10_alunos_criticos'] = top_10[['nome', 'disciplinas_abaixo_12']]

    summary_list = []
    for col in disciplinas_com_notas:
        stats = df_apenas_notas[col].describe()
        summary_list.append({
            'Disciplina': get_simplified_name(col, disciplinas_dict),
            'Média': f"{stats.get('mean', 0):.2f}",
            'Mediana': f"{stats.get('50%', 0):.2f}",
            'Desv. Padrão': f"{stats.get('std', 0):.2f}",
            'Mínimo': f"{stats.get('min', 0):.2f}",
            'Máximo': f"{stats.get('max', 0):.2f}",
        })
    estatisticas['boxplot_summary_df'] = pd.DataFrame(summary_list)
    estatisticas['disciplinas_com_notas'] = disciplinas_com_notas

    return estatisticas


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
    prompt = f"""
    Você é um especialista em análise de dados educacionais. Com base nos dados a seguir, gere uma análise em português. Não faça em formato MarkDown, ou seja, não use * ou #.

    **Contexto:**
    - **Curso:** {nome_curso}
    - **Total de Alunos:** {estatisticas['total_alunos']}

    **Análise Geral da Turma:**
    - **Média Geral (0-20):** {estatisticas['media_geral_turma']:.2f}
    - **Dispersão das Médias (Desvio Padrão):** {estatisticas['desvio_padrao_medias']:.2f}
    - **Taxa de Aprovação Geral (Nota >= 12 em tudo):** {estatisticas['taxa_aprovacao_geral']}

    **Resumo Estatístico por Disciplina:**
    {summary_markdown}

    **Instruções:**
    1. Primeiro Parágrafo: desempenho geral da turma (média satisfatória? turma homogênea ou heterogênea? taxa de aprovação preocupante?).
    2. Segundo Parágrafo: analise a disciplina com menor média ({estatisticas['disciplina_menor_media_nome']}) e o número de alunos com nota baixa ({estatisticas['alunos_abaixo_12_disciplina_critica']}).
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
def grafico_distribuicao_notas(df_notas, nome_curso, disciplinas_dict):
    presentes = [c for c in disciplinas_dict if c in df_notas.columns and df_notas[c].notna().any()]
    if not presentes:
        return None
    media_aluno = df_notas[presentes].mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(media_aluno, kde=True, bins=15, ax=ax)
    ax.set_title(f'Distribuição das Médias Finais - {nome_curso}', fontsize=16)
    ax.set_xlabel('Média Final do Aluno (0 a 20)', fontsize=12)
    ax.set_ylabel('Número de Alunos', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig


def grafico_media_por_disciplina(df_notas, nome_curso, disciplinas_dict):
    presentes = [c for c in disciplinas_dict if c in df_notas.columns and df_notas[c].notna().any()]
    if not presentes:
        return None
    media = df_notas[presentes].mean().sort_values(ascending=False)
    labels = [get_simplified_name(c, disciplinas_dict) for c in media.index]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=media.values, y=labels, ax=ax)
    ax.set_title(f'Média por Disciplina - {nome_curso}', fontsize=16)
    ax.set_xlabel('Média da Turma (0 a 20)', fontsize=12)
    ax.set_ylabel('Disciplina', fontsize=12)
    ax.set_xlim(0, 20)
    fig.tight_layout()
    return fig


def grafico_boxplot_disciplinas(df_notas, nome_curso, disciplinas_dict):
    presentes = [c for c in disciplinas_dict if c in df_notas.columns and df_notas[c].notna().any()]
    if not presentes:
        return None
    df_melted = df_notas.melt(value_vars=presentes, var_name='disciplina_code', value_name='nota')
    df_melted['disciplina_nome'] = df_melted['disciplina_code'].apply(
        lambda c: get_simplified_name(c, disciplinas_dict))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='nota', y='disciplina_nome', data=df_melted, orient='h', ax=ax)
    ax.set_title(f'Dispersão de Notas por Disciplina - {nome_curso}', fontsize=16)
    ax.set_xlabel('Nota (0 a 20)', fontsize=12)
    ax.set_ylabel('Disciplina', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    fig.tight_layout()
    return fig


def grafico_disciplina_critica(df_notas, disciplina_code, disciplina_nome, nome_curso):
    if disciplina_code == "N/A" or disciplina_code not in df_notas.columns:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_notas[disciplina_code].dropna(), kde=True, bins=10, color='indianred', ax=ax)
    ax.set_title(f'Dispersão de Notas: {disciplina_nome} ({nome_curso})', fontsize=16)
    ax.set_xlabel('Nota na Disciplina (0 a 20)', fontsize=12)
    ax.set_ylabel('Número de Alunos', fontsize=12)
    ax.set_xlim(0, 20)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig


def gerar_todos_graficos(df_notas, nome_curso, disciplinas_dict, estatisticas):
    """Gera todas as figuras e devolve um dicionário {chave: Figure|None}."""
    return {
        'distribuicao_geral': grafico_distribuicao_notas(df_notas, nome_curso, disciplinas_dict),
        'media_disciplina': grafico_media_por_disciplina(df_notas, nome_curso, disciplinas_dict),
        'boxplot_disciplinas': grafico_boxplot_disciplinas(df_notas, nome_curso, disciplinas_dict),
        'disciplina_critica': grafico_disciplina_critica(
            df_notas,
            estatisticas.get('disciplina_menor_media_code', 'N/A'),
            estatisticas.get('disciplina_menor_media_nome', 'N/A'),
            nome_curso,
        ),
    }


def _fig_para_imagem(fig):
    """Converte uma figura matplotlib em BytesIO PNG para uso no reportlab."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf


# --------------------------------
# Relatório PDF (em memória)
# --------------------------------
def _cabecalho_factory(logo_path=None):
    def adicionar_cabecalho(canvas, doc):
        canvas.saveState()
        largura, altura = doc.pagesize

        if logo_path:
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


def criar_relatorio_pdf(nome_curso, estatisticas, figuras, logo_path=None):
    """Cria o relatório em PDF e devolve um BytesIO pronto para download."""
    buffer = io.BytesIO()
    doc = BaseDocTemplate(buffer, pagesize=A4)
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
    style_data = ParagraphStyle(name='DataCapa', fontSize=12, alignment=TA_CENTER,
                                spaceBefore=13 * cm, fontName='Times-Roman')
    style_h2 = ParagraphStyle(name='h2custom', parent=styles['h2'], fontName='Times-Bold')
    style_corpo = styles['Justify']

    story = []

    # Capa
    story.append(Spacer(1, 6 * cm))
    story.append(Paragraph("RELATÓRIO DE ACOMPANHAMENTO ACADÊMICO", style_titulo))
    story.append(Paragraph(f"Curso Técnico em {nome_curso.title()}", style_subtitulo))
    data_pt = format_date(datetime.now(), format="d 'de' MMMM 'de' y", locale='pt_BR')
    story.append(Paragraph(data_pt, style_data))
    story.append(PageBreak())

    # Estatísticas gerais
    story.append(Paragraph("Estatísticas Gerais da Turma", style_h2))
    story.append(Spacer(1, 0.5 * cm))
    dados_gerais = [
        ['Total de Alunos:', estatisticas['total_alunos']],
        ['Média Geral da Turma:', f"{estatisticas['media_geral_turma']:.2f}"],
        ['Desvio Padrão das Médias:', f"{estatisticas['desvio_padrao_medias']:.2f}"],
        ['Taxa de Aprovação (>= 12.0 em tudo):', estatisticas['taxa_aprovacao_geral']],
        ['Disciplina com Maior Média:', f"{estatisticas['disciplina_maior_media_nome']} ({estatisticas['maior_media']:.2f})"],
        ['Disciplina com Menor Média:', f"{estatisticas['disciplina_menor_media_nome']} ({estatisticas['menor_media']:.2f})"],
    ]
    tabela_geral = Table(dados_gerais, colWidths=[7 * cm, 9 * cm])
    tabela_geral.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
    ]))
    story.append(tabela_geral)
    story.append(Spacer(1, 1 * cm))

    # Alunos críticos
    story.append(Paragraph("Alunos com Maior Número de Disciplinas Abaixo da Média", style_h2))
    story.append(Spacer(1, 0.5 * cm))
    dados_criticos = [['Aluno', 'Disciplinas < 12']] + estatisticas['top_10_alunos_criticos'].values.tolist()
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

    # Gráficos gerais
    story.append(Paragraph("Visualizações Gráficas Gerais", style_h2))
    story.append(Spacer(1, 0.5 * cm))
    for chave in ['distribuicao_geral', 'media_disciplina', 'boxplot_disciplinas']:
        fig = figuras.get(chave)
        if fig is not None:
            story.append(Image(_fig_para_imagem(fig), width=16 * cm, height=11 * cm, kind='proportional'))
            story.append(Spacer(1, 1 * cm))

    # Resumo estatístico por disciplina
    story.append(Paragraph("Resumo Estatístico por Disciplina", style_h2))
    story.append(Spacer(1, 0.5 * cm))
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
    story.append(Spacer(1, 1 * cm))

    # Disciplina crítica
    fig_critica = figuras.get('disciplina_critica')
    if fig_critica is not None:
        story.append(PageBreak())
        story.append(Paragraph("Análise da Disciplina com Menor Desempenho", style_h2))
        story.append(Spacer(1, 0.5 * cm))
        dados_critica = [
            ["Disciplina:", estatisticas['disciplina_menor_media_nome']],
            ["Média da Turma:", f"{estatisticas['menor_media']:.2f}"],
            ["Desvio Padrão:", f"{estatisticas['desvio_padrao_disciplina_critica']:.2f}"],
            ["Alunos com Nota < 12.0:", f"{estatisticas['alunos_abaixo_12_disciplina_critica']}"],
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

    # Comentário da IA
    if estatisticas.get('comentario_ia'):
        story.append(PageBreak())
        story.append(Paragraph("Análise e Comentários (Gerado por Inteligência Artificial)", style_h2))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(estatisticas['comentario_ia'], style_corpo))

    doc.build(story)
    buffer.seek(0)
    return buffer
