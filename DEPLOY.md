# 🚀 Como colocar o app no ar

Guia passo a passo para publicar a **Gestão Acadêmica EPTNM — CEFET-MG** no
**Streamlit Community Cloud** (gratuito) e configurar o envio de e-mails.

---

## 1. Pré-requisitos

- Conta no **GitHub** (o repositório já está em
  `https://github.com/d-camargo/gestao_tec-transito`).
- Uma conta **Gmail** para ser o *remetente* dos relatórios (de preferência uma
  conta Google Workspace institucional).

---

## 2. Gerar a "Senha de app" do Gmail

O app envia os relatórios por SMTP. O Gmail **não** aceita a senha normal da
conta — é preciso uma *senha de app* (16 caracteres):

1. Ative a **verificação em 2 etapas** na conta Google:
   <https://myaccount.google.com/security>
2. Acesse <https://myaccount.google.com/apppasswords>.
3. Crie uma senha de app (ex.: nome "Streamlit"). O Google mostra algo como
   `abcd efgh ijkl mnop`.
4. **Guarde essa senha** — você vai colá-la nos *secrets* (passo 4).

> Dica: crie/euse uma conta dedicada (ex.: `gestao.eptnm@...`) só para o envio,
> em vez da sua conta pessoal.

---

## 3. Publicar no Streamlit Community Cloud

1. Acesse <https://share.streamlit.io> e faça login com o GitHub.
2. Clique em **Create app → Deploy a public app from GitHub**.
3. Preencha:
   - **Repository:** `d-camargo/gestao_tec-transito`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Clique em **Deploy**. O Streamlit instala o `requirements.txt` sozinho
   (leva 1–3 minutos na primeira vez).

---

## 4. Configurar os *secrets* (credenciais)

No painel do app: **⋮ (menu) → Settings → Secrets**. Cole o conteúdo abaixo,
substituindo pelos seus valores:

```toml
# Envio de e-mail (obrigatório)
GMAIL_USER = "remetente@gmail.com"
GMAIL_APP_PASSWORD = "abcd efgh ijkl mnop"

# Análise por IA (opcional — só se for usar o comentário da IA)
OPENAI_API_KEY = "sk-..."
```

Salve. O app reinicia automaticamente e já estará pronto para enviar e-mails.

---

## 5. Testar

1. Abra a URL do app (algo como
   `https://gestao-tec-transito.streamlit.app`).
2. Informe um e-mail `@cefetmg.br`, o nome do curso e envie um mapa `.xls`.
3. Confira a caixa de entrada (e a pasta de spam) do e-mail informado.

---

## 6. Atualizações futuras

Todo `git push` para a branch `main` **redeploya o app automaticamente**.
Não precisa fazer nada no painel do Streamlit.

```bash
git add -A
git commit -m "minha alteração"
git push origin main
```

---

## 7. Usar um domínio próprio (custom domain)

**Resposta curta:** o **Streamlit Community Cloud (gratuito) não suporta domínio
próprio** — seu app fica sempre num endereço `*.streamlit.app`. Mas há caminhos
para usar o seu domínio:

### Opção A — Subdomínio "mascarado" (mais simples, com ressalvas)
Você pode criar um **redirecionamento** ou um *frame* a partir do seu domínio
(ex.: `gestao.seudominio.com` → `...streamlit.app`). É fácil, mas a barra de
endereço pode continuar mostrando o `streamlit.app` (no caso de redirect) e
*iframes* às vezes têm problemas com upload/websocket. Serve para um atalho
bonito, não para "esconder" totalmente a hospedagem.

### Opção B — Hospedar você mesmo no seu domínio (controle total) ✅
Para o domínio funcionar de verdade (URL própria + HTTPS), rode o app num
servidor/VPS que você controle e aponte o DNS para ele:

1. Suba o app num **VPS** (ex.: uma máquina Linux na nuvem) com Python.
2. Rode o Streamlit (ex.: `streamlit run app.py --server.port 8501`).
3. Coloque um **reverse proxy** (Nginx) na frente, com **HTTPS** via Let's
   Encrypt (Certbot).
4. No painel DNS do seu domínio, crie um registro **A** (ou **CNAME**) apontando
   `gestao.seudominio.com` para o servidor.

### Opção C — Plataforma que aceita domínio próprio
Plataformas como **Render**, **Railway**, **Fly.io** ou **Google Cloud Run**
hospedam apps Streamlit e permitem **adicionar um domínio próprio** (geralmente
via CNAME), muitas com plano gratuito ou de baixo custo. É o meio-termo entre a
facilidade do Streamlit Cloud e o controle do VPS.

> **Resumo:** dá para usar seu domínio, sim — só **não** no Streamlit Cloud
> gratuito de forma nativa. Para começar rápido, fique no `*.streamlit.app`;
> quando quiser o domínio próprio de verdade, vá de **Opção B ou C**.

---

Desenvolvido pelo Professor Diego Camargo — `diegocamargo@cefetmg.br`.
