import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import google.generativeai as genai # Biblioteca ESTÁVEL

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Report Service Desk - ANBIMA", layout="wide", page_icon="📊")



# --- CONFIGURAÇÃO DA IA (MÉTODO ESTÁVEL) ---
ia_ativa = False
model_ia = None
try:
    # O Streamlit busca automaticamente no arquivo .streamlit/secrets.toml
    # ou nas "Secrets" do painel da nuvem (se fizer deploy)
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        
        # Instancia o modelo
        model_ia = genai.GenerativeModel('gemini-flash-latest')
        ia_ativa = True
    else:
        st.warning("⚠️ Chave de API não encontrada. Configure o secrets.toml.")

except Exception as e:
    st.error(f"Erro ao conectar na IA: {e}")
    ia_ativa = False

# --- CSS (ESTILO E IMPRESSÃO) ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    div[data-testid="stColumn"] {
        display: flex;
        align-items: center; 
        justify-content: center; 
    }
    
    /* Ajuste para imagens não ficarem gigantes */
    img[data-testid="stImage"] {
        object-fit: contain;
        max-height: 80px;
    }

    /* --- ESTILO DE IMPRESSÃO --- */
    @media print {
        [data-testid="stSidebar"], header, .stFileUploader, .stButton, .stDeployButton, [data-testid="stToolbar"], .stElementContainer:has(.stSelectbox), .stElementContainer:has(.stMultiSelect), .stElementContainer:has([data-testid="stPills"]) {
            display: none !important;
        }
        .block-container { padding: 0 !important; max-width: 100% !important; }
        [data-testid="stHorizontalBlock"] { display: block !important; margin-bottom: 50px !important; }
        [data-testid="stColumn"] { width: 100% !important; flex: none !important; display: block !important; break-inside: avoid !important; page-break-inside: avoid !important; margin-bottom: 40px !important; }
        .stPlotlyChart { break-inside: avoid !important; page-break-inside: avoid !important; }
        [data-testid="stDataFrame"] { margin-top: 80px !important; break-inside: auto !important; page-break-before: auto !important; display: block !important; }
    }
    </style>
""", unsafe_allow_html=True)

# --- CABEÇALHO ---
c1, c2, c3 = st.columns([1, 6, 1])
with c1:
    if os.path.exists("logo_anbima.png"): st.image("logo_anbima.png", width=100)
    else: st.write("")
with c2:
    st.markdown(f"<h1 style='text-align: center; color: #005c58; margin: 0; padding: 0; font-family: Helvetica, sans-serif;'>Report Service Desk - ANBIMA</h1>", unsafe_allow_html=True)
with c3:
    if os.path.exists("logo_gotobiz.png"): st.image("logo_gotobiz.png", width=100)
    else: st.write("")
st.divider()

# --- FUNÇÕES ---
def verificar_prazo(valor_tempo):
    if pd.isna(valor_tempo) or str(valor_tempo).strip() == '': return None
    return False if str(valor_tempo).strip().startswith('-') else True

def classificar_criador(nome):
    if pd.isna(nome): return 'Desconhecido'
    nome = str(nome).lower().strip()
    tecnicos = ['daniel costa', 'paulo reis', 'felipe souza', 'leandro jesus', 'marco costa', 'alessandro pereira', 'enos oliveira', 'hernan silva', 'paulo saraiva', 'guilherme silva', 'daniel souza']
    sistemas = ['pipefy', 'automation for jira', 'administrador do jira', 'jira infraestrutura', 'usuário de serviço', 'system']
    if nome in tecnicos: return 'Técnico'
    elif any(s in nome for s in sistemas): return 'Sistema/Serviço'
    else: return 'Usuário Final'

def get_location(analyst_name):
    return {'Daniel Costa': 'SP', 'Felipe Souza': 'SP', 'Leandro Jesus': 'SP', 'Paulo Reis': 'RJ', 'Marco Costa': 'SP', 'Alessandro Pereira': 'SP', 'Enos Oliveira': 'SP', 'Daniel souza': 'SP', 'Guilherme Silva': 'SP', 'Hernan Silva': 'RJ', 'Paulo Saraiva': 'SP'}.get(analyst_name, 'Outros')

def criar_tabela_top5(df, coluna):
    if df.empty or coluna not in df.columns: return pd.DataFrame()
    contagem = df[coluna].value_counts().head(5).reset_index()
    contagem.columns = [coluna, 'Qtd']
    contagem['%'] = (contagem['Qtd'] / len(df) * 100).map('{:.1f}%'.format)
    return contagem

# --- SIDEBAR ---
st.sidebar.header("📂 Importação")
uploaded_jira = st.sidebar.file_uploader("1. Arquivo do Jira (CSV/XLSX)", type=["csv", "xlsx"], key="jira")
uploaded_lan = st.sidebar.file_uploader("2. Arquivo Lansweeper (CSV/XLSX)", type=["csv", "xlsx"], key="lan")

# Indicador de Status da IA
if ia_ativa:
    st.sidebar.success("✅ IA Conectada")
else:
    st.sidebar.error("❌ IA Desconectada")

st.sidebar.markdown("---")
st.sidebar.header("📝 Dados Manuais")
qtd_rotinas = st.sidebar.number_input("Qtd. Rotinas", min_value=0, value=0)
pct_rotinas_slm = st.sidebar.slider("% Rotinas (Meta)", 0, 100, 100)
qtd_pesquisas_resp = st.sidebar.number_input("Nº Total de Respostas", min_value=0, value=0)
qtd_avaliacoes_positivas = st.sidebar.number_input("Nº Avaliações 5 Estrelas", min_value=0, value=0)

st.sidebar.subheader("Estoque RJ")
teclado_fio = st.sidebar.number_input("Teclado c/ fio", min_value=0, value=0)
teclado_sem_fio = st.sidebar.number_input("Teclado s/ fio", min_value=0, value=0)
mouse_fio = st.sidebar.number_input("Mouse c/ fio", min_value=0, value=0)
mouse_sem_fio = st.sidebar.number_input("Mouse s/ fio", min_value=0, value=0)
headset = st.sidebar.number_input("Headset", min_value=0, value=0)
estoque_rj_manual = teclado_fio + teclado_sem_fio + mouse_fio + mouse_sem_fio + headset

# --- PROCESSAMENTO ---
df_jira = None
if uploaded_jira:
    try:
        if uploaded_jira.name.endswith('.csv'): df_jira = pd.read_csv(uploaded_jira)
        else: 
            try: df_jira = pd.read_excel(uploaded_jira, sheet_name="your jira issues")
            except: df_jira = pd.read_excel(uploaded_jira, sheet_name=1)
        
        df_jira.columns = df_jira.columns.str.strip()
        if 'Status' in df_jira.columns: df_jira = df_jira[df_jira['Status'] != 'Cancelado']
        if 'Criado' in df_jira.columns: df_jira['Criado'] = pd.to_datetime(df_jira['Criado'], dayfirst=True, errors='coerce')
        
        if 'Criado' in df_jira.columns and not df_jira['Criado'].isnull().all():
            min_date, max_date = df_jira['Criado'].min().date(), df_jira['Criado'].max().date()
            st.sidebar.subheader("📅 Período Jira")
            c1, c2 = st.sidebar.columns(2)
            start_date = c1.date_input("Início", min_date)
            end_date = c2.date_input("Fim", max_date)
            df_jira_full = df_jira.copy() # Copia dos dados completos antes do filtro de data
            mask = (df_jira['Criado'].dt.date >= start_date) & (df_jira['Criado'].dt.date <= end_date)
            df_jira = df_jira.loc[mask].copy()
        
        df_jira['Localidade'] = df_jira['Responsável'].apply(get_location)
        col_criador = 'Criador' if 'Criador' in df_jira.columns else 'Creator'
        if col_criador in df_jira.columns: 
            df_jira['Categoria_Criador'] = df_jira[col_criador].apply(classificar_criador)
        else: 
            df_jira['Categoria_Criador'] = 'Desconhecido'
            
    except Exception as e: st.error(f"Erro Jira: {e}")

df_lan = None
if uploaded_lan:
    try:
        if uploaded_lan.name.endswith('.csv'): df_lan = pd.read_csv(uploaded_lan)
        else: df_lan = pd.read_excel(uploaded_lan)
        df_lan.columns = df_lan.columns.str.strip()
        if 'Model' not in df_lan.columns: df_lan['Model'] = "Desconhecido"
        if 'Statename' not in df_lan.columns: df_lan['Statename'] = "Desconhecido"
        modelos_excluir = ['nginx', 'Virtual Machine', 'VMware7,1', 't3a.large', 'Custom Pc']
        df_lan = df_lan[~df_lan['Model'].isin(modelos_excluir)]
        df_lan['Model'] = df_lan['Model'].replace('Pro 14 PC14250', 'Dell Pro 14 PC14250')
    except Exception as e: st.error(f"Erro Lansweeper: {e}")

# --- DASHBOARD ---
if df_jira is None and df_lan is None:
    st.info("👋 Faça o upload dos arquivos na barra lateral.")
    st.stop()

# --- CÁLCULOS GLOBAIS ---
if df_jira is not None:
    incidentes_df = df_jira[df_jira['Tipo de item'] == 'Incidente']
    requisicoes_df = df_jira[df_jira['Tipo de item'] != 'Incidente']
    total_chamados = len(df_jira)
    
    # SLA
    ok_resol, total_resol = 0, 0
    if 'Tempo de resolução' in df_jira.columns:
        df_jira['SLA_Res_Ok'] = df_jira['Tempo de resolução'].apply(verificar_prazo)
        total_resol = df_jira['SLA_Res_Ok'].notnull().sum()
        ok_resol = df_jira[df_jira['SLA_Res_Ok'] == True].shape[0]
    pct_resol = (ok_resol / total_resol * 100) if total_resol > 0 else 0
    
    # Backlog
    status_fechados = ['Resolvido', 'Fechado', 'Entrega Concluída', 'Devolução Concluída', 'Troca Concluída', 'Concluído']
    status_standby = ['Pendente Informações', 'Aguardando Aprovação', 'Aguardando Pendencia Linkada', 'Pendente Fornecedor', 'Aguardando Retirada da Máquina e Assinatura do Termo', 'Aguardando Devolução da Máquina']
    backlog_df = df_jira[(~df_jira['Status'].isin(status_fechados)) & (~df_jira['Status'].isin(status_standby))]
    qtd_backlog = len(backlog_df)
    pct_satisfacao = (qtd_avaliacoes_positivas / qtd_pesquisas_resp * 100) if qtd_pesquisas_resp > 0 else 0

# --- GERADOR DE RESUMO (IA) ---
def gerar_texto_com_gemini(prompt_text):
    if not ia_ativa or model_ia is None:
        return "⚠️ IA não configurada."
    try:
        response = model_ia.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"Erro na IA: {e}"

if df_jira is not None:
    with st.sidebar.expander("📝 Gerar Resumo", expanded=False):
        if st.button("Gerar"):
            with st.status("🤖 IA Processando...", expanded=True) as status:
                time.sleep(0.5)
                status.update(label="Concluído!", state="complete", expanded=False)
            
            if ia_ativa:
                top3 = incidentes_df['Request Type'].value_counts().head(3).to_dict() if not incidentes_df.empty else "N/A"
                
                # Preparando dados completos para a IA
                csv_jira = df_jira_full.to_csv(index=False) if 'df_jira_full' in locals() else df_jira.to_csv(index=False)
                csv_lan = df_lan.to_csv(index=False) if df_lan is not None else "Sem dados do Lansweeper."
                
                prompt_completo = f"""
                Analise os dados a seguir para criar um resumo executivo:
                
                --- DADOS DO JIRA (CHAMADOS) ---
                {csv_jira}
                
                --- DADOS DO LANSWEEPER (ATIVOS) ---
                {csv_lan}
                
                --- RESUMO ESTATÍSTICO (FILTRADO) ---
                Período: {start_date} a {end_date}. Total Filtrado: {total_chamados}. Incidentes: {len(incidentes_df)}. Backlog: {qtd_backlog}. SLA: {pct_resol:.1f}%. Satisfação: {pct_satisfacao:.1f}%.
                
                Instrução: Crie um resumo executivo formal em bullets com base em TODOS os dados fornecidos (considere o histórico completo, não apenas o filtrado, se relevante para tendências).
                """
                
                texto = gerar_texto_com_gemini(prompt_completo)
                st.markdown(texto)
            else:
                st.error("⚠️ Cole sua chave API no código.")
            st.balloons()

# --- ABAS PRINCIPAIS ---
tabs = st.tabs([t for t in ["Service Desk (Jira)", "📈 Análise de Tendências", "🖥️ Inventário (Lansweeper)", "🤖 Assistente Virtual"] if (df_jira is not None) or (df_lan is not None)])

# ABA 1: JIRA
if df_jira is not None:
    with tabs[0]:
        st.caption(f"Período: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
        
        ok_resp, total_resp = 0, 0
        if 'Tempo até a primeira resposta' in df_jira.columns:
            df_jira['SLA_Resp_Ok'] = df_jira['Tempo até a primeira resposta'].apply(verificar_prazo)
            total_resp = df_jira['SLA_Resp_Ok'].notnull().sum()
            ok_resp = df_jira[df_jira['SLA_Resp_Ok'] == True].shape[0]
        pct_resp = (ok_resp / total_resp * 100) if total_resp > 0 else 0
        
        base_aderencia = total_resol if total_resol > 0 else total_chamados
        pct_aderencia = (qtd_pesquisas_resp / base_aderencia * 100) if base_aderencia > 0 else 0

        st.subheader("🔹 Volumetria")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Geral", total_chamados + qtd_rotinas, border=True)
        k2.metric("Incidentes", len(incidentes_df), border=True)
        k3.metric("Requisições", len(requisicoes_df), border=True)
        k4.metric("Rotinas", qtd_rotinas, border=True)
        k5.metric("Chamados Jira", total_chamados, border=True)

        st.divider()
        st.subheader("🔹 Gestão de Nível de Serviço (SLM)")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("% Resp. Prazo", f"{pct_resp:.1f}%", f"{ok_resp} no prazo", border=True)
        s2.metric("% Resol. Prazo", f"{pct_resol:.1f}%", f"{ok_resol} no prazo", border=True)
        s3.metric("% Rotinas", f"{pct_rotinas_slm}%", "Meta", border=True)
        s4.metric("% Backlog (Atuando)", f"{len(backlog_df)/total_chamados*100:.1f}%" if total_chamados > 0 else "0%", f"{qtd_backlog} na fila", border=True, delta_color="inverse")

        st.divider()
        st.subheader("🔹 Qualidade e Localidade")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Satisfação", f"{pct_satisfacao:.1f}%", delta_color="normal" if pct_satisfacao >= 90 else "inverse", border=True)
        q2.metric("Aderência", f"{pct_aderencia:.1f}%", border=True)
        q3.metric("Volume SP", len(df_jira[df_jira['Localidade']=='SP']), border=True)
        q4.metric("Volume RJ", len(df_jira[df_jira['Localidade']=='RJ']), border=True)

        st.divider()
        st.subheader("🔹 Forma de Abertura (Origem)")
        if 'Categoria_Criador' in df_jira.columns:
            counts_criador = df_jira['Categoria_Criador'].value_counts()
            total_origem = counts_criador.sum()
            orig1, orig2, orig3 = st.columns(3)
            orig1.metric("Usuários Finais", counts_criador.get('Usuário Final', 0), f"{(counts_criador.get('Usuário Final', 0)/total_origem*100):.1f}%", border=True)
            orig2.metric("Técnicos (Interno)", counts_criador.get('Técnico', 0), f"{(counts_criador.get('Técnico', 0)/total_origem*100):.1f}%", border=True)
            orig3.metric("Sistemas/Robôs", counts_criador.get('Sistema/Serviço', 0), f"{(counts_criador.get('Sistema/Serviço', 0)/total_origem*100):.1f}%", border=True)

        st.divider()
        st.subheader("🚀 Planos de Ação & Estoque")
        if 'df_planos' not in st.session_state:
            st.session_state.df_planos = pd.DataFrame([{"Descrição": "Ex: Troca de Switch", "Status": "Em Andamento", "Local": "RJ", "% Conclusão": 50, "Responsável": "Paulo", "Início": None}])
        st.data_editor(st.session_state.df_planos, num_rows="dynamic", use_container_width=True, hide_index=True, column_config={"% Conclusão": st.column_config.NumberColumn(format="%d%%", min_value=0, max_value=100)})
        
        st.markdown("#### 📦 Detalhamento Estoque RJ")
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Teclado c/ Fio", teclado_fio, border=True)
        e2.metric("Teclado s/ Fio", teclado_sem_fio, border=True)
        e3.metric("Mouse c/ Fio", mouse_fio, border=True)
        e4.metric("Mouse s/ Fio", mouse_sem_fio, border=True)
        e5.metric("Headset", headset, border=True)

        st.divider()
        st.markdown("#### Detalhes Operacionais")
        t1, t2 = st.tabs(["Top 5 & Criadores", "Base Completa"])
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                st.error("🔥 Incidentes")
                if not incidentes_df.empty:
                    st.markdown("**Top Assuntos**")
                    if 'Request Type' in incidentes_df.columns: st.dataframe(criar_tabela_top5(incidentes_df, 'Request Type'), use_container_width=True, hide_index=True)
                    st.markdown("**Top Criadores**")
                    if col_criador in incidentes_df.columns: st.dataframe(criar_tabela_top5(incidentes_df, col_criador), use_container_width=True, hide_index=True)
            with c2:
                st.success("📋 Requisições")
                if not requisicoes_df.empty:
                    st.markdown("**Top Assuntos**")
                    if 'Request Type' in requisicoes_df.columns: st.dataframe(criar_tabela_top5(requisicoes_df, 'Request Type'), use_container_width=True, hide_index=True)
                    st.markdown("**Top Criadores**")
                    if col_criador in requisicoes_df.columns: st.dataframe(criar_tabela_top5(requisicoes_df, col_criador), use_container_width=True, hide_index=True)
        with t2: st.dataframe(df_jira)

# ABA 2: TENDÊNCIAS
if df_jira is not None:
    with tabs[1]:
        st.header("📈 Análise de Tendências")
        
        st.subheader("🌡️ Matriz de Calor: Horários de Pico")
        df_heat = df_jira.copy()
        df_heat['Dia_Semana'] = df_heat['Criado'].dt.day_name()
        df_heat['Hora'] = df_heat['Criado'].dt.hour
        dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_heat['Dia_Semana'] = pd.Categorical(df_heat['Dia_Semana'], categories=dias_ordem, ordered=True)
        heatmap_data = df_heat.groupby(['Dia_Semana', 'Hora']).size().reset_index(name='Chamados')
        
        fig_heat = px.density_heatmap(
            heatmap_data, x='Hora', y='Dia_Semana', z='Chamados', 
            nbinsx=24, text_auto=True, color_continuous_scale='Viridis', template="plotly_white"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.divider()
        
        st.subheader("📆 Evolução do Volume (Semana a Semana)")
        df_heat['Semana'] = df_heat['Criado'].dt.strftime('%U') 
        df_heat['Semana_Label'] = "Semana " + df_heat['Semana']
        trend_data = df_heat.groupby(['Semana_Label', 'Tipo de item']).size().reset_index(name='Volume')
        
        fig_trend = px.bar(
            trend_data, x='Semana_Label', y='Volume', color='Tipo de item', 
            barmode='group', text_auto=True, template="plotly_white",
            color_discrete_map={'Incidente': '#EF553B', 'Requisição': '#636EFA'}
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# ABA 3: LANSWEEPER
if df_lan is not None:
    idx = 2 if df_jira is not None else 0
    with tabs[idx]:
        st.header("🖥️ Gestão de Ativos")
        if 'Statename' in df_lan.columns:
            all_status = sorted(df_lan['Statename'].unique().astype(str))
            try: sel_status = st.pills("Filtrar por Status:", options=all_status, selection_mode="multi", default=all_status)
            except AttributeError: sel_status = st.multiselect("Filtrar por Status:", options=all_status, default=all_status)
            if not sel_status: df_lan_view = df_lan
            else: df_lan_view = df_lan[df_lan['Statename'].isin(sel_status)]
        else: df_lan_view = df_lan

        st.divider()
        c1, c2 = st.columns([1.2, 1])
        model_counts = df_lan_view['Model'].value_counts().reset_index()
        model_counts.columns = ['Modelo', 'Quantidade']
        
        with c1:
            st.subheader("📍 Selecione um Modelo")
            fig = px.bar(model_counts.head(20), x='Quantidade', y='Modelo', orientation='h', text='Quantidade', template="plotly_white", color_discrete_sequence=['#007bff'])
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, clickmode='event+select', plot_bgcolor='rgba(0,0,0,0)', dragmode=False)
            sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
        
        total = len(df_lan_view)
        if len(sel["selection"]["points"]) > 0:
            mod = sel["selection"]["points"][0]["y"]
            qtd = sel["selection"]["points"][0]["x"]
            
            df_table = df_lan_view[df_lan_view['Model'] == mod]
            df_rosca = pd.DataFrame({'Categoria': [mod, 'Outros'], 'Quantidade': [qtd, total-qtd], 'Cor': ['#005c58', '#e0e0e0']})
            title, center = f"Representatividade: {mod}", f"<b>{qtd}</b><br>({(qtd/total*100):.1f}%)"
        else:
            df_table = df_lan_view
            df_rosca = model_counts.head(5).copy()
            df_rosca.columns = ['Categoria', 'Quantidade']
            if total - df_rosca['Quantidade'].sum() > 0: df_rosca = pd.concat([df_rosca, pd.DataFrame({'Categoria':['Outros'], 'Quantidade':[total - df_rosca['Quantidade'].sum()]})])
            df_rosca['Cor'] = px.colors.qualitative.Safe[:len(df_rosca)]
            title, center = "Visão Geral", f"<b>{total}</b><br>Ativos"

        with c2:
            st.subheader("📊 Análise")
            fig_pie = go.Figure(data=[go.Pie(labels=df_rosca['Categoria'], values=df_rosca['Quantidade'], hole=0.6, textinfo='none', marker=dict(colors=df_rosca['Cor']))])
            fig_pie.update_layout(title=title, showlegend=True, legend=dict(orientation="h", y=-0.2), template="plotly_white", annotations=[dict(text=center, x=0.5, y=0.5, font_size=20, showarrow=False)], hovermode=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.divider()
        st.subheader("📂 Base de Ativos Detalhada")
        st.markdown("<br>", unsafe_allow_html=True) 
        st.dataframe(df_table, use_container_width=True)

# ABA 4: CHAT
if df_jira is not None:
    idx_chat = 3 if df_lan is not None else 2
    with tabs[idx_chat]:
        st.header("🤖 Assistente Virtual")
        if "messages" not in st.session_state: st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])
        if prompt := st.chat_input("Pergunte sobre os dados..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analisando..."):
                    if ia_ativa:
                        top3 = incidentes_df['Request Type'].value_counts().head(3).to_dict() if not incidentes_df.empty else "N/A"
                        
                        # Preparando dados completos para a IA
                        csv_jira = df_jira_full.to_csv(index=False) if 'df_jira_full' in locals() else df_jira.to_csv(index=False)
                        csv_lan = df_lan.to_csv(index=False) if df_lan is not None else "Sem dados do Lansweeper."
                        
                        prompt_completo = f"""
                        Você é um assistente de TI inteligente. Use os dados abaixo para responder à pergunta do usuário.
                        
                        --- DADOS DO JIRA (CHAMADOS - COMPLETO) ---
                        {csv_jira}
                        
                        --- DADOS DO LANSWEEPER (ATIVOS - COMPLETO) ---
                        {csv_lan}
                        
                        --- CONTEXTO ATUAL (FILTROS APLICADOS) ---
                        Total na tela: {total_chamados}. SLA: {pct_resol:.1f}%. Incidentes: {len(incidentes_df)}.
                        
                        Pergunta do usuário: {prompt}
                        """
                        
                        resposta = gerar_texto_com_gemini(prompt_completo)
                    else:
                        resposta = "IA desconectada."
                st.markdown(resposta)
            st.session_state.messages.append({"role": "assistant", "content": resposta})