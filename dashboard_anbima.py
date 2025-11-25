import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Dashboard TI - ANBIMA", layout="wide", page_icon="📊")

# --- CSS PARA ESTILO ---
st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# --- 1. FUNÇÕES DE NEGÓCIO ---

def verificar_prazo(valor_tempo):
    """
    Verifica se estourou o prazo baseado no sinal de negativo '-'
    Retorna: True (No Prazo), False (Atrasado), None (Vazio)
    """
    if pd.isna(valor_tempo) or str(valor_tempo).strip() == '':
        return None
    
    valor_str = str(valor_tempo).strip()
    # Se começa com '-', estourou o prazo
    if valor_str.startswith('-'):
        return False # FORA
    # Caso contrário, está DENTRO
    return True

def classificar_criador(nome):
    """
    Separa quem abriu o chamado em 3 categorias:
    1. Técnicos (Equipe TI)
    2. Sistema/Robôs (Pipefy, Jira, Automation)
    3. Usuários (Todos os outros)
    """
    if pd.isna(nome):
        return 'Desconhecido'
    
    nome = str(nome).lower().strip()
    
    # Lista de Técnicos
    tecnicos = [
        'daniel costa', 'paulo reis', 'felipe souza', 'leandro jesus', 
        'marco costa', 'alessandro pereira', 'enos oliveira', 'hernan silva', 
        'paulo saraiva', 'guilherme silva', 'daniel souza'
    ]
    
    # Lista de Robôs/Sistemas
    sistemas = [
        'pipefy', 'automation for jira', 'administrador do jira', 
        'jira infraestrutura', 'usuário de serviço', 'system'
    ]
    
    if nome in tecnicos:
        return 'Técnico'
    elif any(s in nome for s in sistemas): 
        return 'Sistema/Serviço'
    else:
        return 'Usuário Final'

def get_location(analyst_name):
    # Mapeamento para definir SP ou RJ
    ANALYST_MAP = {
        'Daniel Costa': 'SP', 'Felipe Souza': 'SP', 'Leandro Jesus': 'SP',
        'Paulo Reis': 'RJ', 'Marco Costa': 'SP', 'Alessandro Pereira': 'SP',
        'Enos Oliveira': 'SP', 'Daniel souza': 'SP', 'Guilherme Silva': 'SP',
        'Hernan Silva': 'RJ', 'Paulo Saraiva': 'SP'
    }
    return ANALYST_MAP.get(analyst_name, 'Outros')

# --- 2. BARRA LATERAL ---
st.sidebar.image("logo-anbima.png", width=180)
st.sidebar.header("📂 Dados do Jira")
uploaded_file = st.sidebar.file_uploader("Arquivo CSV", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.header("📝 Dados Manuais")

# Inputs separados para Quantidade (Volume) e Porcentagem (Meta SLM)
st.sidebar.subheader("Rotinas")
qtd_rotinas = st.sidebar.number_input("Qtd. Rotinas (Volume)", min_value=0, value=0)
pct_rotinas_slm = st.sidebar.slider("% Rotinas Executadas (Meta)", 0, 100, 100)

st.sidebar.subheader("Pesquisa de Satisfação")
qtd_pesquisas_resp = st.sidebar.number_input("Nº Pesquisas Respondidas", min_value=0, value=0)

st.sidebar.subheader("Outros")
estoque_rj = st.sidebar.number_input("Estoque RJ (Qtd)", value=0)

# --- 3. PROCESSAMENTO ---
if uploaded_file is not None:
    try:
        # Carregar
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        df.columns = df.columns.str.strip()

        # Remover Cancelados
        if 'Status' in df.columns:
            df = df[df['Status'] != 'Cancelado']

        # Datas
        if 'Criado' in df.columns:
            df['Criado'] = pd.to_datetime(df['Criado'], dayfirst=True, errors='coerce')
        
        # Filtro de Data
        if 'Criado' in df.columns and not df['Criado'].isnull().all():
            min_date = df['Criado'].min().date()
            max_date = df['Criado'].max().date()
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("📅 Período")
            c1, c2 = st.sidebar.columns(2)
            start_date = c1.date_input("Início", min_date)
            end_date = c2.date_input("Fim", max_date)
            
            mask = (df['Criado'].dt.date >= start_date) & (df['Criado'].dt.date <= end_date)
            df_filtered = df.loc[mask].copy()
        else:
            df_filtered = df.copy()

        # Enriquecimento de Dados
        df_filtered['Localidade'] = df_filtered['Responsável'].apply(get_location)
        
        # Classificar Origem
        col_criador = 'Criador' if 'Criador' in df_filtered.columns else 'Creator'
        if col_criador in df_filtered.columns:
            df_filtered['Categoria_Criador'] = df_filtered[col_criador].apply(classificar_criador)
        else:
            df_filtered['Categoria_Criador'] = 'Desconhecido'

        # --- 4. CÁLCULOS DE MÉTRICAS ---

        # 4.1 Volumetria (Incidentes vs Requisições)
        incidentes_df = df_filtered[df_filtered['Tipo de item'] == 'Incidente']
        requisicoes_df = df_filtered[df_filtered['Tipo de item'] != 'Incidente']
        
        qtd_incidentes = len(incidentes_df)
        qtd_requisicoes = len(requisicoes_df)
        
        # Total Geral Customizado (Incidente + Requisição + Rotina Manual)
        total_chamados_sistema = len(df_filtered)
        total_geral_KPI = total_chamados_sistema + qtd_rotinas

        # 4.2 SLA (Baseado no sinal '-')
        # SLA Resposta
        col_resp = 'Tempo até a primeira resposta'
        if col_resp in df_filtered.columns:
            df_filtered['SLA_Resp_Ok'] = df_filtered[col_resp].apply(verificar_prazo)
            total_resp_validos = df_filtered['SLA_Resp_Ok'].notnull().sum()
            qtd_resp_prazo = df_filtered[df_filtered['SLA_Resp_Ok'] == True].shape[0]
            
            pct_resp = (qtd_resp_prazo / total_resp_validos * 100) if total_resp_validos > 0 else 0
        else:
            qtd_resp_prazo = 0
            pct_resp = 0

        # SLA Resolução
        col_resol = 'Tempo de resolução'
        if col_resol in df_filtered.columns:
            df_filtered['SLA_Res_Ok'] = df_filtered[col_resol].apply(verificar_prazo)
            total_resol_validos = df_filtered['SLA_Res_Ok'].notnull().sum() # Conta só quem tem tempo preenchido
            qtd_resol_prazo = df_filtered[df_filtered['SLA_Res_Ok'] == True].shape[0]
            
            pct_resol = (qtd_resol_prazo / total_resol_validos * 100) if total_resol_validos > 0 else 0
        else:
            qtd_resol_prazo = 0
            total_resol_validos = 0
            pct_resol = 0

        # 4.3 Backlog
        status_fechados = [
            'Resolvido', 'Fechado', 'Entrega Concluída', 
            'Devolução Concluída', 'Troca Concluída', 'Concluído'
        ]
        backlog_df = df_filtered[~df_filtered['Status'].isin(status_fechados)]
        qtd_backlog = len(backlog_df)
        pct_backlog = (qtd_backlog / total_chamados_sistema * 100) if total_chamados_sistema > 0 else 0

        # 4.4 Satisfação (Média Ponderada)
        if 'Satisfaction' in df_filtered.columns:
            com_nota = df_filtered[df_filtered['Satisfaction'].notnull()]
            if not com_nota.empty:
                media_notas = com_nota['Satisfaction'].mean()
                pct_satisfacao = (media_notas / 5) * 100 
            else:
                pct_satisfacao = 0.0
        else:
            pct_satisfacao = 0.0

        # 4.5 Aderência (Manual / Total Resolvidos Válidos)
        # Usamos total_resol_validos ou len(df_filtered)? Geralmente aderência é sobre resolvidos.
        # Se total_resol_validos for 0, usamos o total do filtro para evitar erro.
        base_aderencia = total_resol_validos if total_resol_validos > 0 else len(df_filtered)
        if base_aderencia > 0:
            pct_aderencia = (qtd_pesquisas_resp / base_aderencia) * 100
        else:
            pct_aderencia = 0.0

        # --- 5. DASHBOARD VISUAL ---
        
        st.title("📊 Painel de Controle TI - Gestão Semanal")
        st.caption(f"Dados filtrados de {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
        st.divider()

        # BLOCO 1: VOLUMETRIA GERAL
        st.subheader("🔹 Volumetria")
        k1, k2, k3, k4, k5 = st.columns(5)
        
        k1.metric("Total Geral", total_geral_KPI, help="Incidentes + Requisições + Rotinas")
        k2.metric("Incidentes", qtd_incidentes, border=True)
        k3.metric("Requisições", qtd_requisicoes, border=True)
        k4.metric("Rotinas (Qtd)", qtd_rotinas, border=True)
        k5.metric("Chamados no Jira", total_chamados_sistema)

        # BLOCO 2: GESTÃO DO NÍVEL DE SERVIÇO - SLM (O PEDIDO PRINCIPAL)
        st.divider()
        st.subheader("🔹 GESTÃO DO NÍVEL DE SERVIÇO - SLM")
        
        slm1, slm2, slm3, slm4 = st.columns(4)
        
        slm1.metric(
            "% Respondidos no Prazo", 
            f"{pct_resp:.1f}%", 
            help="Cálculo automático baseado no sinal '-' do Jira"
        )
        
        slm2.metric(
            "% Resolvidos no Prazo", 
            f"{pct_resol:.1f}%", 
            f"{qtd_resol_prazo} no prazo",
            help="Cálculo automático baseado no sinal '-' do Jira"
        )
        
        slm3.metric(
            "% Rotinas Executadas", 
            f"{pct_rotinas_slm}%", 
            "Meta Manual",
            help="Valor inserido manualmente na barra lateral"
        )
        
        slm4.metric(
            "% Backlog", 
            f"{pct_backlog:.1f}%", 
            f"{qtd_backlog} abertos",
            delta_color="inverse",
            help="Chamados que não estão com status de Concluído/Resolvido"
        )

        # BLOCO 3: QUALIDADE & ACOMPANHAMENTO
        st.divider()
        st.subheader("🔹 Indicadores de Acompanhamento")
        q1, q2, q3, q4 = st.columns(4)
        
        sat_color = "normal" if pct_satisfacao >= 90 else "inverse"
        q1.metric("Satisfação (CSAT)", f"{pct_satisfacao:.1f}%", delta_color=sat_color)
        q2.metric("Aderência Pesquisa", f"{pct_aderencia:.1f}%", f"{qtd_pesquisas_resp} respostas")
        
        sp_vol = len(df_filtered[df_filtered['Localidade']=='SP'])
        rj_vol = len(df_filtered[df_filtered['Localidade']=='RJ'])
        q3.metric("Volume SP", sp_vol)
        q4.metric("Volume RJ", rj_vol)

        # BLOCO 4: QUEM ABRIU OS CHAMADOS?
        st.divider()
        st.subheader("🔹 Análise de Abertura (Origem)")
        
        counts_criador = df_filtered['Categoria_Criador'].value_counts()
        total_origem = counts_criador.sum()
        
        col_origem1, col_origem2 = st.columns([3, 2])
        
        with col_origem1:
            o1, o2, o3 = st.columns(3)
            # Usuários
            qtd_user = counts_criador.get('Usuário Final', 0)
            pct_user = (qtd_user / total_origem * 100) if total_origem > 0 else 0
            o1.metric("Usuários Finais", qtd_user, f"{pct_user:.1f}%")
            
            # Técnicos
            qtd_tec = counts_criador.get('Técnico', 0)
            pct_tec = (qtd_tec / total_origem * 100) if total_origem > 0 else 0
            o2.metric("Técnicos (Interno)", qtd_tec, f"{pct_tec:.1f}%")
            
            # Sistemas
            qtd_sys = counts_criador.get('Sistema/Serviço', 0)
            pct_sys = (qtd_sys / total_origem * 100) if total_origem > 0 else 0
            o3.metric("Sistemas/Robôs", qtd_sys, f"{pct_sys:.1f}%")
            
            # Gráfico de Barras Técnicos
            st.markdown("---")
            st.caption("Detalhamento: Chamados abertos pela própria equipe técnica")
            df_tecnicos = df_filtered[df_filtered['Categoria_Criador'] == 'Técnico']
            if not df_tecnicos.empty and col_criador in df_tecnicos.columns:
                count_tec = df_tecnicos[col_criador].value_counts().reset_index()
                count_tec.columns = ['Técnico', 'Qtd']
                st.bar_chart(count_tec.set_index('Técnico'), height=200)

        with col_origem2:
            fig_pizza = px.pie(
                names=counts_criador.index, 
                values=counts_criador.values, 
                title="Distribuição por Origem",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pizza, use_container_width=True)

        # BLOCO 5: PLANOS DE AÇÃO (TABELA EDITÁVEL)
        st.divider()
        st.subheader("🚀 Planos de Ação")
        
        # Inicializa DataFrame na sessão
        if 'df_planos' not in st.session_state:
            st.session_state.df_planos = pd.DataFrame(
                [{"Descrição": "Ex: Troca de Switch", "Status": "Em Andamento", "Local": "RJ", "% Conclusão": 50, "Responsável": "Paulo", "Início": None, "Obs": ""}]
            )

        edited_df = st.data_editor(
            st.session_state.df_planos,
            num_rows="dynamic",
            column_config={
                "% Conclusão": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
                "Status": st.column_config.SelectboxColumn(options=["Não Iniciado", "Em Andamento", "Concluído", "Pausado"], required=True),
                "Local": st.column_config.SelectboxColumn(options=["SP", "RJ", "Ambos"], required=True),
                "Início": st.column_config.DateColumn(format="DD/MM/YYYY")
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.caption(f"Estoque RJ atual: {estoque_rj} itens")

        # BLOCO 6: GRÁFICOS TOP 5
        st.divider()
        tab1, tab2 = st.tabs(["Top 5 Incidentes & Requisições", "Base Detalhada"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔥 Top 5 Incidentes")
                if 'Tipo de item' in df_filtered.columns:
                    incs = df_filtered[df_filtered['Tipo de item'] == 'Incidente']
                    if not incs.empty and 'Request Type' in incs.columns:
                        top_incs = incs['Request Type'].value_counts().head(5).reset_index()
                        top_incs.columns = ['Tipo', 'Qtd']
                        st.dataframe(top_incs, hide_index=True, use_container_width=True)
                    else:
                        st.info("Sem dados de incidentes.")

            with c2:
                st.markdown("#### 📋 Top 5 Requisições")
                if 'Tipo de item' in df_filtered.columns:
                    reqs = df_filtered[df_filtered['Tipo de item'] != 'Incidente']
                    if not reqs.empty and 'Request Type' in reqs.columns:
                        top_reqs = reqs['Request Type'].value_counts().head(5).reset_index()
                        top_reqs.columns = ['Tipo', 'Qtd']
                        st.dataframe(top_reqs, hide_index=True, use_container_width=True)
                    else:
                        st.info("Sem dados de requisições.")

        with tab2:
            st.dataframe(df_filtered)

    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
else:
    st.info("👋 Faça o upload do arquivo CSV do Jira na barra lateral para começar.")