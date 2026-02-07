import pandas as pd
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform

# Font Configuration for Matplotlib (Fallback)
def configure_fonts():
    """Configures Korean fonts for Matplotlib based on OS."""
    # 1. Try to load bundled font (robust for cloud)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(current_dir, '..', 'assets', 'font.ttf')
    
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            font_prop = fm.FontProperties(fname=font_path)
            plt.rc('font', family=font_prop.get_name())
            plt.rc('axes', unicode_minus=False)
            return
        except Exception:
            pass

    # 2. Fallback to system fonts
    try:
        import koreanize_matplotlib
    except ImportError:
        pass

    system_name = platform.system()
    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin':  # Mac
        plt.rc('font', family='AppleGothic')
    
    plt.rc('axes', unicode_minus=False)

def truncate_label(text, limit=15):
    """Truncates text to limit characters."""
    text = str(text)
    if len(text) > limit:
        return text[:limit] + ".."
    return text

def create_network_graph(rules):
    """
    Creates an interactive network graph using PyVis.
    Nodes: Items (Antecedents/Consequents)
    Edges: Association Rules (Thickness = Lift)
    """
    if rules.empty:
        return None

    # Initialize Network (Increased height)
    net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='black')
    
    # Add nodes and edges
    for _, row in rules.iterrows():
        src = row['antecedents_str']
        dst = row['consequents_str']
        
        # Truncate labels for graph clarity, but keep full info in title (tooltip)
        src_label = truncate_label(src)
        dst_label = truncate_label(dst)
        
        weight = row['lift']
        conf = row['confidence']
        
        # Add nodes if not exist
        # Use full name as ID to link edges correctly
        # title argument provides the tooltip on hover
        net.add_node(src, label=src_label, title=src, color='#1A374D', shape='dot') 
        net.add_node(dst, label=dst_label, title=dst, color='#406882', shape='dot') 
        
        # Add edge
        net.add_edge(src, dst, value=weight, title=f"향상도(Lift): {weight:.2f}\n신뢰도(Conf): {conf:.2f}", color='#B1D0E0')

    # Physics options for better layout
    net.force_atlas_2based()
    net.show_buttons(filter_=['physics'])
    
    return net

def create_sankey_diagram(df):
    """
    Creates a Sankey diagram showing flow: Age -> Surgery -> Interventions.
    Limits to Top 15 Interventions to prevent clutter.
    """
    cols = []
    if '연령대' in df.columns: cols.append('연령대')
    if '수술시간_범주' in df.columns: cols.append('수술시간_범주')
    if '간호중재' in df.columns: 
        cols.append('간호중재')
        
        # Filter Data: Keep only Top 15 Interventions
        top_interventions = df['간호중재'].value_counts().head(15).index
        df = df[df['간호중재'].isin(top_interventions)].copy()
    
    if len(cols) < 2:
        return None

    # Create source-target pairs
    sources = []
    targets = []
    values = []
    
    # Process Nodes
    all_nodes = []
    for col in cols:
        unique_vals = df[col].astype(str).unique()
        all_nodes.extend([f"{col}:{val}" for val in unique_vals])
        
    node_map = {label: i for i, label in enumerate(all_nodes)}
    
    # Generate Display Labels (Remove prefix + Truncate)
    # Full label is stored in customdata for hover
    labels = []
    full_labels = []
    for label in all_nodes:
        text = label.split(':')[-1]
        labels.append(truncate_label(text))
        full_labels.append(text)
    
    # Aggregate flows
    for i in range(len(cols)-1):
        src_col = cols[i]
        dst_col = cols[i+1]
        
        counts = df.groupby([src_col, dst_col]).size().reset_index(name='count')
        
        for _, row in counts.iterrows():
            src_label = f"{src_col}:{row[src_col]}"
            dst_label = f"{dst_col}:{row[dst_col]}"
            
            sources.append(node_map[src_label])
            targets.append(node_map[dst_label])
            values.append(row['count'])
            
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          customdata = full_labels,
          hovertemplate='%{customdata}<br>빈도: %{value}<extra></extra>',
          color = "#406882"
        ),
        link = dict(
          source = sources,
          target = targets,
          value = values,
          color = "#B1D0E0" 
      ))])

    fig.update_layout(
        title_text="환자 흐름 (연령 → 수술시간 → 간호중재 [상위 15개])", 
        font_size=12,
        height=700  # Increased height
    )
    return fig

def create_heatmap(df):
    """
    Creates an interactive Heatmap using Plotly.
    X: Nursing Interventions (Top 20)
    Y: Surgery Time Category
    """
    if '수술시간_범주' not in df.columns or '간호중재' not in df.columns:
        return None
        
    # Crosstab
    ct = pd.crosstab(df['수술시간_범주'], df['간호중재'])
    
    # Select top N interventions if too many
    if ct.shape[1] > 20:
        top_interventions = ct.sum().sort_values(ascending=False).head(20).index
        ct = ct[top_interventions]
        
    # Prepare Labels
    full_labels = ct.columns.tolist() # Keep full names for hover
    short_labels = [truncate_label(c, 12) for c in full_labels] # Truncate for X-axis
    
    # Plotly Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=ct.values,
        x=short_labels, # Use truncated labels
        y=ct.index,
        customdata=full_labels, # Store full labels
        colorscale='Blues',
        colorbar=dict(title='빈도'),
        hovertemplate='<b>수술시간</b>: %{y}<br><b>간호중재</b>: %{customdata}<br><b>빈도</b>: %{z}회<extra></extra>' # Show full label on hover
    ))
    
    fig.update_layout(
        title='수술 시간 유형별 주요 간호중재 빈도 (상위 20개)',
        xaxis_title='간호중재 (주요 항목)',
        yaxis_title='수술 시간 유형',
        height=750, # Increased height significantly
        xaxis=dict(tickangle=-45) # Rotate labels for better readability
    )
    
    return fig
