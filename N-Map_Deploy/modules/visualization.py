import pandas as pd
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform

# Font Configuration for Matplotlib
def configure_fonts():
    """Configures Korean fonts for Matplotlib based on OS."""
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin':  # Mac
        plt.rc('font', family='AppleGothic')
    else:  # Linux (Streamlit Cloud usually)
        # Try to find a fallback if possible, or assume NanumGothic is installed
        try:
            plt.rc('font', family='NanumGothic')
        except:
            pass
    plt.rc('axes', unicode_minus=False)

def truncate_label(text, limit=12):
    """Truncates text to limit characters."""
    text = str(text)
    if len(text) > limit:
        return text[:limit] + "..."
    return text

def create_network_graph(rules):
    """
    Creates an interactive network graph using PyVis.
    Nodes: Items (Antecedents/Consequents)
    Edges: Association Rules (Thickness = Lift)
    """
    if rules.empty:
        return None

    # Initialize Network
    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='black')
    
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
        # Use full name as ID to link edges correctly, but display truncated label
        net.add_node(src, label=src_label, title=src, color='#1A374D', shape='dot') # Deep Blue
        net.add_node(dst, label=dst_label, title=dst, color='#406882', shape='dot') # Teal
        
        # Add edge
        # Width proportional to Lift, Label with Confidence
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
    # Skip Diagnosis for simplicity or keep if needed, lets focus on Intervention
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
    labels = []
    for label in all_nodes:
        text = label.split(':')[-1]
        labels.append(truncate_label(text))
    
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
          color = "#406882"
        ),
        link = dict(
          source = sources,
          target = targets,
          value = values,
          color = "#B1D0E0" # Light blue for flows
      ))])

    fig.update_layout(title_text="환자 흐름 (연령 → 수술시간 → 간호중재 [상위 15개])", font_size=12)
    return fig

def create_heatmap(df):
    """
    Creates a heatmap of Surgery Type vs Nursing Interventions.
    """
    if '수술시간_범주' not in df.columns or '간호중재' not in df.columns:
        return None
        
    # Crosstab
    ct = pd.crosstab(df['수술시간_범주'], df['간호중재'])
    
    # Select top N interventions if too many
    if ct.shape[1] > 15:
        top_interventions = ct.sum().sort_values(ascending=False).head(15).index
        ct = ct[top_interventions]
        
    # Truncate labels for X axis
    ct.columns = [truncate_label(c, 10) for c in ct.columns]
    
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', linewidths=.5)
    plt.title('수술 시간 유형별 주요 간호중재 빈도 (상위 15개)')
    plt.ylabel('수술 시간 유형')
    plt.xlabel('간호중재 (주요 항목)')
    
    return fig
