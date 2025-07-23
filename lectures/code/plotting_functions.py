import numpy as np
import pandas as pd
import graphviz
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

def plot_img_caption_similarities(original_images, captions, similarity_np):
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 10], wspace=0.05)
    
    # Left column: image thumbnails
    ax_imgs = fig.add_subplot(gs[0])
    ax_imgs.axis("off")
    
    # Stack images into one vertical strip
    thumbs = [img.resize((60, 60)) for img in original_images]
    thumb_strip = np.vstack([np.asarray(thumb) for thumb in thumbs])
    ax_imgs.imshow(thumb_strip)
    ax_imgs.set_ylim(len(original_images)*60, 0)  # invert y-axis
    ax_imgs.set_xlim(0, 60)
    
    # Right column: similarity heatmap
    ax_heatmap = fig.add_subplot(gs[1])
    sns.heatmap(
        similarity_np,
        xticklabels=captions,
        yticklabels=False,  # no text ticks
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        ax=ax_heatmap,
        cbar_kws={"label": "Similarity"},
    )
    ax_heatmap.set_title("CLIP Image-Caption Similarity", fontsize=14)
    ax_heatmap.set_xlabel("Captions", fontsize=12)
    ax_heatmap.set_ylabel("Images", fontsize=12)
    ax_heatmap.set_yticks(np.arange(len(original_images)) + 0.5)
    ax_heatmap.set_yticklabels([""] * len(original_images))  # suppress matplotlib warnings
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def visualize_hmm(model, states=["s0", "s1"]):
    transmat_df = pd.DataFrame(model.transmat_, index=states, columns=states)

    # Get edges and their weights from the transition matrix
    edges = {}
    for col in transmat_df.columns:
        for idx in transmat_df.index:
            val = round(transmat_df.loc[idx, col], 3)
            if val > 0:
                edges[(idx, col)] = val

    # Create graph
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(states)  # nodes correspond to states
    print(f"Nodes:\n{graph.nodes()}\n")

    # edges represent transition probabilities
    for k, v in edges.items():
        tmp_source, tmp_dest = k[0], k[1]
        graph.add_edge(tmp_source, tmp_dest, weight=v, label=v)
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    # pos = nx.drawing.nx_pydot.graphviz_layout(graph, prog="dot")
    # nx.draw_networkx(graph, pos)

    edge_labels = {(n1, n2): d["label"] for n1, n2, d in graph.edges(data=True)}
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    dot_filename = "test.dot"
    nx.drawing.nx_pydot.write_dot(graph, dot_filename)
    with open(dot_filename) as f:
        dot_graph = f.read()
    return graphviz.Source(dot_graph)


import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

def plot_lda_w_vectors(W, component_labels, feature_names, width=800, height=600): 
    
    fig = px.imshow(
        W,
        y=component_labels,
        x=feature_names,
        color_continuous_scale="viridis",
    )

    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Principal Components",
        xaxis = {'side': 'top',  'tickangle':300}, 
    )
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
    )    

    return fig

# Source: https://github.com/amueller/mglearn/blob/master/mglearn/tools.py
def print_topics(topics, feature_names, sorting, topics_per_chunk=6,
                 n_words=20):
    for i in range(0, len(topics), topics_per_chunk):
        # for each chunk:
        these_topics = topics[i: i + topics_per_chunk]
        # maybe we have less than topics_per_chunk left
        len_this_chunk = len(these_topics)
        # print topic headers
        print(("topic {:<8}" * len_this_chunk).format(*these_topics))
        print(("-------- {0:<5}" * len_this_chunk).format(""))
        # print top n_words frequent words
        for i in range(n_words):
            try:
                print(("{:<14}" * len_this_chunk).format(
                    *feature_names[sorting[these_topics, i]]))
            except:
                pass
        print("\n")