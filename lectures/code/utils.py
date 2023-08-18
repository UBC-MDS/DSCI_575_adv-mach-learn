import pandas as pd
import numpy as np

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

def plot_components(W, image_shape): 
    fig, axes = plt.subplots(2, 5, figsize=(10, 4), subplot_kw={"xticks": (), "yticks": ()})
    for i, (component, ax) in enumerate(zip(W, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap="viridis")
        ax.set_title("{}. component".format((i)))
    plt.show()            