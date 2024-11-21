import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.ticker import MultipleLocator, FixedLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)

def make_pct_frame(
    size=5
):
    fontsize_large = 2.4*size
    fontsize_small = 1.6*size
    # freely inspired from: https://stackoverflow.com/questions/78369780/political-compass-graph-using-matplotlib
    fig, ax = plt.subplots(figsize=(size,size))

    # remove external spines
    for spine in ["left", "right", "bottom", "top"]:
        if spine in ["left", "bottom"]:
            ax.spines[spine].set_position("zero")
        else:
            ax.spines[spine].set_visible(False)

    # set limits
    lower_val, upper_val = -10, 10
    ax.tick_params(direction="inout", which="both")
    ax.set(xlim=(lower_val, upper_val), ylim=(lower_val, upper_val), xticks=[], yticks=[])

    # grid
    ticks = np.linspace(lower_val+1, upper_val, np.abs(lower_val)+np.abs(upper_val))
    ax.yaxis.set_minor_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator(ticks))
    ax.grid(True, which="minor")
    for ax_axis in [ax.xaxis, ax.yaxis]:
        for tick in ax_axis.get_minor_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)


    # add text labels to axes
    font_kwgs = {"fontsize": fontsize_large, "fontname": "Sans", "weight":"bold"}
    ax.text(0, upper_val, "Authoritarian", ha="center", **font_kwgs)
    ax.text(0, lower_val, "Libertarian", va="top", ha="center", **font_kwgs)
    ax.text(lower_val, 0, "Left", ha="right", va="center", **font_kwgs)
    ax.text(upper_val, 0, "Right", ha="left", va="center", **font_kwgs)
    font_kwgs = {"fontsize": fontsize_small, "fontname": "Sans", "style":"italic", "color":"blue"}
    ax.text(lower_val/2, 0, "←economic scale→", ha="center", **font_kwgs)
    ax.text(0, lower_val/2, "←social scale→", rotation=90, va="center", ha="right", **font_kwgs)
    # ←social scale→
    # ←economic scale→

    # color quadrants
    alpha_c = .8
    ll_c = (173/255, 235/255, 159/255, alpha_c)
    ul_c = (238/255, 125/255, 121/255, alpha_c)
    ur_c = (96/255, 168/255, 248/255, alpha_c)
    lr_c = (186/255, 156/255, 231/255, alpha_c)
    for offset, color in zip(
        [(lower_val, lower_val), (lower_val, 0), (0, 0), (0, lower_val)],
        [ll_c, ul_c, ur_c, lr_c],
    ):
        ax.add_patch(
            Polygon(
                [
                    offset,
                    (offset[0] + upper_val, offset[1]),
                    (offset[0] + upper_val, offset[1] + upper_val),
                    (offset[0], offset[1] + upper_val),
                ],
                facecolor=color,
            )
        )

    # equal aspects
    ax.set_aspect('equal', adjustable='box')

    return fig, ax

def place_tick(
    x_pos, y_pos,
    ax,
    marker_size=200,
    **kwargs
):
    ax.scatter(
        [x_pos], [y_pos],
        s=marker_size,
        #marker="o", s=200,
        #color="red", alpha=1,
        #edgecolors='black',
        zorder=10, clip_on=False,
        **kwargs
    )
    return ax

def place_image(
    im_path,
    x_pos, y_pos,
    ax,
    zoom=.05,
):
    # Display images instead!
    x = [x_pos]
    y = [y_pos]
    #ILLUSTRATIONS_PATH = "/Users/noedurandard/Desktop/llm_questionnaire/illustrations/"
    paths = [im_path]
    for x0, y0, path in zip(x, y,paths):
        ab = AnnotationBbox(
            getImage(path, zoom=zoom), 
            (x0, y0), 
            frameon=False, 
            zorder=10
        )
        ax.add_artist(ab)
    
    return ax