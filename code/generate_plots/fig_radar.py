import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from math import pi


def radar_plot(values, color_line, color_fill, filename):
    # Set data
    # values = [94.74, 49.06, 50.0, 87.5, 66.03, 75.68]
    values += values[:1]
    values

    categories = ["Sparing\nhumans", "Sparing\nthe young", "Sparing\nthe fit", "Sparing\nfemales", "Sparing\nhigher status", "Sparing\nmore" ]

    N = 6
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    plt.figure(figsize=(22,15))
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='black', size=6, fontsize=97)
    rstep = int(ax.get_theta_direction())
    if rstep > 0:
        rmin = 0
        rmax = len(angles)
    else:
        rmin = len(angles)-1
        rmax = -1

    for label, i in zip(ax.get_xticklabels(), range(rmin, rmax, rstep)):
        angle_rad = angles[i] + ax.get_theta_offset()
        if angle_rad == 0:
            ha = 'left'
            va = "center"
        elif angle_rad == pi:
            ha = 'right'
            va = "center"
        elif angle_rad <= pi / 2:
            ha = 'left'
            va = "bottom"
        elif pi / 2 < angle_rad <= pi:
            ha = 'right'
            va = "bottom"
        elif pi < angle_rad <= (3 * pi / 2):
            ha = 'right'
            va = "top"
        else:
            ha = 'left'
            va = "top"
        label.set_verticalalignment(va)
        label.set_horizontalalignment(ha)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([20,40,60,80],labels=None, color="grey", size=10, fontsize=0)
    plt.ylim(0,100)
    
    for tick in ax.get_xticklabels():
        tick.set_fontname("Arial")
        
    # Plot data
    ax.plot(angles, values, linewidth=10, linestyle='solid', color=color_line)
    
    # Fill area
    ax.fill(angles, values, color=color_fill, alpha=0.6)
    plt.tight_layout(pad=5)
    # Show the graph
    # plt.show()
    plt.savefig(filename, dpi=300)
    




def main():
   df = pd.read_csv("model_preferences_by_lang.csv")
   df = df[["criterion", "de", "zh-cn", "sw"]] 
   print(list(df["de"]))
   radar_plot(list(df["de"]), color_line="#9491E2",  color_fill="#99A1E0", filename="fig_radar_de.pdf")
   radar_plot(list(df["zh-cn"]), color_line="#7B9E87" , color_fill="#98b4a2" , filename="fig_radar_zh.pdf")
   radar_plot(list(df["sw"]), color_line="#E0A579", color_fill = "#EEB86D", filename="fig_radar_sw.pdf" )

if __name__ == "__main__":
    main()