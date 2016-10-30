import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

def get_data_x_cat(reduced_ds):
    data_x_cat = defaultdict(dict)
    for data in reduced_ds:
        cat = int(data[0])
        if not data_x_cat[cat]:
            data_x_cat[cat] = defaultdict(list)

        for i in range(1,len(reduced_ds[0])):
            data_x_cat[cat][i].append(data[i])
    return data_x_cat


def plot3d(reduced_ds_train, reduced_ds_val):
    colors = ['b', 'g', 'r', 'c', 'm', 'gold', 'k', 'pink', '0.8']        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data_x_cat = get_data_x_cat(reduced_ds_train)

    for c in range(0,9):
        ax.scatter(data_x_cat[c+1][1], data_x_cat[c+1][2], data_x_cat[c+1][3], marker='o', s=30, c=colors[c],label='cat'+ str(c + 1) + ' train', edgecolors='none')

    data_x_cat = get_data_x_cat(reduced_ds_val)

    for c in range(0,9):
        ax.scatter(data_x_cat[c+1][1], data_x_cat[c+1][2], data_x_cat[c+1][3], marker='^', s=30, c=colors[c], label='cat'+ str(c + 1) + ' val',edgecolors='none')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    plt.legend(numpoints=1,ncol=6,shadow=True, fancybox=True)

    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()  # all the text.Text instance in the legend
    llines = leg.get_lines()  # all the lines.Line2D instance in the legend
    frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend

    plt.setp(ltext, fontsize='small')    # the legend text fontsize
    plt.setp(llines, linewidth=1.5)      # the legend linewidth

    plt.show()

def plot2d(data_x_cat, data_x_cat_val, ax, pc1, pc2):
    colors = ['b', 'g', 'r', 'c', 'm', 'gold', 'k', 'pink', '0.8']

    for c in range(0,9):
        ax.scatter(data_x_cat[c+1][pc1], data_x_cat[c+1][pc2], marker='o', color=colors[c], label='c'+ str(c + 1) + ' training')
        ax.scatter(data_x_cat_val[c+1][pc1], data_x_cat_val[c+1][pc2], marker='^', color=colors[c], label='c'+ str(c + 1) + ' validation')
        pass
    
    ax.set_xlabel('PC'+str(pc1))
    ax.set_ylabel('PC'+str(pc2))
    
