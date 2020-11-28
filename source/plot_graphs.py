import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from extract_helpers import *
from mpl_toolkits.mplot3d import axes3d, Axes3D

def pca(x, labels=[], barriers=np.array([])):
    pca = PCA(n_components=3)
    x = StandardScaler().fit_transform(x)
    pca.fit(x)
    result = pd.DataFrame(pca.transform(x), columns=['PCA%i' % i for i in range(3)])
    barriers = np.delete(barriers, np.argmax(result['PCA0']), 0)
    pc_0 = result['PCA0'].to_numpy()
    pc_1 = result['PCA1'].to_numpy()
    pc_2 = result['PCA2'].to_numpy()
    pc_0 = np.delete(pc_0, np.argmax(result['PCA0']), 0)
    pc_1 = np.delete(pc_1, np.argmax(result['PCA0']), 0)
    pc_2 = np.delete(pc_2, np.argmax(result['PCA0']), 0)

    min = np.min(barriers)
    max = np.max(barriers)
    mid = (min + max)/2
    norm = cm.colors.Normalize(vmin=np.min(barriers), vmax=np.max(barriers))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pl = ax.scatter(pc_0, pc_1, pc_2,
                    s=10, cmap="bwr",
                    c=[norm(item) for item in barriers],
                    alpha=0.5)
    ax.set_xlabel('PCA0')
    ax.set_ylabel('PCA1')
    ax.set_zlabel('PCA2')
    cbar = fig.colorbar(pl, cmap="bwr", ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.set_label("Barrier Energy")
    cbar.ax.set_yticklabels([str(min), str((min + mid)/2), str(mid), str((max+mid)/2), str(max)])
    plt.show()

    comp = []
    coeff_pca = []
    for i in pca.components_:
        for ind, j in enumerate(labels):
            if(np.absolute(i[ind]) > 0.1):
                comp.append(labels[ind])
                coeff_pca.append(i[ind])
        print(comp)
        print(len(comp))

        #print(np.sort(np.absolute(np.array(i))))
        #print(len(np.array(i)))
        # top n features
        #i = np.array(i)
        #ind = np.argsort(np.absolute(np.array(i))).tolist()
        #print(i[ind[-20:-1]])
        #for ind_temp in ind:
        #    print(i[ind_temp])
        ind = []
        comp = []

    #print(pca.components_)
    #print(pca.explained_variance_ratio_)
    #print(sum(pca.explained_variance_ratio_))

    fig, axs = plt.subplots(2,2)
    axs[0,0].scatter(pc_1, pc_2,
                    s=10, cmap="seismic",
                    c=[norm(item) for item in barriers],
                    alpha=0.3)
    axs[1,0].scatter(pc_0, pc_2,
                    s=10, cmap="seismic",
                    c=[norm(item) for item in barriers],
                    alpha=0.3)
    tmp = axs[0,1].scatter(pc_1, pc_0,
                    s=10, cmap="seismic",
                    c=[norm(item) for item in barriers],
                    alpha=0.3)

    axs[0,0].set(xlabel="PC1", ylabel='PC2')
    axs[1,0].set(xlabel="PC0", ylabel='PC2')
    axs[0,1].set(xlabel="PC1", ylabel='PC0')

    #fig.subplots_adjust(right=0.8)
    cbaxes = fig.add_axes([0.90, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(tmp, cmap="seismic", ticks=[0, 0.25, 0.5, 0.75, 1],
                        ax = axs[:,1], cax=cbaxes)

    cbar.set_label("Barrier Energy")
    cbar.ax.set_yticklabels([str(int(min)), str(int((min + mid) / 2)), str(int(mid)), str(int((max + mid) / 2)), str(int(max))])

    plt.tight_layout(rect=[0,0,0.9,1])
    fig.suptitle("2D Principal Components")
    fig.subplots_adjust(top=0.88)
    plt.show()


    variance = pca.explained_variance_ratio_
    var = np.cumsum(np.round(variance, decimals=3) * 100)
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots()
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('Explained Variability vs. # Eigenvectors')
    plt.ylim(30,100)
    plt.xlim(0,50)
    plt.style.context('seaborn-whitegrid')
    #plt.plot(var)
    sns.lineplot(range(len(var)), var)
    plt.hlines(95, linestyles= "dashdot", xmin=0, xmax=50)
    plt.hlines(90, linestyles= "dashdot", xmin=0, xmax=50)
    plt.hlines(85, linestyles= "dashdot", xmin=0, xmax=50)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(33, 80, "95% Variance Explained", fontsize=9,
            verticalalignment='top', bbox=props)
    ax.text(24, 75, "90% Variance Explained", fontsize=9,
            verticalalignment='top', bbox=props)
    ax.text(14, 70, "85% Variance Explained", fontsize=9,
            verticalalignment='top', bbox=props)
    ax.arrow(21,71, 0, 14, fc='black',             #arrow fill color
             ec='black')
    ax.arrow(30,76, 0, 14,fc='black',             #arrow fill color
             ec='black')
    ax.arrow(40,81, 0, 14, fc='black',             #arrow fill color
             ec='black')
    plt.show()



x, y = extract_all()
min = np.min(y)
max = np.max(y)
y_scale = (y - min) / (max - min)
sns.set_theme()

plt.hist(y)
plt.title("Distribution of Dataset Energies")
plt.xlabel("Energy (kJ/Mol)")
plt.ylabel("Counts")
plt.show()


# final trial with full dataset, pooled
importance_vars_v5 = \
    [
    "-DivStress_1","-DivStress_2", "-DivStress_3","-DivStress_6","-DivStress_8",
    "GradRho_a_10","GradRho_b_8","GradRho_c_7",
    "HessRho_EigVals_a_10","HessRho_EigVals_b_10","HessRho_EigVals_c_7",
    "K|Scaled|_basic_2","K|Scaled|_basic_3","K|Scaled|_basic_4","K|Scaled|_basic_5",
    "Lagr_basic_1","Lagr_basic_2","Lagr_basic_3","Lagr_basic_4","Lagr_basic_5","Lagr_basic_6",
    "V_10","Vnuc_1","Vnuc_2","Vnuc_3","Vnuc_4","Vnuc_5", "Vnuc_6",
    
    "ESP_1", "ESP_10", "ESP_2", "ESP_3", "ESP_4", "ESP_5", "ESP_6", "ESP_10",
    "ESPe_1", "ESPe_10",
    "ESPn_11", "ESPn_4", "ESPn_5", "ESPn_6",
    ]

# physical set - 1
importance_final_feats = \
    [
    "ESP_1","ESP_2","ESP_11","ESP_3","ESP_4","ESP_5","ESP_6","ESP_10",
    "ESPe_1","ESPe_10",
    "ESPn_11","ESPn_4","ESPn_5","ESPn_6",
    "K|Scaled|_basic_1","K|Scaled|_basic_2","K|Scaled|_basic_3","K|Scaled|_basic_5","Lagr_basic_6",
    "Lagr_basic_1","Lagr_basic_2","Lagr_basic_3","Lagr_basic_4","Lagr_basic_5",
    "Vnuc_1","Vnuc_2","Vnuc_3","Vnuc_4","Vnuc_5","Vnuc_6"
    ]


# final trial with full dataset, no correlation
importance_vars_v6 = \
    [
    "-DivStress_1","-DivStress_1","-DivStress_2","-DivStress_6","-DivStress_8",
    "ESP_1","ESP_2","ESP_3","ESP_4", "ESP_5","ESP_6","ESPe_1","ESPe_9",
    "GradRho_a_10","GradRho_b_8","GradRho_c_7",
    "K|Scaled|_basic_1","K|Scaled|_basic_2","K|Scaled|_basic_3","K|Scaled|_basic_5",
    "Lagr_basic_1","Lagr_basic_2","Lagr_basic_3","Lagr_basic_4","Lagr_basic_5","Lagr_basic_6",
    "Vnuc_1","Vnuc_2","Vnuc_3","Vnuc_4","Vnuc_5", "Vnuc_6"
]
# physical set, general model
physical = \
    [
    "ESP_1","ESP_2","ESP_3","ESP_4","ESP_5","ESP_6",
    "ESP_7", "ESP_8" ,"ESP_9","ESP_10"
    "ESPn_1", "ESPn_2", "ESPn_3", "ESPn_4","ESPn_5","ESPn_6"
    "K|Scaled|_basic_1","K|Scaled|_basic_2", "K|Scaled|_basic_3","K|Scaled|_basic_4","K|Scaled|_basic_5","K|Scaled|_basic_6"
    "Lagr_basic_1","Lagr_basic_2","Lagr_basic_3","Lagr_basic_4","Lagr_basic_5","Lagr_basic_6",
    "Vnuc_1","Vnuc_2", "Vnuc_3","Vnuc_4","Vnuc_5", "Vnuc_6"
]
# select subset of full dictionary

reduced_x_5_df = x[importance_vars_v5]
reduced_x_6_df = x[importance_vars_v6]
reduce_x_final_df = x[physical]
#  scales features down
reduced_x_physical = scale(reduce_x_final_df)
reduced_x_6 = scale(reduced_x_6_df)
reduced_x_5 = scale(reduced_x_5_df)



corr = reduce_x_final_df.corr()
ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0,  cmap=sns.diverging_palette(20, 220, n=200), square=True,
                 yticklabels=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=60, horizontalalignment='center', fontsize='x-small')
ax.set_yticklabels([i for i in reduce_x_final_df], rotation="0", fontsize = "x-small", va="center")
plt.title("Correlation, Physical Descriptors")
plt.show()

#plot_corr = reduced_x_3_df
#plot_corr["barrier"] = y_scale
#corr = np.array(plot_corr.corr()["barrier"].to_numpy()[0:-1])

#ax = plt.subplot(1,1,1)
#plt.title("Correlation Top Features vs. Barrier")
#ax.barh(range(np.shape(corr)[0]), corr)
#plt.xlabel("Correlation w/")
#print([str(i) for i in importance_vars_v3])
#ax.set_yticklabels([i for i in importance_vars_v3], rotation="0")
#ax.set_yticks(np.arange(len(importance_vars_v3)))
#plt.show()

#reduced_x = x[importance_vars_v4]
#corr = reduced_x.corr()
#ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0,
#    cmap=sns.diverging_palette(20, 220, n=200), square=False)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=70, horizontalalignment='right', fontsize='x-small')
#plt.show()

"""
corr = x.corr()
ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0,  cmap=sns.diverging_palette(20, 220, n=200), square=True,
                 yticklabels=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=60, horizontalalignment='center', fontsize='medium')
ax.set_yticklabels([i for i in x], rotation="0", fontsize = "x-small", va="center")
plt.title("Correlation, Selected Features")
plt.show()
"""

plot_corr = reduced_x_5_df
plot_corr["barrier"] = y_scale
corr = np.array(plot_corr.corr()["barrier"].to_numpy()[0:-1])
corr_barrier = plot_corr.corr()["barrier"].to_numpy()[0:-1]
corr_barriers_labels = plot_corr.corr()["barrier"].keys()[0:-1]


ax = plt.subplot(1,1,1)
plt.title("Compiled Descriptor Correlation vs. Barrier")
plt.xlabel("Correlation w/Barrier")
ax.barh(range(np.shape(corr_barrier)[0]), corr_barrier)
ax.set_yticklabels([i for i in corr_barriers_labels], rotation="0")
ax.set_yticks(np.arange(np.shape(corr_barriers_labels)[0]))
plt.show()


#pca(x,list(x), y)
# Box Plots
#fig1, ax1 = plt.subplots()
#V_Var =\
#[ "V_11","Vnuc_0","Vnuc_1","Vnuc_2","Vnuc_3","Vnuc_4","Vnuc_5"]
#esp =\
#[ "ESP_0","ESP_1","ESP_2","ESP_3","ESP_4","ESP_5","ESP_9"]
#v =  reduced_x_barrier_corr[V_Var]
#ESP = reduced_x_barrier_corr[esp]
#ax = sns.boxplot(data=v, orient="h", palette="Set2")
#plt.show()
#ax = sns.boxplot(data=ESP, orient="h", palette="Set2")
#plt.show()