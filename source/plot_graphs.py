import seaborn as sns
sns.set_style("ticks")
sns.set_context("paper")
from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from extract_helpers_shifted import *
from mpl_toolkits.mplot3d import axes3d, Axes3D
matplotlib.rcParams.update({# Use mathtext, not LaTeX
                            'text.usetex': False,
                            # Use the Computer modern font
                            'font.family': 'serif',
                            'font.serif': 'cmr10',
                            'mathtext.fontset': 'cm',
                            # Use ASCII minus
                            'axes.unicode_minus': False,
                            'font.size': 20,
                            })

def label_rewrite(lbl_raw):
    lbls = []
    for i in lbl_raw:
        if ("basic" in i):
            i = i.replace("basic,", "")
        if (str.split(i, "\mathcal")[1][1:5] == "Bond"):
            if(str(i[-3]) == "0"):
                tmp = r"$\mathrm{\epsilon_{" + i[-4] + i[-3] + "}}$"
            else:
                tmp = r"$\mathrm{\epsilon_" + i[-3] + "}$"
            lbls.append(tmp)
        elif (str.split(i, "\mathcal")[1][1:4] == "Rho"):
            tmp = r"$\mathrm{\rho_{" + i[-3] + "}}$"
            lbls.append(tmp)
        elif (str.split(i, "\mathcal")[1][1:6] == "DelSq"):
            if (str.split(i, "\mathcal")[1][6] == "V"):
                tmp = r"$\mathrm{\nabla^2 V_{" + i[-3] + "}}$"
                lbls.append(tmp)
            else:
                tmp = r"$\mathrm{\nabla^2 \rho _{" + i[-3] + "}}$"
                lbls.append(tmp)
        elif (i == "$\mathcal{ESPe}_{10}$"):
            lbls.append("$\Phi^{e}_{10}$")
        elif ("Stress" in i):
            lbls.append("$\lambda^{(1)}_{\sigma, " + i[-3] + "}$")
        elif ("DelocInd" in i):
            lbls.append("$\delta_" + i[-3] + "}$")
        elif ("Vnuc" in i):
            lbls.append(r"$\tilde{\Phi}^{nuc}_{" + i[-3] + "}$")
        elif ("ESPn" in i):
            lbls.append("$\Phi^{nuc}_{" + i[-3] + "}$")
        elif ("{ESP}" in i):
            lbls.append("$\Phi_{" + i[-3] + "}$")
        elif (i[0:15] == "$\mathcal{Lagr}" or i[0:21] == "$\mathcal{Lagrangian}"):
            lbls.append("$\mathcal{L}_{" + i[-3] + "}$")
        elif ("HessRhoEigVals" in i):
            tmp = r"$\lambda^{(1)}_{\mathbf{H} \rho," + i[-3] + "}$"
            lbls.append(tmp)
        elif ("K" in i):
            lbls.append("$\mathcal{K}_{" + i[-3] + "}}$")
        else:
            lbls.append(r"$\mathrm" + str.split(i, "\mathcal")[1])
    return lbls

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

    '''
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
    '''

if __name__ == "__main__":
    x, y = extract_all()
    min = np.min(y)
    max = np.max(y)
    y_scale = (y - min) / (max - min)

    plt.hist(y)
    plt.title("Distribution of Dataset Energies")
    plt.xlabel("Energy [kJ/Mol]")
    plt.ylabel("Frequency")
    plt.show()

    pooled_set = \
        [
            "$\mathcal{Bond}_{7}$", "$\mathcal{Bond}_{8}$", "$\mathcal{Bond}_{9}$",
            "$\mathcal{DelocIndBond}_{5}$",
            "$\mathcal{DelSqRho}_{1}$",
            "$\mathcal{DelSqV}_{7}$",
            "$\mathcal{ESP}_{1}$", "$\mathcal{ESP}_{2}$", "$\mathcal{ESP}_{4}$", "$\mathcal{ESP}_{5}$",
            "$\mathcal{ESP}_{6}$",
            "$\mathcal{ESPe}_{10}$",
            "$\mathcal{ESPn}_{4}$", "$\mathcal{ESPn}_{5}$",
            "$\mathcal{HessRhoEigVals}_{c,7}$",
            "$\mathcal{K|Scaled|}_{basic,1}$", "$\mathcal{K|Scaled|}_{basic,2}$",
            "$\mathcal{K|Scaled|}_{basic,3}$", "$\mathcal{K|Scaled|}_{basic,4}$",
            "$\mathcal{K|Scaled|}_{basic,6}$",
            "$\mathcal{Kinetic}_{basic,5}$",
            "$\mathcal{Lagr}_{basic,1}$", "$\mathcal{Lagr}_{basic,5}$", "$\mathcal{Lagrangian}_{2}$",
            "$\mathcal{Stress_EigVals}_{c,7}$",
            "$\mathcal{Vnuc}_{1}$", "$\mathcal{Vnuc}_{2}$", "$\mathcal{Vnuc}_{3}$",
            "$\mathcal{Vnuc}_{4}$", "$\mathcal{Vnuc}_{5}$", "$\mathcal{Vnuc}_{6}$"
        ]

    pool_uncorr = \
        [
            "$\mathcal{Bond}_{7}$", "$\mathcal{Bond}_{8}$", "$\mathcal{Bond}_{9}$",
            "$\mathcal{DelocIndBond}_{5}$",
            "$\mathcal{DelSqRho}_{1}$",
            "$\mathcal{ESP}_{1}$", "$\mathcal{ESP}_{2}$", "$\mathcal{ESP}_{4}$", "$\mathcal{ESP}_{6}$",
            "$\mathcal{ESPn}_{5}$",
            "$\mathcal{HessRhoEigVals}_{c,7}$",
            "$\mathcal{K|Scaled|}_{basic,1}$", "$\mathcal{K|Scaled|}_{basic,2}$",
            "$\mathcal{K|Scaled|}_{basic,3}$", "$\mathcal{K|Scaled|}_{basic,4}$",
            "$\mathcal{Kinetic}_{basic,5}$", "$\mathcal{Kinetic}_{basic,6}$",
            "$\mathcal{Lagr}_{basic,1}$", "$\mathcal{Lagr}_{basic,5}$", "$\mathcal{Lagrangian}_{2}$",
            "$\mathcal{Vnuc}_{1}$", "$\mathcal{Vnuc}_{2}$", "$\mathcal{Vnuc}_{3}$",
            "$\mathcal{Vnuc}_{4}$", "$\mathcal{Vnuc}_{5}$", "$\mathcal{Vnuc}_{6}$"
        ]

    # physical set, general model
    physical = \
        [
            "$\mathcal{Bond}_{7}$", "$\mathcal{Bond}_{8}$", "$\mathcal{Bond}_{9}$", "$\mathcal{Bond}_{10}$",
            "$\mathcal{DelocIndBond}_{5}$",
            "$\mathcal{ESP}_{1}$", "$\mathcal{ESP}_{2}$", "$\mathcal{ESP}_{3}$", "$\mathcal{ESP}_{4}$",
            "$\mathcal{ESP}_{5}$", "$\mathcal{ESP}_{6}$",
            "$\mathcal{ESPn}_{4}$", "$\mathcal{ESPn}_{5}$",
            "$\mathcal{HessRhoEigVals}_{c,7}$",
            "$\mathcal{K|Scaled|}_{basic,1}$", "$\mathcal{K|Scaled|}_{basic,2}$", "$\mathcal{K|Scaled|}_{basic,3}$",
            "$\mathcal{K|Scaled|}_{basic,4}$", "$\mathcal{K|Scaled|}_{basic,5}$", "$\mathcal{K|Scaled|}_{basic,6}$",
            "$\mathcal{Vnuc}_{1}$", "$\mathcal{Vnuc}_{2}$", "$\mathcal{Vnuc}_{3}$", "$\mathcal{Vnuc}_{4}$",
            "$\mathcal{Vnuc}_{5}$",
            "$\mathcal{Vnuc}_{6}$"
        ]

    pool_x_df           = x[pooled_set]
    phys_x_df           = x[physical]
    pool_x_uncorr_df    = x[pool_uncorr]

    full_input = scale(x)
    pool_x          = scale(x[pooled_set].to_numpy())
    phys_x          = scale(x[physical].to_numpy())
    pool_x_uncorr   = scale(x[pool_uncorr].to_numpy())

    #--------------------------------------------------------------
    sns.axes_style("ticks")
    corr = pool_x_df.corr()
    lbls = label_rewrite(pool_x_df)
    ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0,  cmap=sns.diverging_palette(20, 220, n=200), square=True,
                     yticklabels=lbls, xticklabels=False, cbar_kws={'label': 'R'})
    ax.set_yticklabels(lbls, fontsize=16)
    ax.figure.axes[-1].yaxis.label.set_size(12)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    plt.title("Correlation, Pooled Descriptors", fontsize=16)
    plt.show()

    #--------------------------------------------------------------
    plot_corr = pool_x_df
    plot_corr["barrier"] = y_scale
    corr = np.array(plot_corr.corr()["barrier"].to_numpy()[0:-1])
    corr_barrier = plot_corr.corr()["barrier"].to_numpy()[0:-1]
    corr_barriers_labels = plot_corr.corr()["barrier"].keys()[0:-1]
    lbls = label_rewrite(corr_barriers_labels)
    ax = plt.subplot(1,1,1)
    plt.title("Pooled Descriptor Correlation vs. Barrier", fontsize=16)
    plt.xlabel("Correlation w/Barrier (R)", fontsize=16)
    ax.barh(range(np.shape(corr_barrier)[0]), corr_barrier,
            color = "peachpuff", edgecolor="k")

    ax.tick_params(labelsize=12)
    ax.set_yticklabels(lbls, rotation="0", fontsize=16)
    ax.set_yticks(np.arange(np.shape(corr_barriers_labels)[0]))
    plt.show()


