import os
import seaborn as sns
import numpy as np
import pandas as pd
from extract_helpers import *
from feature_sel_util import *

import matplotlib.pyplot as plt
from skopt.searchcv import BayesSearchCV

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier



#------------------------Classifiers-------------------------
x, y = extract_all()
class_dist = []
plt.title("Distribution of Barriers(kj/mol)")
plt.xlabel("Barrier(kj/mol)")
plt.ylabel("Counts")
plt.hist(y,bins=30)
plt.show()

for i in y:
    if (float(i) < float(90)):
        class_dist.append([0])
    else:
        class_dist.append([1])
# this plus rf is pretty good
np.ravel(class_dist)

y = pd.DataFrame({"y":y})
data = x.join(y)

#y = data["y"].loc[data["y"] < 90]
#x = data.loc[data["y"] < 90].drop(columns = ['y'])

min = np.min(y)
max = np.max(y)
y_scale = (y - min) / (max - min)

importance_vars_v1 = [
    "DelSqV_6",
    "ESP_0", "ESP_4", "ESP_5",
    "ESPe_4", "ESPe_9", "ESPn_4",
    "G_0", "G_4", "GradRho_b_10",
    "HessRho_EigVals_a_9", "HessRho_EigVals_b_9", "HessRho_EigVals_c_6",
    "K|Scaled|_basic_1", "K|Scaled|_basic_3",
    "Lagr_basic_0",
    "NetCharge_5", "NetCharge_basic_1",
    "Rho_0",
    "Stress_EigVals_c_6",
    "Spin_tot_5",
    "V_9",  "Vnuc_0", "Vnuc_1", "Vnuc_2", "Vnuc_3", "Vnuc_9",
    "x_basic_4", "x_basic_5", "z_basic_3", "z_basic_4", "z_basic_5"]

print(len(importance_vars_v1))
#reduced_x = x[importance_vars_v1]
#reduced_x = scale(reduced_x)
#plt.matshow(reduced_x.corr())
#plt.colorbar()
#plt.show()

# plots selected variable correlation
#reduced_x = x[importance_vars_v1]
#corr = reduced_x.corr()
#ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0,
#    cmap=sns.diverging_palette(20, 220, n=200), square=False)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right', fontsize='small')
#plt.show()


importance_vars_v2 = \
    [
        "DelSqV_6",
        "ESP_0", "ESP_4", "ESP_5",
        "ESPe_4",
        "G_0", "GradRho_b_10",
        "K|Scaled|_basic_1", "K|Scaled|_basic_3",
        "Lagr_basic_0",
        "NetCharge_5", "NetCharge_basic_1",
        "Rho_0",
        "Vnuc_0", "Vnuc_1", "Vnuc_2", "Vnuc_3",
        "x_basic_4", "x_basic_5", "z_basic_3", "z_basic_4", "z_basic_5"]
reduced_x_1 = x[importance_vars_v1]
reduced_x_2 = x[importance_vars_v2]
corr = reduced_x_2.corr()
#ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=False)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right', fontsize='small')
#plt.show()

#print(corr)
# feature selection
# variance_thresh(x,y)
# -------------------------------------
#pca(x)
# 15 pca components has 82% explained variance
# 20 pca components has 87% explained variance
# 25 pca components has 90% explained variance
# ----------------Done and good
#lasso(x, y)
#boruta(x,y)
#recursive_feat_elim(x, y)

reduced_x_2 = scale(reduced_x_2)
reduced_x_1 = scale(reduced_x_1)

pca = PCA(0.90)
principal_components = pca.fit_transform(x)

# principal_df = pd.DataFrame(data = principal_components)
# principal_df

x_train, x_test, y_train, y_test = train_test_split(reduced_x_2, np.ravel(class_dist) , test_size=0.2)

#sklearn nns
#
#classifier = MLPClassifier(early_stopping = True, n_iter_no_change = 50, hidden_layer_sizes=(500,),
#                     solver = "adam", alpha = 0.002)
classifier = RandomForestClassifier(max_depth=10, random_state=0)
#classifier = LinearDiscriminantAnalysis()

classifier.fit(x_train, np.ravel(y_train))
#sklearn nn
y_pred = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
y_pred = classifier.predict(x_train)
print(accuracy_score(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))
y_prob = classifier.predict_proba(x_test)
preds = y_prob[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
