import numpy  as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.layers import Dropout

seed=13
np.random.seed()
c="battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time,three_g,touch_screen,wifi,price_range".split(',')


train=pandas.read_csv("C:\\Users\\Anas\Downloads\Philips\MobilePriceClassification\\train.csv",header=0)
train=train
y=train.price_range
X=train.loc[:,train.columns != 'price_range'].values
test=pandas.read_csv("C:\\Users\\Anas\Downloads\Philips\MobilePriceClassification\\test.csv",header=0)
test_X=test.loc[:,test.columns != 'id'].values



from sklearn.feature_selection import VarianceThreshold
# Features are in train and labels are in train_labels
sel = VarianceThreshold(threshold=(.20))
X=sel.fit_transform(X)
test_X=sel.fit_transform(test_X)
print("X",X.shape)
print("test",test_X.shape)


#print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=31)


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

sc=StandardScaler()
# minmax=MinMaxScaler()
# norm=Normalizer()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
test_X=sc.transform(test_X)

#print(X_train,X_test)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)
# X_train=pca.fit_transform(X_train)
# X_test=pca.fit(X_test)
# print(pca.explained_variance_ratio_)



from sklearn.linear_model import SGDClassifier
# from sklearn import grid_search
# clf = SGDClassifier(random_state=0, class_weight='balanced')
# clf.fit(X_train,y_train)
# y_pred=clf.predict(X_test)

# from sklearn.ensemble import RandomForestClassifier
# forest_clf=RandomForestClassifier()
# forest_clf.fit(X_train,y_train)
# y_pred=forest_clf.predict(X_test)

print(X_train)
print(y_train)
from sklearn.multiclass import OneVsOneClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
import parfit.parfit as pf

grid = {
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate
    'n_iter': [1000], # number of epochs
    'loss': ['log'], # logistic regression,
    'penalty': ['l2'],
    'n_jobs': [-1]
}
paramGrid = ParameterGrid(grid)

bestModel, bestScore, allModels, allScores = pf.bestFit(SGDClassifier(random_state=42), paramGrid,
           X_train, y_train, X_test, y_test,metric = roc_auc_score, greater_is_better=True,scoreLabel = 'AUC',n_jobs=-1)

print(bestModel, bestScore)










from sklearn import svm
from sklearn.linear_model import Lasso,ElasticNet

clf=OneVsOneClassifier(SGDClassifier(random_state=42))
# clf=OneVsOneClassifier(svm.SVC(random_state=42))

# clf=svm.SVC(gamma='scale',decision_function_shape='ovo')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
test_pred=clf.predict(test_X)
print(test_pred)
#
# acc = accuracy_score(y_test, y_pred, normalize = True)
#
# print("acc",acc)
#
# print(f1_score(y_test, y_pred, average="macro"))
# print(precision_score(y_test, y_pred, average="macro"))
# print(recall_score(y_test, y_pred, average="macro"))
#
# # df=pandas.DataFrame(test_pred)
#
# # print(df)
#
# # df.to_csv("result.csv")