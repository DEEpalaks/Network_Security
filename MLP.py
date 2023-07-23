import pandas as pd
import numpy as np
import sys
import sklearn
df = pd.read_csv("kdd_train.csv)
df_test = pd.read_csv("kdd_test.csv")
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
       for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len(df_test[col_name].unique())
       from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']
df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]
unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
dumcols=unique_protocol2 + unique_service2 + unique_flag2
unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2
df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)
enc = OneHotEncoder()
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_enc.toarray(),columns=testdumcols)
trainservice=df['service'].tolist()
testservice= df_test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
for col in difference:
    testdf_cat_data[col] = 0
newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)
newdf_test=df_test.join(testdf_cat_data)
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)
labeldf=newdf['label']
labeldf_test=newdf_test['label']
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test
to_drop_DoS = [2,3,4]
to_drop_Probe = [1,3,4]
to_drop_R2L = [1,2,4]
to_drop_U2R = [1,2,3]
DoS_df=newdf[~newdf['label'].isin(to_drop_DoS)];
Probe_df=newdf[~newdf['label'].isin(to_drop_Probe)];
R2L_df=newdf[~newdf['label'].isin(to_drop_R2L)];
U2R_df=newdf[~newdf['label'].isin(to_drop_U2R)];
DoS_df_test=newdf_test[~newdf_test['label'].isin(to_drop_DoS)];
Probe_df_test=newdf_test[~newdf_test['label'].isin(to_drop_Probe)];
R2L_df_test=newdf_test[~newdf_test['label'].isin(to_drop_R2L)];
U2R_df_test=newdf_test[~newdf_test['label'].isin(to_drop_U2R)];
X_DoS = DoS_df.drop('label',1)
Y_DoS = DoS_df.label
X_Probe = Probe_df.drop('label',1)
Y_Probe = Probe_df.label
X_R2L = R2L_df.drop('label',1)
Y_R2L = R2L_df.label
X_U2R = U2R_df.drop('label',1)
Y_U2R = U2R_df.label
X_DoS_test = DoS_df_test.drop('label',1)
Y_DoS_test = DoS_df_test.label
X_Probe_test = Probe_df_test.drop('label',1)
Y_Probe_test = Probe_df_test.label
X_R2L_test = R2L_df_test.drop('label',1)
Y_R2L_test = R2L_df_test.label
X_U2R_test = U2R_df_test.drop('label',1)
Y_U2R_test = U2R_df_test.label
colNames=list(X_DoS)
colNames_test=list(X_DoS_test)
from sklearn import preprocessing StandardScaler
X_DoS=StandardScaler().fit_transform(X_DoS) 
X_Probe=StandardScaler().fit_transform(X_Probe) 
X_R2L=StandardScaler().fit_transform(X_R2L) 
X_U2R=StandardScaler().fit_transform(X_U2R) 
X_DoS_test=StandardScaler().fit_transform(X_DoS_test) 
X_Probe_test=StandardScaler().fit_transform(X_Probe_test) 
X_R2L_test=StandardScaler().fit_transform(X_R2L_test) 
scaler8 = preprocessing.StandardScaler().fit(X_U2R_test)
X_U2R_test=scaler8.transform(X_U2R_test) 
X_Probe.std(axis=0);
X_R2L.std(axis=0);
from sklearn.feature_selection import SelectPercentile, f_classif
np.seterr(divide='ignore', invalid='ignore');
selector=SelectPercentile(f_classif, percentile=10)
X_newDoS = selector.fit_transform(X_DoS,Y_DoS)
from sklearn.feature_selection import SelectPercentile, f_classif
np.seterr(divide='ignore', invalid='ignore');
selector=SelectPercentile(f_classif, percentile=10)
X_newDoS = selector.fit_transform(X_DoS,Y_DoS)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,50,50), random_state=13,verbose=True, learning_rate_init=0.001,max_iter=200)
from sklearn.feature_selection import RFE
mlp.fit(X_newDoS, Y_DoS)
mlp.fit(X_newProbe, Y_Probe)
mlp.fit(X_newR2L, Y_R2L)
mlp.fit(X_newU2R, Y_U2R)
from sklearn.feature_selection import SelectPercentile, f_classif
np.seterr(divide='ignore', invalid='ignore');
selector=SelectPercentile(f_classif, percentile=10)
X_newDoS_test = selector.fit_transform(X_DoS_test,Y_DoS_test)
X_newDoS_test.shape
Y_DoS_pred=mlp.predict(X_newDoS_test)
pd.crosstab(Y_DoS_test, Y_DoS_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
X_newProbe_test = selector.fit_transform(X_Probe_test,Y_Probe_test)
X_newProbe_test.shape
Y_Probe_pred=mlp.predict(X_newProbe_test)
from sklearn.neural_network import MLPClassifier
mlp_DoS=MLPClassifier(random_state=42,)
mlp_Probe=MLPClassifier(random_state=42)
mlp_R2L=MLPClassifier(random_state=42)
mlp_U2R=MLPClassifier(random_state=42)
mlp_DoS.fit(X_DoS, Y_DoS)
mlp_Probe.fit(X_Probe, Y_Probe)
mlp_R2L.fit(X_R2L, Y_R2L)
mlp_U2R.fit(X_U2R, Y_U2R)
from sklearn.model_selection import cross_val_score
from sklearn import metrics
accuracy = cross_val_score(mlp_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(mlp_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(mlp_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(mlp_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
      plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
           else:
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(Y_DoS_test, Y_DoS_pred)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix  , classes=class_names, title='Confusion matrix')
plt.show()
cnf_matrix = confusion_matrix(Y_Probe_test, Y_Probe_pred)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()
cnf_matrix = confusion_matrix(Y_R2L_test, Y_R2L_pred)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix , classes=class_names , title='Confusion matrix')
plt.show()
cnf_matrix = confusion_matrix(Y_U2R_test, Y_U2R_pred)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix , classes=class_names , title='Confusion matrix')
plt.show()
X_newU2R_test = selector.fit_transform(X_U2R_test,Y_U2R_test)
X_newU2R_test.shape
Y_U2R_pred=mlp.predict(X_newU2R_test)
cnf_matrix = confusion_matrix(Y_U2R_test, Y_U2R_pred)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()
clf_Dos=MLPClassifier(hidden_layer_sizes=(100,50,50),random_state=13,verbose=True,learning_rate_init=0.001,max_iter=200)
clf_Dos.fit(X_DoS,Y_DoS)
clf_R2L=MLPClassifier(hidden_layer_sizes=(100,50,50),random_state=13,verbose=True,learning_rate_init=0.001,max_iter=200)
clf_R2L.fit(X_R2L,Y_R2L)
clf_Probe=MLPClassifier(hidden_layer_sizes=(100,50,50),random_state=13,verbose=True,learning_rate_init=0.001,max_iter=200)
clf_Probe.fit(X_Probe,Y_Probe)
clf_U2R=MLPClassifier(hidden_layer_sizes=(100,50,50),random_state=13,verbose=True,learning_rate_init=0.001,max_iter=200)
clf_U2R.fit(X_U2R,Y_U2R)
