import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy import arange
from statsmodels.graphics.mosaicplot import mosaic
#----call data----
df = pd.read_csv ("heart_data.csv")
#----ageapriorX2-----
plt.hist(df['age'], bins=40, color='darkblue')
plt.title("Age Histogram", fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(arange(2200, step=100))
plt.show()

#----genderapriorX3----
genderprob = df['gender'].value_counts() / df['gender'].shape[0]
print (genderprob)
sns.countplot(x='gender', data=df)
plt.title("Gender BarPlot", fontsize=20)
plt.xlabel('Gender', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#----cpapriorX4----
cpprob = df['cp'].value_counts() / df['cp'].shape[0]
print (cpprob)
sns.countplot(x='cp', data=df)
plt.title("cp BarPlot", fontsize=20)
plt.xlabel('cp', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#----trestbpsapriorX5----
plt.hist(df['trestbps'], bins=20, color='darkblue')
plt.title("trestbps Histogram", fontsize=20)
plt.xlabel('trestbps', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(arange(200, step=10))
plt.show()

#----cholapriorX6----
plt.hist(df['chol'], bins=30, color='darkblue')
plt.title("chol Histogram", fontsize=20)
plt.xlabel('chol', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#----fbsapriorX7----
fbsprob = df['fbs'].value_counts() / df['fbs'].shape[0]
print (cpprob)
sns.countplot(x='fbs', data=df)
plt.title("fbs BarPlot", fontsize=20)
plt.xlabel('fbs', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#----restecgapriorX8----
restecprob = df['restecg'].value_counts() / df['restecg'].shape[0]
print (cpprob)
sns.countplot(x='restecg', data=df)
plt.title("restecg BarPlot", fontsize=20)
plt.xlabel('restecg', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#----thalachapriorX9----
plt.hist(df['thalach'], bins=30, color='darkblue')
plt.title("thalach Histogram", fontsize=20)
plt.xlabel('thalach', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#----exangapriorX10----
exangprob = df['exang'].value_counts() / df['exang'].shape[0]
print (cpprob)
sns.countplot(x='exang', data=df)
plt.title("exang BarPlot", fontsize=20)
plt.xlabel('exang', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#----oldpeakapriorX11----
plt.hist(df['oldpeak'], bins=30, color='darkblue')
plt.title("oldpeak Histogram", fontsize=20)
plt.xlabel('oldpeak', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#----slopeapriorX12----
slopprob = df['slope'].value_counts() / df['slope'].shape[0]
print (cpprob)
sns.countplot(x='slope', data=df)
plt.title("slope BarPlot", fontsize=20)
plt.xlabel('slope', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#----caeapriorX13----
caprob = df['ca'].value_counts() / df['ca'].shape[0]
print (cpprob)
sns.countplot(x='ca', data=df)
plt.title("ca BarPlot", fontsize=20)
plt.xlabel('ca', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#----thalapriorX14----
thalprob = df['thal'].value_counts() / df['thal'].shape[0]
print (cpprob)
sns.countplot(x='thal', data=df)
plt.title("thal BarPlot", fontsize=20)
plt.xlabel('thal', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#----YapriorY----
yprob = df['y'].value_counts() / df['y'].shape[0]
print (cpprob)
sns.countplot(x='y', data=df)
plt.title("Y BarPlot", fontsize=20)
plt.xlabel('Y', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#fbs VS gender
df['gender'].replace(0, 'Female',inplace=True)
df['gender'].replace(1, 'Male',inplace=True)
props = {}
props[('0', 'Female')] = {'color': 'xkcd:light yellow'}
props[('0','Male')] = {'color': 'xkcd:light grey'}
props[('1','Female')] = {'color':'xkcd:grey'}
props[('1', 'Male')] = {'color': 'xkcd:light pink'}
data = pd.DataFrame({'fbs': df['fbs'], 'gender': df['gender']})
mosaic(data, ['fbs', 'gender'], title='fbs VS gender',properties= props)
plt.show()

#trestbps VS chol
plt.title("trestbps VS chol", fontsize=20)
sns.scatterplot(x='trestbps', y='chol', data=df)
plt.show()

#age VS thalach
plt.title("Age VS Thalach", fontsize=20)
sns.scatterplot(x='age', y='thalach',data=df)
plt.show()

#without Values that doesnt make sense - age:
df1 = pd.read_csv ("‏‏heart_data_changes_avarage.csv")
plt.title("Age VS Thalach", fontsize=20)
sns.scatterplot(x='age', y='thalach',data=df1)
plt.show()

#trestbps VS cp
plt.title("Trestbps VS cp", fontsize=20)
sns.boxplot(x = 'cp', y='trestbps', data=df)
plt.show()

#Y VS gender
df['gender'].replace(0, 'Female',inplace=True)
df['gender'].replace(1, 'Male',inplace=True)
props = {}
props[('0', 'Female')] = {'color': 'xkcd:light yellow'}
props[('0','Male')] = {'color': 'xkcd:light grey'}
props[('1','Female')] = {'color':'xkcd:grey'}
props[('1', 'Male')] = {'color': 'xkcd:light pink'}
data = pd.DataFrame({'y': df['y'], 'gender': df['gender']})
mosaic(data, ['y', 'gender'], title='y VS Gender',properties= props)
plt.show()

#Y VS exang
df['exang'].replace(0, 'no',inplace=True)
df['exang'].replace(1, 'yes',inplace=True)
props = {}
props[('0', 'no')] = {'color': 'xkcd:light yellow'}
props[('0','yes')] = {'color': 'xkcd:light grey'}
props[('1','no')] = {'color':'xkcd:grey'}
props[('1', 'yes')] = {'color': 'xkcd:light pink'}
data = pd.DataFrame({'y': df['y'], 'exang': df['exang']})
mosaic(data, ['y', 'exang'], title='y VS exang',properties= props)
plt.show()

#Y VS chol
plt.title("Y VS chol", fontsize=20)
sns.boxplot(x = 'y', y='chol', data=df)
plt.show()

 # correlation matrix
matrix = pd.DataFrame(df.corr())
print(matrix)
matrix.to_csv("corr_matrix.csv")
sns.heatmap(df.drop('y', 1).corr(), annot=True, cmap='coolwarm')
plt.show()

# boxplot
plt.boxplot(df['age'])
plt.title("age", fontsize=20)
plt.show()

plt.boxplot(df['trestbps'])
plt.title("trestbps", fontsize=20)
plt.show()

plt.boxplot(df['chol'])
plt.title("chol", fontsize=20)
plt.show()

plt.boxplot(df['thalach'])
plt.title("thalach", fontsize=20)
plt.show()

plt.boxplot(df['oldpeak'])
plt.title("oldpeak", fontsize=20)
plt.show()

 #changing unreasonable samples to avarage
df1 = pd.read_csv ("‏‏heart_data_changes_avarage.csv")
print(df1)
plt.boxplot(df1['age'])
plt.title("age", fontsize=20)
plt.show()

sns.boxplot(x = 'y', y='age', data=df1)
plt.show()

plt.scatter(x=df1['y'], y=df1['age'])
plt.title("age", fontsize=20)
plt.xlabel('y')
plt.ylabel('age')
plt.show()

sns.stripplot(x='y', y='age', data=df1, jitter=0.1)
plt.show()
sns.distplot(df1[df1['y'] == 0]['age'], hist=False, kde=True, label='0')
sns.distplot(df1[df1['y'] == 1]['age'], hist=False, kde=True, label='1')
plt.title("age", fontsize=20)
plt.legend()
plt.show()

# remove unreasonable samples
df2 = pd.read_csv ("‏‏heart_data_changes_without.csv")
print(df2)
plt.boxplot(df2['age'])
plt.title("age", fontsize=20)
plt.show()

sns.boxplot(x = 'y', y='age', data=df2)
plt.show()

plt.scatter(x=df2['y'], y=df2['age'])
plt.title("age", fontsize=20)
plt.xlabel('y')
plt.ylabel('age')
plt.show()

sns.distplot(df2[df2['y'] == 0]['age'], hist=False, kde=True, label='0')
sns.distplot(df2[df2['y'] == 1]['age'], hist=False, kde=True, label='1')
plt.title("age", fontsize=20)
plt.legend()
plt.show()

sns.stripplot(x='y', y='age', data=df2, jitter=0.1)
plt.show()

#categorization age
df4= pd.read_csv ("age_y_new.csv")
sns.countplot(x='age', data=df4)
plt.show()

df4['age'].replace(1, '0-51',inplace=True)
df4['age'].replace(2, '52-57',inplace=True)
df4['age'].replace(3, '58+',inplace=True)
props = {}
data = pd.DataFrame({'age': df4['age'], 'y': df4['y']})
mosaic(data, ['age', 'y'], title='age VS y',properties= props)
plt.show()


#categorization trestbps
df6= pd.read_csv ("trestbps_y_new.csv")
sns.countplot(x='trestbps', data=df6)
plt.show()

df6['trestbps'].replace(1, 'optimal',inplace=True)
df6['trestbps'].replace(2, 'normal',inplace=True)
df6['trestbps'].replace(3, 'high normal',inplace=True)
df6['trestbps'].replace(4, 'high',inplace=True)
props = {}
data = pd.DataFrame({'trestbps': df6['trestbps'], 'y': df6['y']})
mosaic(data, ['trestbps', 'y'], title='trestbps VS y',properties= props)
plt.show()

#categorization oldpeak
df7= pd.read_csv ("‏‏‏‏oldpeak_y_new.csv")
sns.countplot(x='oldpeak', data=df7)
plt.show()

df7['oldpeak'].replace(1, 'les 1 mm',inplace=True)
df7['oldpeak'].replace(2, 'less 2 mm',inplace=True)
df7['oldpeak'].replace(3, 'other',inplace=True)
props = {}
data = pd.DataFrame({'oldpeak': df7['oldpeak'], 'y': df7['y']})
mosaic(data, ['oldpeak', 'y'], title='oldpeak VS y',properties= props)
plt.show()



