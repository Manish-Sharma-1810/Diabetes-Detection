# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import pickle


# IMPORT TRAINED MODELS
svm = pickle.load(open('SVC_model.pkl','rb'))
rf = pickle.load(open('rf.pkl','rb'))
grid = pickle.load(open('grid.pkl','rb'))
log_model = pickle.load(open('log_model.pkl','rb'))

# LOAD DATASET
df = pd.read_csv('data.csv')
df = df.drop(df.columns[0], axis=1)


# HEADINGS
st.title('PIMA Diabetes Detection')

html_temp1 = """
    <br>
    <div style="background-color:red ;padding:2px">
    <h1 style="color:white;text-align:center; font-size:35px"><b>Diabetes Checkup</b></h1>
    </div>
    <br>
    <br>
    
    """
st.markdown(html_temp1, unsafe_allow_html=True)
activities=['SVM', 'RandomForest', 'GridSearchCV', 'Logistic Regression']
option=st.sidebar.selectbox('Which model would you like to use?',activities)
st.sidebar.header('Patient Data')


if st.checkbox("About Dataset"):
    html_temp2 = """
    <br>
    <p>
    This dataset is originally from the <i><b>National Institute of Diabetes and Digestive and Kidney Diseases.</b></i> 
    The objective of the dataset is to diagnostically predict whether or not a patient has diabetes,
    based on certain diagnostic measurements included in the dataset.
    Several constraints were placed on the selection of these instances from a larger database. 
    In particular, all patients here are females at least 21 years old of Pima Indian heritage.
    <br>
    <br>
    The datasets consists of several medical predictor variables and one target variable, Outcome. 
    Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin 
    level, age, and so on.
    </p>
    <br>
    Reference : <a style='display: block;' href="https://www.kaggle.com/uciml/pima-indians-diabetes-database" target="_blank">Kaggle</a>
    """
    st.markdown(html_temp2, unsafe_allow_html=True)
    st.subheader("Dataset")
    st.write(df.head(10))
    st.subheader("Describe dataset")
    st.write(df.describe())


# Set White Grid
sns.set_style("whitegrid")

# VISUALIZATION
if st.checkbox("Exploratory Data Analysis (EDA)"):
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)



# Train-Test Split
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=101)



# Training Data
if st.checkbox("Train-Test Datasets"):
    st.subheader("X_train")
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader("X_test")
    st.write(X_test.head())
    st.write(X_test.shape)
    st.subheader("y_test")
    st.write(y_test.head())
    st.write(y_test.shape)


# FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 1)
    glucose = st.sidebar.slider('Glucose', 0,200, 108)
    bp = st.sidebar.slider('Blood Pressure', 0,140, 40)
    skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 25)
    insulin = st.sidebar.slider('Insulin', 0,1000, 120)
    bmi = st.sidebar.slider('BMI', 0,80, 25)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.05,2.5, 0.45)
    age = st.sidebar.slider('Age', 21,100, 24)
    skin = st.sidebar.slider('Skin', 0.0,5.0, 1.04)

    user_report_data = {
      'num_preg':pregnancies,
      'glucose_conc':glucose,
      'diastolic_bp':bp,
      'thickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'diab_pred':dpf,
      'age':age,
      'skin':skin
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)


# MODELS
if option=='SVM':
  user_result = svm.predict(user_data)
  svc_score = accuracy_score(y_test, svm.predict(X_test))
elif option=='RandomForest':
  user_result = rf.predict(user_data)
  rf_score = accuracy_score(y_test, rf.predict(X_test))
elif option=='GridSearchCV':
  user_result = grid.predict(user_data)
  grid_score = accuracy_score(y_test, grid.predict(X_test))
else:
  user_result = log_model.predict(user_data)
  log_score = accuracy_score(y_test, log_model.predict(X_test))



# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)
st.subheader('Model Used: \n'+option)
st.subheader('Accuracy: ')
if option=='SVM':
  st.write(str(svc_score*100)+'%')
elif option=='RandomForest':
  st.write(str(rf_score*100)+'%')
elif option=='GridSearchCV':
  st.write(str(grid_score*100)+'%')
else:
  st.write(str(log_score*100)+'%')


# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

# VISUALISATIONS REPORT
st.title('Visualised Report')
if st.checkbox("Show Visualized Report"):

    # Age vs Pregnancies
    st.header('Pregnancy count Graph (Others vs Yours)')
    fig_preg = plt.figure()
    ax1 = sns.scatterplot(x = 'age', y = 'num_preg', data = df, hue = 'diabetes', palette = 'Dark2')
    ax2 = sns.scatterplot(x = user_data['age'], y = user_data['num_preg'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,20,2))
    plt.xlabel('Age')
    plt.ylabel('Pregnencies')
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_preg)


    # Age vs Glucose
    st.header('Glucose Value Graph (Others vs Yours)')
    fig_glucose = plt.figure()
    ax3 = sns.scatterplot(x = 'age', y = 'glucose_conc', data = df, hue = 'diabetes' , palette='magma')
    ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose_conc'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,220,10))
    plt.xlabel('Age')
    plt.ylabel('Glucose conc.')
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_glucose)

    # Age vs Bp
    st.header('Blood Pressure Value Graph (Others vs Yours)')
    fig_bp = plt.figure()
    ax5 = sns.scatterplot(x = 'age', y = 'diastolic_bp', data = df, hue = 'diabetes', palette='Reds')
    ax6 = sns.scatterplot(x = user_data['age'], y = user_data['diastolic_bp'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,130,10))
    plt.xlabel('Age')
    plt.ylabel('Diastolic Blood Pressure')
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bp)

    # Age vs Skin Thickness
    st.header('Skin Thickness Value Graph (Others vs Yours)')
    fig_st = plt.figure()
    ax7 = sns.scatterplot(x = 'age', y = 'thickness', data = df, hue = 'diabetes', palette='winter_r')
    ax8 = sns.scatterplot(x = user_data['age'], y = user_data['thickness'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,110,10))
    plt.xlabel('Age')
    plt.ylabel('Skin Thickness')
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_st)

    # Age vs Insulin
    st.header('Insulin Value Graph (Others vs Yours)')
    fig_i = plt.figure()
    ax9 = sns.scatterplot(x = 'age', y = 'insulin', data = df, hue = 'diabetes', palette='rocket')
    ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,900,50))
    plt.xlabel('Age')
    plt.ylabel('Insulin')
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_i)

    # Age vs BMI
    st.header('BMI Value Graph (Others vs Yours)')
    fig_bmi = plt.figure()
    ax11 = sns.scatterplot(x = 'age', y = 'bmi', data = df, hue = 'diabetes', palette='tab20_r')
    ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,70,5))
    plt.xlabel('Age')
    plt.ylabel('BMI')
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bmi)

    # Age vs Dpf
    st.header('DPF Value Graph (Others vs Yours)')
    fig_dpf = plt.figure()
    ax13 = sns.scatterplot(x = 'age', y = 'diab_pred', data = df, hue = 'diabetes', palette='YlOrBr')
    ax14 = sns.scatterplot(x = user_data['age'], y = user_data['diab_pred'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,3,0.2))
    plt.xlabel('Age')
    plt.ylabel('DiabetesPedigreeFunction')
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_dpf)

    # Age vs Skin
    st.header('Skin Value Graph (Others vs Yours)')
    fig_sk = plt.figure()
    ax15 = sns.scatterplot(x = 'age', y = 'skin', data = df, hue = 'diabetes', palette='twilight_r')
    ax16 = sns.scatterplot(x = user_data['age'], y = user_data['skin'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,4,0.2))
    plt.xlabel('Age')
    plt.ylabel('Skin')
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_sk)

# Add Image

st.header('**About Me**')
col1, col2 = st.beta_columns([10,20])
with col1:
    st.image('my_img.jpg', width=200)
with col2:
    st.write(
      '''
      *Hi! I'm **Manish** and I am a B.Tech final-year student,*
      *specialized in Computer Scince and Engineering.*

      *I'm passionate about Data Scince. Machine Learning and Deep Learning*
      *enthusiast, and like to develope ML based products.*

      * **Email:** *manishkumar.sharma0103@gmail.com*

      * **Mobile:** +91 9026822669


      '''
      )


    st.markdown(
    """
      * <a style='display: block;' href="https://github.com/Manish-Sharma-1810" target="_blank">GitHub</a>
      * <a style='display: block;' href="https://www.linkedin.com/in/manish-kumar-1810" target="_blank">LinkedIn</a>
    """, unsafe_allow_html=True)


# PAGE VIEWS FUNCTION
@st.cache(allow_output_mutation=True)
def Pageviews():
    return []

pageviews=Pageviews()
pageviews.append('dummy')

try:
    st.markdown('Page viewed : {} times.'.format(len(pageviews)))
except ValueError:
    st.markdown('Page viewed : {} times.'.format(1))


html_temp3 = """
    <br>
    <div>
    <h1 style="text-align:center; font-size:25px"><b>Thank You Streamlit! For giving such Awsome Interface and make alive to my models.</b></h1>
    </div>
    <br>
    
    """
st.markdown(html_temp3, unsafe_allow_html=True)

st.write('**Learn more about [Streamlit](https://streamlit.io/)**')

################## END OF THE CODE ####################
