# PACKAGES

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE


from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 



def main():
    st.title("CAPYTAL PROJECT")
    st.sidebar.title("BANK MARKETING WEB APP")
    st.markdown("New deposit project")
    st.sidebar.markdown("Selection of the best models")


    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('bank-additional-full_processing.csv')
        return data

    @st.cache(persist=True)
    def split(data):
        y = data['deposit']
        x = data.drop(columns =['deposit'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        return x_train, x_test, y_train, y_test


    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)



        if 'Classification report' in metrics_list:
            st.subheader('Classification report')
            st.text('Model Report:\n ' + classification_report(y_test, y_pred))
        
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix") 
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names, cmap='Reds')
            st.pyplot()
            
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve") 
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
            
       
        


    df = load_data()
    class_names = ['Yes', 'No']

    x_train, x_test, y_train, y_test = split(df)
    
    smote = SMOTE(random_state = 101)
    x_train_over, y_train_over = smote.fit_resample(x_train, y_train)

    
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Decision tree", "KNN", "Random Forest"))

    if classifier == 'Decision tree':
        metrics = st.sidebar.multiselect("What metrics to plot?",('Classification report', 'Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("Decision tree")
            model = DecisionTreeClassifier(criterion = 'gini', max_depth=9, min_samples_leaf=1, min_samples_split=2)
            model.fit(x_train_over, y_train_over)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)



    if classifier == 'Random Forest':
        metrics = st.sidebar.multiselect("What metrics to plot?",('Classification report', 'Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("")
            model = RandomForestClassifier(n_jobs=-1, random_state=321, max_features='auto', n_estimators=700)
            model.fit(x_train_over, y_train_over)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)


    if classifier == 'KNN':
        metrics = st.sidebar.multiselect("What metrics to plot?",('Classification report', 'Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("KNN")
            model = KNeighborsClassifier(metric='manhattan', n_neighbors=1)
            model.fit(x_train_over, y_train_over)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)




if __name__ == '__main__':
    main()



