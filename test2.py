# PACKAGES

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split


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
    st.sidebar.markdown("Is the prospect willing to sign")


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



        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix") 
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
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
    
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Decision tree", "KNN ", "Random Forest"))

    if classifier == 'Decision tree':
        metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("Decision tree")
            model = DecisionTreeClassifier(criterion = 'gini', max_depth=9, min_samples_leaf=1, min_samples_split=2)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'KNN':
        metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("KNN")
            model = KNeighborsClassifier(metric='manhattan', n_neighbors=1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)



    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("")
            model = RandomForestClassifier(n_jobs=-1, random_state=321, max_features='auto', n_estimators=700)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)


    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)


if __name__ == '__main__':
    main()


###


#Use Random Forest
from sklearn.ensemble import RandomForestRegressor

#SHAP(SHapley Additive exPlanations)
import shap
shap.initjs() # for visualization
X, y = shap.datasets.boston()

clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)
clf.fit(X, y)

explainer = shap.TreeExplainer(clf, X)
shap_values = explainer.shap_values(X)
st.set_option('deprecation.showPyplotGlobalUse', False)

#force_plot
i = 0
shap.force_plot(explainer.expected_value, shap_values[i,:], X.iloc[i,:])
st.pyplot()

#summary_plot_bar
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot()

#summary_plot
shap.summary_plot(shap_values, X)
st.pyplot()

#dependance_plot
shap.dependence_plot("LSTAT", shap_values, X)
st.pyplot()

###########################
