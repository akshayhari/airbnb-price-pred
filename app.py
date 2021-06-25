from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import FMRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd
#from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
import matplotlib.pyplot as plt

sc = SparkSession.builder.appName('airbnb_price_pred') \
            .getOrCreate()


st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title('''New York Airbnb Price Prediction''')
st.subheader('Machine Learning Regression model comparison in MLlib. \
    Dataset [link](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)')
st.subheader('Built by [Akshay Hari](https://www.linkedin.com/in/akshayhari/) \
    Github repo [here](https://github.com/akshayhari/airbnb-price-pred)')


#########   DATA

#df = sc.read.csv("file:///home/hduser/programs/airbnb-price-pred/airbnb.csv", header=True)
df = sc.read.csv("airbnb.csv", header=True)
#Preprocessed data is given as input to save computation
#df4 = sc.read.load("file:///home/hduser/programs/airbnb-price-pred/processed_data.parquet")
df4 = sc.read.load("processed_data.parquet")
splits = df4.randomSplit([0.7, 0.3], seed=12345)
train_df = splits[0]
test_df = splits[1]

##########  SIDEBAR

st.sidebar.title('MLlib Regression models')
st.sidebar.subheader('Select your model')
mllib_model = st.sidebar.selectbox("Regression Models", \
        ('Linear Regression', 'Gradient Boost Tree', 'Decision Tree Regressor', \
            'Random Forest Regressor', 'Factorization machine Regressor'))
st.sidebar.text('70 - 30 split')

def regression_model(mllib_model, train_df, test_df):
    if mllib_model == 'Linear Regression':
        lr = LinearRegression(featuresCol = 'features', labelCol='label')   
        lr_model = lr.fit(train_df)
        fullPredictions = lr_model.transform(test_df).cache()
        lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
        r2 = lr_evaluator.evaluate(fullPredictions)
        lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="rmse")
        rmse = lr_evaluator.evaluate(fullPredictions)
        pred = [int(row['prediction']) for row in fullPredictions.select('prediction').collect()]
        actual = [int(row['label']) for row in fullPredictions.select('label').collect()]
        return r2,rmse,pred,actual

    elif mllib_model == 'Decision Tree Regressor':
        dt = DecisionTreeRegressor(featuresCol = 'features', labelCol='label')
        dt_model = dt.fit(train_df)
        dtPrediction = dt_model.transform(test_df).cache()
        dt_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
        r2 = dt_evaluator.evaluate(dtPrediction)
        dt_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="rmse")
        rmse = dt_evaluator.evaluate(dtPrediction)
        pred = [int(row['prediction']) for row in dtPrediction.select('prediction').collect()]
        actual = [int(row['label']) for row in dtPrediction.select('label').collect()]
        return r2,rmse,pred,actual

    elif mllib_model == 'Gradient Boost Tree':
        gb = GBTRegressor(featuresCol = 'features', labelCol='label')
        gb_model = gb.fit(train_df)
        gbPredictions = gb_model.transform(test_df).cache()
        gb_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
        r2 = gb_evaluator.evaluate(gbPredictions)
        gb_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="rmse")
        rmse = gb_evaluator.evaluate(gbPredictions)
        pred = [int(row['prediction']) for row in gbPredictions.select('prediction').collect()]
        actual = [int(row['label']) for row in gbPredictions.select('label').collect()]
        return r2,rmse,pred,actual   

    elif mllib_model == 'Random Forest Regressor':
        rf = RandomForestRegressor(featuresCol = 'features', labelCol='label')
        rf_model = rf.fit(train_df)
        rfPredictions = rf_model.transform(test_df).cache()
        rf_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
        r2 = rf_evaluator.evaluate(rfPredictions)
        rf_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="rmse")
        rmse = rf_evaluator.evaluate(rfPredictions)
        pred = [int(row['prediction']) for row in rfPredictions.select('prediction').collect()]
        actual = [int(row['label']) for row in rfPredictions.select('label').collect()]
        return r2,rmse,pred,actual

    else:
        fm = FMRegressor(featuresCol = 'features', labelCol='label')
        fm_model = fm.fit(train_df)
        fmPredictions = fm_model.transform(test_df).cache()
        fm_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
        r2 = fm_evaluator.evaluate(fmPredictions)
        fm_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="rmse")
        rmse = fm_evaluator.evaluate(fmPredictions)
        pred = [int(row['prediction']) for row in fmPredictions.select('prediction').collect()]
        actual = [int(row['label']) for row in fmPredictions.select('label').collect()]
        return r2,rmse,pred,actual


st.dataframe(data = df.toPandas().head(10))
st.text('Our target variable is price and we are giving vectorized data to the mllib.')
st.text('Below shown data are results of the testing data.')
r2,rmse,actual,pred = regression_model(mllib_model, train_df, test_df)

st.write(mllib_model," model")


col3, col4, col5 = st.beta_columns((1,1,2))
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-Weight: bold;
}
</style>
""", unsafe_allow_html=True)

col3.header("R2 score")
col3.markdown(f'<p class="big-font">{"{:.2f}".format(r2)}</p>', unsafe_allow_html=True)

#
col4.header("RMS Error")
col4.markdown(f'<p class="big-font">{"{:.2f}".format(rmse)}</p>', unsafe_allow_html=True)

#
fig, ax = plt.subplots()
ax.scatter(actual, pred, color='b', s=60, alpha=0.1)
plt.plot([5,250], [5,250], color='r')
plt.xlim([0, 260])
plt.ylim([0, 260])
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted Prices',fontsize=20)
col5.pyplot(fig)

st.text('All the models have average performance and Gradient Boost Tree Regression gave best results among them.')

