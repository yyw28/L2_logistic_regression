import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

#Experiment data
#dataset_experiment[:50].to_csv('/Users/yuwenyu/Desktop/HW1_dataset/train-50(1000)-100.csv')
#dataset_experiment[:100].to_csv('/Users/yuwenyu/Desktop/HW1_dataset/train-100(1000)-100.csv')
#dataset_experiment[:150].to_csv('/Users/yuwenyu/Desktop/HW1_dataset/train-150(1000)-100.csv')

#read data
dataset1_train = pd.read_csv('train-100-10.csv').drop(['Unnamed: 11','Unnamed: 12'], axis=1)
dataset2_train = pd.read_csv('train-100-100.csv')
dataset3_train = pd.read_csv('train-1000-100.csv')
dataset4_train = pd.read_csv('train-50(1000)-100.csv').drop('Unnamed: 0',axis=1)
dataset5_train = pd.read_csv('train-100(1000)-100.csv').drop('Unnamed: 0',axis=1)
dataset6_train = pd.read_csv('train-150(1000)-100.csv').drop('Unnamed: 0',axis=1)
dataset1_test = pd.read_csv('test-100-10.csv')
dataset2_test = pd.read_csv('test-100-100.csv')
dataset3_test = pd.read_csv('test-1000-100.csv')
dataset4_test = pd.read_csv('test-1000-100.csv')
dataset5_test = pd.read_csv('test-1000-100.csv')
dataset6_test = pd.read_csv('test-1000-100.csv')

#created train and test container for all datasets
Dataset_train=[dataset1_train,dataset2_train,dataset3_train,dataset4_train,dataset5_train,dataset6_train]
Dataset_test=[dataset1_test,dataset2_test,dataset3_test,dataset4_test,dataset5_test,dataset6_test]

print('Q2.a')

#Split data into train and validation with 80/20
for i in range(len(Dataset_train)):
    X_fit, X_val, y_fit, y_val = train_test_split(Dataset_train[i].drop('y',axis=1), Dataset_train[i]['y'], test_size=0.2)
    X_test,y_test = Dataset_test[i].drop('y',axis=1),Dataset_test[i]['y']
    n,p = X_fit.shape
    I=np.eye(p)
    
    #Created L2 closed form to calculate W vector and MSE
    lambdas_range=range(1,150,1)
    
    MSE_Fit=[]
    MSE_Val=[]
    MSE_Test = []
    for lambdas in lambdas_range:
        #Fitting data
        w =inv((X_fit.T.dot(X_fit))+lambdas*I).dot(X_fit.T).dot(y_fit)
        yhat_fit = X_fit.dot(w)
        mse_fit=round((yhat_fit-y_fit).T.dot((yhat_fit-y_fit)),2)
        MSE_Fit.append(mse_fit)
        
        #validation data
        yhat_val = X_val.dot(w)
        mse_val=round((yhat_val-y_val).T.dot((yhat_val-y_val)),2)
        MSE_Val.append(mse_val)
        
        #Test data
        yhat_test = X_test.dot(w)
        mse_test=round((yhat_test-y_test).T.dot((yhat_test-y_test)),2)
        MSE_Test.append(mse_test)
        
    # Question 2a.
    #MSE tables
    a=np.array(MSE_Fit)
    b=np.array(MSE_Val)
    c=np.array(MSE_Test)
    d=np.array(lambdas_range)
    data=np.array([a,b,c,d]).T
    MSEs=pd.DataFrame(data=data,columns=['MSE_Fit','MSE_Val','MSE_Test','lambdas']).set_index('lambdas')
    
    #find minimum validation and test set's MSE and corresponding lambda values separately
    min_MSE_val = MSEs['MSE_Val'].min()
    min_lambdas_val = int(MSEs['MSE_Val'].idxmin())

    min_MSE_test = MSEs['MSE_Test'].min()
    min_lambdas_test = int(MSEs['MSE_Test'].idxmin())

    print('Dataset'+ str(i+1)+': the least test set MSE is '+ str(min_MSE_test) + '. the corresponding lambda value is '+ str(min_lambdas_test))
    #print('Dataset'+str(i+1)+': the least valuation set MSE is '+ str(min_MSE_val) + '. the corresponding lambda value is '+ str(min_lambdas_val))
    
    #Test MSE with selected Lambda
    selected_labmda_test_mse=MSEs.iloc[min_lambdas_val]['MSE_Test']
    dif_mse_test= min_MSE_test -selected_labmda_test_mse
    #print('with selected labmda from valuation set, the test set MSE is ' + str(selected_labmda_test_mse))
    
    #Question 2b.
    #plot MSEs for both train and test datasets
    fig, ax = plt.subplots()
    plt.scatter(lambdas_range, MSE_Val, color='red')
    plt.scatter(lambdas_range, MSE_Test, color='blue')
    plt.title('Dataset_%s'% (i+1))
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.legend(['MSE_Val','MSE_Test'])
    plt.savefig('Q2.plot_Dataset_%s.png' %(i+1))
    plt.show()
    




