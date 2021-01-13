import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

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

#Split data into train and validation with 80/20
Dataset_train=[dataset1_train,dataset2_train,dataset3_train,dataset4_train,dataset5_train,dataset6_train]
Dataset_test=[dataset1_test,dataset2_test,dataset3_test,dataset4_test,dataset5_test,dataset6_test]

print('Q3.a')

for i in range(len(Dataset_train)):
    D=Dataset_train[i]
    
    MSE_Train=[]        
    lambdas_range=range(1,150,1)
    for lambdas in lambdas_range:                    
        
        #split train set into 10 folds of fit and validation sets
        kf=KFold(n_splits=10, random_state=None, shuffle=False).split(D)
                
        mse_Train=[]
        for train_index, test_index in kf:
            Dataset_f=D.loc[train_index]
            Dataset_v=D.loc[test_index]
            
            X_fit, y_fit = Dataset_f.drop('y',axis=1),Dataset_f['y']
            X_val, y_val = Dataset_v.drop('y',axis=1),Dataset_v['y']
            
            #Created L2 closed form to calculate W vector and MSE
            n,p = X_fit.shape
            I=np.eye(p)
            w =inv((X_fit.T.dot(X_fit))+lambdas*I).dot(X_fit.T).dot(y_fit)

            #validation set inside of Train data
            yhat_val = X_val.dot(w)
            mse_train=round((yhat_val-y_val).T.dot((yhat_val-y_val)),2)
            mse_Train.append(mse_train)
            
        mse_Train_avg=round(sum(mse_Train)/len(mse_Train),2)
        MSE_Train.append(mse_Train_avg)   
    
    a=np.array(MSE_Train)
    b=np.array(lambdas_range)
    data=np.array([a,b]).T
    MSEs=pd.DataFrame(data=data,columns=['MSE_Train','lambdas']).set_index('lambdas')
    
    # plot MSEs for train set
    fig, ax = plt.subplots()
    plt.scatter(lambdas_range, MSE_Train, color='blue')
    plt.title('Dataset_%s'% (i+1))
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.legend(['MSE_Train'])
    plt.savefig('Q3_plot_Dataset_%s.png' %(i+1)) 
    plt.show() 
    
    #find minimum train set MSE and corresponding lambda value  
    min_MSE = MSEs['MSE_Train'].min()
    selected_lambdas = int(MSEs['MSE_Train'].idxmin())

    #Re_train on the entire training dataset uisng the chosen parameter value
    X_re_train, y_re_train = D.drop('y',axis=1),D['y']
    n,p = X_re_train.shape
    I=np.eye(p)
    w_retrain =inv((X_re_train.T.dot(X_re_train))+selected_lambdas*I).dot(X_re_train.T).dot(y_re_train)
  
    #Train set MSE
    yhat_re_train = X_re_train.dot(w_retrain)
    mse_re_train=round((yhat_re_train-y_re_train).T.dot((yhat_re_train-y_re_train)),2)

    #Test set MSE
    X_test,y_test = Dataset_test[i].drop('y',axis=1),Dataset_test[i]['y']
    yhat_test = X_test.dot(w_retrain)
    mse_test=round((yhat_test-y_test).T.dot((yhat_test-y_test)),2)
    
   # print('for dataset'+str(i+1)+' mse_test:'+str(mse_test)+'; re_train_mse:'+str(mse_re_train))
    
    #Question 3a.
    print('for dataset' + str(i+1)+ ' the best choice of λ value is '+ str(selected_lambdas) + '， the corresponding test set MSE is '+ str(min_MSE))
print('\n')

        
        
                

        


    

            

        


    
    
