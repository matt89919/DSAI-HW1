# You can write code above the if-main block.
import csv
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    # -*- coding: utf-8 -*- 

    df=pd.read_csv(args.training)
    #df = pd.read_csv('./data.csv')
    #將每筆資料加入index
    df['index']=list(range(1,len(df)+1))
    #將每筆資料加上是星期幾
    df['days']=df['index']%7
    s ={0:2,1:3,2:4,3:5,4:6,5:7,6:1}
    df['days']=df['days'].map(s)
    #df['month']=df['日期'][4:6]
    #print(df['days'])
    df['日期']=df['日期'].astype("string")
    l=[]
    for a in range(len(df)):
        l.append(df['日期'][a][4:6])
    #print(l)
    df['month']=pd.Series(l)
    #df=df[14:487]
    df_y=df[0:486]
    df_arry=df['備轉容量(MW)'].values
    avrg=0
    #print(len(df_arry))
    temp=df_arry
    for i in range(len(df_y)-13):
        avrg=int((df_arry[i]+df_arry[i+1]+df_arry[i+2]+df_arry[i+3]+df_arry[i+4]+df_arry[i+5]+df_arry[i+6]+
        df_arry[i+7]+df_arry[i+8]+df_arry[i+9]+df_arry[i+10]+df_arry[i+11]+df_arry[i+12]+df_arry[i+13])/14)
    temp[i]=avrg
    #print(temp)
    df['before_target']=pd.DataFrame(temp)
    #print(df['before_target'])
    #print(df_arry)
    #X=df[['month','days','before_target']]
    X=df[['month','days']]
    #print(X)
    y=df['備轉容量(MW)']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)
    '''sc=StandardScaler()
    sc.fit(X_train)
    X_train_std=sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    svm=SVC(kernel='rbf',probabil1ty=True)
    svm.fit(X_train_std,y_train)
    pred=svm.predict(X_test_std)'''
    k=1
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    pred=knn.predict(X_test)
    #print(pred)
    y_test_np=y_test.values
    err=0
    for i in range(len(y_test_np)):
        err=err+((pred[i]-y_test_np[i])**2)**0.5
    err=(err)/(len(y_test_np))
    print(err)
    month_list=[3,3,4,4,4,4,4,4,4,4,4,4,4,4,4]
    days_list=[3,4,5,6,7,1,2,3,4,5,6,7,1,2,3]
    before_target=[]
    input=pd.DataFrame(list(zip(month_list,days_list)),columns=['month','days'])
    pred=knn.predict(input)
    '''table=[['date','operating_reserve(MW)'],
        ['20220330',pred[0]],
        ['20220331',pred[1]],
        ['20220401',pred[2]],
        ['20220402',pred[3]],
        ['20220403',pred[4]],
        ['20220404',pred[5]],
        ['20220405',pred[6]],
        ['20220406',pred[7]],
        ['20220407',pred[8]],
        ['20220408',pred[9]],
        ['20220409',pred[10]],
        ['20220410',pred[11]],
        ['20220411',pred[12]],
        ['20220412',pred[13]],
        ['20220413',pred[14]]]'''
    date_list=['20220330','20220331','20220401','20220402','20220403','20220404','20220405','20220406','20220407',
                '20220408','20220409','20220410','20220411','20220412','20220413']

    res=pd.DataFrame(list(zip(date_list,pred)),columns=['date','operating_reserve(MW)'])
    print(res)
    print(pred)
    res.to_csv(args.output,index=False)

  


