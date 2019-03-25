from DataOPC import DataOPC
import pandas as pd
import time
import datetime

#Código para crear un data frame con los datos obetnidos desde el OPC.

#Se van a crear cuatro ficheros con cuatro data frames procedente del OPC
#Tomamos 10 datos durante 1 sg
for iter in range(4):
    print('Iniciamos Iteración ##',iter)
    list_var=DataOPC(24, 3600)
    data=pd.DataFrame()
    k=0
    for i in list_var[1]:
        data_l=[]
        for j in range(len(list_var[0])):
            data_n=list_var[0][j][k]
            data_l.append(data_n)
        data[i]=data_l
        k+=1
    data['Time'] = list_var[2]
    data=data.set_index('Time')
    print(data)
    #Escribimos un nuevo .csv con los datos listo para ser trabajados.
    #Nombre será la fecha del día y hora y el número de la iteración
    #Obtenemos la fecha
    date=datetime.datetime.now()
    #Generamos el nombre del fichero
    y=date.year
    m=date.month
    d=date.day
    h=date.hour
    data.to_csv('df_%d_%d_%d_%d_%d.csv'%(iter,y,m,d,h),encoding='ISO-8859-1',decimal=',',sep=';')
    
