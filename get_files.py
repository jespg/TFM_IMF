def crit_date(date_i, date_f, file):
    import pandas as pd
    import numpy as np
    import shutil
    from datetime import datetime
    
    #Obtenemos el dataframe desde el fichero
    df=pd.read_csv(file,sep=';',decimal=',',encoding='ISO-8859-1',low_memory=False)

    # Convertimos los criterios a formato de fecha
    date_i = datetime.strptime(date_i, '%d/%m/%Y %H:%M')
    date_f = datetime.strptime(date_f, '%d/%m/%Y %H:%M')
       
    # Convertimos las fechas del data set a formato con fecha.
    dates=list(df['Date'])
    dates_f=[]
    for i in dates:
        dates_f.append(datetime.strptime(i, '%d/%m/%y %H:%M'))
    
    # Añadimos una nueva columna con las fecha en el formato correcto
    df['Dates_F']=dates_f
    
    #Obetenemos el dataset entre fechas.
    df_train=df[(df['Dates_F']>date_i) & (df['Dates_F']<date_f)]
    df_train=df_train.set_index('Date')
    
    # Generamos un fichero csv para el modelo de entrenamiento.
    f='data_train.csv'
    df_train.to_csv(f,sep=';',decimal=',',encoding='ISO-8859-1')
    print('<>'*5,'Se han creado el fichero',f,'<>'*5)
    return df_train
    
def crit_n(n,file_train):
    import pandas as pd
    
    #Obtenemos el dataframe desde el fichero
    df=pd.read_csv(file_train,sep=';',decimal=',',encoding='ISO-8859-1',low_memory=False)
    
    #Crietrio de construcción de los ficheros: cada n horas
    new_row=0 
    file_list=[]
    for i in range(int(len(df)/n)):
        df_1=df[new_row:new_row+n]
        new_row += n
        df_1=df_1.set_index('Date')
        f='data_train_'+str(i)+'.csv'
        file_list.append(f)
        df_1.to_csv(f,sep=';',decimal=',',encoding='ISO-8859-1')
    print('====== Se han creado %i ficheros ========='%i)