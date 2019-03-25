def ETL_Scada(file,file_exit):
    import pandas as pd
    import numpy as np
    import shutil
    #Exportamos el fichero con los datos del modelo
    df=pd.read_csv(file,sep=';',decimal=',',encoding='ISO-8859-1',low_memory=False)
    lista=list(df.columns)
    #Tamaño del dataset
    print('Tamaño DataSet:',df.shape[0],'filas de datos y ',df.shape[1],'variables registradas')
    #Numero de valores válidos en cada variable
    len_lista=[]
    j=0
    for i in lista:
        len_lista.append(len(df[i])-df[i].isnull().values.sum())
        j+=1
    
    #Lista de valores de la variable de referencia 
    #con el que se construirá el dataset final    
    df1=df[lista[len_lista.index(min(len_lista))]]
    df1=df1.dropna()
    list_df1=list(df1)

    #Generamos las diferentes listas de las variable con los valores 
    #validos y depurados
    df_aux=pd.DataFrame()
    s_list=[]
    for i in range(int(df.shape[1])-1):
        df_aux[lista[i]]=list(df[lista[i]])
        df_aux[lista[i+1]]=list(df[lista[i+1]])
        df_aux_1=df_aux[np.in1d(list(df_aux[lista[i]]),list_df1)]
        df_aux_1.drop_duplicates(lista[i],keep="last",inplace=True)
        s_list.append(list(df_aux_1[lista[i+1]]))
    for i in range(int(len(s_list)/2)):
        i+=1
        s_list.remove(s_list[i])
    #Construimos el data frame final
    data=pd.DataFrame()
    j=0
    data['Fecha']=list(df_aux_1.iloc[:,0])
    for i in range(len(s_list)):
        data[lista[j+1]]=s_list[i]
        j+=2
    # Eliminamos datos en parada o en regulación de secundaria
    data=data.drop(data[data['Potencia TG ValueY'].astype(float)<12500].index)
    data=data.set_index('Fecha')
    #Escribimos un nuevo .csv con los datos listos para ser trabajados en la RNA.
    data.to_csv(file_exit,encoding='iso-8859-1',decimal=',',sep=';')
    #Escribimos un nuevo .csv con los datos listos para ser almacenado en MongoDB
    f='data'+file_exit
    data.to_csv(f,sep=',',decimal='.',encoding='UTF-8')
    ruta_origen = '/Users/jpgmacbookpro/Documents/Modelizaciones_Plantas/Cabra/Modelo_TG/Machine_Learning/Modelo_2018/'+f
    ruta_destino = '/Users/jpgmacbookpro/Documents/Modelizaciones_Plantas/Cabra/Modelo_TG/Machine_Learning/Modelo_2018/'+f
    ruta=shutil.move(ruta_origen, ruta_destino)
