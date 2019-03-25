### Función Verificación y comprobación de los modelos ###

# Error entre el modelo entrenado, el modelo del día y el dato promedio real 

def f_test(x,y,reg):
    import numpy as np
    import pandas as pd
    
    df_var_mod=pd.read_csv('file_model_r2.csv',sep=';',decimal=',',encoding='ISO-8859-1')
    ##############df_mod_amb=pd.read_csv('Model_Amb.csv',sep=';',decimal=',',encoding='ISO-8859-1')
    
    array=[]
    l=list(x.columns)
    for i in l:
        array.append(np.mean(x[i]))
    prod=np.dot(reg[0],array)+reg[1]
    suma=0
    #########suma2=0
    for i in range(len(df_var_mod)-1):
        suma=suma+np.mean(x[list(df_var_mod.loc[i])[0]])*list(df_var_mod.loc[i])[1]
    #for i in range(len(df_mod_amb)-1):
        ###########suma2=suma2+np.mean(x[list(df_mod_amb.loc[i])[0]])*list(df_mod_amb.loc[i])[1]
    T1 = suma+df_var_mod[df_var_mod['Variable']=='bias']['Coeficiente']
    #############T2 = suma2+df_mod_amb[df_mod_amb['Variable']=='bias']['Coeficiente']
    print('#'*50,'\nPot Calculada según Modelo = %0.3f'% prod)
    print('\nPotencia Real alcanzada:',np.mean(y))
    print('Desvio Técnico: %.3f'%((np.mean(y)/T1-1)*100),'%')
    print('\nPotencia Máxima esperada: %.3f'%(T1))
    ########print('\nPotencia Esperada por Condiciones Ambientales: %.3f'%(T2))
    ###########print('Desvio del Modelo de Condiciones Ambientales:%.3f'%((np.mean(y)/T2-1)*100),'%')
    print('#'*50)
    res_t=['Pot Calculada según Modelo','Potencia Real alcanzada','Desvio Técnica(%)','Potencia Máxima esperada']
    res_n=[prod,np.mean(y),float(np.mean(y)/T1-1)*100,float(T1)]
    
    return float(np.mean(y)/T1-1)*100


# Función para la obtención de los ficheros con los reultados de los análisis.
    
def Data_Analysis(i,var_remove):
    
    # Importamos funciones auxliares creadas en los Scripts
    from ModReg_TG import imp_ds, Fit_Var, sense_var, f_model_reg, plot_var, f_score
    from f_selection import ForwardSelection_r2, ForwardSelection_r2_Ac
    from ModDL import f_test
    from datetime import datetime
    import pandas as pd
    import shutil
    
    file_data = 'data_train_'+str(i)+'.csv'
    #Importamos el dataset con todas las variables de la máquina objetivo de estudio
    file=imp_ds(file_data)
    # devuelve lista de variable, el dataframe completo, y fecha de los datos sometidos a análisis.
    #Función que elimina las variables no numéricas y establece la variable dependiente
    df_var=Fit_Var(file,var_remove)
    # devuelve lista de variables tratadas, la X y la Y para el modelo de entrenamiento

    #Obtención de los primeros estadísticos.
    #Ejecutamos una función que devuelve el valor medio y la desviación std
    stat=sense_var(df_var[1],df_var[2])
    # devuelve lista de variables y lista con las medias y lista con las descv std.

    #Una vez que el dataset ha sido tratado, se lanza la función de regresión para el Modelo
    #Se lanza tantas veces como sea necesario para asegurar que no haya sobreajuste
    e=0.1
    i=0
    while (e >0.05 and i <10):
        reg=f_model_reg(df_var[1],df_var[2])
        e=reg[2]
        if i >10:
            print('Modelo Presenta Sobre ajuste.')
        i+=1
    print('------------------\nIteraciones realizadas >>>',i)
    #devuelve una lista con los coeficientes de la regresion, el bias de la regresion y la diferencia entre el r2 del train y el test para valorar un posible sobre ajuste

    # Función de selección de caracteristicas por orden decreciente de r2
    var_s = ForwardSelection_r2_Ac(list(df_var[1].columns),df_var[1],df_var[2])
    var_r2 = ForwardSelection_r2(df_var)
    #La función nos devuelve la lista de variables ordenadas por r2 y la lista con los r2

    # Verificamos el rendimiento del modelo con los datos reales
    test = f_test(df_var[1],df_var[2],reg)
    
    return test

# Función de analisis de sensibilidad de r2

def r2_Analysis(i,var_remove,test):
    
    # Importamos funciones auxliares creadas en los Scripts
    from ModReg_TG import imp_ds, Fit_Var, sense_var, f_model_reg, plot_var, f_score
    from f_selection import ForwardSelection_r2, ForwardSelection_r2_Ac
    from ModDL import f_test
    from datetime import datetime
    import pandas as pd
    import shutil
    
    file_data = 'data_train_'+str(i)+'.csv'
    #Importamos el dataset con todas las variables de la máquina objetivo de estudio
    file=imp_ds(file_data)
    # devuelve lista de variable, el dataframe completo, y fecha de los datos sometidos a análisis.
    #Función que elimina las variables no numéricas y establece la variable dependiente
    df_var=Fit_Var(file,var_remove)
    # devuelve lista de variables tratadas, la X y la Y para el modelo de entrenamiento

    #Obtención de los primeros estadísticos.
    #Ejecutamos una función que devuelve el valor medio y la desviación std
    stat=sense_var(df_var[1],df_var[2])
    # devuelve lista de variables y lista con las medias y lista con las descv std.

    #Una vez que el dataset ha sido tratado, se lanza la función de regresión para el Modelo
    #Se lanza tantas veces como sea necesario para asegurar que no haya sobreajuste
    e=0.1
    i=0
    while (e >0.05 and i <10):
        reg=f_model_reg(df_var[1],df_var[2])
        e=reg[2]
        if i >10:
            print('Modelo Presenta Sobre ajuste.')
        i+=1
    print('------------------\nIteraciones realizadas >>>',i)
    #devuelve una lista con los coeficientes de la regresion, el bias de la regresion y la diferencia entre el r2 del train y el test para valorar un posible sobre ajuste
    
    ### Creamos un Data Frame Resumen con los Estadísticos principales.
    ##  Con ellos realizaremos el análisis de sensibilidad para el diagnostico del sistema
    df_summary=pd.DataFrame()

    # Creamos la lista con todas las variables
    Variable=list(stat.index)

    # Creamos la lista con los coeficientes del modelo
    Coef=[0]
    for i in list(reg[0]):
        Coef.append(i)

    # Creamos la lista con los valores promedios
    Media=list(stat['Media'])

    # Creamos la lista con los valores de desviación Std.
    Dev_Std=list(stat['Std Dev'])

    # Creamos la lista con los valores de r2
    r2_score=[reg[1]]
    r2_list=f_score(df_var[1],df_var[2],list(df_var[1].columns))
    for i in r2_list:
        r2_score.append(i)
    
    # Creamos el campo fecha con formato
    fecha_f=datetime.strptime(file[2].strip(' \t\r\n'),'%d/%m/%y')
    fecha_fs=datetime.strftime(fecha_f,'%d/%m/%y')
    
    list_fecha=[fecha_fs]*len(Variable)

    # Creamos el dataframe
    df_summary['Fecha']=list_fecha
    df_summary['Variable']=Variable
    df_summary['r2_score']=r2_score
    df_summary['Valor Medio']=Media
    df_summary['Std Dev']=Dev_Std
    df_summary['Coef Model']=Coef
    df_summary.loc[len(Variable)+1]=[fecha_fs,'Desvio Tecnico',0,test,0,0]
    ######df_summary.loc[16]=[fecha_fs,'Desvio CC Amb',0,test[0][3],0,0]
    df_summary.loc[len(Variable)+2]=[fecha_fs,'r2 Train',reg[3],0,0,0]
    df_summary.loc[len(Variable)+3]=[fecha_fs,'r2 Test',reg[4],0,0,0]
    print(df_summary.sort_values(by='r2_score',ascending=False))

    # Creamos un fichero con los datos del resumen
    f='fa_'+file[2].replace('/','-')+'.csv'
    df_summary.to_csv(f,sep=';',decimal=',',encoding='ISO-8859-1')

    # Movemos el fichero creado a la carpera Salida_Analisis
    ruta_origen = '/Users/jpgmacbookpro/Google Drive/Master Big Data/TRABAJO FIN DE MASTER/Codigo_TFM/TBM DATA/'+f
    ruta_destino = '/Users/jpgmacbookpro/Google Drive/Master Big Data/TRABAJO FIN DE MASTER/Codigo_TFM/TBM DATA/Salida_Analisis/'+f
    ruta=shutil.move(ruta_origen, ruta_destino)
    print('='*75)
    print("Archivo copiado a",ruta)
    print('='*75)
    
    return df_summary


# Función para el volcado de datos de analisis en MomgoDB
def mn_Connect(DB,col_name,df):
    import pymongo, json
    # Habilitamos la conexión a la Base Datos MongoDB
    mng_client = pymongo.MongoClient('localhost', 27017)
    
    # Direccionamos a la bases y colecciones de MongoDB
    mng_db=mng_client[DB]
    collection_name=col_name
    db_cm=mng_db[collection_name]
    
    # Creamos los objetos json para exportar a Mongo
    data_json = json.loads(df.to_json(orient='records'))
    db_cm.insert_many(data_json)