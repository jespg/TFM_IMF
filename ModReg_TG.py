###############################################################
# SCRIPT DE FUNCIONES PARA EL TRATAMIENTO, CARGA Y ENTRENAMIENTO DE DATOS 
###############################################################

# Función para carga del fichero con dataset con todas las variables de la máquina objetivo de estudio
def imp_ds(file):
    import pandas as pd
    # Cargamos el fichero en un dataframe
    df=pd.read_csv(file,sep=';',decimal=',', encoding='ISO-8859-1',low_memory=False)
    # Eliminamos las filas con valors nulos o péridos
    df=df.dropna()
    
    # Capturamos la fecha del dataser para su trazabilidad
    Fecha=df.loc[0][0][0:-5]
    
    # Condición lógica que descarta aquellos valores de potencia que están bajo procesos de regulación o limitación.
    df=df[df['Active power']+200 < df['Power setpoint']]
    df=df[df['Active power']>12000]
    if df.shape[0] > 24:
        # Creamos un fichero con la lista de todas las variables
        var=list(df.columns)
        f=open('file_var.csv','w')
        for i in var:
            f.write(i+'\n')
        f.close()
        print('*'*60)
        print('Tamaño DataSet:',df.shape[0],'filas de datos y ', df.shape[1],'variables registradas')
        print('*'*60)
    else:
        print('Datos en modo regulación o limitación. No hay datos suficientes para una buena simulación.')
    
    return var,df,Fecha
    
#Función Modelo de Regresión Lineal
def f_model_reg(x,y):
    #Entrenamos el modelos Mediante Regresión Lineal Multivariable
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import median_absolute_error

    #Generamos las variables de entrenamiento y test
    x_train,x_test,y_train,y_test=train_test_split(x,y)

    #Modelo de Regresión Lineal
    model=LinearRegression(fit_intercept=True)
    model.fit(x_train,y_train)
    predit_train=model.predict(x_train)
    predit_test=model.predict(x_test)
    
    #Evaluamos los resultados de la Regresión
    r2_training = model.score(x_train,y_train)
    r2_test = model.score(x_test,y_test)
    print('='*70)
    print('R2 en training: %.4f' %(model.score(x_train,y_train)))
    print('R2 en test: %.4f' % (model.score(x_test,y_test)))
    e=abs(model.score(x_train,y_train)-model.score(x_test,y_test))/model.score(x_train,y_train)
    print('Sobre Ajuste--->>> %.4f' % (e))
   
    #Determinación de los estimadores de la calidad del modelo.
    print("Error caudratico medio: %.2f" % (mean_squared_error(predit_train,y_train)))
    print("Error absoluto medio: %.2f"% (mean_absolute_error(predit_train,y_train)))
    print("Mediana del error Absoluto: %.2f" % (median_absolute_error(predit_train,y_train)))
    
    #Obtenemos los coeficientes de la regresión y los llevamos a un fichero
    list_model=[]
    list_xvar=[]
    #print('Termino Ind:%.2f'% (model.intercept_))
    j=0
    for i in list(x.columns):
        list_xvar.append(i)
        list_model.append(model.coef_[j])
        #print(i,'-- %.2f' %(model.coef_[j]))
        j+=1
    list_xvar.append('bias')
    list_model.append(model.intercept_)
    df_fm=pd.DataFrame()
    df_fm['Variable']=list_xvar
    df_fm['Coeficiente']=list_model
    df_fm=df_fm.set_index('Variable')
    df_fm.to_csv('file_model.csv',sep=';',decimal=',',encoding='ISO-8859-1')
    
    return model.coef_,model.intercept_,e,r2_training,r2_test

#Función que obtiene una lista con los valores de r2
def f_score(x,y,var):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import median_absolute_error
    model=LinearRegression(fit_intercept=True)
    list_score=[]
    for i in var:
        xs=x[i].values.reshape(x[i].shape[0],1)
        model.fit(xs,y)
        model_score=model.score(xs,y)
        list_score.append(model_score)
    
    return list_score 
        
# Funcion para la Representacion Grafica de la linealidad de cada variable
def plot_var(var_s,x,y):    
    
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    plt.figure(1,figsize=(30,30))
    j=0
    model_score=[]
    
    #Modelo Lineal de ajuste
    model=LinearRegression()
    
    for i in var_s:
        x_d=x[i].values.reshape(x[i].shape[0],1)
        model.fit(x_d,y)
        xp=[[np.max(x_d)],[np.min(x_d)]]
        yp=model.predict(xp)
    
        plt.subplot(len(var_s)/3+1,3,j+1)
        plt.plot(x_d,y,'r.',label='Datos')
        plt.plot(xp,yp,'b-',label='Modelo')
        plt.legend(loc=1)
        plt.title(i)
        j+=1                 
    plt.savefig("fix_features.png")
    plt.show()
    plt.close()
    
    
#Función que elimina las variables no numéricas y establece la variable dependiente

def Fit_Var(file,var_remove):
    import pandas as pd
    var=file[0]
    df=file[1]
    #y=input('Escribe el nombre de la variables dependiente >>>>>  ')
    y='Active power'
    df_var=pd.read_csv(var_remove,sep=';',encoding='ISO-8859-1',header=None)
    varNoNum=list(df_var[0])
    for i in varNoNum:
        var.remove(i)
    x=df[var].astype(float)
    y=df[y].astype(float)
    return var,x,y
   
#Función que nos devuelve un dataframe con los valores medios y desviación estándar de cada variable
def sense_var(x,y):
    import numpy as np
    import pandas as pd
    var_n=['Active power']
    for i in list(x.columns):
        var_n.append(i)
    val_mean=[np.mean(y)]
    val_std=[np.std(y)/np.mean(y)*100]
    #Creamos las tres listas con las variables y sus estadisticos
    j=0
    
    for i in list(x.columns):
        val_mean.append(np.mean(x[i]))
        val_std.append(np.std(x[i])/np.mean(y)*100)
        j+=1
    df = pd.DataFrame()
    df['Feat']=var_n
    df['Media']=val_mean
    df['Std Dev']=val_std
    df=df.set_index('Feat')
    
    return df
    
# Función que lanza el nuevo modelo de regresion con las variables seleccionadas
def new_model(var_sel,x,y):
    from ModReg_TG import f_model_reg
    import pandas as pd
    x_nm=pd.DataFrame()
    for i in var_sel:
        x_nm[i]=x[i]
    model=f_model_reg(x_nm,y)
    return model

