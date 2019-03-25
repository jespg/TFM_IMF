#### FUNCIONES PARA SELECCIÓN DE VARIABLES ####

### 1. FUNCIÓN EXISTENTE EN SCIKIT-LEARN ###

## Metodo VarianceThreshold. Requiere Normalización.
# VARIANZA = SUM(Xi - Xbar)**2 ( (n-1).

def var_umbral(x,u):
    from sklearn.feature_selection import VarianceThreshold
    import numpy as np

    var_th = VarianceThreshold(threshold = u)
    x_var = var_th.fit_transform(x)

    print("Variables originales ", x.shape[1])
    print("Variables finales ", x_var.shape[1])
    
    var_s=np.asarray(list(x))[var_th.get_support()]
    
    print("Listado de variables ", var_s)
    
    return var_s

## Selección de un número caracterisiticas en función de funciones
# función f_regression para modelos de regresion lineal
# función chi2 para modelos de clasificación

def selectKvar(x,y,k):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    import numpy as np
    import pandas as pd

    var_sk = SelectKBest(f_regression, k)
    x_sk = var_sk.fit_transform(x,y)
    
    # Obtenemos los f_score de cada variable
    values = f_regression(x,y)
    f_score=list(values[0])
    
    # Obtenemos las k variables con mejor f_score
    var_s=np.asarray(list(x))[var_sk.get_support()]
    
    # Obtenemos un data frame con las k mejores variables y sus f_score
    df_s=pd.DataFrame()
    df_s['Variables']=var_s
    df_s['f_score']=f_score[0:k]
    df_s=df_s.sort_values(by='f_score',ascending=False)
    
    return df_s

### 2. METODO BASADO EN EL FACTOR DE INFLACIÓN DE LA VARIANZA.
## VIF = 1 / (1-Ri2). Detecta la colinealidad entre variables 

def calculateVIF(data):
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    
    features = list(data.columns)
    num_features=len(features)
    
    model = LinearRegression()
    
    result=pd.DataFrame(index=['VIF'],columns = features)
    result=result.fillna(0)
    
    for ite in range(num_features):
        x_features=features[:]
        y_featue=features[ite]
        x_features.remove(y_featue)
        
        x=data[x_features]
        y=data[y_featue]
        
        model.fit(data[x_features],data[y_featue])
        R2=model.score(data[x_features],data[y_featue])
        
        if R2 == 1:
            result[y_featue]=1000
        else:
            result[y_featue]=1/(1-R2)
   
    return result

def selectDataUsingVIF(data,max_VIF):
    
    import numpy as np
    
    result=data.copy(deep=True)
    
    VIF = calculateVIF(result)
    
    while VIF.as_matrix().max() > max_VIF:
        col_max=np.where(VIF == VIF.as_matrix().max())[1][0]
        features=list(result.columns)
        features.remove(features[col_max])
        
        result = result[features]
        
        VIF = calculateVIF(result)
       
    return result

### 3. METODO BASADO EN LASSO REGRESION 

def Lasso_Selection(x,y):
    from sklearn.linear_model import Lasso
    
    model_ridge = Lasso(alpha=1)
    model_ridge.fit(x,y)
    
    for i in range(len(list(x.columns))):
        if model_ridge.coef_[i] != 0:
            print(x.columns[i],'>>>',model_ridge.coef_[i])
    

### MODELOS TIPO STEPWISE. TIPO FORWARDSELECTION.
## Selección del modelo en funcion del r2

def ForwardSelection_r2(df_var):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    
    var=df_var[0]
    x=df_var[1]
    y=df_var[2]
    
    #Declaramos las variable donde se almacenan los coeficientes y las variables ordenadas por mejor coeficiente de correlación
    r2_s=[]
    var_s=[]
    
    #Modelo Lineal de ajuste
    model=LinearRegression()
    
    #Calculamos los r2 y los ordenamos por coeficiente de correlación
    for i in range(len(var)):
        var_list=var
        r2_list=[]
        for j in var_list:
            xd=x[j].values.reshape(x[j].shape[0],1)
            model.fit(xd,y)
            r2_list.append(model.score(xd,y)) 
        pos_best=np.argmax(r2_list)
        #print('En el Paso',i,'se ha insertado la variable',var[pos_best],
         #     'con un r2 de %.3f'% r2_list[pos_best])
        r2_s.append(r2_list[pos_best])
        var_s.append(var[pos_best])
        var_list.remove(var[pos_best])
    
    #Añadimos variables en función del mejor r2 y vemos la respuesta del modelo.
    x_row=pd.DataFrame()
    it_r2=[]
    j=0
    #print('='*85)
    for i in var_s:
        x_row[i]=x[i]
        model.fit(x_row,y)
        it_r2.append(model.score(x_row,y))
       # print('En el Paso',j,'se ha insertado la variable',i,
       #   'con un r2 global de %.3f'% it_r2[j])
        j+=1
    
    return var_s, r2_s

## Selección del modelo matematico atendiendo al mejor r2 Acumulado 

def ForwardSelection_r2_Ac(var,x,y):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Modelo para realizar los ajustes
    model = LinearRegression()

    # Variable para almecena los índices de la lista de atributos usados
    var_order =  []
    var_r2 = []

    # Iteración sobre todas las variables
    for i in range(len(var)):
        idx_try = [val for val in range(len(var)) if val not in var_order]
        iter_r2 = []

        for i_try in idx_try:
            useRow = var_order[:]
            useRow.append(i_try)

            use = x[x.columns[useRow]]

            model.fit(use, y)
            r2=model.score(use,y)
            iter_r2.append(r2)

        pos_best = np.argmax(iter_r2)
        var_order.append(idx_try[pos_best])
        var_r2.append(iter_r2[pos_best])
        
    # Obtenemos la lista de variables ordenadas por mejor r2
    var_s=[]
    for i in range(len(var)):
        var_s.append(var[var_order[i]])
    df_s=pd.DataFrame()
    df_s['Variables']=var_s
    df_s['r2']=var_r2
        
    # Grafico en el que representamos r2 vs Variable Añadidas
    plt.plot(range(len(var)),var_r2,'rx-',label='Error vs Nº de Atributo')
    plt.title('Selección de Variables')
    plt.xlabel("Numero de atributos")
    plt.ylabel("r2 Score")
    plt.savefig("select_r2.png")
    plt.show()
        
    return df_s

## Función con los criterios de eliminación y/o selección de variables
def crit_selection(criteria, df_var, var_s):
    import pandas as pd
    from f_selection import calculateVIF, selectDataUsingVIF
    from ModReg_TG import f_score
    
    ## Criterio 1.
    #Eliminamos caracateristicas según un criterio de una desv std menor a un valor dado
    stat=df_var[1]
    df_ds=stat[stat['Std Dev']>criteria[0]]
    var_ds=list(df_ds.index)
    var_ds.remove('Active power')
    
    ## Criterio2.
    #Eliminamos caracteristicas según criterio de un r2 menor que el valor dado
    x=df_var[0][1]
    y=df_var[0][2]
    r2_s=f_score(x,y,var_ds)
    df_r2=pd.DataFrame()
    df_r2['Feat']=var_ds
    df_r2['r2_score']=r2_s
    df_r2=df_r2[df_r2['r2_score']>criteria[1]]
    var_ds_r2=list(df_r2['Feat'])
    
    ## Criterio 3.
    # Crieterio Multicolinealidad VIF superior a un criterio dado
    
    var_=var_ds_r2
    new_x=x[var_]
    
    # Obtenemos los valores del VIF y eliminamos las caracteristicas segun criterio
    df_VIF = calculateVIF(new_x)
    df_VIF_sel=calculateVIF(selectDataUsingVIF(new_x,criteria[2]))
    
    ## Vamos a obtener el data frame con las variables seleccionadas, que incluya:
    # El r2, desv Std y el VIF
    
    #Obtenemos las variables seleccionadas
    var_s_VIF = list(df_VIF_sel.columns)
    
    #Obtenemos los r2 de las variables seleccionadas
    r2_values=f_score(x,y,var_s_VIF)
    
    #Obtenemos los VIF de las variables seleccionadas
    VIF_values=list(df_VIF_sel.values[0])
    
    #Obtenemos las Desv Std de las variables seleccionadas
    list_std=[]
    list_i=[]
    for i in var_ds:
        if i in var_s_VIF:
            list_std.append(df_ds['Std Dev'][i])
    
    # Construimos el data frame final
    df_r2_VIF=pd.DataFrame()
    df_r2_VIF['Feat']=var_s_VIF
    df_r2_VIF['r2 Score']=r2_values
    df_r2_VIF['Desv Std']=list_std
    df_r2_VIF.set_index('Feat')
    df_r2_VIF=df_r2_VIF.sort_values(by='r2 Score',ascending=False)
    print('Variables Seleccionadas Final:',len(var_s_VIF))
    
    return list(df_r2_VIF['Feat'])
    
# Función pora obtener una lista de variables descartadas 

def var_remove_DL(file,var_sel):
    import pandas as pd
    df=pd.read_csv(file,sep=';',decimal=',', encoding='ISO-8859-1')
    l=list(df.columns)
    ll=[]
    f=open('var_remove_DL_XXXXX.csv','w')
    for i in l:
        if i not in var_sel:
            f.write(i+'\n')
    f.close()

    



