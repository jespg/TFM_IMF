# TFM_IMF
Monetización de datos mediante Machine Learning en una Turbina de Gas.

Lista de Ficheros del Repositorio.
- Ficheros .csv
  - data_train_0 a 210.csv. Contiene los data set empleados en la simulación de control diario
  - data_train.csv. Contiene el dataset para el entranmiento del modelo inicial
  - file_model.csv. Coeficiente y bias de los modelos entrenados
  - file_var.csv. Contiene la lista de variables procesadas en el proceso de entrenamiento
  - SC_TBM.csv. Lista de variables para la captura de datos vía OPC Server
  - TBM_data.csv. Dataset Original con los datos procedentes del scada
  - var_remove_TBM.csv. Lista de variables que son descartadas preliminarmente
  - var_remove_DL_xxx.csv, Lista de variables a eliminar tras el proceso de entrenamiento
  - data_CCAmb_1 y 2.csv. dataset Modelo 1 y 2 repectivamente, basado en condiciones ambiantales
  - ds_data.csv. Dataset generado para el análisis de series temporales basado en condiciones ambientales
  - file_model_1 y 2.csv, fichero con los coeficientes y bias de modelos 1 y 2

  
- Ficheros ypynb. Notebooks de Jupyter
  - MLconSeleccion_K.ipynb. Notebook con el algotirmo de regresion lineal y selección de variables mediante método SelectKBest
  - MLconSeleccion_r2.ipynb. Notebook con el algoritmo de regresion lineal y selección de variables mediante método Forward de mejor r2.
  - Obtención_Fichero.ipynb. Notebook con el algoritmo para la seleccion de los datasets, basado en intervalos de tiempo o tamaño de las tuplas de datos.
  - OYMconDL_TBM.ipynb. Notebook para la simulación de control diario y almacenamiento de parámetros del modelo en MongoDB.
  - ML_CCAmb.ipynb. Notebook para obtener el modelo entrenado basado en condiciones ambientales
  - Series_CCAmb.ipynb. Notebook, para el anañisis mediante series temporales
  
- Ficheros .py, Script de Python.
  - DataOPC.py. Crea una conexion con el servidor OPC donde corren los datos 
  - OPC_DF.py, crea un fichero csv con los datos capturados desde el OPC Server
  - ETL_V1.py y ETL_Scada.py. Tratamiento del dataset de los datoscaprturados desde el regsitro de historicos del Scada
  - f_selection.py. Contiene diferentes funciones para los algoritmos de seleccion de variables
  - get_files.py. Contiene diferentes funciones para caprutrar y dimensionar los dataset originales.
  - ModDL.py. Contiene las funciones para el test del modelo, registro y almacenamiento de los indicadores del modelo
  - ModReg_TG.py. Contiene las funciones para la adecuacion de los dataset y el entrenamiento del modelo
  - ModReg_TG:CCA.py. Contiene las funciones para la adecuacion de los dataset y el entrenamiento del modelo basado en condiones ambientales.
  - TempBH.py, algoritmo de cálculo iterativo para obtener la Temperatura de Bulbo húmedo.
  
- Ficheros .twb, libros para la visualición tableau.
  - Mod_TG_TBM.twb, Conectado a la fuente de datos la base de datos TG_DB de MongoDB
  
1.Obtenemos desde el Scada un año de datos de procesos del funcionamiento de una turbina de gas. “TBM_data.csv”. Este  fichero consta de 169 variables y más de 10.000 registros.

2. En el Notebook Obtencion_Ficheros.ipynb, y en el script get_files.py, tenemos el código para obtener:
i. Un fichero de entrenamiento para el modelo con los seis primeros meses del año (u otro intervalo de tiempo deseado)
ii. Un obtención masiva de ficheros más pequeños para simular el control diario, que nos servirán para el análisis de patologías de la máquina, sensibilidad a las condiciones del sistema, etc.

3. Ejecutar la primera celda del Notebook Obtencion_Ficheros.ipynb  nos debe dar un fichero “data.train.csv” que lo usaremos en el siguiente para el proceso de aprendizaje automático. Ejecutar la segunda celda nos devolverá 211 ficheros “data_train_(i).csv” para la simulación de control diario.

4. En el Notebook MLconSeleccion_r2.ipynb, obtenemos el modelo entrenado y con las variables seleccionadas, atendiendo al criterio de selección de características Forward de mejor r2, Igualmente en el Notebook MLconSeleccion_K.ipynb, obtenemos el modelo entrenado con el criterio de selección de características SElectKBest. Se utilizan dos scripts ModReg_TG.py y f_Selection.py, que poseen diferentes funciones para el desarrollo del código y obtención de estadísticos.

5. Conclusiones sobre, qué modelo de selección es más eficiente.

6. En Notebook OYMconDL_TBM.ipynb, entrenamos los modelos de control diario. En este caso sólo se utilizan las variables seleccionadas en el paso anterior. Utilizamos el criterio de selección de variables Forward de mejor r2, para obtener la sensibilidad del sistema a cada variable.

7. Mediante el mismo Notebook del paso 6., obtenemos un dataframe con la estructura mostrada en la siguiente tabla

Fecha || Análisis|| Nombre Variable	|| R2	|| Valor Medio || 	% desv Std	||Coef  ||  Variable (W)

viendo 

Estos mismos datos son pasados a formato json, enviados a MongoDB y mediante una conexión entre Mongo y Tableau, pueden ser visualizados. (Los pasos de configuración serán explicados en al memoria del TFM).

El código se recoge en el Script ModDL.py, que contiene las funciones.

8. Por último realizamos un analsis, mediante machine learning basado en las condiciones ambientales. 

La memoria del TFM se presenta en el fichero TFM-comprimido.pdf
