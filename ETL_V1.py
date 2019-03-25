#Seleccionamos el fichero con los datos a procesar y damos nombre al fichero procesado
from ETL_Scada import ETL_Scada

file=input('Introduce el nombre con los datos en bruto:\n>>>>')
file_exit=input('Introduce el nombre del fichero con los datos procesados:\n>>>>')

file_exit=ETL_Scada(file,file_exit)