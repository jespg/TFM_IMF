import OpenOPC
import pandas as pd
import time

def DataOPC(n,timer):
    opc = OpenOPC.open_client('localhost')
    opc.connect('Matrikon.OPC.Server')
    i=0
    list_val=[]
    list_time=[]
    f=open('SC_TBM.csv','r')
    sc_var=f.read().split('\n')
    while i <= n:
        val = opc.read(sc_var)
        df=pd.DataFrame(val)
        list_val.append(list(df[1]))
        list_time_all=(list(df[3]))
        list_time.append(list_time_all[0])
        print('Dato It',i,'>>>>',list_val[i])
        list_var=list(df[0])
        i+=1
        time.sleep(timer)
    return list_val, list_var, list_time

# Incluir código asociado a la generación de un data frame desde el OPC


############################################################################
############################################################################
