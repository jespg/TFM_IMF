### Función de cálcuolo de la Temperatura de Bulbo húmedo ###

def TempBH(Tamb,HR,Patm):
    
    import math
    # Lista de Constantes del método de cálculo de Carrier.
    A = 1.2378847e-5
    B = -1.9121316e-2
    C = 33.93711047
    D = -6343.1645
    THETA = 1940
    JI = -1.44

    # Método de Carrier
    e = 0 # Variable comparativa del error en cada iteracions 
    
    Ps = math.exp(A * (Tamb + 273.15) ** 2 + B * (Tamb + 273.15) + C + D / (Tamb + 273.15))
    P = Ps * HR
    Tbh0 = 0
    Tbh = []
    i = 0 # Contador de iteraciones
    Tbh.append(Tbh0)
    # Iteramos hasta que el error entre ambos término sea de 0.01
    while (e < 0.99):
        Psw = math.exp(A * (Tbh[i] + 273.15) ** 2 + B * 
                       (Tbh[i] + 273.15) + C + D / (Tbh[i] + 273.15))
        Pw = Psw -((Patm * 100 - Psw) * (Tamb - Tbh[i])) / (THETA + JI * (Tbh[i] + 273.15))
        T = Tbh[i] + 0.1
        i = i + 1
        e = Pw / P
        Tbh.append(T)
    print("La Temperatura de bulbo humedo es: %.1fºC en %d iteracoines realizadas"%(T,i))
    
    return T

