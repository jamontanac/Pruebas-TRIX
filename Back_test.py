import time
from Tecnicos import Tecnichal_Analisis as tc
from Tecnicos import Back_Testing as BT
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pylab as plt
import numpy as np
import os
import itertools
from scipy.optimize import curve_fit
import sys
df=pd.read_csv("periodo3.csv",header=[0,1],index_col=[0])
df.index = pd.to_datetime(df.index)
df=df.sort_index()
def l(x,a,b):
	return a*x+b
a=range(3,40)
resampled=sys.argv[1]
archivo_best=open("Resultados_95_"+resample+".txt","w")
archivo_rest=open("Resultados_8_"+resample+".txt","w")
archivo_best.write("Period"+"\t"+"Period mid"+"\t"+"R\n")
archivo_rest.write("Period"+"\t"+"Period mid"+"\t"+"R\n")
for i in itertools.product(a,repeat=2):
	period=i[0]
	period_mid=i[1]
	A=BT.Resumen_acomulado_estrategia_2(df,period=period,period_mid=period_mid,resample=resampled)
	R=np.corrcoef(A["Time past"],A["Cumulative"])[0,1]
	if np.abs(R) >= 0.95:
		popt, pcov = curve_fit(l, A["Time past"], A["Cumulative"])
		BT.Plot_Resumen_estrategia(A,plot_cumulative=True,plot_rounds=False,title="Period = {} and Middle Period = {}".format(period,period_mid))
		l(A["Time past"], *popt).plot(color="forestgreen",style="--",label="Ajuste")
		archivo_best.write(str(period)+"\t"+str(period_mid)+"\t"+str(R)+"\n")
		plt.legend()
		plt.savefig("./"+resampled+"/Resultado_period"+str(period)+"periodmid"+str(period_mid)+".png")
		plt.close()
	elif np.abs(R)>=0.8 and np.abs(R)<0.95:
		archivo_rest.write(str(period)+"\t"+str(period_mid)+"\t"+str(R)+"\n")
archivo_best.close()
archivo_rest.close()
