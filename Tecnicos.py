import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pylab as plt
class Tecnichal_Analisis:

    @classmethod
    def SMA(cls,df:DataFrame, period: int = 9,  column: str = "Cierre") -> Series:
        """
        SMA o "simple moving average se refiere a el promedio movil simple,
        Este es el elemento más basico utilizado en traiding, sólo se le define el periodo y
        el nombre de la columna de los datos sobre los cuales se quiere calcular
        """
        return pd.Series(df[column].rolling(center=False, window=period, min_periods = period -1).mean()
                         ,name="{} period SMA".format(period))
    @classmethod
    def SMM(cls, df: DataFrame, period: int = 9, column: str = "Cierre") -> Series:
        """"
        La mediana movil simple es una alternativa a SMA. SMA es usado para hacer estimaciones de tendencias en series de tiempo,
        es suceptible a eventos raros tales como cambios bruscos y otro tipo de anomalias. Un estimador más robusto
        para estimar una tendencia es la mediana movil simple.
        """
        return pd.Series(df[column].rolling(center=False,window=period, min_periods=period -1 ).median()
                         ,name="{} period SMM".format(period))
    @classmethod
    def EMA(cls,df:DataFrame, period: int = 9, column: str = "Cierre") -> Series:
        """
        Media movil pesada exponencialmente, esta parece ser más utilizada en los sistemas de mercado.
        Si el mercado muestra un alza fuerte y bien definida, el indicador EMA mostrará también el
        crecimiento y vice-versa for un comportamiento de bajada. EMA es utilizado comunmente en conjunto con otros indicadores
        que confirman el movimiento en el mercado y así calibrar su validez. Se calibra tal que el peso exponencial cumpla que el
        centro de masa del SMA y el del EMA sean los mismos.
        """
        return pd.Series(df[column].ewm(ignore_na=False, min_periods=period - 1, span=period).mean(),
                        name="{} period EMA".format(period))
    @classmethod
    def DEMA(cls,df:DataFrame, period: int = 9,column: str = "Cierre")-> Series:
        """
        Media movil doble exponencial, este indicador intenta eliminar el retraso asociado a las medias moviles poniendo
        más peso en los valores recientes. Este nombre sugiere que se puede hacer aplicando 2 veces un promedio de una
        media movil exponencial, lo cual no es cierto. Este nombre vien del hecho de que el valor de un EMA es doblado.
        Con el fin de mantener este indicador pegado a la linea de los datos, y remover el retras, el valor EMA de EMA es
        restado del valor doblado del EMA.
        Debido a que EMA(EMA) está involucrado en los calculos, DEMA necesita 2*period -4 muestras para empezar a producir datos
        no como en el caso del EMA que requiere period-1
        """
        DEMA = (2*cls.EMA(df, period,column)- cls.EMA(df, period,column).ewm(ignore_na=False, min_periods=period - 1, span=period).mean())

        return pd.Series(DEMA,name="{} period DEMA".format(period))
    @classmethod
    def TEMA(cls,df:DataFrame,period: int = 9, column: str= "Cierre") -> Series:
        """"
        Media movil triple exponencial, intenta remover el retraso inherente asociado a las medias moviles
        poniendo más peso sobre los valores más recientes. Aunque su nombre sugiere que que se obtiene al aplicar
        tres veces el promedio movil exponencial, esto no es cierto. este estimador se obtiene al hacer una media movil
        exponencial el triple de su valor.
        Para mantener la linea del promedio en el mismo rango de los datos, el valor del "EMA de EMA" se le resta 3 veces al valor
        del triple del EMA y por ultimo se le agrega el EMA del EMA del EMA.
        Dado que el calculo de EMA(EMA(EMA)) está involucrado en nuestros calculos, el indicador TEMA necesita 3*period-4
        (se puede imponer para que tenga al igual que el DEMA 2N-2)
        https://en.wikipedia.org/wiki/Exponential_smoothing
        """
        triple_ema = 3*cls.EMA(df,period,column)
        ema_ema_ema = cls.EMA(df,period,column).ewm(ignore_na=False,span=period).mean().ewm(ignore_na=False,span=period).mean()
        #calculo con 3N-4
        #TEMA=triple_ema - 3*cls.EMA(df,period,column).ewm(ignore_na=False,min_periods=2*period -1 ,span=period).mean() + ema_ema_ema
        #----------------------------------------------------------------------
        #calculo con 2N-2
        TEMA=triple_ema - 3*cls.EMA(df,period,column).ewm(ignore_na=False,min_periods=period -1 ,span=period).mean() + ema_ema_ema
        return pd.Series(TEMA,name="{} period TEMA".format(period))
        #raise NotImplementedError
    @classmethod
    def TRIMA(cls, df: DataFrame, period: int = 18,column: str="Cierre") -> Series:
        """"
        El promedio movil triangular (TRIMA) o TMA representa un promedio de precios, salvo que se impone el peso sobre
        la mitad del tiempo establecido en el periodo.
        El calculo suaviza dos veces los datos haceindo uso de una ventana de tamaño la mitad de la longitud de la serie.
        https://www.thebalance.com/triangular-moving-average-tma-description-and-uses-1031203
        """
        TRIMA=cls.SMA(df,period,column).rolling(window=period,min_periods=period-1).mean()
        return pd.Series(TRIMA,name="{} period TRIMA".format(period))
        #raise NotImplementedError
    @classmethod
    def TRIX(cls,df:DataFrame,period:int=15,column: str="Cierre") -> Series:
        """"
        El indicador de la triple media exponencial oscilatoria movil (TRIX) hecho por Jack Hutson es un indicador de
        "momentum", el cual oscila al rededor de cero, valores negativos significan tendencia a decrecer y valores
        positivos indican tendencia a crecer.
        Esto muestra el porcentaje de la rata de cambio entre 2 medias moviles exponenciales triples.
        https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/triple-exponential-moving-average-oscillator-trix/
        se recomienda tener el TRIX en un valor de 14 periodos pero para responder de forma más rapida, se recomienda un valor de 12
        con una linea de señal en 4.
        https://forex-indicators.net/momentum-indicators/trix
        Para calcular el TRIX se calcula el triple promedio exponencial de n periodos y luego se resta su diferencia y se divide por esa diferencia
        para ver el porcentaje relativo de cambio.
        """
        EMA3=cls.EMA(df,period,column).ewm(span=period).mean().ewm(span=period).mean()
        TRIX = EMA3.diff()/(EMA3-EMA3.diff())
        return pd.Series(TRIX, name="{} period TRIX".format(period))

    @classmethod
    def VAMA(cls,df:DataFrame, period: int =8, column: str="Cierre",volume: str= "Cantidad") -> Series:
        """
        VAMA o VWAP consiste en el promedio movil ponderado por el volumen tranzado.
        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vwap_intraday
        """
        vp = df[volume]*df[column]
        volsum = df[volume].rolling(window=period).mean()
        volRatio = pd.Series(vp/volsum,name="VAMA")
        cumSum = (volRatio*df[column]).rolling(window=period).sum()
        cumDiv = volRatio.rolling(window=period).sum()
        return pd.Series(cumSum/cumDiv,name="{} period VAMA".format(period))
    @classmethod
    def VIDYA(cls,df:DataFrame,smoothing_period_slow:int = 30, smoothing_period_fast: int = 9, a :float = 0.5,column: str = "Cierre") -> Series:
        """
        el indicador VIDYA "promedio dinamico variable del indice" (variable index dynamic average) es una modificación de
        la media movil exponencial EMA. La diferencia principal entre el EMA y VIDYA es que el factor de "suavizado" se calcula de
        forma distinta.
        En el EMA el valor del peso de los promedios está dado por F=2/(period+1)
        en VIDYA el factor de suavizado es algo que fluctua en el tiempo en función del movimiento de los precios.
        https://tulipindicators.org/vidya
        el valor de la cosntante que multiplica será 0.5
        """
        Small_Move=df[column].rolling(window=smoothing_period_fast).std()
        Large_Move=df[column].rolling(window=smoothing_period_slow).std()
        d={"Std short":Small_Move,"Std long":Large_Move}
        variances=pd.DataFrame(data=d)
        variances["Alpha"]=a*variances["Std short"]/variances["Std long"]
        variances.loc[(variances["Std long"]<10**(-6)) | (variances["Alpha"]>1),["Alpha"]]=2.0/(smoothing_period_slow+1)
        vidya=[]
        sma=pd.Series(df[column].rolling(smoothing_period_slow).mean(),name="Simple moving average")
        for s, ma, price in zip(variances["Alpha"].items(),sma.shift().items(),df[column].items()):
            try:
                vidya.append((1-s[1])*vidya[-1]+s[1]*price[1])
            except(IndexError,TypeError):
                if pd.notnull(ma[1]):
                    vidya.append((1-s[1])*ma[1]+s[1]*price[1])
                else:
                    vidya.append(None)
        sma["VIDYA"]=pd.Series(vidya,index=sma.index,name="{} period VIDYA".format(smoothing_period_slow))
        return sma["VIDYA"]
        #raise NotImplementedError
    @classmethod
    def ER(cls,df:DataFrame,period: int = 10,column:str="Cierre")-> Series:
        """
        El indicador de eficiencia de Kaufman es un indicador que oscila entre valores de 1 y -1, en donde cero es
        el punto medio, se recomienda esperar a que esté por encia o por debajo de .3 (-.3).
        valores cercanos a 1 indican una tendencia al laza y valores cercanos a -1 indican tendencias a la baja.
        Se calcula dicidiendo el cambio del precio sobre un periodo establecido sobre la volatilidad la cual
        corresponde a la suma de las medias moviles de las diferencias del periodo establecido.
        """
        change= df[column].diff(period)
        volatility=df[column].diff().abs().rolling(window=period).sum()
        d={"Change":change,"Volatility":volatility}
        resume=pd.DataFrame(data=d)
        resume["ER"]=resume["Change"]/resume["Volatility"]
        resume.loc[resume["Volatility"]==0,["ER"]]=1
        return pd.Series(resume["ER"], name="{} period ER".format(period))
    @classmethod
    def KAMA(cls,df:DataFrame,er:int = 10, ema_fast: int = 2, ema_slow: int = 30, period:int =20,column:str="Cierre")-> Series:
        '''
        El indicador de la media movil adaptativa de Kaufman intenta ajustar con respecto a la condición actual
        del mercado. este indicador se adapta a un movimiento promedio rapido cuando los precios parecen no
        cambiar mucho y se adapta a un promedio lento cuando el mercado presenta mucho ruido. La ventaja es que no sólo
        tiene en cuanta la dirección del mercado si no la volatilidad de este.
        TOma un parámetro de eficiencia de periodo n
        https://tulipindicators.org/kama
        '''
        er=cls.ER(df,er,column)
        fast_alpha=2/(ema_fast+1)
        slow_alpha=2/(ema_slow+1)
        sc=pd.Series((er*(fast_alpha-slow_alpha)+slow_alpha)**2,name="Smoothing constant")
        sma=pd.Series(df[column].rolling(period).mean(),name="Simple moving average")
        kama=[]
        aux=0
        for s, ma, price in zip(sc.items(),sma.shift().items(),df[column].items()):
            try:
                kama.append(kama[-1]+s[1]*(price[1]-kama[-1]))
            except(IndexError,TypeError):
                if pd.notnull(ma[1]):
                    kama.append(ma[1]+s[1]*(price[1]-ma[1]))
                else:
                    kama.append(None)
        sma["KAMA"]=pd.Series(kama,index=sma.index,name="{} period KAMA".format(period))
        return sma["KAMA"]
    @classmethod
    def MACD(cls,df:DataFrame,period_fast=12,period_slow: int =26, signal:int =9,column: str = "Cierre")-> Series:
        EMA_fast = pd.Series(df[column].ewm(ignore_na=False,min_periods=period_slow-1,span=period_fast).mean(),name="EMA fast")
        EMA_slow = pd.Series(df[column].ewm(ignore_na=False,min_periods=period_slow-1,span=period_slow).mean(),name="EMA slow")
        MACD = pd.Series(EMA_fast-EMA_slow,name="MACD")
        MACD_signal = pd.Series(MACD.ewm(ignore_na=False, span = signal).mean(),name="SIGNAL")
        MACD_histogram = pd.Series(MACD-MACD_signal,name="Histogram")
        return pd.concat([MACD,MACD_signal,MACD_histogram],axis=1)

    @classmethod
    def MACD_TRIX(cls, df:DataFrame, period:int = 5 ,period_mid: int= 5, column:str="last"):
        TRIX = pd.Series(cls.TRIX(df,column=column,period=period),name="TRIX")
        TRIX_signal = pd.Series(TRIX.ewm(ignore_na=False,span=period_mid).mean(),name="SIGNAL")
        TRIX_histogram = pd.Series(TRIX-TRIX_signal,name="Histogram")
        return pd.concat([TRIX,TRIX_signal,TRIX_histogram],axis=1)

class Back_Testing:
    @classmethod
    def Inicializar_resumen_TRIX_1(cls,df:DataFrame,period:int=12,period_mid:int=9,name_offer:str="Offer",name_bid:str="Bid",resample: str="1D"):
        """
        Esta clase es la encargada de definir la estrategia para comprar o vender, compra en el caso en el que encuentre un
        cambio en la concavidad en el oscilador.
        la idea es que este vende cuando se encuentra un valor negativo y compra en el caso contrario
        """
        tc=Tecnichal_Analisis()
        data=(df[name_bid]+df[name_offer])*0.5
        resampled_data=data.resample(resample,label="left").std()
        Id_begin = data.index[0]
        Id_final = resampled_data.index[0]
        resume={"Precio":[],"Estado":[]}
        resume=pd.DataFrame(data=resume)
        for i in range(len(resampled_data)):
            histogram=tc.MACD_TRIX(data.loc[Id_begin:Id_final],period=period,period_mid=period_mid,column="last")["Histogram"]
            sign_y = histogram.loc[((histogram.diff()*histogram.shift(1).diff()<0) & (np.abs(histogram)>histogram.std()))]
            sign_x=sign_y.index
            condicion_c=sign_y.loc[sign_y>0].index
            condicion_v=sign_y.loc[sign_y<0].index
            Precio_venta=df[name_offer].loc[condicion_v]["first"]
            Precio_compra=df[name_bid].loc[condicion_c]["first"]
            Compra = ["C"]*len(Precio_compra)
            Venta = ["V"]*len(Precio_venta)
            temporal_data_C={"Precio":Precio_compra,"Estado":Compra}
            temporal_data_V={"Precio":Precio_venta,"Estado":Venta}
            temporal_data_C=pd.DataFrame(data=temporal_data_C)
            temporal_data_V=pd.DataFrame(data=temporal_data_V)
            resume=pd.concat([resume,temporal_data_C,temporal_data_V])
            if i+1 == len(resampled_data):
                continue
            else:
                Id_begin=Id_final
                Id_final=resampled_data.index[i+1]
        resume=resume.sort_index()
        return resume

    @classmethod
    def Inicializar_resumen_TRIX_2(cls,df:DataFrame,period:int=12,period_mid:int=9,name_offer:str="Offer",name_bid:str="Bid",resample:str="1D"):
        """
        Esta clase es la encargada de definir la estrategia para comprar o vender, compra en el caso en el que encuentre un
        cambio en la concavidad en el oscilador.
        la idea es que este vende cuando se encuentra un valor positivo y compra en el caso contrario
        """
        tc=Tecnichal_Analisis()
        data=(df[name_bid]+df[name_offer])*0.5
        resampled_data=data.resample(resample,label="left").std()
        Id_begin = data.index[0]
        Id_final = resampled_data.index[0]
        resume={"Precio":[],"Estado":[]}
        resume=pd.DataFrame(data=resume)
        for i in range(len(resampled_data)):
            histogram=tc.MACD_TRIX(data.loc[Id_begin:Id_final],period=period,period_mid=period_mid,column="last")["Histogram"]
            sign_y = histogram.loc[((histogram.diff()*histogram.shift(1).diff()<0) & (np.abs(histogram)>histogram.std()))]
            sign_x=sign_y.index
            condicion_c=sign_y.loc[sign_y<0].index
            condicion_v=sign_y.loc[sign_y>0].index
            Precio_venta=df[name_offer].loc[condicion_v]["first"]
            Precio_compra=df[name_bid].loc[condicion_c]["first"]
            Compra = ["C"]*len(Precio_compra)
            Venta = ["V"]*len(Precio_venta)
            temporal_data_C={"Precio":Precio_compra,"Estado":Compra}
            temporal_data_V={"Precio":Precio_venta,"Estado":Venta}
            temporal_data_C=pd.DataFrame(data=temporal_data_C)
            temporal_data_V=pd.DataFrame(data=temporal_data_V)
            resume=pd.concat([resume,temporal_data_C,temporal_data_V])
            if i+1 == len(resampled_data):
                continue
            else:
                Id_begin=Id_final
                Id_final=resampled_data.index[i+1]
        resume=resume.sort_index()
        return resume


    @classmethod
    def Filtro_Compra_Venta(cls,df_2:DataFrame,Quantity:int=250,Max_trade:int=4,column:str="Estado",name_sale:str="V",name_buy:str="C"):
        """"
        El data frame que ingrese debe estar ordenado cronologicamente con las ordenes de compra y venta
        de lo contrario este metodo no funciona
        Especificamente el dataFrame que ingresa acá es uno con las posibles señales de compra y venta que pueden
        surgir a partir de un algoritmo, esta funcion resulta ser general para cualquier algoritmo, ayuda a que se
        tenga un control del maximo de la posición abierta y así evaluar la posibilidad de que la estrategia cambie
        """
        df=df_2.copy().reset_index()
        Max_quantity=Quantity*Max_trade
        Total=0
        status_Compra=False
        status_Venta=False
        Dicc_Compra={}
        Dicc_Venta={}
        for i in df.index:
            if df[column].iloc[i] == name_sale:
                if Total < Max_quantity: # it can keep selling
                    if status_Compra: # it has to sell everything that have bought
                        Dicc_Venta[i] = -Total
                        Total += -Total
                        status_Compra = False # it has nothing left, here start a new round
                        status_Venta = False
                    else:
                        Total += Quantity # it sell what we tell it to sell
                        Dicc_Venta[i]=Quantity
                        status_Venta=True # we keep track of how many it is selling
                        status_Compra=False
            else:
                if Total > -Max_quantity:
                    if status_Venta:
                        Dicc_Compra[i] = -Total
                        Total += -Total
                        status_Venta = False
                        status_Compra = False
                    else:
                        Total += -Quantity
                        Dicc_Compra[i]= -Quantity
                        status_Compra=True
                        status_Venta=False
        df.loc[list(Dicc_Compra.keys()),"Total"]=list(Dicc_Compra.values())
        df.loc[list(Dicc_Venta.keys()),"Total"]=list(Dicc_Venta.values())
        df["Cumulative"]=df.Total.cumsum()
        df.set_index("index",inplace=True)
        df.index.name=None
        return df

    @classmethod
    def  Obtener_resumen_estrategia(cls,df:DataFrame,column_price:str="Precio",column_total:str="Total"):
        rounds=df.reset_index().loc[df.reset_index()["Cumulative"]==0].index.tolist()
        aux=0
        close_position=[]
        dates_round=[]
        for i in rounds:
            close_position.append((df.iloc[aux:i+1][column_price]*df.iloc[aux:i+1][column_total]).sum()*1000)
            dates_round.append(df.iloc[aux:i+1].index[-1])
            aux=i+1
        data={"Position":pd.Series(close_position,index=dates_round),"Cumulative":pd.Series(np.cumsum(close_position),index=dates_round)}
        data=DataFrame(data=data)
        data["Time past"]=(data["Cumulative"].index - data.index[0]).astype('timedelta64[h]')
        return data

    @classmethod
    def Plot_Resumen_estrategia(cls,df:DataFrame,title:str=None,plot_rounds:bool=False,plot_cumulative:bool=True,figsize:tuple=(17,7),column_position:str="Position",column_cumulative:str="Cumulative",color_round:str="rebeccapurple",color_cumulative:str="royalblue"):
        if plot_rounds and plot_cumulative:
            fig=plt.figure(figsize=(25,10))
            plt.subplot(121)
            df[column_position].plot(figsize=figsize,color=color_round)
            plt.ticklabel_format(axis="y",style="sci",scilimits=(4,4),useMathText=True)
            plt.xlabel("Dates")
            plt.ylabel("USD")
            plt.subplot(122)
            df[column_cumulative].plot(figsize=figsize,color=color_cumulative)
            plt.ticklabel_format(axis="y",style="sci",scilimits=(4,4),useMathText=True)
            plt.xlabel("Dates")
            plt.ylabel("USD")
        elif plot_rounds and not plot_cumulative:
            df[column_position].plot(figsize=figsize,color=color_round,title=title)
            plt.ticklabel_format(axis="y",style="sci",scilimits=(4,4),useMathText=True)
            plt.xlabel("Dates")
            plt.ylabel("USD")
        elif plot_cumulative and not plot_rounds:
            df[column_cumulative].plot(figsize=figsize,color=color_cumulative,title=title)
            plt.ticklabel_format(axis="y",style="sci",scilimits=(4,4),useMathText=True)
            plt.xlabel("Dates")
            plt.ylabel("USD")
    @classmethod
    def Resumen_acomulado_estrategia_1(cls,df:DataFrame,period:int=12,period_mid:int=9,name_offer:str="Offer",name_bid:str="Bid",resample: str="1D",Quantity:int=250,Max_trade:int=4):
        orders=cls.Inicializar_resumen_TRIX_1(df,period=period,period_mid=period_mid,name_offer=name_offer,name_bid=name_bid,resample=resample)
        resume=cls.Filtro_Compra_Venta(orders,Quantity=Quantity,Max_trade=Max_trade)
        Acumulate=cls.Obtener_resumen_estrategia(resume)
        return Acumulate

    @classmethod
    def Resumen_acomulado_estrategia_2(cls,df:DataFrame,period:int=12,period_mid:int=9,name_offer:str="Offer",name_bid:str="Bid",resample: str="1D",Quantity:int=250,Max_trade:int=4):
        orders=cls.Inicializar_resumen_TRIX_2(df,period=period,period_mid=period_mid,name_offer=name_offer,name_bid=name_bid,resample=resample)
        resume=cls.Filtro_Compra_Venta(orders,Quantity=Quantity,Max_trade=Max_trade)
        Acumulate=cls.Obtener_resumen_estrategia(resume)
        return Acumulate
