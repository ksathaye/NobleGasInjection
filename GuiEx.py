# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:11:55 2015

Plotting Examples Interactive

@author: kiransathaye
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import math


def Fg(SG,InitialValue):
    #sets up fractional flow function as a function of gas saturation and residual water saturation

    MobRatio=InitialValue['ViscRat']
    #ResidualW=InitialValue['ResWater']
    #ResidualG=InitialValue['ResGas']

    ViscG=1e-5;# set viscosities
    ViscW=MobRatio*ViscG;

    KW = RKW(SG,InitialValue,0); # call relative permeabilities
    KG = RKG(SG,InitialValue,0);
    MG=KG/ViscG;
    MW=KW/ViscW;
    F=MG/(MW+MG);
    #if np.shape(F)>1:
        #F[SG>1-ResidualW]=1;
    return F;

def RKW(SG,InitialValue,deriv):
    #relative permeability of liquid phase as function of gas saturation and residual water saturation
    #deriv option for derivative of rel perm
    ResidualW=InitialValue['ResWater']
    ResidualG=InitialValue['ResGas']
    WaterExp=InitialValue['LiqExp']
    MaxRelPerm=InitialValue['MaxWater']

    KTest=((1-SG-ResidualW)/(1-ResidualG-ResidualW))
    if np.size(KTest)==1:
        if KTest<0:
            KTest=0

    K=MaxRelPerm*KTest**WaterExp;
    if np.size(SG)>1:
        KTest[KTest<0]=0
        K=KTest**WaterExp;
        K[SG>1-ResidualW]=0
        K[SG<ResidualG]=1
    if deriv==1:
        dKTest=(1-SG-ResidualW)/(1-ResidualG-ResidualW)
        if np.size(dKTest)==1:
            if dKTest<0:
                dKTest=0
        dK=-MaxRelPerm*WaterExp*dKTest**(WaterExp-1)
        if np.size(SG)>1:
           dKTest[dKTest<0]=0
           dK=-WaterExp*dKTest**(WaterExp-1)
           dK[SG>1-ResidualW]=0
           dK[SG<ResidualG]=0
        return dK;
    else:
        return K;

def RKG(SG,InitialValue,deriv):
    #relative permeability of gas phase as function of gas saturation and residual water saturation
    #deriv option for derivative of rel perm

    ResidualW=InitialValue['ResWater']
    ResidualG=InitialValue['ResGas']
    GasExp=InitialValue['GasExp']
    MaxGas=InitialValue['MaxGas']

    K=MaxGas*((SG-ResidualG)/(1-ResidualG-ResidualW))**GasExp;

    if np.size(SG)>1:
       K[SG<ResidualG]=0
       K[SG>1-ResidualW]=InitialValue['MaxGas']

    if deriv==1:
        dKTest=(SG-ResidualG)/(1-ResidualG-ResidualW)
        if np.size(SG)>1:
           dKTest[dKTest<0]=0
           dK=MaxGas*GasExp*(dKTest**(GasExp-1));
           dK[SG<ResidualG]=0
           dK[SG>1-ResidualW]=0
        else:
             if dKTest<0:
                dKTest=0
        dK=MaxGas*GasExp*(dKTest**(GasExp-1));
        return dK;
    else:
        return K;

def df(SG,InitialValue):
    #computes derivative of fractional flow function as a function of gas saturation and residual water

    MobRatio=InitialValue['ViscRat']
    #ResidualW=InitialValue['ResWater']
    #ResidualG=InitialValue['ResGas']

    ViscG=1e-5;# set viscosities
    ViscW=MobRatio*ViscG;

    KW = RKW(SG,InitialValue,0);
    KG = RKG(SG,InitialValue,0);
    MG=KG/ViscG;
    MW=KW/ViscW;

    dKW = RKW(SG,InitialValue,1);
    dKG = RKG(SG,InitialValue,1);
    dMG=dKG/ViscG;
    dMW=dKW/ViscW;

    dF=(dMG*MW-dMW*MG)/((MG+MW)**2);
    return dF;

def TanFind(InitialValue):

    #ResidualG=InitialValue['ResGas']
    ResidualW=InitialValue['ResWater']
    #MobRatio=InitialValue['ViscRat']
    InjSg=1-InitialValue['ResWater']

    Sg=np.linspace(0,1,1e6)
    F=Fg(Sg,InitialValue)
    dF=df(Sg,InitialValue)
    FInj=Fg(InjSg,InitialValue)
    dFInj=df(InjSg,InitialValue)

    Obj=np.abs((FInj-F)/(InjSg-Sg)-dF)
    Obj[np.abs(Obj)>2]=np.nan
    Obj[Sg>=1-ResidualW]=np.nan
    Obj[Sg>=InjSg]=np.nan
    dObj=np.diff(Obj)
    dS=Sg[1]-Sg[0]

    MinInd=np.nanargmin(dObj)
    MinS=Sg[MinInd]
    ShockSpeed=dF[MinInd]

    dF[dF>ShockSpeed]=np.nan
    ShockSatObj=(np.abs(dF-ShockSpeed))
    ShockSatObj[ShockSatObj>ShockSpeed]=np.nan
    ShockSatObj[0:MinInd]=np.nan
    ShockSatInd=np.nanargmin(ShockSatObj)

    N=int(np.max(np.where(np.isnan(dF))))

    ShockSat=Sg[N]
    SpeedProf=dF[N:]
    SatProf=Sg[N:]
    SpeedProf[0]=SpeedProf[1]
    SatProf[0]=0
    outDict={'Speed': SpeedProf}
    outDict['Saturation']=SatProf
    #outFrame=pd.DataFrame(np.transpose([SpeedProf,SatProf]),columns=('Speed','Saturation'))

    return outDict

InitialValue={'LiqExp':2.5}
InitialValue['ResGas']=0
InitialValue['ResWater']=0.2
InitialValue['GasExp']=2
InitialValue['MaxWater']=1
InitialValue['MaxGas']=0.8
InitialValue['ViscRat']=100

outFrame=TanFind(InitialValue)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.05, bottom=0.4)
l, = plt.plot(outFrame['Speed'], outFrame['Saturation'], lw=2, color='red')
plt.axis([0, 2, -0.05, 1])
plt.grid()
plt.ylabel('Gas Saturation')
plt.xlabel('Speed (x/t)')

axcolor = 'lightgoldenrodyellow'

axDict={'ResWater': plt.axes([0.25, 0.2, 0.65, 0.03], axisbg=axcolor)}
axDict['ResGas']= plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
axDict['GasExp']= plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axDict['LiqExp']=plt.axes([0.25, 0.05, 0.65, 0.03], axisbg=axcolor)
#axDict['MaxWater']=plt.axes([0.25, 0.25, 0.65, 0.03], axisbg=axcolor)
axDict['MaxGas']= plt.axes([0.25, 0.25, 0.65, 0.03], axisbg=axcolor)
axDict['ViscRat']= plt.axes([0.25, 0.3, 0.65, 0.03], axisbg=axcolor)

SliderDict={'LiqExp': Slider(axDict['LiqExp'], 'Liquid Perm Exponent', 1,4, valinit=InitialValue['LiqExp'])}
SliderDict['ResGas']=Slider(axDict['ResGas'], 'Residual Gas', 0., .5, valinit=InitialValue['ResGas'])
SliderDict['ResWater']= Slider(axDict['ResWater'], 'Residual Water', 0., .5, valinit=InitialValue['ResWater'])
SliderDict['GasExp']= Slider(axDict['GasExp'], 'Gas Perm Exponent', 2, 4, valinit=InitialValue['GasExp'])
#SliderDict['MaxWater']= Slider(axDict['MaxWater'], 'Max Water Rel Perm', 0, 1, valinit=InitialValue['MaxWater'])
SliderDict['MaxGas']= Slider(axDict['MaxGas'], 'Max Gas Rel Perm', 0, 1, valinit=InitialValue['MaxGas'])
SliderDict['ViscRat']= Slider(axDict['ViscRat'], 'Viscosity Ratio', 1, 1000, valinit=InitialValue['ViscRat'])

resetax = plt.axes([.78, 0.8, 0.1, 0.07])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def update(val):
    TanInput={'LiqExp':SliderDict['LiqExp'].val}
    TanInput['ResGas']=SliderDict['ResGas'].val
    TanInput['ResWater']=SliderDict['ResWater'].val
    TanInput['GasExp']=SliderDict['GasExp'].val
    TanInput['MaxWater']=1
    TanInput['MaxGas']=SliderDict['MaxGas'].val
    TanInput['ViscRat']=SliderDict['ViscRat'].val

    outFrame=TanFind(TanInput)
    l.set_ydata(outFrame['Saturation'])
    l.set_xdata(outFrame['Speed'])
    ax.axis([0, np.ceil(max(outFrame['Speed'])), -0.05, 1])
    fig.canvas.draw_idle()

SliderDict['ResGas'].on_changed(update)
SliderDict['ResWater'].on_changed(update)
SliderDict['GasExp'].on_changed(update)
SliderDict['LiqExp'].on_changed(update)
#SliderDict['MaxWater'].on_changed(update)
SliderDict['MaxGas'].on_changed(update)
SliderDict['ViscRat'].on_changed(update)

def reset(event):
    SliderDict['ResGas'].reset()
    SliderDict['ResWater'].reset()
    SliderDict['GasExp'].reset()
    SliderDict['LiqExp'].reset()
    #SliderDict['MaxWater'].reset()
    SliderDict['MaxGas'].reset()
    SliderDict['ViscRat'].reset()
button.on_clicked(reset)

plt.show()

