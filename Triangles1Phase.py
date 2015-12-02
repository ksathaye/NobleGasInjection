import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import ndimage
from scipy.integrate import odeint
import NewtonSat
from scipy import interpolate 
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
import scipy


def GetSpeeds(CrossoverPoints,K,InjC,IniC,TotalSoln):
    props = dict(boxstyle='round', facecolor='w', alpha=1);

    injc=np.zeros(4);# get injection gas composition
    injc=CrossoverPoints[0,0:-1]//(CrossoverPoints[0,-1]+K*(1-CrossoverPoints[0,-1]));
    
    Rare1S=np.linspace(CrossoverPoints[0,-1],CrossoverPoints[2,-1],1000); # draw saturations in rarefaction
    Rare1df=df(Rare1S,0.2); # get speeds of rarefaction
    Rare1df[Rare1S>CrossoverPoints[1,-1]]=0; #saturations above residual not permitted 
    #Rare1df[Rare1S>CrossoverPoints[1,-1]]=0;
    Shock1S=np.zeros(2);# set up shock 1 vector
    Shock1S[0]=Rare1S[-1]; # shock 1 start at end of rarefaction
    
    TriangleshockSpeed=np.min(Rare1df[np.nonzero(Rare1df)]);
    TriangleshockIndex=np.argmin(np.abs(Rare1df-TriangleshockSpeed));
    
    Shock1S[-1]=CrossoverPoints[3,-1]; # shocks down to 2nd crossover
    Shock1Speed=np.zeros(2); # set up vector for shock 1 speed
    Shock1Speed=Rare1df[-1]+Shock1Speed; # set shocks 1 speed

    EndShockSpeed=(Fg(CrossoverPoints[4,-1],.2)-Fg(CrossoverPoints[-1,-1],.2))/(CrossoverPoints[4,-1]-CrossoverPoints[-1,-1]); # speed for final shock to initial concentrations
    c11Shock2Speed=CrossoverPoints[3,0]/(CrossoverPoints[3,4]+K[0]*(1-CrossoverPoints[3,4]));# find gas composition before final shock 
    fShock2=Fg(CrossoverPoints[3,-1],.2); # fractional flow before 2nd shock
    FShock2=c11Shock2Speed*fShock2+(1-fShock2)*K[0]*c11Shock2Speed;# fractional flow C1 before 2nd shock
    Shock2Speed=FShock2/CrossoverPoints[3,0]; # Speed of 2nd shock
    
    IndexCross=np.zeros(6); #initialize vector of crossover indices
    IndexCross[0]=0; # Location of injection composition in TotalSoln
    IndexCross[1]=1;  # Location of S=1 composition in TotalSoln
    IndexCross[2]=2;# Location of drop to residual saturation
    A=TotalSoln==CrossoverPoints[3,:];  # find location of 1st shock 
    AS=np.sum(A,1);
    IndexCross[3]=np.argmax(AS); #find by minimizing difference
    A=TotalSoln==CrossoverPoints[4,:]; # find location of 2nd shock 
    AS=np.sum(A,1);
    IndexCross[4]=np.argmax(AS);#find by minimizing difference
    IndexCross[5]=len(TotalSoln); #location of final composition
        
    Speed=np.zeros(len(Rare1S)+6); # initialize vector of all speeds
    Speed[0:len(Rare1df)]=Rare1df; # speed during rarefaction
    Speed[len(Rare1df)]=Shock1Speed[0]; # speed of 1st shock
    Speed[len(Rare1df)+1]=Shock2Speed;# speed of 2nd shock
    Speed[len(Rare1df)+2]=Shock2Speed;# speed of 2nd shock
    Speed[len(Rare1df)+3]=EndShockSpeed;# speed of end shock.5
    Speed[len(Rare1df)+4]=EndShockSpeed;# speed of end shock
    Speed[len(Rare1df)+5]=np.ceil(EndShockSpeed);# initial composition
    
    CSpeed=np.zeros([len(Speed),4]); #initialize vector of total compositions C_i
    cspeed=np.zeros([len(Speed),4])+.1; #initialize vector of gas phase compositions c_i
    SSpeed=np.zeros([len(Speed)]); #initialize vector of saturations

    SSpeed[0:len(Rare1S)]=Rare1S;# saturation during rarefaction
    SSpeed[len(Rare1df)+0]=CrossoverPoints[3,-1]; # shock 1 end 
    SSpeed[len(Rare1df)+1]=CrossoverPoints[3,-1];# shock 1 end 
    SSpeed[len(Rare1df)+2]=CrossoverPoints[4,-1]; # jump to shock 2 saturation
    SSpeed[len(Rare1df)+3]=CrossoverPoints[4,-1];# jump to shock 2 saturation
    SSpeed[len(Rare1df)+4]=CrossoverPoints[-1,-1]; #end at initial composition saturation
    SSpeed[len(Rare1df)+5]=CrossoverPoints[-1,-1];#end at initial composition saturation    

    for i in range(4): # fill CSpeed with total compositions for each component
        CSpeed[0:len(Rare1S),i]=np.linspace(CrossoverPoints[0,i],CrossoverPoints[2,i],len(Rare1S)); # fill rarefaction compositions
        CSpeed[len(Rare1df)+0,i]=CrossoverPoints[3,i]; # shock 1 points
        CSpeed[len(Rare1df)+1,i]=CrossoverPoints[3,i]; # shock 1 points
        CSpeed[len(Rare1df)+2,i]=CrossoverPoints[4,i];# shock 2 points
        CSpeed[len(Rare1df)+3,i]=CrossoverPoints[4,i];# shock 2 points
        CSpeed[len(Rare1df)+4,i]=CrossoverPoints[-1,i]; # end composition
        CSpeed[len(Rare1df)+5,i]=CrossoverPoints[-1,i];# end composition
        cspeed[:,i]=CSpeed[:,i]/(SSpeed+K[i]*(1-SSpeed)); # compute gas phase compositions
    cspeed[0,3]=1-cspeed[0,1]-cspeed[0,2]-cspeed[0,0];
    plt.close('all') # start plotting stuff
    
    SSpeed[TriangleshockIndex-1]=0.8;
    SSpeed[TriangleshockIndex]=0.8;
    SSpeed[0:TriangleshockIndex-1]=0.8;
    Speed[0]=-.1;
    Speed[1]=-0;
    SSpeed[0]=1#CrossoverPoints[0,-1];
    SSpeed[1]=1#CrossoverPoints[0,-1];

    plt.subplot(3,1,1) # subplot for gas saturation
    plt.xlim([-0.1,3]) # set xlim below zero to start point
    plt.ylim([-0.1,1.1]) # set ylim below zero to show end point
    plt.xticks([0,.5,1,1.5,2,2.5,3],['','','','','','']) # remove ticklabels
    #plt.legend(loc=1)
    plt.scatter(0,1,c='g',s=100,zorder=2) # initial composition point marker
    plt.scatter(TriangleshockSpeed,CrossoverPoints[1,-1],s=100,marker='^',color='green',edgecolor='k',zorder=2) #start of rarefaction point
    plt.scatter(Rare1df[-1],Rare1S[-1],s=100,marker='s',color='green',edgecolor='k',zorder=2) #start of rarefaction point
    plt.text(2.8,.8,'$A$',fontsize=18,color='k',rotation=0,bbox=props)
    plt.scatter(.5*(Speed[len(Rare1df)+0]+Speed[len(Rare1df)+2]),SSpeed[len(Rare1df)+0],marker='v',s=100,color='green',edgecolor='k',zorder=2) #marker for shock 1
    plt.scatter(.5*(Speed[len(Rare1df)+3]+Speed[len(Rare1df)+2]),SSpeed[len(Rare1df)+3],marker='p',s=150,color='green',edgecolor='k',zorder=2)#marker for shock 2
    plt.scatter(.5*(Speed[len(Rare1df)+4]+3),SSpeed[len(Rare1df)+4],marker='D',s=100,color='green',edgecolor='k',zorder=2)#marker for initial comp
    plt.title('Gas Saturation') # title for plot
    plt.text(Rare1df[-1]-.05,Rare1S[-1]+.35,'$\mathcal{W}_2$',fontsize=15)
    plt.text(Rare1df[-1]-.09,Rare1S[-1]+.15,'$\leftarrow$',fontsize=25,rotation=90)
    plt.text(Speed[len(Rare1df)+2]-.05,.45,'$\mathcal{W}_3$',fontsize=15)
    plt.text(Speed[len(Rare1df)+2]-.09,SSpeed[len(Rare1df)+3]+.1,'$\leftarrow$',fontsize=25,rotation=90)
    plt.text(Speed[len(Rare1df)+3]-.05,.45,'$\mathcal{W}_4$',fontsize=15)
    plt.text(Speed[len(Rare1df)+3]-.09,SSpeed[len(Rare1df)+3]+.1,'$\leftarrow$',fontsize=25,rotation=90)
    plt.plot(Speed,SSpeed,c='k',lw=2,label='Gas Saturation', zorder=1) # plot saturation
    plt.plot([0,Speed[len(Rare1df)]],[0,0],c='k',ls='--')
    plt.text(.5*Speed[len(Rare1df)]-.1,0.08,'$\mathcal{W}_1$',fontsize=15)

    FillPlot=1;
    if FillPlot==0:
        plt.subplot(3,1,2)
        plt.plot(Speed,cspeed[:,0],c='red',lw=2,label='c$_g$')
        plt.plot(Speed,cspeed[:,1],c='g',lw=1,label='c$_G$')
        plt.xlim([-0.1,3])
        plt.ylim([-0.,1])
        plt.xticks([0,.5,1,1.5,2,2.5,3],['','','','','',''])
        #plt.legend(loc=1)
        
        plt.subplot(3,1,3)
        plt.plot(Speed,cspeed[:,2],c='r',lw=1,label='c$_b$')
        plt.plot(Speed,cspeed[:,3],c='g',ls='-',lw=2,label='c$_B$')
        plt.xlim([-0.1,3])
        plt.ylim([-0.,1])
        plt.legend(loc=2)
        
        plt.subplot(3,1,2)#subplot for total composition C_i
        plt.stackplot(Speed,CSpeed[:,0],CSpeed[:,1],CSpeed[:,2],CSpeed[:,3],colors=[[1,.7,.7],[1,.2,.2],[.7,.7,1],[.2,.2,1]]) # stackplot for all 4 components
        plt.ylim([0,1]) # ylimits between 0 and 1 volume fraction
        plt.xlim([-0.1,3]) # x limits to match saturation
        plt.text(.45,.3,'$G$',fontsize=16,color='w') # label major gas component
        plt.text(1.3,.07,'$g$',fontsize=16,color='k') #label trace gas component
        plt.text(2.3,.07,'$l$',fontsize=16,color='k')#label trace liquid component
        plt.text(1.3,.6,'$L$',fontsize=16,color='w') #label major liquid component
        plt.title('Total Composition')
        plt.xticks([0,.5,1,1.5,2,2.5,3],['','','','','',''])
    else:
        plt.subplot(3,1,2) # set up plot for gas phase composition
        cspeed[Speed>EndShockSpeed]=np.nan;
        plt.stackplot(Speed,cspeed[:,0],cspeed[:,1],cspeed[:,2],cspeed[:,3],colors=[[1,.7,.7],[1,.2,.2],[.7,.7,1],[.2,.2,1]]) # stack plot of all gas phase comp
        plt.text(.45,.4,'$G$',fontsize=16,color='w')# label major gas component
        plt.text(1.4,.4,'$g$',fontsize=16,color='k')#label trace gas component
        plt.text(2.1,.4,'$l$',fontsize=16,color='k')#label trace liquid component
        plt.title('Gas Composition') # title for gas phase composition
        plt.ylim([0,1]) # ylim for volume fraction
        plt.xlim([-0.1,3]) # same xlim as above
        plt.text(Speed[-2]-.02,.87,'$\leftarrow L$',fontsize=16,color='b')
        plt.text(2.8,.15,'$B$',fontsize=18,color='k',rotation=0,bbox=props)
        plt.xticks([0,.5,1,1.5,2,2.5,3],['','','','','',''])

        plt.subplot(3,1,3)
        Speed[-1]=3;
        Speed[-2]=3;
        Speed[-3]=3;
        print(Speed)
        plt.stackplot(Speed,cspeed[:,0]*K[0],cspeed[:,1]*K[1],cspeed[:,2]*K[2],cspeed[:,3]*K[3],colors=[[1,.7,.7],[1,.2,.2],[.7,.7,1],[.2,.2,1]]) # stack plot of all gas phase comp
        #plt.stackplot([2.6,3],[cspeed[-1,0]*K[0],cspeed[-1,0]*K[0]],[cspeed[-1,1]*K[1],cspeed[-1,1]*K[1]],[cspeed[-1,2]*K[2],cspeed[-1,2]*K[2]],[cspeed[-1,3]*K[3],cspeed[-1,3]*K[3]],colors=[[1,.7,.7],[1,.2,.2],[.7,.7,1],[.2,.2,1]]) # stack plot of all gas phase comp
        plt.xlabel('Speed') #xlabel for all plots
        plt.ylim([0,1]) # ylim for volume fraction
        plt.xlim([-0.1,3]) # same xlim as above
        plt.text(2.8,.75,'$C$',fontsize=18,color='k',rotation=0,bbox=props)
        plt.text(.45,.2,'$G$',color='w',fontsize=16)
        plt.text(1.4,.12,'$g$',color=[1,.7,.7],fontsize=16)
        plt.text(2.2,.03,'$l$',color='k',fontsize=16)
        plt.text(1.4,.75,'$L$',fontsize=18,color='w')
        plt.title('Liquid Composition')

    plt.savefig('SpeedProf.pdf') # save to PDF plot
    
def TernCoords3D(xi): # function to convert cartesian to tetrahedral coordinates

    fA=xi[:,0]; # get first column
    fB=xi[:,1]; # get 2nd column
    fD=xi[:,3]; # get C4 column
    
    theta = np.deg2rad(30); #rotation angle
    t = 0.5*np.tan(theta);  # 2nd rotation angle
    S = np.sqrt(t**2 + 0.5**2); # rotation distance
    vS = fD * S;     
    vt = vS * np.sin(theta); 
    va = vS * np.cos(theta);
    IncY = vt;
    IncX = va;
    y = fA*np.sin(np.deg2rad(60));# set up rotated y coordinates
    x = fB + y* 1/np.tan(np.deg2rad(60)); #rotated x coordinates
    y = y + IncY; #rotated angles
    x = x + IncX;
    z = fD*(np.sin(np.deg2rad(60)))**2;

    TernOut=xi*0; # assign rotated output
    TernOut[:,0]=x;
    TernOut[:,1]=y;
    TernOut[:,2]=z;
    
    return TernOut;

    
def Pyramid3D(CrossPoints,PhaseBound): #plot 3D tetrahedron of composition
    
    BaseCoords=np.zeros([4,4]);  #set up array of vertex coordinates
    BaseCoords[:,0]=[0,1,0,0];
    BaseCoords[:,1]=[0,0,1,0];
    BaseCoords[:,2]=[0,0,0,1];
    BaseCoords[:,3]=[1,0,0,0];
    
    BaseCoordsTrans=TernCoords3D(BaseCoords); #transform bases to tetrahedral coordinates
    PhaseBound0=TernCoords3D(PhaseBound[0]); # transform 1 phase boundaries to tetrahedral coordinatess
    PhaseBound1=TernCoords3D(PhaseBound[1]);

    PB01=np.zeros([3,3]);  #phase boundary 0 saturation composed of 2 triangles in 3-space
    PB01[1:,:]=PhaseBound0[1:,0:3]; # set up 1st triangle
    PB01[0,:]=BaseCoordsTrans[0,0:3];

    PB02=np.zeros([3,3]);#phase boundary 0 saturation composed of 2 triangles in 3-space
    PB02[0:2,:]=PhaseBound0[0:2,0:3];#2nd triangle
    PB02[2,:]=BaseCoordsTrans[0,0:3];
    
    PB11=np.zeros([4,3]);#phase boundary 0 saturation composed of 2 quadrilateral in 3-space
    PB11[0:2,:]=BaseCoordsTrans[1:3,0:3];
    PB11[2:,:]=PhaseBound1[0:2,0:3];
    
    PB12=np.zeros([4,3]);
    PB12[0,:]=BaseCoordsTrans[1,0:3];
    PB12[3,:]=BaseCoordsTrans[3,0:3];
    PB12[1,:]=PhaseBound1[1,0:3];
    PB12[2,:]=PhaseBound1[2,0:3];
    
    CP=TernCoords3D(CrossPoints); # get 3D transform of crossover points
    
    plt.close('all') # begin plotting
    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d'); #create 3D figure
    #ax.scatter(BaseCoordsTrans[2:,0],BaseCoordsTrans[2:,1],BaseCoordsTrans[2:,2],c='k'); 
    ax.plot(CP[:,0],CP[:,1],CP[:,2],c='g'); # plot solution in 3D space
    ax.scatter(CP[0,0],CP[0,1],CP[0,2],c='g',edgecolor='g',marker='o'); # scatter crossover points
    ax.scatter(CP[1,0],CP[1,1],CP[1,2],c='g',edgecolor='g',marker='^'); 
    ax.scatter(CP[2,0],CP[2,1],CP[2,2],c='g',edgecolor='g',marker='s'); 
    ax.scatter(CP[3,0],CP[3,1],CP[3,2],c='g',edgecolor='g',marker='v'); 
    ax.scatter(CP[4,0],CP[4,1],CP[4,2],c='g',edgecolor='g',marker='p'); 
    ax.scatter(CP[5,0],CP[5,1],CP[5,2],c='g',edgecolor='g',marker='D'); 

    base1=np.zeros([4,2]);# plot base triangles edges
    base1[0:3,0]=BaseCoordsTrans[1:,0];
    base1[0:3,1]=BaseCoordsTrans[1:,1];
    base1[-1,0]=BaseCoordsTrans[1,0];
    base1[-1,1]=BaseCoordsTrans[1,1];
    
    base2=np.zeros([4,3]); # plot triangle face edge 2
    base2[0:3,0]=BaseCoordsTrans[0:-1,0];
    base2[0:3,1]=BaseCoordsTrans[0:-1,1];
    base2[0:3,2]=BaseCoordsTrans[0:-1,2];
    base2[-1,0]=BaseCoordsTrans[0,0];
    base2[-1,1]=BaseCoordsTrans[0,1];
    base2[-1,2]=BaseCoordsTrans[0,2];
    
    base3=np.zeros([4,3]); # plot triangle face edge 3
    base3[0:2,0]=BaseCoordsTrans[0:2,0];
    base3[0:2,1]=BaseCoordsTrans[0:2,1];
    base3[0:2,2]=BaseCoordsTrans[0:2,2];
    base3[2,0]=BaseCoordsTrans[-1,0];
    base3[2,1]=BaseCoordsTrans[-1,1];
    base3[2,2]=BaseCoordsTrans[-1,2];
    
    base3[-1,0]=BaseCoordsTrans[0,0];
    base3[-1,1]=BaseCoordsTrans[0,1];
    base3[-1,2]=BaseCoordsTrans[0,2];
    
    plt.plot(base1[:,0],base1[:,1],c='k',ls='--',lw=0.5) # plot bottom triangle edge

    vtx = PB01;
    tri = a3.art3d.Poly3DCollection([vtx])
    tri.set_color('blue')
    tri.set_edgecolor('')
    tri.set_alpha(0.2)
    ax.add_collection3d(tri) #set up fill for liquids phase regions
    
    vtx = PB02;
    tri = a3.art3d.Poly3DCollection([vtx])
    tri.set_color('blue')
    tri.set_edgecolor('')
    tri.set_alpha(0.2)
    ax.add_collection3d(tri)
    
    vtx = PB11; # fill for pure gas region
    tri = a3.art3d.Poly3DCollection([vtx])
    tri.set_color('red')
    tri.set_edgecolor('')
    tri.set_alpha(0.2)
    ax.add_collection3d(tri)

    vtx = PB12;
    tri = a3.art3d.Poly3DCollection([vtx])
    tri.set_color('red')
    tri.set_edgecolor('')
    tri.set_alpha(0.2)
    ax.add_collection3d(tri)

    ax.plot(base2[:,0],base2[:,1],base2[:,2],c='k',lw=0.5) #plot edge face triangles
    ax.plot(base3[:,0],base3[:,1],base3[:,2],c='k',lw=0.5)
    ax.view_init(elev=-45,azim=90)# set up camera angle
    ax.invert_zaxis() # invert axes to match diamond plot
    ax.invert_xaxis()
    plt.axis('off')
    ax.text(BaseCoordsTrans[0,0],BaseCoordsTrans[1,0]+.05,BaseCoordsTrans[2,0],'$L$',fontsize=18) # label vertices with components
    ax.text(BaseCoordsTrans[0,1]+.20,BaseCoordsTrans[1,1]+.03,BaseCoordsTrans[2,1],'$g$',fontsize=18)
    ax.text(BaseCoordsTrans[0,2]+.25,BaseCoordsTrans[1,2],BaseCoordsTrans[2,2],'$G$',fontsize=18)
    ax.text(BaseCoordsTrans[0,3]-.05,BaseCoordsTrans[1,3],BaseCoordsTrans[2,3],'$l$',fontsize=18)

    plt.savefig('3DTest.pdf',format='pdf') #save plot to PDF

def FgPlots(Injc,Inic,K,CrossPoints): # plot fractional flow curves and switch points
    
    InjTLSLope=Injc[0]*(K[0]-1)/(Injc[1]*(K[1]-1)); #injection tie line slope
    InjTLB=CrossPoints[0,0]-InjTLSLope*CrossPoints[0,1]; # injection tie line intercepts
    InjTLB=1-InjTLB;
    
    K4=K[-1]; # get K value of water
    c41=Injc[-1]; # get main liquid component in gas phase fraction
    C=np.linspace(0,InjTLB,100000); # set up vector of C4 values
    S=(C-c41*K4)/(c41-c41*K4); # determine gas saturation from K value
    fg=Fg(S,0.2); # get fractional flow from Saturation
    F=c41*fg+(1-fg)*c41*K4; # get Fractional flow of Component 4
    F[S<0]=C[S<0]; # linear flux term in 1 phase regions
    F[S>1]=C[S>1];


    CrossoverTLc41=CrossPoints[3,3]/(CrossPoints[3,-1]+K4*(1-CrossPoints[3,-1]));
    CrossoverTLS=(C-CrossoverTLc41*K4)/(CrossoverTLc41-CrossoverTLc41*K4); 
    Crossoverfg=Fg(CrossoverTLS,0.2);
    Crossoverfg[CrossoverTLS>0.8]=1;
    Crossoverfg[CrossoverTLS<0]=0;
    CrossoverTLF=CrossoverTLc41*Crossoverfg+(1-Crossoverfg)*CrossoverTLc41*K4; 

    CrossoverTLF[CrossoverTLS<=0]=C[CrossoverTLS<=0]; # linear flux term in 1 phase regions
    CrossoverTLF[CrossoverTLS>=1]=C[CrossoverTLS>=1];

    FInterp=interpolate.interp1d(C,F); # set up 1D interploation of flux 
    CPY1=FInterp(CrossPoints[0:3,3]); # get flux at cross points

    plt.close('all') # set up plotting
    plt.subplot(2,2,1)  # set up plot number 1
    plt.plot(C,F,c='red',label='Injection \nTie Line')  # plot fractional flow curve with injection tie line
    plt.scatter(CrossPoints[0,3],CPY1[0],color='red',s=100) # plot crossover points on tie line 1
    plt.scatter(CrossPoints[1,3],CPY1[1],marker='^',color='red',s=100)
    plt.plot([CrossPoints[0,3],CrossPoints[1,3]],[CPY1[0],CPY1[1]],c='k')
    plt.scatter(CrossPoints[2,3],CPY1[2],marker='s',color='red',s=100)
    plt.plot([CrossPoints[2,3],InjTLB ],[CPY1[2],InjTLB],color='k',ls='-',lw=1,marker='',markersize=2,markerfacecolor='red')
    plt.legend(loc=4,fontsize=13)
    plt.ylim([-0.0,1])
    #plt.axis('equal')
    plt.xlim([-0.05,1.1])
    plt.ylim([-0.05,1.1])
    plt.yticks([0,.2,.4,.6,.8,1],fontsize=14)
    plt.xticks([0,.25,.5,.75,1],['0','0.25','0.50','0.75','1'],fontsize=14)
    plt.xlabel('$C_L$',fontsize=16)
    plt.ylabel('$F_L$  ',rotation=0,fontsize=16)
    plt.text(CrossPoints[1,3]-.15,.8,'$\mathcal{W_1}$',color='red',fontsize=16)
    plt.text(0,.8,'$\leftarrow$',color='red',fontsize=16,ha='left')
    plt.text(CrossPoints[2,3],.805,'$\leftarrow$',rotation=180,color='red',ha='right',fontsize=16)
    props = dict(boxstyle='round', facecolor='w', alpha=1)
    plt.text(0.5,.9,'A',fontsize=20,bbox=props)
    
    #plt.plot([CrossPoints[1,3],CrossPoints[1,3]],[-.05,1.1],ls='--',c='red')
    plt.plot([CrossPoints[2,3],CrossPoints[2,3]],[-.05,1.1],ls='--',c='red')
    plt.plot([0,0],[-.05,1.1],ls='--',c='red')

    plt.text(.65,.4,'$\mathcal{W_2}$',color='k',fontsize=16)
    
    plt.subplot(2,2,2) # second plot along the initial tie line
    c41=Inic[-1]/K4; # get initial c4 composition
    S2=(C-c41*K4)/(c41-c41*K4); # plot gas saturation along initial tie line
    fg2=Fg(S2,0.2); # plot fractional flow along initial TL
    F2=c41*fg2+(1-fg2)*c41*K4; # flux for initial TL
    F2[S2<0]=C[S2<0]; # linear in 1 phase region
    F2[S2>1]=C[S2>1];
    
    F2Interp=interpolate.interp1d(C,F2); # 1D interp of flux function
    CPY=F2Interp(CrossPoints[4:,3]); # flux at cross points
    
    CrossoverTLInterp=interpolate.interp1d(C,CrossoverTLF);
    CPYCrossover=CrossoverTLInterp(CrossPoints[3,3]);
    
    plt.plot(C,F2,label='Initial \nTie Line') # flux function intial TL plot
    #plt.plot(C,F,c='red')
    plt.plot(C,CrossoverTLF,color='green',label='Crossover \nTie Line')
    plt.scatter(CrossPoints[3,3],[CPYCrossover],color='green',marker='v',s=100)
    plt.text(CrossPoints[3,3]-.02,CPYCrossover-.05,'$\leftarrow \mathcal{W_3}$',color='k',ha='left',fontsize=16)
    plt.text(CrossPoints[-1,3]-.1,CPY[-1]-.03,'$\mathcal{W_4}$',color='k',ha='right',fontsize=16)
    plt.text(CrossPoints[-1,3],CPY[-1]-.1,'$\leftarrow$',color='k',ha='right',rotation=180,weight='bold',fontsize=20)

    plt.plot([CrossPoints[3,3],CrossPoints[4,3]],[CPYCrossover,CPY[0]],color='k',ls='-',lw=1,marker='',markersize=2,markerfacecolor='red')
    plt.plot([CrossPoints[-1,3],CrossPoints[4,3]],[CPY[-1],CPY[0]],color='k',ls='-',lw=1,marker='',markersize=2,markerfacecolor='red')

    plt.legend(loc=2,fontsize=13)
    plt.scatter(CrossPoints[4,3],CPY[0],marker='p',color='b',s=100) # plot cross over points on initial TL
    plt.scatter(CrossPoints[-1,3],CPY[-1],marker='D',color='b',s=100)
    plt.xlim([-0,1])
    plt.ylim([-0.05,1.1])
    plt.yticks([0,.2,.4,.6,.8,1],['','','','',''])
    plt.xticks([0,.25,.5,.75,1],['0','0.25','0.50','0.75','1'],fontsize=14)

    plt.xlabel('$C_L$',fontsize=16)
    plt.ylabel('$F_L$  ',rotation=0,fontsize=16)
    plt.text(0.85,0.1,'B',fontsize=20,bbox=props)

    print('end speed 2')
    print((CPY[-1]-CPY[0])/(CrossPoints[-1,3]-CrossPoints[4,3]))
    
    if 0==1:
        plt.subplot(3,3,3) # plot fractional flow as a function of gas saturation
        SG=np.linspace(0,1.2,100000); # set up vector of saturations
        FG2=Fg(SG,0.2); # compute fractional flow curve
        FG2[SG>.8]=1; # flux equals 1 above residual water
        fInterp=interpolate.interp1d(SG,FG2); # interploation of fractional flow function
        fI=fInterp(CrossPoints[:,-1]);
        fI=1-fI;
        CrossPointsL=1-CrossPoints;
        
        plt.scatter(CrossPointsL[0,-1],fI[0],marker='o',color='red')
        plt.scatter(CrossPointsL[1,-1],fI[1],marker='^',color='red')
        plt.scatter(CrossPointsL[2,-1],fI[2],marker='s',color='red')
        plt.scatter(CrossPointsL[3,-1],fI[3],marker='v',color='g')
        plt.scatter(CrossPointsL[4,-1],fI[4],marker='p',color='b')
        plt.scatter(CrossPointsL[5,-1],fI[5],marker='D',color='b')
        plt.plot([CrossPointsL[4,-1],CrossPointsL[5,-1]],[fI[4],fI[5]],c='k',lw=0.5)
        
        
        plt.scatter([CrossPointsL[4,-1],CrossPointsL[5,-1]],[fI[4],fI[5]],color='k',edgecolor='k',s=6)
    
        plt.plot(1-SG,1-FG2,c='k') #plot fractional flow 
        plt.xlim([-0.2,1.05])
        plt.ylim([-0.05,1.05])
    
        plt.yticks([0,.2,.4,.6,.8,1],['','','','',''])
        plt.xticks([-.2,0,.2,.4,.6,.8,1],['-0.2','','0.2','','0.6','','1'])
        plt.ylabel('$f_{liq}$')
        plt.xlabel('$S_{liq}$')
        plt.text(CrossPointsL[4,-1],fI[3],'$\mathcal{W_4}$',color='k',ha='right')

    plt.savefig('FgTest.pdf',format='pdf') #save plot
    

def TangentFind(Injc,K,Intercept,ResidualW): # finds location of 1st shock using tangent method
    PlotOn=1;
    C=np.linspace(Intercept,1,100000); # set up vector of C values
    K1=K[0];
    K4=K[3];
    c11=Injc[0];
    c41=Injc[-1];

    InjS=(0-K4*c41)/(c41-K4*c41);
    InjC1=InjS*c11+K1*c11*(1-InjS);
    
    S=(C-c11*K1)/(c11-c11*K1); # get saturation from K and C
    
    fg=Fg(S,ResidualW); # fractional flow 
    dfg=df(S,ResidualW); # derivative of fractional flow 
    F=c11*fg+(1-fg)*c11*K1; # flux term for C1s
    dF=dfg; # derivative of C1 flux term
    F[S<0]=C[S<0];# linear flux in one phase region
    F[S>1]=C[S>1];

    dF[1:]=(F[1:]-F[:-1])/(C[1:]-C[:-1]); #numerical derivative
    dF[S<0]=1;
    dF[S>1]=1;
    
    Obj=np.abs((F-Intercept)/(C-Intercept)-dF); #objective function for tangent find
    Obj[S<0]=1;
    Obj[S>1]=1;
    Obj2=np.abs((F-InjC1)/(C-InjC1)-dF); 
    Obj2[S<0]=100;
    Obj2[S>1]=100;

    
    minTan=np.argmin(Obj); # find objective minimum
    minTan2=np.argmin(Obj2);
    if PlotOn==1:
        plt.close('all')
        plt.plot(C,F)
        #plt.plot(C,dF)
        #plt.plot(C,np.abs((F-Intercept)/(C-Intercept)))
        plt.plot([Intercept,C[minTan]],[Intercept,F[minTan]],c='red')
        plt.plot([InjC1,C[minTan2]],[InjC1,F[minTan2]],c='red')
        plt.grid();
        plt.scatter(InjC1,InjC1)
        plt.savefig('TanFind.png',format='png')
        
    
    return [C[minTan],C[minTan2]];


def numIntPath(K,ResidualW,Spec,Face,StartInt): # numerically integrate non-tie line path
    
    Test=dxdSFunc(K,.9,1,Face); # determine direction of eigenvectors
    d1=Test[1]/Test[0]; 
    NumIt=1000;
    
    if Test[1]<0: # negative eigenvector start integrations
        if np.isnan(Spec[0])==1:
            if Face==2: # start for face 2
                StartS=np.array([.6,.75,.85,.9636,StartInt[0]]); 
                Startc11=np.ones(len(StartS));
                Startc11[-1]=StartInt[1];
                dS=(StartS-0.05)/NumIt;
            elif Face==1: # start for face 1
                StartS=np.array([.285,.3,.33,StartInt[0]]);
                Startc11=np.ones(len(StartS));
                Startc11[-1]=StartInt[1];
                dS=(StartS-0.95)/NumIt;
                
        else: 
            StartS=Spec[0];
            Startc11=Spec[1];
            dS=(StartS-0)/NumIt;
            
        if np.isnan(Spec[0])==1:
            SAll=np.zeros([NumIt,len(StartS)]);
            c11All=np.zeros([NumIt,len(StartS)]);
            c11All[0,:]=Startc11;
            SAll[0,:]=StartS;
        else:
            SAll=np.zeros(NumIt);
            c11All=np.zeros(NumIt);
            c11All[0]=Startc11;
            SAll[0]=StartS;
            
        if np.isnan(Spec[0])==1:
            for i in range(1,NumIt): #begin numerical integration
                d=dxdSFunc(K,SAll[i-1],c11All[i-1,:],Face);
                dx=d[0];
                dy=d[1];
                SAll[i,:]=SAll[i-1]-dS;
                c11All[i,:]=c11All[i-1,:]-dy/dx*dS;
            c11All[0,:]=Startc11;
            SAll[0,:]=StartS;

        else:
            for i in range(1,NumIt):
                d=dxdSFunc(K,SAll[i-1],c11All[i-1],Face); 
                dx=d[0];
                dy=d[1];
                SAll[i]=SAll[i-1]-dS;
                c11All[i]=c11All[i-1]+dy/dx*dS;
            c11All[0]=Startc11;
            SAll[0]=StartS;

    return [SAll,c11All]; # return integral path 

def dxdSFunc(K,S,c11,Face): # compute eigenvalue of Non-Tie Line path
    if Face==1: #determines which K values to use dependending on face

        K1=K[0];
        K2=K[2];
        K3=K[3];

    elif Face==2:
        K1=K[0];
        K2=K[1];
        K3=K[3];
            
    ResidualW=0.2;
    gamma=(1-K3)*(K2-1)/((K1-K2)*(K1-K3)); #simplification variable
    derivf=df(S,ResidualW); #compute derivative of fractoinal flow
    f=Fg(S,ResidualW); #compute fractional flow for various saturations
    
    p=(derivf-1)/(f-S); #2nd simplification variable
    Q=(gamma*(K1-1))+gamma*(1+(K1-1)*S)*(1-derivf)/(f-S); #3rd simplification variable
    dxdS=Q-p*c11; #eigenvector slope for non-tie line path
    t=np.arctan(dxdS); #regularize vector lengths to 1
    if Face==2:
        dx=-np.cos(t);
        dy=-np.sin(t);
    elif Face==1:
        dx=-np.cos(t);
        dy=-np.sin(t);
    return [dx,dy]; # return eigenvector x,y

def NTLPaths(K,Face,StartInt): # computes series of NTL paths for plotting purposes
    #computes non-tie line eigenvector paths on face 1 or 2 of tetrahedron
    
    if Face==1: #determines which K values to use dependending on face
        K1=K[0];
        K2=K[2];
        K3=K[3];

    elif Face==2:
        K1=K[0];
        K2=K[1];
        K3=K[3];
        
    ResidualW=0.2; #set residual water saturation
    ResidualG=0;#set residual gas saturation

    vc11=np.linspace(0-.3,1.1,21); #set up c11 for phase behavior
    vS=np.linspace(-0,1,21);# set up look up table for gas saturation
    S,c11=np.meshgrid(vS,vc11); #meshgrid for scope of gas saturation and component volume fraction
    f=Fg(S,ResidualW); #compute fractional flow for various saturations
    derivf=df(S,ResidualW); #compute derivative of fractoinal flow

    gamma=(1-K3)*(K2-1)/((K1-K2)*(K1-K3)); #simplification variable
    p=(derivf-1)/(f-S); #2nd simplification variable
    Q=(gamma*(K1-1))+gamma*(1+(K1-1)*S)*(1-derivf)/(f-S); #3rd simplification variable
    dxdS=Q-p*c11; #eigenvector slope for non-tie line path
    t=np.arctan(dxdS); #regularize vector lengths to 1
    d=dxdSFunc(K,S,c11,Face);
    dx=d[0];
    dy=d[1];
    #dx=-np.cos(t);
    #dy=-np.sin(t);
    
    C1Quiv=S*c11+(1-S)*K1*c11; # C1 values from S, c11
    c2Quiv=(1-K1*S-K3+K3*c11)/(K2-K3);
    C2Quiv=S*c2Quiv+(1-S)*K2*c2Quiv; # corresponding C2 values
    ListPaths=0;
    
    T=numIntPath(K,0.2,np.zeros(2)*np.nan,Face,StartInt); # numerically integrate sample paths
    SPath=T[0];# get S data
    c11Path=T[1]; # get c11 data
    C1Path=c11Path*SPath+(1-SPath)*c11Path*K1;
    c21Path=(1-K1*c11Path-K3+K3*c11Path)/(K2-K3);
    c21Zero=np.argmin(np.abs(c21Path[:,-1]));
    C2Path=c21Path*SPath+(1-SPath)*c21Path*K2;
    C2Path[C2Path+C1Path>1]=np.nan;
    C1Path[C1Path<-1]=np.nan;
    #C2Path[C2Path<0]=np.nan;
    #CPath=np.zeros([len(C1Path),2]);
    #CPath[:,0]=C1Path;
    #CPath[:,1]=C2Path;
    SHPath=np.shape(SPath);
    props = dict(boxstyle='round', facecolor='w', alpha=1)
    SPathPlot=np.zeros(SHPath);
    SPathPlot=SPathPlot+SPath;
    SPathPlot[SPathPlot<.1]=np.nan;
    PlotOn=1;
    if PlotOn==1:
        plt.close('all')
        for i in range(SHPath[1]):
            SP=SPathPlot[:,i];
            c11P=c11Path[:,i];
            c11P[SP<0.1]=np.nan;
            plt.plot(SP,c11P,c='red',lw=2);
        #if Face==2:
            #plt.close('all')
        plt.ylim([0,1])
        if Face==1:
            c1gZero=np.argmin(np.abs(c11Path[:,-1]));

            plt.scatter(StartInt[0],StartInt[1],color='red',s=200,marker='v')
            plt.quiver(S,c11,-dx*1e5,-dy*1e5)
            plt.xlim([-0.05,1])
            plt.ylim([-.05,1])
            plt.xlabel('$S_{gas}$',fontsize=16)
            plt.xticks(np.linspace(0,1,11),fontsize=16);
            plt.yticks(np.linspace(0,1,11),fontsize=16);
            plt.ylabel('$c_{g,gas}$',fontsize=20)
            plt.title('Non-Tie Line Vectors: $g-L-l$ plane',fontsize=20)
            #ax = plt.axes()
            #ax.arrow(SPath[c1gZero,-1], 0, -SPath[c1gZero,-1]/2, 0, head_width=0.03, head_length=0.03, fc='b', ec='k')
            plt.scatter(SPath[c1gZero,-1],0,marker='p',s=200,color='red')
            plt.plot([0,SPath[c1gZero,-1]],[0,0],lw=2,color='b')
            plt.scatter(0,0,marker='D',s=120,color='b')
            plt.text(.23,.4,'$\mathcal{W}_3$',backgroundcolor='w',color='red',fontsize=25)
            plt.text(.23,.4,'$\leftarrow$',backgroundcolor='none',color='red',fontsize=25,ha='right')
            plt.text(.9,.9,'B',fontsize=25,bbox=props)
            plt.text(SPath[c1gZero,-1]/4,0.05,'$\mathcal{W}_4$',backgroundcolor='w',color='b',fontsize=25)
            plt.savefig('Vector1.pdf');
            
        else: 
            plt.scatter(StartInt[0],StartInt[1],c='red',s=150,marker='s',edgecolor='red')
            plt.quiver(S,c11,dx*1e5,dy*1e5)
            plt.ylim([-0.05,1])
            plt.xlim([0,1.2])
            plt.xticks(np.linspace(0,1.2,13));
            plt.xlabel('$S_{gas}$',fontsize=16)
            plt.ylabel('$c_{g,gas}$',fontsize=20)
            plt.xticks(np.linspace(0,1.2,13),fontsize=16);
            plt.yticks(np.linspace(0,1,11),fontsize=16);

            plt.scatter(1.14285714,StartInt[1],s=150,color='b')
            plt.scatter(0.53068197,StartInt[1],s=150,marker='^',color='b')
            plt.plot([StartInt[0],1.14285714],[StartInt[1],StartInt[1]],lw=2,c='b')
            plt.title('Non-Tie Line Vectors: $g-G-L$ plane',fontsize=20)
            plt.scatter(SPath[c21Zero,-1],c11Path[c21Zero,-1],c='red',s=200,marker='v',edgecolor='red')
            plt.text(.2,.2,'$\mathcal{W}_2$',backgroundcolor='w',color='red',fontsize=25,ha='right')
            plt.text(.18,.22,'$\leftarrow$',backgroundcolor='none',color='red',fontsize=25,ha='left',rotation=180)
            plt.text(1.1,.9,'A',fontsize=25,bbox=props)
            plt.text(1,StartInt[1]*1.5,'$\mathcal{W}_1$',backgroundcolor='none',color='b',fontsize=25)

            plt.savefig('Vector2.pdf');

    if 1==0: #set up streamlines through vector field
        #STR=plt.streamplot(S, c11, np.ones(np.shape(dx)), dxdS, color='r');
        STR=plt.streamplot(S, c11, -dx, -dy, color='r',density=1.5);
        plt.close();#close streamline plot

        L=STR.lines.get_segments(); # get streamline data

        LData=np.zeros([len(L),3]); #set up array for streamline data
        for i in range(len(L)):
            LData[i,[0,1]]=L[i][0,:]; #put streamline data into array
            LData[i,2]=i;

        Lim=len(L);
        R=np.zeros(Lim);
        R[0]=1;
        R[1:]=np.sqrt((LData[1:,1]-LData[0:-1,1])**2)+np.sqrt((LData[1:,0]-LData[0:-1,0])**2);#regularize  distance vectors

        R[R==0]=np.nan;
        R[R>0.08]=1;

        count=0;
        for i in range(len(R)):
            LData[i,2]=count;
            if R[i]==1:
                    count=count+1;
        ListPaths=list();
        ListPathsC1C2=list();#initialize list of streamline paths
        #plt.figure();
        a=0;
        for i in range(int(count)): #assign list of streamline paths
            Path=LData[LData[:,2]==i,:];
            Path[Path[:,1]>1]=np.nan;
            Path[Path[:,1]<0]=np.nan;
            if len(Path)>17:

                PathC1C2=np.zeros([len(Path),2]);
                c21Path=(1-K1*Path[:,1]-K3+K3*Path[:,1])/(K2-K3);
                #c21Path=(1-Path[:,1]-K3*(1-Path[:,1]/K1))/(K2-K3);
                ListPaths.append(Path);

                PathC1C2[:,0]=Path[:,1]*(Path[:,0])+(1-Path[:,0])*Path[:,1]*K1;
                PathC1C2[:,1]=c21Path*(Path[:,0])+(1-Path[:,0])*c21Path*K2;
                plt.plot(PathC1C2[:,0],PathC1C2[:,1])
                ListPathsC1C2.append(PathC1C2);

                a=a+1;
                
        #QuivPath=[C1Quiv,C2Quiv,dx,dy];
        plt.show();
    
    return [C1Path,C2Path,SPath,c11Path]; # return all paths

def Fg(SG,ResidualW):
    #sets up fractional flow function as a function of gas saturation and residual water saturation
    ViscG=1e-5;# set viscosities
    ViscW=1e-4;

    KW = RKW(SG,ResidualW,0); # call relative permeabilities
    KG = RKG(SG,ResidualW,0);
    MG=KG/ViscG;
    MW=KW/ViscW;
    F=MG/(MW+MG);
    #if np.shape(F)>1:
        #F[SG>1-ResidualW]=1;
    return F;

def TernCoords(xin,yin): # convert cartesian to 2D ternary coordinates
    xin[xin<0]=np.nan;
    yin[yin<0]=np.nan;
    xin[xin+yin>1]=np.nan;
    y = yin*np.sin(np.deg2rad(60));
    x = xin + y*np.cot(np.deg2rad(60));
    
    return x,y

def df(SG,ResidualW):
    #computes derivative of fractional flow function as a function of gas saturation and residual water
    ResidualG=0;
    ViscG=1e-5;
    ViscW=1e-4;
    KW = RKW(SG,ResidualW,0);
    KG = RKG(SG,ResidualW,0);
    MG=KG/ViscG;
    MW=KW/ViscW;
    F=MG/(MW+MG);

    dKW = RKW(SG,ResidualG,1);
    dKG = RKG(SG,ResidualG,1);
    dMG=dKG/ViscG;
    dMW=dKW/ViscW;

    dF=(dMG*MW-dMW*MG)/((MG+MW)**2);
    return dF;

def RKW(SG,ResidualW,deriv):
    #relative permeability of liquid phase as function of gas saturation and residual water saturation
    #deriv option for derivative of rel perm
    ResidualG=0;

    K=((1-SG-ResidualW)/(1-ResidualG-ResidualW))**2;
    if deriv==1:
        dK=-2*(1-SG-ResidualW)/((1-ResidualG-ResidualW)**2);
        return dK;
    else:
        return K;

def RKG(SG,ResidualW,deriv):
    #relative permeability of gas phase as function of gas saturation and residual water saturation
    #deriv option for derivative of rel perm

    ResidualG=0;
    K=1*((SG-ResidualG)/(1-ResidualG-ResidualW))**2;
    if deriv==1:
        dK=2*1*(SG-ResidualG)/((1-ResidualG-ResidualW)**2);
        return dK;
    else:
        return K;

def Tern():  
    
    #computes solution path and 1 phase regions
    
    PlotTri=0;
    K=np.zeros(4);
    K[0]=.1;#Henrys law helium
    K[2]=.2; #Henrys law Neon
    K[1]=.5; #Henrys Law for CO2/Methane
    K[3]=8; #Henrys Law water
    Face=2;
    if Face==1: #determines which K values to use dependending on face

        K1=K[0];
        K2=K[2];
        K3=K[3];

    elif Face==2:
        K1=K[0];
        K2=K[1];
        K3=K[3];
    
    m=np.zeros(2);
    b=np.zeros(2);
    m[0]=-(K2-K3)/(K1-K3);# sat=0 slope
    b[0]=(1-K3)/(K1-K3);
    
    m[1]=K1/K2*(K3-K2)/(K1-K3);#Sat=1 slope
    b[1]=K1*(1-K3)/(K1-K3);
    
    b0=-b/m; # intercept of saturation lines 
    mP1=-(K2-K3)/(K1-K3); 
    mP2=K1*(K3-K2)/(K2*(K1-K3));
    bP1=(1-K3)/(K1-K3);
    bP2=K1*(1-K3)/(K1-K3);
    
    InjC=np.zeros(4);
    InjC[1]=.85;
    InjC[0]=InjC[1]*mP1+bP1;
    InjC[3]=1-InjC[0]-InjC[1];
    injc=InjC;

    C1TL=np.zeros([21,2]); #tie line 1 coordinates
    C1TL[:,0]=np.linspace(0,1,len(C1TL))*b0[0];
    C1TL[:,1]=np.linspace(0,1,len(C1TL))*b0[1];
    C2TL=np.zeros([len(C1TL),2]);
    C2TL[:,0]=C1TL[:,0]*m[0]+b[0];
    C2TL[:,1]=C1TL[:,1]*m[1]+b[1];
    
    InjTLSLope=injc[0]*(K1-1)/(injc[1]*(K2-1)); #injection tie line slope
    InjTLB=InjC[0]-InjTLSLope*InjC[1]; # injection tie line intercepts
    

    Crossover1=np.zeros(4); # set up array of crossover points

    TL1SRS=TangentFind(injc,K,InjTLB,0.2)

    Crossover1[0]=TL1SRS[0];
    Crossover1[2]=(Crossover1[0]-InjTLB)/InjTLSLope;
    Crossover1[3]=1-Crossover1[0]-Crossover1[2];

    SRS1=np.zeros(5);
    SRS1[0]=TL1SRS[1];
    SRS1[1]=(SRS1[0]-InjTLB)/InjTLSLope;
    SRS1[3]=1-SRS1[0]-SRS1[1];
    SRS1[4]=(SRS1[0]-injc[0]*K1)/(injc[0]-injc[0]*K1);
    
    SCrossover=(Crossover1[0]-injc[0]*K1)/(injc[0]-injc[0]*K1); #saturation at crossover points

    T=NTLPaths(K,2,[SCrossover,injc[0]]); # get Non-tieline paths for plotting
    SHPath=np.shape(T[0]); 
    C1Path=T[0]; #split C1 and C2 
    C2Path=T[1];
    A=np.argmin(C2Path[:,4]);
    C2Path[A:,4]=np.nan;

    C2Path[C2Path<-0]=0;  # remove points outside of domain
    C1Path[C1Path>1]=1;
    
    SPath=T[2]; #get saturation points on path
    c11Path=T[3];
    c21Path=(1-K1*c11Path-K3+K3*c11Path)/(K2-K3); # compute c21 from c11
    
    c11Path[c11Path<0]=1000; # 
    Minc11=np.argmin(c11Path,0);
    c21Path[c11Path==1000]=np.nan;
    c11Path[c11Path==1000]=np.nan; #remove gas fraction values outside of domain
    InjTLSat1=np.zeros(2);  # vector for injection tie line saturations
    InjTLSat1[0]=(InjTLB-bP1)/(mP1-InjTLSLope);
    InjTLSat1[1]=InjTLSat1[0]*InjTLSLope+InjTLB; #
    
    Curve1S=SPath[:,-1]; #
    Curve1c11=c11Path[:,-1];
    
    Curve1c21=(1-K1*Curve1c11-K3+K3*Curve1c11)/(K2-K3);
    
    Curve1C1=Curve1c11*Curve1S+K1*(1-Curve1S)*Curve1c11;
    Curve1C2=Curve1c21*Curve1S+K2*(1-Curve1S)*Curve1c21;
    
    ArgC2=np.argmin(np.abs(Curve1C2));
    Curve1C2[ArgC2+2:]=np.nan;
    Curve1C1[ArgC2+2:]=np.nan;
    
    Crossover2=np.polyfit(Curve1C2[ArgC2:ArgC2+2],Curve1C1[ArgC2:ArgC2+2],1);
    Crossover2[0]=Crossover2[1];
    Crossover2[1]=0;
    
    Curve1C1[ArgC2+1]=Crossover2[0];
    Curve1C2[ArgC2+1]=0;
    Curve1C1[Curve1C1>1]=1;
    
    CrossoverTLc11=1-b[1];
    SCrossover2=((Crossover2[0])-CrossoverTLc11*K1)/(CrossoverTLc11-CrossoverTLc11*K1);
    
    SCrossover2=(Crossover2[0]-b[1])/(b[0]-b[1]);
    
    #print('CrossoverSTest')
    #print(CrossoverSTest)
    
    #T=numIntPath(K,0.2,np.zeros(2)*np.nan);
    #print('Crossover Point S,c11,c21')
    #print(SCrossover,injc[0],injc[2])
    #print('Curve Start S,c11,c21')
    #print(Curve1S[0],Curve1c11[0],Curve1c21[0])
    #print('Crossover Point C1,C2')
    #print(Crossover1[[0,2]])
    #print('Curve Start C1,C2')
    #print(Curve1C1[0],Curve1C2[01])
    #print(InjTLB)
    
    PhaseBound1=np.zeros([3,4]);# intialize array for phase boundaries
    PhaseBound0=np.zeros([3,4]);
    TLOut1=np.zeros([2,7]); # initialize vector for demonstration tie lines
    TLOut2=np.zeros([2,7]);
    TLOut3=np.zeros([2,7]);
    TLOut4=np.zeros([2,7]);
    NTLPathsOutC1=np.zeros([len(C2Path),7]);# initialize vector for demonstration curves
    NTLPathsOutC2=np.zeros([len(C2Path),7]);
    NTLPathsOutC3=np.zeros([len(C2Path),7]);
    NTLPathsOutC4=np.zeros([len(C2Path),7]);    
    
    for i in range(SHPath[1]):  #fill demonstration arrays
        NTLPathsOutC1[:,i]=C1Path[:,i];
        NTLPathsOutC2[:,i]=C2Path[:,i];
        NTLPathsOutC4[:,i]=1-C2Path[:,i];-C1Path[:,i];
        TLOut1[:,i]=[c11Path[Minc11[i],i],K1*c11Path[Minc11[i],i]];
        TLOut2[:,i]=[c21Path[Minc11[i],i],K2*c21Path[Minc11[i],i]];
        TLOut4[:,i]=1-TLOut1[:,i]-TLOut2[:,i];

    PhaseBound1[0,:]=[0,b0[0],0,1-b0[0]];# fill phase boundary arrays
    PhaseBound1[1,:]=[b[0],0,0,1-b[0]];
    
    PhaseBound0[0,:]=[0,b0[1],0,1-b0[1]];
    PhaseBound0[1,:]=[b[1],0,0,1-b[1]];
    
    NTLSet1=i;
    
    InjTLx1=(InjTLB-bP2)/(-InjTLSLope+mP2);
    InjTLy1=InjTLSLope*InjTLx1+InjTLB;
    if 1==PlotTri: 
        plt.close('all')
        plt.plot([0,0,1,0],[0,1,0,0],lw=2,c='k')
        plt.plot([0,b0[0]],[b[0],0],c='b',lw=2)
        plt.scatter(0,b[1],c='b',s=100)
        plt.plot([0,b0[1]],[b[1],0],c='b',lw=2)

        for i in range(SHPath[1]):
            plt.plot(C2Path[:,i],C1Path[:,i],c='k',lw=1); 
            plt.scatter(C2Path[Minc11[i],i],C1Path[Minc11[i],i],c='k',s=50,edgecolor='none');
            plt.plot([c21Path[Minc11[i],i],K2*c21Path[Minc11[i],i]],[c11Path[Minc11[i],i],K1*c11Path[Minc11[i],i]],c='k',ls='--');
        plt.scatter(InjC[1],InjC[0],c='k',s=50)
        plt.scatter(InjTLSat1[0],InjTLSat1[1],c='r',s=100)
        plt.plot([Crossover1[2],InjC[1]],[Crossover1[0],InjC[0]],c='r',lw=2)
        plt.plot(Curve1C2,Curve1C1,c='r',lw=2)
        plt.scatter(Crossover1[2],Crossover1[0],c='r',s=100)
        plt.scatter(Crossover2[1],Crossover2[0],c='r',s=100)

        plt.text(InjC[2],InjC[0],'Injection Gas');
 
    
    C1PathA=C1Path;
    C2PathA=C2Path;
    K1=K[0];
    K2=K[2];
    K3=K[3];
    Face=1;
    #initS=(1-(1-initc[2])*K3+initc[2]*K2)/(initc[2]-initc[2]*K2+(1-initc[2])-(1-initc[0])*K3);
    #InjC=initS*initc+(1-initS)*K*initc;

    T=NTLPaths(K,1,[SCrossover2,CrossoverTLc11]); # face 2 paths
    SHPath=np.shape(T[0]);
    C1Path=T[0];
    C2Path=T[1];
    C2Path[C2Path<0]=np.nan;
    C1Path[C1Path<0]=np.nan;
    SPath=T[2];
    c11Path=T[3];
    c21Path=(1-K1*c11Path-K3+K3*c11Path)/(K2-K3);
    c31Path=(1-K1*c11Path-K2+K2*c11Path)/(-K2+K3);

    Minc11=np.argmin(c11Path,0);
    c11Path[c11Path==1000]=np.nan;
    Curve2S=SPath[:,-1];
    Curve2c11=c11Path[:,-1];

    Curve2c21=(1-K1*Curve2c11-K3+K3*Curve2c11)/(K2-K3);
    Curve2c31=(1-K1*Curve2c11-K2+K2*Curve2c11)/(-K2+K3);

    Curve2C1=Curve2c11*Curve2S+K1*(1-Curve2S)*Curve2c11;
    Curve2C2=Curve2c21*Curve2S+K2*(1-Curve2S)*Curve2c21;
    AM1=np.argmin(np.abs(Curve2C2));
    Curve2C1[Curve2C2<0]=Crossover2[0];
    Curve2C2[Curve2C2<0]=Crossover2[1];
    #Curve2S[Curve2C2<0]=SCrossover2;
    
    AM1=np.argmin(np.abs(Curve2C1));
    if AM1>Curve2C1[0]:
        PCross=np.polyfit([Curve2C1[AM1],Curve2C1[AM1+1]],[Curve2C2[AM1],Curve2C2[AM1+1]],1);
    
    
    Curve2C1[AM1+1:]=0;
    Curve2C2[AM1+1:]=PCross[1];
    Curve2S[AM1+1:]=Curve2S[AM1];
    
    m=np.zeros(2);
    b=np.zeros(2);
    m[0]=-(K2-K3)/(K1-K3);
    b[0]=(1-K3)/(K1-K3);
    
    m[1]=K1/K2*(K3-K2)/(K1-K3);
    b[1]=K1*(1-K3)/(K1-K3);
    
    b0=-b/m;
    mP1=-(K2-K3)/(K1-K3);
    mP2=K1*(K3-K2)/(K2*(K1-K3));
    bP1=(1-K3)/(K1-K3);
    bP2=K1*(1-K3)/(K1-K3);
    
    InitC=np.zeros(4);
    InitC[2]=b0[1];
    InitC[3]=1-b0[1];
    initc=InitC;

    C1TL=np.zeros([11,2]);
    C1TL[:,0]=np.linspace(0,1,len(C1TL))*b0[0];
    C1TL[:,1]=np.linspace(0,1,len(C1TL))*b0[1];
    C2TL=np.zeros([len(C1TL),2]);
    C2TL[:,0]=C1TL[:,0]*m[0]+b[0];
    C2TL[:,1]=C1TL[:,1]*m[1]+b[1];

    InitTLSLope=.9*(1-K2)/(.1*(1-K1));
    InitTLB=-InitTLSLope*.1;
    InitTLx1=(InitTLB-bP1)/(-InitTLSLope+mP1);
    InitTLy1=InitTLSLope*InitTLx1+InitTLB;
    
    for i in range(SHPath[1]-1):
        NTLPathsOutC1[:,i+NTLSet1]=C1Path[:,i];
        NTLPathsOutC3[:,i+NTLSet1]=C2Path[:,i];
        NTLPathsOutC4[:,i+NTLSet1]=1-C2Path[:,i];-C1Path[:,i];
        TLOut1[:,i+NTLSet1]=[c11Path[Minc11[i],i],K1*c11Path[Minc11[i],i]];
        TLOut3[:,i+NTLSet1]=[c21Path[Minc11[i],i],K2*c21Path[Minc11[i],i]];
        TLOut4[0,i+NTLSet1]=1-c11Path[Minc11[i],i]-c21Path[Minc11[i],i];
        TLOut4[1,i+NTLSet1]=1-K2*c21Path[Minc11[i],i]-c21Path[Minc11[i],i];
        
    PhaseBound1[2,:]=[0,0,b0[0],1-b0[0]];
    PhaseBound0[2,:]=[0,0,b0[1],1-b0[1]];

    if 1==PlotTri:
        plt.plot([0,0,-1,0],[0,1,0,0],lw=2,c='k')
        plt.plot([0,-b0[0]],[b[0],0],c='b',lw=2)
        plt.plot([0,-b0[1]],[b[1],0],c='b',lw=2)
        #for i in range(len(C1TL)):
            #plt.plot([C1TL[i,0],C1TL[i,1]],[C2TL[i,0],C2TL[i,1]],lw=2,color=[.5,.5,.5],ls=':');
        for i in range(SHPath[1]):
            plt.plot(-C2Path[:,i],C1Path[:,i],c='k',lw=1);
            plt.scatter(-C2Path[Minc11[i],i],C1Path[Minc11[i],i],c='k',s=50,edgecolor='none');
            if K2*c21Path[Minc11[i],i]<0:
                c21Path[Minc11[i],i]=np.nan;
            if K1*c11Path[Minc11[i],i]>1:
                c21Path[Minc11[i],i]=np.nan;
            plt.plot([-c21Path[Minc11[i],i],-K2*c21Path[Minc11[i],i]],[c11Path[Minc11[i],i],K1*c11Path[Minc11[i],i]],c='k',ls='--');    
            plt.scatter([-c21Path[Minc11[i],i],-K2*c21Path[Minc11[i],i]],[c11Path[Minc11[i],i],K1*c11Path[Minc11[i],i]],c='k');    

        plt.scatter(-InitC[2],0,c='k',s=50)
        plt.plot(-Curve2C2,Curve2C1,c='r',lw=2)
        plt.text(-0.1,0.,'Initial Liq')
        plt.scatter(-PCross[1],0,c='r',s=100)

        plt.xlim([-1,1])
        plt.ylim([0,1])


    Curve1L=np.shape(Curve1C1)[0];
    Curve2L=np.shape(Curve2C1)[0];
    
    TotalSoln=np.zeros([1+Curve1L+Curve2L+1,5]); # aggregate total solution into 4D array
    TotalSoln[0,0:4]=InjC;
    TotalSoln[0,4]=1;
    TotalSoln[1:Curve1L+1,0]=Curve1C1;
    TotalSoln[1:Curve1L+1,1]=Curve1C2;
    TotalSoln[1:Curve1L+1,3]=1-Curve1C2-Curve1C1;
    TotalSoln[1:Curve1L+1,4]=Curve1S;
    
    TotalSoln[Curve1L+1:Curve2L+Curve1L+1,0]=Curve2C1;
    TotalSoln[Curve1L+1:Curve2L+Curve1L+1,2]=Curve2C2;
    TotalSoln[Curve1L+1:Curve2L+Curve1L+1,3]=1-Curve2C2-Curve2C1;
    TotalSoln[Curve1L+1:Curve2L+Curve1L+1,4]=Curve2S;
    TotalSoln[-1,0:4]=InitC;
    TotalSoln[-1,4]=0;
    TotalSoln[1,:]=TotalSoln[0,:];
    TotalSoln[0,:]=0;
    TotalSoln[0,1]=(1-InjTLB)/(InjTLSLope+1);
    TotalSoln[0,0]=1-(1-InjTLB)/(InjTLSLope+1);
        
    CrossPoints=np.zeros([6,5]);
    CrossPoints[0,:]=TotalSoln[0,:];
    #CrossPoints[1,:]=#TotalSoln[1,:];
    #CrossPoints[1,0]=injc[0]*0.8+(1-.8)*K[0]*injc[0];
    #CrossPoints[1,1]=injc[1]*0.8+(1-.8)*K[1]*injc[1];
    #CrossPoints[1,2]=injc[2]*0.8+(1-.8)*K[2]*injc[2];
    #CrossPoints[1,3]=injc[3]*0.8+(1-.8)*K[3]*injc[3];
    #CrossPoints[1,4]=0.8;
    
    CrossPoints[1,:]=SRS1;
    
    CrossPoints[2,:]=TotalSoln[2,:];
    CrossPoints[3,:]=TotalSoln[Curve1L+1,:];
    CrossPoints[4,:]=TotalSoln[Curve2L+Curve1L,:];
    CrossPoints[5,:]=TotalSoln[-1,:];
    CrossPoints[0,4]=(CrossPoints[0,0]-injc[0]*K[0])/(injc[0]-injc[0]*K[0]);
    TotalSoln[0,:]=CrossPoints[0,:];
    print(CrossPoints)
    
    FgPlots(injc,initc,K,CrossPoints); 

    NTLPathsOut=[NTLPathsOutC1,NTLPathsOutC2,NTLPathsOutC3,NTLPathsOutC4];
    TLOut=[TLOut1,TLOut2,TLOut3,TLOut4];
    PhaseBounds=[PhaseBound0,PhaseBound1];
    GetSpeeds(CrossPoints,K,InjC,InitC,TotalSoln);
        
    return TotalSoln,CrossPoints,NTLPathsOut,TLOut,PhaseBounds; # return data for plotting
    
def RotateInjFace(Xin):
    #rotates triangle to horizontal
    
    sint1=np.sin(np.deg2rad(30));
    cost1=np.cos(np.deg2rad(30));
    
    Xout2=np.zeros([len(Xin),2])*np.nan;
    Xout2[:,0]=cost1*Xin[:,1]-sint1*Xin[:,0];
    Xout2[:,1]=sint1*Xin[:,1]+cost1*Xin[:,0];
    
    return Xout2;
    

def RotateDiam(TotalSoln,CrossPoints,NTLPathsOut,TLOut,PhaseBounds):

    #rotates solution data and saves plot
    
    X1=np.array([0,0,-1,0]);
    Y1=np.array([0,1,0,0]);
    X2=np.array([0,0,1,0]);
    Y2=np.array([0,1,0,0]);
        
    Y1R = Y1*np.sin(np.deg2rad(60));
    X1R = X1 - Y1R/np.tan(np.deg2rad(60));  
    
    Y2R = Y2*np.sin(np.deg2rad(60));
    X2R = X2 + Y2R/np.tan(np.deg2rad(60)); #plots edges of domain
        
    TotalSolnRot=np.zeros([len(TotalSoln),2])*np.nan; # rotates total solution path
    TotalSolnRot[TotalSoln[:,2]==0,0]=TotalSoln[TotalSoln[:,2]==0,0]*np.sin(np.deg2rad(60));
    TotalSolnRot[TotalSoln[:,2]==0,1]=TotalSoln[TotalSoln[:,2]==0,1] + TotalSolnRot[TotalSoln[:,2]==0,0]/np.tan(np.deg2rad(60));  
    
    TotalSolnRot[TotalSoln[:,2]>0,0]=TotalSoln[TotalSoln[:,2]>0,0]*np.sin(np.deg2rad(60));
    TotalSolnRot[TotalSoln[:,2]>0,1]=TotalSoln[TotalSoln[:,2]>0,2] + TotalSolnRot[TotalSoln[:,2]>0,0]/np.tan(np.deg2rad(60));  

    TotalSolnRot2=RotateInjFace(TotalSolnRot); # Rotates face 2 solution
    TotalSolnRot2[TotalSoln[:,2]>0,0]=-TotalSolnRot2[TotalSoln[:,2]>0,0];
    
    CrossRot=np.zeros([len(CrossPoints),2])*np.nan; # rotates crossover points
    CrossRot[CrossPoints[:,2]==0,0]=CrossPoints[CrossPoints[:,2]==0,0]*np.sin(np.deg2rad(60));
    CrossRot[CrossPoints[:,2]==0,1]=CrossPoints[CrossPoints[:,2]==0,1] + CrossRot[CrossPoints[:,2]==0,0]/np.tan(np.deg2rad(60));  
    CrossRot[CrossPoints[:,2]>0,0]=CrossPoints[CrossPoints[:,2]>0,0]*np.sin(np.deg2rad(60));
    CrossRot[CrossPoints[:,2]>0,1]=CrossPoints[CrossPoints[:,2]>0,2] + CrossRot[CrossPoints[:,2]>0,0]/np.tan(np.deg2rad(60));  
    CrossRot=RotateInjFace(CrossRot); 
    CrossRot[CrossPoints[:,2]>0,0]=-CrossRot[CrossPoints[:,2]>0,0];
    
    PhaseBounds0=PhaseBounds[0]; # rotates phase boundaries
    PhaseBound0Rot=np.zeros([3,2])*np.nan;
    PhaseBound0Rot[PhaseBounds0[:,2]==0,0]=PhaseBounds0[PhaseBounds0[:,2]==0,0]*np.sin(np.deg2rad(60));
    PhaseBound0Rot[PhaseBounds0[:,2]==0,1]=PhaseBounds0[PhaseBounds0[:,2]==0,1]+PhaseBound0Rot[PhaseBounds0[:,2]==0,0]/np.tan(np.deg2rad(60));  

    PhaseBound0Rot[PhaseBounds0[:,1]==0,0]=PhaseBounds0[PhaseBounds0[:,1]==0,0]*np.sin(np.deg2rad(60));
    PhaseBound0Rot[PhaseBounds0[:,1]==0,1]=PhaseBounds0[PhaseBounds0[:,1]==0,1]+PhaseBound0Rot[PhaseBounds0[:,1]==0,0]/np.tan(np.deg2rad(60));  
    PhaseBound0Rot[-1,0]=PhaseBounds0[-1,0]*np.sin(np.deg2rad(60));
    PhaseBound0Rot[-1,1]=(PhaseBounds0[-1,2]+PhaseBound0Rot[-1,0]/np.tan(np.deg2rad(60)));  
    PhaseBound0Rot=RotateInjFace(PhaseBound0Rot);
    PhaseBound0Rot[-1,0]=-PhaseBound0Rot[-1,0];

    PhaseBounds1=PhaseBounds[1];
    PhaseBound1Rot=np.zeros([3,2])*np.nan;
    PhaseBound1Rot[PhaseBounds1[:,2]==0,0]=PhaseBounds1[PhaseBounds1[:,2]==0,0]*np.sin(np.deg2rad(60));
    PhaseBound1Rot[PhaseBounds1[:,2]==0,1]=PhaseBounds1[PhaseBounds1[:,2]==0,1]+PhaseBound1Rot[PhaseBounds1[:,2]==0,0]/np.tan(np.deg2rad(60));  
    PhaseBound1Rot[-1,0]=PhaseBounds1[-1,0]*np.sin(np.deg2rad(60));
    PhaseBound1Rot[-1,1]=(PhaseBounds1[-1,2]+PhaseBound1Rot[-1,0]/np.tan(np.deg2rad(60)));  
    PhaseBound1Rot=RotateInjFace(PhaseBound1Rot);
    PhaseBound1Rot[-1,0]=-PhaseBound1Rot[-1,0];
    
    NLTShape=np.shape(NTLPathsOut[0]);
    A0=NTLPathsOut[0];
    A1=NTLPathsOut[1];
    A2=NTLPathsOut[2];
    A3=NTLPathsOut[3];
    
    TL0=TLOut[0];
    TL1=TLOut[1];
    TL2=TLOut[2];
    TL3=TLOut[3];
    NTLRotAll=list();
    
    plt.close('all')

    for i in range(NLTShape[1]): # rotates and plots demonstration paths
        A=np.zeros([len(NTLPathsOut[0]),4]);
        A[:,0]=A0[:,i];
        A[:,1]=A1[:,i];
        A[:,2]=A2[:,i];
        A[:,3]=A3[:,i];
        
        NTLRot=np.zeros([len(A),2]);
        NTLRot[A[:,2]==0,0]=A[A[:,2]==0,0]*np.sin(np.deg2rad(60));
        NTLRot[A[:,2]==0,1]=A[A[:,2]==0,1] + NTLRot[A[:,2]==0,0]/np.tan(np.deg2rad(60));  
        
        NTLRot[A[:,2]>0,0]=A[A[:,2]>0,0]*np.sin(np.deg2rad(60));
        NTLRot[A[:,2]>0,1]=A[A[:,2]>0,2] + NTLRot[A[:,2]>0,0]/np.tan(np.deg2rad(60));  
        
        NTLRot2=RotateInjFace(NTLRot);
        NTLRot2[A[:,2]>0,0]=-NTLRot2[A[:,2]>0,0];
        
        TL=np.zeros([len(TL3),4]);
        TL[:,0]=TL0[:,i];
        TL[:,1]=TL1[:,i];
        TL[:,2]=TL2[:,i];
        TL[:,3]=TL3[:,i];
        
        TLRot=np.zeros([len(TL),2]);
        TLRot[TL[:,2]==0,0]=TL[TL[:,2]==0,0]*np.sin(np.deg2rad(60));
        TLRot[TL[:,2]==0,1]=TL[TL[:,2]==0,1] + TLRot[TL[:,2]==0,0]/np.tan(np.deg2rad(60));  
        
        TLRot[TL[:,2]>0,0]=TL[TL[:,2]>0,0]*np.sin(np.deg2rad(60));
        TLRot[TL[:,2]>0,1]=TL[TL[:,2]>0,2] + TLRot[TL[:,2]>0,0]/np.tan(np.deg2rad(60));  

        TLRot=RotateInjFace(TLRot);
        TLRot[TL[:,2]>0,0]=-TLRot[TL[:,2]>0,0];
        
        plt.plot(TLRot[:,0],TLRot[:,1],c='k',ls='--');
        plt.plot(NTLRot2[:,0],NTLRot2[:,1],c='k');
        
    plt.plot(-Y1R,-X1R,lw=1,c='k')
    plt.plot(Y2R,X2R,lw=1,c='k') # plot boundaries
    plt.plot(TotalSolnRot2[:,0],TotalSolnRot2[:,1],c='g',lw=2);# plot total particular solution
    
    Fill1=np.zeros([len(PhaseBound0Rot[:,0]),3]); # fill coordinates for 1 phase regions
    Fill1[:,0]=PhaseBound0Rot[:,0];
    Fill1[:,1]=PhaseBound0Rot[:,1];
    Fill1[0:-1,2]=PhaseBound0Rot[0:-1,0]/np.sqrt(3)
    Fill1[-1,2]=-PhaseBound0Rot[-1,0]/np.sqrt(3)
    
    Fill2=np.zeros([len(PhaseBound0Rot[:,0]),3]);
    Fill2[:,0]=PhaseBound1Rot[:,0];
    Fill2[:,1]=PhaseBound1Rot[:,1];
    Fill2[:,2]=[-PhaseBound1Rot[0,0]/np.sqrt(3)+1,1,PhaseBound1Rot[-1,0]/np.sqrt(3)+1];
    plt.fill_between(Fill1[:,0],Fill1[:,1],Fill1[:,2],alpha=.2,edgecolor='none') # plots 1 phase regions
    plt.fill_between(Fill2[:,0],Fill2[:,1],Fill2[:,2],alpha=.2,edgecolor='none',color='red')
    plt.fill_between([PhaseBound1Rot[0,0],np.sqrt(3)/2],[PhaseBound1Rot[0,1],1/2.0],[-PhaseBound1Rot[0,0]/np.sqrt(3)+1,1.0/2],edgecolor='none',alpha=.2,color='red')
    plt.fill_between([PhaseBound1Rot[-1,0],-np.sqrt(3)/2],[PhaseBound1Rot[-1,1],1/2.0],[PhaseBound1Rot[-1,0]/np.sqrt(3)+1,1.0/2],edgecolor='none',alpha=.2,color='red')
    plt.scatter(CrossRot[0,0],CrossRot[0,1],color='g',s=100,marker='o') # scatter crossover points
    plt.scatter(CrossRot[1,0],CrossRot[1,1],color='g',s=100,marker='^')
    plt.scatter(CrossRot[2,0],CrossRot[2,1],color='g',s=100,marker='s')
    plt.scatter(CrossRot[3,0],CrossRot[3,1],color='g',s=100,marker='v')
    plt.scatter(CrossRot[4,0],CrossRot[4,1],color='g',s=100,marker='p')
    plt.scatter(CrossRot[5,0],CrossRot[5,1],color='g',s=100,marker='D')
    plt.text(CrossRot[1,0]-.05,CrossRot[1,1]+.08,'$\mathcal{W}_1$',rotation=37,fontsize=16,color='green')
    plt.text(CrossRot[3,0]+.07,CrossRot[3,1]-.02,'$\mathcal{W}_2$',rotation=12,fontsize=16,color='green')
    plt.text(CrossRot[3,0]-.24,CrossRot[3,1],'$\mathcal{W}_3$',rotation=14,fontsize=16,color='green')

    plt.xlim([-1,1])

    plt.text(0,-0.08,'$L$',fontsize=18) #label vertices
    plt.text(-np.sqrt(3)/2-0.07,.5,'$l$',fontsize=18)
    plt.text(0,1+0.04,'$g$',fontsize=18)
    plt.text(np.sqrt(3)/2+.02,.5,'$G$',fontsize=18)
    plt.text(TotalSolnRot2[0,0],TotalSolnRot2[0,1]+.05,'Injection \nGas',color='g',fontsize=14,ha='center') # label injection and initial
    plt.text(TotalSolnRot2[-1,0],TotalSolnRot2[-1,1]-.17,'Initial \nWater',color='g',fontsize=14,ha='center')
    plt.text(-.1,1-.25/np.sqrt(3),'Pure Gas',color='r',fontsize=14,rotation=30,ha='right')#lable 1 phase regions
    plt.text(.1,.05,'$\leftarrow$Pure Liquid',color='blue',fontsize=16)
    plt.text(TotalSolnRot2[-1,0]-.22,TotalSolnRot2[-1,1],'$\mathcal{W}_4$',fontsize=16,color='g')
    plt.text(TotalSolnRot2[-1,0]-.16,TotalSolnRot2[-1,1]+.015,'$\leftarrow$',fontsize=18,color='g',rotation=225)
    props = dict(boxstyle='round', facecolor='w', alpha=1)


    plt.scatter([0,-np.sqrt(3)/2,0,np.sqrt(3)/2],[0,.5,1,.5],s=50,c='k') # scatter vertex
    plt.axis('equal')
    plt.axis('off')# turn off axes
    plt.savefig('Diamond.pdf',dpi=1000,format='pdf') # save plot

    Pyramid3D(CrossPoints,PhaseBounds); # call function for 3D plot
    
if 1==1:
    K=np.zeros(4);
    K[0]=1/200.0;#.1548*.822; # Henry's Law for Helium
    K[1]=1/30.0; #Henrys law Neon
    K[2]=1/10.0; #Henrys Law for Methane
    K[3]=40.0; #Henrys Law water
        
    K1=1/K[0];
    K2=1/K[1];
    K3=1/K[3];
    K4=1/K[0];
   
TotalSoln,CrossPoints,NTLPathsOut,TLOut,PhaseBounds=Tern(); # call function for computation
RotateDiam(TotalSoln,CrossPoints,NTLPathsOut,TLOut,PhaseBounds); # call function for plotting