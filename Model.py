import numpy as np
import pylab as plt
plt.ion()
plt.close('all')
#constants
TRAIN=0
TESTLOC=1
TESTHIC=2
D=10
a2str=['B','W','G','RS','RM','RL','S','M','L','XL']
a2data=np.array([[0,1,2,2,1,0,2,1,0,np.nan],
    [2,1,0,2,1,0,2,1,0,np.nan],[np.nan,np.nan,np.nan,2,1,0,np.nan,2,1,0]])
data2a=np.zeros((3,D,3))
for i in range(3):
    data2a[i,:,:] = np.int32(a2data==i).T
bla 
feedback=np.array([[1,0,0,0,0,1,0,0,1,np.nan],
    [0,0,1,0,0,1,0,0,1,np.nan],[np.nan,np.nan,np.nan,0,0,1,0,0,0,1]])
w=np.array([1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5])

# functions
def getProb(a,d): 
    p=np.power(a,d)
    p/=np.nansum(p)
    return p
def chooseAction(p):
    action=np.random.multinomial(1,p)
    return action.nonzero()[0][0]

class Model():
    def __init__(self,q0=0.5,u0=0.5,d=1,g=0.7,h=0.5,m=1):
        ''' q0 - prior preference of color over length (0,1)
            u0 - prior preference of rel. over abs. length (0,1)
            d - decision consistency (0,inf)
            g - learning from positive feedback (0,1)
            h - learning from negative feedback (0,1)
            m - attentional focus (0, inf)
        '''
        self.q0=q0; self.u0=u0; self.d=d
        self.g=g; self.h=h; self.m=m
    def exp1run(self):
        T=20
        #initialize
        q=np.zeros(T+1); q[0]=self.q0
        u=np.zeros(T+1); u[0]=self.u0
        a=np.zeros((T+1,D));self.f=[]
        p=np.zeros((T+1,D));dat=np.zeros(T)
        a[0,:]=np.ones(10)/3.0
        a[0,-1]=np.nan
        a[0,:3]*=q[0]
        a[0,3:6]*=(1-q[0])*u[0]
        a[0,6:]*=(1-q[0])*(1-u[0])
        b=np.zeros(T)# observed behavior
        phase=0
        #print a[0,:]
        for t in range(T):
            if t>10: phase=1
            else: phase=0
            p[t,:]=getProb(a[t,:],self.d)
            b[t]=chooseAction(p[t,:])
            dat[t]=a2data[phase,b[t]]
            m=data2a[dat[t],:,phase]
            f=feedback[phase,b[t]]
            w=np.power(a[t,:],self.m)
            self.f.append(f)
            if f==1:
                s=m*w
                a[t+1,:]= self.g*s/np.nansum(s) + (1-self.g)*a[t,:]
            else:
                s=(1-m)*w
                a[t+1,:]= self.h*s/np.nansum(s) + (1-self.h)*a[t,:]
            u[t+1]= np.nansum(a[t+1,3:6])/np.nansum(a[t+1,3:])
            q[t+1]= np.nansum(a[t+1,:3])/np.nansum(a[t+1,:])
            #(np.nansum(a[t+1,:3])+(1-u[t+1])*np.nansum(a[t+1,6:])+u[t+1]*np.nansum(a[t+1,3:6])
        self.a=a
        self.b=b
        self.dat=dat
        self.f=np.array(self.f)
        return self.dat,self.f

    
    def exp1computeLL(self,dat,f):
        T=20
        #initialize
        q=np.zeros(T+1); q[0]=self.q0
        u=np.zeros(T+1); u[0]=self.u0
        a=np.zeros((T+1,D));self.f=[]
        p=np.zeros((T+1,D));
        a[0,:]=np.ones(10)/3.0
        a[0,-1]=np.nan
        a[0,:3]*=q[0]
        a[0,3:6]*=(1-q[0])*u[0]
        a[0,6:]*=(1-q[0])*(1-u[0])
        phase=0
        LL=0
        #print a[0,:]
        for t in range(T):
            if t>10: phase=1
            else: phase=0
            p[t,:]=getProb(a[t,:],self.d)
            m=data2a[dat[t],:,phase]
            w=np.power(a[t,:],self.m)
            loglik= np.nansum(np.log(np.maximum(0.001,p[t,m==f[t]])))
            if f[t]==1:
                s=m*w
                a[t+1,:]= self.g*s/np.nansum(s) + (1-self.g)*a[t,:]
            else:
                s=(1-m)*w
                a[t+1,:]= self.h*s/np.nansum(s) + (1-self.h)*a[t,:]
            #print t,dat[t],f[t],np.nansum(p[t,m==f[t]]),loglik
            #print 'm= ',m
            #print 'p= ',p
            LL+=loglik
        return LL
        
    def plothistory(self):
        a=self.a
        b=self.b
        plt.figure(figsize=(12,6))
        I=np.concatenate([a.T,np.array(np.nansum(a[:,:3],1),ndmin=2),
            np.array(np.nansum(a[:,3:6],1),ndmin=2),np.array(np.nansum(a[:,6:],1),ndmin=2)],axis=0)
        plt.plot(range(b.size),b,'rx',ms=8,mew=2)
        plt.plot([10.5,10.5],[-1,I.shape[1]],'r',lw=2)
        plt.imshow(I,interpolation='nearest',cmap='winter')
        plt.colorbar()
        ax=plt.gca()
        ax.set_yticks(range(I.shape[0]))
        ax.set_yticklabels(['']*a.shape[0]+['color','rel len','abs len'])
        c1=plt.Circle((-1.5,0),radius=0.4,color='blue',clip_on=False)
        c2=plt.Circle((-1.5,1),radius=0.4,color='white',clip_on=False)
        c3=plt.Circle((-1.5,2),radius=0.4,color='yellow',clip_on=False)
        ax.add_patch(c1);ax.add_patch(c2);ax.add_patch(c3);
        c1=plt.Rectangle((-2,3),1,0.2,color='white',clip_on=False)
        c2=plt.Rectangle((-2.5,4),1.5,0.2,color='white',clip_on=False)
        c3=plt.Rectangle((-3,5),2,0.2,color='white',clip_on=False)
        ax.add_patch(c1);ax.add_patch(c2);ax.add_patch(c3);
        c1=plt.Rectangle((-2,6),1,0.2,color='gray',clip_on=False)
        c2=plt.Rectangle((-2.5,7),1.5,0.2,color='gray',clip_on=False)
        c3=plt.Rectangle((-3,8),2,0.2,color='gray',clip_on=False)
        c4=plt.Rectangle((-3.5,9),2.5,0.2,color='gray',clip_on=False)
        ax.add_patch(c1);ax.add_patch(c2);ax.add_patch(c3);ax.add_patch(c4);
        print I[-3,-1]

def LLsample(M,Y):
    LL=0
    for y in Y:
        LL+= M.exp1computeLL(y[0],y[1])
    return LL
        
np.random.seed(5)
M=Model()
##dat,f=M.exp1run()
##print M.exp1computeLL(dat,f)

n=50
Y=[]
for i in range(n):
    dat,f=M.exp1run()
    Y.append([dat,f])
#M.plothistory()
h= np.linspace(0,1,21)#np.array([1])
g= np.linspace(0,1,21)
m=np.linspace(0,2,21)
d=np.linspace(0,2,21)
import time
t0=time.time()
out=[]
for hh in h.tolist():
    print hh
    out.append([])
    for gg in g.tolist():
        out[-1].append([])
        for mm in m.tolist():
            out[-1][-1].append([])
            for dd in d.tolist():
                M.h=hh
                M.g=gg
                M.m=mm
                M.d=dd
                out[-1][-1][-1].append(LLsample(M,Y))
print time.time()-t0
np.save('out5',out)

def plotLL(fname='out4.npy'):
    h= np.linspace(0,1,21)
    g= np.linspace(0,1,21)
    m=np.linspace(0,2,21)
    d=np.linspace(0,2,21)
    out=np.load('out4.npy')
    for mm in range(m.size/2):
        for dd in range(d.size/2):
            plt.subplot(10,10,(9-mm)*10+dd+1)
            plt.pcolormesh(h,g,out[:,:,mm*2,dd*2],cmap='winter',vmax=0,vmin=-500)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            if dd==0:
                print mm,dd
                plt.ylabel('%.1f'%m[mm*2])
            if mm==0: plt.xlabel('%.1f'%d[dd*2])
        



    
