import numpy as np
import pylab as plt
plt.ion()
plt.close('all')
np.random.seed(3)
T=20
S=3
U=3#4
D=np.zeros((T,U+1,S))*np.nan
##for t in range(T):
##    for i in range(S):
##        D[t,:-1,i]=np.random.permutation(U)
##    D[t,-1,:]= np.random.randint(U,size=S)
##
### constant feedback rule
##rule=np.zeros(T,dtype=int)
##for i in range(T/10):
##    a=rule[i*T/10-1]
##    while a==rule[i*T/10-1]: a=np.random.randint(S)
##    rule[i*10:(i+1)*10]=a
### identify target
##target=np.zeros(T,dtype=int)   
##for i in range(T):
##    target[i]= np.nonzero(D[i,-1,rule[i]]==D[i,:-1,rule[i]])[0][0]
###out3
##target[:10]=0
##D[:10,:]=D[0,:]
##D[10:,:]=D[10,:]
##target[10:]=3
#out4
D[0,0,:]=0
D[0,1,:]=1
D[0,2,:]=2
D[0,3,:]=D[0,0,:]
D[:,:,:]=D[0,:,:]

D[10:,0,0]=1
D[10:,1,0]=2
D[10:,2,0]=0
D[10:,3,0]=1

D=np.int32(D)
target=np.zeros(T,dtype=int)

def sampleModel(D,target,r=0.5,q=0.5,d=1,f=1,B=None,blind=True):
    '''
        blind - whether target features are visible to solver 
    '''
    a=np.zeros((T+1,S))*np.nan
    m=np.zeros((T,S),dtype=int)
    s=np.zeros((T,S))
    p=np.zeros((T,U))
    cor=np.zeros(T,dtype=int)
    if B is None: choice=np.zeros(T,dtype=int)*np.nan
    else: choice=B
    a[0,:]=1/float(S)
    LL=0
    for t in range(T):
        if blind: n=np.ones((U,S))
        else: n=np.array(D[t,-1,:],ndmin=2)==D[t,:-1,:]
        p[t,:]= n.dot(np.power(a[t,:],d))
        p[t,:]/= p[t,:].sum()

        if B is None:
            choice[t]=int(np.random.multinomial(1,p[t,:]).nonzero()[0][0])
        else:
            print p[t,:],choice[t]
            LL+=np.log(max(p[t,choice[t]],0.0001))
        cor[t]=target[t]==choice[t]
        m[t,:]= D[t,-1,:]==D[t,choice[t],:]#n[choice[t],:]
        #if m[t,:].sum()==0: print 'mproblem',r,q,d,f
        if cor[t]:
            s[t,:]=m[t,:]*np.power(a[t,:],f)
            #if s[t,:].sum()==0: print 'divzero1',r,q,d,f
            s[t,:]/= s[t,:].sum()
            a[t+1,:]= (1-r)*a[t,:]+r*s[t,:]
        else:
            s[t,:]=(1-m[t,:])*np.power(a[t,:],f)
            print s[t,:]
            #if s[t,:].sum()==0: print 'divzero2',r,q,d,f
            s[t,:]/= s[t,:].sum()
            a[t+1,:]= (1-q)*a[t,:]+q*s[t,:]
        if np.any(a[t+1,:]==1):# perturb to avoid extreme values
            a[t+1,:]+= 0.0001
            a[t+1,:]/= 1+0.0001*S
    print choice,cor
    print p
    print m
    print a
    if B is None: return choice
    else:return LL

B=sampleModel(D,target)
bla
N=50
B=np.zeros((N,T),dtype=int)
for i in range(N): B[i,:]=sampleModel(D,target)

r= np.linspace(0,1,21)#np.array([1])
q= np.linspace(0,1,21)
d=np.linspace(0,5,21)
f=np.linspace(0,5,21)

import time
t0=time.time()
out=[]
for rr in r.tolist():
    print rr
    out.append([])
    for qq in q.tolist():
        out[-1].append([])
        for dd in d.tolist():
            out[-1][-1].append([])
            for ff in f.tolist():
                LL=0
                for k in range(N):
                    LL+=sampleModel(D,target,r=rr,
                            q=qq,d=dd,f=ff,B=B[k,:])
                out[-1][-1][-1].append(LL)
print time.time()-t0
np.save('out4',out)


