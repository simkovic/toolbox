import numpy as np
import pylab as plt
plt.ion()
plt.close('all')
np.random.seed(3)
T=100
D=np.zeros((T,5,3))*np.nan
for t in range(T):
    for i in range(3):
        D[t,:-1,i]=np.random.permutation(4)
    D[t,-1,:]= np.random.randint(4,size=3)
D=np.int32(D)
# constant feedback rule
rule=np.zeros(T,dtype=int)
for i in range(T/10):
    a=rule[i*T/10-1]
    while a==rule[i*T/10-1]: a=np.random.randint(3)
    rule[i*T/10:(i+1)*T/10]=a
# identify target
target=np.zeros(T,dtype=int)   
for i in range(T):
    target[i]= np.nonzero(D[i,-1,rule[i]]==D[i,:-1,rule[i]])[0][0]


def sampleModel(D,target,r=0.5,q=0.5,d=1,f=1,B=None):    
    a=np.zeros((T+1,3))*np.nan
    m=np.zeros((T,3),dtype=int)
    s=np.zeros((T,3))
    p=np.zeros((T,4))
    cor=np.zeros(T,dtype=int)
    if B is None: choice=np.zeros(T,dtype=int)
    else: choice=B
    a[0,:]=1/3.
    LL=0
    for t in range(T):
        n=np.array(D[t,-1,:],ndmin=2)==D[t,:-1,:]
        p[t,:]= n.dot(np.power(a[t,:],d))
        p[t,:]/= p[t,:].sum()

        if B is None:
            choice[t]=int(np.random.multinomial(1,p[t,:]).nonzero()[0][0])
        else: LL+=np.log(max(p[t,choice[t]],0.0001))
        cor[t]=target[t]==choice[t]
        m[t,:]= D[t,-1,:]==D[t,choice[t],:]
        if cor[t]:
            s[t,:]=m[t,:]*np.power(a[t,:],f)
            s[t,:]/= s[t,:].sum()
            a[t+1,:]= (1-r)*a[t,:]+r*s[t,:]
        else:
            s[t,:]=(1-m[t,:])*np.power(a[t,:],f)
            s[t,:]/= s[t,:].sum()
            a[t+1,:]= (1-q)*a[t,:]+q*s[t,:]
        if np.any(a[t+1,:]==1):
            a[t+1,:]+= 0.0001
            a[t+1,:]/= 1.0003
    if B is None: return choice
    else: return LL


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
np.save('out',out)


