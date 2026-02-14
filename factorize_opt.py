
import pyscf
from pyscf import gto,scf,ao2mo
from pyscf_utils import *
import mlx.core as mx
from mlx.optimizers import Adam

def XDF(I): # input ERI
    
    s0 = int(I.shape[0])
    s1 = s0**2
    
    IS=I.reshape(s1,s1)
    a,b=np.linalg.eigh(IS)
    Us = np.zeros([s1,s0,s0])
    Zs = np.zeros([s1,s0,s0])
    ind = 0
    for j in range(s1-1,0-1,-1):
        a1,b1=np.linalg.eigh(b[:,j].reshape(s0,s0))
        Us[ind,:,:]=b1
        Zs[ind,:,:]=a[j]*np.outer(a1,a1)
        ind+=1
    return mx.array(Us),mx.array(Zs)
    
def CDF_mlx(eri,U,Z,B=None,batch=None,iso=1,l1=1e-6,lr=.001,maxiter=10000,iprint=True):
    n0 = eri.shape[0]
    if B is None:
        params={'U':U,'Z':Z}
    else:
        params={'U':U,'Z':Z,'B':B}
    N = eri.shape[0]

    def B_pass(params):
        X = 0.5*( params['B'] + params['B'].T)
        D = mx.eye(n0)
        B = mx.kron(D,X).reshape(n0,n0,n0,n0) 
        B=B.swapaxes(1,2)
    
        A = mx.kron(X,D).reshape(n0,n0,n0,n0) 
        A=A.swapaxes(1,2)
    
        B = 0.5*(A+B)
        return B
        
    def forward_pass(params):
        if B is None:
            if batch is None:
                return mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U'],params['U'],params['Z'],params['U'],params['U'],stream=mx.gpu)
            else:
                erx=mx.zeros(eri.shape)
                for j in range(len(batch)-1):
                    erx+=mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U'][batch[j]:batch[j+1],:,:],params['U'][batch[j]:batch[j+1],:,:],
                                   params['Z'][batch[j]:batch[j+1],:,:],params['U'][batch[j]:batch[j+1],:,:],
                                   params['U'][batch[j]:batch[j+1],:,:],stream=mx.gpu)
                return erx
        else:
            if batch is None:
                return mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U'],params['U'],
                                 params['Z'],params['U'],params['U'],stream=mx.gpu) + B_pass(params)
            else:
                erx=mx.zeros(eri.shape)
                for j in range(len(batch)-1):
                    erx+=mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U'][batch[j]:batch[j+1],:,:],params['U'][batch[j]:batch[j+1],:,:],
                                   params['Z'][batch[j]:batch[j+1],:,:],params['U'][batch[j]:batch[j+1],:,:],
                                   params['U'][batch[j]:batch[j+1],:,:],stream=mx.gpu)
                return erx + B_pass(params)
                

    def cdf_norm(params):
        return mx.sum(mx.abs(params['Z']))

    def iso_norm(params):
        II =  mx.einsum('tpq,trq->tpr',params['U'],params['U'],stream=mx.gpu)
        return mx.sum((mx.eye(N) - II)**2)

    def loss_fn(params):
        return mx.sum((eri-forward_pass(params))**2) + iso*iso_norm(params) + l1*cdf_norm(params)

    loss_fn = mx.compile(loss_fn)
    grad_fn = mx.grad(loss_fn, argnums=0)

    import time
    t0=time.time()
    optimizer = Adam(learning_rate=lr)
    maxiter = maxiter
    data = mx.zeros(maxiter)
    for step in range(maxiter):
        grads = grad_fn(params)
        loss = loss_fn(params)
    
        params = optimizer.apply_gradients(grads, params)
    
        mx.eval(loss)
        data[step]=loss.item()
        if step % 500 == 0:
            if iprint==True:
                print(f"Step {step}: Loss = {loss.item():.8f}")
    t1 = time.time()
    print('************************************')
    print('Final Loss: ',loss.item())
    print('Elapsed Time: ',t1-t0,' s')

    return params,data

def iTHC_spatial(eri,X,Z,B=None,maxiter=10000,l1=1e-6,iso=1.,lr=.001,iprint=True):
    if B is None:
        params={'X':X,'Z':Z}
    else:
        params={'X':X,'Z':Z,'B':B}
        
    n0=eri.shape[0]
    M=Z.shape[0]

    def B_pass(params):
        X = 0.5*( params['B'] + params['B'].T)
        D = mx.eye(n0)
        B = mx.kron(D,X).reshape(n0,n0,n0,n0) 
        B=B.swapaxes(1,2)
    
        A = mx.kron(X,D).reshape(n0,n0,n0,n0) 
        A=A.swapaxes(1,2)
    
        B = 0.5*(A+B)
        return B

    def forward_pass(params):
        if B is None:
            return mx.einsum('pk,qk,kl,rl,sl,->pqrs',params['X'],params['X'],params['Z'],params['X'],params['X'],stream=mx.gpu)
        else:
            return mx.einsum('pk,qk,kl,rl,sl,->pqrs',params['X'],params['X'],params['Z'],params['X'],params['X'],stream=mx.gpu) + B_pass(params)

    def thc_norm(params):
        return mx.sum(mx.abs(params['Z']))

    def iso_norm(params):
        return mx.sum((mx.eye(n0)-params['X']@params['X'].T)**2)

    def loss_fn(params):
        return mx.sum((eri - forward_pass(params))**2) + iso*iso_norm(params) + l1*thc_norm(params)

    loss_fn = mx.compile(loss_fn)
    grad_fn = mx.grad(loss_fn, argnums=0)

    import time
    t0=time.time()
    optimizer = Adam(learning_rate=lr)
    maxiter = maxiter
    data = mx.zeros(maxiter)
    for step in range(maxiter):
        grads = grad_fn(params)
        loss = loss_fn(params)
    
        params = optimizer.apply_gradients(grads, params)
    
        mx.eval(loss)
        data[step]=loss.item()
        if step % 500 == 0:
            if iprint==True:
                print(f"Step {step}: Loss = {loss.item():.8f}")
    t1 = time.time()
    print('************************************')
    print('Final Loss: ',loss.item())
    print('Elapsed Time: ',t1-t0,' s')

    return params,data

def iTHC_spin_BS(eri,U0,Z0,X,Z,B=None,maxiter=10000,l1=1e-6,iso=1.,lr=.001,iprint=True):
    if B is None:
        params={'U0':U0,'Z0':Z0,'X':X,'Z':Z,'SXY':mx.array(.1)}
    else:
        params={'U0':U0,'Z0':Z0,'X':X,'Z':Z,'Ba':B[0,:,:],'Bb':B[1,:,:],'SXY':mx.array(.1)}
    print(params['U0'].shape)
    n0=eri.shape[0]
    N = 2*eri.shape[0]
    M=Z.shape[0]
    I = mx.eye(n0)
    II = mx.outer(mx.ones(n0),mx.ones(n0))
    N2 = mx.einsum('pk,qk,kl,rl,sl->pqrs',I,I,II,I,I,stream=mx.gpu)
    #print((params['Nab'][0,0]*N2).shape)

    def Ba_pass(params):
        X = 0.5*( params['Ba'] + params['Ba'].T)
        D = mx.eye(n0)
        B = mx.kron(D,X).reshape(n0,n0,n0,n0) 
        B=B.swapaxes(1,2)
    
        A = mx.kron(X,D).reshape(n0,n0,n0,n0) 
        A=A.swapaxes(1,2)
    
        B = 0.5*(A+B)
        return B

    def Bb_pass(params):
        X = 0.5*( params['Bb'] + params['Bb'].T)
        D = mx.eye(n0)
        B = mx.kron(D,X).reshape(n0,n0,n0,n0) 
        B=B.swapaxes(1,2)
    
        A = mx.kron(X,D).reshape(n0,n0,n0,n0) 
        A=A.swapaxes(1,2)
    
        B = 0.5*(A+B)
        return B

    def forward_pass(params):
        if B is None:
            er00 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', params['X'][0:n0,:], params['X'][0:n0,:],
                              params['Z'], params['X'][0:n0,:], params['X'][0:n0,:],stream=mx.gpu) 
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U0'],params['U0'],
                               params['Z0'][:,0:n0,0:n0],params['U0'],params['U0'],stream=mx.gpu))
            er11 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', params['X'][n0:2*n0,:], params['X'][n0:2*n0,:],
                            params['Z'], params['X'][n0:2*n0,:], params['X'][n0:2*n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U0'],params['U0'],
                               params['Z0'][:,n0:,n0:],params['U0'],params['U0'],stream=mx.gpu))
            er10 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', params['X'][n0:2*n0,:], params['X'][n0:2*n0,:],
                        params['Z'], params['X'][0:n0,:], params['X'][0:n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U0'],params['U0'],
                               params['Z0'][:,n0:,0:n0],params['U0'],params['U0'],stream=mx.gpu))
            er01 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', params['X'][0:n0,:], params['X'][0:n0,:],
                        params['Z'], params['X'][n0:2*n0,:], params['X'][n0:2*n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U0'],params['U0'],
                               params['Z0'][:,0:n0,n0:],params['U0'],params['U0'],stream=mx.gpu))
        else:
            er00 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', params['X'][0:n0,:], params['X'][0:n0,:],
                              params['Z'], params['X'][0:n0,:], params['X'][0:n0,:],stream=mx.gpu) 
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U0'],params['U0'],
                               params['Z0'][:,0:n0,0:n0],params['U0'],params['U0'],stream=mx.gpu) + Ba_pass(params))
            er11 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', params['X'][n0:2*n0,:], params['X'][n0:2*n0,:],
                            params['Z'], params['X'][n0:2*n0,:], params['X'][n0:2*n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U0'],params['U0'],
                               params['Z0'][:,n0:,n0:],params['U0'],params['U0'],stream=mx.gpu) + Ba_pass(params))
            er10 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', params['X'][n0:2*n0,:], params['X'][n0:2*n0,:],
                        params['Z'], params['X'][0:n0,:], params['X'][0:n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U0'],params['U0'],
                               params['Z0'][:,n0:,0:n0],params['U0'],params['U0'],stream=mx.gpu) + Bb_pass(params))
            er01 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', params['X'][0:n0,:], params['X'][0:n0,:],
                        params['Z'], params['X'][n0:2*n0,:], params['X'][n0:2*n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',params['U0'],params['U0'],
                               params['Z0'][:,0:n0,n0:],params['U0'],params['U0'],stream=mx.gpu) + Bb_pass(params))
        
        

        er0110 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', params['X'][0:n0,:], params['X'][n0:2*n0,:],
                        params['Z'], params['X'][n0:2*n0,:], params['X'][0:n0,:],stream=mx.gpu) + params['SXY']*N2) 

        er1001 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', params['X'][n0:2*n0,:], params['X'][0:n0,:],
                        params['Z'], params['X'][0:n0,:], params['X'][n0:2*n0,:],stream=mx.gpu) + params['SXY']*N2) 

        

        return er00,er11,er01-er0110.transpose(0,3,2,1),er10-er1001.transpose(0,3,2,1)
        

    def ithc_norm(params):
        return mx.sum(mx.abs(params['Z'])) + mx.sum(mx.abs(params['Z0']))

    def iso_norm(params):
        II =  mx.einsum('tpq,trq->tpr',params['U0'],params['U0'],stream=mx.gpu)
        return mx.sum((mx.eye(N)-params['X']@params['X'].T)**2) + 4*mx.sum((mx.eye(n0) - II)**2)

    def loss_fn(params):
        EX=forward_pass(params)
        return (mx.sum((eri-EX[0])**2) + mx.sum((eri-EX[1])**2) + mx.sum((eri-EX[2])**2) + mx.sum((eri-EX[3])**2) 
                + iso*iso_norm(params) + l1*ithc_norm(params))


    loss_fn = mx.compile(loss_fn)
    grad_fn = mx.grad(loss_fn, argnums=0)

    import time
    t0=time.time()
    optimizer = Adam(learning_rate=lr)
    maxiter = maxiter
    data = mx.zeros(maxiter)
    for step in range(maxiter):
        grads = grad_fn(params)
        loss = loss_fn(params)
    
        params = optimizer.apply_gradients(grads, params)
    
        mx.eval(loss)
        data[step]=loss.item()
        if step % 500 == 0:
            if iprint==True:
                print(f"Step {step}: Loss = {loss.item():.8f}")
    t1 = time.time()
    print('************************************')
    print('Final Loss: ',loss.item())
    print('Elapsed Time: ',t1-t0,' s')

    return params,data  

def ISO_X(X):
    N = X.shape[0]
    M = X.shape[1]
    Xi = mx.zeros([N,M])
    si = mx.zeros([N,M])
    U,S,V = mx.linalg.svd(X,stream=mx.cpu)
    for j in range(N):
        si[j,j]=1
    return U@si@V

    