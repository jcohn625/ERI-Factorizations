
import pyscf
from pyscf import gto,scf,ao2mo
from pyscf_utils import *
import mlx.core as mx
from mlx.optimizers import Adam
from factorize_opt import *
import scipy
from scipy.optimize import minimize


class CDF:

    def __init__(self,system):
        self.ecore=system['ecore']
        self.h1 = system['h1']
        self.eri = mx.array(system['eri'])
        self.NA = system['NA']
        self.NB = system['NB']
        self.norb = system['h1'].shape[0]


    def initial_guess(self,ndf, B=False):
        U0,Z0=XDF(self.eri)
        self.U=U0[0:ndf,:,:]
        self.Z=Z0[0:ndf,:,:]
        self.ndf = ndf

    
        if B:
            self.B = .001*mx.eye(self.norb)
        else:
            self.B = None

    def optimize(self,batch=None,iso=1,l1=1e-6,lr=.001,maxiter=10000,iprint=True):
        self.batch = batch
        X=CDF_mlx(self.eri,self.U,self.Z,self.B,batch=self.batch,iso=iso,l1=l1,lr=lr,maxiter=maxiter,iprint=iprint)
        self.data = X[1]
        self.U = X[0]['U']
        self.Z = X[0]['Z']
        if self.B is not None:
            self.B = X[0]['B']

    def B_pass(self):
        if self.B is None:
            return 0.
        else:
            n0 = self.norb
            X = 0.5*( self.B + self.B.T)
            D = mx.eye(n0)
            BB = mx.kron(D,X).reshape(n0,n0,n0,n0) 
            BB=BB.swapaxes(1,2)
        
            AA = mx.kron(X,D).reshape(n0,n0,n0,n0) 
            AA=AA.swapaxes(1,2)
        
            Bb = 0.5*(AA+BB)
            return Bb
        
    def approx_eri(self):
        if self.B is None:
            if self.batch is None:
                return mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U,self.U,self.Z,self.U,self.U,stream=mx.gpu)
            else: 
                erx=mx.zeros(self.eri.shape)
                for j in range(len(self.batch)-1):
                    erx+=mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U[self.batch[j]:self.batch[j+1],:,:],self.U[self.batch[j]:self.batch[j+1],:,:],
                                   self.Z[self.batch[j]:self.batch[j+1],:,:],self.U[self.batch[j]:self.batch[j+1],:,:],
                                   self.U[self.batch[j]:self.batch[j+1],:,:],stream=mx.gpu)
                return erx
        else:
            if self.batch is None:
                return mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U,self.U,self.Z,self.U,self.U,stream=mx.gpu) + self.B_pass()
            else:
                erx=mx.zeros(self.eri.shape)
                for j in range(len(self.batch)-1):
                    erx+=mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U[self.batch[j]:self.batch[j+1],:,:],self.U[self.batch[j]:self.batch[j+1],:,:],
                                   self.Z[self.batch[j]:self.batch[j+1],:,:],self.U[self.batch[j]:self.batch[j+1],:,:],
                                   self.U[self.batch[j]:self.batch[j+1],:,:],stream=mx.gpu)
                return erx + self.B_pass()
                

    def l2_error(self):
        
        return mx.sum((self.eri - self.approx_eri())**2)

    def rel_error(self):
        return mx.sqrt(self.l2_error()/mx.sum(self.eri**2))

    def opt_1_norm(self):
        z0=0
        n0=self.norb
        I = mx.eye(n0)
        II = mx.outer(mx.ones(n0),mx.ones(n0))
        for j in range(self.ndf):
            Zi = self.Z[j,:,:]
            def loss(a):
                return mx.sum(mx.abs(Zi - a*II))
            x0 = minimize(loss,0.0)
            z0+= x0.fun
        return z0

class iTHC:

    def __init__(self,system):
        self.ecore=system['ecore']
        self.h1 = system['h1']
        self.eri = mx.array(system['eri'])
        self.NA = system['NA']
        self.NB = system['NB']
        self.norb = system['h1'].shape[0]

    def initial_guess(self,M, B=False):
        self.M=M

        self.X = ISO_X(mx.random.normal((self.norb,self.M)))
        Z = mx.random.normal((self.M,self.M))
        self.Z = 0.5*(Z + Z.T)
    
        if B:
            self.B = .001*mx.eye(self.norb)
        else:
            self.B = None

    def optimize(self,maxiter=10000,l1=1e-6,iso=1.,lr=.001,iprint=True):
        res=iTHC_spatial(self.eri,self.X,self.Z,self.B,maxiter=maxiter,l1=l1,iso=iso,lr=lr,iprint=iprint)
        self.data = res[1]
        self.X = res[0]['X']
        self.Z = res[0]['Z']
        if self.B is not None:
            self.B = res[0]['B']

    def B_pass(self):
        if self.B is None:
            return 0.
        else:
            n0 = self.norb
            X = 0.5*( self.B + self.B.T)
            D = mx.eye(n0)
            BB = mx.kron(D,X).reshape(n0,n0,n0,n0) 
            BB=BB.swapaxes(1,2)
        
            AA = mx.kron(X,D).reshape(n0,n0,n0,n0) 
            AA=AA.swapaxes(1,2)
        
            Bb = 0.5*(AA+BB)
            return Bb

    def approx_eri(self):
        if self.B is None:
            return mx.einsum('pk,qk,kl,rl,sl,->pqrs',self.X,self.X,self.Z,self.X,self.X,stream=mx.gpu)
        else:
            return mx.einsum('pk,qk,kl,rl,sl,->pqrs',self.X,self.X,self.Z,self.X,self.X,stream=mx.gpu) + self.B_pass()

    def l2_error(self):
        return mx.sum((self.eri - self.approx_eri())**2)

    def rel_error(self):
        return mx.sqrt(self.l2_error()/mx.sum(self.eri**2))

    # def norm1(self):

    #     h0 = self.h1 - 0.5*mx.einsum('prrs->ps',self.eri,stream=mx.gpu) + mx.einsum('prss->pr',self.eri,stream=mx.gpu) self.NA*self.B
        
class iTHC_SO:

    def __init__(self,system):
        self.ecore=system['ecore']
        self.h1 = system['h1']
        self.eri = mx.array(system['eri'])
        self.NA = system['NA']
        self.NB = system['NB']
        self.norb = system['h1'].shape[0]

    def initial_guess(self,ndf,M, B=False):
        self.M=M
        self.ndf=ndf

        self.X = ISO_X(mx.random.normal((2*self.norb,self.M)))
        Z = mx.random.normal((self.M,self.M))
        self.Z = 0.5*(Z + Z.T)
        U0,Z0 = XDF(self.eri)
        self.U0=U0[0:ndf,:,:]
        self.Z0 = mx.random.normal((self.ndf,2*self.norb,2*self.norb))
    
        if B:
            b=.001*mx.eye(self.norb)
            bb = mx.zeros((2,self.norb,self.norb))
            bb[0,:,:]=b
            bb[1,:,:]=b
            self.B = bb
        else:
            self.B = None

    def optimize(self,maxiter=10000,l1=1e-6,iso=1.,lr=.001,iprint=True):
        res=iTHC_spin_BS(self.eri,self.U0,self.Z0,self.X,self.Z,self.B,maxiter=maxiter,l1=l1,iso=iso,lr=lr,iprint=iprint)
        self.data = res[1]
        self.U0 = res[0]['U0']
        self.Z0 = res[0]['Z0']
        self.X = res[0]['X']
        self.Z = res[0]['Z']
        self.SXY = res[0]['SXY']
        if self.B is not None:
            Ba = res[0]['Ba']
            Bb = res[0]['Bb']
            bb = bb = mx.zeros((2,self.norb,self.norb))
            bb[0,:,:] = Ba
            bb[1,:,:] = Bb
            self.B = bb


    def Ba_pass(self):
        if self.B is None:
            return 0.
        else:
            n0 = self.norb
            X = 0.5*( self.B[0,:,:] + self.B[0,:,:].T)
            D = mx.eye(n0)
            B = mx.kron(D,X).reshape(n0,n0,n0,n0) 
            B=B.swapaxes(1,2)
        
            A = mx.kron(X,D).reshape(n0,n0,n0,n0) 
            A=A.swapaxes(1,2)
        
            B = 0.5*(A+B)
            return B

    def Bb_pass(self):
        if self.B is None:
            return 0.
        else:
            n0 = self.norb
            X = 0.5*( self.B[1,:,:] + self.B[1,:,:].T)
            D = mx.eye(n0)
            B = mx.kron(D,X).reshape(n0,n0,n0,n0) 
            B=B.swapaxes(1,2)
        
            A = mx.kron(X,D).reshape(n0,n0,n0,n0) 
            A=A.swapaxes(1,2)
        
            B = 0.5*(A+B)
            return B

    def approx_eris(self):
        n0 = self.norb
        I = mx.eye(n0)
        II = mx.outer(mx.ones(n0),mx.ones(n0))
        N2 = mx.einsum('pk,qk,kl,rl,sl->pqrs',I,I,II,I,I,stream=mx.gpu)
        if self.B is None:
            
            er00 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', self.X[0:n0,:], self.X[0:n0,:],
                              self.Z, self.X[0:n0,:], self.X[0:n0,:],stream=mx.gpu) 
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U0,params['U0'],
                               self.Z0[:,0:n0,0:n0],self.U0,self.U0,stream=mx.gpu))
            er11 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', self.X[n0:2*n0,:], self.X[n0:2*n0,:],
                            self.Z, self.X[n0:2*n0,:], self.X[n0:2*n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U0,self.U0,
                               self.Z0[:,n0:,n0:],self.U0,self.U0,stream=mx.gpu))
            er10 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', self.X[n0:2*n0,:], sself.X[n0:2*n0,:],
                        self.Z, self.X[0:n0,:], self.X[0:n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U0,self.U0,
                               self.Z0[:,n0:,0:n0],self.U0,self.U0,stream=mx.gpu))
            er01 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', self.X[0:n0,:], self.X[0:n0,:],
                        self.Z, self.X[n0:2*n0,:], self.X[n0:2*n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U0,self.U0,
                               self.Z0[:,0:n0,n0:],self.U0,self.U0,stream=mx.gpu))
        else:
            er00 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', self.X[0:n0,:], self.X[0:n0,:],
                              self.Z, self.X[0:n0,:], self.X[0:n0,:],stream=mx.gpu) 
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U0,self.U0,
                               self.Z0[:,0:n0,0:n0],self.U0,self.U0,stream=mx.gpu) + self.Ba_pass())
            er11 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', self.X[n0:2*n0,:], self.X[n0:2*n0,:],
                            self.Z, self.X[n0:2*n0,:], self.X[n0:2*n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U0,self.U0,
                               self.Z0[:,n0:,n0:],self.U0,self.U0,stream=mx.gpu) + self.Ba_pass())
            er10 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', self.X[n0:2*n0,:], self.X[n0:2*n0,:],
                        self.Z, self.X[0:n0,:], self.X[0:n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U0,self.U0,
                               self.Z0[:,n0:,0:n0],self.U0,self.U0,stream=mx.gpu) + self.Bb_pass())
            er01 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', self.X[0:n0,:], self.X[0:n0,:],
                        self.Z, self.X[n0:2*n0,:], self.X[n0:2*n0,:],stream=mx.gpu)
                   + mx.einsum('tpk,tqk,tkl,trl,tsl->pqrs',self.U0,self.U0,
                               self.Z0[:,0:n0,n0:],self.U0,self.U0,stream=mx.gpu) + self.Bb_pass())
        
        

        er0110 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', self.X[0:n0,:], self.X[n0:2*n0,:],
                        self.Z, self.X[n0:2*n0,:],self.X[0:n0,:],stream=mx.gpu) + self.SXY*N2) 

        er1001 = (mx.einsum('pk,qk,kl,rl,sl->pqrs', self.X[n0:2*n0,:], self.X[0:n0,:],
                        self.Z, self.X[0:n0,:], self.X[n0:2*n0,:],stream=mx.gpu) + self.SXY*N2) 

        

        return er00,er11,er01-er0110.transpose(0,3,2,1),er10-er1001.transpose(0,3,2,1)

    def l2_error(self):
        e00,e11,e01,e10 = self.approx_eris()
        return mx.sum((self.eri - e00)**2),mx.sum((self.eri - e11)**2),mx.sum((self.eri - e01)**2),mx.sum((self.eri - e10)**2)

    def rel_error(self):
        return mx.sqrt(self.l2_error()/mx.sum(self.eri**2))

    
            

            

            

    

    
    



    

    

    
             
        
        