# ERI-Factorizations
* Build for Apple Silicon GPUs so reiquires Apple Silicon and installation of the MLX package
* Explicit Double Factorization (XDF) , Compressed Double Factorization (CDF), isometric-Tensor Hyper-Contraction (iTHC)
* XDF,CDF,iTHC, and spin orbitial iTHC codes
* XDF is called from factorize_opt.py 
  * Currently uses a double eigen-decomposition, need to update to the pivoted Cholesky method
* CDF and iTHC  have block invariant shift option in the optimization

## Examples
```python
from Factorize import *
from factorize_opt import *
from pyscf import gto,scf,ao2mo
from mol_obj import *

# Define molecule with pyscf
n=10
dx=2.0 # atom spacing
basis = 'sto-6g'
mol = gto.Mole()
HL = []
for j in range(n):
    HL.append(['H',(0,0,j*dx)])
mol.atom =HL
if basis==None:
    mol.basis = 'sto-6g'
else:
    mol.basis = basis
#mol.basis = '6-31g'
mol.build()

ints=RHFIntegrals(mol)
ints.build() # Container for molecular properites

# XDF

U_xdf,Z_xdf= XDF(ints.eri)

# CDF
system={'ecore':ints.e_nuc,'h1':ints.h1[:10,:10],'eri':ints.eri,'NA':ints.nelec[0],'NB':ints.nelec[1]}
eri_cdf=CDF(system)
eri_cdf.initial_guess(ndf=4, B=False) # ndf = number of layers or stages, used the first ndf layers of XDF calculation as initial guess
# B=False means set block invariant option in optimization to False. I don't find it help must for CDF

eri_cdf.optimize(batch=None,iso=1,l1=1e-6,lr=.001,maxiter = 40000) # Optimize with ADAM lr = learning rate, maxiter = max iterations
# l1 is regularization of core tensor: l1*|Z^t_{pq}|_1
# iso is regulation of each leaf U^t being orthogonal

# iTHC

eri_ithc = iTHC(system)
eri_ithc.initial_guess(M, B=True) # M= THC rank , and here we set block invariant shift to True for the optimization

eri_ithc.optimize(maxiter=20000,l1=1e-7,iso=1.,lr=.001,iprint=True)
```









