"""FEA From Scratch #07 - Stress, Strain & Reaction Forces."""
import numpy as np

NODES=np.array([[0.,0.],[2.,0.],[1.,1.5]])
ELEMENTS=((0,1),(0,2),(1,2))
A=100e-6
E=200e9
P=-10000.

def element(i,j):
    dx,dy=NODES[j]-NODES[i]; L=float(np.hypot(dx,dy))
    c,s=dx/L,dy/L
    k=A*E/L*np.array([[c*c,c*s,-c*c,-c*s],[c*s,s*s,-c*s,-s*s],[-c*c,-c*s,c*c,c*s],[-c*s,-s*s,c*s,s*s]])
    dofs=np.array([2*i,2*i+1,2*j,2*j+1])
    return dict(nodes=(i+1,j+1),dofs=dofs,L=L,c=c,s=s,ke=k)

def solve_structure():
    K=np.zeros((6,6)); els=[]
    for i,j in ELEMENTS:
        e=element(i,j); d=e['dofs']
        for a,I in enumerate(d):
            for b,J in enumerate(d): K[I,J]+=e['ke'][a,b]
        els.append(e)
    F=np.zeros(6); F[5]=P
    constrained=np.array([0,1,3]); free=np.array([2,4,5])
    Kff=K[np.ix_(free,free)]; Uf=np.linalg.solve(Kff,F[free])
    U=np.zeros(6); U[free]=Uf
    R=K@U-F
    for e in els:
        c,s,L,d=e['c'],e['s'],e['L'],e['dofs']; ue=U[d]
        delta=float(np.dot([-c,-s,c,s],ue))
        strain=delta/L; stress=E*strain; force=A*stress
        e.update(ue=ue,delta=delta,strain=strain,stress=stress,force=force,
                 local_force=A*E/L*delta*np.array([-1.,1.]))
    return dict(nodes=NODES,elements=els,A=A,E=E,P=P,K=K,F=F,U=U,Uf=Uf,
                free_dofs=free,constrained_dofs=constrained,reactions=R,
                symmetry_error=np.max(np.abs(K-K.T)),
                vertical_equilibrium=R[1]+R[3]+P,
                horizontal_equilibrium=R[0])

def solve_fea(): return solve_structure()

if __name__=='__main__':
    r=solve_structure(); np.set_printoptions(precision=9,suppress=True)
    print('DISPLACEMENTS [mm]\n',r['U']*1000)
    print('REACTIONS [kN]\n',r['reactions']/1000)
    for n,e in enumerate(r['elements'],1):
        print(f'E{n}: L={e["L"]:.6f} m, delta={e["delta"]*1000:.6f} mm, strain={e["strain"]:.9e}, stress={e["stress"]/1e6:.6f} MPa, force={e["force"]/1000:.6f} kN')
    print('K symmetry error:',r['symmetry_error'])
    print('Vertical equilibrium [N]:',r['vertical_equilibrium'])
    print('Horizontal equilibrium [N]:',r['horizontal_equilibrium'])
