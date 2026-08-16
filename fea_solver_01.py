from __future__ import annotations

import numpy as np

TOTAL_LENGTH = 2.0
AREA = 100e-6
YOUNGS_MODULUS = 200e9
APPLIED_FORCE = 10e3
NUMBER_OF_ELEMENTS = 2

def axial_bar_stiffness(E: float, A: float, length: float) -> np.ndarray:
    if E <= 0: raise ValueError("Young's modulus must be positive.")
    if A <= 0: raise ValueError("Cross-sectional area must be positive.")
    if length <= 0: raise ValueError("Element length must be positive.")
    return E*A/length*np.array([[1.0,-1.0],[-1.0,1.0]])

def assemble_global_stiffness(E,A,total_length,number_of_elements):
    number_of_nodes=number_of_elements+1; element_length=total_length/number_of_elements
    K=np.zeros((number_of_nodes,number_of_nodes)); element_matrices=[]
    for element in range(number_of_elements):
        ke=axial_bar_stiffness(E,A,element_length); element_matrices.append(ke.copy())
        i,j=element,element+1
        K[i,i]+=ke[0,0]; K[i,j]+=ke[0,1]; K[j,i]+=ke[1,0]; K[j,j]+=ke[1,1]
    return K,element_matrices

def solve_linear_system(K,force_vector,fixed_dofs):
    all_dofs=np.arange(len(force_vector)); free_dofs=np.array([d for d in all_dofs if d not in fixed_dofs],dtype=int)
    K_ff=K[np.ix_(free_dofs,free_dofs)]; F_f=force_vector[free_dofs]
    U=np.zeros(len(force_vector)); U[free_dofs]=np.linalg.solve(K_ff,F_f)
    return U,K@U-force_vector

def calculate_element_results(U,E,A,total_length,number_of_elements):
    Le=total_length/number_of_elements; strains=[]; stresses=[]; internal_forces=[]
    for element in range(number_of_elements):
        i,j=element,element+1; strain=(U[j]-U[i])/Le; stress=E*strain; internal_forces.append(A*stress); strains.append(strain); stresses.append(stress)
    return np.array(strains),np.array(stresses),np.array(internal_forces)

def analytical_solution(E,A,total_length,force): return force*total_length/(A*E),force/A

def solve_fea_bar(total_length=TOTAL_LENGTH,area=AREA,youngs_modulus=YOUNGS_MODULUS,applied_force=APPLIED_FORCE,number_of_elements=NUMBER_OF_ELEMENTS):
    n=number_of_elements+1; Le=total_length/number_of_elements
    K,element_matrices=assemble_global_stiffness(youngs_modulus,area,total_length,number_of_elements)
    F=np.zeros(n); F[-1]=applied_force
    U,reactions=solve_linear_system(K,F,[0])
    strains,stresses,internal_forces=calculate_element_results(U,youngs_modulus,area,total_length,number_of_elements)
    analytical_displacement,analytical_stress=analytical_solution(youngs_modulus,area,total_length,applied_force)
    error=abs(U[-1]-analytical_displacement)/abs(analytical_displacement)*100 if analytical_displacement else 0.0
    return {'total_length':total_length,'area':area,'youngs_modulus':youngs_modulus,'applied_force':applied_force,'number_of_elements':number_of_elements,'number_of_nodes':n,'element_length':Le,'element_matrices':element_matrices,'global_stiffness':K,'force_vector':F,'fixed_dofs':[0],'displacements':U,'reactions':reactions,'strains':strains,'stresses':stresses,'internal_forces':internal_forces,'analytical_displacement':analytical_displacement,'analytical_stress':analytical_stress,'displacement_error_percent':error}

def print_report(results):
    print('\n'+'='*70); print('FINITE ELEMENT ANALYSIS — TUTORIAL 01'); print('1D AXIAL BAR'); print('='*70)
    print(f"\nTip displacement: {results['displacements'][-1]*1000:.6f} mm")
    print(f"Analytical displacement: {results['analytical_displacement']*1000:.6f} mm")
    print(f"Displacement error: {results['displacement_error_percent']:.6e} %")
    print(f"Reaction: {results['reactions'][0]:.6f} N")

if __name__=='__main__': print_report(solve_fea_bar())
