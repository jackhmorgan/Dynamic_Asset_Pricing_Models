import numpy as np
import os
#import json
from AssetPricing import (Generate_D_Minus_E_problem,
                          Generate_D_Minus_E_problem_SV,
                          Generate_D_Minus_E_problem_RD,
#                          GenerateEmpiricalProblems,
#                          GenerateBenchmarkModel,
                          calculate_d_vector
)
from quantum_linear_system import (QuantumLinearSystemSolver,)
#                                 QuantumLinearSystemProblem)
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

file_name = 'benchmark_CRRA_10_4.txt'
script_dir = os.path.dirname(os.path.realpath(__file__))
# Define the file path
file_path = os.path.join(script_dir, file_name)

benchmark_model = np.loadtxt(file_path, dtype=complex).view(complex)

utility_function = 'CRRA'
gamma = 2
size = 4
gamma_G_1 = 0.3
gamma_G_2 = 0.01

pi_G_1 = 0.8
pi_G_2 = 0.95

p_data = 0.1

# Classical Ambiguity
d_vector = np.kron([[0],[1]], (calculate_d_vector(size)))
#d_vector /= np.linalg.norm(d_vector)
s1_vector = Statevector(d_vector)

tail_vector = np.zeros(d_vector.shape)
tail_vector[int(tail_vector.size/2)]=1
tail_operator = Statevector(tail_vector).to_operator()

s2_vector = Statevector(benchmark_model)
# `s3_problem = Generate_D_Minus_E_problem_SV(utility_function=utility_function, gamma=gamma,
# size=size, gamma_G=gamma_G)` is creating a problem instance for solving a quantum linear system
# related to the utility function. The function `Generate_D_Minus_E_problem_SV` is likely defined in
# the `AssetPricing` module and is used to generate a specific type of problem based on the provided
# utility function, gamma values, and size parameters. This problem instance is then used in the
# quantum linear system solver to find the ideal state vector `s3_vector` that represents the solution
# to the problem.

s3p_list = []
s3p_list.append(Generate_D_Minus_E_problem(utility_function=utility_function, gamma=gamma, size=size))
s3p_list.append(Generate_D_Minus_E_problem_SV(utility_function=utility_function, gamma=gamma, size=size, gamma_G=gamma_G_1, pi_G = pi_G_1))
s3p_list.append(Generate_D_Minus_E_problem_SV(utility_function=utility_function, gamma=gamma, size=size, gamma_G=gamma_G_2, pi_G=pi_G_2))
s3p_list.append(Generate_D_Minus_E_problem_RD(utility_function=utility_function, gamma=gamma, size=size))

cu_list = []
ub_list = []
lb_list = []

for s3_problem in s3p_list:
    s3_vector = QuantumLinearSystemSolver(s3_problem).ideal_x_statevector
    np.savez(utility_function+f'_{gamma}_{size}.npz', a_matrix=s3_problem.A_matrix, b_vector=s3_problem.b_vector)
    print('tail event expectation value: ', s3_vector.expectation_value(tail_operator))
    a3 = np.arccos(np.real(s3_vector.inner(s1_vector)))
    ab = np.arccos(np.real(s2_vector.inner(s1_vector)))

    s3 = 2 - (2*np.cos(a3))
    sb = 2 - (2*np.cos(ab))

    s1_observable = s1_vector.to_operator()
    s2_observable = s2_vector.to_operator()

    #Quantum Ambiguity
    a3 = np.arccos(np.real(s3_vector.inner(s1_vector)))
    ab = np.arccos(np.real(s2_vector.inner(s1_vector)))

    benchmark_dv = s2_vector - s1_vector

    benchmark_utility = (s2_vector).inner(benchmark_dv)
    print('-----')
    print(benchmark_utility)
    print(p_data+((1-p_data)*(s2_vector).inner(benchmark_dv)))
    benchmark_utility = p_data+((1-p_data)*(s2_vector).inner(benchmark_dv))

    s3 = 2 - (2*np.cos(a3))
    sb = 2 - (2*np.cos(ab))

    def QuantumAmbiguity(s1, s2, alpha, delta):
        return (np.sqrt(1-(alpha**2))*s1) + np.exp(delta*1j)*alpha*s2

    probs = list(np.linspace(0,1,100))

    classical_utilities = []
    upper_bounds = []
    lower_bounds = []

    classical_operators = []
    quantum_upper_bound_operators = []
    quantum_lower_bound_operators = []

    deltas = list(np.linspace(0,np.pi,10))
    quantum_utilities = {key : [] for key in deltas}
    for p in probs:
        # Classical
        a = np.sqrt(p)
        observable = ((1-p)*s1_observable)+(p*s2_observable)
        classical_operators.append(observable)
        utility = s3_vector.expectation_value(observable.data)
        utility *= s3/sb
        classical_utilities.append(utility)

        offset = 0 #17*np.pi/48

        # Upper bound
        s12_vector = QuantumAmbiguity(s1=s1_vector, s2=s2_vector, alpha=a, delta=0+offset)
        s12_operator = s12_vector.to_operator()
        quantum_upper_bound_operators.append(s12_operator.data)
        utility = s3_vector.expectation_value(s12_operator)
        utility *= s3/sb
        upper_bounds.append(abs(utility))

        # Lower bound
        s12_vector = QuantumAmbiguity(s1=s1_vector, s2=s2_vector, alpha=a, delta=np.pi-offset)
        s12_operator = s12_vector.to_operator()
        quantum_lower_bound_operators.append(s12_operator.data)
        utility = s3_vector.expectation_value(s12_operator)
        utility *= s3/sb
        lower_bounds.append(abs(utility))

        #for d in deltas:
        #    s12_vector = QuantumAmbiguity(s1=s1_vector, s2=s2_vector, alpha=a, delta=d)
        #    s12_operator = s12_vector.to_operator()
        #    utility = s3_vector.expectation_value(s12_operator)
        #    utility *= s3/sb
        #    quantum_utilities[d].append(abs(utility))

    cu_list.append(classical_utilities)
    ub_list.append(upper_bounds)
    lb_list.append(lower_bounds)



# Save the lists to files
#np.save('classical_operators.npy', classical_operators)
#np.save('quantum_upper_bound_operators.npy', quantum_upper_bound_operators)
#np.save('quantum_lower_bound_operators.npy', quantum_lower_bound_operators)
#np.savez(utility_function+'_'+str(gamma)+'tasks_3_4.npz', 
#         classical_operators=classical_operators, 
#         quantum_upper_bound_operators=quantum_upper_bound_operators, 
#         quantum_lower_bound_operators=quantum_lower_bound_operators,
#         benchmark_model=benchmark_model.data)

#for delta, list in quantum_utilities.items():
    #plt.plot(probs, list,'--', label = delta)
plt.plot(probs, cu_list[0], color='blue')
plt.plot(probs, ub_list[0], color = 'blue', linestyle='--')
plt.plot(probs, lb_list[0], color = 'blue', linestyle='--')

#plt.plot(probs, cu_list[1], label=f'sv \gamma_g = {gamma_G_1}')
#plt.plot(probs, cu_list[2], label=f'sv \gamma_g = {gamma_G_2}')
plt.plot(probs, cu_list[3], label='rd', color='red')
plt.plot(probs, ub_list[3], color = 'red', linestyle='--')
plt.plot(probs, lb_list[3], color = 'red', linestyle='--')
plt.text(probs[-1], cu_list[0][-1], r'$ \ \ Constant \ Volatility$', va='center', ha='left', color='black')
# plt.text(probs[-1], cu_list[1][-1], r'$ \ \ \pi_G = 0.8, \ \gamma_G = 0.3$', va='center', ha='left', color='black')
# plt.text(probs[-1], cu_list[2][-1], r'$ \ \ \pi_G = 0.95, \ \gamma_G = 0.01$', va='center', ha='left', color='black')
plt.text(probs[-1], cu_list[3][-1], r'$ \ \ Rare \ Disasters$', va='center', ha='left', color='black')


#plt.title('Utility Function: '+utility_function+" Gamma: "+str(gamma))

#plt.hlines(benchmark_utility, 0, 1, color = 'red')

i0 = min(range(len(ub_list[0])), key=lambda i: abs(lb_list[0][i] - ub_list[3][i]))
# i1 = min(range(len(lb_list[0])), key=lambda i: abs(ub_list[0][i] - benchmark_utility))
# i2 = min(range(len(lb_list[0])), key=lambda i: abs(lb_list[0][i] - benchmark_utility))

i0_prob, i0_value = probs[i0], lb_list[0][i0]
# i2_prob, i2_value = probs[i2], lb_list[0][i2]

print(i0_prob, i0_value)

plt.vlines(i0_prob, ymin=0, ymax = i0_value, linestyle='--', colors='black')
# plt.text(i0_prob, -1.1, f'p', ha='center', va='top')

# plt.vlines(i1_prob, ymin=0, ymax = i1_value, linestyle='--', colors='blue')
# plt.text(i1_prob, -1.1, f'p_L', ha='center', va='top')
# plt.vlines(i2_prob, ymin=0, ymax = i2_value, linestyle='--', colors='orange')
# plt.text(i2_prob, -1.1, f'p_U', ha='center', va='top')

#plt.text(probs[-1], cu_list[3][-1], r'$ \ \ (1-p)tr(P_d, P_t) + p tr(P_B, p_t)$', va='center', ha='left', color='black')


#plt.text(probs[-1], benchmark_utility, r'$ \ \ (1-p)tr(P_d, P_t) + p$', va='center', ha='left', color='black')
#plt.text(probs[-1], benchmark_utility, r'$ \ \ Benchmark$', va='center', ha='left', color='black')

plt.xlabel('Probability', fontsize=16)
plt.ylabel('Loss', fontsize=16)

plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_position('zero')
plt.gca().spines['left'].set_position('zero')
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

plt.xlim(right=1.5)
plt.ylim(bottom=0)
#plt.tight_layout()

plt.show()