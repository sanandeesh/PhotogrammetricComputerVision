"""
    Compute Jacobians for ...
        (1) `State-Transition Model` wrt. motion parameters (translational + rotational velocity)
        (2) `Measurement Model` wrt. state parameters (X, Y, Z) 
    (1) is for Process Noise in Prediction-Step, and (2) is for Measurement-Correction-Step

Usage: python3 vitekf_compute_jacobians.py
Requirements: sympy
"""
import sympy as sp
from sympy import MatMul

"""
I.
--------------------------------------------------------------------------------------------------------------------
------------------------------------------ State Transition Model --------------------------------------------------- 

     ⎡1.0      0.0            0.0     ⎤ ⎡cos(δ_yaw)   0.0  sin(δ_yaw)⎤ ⎡ -δₓ + μₓ ⎤
     ⎢                                ⎥ ⎢                            ⎥ ⎢          ⎥
F =  ⎢0.0  cos(δ_pitch)  -sin(δ_pitch)⎥⋅⎢    0.0      1.0     0.0    ⎥⋅⎢   μ_y    ⎥
     ⎢                                ⎥ ⎢                            ⎥ ⎢          ⎥
     ⎣0.0  sin(δ_pitch)  cos(δ_pitch) ⎦ ⎣-sin(δ_yaw)  0.0  cos(δ_yaw)⎦ ⎣-δ_z + μ_z⎦

     Is a function of (1) State and (2) Control Inputs

------------ The Jacobian of the State Transition Model wrt "Control Inputs", `δ_x, δ_z, δ_yaw, δ_pitch`------------ 
Original Jacobian wrt Control Params:
     ⎡    -1.0⋅cos(δ_yaw)           -1.0⋅sin(δ_yaw)                -1.0⋅(μₓ - δₓ)⋅sin(δ_yaw) + 1.0⋅(μ_z - δ_z)⋅cos(δ_yaw)                                                           0                                              ⎤
     ⎢                                                                                                                                                                                                                             ⎥
 J = ⎢-sin(δ_pitch)⋅sin(δ_yaw)  sin(δ_pitch)⋅cos(δ_yaw)   (μₓ - δₓ)⋅sin(δ_pitch)⋅cos(δ_yaw) + (μ_z - δ_z)⋅sin(δ_pitch)⋅sin(δ_yaw)   -1.0⋅μ_y⋅sin(δ_pitch) + (μₓ - δₓ)⋅sin(δ_yaw)⋅cos(δ_pitch) - (μ_z - δ_z)⋅cos(δ_pitch)⋅cos(δ_yaw)⎥ 
     ⎢                                                                                                                                                                                                                             ⎥
     ⎣sin(δ_yaw)⋅cos(δ_pitch)   -cos(δ_pitch)⋅cos(δ_yaw)  -(μₓ - δₓ)⋅cos(δ_pitch)⋅cos(δ_yaw) - (μ_z - δ_z)⋅sin(δ_yaw)⋅cos(δ_pitch)  1.0⋅μ_y⋅cos(δ_pitch) + (μₓ - δₓ)⋅sin(δ_pitch)⋅sin(δ_yaw) - (μ_z - δ_z)⋅sin(δ_pitch)⋅cos(δ_yaw) ⎦

ASSUMING (δ_yaw == δ_pitch == 0.0):
     ⎡-1.0  0   -1.0⋅δ_z + 1.0⋅μ_z      0    ⎤
     ⎢                                       ⎥
 J = ⎢ 0    0           0           δ_z - μ_z⎥
     ⎢                                       ⎥
     ⎣ 0    -1       δₓ - μₓ         1.0⋅μ_y ⎦

     Hence, process noise covariance Q is defined as
    Q = J Sigma_u J.T
    where `Sigma_u` is covariance matrix of Control Inputs.
"""
# 1. Define all necessary symbols
delta_x, delta_z, delta_yaw, delta_pitch = sp.symbols('delta_x delta_z delta_yaw delta_pitch')
mu_x, mu_y, mu_z = sp.symbols('mu_x mu_y mu_z')

# 2. Define the symbolic matrices as provided
R_x = sp.Matrix([[1.0,                    0.0,                   0.0],
                 [0.0,  sp.cos(delta_pitch), -sp.sin(delta_pitch)],
                 [0.0,  sp.sin(delta_pitch),  sp.cos(delta_pitch)]])

R_y = sp.Matrix([[ sp.cos(delta_yaw), 0.0, sp.sin(delta_yaw)],
                 [              0.0, 1.0,              0.0],
                 [-sp.sin(delta_yaw), 0.0, sp.cos(delta_yaw)]])

mu = sp.Matrix([[mu_x],
                [mu_y],
                [mu_z]])

delta_shift = sp.Matrix([[delta_x],
                         [0.0],
                         [delta_z]])

# 3. Compute 'mu_predicted' symbolically
mu_predicted = R_x * R_y * (mu - delta_shift)

# 4. Compute the Jacobian matrix 
# The variables are passed as a list in the order specified
variables = [delta_x, delta_z, delta_yaw, delta_pitch]
jacobian = mu_predicted.jacobian(variables)

# 5. Simplify the Jacobian matrix assuming delta_yaw = 0 and delta_pitch = 0
simplified_jacobian = jacobian.subs({delta_yaw: 0, delta_pitch: 0})


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
sp.init_printing()
print("State Transition Model:")
mu_predicted_unevaluated = MatMul(R_x, R_y, (mu - delta_shift), evaluate=False)
sp.pprint(mu_predicted_unevaluated)
print("--------------------------------------------------")
print("Original Jacobian wrt Control Params:")
sp.pprint(jacobian)
print("\nSimplified Jacobian (where delta_yaw and delta_pitch are zero):")
sp.pprint(simplified_jacobian)
print('\n')
print('===================================================================================')
print('\n')

"""
II.
--------------------------------------------------------------------------------------------------------------------
------------------------------------------ Measurement Model --------------------------------------------------- 
Measurement Model: Non-Linear Normalized Camera Projection
⎡fₓ        cₓ ⎤      
⎢───  0.0  ───⎥ ⎡μₓ ⎤
⎢μ_z       μ_z⎥ ⎢   ⎥
⎢             ⎥⋅⎢μ_y⎥
⎢     f_y  c_y⎥ ⎢   ⎥
⎢0.0  ───  ───⎥ ⎣μ_z⎦
⎣     μ_z  μ_z⎦      
--------------------------------------------------
Original Jacobian wrt State:
⎡fₓ         -fₓ⋅μₓ  ⎤
⎢───   0    ─────── ⎥
⎢μ_z            2   ⎥
⎢            μ_z    ⎥
⎢                   ⎥
⎢     f_y  -f_y⋅μ_y ⎥
⎢ 0   ───  ─────────⎥
⎢     μ_z       2   ⎥
⎣            μ_z    ⎦
--------------------------------------------------------------------------------------------------------------------
"""
# 1. Define all necessary symbols
mu_x, mu_y, mu_z = sp.symbols('mu_x mu_y mu_z')
f_x, f_y, c_x, c_y = sp.symbols('f_x f_y c_x c_y ')

# 2. Define the symbolic matrices as provided
P = sp.Matrix([[f_x/mu_z,      0.0, c_x/mu_z],
               [     0.0, f_y/mu_z, c_y/mu_z]])

mu = sp.Matrix([[mu_x],
                [mu_y],
                [mu_z]])

# 3. Compute 'mu_predicted' symbolically
h = P * mu

# 4. Compute the Jacobian matrix 
# The variables are passed as a list in the order specified
variables = [mu_x, mu_y, mu_z]
jacobian = h.jacobian(variables)


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
sp.init_printing()
print("Measurement Model: Non-Linear Normalized Camera Projection")
h_unevaluated = MatMul(P, mu, evaluate=False)
sp.pprint(h_unevaluated)
print("--------------------------------------------------")
print("Original Jacobian wrt State:")
sp.pprint(jacobian)


# ----------------------------------------------------------
def ComputeProcessNoiseJacobian():
    # 1. Define all necessary symbols
    delta_x, delta_z, delta_yaw, delta_pitch = sp.symbols('delta_x delta_z delta_yaw delta_pitch')
    mu_x, mu_y, mu_z = sp.symbols('mu_x mu_y mu_z')

    # 2. Define the symbolic matrices as provided
    R_x = sp.Matrix([[1.0,                    0.0,                   0.0],
                     [0.0,  sp.cos(delta_pitch), -sp.sin(delta_pitch)],
                     [0.0,  sp.sin(delta_pitch),  sp.cos(delta_pitch)]])

    R_y = sp.Matrix([[ sp.cos(delta_yaw), 0.0, sp.sin(delta_yaw)],
                     [              0.0, 1.0,              0.0],
                     [-sp.sin(delta_yaw), 0.0, sp.cos(delta_yaw)]])

    mu = sp.Matrix([[mu_x],
                    [mu_y],
                    [mu_z]])

    delta_shift = sp.Matrix([[delta_x],
                             [0.0],
                             [delta_z]])

    # 3. Compute 'mu_predicted' symbolically
    mu_predicted = R_x * R_y * (mu - delta_shift)

    # 4. Compute the Jacobian matrix 
    # The variables are passed as a list in the order specified
    variables = [delta_x, delta_z, delta_yaw, delta_pitch]
    jacobian_Q = mu_predicted.jacobian(variables)

    # 1. List all input arguments in your desired order
    all_args = [delta_x, delta_z, delta_yaw, delta_pitch, mu_x, mu_y, mu_z]

    # 2. Compile the symbolic matrix into a fast NumPy function
    jacobian_func = sp.lambdify(all_args, jacobian_Q, modules='numpy')

    return jacobian_func
