import numpy as np, time

d_a = 101; # d_alpha
d_b = 102; # d_beta
d_c = 103; # d_gamma
d_d = 104; # d_delta
d_m = 105; # d_mu

A = np.random.rand(d_c,d_d); # tensor A(gamma,delta)
B = np.random.rand(d_a,d_m,d_c); # tensor B(alpha,mu,gamma)
C = np.random.rand(d_b,d_m,d_d); # tensor C(beta,mu,delta)

now = time.time()

B1 = B.transpose((0, 2, 1)).reshape((d_a*d_c,d_m))
C1 = C.transpose((1, 0, 2)).reshape((d_m,d_b*d_d))
BC = B1 @ C1
BC = BC.reshape((d_a,d_c,d_b,d_d))
BC = BC.transpose((0, 2, 1, 3)).reshape((d_a*d_b,d_c*d_d))
A1 = A.flatten()

ABC1 = BC @ A1
ABC1 = ABC1.reshape((d_a,d_b))

print(time.time() - now) # 0.5828

now = time.time()
np.einsum('cd,amc,bmd->ab', A, B, C, optimize=True)
print(time.time() - now) # 0.0247

now = time.time()
np.einsum('cd,amc,bmd->ab', A, B, C, optimize=False)
print(time.time() - now) # 20.24