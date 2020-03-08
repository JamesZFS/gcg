# The generalized conjugate gradient algorithm for solving quadratic programming
# courtesy of Dianne Prost O'Leary et.al.

from special_mat import *

DEFAULT_EPS = 1e-5
DEFAULT_EPS_ZERO = 1e-5


def generalized_conjugate_gradient(A: MatrixA, b: float, x0: np.ndarray, eps=DEFAULT_EPS, eps_zero=DEFAULT_EPS_ZERO):
	'''The generalized conjugate gradient algorithm for solving quadratic programming:
	min 1/2 x^T A x - b x  s.t. x_i >= 0

	:param A: our special matrix (m, m)
	:param b: denotes vec b, float
	:param x0: initial x (>=0) (m,)
	:return: argmin x (m,)
	'''
	assert x0.shape == A.b.shape
	assert (x0 >= 0).all()
	# Initialization:
	m = A.m
	k = 0  # n_iter_outer
	n_step = 0
	# I = np.arange(0, m)
	I = np.ones(m, dtype=bool)  # I = {i: xi = 0 and yi > 0} boundary set
	x = x0  # x_k

	# Outer Iteration:
	while True:
		# print('\033[32mnew outer iter k = %d\033[0m' % k)
		k += 1
		y = A.mul(x) - b  # y_k
		I_ = I  # I_(k-1)
		I = (x < eps_zero) & (y > 0)  # xi==0 and yi>0
		J = ~I  # interior set
		# print('I_=', I_, 'I=', I)
		if (I == I_).all(): break  # return x

		# Inner Iteration:
		while True:
			# print('\033[34m  restart inner iter\033[0m')
			# (a) Partition and rearrange A, b:
			x_J = x[J]
			A_JJ = A.split(J)
			# b_J = b

			z: np.ndarray = x_J  # z(0), z will be the approx solution to x_J
			r: np.ndarray = b - A_JJ.mul(z)  # r(0), (x_I == 0)
			p: np.ndarray = r  # p(0)

			restart_outer: bool
			q = 0  # n_iter_inner
			while True:
				# print('    inner iter q = %d' % q)
				# (b) Calc new iterate and residue:
				A_JJ_p = A_JJ.mul(p)  # A_JJ @ p(q)
				rdr = r.dot(r)  # r(q) dot r(q)
				a_cg = rdr / p.dot(A_JJ_p)
				p_neg_pos = p < 0
				p_neg = p[p_neg_pos]
				# max step that doesn't violate x >= 0:
				a_max = np.min(-z[p_neg_pos] / p_neg) if len(p_neg) > 0 else np.inf
				alpha = min(a_cg, a_max)  # step size a(q)
				# step z:
				z = z + alpha * p  # z(q+1)
				n_step += 1
				r = r - alpha * A_JJ_p  # r(q+1)
				rdr_ = rdr  # r(q) dot r(q)
				rdr = r.dot(r)  # r(q+1) dot r(q+1)

				# (c) Test for termination of inner iteration:
				# print('    rdr = %f' % rdr)
				if rdr < eps:
					x_J = z
					x[J] = x_J  # reconstruct x_k
					restart_outer = True
					break  # goto (e)
				if (z < eps_zero).any():  # {i: z(q+1)_i == 0} is empty, I doesn't change
					# some x_J should be added to boundary set I
					x[J] = z  # reconstruct x_k
					I = x < eps_zero  # we always have I_k > I_(k-1)
					J = ~I
					# if I.all():  # I == {0,1,..,m-1}
					# 	print('\033[31mwarning: all x == 0!\033[0m')
					# 	restart_outer = True
					# else:
					restart_outer = False  # restart inner
					break  # goto (e)
				# else: boundary set does not expand
				# (d) Calc new search direction
				beta = rdr / rdr_
				p = r + beta * p  # p(q+1)
				q += 1  # goto (b)

			# Loop (b)
			if restart_outer: break  # (e)
		# Inner Loop
		pass
	# Outer Loop
	print('k =', k, 'n_step =', n_step)
	return x
