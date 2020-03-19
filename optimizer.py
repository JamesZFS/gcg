import cvxpy as cp
import numpy as np

from cg import conjugate_gradient
from gcg import generalized_conjugate_gradient
from special_mat import MatrixA


def init_trivial(b, V, penalty, A=None) -> np.ndarray:
	m = len(b)
	return np.ones(m) / m


def init_bias_var(b, V, penalty, A=None) -> np.ndarray:
	x0 = 1 / (np.array(b) ** 2 + np.array(V))
	return x0 / x0.sum()


def init_no_constr(b, V, penalty, A) -> np.ndarray:
	x0, n_iter = conjugate_gradient(A, penalty, init_bias_var(b, V, penalty), eps_zero=1e-5)
	# print('init with cg, iters=', n_iter)
	x0[x0 < 0] = 0
	return x0 / x0.sum()


def opt_OSQP(b: list, V: list, **kwargs):
	m = len(V)
	b = np.array(b)
	Q = np.diag(V) + np.outer(b, b)
	cond = np.linalg.cond(Q)
	print('cond(Q)=', cond)
	W = cp.Variable((m, 1), "optimal weights")
	objective = cp.quad_form(W, Q)  # W^T Q W
	constraints = [
		W >= 0,
		cp.sum(W) == 1
	]
	prob = cp.Problem(cp.Minimize(objective), constraints)
	print('solving...')
	prob.solve(solver=cp.OSQP, **kwargs)
	W = (W.value.T)[0]
	# kick out negative values and re-normalize
	W[W < 0] = 0
	W = W / np.sum(W)
	return W


def opt_GCG(b: list, V: list, penalty=1, init_method=init_bias_var, ret_steps=False, **kwargs):
	A = MatrixA(b, V, penalty)
	x0 = init_method(b, V, penalty, A)
	if kwargs.get('max_iter', np.inf) == 0: return x0
	result = generalized_conjugate_gradient(A, penalty, x0, ret_steps, **kwargs)
	if ret_steps == True:
		W, n_outer, n_step = result
		W = W / np.sum(W)
		return W, n_outer, n_step
	else:
		W = result / np.sum(result)
		return W
