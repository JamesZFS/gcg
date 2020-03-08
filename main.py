import time

import cvxpy as cp
import numpy as np

from special_mat import MatrixA
from cg import conjugate_gradient
from gcg import generalized_conjugate_gradient, DEFAULT_EPS


def opt_OSQP(b: list, V: list):
	assert b[0] == 0
	m = len(V)
	b = np.array(b)
	Q = np.diag(V) + np.outer(b, b)
	W = cp.Variable((m, 1), "optimal weights")
	objective = cp.quad_form(W, Q)  # W^T Q W
	constraints = [
		W >= 0,
		cp.sum(W) == 1
	]
	prob = cp.Problem(cp.Minimize(objective), constraints)
	print('solving...')
	prob.solve(solver=cp.OSQP, eps_rel=1e-3, verbose=False)
	W = (W.value.T)[0]
	# kick out negative values and re-normalize
	W[W < 0] = 0
	W = W / np.sum(W)
	return W


def opt_GCG(b: list, V: list, penalty: float = 1e3, eps: float = DEFAULT_EPS):
	A = MatrixA(b, V, penalty)
	m = len(b)
	x0 = np.ones(m) / m
	return generalized_conjugate_gradient(A, penalty, x0, eps)


def simple_test(method):
	V = [1, 1, 1, 2, 3]
	b = [0, 10, 10, 10, 2]
	# alpha = 1000
	n = 32
	W = np.array(method(b, V, n))
	W_ref = np.array([0.992424, 9.66144e-9, 9.66144e-9, 9.66144e-9, 0.00757565])
	assert np.linalg.norm(W - W_ref, ord=2) <= 1e-6
	print('\033[32m[ pass simple test ]\033[0m')


def massive_test(method):
	m = 4096
	np.random.seed(1)
	V = np.abs(np.random.normal(0, 10, m))
	b = np.random.random(m)
	b[0] = 0
	# alpha = 1000
	n = 32
	tic = time.time()
	W = np.array(method(b, V, n))
	toc = time.time()
	W_ref = np.load('massive_test_ref.npy')
	err = np.linalg.norm(W - W_ref)
	print('\033[32m[ pass massive test, \033[31merr = %f\033[32m, elapse = %f sec ]\033[0m' % (err, toc - tic))


# ! This below test proves that exploiting the form(sparsity pattern) of matrix Q can accelerate the quadratic computing around 100 times!
def show_form_benefit(seed=time.time_ns() % 1000, iter=200):
	m = 4096
	n = 32
	np.random.seed(seed)
	V = np.abs(np.random.normal(0, 10, m)) / n
	b = np.random.random(m)
	b[0] = 0
	Q = np.diag(V) + np.outer(b, b)
	res1, res2, res3 = [], [], []

	# O(n^2)
	tic = time.time()
	np.random.seed(seed)
	for i in range(iter):
		x = np.array([np.random.random(m)])
		y = (x @ Q @ x.T).item()
		res1.append(y)
	toc = time.time()
	t1 = toc - tic

	# O(n)
	tic = time.time()
	np.random.seed(seed)
	for i in range(iter):
		x = np.random.random(m)
		y = np.sum(x ** 2 * V) + np.dot(b, x) ** 2
		res2.append(y)
	toc = time.time()
	t2 = toc - tic

	# O(n)
	A = MatrixA(b, V, 0)
	tic = time.time()
	np.random.seed(seed)
	for i in range(iter):
		x = np.random.random(m)
		y = A.quad(x)
		res3.append(y)
	toc = time.time()
	t3 = toc - tic

	diff2 = np.linalg.norm(np.subtract(res2, res1))
	diff3 = np.linalg.norm(np.subtract(res3, res1))
	print('diff2-1: ', diff2)
	print('diff3-1: ', diff3)
	print('time: %.4f sec : %.4f sec : %.4f, improve = %.2f, %.2f' % (t1, t2, t3, t1 / t2, t1 / t3))
	assert diff2 < 1e-3
	assert diff3 < 1e-3


# simple_test(opt1)
# massive_test(opt1)
# show_form_benefit(iter=300)
m = 100
n = 32
np.random.seed(1)
V = np.abs(np.random.normal(0, 10, m)) / n
b = np.random.random(m)
b[0] = 0

tic = time.time()
W1 = opt_OSQP(b, V)
t1 = time.time() - tic
tic = time.time()
W2 = opt_GCG(b, V)
t2 = time.time() - tic
print('std: ', t1, 'sec')
print('ours:', t2, 'sec')
print('diff=', np.linalg.norm(W1 - W2))
