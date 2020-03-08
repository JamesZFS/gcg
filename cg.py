import time

from special_mat import *

DEFAULT_EPS = 1e-5


def conjugate_gradient(A: MatrixA, b: np.ndarray, x0: np.ndarray, eps=DEFAULT_EPS):
	'''solve A x = b using conjugate gradient

	:param A: MatrixA
	:param b: row vec
	:param x0: initial x, row vec
	:return: solution(row vec) and n_iteration
	'''
	r = b - A.mul(x0)  # residue
	rdr = r.dot(r)
	p = r  # explore direction
	k = 0  # n_iter
	max_iter = len(b)
	x = x0
	while k < max_iter:
		Ap = A.mul(p)
		alpha = rdr / p.dot(Ap)  # ak = <rk, rk> / <pk, A pk>
		x = x + alpha * p
		k += 1
		r = r - alpha * Ap  # rk+1
		rdr_ = rdr  # rk dot rk
		rdr = r.dot(r)  # rk+1 dot rk+1
		if rdr < eps: break  # if r is sufficiently small
		beta = rdr / rdr_
		p = r + beta * p  # pk+1
	return x, k


def conjgrad_ref(A, b, x, eps=DEFAULT_EPS):
	r = b - A @ x
	p = r
	rsold = r.dot(r)
	k = 0
	for i in range(len(b)):
		Ap = A @ p
		alpha = rsold / p.dot(Ap)
		x = x + alpha * p
		r = r - alpha * Ap
		rsnew = r.dot(r)
		if rsnew < eps:
			k = i + 1
			break
		p = r + (rsnew / rsold) * p
		rsold = rsnew
	return x, k


if __name__ == '__main__':
	m = 2048
	n = 32
	print('m=', m)
	V = np.abs(np.random.normal(0, 10, m)) / n
	b = np.random.random(m)
	b = b - V
	p = 100
	b[0] = 0
	Q = np.diag(V) + np.outer(b, b) + p * np.ones((m, m))
	A = MatrixA(b, V, p)
	B = p * np.ones(m)
	x0 = np.random.randn(m)
	tic = time.time()
	x1 = np.linalg.solve(Q, B)
	print('linsolve:', time.time() - tic, 'sec')
	tic = time.time()
	for i in range(10):
		x2, k = conjugate_gradient(A, B, x0)
	print('our cg:  ', time.time() - tic, 'sec', 'iter=', k, 'diff=', np.linalg.norm(x2 - x1))
	tic = time.time()
	for i in range(10):
		x3, k = conjgrad_ref(Q, B, x0)
	print('ref cg:  ', time.time() - tic, 'sec', 'iter=', k, 'diff=', np.linalg.norm(x3 - x1))