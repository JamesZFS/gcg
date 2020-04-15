import time

from matplotlib import pyplot as plt

from optimizer import *


def simple_test(method, **kwargs):
	V = [1, 1, 1, 2, 3]
	b = [0, 10, 10, 10, 2]
	# alpha = 1000
	n = 32
	V = [v / n for v in V]
	W = np.array(method(b, V, **kwargs))
	W_ref = np.array([0.992424, 9.66144e-9, 9.66144e-9, 9.66144e-9, 0.00757565])
	assert np.linalg.norm(W - W_ref) <= 1e-5
	print('\033[32m[ pass simple test ]\033[0m')


def massive_test(method, **kwargs):
	m = 4096
	n = 32
	np.random.seed(1)
	V = np.abs(np.random.normal(0, 10, m)) / n
	b = np.random.random(m)
	b[0] = 0
	# alpha = 1000
	tic = time.time()
	W = np.array(method(b, V, **kwargs))
	toc = time.time()
	W_ref = np.load('massive_test_ref.npy')
	err = np.linalg.norm(W - W_ref, ord=1)
	# OSQP with eps_rel=1e-2 err=0.022
	print('first 20:', W[:20])
	print('\033[32m[ pass massive test, \033[31merr = %f\033[32m, elapse = %f sec ]\033[0m' % (err, toc - tic))
	err = np.linalg.norm(np.ones(m) / m - W_ref, ord=1)
	print('max err=', err)
	plt.plot(W_ref, color='red', alpha=0.3)
	plt.plot(W, color='blue', alpha=0.3)
	plt.xlabel('i')
	plt.ylabel('w[i]')
	plt.legend(['ref', 'ours'])
	plt.show()


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


def compare_two(m=128, plot=False):
	print(f'm = {m}')
	n = 32
	np.random.seed(10)
	V = np.abs(np.random.normal(0, 1, m)) / n
	b = np.random.random(m)
	# b[m / 2] = 0
	penalty = 1
	eps = 1e-10

	tic = time.time()
	W1 = opt_OSQP(b, V, max_iter=100_000, verbose=False)
	t1 = time.time() - tic

	tic = time.time()
	W2, n_outer_tv, n_step_tv = opt_GCG(b, V, penalty, ret_steps=True, eps=eps)  # ours
	t2 = time.time() - tic

	tic = time.time()
	W3, n_outer_bv, n_step_bv = opt_GCG(b, V, penalty, init_method=init_bias_var, ret_steps=True, eps=eps)  # ours
	t3 = time.time() - tic

	tic = time.time()
	W4, n_outer_cg, n_step_cg = opt_GCG(b, V, penalty, init_method=init_no_constr, ret_steps=True, eps=eps)  # ours
	t4 = time.time() - tic

	print('std: ', t1, 'sec')
	print('ours:', t2, 'sec', '  outers=', n_outer_tv, '  steps=', n_step_tv, '  \tdiff=',
		  np.linalg.norm(W1 - W2, ord=1), '  accl=', t1 / t2, '  trivial init')
	print('ours:', t3, 'sec', '  outers=', n_outer_bv, '  steps=', n_step_bv, '  \tdiff=',
		  np.linalg.norm(W1 - W3, ord=1), '  accl=', t1 / t3, '  bias-variance init')
	print('\033[32mours:', t4, 'sec', '  outers=', n_outer_cg, '  steps=', n_step_cg, '  \tdiff=',
		  np.linalg.norm(W1 - W4, ord=1), '  accl=', t1 / t4, '  no_constr+cg init\033[0m')

	if plot:
		plt.plot(W1, color='red', alpha=0.3)
		plt.plot(W2, color='blue', alpha=0.3)
		plt.plot(W3, color='yellow', alpha=0.3)
		plt.xlabel('i')
		plt.ylabel('w[i]')
		plt.legend(['std', 'ours1, ours2'])
		plt.ylim((0, 0.2))
		plt.show()

	# save data:
	np.savetxt('mat-data/variance.csv', V, delimiter=',')
	np.savetxt('mat-data/bias.csv', b, delimiter=',')
	np.savetxt('mat-data/weight.csv', W3, delimiter=',')
	pass


# simple_test(opt_GCG, eps=1e-20)
# massive_test(opt_GCG, penalty=1e2, eps=1e-10, eps_zero=1e-10)
# show_form_benefit(iter=300)
compare_two(m=128)
