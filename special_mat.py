import numpy as np


class MatrixA(object):
	'''
	Our special patterned matrix: A = Q + <Penalty Term> = bb^T + p ee^T + Diag(v)
	'''

	def __init__(self, bias: list, variance: list, penalty: float = 1000.):
		''' A = Q + <Penalty Term> = bb^T + p ee^T + Diag(v)

		:param bias: (m,)
		:param variance: V/n (m,)
		:param penalty: float
		'''
		# assert len(bias) == len(variance)
		# for vi in variance:
		# 	assert vi > 0  # vi == 0 makes the problem degenerate
		self.m = len(bias)  # number of strategies
		self.b = np.array(bias, dtype=np.float)  # row vec
		self.v = np.array(variance, dtype=np.float)  # row vec
		self.p = penalty  # float
		self.e = np.ones(self.m)  # row vec

	def mul(self, x: np.ndarray):
		'''mul vector, O(m)

		:param x: row vector (m,)
		:return: A.x (m,)
		'''
		return self.b * self.b.dot(x) + self.p * self.e * x.sum() + self.v * x

	def quad(self, x: np.ndarray):
		'''
		:param x: row vector (m,)
		:return: x^T.A.x (float)
		'''
		return self.b.dot(x) ** 2 + self.p * x.sum() ** 2 + (self.v * x ** 2).sum()

	def split(self, J):
		'''split the matrix into [[A_II, A_JI^T], [A_JI, A_JJ]]

		:param J: mask vector
		:return: MatrixA, A_JJ
		'''
		return MatrixA(self.b[J], self.v[J], self.p)
