from threading import Thread
from PIL import Image
from tqdm import tqdm

from optimizer import *
from multiprocessing import cpu_count
from matplotlib import pyplot as plt

EPSILON = 1e-5


def check_shape(img: np.ndarray) -> np.ndarray:
	assert isinstance(img, np.ndarray)
	assert img.ndim == 3
	if img.shape[2] == 4:
		img = img[:, :, :3]  # ignore alpha channel
	assert img.shape[2] == 3
	return img


def get_todo_list(amount: int, n_job: int) -> list:
	piece = amount / n_job
	return [range(int(piece * i), int(piece * (i + 1))) for i in range(n_job)]


def denoise(im_mean: np.ndarray, im_var: np.ndarray, im_original: np.ndarray, filter_size: tuple,
			**kwargs) -> np.ndarray:
	'''Compute optimal filter and denoise in place

	:return: denoised image
	'''
	im_mean, im_var, im_original = check_shape(im_mean), check_shape(im_var), check_shape(im_original)
	assert im_mean.shape == im_var.shape == im_original.shape
	n_row, n_col = im_mean.shape[0:2]
	f_r, f_c = filter_size[0:2]
	if not (f_r & 1 and f_c & 1):
		raise ValueError('Bad filter size, expect odd size!')
	f_r_2, f_c_2 = int(f_r / 2), int(f_c / 2)
	im_result = np.zeros_like(im_mean)
	progress = tqdm(total=n_row * n_col, desc='denoising', postfix=dict(current=''))

	class Job(Thread):
		def __init__(self, rows_todo):
			super().__init__()
			self.rows_todo = rows_todo

		def run(self):
			print(f'hello from thread {self.ident}, todo: {self.rows_todo}')
			for i_r in self.rows_todo:
				for i_c in range(n_col):
					progress.set_postfix(current=f'({i_r} / {n_row}, {i_c} / {n_col})')
					progress.update()
					if (im_var[i_r, i_c] < EPSILON).any():  # very smooth
						im_result[i_r, i_c] = im_original[i_r, i_c]  # inherit from original image
						continue
					# filter pixel i using neighbors j:
					j_r_begin = max(0, i_r - f_r_2)
					j_r_end = min(n_row, i_r + f_r_2 + 1)
					j_c_begin = max(0, i_c - f_c_2)
					j_c_end = min(n_col, i_c + f_c_2 + 1)
					b_multi = im_mean[j_r_begin: j_r_end, j_c_begin: j_c_end, :] - im_mean[i_r, i_c, :]
					v_multi = im_var[j_r_begin: j_r_end, j_c_begin: j_c_end, :]
					o_multi = im_original[j_r_begin: j_r_end, j_c_begin: j_c_end, :]
					# compute filter for each channel:
					for channel in range(3):
						b = b_multi[:, :, channel].flatten()
						v = v_multi[:, :, channel].flatten() + EPSILON
						o = o_multi[:, :, channel].flatten()
						assert (v > 0).all()
						w = opt_GCG(b, v, ret_steps=False, **kwargs)
						im_result[i_r, i_c, channel] = o.dot(w)  # apply filter on original image

	jobs = [Job(rows_todo) for rows_todo in get_todo_list(n_row, cpu_count())]
	for job in jobs:
		job.start()
	for job in jobs:
		job.join()
	return im_result


def compute_filter(im_mean, im_var, filter_size: tuple, at: tuple, **kwargs):
	im_mean, im_var = check_shape(im_mean), check_shape(im_var)
	assert im_mean.shape == im_var.shape
	n_row, n_col = im_mean.shape[0:2]
	f_r, f_c = filter_size[0:2]
	if not (f_r & 1 and f_c & 1):
		raise ValueError('Bad filter size, expect odd size!')
	f_r_2, f_c_2 = int(f_r / 2), int(f_c / 2)
	i_r, i_c = at
	# filter pixel i using neighbors j:
	j_r_begin = max(0, i_r - f_r_2)
	j_r_end = min(n_row, i_r + f_r_2 + 1)
	j_c_begin = max(0, i_c - f_c_2)
	j_c_end = min(n_col, i_c + f_c_2 + 1)
	b_multi = im_mean[j_r_begin: j_r_end, j_c_begin: j_c_end, :] - im_mean[i_r, i_c, :]
	v_multi = im_var[j_r_begin: j_r_end, j_c_begin: j_c_end, :]
	w_multi = np.zeros_like(b_multi)
	# compute filter for each channel:
	for channel in range(3):
		b = b_multi[:, :, channel].flatten()
		v = v_multi[:, :, channel].flatten() + EPSILON
		assert (v > 0).all()
		w = opt_GCG(b, v, ret_steps=False, **kwargs)
		w_multi[:, :, channel] = w.reshape((j_r_end - j_r_begin, j_c_end - j_c_begin))
	return w_multi


def imread(path) -> np.ndarray:
	return np.array(Image.open(path), dtype='float64') / 255.


def imwrite(img: np.ndarray, path):
	img[img > 1.0] = 1.0
	Image.fromarray((255 * img).astype('uint8')).save(path)


def imshow(img: np.ndarray, title=None):
	img[img > 1.0] = 1.0
	Image.fromarray((255 * img).astype('uint8')).show(title)


def imdiff(img1, img2):
	diff_img = img1 - img2
	diff = diff_img.mean()
	return diff_img, diff


def show_diff(img1, img2):
	diff_img = img1 - img2
	diff = diff_img.mean()
	imshow(diff_img)
	print(f'mean difference = {diff}')
	return diff


def visualize_weight(im_mean, im_var, im_background, filter_size: tuple, at: tuple, gamma=0.1, alpha=1.0,
					 in_place=False, **kwargs):
	weight = compute_filter(im_mean, im_var, filter_size, at, **kwargs).mean(axis=2) + EPSILON
	weight = weight ** gamma
	color_map = plt.get_cmap('afmhot')
	colored_weight = check_shape(color_map(weight))

	n_row, n_col = im_mean.shape[0:2]
	f_r_2, f_c_2 = int(filter_size[0] / 2), int(filter_size[1] / 2)
	i_r, i_c = at
	j_r_begin = max(0, i_r - f_r_2)
	j_r_end = min(n_row, i_r + f_r_2 + 1)
	j_c_begin = max(0, i_c - f_c_2)
	j_c_end = min(n_col, i_c + f_c_2 + 1)
	im = im_background if in_place else im_background.copy()
	im[j_r_begin:j_r_end, j_c_begin:j_c_end] = \
		alpha * colored_weight + (1 - alpha) * im[j_r_begin:j_r_end, j_c_begin:j_c_end]
	return im


# small test:
it = 100
im_mean = imread('data/16mean.png')[200:300, 300:450, :3]
im_var = imread('data/16variance.png')[200:300, 300:450, :3] / 16
im_original = imread('data/16mean.png')[200:300, 300:450, :3]
imwrite(im_mean, 'out/bias/org.png')
imwrite(im_var, 'out/bias/var.png')
im_result = denoise(im_mean, im_var, im_original, (63, 63), max_iter=it)
imwrite(im_result, f'out/bias/it={it}.result.png')
im_diff, diff = imdiff(im_result, im_original)
imwrite(im_diff, f'out/bias/it={it}.diff.png')
with open(f'out/bias/it={it}.diff.txt', 'w') as f:
	f.write(str(diff))

# massive test:
# im_mean = imread('data/images-png-part1/16spp/mean_cross_filtered_iter2.png')
# im_var = imread('data/images-png-part1/16spp/variance_cross_filtered_iter2.png') / 16
# im_original = check_shape(imread('data/16mean.png'))
# visualize filter weight:
# points = [(230, 520), (358, 367), (446, 552), (470, 181), (386, 181), (310, 636), (552, 826)]
# for point in points:
# 	visualize_weight(im_mean, im_var, im_original, (63, 63), at=point, alpha=1, gamma=0.1, in_place=True, max_iter=4000)
# imwrite(im_original, 'out/full 63x63/weight_iter_4000.png')
# draw weights on im_original

# imwrite(im_original, 'out/full 63x63/16original.png')
# imwrite(im_mean, 'out/full 63x63/mean_cross_filtered_iter2.png')
# im_result = denoise(im_mean, im_var, im_original, (63, 63), max_iter=0)
# imwrite(im_result, 'out/full 63x63/iter_0_init_bias_var_guided_by_cross++++.png')

# diff
# imwrite(imread('out/full 63x63/16reference.png')[:,:,0:3] - imread('out/full 63x63/iter_0_init_bias_var.png'), 'out/full 63x63/diff.png')
# im_org = imread('out/small 63x63/16original.png')[:, :, :3]
# im_our = imread('out/small 63x63/filtered_iter_0.png')[:, :, :3]
# show_diff(im_org, im_our)
# im_our = imread('out/small 63x63/filtered_iter_10.png')[:, :, :3]
# show_diff(im_org, im_our)

# im = Image.open('data/16mean.png')
# since = time.time()
# N = 100
# for i in range(N):
# 	im = im.filter(ImageFilter.BLUR)
# print(f'{(time.time() - since) / N} sec')
# im.save('out/blur_ref.png')

# im = imread('data/16mean.png')
# imwrite(im, 'data/16mean2.png')
# ! (64 / 100, 9 / 150)
