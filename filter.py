from optimizer import *
from PIL import Image, ImageFilter
from tqdm import tqdm

EPSILON = 1e-5


def check_shape(img: np.ndarray) -> np.ndarray:
	assert isinstance(img, np.ndarray)
	assert img.ndim == 3
	if img.shape[2] == 4:
		img = img[:, :, :3]  # ignore alpha channel
	assert img.shape[2] == 3
	return img


def denoise(im_mean: np.ndarray, im_var: np.ndarray, im_original: np.ndarray, filter_size: tuple,
			**kwargs) -> np.ndarray:
	'''Compute optimal filter and denoise in place

	:return: denoised image
	'''
	im_mean, im_var, im_original = check_shape(im_mean), check_shape(im_var), check_shape(im_original)
	assert im_mean.shape == im_var.shape == im_original.shape
	n_row, n_col = im_mean.shape[0:2]
	f_r, f_c = filter_size[0:2]
	f_r_2, f_c_2 = int(f_r / 2), int(f_c / 2)
	if not (f_r & 1 and f_c & 1):
		raise ValueError('Bad filter size, expect odd size!')
	im_result = np.zeros_like(im_mean)
	progress = tqdm(total=n_row * n_col, desc='denoising', postfix=dict(current=''))
	w = np.array([
		1, 1, 1, 1, 1,
		1, 0, 0, 0, 1,
		1, 0, 0, 0, 1,
		1, 0, 0, 0, 1,
		1, 1, 1, 1, 1,
	], dtype='float64')
	w /= w.sum()
	for i_r in range(n_row):
		for i_c in range(n_col):
			progress.set_postfix(current=f'({i_r} / {n_row}, {i_c} / {n_col})')
			progress.update()
			if (im_var[i_r, i_c] < EPSILON).any():  # very smooth
				im_result[i_r, i_c] = im_original[i_r, i_c]
				continue
			# filter pixel i using neighbors j:
			j_r_begin = max(0, i_r - f_r_2)
			j_r_end = min(n_row, i_r + f_r_2 + 1)
			j_c_begin = max(0, i_c - f_c_2)
			j_c_end = min(n_col, i_c + f_c_2 + 1)
			# b_multi = im_mean[j_r_begin: j_r_end, j_c_begin: j_c_end, :] - im_mean[i_r, i_c, :]
			# v_multi = im_var[j_r_begin: j_r_end, j_c_begin: j_c_end, :]
			o_multi = im_original[j_r_begin: j_r_end, j_c_begin: j_c_end, :]
			# compute filter for each channel:
			for channel in range(3):
				# b = b_multi[:, :, channel].flatten()
				# v = v_multi[:, :, channel].flatten() + EPSILON
				o = o_multi[:, :, channel].flatten()
				if o.shape != w.shape:
					im_result[i_r, i_c, channel] = 0 # skip
					continue
				# print(b, v)
				# assert (v > 0).all()
				# w = opt_GCG(b, v, ret_steps=False, **kwargs)
				im_result[i_r, i_c, channel] = o.dot(w)  # reconstruct
	return im_result


def imread(path) -> np.ndarray:
	return np.array(Image.open(path), dtype='float64') / 255.


def imwrite(img: np.ndarray, path):
	print(np.max(img[:, :, 0]), np.max(img[:, :, 1]), np.max(img[:, :, 2]))
	Image.fromarray((255 * img / np.max(img)).astype('uint8')).save(path)


def imshow(img: np.ndarray):
	Image.fromarray((255 * img / np.max(img)).astype('uint8')).show()


# small test:
# im_mean = imread('data/8196mean.png')[200:300, 300:450, :]
# im_var = imread('data/8196variance.png')[200:300, 300:450, :]
# im_original = imread('data/16mean.png')[200:300, 300:450, :]
# imwrite(im_original, 'out/small 5x5/16original.png')
# imwrite(im_mean, 'out/small 5x5/16reference.png')
# im_result = denoise(im_mean, im_var, im_original, (5, 5))
# imwrite(im_result, 'out/small 5x5/16filtered.png')

im_mean = imread('data/8196mean.png')
im_var = imread('data/8196variance.png')
im_original = imread('data/16mean.png')
# imwrite(im_original, 'out/full 63x63/16original.png')
# imwrite(im_mean, 'out/full 63x63/16reference.png')

# im = Image.open('data/16mean.png')
# im = im.filter(ImageFilter.BLUR)
# im.save('out/blur_ref.png')
im_result = denoise(im_mean, im_var, im_original, (5, 5))
imwrite(im_result, 'out/blur.png')

# im = imread('data/16mean.png')
# imwrite(im, 'data/16mean2.png')
