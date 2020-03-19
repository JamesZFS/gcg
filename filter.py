from optimizer import *
from PIL import Image, ImageFilter
from tqdm import tqdm
import time

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
	if not (f_r & 1 and f_c & 1):
		raise ValueError('Bad filter size, expect odd size!')
	f_r_2, f_c_2 = int(f_r / 2), int(f_c / 2)
	im_result = np.zeros_like(im_mean)
	progress = tqdm(total=n_row * n_col, desc='denoising', postfix=dict(current=''))
	for i_r in range(n_row):
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
	return im_result


def imread(path) -> np.ndarray:
	return np.array(Image.open(path), dtype='float64') / 255.


def imwrite(img: np.ndarray, path):
	img[img > 1.0] = 1.0
	Image.fromarray((255 * img).astype('uint8')).save(path)


def imshow(img: np.ndarray):
	img[img > 1.0] = 1.0
	Image.fromarray((255 * img).astype('uint8')).show()


def show_diff(img1, img2):
	imshow(img1 - img2)


# small test:
# im_mean = imread('data/images-png-part1/16spp/mean_cross_filtered_iter2.png')[200:300, 300:450, :]
# im_mean = imread('data/16mean.png')[200:300, 300:450, :]
# im_var = imread('data/16variance.png')[200:300, 300:450, :] / (16 ** 2)
# im_var = imread('data/images-png-part1/16spp/variance_cross_filtered_iter2.png')[200:300, 300:450, :] / (16 ** 2)
# im_original = imread('data/16mean.png')[200:300, 300:450, :]
# im_result = denoise(im_mean, im_var, im_original, (63, 63), max_iter=0)
# imwrite(im_result, 'out/small 63x63/filtered_iter_0_bias_var_guided_by_self.png')

# massive test
im_mean = imread('out/full 63x63/mean_cross_filtered_iter2.png')
im_var = imread('data/images-png-part1/16spp/variance_cross_filtered_iter2.png') / (16 ** 2)
im_original = imread('data/16mean.png')
# imwrite(im_original, 'out/full 63x63/16original.png')
# imwrite(im_mean, 'out/full 63x63/mean_cross_filtered_iter2.png')
im_result = denoise(im_mean, im_var, im_original, (63, 63), max_iter=0)
imwrite(im_result, 'out/full 63x63/iter_0_init_bias_var_guided_by_cross.png')

# diff
# imwrite(imread('out/full 63x63/16reference.png')[:,:,0:3] - imread('out/full 63x63/iter_0_init_bias_var.png'), 'out/full 63x63/diff.png')

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
