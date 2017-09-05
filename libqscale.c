/*
 * qscale: Quick, high-quality JPEG-to-JPEG scaler.
 * Copyright (C) 2008 Steinar H. Gunderson <sgunderson@bigfoot.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#include "libqscale.h"

/* The number of pixels to process at a time when scaling vertically. */
#define CACHE_LINE_FACTOR 16

/* Whether to use SSE for horizontal scaling or not (requires SSE3). */
#define USE_HORIZONTAL_SSE 1

/* Whether to use SSE for vertical scaling or not (requires only SSE1). */
#define USE_VERTICAL_SSE 1

#if USE_VERTICAL_SSE
#undef CACHE_LINE_FACTOR
#define CACHE_LINE_FACTOR 16
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264
#endif

#if USE_VERTICAL_SSE || USE_HORIZONTAL_SSE
typedef float v4sf __attribute__((vector_size(16)));
typedef int v4si __attribute__((vector_size(16)));
typedef short v8hi __attribute__((vector_size(16)));
typedef char v16qi __attribute__((vector_size(16)));
#endif

qscale_img *qscale_load_jpeg(const char *filename)
{
	FILE *file = fopen(filename, "rb");
	qscale_img *img;
	if (file == NULL) {
		return NULL;
	}

	img = qscale_load_jpeg_from_stdio(file);

	fclose(file);
	return img;
}

qscale_img *qscale_load_jpeg_from_stdio(FILE *file)
{
	qscale_img *img = (qscale_img *)malloc(sizeof(qscale_img));
	if (img == NULL) {
		return NULL;
	}

	img->data_y = img->data_cb = img->data_cr = NULL;

	/* FIXME: Better error handling here (ie., return NULL). */
	struct jpeg_decompress_struct dinfo;
	struct jpeg_error_mgr jerr;
	dinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&dinfo);
	jpeg_stdio_src(&dinfo, file);
	jpeg_read_header(&dinfo, TRUE);
	dinfo.raw_data_out = TRUE;
	jpeg_start_decompress(&dinfo);
	
	if (dinfo.num_components != 1 && dinfo.num_components != 3) {
		qscale_destroy(img);
		return NULL;
	}
	img->num_components = dinfo.num_components;

	img->width = dinfo.image_width;
	img->height = dinfo.image_height;

	img->w0 = dinfo.image_width * dinfo.comp_info[0].h_samp_factor / dinfo.max_h_samp_factor;
	img->h0 = dinfo.image_height * dinfo.comp_info[0].v_samp_factor / dinfo.max_v_samp_factor;

	if (img->num_components == 3) {
		img->w1 = dinfo.image_width * dinfo.comp_info[1].h_samp_factor / dinfo.max_h_samp_factor;
		img->h1 = dinfo.image_height * dinfo.comp_info[1].v_samp_factor / dinfo.max_v_samp_factor;

		img->w2 = dinfo.image_width * dinfo.comp_info[2].h_samp_factor / dinfo.max_h_samp_factor;
		img->h2 = dinfo.image_height * dinfo.comp_info[2].v_samp_factor / dinfo.max_v_samp_factor;
	}

	img->samp_h0 = dinfo.comp_info[0].h_samp_factor;
	img->samp_v0 = dinfo.comp_info[0].v_samp_factor;

	if (img->num_components == 3) {
		img->samp_h1 = dinfo.comp_info[1].h_samp_factor;
		img->samp_v1 = dinfo.comp_info[1].v_samp_factor;

		img->samp_h2 = dinfo.comp_info[2].h_samp_factor;
		img->samp_v2 = dinfo.comp_info[2].v_samp_factor;
	}

	img->data_y  = (JSAMPLE*)memalign(16, dinfo.comp_info[0].height_in_blocks * dinfo.comp_info[0].width_in_blocks * DCTSIZE * DCTSIZE);
	if (img->data_y == NULL) {
		qscale_destroy(img);
		return NULL;
	}

	if (img->num_components == 3) {
		img->data_cb = (JSAMPLE*)memalign(16, dinfo.comp_info[1].height_in_blocks * dinfo.comp_info[1].width_in_blocks * DCTSIZE * DCTSIZE);
		img->data_cr = (JSAMPLE*)memalign(16, dinfo.comp_info[2].height_in_blocks * dinfo.comp_info[2].width_in_blocks * DCTSIZE * DCTSIZE);
		if (img->data_cb == NULL || img->data_cr == NULL) {
			qscale_destroy(img);
			return NULL;
		}
	}

	int total_lines = 0, blocks = 0;
        while (total_lines < dinfo.comp_info[0].height_in_blocks * DCTSIZE) {
                unsigned max_lines = dinfo.max_v_samp_factor * DCTSIZE;

                JSAMPROW y_row_ptrs[max_lines];
                JSAMPROW cb_row_ptrs[max_lines];
                JSAMPROW cr_row_ptrs[max_lines];
                JSAMPROW* ptrs[] = { y_row_ptrs, cb_row_ptrs, cr_row_ptrs };

		int i;
                for (i = 0; i < max_lines; ++i) {
                        y_row_ptrs[i]  = img->data_y  + (i+blocks*DCTSIZE*dinfo.comp_info[0].v_samp_factor) * dinfo.comp_info[0].width_in_blocks * DCTSIZE;
			if (img->num_components == 3) {
				cb_row_ptrs[i] = img->data_cb + (i+blocks*DCTSIZE*dinfo.comp_info[1].v_samp_factor) * dinfo.comp_info[1].width_in_blocks * DCTSIZE;
				cr_row_ptrs[i] = img->data_cr + (i+blocks*DCTSIZE*dinfo.comp_info[2].v_samp_factor) * dinfo.comp_info[2].width_in_blocks * DCTSIZE;
			}
                }

                total_lines += max_lines;
                ++blocks;

                if (jpeg_read_raw_data(&dinfo, ptrs, max_lines) == 0)
                        break;
        }

	jpeg_destroy_decompress(&dinfo);
	return img;
}

void qscale_destroy(qscale_img *img)
{
	free(img->data_y);
	free(img->data_cb);
	free(img->data_cr);
	free(img);
}


static double sinc(double x)
{
	static const double cutoff = 1.220703668e-4;  /* sqrt(sqrt(eps)) */

	if (abs(x) < cutoff) {
		/* For small |x|, use Taylor series instead */
		const double x2 = x * x;
		const double x4 = x2 * x2;

		return 1.0 - x2 / 6.0 + x4 / 120.0;
	} else {
		return sin(x) / x;
	}
}

static double lanczos_tap(double x)
{
	if (x < -3.0 || x > 3.0)
		return 0.0;
	if (x < 0.0)
		return sinc(-x*M_PI) * sinc(-x*M_PI / 3.0);
	else
		return sinc(x*M_PI) * sinc(x*M_PI / 3.0);
}

static double mitchell_tap(double x)
{
	const double b = 1.0 / 3.0;
	const double c = 1.0 / 3.0;
	const double p0 = (  6.0 -  2.0*b         ) / 6.0;
	const double p2 = (-18.0 + 12.0*b +  6.0*c) / 6.0;
	const double p3 = ( 12.0 -  9.0*b -  6.0*c) / 6.0;
	const double q0 = (         8.0*b + 24.0*c) / 6.0;
	const double q1 = (      - 12.0*b - 48.0*c) / 6.0;
	const double q2 = (         6.0*b + 30.0*c) / 6.0;
	const double q3 = (      -      b -  6.0*c) / 6.0;

	if (x < -2.0) {
		return 0.0;
	} else if (x < -1.0) {
		return q0 - x * (q1 - x * (q2 - x * q3));
	} else if (x < 0.0) {
		return p0 + x * x * (p2 - x * p3);
	} else if (x < 1.0) {
		return p0 + x * x * (p2 + x * p3);
	} else if (x < 2.0) {
		return q0 + x * (q1 + x * (q2 + x * q3));
	} else {
		return 0.0;
	}
}

struct pix_desc {
	unsigned start, end;
	unsigned startcoeff;
};

static void hscale(float *pix, unsigned char *npix, unsigned w, unsigned h, unsigned nw, unsigned sstride, unsigned dstride, enum qscale_scaling_filter scaling_filter)
{
	struct pix_desc *pd = (struct pix_desc *)malloc(nw * sizeof(struct pix_desc));
	int size_coeffs = 8;
	float *coeffs = (float *)malloc(size_coeffs * sizeof(float));
	int num_coeffs = 0;
	int x, y;
	double sf = (double)w / (double)nw;
	double support;
	
	if (scaling_filter == LANCZOS) {
		support = (w > nw) ? (3.0 * sf) : (3.0 / sf);
	} else {  /* Mitchell */
		support = (w > nw) ? (2.0 * sf) : (2.0 / sf);
	}

	/* calculate the filter */
	for (x = 0; x < nw; ++x) {
		int start = ceil(x * sf - support);
		int end = floor(x * sf + support);
		int sx;
		double sum = 0.0;

		if (start < 0) {
			start = 0;
		}
		if (end > w - 1) {
			end = w - 1;
		}

#if USE_HORIZONTAL_SSE
		/* round up so we get a multiple of four for the SSE code */
		int num = (end - start + 1);
		if (num % 4 != 0) {
			/* prefer aligning it if possible */
			if (start % 4 != 0 && start % 4 <= num % 4) {
				num += start % 4;
				start -= start % 4;
			}
			if (num % 4 != 0) {
				end += 4 - (num % 4);
			}
		}
#endif

		pd[x].start = start;
		pd[x].end = end;
		pd[x].startcoeff = num_coeffs;

		for (sx = start; sx <= end; ++sx) {
			double nd = (w > nw) ? (sx/sf - x) : (sx - x*sf);
			double f;
			if (scaling_filter == LANCZOS) {
				f = lanczos_tap(nd);
			} else {  /* Mitchell */
				f = mitchell_tap(nd);
			}
			if (num_coeffs == size_coeffs) {
				size_coeffs <<= 1;
				coeffs = (float *)realloc(coeffs, size_coeffs * sizeof(float));
			}

			coeffs[num_coeffs++] = f;
			sum += f;
		}

		for (sx = start; sx <= end; ++sx) {
			coeffs[pd[x].startcoeff + sx - start] /= sum;
		}
	}

	for (y = 0; y < h; ++y) {
		float *sptr = pix + y*sstride;
		unsigned char *dptr = npix + y*dstride;
		unsigned char ch;
		for (x = 0; x < nw; ++x) {
#if USE_HORIZONTAL_SSE
			v4sf acc = { 0.0f, 0.0f, 0.0f, 0.0f };
			static const v4sf low = { 0.0f, 0.0f, 0.0f, 0.0f };
			static const v4sf high = { 255.0f, 255.0f, 255.0f, 255.0f };
			int result;
			int i;
		
			const float *sptr_xmm = &sptr[pd[x].start];
			const float *coeffptr = &coeffs[pd[x].startcoeff];
			const int filter_len = (pd[x].end - pd[x].start + 1) / 4;

			for (i = 0; i < filter_len; ++i) {
				v4sf pixels = __builtin_ia32_loadups(&sptr_xmm[i * 4]);
				v4sf coeffs = __builtin_ia32_loadups(&coeffptr[i * 4]);
				acc = __builtin_ia32_addps(acc, __builtin_ia32_mulps(pixels, coeffs));
			}
			acc = __builtin_ia32_haddps(acc, acc);	
			acc = __builtin_ia32_haddps(acc, acc);
			acc = __builtin_ia32_maxss(acc, low);
			acc = __builtin_ia32_minss(acc, high);
			result = __builtin_ia32_cvtss2si(acc);

			*dptr++ = (unsigned char)result;
#else
			float acc = 0.0;
			float *cf = &coeffs[pd[x].startcoeff];
			unsigned sx;
			
			for (sx = pd[x].start; sx <= pd[x].end; ++sx) {
				acc += sptr[sx] * *cf++;
			}

			if (acc < 0.0)
				ch = 0;
			else if (acc > 255.0)
				ch = 255;
			else
				ch = (unsigned char)acc;
			*dptr++ = ch;
#endif
		}
		ch = dptr[-1];
		for ( ; x < dstride; ++x) {
			*dptr++ = ch;
		}
	}

	free(pd);
	free(coeffs);
}

static void vscale(unsigned char *pix, float *npix, unsigned w, unsigned h, unsigned nh, unsigned dstride, enum qscale_scaling_filter scaling_filter)
{
	struct pix_desc *pd = (struct pix_desc *)malloc(nh * sizeof(struct pix_desc));
	int size_coeffs = 8;
	float *coeffs = (float *)malloc(size_coeffs * sizeof(float));
	int num_coeffs = 0;
	int x, y, sy;
	double sf = (double)h / (double)nh;
	double support;
	
	if (scaling_filter == LANCZOS) {
		support = (h > nh) ? (3.0 * sf) : (3.0 / sf);
	} else {  /* Mitchell */
		support = (h > nh) ? (2.0 * sf) : (2.0 / sf);
	}

	/* calculate the filter */
	for (y = 0; y < nh; ++y) {
		int start = ceil(y * sf - support);
		int end = floor(y * sf + support);
		double sum = 0.0;

		if (start < 0) {
			start = 0;
		}
		if (end > h - 1) {
			end = h - 1;
		}

		pd[y].start = start;
		pd[y].end = end;
		pd[y].startcoeff = num_coeffs;

		for (sy = start; sy <= end; ++sy) {
			double nd = (h > nh) ? (sy/sf - y) : (sy - y*sf);
			double f;
			if (scaling_filter == LANCZOS) {
				f = lanczos_tap(nd);
			} else {  /* Mitchell */
				f = mitchell_tap(nd);
			}
			if (num_coeffs == size_coeffs) {
				size_coeffs <<= 1;
				coeffs = (float *)realloc(coeffs, size_coeffs * sizeof(float));
			}
			
			coeffs[num_coeffs++] = f;
			sum += f;
		}

		for (sy = start; sy <= end; ++sy) {
			coeffs[pd[y].startcoeff + sy - start] /= sum;
		}
	}

#if CACHE_LINE_FACTOR > 1
	for (x = 0; x < (w/CACHE_LINE_FACTOR) * CACHE_LINE_FACTOR; x += CACHE_LINE_FACTOR) {
		unsigned char *sptr = pix + x;
		float *dptr = npix + x;
		for (y = 0; y < nh; ++y) {
#if USE_VERTICAL_SSE
			/* A zero is useful during unpacking. */
			static const v4sf zero = { 0.0f, 0.0f, 0.0f, 0.0f };
			const unsigned char *sptr_xmm = &sptr[pd[y].start * w];
			const float *coeffptr = &coeffs[pd[y].startcoeff];
			const int filter_len = pd[y].end - pd[y].start + 1;
			int i;

			v4sf acc0 = { 0.0f, 0.0f, 0.0f, 0.0f };
			v4sf acc1 = { 0.0f, 0.0f, 0.0f, 0.0f };
			v4sf acc2 = { 0.0f, 0.0f, 0.0f, 0.0f };
			v4sf acc3 = { 0.0f, 0.0f, 0.0f, 0.0f };
			
			for (i = 0; i < filter_len; ++i, ++coeffptr, sptr_xmm += w) {
				__builtin_prefetch(sptr_xmm + w, 0);
				v16qi src = (v16qi)__builtin_ia32_loadups((float*)sptr_xmm);

				// unpack into words
				v8hi src_lo = (v8hi)__builtin_ia32_punpcklbw128(src, (v16qi)zero);
				v8hi src_hi = (v8hi)__builtin_ia32_punpckhbw128(src, (v16qi)zero);

				// unpack into dwords, convert to floats
				v4si src0_i = (v4si)__builtin_ia32_punpcklwd128(src_lo, (v8hi)zero);
				v4si src1_i = (v4si)__builtin_ia32_punpckhwd128(src_lo, (v8hi)zero);
				v4si src2_i = (v4si)__builtin_ia32_punpcklwd128(src_hi, (v8hi)zero);
				v4si src3_i = (v4si)__builtin_ia32_punpckhwd128(src_hi, (v8hi)zero);

				v4sf src0 = __builtin_ia32_cvtdq2ps(src0_i);
				v4sf src1 = __builtin_ia32_cvtdq2ps(src1_i);
				v4sf src2 = __builtin_ia32_cvtdq2ps(src2_i);
				v4sf src3 = __builtin_ia32_cvtdq2ps(src3_i);
			
				// fetch the coefficient, and replicate it
				v4sf coeff = { *coeffptr, *coeffptr, *coeffptr, *coeffptr };

				// do the actual muladds
				acc0 = __builtin_ia32_addps(acc0, __builtin_ia32_mulps(src0, coeff));
				acc1 = __builtin_ia32_addps(acc1, __builtin_ia32_mulps(src1, coeff));
				acc2 = __builtin_ia32_addps(acc2, __builtin_ia32_mulps(src2, coeff));
				acc3 = __builtin_ia32_addps(acc3, __builtin_ia32_mulps(src3, coeff));
			}

			*(v4sf *)(&dptr[0]) = acc0;
			*(v4sf *)(&dptr[4]) = acc1;
			*(v4sf *)(&dptr[8]) = acc2;
			*(v4sf *)(&dptr[12]) = acc3;
#else
			int i;
			float acc[CACHE_LINE_FACTOR];
			for (i = 0; i < CACHE_LINE_FACTOR; ++i)
				acc[i] = 0.0;
			float *cf = &coeffs[pd[y].startcoeff];
			unsigned sy;
		
			for (sy = pd[y].start; sy <= pd[y].end; ++sy) {
				for (i = 0; i < CACHE_LINE_FACTOR; ++i) {
					acc[i] += sptr[sy * w + i] * *cf;
				}
				++cf;
			}

			for (i = 0; i < CACHE_LINE_FACTOR; ++i) {
				dptr[i] = acc[i];
			}
#endif
			dptr += dstride;
		}
	}
	for (x = (x/CACHE_LINE_FACTOR)*CACHE_LINE_FACTOR; x < w; ++x) {
#else
	for (x = 0; x < w; ++x) {
#endif
		unsigned char *sptr = pix + x;
		float *dptr = npix + x;
		for (y = 0; y < nh; ++y) {
			float acc = 0.0;
			float *cf = &coeffs[pd[y].startcoeff];
			unsigned sy;
			
			for (sy = pd[y].start; sy <= pd[y].end; ++sy) {
				acc += sptr[sy * w] * *cf++;
			}

			*dptr = acc;
			dptr += dstride;
		}
	}
	
	free(pd);
	free(coeffs);
}

qscale_img *qscale_clone(const qscale_img *img)
{
	qscale_img *dst = (qscale_img *)malloc(sizeof(qscale_img));
	if (dst == NULL) {
		return NULL;
	}

	*dst = *img;

	unsigned dstride0 = (dst->w0 + DCTSIZE-1) & ~(DCTSIZE-1);
	unsigned dstride1 = (dst->w1 + DCTSIZE-1) & ~(DCTSIZE-1);
	unsigned dstride2 = (dst->w2 + DCTSIZE-1) & ~(DCTSIZE-1);

	/* FIXME: handle out-of-memory gracefully */
	{
		dst->data_y = (unsigned char *)malloc(dst->h0 * dstride0);
		memcpy(dst->data_y, img->data_y, dst->h0 * dstride0);
	}
	{
		dst->data_cb = (unsigned char *)malloc(dst->h1 * dstride1);
		memcpy(dst->data_cb, img->data_cb, dst->h1 * dstride1);
	}
	{
		dst->data_cr = (unsigned char *)malloc(dst->h2 * dstride2);
		memcpy(dst->data_cr, img->data_cr, dst->h2 * dstride2);
	}

	return dst;
}

qscale_img *qscale_scale(qscale_img *src, unsigned width, unsigned height, unsigned samp_h0, unsigned samp_v0, unsigned samp_h1, unsigned samp_v1, unsigned samp_h2, unsigned samp_v2, enum qscale_scaling_filter scaling_filter)
{
	qscale_img *dst = (qscale_img *)malloc(sizeof(qscale_img));
	if (dst == NULL) {
		return NULL;
	}

	dst->width = width;
	dst->height = height;
	dst->num_components = src->num_components;

	unsigned max_samp_h, max_samp_v;
        max_samp_h = samp_h0;
	if (src->num_components == 3) {
		if (samp_h1 > max_samp_h)
			max_samp_h = samp_h1;
		if (samp_h2 > max_samp_h)
			max_samp_h = samp_h2;
	}

        max_samp_v = samp_v0;
	if (src->num_components == 3) {
		if (samp_v1 > max_samp_v)
			max_samp_v = samp_v1;
		if (samp_v2 > max_samp_v)
			max_samp_v = samp_v2;
	}

	dst->w0 = width * samp_h0 / max_samp_h;
	dst->h0 = height * samp_v0 / max_samp_v;

	if (src->num_components == 3) {
		dst->w1 = width * samp_h1 / max_samp_h;
		dst->h1 = height * samp_v1 / max_samp_v;

		dst->w2 = width * samp_h2 / max_samp_h;
		dst->h2 = height * samp_v2 / max_samp_v;
	}

	dst->samp_h0 = samp_h0;
	dst->samp_v0 = samp_v0;

	if (src->num_components == 3) {
		dst->samp_h1 = samp_h1;
		dst->samp_v1 = samp_v1;

		dst->samp_h2 = samp_h2;
		dst->samp_v2 = samp_v2;
	}

	unsigned dstride0 = (dst->w0 + DCTSIZE-1) & ~(DCTSIZE-1);
	unsigned dstride1 = (dst->w1 + DCTSIZE-1) & ~(DCTSIZE-1);
	unsigned dstride2 = (dst->w2 + DCTSIZE-1) & ~(DCTSIZE-1);

	unsigned sstride0 = (src->w0 + DCTSIZE-1) & ~(DCTSIZE-1);
	unsigned sstride1 = (src->w1 + DCTSIZE-1) & ~(DCTSIZE-1);
	unsigned sstride2 = (src->w2 + DCTSIZE-1) & ~(DCTSIZE-1);

	/* FIXME: handle out-of-memory gracefully */
	{
		float *npix = (float*)memalign(16, sstride0 * dst->h0 * sizeof(float));
		vscale(src->data_y, npix, sstride0, src->h0, dst->h0, sstride0, scaling_filter);
		dst->data_y = (unsigned char *)malloc(dst->h0 * dstride0);
		hscale(npix, dst->data_y, src->w0, dst->h0, dst->w0, sstride0, dstride0, scaling_filter);
		free(npix);
	}
	if (src->num_components == 3) {
		{
			float *npix = (float*)memalign(16, sstride1 * dst->h1 * sizeof(float));
			vscale(src->data_cr, npix, sstride1, src->h1, dst->h1, sstride1, scaling_filter);
			dst->data_cr = (unsigned char *)malloc(dst->h1 * dstride1);
			hscale(npix, dst->data_cr, src->w1, dst->h1, dst->w1, sstride1, dstride1, scaling_filter);
			free(npix);
		}
		{
			float *npix = (float*)memalign(16, sstride2 * dst->h2 * sizeof(float));
			vscale(src->data_cb, npix, sstride2, src->h2, dst->h2, sstride2, scaling_filter);
			dst->data_cb = (unsigned char *)malloc(dst->h2 * dstride2);
			hscale(npix, dst->data_cb, src->w2, dst->h2, dst->w2, sstride2, dstride2, scaling_filter);
			free(npix);
		}
	}

	return dst;
}

int qscale_save_jpeg(const qscale_img *img, const char *filename, unsigned jpeg_quality, enum qscale_jpeg_mode jpeg_mode)
{
	FILE *file = fopen(filename, "wb");
	if (file == NULL) {
		return -1;
	}

	int err = qscale_save_jpeg_to_stdio(img, file, jpeg_quality, jpeg_mode);

	fclose(file);
	return err;
}

int qscale_save_jpeg_to_stdio(const qscale_img *img, FILE *file, unsigned jpeg_quality, enum qscale_jpeg_mode jpeg_mode)
{
        struct jpeg_compress_struct cinfo;
        struct jpeg_error_mgr jerr;
        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);
        jpeg_stdio_dest(&cinfo, file);
        cinfo.input_components = img->num_components;
        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, jpeg_quality, FALSE);

	if (jpeg_mode == PROGRESSIVE) {
		jpeg_simple_progression(&cinfo);
	}

        cinfo.image_width = img->width;
        cinfo.image_height = img->height;
        cinfo.raw_data_in = TRUE;
	if (img->num_components == 3) {
		jpeg_set_colorspace(&cinfo, JCS_YCbCr);
	} else {
		jpeg_set_colorspace(&cinfo, JCS_GRAYSCALE);
	}
        cinfo.comp_info[0].h_samp_factor = img->samp_h0;
        cinfo.comp_info[0].v_samp_factor = img->samp_v0;
	if (img->num_components == 3) {
		cinfo.comp_info[1].h_samp_factor = img->samp_h1;
		cinfo.comp_info[1].v_samp_factor = img->samp_v1;
		cinfo.comp_info[2].h_samp_factor = img->samp_h2;
		cinfo.comp_info[2].v_samp_factor = img->samp_v2;
	}
        jpeg_start_compress(&cinfo, TRUE);

        unsigned dstride0 = (img->w0 + DCTSIZE-1) & ~(DCTSIZE-1);
        unsigned dstride1 = (img->w1 + DCTSIZE-1) & ~(DCTSIZE-1);
        unsigned dstride2 = (img->w2 + DCTSIZE-1) & ~(DCTSIZE-1);

        int total_lines = 0;
        int blocks = 0;
        while (total_lines < cinfo.comp_info[0].height_in_blocks * DCTSIZE) {
                unsigned max_lines = cinfo.max_v_samp_factor * DCTSIZE;

                JSAMPROW y_row_ptrs[max_lines];
                JSAMPROW cb_row_ptrs[max_lines];
                JSAMPROW cr_row_ptrs[max_lines];
                JSAMPROW* ptrs[] = { y_row_ptrs, cb_row_ptrs, cr_row_ptrs };
                int i;

                for (i = 0; i < max_lines; ++i) {
                        /* simple edge extension */
                        int yline = i + blocks*DCTSIZE*cinfo.comp_info[0].v_samp_factor;
                        if (yline > img->h0 - 1)
                                yline = img->h0 - 1;

                        y_row_ptrs[i]  = img->data_y  + yline * dstride0;

			if (img->num_components == 3) {
				int cbline = i + blocks*DCTSIZE*cinfo.comp_info[1].v_samp_factor;
				if (cbline > img->h1 - 1)
					cbline = img->h1 - 1;

				int crline = i + blocks*DCTSIZE*cinfo.comp_info[2].v_samp_factor;
				if (crline > img->h2 - 1)
					crline = img->h2 - 1;

				cb_row_ptrs[i] = img->data_cb + cbline * dstride1;
				cr_row_ptrs[i] = img->data_cr + crline * dstride2;
			}
                }

                total_lines += max_lines;
                ++blocks;

                jpeg_write_raw_data(&cinfo, ptrs, max_lines);
        }
        jpeg_finish_compress(&cinfo);
        jpeg_destroy_compress(&cinfo);

	return 0;
}
