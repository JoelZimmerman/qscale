#ifndef _LIBQSCALE_H
#define _LIBQSCALE_H

#include <stdio.h>
#include "jpeglib.h"

typedef struct {
	/* True image size */
	unsigned width, height;

	/* 1 = B/W, 3 = YCbCr */	
	unsigned num_components;

	/* Component image sizes (possibly subsampled) */
	unsigned w0, h0;
	unsigned w1, h1;
	unsigned w2, h2;

	/* Sampling factors */
	unsigned samp_h0, samp_v0;
	unsigned samp_h1, samp_v1;
	unsigned samp_h2, samp_v2;

	/* The data itself */
	JSAMPLE *data_y, *data_cb, *data_cr;
} qscale_img;

enum qscale_scaling_filter {
	LANCZOS = 0,
	MITCHELL = 1,
};

enum qscale_jpeg_mode {
	SEQUENTIAL = 0,
	PROGRESSIVE = 1,
};

qscale_img *qscale_load_jpeg(const char *filename);
qscale_img *qscale_load_jpeg_from_stdio(FILE *file);
int qscale_save_jpeg(const qscale_img *img, const char *filename, unsigned jpeg_quality, enum qscale_jpeg_mode jpeg_mode);
int qscale_save_jpeg_to_stdio(const qscale_img *img, FILE *file, unsigned jpeg_quality, enum qscale_jpeg_mode jpeg_mode);

qscale_img *qscale_clone(const qscale_img *img);
qscale_img *qscale_scale(qscale_img *src, unsigned width, unsigned height, unsigned samp_h0, unsigned samp_v0, unsigned samp_h1, unsigned samp_v1, unsigned samp_h2, unsigned samp_v2, enum qscale_scaling_filter scaling_filter);
void qscale_destroy(qscale_img *img);

#endif /* !defined(_LIBQSCALE_H) */
