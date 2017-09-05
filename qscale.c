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

#include <stdio.h>
#include <stdlib.h>
#include "libqscale.h"

int main(int argc, char **argv)
{
	/* user-settable parameters */
	unsigned nominal_w = atoi(argv[1]);
	unsigned nominal_h = atoi(argv[2]);
	unsigned samp_h0 = 2, samp_v0 = 2;
	unsigned samp_h1 = 1, samp_v1 = 1;
	unsigned samp_h2 = 1, samp_v2 = 1;
	unsigned jpeg_quality = 85;
	/* end */

	qscale_img *img = qscale_load_jpeg_from_stdio(stdin);
	qscale_img *scaled = qscale_scale(img, nominal_w, nominal_h, samp_h0, samp_v0, samp_h1, samp_v1, samp_h2, samp_v2, LANCZOS);
	qscale_destroy(img);
	qscale_save_jpeg_to_stdio(scaled, stdout, jpeg_quality, SEQUENTIAL);

	return 0;
}

