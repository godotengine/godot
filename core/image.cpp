/*************************************************************************/
/*  image.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "image.h"

#include "core/io/image_loader.h"
#include "core/os/copymem.h"
#include "hash_map.h"
#include "hq2x.h"
#include "print_string.h"

#include <stdio.h>

const char *Image::format_names[Image::FORMAT_MAX] = {
	"Lum8", //luminance
	"LumAlpha8", //luminance-alpha
	"Red8",
	"RedGreen",
	"RGB8",
	"RGBA8",
	"RGB565", //16 bit
	"RGBA4444",
	"RGBA5551",
	"RFloat", //float
	"RGFloat",
	"RGBFloat",
	"RGBAFloat",
	"RHalf", //half float
	"RGHalf",
	"RGBHalf",
	"RGBAHalf",
	"DXT1", //s3tc
	"DXT3",
	"DXT5",
	"ATI1",
	"ATI2",
	"BPTC_RGBA",
	"BPTC_RGBF",
	"BPTC_RGBFU",
	"PVRTC2", //pvrtc
	"PVRTC2A",
	"PVRTC4",
	"PVRTC4A",
	"ETC", //etc1
	"ETC2_R11", //etc2
	"ETC2_R11S", //signed", NOT srgb.
	"ETC2_RG11",
	"ETC2_RG11S",
	"ETC2_RGB8",
	"ETC2_RGBA8",
	"ETC2_RGB8A1",

};

SavePNGFunc Image::save_png_func = NULL;

void Image::_put_pixelb(int p_x, int p_y, uint32_t p_pixelsize, uint8_t *p_dst, const uint8_t *p_src) {

	uint32_t ofs = (p_y * width + p_x) * p_pixelsize;

	for (uint32_t i = 0; i < p_pixelsize; i++) {
		p_dst[ofs + i] = p_src[i];
	}
}

void Image::_get_pixelb(int p_x, int p_y, uint32_t p_pixelsize, const uint8_t *p_src, uint8_t *p_dst) {

	uint32_t ofs = (p_y * width + p_x) * p_pixelsize;

	for (uint32_t i = 0; i < p_pixelsize; i++) {
		p_dst[ofs] = p_src[ofs + i];
	}
}

int Image::get_format_pixel_size(Format p_format) {

	switch (p_format) {
		case FORMAT_L8:
			return 1; //luminance
		case FORMAT_LA8:
			return 2; //luminance-alpha
		case FORMAT_R8: return 1;
		case FORMAT_RG8: return 2;
		case FORMAT_RGB8: return 3;
		case FORMAT_RGBA8: return 4;
		case FORMAT_RGB565:
			return 2; //16 bit
		case FORMAT_RGBA4444: return 2;
		case FORMAT_RGBA5551: return 2;
		case FORMAT_RF:
			return 4; //float
		case FORMAT_RGF: return 8;
		case FORMAT_RGBF: return 12;
		case FORMAT_RGBAF: return 16;
		case FORMAT_RH:
			return 2; //half float
		case FORMAT_RGH: return 4;
		case FORMAT_RGBH: return 8;
		case FORMAT_RGBAH: return 12;
		case FORMAT_DXT1:
			return 1; //s3tc bc1
		case FORMAT_DXT3:
			return 1; //bc2
		case FORMAT_DXT5:
			return 1; //bc3
		case FORMAT_ATI1:
			return 1; //bc4
		case FORMAT_ATI2:
			return 1; //bc5
		case FORMAT_BPTC_RGBA:
			return 1; //btpc bc6h
		case FORMAT_BPTC_RGBF:
			return 1; //float /
		case FORMAT_BPTC_RGBFU:
			return 1; //unsigned float
		case FORMAT_PVRTC2:
			return 1; //pvrtc
		case FORMAT_PVRTC2A: return 1;
		case FORMAT_PVRTC4: return 1;
		case FORMAT_PVRTC4A: return 1;
		case FORMAT_ETC:
			return 1; //etc1
		case FORMAT_ETC2_R11:
			return 1; //etc2
		case FORMAT_ETC2_R11S:
			return 1; //signed: return 1; NOT srgb.
		case FORMAT_ETC2_RG11: return 1;
		case FORMAT_ETC2_RG11S: return 1;
		case FORMAT_ETC2_RGB8: return 1;
		case FORMAT_ETC2_RGBA8: return 1;
		case FORMAT_ETC2_RGB8A1: return 1;
		case FORMAT_MAX: {
		}
	}
	return 0;
}

void Image::get_format_min_pixel_size(Format p_format, int &r_w, int &r_h) {

	switch (p_format) {
		case FORMAT_DXT1: //s3tc bc1
		case FORMAT_DXT3: //bc2
		case FORMAT_DXT5: //bc3
		case FORMAT_ATI1: //bc4
		case FORMAT_ATI2: { //bc5		case case FORMAT_DXT1:

			r_w = 4;
			r_h = 4;
		} break;
		case FORMAT_PVRTC2:
		case FORMAT_PVRTC2A: {

			r_w = 16;
			r_h = 8;
		} break;
		case FORMAT_PVRTC4A:
		case FORMAT_PVRTC4: {

			r_w = 8;
			r_h = 8;
		} break;
		case FORMAT_ETC: {

			r_w = 4;
			r_h = 4;
		} break;
		case FORMAT_BPTC_RGBA:
		case FORMAT_BPTC_RGBF:
		case FORMAT_BPTC_RGBFU: {

			r_w = 4;
			r_h = 4;
		} break;
		case FORMAT_ETC2_R11: //etc2
		case FORMAT_ETC2_R11S: //signed: NOT srgb.
		case FORMAT_ETC2_RG11:
		case FORMAT_ETC2_RG11S:
		case FORMAT_ETC2_RGB8:
		case FORMAT_ETC2_RGBA8:
		case FORMAT_ETC2_RGB8A1: {

			r_w = 4;
			r_h = 4;

		} break;

		default: {
			r_w = 1;
			r_h = 1;
		} break;
	}
}

int Image::get_format_pixel_rshift(Format p_format) {

	if (p_format == FORMAT_DXT1 || p_format == FORMAT_ATI1 || p_format == FORMAT_PVRTC4 || p_format == FORMAT_PVRTC4A || p_format == FORMAT_ETC || p_format == FORMAT_ETC2_R11 || p_format == FORMAT_ETC2_R11S || p_format == FORMAT_ETC2_RGB8 || p_format == FORMAT_ETC2_RGB8A1)
		return 1;
	else if (p_format == FORMAT_PVRTC2 || p_format == FORMAT_PVRTC2A)
		return 2;
	else
		return 0;
}

void Image::_get_mipmap_offset_and_size(int p_mipmap, int &r_offset, int &r_width, int &r_height) const {

	int w = width;
	int h = height;
	int ofs = 0;

	int pixel_size = get_format_pixel_size(format);
	int pixel_rshift = get_format_pixel_rshift(format);
	int minw, minh;
	get_format_min_pixel_size(format, minw, minh);

	for (int i = 0; i < p_mipmap; i++) {
		int s = w * h;
		s *= pixel_size;
		s >>= pixel_rshift;
		ofs += s;
		w = MAX(minw, w >> 1);
		h = MAX(minh, h >> 1);
	}

	r_offset = ofs;
	r_width = w;
	r_height = h;
}
int Image::get_mipmap_offset(int p_mipmap) const {

	ERR_FAIL_INDEX_V(p_mipmap, get_mipmap_count() + 1, -1);

	int ofs, w, h;
	_get_mipmap_offset_and_size(p_mipmap, ofs, w, h);
	return ofs;
}

void Image::get_mipmap_offset_and_size(int p_mipmap, int &r_ofs, int &r_size) const {

	int ofs, w, h;
	_get_mipmap_offset_and_size(p_mipmap, ofs, w, h);
	int ofs2;
	_get_mipmap_offset_and_size(p_mipmap + 1, ofs2, w, h);
	r_ofs = ofs;
	r_size = ofs2 - ofs;
}

void Image::get_mipmap_offset_size_and_dimensions(int p_mipmap, int &r_ofs, int &r_size, int &w, int &h) const {

	int ofs;
	_get_mipmap_offset_and_size(p_mipmap, ofs, w, h);
	int ofs2, w2, h2;
	_get_mipmap_offset_and_size(p_mipmap + 1, ofs2, w2, h2);
	r_ofs = ofs;
	r_size = ofs2 - ofs;
}

int Image::get_width() const {

	return width;
}

int Image::get_height() const {

	return height;
}

bool Image::has_mipmaps() const {

	return mipmaps;
}

int Image::get_mipmap_count() const {

	if (mipmaps)
		return get_image_required_mipmaps(width, height, format);
	else
		return 0;
}

//using template generates perfectly optimized code due to constant expression reduction and unused variable removal present in all compilers
template <uint32_t read_bytes, bool read_alpha, uint32_t write_bytes, bool write_alpha, bool read_gray, bool write_gray>
static void _convert(int p_width, int p_height, const uint8_t *p_src, uint8_t *p_dst) {

	for (int y = 0; y < p_height; y++) {
		for (int x = 0; x < p_width; x++) {

			const uint8_t *rofs = &p_src[((y * p_width) + x) * (read_bytes + (read_alpha ? 1 : 0))];
			uint8_t *wofs = &p_dst[((y * p_width) + x) * (write_bytes + (write_alpha ? 1 : 0))];

			uint8_t rgba[4];

			if (read_gray) {
				rgba[0] = rofs[0];
				rgba[1] = rofs[0];
				rgba[2] = rofs[0];
			} else {
				for (uint32_t i = 0; i < MAX(read_bytes, write_bytes); i++) {

					rgba[i] = (i < read_bytes) ? rofs[i] : 0;
				}
			}

			if (read_alpha || write_alpha) {
				rgba[3] = read_alpha ? rofs[read_bytes] : 255;
			}

			if (write_gray) {
				//TODO: not correct grayscale, should use fixed point version of actual weights
				wofs[0] = uint8_t((uint16_t(rofs[0]) + uint16_t(rofs[1]) + uint16_t(rofs[2])) / 3);
			} else {
				for (uint32_t i = 0; i < write_bytes; i++) {

					wofs[i] = rgba[i];
				}
			}

			if (write_alpha) {
				wofs[write_bytes] = rgba[3];
			}
		}
	}
}

void Image::convert(Format p_new_format) {

	if (data.size() == 0)
		return;

	if (p_new_format == format)
		return;

	if (format >= FORMAT_RGB565 || p_new_format >= FORMAT_RGB565) {

		ERR_EXPLAIN("Cannot convert to <-> from non byte formats.");
		ERR_FAIL();
	}

	Image new_img(width, height, 0, p_new_format);

	//int len=data.size();

	PoolVector<uint8_t>::Read r = data.read();
	PoolVector<uint8_t>::Write w = new_img.data.write();

	const uint8_t *rptr = r.ptr();
	uint8_t *wptr = w.ptr();

	int conversion_type = format | p_new_format << 8;

	switch (conversion_type) {

		case FORMAT_L8 | (FORMAT_LA8 << 8): _convert<1, false, 1, true, true, true>(width, height, rptr, wptr); break;
		case FORMAT_L8 | (FORMAT_R8 << 8): _convert<1, false, 1, false, true, false>(width, height, rptr, wptr); break;
		case FORMAT_L8 | (FORMAT_RG8 << 8): _convert<1, false, 2, false, true, false>(width, height, rptr, wptr); break;
		case FORMAT_L8 | (FORMAT_RGB8 << 8): _convert<1, false, 3, false, true, false>(width, height, rptr, wptr); break;
		case FORMAT_L8 | (FORMAT_RGBA8 << 8): _convert<1, false, 3, true, true, false>(width, height, rptr, wptr); break;
		case FORMAT_LA8 | (FORMAT_L8 << 8): _convert<1, true, 1, false, true, true>(width, height, rptr, wptr); break;
		case FORMAT_LA8 | (FORMAT_R8 << 8): _convert<1, true, 1, false, true, false>(width, height, rptr, wptr); break;
		case FORMAT_LA8 | (FORMAT_RG8 << 8): _convert<1, true, 2, false, true, false>(width, height, rptr, wptr); break;
		case FORMAT_LA8 | (FORMAT_RGB8 << 8): _convert<1, true, 3, false, true, false>(width, height, rptr, wptr); break;
		case FORMAT_LA8 | (FORMAT_RGBA8 << 8): _convert<1, true, 3, true, true, false>(width, height, rptr, wptr); break;
		case FORMAT_R8 | (FORMAT_L8 << 8): _convert<1, false, 1, false, false, true>(width, height, rptr, wptr); break;
		case FORMAT_R8 | (FORMAT_LA8 << 8): _convert<1, false, 1, true, false, true>(width, height, rptr, wptr); break;
		case FORMAT_R8 | (FORMAT_RG8 << 8): _convert<1, false, 2, false, false, false>(width, height, rptr, wptr); break;
		case FORMAT_R8 | (FORMAT_RGB8 << 8): _convert<1, false, 3, false, false, false>(width, height, rptr, wptr); break;
		case FORMAT_R8 | (FORMAT_RGBA8 << 8): _convert<1, false, 3, true, false, false>(width, height, rptr, wptr); break;
		case FORMAT_RG8 | (FORMAT_L8 << 8): _convert<2, false, 1, false, false, true>(width, height, rptr, wptr); break;
		case FORMAT_RG8 | (FORMAT_LA8 << 8): _convert<2, false, 1, true, false, true>(width, height, rptr, wptr); break;
		case FORMAT_RG8 | (FORMAT_R8 << 8): _convert<2, false, 1, false, false, false>(width, height, rptr, wptr); break;
		case FORMAT_RG8 | (FORMAT_RGB8 << 8): _convert<2, false, 3, false, false, false>(width, height, rptr, wptr); break;
		case FORMAT_RG8 | (FORMAT_RGBA8 << 8): _convert<2, false, 3, true, false, false>(width, height, rptr, wptr); break;
		case FORMAT_RGB8 | (FORMAT_L8 << 8): _convert<3, false, 1, false, false, true>(width, height, rptr, wptr); break;
		case FORMAT_RGB8 | (FORMAT_LA8 << 8): _convert<3, false, 1, true, false, true>(width, height, rptr, wptr); break;
		case FORMAT_RGB8 | (FORMAT_R8 << 8): _convert<3, false, 1, false, false, false>(width, height, rptr, wptr); break;
		case FORMAT_RGB8 | (FORMAT_RG8 << 8): _convert<3, false, 2, false, false, false>(width, height, rptr, wptr); break;
		case FORMAT_RGB8 | (FORMAT_RGBA8 << 8): _convert<3, false, 3, true, false, false>(width, height, rptr, wptr); break;
		case FORMAT_RGBA8 | (FORMAT_L8 << 8): _convert<3, true, 1, false, false, true>(width, height, rptr, wptr); break;
		case FORMAT_RGBA8 | (FORMAT_LA8 << 8): _convert<3, true, 1, true, false, true>(width, height, rptr, wptr); break;
		case FORMAT_RGBA8 | (FORMAT_R8 << 8): _convert<3, true, 1, false, false, false>(width, height, rptr, wptr); break;
		case FORMAT_RGBA8 | (FORMAT_RG8 << 8): _convert<3, true, 2, false, false, false>(width, height, rptr, wptr); break;
		case FORMAT_RGBA8 | (FORMAT_RGB8 << 8): _convert<3, true, 3, false, false, false>(width, height, rptr, wptr); break;
	}

	r = PoolVector<uint8_t>::Read();
	w = PoolVector<uint8_t>::Write();

	bool gen_mipmaps = mipmaps;

	//mipmaps=false;

	*this = new_img;

	if (gen_mipmaps)
		generate_mipmaps();
}

Image::Format Image::get_format() const {

	return format;
}

static double _bicubic_interp_kernel(double x) {

	x = ABS(x);

	double bc = 0;

	if (x <= 1)
		bc = (1.5 * x - 2.5) * x * x + 1;
	else if (x < 2)
		bc = ((-0.5 * x + 2.5) * x - 4) * x + 2;

	return bc;
}

template <int CC>
static void _scale_cubic(const uint8_t *p_src, uint8_t *p_dst, uint32_t p_src_width, uint32_t p_src_height, uint32_t p_dst_width, uint32_t p_dst_height) {

	// get source image size
	int width = p_src_width;
	int height = p_src_height;
	double xfac = (double)width / p_dst_width;
	double yfac = (double)height / p_dst_height;
	// coordinates of source points and cooefficiens
	double ox, oy, dx, dy, k1, k2;
	int ox1, oy1, ox2, oy2;
	// destination pixel values
	// width and height decreased by 1
	int ymax = height - 1;
	int xmax = width - 1;
	// temporary pointer

	for (uint32_t y = 0; y < p_dst_height; y++) {
		// Y coordinates
		oy = (double)y * yfac - 0.5f;
		oy1 = (int)oy;
		dy = oy - (double)oy1;

		for (uint32_t x = 0; x < p_dst_width; x++) {
			// X coordinates
			ox = (double)x * xfac - 0.5f;
			ox1 = (int)ox;
			dx = ox - (double)ox1;

			// initial pixel value

			uint8_t *dst = p_dst + (y * p_dst_width + x) * CC;

			double color[CC];
			for (int i = 0; i < CC; i++) {
				color[i] = 0;
			}

			for (int n = -1; n < 3; n++) {
				// get Y cooefficient
				k1 = _bicubic_interp_kernel(dy - (double)n);

				oy2 = oy1 + n;
				if (oy2 < 0)
					oy2 = 0;
				if (oy2 > ymax)
					oy2 = ymax;

				for (int m = -1; m < 3; m++) {
					// get X cooefficient
					k2 = k1 * _bicubic_interp_kernel((double)m - dx);

					ox2 = ox1 + m;
					if (ox2 < 0)
						ox2 = 0;
					if (ox2 > xmax)
						ox2 = xmax;

					// get pixel of original image
					const uint8_t *p = p_src + (oy2 * p_src_width + ox2) * CC;

					for (int i = 0; i < CC; i++) {

						color[i] += p[i] * k2;
					}
				}
			}

			for (int i = 0; i < CC; i++) {
				dst[i] = CLAMP(Math::fast_ftoi(color[i]), 0, 255);
			}
		}
	}
}

template <int CC>
static void _scale_bilinear(const uint8_t *p_src, uint8_t *p_dst, uint32_t p_src_width, uint32_t p_src_height, uint32_t p_dst_width, uint32_t p_dst_height) {

	enum {
		FRAC_BITS = 8,
		FRAC_LEN = (1 << FRAC_BITS),
		FRAC_MASK = FRAC_LEN - 1

	};

	for (uint32_t i = 0; i < p_dst_height; i++) {

		uint32_t src_yofs_up_fp = (i * p_src_height * FRAC_LEN / p_dst_height);
		uint32_t src_yofs_frac = src_yofs_up_fp & FRAC_MASK;
		uint32_t src_yofs_up = src_yofs_up_fp >> FRAC_BITS;

		uint32_t src_yofs_down = (i + 1) * p_src_height / p_dst_height;
		if (src_yofs_down >= p_src_height)
			src_yofs_down = p_src_height - 1;

		//src_yofs_up*=CC;
		//src_yofs_down*=CC;

		uint32_t y_ofs_up = src_yofs_up * p_src_width * CC;
		uint32_t y_ofs_down = src_yofs_down * p_src_width * CC;

		for (uint32_t j = 0; j < p_dst_width; j++) {

			uint32_t src_xofs_left_fp = (j * p_src_width * FRAC_LEN / p_dst_width);
			uint32_t src_xofs_frac = src_xofs_left_fp & FRAC_MASK;
			uint32_t src_xofs_left = src_xofs_left_fp >> FRAC_BITS;
			uint32_t src_xofs_right = (j + 1) * p_src_width / p_dst_width;
			if (src_xofs_right >= p_src_width)
				src_xofs_right = p_src_width - 1;

			src_xofs_left *= CC;
			src_xofs_right *= CC;

			for (uint32_t l = 0; l < CC; l++) {

				uint32_t p00 = p_src[y_ofs_up + src_xofs_left + l] << FRAC_BITS;
				uint32_t p10 = p_src[y_ofs_up + src_xofs_right + l] << FRAC_BITS;
				uint32_t p01 = p_src[y_ofs_down + src_xofs_left + l] << FRAC_BITS;
				uint32_t p11 = p_src[y_ofs_down + src_xofs_right + l] << FRAC_BITS;

				uint32_t interp_up = p00 + (((p10 - p00) * src_xofs_frac) >> FRAC_BITS);
				uint32_t interp_down = p01 + (((p11 - p01) * src_xofs_frac) >> FRAC_BITS);
				uint32_t interp = interp_up + (((interp_down - interp_up) * src_yofs_frac) >> FRAC_BITS);
				interp >>= FRAC_BITS;
				p_dst[i * p_dst_width * CC + j * CC + l] = interp;
			}
		}
	}
}

template <int CC>
static void _scale_nearest(const uint8_t *p_src, uint8_t *p_dst, uint32_t p_src_width, uint32_t p_src_height, uint32_t p_dst_width, uint32_t p_dst_height) {

	for (uint32_t i = 0; i < p_dst_height; i++) {

		uint32_t src_yofs = i * p_src_height / p_dst_height;
		uint32_t y_ofs = src_yofs * p_src_width * CC;

		for (uint32_t j = 0; j < p_dst_width; j++) {

			uint32_t src_xofs = j * p_src_width / p_dst_width;
			src_xofs *= CC;

			for (uint32_t l = 0; l < CC; l++) {

				uint32_t p = p_src[y_ofs + src_xofs + l];
				p_dst[i * p_dst_width * CC + j * CC + l] = p;
			}
		}
	}
}

void Image::resize_to_po2(bool p_square) {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot resize in indexed, compressed or custom image formats.");
		ERR_FAIL();
	}

	int w = nearest_power_of_2(width);
	int h = nearest_power_of_2(height);

	if (w == width && h == height) {

		if (!p_square || w == h)
			return; //nothing to do
	}

	resize(w, h);
}

Image Image::resized(int p_width, int p_height, int p_interpolation) {

	Image ret = *this;
	ret.resize(p_width, p_height, (Interpolation)p_interpolation);

	return ret;
};

void Image::resize(int p_width, int p_height, Interpolation p_interpolation) {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot resize in indexed, compressed or custom image formats.");
		ERR_FAIL();
	}

	ERR_FAIL_COND(p_width <= 0);
	ERR_FAIL_COND(p_height <= 0);
	ERR_FAIL_COND(p_width > MAX_WIDTH);
	ERR_FAIL_COND(p_height > MAX_HEIGHT);

	if (p_width == width && p_height == height)
		return;

	Image dst(p_width, p_height, 0, format);

	PoolVector<uint8_t>::Read r = data.read();
	const unsigned char *r_ptr = r.ptr();

	PoolVector<uint8_t>::Write w = dst.data.write();
	unsigned char *w_ptr = w.ptr();

	switch (p_interpolation) {

		case INTERPOLATE_NEAREST: {

			switch (get_format_pixel_size(format)) {
				case 1: _scale_nearest<1>(r_ptr, w_ptr, width, height, p_width, p_height); break;
				case 2: _scale_nearest<2>(r_ptr, w_ptr, width, height, p_width, p_height); break;
				case 3: _scale_nearest<3>(r_ptr, w_ptr, width, height, p_width, p_height); break;
				case 4: _scale_nearest<4>(r_ptr, w_ptr, width, height, p_width, p_height); break;
			}
		} break;
		case INTERPOLATE_BILINEAR: {

			switch (get_format_pixel_size(format)) {
				case 1: _scale_bilinear<1>(r_ptr, w_ptr, width, height, p_width, p_height); break;
				case 2: _scale_bilinear<2>(r_ptr, w_ptr, width, height, p_width, p_height); break;
				case 3: _scale_bilinear<3>(r_ptr, w_ptr, width, height, p_width, p_height); break;
				case 4: _scale_bilinear<4>(r_ptr, w_ptr, width, height, p_width, p_height); break;
			}

		} break;
		case INTERPOLATE_CUBIC: {

			switch (get_format_pixel_size(format)) {
				case 1: _scale_cubic<1>(r_ptr, w_ptr, width, height, p_width, p_height); break;
				case 2: _scale_cubic<2>(r_ptr, w_ptr, width, height, p_width, p_height); break;
				case 3: _scale_cubic<3>(r_ptr, w_ptr, width, height, p_width, p_height); break;
				case 4: _scale_cubic<4>(r_ptr, w_ptr, width, height, p_width, p_height); break;
			}

		} break;
	}

	r = PoolVector<uint8_t>::Read();
	w = PoolVector<uint8_t>::Write();

	if (mipmaps > 0)
		dst.generate_mipmaps();

	*this = dst;
}
void Image::crop(int p_width, int p_height) {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot crop in indexed, compressed or custom image formats.");
		ERR_FAIL();
	}
	ERR_FAIL_COND(p_width <= 0);
	ERR_FAIL_COND(p_height <= 0);
	ERR_FAIL_COND(p_width > MAX_WIDTH);
	ERR_FAIL_COND(p_height > MAX_HEIGHT);

	/* to save memory, cropping should be done in-place, however, since this function
	   will most likely either not be used much, or in critical areas, for now it wont, because
	   it's a waste of time. */

	if (p_width == width && p_height == height)
		return;

	uint8_t pdata[16]; //largest is 16
	uint32_t pixel_size = get_format_pixel_size(format);

	Image dst(p_width, p_height, 0, format);

	{
		PoolVector<uint8_t>::Read r = data.read();
		PoolVector<uint8_t>::Write w = dst.data.write();

		for (int y = 0; y < p_height; y++) {

			for (int x = 0; x < p_width; x++) {

				if ((x >= width || y >= height)) {
					for (uint32_t i = 0; i < pixel_size; i++)
						pdata[i] = 0;
				} else {
					_get_pixelb(x, y, pixel_size, r.ptr(), pdata);
				}

				dst._put_pixelb(x, y, pixel_size, w.ptr(), pdata);
			}
		}
	}

	if (mipmaps > 0)
		dst.generate_mipmaps();
	*this = dst;
}

void Image::flip_y() {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot flip_y in indexed, compressed or custom image formats.");
		ERR_FAIL();
	}

	bool gm = mipmaps;

	if (gm)
		clear_mipmaps();

	{
		PoolVector<uint8_t>::Write w = data.write();
		uint8_t up[16];
		uint8_t down[16];
		uint32_t pixel_size = get_format_pixel_size(format);

		for (int y = 0; y < height; y++) {

			for (int x = 0; x < width; x++) {

				_get_pixelb(x, y, pixel_size, w.ptr(), up);
				_get_pixelb(x, height - y - 1, pixel_size, w.ptr(), down);

				_put_pixelb(x, height - y - 1, pixel_size, w.ptr(), up);
				_put_pixelb(x, y, pixel_size, w.ptr(), down);
			}
		}
	}

	if (gm)
		generate_mipmaps();
}

void Image::flip_x() {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot flip_x in indexed, compressed or custom image formats.");
		ERR_FAIL();
	}

	bool gm = mipmaps;
	if (gm)
		clear_mipmaps();

	{
		PoolVector<uint8_t>::Write w = data.write();
		uint8_t up[16];
		uint8_t down[16];
		uint32_t pixel_size = get_format_pixel_size(format);

		for (int y = 0; y < height; y++) {

			for (int x = 0; x < width; x++) {

				_get_pixelb(x, y, pixel_size, w.ptr(), up);
				_get_pixelb(width - x - 1, y, pixel_size, w.ptr(), down);

				_put_pixelb(width - x - 1, y, pixel_size, w.ptr(), up);
				_put_pixelb(x, y, pixel_size, w.ptr(), down);
			}
		}
	}

	if (gm)
		generate_mipmaps();
}

int Image::_get_dst_image_size(int p_width, int p_height, Format p_format, int &r_mipmaps, int p_mipmaps) {

	int size = 0;
	int w = p_width;
	int h = p_height;
	int mm = 0;

	int pixsize = get_format_pixel_size(p_format);
	int pixshift = get_format_pixel_rshift(p_format);
	int minw, minh;
	get_format_min_pixel_size(p_format, minw, minh);

	while (true) {

		int s = w * h;
		s *= pixsize;
		s >>= pixshift;

		size += s;

		if (p_mipmaps >= 0 && mm == p_mipmaps)
			break;

		if (p_mipmaps >= 0) {

			w = MAX(minw, w >> 1);
			h = MAX(minh, h >> 1);
		} else {
			if (w == minw && h == minh)
				break;
			w = MAX(minw, w >> 1);
			h = MAX(minh, h >> 1);
		}
		mm++;
	};

	r_mipmaps = mm;
	return size;
}

bool Image::_can_modify(Format p_format) const {

	return p_format < FORMAT_RGB565;
}

template <int CC>
static void _generate_po2_mipmap(const uint8_t *p_src, uint8_t *p_dst, uint32_t p_width, uint32_t p_height) {

	//fast power of 2 mipmap generation
	uint32_t dst_w = p_width >> 1;
	uint32_t dst_h = p_height >> 1;

	for (uint32_t i = 0; i < dst_h; i++) {

		const uint8_t *rup_ptr = &p_src[i * 2 * p_width * CC];
		const uint8_t *rdown_ptr = rup_ptr + p_width * CC;
		uint8_t *dst_ptr = &p_dst[i * dst_w * CC];
		uint32_t count = dst_w;

		while (count--) {

			for (int j = 0; j < CC; j++) {

				uint16_t val = 0;
				val += rup_ptr[j];
				val += rup_ptr[j + CC];
				val += rdown_ptr[j];
				val += rdown_ptr[j + CC];
				dst_ptr[j] = val >> 2;
			}

			dst_ptr += CC;
			rup_ptr += CC * 2;
			rdown_ptr += CC * 2;
		}
	}
}

void Image::expand_x2_hq2x() {

	ERR_FAIL_COND(!_can_modify(format));

	Format current = format;
	bool mm = has_mipmaps();
	if (mm) {
		clear_mipmaps();
	}

	if (current != FORMAT_RGBA8)
		convert(FORMAT_RGBA8);

	PoolVector<uint8_t> dest;
	dest.resize(width * 2 * height * 2 * 4);

	{
		PoolVector<uint8_t>::Read r = data.read();
		PoolVector<uint8_t>::Write w = dest.write();

		hq2x_resize((const uint32_t *)r.ptr(), width, height, (uint32_t *)w.ptr());
	}

	width *= 2;
	height *= 2;
	data = dest;

	if (current != FORMAT_RGBA8)
		convert(current);

	if (mipmaps) {
		generate_mipmaps();
	}
}

void Image::shrink_x2() {

	ERR_FAIL_COND(data.size() == 0);

	if (mipmaps) {

		//just use the lower mipmap as base and copy all
		PoolVector<uint8_t> new_img;

		int ofs = get_mipmap_offset(1);

		int new_size = data.size() - ofs;
		new_img.resize(new_size);

		{
			PoolVector<uint8_t>::Write w = new_img.write();
			PoolVector<uint8_t>::Read r = data.read();

			copymem(w.ptr(), &r[ofs], new_size);
		}

		width /= 2;
		height /= 2;
		data = new_img;

	} else {

		PoolVector<uint8_t> new_img;

		ERR_FAIL_COND(!_can_modify(format));
		int ps = get_format_pixel_size(format);
		new_img.resize((width / 2) * (height / 2) * ps);

		{
			PoolVector<uint8_t>::Write w = new_img.write();
			PoolVector<uint8_t>::Read r = data.read();

			switch (format) {

				case FORMAT_L8:
				case FORMAT_R8: _generate_po2_mipmap<1>(r.ptr(), w.ptr(), width, height); break;
				case FORMAT_LA8: _generate_po2_mipmap<2>(r.ptr(), w.ptr(), width, height); break;
				case FORMAT_RG8: _generate_po2_mipmap<2>(r.ptr(), w.ptr(), width, height); break;
				case FORMAT_RGB8: _generate_po2_mipmap<3>(r.ptr(), w.ptr(), width, height); break;
				case FORMAT_RGBA8: _generate_po2_mipmap<4>(r.ptr(), w.ptr(), width, height); break;
				default: {}
			}
		}

		width /= 2;
		height /= 2;
		data = new_img;
	}
}

Error Image::generate_mipmaps() {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot generate mipmaps in indexed, compressed or custom image formats.");
		ERR_FAIL_V(ERR_UNAVAILABLE);
	}

	ERR_FAIL_COND_V(width == 0 || height == 0, ERR_UNCONFIGURED);

	int mmcount;

	int size = _get_dst_image_size(width, height, format, mmcount);

	data.resize(size);
	print_line("to gen mipmaps w " + itos(width) + " h " + itos(height) + " format " + get_format_name(format) + " mipmaps " + itos(mmcount) + " new size is: " + itos(size));

	PoolVector<uint8_t>::Write wp = data.write();

	if (nearest_power_of_2(width) == uint32_t(width) && nearest_power_of_2(height) == uint32_t(height)) {
		//use fast code for powers of 2
		int prev_ofs = 0;
		int prev_h = height;
		int prev_w = width;

		for (int i = 1; i < mmcount; i++) {

			int ofs, w, h;
			_get_mipmap_offset_and_size(i, ofs, w, h);

			switch (format) {

				case FORMAT_L8:
				case FORMAT_R8: _generate_po2_mipmap<1>(&wp[prev_ofs], &wp[ofs], prev_w, prev_h); break;
				case FORMAT_LA8:
				case FORMAT_RG8: _generate_po2_mipmap<2>(&wp[prev_ofs], &wp[ofs], prev_w, prev_h); break;
				case FORMAT_RGB8: _generate_po2_mipmap<3>(&wp[prev_ofs], &wp[ofs], prev_w, prev_h); break;
				case FORMAT_RGBA8: _generate_po2_mipmap<4>(&wp[prev_ofs], &wp[ofs], prev_w, prev_h); break;
				default: {}
			}

			prev_ofs = ofs;
			prev_w = w;
			prev_h = h;
		}

	} else {
		//use slow code..

		//use bilinear filtered code for non powers of 2
		int prev_ofs = 0;
		int prev_h = height;
		int prev_w = width;

		for (int i = 1; i < mmcount; i++) {

			int ofs, w, h;
			_get_mipmap_offset_and_size(i, ofs, w, h);

			switch (format) {

				case FORMAT_L8:
				case FORMAT_R8: _scale_bilinear<1>(&wp[prev_ofs], &wp[ofs], prev_w, prev_h, w, h); break;
				case FORMAT_LA8:
				case FORMAT_RG8: _scale_bilinear<2>(&wp[prev_ofs], &wp[ofs], prev_w, prev_h, w, h); break;
				case FORMAT_RGB8: _scale_bilinear<3>(&wp[prev_ofs], &wp[ofs], prev_w, prev_h, w, h); break;
				case FORMAT_RGBA8: _scale_bilinear<4>(&wp[prev_ofs], &wp[ofs], prev_w, prev_h, w, h); break;
				default: {}
			}

			prev_ofs = ofs;
			prev_w = w;
			prev_h = h;
		}
	}

	mipmaps = true;

	return OK;
}

void Image::clear_mipmaps() {

	if (!mipmaps)
		return;

	if (empty())
		return;

	int ofs, w, h;
	_get_mipmap_offset_and_size(1, ofs, w, h);
	data.resize(ofs);

	mipmaps = false;
}

bool Image::empty() const {

	return (data.size() == 0);
}

PoolVector<uint8_t> Image::get_data() const {

	return data;
}

void Image::create(int p_width, int p_height, bool p_use_mipmaps, Format p_format) {

	int mm = 0;
	int size = _get_dst_image_size(p_width, p_height, p_format, mm, p_use_mipmaps ? -1 : 0);
	data.resize(size);
	{
		PoolVector<uint8_t>::Write w = data.write();
		zeromem(w.ptr(), size);
	}

	width = p_width;
	height = p_height;
	mipmaps = p_use_mipmaps;
	format = p_format;
}

void Image::create(int p_width, int p_height, bool p_use_mipmaps, Format p_format, const PoolVector<uint8_t> &p_data) {

	ERR_FAIL_INDEX(p_width - 1, MAX_WIDTH);
	ERR_FAIL_INDEX(p_height - 1, MAX_HEIGHT);

	int mm;
	int size = _get_dst_image_size(p_width, p_height, p_format, mm, p_use_mipmaps ? -1 : 0);

	if (size != p_data.size()) {
		ERR_EXPLAIN("Expected data size of " + itos(size) + " in Image::create()");
		ERR_FAIL_COND(p_data.size() != size);
	}

	height = p_height;
	width = p_width;
	format = p_format;
	data = p_data;
	mipmaps = p_use_mipmaps;
}

void Image::create(const char **p_xpm) {

	int size_width, size_height;
	int pixelchars = 0;
	mipmaps = false;
	bool has_alpha = false;

	enum Status {
		READING_HEADER,
		READING_COLORS,
		READING_PIXELS,
		DONE
	};

	Status status = READING_HEADER;
	int line = 0;

	HashMap<String, Color> colormap;
	int colormap_size;
	uint32_t pixel_size;
	PoolVector<uint8_t>::Write w;

	while (status != DONE) {

		const char *line_ptr = p_xpm[line];

		switch (status) {

			case READING_HEADER: {

				String line_str = line_ptr;
				line_str.replace("\t", " ");

				size_width = line_str.get_slicec(' ', 0).to_int();
				size_height = line_str.get_slicec(' ', 1).to_int();
				colormap_size = line_str.get_slicec(' ', 2).to_int();
				pixelchars = line_str.get_slicec(' ', 3).to_int();
				ERR_FAIL_COND(colormap_size > 32766);
				ERR_FAIL_COND(pixelchars > 5);
				ERR_FAIL_COND(size_width > 32767);
				ERR_FAIL_COND(size_height > 32767);
				status = READING_COLORS;
			} break;
			case READING_COLORS: {

				String colorstring;
				for (int i = 0; i < pixelchars; i++) {

					colorstring += *line_ptr;
					line_ptr++;
				}
				//skip spaces
				while (*line_ptr == ' ' || *line_ptr == '\t' || *line_ptr == 0) {
					if (*line_ptr == 0)
						break;
					line_ptr++;
				}
				if (*line_ptr == 'c') {

					line_ptr++;
					while (*line_ptr == ' ' || *line_ptr == '\t' || *line_ptr == 0) {
						if (*line_ptr == 0)
							break;
						line_ptr++;
					}

					if (*line_ptr == '#') {
						line_ptr++;
						uint8_t col_r;
						uint8_t col_g;
						uint8_t col_b;
						//uint8_t col_a=255;

						for (int i = 0; i < 6; i++) {

							char v = line_ptr[i];

							if (v >= '0' && v <= '9')
								v -= '0';
							else if (v >= 'A' && v <= 'F')
								v = (v - 'A') + 10;
							else if (v >= 'a' && v <= 'f')
								v = (v - 'a') + 10;
							else
								break;

							switch (i) {
								case 0: col_r = v << 4; break;
								case 1: col_r |= v; break;
								case 2: col_g = v << 4; break;
								case 3: col_g |= v; break;
								case 4: col_b = v << 4; break;
								case 5: col_b |= v; break;
							};
						}

						// magenta mask
						if (col_r == 255 && col_g == 0 && col_b == 255) {

							colormap[colorstring] = Color(0, 0, 0, 0);
							has_alpha = true;
						} else {

							colormap[colorstring] = Color(col_r / 255.0, col_g / 255.0, col_b / 255.0, 1.0);
						}
					}
				}
				if (line == colormap_size) {

					status = READING_PIXELS;
					create(size_width, size_height, 0, has_alpha ? FORMAT_RGBA8 : FORMAT_RGB8);
					w = data.write();
					pixel_size = has_alpha ? 4 : 3;
				}
			} break;
			case READING_PIXELS: {

				int y = line - colormap_size - 1;
				for (int x = 0; x < size_width; x++) {

					char pixelstr[6] = { 0, 0, 0, 0, 0, 0 };
					for (int i = 0; i < pixelchars; i++)
						pixelstr[i] = line_ptr[x * pixelchars + i];

					Color *colorptr = colormap.getptr(pixelstr);
					ERR_FAIL_COND(!colorptr);
					uint8_t pixel[4];
					for (uint32_t i = 0; i < pixel_size; i++) {
						pixel[i] = CLAMP((*colorptr)[i] * 255, 0, 255);
					}
					_put_pixelb(x, y, pixel_size, w.ptr(), pixel);
				}

				if (y == (size_height - 1))
					status = DONE;
			} break;
			default: {}
		}

		line++;
	}
}
#define DETECT_ALPHA_MAX_TRESHOLD 254
#define DETECT_ALPHA_MIN_TRESHOLD 2

#define DETECT_ALPHA(m_value)                         \
	{                                                 \
		uint8_t value = m_value;                      \
		if (value < DETECT_ALPHA_MIN_TRESHOLD)        \
			bit = true;                               \
		else if (value < DETECT_ALPHA_MAX_TRESHOLD) { \
                                                      \
			detected = true;                          \
			break;                                    \
		}                                             \
	}

#define DETECT_NON_ALPHA(m_value) \
	{                             \
		uint8_t value = m_value;  \
		if (value > 0) {          \
                                  \
			detected = true;      \
			break;                \
		}                         \
	}

bool Image::is_invisible() const {

	if (format == FORMAT_L8 ||
			format == FORMAT_RGB8 || format == FORMAT_RG8)
		return false;

	int len = data.size();

	if (len == 0)
		return true;

	int w, h;
	_get_mipmap_offset_and_size(1, len, w, h);

	PoolVector<uint8_t>::Read r = data.read();
	const unsigned char *data_ptr = r.ptr();

	bool detected = false;

	switch (format) {

		case FORMAT_LA8: {

			for (int i = 0; i < (len >> 1); i++) {
				DETECT_NON_ALPHA(data_ptr[(i << 1) + 1]);
			}

		} break;
		case FORMAT_RGBA8: {

			for (int i = 0; i < (len >> 2); i++) {
				DETECT_NON_ALPHA(data_ptr[(i << 2) + 3])
			}

		} break;

		case FORMAT_PVRTC2A:
		case FORMAT_PVRTC4A:
		case FORMAT_DXT3:
		case FORMAT_DXT5: {
			detected = true;
		} break;
		default: {}
	}

	return !detected;
}

Image::AlphaMode Image::detect_alpha() const {

	int len = data.size();

	if (len == 0)
		return ALPHA_NONE;

	int w, h;
	_get_mipmap_offset_and_size(1, len, w, h);

	PoolVector<uint8_t>::Read r = data.read();
	const unsigned char *data_ptr = r.ptr();

	bool bit = false;
	bool detected = false;

	switch (format) {

		case FORMAT_LA8: {

			for (int i = 0; i < (len >> 1); i++) {
				DETECT_ALPHA(data_ptr[(i << 1) + 1]);
			}

		} break;
		case FORMAT_RGBA8: {

			for (int i = 0; i < (len >> 2); i++) {
				DETECT_ALPHA(data_ptr[(i << 2) + 3])
			}

		} break;
		case FORMAT_PVRTC2A:
		case FORMAT_PVRTC4A:
		case FORMAT_DXT3:
		case FORMAT_DXT5: {
			detected = true;
		} break;
		default: {}
	}

	if (detected)
		return ALPHA_BLEND;
	else if (bit)
		return ALPHA_BIT;
	else
		return ALPHA_NONE;
}

Error Image::load(const String &p_path) {

	return ImageLoader::load_image(p_path, this);
}

Error Image::save_png(const String &p_path) {

	if (save_png_func == NULL)
		return ERR_UNAVAILABLE;

	return save_png_func(p_path, *this);
}

bool Image::operator==(const Image &p_image) const {

	if (data.size() == 0 && p_image.data.size() == 0)
		return true;
	PoolVector<uint8_t>::Read r = data.read();
	PoolVector<uint8_t>::Read pr = p_image.data.read();

	return r.ptr() == pr.ptr();
}

int Image::get_image_data_size(int p_width, int p_height, Format p_format, int p_mipmaps) {

	int mm;
	return _get_dst_image_size(p_width, p_height, p_format, mm, p_mipmaps);
}

int Image::get_image_required_mipmaps(int p_width, int p_height, Format p_format) {

	int mm;
	_get_dst_image_size(p_width, p_height, p_format, mm, -1);
	return mm;
}

Error Image::_decompress_bc() {

	int wd = width, ht = height;
	if (wd % 4 != 0) {
		wd += 4 - (wd % 4);
	}
	if (ht % 4 != 0) {
		ht += 4 - (ht % 4);
	}

	int mm;
	int size = _get_dst_image_size(wd, ht, FORMAT_RGBA8, mm);

	PoolVector<uint8_t> newdata;
	newdata.resize(size);

	PoolVector<uint8_t>::Write w = newdata.write();
	PoolVector<uint8_t>::Read r = data.read();

	int rofs = 0;
	int wofs = 0;

	//print_line("width: "+itos(wd)+" height: "+itos(ht));

	for (int i = 0; i <= mm; i++) {

		switch (format) {

			case FORMAT_DXT1: {

				int len = (wd * ht) / 16;
				uint8_t *dst = &w[wofs];

				uint32_t ofs_table[16];
				for (int x = 0; x < 4; x++) {

					for (int y = 0; y < 4; y++) {

						ofs_table[15 - (y * 4 + (3 - x))] = (x + y * wd) * 4;
					}
				}

				for (int j = 0; j < len; j++) {

					const uint8_t *src = &r[rofs + j * 8];
					uint16_t col_a = src[1];
					col_a <<= 8;
					col_a |= src[0];
					uint16_t col_b = src[3];
					col_b <<= 8;
					col_b |= src[2];

					uint8_t table[4][4] = {
						{ uint8_t((col_a >> 11) << 3), uint8_t(((col_a >> 5) & 0x3f) << 2), uint8_t(((col_a)&0x1f) << 3), 255 },
						{ uint8_t((col_b >> 11) << 3), uint8_t(((col_b >> 5) & 0x3f) << 2), uint8_t(((col_b)&0x1f) << 3), 255 },
						{ 0, 0, 0, 255 },
						{ 0, 0, 0, 255 }
					};

					if (col_a < col_b) {
						//punchrough
						table[2][0] = (int(table[0][0]) + int(table[1][0])) >> 1;
						table[2][1] = (int(table[0][1]) + int(table[1][1])) >> 1;
						table[2][2] = (int(table[0][2]) + int(table[1][2])) >> 1;
						table[3][3] = 0; //premul alpha black
					} else {
						//gradient
						table[2][0] = (int(table[0][0]) * 2 + int(table[1][0])) / 3;
						table[2][1] = (int(table[0][1]) * 2 + int(table[1][1])) / 3;
						table[2][2] = (int(table[0][2]) * 2 + int(table[1][2])) / 3;
						table[3][0] = (int(table[0][0]) + int(table[1][0]) * 2) / 3;
						table[3][1] = (int(table[0][1]) + int(table[1][1]) * 2) / 3;
						table[3][2] = (int(table[0][2]) + int(table[1][2]) * 2) / 3;
					}

					uint32_t block = src[4];
					block <<= 8;
					block |= src[5];
					block <<= 8;
					block |= src[6];
					block <<= 8;
					block |= src[7];

					int y = (j / (wd / 4)) * 4;
					int x = (j % (wd / 4)) * 4;
					int pixofs = (y * wd + x) * 4;

					for (int k = 0; k < 16; k++) {
						int idx = pixofs + ofs_table[k];
						dst[idx + 0] = table[block & 0x3][0];
						dst[idx + 1] = table[block & 0x3][1];
						dst[idx + 2] = table[block & 0x3][2];
						dst[idx + 3] = table[block & 0x3][3];
						block >>= 2;
					}
				}

				rofs += len * 8;
				wofs += wd * ht * 4;

				wd /= 2;
				ht /= 2;

			} break;
			case FORMAT_DXT3: {

				int len = (wd * ht) / 16;
				uint8_t *dst = &w[wofs];

				uint32_t ofs_table[16];
				for (int x = 0; x < 4; x++) {

					for (int y = 0; y < 4; y++) {

						ofs_table[15 - (y * 4 + (3 - x))] = (x + y * wd) * 4;
					}
				}

				for (int j = 0; j < len; j++) {

					const uint8_t *src = &r[rofs + j * 16];

					uint64_t ablock = src[1];
					ablock <<= 8;
					ablock |= src[0];
					ablock <<= 8;
					ablock |= src[3];
					ablock <<= 8;
					ablock |= src[2];
					ablock <<= 8;
					ablock |= src[5];
					ablock <<= 8;
					ablock |= src[4];
					ablock <<= 8;
					ablock |= src[7];
					ablock <<= 8;
					ablock |= src[6];

					uint16_t col_a = src[8 + 1];
					col_a <<= 8;
					col_a |= src[8 + 0];
					uint16_t col_b = src[8 + 3];
					col_b <<= 8;
					col_b |= src[8 + 2];

					uint8_t table[4][4] = {
						{ uint8_t((col_a >> 11) << 3), uint8_t(((col_a >> 5) & 0x3f) << 2), uint8_t(((col_a)&0x1f) << 3), 255 },
						{ uint8_t((col_b >> 11) << 3), uint8_t(((col_b >> 5) & 0x3f) << 2), uint8_t(((col_b)&0x1f) << 3), 255 },

						{ 0, 0, 0, 255 },
						{ 0, 0, 0, 255 }
					};

					//always gradient
					table[2][0] = (int(table[0][0]) * 2 + int(table[1][0])) / 3;
					table[2][1] = (int(table[0][1]) * 2 + int(table[1][1])) / 3;
					table[2][2] = (int(table[0][2]) * 2 + int(table[1][2])) / 3;
					table[3][0] = (int(table[0][0]) + int(table[1][0]) * 2) / 3;
					table[3][1] = (int(table[0][1]) + int(table[1][1]) * 2) / 3;
					table[3][2] = (int(table[0][2]) + int(table[1][2]) * 2) / 3;

					uint32_t block = src[4 + 8];
					block <<= 8;
					block |= src[5 + 8];
					block <<= 8;
					block |= src[6 + 8];
					block <<= 8;
					block |= src[7 + 8];

					int y = (j / (wd / 4)) * 4;
					int x = (j % (wd / 4)) * 4;
					int pixofs = (y * wd + x) * 4;

					for (int k = 0; k < 16; k++) {
						uint8_t alpha = ablock & 0xf;
						alpha = int(alpha) * 255 / 15; //right way for alpha
						int idx = pixofs + ofs_table[k];
						dst[idx + 0] = table[block & 0x3][0];
						dst[idx + 1] = table[block & 0x3][1];
						dst[idx + 2] = table[block & 0x3][2];
						dst[idx + 3] = alpha;
						block >>= 2;
						ablock >>= 4;
					}
				}

				rofs += len * 16;
				wofs += wd * ht * 4;

				wd /= 2;
				ht /= 2;

			} break;
			case FORMAT_DXT5: {

				int len = (wd * ht) / 16;
				uint8_t *dst = &w[wofs];

				uint32_t ofs_table[16];
				for (int x = 0; x < 4; x++) {

					for (int y = 0; y < 4; y++) {

						ofs_table[15 - (y * 4 + (3 - x))] = (x + y * wd) * 4;
					}
				}

				for (int j = 0; j < len; j++) {

					const uint8_t *src = &r[rofs + j * 16];

					uint8_t a_start = src[1];
					uint8_t a_end = src[0];

					uint64_t ablock = src[3];
					ablock <<= 8;
					ablock |= src[2];
					ablock <<= 8;
					ablock |= src[5];
					ablock <<= 8;
					ablock |= src[4];
					ablock <<= 8;
					ablock |= src[7];
					ablock <<= 8;
					ablock |= src[6];

					uint8_t atable[8];

					if (a_start > a_end) {

						atable[0] = (int(a_start) * 7 + int(a_end) * 0) / 7;
						atable[1] = (int(a_start) * 6 + int(a_end) * 1) / 7;
						atable[2] = (int(a_start) * 5 + int(a_end) * 2) / 7;
						atable[3] = (int(a_start) * 4 + int(a_end) * 3) / 7;
						atable[4] = (int(a_start) * 3 + int(a_end) * 4) / 7;
						atable[5] = (int(a_start) * 2 + int(a_end) * 5) / 7;
						atable[6] = (int(a_start) * 1 + int(a_end) * 6) / 7;
						atable[7] = (int(a_start) * 0 + int(a_end) * 7) / 7;
					} else {

						atable[0] = (int(a_start) * 5 + int(a_end) * 0) / 5;
						atable[1] = (int(a_start) * 4 + int(a_end) * 1) / 5;
						atable[2] = (int(a_start) * 3 + int(a_end) * 2) / 5;
						atable[3] = (int(a_start) * 2 + int(a_end) * 3) / 5;
						atable[4] = (int(a_start) * 1 + int(a_end) * 4) / 5;
						atable[5] = (int(a_start) * 0 + int(a_end) * 5) / 5;
						atable[6] = 0;
						atable[7] = 255;
					}

					uint16_t col_a = src[8 + 1];
					col_a <<= 8;
					col_a |= src[8 + 0];
					uint16_t col_b = src[8 + 3];
					col_b <<= 8;
					col_b |= src[8 + 2];

					uint8_t table[4][4] = {
						{ uint8_t((col_a >> 11) << 3), uint8_t(((col_a >> 5) & 0x3f) << 2), uint8_t(((col_a)&0x1f) << 3), 255 },
						{ uint8_t((col_b >> 11) << 3), uint8_t(((col_b >> 5) & 0x3f) << 2), uint8_t(((col_b)&0x1f) << 3), 255 },

						{ 0, 0, 0, 255 },
						{ 0, 0, 0, 255 }
					};

					//always gradient
					table[2][0] = (int(table[0][0]) * 2 + int(table[1][0])) / 3;
					table[2][1] = (int(table[0][1]) * 2 + int(table[1][1])) / 3;
					table[2][2] = (int(table[0][2]) * 2 + int(table[1][2])) / 3;
					table[3][0] = (int(table[0][0]) + int(table[1][0]) * 2) / 3;
					table[3][1] = (int(table[0][1]) + int(table[1][1]) * 2) / 3;
					table[3][2] = (int(table[0][2]) + int(table[1][2]) * 2) / 3;

					uint32_t block = src[4 + 8];
					block <<= 8;
					block |= src[5 + 8];
					block <<= 8;
					block |= src[6 + 8];
					block <<= 8;
					block |= src[7 + 8];

					int y = (j / (wd / 4)) * 4;
					int x = (j % (wd / 4)) * 4;
					int pixofs = (y * wd + x) * 4;

					for (int k = 0; k < 16; k++) {
						uint8_t alpha = ablock & 0x7;
						int idx = pixofs + ofs_table[k];
						dst[idx + 0] = table[block & 0x3][0];
						dst[idx + 1] = table[block & 0x3][1];
						dst[idx + 2] = table[block & 0x3][2];
						dst[idx + 3] = atable[alpha];
						block >>= 2;
						ablock >>= 3;
					}
				}

				rofs += len * 16;
				wofs += wd * ht * 4;

				wd /= 2;
				ht /= 2;

			} break;
			default: {}
		}
	}

	w = PoolVector<uint8_t>::Write();
	r = PoolVector<uint8_t>::Read();

	data = newdata;
	format = FORMAT_RGBA8;
	if (wd != width || ht != height) {

		SWAP(width, wd);
		SWAP(height, ht);
		crop(wd, ht);
	}

	return OK;
}

bool Image::is_compressed() const {
	return format >= FORMAT_RGB565;
}

Image Image::decompressed() const {

	Image img = *this;
	img.decompress();
	return img;
}

Error Image::decompress() {

	if (format >= FORMAT_DXT1 && format <= FORMAT_ATI2)
		_decompress_bc(); //_image_decompress_bc(this);
	else if (format >= FORMAT_PVRTC2 && format <= FORMAT_PVRTC4A && _image_decompress_pvrtc)
		_image_decompress_pvrtc(this);
	else if (format == FORMAT_ETC && _image_decompress_etc)
		_image_decompress_etc(this);
	else if (format >= FORMAT_ETC2_R11 && format <= FORMAT_ETC2_RGB8A1 && _image_decompress_etc)
		_image_decompress_etc2(this);
	else
		return ERR_UNAVAILABLE;
	return OK;
}

Error Image::compress(CompressMode p_mode) {

	switch (p_mode) {

		case COMPRESS_16BIT: {

			//ERR_FAIL_COND_V(!_image_compress_bc_func, ERR_UNAVAILABLE);
			//_image_compress_bc_func(this);
		} break;
		case COMPRESS_S3TC: {

			ERR_FAIL_COND_V(!_image_compress_bc_func, ERR_UNAVAILABLE);
			_image_compress_bc_func(this);
		} break;
		case COMPRESS_PVRTC2: {

			ERR_FAIL_COND_V(!_image_compress_pvrtc2_func, ERR_UNAVAILABLE);
			_image_compress_pvrtc2_func(this);
		} break;
		case COMPRESS_PVRTC4: {

			ERR_FAIL_COND_V(!_image_compress_pvrtc4_func, ERR_UNAVAILABLE);
			_image_compress_pvrtc4_func(this);
		} break;
		case COMPRESS_ETC: {

			ERR_FAIL_COND_V(!_image_compress_etc_func, ERR_UNAVAILABLE);
			_image_compress_etc_func(this);
		} break;
		case COMPRESS_ETC2: {

			ERR_FAIL_COND_V(!_image_compress_etc_func, ERR_UNAVAILABLE);
			_image_compress_etc_func(this);
		} break;
	}

	return OK;
}

Image Image::compressed(int p_mode) {

	Image ret = *this;
	ret.compress((Image::CompressMode)p_mode);

	return ret;
}

Image::Image(const char **p_xpm) {

	width = 0;
	height = 0;
	mipmaps = false;
	format = FORMAT_L8;

	create(p_xpm);
}

Image::Image(int p_width, int p_height, bool p_use_mipmaps, Format p_format) {

	width = 0;
	height = 0;
	mipmaps = p_use_mipmaps;
	format = FORMAT_L8;

	create(p_width, p_height, p_use_mipmaps, p_format);
}

Image::Image(int p_width, int p_height, bool p_mipmaps, Format p_format, const PoolVector<uint8_t> &p_data) {

	width = 0;
	height = 0;
	mipmaps = p_mipmaps;
	format = FORMAT_L8;

	create(p_width, p_height, p_mipmaps, p_format, p_data);
}

Rect2 Image::get_used_rect() const {

	if (format != FORMAT_LA8 && format != FORMAT_RGBA8)
		return Rect2(Point2(), Size2(width, height));

	int len = data.size();

	if (len == 0)
		return Rect2();

	//int data_size = len;
	PoolVector<uint8_t>::Read r = data.read();
	const unsigned char *rptr = r.ptr();

	int ps = format == FORMAT_LA8 ? 2 : 4;
	int minx = 0xFFFFFF, miny = 0xFFFFFFF;
	int maxx = -1, maxy = -1;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {

			bool opaque = rptr[(j * width + i) * ps + (ps - 1)] > 2;
			if (!opaque)
				continue;
			if (i > maxx)
				maxx = i;
			if (j > maxy)
				maxy = j;
			if (i < minx)
				minx = i;
			if (j < miny)
				miny = j;
		}
	}

	if (maxx == -1)
		return Rect2();
	else
		return Rect2(minx, miny, maxx - minx + 1, maxy - miny + 1);
}

Image Image::get_rect(const Rect2 &p_area) const {

	Image img(p_area.size.x, p_area.size.y, mipmaps, format);
	img.blit_rect(*this, p_area, Point2(0, 0));

	return img;
}

void Image::blit_rect(const Image &p_src, const Rect2 &p_src_rect, const Point2 &p_dest) {

	int dsize = data.size();
	int srcdsize = p_src.data.size();
	ERR_FAIL_COND(dsize == 0);
	ERR_FAIL_COND(srcdsize == 0);
	ERR_FAIL_COND(format != p_src.format);

	Rect2i local_src_rect = Rect2i(0, 0, width, height).clip(Rect2i(p_dest + p_src_rect.pos, p_src_rect.size));

	if (local_src_rect.size.x <= 0 || local_src_rect.size.y <= 0)
		return;
	Rect2i src_rect(p_src_rect.pos + (local_src_rect.pos - p_dest), local_src_rect.size);

	PoolVector<uint8_t>::Write wp = data.write();
	uint8_t *dst_data_ptr = wp.ptr();

	PoolVector<uint8_t>::Read rp = p_src.data.read();
	const uint8_t *src_data_ptr = rp.ptr();

	int pixel_size = get_format_pixel_size(format);

	for (int i = 0; i < src_rect.size.y; i++) {

		for (int j = 0; j < src_rect.size.x; j++) {

			int src_x = src_rect.pos.x + j;
			int src_y = src_rect.pos.y + i;

			int dst_x = local_src_rect.pos.x + j;
			int dst_y = local_src_rect.pos.y + i;

			const uint8_t *src = &src_data_ptr[(src_y * p_src.width + src_x) * pixel_size];
			uint8_t *dst = &dst_data_ptr[(dst_y * width + dst_x) * pixel_size];

			for (int k = 0; k < pixel_size; k++) {
				dst[k] = src[k];
			}
		}
	}
}

Image (*Image::_png_mem_loader_func)(const uint8_t *, int) = NULL;
Image (*Image::_jpg_mem_loader_func)(const uint8_t *, int) = NULL;

void (*Image::_image_compress_bc_func)(Image *) = NULL;
void (*Image::_image_compress_pvrtc2_func)(Image *) = NULL;
void (*Image::_image_compress_pvrtc4_func)(Image *) = NULL;
void (*Image::_image_compress_etc_func)(Image *) = NULL;
void (*Image::_image_compress_etc2_func)(Image *) = NULL;
void (*Image::_image_decompress_pvrtc)(Image *) = NULL;
void (*Image::_image_decompress_bc)(Image *) = NULL;
void (*Image::_image_decompress_etc)(Image *) = NULL;
void (*Image::_image_decompress_etc2)(Image *) = NULL;

PoolVector<uint8_t> (*Image::lossy_packer)(const Image &, float) = NULL;
Image (*Image::lossy_unpacker)(const PoolVector<uint8_t> &) = NULL;
PoolVector<uint8_t> (*Image::lossless_packer)(const Image &) = NULL;
Image (*Image::lossless_unpacker)(const PoolVector<uint8_t> &) = NULL;

void Image::set_compress_bc_func(void (*p_compress_func)(Image *)) {

	_image_compress_bc_func = p_compress_func;
}

void Image::normalmap_to_xy() {

	convert(Image::FORMAT_RGBA8);

	{
		int len = data.size() / 4;
		PoolVector<uint8_t>::Write wp = data.write();
		unsigned char *data_ptr = wp.ptr();

		for (int i = 0; i < len; i++) {

			data_ptr[(i << 2) + 3] = data_ptr[(i << 2) + 0]; //x to w
			data_ptr[(i << 2) + 0] = data_ptr[(i << 2) + 1]; //y to xz
			data_ptr[(i << 2) + 2] = data_ptr[(i << 2) + 1];
		}
	}

	convert(Image::FORMAT_LA8);
}

void Image::srgb_to_linear() {

	if (data.size() == 0)
		return;

	static const uint8_t srgb2lin[256] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 22, 22, 23, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 31, 31, 32, 33, 33, 34, 35, 36, 36, 37, 38, 38, 39, 40, 41, 42, 42, 43, 44, 45, 46, 47, 47, 48, 49, 50, 51, 52, 53, 54, 55, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 92, 93, 94, 95, 97, 98, 99, 101, 102, 103, 105, 106, 107, 109, 110, 112, 113, 114, 116, 117, 119, 120, 122, 123, 125, 126, 128, 129, 131, 132, 134, 135, 137, 139, 140, 142, 144, 145, 147, 148, 150, 152, 153, 155, 157, 159, 160, 162, 164, 166, 167, 169, 171, 173, 175, 176, 178, 180, 182, 184, 186, 188, 190, 192, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 218, 220, 222, 224, 226, 228, 230, 232, 235, 237, 239, 241, 243, 245, 248, 250, 252 };

	ERR_FAIL_COND(format != FORMAT_RGB8 && format != FORMAT_RGBA8);

	if (format == FORMAT_RGBA8) {

		int len = data.size() / 4;
		PoolVector<uint8_t>::Write wp = data.write();
		unsigned char *data_ptr = wp.ptr();

		for (int i = 0; i < len; i++) {

			data_ptr[(i << 2) + 0] = srgb2lin[data_ptr[(i << 2) + 0]];
			data_ptr[(i << 2) + 1] = srgb2lin[data_ptr[(i << 2) + 1]];
			data_ptr[(i << 2) + 2] = srgb2lin[data_ptr[(i << 2) + 2]];
		}

	} else if (format == FORMAT_RGB8) {

		int len = data.size() / 3;
		PoolVector<uint8_t>::Write wp = data.write();
		unsigned char *data_ptr = wp.ptr();

		for (int i = 0; i < len; i++) {

			data_ptr[(i * 3) + 0] = srgb2lin[data_ptr[(i * 3) + 0]];
			data_ptr[(i * 3) + 1] = srgb2lin[data_ptr[(i * 3) + 1]];
			data_ptr[(i * 3) + 2] = srgb2lin[data_ptr[(i * 3) + 2]];
		}
	}
}

void Image::premultiply_alpha() {

	if (data.size() == 0)
		return;

	if (format != FORMAT_RGBA8)
		return; //not needed

	PoolVector<uint8_t>::Write wp = data.write();
	unsigned char *data_ptr = wp.ptr();

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			uint8_t *ptr = &data_ptr[(i * width + j) * 4];

			ptr[0] = (uint16_t(ptr[0]) * uint16_t(ptr[3])) >> 8;
			ptr[1] = (uint16_t(ptr[1]) * uint16_t(ptr[3])) >> 8;
			ptr[2] = (uint16_t(ptr[2]) * uint16_t(ptr[3])) >> 8;
		}
	}
}

void Image::fix_alpha_edges() {

	if (data.size() == 0)
		return;

	if (format != FORMAT_RGBA8)
		return; //not needed

	PoolVector<uint8_t> dcopy = data;
	PoolVector<uint8_t>::Read rp = dcopy.read();
	const uint8_t *srcptr = rp.ptr();

	PoolVector<uint8_t>::Write wp = data.write();
	unsigned char *data_ptr = wp.ptr();

	const int max_radius = 4;
	const int alpha_treshold = 20;
	const int max_dist = 0x7FFFFFFF;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			const uint8_t *rptr = &srcptr[(i * width + j) * 4];
			uint8_t *wptr = &data_ptr[(i * width + j) * 4];

			if (rptr[3] >= alpha_treshold)
				continue;

			int closest_dist = max_dist;
			uint8_t closest_color[3];

			int from_x = MAX(0, j - max_radius);
			int to_x = MIN(width - 1, j + max_radius);
			int from_y = MAX(0, i - max_radius);
			int to_y = MIN(height - 1, i + max_radius);

			for (int k = from_y; k <= to_y; k++) {
				for (int l = from_x; l <= to_x; l++) {

					int dy = i - k;
					int dx = j - l;
					int dist = dy * dy + dx * dx;
					if (dist >= closest_dist)
						continue;

					const uint8_t *rp = &srcptr[(k * width + l) << 2];

					if (rp[3] < alpha_treshold)
						continue;

					closest_color[0] = rp[0];
					closest_color[1] = rp[1];
					closest_color[2] = rp[2];
				}
			}

			if (closest_dist != max_dist) {

				wptr[0] = closest_color[0];
				wptr[1] = closest_color[1];
				wptr[2] = closest_color[2];
			}
		}
	}
}

String Image::get_format_name(Format p_format) {

	ERR_FAIL_INDEX_V(p_format, FORMAT_MAX, String());
	return format_names[p_format];
}

Image::Image(const uint8_t *p_mem_png_jpg, int p_len) {

	width = 0;
	height = 0;
	mipmaps = false;
	format = FORMAT_L8;

	if (_png_mem_loader_func) {
		*this = _png_mem_loader_func(p_mem_png_jpg, p_len);
	}

	if (empty() && _jpg_mem_loader_func) {
		*this = _jpg_mem_loader_func(p_mem_png_jpg, p_len);
	}
}

Image::Image() {

	width = 0;
	height = 0;
	mipmaps = false;
	format = FORMAT_L8;
}

Image::~Image() {
}
