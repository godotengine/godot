/*************************************************************************/
/*  image_etc.cpp                                                        */
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
#include "image_etc.h"
#include "image.h"
#include "os/copymem.h"
#include "print_string.h"
#include "rg_etc1.h"
static void _decompress_etc(Image *p_img) {

	ERR_FAIL_COND(p_img->get_format() != Image::FORMAT_ETC);

	int imgw = p_img->get_width();
	int imgh = p_img->get_height();
	PoolVector<uint8_t> src = p_img->get_data();
	PoolVector<uint8_t> dst;

	PoolVector<uint8_t>::Read r = src.read();

	int mmc = p_img->get_mipmap_count();

	for (int i = 0; i <= mmc; i++) {

		dst.resize(dst.size() + imgw * imgh * 3);
		const uint8_t *srcbr = &r[p_img->get_mipmap_offset(i)];
		PoolVector<uint8_t>::Write w = dst.write();

		uint8_t *wptr = &w[dst.size() - imgw * imgh * 3];

		int bw = MAX(imgw / 4, 1);
		int bh = MAX(imgh / 4, 1);

		for (int y = 0; y < bh; y++) {

			for (int x = 0; x < bw; x++) {

				uint8_t block[4 * 4 * 4];

				rg_etc1::unpack_etc1_block(srcbr, (unsigned int *)block);
				srcbr += 8;

				int maxx = MIN(imgw, 4);
				int maxy = MIN(imgh, 4);

				for (int yy = 0; yy < maxy; yy++) {

					for (int xx = 0; xx < maxx; xx++) {

						uint32_t src_ofs = (yy * 4 + xx) * 4;
						uint32_t dst_ofs = ((y * 4 + yy) * imgw + x * 4 + xx) * 3;
						wptr[dst_ofs + 0] = block[src_ofs + 0];
						wptr[dst_ofs + 1] = block[src_ofs + 1];
						wptr[dst_ofs + 2] = block[src_ofs + 2];
					}
				}
			}
		}

		imgw = MAX(1, imgw / 2);
		imgh = MAX(1, imgh / 2);
	}

	r = PoolVector<uint8_t>::Read();
	//print_line("Re Creating ETC into regular image: w "+itos(p_img->get_width())+" h "+itos(p_img->get_height())+" mm "+itos(p_img->get_mipmaps()));
	*p_img = Image(p_img->get_width(), p_img->get_height(), p_img->has_mipmaps(), Image::FORMAT_RGB8, dst);
	if (p_img->has_mipmaps())
		p_img->generate_mipmaps();
}

static void _compress_etc(Image *p_img) {

	Image img = *p_img;

	int imgw = img.get_width(), imgh = img.get_height();

	ERR_FAIL_COND(nearest_power_of_2(imgw) != imgw || nearest_power_of_2(imgh) != imgh);

	if (img.get_format() != Image::FORMAT_RGB8)
		img.convert(Image::FORMAT_RGB8);

	PoolVector<uint8_t> res_data;
	PoolVector<uint8_t> dst_data;
	PoolVector<uint8_t>::Read r = img.get_data().read();

	int target_size = Image::get_image_data_size(p_img->get_width(), p_img->get_height(), Image::FORMAT_ETC, p_img->has_mipmaps() ? -1 : 0);
	int mmc = p_img->has_mipmaps() ? Image::get_image_required_mipmaps(p_img->get_width(), p_img->get_height(), Image::FORMAT_ETC) : 0;

	dst_data.resize(target_size);
	int mc = 0;
	int ofs = 0;
	PoolVector<uint8_t>::Write w = dst_data.write();

	rg_etc1::etc1_pack_params pp;
	pp.m_quality = rg_etc1::cLowQuality;
	for (int i = 0; i <= mmc; i++) {

		int bw = MAX(imgw / 4, 1);
		int bh = MAX(imgh / 4, 1);
		const uint8_t *src = &r[img.get_mipmap_offset(i)];
		int mmsize = MAX(bw, 1) * MAX(bh, 1) * 8;

		uint8_t *dst = &w[ofs];
		ofs += mmsize;

		//print_line("bh: "+itos(bh)+" bw: "+itos(bw));

		for (int y = 0; y < bh; y++) {

			for (int x = 0; x < bw; x++) {

				//print_line("x: "+itos(x)+" y: "+itos(y));

				uint8_t block[4 * 4 * 4];
				zeromem(block, 4 * 4 * 4);
				uint8_t cblock[8];

				int maxy = MIN(imgh, 4);
				int maxx = MIN(imgw, 4);

				for (int yy = 0; yy < maxy; yy++) {

					for (int xx = 0; xx < maxx; xx++) {

						uint32_t dst_ofs = (yy * 4 + xx) * 4;
						uint32_t src_ofs = ((y * 4 + yy) * imgw + x * 4 + xx) * 3;
						block[dst_ofs + 0] = src[src_ofs + 0];
						block[dst_ofs + 1] = src[src_ofs + 1];
						block[dst_ofs + 2] = src[src_ofs + 2];
						block[dst_ofs + 3] = 255;
					}
				}

				rg_etc1::pack_etc1_block(cblock, (const unsigned int *)block, pp);
				for (int j = 0; j < 8; j++) {

					dst[j] = cblock[j];
				}

				dst += 8;
			}
		}

		imgw = MAX(1, imgw / 2);
		imgh = MAX(1, imgh / 2);
		mc++;
	}

	*p_img = Image(p_img->get_width(), p_img->get_height(), (mc - 1) ? true : false, Image::FORMAT_ETC, dst_data);
}

void _register_etc1_compress_func() {

	rg_etc1::pack_etc1_block_init();
	Image::_image_compress_etc_func = _compress_etc;
	Image::_image_decompress_etc = _decompress_etc;
}
