/*************************************************************************/
/*  image_compress_cvtt.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "image_compress_cvtt.h"

#include "print_string.h"

#include <ConvectionKernels.h>

void image_compress_cvtt(Image *p_image, float p_lossy_quality, Image::CompressSource p_source) {

	if (p_image->get_format() >= Image::FORMAT_BPTC_RGBA)
		return; //do not compress, already compressed

	int w = p_image->get_width();
	int h = p_image->get_height();

	bool is_ldr = (p_image->get_format() <= Image::FORMAT_RGBA8);
	bool is_hdr = (p_image->get_format() == Image::FORMAT_RGBH);

	if (!is_ldr && !is_hdr) {
		return; // Not a usable source format
	}

	cvtt::Options options;
	uint32_t flags = cvtt::Flags::Fastest;

	if (p_lossy_quality > 0.85)
		flags = cvtt::Flags::Ultra;
	else if (p_lossy_quality > 0.75)
		flags = cvtt::Flags::Better;
	else if (p_lossy_quality > 0.55)
		flags = cvtt::Flags::Default;
	else if (p_lossy_quality > 0.35)
		flags = cvtt::Flags::Fast;
	else if (p_lossy_quality > 0.15)
		flags = cvtt::Flags::Faster;

	flags |= cvtt::Flags::BC7_RespectPunchThrough;

	if (p_source == Image::COMPRESS_SOURCE_NORMAL) {
		flags |= cvtt::Flags::Uniform;
	}

	Image::Format target_format = Image::FORMAT_BPTC_RGBA;

	bool is_signed = false;
	if (is_hdr) {
		PoolVector<uint8_t>::Read rb = p_image->get_data().read();

		const uint16_t *source_data = reinterpret_cast<const uint16_t *>(&rb[0]);
		int pixel_element_count = w * h * 3;
		for (int i = 0; i < pixel_element_count; i++) {
			if ((source_data[i] & 0x8000) != 0 && (source_data[i] & 0x7fff) != 0) {
				is_signed = true;
				break;
			}
		}

		target_format = is_signed ? Image::FORMAT_BPTC_RGBF : Image::FORMAT_BPTC_RGBFU;
	} else {
		p_image->convert(Image::FORMAT_RGBA8); //still uses RGBA to convert
	}

	PoolVector<uint8_t>::Read rb = p_image->get_data().read();

	PoolVector<uint8_t> data;
	int target_size = Image::get_image_data_size(w, h, target_format, p_image->has_mipmaps());
	int mm_count = p_image->has_mipmaps() ? Image::get_image_required_mipmaps(w, h, target_format) : 0;
	data.resize(target_size);
	int shift = Image::get_format_pixel_rshift(target_format);

	PoolVector<uint8_t>::Write wb = data.write();

	int dst_ofs = 0;

	for (int i = 0; i <= mm_count; i++) {

		int bw = w % 4 != 0 ? w + (4 - w % 4) : w;
		int bh = h % 4 != 0 ? h + (4 - h % 4) : h;

		int src_ofs = p_image->get_mipmap_offset(i);

		const uint8_t *in_bytes = &rb[src_ofs];
		uint8_t *out_bytes = &wb[dst_ofs];

		cvtt::PixelBlockU8 input_blocks_ldr[cvtt::NumParallelBlocks];
		cvtt::PixelBlockF16 input_blocks_hdr[cvtt::NumParallelBlocks];

		int bytes_per_pixel = is_hdr ? 6 : 4;

		for (int y_start = 0; y_start < h; y_start += 4) {
			int y_end = y_start + 4;

			for (int x_start = 0; x_start < w; x_start += 4 * cvtt::NumParallelBlocks) {
				int x_end = x_start + 4 * cvtt::NumParallelBlocks;

				for (int y = y_start; y < y_end; y++) {
					int first_input_element = (y - y_start) * 4;
					const uint8_t *row_start;
					if (y >= h) {
						row_start = in_bytes + (h - 1) * (w * bytes_per_pixel);
					} else {
						row_start = in_bytes + y * (w * bytes_per_pixel);
					}

					for (int x = x_start; x < x_end; x++) {
						const uint8_t *pixel_start;
						if (x >= w) {
							pixel_start = row_start + (w - 1) * bytes_per_pixel;
						} else {
							pixel_start = row_start + x * bytes_per_pixel;
						}

						int block_index = (x - x_start) / 4;
						int block_element = (x - x_start) % 4 + first_input_element;
						if (is_hdr) {
							memcpy(input_blocks_hdr[block_index].m_pixels[block_element], pixel_start, bytes_per_pixel);
							input_blocks_hdr[block_index].m_pixels[block_element][3] = 0x3c00; // 1.0 (unused)
						} else {
							memcpy(input_blocks_ldr[block_index].m_pixels[block_element], pixel_start, bytes_per_pixel);
						}
					}
				}

				uint8_t output_blocks[16 * cvtt::NumParallelBlocks];

				if (is_hdr) {
					if (is_signed) {
						cvtt::Kernels::EncodeBC6HS(output_blocks, input_blocks_hdr, options);
					} else {
						cvtt::Kernels::EncodeBC6HU(output_blocks, input_blocks_hdr, options);
					}
				} else {
					cvtt::Kernels::EncodeBC7(output_blocks, input_blocks_ldr, options);
				}

				int num_real_blocks = ((w - x_start) + 3) / 4;
				if (num_real_blocks > cvtt::NumParallelBlocks) {
					num_real_blocks = cvtt::NumParallelBlocks;
				}

				memcpy(out_bytes, output_blocks, 16 * num_real_blocks);
				out_bytes += 16 * num_real_blocks;
			}
		}

		dst_ofs += (MAX(4, bw) * MAX(4, bh)) >> shift;
		w >>= 1;
		h >>= 1;
	}

	rb = PoolVector<uint8_t>::Read();
	wb = PoolVector<uint8_t>::Write();

	p_image->create(p_image->get_width(), p_image->get_height(), p_image->has_mipmaps(), target_format, data);
}

void image_decompress_cvtt(Image *p_image) {

	Image::Format target_format;
	bool is_signed = false;
	bool is_hdr = false;

	Image::Format input_format = p_image->get_format();

	switch (input_format) {
		case Image::FORMAT_BPTC_RGBA:
			target_format = Image::FORMAT_RGBA8;
			break;
		case Image::FORMAT_BPTC_RGBF:
		case Image::FORMAT_BPTC_RGBFU:
			target_format = Image::FORMAT_RGBH;
			is_signed = (input_format == Image::FORMAT_BPTC_RGBF);
			is_hdr = true;
			break;
		default:
			return; // Invalid input format
	};

	int w = p_image->get_width();
	int h = p_image->get_height();

	PoolVector<uint8_t>::Read rb = p_image->get_data().read();

	PoolVector<uint8_t> data;
	int target_size = Image::get_image_data_size(w, h, target_format, p_image->has_mipmaps());
	int mm_count = p_image->get_mipmap_count();
	data.resize(target_size);
	int shift = Image::get_format_pixel_rshift(target_format);

	PoolVector<uint8_t>::Write wb = data.write();

	int bytes_per_pixel = is_hdr ? 6 : 4;

	int dst_ofs = 0;

	for (int i = 0; i <= mm_count; i++) {

		int src_ofs = p_image->get_mipmap_offset(i);

		const uint8_t *in_bytes = &rb[src_ofs];
		uint8_t *out_bytes = &wb[dst_ofs];

		cvtt::PixelBlockU8 output_blocks_ldr[cvtt::NumParallelBlocks];
		cvtt::PixelBlockF16 output_blocks_hdr[cvtt::NumParallelBlocks];

		for (int y_start = 0; y_start < h; y_start += 4) {
			int y_end = y_start + 4;

			for (int x_start = 0; x_start < w; x_start += 4 * cvtt::NumParallelBlocks) {
				int x_end = x_start + 4 * cvtt::NumParallelBlocks;

				uint8_t input_blocks[16 * cvtt::NumParallelBlocks];
				memset(input_blocks, 0, sizeof(input_blocks));

				int num_real_blocks = ((w - x_start) + 3) / 4;
				if (num_real_blocks > cvtt::NumParallelBlocks) {
					num_real_blocks = cvtt::NumParallelBlocks;
				}

				memcpy(input_blocks, in_bytes, 16 * num_real_blocks);
				in_bytes += 16 * num_real_blocks;

				if (is_hdr) {
					if (is_signed) {
						cvtt::Kernels::DecodeBC6HS(output_blocks_hdr, input_blocks);
					} else {
						cvtt::Kernels::DecodeBC6HU(output_blocks_hdr, input_blocks);
					}
				} else {
					cvtt::Kernels::DecodeBC7(output_blocks_ldr, input_blocks);
				}

				for (int y = y_start; y < y_end; y++) {
					int first_input_element = (y - y_start) * 4;
					uint8_t *row_start;
					if (y >= h) {
						row_start = out_bytes + (h - 1) * (w * bytes_per_pixel);
					} else {
						row_start = out_bytes + y * (w * bytes_per_pixel);
					}

					for (int x = x_start; x < x_end; x++) {
						uint8_t *pixel_start;
						if (x >= w) {
							pixel_start = row_start + (w - 1) * bytes_per_pixel;
						} else {
							pixel_start = row_start + x * bytes_per_pixel;
						}

						int block_index = (x - x_start) / 4;
						int block_element = (x - x_start) % 4 + first_input_element;
						if (is_hdr) {
							memcpy(pixel_start, output_blocks_hdr[block_index].m_pixels[block_element], bytes_per_pixel);
						} else {
							memcpy(pixel_start, output_blocks_ldr[block_index].m_pixels[block_element], bytes_per_pixel);
						}
					}
				}
			}
		}

		dst_ofs += w * h * bytes_per_pixel;
		w >>= 1;
		h >>= 1;
	}

	rb = PoolVector<uint8_t>::Read();
	wb = PoolVector<uint8_t>::Write();

	p_image->create(p_image->get_width(), p_image->get_height(), p_image->has_mipmaps(), target_format, data);
}
