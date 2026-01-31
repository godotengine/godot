/**************************************************************************/
/*  image_decompress_etcpak.cpp                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "image_decompress_etcpak.h"

#include "core/os/os.h"
#include "core/string/print_string.h"

#include <DecodeRGB.hpp>

#define ETCPAK_R_BLOCK_SIZE 8
#define ETCPAK_RG_BLOCK_SIZE 16
#define ETCPAK_RGB_BLOCK_SIZE 8
#define ETCPAK_RGBA_BLOCK_SIZE 16

template <void (*decompress_func)(const void *, void *, size_t), int block_size, int pixel_size>
static inline void _safe_decompress_mipmap(int width, int height, const uint8_t *src, uint8_t *dst) {
	// A stack-allocated output buffer large enough to contain an entire uncompressed block.
	uint8_t temp_buf[4 * 4 * pixel_size];

	// The amount of misaligned pixels on each axis.
	const int width_diff = width - (width & ~0x03);
	const int height_diff = height - (height & ~0x03);

	// The amount of uncompressed blocks on each axis.
	const int width_blocks = (width & ~0x03) / 4;
	const int height_blocks = (height & ~0x03) / 4;

	// The pitch of the image in bytes.
	const int image_pitch = width * pixel_size;
	// The pitch of a block in bytes.
	const int block_pitch = 4 * pixel_size;
	// The pitch of the last block in bytes.
	const int odd_pitch = width_diff * pixel_size;

	size_t src_pos = 0;
	size_t dst_pos = 0;

	// Decompress the blocks, starting from the top.
	for (int y = 0; y < height_blocks; y += 1) {
		// Decompress the blocks, starting from the left.
		for (int x = 0; x < width_blocks; x += 1) {
			decompress_func(&src[src_pos], &dst[dst_pos], width);
			src_pos += block_size;
			dst_pos += block_pitch;
		}

		// Decompress the block on the right.
		if (width_diff > 0) {
			decompress_func(&src[src_pos], temp_buf, 4);

			// Copy the data from the temporary buffer to the output.
			for (int i = 0; i < 4; i++) {
				memcpy(&dst[dst_pos + i * image_pitch], &temp_buf[i * block_pitch], odd_pitch);
			}

			src_pos += block_size;
			dst_pos += odd_pitch;
		}

		// Skip to the next row of blocks, the current one has already been filled.
		dst_pos += 3 * image_pitch;
	}

	// Decompress the blocks at the bottom of the image.
	if (height_diff > 0) {
		// Decompress the blocks at the bottom.
		for (int x = 0; x < width_blocks; x += 1) {
			decompress_func(&src[src_pos], temp_buf, 4);

			// Copy the data from the temporary buffer to the output.
			for (int i = 0; i < height_diff; i++) {
				memcpy(&dst[dst_pos + i * image_pitch], &temp_buf[i * block_pitch], block_pitch);
			}

			src_pos += block_size;
			dst_pos += block_pitch;
		}

		// Decompress the block in the lower-right corner.
		if (width_diff > 0) {
			decompress_func(&src[src_pos], temp_buf, 4);

			// Copy the data from the temporary buffer to the output.
			for (int i = 0; i < height_diff; i++) {
				memcpy(&dst[dst_pos + i * image_pitch], &temp_buf[i * block_pitch], odd_pitch);
			}

			src_pos += block_size;
			dst_pos += odd_pitch;
		}
	}
}

template <void (*decompress_func)(const void *, void *, size_t), int block_size, int pixel_size>
static inline void _decompress_mipmap(int width, int height, const uint8_t *src, uint8_t *dst) {
	size_t src_pos = 0;
	size_t dst_pos = 0;

	// The size of a single block in bytes.
	const int block_pitch = 4 * pixel_size;

	for (int y = 0; y < height; y += 4) {
		for (int x = 0; x < width; x += 4) {
			decompress_func(&src[src_pos], &dst[dst_pos], width);
			src_pos += block_size;
			dst_pos += block_pitch;
		}

		// Skip to the next row of blocks, the current one has already been filled.
		dst_pos += 3 * width * pixel_size;
	}
}

static void decompress_image(EtcpakFormat format, const void *src, void *dst, const uint64_t width, const uint64_t height) {
	const uint8_t *src_blocks = reinterpret_cast<const uint8_t *>(src);
	uint8_t *dec_blocks = reinterpret_cast<uint8_t *>(dst);

	const uint64_t aligned_width = (width + 3) & ~0x03;
	const uint64_t aligned_height = (height + 3) & ~0x03;

	if (width != aligned_width || height != aligned_height) {
		switch (format) {
			case Etcpak_R: {
				_safe_decompress_mipmap<DecodeRBlock, ETCPAK_R_BLOCK_SIZE, 4>(width, height, src_blocks, dec_blocks);
			} break;
			case Etcpak_RG: {
				_safe_decompress_mipmap<DecodeRGBlock, ETCPAK_RG_BLOCK_SIZE, 4>(width, height, src_blocks, dec_blocks);
			} break;
			case Etcpak_RGB: {
				_safe_decompress_mipmap<DecodeRGBBlock, ETCPAK_RGB_BLOCK_SIZE, 4>(width, height, src_blocks, dec_blocks);
			} break;
			case Etcpak_RGBA: {
				_safe_decompress_mipmap<DecodeRGBABlock, ETCPAK_RGBA_BLOCK_SIZE, 4>(width, height, src_blocks, dec_blocks);
			} break;
		}
	} else {
		switch (format) {
			case Etcpak_R: {
				_decompress_mipmap<DecodeRBlock, ETCPAK_R_BLOCK_SIZE, 4>(width, height, src_blocks, dec_blocks);
			} break;
			case Etcpak_RG: {
				_decompress_mipmap<DecodeRGBlock, ETCPAK_RG_BLOCK_SIZE, 4>(width, height, src_blocks, dec_blocks);
			} break;
			case Etcpak_RGB: {
				_decompress_mipmap<DecodeRGBBlock, ETCPAK_RGB_BLOCK_SIZE, 4>(width, height, src_blocks, dec_blocks);
			} break;
			case Etcpak_RGBA: {
				_decompress_mipmap<DecodeRGBABlock, ETCPAK_RGBA_BLOCK_SIZE, 4>(width, height, src_blocks, dec_blocks);
			} break;
		}
	}
}

void _decompress_etc(Image *p_image) {
	uint64_t start_time = OS::get_singleton()->get_ticks_msec();

	int width = p_image->get_width();
	int height = p_image->get_height();

	Image::Format source_format = p_image->get_format();
	Image::Format target_format = Image::FORMAT_RGBA8;

	EtcpakFormat etcpak_format = Etcpak_R;

	switch (source_format) {
		case Image::FORMAT_ETC:
		case Image::FORMAT_ETC2_RGB8:
			etcpak_format = Etcpak_RGB;
			break;

		case Image::FORMAT_ETC2_RGBA8:
		case Image::FORMAT_ETC2_RA_AS_RG:
			etcpak_format = Etcpak_RGBA;
			break;

		case Image::FORMAT_ETC2_R11:
			etcpak_format = Etcpak_R;
			break;

		case Image::FORMAT_ETC2_RG11:
			etcpak_format = Etcpak_RG;
			break;

		default:
			ERR_FAIL_MSG(vformat("etcpak: Can't decompress image %s with an unknown format: %s.", p_image->get_path(), Image::get_format_name(source_format)));
			break;
	}

	int mm_count = p_image->get_mipmap_count();
	int64_t target_size = Image::get_image_data_size(width, height, target_format, p_image->has_mipmaps());

	// Decompressed data.
	Vector<uint8_t> data;
	data.resize(target_size);
	uint8_t *wb = data.ptrw();

	// Source data.
	const uint8_t *rb = p_image->ptr();

	// Decompress mipmaps.
	for (int i = 0; i <= mm_count; i++) {
		int mipmap_w = 0, mipmap_h = 0;
		int64_t src_ofs = Image::get_image_mipmap_offset(width, height, source_format, i);
		int64_t dst_ofs = Image::get_image_mipmap_offset_and_dimensions(width, height, target_format, i, mipmap_w, mipmap_h);
		decompress_image(etcpak_format, rb + src_ofs, wb + dst_ofs, mipmap_w, mipmap_h);
	}

	p_image->set_data(p_image->get_width(), p_image->get_height(), p_image->has_mipmaps(), target_format, data);

	// Swap channels if the format is using a channel swizzle.
	if (source_format == Image::FORMAT_ETC2_RA_AS_RG) {
		p_image->convert_ra_rgba8_to_rg();
	}

	print_verbose(vformat("etcpak: Decompression of %dx%d %s image %s with %d mipmaps took %d ms.",
			p_image->get_width(), p_image->get_height(), Image::get_format_name(source_format), p_image->get_path(), p_image->get_mipmap_count(), OS::get_singleton()->get_ticks_msec() - start_time));
}
