/**************************************************************************/
/*  image_decompress_bcdec.cpp                                            */
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

#include "image_decompress_bcdec.h"

#include "core/os/os.h"
#include "core/string/print_string.h"

#define BCDEC_IMPLEMENTATION
#include "thirdparty/misc/bcdec.h"

inline void bcdec_bc6h_half_s(const void *compressedBlock, void *decompressedBlock, int destinationPitch) {
	bcdec_bc6h_half(compressedBlock, decompressedBlock, destinationPitch, true);
}

inline void bcdec_bc6h_half_u(const void *compressedBlock, void *decompressedBlock, int destinationPitch) {
	bcdec_bc6h_half(compressedBlock, decompressedBlock, destinationPitch, false);
}

template <void (*decompress_func)(const void *, void *, int), int block_size, int pixel_size, int component_size>
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
			decompress_func(&src[src_pos], &dst[dst_pos], image_pitch / component_size);
			src_pos += block_size;
			dst_pos += block_pitch;
		}

		// Decompress the block on the right.
		if (width_diff > 0) {
			decompress_func(&src[src_pos], temp_buf, block_pitch / component_size);

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
			decompress_func(&src[src_pos], temp_buf, block_pitch / component_size);

			// Copy the data from the temporary buffer to the output.
			for (int i = 0; i < height_diff; i++) {
				memcpy(&dst[dst_pos + i * image_pitch], &temp_buf[i * block_pitch], block_pitch);
			}

			src_pos += block_size;
			dst_pos += block_pitch;
		}

		// Decompress the block in the lower-right corner.
		if (width_diff > 0) {
			decompress_func(&src[src_pos], temp_buf, block_pitch / component_size);

			// Copy the data from the temporary buffer to the output.
			for (int i = 0; i < height_diff; i++) {
				memcpy(&dst[dst_pos + i * image_pitch], &temp_buf[i * block_pitch], odd_pitch);
			}

			src_pos += block_size;
			dst_pos += odd_pitch;
		}
	}
}

template <void (*decompress_func)(const void *, void *, int), int block_size, int pixel_size, int component_size>
static inline void _decompress_mipmap(int width, int height, const uint8_t *src, uint8_t *dst) {
	size_t src_pos = 0;
	size_t dst_pos = 0;

	// The size of a single block in bytes.
	const int block_pitch = 4 * pixel_size;
	// The pitch of the image in bytes.
	const int image_pitch = width * pixel_size;

	for (int y = 0; y < height; y += 4) {
		for (int x = 0; x < width; x += 4) {
			decompress_func(&src[src_pos], &dst[dst_pos], image_pitch / component_size);
			src_pos += block_size;
			dst_pos += block_pitch;
		}

		// Skip to the next row of blocks, the current one has already been filled.
		dst_pos += 3 * image_pitch;
	}
}

static void decompress_image(BCdecFormat format, const void *src, void *dst, const uint64_t width, const uint64_t height) {
	const uint8_t *src_blocks = reinterpret_cast<const uint8_t *>(src);
	uint8_t *dec_blocks = reinterpret_cast<uint8_t *>(dst);

	const uint64_t aligned_width = (width + 3) & ~0x03;
	const uint64_t aligned_height = (height + 3) & ~0x03;

	if (width != aligned_width || height != aligned_height) {
		// Decompress the mipmap in a 'safe' way, which involves starting from the top left.
		// For each block row, decompress all of the 'full' blocks, then the misaligned one (on the x axis).
		// Then, decompress the final misaligned block row at the bottom.
		// Finally, decompress the misaligned block at the bottom right.
		switch (format) {
			case BCdec_BC1: {
				_safe_decompress_mipmap<bcdec_bc1, BCDEC_BC1_BLOCK_SIZE, 4, 1>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC2: {
				_safe_decompress_mipmap<bcdec_bc2, BCDEC_BC2_BLOCK_SIZE, 4, 1>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC3: {
				_safe_decompress_mipmap<bcdec_bc3, BCDEC_BC3_BLOCK_SIZE, 4, 1>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC4: {
				_safe_decompress_mipmap<bcdec_bc4, BCDEC_BC4_BLOCK_SIZE, 1, 1>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC5: {
				_safe_decompress_mipmap<bcdec_bc5, BCDEC_BC5_BLOCK_SIZE, 2, 1>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC6U: {
				_safe_decompress_mipmap<bcdec_bc6h_half_u, BCDEC_BC6H_BLOCK_SIZE, 6, 2>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC6S: {
				_safe_decompress_mipmap<bcdec_bc6h_half_s, BCDEC_BC6H_BLOCK_SIZE, 6, 2>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC7: {
				_safe_decompress_mipmap<bcdec_bc7, BCDEC_BC7_BLOCK_SIZE, 4, 1>(width, height, src_blocks, dec_blocks);
			} break;
		}
	} else {
		// Just decompress as usual, as fast as possible.
		switch (format) {
			case BCdec_BC1: {
				_decompress_mipmap<bcdec_bc1, BCDEC_BC1_BLOCK_SIZE, 4, 1>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC2: {
				_decompress_mipmap<bcdec_bc2, BCDEC_BC2_BLOCK_SIZE, 4, 1>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC3: {
				_decompress_mipmap<bcdec_bc3, BCDEC_BC3_BLOCK_SIZE, 4, 1>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC4: {
				_decompress_mipmap<bcdec_bc4, BCDEC_BC4_BLOCK_SIZE, 1, 1>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC5: {
				_decompress_mipmap<bcdec_bc5, BCDEC_BC5_BLOCK_SIZE, 2, 1>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC6U: {
				_decompress_mipmap<bcdec_bc6h_half_u, BCDEC_BC6H_BLOCK_SIZE, 6, 2>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC6S: {
				_decompress_mipmap<bcdec_bc6h_half_s, BCDEC_BC6H_BLOCK_SIZE, 6, 2>(width, height, src_blocks, dec_blocks);
			} break;
			case BCdec_BC7: {
				_decompress_mipmap<bcdec_bc7, BCDEC_BC7_BLOCK_SIZE, 4, 1>(width, height, src_blocks, dec_blocks);
			} break;
		}
	}
}

void image_decompress_bcdec(Image *p_image) {
	uint64_t start_time = OS::get_singleton()->get_ticks_msec();

	int width = p_image->get_width();
	int height = p_image->get_height();

	Image::Format source_format = p_image->get_format();
	Image::Format target_format = Image::FORMAT_MAX;

	BCdecFormat bcdec_format = BCdec_BC1;

	switch (source_format) {
		case Image::FORMAT_DXT1:
			bcdec_format = BCdec_BC1;
			target_format = Image::FORMAT_RGBA8;
			break;

		case Image::FORMAT_DXT3:
			bcdec_format = BCdec_BC2;
			target_format = Image::FORMAT_RGBA8;
			break;

		case Image::FORMAT_DXT5:
		case Image::FORMAT_DXT5_RA_AS_RG:
			bcdec_format = BCdec_BC3;
			target_format = Image::FORMAT_RGBA8;
			break;

		case Image::FORMAT_RGTC_R:
			bcdec_format = BCdec_BC4;
			target_format = Image::FORMAT_R8;
			break;

		case Image::FORMAT_RGTC_RG:
			bcdec_format = BCdec_BC5;
			target_format = Image::FORMAT_RG8;
			break;

		case Image::FORMAT_BPTC_RGBFU:
			bcdec_format = BCdec_BC6U;
			target_format = Image::FORMAT_RGBH;
			break;

		case Image::FORMAT_BPTC_RGBF:
			bcdec_format = BCdec_BC6S;
			target_format = Image::FORMAT_RGBH;
			break;

		case Image::FORMAT_BPTC_RGBA:
			bcdec_format = BCdec_BC7;
			target_format = Image::FORMAT_RGBA8;
			break;

		default:
			ERR_FAIL_MSG("bcdec: Can't decompress unknown format: " + Image::get_format_name(source_format) + ".");
			break;
	}

	int mm_count = p_image->get_mipmap_count();
	int64_t target_size = Image::get_image_data_size(width, height, target_format, p_image->has_mipmaps());

	// Decompressed data.
	Vector<uint8_t> data;
	data.resize(target_size);
	uint8_t *wb = data.ptrw();

	// Source data.
	const uint8_t *rb = p_image->get_data().ptr();

	// Decompress mipmaps.
	for (int i = 0; i <= mm_count; i++) {
		int mipmap_w = 0, mipmap_h = 0;
		int64_t src_ofs = Image::get_image_mipmap_offset(width, height, source_format, i);
		int64_t dst_ofs = Image::get_image_mipmap_offset_and_dimensions(width, height, target_format, i, mipmap_w, mipmap_h);
		decompress_image(bcdec_format, rb + src_ofs, wb + dst_ofs, mipmap_w, mipmap_h);
	}

	p_image->set_data(width, height, p_image->has_mipmaps(), target_format, data);

	// Swap channels if the format is using a channel swizzle.
	if (source_format == Image::FORMAT_DXT5_RA_AS_RG) {
		p_image->convert_ra_rgba8_to_rg();
	}

	print_verbose(vformat("bcdec: Decompression of a %dx%d %s image with %d mipmaps took %d ms.",
			p_image->get_width(), p_image->get_height(), Image::get_format_name(source_format), p_image->get_mipmap_count(), OS::get_singleton()->get_ticks_msec() - start_time));
}
