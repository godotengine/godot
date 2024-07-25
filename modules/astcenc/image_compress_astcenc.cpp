/**************************************************************************/
/*  image_compress_astcenc.cpp                                            */
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

#include "image_compress_astcenc.h"

#include "core/os/os.h"
#include "core/string/print_string.h"

#include <astcenc.h>

void _compress_astc(Image *r_img, Image::ASTCFormat p_format) {
	uint64_t start_time = OS::get_singleton()->get_ticks_msec();

	// TODO: See how to handle lossy quality.

	Image::Format img_format = r_img->get_format();
	if (Image::is_format_compressed(img_format)) {
		return; // Do not compress, already compressed.
	}

	bool is_hdr = false;
	if ((img_format >= Image::FORMAT_RH) && (img_format <= Image::FORMAT_RGBE9995)) {
		is_hdr = true;
		r_img->convert(Image::FORMAT_RGBAF);
	} else {
		r_img->convert(Image::FORMAT_RGBA8);
	}

	// Determine encoder output format from our enum.

	Image::Format target_format = Image::FORMAT_RGBA8;
	astcenc_profile profile = ASTCENC_PRF_LDR;
	unsigned int block_x = 4;
	unsigned int block_y = 4;

	if (p_format == Image::ASTCFormat::ASTC_FORMAT_4x4) {
		if (is_hdr) {
			target_format = Image::FORMAT_ASTC_4x4_HDR;
			profile = ASTCENC_PRF_HDR;
		} else {
			target_format = Image::FORMAT_ASTC_4x4;
		}
	} else if (p_format == Image::ASTCFormat::ASTC_FORMAT_8x8) {
		if (is_hdr) {
			target_format = Image::FORMAT_ASTC_8x8_HDR;
			profile = ASTCENC_PRF_HDR;
		} else {
			target_format = Image::FORMAT_ASTC_8x8;
		}
		block_x = 8;
		block_y = 8;
	}

	// Compress image data and (if required) mipmaps.

	const bool mipmaps = r_img->has_mipmaps();
	int width = r_img->get_width();
	int height = r_img->get_height();
	int required_width = (width % block_x) != 0 ? width + (block_x - (width % block_x)) : width;
	int required_height = (height % block_y) != 0 ? height + (block_y - (height % block_y)) : height;

	if (width != required_width || height != required_height) {
		// Resize texture to fit block size.
		r_img->resize(required_width, required_height);
		width = required_width;
		height = required_height;
	}

	print_verbose(vformat("astcenc: Encoding image size %dx%d to format %s%s.", width, height, Image::get_format_name(target_format), mipmaps ? ", with mipmaps" : ""));

	// Initialize astcenc.

	int64_t dest_size = Image::get_image_data_size(width, height, target_format, mipmaps);
	Vector<uint8_t> dest_data;
	dest_data.resize(dest_size);
	uint8_t *dest_write = dest_data.ptrw();

	astcenc_config config;
	config.block_x = block_x;
	config.block_y = block_y;
	config.profile = profile;

	const float quality = ASTCENC_PRE_MEDIUM;
	astcenc_error status = astcenc_config_init(profile, block_x, block_y, 1, quality, 0, &config);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS,
			vformat("astcenc: Configuration initialization failed: %s.", astcenc_get_error_string(status)));

	// Context allocation.

	astcenc_context *context;
	const unsigned int thread_count = 1; // Godot compresses multiple images each on a thread, which is more efficient for large amount of images imported.
	status = astcenc_context_alloc(&config, thread_count, &context);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS,
			vformat("astcenc: Context allocation failed: %s.", astcenc_get_error_string(status)));

	Vector<uint8_t> image_data = r_img->get_data();

	int mip_count = mipmaps ? Image::get_image_required_mipmaps(width, height, target_format) : 0;
	for (int i = 0; i < mip_count + 1; i++) {
		int src_mip_w, src_mip_h;
		int64_t src_ofs = Image::get_image_mipmap_offset_and_dimensions(width, height, r_img->get_format(), i, src_mip_w, src_mip_h);

		const uint8_t *slices = &image_data.ptr()[src_ofs];

		int dst_mip_w, dst_mip_h;
		int64_t dst_ofs = Image::get_image_mipmap_offset_and_dimensions(width, height, target_format, i, dst_mip_w, dst_mip_h);
		// Ensure that mip offset is a multiple of 8 (etcpak expects uint64_t pointer).
		if (unlikely(dst_ofs % 8 != 0)) {
			astcenc_context_free(context);
			ERR_FAIL_MSG("astcenc: Mip offset is not a multiple of 8.");
		}
		uint8_t *dest_mip_write = (uint8_t *)&dest_write[dst_ofs];

		// Compress image.

		astcenc_image image;
		image.dim_x = src_mip_w;
		image.dim_y = src_mip_h;
		image.dim_z = 1;
		image.data_type = ASTCENC_TYPE_U8;
		if (is_hdr) {
			image.data_type = ASTCENC_TYPE_F32;
		}
		image.data = (void **)(&slices);

		// Compute the number of ASTC blocks in each dimension.
		unsigned int block_count_x = (src_mip_w + block_x - 1) / block_x;
		unsigned int block_count_y = (src_mip_h + block_y - 1) / block_y;
		size_t comp_len = block_count_x * block_count_y * 16;

		const astcenc_swizzle swizzle = {
			ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
		};

		status = astcenc_compress_image(context, &image, &swizzle, dest_mip_write, comp_len, 0);

		ERR_BREAK_MSG(status != ASTCENC_SUCCESS,
				vformat("astcenc: ASTC image compression failed: %s.", astcenc_get_error_string(status)));
		astcenc_compress_reset(context);
	}

	astcenc_context_free(context);

	// Replace original image with compressed one.

	r_img->set_data(width, height, mipmaps, target_format, dest_data);

	print_verbose(vformat("astcenc: Encoding took %d ms.", OS::get_singleton()->get_ticks_msec() - start_time));
}

void _decompress_astc(Image *r_img) {
	uint64_t start_time = OS::get_singleton()->get_ticks_msec();

	// Determine decompression parameters from image format.

	Image::Format img_format = r_img->get_format();
	bool is_hdr = false;
	unsigned int block_x = 0;
	unsigned int block_y = 0;
	if (img_format == Image::FORMAT_ASTC_4x4) {
		block_x = 4;
		block_y = 4;
		is_hdr = false;
	} else if (img_format == Image::FORMAT_ASTC_4x4_HDR) {
		block_x = 4;
		block_y = 4;
		is_hdr = true;
	} else if (img_format == Image::FORMAT_ASTC_8x8) {
		block_x = 8;
		block_y = 8;
		is_hdr = false;
	} else if (img_format == Image::FORMAT_ASTC_8x8_HDR) {
		block_x = 8;
		block_y = 8;
		is_hdr = true;
	} else {
		ERR_FAIL_MSG("astcenc: Cannot decompress Image with a non-ASTC format.");
	}

	// Initialize astcenc.

	astcenc_profile profile = ASTCENC_PRF_LDR;
	if (is_hdr) {
		profile = ASTCENC_PRF_HDR;
	}
	astcenc_config config;
	const float quality = ASTCENC_PRE_MEDIUM;

	astcenc_error status = astcenc_config_init(profile, block_x, block_y, 1, quality, 0, &config);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS,
			vformat("astcenc: Configuration initialization failed: %s.", astcenc_get_error_string(status)));

	// Context allocation.

	astcenc_context *context = nullptr;
	const unsigned int thread_count = 1;

	status = astcenc_context_alloc(&config, thread_count, &context);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS,
			vformat("astcenc: Context allocation failed: %s.", astcenc_get_error_string(status)));

	Image::Format target_format = is_hdr ? Image::FORMAT_RGBAF : Image::FORMAT_RGBA8;

	const bool mipmaps = r_img->has_mipmaps();
	int width = r_img->get_width();
	int height = r_img->get_height();
	int64_t dest_size = Image::get_image_data_size(width, height, target_format, mipmaps);
	Vector<uint8_t> dest_data;
	dest_data.resize(dest_size);
	uint8_t *dest_write = dest_data.ptrw();

	// Decompress image.

	Vector<uint8_t> image_data = r_img->get_data();
	int mip_count = mipmaps ? Image::get_image_required_mipmaps(width, height, target_format) : 0;

	for (int i = 0; i < mip_count + 1; i++) {
		int src_mip_w, src_mip_h;

		int64_t src_ofs = Image::get_image_mipmap_offset_and_dimensions(width, height, r_img->get_format(), i, src_mip_w, src_mip_h);
		const uint8_t *src_data = &image_data.ptr()[src_ofs];
		int64_t src_size;
		if (i == mip_count) {
			src_size = image_data.size() - src_ofs;
		} else {
			int auxw, auxh;
			src_size = Image::get_image_mipmap_offset_and_dimensions(width, height, r_img->get_format(), i + 1, auxw, auxh) - src_ofs;
		}

		int dst_mip_w, dst_mip_h;
		int64_t dst_ofs = Image::get_image_mipmap_offset_and_dimensions(width, height, target_format, i, dst_mip_w, dst_mip_h);
		// Ensure that mip offset is a multiple of 8 (etcpak expects uint64_t pointer).
		ERR_FAIL_COND(dst_ofs % 8 != 0);
		uint8_t *dest_mip_write = (uint8_t *)&dest_write[dst_ofs];

		astcenc_image image;
		image.dim_x = dst_mip_w;
		image.dim_y = dst_mip_h;
		image.dim_z = 1;
		image.data_type = ASTCENC_TYPE_U8;
		if (is_hdr) {
			target_format = Image::FORMAT_RGBAF;
			image.data_type = ASTCENC_TYPE_F32;
		}

		image.data = (void **)(&dest_mip_write);

		const astcenc_swizzle swizzle = {
			ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
		};

		status = astcenc_decompress_image(context, src_data, src_size, &image, &swizzle, 0);
		ERR_BREAK_MSG(status != ASTCENC_SUCCESS,
				vformat("astcenc: ASTC decompression failed: %s.", astcenc_get_error_string(status)));
		ERR_BREAK_MSG(image.dim_z > 1,
				"astcenc: ASTC decompression failed because this is a 3D texture, which is not supported.");
		astcenc_compress_reset(context);
	}
	astcenc_context_free(context);

	// Replace original image with compressed one.

	r_img->set_data(width, height, mipmaps, target_format, dest_data);

	print_verbose(vformat("astcenc: Decompression took %d ms.", OS::get_singleton()->get_ticks_msec() - start_time));
}
