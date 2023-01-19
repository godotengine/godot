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

void _compress_astc(Image *r_img, float p_lossy_quality, Image::ASTCFormat p_format) {
	uint64_t start_time = OS::get_singleton()->get_ticks_msec();

	// TODO: See how to handle lossy quality.

	Image::Format img_format = r_img->get_format();
	if (img_format >= Image::FORMAT_DXT1) {
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

	print_verbose(vformat("astcenc: Encoding image size %dx%d to format %s%s.", width, height, Image::get_format_name(target_format), mipmaps ? ", with mipmaps" : ""));

	// Initialize astcenc.

	astcenc_config config;
	config.block_x = block_x;
	config.block_y = block_y;
	config.profile = profile;
	const float quality = ASTCENC_PRE_MEDIUM;

	astcenc_error status = astcenc_config_init(profile, block_x, block_y, block_x, quality, 0, &config);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS,
			vformat("astcenc: Configuration initialization failed: %s.", astcenc_get_error_string(status)));

	// Context allocation.

	astcenc_context *context;
	const unsigned int thread_count = OS::get_singleton()->get_processor_count();

	status = astcenc_context_alloc(&config, thread_count, &context);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS,
			vformat("astcenc: Context allocation failed: %s.", astcenc_get_error_string(status)));

	// Compress image.

	Vector<uint8_t> image_data = r_img->get_data();
	uint8_t *slices = image_data.ptrw();

	astcenc_image image;
	image.dim_x = width;
	image.dim_y = height;
	image.dim_z = 1;
	image.data_type = ASTCENC_TYPE_U8;
	if (is_hdr) {
		image.data_type = ASTCENC_TYPE_F32;
	}
	image.data = reinterpret_cast<void **>(&slices);

	// Compute the number of ASTC blocks in each dimension.
	unsigned int block_count_x = (width + block_x - 1) / block_x;
	unsigned int block_count_y = (height + block_y - 1) / block_y;
	size_t comp_len = block_count_x * block_count_y * 16;

	Vector<uint8_t> compressed_data;
	compressed_data.resize(comp_len);
	compressed_data.fill(0);

	const astcenc_swizzle swizzle = {
		ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
	};

	status = astcenc_compress_image(context, &image, &swizzle, compressed_data.ptrw(), comp_len, 0);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS,
			vformat("astcenc: ASTC image compression failed: %s.", astcenc_get_error_string(status)));

	// Replace original image with compressed one.

	r_img->set_data(width, height, mipmaps, target_format, compressed_data);

	print_verbose(vformat("astcenc: Encoding took %s ms.", rtos(OS::get_singleton()->get_ticks_msec() - start_time)));
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

	astcenc_error status = astcenc_config_init(profile, block_x, block_y, block_x, quality, 0, &config);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS,
			vformat("astcenc: Configuration initialization failed: %s.", astcenc_get_error_string(status)));

	// Context allocation.

	astcenc_context *context = nullptr;
	const unsigned int thread_count = OS::get_singleton()->get_processor_count();

	status = astcenc_context_alloc(&config, thread_count, &context);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS,
			vformat("astcenc: Context allocation failed: %s.", astcenc_get_error_string(status)));

	// Decompress image.

	const bool mipmaps = r_img->has_mipmaps();
	int width = r_img->get_width();
	int height = r_img->get_height();

	astcenc_image image;
	image.dim_x = width;
	image.dim_y = height;
	image.dim_z = 1;
	image.data_type = ASTCENC_TYPE_U8;
	Image::Format target_format = Image::FORMAT_RGBA8;
	if (is_hdr) {
		target_format = Image::FORMAT_RGBAF;
		image.data_type = ASTCENC_TYPE_F32;
	}

	Vector<uint8_t> image_data = r_img->get_data();

	Vector<uint8_t> new_image_data;
	new_image_data.resize(Image::get_image_data_size(width, height, target_format, false));
	new_image_data.fill(0);
	uint8_t *slices = new_image_data.ptrw();
	image.data = reinterpret_cast<void **>(&slices);

	const astcenc_swizzle swizzle = {
		ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
	};

	status = astcenc_decompress_image(context, image_data.ptr(), image_data.size(), &image, &swizzle, 0);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS,
			vformat("astcenc: ASTC decompression failed: %s.", astcenc_get_error_string(status)));
	ERR_FAIL_COND_MSG(image.dim_z > 1,
			"astcenc: ASTC decompression failed because this is a 3D texture, which is not supported.");

	// Replace original image with compressed one.

	Image::Format image_format = Image::FORMAT_RGBA8;
	if (image.data_type == ASTCENC_TYPE_F32) {
		image_format = Image::FORMAT_RGBAF;
	} else if (image.data_type == ASTCENC_TYPE_U8) {
		image_format = Image::FORMAT_RGBA8;
	} else if (image.data_type == ASTCENC_TYPE_F16) {
		image_format = Image::FORMAT_RGBAH;
	} else {
		ERR_FAIL_MSG("astcenc: ASTC decompression failed with an unknown format.");
	}

	r_img->set_data(image.dim_x, image.dim_y, mipmaps, image_format, new_image_data);

	print_verbose(vformat("astcenc: Decompression took %s ms.", rtos(OS::get_singleton()->get_ticks_msec() - start_time)));
}
