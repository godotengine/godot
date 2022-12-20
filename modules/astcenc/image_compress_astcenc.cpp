/*************************************************************************/
/*  image_compress_astcenc.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "image_compress_astcenc.h"

#include "core/os/os.h"
#include "core/string/print_string.h"

#include "thirdparty/astcenc/astcenc.h"
#include <cstring>

void _compress_astc(Image *r_img, float p_quality, Image::ASTCFormat p_format) {
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
	Image::Format target_format = Image::FORMAT_RGBA8;

	const bool mipmaps = r_img->has_mipmaps();
	int width = r_img->get_width();
	int height = r_img->get_height();

	print_verbose(vformat("astcenc: Encoding image size %dx%d to format %s.", width, height, Image::get_format_name(target_format)));
	Vector<uint8_t> image_data = r_img->get_data();
	ERR_FAIL_NULL(image_data.ptrw());

	unsigned int block_x = 4;
	unsigned int block_y = 4;
	astcenc_profile profile = ASTCENC_PRF_LDR;

	if (p_format == Image::ASTCFormat::ASTC_FORMAT_4x4 && !is_hdr) {
		target_format = Image::FORMAT_ASTC_4x4;
	} else if (p_format == Image::ASTCFormat::ASTC_FORMAT_4x4 && is_hdr) {
		target_format = Image::FORMAT_ASTC_4x4_HDR;
		profile = ASTCENC_PRF_HDR;
	} else if (p_format == Image::ASTCFormat::ASTC_FORMAT_8x8 && !is_hdr) {
		target_format = Image::FORMAT_ASTC_8x8;
		block_x = 8;
		block_y = 8;
	} else if (p_format == Image::ASTCFormat::ASTC_FORMAT_8x8 && is_hdr) {
		target_format = Image::FORMAT_ASTC_8x8_HDR;
		block_x = 8;
		block_y = 8;
		profile = ASTCENC_PRF_HDR;
	}
	const unsigned int thread_count = 1;
	const float quality = ASTCENC_PRE_MEDIUM;
	const astcenc_swizzle swizzle{
		ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
	};

	// Compute the number of ASTC blocks in each dimension.
	unsigned int block_count_x = (width + block_x - 1) / block_x;
	unsigned int block_count_y = (height + block_y - 1) / block_y;

	astcenc_config config;
	config.block_x = block_x;
	config.block_y = block_y;
	config.profile = profile;

	astcenc_error status;
	status = astcenc_config_init(profile, width, height, 1, quality, 0, &config);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS, vformat("ASTC configuration initialization failed. %s", astcenc_get_error_string(status)));

	astcenc_context *context;
	status = astcenc_context_alloc(&config, thread_count, &context);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS, vformat("ASTC context allocation failed. %s", astcenc_get_error_string(status)));

	astcenc_image image;
	image.dim_x = width;
	image.dim_y = height;
	image.dim_z = 1;
	image.data_type = ASTCENC_TYPE_U8;
	if (is_hdr) {
		image.data_type = ASTCENC_TYPE_F32;
	}
	uint8_t *slices = image_data.ptrw();
	image.data = reinterpret_cast<void **>(&slices);

	size_t comp_len = block_count_x * block_count_y * 16;
	Vector<uint8_t> compressed_data;
	compressed_data.resize(comp_len);
	compressed_data.fill(0);

	status = astcenc_compress_image(context, &image, &swizzle, compressed_data.ptrw(), comp_len, 0);
	ERR_FAIL_COND_MSG(status != ASTCENC_SUCCESS, vformat("ASTC compression failed. %s", astcenc_get_error_string(status)));

	r_img->create_from_data(width, height, mipmaps, target_format, compressed_data);
}
