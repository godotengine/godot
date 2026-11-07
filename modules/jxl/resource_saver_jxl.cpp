/**************************************************************************/
/*  resource_saver_jxl.cpp                                                */
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

#include "resource_saver_jxl.h"

#include "jxl_parallel_runner.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "scene/resources/image_texture.h"

#include <jxl/encode.h>

// Replaces infinity with the largest finite value and NaN with 0.
static void _replace_non_finite(Vector<uint8_t> &r_data, JxlDataType p_type) {
	if (p_type == JXL_TYPE_FLOAT) {
		uint32_t *samples = reinterpret_cast<uint32_t *>(r_data.ptrw());
		for (int64_t i = 0; i < r_data.size() / 4; i++) {
			if ((samples[i] & 0x7f800000) == 0x7f800000) {
				samples[i] = (samples[i] & 0x007fffff) ? 0 : (samples[i] & 0x80000000) | 0x7f7fffff;
			}
		}
	} else if (p_type == JXL_TYPE_FLOAT16) {
		uint16_t *samples = reinterpret_cast<uint16_t *>(r_data.ptrw());
		for (int64_t i = 0; i < r_data.size() / 2; i++) {
			if ((samples[i] & 0x7c00) == 0x7c00) {
				samples[i] = (samples[i] & 0x03ff) ? 0 : (samples[i] & 0x8000) | 0x7bff;
			}
		}
	}
}

static bool _has_mixed_sign_channel(const Vector<uint8_t> &p_data, uint32_t p_channels) {
	const uint32_t *samples = reinterpret_cast<const uint32_t *>(p_data.ptr());
	const int64_t count = p_data.size() / sizeof(uint32_t);
	uint32_t has_negative = 0;
	uint32_t has_positive = 0;
	for (int64_t i = 0; i < count; i++) {
		const uint32_t channel_bit = 1u << (i % p_channels);
		if (samples[i] >> 31) {
			has_negative |= channel_bit;
		} else {
			has_positive |= channel_bit;
		}
		if (has_negative & has_positive) {
			return true;
		}
	}
	return false;
}

static Vector<uint8_t> _jxl_encode(JxlEncoder *p_enc, const Ref<Image> &p_image, bool p_lossy, float p_quality) {
	Ref<Image> img = p_image->duplicate();
	if (img->is_compressed()) {
		ERR_FAIL_COND_V_MSG(img->decompress() != OK, Vector<uint8_t>(), "Couldn't decompress image.");
	}
	img->clear_mipmaps();

	const Image::Format src_format = img->get_format();
	// detect_alpha() doesn't work on float formats.
	bool has_alpha = src_format == Image::FORMAT_RGBAF || src_format == Image::FORMAT_RGBAH;
	bool is_gray = false;
	JxlPixelFormat format = {};
	JxlBasicInfo info;
	JxlEncoderInitBasicInfo(&info);

	switch (src_format) {
		case Image::FORMAT_RF:
		case Image::FORMAT_RGF:
		case Image::FORMAT_RGBF:
		case Image::FORMAT_RGBAF:
		case Image::FORMAT_RGBE9995: {
			is_gray = src_format == Image::FORMAT_RF;
			img->convert(is_gray ? Image::FORMAT_RF : (has_alpha ? Image::FORMAT_RGBAF : Image::FORMAT_RGBF));
			format.data_type = JXL_TYPE_FLOAT;
			info.bits_per_sample = 32;
			info.exponent_bits_per_sample = 8;
		} break;
		case Image::FORMAT_RH:
		case Image::FORMAT_RGH:
		case Image::FORMAT_RGBH:
		case Image::FORMAT_RGBAH: {
			is_gray = src_format == Image::FORMAT_RH;
			img->convert(is_gray ? Image::FORMAT_RH : (has_alpha ? Image::FORMAT_RGBAH : Image::FORMAT_RGBH));
			format.data_type = JXL_TYPE_FLOAT16;
			info.bits_per_sample = 16;
			info.exponent_bits_per_sample = 5;
		} break;
		default: {
			has_alpha = img->detect_alpha() != Image::ALPHA_NONE;
			is_gray = src_format == Image::FORMAT_L8 || src_format == Image::FORMAT_LA8;
			if (is_gray) {
				img->convert(has_alpha ? Image::FORMAT_LA8 : Image::FORMAT_L8);
			} else {
				img->convert(has_alpha ? Image::FORMAT_RGBA8 : Image::FORMAT_RGB8);
			}
			format.data_type = JXL_TYPE_UINT8;
			info.bits_per_sample = 8;
			info.exponent_bits_per_sample = 0;
		} break;
	}

	info.xsize = img->get_width();
	info.ysize = img->get_height();
	info.num_color_channels = is_gray ? 1 : 3;
	if (has_alpha) {
		info.num_extra_channels = 1;
		info.alpha_bits = info.bits_per_sample;
		info.alpha_exponent_bits = info.exponent_bits_per_sample;
	}
	// Required for lossless.
	info.uses_original_profile = p_lossy ? JXL_FALSE : JXL_TRUE;

	format.num_channels = info.num_color_channels + (has_alpha ? 1 : 0);
	format.endianness = JXL_NATIVE_ENDIAN;
	format.align = 0;

	JxlColorEncoding color_encoding;
	if (format.data_type == JXL_TYPE_UINT8) {
		JxlColorEncodingSetToSRGB(&color_encoding, is_gray);
	} else {
		JxlColorEncodingSetToLinearSRGB(&color_encoding, is_gray);
	}

	ERR_FAIL_COND_V(JxlEncoderSetBasicInfo(p_enc, &info) != JXL_ENC_SUCCESS, Vector<uint8_t>());
	ERR_FAIL_COND_V(JxlEncoderSetColorEncoding(p_enc, &color_encoding) != JXL_ENC_SUCCESS, Vector<uint8_t>());

	JxlEncoderFrameSettings *settings = JxlEncoderFrameSettingsCreate(p_enc, nullptr);
	ERR_FAIL_NULL_V(settings, Vector<uint8_t>());
	const int effort = CLAMP(int(GLOBAL_GET("rendering/textures/jxl_compression/effort")), 1, 10);
	ERR_FAIL_COND_V(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, effort) != JXL_ENC_SUCCESS, Vector<uint8_t>());
	if (p_lossy) {
		// Otherwise quality 1.0 would switch to lossless.
		const float distance = MAX(JxlEncoderDistanceFromQuality(p_quality * 100.0f), 0.01f);
		ERR_FAIL_COND_V(JxlEncoderSetFrameDistance(settings, distance) != JXL_ENC_SUCCESS, Vector<uint8_t>());
	}

	Vector<uint8_t> data = img->get_data();
	if (p_lossy) {
		// libjxl fails on infinite or NaN values when encoding lossy.
		_replace_non_finite(data, format.data_type);
	} else {
		ERR_FAIL_COND_V(JxlEncoderSetFrameLossless(settings, JXL_TRUE) != JXL_ENC_SUCCESS, Vector<uint8_t>());
		// libjxl fails on float channels with both negative and positive values unless prediction is off.
		if (format.data_type == JXL_TYPE_FLOAT && _has_mixed_sign_channel(data, format.num_channels)) {
			ERR_FAIL_COND_V(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 0) != JXL_ENC_SUCCESS, Vector<uint8_t>());
		}
	}

	ERR_FAIL_COND_V_MSG(JxlEncoderAddImageFrame(settings, &format, data.ptr(), data.size()) != JXL_ENC_SUCCESS, Vector<uint8_t>(), "JXL encoding failed.");
	JxlEncoderCloseInput(p_enc);

	Vector<uint8_t> dst;
	dst.resize(64 * 1024);
	size_t written = 0;
	for (;;) {
		uint8_t *next_out = dst.ptrw() + written;
		size_t avail_out = dst.size() - written;
		const JxlEncoderStatus status = JxlEncoderProcessOutput(p_enc, &next_out, &avail_out);
		written = next_out - dst.ptr();
		if (status == JXL_ENC_SUCCESS) {
			break;
		}
		ERR_FAIL_COND_V_MSG(status != JXL_ENC_NEED_MORE_OUTPUT, Vector<uint8_t>(), "JXL encoding failed.");
		ERR_FAIL_COND_V(dst.resize(dst.size() * 2) != OK, Vector<uint8_t>());
	}
	dst.resize(written);
	return dst;
}

Error ResourceSaverJXL::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	Ref<ImageTexture> texture = p_resource;

	ERR_FAIL_COND_V_MSG(texture.is_null(), ERR_INVALID_PARAMETER, "Can't save invalid texture as JXL.");
	ERR_FAIL_COND_V_MSG(!texture->get_width(), ERR_INVALID_PARAMETER, "Can't save empty texture as JXL.");

	return save_image(p_path, texture->get_image());
}

Error ResourceSaverJXL::save_image(const String &p_path, const Ref<Image> &p_img, const bool p_lossy, const float p_quality) {
	Vector<uint8_t> buffer = save_image_to_buffer(p_img, p_lossy, p_quality);
	ERR_FAIL_COND_V(buffer.is_empty(), ERR_CANT_CREATE);

	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V_MSG(err, err, vformat("Can't save JXL at path: '%s'.", p_path));

	file->store_buffer(buffer.ptr(), buffer.size());
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		return ERR_CANT_CREATE;
	}

	return OK;
}

Vector<uint8_t> ResourceSaverJXL::save_image_to_buffer(const Ref<Image> &p_img, const bool p_lossy, const float p_quality) {
	ERR_FAIL_COND_V(p_img.is_null() || p_img->is_empty(), Vector<uint8_t>());

	JxlEncoder *enc = JxlEncoderCreate(nullptr);
	ERR_FAIL_NULL_V(enc, Vector<uint8_t>());
	if (JxlEncoderSetParallelRunner(enc, jxl_parallel_runner, nullptr) != JXL_ENC_SUCCESS) {
		JxlEncoderDestroy(enc);
		ERR_FAIL_V(Vector<uint8_t>());
	}
	Vector<uint8_t> buffer = _jxl_encode(enc, p_img, p_lossy, CLAMP(p_quality, 0.0f, 1.0f));
	JxlEncoderDestroy(enc);

	return buffer;
}

bool ResourceSaverJXL::recognize(const Ref<Resource> &p_resource) const {
	return p_resource.is_valid() && p_resource->is_class("ImageTexture");
}

void ResourceSaverJXL::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<ImageTexture>(*p_resource)) {
		p_extensions->push_back("jxl");
	}
}

ResourceSaverJXL::ResourceSaverJXL() {
	Image::save_jxl_func = &save_image;
	Image::save_jxl_buffer_func = &save_image_to_buffer;
}
