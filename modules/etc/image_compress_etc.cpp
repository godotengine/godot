/*************************************************************************/
/*  image_compress_etc.cpp                                               */
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

#include "image_compress_etc.h"

#include "core/image.h"
#include "core/os/os.h"
#include "core/print_string.h"

#include <Etc.h>
#include <EtcFilter.h>

static Image::Format _get_etc2_mode(Image::DetectChannels format) {
	switch (format) {
		case Image::DETECTED_R:
			return Image::FORMAT_ETC2_R11;

		case Image::DETECTED_RG:
			return Image::FORMAT_ETC2_RG11;

		case Image::DETECTED_RGB:
			return Image::FORMAT_ETC2_RGB8;

		case Image::DETECTED_RGBA:
			return Image::FORMAT_ETC2_RGBA8;

		// TODO: would be nice if we could use FORMAT_ETC2_RGB8A1 for FORMAT_RGBA5551
		default:
			// TODO: Kept for compatibility, but should be investigated whether it's correct or if it should error out
			return Image::FORMAT_ETC2_RGBA8;
	}
}

static Etc::Image::Format _image_format_to_etc2comp_format(Image::Format format) {
	switch (format) {
		case Image::FORMAT_ETC:
			return Etc::Image::Format::ETC1;

		case Image::FORMAT_ETC2_R11:
			return Etc::Image::Format::R11;

		case Image::FORMAT_ETC2_R11S:
			return Etc::Image::Format::SIGNED_R11;

		case Image::FORMAT_ETC2_RG11:
			return Etc::Image::Format::RG11;

		case Image::FORMAT_ETC2_RG11S:
			return Etc::Image::Format::SIGNED_RG11;

		case Image::FORMAT_ETC2_RGB8:
			return Etc::Image::Format::RGB8;

		case Image::FORMAT_ETC2_RGBA8:
			return Etc::Image::Format::RGBA8;

		case Image::FORMAT_ETC2_RGB8A1:
			return Etc::Image::Format::RGB8A1;

		default:
			ERR_FAIL_V(Etc::Image::Format::UNKNOWN);
	}
}

static void _compress_etc(Image *p_img, float p_lossy_quality, bool force_etc1_format, Image::CompressSource p_source) {
	Image::Format img_format = p_img->get_format();
	Image::DetectChannels detected_channels = p_img->get_detected_channels();

	if (p_source == Image::COMPRESS_SOURCE_LAYERED) {
		//keep what comes in
		switch (p_img->get_format()) {
			case Image::FORMAT_L8: {
				detected_channels = Image::DETECTED_L;
			} break;
			case Image::FORMAT_LA8: {
				detected_channels = Image::DETECTED_LA;
			} break;
			case Image::FORMAT_R8: {
				detected_channels = Image::DETECTED_R;
			} break;
			case Image::FORMAT_RG8: {
				detected_channels = Image::DETECTED_RG;
			} break;
			case Image::FORMAT_RGB8: {
				detected_channels = Image::DETECTED_RGB;
			} break;
			case Image::FORMAT_RGBA8:
			case Image::FORMAT_RGBA4444:
			case Image::FORMAT_RGBA5551: {
				detected_channels = Image::DETECTED_RGBA;
			} break;
			default: {
			}
		}
	}

	if (p_source == Image::COMPRESS_SOURCE_SRGB && (detected_channels == Image::DETECTED_R || detected_channels == Image::DETECTED_RG)) {
		//R and RG do not support SRGB
		detected_channels = Image::DETECTED_RGB;
	}

	if (p_source == Image::COMPRESS_SOURCE_NORMAL) {
		//use RG channels only for normal
		detected_channels = Image::DETECTED_RG;
	}

	if (img_format >= Image::FORMAT_DXT1) {
		return; //do not compress, already compressed
	}

	if (img_format > Image::FORMAT_RGBA8) {
		// TODO: we should be able to handle FORMAT_RGBA4444 and FORMAT_RGBA5551 eventually
		return;
	}

	if (force_etc1_format) {
		// If VRAM compression is using ETC, but image has alpha, convert to RGBA4444 or LA8
		// This saves space while maintaining the alpha channel
		if (detected_channels == Image::DETECTED_RGBA) {
			if (p_img->has_mipmaps()) {
				// Image doesn't support mipmaps with RGBA4444 textures
				p_img->clear_mipmaps();
			}
			p_img->convert(Image::FORMAT_RGBA4444);
			return;
		} else if (detected_channels == Image::DETECTED_LA) {
			p_img->convert(Image::FORMAT_LA8);
			return;
		}
	}

	uint32_t imgw = p_img->get_width(), imgh = p_img->get_height();

	Image::Format etc_format = force_etc1_format ? Image::FORMAT_ETC : _get_etc2_mode(detected_channels);

	Ref<Image> img = p_img->duplicate();

	if (img->get_format() != Image::FORMAT_RGBA8) {
		img->convert(Image::FORMAT_RGBA8); //still uses RGBA to convert
	}

	if (img->has_mipmaps()) {
		if (next_power_of_2(imgw) != imgw || next_power_of_2(imgh) != imgh) {
			img->resize_to_po2();
			imgw = img->get_width();
			imgh = img->get_height();
		}
	} else {
		if (imgw % 4 != 0 || imgh % 4 != 0) {
			if (imgw % 4) {
				imgw += 4 - imgw % 4;
			}
			if (imgh % 4) {
				imgh += 4 - imgh % 4;
			}

			img->resize(imgw, imgh);
		}
	}

	PoolVector<uint8_t>::Read r = img->get_data().read();
	ERR_FAIL_COND(!r.ptr());

	unsigned int target_size = Image::get_image_data_size(imgw, imgh, etc_format, p_img->has_mipmaps());
	int mmc = 1 + (p_img->has_mipmaps() ? Image::get_image_required_mipmaps(imgw, imgh, etc_format) : 0);

	PoolVector<uint8_t> dst_data;
	dst_data.resize(target_size);

	PoolVector<uint8_t>::Write w = dst_data.write();

	// prepare parameters to be passed to etc2comp
	int num_cpus = OS::get_singleton()->get_processor_count();
	int encoding_time = 0;

	float effort = 0.0; //default, reasonable time

	if (p_lossy_quality > 0.95) {
		effort = 80;
	} else if (p_lossy_quality > 0.85) {
		effort = 60;
	} else if (p_lossy_quality > 0.75) {
		effort = 40;
	}

	Etc::ErrorMetric error_metric = Etc::ErrorMetric::RGBX; // NOTE: we can experiment with other error metrics
	Etc::Image::Format etc2comp_etc_format = _image_format_to_etc2comp_format(etc_format);

	int wofs = 0;

	print_verbose("ETC: Begin encoding, format: " + Image::get_format_name(etc_format));
	uint64_t t = OS::get_singleton()->get_ticks_msec();
	for (int i = 0; i < mmc; i++) {
		// convert source image to internal etc2comp format (which is equivalent to Image::FORMAT_RGBAF)
		// NOTE: We can alternatively add a case to Image::convert to handle Image::FORMAT_RGBAF conversion.
		int mipmap_ofs = 0, mipmap_size = 0, mipmap_w = 0, mipmap_h = 0;
		img->get_mipmap_offset_size_and_dimensions(i, mipmap_ofs, mipmap_size, mipmap_w, mipmap_h);
		const uint8_t *src = &r[mipmap_ofs];

		Etc::ColorFloatRGBA *src_rgba_f = new Etc::ColorFloatRGBA[mipmap_w * mipmap_h];
		for (int j = 0; j < mipmap_w * mipmap_h; j++) {
			int si = j * 4; // RGBA8
			src_rgba_f[j] = Etc::ColorFloatRGBA::ConvertFromRGBA8(src[si], src[si + 1], src[si + 2], src[si + 3]);
		}

		unsigned char *etc_data = nullptr;
		unsigned int etc_data_len = 0;
		unsigned int extended_width = 0, extended_height = 0;
		Etc::Encode((float *)src_rgba_f, mipmap_w, mipmap_h, etc2comp_etc_format, error_metric, effort, num_cpus, num_cpus, &etc_data, &etc_data_len, &extended_width, &extended_height, &encoding_time);

		CRASH_COND(wofs + etc_data_len > target_size);
		memcpy(&w[wofs], etc_data, etc_data_len);
		wofs += etc_data_len;

		delete[] etc_data;
		delete[] src_rgba_f;
	}

	print_verbose("ETC: Time encoding: " + rtos(OS::get_singleton()->get_ticks_msec() - t));

	p_img->create(imgw, imgh, p_img->has_mipmaps(), etc_format, dst_data);
}

static void _compress_etc1(Image *p_img, float p_lossy_quality) {
	_compress_etc(p_img, p_lossy_quality, true, Image::COMPRESS_SOURCE_GENERIC);
}

static void _compress_etc2(Image *p_img, float p_lossy_quality, Image::CompressSource p_source) {
	_compress_etc(p_img, p_lossy_quality, false, p_source);
}

void _register_etc_compress_func() {
	Image::_image_compress_etc1_func = _compress_etc1;
	Image::_image_compress_etc2_func = _compress_etc2;
}
