/*************************************************************************/
/*  image_etc.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <future>
#include <limits>
#include <memory>

#include "thirdparty/etcpak/Bitmap.hpp"
#include "thirdparty/etcpak/BlockData.hpp"
#include "thirdparty/etcpak/CpuArch.hpp"
#include "thirdparty/etcpak/DataProvider.hpp"
#include "thirdparty/etcpak/Dither.hpp"
#include "thirdparty/etcpak/Error.hpp"
#include "thirdparty/etcpak/System.hpp"
#include "thirdparty/etcpak/TaskDispatch.hpp"
#include "thirdparty/etcpak/Timing.hpp"

#include "core/image.h"
#include "core/os/copymem.h"
#include "core/os/os.h"
#include "core/print_string.h"
#include "image_etc.h"

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

	uint32_t imgw = p_img->get_width(), imgh = p_img->get_height();

	Image::Format etc_format = force_etc1_format ? Image::FORMAT_ETC : _get_etc2_mode(detected_channels);

	Ref<Image> img = p_img->duplicate();

	if (img->get_format() != Image::FORMAT_RGBA8) {
		img->convert(Image::FORMAT_RGBA8);
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
	print_verbose("ETCPAK: Begin encoding, format: " + Image::get_format_name(etc_format));
	uint64_t t = OS::get_singleton()->get_ticks_msec();
	BlockData::Type type = BlockData::Etc1;
	if (etc_format == Image::FORMAT_ETC || force_etc1_format) {
		etc_format = Image::FORMAT_ETC;
	} else if (etc_format == Image::FORMAT_ETC2_RGB8) {
		type = BlockData::Etc2_RGB;
		etc_format = Image::FORMAT_ETC2_RGB8;
	} else if (etc_format == Image::FORMAT_ETC2_RGBA8) {
		type = BlockData::Etc2_RGBA;
		etc_format = Image::FORMAT_ETC2_RGBA8;
	} else if (etc_format == Image::FORMAT_ETC2_R11) {
		type = BlockData::Etc2_RGB;
		etc_format = Image::FORMAT_ETC2_RGB8;
	} else if (etc_format == Image::FORMAT_ETC2_RG11) {
		type = BlockData::Etc2_RGB;
		etc_format = Image::FORMAT_ETC2_RGB8;
	} else {
		type = BlockData::Etc2_RGBA;
		etc_format = Image::FORMAT_ETC2_RGBA8;
	}
	unsigned int target_size = Image::get_image_data_size(imgw, imgh, etc_format, p_img->has_mipmaps());
	PoolVector<uint8_t> dst_data;
	dst_data.resize(target_size);
	PoolVector<uint8_t>::Write w = dst_data.write();
	const bool dither = false;
	const bool mipmap = true;
	const size_t stride = 4;
	const size_t block = stride * stride;
	BlockDataPtr bd = std::make_shared<BlockData>(v2i(img->get_size().x, img->get_size().y), mipmap, type);

	Vector<uint32_t> tex;
	tex.resize(imgh * imgw);
	img->lock();
	size_t count = 0;
	for (size_t y = 0; y < imgh; y++) {
		for (size_t x = 0; x < imgw; x++) {
			Color c = img->get_pixel(x, y);
			tex.ptrw()[count] = c.to_argb32();
			count++;
		}
	}
	img->unlock();
	if (etc_format == Image::FORMAT_ETC2_RGBA8) {
		bd->ProcessRGBA(tex.ptr(), imgh * imgw / block, 0, MAX(4, imgw), dither);
	} else {
		bd->Process(tex.ptr(), imgh * imgw / block, 0, MAX(4, imgw), Channels::RGB, dither);
	}
	int wofs = 0;
	memcpy(&w[wofs], bd->Data(), target_size);
	print_verbose("ETCPAK: Time encoding: " + rtos(OS::get_singleton()->get_ticks_msec() - t));
	p_img->create(imgw, imgh, p_img->has_mipmaps(), etc_format, dst_data);
	bd.reset();
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
