/*************************************************************************/
/*  image_etcpak.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "image_etcpak.h"

#include "core/os/copymem.h"
#include "core/os/os.h"
#include "core/string/print_string.h"

#include "thirdparty/etcpak/ProcessDxtc.hpp"
#include "thirdparty/etcpak/ProcessRGB.hpp"

// thresholds for the early compression-mode decision scheme in QuickETC2
// which can be changed by the option -e
float ecmd_threshold[3] = { 0.03f, 0.09f, 0.38f };

EtcpakType _determine_etc_type(Image::UsedChannels p_source) {
	switch (p_source) {
		case Image::USED_CHANNELS_L:
			return EtcpakType::ETCPAK_TYPE_ETC1;
		case Image::USED_CHANNELS_LA:
			return EtcpakType::ETCPAK_TYPE_ETC2_ALPHA;
		case Image::USED_CHANNELS_R:
			return EtcpakType::ETCPAK_TYPE_ETC2;
		case Image::USED_CHANNELS_RG:
			return EtcpakType::ETCPAK_TYPE_ETC2_RA_AS_RG;
		case Image::USED_CHANNELS_RGB:
			return EtcpakType::ETCPAK_TYPE_ETC2;
		case Image::USED_CHANNELS_RGBA:
			return EtcpakType::ETCPAK_TYPE_ETC2_ALPHA;
		default:
			return EtcpakType::ETCPAK_TYPE_ETC2_ALPHA;
	}
}

EtcpakType _determine_dxt_type(Image::UsedChannels p_source) {
	switch (p_source) {
		case Image::USED_CHANNELS_L:
			return EtcpakType::ETCPAK_TYPE_DXT1;
		case Image::USED_CHANNELS_LA:
			return EtcpakType::ETCPAK_TYPE_DXT5;
		case Image::USED_CHANNELS_R:
			return EtcpakType::ETCPAK_TYPE_DXT5;
		case Image::USED_CHANNELS_RG:
			return EtcpakType::ETCPAK_TYPE_DXT5_RA_AS_RG;
		case Image::USED_CHANNELS_RGB:
			return EtcpakType::ETCPAK_TYPE_DXT5;
		case Image::USED_CHANNELS_RGBA:
			return EtcpakType::ETCPAK_TYPE_DXT5;
		default:
			return EtcpakType::ETCPAK_TYPE_DXT5;
	}
}
void _compress_etc2(Image *p_img, float p_lossy_quality, Image::UsedChannels p_source) {
	EtcpakType type = _determine_etc_type(p_source);
	_compress_etcpak(type, p_img, p_lossy_quality, false, p_source);
}
void _compress_bc(Image *p_img, float p_lossy_quality, Image::UsedChannels p_source) {
	EtcpakType type = _determine_dxt_type(p_source);
	_compress_etcpak(type, p_img, p_lossy_quality, false, p_source);
}
void _compress_etc1(Image *p_img, float p_lossy_quality) {
	_compress_etcpak(EtcpakType::ETCPAK_TYPE_ETC1, p_img, p_lossy_quality, true, Image::USED_CHANNELS_RGB);
}

void _compress_etcpak(EtcpakType p_compresstype, Image *p_img, float p_lossy_quality, bool force_etc1_format, Image::UsedChannels p_channels) {
	uint64_t t = OS::get_singleton()->get_ticks_msec();
	Image::Format img_format = p_img->get_format();

	if (img_format >= Image::FORMAT_DXT1) {
		return; //do not compress, already compressed
	}

	if (img_format > Image::FORMAT_RGBA8) {
		// TODO: we should be able to handle FORMAT_RGBA4444 and FORMAT_RGBA5551 eventually
		return;
	}

	Image::Format format = Image::FORMAT_RGBA8;
	if (p_img->get_format() != Image::FORMAT_RGBA8) {
		p_img->convert(Image::FORMAT_RGBA8);
	}
	if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC1 || force_etc1_format) {
		format = Image::FORMAT_ETC;
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2) {
		format = Image::FORMAT_ETC2_RGB8;
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2_RA_AS_RG) {
		format = Image::FORMAT_ETC2_RA_AS_RG;
		p_img->convert_rg_to_ra_rgba8();
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2_ALPHA) {
		format = Image::FORMAT_ETC2_RGBA8;
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_DXT1) {
		format = Image::FORMAT_DXT1;
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_DXT5_RA_AS_RG) {
		format = Image::FORMAT_DXT5_RA_AS_RG;
		p_img->convert_rg_to_ra_rgba8();
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_DXT5) {
		format = Image::FORMAT_DXT5;
	} else {
		ERR_FAIL();
	}

	const bool mipmap = p_img->has_mipmaps();
	print_verbose("Encoding format: " + Image::get_format_name(format));

	Ref<Image> new_img;
	new_img.instance();
	new_img->create(p_img->get_width(), p_img->get_height(), mipmap, format);
	Vector<uint8_t> data = new_img->get_data();
	uint8_t *wr = data.ptrw();

	Ref<Image> image = p_img->duplicate();
	int mmc = 1 + (mipmap ? Image::get_image_required_mipmaps(new_img->get_width(), new_img->get_height(), format) : 0);
	for (int i = 0; i < mmc; i++) {
		int ofs, size, mip_w, mip_h;
		new_img->get_mipmap_offset_size_and_dimensions(i, ofs, size, mip_w, mip_h);
		mip_w = (mip_w + 3) & ~3;
		mip_h = (mip_h + 3) & ~3;
		Vector<uint8_t> dst_data;
		dst_data.resize(size);
		int mipmap_ofs = image->get_mipmap_offset(i);

		const uint32_t *image_read = (const uint32_t *)&image->get_data().ptr()[mipmap_ofs];
		uint64_t *dst_write = (uint64_t *)dst_data.ptrw();
		if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC1 || force_etc1_format) {
			CompressEtc1RgbDither(image_read, dst_write, mip_w * mip_h / 16, mip_w);
		} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2 || p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2_RA_AS_RG) {
			CompressEtc2Rgb(image_read, dst_write, mip_w * mip_h / 16, mip_w);
		} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2_ALPHA) {
			CompressEtc2Rgba(image_read, dst_write, mip_w * mip_h / 16, mip_w);
		} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_DXT5 || p_compresstype == EtcpakType::ETCPAK_TYPE_DXT5_RA_AS_RG) {
			CompressDxt5(image_read, dst_write, mip_w * mip_h / 16, mip_w);
		} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_DXT1) {
			CompressDxt1Dither(image_read, dst_write, mip_w * mip_h / 16, mip_w);
		} else {
			ERR_FAIL();
		}
		copymem(&wr[ofs], dst_data.ptr(), size);
	}
	p_img->create(new_img->get_width(), new_img->get_height(), mipmap, format, data);

	print_verbose(vformat("ETCPAK encode took %s ms.", rtos(OS::get_singleton()->get_ticks_msec() - t)));
}
