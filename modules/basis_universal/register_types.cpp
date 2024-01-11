/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "core/os/os.h"
#include "servers/rendering_server.h"

#include <transcoder/basisu_transcoder.h>

#ifdef TOOLS_ENABLED
#include <encoder/basisu_comp.h>
#endif

enum BasisDecompressFormat {
	BASIS_DECOMPRESS_RG,
	BASIS_DECOMPRESS_RGB,
	BASIS_DECOMPRESS_RGBA,
	BASIS_DECOMPRESS_RG_AS_RA
};

//workaround for lack of ETC2 RG
#define USE_RG_AS_RGBA

#ifdef TOOLS_ENABLED
static Vector<uint8_t> basis_universal_packer(const Ref<Image> &p_image, Image::UsedChannels p_channels) {
	Vector<uint8_t> budata;
	{
		basisu::basis_compressor_params params;
		Ref<Image> image = p_image->duplicate();
		if (image->get_format() != Image::FORMAT_RGBA8) {
			image->convert(Image::FORMAT_RGBA8);
		}

		params.m_uastc = true;
		params.m_quality_level = basisu::BASISU_QUALITY_MIN;

		params.m_pack_uastc_flags &= ~basisu::cPackUASTCLevelMask;

		static const uint32_t s_level_flags[basisu::TOTAL_PACK_UASTC_LEVELS] = { basisu::cPackUASTCLevelFastest, basisu::cPackUASTCLevelFaster, basisu::cPackUASTCLevelDefault, basisu::cPackUASTCLevelSlower, basisu::cPackUASTCLevelVerySlow };
		params.m_pack_uastc_flags |= s_level_flags[0];
		params.m_rdo_uastc = 0.0f;
		params.m_rdo_uastc_quality_scalar = 0.0f;
		params.m_rdo_uastc_dict_size = 1024;

		params.m_mip_fast = true;
		params.m_multithreading = true;

		basisu::job_pool jpool(OS::get_singleton()->get_processor_count());
		params.m_pJob_pool = &jpool;

		BasisDecompressFormat decompress_format = BASIS_DECOMPRESS_RG;
		params.m_check_for_alpha = false;

		switch (p_channels) {
			case Image::USED_CHANNELS_L: {
				decompress_format = BASIS_DECOMPRESS_RGB;
			} break;
			case Image::USED_CHANNELS_LA: {
				params.m_force_alpha = true;
				decompress_format = BASIS_DECOMPRESS_RGBA;
			} break;
			case Image::USED_CHANNELS_R: {
				decompress_format = BASIS_DECOMPRESS_RGB;
			} break;
			case Image::USED_CHANNELS_RG: {
#ifdef USE_RG_AS_RGBA
				params.m_force_alpha = true;
				image->convert_rg_to_ra_rgba8();
				decompress_format = BASIS_DECOMPRESS_RG_AS_RA;
#else
				params.m_seperate_rg_to_color_alpha = true;
				decompress_format = BASIS_DECOMPRESS_RG;
#endif
			} break;
			case Image::USED_CHANNELS_RGB: {
				decompress_format = BASIS_DECOMPRESS_RGB;
			} break;
			case Image::USED_CHANNELS_RGBA: {
				params.m_force_alpha = true;
				decompress_format = BASIS_DECOMPRESS_RGBA;
			} break;
		}

		if (!image->has_mipmaps()) {
			basisu::image buimg(image->get_width(), image->get_height());
			Vector<uint8_t> vec = image->get_data();
			const uint8_t *r = vec.ptr();
			memcpy(buimg.get_ptr(), r, vec.size());
			params.m_source_images.push_back(buimg);
		} else {
			{
				Ref<Image> base_image = image->get_image_from_mipmap(0);
				Vector<uint8_t> image_vec = base_image->get_data();
				basisu::image buimg_image(base_image->get_width(), base_image->get_height());
				const uint8_t *r = image_vec.ptr();
				memcpy(buimg_image.get_ptr(), r, image_vec.size());
				params.m_source_images.push_back(buimg_image);
			}
			basisu::vector<basisu::image> images;
			for (int32_t mip_map_i = 1; mip_map_i <= image->get_mipmap_count(); mip_map_i++) {
				Ref<Image> mip_map = image->get_image_from_mipmap(mip_map_i);
				Vector<uint8_t> mip_map_vec = mip_map->get_data();
				basisu::image buimg_mipmap(mip_map->get_width(), mip_map->get_height());
				const uint8_t *r = mip_map_vec.ptr();
				memcpy(buimg_mipmap.get_ptr(), r, mip_map_vec.size());
				images.push_back(buimg_mipmap);
			}
			params.m_source_mipmap_images.push_back(images);
		}

		basisu::basis_compressor c;
		c.init(params);

		int buerr = c.process();
		ERR_FAIL_COND_V(buerr != basisu::basis_compressor::cECSuccess, budata);

		const basisu::uint8_vec &buvec = c.get_output_basis_file();
		budata.resize(buvec.size() + 4);

		{
			uint8_t *w = budata.ptrw();
			uint32_t *decf = (uint32_t *)w;
			*decf = decompress_format;
			memcpy(w + 4, &buvec[0], buvec.size());
		}
	}

	return budata;
}
#endif // TOOLS_ENABLED

static Ref<Image> basis_universal_unpacker_ptr(const uint8_t *p_data, int p_size) {
	Ref<Image> image;

	const uint8_t *ptr = p_data;
	int size = p_size;
	ERR_FAIL_NULL_V_MSG(p_data, image, "Cannot unpack invalid basis universal data.");

	basist::transcoder_texture_format format = basist::transcoder_texture_format::cTFTotalTextureFormats;
	Image::Format imgfmt = Image::FORMAT_MAX;

	switch (*(uint32_t *)(ptr)) {
		case BASIS_DECOMPRESS_RG: {
			if (RS::get_singleton()->has_os_feature("rgtc")) {
				format = basist::transcoder_texture_format::cTFBC5; // get this from renderer
				imgfmt = Image::FORMAT_RGTC_RG;
			} else if (RS::get_singleton()->has_os_feature("etc2")) {
				//unfortunately, basis universal does not support
				//
				ERR_FAIL_V(image); //unimplemented here
				//format = basist::transcoder_texture_format::cTFETC1; // get this from renderer
				//imgfmt = Image::FORMAT_RGTC_RG;
			} else {
				// FIXME: There wasn't anything here, but then imgformat is used uninitialized.
				ERR_FAIL_V(image);
			}
		} break;
		case BASIS_DECOMPRESS_RGB: {
			if (RS::get_singleton()->has_os_feature("bptc")) {
				format = basist::transcoder_texture_format::cTFBC7_M6_OPAQUE_ONLY; // get this from renderer
				imgfmt = Image::FORMAT_BPTC_RGBA;
			} else if (RS::get_singleton()->has_os_feature("s3tc")) {
				format = basist::transcoder_texture_format::cTFBC1; // get this from renderer
				imgfmt = Image::FORMAT_DXT1;
			} else if (RS::get_singleton()->has_os_feature("etc")) {
				format = basist::transcoder_texture_format::cTFETC1; // get this from renderer
				imgfmt = Image::FORMAT_ETC;
			} else {
				format = basist::transcoder_texture_format::cTFBGR565; // get this from renderer
				imgfmt = Image::FORMAT_RGB565;
			}

		} break;
		case BASIS_DECOMPRESS_RGBA: {
			if (RS::get_singleton()->has_os_feature("bptc")) {
				format = basist::transcoder_texture_format::cTFBC7_M5; // get this from renderer
				imgfmt = Image::FORMAT_BPTC_RGBA;
			} else if (RS::get_singleton()->has_os_feature("s3tc")) {
				format = basist::transcoder_texture_format::cTFBC3; // get this from renderer
				imgfmt = Image::FORMAT_DXT5;
			} else if (RS::get_singleton()->has_os_feature("etc2")) {
				format = basist::transcoder_texture_format::cTFETC2; // get this from renderer
				imgfmt = Image::FORMAT_ETC2_RGBA8;
			} else {
				//opengl most likely
				format = basist::transcoder_texture_format::cTFRGBA4444; // get this from renderer
				imgfmt = Image::FORMAT_RGBA4444;
			}
		} break;
		case BASIS_DECOMPRESS_RG_AS_RA: {
			if (RS::get_singleton()->has_os_feature("s3tc")) {
				format = basist::transcoder_texture_format::cTFBC3; // get this from renderer
				imgfmt = Image::FORMAT_DXT5_RA_AS_RG;
			} else if (RS::get_singleton()->has_os_feature("etc2")) {
				format = basist::transcoder_texture_format::cTFETC2; // get this from renderer
				imgfmt = Image::FORMAT_ETC2_RA_AS_RG;
			} else {
				//opengl most likely, bad for normal maps, nothing to do about this.
				format = basist::transcoder_texture_format::cTFRGBA32;
				imgfmt = Image::FORMAT_RGBA8;
			}
		} break;
	}

	ptr += 4;
	size -= 4;

	basist::basisu_transcoder tr;

	ERR_FAIL_COND_V(!tr.validate_header(ptr, size), image);

	tr.start_transcoding(ptr, size);

	basist::basisu_image_info info;
	tr.get_image_info(ptr, size, info, 0);
	Vector<uint8_t> gpudata;
	gpudata.resize(Image::get_image_data_size(info.m_width, info.m_height, imgfmt, info.m_total_levels > 1));

	uint8_t *w = gpudata.ptrw();
	uint8_t *dst = w;
	for (int i = 0; i < gpudata.size(); i++) {
		dst[i] = 0x00;
	}
	uint32_t mip_count = Image::get_image_required_mipmaps(info.m_orig_width, info.m_orig_height, imgfmt);
	for (uint32_t level_i = 0; level_i <= mip_count; level_i++) {
		basist::basisu_image_level_info level;
		tr.get_image_level_info(ptr, size, level, 0, level_i);
		int ofs = Image::get_image_mipmap_offset(info.m_width, info.m_height, imgfmt, level_i);
		bool ret = tr.transcode_image_level(ptr, size, 0, level_i, dst + ofs, level.m_total_blocks, format);
		if (!ret) {
			print_line(vformat("Basis universal cannot unpack level %d.", level_i));
			break;
		};
	}

	image = Image::create_from_data(info.m_width, info.m_height, info.m_total_levels > 1, imgfmt, gpudata);

	return image;
}

static Ref<Image> basis_universal_unpacker(const Vector<uint8_t> &p_buffer) {
	Ref<Image> image;

	const uint8_t *r = p_buffer.ptr();
	int size = p_buffer.size();
	return basis_universal_unpacker_ptr(r, size);
}

void initialize_basis_universal_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

#ifdef TOOLS_ENABLED
	using namespace basisu;
	using namespace basist;
	basisu_encoder_init();
	Image::basis_universal_packer = basis_universal_packer;
#endif
	basist::basisu_transcoder_init();
	Image::basis_universal_unpacker = basis_universal_unpacker;
	Image::basis_universal_unpacker_ptr = basis_universal_unpacker_ptr;
}

void uninitialize_basis_universal_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

#ifdef TOOLS_ENABLED
	Image::basis_universal_packer = nullptr;
#endif
	Image::basis_universal_unpacker = nullptr;
	Image::basis_universal_unpacker_ptr = nullptr;
}
