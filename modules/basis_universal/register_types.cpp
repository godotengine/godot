/*************************************************************************/
/*  register_types.cpp                                                   */
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

#include "register_types.h"

#include "core/os/os.h"
#include "servers/rendering_server.h"
#include "texture_basisu.h"

#ifdef TOOLS_ENABLED
#include <encoder/basisu_comp.h>
#endif

#include <transcoder/basisu_transcoder.h>

enum BasisDecompressFormat {
	BASIS_DECOMPRESS_RG,
	BASIS_DECOMPRESS_RGB,
	BASIS_DECOMPRESS_RGBA,
	BASIS_DECOMPRESS_RG_AS_RA
};

//workaround for lack of ETC2 RG
#define USE_RG_AS_RGBA

basist::etc1_global_selector_codebook *sel_codebook = nullptr;

#ifdef TOOLS_ENABLED
static Vector<uint8_t> basis_universal_packer(const Ref<Image> &p_image, Image::UsedChannels p_channels) {
	Vector<uint8_t> budata;

	{
		Ref<Image> image = p_image->duplicate();

		// unfortunately, basis universal does not support compressing supplied mipmaps,
		// so for the time being, only compressing individual images will have to do.

		if (image->has_mipmaps()) {
			image->clear_mipmaps();
		}
		if (image->get_format() != Image::FORMAT_RGBA8) {
			image->convert(Image::FORMAT_RGBA8);
		}

		basisu::image buimg(image->get_width(), image->get_height());

		{
			Vector<uint8_t> vec = image->get_data();
			const uint8_t *r = vec.ptr();

			memcpy(buimg.get_ptr(), r, vec.size());
		}

		//image->save_png("pepeche.png");

		basisu::basis_compressor_params params;
		params.m_max_endpoint_clusters = 512;
		params.m_max_selector_clusters = 512;
		params.m_multithreading = true;
		//params.m_no_hybrid_sel_cb = true; //fixme, default on this causes crashes //seems fixed?
		params.m_pSel_codebook = sel_codebook;
		//params.m_quality_level = 0;
		//params.m_disable_hierarchical_endpoint_codebooks = true;
		//params.m_no_selector_rdo = true;
		params.m_auto_global_sel_pal = false;

		basisu::job_pool jpool(OS::get_singleton()->get_processor_count());
		params.m_pJob_pool = &jpool;

		params.m_mip_gen = false; //sorry, please some day support provided mipmaps.
		params.m_source_images.push_back(buimg);

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

static Ref<Image> basis_universal_unpacker(const Vector<uint8_t> &p_buffer) {
	Ref<Image> image;

	const uint8_t *r = p_buffer.ptr();
	const uint8_t *ptr = r;
	int size = p_buffer.size();

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
				imgfmt = Image::FORMAT_ETC2_RGBA8;
			} else {
				//opengl most likely, bad for normal maps, nothing to do about this.
				format = basist::transcoder_texture_format::cTFRGBA32;
				imgfmt = Image::FORMAT_RGBA8;
			}
		} break;
	}

	ptr += 4;
	size -= 4;

	basist::basisu_transcoder tr(nullptr);

	ERR_FAIL_COND_V(!tr.validate_header(ptr, size), image);

	basist::basisu_image_info info;
	tr.get_image_info(ptr, size, info, 0);

	int block_size = basist::basis_get_bytes_per_block_or_pixel(format);
	Vector<uint8_t> gpudata;
	gpudata.resize(info.m_total_blocks * block_size);

	{
		uint8_t *w = gpudata.ptrw();
		uint8_t *dst = w;
		for (int i = 0; i < gpudata.size(); i++) {
			dst[i] = 0x00;
		}

		int ofs = 0;
		tr.start_transcoding(ptr, size);
		for (uint32_t i = 0; i < info.m_total_levels; i++) {
			basist::basisu_image_level_info level;
			tr.get_image_level_info(ptr, size, level, 0, i);

			bool ret = tr.transcode_image_level(ptr, size, 0, i, dst + ofs, level.m_total_blocks - i, format);
			if (!ret) {
				printf("failed! on level %i\n", i);
				break;
			};

			ofs += level.m_total_blocks * block_size;
		};
	};

	image.instantiate();
	image->create(info.m_width, info.m_height, info.m_total_levels > 1, imgfmt, gpudata);

	return image;
}

void register_basis_universal_types() {
#ifdef TOOLS_ENABLED
	sel_codebook = new basist::etc1_global_selector_codebook(basist::g_global_selector_cb_size, basist::g_global_selector_cb);
	Image::basis_universal_packer = basis_universal_packer;
#endif
	Image::basis_universal_unpacker = basis_universal_unpacker;
	//GDREGISTER_CLASS(TextureBasisU);
}

void unregister_basis_universal_types() {
#ifdef TOOLS_ENABLED
	delete sel_codebook;
	Image::basis_universal_packer = nullptr;
#endif
	Image::basis_universal_unpacker = nullptr;
}
