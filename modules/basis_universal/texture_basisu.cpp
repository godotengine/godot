/*************************************************************************/
/*  texture_basisu.cpp                                                   */
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

#include "texture_basisu.h"
#if 0
#include "core/os/os.h"

#ifdef TOOLS_ENABLED
#include <encoder/basisu_comp.h>
#endif

#include <transcoder/basisu_transcoder.h>

void TextureBasisU::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_basisu_data", "data"), &TextureBasisU::set_basisu_data);
	ClassDB::bind_method(D_METHOD("get_basisu_data"), &TextureBasisU::get_data);
	ClassDB::bind_method(D_METHOD("import"), &TextureBasisU::import);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "basisu_data"), "set_basisu_data", "get_basisu_data");
};

int TextureBasisU::get_width() const {
	return tex_size.x;
};

int TextureBasisU::get_height() const {
	return tex_size.y;
};

RID TextureBasisU::get_rid() const {
	return texture;
};


bool TextureBasisU::has_alpha() const {
	return false;
};

void TextureBasisU::set_flags(uint32_t p_flags) {
	flags = p_flags;
	RenderingServer::get_singleton()->texture_set_flags(texture, p_flags);
};

uint32_t TextureBasisU::get_flags() const {
	return flags;
};


void TextureBasisU::set_basisu_data(const Vector<uint8_t>& p_data) {

#ifdef TOOLS_ENABLED
	data = p_data;
#endif

	const uint8_t* r = p_data.ptr();
	const void* ptr = r.ptr();
	int size = p_data.size();

	basist::transcoder_texture_format format;
	Image::Format imgfmt;

	if (OS::get_singleton()->has_feature("s3tc")) {
		format = basist::cTFBC3; // get this from renderer
		imgfmt = Image::FORMAT_DXT5;

	} else if (OS::get_singleton()->has_feature("etc2")) {
		format = basist::cTFETC2;
		imgfmt = Image::FORMAT_ETC2_RGBA8;
	};

	basist::basisu_transcoder tr(nullptr);

	ERR_FAIL_COND(!tr.validate_header(ptr, size));

	basist::basisu_image_info info;
	tr.get_image_info(ptr, size, info, 0);
	tex_size = Size2(info.m_width, info.m_height);

	int block_size = basist::basis_get_bytes_per_block(format);
	Vector<uint8_t> gpudata;
	gpudata.resize(info.m_total_blocks * block_size);

	{
		uint8_t* w = gpudata.ptrw();
		uint8_t* dst = w.ptr();
		for (int i=0; i<gpudata.size(); i++)
			dst[i] = 0x00;

		int ofs = 0;
		tr.start_transcoding(ptr, size);
		for (int i=0; i<info.m_total_levels; i++) {
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

	Ref<Image> img;
	img.instantiate();
	img->create(info.m_width, info.m_height, info.m_total_levels > 1, imgfmt, gpudata);

	RenderingServer::get_singleton()->texture_allocate(texture, tex_size.x, tex_size.y, 0, img->get_format(), RS::TEXTURE_TYPE_2D, flags);
	RenderingServer::get_singleton()->texture_set_data(texture, img);
};

Error TextureBasisU::import(const Ref<Image>& p_img) {

#ifdef TOOLS_ENABLED

	Vector<uint8_t> budata;

	{
		Image::Format format = p_img->get_format();
		if (format != Image::FORMAT_RGB8 && format != Image::FORMAT_RGBA8) {
			ERR_FAIL_V(ERR_INVALID_PARAMETER);
			return ERR_INVALID_PARAMETER;
		};

		Ref<Image> copy = p_img->duplicate();
		if (format == Image::FORMAT_RGB8)
			copy->convert(Image::FORMAT_RGBA8);

		basisu::image buimg(p_img->get_width(), p_img->get_height());
		int size = p_img->get_width() * p_img->get_height() * 4;

		Vector<uint8_t> vec = copy->get_data();
		{
			const uint8_t* r = vec.ptr();
			memcpy(buimg.get_ptr(), r.ptr(), size);
		};

		basisu::basis_compressor_params params;
		params.m_max_endpoint_clusters = 512;
		params.m_max_selector_clusters = 512;
		params.m_multithreading = true;

		basisu::job_pool jpool(1);
		params.m_pJob_pool = &jpool;

		params.m_mip_gen = p_img->get_mipmap_count() > 0;
		params.m_source_images.push_back(buimg);

		basisu::basis_compressor c;
		c.init(params);

		int buerr = c.process();
		if (buerr != basisu::basis_compressor::cECSuccess) {
			ERR_FAIL_V(ERR_INVALID_PARAMETER);
			return ERR_INVALID_PARAMETER;
		};

		const basisu::uint8_vec& buvec = c.get_output_basis_file();
		budata.resize(buvec.size());

		{
			uint8_t* w = budata.ptrw();
			memcpy(w.ptr(), &buvec[0], budata.size());
		};
	};

	set_basisu_data(budata);

	return OK;
#else

	return ERR_UNAVAILABLE;
#endif
};


Vector<uint8_t> TextureBasisU::get_basisu_data() const {
	return data;
};

TextureBasisU::TextureBasisU() {
	texture = RenderingServer::get_singleton()->texture_create();
};


TextureBasisU::~TextureBasisU() {
	RenderingServer::get_singleton()->free(texture);
};

#endif
