/**************************************************************************/
/*  image_compress_betsy.cpp                                              */
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

#include "image_compress_betsy.h"

#include "bc1.glsl.gen.h"
#include "bc4.glsl.gen.h"
#include "bc6h.glsl.gen.h"
#include "format_converter.glsl.gen.h"

#include "core/os/os.h"
#include "core/string/print_string.h"
#include "servers/rendering/rendering_device_binds.h"

static const float dxt1_encoding_table[1024] = {
	0,
	0,
	0,
	0,
	0,
	1,
	0,
	1,
	1,
	0,
	1,
	0,
	1,
	0,
	1,
	1,
	1,
	1,
	2,
	0,
	2,
	0,
	0,
	4,
	2,
	1,
	2,
	1,
	2,
	1,
	3,
	0,
	3,
	0,
	3,
	0,
	3,
	1,
	1,
	5,
	3,
	2,
	3,
	2,
	4,
	0,
	4,
	0,
	4,
	1,
	4,
	1,
	4,
	2,
	4,
	2,
	4,
	2,
	3,
	5,
	5,
	1,
	5,
	1,
	5,
	2,
	4,
	4,
	5,
	3,
	5,
	3,
	5,
	3,
	6,
	2,
	6,
	2,
	6,
	2,
	6,
	3,
	5,
	5,
	6,
	4,
	6,
	4,
	4,
	8,
	7,
	3,
	7,
	3,
	7,
	3,
	7,
	4,
	7,
	4,
	7,
	4,
	7,
	5,
	5,
	9,
	7,
	6,
	7,
	6,
	8,
	4,
	8,
	4,
	8,
	5,
	8,
	5,
	8,
	6,
	8,
	6,
	8,
	6,
	7,
	9,
	9,
	5,
	9,
	5,
	9,
	6,
	8,
	8,
	9,
	7,
	9,
	7,
	9,
	7,
	10,
	6,
	10,
	6,
	10,
	6,
	10,
	7,
	9,
	9,
	10,
	8,
	10,
	8,
	8,
	12,
	11,
	7,
	11,
	7,
	11,
	7,
	11,
	8,
	11,
	8,
	11,
	8,
	11,
	9,
	9,
	13,
	11,
	10,
	11,
	10,
	12,
	8,
	12,
	8,
	12,
	9,
	12,
	9,
	12,
	10,
	12,
	10,
	12,
	10,
	11,
	13,
	13,
	9,
	13,
	9,
	13,
	10,
	12,
	12,
	13,
	11,
	13,
	11,
	13,
	11,
	14,
	10,
	14,
	10,
	14,
	10,
	14,
	11,
	13,
	13,
	14,
	12,
	14,
	12,
	12,
	16,
	15,
	11,
	15,
	11,
	15,
	11,
	15,
	12,
	15,
	12,
	15,
	12,
	15,
	13,
	13,
	17,
	15,
	14,
	15,
	14,
	16,
	12,
	16,
	12,
	16,
	13,
	16,
	13,
	16,
	14,
	16,
	14,
	16,
	14,
	15,
	17,
	17,
	13,
	17,
	13,
	17,
	14,
	16,
	16,
	17,
	15,
	17,
	15,
	17,
	15,
	18,
	14,
	18,
	14,
	18,
	14,
	18,
	15,
	17,
	17,
	18,
	16,
	18,
	16,
	16,
	20,
	19,
	15,
	19,
	15,
	19,
	15,
	19,
	16,
	19,
	16,
	19,
	16,
	19,
	17,
	17,
	21,
	19,
	18,
	19,
	18,
	20,
	16,
	20,
	16,
	20,
	17,
	20,
	17,
	20,
	18,
	20,
	18,
	20,
	18,
	19,
	21,
	21,
	17,
	21,
	17,
	21,
	18,
	20,
	20,
	21,
	19,
	21,
	19,
	21,
	19,
	22,
	18,
	22,
	18,
	22,
	18,
	22,
	19,
	21,
	21,
	22,
	20,
	22,
	20,
	20,
	24,
	23,
	19,
	23,
	19,
	23,
	19,
	23,
	20,
	23,
	20,
	23,
	20,
	23,
	21,
	21,
	25,
	23,
	22,
	23,
	22,
	24,
	20,
	24,
	20,
	24,
	21,
	24,
	21,
	24,
	22,
	24,
	22,
	24,
	22,
	23,
	25,
	25,
	21,
	25,
	21,
	25,
	22,
	24,
	24,
	25,
	23,
	25,
	23,
	25,
	23,
	26,
	22,
	26,
	22,
	26,
	22,
	26,
	23,
	25,
	25,
	26,
	24,
	26,
	24,
	24,
	28,
	27,
	23,
	27,
	23,
	27,
	23,
	27,
	24,
	27,
	24,
	27,
	24,
	27,
	25,
	25,
	29,
	27,
	26,
	27,
	26,
	28,
	24,
	28,
	24,
	28,
	25,
	28,
	25,
	28,
	26,
	28,
	26,
	28,
	26,
	27,
	29,
	29,
	25,
	29,
	25,
	29,
	26,
	28,
	28,
	29,
	27,
	29,
	27,
	29,
	27,
	30,
	26,
	30,
	26,
	30,
	26,
	30,
	27,
	29,
	29,
	30,
	28,
	30,
	28,
	30,
	28,
	31,
	27,
	31,
	27,
	31,
	27,
	31,
	28,
	31,
	28,
	31,
	28,
	31,
	29,
	31,
	29,
	31,
	30,
	31,
	30,
	31,
	30,
	31,
	31,
	31,
	31,
	0,
	0,
	0,
	1,
	1,
	0,
	1,
	0,
	1,
	1,
	2,
	0,
	2,
	1,
	3,
	0,
	3,
	0,
	3,
	1,
	4,
	0,
	4,
	0,
	4,
	1,
	5,
	0,
	5,
	1,
	6,
	0,
	6,
	0,
	6,
	1,
	7,
	0,
	7,
	0,
	7,
	1,
	8,
	0,
	8,
	1,
	8,
	1,
	8,
	2,
	9,
	1,
	9,
	2,
	9,
	2,
	9,
	3,
	10,
	2,
	10,
	3,
	10,
	3,
	10,
	4,
	11,
	3,
	11,
	4,
	11,
	4,
	11,
	5,
	12,
	4,
	12,
	5,
	12,
	5,
	12,
	6,
	13,
	5,
	13,
	6,
	8,
	16,
	13,
	7,
	14,
	6,
	14,
	7,
	9,
	17,
	14,
	8,
	15,
	7,
	15,
	8,
	11,
	16,
	15,
	9,
	15,
	10,
	16,
	8,
	16,
	9,
	16,
	10,
	15,
	13,
	17,
	9,
	17,
	10,
	17,
	11,
	15,
	16,
	18,
	10,
	18,
	11,
	18,
	12,
	16,
	16,
	19,
	11,
	19,
	12,
	19,
	13,
	17,
	17,
	20,
	12,
	20,
	13,
	20,
	14,
	19,
	16,
	21,
	13,
	21,
	14,
	21,
	15,
	20,
	17,
	22,
	14,
	22,
	15,
	25,
	10,
	22,
	16,
	23,
	15,
	23,
	16,
	26,
	11,
	23,
	17,
	24,
	16,
	24,
	17,
	27,
	12,
	24,
	18,
	25,
	17,
	25,
	18,
	28,
	13,
	25,
	19,
	26,
	18,
	26,
	19,
	29,
	14,
	26,
	20,
	27,
	19,
	27,
	20,
	30,
	15,
	27,
	21,
	28,
	20,
	28,
	21,
	28,
	21,
	28,
	22,
	29,
	21,
	29,
	22,
	24,
	32,
	29,
	23,
	30,
	22,
	30,
	23,
	25,
	33,
	30,
	24,
	31,
	23,
	31,
	24,
	27,
	32,
	31,
	25,
	31,
	26,
	32,
	24,
	32,
	25,
	32,
	26,
	31,
	29,
	33,
	25,
	33,
	26,
	33,
	27,
	31,
	32,
	34,
	26,
	34,
	27,
	34,
	28,
	32,
	32,
	35,
	27,
	35,
	28,
	35,
	29,
	33,
	33,
	36,
	28,
	36,
	29,
	36,
	30,
	35,
	32,
	37,
	29,
	37,
	30,
	37,
	31,
	36,
	33,
	38,
	30,
	38,
	31,
	41,
	26,
	38,
	32,
	39,
	31,
	39,
	32,
	42,
	27,
	39,
	33,
	40,
	32,
	40,
	33,
	43,
	28,
	40,
	34,
	41,
	33,
	41,
	34,
	44,
	29,
	41,
	35,
	42,
	34,
	42,
	35,
	45,
	30,
	42,
	36,
	43,
	35,
	43,
	36,
	46,
	31,
	43,
	37,
	44,
	36,
	44,
	37,
	44,
	37,
	44,
	38,
	45,
	37,
	45,
	38,
	40,
	48,
	45,
	39,
	46,
	38,
	46,
	39,
	41,
	49,
	46,
	40,
	47,
	39,
	47,
	40,
	43,
	48,
	47,
	41,
	47,
	42,
	48,
	40,
	48,
	41,
	48,
	42,
	47,
	45,
	49,
	41,
	49,
	42,
	49,
	43,
	47,
	48,
	50,
	42,
	50,
	43,
	50,
	44,
	48,
	48,
	51,
	43,
	51,
	44,
	51,
	45,
	49,
	49,
	52,
	44,
	52,
	45,
	52,
	46,
	51,
	48,
	53,
	45,
	53,
	46,
	53,
	47,
	52,
	49,
	54,
	46,
	54,
	47,
	57,
	42,
	54,
	48,
	55,
	47,
	55,
	48,
	58,
	43,
	55,
	49,
	56,
	48,
	56,
	49,
	59,
	44,
	56,
	50,
	57,
	49,
	57,
	50,
	60,
	45,
	57,
	51,
	58,
	50,
	58,
	51,
	61,
	46,
	58,
	52,
	59,
	51,
	59,
	52,
	62,
	47,
	59,
	53,
	60,
	52,
	60,
	53,
	60,
	53,
	60,
	54,
	61,
	53,
	61,
	54,
	61,
	54,
	61,
	55,
	62,
	54,
	62,
	55,
	62,
	55,
	62,
	56,
	63,
	55,
	63,
	56,
	63,
	56,
	63,
	57,
	63,
	58,
	63,
	59,
	63,
	59,
	63,
	60,
	63,
	61,
	63,
	62,
	63,
	62,
	63,
	63,
};

static int get_next_multiple(int n, int m) {
	return n + (m - (n % m));
}

void _compress_betsy(BetsyFormat p_format, Image *r_img) {
	ERR_FAIL_COND_MSG(r_img->is_compressed(), "Image is already compressed.");

	Ref<Image> img_clone = memnew(Image);
	img_clone->copy_from(r_img);

	// Create local RD.
	RenderingDevice *rd = RenderingDevice::get_singleton()->create_local_device();

	Ref<RDShaderFile> compute_shader;
	compute_shader.instantiate();

	// Destination format.
	Image::Format dest_format = Image::FORMAT_MAX;

	Error err = OK;
	switch (p_format) {
		case BETSY_FORMAT_BC6UF:
			err = compute_shader->parse_versions_from_text(bc6h_shader_glsl);
			dest_format = Image::FORMAT_BPTC_RGBFU;
			break;

		case BETSY_FORMAT_BC6SF:
			err = compute_shader->parse_versions_from_text(bc6h_shader_glsl, "#define SIGNED");
			dest_format = Image::FORMAT_BPTC_RGBF;
			break;

		case BETSY_FORMAT_BC1:
			err = compute_shader->parse_versions_from_text(bc1_shader_glsl);
			dest_format = Image::FORMAT_DXT1;
			break;

		case BETSY_FORMAT_BC4U:
			err = compute_shader->parse_versions_from_text(bc4_shader_glsl);
			dest_format = Image::FORMAT_RGTC_R;
			break;

		case BETSY_FORMAT_BC5U:
			err = compute_shader->parse_versions_from_text(bc4_shader_glsl);
			dest_format = Image::FORMAT_RGTC_RG;
			break;

		default:
			return;
	}

	ERR_FAIL_COND(err != OK);

	// Compile the shader, return early if invalid.
	RID shader = rd->shader_create_from_spirv(compute_shader->get_spirv_stages());
	ERR_FAIL_COND(shader.is_null());

	RID pipeline = rd->compute_pipeline_create(shader);

	//RID dst_texture;
	//RID src_texture;
	RID src_sampler;
	RID encoding_table_buffer; // Encoding table only for BC1/ETC2

	bool uses_encoding_table = false;

	// src_texture format information.
	RD::TextureFormat src_texture_format;
	{
		src_texture_format.array_layers = 1;
		src_texture_format.depth = 1;
		src_texture_format.mipmaps = 1;
		src_texture_format.texture_type = RD::TEXTURE_TYPE_2D;
		src_texture_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	}

	if (img_clone->get_format() >= Image::FORMAT_L8 && img_clone->get_format() <= Image::FORMAT_RGB565) {
		// RGBA8.
		src_texture_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		img_clone->convert(Image::FORMAT_RGBA8);
	} else if (img_clone->get_format() >= Image::FORMAT_RF && img_clone->get_format() <= Image::FORMAT_RGBAF) {
		// RGBAF.
		src_texture_format.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
		img_clone->convert(Image::FORMAT_RGBAF);
	} else if (img_clone->get_format() >= Image::FORMAT_RH && img_clone->get_format() <= Image::FORMAT_RGBE9995) {
		// RGBAH.
		src_texture_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		img_clone->convert(Image::FORMAT_RGBAH);
	}

	/*
	// Convert the image to a format which can be sampled.
	if (img_clone->get_format() != Image::FORMAT_RGBA8 || img_clone->get_format() != Image::FORMAT_RGBAH || img_clone->get_format() != Image::FORMAT_RGBAF) {
		Ref<RDShaderFile> convert_shader;
		convert_shader.instantiate();

		err = convert_shader->parse_versions_from_text(format_converter_glsl);
		ERR_FAIL_COND(err != OK);

		// Compile the shader, return early if invalid.
		RID convert_shader_rid = rd->shader_create_from_spirv(convert_shader->get_spirv_stages());
		ERR_FAIL_COND(convert_shader_rid.is_null());

		RID convert_pipeline = rd->compute_pipeline_create(convert_shader_rid);

		Vector<RD::Uniform> uniforms;
		{
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 0;
				u.append_id(src_sampler);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 1;
				u.append_id(dst_textures_ptr[i]);
				uniforms.push_back(u);
			}
		}

		RID uniform_set = rd->uniform_set_create(uniforms, convert_shader_rid, 0);
		RD::ComputeListID compute_list = rd->compute_list_begin();

		rd->compute_list_bind_compute_pipeline(compute_list, convert_pipeline);
		rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
		rd->compute_list_dispatch(compute_list, get_next_multiple(img_clone->get_width(), 32) / 32, get_next_multiple(img_clone->get_height(), 32) / 32, 1);
		rd->compute_list_end();
	}*/

	// For the destination format just copy the source format and change the usage bits.
	RD::TextureFormat dst_texture_format = src_texture_format;
	dst_texture_format.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;

	RD::SamplerState src_sampler_state;
	{
		src_sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
		src_sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
		src_sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
		src_sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
		src_sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
	}

	src_sampler = rd->sampler_create(src_sampler_state);

	// Set the destination texture format to match the compressed block size.
	switch (dest_format) {
		case Image::FORMAT_DXT1:
		case Image::FORMAT_RGTC_R:
		case Image::FORMAT_ETC2_R11:
		case Image::FORMAT_ETC2_R11S:
		case Image::FORMAT_ETC2_RGB8:
		case Image::FORMAT_ETC2_RGB8A1:
			dst_texture_format.format = RD::DATA_FORMAT_R32G32_UINT;
			break;

		case Image::FORMAT_DXT5:
		case Image::FORMAT_RGTC_RG:
		case Image::FORMAT_BPTC_RGBA:
		case Image::FORMAT_BPTC_RGBF:
		case Image::FORMAT_BPTC_RGBFU:
		case Image::FORMAT_ETC2_RG11:
		case Image::FORMAT_ETC2_RG11S:
		case Image::FORMAT_ETC2_RGBA8:
			dst_texture_format.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
			break;

		default:
			break;
	}

	const int mip_count = img_clone->get_mipmap_count() + 1;

	// Encoding table setup.
	if (dest_format == Image::FORMAT_DXT1) {
		Vector<uint8_t> data;
		data.resize(1024 * 4);
		memcpy(data.ptrw(), dxt1_encoding_table, 1024 * 4);

		encoding_table_buffer = rd->storage_buffer_create(1024 * 4, data);
		uses_encoding_table = true;
	}

	Vector<RID> src_textures;
	src_textures.resize(mip_count);
	RID *src_textures_ptr = src_textures.ptrw();

	Vector<RID> dst_textures;
	dst_textures.resize(mip_count);
	RID *dst_textures_ptr = dst_textures.ptrw();

	for (int i = 0; i < mip_count; i++) {
		int ofs, size, width, height;
		img_clone->get_mipmap_offset_size_and_dimensions(i, ofs, size, width, height);

		// Set the source texture width and size.
		src_texture_format.height = height;
		src_texture_format.width = width;

		// Set the destination texture width and size.
		dst_texture_format.height = (height + 3) >> 2;
		dst_texture_format.width = (width + 3) >> 2;

		Vector<Vector<uint8_t>> src_images;
		src_images.push_back(Vector<uint8_t>());
		src_images.ptrw()[0].resize(size);
		memcpy(src_images.ptrw()[0].ptrw(), img_clone->get_data().ptr() + ofs, size);

		src_textures_ptr[i] = rd->texture_create(src_texture_format, RD::TextureView(), src_images);
		dst_textures_ptr[i] = rd->texture_create(dst_texture_format, RD::TextureView());
		rd->texture_clear(dst_textures_ptr[i], Color(0, 0, 0, 0), 0, 1, 0, 1);

		if (dest_format == Image::FORMAT_DXT1) {
			Vector<uint32_t> push_constant;
			push_constant.push_back(2);
			push_constant.push_back(0);
			push_constant.push_back(0);
			push_constant.push_back(0);

			Vector<RD::Uniform> uniforms;
			{
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
					u.binding = 0;
					u.append_id(src_sampler);
					u.append_id(src_textures_ptr[i]);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 1;
					u.append_id(dst_textures_ptr[i]);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 2;
					u.append_id(encoding_table_buffer);
					uniforms.push_back(u);
				}
			}

			RID uniform_set = rd->uniform_set_create(uniforms, shader, 0);
			RD::ComputeListID compute_list = rd->compute_list_begin();

			rd->compute_list_bind_compute_pipeline(compute_list, pipeline);
			rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
			rd->compute_list_set_push_constant(compute_list, push_constant.ptr(), push_constant.size() * 4);
			rd->compute_list_dispatch(compute_list, get_next_multiple(width, 32) / 32, get_next_multiple(height, 32) / 32, 1);
			rd->compute_list_end();

		} else if (dest_format == Image::FORMAT_RGTC_R) {
			Vector<uint32_t> push_constant;
			push_constant.push_back(0);
			push_constant.push_back(0);
			push_constant.push_back(0);
			push_constant.push_back(0);

			Vector<RD::Uniform> uniforms;
			{
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
					u.binding = 0;
					u.append_id(src_sampler);
					u.append_id(src_textures_ptr[i]);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 1;
					u.append_id(dst_textures_ptr[i]);
					uniforms.push_back(u);
				}
			}

			RID uniform_set = rd->uniform_set_create(uniforms, shader, 0);
			RD::ComputeListID compute_list = rd->compute_list_begin();

			rd->compute_list_bind_compute_pipeline(compute_list, pipeline);
			rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
			rd->compute_list_set_push_constant(compute_list, push_constant.ptr(), push_constant.size() * 4);
			rd->compute_list_dispatch(compute_list, 1, get_next_multiple(width, 16) / 16, get_next_multiple(height, 16) / 16);
			rd->compute_list_end();

		} else if (dest_format == Image::FORMAT_RGTC_RG) {
			Vector<RD::Uniform> uniforms;
			{
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
					u.binding = 0;
					u.append_id(src_sampler);
					u.append_id(src_textures_ptr[i]);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 1;
					u.append_id(dst_textures_ptr[i]);
					uniforms.push_back(u);
				}
			}

			RID uniform_set = rd->uniform_set_create(uniforms, shader, 0);
			RD::ComputeListID compute_list = rd->compute_list_begin();

			for (size_t j = 0; j < 2; j++) {
				Vector<uint32_t> push_constant;
				push_constant.push_back(j);
				push_constant.push_back(0);
				push_constant.push_back(0);
				push_constant.push_back(0);

				rd->compute_list_bind_compute_pipeline(compute_list, pipeline);
				rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
				rd->compute_list_set_push_constant(compute_list, push_constant.ptr(), push_constant.size() * 4);
				rd->compute_list_dispatch(compute_list, 1, get_next_multiple(width, 16) / 16, get_next_multiple(height, 16) / 16);
			}

			rd->compute_list_end();

		} else if (dest_format == Image::FORMAT_BPTC_RGBFU || dest_format == Image::FORMAT_BPTC_RGBF) {
			PackedFloat32Array push_constant;
			push_constant.push_back(1.0f / width);
			push_constant.push_back(1.0f / height);
			push_constant.push_back(0);
			push_constant.push_back(0);

			Vector<RD::Uniform> uniforms;
			{
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
					u.binding = 0;
					u.append_id(src_sampler);
					u.append_id(src_textures_ptr[i]);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 1;
					u.append_id(dst_textures_ptr[i]);
					uniforms.push_back(u);
				}
			}

			RID uniform_set = rd->uniform_set_create(uniforms, shader, 0);
			RD::ComputeListID compute_list = rd->compute_list_begin();

			rd->compute_list_bind_compute_pipeline(compute_list, pipeline);
			rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
			rd->compute_list_set_push_constant(compute_list, push_constant.ptr(), push_constant.size() * 4);
			rd->compute_list_dispatch(compute_list, get_next_multiple(width, 32) / 32, get_next_multiple(height, 32) / 32, 1);
			rd->compute_list_end();
		}

		rd->submit();
		rd->sync();
	}

	//rd->sync();

	// Container for the compressed data.
	Vector<uint8_t> dst_data;
	dst_data.resize(Image::get_image_data_size(img_clone->get_width(), img_clone->get_height(), dest_format, img_clone->has_mipmaps()));
	uint8_t *dst_data_ptr = dst_data.ptrw();

	// Copy data from the GPU to the buffer.
	for (int i = 0; i < mip_count; i++) {
		const Vector<uint8_t> &texture_data = rd->texture_get_data(dst_textures_ptr[i], 0);
		int ofs = Image::get_image_mipmap_offset(img_clone->get_width(), img_clone->get_height(), dest_format, i);

		memcpy(dst_data_ptr + ofs, texture_data.ptr(), texture_data.size());

		// Free the source and dest texture.
		rd->free(dst_textures_ptr[i]);
		rd->free(src_textures_ptr[i]);
	}

	// Set the compressed data to the image.
	r_img->set_data(img_clone->get_width(), img_clone->get_height(), img_clone->has_mipmaps(), dest_format, dst_data);

	// Free the shader (dependencies will be cleared automatically).
	rd->free(src_sampler);
	rd->free(shader);

	// Free the encoding table.
	if (uses_encoding_table) {
		rd->free(encoding_table_buffer);
	}
}

void _betsy_compress_bptc(Image *r_img, Image::UsedChannels p_channels) {
	Image::Format format = r_img->get_format();

	if (format >= Image::FORMAT_RH && format <= Image::FORMAT_RGBAH) {
		_compress_betsy(BETSY_FORMAT_BC6UF, r_img);
	}
}

void _betsy_compress_etc1(Image *r_img) {
	_compress_betsy(BETSY_FORMAT_ETC1, r_img);
}

void _betsy_compress_etc2(Image *r_img, Image::UsedChannels p_channels) {
	switch (p_channels) {
		case Image::USED_CHANNELS_R:
			//	_compress_betsy(BETSY_FORMAT_ETC2_R11, r_img);
			//	break;

		case Image::USED_CHANNELS_RG:
			//	_compress_betsy(BETSY_FORMAT_ETC2_RG11, r_img);
			//	break;

		case Image::USED_CHANNELS_RGB:
		case Image::USED_CHANNELS_L:
			_compress_betsy(BETSY_FORMAT_ETC2_RGB8, r_img);
			break;

		case Image::USED_CHANNELS_RGBA:
		case Image::USED_CHANNELS_LA:
			_compress_betsy(BETSY_FORMAT_ETC2_RGBA8, r_img);
			break;
	}
}

void _betsy_compress_bc(Image *r_img, Image::UsedChannels p_channels) {
	switch (p_channels) {
		case Image::USED_CHANNELS_R:
			//	_compress_betsy(BETSY_FORMAT_BC4U, r_img);
			//	break;

		case Image::USED_CHANNELS_RG:
			//	_compress_betsy(BETSY_FORMAT_BC5U, r_img);
			//	break;

		case Image::USED_CHANNELS_RGB:
		case Image::USED_CHANNELS_L:
			_compress_betsy(BETSY_FORMAT_BC1, r_img);
			break;

		case Image::USED_CHANNELS_RGBA:
		case Image::USED_CHANNELS_LA:
			_compress_betsy(BETSY_FORMAT_BC3, r_img);
			break;
	}
}
