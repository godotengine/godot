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

#include "servers/rendering/rendering_device_binds.h"
#include "servers/rendering/rendering_server_default.h"

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_context_driver_vulkan.h"
#endif

#include "bc6h.glsl.gen.h"

struct BC6PushConstant {
	float sizeX;
	float sizeY;
	uint32_t padding[2];
};

static int get_next_multiple(int n, int m) {
	return n + (m - (n % m));
}

Error _compress_betsy(BetsyFormat p_format, Image *r_img) {
	uint64_t start_time = OS::get_singleton()->get_ticks_msec();

	if (r_img->is_compressed()) {
		return ERR_INVALID_DATA;
	}

	ERR_FAIL_COND_V_MSG(r_img->get_format() < Image::FORMAT_RF || r_img->get_format() > Image::FORMAT_RGBE9995, ERR_INVALID_DATA, "Image is not an HDR image.");

	Error err = OK;

	// Create local RD.
	RenderingContextDriver *rcd = nullptr;
	RenderingDevice *rd = RenderingServer::get_singleton()->create_local_rendering_device();

	if (rd == nullptr) {
#if defined(RD_ENABLED)
#if defined(VULKAN_ENABLED)
		rcd = memnew(RenderingContextDriverVulkan);
		rd = memnew(RenderingDevice);
#endif
#endif
		if (rcd != nullptr && rd != nullptr) {
			err = rcd->initialize();
			if (err == OK) {
				err = rd->initialize(rcd);
			}

			if (err != OK) {
				memdelete(rd);
				memdelete(rcd);
				rd = nullptr;
				rcd = nullptr;
			}
		}
	}

	ERR_FAIL_NULL_V_MSG(rd, err, "Unable to create a local RenderingDevice.");

	Ref<RDShaderFile> compute_shader;
	compute_shader.instantiate();

	// Destination format.
	Image::Format dest_format = Image::FORMAT_MAX;

	String version = "";

	switch (p_format) {
		case BETSY_FORMAT_BC6: {
			err = compute_shader->parse_versions_from_text(bc6h_shader_glsl);

			if (r_img->detect_signed(true)) {
				dest_format = Image::FORMAT_BPTC_RGBF;
				version = "signed";
			} else {
				dest_format = Image::FORMAT_BPTC_RGBFU;
				version = "unsigned";
			}

		} break;

		default:
			err = ERR_INVALID_PARAMETER;
			break;
	}

	if (err != OK) {
		memdelete(rd);
		if (rcd != nullptr) {
			memdelete(rcd);
		}

		return err;
	}

	// Compile the shader, return early if invalid.
	RID shader = rd->shader_create_from_spirv(compute_shader->get_spirv_stages(version));

	if (shader.is_null()) {
		memdelete(rd);
		if (rcd != nullptr) {
			memdelete(rcd);
		}

		return err;
	}

	RID pipeline = rd->compute_pipeline_create(shader);

	// src_texture format information.
	RD::TextureFormat src_texture_format;
	{
		src_texture_format.array_layers = 1;
		src_texture_format.depth = 1;
		src_texture_format.mipmaps = 1;
		src_texture_format.texture_type = RD::TEXTURE_TYPE_2D;
		src_texture_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	}

	switch (r_img->get_format()) {
		case Image::FORMAT_RH:
			src_texture_format.format = RD::DATA_FORMAT_R16_SFLOAT;
			break;

		case Image::FORMAT_RGH:
			src_texture_format.format = RD::DATA_FORMAT_R16G16_SFLOAT;
			break;

		case Image::FORMAT_RGBH:
			r_img->convert(Image::FORMAT_RGBAH);
			src_texture_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			break;

		case Image::FORMAT_RGBAH:
			src_texture_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			break;

		case Image::FORMAT_RF:
			src_texture_format.format = RD::DATA_FORMAT_R32_SFLOAT;
			break;

		case Image::FORMAT_RGF:
			src_texture_format.format = RD::DATA_FORMAT_R32G32_SFLOAT;
			break;

		case Image::FORMAT_RGBF:
			r_img->convert(Image::FORMAT_RGBAF);
			src_texture_format.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			break;

		case Image::FORMAT_RGBAF:
			src_texture_format.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			break;

		case Image::FORMAT_RGBE9995:
			src_texture_format.format = RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32;
			break;

		default: {
			rd->free(shader);

			memdelete(rd);
			if (rcd != nullptr) {
				memdelete(rcd);
			}

			return err;
		}
	}

	// Create the sampler state.
	RD::SamplerState src_sampler_state;
	{
		src_sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
		src_sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
		src_sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
		src_sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
		src_sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
	}

	RID src_sampler = rd->sampler_create(src_sampler_state);

	// For the destination format just copy the source format and change the usage bits.
	RD::TextureFormat dst_texture_format = src_texture_format;
	dst_texture_format.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
	dst_texture_format.format = RD::DATA_FORMAT_R32G32B32A32_UINT;

	const int mip_count = r_img->get_mipmap_count() + 1;

	// Container for the compressed data.
	Vector<uint8_t> dst_data;
	dst_data.resize(Image::get_image_data_size(r_img->get_width(), r_img->get_height(), dest_format, r_img->has_mipmaps()));
	uint8_t *dst_data_ptr = dst_data.ptrw();

	Vector<Vector<uint8_t>> src_images;
	src_images.push_back(Vector<uint8_t>());
	Vector<uint8_t> *src_image_ptr = src_images.ptrw();

	// Compress each mipmap.
	for (int i = 0; i < mip_count; i++) {
		int64_t ofs, size;
		int width, height;
		r_img->get_mipmap_offset_size_and_dimensions(i, ofs, size, width, height);

		// Set the source texture width and size.
		src_texture_format.height = height;
		src_texture_format.width = width;

		// Set the destination texture width and size.
		dst_texture_format.height = (height + 3) >> 2;
		dst_texture_format.width = (width + 3) >> 2;

		// Create a buffer filled with the source mip layer data.
		src_image_ptr[0].resize(size);
		memcpy(src_image_ptr[0].ptrw(), r_img->ptr() + ofs, size);

		// Create the textures on the GPU.
		RID src_texture = rd->texture_create(src_texture_format, RD::TextureView(), src_images);
		RID dst_texture = rd->texture_create(dst_texture_format, RD::TextureView());

		if (dest_format == Image::FORMAT_BPTC_RGBFU || dest_format == Image::FORMAT_BPTC_RGBF) {
			BC6PushConstant push_constant;
			push_constant.sizeX = 1.0f / width;
			push_constant.sizeY = 1.0f / height;
			push_constant.padding[0] = 0;
			push_constant.padding[1] = 0;

			Vector<RD::Uniform> uniforms;
			{
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
					u.binding = 0;
					u.append_id(src_sampler);
					u.append_id(src_texture);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 1;
					u.append_id(dst_texture);
					uniforms.push_back(u);
				}
			}

			RID uniform_set = rd->uniform_set_create(uniforms, shader, 0);
			RD::ComputeListID compute_list = rd->compute_list_begin();

			rd->compute_list_bind_compute_pipeline(compute_list, pipeline);
			rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(BC6PushConstant));
			rd->compute_list_dispatch(compute_list, get_next_multiple(width, 32) / 32, get_next_multiple(height, 32) / 32, 1);
			rd->compute_list_end();
		}

		rd->submit();
		rd->sync();

		// Copy data from the GPU to the buffer.
		const Vector<uint8_t> texture_data = rd->texture_get_data(dst_texture, 0);
		int64_t dst_ofs = Image::get_image_mipmap_offset(r_img->get_width(), r_img->get_height(), dest_format, i);

		memcpy(dst_data_ptr + dst_ofs, texture_data.ptr(), texture_data.size());

		// Free the source and dest texture.
		rd->free(dst_texture);
		rd->free(src_texture);
	}

	src_images.clear();

	// Set the compressed data to the image.
	r_img->set_data(r_img->get_width(), r_img->get_height(), r_img->has_mipmaps(), dest_format, dst_data);

	// Free the shader (dependencies will be cleared automatically).
	rd->free(src_sampler);
	rd->free(shader);

	memdelete(rd);
	if (rcd != nullptr) {
		memdelete(rcd);
	}

	print_verbose(vformat("Betsy: Encoding took %d ms.", OS::get_singleton()->get_ticks_msec() - start_time));

	return OK;
}

Error _betsy_compress_bptc(Image *r_img, Image::UsedChannels p_channels) {
	Image::Format format = r_img->get_format();

	if (format >= Image::FORMAT_RF && format <= Image::FORMAT_RGBE9995) {
		return _compress_betsy(BETSY_FORMAT_BC6, r_img);
	}

	return ERR_UNAVAILABLE;
}
