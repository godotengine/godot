/*************************************************************************/
/*  rasterizer_scene_rd.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "rasterizer_scene_rd.h"

#include "core/os/os.h"
#include "core/project_settings.h"
#include "rasterizer_rd.h"
#include "servers/rendering/rendering_server_raster.h"

uint64_t RasterizerSceneRD::auto_exposure_counter = 2;

void get_vogel_disk(float *r_kernel, int p_sample_count) {
	const float golden_angle = 2.4;

	for (int i = 0; i < p_sample_count; i++) {
		float r = Math::sqrt(float(i) + 0.5) / Math::sqrt(float(p_sample_count));
		float theta = float(i) * golden_angle;

		r_kernel[i * 4] = Math::cos(theta) * r;
		r_kernel[i * 4 + 1] = Math::sin(theta) * r;
	}
}

void RasterizerSceneRD::_clear_reflection_data(ReflectionData &rd) {
	rd.layers.clear();
	rd.radiance_base_cubemap = RID();
	if (rd.downsampled_radiance_cubemap.is_valid()) {
		RD::get_singleton()->free(rd.downsampled_radiance_cubemap);
	}
	rd.downsampled_radiance_cubemap = RID();
	rd.downsampled_layer.mipmaps.clear();
	rd.coefficient_buffer = RID();
}

void RasterizerSceneRD::_update_reflection_data(ReflectionData &rd, int p_size, int p_mipmaps, bool p_use_array, RID p_base_cube, int p_base_layer, bool p_low_quality) {
	//recreate radiance and all data

	int mipmaps = p_mipmaps;
	uint32_t w = p_size, h = p_size;

	if (p_use_array) {
		int layers = p_low_quality ? 8 : roughness_layers;

		for (int i = 0; i < layers; i++) {
			ReflectionData::Layer layer;
			uint32_t mmw = w;
			uint32_t mmh = h;
			layer.mipmaps.resize(mipmaps);
			layer.views.resize(mipmaps);
			for (int j = 0; j < mipmaps; j++) {
				ReflectionData::Layer::Mipmap &mm = layer.mipmaps.write[j];
				mm.size.width = mmw;
				mm.size.height = mmh;
				for (int k = 0; k < 6; k++) {
					mm.views[k] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer + i * 6 + k, j);
					Vector<RID> fbtex;
					fbtex.push_back(mm.views[k]);
					mm.framebuffers[k] = RD::get_singleton()->framebuffer_create(fbtex);
				}

				layer.views.write[j] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer + i * 6, j, RD::TEXTURE_SLICE_CUBEMAP);

				mmw = MAX(1, mmw >> 1);
				mmh = MAX(1, mmh >> 1);
			}

			rd.layers.push_back(layer);
		}

	} else {
		mipmaps = p_low_quality ? 8 : mipmaps;
		//regular cubemap, lower quality (aliasing, less memory)
		ReflectionData::Layer layer;
		uint32_t mmw = w;
		uint32_t mmh = h;
		layer.mipmaps.resize(mipmaps);
		layer.views.resize(mipmaps);
		for (int j = 0; j < mipmaps; j++) {
			ReflectionData::Layer::Mipmap &mm = layer.mipmaps.write[j];
			mm.size.width = mmw;
			mm.size.height = mmh;
			for (int k = 0; k < 6; k++) {
				mm.views[k] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer + k, j);
				Vector<RID> fbtex;
				fbtex.push_back(mm.views[k]);
				mm.framebuffers[k] = RD::get_singleton()->framebuffer_create(fbtex);
			}

			layer.views.write[j] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer, j, RD::TEXTURE_SLICE_CUBEMAP);

			mmw = MAX(1, mmw >> 1);
			mmh = MAX(1, mmh >> 1);
		}

		rd.layers.push_back(layer);
	}

	rd.radiance_base_cubemap = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer, 0, RD::TEXTURE_SLICE_CUBEMAP);

	RD::TextureFormat tf;
	tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
	tf.width = 64; // Always 64x64
	tf.height = 64;
	tf.type = RD::TEXTURE_TYPE_CUBE;
	tf.array_layers = 6;
	tf.mipmaps = 7;
	tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

	rd.downsampled_radiance_cubemap = RD::get_singleton()->texture_create(tf, RD::TextureView());
	{
		uint32_t mmw = 64;
		uint32_t mmh = 64;
		rd.downsampled_layer.mipmaps.resize(7);
		for (int j = 0; j < rd.downsampled_layer.mipmaps.size(); j++) {
			ReflectionData::DownsampleLayer::Mipmap &mm = rd.downsampled_layer.mipmaps.write[j];
			mm.size.width = mmw;
			mm.size.height = mmh;
			mm.view = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rd.downsampled_radiance_cubemap, 0, j, RD::TEXTURE_SLICE_CUBEMAP);

			mmw = MAX(1, mmw >> 1);
			mmh = MAX(1, mmh >> 1);
		}
	}
}

void RasterizerSceneRD::_create_reflection_fast_filter(ReflectionData &rd, bool p_use_arrays) {
	storage->get_effects()->cubemap_downsample(rd.radiance_base_cubemap, rd.downsampled_layer.mipmaps[0].view, rd.downsampled_layer.mipmaps[0].size);

	for (int i = 1; i < rd.downsampled_layer.mipmaps.size(); i++) {
		storage->get_effects()->cubemap_downsample(rd.downsampled_layer.mipmaps[i - 1].view, rd.downsampled_layer.mipmaps[i].view, rd.downsampled_layer.mipmaps[i].size);
	}

	Vector<RID> views;
	if (p_use_arrays) {
		for (int i = 1; i < rd.layers.size(); i++) {
			views.push_back(rd.layers[i].views[0]);
		}
	} else {
		for (int i = 1; i < rd.layers[0].views.size(); i++) {
			views.push_back(rd.layers[0].views[i]);
		}
	}

	storage->get_effects()->cubemap_filter(rd.downsampled_radiance_cubemap, views, p_use_arrays);
}

void RasterizerSceneRD::_create_reflection_importance_sample(ReflectionData &rd, bool p_use_arrays, int p_cube_side, int p_base_layer) {
	if (p_use_arrays) {
		//render directly to the layers
		storage->get_effects()->cubemap_roughness(rd.radiance_base_cubemap, rd.layers[p_base_layer].views[0], p_cube_side, sky_ggx_samples_quality, float(p_base_layer) / (rd.layers.size() - 1.0), rd.layers[p_base_layer].mipmaps[0].size.x);
	} else {
		storage->get_effects()->cubemap_roughness(rd.layers[0].views[p_base_layer - 1], rd.layers[0].views[p_base_layer], p_cube_side, sky_ggx_samples_quality, float(p_base_layer) / (rd.layers[0].mipmaps.size() - 1.0), rd.layers[0].mipmaps[p_base_layer].size.x);
	}
}

void RasterizerSceneRD::_update_reflection_mipmaps(ReflectionData &rd, int p_start, int p_end) {
	for (int i = p_start; i < p_end; i++) {
		for (int j = 0; j < rd.layers[i].mipmaps.size() - 1; j++) {
			for (int k = 0; k < 6; k++) {
				RID view = rd.layers[i].mipmaps[j].views[k];
				RID texture = rd.layers[i].mipmaps[j + 1].views[k];
				Size2i size = rd.layers[i].mipmaps[j + 1].size;
				storage->get_effects()->make_mipmap(view, texture, size);
			}
		}
	}
}

void RasterizerSceneRD::_sdfgi_erase(RenderBuffers *rb) {
	for (uint32_t i = 0; i < rb->sdfgi->cascades.size(); i++) {
		const SDFGI::Cascade &c = rb->sdfgi->cascades[i];
		RD::get_singleton()->free(c.light_data);
		RD::get_singleton()->free(c.light_aniso_0_tex);
		RD::get_singleton()->free(c.light_aniso_1_tex);
		RD::get_singleton()->free(c.sdf_tex);
		RD::get_singleton()->free(c.solid_cell_dispatch_buffer);
		RD::get_singleton()->free(c.solid_cell_buffer);
		RD::get_singleton()->free(c.lightprobe_history_tex);
		RD::get_singleton()->free(c.lightprobe_average_tex);
		RD::get_singleton()->free(c.lights_buffer);
	}

	RD::get_singleton()->free(rb->sdfgi->render_albedo);
	RD::get_singleton()->free(rb->sdfgi->render_emission);
	RD::get_singleton()->free(rb->sdfgi->render_emission_aniso);

	RD::get_singleton()->free(rb->sdfgi->render_sdf[0]);
	RD::get_singleton()->free(rb->sdfgi->render_sdf[1]);

	RD::get_singleton()->free(rb->sdfgi->render_sdf_half[0]);
	RD::get_singleton()->free(rb->sdfgi->render_sdf_half[1]);

	for (int i = 0; i < 8; i++) {
		RD::get_singleton()->free(rb->sdfgi->render_occlusion[i]);
	}

	RD::get_singleton()->free(rb->sdfgi->render_geom_facing);

	RD::get_singleton()->free(rb->sdfgi->lightprobe_data);
	RD::get_singleton()->free(rb->sdfgi->lightprobe_history_scroll);
	RD::get_singleton()->free(rb->sdfgi->occlusion_data);
	RD::get_singleton()->free(rb->sdfgi->ambient_texture);

	RD::get_singleton()->free(rb->sdfgi->cascades_ubo);

	memdelete(rb->sdfgi);

	rb->sdfgi = nullptr;
}

const Vector3i RasterizerSceneRD::SDFGI::Cascade::DIRTY_ALL = Vector3i(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);

void RasterizerSceneRD::sdfgi_update(RID p_render_buffers, RID p_environment, const Vector3 &p_world_position) {
	Environment *env = environment_owner.getornull(p_environment);
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	bool needs_sdfgi = env && env->sdfgi_enabled;

	if (!needs_sdfgi) {
		if (rb->sdfgi != nullptr) {
			//erase it
			_sdfgi_erase(rb);
			_render_buffers_uniform_set_changed(p_render_buffers);
		}
		return;
	}

	static const uint32_t history_frames_to_converge[RS::ENV_SDFGI_CONVERGE_MAX] = { 5, 10, 15, 20, 25, 30 };
	uint32_t requested_history_size = history_frames_to_converge[sdfgi_frames_to_converge];

	if (rb->sdfgi && (rb->sdfgi->cascade_mode != env->sdfgi_cascades || rb->sdfgi->min_cell_size != env->sdfgi_min_cell_size || requested_history_size != rb->sdfgi->history_size || rb->sdfgi->uses_occlusion != env->sdfgi_use_occlusion || rb->sdfgi->y_scale_mode != env->sdfgi_y_scale)) {
		//configuration changed, erase
		_sdfgi_erase(rb);
	}

	SDFGI *sdfgi = rb->sdfgi;
	if (sdfgi == nullptr) {
		//re-create
		rb->sdfgi = memnew(SDFGI);
		sdfgi = rb->sdfgi;
		sdfgi->cascade_mode = env->sdfgi_cascades;
		sdfgi->min_cell_size = env->sdfgi_min_cell_size;
		sdfgi->uses_occlusion = env->sdfgi_use_occlusion;
		sdfgi->y_scale_mode = env->sdfgi_y_scale;
		static const float y_scale[3] = { 1.0, 1.5, 2.0 };
		sdfgi->y_mult = y_scale[sdfgi->y_scale_mode];
		static const int cascasde_size[3] = { 4, 6, 8 };
		sdfgi->cascades.resize(cascasde_size[sdfgi->cascade_mode]);
		sdfgi->probe_axis_count = SDFGI::PROBE_DIVISOR + 1;
		sdfgi->solid_cell_ratio = sdfgi_solid_cell_ratio;
		sdfgi->solid_cell_count = uint32_t(float(sdfgi->cascade_size * sdfgi->cascade_size * sdfgi->cascade_size) * sdfgi->solid_cell_ratio);

		float base_cell_size = sdfgi->min_cell_size;

		RD::TextureFormat tf_sdf;
		tf_sdf.format = RD::DATA_FORMAT_R8_UNORM;
		tf_sdf.width = sdfgi->cascade_size; // Always 64x64
		tf_sdf.height = sdfgi->cascade_size;
		tf_sdf.depth = sdfgi->cascade_size;
		tf_sdf.type = RD::TEXTURE_TYPE_3D;
		tf_sdf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

		{
			RD::TextureFormat tf_render = tf_sdf;
			tf_render.format = RD::DATA_FORMAT_R16_UINT;
			sdfgi->render_albedo = RD::get_singleton()->texture_create(tf_render, RD::TextureView());
			tf_render.format = RD::DATA_FORMAT_R32_UINT;
			sdfgi->render_emission = RD::get_singleton()->texture_create(tf_render, RD::TextureView());
			sdfgi->render_emission_aniso = RD::get_singleton()->texture_create(tf_render, RD::TextureView());

			tf_render.format = RD::DATA_FORMAT_R8_UNORM; //at least its easy to visualize

			for (int i = 0; i < 8; i++) {
				sdfgi->render_occlusion[i] = RD::get_singleton()->texture_create(tf_render, RD::TextureView());
			}

			tf_render.format = RD::DATA_FORMAT_R32_UINT;
			sdfgi->render_geom_facing = RD::get_singleton()->texture_create(tf_render, RD::TextureView());

			tf_render.format = RD::DATA_FORMAT_R8G8B8A8_UINT;
			sdfgi->render_sdf[0] = RD::get_singleton()->texture_create(tf_render, RD::TextureView());
			sdfgi->render_sdf[1] = RD::get_singleton()->texture_create(tf_render, RD::TextureView());

			tf_render.width /= 2;
			tf_render.height /= 2;
			tf_render.depth /= 2;

			sdfgi->render_sdf_half[0] = RD::get_singleton()->texture_create(tf_render, RD::TextureView());
			sdfgi->render_sdf_half[1] = RD::get_singleton()->texture_create(tf_render, RD::TextureView());
		}

		RD::TextureFormat tf_occlusion = tf_sdf;
		tf_occlusion.format = RD::DATA_FORMAT_R16_UINT;
		tf_occlusion.shareable_formats.push_back(RD::DATA_FORMAT_R16_UINT);
		tf_occlusion.shareable_formats.push_back(RD::DATA_FORMAT_R4G4B4A4_UNORM_PACK16);
		tf_occlusion.depth *= sdfgi->cascades.size(); //use depth for occlusion slices
		tf_occlusion.width *= 2; //use width for the other half

		RD::TextureFormat tf_light = tf_sdf;
		tf_light.format = RD::DATA_FORMAT_R32_UINT;
		tf_light.shareable_formats.push_back(RD::DATA_FORMAT_R32_UINT);
		tf_light.shareable_formats.push_back(RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32);

		RD::TextureFormat tf_aniso0 = tf_sdf;
		tf_aniso0.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		RD::TextureFormat tf_aniso1 = tf_sdf;
		tf_aniso1.format = RD::DATA_FORMAT_R8G8_UNORM;

		int passes = nearest_shift(sdfgi->cascade_size) - 1;

		//store lightprobe SH
		RD::TextureFormat tf_probes;
		tf_probes.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		tf_probes.width = sdfgi->probe_axis_count * sdfgi->probe_axis_count;
		tf_probes.height = sdfgi->probe_axis_count * SDFGI::SH_SIZE;
		tf_probes.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		tf_probes.type = RD::TEXTURE_TYPE_2D_ARRAY;

		sdfgi->history_size = requested_history_size;

		RD::TextureFormat tf_probe_history = tf_probes;
		tf_probe_history.format = RD::DATA_FORMAT_R16G16B16A16_SINT; //signed integer because SH are signed
		tf_probe_history.array_layers = sdfgi->history_size;

		RD::TextureFormat tf_probe_average = tf_probes;
		tf_probe_average.format = RD::DATA_FORMAT_R32G32B32A32_SINT; //signed integer because SH are signed
		tf_probe_average.type = RD::TEXTURE_TYPE_2D_ARRAY;
		tf_probe_average.array_layers = 1;

		sdfgi->lightprobe_history_scroll = RD::get_singleton()->texture_create(tf_probe_history, RD::TextureView());
		sdfgi->lightprobe_average_scroll = RD::get_singleton()->texture_create(tf_probe_average, RD::TextureView());

		{
			//octahedral lightprobes
			RD::TextureFormat tf_octprobes = tf_probes;
			tf_octprobes.array_layers = sdfgi->cascades.size() * 2;
			tf_octprobes.format = RD::DATA_FORMAT_R32_UINT; //pack well with RGBE
			tf_octprobes.width = sdfgi->probe_axis_count * sdfgi->probe_axis_count * (SDFGI::LIGHTPROBE_OCT_SIZE + 2);
			tf_octprobes.height = sdfgi->probe_axis_count * (SDFGI::LIGHTPROBE_OCT_SIZE + 2);
			tf_octprobes.shareable_formats.push_back(RD::DATA_FORMAT_R32_UINT);
			tf_octprobes.shareable_formats.push_back(RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32);
			//lightprobe texture is an octahedral texture

			sdfgi->lightprobe_data = RD::get_singleton()->texture_create(tf_octprobes, RD::TextureView());
			RD::TextureView tv;
			tv.format_override = RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32;
			sdfgi->lightprobe_texture = RD::get_singleton()->texture_create_shared(tv, sdfgi->lightprobe_data);

			//texture handling ambient data, to integrate with volumetric foc
			RD::TextureFormat tf_ambient = tf_probes;
			tf_ambient.array_layers = sdfgi->cascades.size();
			tf_ambient.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT; //pack well with RGBE
			tf_ambient.width = sdfgi->probe_axis_count * sdfgi->probe_axis_count;
			tf_ambient.height = sdfgi->probe_axis_count;
			tf_ambient.type = RD::TEXTURE_TYPE_2D_ARRAY;
			//lightprobe texture is an octahedral texture
			sdfgi->ambient_texture = RD::get_singleton()->texture_create(tf_ambient, RD::TextureView());
		}

		sdfgi->cascades_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(SDFGI::Cascade::UBO) * SDFGI::MAX_CASCADES);

		sdfgi->occlusion_data = RD::get_singleton()->texture_create(tf_occlusion, RD::TextureView());
		{
			RD::TextureView tv;
			tv.format_override = RD::DATA_FORMAT_R4G4B4A4_UNORM_PACK16;
			sdfgi->occlusion_texture = RD::get_singleton()->texture_create_shared(tv, sdfgi->occlusion_data);
		}

		for (uint32_t i = 0; i < sdfgi->cascades.size(); i++) {
			SDFGI::Cascade &cascade = sdfgi->cascades[i];

			/* 3D Textures */

			cascade.sdf_tex = RD::get_singleton()->texture_create(tf_sdf, RD::TextureView());

			cascade.light_data = RD::get_singleton()->texture_create(tf_light, RD::TextureView());

			cascade.light_aniso_0_tex = RD::get_singleton()->texture_create(tf_aniso0, RD::TextureView());
			cascade.light_aniso_1_tex = RD::get_singleton()->texture_create(tf_aniso1, RD::TextureView());

			{
				RD::TextureView tv;
				tv.format_override = RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32;
				cascade.light_tex = RD::get_singleton()->texture_create_shared(tv, cascade.light_data);

				RD::get_singleton()->texture_clear(cascade.light_tex, Color(0, 0, 0, 0), 0, 1, 0, 1);
				RD::get_singleton()->texture_clear(cascade.light_aniso_0_tex, Color(0, 0, 0, 0), 0, 1, 0, 1);
				RD::get_singleton()->texture_clear(cascade.light_aniso_1_tex, Color(0, 0, 0, 0), 0, 1, 0, 1);
			}

			cascade.cell_size = base_cell_size;
			Vector3 world_position = p_world_position;
			world_position.y *= sdfgi->y_mult;
			int32_t probe_cells = sdfgi->cascade_size / SDFGI::PROBE_DIVISOR;
			Vector3 probe_size = Vector3(1, 1, 1) * cascade.cell_size * probe_cells;
			Vector3i probe_pos = Vector3i((world_position / probe_size + Vector3(0.5, 0.5, 0.5)).floor());
			cascade.position = probe_pos * probe_cells;

			cascade.dirty_regions = SDFGI::Cascade::DIRTY_ALL;

			base_cell_size *= 2.0;

			/* Probe History */

			cascade.lightprobe_history_tex = RD::get_singleton()->texture_create(tf_probe_history, RD::TextureView());
			RD::get_singleton()->texture_clear(cascade.lightprobe_history_tex, Color(0, 0, 0, 0), 0, 1, 0, tf_probe_history.array_layers); //needs to be cleared for average to work

			cascade.lightprobe_average_tex = RD::get_singleton()->texture_create(tf_probe_average, RD::TextureView());
			RD::get_singleton()->texture_clear(cascade.lightprobe_average_tex, Color(0, 0, 0, 0), 0, 1, 0, 1); //needs to be cleared for average to work

			/* Buffers */

			cascade.solid_cell_buffer = RD::get_singleton()->storage_buffer_create(sizeof(SDFGI::Cascade::SolidCell) * sdfgi->solid_cell_count);
			cascade.solid_cell_dispatch_buffer = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t) * 4, Vector<uint8_t>(), RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
			cascade.lights_buffer = RD::get_singleton()->storage_buffer_create(sizeof(SDGIShader::Light) * MAX(SDFGI::MAX_STATIC_LIGHTS, SDFGI::MAX_DYNAMIC_LIGHTS));
			{
				Vector<RD::Uniform> uniforms;
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 1;
					u.ids.push_back(sdfgi->render_sdf[(passes & 1) ? 1 : 0]); //if passes are even, we read from buffer 0, else we read from buffer 1
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 2;
					u.ids.push_back(sdfgi->render_albedo);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 3;
					for (int j = 0; j < 8; j++) {
						u.ids.push_back(sdfgi->render_occlusion[j]);
					}
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 4;
					u.ids.push_back(sdfgi->render_emission);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 5;
					u.ids.push_back(sdfgi->render_emission_aniso);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 6;
					u.ids.push_back(sdfgi->render_geom_facing);
					uniforms.push_back(u);
				}

				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 7;
					u.ids.push_back(cascade.sdf_tex);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 8;
					u.ids.push_back(sdfgi->occlusion_data);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 10;
					u.ids.push_back(cascade.solid_cell_dispatch_buffer);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 11;
					u.ids.push_back(cascade.solid_cell_buffer);
					uniforms.push_back(u);
				}

				cascade.sdf_store_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, SDGIShader::PRE_PROCESS_STORE), 0);
			}

			{
				Vector<RD::Uniform> uniforms;
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 1;
					u.ids.push_back(sdfgi->render_albedo);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 2;
					u.ids.push_back(sdfgi->render_geom_facing);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 3;
					u.ids.push_back(sdfgi->render_emission);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 4;
					u.ids.push_back(sdfgi->render_emission_aniso);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 5;
					u.ids.push_back(cascade.solid_cell_dispatch_buffer);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 6;
					u.ids.push_back(cascade.solid_cell_buffer);
					uniforms.push_back(u);
				}

				cascade.scroll_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, SDGIShader::PRE_PROCESS_SCROLL), 0);
			}
			{
				Vector<RD::Uniform> uniforms;
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 1;
					for (int j = 0; j < 8; j++) {
						u.ids.push_back(sdfgi->render_occlusion[j]);
					}
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 2;
					u.ids.push_back(sdfgi->occlusion_data);
					uniforms.push_back(u);
				}

				cascade.scroll_occlusion_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, SDGIShader::PRE_PROCESS_SCROLL_OCCLUSION), 0);
			}
		}

		//direct light
		for (uint32_t i = 0; i < sdfgi->cascades.size(); i++) {
			SDFGI::Cascade &cascade = sdfgi->cascades[i];

			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.binding = 1;
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				for (uint32_t j = 0; j < SDFGI::MAX_CASCADES; j++) {
					if (j < rb->sdfgi->cascades.size()) {
						u.ids.push_back(rb->sdfgi->cascades[j].sdf_tex);
					} else {
						u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
					}
				}
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 2;
				u.type = RD::UNIFORM_TYPE_SAMPLER;
				u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 3;
				u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.ids.push_back(cascade.solid_cell_dispatch_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 4;
				u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.ids.push_back(cascade.solid_cell_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 5;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.ids.push_back(cascade.light_data);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 6;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.ids.push_back(cascade.light_aniso_0_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 7;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.ids.push_back(cascade.light_aniso_1_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 8;
				u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
				u.ids.push_back(rb->sdfgi->cascades_ubo);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 9;
				u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.ids.push_back(cascade.lights_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 10;
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				u.ids.push_back(rb->sdfgi->lightprobe_texture);
				uniforms.push_back(u);
			}

			cascade.sdf_direct_light_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.direct_light.version_get_shader(sdfgi_shader.direct_light_shader, 0), 0);
		}

		//preprocess initialize uniform set
		{
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 1;
				u.ids.push_back(sdfgi->render_albedo);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 2;
				u.ids.push_back(sdfgi->render_sdf[0]);
				uniforms.push_back(u);
			}

			sdfgi->sdf_initialize_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, SDGIShader::PRE_PROCESS_JUMP_FLOOD_INITIALIZE), 0);
		}

		{
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 1;
				u.ids.push_back(sdfgi->render_albedo);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 2;
				u.ids.push_back(sdfgi->render_sdf_half[0]);
				uniforms.push_back(u);
			}

			sdfgi->sdf_initialize_half_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, SDGIShader::PRE_PROCESS_JUMP_FLOOD_INITIALIZE_HALF), 0);
		}

		//jump flood uniform set
		{
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 1;
				u.ids.push_back(sdfgi->render_sdf[0]);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 2;
				u.ids.push_back(sdfgi->render_sdf[1]);
				uniforms.push_back(u);
			}

			sdfgi->jump_flood_uniform_set[0] = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, SDGIShader::PRE_PROCESS_JUMP_FLOOD), 0);
			SWAP(uniforms.write[0].ids.write[0], uniforms.write[1].ids.write[0]);
			sdfgi->jump_flood_uniform_set[1] = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, SDGIShader::PRE_PROCESS_JUMP_FLOOD), 0);
		}
		//jump flood half uniform set
		{
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 1;
				u.ids.push_back(sdfgi->render_sdf_half[0]);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 2;
				u.ids.push_back(sdfgi->render_sdf_half[1]);
				uniforms.push_back(u);
			}

			sdfgi->jump_flood_half_uniform_set[0] = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, SDGIShader::PRE_PROCESS_JUMP_FLOOD), 0);
			SWAP(uniforms.write[0].ids.write[0], uniforms.write[1].ids.write[0]);
			sdfgi->jump_flood_half_uniform_set[1] = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, SDGIShader::PRE_PROCESS_JUMP_FLOOD), 0);
		}

		//upscale half size sdf
		{
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 1;
				u.ids.push_back(sdfgi->render_albedo);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 2;
				u.ids.push_back(sdfgi->render_sdf_half[(passes & 1) ? 0 : 1]); //reverse pass order because half size
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 3;
				u.ids.push_back(sdfgi->render_sdf[(passes & 1) ? 0 : 1]); //reverse pass order because it needs an extra JFA pass
				uniforms.push_back(u);
			}

			sdfgi->upscale_jfa_uniform_set_index = (passes & 1) ? 0 : 1;
			sdfgi->sdf_upscale_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, SDGIShader::PRE_PROCESS_JUMP_FLOOD_UPSCALE), 0);
		}

		//occlusion uniform set
		{
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 1;
				u.ids.push_back(sdfgi->render_albedo);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 2;
				for (int i = 0; i < 8; i++) {
					u.ids.push_back(sdfgi->render_occlusion[i]);
				}
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 3;
				u.ids.push_back(sdfgi->render_geom_facing);
				uniforms.push_back(u);
			}

			sdfgi->occlusion_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, SDGIShader::PRE_PROCESS_OCCLUSION), 0);
		}

		for (uint32_t i = 0; i < sdfgi->cascades.size(); i++) {
			//integrate uniform

			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.binding = 1;
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				for (uint32_t j = 0; j < SDFGI::MAX_CASCADES; j++) {
					if (j < sdfgi->cascades.size()) {
						u.ids.push_back(sdfgi->cascades[j].sdf_tex);
					} else {
						u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
					}
				}
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 2;
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				for (uint32_t j = 0; j < SDFGI::MAX_CASCADES; j++) {
					if (j < sdfgi->cascades.size()) {
						u.ids.push_back(sdfgi->cascades[j].light_tex);
					} else {
						u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
					}
				}
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 3;
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				for (uint32_t j = 0; j < SDFGI::MAX_CASCADES; j++) {
					if (j < sdfgi->cascades.size()) {
						u.ids.push_back(sdfgi->cascades[j].light_aniso_0_tex);
					} else {
						u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
					}
				}
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.binding = 4;
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				for (uint32_t j = 0; j < SDFGI::MAX_CASCADES; j++) {
					if (j < sdfgi->cascades.size()) {
						u.ids.push_back(sdfgi->cascades[j].light_aniso_1_tex);
					} else {
						u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
					}
				}
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_SAMPLER;
				u.binding = 6;
				u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
				uniforms.push_back(u);
			}

			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
				u.binding = 7;
				u.ids.push_back(sdfgi->cascades_ubo);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 8;
				u.ids.push_back(sdfgi->lightprobe_data);
				uniforms.push_back(u);
			}

			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 9;
				u.ids.push_back(sdfgi->cascades[i].lightprobe_history_tex);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 10;
				u.ids.push_back(sdfgi->cascades[i].lightprobe_average_tex);
				uniforms.push_back(u);
			}

			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 11;
				u.ids.push_back(sdfgi->lightprobe_history_scroll);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 12;
				u.ids.push_back(sdfgi->lightprobe_average_scroll);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 13;
				RID parent_average;
				if (i < sdfgi->cascades.size() - 1) {
					parent_average = sdfgi->cascades[i + 1].lightprobe_average_tex;
				} else {
					parent_average = sdfgi->cascades[i - 1].lightprobe_average_tex; //to use something, but it won't be used
				}
				u.ids.push_back(parent_average);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 14;
				u.ids.push_back(sdfgi->ambient_texture);
				uniforms.push_back(u);
			}

			sdfgi->cascades[i].integrate_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.integrate.version_get_shader(sdfgi_shader.integrate_shader, 0), 0);
		}

		sdfgi->uses_multibounce = env->sdfgi_use_multibounce;
		sdfgi->energy = env->sdfgi_energy;
		sdfgi->normal_bias = env->sdfgi_normal_bias;
		sdfgi->probe_bias = env->sdfgi_probe_bias;
		sdfgi->reads_sky = env->sdfgi_read_sky_light;

		_render_buffers_uniform_set_changed(p_render_buffers);

		return; //done. all levels will need to be rendered which its going to take a bit
	}

	//check for updates

	sdfgi->uses_multibounce = env->sdfgi_use_multibounce;
	sdfgi->energy = env->sdfgi_energy;
	sdfgi->normal_bias = env->sdfgi_normal_bias;
	sdfgi->probe_bias = env->sdfgi_probe_bias;
	sdfgi->reads_sky = env->sdfgi_read_sky_light;

	int32_t drag_margin = (sdfgi->cascade_size / SDFGI::PROBE_DIVISOR) / 2;

	for (uint32_t i = 0; i < sdfgi->cascades.size(); i++) {
		SDFGI::Cascade &cascade = sdfgi->cascades[i];
		cascade.dirty_regions = Vector3i();

		Vector3 probe_half_size = Vector3(1, 1, 1) * cascade.cell_size * float(sdfgi->cascade_size / SDFGI::PROBE_DIVISOR) * 0.5;
		probe_half_size = Vector3(0, 0, 0);

		Vector3 world_position = p_world_position;
		world_position.y *= sdfgi->y_mult;
		Vector3i pos_in_cascade = Vector3i((world_position + probe_half_size) / cascade.cell_size);

		for (int j = 0; j < 3; j++) {
			if (pos_in_cascade[j] < cascade.position[j]) {
				while (pos_in_cascade[j] < (cascade.position[j] - drag_margin)) {
					cascade.position[j] -= drag_margin * 2;
					cascade.dirty_regions[j] += drag_margin * 2;
				}
			} else if (pos_in_cascade[j] > cascade.position[j]) {
				while (pos_in_cascade[j] > (cascade.position[j] + drag_margin)) {
					cascade.position[j] += drag_margin * 2;
					cascade.dirty_regions[j] -= drag_margin * 2;
				}
			}

			if (cascade.dirty_regions[j] == 0) {
				continue; // not dirty
			} else if (uint32_t(ABS(cascade.dirty_regions[j])) >= sdfgi->cascade_size) {
				//moved too much, just redraw everything (make all dirty)
				cascade.dirty_regions = SDFGI::Cascade::DIRTY_ALL;
				break;
			}
		}

		if (cascade.dirty_regions != Vector3i() && cascade.dirty_regions != SDFGI::Cascade::DIRTY_ALL) {
			//see how much the total dirty volume represents from the total volume
			uint32_t total_volume = sdfgi->cascade_size * sdfgi->cascade_size * sdfgi->cascade_size;
			uint32_t safe_volume = 1;
			for (int j = 0; j < 3; j++) {
				safe_volume *= sdfgi->cascade_size - ABS(cascade.dirty_regions[j]);
			}
			uint32_t dirty_volume = total_volume - safe_volume;
			if (dirty_volume > (safe_volume / 2)) {
				//more than half the volume is dirty, make all dirty so its only rendered once
				cascade.dirty_regions = SDFGI::Cascade::DIRTY_ALL;
			}
		}
	}
}

int RasterizerSceneRD::sdfgi_get_pending_region_count(RID p_render_buffers) const {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);

	ERR_FAIL_COND_V(rb == nullptr, 0);

	if (rb->sdfgi == nullptr) {
		return 0;
	}

	int dirty_count = 0;
	for (uint32_t i = 0; i < rb->sdfgi->cascades.size(); i++) {
		const SDFGI::Cascade &c = rb->sdfgi->cascades[i];

		if (c.dirty_regions == SDFGI::Cascade::DIRTY_ALL) {
			dirty_count++;
		} else {
			for (int j = 0; j < 3; j++) {
				if (c.dirty_regions[j] != 0) {
					dirty_count++;
				}
			}
		}
	}

	return dirty_count;
}

int RasterizerSceneRD::_sdfgi_get_pending_region_data(RID p_render_buffers, int p_region, Vector3i &r_local_offset, Vector3i &r_local_size, AABB &r_bounds) const {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(rb == nullptr, -1);
	ERR_FAIL_COND_V(rb->sdfgi == nullptr, -1);

	int dirty_count = 0;
	for (uint32_t i = 0; i < rb->sdfgi->cascades.size(); i++) {
		const SDFGI::Cascade &c = rb->sdfgi->cascades[i];

		if (c.dirty_regions == SDFGI::Cascade::DIRTY_ALL) {
			if (dirty_count == p_region) {
				r_local_offset = Vector3i();
				r_local_size = Vector3i(1, 1, 1) * rb->sdfgi->cascade_size;

				r_bounds.position = Vector3((Vector3i(1, 1, 1) * -int32_t(rb->sdfgi->cascade_size >> 1) + c.position)) * c.cell_size * Vector3(1, 1.0 / rb->sdfgi->y_mult, 1);
				r_bounds.size = Vector3(r_local_size) * c.cell_size * Vector3(1, 1.0 / rb->sdfgi->y_mult, 1);
				return i;
			}
			dirty_count++;
		} else {
			for (int j = 0; j < 3; j++) {
				if (c.dirty_regions[j] != 0) {
					if (dirty_count == p_region) {
						Vector3i from = Vector3i(0, 0, 0);
						Vector3i to = Vector3i(1, 1, 1) * rb->sdfgi->cascade_size;

						if (c.dirty_regions[j] > 0) {
							//fill from the beginning
							to[j] = c.dirty_regions[j];
						} else {
							//fill from the end
							from[j] = to[j] + c.dirty_regions[j];
						}

						for (int k = 0; k < j; k++) {
							// "chip" away previous regions to avoid re-voxelizing the same thing
							if (c.dirty_regions[k] > 0) {
								from[k] += c.dirty_regions[k];
							} else if (c.dirty_regions[k] < 0) {
								to[k] += c.dirty_regions[k];
							}
						}

						r_local_offset = from;
						r_local_size = to - from;

						r_bounds.position = Vector3(from + Vector3i(1, 1, 1) * -int32_t(rb->sdfgi->cascade_size >> 1) + c.position) * c.cell_size * Vector3(1, 1.0 / rb->sdfgi->y_mult, 1);
						r_bounds.size = Vector3(r_local_size) * c.cell_size * Vector3(1, 1.0 / rb->sdfgi->y_mult, 1);

						return i;
					}

					dirty_count++;
				}
			}
		}
	}
	return -1;
}

AABB RasterizerSceneRD::sdfgi_get_pending_region_bounds(RID p_render_buffers, int p_region) const {
	AABB bounds;
	Vector3i from;
	Vector3i size;

	int c = _sdfgi_get_pending_region_data(p_render_buffers, p_region, from, size, bounds);
	ERR_FAIL_COND_V(c == -1, AABB());
	return bounds;
}

uint32_t RasterizerSceneRD::sdfgi_get_pending_region_cascade(RID p_render_buffers, int p_region) const {
	AABB bounds;
	Vector3i from;
	Vector3i size;

	return _sdfgi_get_pending_region_data(p_render_buffers, p_region, from, size, bounds);
}

void RasterizerSceneRD::_sdfgi_update_cascades(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(rb == nullptr);
	if (rb->sdfgi == nullptr) {
		return;
	}

	//update cascades
	SDFGI::Cascade::UBO cascade_data[SDFGI::MAX_CASCADES];
	int32_t probe_divisor = rb->sdfgi->cascade_size / SDFGI::PROBE_DIVISOR;

	for (uint32_t i = 0; i < rb->sdfgi->cascades.size(); i++) {
		Vector3 pos = Vector3((Vector3i(1, 1, 1) * -int32_t(rb->sdfgi->cascade_size >> 1) + rb->sdfgi->cascades[i].position)) * rb->sdfgi->cascades[i].cell_size;

		cascade_data[i].offset[0] = pos.x;
		cascade_data[i].offset[1] = pos.y;
		cascade_data[i].offset[2] = pos.z;
		cascade_data[i].to_cell = 1.0 / rb->sdfgi->cascades[i].cell_size;
		cascade_data[i].probe_offset[0] = rb->sdfgi->cascades[i].position.x / probe_divisor;
		cascade_data[i].probe_offset[1] = rb->sdfgi->cascades[i].position.y / probe_divisor;
		cascade_data[i].probe_offset[2] = rb->sdfgi->cascades[i].position.z / probe_divisor;
		cascade_data[i].pad = 0;
	}

	RD::get_singleton()->buffer_update(rb->sdfgi->cascades_ubo, 0, sizeof(SDFGI::Cascade::UBO) * SDFGI::MAX_CASCADES, cascade_data, true);
}

void RasterizerSceneRD::sdfgi_update_probes(RID p_render_buffers, RID p_environment, const RID *p_directional_light_instances, uint32_t p_directional_light_count, const RID *p_positional_light_instances, uint32_t p_positional_light_count) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(rb == nullptr);
	if (rb->sdfgi == nullptr) {
		return;
	}
	Environment *env = environment_owner.getornull(p_environment);

	RENDER_TIMESTAMP(">SDFGI Update Probes");

	/* Update Cascades UBO */
	_sdfgi_update_cascades(p_render_buffers);
	/* Update Dynamic Lights Buffer */

	RENDER_TIMESTAMP("Update Lights");

	/* Update dynamic lights */

	{
		int32_t cascade_light_count[SDFGI::MAX_CASCADES];

		for (uint32_t i = 0; i < rb->sdfgi->cascades.size(); i++) {
			SDFGI::Cascade &cascade = rb->sdfgi->cascades[i];

			SDGIShader::Light lights[SDFGI::MAX_DYNAMIC_LIGHTS];
			uint32_t idx = 0;
			for (uint32_t j = 0; j < p_directional_light_count; j++) {
				if (idx == SDFGI::MAX_DYNAMIC_LIGHTS) {
					break;
				}

				LightInstance *li = light_instance_owner.getornull(p_directional_light_instances[j]);
				ERR_CONTINUE(!li);
				Vector3 dir = -li->transform.basis.get_axis(Vector3::AXIS_Z);
				dir.y *= rb->sdfgi->y_mult;
				dir.normalize();
				lights[idx].direction[0] = dir.x;
				lights[idx].direction[1] = dir.y;
				lights[idx].direction[2] = dir.z;
				Color color = storage->light_get_color(li->light);
				color = color.to_linear();
				lights[idx].color[0] = color.r;
				lights[idx].color[1] = color.g;
				lights[idx].color[2] = color.b;
				lights[idx].type = RS::LIGHT_DIRECTIONAL;
				lights[idx].energy = storage->light_get_param(li->light, RS::LIGHT_PARAM_ENERGY);
				lights[idx].has_shadow = storage->light_has_shadow(li->light);

				idx++;
			}

			AABB cascade_aabb;
			cascade_aabb.position = Vector3((Vector3i(1, 1, 1) * -int32_t(rb->sdfgi->cascade_size >> 1) + cascade.position)) * cascade.cell_size;
			cascade_aabb.size = Vector3(1, 1, 1) * rb->sdfgi->cascade_size * cascade.cell_size;

			for (uint32_t j = 0; j < p_positional_light_count; j++) {
				if (idx == SDFGI::MAX_DYNAMIC_LIGHTS) {
					break;
				}

				LightInstance *li = light_instance_owner.getornull(p_positional_light_instances[j]);
				ERR_CONTINUE(!li);

				uint32_t max_sdfgi_cascade = storage->light_get_max_sdfgi_cascade(li->light);
				if (i > max_sdfgi_cascade) {
					continue;
				}

				if (!cascade_aabb.intersects(li->aabb)) {
					continue;
				}

				Vector3 dir = -li->transform.basis.get_axis(Vector3::AXIS_Z);
				//faster to not do this here
				//dir.y *= rb->sdfgi->y_mult;
				//dir.normalize();
				lights[idx].direction[0] = dir.x;
				lights[idx].direction[1] = dir.y;
				lights[idx].direction[2] = dir.z;
				Vector3 pos = li->transform.origin;
				pos.y *= rb->sdfgi->y_mult;
				lights[idx].position[0] = pos.x;
				lights[idx].position[1] = pos.y;
				lights[idx].position[2] = pos.z;
				Color color = storage->light_get_color(li->light);
				color = color.to_linear();
				lights[idx].color[0] = color.r;
				lights[idx].color[1] = color.g;
				lights[idx].color[2] = color.b;
				lights[idx].type = storage->light_get_type(li->light);
				lights[idx].energy = storage->light_get_param(li->light, RS::LIGHT_PARAM_ENERGY);
				lights[idx].has_shadow = storage->light_has_shadow(li->light);
				lights[idx].attenuation = storage->light_get_param(li->light, RS::LIGHT_PARAM_ATTENUATION);
				lights[idx].radius = storage->light_get_param(li->light, RS::LIGHT_PARAM_RANGE);
				lights[idx].spot_angle = Math::deg2rad(storage->light_get_param(li->light, RS::LIGHT_PARAM_SPOT_ANGLE));
				lights[idx].spot_attenuation = storage->light_get_param(li->light, RS::LIGHT_PARAM_SPOT_ATTENUATION);

				idx++;
			}

			if (idx > 0) {
				RD::get_singleton()->buffer_update(cascade.lights_buffer, 0, idx * sizeof(SDGIShader::Light), lights, true);
			}

			cascade_light_count[i] = idx;
		}

		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.direct_light_pipeline[SDGIShader::DIRECT_LIGHT_MODE_DYNAMIC]);

		SDGIShader::DirectLightPushConstant push_constant;

		push_constant.grid_size[0] = rb->sdfgi->cascade_size;
		push_constant.grid_size[1] = rb->sdfgi->cascade_size;
		push_constant.grid_size[2] = rb->sdfgi->cascade_size;
		push_constant.max_cascades = rb->sdfgi->cascades.size();
		push_constant.probe_axis_size = rb->sdfgi->probe_axis_count;
		push_constant.multibounce = rb->sdfgi->uses_multibounce;
		push_constant.y_mult = rb->sdfgi->y_mult;

		push_constant.process_offset = 0;
		push_constant.process_increment = 1;

		for (uint32_t i = 0; i < rb->sdfgi->cascades.size(); i++) {
			SDFGI::Cascade &cascade = rb->sdfgi->cascades[i];
			push_constant.light_count = cascade_light_count[i];
			push_constant.cascade = i;

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, cascade.sdf_direct_light_uniform_set, 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::DirectLightPushConstant));
			RD::get_singleton()->compute_list_dispatch_indirect(compute_list, cascade.solid_cell_dispatch_buffer, 0);
		}
		RD::get_singleton()->compute_list_end();
	}

	RENDER_TIMESTAMP("Raytrace");

	SDGIShader::IntegratePushConstant push_constant;
	push_constant.grid_size[1] = rb->sdfgi->cascade_size;
	push_constant.grid_size[2] = rb->sdfgi->cascade_size;
	push_constant.grid_size[0] = rb->sdfgi->cascade_size;
	push_constant.max_cascades = rb->sdfgi->cascades.size();
	push_constant.probe_axis_size = rb->sdfgi->probe_axis_count;
	push_constant.history_index = rb->sdfgi->render_pass % rb->sdfgi->history_size;
	push_constant.history_size = rb->sdfgi->history_size;
	static const uint32_t ray_count[RS::ENV_SDFGI_RAY_COUNT_MAX] = { 8, 16, 32, 64, 96, 128 };
	push_constant.ray_count = ray_count[sdfgi_ray_count];
	push_constant.ray_bias = rb->sdfgi->probe_bias;
	push_constant.image_size[0] = rb->sdfgi->probe_axis_count * rb->sdfgi->probe_axis_count;
	push_constant.image_size[1] = rb->sdfgi->probe_axis_count;
	push_constant.store_ambient_texture = env->volumetric_fog_enabled;

	RID sky_uniform_set = sdfgi_shader.integrate_default_sky_uniform_set;
	push_constant.sky_mode = SDGIShader::IntegratePushConstant::SKY_MODE_DISABLED;
	push_constant.y_mult = rb->sdfgi->y_mult;

	if (rb->sdfgi->reads_sky && env) {
		push_constant.sky_energy = env->bg_energy;

		if (env->background == RS::ENV_BG_CLEAR_COLOR) {
			push_constant.sky_mode = SDGIShader::IntegratePushConstant::SKY_MODE_COLOR;
			Color c = storage->get_default_clear_color().to_linear();
			push_constant.sky_color[0] = c.r;
			push_constant.sky_color[1] = c.g;
			push_constant.sky_color[2] = c.b;
		} else if (env->background == RS::ENV_BG_COLOR) {
			push_constant.sky_mode = SDGIShader::IntegratePushConstant::SKY_MODE_COLOR;
			Color c = env->bg_color;
			push_constant.sky_color[0] = c.r;
			push_constant.sky_color[1] = c.g;
			push_constant.sky_color[2] = c.b;

		} else if (env->background == RS::ENV_BG_SKY) {
			Sky *sky = sky_owner.getornull(env->sky);
			if (sky && sky->radiance.is_valid()) {
				if (sky->sdfgi_integrate_sky_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(sky->sdfgi_integrate_sky_uniform_set)) {
					Vector<RD::Uniform> uniforms;

					{
						RD::Uniform u;
						u.type = RD::UNIFORM_TYPE_TEXTURE;
						u.binding = 0;
						u.ids.push_back(sky->radiance);
						uniforms.push_back(u);
					}

					{
						RD::Uniform u;
						u.type = RD::UNIFORM_TYPE_SAMPLER;
						u.binding = 1;
						u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
						uniforms.push_back(u);
					}

					sky->sdfgi_integrate_sky_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.integrate.version_get_shader(sdfgi_shader.integrate_shader, 0), 1);
				}
				sky_uniform_set = sky->sdfgi_integrate_sky_uniform_set;
				push_constant.sky_mode = SDGIShader::IntegratePushConstant::SKY_MODE_SKY;
			}
		}
	}

	rb->sdfgi->render_pass++;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.integrate_pipeline[SDGIShader::INTEGRATE_MODE_PROCESS]);

	int32_t probe_divisor = rb->sdfgi->cascade_size / SDFGI::PROBE_DIVISOR;
	for (uint32_t i = 0; i < rb->sdfgi->cascades.size(); i++) {
		push_constant.cascade = i;
		push_constant.world_offset[0] = rb->sdfgi->cascades[i].position.x / probe_divisor;
		push_constant.world_offset[1] = rb->sdfgi->cascades[i].position.y / probe_divisor;
		push_constant.world_offset[2] = rb->sdfgi->cascades[i].position.z / probe_divisor;

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->cascades[i].integrate_uniform_set, 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, sky_uniform_set, 1);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::IntegratePushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->sdfgi->probe_axis_count * rb->sdfgi->probe_axis_count, rb->sdfgi->probe_axis_count, 1, 8, 8, 1);
	}

	RD::get_singleton()->compute_list_add_barrier(compute_list); //wait until done

	// Then store values into the lightprobe texture. Separating these steps has a small performance hit, but it allows for multiple bounces
	RENDER_TIMESTAMP("Average Probes");

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.integrate_pipeline[SDGIShader::INTEGRATE_MODE_STORE]);

	//convert to octahedral to store
	push_constant.image_size[0] *= SDFGI::LIGHTPROBE_OCT_SIZE;
	push_constant.image_size[1] *= SDFGI::LIGHTPROBE_OCT_SIZE;

	for (uint32_t i = 0; i < rb->sdfgi->cascades.size(); i++) {
		push_constant.cascade = i;
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->cascades[i].integrate_uniform_set, 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::IntegratePushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->sdfgi->probe_axis_count * rb->sdfgi->probe_axis_count * SDFGI::LIGHTPROBE_OCT_SIZE, rb->sdfgi->probe_axis_count * SDFGI::LIGHTPROBE_OCT_SIZE, 1, 8, 8, 1);
	}

	RD::get_singleton()->compute_list_end();

	RENDER_TIMESTAMP("<SDFGI Update Probes");
}

void RasterizerSceneRD::_setup_giprobes(RID p_render_buffers, const Transform &p_transform, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count, uint32_t &r_gi_probes_used) {
	r_gi_probes_used = 0;
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(rb == nullptr);

	RID gi_probe_buffer = render_buffers_get_gi_probe_buffer(p_render_buffers);
	GI::GIProbeData gi_probe_data[RenderBuffers::MAX_GIPROBES];

	bool giprobes_changed = false;

	Transform to_camera;
	to_camera.origin = p_transform.origin; //only translation, make local

	for (int i = 0; i < RenderBuffers::MAX_GIPROBES; i++) {
		RID texture;
		if (i < p_gi_probe_cull_count) {
			GIProbeInstance *gipi = gi_probe_instance_owner.getornull(p_gi_probe_cull_result[i]);

			if (gipi) {
				texture = gipi->texture;
				GI::GIProbeData &gipd = gi_probe_data[i];

				RID base_probe = gipi->probe;

				Transform to_cell = storage->gi_probe_get_to_cell_xform(gipi->probe) * gipi->transform.affine_inverse() * to_camera;

				gipd.xform[0] = to_cell.basis.elements[0][0];
				gipd.xform[1] = to_cell.basis.elements[1][0];
				gipd.xform[2] = to_cell.basis.elements[2][0];
				gipd.xform[3] = 0;
				gipd.xform[4] = to_cell.basis.elements[0][1];
				gipd.xform[5] = to_cell.basis.elements[1][1];
				gipd.xform[6] = to_cell.basis.elements[2][1];
				gipd.xform[7] = 0;
				gipd.xform[8] = to_cell.basis.elements[0][2];
				gipd.xform[9] = to_cell.basis.elements[1][2];
				gipd.xform[10] = to_cell.basis.elements[2][2];
				gipd.xform[11] = 0;
				gipd.xform[12] = to_cell.origin.x;
				gipd.xform[13] = to_cell.origin.y;
				gipd.xform[14] = to_cell.origin.z;
				gipd.xform[15] = 1;

				Vector3 bounds = storage->gi_probe_get_octree_size(base_probe);

				gipd.bounds[0] = bounds.x;
				gipd.bounds[1] = bounds.y;
				gipd.bounds[2] = bounds.z;

				gipd.dynamic_range = storage->gi_probe_get_dynamic_range(base_probe) * storage->gi_probe_get_energy(base_probe);
				gipd.bias = storage->gi_probe_get_bias(base_probe);
				gipd.normal_bias = storage->gi_probe_get_normal_bias(base_probe);
				gipd.blend_ambient = !storage->gi_probe_is_interior(base_probe);
				gipd.anisotropy_strength = 0;
				gipd.ao = storage->gi_probe_get_ao(base_probe);
				gipd.ao_size = Math::pow(storage->gi_probe_get_ao_size(base_probe), 4.0f);
				gipd.mipmaps = gipi->mipmaps.size();
			}

			r_gi_probes_used++;
		}

		if (texture == RID()) {
			texture = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE);
		}

		if (texture != rb->giprobe_textures[i]) {
			giprobes_changed = true;
			rb->giprobe_textures[i] = texture;
		}
	}

	if (giprobes_changed) {
		RD::get_singleton()->free(rb->gi_uniform_set);
		rb->gi_uniform_set = RID();
		if (rb->volumetric_fog) {
			if (RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->uniform_set)) {
				RD::get_singleton()->free(rb->volumetric_fog->uniform_set);
				RD::get_singleton()->free(rb->volumetric_fog->uniform_set2);
			}
			rb->volumetric_fog->uniform_set = RID();
			rb->volumetric_fog->uniform_set2 = RID();
		}
	}

	if (p_gi_probe_cull_count > 0) {
		RD::get_singleton()->buffer_update(gi_probe_buffer, 0, sizeof(GI::GIProbeData) * MIN(RenderBuffers::MAX_GIPROBES, p_gi_probe_cull_count), gi_probe_data, true);
	}
}

void RasterizerSceneRD::_process_gi(RID p_render_buffers, RID p_normal_roughness_buffer, RID p_ambient_buffer, RID p_reflection_buffer, RID p_gi_probe_buffer, RID p_environment, const CameraMatrix &p_projection, const Transform &p_transform, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count) {
	RENDER_TIMESTAMP("Render GI");

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(rb == nullptr);
	Environment *env = environment_owner.getornull(p_environment);

	GI::PushConstant push_constant;

	push_constant.screen_size[0] = rb->width;
	push_constant.screen_size[1] = rb->height;
	push_constant.z_near = p_projection.get_z_near();
	push_constant.z_far = p_projection.get_z_far();
	push_constant.orthogonal = p_projection.is_orthogonal();
	push_constant.proj_info[0] = -2.0f / (rb->width * p_projection.matrix[0][0]);
	push_constant.proj_info[1] = -2.0f / (rb->height * p_projection.matrix[1][1]);
	push_constant.proj_info[2] = (1.0f - p_projection.matrix[0][2]) / p_projection.matrix[0][0];
	push_constant.proj_info[3] = (1.0f + p_projection.matrix[1][2]) / p_projection.matrix[1][1];
	push_constant.max_giprobes = MIN(RenderBuffers::MAX_GIPROBES, p_gi_probe_cull_count);
	push_constant.high_quality_vct = gi_probe_quality == RS::GI_PROBE_QUALITY_HIGH;
	push_constant.use_sdfgi = rb->sdfgi != nullptr;

	if (env) {
		push_constant.ao_color[0] = env->ao_color.r;
		push_constant.ao_color[1] = env->ao_color.g;
		push_constant.ao_color[2] = env->ao_color.b;
	} else {
		push_constant.ao_color[0] = 0;
		push_constant.ao_color[1] = 0;
		push_constant.ao_color[2] = 0;
	}

	push_constant.cam_rotation[0] = p_transform.basis[0][0];
	push_constant.cam_rotation[1] = p_transform.basis[1][0];
	push_constant.cam_rotation[2] = p_transform.basis[2][0];
	push_constant.cam_rotation[3] = 0;
	push_constant.cam_rotation[4] = p_transform.basis[0][1];
	push_constant.cam_rotation[5] = p_transform.basis[1][1];
	push_constant.cam_rotation[6] = p_transform.basis[2][1];
	push_constant.cam_rotation[7] = 0;
	push_constant.cam_rotation[8] = p_transform.basis[0][2];
	push_constant.cam_rotation[9] = p_transform.basis[1][2];
	push_constant.cam_rotation[10] = p_transform.basis[2][2];
	push_constant.cam_rotation[11] = 0;

	if (rb->sdfgi) {
		GI::SDFGIData sdfgi_data;

		sdfgi_data.grid_size[0] = rb->sdfgi->cascade_size;
		sdfgi_data.grid_size[1] = rb->sdfgi->cascade_size;
		sdfgi_data.grid_size[2] = rb->sdfgi->cascade_size;

		sdfgi_data.max_cascades = rb->sdfgi->cascades.size();
		sdfgi_data.probe_axis_size = rb->sdfgi->probe_axis_count;
		sdfgi_data.cascade_probe_size[0] = sdfgi_data.probe_axis_size - 1; //float version for performance
		sdfgi_data.cascade_probe_size[1] = sdfgi_data.probe_axis_size - 1;
		sdfgi_data.cascade_probe_size[2] = sdfgi_data.probe_axis_size - 1;

		float csize = rb->sdfgi->cascade_size;
		sdfgi_data.probe_to_uvw = 1.0 / float(sdfgi_data.cascade_probe_size[0]);
		sdfgi_data.use_occlusion = rb->sdfgi->uses_occlusion;
		//sdfgi_data.energy = rb->sdfgi->energy;

		sdfgi_data.y_mult = rb->sdfgi->y_mult;

		float cascade_voxel_size = (csize / sdfgi_data.cascade_probe_size[0]);
		float occlusion_clamp = (cascade_voxel_size - 0.5) / cascade_voxel_size;
		sdfgi_data.occlusion_clamp[0] = occlusion_clamp;
		sdfgi_data.occlusion_clamp[1] = occlusion_clamp;
		sdfgi_data.occlusion_clamp[2] = occlusion_clamp;
		sdfgi_data.normal_bias = (rb->sdfgi->normal_bias / csize) * sdfgi_data.cascade_probe_size[0];

		//vec2 tex_pixel_size = 1.0 / vec2(ivec2( (OCT_SIZE+2) * params.probe_axis_size * params.probe_axis_size, (OCT_SIZE+2) * params.probe_axis_size ) );
		//vec3 probe_uv_offset = (ivec3(OCT_SIZE+2,OCT_SIZE+2,(OCT_SIZE+2) * params.probe_axis_size)) * tex_pixel_size.xyx;

		uint32_t oct_size = SDFGI::LIGHTPROBE_OCT_SIZE;

		sdfgi_data.lightprobe_tex_pixel_size[0] = 1.0 / ((oct_size + 2) * sdfgi_data.probe_axis_size * sdfgi_data.probe_axis_size);
		sdfgi_data.lightprobe_tex_pixel_size[1] = 1.0 / ((oct_size + 2) * sdfgi_data.probe_axis_size);
		sdfgi_data.lightprobe_tex_pixel_size[2] = 1.0;

		sdfgi_data.energy = rb->sdfgi->energy;

		sdfgi_data.lightprobe_uv_offset[0] = float(oct_size + 2) * sdfgi_data.lightprobe_tex_pixel_size[0];
		sdfgi_data.lightprobe_uv_offset[1] = float(oct_size + 2) * sdfgi_data.lightprobe_tex_pixel_size[1];
		sdfgi_data.lightprobe_uv_offset[2] = float((oct_size + 2) * sdfgi_data.probe_axis_size) * sdfgi_data.lightprobe_tex_pixel_size[0];

		sdfgi_data.occlusion_renormalize[0] = 0.5;
		sdfgi_data.occlusion_renormalize[1] = 1.0;
		sdfgi_data.occlusion_renormalize[2] = 1.0 / float(sdfgi_data.max_cascades);

		int32_t probe_divisor = rb->sdfgi->cascade_size / SDFGI::PROBE_DIVISOR;

		for (uint32_t i = 0; i < sdfgi_data.max_cascades; i++) {
			GI::SDFGIData::ProbeCascadeData &c = sdfgi_data.cascades[i];
			Vector3 pos = Vector3((Vector3i(1, 1, 1) * -int32_t(rb->sdfgi->cascade_size >> 1) + rb->sdfgi->cascades[i].position)) * rb->sdfgi->cascades[i].cell_size;
			Vector3 cam_origin = p_transform.origin;
			cam_origin.y *= rb->sdfgi->y_mult;
			pos -= cam_origin; //make pos local to camera, to reduce numerical error
			c.position[0] = pos.x;
			c.position[1] = pos.y;
			c.position[2] = pos.z;
			c.to_probe = 1.0 / (float(rb->sdfgi->cascade_size) * rb->sdfgi->cascades[i].cell_size / float(rb->sdfgi->probe_axis_count - 1));

			Vector3i probe_ofs = rb->sdfgi->cascades[i].position / probe_divisor;
			c.probe_world_offset[0] = probe_ofs.x;
			c.probe_world_offset[1] = probe_ofs.y;
			c.probe_world_offset[2] = probe_ofs.z;

			c.to_cell = 1.0 / rb->sdfgi->cascades[i].cell_size;
		}

		RD::get_singleton()->buffer_update(gi.sdfgi_ubo, 0, sizeof(GI::SDFGIData), &sdfgi_data, true);
	}

	if (rb->gi_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(rb->gi_uniform_set)) {
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.binding = 1;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			for (uint32_t j = 0; j < SDFGI::MAX_CASCADES; j++) {
				if (rb->sdfgi && j < rb->sdfgi->cascades.size()) {
					u.ids.push_back(rb->sdfgi->cascades[j].sdf_tex);
				} else {
					u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
				}
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 2;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			for (uint32_t j = 0; j < SDFGI::MAX_CASCADES; j++) {
				if (rb->sdfgi && j < rb->sdfgi->cascades.size()) {
					u.ids.push_back(rb->sdfgi->cascades[j].light_tex);
				} else {
					u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
				}
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 3;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			for (uint32_t j = 0; j < SDFGI::MAX_CASCADES; j++) {
				if (rb->sdfgi && j < rb->sdfgi->cascades.size()) {
					u.ids.push_back(rb->sdfgi->cascades[j].light_aniso_0_tex);
				} else {
					u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
				}
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 4;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			for (uint32_t j = 0; j < SDFGI::MAX_CASCADES; j++) {
				if (rb->sdfgi && j < rb->sdfgi->cascades.size()) {
					u.ids.push_back(rb->sdfgi->cascades[j].light_aniso_1_tex);
				} else {
					u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
				}
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 5;
			if (rb->sdfgi) {
				u.ids.push_back(rb->sdfgi->occlusion_texture);
			} else {
				u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 6;
			u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 7;
			u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 9;
			u.ids.push_back(p_ambient_buffer);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 10;
			u.ids.push_back(p_reflection_buffer);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 11;
			if (rb->sdfgi) {
				u.ids.push_back(rb->sdfgi->lightprobe_texture);
			} else {
				u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE));
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 12;
			u.ids.push_back(rb->depth_texture);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 13;
			u.ids.push_back(p_normal_roughness_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 14;
			RID buffer = p_gi_probe_buffer.is_valid() ? p_gi_probe_buffer : storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_BLACK);
			u.ids.push_back(buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 15;
			u.ids.push_back(gi.sdfgi_ubo);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 16;
			u.ids.push_back(rb->giprobe_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 17;
			for (int i = 0; i < RenderBuffers::MAX_GIPROBES; i++) {
				u.ids.push_back(rb->giprobe_textures[i]);
			}
			uniforms.push_back(u);
		}

		rb->gi_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, gi.shader.version_get_shader(gi.shader_version, 0), 0);
	}

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi.pipelines[0]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->gi_uniform_set, 0);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(GI::PushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->width, rb->height, 1, 8, 8, 1);
	RD::get_singleton()->compute_list_end();
}

RID RasterizerSceneRD::sky_create() {
	return sky_owner.make_rid(Sky());
}

void RasterizerSceneRD::_sky_invalidate(Sky *p_sky) {
	if (!p_sky->dirty) {
		p_sky->dirty = true;
		p_sky->dirty_list = dirty_sky_list;
		dirty_sky_list = p_sky;
	}
}

void RasterizerSceneRD::sky_set_radiance_size(RID p_sky, int p_radiance_size) {
	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND(!sky);
	ERR_FAIL_COND(p_radiance_size < 32 || p_radiance_size > 2048);
	if (sky->radiance_size == p_radiance_size) {
		return;
	}
	sky->radiance_size = p_radiance_size;

	if (sky->mode == RS::SKY_MODE_REALTIME && sky->radiance_size != 256) {
		WARN_PRINT("Realtime Skies can only use a radiance size of 256. Radiance size will be set to 256 internally.");
		sky->radiance_size = 256;
	}

	_sky_invalidate(sky);
	if (sky->radiance.is_valid()) {
		RD::get_singleton()->free(sky->radiance);
		sky->radiance = RID();
	}
	_clear_reflection_data(sky->reflection);
}

void RasterizerSceneRD::sky_set_mode(RID p_sky, RS::SkyMode p_mode) {
	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND(!sky);

	if (sky->mode == p_mode) {
		return;
	}

	sky->mode = p_mode;

	if (sky->mode == RS::SKY_MODE_REALTIME && sky->radiance_size != 256) {
		WARN_PRINT("Realtime Skies can only use a radiance size of 256. Radiance size will be set to 256 internally.");
		sky_set_radiance_size(p_sky, 256);
	}

	_sky_invalidate(sky);
	if (sky->radiance.is_valid()) {
		RD::get_singleton()->free(sky->radiance);
		sky->radiance = RID();
	}
	_clear_reflection_data(sky->reflection);
}

void RasterizerSceneRD::sky_set_material(RID p_sky, RID p_material) {
	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND(!sky);
	sky->material = p_material;
	_sky_invalidate(sky);
}

Ref<Image> RasterizerSceneRD::sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) {
	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND_V(!sky, Ref<Image>());

	_update_dirty_skys();

	if (sky->radiance.is_valid()) {
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
		tf.width = p_size.width;
		tf.height = p_size.height;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

		RID rad_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());
		storage->get_effects()->copy_cubemap_to_panorama(sky->radiance, rad_tex, p_size, p_bake_irradiance ? roughness_layers : 0, sky->reflection.layers.size() > 1);
		Vector<uint8_t> data = RD::get_singleton()->texture_get_data(rad_tex, 0);
		RD::get_singleton()->free(rad_tex);

		Ref<Image> img;
		img.instance();
		img->create(p_size.width, p_size.height, false, Image::FORMAT_RGBAF, data);
		for (int i = 0; i < p_size.width; i++) {
			for (int j = 0; j < p_size.height; j++) {
				Color c = img->get_pixel(i, j);
				c.r *= p_energy;
				c.g *= p_energy;
				c.b *= p_energy;
				img->set_pixel(i, j, c);
			}
		}
		return img;
	}

	return Ref<Image>();
}

void RasterizerSceneRD::_update_dirty_skys() {
	Sky *sky = dirty_sky_list;

	while (sky) {
		bool texture_set_dirty = false;
		//update sky configuration if texture is missing

		if (sky->radiance.is_null()) {
			int mipmaps = Image::get_image_required_mipmaps(sky->radiance_size, sky->radiance_size, Image::FORMAT_RGBAH) + 1;

			uint32_t w = sky->radiance_size, h = sky->radiance_size;
			int layers = roughness_layers;
			if (sky->mode == RS::SKY_MODE_REALTIME) {
				layers = 8;
				if (roughness_layers != 8) {
					WARN_PRINT("When using REALTIME skies, roughness_layers should be set to 8 in the project settings for best quality reflections");
				}
			}

			if (sky_use_cubemap_array) {
				//array (higher quality, 6 times more memory)
				RD::TextureFormat tf;
				tf.array_layers = layers * 6;
				tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
				tf.type = RD::TEXTURE_TYPE_CUBE_ARRAY;
				tf.mipmaps = mipmaps;
				tf.width = w;
				tf.height = h;
				tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

				sky->radiance = RD::get_singleton()->texture_create(tf, RD::TextureView());

				_update_reflection_data(sky->reflection, sky->radiance_size, mipmaps, true, sky->radiance, 0, sky->mode == RS::SKY_MODE_REALTIME);

			} else {
				//regular cubemap, lower quality (aliasing, less memory)
				RD::TextureFormat tf;
				tf.array_layers = 6;
				tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
				tf.type = RD::TEXTURE_TYPE_CUBE;
				tf.mipmaps = MIN(mipmaps, layers);
				tf.width = w;
				tf.height = h;
				tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

				sky->radiance = RD::get_singleton()->texture_create(tf, RD::TextureView());

				_update_reflection_data(sky->reflection, sky->radiance_size, MIN(mipmaps, layers), false, sky->radiance, 0, sky->mode == RS::SKY_MODE_REALTIME);
			}
			texture_set_dirty = true;
		}

		// Create subpass buffers if they haven't been created already
		if (sky->half_res_pass.is_null() && !RD::get_singleton()->texture_is_valid(sky->half_res_pass) && sky->screen_size.x >= 4 && sky->screen_size.y >= 4) {
			RD::TextureFormat tformat;
			tformat.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			tformat.width = sky->screen_size.x / 2;
			tformat.height = sky->screen_size.y / 2;
			tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
			tformat.type = RD::TEXTURE_TYPE_2D;

			sky->half_res_pass = RD::get_singleton()->texture_create(tformat, RD::TextureView());
			Vector<RID> texs;
			texs.push_back(sky->half_res_pass);
			sky->half_res_framebuffer = RD::get_singleton()->framebuffer_create(texs);
			texture_set_dirty = true;
		}

		if (sky->quarter_res_pass.is_null() && !RD::get_singleton()->texture_is_valid(sky->quarter_res_pass) && sky->screen_size.x >= 4 && sky->screen_size.y >= 4) {
			RD::TextureFormat tformat;
			tformat.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			tformat.width = sky->screen_size.x / 4;
			tformat.height = sky->screen_size.y / 4;
			tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
			tformat.type = RD::TEXTURE_TYPE_2D;

			sky->quarter_res_pass = RD::get_singleton()->texture_create(tformat, RD::TextureView());
			Vector<RID> texs;
			texs.push_back(sky->quarter_res_pass);
			sky->quarter_res_framebuffer = RD::get_singleton()->framebuffer_create(texs);
			texture_set_dirty = true;
		}

		if (texture_set_dirty) {
			for (int i = 0; i < SKY_TEXTURE_SET_MAX; i++) {
				if (sky->texture_uniform_sets[i].is_valid() && RD::get_singleton()->uniform_set_is_valid(sky->texture_uniform_sets[i])) {
					RD::get_singleton()->free(sky->texture_uniform_sets[i]);
					sky->texture_uniform_sets[i] = RID();
				}
			}
		}

		sky->reflection.dirty = true;
		sky->processing_layer = 0;

		Sky *next = sky->dirty_list;
		sky->dirty_list = nullptr;
		sky->dirty = false;
		sky = next;
	}

	dirty_sky_list = nullptr;
}

RID RasterizerSceneRD::sky_get_radiance_texture_rd(RID p_sky) const {
	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND_V(!sky, RID());

	return sky->radiance;
}

RID RasterizerSceneRD::sky_get_radiance_uniform_set_rd(RID p_sky, RID p_shader, int p_set) const {
	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND_V(!sky, RID());

	if (sky->uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(sky->uniform_set)) {
		sky->uniform_set = RID();
		if (sky->radiance.is_valid()) {
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 0;
				u.ids.push_back(sky->radiance);
				uniforms.push_back(u);
			}

			sky->uniform_set = RD::get_singleton()->uniform_set_create(uniforms, p_shader, p_set);
		}
	}

	return sky->uniform_set;
}

RID RasterizerSceneRD::_get_sky_textures(Sky *p_sky, SkyTextureSetVersion p_version) {
	if (p_sky->texture_uniform_sets[p_version].is_valid() && RD::get_singleton()->uniform_set_is_valid(p_sky->texture_uniform_sets[p_version])) {
		return p_sky->texture_uniform_sets[p_version];
	}
	Vector<RD::Uniform> uniforms;
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 0;
		if (p_sky->radiance.is_valid() && p_version <= SKY_TEXTURE_SET_QUARTER_RES) {
			u.ids.push_back(p_sky->radiance);
		} else {
			u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK));
		}
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 1; // half res
		if (p_sky->half_res_pass.is_valid() && p_version != SKY_TEXTURE_SET_HALF_RES && p_version != SKY_TEXTURE_SET_CUBEMAP_HALF_RES) {
			if (p_version >= SKY_TEXTURE_SET_CUBEMAP) {
				u.ids.push_back(p_sky->reflection.layers[0].views[1]);
			} else {
				u.ids.push_back(p_sky->half_res_pass);
			}
		} else {
			if (p_version < SKY_TEXTURE_SET_CUBEMAP) {
				u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE));
			} else {
				u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK));
			}
		}
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 2; // quarter res
		if (p_sky->quarter_res_pass.is_valid() && p_version != SKY_TEXTURE_SET_QUARTER_RES && p_version != SKY_TEXTURE_SET_CUBEMAP_QUARTER_RES) {
			if (p_version >= SKY_TEXTURE_SET_CUBEMAP) {
				u.ids.push_back(p_sky->reflection.layers[0].views[2]);
			} else {
				u.ids.push_back(p_sky->quarter_res_pass);
			}
		} else {
			if (p_version < SKY_TEXTURE_SET_CUBEMAP) {
				u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE));
			} else {
				u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK));
			}
		}
		uniforms.push_back(u);
	}

	p_sky->texture_uniform_sets[p_version] = RD::get_singleton()->uniform_set_create(uniforms, sky_shader.default_shader_rd, SKY_SET_TEXTURES);
	return p_sky->texture_uniform_sets[p_version];
}

RID RasterizerSceneRD::sky_get_material(RID p_sky) const {
	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND_V(!sky, RID());

	return sky->material;
}

void RasterizerSceneRD::_draw_sky(bool p_can_continue_color, bool p_can_continue_depth, RID p_fb, RID p_environment, const CameraMatrix &p_projection, const Transform &p_transform) {
	ERR_FAIL_COND(!is_environment(p_environment));

	SkyMaterialData *material = nullptr;

	Sky *sky = sky_owner.getornull(environment_get_sky(p_environment));

	RID sky_material;

	RS::EnvironmentBG background = environment_get_background(p_environment);

	if (!(background == RS::ENV_BG_CLEAR_COLOR || background == RS::ENV_BG_COLOR) || sky) {
		ERR_FAIL_COND(!sky);
		sky_material = sky_get_material(environment_get_sky(p_environment));

		if (sky_material.is_valid()) {
			material = (SkyMaterialData *)storage->material_get_data(sky_material, RasterizerStorageRD::SHADER_TYPE_SKY);
			if (!material || !material->shader_data->valid) {
				material = nullptr;
			}
		}

		if (!material) {
			sky_material = sky_shader.default_material;
			material = (SkyMaterialData *)storage->material_get_data(sky_material, RasterizerStorageRD::SHADER_TYPE_SKY);
		}
	}

	if (background == RS::ENV_BG_CLEAR_COLOR || background == RS::ENV_BG_COLOR) {
		sky_material = sky_scene_state.fog_material;
		material = (SkyMaterialData *)storage->material_get_data(sky_material, RasterizerStorageRD::SHADER_TYPE_SKY);
	}

	ERR_FAIL_COND(!material);

	SkyShaderData *shader_data = material->shader_data;

	ERR_FAIL_COND(!shader_data);

	Basis sky_transform = environment_get_sky_orientation(p_environment);
	sky_transform.invert();

	float multiplier = environment_get_bg_energy(p_environment);
	float custom_fov = environment_get_sky_custom_fov(p_environment);
	// Camera
	CameraMatrix camera;

	if (custom_fov) {
		float near_plane = p_projection.get_z_near();
		float far_plane = p_projection.get_z_far();
		float aspect = p_projection.get_aspect();

		camera.set_perspective(custom_fov, aspect, near_plane, far_plane);

	} else {
		camera = p_projection;
	}

	sky_transform = p_transform.basis * sky_transform;

	if (shader_data->uses_quarter_res) {
		RenderPipelineVertexFormatCacheRD *pipeline = &shader_data->pipelines[SKY_VERSION_QUARTER_RES];

		RID texture_uniform_set = _get_sky_textures(sky, SKY_TEXTURE_SET_QUARTER_RES);

		Vector<Color> clear_colors;
		clear_colors.push_back(Color(0.0, 0.0, 0.0));

		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(sky->quarter_res_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, clear_colors);
		storage->get_effects()->render_sky(draw_list, time, sky->quarter_res_framebuffer, sky_scene_state.uniform_set, sky_scene_state.fog_uniform_set, pipeline, material->uniform_set, texture_uniform_set, camera, sky_transform, multiplier, p_transform.origin);
		RD::get_singleton()->draw_list_end();
	}

	if (shader_data->uses_half_res) {
		RenderPipelineVertexFormatCacheRD *pipeline = &shader_data->pipelines[SKY_VERSION_HALF_RES];

		RID texture_uniform_set = _get_sky_textures(sky, SKY_TEXTURE_SET_HALF_RES);

		Vector<Color> clear_colors;
		clear_colors.push_back(Color(0.0, 0.0, 0.0));

		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(sky->half_res_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, clear_colors);
		storage->get_effects()->render_sky(draw_list, time, sky->half_res_framebuffer, sky_scene_state.uniform_set, sky_scene_state.fog_uniform_set, pipeline, material->uniform_set, texture_uniform_set, camera, sky_transform, multiplier, p_transform.origin);
		RD::get_singleton()->draw_list_end();
	}

	RenderPipelineVertexFormatCacheRD *pipeline = &shader_data->pipelines[SKY_VERSION_BACKGROUND];

	RID texture_uniform_set;
	if (sky) {
		texture_uniform_set = _get_sky_textures(sky, SKY_TEXTURE_SET_BACKGROUND);
	} else {
		texture_uniform_set = sky_scene_state.fog_only_texture_uniform_set;
	}

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_fb, RD::INITIAL_ACTION_CONTINUE, p_can_continue_color ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CONTINUE, p_can_continue_depth ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ);
	storage->get_effects()->render_sky(draw_list, time, p_fb, sky_scene_state.uniform_set, sky_scene_state.fog_uniform_set, pipeline, material->uniform_set, texture_uniform_set, camera, sky_transform, multiplier, p_transform.origin);
	RD::get_singleton()->draw_list_end();
}

void RasterizerSceneRD::_setup_sky(RID p_environment, RID p_render_buffers, const CameraMatrix &p_projection, const Transform &p_transform, const Size2i p_screen_size) {
	ERR_FAIL_COND(!is_environment(p_environment));

	SkyMaterialData *material = nullptr;

	Sky *sky = sky_owner.getornull(environment_get_sky(p_environment));

	RID sky_material;

	SkyShaderData *shader_data = nullptr;

	RS::EnvironmentBG background = environment_get_background(p_environment);

	if (!(background == RS::ENV_BG_CLEAR_COLOR || background == RS::ENV_BG_COLOR) || sky) {
		ERR_FAIL_COND(!sky);
		sky_material = sky_get_material(environment_get_sky(p_environment));

		if (sky_material.is_valid()) {
			material = (SkyMaterialData *)storage->material_get_data(sky_material, RasterizerStorageRD::SHADER_TYPE_SKY);
			if (!material || !material->shader_data->valid) {
				material = nullptr;
			}
		}

		if (!material) {
			sky_material = sky_shader.default_material;
			material = (SkyMaterialData *)storage->material_get_data(sky_material, RasterizerStorageRD::SHADER_TYPE_SKY);
		}

		ERR_FAIL_COND(!material);

		shader_data = material->shader_data;

		ERR_FAIL_COND(!shader_data);
	}

	if (sky) {
		// Invalidate supbass buffers if screen size changes
		if (sky->screen_size != p_screen_size) {
			sky->screen_size = p_screen_size;
			sky->screen_size.x = sky->screen_size.x < 4 ? 4 : sky->screen_size.x;
			sky->screen_size.y = sky->screen_size.y < 4 ? 4 : sky->screen_size.y;
			if (shader_data->uses_half_res) {
				if (sky->half_res_pass.is_valid()) {
					RD::get_singleton()->free(sky->half_res_pass);
					sky->half_res_pass = RID();
				}
				_sky_invalidate(sky);
			}
			if (shader_data->uses_quarter_res) {
				if (sky->quarter_res_pass.is_valid()) {
					RD::get_singleton()->free(sky->quarter_res_pass);
					sky->quarter_res_pass = RID();
				}
				_sky_invalidate(sky);
			}
		}

		// Create new subpass buffers if necessary
		if ((shader_data->uses_half_res && sky->half_res_pass.is_null()) ||
				(shader_data->uses_quarter_res && sky->quarter_res_pass.is_null()) ||
				sky->radiance.is_null()) {
			_sky_invalidate(sky);
			_update_dirty_skys();
		}

		if (shader_data->uses_time && time - sky->prev_time > 0.00001) {
			sky->prev_time = time;
			sky->reflection.dirty = true;
			RenderingServerRaster::redraw_request();
		}

		if (material != sky->prev_material) {
			sky->prev_material = material;
			sky->reflection.dirty = true;
		}

		if (material->uniform_set_updated) {
			material->uniform_set_updated = false;
			sky->reflection.dirty = true;
		}

		if (!p_transform.origin.is_equal_approx(sky->prev_position) && shader_data->uses_position) {
			sky->prev_position = p_transform.origin;
			sky->reflection.dirty = true;
		}

		if (shader_data->uses_light) {
			// Check whether the directional_light_buffer changes
			bool light_data_dirty = false;

			if (sky_scene_state.ubo.directional_light_count != sky_scene_state.last_frame_directional_light_count) {
				light_data_dirty = true;
				for (uint32_t i = sky_scene_state.ubo.directional_light_count; i < sky_scene_state.max_directional_lights; i++) {
					sky_scene_state.directional_lights[i].enabled = false;
				}
			}
			if (!light_data_dirty) {
				for (uint32_t i = 0; i < sky_scene_state.ubo.directional_light_count; i++) {
					if (sky_scene_state.directional_lights[i].direction[0] != sky_scene_state.last_frame_directional_lights[i].direction[0] ||
							sky_scene_state.directional_lights[i].direction[1] != sky_scene_state.last_frame_directional_lights[i].direction[1] ||
							sky_scene_state.directional_lights[i].direction[2] != sky_scene_state.last_frame_directional_lights[i].direction[2] ||
							sky_scene_state.directional_lights[i].energy != sky_scene_state.last_frame_directional_lights[i].energy ||
							sky_scene_state.directional_lights[i].color[0] != sky_scene_state.last_frame_directional_lights[i].color[0] ||
							sky_scene_state.directional_lights[i].color[1] != sky_scene_state.last_frame_directional_lights[i].color[1] ||
							sky_scene_state.directional_lights[i].color[2] != sky_scene_state.last_frame_directional_lights[i].color[2] ||
							sky_scene_state.directional_lights[i].enabled != sky_scene_state.last_frame_directional_lights[i].enabled ||
							sky_scene_state.directional_lights[i].size != sky_scene_state.last_frame_directional_lights[i].size) {
						light_data_dirty = true;
						break;
					}
				}
			}

			if (light_data_dirty) {
				RD::get_singleton()->buffer_update(sky_scene_state.directional_light_buffer, 0, sizeof(SkyDirectionalLightData) * sky_scene_state.max_directional_lights, sky_scene_state.directional_lights, true);

				RasterizerSceneRD::SkyDirectionalLightData *temp = sky_scene_state.last_frame_directional_lights;
				sky_scene_state.last_frame_directional_lights = sky_scene_state.directional_lights;
				sky_scene_state.directional_lights = temp;
				sky_scene_state.last_frame_directional_light_count = sky_scene_state.ubo.directional_light_count;
				sky->reflection.dirty = true;
			}
		}
	}

	//setup fog variables
	sky_scene_state.ubo.volumetric_fog_enabled = false;
	if (p_render_buffers.is_valid()) {
		if (render_buffers_has_volumetric_fog(p_render_buffers)) {
			sky_scene_state.ubo.volumetric_fog_enabled = true;

			float fog_end = render_buffers_get_volumetric_fog_end(p_render_buffers);
			if (fog_end > 0.0) {
				sky_scene_state.ubo.volumetric_fog_inv_length = 1.0 / fog_end;
			} else {
				sky_scene_state.ubo.volumetric_fog_inv_length = 1.0;
			}

			float fog_detail_spread = render_buffers_get_volumetric_fog_detail_spread(p_render_buffers); //reverse lookup
			if (fog_detail_spread > 0.0) {
				sky_scene_state.ubo.volumetric_fog_detail_spread = 1.0 / fog_detail_spread;
			} else {
				sky_scene_state.ubo.volumetric_fog_detail_spread = 1.0;
			}
		}

		RID fog_uniform_set = render_buffers_get_volumetric_fog_sky_uniform_set(p_render_buffers);

		if (fog_uniform_set != RID()) {
			sky_scene_state.fog_uniform_set = fog_uniform_set;
		} else {
			sky_scene_state.fog_uniform_set = sky_scene_state.default_fog_uniform_set;
		}
	}

	sky_scene_state.ubo.z_far = p_projection.get_z_far();
	sky_scene_state.ubo.fog_enabled = environment_is_fog_enabled(p_environment);
	sky_scene_state.ubo.fog_density = environment_get_fog_density(p_environment);
	sky_scene_state.ubo.fog_aerial_perspective = environment_get_fog_aerial_perspective(p_environment);
	Color fog_color = environment_get_fog_light_color(p_environment).to_linear();
	float fog_energy = environment_get_fog_light_energy(p_environment);
	sky_scene_state.ubo.fog_light_color[0] = fog_color.r * fog_energy;
	sky_scene_state.ubo.fog_light_color[1] = fog_color.g * fog_energy;
	sky_scene_state.ubo.fog_light_color[2] = fog_color.b * fog_energy;
	sky_scene_state.ubo.fog_sun_scatter = environment_get_fog_sun_scatter(p_environment);

	RD::get_singleton()->buffer_update(sky_scene_state.uniform_buffer, 0, sizeof(SkySceneState::UBO), &sky_scene_state.ubo, true);
}

void RasterizerSceneRD::_update_sky(RID p_environment, const CameraMatrix &p_projection, const Transform &p_transform) {
	ERR_FAIL_COND(!is_environment(p_environment));

	Sky *sky = sky_owner.getornull(environment_get_sky(p_environment));
	ERR_FAIL_COND(!sky);

	RID sky_material = sky_get_material(environment_get_sky(p_environment));

	SkyMaterialData *material = nullptr;

	if (sky_material.is_valid()) {
		material = (SkyMaterialData *)storage->material_get_data(sky_material, RasterizerStorageRD::SHADER_TYPE_SKY);
		if (!material || !material->shader_data->valid) {
			material = nullptr;
		}
	}

	if (!material) {
		sky_material = sky_shader.default_material;
		material = (SkyMaterialData *)storage->material_get_data(sky_material, RasterizerStorageRD::SHADER_TYPE_SKY);
	}

	ERR_FAIL_COND(!material);

	SkyShaderData *shader_data = material->shader_data;

	ERR_FAIL_COND(!shader_data);

	float multiplier = environment_get_bg_energy(p_environment);

	bool update_single_frame = sky->mode == RS::SKY_MODE_REALTIME || sky->mode == RS::SKY_MODE_QUALITY;
	RS::SkyMode sky_mode = sky->mode;

	if (sky_mode == RS::SKY_MODE_AUTOMATIC) {
		if (shader_data->uses_time || shader_data->uses_position) {
			update_single_frame = true;
			sky_mode = RS::SKY_MODE_REALTIME;
		} else if (shader_data->uses_light || shader_data->ubo_size > 0) {
			update_single_frame = false;
			sky_mode = RS::SKY_MODE_INCREMENTAL;
		} else {
			update_single_frame = true;
			sky_mode = RS::SKY_MODE_QUALITY;
		}
	}

	if (sky->processing_layer == 0 && sky_mode == RS::SKY_MODE_INCREMENTAL) {
		// On the first frame after creating sky, rebuild in single frame
		update_single_frame = true;
		sky_mode = RS::SKY_MODE_QUALITY;
	}

	int max_processing_layer = sky_use_cubemap_array ? sky->reflection.layers.size() : sky->reflection.layers[0].mipmaps.size();

	// Update radiance cubemap
	if (sky->reflection.dirty && (sky->processing_layer >= max_processing_layer || update_single_frame)) {
		static const Vector3 view_normals[6] = {
			Vector3(+1, 0, 0),
			Vector3(-1, 0, 0),
			Vector3(0, +1, 0),
			Vector3(0, -1, 0),
			Vector3(0, 0, +1),
			Vector3(0, 0, -1)
		};
		static const Vector3 view_up[6] = {
			Vector3(0, -1, 0),
			Vector3(0, -1, 0),
			Vector3(0, 0, +1),
			Vector3(0, 0, -1),
			Vector3(0, -1, 0),
			Vector3(0, -1, 0)
		};

		CameraMatrix cm;
		cm.set_perspective(90, 1, 0.01, 10.0);
		CameraMatrix correction;
		correction.set_depth_correction(true);
		cm = correction * cm;

		if (shader_data->uses_quarter_res) {
			RenderPipelineVertexFormatCacheRD *pipeline = &shader_data->pipelines[SKY_VERSION_CUBEMAP_QUARTER_RES];

			Vector<Color> clear_colors;
			clear_colors.push_back(Color(0.0, 0.0, 0.0));
			RD::DrawListID cubemap_draw_list;

			for (int i = 0; i < 6; i++) {
				Transform local_view;
				local_view.set_look_at(Vector3(0, 0, 0), view_normals[i], view_up[i]);
				RID texture_uniform_set = _get_sky_textures(sky, SKY_TEXTURE_SET_CUBEMAP_QUARTER_RES);

				cubemap_draw_list = RD::get_singleton()->draw_list_begin(sky->reflection.layers[0].mipmaps[2].framebuffers[i], RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
				storage->get_effects()->render_sky(cubemap_draw_list, time, sky->reflection.layers[0].mipmaps[2].framebuffers[i], sky_scene_state.uniform_set, sky_scene_state.fog_uniform_set, pipeline, material->uniform_set, texture_uniform_set, cm, local_view.basis, multiplier, p_transform.origin);
				RD::get_singleton()->draw_list_end();
			}
		}

		if (shader_data->uses_half_res) {
			RenderPipelineVertexFormatCacheRD *pipeline = &shader_data->pipelines[SKY_VERSION_CUBEMAP_HALF_RES];

			Vector<Color> clear_colors;
			clear_colors.push_back(Color(0.0, 0.0, 0.0));
			RD::DrawListID cubemap_draw_list;

			for (int i = 0; i < 6; i++) {
				Transform local_view;
				local_view.set_look_at(Vector3(0, 0, 0), view_normals[i], view_up[i]);
				RID texture_uniform_set = _get_sky_textures(sky, SKY_TEXTURE_SET_CUBEMAP_HALF_RES);

				cubemap_draw_list = RD::get_singleton()->draw_list_begin(sky->reflection.layers[0].mipmaps[1].framebuffers[i], RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
				storage->get_effects()->render_sky(cubemap_draw_list, time, sky->reflection.layers[0].mipmaps[1].framebuffers[i], sky_scene_state.uniform_set, sky_scene_state.fog_uniform_set, pipeline, material->uniform_set, texture_uniform_set, cm, local_view.basis, multiplier, p_transform.origin);
				RD::get_singleton()->draw_list_end();
			}
		}

		RD::DrawListID cubemap_draw_list;
		RenderPipelineVertexFormatCacheRD *pipeline = &shader_data->pipelines[SKY_VERSION_CUBEMAP];

		for (int i = 0; i < 6; i++) {
			Transform local_view;
			local_view.set_look_at(Vector3(0, 0, 0), view_normals[i], view_up[i]);
			RID texture_uniform_set = _get_sky_textures(sky, SKY_TEXTURE_SET_CUBEMAP);

			cubemap_draw_list = RD::get_singleton()->draw_list_begin(sky->reflection.layers[0].mipmaps[0].framebuffers[i], RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
			storage->get_effects()->render_sky(cubemap_draw_list, time, sky->reflection.layers[0].mipmaps[0].framebuffers[i], sky_scene_state.uniform_set, sky_scene_state.fog_uniform_set, pipeline, material->uniform_set, texture_uniform_set, cm, local_view.basis, multiplier, p_transform.origin);
			RD::get_singleton()->draw_list_end();
		}

		if (sky_mode == RS::SKY_MODE_REALTIME) {
			_create_reflection_fast_filter(sky->reflection, sky_use_cubemap_array);
			if (sky_use_cubemap_array) {
				_update_reflection_mipmaps(sky->reflection, 0, sky->reflection.layers.size());
			}
		} else {
			if (update_single_frame) {
				for (int i = 1; i < max_processing_layer; i++) {
					_create_reflection_importance_sample(sky->reflection, sky_use_cubemap_array, 10, i);
				}
				if (sky_use_cubemap_array) {
					_update_reflection_mipmaps(sky->reflection, 0, sky->reflection.layers.size());
				}
			} else {
				if (sky_use_cubemap_array) {
					// Multi-Frame so just update the first array level
					_update_reflection_mipmaps(sky->reflection, 0, 1);
				}
			}
			sky->processing_layer = 1;
		}

		sky->reflection.dirty = false;

	} else {
		if (sky_mode == RS::SKY_MODE_INCREMENTAL && sky->processing_layer < max_processing_layer) {
			_create_reflection_importance_sample(sky->reflection, sky_use_cubemap_array, 10, sky->processing_layer);

			if (sky_use_cubemap_array) {
				_update_reflection_mipmaps(sky->reflection, sky->processing_layer, sky->processing_layer + 1);
			}

			sky->processing_layer++;
		}
	}
}

/* SKY SHADER */

void RasterizerSceneRD::SkyShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();

	if (code == String()) {
		return; //just invalid, but no error
	}

	ShaderCompilerRD::GeneratedCode gen_code;
	ShaderCompilerRD::IdentifierActions actions;

	uses_time = false;
	uses_half_res = false;
	uses_quarter_res = false;
	uses_position = false;
	uses_light = false;

	actions.render_mode_flags["use_half_res_pass"] = &uses_half_res;
	actions.render_mode_flags["use_quarter_res_pass"] = &uses_quarter_res;

	actions.usage_flag_pointers["TIME"] = &uses_time;
	actions.usage_flag_pointers["POSITION"] = &uses_position;
	actions.usage_flag_pointers["LIGHT0_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_SIZE"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_SIZE"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_SIZE"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_SIZE"] = &uses_light;

	actions.uniforms = &uniforms;

	RasterizerSceneRD *scene_singleton = (RasterizerSceneRD *)RasterizerSceneRD::singleton;

	Error err = scene_singleton->sky_shader.compiler.compile(RS::SHADER_SKY, code, &actions, path, gen_code);

	ERR_FAIL_COND(err != OK);

	if (version.is_null()) {
		version = scene_singleton->sky_shader.shader.version_create();
	}

#if 0
	print_line("**compiling shader:");
	print_line("**defines:\n");
	for (int i = 0; i < gen_code.defines.size(); i++) {
		print_line(gen_code.defines[i]);
	}
	print_line("\n**uniforms:\n" + gen_code.uniforms);
	//	print_line("\n**vertex_globals:\n" + gen_code.vertex_global);
	//	print_line("\n**vertex_code:\n" + gen_code.vertex);
	print_line("\n**fragment_globals:\n" + gen_code.fragment_global);
	print_line("\n**fragment_code:\n" + gen_code.fragment);
	print_line("\n**light_code:\n" + gen_code.light);
#endif

	scene_singleton->sky_shader.shader.version_set_code(version, gen_code.uniforms, gen_code.vertex_global, gen_code.vertex, gen_code.fragment_global, gen_code.light, gen_code.fragment, gen_code.defines);
	ERR_FAIL_COND(!scene_singleton->sky_shader.shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	//update pipelines

	for (int i = 0; i < SKY_VERSION_MAX; i++) {
		RD::PipelineDepthStencilState depth_stencil_state;
		depth_stencil_state.enable_depth_test = true;
		depth_stencil_state.depth_compare_operator = RD::COMPARE_OP_LESS_OR_EQUAL;

		RID shader_variant = scene_singleton->sky_shader.shader.version_get_shader(version, i);
		pipelines[i].setup(shader_variant, RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), depth_stencil_state, RD::PipelineColorBlendState::create_disabled(), 0);
	}

	valid = true;
}

void RasterizerSceneRD::SkyShaderData::set_default_texture_param(const StringName &p_name, RID p_texture) {
	if (!p_texture.is_valid()) {
		default_texture_params.erase(p_name);
	} else {
		default_texture_params[p_name] = p_texture;
	}
}

void RasterizerSceneRD::SkyShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {
	Map<int, StringName> order;

	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = uniforms.front(); E; E = E->next()) {
		if (E->get().scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL || E->get().scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		if (E->get().texture_order >= 0) {
			order[E->get().texture_order + 100000] = E->key();
		} else {
			order[E->get().order] = E->key();
		}
	}

	for (Map<int, StringName>::Element *E = order.front(); E; E = E->next()) {
		PropertyInfo pi = ShaderLanguage::uniform_to_property_info(uniforms[E->get()]);
		pi.name = E->get();
		p_param_list->push_back(pi);
	}
}

void RasterizerSceneRD::SkyShaderData::get_instance_param_list(List<RasterizerStorage::InstanceShaderParam> *p_param_list) const {
	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = uniforms.front(); E; E = E->next()) {
		if (E->get().scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		RasterizerStorage::InstanceShaderParam p;
		p.info = ShaderLanguage::uniform_to_property_info(E->get());
		p.info.name = E->key(); //supply name
		p.index = E->get().instance_index;
		p.default_value = ShaderLanguage::constant_value_to_variant(E->get().default_value, E->get().type, E->get().hint);
		p_param_list->push_back(p);
	}
}

bool RasterizerSceneRD::SkyShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool RasterizerSceneRD::SkyShaderData::is_animated() const {
	return false;
}

bool RasterizerSceneRD::SkyShaderData::casts_shadows() const {
	return false;
}

Variant RasterizerSceneRD::SkyShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.hint);
	}
	return Variant();
}

RasterizerSceneRD::SkyShaderData::SkyShaderData() {
	valid = false;
}

RasterizerSceneRD::SkyShaderData::~SkyShaderData() {
	RasterizerSceneRD *scene_singleton = (RasterizerSceneRD *)RasterizerSceneRD::singleton;
	ERR_FAIL_COND(!scene_singleton);
	//pipeline variants will clear themselves if shader is gone
	if (version.is_valid()) {
		scene_singleton->sky_shader.shader.version_free(version);
	}
}

RasterizerStorageRD::ShaderData *RasterizerSceneRD::_create_sky_shader_func() {
	SkyShaderData *shader_data = memnew(SkyShaderData);
	return shader_data;
}

void RasterizerSceneRD::SkyMaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	RasterizerSceneRD *scene_singleton = (RasterizerSceneRD *)RasterizerSceneRD::singleton;

	uniform_set_updated = true;

	if ((uint32_t)ubo_data.size() != shader_data->ubo_size) {
		p_uniform_dirty = true;
		if (uniform_buffer.is_valid()) {
			RD::get_singleton()->free(uniform_buffer);
			uniform_buffer = RID();
		}

		ubo_data.resize(shader_data->ubo_size);
		if (ubo_data.size()) {
			uniform_buffer = RD::get_singleton()->uniform_buffer_create(ubo_data.size());
			memset(ubo_data.ptrw(), 0, ubo_data.size()); //clear
		}

		//clear previous uniform set
		if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			RD::get_singleton()->free(uniform_set);
			uniform_set = RID();
		}
	}

	//check whether buffer changed
	if (p_uniform_dirty && ubo_data.size()) {
		update_uniform_buffer(shader_data->uniforms, shader_data->ubo_offsets.ptr(), p_parameters, ubo_data.ptrw(), ubo_data.size(), false);
		RD::get_singleton()->buffer_update(uniform_buffer, 0, ubo_data.size(), ubo_data.ptrw());
	}

	uint32_t tex_uniform_count = shader_data->texture_uniforms.size();

	if ((uint32_t)texture_cache.size() != tex_uniform_count) {
		texture_cache.resize(tex_uniform_count);
		p_textures_dirty = true;

		//clear previous uniform set
		if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			RD::get_singleton()->free(uniform_set);
			uniform_set = RID();
		}
	}

	if (p_textures_dirty && tex_uniform_count) {
		update_textures(p_parameters, shader_data->default_texture_params, shader_data->texture_uniforms, texture_cache.ptrw(), true);
	}

	if (shader_data->ubo_size == 0 && shader_data->texture_uniforms.size() == 0) {
		// This material does not require an uniform set, so don't create it.
		return;
	}

	if (!p_textures_dirty && uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		//no reason to update uniform set, only UBO (or nothing) was needed to update
		return;
	}

	Vector<RD::Uniform> uniforms;

	{
		if (shader_data->ubo_size) {
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 0;
			u.ids.push_back(uniform_buffer);
			uniforms.push_back(u);
		}

		const RID *textures = texture_cache.ptrw();
		for (uint32_t i = 0; i < tex_uniform_count; i++) {
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1 + i;
			u.ids.push_back(textures[i]);
			uniforms.push_back(u);
		}
	}

	uniform_set = RD::get_singleton()->uniform_set_create(uniforms, scene_singleton->sky_shader.shader.version_get_shader(shader_data->version, 0), SKY_SET_MATERIAL);
}

RasterizerSceneRD::SkyMaterialData::~SkyMaterialData() {
	if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		RD::get_singleton()->free(uniform_set);
	}

	if (uniform_buffer.is_valid()) {
		RD::get_singleton()->free(uniform_buffer);
	}
}

RasterizerStorageRD::MaterialData *RasterizerSceneRD::_create_sky_material_func(SkyShaderData *p_shader) {
	SkyMaterialData *material_data = memnew(SkyMaterialData);
	material_data->shader_data = p_shader;
	material_data->last_frame = false;
	//update will happen later anyway so do nothing.
	return material_data;
}

RID RasterizerSceneRD::environment_create() {
	return environment_owner.make_rid(Environment());
}

void RasterizerSceneRD::environment_set_background(RID p_env, RS::EnvironmentBG p_bg) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->background = p_bg;
}

void RasterizerSceneRD::environment_set_sky(RID p_env, RID p_sky) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->sky = p_sky;
}

void RasterizerSceneRD::environment_set_sky_custom_fov(RID p_env, float p_scale) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->sky_custom_fov = p_scale;
}

void RasterizerSceneRD::environment_set_sky_orientation(RID p_env, const Basis &p_orientation) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->sky_orientation = p_orientation;
}

void RasterizerSceneRD::environment_set_bg_color(RID p_env, const Color &p_color) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->bg_color = p_color;
}

void RasterizerSceneRD::environment_set_bg_energy(RID p_env, float p_energy) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->bg_energy = p_energy;
}

void RasterizerSceneRD::environment_set_canvas_max_layer(RID p_env, int p_max_layer) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->canvas_max_layer = p_max_layer;
}

void RasterizerSceneRD::environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient, float p_energy, float p_sky_contribution, RS::EnvironmentReflectionSource p_reflection_source, const Color &p_ao_color) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->ambient_light = p_color;
	env->ambient_source = p_ambient;
	env->ambient_light_energy = p_energy;
	env->ambient_sky_contribution = p_sky_contribution;
	env->reflection_source = p_reflection_source;
	env->ao_color = p_ao_color;
}

RS::EnvironmentBG RasterizerSceneRD::environment_get_background(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, RS::ENV_BG_MAX);
	return env->background;
}

RID RasterizerSceneRD::environment_get_sky(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, RID());
	return env->sky;
}

float RasterizerSceneRD::environment_get_sky_custom_fov(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->sky_custom_fov;
}

Basis RasterizerSceneRD::environment_get_sky_orientation(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, Basis());
	return env->sky_orientation;
}

Color RasterizerSceneRD::environment_get_bg_color(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, Color());
	return env->bg_color;
}

float RasterizerSceneRD::environment_get_bg_energy(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->bg_energy;
}

int RasterizerSceneRD::environment_get_canvas_max_layer(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->canvas_max_layer;
}

Color RasterizerSceneRD::environment_get_ambient_light_color(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, Color());
	return env->ambient_light;
}

RS::EnvironmentAmbientSource RasterizerSceneRD::environment_get_ambient_source(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, RS::ENV_AMBIENT_SOURCE_BG);
	return env->ambient_source;
}

float RasterizerSceneRD::environment_get_ambient_light_energy(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->ambient_light_energy;
}

float RasterizerSceneRD::environment_get_ambient_sky_contribution(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->ambient_sky_contribution;
}

RS::EnvironmentReflectionSource RasterizerSceneRD::environment_get_reflection_source(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, RS::ENV_REFLECTION_SOURCE_DISABLED);
	return env->reflection_source;
}

Color RasterizerSceneRD::environment_get_ao_color(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, Color());
	return env->ao_color;
}

void RasterizerSceneRD::environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->exposure = p_exposure;
	env->tone_mapper = p_tone_mapper;
	if (!env->auto_exposure && p_auto_exposure) {
		env->auto_exposure_version = ++auto_exposure_counter;
	}
	env->auto_exposure = p_auto_exposure;
	env->white = p_white;
	env->min_luminance = p_min_luminance;
	env->max_luminance = p_max_luminance;
	env->auto_exp_speed = p_auto_exp_speed;
	env->auto_exp_scale = p_auto_exp_scale;
}

void RasterizerSceneRD::environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	ERR_FAIL_COND_MSG(p_levels.size() != 7, "Size of array of glow levels must be 7");
	env->glow_enabled = p_enable;
	env->glow_levels = p_levels;
	env->glow_intensity = p_intensity;
	env->glow_strength = p_strength;
	env->glow_mix = p_mix;
	env->glow_bloom = p_bloom_threshold;
	env->glow_blend_mode = p_blend_mode;
	env->glow_hdr_bleed_threshold = p_hdr_bleed_threshold;
	env->glow_hdr_bleed_scale = p_hdr_bleed_scale;
	env->glow_hdr_luminance_cap = p_hdr_luminance_cap;
}

void RasterizerSceneRD::environment_glow_set_use_bicubic_upscale(bool p_enable) {
	glow_bicubic_upscale = p_enable;
}

void RasterizerSceneRD::environment_glow_set_use_high_quality(bool p_enable) {
	glow_high_quality = p_enable;
}

void RasterizerSceneRD::environment_set_sdfgi(RID p_env, bool p_enable, RS::EnvironmentSDFGICascades p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, bool p_use_multibounce, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->sdfgi_enabled = p_enable;
	env->sdfgi_cascades = p_cascades;
	env->sdfgi_min_cell_size = p_min_cell_size;
	env->sdfgi_use_occlusion = p_use_occlusion;
	env->sdfgi_use_multibounce = p_use_multibounce;
	env->sdfgi_read_sky_light = p_read_sky;
	env->sdfgi_energy = p_energy;
	env->sdfgi_normal_bias = p_normal_bias;
	env->sdfgi_probe_bias = p_probe_bias;
	env->sdfgi_y_scale = p_y_scale;
}

void RasterizerSceneRD::environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_fog_aerial_perspective) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->fog_enabled = p_enable;
	env->fog_light_color = p_light_color;
	env->fog_light_energy = p_light_energy;
	env->fog_sun_scatter = p_sun_scatter;
	env->fog_density = p_density;
	env->fog_height = p_height;
	env->fog_height_density = p_height_density;
	env->fog_aerial_perspective = p_fog_aerial_perspective;
}

bool RasterizerSceneRD::environment_is_fog_enabled(RID p_env) const {
	const Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, false);

	return env->fog_enabled;
}
Color RasterizerSceneRD::environment_get_fog_light_color(RID p_env) const {
	const Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, Color());
	return env->fog_light_color;
}
float RasterizerSceneRD::environment_get_fog_light_energy(RID p_env) const {
	const Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->fog_light_energy;
}
float RasterizerSceneRD::environment_get_fog_sun_scatter(RID p_env) const {
	const Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->fog_sun_scatter;
}
float RasterizerSceneRD::environment_get_fog_density(RID p_env) const {
	const Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->fog_density;
}
float RasterizerSceneRD::environment_get_fog_height(RID p_env) const {
	const Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);

	return env->fog_height;
}
float RasterizerSceneRD::environment_get_fog_height_density(RID p_env) const {
	const Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->fog_height_density;
}

float RasterizerSceneRD::environment_get_fog_aerial_perspective(RID p_env) const {
	const Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->fog_aerial_perspective;
}

void RasterizerSceneRD::environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_light, float p_light_energy, float p_length, float p_detail_spread, float p_gi_inject, RenderingServer::EnvVolumetricFogShadowFilter p_shadow_filter) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->volumetric_fog_enabled = p_enable;
	env->volumetric_fog_density = p_density;
	env->volumetric_fog_light = p_light;
	env->volumetric_fog_light_energy = p_light_energy;
	env->volumetric_fog_length = p_length;
	env->volumetric_fog_detail_spread = p_detail_spread;
	env->volumetric_fog_shadow_filter = p_shadow_filter;
	env->volumetric_fog_gi_inject = p_gi_inject;
}

void RasterizerSceneRD::environment_set_volumetric_fog_volume_size(int p_size, int p_depth) {
	volumetric_fog_size = p_size;
	volumetric_fog_depth = p_depth;
}

void RasterizerSceneRD::environment_set_volumetric_fog_filter_active(bool p_enable) {
	volumetric_fog_filter_active = p_enable;
}
void RasterizerSceneRD::environment_set_volumetric_fog_directional_shadow_shrink_size(int p_shrink_size) {
	p_shrink_size = nearest_power_of_2_templated(p_shrink_size);
	if (volumetric_fog_directional_shadow_shrink == (uint32_t)p_shrink_size) {
		return;
	}

	_clear_shadow_shrink_stages(directional_shadow.shrink_stages);
}
void RasterizerSceneRD::environment_set_volumetric_fog_positional_shadow_shrink_size(int p_shrink_size) {
	p_shrink_size = nearest_power_of_2_templated(p_shrink_size);
	if (volumetric_fog_positional_shadow_shrink == (uint32_t)p_shrink_size) {
		return;
	}

	for (uint32_t i = 0; i < shadow_atlas_owner.get_rid_count(); i++) {
		ShadowAtlas *sa = shadow_atlas_owner.get_ptr_by_index(i);
		_clear_shadow_shrink_stages(sa->shrink_stages);
	}
}

void RasterizerSceneRD::environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count) {
	sdfgi_ray_count = p_ray_count;
}

void RasterizerSceneRD::environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames) {
	sdfgi_frames_to_converge = p_frames;
}

void RasterizerSceneRD::environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->ssr_enabled = p_enable;
	env->ssr_max_steps = p_max_steps;
	env->ssr_fade_in = p_fade_int;
	env->ssr_fade_out = p_fade_out;
	env->ssr_depth_tolerance = p_depth_tolerance;
}

void RasterizerSceneRD::environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) {
	ssr_roughness_quality = p_quality;
}

RS::EnvironmentSSRRoughnessQuality RasterizerSceneRD::environment_get_ssr_roughness_quality() const {
	return ssr_roughness_quality;
}

void RasterizerSceneRD::environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_bias, float p_light_affect, float p_ao_channel_affect, RS::EnvironmentSSAOBlur p_blur, float p_bilateral_sharpness) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->ssao_enabled = p_enable;
	env->ssao_radius = p_radius;
	env->ssao_intensity = p_intensity;
	env->ssao_bias = p_bias;
	env->ssao_direct_light_affect = p_light_affect;
	env->ssao_ao_channel_affect = p_ao_channel_affect;
	env->ssao_blur = p_blur;
}

void RasterizerSceneRD::environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size) {
	ssao_quality = p_quality;
	ssao_half_size = p_half_size;
}

bool RasterizerSceneRD::environment_is_ssao_enabled(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->ssao_enabled;
}

float RasterizerSceneRD::environment_get_ssao_ao_affect(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->ssao_ao_channel_affect;
}

float RasterizerSceneRD::environment_get_ssao_light_affect(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->ssao_direct_light_affect;
}

bool RasterizerSceneRD::environment_is_ssr_enabled(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->ssr_enabled;
}
bool RasterizerSceneRD::environment_is_sdfgi_enabled(RID p_env) const {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->sdfgi_enabled;
}

bool RasterizerSceneRD::is_environment(RID p_env) const {
	return environment_owner.owns(p_env);
}

Ref<Image> RasterizerSceneRD::environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, Ref<Image>());

	if (env->background == RS::ENV_BG_CAMERA_FEED || env->background == RS::ENV_BG_CANVAS || env->background == RS::ENV_BG_KEEP) {
		return Ref<Image>(); //nothing to bake
	}

	if (env->background == RS::ENV_BG_CLEAR_COLOR || env->background == RS::ENV_BG_COLOR) {
		Color color;
		if (env->background == RS::ENV_BG_CLEAR_COLOR) {
			color = storage->get_default_clear_color();
		} else {
			color = env->bg_color;
		}
		color.r *= env->bg_energy;
		color.g *= env->bg_energy;
		color.b *= env->bg_energy;

		Ref<Image> ret;
		ret.instance();
		ret->create(p_size.width, p_size.height, false, Image::FORMAT_RGBAF);
		for (int i = 0; i < p_size.width; i++) {
			for (int j = 0; j < p_size.height; j++) {
				ret->set_pixel(i, j, color);
			}
		}
		return ret;
	}

	if (env->background == RS::ENV_BG_SKY && env->sky.is_valid()) {
		return sky_bake_panorama(env->sky, env->bg_energy, p_bake_irradiance, p_size);
	}

	return Ref<Image>();
}

////////////////////////////////////////////////////////////

RID RasterizerSceneRD::reflection_atlas_create() {
	ReflectionAtlas ra;
	ra.count = GLOBAL_GET("rendering/quality/reflection_atlas/reflection_count");
	ra.size = GLOBAL_GET("rendering/quality/reflection_atlas/reflection_size");

	return reflection_atlas_owner.make_rid(ra);
}

void RasterizerSceneRD::reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) {
	ReflectionAtlas *ra = reflection_atlas_owner.getornull(p_ref_atlas);
	ERR_FAIL_COND(!ra);

	if (ra->size == p_reflection_size && ra->count == p_reflection_count) {
		return; //no changes
	}

	ra->size = p_reflection_size;
	ra->count = p_reflection_count;

	if (ra->reflection.is_valid()) {
		//clear and invalidate everything
		RD::get_singleton()->free(ra->reflection);
		ra->reflection = RID();
		RD::get_singleton()->free(ra->depth_buffer);
		ra->depth_buffer = RID();

		for (int i = 0; i < ra->reflections.size(); i++) {
			_clear_reflection_data(ra->reflections.write[i].data);
			if (ra->reflections[i].owner.is_null()) {
				continue;
			}
			reflection_probe_release_atlas_index(ra->reflections[i].owner);
			//rp->atlasindex clear
		}

		ra->reflections.clear();
	}
}

////////////////////////
RID RasterizerSceneRD::reflection_probe_instance_create(RID p_probe) {
	ReflectionProbeInstance rpi;
	rpi.probe = p_probe;
	return reflection_probe_instance_owner.make_rid(rpi);
}

void RasterizerSceneRD::reflection_probe_instance_set_transform(RID p_instance, const Transform &p_transform) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND(!rpi);

	rpi->transform = p_transform;
	rpi->dirty = true;
}

void RasterizerSceneRD::reflection_probe_release_atlas_index(RID p_instance) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND(!rpi);

	if (rpi->atlas.is_null()) {
		return; //nothing to release
	}
	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(rpi->atlas);
	ERR_FAIL_COND(!atlas);
	ERR_FAIL_INDEX(rpi->atlas_index, atlas->reflections.size());
	atlas->reflections.write[rpi->atlas_index].owner = RID();
	rpi->atlas_index = -1;
	rpi->atlas = RID();
}

bool RasterizerSceneRD::reflection_probe_instance_needs_redraw(RID p_instance) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	if (rpi->rendering) {
		return false;
	}

	if (rpi->dirty) {
		return true;
	}

	if (storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS) {
		return true;
	}

	return rpi->atlas_index == -1;
}

bool RasterizerSceneRD::reflection_probe_instance_has_reflection(RID p_instance) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	return rpi->atlas.is_valid();
}

bool RasterizerSceneRD::reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) {
	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(p_reflection_atlas);

	ERR_FAIL_COND_V(!atlas, false);

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	if (storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS && atlas->reflection.is_valid() && atlas->size != 256) {
		WARN_PRINT("ReflectionProbes set to UPDATE_ALWAYS must have an atlas size of 256. Please update the atlas size in the ProjectSettings.");
		reflection_atlas_set_size(p_reflection_atlas, 256, atlas->count);
	}

	if (storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS && atlas->reflection.is_valid() && atlas->reflections[0].data.layers[0].mipmaps.size() != 8) {
		// Invalidate reflection atlas, need to regenerate
		RD::get_singleton()->free(atlas->reflection);
		atlas->reflection = RID();

		for (int i = 0; i < atlas->reflections.size(); i++) {
			if (atlas->reflections[i].owner.is_null()) {
				continue;
			}
			reflection_probe_release_atlas_index(atlas->reflections[i].owner);
		}

		atlas->reflections.clear();
	}

	if (atlas->reflection.is_null()) {
		int mipmaps = MIN(roughness_layers, Image::get_image_required_mipmaps(atlas->size, atlas->size, Image::FORMAT_RGBAH) + 1);
		mipmaps = storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS ? 8 : mipmaps; // always use 8 mipmaps with real time filtering
		{
			//reflection atlas was unused, create:
			RD::TextureFormat tf;
			tf.array_layers = 6 * atlas->count;
			tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			tf.type = RD::TEXTURE_TYPE_CUBE_ARRAY;
			tf.mipmaps = mipmaps;
			tf.width = atlas->size;
			tf.height = atlas->size;
			tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

			atlas->reflection = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}
		{
			RD::TextureFormat tf;
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
			tf.width = atlas->size;
			tf.height = atlas->size;
			tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
			atlas->depth_buffer = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}
		atlas->reflections.resize(atlas->count);
		for (int i = 0; i < atlas->count; i++) {
			_update_reflection_data(atlas->reflections.write[i].data, atlas->size, mipmaps, false, atlas->reflection, i * 6, storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS);
			for (int j = 0; j < 6; j++) {
				Vector<RID> fb;
				fb.push_back(atlas->reflections.write[i].data.layers[0].mipmaps[0].views[j]);
				fb.push_back(atlas->depth_buffer);
				atlas->reflections.write[i].fbs[j] = RD::get_singleton()->framebuffer_create(fb);
			}
		}

		Vector<RID> fb;
		fb.push_back(atlas->depth_buffer);
		atlas->depth_fb = RD::get_singleton()->framebuffer_create(fb);
	}

	if (rpi->atlas_index == -1) {
		for (int i = 0; i < atlas->reflections.size(); i++) {
			if (atlas->reflections[i].owner.is_null()) {
				rpi->atlas_index = i;
				break;
			}
		}
		//find the one used last
		if (rpi->atlas_index == -1) {
			//everything is in use, find the one least used via LRU
			uint64_t pass_min = 0;

			for (int i = 0; i < atlas->reflections.size(); i++) {
				ReflectionProbeInstance *rpi2 = reflection_probe_instance_owner.getornull(atlas->reflections[i].owner);
				if (rpi2->last_pass < pass_min) {
					pass_min = rpi2->last_pass;
					rpi->atlas_index = i;
				}
			}
		}
	}

	rpi->atlas = p_reflection_atlas;
	rpi->rendering = true;
	rpi->dirty = false;
	rpi->processing_layer = 1;
	rpi->processing_side = 0;

	return true;
}

bool RasterizerSceneRD::reflection_probe_instance_postprocess_step(RID p_instance) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);
	ERR_FAIL_COND_V(!rpi->rendering, false);
	ERR_FAIL_COND_V(rpi->atlas.is_null(), false);

	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(rpi->atlas);
	if (!atlas || rpi->atlas_index == -1) {
		//does not belong to an atlas anymore, cancel (was removed from atlas or atlas changed while rendering)
		rpi->rendering = false;
		return false;
	}

	if (storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS) {
		// Using real time reflections, all roughness is done in one step
		_create_reflection_fast_filter(atlas->reflections.write[rpi->atlas_index].data, false);
		rpi->rendering = false;
		rpi->processing_side = 0;
		rpi->processing_layer = 1;
		return true;
	}

	if (rpi->processing_layer > 1) {
		_create_reflection_importance_sample(atlas->reflections.write[rpi->atlas_index].data, false, 10, rpi->processing_layer);
		rpi->processing_layer++;
		if (rpi->processing_layer == atlas->reflections[rpi->atlas_index].data.layers[0].mipmaps.size()) {
			rpi->rendering = false;
			rpi->processing_side = 0;
			rpi->processing_layer = 1;
			return true;
		}
		return false;

	} else {
		_create_reflection_importance_sample(atlas->reflections.write[rpi->atlas_index].data, false, rpi->processing_side, rpi->processing_layer);
	}

	rpi->processing_side++;
	if (rpi->processing_side == 6) {
		rpi->processing_side = 0;
		rpi->processing_layer++;
	}

	return false;
}

uint32_t RasterizerSceneRD::reflection_probe_instance_get_resolution(RID p_instance) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, 0);

	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(rpi->atlas);
	ERR_FAIL_COND_V(!atlas, 0);
	return atlas->size;
}

RID RasterizerSceneRD::reflection_probe_instance_get_framebuffer(RID p_instance, int p_index) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, RID());
	ERR_FAIL_INDEX_V(p_index, 6, RID());

	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(rpi->atlas);
	ERR_FAIL_COND_V(!atlas, RID());
	return atlas->reflections[rpi->atlas_index].fbs[p_index];
}

RID RasterizerSceneRD::reflection_probe_instance_get_depth_framebuffer(RID p_instance, int p_index) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, RID());
	ERR_FAIL_INDEX_V(p_index, 6, RID());

	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(rpi->atlas);
	ERR_FAIL_COND_V(!atlas, RID());
	return atlas->depth_fb;
}

///////////////////////////////////////////////////////////

RID RasterizerSceneRD::shadow_atlas_create() {
	return shadow_atlas_owner.make_rid(ShadowAtlas());
}

void RasterizerSceneRD::shadow_atlas_set_size(RID p_atlas, int p_size) {
	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_atlas);
	ERR_FAIL_COND(!shadow_atlas);
	ERR_FAIL_COND(p_size < 0);
	p_size = next_power_of_2(p_size);

	if (p_size == shadow_atlas->size) {
		return;
	}

	// erasing atlas
	if (shadow_atlas->depth.is_valid()) {
		RD::get_singleton()->free(shadow_atlas->depth);
		shadow_atlas->depth = RID();
		_clear_shadow_shrink_stages(shadow_atlas->shrink_stages);
	}
	for (int i = 0; i < 4; i++) {
		//clear subdivisions
		shadow_atlas->quadrants[i].shadows.resize(0);
		shadow_atlas->quadrants[i].shadows.resize(1 << shadow_atlas->quadrants[i].subdivision);
	}

	//erase shadow atlas reference from lights
	for (Map<RID, uint32_t>::Element *E = shadow_atlas->shadow_owners.front(); E; E = E->next()) {
		LightInstance *li = light_instance_owner.getornull(E->key());
		ERR_CONTINUE(!li);
		li->shadow_atlases.erase(p_atlas);
	}

	//clear owners
	shadow_atlas->shadow_owners.clear();

	shadow_atlas->size = p_size;

	if (shadow_atlas->size) {
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		tf.width = shadow_atlas->size;
		tf.height = shadow_atlas->size;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

		shadow_atlas->depth = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}
}

void RasterizerSceneRD::shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) {
	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_atlas);
	ERR_FAIL_COND(!shadow_atlas);
	ERR_FAIL_INDEX(p_quadrant, 4);
	ERR_FAIL_INDEX(p_subdivision, 16384);

	uint32_t subdiv = next_power_of_2(p_subdivision);
	if (subdiv & 0xaaaaaaaa) { //sqrt(subdiv) must be integer
		subdiv <<= 1;
	}

	subdiv = int(Math::sqrt((float)subdiv));

	//obtain the number that will be x*x

	if (shadow_atlas->quadrants[p_quadrant].subdivision == subdiv) {
		return;
	}

	//erase all data from quadrant
	for (int i = 0; i < shadow_atlas->quadrants[p_quadrant].shadows.size(); i++) {
		if (shadow_atlas->quadrants[p_quadrant].shadows[i].owner.is_valid()) {
			shadow_atlas->shadow_owners.erase(shadow_atlas->quadrants[p_quadrant].shadows[i].owner);
			LightInstance *li = light_instance_owner.getornull(shadow_atlas->quadrants[p_quadrant].shadows[i].owner);
			ERR_CONTINUE(!li);
			li->shadow_atlases.erase(p_atlas);
		}
	}

	shadow_atlas->quadrants[p_quadrant].shadows.resize(0);
	shadow_atlas->quadrants[p_quadrant].shadows.resize(subdiv * subdiv);
	shadow_atlas->quadrants[p_quadrant].subdivision = subdiv;

	//cache the smallest subdiv (for faster allocation in light update)

	shadow_atlas->smallest_subdiv = 1 << 30;

	for (int i = 0; i < 4; i++) {
		if (shadow_atlas->quadrants[i].subdivision) {
			shadow_atlas->smallest_subdiv = MIN(shadow_atlas->smallest_subdiv, shadow_atlas->quadrants[i].subdivision);
		}
	}

	if (shadow_atlas->smallest_subdiv == 1 << 30) {
		shadow_atlas->smallest_subdiv = 0;
	}

	//resort the size orders, simple bublesort for 4 elements..

	int swaps = 0;
	do {
		swaps = 0;

		for (int i = 0; i < 3; i++) {
			if (shadow_atlas->quadrants[shadow_atlas->size_order[i]].subdivision < shadow_atlas->quadrants[shadow_atlas->size_order[i + 1]].subdivision) {
				SWAP(shadow_atlas->size_order[i], shadow_atlas->size_order[i + 1]);
				swaps++;
			}
		}
	} while (swaps > 0);
}

bool RasterizerSceneRD::_shadow_atlas_find_shadow(ShadowAtlas *shadow_atlas, int *p_in_quadrants, int p_quadrant_count, int p_current_subdiv, uint64_t p_tick, int &r_quadrant, int &r_shadow) {
	for (int i = p_quadrant_count - 1; i >= 0; i--) {
		int qidx = p_in_quadrants[i];

		if (shadow_atlas->quadrants[qidx].subdivision == (uint32_t)p_current_subdiv) {
			return false;
		}

		//look for an empty space
		int sc = shadow_atlas->quadrants[qidx].shadows.size();
		ShadowAtlas::Quadrant::Shadow *sarr = shadow_atlas->quadrants[qidx].shadows.ptrw();

		int found_free_idx = -1; //found a free one
		int found_used_idx = -1; //found existing one, must steal it
		uint64_t min_pass = 0; // pass of the existing one, try to use the least recently used one (LRU fashion)

		for (int j = 0; j < sc; j++) {
			if (!sarr[j].owner.is_valid()) {
				found_free_idx = j;
				break;
			}

			LightInstance *sli = light_instance_owner.getornull(sarr[j].owner);
			ERR_CONTINUE(!sli);

			if (sli->last_scene_pass != scene_pass) {
				//was just allocated, don't kill it so soon, wait a bit..
				if (p_tick - sarr[j].alloc_tick < shadow_atlas_realloc_tolerance_msec) {
					continue;
				}

				if (found_used_idx == -1 || sli->last_scene_pass < min_pass) {
					found_used_idx = j;
					min_pass = sli->last_scene_pass;
				}
			}
		}

		if (found_free_idx == -1 && found_used_idx == -1) {
			continue; //nothing found
		}

		if (found_free_idx == -1 && found_used_idx != -1) {
			found_free_idx = found_used_idx;
		}

		r_quadrant = qidx;
		r_shadow = found_free_idx;

		return true;
	}

	return false;
}

bool RasterizerSceneRD::shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) {
	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_atlas);
	ERR_FAIL_COND_V(!shadow_atlas, false);

	LightInstance *li = light_instance_owner.getornull(p_light_intance);
	ERR_FAIL_COND_V(!li, false);

	if (shadow_atlas->size == 0 || shadow_atlas->smallest_subdiv == 0) {
		return false;
	}

	uint32_t quad_size = shadow_atlas->size >> 1;
	int desired_fit = MIN(quad_size / shadow_atlas->smallest_subdiv, next_power_of_2(quad_size * p_coverage));

	int valid_quadrants[4];
	int valid_quadrant_count = 0;
	int best_size = -1; //best size found
	int best_subdiv = -1; //subdiv for the best size

	//find the quadrants this fits into, and the best possible size it can fit into
	for (int i = 0; i < 4; i++) {
		int q = shadow_atlas->size_order[i];
		int sd = shadow_atlas->quadrants[q].subdivision;
		if (sd == 0) {
			continue; //unused
		}

		int max_fit = quad_size / sd;

		if (best_size != -1 && max_fit > best_size) {
			break; //too large
		}

		valid_quadrants[valid_quadrant_count++] = q;
		best_subdiv = sd;

		if (max_fit >= desired_fit) {
			best_size = max_fit;
		}
	}

	ERR_FAIL_COND_V(valid_quadrant_count == 0, false);

	uint64_t tick = OS::get_singleton()->get_ticks_msec();

	//see if it already exists

	if (shadow_atlas->shadow_owners.has(p_light_intance)) {
		//it does!
		uint32_t key = shadow_atlas->shadow_owners[p_light_intance];
		uint32_t q = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
		uint32_t s = key & ShadowAtlas::SHADOW_INDEX_MASK;

		bool should_realloc = shadow_atlas->quadrants[q].subdivision != (uint32_t)best_subdiv && (shadow_atlas->quadrants[q].shadows[s].alloc_tick - tick > shadow_atlas_realloc_tolerance_msec);
		bool should_redraw = shadow_atlas->quadrants[q].shadows[s].version != p_light_version;

		if (!should_realloc) {
			shadow_atlas->quadrants[q].shadows.write[s].version = p_light_version;
			//already existing, see if it should redraw or it's just OK
			return should_redraw;
		}

		int new_quadrant, new_shadow;

		//find a better place
		if (_shadow_atlas_find_shadow(shadow_atlas, valid_quadrants, valid_quadrant_count, shadow_atlas->quadrants[q].subdivision, tick, new_quadrant, new_shadow)) {
			//found a better place!
			ShadowAtlas::Quadrant::Shadow *sh = &shadow_atlas->quadrants[new_quadrant].shadows.write[new_shadow];
			if (sh->owner.is_valid()) {
				//is taken, but is invalid, erasing it
				shadow_atlas->shadow_owners.erase(sh->owner);
				LightInstance *sli = light_instance_owner.getornull(sh->owner);
				sli->shadow_atlases.erase(p_atlas);
			}

			//erase previous
			shadow_atlas->quadrants[q].shadows.write[s].version = 0;
			shadow_atlas->quadrants[q].shadows.write[s].owner = RID();

			sh->owner = p_light_intance;
			sh->alloc_tick = tick;
			sh->version = p_light_version;
			li->shadow_atlases.insert(p_atlas);

			//make new key
			key = new_quadrant << ShadowAtlas::QUADRANT_SHIFT;
			key |= new_shadow;
			//update it in map
			shadow_atlas->shadow_owners[p_light_intance] = key;
			//make it dirty, as it should redraw anyway
			return true;
		}

		//no better place for this shadow found, keep current

		//already existing, see if it should redraw or it's just OK

		shadow_atlas->quadrants[q].shadows.write[s].version = p_light_version;

		return should_redraw;
	}

	int new_quadrant, new_shadow;

	//find a better place
	if (_shadow_atlas_find_shadow(shadow_atlas, valid_quadrants, valid_quadrant_count, -1, tick, new_quadrant, new_shadow)) {
		//found a better place!
		ShadowAtlas::Quadrant::Shadow *sh = &shadow_atlas->quadrants[new_quadrant].shadows.write[new_shadow];
		if (sh->owner.is_valid()) {
			//is taken, but is invalid, erasing it
			shadow_atlas->shadow_owners.erase(sh->owner);
			LightInstance *sli = light_instance_owner.getornull(sh->owner);
			sli->shadow_atlases.erase(p_atlas);
		}

		sh->owner = p_light_intance;
		sh->alloc_tick = tick;
		sh->version = p_light_version;
		li->shadow_atlases.insert(p_atlas);

		//make new key
		uint32_t key = new_quadrant << ShadowAtlas::QUADRANT_SHIFT;
		key |= new_shadow;
		//update it in map
		shadow_atlas->shadow_owners[p_light_intance] = key;
		//make it dirty, as it should redraw anyway

		return true;
	}

	//no place to allocate this light, apologies

	return false;
}

void RasterizerSceneRD::directional_shadow_atlas_set_size(int p_size) {
	p_size = nearest_power_of_2_templated(p_size);

	if (directional_shadow.size == p_size) {
		return;
	}

	directional_shadow.size = p_size;

	if (directional_shadow.depth.is_valid()) {
		RD::get_singleton()->free(directional_shadow.depth);
		_clear_shadow_shrink_stages(directional_shadow.shrink_stages);
		directional_shadow.depth = RID();
	}

	if (p_size > 0) {
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		tf.width = p_size;
		tf.height = p_size;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

		directional_shadow.depth = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	_base_uniforms_changed();
}

void RasterizerSceneRD::set_directional_shadow_count(int p_count) {
	directional_shadow.light_count = p_count;
	directional_shadow.current_light = 0;
}

static Rect2i _get_directional_shadow_rect(int p_size, int p_shadow_count, int p_shadow_index) {
	int split_h = 1;
	int split_v = 1;

	while (split_h * split_v < p_shadow_count) {
		if (split_h == split_v) {
			split_h <<= 1;
		} else {
			split_v <<= 1;
		}
	}

	Rect2i rect(0, 0, p_size, p_size);
	rect.size.width /= split_h;
	rect.size.height /= split_v;

	rect.position.x = rect.size.width * (p_shadow_index % split_h);
	rect.position.y = rect.size.height * (p_shadow_index / split_h);

	return rect;
}

int RasterizerSceneRD::get_directional_light_shadow_size(RID p_light_intance) {
	ERR_FAIL_COND_V(directional_shadow.light_count == 0, 0);

	Rect2i r = _get_directional_shadow_rect(directional_shadow.size, directional_shadow.light_count, 0);

	LightInstance *light_instance = light_instance_owner.getornull(p_light_intance);
	ERR_FAIL_COND_V(!light_instance, 0);

	switch (storage->light_directional_get_shadow_mode(light_instance->light)) {
		case RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL:
			break; //none
		case RS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS:
			r.size.height /= 2;
			break;
		case RS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS:
			r.size /= 2;
			break;
	}

	return MAX(r.size.width, r.size.height);
}

//////////////////////////////////////////////////

RID RasterizerSceneRD::camera_effects_create() {
	return camera_effects_owner.make_rid(CameraEffects());
}

void RasterizerSceneRD::camera_effects_set_dof_blur_quality(RS::DOFBlurQuality p_quality, bool p_use_jitter) {
	dof_blur_quality = p_quality;
	dof_blur_use_jitter = p_use_jitter;
}

void RasterizerSceneRD::camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape p_shape) {
	dof_blur_bokeh_shape = p_shape;
}

void RasterizerSceneRD::camera_effects_set_dof_blur(RID p_camera_effects, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount) {
	CameraEffects *camfx = camera_effects_owner.getornull(p_camera_effects);
	ERR_FAIL_COND(!camfx);

	camfx->dof_blur_far_enabled = p_far_enable;
	camfx->dof_blur_far_distance = p_far_distance;
	camfx->dof_blur_far_transition = p_far_transition;

	camfx->dof_blur_near_enabled = p_near_enable;
	camfx->dof_blur_near_distance = p_near_distance;
	camfx->dof_blur_near_transition = p_near_transition;

	camfx->dof_blur_amount = p_amount;
}

void RasterizerSceneRD::camera_effects_set_custom_exposure(RID p_camera_effects, bool p_enable, float p_exposure) {
	CameraEffects *camfx = camera_effects_owner.getornull(p_camera_effects);
	ERR_FAIL_COND(!camfx);

	camfx->override_exposure_enabled = p_enable;
	camfx->override_exposure = p_exposure;
}

RID RasterizerSceneRD::light_instance_create(RID p_light) {
	RID li = light_instance_owner.make_rid(LightInstance());

	LightInstance *light_instance = light_instance_owner.getornull(li);

	light_instance->self = li;
	light_instance->light = p_light;
	light_instance->light_type = storage->light_get_type(p_light);

	return li;
}

void RasterizerSceneRD::light_instance_set_transform(RID p_light_instance, const Transform &p_transform) {
	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->transform = p_transform;
}

void RasterizerSceneRD::light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) {
	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->aabb = p_aabb;
}

void RasterizerSceneRD::light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale, float p_range_begin, const Vector2 &p_uv_scale) {
	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	if (storage->light_get_type(light_instance->light) != RS::LIGHT_DIRECTIONAL) {
		p_pass = 0;
	}

	ERR_FAIL_INDEX(p_pass, 4);

	light_instance->shadow_transform[p_pass].camera = p_projection;
	light_instance->shadow_transform[p_pass].transform = p_transform;
	light_instance->shadow_transform[p_pass].farplane = p_far;
	light_instance->shadow_transform[p_pass].split = p_split;
	light_instance->shadow_transform[p_pass].bias_scale = p_bias_scale;
	light_instance->shadow_transform[p_pass].range_begin = p_range_begin;
	light_instance->shadow_transform[p_pass].shadow_texel_size = p_shadow_texel_size;
	light_instance->shadow_transform[p_pass].uv_scale = p_uv_scale;
}

void RasterizerSceneRD::light_instance_mark_visible(RID p_light_instance) {
	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->last_scene_pass = scene_pass;
}

RasterizerSceneRD::ShadowCubemap *RasterizerSceneRD::_get_shadow_cubemap(int p_size) {
	if (!shadow_cubemaps.has(p_size)) {
		ShadowCubemap sc;
		{
			RD::TextureFormat tf;
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
			tf.width = p_size;
			tf.height = p_size;
			tf.type = RD::TEXTURE_TYPE_CUBE;
			tf.array_layers = 6;
			tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
			sc.cubemap = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}

		for (int i = 0; i < 6; i++) {
			RID side_texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), sc.cubemap, i, 0);
			Vector<RID> fbtex;
			fbtex.push_back(side_texture);
			sc.side_fb[i] = RD::get_singleton()->framebuffer_create(fbtex);
		}

		shadow_cubemaps[p_size] = sc;
	}

	return &shadow_cubemaps[p_size];
}

RasterizerSceneRD::ShadowMap *RasterizerSceneRD::_get_shadow_map(const Size2i &p_size) {
	if (!shadow_maps.has(p_size)) {
		ShadowMap sm;
		{
			RD::TextureFormat tf;
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
			tf.width = p_size.width;
			tf.height = p_size.height;
			tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

			sm.depth = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}

		Vector<RID> fbtex;
		fbtex.push_back(sm.depth);
		sm.fb = RD::get_singleton()->framebuffer_create(fbtex);

		shadow_maps[p_size] = sm;
	}

	return &shadow_maps[p_size];
}

//////////////////////////

RID RasterizerSceneRD::decal_instance_create(RID p_decal) {
	DecalInstance di;
	di.decal = p_decal;
	return decal_instance_owner.make_rid(di);
}

void RasterizerSceneRD::decal_instance_set_transform(RID p_decal, const Transform &p_transform) {
	DecalInstance *di = decal_instance_owner.getornull(p_decal);
	ERR_FAIL_COND(!di);
	di->transform = p_transform;
}

/////////////////////////////////

RID RasterizerSceneRD::gi_probe_instance_create(RID p_base) {
	GIProbeInstance gi_probe;
	gi_probe.probe = p_base;
	RID rid = gi_probe_instance_owner.make_rid(gi_probe);
	return rid;
}

void RasterizerSceneRD::gi_probe_instance_set_transform_to_data(RID p_probe, const Transform &p_xform) {
	GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->transform = p_xform;
}

bool RasterizerSceneRD::gi_probe_needs_update(RID p_probe) const {
	GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!gi_probe, false);

	//return true;
	return gi_probe->last_probe_version != storage->gi_probe_get_version(gi_probe->probe);
}

void RasterizerSceneRD::gi_probe_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, int p_dynamic_object_count, InstanceBase **p_dynamic_objects) {
	GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_probe);
	ERR_FAIL_COND(!gi_probe);

	uint32_t data_version = storage->gi_probe_get_data_version(gi_probe->probe);

	// (RE)CREATE IF NEEDED

	if (gi_probe->last_probe_data_version != data_version) {
		//need to re-create everything
		if (gi_probe->texture.is_valid()) {
			RD::get_singleton()->free(gi_probe->texture);
			RD::get_singleton()->free(gi_probe->write_buffer);
			gi_probe->mipmaps.clear();
		}

		for (int i = 0; i < gi_probe->dynamic_maps.size(); i++) {
			RD::get_singleton()->free(gi_probe->dynamic_maps[i].texture);
			RD::get_singleton()->free(gi_probe->dynamic_maps[i].depth);
		}

		gi_probe->dynamic_maps.clear();

		Vector3i octree_size = storage->gi_probe_get_octree_size(gi_probe->probe);

		if (octree_size != Vector3i()) {
			//can create a 3D texture
			Vector<int> levels = storage->gi_probe_get_level_counts(gi_probe->probe);

			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
			tf.width = octree_size.x;
			tf.height = octree_size.y;
			tf.depth = octree_size.z;
			tf.type = RD::TEXTURE_TYPE_3D;
			tf.mipmaps = levels.size();

			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;

			gi_probe->texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

			RD::get_singleton()->texture_clear(gi_probe->texture, Color(0, 0, 0, 0), 0, levels.size(), 0, 1, false);

			{
				int total_elements = 0;
				for (int i = 0; i < levels.size(); i++) {
					total_elements += levels[i];
				}

				gi_probe->write_buffer = RD::get_singleton()->storage_buffer_create(total_elements * 16);
			}

			for (int i = 0; i < levels.size(); i++) {
				GIProbeInstance::Mipmap mipmap;
				mipmap.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), gi_probe->texture, 0, i, RD::TEXTURE_SLICE_3D);
				mipmap.level = levels.size() - i - 1;
				mipmap.cell_offset = 0;
				for (uint32_t j = 0; j < mipmap.level; j++) {
					mipmap.cell_offset += levels[j];
				}
				mipmap.cell_count = levels[mipmap.level];

				Vector<RD::Uniform> uniforms;
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 1;
					u.ids.push_back(storage->gi_probe_get_octree_buffer(gi_probe->probe));
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 2;
					u.ids.push_back(storage->gi_probe_get_data_buffer(gi_probe->probe));
					uniforms.push_back(u);
				}

				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 4;
					u.ids.push_back(gi_probe->write_buffer);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_TEXTURE;
					u.binding = 9;
					u.ids.push_back(storage->gi_probe_get_sdf_texture(gi_probe->probe));
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_SAMPLER;
					u.binding = 10;
					u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
					uniforms.push_back(u);
				}

				{
					Vector<RD::Uniform> copy_uniforms = uniforms;
					if (i == 0) {
						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
							u.binding = 3;
							u.ids.push_back(gi_probe_lights_uniform);
							copy_uniforms.push_back(u);
						}

						mipmap.uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, giprobe_lighting_shader_version_shaders[GI_PROBE_SHADER_VERSION_COMPUTE_LIGHT], 0);

						copy_uniforms = uniforms; //restore

						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_TEXTURE;
							u.binding = 5;
							u.ids.push_back(gi_probe->texture);
							copy_uniforms.push_back(u);
						}
						mipmap.second_bounce_uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, giprobe_lighting_shader_version_shaders[GI_PROBE_SHADER_VERSION_COMPUTE_SECOND_BOUNCE], 0);
					} else {
						mipmap.uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, giprobe_lighting_shader_version_shaders[GI_PROBE_SHADER_VERSION_COMPUTE_MIPMAP], 0);
					}
				}

				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 5;
					u.ids.push_back(mipmap.texture);
					uniforms.push_back(u);
				}

				mipmap.write_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, giprobe_lighting_shader_version_shaders[GI_PROBE_SHADER_VERSION_WRITE_TEXTURE], 0);

				gi_probe->mipmaps.push_back(mipmap);
			}

			{
				uint32_t dynamic_map_size = MAX(MAX(octree_size.x, octree_size.y), octree_size.z);
				uint32_t oversample = nearest_power_of_2_templated(4);
				int mipmap_index = 0;

				while (mipmap_index < gi_probe->mipmaps.size()) {
					GIProbeInstance::DynamicMap dmap;

					if (oversample > 0) {
						dmap.size = dynamic_map_size * (1 << oversample);
						dmap.mipmap = -1;
						oversample--;
					} else {
						dmap.size = dynamic_map_size >> mipmap_index;
						dmap.mipmap = mipmap_index;
						mipmap_index++;
					}

					RD::TextureFormat dtf;
					dtf.width = dmap.size;
					dtf.height = dmap.size;
					dtf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
					dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

					if (gi_probe->dynamic_maps.size() == 0) {
						dtf.usage_bits |= RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
					}
					dmap.texture = RD::get_singleton()->texture_create(dtf, RD::TextureView());

					if (gi_probe->dynamic_maps.size() == 0) {
						//render depth for first one
						dtf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
						dtf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
						dmap.fb_depth = RD::get_singleton()->texture_create(dtf, RD::TextureView());
					}

					//just use depth as-is
					dtf.format = RD::DATA_FORMAT_R32_SFLOAT;
					dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

					dmap.depth = RD::get_singleton()->texture_create(dtf, RD::TextureView());

					if (gi_probe->dynamic_maps.size() == 0) {
						dtf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
						dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
						dmap.albedo = RD::get_singleton()->texture_create(dtf, RD::TextureView());
						dmap.normal = RD::get_singleton()->texture_create(dtf, RD::TextureView());
						dmap.orm = RD::get_singleton()->texture_create(dtf, RD::TextureView());

						Vector<RID> fb;
						fb.push_back(dmap.albedo);
						fb.push_back(dmap.normal);
						fb.push_back(dmap.orm);
						fb.push_back(dmap.texture); //emission
						fb.push_back(dmap.depth);
						fb.push_back(dmap.fb_depth);

						dmap.fb = RD::get_singleton()->framebuffer_create(fb);

						{
							Vector<RD::Uniform> uniforms;
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
								u.binding = 3;
								u.ids.push_back(gi_probe_lights_uniform);
								uniforms.push_back(u);
							}

							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 5;
								u.ids.push_back(dmap.albedo);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 6;
								u.ids.push_back(dmap.normal);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 7;
								u.ids.push_back(dmap.orm);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_TEXTURE;
								u.binding = 8;
								u.ids.push_back(dmap.fb_depth);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_TEXTURE;
								u.binding = 9;
								u.ids.push_back(storage->gi_probe_get_sdf_texture(gi_probe->probe));
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_SAMPLER;
								u.binding = 10;
								u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 11;
								u.ids.push_back(dmap.texture);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 12;
								u.ids.push_back(dmap.depth);
								uniforms.push_back(u);
							}

							dmap.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, giprobe_lighting_shader_version_shaders[GI_PROBE_SHADER_VERSION_DYNAMIC_OBJECT_LIGHTING], 0);
						}
					} else {
						bool plot = dmap.mipmap >= 0;
						bool write = dmap.mipmap < (gi_probe->mipmaps.size() - 1);

						Vector<RD::Uniform> uniforms;

						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_IMAGE;
							u.binding = 5;
							u.ids.push_back(gi_probe->dynamic_maps[gi_probe->dynamic_maps.size() - 1].texture);
							uniforms.push_back(u);
						}
						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_IMAGE;
							u.binding = 6;
							u.ids.push_back(gi_probe->dynamic_maps[gi_probe->dynamic_maps.size() - 1].depth);
							uniforms.push_back(u);
						}

						if (write) {
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 7;
								u.ids.push_back(dmap.texture);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 8;
								u.ids.push_back(dmap.depth);
								uniforms.push_back(u);
							}
						}

						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_TEXTURE;
							u.binding = 9;
							u.ids.push_back(storage->gi_probe_get_sdf_texture(gi_probe->probe));
							uniforms.push_back(u);
						}
						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_SAMPLER;
							u.binding = 10;
							u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
							uniforms.push_back(u);
						}

						if (plot) {
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 11;
								u.ids.push_back(gi_probe->mipmaps[dmap.mipmap].texture);
								uniforms.push_back(u);
							}
						}

						dmap.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, giprobe_lighting_shader_version_shaders[(write && plot) ? GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_WRITE_PLOT : write ? GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_WRITE : GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_PLOT], 0);
					}

					gi_probe->dynamic_maps.push_back(dmap);
				}
			}
		}

		gi_probe->last_probe_data_version = data_version;
		p_update_light_instances = true; //just in case

		_base_uniforms_changed();
	}

	// UDPDATE TIME

	if (gi_probe->has_dynamic_object_data) {
		//if it has dynamic object data, it needs to be cleared
		RD::get_singleton()->texture_clear(gi_probe->texture, Color(0, 0, 0, 0), 0, gi_probe->mipmaps.size(), 0, 1, true);
	}

	uint32_t light_count = 0;

	if (p_update_light_instances || p_dynamic_object_count > 0) {
		light_count = MIN(gi_probe_max_lights, (uint32_t)p_light_instances.size());

		{
			Transform to_cell = storage->gi_probe_get_to_cell_xform(gi_probe->probe);
			Transform to_probe_xform = (gi_probe->transform * to_cell.affine_inverse()).affine_inverse();
			//update lights

			for (uint32_t i = 0; i < light_count; i++) {
				GIProbeLight &l = gi_probe_lights[i];
				RID light_instance = p_light_instances[i];
				RID light = light_instance_get_base_light(light_instance);

				l.type = storage->light_get_type(light);
				l.attenuation = storage->light_get_param(light, RS::LIGHT_PARAM_ATTENUATION);
				l.energy = storage->light_get_param(light, RS::LIGHT_PARAM_ENERGY) * storage->light_get_param(light, RS::LIGHT_PARAM_INDIRECT_ENERGY);
				l.radius = to_cell.basis.xform(Vector3(storage->light_get_param(light, RS::LIGHT_PARAM_RANGE), 0, 0)).length();
				Color color = storage->light_get_color(light).to_linear();
				l.color[0] = color.r;
				l.color[1] = color.g;
				l.color[2] = color.b;

				l.spot_angle_radians = Math::deg2rad(storage->light_get_param(light, RS::LIGHT_PARAM_SPOT_ANGLE));
				l.spot_attenuation = storage->light_get_param(light, RS::LIGHT_PARAM_SPOT_ATTENUATION);

				Transform xform = light_instance_get_base_transform(light_instance);

				Vector3 pos = to_probe_xform.xform(xform.origin);
				Vector3 dir = to_probe_xform.basis.xform(-xform.basis.get_axis(2)).normalized();

				l.position[0] = pos.x;
				l.position[1] = pos.y;
				l.position[2] = pos.z;

				l.direction[0] = dir.x;
				l.direction[1] = dir.y;
				l.direction[2] = dir.z;

				l.has_shadow = storage->light_has_shadow(light);
			}

			RD::get_singleton()->buffer_update(gi_probe_lights_uniform, 0, sizeof(GIProbeLight) * light_count, gi_probe_lights, true);
		}
	}

	if (gi_probe->has_dynamic_object_data || p_update_light_instances || p_dynamic_object_count) {
		// PROCESS MIPMAPS
		if (gi_probe->mipmaps.size()) {
			//can update mipmaps

			Vector3i probe_size = storage->gi_probe_get_octree_size(gi_probe->probe);

			GIProbePushConstant push_constant;

			push_constant.limits[0] = probe_size.x;
			push_constant.limits[1] = probe_size.y;
			push_constant.limits[2] = probe_size.z;
			push_constant.stack_size = gi_probe->mipmaps.size();
			push_constant.emission_scale = 1.0;
			push_constant.propagation = storage->gi_probe_get_propagation(gi_probe->probe);
			push_constant.dynamic_range = storage->gi_probe_get_dynamic_range(gi_probe->probe);
			push_constant.light_count = light_count;
			push_constant.aniso_strength = 0;

			/*		print_line("probe update to version " + itos(gi_probe->last_probe_version));
			print_line("propagation " + rtos(push_constant.propagation));
			print_line("dynrange " + rtos(push_constant.dynamic_range));
	*/
			RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

			int passes;
			if (p_update_light_instances) {
				passes = storage->gi_probe_is_using_two_bounces(gi_probe->probe) ? 2 : 1;
			} else {
				passes = 1; //only re-blitting is necessary
			}
			int wg_size = 64;
			int wg_limit_x = RD::get_singleton()->limit_get(RD::LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X);

			for (int pass = 0; pass < passes; pass++) {
				if (p_update_light_instances) {
					for (int i = 0; i < gi_probe->mipmaps.size(); i++) {
						if (i == 0) {
							RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[pass == 0 ? GI_PROBE_SHADER_VERSION_COMPUTE_LIGHT : GI_PROBE_SHADER_VERSION_COMPUTE_SECOND_BOUNCE]);
						} else if (i == 1) {
							RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_COMPUTE_MIPMAP]);
						}

						if (pass == 1 || i > 0) {
							RD::get_singleton()->compute_list_add_barrier(compute_list); //wait til previous step is done
						}
						if (pass == 0 || i > 0) {
							RD::get_singleton()->compute_list_bind_uniform_set(compute_list, gi_probe->mipmaps[i].uniform_set, 0);
						} else {
							RD::get_singleton()->compute_list_bind_uniform_set(compute_list, gi_probe->mipmaps[i].second_bounce_uniform_set, 0);
						}

						push_constant.cell_offset = gi_probe->mipmaps[i].cell_offset;
						push_constant.cell_count = gi_probe->mipmaps[i].cell_count;

						int wg_todo = (gi_probe->mipmaps[i].cell_count - 1) / wg_size + 1;
						while (wg_todo) {
							int wg_count = MIN(wg_todo, wg_limit_x);
							RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(GIProbePushConstant));
							RD::get_singleton()->compute_list_dispatch(compute_list, wg_count, 1, 1);
							wg_todo -= wg_count;
							push_constant.cell_offset += wg_count * wg_size;
						}
					}

					RD::get_singleton()->compute_list_add_barrier(compute_list); //wait til previous step is done
				}

				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_WRITE_TEXTURE]);

				for (int i = 0; i < gi_probe->mipmaps.size(); i++) {
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, gi_probe->mipmaps[i].write_uniform_set, 0);

					push_constant.cell_offset = gi_probe->mipmaps[i].cell_offset;
					push_constant.cell_count = gi_probe->mipmaps[i].cell_count;

					int wg_todo = (gi_probe->mipmaps[i].cell_count - 1) / wg_size + 1;
					while (wg_todo) {
						int wg_count = MIN(wg_todo, wg_limit_x);
						RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(GIProbePushConstant));
						RD::get_singleton()->compute_list_dispatch(compute_list, wg_count, 1, 1);
						wg_todo -= wg_count;
						push_constant.cell_offset += wg_count * wg_size;
					}
				}
			}

			RD::get_singleton()->compute_list_end();
		}
	}

	gi_probe->has_dynamic_object_data = false; //clear until dynamic object data is used again

	if (p_dynamic_object_count && gi_probe->dynamic_maps.size()) {
		Vector3i octree_size = storage->gi_probe_get_octree_size(gi_probe->probe);
		int multiplier = gi_probe->dynamic_maps[0].size / MAX(MAX(octree_size.x, octree_size.y), octree_size.z);

		Transform oversample_scale;
		oversample_scale.basis.scale(Vector3(multiplier, multiplier, multiplier));

		Transform to_cell = oversample_scale * storage->gi_probe_get_to_cell_xform(gi_probe->probe);
		Transform to_world_xform = gi_probe->transform * to_cell.affine_inverse();
		Transform to_probe_xform = to_world_xform.affine_inverse();

		AABB probe_aabb(Vector3(), octree_size);

		//this could probably be better parallelized in compute..
		for (int i = 0; i < p_dynamic_object_count; i++) {
			InstanceBase *instance = p_dynamic_objects[i];
			//not used, so clear
			instance->depth_layer = 0;
			instance->depth = 0;

			//transform aabb to giprobe
			AABB aabb = (to_probe_xform * instance->transform).xform(instance->aabb);

			//this needs to wrap to grid resolution to avoid jitter
			//also extend margin a bit just in case
			Vector3i begin = aabb.position - Vector3i(1, 1, 1);
			Vector3i end = aabb.position + aabb.size + Vector3i(1, 1, 1);

			for (int j = 0; j < 3; j++) {
				if ((end[j] - begin[j]) & 1) {
					end[j]++; //for half extents split, it needs to be even
				}
				begin[j] = MAX(begin[j], 0);
				end[j] = MIN(end[j], octree_size[j] * multiplier);
			}

			//aabb = aabb.intersection(probe_aabb); //intersect
			aabb.position = begin;
			aabb.size = end - begin;

			//print_line("aabb: " + aabb);

			for (int j = 0; j < 6; j++) {
				//if (j != 0 && j != 3) {
				//	continue;
				//}
				static const Vector3 render_z[6] = {
					Vector3(1, 0, 0),
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(-1, 0, 0),
					Vector3(0, -1, 0),
					Vector3(0, 0, -1),
				};
				static const Vector3 render_up[6] = {
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(0, 1, 0),
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(0, 1, 0),
				};

				Vector3 render_dir = render_z[j];
				Vector3 up_dir = render_up[j];

				Vector3 center = aabb.position + aabb.size * 0.5;
				Transform xform;
				xform.set_look_at(center - aabb.size * 0.5 * render_dir, center, up_dir);

				Vector3 x_dir = xform.basis.get_axis(0).abs();
				int x_axis = int(Vector3(0, 1, 2).dot(x_dir));
				Vector3 y_dir = xform.basis.get_axis(1).abs();
				int y_axis = int(Vector3(0, 1, 2).dot(y_dir));
				Vector3 z_dir = -xform.basis.get_axis(2);
				int z_axis = int(Vector3(0, 1, 2).dot(z_dir.abs()));

				Rect2i rect(aabb.position[x_axis], aabb.position[y_axis], aabb.size[x_axis], aabb.size[y_axis]);
				bool x_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_axis(0)) < 0);
				bool y_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_axis(1)) < 0);
				bool z_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_axis(2)) > 0);

				CameraMatrix cm;
				cm.set_orthogonal(-rect.size.width / 2, rect.size.width / 2, -rect.size.height / 2, rect.size.height / 2, 0.0001, aabb.size[z_axis]);

				_render_material(to_world_xform * xform, cm, true, &instance, 1, gi_probe->dynamic_maps[0].fb, Rect2i(Vector2i(), rect.size));

				GIProbeDynamicPushConstant push_constant;
				zeromem(&push_constant, sizeof(GIProbeDynamicPushConstant));
				push_constant.limits[0] = octree_size.x;
				push_constant.limits[1] = octree_size.y;
				push_constant.limits[2] = octree_size.z;
				push_constant.light_count = p_light_instances.size();
				push_constant.x_dir[0] = x_dir[0];
				push_constant.x_dir[1] = x_dir[1];
				push_constant.x_dir[2] = x_dir[2];
				push_constant.y_dir[0] = y_dir[0];
				push_constant.y_dir[1] = y_dir[1];
				push_constant.y_dir[2] = y_dir[2];
				push_constant.z_dir[0] = z_dir[0];
				push_constant.z_dir[1] = z_dir[1];
				push_constant.z_dir[2] = z_dir[2];
				push_constant.z_base = xform.origin[z_axis];
				push_constant.z_sign = (z_flip ? -1.0 : 1.0);
				push_constant.pos_multiplier = float(1.0) / multiplier;
				push_constant.dynamic_range = storage->gi_probe_get_dynamic_range(gi_probe->probe);
				push_constant.flip_x = x_flip;
				push_constant.flip_y = y_flip;
				push_constant.rect_pos[0] = rect.position[0];
				push_constant.rect_pos[1] = rect.position[1];
				push_constant.rect_size[0] = rect.size[0];
				push_constant.rect_size[1] = rect.size[1];
				push_constant.prev_rect_ofs[0] = 0;
				push_constant.prev_rect_ofs[1] = 0;
				push_constant.prev_rect_size[0] = 0;
				push_constant.prev_rect_size[1] = 0;
				push_constant.on_mipmap = false;
				push_constant.propagation = storage->gi_probe_get_propagation(gi_probe->probe);
				push_constant.pad[0] = 0;
				push_constant.pad[1] = 0;
				push_constant.pad[2] = 0;

				//process lighting
				RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_DYNAMIC_OBJECT_LIGHTING]);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, gi_probe->dynamic_maps[0].uniform_set, 0);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(GIProbeDynamicPushConstant));
				RD::get_singleton()->compute_list_dispatch(compute_list, (rect.size.x - 1) / 8 + 1, (rect.size.y - 1) / 8 + 1, 1);
				//print_line("rect: " + itos(i) + ": " + rect);

				for (int k = 1; k < gi_probe->dynamic_maps.size(); k++) {
					// enlarge the rect if needed so all pixels fit when downscaled,
					// this ensures downsampling is smooth and optimal because no pixels are left behind

					//x
					if (rect.position.x & 1) {
						rect.size.x++;
						push_constant.prev_rect_ofs[0] = 1; //this is used to ensure reading is also optimal
					} else {
						push_constant.prev_rect_ofs[0] = 0;
					}
					if (rect.size.x & 1) {
						rect.size.x++;
					}

					rect.position.x >>= 1;
					rect.size.x = MAX(1, rect.size.x >> 1);

					//y
					if (rect.position.y & 1) {
						rect.size.y++;
						push_constant.prev_rect_ofs[1] = 1;
					} else {
						push_constant.prev_rect_ofs[1] = 0;
					}
					if (rect.size.y & 1) {
						rect.size.y++;
					}

					rect.position.y >>= 1;
					rect.size.y = MAX(1, rect.size.y >> 1);

					//shrink limits to ensure plot does not go outside map
					if (gi_probe->dynamic_maps[k].mipmap > 0) {
						for (int l = 0; l < 3; l++) {
							push_constant.limits[l] = MAX(1, push_constant.limits[l] >> 1);
						}
					}

					//print_line("rect: " + itos(i) + ": " + rect);
					push_constant.rect_pos[0] = rect.position[0];
					push_constant.rect_pos[1] = rect.position[1];
					push_constant.prev_rect_size[0] = push_constant.rect_size[0];
					push_constant.prev_rect_size[1] = push_constant.rect_size[1];
					push_constant.rect_size[0] = rect.size[0];
					push_constant.rect_size[1] = rect.size[1];
					push_constant.on_mipmap = gi_probe->dynamic_maps[k].mipmap > 0;

					RD::get_singleton()->compute_list_add_barrier(compute_list);

					if (gi_probe->dynamic_maps[k].mipmap < 0) {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_WRITE]);
					} else if (k < gi_probe->dynamic_maps.size() - 1) {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_WRITE_PLOT]);
					} else {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_PLOT]);
					}
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, gi_probe->dynamic_maps[k].uniform_set, 0);
					RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(GIProbeDynamicPushConstant));
					RD::get_singleton()->compute_list_dispatch(compute_list, (rect.size.x - 1) / 8 + 1, (rect.size.y - 1) / 8 + 1, 1);
				}

				RD::get_singleton()->compute_list_end();
			}
		}

		gi_probe->has_dynamic_object_data = true; //clear until dynamic object data is used again
	}

	gi_probe->last_probe_version = storage->gi_probe_get_version(gi_probe->probe);
}

void RasterizerSceneRD::_debug_giprobe(RID p_gi_probe, RD::DrawListID p_draw_list, RID p_framebuffer, const CameraMatrix &p_camera_with_transform, bool p_lighting, bool p_emission, float p_alpha) {
	GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	if (gi_probe->mipmaps.size() == 0) {
		return;
	}

	CameraMatrix transform = (p_camera_with_transform * CameraMatrix(gi_probe->transform)) * CameraMatrix(storage->gi_probe_get_to_cell_xform(gi_probe->probe).affine_inverse());

	int level = 0;
	Vector3i octree_size = storage->gi_probe_get_octree_size(gi_probe->probe);

	GIProbeDebugPushConstant push_constant;
	push_constant.alpha = p_alpha;
	push_constant.dynamic_range = storage->gi_probe_get_dynamic_range(gi_probe->probe);
	push_constant.cell_offset = gi_probe->mipmaps[level].cell_offset;
	push_constant.level = level;

	push_constant.bounds[0] = octree_size.x >> level;
	push_constant.bounds[1] = octree_size.y >> level;
	push_constant.bounds[2] = octree_size.z >> level;
	push_constant.pad = 0;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			push_constant.projection[i * 4 + j] = transform.matrix[i][j];
		}
	}

	if (giprobe_debug_uniform_set.is_valid()) {
		RD::get_singleton()->free(giprobe_debug_uniform_set);
	}
	Vector<RD::Uniform> uniforms;
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.binding = 1;
		u.ids.push_back(storage->gi_probe_get_data_buffer(gi_probe->probe));
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 2;
		u.ids.push_back(gi_probe->texture);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_SAMPLER;
		u.binding = 3;
		u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
		uniforms.push_back(u);
	}

	int cell_count;
	if (!p_emission && p_lighting && gi_probe->has_dynamic_object_data) {
		cell_count = push_constant.bounds[0] * push_constant.bounds[1] * push_constant.bounds[2];
	} else {
		cell_count = gi_probe->mipmaps[level].cell_count;
	}

	giprobe_debug_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, giprobe_debug_shader_version_shaders[0], 0);
	RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, giprobe_debug_shader_version_pipelines[p_emission ? GI_PROBE_DEBUG_EMISSION : p_lighting ? (gi_probe->has_dynamic_object_data ? GI_PROBE_DEBUG_LIGHT_FULL : GI_PROBE_DEBUG_LIGHT) : GI_PROBE_DEBUG_COLOR].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, giprobe_debug_uniform_set, 0);
	RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(GIProbeDebugPushConstant));
	RD::get_singleton()->draw_list_draw(p_draw_list, false, cell_count, 36);
}

void RasterizerSceneRD::_debug_sdfgi_probes(RID p_render_buffers, RD::DrawListID p_draw_list, RID p_framebuffer, const CameraMatrix &p_camera_with_transform) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);

	if (!rb->sdfgi) {
		return; //nothing to debug
	}

	SDGIShader::DebugProbesPushConstant push_constant;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			push_constant.projection[i * 4 + j] = p_camera_with_transform.matrix[i][j];
		}
	}

	//gen spheres from strips
	uint32_t band_points = 16;
	push_constant.band_power = 4;
	push_constant.sections_in_band = ((band_points / 2) - 1);
	push_constant.band_mask = band_points - 2;
	push_constant.section_arc = (Math_PI * 2.0) / float(push_constant.sections_in_band);
	push_constant.y_mult = rb->sdfgi->y_mult;

	uint32_t total_points = push_constant.sections_in_band * band_points;
	uint32_t total_probes = rb->sdfgi->probe_axis_count * rb->sdfgi->probe_axis_count * rb->sdfgi->probe_axis_count;

	push_constant.grid_size[0] = rb->sdfgi->cascade_size;
	push_constant.grid_size[1] = rb->sdfgi->cascade_size;
	push_constant.grid_size[2] = rb->sdfgi->cascade_size;
	push_constant.cascade = 0;

	push_constant.probe_axis_size = rb->sdfgi->probe_axis_count;

	if (!rb->sdfgi->debug_probes_uniform_set.is_valid() || !RD::get_singleton()->uniform_set_is_valid(rb->sdfgi->debug_probes_uniform_set)) {
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.binding = 1;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.ids.push_back(rb->sdfgi->cascades_ubo);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 2;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.ids.push_back(rb->sdfgi->lightprobe_texture);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 3;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
			u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 4;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.ids.push_back(rb->sdfgi->occlusion_texture);
			uniforms.push_back(u);
		}

		rb->sdfgi->debug_probes_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.debug_probes.version_get_shader(sdfgi_shader.debug_probes_shader, 0), 0);
	}

	RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, sdfgi_shader.debug_probes_pipeline[SDGIShader::PROBE_DEBUG_PROBES].get_render_pipeline(RD::INVALID_FORMAT_ID, RD::get_singleton()->framebuffer_get_format(p_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, rb->sdfgi->debug_probes_uniform_set, 0);
	RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(SDGIShader::DebugProbesPushConstant));
	RD::get_singleton()->draw_list_draw(p_draw_list, false, total_probes, total_points);

	if (sdfgi_debug_probe_dir != Vector3()) {
		print_line("CLICK DEBUG ME?");
		uint32_t cascade = 0;
		Vector3 offset = Vector3((Vector3i(1, 1, 1) * -int32_t(rb->sdfgi->cascade_size >> 1) + rb->sdfgi->cascades[cascade].position)) * rb->sdfgi->cascades[cascade].cell_size * Vector3(1.0, 1.0 / rb->sdfgi->y_mult, 1.0);
		Vector3 probe_size = rb->sdfgi->cascades[cascade].cell_size * (rb->sdfgi->cascade_size / SDFGI::PROBE_DIVISOR) * Vector3(1.0, 1.0 / rb->sdfgi->y_mult, 1.0);
		Vector3 ray_from = sdfgi_debug_probe_pos;
		Vector3 ray_to = sdfgi_debug_probe_pos + sdfgi_debug_probe_dir * rb->sdfgi->cascades[cascade].cell_size * Math::sqrt(3.0) * rb->sdfgi->cascade_size;
		float sphere_radius = 0.2;
		float closest_dist = 1e20;
		sdfgi_debug_probe_enabled = false;

		Vector3i probe_from = rb->sdfgi->cascades[cascade].position / (rb->sdfgi->cascade_size / SDFGI::PROBE_DIVISOR);
		for (int i = 0; i < (SDFGI::PROBE_DIVISOR + 1); i++) {
			for (int j = 0; j < (SDFGI::PROBE_DIVISOR + 1); j++) {
				for (int k = 0; k < (SDFGI::PROBE_DIVISOR + 1); k++) {
					Vector3 pos = offset + probe_size * Vector3(i, j, k);
					Vector3 res;
					if (Geometry3D::segment_intersects_sphere(ray_from, ray_to, pos, sphere_radius, &res)) {
						float d = ray_from.distance_to(res);
						if (d < closest_dist) {
							closest_dist = d;
							sdfgi_debug_probe_enabled = true;
							sdfgi_debug_probe_index = probe_from + Vector3i(i, j, k);
						}
					}
				}
			}
		}

		if (sdfgi_debug_probe_enabled) {
			print_line("found: " + sdfgi_debug_probe_index);
		} else {
			print_line("no found");
		}
		sdfgi_debug_probe_dir = Vector3();
	}

	if (sdfgi_debug_probe_enabled) {
		uint32_t cascade = 0;
		uint32_t probe_cells = (rb->sdfgi->cascade_size / SDFGI::PROBE_DIVISOR);
		Vector3i probe_from = rb->sdfgi->cascades[cascade].position / probe_cells;
		Vector3i ofs = sdfgi_debug_probe_index - probe_from;
		if (ofs.x < 0 || ofs.y < 0 || ofs.z < 0) {
			return;
		}
		if (ofs.x > SDFGI::PROBE_DIVISOR || ofs.y > SDFGI::PROBE_DIVISOR || ofs.z > SDFGI::PROBE_DIVISOR) {
			return;
		}

		uint32_t mult = (SDFGI::PROBE_DIVISOR + 1);
		uint32_t index = ofs.z * mult * mult + ofs.y * mult + ofs.x;

		push_constant.probe_debug_index = index;

		uint32_t cell_count = probe_cells * 2 * probe_cells * 2 * probe_cells * 2;

		RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, sdfgi_shader.debug_probes_pipeline[SDGIShader::PROBE_DEBUG_VISIBILITY].get_render_pipeline(RD::INVALID_FORMAT_ID, RD::get_singleton()->framebuffer_get_format(p_framebuffer)));
		RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, rb->sdfgi->debug_probes_uniform_set, 0);
		RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(SDGIShader::DebugProbesPushConstant));
		RD::get_singleton()->draw_list_draw(p_draw_list, false, cell_count, total_points);
	}
}

////////////////////////////////
RID RasterizerSceneRD::render_buffers_create() {
	RenderBuffers rb;
	rb.data = _create_render_buffer_data();
	return render_buffers_owner.make_rid(rb);
}

void RasterizerSceneRD::_allocate_blur_textures(RenderBuffers *rb) {
	ERR_FAIL_COND(!rb->blur[0].texture.is_null());

	uint32_t mipmaps_required = Image::get_image_required_mipmaps(rb->width, rb->height, Image::FORMAT_RGBAH);

	RD::TextureFormat tf;
	tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
	tf.width = rb->width;
	tf.height = rb->height;
	tf.type = RD::TEXTURE_TYPE_2D;
	tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	tf.mipmaps = mipmaps_required;

	rb->blur[0].texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
	//the second one is smaller (only used for separatable part of blur)
	tf.width >>= 1;
	tf.height >>= 1;
	tf.mipmaps--;
	rb->blur[1].texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

	int base_width = rb->width;
	int base_height = rb->height;

	for (uint32_t i = 0; i < mipmaps_required; i++) {
		RenderBuffers::Blur::Mipmap mm;
		mm.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->blur[0].texture, 0, i);

		mm.width = base_width;
		mm.height = base_height;

		rb->blur[0].mipmaps.push_back(mm);

		if (i > 0) {
			mm.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->blur[1].texture, 0, i - 1);

			rb->blur[1].mipmaps.push_back(mm);
		}

		base_width = MAX(1, base_width >> 1);
		base_height = MAX(1, base_height >> 1);
	}
}

void RasterizerSceneRD::_allocate_luminance_textures(RenderBuffers *rb) {
	ERR_FAIL_COND(!rb->luminance.current.is_null());

	int w = rb->width;
	int h = rb->height;

	while (true) {
		w = MAX(w / 8, 1);
		h = MAX(h / 8, 1);

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		tf.width = w;
		tf.height = h;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

		bool final = w == 1 && h == 1;

		if (final) {
			tf.usage_bits |= RD::TEXTURE_USAGE_SAMPLING_BIT;
		}

		RID texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

		rb->luminance.reduce.push_back(texture);

		if (final) {
			rb->luminance.current = RD::get_singleton()->texture_create(tf, RD::TextureView());
			break;
		}
	}
}

void RasterizerSceneRD::_free_render_buffer_data(RenderBuffers *rb) {
	if (rb->texture.is_valid()) {
		RD::get_singleton()->free(rb->texture);
		rb->texture = RID();
	}

	if (rb->depth_texture.is_valid()) {
		RD::get_singleton()->free(rb->depth_texture);
		rb->depth_texture = RID();
	}

	for (int i = 0; i < 2; i++) {
		if (rb->blur[i].texture.is_valid()) {
			RD::get_singleton()->free(rb->blur[i].texture);
			rb->blur[i].texture = RID();
			rb->blur[i].mipmaps.clear();
		}
	}

	for (int i = 0; i < rb->luminance.reduce.size(); i++) {
		RD::get_singleton()->free(rb->luminance.reduce[i]);
	}

	for (int i = 0; i < rb->luminance.reduce.size(); i++) {
		RD::get_singleton()->free(rb->luminance.reduce[i]);
	}
	rb->luminance.reduce.clear();

	if (rb->luminance.current.is_valid()) {
		RD::get_singleton()->free(rb->luminance.current);
		rb->luminance.current = RID();
	}

	if (rb->ssao.ao[0].is_valid()) {
		RD::get_singleton()->free(rb->ssao.depth);
		RD::get_singleton()->free(rb->ssao.ao[0]);
		if (rb->ssao.ao[1].is_valid()) {
			RD::get_singleton()->free(rb->ssao.ao[1]);
		}
		if (rb->ssao.ao_full.is_valid()) {
			RD::get_singleton()->free(rb->ssao.ao_full);
		}

		rb->ssao.depth = RID();
		rb->ssao.ao[0] = RID();
		rb->ssao.ao[1] = RID();
		rb->ssao.ao_full = RID();
		rb->ssao.depth_slices.clear();
	}

	if (rb->ssr.blur_radius[0].is_valid()) {
		RD::get_singleton()->free(rb->ssr.blur_radius[0]);
		RD::get_singleton()->free(rb->ssr.blur_radius[1]);
		rb->ssr.blur_radius[0] = RID();
		rb->ssr.blur_radius[1] = RID();
	}

	if (rb->ssr.depth_scaled.is_valid()) {
		RD::get_singleton()->free(rb->ssr.depth_scaled);
		rb->ssr.depth_scaled = RID();
		RD::get_singleton()->free(rb->ssr.normal_scaled);
		rb->ssr.normal_scaled = RID();
	}
}

void RasterizerSceneRD::_process_sss(RID p_render_buffers, const CameraMatrix &p_camera) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);

	bool can_use_effects = rb->width >= 8 && rb->height >= 8;

	if (!can_use_effects) {
		//just copy
		return;
	}

	if (rb->blur[0].texture.is_null()) {
		_allocate_blur_textures(rb);
		_render_buffers_uniform_set_changed(p_render_buffers);
	}

	storage->get_effects()->sub_surface_scattering(rb->texture, rb->blur[0].mipmaps[0].texture, rb->depth_texture, p_camera, Size2i(rb->width, rb->height), sss_scale, sss_depth_scale, sss_quality);
}

void RasterizerSceneRD::_process_ssr(RID p_render_buffers, RID p_dest_framebuffer, RID p_normal_buffer, RID p_specular_buffer, RID p_metallic, const Color &p_metallic_mask, RID p_environment, const CameraMatrix &p_projection, bool p_use_additive) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);

	bool can_use_effects = rb->width >= 8 && rb->height >= 8;

	if (!can_use_effects) {
		//just copy
		storage->get_effects()->merge_specular(p_dest_framebuffer, p_specular_buffer, p_use_additive ? RID() : rb->texture, RID());
		return;
	}

	Environment *env = environment_owner.getornull(p_environment);
	ERR_FAIL_COND(!env);

	ERR_FAIL_COND(!env->ssr_enabled);

	if (rb->ssr.depth_scaled.is_null()) {
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		tf.width = rb->width / 2;
		tf.height = rb->height / 2;
		tf.type = RD::TEXTURE_TYPE_2D;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

		rb->ssr.depth_scaled = RD::get_singleton()->texture_create(tf, RD::TextureView());

		tf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;

		rb->ssr.normal_scaled = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	if (ssr_roughness_quality != RS::ENV_SSR_ROUGNESS_QUALITY_DISABLED && !rb->ssr.blur_radius[0].is_valid()) {
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R8_UNORM;
		tf.width = rb->width / 2;
		tf.height = rb->height / 2;
		tf.type = RD::TEXTURE_TYPE_2D;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

		rb->ssr.blur_radius[0] = RD::get_singleton()->texture_create(tf, RD::TextureView());
		rb->ssr.blur_radius[1] = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	if (rb->blur[0].texture.is_null()) {
		_allocate_blur_textures(rb);
		_render_buffers_uniform_set_changed(p_render_buffers);
	}

	storage->get_effects()->screen_space_reflection(rb->texture, p_normal_buffer, ssr_roughness_quality, rb->ssr.blur_radius[0], rb->ssr.blur_radius[1], p_metallic, p_metallic_mask, rb->depth_texture, rb->ssr.depth_scaled, rb->ssr.normal_scaled, rb->blur[0].mipmaps[1].texture, rb->blur[1].mipmaps[0].texture, Size2i(rb->width / 2, rb->height / 2), env->ssr_max_steps, env->ssr_fade_in, env->ssr_fade_out, env->ssr_depth_tolerance, p_projection);
	storage->get_effects()->merge_specular(p_dest_framebuffer, p_specular_buffer, p_use_additive ? RID() : rb->texture, rb->blur[0].mipmaps[1].texture);
}

void RasterizerSceneRD::_process_ssao(RID p_render_buffers, RID p_environment, RID p_normal_buffer, const CameraMatrix &p_projection) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);

	Environment *env = environment_owner.getornull(p_environment);
	ERR_FAIL_COND(!env);

	RENDER_TIMESTAMP("Process SSAO");

	if (rb->ssao.ao[0].is_valid() && rb->ssao.ao_full.is_valid() != ssao_half_size) {
		RD::get_singleton()->free(rb->ssao.depth);
		RD::get_singleton()->free(rb->ssao.ao[0]);
		if (rb->ssao.ao[1].is_valid()) {
			RD::get_singleton()->free(rb->ssao.ao[1]);
		}
		if (rb->ssao.ao_full.is_valid()) {
			RD::get_singleton()->free(rb->ssao.ao_full);
		}

		rb->ssao.depth = RID();
		rb->ssao.ao[0] = RID();
		rb->ssao.ao[1] = RID();
		rb->ssao.ao_full = RID();
		rb->ssao.depth_slices.clear();
	}

	if (!rb->ssao.ao[0].is_valid()) {
		//allocate depth slices

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R32_SFLOAT;
			tf.width = rb->width / 2;
			tf.height = rb->height / 2;
			tf.mipmaps = Image::get_image_required_mipmaps(tf.width, tf.height, Image::FORMAT_RF) + 1;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ssao.depth = RD::get_singleton()->texture_create(tf, RD::TextureView());
			for (uint32_t i = 0; i < tf.mipmaps; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->ssao.depth, 0, i);
				rb->ssao.depth_slices.push_back(slice);
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = ssao_half_size ? rb->width / 2 : rb->width;
			tf.height = ssao_half_size ? rb->height / 2 : rb->height;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ssao.ao[0] = RD::get_singleton()->texture_create(tf, RD::TextureView());
			rb->ssao.ao[1] = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}

		if (ssao_half_size) {
			//upsample texture
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = rb->width;
			tf.height = rb->height;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ssao.ao_full = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}

		_render_buffers_uniform_set_changed(p_render_buffers);
	}

	storage->get_effects()->generate_ssao(rb->depth_texture, p_normal_buffer, Size2i(rb->width, rb->height), rb->ssao.depth, rb->ssao.depth_slices, rb->ssao.ao[0], rb->ssao.ao_full.is_valid(), rb->ssao.ao[1], rb->ssao.ao_full, env->ssao_intensity, env->ssao_radius, env->ssao_bias, p_projection, ssao_quality, env->ssao_blur, env->ssao_blur_edge_sharpness);
}

void RasterizerSceneRD::_render_buffers_post_process_and_tonemap(RID p_render_buffers, RID p_environment, RID p_camera_effects, const CameraMatrix &p_projection) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);

	Environment *env = environment_owner.getornull(p_environment);
	//glow (if enabled)
	CameraEffects *camfx = camera_effects_owner.getornull(p_camera_effects);

	bool can_use_effects = rb->width >= 8 && rb->height >= 8;

	if (can_use_effects && camfx && (camfx->dof_blur_near_enabled || camfx->dof_blur_far_enabled) && camfx->dof_blur_amount > 0.0) {
		if (rb->blur[0].texture.is_null()) {
			_allocate_blur_textures(rb);
			_render_buffers_uniform_set_changed(p_render_buffers);
		}

		float bokeh_size = camfx->dof_blur_amount * 64.0;
		storage->get_effects()->bokeh_dof(rb->texture, rb->depth_texture, Size2i(rb->width, rb->height), rb->blur[0].mipmaps[0].texture, rb->blur[1].mipmaps[0].texture, rb->blur[0].mipmaps[1].texture, camfx->dof_blur_far_enabled, camfx->dof_blur_far_distance, camfx->dof_blur_far_transition, camfx->dof_blur_near_enabled, camfx->dof_blur_near_distance, camfx->dof_blur_near_transition, bokeh_size, dof_blur_bokeh_shape, dof_blur_quality, dof_blur_use_jitter, p_projection.get_z_near(), p_projection.get_z_far(), p_projection.is_orthogonal());
	}

	if (can_use_effects && env && env->auto_exposure) {
		if (rb->luminance.current.is_null()) {
			_allocate_luminance_textures(rb);
			_render_buffers_uniform_set_changed(p_render_buffers);
		}

		bool set_immediate = env->auto_exposure_version != rb->auto_exposure_version;
		rb->auto_exposure_version = env->auto_exposure_version;

		double step = env->auto_exp_speed * time_step;
		storage->get_effects()->luminance_reduction(rb->texture, Size2i(rb->width, rb->height), rb->luminance.reduce, rb->luminance.current, env->min_luminance, env->max_luminance, step, set_immediate);

		//swap final reduce with prev luminance
		SWAP(rb->luminance.current, rb->luminance.reduce.write[rb->luminance.reduce.size() - 1]);
		RenderingServerRaster::redraw_request(); //redraw all the time if auto exposure rendering is on
	}

	int max_glow_level = -1;

	if (can_use_effects && env && env->glow_enabled) {
		/* see that blur textures are allocated */

		if (rb->blur[1].texture.is_null()) {
			_allocate_blur_textures(rb);
			_render_buffers_uniform_set_changed(p_render_buffers);
		}

		for (int i = 0; i < RS::MAX_GLOW_LEVELS; i++) {
			if (env->glow_levels[i] > 0.0) {
				if (i >= rb->blur[1].mipmaps.size()) {
					max_glow_level = rb->blur[1].mipmaps.size() - 1;
				} else {
					max_glow_level = i;
				}
			}
		}

		for (int i = 0; i < (max_glow_level + 1); i++) {
			int vp_w = rb->blur[1].mipmaps[i].width;
			int vp_h = rb->blur[1].mipmaps[i].height;

			if (i == 0) {
				RID luminance_texture;
				if (env->auto_exposure && rb->luminance.current.is_valid()) {
					luminance_texture = rb->luminance.current;
				}
				storage->get_effects()->gaussian_glow(rb->texture, rb->blur[1].mipmaps[i].texture, Size2i(vp_w, vp_h), env->glow_strength, glow_high_quality, true, env->glow_hdr_luminance_cap, env->exposure, env->glow_bloom, env->glow_hdr_bleed_threshold, env->glow_hdr_bleed_scale, luminance_texture, env->auto_exp_scale);
			} else {
				storage->get_effects()->gaussian_glow(rb->blur[1].mipmaps[i - 1].texture, rb->blur[1].mipmaps[i].texture, Size2i(vp_w, vp_h), env->glow_strength, glow_high_quality);
			}
		}
	}

	{
		//tonemap
		RasterizerEffectsRD::TonemapSettings tonemap;

		tonemap.color_correction_texture = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE);

		if (can_use_effects && env && env->auto_exposure && rb->luminance.current.is_valid()) {
			tonemap.use_auto_exposure = true;
			tonemap.exposure_texture = rb->luminance.current;
			tonemap.auto_exposure_grey = env->auto_exp_scale;
		} else {
			tonemap.exposure_texture = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE);
		}

		if (can_use_effects && env && env->glow_enabled) {
			tonemap.use_glow = true;
			tonemap.glow_mode = RasterizerEffectsRD::TonemapSettings::GlowMode(env->glow_blend_mode);
			tonemap.glow_intensity = env->glow_blend_mode == RS::ENV_GLOW_BLEND_MODE_MIX ? env->glow_mix : env->glow_intensity;
			for (int i = 0; i < RS::MAX_GLOW_LEVELS; i++) {
				tonemap.glow_levels[i] = env->glow_levels[i];
			}
			tonemap.glow_texture_size.x = rb->blur[1].mipmaps[0].width;
			tonemap.glow_texture_size.y = rb->blur[1].mipmaps[0].height;
			tonemap.glow_use_bicubic_upscale = glow_bicubic_upscale;
			tonemap.glow_texture = rb->blur[1].texture;
		} else {
			tonemap.glow_texture = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_BLACK);
		}

		if (rb->screen_space_aa == RS::VIEWPORT_SCREEN_SPACE_AA_FXAA) {
			tonemap.use_fxaa = true;
		}

		tonemap.use_debanding = rb->use_debanding;
		tonemap.texture_size = Vector2i(rb->width, rb->height);

		if (env) {
			tonemap.tonemap_mode = env->tone_mapper;
			tonemap.white = env->white;
			tonemap.exposure = env->exposure;
		}

		storage->get_effects()->tonemapper(rb->texture, storage->render_target_get_rd_framebuffer(rb->render_target), tonemap);
	}

	storage->render_target_disable_clear_request(rb->render_target);
}

void RasterizerSceneRD::_render_buffers_debug_draw(RID p_render_buffers, RID p_shadow_atlas) {
	RasterizerEffectsRD *effects = storage->get_effects();

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_SHADOW_ATLAS) {
		if (p_shadow_atlas.is_valid()) {
			RID shadow_atlas_texture = shadow_atlas_get_texture(p_shadow_atlas);
			Size2 rtsize = storage->render_target_get_size(rb->render_target);

			effects->copy_to_fb_rect(shadow_atlas_texture, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2i(Vector2(), rtsize / 2), false, true);
		}
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_DIRECTIONAL_SHADOW_ATLAS) {
		if (directional_shadow_get_texture().is_valid()) {
			RID shadow_atlas_texture = directional_shadow_get_texture();
			Size2 rtsize = storage->render_target_get_size(rb->render_target);

			effects->copy_to_fb_rect(shadow_atlas_texture, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2i(Vector2(), rtsize / 2), false, true);
		}
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_DECAL_ATLAS) {
		RID decal_atlas = storage->decal_atlas_get_texture();

		if (decal_atlas.is_valid()) {
			Size2 rtsize = storage->render_target_get_size(rb->render_target);

			effects->copy_to_fb_rect(decal_atlas, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2i(Vector2(), rtsize / 2), false, false, true);
		}
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_SCENE_LUMINANCE) {
		if (rb->luminance.current.is_valid()) {
			Size2 rtsize = storage->render_target_get_size(rb->render_target);

			effects->copy_to_fb_rect(rb->luminance.current, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize / 8), false, true);
		}
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_SSAO && rb->ssao.ao[0].is_valid()) {
		Size2 rtsize = storage->render_target_get_size(rb->render_target);
		RID ao_buf = rb->ssao.ao_full.is_valid() ? rb->ssao.ao_full : rb->ssao.ao[0];
		effects->copy_to_fb_rect(ao_buf, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize), false, true);
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_NORMAL_BUFFER && _render_buffers_get_normal_texture(p_render_buffers).is_valid()) {
		Size2 rtsize = storage->render_target_get_size(rb->render_target);
		effects->copy_to_fb_rect(_render_buffers_get_normal_texture(p_render_buffers), storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize), false, false);
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_GI_BUFFER && _render_buffers_get_ambient_texture(p_render_buffers).is_valid()) {
		Size2 rtsize = storage->render_target_get_size(rb->render_target);
		RID ambient_texture = _render_buffers_get_ambient_texture(p_render_buffers);
		RID reflection_texture = _render_buffers_get_reflection_texture(p_render_buffers);
		effects->copy_to_fb_rect(ambient_texture, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize), false, false, false, true, reflection_texture);
	}
}

void RasterizerSceneRD::_sdfgi_debug_draw(RID p_render_buffers, const CameraMatrix &p_projection, const Transform &p_transform) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);

	if (!rb->sdfgi) {
		return; //eh
	}

	if (!rb->sdfgi->debug_uniform_set.is_valid() || !RD::get_singleton()->uniform_set_is_valid(rb->sdfgi->debug_uniform_set)) {
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.binding = 1;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			for (uint32_t i = 0; i < SDFGI::MAX_CASCADES; i++) {
				if (i < rb->sdfgi->cascades.size()) {
					u.ids.push_back(rb->sdfgi->cascades[i].sdf_tex);
				} else {
					u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
				}
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 2;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			for (uint32_t i = 0; i < SDFGI::MAX_CASCADES; i++) {
				if (i < rb->sdfgi->cascades.size()) {
					u.ids.push_back(rb->sdfgi->cascades[i].light_tex);
				} else {
					u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
				}
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 3;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			for (uint32_t i = 0; i < SDFGI::MAX_CASCADES; i++) {
				if (i < rb->sdfgi->cascades.size()) {
					u.ids.push_back(rb->sdfgi->cascades[i].light_aniso_0_tex);
				} else {
					u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
				}
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 4;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			for (uint32_t i = 0; i < SDFGI::MAX_CASCADES; i++) {
				if (i < rb->sdfgi->cascades.size()) {
					u.ids.push_back(rb->sdfgi->cascades[i].light_aniso_1_tex);
				} else {
					u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
				}
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 5;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.ids.push_back(rb->sdfgi->occlusion_texture);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 8;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
			u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 9;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.ids.push_back(rb->sdfgi->cascades_ubo);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 10;
			u.type = RD::UNIFORM_TYPE_IMAGE;
			u.ids.push_back(rb->texture);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 11;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.ids.push_back(rb->sdfgi->lightprobe_texture);
			uniforms.push_back(u);
		}
		rb->sdfgi->debug_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.debug_shader_version, 0);
	}

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.debug_pipeline);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->debug_uniform_set, 0);

	SDGIShader::DebugPushConstant push_constant;
	push_constant.grid_size[0] = rb->sdfgi->cascade_size;
	push_constant.grid_size[1] = rb->sdfgi->cascade_size;
	push_constant.grid_size[2] = rb->sdfgi->cascade_size;
	push_constant.max_cascades = rb->sdfgi->cascades.size();
	push_constant.screen_size[0] = rb->width;
	push_constant.screen_size[1] = rb->height;
	push_constant.probe_axis_size = rb->sdfgi->probe_axis_count;
	push_constant.use_occlusion = rb->sdfgi->uses_occlusion;
	push_constant.y_mult = rb->sdfgi->y_mult;

	Vector2 vp_half = p_projection.get_viewport_half_extents();
	push_constant.cam_extent[0] = vp_half.x;
	push_constant.cam_extent[1] = vp_half.y;
	push_constant.cam_extent[2] = -p_projection.get_z_near();

	push_constant.cam_transform[0] = p_transform.basis.elements[0][0];
	push_constant.cam_transform[1] = p_transform.basis.elements[1][0];
	push_constant.cam_transform[2] = p_transform.basis.elements[2][0];
	push_constant.cam_transform[3] = 0;
	push_constant.cam_transform[4] = p_transform.basis.elements[0][1];
	push_constant.cam_transform[5] = p_transform.basis.elements[1][1];
	push_constant.cam_transform[6] = p_transform.basis.elements[2][1];
	push_constant.cam_transform[7] = 0;
	push_constant.cam_transform[8] = p_transform.basis.elements[0][2];
	push_constant.cam_transform[9] = p_transform.basis.elements[1][2];
	push_constant.cam_transform[10] = p_transform.basis.elements[2][2];
	push_constant.cam_transform[11] = 0;
	push_constant.cam_transform[12] = p_transform.origin.x;
	push_constant.cam_transform[13] = p_transform.origin.y;
	push_constant.cam_transform[14] = p_transform.origin.z;
	push_constant.cam_transform[15] = 1;

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::DebugPushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->width, rb->height, 1, 8, 8, 1);
	RD::get_singleton()->compute_list_end();

	Size2 rtsize = storage->render_target_get_size(rb->render_target);
	storage->get_effects()->copy_to_fb_rect(rb->texture, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize), true);
}

RID RasterizerSceneRD::render_buffers_get_back_buffer_texture(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	if (!rb->blur[0].texture.is_valid()) {
		return RID(); //not valid at the moment
	}
	return rb->blur[0].texture;
}

RID RasterizerSceneRD::render_buffers_get_ao_texture(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());

	return rb->ssao.ao_full.is_valid() ? rb->ssao.ao_full : rb->ssao.ao[0];
}

RID RasterizerSceneRD::render_buffers_get_gi_probe_buffer(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	if (rb->giprobe_buffer.is_null()) {
		rb->giprobe_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(GI::GIProbeData) * RenderBuffers::MAX_GIPROBES);
	}
	return rb->giprobe_buffer;
}

RID RasterizerSceneRD::render_buffers_get_default_gi_probe_buffer() {
	return default_giprobe_buffer;
}

uint32_t RasterizerSceneRD::render_buffers_get_sdfgi_cascade_count(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0);
	ERR_FAIL_COND_V(!rb->sdfgi, 0);

	return rb->sdfgi->cascades.size();
}
bool RasterizerSceneRD::render_buffers_is_sdfgi_enabled(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, false);

	return rb->sdfgi != nullptr;
}
RID RasterizerSceneRD::render_buffers_get_sdfgi_irradiance_probes(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	ERR_FAIL_COND_V(!rb->sdfgi, RID());

	return rb->sdfgi->lightprobe_texture;
}

Vector3 RasterizerSceneRD::render_buffers_get_sdfgi_cascade_offset(RID p_render_buffers, uint32_t p_cascade) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, Vector3());
	ERR_FAIL_COND_V(!rb->sdfgi, Vector3());
	ERR_FAIL_UNSIGNED_INDEX_V(p_cascade, rb->sdfgi->cascades.size(), Vector3());

	return Vector3((Vector3i(1, 1, 1) * -int32_t(rb->sdfgi->cascade_size >> 1) + rb->sdfgi->cascades[p_cascade].position)) * rb->sdfgi->cascades[p_cascade].cell_size;
}

Vector3i RasterizerSceneRD::render_buffers_get_sdfgi_cascade_probe_offset(RID p_render_buffers, uint32_t p_cascade) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, Vector3i());
	ERR_FAIL_COND_V(!rb->sdfgi, Vector3i());
	ERR_FAIL_UNSIGNED_INDEX_V(p_cascade, rb->sdfgi->cascades.size(), Vector3i());
	int32_t probe_divisor = rb->sdfgi->cascade_size / SDFGI::PROBE_DIVISOR;

	return rb->sdfgi->cascades[p_cascade].position / probe_divisor;
}

float RasterizerSceneRD::render_buffers_get_sdfgi_normal_bias(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0);
	ERR_FAIL_COND_V(!rb->sdfgi, 0);

	return rb->sdfgi->normal_bias;
}
float RasterizerSceneRD::render_buffers_get_sdfgi_cascade_probe_size(RID p_render_buffers, uint32_t p_cascade) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0);
	ERR_FAIL_COND_V(!rb->sdfgi, 0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_cascade, rb->sdfgi->cascades.size(), 0);

	return float(rb->sdfgi->cascade_size) * rb->sdfgi->cascades[p_cascade].cell_size / float(rb->sdfgi->probe_axis_count - 1);
}
uint32_t RasterizerSceneRD::render_buffers_get_sdfgi_cascade_probe_count(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0);
	ERR_FAIL_COND_V(!rb->sdfgi, 0);

	return rb->sdfgi->probe_axis_count;
}

uint32_t RasterizerSceneRD::render_buffers_get_sdfgi_cascade_size(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0);
	ERR_FAIL_COND_V(!rb->sdfgi, 0);

	return rb->sdfgi->cascade_size;
}

bool RasterizerSceneRD::render_buffers_is_sdfgi_using_occlusion(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, false);
	ERR_FAIL_COND_V(!rb->sdfgi, false);

	return rb->sdfgi->uses_occlusion;
}

float RasterizerSceneRD::render_buffers_get_sdfgi_energy(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0);
	ERR_FAIL_COND_V(!rb->sdfgi, false);

	return rb->sdfgi->energy;
}
RID RasterizerSceneRD::render_buffers_get_sdfgi_occlusion_texture(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	ERR_FAIL_COND_V(!rb->sdfgi, RID());

	return rb->sdfgi->occlusion_texture;
}

bool RasterizerSceneRD::render_buffers_has_volumetric_fog(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, false);

	return rb->volumetric_fog != nullptr;
}
RID RasterizerSceneRD::render_buffers_get_volumetric_fog_texture(RID p_render_buffers) {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb || !rb->volumetric_fog, RID());

	return rb->volumetric_fog->fog_map;
}

RID RasterizerSceneRD::render_buffers_get_volumetric_fog_sky_uniform_set(RID p_render_buffers) {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());

	if (!rb->volumetric_fog) {
		return RID();
	}

	return rb->volumetric_fog->sky_uniform_set;
}

float RasterizerSceneRD::render_buffers_get_volumetric_fog_end(RID p_render_buffers) {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb || !rb->volumetric_fog, 0);
	return rb->volumetric_fog->length;
}
float RasterizerSceneRD::render_buffers_get_volumetric_fog_detail_spread(RID p_render_buffers) {
	const RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb || !rb->volumetric_fog, 0);
	return rb->volumetric_fog->spread;
}

void RasterizerSceneRD::render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_width, int p_height, RS::ViewportMSAA p_msaa, RenderingServer::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_debanding) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	rb->width = p_width;
	rb->height = p_height;
	rb->render_target = p_render_target;
	rb->msaa = p_msaa;
	rb->screen_space_aa = p_screen_space_aa;
	rb->use_debanding = p_use_debanding;
	_free_render_buffer_data(rb);

	{
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		tf.width = rb->width;
		tf.height = rb->height;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		if (rb->msaa != RS::VIEWPORT_MSAA_DISABLED) {
			tf.usage_bits |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		} else {
			tf.usage_bits |= RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
		}

		rb->texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	{
		RD::TextureFormat tf;
		if (rb->msaa == RS::VIEWPORT_MSAA_DISABLED) {
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D24_UNORM_S8_UINT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D24_UNORM_S8_UINT : RD::DATA_FORMAT_D32_SFLOAT_S8_UINT;
		} else {
			tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		}

		tf.width = p_width;
		tf.height = p_height;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;

		if (rb->msaa != RS::VIEWPORT_MSAA_DISABLED) {
			tf.usage_bits |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		} else {
			tf.usage_bits |= RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		}

		rb->depth_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	rb->data->configure(rb->texture, rb->depth_texture, p_width, p_height, p_msaa);
	_render_buffers_uniform_set_changed(p_render_buffers);
}

void RasterizerSceneRD::sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) {
	sss_quality = p_quality;
}

RS::SubSurfaceScatteringQuality RasterizerSceneRD::sub_surface_scattering_get_quality() const {
	return sss_quality;
}

void RasterizerSceneRD::sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) {
	sss_scale = p_scale;
	sss_depth_scale = p_depth_scale;
}

void RasterizerSceneRD::shadows_quality_set(RS::ShadowQuality p_quality) {
	ERR_FAIL_INDEX_MSG(p_quality, RS::SHADOW_QUALITY_MAX, "Shadow quality too high, please see RenderingServer's ShadowQuality enum");

	if (shadows_quality != p_quality) {
		shadows_quality = p_quality;

		switch (shadows_quality) {
			case RS::SHADOW_QUALITY_HARD: {
				penumbra_shadow_samples = 4;
				soft_shadow_samples = 1;
				shadows_quality_radius = 1.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_LOW: {
				penumbra_shadow_samples = 8;
				soft_shadow_samples = 4;
				shadows_quality_radius = 2.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_MEDIUM: {
				penumbra_shadow_samples = 12;
				soft_shadow_samples = 8;
				shadows_quality_radius = 2.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_HIGH: {
				penumbra_shadow_samples = 24;
				soft_shadow_samples = 16;
				shadows_quality_radius = 3.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_ULTRA: {
				penumbra_shadow_samples = 32;
				soft_shadow_samples = 32;
				shadows_quality_radius = 4.0;
			} break;
			case RS::SHADOW_QUALITY_MAX:
				break;
		}
		get_vogel_disk(penumbra_shadow_kernel, penumbra_shadow_samples);
		get_vogel_disk(soft_shadow_kernel, soft_shadow_samples);
	}
}

void RasterizerSceneRD::directional_shadow_quality_set(RS::ShadowQuality p_quality) {
	ERR_FAIL_INDEX_MSG(p_quality, RS::SHADOW_QUALITY_MAX, "Shadow quality too high, please see RenderingServer's ShadowQuality enum");

	if (directional_shadow_quality != p_quality) {
		directional_shadow_quality = p_quality;

		switch (directional_shadow_quality) {
			case RS::SHADOW_QUALITY_HARD: {
				directional_penumbra_shadow_samples = 4;
				directional_soft_shadow_samples = 1;
				directional_shadow_quality_radius = 1.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_LOW: {
				directional_penumbra_shadow_samples = 8;
				directional_soft_shadow_samples = 4;
				directional_shadow_quality_radius = 2.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_MEDIUM: {
				directional_penumbra_shadow_samples = 12;
				directional_soft_shadow_samples = 8;
				directional_shadow_quality_radius = 2.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_HIGH: {
				directional_penumbra_shadow_samples = 24;
				directional_soft_shadow_samples = 16;
				directional_shadow_quality_radius = 3.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_ULTRA: {
				directional_penumbra_shadow_samples = 32;
				directional_soft_shadow_samples = 32;
				directional_shadow_quality_radius = 4.0;
			} break;
			case RS::SHADOW_QUALITY_MAX:
				break;
		}
		get_vogel_disk(directional_penumbra_shadow_kernel, directional_penumbra_shadow_samples);
		get_vogel_disk(directional_soft_shadow_kernel, directional_soft_shadow_samples);
	}
}

int RasterizerSceneRD::get_roughness_layers() const {
	return roughness_layers;
}

bool RasterizerSceneRD::is_using_radiance_cubemap_array() const {
	return sky_use_cubemap_array;
}

RasterizerSceneRD::RenderBufferData *RasterizerSceneRD::render_buffers_get_data(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, nullptr);
	return rb->data;
}

void RasterizerSceneRD::_setup_reflections(RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, const Transform &p_camera_inverse_transform, RID p_environment) {
	for (int i = 0; i < p_reflection_probe_cull_count; i++) {
		RID rpi = p_reflection_probe_cull_result[i];

		if (i >= (int)cluster.max_reflections) {
			reflection_probe_instance_set_render_index(rpi, 0); //invalid, but something needs to be set
			continue;
		}

		reflection_probe_instance_set_render_index(rpi, i);

		RID base_probe = reflection_probe_instance_get_probe(rpi);

		Cluster::ReflectionData &reflection_ubo = cluster.reflections[i];

		Vector3 extents = storage->reflection_probe_get_extents(base_probe);

		reflection_ubo.box_extents[0] = extents.x;
		reflection_ubo.box_extents[1] = extents.y;
		reflection_ubo.box_extents[2] = extents.z;
		reflection_ubo.index = reflection_probe_instance_get_atlas_index(rpi);

		Vector3 origin_offset = storage->reflection_probe_get_origin_offset(base_probe);

		reflection_ubo.box_offset[0] = origin_offset.x;
		reflection_ubo.box_offset[1] = origin_offset.y;
		reflection_ubo.box_offset[2] = origin_offset.z;
		reflection_ubo.mask = storage->reflection_probe_get_cull_mask(base_probe);

		float intensity = storage->reflection_probe_get_intensity(base_probe);
		bool interior = storage->reflection_probe_is_interior(base_probe);
		bool box_projection = storage->reflection_probe_is_box_projection(base_probe);

		reflection_ubo.params[0] = intensity;
		reflection_ubo.params[1] = 0;
		reflection_ubo.params[2] = interior ? 1.0 : 0.0;
		reflection_ubo.params[3] = box_projection ? 1.0 : 0.0;

		Color ambient_linear = storage->reflection_probe_get_ambient_color(base_probe).to_linear();
		float interior_ambient_energy = storage->reflection_probe_get_ambient_color_energy(base_probe);
		uint32_t ambient_mode = storage->reflection_probe_get_ambient_mode(base_probe);
		reflection_ubo.ambient[0] = ambient_linear.r * interior_ambient_energy;
		reflection_ubo.ambient[1] = ambient_linear.g * interior_ambient_energy;
		reflection_ubo.ambient[2] = ambient_linear.b * interior_ambient_energy;
		reflection_ubo.ambient_mode = ambient_mode;

		Transform transform = reflection_probe_instance_get_transform(rpi);
		Transform proj = (p_camera_inverse_transform * transform).inverse();
		RasterizerStorageRD::store_transform(proj, reflection_ubo.local_matrix);

		cluster.builder.add_reflection_probe(transform, extents);

		reflection_probe_instance_set_render_pass(rpi, RSG::rasterizer->get_frame_number());
	}

	if (p_reflection_probe_cull_count) {
		RD::get_singleton()->buffer_update(cluster.reflection_buffer, 0, MIN(cluster.max_reflections, (unsigned int)p_reflection_probe_cull_count) * sizeof(ReflectionData), cluster.reflections, true);
	}
}

void RasterizerSceneRD::_setup_lights(RID *p_light_cull_result, int p_light_cull_count, const Transform &p_camera_inverse_transform, RID p_shadow_atlas, bool p_using_shadows, uint32_t &r_directional_light_count, uint32_t &r_positional_light_count) {
	uint32_t light_count = 0;
	r_directional_light_count = 0;
	r_positional_light_count = 0;
	sky_scene_state.ubo.directional_light_count = 0;

	for (int i = 0; i < p_light_cull_count; i++) {
		RID li = p_light_cull_result[i];
		RID base = light_instance_get_base_light(li);

		ERR_CONTINUE(base.is_null());

		RS::LightType type = storage->light_get_type(base);
		switch (type) {
			case RS::LIGHT_DIRECTIONAL: {
				if (r_directional_light_count >= cluster.max_directional_lights) {
					continue;
				}

				Cluster::DirectionalLightData &light_data = cluster.directional_lights[r_directional_light_count];

				Transform light_transform = light_instance_get_base_transform(li);

				Vector3 direction = p_camera_inverse_transform.basis.xform(light_transform.basis.xform(Vector3(0, 0, 1))).normalized();

				light_data.direction[0] = direction.x;
				light_data.direction[1] = direction.y;
				light_data.direction[2] = direction.z;

				float sign = storage->light_is_negative(base) ? -1 : 1;

				light_data.energy = sign * storage->light_get_param(base, RS::LIGHT_PARAM_ENERGY) * Math_PI;

				Color linear_col = storage->light_get_color(base).to_linear();
				light_data.color[0] = linear_col.r;
				light_data.color[1] = linear_col.g;
				light_data.color[2] = linear_col.b;

				light_data.specular = storage->light_get_param(base, RS::LIGHT_PARAM_SPECULAR);
				light_data.mask = storage->light_get_cull_mask(base);

				float size = storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);

				light_data.size = 1.0 - Math::cos(Math::deg2rad(size)); //angle to cosine offset

				Color shadow_col = storage->light_get_shadow_color(base).to_linear();

				if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_PSSM_SPLITS) {
					light_data.shadow_color1[0] = 1.0;
					light_data.shadow_color1[1] = 0.0;
					light_data.shadow_color1[2] = 0.0;
					light_data.shadow_color1[3] = 1.0;
					light_data.shadow_color2[0] = 0.0;
					light_data.shadow_color2[1] = 1.0;
					light_data.shadow_color2[2] = 0.0;
					light_data.shadow_color2[3] = 1.0;
					light_data.shadow_color3[0] = 0.0;
					light_data.shadow_color3[1] = 0.0;
					light_data.shadow_color3[2] = 1.0;
					light_data.shadow_color3[3] = 1.0;
					light_data.shadow_color4[0] = 1.0;
					light_data.shadow_color4[1] = 1.0;
					light_data.shadow_color4[2] = 0.0;
					light_data.shadow_color4[3] = 1.0;

				} else {
					light_data.shadow_color1[0] = shadow_col.r;
					light_data.shadow_color1[1] = shadow_col.g;
					light_data.shadow_color1[2] = shadow_col.b;
					light_data.shadow_color1[3] = 1.0;
					light_data.shadow_color2[0] = shadow_col.r;
					light_data.shadow_color2[1] = shadow_col.g;
					light_data.shadow_color2[2] = shadow_col.b;
					light_data.shadow_color2[3] = 1.0;
					light_data.shadow_color3[0] = shadow_col.r;
					light_data.shadow_color3[1] = shadow_col.g;
					light_data.shadow_color3[2] = shadow_col.b;
					light_data.shadow_color3[3] = 1.0;
					light_data.shadow_color4[0] = shadow_col.r;
					light_data.shadow_color4[1] = shadow_col.g;
					light_data.shadow_color4[2] = shadow_col.b;
					light_data.shadow_color4[3] = 1.0;
				}

				light_data.shadow_enabled = p_using_shadows && storage->light_has_shadow(base);

				float angular_diameter = storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);
				if (angular_diameter > 0.0) {
					// I know tan(0) is 0, but let's not risk it with numerical precision.
					// technically this will keep expanding until reaching the sun, but all we care
					// is expand until we reach the radius of the near plane (there can't be more occluders than that)
					angular_diameter = Math::tan(Math::deg2rad(angular_diameter));
				} else {
					angular_diameter = 0.0;
				}

				if (light_data.shadow_enabled) {
					RS::LightDirectionalShadowMode smode = storage->light_directional_get_shadow_mode(base);

					int limit = smode == RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL ? 0 : (smode == RS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS ? 1 : 3);
					light_data.blend_splits = storage->light_directional_get_blend_splits(base);
					for (int j = 0; j < 4; j++) {
						Rect2 atlas_rect = light_instance_get_directional_shadow_atlas_rect(li, j);
						CameraMatrix matrix = light_instance_get_shadow_camera(li, j);
						float split = light_instance_get_directional_shadow_split(li, MIN(limit, j));

						CameraMatrix bias;
						bias.set_light_bias();
						CameraMatrix rectm;
						rectm.set_light_atlas_rect(atlas_rect);

						Transform modelview = (p_camera_inverse_transform * light_instance_get_shadow_transform(li, j)).inverse();

						CameraMatrix shadow_mtx = rectm * bias * matrix * modelview;
						light_data.shadow_split_offsets[j] = split;
						float bias_scale = light_instance_get_shadow_bias_scale(li, j);
						light_data.shadow_bias[j] = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BIAS) * bias_scale;
						light_data.shadow_normal_bias[j] = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS) * light_instance_get_directional_shadow_texel_size(li, j);
						light_data.shadow_transmittance_bias[j] = storage->light_get_transmittance_bias(base) * bias_scale;
						light_data.shadow_z_range[j] = light_instance_get_shadow_range(li, j);
						light_data.shadow_range_begin[j] = light_instance_get_shadow_range_begin(li, j);
						RasterizerStorageRD::store_camera(shadow_mtx, light_data.shadow_matrices[j]);

						Vector2 uv_scale = light_instance_get_shadow_uv_scale(li, j);
						uv_scale *= atlas_rect.size; //adapt to atlas size
						switch (j) {
							case 0: {
								light_data.uv_scale1[0] = uv_scale.x;
								light_data.uv_scale1[1] = uv_scale.y;
							} break;
							case 1: {
								light_data.uv_scale2[0] = uv_scale.x;
								light_data.uv_scale2[1] = uv_scale.y;
							} break;
							case 2: {
								light_data.uv_scale3[0] = uv_scale.x;
								light_data.uv_scale3[1] = uv_scale.y;
							} break;
							case 3: {
								light_data.uv_scale4[0] = uv_scale.x;
								light_data.uv_scale4[1] = uv_scale.y;
							} break;
						}
					}

					float fade_start = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_FADE_START);
					light_data.fade_from = -light_data.shadow_split_offsets[3] * MIN(fade_start, 0.999); //using 1.0 would break smoothstep
					light_data.fade_to = -light_data.shadow_split_offsets[3];
					light_data.shadow_volumetric_fog_fade = 1.0 / storage->light_get_shadow_volumetric_fog_fade(base);

					light_data.soft_shadow_scale = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BLUR);
					light_data.softshadow_angle = angular_diameter;

					if (angular_diameter <= 0.0) {
						light_data.soft_shadow_scale *= directional_shadow_quality_radius_get(); // Only use quality radius for PCF
					}
				}

				//	Copy to SkyDirectionalLightData
				if (r_directional_light_count < sky_scene_state.max_directional_lights) {
					SkyDirectionalLightData &sky_light_data = sky_scene_state.directional_lights[r_directional_light_count];

					Vector3 world_direction = light_transform.basis.xform(Vector3(0, 0, 1)).normalized();

					sky_light_data.direction[0] = world_direction.x;
					sky_light_data.direction[1] = world_direction.y;
					sky_light_data.direction[2] = -world_direction.z;

					sky_light_data.energy = light_data.energy / Math_PI;

					sky_light_data.color[0] = light_data.color[0];
					sky_light_data.color[1] = light_data.color[1];
					sky_light_data.color[2] = light_data.color[2];

					sky_light_data.enabled = true;
					sky_light_data.size = angular_diameter;
					sky_scene_state.ubo.directional_light_count++;
				}

				r_directional_light_count++;
			} break;
			case RS::LIGHT_SPOT:
			case RS::LIGHT_OMNI: {
				if (light_count >= cluster.max_lights) {
					continue;
				}

				Transform light_transform = light_instance_get_base_transform(li);

				Cluster::LightData &light_data = cluster.lights[light_count];
				cluster.lights_instances[light_count] = li;

				float sign = storage->light_is_negative(base) ? -1 : 1;
				Color linear_col = storage->light_get_color(base).to_linear();

				light_data.attenuation_energy[0] = Math::make_half_float(storage->light_get_param(base, RS::LIGHT_PARAM_ATTENUATION));
				light_data.attenuation_energy[1] = Math::make_half_float(sign * storage->light_get_param(base, RS::LIGHT_PARAM_ENERGY) * Math_PI);

				light_data.color_specular[0] = MIN(uint32_t(linear_col.r * 255), 255);
				light_data.color_specular[1] = MIN(uint32_t(linear_col.g * 255), 255);
				light_data.color_specular[2] = MIN(uint32_t(linear_col.b * 255), 255);
				light_data.color_specular[3] = MIN(uint32_t(storage->light_get_param(base, RS::LIGHT_PARAM_SPECULAR) * 255), 255);

				float radius = MAX(0.001, storage->light_get_param(base, RS::LIGHT_PARAM_RANGE));
				light_data.inv_radius = 1.0 / radius;

				Vector3 pos = p_camera_inverse_transform.xform(light_transform.origin);

				light_data.position[0] = pos.x;
				light_data.position[1] = pos.y;
				light_data.position[2] = pos.z;

				Vector3 direction = p_camera_inverse_transform.basis.xform(light_transform.basis.xform(Vector3(0, 0, -1))).normalized();

				light_data.direction[0] = direction.x;
				light_data.direction[1] = direction.y;
				light_data.direction[2] = direction.z;

				float size = storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);

				light_data.size = size;

				light_data.cone_attenuation_angle[0] = Math::make_half_float(storage->light_get_param(base, RS::LIGHT_PARAM_SPOT_ATTENUATION));
				float spot_angle = storage->light_get_param(base, RS::LIGHT_PARAM_SPOT_ANGLE);
				light_data.cone_attenuation_angle[1] = Math::make_half_float(Math::cos(Math::deg2rad(spot_angle)));

				light_data.mask = storage->light_get_cull_mask(base);

				light_data.atlas_rect[0] = 0;
				light_data.atlas_rect[1] = 0;
				light_data.atlas_rect[2] = 0;
				light_data.atlas_rect[3] = 0;

				RID projector = storage->light_get_projector(base);

				if (projector.is_valid()) {
					Rect2 rect = storage->decal_atlas_get_texture_rect(projector);

					if (type == RS::LIGHT_SPOT) {
						light_data.projector_rect[0] = rect.position.x;
						light_data.projector_rect[1] = rect.position.y + rect.size.height; //flip because shadow is flipped
						light_data.projector_rect[2] = rect.size.width;
						light_data.projector_rect[3] = -rect.size.height;
					} else {
						light_data.projector_rect[0] = rect.position.x;
						light_data.projector_rect[1] = rect.position.y;
						light_data.projector_rect[2] = rect.size.width;
						light_data.projector_rect[3] = rect.size.height * 0.5; //used by dp, so needs to be half
					}
				} else {
					light_data.projector_rect[0] = 0;
					light_data.projector_rect[1] = 0;
					light_data.projector_rect[2] = 0;
					light_data.projector_rect[3] = 0;
				}

				if (p_using_shadows && p_shadow_atlas.is_valid() && shadow_atlas_owns_light_instance(p_shadow_atlas, li)) {
					// fill in the shadow information

					Color shadow_color = storage->light_get_shadow_color(base);

					light_data.shadow_color_enabled[0] = MIN(uint32_t(shadow_color.r * 255), 255);
					light_data.shadow_color_enabled[1] = MIN(uint32_t(shadow_color.g * 255), 255);
					light_data.shadow_color_enabled[2] = MIN(uint32_t(shadow_color.b * 255), 255);
					light_data.shadow_color_enabled[3] = 255;

					if (type == RS::LIGHT_SPOT) {
						light_data.shadow_bias = (storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BIAS) * radius / 10.0);
						float shadow_texel_size = Math::tan(Math::deg2rad(spot_angle)) * radius * 2.0;
						shadow_texel_size *= light_instance_get_shadow_texel_size(li, p_shadow_atlas);

						light_data.shadow_normal_bias = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS) * shadow_texel_size;

					} else { //omni
						light_data.shadow_bias = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BIAS) * radius / 10.0;
						float shadow_texel_size = light_instance_get_shadow_texel_size(li, p_shadow_atlas);
						light_data.shadow_normal_bias = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS) * shadow_texel_size * 2.0; // applied in -1 .. 1 space
					}

					light_data.transmittance_bias = storage->light_get_transmittance_bias(base);

					Rect2 rect = light_instance_get_shadow_atlas_rect(li, p_shadow_atlas);

					light_data.atlas_rect[0] = rect.position.x;
					light_data.atlas_rect[1] = rect.position.y;
					light_data.atlas_rect[2] = rect.size.width;
					light_data.atlas_rect[3] = rect.size.height;

					light_data.soft_shadow_scale = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BLUR);
					light_data.shadow_volumetric_fog_fade = 1.0 / storage->light_get_shadow_volumetric_fog_fade(base);

					if (type == RS::LIGHT_OMNI) {
						light_data.atlas_rect[3] *= 0.5; //one paraboloid on top of another
						Transform proj = (p_camera_inverse_transform * light_transform).inverse();

						RasterizerStorageRD::store_transform(proj, light_data.shadow_matrix);

						if (size > 0.0) {
							light_data.soft_shadow_size = size;
						} else {
							light_data.soft_shadow_size = 0.0;
							light_data.soft_shadow_scale *= shadows_quality_radius_get(); // Only use quality radius for PCF
						}

					} else if (type == RS::LIGHT_SPOT) {
						Transform modelview = (p_camera_inverse_transform * light_transform).inverse();
						CameraMatrix bias;
						bias.set_light_bias();

						CameraMatrix shadow_mtx = bias * light_instance_get_shadow_camera(li, 0) * modelview;
						RasterizerStorageRD::store_camera(shadow_mtx, light_data.shadow_matrix);

						if (size > 0.0) {
							CameraMatrix cm = light_instance_get_shadow_camera(li, 0);
							float half_np = cm.get_z_near() * Math::tan(Math::deg2rad(spot_angle));
							light_data.soft_shadow_size = (size * 0.5 / radius) / (half_np / cm.get_z_near()) * rect.size.width;
						} else {
							light_data.soft_shadow_size = 0.0;
							light_data.soft_shadow_scale *= shadows_quality_radius_get(); // Only use quality radius for PCF
						}
					}
				} else {
					light_data.shadow_color_enabled[3] = 0;
				}

				light_instance_set_index(li, light_count);

				cluster.builder.add_light(type == RS::LIGHT_SPOT ? LightClusterBuilder::LIGHT_TYPE_SPOT : LightClusterBuilder::LIGHT_TYPE_OMNI, light_transform, radius, spot_angle);

				light_count++;
				r_positional_light_count++;
			} break;
		}

		light_instance_set_render_pass(li, RSG::rasterizer->get_frame_number());

		//update UBO for forward rendering, blit to texture for clustered
	}

	if (light_count) {
		RD::get_singleton()->buffer_update(cluster.light_buffer, 0, sizeof(Cluster::LightData) * light_count, cluster.lights, true);
	}

	if (r_directional_light_count) {
		RD::get_singleton()->buffer_update(cluster.directional_light_buffer, 0, sizeof(Cluster::DirectionalLightData) * r_directional_light_count, cluster.directional_lights, true);
	}
}

void RasterizerSceneRD::_setup_decals(const RID *p_decal_instances, int p_decal_count, const Transform &p_camera_inverse_xform) {
	Transform uv_xform;
	uv_xform.basis.scale(Vector3(2.0, 1.0, 2.0));
	uv_xform.origin = Vector3(-1.0, 0.0, -1.0);

	p_decal_count = MIN((uint32_t)p_decal_count, cluster.max_decals);
	int idx = 0;
	for (int i = 0; i < p_decal_count; i++) {
		RID di = p_decal_instances[i];
		RID decal = decal_instance_get_base(di);

		Transform xform = decal_instance_get_transform(di);

		float fade = 1.0;

		if (storage->decal_is_distance_fade_enabled(decal)) {
			real_t distance = -p_camera_inverse_xform.xform(xform.origin).z;
			float fade_begin = storage->decal_get_distance_fade_begin(decal);
			float fade_length = storage->decal_get_distance_fade_length(decal);

			if (distance > fade_begin) {
				if (distance > fade_begin + fade_length) {
					continue; // do not use this decal, its invisible
				}

				fade = 1.0 - (distance - fade_begin) / fade_length;
			}
		}

		Cluster::DecalData &dd = cluster.decals[idx];

		Vector3 decal_extents = storage->decal_get_extents(decal);

		Transform scale_xform;
		scale_xform.basis.scale(Vector3(decal_extents.x, decal_extents.y, decal_extents.z));
		Transform to_decal_xform = (p_camera_inverse_xform * decal_instance_get_transform(di) * scale_xform * uv_xform).affine_inverse();
		RasterizerStorageRD::store_transform(to_decal_xform, dd.xform);

		Vector3 normal = xform.basis.get_axis(Vector3::AXIS_Y).normalized();
		normal = p_camera_inverse_xform.basis.xform(normal); //camera is normalized, so fine

		dd.normal[0] = normal.x;
		dd.normal[1] = normal.y;
		dd.normal[2] = normal.z;
		dd.normal_fade = storage->decal_get_normal_fade(decal);

		RID albedo_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_ALBEDO);
		RID emission_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_EMISSION);
		if (albedo_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(albedo_tex);
			dd.albedo_rect[0] = rect.position.x;
			dd.albedo_rect[1] = rect.position.y;
			dd.albedo_rect[2] = rect.size.x;
			dd.albedo_rect[3] = rect.size.y;
		} else {
			if (!emission_tex.is_valid()) {
				continue; //no albedo, no emission, no decal.
			}
			dd.albedo_rect[0] = 0;
			dd.albedo_rect[1] = 0;
			dd.albedo_rect[2] = 0;
			dd.albedo_rect[3] = 0;
		}

		RID normal_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_NORMAL);

		if (normal_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(normal_tex);
			dd.normal_rect[0] = rect.position.x;
			dd.normal_rect[1] = rect.position.y;
			dd.normal_rect[2] = rect.size.x;
			dd.normal_rect[3] = rect.size.y;

			Basis normal_xform = p_camera_inverse_xform.basis * xform.basis.orthonormalized();
			RasterizerStorageRD::store_basis_3x4(normal_xform, dd.normal_xform);
		} else {
			dd.normal_rect[0] = 0;
			dd.normal_rect[1] = 0;
			dd.normal_rect[2] = 0;
			dd.normal_rect[3] = 0;
		}

		RID orm_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_ORM);
		if (orm_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(orm_tex);
			dd.orm_rect[0] = rect.position.x;
			dd.orm_rect[1] = rect.position.y;
			dd.orm_rect[2] = rect.size.x;
			dd.orm_rect[3] = rect.size.y;
		} else {
			dd.orm_rect[0] = 0;
			dd.orm_rect[1] = 0;
			dd.orm_rect[2] = 0;
			dd.orm_rect[3] = 0;
		}

		if (emission_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(emission_tex);
			dd.emission_rect[0] = rect.position.x;
			dd.emission_rect[1] = rect.position.y;
			dd.emission_rect[2] = rect.size.x;
			dd.emission_rect[3] = rect.size.y;
		} else {
			dd.emission_rect[0] = 0;
			dd.emission_rect[1] = 0;
			dd.emission_rect[2] = 0;
			dd.emission_rect[3] = 0;
		}

		Color modulate = storage->decal_get_modulate(decal);
		dd.modulate[0] = modulate.r;
		dd.modulate[1] = modulate.g;
		dd.modulate[2] = modulate.b;
		dd.modulate[3] = modulate.a * fade;
		dd.emission_energy = storage->decal_get_emission_energy(decal) * fade;
		dd.albedo_mix = storage->decal_get_albedo_mix(decal);
		dd.mask = storage->decal_get_cull_mask(decal);
		dd.upper_fade = storage->decal_get_upper_fade(decal);
		dd.lower_fade = storage->decal_get_lower_fade(decal);

		cluster.builder.add_decal(xform, decal_extents);

		idx++;
	}

	if (idx > 0) {
		RD::get_singleton()->buffer_update(cluster.decal_buffer, 0, sizeof(Cluster::DecalData) * idx, cluster.decals, true);
	}
}

void RasterizerSceneRD::_volumetric_fog_erase(RenderBuffers *rb) {
	ERR_FAIL_COND(!rb->volumetric_fog);

	RD::get_singleton()->free(rb->volumetric_fog->light_density_map);
	RD::get_singleton()->free(rb->volumetric_fog->fog_map);

	if (rb->volumetric_fog->uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->uniform_set)) {
		RD::get_singleton()->free(rb->volumetric_fog->uniform_set);
	}
	if (rb->volumetric_fog->uniform_set2.is_valid() && RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->uniform_set2)) {
		RD::get_singleton()->free(rb->volumetric_fog->uniform_set2);
	}
	if (rb->volumetric_fog->sdfgi_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->sdfgi_uniform_set)) {
		RD::get_singleton()->free(rb->volumetric_fog->sdfgi_uniform_set);
	}
	if (rb->volumetric_fog->sky_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->sky_uniform_set)) {
		RD::get_singleton()->free(rb->volumetric_fog->sky_uniform_set);
	}

	memdelete(rb->volumetric_fog);

	rb->volumetric_fog = nullptr;
}

void RasterizerSceneRD::_allocate_shadow_shrink_stages(RID p_base, int p_base_size, Vector<ShadowShrinkStage> &shrink_stages, uint32_t p_target_size) {
	//create fog mipmaps
	uint32_t fog_texture_size = p_target_size;
	uint32_t base_texture_size = p_base_size;

	ShadowShrinkStage first;
	first.size = base_texture_size;
	first.texture = p_base;
	shrink_stages.push_back(first); //put depth first in case we dont find smaller ones

	while (fog_texture_size < base_texture_size) {
		base_texture_size = MAX(base_texture_size / 8, fog_texture_size);

		ShadowShrinkStage s;
		s.size = base_texture_size;

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		tf.width = base_texture_size;
		tf.height = base_texture_size;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

		if (base_texture_size == fog_texture_size) {
			s.filter_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
			tf.usage_bits |= RD::TEXTURE_USAGE_SAMPLING_BIT;
		}

		s.texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

		shrink_stages.push_back(s);
	}
}

void RasterizerSceneRD::_clear_shadow_shrink_stages(Vector<ShadowShrinkStage> &shrink_stages) {
	for (int i = 1; i < shrink_stages.size(); i++) {
		RD::get_singleton()->free(shrink_stages[i].texture);
		if (shrink_stages[i].filter_texture.is_valid()) {
			RD::get_singleton()->free(shrink_stages[i].filter_texture);
		}
	}
	shrink_stages.clear();
}

void RasterizerSceneRD::_update_volumetric_fog(RID p_render_buffers, RID p_environment, const CameraMatrix &p_cam_projection, const Transform &p_cam_transform, RID p_shadow_atlas, int p_directional_light_count, bool p_use_directional_shadows, int p_positional_light_count, int p_gi_probe_count) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);
	Environment *env = environment_owner.getornull(p_environment);

	float ratio = float(rb->width) / float((rb->width + rb->height) / 2);
	uint32_t target_width = uint32_t(float(volumetric_fog_size) * ratio);
	uint32_t target_height = uint32_t(float(volumetric_fog_size) / ratio);

	if (rb->volumetric_fog) {
		//validate
		if (!env || !env->volumetric_fog_enabled || rb->volumetric_fog->width != target_width || rb->volumetric_fog->height != target_height || rb->volumetric_fog->depth != volumetric_fog_depth) {
			_volumetric_fog_erase(rb);
			_render_buffers_uniform_set_changed(p_render_buffers);
		}
	}

	if (!env || !env->volumetric_fog_enabled) {
		//no reason to enable or update, bye
		return;
	}

	if (env && env->volumetric_fog_enabled && !rb->volumetric_fog) {
		//required volumetric fog but not existing, create
		rb->volumetric_fog = memnew(VolumetricFog);
		rb->volumetric_fog->width = target_width;
		rb->volumetric_fog->height = target_height;
		rb->volumetric_fog->depth = volumetric_fog_depth;

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		tf.width = target_width;
		tf.height = target_height;
		tf.depth = volumetric_fog_depth;
		tf.type = RD::TEXTURE_TYPE_3D;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

		rb->volumetric_fog->light_density_map = RD::get_singleton()->texture_create(tf, RD::TextureView());

		tf.usage_bits |= RD::TEXTURE_USAGE_SAMPLING_BIT;

		rb->volumetric_fog->fog_map = RD::get_singleton()->texture_create(tf, RD::TextureView());
		_render_buffers_uniform_set_changed(p_render_buffers);

		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.binding = 0;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.ids.push_back(rb->volumetric_fog->fog_map);
			uniforms.push_back(u);
		}

		rb->volumetric_fog->sky_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sky_shader.default_shader_rd, SKY_SET_FOG);
	}

	//update directional shadow

	if (p_use_directional_shadows) {
		if (directional_shadow.shrink_stages.empty()) {
			if (rb->volumetric_fog->uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->uniform_set)) {
				//invalidate uniform set, we will need a new one
				RD::get_singleton()->free(rb->volumetric_fog->uniform_set);
				rb->volumetric_fog->uniform_set = RID();
			}
			_allocate_shadow_shrink_stages(directional_shadow.depth, directional_shadow.size, directional_shadow.shrink_stages, volumetric_fog_directional_shadow_shrink);
		}

		if (directional_shadow.shrink_stages.size() > 1) {
			RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
			for (int i = 1; i < directional_shadow.shrink_stages.size(); i++) {
				int32_t src_size = directional_shadow.shrink_stages[i - 1].size;
				int32_t dst_size = directional_shadow.shrink_stages[i].size;
				Rect2i r(0, 0, src_size, src_size);
				int32_t shrink_limit = 8 / (src_size / dst_size);

				storage->get_effects()->reduce_shadow(directional_shadow.shrink_stages[i - 1].texture, directional_shadow.shrink_stages[i].texture, Size2i(src_size, src_size), r, shrink_limit, compute_list);
				RD::get_singleton()->compute_list_add_barrier(compute_list);
				if (env->volumetric_fog_shadow_filter != RS::ENV_VOLUMETRIC_FOG_SHADOW_FILTER_DISABLED && directional_shadow.shrink_stages[i].filter_texture.is_valid()) {
					Rect2i rf(0, 0, dst_size, dst_size);
					storage->get_effects()->filter_shadow(directional_shadow.shrink_stages[i].texture, directional_shadow.shrink_stages[i].filter_texture, Size2i(dst_size, dst_size), rf, env->volumetric_fog_shadow_filter, compute_list);
				}
			}
			RD::get_singleton()->compute_list_end();
		}
	}

	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_shadow_atlas);

	if (shadow_atlas) {
		//shrink shadows that need to be shrunk

		bool force_shrink_shadows = false;

		if (shadow_atlas->shrink_stages.empty()) {
			if (rb->volumetric_fog->uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->uniform_set)) {
				//invalidate uniform set, we will need a new one
				RD::get_singleton()->free(rb->volumetric_fog->uniform_set);
				rb->volumetric_fog->uniform_set = RID();
			}
			_allocate_shadow_shrink_stages(shadow_atlas->depth, shadow_atlas->size, shadow_atlas->shrink_stages, volumetric_fog_positional_shadow_shrink);
			force_shrink_shadows = true;
		}

		if (rb->volumetric_fog->last_shadow_filter != env->volumetric_fog_shadow_filter) {
			//if shadow filter changed, invalidate caches
			rb->volumetric_fog->last_shadow_filter = env->volumetric_fog_shadow_filter;
			force_shrink_shadows = true;
		}

		cluster.lights_shadow_rect_cache_count = 0;

		for (int i = 0; i < p_positional_light_count; i++) {
			if (cluster.lights[i].shadow_color_enabled[3] > 127) {
				RID li = cluster.lights_instances[i];

				ERR_CONTINUE(!shadow_atlas->shadow_owners.has(li));

				uint32_t key = shadow_atlas->shadow_owners[li];

				uint32_t quadrant = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
				uint32_t shadow = key & ShadowAtlas::SHADOW_INDEX_MASK;

				ERR_CONTINUE((int)shadow >= shadow_atlas->quadrants[quadrant].shadows.size());

				ShadowAtlas::Quadrant::Shadow &s = shadow_atlas->quadrants[quadrant].shadows.write[shadow];

				if (!force_shrink_shadows && s.fog_version == s.version) {
					continue; //do not update, no need
				}

				s.fog_version = s.version;

				uint32_t quadrant_size = shadow_atlas->size >> 1;

				Rect2i atlas_rect;

				atlas_rect.position.x = (quadrant & 1) * quadrant_size;
				atlas_rect.position.y = (quadrant >> 1) * quadrant_size;

				uint32_t shadow_size = (quadrant_size / shadow_atlas->quadrants[quadrant].subdivision);
				atlas_rect.position.x += (shadow % shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;
				atlas_rect.position.y += (shadow / shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;

				atlas_rect.size.x = shadow_size;
				atlas_rect.size.y = shadow_size;

				cluster.lights_shadow_rect_cache[cluster.lights_shadow_rect_cache_count] = atlas_rect;

				cluster.lights_shadow_rect_cache_count++;

				if (cluster.lights_shadow_rect_cache_count == cluster.max_lights) {
					break; //light limit reached
				}
			}
		}

		if (cluster.lights_shadow_rect_cache_count > 0) {
			//there are shadows to be shrunk, try to do them in parallel
			RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

			for (int i = 1; i < shadow_atlas->shrink_stages.size(); i++) {
				int32_t base_size = shadow_atlas->shrink_stages[0].size;
				int32_t src_size = shadow_atlas->shrink_stages[i - 1].size;
				int32_t dst_size = shadow_atlas->shrink_stages[i].size;

				uint32_t rect_divisor = base_size / src_size;

				int32_t shrink_limit = 8 / (src_size / dst_size);

				//shrink in parallel for more performance
				for (uint32_t j = 0; j < cluster.lights_shadow_rect_cache_count; j++) {
					Rect2i src_rect = cluster.lights_shadow_rect_cache[j];

					src_rect.position /= rect_divisor;
					src_rect.size /= rect_divisor;

					storage->get_effects()->reduce_shadow(shadow_atlas->shrink_stages[i - 1].texture, shadow_atlas->shrink_stages[i].texture, Size2i(src_size, src_size), src_rect, shrink_limit, compute_list);
				}

				RD::get_singleton()->compute_list_add_barrier(compute_list);

				if (env->volumetric_fog_shadow_filter != RS::ENV_VOLUMETRIC_FOG_SHADOW_FILTER_DISABLED && shadow_atlas->shrink_stages[i].filter_texture.is_valid()) {
					uint32_t filter_divisor = base_size / dst_size;

					//filter in parallel for more performance
					for (uint32_t j = 0; j < cluster.lights_shadow_rect_cache_count; j++) {
						Rect2i dst_rect = cluster.lights_shadow_rect_cache[j];

						dst_rect.position /= filter_divisor;
						dst_rect.size /= filter_divisor;

						storage->get_effects()->filter_shadow(shadow_atlas->shrink_stages[i].texture, shadow_atlas->shrink_stages[i].filter_texture, Size2i(dst_size, dst_size), dst_rect, env->volumetric_fog_shadow_filter, compute_list, true, false);
					}

					RD::get_singleton()->compute_list_add_barrier(compute_list);

					for (uint32_t j = 0; j < cluster.lights_shadow_rect_cache_count; j++) {
						Rect2i dst_rect = cluster.lights_shadow_rect_cache[j];

						dst_rect.position /= filter_divisor;
						dst_rect.size /= filter_divisor;

						storage->get_effects()->filter_shadow(shadow_atlas->shrink_stages[i].texture, shadow_atlas->shrink_stages[i].filter_texture, Size2i(dst_size, dst_size), dst_rect, env->volumetric_fog_shadow_filter, compute_list, false, true);
					}
				}
			}

			RD::get_singleton()->compute_list_end();
		}
	}

	//update volumetric fog

	if (rb->volumetric_fog->uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->uniform_set)) {
		//re create uniform set if needed

		Vector<RD::Uniform> uniforms;

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1;
			if (shadow_atlas == nullptr || shadow_atlas->shrink_stages.size() == 0) {
				u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_BLACK));
			} else {
				u.ids.push_back(shadow_atlas->shrink_stages[shadow_atlas->shrink_stages.size() - 1].texture);
			}

			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 2;
			if (directional_shadow.shrink_stages.size() == 0) {
				u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_BLACK));
			} else {
				u.ids.push_back(directional_shadow.shrink_stages[directional_shadow.shrink_stages.size() - 1].texture);
			}
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 3;
			u.ids.push_back(get_positional_light_buffer());
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 4;
			u.ids.push_back(get_directional_light_buffer());
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 5;
			u.ids.push_back(get_cluster_builder_texture());
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 6;
			u.ids.push_back(get_cluster_builder_indices_buffer());
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 7;
			u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 8;
			u.ids.push_back(rb->volumetric_fog->light_density_map);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 9;
			u.ids.push_back(rb->volumetric_fog->fog_map);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 10;
			u.ids.push_back(shadow_sampler);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 11;
			u.ids.push_back(render_buffers_get_gi_probe_buffer(p_render_buffers));
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 12;
			for (int i = 0; i < RenderBuffers::MAX_GIPROBES; i++) {
				u.ids.push_back(rb->giprobe_textures[i]);
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 13;
			u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
			uniforms.push_back(u);
		}

		rb->volumetric_fog->uniform_set = RD::get_singleton()->uniform_set_create(uniforms, volumetric_fog.shader.version_get_shader(volumetric_fog.shader_version, 0), 0);

		SWAP(uniforms.write[7].ids.write[0], uniforms.write[8].ids.write[0]);

		rb->volumetric_fog->uniform_set2 = RD::get_singleton()->uniform_set_create(uniforms, volumetric_fog.shader.version_get_shader(volumetric_fog.shader_version, 0), 0);
	}

	bool using_sdfgi = env->volumetric_fog_gi_inject > 0.0001 && env->sdfgi_enabled && (rb->sdfgi != nullptr);

	if (using_sdfgi) {
		if (rb->volumetric_fog->sdfgi_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->sdfgi_uniform_set)) {
			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
				u.binding = 0;
				u.ids.push_back(gi.sdfgi_ubo);
				uniforms.push_back(u);
			}

			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 1;
				u.ids.push_back(rb->sdfgi->ambient_texture);
				uniforms.push_back(u);
			}

			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 2;
				u.ids.push_back(rb->sdfgi->occlusion_texture);
				uniforms.push_back(u);
			}

			rb->volumetric_fog->sdfgi_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, volumetric_fog.shader.version_get_shader(volumetric_fog.shader_version, VOLUMETRIC_FOG_SHADER_DENSITY_WITH_SDFGI), 1);
		}
	}

	rb->volumetric_fog->length = env->volumetric_fog_length;
	rb->volumetric_fog->spread = env->volumetric_fog_detail_spread;

	VolumetricFogShader::PushConstant push_constant;

	Vector2 frustum_near_size = p_cam_projection.get_viewport_half_extents();
	Vector2 frustum_far_size = p_cam_projection.get_far_plane_half_extents();
	float z_near = p_cam_projection.get_z_near();
	float z_far = p_cam_projection.get_z_far();
	float fog_end = env->volumetric_fog_length;

	Vector2 fog_far_size = frustum_near_size.lerp(frustum_far_size, (fog_end - z_near) / (z_far - z_near));
	Vector2 fog_near_size;
	if (p_cam_projection.is_orthogonal()) {
		fog_near_size = fog_far_size;
	} else {
		fog_near_size = Vector2();
	}

	push_constant.fog_frustum_size_begin[0] = fog_near_size.x;
	push_constant.fog_frustum_size_begin[1] = fog_near_size.y;

	push_constant.fog_frustum_size_end[0] = fog_far_size.x;
	push_constant.fog_frustum_size_end[1] = fog_far_size.y;

	push_constant.z_near = z_near;
	push_constant.z_far = z_far;

	push_constant.fog_frustum_end = fog_end;

	push_constant.fog_volume_size[0] = rb->volumetric_fog->width;
	push_constant.fog_volume_size[1] = rb->volumetric_fog->height;
	push_constant.fog_volume_size[2] = rb->volumetric_fog->depth;

	push_constant.directional_light_count = p_directional_light_count;

	Color light = env->volumetric_fog_light.to_linear();
	push_constant.light_energy[0] = light.r * env->volumetric_fog_light_energy;
	push_constant.light_energy[1] = light.g * env->volumetric_fog_light_energy;
	push_constant.light_energy[2] = light.b * env->volumetric_fog_light_energy;
	push_constant.base_density = env->volumetric_fog_density;

	push_constant.detail_spread = env->volumetric_fog_detail_spread;
	push_constant.gi_inject = env->volumetric_fog_gi_inject;

	push_constant.cam_rotation[0] = p_cam_transform.basis[0][0];
	push_constant.cam_rotation[1] = p_cam_transform.basis[1][0];
	push_constant.cam_rotation[2] = p_cam_transform.basis[2][0];
	push_constant.cam_rotation[3] = 0;
	push_constant.cam_rotation[4] = p_cam_transform.basis[0][1];
	push_constant.cam_rotation[5] = p_cam_transform.basis[1][1];
	push_constant.cam_rotation[6] = p_cam_transform.basis[2][1];
	push_constant.cam_rotation[7] = 0;
	push_constant.cam_rotation[8] = p_cam_transform.basis[0][2];
	push_constant.cam_rotation[9] = p_cam_transform.basis[1][2];
	push_constant.cam_rotation[10] = p_cam_transform.basis[2][2];
	push_constant.cam_rotation[11] = 0;
	push_constant.filter_axis = 0;
	push_constant.max_gi_probes = env->volumetric_fog_gi_inject > 0.001 ? p_gi_probe_count : 0;

	/*	Vector2 dssize = directional_shadow_get_size();
	push_constant.directional_shadow_pixel_size[0] = 1.0 / dssize.x;
	push_constant.directional_shadow_pixel_size[1] = 1.0 / dssize.y;
*/
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	bool use_filter = volumetric_fog_filter_active;

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, volumetric_fog.pipelines[using_sdfgi ? VOLUMETRIC_FOG_SHADER_DENSITY_WITH_SDFGI : VOLUMETRIC_FOG_SHADER_DENSITY]);

	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->uniform_set, 0);
	if (using_sdfgi) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->sdfgi_uniform_set, 1);
	}
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VolumetricFogShader::PushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth, 4, 4, 4);

	RD::get_singleton()->compute_list_add_barrier(compute_list);

	if (use_filter) {
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, volumetric_fog.pipelines[VOLUMETRIC_FOG_SHADER_FILTER]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->uniform_set, 0);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VolumetricFogShader::PushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth, 8, 8, 1);

		RD::get_singleton()->compute_list_add_barrier(compute_list);

		push_constant.filter_axis = 1;

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->uniform_set2, 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VolumetricFogShader::PushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth, 8, 8, 1);

		RD::get_singleton()->compute_list_add_barrier(compute_list);
	}

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, volumetric_fog.pipelines[VOLUMETRIC_FOG_SHADER_FOG]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->uniform_set, 0);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VolumetricFogShader::PushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->volumetric_fog->width, rb->volumetric_fog->height, 1, 8, 8, 1);

	RD::get_singleton()->compute_list_end();
}

void RasterizerSceneRD::render_scene(RID p_render_buffers, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count, RID *p_decal_cull_result, int p_decal_cull_count, InstanceBase **p_lightmap_cull_result, int p_lightmap_cull_count, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass) {
	Color clear_color;
	if (p_render_buffers.is_valid()) {
		RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
		ERR_FAIL_COND(!rb);
		clear_color = storage->render_target_get_clear_request_color(rb->render_target);
	} else {
		clear_color = storage->get_default_clear_color();
	}

	//assign render indices to giprobes
	for (int i = 0; i < p_gi_probe_cull_count; i++) {
		GIProbeInstance *giprobe_inst = gi_probe_instance_owner.getornull(p_gi_probe_cull_result[i]);
		if (giprobe_inst) {
			giprobe_inst->render_index = i;
		}
	}

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_UNSHADED) {
		p_light_cull_count = 0;
		p_reflection_probe_cull_count = 0;
		p_gi_probe_cull_count = 0;
	}

	cluster.builder.begin(p_cam_transform.affine_inverse(), p_cam_projection); //prepare cluster

	bool using_shadows = true;

	if (p_reflection_probe.is_valid()) {
		if (!storage->reflection_probe_renders_shadows(reflection_probe_instance_get_probe(p_reflection_probe))) {
			using_shadows = false;
		}
	} else {
		//do not render reflections when rendering a reflection probe
		_setup_reflections(p_reflection_probe_cull_result, p_reflection_probe_cull_count, p_cam_transform.affine_inverse(), p_environment);
	}

	uint32_t directional_light_count = 0;
	uint32_t positional_light_count = 0;
	_setup_lights(p_light_cull_result, p_light_cull_count, p_cam_transform.affine_inverse(), p_shadow_atlas, using_shadows, directional_light_count, positional_light_count);
	_setup_decals(p_decal_cull_result, p_decal_cull_count, p_cam_transform.affine_inverse());
	cluster.builder.bake_cluster(); //bake to cluster

	uint32_t gi_probe_count = 0;
	_setup_giprobes(p_render_buffers, p_cam_transform, p_gi_probe_cull_result, p_gi_probe_cull_count, gi_probe_count);

	if (p_render_buffers.is_valid()) {
		bool directional_shadows = false;
		for (uint32_t i = 0; i < directional_light_count; i++) {
			if (cluster.directional_lights[i].shadow_enabled) {
				directional_shadows = true;
				break;
			}
		}
		_update_volumetric_fog(p_render_buffers, p_environment, p_cam_projection, p_cam_transform, p_shadow_atlas, directional_light_count, directional_shadows, positional_light_count, gi_probe_count);
	}

	_render_scene(p_render_buffers, p_cam_transform, p_cam_projection, p_cam_ortogonal, p_cull_result, p_cull_count, directional_light_count, p_gi_probe_cull_result, p_gi_probe_cull_count, p_lightmap_cull_result, p_lightmap_cull_count, p_environment, p_camera_effects, p_shadow_atlas, p_reflection_atlas, p_reflection_probe, p_reflection_probe_pass, clear_color);

	if (p_render_buffers.is_valid()) {
		RENDER_TIMESTAMP("Tonemap");

		_render_buffers_post_process_and_tonemap(p_render_buffers, p_environment, p_camera_effects, p_cam_projection);
		_render_buffers_debug_draw(p_render_buffers, p_shadow_atlas);
		if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_SDFGI) {
			_sdfgi_debug_draw(p_render_buffers, p_cam_projection, p_cam_transform);
		}
	}
}

void RasterizerSceneRD::render_shadow(RID p_light, RID p_shadow_atlas, int p_pass, InstanceBase **p_cull_result, int p_cull_count) {
	LightInstance *light_instance = light_instance_owner.getornull(p_light);
	ERR_FAIL_COND(!light_instance);

	Rect2i atlas_rect;
	RID atlas_texture;

	bool using_dual_paraboloid = false;
	bool using_dual_paraboloid_flip = false;
	float znear = 0;
	float zfar = 0;
	RID render_fb;
	RID render_texture;
	float bias = 0;
	float normal_bias = 0;

	bool use_pancake = false;
	bool use_linear_depth = false;
	bool render_cubemap = false;
	bool finalize_cubemap = false;

	CameraMatrix light_projection;
	Transform light_transform;

	if (storage->light_get_type(light_instance->light) == RS::LIGHT_DIRECTIONAL) {
		//set pssm stuff
		if (light_instance->last_scene_shadow_pass != scene_pass) {
			light_instance->directional_rect = _get_directional_shadow_rect(directional_shadow.size, directional_shadow.light_count, directional_shadow.current_light);
			directional_shadow.current_light++;
			light_instance->last_scene_shadow_pass = scene_pass;
		}

		use_pancake = storage->light_get_param(light_instance->light, RS::LIGHT_PARAM_SHADOW_PANCAKE_SIZE) > 0;
		light_projection = light_instance->shadow_transform[p_pass].camera;
		light_transform = light_instance->shadow_transform[p_pass].transform;

		atlas_rect.position.x = light_instance->directional_rect.position.x;
		atlas_rect.position.y = light_instance->directional_rect.position.y;
		atlas_rect.size.width = light_instance->directional_rect.size.x;
		atlas_rect.size.height = light_instance->directional_rect.size.y;

		if (storage->light_directional_get_shadow_mode(light_instance->light) == RS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS) {
			atlas_rect.size.width /= 2;
			atlas_rect.size.height /= 2;

			if (p_pass == 1) {
				atlas_rect.position.x += atlas_rect.size.width;
			} else if (p_pass == 2) {
				atlas_rect.position.y += atlas_rect.size.height;
			} else if (p_pass == 3) {
				atlas_rect.position.x += atlas_rect.size.width;
				atlas_rect.position.y += atlas_rect.size.height;
			}

		} else if (storage->light_directional_get_shadow_mode(light_instance->light) == RS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS) {
			atlas_rect.size.height /= 2;

			if (p_pass == 0) {
			} else {
				atlas_rect.position.y += atlas_rect.size.height;
			}
		}

		light_instance->shadow_transform[p_pass].atlas_rect = atlas_rect;

		light_instance->shadow_transform[p_pass].atlas_rect.position /= directional_shadow.size;
		light_instance->shadow_transform[p_pass].atlas_rect.size /= directional_shadow.size;

		float bias_mult = light_instance->shadow_transform[p_pass].bias_scale;
		zfar = storage->light_get_param(light_instance->light, RS::LIGHT_PARAM_RANGE);
		bias = storage->light_get_param(light_instance->light, RS::LIGHT_PARAM_SHADOW_BIAS) * bias_mult;
		normal_bias = storage->light_get_param(light_instance->light, RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS) * bias_mult;

		ShadowMap *shadow_map = _get_shadow_map(atlas_rect.size);
		render_fb = shadow_map->fb;
		render_texture = shadow_map->depth;
		atlas_texture = directional_shadow.depth;

	} else {
		//set from shadow atlas

		ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_shadow_atlas);
		ERR_FAIL_COND(!shadow_atlas);
		ERR_FAIL_COND(!shadow_atlas->shadow_owners.has(p_light));

		uint32_t key = shadow_atlas->shadow_owners[p_light];

		uint32_t quadrant = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
		uint32_t shadow = key & ShadowAtlas::SHADOW_INDEX_MASK;

		ERR_FAIL_INDEX((int)shadow, shadow_atlas->quadrants[quadrant].shadows.size());

		uint32_t quadrant_size = shadow_atlas->size >> 1;

		atlas_rect.position.x = (quadrant & 1) * quadrant_size;
		atlas_rect.position.y = (quadrant >> 1) * quadrant_size;

		uint32_t shadow_size = (quadrant_size / shadow_atlas->quadrants[quadrant].subdivision);
		atlas_rect.position.x += (shadow % shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;
		atlas_rect.position.y += (shadow / shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;

		atlas_rect.size.width = shadow_size;
		atlas_rect.size.height = shadow_size;
		atlas_texture = shadow_atlas->depth;

		zfar = storage->light_get_param(light_instance->light, RS::LIGHT_PARAM_RANGE);
		bias = storage->light_get_param(light_instance->light, RS::LIGHT_PARAM_SHADOW_BIAS);
		normal_bias = storage->light_get_param(light_instance->light, RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS);

		if (storage->light_get_type(light_instance->light) == RS::LIGHT_OMNI) {
			if (storage->light_omni_get_shadow_mode(light_instance->light) == RS::LIGHT_OMNI_SHADOW_CUBE) {
				ShadowCubemap *cubemap = _get_shadow_cubemap(shadow_size / 2);

				render_fb = cubemap->side_fb[p_pass];
				render_texture = cubemap->cubemap;

				light_projection = light_instance->shadow_transform[0].camera;
				light_transform = light_instance->shadow_transform[0].transform;
				render_cubemap = true;
				finalize_cubemap = p_pass == 5;

			} else {
				light_projection = light_instance->shadow_transform[0].camera;
				light_transform = light_instance->shadow_transform[0].transform;

				atlas_rect.size.height /= 2;
				atlas_rect.position.y += p_pass * atlas_rect.size.height;

				using_dual_paraboloid = true;
				using_dual_paraboloid_flip = p_pass == 1;

				ShadowMap *shadow_map = _get_shadow_map(atlas_rect.size);
				render_fb = shadow_map->fb;
				render_texture = shadow_map->depth;
			}

		} else if (storage->light_get_type(light_instance->light) == RS::LIGHT_SPOT) {
			light_projection = light_instance->shadow_transform[0].camera;
			light_transform = light_instance->shadow_transform[0].transform;

			ShadowMap *shadow_map = _get_shadow_map(atlas_rect.size);
			render_fb = shadow_map->fb;
			render_texture = shadow_map->depth;

			znear = light_instance->shadow_transform[0].camera.get_z_near();
			use_linear_depth = true;
		}
	}

	if (render_cubemap) {
		//rendering to cubemap
		_render_shadow(render_fb, p_cull_result, p_cull_count, light_projection, light_transform, zfar, 0, 0, false, false, use_pancake);
		if (finalize_cubemap) {
			//reblit
			atlas_rect.size.height /= 2;
			storage->get_effects()->copy_cubemap_to_dp(render_texture, atlas_texture, atlas_rect, light_projection.get_z_near(), light_projection.get_z_far(), 0.0, false);
			atlas_rect.position.y += atlas_rect.size.height;
			storage->get_effects()->copy_cubemap_to_dp(render_texture, atlas_texture, atlas_rect, light_projection.get_z_near(), light_projection.get_z_far(), 0.0, true);
		}
	} else {
		//render shadow

		_render_shadow(render_fb, p_cull_result, p_cull_count, light_projection, light_transform, zfar, bias, normal_bias, using_dual_paraboloid, using_dual_paraboloid_flip, use_pancake);

		//copy to atlas
		if (use_linear_depth) {
			storage->get_effects()->copy_depth_to_rect_and_linearize(render_texture, atlas_texture, atlas_rect, true, znear, zfar);
		} else {
			storage->get_effects()->copy_depth_to_rect(render_texture, atlas_texture, atlas_rect, true);
		}

		//does not work from depth to color
		//RD::get_singleton()->texture_copy(render_texture, atlas_texture, Vector3(0, 0, 0), Vector3(atlas_rect.position.x, atlas_rect.position.y, 0), Vector3(atlas_rect.size.x, atlas_rect.size.y, 1), 0, 0, 0, 0, true);
	}
}

void RasterizerSceneRD::render_material(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID p_framebuffer, const Rect2i &p_region) {
	_render_material(p_cam_transform, p_cam_projection, p_cam_ortogonal, p_cull_result, p_cull_count, p_framebuffer, p_region);
}

void RasterizerSceneRD::render_sdfgi(RID p_render_buffers, int p_region, InstanceBase **p_cull_result, int p_cull_count) {
	//print_line("rendering region " + itos(p_region));
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);
	ERR_FAIL_COND(!rb->sdfgi);
	AABB bounds;
	Vector3i from;
	Vector3i size;

	int cascade_prev = _sdfgi_get_pending_region_data(p_render_buffers, p_region - 1, from, size, bounds);
	int cascade_next = _sdfgi_get_pending_region_data(p_render_buffers, p_region + 1, from, size, bounds);
	int cascade = _sdfgi_get_pending_region_data(p_render_buffers, p_region, from, size, bounds);
	ERR_FAIL_COND(cascade < 0);

	if (cascade_prev != cascade) {
		//initialize render
		RD::get_singleton()->texture_clear(rb->sdfgi->render_albedo, Color(0, 0, 0, 0), 0, 1, 0, 1, true);
		RD::get_singleton()->texture_clear(rb->sdfgi->render_emission, Color(0, 0, 0, 0), 0, 1, 0, 1, true);
		RD::get_singleton()->texture_clear(rb->sdfgi->render_emission_aniso, Color(0, 0, 0, 0), 0, 1, 0, 1, true);
		RD::get_singleton()->texture_clear(rb->sdfgi->render_geom_facing, Color(0, 0, 0, 0), 0, 1, 0, 1, true);
	}

	//print_line("rendering cascade " + itos(p_region) + " objects: " + itos(p_cull_count) + " bounds: " + bounds + " from: " + from + " size: " + size + " cell size: " + rtos(rb->sdfgi->cascades[cascade].cell_size));
	_render_sdfgi(p_render_buffers, from, size, bounds, p_cull_result, p_cull_count, rb->sdfgi->render_albedo, rb->sdfgi->render_emission, rb->sdfgi->render_emission_aniso, rb->sdfgi->render_geom_facing);

	if (cascade_next != cascade) {
		RENDER_TIMESTAMP(">SDFGI Update SDF");
		//done rendering! must update SDF
		//clear dispatch indirect data

		SDGIShader::PreprocessPushConstant push_constant;
		zeromem(&push_constant, sizeof(SDGIShader::PreprocessPushConstant));

		RENDER_TIMESTAMP("Scroll SDF");

		//scroll
		if (rb->sdfgi->cascades[cascade].dirty_regions != SDFGI::Cascade::DIRTY_ALL) {
			//for scroll
			Vector3i dirty = rb->sdfgi->cascades[cascade].dirty_regions;
			push_constant.scroll[0] = dirty.x;
			push_constant.scroll[1] = dirty.y;
			push_constant.scroll[2] = dirty.z;
		} else {
			//for no scroll
			push_constant.scroll[0] = 0;
			push_constant.scroll[1] = 0;
			push_constant.scroll[2] = 0;
		}
		push_constant.grid_size = rb->sdfgi->cascade_size;
		push_constant.cascade = cascade;

		if (rb->sdfgi->cascades[cascade].dirty_regions != SDFGI::Cascade::DIRTY_ALL) {
			RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

			//must pre scroll existing data because not all is dirty
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_SCROLL]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->cascades[cascade].scroll_uniform_set, 0);

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));
			RD::get_singleton()->compute_list_dispatch_indirect(compute_list, rb->sdfgi->cascades[cascade].solid_cell_dispatch_buffer, 0);
			// no barrier do all together

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_SCROLL_OCCLUSION]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->cascades[cascade].scroll_occlusion_uniform_set, 0);

			Vector3i dirty = rb->sdfgi->cascades[cascade].dirty_regions;
			Vector3i groups;
			groups.x = rb->sdfgi->cascade_size - ABS(dirty.x);
			groups.y = rb->sdfgi->cascade_size - ABS(dirty.y);
			groups.z = rb->sdfgi->cascade_size - ABS(dirty.z);

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, groups.x, groups.y, groups.z, 4, 4, 4);

			//no barrier, continue together

			{
				//scroll probes and their history also

				SDGIShader::IntegratePushConstant ipush_constant;
				ipush_constant.grid_size[1] = rb->sdfgi->cascade_size;
				ipush_constant.grid_size[2] = rb->sdfgi->cascade_size;
				ipush_constant.grid_size[0] = rb->sdfgi->cascade_size;
				ipush_constant.max_cascades = rb->sdfgi->cascades.size();
				ipush_constant.probe_axis_size = rb->sdfgi->probe_axis_count;
				ipush_constant.history_index = 0;
				ipush_constant.history_size = rb->sdfgi->history_size;
				ipush_constant.ray_count = 0;
				ipush_constant.ray_bias = 0;
				ipush_constant.sky_mode = 0;
				ipush_constant.sky_energy = 0;
				ipush_constant.sky_color[0] = 0;
				ipush_constant.sky_color[1] = 0;
				ipush_constant.sky_color[2] = 0;
				ipush_constant.y_mult = rb->sdfgi->y_mult;
				ipush_constant.store_ambient_texture = false;

				ipush_constant.image_size[0] = rb->sdfgi->probe_axis_count * rb->sdfgi->probe_axis_count;
				ipush_constant.image_size[1] = rb->sdfgi->probe_axis_count;
				ipush_constant.image_size[1] = rb->sdfgi->probe_axis_count;

				int32_t probe_divisor = rb->sdfgi->cascade_size / SDFGI::PROBE_DIVISOR;
				ipush_constant.cascade = cascade;
				ipush_constant.world_offset[0] = rb->sdfgi->cascades[cascade].position.x / probe_divisor;
				ipush_constant.world_offset[1] = rb->sdfgi->cascades[cascade].position.y / probe_divisor;
				ipush_constant.world_offset[2] = rb->sdfgi->cascades[cascade].position.z / probe_divisor;

				ipush_constant.scroll[0] = dirty.x / probe_divisor;
				ipush_constant.scroll[1] = dirty.y / probe_divisor;
				ipush_constant.scroll[2] = dirty.z / probe_divisor;

				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.integrate_pipeline[SDGIShader::INTEGRATE_MODE_SCROLL]);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->cascades[cascade].integrate_uniform_set, 0);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, sdfgi_shader.integrate_default_sky_uniform_set, 1);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, &ipush_constant, sizeof(SDGIShader::IntegratePushConstant));
				RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->sdfgi->probe_axis_count * rb->sdfgi->probe_axis_count, rb->sdfgi->probe_axis_count, 1, 8, 8, 1);

				RD::get_singleton()->compute_list_add_barrier(compute_list);

				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.integrate_pipeline[SDGIShader::INTEGRATE_MODE_SCROLL_STORE]);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->cascades[cascade].integrate_uniform_set, 0);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, sdfgi_shader.integrate_default_sky_uniform_set, 1);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, &ipush_constant, sizeof(SDGIShader::IntegratePushConstant));
				RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->sdfgi->probe_axis_count * rb->sdfgi->probe_axis_count, rb->sdfgi->probe_axis_count, 1, 8, 8, 1);
			}

			//ok finally barrier
			RD::get_singleton()->compute_list_end();
		}

		//clear dispatch indirect data
		uint32_t dispatch_indirct_data[4] = { 0, 0, 0, 0 };
		RD::get_singleton()->buffer_update(rb->sdfgi->cascades[cascade].solid_cell_dispatch_buffer, 0, sizeof(uint32_t) * 4, dispatch_indirct_data, true);

		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

		bool half_size = true; //much faster, very little difference
		static const int optimized_jf_group_size = 8;

		if (half_size) {
			push_constant.grid_size >>= 1;

			uint32_t cascade_half_size = rb->sdfgi->cascade_size >> 1;
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_JUMP_FLOOD_INITIALIZE_HALF]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->sdf_initialize_half_uniform_set, 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, cascade_half_size, cascade_half_size, cascade_half_size, 4, 4, 4);
			RD::get_singleton()->compute_list_add_barrier(compute_list);

			//must start with regular jumpflood

			push_constant.half_size = true;
			{
				RENDER_TIMESTAMP("SDFGI Jump Flood (Half Size)");

				uint32_t s = cascade_half_size;

				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_JUMP_FLOOD]);

				int jf_us = 0;
				//start with regular jump flood for very coarse reads, as this is impossible to optimize
				while (s > 1) {
					s /= 2;
					push_constant.step_size = s;
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->jump_flood_half_uniform_set[jf_us], 0);
					RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));
					RD::get_singleton()->compute_list_dispatch_threads(compute_list, cascade_half_size, cascade_half_size, cascade_half_size, 4, 4, 4);
					RD::get_singleton()->compute_list_add_barrier(compute_list);
					jf_us = jf_us == 0 ? 1 : 0;

					if (cascade_half_size / (s / 2) >= optimized_jf_group_size) {
						break;
					}
				}

				RENDER_TIMESTAMP("SDFGI Jump Flood Optimized (Half Size)");

				//continue with optimized jump flood for smaller reads
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_JUMP_FLOOD_OPTIMIZED]);
				while (s > 1) {
					s /= 2;
					push_constant.step_size = s;
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->jump_flood_half_uniform_set[jf_us], 0);
					RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));
					RD::get_singleton()->compute_list_dispatch_threads(compute_list, cascade_half_size, cascade_half_size, cascade_half_size, optimized_jf_group_size, optimized_jf_group_size, optimized_jf_group_size);
					RD::get_singleton()->compute_list_add_barrier(compute_list);
					jf_us = jf_us == 0 ? 1 : 0;
				}
			}

			// restore grid size for last passes
			push_constant.grid_size = rb->sdfgi->cascade_size;

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_JUMP_FLOOD_UPSCALE]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->sdf_upscale_uniform_set, 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, 4, 4, 4);
			RD::get_singleton()->compute_list_add_barrier(compute_list);

			//run one pass of fullsize jumpflood to fix up half size arctifacts

			push_constant.half_size = false;
			push_constant.step_size = 1;
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_JUMP_FLOOD_OPTIMIZED]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->jump_flood_uniform_set[rb->sdfgi->upscale_jfa_uniform_set_index], 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, optimized_jf_group_size, optimized_jf_group_size, optimized_jf_group_size);
			RD::get_singleton()->compute_list_add_barrier(compute_list);

		} else {
			//full size jumpflood
			RENDER_TIMESTAMP("SDFGI Jump Flood");

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_JUMP_FLOOD_INITIALIZE]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->sdf_initialize_uniform_set, 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, 4, 4, 4);

			RD::get_singleton()->compute_list_add_barrier(compute_list);

			push_constant.half_size = false;
			{
				uint32_t s = rb->sdfgi->cascade_size;

				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_JUMP_FLOOD]);

				int jf_us = 0;
				//start with regular jump flood for very coarse reads, as this is impossible to optimize
				while (s > 1) {
					s /= 2;
					push_constant.step_size = s;
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->jump_flood_uniform_set[jf_us], 0);
					RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));
					RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, 4, 4, 4);
					RD::get_singleton()->compute_list_add_barrier(compute_list);
					jf_us = jf_us == 0 ? 1 : 0;

					if (rb->sdfgi->cascade_size / (s / 2) >= optimized_jf_group_size) {
						break;
					}
				}

				RENDER_TIMESTAMP("SDFGI Jump Flood Optimized");

				//continue with optimized jump flood for smaller reads
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_JUMP_FLOOD_OPTIMIZED]);
				while (s > 1) {
					s /= 2;
					push_constant.step_size = s;
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->jump_flood_uniform_set[jf_us], 0);
					RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));
					RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, optimized_jf_group_size, optimized_jf_group_size, optimized_jf_group_size);
					RD::get_singleton()->compute_list_add_barrier(compute_list);
					jf_us = jf_us == 0 ? 1 : 0;
				}
			}
		}

		RENDER_TIMESTAMP("SDFGI Occlusion");

		// occlusion
		{
			uint32_t probe_size = rb->sdfgi->cascade_size / SDFGI::PROBE_DIVISOR;
			Vector3i probe_global_pos = rb->sdfgi->cascades[cascade].position / probe_size;

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_OCCLUSION]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->occlusion_uniform_set, 0);
			for (int i = 0; i < 8; i++) {
				//dispatch all at once for performance
				Vector3i offset(i & 1, (i >> 1) & 1, (i >> 2) & 1);

				if ((probe_global_pos.x & 1) != 0) {
					offset.x = (offset.x + 1) & 1;
				}
				if ((probe_global_pos.y & 1) != 0) {
					offset.y = (offset.y + 1) & 1;
				}
				if ((probe_global_pos.z & 1) != 0) {
					offset.z = (offset.z + 1) & 1;
				}
				push_constant.probe_offset[0] = offset.x;
				push_constant.probe_offset[1] = offset.y;
				push_constant.probe_offset[2] = offset.z;
				push_constant.occlusion_index = i;
				RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));

				Vector3i groups = Vector3i(probe_size + 1, probe_size + 1, probe_size + 1) - offset; //if offset, it's one less probe per axis to compute
				RD::get_singleton()->compute_list_dispatch(compute_list, groups.x, groups.y, groups.z);
			}
			RD::get_singleton()->compute_list_add_barrier(compute_list);
		}

		RENDER_TIMESTAMP("SDFGI Store");

		// store
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.preprocess_pipeline[SDGIShader::PRE_PROCESS_STORE]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->sdfgi->cascades[cascade].sdf_store_uniform_set, 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SDGIShader::PreprocessPushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, 4, 4, 4);

		RD::get_singleton()->compute_list_end();

		//clear these textures, as they will have previous garbage on next draw
		RD::get_singleton()->texture_clear(rb->sdfgi->cascades[cascade].light_tex, Color(0, 0, 0, 0), 0, 1, 0, 1, true);
		RD::get_singleton()->texture_clear(rb->sdfgi->cascades[cascade].light_aniso_0_tex, Color(0, 0, 0, 0), 0, 1, 0, 1, true);
		RD::get_singleton()->texture_clear(rb->sdfgi->cascades[cascade].light_aniso_1_tex, Color(0, 0, 0, 0), 0, 1, 0, 1, true);

#if 0
		Vector<uint8_t> data = RD::get_singleton()->texture_get_data(rb->sdfgi->cascades[cascade].sdf, 0);
		Ref<Image> img;
		img.instance();
		for (uint32_t i = 0; i < rb->sdfgi->cascade_size; i++) {
			Vector<uint8_t> subarr = data.subarray(128 * 128 * i, 128 * 128 * (i + 1) - 1);
			img->create(rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, false, Image::FORMAT_L8, subarr);
			img->save_png("res://cascade_sdf_" + itos(cascade) + "_" + itos(i) + ".png");
		}

		//finalize render and update sdf
#endif

#if 0
		Vector<uint8_t> data = RD::get_singleton()->texture_get_data(rb->sdfgi->render_albedo, 0);
		Ref<Image> img;
		img.instance();
		for (uint32_t i = 0; i < rb->sdfgi->cascade_size; i++) {
			Vector<uint8_t> subarr = data.subarray(128 * 128 * i * 2, 128 * 128 * (i + 1) * 2 - 1);
			img->create(rb->sdfgi->cascade_size, rb->sdfgi->cascade_size, false, Image::FORMAT_RGB565, subarr);
			img->convert(Image::FORMAT_RGBA8);
			img->save_png("res://cascade_" + itos(cascade) + "_" + itos(i) + ".png");
		}

		//finalize render and update sdf
#endif

		RENDER_TIMESTAMP("<SDFGI Update SDF");
	}
}

void RasterizerSceneRD::render_particle_collider_heightfield(RID p_collider, const Transform &p_transform, InstanceBase **p_cull_result, int p_cull_count) {
	ERR_FAIL_COND(!storage->particles_collision_is_heightfield(p_collider));
	Vector3 extents = storage->particles_collision_get_extents(p_collider) * p_transform.basis.get_scale();
	CameraMatrix cm;
	cm.set_orthogonal(-extents.x, extents.x, -extents.z, extents.z, 0, extents.y * 2.0);

	Vector3 cam_pos = p_transform.origin;
	cam_pos.y += extents.y;

	Transform cam_xform;
	cam_xform.set_look_at(cam_pos, cam_pos - p_transform.basis.get_axis(Vector3::AXIS_Y), -p_transform.basis.get_axis(Vector3::AXIS_Z).normalized());

	RID fb = storage->particles_collision_get_heightfield_framebuffer(p_collider);

	_render_particle_collider_heightfield(fb, cam_xform, cm, p_cull_result, p_cull_count);
}

void RasterizerSceneRD::render_sdfgi_static_lights(RID p_render_buffers, uint32_t p_cascade_count, const uint32_t *p_cascade_indices, const RID **p_positional_light_cull_result, const uint32_t *p_positional_light_cull_count) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);
	ERR_FAIL_COND(!rb->sdfgi);

	ERR_FAIL_COND(p_positional_light_cull_count == 0);

	_sdfgi_update_cascades(p_render_buffers); //need cascades updated for this

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sdfgi_shader.direct_light_pipeline[SDGIShader::DIRECT_LIGHT_MODE_STATIC]);

	SDGIShader::DirectLightPushConstant dl_push_constant;

	dl_push_constant.grid_size[0] = rb->sdfgi->cascade_size;
	dl_push_constant.grid_size[1] = rb->sdfgi->cascade_size;
	dl_push_constant.grid_size[2] = rb->sdfgi->cascade_size;
	dl_push_constant.max_cascades = rb->sdfgi->cascades.size();
	dl_push_constant.probe_axis_size = rb->sdfgi->probe_axis_count;
	dl_push_constant.multibounce = false; // this is static light, do not multibounce yet
	dl_push_constant.y_mult = rb->sdfgi->y_mult;

	//all must be processed
	dl_push_constant.process_offset = 0;
	dl_push_constant.process_increment = 1;

	SDGIShader::Light lights[SDFGI::MAX_STATIC_LIGHTS];

	for (uint32_t i = 0; i < p_cascade_count; i++) {
		ERR_CONTINUE(p_cascade_indices[i] >= rb->sdfgi->cascades.size());

		SDFGI::Cascade &cc = rb->sdfgi->cascades[p_cascade_indices[i]];

		{ //fill light buffer

			AABB cascade_aabb;
			cascade_aabb.position = Vector3((Vector3i(1, 1, 1) * -int32_t(rb->sdfgi->cascade_size >> 1) + cc.position)) * cc.cell_size;
			cascade_aabb.size = Vector3(1, 1, 1) * rb->sdfgi->cascade_size * cc.cell_size;

			int idx = 0;

			for (uint32_t j = 0; j < p_positional_light_cull_count[i]; j++) {
				if (idx == SDFGI::MAX_STATIC_LIGHTS) {
					break;
				}

				LightInstance *li = light_instance_owner.getornull(p_positional_light_cull_result[i][j]);
				ERR_CONTINUE(!li);

				uint32_t max_sdfgi_cascade = storage->light_get_max_sdfgi_cascade(li->light);
				if (p_cascade_indices[i] > max_sdfgi_cascade) {
					continue;
				}

				if (!cascade_aabb.intersects(li->aabb)) {
					continue;
				}

				lights[idx].type = storage->light_get_type(li->light);

				Vector3 dir = -li->transform.basis.get_axis(Vector3::AXIS_Z);
				if (lights[idx].type == RS::LIGHT_DIRECTIONAL) {
					dir.y *= rb->sdfgi->y_mult; //only makes sense for directional
					dir.normalize();
				}
				lights[idx].direction[0] = dir.x;
				lights[idx].direction[1] = dir.y;
				lights[idx].direction[2] = dir.z;
				Vector3 pos = li->transform.origin;
				pos.y *= rb->sdfgi->y_mult;
				lights[idx].position[0] = pos.x;
				lights[idx].position[1] = pos.y;
				lights[idx].position[2] = pos.z;
				Color color = storage->light_get_color(li->light);
				color = color.to_linear();
				lights[idx].color[0] = color.r;
				lights[idx].color[1] = color.g;
				lights[idx].color[2] = color.b;
				lights[idx].energy = storage->light_get_param(li->light, RS::LIGHT_PARAM_ENERGY);
				lights[idx].has_shadow = storage->light_has_shadow(li->light);
				lights[idx].attenuation = storage->light_get_param(li->light, RS::LIGHT_PARAM_ATTENUATION);
				lights[idx].radius = storage->light_get_param(li->light, RS::LIGHT_PARAM_RANGE);
				lights[idx].spot_angle = Math::deg2rad(storage->light_get_param(li->light, RS::LIGHT_PARAM_SPOT_ANGLE));
				lights[idx].spot_attenuation = storage->light_get_param(li->light, RS::LIGHT_PARAM_SPOT_ATTENUATION);

				idx++;
			}

			if (idx > 0) {
				RD::get_singleton()->buffer_update(cc.lights_buffer, 0, idx * sizeof(SDGIShader::Light), lights, true);
			}
			dl_push_constant.light_count = idx;
		}

		dl_push_constant.cascade = p_cascade_indices[i];

		if (dl_push_constant.light_count > 0) {
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, cc.sdf_direct_light_uniform_set, 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &dl_push_constant, sizeof(SDGIShader::DirectLightPushConstant));
			RD::get_singleton()->compute_list_dispatch_indirect(compute_list, cc.solid_cell_dispatch_buffer, 0);
		}
	}

	RD::get_singleton()->compute_list_end();
}

bool RasterizerSceneRD::free(RID p_rid) {
	if (render_buffers_owner.owns(p_rid)) {
		RenderBuffers *rb = render_buffers_owner.getornull(p_rid);
		_free_render_buffer_data(rb);
		memdelete(rb->data);
		if (rb->sdfgi) {
			_sdfgi_erase(rb);
		}
		if (rb->volumetric_fog) {
			_volumetric_fog_erase(rb);
		}
		render_buffers_owner.free(p_rid);
	} else if (environment_owner.owns(p_rid)) {
		//not much to delete, just free it
		environment_owner.free(p_rid);
	} else if (camera_effects_owner.owns(p_rid)) {
		//not much to delete, just free it
		camera_effects_owner.free(p_rid);
	} else if (reflection_atlas_owner.owns(p_rid)) {
		reflection_atlas_set_size(p_rid, 0, 0);
		reflection_atlas_owner.free(p_rid);
	} else if (reflection_probe_instance_owner.owns(p_rid)) {
		//not much to delete, just free it
		//ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_rid);
		reflection_probe_release_atlas_index(p_rid);
		reflection_probe_instance_owner.free(p_rid);
	} else if (decal_instance_owner.owns(p_rid)) {
		decal_instance_owner.free(p_rid);
	} else if (gi_probe_instance_owner.owns(p_rid)) {
		GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_rid);
		if (gi_probe->texture.is_valid()) {
			RD::get_singleton()->free(gi_probe->texture);
			RD::get_singleton()->free(gi_probe->write_buffer);
		}

		for (int i = 0; i < gi_probe->dynamic_maps.size(); i++) {
			RD::get_singleton()->free(gi_probe->dynamic_maps[i].texture);
			RD::get_singleton()->free(gi_probe->dynamic_maps[i].depth);
		}

		gi_probe_instance_owner.free(p_rid);
	} else if (sky_owner.owns(p_rid)) {
		_update_dirty_skys();
		Sky *sky = sky_owner.getornull(p_rid);

		if (sky->radiance.is_valid()) {
			RD::get_singleton()->free(sky->radiance);
			sky->radiance = RID();
		}
		_clear_reflection_data(sky->reflection);

		if (sky->uniform_buffer.is_valid()) {
			RD::get_singleton()->free(sky->uniform_buffer);
			sky->uniform_buffer = RID();
		}

		if (sky->half_res_pass.is_valid()) {
			RD::get_singleton()->free(sky->half_res_pass);
			sky->half_res_pass = RID();
		}

		if (sky->quarter_res_pass.is_valid()) {
			RD::get_singleton()->free(sky->quarter_res_pass);
			sky->quarter_res_pass = RID();
		}

		if (sky->material.is_valid()) {
			storage->free(sky->material);
		}

		sky_owner.free(p_rid);
	} else if (light_instance_owner.owns(p_rid)) {
		LightInstance *light_instance = light_instance_owner.getornull(p_rid);

		//remove from shadow atlases..
		for (Set<RID>::Element *E = light_instance->shadow_atlases.front(); E; E = E->next()) {
			ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(E->get());
			ERR_CONTINUE(!shadow_atlas->shadow_owners.has(p_rid));
			uint32_t key = shadow_atlas->shadow_owners[p_rid];
			uint32_t q = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
			uint32_t s = key & ShadowAtlas::SHADOW_INDEX_MASK;

			shadow_atlas->quadrants[q].shadows.write[s].owner = RID();
			shadow_atlas->shadow_owners.erase(p_rid);
		}

		light_instance_owner.free(p_rid);

	} else if (shadow_atlas_owner.owns(p_rid)) {
		shadow_atlas_set_size(p_rid, 0);
		shadow_atlas_owner.free(p_rid);

	} else {
		return false;
	}

	return true;
}

void RasterizerSceneRD::set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) {
	debug_draw = p_debug_draw;
}

void RasterizerSceneRD::update() {
	_update_dirty_skys();
}

void RasterizerSceneRD::set_time(double p_time, double p_step) {
	time = p_time;
	time_step = p_step;
}

void RasterizerSceneRD::screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_limit) {
	screen_space_roughness_limiter = p_enable;
	screen_space_roughness_limiter_amount = p_amount;
	screen_space_roughness_limiter_limit = p_limit;
}

bool RasterizerSceneRD::screen_space_roughness_limiter_is_active() const {
	return screen_space_roughness_limiter;
}

float RasterizerSceneRD::screen_space_roughness_limiter_get_amount() const {
	return screen_space_roughness_limiter_amount;
}

float RasterizerSceneRD::screen_space_roughness_limiter_get_limit() const {
	return screen_space_roughness_limiter_limit;
}

TypedArray<Image> RasterizerSceneRD::bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size) {
	RD::TextureFormat tf;
	tf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	tf.width = p_image_size.width; // Always 64x64
	tf.height = p_image_size.height;
	tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

	RID albedo_alpha_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());
	RID normal_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());
	RID orm_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());

	tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
	RID emission_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());

	tf.format = RD::DATA_FORMAT_R32_SFLOAT;
	RID depth_write_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());

	tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
	tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
	RID depth_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());

	Vector<RID> fb_tex;
	fb_tex.push_back(albedo_alpha_tex);
	fb_tex.push_back(normal_tex);
	fb_tex.push_back(orm_tex);
	fb_tex.push_back(emission_tex);
	fb_tex.push_back(depth_write_tex);
	fb_tex.push_back(depth_tex);

	RID fb = RD::get_singleton()->framebuffer_create(fb_tex);

	//RID sampled_light;

	InstanceBase ins;

	ins.base_type = RSG::storage->get_base_type(p_base);
	ins.base = p_base;
	ins.materials.resize(RSG::storage->mesh_get_surface_count(p_base));
	for (int i = 0; i < ins.materials.size(); i++) {
		if (i < p_material_overrides.size()) {
			ins.materials.write[i] = p_material_overrides[i];
		}
	}

	InstanceBase *cull = &ins;
	_render_uv2(&cull, 1, fb, Rect2i(0, 0, p_image_size.width, p_image_size.height));

	TypedArray<Image> ret;

	{
		PackedByteArray data = RD::get_singleton()->texture_get_data(albedo_alpha_tex, 0);
		Ref<Image> img;
		img.instance();
		img->create(p_image_size.width, p_image_size.height, false, Image::FORMAT_RGBA8, data);
		RD::get_singleton()->free(albedo_alpha_tex);
		ret.push_back(img);
	}

	{
		PackedByteArray data = RD::get_singleton()->texture_get_data(normal_tex, 0);
		Ref<Image> img;
		img.instance();
		img->create(p_image_size.width, p_image_size.height, false, Image::FORMAT_RGBA8, data);
		RD::get_singleton()->free(normal_tex);
		ret.push_back(img);
	}

	{
		PackedByteArray data = RD::get_singleton()->texture_get_data(orm_tex, 0);
		Ref<Image> img;
		img.instance();
		img->create(p_image_size.width, p_image_size.height, false, Image::FORMAT_RGBA8, data);
		RD::get_singleton()->free(orm_tex);
		ret.push_back(img);
	}

	{
		PackedByteArray data = RD::get_singleton()->texture_get_data(emission_tex, 0);
		Ref<Image> img;
		img.instance();
		img->create(p_image_size.width, p_image_size.height, false, Image::FORMAT_RGBAH, data);
		RD::get_singleton()->free(emission_tex);
		ret.push_back(img);
	}

	RD::get_singleton()->free(depth_write_tex);
	RD::get_singleton()->free(depth_tex);

	return ret;
}

void RasterizerSceneRD::sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) {
	sdfgi_debug_probe_pos = p_position;
	sdfgi_debug_probe_dir = p_dir;
}

RasterizerSceneRD *RasterizerSceneRD::singleton = nullptr;

RID RasterizerSceneRD::get_cluster_builder_texture() {
	return cluster.builder.get_cluster_texture();
}

RID RasterizerSceneRD::get_cluster_builder_indices_buffer() {
	return cluster.builder.get_cluster_indices_buffer();
}

RID RasterizerSceneRD::get_reflection_probe_buffer() {
	return cluster.reflection_buffer;
}
RID RasterizerSceneRD::get_positional_light_buffer() {
	return cluster.light_buffer;
}
RID RasterizerSceneRD::get_directional_light_buffer() {
	return cluster.directional_light_buffer;
}
RID RasterizerSceneRD::get_decal_buffer() {
	return cluster.decal_buffer;
}
int RasterizerSceneRD::get_max_directional_lights() const {
	return cluster.max_directional_lights;
}

RasterizerSceneRD::RasterizerSceneRD(RasterizerStorageRD *p_storage) {
	storage = p_storage;
	singleton = this;

	roughness_layers = GLOBAL_GET("rendering/quality/reflections/roughness_layers");
	sky_ggx_samples_quality = GLOBAL_GET("rendering/quality/reflections/ggx_samples");
	sky_use_cubemap_array = GLOBAL_GET("rendering/quality/reflections/texture_array_reflections");
	//	sky_use_cubemap_array = false;

	//uint32_t textures_per_stage = RD::get_singleton()->limit_get(RD::LIMIT_MAX_TEXTURES_PER_SHADER_STAGE);

	{
		//kinda complicated to compute the amount of slots, we try to use as many as we can

		gi_probe_max_lights = 32;

		gi_probe_lights = memnew_arr(GIProbeLight, gi_probe_max_lights);
		gi_probe_lights_uniform = RD::get_singleton()->uniform_buffer_create(gi_probe_max_lights * sizeof(GIProbeLight));
		gi_probe_quality = RS::GIProbeQuality(CLAMP(int(GLOBAL_GET("rendering/quality/gi_probes/quality")), 0, 1));

		String defines = "\n#define MAX_LIGHTS " + itos(gi_probe_max_lights) + "\n";

		Vector<String> versions;
		versions.push_back("\n#define MODE_COMPUTE_LIGHT\n");
		versions.push_back("\n#define MODE_SECOND_BOUNCE\n");
		versions.push_back("\n#define MODE_UPDATE_MIPMAPS\n");
		versions.push_back("\n#define MODE_WRITE_TEXTURE\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_LIGHTING\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_SHRINK\n#define MODE_DYNAMIC_SHRINK_WRITE\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_SHRINK\n#define MODE_DYNAMIC_SHRINK_PLOT\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_SHRINK\n#define MODE_DYNAMIC_SHRINK_PLOT\n#define MODE_DYNAMIC_SHRINK_WRITE\n");

		giprobe_shader.initialize(versions, defines);
		giprobe_lighting_shader_version = giprobe_shader.version_create();
		for (int i = 0; i < GI_PROBE_SHADER_VERSION_MAX; i++) {
			giprobe_lighting_shader_version_shaders[i] = giprobe_shader.version_get_shader(giprobe_lighting_shader_version, i);
			giprobe_lighting_shader_version_pipelines[i] = RD::get_singleton()->compute_pipeline_create(giprobe_lighting_shader_version_shaders[i]);
		}
	}

	{
		String defines;
		Vector<String> versions;
		versions.push_back("\n#define MODE_DEBUG_COLOR\n");
		versions.push_back("\n#define MODE_DEBUG_LIGHT\n");
		versions.push_back("\n#define MODE_DEBUG_EMISSION\n");
		versions.push_back("\n#define MODE_DEBUG_LIGHT\n#define MODE_DEBUG_LIGHT_FULL\n");

		giprobe_debug_shader.initialize(versions, defines);
		giprobe_debug_shader_version = giprobe_debug_shader.version_create();
		for (int i = 0; i < GI_PROBE_DEBUG_MAX; i++) {
			giprobe_debug_shader_version_shaders[i] = giprobe_debug_shader.version_get_shader(giprobe_debug_shader_version, i);

			RD::PipelineRasterizationState rs;
			rs.cull_mode = RD::POLYGON_CULL_FRONT;
			RD::PipelineDepthStencilState ds;
			ds.enable_depth_test = true;
			ds.enable_depth_write = true;
			ds.depth_compare_operator = RD::COMPARE_OP_LESS_OR_EQUAL;

			giprobe_debug_shader_version_pipelines[i].setup(giprobe_debug_shader_version_shaders[i], RD::RENDER_PRIMITIVE_TRIANGLES, rs, RD::PipelineMultisampleState(), ds, RD::PipelineColorBlendState::create_disabled(), 0);
		}
	}

	/* SKY SHADER */

	{
		// Start with the directional lights for the sky
		sky_scene_state.max_directional_lights = 4;
		uint32_t directional_light_buffer_size = sky_scene_state.max_directional_lights * sizeof(SkyDirectionalLightData);
		sky_scene_state.directional_lights = memnew_arr(SkyDirectionalLightData, sky_scene_state.max_directional_lights);
		sky_scene_state.last_frame_directional_lights = memnew_arr(SkyDirectionalLightData, sky_scene_state.max_directional_lights);
		sky_scene_state.last_frame_directional_light_count = sky_scene_state.max_directional_lights + 1;
		sky_scene_state.directional_light_buffer = RD::get_singleton()->uniform_buffer_create(directional_light_buffer_size);

		String defines = "\n#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS " + itos(sky_scene_state.max_directional_lights) + "\n";

		// Initialize sky
		Vector<String> sky_modes;
		sky_modes.push_back(""); // Full size
		sky_modes.push_back("\n#define USE_HALF_RES_PASS\n"); // Half Res
		sky_modes.push_back("\n#define USE_QUARTER_RES_PASS\n"); // Quarter res
		sky_modes.push_back("\n#define USE_CUBEMAP_PASS\n"); // Cubemap
		sky_modes.push_back("\n#define USE_CUBEMAP_PASS\n#define USE_HALF_RES_PASS\n"); // Half Res Cubemap
		sky_modes.push_back("\n#define USE_CUBEMAP_PASS\n#define USE_QUARTER_RES_PASS\n"); // Quarter res Cubemap
		sky_shader.shader.initialize(sky_modes, defines);
	}

	// register our shader funds
	storage->shader_set_data_request_function(RasterizerStorageRD::SHADER_TYPE_SKY, _create_sky_shader_funcs);
	storage->material_set_data_request_function(RasterizerStorageRD::SHADER_TYPE_SKY, _create_sky_material_funcs);

	{
		ShaderCompilerRD::DefaultIdentifierActions actions;

		actions.renames["COLOR"] = "color";
		actions.renames["ALPHA"] = "alpha";
		actions.renames["EYEDIR"] = "cube_normal";
		actions.renames["POSITION"] = "params.position_multiplier.xyz";
		actions.renames["SKY_COORDS"] = "panorama_coords";
		actions.renames["SCREEN_UV"] = "uv";
		actions.renames["TIME"] = "params.time";
		actions.renames["HALF_RES_COLOR"] = "half_res_color";
		actions.renames["QUARTER_RES_COLOR"] = "quarter_res_color";
		actions.renames["RADIANCE"] = "radiance";
		actions.renames["FOG"] = "custom_fog";
		actions.renames["LIGHT0_ENABLED"] = "directional_lights.data[0].enabled";
		actions.renames["LIGHT0_DIRECTION"] = "directional_lights.data[0].direction_energy.xyz";
		actions.renames["LIGHT0_ENERGY"] = "directional_lights.data[0].direction_energy.w";
		actions.renames["LIGHT0_COLOR"] = "directional_lights.data[0].color_size.xyz";
		actions.renames["LIGHT0_SIZE"] = "directional_lights.data[0].color_size.w";
		actions.renames["LIGHT1_ENABLED"] = "directional_lights.data[1].enabled";
		actions.renames["LIGHT1_DIRECTION"] = "directional_lights.data[1].direction_energy.xyz";
		actions.renames["LIGHT1_ENERGY"] = "directional_lights.data[1].direction_energy.w";
		actions.renames["LIGHT1_COLOR"] = "directional_lights.data[1].color_size.xyz";
		actions.renames["LIGHT1_SIZE"] = "directional_lights.data[1].color_size.w";
		actions.renames["LIGHT2_ENABLED"] = "directional_lights.data[2].enabled";
		actions.renames["LIGHT2_DIRECTION"] = "directional_lights.data[2].direction_energy.xyz";
		actions.renames["LIGHT2_ENERGY"] = "directional_lights.data[2].direction_energy.w";
		actions.renames["LIGHT2_COLOR"] = "directional_lights.data[2].color_size.xyz";
		actions.renames["LIGHT2_SIZE"] = "directional_lights.data[2].color_size.w";
		actions.renames["LIGHT3_ENABLED"] = "directional_lights.data[3].enabled";
		actions.renames["LIGHT3_DIRECTION"] = "directional_lights.data[3].direction_energy.xyz";
		actions.renames["LIGHT3_ENERGY"] = "directional_lights.data[3].direction_energy.w";
		actions.renames["LIGHT3_COLOR"] = "directional_lights.data[3].color_size.xyz";
		actions.renames["LIGHT3_SIZE"] = "directional_lights.data[3].color_size.w";
		actions.renames["AT_CUBEMAP_PASS"] = "AT_CUBEMAP_PASS";
		actions.renames["AT_HALF_RES_PASS"] = "AT_HALF_RES_PASS";
		actions.renames["AT_QUARTER_RES_PASS"] = "AT_QUARTER_RES_PASS";
		actions.custom_samplers["RADIANCE"] = "material_samplers[3]";
		actions.usage_defines["HALF_RES_COLOR"] = "\n#define USES_HALF_RES_COLOR\n";
		actions.usage_defines["QUARTER_RES_COLOR"] = "\n#define USES_QUARTER_RES_COLOR\n";
		actions.render_mode_defines["disable_fog"] = "#define DISABLE_FOG\n";

		actions.sampler_array_name = "material_samplers";
		actions.base_texture_binding_index = 1;
		actions.texture_layout_set = 1;
		actions.base_uniform_string = "material.";
		actions.base_varying_index = 10;

		actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
		actions.default_repeat = ShaderLanguage::REPEAT_ENABLE;
		actions.global_buffer_array_variable = "global_variables.data";

		sky_shader.compiler.initialize(actions);
	}

	{
		// default material and shader for sky shader
		sky_shader.default_shader = storage->shader_create();
		storage->shader_set_code(sky_shader.default_shader, "shader_type sky; void fragment() { COLOR = vec3(0.0); } \n");
		sky_shader.default_material = storage->material_create();
		storage->material_set_shader(sky_shader.default_material, sky_shader.default_shader);

		SkyMaterialData *md = (SkyMaterialData *)storage->material_get_data(sky_shader.default_material, RasterizerStorageRD::SHADER_TYPE_SKY);
		sky_shader.default_shader_rd = sky_shader.shader.version_get_shader(md->shader_data->version, SKY_VERSION_BACKGROUND);

		sky_scene_state.uniform_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(SkySceneState::UBO));

		Vector<RD::Uniform> uniforms;

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 0;
			u.ids.resize(12);
			RID *ids_ptr = u.ids.ptrw();
			ids_ptr[0] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[1] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[2] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[3] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[4] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[5] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[6] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[7] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[8] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[9] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[10] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[11] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 1;
			u.ids.push_back(storage->global_variables_get_storage_buffer());
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 2;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.ids.push_back(sky_scene_state.uniform_buffer);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 3;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.ids.push_back(sky_scene_state.directional_light_buffer);
			uniforms.push_back(u);
		}

		sky_scene_state.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sky_shader.default_shader_rd, SKY_SET_UNIFORMS);
	}

	{
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.binding = 0;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			RID vfog = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE);
			u.ids.push_back(vfog);
			uniforms.push_back(u);
		}

		sky_scene_state.default_fog_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sky_shader.default_shader_rd, SKY_SET_FOG);
	}

	{
		// Need defaults for using fog with clear color
		sky_scene_state.fog_shader = storage->shader_create();
		storage->shader_set_code(sky_scene_state.fog_shader, "shader_type sky; uniform vec4 clear_color; void fragment() { COLOR = clear_color.rgb; } \n");
		sky_scene_state.fog_material = storage->material_create();
		storage->material_set_shader(sky_scene_state.fog_material, sky_scene_state.fog_shader);

		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 0;
			u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK));
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1;
			u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE));
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 2;
			u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE));
			uniforms.push_back(u);
		}

		sky_scene_state.fog_only_texture_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sky_shader.default_shader_rd, SKY_SET_TEXTURES);
	}

	{
		Vector<String> preprocess_modes;
		preprocess_modes.push_back("\n#define MODE_SCROLL\n");
		preprocess_modes.push_back("\n#define MODE_SCROLL_OCCLUSION\n");
		preprocess_modes.push_back("\n#define MODE_INITIALIZE_JUMP_FLOOD\n");
		preprocess_modes.push_back("\n#define MODE_INITIALIZE_JUMP_FLOOD_HALF\n");
		preprocess_modes.push_back("\n#define MODE_JUMPFLOOD\n");
		preprocess_modes.push_back("\n#define MODE_JUMPFLOOD_OPTIMIZED\n");
		preprocess_modes.push_back("\n#define MODE_UPSCALE_JUMP_FLOOD\n");
		preprocess_modes.push_back("\n#define MODE_OCCLUSION\n");
		preprocess_modes.push_back("\n#define MODE_STORE\n");
		String defines = "\n#define OCCLUSION_SIZE " + itos(SDFGI::CASCADE_SIZE / SDFGI::PROBE_DIVISOR) + "\n";
		sdfgi_shader.preprocess.initialize(preprocess_modes, defines);
		sdfgi_shader.preprocess_shader = sdfgi_shader.preprocess.version_create();
		for (int i = 0; i < SDGIShader::PRE_PROCESS_MAX; i++) {
			sdfgi_shader.preprocess_pipeline[i] = RD::get_singleton()->compute_pipeline_create(sdfgi_shader.preprocess.version_get_shader(sdfgi_shader.preprocess_shader, i));
		}
	}

	{
		//calculate tables
		String defines = "\n#define OCT_SIZE " + itos(SDFGI::LIGHTPROBE_OCT_SIZE) + "\n";

		Vector<String> direct_light_modes;
		direct_light_modes.push_back("\n#define MODE_PROCESS_STATIC\n");
		direct_light_modes.push_back("\n#define MODE_PROCESS_DYNAMIC\n");
		sdfgi_shader.direct_light.initialize(direct_light_modes, defines);
		sdfgi_shader.direct_light_shader = sdfgi_shader.direct_light.version_create();
		for (int i = 0; i < SDGIShader::DIRECT_LIGHT_MODE_MAX; i++) {
			sdfgi_shader.direct_light_pipeline[i] = RD::get_singleton()->compute_pipeline_create(sdfgi_shader.direct_light.version_get_shader(sdfgi_shader.direct_light_shader, i));
		}
	}

	{
		//calculate tables
		String defines = "\n#define OCT_SIZE " + itos(SDFGI::LIGHTPROBE_OCT_SIZE) + "\n";
		defines += "\n#define SH_SIZE " + itos(SDFGI::SH_SIZE) + "\n";

		Vector<String> integrate_modes;
		integrate_modes.push_back("\n#define MODE_PROCESS\n");
		integrate_modes.push_back("\n#define MODE_STORE\n");
		integrate_modes.push_back("\n#define MODE_SCROLL\n");
		integrate_modes.push_back("\n#define MODE_SCROLL_STORE\n");
		sdfgi_shader.integrate.initialize(integrate_modes, defines);
		sdfgi_shader.integrate_shader = sdfgi_shader.integrate.version_create();

		for (int i = 0; i < SDGIShader::INTEGRATE_MODE_MAX; i++) {
			sdfgi_shader.integrate_pipeline[i] = RD::get_singleton()->compute_pipeline_create(sdfgi_shader.integrate.version_get_shader(sdfgi_shader.integrate_shader, i));
		}

		{
			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 0;
				u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE));
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_SAMPLER;
				u.binding = 1;
				u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
				uniforms.push_back(u);
			}

			sdfgi_shader.integrate_default_sky_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sdfgi_shader.integrate.version_get_shader(sdfgi_shader.integrate_shader, 0), 1);
		}
	}
	{
		//calculate tables
		String defines = "\n#define SDFGI_OCT_SIZE " + itos(SDFGI::LIGHTPROBE_OCT_SIZE) + "\n";
		Vector<String> gi_modes;
		gi_modes.push_back("");
		gi.shader.initialize(gi_modes, defines);
		gi.shader_version = gi.shader.version_create();
		for (int i = 0; i < GI::MODE_MAX; i++) {
			gi.pipelines[i] = RD::get_singleton()->compute_pipeline_create(gi.shader.version_get_shader(gi.shader_version, i));
		}

		gi.sdfgi_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(GI::SDFGIData));
	}
	{
		String defines = "\n#define OCT_SIZE " + itos(SDFGI::LIGHTPROBE_OCT_SIZE) + "\n";
		Vector<String> debug_modes;
		debug_modes.push_back("");
		sdfgi_shader.debug.initialize(debug_modes, defines);
		sdfgi_shader.debug_shader = sdfgi_shader.debug.version_create();
		sdfgi_shader.debug_shader_version = sdfgi_shader.debug.version_get_shader(sdfgi_shader.debug_shader, 0);
		sdfgi_shader.debug_pipeline = RD::get_singleton()->compute_pipeline_create(sdfgi_shader.debug_shader_version);
	}
	{
		String defines = "\n#define OCT_SIZE " + itos(SDFGI::LIGHTPROBE_OCT_SIZE) + "\n";

		Vector<String> versions;
		versions.push_back("\n#define MODE_PROBES\n");
		versions.push_back("\n#define MODE_VISIBILITY\n");

		sdfgi_shader.debug_probes.initialize(versions, defines);
		sdfgi_shader.debug_probes_shader = sdfgi_shader.debug_probes.version_create();

		{
			RD::PipelineRasterizationState rs;
			rs.cull_mode = RD::POLYGON_CULL_DISABLED;
			RD::PipelineDepthStencilState ds;
			ds.enable_depth_test = true;
			ds.enable_depth_write = true;
			ds.depth_compare_operator = RD::COMPARE_OP_LESS_OR_EQUAL;
			for (int i = 0; i < SDGIShader::PROBE_DEBUG_MAX; i++) {
				RID debug_probes_shader_version = sdfgi_shader.debug_probes.version_get_shader(sdfgi_shader.debug_probes_shader, i);
				sdfgi_shader.debug_probes_pipeline[i].setup(debug_probes_shader_version, RD::RENDER_PRIMITIVE_TRIANGLE_STRIPS, rs, RD::PipelineMultisampleState(), ds, RD::PipelineColorBlendState::create_disabled(), 0);
			}
		}
	}

	//cluster setup
	uint32_t uniform_max_size = RD::get_singleton()->limit_get(RD::LIMIT_MAX_UNIFORM_BUFFER_SIZE);

	{ //reflections
		uint32_t reflection_buffer_size;
		if (uniform_max_size < 65536) {
			//Yes, you guessed right, ARM again
			reflection_buffer_size = uniform_max_size;
		} else {
			reflection_buffer_size = 65536;
		}

		cluster.max_reflections = reflection_buffer_size / sizeof(Cluster::ReflectionData);
		cluster.reflections = memnew_arr(Cluster::ReflectionData, cluster.max_reflections);
		cluster.reflection_buffer = RD::get_singleton()->storage_buffer_create(reflection_buffer_size);
	}

	{ //lights
		cluster.max_lights = MIN(1024 * 1024, uniform_max_size) / sizeof(Cluster::LightData); //1mb of lights
		uint32_t light_buffer_size = cluster.max_lights * sizeof(Cluster::LightData);
		cluster.lights = memnew_arr(Cluster::LightData, cluster.max_lights);
		cluster.light_buffer = RD::get_singleton()->storage_buffer_create(light_buffer_size);
		//defines += "\n#define MAX_LIGHT_DATA_STRUCTS " + itos(cluster.max_lights) + "\n";
		cluster.lights_instances = memnew_arr(RID, cluster.max_lights);
		cluster.lights_shadow_rect_cache = memnew_arr(Rect2i, cluster.max_lights);

		cluster.max_directional_lights = 8;
		uint32_t directional_light_buffer_size = cluster.max_directional_lights * sizeof(Cluster::DirectionalLightData);
		cluster.directional_lights = memnew_arr(Cluster::DirectionalLightData, cluster.max_directional_lights);
		cluster.directional_light_buffer = RD::get_singleton()->uniform_buffer_create(directional_light_buffer_size);
	}

	{ //decals
		cluster.max_decals = MIN(1024 * 1024, uniform_max_size) / sizeof(Cluster::DecalData); //1mb of decals
		uint32_t decal_buffer_size = cluster.max_decals * sizeof(Cluster::DecalData);
		cluster.decals = memnew_arr(Cluster::DecalData, cluster.max_decals);
		cluster.decal_buffer = RD::get_singleton()->storage_buffer_create(decal_buffer_size);
	}

	cluster.builder.setup(16, 8, 24);

	{
		String defines = "\n#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS " + itos(cluster.max_directional_lights) + "\n";
		Vector<String> volumetric_fog_modes;
		volumetric_fog_modes.push_back("\n#define MODE_DENSITY\n");
		volumetric_fog_modes.push_back("\n#define MODE_DENSITY\n#define ENABLE_SDFGI\n");
		volumetric_fog_modes.push_back("\n#define MODE_FILTER\n");
		volumetric_fog_modes.push_back("\n#define MODE_FOG\n");
		volumetric_fog.shader.initialize(volumetric_fog_modes, defines);
		volumetric_fog.shader_version = volumetric_fog.shader.version_create();
		for (int i = 0; i < VOLUMETRIC_FOG_SHADER_MAX; i++) {
			volumetric_fog.pipelines[i] = RD::get_singleton()->compute_pipeline_create(volumetric_fog.shader.version_get_shader(volumetric_fog.shader_version, i));
		}
	}
	default_giprobe_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(GI::GIProbeData) * RenderBuffers::MAX_GIPROBES);

	{
		RD::SamplerState sampler;
		sampler.mag_filter = RD::SAMPLER_FILTER_NEAREST;
		sampler.min_filter = RD::SAMPLER_FILTER_NEAREST;
		sampler.enable_compare = true;
		sampler.compare_op = RD::COMPARE_OP_LESS;
		shadow_sampler = RD::get_singleton()->sampler_create(sampler);
	}

	camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape(int(GLOBAL_GET("rendering/quality/depth_of_field/depth_of_field_bokeh_shape"))));
	camera_effects_set_dof_blur_quality(RS::DOFBlurQuality(int(GLOBAL_GET("rendering/quality/depth_of_field/depth_of_field_bokeh_quality"))), GLOBAL_GET("rendering/quality/depth_of_field/depth_of_field_use_jitter"));
	environment_set_ssao_quality(RS::EnvironmentSSAOQuality(int(GLOBAL_GET("rendering/quality/ssao/quality"))), GLOBAL_GET("rendering/quality/ssao/half_size"));
	screen_space_roughness_limiter = GLOBAL_GET("rendering/quality/screen_filters/screen_space_roughness_limiter_enabled");
	screen_space_roughness_limiter_amount = GLOBAL_GET("rendering/quality/screen_filters/screen_space_roughness_limiter_amount");
	screen_space_roughness_limiter_limit = GLOBAL_GET("rendering/quality/screen_filters/screen_space_roughness_limiter_limit");
	glow_bicubic_upscale = int(GLOBAL_GET("rendering/quality/glow/upscale_mode")) > 0;
	glow_high_quality = GLOBAL_GET("rendering/quality/glow/use_high_quality");
	ssr_roughness_quality = RS::EnvironmentSSRRoughnessQuality(int(GLOBAL_GET("rendering/quality/screen_space_reflection/roughness_quality")));
	sss_quality = RS::SubSurfaceScatteringQuality(int(GLOBAL_GET("rendering/quality/subsurface_scattering/subsurface_scattering_quality")));
	sss_scale = GLOBAL_GET("rendering/quality/subsurface_scattering/subsurface_scattering_scale");
	sss_depth_scale = GLOBAL_GET("rendering/quality/subsurface_scattering/subsurface_scattering_depth_scale");
	directional_penumbra_shadow_kernel = memnew_arr(float, 128);
	directional_soft_shadow_kernel = memnew_arr(float, 128);
	penumbra_shadow_kernel = memnew_arr(float, 128);
	soft_shadow_kernel = memnew_arr(float, 128);
	shadows_quality_set(RS::ShadowQuality(int(GLOBAL_GET("rendering/quality/shadows/soft_shadow_quality"))));
	directional_shadow_quality_set(RS::ShadowQuality(int(GLOBAL_GET("rendering/quality/directional_shadow/soft_shadow_quality"))));

	environment_set_volumetric_fog_volume_size(GLOBAL_GET("rendering/volumetric_fog/volume_size"), GLOBAL_GET("rendering/volumetric_fog/volume_depth"));
	environment_set_volumetric_fog_filter_active(GLOBAL_GET("rendering/volumetric_fog/use_filter"));
	environment_set_volumetric_fog_directional_shadow_shrink_size(GLOBAL_GET("rendering/volumetric_fog/directional_shadow_shrink"));
	environment_set_volumetric_fog_positional_shadow_shrink_size(GLOBAL_GET("rendering/volumetric_fog/positional_shadow_shrink"));
}

RasterizerSceneRD::~RasterizerSceneRD() {
	for (Map<Vector2i, ShadowMap>::Element *E = shadow_maps.front(); E; E = E->next()) {
		RD::get_singleton()->free(E->get().depth);
	}
	for (Map<int, ShadowCubemap>::Element *E = shadow_cubemaps.front(); E; E = E->next()) {
		RD::get_singleton()->free(E->get().cubemap);
	}

	if (sky_scene_state.uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(sky_scene_state.uniform_set)) {
		RD::get_singleton()->free(sky_scene_state.uniform_set);
	}

	RD::get_singleton()->free(default_giprobe_buffer);
	RD::get_singleton()->free(gi_probe_lights_uniform);
	RD::get_singleton()->free(gi.sdfgi_ubo);

	giprobe_debug_shader.version_free(giprobe_debug_shader_version);
	giprobe_shader.version_free(giprobe_lighting_shader_version);
	gi.shader.version_free(gi.shader_version);
	sdfgi_shader.debug_probes.version_free(sdfgi_shader.debug_probes_shader);
	sdfgi_shader.debug.version_free(sdfgi_shader.debug_shader);
	sdfgi_shader.direct_light.version_free(sdfgi_shader.direct_light_shader);
	sdfgi_shader.integrate.version_free(sdfgi_shader.integrate_shader);
	sdfgi_shader.preprocess.version_free(sdfgi_shader.preprocess_shader);

	volumetric_fog.shader.version_free(volumetric_fog.shader_version);

	memdelete_arr(gi_probe_lights);
	SkyMaterialData *md = (SkyMaterialData *)storage->material_get_data(sky_shader.default_material, RasterizerStorageRD::SHADER_TYPE_SKY);
	sky_shader.shader.version_free(md->shader_data->version);
	RD::get_singleton()->free(sky_scene_state.directional_light_buffer);
	RD::get_singleton()->free(sky_scene_state.uniform_buffer);
	memdelete_arr(sky_scene_state.directional_lights);
	memdelete_arr(sky_scene_state.last_frame_directional_lights);
	storage->free(sky_shader.default_shader);
	storage->free(sky_shader.default_material);
	storage->free(sky_scene_state.fog_shader);
	storage->free(sky_scene_state.fog_material);
	memdelete_arr(directional_penumbra_shadow_kernel);
	memdelete_arr(directional_soft_shadow_kernel);
	memdelete_arr(penumbra_shadow_kernel);
	memdelete_arr(soft_shadow_kernel);

	{
		RD::get_singleton()->free(cluster.directional_light_buffer);
		RD::get_singleton()->free(cluster.light_buffer);
		RD::get_singleton()->free(cluster.reflection_buffer);
		RD::get_singleton()->free(cluster.decal_buffer);
		memdelete_arr(cluster.directional_lights);
		memdelete_arr(cluster.lights);
		memdelete_arr(cluster.lights_shadow_rect_cache);
		memdelete_arr(cluster.lights_instances);
		memdelete_arr(cluster.reflections);
		memdelete_arr(cluster.decals);
	}

	RD::get_singleton()->free(shadow_sampler);

	directional_shadow_atlas_set_size(0);
}
