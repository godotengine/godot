/*************************************************************************/
/*  render_forward_mobile.cpp                                            */
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

#include "render_forward_mobile.h"
#include "core/config/project_settings.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_server_default.h"

using namespace RendererSceneRenderImplementation;

RenderForwardMobile::ForwardID RenderForwardMobile::_allocate_forward_id(ForwardIDType p_type) {
	int32_t index = -1;
	for (uint32_t i = 0; i < forward_id_allocators[p_type].allocations.size(); i++) {
		if (forward_id_allocators[p_type].allocations[i] == false) {
			index = i;
			break;
		}
	}

	if (index == -1) {
		index = forward_id_allocators[p_type].allocations.size();
		forward_id_allocators[p_type].allocations.push_back(true);
		forward_id_allocators[p_type].map.push_back(0xFF);
	} else {
		forward_id_allocators[p_type].allocations[index] = true;
	}

	return index;
}
void RenderForwardMobile::_free_forward_id(ForwardIDType p_type, ForwardID p_id) {
	ERR_FAIL_INDEX(p_id, (ForwardID)forward_id_allocators[p_type].allocations.size());
	forward_id_allocators[p_type].allocations[p_id] = false;
}

void RenderForwardMobile::_map_forward_id(ForwardIDType p_type, ForwardID p_id, uint32_t p_index) {
	forward_id_allocators[p_type].map[p_id] = p_index;
}

/* Render buffer */

void RenderForwardMobile::RenderBufferDataForwardMobile::clear() {
	if (color_msaa.is_valid()) {
		RD::get_singleton()->free(color_msaa);
		color_msaa = RID();
	}

	if (depth_msaa.is_valid()) {
		RD::get_singleton()->free(depth_msaa);
		depth_msaa = RID();
	}

	color = RID();
	depth = RID();
	for (int i = 0; i < FB_CONFIG_MAX; i++) {
		color_fbs[i] = RID();
	}
}

void RenderForwardMobile::RenderBufferDataForwardMobile::configure(RID p_color_buffer, RID p_depth_buffer, RID p_target_buffer, int p_width, int p_height, RS::ViewportMSAA p_msaa, uint32_t p_view_count) {
	clear();

	msaa = p_msaa;

	Size2i target_size = RD::get_singleton()->texture_size(p_target_buffer);

	width = p_width;
	height = p_height;
	bool is_scaled = (target_size.width != p_width) || (target_size.height != p_height);
	view_count = p_view_count;

	color = p_color_buffer;
	depth = p_depth_buffer;

	// We are creating 4 configurations here for our framebuffers.

	if (p_msaa == RS::VIEWPORT_MSAA_DISABLED) {
		Vector<RID> fb;
		fb.push_back(p_color_buffer); // 0 - color buffer
		fb.push_back(depth); // 1 - depth buffer

		// Now define our subpasses
		Vector<RD::FramebufferPass> passes;
		RD::FramebufferPass pass;

		// re-using the same attachments
		pass.color_attachments.push_back(0);
		pass.depth_attachment = 1;

		// - opaque pass
		passes.push_back(pass);
		color_fbs[FB_CONFIG_ONE_PASS] = RD::get_singleton()->framebuffer_create_multipass(fb, passes, RenderingDevice::INVALID_ID, view_count);

		// - add sky pass
		passes.push_back(pass);
		color_fbs[FB_CONFIG_TWO_SUBPASSES] = RD::get_singleton()->framebuffer_create_multipass(fb, passes, RenderingDevice::INVALID_ID, view_count);

		// - add alpha pass
		passes.push_back(pass);
		color_fbs[FB_CONFIG_THREE_SUBPASSES] = RD::get_singleton()->framebuffer_create_multipass(fb, passes, RenderingDevice::INVALID_ID, view_count);

		if (!is_scaled) {
			// - add blit to 2D pass
			fb.push_back(p_target_buffer); // 2 - target buffer

			RD::FramebufferPass blit_pass;
			blit_pass.color_attachments.push_back(2);
			blit_pass.input_attachments.push_back(0);
			passes.push_back(blit_pass);

			color_fbs[FB_CONFIG_FOUR_SUBPASSES] = RD::get_singleton()->framebuffer_create_multipass(fb, passes, RenderingDevice::INVALID_ID, view_count);
		} else {
			// can't do our blit pass if resolutions don't match
			color_fbs[FB_CONFIG_FOUR_SUBPASSES] = RID();
		}
	} else {
		RD::DataFormat color_format = RenderForwardMobile::singleton->_render_buffers_get_color_format();

		RD::TextureFormat tf;
		if (view_count > 1) {
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		} else {
			tf.texture_type = RD::TEXTURE_TYPE_2D;
		}
		tf.format = color_format;
		tf.width = p_width;
		tf.height = p_height;
		tf.array_layers = view_count; // create a layer for every view
		tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

		RD::TextureSamples ts[RS::VIEWPORT_MSAA_MAX] = {
			RD::TEXTURE_SAMPLES_1,
			RD::TEXTURE_SAMPLES_2,
			RD::TEXTURE_SAMPLES_4,
			RD::TEXTURE_SAMPLES_8,
		};

		texture_samples = ts[p_msaa];
		tf.samples = texture_samples;

		color_msaa = RD::get_singleton()->texture_create(tf, RD::TextureView());

		tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D24_UNORM_S8_UINT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D24_UNORM_S8_UINT : RD::DATA_FORMAT_D32_SFLOAT_S8_UINT;
		tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

		depth_msaa = RD::get_singleton()->texture_create(tf, RD::TextureView());

		{
			Vector<RID> fb;
			fb.push_back(color_msaa); // 0 - msaa color buffer
			fb.push_back(depth_msaa); // 1 - msaa depth buffer

			// Now define our subpasses
			Vector<RD::FramebufferPass> passes;
			RD::FramebufferPass pass;

			// re-using the same attachments
			pass.color_attachments.push_back(0);
			pass.depth_attachment = 1;

			// - opaque pass
			passes.push_back(pass);

			// - add sky pass
			fb.push_back(color); // 2 - color buffer
			passes.push_back(pass); // without resolve for our 3 + 4 subpass config
			{
				// but with resolve for our 2 subpass config
				Vector<RD::FramebufferPass> two_passes;
				two_passes.push_back(pass); // opaque subpass without resolve
				pass.resolve_attachments.push_back(2);
				two_passes.push_back(pass); // sky subpass with resolve

				color_fbs[FB_CONFIG_TWO_SUBPASSES] = RD::get_singleton()->framebuffer_create_multipass(fb, two_passes, RenderingDevice::INVALID_ID, view_count);
			}

			// - add alpha pass (with resolve, we just added that above)
			passes.push_back(pass);
			color_fbs[FB_CONFIG_THREE_SUBPASSES] = RD::get_singleton()->framebuffer_create_multipass(fb, passes, RenderingDevice::INVALID_ID, view_count);

			{
				// we also need our one pass with resolve
				Vector<RD::FramebufferPass> one_pass_with_resolve;
				one_pass_with_resolve.push_back(pass); // note our pass configuration already has resolve..
				color_fbs[FB_CONFIG_ONE_PASS] = RD::get_singleton()->framebuffer_create_multipass(fb, one_pass_with_resolve, RenderingDevice::INVALID_ID, view_count);
			}

			if (!is_scaled) {
				// - add blit to 2D pass
				fb.push_back(p_target_buffer); // 3 - target buffer
				RD::FramebufferPass blit_pass;
				blit_pass.color_attachments.push_back(3);
				blit_pass.input_attachments.push_back(2);
				passes.push_back(blit_pass);

				color_fbs[FB_CONFIG_FOUR_SUBPASSES] = RD::get_singleton()->framebuffer_create_multipass(fb, passes, RenderingDevice::INVALID_ID, view_count);
			} else {
				// can't do our blit pass if resolutions don't match
				color_fbs[FB_CONFIG_FOUR_SUBPASSES] = RID();
			}
		}
	}
}

RID RenderForwardMobile::reflection_probe_create_framebuffer(RID p_color, RID p_depth) {
	// Our attachments
	Vector<RID> fb;
	fb.push_back(p_color); // 0
	fb.push_back(p_depth); // 1

	// Now define our subpasses
	Vector<RD::FramebufferPass> passes;
	RD::FramebufferPass pass;

	// re-using the same attachments
	pass.color_attachments.push_back(0);
	pass.depth_attachment = 1;

	// - opaque pass
	passes.push_back(pass);

	// - sky pass
	passes.push_back(pass);

	// - alpha pass
	passes.push_back(pass);

	return RD::get_singleton()->framebuffer_create_multipass(fb, passes);
}

RenderForwardMobile::RenderBufferDataForwardMobile::~RenderBufferDataForwardMobile() {
	clear();
}

RendererSceneRenderRD::RenderBufferData *RenderForwardMobile::_create_render_buffer_data() {
	return memnew(RenderBufferDataForwardMobile);
}

bool RenderForwardMobile::free(RID p_rid) {
	if (RendererSceneRenderRD::free(p_rid)) {
		return true;
	}
	return false;
}

/* Render functions */

float RenderForwardMobile::_render_buffers_get_luminance_multiplier() {
	// On mobile renderer we need to multiply source colors by 2 due to using a UNORM buffer
	// and multiplying by the output color during 3D rendering by 0.5
	return 2.0;
}

RD::DataFormat RenderForwardMobile::_render_buffers_get_color_format() {
	// Using 32bit buffers enables AFBC on mobile devices which should have a definite performance improvement (MALI G710 and newer support this on 64bit RTs)
	return RD::DATA_FORMAT_A2B10G10R10_UNORM_PACK32;
}

bool RenderForwardMobile::_render_buffers_can_be_storage() {
	// Using 32bit buffers enables AFBC on mobile devices which should have a definite performance improvement (MALI G710 and newer support this on 64bit RTs)
	// Doesn't support storage
	return false;
}

RID RenderForwardMobile::_setup_render_pass_uniform_set(RenderListType p_render_list, const RenderDataRD *p_render_data, RID p_radiance_texture, bool p_use_directional_shadow_atlas, int p_index) {
	//there should always be enough uniform buffers for render passes, otherwise bugs
	ERR_FAIL_INDEX_V(p_index, (int)scene_state.uniform_buffers.size(), RID());

	RenderBufferDataForwardMobile *rb = nullptr;
	if (p_render_data && p_render_data->render_buffers.is_valid()) {
		rb = (RenderBufferDataForwardMobile *)render_buffers_get_data(p_render_data->render_buffers);
	}

	// default render buffer and scene state uniform set
	// loaded into set 1

	Vector<RD::Uniform> uniforms;

	{
		RD::Uniform u;
		u.binding = 0;
		u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
		u.ids.push_back(scene_state.uniform_buffers[p_index]);
		uniforms.push_back(u);
	}

	{
		RID radiance_texture;
		if (p_radiance_texture.is_valid()) {
			radiance_texture = p_radiance_texture;
		} else {
			radiance_texture = storage->texture_rd_get_default(is_using_radiance_cubemap_array() ? RendererStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK : RendererStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK);
		}
		RD::Uniform u;
		u.binding = 2;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.ids.push_back(radiance_texture);
		uniforms.push_back(u);
	}

	{
		RID ref_texture = (p_render_data && p_render_data->reflection_atlas.is_valid()) ? reflection_atlas_get_texture(p_render_data->reflection_atlas) : RID();
		RD::Uniform u;
		u.binding = 3;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		if (ref_texture.is_valid()) {
			u.ids.push_back(ref_texture);
		} else {
			u.ids.push_back(storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK));
		}
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.binding = 4;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		RID texture;
		if (p_render_data && p_render_data->shadow_atlas.is_valid()) {
			texture = shadow_atlas_get_texture(p_render_data->shadow_atlas);
		}
		if (!texture.is_valid()) {
			texture = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_WHITE);
		}
		u.ids.push_back(texture);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.binding = 5;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		if (p_use_directional_shadow_atlas && directional_shadow_get_texture().is_valid()) {
			u.ids.push_back(directional_shadow_get_texture());
		} else {
			u.ids.push_back(storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_WHITE));
		}
		uniforms.push_back(u);
	}

	/* we have limited ability to keep textures like this so we're moving this to a set we change before drawing geometry and just pushing the needed texture in */
	{
		RD::Uniform u;
		u.binding = 6;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.ids.resize(scene_state.max_lightmaps);
		RID default_tex = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE);
		for (uint32_t i = 0; i < scene_state.max_lightmaps; i++) {
			if (p_render_data && i < p_render_data->lightmaps->size()) {
				RID base = lightmap_instance_get_lightmap((*p_render_data->lightmaps)[i]);
				RID texture = storage->lightmap_get_texture(base);
				RID rd_texture = storage->texture_get_rd_texture(texture);
				u.ids.write[i] = rd_texture;
			} else {
				u.ids.write[i] = default_tex;
			}
		}

		uniforms.push_back(u);
	}

	/*
	{
		RD::Uniform u;
		u.binding = 7;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.ids.resize(MAX_VOXEL_GI_INSTANCESS);
		RID default_tex = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE);
		for (int i = 0; i < MAX_VOXEL_GI_INSTANCESS; i++) {
			if (i < (int)p_voxel_gi_instances.size()) {
				RID tex = gi.voxel_gi_instance_get_texture(p_voxel_gi_instances[i]);
				if (!tex.is_valid()) {
					tex = default_tex;
				}
				u.ids.write[i] = tex;
			} else {
				u.ids.write[i] = default_tex;
			}
		}

		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.binding = 8;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		RID cb = p_cluster_buffer.is_valid() ? p_cluster_buffer : default_vec4_xform_buffer;
		u.ids.push_back(cb);
		uniforms.push_back(u);
	}
	*/

	{
		RD::Uniform u;
		u.binding = 9;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		RID dbt = rb ? render_buffers_get_back_depth_texture(p_render_data->render_buffers) : RID();
		RID texture = (dbt.is_valid()) ? dbt : storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_WHITE);
		u.ids.push_back(texture);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.binding = 10;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		RID bbt = rb ? render_buffers_get_back_buffer_texture(p_render_data->render_buffers) : RID();
		RID texture = bbt.is_valid() ? bbt : storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_BLACK);
		u.ids.push_back(texture);
		uniforms.push_back(u);
	}

	if (p_index >= (int)render_pass_uniform_sets.size()) {
		render_pass_uniform_sets.resize(p_index + 1);
	}

	if (render_pass_uniform_sets[p_index].is_valid() && RD::get_singleton()->uniform_set_is_valid(render_pass_uniform_sets[p_index])) {
		RD::get_singleton()->free(render_pass_uniform_sets[p_index]);
	}

	render_pass_uniform_sets[p_index] = RD::get_singleton()->uniform_set_create(uniforms, scene_shader.default_shader_rd, RENDER_PASS_UNIFORM_SET);
	return render_pass_uniform_sets[p_index];
}

void RenderForwardMobile::_setup_lightmaps(const PagedArray<RID> &p_lightmaps, const Transform3D &p_cam_transform) {
	// This probably needs to change...
	scene_state.lightmaps_used = 0;
	for (int i = 0; i < (int)p_lightmaps.size(); i++) {
		if (i >= (int)scene_state.max_lightmaps) {
			break;
		}

		RID lightmap = lightmap_instance_get_lightmap(p_lightmaps[i]);

		Basis to_lm = lightmap_instance_get_transform(p_lightmaps[i]).basis.inverse() * p_cam_transform.basis;
		to_lm = to_lm.inverse().transposed(); //will transform normals
		RendererStorageRD::store_transform_3x3(to_lm, scene_state.lightmaps[i].normal_xform);
		scene_state.lightmap_ids[i] = p_lightmaps[i];
		scene_state.lightmap_has_sh[i] = storage->lightmap_uses_spherical_harmonics(lightmap);

		scene_state.lightmaps_used++;
	}
	if (scene_state.lightmaps_used > 0) {
		RD::get_singleton()->buffer_update(scene_state.lightmap_buffer, 0, sizeof(LightmapData) * scene_state.lightmaps_used, scene_state.lightmaps, RD::BARRIER_MASK_RASTER);
	}
}

void RenderForwardMobile::_render_scene(RenderDataRD *p_render_data, const Color &p_default_bg_color) {
	RenderBufferDataForwardMobile *render_buffer = nullptr;
	if (p_render_data->render_buffers.is_valid()) {
		render_buffer = (RenderBufferDataForwardMobile *)render_buffers_get_data(p_render_data->render_buffers);
	}
	RendererSceneEnvironmentRD *env = get_environment(p_render_data->environment);

	RENDER_TIMESTAMP("Setup 3D Scene");

	Vector2 vp_he = p_render_data->cam_projection.get_viewport_half_extents();
	scene_state.ubo.viewport_size[0] = vp_he.x;
	scene_state.ubo.viewport_size[1] = vp_he.y;
	scene_state.ubo.directional_light_count = 0;

	// We can only use our full subpass approach if we're:
	// - not reading from SCREEN_TEXTURE/DEPTH_TEXTURE
	// - not using ssr/sss (currently not supported)
	// - not using glow or other post effects (can't do 4th subpass)
	// - rendering to a half sized render buffer (can't do 4th subpass)
	// We'll need to restrict how far we're going with subpasses based on this.

	Size2i screen_size;
	RID framebuffer;
	bool reverse_cull = false;
	bool using_subpass_transparent = true;
	bool using_subpass_post_process = true;

	bool using_ssr = false; // I don't think we support this in our mobile renderer so probably should phase it out
	bool using_sss = false; // I don't think we support this in our mobile renderer so probably should phase it out

	// fill our render lists early so we can find out if we use various features
	_fill_render_list(RENDER_LIST_OPAQUE, p_render_data, PASS_MODE_COLOR);
	render_list[RENDER_LIST_OPAQUE].sort_by_key();
	render_list[RENDER_LIST_ALPHA].sort_by_reverse_depth_and_priority();
	_fill_element_info(RENDER_LIST_OPAQUE);
	_fill_element_info(RENDER_LIST_ALPHA);

	if (p_render_data->render_info) {
		p_render_data->render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RS::VIEWPORT_RENDER_INFO_DRAW_CALLS_IN_FRAME] = p_render_data->instances->size();
		p_render_data->render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RS::VIEWPORT_RENDER_INFO_OBJECTS_IN_FRAME] = p_render_data->instances->size();
	}

	if (render_buffer) {
		// setup rendering to render buffer
		screen_size.x = render_buffer->width;
		screen_size.y = render_buffer->height;

		if (render_buffer->color_fbs[FB_CONFIG_FOUR_SUBPASSES].is_null()) {
			// can't do blit subpass
			using_subpass_post_process = false;
		} else if (env && (env->glow_enabled || env->auto_exposure || camera_effects_uses_dof(p_render_data->camera_effects))) {
			// can't do blit subpass
			using_subpass_post_process = false;
		}

		if (using_ssr || using_sss || scene_state.used_screen_texture || scene_state.used_depth_texture) {
			// can't use our last two subpasses
			using_subpass_transparent = false;
			using_subpass_post_process = false;
		}

		if (using_subpass_post_process) {
			// all as subpasses
			framebuffer = render_buffer->color_fbs[FB_CONFIG_FOUR_SUBPASSES];
		} else if (using_subpass_transparent) {
			// our tonemap pass is separate
			framebuffer = render_buffer->color_fbs[FB_CONFIG_THREE_SUBPASSES];
		} else {
			// only opaque and sky as subpasses
			framebuffer = render_buffer->color_fbs[FB_CONFIG_TWO_SUBPASSES];
		}
	} else if (p_render_data->reflection_probe.is_valid()) {
		uint32_t resolution = reflection_probe_instance_get_resolution(p_render_data->reflection_probe);
		screen_size.x = resolution;
		screen_size.y = resolution;

		framebuffer = reflection_probe_instance_get_framebuffer(p_render_data->reflection_probe, p_render_data->reflection_probe_pass);

		if (storage->reflection_probe_is_interior(reflection_probe_instance_get_probe(p_render_data->reflection_probe))) {
			p_render_data->environment = RID(); //no environment on interiors
			env = nullptr;
		}

		reverse_cull = true;
		using_subpass_transparent = true; // we ignore our screen/depth texture here
		using_subpass_post_process = false; // not applicable at all for reflection probes.
	} else {
		ERR_FAIL(); //bug?
	}

	RD::get_singleton()->draw_command_begin_label("Render Setup");

	_setup_lightmaps(*p_render_data->lightmaps, p_render_data->cam_transform);
	_setup_environment(p_render_data, p_render_data->reflection_probe.is_valid(), screen_size, !p_render_data->reflection_probe.is_valid(), p_default_bg_color, false);

	_update_render_base_uniform_set(); //may have changed due to the above (light buffer enlarged, as an example)

	RD::get_singleton()->draw_command_end_label(); // Render Setup

	// setup environment
	RID radiance_texture;
	bool draw_sky = false;
	bool draw_sky_fog_only = false;

	Color clear_color = p_default_bg_color;
	bool keep_color = false;

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_OVERDRAW) {
		clear_color = Color(0, 0, 0, 1); //in overdraw mode, BG should always be black
	} else if (is_environment(p_render_data->environment)) {
		RS::EnvironmentBG bg_mode = environment_get_background(p_render_data->environment);
		float bg_energy = environment_get_bg_energy(p_render_data->environment);
		switch (bg_mode) {
			case RS::ENV_BG_CLEAR_COLOR: {
				clear_color = p_default_bg_color;
				clear_color.r *= bg_energy;
				clear_color.g *= bg_energy;
				clear_color.b *= bg_energy;
				/*
				if (render_buffers_has_volumetric_fog(p_render_data->render_buffers) || environment_is_fog_enabled(p_render_data->environment)) {
					draw_sky_fog_only = true;
					storage->material_set_param(sky.sky_scene_state.fog_material, "clear_color", Variant(clear_color.to_linear()));
				}
				*/
			} break;
			case RS::ENV_BG_COLOR: {
				clear_color = environment_get_bg_color(p_render_data->environment);
				clear_color.r *= bg_energy;
				clear_color.g *= bg_energy;
				clear_color.b *= bg_energy;
				/*
				if (render_buffers_has_volumetric_fog(p_render_data->render_buffers) || environment_is_fog_enabled(p_render_data->environment)) {
					draw_sky_fog_only = true;
					storage->material_set_param(sky.sky_scene_state.fog_material, "clear_color", Variant(clear_color.to_linear()));
				}
				*/
			} break;
			case RS::ENV_BG_SKY: {
				draw_sky = true;
			} break;
			case RS::ENV_BG_CANVAS: {
				keep_color = true;
			} break;
			case RS::ENV_BG_KEEP: {
				keep_color = true;
			} break;
			case RS::ENV_BG_CAMERA_FEED: {
			} break;
			default: {
			}
		}
		// setup sky if used for ambient, reflections, or background
		if (draw_sky || draw_sky_fog_only || environment_get_reflection_source(p_render_data->environment) == RS::ENV_REFLECTION_SOURCE_SKY || environment_get_ambient_source(p_render_data->environment) == RS::ENV_AMBIENT_SOURCE_SKY) {
			RENDER_TIMESTAMP("Setup Sky");
			RD::get_singleton()->draw_command_begin_label("Setup Sky");
			CameraMatrix projection = p_render_data->cam_projection;
			if (p_render_data->reflection_probe.is_valid()) {
				CameraMatrix correction;
				correction.set_depth_correction(true);
				projection = correction * p_render_data->cam_projection;
			}

			sky.setup(env, p_render_data->render_buffers, projection, p_render_data->cam_transform, screen_size, this);

			RID sky_rid = env->sky;
			if (sky_rid.is_valid()) {
				sky.update(env, projection, p_render_data->cam_transform, time, _render_buffers_get_luminance_multiplier());
				radiance_texture = sky.sky_get_radiance_texture_rd(sky_rid);
			} else {
				// do not try to draw sky if invalid
				draw_sky = false;
			}
			RD::get_singleton()->draw_command_end_label(); // Setup Sky
		}
	} else {
		clear_color = p_default_bg_color;
	}

	// update sky buffers (if required)
	if (draw_sky || draw_sky_fog_only) {
		// !BAS! @TODO See if we can limit doing some things double and maybe even move this into _pre_opaque_render
		// and change Forward Clustered in the same way as we have here (but without using subpasses)
		RENDER_TIMESTAMP("Setup Sky resolution buffers");

		RD::get_singleton()->draw_command_begin_label("Setup Sky resolution buffers");

		if (p_render_data->reflection_probe.is_valid()) {
			CameraMatrix correction;
			correction.set_depth_correction(true);
			CameraMatrix projection = correction * p_render_data->cam_projection;
			sky.update_res_buffers(env, 1, &projection, p_render_data->cam_transform, time);
		} else {
			sky.update_res_buffers(env, p_render_data->view_count, p_render_data->view_projection, p_render_data->cam_transform, time);
		}

		RD::get_singleton()->draw_command_end_label(); // Setup Sky resolution buffers
	}

	_pre_opaque_render(p_render_data, false, false, false, RID(), RID());

	uint32_t spec_constant_base_flags = 0;

	{
		//figure out spec constants

		if (p_render_data->directional_light_count > 0) {
			if (p_render_data->directional_light_soft_shadows) {
				spec_constant_base_flags |= 1 << SPEC_CONSTANT_USING_DIRECTIONAL_SOFT_SHADOWS;
			}
		} else {
			spec_constant_base_flags |= 1 << SPEC_CONSTANT_DISABLE_DIRECTIONAL_LIGHTS;
		}

		if (!is_environment(p_render_data->environment) || environment_is_fog_enabled(p_render_data->environment)) {
			spec_constant_base_flags |= 1 << SPEC_CONSTANT_DISABLE_FOG;
		}
	}
	{
		if (render_buffer) {
			RD::get_singleton()->draw_command_begin_label("Render 3D Pass");
		} else {
			RD::get_singleton()->draw_command_begin_label("Render Reflection Probe Pass");
		}

		// opaque pass

		RD::get_singleton()->draw_command_begin_label("Render Opaque Subpass");

		scene_state.ubo.directional_light_count = p_render_data->directional_light_count;

		_setup_environment(p_render_data, p_render_data->reflection_probe.is_valid(), screen_size, !p_render_data->reflection_probe.is_valid(), p_default_bg_color, p_render_data->render_buffers.is_valid());

		if (using_subpass_transparent && using_subpass_post_process) {
			RENDER_TIMESTAMP("Render Opaque + Transparent + Tonemap");
		} else if (using_subpass_transparent) {
			RENDER_TIMESTAMP("Render Opaque + Transparent");
		} else {
			RENDER_TIMESTAMP("Render Opaque");
		}

		RID rp_uniform_set = _setup_render_pass_uniform_set(RENDER_LIST_OPAQUE, p_render_data, radiance_texture, true);

		bool can_continue_color = !using_subpass_transparent && !scene_state.used_screen_texture && !using_ssr && !using_sss;
		bool can_continue_depth = !using_subpass_transparent && !scene_state.used_depth_texture && !using_ssr && !using_sss;

		{
			// regular forward for now
			Vector<Color> c;
			c.push_back(clear_color.to_linear()); // our render buffer
			if (render_buffer) {
				if (render_buffer->msaa != RS::VIEWPORT_MSAA_DISABLED) {
					c.push_back(clear_color.to_linear()); // our resolve buffer
				}
				if (using_subpass_post_process) {
					c.push_back(Color()); // our 2D buffer we're copying into
				}
			}

			RD::FramebufferFormatID fb_format = RD::get_singleton()->framebuffer_get_format(framebuffer);
			RenderListParameters render_list_params(render_list[RENDER_LIST_OPAQUE].elements.ptr(), render_list[RENDER_LIST_OPAQUE].element_info.ptr(), render_list[RENDER_LIST_OPAQUE].elements.size(), reverse_cull, PASS_MODE_COLOR, rp_uniform_set, spec_constant_base_flags, get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_WIREFRAME, Vector2(), p_render_data->lod_camera_plane, p_render_data->lod_distance_multiplier, p_render_data->screen_lod_threshold, p_render_data->view_count);
			render_list_params.framebuffer_format = fb_format;
			if ((uint32_t)render_list_params.element_count > render_list_thread_threshold && false) {
				// secondary command buffers need more testing at this time
				//multi threaded
				thread_draw_lists.resize(RendererThreadPool::singleton->thread_work_pool.get_thread_count());
				RD::get_singleton()->draw_list_begin_split(framebuffer, thread_draw_lists.size(), thread_draw_lists.ptr(), keep_color ? RD::INITIAL_ACTION_KEEP : RD::INITIAL_ACTION_CLEAR, can_continue_color ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, can_continue_depth ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ, c, 1.0, 0);
				RendererThreadPool::singleton->thread_work_pool.do_work(thread_draw_lists.size(), this, &RenderForwardMobile::_render_list_thread_function, &render_list_params);
			} else {
				//single threaded
				RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer, keep_color ? RD::INITIAL_ACTION_KEEP : RD::INITIAL_ACTION_CLEAR, can_continue_color ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, can_continue_depth ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ, c, 1.0, 0);
				_render_list(draw_list, fb_format, &render_list_params, 0, render_list_params.element_count);
			}
		}

		RD::get_singleton()->draw_command_end_label(); //Render Opaque Subpass

		if (draw_sky || draw_sky_fog_only) {
			RD::get_singleton()->draw_command_begin_label("Draw Sky Subpass");

			RD::DrawListID draw_list = RD::get_singleton()->draw_list_switch_to_next_pass();

			if (p_render_data->reflection_probe.is_valid()) {
				CameraMatrix correction;
				correction.set_depth_correction(true);
				CameraMatrix projection = correction * p_render_data->cam_projection;
				sky.draw(draw_list, env, framebuffer, 1, &projection, p_render_data->cam_transform, time, _render_buffers_get_luminance_multiplier());
			} else {
				sky.draw(draw_list, env, framebuffer, p_render_data->view_count, p_render_data->view_projection, p_render_data->cam_transform, time, _render_buffers_get_luminance_multiplier());
			}

			RD::get_singleton()->draw_command_end_label(); // Draw Sky Subpass

			// note, if MSAA is used in 2-subpass approach we should get an automatic resolve here
		} else {
			// switch to subpass but we do nothing here so basically we skip (though this should trigger resolve with 2-subpass MSAA).
			RD::get_singleton()->draw_list_switch_to_next_pass();
		}

		if (!using_subpass_transparent) {
			// We're done with our subpasses so end our container pass
			RD::get_singleton()->draw_list_end(RD::BARRIER_MASK_ALL);

			RD::get_singleton()->draw_command_end_label(); // Render 3D Pass / Render Reflection Probe Pass
		}

		if (scene_state.used_screen_texture) {
			// Copy screen texture to backbuffer so we can read from it
			_render_buffers_copy_screen_texture(p_render_data);
		}

		if (scene_state.used_depth_texture) {
			// Copy depth texture to backbuffer so we can read from it
			_render_buffers_copy_depth_texture(p_render_data);
		}

		// transparent pass

		RD::get_singleton()->draw_command_begin_label("Render Transparent Subpass");

		rp_uniform_set = _setup_render_pass_uniform_set(RENDER_LIST_ALPHA, p_render_data, radiance_texture, true);

		if (using_subpass_transparent) {
			RD::FramebufferFormatID fb_format = RD::get_singleton()->framebuffer_get_format(framebuffer);
			RenderListParameters render_list_params(render_list[RENDER_LIST_ALPHA].elements.ptr(), render_list[RENDER_LIST_ALPHA].element_info.ptr(), render_list[RENDER_LIST_ALPHA].elements.size(), reverse_cull, PASS_MODE_COLOR, rp_uniform_set, spec_constant_base_flags, get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_WIREFRAME, Vector2(), p_render_data->lod_camera_plane, p_render_data->lod_distance_multiplier, p_render_data->screen_lod_threshold, p_render_data->view_count);
			render_list_params.framebuffer_format = fb_format;
			if ((uint32_t)render_list_params.element_count > render_list_thread_threshold && false) {
				// secondary command buffers need more testing at this time
				//multi threaded
				thread_draw_lists.resize(RendererThreadPool::singleton->thread_work_pool.get_thread_count());
				RD::get_singleton()->draw_list_switch_to_next_pass_split(thread_draw_lists.size(), thread_draw_lists.ptr());
				render_list_params.subpass = RD::get_singleton()->draw_list_get_current_pass();
				RendererThreadPool::singleton->thread_work_pool.do_work(thread_draw_lists.size(), this, &RenderForwardMobile::_render_list_thread_function, &render_list_params);
			} else {
				//single threaded
				RD::DrawListID draw_list = RD::get_singleton()->draw_list_switch_to_next_pass();
				render_list_params.subpass = RD::get_singleton()->draw_list_get_current_pass();
				_render_list(draw_list, fb_format, &render_list_params, 0, render_list_params.element_count);
			}

			RD::get_singleton()->draw_command_end_label(); // Render Transparent Subpass

			// note if we are using MSAA we should get an automatic resolve through our subpass configuration.

			// blit to tonemap
			if (render_buffer && using_subpass_post_process) {
				_post_process_subpass(render_buffer->color, framebuffer, p_render_data);
			}

			RD::get_singleton()->draw_command_end_label(); // Render 3D Pass / Render Reflection Probe Pass

			RD::get_singleton()->draw_list_end(RD::BARRIER_MASK_ALL);
		} else {
			RENDER_TIMESTAMP("Render Transparent");

			framebuffer = render_buffer->color_fbs[FB_CONFIG_ONE_PASS];

			// this may be needed if we re-introduced steps that change info, not sure which do so in the previous implementation
			// _setup_environment(p_render_data, p_render_data->reflection_probe.is_valid(), screen_size, !p_render_data->reflection_probe.is_valid(), p_default_bg_color, false);

			RD::FramebufferFormatID fb_format = RD::get_singleton()->framebuffer_get_format(framebuffer);
			RenderListParameters render_list_params(render_list[RENDER_LIST_ALPHA].elements.ptr(), render_list[RENDER_LIST_ALPHA].element_info.ptr(), render_list[RENDER_LIST_ALPHA].elements.size(), reverse_cull, PASS_MODE_COLOR, rp_uniform_set, spec_constant_base_flags, get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_WIREFRAME, Vector2(), p_render_data->lod_camera_plane, p_render_data->lod_distance_multiplier, p_render_data->screen_lod_threshold, p_render_data->view_count);
			render_list_params.framebuffer_format = fb_format;
			if ((uint32_t)render_list_params.element_count > render_list_thread_threshold && false) {
				// secondary command buffers need more testing at this time
				//multi threaded
				thread_draw_lists.resize(RendererThreadPool::singleton->thread_work_pool.get_thread_count());
				RD::get_singleton()->draw_list_begin_split(framebuffer, thread_draw_lists.size(), thread_draw_lists.ptr(), can_continue_color ? RD::INITIAL_ACTION_CONTINUE : RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, can_continue_depth ? RD::INITIAL_ACTION_CONTINUE : RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ);
				RendererThreadPool::singleton->thread_work_pool.do_work(thread_draw_lists.size(), this, &RenderForwardMobile::_render_list_thread_function, &render_list_params);
				RD::get_singleton()->draw_list_end(RD::BARRIER_MASK_ALL);
			} else {
				//single threaded
				RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer, can_continue_color ? RD::INITIAL_ACTION_CONTINUE : RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, can_continue_depth ? RD::INITIAL_ACTION_CONTINUE : RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ);
				_render_list(draw_list, fb_format, &render_list_params, 0, render_list_params.element_count);
				RD::get_singleton()->draw_list_end(RD::BARRIER_MASK_ALL);
			}

			RD::get_singleton()->draw_command_end_label(); // Render Transparent Subpass
		}
	}

	if (render_buffer && !using_subpass_post_process) {
		RD::get_singleton()->draw_command_begin_label("Post process pass");

		// If we need extra effects we do this in its own pass
		RENDER_TIMESTAMP("Tonemap");

		_render_buffers_post_process_and_tonemap(p_render_data);

		RD::get_singleton()->draw_command_end_label(); // Post process pass
	}

	if (render_buffer) {
		_disable_clear_request(p_render_data);
	}
}

/* these are being called from RendererSceneRenderRD::_pre_opaque_render */

void RenderForwardMobile::_render_shadow_begin() {
	scene_state.shadow_passes.clear();
	RD::get_singleton()->draw_command_begin_label("Shadow Setup");
	_update_render_base_uniform_set();

	render_list[RENDER_LIST_SECONDARY].clear();
}

void RenderForwardMobile::_render_shadow_append(RID p_framebuffer, const PagedArray<GeometryInstance *> &p_instances, const CameraMatrix &p_projection, const Transform3D &p_transform, float p_zfar, float p_bias, float p_normal_bias, bool p_use_dp, bool p_use_dp_flip, bool p_use_pancake, const Plane &p_camera_plane, float p_lod_distance_multiplier, float p_screen_lod_threshold, const Rect2i &p_rect, bool p_flip_y, bool p_clear_region, bool p_begin, bool p_end, RendererScene::RenderInfo *p_render_info) {
	uint32_t shadow_pass_index = scene_state.shadow_passes.size();

	SceneState::ShadowPass shadow_pass;

	if (p_render_info) {
		p_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_SHADOW][RS::VIEWPORT_RENDER_INFO_DRAW_CALLS_IN_FRAME] = p_instances.size();
		p_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_SHADOW][RS::VIEWPORT_RENDER_INFO_OBJECTS_IN_FRAME] = p_instances.size();
	}
	RenderDataRD render_data;
	render_data.cam_projection = p_projection;
	render_data.cam_transform = p_transform;
	render_data.view_projection[0] = p_projection;
	render_data.z_near = 0.0;
	render_data.z_far = p_zfar;
	render_data.instances = &p_instances;
	render_data.render_info = p_render_info;
	render_data.lod_camera_plane = p_camera_plane;
	render_data.lod_distance_multiplier = p_lod_distance_multiplier;

	scene_state.ubo.dual_paraboloid_side = p_use_dp_flip ? -1 : 1;

	_setup_environment(&render_data, true, Vector2(1, 1), !p_flip_y, Color(), false, p_use_pancake, shadow_pass_index);

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_DISABLE_LOD) {
		render_data.screen_lod_threshold = 0.0;
	} else {
		render_data.screen_lod_threshold = p_screen_lod_threshold;
	}

	PassMode pass_mode = p_use_dp ? PASS_MODE_SHADOW_DP : PASS_MODE_SHADOW;

	uint32_t render_list_from = render_list[RENDER_LIST_SECONDARY].elements.size();
	_fill_render_list(RENDER_LIST_SECONDARY, &render_data, pass_mode, true);
	uint32_t render_list_size = render_list[RENDER_LIST_SECONDARY].elements.size() - render_list_from;
	render_list[RENDER_LIST_SECONDARY].sort_by_key_range(render_list_from, render_list_size);
	_fill_element_info(RENDER_LIST_SECONDARY, render_list_from, render_list_size);

	{
		//regular forward for now
		bool flip_cull = p_use_dp_flip;
		if (p_flip_y) {
			flip_cull = !flip_cull;
		}

		shadow_pass.element_from = render_list_from;
		shadow_pass.element_count = render_list_size;
		shadow_pass.flip_cull = flip_cull;
		shadow_pass.pass_mode = pass_mode;

		shadow_pass.rp_uniform_set = RID(); //will be filled later when instance buffer is complete
		shadow_pass.camera_plane = p_camera_plane;
		shadow_pass.screen_lod_threshold = render_data.screen_lod_threshold;
		shadow_pass.lod_distance_multiplier = render_data.lod_distance_multiplier;

		shadow_pass.framebuffer = p_framebuffer;
		shadow_pass.initial_depth_action = p_begin ? (p_clear_region ? RD::INITIAL_ACTION_CLEAR_REGION : RD::INITIAL_ACTION_CLEAR) : (p_clear_region ? RD::INITIAL_ACTION_CLEAR_REGION_CONTINUE : RD::INITIAL_ACTION_CONTINUE);
		shadow_pass.final_depth_action = p_end ? RD::FINAL_ACTION_READ : RD::FINAL_ACTION_CONTINUE;
		shadow_pass.rect = p_rect;

		scene_state.shadow_passes.push_back(shadow_pass);
	}
}

void RenderForwardMobile::_render_shadow_process() {
	//render shadows one after the other, so this can be done un-barriered and the driver can optimize (as well as allow us to run compute at the same time)

	for (uint32_t i = 0; i < scene_state.shadow_passes.size(); i++) {
		//render passes need to be configured after instance buffer is done, since they need the latest version
		SceneState::ShadowPass &shadow_pass = scene_state.shadow_passes[i];
		shadow_pass.rp_uniform_set = _setup_render_pass_uniform_set(RENDER_LIST_SECONDARY, nullptr, RID(), false, i);
	}

	RD::get_singleton()->draw_command_end_label();
}

void RenderForwardMobile::_render_shadow_end(uint32_t p_barrier) {
	RD::get_singleton()->draw_command_begin_label("Shadow Render");

	for (uint32_t i = 0; i < scene_state.shadow_passes.size(); i++) {
		SceneState::ShadowPass &shadow_pass = scene_state.shadow_passes[i];
		RenderListParameters render_list_parameters(render_list[RENDER_LIST_SECONDARY].elements.ptr() + shadow_pass.element_from, render_list[RENDER_LIST_SECONDARY].element_info.ptr() + shadow_pass.element_from, shadow_pass.element_count, shadow_pass.flip_cull, shadow_pass.pass_mode, shadow_pass.rp_uniform_set, 0, false, Vector2(), shadow_pass.camera_plane, shadow_pass.lod_distance_multiplier, shadow_pass.screen_lod_threshold, 1, shadow_pass.element_from, RD::BARRIER_MASK_NO_BARRIER);
		_render_list_with_threads(&render_list_parameters, shadow_pass.framebuffer, RD::INITIAL_ACTION_DROP, RD::FINAL_ACTION_DISCARD, shadow_pass.initial_depth_action, shadow_pass.final_depth_action, Vector<Color>(), 1.0, 0, shadow_pass.rect);
	}

	if (p_barrier != RD::BARRIER_MASK_NO_BARRIER) {
		RD::get_singleton()->barrier(RD::BARRIER_MASK_RASTER, p_barrier);
	}
	RD::get_singleton()->draw_command_end_label();
}

/* */

void RenderForwardMobile::_render_material(const Transform3D &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) {
	RENDER_TIMESTAMP("Setup Rendering Material");

	RD::get_singleton()->draw_command_begin_label("Render Material");

	_update_render_base_uniform_set();

	scene_state.ubo.dual_paraboloid_side = 0;
	scene_state.ubo.material_uv2_mode = false;

	RenderDataRD render_data;
	render_data.cam_projection = p_cam_projection;
	render_data.cam_transform = p_cam_transform;
	render_data.view_projection[0] = p_cam_projection;
	render_data.instances = &p_instances;

	_setup_environment(&render_data, true, Vector2(1, 1), false, Color());

	PassMode pass_mode = PASS_MODE_DEPTH_MATERIAL;
	_fill_render_list(RENDER_LIST_SECONDARY, &render_data, pass_mode);
	render_list[RENDER_LIST_SECONDARY].sort_by_key();
	_fill_element_info(RENDER_LIST_SECONDARY);

	RID rp_uniform_set = _setup_render_pass_uniform_set(RENDER_LIST_SECONDARY, nullptr, RID());

	RENDER_TIMESTAMP("Render Material");

	{
		RenderListParameters render_list_params(render_list[RENDER_LIST_SECONDARY].elements.ptr(), render_list[RENDER_LIST_SECONDARY].element_info.ptr(), render_list[RENDER_LIST_SECONDARY].elements.size(), true, pass_mode, rp_uniform_set, 0);
		//regular forward for now
		Vector<Color> clear;
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, clear, 1.0, 0, p_region);
		_render_list(draw_list, RD::get_singleton()->framebuffer_get_format(p_framebuffer), &render_list_params, 0, render_list_params.element_count);
		RD::get_singleton()->draw_list_end();
	}

	RD::get_singleton()->draw_command_end_label();
}

void RenderForwardMobile::_render_uv2(const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) {
	RENDER_TIMESTAMP("Setup Rendering UV2");

	RD::get_singleton()->draw_command_begin_label("Render UV2");

	_update_render_base_uniform_set();

	scene_state.ubo.dual_paraboloid_side = 0;
	scene_state.ubo.material_uv2_mode = true;

	RenderDataRD render_data;
	render_data.instances = &p_instances;

	_setup_environment(&render_data, true, Vector2(1, 1), false, Color());

	PassMode pass_mode = PASS_MODE_DEPTH_MATERIAL;
	_fill_render_list(RENDER_LIST_SECONDARY, &render_data, pass_mode);
	render_list[RENDER_LIST_SECONDARY].sort_by_key();
	_fill_element_info(RENDER_LIST_SECONDARY);

	RID rp_uniform_set = _setup_render_pass_uniform_set(RENDER_LIST_SECONDARY, nullptr, RID());

	RENDER_TIMESTAMP("Render Material");

	{
		RenderListParameters render_list_params(render_list[RENDER_LIST_SECONDARY].elements.ptr(), render_list[RENDER_LIST_SECONDARY].element_info.ptr(), render_list[RENDER_LIST_SECONDARY].elements.size(), true, pass_mode, rp_uniform_set, true, 0);
		//regular forward for now
		Vector<Color> clear;
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, clear, 1.0, 0, p_region);

		const int uv_offset_count = 9;
		static const Vector2 uv_offsets[uv_offset_count] = {
			Vector2(-1, 1),
			Vector2(1, 1),
			Vector2(1, -1),
			Vector2(-1, -1),
			Vector2(-1, 0),
			Vector2(1, 0),
			Vector2(0, -1),
			Vector2(0, 1),
			Vector2(0, 0),

		};

		for (int i = 0; i < uv_offset_count; i++) {
			Vector2 ofs = uv_offsets[i];
			ofs.x /= p_region.size.width;
			ofs.y /= p_region.size.height;
			render_list_params.uv_offset = ofs;
			_render_list(draw_list, RD::get_singleton()->framebuffer_get_format(p_framebuffer), &render_list_params, 0, render_list_params.element_count); //first wireframe, for pseudo conservative
		}
		render_list_params.uv_offset = Vector2();
		_render_list(draw_list, RD::get_singleton()->framebuffer_get_format(p_framebuffer), &render_list_params, 0, render_list_params.element_count); //second regular triangles

		RD::get_singleton()->draw_list_end();
	}

	RD::get_singleton()->draw_command_end_label();
}

void RenderForwardMobile::_render_sdfgi(RID p_render_buffers, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, const PagedArray<GeometryInstance *> &p_instances, const RID &p_albedo_texture, const RID &p_emission_texture, const RID &p_emission_aniso_texture, const RID &p_geom_facing_texture) {
	// we don't do GI in low end..
}

void RenderForwardMobile::_render_particle_collider_heightfield(RID p_fb, const Transform3D &p_cam_transform, const CameraMatrix &p_cam_projection, const PagedArray<GeometryInstance *> &p_instances) {
	RENDER_TIMESTAMP("Setup Render Collider Heightfield");

	RD::get_singleton()->draw_command_begin_label("Render Collider Heightfield");

	_update_render_base_uniform_set();
	scene_state.ubo.dual_paraboloid_side = 0;

	RenderDataRD render_data;
	render_data.cam_projection = p_cam_projection;
	render_data.cam_transform = p_cam_transform;
	render_data.view_projection[0] = p_cam_projection;
	render_data.z_near = 0.0;
	render_data.z_far = p_cam_projection.get_z_far();
	render_data.instances = &p_instances;

	_setup_environment(&render_data, true, Vector2(1, 1), true, Color(), false, false);

	PassMode pass_mode = PASS_MODE_SHADOW;

	_fill_render_list(RENDER_LIST_SECONDARY, &render_data, pass_mode);
	render_list[RENDER_LIST_SECONDARY].sort_by_key();
	_fill_element_info(RENDER_LIST_SECONDARY);

	RID rp_uniform_set = _setup_render_pass_uniform_set(RENDER_LIST_SECONDARY, nullptr, RID());

	RENDER_TIMESTAMP("Render Collider Heightfield");

	{
		//regular forward for now
		RenderListParameters render_list_params(render_list[RENDER_LIST_SECONDARY].elements.ptr(), render_list[RENDER_LIST_SECONDARY].element_info.ptr(), render_list[RENDER_LIST_SECONDARY].elements.size(), false, pass_mode, rp_uniform_set, 0);
		_render_list_with_threads(&render_list_params, p_fb, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ);
	}
	RD::get_singleton()->draw_command_end_label();
}

void RenderForwardMobile::_base_uniforms_changed() {
	if (!render_base_uniform_set.is_null() && RD::get_singleton()->uniform_set_is_valid(render_base_uniform_set)) {
		RD::get_singleton()->free(render_base_uniform_set);
	}
	render_base_uniform_set = RID();
}

void RenderForwardMobile::_update_render_base_uniform_set() {
	if (render_base_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(render_base_uniform_set) || (lightmap_texture_array_version != storage->lightmap_array_get_version())) {
		if (render_base_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(render_base_uniform_set)) {
			RD::get_singleton()->free(render_base_uniform_set);
		}

		// This is all loaded into set 0

		lightmap_texture_array_version = storage->lightmap_array_get_version();

		Vector<RD::Uniform> uniforms;

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 1;
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
			u.binding = 2;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			u.ids.push_back(scene_shader.shadow_sampler);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 3;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			RID sampler;
			switch (decals_get_filter()) {
				case RS::DECAL_FILTER_NEAREST: {
					sampler = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				} break;
				case RS::DECAL_FILTER_NEAREST_MIPMAPS: {
					sampler = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				} break;
				case RS::DECAL_FILTER_LINEAR: {
					sampler = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				} break;
				case RS::DECAL_FILTER_LINEAR_MIPMAPS: {
					sampler = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				} break;
				case RS::DECAL_FILTER_LINEAR_MIPMAPS_ANISOTROPIC: {
					sampler = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				} break;
			}

			u.ids.push_back(sampler);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 4;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			RID sampler;
			switch (light_projectors_get_filter()) {
				case RS::LIGHT_PROJECTOR_FILTER_NEAREST: {
					sampler = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				} break;
				case RS::LIGHT_PROJECTOR_FILTER_NEAREST_MIPMAPS: {
					sampler = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				} break;
				case RS::LIGHT_PROJECTOR_FILTER_LINEAR: {
					sampler = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				} break;
				case RS::LIGHT_PROJECTOR_FILTER_LINEAR_MIPMAPS: {
					sampler = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				} break;
				case RS::LIGHT_PROJECTOR_FILTER_LINEAR_MIPMAPS_ANISOTROPIC: {
					sampler = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				} break;
			}

			u.ids.push_back(sampler);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 5;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(get_omni_light_buffer());
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 6;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(get_spot_light_buffer());
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 7;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(get_reflection_probe_buffer());
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 8;
			u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.ids.push_back(get_directional_light_buffer());
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 9;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(scene_state.lightmap_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 10;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(scene_state.lightmap_capture_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 11;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			RID decal_atlas = storage->decal_atlas_get_texture();
			u.ids.push_back(decal_atlas);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 12;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			RID decal_atlas = storage->decal_atlas_get_texture_srgb();
			u.ids.push_back(decal_atlas);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 13;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(get_decal_buffer());
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 14;
			u.ids.push_back(storage->global_variables_get_storage_buffer());
			uniforms.push_back(u);
		}

		render_base_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, scene_shader.default_shader_rd, SCENE_UNIFORM_SET);
	}
}

RID RenderForwardMobile::_render_buffers_get_normal_texture(RID p_render_buffers) {
	// RenderBufferDataForwardMobile *rb = (RenderBufferDataForwardMobile *)render_buffers_get_data(p_render_buffers);

	// We don't have this. This is for debugging
	// return rb->normal_roughness_buffer;
	return RID();
}

_FORCE_INLINE_ static uint32_t _indices_to_primitives(RS::PrimitiveType p_primitive, uint32_t p_indices) {
	static const uint32_t divisor[RS::PRIMITIVE_MAX] = { 1, 2, 1, 3, 1 };
	static const uint32_t subtractor[RS::PRIMITIVE_MAX] = { 0, 0, 1, 0, 1 };
	return (p_indices - subtractor[p_primitive]) / divisor[p_primitive];
}

void RenderForwardMobile::_fill_render_list(RenderListType p_render_list, const RenderDataRD *p_render_data, PassMode p_pass_mode, bool p_append) {
	if (p_render_list == RENDER_LIST_OPAQUE) {
		scene_state.used_sss = false;
		scene_state.used_screen_texture = false;
		scene_state.used_normal_texture = false;
		scene_state.used_depth_texture = false;
	}
	uint32_t lightmap_captures_used = 0;

	Plane near_plane(-p_render_data->cam_transform.basis.get_axis(Vector3::AXIS_Z), p_render_data->cam_transform.origin);
	near_plane.d += p_render_data->cam_projection.get_z_near();
	float z_max = p_render_data->cam_projection.get_z_far() - p_render_data->cam_projection.get_z_near();

	RenderList *rl = &render_list[p_render_list];

	// Parse any updates on our geometry, updates surface caches and such
	_update_dirty_geometry_instances();

	if (!p_append) {
		rl->clear();
		if (p_render_list == RENDER_LIST_OPAQUE) {
			render_list[RENDER_LIST_ALPHA].clear(); //opaque fills alpha too
		}
	}

	//fill list

	for (int i = 0; i < (int)p_render_data->instances->size(); i++) {
		GeometryInstanceForwardMobile *inst = static_cast<GeometryInstanceForwardMobile *>((*p_render_data->instances)[i]);

		Vector3 support_min = inst->transformed_aabb.get_support(-near_plane.normal);
		inst->depth = near_plane.distance_to(support_min);
		uint32_t depth_layer = CLAMP(int(inst->depth * 16 / z_max), 0, 15);

		uint32_t flags = inst->base_flags; //fill flags if appropriate

		if (inst->non_uniform_scale) {
			flags |= INSTANCE_DATA_FLAGS_NON_UNIFORM_SCALE;
		}

		bool uses_lightmap = false;
		// bool uses_gi = false;

		if (p_render_list == RENDER_LIST_OPAQUE) {
			if (inst->lightmap_instance.is_valid()) {
				int32_t lightmap_cull_index = -1;
				for (uint32_t j = 0; j < scene_state.lightmaps_used; j++) {
					if (scene_state.lightmap_ids[j] == inst->lightmap_instance) {
						lightmap_cull_index = j;
						break;
					}
				}
				if (lightmap_cull_index >= 0) {
					inst->gi_offset_cache = inst->lightmap_slice_index << 16;
					inst->gi_offset_cache |= lightmap_cull_index;
					flags |= INSTANCE_DATA_FLAG_USE_LIGHTMAP;
					if (scene_state.lightmap_has_sh[lightmap_cull_index]) {
						flags |= INSTANCE_DATA_FLAG_USE_SH_LIGHTMAP;
					}
					uses_lightmap = true;
				} else {
					inst->gi_offset_cache = 0xFFFFFFFF;
				}

			} else if (inst->lightmap_sh) {
				if (lightmap_captures_used < scene_state.max_lightmap_captures) {
					const Color *src_capture = inst->lightmap_sh->sh;
					LightmapCaptureData &lcd = scene_state.lightmap_captures[lightmap_captures_used];
					for (int j = 0; j < 9; j++) {
						lcd.sh[j * 4 + 0] = src_capture[j].r;
						lcd.sh[j * 4 + 1] = src_capture[j].g;
						lcd.sh[j * 4 + 2] = src_capture[j].b;
						lcd.sh[j * 4 + 3] = src_capture[j].a;
					}
					flags |= INSTANCE_DATA_FLAG_USE_LIGHTMAP_CAPTURE;
					inst->gi_offset_cache = lightmap_captures_used;
					lightmap_captures_used++;
					uses_lightmap = true;
				}
			}
		}
		inst->flags_cache = flags;

		GeometryInstanceSurfaceDataCache *surf = inst->surface_caches;

		while (surf) {
			surf->sort.uses_lightmap = 0;

			// LOD

			if (p_render_data->screen_lod_threshold > 0.0 && storage->mesh_surface_has_lod(surf->surface)) {
				//lod
				Vector3 lod_support_min = inst->transformed_aabb.get_support(-p_render_data->lod_camera_plane.normal);
				Vector3 lod_support_max = inst->transformed_aabb.get_support(p_render_data->lod_camera_plane.normal);

				float distance_min = p_render_data->lod_camera_plane.distance_to(lod_support_min);
				float distance_max = p_render_data->lod_camera_plane.distance_to(lod_support_max);

				float distance = 0.0;

				if (distance_min * distance_max < 0.0) {
					//crossing plane
					distance = 0.0;
				} else if (distance_min >= 0.0) {
					distance = distance_min;
				} else if (distance_max <= 0.0) {
					distance = -distance_max;
				}

				uint32_t indices;
				surf->lod_index = storage->mesh_surface_get_lod(surf->surface, inst->lod_model_scale * inst->lod_bias, distance * p_render_data->lod_distance_multiplier, p_render_data->screen_lod_threshold, &indices);
				if (p_render_data->render_info) {
					indices = _indices_to_primitives(surf->primitive, indices);
					if (p_render_list == RENDER_LIST_OPAQUE) { //opaque
						p_render_data->render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += indices;
					} else if (p_render_list == RENDER_LIST_SECONDARY) { //shadow
						p_render_data->render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_SHADOW][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += indices;
					}
				}
			} else {
				surf->lod_index = 0;
				if (p_render_data->render_info) {
					uint32_t to_draw = storage->mesh_surface_get_vertices_drawn_count(surf->surface);
					to_draw = _indices_to_primitives(surf->primitive, to_draw);
					to_draw *= inst->instance_count;
					if (p_render_list == RENDER_LIST_OPAQUE) { //opaque
						p_render_data->render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += storage->mesh_surface_get_vertices_drawn_count(surf->surface);
					} else if (p_render_list == RENDER_LIST_SECONDARY) { //shadow
						p_render_data->render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_SHADOW][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += storage->mesh_surface_get_vertices_drawn_count(surf->surface);
					}
				}
			}

			// ADD Element
			if (p_pass_mode == PASS_MODE_COLOR) {
#ifdef DEBUG_ENABLED
				bool force_alpha = unlikely(get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_OVERDRAW);
#else
				bool force_alpha = false;
#endif
				if (!force_alpha && (surf->flags & (GeometryInstanceSurfaceDataCache::FLAG_PASS_DEPTH | GeometryInstanceSurfaceDataCache::FLAG_PASS_OPAQUE))) {
					rl->add_element(surf);
				}
				if (force_alpha || (surf->flags & GeometryInstanceSurfaceDataCache::FLAG_PASS_ALPHA)) {
					render_list[RENDER_LIST_ALPHA].add_element(surf);
				}

				if (uses_lightmap) {
					surf->sort.uses_lightmap = 1; // This needs to become our lightmap index but we'll do that in a separate PR.
				}

				if (surf->flags & GeometryInstanceSurfaceDataCache::FLAG_USES_SUBSURFACE_SCATTERING) {
					scene_state.used_sss = true;
				}
				if (surf->flags & GeometryInstanceSurfaceDataCache::FLAG_USES_SCREEN_TEXTURE) {
					scene_state.used_screen_texture = true;
				}
				if (surf->flags & GeometryInstanceSurfaceDataCache::FLAG_USES_NORMAL_TEXTURE) {
					scene_state.used_normal_texture = true;
				}
				if (surf->flags & GeometryInstanceSurfaceDataCache::FLAG_USES_DEPTH_TEXTURE) {
					scene_state.used_depth_texture = true;
				}

			} else if (p_pass_mode == PASS_MODE_SHADOW || p_pass_mode == PASS_MODE_SHADOW_DP) {
				if (surf->flags & GeometryInstanceSurfaceDataCache::FLAG_PASS_SHADOW) {
					rl->add_element(surf);
				}
			} else {
				if (surf->flags & (GeometryInstanceSurfaceDataCache::FLAG_PASS_DEPTH | GeometryInstanceSurfaceDataCache::FLAG_PASS_OPAQUE)) {
					rl->add_element(surf);
				}
			}

			surf->sort.depth_layer = depth_layer;

			surf = surf->next;
		}
	}
}

void RenderForwardMobile::_setup_environment(const RenderDataRD *p_render_data, bool p_no_fog, const Size2i &p_screen_size, bool p_flip_y, const Color &p_default_bg_color, bool p_opaque_render_buffers, bool p_pancake_shadows, int p_index) {
	//!BAS! need to go through this and find out what we don't need anymore

	// This populates our UBO with main scene data that is pushed into set 1

	//CameraMatrix projection = p_render_data->cam_projection;
	//projection.flip_y(); // Vulkan and modern APIs use Y-Down
	CameraMatrix correction;
	correction.set_depth_correction(p_flip_y);
	CameraMatrix projection = correction * p_render_data->cam_projection;

	//store camera into ubo
	RendererStorageRD::store_camera(projection, scene_state.ubo.projection_matrix);
	RendererStorageRD::store_camera(projection.inverse(), scene_state.ubo.inv_projection_matrix);
	RendererStorageRD::store_transform(p_render_data->cam_transform, scene_state.ubo.camera_matrix);
	RendererStorageRD::store_transform(p_render_data->cam_transform.affine_inverse(), scene_state.ubo.inv_camera_matrix);

	for (uint32_t v = 0; v < p_render_data->view_count; v++) {
		projection = correction * p_render_data->view_projection[v];
		RendererStorageRD::store_camera(projection, scene_state.ubo.projection_matrix_view[v]);
		RendererStorageRD::store_camera(projection.inverse(), scene_state.ubo.inv_projection_matrix_view[v]);
	}

	scene_state.ubo.z_far = p_render_data->z_far;
	scene_state.ubo.z_near = p_render_data->z_near;

	scene_state.ubo.pancake_shadows = p_pancake_shadows;

	RendererStorageRD::store_soft_shadow_kernel(directional_penumbra_shadow_kernel_get(), scene_state.ubo.directional_penumbra_shadow_kernel);
	RendererStorageRD::store_soft_shadow_kernel(directional_soft_shadow_kernel_get(), scene_state.ubo.directional_soft_shadow_kernel);
	RendererStorageRD::store_soft_shadow_kernel(penumbra_shadow_kernel_get(), scene_state.ubo.penumbra_shadow_kernel);
	RendererStorageRD::store_soft_shadow_kernel(soft_shadow_kernel_get(), scene_state.ubo.soft_shadow_kernel);

	Size2 screen_pixel_size = Vector2(1.0, 1.0) / Size2(p_screen_size);
	scene_state.ubo.screen_pixel_size[0] = screen_pixel_size.x;
	scene_state.ubo.screen_pixel_size[1] = screen_pixel_size.y;

	if (p_render_data->shadow_atlas.is_valid()) {
		Vector2 sas = shadow_atlas_get_size(p_render_data->shadow_atlas);
		scene_state.ubo.shadow_atlas_pixel_size[0] = 1.0 / sas.x;
		scene_state.ubo.shadow_atlas_pixel_size[1] = 1.0 / sas.y;
	}
	{
		Vector2 dss = directional_shadow_get_size();
		scene_state.ubo.directional_shadow_pixel_size[0] = 1.0 / dss.x;
		scene_state.ubo.directional_shadow_pixel_size[1] = 1.0 / dss.y;
	}

	//time global variables
	scene_state.ubo.time = time;

	/*
	scene_state.ubo.gi_upscale_for_msaa = false;
	scene_state.ubo.volumetric_fog_enabled = false;
	scene_state.ubo.fog_enabled = false;

	if (p_render_data->render_buffers.is_valid()) {
		RenderBufferDataForwardMobile *render_buffers = (RenderBufferDataForwardMobile *)render_buffers_get_data(p_render_data->render_buffers);
		if (render_buffers->msaa != RS::VIEWPORT_MSAA_DISABLED) {
			scene_state.ubo.gi_upscale_for_msaa = true;
		}

		if (render_buffers_has_volumetric_fog(p_render_data->render_buffers)) {
			scene_state.ubo.volumetric_fog_enabled = true;
			float fog_end = render_buffers_get_volumetric_fog_end(p_render_data->render_buffers);
			if (fog_end > 0.0) {
				scene_state.ubo.volumetric_fog_inv_length = 1.0 / fog_end;
			} else {
				scene_state.ubo.volumetric_fog_inv_length = 1.0;
			}

			float fog_detail_spread = render_buffers_get_volumetric_fog_detail_spread(p_render_data->render_buffers); //reverse lookup
			if (fog_detail_spread > 0.0) {
				scene_state.ubo.volumetric_fog_detail_spread = 1.0 / fog_detail_spread;
			} else {
				scene_state.ubo.volumetric_fog_detail_spread = 1.0;
			}
		}
	}

	*/

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_UNSHADED) {
		scene_state.ubo.use_ambient_light = true;
		scene_state.ubo.ambient_light_color_energy[0] = 1;
		scene_state.ubo.ambient_light_color_energy[1] = 1;
		scene_state.ubo.ambient_light_color_energy[2] = 1;
		scene_state.ubo.ambient_light_color_energy[3] = 1.0;
		scene_state.ubo.use_ambient_cubemap = false;
		scene_state.ubo.use_reflection_cubemap = false;
		scene_state.ubo.ssao_enabled = false;

	} else if (is_environment(p_render_data->environment)) {
		RS::EnvironmentBG env_bg = environment_get_background(p_render_data->environment);
		RS::EnvironmentAmbientSource ambient_src = environment_get_ambient_source(p_render_data->environment);

		float bg_energy = environment_get_bg_energy(p_render_data->environment);
		scene_state.ubo.ambient_light_color_energy[3] = bg_energy;

		scene_state.ubo.ambient_color_sky_mix = environment_get_ambient_sky_contribution(p_render_data->environment);

		//ambient
		if (ambient_src == RS::ENV_AMBIENT_SOURCE_BG && (env_bg == RS::ENV_BG_CLEAR_COLOR || env_bg == RS::ENV_BG_COLOR)) {
			Color color = env_bg == RS::ENV_BG_CLEAR_COLOR ? p_default_bg_color : environment_get_bg_color(p_render_data->environment);
			color = color.to_linear();

			scene_state.ubo.ambient_light_color_energy[0] = color.r * bg_energy;
			scene_state.ubo.ambient_light_color_energy[1] = color.g * bg_energy;
			scene_state.ubo.ambient_light_color_energy[2] = color.b * bg_energy;
			scene_state.ubo.use_ambient_light = true;
			scene_state.ubo.use_ambient_cubemap = false;
		} else {
			float energy = environment_get_ambient_light_energy(p_render_data->environment);
			Color color = environment_get_ambient_light_color(p_render_data->environment);
			color = color.to_linear();
			scene_state.ubo.ambient_light_color_energy[0] = color.r * energy;
			scene_state.ubo.ambient_light_color_energy[1] = color.g * energy;
			scene_state.ubo.ambient_light_color_energy[2] = color.b * energy;

			Basis sky_transform = environment_get_sky_orientation(p_render_data->environment);
			sky_transform = sky_transform.inverse() * p_render_data->cam_transform.basis;
			RendererStorageRD::store_transform_3x3(sky_transform, scene_state.ubo.radiance_inverse_xform);

			scene_state.ubo.use_ambient_cubemap = (ambient_src == RS::ENV_AMBIENT_SOURCE_BG && env_bg == RS::ENV_BG_SKY) || ambient_src == RS::ENV_AMBIENT_SOURCE_SKY;
			scene_state.ubo.use_ambient_light = scene_state.ubo.use_ambient_cubemap || ambient_src == RS::ENV_AMBIENT_SOURCE_COLOR;
		}

		//specular
		RS::EnvironmentReflectionSource ref_src = environment_get_reflection_source(p_render_data->environment);
		if ((ref_src == RS::ENV_REFLECTION_SOURCE_BG && env_bg == RS::ENV_BG_SKY) || ref_src == RS::ENV_REFLECTION_SOURCE_SKY) {
			scene_state.ubo.use_reflection_cubemap = true;
		} else {
			scene_state.ubo.use_reflection_cubemap = false;
		}

		scene_state.ubo.ssao_enabled = p_opaque_render_buffers && environment_is_ssao_enabled(p_render_data->environment);
		scene_state.ubo.ssao_ao_affect = environment_get_ssao_ao_affect(p_render_data->environment);
		scene_state.ubo.ssao_light_affect = environment_get_ssao_light_affect(p_render_data->environment);

		scene_state.ubo.fog_enabled = environment_is_fog_enabled(p_render_data->environment);
		scene_state.ubo.fog_density = environment_get_fog_density(p_render_data->environment);
		scene_state.ubo.fog_height = environment_get_fog_height(p_render_data->environment);
		scene_state.ubo.fog_height_density = environment_get_fog_height_density(p_render_data->environment);
		scene_state.ubo.fog_aerial_perspective = environment_get_fog_aerial_perspective(p_render_data->environment);

		Color fog_color = environment_get_fog_light_color(p_render_data->environment).to_linear();
		float fog_energy = environment_get_fog_light_energy(p_render_data->environment);

		scene_state.ubo.fog_light_color[0] = fog_color.r * fog_energy;
		scene_state.ubo.fog_light_color[1] = fog_color.g * fog_energy;
		scene_state.ubo.fog_light_color[2] = fog_color.b * fog_energy;

		scene_state.ubo.fog_sun_scatter = environment_get_fog_sun_scatter(p_render_data->environment);

	} else {
		if (p_render_data->reflection_probe.is_valid() && storage->reflection_probe_is_interior(reflection_probe_instance_get_probe(p_render_data->reflection_probe))) {
			scene_state.ubo.use_ambient_light = false;
		} else {
			scene_state.ubo.use_ambient_light = true;
			Color clear_color = p_default_bg_color;
			clear_color = clear_color.to_linear();
			scene_state.ubo.ambient_light_color_energy[0] = clear_color.r;
			scene_state.ubo.ambient_light_color_energy[1] = clear_color.g;
			scene_state.ubo.ambient_light_color_energy[2] = clear_color.b;
			scene_state.ubo.ambient_light_color_energy[3] = 1.0;
		}

		scene_state.ubo.use_ambient_cubemap = false;
		scene_state.ubo.use_reflection_cubemap = false;
		scene_state.ubo.ssao_enabled = false;
	}

	scene_state.ubo.roughness_limiter_enabled = p_opaque_render_buffers && screen_space_roughness_limiter_is_active();
	scene_state.ubo.roughness_limiter_amount = screen_space_roughness_limiter_get_amount();
	scene_state.ubo.roughness_limiter_limit = screen_space_roughness_limiter_get_limit();

	if (p_index >= (int)scene_state.uniform_buffers.size()) {
		uint32_t from = scene_state.uniform_buffers.size();
		scene_state.uniform_buffers.resize(p_index + 1);
		render_pass_uniform_sets.resize(p_index + 1);
		for (uint32_t i = from; i < scene_state.uniform_buffers.size(); i++) {
			scene_state.uniform_buffers[i] = RD::get_singleton()->uniform_buffer_create(sizeof(SceneState::UBO));
		}
	}
	RD::get_singleton()->buffer_update(scene_state.uniform_buffers[p_index], 0, sizeof(SceneState::UBO), &scene_state.ubo, RD::BARRIER_MASK_RASTER);
}

void RenderForwardMobile::_fill_element_info(RenderListType p_render_list, uint32_t p_offset, int32_t p_max_elements) {
	RenderList *rl = &render_list[p_render_list];
	uint32_t element_total = p_max_elements >= 0 ? uint32_t(p_max_elements) : rl->elements.size();

	rl->element_info.resize(p_offset + element_total);

	for (uint32_t i = 0; i < element_total; i++) {
		GeometryInstanceSurfaceDataCache *surface = rl->elements[i + p_offset];
		RenderElementInfo &element_info = rl->element_info[p_offset + i];

		element_info.lod_index = surface->lod_index;
		element_info.uses_lightmap = surface->sort.uses_lightmap;
	}
}

/// RENDERING ///

void RenderForwardMobile::_render_list(RenderingDevice::DrawListID p_draw_list, RenderingDevice::FramebufferFormatID p_framebuffer_Format, RenderListParameters *p_params, uint32_t p_from_element, uint32_t p_to_element) {
	//use template for faster performance (pass mode comparisons are inlined)

	switch (p_params->pass_mode) {
		case PASS_MODE_COLOR: {
			_render_list_template<PASS_MODE_COLOR>(p_draw_list, p_framebuffer_Format, p_params, p_from_element, p_to_element);
		} break;
		case PASS_MODE_COLOR_TRANSPARENT: {
			_render_list_template<PASS_MODE_COLOR_TRANSPARENT>(p_draw_list, p_framebuffer_Format, p_params, p_from_element, p_to_element);
		} break;
		case PASS_MODE_SHADOW: {
			_render_list_template<PASS_MODE_SHADOW>(p_draw_list, p_framebuffer_Format, p_params, p_from_element, p_to_element);
		} break;
		case PASS_MODE_SHADOW_DP: {
			_render_list_template<PASS_MODE_SHADOW_DP>(p_draw_list, p_framebuffer_Format, p_params, p_from_element, p_to_element);
		} break;
		case PASS_MODE_DEPTH_MATERIAL: {
			_render_list_template<PASS_MODE_DEPTH_MATERIAL>(p_draw_list, p_framebuffer_Format, p_params, p_from_element, p_to_element);
		} break;
	}
}

void RenderForwardMobile::_render_list_thread_function(uint32_t p_thread, RenderListParameters *p_params) {
	uint32_t render_total = p_params->element_count;
	uint32_t total_threads = RendererThreadPool::singleton->thread_work_pool.get_thread_count();
	uint32_t render_from = p_thread * render_total / total_threads;
	uint32_t render_to = (p_thread + 1 == total_threads) ? render_total : ((p_thread + 1) * render_total / total_threads);
	_render_list(thread_draw_lists[p_thread], p_params->framebuffer_format, p_params, render_from, render_to);
}

void RenderForwardMobile::_render_list_with_threads(RenderListParameters *p_params, RID p_framebuffer, RD::InitialAction p_initial_color_action, RD::FinalAction p_final_color_action, RD::InitialAction p_initial_depth_action, RD::FinalAction p_final_depth_action, const Vector<Color> &p_clear_color_values, float p_clear_depth, uint32_t p_clear_stencil, const Rect2 &p_region, const Vector<RID> &p_storage_textures) {
	RD::FramebufferFormatID fb_format = RD::get_singleton()->framebuffer_get_format(p_framebuffer);
	p_params->framebuffer_format = fb_format;

	if ((uint32_t)p_params->element_count > render_list_thread_threshold && false) { // secondary command buffers need more testing at this time
		//multi threaded
		thread_draw_lists.resize(RendererThreadPool::singleton->thread_work_pool.get_thread_count());
		RD::get_singleton()->draw_list_begin_split(p_framebuffer, thread_draw_lists.size(), thread_draw_lists.ptr(), p_initial_color_action, p_final_color_action, p_initial_depth_action, p_final_depth_action, p_clear_color_values, p_clear_depth, p_clear_stencil, p_region, p_storage_textures);
		RendererThreadPool::singleton->thread_work_pool.do_work(thread_draw_lists.size(), this, &RenderForwardMobile::_render_list_thread_function, p_params);
		RD::get_singleton()->draw_list_end(p_params->barrier);
	} else {
		//single threaded
		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_framebuffer, p_initial_color_action, p_final_color_action, p_initial_depth_action, p_final_depth_action, p_clear_color_values, p_clear_depth, p_clear_stencil, p_region, p_storage_textures);
		_render_list(draw_list, fb_format, p_params, 0, p_params->element_count);
		RD::get_singleton()->draw_list_end(p_params->barrier);
	}
}

void RenderForwardMobile::_fill_push_constant_instance_indices(GeometryInstanceForwardMobile::PushConstant *p_push_constant, uint32_t &spec_constants, const GeometryInstanceForwardMobile *p_instance) {
	// first zero out our indices

	p_push_constant->omni_lights[0] = 0xFFFF;
	p_push_constant->omni_lights[1] = 0xFFFF;

	p_push_constant->spot_lights[0] = 0xFFFF;
	p_push_constant->spot_lights[1] = 0xFFFF;

	p_push_constant->decals[0] = 0xFFFF;
	p_push_constant->decals[1] = 0xFFFF;

	p_push_constant->reflection_probes[0] = 0xFFFF;
	p_push_constant->reflection_probes[1] = 0xFFFF;

	if (p_instance->omni_light_count == 0) {
		spec_constants |= 1 << SPEC_CONSTANT_DISABLE_OMNI_LIGHTS;
	}
	if (p_instance->spot_light_count == 0) {
		spec_constants |= 1 << SPEC_CONSTANT_DISABLE_SPOT_LIGHTS;
	}
	if (p_instance->reflection_probe_count == 0) {
		spec_constants |= 1 << SPEC_CONSTANT_DISABLE_REFLECTION_PROBES;
	}
	if (p_instance->decals_count == 0) {
		spec_constants |= 1 << SPEC_CONSTANT_DISABLE_DECALS;
	}

	for (uint32_t i = 0; i < MAX_RDL_CULL; i++) {
		uint32_t ofs = i < 4 ? 0 : 1;
		uint32_t shift = (i & 0x3) << 3;
		uint32_t mask = ~(0xFF << shift);
		if (i < p_instance->omni_light_count) {
			p_push_constant->omni_lights[ofs] &= mask;
			p_push_constant->omni_lights[ofs] |= uint32_t(forward_id_allocators[FORWARD_ID_TYPE_OMNI_LIGHT].map[p_instance->omni_lights[i]]) << shift;
		}
		if (i < p_instance->spot_light_count) {
			p_push_constant->spot_lights[ofs] &= mask;
			p_push_constant->spot_lights[ofs] |= uint32_t(forward_id_allocators[FORWARD_ID_TYPE_SPOT_LIGHT].map[p_instance->spot_lights[i]]) << shift;
		}
		if (i < p_instance->decals_count) {
			p_push_constant->decals[ofs] &= mask;
			p_push_constant->decals[ofs] |= uint32_t(forward_id_allocators[FORWARD_ID_TYPE_DECAL].map[p_instance->decals[i]]) << shift;
		}
		if (i < p_instance->reflection_probe_count) {
			p_push_constant->reflection_probes[ofs] &= mask;
			p_push_constant->reflection_probes[ofs] |= uint32_t(forward_id_allocators[FORWARD_ID_TYPE_REFLECTION_PROBE].map[p_instance->reflection_probes[i]]) << shift;
		}
	}
}

template <RenderForwardMobile::PassMode p_pass_mode>
void RenderForwardMobile::_render_list_template(RenderingDevice::DrawListID p_draw_list, RenderingDevice::FramebufferFormatID p_framebuffer_Format, RenderListParameters *p_params, uint32_t p_from_element, uint32_t p_to_element) {
	RD::DrawListID draw_list = p_draw_list;
	RD::FramebufferFormatID framebuffer_format = p_framebuffer_Format;

	//global scope bindings
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, render_base_uniform_set, SCENE_UNIFORM_SET);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, p_params->render_pass_uniform_set, RENDER_PASS_UNIFORM_SET);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, scene_shader.default_vec4_xform_uniform_set, TRANSFORMS_UNIFORM_SET);

	RID prev_material_uniform_set;

	RID prev_vertex_array_rd;
	RID prev_index_array_rd;
	RID prev_pipeline_rd;
	RID prev_xforms_uniform_set;

	bool shadow_pass = (p_params->pass_mode == PASS_MODE_SHADOW) || (p_params->pass_mode == PASS_MODE_SHADOW_DP);

	for (uint32_t i = p_from_element; i < p_to_element; i++) {
		const GeometryInstanceSurfaceDataCache *surf = p_params->elements[i];
		const RenderElementInfo &element_info = p_params->element_info[i];
		const GeometryInstanceForwardMobile *inst = surf->owner;

		uint32_t base_spec_constants = p_params->spec_constant_base_flags;

		// GeometryInstanceForwardMobile::PushConstant push_constant = inst->push_constant;
		GeometryInstanceForwardMobile::PushConstant push_constant;

		if (inst->store_transform_cache) {
			RendererStorageRD::store_transform(inst->transform, push_constant.transform);
		} else {
			RendererStorageRD::store_transform(Transform3D(), push_constant.transform);
		}

		push_constant.flags = inst->flags_cache;
		push_constant.gi_offset = inst->gi_offset_cache;
		push_constant.layer_mask = inst->layer_mask;
		push_constant.instance_uniforms_ofs = uint32_t(inst->shader_parameters_offset);

		if (p_params->pass_mode == PASS_MODE_DEPTH_MATERIAL) {
			// abuse lightmap_uv_scale[0] here, should not be needed here
			push_constant.lightmap_uv_scale[0] = p_params->uv_offset.x;
			push_constant.lightmap_uv_scale[1] = p_params->uv_offset.y;
		} else {
			push_constant.lightmap_uv_scale[0] = inst->lightmap_uv_scale.position.x;
			push_constant.lightmap_uv_scale[1] = inst->lightmap_uv_scale.position.y;
			push_constant.lightmap_uv_scale[2] = inst->lightmap_uv_scale.size.x;
			push_constant.lightmap_uv_scale[3] = inst->lightmap_uv_scale.size.y;
		};

		RID material_uniform_set;
		SceneShaderForwardMobile::ShaderData *shader;
		void *mesh_surface;

		if (shadow_pass) {
			material_uniform_set = surf->material_uniform_set_shadow;
			shader = surf->shader_shadow;
			mesh_surface = surf->surface_shadow;

		} else {
			if (inst->use_projector) {
				base_spec_constants |= 1 << SPEC_CONSTANT_USING_PROJECTOR;
			}
			if (inst->use_soft_shadow) {
				base_spec_constants |= 1 << SPEC_CONSTANT_USING_SOFT_SHADOWS;
			}
			_fill_push_constant_instance_indices(&push_constant, base_spec_constants, inst);

#ifdef DEBUG_ENABLED
			if (unlikely(get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_LIGHTING)) {
				material_uniform_set = scene_shader.default_material_uniform_set;
				shader = scene_shader.default_material_shader_ptr;
			} else if (unlikely(get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_OVERDRAW)) {
				material_uniform_set = scene_shader.overdraw_material_uniform_set;
				shader = scene_shader.overdraw_material_shader_ptr;
			} else {
#endif
				material_uniform_set = surf->material_uniform_set;
				shader = surf->shader;
#ifdef DEBUG_ENABLED
			}
#endif
			mesh_surface = surf->surface;
		}

		if (!mesh_surface) {
			continue;
		}

		//find cull variant
		SceneShaderForwardMobile::ShaderData::CullVariant cull_variant;

		if (p_params->pass_mode == PASS_MODE_DEPTH_MATERIAL || ((p_params->pass_mode == PASS_MODE_SHADOW || p_params->pass_mode == PASS_MODE_SHADOW_DP) && surf->flags & GeometryInstanceSurfaceDataCache::FLAG_USES_DOUBLE_SIDED_SHADOWS)) {
			cull_variant = SceneShaderForwardMobile::ShaderData::CULL_VARIANT_DOUBLE_SIDED;
		} else {
			bool mirror = surf->owner->mirror;
			if (p_params->reverse_cull) {
				mirror = !mirror;
			}
			cull_variant = mirror ? SceneShaderForwardMobile::ShaderData::CULL_VARIANT_REVERSED : SceneShaderForwardMobile::ShaderData::CULL_VARIANT_NORMAL;
		}

		RS::PrimitiveType primitive = surf->primitive;
		RID xforms_uniform_set = surf->owner->transforms_uniform_set;

		SceneShaderForwardMobile::ShaderVersion shader_version = SceneShaderForwardMobile::SHADER_VERSION_MAX; // Assigned to silence wrong -Wmaybe-initialized.

		switch (p_params->pass_mode) {
			case PASS_MODE_COLOR:
			case PASS_MODE_COLOR_TRANSPARENT: {
				if (element_info.uses_lightmap) {
					shader_version = p_params->view_count > 1 ? SceneShaderForwardMobile::SHADER_VERSION_LIGHTMAP_COLOR_PASS_MULTIVIEW : SceneShaderForwardMobile::SHADER_VERSION_LIGHTMAP_COLOR_PASS;
				} else {
					shader_version = p_params->view_count > 1 ? SceneShaderForwardMobile::SHADER_VERSION_COLOR_PASS_MULTIVIEW : SceneShaderForwardMobile::SHADER_VERSION_COLOR_PASS;
				}
			} break;
			case PASS_MODE_SHADOW: {
				shader_version = p_params->view_count > 1 ? SceneShaderForwardMobile::SHADER_VERSION_SHADOW_PASS_MULTIVIEW : SceneShaderForwardMobile::SHADER_VERSION_SHADOW_PASS;
			} break;
			case PASS_MODE_SHADOW_DP: {
				ERR_FAIL_COND_MSG(p_params->view_count > 1, "Multiview not supported for shadow DP pass");
				shader_version = SceneShaderForwardMobile::SHADER_VERSION_SHADOW_PASS_DP;
			} break;
			case PASS_MODE_DEPTH_MATERIAL: {
				ERR_FAIL_COND_MSG(p_params->view_count > 1, "Multiview not supported for material pass");
				shader_version = SceneShaderForwardMobile::SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL;
			} break;
		}

		PipelineCacheRD *pipeline = nullptr;

		pipeline = &shader->pipelines[cull_variant][primitive][shader_version];

		RD::VertexFormatID vertex_format = -1;
		RID vertex_array_rd;
		RID index_array_rd;

		//skeleton and blend shape
		if (surf->owner->mesh_instance.is_valid()) {
			storage->mesh_instance_surface_get_vertex_arrays_and_format(surf->owner->mesh_instance, surf->surface_index, pipeline->get_vertex_input_mask(), vertex_array_rd, vertex_format);
		} else {
			storage->mesh_surface_get_vertex_arrays_and_format(mesh_surface, pipeline->get_vertex_input_mask(), vertex_array_rd, vertex_format);
		}

		index_array_rd = storage->mesh_surface_get_index_array(mesh_surface, element_info.lod_index);

		if (prev_vertex_array_rd != vertex_array_rd) {
			RD::get_singleton()->draw_list_bind_vertex_array(draw_list, vertex_array_rd);
			prev_vertex_array_rd = vertex_array_rd;
		}

		if (prev_index_array_rd != index_array_rd) {
			if (index_array_rd.is_valid()) {
				RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array_rd);
			}
			prev_index_array_rd = index_array_rd;
		}

		RID pipeline_rd = pipeline->get_render_pipeline(vertex_format, framebuffer_format, p_params->force_wireframe, p_params->subpass, base_spec_constants);

		if (pipeline_rd != prev_pipeline_rd) {
			// checking with prev shader does not make so much sense, as
			// the pipeline may still be different.
			RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, pipeline_rd);
			prev_pipeline_rd = pipeline_rd;
		}

		if (xforms_uniform_set.is_valid() && prev_xforms_uniform_set != xforms_uniform_set) {
			RD::get_singleton()->draw_list_bind_uniform_set(draw_list, xforms_uniform_set, TRANSFORMS_UNIFORM_SET);
			prev_xforms_uniform_set = xforms_uniform_set;
		}

		if (material_uniform_set != prev_material_uniform_set) {
			// Update uniform set.
			if (material_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(material_uniform_set)) { // Material may not have a uniform set.
				RD::get_singleton()->draw_list_bind_uniform_set(draw_list, material_uniform_set, MATERIAL_UNIFORM_SET);
			}

			prev_material_uniform_set = material_uniform_set;
		}

		RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(GeometryInstanceForwardMobile::PushConstant));

		uint32_t instance_count = surf->owner->instance_count > 1 ? surf->owner->instance_count : 1;
		if (surf->flags & GeometryInstanceSurfaceDataCache::FLAG_USES_PARTICLE_TRAILS) {
			instance_count /= surf->owner->trail_steps;
		}

		RD::get_singleton()->draw_list_draw(draw_list, index_array_rd.is_valid(), instance_count);
	}
}

/* Geometry instance */

RendererSceneRender::GeometryInstance *RenderForwardMobile::geometry_instance_create(RID p_base) {
	RS::InstanceType type = storage->get_base_type(p_base);
	ERR_FAIL_COND_V(!((1 << type) & RS::INSTANCE_GEOMETRY_MASK), nullptr);

	GeometryInstanceForwardMobile *ginstance = geometry_instance_alloc.alloc();
	ginstance->data = memnew(GeometryInstanceForwardMobile::Data);

	ginstance->data->base = p_base;
	ginstance->data->base_type = type;

	_geometry_instance_mark_dirty(ginstance);

	return ginstance;
}

void RenderForwardMobile::geometry_instance_set_skeleton(GeometryInstance *p_geometry_instance, RID p_skeleton) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->data->skeleton = p_skeleton;

	_geometry_instance_mark_dirty(ginstance);
	ginstance->data->dirty_dependencies = true;
}

void RenderForwardMobile::geometry_instance_set_material_override(GeometryInstance *p_geometry_instance, RID p_override) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->data->material_override = p_override;

	_geometry_instance_mark_dirty(ginstance);
	ginstance->data->dirty_dependencies = true;
}

void RenderForwardMobile::geometry_instance_set_surface_materials(GeometryInstance *p_geometry_instance, const Vector<RID> &p_materials) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->data->surface_materials = p_materials;

	_geometry_instance_mark_dirty(ginstance);
	ginstance->data->dirty_dependencies = true;
}

void RenderForwardMobile::geometry_instance_set_mesh_instance(GeometryInstance *p_geometry_instance, RID p_mesh_instance) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->mesh_instance = p_mesh_instance;

	_geometry_instance_mark_dirty(ginstance);
}

void RenderForwardMobile::geometry_instance_set_transform(GeometryInstance *p_geometry_instance, const Transform3D &p_transform, const AABB &p_aabb, const AABB &p_transformed_aabb) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->transform = p_transform;
	ginstance->mirror = p_transform.basis.determinant() < 0;
	ginstance->data->aabb = p_aabb;
	ginstance->transformed_aabb = p_transformed_aabb;

	Vector3 model_scale_vec = p_transform.basis.get_scale_abs();
	// handle non uniform scale here

	float max_scale = MAX(model_scale_vec.x, MAX(model_scale_vec.y, model_scale_vec.z));
	float min_scale = MIN(model_scale_vec.x, MIN(model_scale_vec.y, model_scale_vec.z));
	ginstance->non_uniform_scale = max_scale >= 0.0 && (min_scale / max_scale) < 0.9;

	ginstance->lod_model_scale = max_scale;
}

void RenderForwardMobile::geometry_instance_set_layer_mask(GeometryInstance *p_geometry_instance, uint32_t p_layer_mask) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->layer_mask = p_layer_mask;
}

void RenderForwardMobile::geometry_instance_set_lod_bias(GeometryInstance *p_geometry_instance, float p_lod_bias) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->lod_bias = p_lod_bias;
}

void RenderForwardMobile::geometry_instance_set_fade_range(GeometryInstance *p_geometry_instance, bool p_enable_near, float p_near_begin, float p_near_end, bool p_enable_far, float p_far_begin, float p_far_end) {
}

void RenderForwardMobile::geometry_instance_set_transparency(GeometryInstance *p_geometry_instance, float p_transparency) {
}

void RenderForwardMobile::geometry_instance_set_parent_fade_alpha(GeometryInstance *p_geometry_instance, float p_alpha) {
}

void RenderForwardMobile::geometry_instance_set_use_baked_light(GeometryInstance *p_geometry_instance, bool p_enable) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->data->use_baked_light = p_enable;

	_geometry_instance_mark_dirty(ginstance);
}

void RenderForwardMobile::geometry_instance_set_use_dynamic_gi(GeometryInstance *p_geometry_instance, bool p_enable) {
	// !BAS! do we support this in mobile?
	// GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	// ERR_FAIL_COND(!ginstance);
	// ginstance->data->use_dynamic_gi = p_enable;
	// _geometry_instance_mark_dirty(ginstance);
}

void RenderForwardMobile::geometry_instance_set_use_lightmap(GeometryInstance *p_geometry_instance, RID p_lightmap_instance, const Rect2 &p_lightmap_uv_scale, int p_lightmap_slice_index) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->lightmap_instance = p_lightmap_instance;
	ginstance->lightmap_uv_scale = p_lightmap_uv_scale;
	ginstance->lightmap_slice_index = p_lightmap_slice_index;
	_geometry_instance_mark_dirty(ginstance);
}

void RenderForwardMobile::geometry_instance_set_lightmap_capture(GeometryInstance *p_geometry_instance, const Color *p_sh9) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	if (p_sh9) {
		if (ginstance->lightmap_sh == nullptr) {
			ginstance->lightmap_sh = geometry_instance_lightmap_sh.alloc();
		}

		memcpy(ginstance->lightmap_sh->sh, p_sh9, sizeof(Color) * 9);
	} else {
		if (ginstance->lightmap_sh != nullptr) {
			geometry_instance_lightmap_sh.free(ginstance->lightmap_sh);
			ginstance->lightmap_sh = nullptr;
		}
	}
	_geometry_instance_mark_dirty(ginstance);
}

void RenderForwardMobile::geometry_instance_set_instance_shader_parameters_offset(GeometryInstance *p_geometry_instance, int32_t p_offset) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	ginstance->shader_parameters_offset = p_offset;
	_geometry_instance_mark_dirty(ginstance);
}

void RenderForwardMobile::geometry_instance_set_cast_double_sided_shadows(GeometryInstance *p_geometry_instance, bool p_enable) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);

	ginstance->data->cast_double_sided_shadows = p_enable;
	_geometry_instance_mark_dirty(ginstance);
}

Transform3D RenderForwardMobile::geometry_instance_get_transform(GeometryInstance *p_instance) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_instance);
	ERR_FAIL_COND_V(!ginstance, Transform3D());
	return ginstance->transform;
}

AABB RenderForwardMobile::geometry_instance_get_aabb(GeometryInstance *p_instance) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_instance);
	ERR_FAIL_COND_V(!ginstance, AABB());
	return ginstance->data->aabb;
}

void RenderForwardMobile::geometry_instance_free(GeometryInstance *p_geometry_instance) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);
	if (ginstance->lightmap_sh != nullptr) {
		geometry_instance_lightmap_sh.free(ginstance->lightmap_sh);
	}
	GeometryInstanceSurfaceDataCache *surf = ginstance->surface_caches;
	while (surf) {
		GeometryInstanceSurfaceDataCache *next = surf->next;
		geometry_instance_surface_alloc.free(surf);
		surf = next;
	}
	memdelete(ginstance->data);
	geometry_instance_alloc.free(ginstance);
}

uint32_t RenderForwardMobile::geometry_instance_get_pair_mask() {
	return ((1 << RS::INSTANCE_LIGHT) + (1 << RS::INSTANCE_REFLECTION_PROBE) + (1 << RS::INSTANCE_DECAL));
}

void RenderForwardMobile::geometry_instance_pair_light_instances(GeometryInstance *p_geometry_instance, const RID *p_light_instances, uint32_t p_light_instance_count) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);

	ginstance->omni_light_count = 0;
	ginstance->spot_light_count = 0;

	for (uint32_t i = 0; i < p_light_instance_count; i++) {
		RS::LightType type = light_instance_get_type(p_light_instances[i]);
		switch (type) {
			case RS::LIGHT_OMNI: {
				if (ginstance->omni_light_count < (uint32_t)MAX_RDL_CULL) {
					ginstance->omni_lights[ginstance->omni_light_count] = light_instance_get_forward_id(p_light_instances[i]);
					ginstance->omni_light_count++;
				}
			} break;
			case RS::LIGHT_SPOT: {
				if (ginstance->spot_light_count < (uint32_t)MAX_RDL_CULL) {
					ginstance->spot_lights[ginstance->spot_light_count] = light_instance_get_forward_id(p_light_instances[i]);
					ginstance->spot_light_count++;
				}
			} break;
			default:
				break;
		}
	}
}

void RenderForwardMobile::geometry_instance_pair_reflection_probe_instances(GeometryInstance *p_geometry_instance, const RID *p_reflection_probe_instances, uint32_t p_reflection_probe_instance_count) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);

	ginstance->reflection_probe_count = p_reflection_probe_instance_count < (uint32_t)MAX_RDL_CULL ? p_reflection_probe_instance_count : (uint32_t)MAX_RDL_CULL;
	for (uint32_t i = 0; i < ginstance->reflection_probe_count; i++) {
		ginstance->reflection_probes[i] = reflection_probe_instance_get_forward_id(p_reflection_probe_instances[i]);
	}
}

void RenderForwardMobile::geometry_instance_pair_decal_instances(GeometryInstance *p_geometry_instance, const RID *p_decal_instances, uint32_t p_decal_instance_count) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);

	ginstance->decals_count = p_decal_instance_count < (uint32_t)MAX_RDL_CULL ? p_decal_instance_count : (uint32_t)MAX_RDL_CULL;
	for (uint32_t i = 0; i < ginstance->decals_count; i++) {
		ginstance->decals[i] = decal_instance_get_forward_id(p_decal_instances[i]);
	}
}

void RenderForwardMobile::geometry_instance_pair_voxel_gi_instances(GeometryInstance *p_geometry_instance, const RID *p_voxel_gi_instances, uint32_t p_voxel_gi_instance_count) {
	// We do not have this here!
}

void RenderForwardMobile::geometry_instance_set_softshadow_projector_pairing(GeometryInstance *p_geometry_instance, bool p_softshadow, bool p_projector) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	ERR_FAIL_COND(!ginstance);

	ginstance->use_projector = p_projector;
	ginstance->use_soft_shadow = p_softshadow;
}

void RenderForwardMobile::_geometry_instance_mark_dirty(GeometryInstance *p_geometry_instance) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);
	if (ginstance->dirty_list_element.in_list()) {
		return;
	}

	//clear surface caches
	GeometryInstanceSurfaceDataCache *surf = ginstance->surface_caches;

	while (surf) {
		GeometryInstanceSurfaceDataCache *next = surf->next;
		geometry_instance_surface_alloc.free(surf);
		surf = next;
	}

	ginstance->surface_caches = nullptr;

	geometry_instance_dirty_list.add(&ginstance->dirty_list_element);
}

void RenderForwardMobile::_geometry_instance_add_surface_with_material(GeometryInstanceForwardMobile *ginstance, uint32_t p_surface, SceneShaderForwardMobile::MaterialData *p_material, uint32_t p_material_id, uint32_t p_shader_id, RID p_mesh) {
	bool has_read_screen_alpha = p_material->shader_data->uses_screen_texture || p_material->shader_data->uses_depth_texture || p_material->shader_data->uses_normal_texture;
	bool has_base_alpha = (p_material->shader_data->uses_alpha || has_read_screen_alpha);
	bool has_blend_alpha = p_material->shader_data->uses_blend_alpha;
	bool has_alpha = has_base_alpha || has_blend_alpha;

	uint32_t flags = 0;

	if (p_material->shader_data->uses_sss) {
		flags |= GeometryInstanceSurfaceDataCache::FLAG_USES_SUBSURFACE_SCATTERING;
	}

	if (p_material->shader_data->uses_screen_texture) {
		flags |= GeometryInstanceSurfaceDataCache::FLAG_USES_SCREEN_TEXTURE;
	}

	if (p_material->shader_data->uses_depth_texture) {
		flags |= GeometryInstanceSurfaceDataCache::FLAG_USES_DEPTH_TEXTURE;
	}

	if (p_material->shader_data->uses_normal_texture) {
		flags |= GeometryInstanceSurfaceDataCache::FLAG_USES_NORMAL_TEXTURE;
	}

	if (ginstance->data->cast_double_sided_shadows) {
		flags |= GeometryInstanceSurfaceDataCache::FLAG_USES_DOUBLE_SIDED_SHADOWS;
	}

	if (has_alpha || has_read_screen_alpha || p_material->shader_data->depth_draw == SceneShaderForwardMobile::ShaderData::DEPTH_DRAW_DISABLED || p_material->shader_data->depth_test == SceneShaderForwardMobile::ShaderData::DEPTH_TEST_DISABLED) {
		//material is only meant for alpha pass
		flags |= GeometryInstanceSurfaceDataCache::FLAG_PASS_ALPHA;
		if (p_material->shader_data->uses_depth_pre_pass && !(p_material->shader_data->depth_draw == SceneShaderForwardMobile::ShaderData::DEPTH_DRAW_DISABLED || p_material->shader_data->depth_test == SceneShaderForwardMobile::ShaderData::DEPTH_TEST_DISABLED)) {
			flags |= GeometryInstanceSurfaceDataCache::FLAG_PASS_DEPTH;
			flags |= GeometryInstanceSurfaceDataCache::FLAG_PASS_SHADOW;
		}
	} else {
		flags |= GeometryInstanceSurfaceDataCache::FLAG_PASS_OPAQUE;
		flags |= GeometryInstanceSurfaceDataCache::FLAG_PASS_DEPTH;
		flags |= GeometryInstanceSurfaceDataCache::FLAG_PASS_SHADOW;
	}

	if (p_material->shader_data->uses_particle_trails) {
		flags |= GeometryInstanceSurfaceDataCache::FLAG_USES_PARTICLE_TRAILS;
	}

	SceneShaderForwardMobile::MaterialData *material_shadow = nullptr;
	void *surface_shadow = nullptr;
	if (!p_material->shader_data->uses_particle_trails && !p_material->shader_data->writes_modelview_or_projection && !p_material->shader_data->uses_vertex && !p_material->shader_data->uses_discard && !p_material->shader_data->uses_depth_pre_pass) {
		flags |= GeometryInstanceSurfaceDataCache::FLAG_USES_SHARED_SHADOW_MATERIAL;
		material_shadow = (SceneShaderForwardMobile::MaterialData *)storage->material_get_data(scene_shader.default_material, RendererStorageRD::SHADER_TYPE_3D);

		RID shadow_mesh = storage->mesh_get_shadow_mesh(p_mesh);

		if (shadow_mesh.is_valid()) {
			surface_shadow = storage->mesh_get_surface(shadow_mesh, p_surface);
		}

	} else {
		material_shadow = p_material;
	}

	GeometryInstanceSurfaceDataCache *sdcache = geometry_instance_surface_alloc.alloc();

	sdcache->flags = flags;

	sdcache->shader = p_material->shader_data;
	sdcache->material_uniform_set = p_material->uniform_set;
	sdcache->surface = storage->mesh_get_surface(p_mesh, p_surface);
	sdcache->primitive = storage->mesh_surface_get_primitive(sdcache->surface);
	sdcache->surface_index = p_surface;

	if (ginstance->data->dirty_dependencies) {
		storage->base_update_dependency(p_mesh, &ginstance->data->dependency_tracker);
	}

	//shadow
	sdcache->shader_shadow = material_shadow->shader_data;
	sdcache->material_uniform_set_shadow = material_shadow->uniform_set;

	sdcache->surface_shadow = surface_shadow ? surface_shadow : sdcache->surface;

	sdcache->owner = ginstance;

	sdcache->next = ginstance->surface_caches;
	ginstance->surface_caches = sdcache;

	//sortkey

	sdcache->sort.sort_key1 = 0;
	sdcache->sort.sort_key2 = 0;

	sdcache->sort.surface_index = p_surface;
	sdcache->sort.material_id_low = p_material_id & 0x0000FFFF;
	sdcache->sort.material_id_hi = p_material_id >> 16;
	sdcache->sort.shader_id = p_shader_id;
	sdcache->sort.geometry_id = p_mesh.get_local_index();
	// sdcache->sort.uses_forward_gi = ginstance->can_sdfgi;
	sdcache->sort.priority = p_material->priority;
}

void RenderForwardMobile::_geometry_instance_add_surface(GeometryInstanceForwardMobile *ginstance, uint32_t p_surface, RID p_material, RID p_mesh) {
	RID m_src;

	m_src = ginstance->data->material_override.is_valid() ? ginstance->data->material_override : p_material;

	SceneShaderForwardMobile::MaterialData *material = nullptr;

	if (m_src.is_valid()) {
		material = (SceneShaderForwardMobile::MaterialData *)storage->material_get_data(m_src, RendererStorageRD::SHADER_TYPE_3D);
		if (!material || !material->shader_data->valid) {
			material = nullptr;
		}
	}

	if (material) {
		if (ginstance->data->dirty_dependencies) {
			storage->material_update_dependency(m_src, &ginstance->data->dependency_tracker);
		}
	} else {
		material = (SceneShaderForwardMobile::MaterialData *)storage->material_get_data(scene_shader.default_material, RendererStorageRD::SHADER_TYPE_3D);
		m_src = scene_shader.default_material;
	}

	ERR_FAIL_COND(!material);

	_geometry_instance_add_surface_with_material(ginstance, p_surface, material, m_src.get_local_index(), storage->material_get_shader_id(m_src), p_mesh);

	while (material->next_pass.is_valid()) {
		RID next_pass = material->next_pass;
		material = (SceneShaderForwardMobile::MaterialData *)storage->material_get_data(next_pass, RendererStorageRD::SHADER_TYPE_3D);
		if (!material || !material->shader_data->valid) {
			break;
		}
		if (ginstance->data->dirty_dependencies) {
			storage->material_update_dependency(next_pass, &ginstance->data->dependency_tracker);
		}
		_geometry_instance_add_surface_with_material(ginstance, p_surface, material, next_pass.get_local_index(), storage->material_get_shader_id(next_pass), p_mesh);
	}
}

void RenderForwardMobile::_geometry_instance_update(GeometryInstance *p_geometry_instance) {
	GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_geometry_instance);

	if (ginstance->data->dirty_dependencies) {
		ginstance->data->dependency_tracker.update_begin();
	}

	//add geometry for drawing
	switch (ginstance->data->base_type) {
		case RS::INSTANCE_MESH: {
			const RID *materials = nullptr;
			uint32_t surface_count;
			RID mesh = ginstance->data->base;

			materials = storage->mesh_get_surface_count_and_materials(mesh, surface_count);
			if (materials) {
				//if no materials, no surfaces.
				const RID *inst_materials = ginstance->data->surface_materials.ptr();
				uint32_t surf_mat_count = ginstance->data->surface_materials.size();

				for (uint32_t j = 0; j < surface_count; j++) {
					RID material = (j < surf_mat_count && inst_materials[j].is_valid()) ? inst_materials[j] : materials[j];
					_geometry_instance_add_surface(ginstance, j, material, mesh);
				}
			}

			ginstance->instance_count = 1;

		} break;

		case RS::INSTANCE_MULTIMESH: {
			RID mesh = storage->multimesh_get_mesh(ginstance->data->base);
			if (mesh.is_valid()) {
				const RID *materials = nullptr;
				uint32_t surface_count;

				materials = storage->mesh_get_surface_count_and_materials(mesh, surface_count);
				if (materials) {
					for (uint32_t j = 0; j < surface_count; j++) {
						_geometry_instance_add_surface(ginstance, j, materials[j], mesh);
					}
				}

				ginstance->instance_count = storage->multimesh_get_instances_to_draw(ginstance->data->base);
			}

		} break;
#if 0
		case RS::INSTANCE_IMMEDIATE: {
			RasterizerStorageGLES3::Immediate *immediate = storage->immediate_owner.get_or_null(inst->base);
			ERR_CONTINUE(!immediate);

			_add_geometry(immediate, inst, nullptr, -1, p_depth_pass, p_shadow_pass);

		} break;
#endif
		case RS::INSTANCE_PARTICLES: {
			int draw_passes = storage->particles_get_draw_passes(ginstance->data->base);

			for (int j = 0; j < draw_passes; j++) {
				RID mesh = storage->particles_get_draw_pass_mesh(ginstance->data->base, j);
				if (!mesh.is_valid()) {
					continue;
				}

				const RID *materials = nullptr;
				uint32_t surface_count;

				materials = storage->mesh_get_surface_count_and_materials(mesh, surface_count);
				if (materials) {
					for (uint32_t k = 0; k < surface_count; k++) {
						_geometry_instance_add_surface(ginstance, k, materials[k], mesh);
					}
				}
			}

			ginstance->instance_count = storage->particles_get_amount(ginstance->data->base, ginstance->trail_steps);

		} break;

		default: {
		}
	}

	//Fill push constant

	bool store_transform = true;
	ginstance->base_flags = 0;

	if (ginstance->data->base_type == RS::INSTANCE_MULTIMESH) {
		ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH;
		if (storage->multimesh_get_transform_format(ginstance->data->base) == RS::MULTIMESH_TRANSFORM_2D) {
			ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH_FORMAT_2D;
		}
		if (storage->multimesh_uses_colors(ginstance->data->base)) {
			ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH_HAS_COLOR;
		}
		if (storage->multimesh_uses_custom_data(ginstance->data->base)) {
			ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH_HAS_CUSTOM_DATA;
		}

		ginstance->transforms_uniform_set = storage->multimesh_get_3d_uniform_set(ginstance->data->base, scene_shader.default_shader_rd, TRANSFORMS_UNIFORM_SET);

	} else if (ginstance->data->base_type == RS::INSTANCE_PARTICLES) {
		ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH;
		if (false) { // 2D particles
			ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH_FORMAT_2D;
		}

		ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH_HAS_COLOR;
		ginstance->base_flags |= INSTANCE_DATA_FLAG_MULTIMESH_HAS_CUSTOM_DATA;

		//for particles, stride is the trail size
		ginstance->base_flags |= (ginstance->trail_steps << INSTANCE_DATA_FLAGS_PARTICLE_TRAIL_SHIFT);

		if (!storage->particles_is_using_local_coords(ginstance->data->base)) {
			store_transform = false;
		}
		ginstance->transforms_uniform_set = storage->particles_get_instance_buffer_uniform_set(ginstance->data->base, scene_shader.default_shader_rd, TRANSFORMS_UNIFORM_SET);

	} else if (ginstance->data->base_type == RS::INSTANCE_MESH) {
		if (storage->skeleton_is_valid(ginstance->data->skeleton)) {
			ginstance->transforms_uniform_set = storage->skeleton_get_3d_uniform_set(ginstance->data->skeleton, scene_shader.default_shader_rd, TRANSFORMS_UNIFORM_SET);
			if (ginstance->data->dirty_dependencies) {
				storage->skeleton_update_dependency(ginstance->data->skeleton, &ginstance->data->dependency_tracker);
			}
		}
	}

	ginstance->store_transform_cache = store_transform;

	if (ginstance->data->dirty_dependencies) {
		ginstance->data->dependency_tracker.update_end();
		ginstance->data->dirty_dependencies = false;
	}

	ginstance->dirty_list_element.remove_from_list();
}

void RenderForwardMobile::_update_dirty_geometry_instances() {
	while (geometry_instance_dirty_list.first()) {
		_geometry_instance_update(geometry_instance_dirty_list.first()->self());
	}
}

void RenderForwardMobile::_geometry_instance_dependency_changed(RendererStorage::DependencyChangedNotification p_notification, RendererStorage::DependencyTracker *p_tracker) {
	switch (p_notification) {
		case RendererStorage::DEPENDENCY_CHANGED_MATERIAL:
		case RendererStorage::DEPENDENCY_CHANGED_MESH:
		case RendererStorage::DEPENDENCY_CHANGED_PARTICLES:
		case RendererStorage::DEPENDENCY_CHANGED_MULTIMESH:
		case RendererStorage::DEPENDENCY_CHANGED_SKELETON_DATA: {
			static_cast<RenderForwardMobile *>(singleton)->_geometry_instance_mark_dirty(static_cast<GeometryInstance *>(p_tracker->userdata));
		} break;
		case RendererStorage::DEPENDENCY_CHANGED_MULTIMESH_VISIBLE_INSTANCES: {
			GeometryInstanceForwardMobile *ginstance = static_cast<GeometryInstanceForwardMobile *>(p_tracker->userdata);
			if (ginstance->data->base_type == RS::INSTANCE_MULTIMESH) {
				ginstance->instance_count = static_cast<RenderForwardMobile *>(singleton)->storage->multimesh_get_instances_to_draw(ginstance->data->base);
			}
		} break;
		default: {
			//rest of notifications of no interest
		} break;
	}
}
void RenderForwardMobile::_geometry_instance_dependency_deleted(const RID &p_dependency, RendererStorage::DependencyTracker *p_tracker) {
	static_cast<RenderForwardMobile *>(singleton)->_geometry_instance_mark_dirty(static_cast<GeometryInstance *>(p_tracker->userdata));
}

/* misc */

bool RenderForwardMobile::is_dynamic_gi_supported() const {
	return false;
}

bool RenderForwardMobile::is_clustered_enabled() const {
	return false;
}

bool RenderForwardMobile::is_volumetric_supported() const {
	return false;
}

uint32_t RenderForwardMobile::get_max_elements() const {
	return 256;
}

RenderForwardMobile *RenderForwardMobile::singleton = nullptr;

void RenderForwardMobile::_update_shader_quality_settings() {
	Vector<RD::PipelineSpecializationConstant> spec_constants;

	RD::PipelineSpecializationConstant sc;
	sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT;

	sc.constant_id = SPEC_CONSTANT_SOFT_SHADOW_SAMPLES;
	sc.int_value = soft_shadow_samples_get();

	spec_constants.push_back(sc);

	sc.constant_id = SPEC_CONSTANT_PENUMBRA_SHADOW_SAMPLES;
	sc.int_value = penumbra_shadow_samples_get();

	spec_constants.push_back(sc);

	sc.constant_id = SPEC_CONSTANT_DIRECTIONAL_SOFT_SHADOW_SAMPLES;
	sc.int_value = directional_soft_shadow_samples_get();

	spec_constants.push_back(sc);

	sc.constant_id = SPEC_CONSTANT_DIRECTIONAL_PENUMBRA_SHADOW_SAMPLES;
	sc.int_value = directional_penumbra_shadow_samples_get();

	spec_constants.push_back(sc);

	sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
	sc.constant_id = SPEC_CONSTANT_DECAL_USE_MIPMAPS;
	sc.bool_value = decals_get_filter() == RS::DECAL_FILTER_NEAREST_MIPMAPS || decals_get_filter() == RS::DECAL_FILTER_LINEAR_MIPMAPS || decals_get_filter() == RS::DECAL_FILTER_LINEAR_MIPMAPS_ANISOTROPIC;

	spec_constants.push_back(sc);

	sc.constant_id = SPEC_CONSTANT_PROJECTOR_USE_MIPMAPS;
	sc.bool_value = light_projectors_get_filter() == RS::LIGHT_PROJECTOR_FILTER_NEAREST_MIPMAPS || light_projectors_get_filter() == RS::LIGHT_PROJECTOR_FILTER_LINEAR_MIPMAPS || light_projectors_get_filter() == RS::LIGHT_PROJECTOR_FILTER_LINEAR_MIPMAPS_ANISOTROPIC;

	spec_constants.push_back(sc);

	scene_shader.set_default_specialization_constants(spec_constants);

	_base_uniforms_changed(); //also need this
}

RenderForwardMobile::RenderForwardMobile(RendererStorageRD *p_storage) :
		RendererSceneRenderRD(p_storage) {
	singleton = this;

	sky.set_texture_format(_render_buffers_get_color_format());

	String defines;

	defines += "\n#define MAX_ROUGHNESS_LOD " + itos(get_roughness_layers() - 1) + ".0\n";
	if (is_using_radiance_cubemap_array()) {
		defines += "\n#define USE_RADIANCE_CUBEMAP_ARRAY \n";
	}
	// defines += "\n#define SDFGI_OCT_SIZE " + itos(gi.sdfgi_get_lightprobe_octahedron_size()) + "\n";
	defines += "\n#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS " + itos(MAX_DIRECTIONAL_LIGHTS) + "\n";

	{
		//lightmaps
		scene_state.max_lightmaps = 2;
		defines += "\n#define MAX_LIGHTMAP_TEXTURES " + itos(scene_state.max_lightmaps) + "\n";
		defines += "\n#define MAX_LIGHTMAPS " + itos(scene_state.max_lightmaps) + "\n";

		scene_state.lightmap_buffer = RD::get_singleton()->storage_buffer_create(sizeof(LightmapData) * scene_state.max_lightmaps);
	}
	{
		//captures
		scene_state.max_lightmap_captures = 2048;
		scene_state.lightmap_captures = memnew_arr(LightmapCaptureData, scene_state.max_lightmap_captures);
		scene_state.lightmap_capture_buffer = RD::get_singleton()->storage_buffer_create(sizeof(LightmapCaptureData) * scene_state.max_lightmap_captures);
	}
	{
		defines += "\n#define MATERIAL_UNIFORM_SET " + itos(MATERIAL_UNIFORM_SET) + "\n";
	}

	scene_shader.init(p_storage, defines);

	// !BAS! maybe we need a mobile version of this setting?
	render_list_thread_threshold = GLOBAL_GET("rendering/limits/forward_renderer/threaded_render_minimum_instances");

	_update_shader_quality_settings();
}

RenderForwardMobile::~RenderForwardMobile() {
	directional_shadow_atlas_set_size(0);

	//clear base uniform set if still valid
	for (uint32_t i = 0; i < render_pass_uniform_sets.size(); i++) {
		if (render_pass_uniform_sets[i].is_valid() && RD::get_singleton()->uniform_set_is_valid(render_pass_uniform_sets[i])) {
			RD::get_singleton()->free(render_pass_uniform_sets[i]);
		}
	}

	{
		for (uint32_t i = 0; i < scene_state.uniform_buffers.size(); i++) {
			RD::get_singleton()->free(scene_state.uniform_buffers[i]);
		}
		RD::get_singleton()->free(scene_state.lightmap_buffer);
		RD::get_singleton()->free(scene_state.lightmap_capture_buffer);
		memdelete_arr(scene_state.lightmap_captures);
	}
}
