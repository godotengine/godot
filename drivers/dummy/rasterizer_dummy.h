/*************************************************************************/
/*  rasterizer_dummy.h                                                   */
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

#ifndef RASTERIZER_DUMMY_H
#define RASTERIZER_DUMMY_H

#include "core/math/camera_matrix.h"
#include "core/rid_owner.h"
#include "core/self_list.h"
#include "scene/resources/mesh.h"
#include "servers/rendering/rasterizer.h"
#include "servers/rendering_server.h"

class RasterizerSceneDummy : public RasterizerScene {
public:
	/* SHADOW ATLAS API */

	RID shadow_atlas_create() { return RID(); }
	void shadow_atlas_set_size(RID p_atlas, int p_size) {}
	void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) {}
	bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) { return false; }

	void directional_shadow_atlas_set_size(int p_size) {}
	int get_directional_light_shadow_size(RID p_light_intance) { return 0; }
	void set_directional_shadow_count(int p_count) {}

	/* SKY API */

	RID sky_create() { return RID(); }
	void sky_set_radiance_size(RID p_sky, int p_radiance_size) {}
	void sky_set_mode(RID p_sky, RS::SkyMode p_samples) {}
	void sky_set_texture(RID p_sky, RID p_panorama) {}
	void sky_set_texture(RID p_sky, RID p_cube_map, int p_radiance_size) {}
	void sky_set_material(RID p_sky, RID p_material) {}

	/* ENVIRONMENT API */

	RID environment_create() { return RID(); }

	void environment_set_background(RID p_env, RS::EnvironmentBG p_bg) {}
	void environment_set_sky(RID p_env, RID p_sky) {}
	void environment_set_sky_custom_fov(RID p_env, float p_scale) {}
	void environment_set_sky_orientation(RID p_env, const Basis &p_orientation) {}
	void environment_set_bg_color(RID p_env, const Color &p_color) {}
	void environment_set_bg_energy(RID p_env, float p_energy) {}
	void environment_set_canvas_max_layer(RID p_env, int p_max_layer) {}
	void environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient = RS::ENV_AMBIENT_SOURCE_BG, float p_energy = 1.0, float p_sky_contribution = 0.0, RS::EnvironmentReflectionSource p_reflection_source = RS::ENV_REFLECTION_SOURCE_BG, const Color &p_ao_color = Color()) {}
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	void environment_set_camera_feed_id(RID p_env, int p_camera_feed_id) {}
#endif

	void environment_set_glow(RID p_env, bool p_enable, int p_level_flags, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap) {}

	void environment_set_fog(RID p_env, bool p_enable, float p_begin, float p_end, RID p_gradient_texture) {}

	void environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance, bool p_roughness) {}
	virtual void environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_bias, float p_light_affect, float p_ao_channel_affect, RS::EnvironmentSSAOBlur p_blur, float p_bilateral_sharpness) {}
	virtual void environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size) {}

	void environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) {}

	void environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, RID p_ramp) {}

	void environment_set_fog(RID p_env, bool p_enable, const Color &p_color, const Color &p_sun_color, float p_sun_amount) {}
	void environment_set_fog_depth(RID p_env, bool p_enable, float p_depth_begin, float p_depth_end, float p_depth_curve, bool p_transmit, float p_transmit_curve) {}
	void environment_set_fog_height(RID p_env, bool p_enable, float p_min_height, float p_max_height, float p_height_curve) {}

	bool is_environment(RID p_env) const { return false; }
	RS::EnvironmentBG environment_get_background(RID p_env) const { return RS::ENV_BG_KEEP; }
	int environment_get_canvas_max_layer(RID p_env) const { return 0; }

	virtual RID camera_effects_create() { return RID(); }

	virtual void camera_effects_set_dof_blur_quality(RS::DOFBlurQuality p_quality, bool p_use_jitter) {}
	virtual void camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape p_shape) {}

	virtual void camera_effects_set_dof_blur(RID p_camera_effects, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount) {}
	virtual void camera_effects_set_custom_exposure(RID p_camera_effects, bool p_enable, float p_exposure) {}

	RID light_instance_create(RID p_light) { return RID(); }
	void light_instance_set_transform(RID p_light_instance, const Transform &p_transform) {}
	void light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform &p_transform, float p_far, float p_split, int p_pass, float p_bias_scale = 1.0) {}
	void light_instance_mark_visible(RID p_light_instance) {}

	RID reflection_atlas_create() { return RID(); }
	virtual void reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) {}

	RID reflection_probe_instance_create(RID p_probe) { return RID(); }
	void reflection_probe_instance_set_transform(RID p_instance, const Transform &p_transform) {}
	void reflection_probe_release_atlas_index(RID p_instance) {}
	bool reflection_probe_instance_needs_redraw(RID p_instance) { return false; }
	bool reflection_probe_instance_has_reflection(RID p_instance) { return false; }
	bool reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) { return false; }
	bool reflection_probe_instance_postprocess_step(RID p_instance) { return true; }

	virtual RID gi_probe_instance_create(RID p_gi_probe) { return RID(); }
	void gi_probe_instance_set_light_data(RID p_probe, RID p_base, RID p_data) {}
	void gi_probe_instance_set_transform_to_data(RID p_probe, const Transform &p_xform) {}
	virtual bool gi_probe_needs_update(RID p_probe) const { return false; }
	virtual void gi_probe_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, int p_dynamic_object_count, InstanceBase **p_dynamic_objects) {}

	virtual void render_scene(RID p_render_buffers, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass) {}
	void render_shadow(RID p_light, RID p_shadow_atlas, int p_pass, InstanceBase **p_cull_result, int p_cull_count) {}
	virtual void render_material(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID p_framebuffer, const Rect2i &p_region) {}

	void set_scene_pass(uint64_t p_pass) {}
	virtual void set_time(double p_time, double p_step) {}
	void set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) {}

	virtual RID render_buffers_create() { return RID(); }
	virtual void render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_width, int p_height, RS::ViewportMSAA p_msaa) {}

	virtual void screen_space_roughness_limiter_set_active(bool p_enable, float p_curve) {}
	virtual bool screen_space_roughness_limiter_is_active() const { return false; }

	bool free(RID p_rid) { return true; }
	virtual void update() {}

	RasterizerSceneDummy() {}
	~RasterizerSceneDummy() {}
};

class RasterizerStorageDummy : public RasterizerStorage {
public:
	/* TEXTURE API */
	struct DummyTexture {
		int width;
		int height;
		uint32_t flags;
		Image::Format format;
		Ref<Image> image;
		String path;
	};

	struct DummySurface {
		uint32_t format;
		RS::PrimitiveType primitive;
		Vector<uint8_t> array;
		int vertex_count;
		Vector<uint8_t> index_array;
		int index_count;
		AABB aabb;
		Vector<Vector<uint8_t>> blend_shapes;
		Vector<AABB> bone_aabbs;
	};

	struct DummyMesh {
		Vector<DummySurface> surfaces;
		int blend_shape_count;
		RS::BlendShapeMode blend_shape_mode;
	};

	mutable RID_PtrOwner<DummyTexture> texture_owner;
	mutable RID_PtrOwner<DummyMesh> mesh_owner;

	virtual RID texture_2d_create(const Ref<Image> &p_image) { return RID(); }
	virtual RID texture_2d_layered_create(const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) { return RID(); }
	virtual RID texture_3d_create(const Vector<Ref<Image>> &p_slices) { return RID(); }
	virtual RID texture_proxy_create(RID p_base) { return RID(); }

	virtual void texture_2d_update_immediate(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) {}
	virtual void texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) {}
	virtual void texture_3d_update(RID p_texture, const Ref<Image> &p_image, int p_depth, int p_mipmap) {}
	virtual void texture_proxy_update(RID p_proxy, RID p_base) {}

	virtual RID texture_2d_placeholder_create() { return RID(); }
	virtual RID texture_2d_layered_placeholder_create() { return RID(); }
	virtual RID texture_3d_placeholder_create() { return RID(); }

	virtual Ref<Image> texture_2d_get(RID p_texture) const { return Ref<Image>(); }
	virtual Ref<Image> texture_2d_layer_get(RID p_texture, int p_layer) const { return Ref<Image>(); }
	virtual Ref<Image> texture_3d_slice_get(RID p_texture, int p_depth, int p_mipmap) const { return Ref<Image>(); }

	virtual void texture_replace(RID p_texture, RID p_by_texture) {}
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height) {}
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	virtual void texture_bind(RID p_texture, uint32_t p_texture_no) = 0;
#endif

	virtual void texture_set_path(RID p_texture, const String &p_path) {}
	virtual String texture_get_path(RID p_texture) const { return String(); }

	virtual void texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) {}
	virtual void texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) {}
	virtual void texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) {}

	virtual void texture_debug_usage(List<RS::TextureInfo> *r_info) {}
	virtual void texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) {}
	virtual Size2 texture_size_with_proxy(RID p_proxy) { return Size2(); }

#if 0
	RID texture_create() {

		DummyTexture *texture = memnew(DummyTexture);
		ERR_FAIL_COND_V(!texture, RID());
		return texture_owner.make_rid(texture);
	}

	void texture_allocate(RID p_texture, int p_width, int p_height, int p_depth_3d, Image::Format p_format, RenderingServer::TextureType p_type = RS::TEXTURE_TYPE_2D, uint32_t p_flags = RS::TEXTURE_FLAGS_DEFAULT) {
		DummyTexture *t = texture_owner.getornull(p_texture);
		ERR_FAIL_COND(!t);
		t->width = p_width;
		t->height = p_height;
		t->flags = p_flags;
		t->format = p_format;
		t->image = Ref<Image>(memnew(Image));
		t->image->create(p_width, p_height, false, p_format);
	}
	void texture_set_data(RID p_texture, const Ref<Image> &p_image, int p_level) {
		DummyTexture *t = texture_owner.getornull(p_texture);
		ERR_FAIL_COND(!t);
		t->width = p_image->get_width();
		t->height = p_image->get_height();
		t->format = p_image->get_format();
		t->image->create(t->width, t->height, false, t->format, p_image->get_data());
	}

	void texture_set_data_partial(RID p_texture, const Ref<Image> &p_image, int src_x, int src_y, int src_w, int src_h, int dst_x, int dst_y, int p_dst_mip, int p_level) {
		DummyTexture *t = texture_owner.getornull(p_texture);

		ERR_FAIL_COND(!t);
		ERR_FAIL_COND_MSG(p_image.is_null(), "It's not a reference to a valid Image object.");
		ERR_FAIL_COND(t->format != p_image->get_format());
		ERR_FAIL_COND(src_w <= 0 || src_h <= 0);
		ERR_FAIL_COND(src_x < 0 || src_y < 0 || src_x + src_w > p_image->get_width() || src_y + src_h > p_image->get_height());
		ERR_FAIL_COND(dst_x < 0 || dst_y < 0 || dst_x + src_w > t->width || dst_y + src_h > t->height);

		t->image->blit_rect(p_image, Rect2(src_x, src_y, src_w, src_h), Vector2(dst_x, dst_y));
	}

	Ref<Image> texture_get_data(RID p_texture, int p_level) const {
		DummyTexture *t = texture_owner.getornull(p_texture);
		ERR_FAIL_COND_V(!t, Ref<Image>());
		return t->image;
	}
	void texture_set_flags(RID p_texture, uint32_t p_flags) {
		DummyTexture *t = texture_owner.getornull(p_texture);
		ERR_FAIL_COND(!t);
		t->flags = p_flags;
	}
	uint32_t texture_get_flags(RID p_texture) const {
		DummyTexture *t = texture_owner.getornull(p_texture);
		ERR_FAIL_COND_V(!t, 0);
		return t->flags;
	}
	Image::Format texture_get_format(RID p_texture) const {
		DummyTexture *t = texture_owner.getornull(p_texture);
		ERR_FAIL_COND_V(!t, Image::FORMAT_RGB8);
		return t->format;
	}

	RenderingServer::TextureType texture_get_type(RID p_texture) const { return RS::TEXTURE_TYPE_2D; }
	uint32_t texture_get_texid(RID p_texture) const { return 0; }
	uint32_t texture_get_width(RID p_texture) const { return 0; }
	uint32_t texture_get_height(RID p_texture) const { return 0; }
	uint32_t texture_get_depth(RID p_texture) const { return 0; }
	void texture_set_size_override(RID p_texture, int p_width, int p_height, int p_depth_3d) {}
	void texture_bind(RID p_texture, uint32_t p_texture_no) {}

	void texture_set_path(RID p_texture, const String &p_path) {
		DummyTexture *t = texture_owner.getornull(p_texture);
		ERR_FAIL_COND(!t);
		t->path = p_path;
	}
	String texture_get_path(RID p_texture) const {
		DummyTexture *t = texture_owner.getornull(p_texture);
		ERR_FAIL_COND_V(!t, String());
		return t->path;
	}

	void texture_set_shrink_all_x2_on_set_data(bool p_enable) {}

	void texture_debug_usage(List<RS::TextureInfo> *r_info) {}

	RID texture_create_radiance_cubemap(RID p_source, int p_resolution = -1) const { return RID(); }

	void texture_set_detect_3d_callback(RID p_texture, RenderingServer::TextureDetectCallback p_callback, void *p_userdata) {}
	void texture_set_detect_srgb_callback(RID p_texture, RenderingServer::TextureDetectCallback p_callback, void *p_userdata) {}
	void texture_set_detect_normal_callback(RID p_texture, RenderingServer::TextureDetectCallback p_callback, void *p_userdata) {}

	void textures_keep_original(bool p_enable) {}

	void texture_set_proxy(RID p_proxy, RID p_base) {}
	virtual Size2 texture_size_with_proxy(RID p_texture) const { return Size2(); }
	void texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) {}
#endif

	/* SKY API */

	RID sky_create() { return RID(); }
	void sky_set_texture(RID p_sky, RID p_cube_map, int p_radiance_size) {}

	/* SHADER API */

	RID shader_create() { return RID(); }

	void shader_set_code(RID p_shader, const String &p_code) {}
	String shader_get_code(RID p_shader) const { return ""; }
	void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const {}

	void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture) {}
	RID shader_get_default_texture_param(RID p_shader, const StringName &p_name) const { return RID(); }
	virtual Variant shader_get_param_default(RID p_material, const StringName &p_param) const { return Variant(); }

	/* COMMON MATERIAL API */

	RID material_create() { return RID(); }

	void material_set_render_priority(RID p_material, int priority) {}
	void material_set_shader(RID p_shader_material, RID p_shader) {}

	void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) {}
	Variant material_get_param(RID p_material, const StringName &p_param) const { return Variant(); }

	void material_set_next_pass(RID p_material, RID p_next_material) {}

	bool material_is_animated(RID p_material) { return false; }
	bool material_casts_shadows(RID p_material) { return false; }
	void material_update_dependency(RID p_material, RasterizerScene::InstanceBase *p_instance) {}

	/* MESH API */

	RID mesh_create() {
		DummyMesh *mesh = memnew(DummyMesh);
		ERR_FAIL_COND_V(!mesh, RID());
		mesh->blend_shape_count = 0;
		mesh->blend_shape_mode = RS::BLEND_SHAPE_MODE_NORMALIZED;
		return mesh_owner.make_rid(mesh);
	}

	void mesh_add_surface(RID p_mesh, const RS::SurfaceData &p_surface) {}

#if 0
	void mesh_add_surface(RID p_mesh, uint32_t p_format, RS::PrimitiveType p_primitive, const Vector<uint8_t> &p_array, int p_vertex_count, const Vector<uint8_t> &p_index_array, int p_index_count, const AABB &p_aabb, const Vector<Vector<uint8_t> > &p_blend_shapes = Vector<Vector<uint8_t> >(), const Vector<AABB> &p_bone_aabbs = Vector<AABB>()) {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND(!m);

		m->surfaces.push_back(DummySurface());
		DummySurface *s = &m->surfaces.write[m->surfaces.size() - 1];
		s->format = p_format;
		s->primitive = p_primitive;
		s->array = p_array;
		s->vertex_count = p_vertex_count;
		s->index_array = p_index_array;
		s->index_count = p_index_count;
		s->aabb = p_aabb;
		s->blend_shapes = p_blend_shapes;
		s->bone_aabbs = p_bone_aabbs;
	}

	void mesh_set_blend_shape_count(RID p_mesh, int p_amount) {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND(!m);
		m->blend_shape_count = p_amount;
	}
#endif

	int mesh_get_blend_shape_count(RID p_mesh) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, 0);
		return m->blend_shape_count;
	}

	void mesh_set_blend_shape_mode(RID p_mesh, RS::BlendShapeMode p_mode) {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND(!m);
		m->blend_shape_mode = p_mode;
	}
	RS::BlendShapeMode mesh_get_blend_shape_mode(RID p_mesh) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, RS::BLEND_SHAPE_MODE_NORMALIZED);
		return m->blend_shape_mode;
	}

	void mesh_surface_update_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {}

	void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) {}
	RID mesh_surface_get_material(RID p_mesh, int p_surface) const { return RID(); }

#if 0
	int mesh_surface_get_array_len(RID p_mesh, int p_surface) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, 0);

		return m->surfaces[p_surface].vertex_count;
	}
	int mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, 0);

		return m->surfaces[p_surface].index_count;
	}

	Vector<uint8_t> mesh_surface_get_array(RID p_mesh, int p_surface) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, Vector<uint8_t>());

		return m->surfaces[p_surface].array;
	}
	Vector<uint8_t> mesh_surface_get_index_array(RID p_mesh, int p_surface) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, Vector<uint8_t>());

		return m->surfaces[p_surface].index_array;
	}

	uint32_t mesh_surface_get_format(RID p_mesh, int p_surface) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, 0);

		return m->surfaces[p_surface].format;
	}
	RS::PrimitiveType mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, RS::PRIMITIVE_POINTS);

		return m->surfaces[p_surface].primitive;
	}

	AABB mesh_surface_get_aabb(RID p_mesh, int p_surface) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, AABB());

		return m->surfaces[p_surface].aabb;
	}
	Vector<Vector<uint8_t> > mesh_surface_get_blend_shapes(RID p_mesh, int p_surface) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, Vector<Vector<uint8_t> >());

		return m->surfaces[p_surface].blend_shapes;
	}
	Vector<AABB> mesh_surface_get_skeleton_aabb(RID p_mesh, int p_surface) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, Vector<AABB>());

		return m->surfaces[p_surface].bone_aabbs;
	}

	void mesh_remove_surface(RID p_mesh, int p_index) {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND(!m);
		ERR_FAIL_COND(p_index >= m->surfaces.size());

		m->surfaces.remove(p_index);
	}
#endif

	RS::SurfaceData mesh_get_surface(RID p_mesh, int p_surface) const { return RS::SurfaceData(); }
	int mesh_get_surface_count(RID p_mesh) const {
		DummyMesh *m = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!m, 0);
		return m->surfaces.size();
	}

	void mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) {}
	AABB mesh_get_custom_aabb(RID p_mesh) const { return AABB(); }

	AABB mesh_get_aabb(RID p_mesh, RID p_skeleton = RID()) { return AABB(); }
	void mesh_clear(RID p_mesh) {}

	/* MULTIMESH API */

	virtual RID multimesh_create() { return RID(); }

	virtual void multimesh_allocate(RID p_multimesh, int p_instances, RS::MultimeshTransformFormat p_transform_format, bool p_use_colors = false, bool p_use_custom_data = false) {}
	int multimesh_get_instance_count(RID p_multimesh) const { return 0; }

	void multimesh_set_mesh(RID p_multimesh, RID p_mesh) {}
	void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform) {}
	void multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) {}
	void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) {}
	void multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) {}

	RID multimesh_get_mesh(RID p_multimesh) const { return RID(); }
	AABB multimesh_get_aabb(RID p_multimesh) const { return AABB(); }

	Transform multimesh_instance_get_transform(RID p_multimesh, int p_index) const { return Transform(); }
	Transform2D multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const { return Transform2D(); }
	Color multimesh_instance_get_color(RID p_multimesh, int p_index) const { return Color(); }
	Color multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const { return Color(); }
	virtual void multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer) {}
	virtual Vector<float> multimesh_get_buffer(RID p_multimesh) const { return Vector<float>(); }

	void multimesh_set_visible_instances(RID p_multimesh, int p_visible) {}
	int multimesh_get_visible_instances(RID p_multimesh) const { return 0; }

	/* IMMEDIATE API */

	RID immediate_create() { return RID(); }
	void immediate_begin(RID p_immediate, RS::PrimitiveType p_rimitive, RID p_texture = RID()) {}
	void immediate_vertex(RID p_immediate, const Vector3 &p_vertex) {}
	void immediate_normal(RID p_immediate, const Vector3 &p_normal) {}
	void immediate_tangent(RID p_immediate, const Plane &p_tangent) {}
	void immediate_color(RID p_immediate, const Color &p_color) {}
	void immediate_uv(RID p_immediate, const Vector2 &tex_uv) {}
	void immediate_uv2(RID p_immediate, const Vector2 &tex_uv) {}
	void immediate_end(RID p_immediate) {}
	void immediate_clear(RID p_immediate) {}
	void immediate_set_material(RID p_immediate, RID p_material) {}
	RID immediate_get_material(RID p_immediate) const { return RID(); }
	AABB immediate_get_aabb(RID p_immediate) const { return AABB(); }

	/* SKELETON API */

	RID skeleton_create() { return RID(); }
	void skeleton_allocate(RID p_skeleton, int p_bones, bool p_2d_skeleton = false) {}
	void skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) {}
	void skeleton_set_world_transform(RID p_skeleton, bool p_enable, const Transform &p_world_transform) {}
	int skeleton_get_bone_count(RID p_skeleton) const { return 0; }
	void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform) {}
	Transform skeleton_bone_get_transform(RID p_skeleton, int p_bone) const { return Transform(); }
	void skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) {}
	Transform2D skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const { return Transform2D(); }

	/* Light API */

	RID light_create(RS::LightType p_type) { return RID(); }

	RID directional_light_create() { return light_create(RS::LIGHT_DIRECTIONAL); }
	RID omni_light_create() { return light_create(RS::LIGHT_OMNI); }
	RID spot_light_create() { return light_create(RS::LIGHT_SPOT); }

	void light_set_color(RID p_light, const Color &p_color) {}
	void light_set_param(RID p_light, RS::LightParam p_param, float p_value) {}
	void light_set_shadow(RID p_light, bool p_enabled) {}
	void light_set_shadow_color(RID p_light, const Color &p_color) {}
	void light_set_projector(RID p_light, RID p_texture) {}
	void light_set_negative(RID p_light, bool p_enable) {}
	void light_set_cull_mask(RID p_light, uint32_t p_mask) {}
	void light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) {}
	void light_set_use_gi(RID p_light, bool p_enabled) {}

	void light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) {}

	void light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) {}
	void light_directional_set_blend_splits(RID p_light, bool p_enable) {}
	bool light_directional_get_blend_splits(RID p_light) const { return false; }
	void light_directional_set_shadow_depth_range_mode(RID p_light, RS::LightDirectionalShadowDepthRangeMode p_range_mode) {}
	RS::LightDirectionalShadowDepthRangeMode light_directional_get_shadow_depth_range_mode(RID p_light) const { return RS::LIGHT_DIRECTIONAL_SHADOW_DEPTH_RANGE_STABLE; }

	RS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) { return RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL; }
	RS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) { return RS::LIGHT_OMNI_SHADOW_DUAL_PARABOLOID; }

	bool light_has_shadow(RID p_light) const { return false; }

	RS::LightType light_get_type(RID p_light) const { return RS::LIGHT_OMNI; }
	AABB light_get_aabb(RID p_light) const { return AABB(); }
	float light_get_param(RID p_light, RS::LightParam p_param) { return 0.0; }
	Color light_get_color(RID p_light) { return Color(); }
	bool light_get_use_gi(RID p_light) { return false; }
	uint64_t light_get_version(RID p_light) const { return 0; }

	/* PROBE API */

	RID reflection_probe_create() { return RID(); }

	void reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) {}
	void reflection_probe_set_intensity(RID p_probe, float p_intensity) {}
	void reflection_probe_set_interior_ambient(RID p_probe, const Color &p_ambient) {}
	void reflection_probe_set_interior_ambient_energy(RID p_probe, float p_energy) {}
	void reflection_probe_set_interior_ambient_probe_contribution(RID p_probe, float p_contrib) {}
	void reflection_probe_set_max_distance(RID p_probe, float p_distance) {}
	void reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) {}
	void reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) {}
	void reflection_probe_set_as_interior(RID p_probe, bool p_enable) {}
	void reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) {}
	void reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) {}
	void reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) {}
	void reflection_probe_set_resolution(RID p_probe, int p_resolution) {}

	AABB reflection_probe_get_aabb(RID p_probe) const { return AABB(); }
	RS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const { return RenderingServer::REFLECTION_PROBE_UPDATE_ONCE; }
	uint32_t reflection_probe_get_cull_mask(RID p_probe) const { return 0; }
	Vector3 reflection_probe_get_extents(RID p_probe) const { return Vector3(); }
	Vector3 reflection_probe_get_origin_offset(RID p_probe) const { return Vector3(); }
	float reflection_probe_get_origin_max_distance(RID p_probe) const { return 0.0; }
	bool reflection_probe_renders_shadows(RID p_probe) const { return false; }

	virtual void base_update_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance) {}
	virtual void skeleton_update_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance) {}

	/* GI PROBE API */

	RID gi_probe_create() { return RID(); }

	virtual void gi_probe_allocate(RID p_gi_probe, const Transform &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) {}

	virtual AABB gi_probe_get_bounds(RID p_gi_probe) const { return AABB(); }
	virtual Vector3i gi_probe_get_octree_size(RID p_gi_probe) const { return Vector3i(); }
	virtual Vector<uint8_t> gi_probe_get_octree_cells(RID p_gi_probe) const { return Vector<uint8_t>(); }
	virtual Vector<uint8_t> gi_probe_get_data_cells(RID p_gi_probe) const { return Vector<uint8_t>(); }
	virtual Vector<uint8_t> gi_probe_get_distance_field(RID p_gi_probe) const { return Vector<uint8_t>(); }

	virtual Vector<int> gi_probe_get_level_counts(RID p_gi_probe) const { return Vector<int>(); }
	virtual Transform gi_probe_get_to_cell_xform(RID p_gi_probe) const { return Transform(); }

	virtual void gi_probe_set_dynamic_range(RID p_gi_probe, float p_range) {}
	virtual float gi_probe_get_dynamic_range(RID p_gi_probe) const { return 0; }

	virtual void gi_probe_set_propagation(RID p_gi_probe, float p_range) {}
	virtual float gi_probe_get_propagation(RID p_gi_probe) const { return 0; }

	void gi_probe_set_energy(RID p_gi_probe, float p_range) {}
	float gi_probe_get_energy(RID p_gi_probe) const { return 0.0; }

	virtual void gi_probe_set_ao(RID p_gi_probe, float p_ao) {}
	virtual float gi_probe_get_ao(RID p_gi_probe) const { return 0; }

	virtual void gi_probe_set_ao_size(RID p_gi_probe, float p_strength) {}
	virtual float gi_probe_get_ao_size(RID p_gi_probe) const { return 0; }

	void gi_probe_set_bias(RID p_gi_probe, float p_range) {}
	float gi_probe_get_bias(RID p_gi_probe) const { return 0.0; }

	void gi_probe_set_normal_bias(RID p_gi_probe, float p_range) {}
	float gi_probe_get_normal_bias(RID p_gi_probe) const { return 0.0; }

	void gi_probe_set_interior(RID p_gi_probe, bool p_enable) {}
	bool gi_probe_is_interior(RID p_gi_probe) const { return false; }

	virtual void gi_probe_set_use_two_bounces(RID p_gi_probe, bool p_enable) {}
	virtual bool gi_probe_is_using_two_bounces(RID p_gi_probe) const { return false; }

	virtual void gi_probe_set_anisotropy_strength(RID p_gi_probe, float p_strength) {}
	virtual float gi_probe_get_anisotropy_strength(RID p_gi_probe) const { return 0; }

	uint32_t gi_probe_get_version(RID p_gi_probe) { return 0; }

	/* LIGHTMAP CAPTURE */
	struct Instantiable {

		SelfList<RasterizerScene::InstanceBase>::List instance_list;

		_FORCE_INLINE_ void instance_change_notify(bool p_aabb = true, bool p_materials = true) {

			SelfList<RasterizerScene::InstanceBase> *instances = instance_list.first();
			while (instances) {

				//instances->self()->base_changed(p_aabb, p_materials);
				instances = instances->next();
			}
		}

		_FORCE_INLINE_ void instance_remove_deps() {
			SelfList<RasterizerScene::InstanceBase> *instances = instance_list.first();
			while (instances) {

				SelfList<RasterizerScene::InstanceBase> *next = instances->next();
				//instances->self()->base_removed();
				instances = next;
			}
		}

		Instantiable() {}
		virtual ~Instantiable() {
		}
	};

	struct LightmapCapture : public Instantiable {

		Vector<LightmapCaptureOctree> octree;
		AABB bounds;
		Transform cell_xform;
		int cell_subdiv;
		float energy;
		LightmapCapture() {
			energy = 1.0;
			cell_subdiv = 1;
		}
	};

	mutable RID_PtrOwner<LightmapCapture> lightmap_capture_data_owner;
	void lightmap_capture_set_bounds(RID p_capture, const AABB &p_bounds) {}
	AABB lightmap_capture_get_bounds(RID p_capture) const { return AABB(); }
	void lightmap_capture_set_octree(RID p_capture, const Vector<uint8_t> &p_octree) {}
	RID lightmap_capture_create() {
		LightmapCapture *capture = memnew(LightmapCapture);
		return lightmap_capture_data_owner.make_rid(capture);
	}
	Vector<uint8_t> lightmap_capture_get_octree(RID p_capture) const {
		const LightmapCapture *capture = lightmap_capture_data_owner.getornull(p_capture);
		ERR_FAIL_COND_V(!capture, Vector<uint8_t>());
		return Vector<uint8_t>();
	}
	void lightmap_capture_set_octree_cell_transform(RID p_capture, const Transform &p_xform) {}
	Transform lightmap_capture_get_octree_cell_transform(RID p_capture) const { return Transform(); }
	void lightmap_capture_set_octree_cell_subdiv(RID p_capture, int p_subdiv) {}
	int lightmap_capture_get_octree_cell_subdiv(RID p_capture) const { return 0; }
	void lightmap_capture_set_energy(RID p_capture, float p_energy) {}
	float lightmap_capture_get_energy(RID p_capture) const { return 0.0; }
	const Vector<LightmapCaptureOctree> *lightmap_capture_get_octree_ptr(RID p_capture) const {
		const LightmapCapture *capture = lightmap_capture_data_owner.getornull(p_capture);
		ERR_FAIL_COND_V(!capture, nullptr);
		return &capture->octree;
	}

	/* PARTICLES */

	RID particles_create() { return RID(); }

	void particles_set_emitting(RID p_particles, bool p_emitting) {}
	void particles_set_amount(RID p_particles, int p_amount) {}
	void particles_set_lifetime(RID p_particles, float p_lifetime) {}
	void particles_set_one_shot(RID p_particles, bool p_one_shot) {}
	void particles_set_pre_process_time(RID p_particles, float p_time) {}
	void particles_set_explosiveness_ratio(RID p_particles, float p_ratio) {}
	void particles_set_randomness_ratio(RID p_particles, float p_ratio) {}
	void particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) {}
	void particles_set_speed_scale(RID p_particles, float p_scale) {}
	void particles_set_use_local_coordinates(RID p_particles, bool p_enable) {}
	void particles_set_process_material(RID p_particles, RID p_material) {}
	void particles_set_fixed_fps(RID p_particles, int p_fps) {}
	void particles_set_fractional_delta(RID p_particles, bool p_enable) {}
	void particles_restart(RID p_particles) {}

	void particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order) {}

	void particles_set_draw_passes(RID p_particles, int p_count) {}
	void particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) {}

	void particles_request_process(RID p_particles) {}
	AABB particles_get_current_aabb(RID p_particles) { return AABB(); }
	AABB particles_get_aabb(RID p_particles) const { return AABB(); }

	void particles_set_emission_transform(RID p_particles, const Transform &p_transform) {}

	bool particles_get_emitting(RID p_particles) { return false; }
	int particles_get_draw_passes(RID p_particles) const { return 0; }
	RID particles_get_draw_pass_mesh(RID p_particles, int p_pass) const { return RID(); }

	virtual bool particles_is_inactive(RID p_particles) const { return false; }

	/* RENDER TARGET */

	RID render_target_create() { return RID(); }
	void render_target_set_position(RID p_render_target, int p_x, int p_y) {}
	void render_target_set_size(RID p_render_target, int p_width, int p_height) {}
	RID render_target_get_texture(RID p_render_target) { return RID(); }
	void render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) {}
	void render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) {}
	bool render_target_was_used(RID p_render_target) { return false; }
	void render_target_set_as_unused(RID p_render_target) {}

	virtual void render_target_request_clear(RID p_render_target, const Color &p_clear_color) {}
	virtual bool render_target_is_clear_requested(RID p_render_target) { return false; }
	virtual Color render_target_get_clear_request_color(RID p_render_target) { return Color(); }
	virtual void render_target_disable_clear_request(RID p_render_target) {}
	virtual void render_target_do_clear_request(RID p_render_target) {}

	RS::InstanceType get_base_type(RID p_rid) const {
		if (mesh_owner.owns(p_rid)) {
			return RS::INSTANCE_MESH;
		}

		return RS::INSTANCE_NONE;
	}

	bool free(RID p_rid) {

		if (texture_owner.owns(p_rid)) {
			// delete the texture
			DummyTexture *texture = texture_owner.getornull(p_rid);
			texture_owner.free(p_rid);
			memdelete(texture);
		}
		return true;
	}

	bool has_os_feature(const String &p_feature) const { return false; }

	void update_dirty_resources() {}

	void set_debug_generate_wireframes(bool p_generate) {}

	void render_info_begin_capture() {}
	void render_info_end_capture() {}
	int get_captured_render_info(RS::RenderInfo p_info) { return 0; }

	int get_render_info(RS::RenderInfo p_info) { return 0; }
	String get_video_adapter_name() const { return String(); }
	String get_video_adapter_vendor() const { return String(); }

	static RasterizerStorage *base_singleton;

	virtual void capture_timestamps_begin() {}
	virtual void capture_timestamp(const String &p_name) {}
	virtual uint32_t get_captured_timestamps_count() const { return 0; }
	virtual uint64_t get_captured_timestamps_frame() const { return 0; }
	virtual uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const { return 0; }
	virtual uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const { return 0; }
	virtual String get_captured_timestamp_name(uint32_t p_index) const { return String(); }

	RasterizerStorageDummy() {}
	~RasterizerStorageDummy() {}
};

class RasterizerCanvasDummy : public RasterizerCanvas {
public:
	virtual TextureBindingID request_texture_binding(RID p_texture, RID p_normalmap, RID p_specular, RS::CanvasItemTextureFilter p_filter, RS::CanvasItemTextureRepeat p_repeat, RID p_multimesh) { return 0; }
	virtual void free_texture_binding(TextureBindingID p_binding) {}

	virtual PolygonID request_polygon(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs = Vector<Point2>(), const Vector<int> &p_bones = Vector<int>(), const Vector<float> &p_weights = Vector<float>()) { return 0; }
	virtual void free_polygon(PolygonID p_polygon) {}

	virtual void canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, const Transform2D &p_canvas_transform) {}
	virtual void canvas_debug_viewport_shadows(Light *p_lights_with_shadow) {}

	virtual RID light_create() { return RID(); }
	virtual void light_set_texture(RID p_rid, RID p_texture) {}
	virtual void light_set_use_shadow(RID p_rid, bool p_enable, int p_resolution) {}
	virtual void light_update_shadow(RID p_rid, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders) {}

	virtual RID occluder_polygon_create() { return RID(); }
	virtual void occluder_polygon_set_shape_as_lines(RID p_occluder, const Vector<Vector2> &p_lines) {}
	virtual void occluder_polygon_set_cull_mode(RID p_occluder, RS::CanvasOccluderPolygonCullMode p_mode) {}

	void draw_window_margins(int *p_margins, RID *p_margin_textures) {}

	virtual bool free(RID p_rid) { return true; }
	virtual void update() {}

	RasterizerCanvasDummy() {}
	~RasterizerCanvasDummy() {}
};

class RasterizerDummy : public Rasterizer {
protected:
	RasterizerCanvasDummy canvas;
	RasterizerStorageDummy storage;
	RasterizerSceneDummy scene;

public:
	RasterizerStorage *get_storage() { return &storage; }
	RasterizerCanvas *get_canvas() { return &canvas; }
	RasterizerScene *get_scene() { return &scene; }

	void set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter = true) {}

	void initialize() {}
	void begin_frame(double frame_step) {}

	virtual void prepare_for_blitting_render_targets() {}
	virtual void blit_render_targets_to_screen(int p_screen, const BlitToScreen *p_render_targets, int p_amount) {}

	void end_frame(bool p_swap_buffers) { OS::get_singleton()->swap_buffers(); }
	void finalize() {}

	static Error is_viable() {
		return OK;
	}

	static Rasterizer *_create_current() {
		return memnew(RasterizerDummy);
	}

	static void make_current() {
		_create_func = _create_current;
	}

	virtual bool is_low_end() const { return true; }

	RasterizerDummy() {}
	~RasterizerDummy() {}
};

#endif // RASTERIZER_DUMMY_H
