/*************************************************************************/
/*  rasterizer_storage_gles2.h                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef RASTERIZERSTORAGEGLES2_H
#define RASTERIZERSTORAGEGLES2_H

#include "self_list.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual/shader_language.h"
#include "shader_compiler_gles2.h"
#include "shader_gles2.h"

#include "shaders/copy.glsl.gen.h"
/*
#include "shaders/blend_shape.glsl.gen.h"
#include "shaders/canvas.glsl.gen.h"
#include "shaders/copy.glsl.gen.h"
#include "shaders/cubemap_filter.glsl.gen.h"
#include "shaders/particles.glsl.gen.h"
*/

class RasterizerCanvasGLES2;
class RasterizerSceneGLES2;

class RasterizerStorageGLES2 : public RasterizerStorage {
public:
	RasterizerCanvasGLES2 *canvas;
	RasterizerSceneGLES2 *scene;

	static GLuint system_fbo;

	struct Config {

		bool shrink_textures_x2;
		bool use_fast_texture_filter;
		// bool use_anisotropic_filter;

		bool hdr_supported;

		bool use_rgba_2d_shadows;

		// float anisotropic_level;

		int max_texture_image_units;
		int max_texture_size;

		bool generate_wireframes;

		bool use_texture_array_environment;

		Set<String> extensions;

		bool keep_original_textures;

		bool no_depth_prepass;
		bool force_vertex_shading;
	} config;

	struct Resources {

		GLuint white_tex;
		GLuint black_tex;
		GLuint normal_tex;
		GLuint aniso_tex;

		GLuint quadie;
		GLuint quadie_array;

	} resources;

	mutable struct Shaders {

		ShaderCompilerGLES2 compiler;

		CopyShaderGLES2 copy;

		ShaderCompilerGLES2::IdentifierActions actions_canvas;
		ShaderCompilerGLES2::IdentifierActions actions_scene;
		ShaderCompilerGLES2::IdentifierActions actions_particles;

	} shaders;

	struct Info {

		uint64_t texture_mem;
		uint64_t vertex_mem;

		struct Render {
			uint32_t object_count;
			uint32_t draw_call_count;
			uint32_t material_switch_count;
			uint32_t surface_switch_count;
			uint32_t shader_rebind_count;
			uint32_t vertices_count;

			void reset() {
				object_count = 0;
				draw_call_count = 0;
				material_switch_count = 0;
				surface_switch_count = 0;
				shader_rebind_count = 0;
				vertices_count = 0;
			}
		} render, render_final, snap;

		Info() {

			texture_mem = 0;
			vertex_mem = 0;
			render.reset();
			render_final.reset();
		}

	} info;

	/////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////DATA///////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////API////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////

	/* TEXTURE API */

	struct RenderTarget;

	struct Texture : RID_Data {

		Texture *proxy;
		Set<Texture *> proxy_owners;

		String path;
		uint32_t flags;
		int width, height;
		int alloc_width, alloc_height;
		Image::Format format;

		GLenum target;
		GLenum gl_format_cache;
		GLenum gl_internal_format_cache;
		GLenum gl_type_cache;

		int data_size;
		int total_data_size;
		bool ignore_mipmaps;

		int mipmaps;

		bool active;
		GLenum tex_id;

		uint16_t stored_cube_sides;

		RenderTarget *render_target;

		Ref<Image> images[6];

		Texture() {
			flags = 0;
			width = 0;
			height = 0;
			alloc_width = 0;
			alloc_height = 0;
			format = Image::FORMAT_L8;

			target = 0;

			data_size = 0;
			total_data_size = 0;
			ignore_mipmaps = false;

			active = false;

			tex_id = 0;

			stored_cube_sides = 0;

			proxy = NULL;

			render_target = NULL;
		}

		_ALWAYS_INLINE_ Texture *get_ptr() {
			if (proxy) {
				return proxy; //->get_ptr(); only one level of indirection, else not inlining possible.
			} else {
				return this;
			}
		}

		~Texture() {
			if (tex_id != 0) {
				glDeleteTextures(1, &tex_id);
			}

			for (Set<Texture *>::Element *E = proxy_owners.front(); E; E = E->next()) {
				E->get()->proxy = NULL;
			}

			if (proxy) {
				proxy->proxy_owners.erase(this);
			}
		}
	};

	mutable RID_Owner<Texture> texture_owner;

	Ref<Image> _get_gl_image_and_format(const Ref<Image> &p_image, Image::Format p_format, uint32_t p_flags, GLenum &r_gl_format, GLenum &r_gl_internal_format, GLenum &r_gl_type);

	virtual RID texture_create();
	virtual void texture_allocate(RID p_texture, int p_width, int p_height, Image::Format p_format, uint32_t p_flags = VS::TEXTURE_FLAGS_DEFAULT);
	virtual void texture_set_data(RID p_texture, const Ref<Image> &p_image, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT);
	virtual Ref<Image> texture_get_data(RID p_texture, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT) const;
	virtual void texture_set_flags(RID p_texture, uint32_t p_flags);
	virtual uint32_t texture_get_flags(RID p_texture) const;
	virtual Image::Format texture_get_format(RID p_texture) const;
	virtual uint32_t texture_get_texid(RID p_texture) const;
	virtual uint32_t texture_get_width(RID p_texture) const;
	virtual uint32_t texture_get_height(RID p_texture) const;
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height);

	virtual void texture_set_path(RID p_texture, const String &p_path);
	virtual String texture_get_path(RID p_texture) const;

	virtual void texture_set_shrink_all_x2_on_set_data(bool p_enable);

	virtual void texture_debug_usage(List<VS::TextureInfo> *r_info);

	virtual RID texture_create_radiance_cubemap(RID p_source, int p_resolution = -1) const;

	virtual void textures_keep_original(bool p_enable);

	virtual void texture_set_proxy(RID p_texture, RID p_proxy);

	virtual void texture_set_detect_3d_callback(RID p_texture, VisualServer::TextureDetectCallback p_callback, void *p_userdata);
	virtual void texture_set_detect_srgb_callback(RID p_texture, VisualServer::TextureDetectCallback p_callback, void *p_userdata);
	virtual void texture_set_detect_normal_callback(RID p_texture, VisualServer::TextureDetectCallback p_callback, void *p_userdata);

	/* SKY API */

	virtual RID sky_create();
	virtual void sky_set_texture(RID p_sky, RID p_panorama, int p_radiance_size);

	/* SHADER API */

	struct Material;

	struct Shader : public RID_Data {

		RID self;

		VS::ShaderMode mode;
		ShaderGLES2 *shader;
		String code;
		SelfList<Material>::List materials;

		Map<StringName, ShaderLanguage::ShaderNode::Uniform> uniforms;

		uint32_t texture_count;

		uint32_t custom_code_id;
		uint32_t version;

		SelfList<Shader> dirty_list;

		Map<StringName, RID> default_textures;

		Vector<ShaderLanguage::ShaderNode::Uniform::Hint> texture_hints;

		bool valid;

		String path;

		struct CanvasItem {

			enum BlendMode {
				BLEND_MODE_MIX,
				BLEND_MODE_ADD,
				BLEND_MODE_SUB,
				BLEND_MODE_MUL,
				BLEND_MODE_PMALPHA,
			};

			int blend_mode;

			/*
			enum LightMode {
				LIGHT_MODE_NORMAL,
				LIGHT_MODE_UNSHADED,
				LIGHT_MODE_LIGHT_ONLY
			};

			int light_mode;
			*/
			bool uses_screen_texture;
			bool uses_screen_uv;
			bool uses_time;

		} canvas_item;

		/*
		struct Spatial {

			enum BlendMode {
				BLEND_MODE_MIX,
				BLEND_MODE_ADD,
				BLEND_MODE_SUB,
				BLEND_MODE_MUL,
			};

			int blend_mode;

			enum DepthDrawMode {
				DEPTH_DRAW_OPAQUE,
				DEPTH_DRAW_ALWAYS,
				DEPTH_DRAW_NEVER,
				DEPTH_DRAW_ALPHA_PREPASS,
			};

			int depth_draw_mode;

			enum CullMode {
				CULL_MODE_FRONT,
				CULL_MODE_BACK,
				CULL_MODE_DISABLED,
			};

			int cull_mode;

			bool uses_alpha;
			bool uses_alpha_scissor;
			bool unshaded;
			bool no_depth_test;
			bool uses_vertex;
			bool uses_discard;
			bool uses_sss;
			bool uses_screen_texture;
			bool uses_time;
			bool writes_modelview_or_projection;
			bool uses_vertex_lighting;
			bool uses_world_coordinates;

		} spatial;

		struct Particles {

		} particles;
		*/

		bool uses_vertex_time;
		bool uses_fragment_time;

		Shader() :
				dirty_list(this) {

			shader = NULL;
			valid = false;
			custom_code_id = 0;
			version = 1;
		}
	};

	mutable RID_Owner<Shader> shader_owner;
	mutable SelfList<Shader>::List _shader_dirty_list;

	void _shader_make_dirty(Shader *p_shader);

	virtual RID shader_create();

	virtual void shader_set_code(RID p_shader, const String &p_code);
	virtual String shader_get_code(RID p_shader) const;
	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const;

	virtual void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture);
	virtual RID shader_get_default_texture_param(RID p_shader, const StringName &p_name) const;

	void _update_shader(Shader *p_shader) const;
	void update_dirty_shaders();

	/* COMMON MATERIAL API */

	struct Material : public RID_Data {

		Shader *shader;
		Map<StringName, Variant> params;
		SelfList<Material> list;
		SelfList<Material> dirty_list;
		Vector<RID> textures;
		float line_width;
		int render_priority;

		RID next_pass;

		uint32_t index;
		uint64_t last_pass;

		Map<Geometry *, int> geometry_owners;
		Map<RasterizerScene::InstanceBase *, int> instance_owners;

		bool can_cast_shadow_cache;
		bool is_animated_cache;

		Material() :
				list(this),
				dirty_list(this) {
			can_cast_shadow_cache = false;
			is_animated_cache = false;
			shader = NULL;
			line_width = 1.0;
			last_pass = 0;
			render_priority = 0;
		}
	};

	mutable SelfList<Material>::List _material_dirty_list;
	void _material_make_dirty(Material *p_material) const;

	mutable RID_Owner<Material> material_owner;

	virtual RID material_create();

	virtual void material_set_shader(RID p_material, RID p_shader);
	virtual RID material_get_shader(RID p_material) const;

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value);
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const;

	virtual void material_set_line_width(RID p_material, float p_width);
	virtual void material_set_next_pass(RID p_material, RID p_next_material);

	virtual bool material_is_animated(RID p_material);
	virtual bool material_casts_shadows(RID p_material);

	virtual void material_add_instance_owner(RID p_material, RasterizerScene::InstanceBase *p_instance);
	virtual void material_remove_instance_owner(RID p_material, RasterizerScene::InstanceBase *p_instance);

	virtual void material_set_render_priority(RID p_material, int priority);

	void update_dirty_materials();

	/* MESH API */
	virtual RID mesh_create();

	virtual void mesh_add_surface(RID p_mesh, uint32_t p_format, VS::PrimitiveType p_primitive, const PoolVector<uint8_t> &p_array, int p_vertex_count, const PoolVector<uint8_t> &p_index_array, int p_index_count, const AABB &p_aabb, const Vector<PoolVector<uint8_t> > &p_blend_shapes = Vector<PoolVector<uint8_t> >(), const Vector<AABB> &p_bone_aabbs = Vector<AABB>());

	virtual void mesh_set_blend_shape_count(RID p_mesh, int p_amount);
	virtual int mesh_get_blend_shape_count(RID p_mesh) const;

	virtual void mesh_set_blend_shape_mode(RID p_mesh, VS::BlendShapeMode p_mode);
	virtual VS::BlendShapeMode mesh_get_blend_shape_mode(RID p_mesh) const;

	virtual void mesh_surface_update_region(RID p_mesh, int p_surface, int p_offset, const PoolVector<uint8_t> &p_data);

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material);
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const;

	virtual int mesh_surface_get_array_len(RID p_mesh, int p_surface) const;
	virtual int mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const;

	virtual PoolVector<uint8_t> mesh_surface_get_array(RID p_mesh, int p_surface) const;
	virtual PoolVector<uint8_t> mesh_surface_get_index_array(RID p_mesh, int p_surface) const;

	virtual uint32_t mesh_surface_get_format(RID p_mesh, int p_surface) const;
	virtual VS::PrimitiveType mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const;

	virtual AABB mesh_surface_get_aabb(RID p_mesh, int p_surface) const;
	virtual Vector<PoolVector<uint8_t> > mesh_surface_get_blend_shapes(RID p_mesh, int p_surface) const;
	virtual Vector<AABB> mesh_surface_get_skeleton_aabb(RID p_mesh, int p_surface) const;

	virtual void mesh_remove_surface(RID p_mesh, int p_surface);
	virtual int mesh_get_surface_count(RID p_mesh) const;

	virtual void mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb);
	virtual AABB mesh_get_custom_aabb(RID p_mesh) const;

	virtual AABB mesh_get_aabb(RID p_mesh, RID p_skeleton) const;
	virtual void mesh_clear(RID p_mesh);

	/* MULTIMESH API */

	virtual RID multimesh_create();

	virtual void multimesh_allocate(RID p_multimesh, int p_instances, VS::MultimeshTransformFormat p_transform_format, VS::MultimeshColorFormat p_color_format);
	virtual int multimesh_get_instance_count(RID p_multimesh) const;

	virtual void multimesh_set_mesh(RID p_multimesh, RID p_mesh);
	virtual void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform);
	virtual void multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform);
	virtual void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color);

	virtual RID multimesh_get_mesh(RID p_multimesh) const;

	virtual Transform multimesh_instance_get_transform(RID p_multimesh, int p_index) const;
	virtual Transform2D multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const;
	virtual Color multimesh_instance_get_color(RID p_multimesh, int p_index) const;

	virtual void multimesh_set_visible_instances(RID p_multimesh, int p_visible);
	virtual int multimesh_get_visible_instances(RID p_multimesh) const;

	virtual AABB multimesh_get_aabb(RID p_multimesh) const;

	void update_dirty_multimeshes();

	/* IMMEDIATE API */

	virtual RID immediate_create();
	virtual void immediate_begin(RID p_immediate, VS::PrimitiveType p_rimitive, RID p_texture = RID());
	virtual void immediate_vertex(RID p_immediate, const Vector3 &p_vertex);
	virtual void immediate_normal(RID p_immediate, const Vector3 &p_normal);
	virtual void immediate_tangent(RID p_immediate, const Plane &p_tangent);
	virtual void immediate_color(RID p_immediate, const Color &p_color);
	virtual void immediate_uv(RID p_immediate, const Vector2 &tex_uv);
	virtual void immediate_uv2(RID p_immediate, const Vector2 &tex_uv);
	virtual void immediate_end(RID p_immediate);
	virtual void immediate_clear(RID p_immediate);
	virtual void immediate_set_material(RID p_immediate, RID p_material);
	virtual RID immediate_get_material(RID p_immediate) const;
	virtual AABB immediate_get_aabb(RID p_immediate) const;

	/* SKELETON API */

	void update_dirty_skeletons();

	virtual RID skeleton_create();
	virtual void skeleton_allocate(RID p_skeleton, int p_bones, bool p_2d_skeleton = false);
	virtual int skeleton_get_bone_count(RID p_skeleton) const;
	virtual void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform);
	virtual Transform skeleton_bone_get_transform(RID p_skeleton, int p_bone) const;
	virtual void skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform);
	virtual Transform2D skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const;

	/* Light API */

	virtual RID light_create(VS::LightType p_type);

	virtual void light_set_color(RID p_light, const Color &p_color);
	virtual void light_set_param(RID p_light, VS::LightParam p_param, float p_value);
	virtual void light_set_shadow(RID p_light, bool p_enabled);
	virtual void light_set_shadow_color(RID p_light, const Color &p_color);
	virtual void light_set_projector(RID p_light, RID p_texture);
	virtual void light_set_negative(RID p_light, bool p_enable);
	virtual void light_set_cull_mask(RID p_light, uint32_t p_mask);
	virtual void light_set_reverse_cull_face_mode(RID p_light, bool p_enabled);

	virtual void light_omni_set_shadow_mode(RID p_light, VS::LightOmniShadowMode p_mode);
	virtual void light_omni_set_shadow_detail(RID p_light, VS::LightOmniShadowDetail p_detail);

	virtual void light_directional_set_shadow_mode(RID p_light, VS::LightDirectionalShadowMode p_mode);
	virtual void light_directional_set_blend_splits(RID p_light, bool p_enable);
	virtual bool light_directional_get_blend_splits(RID p_light) const;

	virtual VS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light);
	virtual VS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light);

	virtual void light_directional_set_shadow_depth_range_mode(RID p_light, VS::LightDirectionalShadowDepthRangeMode p_range_mode);
	virtual VS::LightDirectionalShadowDepthRangeMode light_directional_get_shadow_depth_range_mode(RID p_light) const;

	virtual bool light_has_shadow(RID p_light) const;

	virtual VS::LightType light_get_type(RID p_light) const;
	virtual float light_get_param(RID p_light, VS::LightParam p_param);
	virtual Color light_get_color(RID p_light);

	virtual AABB light_get_aabb(RID p_light) const;
	virtual uint64_t light_get_version(RID p_light) const;

	/* PROBE API */
	virtual RID reflection_probe_create();

	virtual void reflection_probe_set_update_mode(RID p_probe, VS::ReflectionProbeUpdateMode p_mode);
	virtual void reflection_probe_set_intensity(RID p_probe, float p_intensity);
	virtual void reflection_probe_set_interior_ambient(RID p_probe, const Color &p_ambient);
	virtual void reflection_probe_set_interior_ambient_energy(RID p_probe, float p_energy);
	virtual void reflection_probe_set_interior_ambient_probe_contribution(RID p_probe, float p_contrib);
	virtual void reflection_probe_set_max_distance(RID p_probe, float p_distance);
	virtual void reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents);
	virtual void reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset);
	virtual void reflection_probe_set_as_interior(RID p_probe, bool p_enable);
	virtual void reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable);
	virtual void reflection_probe_set_enable_shadows(RID p_probe, bool p_enable);
	virtual void reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers);

	virtual AABB reflection_probe_get_aabb(RID p_probe) const;
	virtual VS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const;
	virtual uint32_t reflection_probe_get_cull_mask(RID p_probe) const;

	virtual Vector3 reflection_probe_get_extents(RID p_probe) const;
	virtual Vector3 reflection_probe_get_origin_offset(RID p_probe) const;
	virtual float reflection_probe_get_origin_max_distance(RID p_probe) const;
	virtual bool reflection_probe_renders_shadows(RID p_probe) const;

	/* GI PROBE API */
	virtual RID gi_probe_create();

	virtual void gi_probe_set_bounds(RID p_probe, const AABB &p_bounds);
	virtual AABB gi_probe_get_bounds(RID p_probe) const;

	virtual void gi_probe_set_cell_size(RID p_probe, float p_size);
	virtual float gi_probe_get_cell_size(RID p_probe) const;

	virtual void gi_probe_set_to_cell_xform(RID p_probe, const Transform &p_xform);
	virtual Transform gi_probe_get_to_cell_xform(RID p_probe) const;

	virtual void gi_probe_set_dynamic_data(RID p_probe, const PoolVector<int> &p_data);
	virtual PoolVector<int> gi_probe_get_dynamic_data(RID p_probe) const;

	virtual void gi_probe_set_dynamic_range(RID p_probe, int p_range);
	virtual int gi_probe_get_dynamic_range(RID p_probe) const;

	virtual void gi_probe_set_energy(RID p_probe, float p_range);
	virtual float gi_probe_get_energy(RID p_probe) const;

	virtual void gi_probe_set_bias(RID p_probe, float p_range);
	virtual float gi_probe_get_bias(RID p_probe) const;

	virtual void gi_probe_set_normal_bias(RID p_probe, float p_range);
	virtual float gi_probe_get_normal_bias(RID p_probe) const;

	virtual void gi_probe_set_propagation(RID p_probe, float p_range);
	virtual float gi_probe_get_propagation(RID p_probe) const;

	virtual void gi_probe_set_interior(RID p_probe, bool p_enable);
	virtual bool gi_probe_is_interior(RID p_probe) const;

	virtual void gi_probe_set_compress(RID p_probe, bool p_enable);
	virtual bool gi_probe_is_compressed(RID p_probe) const;

	virtual uint32_t gi_probe_get_version(RID p_probe);

	virtual GIProbeCompression gi_probe_get_dynamic_data_get_preferred_compression() const;
	virtual RID gi_probe_dynamic_data_create(int p_width, int p_height, int p_depth, GIProbeCompression p_compression);
	virtual void gi_probe_dynamic_data_update(RID p_gi_probe_data, int p_depth_slice, int p_slice_count, int p_mipmap, const void *p_data);

	/* LIGHTMAP */

	virtual RID lightmap_capture_create();
	virtual void lightmap_capture_set_bounds(RID p_capture, const AABB &p_bounds);
	virtual AABB lightmap_capture_get_bounds(RID p_capture) const;
	virtual void lightmap_capture_set_octree(RID p_capture, const PoolVector<uint8_t> &p_octree);
	virtual PoolVector<uint8_t> lightmap_capture_get_octree(RID p_capture) const;
	virtual void lightmap_capture_set_octree_cell_transform(RID p_capture, const Transform &p_xform);
	virtual Transform lightmap_capture_get_octree_cell_transform(RID p_capture) const;
	virtual void lightmap_capture_set_octree_cell_subdiv(RID p_capture, int p_subdiv);
	virtual int lightmap_capture_get_octree_cell_subdiv(RID p_capture) const;
	virtual void lightmap_capture_set_energy(RID p_capture, float p_energy);
	virtual float lightmap_capture_get_energy(RID p_capture) const;
	virtual const PoolVector<LightmapCaptureOctree> *lightmap_capture_get_octree_ptr(RID p_capture) const;

	/* PARTICLES */
	void update_particles();

	virtual RID particles_create();

	virtual void particles_set_emitting(RID p_particles, bool p_emitting);
	virtual bool particles_get_emitting(RID p_particles);

	virtual void particles_set_amount(RID p_particles, int p_amount);
	virtual void particles_set_lifetime(RID p_particles, float p_lifetime);
	virtual void particles_set_one_shot(RID p_particles, bool p_one_shot);
	virtual void particles_set_pre_process_time(RID p_particles, float p_time);
	virtual void particles_set_explosiveness_ratio(RID p_particles, float p_ratio);
	virtual void particles_set_randomness_ratio(RID p_particles, float p_ratio);
	virtual void particles_set_custom_aabb(RID p_particles, const AABB &p_aabb);
	virtual void particles_set_speed_scale(RID p_particles, float p_scale);
	virtual void particles_set_use_local_coordinates(RID p_particles, bool p_enable);
	virtual void particles_set_process_material(RID p_particles, RID p_material);
	virtual void particles_set_fixed_fps(RID p_particles, int p_fps);
	virtual void particles_set_fractional_delta(RID p_particles, bool p_enable);
	virtual void particles_restart(RID p_particles);

	virtual void particles_set_draw_order(RID p_particles, VS::ParticlesDrawOrder p_order);

	virtual void particles_set_draw_passes(RID p_particles, int p_passes);
	virtual void particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh);

	virtual void particles_request_process(RID p_particles);
	virtual AABB particles_get_current_aabb(RID p_particles);
	virtual AABB particles_get_aabb(RID p_particles) const;

	virtual void particles_set_emission_transform(RID p_particles, const Transform &p_transform);

	virtual int particles_get_draw_passes(RID p_particles) const;
	virtual RID particles_get_draw_pass_mesh(RID p_particles, int p_pass) const;

	/* INSTANCE */

	virtual void instance_add_skeleton(RID p_skeleton, RasterizerScene::InstanceBase *p_instance);
	virtual void instance_remove_skeleton(RID p_skeleton, RasterizerScene::InstanceBase *p_instance);

	virtual void instance_add_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance);
	virtual void instance_remove_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance);

	/* RENDER TARGET */

	struct RenderTarget : public RID_Data {
		GLuint fbo;

		GLuint color;
		GLuint depth;

		// TODO post processing effects?

		// TODO HDR?

		// TODO this is hardcoded for texscreen copies for now

		struct Effect {
			GLuint fbo;
			int width;
			int height;

			GLuint color;

			Effect() {
				fbo = 0;
				width = 0;
				height = 0;
				color = 0;
			}
		};

		Effect copy_screen_effect;

		int width, height;

		bool flags[RENDER_TARGET_FLAG_MAX];

		bool used_in_frame;
		VS::ViewportMSAA msaa;

		RID texture;

		RenderTarget() {
			fbo = 0;

			color = 0;
			depth = 0;

			width = 0;
			height = 0;

			for (int i = 0; i < RENDER_TARGET_FLAG_MAX; i++) {
				flags[i] = false;
			}

			used_in_frame = false;

			msaa = VS::VIEWPORT_MSAA_DISABLED;
		}
	};

	mutable RID_Owner<RenderTarget> render_target_owner;

	void _render_target_clear(RenderTarget *rt);
	void _render_target_allocate(RenderTarget *rt);

	virtual RID render_target_create();
	virtual void render_target_set_size(RID p_render_target, int p_width, int p_height);
	virtual RID render_target_get_texture(RID p_render_target) const;

	virtual void render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value);
	virtual bool render_target_was_used(RID p_render_target);
	virtual void render_target_clear_used(RID p_render_target);
	virtual void render_target_set_msaa(RID p_render_target, VS::ViewportMSAA p_msaa);

	/* CANVAS SHADOW */

	virtual RID canvas_light_shadow_buffer_create(int p_width);

	/* LIGHT SHADOW MAPPING */

	virtual RID canvas_light_occluder_create();
	virtual void canvas_light_occluder_set_polylines(RID p_occluder, const PoolVector<Vector2> &p_lines);

	virtual VS::InstanceType get_base_type(RID p_rid) const;

	virtual bool free(RID p_rid);

	struct Frame {

		RenderTarget *current_rt;

		bool clear_request;
		Color clear_request_color;
		int canvas_draw_commands;
		float time[4];
		float delta;
		uint64_t prev_tick;
		uint64_t count;

	} frame;

	void initialize();
	void finalize();

	virtual bool has_os_feature(const String &p_feature) const;

	virtual void update_dirty_resources();

	virtual void set_debug_generate_wireframes(bool p_generate);

	virtual void render_info_begin_capture();
	virtual void render_info_end_capture();
	virtual int get_captured_render_info(VS::RenderInfo p_info);

	virtual int get_render_info(VS::RenderInfo p_info);

	RasterizerStorageGLES2();
};

#endif // RASTERIZERSTORAGEGLES2_H
