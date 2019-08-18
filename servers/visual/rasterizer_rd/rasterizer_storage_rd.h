/*************************************************************************/
/*  rasterizer_storage_rd.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef RASTERIZER_STORAGE_RD_H
#define RASTERIZER_STORAGE_RD_H

#include "core/rid_owner.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual/rasterizer_rd/rasterizer_effects_rd.h"
#include "servers/visual/rasterizer_rd/shader_compiler_rd.h"
#include "servers/visual/rendering_device.h"

class RasterizerStorageRD : public RasterizerStorage {
public:
	enum ShaderType {
		SHADER_TYPE_2D,
		SHADER_TYPE_3D,
		SHADER_TYPE_PARTICLES,
		SHADER_TYPE_MAX
	};

	struct ShaderData {
		virtual void set_code(const String &p_Code) = 0;
		virtual void set_default_texture_param(const StringName &p_name, RID p_texture) = 0;
		virtual void get_param_list(List<PropertyInfo> *p_param_list) const = 0;
		virtual bool is_param_texture(const StringName &p_param) const = 0;
		virtual bool is_animated() const = 0;
		virtual bool casts_shadows() const = 0;
		virtual Variant get_default_parameter(const StringName &p_parameter) const = 0;
		virtual ~ShaderData() {}
	};

	typedef ShaderData *(*ShaderDataRequestFunction)();

	struct MaterialData {

		void update_uniform_buffer(const Map<StringName, ShaderLanguage::ShaderNode::Uniform> &p_uniforms, const uint32_t *p_uniform_offsets, const Map<StringName, Variant> &p_parameters, uint8_t *p_buffer, uint32_t p_buffer_size, bool p_use_linear_color);
		void update_textures(const Map<StringName, Variant> &p_parameters, const Map<StringName, RID> &p_default_textures, const Vector<ShaderCompilerRD::GeneratedCode::Texture> &p_texture_uniforms, RID *p_textures);

		virtual void set_render_priority(int p_priority) = 0;
		virtual void set_next_pass(RID p_pass) = 0;
		virtual void update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) = 0;
		virtual ~MaterialData() {}
	};
	typedef MaterialData *(*MaterialDataRequestFunction)(ShaderData *);

	enum DefaultRDTexture {
		DEFAULT_RD_TEXTURE_WHITE,
		DEFAULT_RD_TEXTURE_BLACK,
		DEFAULT_RD_TEXTURE_NORMAL,
		DEFAULT_RD_TEXTURE_ANISO,
		DEFAULT_RD_TEXTURE_MULTIMESH_BUFFER,
		DEFAULT_RD_TEXTURE_MAX
	};

	enum DefaultRDBuffer {
		DEFAULT_RD_BUFFER_VERTEX,
		DEFAULT_RD_BUFFER_NORMAL,
		DEFAULT_RD_BUFFER_TANGENT,
		DEFAULT_RD_BUFFER_COLOR,
		DEFAULT_RD_BUFFER_TEX_UV,
		DEFAULT_RD_BUFFER_TEX_UV2,
		DEFAULT_RD_BUFFER_BONES,
		DEFAULT_RD_BUFFER_WEIGHTS,
		DEFAULT_RD_BUFFER_MAX,
	};

private:
	/* TEXTURE API */
	struct Texture {

		enum Type {
			TYPE_2D,
			TYPE_LAYERED,
			TYPE_3D
		};

		Type type;

		RenderingDevice::TextureType rd_type;
		RID rd_texture;
		RID rd_texture_srgb;
		RenderingDevice::DataFormat rd_format;
		RenderingDevice::DataFormat rd_format_srgb;

		RD::TextureView rd_view;

		Image::Format format;
		Image::Format validated_format;

		int width;
		int height;
		int depth;
		int layers;
		int mipmaps;

		int height_2d;
		int width_2d;

		bool is_render_target;
		bool is_proxy;

		Ref<Image> image_cache_2d;
		String path;

		RID proxy_to;
		Vector<RID> proxies;
	};

	struct TextureToRDFormat {
		RD::DataFormat format;
		RD::DataFormat format_srgb;
		RD::TextureSwizzle swizzle_r;
		RD::TextureSwizzle swizzle_g;
		RD::TextureSwizzle swizzle_b;
		RD::TextureSwizzle swizzle_a;
		TextureToRDFormat() {
			format = RD::DATA_FORMAT_MAX;
			format_srgb = RD::DATA_FORMAT_MAX;
			swizzle_r = RD::TEXTURE_SWIZZLE_R;
			swizzle_g = RD::TEXTURE_SWIZZLE_G;
			swizzle_b = RD::TEXTURE_SWIZZLE_B;
			swizzle_a = RD::TEXTURE_SWIZZLE_A;
		}
	};

	//textures can be created from threads, so this RID_Owner is thread safe
	mutable RID_Owner<Texture, true> texture_owner;

	Ref<Image> _validate_texture_format(const Ref<Image> &p_image, TextureToRDFormat &r_format);

	RID default_rd_textures[DEFAULT_RD_TEXTURE_MAX];
	RID default_rd_samplers[VS::CANVAS_ITEM_TEXTURE_FILTER_MAX][VS::CANVAS_ITEM_TEXTURE_REPEAT_MAX];

	/* SHADER */

	struct Material;

	struct Shader {
		ShaderData *data;
		String code;
		ShaderType type;
		Map<StringName, RID> default_texture_parameter;
		Set<Material *> owners;
	};

	ShaderDataRequestFunction shader_data_request_func[SHADER_TYPE_MAX];
	mutable RID_Owner<Shader> shader_owner;

	/* Material */

	struct Material {
		RID self;
		MaterialData *data;
		Shader *shader;
		//shortcut to shader data and type
		ShaderType shader_type;
		bool update_requested;
		bool uniform_dirty;
		bool texture_dirty;
		Material *update_next;
		Map<StringName, Variant> params;
		int32_t priority;
		RID next_pass;
		RasterizerScene::InstanceDependency instance_dependency;
	};

	MaterialDataRequestFunction material_data_request_func[SHADER_TYPE_MAX];
	mutable RID_Owner<Material> material_owner;

	Material *material_update_list;
	void _material_queue_update(Material *material, bool p_uniform, bool p_texture);
	void _update_queued_materials();

	/* Mesh */

	struct Mesh {

		struct Surface {
			VS::PrimitiveType primitive;
			uint32_t format = 0;

			RID vertex_buffer;
			uint32_t vertex_count = 0;

			// A different pipeline needs to be allocated
			// depending on the inputs available in the
			// material.
			// There are never that many geometry/material
			// combinations, so a simple array is the most
			// cache-efficient structure.

			struct Version {
				uint32_t input_mask;
				RD::VertexFormatID vertex_format;
				RID vertex_array;
			};

			SpinLock version_lock; //needed to access versions
			Version *versions = nullptr; //allocated on demand
			uint32_t version_count = 0;

			RID index_buffer;
			RID index_array;
			uint32_t index_count = 0;

			struct LOD {
				float edge_length;
				RID index_buffer;
				RID index_array;
			};

			LOD *lods = nullptr;
			uint32_t lod_count = 0;

			AABB aabb;

			Vector<AABB> bone_aabbs;

			Vector<RID> blend_shapes;
			RID blend_shape_base_buffer; //source buffer goes here when using blend shapes, and main one is uncompressed

			RID material;
		};

		uint32_t blend_shape_count = 0;
		VS::BlendShapeMode blend_shape_mode = VS::BLEND_SHAPE_MODE_NORMALIZED;

		Surface **surfaces = nullptr;
		uint32_t surface_count = 0;

		Vector<AABB> bone_aabbs;

		AABB aabb;
		AABB custom_aabb;

		Vector<RID> material_cache;

		RasterizerScene::InstanceDependency instance_dependency;
	};

	mutable RID_Owner<Mesh> mesh_owner;

	void _mesh_surface_generate_version_for_input_mask(Mesh::Surface *s, uint32_t p_input_mask);

	RID mesh_default_rd_buffers[DEFAULT_RD_BUFFER_MAX];

	/* RENDER TARGET */

	struct RenderTarget {

		Size2i size;
		RID framebuffer;
		RID color;

		//used for retrieving from CPU
		RD::DataFormat color_format;
		RD::DataFormat color_format_srgb;
		Image::Format image_format;

		bool flags[RENDER_TARGET_FLAG_MAX];

		RID backbuffer; //used for effects
		RID backbuffer_fb;

		struct BackbufferMipmap {
			RID mipmap;
			RID mipmap_fb;
			RID mipmap_copy;
			RID mipmap_copy_fb;
		};

		Vector<BackbufferMipmap> backbuffer_mipmaps;
		RID backbuffer_uniform_set;

		//texture generated for this owner (nor RD).
		RID texture;
		bool was_used;

		//clear request
		bool clear_requested;
		Color clear_color;
	};

	RID_Owner<RenderTarget> render_target_owner;

	void _clear_render_target(RenderTarget *rt);
	void _update_render_target(RenderTarget *rt);
	void _create_render_target_backbuffer(RenderTarget *rt);

	/* EFFECTS */

	RasterizerEffectsRD effects;

public:
	/* TEXTURE API */

	virtual RID texture_2d_create(const Ref<Image> &p_image);
	virtual RID texture_2d_layered_create(const Vector<Ref<Image> > &p_layers, VS::TextureLayeredType p_layered_type);
	virtual RID texture_3d_create(const Vector<Ref<Image> > &p_slices); //all slices, then all the mipmaps, must be coherent
	virtual RID texture_proxy_create(RID p_base);

	virtual void _texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer, bool p_immediate);

	virtual void texture_2d_update_immediate(RID p_texture, const Ref<Image> &p_image, int p_layer = 0); //mostly used for video and streaming
	virtual void texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0);
	virtual void texture_3d_update(RID p_texture, const Ref<Image> &p_image, int p_depth, int p_mipmap);
	virtual void texture_proxy_update(RID p_texture, RID p_proxy_to);

	//these two APIs can be used together or in combination with the others.
	virtual RID texture_2d_placeholder_create();
	virtual RID texture_2d_layered_placeholder_create();
	virtual RID texture_3d_placeholder_create();

	virtual Ref<Image> texture_2d_get(RID p_texture) const;
	virtual Ref<Image> texture_2d_layer_get(RID p_texture, int p_layer) const;
	virtual Ref<Image> texture_3d_slice_get(RID p_texture, int p_depth, int p_mipmap) const;

	virtual void texture_replace(RID p_texture, RID p_by_texture);
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height);

	virtual void texture_set_path(RID p_texture, const String &p_path);
	virtual String texture_get_path(RID p_texture) const;

	virtual void texture_set_detect_3d_callback(RID p_texture, VS::TextureDetectCallback p_callback, void *p_userdata);
	virtual void texture_set_detect_normal_callback(RID p_texture, VS::TextureDetectCallback p_callback, void *p_userdata);
	virtual void texture_set_detect_roughness_callback(RID p_texture, VS::TextureDetectRoughnessCallback p_callback, void *p_userdata);

	virtual void texture_debug_usage(List<VS::TextureInfo> *r_info);

	virtual void texture_set_proxy(RID p_proxy, RID p_base);
	virtual void texture_set_force_redraw_if_visible(RID p_texture, bool p_enable);

	virtual Size2 texture_size_with_proxy(RID p_proxy);

	//internal usage

	_FORCE_INLINE_ RID texture_get_rd_texture(RID p_texture, bool p_srgb = false) {
		if (p_texture.is_null()) {
			return RID();
		}
		Texture *tex = texture_owner.getornull(p_texture);

		if (!tex) {
			return RID();
		}
		return (p_srgb && tex->rd_texture_srgb.is_valid()) ? tex->rd_texture_srgb : tex->rd_texture;
	}

	_FORCE_INLINE_ Size2i texture_2d_get_size(RID p_texture) {
		if (p_texture.is_null()) {
			return Size2i();
		}
		Texture *tex = texture_owner.getornull(p_texture);

		if (!tex) {
			return Size2i();
		}
		return Size2i(tex->width_2d, tex->height_2d);
	}

	_FORCE_INLINE_ RID texture_rd_get_default(DefaultRDTexture p_texture) {
		return default_rd_textures[p_texture];
	}
	_FORCE_INLINE_ RID sampler_rd_get_default(VS::CanvasItemTextureFilter p_filter, VS::CanvasItemTextureRepeat p_repeat) {
		return default_rd_samplers[p_filter][p_repeat];
	}

	/* SKY API */

	RID sky_create() { return RID(); }
	void sky_set_texture(RID p_sky, RID p_cube_map, int p_radiance_size) {}

	/* SHADER API */

	RID shader_create();

	void shader_set_code(RID p_shader, const String &p_code);
	String shader_get_code(RID p_shader) const;
	void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const;

	void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture);
	RID shader_get_default_texture_param(RID p_shader, const StringName &p_name) const;
	Variant shader_get_param_default(RID p_shader, const StringName &p_param) const;
	void shader_set_data_request_function(ShaderType p_shader_type, ShaderDataRequestFunction p_function);

	/* COMMON MATERIAL API */

	RID material_create();

	void material_set_shader(RID p_material, RID p_shader);

	void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value);
	Variant material_get_param(RID p_material, const StringName &p_param) const;

	void material_set_next_pass(RID p_material, RID p_next_material);
	void material_set_render_priority(RID p_material, int priority);

	bool material_is_animated(RID p_material);
	bool material_casts_shadows(RID p_material);

	void material_update_dependency(RID p_material, RasterizerScene::InstanceBase *p_instance);
	void material_force_update_textures(RID p_material, ShaderType p_shader_type);

	void material_set_data_request_function(ShaderType p_shader_type, MaterialDataRequestFunction p_function);

	_FORCE_INLINE_ MaterialData *material_get_data(RID p_material, ShaderType p_shader_type) {
		Material *material = material_owner.getornull(p_material);
		if (material->shader_type != p_shader_type) {
			return NULL;
		} else {
			return material->data;
		}
	}

	/* MESH API */

	virtual RID mesh_create();

	/// Return stride
	virtual void mesh_add_surface(RID p_mesh, const VS::SurfaceData &p_surface);

	virtual int mesh_get_blend_shape_count(RID p_mesh) const;

	virtual void mesh_set_blend_shape_mode(RID p_mesh, VS::BlendShapeMode p_mode);
	virtual VS::BlendShapeMode mesh_get_blend_shape_mode(RID p_mesh) const;

	virtual void mesh_surface_update_region(RID p_mesh, int p_surface, int p_offset, const PoolVector<uint8_t> &p_data);

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material);
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const;

	virtual VS::SurfaceData mesh_get_surface(RID p_mesh, int p_surface) const;

	virtual int mesh_get_surface_count(RID p_mesh) const;

	virtual void mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb);
	virtual AABB mesh_get_custom_aabb(RID p_mesh) const;

	virtual AABB mesh_get_aabb(RID p_mesh, RID p_skeleton = RID());

	virtual void mesh_clear(RID p_mesh);

	_FORCE_INLINE_ const RID *mesh_get_surface_count_and_materials(RID p_mesh, uint32_t &r_surface_count) {
		Mesh *mesh = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!mesh, NULL);
		r_surface_count = mesh->surface_count;
		if (r_surface_count == 0) {
			return NULL;
		}
		if (mesh->material_cache.empty()) {
			mesh->material_cache.resize(mesh->surface_count);
			for (uint32_t i = 0; i < r_surface_count; i++) {
				mesh->material_cache.write[i] = mesh->surfaces[i]->material;
			}
		}

		return mesh->material_cache.ptr();
	}

	_FORCE_INLINE_ void mesh_get_arrays_primitive_and_format(RID p_mesh, uint32_t p_surface_index, uint32_t p_input_mask, VS::PrimitiveType &r_primitive, RID &r_vertex_array_rd, RID &r_index_array_rd, RD::VertexFormatID &r_vertex_format) {
		Mesh *mesh = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND(!mesh);
		ERR_FAIL_INDEX(p_surface_index, mesh->surface_count);

		Mesh::Surface *s = mesh->surfaces[p_surface_index];

		r_index_array_rd = s->index_array;

		s->version_lock.lock();

		//there will never be more than, at much, 3 or 4 versions, so iterating is the fastest way

		for (uint32_t i = 0; i < s->version_count; i++) {
			if (s->versions[i].input_mask != p_input_mask) {
				continue;
			}
			//we have this version, hooray
			r_vertex_format = s->versions[i].vertex_format;
			r_vertex_array_rd = s->versions[i].vertex_array;
			s->version_lock.unlock();
			return;
		}

		uint32_t version = s->version_count; //gets added at the end

		_mesh_surface_generate_version_for_input_mask(s, p_input_mask);

		r_vertex_format = s->versions[version].vertex_format;
		r_vertex_array_rd = s->versions[version].vertex_array;

		s->version_lock.unlock();
	}

	_FORCE_INLINE_ RID mesh_get_default_rd_buffer(DefaultRDBuffer p_buffer) {
		ERR_FAIL_INDEX_V(p_buffer, DEFAULT_RD_BUFFER_MAX, RID());
		return mesh_default_rd_buffers[p_buffer];
	}

	/* MULTIMESH API */

	virtual RID multimesh_create() { return RID(); }

	void multimesh_allocate(RID p_multimesh, int p_instances, VS::MultimeshTransformFormat p_transform_format, VS::MultimeshColorFormat p_color_format, VS::MultimeshCustomDataFormat p_data = VS::MULTIMESH_CUSTOM_DATA_NONE) {}
	int multimesh_get_instance_count(RID p_multimesh) const { return 0; }

	void multimesh_set_mesh(RID p_multimesh, RID p_mesh) {}
	void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform) {}
	void multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) {}
	void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) {}
	void multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) {}

	RID multimesh_get_mesh(RID p_multimesh) const { return RID(); }

	Transform multimesh_instance_get_transform(RID p_multimesh, int p_index) const { return Transform(); }
	Transform2D multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const { return Transform2D(); }
	Color multimesh_instance_get_color(RID p_multimesh, int p_index) const { return Color(); }
	Color multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const { return Color(); }

	void multimesh_set_as_bulk_array(RID p_multimesh, const PoolVector<float> &p_array) {}

	void multimesh_set_visible_instances(RID p_multimesh, int p_visible) {}
	int multimesh_get_visible_instances(RID p_multimesh) const { return 0; }

	AABB multimesh_get_aabb(RID p_multimesh) const { return AABB(); }

	/* IMMEDIATE API */

	RID immediate_create() { return RID(); }
	void immediate_begin(RID p_immediate, VS::PrimitiveType p_rimitive, RID p_texture = RID()) {}
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

	RID light_create(VS::LightType p_type) { return RID(); }

	RID directional_light_create() { return light_create(VS::LIGHT_DIRECTIONAL); }
	RID omni_light_create() { return light_create(VS::LIGHT_OMNI); }
	RID spot_light_create() { return light_create(VS::LIGHT_SPOT); }

	void light_set_color(RID p_light, const Color &p_color) {}
	void light_set_param(RID p_light, VS::LightParam p_param, float p_value) {}
	void light_set_shadow(RID p_light, bool p_enabled) {}
	void light_set_shadow_color(RID p_light, const Color &p_color) {}
	void light_set_projector(RID p_light, RID p_texture) {}
	void light_set_negative(RID p_light, bool p_enable) {}
	void light_set_cull_mask(RID p_light, uint32_t p_mask) {}
	void light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) {}
	void light_set_use_gi(RID p_light, bool p_enabled) {}

	void light_omni_set_shadow_mode(RID p_light, VS::LightOmniShadowMode p_mode) {}
	void light_omni_set_shadow_detail(RID p_light, VS::LightOmniShadowDetail p_detail) {}

	void light_directional_set_shadow_mode(RID p_light, VS::LightDirectionalShadowMode p_mode) {}
	void light_directional_set_blend_splits(RID p_light, bool p_enable) {}
	bool light_directional_get_blend_splits(RID p_light) const { return false; }
	void light_directional_set_shadow_depth_range_mode(RID p_light, VS::LightDirectionalShadowDepthRangeMode p_range_mode) {}
	VS::LightDirectionalShadowDepthRangeMode light_directional_get_shadow_depth_range_mode(RID p_light) const { return VS::LIGHT_DIRECTIONAL_SHADOW_DEPTH_RANGE_STABLE; }

	VS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) { return VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL; }
	VS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) { return VS::LIGHT_OMNI_SHADOW_DUAL_PARABOLOID; }

	bool light_has_shadow(RID p_light) const { return false; }

	VS::LightType light_get_type(RID p_light) const { return VS::LIGHT_OMNI; }
	AABB light_get_aabb(RID p_light) const { return AABB(); }
	float light_get_param(RID p_light, VS::LightParam p_param) { return 0.0; }
	Color light_get_color(RID p_light) { return Color(); }
	bool light_get_use_gi(RID p_light) { return false; }
	uint64_t light_get_version(RID p_light) const { return 0; }

	/* PROBE API */

	RID reflection_probe_create() { return RID(); }

	void reflection_probe_set_update_mode(RID p_probe, VS::ReflectionProbeUpdateMode p_mode) {}
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
	VS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const { return VisualServer::REFLECTION_PROBE_UPDATE_ONCE; }
	uint32_t reflection_probe_get_cull_mask(RID p_probe) const { return 0; }
	Vector3 reflection_probe_get_extents(RID p_probe) const { return Vector3(); }
	Vector3 reflection_probe_get_origin_offset(RID p_probe) const { return Vector3(); }
	float reflection_probe_get_origin_max_distance(RID p_probe) const { return 0.0; }
	bool reflection_probe_renders_shadows(RID p_probe) const { return false; }

	void base_update_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance);
	void skeleton_update_dependency(RID p_skeleton, RasterizerScene::InstanceBase *p_instance) {}

	/* GI PROBE API */

	RID gi_probe_create() { return RID(); }

	void gi_probe_set_bounds(RID p_probe, const AABB &p_bounds) {}
	AABB gi_probe_get_bounds(RID p_probe) const { return AABB(); }

	void gi_probe_set_cell_size(RID p_probe, float p_range) {}
	float gi_probe_get_cell_size(RID p_probe) const { return 0.0; }

	void gi_probe_set_to_cell_xform(RID p_probe, const Transform &p_xform) {}
	Transform gi_probe_get_to_cell_xform(RID p_probe) const { return Transform(); }

	void gi_probe_set_dynamic_data(RID p_probe, const PoolVector<int> &p_data) {}
	PoolVector<int> gi_probe_get_dynamic_data(RID p_probe) const {
		PoolVector<int> p;
		return p;
	}

	void gi_probe_set_dynamic_range(RID p_probe, int p_range) {}
	int gi_probe_get_dynamic_range(RID p_probe) const { return 0; }

	void gi_probe_set_energy(RID p_probe, float p_range) {}
	float gi_probe_get_energy(RID p_probe) const { return 0.0; }

	void gi_probe_set_bias(RID p_probe, float p_range) {}
	float gi_probe_get_bias(RID p_probe) const { return 0.0; }

	void gi_probe_set_normal_bias(RID p_probe, float p_range) {}
	float gi_probe_get_normal_bias(RID p_probe) const { return 0.0; }

	void gi_probe_set_propagation(RID p_probe, float p_range) {}
	float gi_probe_get_propagation(RID p_probe) const { return 0.0; }

	void gi_probe_set_interior(RID p_probe, bool p_enable) {}
	bool gi_probe_is_interior(RID p_probe) const { return false; }

	void gi_probe_set_compress(RID p_probe, bool p_enable) {}
	bool gi_probe_is_compressed(RID p_probe) const { return false; }

	uint32_t gi_probe_get_version(RID p_probe) { return 0; }

	GIProbeCompression gi_probe_get_dynamic_data_get_preferred_compression() const { return GI_PROBE_UNCOMPRESSED; }
	RID gi_probe_dynamic_data_create(int p_width, int p_height, int p_depth, GIProbeCompression p_compression) { return RID(); }
	void gi_probe_dynamic_data_update(RID p_gi_probe_data, int p_depth_slice, int p_slice_count, int p_mipmap, const void *p_data) {}

	/* LIGHTMAP CAPTURE */

	void lightmap_capture_set_bounds(RID p_capture, const AABB &p_bounds) {}
	AABB lightmap_capture_get_bounds(RID p_capture) const { return AABB(); }
	void lightmap_capture_set_octree(RID p_capture, const PoolVector<uint8_t> &p_octree) {}
	RID lightmap_capture_create() {
		return RID();
	}
	PoolVector<uint8_t> lightmap_capture_get_octree(RID p_capture) const {
		return PoolVector<uint8_t>();
	}
	void lightmap_capture_set_octree_cell_transform(RID p_capture, const Transform &p_xform) {}
	Transform lightmap_capture_get_octree_cell_transform(RID p_capture) const { return Transform(); }
	void lightmap_capture_set_octree_cell_subdiv(RID p_capture, int p_subdiv) {}
	int lightmap_capture_get_octree_cell_subdiv(RID p_capture) const { return 0; }
	void lightmap_capture_set_energy(RID p_capture, float p_energy) {}
	float lightmap_capture_get_energy(RID p_capture) const { return 0.0; }
	const PoolVector<LightmapCaptureOctree> *lightmap_capture_get_octree_ptr(RID p_capture) const {
		return NULL;
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

	void particles_set_draw_order(RID p_particles, VS::ParticlesDrawOrder p_order) {}

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

	/* RENDER TARGET API */

	RID render_target_create();
	void render_target_set_position(RID p_render_target, int p_x, int p_y);
	void render_target_set_size(RID p_render_target, int p_width, int p_height);
	RID render_target_get_texture(RID p_render_target);
	void render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id);
	void render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value);
	bool render_target_was_used(RID p_render_target);
	void render_target_set_as_unused(RID p_render_target);
	void render_target_copy_to_back_buffer(RID p_render_target, const Rect2i &p_region);
	RID render_target_get_back_buffer_uniform_set(RID p_render_target, RID p_base_shader);

	virtual void render_target_request_clear(RID p_render_target, const Color &p_clear_color);
	virtual bool render_target_is_clear_requested(RID p_render_target);
	virtual Color render_target_get_clear_request_color(RID p_render_target);
	virtual void render_target_disable_clear_request(RID p_render_target);
	virtual void render_target_do_clear_request(RID p_render_target);

	Size2 render_target_get_size(RID p_render_target);
	RID render_target_get_rd_framebuffer(RID p_render_target);

	VS::InstanceType get_base_type(RID p_rid) const {

		return VS::INSTANCE_NONE;
	}

	bool free(RID p_rid);

	bool has_os_feature(const String &p_feature) const { return false; }

	void update_dirty_resources();

	void set_debug_generate_wireframes(bool p_generate) {}

	void render_info_begin_capture() {}
	void render_info_end_capture() {}
	int get_captured_render_info(VS::RenderInfo p_info) { return 0; }

	int get_render_info(VS::RenderInfo p_info) { return 0; }

	static RasterizerStorage *base_singleton;

	RasterizerEffectsRD *get_effects();

	RasterizerStorageRD();
	~RasterizerStorageRD();
};

#endif // RASTERIZER_STORAGE_RD_H
