/*************************************************************************/
/*  rasterizer_storage_rd.h                                              */
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

#ifndef RASTERIZER_STORAGE_RD_H
#define RASTERIZER_STORAGE_RD_H

#include "core/rid_owner.h"
#include "servers/rendering/rasterizer.h"
#include "servers/rendering/rasterizer_rd/rasterizer_effects_rd.h"
#include "servers/rendering/rasterizer_rd/shader_compiler_rd.h"
#include "servers/rendering/rasterizer_rd/shaders/giprobe_sdf.glsl.gen.h"
#include "servers/rendering/rasterizer_rd/shaders/particles.glsl.gen.h"
#include "servers/rendering/rasterizer_rd/shaders/particles_copy.glsl.gen.h"
#include "servers/rendering/rendering_device.h"

class RasterizerStorageRD : public RasterizerStorage {
public:
	static _FORCE_INLINE_ void store_transform(const Transform &p_mtx, float *p_array) {
		p_array[0] = p_mtx.basis.elements[0][0];
		p_array[1] = p_mtx.basis.elements[1][0];
		p_array[2] = p_mtx.basis.elements[2][0];
		p_array[3] = 0;
		p_array[4] = p_mtx.basis.elements[0][1];
		p_array[5] = p_mtx.basis.elements[1][1];
		p_array[6] = p_mtx.basis.elements[2][1];
		p_array[7] = 0;
		p_array[8] = p_mtx.basis.elements[0][2];
		p_array[9] = p_mtx.basis.elements[1][2];
		p_array[10] = p_mtx.basis.elements[2][2];
		p_array[11] = 0;
		p_array[12] = p_mtx.origin.x;
		p_array[13] = p_mtx.origin.y;
		p_array[14] = p_mtx.origin.z;
		p_array[15] = 1;
	}

	static _FORCE_INLINE_ void store_basis_3x4(const Basis &p_mtx, float *p_array) {
		p_array[0] = p_mtx.elements[0][0];
		p_array[1] = p_mtx.elements[1][0];
		p_array[2] = p_mtx.elements[2][0];
		p_array[3] = 0;
		p_array[4] = p_mtx.elements[0][1];
		p_array[5] = p_mtx.elements[1][1];
		p_array[6] = p_mtx.elements[2][1];
		p_array[7] = 0;
		p_array[8] = p_mtx.elements[0][2];
		p_array[9] = p_mtx.elements[1][2];
		p_array[10] = p_mtx.elements[2][2];
		p_array[11] = 0;
	}

	static _FORCE_INLINE_ void store_transform_3x3(const Basis &p_mtx, float *p_array) {
		p_array[0] = p_mtx.elements[0][0];
		p_array[1] = p_mtx.elements[1][0];
		p_array[2] = p_mtx.elements[2][0];
		p_array[3] = 0;
		p_array[4] = p_mtx.elements[0][1];
		p_array[5] = p_mtx.elements[1][1];
		p_array[6] = p_mtx.elements[2][1];
		p_array[7] = 0;
		p_array[8] = p_mtx.elements[0][2];
		p_array[9] = p_mtx.elements[1][2];
		p_array[10] = p_mtx.elements[2][2];
		p_array[11] = 0;
	}

	static _FORCE_INLINE_ void store_camera(const CameraMatrix &p_mtx, float *p_array) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				p_array[i * 4 + j] = p_mtx.matrix[i][j];
			}
		}
	}

	static _FORCE_INLINE_ void store_soft_shadow_kernel(const float *p_kernel, float *p_array) {
		for (int i = 0; i < 128; i++) {
			p_array[i] = p_kernel[i];
		}
	}

	enum ShaderType {
		SHADER_TYPE_2D,
		SHADER_TYPE_3D,
		SHADER_TYPE_PARTICLES,
		SHADER_TYPE_SKY,
		SHADER_TYPE_MAX
	};

	struct ShaderData {
		virtual void set_code(const String &p_Code) = 0;
		virtual void set_default_texture_param(const StringName &p_name, RID p_texture) = 0;
		virtual void get_param_list(List<PropertyInfo> *p_param_list) const = 0;

		virtual void get_instance_param_list(List<InstanceShaderParam> *p_param_list) const = 0;
		virtual bool is_param_texture(const StringName &p_param) const = 0;
		virtual bool is_animated() const = 0;
		virtual bool casts_shadows() const = 0;
		virtual Variant get_default_parameter(const StringName &p_parameter) const = 0;
		virtual ~ShaderData() {}
	};

	typedef ShaderData *(*ShaderDataRequestFunction)();

	struct MaterialData {
		void update_uniform_buffer(const Map<StringName, ShaderLanguage::ShaderNode::Uniform> &p_uniforms, const uint32_t *p_uniform_offsets, const Map<StringName, Variant> &p_parameters, uint8_t *p_buffer, uint32_t p_buffer_size, bool p_use_linear_color);
		void update_textures(const Map<StringName, Variant> &p_parameters, const Map<StringName, RID> &p_default_textures, const Vector<ShaderCompilerRD::GeneratedCode::Texture> &p_texture_uniforms, RID *p_textures, bool p_use_linear_color);

		virtual void set_render_priority(int p_priority) = 0;
		virtual void set_next_pass(RID p_pass) = 0;
		virtual void update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) = 0;
		virtual ~MaterialData();

	private:
		friend class RasterizerStorageRD;
		RID self;
		List<RID>::Element *global_buffer_E = nullptr;
		List<RID>::Element *global_texture_E = nullptr;
		uint64_t global_textures_pass = 0;
		Map<StringName, uint64_t> used_global_textures;
	};
	typedef MaterialData *(*MaterialDataRequestFunction)(ShaderData *);

	enum DefaultRDTexture {
		DEFAULT_RD_TEXTURE_WHITE,
		DEFAULT_RD_TEXTURE_BLACK,
		DEFAULT_RD_TEXTURE_NORMAL,
		DEFAULT_RD_TEXTURE_ANISO,
		DEFAULT_RD_TEXTURE_MULTIMESH_BUFFER,
		DEFAULT_RD_TEXTURE_CUBEMAP_BLACK,
		DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK,
		DEFAULT_RD_TEXTURE_3D_WHITE,
		DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE,
		DEFAULT_RD_TEXTURE_2D_UINT,
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
	/* CANVAS TEXTURE API (2D) */

	struct CanvasTexture {
		RID diffuse;
		RID normalmap;
		RID specular;
		Color specular_color = Color(1, 1, 1, 1);
		float shininess = 1.0;

		RS::CanvasItemTextureFilter texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT;
		RS::CanvasItemTextureRepeat texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT;
		RID uniform_sets[RS::CANVAS_ITEM_TEXTURE_FILTER_MAX][RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX];

		Size2i size_cache = Size2i(1, 1);
		bool use_normal_cache = false;
		bool use_specular_cache = false;
		bool cleared_cache = true;
		void clear_sets();
		~CanvasTexture();
	};

	RID_PtrOwner<CanvasTexture> canvas_texture_owner;

	/* TEXTURE API */
	struct Texture {
		enum Type {
			TYPE_2D,
			TYPE_LAYERED,
			TYPE_3D
		};

		Type type;
		RS::TextureLayeredType layered_type = RS::TEXTURE_LAYERED_2D_ARRAY;

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

		struct BufferSlice3D {
			Size2i size;
			uint32_t offset = 0;
			uint32_t buffer_size = 0;
		};
		Vector<BufferSlice3D> buffer_slices_3d;
		uint32_t buffer_size_3d = 0;

		bool is_render_target;
		bool is_proxy;

		Ref<Image> image_cache_2d;
		String path;

		RID proxy_to;
		Vector<RID> proxies;
		Set<RID> lightmap_users;

		RS::TextureDetectCallback detect_3d_callback = nullptr;
		void *detect_3d_callback_ud = nullptr;

		RS::TextureDetectCallback detect_normal_callback = nullptr;
		void *detect_normal_callback_ud = nullptr;

		RS::TextureDetectRoughnessCallback detect_roughness_callback = nullptr;
		void *detect_roughness_callback_ud = nullptr;

		CanvasTexture *canvas_texture = nullptr;
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
	RID default_rd_samplers[RS::CANVAS_ITEM_TEXTURE_FILTER_MAX][RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX];
	RID default_rd_storage_buffer;

	/* DECAL ATLAS */

	struct DecalAtlas {
		struct Texture {
			int panorama_to_dp_users;
			int users;
			Rect2 uv_rect;
		};

		struct SortItem {
			RID texture;
			Size2i pixel_size;
			Size2i size;
			Point2i pos;

			bool operator<(const SortItem &p_item) const {
				//sort larger to smaller
				if (size.height == p_item.size.height) {
					return size.width > p_item.size.width;
				} else {
					return size.height > p_item.size.height;
				}
			}
		};

		HashMap<RID, Texture> textures;
		bool dirty = true;
		int mipmaps = 5;

		RID texture;
		RID texture_srgb;
		struct MipMap {
			RID fb;
			RID texture;
			Size2i size;
		};
		Vector<MipMap> texture_mipmaps;

		Size2i size;

	} decal_atlas;

	void _update_decal_atlas();

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
			RS::PrimitiveType primitive = RS::PRIMITIVE_POINTS;
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
				uint32_t input_mask = 0;
				RD::VertexFormatID vertex_format = 0;
				RID vertex_array;
			};

			SpinLock version_lock; //needed to access versions
			Version *versions = nullptr; //allocated on demand
			uint32_t version_count = 0;

			RID index_buffer;
			RID index_array;
			uint32_t index_count = 0;

			struct LOD {
				float edge_length = 0.0;
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

			uint32_t render_index = 0;
			uint64_t render_pass = 0;

			uint32_t multimesh_render_index = 0;
			uint64_t multimesh_render_pass = 0;

			uint32_t particles_render_index = 0;
			uint64_t particles_render_pass = 0;
		};

		uint32_t blend_shape_count = 0;
		RS::BlendShapeMode blend_shape_mode = RS::BLEND_SHAPE_MODE_NORMALIZED;

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

	/* MultiMesh */
	struct MultiMesh {
		RID mesh;
		int instances = 0;
		RS::MultimeshTransformFormat xform_format = RS::MULTIMESH_TRANSFORM_3D;
		bool uses_colors = false;
		bool uses_custom_data = false;
		int visible_instances = -1;
		AABB aabb;
		bool aabb_dirty = false;
		bool buffer_set = false;
		uint32_t stride_cache = 0;
		uint32_t color_offset_cache = 0;
		uint32_t custom_data_offset_cache = 0;

		Vector<float> data_cache; //used if individual setting is used
		bool *data_cache_dirty_regions = nullptr;
		uint32_t data_cache_used_dirty_regions = 0;

		RID buffer; //storage buffer
		RID uniform_set_3d;

		bool dirty = false;
		MultiMesh *dirty_list = nullptr;

		RasterizerScene::InstanceDependency instance_dependency;
	};

	mutable RID_Owner<MultiMesh> multimesh_owner;

	MultiMesh *multimesh_dirty_list = nullptr;

	_FORCE_INLINE_ void _multimesh_make_local(MultiMesh *multimesh) const;
	_FORCE_INLINE_ void _multimesh_mark_dirty(MultiMesh *multimesh, int p_index, bool p_aabb);
	_FORCE_INLINE_ void _multimesh_mark_all_dirty(MultiMesh *multimesh, bool p_data, bool p_aabb);
	_FORCE_INLINE_ void _multimesh_re_create_aabb(MultiMesh *multimesh, const float *p_data, int p_instances);
	void _update_dirty_multimeshes();

	/* PARTICLES */

	struct ParticleData {
		float xform[16];
		float velocity[3];
		uint32_t active;
		float color[4];
		float custom[3];
		float lifetime;
		uint32_t pad[3];
	};

	struct ParticlesFrameParams {
		enum {
			MAX_ATTRACTORS = 32,
			MAX_COLLIDERS = 32,
			MAX_3D_TEXTURES = 7
		};

		enum AttractorType {
			ATTRACTOR_TYPE_SPHERE,
			ATTRACTOR_TYPE_BOX,
			ATTRACTOR_TYPE_VECTOR_FIELD,
		};

		struct Attractor {
			float transform[16];
			float extents[3]; //exents or radius
			uint32_t type;

			uint32_t texture_index; //texture index for vector field
			float strength;
			float attenuation;
			float directionality;
		};

		enum CollisionType {
			COLLISION_TYPE_SPHERE,
			COLLISION_TYPE_BOX,
			COLLISION_TYPE_SDF,
			COLLISION_TYPE_HEIGHT_FIELD
		};

		struct Collider {
			float transform[16];
			float extents[3]; //exents or radius
			uint32_t type;

			uint32_t texture_index; //texture index for vector field
			float scale;
			uint32_t pad[2];
		};

		uint32_t emitting;
		float system_phase;
		float prev_system_phase;
		uint32_t cycle;

		float explosiveness;
		float randomness;
		float time;
		float delta;

		uint32_t random_seed;
		uint32_t attractor_count;
		uint32_t collider_count;
		float particle_size;

		float emission_transform[16];

		Attractor attractors[MAX_ATTRACTORS];
		Collider colliders[MAX_COLLIDERS];
	};

	struct ParticleEmissionBufferData {
	};

	struct ParticleEmissionBuffer {
		struct Data {
			float xform[16];
			float velocity[3];
			uint32_t flags;
			float color[4];
			float custom[4];
		};

		int32_t particle_count;
		int32_t particle_max;
		uint32_t pad1;
		uint32_t pad2;
		Data data[1]; //its 2020 and empty arrays are still non standard in C++
	};

	struct Particles {
		bool inactive;
		float inactive_time;
		bool emitting;
		bool one_shot;
		int amount;
		float lifetime;
		float pre_process_time;
		float explosiveness;
		float randomness;
		bool restart_request;
		AABB custom_aabb;
		bool use_local_coords;
		RID process_material;

		RS::ParticlesDrawOrder draw_order;

		Vector<RID> draw_passes;

		RID particle_buffer;
		RID particle_instance_buffer;
		RID frame_params_buffer;

		RID particles_material_uniform_set;
		RID particles_copy_uniform_set;
		RID particles_transforms_buffer_uniform_set;
		RID collision_textures_uniform_set;

		RID collision_3d_textures[ParticlesFrameParams::MAX_3D_TEXTURES];
		uint32_t collision_3d_textures_used = 0;
		RID collision_heightmap_texture;

		RID particles_sort_buffer;
		RID particles_sort_uniform_set;

		bool dirty = false;
		Particles *update_list = nullptr;

		RID sub_emitter;

		float phase;
		float prev_phase;
		uint64_t prev_ticks;
		uint32_t random_seed;

		uint32_t cycle_number;

		float speed_scale;

		int fixed_fps;
		bool fractional_delta;
		float frame_remainder;
		float collision_base_size;

		bool clear;

		bool force_sub_emit = false;

		Transform emission_transform;

		Vector<uint8_t> emission_buffer_data;

		ParticleEmissionBuffer *emission_buffer = nullptr;
		RID emission_storage_buffer;

		Set<RasterizerScene::InstanceBase *> collisions;

		Particles() :
				inactive(true),
				inactive_time(0.0),
				emitting(false),
				one_shot(false),
				amount(0),
				lifetime(1.0),
				pre_process_time(0.0),
				explosiveness(0.0),
				randomness(0.0),
				restart_request(false),
				custom_aabb(AABB(Vector3(-4, -4, -4), Vector3(8, 8, 8))),
				use_local_coords(true),
				draw_order(RS::PARTICLES_DRAW_ORDER_INDEX),
				prev_ticks(0),
				random_seed(0),
				cycle_number(0),
				speed_scale(1.0),
				fixed_fps(0),
				fractional_delta(false),
				frame_remainder(0),
				collision_base_size(0.01),
				clear(true) {
		}

		RasterizerScene::InstanceDependency instance_dependency;

		ParticlesFrameParams frame_params;
	};

	void _particles_process(Particles *p_particles, float p_delta);
	void _particles_allocate_emission_buffer(Particles *particles);
	void _particles_free_data(Particles *particles);

	struct ParticlesShader {
		struct PushConstant {
			float lifetime;
			uint32_t clear;
			uint32_t total_particles;
			uint32_t trail_size;

			uint32_t use_fractional_delta;
			uint32_t sub_emitter_mode;
			uint32_t can_emit;
			uint32_t pad;
		};

		ParticlesShaderRD shader;
		ShaderCompilerRD compiler;

		RID default_shader;
		RID default_material;
		RID default_shader_rd;

		RID base_uniform_set;

		struct CopyPushConstant {
			float sort_direction[3];
			uint32_t total_particles;
		};

		enum {
			COPY_MODE_FILL_INSTANCES,
			COPY_MODE_FILL_SORT_BUFFER,
			COPY_MODE_FILL_INSTANCES_WITH_SORT_BUFFER,
			COPY_MODE_MAX,
		};

		ParticlesCopyShaderRD copy_shader;
		RID copy_shader_version;
		RID copy_pipelines[COPY_MODE_MAX];

	} particles_shader;

	Particles *particle_update_list = nullptr;

	struct ParticlesShaderData : public ShaderData {
		bool valid;
		RID version;

		//RenderPipelineVertexFormatCacheRD pipelines[SKY_VERSION_MAX];
		Map<StringName, ShaderLanguage::ShaderNode::Uniform> uniforms;
		Vector<ShaderCompilerRD::GeneratedCode::Texture> texture_uniforms;

		Vector<uint32_t> ubo_offsets;
		uint32_t ubo_size;

		String path;
		String code;
		Map<StringName, RID> default_texture_params;

		RID pipeline;

		bool uses_time;

		virtual void set_code(const String &p_Code);
		virtual void set_default_texture_param(const StringName &p_name, RID p_texture);
		virtual void get_param_list(List<PropertyInfo> *p_param_list) const;
		virtual void get_instance_param_list(List<RasterizerStorage::InstanceShaderParam> *p_param_list) const;
		virtual bool is_param_texture(const StringName &p_param) const;
		virtual bool is_animated() const;
		virtual bool casts_shadows() const;
		virtual Variant get_default_parameter(const StringName &p_parameter) const;
		ParticlesShaderData();
		virtual ~ParticlesShaderData();
	};

	ShaderData *_create_particles_shader_func();
	static RasterizerStorageRD::ShaderData *_create_particles_shader_funcs() {
		return base_singleton->_create_particles_shader_func();
	}

	struct ParticlesMaterialData : public MaterialData {
		uint64_t last_frame;
		ParticlesShaderData *shader_data;
		RID uniform_buffer;
		RID uniform_set;
		Vector<RID> texture_cache;
		Vector<uint8_t> ubo_data;
		bool uniform_set_updated;

		virtual void set_render_priority(int p_priority) {}
		virtual void set_next_pass(RID p_pass) {}
		virtual void update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty);
		virtual ~ParticlesMaterialData();
	};

	MaterialData *_create_particles_material_func(ParticlesShaderData *p_shader);
	static RasterizerStorageRD::MaterialData *_create_particles_material_funcs(ShaderData *p_shader) {
		return base_singleton->_create_particles_material_func(static_cast<ParticlesShaderData *>(p_shader));
	}

	void update_particles();

	mutable RID_Owner<Particles> particles_owner;

	/* Particles Collision */

	struct ParticlesCollision {
		RS::ParticlesCollisionType type = RS::PARTICLES_COLLISION_TYPE_SPHERE_ATTRACT;
		uint32_t cull_mask = 0xFFFFFFFF;
		float radius = 1.0;
		Vector3 extents = Vector3(1, 1, 1);
		float attractor_strength = 1.0;
		float attractor_attenuation = 1.0;
		float attractor_directionality = 0.0;
		RID field_texture;
		RID heightfield_texture;
		RID heightfield_fb;
		Size2i heightfield_fb_size;

		RS::ParticlesCollisionHeightfieldResolution heightfield_resolution = RS::PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_1024;

		RasterizerScene::InstanceDependency instance_dependency;
	};

	mutable RID_Owner<ParticlesCollision> particles_collision_owner;

	/* Skeleton */

	struct Skeleton {
		bool use_2d = false;
		int size = 0;
		Vector<float> data;
		RID buffer;

		bool dirty = false;
		Skeleton *dirty_list = nullptr;
		Transform2D base_transform_2d;

		RID uniform_set_3d;

		RasterizerScene::InstanceDependency instance_dependency;
	};

	mutable RID_Owner<Skeleton> skeleton_owner;

	_FORCE_INLINE_ void _skeleton_make_dirty(Skeleton *skeleton);

	Skeleton *skeleton_dirty_list = nullptr;

	void _update_dirty_skeletons();

	/* LIGHT */

	struct Light {
		RS::LightType type;
		float param[RS::LIGHT_PARAM_MAX];
		Color color = Color(1, 1, 1, 1);
		Color shadow_color;
		RID projector;
		bool shadow = false;
		bool negative = false;
		bool reverse_cull = false;
		RS::LightBakeMode bake_mode = RS::LIGHT_BAKE_DYNAMIC;
		uint32_t max_sdfgi_cascade = 2;
		uint32_t cull_mask = 0xFFFFFFFF;
		RS::LightOmniShadowMode omni_shadow_mode = RS::LIGHT_OMNI_SHADOW_DUAL_PARABOLOID;
		RS::LightDirectionalShadowMode directional_shadow_mode = RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL;
		RS::LightDirectionalShadowDepthRangeMode directional_range_mode = RS::LIGHT_DIRECTIONAL_SHADOW_DEPTH_RANGE_STABLE;
		bool directional_blend_splits = false;
		uint64_t version = 0;

		RasterizerScene::InstanceDependency instance_dependency;
	};

	mutable RID_Owner<Light> light_owner;

	/* REFLECTION PROBE */

	struct ReflectionProbe {
		RS::ReflectionProbeUpdateMode update_mode = RS::REFLECTION_PROBE_UPDATE_ONCE;
		int resolution = 256;
		float intensity = 1.0;
		RS::ReflectionProbeAmbientMode ambient_mode = RS::REFLECTION_PROBE_AMBIENT_ENVIRONMENT;
		Color ambient_color;
		float ambient_color_energy = 1.0;
		float max_distance = 0;
		Vector3 extents = Vector3(1, 1, 1);
		Vector3 origin_offset;
		bool interior = false;
		bool box_projection = false;
		bool enable_shadows = false;
		uint32_t cull_mask = (1 << 20) - 1;

		RasterizerScene::InstanceDependency instance_dependency;
	};

	mutable RID_Owner<ReflectionProbe> reflection_probe_owner;

	/* DECAL */

	struct Decal {
		Vector3 extents = Vector3(1, 1, 1);
		RID textures[RS::DECAL_TEXTURE_MAX];
		float emission_energy = 1.0;
		float albedo_mix = 1.0;
		Color modulate = Color(1, 1, 1, 1);
		uint32_t cull_mask = (1 << 20) - 1;
		float upper_fade = 0.3;
		float lower_fade = 0.3;
		bool distance_fade = false;
		float distance_fade_begin = 10;
		float distance_fade_length = 1;
		float normal_fade = 0.0;

		RasterizerScene::InstanceDependency instance_dependency;
	};

	mutable RID_Owner<Decal> decal_owner;

	/* GI PROBE */

	struct GIProbe {
		RID octree_buffer;
		RID data_buffer;
		RID sdf_texture;

		uint32_t octree_buffer_size = 0;
		uint32_t data_buffer_size = 0;

		Vector<int> level_counts;

		int cell_count = 0;

		Transform to_cell_xform;
		AABB bounds;
		Vector3i octree_size;

		float dynamic_range = 4.0;
		float energy = 1.0;
		float ao = 0.0;
		float ao_size = 0.5;
		float bias = 1.4;
		float normal_bias = 0.0;
		float propagation = 0.7;
		bool interior = false;
		bool use_two_bounces = false;

		float anisotropy_strength = 0.5;

		uint32_t version = 1;
		uint32_t data_version = 1;

		RasterizerScene::InstanceDependency instance_dependency;
	};

	GiprobeSdfShaderRD giprobe_sdf_shader;
	RID giprobe_sdf_shader_version;
	RID giprobe_sdf_shader_version_shader;
	RID giprobe_sdf_shader_pipeline;

	mutable RID_Owner<GIProbe> gi_probe_owner;

	/* REFLECTION PROBE */

	struct Lightmap {
		RID light_texture;
		bool uses_spherical_harmonics = false;
		bool interior = false;
		AABB bounds = AABB(Vector3(), Vector3(1, 1, 1));
		int32_t array_index = -1; //unassigned
		PackedVector3Array points;
		PackedColorArray point_sh;
		PackedInt32Array tetrahedra;
		PackedInt32Array bsp_tree;

		struct BSP {
			static const int32_t EMPTY_LEAF = INT32_MIN;
			float plane[4];
			int32_t over = EMPTY_LEAF, under = EMPTY_LEAF;
		};

		RasterizerScene::InstanceDependency instance_dependency;
	};

	bool using_lightmap_array; //high end uses this
	/* for high end */

	Vector<RID> lightmap_textures;

	uint64_t lightmap_array_version = 0;

	mutable RID_Owner<Lightmap> lightmap_owner;

	float lightmap_probe_capture_update_speed = 4;

	/* RENDER TARGET */

	struct RenderTarget {
		Size2i size;
		RID framebuffer;
		RID color;

		//used for retrieving from CPU
		RD::DataFormat color_format = RD::DATA_FORMAT_R4G4_UNORM_PACK8;
		RD::DataFormat color_format_srgb = RD::DATA_FORMAT_R4G4_UNORM_PACK8;
		Image::Format image_format = Image::FORMAT_L8;

		bool flags[RENDER_TARGET_FLAG_MAX];

		RID backbuffer; //used for effects
		RID backbuffer_fb;
		RID backbuffer_mipmap0;

		struct BackbufferMipmap {
			RID mipmap;
			RID mipmap_copy;
		};

		Vector<BackbufferMipmap> backbuffer_mipmaps;

		RID framebuffer_uniform_set;
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

	/* GLOBAL SHADER VARIABLES */

	struct GlobalVariables {
		enum {
			BUFFER_DIRTY_REGION_SIZE = 1024
		};
		struct Variable {
			Set<RID> texture_materials; // materials using this

			RS::GlobalVariableType type;
			Variant value;
			Variant override;
			int32_t buffer_index; //for vectors
			int32_t buffer_elements; //for vectors
		};

		HashMap<StringName, Variable> variables;

		struct Value {
			float x;
			float y;
			float z;
			float w;
		};

		struct ValueInt {
			int32_t x;
			int32_t y;
			int32_t z;
			int32_t w;
		};

		struct ValueUInt {
			uint32_t x;
			uint32_t y;
			uint32_t z;
			uint32_t w;
		};

		struct ValueUsage {
			uint32_t elements = 0;
		};

		List<RID> materials_using_buffer;
		List<RID> materials_using_texture;

		RID buffer;
		Value *buffer_values;
		ValueUsage *buffer_usage;
		bool *buffer_dirty_regions;
		uint32_t buffer_dirty_region_count = 0;

		uint32_t buffer_size;

		bool must_update_texture_materials = false;
		bool must_update_buffer_materials = false;

		HashMap<RID, int32_t> instance_buffer_pos;

	} global_variables;

	int32_t _global_variable_allocate(uint32_t p_elements);
	void _global_variable_store_in_buffer(int32_t p_index, RS::GlobalVariableType p_type, const Variant &p_value);
	void _global_variable_mark_buffer_dirty(int32_t p_index, int32_t p_elements);

	void _update_global_variables();
	/* EFFECTS */

	RasterizerEffectsRD effects;

public:
	/* TEXTURE API */

	virtual RID texture_2d_create(const Ref<Image> &p_image);
	virtual RID texture_2d_layered_create(const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type);
	virtual RID texture_3d_create(Image::Format p_format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data); //all slices, then all the mipmaps, must be coherent
	virtual RID texture_proxy_create(RID p_base);

	virtual void _texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer, bool p_immediate);

	virtual void texture_2d_update_immediate(RID p_texture, const Ref<Image> &p_image, int p_layer = 0); //mostly used for video and streaming
	virtual void texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0);
	virtual void texture_3d_update(RID p_texture, const Vector<Ref<Image>> &p_data);
	virtual void texture_proxy_update(RID p_texture, RID p_proxy_to);

	//these two APIs can be used together or in combination with the others.
	virtual RID texture_2d_placeholder_create();
	virtual RID texture_2d_layered_placeholder_create(RenderingServer::TextureLayeredType p_layered_type);
	virtual RID texture_3d_placeholder_create();

	virtual Ref<Image> texture_2d_get(RID p_texture) const;
	virtual Ref<Image> texture_2d_layer_get(RID p_texture, int p_layer) const;
	virtual Vector<Ref<Image>> texture_3d_get(RID p_texture) const;

	virtual void texture_replace(RID p_texture, RID p_by_texture);
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height);

	virtual void texture_set_path(RID p_texture, const String &p_path);
	virtual String texture_get_path(RID p_texture) const;

	virtual void texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata);
	virtual void texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata);
	virtual void texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata);

	virtual void texture_debug_usage(List<RS::TextureInfo> *r_info);

	virtual void texture_set_proxy(RID p_proxy, RID p_base);
	virtual void texture_set_force_redraw_if_visible(RID p_texture, bool p_enable);

	virtual Size2 texture_size_with_proxy(RID p_proxy);

	virtual void texture_add_to_decal_atlas(RID p_texture, bool p_panorama_to_dp = false);
	virtual void texture_remove_from_decal_atlas(RID p_texture, bool p_panorama_to_dp = false);

	RID decal_atlas_get_texture() const;
	RID decal_atlas_get_texture_srgb() const;
	_FORCE_INLINE_ Rect2 decal_atlas_get_texture_rect(RID p_texture) {
		DecalAtlas::Texture *t = decal_atlas.textures.getptr(p_texture);
		if (!t) {
			return Rect2();
		}

		return t->uv_rect;
	}

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
	_FORCE_INLINE_ RID sampler_rd_get_default(RS::CanvasItemTextureFilter p_filter, RS::CanvasItemTextureRepeat p_repeat) {
		return default_rd_samplers[p_filter][p_repeat];
	}

	/* CANVAS TEXTURE API */

	virtual RID canvas_texture_create();

	virtual void canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture);
	virtual void canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_specular_color, float p_shininess);

	virtual void canvas_texture_set_texture_filter(RID p_canvas_texture, RS::CanvasItemTextureFilter p_filter);
	virtual void canvas_texture_set_texture_repeat(RID p_canvas_texture, RS::CanvasItemTextureRepeat p_repeat);

	bool canvas_texture_get_unifom_set(RID p_texture, RS::CanvasItemTextureFilter p_base_filter, RS::CanvasItemTextureRepeat p_base_repeat, RID p_base_shader, int p_base_set, RID &r_uniform_set, Size2i &r_size, Color &r_specular_shininess, bool &r_use_normal, bool &r_use_specular);

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

	void material_get_instance_shader_parameters(RID p_material, List<InstanceShaderParam> *r_parameters);

	void material_update_dependency(RID p_material, RasterizerScene::InstanceBase *p_instance);
	void material_force_update_textures(RID p_material, ShaderType p_shader_type);

	void material_set_data_request_function(ShaderType p_shader_type, MaterialDataRequestFunction p_function);

	_FORCE_INLINE_ MaterialData *material_get_data(RID p_material, ShaderType p_shader_type) {
		Material *material = material_owner.getornull(p_material);
		if (!material || material->shader_type != p_shader_type) {
			return nullptr;
		} else {
			return material->data;
		}
	}

	/* MESH API */

	virtual RID mesh_create();

	/// Return stride
	virtual void mesh_add_surface(RID p_mesh, const RS::SurfaceData &p_surface);

	virtual int mesh_get_blend_shape_count(RID p_mesh) const;

	virtual void mesh_set_blend_shape_mode(RID p_mesh, RS::BlendShapeMode p_mode);
	virtual RS::BlendShapeMode mesh_get_blend_shape_mode(RID p_mesh) const;

	virtual void mesh_surface_update_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data);

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material);
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const;

	virtual RS::SurfaceData mesh_get_surface(RID p_mesh, int p_surface) const;

	virtual int mesh_get_surface_count(RID p_mesh) const;

	virtual void mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb);
	virtual AABB mesh_get_custom_aabb(RID p_mesh) const;

	virtual AABB mesh_get_aabb(RID p_mesh, RID p_skeleton = RID());

	virtual void mesh_clear(RID p_mesh);

	_FORCE_INLINE_ const RID *mesh_get_surface_count_and_materials(RID p_mesh, uint32_t &r_surface_count) {
		Mesh *mesh = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!mesh, nullptr);
		r_surface_count = mesh->surface_count;
		if (r_surface_count == 0) {
			return nullptr;
		}
		if (mesh->material_cache.empty()) {
			mesh->material_cache.resize(mesh->surface_count);
			for (uint32_t i = 0; i < r_surface_count; i++) {
				mesh->material_cache.write[i] = mesh->surfaces[i]->material;
			}
		}

		return mesh->material_cache.ptr();
	}

	_FORCE_INLINE_ RS::PrimitiveType mesh_surface_get_primitive(RID p_mesh, uint32_t p_surface_index) {
		Mesh *mesh = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND_V(!mesh, RS::PRIMITIVE_MAX);
		ERR_FAIL_UNSIGNED_INDEX_V(p_surface_index, mesh->surface_count, RS::PRIMITIVE_MAX);

		return mesh->surfaces[p_surface_index]->primitive;
	}

	_FORCE_INLINE_ void mesh_surface_get_arrays_and_format(RID p_mesh, uint32_t p_surface_index, uint32_t p_input_mask, RID &r_vertex_array_rd, RID &r_index_array_rd, RD::VertexFormatID &r_vertex_format) {
		Mesh *mesh = mesh_owner.getornull(p_mesh);
		ERR_FAIL_COND(!mesh);
		ERR_FAIL_UNSIGNED_INDEX(p_surface_index, mesh->surface_count);

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

	_FORCE_INLINE_ uint32_t mesh_surface_get_render_pass_index(RID p_mesh, uint32_t p_surface_index, uint64_t p_render_pass, uint32_t *r_index) {
		Mesh *mesh = mesh_owner.getornull(p_mesh);
		Mesh::Surface *s = mesh->surfaces[p_surface_index];

		if (s->render_pass != p_render_pass) {
			(*r_index)++;
			s->render_pass = p_render_pass;
			s->render_index = *r_index;
		}

		return s->render_index;
	}

	_FORCE_INLINE_ uint32_t mesh_surface_get_multimesh_render_pass_index(RID p_mesh, uint32_t p_surface_index, uint64_t p_render_pass, uint32_t *r_index) {
		Mesh *mesh = mesh_owner.getornull(p_mesh);
		Mesh::Surface *s = mesh->surfaces[p_surface_index];

		if (s->multimesh_render_pass != p_render_pass) {
			(*r_index)++;
			s->multimesh_render_pass = p_render_pass;
			s->multimesh_render_index = *r_index;
		}

		return s->multimesh_render_index;
	}

	_FORCE_INLINE_ uint32_t mesh_surface_get_particles_render_pass_index(RID p_mesh, uint32_t p_surface_index, uint64_t p_render_pass, uint32_t *r_index) {
		Mesh *mesh = mesh_owner.getornull(p_mesh);
		Mesh::Surface *s = mesh->surfaces[p_surface_index];

		if (s->particles_render_pass != p_render_pass) {
			(*r_index)++;
			s->particles_render_pass = p_render_pass;
			s->particles_render_index = *r_index;
		}

		return s->particles_render_index;
	}

	/* MULTIMESH API */

	RID multimesh_create();

	void multimesh_allocate(RID p_multimesh, int p_instances, RS::MultimeshTransformFormat p_transform_format, bool p_use_colors = false, bool p_use_custom_data = false);
	int multimesh_get_instance_count(RID p_multimesh) const;

	void multimesh_set_mesh(RID p_multimesh, RID p_mesh);
	void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform);
	void multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform);
	void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color);
	void multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color);

	RID multimesh_get_mesh(RID p_multimesh) const;

	Transform multimesh_instance_get_transform(RID p_multimesh, int p_index) const;
	Transform2D multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const;
	Color multimesh_instance_get_color(RID p_multimesh, int p_index) const;
	Color multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const;

	void multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer);
	Vector<float> multimesh_get_buffer(RID p_multimesh) const;

	void multimesh_set_visible_instances(RID p_multimesh, int p_visible);
	int multimesh_get_visible_instances(RID p_multimesh) const;

	AABB multimesh_get_aabb(RID p_multimesh) const;

	_FORCE_INLINE_ RS::MultimeshTransformFormat multimesh_get_transform_format(RID p_multimesh) const {
		MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
		return multimesh->xform_format;
	}

	_FORCE_INLINE_ bool multimesh_uses_colors(RID p_multimesh) const {
		MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
		return multimesh->uses_colors;
	}

	_FORCE_INLINE_ bool multimesh_uses_custom_data(RID p_multimesh) const {
		MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
		return multimesh->uses_custom_data;
	}

	_FORCE_INLINE_ uint32_t multimesh_get_instances_to_draw(RID p_multimesh) const {
		MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
		if (multimesh->visible_instances >= 0) {
			return multimesh->visible_instances;
		}
		return multimesh->instances;
	}

	_FORCE_INLINE_ RID multimesh_get_3d_uniform_set(RID p_multimesh, RID p_shader, uint32_t p_set) const {
		MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
		if (!multimesh->uniform_set_3d.is_valid()) {
			Vector<RD::Uniform> uniforms;
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 0;
			u.ids.push_back(multimesh->buffer);
			uniforms.push_back(u);
			multimesh->uniform_set_3d = RD::get_singleton()->uniform_set_create(uniforms, p_shader, p_set);
		}

		return multimesh->uniform_set_3d;
	}

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

	RID skeleton_create();
	void skeleton_allocate(RID p_skeleton, int p_bones, bool p_2d_skeleton = false);
	void skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform);
	void skeleton_set_world_transform(RID p_skeleton, bool p_enable, const Transform &p_world_transform);
	int skeleton_get_bone_count(RID p_skeleton) const;
	void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform);
	Transform skeleton_bone_get_transform(RID p_skeleton, int p_bone) const;
	void skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform);
	Transform2D skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const;

	_FORCE_INLINE_ RID skeleton_get_3d_uniform_set(RID p_skeleton, RID p_shader, uint32_t p_set) const {
		Skeleton *skeleton = skeleton_owner.getornull(p_skeleton);
		ERR_FAIL_COND_V(!skeleton, RID());
		ERR_FAIL_COND_V(skeleton->size == 0, RID());
		if (skeleton->use_2d) {
			return RID();
		}
		if (!skeleton->uniform_set_3d.is_valid()) {
			Vector<RD::Uniform> uniforms;
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 0;
			u.ids.push_back(skeleton->buffer);
			uniforms.push_back(u);
			skeleton->uniform_set_3d = RD::get_singleton()->uniform_set_create(uniforms, p_shader, p_set);
		}

		return skeleton->uniform_set_3d;
	}
	/* Light API */

	RID light_create(RS::LightType p_type);

	RID directional_light_create() { return light_create(RS::LIGHT_DIRECTIONAL); }
	RID omni_light_create() { return light_create(RS::LIGHT_OMNI); }
	RID spot_light_create() { return light_create(RS::LIGHT_SPOT); }

	void light_set_color(RID p_light, const Color &p_color);
	void light_set_param(RID p_light, RS::LightParam p_param, float p_value);
	void light_set_shadow(RID p_light, bool p_enabled);
	void light_set_shadow_color(RID p_light, const Color &p_color);
	void light_set_projector(RID p_light, RID p_texture);
	void light_set_negative(RID p_light, bool p_enable);
	void light_set_cull_mask(RID p_light, uint32_t p_mask);
	void light_set_reverse_cull_face_mode(RID p_light, bool p_enabled);
	void light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode);
	void light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade);

	void light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode);

	void light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode);
	void light_directional_set_blend_splits(RID p_light, bool p_enable);
	bool light_directional_get_blend_splits(RID p_light) const;
	void light_directional_set_shadow_depth_range_mode(RID p_light, RS::LightDirectionalShadowDepthRangeMode p_range_mode);
	RS::LightDirectionalShadowDepthRangeMode light_directional_get_shadow_depth_range_mode(RID p_light) const;

	RS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light);
	RS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light);

	_FORCE_INLINE_ RS::LightType light_get_type(RID p_light) const {
		const Light *light = light_owner.getornull(p_light);
		ERR_FAIL_COND_V(!light, RS::LIGHT_DIRECTIONAL);

		return light->type;
	}
	AABB light_get_aabb(RID p_light) const;

	_FORCE_INLINE_ float light_get_param(RID p_light, RS::LightParam p_param) {
		const Light *light = light_owner.getornull(p_light);
		ERR_FAIL_COND_V(!light, 0);

		return light->param[p_param];
	}

	_FORCE_INLINE_ RID light_get_projector(RID p_light) {
		const Light *light = light_owner.getornull(p_light);
		ERR_FAIL_COND_V(!light, RID());

		return light->projector;
	}

	_FORCE_INLINE_ Color light_get_color(RID p_light) {
		const Light *light = light_owner.getornull(p_light);
		ERR_FAIL_COND_V(!light, Color());

		return light->color;
	}

	_FORCE_INLINE_ Color light_get_shadow_color(RID p_light) {
		const Light *light = light_owner.getornull(p_light);
		ERR_FAIL_COND_V(!light, Color());

		return light->shadow_color;
	}

	_FORCE_INLINE_ uint32_t light_get_cull_mask(RID p_light) {
		const Light *light = light_owner.getornull(p_light);
		ERR_FAIL_COND_V(!light, 0);

		return light->cull_mask;
	}

	_FORCE_INLINE_ bool light_has_shadow(RID p_light) const {
		const Light *light = light_owner.getornull(p_light);
		ERR_FAIL_COND_V(!light, RS::LIGHT_DIRECTIONAL);

		return light->shadow;
	}

	_FORCE_INLINE_ bool light_is_negative(RID p_light) const {
		const Light *light = light_owner.getornull(p_light);
		ERR_FAIL_COND_V(!light, RS::LIGHT_DIRECTIONAL);

		return light->negative;
	}

	_FORCE_INLINE_ float light_get_transmittance_bias(RID p_light) const {
		const Light *light = light_owner.getornull(p_light);
		ERR_FAIL_COND_V(!light, 0.0);

		return light->param[RS::LIGHT_PARAM_TRANSMITTANCE_BIAS];
	}

	_FORCE_INLINE_ float light_get_shadow_volumetric_fog_fade(RID p_light) const {
		const Light *light = light_owner.getornull(p_light);
		ERR_FAIL_COND_V(!light, 0.0);

		return light->param[RS::LIGHT_PARAM_SHADOW_VOLUMETRIC_FOG_FADE];
	}

	RS::LightBakeMode light_get_bake_mode(RID p_light);
	uint32_t light_get_max_sdfgi_cascade(RID p_light);
	uint64_t light_get_version(RID p_light) const;

	/* PROBE API */

	RID reflection_probe_create();

	void reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode);
	void reflection_probe_set_intensity(RID p_probe, float p_intensity);
	void reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode);
	void reflection_probe_set_ambient_color(RID p_probe, const Color &p_color);
	void reflection_probe_set_ambient_energy(RID p_probe, float p_energy);
	void reflection_probe_set_max_distance(RID p_probe, float p_distance);
	void reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents);
	void reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset);
	void reflection_probe_set_as_interior(RID p_probe, bool p_enable);
	void reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable);
	void reflection_probe_set_enable_shadows(RID p_probe, bool p_enable);
	void reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers);
	void reflection_probe_set_resolution(RID p_probe, int p_resolution);

	AABB reflection_probe_get_aabb(RID p_probe) const;
	RS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const;
	uint32_t reflection_probe_get_cull_mask(RID p_probe) const;
	Vector3 reflection_probe_get_extents(RID p_probe) const;
	Vector3 reflection_probe_get_origin_offset(RID p_probe) const;
	float reflection_probe_get_origin_max_distance(RID p_probe) const;
	int reflection_probe_get_resolution(RID p_probe) const;
	bool reflection_probe_renders_shadows(RID p_probe) const;

	float reflection_probe_get_intensity(RID p_probe) const;
	bool reflection_probe_is_interior(RID p_probe) const;
	bool reflection_probe_is_box_projection(RID p_probe) const;
	RS::ReflectionProbeAmbientMode reflection_probe_get_ambient_mode(RID p_probe) const;
	Color reflection_probe_get_ambient_color(RID p_probe) const;
	float reflection_probe_get_ambient_color_energy(RID p_probe) const;

	void base_update_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance);
	void skeleton_update_dependency(RID p_skeleton, RasterizerScene::InstanceBase *p_instance);

	/* DECAL API */

	virtual RID decal_create();
	virtual void decal_set_extents(RID p_decal, const Vector3 &p_extents);
	virtual void decal_set_texture(RID p_decal, RS::DecalTexture p_type, RID p_texture);
	virtual void decal_set_emission_energy(RID p_decal, float p_energy);
	virtual void decal_set_albedo_mix(RID p_decal, float p_mix);
	virtual void decal_set_modulate(RID p_decal, const Color &p_modulate);
	virtual void decal_set_cull_mask(RID p_decal, uint32_t p_layers);
	virtual void decal_set_distance_fade(RID p_decal, bool p_enabled, float p_begin, float p_length);
	virtual void decal_set_fade(RID p_decal, float p_above, float p_below);
	virtual void decal_set_normal_fade(RID p_decal, float p_fade);

	_FORCE_INLINE_ Vector3 decal_get_extents(RID p_decal) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->extents;
	}

	_FORCE_INLINE_ RID decal_get_texture(RID p_decal, RS::DecalTexture p_texture) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->textures[p_texture];
	}

	_FORCE_INLINE_ Color decal_get_modulate(RID p_decal) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->modulate;
	}

	_FORCE_INLINE_ float decal_get_emission_energy(RID p_decal) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->emission_energy;
	}

	_FORCE_INLINE_ float decal_get_albedo_mix(RID p_decal) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->albedo_mix;
	}

	_FORCE_INLINE_ uint32_t decal_get_cull_mask(RID p_decal) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->cull_mask;
	}

	_FORCE_INLINE_ float decal_get_upper_fade(RID p_decal) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->upper_fade;
	}

	_FORCE_INLINE_ float decal_get_lower_fade(RID p_decal) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->lower_fade;
	}

	_FORCE_INLINE_ float decal_get_normal_fade(RID p_decal) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->normal_fade;
	}

	_FORCE_INLINE_ bool decal_is_distance_fade_enabled(RID p_decal) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->distance_fade;
	}

	_FORCE_INLINE_ float decal_get_distance_fade_begin(RID p_decal) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->distance_fade_begin;
	}

	_FORCE_INLINE_ float decal_get_distance_fade_length(RID p_decal) {
		const Decal *decal = decal_owner.getornull(p_decal);
		return decal->distance_fade_length;
	}

	virtual AABB decal_get_aabb(RID p_decal) const;

	/* GI PROBE API */

	RID gi_probe_create();

	void gi_probe_allocate(RID p_gi_probe, const Transform &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts);

	AABB gi_probe_get_bounds(RID p_gi_probe) const;
	Vector3i gi_probe_get_octree_size(RID p_gi_probe) const;
	Vector<uint8_t> gi_probe_get_octree_cells(RID p_gi_probe) const;
	Vector<uint8_t> gi_probe_get_data_cells(RID p_gi_probe) const;
	Vector<uint8_t> gi_probe_get_distance_field(RID p_gi_probe) const;

	Vector<int> gi_probe_get_level_counts(RID p_gi_probe) const;
	Transform gi_probe_get_to_cell_xform(RID p_gi_probe) const;

	void gi_probe_set_dynamic_range(RID p_gi_probe, float p_range);
	float gi_probe_get_dynamic_range(RID p_gi_probe) const;

	void gi_probe_set_propagation(RID p_gi_probe, float p_range);
	float gi_probe_get_propagation(RID p_gi_probe) const;

	void gi_probe_set_energy(RID p_gi_probe, float p_energy);
	float gi_probe_get_energy(RID p_gi_probe) const;

	void gi_probe_set_ao(RID p_gi_probe, float p_ao);
	float gi_probe_get_ao(RID p_gi_probe) const;

	void gi_probe_set_ao_size(RID p_gi_probe, float p_strength);
	float gi_probe_get_ao_size(RID p_gi_probe) const;

	void gi_probe_set_bias(RID p_gi_probe, float p_bias);
	float gi_probe_get_bias(RID p_gi_probe) const;

	void gi_probe_set_normal_bias(RID p_gi_probe, float p_range);
	float gi_probe_get_normal_bias(RID p_gi_probe) const;

	void gi_probe_set_interior(RID p_gi_probe, bool p_enable);
	bool gi_probe_is_interior(RID p_gi_probe) const;

	void gi_probe_set_use_two_bounces(RID p_gi_probe, bool p_enable);
	bool gi_probe_is_using_two_bounces(RID p_gi_probe) const;

	void gi_probe_set_anisotropy_strength(RID p_gi_probe, float p_strength);
	float gi_probe_get_anisotropy_strength(RID p_gi_probe) const;

	uint32_t gi_probe_get_version(RID p_probe);
	uint32_t gi_probe_get_data_version(RID p_probe);

	RID gi_probe_get_octree_buffer(RID p_gi_probe) const;
	RID gi_probe_get_data_buffer(RID p_gi_probe) const;

	RID gi_probe_get_sdf_texture(RID p_gi_probe);

	/* LIGHTMAP CAPTURE */

	virtual RID lightmap_create();

	virtual void lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics);
	virtual void lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds);
	virtual void lightmap_set_probe_interior(RID p_lightmap, bool p_interior);
	virtual void lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree);
	virtual PackedVector3Array lightmap_get_probe_capture_points(RID p_lightmap) const;
	virtual PackedColorArray lightmap_get_probe_capture_sh(RID p_lightmap) const;
	virtual PackedInt32Array lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const;
	virtual PackedInt32Array lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const;
	virtual AABB lightmap_get_aabb(RID p_lightmap) const;
	virtual bool lightmap_is_interior(RID p_lightmap) const;
	virtual void lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh);
	virtual void lightmap_set_probe_capture_update_speed(float p_speed);
	_FORCE_INLINE_ float lightmap_get_probe_capture_update_speed() const {
		return lightmap_probe_capture_update_speed;
	}

	_FORCE_INLINE_ int32_t lightmap_get_array_index(RID p_lightmap) const {
		ERR_FAIL_COND_V(!using_lightmap_array, -1); //only for arrays
		const Lightmap *lm = lightmap_owner.getornull(p_lightmap);
		return lm->array_index;
	}
	_FORCE_INLINE_ bool lightmap_uses_spherical_harmonics(RID p_lightmap) const {
		ERR_FAIL_COND_V(!using_lightmap_array, false); //only for arrays
		const Lightmap *lm = lightmap_owner.getornull(p_lightmap);
		return lm->uses_spherical_harmonics;
	}
	_FORCE_INLINE_ uint64_t lightmap_array_get_version() const {
		ERR_FAIL_COND_V(!using_lightmap_array, 0); //only for arrays
		return lightmap_array_version;
	}

	_FORCE_INLINE_ int lightmap_array_get_size() const {
		ERR_FAIL_COND_V(!using_lightmap_array, 0); //only for arrays
		return lightmap_textures.size();
	}

	_FORCE_INLINE_ const Vector<RID> &lightmap_array_get_textures() const {
		ERR_FAIL_COND_V(!using_lightmap_array, lightmap_textures); //only for arrays
		return lightmap_textures;
	}

	/* PARTICLES */

	RID particles_create();

	void particles_set_emitting(RID p_particles, bool p_emitting);
	void particles_set_amount(RID p_particles, int p_amount);
	void particles_set_lifetime(RID p_particles, float p_lifetime);
	void particles_set_one_shot(RID p_particles, bool p_one_shot);
	void particles_set_pre_process_time(RID p_particles, float p_time);
	void particles_set_explosiveness_ratio(RID p_particles, float p_ratio);
	void particles_set_randomness_ratio(RID p_particles, float p_ratio);
	void particles_set_custom_aabb(RID p_particles, const AABB &p_aabb);
	void particles_set_speed_scale(RID p_particles, float p_scale);
	void particles_set_use_local_coordinates(RID p_particles, bool p_enable);
	void particles_set_process_material(RID p_particles, RID p_material);
	void particles_set_fixed_fps(RID p_particles, int p_fps);
	void particles_set_fractional_delta(RID p_particles, bool p_enable);
	void particles_set_collision_base_size(RID p_particles, float p_size);
	void particles_restart(RID p_particles);
	void particles_emit(RID p_particles, const Transform &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags);
	void particles_set_subemitter(RID p_particles, RID p_subemitter_particles);

	void particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order);

	void particles_set_draw_passes(RID p_particles, int p_count);
	void particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh);

	void particles_request_process(RID p_particles);
	AABB particles_get_current_aabb(RID p_particles);
	AABB particles_get_aabb(RID p_particles) const;

	void particles_set_emission_transform(RID p_particles, const Transform &p_transform);

	bool particles_get_emitting(RID p_particles);
	int particles_get_draw_passes(RID p_particles) const;
	RID particles_get_draw_pass_mesh(RID p_particles, int p_pass) const;

	void particles_set_view_axis(RID p_particles, const Vector3 &p_axis);

	virtual bool particles_is_inactive(RID p_particles) const;

	_FORCE_INLINE_ uint32_t particles_get_amount(RID p_particles) {
		Particles *particles = particles_owner.getornull(p_particles);
		ERR_FAIL_COND_V(!particles, 0);

		return particles->amount;
	}

	_FORCE_INLINE_ uint32_t particles_is_using_local_coords(RID p_particles) {
		Particles *particles = particles_owner.getornull(p_particles);
		ERR_FAIL_COND_V(!particles, false);

		return particles->use_local_coords;
	}

	_FORCE_INLINE_ RID particles_get_instance_buffer_uniform_set(RID p_particles, RID p_shader, uint32_t p_set) {
		Particles *particles = particles_owner.getornull(p_particles);
		ERR_FAIL_COND_V(!particles, RID());
		if (particles->particles_transforms_buffer_uniform_set.is_null()) {
			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 0;
				u.ids.push_back(particles->particle_instance_buffer);
				uniforms.push_back(u);
			}

			particles->particles_transforms_buffer_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, p_shader, p_set);
		}

		return particles->particles_transforms_buffer_uniform_set;
	}

	virtual void particles_add_collision(RID p_particles, RasterizerScene::InstanceBase *p_instance);
	virtual void particles_remove_collision(RID p_particles, RasterizerScene::InstanceBase *p_instance);

	/* PARTICLES COLLISION */

	virtual RID particles_collision_create();
	virtual void particles_collision_set_collision_type(RID p_particles_collision, RS::ParticlesCollisionType p_type);
	virtual void particles_collision_set_cull_mask(RID p_particles_collision, uint32_t p_cull_mask);
	virtual void particles_collision_set_sphere_radius(RID p_particles_collision, float p_radius); //for spheres
	virtual void particles_collision_set_box_extents(RID p_particles_collision, const Vector3 &p_extents); //for non-spheres
	virtual void particles_collision_set_attractor_strength(RID p_particles_collision, float p_strength);
	virtual void particles_collision_set_attractor_directionality(RID p_particles_collision, float p_directionality);
	virtual void particles_collision_set_attractor_attenuation(RID p_particles_collision, float p_curve);
	virtual void particles_collision_set_field_texture(RID p_particles_collision, RID p_texture); //for SDF and vector field, heightfield is dynamic
	virtual void particles_collision_height_field_update(RID p_particles_collision); //for SDF and vector field
	virtual void particles_collision_set_height_field_resolution(RID p_particles_collision, RS::ParticlesCollisionHeightfieldResolution p_resolution); //for SDF and vector field
	virtual AABB particles_collision_get_aabb(RID p_particles_collision) const;
	virtual Vector3 particles_collision_get_extents(RID p_particles_collision) const;
	virtual bool particles_collision_is_heightfield(RID p_particles_collision) const;
	RID particles_collision_get_heightfield_framebuffer(RID p_particles_collision) const;

	/* GLOBAL VARIABLES API */

	virtual void global_variable_add(const StringName &p_name, RS::GlobalVariableType p_type, const Variant &p_value);
	virtual void global_variable_remove(const StringName &p_name);
	virtual Vector<StringName> global_variable_get_list() const;

	virtual void global_variable_set(const StringName &p_name, const Variant &p_value);
	virtual void global_variable_set_override(const StringName &p_name, const Variant &p_value);
	virtual Variant global_variable_get(const StringName &p_name) const;
	virtual RS::GlobalVariableType global_variable_get_type(const StringName &p_name) const;
	RS::GlobalVariableType global_variable_get_type_internal(const StringName &p_name) const;

	virtual void global_variables_load_settings(bool p_load_textures = true);
	virtual void global_variables_clear();

	virtual int32_t global_variables_instance_allocate(RID p_instance);
	virtual void global_variables_instance_free(RID p_instance);
	virtual void global_variables_instance_update(RID p_instance, int p_index, const Variant &p_value);

	RID global_variables_get_storage_buffer() const;

	/* RENDER TARGET API */

	RID render_target_create();
	void render_target_set_position(RID p_render_target, int p_x, int p_y);
	void render_target_set_size(RID p_render_target, int p_width, int p_height);
	RID render_target_get_texture(RID p_render_target);
	void render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id);
	void render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value);
	bool render_target_was_used(RID p_render_target);
	void render_target_set_as_unused(RID p_render_target);
	void render_target_copy_to_back_buffer(RID p_render_target, const Rect2i &p_region, bool p_gen_mipmaps);
	void render_target_clear_back_buffer(RID p_render_target, const Rect2i &p_region, const Color &p_color);
	void render_target_gen_back_buffer_mipmaps(RID p_render_target, const Rect2i &p_region);

	RID render_target_get_back_buffer_uniform_set(RID p_render_target, RID p_base_shader);

	virtual void render_target_request_clear(RID p_render_target, const Color &p_clear_color);
	virtual bool render_target_is_clear_requested(RID p_render_target);
	virtual Color render_target_get_clear_request_color(RID p_render_target);
	virtual void render_target_disable_clear_request(RID p_render_target);
	virtual void render_target_do_clear_request(RID p_render_target);

	Size2 render_target_get_size(RID p_render_target);
	RID render_target_get_rd_framebuffer(RID p_render_target);
	RID render_target_get_rd_texture(RID p_render_target);
	RID render_target_get_rd_backbuffer(RID p_render_target);
	RID render_target_get_rd_backbuffer_framebuffer(RID p_render_target);

	RID render_target_get_framebuffer_uniform_set(RID p_render_target);
	RID render_target_get_backbuffer_uniform_set(RID p_render_target);

	void render_target_set_framebuffer_uniform_set(RID p_render_target, RID p_uniform_set);
	void render_target_set_backbuffer_uniform_set(RID p_render_target, RID p_uniform_set);

	RS::InstanceType get_base_type(RID p_rid) const;

	bool free(RID p_rid);

	bool has_os_feature(const String &p_feature) const;

	void update_dirty_resources();

	void set_debug_generate_wireframes(bool p_generate) {}

	void render_info_begin_capture() {}
	void render_info_end_capture() {}
	int get_captured_render_info(RS::RenderInfo p_info) { return 0; }

	int get_render_info(RS::RenderInfo p_info) { return 0; }
	String get_video_adapter_name() const { return String(); }
	String get_video_adapter_vendor() const { return String(); }

	virtual void capture_timestamps_begin();
	virtual void capture_timestamp(const String &p_name);
	virtual uint32_t get_captured_timestamps_count() const;
	virtual uint64_t get_captured_timestamps_frame() const;
	virtual uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const;
	virtual uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const;
	virtual String get_captured_timestamp_name(uint32_t p_index) const;

	RID get_default_rd_storage_buffer() { return default_rd_storage_buffer; }

	static RasterizerStorageRD *base_singleton;

	RasterizerEffectsRD *get_effects();

	RasterizerStorageRD();
	~RasterizerStorageRD();
};

#endif // RASTERIZER_STORAGE_RD_H
