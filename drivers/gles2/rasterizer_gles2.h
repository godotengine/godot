/*************************************************************************/
/*  rasterizer_gles2.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef RASTERIZER_GLES2_H
#define RASTERIZER_GLES2_H

#include "servers/visual/rasterizer.h"

#define MAX_POLYGON_VERTICES 4096 //used for WebGL canvas_draw_polygon call.

#ifdef GLES2_ENABLED

#include "camera_matrix.h"
#include "image.h"
#include "list.h"
#include "map.h"
#include "rid.h"
#include "self_list.h"
#include "servers/visual_server.h"
#include "sort.h"

#include "platform_config.h"
#ifndef GLES2_INCLUDE_H
#include <GLES2/gl2.h>
#else
#include GLES2_INCLUDE_H
#endif

#include "drivers/gles2/shader_compiler_gles2.h"
#include "drivers/gles2/shaders/blur.glsl.h"
#include "drivers/gles2/shaders/canvas.glsl.h"
#include "drivers/gles2/shaders/canvas_shadow.glsl.h"
#include "drivers/gles2/shaders/copy.glsl.h"
#include "drivers/gles2/shaders/material.glsl.h"
#include "servers/visual/particle_system_sw.h"

/**
        @author Juan Linietsky <reduzio@gmail.com>
*/
class RasterizerGLES2 : public Rasterizer {

	enum {

		MAX_SCENE_LIGHTS = 2048,
		LIGHT_SPOT_BIT = 0x80,
		DEFAULT_SKINNED_BUFFER_SIZE = 2048, // 10k vertices
		MAX_HW_LIGHTS = 1,
	};

	uint8_t *skinned_buffer;
	int skinned_buffer_size;
	bool pvr_supported;
	bool pvr_srgb_supported;
	bool s3tc_supported;
	bool s3tc_srgb_supported;
	bool latc_supported;
	bool etc_supported;
	bool atitc_supported;
	bool npo2_textures_available;
	bool read_depth_supported;
	bool use_framebuffers;
	bool full_float_fb_supported;
	bool use_shadow_mapping;
	bool use_fp16_fb;
	bool srgb_supported;
	bool float_supported;
	bool float_linear_supported;
	bool use_16bits_fbo;

	ShadowFilterTechnique shadow_filter;

	bool use_shadow_esm;
	bool use_shadow_pcf;
	bool use_hw_skeleton_xform;
	bool use_depth24;
	bool use_texture_instancing;
	bool use_attribute_instancing;
	bool use_rgba_shadowmaps;
	bool use_anisotropic_filter;
	float anisotropic_level;

	bool use_half_float;
	bool low_memory_2d;

	bool shrink_textures_x2;

	Vector<float> skel_default;

	Image _get_gl_image_and_format(const Image &p_image, Image::Format p_format, uint32_t p_flags, GLenum &r_gl_format, GLenum &r_gl_internal_format, int &r_gl_components, bool &r_has_alpha_cache, bool &r_compressed);

	struct RenderTarget;

	struct Texture {

		String path;
		uint32_t flags;
		int width, height;
		int alloc_width, alloc_height;
		Image::Format format;

		GLenum target;
		GLenum gl_format_cache;
		GLenum gl_internal_format_cache;
		int gl_components_cache;
		int data_size; //original data size, useful for retrieving back
		bool has_alpha;
		bool format_has_alpha;
		bool compressed;
		bool disallow_mipmaps;
		int total_data_size;
		bool ignore_mipmaps;

		ObjectID reloader;
		StringName reloader_func;
		Image image[6];

		int mipmaps;

		bool active;
		GLuint tex_id;

		RenderTarget *render_target;

		Texture() {

			ignore_mipmaps = false;
			render_target = NULL;
			flags = width = height = 0;
			tex_id = 0;
			data_size = 0;
			format = Image::FORMAT_L8;
			gl_components_cache = 0;
			format_has_alpha = false;
			has_alpha = false;
			active = false;
			disallow_mipmaps = false;
			compressed = false;
			total_data_size = 0;
			target = GL_TEXTURE_2D;
			mipmaps = 0;

			reloader = 0;
		}

		~Texture() {

			if (tex_id != 0) {

				glDeleteTextures(1, &tex_id);
			}
		}
	};

	mutable RID_Owner<Texture> texture_owner;

	struct Shader {

		String vertex_code;
		String fragment_code;
		String light_code;
		int vertex_line;
		int fragment_line;
		int light_line;
		VS::ShaderMode mode;

		uint32_t custom_code_id;
		uint32_t version;

		bool valid;
		bool has_alpha;
		bool can_zpass;
		bool has_texscreen;
		bool has_screen_uv;
		bool writes_vertex;
		bool uses_discard;
		bool uses_time;
		bool uses_normal;
		bool uses_texpixel_size;

		Map<StringName, ShaderLanguage::Uniform> uniforms;
		StringName first_texture;

		Map<StringName, RID> default_textures;

		SelfList<Shader> dirty_list;

		Shader()
			: dirty_list(this) {

			valid = false;
			custom_code_id = 0;
			has_alpha = false;
			version = 1;
			vertex_line = 0;
			fragment_line = 0;
			light_line = 0;
			can_zpass = true;
			has_texscreen = false;
			has_screen_uv = false;
			writes_vertex = false;
			uses_discard = false;
			uses_time = false;
			uses_normal = false;
		}
	};

	mutable RID_Owner<Shader> shader_owner;
	mutable SelfList<Shader>::List _shader_dirty_list;
	_FORCE_INLINE_ void _shader_make_dirty(Shader *p_shader);
	void _update_shader(Shader *p_shader) const;

	struct Material {

		bool flags[VS::MATERIAL_FLAG_MAX];

		VS::MaterialBlendMode blend_mode;
		VS::MaterialDepthDrawMode depth_draw_mode;

		float line_width;
		bool has_alpha;

		mutable uint32_t shader_version;

		RID shader; // shader material
		Shader *shader_cache;

		struct UniformData {

			bool inuse;
			bool istexture;
			Variant value;
			int index;
		};

		mutable Map<StringName, UniformData> shader_params;

		uint64_t last_pass;

		Material() {

			for (int i = 0; i < VS::MATERIAL_FLAG_MAX; i++)
				flags[i] = false;
			flags[VS::MATERIAL_FLAG_VISIBLE] = true;

			line_width = 1;
			has_alpha = false;
			depth_draw_mode = VS::MATERIAL_DEPTH_DRAW_OPAQUE_ONLY;
			blend_mode = VS::MATERIAL_BLEND_MODE_MIX;
			last_pass = 0;
			shader_version = 0;
			shader_cache = NULL;
		}
	};

	_FORCE_INLINE_ void _update_material_shader_params(Material *p_material) const;
	mutable RID_Owner<Material> material_owner;

	struct Geometry {

		enum Type {
			GEOMETRY_INVALID,
			GEOMETRY_SURFACE,
			GEOMETRY_IMMEDIATE,
			GEOMETRY_PARTICLES,
			GEOMETRY_MULTISURFACE,
		};

		Type type;
		RID material;
		bool has_alpha;
		bool material_owned;

		Geometry() {
			has_alpha = false;
			material_owned = false;
		}
		virtual ~Geometry(){};
	};

	struct GeometryOwner {

		virtual ~GeometryOwner() {}
	};

	struct Mesh;

	struct Surface : public Geometry {

		struct ArrayData {

			uint32_t ofs, size, datatype, count;
			bool normalize;
			bool bind;

			ArrayData() {
				ofs = 0;
				size = 0;
				count = 0;
				datatype = 0;
				normalize = 0;
				bind = false;
			}
		};

		Mesh *mesh;

		Array data;
		Array morph_data;
		ArrayData array[VS::ARRAY_MAX];
		// support for vertex array objects
		GLuint array_object_id;
		// support for vertex buffer object
		GLuint vertex_id; // 0 means, unconfigured
		GLuint index_id; // 0 means, unconfigured
		// no support for the above, array in localmem.
		uint8_t *array_local;
		uint8_t *index_array_local;
		Vector<AABB> skeleton_bone_aabb;
		Vector<bool> skeleton_bone_used;

		//bool packed;

		struct MorphTarget {
			uint32_t configured_format;
			uint8_t *array;
		};

		MorphTarget *morph_targets_local;
		int morph_target_count;
		AABB aabb;

		int array_len;
		int index_array_len;
		int max_bone;

		float vertex_scale;
		float uv_scale;
		float uv2_scale;

		bool alpha_sort;

		VS::PrimitiveType primitive;

		uint32_t format;
		uint32_t configured_format;

		int stride;
		int local_stride;
		uint32_t morph_format;

		bool active;

		Point2 uv_min;
		Point2 uv_max;

		Surface() {

			array_len = 0;
			local_stride = 0;
			morph_format = 0;
			type = GEOMETRY_SURFACE;
			primitive = VS::PRIMITIVE_POINTS;
			index_array_len = 0;
			vertex_scale = 1.0;
			uv_scale = 1.0;
			uv2_scale = 1.0;

			alpha_sort = false;

			format = 0;
			stride = 0;
			morph_targets_local = 0;
			morph_target_count = 0;

			array_local = index_array_local = 0;
			vertex_id = index_id = 0;

			active = false;
			//packed=false;
		}

		~Surface() {
		}
	};

	struct Mesh {

		bool active;
		Vector<Surface *> surfaces;
		int morph_target_count;
		VS::MorphTargetMode morph_target_mode;
		AABB custom_aabb;

		mutable uint64_t last_pass;
		Mesh() {
			morph_target_mode = VS::MORPH_MODE_NORMALIZED;
			morph_target_count = 0;
			last_pass = 0;
			active = false;
		}
	};
	mutable RID_Owner<Mesh> mesh_owner;

	Error _surface_set_arrays(Surface *p_surface, uint8_t *p_mem, uint8_t *p_index_mem, const Array &p_arrays, bool p_main);

	struct MultiMesh;

	struct MultiMeshSurface : public Geometry {

		Surface *surface;
		MultiMeshSurface() { type = GEOMETRY_MULTISURFACE; }
	};

	struct MultiMesh : public GeometryOwner {

		struct Element {

			float matrix[16];
			uint8_t color[4];
			Element() {
				matrix[0] = 1;
				matrix[1] = 0;
				matrix[2] = 0;
				matrix[3] = 0;

				matrix[4] = 0;
				matrix[5] = 1;
				matrix[6] = 0;
				matrix[7] = 0;

				matrix[8] = 0;
				matrix[9] = 0;
				matrix[10] = 1;
				matrix[11] = 0;

				matrix[12] = 0;
				matrix[13] = 0;
				matrix[14] = 0;
				matrix[15] = 1;
			};
		};

		AABB aabb;
		RID mesh;
		int visible;

		//IDirect3DVertexBuffer9* instance_buffer;
		Vector<Element> elements;
		Vector<MultiMeshSurface> cache_surfaces;
		mutable uint64_t last_pass;
		GLuint tex_id;
		int tw;
		int th;

		SelfList<MultiMesh> dirty_list;

		MultiMesh()
			: dirty_list(this) {

			tw = 1;
			th = 1;
			tex_id = 0;
			last_pass = 0;
			visible = -1;
		}
	};

	mutable RID_Owner<MultiMesh> multimesh_owner;
	mutable SelfList<MultiMesh>::List _multimesh_dirty_list;

	struct Immediate : public Geometry {

		struct Chunk {

			RID texture;
			VS::PrimitiveType primitive;
			Vector<Vector3> vertices;
			Vector<Vector3> normals;
			Vector<Plane> tangents;
			Vector<Color> colors;
			Vector<Vector2> uvs;
			Vector<Vector2> uvs2;
		};

		List<Chunk> chunks;
		bool building;
		int mask;
		AABB aabb;

		Immediate() {
			type = GEOMETRY_IMMEDIATE;
			building = false;
		}
	};

	mutable RID_Owner<Immediate> immediate_owner;

	struct Particles : public Geometry {

		ParticleSystemSW data; // software particle system

		Particles() {
			type = GEOMETRY_PARTICLES;
		}
	};

	mutable RID_Owner<Particles> particles_owner;

	struct ParticlesInstance : public GeometryOwner {

		RID particles;

		ParticleSystemProcessSW particles_process;
		Transform transform;

		ParticlesInstance() {}
	};

	mutable RID_Owner<ParticlesInstance> particles_instance_owner;
	ParticleSystemDrawInfoSW particle_draw_info;

	struct Skeleton {

		struct Bone {

			float mtx[4][4]; //used

			Bone() {
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {

						mtx[i][j] = (i == j) ? 1 : 0;
					}
				}
			}

			_ALWAYS_INLINE_ void transform_add_mul3(const float *p_src, float *r_dst, float p_weight) const {

				r_dst[0] += ((mtx[0][0] * p_src[0]) + (mtx[1][0] * p_src[1]) + (mtx[2][0] * p_src[2]) + mtx[3][0]) * p_weight;
				r_dst[1] += ((mtx[0][1] * p_src[0]) + (mtx[1][1] * p_src[1]) + (mtx[2][1] * p_src[2]) + mtx[3][1]) * p_weight;
				r_dst[2] += ((mtx[0][2] * p_src[0]) + (mtx[1][2] * p_src[1]) + (mtx[2][2] * p_src[2]) + mtx[3][2]) * p_weight;
			}
			_ALWAYS_INLINE_ void transform3_add_mul3(const float *p_src, float *r_dst, float p_weight) const {

				r_dst[0] += ((mtx[0][0] * p_src[0]) + (mtx[1][0] * p_src[1]) + (mtx[2][0] * p_src[2])) * p_weight;
				r_dst[1] += ((mtx[0][1] * p_src[0]) + (mtx[1][1] * p_src[1]) + (mtx[2][1] * p_src[2])) * p_weight;
				r_dst[2] += ((mtx[0][2] * p_src[0]) + (mtx[1][2] * p_src[1]) + (mtx[2][2] * p_src[2])) * p_weight;
			}

			_ALWAYS_INLINE_ AABB transform_aabb(const AABB &p_aabb) const {

				float vertices[8][3] = {
					{ p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z + p_aabb.size.z },
					{ p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z },
					{ p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y, p_aabb.pos.z + p_aabb.size.z },
					{ p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y, p_aabb.pos.z },
					{ p_aabb.pos.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z + p_aabb.size.z },
					{ p_aabb.pos.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z },
					{ p_aabb.pos.x, p_aabb.pos.y, p_aabb.pos.z + p_aabb.size.z },
					{ p_aabb.pos.x, p_aabb.pos.y, p_aabb.pos.z }
				};

				AABB ret;

				for (int i = 0; i < 8; i++) {

					Vector3 xv(

							((mtx[0][0] * vertices[i][0]) + (mtx[1][0] * vertices[i][1]) + (mtx[2][0] * vertices[i][2]) + mtx[3][0]),
							((mtx[0][1] * vertices[i][0]) + (mtx[1][1] * vertices[i][1]) + (mtx[2][1] * vertices[i][2]) + mtx[3][1]),
							((mtx[0][2] * vertices[i][0]) + (mtx[1][2] * vertices[i][1]) + (mtx[2][2] * vertices[i][2]) + mtx[3][2]));

					if (i == 0)
						ret.pos = xv;
					else
						ret.expand_to(xv);
				}

				return ret;
			}
		};

		GLuint tex_id;
		float pixel_size; //for texture
		Vector<Bone> bones;

		SelfList<Skeleton> dirty_list;

		Skeleton()
			: dirty_list(this) {
			tex_id = 0;
			pixel_size = 1.0;
		}
	};

	mutable RID_Owner<Skeleton> skeleton_owner;
	mutable SelfList<Skeleton>::List _skeleton_dirty_list;

	template <bool USE_NORMAL, bool USE_TANGENT, bool INPLACE>
	void _skeleton_xform(const uint8_t *p_src_array, int p_src_stride, uint8_t *p_dst_array, int p_dst_stride, int p_elements, const uint8_t *p_src_bones, const uint8_t *p_src_weights, const Skeleton::Bone *p_bone_xforms);

	struct Light {

		VS::LightType type;
		float vars[VS::LIGHT_PARAM_MAX];
		Color colors[3];
		bool shadow_enabled;
		RID projector;
		bool volumetric_enabled;
		Color volumetric_color;
		VS::LightOmniShadowMode omni_shadow_mode;
		VS::LightDirectionalShadowMode directional_shadow_mode;
		float directional_shadow_param[3];

		Light() {

			vars[VS::LIGHT_PARAM_SPOT_ATTENUATION] = 1;
			vars[VS::LIGHT_PARAM_SPOT_ANGLE] = 45;
			vars[VS::LIGHT_PARAM_ATTENUATION] = 1.0;
			vars[VS::LIGHT_PARAM_ENERGY] = 1.0;
			vars[VS::LIGHT_PARAM_RADIUS] = 1.0;
			vars[VS::LIGHT_PARAM_SHADOW_DARKENING] = 0.0;
			vars[VS::LIGHT_PARAM_SHADOW_Z_OFFSET] = 0.2;
			vars[VS::LIGHT_PARAM_SHADOW_Z_SLOPE_SCALE] = 1.4;
			vars[VS::LIGHT_PARAM_SHADOW_ESM_MULTIPLIER] = 60.0;
			vars[VS::LIGHT_PARAM_SHADOW_BLUR_PASSES] = 1;
			colors[VS::LIGHT_COLOR_DIFFUSE] = Color(1, 1, 1);
			colors[VS::LIGHT_COLOR_SPECULAR] = Color(1, 1, 1);
			shadow_enabled = false;
			volumetric_enabled = false;

			directional_shadow_param[VS::LIGHT_DIRECTIONAL_SHADOW_PARAM_PSSM_SPLIT_WEIGHT] = 0.5;
			directional_shadow_param[VS::LIGHT_DIRECTIONAL_SHADOW_PARAM_MAX_DISTANCE] = 0;
			directional_shadow_param[VS::LIGHT_DIRECTIONAL_SHADOW_PARAM_PSSM_ZOFFSET_SCALE] = 2.0;
			omni_shadow_mode = VS::LIGHT_OMNI_SHADOW_DEFAULT;
			directional_shadow_mode = VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL;
		}
	};

	struct Environment {

		VS::EnvironmentBG bg_mode;
		Variant bg_param[VS::ENV_BG_PARAM_MAX];
		bool fx_enabled[VS::ENV_FX_MAX];
		Variant fx_param[VS::ENV_FX_PARAM_MAX];

		Environment() {

			bg_mode = VS::ENV_BG_DEFAULT_COLOR;
			bg_param[VS::ENV_BG_PARAM_COLOR] = Color(0, 0, 0);
			bg_param[VS::ENV_BG_PARAM_TEXTURE] = RID();
			bg_param[VS::ENV_BG_PARAM_CUBEMAP] = RID();
			bg_param[VS::ENV_BG_PARAM_ENERGY] = 1.0;
			bg_param[VS::ENV_BG_PARAM_SCALE] = 1.0;
			bg_param[VS::ENV_BG_PARAM_GLOW] = 0.0;
			bg_param[VS::ENV_BG_PARAM_CANVAS_MAX_LAYER] = 0;

			for (int i = 0; i < VS::ENV_FX_MAX; i++)
				fx_enabled[i] = false;

			fx_param[VS::ENV_FX_PARAM_GLOW_BLUR_PASSES] = 1;
			fx_param[VS::ENV_FX_PARAM_GLOW_BLUR_SCALE] = 1.0;
			fx_param[VS::ENV_FX_PARAM_GLOW_BLUR_STRENGTH] = 1.0;
			fx_param[VS::ENV_FX_PARAM_GLOW_BLUR_BLEND_MODE] = 0;
			fx_param[VS::ENV_FX_PARAM_GLOW_BLOOM] = 0.0;
			fx_param[VS::ENV_FX_PARAM_GLOW_BLOOM_TRESHOLD] = 0.5;
			fx_param[VS::ENV_FX_PARAM_DOF_BLUR_PASSES] = 1;
			fx_param[VS::ENV_FX_PARAM_DOF_BLUR_BEGIN] = 100.0;
			fx_param[VS::ENV_FX_PARAM_DOF_BLUR_RANGE] = 10.0;
			fx_param[VS::ENV_FX_PARAM_HDR_TONEMAPPER] = VS::ENV_FX_HDR_TONE_MAPPER_LINEAR;
			fx_param[VS::ENV_FX_PARAM_HDR_EXPOSURE] = 0.4;
			fx_param[VS::ENV_FX_PARAM_HDR_WHITE] = 1.0;
			fx_param[VS::ENV_FX_PARAM_HDR_GLOW_TRESHOLD] = 0.95;
			fx_param[VS::ENV_FX_PARAM_HDR_GLOW_SCALE] = 0.2;
			fx_param[VS::ENV_FX_PARAM_HDR_MIN_LUMINANCE] = 0.4;
			fx_param[VS::ENV_FX_PARAM_HDR_MAX_LUMINANCE] = 8.0;
			fx_param[VS::ENV_FX_PARAM_HDR_EXPOSURE_ADJUST_SPEED] = 0.5;
			fx_param[VS::ENV_FX_PARAM_FOG_BEGIN] = 100.0;
			fx_param[VS::ENV_FX_PARAM_FOG_ATTENUATION] = 1.0;
			fx_param[VS::ENV_FX_PARAM_FOG_BEGIN_COLOR] = Color(0, 0, 0);
			fx_param[VS::ENV_FX_PARAM_FOG_END_COLOR] = Color(0, 0, 0);
			fx_param[VS::ENV_FX_PARAM_FOG_BG] = true;
			fx_param[VS::ENV_FX_PARAM_BCS_BRIGHTNESS] = 1.0;
			fx_param[VS::ENV_FX_PARAM_BCS_CONTRAST] = 1.0;
			fx_param[VS::ENV_FX_PARAM_BCS_SATURATION] = 1.0;
		}
	};

	mutable RID_Owner<Environment> environment_owner;

	struct SampledLight {

		int w, h;
		GLuint texture;
		float multiplier;
		bool is_float;
	};

	mutable RID_Owner<SampledLight> sampled_light_owner;

	struct ViewportData {

		//1x1 fbo+texture for storing previous HDR value
		GLuint lum_fbo;
		GLuint lum_color;

		ViewportData() {
			lum_fbo = 0;
			lum_color = 0;
		}
	};

	mutable RID_Owner<ViewportData> viewport_data_owner;

	struct RenderTarget {

		Texture *texture_ptr;
		RID texture;
		GLuint fbo;
		GLuint color;
		GLuint depth;
		int width, height;
		uint64_t last_pass;
	};

	mutable RID_Owner<RenderTarget> render_target_owner;

	struct ShadowBuffer;

	struct LightInstance {

		struct SplitInfo {

			CameraMatrix camera;
			Transform transform;
			float near;
			float far;
		};

		RID light;
		Light *base;
		Transform transform;
		CameraMatrix projection;

		Transform custom_transform[4];
		CameraMatrix custom_projection[4];

		Vector3 light_vector;
		Vector3 spot_vector;
		float linear_att;

		uint64_t shadow_pass;
		uint64_t last_pass;
		uint16_t sort_key;

		Vector2 dp;

		CameraMatrix shadow_projection[4];
		float shadow_split[4];

		ShadowBuffer *near_shadow_buffer;

		void clear_shadow_buffers() {

			clear_near_shadow_buffers();
		}

		void clear_near_shadow_buffers() {

			if (near_shadow_buffer) {
				near_shadow_buffer->owner = NULL;
				near_shadow_buffer = NULL;
			}
		}

		LightInstance() {
			shadow_pass = 0;
			last_pass = 0;
			sort_key = 0;
			near_shadow_buffer = NULL;
		}
	};
	mutable RID_Owner<Light> light_owner;
	mutable RID_Owner<LightInstance> light_instance_owner;

	LightInstance *light_instances[MAX_SCENE_LIGHTS];
	LightInstance *directional_lights[4];
	int light_instance_count;
	int directional_light_count;
	int last_light_id;
	bool current_depth_test;
	bool current_depth_mask;
	VS::MaterialBlendMode current_blend_mode;
	bool use_fast_texture_filter;
	int max_texture_size;

	bool fragment_lighting;
	RID shadow_material;
	RID shadow_material_double_sided;
	Material *shadow_mat_ptr;
	Material *shadow_mat_double_sided_ptr;

	int max_texture_units;
	GLuint base_framebuffer;

	GLuint gui_quad_buffer;
	GLuint indices_buffer;

	struct RenderList {

		enum {
			DEFAULT_MAX_ELEMENTS = 4096,
			MAX_LIGHTS = 4,
			SORT_FLAG_SKELETON = 1,
			SORT_FLAG_INSTANCING = 2,
		};

		static int max_elements;

		struct Element {

			float depth;
			const InstanceData *instance;
			const Skeleton *skeleton;
			const Geometry *geometry;
			const Geometry *geometry_cmp;
			const Material *material;
			const GeometryOwner *owner;
			bool *additive_ptr;
			bool additive;
			bool mirror;
			union {
#ifdef BIG_ENDIAN_ENABLED
				struct {
					uint8_t sort_flags;
					uint8_t light_type;
					uint16_t light;
				};
#else
				struct {
					uint16_t light;
					uint8_t light_type;
					uint8_t sort_flags;
				};
#endif
				uint32_t sort_key;
			};
		};

		Element *_elements;
		Element **elements;
		int element_count;

		void clear() {

			element_count = 0;
		}

		struct SortZ {

			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {

				return A->depth > B->depth;
			}
		};

		void sort_z() {

			SortArray<Element *, SortZ> sorter;
			sorter.sort(elements, element_count);
		}

		struct SortMatGeom {

			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {
				// TODO move to a single uint64 (one comparison)
				if (A->material->shader_cache == B->material->shader_cache) {
					if (A->material == B->material) {

						return A->geometry_cmp < B->geometry_cmp;
					} else {

						return (A->material < B->material);
					}
				} else {

					return A->material->shader_cache < B->material->shader_cache;
				}
			}
		};

		void sort_mat_geom() {

			SortArray<Element *, SortMatGeom> sorter;
			sorter.sort(elements, element_count);
		}

		struct SortMatLight {

			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {

				if (A->geometry_cmp == B->geometry_cmp) {

					if (A->material == B->material) {

						return A->light < B->light;
					} else {

						return (A->material < B->material);
					}
				} else {

					return (A->geometry_cmp < B->geometry_cmp);
				}
			}
		};

		void sort_mat_light() {

			SortArray<Element *, SortMatLight> sorter;
			sorter.sort(elements, element_count);
		}

		struct SortMatLightType {

			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {

				if (A->light_type == B->light_type) {
					if (A->material->shader_cache == B->material->shader_cache) {
						if (A->material == B->material) {

							return (A->geometry_cmp < B->geometry_cmp);
						} else {

							return (A->material < B->material);
						}
					} else {

						return (A->material->shader_cache < B->material->shader_cache);
					}
				} else {

					return A->light_type < B->light_type;
				}
			}
		};

		void sort_mat_light_type() {

			SortArray<Element *, SortMatLightType> sorter;
			sorter.sort(elements, element_count);
		}

		struct SortMatLightTypeFlags {

			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {

				if (A->sort_key == B->sort_key) {
					if (A->material->shader_cache == B->material->shader_cache) {
						if (A->material == B->material) {

							return (A->geometry_cmp < B->geometry_cmp);
						} else {

							return (A->material < B->material);
						}
					} else {

						return (A->material->shader_cache < B->material->shader_cache);
					}
				} else {

					return A->sort_key < B->sort_key; //one is null and one is not
				}
			}
		};

		void sort_mat_light_type_flags() {

			SortArray<Element *, SortMatLightTypeFlags> sorter;
			sorter.sort(elements, element_count);
		}
		_FORCE_INLINE_ Element *add_element() {

			if (element_count >= max_elements)
				return NULL;
			elements[element_count] = &_elements[element_count];
			return elements[element_count++];
		}

		void init() {

			element_count = 0;
			elements = memnew_arr(Element *, max_elements);
			_elements = memnew_arr(Element, max_elements);
			for (int i = 0; i < max_elements; i++)
				elements[i] = &_elements[i]; // assign elements
		}

		RenderList() {
		}
		~RenderList() {
			memdelete_arr(elements);
			memdelete_arr(_elements);
		}
	};

	RenderList opaque_render_list;
	RenderList alpha_render_list;

	RID default_material;

	CameraMatrix camera_projection;
	Transform camera_transform;
	Transform camera_transform_inverse;
	float camera_z_near;
	float camera_z_far;
	Size2 camera_vp_size;
	bool camera_ortho;
	Set<String> extensions;
	bool texscreen_copied;
	bool texscreen_used;

	Plane camera_plane;

	void _add_geometry(const Geometry *p_geometry, const InstanceData *p_instance, const Geometry *p_geometry_cmp, const GeometryOwner *p_owner, int p_material = -1);
	void _render_list_forward(RenderList *p_render_list, const Transform &p_view_transform, const Transform &p_view_transform_inverse, const CameraMatrix &p_projection, bool p_reverse_cull = false, bool p_fragment_light = false, bool p_alpha_pass = false);

	//void _setup_light(LightInstance* p_instance, int p_idx);
	void _setup_light(uint16_t p_light);

	_FORCE_INLINE_ void _setup_shader_params(const Material *p_material);
	bool _setup_material(const Geometry *p_geometry, const Material *p_material, bool p_no_const_light, bool p_opaque_pass);
	void _setup_skeleton(const Skeleton *p_skeleton);

	Error _setup_geometry(const Geometry *p_geometry, const Material *p_material, const Skeleton *p_skeleton, const float *p_morphs);
	void _render(const Geometry *p_geometry, const Material *p_material, const Skeleton *p_skeleton, const GeometryOwner *p_owner, const Transform &p_xform);

	/***********/
	/* SHADOWS */
	/***********/

	struct ShadowBuffer {

		int size;
		GLuint fbo;
		GLuint rbo;
		GLuint depth;
		GLuint rgba; //for older devices
#if 0
		GLuint fbo_blur;
		GLuint rbo_blur;
		GLuint blur;
#endif

		LightInstance *owner;
		bool init(int p_size, bool p_use_depth);
		ShadowBuffer() {
			size = 0;
			depth = 0;
			owner = NULL;
		}
	};

	Vector<ShadowBuffer> near_shadow_buffers;
	ShadowBuffer blur_shadow_buffer;

	Vector<ShadowBuffer> far_shadow_buffers;

	LightInstance *shadow;
	int shadow_pass;

	float shadow_near_far_split_size_ratio;
	bool _allocate_shadow_buffers(LightInstance *p_instance, Vector<ShadowBuffer> &p_buffers);
	void _debug_draw_shadow(GLuint tex, const Rect2 &p_rect);
	void _debug_draw_shadows_type(Vector<ShadowBuffer> &p_shadows, Point2 &ofs);
	void _debug_shadows();
	void _debug_luminances();
	void _debug_samplers();

	/***********/
	/*  FBOs   */
	/***********/

	struct FrameBuffer {

		GLuint fbo;
		GLuint color;
		GLuint depth;

		int width, height;
		int scale;
		bool active;

		int blur_size;

		struct Blur {

			GLuint fbo;
			GLuint color;

			Blur() {
				fbo = 0;
				color = 0;
			}
		} blur[3];

		struct Luminance {

			int size;
			GLuint fbo;
			GLuint color;

			Luminance() {
				fbo = 0;
				color = 0;
				size = 0;
			}
		};

		Vector<Luminance> luminance;

		GLuint sample_fbo;
		GLuint sample_color;

		FrameBuffer() {
			blur_size = 0;
		}

	} framebuffer;

	void _update_framebuffer();
	void _process_glow_and_bloom();
	//void _update_blur_buffer();

	/*********/
	/* FRAME */
	/*********/

	struct _Rinfo {

		int texture_mem;
		int vertex_count;
		int object_count;
		int mat_change_count;
		int surface_count;
		int shader_change_count;
		int ci_draw_commands;
		int draw_calls;

	} _rinfo;

	/*******************/
	/* CANVAS OCCLUDER */
	/*******************/

	struct CanvasOccluder {

		GLuint vertex_id; // 0 means, unconfigured
		GLuint index_id; // 0 means, unconfigured
		DVector<Vector2> lines;
		int len;
	};

	RID_Owner<CanvasOccluder> canvas_occluder_owner;

	/***********************/
	/* CANVAS LIGHT SHADOW */
	/***********************/

	struct CanvasLightShadow {

		int size;
		int height;
		GLuint fbo;
		GLuint rbo;
		GLuint depth;
		GLuint rgba; //for older devices

		GLuint blur;
	};

	RID_Owner<CanvasLightShadow> canvas_light_shadow_owner;

	RID canvas_shadow_blur;

	/* ETC */

	RenderTarget *current_rt;
	bool current_rt_transparent;
	bool current_rt_vflip;
	ViewportData *current_vd;

	GLuint white_tex;
	RID canvas_tex;
	float canvas_opacity;
	Color canvas_modulate;
	bool canvas_use_modulate;
	bool uses_texpixel_size;
	bool rebind_texpixel_size;
	Transform canvas_transform;
	ShaderMaterial *canvas_last_material;
	bool canvas_texscreen_used;
	Vector2 normal_flip;
	_FORCE_INLINE_ void _canvas_normal_set_flip(const Vector2 &p_flip);

	_FORCE_INLINE_ Texture *_bind_canvas_texture(const RID &p_texture);
	VS::MaterialBlendMode canvas_blend_mode;

	int _setup_geometry_vinfo;

	bool pack_arrays;
	bool keep_copies;
	bool use_reload_hooks;
	bool cull_front;
	bool lights_use_shadow;
	_FORCE_INLINE_ void _set_cull(bool p_front, bool p_reverse_cull = false);
	_FORCE_INLINE_ Color _convert_color(const Color &p_color);

	void _process_glow_bloom();
	void _process_hdr();
	void _draw_tex_bg();

	bool using_canvas_bg;
	Size2 window_size;
	VS::ViewportRect viewport;
	double last_time;
	double time_delta;
	uint64_t frame;
	uint64_t scene_pass;
	bool draw_next_frame;
	Environment *current_env;
	VS::ScenarioDebugMode current_debug;
	RID overdraw_material;
	float shader_time_rollback;

	mutable MaterialShaderGLES2 material_shader;
	mutable CanvasShaderGLES2 canvas_shader;
	BlurShaderGLES2 blur_shader;
	CopyShaderGLES2 copy_shader;
	mutable CanvasShadowShaderGLES2 canvas_shadow_shader;

	mutable ShaderCompilerGLES2 shader_precompiler;

	void _draw_primitive(int p_points, const Vector3 *p_vertices, const Vector3 *p_normals, const Color *p_colors, const Vector3 *p_uvs, const Plane *p_tangents = NULL, int p_instanced = 1);
	_FORCE_INLINE_ void _draw_gui_primitive(int p_points, const Vector2 *p_vertices, const Color *p_colors, const Vector2 *p_uvs);
	_FORCE_INLINE_ void _draw_gui_primitive2(int p_points, const Vector2 *p_vertices, const Color *p_colors, const Vector2 *p_uvs, const Vector2 *p_uvs2);
	void _draw_textured_quad(const Rect2 &p_rect, const Rect2 &p_src_region, const Size2 &p_tex_size, bool p_h_flip = false, bool p_v_flip = false, bool p_transpose = false);
	void _draw_quad(const Rect2 &p_rect);
	void _copy_screen_quad();
	void _copy_to_texscreen();

	bool _test_depth_shadow_buffer();

	Vector3 chunk_vertex;
	Vector3 chunk_normal;
	Plane chunk_tangent;
	Color chunk_color;
	Vector2 chunk_uv;
	Vector2 chunk_uv2;
	GLuint tc0_id_cache;
	GLuint tc0_idx;

	template <bool use_normalmap>
	_FORCE_INLINE_ void _canvas_item_render_commands(CanvasItem *p_item, CanvasItem *current_clip, bool &reclip);
	_FORCE_INLINE_ void _canvas_item_setup_shader_params(ShaderMaterial *material, Shader *p_shader);
	_FORCE_INLINE_ void _canvas_item_setup_shader_uniforms(ShaderMaterial *material, Shader *p_shader);

public:
	/* TEXTURE API */

	virtual RID texture_create();
	virtual void texture_allocate(RID p_texture, int p_width, int p_height, Image::Format p_format, uint32_t p_flags = VS::TEXTURE_FLAGS_DEFAULT);
	virtual void texture_set_data(RID p_texture, const Image &p_image, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT);
	virtual Image texture_get_data(RID p_texture, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT) const;
	virtual void texture_set_flags(RID p_texture, uint32_t p_flags);
	virtual uint32_t texture_get_flags(RID p_texture) const;
	virtual Image::Format texture_get_format(RID p_texture) const;
	virtual uint32_t texture_get_width(RID p_texture) const;
	virtual uint32_t texture_get_height(RID p_texture) const;
	virtual bool texture_has_alpha(RID p_texture) const;
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height);
	virtual void texture_set_reload_hook(RID p_texture, ObjectID p_owner, const StringName &p_function) const;

	virtual void texture_set_path(RID p_texture, const String &p_path);
	virtual String texture_get_path(RID p_texture) const;
	virtual void texture_debug_usage(List<VS::TextureInfo> *r_info);

	virtual void texture_set_shrink_all_x2_on_set_data(bool p_enable);

	GLuint _texture_get_name(RID p_tex);

	/* SHADER API */

	virtual RID shader_create(VS::ShaderMode p_mode = VS::SHADER_MATERIAL);

	virtual void shader_set_mode(RID p_shader, VS::ShaderMode p_mode);
	virtual VS::ShaderMode shader_get_mode(RID p_shader) const;

	virtual void shader_set_code(RID p_shader, const String &p_vertex, const String &p_fragment, const String &p_light, int p_vertex_ofs = 0, int p_fragment_ofs = 0, int p_light_ofs = 0);
	virtual String shader_get_fragment_code(RID p_shader) const;
	virtual String shader_get_vertex_code(RID p_shader) const;
	virtual String shader_get_light_code(RID p_shader) const;

	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const;

	virtual void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture);
	virtual RID shader_get_default_texture_param(RID p_shader, const StringName &p_name) const;

	virtual Variant shader_get_default_param(RID p_shader, const StringName &p_name);

	/* COMMON MATERIAL API */

	virtual RID material_create();

	virtual void material_set_shader(RID p_shader_material, RID p_shader);
	virtual RID material_get_shader(RID p_shader_material) const;

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value);
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const;

	virtual void material_set_flag(RID p_material, VS::MaterialFlag p_flag, bool p_enabled);
	virtual bool material_get_flag(RID p_material, VS::MaterialFlag p_flag) const;

	virtual void material_set_depth_draw_mode(RID p_material, VS::MaterialDepthDrawMode p_mode);
	virtual VS::MaterialDepthDrawMode material_get_depth_draw_mode(RID p_material) const;

	virtual void material_set_blend_mode(RID p_material, VS::MaterialBlendMode p_mode);
	virtual VS::MaterialBlendMode material_get_blend_mode(RID p_material) const;

	virtual void material_set_line_width(RID p_material, float p_line_width);
	virtual float material_get_line_width(RID p_material) const;

	/* MESH API */

	virtual RID mesh_create();

	virtual void mesh_add_surface(RID p_mesh, VS::PrimitiveType p_primitive, const Array &p_arrays, const Array &p_blend_shapes = Array(), bool p_alpha_sort = false);
	virtual Array mesh_get_surface_arrays(RID p_mesh, int p_surface) const;
	virtual Array mesh_get_surface_morph_arrays(RID p_mesh, int p_surface) const;
	virtual void mesh_add_custom_surface(RID p_mesh, const Variant &p_dat);

	virtual void mesh_set_morph_target_count(RID p_mesh, int p_amount);
	virtual int mesh_get_morph_target_count(RID p_mesh) const;

	virtual void mesh_set_morph_target_mode(RID p_mesh, VS::MorphTargetMode p_mode);
	virtual VS::MorphTargetMode mesh_get_morph_target_mode(RID p_mesh) const;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material, bool p_owned = false);
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const;

	virtual int mesh_surface_get_array_len(RID p_mesh, int p_surface) const;
	virtual int mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const;
	virtual uint32_t mesh_surface_get_format(RID p_mesh, int p_surface) const;
	virtual VS::PrimitiveType mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const;

	virtual void mesh_remove_surface(RID p_mesh, int p_index);
	virtual int mesh_get_surface_count(RID p_mesh) const;

	virtual AABB mesh_get_aabb(RID p_mesh, RID p_skeleton = RID()) const;

	virtual void mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb);
	virtual AABB mesh_get_custom_aabb(RID p_mesh) const;

	/* MULTIMESH API */

	virtual RID multimesh_create();

	virtual void multimesh_set_instance_count(RID p_multimesh, int p_count);
	virtual int multimesh_get_instance_count(RID p_multimesh) const;

	virtual void multimesh_set_mesh(RID p_multimesh, RID p_mesh);
	virtual void multimesh_set_aabb(RID p_multimesh, const AABB &p_aabb);
	virtual void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform);
	virtual void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color);

	virtual RID multimesh_get_mesh(RID p_multimesh) const;
	virtual AABB multimesh_get_aabb(RID p_multimesh) const;

	virtual Transform multimesh_instance_get_transform(RID p_multimesh, int p_index) const;
	virtual Color multimesh_instance_get_color(RID p_multimesh, int p_index) const;

	virtual void multimesh_set_visible_instances(RID p_multimesh, int p_visible);
	virtual int multimesh_get_visible_instances(RID p_multimesh) const;

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
	virtual AABB immediate_get_aabb(RID p_immediate) const;
	virtual void immediate_set_material(RID p_immediate, RID p_material);
	virtual RID immediate_get_material(RID p_immediate) const;

	/* PARTICLES API */

	virtual RID particles_create();

	virtual void particles_set_amount(RID p_particles, int p_amount);
	virtual int particles_get_amount(RID p_particles) const;

	virtual void particles_set_emitting(RID p_particles, bool p_emitting);
	virtual bool particles_is_emitting(RID p_particles) const;

	virtual void particles_set_visibility_aabb(RID p_particles, const AABB &p_visibility);
	virtual AABB particles_get_visibility_aabb(RID p_particles) const;

	virtual void particles_set_emission_half_extents(RID p_particles, const Vector3 &p_half_extents);
	virtual Vector3 particles_get_emission_half_extents(RID p_particles) const;

	virtual void particles_set_emission_base_velocity(RID p_particles, const Vector3 &p_base_velocity);
	virtual Vector3 particles_get_emission_base_velocity(RID p_particles) const;

	virtual void particles_set_emission_points(RID p_particles, const DVector<Vector3> &p_points);
	virtual DVector<Vector3> particles_get_emission_points(RID p_particles) const;

	virtual void particles_set_gravity_normal(RID p_particles, const Vector3 &p_normal);
	virtual Vector3 particles_get_gravity_normal(RID p_particles) const;

	virtual void particles_set_variable(RID p_particles, VS::ParticleVariable p_variable, float p_value);
	virtual float particles_get_variable(RID p_particles, VS::ParticleVariable p_variable) const;

	virtual void particles_set_randomness(RID p_particles, VS::ParticleVariable p_variable, float p_randomness);
	virtual float particles_get_randomness(RID p_particles, VS::ParticleVariable p_variable) const;

	virtual void particles_set_color_phase_pos(RID p_particles, int p_phase, float p_pos);
	virtual float particles_get_color_phase_pos(RID p_particles, int p_phase) const;

	virtual void particles_set_color_phases(RID p_particles, int p_phases);
	virtual int particles_get_color_phases(RID p_particles) const;

	virtual void particles_set_color_phase_color(RID p_particles, int p_phase, const Color &p_color);
	virtual Color particles_get_color_phase_color(RID p_particles, int p_phase) const;

	virtual void particles_set_attractors(RID p_particles, int p_attractors);
	virtual int particles_get_attractors(RID p_particles) const;

	virtual void particles_set_attractor_pos(RID p_particles, int p_attractor, const Vector3 &p_pos);
	virtual Vector3 particles_get_attractor_pos(RID p_particles, int p_attractor) const;

	virtual void particles_set_attractor_strength(RID p_particles, int p_attractor, float p_force);
	virtual float particles_get_attractor_strength(RID p_particles, int p_attractor) const;

	virtual void particles_set_material(RID p_particles, RID p_material, bool p_owned = false);
	virtual RID particles_get_material(RID p_particles) const;

	virtual AABB particles_get_aabb(RID p_particles) const;

	virtual void particles_set_height_from_velocity(RID p_particles, bool p_enable);
	virtual bool particles_has_height_from_velocity(RID p_particles) const;

	virtual void particles_set_use_local_coordinates(RID p_particles, bool p_enable);
	virtual bool particles_is_using_local_coordinates(RID p_particles) const;

	/* SKELETON API */

	virtual RID skeleton_create();
	virtual void skeleton_resize(RID p_skeleton, int p_bones);
	virtual int skeleton_get_bone_count(RID p_skeleton) const;
	virtual void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform);
	virtual Transform skeleton_bone_get_transform(RID p_skeleton, int p_bone);

	/* LIGHT API */

	virtual RID light_create(VS::LightType p_type);
	virtual VS::LightType light_get_type(RID p_light) const;

	virtual void light_set_color(RID p_light, VS::LightColor p_type, const Color &p_color);
	virtual Color light_get_color(RID p_light, VS::LightColor p_type) const;

	virtual void light_set_shadow(RID p_light, bool p_enabled);
	virtual bool light_has_shadow(RID p_light) const;

	virtual void light_set_volumetric(RID p_light, bool p_enabled);
	virtual bool light_is_volumetric(RID p_light) const;

	virtual void light_set_projector(RID p_light, RID p_texture);
	virtual RID light_get_projector(RID p_light) const;

	virtual void light_set_var(RID p_light, VS::LightParam p_var, float p_value);
	virtual float light_get_var(RID p_light, VS::LightParam p_var) const;

	virtual void light_set_operator(RID p_light, VS::LightOp p_op);
	virtual VS::LightOp light_get_operator(RID p_light) const;

	virtual void light_omni_set_shadow_mode(RID p_light, VS::LightOmniShadowMode p_mode);
	virtual VS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) const;

	virtual void light_directional_set_shadow_mode(RID p_light, VS::LightDirectionalShadowMode p_mode);
	virtual VS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) const;
	virtual void light_directional_set_shadow_param(RID p_light, VS::LightDirectionalShadowParam p_param, float p_value);
	virtual float light_directional_get_shadow_param(RID p_light, VS::LightDirectionalShadowParam p_param) const;

	virtual AABB light_get_aabb(RID p_poly) const;

	virtual RID light_instance_create(RID p_light);
	virtual void light_instance_set_transform(RID p_light_instance, const Transform &p_transform);

	virtual ShadowType light_instance_get_shadow_type(RID p_light_instance, bool p_far = false) const;
	virtual int light_instance_get_shadow_passes(RID p_light_instance) const;
	virtual bool light_instance_get_pssm_shadow_overlap(RID p_light_instance) const;
	virtual void light_instance_set_shadow_transform(RID p_light_instance, int p_index, const CameraMatrix &p_camera, const Transform &p_transform, float p_split_near = 0, float p_split_far = 0);
	virtual int light_instance_get_shadow_size(RID p_light_instance, int p_index = 0) const;

	virtual void shadow_clear_near();
	virtual bool shadow_allocate_near(RID p_light);
	virtual bool shadow_allocate_far(RID p_light);

	/* SHADOW */

	virtual RID particles_instance_create(RID p_particles);
	virtual void particles_instance_set_transform(RID p_particles_instance, const Transform &p_transform);

	/* VIEWPORT */

	virtual RID viewport_data_create();

	virtual RID render_target_create();
	virtual void render_target_set_size(RID p_render_target, int p_width, int p_height);
	virtual RID render_target_get_texture(RID p_render_target) const;
	virtual bool render_target_renedered_in_frame(RID p_render_target);

	/* RENDER API */
	/* all calls (inside begin/end shadow) are always warranted to be in the following order: */

	virtual void begin_frame();

	virtual void set_viewport(const VS::ViewportRect &p_viewport);
	virtual void set_render_target(RID p_render_target, bool p_transparent_bg = false, bool p_vflip = false);
	virtual void clear_viewport(const Color &p_color);
	virtual void capture_viewport(Image *r_capture);

	virtual void begin_scene(RID p_viewport_data, RID p_env, VS::ScenarioDebugMode p_debug);

	virtual void begin_shadow_map(RID p_light_instance, int p_shadow_pass);

	virtual void set_camera(const Transform &p_world, const CameraMatrix &p_projection, bool p_ortho_hint);

	virtual void add_light(RID p_light_instance); ///< all "add_light" calls happen before add_geometry calls

	typedef Map<StringName, Variant> ParamOverrideMap;

	virtual void add_mesh(const RID &p_mesh, const InstanceData *p_data);
	virtual void add_multimesh(const RID &p_multimesh, const InstanceData *p_data);
	virtual void add_immediate(const RID &p_immediate, const InstanceData *p_data);
	virtual void add_particles(const RID &p_particle_instance, const InstanceData *p_data);

	virtual void end_scene();
	virtual void end_shadow_map();

	virtual void end_frame();

	/* CANVAS API */

	virtual void begin_canvas_bg();

	virtual void canvas_begin();
	virtual void canvas_disable_blending();

	virtual void canvas_set_opacity(float p_opacity);
	virtual void canvas_set_blend_mode(VS::MaterialBlendMode p_mode);
	virtual void canvas_begin_rect(const Transform2D &p_transform);
	virtual void canvas_set_clip(bool p_clip, const Rect2 &p_rect);
	virtual void canvas_end_rect();
	virtual void canvas_draw_line(const Point2 &p_from, const Point2 &p_to, const Color &p_color, float p_width, bool p_antialiased);
	virtual void canvas_draw_rect(const Rect2 &p_rect, int p_flags, const Rect2 &p_source, RID p_texture, const Color &p_modulate);
	virtual void canvas_draw_style_box(const Rect2 &p_rect, const Rect2 &p_src_region, RID p_texture, const float *p_margins, bool p_draw_center = true, const Color &p_modulate = Color(1, 1, 1));
	virtual void canvas_draw_primitive(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, RID p_texture, float p_width);
	virtual void canvas_draw_polygon(int p_vertex_count, const int *p_indices, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, const RID &p_texture, bool p_singlecolor);
	virtual void canvas_set_transform(const Transform2D &p_transform);

	virtual void canvas_render_items(CanvasItem *p_item_list, int p_z, const Color &p_modulate, CanvasLight *p_light);
	virtual void canvas_debug_viewport_shadows(CanvasLight *p_lights_with_shadow);

	/* CANVAS LIGHT SHADOW */

	//buffer
	virtual RID canvas_light_shadow_buffer_create(int p_width);
	virtual void canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, CanvasLightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache);

	//occluder
	virtual RID canvas_light_occluder_create();
	virtual void canvas_light_occluder_set_polylines(RID p_occluder, const DVector<Vector2> &p_lines);

	/* ENVIRONMENT */

	virtual RID environment_create();

	virtual void environment_set_background(RID p_env, VS::EnvironmentBG p_bg);
	virtual VS::EnvironmentBG environment_get_background(RID p_env) const;

	virtual void environment_set_background_param(RID p_env, VS::EnvironmentBGParam p_param, const Variant &p_value);
	virtual Variant environment_get_background_param(RID p_env, VS::EnvironmentBGParam p_param) const;

	virtual void environment_set_enable_fx(RID p_env, VS::EnvironmentFx p_effect, bool p_enabled);
	virtual bool environment_is_fx_enabled(RID p_env, VS::EnvironmentFx p_effect) const;

	virtual void environment_fx_set_param(RID p_env, VS::EnvironmentFxParam p_param, const Variant &p_value);
	virtual Variant environment_fx_get_param(RID p_env, VS::EnvironmentFxParam p_param) const;

	/* SAMPLED LIGHT */
	virtual RID sampled_light_dp_create(int p_width, int p_height);
	virtual void sampled_light_dp_update(RID p_sampled_light, const Color *p_data, float p_multiplier);

	/*MISC*/

	virtual bool is_texture(const RID &p_rid) const;
	virtual bool is_material(const RID &p_rid) const;
	virtual bool is_mesh(const RID &p_rid) const;
	virtual bool is_immediate(const RID &p_rid) const;
	virtual bool is_multimesh(const RID &p_rid) const;
	virtual bool is_particles(const RID &p_beam) const;

	virtual bool is_light(const RID &p_rid) const;
	virtual bool is_light_instance(const RID &p_rid) const;
	virtual bool is_particles_instance(const RID &p_rid) const;
	virtual bool is_skeleton(const RID &p_rid) const;
	virtual bool is_environment(const RID &p_rid) const;
	virtual bool is_shader(const RID &p_rid) const;

	virtual bool is_canvas_light_occluder(const RID &p_rid) const;

	virtual void free(const RID &p_rid);

	virtual void init();
	virtual void finish();

	virtual int get_render_info(VS::RenderInfo p_info);

	void set_base_framebuffer(GLuint p_id, Vector2 p_size = Vector2(0, 0));

	virtual void flush_frame(); //not necessary in most cases
	void set_extensions(const char *p_strings);

	virtual bool needs_to_draw_next_frame() const;

	void set_use_framebuffers(bool p_use);
	void reload_vram();

	virtual bool has_feature(VS::Features p_feature) const;

	virtual void restore_framebuffer();

	static RasterizerGLES2 *get_singleton();

	virtual void set_force_16_bits_fbo(bool p_force);

	RasterizerGLES2(bool p_compress_arrays = false, bool p_keep_ram_copy = true, bool p_default_fragment_lighting = true, bool p_use_reload_hooks = false);
	virtual ~RasterizerGLES2();
};

#endif
#endif
