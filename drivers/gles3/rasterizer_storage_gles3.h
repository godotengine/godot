/*************************************************************************/
/*  rasterizer_storage_gles3.h                                           */
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
#ifndef RASTERIZERSTORAGEGLES3_H
#define RASTERIZERSTORAGEGLES3_H

#include "self_list.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual/shader_language.h"
#include "shader_compiler_gles3.h"
#include "shader_gles3.h"
#include "shaders/blend_shape.glsl.h"
#include "shaders/canvas.glsl.h"
#include "shaders/copy.glsl.h"
#include "shaders/cubemap_filter.glsl.h"
#include "shaders/particles.glsl.h"

class RasterizerCanvasGLES3;
class RasterizerSceneGLES3;

#define _TEXTURE_SRGB_DECODE_EXT 0x8A48
#define _DECODE_EXT 0x8A49
#define _SKIP_DECODE_EXT 0x8A4A

class RasterizerStorageGLES3 : public RasterizerStorage {
public:
	RasterizerCanvasGLES3 *canvas;
	RasterizerSceneGLES3 *scene;
	static GLuint system_fbo; //on some devices, such as apple, screen is rendered to yet another fbo.

	enum RenderArchitecture {
		RENDER_ARCH_MOBILE,
		RENDER_ARCH_DESKTOP,
	};

	struct Config {

		RenderArchitecture render_arch;

		bool shrink_textures_x2;
		bool use_fast_texture_filter;
		bool use_anisotropic_filter;

		bool s3tc_supported;
		bool latc_supported;
		bool bptc_supported;
		bool etc_supported;
		bool etc2_supported;
		bool pvrtc_supported;

		bool hdr_supported;

		bool srgb_decode_supported;

		bool use_rgba_2d_shadows;

		float anisotropic_level;

		int max_texture_image_units;
		int max_texture_size;

		Set<String> extensions;

		bool keep_original_textures;
	} config;

	mutable struct Shaders {

		CopyShaderGLES3 copy;

		ShaderCompilerGLES3 compiler;

		CubemapFilterShaderGLES3 cubemap_filter;

		BlendShapeShaderGLES3 blend_shapes;

		ParticlesShaderGLES3 particles;

		ShaderCompilerGLES3::IdentifierActions actions_canvas;
		ShaderCompilerGLES3::IdentifierActions actions_scene;
		ShaderCompilerGLES3::IdentifierActions actions_particles;
	} shaders;

	struct Resources {

		GLuint white_tex;
		GLuint black_tex;
		GLuint normal_tex;
		GLuint aniso_tex;

		GLuint quadie;
		GLuint quadie_array;

		GLuint transform_feedback_buffers[2];
		GLuint transform_feedback_array;

	} resources;

	struct Info {

		uint64_t texture_mem;

		uint32_t render_object_count;
		uint32_t render_material_switch_count;
		uint32_t render_surface_switch_count;
		uint32_t render_shader_rebind_count;
		uint32_t render_vertices_count;

	} info;

	/////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////DATA///////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////

	struct Instantiable : public RID_Data {

		SelfList<RasterizerScene::InstanceBase>::List instance_list;

		_FORCE_INLINE_ void instance_change_notify() {

			SelfList<RasterizerScene::InstanceBase> *instances = instance_list.first();
			while (instances) {

				instances->self()->base_changed();
				instances = instances->next();
			}
		}

		_FORCE_INLINE_ void instance_material_change_notify() {

			SelfList<RasterizerScene::InstanceBase> *instances = instance_list.first();
			while (instances) {

				instances->self()->base_material_changed();
				instances = instances->next();
			}
		}

		_FORCE_INLINE_ void instance_remove_deps() {
			SelfList<RasterizerScene::InstanceBase> *instances = instance_list.first();
			while (instances) {

				SelfList<RasterizerScene::InstanceBase> *next = instances->next();
				instances->self()->base_removed();
				instances = next;
			}
		}

		Instantiable() {}
		virtual ~Instantiable() {
		}
	};

	struct GeometryOwner : public Instantiable {

		virtual ~GeometryOwner() {}
	};
	struct Geometry : Instantiable {

		enum Type {
			GEOMETRY_INVALID,
			GEOMETRY_SURFACE,
			GEOMETRY_IMMEDIATE,
			GEOMETRY_MULTISURFACE,
		};

		Type type;
		RID material;
		uint64_t last_pass;
		uint32_t index;

		virtual void material_changed_notify() {}

		Geometry() {
			last_pass = 0;
			index = 0;
		}
	};

	/////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////API////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////

	/* TEXTURE API */

	struct RenderTarget;

	struct Texture : public RID_Data {

		String path;
		uint32_t flags;
		int width, height;
		int alloc_width, alloc_height;
		Image::Format format;

		GLenum target;
		GLenum gl_format_cache;
		GLenum gl_internal_format_cache;
		GLenum gl_type_cache;
		int data_size; //original data size, useful for retrieving back
		bool compressed;
		bool srgb;
		int total_data_size;
		bool ignore_mipmaps;

		int mipmaps;

		bool active;
		GLuint tex_id;

		bool using_srgb;

		uint16_t stored_cube_sides;

		RenderTarget *render_target;

		Image images[6];

		VisualServer::TextureDetectCallback detect_3d;
		void *detect_3d_ud;

		VisualServer::TextureDetectCallback detect_srgb;
		void *detect_srgb_ud;

		Texture() {

			using_srgb = false;
			stored_cube_sides = 0;
			ignore_mipmaps = false;
			render_target = NULL;
			flags = width = height = 0;
			tex_id = 0;
			data_size = 0;
			format = Image::FORMAT_L8;
			active = false;
			compressed = false;
			total_data_size = 0;
			target = GL_TEXTURE_2D;
			mipmaps = 0;
			detect_3d = NULL;
			detect_3d_ud = NULL;
			detect_srgb = NULL;
			detect_srgb_ud = NULL;
		}

		~Texture() {

			if (tex_id != 0) {

				glDeleteTextures(1, &tex_id);
			}
		}
	};

	mutable RID_Owner<Texture> texture_owner;

	Image _get_gl_image_and_format(const Image &p_image, Image::Format p_format, uint32_t p_flags, GLenum &r_gl_format, GLenum &r_gl_internal_format, GLenum &r_type, bool &r_compressed, bool &srgb);

	virtual RID texture_create();
	virtual void texture_allocate(RID p_texture, int p_width, int p_height, Image::Format p_format, uint32_t p_flags = VS::TEXTURE_FLAGS_DEFAULT);
	virtual void texture_set_data(RID p_texture, const Image &p_image, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT);
	virtual Image texture_get_data(RID p_texture, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT) const;
	virtual void texture_set_flags(RID p_texture, uint32_t p_flags);
	virtual uint32_t texture_get_flags(RID p_texture) const;
	virtual Image::Format texture_get_format(RID p_texture) const;
	virtual uint32_t texture_get_width(RID p_texture) const;
	virtual uint32_t texture_get_height(RID p_texture) const;
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height);

	virtual void texture_set_path(RID p_texture, const String &p_path);
	virtual String texture_get_path(RID p_texture) const;

	virtual void texture_set_shrink_all_x2_on_set_data(bool p_enable);

	virtual void texture_debug_usage(List<VS::TextureInfo> *r_info);

	virtual RID texture_create_radiance_cubemap(RID p_source, int p_resolution = -1) const;

	virtual void textures_keep_original(bool p_enable);

	virtual void texture_set_detect_3d_callback(RID p_texture, VisualServer::TextureDetectCallback p_callback, void *p_userdata);
	virtual void texture_set_detect_srgb_callback(RID p_texture, VisualServer::TextureDetectCallback p_callback, void *p_userdata);

	/* SKYBOX API */

	struct SkyBox : public RID_Data {

		RID cubemap;
		GLuint radiance;
		int radiance_size;
	};

	mutable RID_Owner<SkyBox> skybox_owner;

	virtual RID skybox_create();
	virtual void skybox_set_texture(RID p_skybox, RID p_cube_map, int p_radiance_size);

	/* SHADER API */

	struct Material;

	struct Shader : public RID_Data {

		RID self;

		VS::ShaderMode mode;
		ShaderGLES3 *shader;
		String code;
		SelfList<Material>::List materials;

		Map<StringName, ShaderLanguage::ShaderNode::Uniform> uniforms;
		Vector<uint32_t> ubo_offsets;
		uint32_t ubo_size;

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

			enum LightMode {
				LIGHT_MODE_NORMAL,
				LIGHT_MODE_UNSHADED,
				LIGHT_MODE_LIGHT_ONLY
			};

			int light_mode;

		} canvas_item;

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
			bool unshaded;
			bool ontop;
			bool uses_vertex;
			bool uses_discard;
			bool uses_sss;
			bool writes_modelview_or_projection;

		} spatial;

		struct Particles {

		} particles;

		bool uses_vertex_time;
		bool uses_fragment_time;

		Shader()
			: dirty_list(this) {

			shader = NULL;
			valid = false;
			custom_code_id = 0;
			version = 1;
		}
	};

	mutable SelfList<Shader>::List _shader_dirty_list;
	void _shader_make_dirty(Shader *p_shader);

	mutable RID_Owner<Shader> shader_owner;

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
		GLuint ubo_id;
		uint32_t ubo_size;
		Map<StringName, Variant> params;
		SelfList<Material> list;
		SelfList<Material> dirty_list;
		Vector<RID> textures;
		float line_width;

		uint32_t index;
		uint64_t last_pass;

		Map<Geometry *, int> geometry_owners;
		Map<RasterizerScene::InstanceBase *, int> instance_owners;

		bool can_cast_shadow_cache;
		bool is_animated_cache;

		Material()
			: list(this), dirty_list(this) {
			can_cast_shadow_cache = false;
			is_animated_cache = false;
			shader = NULL;
			line_width = 1.0;
			ubo_id = 0;
			ubo_size = 0;
			last_pass = 0;
		}
	};

	mutable SelfList<Material>::List _material_dirty_list;
	void _material_make_dirty(Material *p_material) const;
	void _material_add_geometry(RID p_material, Geometry *p_instantiable);
	void _material_remove_geometry(RID p_material, Geometry *p_instantiable);

	mutable RID_Owner<Material> material_owner;

	virtual RID material_create();

	virtual void material_set_shader(RID p_material, RID p_shader);
	virtual RID material_get_shader(RID p_material) const;

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value);
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const;

	virtual void material_set_line_width(RID p_material, float p_width);

	virtual bool material_is_animated(RID p_material);
	virtual bool material_casts_shadows(RID p_material);

	virtual void material_add_instance_owner(RID p_material, RasterizerScene::InstanceBase *p_instance);
	virtual void material_remove_instance_owner(RID p_material, RasterizerScene::InstanceBase *p_instance);

	void _update_material(Material *material);

	void update_dirty_materials();

	/* MESH API */

	struct Mesh;
	struct Surface : public Geometry {

		struct Attrib {

			bool enabled;
			bool integer;
			GLuint index;
			GLint size;
			GLenum type;
			GLboolean normalized;
			GLsizei stride;
			uint32_t offset;
		};

		Attrib attribs[VS::ARRAY_MAX];

		Mesh *mesh;
		uint32_t format;

		GLuint array_id;
		GLuint instancing_array_id;
		GLuint vertex_id;
		GLuint index_id;

		Vector<Rect3> skeleton_bone_aabb;
		Vector<bool> skeleton_bone_used;

		//bool packed;

		struct BlendShape {
			GLuint vertex_id;
			GLuint array_id;
		};

		Vector<BlendShape> blend_shapes;

		Rect3 aabb;

		int array_len;
		int index_array_len;
		int max_bone;

		int array_byte_size;
		int index_array_byte_size;

		VS::PrimitiveType primitive;

		bool active;

		virtual void material_changed_notify() {
			mesh->instance_material_change_notify();
			mesh->update_multimeshes();
		}

		Surface() {

			array_byte_size = 0;
			index_array_byte_size = 0;
			mesh = NULL;
			format = 0;
			array_id = 0;
			vertex_id = 0;
			index_id = 0;
			array_len = 0;
			type = GEOMETRY_SURFACE;
			primitive = VS::PRIMITIVE_POINTS;
			index_array_len = 0;
			active = false;
		}

		~Surface() {
		}
	};

	class MultiMesh;

	struct Mesh : public GeometryOwner {

		bool active;
		Vector<Surface *> surfaces;
		int blend_shape_count;
		VS::BlendShapeMode blend_shape_mode;
		Rect3 custom_aabb;
		mutable uint64_t last_pass;
		SelfList<MultiMesh>::List multimeshes;

		_FORCE_INLINE_ void update_multimeshes() {

			SelfList<MultiMesh> *mm = multimeshes.first();
			while (mm) {
				mm->self()->instance_material_change_notify();
				mm = mm->next();
			}
		}

		Mesh() {
			blend_shape_mode = VS::BLEND_SHAPE_MODE_NORMALIZED;
			blend_shape_count = 0;
			last_pass = 0;
			active = false;
		}
	};

	mutable RID_Owner<Mesh> mesh_owner;

	virtual RID mesh_create();

	virtual void mesh_add_surface(RID p_mesh, uint32_t p_format, VS::PrimitiveType p_primitive, const PoolVector<uint8_t> &p_array, int p_vertex_count, const PoolVector<uint8_t> &p_index_array, int p_index_count, const Rect3 &p_aabb, const Vector<PoolVector<uint8_t> > &p_blend_shapes = Vector<PoolVector<uint8_t> >(), const Vector<Rect3> &p_bone_aabbs = Vector<Rect3>());

	virtual void mesh_set_blend_shape_count(RID p_mesh, int p_amount);
	virtual int mesh_get_blend_shape_count(RID p_mesh) const;

	virtual void mesh_set_blend_shape_mode(RID p_mesh, VS::BlendShapeMode p_mode);
	virtual VS::BlendShapeMode mesh_get_blend_shape_mode(RID p_mesh) const;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material);
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const;

	virtual int mesh_surface_get_array_len(RID p_mesh, int p_surface) const;
	virtual int mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const;

	virtual PoolVector<uint8_t> mesh_surface_get_array(RID p_mesh, int p_surface) const;
	virtual PoolVector<uint8_t> mesh_surface_get_index_array(RID p_mesh, int p_surface) const;

	virtual uint32_t mesh_surface_get_format(RID p_mesh, int p_surface) const;
	virtual VS::PrimitiveType mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const;

	virtual Rect3 mesh_surface_get_aabb(RID p_mesh, int p_surface) const;
	virtual Vector<PoolVector<uint8_t> > mesh_surface_get_blend_shapes(RID p_mesh, int p_surface) const;
	virtual Vector<Rect3> mesh_surface_get_skeleton_aabb(RID p_mesh, int p_surface) const;

	virtual void mesh_remove_surface(RID p_mesh, int p_surface);
	virtual int mesh_get_surface_count(RID p_mesh) const;

	virtual void mesh_set_custom_aabb(RID p_mesh, const Rect3 &p_aabb);
	virtual Rect3 mesh_get_custom_aabb(RID p_mesh) const;

	virtual Rect3 mesh_get_aabb(RID p_mesh, RID p_skeleton) const;
	virtual void mesh_clear(RID p_mesh);

	void mesh_render_blend_shapes(Surface *s, float *p_weights);

	/* MULTIMESH API */

	struct MultiMesh : public GeometryOwner {
		RID mesh;
		int size;
		VS::MultimeshTransformFormat transform_format;
		VS::MultimeshColorFormat color_format;
		Vector<float> data;
		Rect3 aabb;
		SelfList<MultiMesh> update_list;
		SelfList<MultiMesh> mesh_list;
		GLuint buffer;
		int visible_instances;

		int xform_floats;
		int color_floats;

		bool dirty_aabb;
		bool dirty_data;

		MultiMesh()
			: update_list(this), mesh_list(this) {
			dirty_aabb = true;
			dirty_data = true;
			xform_floats = 0;
			color_floats = 0;
			visible_instances = -1;
			size = 0;
			buffer = 0;
			transform_format = VS::MULTIMESH_TRANSFORM_2D;
			color_format = VS::MULTIMESH_COLOR_NONE;
		}
	};

	mutable RID_Owner<MultiMesh> multimesh_owner;

	SelfList<MultiMesh>::List multimesh_update_list;

	void update_dirty_multimeshes();

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

	virtual Rect3 multimesh_get_aabb(RID p_multimesh) const;

	/* IMMEDIATE API */

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
		Rect3 aabb;

		Immediate() {
			type = GEOMETRY_IMMEDIATE;
			building = false;
		}
	};

	Vector3 chunk_vertex;
	Vector3 chunk_normal;
	Plane chunk_tangent;
	Color chunk_color;
	Vector2 chunk_uv;
	Vector2 chunk_uv2;

	mutable RID_Owner<Immediate> immediate_owner;

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
	virtual Rect3 immediate_get_aabb(RID p_immediate) const;

	/* SKELETON API */

	struct Skeleton : RID_Data {
		bool use_2d;
		int size;
		Vector<float> skel_texture;
		GLuint texture;
		SelfList<Skeleton> update_list;
		Set<RasterizerScene::InstanceBase *> instances; //instances using skeleton

		Skeleton()
			: update_list(this) {
			size = 0;

			use_2d = false;
			texture = 0;
		}
	};

	mutable RID_Owner<Skeleton> skeleton_owner;

	SelfList<Skeleton>::List skeleton_update_list;

	void update_dirty_skeletons();

	virtual RID skeleton_create();
	virtual void skeleton_allocate(RID p_skeleton, int p_bones, bool p_2d_skeleton = false);
	virtual int skeleton_get_bone_count(RID p_skeleton) const;
	virtual void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform);
	virtual Transform skeleton_bone_get_transform(RID p_skeleton, int p_bone) const;
	virtual void skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform);
	virtual Transform2D skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const;

	/* Light API */

	struct Light : Instantiable {

		VS::LightType type;
		float param[VS::LIGHT_PARAM_MAX];
		Color color;
		Color shadow_color;
		RID projector;
		bool shadow;
		bool negative;
		uint32_t cull_mask;
		VS::LightOmniShadowMode omni_shadow_mode;
		VS::LightOmniShadowDetail omni_shadow_detail;
		VS::LightDirectionalShadowMode directional_shadow_mode;
		bool directional_blend_splits;
		uint64_t version;
	};

	mutable RID_Owner<Light> light_owner;

	virtual RID light_create(VS::LightType p_type);

	virtual void light_set_color(RID p_light, const Color &p_color);
	virtual void light_set_param(RID p_light, VS::LightParam p_param, float p_value);
	virtual void light_set_shadow(RID p_light, bool p_enabled);
	virtual void light_set_shadow_color(RID p_light, const Color &p_color);
	virtual void light_set_projector(RID p_light, RID p_texture);
	virtual void light_set_negative(RID p_light, bool p_enable);
	virtual void light_set_cull_mask(RID p_light, uint32_t p_mask);

	virtual void light_omni_set_shadow_mode(RID p_light, VS::LightOmniShadowMode p_mode);
	virtual void light_omni_set_shadow_detail(RID p_light, VS::LightOmniShadowDetail p_detail);

	virtual void light_directional_set_shadow_mode(RID p_light, VS::LightDirectionalShadowMode p_mode);
	virtual void light_directional_set_blend_splits(RID p_light, bool p_enable);
	virtual bool light_directional_get_blend_splits(RID p_light) const;

	virtual VS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light);
	virtual VS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light);

	virtual bool light_has_shadow(RID p_light) const;

	virtual VS::LightType light_get_type(RID p_light) const;
	virtual float light_get_param(RID p_light, VS::LightParam p_param);
	virtual Color light_get_color(RID p_light);

	virtual Rect3 light_get_aabb(RID p_light) const;
	virtual uint64_t light_get_version(RID p_light) const;

	/* PROBE API */

	struct ReflectionProbe : Instantiable {

		VS::ReflectionProbeUpdateMode update_mode;
		float intensity;
		Color interior_ambient;
		float interior_ambient_energy;
		float interior_ambient_probe_contrib;
		float max_distance;
		Vector3 extents;
		Vector3 origin_offset;
		bool interior;
		bool box_projection;
		bool enable_shadows;
		uint32_t cull_mask;
	};

	mutable RID_Owner<ReflectionProbe> reflection_probe_owner;

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

	virtual Rect3 reflection_probe_get_aabb(RID p_probe) const;
	virtual VS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const;
	virtual uint32_t reflection_probe_get_cull_mask(RID p_probe) const;

	virtual Vector3 reflection_probe_get_extents(RID p_probe) const;
	virtual Vector3 reflection_probe_get_origin_offset(RID p_probe) const;
	virtual float reflection_probe_get_origin_max_distance(RID p_probe) const;
	virtual bool reflection_probe_renders_shadows(RID p_probe) const;

	/* ROOM API */

	virtual RID room_create();
	virtual void room_add_bounds(RID p_room, const PoolVector<Vector2> &p_convex_polygon, float p_height, const Transform &p_transform);
	virtual void room_clear_bounds(RID p_room);

	/* PORTAL API */

	// portals are only (x/y) points, forming a convex shape, which its clockwise
	// order points outside. (z is 0);

	virtual RID portal_create();
	virtual void portal_set_shape(RID p_portal, const Vector<Point2> &p_shape);
	virtual void portal_set_enabled(RID p_portal, bool p_enabled);
	virtual void portal_set_disable_distance(RID p_portal, float p_distance);
	virtual void portal_set_disabled_color(RID p_portal, const Color &p_color);

	/* GI PROBE API */

	struct GIProbe : public Instantiable {

		Rect3 bounds;
		Transform to_cell;
		float cell_size;

		int dynamic_range;
		float energy;
		float bias;
		float propagation;
		bool interior;
		bool compress;

		uint32_t version;

		PoolVector<int> dynamic_data;
	};

	mutable RID_Owner<GIProbe> gi_probe_owner;

	virtual RID gi_probe_create();

	virtual void gi_probe_set_bounds(RID p_probe, const Rect3 &p_bounds);
	virtual Rect3 gi_probe_get_bounds(RID p_probe) const;

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

	virtual void gi_probe_set_propagation(RID p_probe, float p_range);
	virtual float gi_probe_get_propagation(RID p_probe) const;

	virtual void gi_probe_set_interior(RID p_probe, bool p_enable);
	virtual bool gi_probe_is_interior(RID p_probe) const;

	virtual void gi_probe_set_compress(RID p_probe, bool p_enable);
	virtual bool gi_probe_is_compressed(RID p_probe) const;

	virtual uint32_t gi_probe_get_version(RID p_probe);

	struct GIProbeData : public RID_Data {

		int width;
		int height;
		int depth;
		int levels;
		GLuint tex_id;
		GIProbeCompression compression;

		GIProbeData() {
		}
	};

	mutable RID_Owner<GIProbeData> gi_probe_data_owner;

	virtual GIProbeCompression gi_probe_get_dynamic_data_get_preferred_compression() const;
	virtual RID gi_probe_dynamic_data_create(int p_width, int p_height, int p_depth, GIProbeCompression p_compression);
	virtual void gi_probe_dynamic_data_update(RID p_gi_probe_data, int p_depth_slice, int p_slice_count, int p_mipmap, const void *p_data);

	/* PARTICLES */

	struct Particles : public GeometryOwner {

		bool emitting;
		int amount;
		float lifetime;
		float pre_process_time;
		float explosiveness;
		float randomness;
		Rect3 custom_aabb;
		bool use_local_coords;
		RID process_material;

		VS::ParticlesDrawOrder draw_order;

		Vector<RID> draw_passes;

		GLuint particle_buffers[2];
		GLuint particle_vaos[2];

		GLuint particle_buffer_histories[2];
		GLuint particle_vao_histories[2];
		bool particle_valid_histories[2];
		bool histories_enabled;

		SelfList<Particles> particle_element;

		float phase;
		float prev_phase;
		uint64_t prev_ticks;

		uint32_t cycle_number;

		float speed_scale;

		int fixed_fps;
		bool fractional_delta;
		float frame_remainder;

		bool clear;

		Transform emission_transform;

		Particles()
			: particle_element(this) {
			cycle_number = 0;
			emitting = false;
			amount = 0;
			lifetime = 1.0;
			pre_process_time = 0.0;
			explosiveness = 0.0;
			randomness = 0.0;
			use_local_coords = true;
			fixed_fps = 0;
			fractional_delta = false;
			frame_remainder = 0;
			histories_enabled = false;
			speed_scale = 1.0;

			custom_aabb = Rect3(Vector3(-4, -4, -4), Vector3(8, 8, 8));

			draw_order = VS::PARTICLES_DRAW_ORDER_INDEX;
			particle_buffers[0] = 0;
			particle_buffers[1] = 0;

			prev_ticks = 0;

			clear = true;

			glGenBuffers(2, particle_buffers);
			glGenVertexArrays(2, particle_vaos);
		}

		~Particles() {

			glDeleteBuffers(2, particle_buffers);
			glDeleteVertexArrays(2, particle_vaos);
			if (histories_enabled) {
				glDeleteBuffers(2, particle_buffer_histories);
				glDeleteVertexArrays(2, particle_vao_histories);
			}
		}
	};

	SelfList<Particles>::List particle_update_list;

	void update_particles();

	mutable RID_Owner<Particles> particles_owner;

	virtual RID particles_create();

	virtual void particles_set_emitting(RID p_particles, bool p_emitting);
	virtual void particles_set_amount(RID p_particles, int p_amount);
	virtual void particles_set_lifetime(RID p_particles, float p_lifetime);
	virtual void particles_set_pre_process_time(RID p_particles, float p_time);
	virtual void particles_set_explosiveness_ratio(RID p_particles, float p_ratio);
	virtual void particles_set_randomness_ratio(RID p_particles, float p_ratio);
	virtual void particles_set_custom_aabb(RID p_particles, const Rect3 &p_aabb);
	virtual void particles_set_speed_scale(RID p_particles, float p_scale);
	virtual void particles_set_use_local_coordinates(RID p_particles, bool p_enable);
	virtual void particles_set_process_material(RID p_particles, RID p_material);
	virtual void particles_set_fixed_fps(RID p_particles, int p_fps);
	virtual void particles_set_fractional_delta(RID p_particles, bool p_enable);

	virtual void particles_set_draw_order(RID p_particles, VS::ParticlesDrawOrder p_order);

	virtual void particles_set_draw_passes(RID p_particles, int p_count);
	virtual void particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh);

	virtual void particles_request_process(RID p_particles);
	virtual Rect3 particles_get_current_aabb(RID p_particles);
	virtual Rect3 particles_get_aabb(RID p_particles) const;

	virtual void _particles_update_histories(Particles *particles);

	virtual void particles_set_emission_transform(RID p_particles, const Transform &p_transform);
	void _particles_process(Particles *p_particles, float p_delta);

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

		struct Buffers {
			GLuint fbo;
			GLuint depth;
			GLuint specular;
			GLuint diffuse;
			GLuint normal_rough;
			GLuint motion_sss;

			GLuint effect_fbo;
			GLuint effect;

		} buffers;

		struct Effects {

			struct MipMaps {

				struct Size {
					GLuint fbo;
					int width;
					int height;
				};

				Vector<Size> sizes;
				GLuint color;
				int levels;

				MipMaps() {
					color = 0;
					levels = 0;
				}
			};

			MipMaps mip_maps[2]; //first mipmap chain starts from full-screen
			//GLuint depth2; //depth for the second mipmap chain, in case of desiring upsampling

			struct SSAO {
				GLuint blur_fbo[2]; // blur fbo
				GLuint blur_red[2]; // 8 bits red buffer

				GLuint linear_depth;

				Vector<GLuint> depth_mipmap_fbos; //fbos for depth mipmapsla ver

				SSAO() {
					blur_fbo[0] = 0;
					blur_fbo[1] = 0;
					linear_depth = 0;
				}
			} ssao;

			Effects() {}

		} effects;

		struct Exposure {
			GLuint fbo;
			GLuint color;

			Exposure() { fbo = 0; }
		} exposure;

		uint64_t last_exposure_tick;

		int width, height;

		bool flags[RENDER_TARGET_FLAG_MAX];

		bool used_in_frame;
		VS::ViewportMSAA msaa;

		RID texture;

		RenderTarget() {

			msaa = VS::VIEWPORT_MSAA_DISABLED;
			width = 0;
			height = 0;
			depth = 0;
			fbo = 0;
			exposure.fbo = 0;
			buffers.fbo = 0;
			used_in_frame = false;

			flags[RENDER_TARGET_VFLIP] = false;
			flags[RENDER_TARGET_TRANSPARENT] = false;
			flags[RENDER_TARGET_NO_3D] = false;
			flags[RENDER_TARGET_HDR] = true;
			flags[RENDER_TARGET_NO_SAMPLING] = false;

			last_exposure_tick = 0;
		}
	};

	mutable RID_Owner<RenderTarget> render_target_owner;

	void _render_target_clear(RenderTarget *rt);
	void _render_target_allocate(RenderTarget *rt);

	virtual RID render_target_create();
	virtual void render_target_set_size(RID p_render_target, int p_width, int p_height);
	virtual RID render_target_get_texture(RID p_render_target) const;

	virtual void render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value);
	virtual bool render_target_renedered_in_frame(RID p_render_target);
	virtual void render_target_set_msaa(RID p_render_target, VS::ViewportMSAA p_msaa);

	/* CANVAS SHADOW */

	struct CanvasLightShadow : public RID_Data {

		int size;
		int height;
		GLuint fbo;
		GLuint depth;
		GLuint distance; //for older devices
	};

	RID_Owner<CanvasLightShadow> canvas_light_shadow_owner;

	virtual RID canvas_light_shadow_buffer_create(int p_width);

	/* LIGHT SHADOW MAPPING */

	struct CanvasOccluder : public RID_Data {

		GLuint vertex_id; // 0 means, unconfigured
		GLuint index_id; // 0 means, unconfigured
		PoolVector<Vector2> lines;
		int len;
	};

	RID_Owner<CanvasOccluder> canvas_occluder_owner;

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

	RasterizerStorageGLES3();
};

#endif // RASTERIZERSTORAGEGLES3_H
