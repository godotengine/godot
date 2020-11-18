/*************************************************************************/
/*  rasterizer_storage_gles2.h                                           */
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

#ifndef RASTERIZERSTORAGEGLES2_H
#define RASTERIZERSTORAGEGLES2_H

#include "drivers/gles_common/rasterizer_platforms.h"
#ifdef GLES2_BACKEND_ENABLED

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "drivers/gles_common/rasterizer_asserts.h"
#include "drivers/gles_common/rasterizer_common_stubs.h"
#include "drivers/gles_common/rasterizer_version.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/shader_language.h"
#include "shader_compiler_gles2.h"
#include "shader_gles2.h"

#include "shaders/copy.glsl.gen.h"
#include "shaders/cubemap_filter.glsl.gen.h"

class RasterizerCanvasGLES2;
class RasterizerSceneGLES2;

class RasterizerStorageGLES2 : public StubsStorage {
	friend class RasterizerGLES2;

	Thread::ID _main_thread_id = 0;
	bool _is_main_thread();

public:
	RasterizerCanvasGLES2 *canvas;
	RasterizerSceneGLES2 *scene;

	static GLuint system_fbo;

	struct Config {
		bool shrink_textures_x2;
		bool use_fast_texture_filter;
		bool use_skeleton_software;

		int max_vertex_texture_image_units;
		int max_texture_image_units;
		int max_texture_size;

		// TODO implement wireframe in GLES2
		// bool generate_wireframes;

		Set<String> extensions;

		bool float_texture_supported;
		bool s3tc_supported;
		bool etc1_supported;
		bool pvrtc_supported;
		bool rgtc_supported;
		bool bptc_supported;

		bool keep_original_textures;

		bool force_vertex_shading;

		bool use_rgba_2d_shadows;
		bool use_rgba_3d_shadows;

		bool support_32_bits_indices;
		bool support_write_depth;
		bool support_half_float_vertices;
		bool support_npot_repeat_mipmap;
		bool support_depth_texture;
		bool support_depth_cubemaps;

		bool support_shadow_cubemaps;

		bool multisample_supported;
		bool render_to_mipmap_supported;

		GLuint depth_internalformat;
		GLuint depth_type;
		GLuint depth_buffer_internalformat;

		// in some cases the legacy render didn't orphan. We will mark these
		// so the user can switch orphaning off for them.
		bool should_orphan;
	} config;

	struct Resources {
		GLuint white_tex;
		GLuint black_tex;
		GLuint normal_tex;
		GLuint aniso_tex;

		GLuint mipmap_blur_fbo;
		GLuint mipmap_blur_color;

		GLuint radical_inverse_vdc_cache_tex;
		bool use_rgba_2d_shadows;

		GLuint quadie;

		size_t skeleton_transform_buffer_size;
		GLuint skeleton_transform_buffer;
		LocalVector<float> skeleton_transform_cpu_buffer;

	} resources;

	mutable struct Shaders {
		ShaderCompilerGLES2 compiler;

		CopyShaderGLES2 copy;
		CubemapFilterShaderGLES2 cubemap_filter;

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
			uint32_t _2d_item_count;
			uint32_t _2d_draw_call_count;

			void reset() {
				object_count = 0;
				draw_call_count = 0;
				material_switch_count = 0;
				surface_switch_count = 0;
				shader_rebind_count = 0;
				vertices_count = 0;
				_2d_item_count = 0;
				_2d_draw_call_count = 0;
			}
		} render, render_final, snap;

		Info() :
				texture_mem(0),
				vertex_mem(0) {
			render.reset();
			render_final.reset();
		}

	} info;

	void bind_quad_array() const;

	/////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////DATA///////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////

	/*
	struct Instantiable {
		RID self;

		SelfList<InstanceBaseDependency>::List instance_list;

		_FORCE_INLINE_ void instance_change_notify(bool p_aabb, bool p_materials) {
			SelfList<InstanceBaseDependency> *instances = instance_list.first();
			while (instances) {
				instances->self()->base_changed(p_aabb, p_materials);
				instances = instances->next();
			}
		}

		_FORCE_INLINE_ void instance_remove_deps() {
			SelfList<InstanceBaseDependency> *instances = instance_list.first();

			while (instances) {
				instances->self()->base_removed();
				instances = instances->next();
			}
		}

		Instantiable() {}

		virtual ~Instantiable() {}
	};

	struct GeometryOwner : public Instantiable {
	};

	struct Geometry : public Instantiable {
		enum Type {
			GEOMETRY_INVALID,
			GEOMETRY_SURFACE,
			GEOMETRY_IMMEDIATE,
			GEOMETRY_MULTISURFACE
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
*/
	/////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////API////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////

	// TEXTURE API

	enum GLESTextureFlags {
		TEXTURE_FLAG_MIPMAPS = 1, /// Enable automatic mipmap generation - when available
		TEXTURE_FLAG_REPEAT = 2, /// Repeat texture (Tiling), otherwise Clamping
		TEXTURE_FLAG_FILTER = 4, /// Create texture with linear (or available) filter
		TEXTURE_FLAG_ANISOTROPIC_FILTER = 8,
		TEXTURE_FLAG_CONVERT_TO_LINEAR = 16,
		TEXTURE_FLAG_MIRRORED_REPEAT = 32, /// Repeat texture, with alternate sections mirrored
		TEXTURE_FLAG_USED_FOR_STREAMING = 2048,
		TEXTURE_FLAGS_DEFAULT = TEXTURE_FLAG_REPEAT | TEXTURE_FLAG_MIPMAPS | TEXTURE_FLAG_FILTER
	};

	struct RenderTarget;

	struct Texture {
		RID self;

		Texture *proxy;
		Set<Texture *> proxy_owners;

		String path;
		uint32_t flags;
		int width, height, depth;
		int alloc_width, alloc_height;
		Image::Format format;
		GD_RD::TextureType type;

		GLenum target;
		GLenum gl_format_cache;
		GLenum gl_internal_format_cache;
		GLenum gl_type_cache;

		int data_size;
		int total_data_size;
		bool ignore_mipmaps;

		bool compressed;

		bool srgb;

		int mipmaps;

		bool resize_to_po2;

		bool active;
		GLenum tex_id;

		uint16_t stored_cube_sides;

		RenderTarget *render_target;

		Vector<Ref<Image>> images;

		bool redraw_if_visible;

		GD_VS::TextureDetectCallback detect_3d;
		void *detect_3d_ud;

		GD_VS::TextureDetectCallback detect_srgb;
		void *detect_srgb_ud;

		GD_VS::TextureDetectCallback detect_normal;
		void *detect_normal_ud;

		// some silly opengl shenanigans where
		// texture coords start from bottom left, means we need to draw render target textures upside down
		// to be compatible with vulkan etc.
		bool is_upside_down() const {
			if (proxy)
				return proxy->is_upside_down();

			return render_target != nullptr;
		}

		Texture() {
			create();
		}

		_ALWAYS_INLINE_ Texture *get_ptr() {
			if (proxy) {
				return proxy; //->get_ptr(); only one level of indirection, else not inlining possible.
			} else {
				return this;
			}
		}

		~Texture() {
			destroy();

			if (tex_id != 0) {
				glDeleteTextures(1, &tex_id);
			}
		}

		void copy_from(const Texture &o) {
			proxy = o.proxy;
			flags = o.flags;
			width = o.width;
			height = o.height;
			alloc_width = o.alloc_width;
			alloc_height = o.alloc_height;
			format = o.format;
			type = o.type;
			target = o.target;
			data_size = o.data_size;
			total_data_size = o.total_data_size;
			ignore_mipmaps = o.ignore_mipmaps;
			compressed = o.compressed;
			mipmaps = o.mipmaps;
			resize_to_po2 = o.resize_to_po2;
			active = o.active;
			tex_id = o.tex_id;
			stored_cube_sides = o.stored_cube_sides;
			render_target = o.render_target;
			redraw_if_visible = o.redraw_if_visible;
			detect_3d = o.detect_3d;
			detect_3d_ud = o.detect_3d_ud;
			detect_srgb = o.detect_srgb;
			detect_srgb_ud = o.detect_srgb_ud;
			detect_normal = o.detect_normal;
			detect_normal_ud = o.detect_normal_ud;

			images.clear();
		}

		void create() {
			proxy = nullptr;
			flags = 0;
			width = 0;
			height = 0;
			alloc_width = 0;
			alloc_height = 0;
			format = Image::FORMAT_L8;
			type = GD_RD::TEXTURE_TYPE_2D;
			target = 0;
			data_size = 0;
			total_data_size = 0;
			ignore_mipmaps = false;
			compressed = false;
			mipmaps = 0;
			resize_to_po2 = false;
			active = false;
			tex_id = 0;
			stored_cube_sides = 0;
			render_target = nullptr;
			redraw_if_visible = false;
			detect_3d = nullptr;
			detect_3d_ud = nullptr;
			detect_srgb = nullptr;
			detect_srgb_ud = nullptr;
			detect_normal = nullptr;
			detect_normal_ud = nullptr;
		}
		void destroy() {
			images.clear();

			for (Set<Texture *>::Element *E = proxy_owners.front(); E; E = E->next()) {
				E->get()->proxy = NULL;
			}

			if (proxy) {
				proxy->proxy_owners.erase(this);
			}
		}

		// texture state
		void GLSetFilter(GLenum p_target, RS::CanvasItemTextureFilter p_filter) {
			if (p_filter == state_filter)
				return;
			state_filter = p_filter;
			GLint pmin = GL_LINEAR; // param min
			GLint pmag = GL_LINEAR; // param mag
			switch (state_filter) {
				default: {
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS: {
					pmin = GL_LINEAR_MIPMAP_LINEAR;
					pmag = GL_LINEAR;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST: {
					pmin = GL_NEAREST;
					pmag = GL_NEAREST;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS: {
					pmin = GL_NEAREST_MIPMAP_NEAREST;
					pmag = GL_NEAREST;
				} break;
			}
			glTexParameteri(p_target, GL_TEXTURE_MIN_FILTER, pmin);
			glTexParameteri(p_target, GL_TEXTURE_MAG_FILTER, pmag);
		}
		void GLSetRepeat(RS::CanvasItemTextureRepeat p_repeat) {
			if (p_repeat == state_repeat)
				return;
			state_repeat = p_repeat;
			GLint prep = GL_CLAMP_TO_EDGE; // parameter repeat
			switch (state_repeat) {
				default: {
				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED: {
					prep = GL_REPEAT;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR: {
					prep = GL_MIRRORED_REPEAT;
				} break;
			}
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, prep);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, prep);
		}

	private:
		RS::CanvasItemTextureFilter state_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR;
		RS::CanvasItemTextureRepeat state_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED;
	};

	mutable RID_PtrOwner<Texture> texture_owner;

	Ref<Image> _get_gl_image_and_format(const Ref<Image> &p_image, Image::Format p_format, uint32_t p_flags, Image::Format &r_real_format, GLenum &r_gl_format, GLenum &r_gl_internal_format, GLenum &r_gl_type, bool &r_compressed, bool p_force_decompress) const;

	void _texture_set_state_from_flags(Texture *p_tex);

	// new
	RID texture_allocate() override;
	void texture_2d_initialize(RID p_texture, const Ref<Image> &p_image) override;

	//	RID texture_2d_create(const Ref<Image> &p_image) override;
	//	RID texture_2d_layered_create(const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) override;
	//	RID texture_3d_create(Image::Format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) override { return RID(); }
	//RID texture_proxy_create(RID p_base) override;

	//void texture_2d_update_immediate(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) override;
	void texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) override;
	void texture_3d_update(RID p_texture, const Vector<Ref<Image>> &p_data) override {}
	void texture_proxy_update(RID p_proxy, RID p_base) override {}

	//RID texture_2d_placeholder_create() override;
	//RID texture_2d_layered_placeholder_create(RenderingServer::TextureLayeredType p_layered_type) override { return RID(); }
	//RID texture_3d_placeholder_create() override { return RID(); }

	Ref<Image> texture_2d_get(RID p_texture) const override;
	Ref<Image> texture_2d_layer_get(RID p_texture, int p_layer) const override { return Ref<Image>(); }
	Vector<Ref<Image>> texture_3d_get(RID p_texture) const override { return Vector<Ref<Image>>(); }

	void texture_replace(RID p_texture, RID p_by_texture) override;
	//void texture_set_size_override(RID p_texture, int p_width, int p_height) override {}

	void texture_add_to_decal_atlas(RID p_texture, bool p_panorama_to_dp = false) override {}
	void texture_remove_from_decal_atlas(RID p_texture, bool p_panorama_to_dp = false) override {}

	// old
	virtual uint32_t texture_get_width(RID p_texture) const;
	virtual uint32_t texture_get_height(RID p_texture) const;

private:
	virtual RID texture_create();

	//virtual void texture_allocate(RID p_texture, int p_width, int p_height, int p_depth_3d, Image::Format p_format, GD_RD::TextureType p_type, uint32_t p_flags = TEXTURE_FLAGS_DEFAULT);
	void _texture_allocate_internal(RID p_texture, int p_width, int p_height, int p_depth_3d, Image::Format p_format, GD_RD::TextureType p_type, uint32_t p_flags = TEXTURE_FLAGS_DEFAULT);

	virtual void texture_set_data(RID p_texture, const Ref<Image> &p_image, int p_layer = 0);
	virtual void texture_set_data_partial(RID p_texture, const Ref<Image> &p_image, int src_x, int src_y, int src_w, int src_h, int dst_x, int dst_y, int p_dst_mip, int p_layer = 0);
	//virtual Ref<Image> texture_get_data(RID p_texture, int p_layer = 0) const;
	virtual void texture_set_flags(RID p_texture, uint32_t p_flags);
	virtual uint32_t texture_get_flags(RID p_texture) const;
	virtual Image::Format texture_get_format(RID p_texture) const;
	virtual GD_RD::TextureType texture_get_type(RID p_texture) const;
	virtual uint32_t texture_get_texid(RID p_texture) const;
	virtual uint32_t texture_get_depth(RID p_texture) const;
	void texture_set_size_override(RID p_texture, int p_width, int p_height) override;

	virtual void texture_bind(RID p_texture, uint32_t p_texture_no);

	virtual void texture_set_path(RID p_texture, const String &p_path) override;
	virtual String texture_get_path(RID p_texture) const override;

	virtual void texture_set_shrink_all_x2_on_set_data(bool p_enable);

	virtual void texture_debug_usage(List<GD_VS::TextureInfo> *r_info) override;

	virtual RID texture_create_radiance_cubemap(RID p_source, int p_resolution = -1) const;

	virtual void textures_keep_original(bool p_enable);

	virtual void texture_set_proxy(RID p_texture, RID p_proxy);
	virtual Size2 texture_size_with_proxy(RID p_texture) override;

	virtual void texture_set_detect_3d_callback(RID p_texture, GD_VS::TextureDetectCallback p_callback, void *p_userdata) override;
	virtual void texture_set_detect_srgb_callback(RID p_texture, GD_VS::TextureDetectCallback p_callback, void *p_userdata);
	virtual void texture_set_detect_normal_callback(RID p_texture, GD_VS::TextureDetectCallback p_callback, void *p_userdata) override;
	void texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) override {}

	virtual void texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) override;

public:
	// CANVAS TEXTURE API
	/*
	RID canvas_texture_create() override { return RID(); }
	void canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture) override {}
	void canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_base_color, float p_shininess) override {}

	void canvas_texture_set_texture_filter(RID p_item, RS::CanvasItemTextureFilter p_filter) override {}
	void canvas_texture_set_texture_repeat(RID p_item, RS::CanvasItemTextureRepeat p_repeat) override {}
	*/
	/* SKY API */
	// not sure if used in godot 4?
	struct Sky {
		RID self;
		RID panorama;
		GLuint radiance;
		int radiance_size;
	};

	mutable RID_PtrOwner<Sky> sky_owner;

	virtual RID sky_create();
	virtual void sky_set_texture(RID p_sky, RID p_panorama, int p_radiance_size);

	// SHADER API

	struct Material;

	struct Shader {
		RID self;

		GD_VS::ShaderMode mode;
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

		uint32_t index;
		uint64_t last_pass;

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

			// these flags are specifically for batching
			// some of the logic is thus in rasterizer_storage.cpp
			// we could alternatively set bitflags for each 'uses' and test on the fly
			// defined in RasterizerStorageCommon::BatchFlags
			unsigned int batch_flags;

			bool uses_screen_texture;
			bool uses_screen_uv;
			bool uses_time;
			bool uses_modulate;
			bool uses_color;
			bool uses_vertex;

			// all these should disable item joining if used in a custom shader
			bool uses_world_matrix;
			bool uses_extra_matrix;
			bool uses_projection_matrix;
			bool uses_instance_custom;

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
			bool uses_alpha_scissor;
			bool unshaded;
			bool no_depth_test;
			bool uses_vertex;
			bool uses_discard;
			bool uses_sss;
			bool uses_screen_texture;
			bool uses_depth_texture;
			bool uses_time;
			bool uses_tangent;
			bool uses_ensure_correct_normals;
			bool writes_modelview_or_projection;
			bool uses_vertex_lighting;
			bool uses_world_coordinates;

		} spatial;

		struct Particles {
		} particles;

		bool uses_vertex_time;
		bool uses_fragment_time;

		Shader() :
				dirty_list(this) {
			shader = NULL;
			valid = false;
			custom_code_id = 0;
			version = 1;
			last_pass = 0;
		}
	};

	mutable RID_PtrOwner<Shader> shader_owner;
	mutable SelfList<Shader>::List _shader_dirty_list;

	void _shader_make_dirty(Shader *p_shader);

	RID shader_allocate() override;
	void shader_initialize(RID p_rid) override;

	//virtual RID shader_create() override;

	virtual void shader_set_code(RID p_shader, const String &p_code) override;
	virtual String shader_get_code(RID p_shader) const override;
	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const override;

	virtual void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture) override;
	virtual RID shader_get_default_texture_param(RID p_shader, const StringName &p_name) const override;

	virtual RS::ShaderNativeSourceCode shader_get_native_source_code(RID p_shader) const override { return RS::ShaderNativeSourceCode(); };

	virtual void shader_add_custom_define(RID p_shader, const String &p_define);
	virtual void shader_get_custom_defines(RID p_shader, Vector<String> *p_defines) const;
	virtual void shader_remove_custom_define(RID p_shader, const String &p_define);

	void _update_shader(Shader *p_shader) const;
	void update_dirty_shaders();

	// new
	Variant shader_get_param_default(RID p_material, const StringName &p_param) const override { return Variant(); }

	// COMMON MATERIAL API

	struct Material {
		RID self;
		Shader *shader;
		Map<StringName, Variant> params;
		SelfList<Material> list;
		SelfList<Material> dirty_list;
		Vector<Pair<StringName, RID>> textures;
		float line_width;
		int render_priority;

		RID next_pass;

		uint32_t index;
		uint64_t last_pass;

		//		Map<Geometry *, int> geometry_owners;
		//		Map<InstanceBaseDependency *, int> instance_owners;

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

	//	void _material_add_geometry(RID p_material, Geometry *p_geometry);
	//	void _material_remove_geometry(RID p_material, Geometry *p_geometry);

	void _update_material(Material *p_material);

	mutable RID_PtrOwner<Material> material_owner;

	// new
	void material_get_instance_shader_parameters(RID p_material, List<InstanceShaderParam> *r_parameters) override {}
	void material_update_dependency(RID p_material, DependencyTracker *p_instance) override {}

	// old
	RID material_allocate() override;
	void material_initialize(RID p_rid) override;

	//virtual RID material_create() override;

	virtual void material_set_shader(RID p_material, RID p_shader) override;
	virtual RID material_get_shader(RID p_material) const;

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) override;
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const override;
	virtual Variant material_get_param_default(RID p_material, const StringName &p_param) const;

	virtual void material_set_line_width(RID p_material, float p_width);
	virtual void material_set_next_pass(RID p_material, RID p_next_material) override;

	virtual bool material_is_animated(RID p_material) override;
	virtual bool material_casts_shadows(RID p_material) override;
	virtual bool material_uses_tangents(RID p_material);
	virtual bool material_uses_ensure_correct_normals(RID p_material);

	virtual void material_add_instance_owner(RID p_material, DependencyTracker *p_instance);
	virtual void material_remove_instance_owner(RID p_material, DependencyTracker *p_instance);

	virtual void material_set_render_priority(RID p_material, int priority) override;

	void update_dirty_materials();

	// RENDER TARGET

	struct RenderTarget {
		RID self;
		GLuint fbo;
		GLuint color;
		GLuint depth;

		GLuint multisample_fbo;
		GLuint multisample_color;
		GLuint multisample_depth;
		bool multisample_active;

		struct Effect {
			GLuint fbo;
			int width;
			int height;

			GLuint color;

			Effect() :
					fbo(0),
					width(0),
					height(0),
					color(0) {
			}
		};

		Effect copy_screen_effect;

		struct MipMaps {
			struct Size {
				GLuint fbo;
				GLuint color;
				int width;
				int height;
			};

			Vector<Size> sizes;
			GLuint color;
			int levels;

			MipMaps() :
					color(0),
					levels(0) {
			}
		};

		MipMaps mip_maps[2];

		struct External {
			GLuint fbo;
			GLuint color;
			GLuint depth;
			RID texture;

			External() :
					fbo(0),
					color(0),
					depth(0) {
			}
		} external;

		int x, y, width, height;

		bool flags[RENDER_TARGET_FLAG_MAX];

		// instead of allocating sized render targets immediately,
		// defer this for faster startup
		bool allocate_is_dirty = false;
		bool used_in_frame;
		GD_VS::ViewportMSAA msaa;

		bool use_fxaa;
		bool use_debanding;

		RID texture;

		bool used_dof_blur_near;
		bool mip_maps_allocated;

		Color clear_color;
		bool clear_requested;

		RenderTarget() :
				fbo(0),
				color(0),
				depth(0),
				multisample_fbo(0),
				multisample_color(0),
				multisample_depth(0),
				multisample_active(false),
				x(0),
				y(0),
				width(0),
				height(0),
				used_in_frame(false),
				msaa(GD_VS::VIEWPORT_MSAA_DISABLED),
				use_fxaa(false),
				use_debanding(false),
				used_dof_blur_near(false),
				mip_maps_allocated(false),
				clear_color(Color(1, 1, 1, 1)),
				clear_requested(false) {
			for (int i = 0; i < RENDER_TARGET_FLAG_MAX; ++i) {
				flags[i] = false;
			}
			external.fbo = 0;
		}
	};

	mutable RID_PtrOwner<RenderTarget> render_target_owner;

	void _render_target_clear(RenderTarget *rt);
	void _render_target_allocate(RenderTarget *rt);
	void _set_current_render_target(RID p_render_target);

	virtual RID render_target_create() override;
	virtual void render_target_set_position(RID p_render_target, int p_x, int p_y) override;
	virtual void render_target_set_size(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) override;
	virtual RID render_target_get_texture(RID p_render_target) override;
	virtual void render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) override;

	virtual void render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) override;
	virtual bool render_target_was_used(RID p_render_target) override;
	virtual void render_target_clear_used(RID p_render_target);
	virtual void render_target_set_msaa(RID p_render_target, GD_VS::ViewportMSAA p_msaa);
	virtual void render_target_set_use_fxaa(RID p_render_target, bool p_fxaa);
	virtual void render_target_set_use_debanding(RID p_render_target, bool p_debanding);

	// new
	void render_target_set_as_unused(RID p_render_target) override { render_target_clear_used(p_render_target); }

	void render_target_request_clear(RID p_render_target, const Color &p_clear_color) override;
	bool render_target_is_clear_requested(RID p_render_target) override;
	Color render_target_get_clear_request_color(RID p_render_target) override;
	void render_target_disable_clear_request(RID p_render_target) override;
	void render_target_do_clear_request(RID p_render_target) override;

	// access from canvas
	//	RenderTarget * render_target_get(RID p_render_target);

	/* CANVAS SHADOW */

	struct CanvasLightShadow {
		RID self;
		int size;
		int height;
		GLuint fbo;
		GLuint depth;
		GLuint distance; //for older devices
	};

	RID_PtrOwner<CanvasLightShadow> canvas_light_shadow_owner;

	virtual RID canvas_light_shadow_buffer_create(int p_width);

	/* LIGHT SHADOW MAPPING */
	/*
	struct CanvasOccluder {
		RID self;

		GLuint vertex_id; // 0 means, unconfigured
		GLuint index_id; // 0 means, unconfigured
		LocalVector<Vector2> lines;
		int len;
	};

	RID_Owner<CanvasOccluder> canvas_occluder_owner;

	virtual RID canvas_light_occluder_create();
	virtual void canvas_light_occluder_set_polylines(RID p_occluder, const LocalVector<Vector2> &p_lines);
*/

	virtual GD_VS::InstanceType get_base_type(RID p_rid) const override;

	virtual bool free(RID p_rid) override;

	struct Frame {
		RenderTarget *current_rt;

		// these 2 may have been superceded by the equivalents in the render target.
		// these may be able to be removed.
		bool clear_request;
		Color clear_request_color;

		float time[4];
		float delta;
		uint64_t count;

		Frame() {
			//			current_rt = nullptr;
			//			clear_request = false;
		}
	} frame;

	void initialize();
	void finalize();

	void _copy_screen();

	virtual bool has_os_feature(const String &p_feature) const override;

	virtual void update_dirty_resources() override;

	virtual void set_debug_generate_wireframes(bool p_generate) override;

	//	virtual void render_info_begin_capture() override;
	//	virtual void render_info_end_capture() override;
	//	virtual int get_captured_render_info(GD_VS::RenderInfo p_info) override;

	//	virtual int get_render_info(GD_VS::RenderInfo p_info) override;
	virtual String get_video_adapter_name() const override;
	virtual String get_video_adapter_vendor() const override;

	void capture_timestamps_begin() override {}
	void capture_timestamp(const String &p_name) override {}
	uint32_t get_captured_timestamps_count() const override { return 0; }
	uint64_t get_captured_timestamps_frame() const override { return 0; }
	uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const override { return 0; }
	uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const override { return 0; }
	String get_captured_timestamp_name(uint32_t p_index) const override { return String(); }

	// make access easier to these
	struct Dimensions {
		// render target
		int rt_width;
		int rt_height;

		// window
		int win_width;
		int win_height;
		Dimensions() {
			rt_width = 0;
			rt_height = 0;
			win_width = 0;
			win_height = 0;
		}
	} _dims;

	void buffer_orphan_and_upload(unsigned int p_buffer_size, unsigned int p_offset, unsigned int p_data_size, const void *p_data, GLenum p_target = GL_ARRAY_BUFFER, GLenum p_usage = GL_DYNAMIC_DRAW, bool p_optional_orphan = false) const;
	bool safe_buffer_sub_data(unsigned int p_total_buffer_size, GLenum p_target, unsigned int p_offset, unsigned int p_data_size, const void *p_data, unsigned int &r_offset_after) const;

	void bind_framebuffer(GLuint framebuffer) {
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	}

	void bind_framebuffer_system() {
		glBindFramebuffer(GL_FRAMEBUFFER, RasterizerStorageGLES2::system_fbo);
	}

	RasterizerStorageGLES2();
};

inline bool RasterizerStorageGLES2::safe_buffer_sub_data(unsigned int p_total_buffer_size, GLenum p_target, unsigned int p_offset, unsigned int p_data_size, const void *p_data, unsigned int &r_offset_after) const {
	r_offset_after = p_offset + p_data_size;
#ifdef DEBUG_ENABLED
	// we are trying to write across the edge of the buffer
	if (r_offset_after > p_total_buffer_size)
		return false;
#endif
	glBufferSubData(p_target, p_offset, p_data_size, p_data);
	return true;
}

// standardize the orphan / upload in one place so it can be changed per platform as necessary, and avoid future
// bugs causing pipeline stalls
inline void RasterizerStorageGLES2::buffer_orphan_and_upload(unsigned int p_buffer_size, unsigned int p_offset, unsigned int p_data_size, const void *p_data, GLenum p_target, GLenum p_usage, bool p_optional_orphan) const {
	// Orphan the buffer to avoid CPU/GPU sync points caused by glBufferSubData
	// Was previously #ifndef GLES_OVER_GL however this causes stalls on desktop mac also (and possibly other)
	if (!p_optional_orphan || (config.should_orphan)) {
		glBufferData(p_target, p_buffer_size, NULL, p_usage);
#ifdef RASTERIZER_EXTRA_CHECKS
		// fill with garbage off the end of the array
		if (p_buffer_size) {
			unsigned int start = p_offset + p_data_size;
			unsigned int end = start + 1024;
			if (end < p_buffer_size) {
				uint8_t *garbage = (uint8_t *)alloca(1024);
				for (int n = 0; n < 1024; n++) {
					garbage[n] = Math::random(0, 255);
				}
				glBufferSubData(p_target, start, 1024, garbage);
			}
		}
#endif
	}
	RAST_DEV_DEBUG_ASSERT((p_offset + p_data_size) <= p_buffer_size);
	glBufferSubData(p_target, p_offset, p_data_size, p_data);
}

#endif // GLES2_BACKEND_ENABLED
#endif // RASTERIZERSTORAGEGLES2_H
