/*************************************************************************/
/*  texture_storage.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEXTURE_STORAGE_GLES3_H
#define TEXTURE_STORAGE_GLES3_H

#ifdef GLES3_ENABLED

#include "config.h"
#include "core/os/os.h"
#include "core/templates/rid_owner.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/storage/texture_storage.h"

// This must come first to avoid windows.h mess
#include "platform_config.h"
#ifndef OPENGL_INCLUDE_H
#include <GLES3/gl3.h>
#else
#include OPENGL_INCLUDE_H
#endif

namespace GLES3 {

#define _EXT_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#define _EXT_COMPRESSED_RGBA_S3TC_DXT3_EXT 0x83F2
#define _EXT_COMPRESSED_RGBA_S3TC_DXT5_EXT 0x83F3

#define _EXT_COMPRESSED_RED_RGTC1_EXT 0x8DBB
#define _EXT_COMPRESSED_RED_RGTC1 0x8DBB
#define _EXT_COMPRESSED_SIGNED_RED_RGTC1 0x8DBC
#define _EXT_COMPRESSED_RG_RGTC2 0x8DBD
#define _EXT_COMPRESSED_SIGNED_RG_RGTC2 0x8DBE
#define _EXT_COMPRESSED_SIGNED_RED_RGTC1_EXT 0x8DBC
#define _EXT_COMPRESSED_RED_GREEN_RGTC2_EXT 0x8DBD
#define _EXT_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT 0x8DBE
#define _EXT_ETC1_RGB8_OES 0x8D64

#define _EXT_COMPRESSED_RGBA_BPTC_UNORM 0x8E8C
#define _EXT_COMPRESSED_SRGB_ALPHA_BPTC_UNORM 0x8E8D
#define _EXT_COMPRESSED_RGB_BPTC_SIGNED_FLOAT 0x8E8E
#define _EXT_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT 0x8E8F

#define _GL_TEXTURE_EXTERNAL_OES 0x8D65

#ifdef GLES_OVER_GL
#define _GL_HALF_FLOAT_OES 0x140B
#else
#define _GL_HALF_FLOAT_OES 0x8D61
#endif

#define _EXT_TEXTURE_CUBE_MAP_SEAMLESS 0x884F

#define _RED_OES 0x1903

#define _DEPTH_COMPONENT24_OES 0x81A6

#ifndef GLES_OVER_GL
#define glClearDepth glClearDepthf
#endif //!GLES_OVER_GL

enum OpenGLTextureFlags {
	TEXTURE_FLAG_MIPMAPS = 1, /// Enable automatic mipmap generation - when available
	TEXTURE_FLAG_REPEAT = 2, /// Repeat texture (Tiling), otherwise Clamping
	TEXTURE_FLAG_FILTER = 4, /// Create texture with linear (or available) filter
	TEXTURE_FLAG_ANISOTROPIC_FILTER = 8,
	TEXTURE_FLAG_CONVERT_TO_LINEAR = 16,
	TEXTURE_FLAG_MIRRORED_REPEAT = 32, /// Repeat texture, with alternate sections mirrored
	TEXTURE_FLAG_USED_FOR_STREAMING = 2048,
	TEXTURE_FLAGS_DEFAULT = TEXTURE_FLAG_REPEAT | TEXTURE_FLAG_MIPMAPS | TEXTURE_FLAG_FILTER
};

struct CanvasTexture {
	RID diffuse;
	RID normal_map;
	RID specular;
	Color specular_color = Color(1, 1, 1, 1);
	float shininess = 1.0;

	RS::CanvasItemTextureFilter texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT;
	RS::CanvasItemTextureRepeat texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT;

	Size2i size_cache = Size2i(1, 1);
	bool use_normal_cache = false;
	bool use_specular_cache = false;
	bool cleared_cache = true;
};

struct RenderTarget;

struct Texture {
	RID self;

	Texture *proxy = nullptr;
	Set<Texture *> proxy_owners;

	String path;
	uint32_t flags;
	int width, height, depth;
	int alloc_width, alloc_height;
	Image::Format format;
	RenderingDevice::TextureType type;

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

	RenderTarget *render_target = nullptr;

	Vector<Ref<Image>> images;

	bool redraw_if_visible;

	RS::TextureDetectCallback detect_3d;
	void *detect_3d_ud = nullptr;

	RS::TextureDetectCallback detect_srgb;
	void *detect_srgb_ud = nullptr;

	RS::TextureDetectCallback detect_normal;
	void *detect_normal_ud = nullptr;

	CanvasTexture *canvas_texture = nullptr;

	// some silly opengl shenanigans where
	// texture coords start from bottom left, means we need to draw render target textures upside down
	// to be compatible with vulkan etc.
	bool is_upside_down() const {
		if (proxy) {
			return proxy->is_upside_down();
		}

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
		type = RenderingDevice::TEXTURE_TYPE_2D;
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
			E->get()->proxy = nullptr;
		}

		if (proxy) {
			proxy->proxy_owners.erase(this);
		}
	}

	// texture state
	void GLSetFilter(GLenum p_target, RS::CanvasItemTextureFilter p_filter) {
		if (p_filter == state_filter) {
			return;
		}
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
	void GLSetRepeat(GLenum p_target, RS::CanvasItemTextureRepeat p_repeat) {
		if (p_repeat == state_repeat) {
			return;
		}
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
		glTexParameteri(p_target, GL_TEXTURE_WRAP_S, prep);
		glTexParameteri(p_target, GL_TEXTURE_WRAP_T, prep);
	}

private:
	RS::CanvasItemTextureFilter state_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR;
	RS::CanvasItemTextureRepeat state_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED;
};

struct RenderTarget {
	RID self;
	GLuint fbo = 0;
	GLuint color = 0;
	GLuint depth = 0;

	GLuint multisample_fbo = 0;
	GLuint multisample_color = 0;
	GLuint multisample_depth = 0;
	bool multisample_active = false;

	struct Effect {
		GLuint fbo = 0;
		int width = 0;
		int height = 0;

		GLuint color = 0;
	};

	Effect copy_screen_effect;

	struct MipMaps {
		struct Size {
			GLuint fbo = 0;
			GLuint color = 0;
			int width = 0;
			int height = 0;
		};

		Vector<Size> sizes;
		GLuint color = 0;
		int levels = 0;
	};

	MipMaps mip_maps[2];

	struct External {
		GLuint fbo = 0;
		GLuint color = 0;
		GLuint depth = 0;
		RID texture;
	} external;

	int x = 0;
	int y = 0;
	int width = 0;
	int height = 0;

	bool flags[RendererTextureStorage::RENDER_TARGET_FLAG_MAX] = {};

	// instead of allocating sized render targets immediately,
	// defer this for faster startup
	bool allocate_is_dirty = false;
	bool used_in_frame = false;
	RS::ViewportMSAA msaa = RS::VIEWPORT_MSAA_DISABLED;

	bool use_fxaa = false;
	bool use_debanding = false;

	RID texture;

	bool used_dof_blur_near = false;
	bool mip_maps_allocated = false;

	Color clear_color = Color(1, 1, 1, 1);
	bool clear_requested = false;

	RenderTarget() {
		for (int i = 0; i < RendererTextureStorage::RENDER_TARGET_FLAG_MAX; ++i) {
			flags[i] = false;
		}
		external.fbo = 0;
	}
};

class TextureStorage : public RendererTextureStorage {
private:
	static TextureStorage *singleton;

	Thread::ID _main_thread_id = 0;
	bool _is_main_thread();

	/* Canvas Texture API */

	RID_Owner<CanvasTexture, true> canvas_texture_owner;

	/* Texture API */

	mutable RID_PtrOwner<Texture> texture_owner;

	Ref<Image> _get_gl_image_and_format(const Ref<Image> &p_image, Image::Format p_format, uint32_t p_flags, Image::Format &r_real_format, GLenum &r_gl_format, GLenum &r_gl_internal_format, GLenum &r_gl_type, bool &r_compressed, bool p_force_decompress) const;
	void _texture_set_state_from_flags(Texture *p_tex);

	void texture_set_proxy(RID p_texture, RID p_proxy);

	/* Render Target API */

	mutable RID_PtrOwner<RenderTarget> render_target_owner;

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

public:
	static TextureStorage *get_singleton();

	TextureStorage();
	virtual ~TextureStorage();

	/* Canvas Texture API */

	CanvasTexture *get_canvas_texture(RID p_rid) { return canvas_texture_owner.get_or_null(p_rid); };
	bool owns_canvas_texture(RID p_rid) { return canvas_texture_owner.owns(p_rid); };

	virtual RID canvas_texture_allocate() override;
	virtual void canvas_texture_initialize(RID p_rid) override;
	virtual void canvas_texture_free(RID p_rid) override;

	virtual void canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture) override;
	virtual void canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_base_color, float p_shininess) override;

	virtual void canvas_texture_set_texture_filter(RID p_item, RS::CanvasItemTextureFilter p_filter) override;
	virtual void canvas_texture_set_texture_repeat(RID p_item, RS::CanvasItemTextureRepeat p_repeat) override;

	/* Texture API */

	Texture *get_texture(RID p_rid) { return texture_owner.get_or_null(p_rid); };
	bool owns_texture(RID p_rid) { return texture_owner.owns(p_rid); };
	RID make_rid(Texture *p_texture) { return texture_owner.make_rid(p_texture); };

	void set_main_thread_id(Thread::ID p_id);

	virtual bool can_create_resources_async() const override;

	RID texture_create();
	void _texture_allocate_internal(RID p_texture, int p_width, int p_height, int p_depth_3d, Image::Format p_format, RenderingDevice::TextureType p_type, uint32_t p_flags = TEXTURE_FLAGS_DEFAULT);

	virtual RID texture_allocate() override;
	virtual void texture_free(RID p_rid) override;

	virtual void texture_2d_initialize(RID p_texture, const Ref<Image> &p_image) override;
	virtual void texture_2d_layered_initialize(RID p_texture, const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) override;
	virtual void texture_3d_initialize(RID p_texture, Image::Format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) override;
	virtual void texture_proxy_initialize(RID p_texture, RID p_base) override; //all slices, then all the mipmaps, must be coherent

	virtual void texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) override;
	virtual void texture_3d_update(RID p_texture, const Vector<Ref<Image>> &p_data) override{};
	virtual void texture_proxy_update(RID p_proxy, RID p_base) override{};

	//these two APIs can be used together or in combination with the others.
	virtual void texture_2d_placeholder_initialize(RID p_texture) override;
	virtual void texture_2d_layered_placeholder_initialize(RID p_texture, RenderingServer::TextureLayeredType p_layered_type) override;
	virtual void texture_3d_placeholder_initialize(RID p_texture) override;

	virtual Ref<Image> texture_2d_get(RID p_texture) const override;
	virtual Ref<Image> texture_2d_layer_get(RID p_texture, int p_layer) const override { return Ref<Image>(); };
	virtual Vector<Ref<Image>> texture_3d_get(RID p_texture) const override { return Vector<Ref<Image>>(); };

	virtual void texture_replace(RID p_texture, RID p_by_texture) override;
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height) override;

	virtual void texture_set_path(RID p_texture, const String &p_path) override;
	virtual String texture_get_path(RID p_texture) const override;

	virtual void texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) override;
	void texture_set_detect_srgb_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata);
	virtual void texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) override;
	virtual void texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) override{};

	virtual void texture_debug_usage(List<RS::TextureInfo> *r_info) override;

	virtual void texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) override;

	virtual Size2 texture_size_with_proxy(RID p_proxy) override;

	void texture_set_data(RID p_texture, const Ref<Image> &p_image, int p_layer = 0);
	void texture_set_data_partial(RID p_texture, const Ref<Image> &p_image, int src_x, int src_y, int src_w, int src_h, int dst_x, int dst_y, int p_dst_mip, int p_layer = 0);
	//Ref<Image> texture_get_data(RID p_texture, int p_layer = 0) const;
	void texture_set_flags(RID p_texture, uint32_t p_flags);
	uint32_t texture_get_flags(RID p_texture) const;
	Image::Format texture_get_format(RID p_texture) const;
	RenderingDevice::TextureType texture_get_type(RID p_texture) const;
	uint32_t texture_get_texid(RID p_texture) const;
	uint32_t texture_get_width(RID p_texture) const;
	uint32_t texture_get_height(RID p_texture) const;
	uint32_t texture_get_depth(RID p_texture) const;
	void texture_bind(RID p_texture, uint32_t p_texture_no);
	void texture_set_shrink_all_x2_on_set_data(bool p_enable);
	RID texture_create_radiance_cubemap(RID p_source, int p_resolution = -1) const;
	void textures_keep_original(bool p_enable);

	/* DECAL API */

	virtual RID decal_allocate() override;
	virtual void decal_initialize(RID p_rid) override;
	virtual void decal_free(RID p_rid) override{};

	virtual void decal_set_extents(RID p_decal, const Vector3 &p_extents) override;
	virtual void decal_set_texture(RID p_decal, RS::DecalTexture p_type, RID p_texture) override;
	virtual void decal_set_emission_energy(RID p_decal, float p_energy) override;
	virtual void decal_set_albedo_mix(RID p_decal, float p_mix) override;
	virtual void decal_set_modulate(RID p_decal, const Color &p_modulate) override;
	virtual void decal_set_cull_mask(RID p_decal, uint32_t p_layers) override;
	virtual void decal_set_distance_fade(RID p_decal, bool p_enabled, float p_begin, float p_length) override;
	virtual void decal_set_fade(RID p_decal, float p_above, float p_below) override;
	virtual void decal_set_normal_fade(RID p_decal, float p_fade) override;

	virtual AABB decal_get_aabb(RID p_decal) const override;

	virtual void texture_add_to_decal_atlas(RID p_texture, bool p_panorama_to_dp = false) override {}
	virtual void texture_remove_from_decal_atlas(RID p_texture, bool p_panorama_to_dp = false) override {}

	/* RENDER TARGET API */

	static GLuint system_fbo;

	struct Frame {
		GLES3::RenderTarget *current_rt;

		// these 2 may have been superseded by the equivalents in the render target.
		// these may be able to be removed.
		bool clear_request;
		Color clear_request_color;

		float time;
		float delta;
		uint64_t count;

		Frame() {
			//			current_rt = nullptr;
			//			clear_request = false;
		}
	} frame;

	RenderTarget *get_render_target(RID p_rid) { return render_target_owner.get_or_null(p_rid); };
	bool owns_render_target(RID p_rid) { return render_target_owner.owns(p_rid); };

	void _render_target_clear(RenderTarget *rt);
	void _render_target_allocate(RenderTarget *rt);
	void _set_current_render_target(RID p_render_target);

	virtual RID render_target_create() override;
	virtual void render_target_free(RID p_rid) override;
	virtual void render_target_set_position(RID p_render_target, int p_x, int p_y) override;
	virtual void render_target_set_size(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) override;
	Size2i render_target_get_size(RID p_render_target);
	virtual RID render_target_get_texture(RID p_render_target) override;
	virtual void render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) override;

	virtual void render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) override;
	virtual bool render_target_was_used(RID p_render_target) override;
	void render_target_clear_used(RID p_render_target);
	void render_target_set_msaa(RID p_render_target, RS::ViewportMSAA p_msaa);
	void render_target_set_use_fxaa(RID p_render_target, bool p_fxaa);
	void render_target_set_use_debanding(RID p_render_target, bool p_debanding);

	// new
	void render_target_set_as_unused(RID p_render_target) override {
		render_target_clear_used(p_render_target);
	}

	void render_target_request_clear(RID p_render_target, const Color &p_clear_color) override;
	bool render_target_is_clear_requested(RID p_render_target) override;
	Color render_target_get_clear_request_color(RID p_render_target) override;
	void render_target_disable_clear_request(RID p_render_target) override;
	void render_target_do_clear_request(RID p_render_target) override;

	void render_target_set_sdf_size_and_scale(RID p_render_target, RS::ViewportSDFOversize p_size, RS::ViewportSDFScale p_scale) override;
	Rect2i render_target_get_sdf_rect(RID p_render_target) const override;
	void render_target_mark_sdf_enabled(RID p_render_target, bool p_enabled) override;
};

} // namespace GLES3

#endif // !GLES3_ENABLED

#endif // !TEXTURE_STORAGE_GLES3_H
