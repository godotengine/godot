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

#ifndef TEXTURE_STORAGE_RD_H
#define TEXTURE_STORAGE_RD_H

#include "canvas_texture_storage.h"
#include "core/templates/rid_owner.h"
#include "servers/rendering/storage/texture_storage.h"

namespace RendererRD {

enum DefaultRDTexture {
	DEFAULT_RD_TEXTURE_WHITE,
	DEFAULT_RD_TEXTURE_BLACK,
	DEFAULT_RD_TEXTURE_NORMAL,
	DEFAULT_RD_TEXTURE_ANISO,
	DEFAULT_RD_TEXTURE_MULTIMESH_BUFFER,
	DEFAULT_RD_TEXTURE_CUBEMAP_BLACK,
	DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK,
	DEFAULT_RD_TEXTURE_CUBEMAP_WHITE,
	DEFAULT_RD_TEXTURE_3D_WHITE,
	DEFAULT_RD_TEXTURE_3D_BLACK,
	DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE,
	DEFAULT_RD_TEXTURE_2D_UINT,
	DEFAULT_RD_TEXTURE_MAX
};

class Texture {
public:
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

	void cleanup();
};

class TextureStorage : public RendererTextureStorage {
private:
	static TextureStorage *singleton;

	//textures can be created from threads, so this RID_Owner is thread safe
	mutable RID_Owner<Texture, true> texture_owner;

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

	Ref<Image> _validate_texture_format(const Ref<Image> &p_image, TextureToRDFormat &r_format);
	void _texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0, bool p_immediate = false);

public:
	static TextureStorage *get_singleton();

	RID default_rd_textures[DEFAULT_RD_TEXTURE_MAX];

	_FORCE_INLINE_ RID texture_rd_get_default(DefaultRDTexture p_texture) {
		return default_rd_textures[p_texture];
	}

	TextureStorage();
	virtual ~TextureStorage();

	Texture *get_texture(RID p_rid) { return texture_owner.get_or_null(p_rid); };
	bool owns_texture(RID p_rid) { return texture_owner.owns(p_rid); };

	virtual bool can_create_resources_async() const override;

	virtual RID texture_allocate() override;
	virtual void texture_free(RID p_rid) override;

	virtual void texture_2d_initialize(RID p_texture, const Ref<Image> &p_image) override;
	virtual void texture_2d_layered_initialize(RID p_texture, const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) override;
	virtual void texture_3d_initialize(RID p_texture, Image::Format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) override;
	virtual void texture_proxy_initialize(RID p_texture, RID p_base) override; //all slices, then all the mipmaps, must be coherent

	virtual void texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) override;
	virtual void texture_3d_update(RID p_texture, const Vector<Ref<Image>> &p_data) override;
	virtual void texture_proxy_update(RID p_proxy, RID p_base) override;

	//these two APIs can be used together or in combination with the others.
	virtual void texture_2d_placeholder_initialize(RID p_texture) override;
	virtual void texture_2d_layered_placeholder_initialize(RID p_texture, RenderingServer::TextureLayeredType p_layered_type) override;
	virtual void texture_3d_placeholder_initialize(RID p_texture) override;

	virtual Ref<Image> texture_2d_get(RID p_texture) const override;
	virtual Ref<Image> texture_2d_layer_get(RID p_texture, int p_layer) const override;
	virtual Vector<Ref<Image>> texture_3d_get(RID p_texture) const override;

	virtual void texture_replace(RID p_texture, RID p_by_texture) override;
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height) override;

	virtual void texture_set_path(RID p_texture, const String &p_path) override;
	virtual String texture_get_path(RID p_texture) const override;

	virtual void texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) override;
	virtual void texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) override;
	virtual void texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) override;

	virtual void texture_debug_usage(List<RS::TextureInfo> *r_info) override;

	virtual void texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) override;

	virtual Size2 texture_size_with_proxy(RID p_proxy) override;

	//internal usage

	_FORCE_INLINE_ RID texture_get_rd_texture(RID p_texture, bool p_srgb = false) {
		if (p_texture.is_null()) {
			return RID();
		}
		RendererRD::Texture *tex = texture_owner.get_or_null(p_texture);

		if (!tex) {
			return RID();
		}
		return (p_srgb && tex->rd_texture_srgb.is_valid()) ? tex->rd_texture_srgb : tex->rd_texture;
	}

	_FORCE_INLINE_ Size2i texture_2d_get_size(RID p_texture) {
		if (p_texture.is_null()) {
			return Size2i();
		}
		RendererRD::Texture *tex = texture_owner.get_or_null(p_texture);

		if (!tex) {
			return Size2i();
		}
		return Size2i(tex->width_2d, tex->height_2d);
	}
};

} // namespace RendererRD

#endif // !_TEXTURE_STORAGE_RD_H
