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

#ifndef TEXTURE_STORAGE_DUMMY_H
#define TEXTURE_STORAGE_DUMMY_H

#include "servers/rendering/rendering_server_globals.h"
#include "servers/rendering/storage/texture_storage.h"

namespace RendererDummy {

class TextureStorage : public RendererTextureStorage {
private:
	struct DummyTexture {
		Ref<Image> image;
	};
	mutable RID_PtrOwner<DummyTexture> texture_owner;

public:
	static TextureStorage *get_singleton() {
		// Here we cheat until we can retire RasterizerStorageDummy::free()

		return (TextureStorage *)RSG::texture_storage;
	};

	virtual bool can_create_resources_async() const override { return false; }

	DummyTexture *get_texture(RID p_rid) { return texture_owner.get_or_null(p_rid); };
	bool owns_texture(RID p_rid) { return texture_owner.owns(p_rid); };

	virtual RID texture_allocate() override {
		DummyTexture *texture = memnew(DummyTexture);
		ERR_FAIL_COND_V(!texture, RID());
		return texture_owner.make_rid(texture);
	};

	virtual void texture_free(RID p_rid) override {
		// delete the texture
		DummyTexture *texture = texture_owner.get_or_null(p_rid);
		texture_owner.free(p_rid);
		memdelete(texture);
	};

	virtual void texture_2d_initialize(RID p_texture, const Ref<Image> &p_image) override {
		DummyTexture *t = texture_owner.get_or_null(p_texture);
		ERR_FAIL_COND(!t);
		t->image = p_image->duplicate();
	};
	virtual void texture_2d_layered_initialize(RID p_texture, const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) override{};
	virtual void texture_3d_initialize(RID p_texture, Image::Format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) override{};
	virtual void texture_proxy_initialize(RID p_texture, RID p_base) override{}; //all slices, then all the mipmaps, must be coherent

	virtual void texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) override{};
	virtual void texture_3d_update(RID p_texture, const Vector<Ref<Image>> &p_data) override{};
	virtual void texture_proxy_update(RID p_proxy, RID p_base) override{};

	//these two APIs can be used together or in combination with the others.
	virtual void texture_2d_placeholder_initialize(RID p_texture) override{};
	virtual void texture_2d_layered_placeholder_initialize(RID p_texture, RenderingServer::TextureLayeredType p_layered_type) override{};
	virtual void texture_3d_placeholder_initialize(RID p_texture) override{};

	virtual Ref<Image> texture_2d_get(RID p_texture) const override {
		DummyTexture *t = texture_owner.get_or_null(p_texture);
		ERR_FAIL_COND_V(!t, Ref<Image>());
		return t->image;
	};
	virtual Ref<Image> texture_2d_layer_get(RID p_texture, int p_layer) const override { return Ref<Image>(); };
	virtual Vector<Ref<Image>> texture_3d_get(RID p_texture) const override { return Vector<Ref<Image>>(); };

	virtual void texture_replace(RID p_texture, RID p_by_texture) override { texture_free(p_by_texture); };
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height) override{};

	virtual void texture_set_path(RID p_texture, const String &p_path) override{};
	virtual String texture_get_path(RID p_texture) const override { return String(); };

	virtual void texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) override{};
	virtual void texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) override{};
	virtual void texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) override{};

	virtual void texture_debug_usage(List<RS::TextureInfo> *r_info) override{};

	virtual void texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) override{};

	virtual Size2 texture_size_with_proxy(RID p_proxy) override { return Size2(); };
};

} // namespace RendererDummy

#endif // !TEXTURE_STORAGE_DUMMY_H
