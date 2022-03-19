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

#ifndef TEXTURE_STORAGE_H
#define TEXTURE_STORAGE_H

#include "servers/rendering_server.h"

class RendererTextureStorage {
public:
	virtual bool can_create_resources_async() const = 0;

	virtual ~RendererTextureStorage(){};

	virtual RID texture_allocate() = 0;
	virtual void texture_free(RID p_rid) = 0;

	virtual void texture_2d_initialize(RID p_texture, const Ref<Image> &p_image) = 0;
	virtual void texture_2d_layered_initialize(RID p_texture, const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) = 0;
	virtual void texture_3d_initialize(RID p_texture, Image::Format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) = 0;
	virtual void texture_proxy_initialize(RID p_texture, RID p_base) = 0; //all slices, then all the mipmaps, must be coherent

	virtual void texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) = 0;
	virtual void texture_3d_update(RID p_texture, const Vector<Ref<Image>> &p_data) = 0;
	virtual void texture_proxy_update(RID p_proxy, RID p_base) = 0;

	//these two APIs can be used together or in combination with the others.
	virtual void texture_2d_placeholder_initialize(RID p_texture) = 0;
	virtual void texture_2d_layered_placeholder_initialize(RID p_texture, RenderingServer::TextureLayeredType p_layered_type) = 0;
	virtual void texture_3d_placeholder_initialize(RID p_texture) = 0;

	virtual Ref<Image> texture_2d_get(RID p_texture) const = 0;
	virtual Ref<Image> texture_2d_layer_get(RID p_texture, int p_layer) const = 0;
	virtual Vector<Ref<Image>> texture_3d_get(RID p_texture) const = 0;

	virtual void texture_replace(RID p_texture, RID p_by_texture) = 0;
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height) = 0;

	virtual void texture_set_path(RID p_texture, const String &p_path) = 0;
	virtual String texture_get_path(RID p_texture) const = 0;

	virtual void texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) = 0;
	virtual void texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) = 0;
	virtual void texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) = 0;

	virtual void texture_debug_usage(List<RS::TextureInfo> *r_info) = 0;

	virtual void texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) = 0;

	virtual Size2 texture_size_with_proxy(RID p_proxy) = 0;
};

#endif // !TEXTURE_STORAGE_H
