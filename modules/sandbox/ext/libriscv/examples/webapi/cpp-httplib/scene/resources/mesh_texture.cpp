/**************************************************************************/
/*  mesh_texture.cpp                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "mesh_texture.h"

#include "scene/resources/mesh.h"

int MeshTexture::get_width() const {
	return size.width;
}

int MeshTexture::get_height() const {
	return size.height;
}

RID MeshTexture::get_rid() const {
	return RID();
}

bool MeshTexture::has_alpha() const {
	return false;
}

void MeshTexture::set_mesh(const Ref<Mesh> &p_mesh) {
	mesh = p_mesh;
}

Ref<Mesh> MeshTexture::get_mesh() const {
	return mesh;
}

void MeshTexture::set_image_size(const Size2 &p_size) {
	size = p_size;
}

Size2 MeshTexture::get_image_size() const {
	return size;
}

void MeshTexture::set_base_texture(const Ref<Texture2D> &p_texture) {
	base_texture = p_texture;
}

Ref<Texture2D> MeshTexture::get_base_texture() const {
	return base_texture;
}

void MeshTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	if (mesh.is_null() || base_texture.is_null()) {
		return;
	}
	Transform2D xform;
	xform.set_origin(p_pos);
	if (p_transpose) {
		SWAP(xform.columns[0][1], xform.columns[1][0]);
		SWAP(xform.columns[0][0], xform.columns[1][1]);
	}
	RenderingServer::get_singleton()->canvas_item_add_mesh(p_canvas_item, mesh->get_rid(), xform, p_modulate, base_texture->get_rid());
}

void MeshTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	if (mesh.is_null() || base_texture.is_null()) {
		return;
	}
	Transform2D xform;
	Vector2 origin = p_rect.position;
	if (p_rect.size.x < 0) {
		origin.x += size.x;
	}
	if (p_rect.size.y < 0) {
		origin.y += size.y;
	}
	xform.set_origin(origin);
	xform.set_scale(p_rect.size / size);

	if (p_transpose) {
		SWAP(xform.columns[0][1], xform.columns[1][0]);
		SWAP(xform.columns[0][0], xform.columns[1][1]);
	}
	RenderingServer::get_singleton()->canvas_item_add_mesh(p_canvas_item, mesh->get_rid(), xform, p_modulate, base_texture->get_rid());
}

void MeshTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	if (mesh.is_null() || base_texture.is_null()) {
		return;
	}
	Transform2D xform;
	Vector2 origin = p_rect.position;
	if (p_rect.size.x < 0) {
		origin.x += size.x;
	}
	if (p_rect.size.y < 0) {
		origin.y += size.y;
	}
	xform.set_origin(origin);
	xform.set_scale(p_rect.size / size);

	if (p_transpose) {
		SWAP(xform.columns[0][1], xform.columns[1][0]);
		SWAP(xform.columns[0][0], xform.columns[1][1]);
	}
	RenderingServer::get_singleton()->canvas_item_add_mesh(p_canvas_item, mesh->get_rid(), xform, p_modulate, base_texture->get_rid());
}

bool MeshTexture::get_rect_region(const Rect2 &p_rect, const Rect2 &p_src_rect, Rect2 &r_rect, Rect2 &r_src_rect) const {
	r_rect = p_rect;
	r_src_rect = p_src_rect;
	return true;
}

bool MeshTexture::is_pixel_opaque(int p_x, int p_y) const {
	return true;
}

void MeshTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &MeshTexture::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &MeshTexture::get_mesh);
	ClassDB::bind_method(D_METHOD("set_image_size", "size"), &MeshTexture::set_image_size);
	ClassDB::bind_method(D_METHOD("get_image_size"), &MeshTexture::get_image_size);
	ClassDB::bind_method(D_METHOD("set_base_texture", "texture"), &MeshTexture::set_base_texture);
	ClassDB::bind_method(D_METHOD("get_base_texture"), &MeshTexture::get_base_texture);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "base_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_base_texture", "get_base_texture");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "image_size", PROPERTY_HINT_RANGE, "0,16384,1,suffix:px"), "set_image_size", "get_image_size");
}

MeshTexture::MeshTexture() {
}
