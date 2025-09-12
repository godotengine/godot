/**************************************************************************/
/*  rasterized_mesh_texture.cpp                                           */
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

#include "rasterized_mesh_texture.h"
#include "scene/resources/mesh.h"

int RasterizedMeshTexture::get_width() const {
	return size.width;
}

int RasterizedMeshTexture::get_height() const {
	return size.height;
}

bool RasterizedMeshTexture::has_alpha() const {
	return true;
}

RID RasterizedMeshTexture::get_rid() const {
	if (texture.is_null()) {
		texture = RS::get_singleton()->texture_2d_placeholder_create();
	}
	return texture;
}

Ref<Image> RasterizedMeshTexture::get_image() const {
	if (texture.is_null()) {
		return Ref<Image>();
	}
	return RenderingServer::get_singleton()->texture_2d_get(texture);
}

void RasterizedMeshTexture::set_width(int p_width) {
	ERR_FAIL_COND(p_width <= 0 || p_width > 16384);
	size.width = p_width;
	texture_dirty = true;
	queue_update();
}

void RasterizedMeshTexture::set_height(int p_height) {
	ERR_FAIL_COND(p_height <= 0 || p_height > 16384);
	size.height = p_height;
	texture_dirty = true;
	queue_update();
}

void RasterizedMeshTexture::set_mesh(const Ref<Mesh> &p_mesh) {
	if (mesh.is_valid()) {
		mesh->disconnect_changed(callable_mp(this, &RasterizedMeshTexture::queue_update));
	}
	mesh = p_mesh;
	if (mesh.is_valid()) {
		mesh->connect_changed(callable_mp(this, &RasterizedMeshTexture::queue_update));
	}
	queue_update();
}

Ref<Mesh> RasterizedMeshTexture::get_mesh() const {
	return mesh;
}

void RasterizedMeshTexture::set_clear_color(const Color &p_color) {
	clear_color = p_color;
	queue_update();
}

Color RasterizedMeshTexture::get_clear_color() const {
	return clear_color;
}

void RasterizedMeshTexture::set_material(const Ref<Material> &p_material) {
	Ref<ShaderMaterial> shader_material = material;
	if (shader_material.is_valid()) {
		shader_material->disconnect_changed(callable_mp(this, &RasterizedMeshTexture::queue_update));
		shader_material->disconnect(SNAME("shader_parameter_changed"), callable_mp(this, &RasterizedMeshTexture::queue_update));
		if (shader_material->get_shader().is_valid()) {
			shader_material->get_shader()->disconnect_changed(callable_mp(this, &RasterizedMeshTexture::queue_update));
		}
	}
	material = p_material;
	shader_material = material;
	if (shader_material.is_valid()) {
		shader_material->connect_changed(callable_mp(this, &RasterizedMeshTexture::queue_update));
		shader_material->connect(SNAME("shader_parameter_changed"), callable_mp(this, &RasterizedMeshTexture::queue_update));
		if (shader_material->get_shader().is_valid()) {
			shader_material->get_shader()->connect_changed(callable_mp(this, &RasterizedMeshTexture::queue_update));
		}
	}
	queue_update();
}

Ref<Material> RasterizedMeshTexture::get_material() const {
	return material;
}

void RasterizedMeshTexture::set_surface_index(int p_surface_index) {
	surface_index = p_surface_index;
	queue_update();
}

int RasterizedMeshTexture::get_surface_index() const {
	return surface_index;
}

void RasterizedMeshTexture::set_texture_format(RS::TextureDrawableFormat p_texture_format) {
	texture_format = p_texture_format;
	texture_dirty = true;
	queue_update();
}

RS::TextureDrawableFormat RasterizedMeshTexture::get_texture_format() const {
	return texture_format;
}

void RasterizedMeshTexture::set_generate_mipmaps(bool p_generate_mipmaps) {
	generate_mipmaps = p_generate_mipmaps;
	texture_dirty = true;
	queue_update();
}

bool RasterizedMeshTexture::is_generating_mipmaps() const {
	return generate_mipmaps;
}

RasterizedMeshTexture::~RasterizedMeshTexture() {
	RS::get_singleton()->free(texture);
}

void RasterizedMeshTexture::force_draw() {
	if (!update_queued) {
		return;
	}
	if (texture_dirty) {
		RID new_texture = RS::get_singleton()->texture_drawable_2d_create(size.width, size.height, texture_format, generate_mipmaps);
		if (texture.is_null()) {
			texture = new_texture;
		} else {
			RS::get_singleton()->texture_replace(texture, new_texture);
		}
	}
	Ref<ShaderMaterial> shader_material = material;
	if (shader_material.is_valid() && mesh.is_valid()) {
		RS::get_singleton()->texture_drawable_blit_mesh_advanced(texture, shader_material->get_rid(), mesh->get_rid(), surface_index, RS::TEXTURE_DRAWABLE_BLEND_CLEAR, clear_color);
		if (generate_mipmaps) {
			RS::get_singleton()->texture_drawable_generate_mipmaps(texture);
		}
	}
	texture_dirty = false;
	update_queued = false;
	emit_changed();
}

void RasterizedMeshTexture::queue_update() {
	if (update_queued) {
		return;
	}
	callable_mp(this, &RasterizedMeshTexture::force_draw).call_deferred();
	update_queued = true;
}

void RasterizedMeshTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_width", "width"), &RasterizedMeshTexture::set_width);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &RasterizedMeshTexture::set_height);
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &RasterizedMeshTexture::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &RasterizedMeshTexture::get_mesh);
	ClassDB::bind_method(D_METHOD("set_clear_color", "color"), &RasterizedMeshTexture::set_clear_color);
	ClassDB::bind_method(D_METHOD("get_clear_color"), &RasterizedMeshTexture::get_clear_color);
	ClassDB::bind_method(D_METHOD("set_material", "material"), &RasterizedMeshTexture::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &RasterizedMeshTexture::get_material);
	ClassDB::bind_method(D_METHOD("set_surface_index", "surface_index"), &RasterizedMeshTexture::set_surface_index);
	ClassDB::bind_method(D_METHOD("get_surface_index"), &RasterizedMeshTexture::get_surface_index);
	ClassDB::bind_method(D_METHOD("set_texture_format", "texture_format"), &RasterizedMeshTexture::set_texture_format);
	ClassDB::bind_method(D_METHOD("get_texture_format"), &RasterizedMeshTexture::get_texture_format);
	ClassDB::bind_method(D_METHOD("set_generate_mipmaps", "generate_mipmaps"), &RasterizedMeshTexture::set_generate_mipmaps);
	ClassDB::bind_method(D_METHOD("is_generating_mipmaps"), &RasterizedMeshTexture::is_generating_mipmaps);
	ClassDB::bind_method(D_METHOD("force_draw"), &RasterizedMeshTexture::force_draw);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,2048,or_greater,suffix:px"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "height", PROPERTY_HINT_RANGE, "1,2048,or_greater,suffix:px"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "clear_color"), "set_clear_color", "get_clear_color");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "surface_index", PROPERTY_HINT_RANGE, "0,10,1,or_greater"), "set_surface_index", "get_surface_index");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial"), "set_material", "get_material");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_format", PROPERTY_HINT_ENUM,
						 vformat("R8:%d,RH:%d,RF:%d,RG8:%d,RGH:%d,RGF:%d,RGBA8_SRGB:%d,RGBA8:%d,RGBAH:%d,RGBAF:%d",
								 // R
								 RS::TEXTURE_DRAWABLE_FORMAT_R8, RS::TEXTURE_DRAWABLE_FORMAT_RH, RS::TEXTURE_DRAWABLE_FORMAT_RF,
								 // RG
								 RS::TEXTURE_DRAWABLE_FORMAT_RG8, RS::TEXTURE_DRAWABLE_FORMAT_RGH, RS::TEXTURE_DRAWABLE_FORMAT_RGF,
								 // RGBA
								 RS::TEXTURE_DRAWABLE_FORMAT_RGBA8_SRGB, RS::TEXTURE_DRAWABLE_FORMAT_RGBA8, RS::TEXTURE_DRAWABLE_FORMAT_RGBAH, RS::TEXTURE_DRAWABLE_FORMAT_RGBAF)),
			"set_texture_format", "get_texture_format");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "generate_mipmaps"), "set_generate_mipmaps", "is_generating_mipmaps");
}
