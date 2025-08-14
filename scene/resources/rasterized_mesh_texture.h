/**************************************************************************/
/*  rasterized_mesh_texture.h                                             */
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

#pragma once

#include "scene/resources/texture.h"

class Mesh;
class Material;

class RasterizedMeshTexture : public Texture2D {
	GDCLASS(RasterizedMeshTexture, Texture2D);

	Size2i size = Size2i(256, 256);
	int surface_index = 0;
	Color clear_color = Color(0, 0, 0, 0);
	Ref<Mesh> mesh;
	Ref<Material> material;
	RS::TextureDrawableFormat texture_format = RS::TEXTURE_DRAWABLE_FORMAT_RGBA8;
	bool generate_mipmaps = false;

	mutable RID texture;

	bool texture_dirty = true;

	bool update_queued = false;

	void queue_update();

protected:
	static void _bind_methods();

public:
	int get_width() const override;
	int get_height() const override;
	bool has_alpha() const override;
	RID get_rid() const override;

	Ref<Image> get_image() const override;

	void set_width(int p_width);
	void set_height(int p_height);

	void set_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_mesh() const;

	void set_clear_color(const Color &p_color);
	Color get_clear_color() const;

	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	void set_surface_index(int p_surface_index);
	int get_surface_index() const;

	void set_texture_format(RS::TextureDrawableFormat p_texture_format);
	RS::TextureDrawableFormat get_texture_format() const;

	void set_generate_mipmaps(bool p_generate_mipmaps);
	bool is_generating_mipmaps() const;

	void force_draw();

	~RasterizedMeshTexture();
};
