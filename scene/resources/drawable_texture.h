/**************************************************************************/
/*  drawable_texture.h                                                    */
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

class DrawableTexture2D : public Texture2D {
	GDCLASS(DrawableTexture2D, Texture2D);

	Size2i size = Size2i(256, 256);

	RID texture;

protected:
	static void _bind_methods();

public:
	int get_width() const override;
	int get_height() const override;
	bool has_alpha() const override;
	RID get_rid() const override;

	Ref<Image> get_image() const override;

	void setup(Size2i p_size, RD::DataFormat p_texture_format, bool p_use_mipmaps = false);
	void draw_mesh(const Ref<Material> &p_material, const Ref<Mesh> &p_mesh, uint32_t p_surface_index, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color);
	void blit_rect(Rect2i p_rect, const Ref<Texture2D> &p_source_texture, const Color &p_modulate, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color);

	DrawableTexture2D();
	~DrawableTexture2D();
};
