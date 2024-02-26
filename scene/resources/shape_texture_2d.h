/**************************************************************************/
/*  shape_texture_2d.h                                                    */
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

#ifndef SHAPE_TEXTURE_2D_H
#define SHAPE_TEXTURE_2D_H

#include "scene/resources/texture.h"

class ShapeTexture2D : public Texture2D {
	GDCLASS(ShapeTexture2D, Texture2D);

private:
	int width = 64;
	int height = 64;
	int points = 3;
	bool star_enabled = false;
	real_t star_inset = 0.5;
	real_t rotation_degrees = 0.0;
	Color fill_color = Color(1, 1, 1, 1);
	Color border_color;
	real_t border_width = 0.0;
	bool antialiased = false;

	bool update_pending = false;
	mutable RID texture;
	Color background_color = Color(0, 0, 0, 0);

	void _queue_update();
	void _update();
	Color _get_color(Vector2 p_point, PackedVector2Array &p_border, PackedVector2Array &p_fill, bool p_supersample);

protected:
	static void _bind_methods();

public:
	void set_width(int p_width);
	int get_width() const override;
	void set_height(int p_height);
	int get_height() const override;
	void set_points(int p_points);
	int get_points() const;
	void set_star_enabled(bool p_is_star);
	bool is_star_enabled() const;
	void set_star_inset(real_t p_star_inset);
	real_t get_star_inset() const;
	void set_rotation_degrees(real_t p_rotation_degrees);
	real_t get_rotation_degrees() const;
	void set_fill_color(Color p_fill_color);
	Color get_fill_color() const;
	void set_border_color(Color p_border_color);
	Color get_border_color() const;
	void set_border_width(real_t p_border_width);
	real_t get_border_width() const;
	void set_antialiased(bool p_antialiased);
	bool is_antialiased() const;

	bool is_using_hdr() const;

	virtual RID get_rid() const override;
	virtual bool has_alpha() const override { return true; }

	virtual Ref<Image> get_image() const override;
	void update_now();

	ShapeTexture2D();
	virtual ~ShapeTexture2D();
};

#endif // SHAPE_TEXTURE_2D_H
