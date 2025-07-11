/**************************************************************************/
/*  spx_ext_pen.h                                                      */
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

#ifndef SPX_PEN_H
#define SPX_PEN_H

#include "gdextension_spx_ext.h"
#include "scene/2d/line_2d.h"
#include "scene/2d/node_2d.h"
#include "scene/2d/sprite_2d.h"
#include "spx_base_mgr.h"

class SpxSprite;
class SpxPen {
private:
	GdObj id;
	Node *root;
	Line2D *current_line = nullptr;
	bool is_pen_down = false;
	float min_draw_distance = 1.0f;

	struct PenProperties {
		Color color = Color(0, 0, 0, 1); // BLACK
		float size = 2.0f;
		float saturation = 1.0f;
		float brightness = 1.0f;
		float transparency = 0.0f;
	} pen_properties;

	Vector2 current_pen_pos;
	bool move_by_mouse = false;

	Ref<Texture2D> stamp_texture;

private:
	Line2D *_create_new_line();
	void _start_new_line();
	Color _get_current_color() const;

public:
	void on_create(GdInt id, Node *root);
	void on_destroy();
	void on_update(float delta);

public:
	// Pen APIs
	void erase_all();
	GdObj get_id();
	void on_down(GdBool move_by_mouse);
	void on_up();
	void stamp();
	void move_to(GdVec2 position);
	void set_color_to(GdColor color);
	void change_by(GdInt property, GdFloat amount);
	void set_to(GdInt property, GdFloat value);
	void change_size_by(GdFloat amount);
	void set_size_to(GdFloat size);
	void set_stamp_texture(GdString texture_path);
};


#endif // SPX_PEN_H
