/**************************************************************************/
/*  nine_patch_sprite.h                                                   */
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
#include "scene/2d/node_2d.h"
#include "scene/gui/nine_patch_rect.h"
using AxisStretchMode = NinePatchRect::AxisStretchMode;
class NinePatchSprite : public Node2D {
	GDCLASS(NinePatchSprite, Node2D);

private:
	Vector2 size;
	Vector2 offset = { 0, 0 };
	bool centered = true;
	// Vector2 minimum_size;

public:
	bool draw_center = true;
	int margin[4] = {};
	Rect2 region_rect;

	Ref<Texture2D> texture;

	AxisStretchMode axis_h = AxisStretchMode::AXIS_STRETCH_MODE_STRETCH;
	AxisStretchMode axis_v = AxisStretchMode::AXIS_STRETCH_MODE_STRETCH;

	void _texture_changed();

protected:
	void _notification(int p_what);
	// virtual Size2 get_minimum_size() const override;
	static void _bind_methods();

public:
#ifdef TOOLS_ENABLED
	virtual Dictionary _edit_get_state() const override;
	virtual void _edit_set_state(const Dictionary &p_state) override;

	virtual void _edit_set_pivot(const Point2 &p_pivot) override;
	virtual Point2 _edit_get_pivot() const override;
	virtual bool _edit_use_pivot() const override;
	virtual void _edit_set_rect(const Rect2 &p_edit_rect) override;
	virtual Size2 _edit_get_minimum_size() const override;
#endif // TOOLS_ENABLED

#ifdef DEBUG_ENABLED
	virtual bool _edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const override;

	virtual Rect2 _edit_get_rect() const override;
	virtual bool _edit_use_rect() const override;
#endif // DEBUG_ENABLED
	bool get_centered() const;
	void set_centered(bool p_centered);
	Vector2 get_offset() const;
	void set_offset(Vector2 p_offset);
	Vector2 get_size() const;
	void set_size(Vector2 p_size);

	void set_texture(const Ref<Texture2D> &p_tex);
	Ref<Texture2D> get_texture() const;

	void set_patch_margin(Side p_side, int p_size);
	int get_patch_margin(Side p_side) const;

	void set_region_rect(const Rect2 &p_region_rect);
	Rect2 get_region_rect() const;

	void set_draw_center(bool p_enabled);
	bool is_draw_center_enabled() const;

	void set_h_axis_stretch_mode(AxisStretchMode p_mode);
	AxisStretchMode get_h_axis_stretch_mode() const;

	void set_v_axis_stretch_mode(AxisStretchMode p_mode);
	AxisStretchMode get_v_axis_stretch_mode() const;

	Rect2 get_rect() const;
	NinePatchSprite();
	~NinePatchSprite();
};
