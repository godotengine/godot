/*************************************************************************/
/*  sprite.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef SPRITE_H
#define SPRITE_H

#include "scene/2d/node_2d.h"
#include "scene/resources/texture.h"

class Sprite : public Node2D {

	GDCLASS(Sprite, Node2D);

	Ref<Texture> texture;

	bool centered;
	Point2 offset;

	bool hflip;
	bool vflip;
	bool region;
	Rect2 region_rect;

	int frame;

	int vframes;
	int hframes;

protected:
	void _notification(int p_what);

	static void _bind_methods();

	virtual void _validate_property(PropertyInfo &property) const;

public:
	virtual void edit_set_pivot(const Point2 &p_pivot);
	virtual Point2 edit_get_pivot() const;
	virtual bool edit_has_pivot() const;

	void set_texture(const Ref<Texture> &p_texture);
	Ref<Texture> get_texture() const;

	void set_centered(bool p_center);
	bool is_centered() const;

	void set_offset(const Point2 &p_offset);
	Point2 get_offset() const;

	void set_flip_h(bool p_flip);
	bool is_flipped_h() const;

	void set_flip_v(bool p_flip);
	bool is_flipped_v() const;

	void set_region(bool p_region);
	bool is_region() const;

	void set_region_rect(const Rect2 &p_region_rect);
	Rect2 get_region_rect() const;

	void set_frame(int p_frame);
	int get_frame() const;

	void set_vframes(int p_amount);
	int get_vframes() const;

	void set_hframes(int p_amount);
	int get_hframes() const;

	virtual Rect2 get_item_rect() const;

	Sprite();
};

#if 0
class ViewportSprite : public Node2D {

	GDCLASS( ViewportSprite, Node2D );

	Ref<Texture> texture;
	NodePath viewport_path;

	bool centered;
	Point2 offset;
	Color modulate;

protected:

	void _notification(int p_what);

	static void _bind_methods();

public:

	virtual void edit_set_pivot(const Point2& p_pivot);
	virtual Point2 edit_get_pivot() const;
	virtual bool edit_has_pivot() const;

	void set_viewport_path(const NodePath& p_viewport);
	NodePath get_viewport_path() const;

	void set_centered(bool p_center);
	bool is_centered() const;

	void set_offset(const Point2& p_offset);
	Point2 get_offset() const;

	void set_modulate(const Color& p_color);
	Color get_modulate() const;

	virtual Rect2 get_item_rect() const;

	virtual String get_configuration_warning() const;

	ViewportSprite();
};

#endif
#endif // SPRITE_H
