/*************************************************************************/
/*  animated_sprite.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef ANIMATED_SPRITE_H
#define ANIMATED_SPRITE_H

#include "scene/2d/node_2d.h"
#include "scene/resources/texture.h"


class SpriteFrames : public Resource {

	OBJ_TYPE(SpriteFrames,Resource);

	Vector< Ref<Texture> > frames;

	Array _get_frames() const;
	void _set_frames(const Array& p_frames);
protected:

	static void _bind_methods();

public:


	void add_frame(const Ref<Texture>& p_frame,int p_at_pos=-1);
	int get_frame_count() const;
	_FORCE_INLINE_ Ref<Texture> get_frame(int p_idx) const { ERR_FAIL_INDEX_V(p_idx,frames.size(),Ref<Texture>()); return frames[p_idx]; }
	void set_frame(int p_idx,const Ref<Texture>& p_frame){ ERR_FAIL_INDEX(p_idx,frames.size()); frames[p_idx]=p_frame; }
	void remove_frame(int p_idx);
	void clear();

	SpriteFrames();

};



class AnimatedSprite : public Node2D {

	OBJ_TYPE(AnimatedSprite,Node2D);

	Ref<SpriteFrames> frames;
	int frame;

	bool centered;
	Point2 offset;

	bool hflip;
	bool vflip;

	Color modulate;

	void _res_changed();
protected:

	static void _bind_methods();
	void _notification(int p_what);

public:


	virtual void edit_set_pivot(const Point2& p_pivot);
	virtual Point2 edit_get_pivot() const;
	virtual bool edit_has_pivot() const;

	void set_sprite_frames(const Ref<SpriteFrames> &p_frames);
	Ref<SpriteFrames> get_sprite_frames() const;

	void set_frame(int p_frame);
	int get_frame() const;

	void set_centered(bool p_center);
	bool is_centered() const;

	void set_offset(const Point2& p_offset);
	Point2 get_offset() const;

	void set_flip_h(bool p_flip);
	bool is_flipped_h() const;

	void set_flip_v(bool p_flip);
	bool is_flipped_v() const;

	void set_modulate(const Color& p_color);
	Color get_modulate() const;

	virtual Rect2 get_item_rect() const;


	AnimatedSprite();
};

#endif // ANIMATED_SPRITE_H
