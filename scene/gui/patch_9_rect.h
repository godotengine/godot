/*************************************************************************/
/*  patch_9_rect.h                                                       */
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
#ifndef PATCH_9_FRAME_H
#define PATCH_9_FRAME_H

#include "scene/gui/control.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class NinePatchRect : public Control {

	GDCLASS(NinePatchRect, Control);

	bool draw_center;
	int margin[4];
	Rect2 region_rect;
	Ref<Texture> texture;

protected:
	void _notification(int p_what);
	virtual Size2 get_minimum_size() const;
	static void _bind_methods();

public:
	void set_texture(const Ref<Texture> &p_tex);
	Ref<Texture> get_texture() const;

	void set_patch_margin(Margin p_margin, int p_size);
	int get_patch_margin(Margin p_margin) const;

	void set_region_rect(const Rect2 &p_region_rect);
	Rect2 get_region_rect() const;

	void set_draw_center(bool p_enable);
	bool get_draw_center() const;

	NinePatchRect();
	~NinePatchRect();
};
#endif // PATCH_9_FRAME_H
