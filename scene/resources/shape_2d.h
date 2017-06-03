/*************************************************************************/
/*  shape_2d.h                                                           */
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
#ifndef SHAPE_2D_H
#define SHAPE_2D_H

#include "resource.h"

class Shape2D : public Resource {
	GDCLASS(Shape2D, Resource);
	OBJ_SAVE_TYPE(Shape2D);

	RID shape;
	real_t custom_bias;

protected:
	static void _bind_methods();
	Shape2D(const RID &p_rid);

public:
	void set_custom_solver_bias(real_t p_bias);
	real_t get_custom_solver_bias() const;

	bool collide_with_motion(const Transform2D &p_local_xform, const Vector2 &p_local_motion, const Ref<Shape2D> &p_shape, const Transform2D &p_shape_xform, const Vector2 &p_p_shape_motion);
	bool collide(const Transform2D &p_local_xform, const Ref<Shape2D> &p_shape, const Transform2D &p_shape_xform);

	Variant collide_with_motion_and_get_contacts(const Transform2D &p_local_xform, const Vector2 &p_local_motion, const Ref<Shape2D> &p_shape, const Transform2D &p_shape_xform, const Vector2 &p_p_shape_motion);
	Variant collide_and_get_contacts(const Transform2D &p_local_xform, const Ref<Shape2D> &p_shape, const Transform2D &p_shape_xform);

	virtual void draw(const RID &p_to_rid, const Color &p_color) {}
	virtual Rect2 get_rect() const { return Rect2(); }
	virtual RID get_rid() const;
	Shape2D();
	~Shape2D();
};

#endif // SHAPE_2D_H
