/*************************************************************************/
/*  editable_shape.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef EDITABLE_SHAPE_H
#define EDITABLE_SHAPE_H

#include "scene/3d/spatial.h"
#include "scene/resources/shape.h"

class EditableShape : public Spatial {

	OBJ_TYPE(EditableShape,Spatial);

	//can hold either of those
	BSP_Tree bsp;
	Ref<Shape> shape;

	void _update_parent();
protected:

	void _notification(int p_what);

	void set_bsp_tree(const BSP_Tree& p_bsp);
	void set_shape(const Ref<Shape>& p_shape);
public:
	EditableShape();
};


class EditableSphere : public EditableShape {

	OBJ_TYPE( EditableSphere, EditableShape );


	float radius;
protected:

	static void _bind_methods();
public:

	void set_radius(float p_radius);
	float get_radius() const;

	EditableSphere();
};


#endif // EDITABLE_SHAPE_H
