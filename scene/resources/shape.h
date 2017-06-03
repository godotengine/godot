/*************************************************************************/
/*  shape.h                                                              */
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
#ifndef SHAPE_H
#define SHAPE_H

#include "resource.h"
class Mesh;

class Shape : public Resource {

	GDCLASS(Shape, Resource);
	OBJ_SAVE_TYPE(Shape);
	RES_BASE_EXTENSION("shp");
	RID shape;

	Ref<Mesh> debug_mesh_cache;

protected:
	_FORCE_INLINE_ RID get_shape() const { return shape; }
	Shape(RID p_shape);

	virtual Vector<Vector3> _gen_debug_mesh_lines() = 0; // { return Vector<Vector3>(); }
public:
	virtual RID get_rid() const { return shape; }

	Ref<Mesh> get_debug_mesh();

	void add_vertices_to_array(PoolVector<Vector3> &array, const Transform &p_xform);

	Shape();
	~Shape();
};

#endif // SHAPE_H
