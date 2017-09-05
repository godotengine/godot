/*************************************************************************/
/*  editable_polygon_2d.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef POLYGONNODE2D_H
#define POLYGONNODE2D_H

#include "core/undo_redo.h"
#include "core/resource.h"
#include "scene/resources/texture.h"
#include "scene/2d/node_2d.h"
#include "scene/2d/ring_2d.h"

class PolygonNode2D : public Node2D {

	GDCLASS(PolygonNode2D, Node2D);

protected:
	static void _bind_methods();

public:
	virtual int get_polygon_count() const = 0;
	virtual Ref<Resource> get_nth_polygon(int p_idx) const = 0;
	virtual int get_ring_count(Ref<Resource> p_polygon) const = 0;
	virtual Ref<Ring2D> get_nth_ring(Ref<Resource> p_polygon, int p_idx) const = 0;

	virtual void add_polygon(const Vector<Point2> &p_vertices);
	virtual Ref<Resource> new_polygon(const Ref<Ring2D> &p_ring) const = 0;
	virtual void add_polygon_at_index(Ref<Resource> p_polygon, int p_idx) = 0;
	virtual void remove_polygon(int p_idx) = 0;
};

#endif // POLYGONNODE2D_H
