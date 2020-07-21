/*************************************************************************/
/*  navigation_2d.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef NAVIGATION_2D_H
#define NAVIGATION_2D_H

#include "scene/2d/navigation_region_2d.h"
#include "scene/2d/node_2d.h"

class Navigation2D : public Node2D {
	GDCLASS(Navigation2D, Node2D);

	RID map;
	real_t cell_size;
	real_t edge_connection_margin;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	RID get_rid() const {
		return map;
	}

	void set_cell_size(float p_cell_size);
	float get_cell_size() const {
		return cell_size;
	}

	void set_edge_connection_margin(float p_edge_connection_margin);
	float get_edge_connection_margin() const {
		return edge_connection_margin;
	}

	Vector<Vector2> get_simple_path(const Vector2 &p_start, const Vector2 &p_end, bool p_optimize = true) const;
	Vector2 get_closest_point(const Vector2 &p_point) const;
	RID get_closest_point_owner(const Vector2 &p_point) const;

	Navigation2D();
	~Navigation2D();
};

#endif // NAVIGATION_2D_H
