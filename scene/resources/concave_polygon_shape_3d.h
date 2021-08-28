/*************************************************************************/
/*  concave_polygon_shape_3d.h                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CONCAVE_POLYGON_SHAPE_3D_H
#define CONCAVE_POLYGON_SHAPE_3D_H

#include "scene/resources/shape_3d.h"

class ConcavePolygonShape3D : public Shape3D {
	GDCLASS(ConcavePolygonShape3D, Shape3D);

	Vector<Vector3> faces;
	bool backface_collision = false;

	struct DrawEdge {
		Vector3 a;
		Vector3 b;
		bool operator<(const DrawEdge &p_edge) const {
			if (a == p_edge.a) {
				return b < p_edge.b;
			} else {
				return a < p_edge.a;
			}
		}

		DrawEdge(const Vector3 &p_a = Vector3(), const Vector3 &p_b = Vector3()) {
			a = p_a;
			b = p_b;
			if (a < b) {
				SWAP(a, b);
			}
		}
	};

protected:
	static void _bind_methods();

	virtual void _update_shape() override;

public:
	void set_faces(const Vector<Vector3> &p_faces);
	Vector<Vector3> get_faces() const;

	void set_backface_collision_enabled(bool p_enabled);
	bool is_backface_collision_enabled() const;

	virtual Vector<Vector3> get_debug_mesh_lines() const override;
	virtual real_t get_enclosing_radius() const override;

	ConcavePolygonShape3D();
};

#endif // CONCAVE_POLYGON_SHAPE_H
