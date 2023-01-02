/*************************************************************************/
/*  convex_mesh_shape_3d.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CONVEX_MESH_SHAPE_3D_H
#define CONVEX_MESH_SHAPE_3D_H

#include "scene/resources/mesh.h"
#include "scene/resources/shape_3d.h"

class ConvexMeshShape3D : public Shape3D {
	GDCLASS(ConvexMeshShape3D, Shape3D);

	Ref<Mesh> mesh;
	bool clean = true;
	bool simplify = false;
	Vector<Vector3> points;

protected:
	static void _bind_methods();

	virtual void _update_shape() override;
	void _update_mesh();

public:
	void set_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_mesh() const;

	void set_clean(bool p_clean);
	bool is_clean() const;

	void set_simplify(bool p_simplify);
	bool is_simplify() const;

	void set_points(const Vector<Vector3> &p_points);
	Vector<Vector3> get_points() const;

	virtual Vector<Vector3> get_debug_mesh_lines() const override;
	virtual real_t get_enclosing_radius() const override;

	ConvexMeshShape3D();
	ConvexMeshShape3D(const Ref<Mesh> &p_mesh);
};

#endif // CONVEX_MESH_SHAPE_3D_H
