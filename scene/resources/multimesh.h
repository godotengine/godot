/*************************************************************************/
/*  multimesh.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef MULTIMESH_H
#define MULTIMESH_H

#include "scene/resources/mesh.h"

class MultiMesh : public Resource {

	OBJ_TYPE(MultiMesh, Resource);
	RES_BASE_EXTENSION("mmsh");

	AABB aabb;
	Ref<Mesh> mesh;
	RID multimesh;

protected:
	static void _bind_methods();

	void _set_transform_array(const DVector<Vector3> &p_array);
	DVector<Vector3> _get_transform_array() const;

	void _set_color_array(const DVector<Color> &p_array);
	DVector<Color> _get_color_array() const;

public:
	void set_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_mesh() const;

	void set_instance_count(int p_count);
	int get_instance_count() const;

	void set_instance_transform(int p_instance, const Transform &p_transform);
	Transform get_instance_transform(int p_instance) const;

	void set_instance_color(int p_instance, const Color &p_color);
	Color get_instance_color(int p_instance) const;

	void set_aabb(const AABB &p_aabb);
	virtual AABB get_aabb() const;

	void generate_aabb();

	virtual RID get_rid() const;

	MultiMesh();
	~MultiMesh();
};

#endif // MULTI_MESH_H
