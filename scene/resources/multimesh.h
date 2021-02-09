/*************************************************************************/
/*  multimesh.h                                                          */
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

#ifndef MULTIMESH_H
#define MULTIMESH_H

#include "scene/resources/mesh.h"
#include "servers/rendering_server.h"

class MultiMesh : public Resource {
	GDCLASS(MultiMesh, Resource);
	RES_BASE_EXTENSION("multimesh");

public:
	enum TransformFormat {
		TRANSFORM_2D = RS::MULTIMESH_TRANSFORM_2D,
		TRANSFORM_3D = RS::MULTIMESH_TRANSFORM_3D
	};

private:
	Ref<Mesh> mesh;
	RID multimesh;
	TransformFormat transform_format = TRANSFORM_2D;
	bool use_colors = false;
	bool use_custom_data = false;
	int instance_count = 0;
	int visible_instance_count = -1;

protected:
	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	// Kept for compatibility from 3.x to 4.0.
	void _set_transform_array(const Vector<Vector3> &p_array);
	Vector<Vector3> _get_transform_array() const;

	void _set_transform_2d_array(const Vector<Vector2> &p_array);
	Vector<Vector2> _get_transform_2d_array() const;

	void _set_color_array(const Vector<Color> &p_array);
	Vector<Color> _get_color_array() const;

	void _set_custom_data_array(const Vector<Color> &p_array);
	Vector<Color> _get_custom_data_array() const;
#endif
	void set_buffer(const Vector<float> &p_buffer);
	Vector<float> get_buffer() const;

public:
	void set_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_mesh() const;

	void set_use_colors(bool p_enable);
	bool is_using_colors() const;

	void set_use_custom_data(bool p_enable);
	bool is_using_custom_data() const;

	void set_transform_format(TransformFormat p_transform_format);
	TransformFormat get_transform_format() const;

	void set_instance_count(int p_count);
	int get_instance_count() const;

	void set_visible_instance_count(int p_count);
	int get_visible_instance_count() const;

	void set_instance_transform(int p_instance, const Transform &p_transform);
	void set_instance_transform_2d(int p_instance, const Transform2D &p_transform);
	Transform get_instance_transform(int p_instance) const;
	Transform2D get_instance_transform_2d(int p_instance) const;

	void set_instance_color(int p_instance, const Color &p_color);
	Color get_instance_color(int p_instance) const;

	void set_instance_custom_data(int p_instance, const Color &p_custom_data);
	Color get_instance_custom_data(int p_instance) const;

	virtual AABB get_aabb() const;

	virtual RID get_rid() const override;

	MultiMesh();
	~MultiMesh();
};

VARIANT_ENUM_CAST(MultiMesh::TransformFormat);

#endif // MULTI_MESH_H
