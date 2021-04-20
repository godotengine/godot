/*************************************************************************/
/*  occluder_instance_3d.h                                               */
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

#ifndef OCCLUDER_INSTANCE_3D_H
#define OCCLUDER_INSTANCE_3D_H

#include "scene/3d/visual_instance_3d.h"

class Occluder3D : public Resource {
	GDCLASS(Occluder3D, Resource);
	RES_BASE_EXTENSION("occ");

	mutable RID occluder;
	mutable Ref<ArrayMesh> debug_mesh;
	mutable Vector<Vector3> debug_lines;
	AABB aabb;

	PackedVector3Array vertices;
	PackedInt32Array indices;

	void _update_changes();

protected:
	static void _bind_methods();

public:
	void set_vertices(PackedVector3Array p_vertices);
	PackedVector3Array get_vertices() const;

	void set_indices(PackedInt32Array p_indices);
	PackedInt32Array get_indices() const;

	Vector<Vector3> get_debug_lines() const;
	Ref<ArrayMesh> get_debug_mesh() const;
	AABB get_aabb() const;

	virtual RID get_rid() const override;
	Occluder3D();
	~Occluder3D();
};

class OccluderInstance3D : public VisualInstance3D {
	GDCLASS(OccluderInstance3D, Node3D);

private:
	Ref<Occluder3D> occluder;
	uint32_t bake_mask = 0xFFFFFFFF;

	void _occluder_changed();

	bool _bake_material_check(Ref<Material> p_material);
	void _bake_node(Node *p_node, PackedVector3Array &r_vertices, PackedInt32Array &r_indices);

protected:
	static void _bind_methods();

public:
	enum BakeError {
		BAKE_ERROR_OK,
		BAKE_ERROR_NO_SAVE_PATH,
		BAKE_ERROR_NO_MESHES,
	};

	void set_occluder(const Ref<Occluder3D> &p_occluder);
	Ref<Occluder3D> get_occluder() const;

	virtual AABB get_aabb() const override;
	virtual Vector<Face3> get_faces(uint32_t p_usage_flags) const override;

	void set_bake_mask(uint32_t p_mask);
	uint32_t get_bake_mask() const;

	void set_bake_mask_bit(int p_layer, bool p_enable);
	bool get_bake_mask_bit(int p_layer) const;
	BakeError bake(Node *p_from_node, String p_occluder_path = "");

	OccluderInstance3D();
	~OccluderInstance3D();
};

#endif
