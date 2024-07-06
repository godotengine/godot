/**************************************************************************/
/*  multimesh_instance_3d.cpp                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "multimesh_instance_3d.h"

void MultiMeshInstance3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_multimesh", "multimesh"), &MultiMeshInstance3D::set_multimesh);
	ClassDB::bind_method(D_METHOD("get_multimesh"), &MultiMeshInstance3D::get_multimesh);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "multimesh", PROPERTY_HINT_RESOURCE_TYPE, "MultiMesh"), "set_multimesh", "get_multimesh");
}

void MultiMeshInstance3D::set_multimesh(const Ref<MultiMesh> &p_multimesh) {
	multimesh = p_multimesh;
	if (multimesh.is_valid()) {
		set_base(multimesh->get_rid());
	} else {
		set_base(RID());
	}
}

Ref<MultiMesh> MultiMeshInstance3D::get_multimesh() const {
	return multimesh;
}

Array MultiMeshInstance3D::get_meshes() const {
	if (multimesh.is_null() || multimesh->get_mesh().is_null() || multimesh->get_transform_format() != MultiMesh::TransformFormat::TRANSFORM_3D) {
		return Array();
	}

	int count = multimesh->get_visible_instance_count();
	if (count == -1) {
		count = multimesh->get_instance_count();
	}

	Ref<Mesh> mesh = multimesh->get_mesh();

	Array results;
	for (int i = 0; i < count; i++) {
		results.push_back(multimesh->get_instance_transform(i));
		results.push_back(mesh);
	}
	return results;
}

AABB MultiMeshInstance3D::get_aabb() const {
	if (multimesh.is_null()) {
		return AABB();
	} else {
		return multimesh->get_aabb();
	}
}

MultiMeshInstance3D::MultiMeshInstance3D() {
}

MultiMeshInstance3D::~MultiMeshInstance3D() {
}
