/**************************************************************************/
/*  usd_mesh.cpp                                                          */
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

#include "usd_mesh.h"

void USDMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_original_name"), &USDMesh::get_original_name);
	ClassDB::bind_method(D_METHOD("set_original_name", "original_name"), &USDMesh::set_original_name);
	ClassDB::bind_method(D_METHOD("get_surface_count"), &USDMesh::get_surface_count);
	ClassDB::bind_method(D_METHOD("get_blend_shapes"), &USDMesh::get_blend_shapes);
	ClassDB::bind_method(D_METHOD("set_blend_shapes", "blend_shapes"), &USDMesh::set_blend_shapes);
	ClassDB::bind_method(D_METHOD("to_importer_mesh"), &USDMesh::to_importer_mesh);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "original_name"), "set_original_name", "get_original_name");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "blend_shapes"), "set_blend_shapes", "get_blend_shapes");
}

String USDMesh::get_original_name() const {
	return original_name;
}

void USDMesh::set_original_name(const String &p_name) {
	original_name = p_name;
}

int USDMesh::get_surface_count() const {
	return surfaces.size();
}

Vector<String> USDMesh::get_blend_shapes() const {
	return blend_shapes;
}

void USDMesh::set_blend_shapes(const Vector<String> &p_blend_shapes) {
	blend_shapes = p_blend_shapes;
}

void USDMesh::add_surface(const Array &p_arrays, Mesh::PrimitiveType p_primitive, int p_material, const String &p_name, const Vector<Array> &p_blend_shape_arrays) {
	Surface s;
	s.arrays = p_arrays;
	s.primitive = p_primitive;
	s.material = p_material;
	s.name = p_name;
	s.blend_shape_arrays = p_blend_shape_arrays;
	surfaces.push_back(s);
}

USDMesh::Surface USDMesh::get_surface(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, surfaces.size(), Surface());
	return surfaces[p_index];
}

Ref<ImporterMesh> USDMesh::to_importer_mesh() const {
	Ref<ImporterMesh> importer_mesh;
	importer_mesh.instantiate();

	// Register blend shape names first; ImporterMesh requires them
	// to be added before any surface that references them.
	for (int i = 0; i < blend_shapes.size(); i++) {
		importer_mesh->add_blend_shape(blend_shapes[i]);
	}

	for (int i = 0; i < surfaces.size(); i++) {
		const Surface &surface = surfaces[i];

		TypedArray<Array> bs_arrays;
		for (int j = 0; j < surface.blend_shape_arrays.size(); j++) {
			bs_arrays.push_back(surface.blend_shape_arrays[j]);
		}

		importer_mesh->add_surface(
				surface.primitive,
				surface.arrays,
				bs_arrays,
				Dictionary(),
				Ref<Material>(),
				surface.name,
				0);
	}

	return importer_mesh;
}
