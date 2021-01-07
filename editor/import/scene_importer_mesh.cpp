/*************************************************************************/
/*  scene_importer_mesh.cpp                                              */
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

#include "scene_importer_mesh.h"

#include "scene/resources/surface_tool.h"

void EditorSceneImporterMesh::add_blend_shape(const String &p_name) {
	ERR_FAIL_COND(surfaces.size() > 0);
	blend_shapes.push_back(p_name);
}

int EditorSceneImporterMesh::get_blend_shape_count() const {
	return blend_shapes.size();
}

String EditorSceneImporterMesh::get_blend_shape_name(int p_blend_shape) const {
	ERR_FAIL_INDEX_V(p_blend_shape, blend_shapes.size(), String());
	return blend_shapes[p_blend_shape];
}

void EditorSceneImporterMesh::set_blend_shape_mode(Mesh::BlendShapeMode p_blend_shape_mode) {
	blend_shape_mode = p_blend_shape_mode;
}

Mesh::BlendShapeMode EditorSceneImporterMesh::get_blend_shape_mode() const {
	return blend_shape_mode;
}

void EditorSceneImporterMesh::add_surface(Mesh::PrimitiveType p_primitive, const Array &p_arrays, const Array &p_blend_shapes, const Dictionary &p_lods, const Ref<Material> &p_material, const String &p_name) {
	ERR_FAIL_COND(p_blend_shapes.size() != blend_shapes.size());
	ERR_FAIL_COND(p_arrays.size() != Mesh::ARRAY_MAX);
	Surface s;
	s.primitive = p_primitive;
	s.arrays = p_arrays;
	s.name = p_name;

	Vector<Vector3> vertex_array = p_arrays[Mesh::ARRAY_VERTEX];
	int vertex_count = vertex_array.size();
	ERR_FAIL_COND(vertex_count == 0);

	for (int i = 0; i < blend_shapes.size(); i++) {
		Array bsdata = p_blend_shapes[i];
		ERR_FAIL_COND(bsdata.size() != Mesh::ARRAY_MAX);
		Vector<Vector3> vertex_data = bsdata[Mesh::ARRAY_VERTEX];
		ERR_FAIL_COND(vertex_data.size() != vertex_count);
		Surface::BlendShape bs;
		bs.arrays = bsdata;
		s.blend_shape_data.push_back(bs);
	}

	List<Variant> lods;
	p_lods.get_key_list(&lods);
	for (List<Variant>::Element *E = lods.front(); E; E = E->next()) {
		ERR_CONTINUE(!E->get().is_num());
		Surface::LOD lod;
		lod.distance = E->get();
		lod.indices = p_lods[E->get()];
		ERR_CONTINUE(lod.indices.size() == 0);
		s.lods.push_back(lod);
	}

	s.material = p_material;

	surfaces.push_back(s);
	mesh.unref();
}

int EditorSceneImporterMesh::get_surface_count() const {
	return surfaces.size();
}

Mesh::PrimitiveType EditorSceneImporterMesh::get_surface_primitive_type(int p_surface) {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Mesh::PRIMITIVE_MAX);
	return surfaces[p_surface].primitive;
}
Array EditorSceneImporterMesh::get_surface_arrays(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Array());
	return surfaces[p_surface].arrays;
}
String EditorSceneImporterMesh::get_surface_name(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), String());
	return surfaces[p_surface].name;
}
Array EditorSceneImporterMesh::get_surface_blend_shape_arrays(int p_surface, int p_blend_shape) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Array());
	ERR_FAIL_INDEX_V(p_blend_shape, surfaces[p_surface].blend_shape_data.size(), Array());
	return surfaces[p_surface].blend_shape_data[p_blend_shape].arrays;
}
int EditorSceneImporterMesh::get_surface_lod_count(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), 0);
	return surfaces[p_surface].lods.size();
}
Vector<int> EditorSceneImporterMesh::get_surface_lod_indices(int p_surface, int p_lod) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Vector<int>());
	ERR_FAIL_INDEX_V(p_lod, surfaces[p_surface].lods.size(), Vector<int>());

	return surfaces[p_surface].lods[p_lod].indices;
}

float EditorSceneImporterMesh::get_surface_lod_size(int p_surface, int p_lod) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), 0);
	ERR_FAIL_INDEX_V(p_lod, surfaces[p_surface].lods.size(), 0);
	return surfaces[p_surface].lods[p_lod].distance;
}

Ref<Material> EditorSceneImporterMesh::get_surface_material(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Ref<Material>());
	return surfaces[p_surface].material;
}

void EditorSceneImporterMesh::generate_lods() {
	if (!SurfaceTool::simplify_func) {
		return;
	}

	for (int i = 0; i < surfaces.size(); i++) {
		if (surfaces[i].primitive != Mesh::PRIMITIVE_TRIANGLES) {
			continue;
		}

		surfaces.write[i].lods.clear();
		Vector<Vector3> vertices = surfaces[i].arrays[RS::ARRAY_VERTEX];
		Vector<int> indices = surfaces[i].arrays[RS::ARRAY_INDEX];
		if (indices.size() == 0) {
			continue; //no lods if no indices
		}
		uint32_t vertex_count = vertices.size();
		const Vector3 *vertices_ptr = vertices.ptr();

		int min_indices = 10;
		int index_target = indices.size() / 2;
		print_line("total: " + itos(indices.size()));
		while (index_target > min_indices) {
			float error;
			Vector<int> new_indices;
			new_indices.resize(indices.size());
			size_t new_len = SurfaceTool::simplify_func((unsigned int *)new_indices.ptrw(), (const unsigned int *)indices.ptr(), indices.size(), (const float *)vertices_ptr, vertex_count, sizeof(Vector3), index_target, 1e20, &error);
			print_line("shoot for " + itos(index_target) + ", got " + itos(new_len) + " distance " + rtos(error));
			if ((int)new_len > (index_target * 120 / 100)) {
				break; // 20 percent tolerance
			}
			new_indices.resize(new_len);
			Surface::LOD lod;
			lod.distance = error;
			lod.indices = new_indices;
			surfaces.write[i].lods.push_back(lod);
			index_target /= 2;
		}
	}
}

bool EditorSceneImporterMesh::has_mesh() const {
	return mesh.is_valid();
}

Ref<ArrayMesh> EditorSceneImporterMesh::get_mesh() {
	ERR_FAIL_COND_V(surfaces.size() == 0, Ref<ArrayMesh>());

	if (mesh.is_null()) {
		mesh.instance();
		for (int i = 0; i < blend_shapes.size(); i++) {
			mesh->add_blend_shape(blend_shapes[i]);
		}
		mesh->set_blend_shape_mode(blend_shape_mode);
		for (int i = 0; i < surfaces.size(); i++) {
			Array bs_data;
			if (surfaces[i].blend_shape_data.size()) {
				for (int j = 0; j < surfaces[i].blend_shape_data.size(); j++) {
					bs_data.push_back(surfaces[i].blend_shape_data[j].arrays);
				}
			}
			Dictionary lods;
			if (surfaces[i].lods.size()) {
				for (int j = 0; j < surfaces[i].lods.size(); j++) {
					lods[surfaces[i].lods[j].distance] = surfaces[i].lods[j].indices;
				}
			}

			mesh->add_surface_from_arrays(surfaces[i].primitive, surfaces[i].arrays, bs_data, lods);
			if (surfaces[i].material.is_valid()) {
				mesh->surface_set_material(mesh->get_surface_count() - 1, surfaces[i].material);
			}
			if (surfaces[i].name != String()) {
				mesh->surface_set_name(mesh->get_surface_count() - 1, surfaces[i].name);
			}
		}
	}

	return mesh;
}

void EditorSceneImporterMesh::clear() {
	surfaces.clear();
	blend_shapes.clear();
	mesh.unref();
}

void EditorSceneImporterMesh::_set_data(const Dictionary &p_data) {
	clear();
	if (p_data.has("blend_shape_names")) {
		blend_shapes = p_data["blend_shape_names"];
	}
	if (p_data.has("surfaces")) {
		Array surface_arr = p_data["surfaces"];
		for (int i = 0; i < surface_arr.size(); i++) {
			Dictionary s = surface_arr[i];
			ERR_CONTINUE(!s.has("primitive"));
			ERR_CONTINUE(!s.has("arrays"));
			Mesh::PrimitiveType prim = Mesh::PrimitiveType(int(s["primitive"]));
			ERR_CONTINUE(prim >= Mesh::PRIMITIVE_MAX);
			Array arr = s["arrays"];
			Dictionary lods;
			String name;
			if (s.has("name")) {
				name = s["name"];
			}
			if (s.has("lods")) {
				lods = s["lods"];
			}
			Array blend_shapes;
			if (s.has("blend_shapes")) {
				blend_shapes = s["blend_shapes"];
			}
			Ref<Material> material;
			if (s.has("material")) {
				material = s["material"];
			}
			add_surface(prim, arr, blend_shapes, lods, material, name);
		}
	}
}
Dictionary EditorSceneImporterMesh::_get_data() const {
	Dictionary data;
	if (blend_shapes.size()) {
		data["blend_shape_names"] = blend_shapes;
	}
	Array surface_arr;
	for (int i = 0; i < surfaces.size(); i++) {
		Dictionary d;
		d["primitive"] = surfaces[i].primitive;
		d["arrays"] = surfaces[i].arrays;
		if (surfaces[i].blend_shape_data.size()) {
			Array bs_data;
			for (int j = 0; j < surfaces[i].blend_shape_data.size(); j++) {
				bs_data.push_back(surfaces[i].blend_shape_data[j].arrays);
			}
			d["blend_shapes"] = bs_data;
		}
		if (surfaces[i].lods.size()) {
			Dictionary lods;
			for (int j = 0; j < surfaces[i].lods.size(); j++) {
				lods[surfaces[i].lods[j].distance] = surfaces[i].lods[j].indices;
			}
			d["lods"] = lods;
		}

		if (surfaces[i].material.is_valid()) {
			d["material"] = surfaces[i].material;
		}

		if (surfaces[i].name != String()) {
			d["name"] = surfaces[i].name;
		}

		surface_arr.push_back(d);
	}
	data["surfaces"] = surface_arr;
	return data;
}

void EditorSceneImporterMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_blend_shape", "name"), &EditorSceneImporterMesh::add_blend_shape);
	ClassDB::bind_method(D_METHOD("get_blend_shape_count"), &EditorSceneImporterMesh::get_blend_shape_count);
	ClassDB::bind_method(D_METHOD("get_blend_shape_name", "blend_shape_idx"), &EditorSceneImporterMesh::get_blend_shape_name);

	ClassDB::bind_method(D_METHOD("set_blend_shape_mode", "mode"), &EditorSceneImporterMesh::set_blend_shape_mode);
	ClassDB::bind_method(D_METHOD("get_blend_shape_mode"), &EditorSceneImporterMesh::get_blend_shape_mode);

	ClassDB::bind_method(D_METHOD("add_surface", "primitive", "arrays", "blend_shapes", "lods", "material"), &EditorSceneImporterMesh::add_surface, DEFVAL(Array()), DEFVAL(Dictionary()), DEFVAL(Ref<Material>()), DEFVAL(String()));

	ClassDB::bind_method(D_METHOD("get_surface_count"), &EditorSceneImporterMesh::get_surface_count);
	ClassDB::bind_method(D_METHOD("get_surface_primitive_type", "surface_idx"), &EditorSceneImporterMesh::get_surface_primitive_type);
	ClassDB::bind_method(D_METHOD("get_surface_name", "surface_idx"), &EditorSceneImporterMesh::get_surface_name);
	ClassDB::bind_method(D_METHOD("get_surface_arrays", "surface_idx"), &EditorSceneImporterMesh::get_surface_arrays);
	ClassDB::bind_method(D_METHOD("get_surface_blend_shape_arrays", "surface_idx", "blend_shape_idx"), &EditorSceneImporterMesh::get_surface_blend_shape_arrays);
	ClassDB::bind_method(D_METHOD("get_surface_lod_count", "surface_idx"), &EditorSceneImporterMesh::get_surface_lod_count);
	ClassDB::bind_method(D_METHOD("get_surface_lod_size", "surface_idx", "lod_idx"), &EditorSceneImporterMesh::get_surface_lod_size);
	ClassDB::bind_method(D_METHOD("get_surface_lod_indices", "surface_idx", "lod_idx"), &EditorSceneImporterMesh::get_surface_lod_indices);
	ClassDB::bind_method(D_METHOD("get_surface_material", "surface_idx"), &EditorSceneImporterMesh::get_surface_material);

	ClassDB::bind_method(D_METHOD("get_mesh"), &EditorSceneImporterMesh::get_mesh);
	ClassDB::bind_method(D_METHOD("clear"), &EditorSceneImporterMesh::clear);

	ClassDB::bind_method(D_METHOD("_set_data", "data"), &EditorSceneImporterMesh::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &EditorSceneImporterMesh::_get_data);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_data", "_get_data");
}
