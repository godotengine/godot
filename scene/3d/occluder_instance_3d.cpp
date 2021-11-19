/*************************************************************************/
/*  occluder_instance_3d.cpp                                             */
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

#include "occluder_instance_3d.h"
#include "core/core_string_names.h"
#include "scene/3d/mesh_instance_3d.h"

RID Occluder3D::get_rid() const {
	if (!occluder.is_valid()) {
		occluder = RS::get_singleton()->occluder_create();
		RS::get_singleton()->occluder_set_mesh(occluder, vertices, indices);
	}
	return occluder;
}

void Occluder3D::set_vertices(PackedVector3Array p_vertices) {
	vertices = p_vertices;
	if (occluder.is_valid()) {
		RS::get_singleton()->occluder_set_mesh(occluder, vertices, indices);
	}
	_update_changes();
}

PackedVector3Array Occluder3D::get_vertices() const {
	return vertices;
}

void Occluder3D::set_indices(PackedInt32Array p_indices) {
	indices = p_indices;
	if (occluder.is_valid()) {
		RS::get_singleton()->occluder_set_mesh(occluder, vertices, indices);
	}
	_update_changes();
}

PackedInt32Array Occluder3D::get_indices() const {
	return indices;
}

void Occluder3D::_update_changes() {
	aabb = AABB();

	const Vector3 *ptr = vertices.ptr();
	for (int i = 0; i < vertices.size(); i++) {
		aabb.expand_to(ptr[i]);
	}

	debug_lines.clear();
	debug_mesh.unref();

	emit_changed();
}

Vector<Vector3> Occluder3D::get_debug_lines() const {
	if (!debug_lines.is_empty()) {
		return debug_lines;
	}

	if (indices.size() % 3 != 0) {
		return Vector<Vector3>();
	}

	for (int i = 0; i < indices.size() / 3; i++) {
		for (int j = 0; j < 3; j++) {
			int a = indices[i * 3 + j];
			int b = indices[i * 3 + (j + 1) % 3];
			ERR_FAIL_INDEX_V_MSG(a, vertices.size(), Vector<Vector3>(), "Occluder indices are out of range.");
			ERR_FAIL_INDEX_V_MSG(b, vertices.size(), Vector<Vector3>(), "Occluder indices are out of range.");
			debug_lines.push_back(vertices[a]);
			debug_lines.push_back(vertices[b]);
		}
	}
	return debug_lines;
}

Ref<ArrayMesh> Occluder3D::get_debug_mesh() const {
	if (debug_mesh.is_valid()) {
		return debug_mesh;
	}

	if (indices.size() % 3 != 0) {
		return debug_mesh;
	}

	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_INDEX] = indices;

	debug_mesh.instantiate();
	debug_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	return debug_mesh;
}

AABB Occluder3D::get_aabb() const {
	return aabb;
}

void Occluder3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_vertices", "vertices"), &Occluder3D::set_vertices);
	ClassDB::bind_method(D_METHOD("get_vertices"), &Occluder3D::get_vertices);

	ClassDB::bind_method(D_METHOD("set_indices", "indices"), &Occluder3D::set_indices);
	ClassDB::bind_method(D_METHOD("get_indices"), &Occluder3D::get_indices);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR3_ARRAY, "vertices", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_vertices", "get_vertices");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "indices", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_indices", "get_indices");
}

Occluder3D::Occluder3D() {
}

Occluder3D::~Occluder3D() {
	if (occluder.is_valid()) {
		RS::get_singleton()->free(occluder);
	}
}
/////////////////////////////////////////////////

AABB OccluderInstance3D::get_aabb() const {
	if (occluder.is_valid()) {
		return occluder->get_aabb();
	}
	return AABB();
}

Vector<Face3> OccluderInstance3D::get_faces(uint32_t p_usage_flags) const {
	return Vector<Face3>();
}

void OccluderInstance3D::set_occluder(const Ref<Occluder3D> &p_occluder) {
	if (occluder == p_occluder) {
		return;
	}

	if (occluder.is_valid()) {
		occluder->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &OccluderInstance3D::_occluder_changed));
	}

	occluder = p_occluder;

	if (occluder.is_valid()) {
		set_base(occluder->get_rid());
		occluder->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &OccluderInstance3D::_occluder_changed));
	} else {
		set_base(RID());
	}

	update_gizmos();
	update_configuration_warnings();
}

void OccluderInstance3D::_occluder_changed() {
	update_gizmos();
	update_configuration_warnings();
}

Ref<Occluder3D> OccluderInstance3D::get_occluder() const {
	return occluder;
}

void OccluderInstance3D::set_bake_mask(uint32_t p_mask) {
	bake_mask = p_mask;
	update_configuration_warnings();
}

uint32_t OccluderInstance3D::get_bake_mask() const {
	return bake_mask;
}

void OccluderInstance3D::set_bake_mask_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Render layer number must be between 1 and 20 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 20, "Render layer number must be between 1 and 20 inclusive.");
	uint32_t mask = get_bake_mask();
	if (p_value) {
		mask |= 1 << (p_layer_number - 1);
	} else {
		mask &= ~(1 << (p_layer_number - 1));
	}
	set_bake_mask(mask);
}

bool OccluderInstance3D::get_bake_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Render layer number must be between 1 and 20 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 20, false, "Render layer number must be between 1 and 20 inclusive.");
	return bake_mask & (1 << (p_layer_number - 1));
}

bool OccluderInstance3D::_bake_material_check(Ref<Material> p_material) {
	StandardMaterial3D *standard_mat = Object::cast_to<StandardMaterial3D>(p_material.ptr());
	if (standard_mat && standard_mat->get_transparency() != StandardMaterial3D::TRANSPARENCY_DISABLED) {
		return false;
	}
	return true;
}

void OccluderInstance3D::_bake_node(Node *p_node, PackedVector3Array &r_vertices, PackedInt32Array &r_indices) {
	MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(p_node);
	if (mi && mi->is_visible_in_tree()) {
		Ref<Mesh> mesh = mi->get_mesh();
		bool valid = true;

		if (mesh.is_null()) {
			valid = false;
		}

		if (valid && !_bake_material_check(mi->get_material_override())) {
			valid = false;
		}

		if ((mi->get_layer_mask() & bake_mask) == 0) {
			valid = false;
		}

		if (valid) {
			Transform3D global_to_local = get_global_transform().affine_inverse() * mi->get_global_transform();

			for (int i = 0; i < mesh->get_surface_count(); i++) {
				if (mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
					continue;
				}

				if (mi->get_surface_override_material(i).is_valid()) {
					if (!_bake_material_check(mi->get_surface_override_material(i))) {
						continue;
					}
				} else {
					if (!_bake_material_check(mesh->surface_get_material(i))) {
						continue;
					}
				}

				Array arrays = mesh->surface_get_arrays(i);

				int vertex_offset = r_vertices.size();
				PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
				r_vertices.resize(r_vertices.size() + vertices.size());

				Vector3 *vtx_ptr = r_vertices.ptrw();
				for (int j = 0; j < vertices.size(); j++) {
					vtx_ptr[vertex_offset + j] = global_to_local.xform(vertices[j]);
				}

				int index_offset = r_indices.size();
				PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
				r_indices.resize(r_indices.size() + indices.size());

				int *idx_ptr = r_indices.ptrw();
				for (int j = 0; j < indices.size(); j++) {
					idx_ptr[index_offset + j] = vertex_offset + indices[j];
				}
			}
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		if (!child->get_owner()) {
			continue; //maybe a helper
		}

		_bake_node(child, r_vertices, r_indices);
	}
}

OccluderInstance3D::BakeError OccluderInstance3D::bake(Node *p_from_node, String p_occluder_path) {
	if (p_occluder_path == "") {
		if (get_occluder().is_null()) {
			return BAKE_ERROR_NO_SAVE_PATH;
		}
	}

	PackedVector3Array vertices;
	PackedInt32Array indices;

	_bake_node(p_from_node, vertices, indices);

	if (vertices.is_empty() || indices.is_empty()) {
		return BAKE_ERROR_NO_MESHES;
	}

	Ref<Occluder3D> occ;
	if (get_occluder().is_valid()) {
		occ = get_occluder();
	} else {
		occ.instantiate();
		occ->set_path(p_occluder_path);
	}

	occ->set_vertices(vertices);
	occ->set_indices(indices);
	set_occluder(occ);

	return BAKE_ERROR_OK;
}

TypedArray<String> OccluderInstance3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (!bool(GLOBAL_GET("rendering/occlusion_culling/use_occlusion_culling"))) {
		warnings.push_back(TTR("Occlusion culling is disabled in the Project Settings, which means occlusion culling won't be performed in the root viewport.\nTo resolve this, open the Project Settings and enable Rendering > Occlusion Culling > Use Occlusion Culling."));
	}

	if (bake_mask == 0) {
		warnings.push_back(TTR("The Bake Mask has no bits enabled, which means baking will not produce any occluder meshes for this OccluderInstance3D.\nTo resolve this, enable at least one bit in the Bake Mask property."));
	}

	if (occluder.is_null()) {
		warnings.push_back(TTR("No occluder mesh is defined in the Occluder property, so no occlusion culling will be performed using this OccluderInstance3D.\nTo resolve this, select the OccluderInstance3D then use the Bake Occluders button at the top of the 3D editor viewport."));
	} else if (occluder->get_vertices().size() < 3) {
		// Using the "New Occluder" dropdown button won't result in a correct occluder,
		// so warn the user about this.
		warnings.push_back(TTR("The occluder mesh has less than 3 vertices, so no occlusion culling will be performed using this OccluderInstance3D.\nTo generate a proper occluder mesh, select the OccluderInstance3D then use the Bake Occluders button at the top of the 3D editor viewport."));
	}

	return warnings;
}

void OccluderInstance3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bake_mask", "mask"), &OccluderInstance3D::set_bake_mask);
	ClassDB::bind_method(D_METHOD("get_bake_mask"), &OccluderInstance3D::get_bake_mask);
	ClassDB::bind_method(D_METHOD("set_bake_mask_value", "layer_number", "value"), &OccluderInstance3D::set_bake_mask_value);
	ClassDB::bind_method(D_METHOD("get_bake_mask_value", "layer_number"), &OccluderInstance3D::get_bake_mask_value);

	ClassDB::bind_method(D_METHOD("set_occluder", "occluder"), &OccluderInstance3D::set_occluder);
	ClassDB::bind_method(D_METHOD("get_occluder"), &OccluderInstance3D::get_occluder);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "occluder", PROPERTY_HINT_RESOURCE_TYPE, "Occluder3D"), "set_occluder", "get_occluder");
	ADD_GROUP("Bake", "bake_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bake_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_bake_mask", "get_bake_mask");
}

OccluderInstance3D::OccluderInstance3D() {
}

OccluderInstance3D::~OccluderInstance3D() {
}
