/**************************************************************************/
/*  gltf_document_extension_physics.cpp                                   */
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

#include "gltf_document_extension_physics.h"

#include "scene/3d/area.h"

// Import process.
Error GLTFDocumentExtensionPhysics::import_preflight(Ref<GLTFState> p_state, Vector<String> p_extensions) {
	if (p_extensions.find("OMI_collider") < 0 && p_extensions.find("OMI_physics_body") < 0) {
		return ERR_SKIP;
	}
	Dictionary state_json = p_state->get_json();
	if (state_json.has("extensions")) {
		Dictionary state_extensions = state_json["extensions"];
		if (state_extensions.has("OMI_collider")) {
			Dictionary omi_collider_ext = state_extensions["OMI_collider"];
			if (omi_collider_ext.has("colliders")) {
				Array state_collider_dicts = omi_collider_ext["colliders"];
				if (state_collider_dicts.size() > 0) {
					Array state_colliders;
					for (int i = 0; i < state_collider_dicts.size(); i++) {
						state_colliders.push_back(GLTFCollider::from_dictionary(state_collider_dicts[i]));
					}
					p_state->set_additional_data("GLTFColliders", state_colliders);
				}
			}
		}
	}
	return OK;
}

Vector<String> GLTFDocumentExtensionPhysics::get_supported_extensions() {
	Vector<String> ret;
	ret.push_back("OMI_collider");
	ret.push_back("OMI_physics_body");
	return ret;
}

Error GLTFDocumentExtensionPhysics::parse_node_extensions(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Dictionary &p_extensions) {
	if (p_extensions.has("OMI_collider")) {
		Dictionary node_collider_ext = p_extensions["OMI_collider"];
		if (node_collider_ext.has("collider")) {
			// "collider" is the index of the collider in the state colliders array.
			int node_collider_index = node_collider_ext["collider"];
			Array state_colliders = p_state->get_additional_data("GLTFColliders");
			ERR_FAIL_INDEX_V_MSG(node_collider_index, state_colliders.size(), Error::ERR_FILE_CORRUPT, "GLTF Physics: On node " + p_gltf_node->get_name() + ", the collider index " + itos(node_collider_index) + " is not in the state colliders (size: " + itos(state_colliders.size()) + ").");
			p_gltf_node->set_additional_data("GLTFCollider", state_colliders[node_collider_index]);
		} else {
			p_gltf_node->set_additional_data("GLTFCollider", GLTFCollider::from_dictionary(p_extensions["OMI_collider"]));
		}
	}
	if (p_extensions.has("OMI_physics_body")) {
		p_gltf_node->set_additional_data("GLTFPhysicsBody", GLTFPhysicsBody::from_dictionary(p_extensions["OMI_physics_body"]));
	}
	return OK;
}

void _setup_collider_mesh_resource_from_index_if_needed(Ref<GLTFState> p_state, Ref<GLTFCollider> p_collider) {
	GLTFMeshIndex collider_mesh_index = p_collider->get_mesh_index();
	if (collider_mesh_index == -1) {
		return; // No mesh for this collider.
	}
	Ref<ArrayMesh> array_mesh = p_collider->get_array_mesh();
	if (array_mesh.is_valid()) {
		return; // The mesh resource is already set up.
	}
	Array state_meshes = p_state->get_meshes();
	ERR_FAIL_INDEX_MSG(collider_mesh_index, state_meshes.size(), "GLTF Physics: When importing '" + p_state->get_scene_name() + "', the collider mesh index " + itos(collider_mesh_index) + " is not in the state meshes (size: " + itos(state_meshes.size()) + ").");
	Ref<GLTFMesh> gltf_mesh = state_meshes[collider_mesh_index];
	ERR_FAIL_COND(gltf_mesh.is_null());
	array_mesh = gltf_mesh->get_mesh();
	ERR_FAIL_COND(array_mesh.is_null());
	p_collider->set_array_mesh(array_mesh);
}

CollisionObject *_generate_collision_with_body(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Ref<GLTFCollider> p_collider, Ref<GLTFPhysicsBody> p_physics_body) {
	print_verbose("glTF: Creating collision for: " + p_gltf_node->get_name());
	bool is_trigger = p_collider->get_is_trigger();
	// This method is used for the case where we must generate a parent body.
	// This is can happen for multiple reasons. One possibility is that this
	// GLTF file is using OMI_collider but not OMI_physics_body, or at least
	// this particular node is not using it. Another possibility is that the
	// physics body information is set up on the same GLTF node, not a parent.
	CollisionObject *body;
	if (p_physics_body.is_valid()) {
		// This code is run when the physics body is on the same GLTF node.
		body = p_physics_body->to_node();
		if (is_trigger != (p_physics_body->get_body_type() == "trigger")) {
			// Edge case: If the body's trigger and the collider's trigger
			// are in disagreement, we need to create another new body.
			CollisionObject *child = _generate_collision_with_body(p_state, p_gltf_node, p_collider, nullptr);
			child->set_name(p_gltf_node->get_name() + (is_trigger ? String("Trigger") : String("Solid")));
			body->add_child(child);
			return body;
		}
	} else if (is_trigger) {
		body = memnew(Area);
	} else {
		body = memnew(StaticBody);
	}
	CollisionShape *shape = p_collider->to_node();
	shape->set_name(p_gltf_node->get_name() + "Shape");
	body->add_child(shape);
	return body;
}

Spatial *GLTFDocumentExtensionPhysics::generate_scene_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Node *p_scene_parent) {
	Ref<GLTFPhysicsBody> physics_body = p_gltf_node->get_additional_data("GLTFPhysicsBody");
	Ref<GLTFCollider> collider = p_gltf_node->get_additional_data("GLTFCollider");
	if (collider.is_valid()) {
		_setup_collider_mesh_resource_from_index_if_needed(p_state, collider);
		// If the collider has the correct type of parent, we just return one node.
		if (collider->get_is_trigger()) {
			if (Object::cast_to<Area>(p_scene_parent)) {
				return collider->to_node(true);
			}
		} else {
			if (Object::cast_to<PhysicsBody>(p_scene_parent)) {
				return collider->to_node(true);
			}
		}
		return _generate_collision_with_body(p_state, p_gltf_node, collider, physics_body);
	}
	if (physics_body.is_valid()) {
		return physics_body->to_node();
	}
	return nullptr;
}

// Export process.
bool _are_all_faces_equal(const PoolVector<Face3> &p_a, const PoolVector<Face3> &p_b) {
	if (p_a.size() != p_b.size()) {
		return false;
	}
	for (int i = 0; i < p_a.size(); i++) {
		Face3 a_face = p_a[i];
		Face3 b_face = p_b[i];
		const Vector3 *a_vertices = a_face.vertex;
		const Vector3 *b_vertices = b_face.vertex;
		for (int j = 0; j < 3; j++) {
			if (!a_vertices[j].is_equal_approx(b_vertices[j])) {
				return false;
			}
		}
	}
	return true;
}

GLTFMeshIndex _get_or_insert_mesh_in_state(Ref<GLTFState> p_state, Ref<ArrayMesh> p_mesh) {
	ERR_FAIL_COND_V(p_mesh.is_null(), -1);
	Array state_meshes = p_state->get_meshes();
	PoolVector<Face3> mesh_faces = p_mesh->get_faces();
	// De-duplication: If the state already has the mesh we need, use that one.
	for (GLTFMeshIndex i = 0; i < state_meshes.size(); i++) {
		Ref<GLTFMesh> state_gltf_mesh = state_meshes[i];
		ERR_CONTINUE(state_gltf_mesh.is_null());
		Ref<ArrayMesh> state_array_mesh = state_gltf_mesh->get_mesh();
		ERR_CONTINUE(state_array_mesh.is_null());
		if (state_array_mesh == p_mesh) {
			return i;
		}
		if (_are_all_faces_equal(state_array_mesh->get_faces(), mesh_faces)) {
			return i;
		}
	}
	// After the loop, we have checked that the mesh is not equal to any of the
	// meshes in the state. So we insert a new mesh into the state mesh array.
	Ref<GLTFMesh> gltf_mesh;
	gltf_mesh.instance();
	gltf_mesh->set_mesh(p_mesh);
	GLTFMeshIndex mesh_index = state_meshes.size();
	state_meshes.push_back(gltf_mesh);
	p_state->set_meshes(state_meshes);
	return mesh_index;
}

void GLTFDocumentExtensionPhysics::convert_scene_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Node *p_scene_node) {
	if (cast_to<CollisionShape>(p_scene_node)) {
		CollisionShape *shape = Object::cast_to<CollisionShape>(p_scene_node);
		Ref<GLTFCollider> collider = GLTFCollider::from_node(shape);
		{
			Ref<ArrayMesh> array_mesh = collider->get_array_mesh();
			if (array_mesh.is_valid()) {
				collider->set_mesh_index(_get_or_insert_mesh_in_state(p_state, array_mesh));
			}
		}
		p_gltf_node->set_additional_data("GLTFCollider", collider);
	} else if (cast_to<CollisionObject>(p_scene_node)) {
		CollisionObject *body = Object::cast_to<CollisionObject>(p_scene_node);
		p_gltf_node->set_additional_data("GLTFPhysicsBody", GLTFPhysicsBody::from_node(body));
	}
}

Array _get_or_create_state_colliders_in_state(Ref<GLTFState> p_state) {
	Dictionary state_json = p_state->get_json();
	Dictionary state_extensions;
	if (state_json.has("extensions")) {
		state_extensions = state_json["extensions"];
	} else {
		state_json["extensions"] = state_extensions;
	}
	Dictionary omi_collider_ext;
	if (state_extensions.has("OMI_collider")) {
		omi_collider_ext = state_extensions["OMI_collider"];
	} else {
		state_extensions["OMI_collider"] = omi_collider_ext;
		p_state->add_used_extension("OMI_collider");
	}
	Array state_colliders;
	if (omi_collider_ext.has("colliders")) {
		state_colliders = omi_collider_ext["colliders"];
	} else {
		omi_collider_ext["colliders"] = state_colliders;
	}
	return state_colliders;
}

Error GLTFDocumentExtensionPhysics::export_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Dictionary &r_node_json, Node *p_node) {
	Dictionary node_extensions = r_node_json["extensions"];
	Ref<GLTFPhysicsBody> physics_body = p_gltf_node->get_additional_data("GLTFPhysicsBody");
	if (physics_body.is_valid()) {
		node_extensions["OMI_physics_body"] = physics_body->to_dictionary();
		p_state->add_used_extension("OMI_physics_body");
	}
	Ref<GLTFCollider> collider = p_gltf_node->get_additional_data("GLTFCollider");
	if (collider.is_valid()) {
		Array state_colliders = _get_or_create_state_colliders_in_state(p_state);
		int size = state_colliders.size();
		Dictionary omi_collider_ext;
		node_extensions["OMI_collider"] = omi_collider_ext;
		Dictionary collider_dict = collider->to_dictionary();
		for (int i = 0; i < size; i++) {
			Dictionary other = state_colliders[i];
			if (other == collider_dict) {
				// De-duplication: If we already have an identical collider,
				// set the collider index to the existing one and return.
				omi_collider_ext["collider"] = i;
				return OK;
			}
		}
		// If we don't have an identical collider, add it to the array.
		state_colliders.push_back(collider_dict);
		omi_collider_ext["collider"] = size;
	}
	return OK;
}
