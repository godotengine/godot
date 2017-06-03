/*************************************************************************/
/*  mesh_instance.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "mesh_instance.h"

#include "body_shape.h"
#include "core_string_names.h"
#include "physics_body.h"
#include "scene/scene_string_names.h"
#include "skeleton.h"
bool MeshInstance::_set(const StringName &p_name, const Variant &p_value) {

	//this is not _too_ bad performance wise, really. it only arrives here if the property was not set anywhere else.
	//add to it that it's probably found on first call to _set anyway.

	if (!get_instance().is_valid())
		return false;

	Map<StringName, BlendShapeTrack>::Element *E = blend_shape_tracks.find(p_name);
	if (E) {
		E->get().value = p_value;
		VisualServer::get_singleton()->instance_set_blend_shape_weight(get_instance(), E->get().idx, E->get().value);
		return true;
	}

	if (p_name.operator String().begins_with("material/")) {
		int idx = p_name.operator String().get_slicec('/', 1).to_int();
		if (idx >= materials.size() || idx < 0)
			return false;

		set_surface_material(idx, p_value);
		return true;
	}

	return false;
}

bool MeshInstance::_get(const StringName &p_name, Variant &r_ret) const {

	if (!get_instance().is_valid())
		return false;

	const Map<StringName, BlendShapeTrack>::Element *E = blend_shape_tracks.find(p_name);
	if (E) {
		r_ret = E->get().value;
		return true;
	}

	if (p_name.operator String().begins_with("material/")) {
		int idx = p_name.operator String().get_slicec('/', 1).to_int();
		if (idx >= materials.size() || idx < 0)
			return false;
		r_ret = materials[idx];
		return true;
	}
	return false;
}

void MeshInstance::_get_property_list(List<PropertyInfo> *p_list) const {

	List<String> ls;
	for (const Map<StringName, BlendShapeTrack>::Element *E = blend_shape_tracks.front(); E; E = E->next()) {

		ls.push_back(E->key());
	}

	ls.sort();

	for (List<String>::Element *E = ls.front(); E; E = E->next()) {
		p_list->push_back(PropertyInfo(Variant::REAL, E->get(), PROPERTY_HINT_RANGE, "0,1,0.01"));
	}

	if (mesh.is_valid()) {
		for (int i = 0; i < mesh->get_surface_count(); i++) {
			p_list->push_back(PropertyInfo(Variant::OBJECT, "material/" + itos(i), PROPERTY_HINT_RESOURCE_TYPE, "Material"));
		}
	}
}

void MeshInstance::set_mesh(const Ref<Mesh> &p_mesh) {

	if (mesh == p_mesh)
		return;

	if (mesh.is_valid()) {
		mesh->disconnect(CoreStringNames::get_singleton()->changed, this, SceneStringNames::get_singleton()->_mesh_changed);
		materials.clear();
	}

	mesh = p_mesh;

	blend_shape_tracks.clear();
	if (mesh.is_valid()) {

		for (int i = 0; i < mesh->get_blend_shape_count(); i++) {

			BlendShapeTrack mt;
			mt.idx = i;
			mt.value = 0;
			blend_shape_tracks["blend_shapes/" + String(mesh->get_blend_shape_name(i))] = mt;
		}

		mesh->connect(CoreStringNames::get_singleton()->changed, this, SceneStringNames::get_singleton()->_mesh_changed);
		materials.resize(mesh->get_surface_count());

		set_base(mesh->get_rid());
	} else {

		set_base(RID());
	}

	_change_notify();
}
Ref<Mesh> MeshInstance::get_mesh() const {

	return mesh;
}

void MeshInstance::_resolve_skeleton_path() {

	if (skeleton_path.is_empty())
		return;

	Skeleton *skeleton = get_node(skeleton_path) ? get_node(skeleton_path)->cast_to<Skeleton>() : NULL;
	if (skeleton)
		VisualServer::get_singleton()->instance_attach_skeleton(get_instance(), skeleton->get_skeleton());
}

void MeshInstance::set_skeleton_path(const NodePath &p_skeleton) {

	skeleton_path = p_skeleton;
	if (!is_inside_tree())
		return;
	_resolve_skeleton_path();
}

NodePath MeshInstance::get_skeleton_path() {
	return skeleton_path;
}

Rect3 MeshInstance::get_aabb() const {

	if (!mesh.is_null())
		return mesh->get_aabb();

	return Rect3();
}

PoolVector<Face3> MeshInstance::get_faces(uint32_t p_usage_flags) const {

	if (!(p_usage_flags & (FACES_SOLID | FACES_ENCLOSING)))
		return PoolVector<Face3>();

	if (mesh.is_null())
		return PoolVector<Face3>();

	return mesh->get_faces();
}

Node *MeshInstance::create_trimesh_collision_node() {

	if (mesh.is_null())
		return NULL;

	Ref<Shape> shape = mesh->create_trimesh_shape();
	if (shape.is_null())
		return NULL;

	StaticBody *static_body = memnew(StaticBody);
	static_body->add_shape(shape);
	return static_body;
}

void MeshInstance::create_trimesh_collision() {

	StaticBody *static_body = create_trimesh_collision_node()->cast_to<StaticBody>();
	ERR_FAIL_COND(!static_body);
	static_body->set_name(String(get_name()) + "_col");

	add_child(static_body);
	if (get_owner())
		static_body->set_owner(get_owner());
	CollisionShape *cshape = memnew(CollisionShape);
	cshape->set_shape(static_body->get_shape(0));
	static_body->add_child(cshape);
	if (get_owner())
		cshape->set_owner(get_owner());
}

Node *MeshInstance::create_convex_collision_node() {

	if (mesh.is_null())
		return NULL;

	Ref<Shape> shape = mesh->create_convex_shape();
	if (shape.is_null())
		return NULL;

	StaticBody *static_body = memnew(StaticBody);
	static_body->add_shape(shape);
	return static_body;
}

void MeshInstance::create_convex_collision() {

	StaticBody *static_body = create_convex_collision_node()->cast_to<StaticBody>();
	ERR_FAIL_COND(!static_body);
	static_body->set_name(String(get_name()) + "_col");

	add_child(static_body);
	if (get_owner())
		static_body->set_owner(get_owner());
	CollisionShape *cshape = memnew(CollisionShape);
	cshape->set_shape(static_body->get_shape(0));
	static_body->add_child(cshape);
	if (get_owner())
		cshape->set_owner(get_owner());
}

void MeshInstance::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {
		_resolve_skeleton_path();
	}
}

void MeshInstance::set_surface_material(int p_surface, const Ref<Material> &p_material) {

	ERR_FAIL_INDEX(p_surface, materials.size());

	materials[p_surface] = p_material;

	if (materials[p_surface].is_valid())
		VS::get_singleton()->instance_set_surface_material(get_instance(), p_surface, materials[p_surface]->get_rid());
	else
		VS::get_singleton()->instance_set_surface_material(get_instance(), p_surface, RID());
}

Ref<Material> MeshInstance::get_surface_material(int p_surface) const {

	ERR_FAIL_INDEX_V(p_surface, materials.size(), Ref<Material>());

	return materials[p_surface];
}

void MeshInstance::_mesh_changed() {

	materials.resize(mesh->get_surface_count());
}

void MeshInstance::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_mesh", "mesh:Mesh"), &MeshInstance::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh:Mesh"), &MeshInstance::get_mesh);
	ClassDB::bind_method(D_METHOD("set_skeleton_path", "skeleton_path:NodePath"), &MeshInstance::set_skeleton_path);
	ClassDB::bind_method(D_METHOD("get_skeleton_path:NodePath"), &MeshInstance::get_skeleton_path);

	ClassDB::bind_method(D_METHOD("create_trimesh_collision"), &MeshInstance::create_trimesh_collision);
	ClassDB::set_method_flags("MeshInstance", "create_trimesh_collision", METHOD_FLAGS_DEFAULT);
	ClassDB::bind_method(D_METHOD("create_convex_collision"), &MeshInstance::create_convex_collision);
	ClassDB::set_method_flags("MeshInstance", "create_convex_collision", METHOD_FLAGS_DEFAULT);
	ClassDB::bind_method(D_METHOD("_mesh_changed"), &MeshInstance::_mesh_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "skeleton"), "set_skeleton_path", "get_skeleton_path");
}

MeshInstance::MeshInstance() {
	skeleton_path = NodePath("..");
}

MeshInstance::~MeshInstance() {
}
