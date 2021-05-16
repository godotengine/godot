/*************************************************************************/
/*  mesh_instance_3d.cpp                                                 */
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

#include "mesh_instance_3d.h"

#include "collision_shape_3d.h"
#include "core/core_string_names.h"
#include "physics_body_3d.h"
#include "scene/resources/material.h"
#include "skeleton_3d.h"

bool MeshInstance3D::_set(const StringName &p_name, const Variant &p_value) {
	//this is not _too_ bad performance wise, really. it only arrives here if the property was not set anywhere else.
	//add to it that it's probably found on first call to _set anyway.

	if (!get_instance().is_valid()) {
		return false;
	}

	Map<StringName, BlendShapeTrack>::Element *E = blend_shape_tracks.find(p_name);
	if (E) {
		E->get().value = p_value;
		RenderingServer::get_singleton()->instance_set_blend_shape_weight(get_instance(), E->get().idx, E->get().value);
		return true;
	}

	if (p_name.operator String().begins_with("surface_material_override/")) {
		int idx = p_name.operator String().get_slicec('/', 1).to_int();
		if (idx >= surface_override_materials.size() || idx < 0) {
			return false;
		}

		set_surface_override_material(idx, p_value);
		return true;
	}

	return false;
}

bool MeshInstance3D::_get(const StringName &p_name, Variant &r_ret) const {
	if (!get_instance().is_valid()) {
		return false;
	}

	const Map<StringName, BlendShapeTrack>::Element *E = blend_shape_tracks.find(p_name);
	if (E) {
		r_ret = E->get().value;
		return true;
	}

	if (p_name.operator String().begins_with("surface_material_override/")) {
		int idx = p_name.operator String().get_slicec('/', 1).to_int();
		if (idx >= surface_override_materials.size() || idx < 0) {
			return false;
		}
		r_ret = surface_override_materials[idx];
		return true;
	}
	return false;
}

void MeshInstance3D::_get_property_list(List<PropertyInfo> *p_list) const {
	List<String> ls;
	for (const Map<StringName, BlendShapeTrack>::Element *E = blend_shape_tracks.front(); E; E = E->next()) {
		ls.push_back(E->key());
	}

	ls.sort();

	for (List<String>::Element *E = ls.front(); E; E = E->next()) {
		p_list->push_back(PropertyInfo(Variant::FLOAT, E->get(), PROPERTY_HINT_RANGE, "-1,1,0.00001"));
	}

	if (mesh.is_valid()) {
		for (int i = 0; i < mesh->get_surface_count(); i++) {
			p_list->push_back(PropertyInfo(Variant::OBJECT, "surface_material_override/" + itos(i), PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,StandardMaterial3D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DEFERRED_SET_RESOURCE));
		}
	}
}

void MeshInstance3D::set_mesh(const Ref<Mesh> &p_mesh) {
	if (mesh == p_mesh) {
		return;
	}

	if (mesh.is_valid()) {
		mesh->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &MeshInstance3D::_mesh_changed));
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

		mesh->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &MeshInstance3D::_mesh_changed));
		surface_override_materials.resize(mesh->get_surface_count());

		set_base(mesh->get_rid());
	} else {
		set_base(RID());
	}

	update_gizmo();

	notify_property_list_changed();
}

Ref<Mesh> MeshInstance3D::get_mesh() const {
	return mesh;
}

void MeshInstance3D::_resolve_skeleton_path() {
	Ref<SkinReference> new_skin_reference;

	if (!skeleton_path.is_empty()) {
		Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(get_node(skeleton_path));
		if (skeleton) {
			new_skin_reference = skeleton->register_skin(skin_internal);
			if (skin_internal.is_null()) {
				//a skin was created for us
				skin_internal = new_skin_reference->get_skin();
				notify_property_list_changed();
			}
		}
	}

	skin_ref = new_skin_reference;

	if (skin_ref.is_valid()) {
		RenderingServer::get_singleton()->instance_attach_skeleton(get_instance(), skin_ref->get_skeleton());
	} else {
		RenderingServer::get_singleton()->instance_attach_skeleton(get_instance(), RID());
	}
}

void MeshInstance3D::set_skin(const Ref<Skin> &p_skin) {
	skin_internal = p_skin;
	skin = p_skin;
	if (!is_inside_tree()) {
		return;
	}
	_resolve_skeleton_path();
}

Ref<Skin> MeshInstance3D::get_skin() const {
	return skin;
}

void MeshInstance3D::set_skeleton_path(const NodePath &p_skeleton) {
	skeleton_path = p_skeleton;
	if (!is_inside_tree()) {
		return;
	}
	_resolve_skeleton_path();
}

NodePath MeshInstance3D::get_skeleton_path() {
	return skeleton_path;
}

AABB MeshInstance3D::get_aabb() const {
	if (!mesh.is_null()) {
		return mesh->get_aabb();
	}

	return AABB();
}

Vector<Face3> MeshInstance3D::get_faces(uint32_t p_usage_flags) const {
	if (!(p_usage_flags & (FACES_SOLID | FACES_ENCLOSING))) {
		return Vector<Face3>();
	}

	if (mesh.is_null()) {
		return Vector<Face3>();
	}

	return mesh->get_faces();
}

Node *MeshInstance3D::create_trimesh_collision_node() {
	if (mesh.is_null()) {
		return nullptr;
	}

	Ref<Shape3D> shape = mesh->create_trimesh_shape();
	if (shape.is_null()) {
		return nullptr;
	}

	StaticBody3D *static_body = memnew(StaticBody3D);
	CollisionShape3D *cshape = memnew(CollisionShape3D);
	cshape->set_shape(shape);
	static_body->add_child(cshape);
	return static_body;
}

void MeshInstance3D::create_trimesh_collision() {
	StaticBody3D *static_body = Object::cast_to<StaticBody3D>(create_trimesh_collision_node());
	ERR_FAIL_COND(!static_body);
	static_body->set_name(String(get_name()) + "_col");

	add_child(static_body);
	if (get_owner()) {
		CollisionShape3D *cshape = Object::cast_to<CollisionShape3D>(static_body->get_child(0));
		static_body->set_owner(get_owner());
		cshape->set_owner(get_owner());
	}
}

Node *MeshInstance3D::create_convex_collision_node() {
	if (mesh.is_null()) {
		return nullptr;
	}

	Ref<Shape3D> shape = mesh->create_convex_shape();
	if (shape.is_null()) {
		return nullptr;
	}

	StaticBody3D *static_body = memnew(StaticBody3D);
	CollisionShape3D *cshape = memnew(CollisionShape3D);
	cshape->set_shape(shape);
	static_body->add_child(cshape);
	return static_body;
}

void MeshInstance3D::create_convex_collision() {
	StaticBody3D *static_body = Object::cast_to<StaticBody3D>(create_convex_collision_node());
	ERR_FAIL_COND(!static_body);
	static_body->set_name(String(get_name()) + "_col");

	add_child(static_body);
	if (get_owner()) {
		CollisionShape3D *cshape = Object::cast_to<CollisionShape3D>(static_body->get_child(0));
		static_body->set_owner(get_owner());
		cshape->set_owner(get_owner());
	}
}

Node *MeshInstance3D::create_multiple_convex_collisions_node() {
	if (mesh.is_null()) {
		return nullptr;
	}

	Vector<Ref<Shape3D>> shapes = mesh->convex_decompose();
	if (!shapes.size()) {
		return nullptr;
	}

	StaticBody3D *static_body = memnew(StaticBody3D);
	for (int i = 0; i < shapes.size(); i++) {
		CollisionShape3D *cshape = memnew(CollisionShape3D);
		cshape->set_shape(shapes[i]);
		static_body->add_child(cshape);
	}
	return static_body;
}

void MeshInstance3D::create_multiple_convex_collisions() {
	StaticBody3D *static_body = Object::cast_to<StaticBody3D>(create_multiple_convex_collisions_node());
	ERR_FAIL_COND(!static_body);
	static_body->set_name(String(get_name()) + "_col");

	add_child(static_body);
	if (get_owner()) {
		static_body->set_owner(get_owner());
		int count = static_body->get_child_count();
		for (int i = 0; i < count; i++) {
			CollisionShape3D *cshape = Object::cast_to<CollisionShape3D>(static_body->get_child(i));
			cshape->set_owner(get_owner());
		}
	}
}

void MeshInstance3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		_resolve_skeleton_path();
	}
}

int MeshInstance3D::get_surface_override_material_count() const {
	return surface_override_materials.size();
}

void MeshInstance3D::set_surface_override_material(int p_surface, const Ref<Material> &p_material) {
	ERR_FAIL_INDEX(p_surface, surface_override_materials.size());

	surface_override_materials.write[p_surface] = p_material;

	if (surface_override_materials[p_surface].is_valid()) {
		RS::get_singleton()->instance_set_surface_override_material(get_instance(), p_surface, surface_override_materials[p_surface]->get_rid());
	} else {
		RS::get_singleton()->instance_set_surface_override_material(get_instance(), p_surface, RID());
	}
}

Ref<Material> MeshInstance3D::get_surface_override_material(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surface_override_materials.size(), Ref<Material>());

	return surface_override_materials[p_surface];
}

Ref<Material> MeshInstance3D::get_active_material(int p_surface) const {
	Ref<Material> material_override = get_material_override();
	if (material_override.is_valid()) {
		return material_override;
	}

	Ref<Material> surface_material = get_surface_override_material(p_surface);
	if (surface_material.is_valid()) {
		return surface_material;
	}

	Ref<Mesh> mesh = get_mesh();
	if (mesh.is_valid()) {
		return mesh->surface_get_material(p_surface);
	}

	return Ref<Material>();
}

void MeshInstance3D::_mesh_changed() {
	ERR_FAIL_COND(mesh.is_null());
	surface_override_materials.resize(mesh->get_surface_count());
	update_gizmo();
}

void MeshInstance3D::create_debug_tangents() {
	Vector<Vector3> lines;
	Vector<Color> colors;

	Ref<Mesh> mesh = get_mesh();
	if (!mesh.is_valid()) {
		return;
	}

	for (int i = 0; i < mesh->get_surface_count(); i++) {
		Array arrays = mesh->surface_get_arrays(i);
		Vector<Vector3> verts = arrays[Mesh::ARRAY_VERTEX];
		Vector<Vector3> norms = arrays[Mesh::ARRAY_NORMAL];
		if (norms.size() == 0) {
			continue;
		}
		Vector<float> tangents = arrays[Mesh::ARRAY_TANGENT];
		if (tangents.size() == 0) {
			continue;
		}

		for (int j = 0; j < verts.size(); j++) {
			Vector3 v = verts[j];
			Vector3 n = norms[j];
			Vector3 t = Vector3(tangents[j * 4 + 0], tangents[j * 4 + 1], tangents[j * 4 + 2]);
			Vector3 b = (n.cross(t)).normalized() * tangents[j * 4 + 3];

			lines.push_back(v); //normal
			colors.push_back(Color(0, 0, 1)); //color
			lines.push_back(v + n * 0.04); //normal
			colors.push_back(Color(0, 0, 1)); //color

			lines.push_back(v); //tangent
			colors.push_back(Color(1, 0, 0)); //color
			lines.push_back(v + t * 0.04); //tangent
			colors.push_back(Color(1, 0, 0)); //color

			lines.push_back(v); //binormal
			colors.push_back(Color(0, 1, 0)); //color
			lines.push_back(v + b * 0.04); //binormal
			colors.push_back(Color(0, 1, 0)); //color
		}
	}

	if (lines.size()) {
		Ref<StandardMaterial3D> sm;
		sm.instance();

		sm->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
		sm->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
		sm->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);

		Ref<ArrayMesh> am;
		am.instance();
		Array a;
		a.resize(Mesh::ARRAY_MAX);
		a[Mesh::ARRAY_VERTEX] = lines;
		a[Mesh::ARRAY_COLOR] = colors;

		am->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, a);
		am->surface_set_material(0, sm);

		MeshInstance3D *mi = memnew(MeshInstance3D);
		mi->set_mesh(am);
		mi->set_name("DebugTangents");
		add_child(mi);
#ifdef TOOLS_ENABLED

		if (is_inside_tree() && this == get_tree()->get_edited_scene_root()) {
			mi->set_owner(this);
		} else {
			mi->set_owner(get_owner());
		}
#endif
	}
}

void MeshInstance3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &MeshInstance3D::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &MeshInstance3D::get_mesh);
	ClassDB::bind_method(D_METHOD("set_skeleton_path", "skeleton_path"), &MeshInstance3D::set_skeleton_path);
	ClassDB::bind_method(D_METHOD("get_skeleton_path"), &MeshInstance3D::get_skeleton_path);
	ClassDB::bind_method(D_METHOD("set_skin", "skin"), &MeshInstance3D::set_skin);
	ClassDB::bind_method(D_METHOD("get_skin"), &MeshInstance3D::get_skin);

	ClassDB::bind_method(D_METHOD("get_surface_override_material_count"), &MeshInstance3D::get_surface_override_material_count);
	ClassDB::bind_method(D_METHOD("set_surface_override_material", "surface", "material"), &MeshInstance3D::set_surface_override_material);
	ClassDB::bind_method(D_METHOD("get_surface_override_material", "surface"), &MeshInstance3D::get_surface_override_material);
	ClassDB::bind_method(D_METHOD("get_active_material", "surface"), &MeshInstance3D::get_active_material);

	ClassDB::bind_method(D_METHOD("create_trimesh_collision"), &MeshInstance3D::create_trimesh_collision);
	ClassDB::set_method_flags("MeshInstance3D", "create_trimesh_collision", METHOD_FLAGS_DEFAULT);
	ClassDB::bind_method(D_METHOD("create_convex_collision"), &MeshInstance3D::create_convex_collision);
	ClassDB::set_method_flags("MeshInstance3D", "create_convex_collision", METHOD_FLAGS_DEFAULT);
	ClassDB::bind_method(D_METHOD("create_multiple_convex_collisions"), &MeshInstance3D::create_multiple_convex_collisions);
	ClassDB::set_method_flags("MeshInstance3D", "create_multiple_convex_collisions", METHOD_FLAGS_DEFAULT);

	ClassDB::bind_method(D_METHOD("create_debug_tangents"), &MeshInstance3D::create_debug_tangents);
	ClassDB::set_method_flags("MeshInstance3D", "create_debug_tangents", METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_GROUP("Skeleton", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "skin", PROPERTY_HINT_RESOURCE_TYPE, "Skin"), "set_skin", "get_skin");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "skeleton", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"), "set_skeleton_path", "get_skeleton_path");
	ADD_GROUP("", "");
}

MeshInstance3D::MeshInstance3D() {
}

MeshInstance3D::~MeshInstance3D() {
}
