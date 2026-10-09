/**************************************************************************/
/*  mesh_library.cpp                                                      */
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

#include "mesh_library.h"

#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "scene/resources/texture.h"
#include "servers/rendering/rendering_server.h" // IWYU pragma: keep // Needed to bind RSE enums.

#ifndef PHYSICS_3D_DISABLED
#include "scene/resources/3d/box_shape_3d.h"
#endif // PHYSICS_3D_DISABLED

bool MeshLibrary::_validate_index(int p_idx) {
	if (unlikely(!item_map.has(p_idx))) {
		if (!init_property) {
			ERR_FAIL_V_MSG(false, vformat("Requested for nonexistent MeshLibrary item '%d'.", p_idx));
		}
		create_item(p_idx);
	}

	return true;
}

bool MeshLibrary::_filter_property(const String &p_name, int p_index) const {
	if (p_name == "preview") {
		return get_item_use_custom_preview(p_index);
	}
	return true;
}

bool MeshLibrary::_set(const StringName &p_name, const Variant &p_value) {
	init_property = true;
	if (property_helper.property_set_value(p_name, p_value)) {
		return true;
	}
	init_property = false;

#ifndef DISABLE_DEPRECATED
	const String sname = p_name;
	if (!sname.begins_with("item/")) {
		return false;
	}

	Vector<String> components = sname.split("/", true, 2);
	if (components.size() < 2 || !components[1].is_valid_int()) {
		return false;
	}

#ifndef NAVIGATION_3D_DISABLED
	int index = components[1].to_int();
	if (components[2] == "navmesh") { // Renamed in 4.0 beta 9.
		set_item_navigation_mesh(index, p_value);
		return true;
	} else if (components[2] == "navmesh_transform") { // Renamed in 4.0 beta 9.
		set_item_navigation_mesh_transform(index, p_value);
		return true;
	}
#endif // NAVIGATION_3D_DISABLED
#endif // DISABLE_DEPRECATED

	return false;
}

bool MeshLibrary::_get(const StringName &p_name, Variant &r_ret) const {
	if (property_helper.property_get_value(p_name, r_ret)) {
		return true;
	}

#ifndef DISABLE_DEPRECATED
	const String sname = p_name;
	if (!sname.begins_with("item/")) {
		return false;
	}

	Vector<String> components = sname.split("/", true, 2);
	if (components.size() < 2 || !components[1].is_valid_int()) {
		return false;
	}

#ifndef NAVIGATION_3D_DISABLED
	int index = components[1].to_int();
	if (components[2] == "navmesh") { // Renamed in 4.0 beta 9.
		r_ret = get_item_navigation_mesh(index);
		return true;
	} else if (components[2] == "navmesh_transform") { // Renamed in 4.0 beta 9.
		r_ret = get_item_navigation_mesh_transform(index);
		return true;
	}
#endif // NAVIGATION_3D_DISABLED
#endif // DISABLE_DEPRECATED

	return false;
}

void MeshLibrary::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const KeyValue<int, Item> &E : item_map) {
		property_helper.add_properties_for_index(E.key, p_list);
	}
}

void MeshLibrary::create_item(int p_item) {
	ERR_FAIL_COND(p_item < 0);
	ERR_FAIL_COND(item_map.has(p_item));
	item_map[p_item] = Item();
	emit_changed();
	notify_property_list_changed();
}

void MeshLibrary::set_item_name(int p_item, const String &p_name) {
	if (_validate_index(p_item)) {
		item_map[p_item].name = p_name;
		emit_changed();
	}
}

void MeshLibrary::set_item_category(int p_item, const StringName &p_category) {
	if (_validate_index(p_item)) {
		item_map[p_item].category = p_category;
		emit_changed();
	}
}

void MeshLibrary::set_item_mesh(int p_item, const Ref<Mesh> &p_mesh) {
	if (!_validate_index(p_item)) {
		return;
	}
	if (item_map[p_item].mesh.is_valid()) {
		item_map[p_item].mesh->disconnect_changed(callable_mp(this, &MeshLibrary::_mesh_changed));
	}
	item_map[p_item].mesh = p_mesh;
	if (p_mesh.is_valid()) {
		p_mesh->connect_changed(callable_mp(this, &MeshLibrary::_mesh_changed).bind(p_item));
	}
	emit_changed();
}

void MeshLibrary::set_item_mesh_transform(int p_item, const Transform3D &p_transform) {
	if (_validate_index(p_item)) {
		item_map[p_item].mesh_transform = p_transform;
		emit_changed();
	}
}

void MeshLibrary::set_item_mesh_cast_shadow(int p_item, RSE::ShadowCastingSetting p_shadow_casting_setting) {
	if (_validate_index(p_item)) {
		item_map[p_item].mesh_cast_shadow = p_shadow_casting_setting;
		emit_changed();
	}
}

#ifndef PHYSICS_3D_DISABLED
void MeshLibrary::set_item_shapes(int p_item, const Vector<ShapeData> &p_shapes) {
	if (_validate_index(p_item)) {
		item_map[p_item].shapes = p_shapes;
		emit_changed();
		notify_property_list_changed();
	}
}
#endif // PHYSICS_3D_DISABLED

#ifndef NAVIGATION_3D_DISABLED
void MeshLibrary::set_item_navigation_mesh(int p_item, const Ref<NavigationMesh> &p_navigation_mesh) {
	if (_validate_index(p_item)) {
		item_map[p_item].navigation_mesh = p_navigation_mesh;
		emit_changed();
	}
}

void MeshLibrary::set_item_navigation_mesh_transform(int p_item, const Transform3D &p_transform) {
	if (_validate_index(p_item)) {
		item_map[p_item].navigation_mesh_transform = p_transform;
		emit_changed();
	}
}

void MeshLibrary::set_item_navigation_layers(int p_item, uint32_t p_navigation_layers) {
	if (_validate_index(p_item)) {
		item_map[p_item].navigation_layers = p_navigation_layers;
		emit_changed();
	}
}
#endif // NAVIGATION_3D_DISABLED

void MeshLibrary::set_item_preview(int p_item, const Ref<Texture2D> &p_preview) {
	if (_validate_index(p_item)) {
		item_map[p_item].preview = p_preview;
		emit_changed();
	}
}

void MeshLibrary::set_item_custom_preview(int p_item, const Ref<Texture2D> &p_preview) {
	if (_validate_index(p_item)) {
		set_item_preview(p_item, p_preview);
		item_map[p_item].use_custom_preview = p_preview.is_valid();
	}
}

void MeshLibrary::set_item_use_custom_preview(int p_item, bool p_use_custom_preview) {
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	print_line(p_use_custom_preview);
	item_map[p_item].use_custom_preview = p_use_custom_preview;
}

String MeshLibrary::get_item_name(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), "", "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].name;
}

StringName MeshLibrary::get_item_category(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), StringName(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].category;
}

Ref<Mesh> MeshLibrary::get_item_mesh(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Ref<Mesh>(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].mesh;
}

Transform3D MeshLibrary::get_item_mesh_transform(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Transform3D(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].mesh_transform;
}

RSE::ShadowCastingSetting MeshLibrary::get_item_mesh_cast_shadow(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), RSE::ShadowCastingSetting::SHADOW_CASTING_SETTING_ON, "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].mesh_cast_shadow;
}

#ifndef PHYSICS_3D_DISABLED
Vector<MeshLibrary::ShapeData> MeshLibrary::get_item_shapes(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Vector<ShapeData>(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].shapes;
}
#endif // PHYSICS_3D_DISABLED

#ifndef NAVIGATION_3D_DISABLED
Ref<NavigationMesh> MeshLibrary::get_item_navigation_mesh(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Ref<NavigationMesh>(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].navigation_mesh;
}

Transform3D MeshLibrary::get_item_navigation_mesh_transform(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Transform3D(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].navigation_mesh_transform;
}

uint32_t MeshLibrary::get_item_navigation_layers(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), 0, "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].navigation_layers;
}
#endif // NAVIGATION_3D_DISABLED

Ref<Texture2D> MeshLibrary::get_item_preview(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Ref<Texture2D>(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].preview;
}

bool MeshLibrary::get_item_use_custom_preview(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), false, "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].use_custom_preview;
}

bool MeshLibrary::has_item(int p_item) const {
	return item_map.has(p_item);
}

void MeshLibrary::remove_item(int p_item) {
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	item_map.erase(p_item);
	notify_property_list_changed();
	emit_changed();
}

void MeshLibrary::clear() {
	item_map.clear();
	notify_property_list_changed();
	emit_changed();
}

Vector<int> MeshLibrary::get_item_list() const {
	Vector<int> ret;
	ret.resize(item_map.size());
	int idx = 0;
	for (const KeyValue<int, Item> &E : item_map) {
		ret.write[idx++] = E.key;
	}

	return ret;
}

int MeshLibrary::find_item_by_name(const String &p_name) const {
	for (const KeyValue<int, Item> &E : item_map) {
		if (E.value.name == p_name) {
			return E.key;
		}
	}
	return -1;
}

int MeshLibrary::get_last_unused_item_id() const {
	if (!item_map.size()) {
		return 0;
	} else {
		return item_map.back()->key() + 1;
	}
}

#ifndef PHYSICS_3D_DISABLED
void MeshLibrary::_set_item_shapes(int p_item, const Array &p_shapes) {
	Array arr_shapes = p_shapes;
	int size = p_shapes.size();
	if (size & 1) {
		ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
		int prev_size = item_map[p_item].shapes.size() * 2;

		if (prev_size < size) {
			// Check if last element is a shape.
			Ref<Shape3D> shape = arr_shapes[size - 1];
			if (shape.is_null()) {
				Ref<BoxShape3D> box_shape;
				box_shape.instantiate();
				arr_shapes[size - 1] = box_shape;
			}

			// Make sure the added element is a Transform3D.
			arr_shapes.push_back(Transform3D());
			size++;
		} else {
			size--;
			arr_shapes.resize(size);
		}
	}

	Vector<ShapeData> shapes;
	for (int i = 0; i < size; i += 2) {
		ShapeData sd;
		sd.shape = arr_shapes[i + 0];
		sd.local_transform = arr_shapes[i + 1];

		if (sd.shape.is_valid()) {
			shapes.push_back(sd);
		}
	}

	set_item_shapes(p_item, shapes);
}

Array MeshLibrary::_get_item_shapes(int p_item) const {
	Vector<ShapeData> shapes = get_item_shapes(p_item);
	Array ret;
	for (int i = 0; i < shapes.size(); i++) {
		ret.push_back(shapes[i].shape);
		ret.push_back(shapes[i].local_transform);
	}

	return ret;
}
#endif // PHYSICS_3D_DISABLED

void MeshLibrary::_mesh_changed(int p_idx) {
	ERR_FAIL_COND(!item_map.has(p_idx));
	if (!item_map[p_idx].use_custom_preview) {
		item_map[p_idx].preview = Ref<Texture2D>();
		callable_mp((Resource *)this, &Resource::emit_changed).call_deferred();
	}
}

void MeshLibrary::reset_state() {
	clear();
}

void MeshLibrary::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_item", "id"), &MeshLibrary::create_item);
	ClassDB::bind_method(D_METHOD("set_item_name", "id", "name"), &MeshLibrary::set_item_name);
	ClassDB::bind_method(D_METHOD("set_item_category", "id", "category"), &MeshLibrary::set_item_category);
	ClassDB::bind_method(D_METHOD("set_item_mesh", "id", "mesh"), &MeshLibrary::set_item_mesh);
	ClassDB::bind_method(D_METHOD("set_item_mesh_transform", "id", "mesh_transform"), &MeshLibrary::set_item_mesh_transform);
	ClassDB::bind_method(D_METHOD("set_item_mesh_cast_shadow", "id", "shadow_casting_setting"), &MeshLibrary::set_item_mesh_cast_shadow);

#ifndef NAVIGATION_3D_DISABLED
	ClassDB::bind_method(D_METHOD("set_item_navigation_mesh", "id", "navigation_mesh"), &MeshLibrary::set_item_navigation_mesh);
	ClassDB::bind_method(D_METHOD("set_item_navigation_mesh_transform", "id", "navigation_mesh"), &MeshLibrary::set_item_navigation_mesh_transform);
	ClassDB::bind_method(D_METHOD("set_item_navigation_layers", "id", "navigation_layers"), &MeshLibrary::set_item_navigation_layers);
#endif // NAVIGATION_3D_DISABLED

#ifndef PHYSICS_3D_DISABLED
	ClassDB::bind_method(D_METHOD("set_item_shapes", "id", "shapes"), &MeshLibrary::_set_item_shapes);
#endif // PHYSICS_3D_DISABLED

	ClassDB::bind_method(D_METHOD("set_item_preview", "id", "texture"), &MeshLibrary::set_item_custom_preview);
	ClassDB::bind_method(D_METHOD("set_item_use_custom_preview", "id", "use_custom_preview"), &MeshLibrary::set_item_use_custom_preview);
	ClassDB::bind_method(D_METHOD("get_item_name", "id"), &MeshLibrary::get_item_name);
	ClassDB::bind_method(D_METHOD("get_item_category", "id"), &MeshLibrary::get_item_category);
	ClassDB::bind_method(D_METHOD("get_item_mesh", "id"), &MeshLibrary::get_item_mesh);
	ClassDB::bind_method(D_METHOD("get_item_mesh_transform", "id"), &MeshLibrary::get_item_mesh_transform);
	ClassDB::bind_method(D_METHOD("get_item_mesh_cast_shadow", "id"), &MeshLibrary::get_item_mesh_cast_shadow);

#ifndef NAVIGATION_3D_DISABLED
	ClassDB::bind_method(D_METHOD("get_item_navigation_mesh", "id"), &MeshLibrary::get_item_navigation_mesh);
	ClassDB::bind_method(D_METHOD("get_item_navigation_mesh_transform", "id"), &MeshLibrary::get_item_navigation_mesh_transform);
	ClassDB::bind_method(D_METHOD("get_item_navigation_layers", "id"), &MeshLibrary::get_item_navigation_layers);
#endif // NAVIGATION_3D_DISABLED

#ifndef PHYSICS_3D_DISABLED
	ClassDB::bind_method(D_METHOD("get_item_shapes", "id"), &MeshLibrary::_get_item_shapes);
#endif // PHYSICS_3D_DISABLED

	ClassDB::bind_method(D_METHOD("get_item_preview", "id"), &MeshLibrary::get_item_preview);
	ClassDB::bind_method(D_METHOD("get_item_use_custom_preview", "id"), &MeshLibrary::get_item_use_custom_preview);
	ClassDB::bind_method(D_METHOD("remove_item", "id"), &MeshLibrary::remove_item);
	ClassDB::bind_method(D_METHOD("find_item_by_name", "name"), &MeshLibrary::find_item_by_name);

	ClassDB::bind_method(D_METHOD("clear"), &MeshLibrary::clear);
	ClassDB::bind_method(D_METHOD("get_item_list"), &MeshLibrary::get_item_list);
	ClassDB::bind_method(D_METHOD("get_item_count"), &MeshLibrary::get_item_count);
	ClassDB::bind_method(D_METHOD("get_last_unused_item_id"), &MeshLibrary::get_last_unused_item_id);

	Item defaults;

	base_property_helper.set_prefix("item/");
	base_property_helper.set_array_length_getter(&MeshLibrary::get_item_count);
	base_property_helper.set_property_filter(&MeshLibrary::_filter_property);
	base_property_helper.register_property(PropertyInfo(Variant::STRING, PNAME("name"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), defaults.name, &MeshLibrary::set_item_name, &MeshLibrary::get_item_name);
	base_property_helper.register_property(PropertyInfo(Variant::STRING, PNAME("category"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), defaults.category, &MeshLibrary::set_item_category, &MeshLibrary::get_item_category);
	base_property_helper.register_property(PropertyInfo(Variant::OBJECT, PNAME("mesh"), PROPERTY_HINT_RESOURCE_TYPE, Mesh::get_class_static(), PROPERTY_USAGE_NO_EDITOR), defaults.mesh, &MeshLibrary::set_item_mesh, &MeshLibrary::get_item_mesh);
	base_property_helper.register_property(PropertyInfo(Variant::TRANSFORM3D, PNAME("mesh_transform"), PROPERTY_HINT_NONE, "suffix:m", PROPERTY_USAGE_NO_EDITOR), defaults.mesh_transform, &MeshLibrary::set_item_mesh_transform, &MeshLibrary::get_item_mesh_transform);
	base_property_helper.register_property(PropertyInfo(Variant::INT, PNAME("mesh_cast_shadow"), PROPERTY_HINT_ENUM, "Off,On,Double-Sided,Shadows Only", PROPERTY_USAGE_NO_EDITOR), defaults.mesh_cast_shadow, &MeshLibrary::set_item_mesh_cast_shadow, &MeshLibrary::get_item_mesh_cast_shadow);

#ifndef PHYSICS_3D_DISABLED
	base_property_helper.register_property(PropertyInfo(Variant::ARRAY, PNAME("shapes"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), Array(), &MeshLibrary::_set_item_shapes, &MeshLibrary::_get_item_shapes);
#endif // PHYSICS_3D_DISABLED

#ifndef NAVIGATION_3D_DISABLED
	base_property_helper.register_property(PropertyInfo(Variant::OBJECT, PNAME("navigation_mesh"), PROPERTY_HINT_RESOURCE_TYPE, NavigationMesh::get_class_static(), PROPERTY_USAGE_NO_EDITOR), defaults.navigation_mesh, &MeshLibrary::set_item_navigation_mesh, &MeshLibrary::get_item_navigation_mesh);
	base_property_helper.register_property(PropertyInfo(Variant::TRANSFORM3D, PNAME("navigation_mesh_transform"), PROPERTY_HINT_NONE, "suffix:m", PROPERTY_USAGE_NO_EDITOR), defaults.navigation_mesh_transform, &MeshLibrary::set_item_mesh_transform, &MeshLibrary::get_item_mesh_transform);
	base_property_helper.register_property(PropertyInfo(Variant::INT, PNAME("navigation_layers"), PROPERTY_HINT_LAYERS_3D_NAVIGATION, "", PROPERTY_USAGE_NO_EDITOR), defaults.navigation_layers, &MeshLibrary::set_item_navigation_layers, &MeshLibrary::get_item_navigation_layers);
#endif // NAVIGATION_3D_DISABLED

	base_property_helper.register_property(PropertyInfo(Variant::OBJECT, PNAME("preview"), PROPERTY_HINT_RESOURCE_TYPE, Texture2D::get_class_static(), PROPERTY_USAGE_NO_EDITOR), defaults.preview, &MeshLibrary::set_item_preview, &MeshLibrary::get_item_preview);
	base_property_helper.register_property(PropertyInfo(Variant::BOOL, PNAME("use_custom_preview"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), defaults.use_custom_preview, &MeshLibrary::set_item_use_custom_preview, &MeshLibrary::get_item_use_custom_preview);
	PropertyListHelper::register_base_helper(get_class_static(), &base_property_helper);
}

MeshLibrary::MeshLibrary() {
	property_helper.setup_for_instance(base_property_helper, this);
	property_helper.enable_out_of_bounds_assign();
}
