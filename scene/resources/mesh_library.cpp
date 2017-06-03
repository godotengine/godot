/*************************************************************************/
/*  mesh_library.cpp                                                     */
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
#include "mesh_library.h"

bool MeshLibrary::_set(const StringName &p_name, const Variant &p_value) {

	String name = p_name;
	if (name.begins_with("item/")) {

		int idx = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		if (!item_map.has(idx))
			create_item(idx);

		if (what == "name")
			set_item_name(idx, p_value);
		else if (what == "mesh")
			set_item_mesh(idx, p_value);
		else if (what == "shape")
			set_item_shape(idx, p_value);
		else if (what == "preview")
			set_item_preview(idx, p_value);
		else if (what == "navmesh")
			set_item_navmesh(idx, p_value);
		else
			return false;

		return true;
	}

	return false;
}

bool MeshLibrary::_get(const StringName &p_name, Variant &r_ret) const {

	String name = p_name;
	int idx = name.get_slicec('/', 1).to_int();
	ERR_FAIL_COND_V(!item_map.has(idx), false);
	String what = name.get_slicec('/', 2);

	if (what == "name")
		r_ret = get_item_name(idx);
	else if (what == "mesh")
		r_ret = get_item_mesh(idx);
	else if (what == "shape")
		r_ret = get_item_shape(idx);
	else if (what == "navmesh")
		r_ret = get_item_navmesh(idx);
	else if (what == "preview")
		r_ret = get_item_preview(idx);
	else
		return false;

	return true;
}

void MeshLibrary::_get_property_list(List<PropertyInfo> *p_list) const {

	for (Map<int, Item>::Element *E = item_map.front(); E; E = E->next()) {

		String name = "item/" + itos(E->key()) + "/";
		p_list->push_back(PropertyInfo(Variant::STRING, name + "name"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, name + "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, name + "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, name + "navmesh", PROPERTY_HINT_RESOURCE_TYPE, "NavigationMesh"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, name + "preview", PROPERTY_HINT_RESOURCE_TYPE, "Texture", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_HELPER));
	}
}

void MeshLibrary::create_item(int p_item) {

	ERR_FAIL_COND(p_item < 0);
	ERR_FAIL_COND(item_map.has(p_item));
	item_map[p_item] = Item();
	_change_notify();
}

void MeshLibrary::set_item_name(int p_item, const String &p_name) {

	ERR_FAIL_COND(!item_map.has(p_item));
	item_map[p_item].name = p_name;
	emit_changed();
	_change_notify();
}
void MeshLibrary::set_item_mesh(int p_item, const Ref<Mesh> &p_mesh) {

	ERR_FAIL_COND(!item_map.has(p_item));
	item_map[p_item].mesh = p_mesh;
	notify_change_to_owners();
	emit_changed();
	_change_notify();
}

void MeshLibrary::set_item_shape(int p_item, const Ref<Shape> &p_shape) {

	ERR_FAIL_COND(!item_map.has(p_item));
	item_map[p_item].shape = p_shape;
	_change_notify();
	notify_change_to_owners();
	emit_changed();
	_change_notify();
}

void MeshLibrary::set_item_navmesh(int p_item, const Ref<NavigationMesh> &p_navmesh) {

	ERR_FAIL_COND(!item_map.has(p_item));
	item_map[p_item].navmesh = p_navmesh;
	_change_notify();
	notify_change_to_owners();
	emit_changed();
	_change_notify();
}

void MeshLibrary::set_item_preview(int p_item, const Ref<Texture> &p_preview) {

	ERR_FAIL_COND(!item_map.has(p_item));
	item_map[p_item].preview = p_preview;
	emit_changed();
	_change_notify();
}
String MeshLibrary::get_item_name(int p_item) const {

	ERR_FAIL_COND_V(!item_map.has(p_item), "");
	return item_map[p_item].name;
}
Ref<Mesh> MeshLibrary::get_item_mesh(int p_item) const {

	ERR_FAIL_COND_V(!item_map.has(p_item), Ref<Mesh>());
	return item_map[p_item].mesh;
}

Ref<Shape> MeshLibrary::get_item_shape(int p_item) const {

	ERR_FAIL_COND_V(!item_map.has(p_item), Ref<Shape>());
	return item_map[p_item].shape;
}

Ref<NavigationMesh> MeshLibrary::get_item_navmesh(int p_item) const {

	ERR_FAIL_COND_V(!item_map.has(p_item), Ref<NavigationMesh>());
	return item_map[p_item].navmesh;
}

Ref<Texture> MeshLibrary::get_item_preview(int p_item) const {

	ERR_FAIL_COND_V(!item_map.has(p_item), Ref<Texture>());
	return item_map[p_item].preview;
}

bool MeshLibrary::has_item(int p_item) const {

	return item_map.has(p_item);
}
void MeshLibrary::remove_item(int p_item) {

	ERR_FAIL_COND(!item_map.has(p_item));
	item_map.erase(p_item);
	notify_change_to_owners();
	_change_notify();
	emit_changed();
}

void MeshLibrary::clear() {

	item_map.clear();
	notify_change_to_owners();
	_change_notify();
	emit_changed();
}

Vector<int> MeshLibrary::get_item_list() const {

	Vector<int> ret;
	ret.resize(item_map.size());
	int idx = 0;
	for (Map<int, Item>::Element *E = item_map.front(); E; E = E->next()) {

		ret[idx++] = E->key();
	}

	return ret;
}

int MeshLibrary::find_item_name(const String &p_name) const {

	for (Map<int, Item>::Element *E = item_map.front(); E; E = E->next()) {

		if (E->get().name == p_name)
			return E->key();
	}
	return -1;
}

int MeshLibrary::get_last_unused_item_id() const {

	if (!item_map.size())
		return 0;
	else
		return item_map.back()->key() + 1;
}

void MeshLibrary::_bind_methods() {

	ClassDB::bind_method(D_METHOD("create_item", "id"), &MeshLibrary::create_item);
	ClassDB::bind_method(D_METHOD("set_item_name", "id", "name"), &MeshLibrary::set_item_name);
	ClassDB::bind_method(D_METHOD("set_item_mesh", "id", "mesh:Mesh"), &MeshLibrary::set_item_mesh);
	ClassDB::bind_method(D_METHOD("set_item_navmesh", "id", "navmesh:NavigationMesh"), &MeshLibrary::set_item_navmesh);
	ClassDB::bind_method(D_METHOD("set_item_shape", "id", "shape:Shape"), &MeshLibrary::set_item_shape);
	ClassDB::bind_method(D_METHOD("get_item_name", "id"), &MeshLibrary::get_item_name);
	ClassDB::bind_method(D_METHOD("get_item_mesh:Mesh", "id"), &MeshLibrary::get_item_mesh);
	ClassDB::bind_method(D_METHOD("get_item_navmesh:NavigationMesh", "id"), &MeshLibrary::get_item_navmesh);
	ClassDB::bind_method(D_METHOD("get_item_shape:Shape", "id"), &MeshLibrary::get_item_shape);
	ClassDB::bind_method(D_METHOD("remove_item", "id"), &MeshLibrary::remove_item);
	ClassDB::bind_method(D_METHOD("clear"), &MeshLibrary::clear);
	ClassDB::bind_method(D_METHOD("get_item_list"), &MeshLibrary::get_item_list);
	ClassDB::bind_method(D_METHOD("get_last_unused_item_id"), &MeshLibrary::get_last_unused_item_id);
}

MeshLibrary::MeshLibrary() {
}
MeshLibrary::~MeshLibrary() {
}
