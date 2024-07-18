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

#include "box_shape_3d.h"

bool MeshLibrary::_set(const StringName &p_name, const Variant &p_value) {
	String prop_name = p_name;
	if (prop_name.begins_with("item/")) {
		int idx = prop_name.get_slicec('/', 1).to_int();
		String what = prop_name.get_slicec('/', 2);
		if (!item_map.has(idx)) {
			create_item(idx);
		}

		if (what == "name") {
			set_item_name(idx, p_value);
		} else if (what == "mesh") {
			set_item_mesh(idx, p_value);
		} else if (what == "mesh_transform") {
			set_item_mesh_transform(idx, p_value);
		} else if (what == "shape") {
			Vector<ShapeData> shapes;
			ShapeData sd;
			sd.shape = p_value;
			shapes.push_back(sd);
			set_item_shapes(idx, shapes);
		} else if (what == "shapes") {
			_set_item_shapes(idx, p_value);
		} else if (what == "preview") {
			set_item_preview(idx, p_value);
		} else if (what == "navigation_mesh") {
			set_item_navigation_mesh(idx, p_value);
		} else if (what == "navigation_mesh_transform") {
			set_item_navigation_mesh_transform(idx, p_value);
#ifndef DISABLE_DEPRECATED
		} else if (what == "navmesh") { // Renamed in 4.0 beta 9.
			set_item_navigation_mesh(idx, p_value);
		} else if (what == "navmesh_transform") { // Renamed in 4.0 beta 9.
			set_item_navigation_mesh_transform(idx, p_value);
#endif // DISABLE_DEPRECATED
		} else if (what == "navigation_layers") {
			set_item_navigation_layers(idx, p_value);
		} else if (what == "custom_data") {
			int layer_index = p_value.get("layer_id");
			ERR_FAIL_COND_V(layer_index < 0, false);

			if (layer_index >= item_map[idx].custom_data.size()) {
				item_map[idx].custom_data.resize(layer_index + 1);
			}

			set_custom_data_by_layer_id(idx, p_value.get("layer_id"), p_value.get("value"));
		} else {
			return false;
		}

		return true;
	}

	return false;
}

bool MeshLibrary::_get(const StringName &p_name, Variant &r_ret) const {
	String prop_name = p_name;
	int idx = prop_name.get_slicec('/', 1).to_int();
	ERR_FAIL_COND_V(!item_map.has(idx), false);
	String what = prop_name.get_slicec('/', 2);

	if (what == "name") {
		r_ret = get_item_name(idx);
	} else if (what == "mesh") {
		r_ret = get_item_mesh(idx);
	} else if (what == "mesh_transform") {
		r_ret = get_item_mesh_transform(idx);
	} else if (what == "shapes") {
		r_ret = _get_item_shapes(idx);
	} else if (what == "navigation_mesh") {
		r_ret = get_item_navigation_mesh(idx);
	} else if (what == "navigation_mesh_transform") {
		r_ret = get_item_navigation_mesh_transform(idx);
#ifndef DISABLE_DEPRECATED
	} else if (what == "navmesh") { // Renamed in 4.0 beta 9.
		r_ret = get_item_navigation_mesh(idx);
	} else if (what == "navmesh_transform") { // Renamed in 4.0 beta 9.
		r_ret = get_item_navigation_mesh_transform(idx);
#endif // DISABLE_DEPRECATED
	} else if (what == "navigation_layers") {
		r_ret = get_item_navigation_layers(idx);
	} else if (what == "preview") {
		r_ret = get_item_preview(idx);
	} else if (what == "custom_data") {
		r_ret = get_custom_data_by_layer_id(idx, prop_name.get_slicec('/', 3).to_int());
	} else {
		return false;
	}

	return true;
}

void MeshLibrary::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const KeyValue<int, Item> &E : item_map) {
		String prop_name = vformat("%s/%d/", PNAME("item"), E.key);
		p_list->push_back(PropertyInfo(Variant::STRING, prop_name + PNAME("name")));
		p_list->push_back(PropertyInfo(Variant::OBJECT, prop_name + PNAME("mesh"), PROPERTY_HINT_RESOURCE_TYPE, "Mesh"));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM3D, prop_name + PNAME("mesh_transform"), PROPERTY_HINT_NONE, "suffix:m"));
		p_list->push_back(PropertyInfo(Variant::ARRAY, prop_name + PNAME("shapes")));
		p_list->push_back(PropertyInfo(Variant::OBJECT, prop_name + PNAME("navigation_mesh"), PROPERTY_HINT_RESOURCE_TYPE, "NavigationMesh"));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM3D, prop_name + PNAME("navigation_mesh_transform"), PROPERTY_HINT_NONE, "suffix:m"));
		p_list->push_back(PropertyInfo(Variant::INT, prop_name + PNAME("navigation_layers"), PROPERTY_HINT_LAYERS_3D_NAVIGATION));
		p_list->push_back(PropertyInfo(Variant::OBJECT, prop_name + PNAME("preview"), PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::NIL, prop_name + PNAME("custom_data"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));

		String argt = "Any";
		for (int i = 1; i < Variant::VARIANT_MAX; i++) {
			argt += "," + Variant::get_type_name(Variant::Type(i));
		}
		for (int i = 0; i < custom_data_layers.size(); i++) {
			p_list->push_back(PropertyInfo(Variant::STRING, vformat("custom_data_layer_%d/name", i)));
			p_list->push_back(PropertyInfo(Variant::INT, vformat("custom_data_layer_%d/type", i), PROPERTY_HINT_ENUM, argt));
		}
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
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	item_map[p_item].name = p_name;
	emit_changed();
}

void MeshLibrary::set_item_mesh(int p_item, const Ref<Mesh> &p_mesh) {
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	item_map[p_item].mesh = p_mesh;
	emit_changed();
}

void MeshLibrary::set_item_mesh_transform(int p_item, const Transform3D &p_transform) {
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	item_map[p_item].mesh_transform = p_transform;
	emit_changed();
}

void MeshLibrary::set_item_shapes(int p_item, const Vector<ShapeData> &p_shapes) {
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	item_map[p_item].shapes = p_shapes;
	emit_changed();
	notify_property_list_changed();
}

void MeshLibrary::set_item_navigation_mesh(int p_item, const Ref<NavigationMesh> &p_navigation_mesh) {
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	item_map[p_item].navigation_mesh = p_navigation_mesh;
	emit_changed();
}

void MeshLibrary::set_item_navigation_mesh_transform(int p_item, const Transform3D &p_transform) {
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	item_map[p_item].navigation_mesh_transform = p_transform;
	emit_changed();
}

void MeshLibrary::set_item_navigation_layers(int p_item, uint32_t p_navigation_layers) {
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	item_map[p_item].navigation_layers = p_navigation_layers;
	emit_changed();
}

void MeshLibrary::set_item_preview(int p_item, const Ref<Texture2D> &p_preview) {
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	item_map[p_item].preview = p_preview;
	emit_changed();
}

String MeshLibrary::get_item_name(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), "", "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].name;
}

Ref<Mesh> MeshLibrary::get_item_mesh(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Ref<Mesh>(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].mesh;
}

Transform3D MeshLibrary::get_item_mesh_transform(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Transform3D(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].mesh_transform;
}

Vector<MeshLibrary::ShapeData> MeshLibrary::get_item_shapes(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Vector<ShapeData>(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].shapes;
}

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

Ref<Texture2D> MeshLibrary::get_item_preview(int p_item) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Ref<Texture2D>(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	return item_map[p_item].preview;
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

void MeshLibrary::notify_mesh_library_properties_should_change() {
	// Convert custom data to the new type.
	for (int i = 0; i < item_map.size(); i++) {
		item_map[i].custom_data.resize(get_custom_data_layers_count());
		for (int j = 0; j < item_map[i].custom_data.size(); j++) {
			if (item_map[i].custom_data[j].get_type() != get_custom_data_layer_type(j)) {
				Variant new_val;
				Callable::CallError error;
				if (Variant::can_convert(item_map[i].custom_data[j].get_type(), get_custom_data_layer_type(j))) {
					const Variant *args[] = { &item_map[i].custom_data[j] };
					Variant::construct(get_custom_data_layer_type(j), new_val, args, 1, error);
				} else {
					Variant::construct(get_custom_data_layer_type(j), new_val, nullptr, 0, error);
				}
				item_map[i].custom_data.write[j] = new_val;
			}
		}
	}

	notify_property_list_changed();
	emit_signal(CoreStringName(changed));
}

// Custom data.
int MeshLibrary::get_custom_data_layers_count() const {
	return custom_data_layers.size();
}

void MeshLibrary::add_custom_data_layer(int p_index) {
	if (p_index < 0) {
		p_index = custom_data_layers.size();
	}
	ERR_FAIL_INDEX(p_index, custom_data_layers.size() + 1);
	custom_data_layers.insert(p_index, CustomDataLayer());

	notify_property_list_changed();
	emit_changed();
}

void MeshLibrary::move_custom_data_layer(int p_from_index, int p_to_pos) {
	ERR_FAIL_INDEX(p_from_index, custom_data_layers.size());
	ERR_FAIL_INDEX(p_to_pos, custom_data_layers.size() + 1);
	custom_data_layers.insert(p_to_pos, custom_data_layers[p_from_index]);
	custom_data_layers.remove_at(p_to_pos < p_from_index ? p_from_index + 1 : p_from_index);
	notify_property_list_changed();
	emit_changed();
}

void MeshLibrary::remove_custom_data_layer(int p_index) {
	ERR_FAIL_INDEX(p_index, custom_data_layers.size());
	custom_data_layers.remove_at(p_index);

	String to_erase;
	for (KeyValue<String, int> &E : custom_data_layers_by_name) {
		if (E.value == p_index) {
			to_erase = E.key;
		} else if (E.value > p_index) {
			E.value--;
		}
	}
	custom_data_layers_by_name.erase(to_erase);
	notify_property_list_changed();
	emit_changed();
}

int MeshLibrary::get_custom_data_layer_by_name(String p_value) const {
	if (custom_data_layers_by_name.has(p_value)) {
		return custom_data_layers_by_name[p_value];
	} else {
		return -1;
	}
}

void MeshLibrary::set_custom_data_layer_name(int p_layer_id, String p_value) {
	ERR_FAIL_INDEX(p_layer_id, custom_data_layers.size());

	// Exit if another property has the same name.
	if (!p_value.is_empty()) {
		for (int other_layer_id = 0; other_layer_id < get_custom_data_layers_count(); other_layer_id++) {
			if (other_layer_id != p_layer_id && get_custom_data_layer_name(other_layer_id) == p_value) {
				ERR_FAIL_MSG(vformat("There is already a custom property named %s", p_value));
			}
		}
	}

	if (p_value.is_empty() && custom_data_layers_by_name.has(p_value)) {
		custom_data_layers_by_name.erase(p_value);
	} else {
		custom_data_layers_by_name[p_value] = p_layer_id;
	}

	custom_data_layers.write[p_layer_id].name = p_value;
	emit_changed();
}

String MeshLibrary::get_custom_data_layer_name(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, custom_data_layers.size(), "");
	return custom_data_layers[p_layer_id].name;
}

void MeshLibrary::set_custom_data_layer_type(int p_layer_id, Variant::Type p_value) {
	ERR_FAIL_INDEX(p_layer_id, custom_data_layers.size());
	custom_data_layers.write[p_layer_id].type = p_value;

	notify_mesh_library_properties_should_change();
	emit_changed();
}

Variant::Type MeshLibrary::get_custom_data_layer_type(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, custom_data_layers.size(), Variant::NIL);
	return custom_data_layers[p_layer_id].type;
}

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

// SETTER/GETTER ITEM SPECIFIC

// Custom data
void MeshLibrary::set_custom_data(int p_item, String p_layer_name, Variant p_value) {
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	int p_layer_id = get_custom_data_layer_by_name(p_layer_name);
	ERR_FAIL_COND_MSG(p_layer_id < 0, vformat("MeshLibrary has no layer with name: %s", p_layer_name));
	set_custom_data_by_layer_id(p_item, p_layer_id, p_value);
}

Variant MeshLibrary::get_custom_data(int p_item, String p_layer_name) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Variant(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	int p_layer_id = get_custom_data_layer_by_name(p_layer_name);
	ERR_FAIL_COND_V_MSG(p_layer_id < 0, Variant(), vformat("MeshLibrary has no layer with name: %s", p_layer_name));
	return get_custom_data_by_layer_id(p_item, p_layer_id);
}

void MeshLibrary::set_custom_data_by_layer_id(int p_item, int p_layer_id, Variant p_value) {
	ERR_FAIL_COND_MSG(!item_map.has(p_item), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	ERR_FAIL_INDEX(p_layer_id, item_map[p_item].custom_data.size());
	item_map[p_item].custom_data.write[p_layer_id] = p_value;
	emit_changed();
}

Variant MeshLibrary::get_custom_data_by_layer_id(int p_item, int p_layer_id) const {
	ERR_FAIL_COND_V_MSG(!item_map.has(p_item), Variant(), "Requested for nonexistent MeshLibrary item '" + itos(p_item) + "'.");
	ERR_FAIL_INDEX_V(p_layer_id, item_map[p_item].custom_data.size(), Variant());
	return item_map[p_item].custom_data[p_layer_id];
}

void MeshLibrary::reset_state() {
	clear();
}
void MeshLibrary::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_item", "id"), &MeshLibrary::create_item);
	ClassDB::bind_method(D_METHOD("set_item_name", "id", "name"), &MeshLibrary::set_item_name);
	ClassDB::bind_method(D_METHOD("set_item_mesh", "id", "mesh"), &MeshLibrary::set_item_mesh);
	ClassDB::bind_method(D_METHOD("set_item_mesh_transform", "id", "mesh_transform"), &MeshLibrary::set_item_mesh_transform);
	ClassDB::bind_method(D_METHOD("set_item_navigation_mesh", "id", "navigation_mesh"), &MeshLibrary::set_item_navigation_mesh);
	ClassDB::bind_method(D_METHOD("set_item_navigation_mesh_transform", "id", "navigation_mesh"), &MeshLibrary::set_item_navigation_mesh_transform);
	ClassDB::bind_method(D_METHOD("set_item_navigation_layers", "id", "navigation_layers"), &MeshLibrary::set_item_navigation_layers);
	ClassDB::bind_method(D_METHOD("set_item_shapes", "id", "shapes"), &MeshLibrary::_set_item_shapes);
	ClassDB::bind_method(D_METHOD("set_item_preview", "id", "texture"), &MeshLibrary::set_item_preview);
	ClassDB::bind_method(D_METHOD("get_item_name", "id"), &MeshLibrary::get_item_name);
	ClassDB::bind_method(D_METHOD("get_item_mesh", "id"), &MeshLibrary::get_item_mesh);
	ClassDB::bind_method(D_METHOD("get_item_mesh_transform", "id"), &MeshLibrary::get_item_mesh_transform);
	ClassDB::bind_method(D_METHOD("get_item_navigation_mesh", "id"), &MeshLibrary::get_item_navigation_mesh);
	ClassDB::bind_method(D_METHOD("get_item_navigation_mesh_transform", "id"), &MeshLibrary::get_item_navigation_mesh_transform);
	ClassDB::bind_method(D_METHOD("get_item_navigation_layers", "id"), &MeshLibrary::get_item_navigation_layers);
	ClassDB::bind_method(D_METHOD("get_item_shapes", "id"), &MeshLibrary::_get_item_shapes);
	ClassDB::bind_method(D_METHOD("get_item_preview", "id"), &MeshLibrary::get_item_preview);
	ClassDB::bind_method(D_METHOD("remove_item", "id"), &MeshLibrary::remove_item);
	ClassDB::bind_method(D_METHOD("find_item_by_name", "name"), &MeshLibrary::find_item_by_name);

	ClassDB::bind_method(D_METHOD("clear"), &MeshLibrary::clear);
	ClassDB::bind_method(D_METHOD("get_item_list"), &MeshLibrary::get_item_list);
	ClassDB::bind_method(D_METHOD("get_last_unused_item_id"), &MeshLibrary::get_last_unused_item_id);

	// New binds
	ClassDB::bind_method(D_METHOD("get_custom_data_layers_count"), &MeshLibrary::get_custom_data_layers_count);
	ClassDB::bind_method(D_METHOD("add_custom_data_layer", "index"), &MeshLibrary::add_custom_data_layer, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("move_custom_data_layer", "from_index", "to_pos"), &MeshLibrary::move_custom_data_layer);
	ClassDB::bind_method(D_METHOD("remove_custom_data_layer", "index"), &MeshLibrary::remove_custom_data_layer);
	ClassDB::bind_method(D_METHOD("get_custom_data_layer_by_name", "value"), &MeshLibrary::get_custom_data_layer_by_name);
	ClassDB::bind_method(D_METHOD("set_custom_data_layer_name", "layer_id", "value"), &MeshLibrary::set_custom_data_layer_name);
	ClassDB::bind_method(D_METHOD("get_custom_data_layer_name", "layer_id"), &MeshLibrary::get_custom_data_layer_name);
	ClassDB::bind_method(D_METHOD("set_custom_data_layer_type", "layer_id", "value"), &MeshLibrary::set_custom_data_layer_type);
	ClassDB::bind_method(D_METHOD("get_custom_data_layer_type", "layer_id"), &MeshLibrary::get_custom_data_layer_type);

	ClassDB::bind_method(D_METHOD("set_custom_data", "item", "layer_name", "value"), &MeshLibrary::set_custom_data);
	ClassDB::bind_method(D_METHOD("get_custom_data", "item", "layer_name"), &MeshLibrary::get_custom_data);
	ClassDB::bind_method(D_METHOD("set_custom_data_by_layer_id", "item", "layer_id", "value"), &MeshLibrary::set_custom_data_by_layer_id);
	ClassDB::bind_method(D_METHOD("get_custom_data_by_layer_id", "item", "layer_id"), &MeshLibrary::get_custom_data_by_layer_id);
}

MeshLibrary::MeshLibrary() {
}

MeshLibrary::~MeshLibrary() {
}
