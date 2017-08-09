/*************************************************************************/
/*  tile_set.cpp                                                         */
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
#include "tile_set.h"

bool TileSet::_set(const StringName &p_name, const Variant &p_value) {

	String n = p_name;
	int slash = n.find("/");
	if (slash == -1)
		return false;
	int id = String::to_int(n.c_str(), slash);

	if (!tile_map.has(id))
		create_tile(id);
	String what = n.substr(slash + 1, n.length());

	if (what == "name")
		tile_set_name(id, p_value);
	else if (what == "texture")
		tile_set_texture(id, p_value);
	else if (what == "normal_map")
		tile_set_normal_map(id, p_value);
	else if (what == "tex_offset")
		tile_set_texture_offset(id, p_value);
	else if (what == "material")
		tile_set_material(id, p_value);
	else if (what == "modulate")
		tile_set_modulate(id, p_value);
	else if (what == "region")
		tile_set_region(id, p_value);
	else if (what == "shape")
		tile_set_shape(id, 0, p_value);
	else if (what == "shape_offset") {
		Transform2D xform = tile_get_shape_transform(id, 0);
		xform.set_origin(p_value);
		tile_set_shape_transform(id, 0, xform);
	} else if (what == "shape_transform")
		tile_set_shape_transform(id, 0, p_value);
	else if (what == "shape_one_way")
		tile_set_shape_one_way(id, 0, p_value);
	else if (what == "shapes")
		_tile_set_shapes(id, p_value);
	else if (what == "occluder")
		tile_set_light_occluder(id, p_value);
	else if (what == "occluder_offset")
		tile_set_occluder_offset(id, p_value);
	else if (what == "navigation")
		tile_set_navigation_polygon(id, p_value);
	else if (what == "navigation_offset")
		tile_set_navigation_polygon_offset(id, p_value);
	else
		return false;

	return true;
}

bool TileSet::_get(const StringName &p_name, Variant &r_ret) const {

	String n = p_name;
	int slash = n.find("/");
	if (slash == -1)
		return false;
	int id = String::to_int(n.c_str(), slash);

	ERR_FAIL_COND_V(!tile_map.has(id), false);

	String what = n.substr(slash + 1, n.length());

	if (what == "name")
		r_ret = tile_get_name(id);
	else if (what == "texture")
		r_ret = tile_get_texture(id);
	else if (what == "normal_map")
		r_ret = tile_get_normal_map(id);
	else if (what == "tex_offset")
		r_ret = tile_get_texture_offset(id);
	else if (what == "material")
		r_ret = tile_get_material(id);
	else if (what == "modulate")
		r_ret = tile_get_modulate(id);
	else if (what == "region")
		r_ret = tile_get_region(id);
	else if (what == "shape")
		r_ret = tile_get_shape(id, 0);
	else if (what == "shape_offset")
		r_ret = tile_get_shape_transform(id, 0).get_origin();
	else if (what == "shape_transform")
		r_ret = tile_get_shape_transform(id, 0);
	else if (what == "shape_one_way")
		r_ret = tile_get_shape_one_way(id, 0);
	else if (what == "shapes")
		r_ret = _tile_get_shapes(id);
	else if (what == "occluder")
		r_ret = tile_get_light_occluder(id);
	else if (what == "occluder_offset")
		r_ret = tile_get_occluder_offset(id);
	else if (what == "navigation")
		r_ret = tile_get_navigation_polygon(id);
	else if (what == "navigation_offset")
		r_ret = tile_get_navigation_polygon_offset(id);
	else
		return false;

	return true;
}

void TileSet::_get_property_list(List<PropertyInfo> *p_list) const {

	for (Map<int, TileData>::Element *E = tile_map.front(); E; E = E->next()) {

		int id = E->key();
		String pre = itos(id) + "/";
		p_list->push_back(PropertyInfo(Variant::STRING, pre + "name"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, pre + "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, pre + "normal_map", PROPERTY_HINT_RESOURCE_TYPE, "Texture"));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "tex_offset"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, pre + "material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial"));
		p_list->push_back(PropertyInfo(Variant::COLOR, pre + "modulate"));
		p_list->push_back(PropertyInfo(Variant::RECT2, pre + "region"));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "occluder_offset"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, pre + "occluder", PROPERTY_HINT_RESOURCE_TYPE, "OccluderPolygon2D"));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "navigation_offset"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, pre + "navigation", PROPERTY_HINT_RESOURCE_TYPE, "NavigationPolygon"));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "shape_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "shape_transform", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::OBJECT, pre + "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape2D", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::BOOL, pre + "shape_one_way", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::ARRAY, pre + "shapes", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	}
}

void TileSet::create_tile(int p_id) {

	ERR_FAIL_COND(tile_map.has(p_id));
	tile_map[p_id] = TileData();
	_change_notify("");
	emit_changed();
}

void TileSet::tile_set_texture(int p_id, const Ref<Texture> &p_texture) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].texture = p_texture;
	emit_changed();
}

Ref<Texture> TileSet::tile_get_texture(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Ref<Texture>());
	return tile_map[p_id].texture;
}

void TileSet::tile_set_normal_map(int p_id, const Ref<Texture> &p_normal_map) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].normal_map = p_normal_map;
	emit_changed();
}

Ref<Texture> TileSet::tile_get_normal_map(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Ref<Texture>());
	return tile_map[p_id].normal_map;
}

void TileSet::tile_set_material(int p_id, const Ref<ShaderMaterial> &p_material) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].material = p_material;
	emit_changed();
}

Ref<ShaderMaterial> TileSet::tile_get_material(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Ref<ShaderMaterial>());
	return tile_map[p_id].material;
}

void TileSet::tile_set_modulate(int p_id, const Color &p_modulate) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].modulate = p_modulate;
	emit_changed();
}

Color TileSet::tile_get_modulate(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Color(1, 1, 1));
	return tile_map[p_id].modulate;
}

void TileSet::tile_set_texture_offset(int p_id, const Vector2 &p_offset) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].offset = p_offset;
	emit_changed();
}

Vector2 TileSet::tile_get_texture_offset(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Vector2());
	return tile_map[p_id].offset;
}

void TileSet::tile_set_region(int p_id, const Rect2 &p_region) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].region = p_region;
	emit_changed();
}

Rect2 TileSet::tile_get_region(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Rect2());
	return tile_map[p_id].region;
}

void TileSet::tile_set_name(int p_id, const String &p_name) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].name = p_name;
	emit_changed();
}

String TileSet::tile_get_name(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), String());
	return tile_map[p_id].name;
}

void TileSet::tile_clear_shapes(int p_id) {
	tile_map[p_id].shapes_data.clear();
}

void TileSet::tile_add_shape(int p_id, const Ref<Shape2D> &p_shape, const Transform2D &p_transform, bool p_one_way) {

	ERR_FAIL_COND(!tile_map.has(p_id));

	ShapeData new_data = ShapeData();
	new_data.shape = p_shape;
	new_data.shape_transform = p_transform;
	new_data.one_way_collision = p_one_way;

	tile_map[p_id].shapes_data.push_back(new_data);
};
int TileSet::tile_get_shape_count(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), 0);

	return tile_map[p_id].shapes_data.size();
};

void TileSet::tile_set_shape(int p_id, int p_shape_id, const Ref<Shape2D> &p_shape) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	if (tile_map[p_id].shapes_data.size() <= p_shape_id)
		tile_map[p_id].shapes_data.resize(p_shape_id + 1);
	tile_map[p_id].shapes_data[p_shape_id].shape = p_shape;
	emit_changed();
}

Ref<Shape2D> TileSet::tile_get_shape(int p_id, int p_shape_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Ref<Shape2D>());
	if (tile_map[p_id].shapes_data.size() > p_shape_id)
		return tile_map[p_id].shapes_data[p_shape_id].shape;

	return Ref<Shape2D>();
}

void TileSet::tile_set_shape_transform(int p_id, int p_shape_id, const Transform2D &p_offset) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	if (tile_map[p_id].shapes_data.size() <= p_shape_id)
		tile_map[p_id].shapes_data.resize(p_shape_id + 1);
	tile_map[p_id].shapes_data[p_shape_id].shape_transform = p_offset;
	emit_changed();
}

Transform2D TileSet::tile_get_shape_transform(int p_id, int p_shape_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Transform2D());
	if (tile_map[p_id].shapes_data.size() > p_shape_id)
		return tile_map[p_id].shapes_data[p_shape_id].shape_transform;

	return Transform2D();
}

void TileSet::tile_set_shape_one_way(int p_id, int p_shape_id, const bool p_one_way) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	if (tile_map[p_id].shapes_data.size() <= p_shape_id)
		tile_map[p_id].shapes_data.resize(p_shape_id + 1);
	tile_map[p_id].shapes_data[p_shape_id].one_way_collision = p_one_way;
	emit_changed();
}

bool TileSet::tile_get_shape_one_way(int p_id, int p_shape_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), false);
	if (tile_map[p_id].shapes_data.size() > p_shape_id)
		return tile_map[p_id].shapes_data[p_shape_id].one_way_collision;

	return false;
}

void TileSet::tile_set_light_occluder(int p_id, const Ref<OccluderPolygon2D> &p_light_occluder) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].occluder = p_light_occluder;
}

Ref<OccluderPolygon2D> TileSet::tile_get_light_occluder(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Ref<OccluderPolygon2D>());
	return tile_map[p_id].occluder;
}

void TileSet::tile_set_navigation_polygon_offset(int p_id, const Vector2 &p_offset) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].navigation_polygon_offset = p_offset;
}

Vector2 TileSet::tile_get_navigation_polygon_offset(int p_id) const {
	ERR_FAIL_COND_V(!tile_map.has(p_id), Vector2());
	return tile_map[p_id].navigation_polygon_offset;
}

void TileSet::tile_set_navigation_polygon(int p_id, const Ref<NavigationPolygon> &p_navigation_polygon) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].navigation_polygon = p_navigation_polygon;
}

Ref<NavigationPolygon> TileSet::tile_get_navigation_polygon(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Ref<NavigationPolygon>());
	return tile_map[p_id].navigation_polygon;
}

void TileSet::tile_set_occluder_offset(int p_id, const Vector2 &p_offset) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].occluder_offset = p_offset;
}

Vector2 TileSet::tile_get_occluder_offset(int p_id) const {
	ERR_FAIL_COND_V(!tile_map.has(p_id), Vector2());
	return tile_map[p_id].occluder_offset;
}

void TileSet::tile_set_shapes(int p_id, const Vector<ShapeData> &p_shapes) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].shapes_data = p_shapes;
	emit_changed();
}

Vector<TileSet::ShapeData> TileSet::tile_get_shapes(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Vector<ShapeData>());

	return tile_map[p_id].shapes_data;
}

void TileSet::_tile_set_shapes(int p_id, const Array &p_shapes) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	Vector<ShapeData> shapes_data;
	Transform2D default_transform = tile_get_shape_transform(p_id, 0);
	bool default_one_way = tile_get_shape_one_way(p_id, 0);
	for (int i = 0; i < p_shapes.size(); i++) {
		ShapeData s = ShapeData();

		if (p_shapes[i].get_type() == Variant::OBJECT) {
			Ref<Shape2D> shape = p_shapes[i];
			if (shape.is_null()) continue;

			s.shape = shape;
			s.shape_transform = default_transform;
			s.one_way_collision = default_one_way;
		} else if (p_shapes[i].get_type() == Variant::DICTIONARY) {
			Dictionary d = p_shapes[i];

			if (d.has("shape") && d["shape"].get_type() == Variant::OBJECT)
				s.shape = d["shape"];
			else
				continue;

			if (d.has("shape_transform") && d["shape_transform"].get_type() == Variant::TRANSFORM2D)
				s.shape_transform = d["shape_transform"];
			else if (d.has("shape_offset") && d["shape_offset"].get_type() == Variant::VECTOR2)
				s.shape_transform = Transform2D(0, (Vector2)d["shape_offset"]);
			else
				s.shape_transform = default_transform;

			if (d.has("one_way") && d["one_way"].get_type() == Variant::BOOL)
				s.one_way_collision = d["one_way"];
			else
				s.one_way_collision = default_one_way;

		} else {
			ERR_EXPLAIN("Expected an array of objects or dictionaries for tile_set_shapes");
			ERR_CONTINUE(true);
		}

		shapes_data.push_back(s);
	}

	tile_map[p_id].shapes_data = shapes_data;
}

Array TileSet::_tile_get_shapes(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Array());
	Array arr;

	Vector<ShapeData> data = tile_map[p_id].shapes_data;
	for (int i = 0; i < data.size(); i++) {
		Dictionary shape_data;
		shape_data["shape"] = data[i].shape;
		shape_data["shape_transform"] = data[i].shape_transform;
		shape_data["one_way"] = data[i].one_way_collision;
		arr.push_back(shape_data);
	}

	return arr;
}

Array TileSet::_get_tiles_ids() const {

	Array arr;

	for (Map<int, TileData>::Element *E = tile_map.front(); E; E = E->next()) {
		arr.push_back(E->key());
	}

	return arr;
}

void TileSet::get_tile_list(List<int> *p_tiles) const {

	for (Map<int, TileData>::Element *E = tile_map.front(); E; E = E->next()) {

		p_tiles->push_back(E->key());
	}
}

bool TileSet::has_tile(int p_id) const {

	return tile_map.has(p_id);
}

void TileSet::remove_tile(int p_id) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map.erase(p_id);
	_change_notify("");
	emit_changed();
}

int TileSet::get_last_unused_tile_id() const {

	if (tile_map.size())
		return tile_map.back()->key() + 1;
	else
		return 0;
}

int TileSet::find_tile_by_name(const String &p_name) const {

	for (Map<int, TileData>::Element *E = tile_map.front(); E; E = E->next()) {

		if (p_name == E->get().name)
			return E->key();
	}
	return -1;
}

void TileSet::clear() {

	tile_map.clear();
	_change_notify("");
	emit_changed();
}

void TileSet::_bind_methods() {

	ClassDB::bind_method(D_METHOD("create_tile", "id"), &TileSet::create_tile);
	ClassDB::bind_method(D_METHOD("tile_set_name", "id", "name"), &TileSet::tile_set_name);
	ClassDB::bind_method(D_METHOD("tile_get_name", "id"), &TileSet::tile_get_name);
	ClassDB::bind_method(D_METHOD("tile_set_texture", "id", "texture"), &TileSet::tile_set_texture);
	ClassDB::bind_method(D_METHOD("tile_get_texture", "id"), &TileSet::tile_get_texture);
	ClassDB::bind_method(D_METHOD("tile_set_normal_map", "id", "normal_map"), &TileSet::tile_set_normal_map);
	ClassDB::bind_method(D_METHOD("tile_get_normal_map", "id"), &TileSet::tile_get_normal_map);
	ClassDB::bind_method(D_METHOD("tile_set_material", "id", "material"), &TileSet::tile_set_material);
	ClassDB::bind_method(D_METHOD("tile_get_material", "id"), &TileSet::tile_get_material);
	ClassDB::bind_method(D_METHOD("tile_set_texture_offset", "id", "texture_offset"), &TileSet::tile_set_texture_offset);
	ClassDB::bind_method(D_METHOD("tile_get_texture_offset", "id"), &TileSet::tile_get_texture_offset);
	ClassDB::bind_method(D_METHOD("tile_set_region", "id", "region"), &TileSet::tile_set_region);
	ClassDB::bind_method(D_METHOD("tile_get_region", "id"), &TileSet::tile_get_region);
	ClassDB::bind_method(D_METHOD("tile_set_shape", "id", "shape_id", "shape"), &TileSet::tile_set_shape);
	ClassDB::bind_method(D_METHOD("tile_get_shape", "id", "shape_id"), &TileSet::tile_get_shape);
	ClassDB::bind_method(D_METHOD("tile_set_shape_transform", "id", "shape_id", "shape_transform"), &TileSet::tile_set_shape_transform);
	ClassDB::bind_method(D_METHOD("tile_get_shape_transform", "id", "shape_id"), &TileSet::tile_get_shape_transform);
	ClassDB::bind_method(D_METHOD("tile_set_shape_one_way", "id", "shape_id", "one_way"), &TileSet::tile_set_shape_one_way);
	ClassDB::bind_method(D_METHOD("tile_get_shape_one_way", "id", "shape_id"), &TileSet::tile_get_shape_one_way);
	ClassDB::bind_method(D_METHOD("tile_add_shape", "id", "shape", "shape_transform", "one_way"), &TileSet::tile_add_shape, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("tile_get_shape_count", "id"), &TileSet::tile_get_shape_count);
	ClassDB::bind_method(D_METHOD("tile_set_shapes", "id", "shapes"), &TileSet::_tile_set_shapes);
	ClassDB::bind_method(D_METHOD("tile_get_shapes", "id"), &TileSet::_tile_get_shapes);
	ClassDB::bind_method(D_METHOD("tile_set_navigation_polygon", "id", "navigation_polygon"), &TileSet::tile_set_navigation_polygon);
	ClassDB::bind_method(D_METHOD("tile_get_navigation_polygon", "id"), &TileSet::tile_get_navigation_polygon);
	ClassDB::bind_method(D_METHOD("tile_set_navigation_polygon_offset", "id", "navigation_polygon_offset"), &TileSet::tile_set_navigation_polygon_offset);
	ClassDB::bind_method(D_METHOD("tile_get_navigation_polygon_offset", "id"), &TileSet::tile_get_navigation_polygon_offset);
	ClassDB::bind_method(D_METHOD("tile_set_light_occluder", "id", "light_occluder"), &TileSet::tile_set_light_occluder);
	ClassDB::bind_method(D_METHOD("tile_get_light_occluder", "id"), &TileSet::tile_get_light_occluder);
	ClassDB::bind_method(D_METHOD("tile_set_occluder_offset", "id", "occluder_offset"), &TileSet::tile_set_occluder_offset);
	ClassDB::bind_method(D_METHOD("tile_get_occluder_offset", "id"), &TileSet::tile_get_occluder_offset);

	ClassDB::bind_method(D_METHOD("remove_tile", "id"), &TileSet::remove_tile);
	ClassDB::bind_method(D_METHOD("clear"), &TileSet::clear);
	ClassDB::bind_method(D_METHOD("get_last_unused_tile_id"), &TileSet::get_last_unused_tile_id);
	ClassDB::bind_method(D_METHOD("find_tile_by_name", "name"), &TileSet::find_tile_by_name);
	ClassDB::bind_method(D_METHOD("get_tiles_ids"), &TileSet::_get_tiles_ids);
}

TileSet::TileSet() {
}
