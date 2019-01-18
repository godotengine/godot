/*************************************************************************/
/*  tile_set.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "core/array.h"

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
	else if (what == "tile_mode")
		tile_set_tile_mode(id, (TileMode)((int)p_value));
	else if (what == "is_autotile") {
		// backward compatibility for Godot 3.0.x
		// autotile used to be a bool, it's now an enum
		bool is_autotile = p_value;
		if (is_autotile)
			tile_set_tile_mode(id, AUTO_TILE);
	} else if (what.left(9) == "autotile/") {
		what = what.right(9);
		if (what == "bitmask_mode")
			autotile_set_bitmask_mode(id, (BitmaskMode)((int)p_value));
		else if (what == "icon_coordinate")
			autotile_set_icon_coordinate(id, p_value);
		else if (what == "tile_size")
			autotile_set_size(id, p_value);
		else if (what == "spacing")
			autotile_set_spacing(id, p_value);
		else if (what == "bitmask_flags") {
			tile_map[id].autotile_data.flags.clear();
			if (p_value.is_array()) {
				Array p = p_value;
				Vector2 last_coord;
				while (p.size() > 0) {
					if (p[0].get_type() == Variant::VECTOR2) {
						last_coord = p[0];
					} else if (p[0].get_type() == Variant::INT) {
						autotile_set_bitmask(id, last_coord, p[0]);
					}
					p.pop_front();
				}
			}
		} else if (what == "occluder_map") {
			tile_map[id].autotile_data.occluder_map.clear();
			Array p = p_value;
			Vector2 last_coord;
			while (p.size() > 0) {
				if (p[0].get_type() == Variant::VECTOR2) {
					last_coord = p[0];
				} else if (p[0].get_type() == Variant::OBJECT) {
					autotile_set_light_occluder(id, p[0], last_coord);
				}
				p.pop_front();
			}
		} else if (what == "navpoly_map") {
			tile_map[id].autotile_data.navpoly_map.clear();
			Array p = p_value;
			Vector2 last_coord;
			while (p.size() > 0) {
				if (p[0].get_type() == Variant::VECTOR2) {
					last_coord = p[0];
				} else if (p[0].get_type() == Variant::OBJECT) {
					autotile_set_navigation_polygon(id, p[0], last_coord);
				}
				p.pop_front();
			}
		} else if (what == "priority_map") {
			tile_map[id].autotile_data.priority_map.clear();
			Array p = p_value;
			Vector3 val;
			Vector2 v;
			int priority;
			while (p.size() > 0) {
				val = p[0];
				if (val.z > 1) {
					v.x = val.x;
					v.y = val.y;
					priority = (int)val.z;
					tile_map[id].autotile_data.priority_map[v] = priority;
				}
				p.pop_front();
			}
		} else if (what == "z_index_map") {
			tile_map[id].autotile_data.z_index_map.clear();
			Array p = p_value;
			Vector3 val;
			Vector2 v;
			int z_index;
			while (p.size() > 0) {
				val = p[0];
				if (val.z != 0) {
					v.x = val.x;
					v.y = val.y;
					z_index = (int)val.z;
					tile_map[id].autotile_data.z_index_map[v] = z_index;
				}
				p.pop_front();
			}
		}
	} else if (what == "shape")
		tile_set_shape(id, 0, p_value);
	else if (what == "shape_offset")
		tile_set_shape_offset(id, 0, p_value);
	else if (what == "shape_transform")
		tile_set_shape_transform(id, 0, p_value);
	else if (what == "shape_one_way")
		tile_set_shape_one_way(id, 0, p_value);
	else if (what == "shape_one_way_margin")
		tile_set_shape_one_way_margin(id, 0, p_value);
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
	else if (what == "z_index")
		tile_set_z_index(id, p_value);
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
	else if (what == "tile_mode")
		r_ret = tile_get_tile_mode(id);
	else if (what.left(9) == "autotile/") {
		what = what.right(9);
		if (what == "bitmask_mode")
			r_ret = autotile_get_bitmask_mode(id);
		else if (what == "icon_coordinate")
			r_ret = autotile_get_icon_coordinate(id);
		else if (what == "tile_size")
			r_ret = autotile_get_size(id);
		else if (what == "spacing")
			r_ret = autotile_get_spacing(id);
		else if (what == "bitmask_flags") {
			Array p;
			for (Map<Vector2, uint16_t>::Element *E = tile_map[id].autotile_data.flags.front(); E; E = E->next()) {
				p.push_back(E->key());
				p.push_back(E->value());
			}
			r_ret = p;
		} else if (what == "occluder_map") {
			Array p;
			for (Map<Vector2, Ref<OccluderPolygon2D> >::Element *E = tile_map[id].autotile_data.occluder_map.front(); E; E = E->next()) {
				p.push_back(E->key());
				p.push_back(E->value());
			}
			r_ret = p;
		} else if (what == "navpoly_map") {
			Array p;
			for (Map<Vector2, Ref<NavigationPolygon> >::Element *E = tile_map[id].autotile_data.navpoly_map.front(); E; E = E->next()) {
				p.push_back(E->key());
				p.push_back(E->value());
			}
			r_ret = p;
		} else if (what == "priority_map") {
			Array p;
			Vector3 v;
			for (Map<Vector2, int>::Element *E = tile_map[id].autotile_data.priority_map.front(); E; E = E->next()) {
				if (E->value() > 1) {
					//Don't save default value
					v.x = E->key().x;
					v.y = E->key().y;
					v.z = E->value();
					p.push_back(v);
				}
			}
			r_ret = p;
		} else if (what == "z_index_map") {
			Array p;
			Vector3 v;
			for (Map<Vector2, int>::Element *E = tile_map[id].autotile_data.z_index_map.front(); E; E = E->next()) {
				if (E->value() != 0) {
					//Don't save default value
					v.x = E->key().x;
					v.y = E->key().y;
					v.z = E->value();
					p.push_back(v);
				}
			}
			r_ret = p;
		}
	} else if (what == "shape")
		r_ret = tile_get_shape(id, 0);
	else if (what == "shape_offset")
		r_ret = tile_get_shape_offset(id, 0);
	else if (what == "shape_transform")
		r_ret = tile_get_shape_transform(id, 0);
	else if (what == "shape_one_way")
		r_ret = tile_get_shape_one_way(id, 0);
	else if (what == "shape_one_way_margin")
		r_ret = tile_get_shape_one_way_margin(id, 0);
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
	else if (what == "z_index")
		r_ret = tile_get_z_index(id);
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
		p_list->push_back(PropertyInfo(Variant::INT, pre + "tile_mode", PROPERTY_HINT_ENUM, "SINGLE_TILE,AUTO_TILE,ATLAS_TILE"));
		if (tile_get_tile_mode(id) == AUTO_TILE) {
			p_list->push_back(PropertyInfo(Variant::INT, pre + "autotile/bitmask_mode", PROPERTY_HINT_ENUM, "2X2,3X3 (minimal),3X3", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::ARRAY, pre + "autotile/bitmask_flags", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "autotile/icon_coordinate", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "autotile/tile_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::INT, pre + "autotile/spacing", PROPERTY_HINT_RANGE, "0,256,1", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::ARRAY, pre + "autotile/occluder_map", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::ARRAY, pre + "autotile/navpoly_map", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::ARRAY, pre + "autotile/priority_map", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::ARRAY, pre + "autotile/z_index_map", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
		} else if (tile_get_tile_mode(id) == ATLAS_TILE) {
			p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "autotile/icon_coordinate", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "autotile/tile_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::INT, pre + "autotile/spacing", PROPERTY_HINT_RANGE, "0,256,1", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::ARRAY, pre + "autotile/occluder_map", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::ARRAY, pre + "autotile/navpoly_map", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::ARRAY, pre + "autotile/z_index_map", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
		}
		p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "occluder_offset"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, pre + "occluder", PROPERTY_HINT_RESOURCE_TYPE, "OccluderPolygon2D"));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "navigation_offset"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, pre + "navigation", PROPERTY_HINT_RESOURCE_TYPE, "NavigationPolygon"));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "shape_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, pre + "shape_transform", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::OBJECT, pre + "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape2D", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::BOOL, pre + "shape_one_way", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::REAL, pre + "shape_one_way_margin", PROPERTY_HINT_RANGE, "0,128,0.01", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::ARRAY, pre + "shapes", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::INT, pre + "z_index", PROPERTY_HINT_RANGE, itos(VS::CANVAS_ITEM_Z_MIN) + "," + itos(VS::CANVAS_ITEM_Z_MAX) + ",1"));
	}
}

void TileSet::create_tile(int p_id) {
	ERR_FAIL_COND(tile_map.has(p_id));
	tile_map[p_id] = TileData();
	tile_map[p_id].autotile_data = AutotileData();
	_change_notify("");
	emit_changed();
}

void TileSet::autotile_set_bitmask_mode(int p_id, BitmaskMode p_mode) {
	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].autotile_data.bitmask_mode = p_mode;
	_change_notify("");
	emit_changed();
}

TileSet::BitmaskMode TileSet::autotile_get_bitmask_mode(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), BITMASK_2X2);
	return tile_map[p_id].autotile_data.bitmask_mode;
}

void TileSet::tile_set_texture(int p_id, const Ref<Texture> &p_texture) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].texture = p_texture;
	emit_changed();
	_change_notify("texture");
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
	_change_notify("modulate");
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
	_change_notify("region");
}

Rect2 TileSet::tile_get_region(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Rect2());
	return tile_map[p_id].region;
}

void TileSet::tile_set_tile_mode(int p_id, TileMode p_tile_mode) {
	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].tile_mode = p_tile_mode;
	emit_changed();
	_change_notify("tile_mode");
}

TileSet::TileMode TileSet::tile_get_tile_mode(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), SINGLE_TILE);
	return tile_map[p_id].tile_mode;
}

void TileSet::autotile_set_icon_coordinate(int p_id, Vector2 coord) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].autotile_data.icon_coord = coord;
	emit_changed();
}

Vector2 TileSet::autotile_get_icon_coordinate(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Vector2());
	return tile_map[p_id].autotile_data.icon_coord;
}

void TileSet::autotile_set_spacing(int p_id, int p_spacing) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	ERR_FAIL_COND(p_spacing < 0);
	tile_map[p_id].autotile_data.spacing = p_spacing;
	emit_changed();
}

int TileSet::autotile_get_spacing(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), 0);
	return tile_map[p_id].autotile_data.spacing;
}

void TileSet::autotile_set_size(int p_id, Size2 p_size) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	ERR_FAIL_COND(p_size.x <= 0 || p_size.y <= 0);
	tile_map[p_id].autotile_data.size = p_size;
}

Size2 TileSet::autotile_get_size(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Size2());
	return tile_map[p_id].autotile_data.size;
}

void TileSet::autotile_clear_bitmask_map(int p_id) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].autotile_data.flags.clear();
}

void TileSet::autotile_set_subtile_priority(int p_id, const Vector2 &p_coord, int p_priority) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	ERR_FAIL_COND(p_priority <= 0);
	tile_map[p_id].autotile_data.priority_map[p_coord] = p_priority;
}

int TileSet::autotile_get_subtile_priority(int p_id, const Vector2 &p_coord) {

	ERR_FAIL_COND_V(!tile_map.has(p_id), 1);
	if (tile_map[p_id].autotile_data.priority_map.has(p_coord)) {
		return tile_map[p_id].autotile_data.priority_map[p_coord];
	}
	//When not custom priority set return the default value
	return 1;
}

const Map<Vector2, int> &TileSet::autotile_get_priority_map(int p_id) const {

	static Map<Vector2, int> dummy;
	ERR_FAIL_COND_V(!tile_map.has(p_id), dummy);
	return tile_map[p_id].autotile_data.priority_map;
}

void TileSet::autotile_set_z_index(int p_id, const Vector2 &p_coord, int p_z_index) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].autotile_data.z_index_map[p_coord] = p_z_index;
	emit_changed();
}

int TileSet::autotile_get_z_index(int p_id, const Vector2 &p_coord) {

	ERR_FAIL_COND_V(!tile_map.has(p_id), 1);
	if (tile_map[p_id].autotile_data.z_index_map.has(p_coord)) {
		return tile_map[p_id].autotile_data.z_index_map[p_coord];
	}
	//When not custom z index set return the default value
	return 0;
}

const Map<Vector2, int> &TileSet::autotile_get_z_index_map(int p_id) const {

	static Map<Vector2, int> dummy;
	ERR_FAIL_COND_V(!tile_map.has(p_id), dummy);
	return tile_map[p_id].autotile_data.z_index_map;
}

void TileSet::autotile_set_bitmask(int p_id, Vector2 p_coord, uint16_t p_flag) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	if (p_flag == 0) {
		if (tile_map[p_id].autotile_data.flags.has(p_coord))
			tile_map[p_id].autotile_data.flags.erase(p_coord);
	} else {
		tile_map[p_id].autotile_data.flags[p_coord] = p_flag;
	}
}

uint16_t TileSet::autotile_get_bitmask(int p_id, Vector2 p_coord) {

	ERR_FAIL_COND_V(!tile_map.has(p_id), 0);
	if (!tile_map[p_id].autotile_data.flags.has(p_coord)) {
		return 0;
	}
	return tile_map[p_id].autotile_data.flags[p_coord];
}

const Map<Vector2, uint16_t> &TileSet::autotile_get_bitmask_map(int p_id) {

	static Map<Vector2, uint16_t> dummy;
	static Map<Vector2, uint16_t> dummy_atlas;
	ERR_FAIL_COND_V(!tile_map.has(p_id), dummy);
	if (tile_get_tile_mode(p_id) == ATLAS_TILE) {
		dummy_atlas = Map<Vector2, uint16_t>();
		Rect2 region = tile_get_region(p_id);
		Size2 size = autotile_get_size(p_id);
		float spacing = autotile_get_spacing(p_id);
		for (int x = 0; x < (region.size.x / (size.x + spacing)); x++) {
			for (int y = 0; y < (region.size.y / (size.y + spacing)); y++) {
				dummy_atlas.insert(Vector2(x, y), 0);
			}
		}
		return dummy_atlas;
	} else
		return tile_map[p_id].autotile_data.flags;
}

Vector2 TileSet::autotile_get_subtile_for_bitmask(int p_id, uint16_t p_bitmask, const Node *p_tilemap_node, const Vector2 &p_tile_location) {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Vector2());
	//First try to forward selection to script
	if (p_tilemap_node->get_class_name() == "TileMap") {
		if (get_script_instance() != NULL) {
			if (get_script_instance()->has_method("_forward_subtile_selection")) {
				Variant ret = get_script_instance()->call("_forward_subtile_selection", p_id, p_bitmask, p_tilemap_node, p_tile_location);
				if (ret.get_type() == Variant::VECTOR2) {
					return ret;
				}
			}
		}
	}

	List<Vector2> coords;
	uint16_t mask;
	for (Map<Vector2, uint16_t>::Element *E = tile_map[p_id].autotile_data.flags.front(); E; E = E->next()) {
		mask = E->get();
		if (tile_map[p_id].autotile_data.bitmask_mode == BITMASK_2X2) {
			mask &= (BIND_BOTTOMLEFT | BIND_BOTTOMRIGHT | BIND_TOPLEFT | BIND_TOPRIGHT);
		}
		if (mask == p_bitmask) {
			for (int i = 0; i < autotile_get_subtile_priority(p_id, E->key()); i++) {
				coords.push_back(E->key());
			}
		}
	}
	if (coords.size() == 0) {
		return autotile_get_icon_coordinate(p_id);
	} else {
		return coords[Math::random(0, (int)coords.size())];
	}
}

void TileSet::tile_set_name(int p_id, const String &p_name) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].name = p_name;
	emit_changed();
	_change_notify("name");
}

String TileSet::tile_get_name(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), String());
	return tile_map[p_id].name;
}

void TileSet::tile_clear_shapes(int p_id) {
	tile_map[p_id].shapes_data.clear();
}

void TileSet::tile_add_shape(int p_id, const Ref<Shape2D> &p_shape, const Transform2D &p_transform, bool p_one_way, const Vector2 &p_autotile_coord) {

	ERR_FAIL_COND(!tile_map.has(p_id));

	ShapeData new_data = ShapeData();
	new_data.shape = p_shape;
	new_data.shape_transform = p_transform;
	new_data.one_way_collision = p_one_way;
	new_data.autotile_coord = p_autotile_coord;

	tile_map[p_id].shapes_data.push_back(new_data);
}

int TileSet::tile_get_shape_count(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), 0);

	return tile_map[p_id].shapes_data.size();
}

void TileSet::tile_set_shape(int p_id, int p_shape_id, const Ref<Shape2D> &p_shape) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	if (tile_map[p_id].shapes_data.size() <= p_shape_id)
		tile_map[p_id].shapes_data.resize(p_shape_id + 1);
	tile_map[p_id].shapes_data.write[p_shape_id].shape = p_shape;
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
	tile_map[p_id].shapes_data.write[p_shape_id].shape_transform = p_offset;
	emit_changed();
}

Transform2D TileSet::tile_get_shape_transform(int p_id, int p_shape_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Transform2D());
	if (tile_map[p_id].shapes_data.size() > p_shape_id)
		return tile_map[p_id].shapes_data[p_shape_id].shape_transform;

	return Transform2D();
}

void TileSet::tile_set_shape_offset(int p_id, int p_shape_id, const Vector2 &p_offset) {
	Transform2D transform = tile_get_shape_transform(p_id, p_shape_id);
	transform.set_origin(p_offset);
	tile_set_shape_transform(p_id, p_shape_id, transform);
}

Vector2 TileSet::tile_get_shape_offset(int p_id, int p_shape_id) const {
	return tile_get_shape_transform(p_id, p_shape_id).get_origin();
}

void TileSet::tile_set_shape_one_way(int p_id, int p_shape_id, const bool p_one_way) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	if (tile_map[p_id].shapes_data.size() <= p_shape_id)
		tile_map[p_id].shapes_data.resize(p_shape_id + 1);
	tile_map[p_id].shapes_data.write[p_shape_id].one_way_collision = p_one_way;
	emit_changed();
}

bool TileSet::tile_get_shape_one_way(int p_id, int p_shape_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), false);
	if (tile_map[p_id].shapes_data.size() > p_shape_id)
		return tile_map[p_id].shapes_data[p_shape_id].one_way_collision;

	return false;
}

void TileSet::tile_set_shape_one_way_margin(int p_id, int p_shape_id, float p_margin) {
	ERR_FAIL_COND(!tile_map.has(p_id));
	if (tile_map[p_id].shapes_data.size() <= p_shape_id)
		tile_map[p_id].shapes_data.resize(p_shape_id + 1);
	tile_map[p_id].shapes_data.write[p_shape_id].one_way_collision_margin = p_margin;
	emit_changed();
}

float TileSet::tile_get_shape_one_way_margin(int p_id, int p_shape_id) const {
	ERR_FAIL_COND_V(!tile_map.has(p_id), 0);
	if (tile_map[p_id].shapes_data.size() > p_shape_id)
		return tile_map[p_id].shapes_data[p_shape_id].one_way_collision_margin;

	return 0;
}

void TileSet::tile_set_light_occluder(int p_id, const Ref<OccluderPolygon2D> &p_light_occluder) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].occluder = p_light_occluder;
}

Ref<OccluderPolygon2D> TileSet::tile_get_light_occluder(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Ref<OccluderPolygon2D>());
	return tile_map[p_id].occluder;
}

void TileSet::autotile_set_light_occluder(int p_id, const Ref<OccluderPolygon2D> &p_light_occluder, const Vector2 &p_coord) {
	ERR_FAIL_COND(!tile_map.has(p_id));
	if (p_light_occluder.is_null()) {
		if (tile_map[p_id].autotile_data.occluder_map.has(p_coord)) {
			tile_map[p_id].autotile_data.occluder_map.erase(p_coord);
		}
	} else {
		tile_map[p_id].autotile_data.occluder_map[p_coord] = p_light_occluder;
	}
}

Ref<OccluderPolygon2D> TileSet::autotile_get_light_occluder(int p_id, const Vector2 &p_coord) const {
	ERR_FAIL_COND_V(!tile_map.has(p_id), Ref<OccluderPolygon2D>());
	if (!tile_map[p_id].autotile_data.occluder_map.has(p_coord)) {
		return Ref<OccluderPolygon2D>();
	} else {
		return tile_map[p_id].autotile_data.occluder_map[p_coord];
	}
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

const Map<Vector2, Ref<OccluderPolygon2D> > &TileSet::autotile_get_light_oclusion_map(int p_id) const {

	static Map<Vector2, Ref<OccluderPolygon2D> > dummy;
	ERR_FAIL_COND_V(!tile_map.has(p_id), dummy);
	return tile_map[p_id].autotile_data.occluder_map;
}

void TileSet::autotile_set_navigation_polygon(int p_id, const Ref<NavigationPolygon> &p_navigation_polygon, const Vector2 &p_coord) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	if (p_navigation_polygon.is_null()) {
		if (tile_map[p_id].autotile_data.navpoly_map.has(p_coord)) {
			tile_map[p_id].autotile_data.navpoly_map.erase(p_coord);
		}
	} else {
		tile_map[p_id].autotile_data.navpoly_map[p_coord] = p_navigation_polygon;
	}
}

Ref<NavigationPolygon> TileSet::autotile_get_navigation_polygon(int p_id, const Vector2 &p_coord) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), Ref<NavigationPolygon>());
	if (!tile_map[p_id].autotile_data.navpoly_map.has(p_coord)) {
		return Ref<NavigationPolygon>();
	} else {
		return tile_map[p_id].autotile_data.navpoly_map[p_coord];
	}
}

const Map<Vector2, Ref<NavigationPolygon> > &TileSet::autotile_get_navigation_map(int p_id) const {

	static Map<Vector2, Ref<NavigationPolygon> > dummy;
	ERR_FAIL_COND_V(!tile_map.has(p_id), dummy);
	return tile_map[p_id].autotile_data.navpoly_map;
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

int TileSet::tile_get_z_index(int p_id) const {

	ERR_FAIL_COND_V(!tile_map.has(p_id), 0);
	return tile_map[p_id].z_index;
}

void TileSet::tile_set_z_index(int p_id, int p_z_index) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	tile_map[p_id].z_index = p_z_index;
	emit_changed();
}

void TileSet::_tile_set_shapes(int p_id, const Array &p_shapes) {

	ERR_FAIL_COND(!tile_map.has(p_id));
	Vector<ShapeData> shapes_data;
	Transform2D default_transform = tile_get_shape_transform(p_id, 0);
	bool default_one_way = tile_get_shape_one_way(p_id, 0);
	Vector2 default_autotile_coord = Vector2();
	for (int i = 0; i < p_shapes.size(); i++) {
		ShapeData s = ShapeData();

		if (p_shapes[i].get_type() == Variant::OBJECT) {
			Ref<Shape2D> shape = p_shapes[i];
			if (shape.is_null()) continue;

			s.shape = shape;
			s.shape_transform = default_transform;
			s.one_way_collision = default_one_way;
			s.autotile_coord = default_autotile_coord;
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

			if (d.has("one_way_margin") && d["one_way_margin"].is_num())
				s.one_way_collision_margin = d["one_way_margin"];
			else
				s.one_way_collision = 1.0;

			if (d.has("autotile_coord") && d["autotile_coord"].get_type() == Variant::VECTOR2)
				s.autotile_coord = d["autotile_coord"];
			else
				s.autotile_coord = default_autotile_coord;

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
		shape_data["one_way_margin"] = data[i].one_way_collision_margin;
		shape_data["autotile_coord"] = data[i].autotile_coord;
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

bool TileSet::is_tile_bound(int p_drawn_id, int p_neighbor_id) {

	if (p_drawn_id == p_neighbor_id) {
		return true;
	} else if (get_script_instance() != NULL) {
		if (get_script_instance()->has_method("_is_tile_bound")) {
			Variant ret = get_script_instance()->call("_is_tile_bound", p_drawn_id, p_neighbor_id);
			if (ret.get_type() == Variant::BOOL) {
				return ret;
			}
		}
	}
	return false;
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
	ClassDB::bind_method(D_METHOD("autotile_clear_bitmask_map", "id"), &TileSet::autotile_clear_bitmask_map);
	ClassDB::bind_method(D_METHOD("autotile_set_icon_coordinate", "id", "coord"), &TileSet::autotile_set_icon_coordinate);
	ClassDB::bind_method(D_METHOD("autotile_get_icon_coordinate", "id"), &TileSet::autotile_get_icon_coordinate);
	ClassDB::bind_method(D_METHOD("autotile_set_subtile_priority", "id", "coord", "priority"), &TileSet::autotile_set_subtile_priority);
	ClassDB::bind_method(D_METHOD("autotile_get_subtile_priority", "id", "coord"), &TileSet::autotile_get_subtile_priority);
	ClassDB::bind_method(D_METHOD("autotile_set_z_index", "id", "coord", "z_index"), &TileSet::autotile_set_z_index);
	ClassDB::bind_method(D_METHOD("autotile_get_z_index", "id", "coord"), &TileSet::autotile_get_z_index);
	ClassDB::bind_method(D_METHOD("autotile_set_light_occluder", "id", "light_occluder", "coord"), &TileSet::autotile_set_light_occluder);
	ClassDB::bind_method(D_METHOD("autotile_get_light_occluder", "id", "coord"), &TileSet::autotile_get_light_occluder);
	ClassDB::bind_method(D_METHOD("autotile_set_navigation_polygon", "id", "navigation_polygon", "coord"), &TileSet::autotile_set_navigation_polygon);
	ClassDB::bind_method(D_METHOD("autotile_get_navigation_polygon", "id", "coord"), &TileSet::autotile_get_navigation_polygon);
	ClassDB::bind_method(D_METHOD("autotile_set_bitmask", "id", "bitmask", "flag"), &TileSet::autotile_set_bitmask);
	ClassDB::bind_method(D_METHOD("autotile_get_bitmask", "id", "coord"), &TileSet::autotile_get_bitmask);
	ClassDB::bind_method(D_METHOD("autotile_set_bitmask_mode", "id", "mode"), &TileSet::autotile_set_bitmask_mode);
	ClassDB::bind_method(D_METHOD("autotile_get_bitmask_mode", "id"), &TileSet::autotile_get_bitmask_mode);
	ClassDB::bind_method(D_METHOD("autotile_set_spacing", "id", "spacing"), &TileSet::autotile_set_spacing);
	ClassDB::bind_method(D_METHOD("autotile_get_spacing", "id"), &TileSet::autotile_get_spacing);
	ClassDB::bind_method(D_METHOD("autotile_set_size", "id", "size"), &TileSet::autotile_set_size);
	ClassDB::bind_method(D_METHOD("autotile_get_size", "id"), &TileSet::autotile_get_size);
	ClassDB::bind_method(D_METHOD("tile_set_name", "id", "name"), &TileSet::tile_set_name);
	ClassDB::bind_method(D_METHOD("tile_get_name", "id"), &TileSet::tile_get_name);
	ClassDB::bind_method(D_METHOD("tile_set_texture", "id", "texture"), &TileSet::tile_set_texture);
	ClassDB::bind_method(D_METHOD("tile_get_texture", "id"), &TileSet::tile_get_texture);
	ClassDB::bind_method(D_METHOD("tile_set_normal_map", "id", "normal_map"), &TileSet::tile_set_normal_map);
	ClassDB::bind_method(D_METHOD("tile_get_normal_map", "id"), &TileSet::tile_get_normal_map);
	ClassDB::bind_method(D_METHOD("tile_set_material", "id", "material"), &TileSet::tile_set_material);
	ClassDB::bind_method(D_METHOD("tile_get_material", "id"), &TileSet::tile_get_material);
	ClassDB::bind_method(D_METHOD("tile_set_modulate", "id", "color"), &TileSet::tile_set_modulate);
	ClassDB::bind_method(D_METHOD("tile_get_modulate", "id"), &TileSet::tile_get_modulate);
	ClassDB::bind_method(D_METHOD("tile_set_texture_offset", "id", "texture_offset"), &TileSet::tile_set_texture_offset);
	ClassDB::bind_method(D_METHOD("tile_get_texture_offset", "id"), &TileSet::tile_get_texture_offset);
	ClassDB::bind_method(D_METHOD("tile_set_region", "id", "region"), &TileSet::tile_set_region);
	ClassDB::bind_method(D_METHOD("tile_get_region", "id"), &TileSet::tile_get_region);
	ClassDB::bind_method(D_METHOD("tile_set_shape", "id", "shape_id", "shape"), &TileSet::tile_set_shape);
	ClassDB::bind_method(D_METHOD("tile_get_shape", "id", "shape_id"), &TileSet::tile_get_shape);
	ClassDB::bind_method(D_METHOD("tile_set_shape_offset", "id", "shape_id", "shape_offset"), &TileSet::tile_set_shape_offset);
	ClassDB::bind_method(D_METHOD("tile_get_shape_offset", "id", "shape_id"), &TileSet::tile_get_shape_offset);
	ClassDB::bind_method(D_METHOD("tile_set_shape_transform", "id", "shape_id", "shape_transform"), &TileSet::tile_set_shape_transform);
	ClassDB::bind_method(D_METHOD("tile_get_shape_transform", "id", "shape_id"), &TileSet::tile_get_shape_transform);
	ClassDB::bind_method(D_METHOD("tile_set_shape_one_way", "id", "shape_id", "one_way"), &TileSet::tile_set_shape_one_way);
	ClassDB::bind_method(D_METHOD("tile_get_shape_one_way", "id", "shape_id"), &TileSet::tile_get_shape_one_way);
	ClassDB::bind_method(D_METHOD("tile_set_shape_one_way_margin", "id", "shape_id", "one_way"), &TileSet::tile_set_shape_one_way_margin);
	ClassDB::bind_method(D_METHOD("tile_get_shape_one_way_margin", "id", "shape_id"), &TileSet::tile_get_shape_one_way_margin);
	ClassDB::bind_method(D_METHOD("tile_add_shape", "id", "shape", "shape_transform", "one_way", "autotile_coord"), &TileSet::tile_add_shape, DEFVAL(false), DEFVAL(Vector2()));
	ClassDB::bind_method(D_METHOD("tile_get_shape_count", "id"), &TileSet::tile_get_shape_count);
	ClassDB::bind_method(D_METHOD("tile_set_shapes", "id", "shapes"), &TileSet::_tile_set_shapes);
	ClassDB::bind_method(D_METHOD("tile_get_shapes", "id"), &TileSet::_tile_get_shapes);
	ClassDB::bind_method(D_METHOD("tile_set_tile_mode", "id", "tilemode"), &TileSet::tile_set_tile_mode);
	ClassDB::bind_method(D_METHOD("tile_get_tile_mode", "id"), &TileSet::tile_get_tile_mode);
	ClassDB::bind_method(D_METHOD("tile_set_navigation_polygon", "id", "navigation_polygon"), &TileSet::tile_set_navigation_polygon);
	ClassDB::bind_method(D_METHOD("tile_get_navigation_polygon", "id"), &TileSet::tile_get_navigation_polygon);
	ClassDB::bind_method(D_METHOD("tile_set_navigation_polygon_offset", "id", "navigation_polygon_offset"), &TileSet::tile_set_navigation_polygon_offset);
	ClassDB::bind_method(D_METHOD("tile_get_navigation_polygon_offset", "id"), &TileSet::tile_get_navigation_polygon_offset);
	ClassDB::bind_method(D_METHOD("tile_set_light_occluder", "id", "light_occluder"), &TileSet::tile_set_light_occluder);
	ClassDB::bind_method(D_METHOD("tile_get_light_occluder", "id"), &TileSet::tile_get_light_occluder);
	ClassDB::bind_method(D_METHOD("tile_set_occluder_offset", "id", "occluder_offset"), &TileSet::tile_set_occluder_offset);
	ClassDB::bind_method(D_METHOD("tile_get_occluder_offset", "id"), &TileSet::tile_get_occluder_offset);
	ClassDB::bind_method(D_METHOD("tile_set_z_index", "id", "z_index"), &TileSet::tile_set_z_index);
	ClassDB::bind_method(D_METHOD("tile_get_z_index", "id"), &TileSet::tile_get_z_index);

	ClassDB::bind_method(D_METHOD("remove_tile", "id"), &TileSet::remove_tile);
	ClassDB::bind_method(D_METHOD("clear"), &TileSet::clear);
	ClassDB::bind_method(D_METHOD("get_last_unused_tile_id"), &TileSet::get_last_unused_tile_id);
	ClassDB::bind_method(D_METHOD("find_tile_by_name", "name"), &TileSet::find_tile_by_name);
	ClassDB::bind_method(D_METHOD("get_tiles_ids"), &TileSet::_get_tiles_ids);

	BIND_VMETHOD(MethodInfo(Variant::BOOL, "_is_tile_bound", PropertyInfo(Variant::INT, "drawn_id"), PropertyInfo(Variant::INT, "neighbor_id")));
	BIND_VMETHOD(MethodInfo(Variant::VECTOR2, "_forward_subtile_selection", PropertyInfo(Variant::INT, "autotile_id"), PropertyInfo(Variant::INT, "bitmask"), PropertyInfo(Variant::OBJECT, "tilemap", PROPERTY_HINT_NONE, "TileMap"), PropertyInfo(Variant::VECTOR2, "tile_location")));

	BIND_ENUM_CONSTANT(BITMASK_2X2);
	BIND_ENUM_CONSTANT(BITMASK_3X3_MINIMAL);
	BIND_ENUM_CONSTANT(BITMASK_3X3);

	BIND_ENUM_CONSTANT(BIND_TOPLEFT);
	BIND_ENUM_CONSTANT(BIND_TOP);
	BIND_ENUM_CONSTANT(BIND_TOPRIGHT);
	BIND_ENUM_CONSTANT(BIND_LEFT);
	BIND_ENUM_CONSTANT(BIND_RIGHT);
	BIND_ENUM_CONSTANT(BIND_BOTTOMLEFT);
	BIND_ENUM_CONSTANT(BIND_BOTTOM);
	BIND_ENUM_CONSTANT(BIND_BOTTOMRIGHT);

	BIND_ENUM_CONSTANT(SINGLE_TILE);
	BIND_ENUM_CONSTANT(AUTO_TILE);
	BIND_ENUM_CONSTANT(ATLAS_TILE);
}

TileSet::TileSet() {
}
