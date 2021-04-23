/*************************************************************************/
/*  tile_set.cpp                                                         */
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

#include "tile_set.h"

#include "scene/gui/control.h"

#include "core/math/geometry_2d.h"

void TileData::set_tile_set(const TileSet *p_tile_set) {
	tile_set = p_tile_set;
	if (tile_set) {
		occluders.resize(tile_set->get_occlusion_layers_count());
		physics.resize(tile_set->get_physics_layers_count());
		navigation.resize(tile_set->get_navigation_layers_count());
		custom_data.resize(tile_set->get_custom_data_layers_count());
	}
	notify_property_list_changed();
}

void TileData::notify_tile_data_properties_should_change() {
	occluders.resize(tile_set->get_occlusion_layers_count());
	physics.resize(tile_set->get_physics_layers_count());
	for (int bit_index = 0; bit_index < 16; bit_index++) {
		if (terrain_set < 0 || terrain_peering_bits[bit_index] >= tile_set->get_terrains_count(terrain_set)) {
			terrain_peering_bits[bit_index] = -1;
		}
	}
	navigation.resize(tile_set->get_navigation_layers_count());

	// Convert custom data to the new type.
	custom_data.resize(tile_set->get_custom_data_layers_count());
	for (int i = 0; i < custom_data.size(); i++) {
		if (custom_data[i].get_type() != tile_set->get_custom_data_type(i)) {
			Variant new_val;
			Callable::CallError error;
			if (Variant::can_convert(custom_data[i].get_type(), tile_set->get_custom_data_type(i))) {
				const Variant *args[] = { &custom_data[i] };
				Variant::construct(tile_set->get_custom_data_type(i), new_val, args, 1, error);
			} else {
				Variant::construct(tile_set->get_custom_data_type(i), new_val, nullptr, 0, error);
			}
			custom_data.write[i] = new_val;
		}
	}

	notify_property_list_changed();
	emit_signal("changed");
}

void TileData::reset_state() {
	occluders.clear();
	physics.clear();
	navigation.clear();
	custom_data.clear();
}

void TileData::set_allow_transform(bool p_allow_transform) {
	allow_transform = p_allow_transform;
}

bool TileData::is_allowing_transform() const {
	return allow_transform;
}

// Rendering
void TileData::set_flip_h(bool p_flip_h) {
	ERR_FAIL_COND_MSG(!allow_transform && p_flip_h, "Transform is only allowed for alternative tiles (with its alternative_id != 0)");
	flip_h = p_flip_h;
	emit_signal("changed");
}
bool TileData::get_flip_h() const {
	return flip_h;
}

void TileData::set_flip_v(bool p_flip_v) {
	ERR_FAIL_COND_MSG(!allow_transform && p_flip_v, "Transform is only allowed for alternative tiles (with its alternative_id != 0)");
	flip_v = p_flip_v;
	emit_signal("changed");
}

bool TileData::get_flip_v() const {
	return flip_v;
}

void TileData::set_transpose(bool p_transpose) {
	ERR_FAIL_COND_MSG(!allow_transform && p_transpose, "Transform is only allowed for alternative tiles (with its alternative_id != 0)");
	transpose = p_transpose;
	emit_signal("changed");
}
bool TileData::get_transpose() const {
	return transpose;
}

void TileData::set_texture_offset(Vector2i p_texture_offset) {
	tex_offset = p_texture_offset;
	emit_signal("changed");
}

Vector2i TileData::get_texture_offset() const {
	return tex_offset;
}

void TileData::tile_set_material(Ref<ShaderMaterial> p_material) {
	material = p_material;
	emit_signal("changed");
}
Ref<ShaderMaterial> TileData::tile_get_material() const {
	return material;
}

void TileData::set_modulate(Color p_modulate) {
	modulate = p_modulate;
	emit_signal("changed");
}
Color TileData::get_modulate() const {
	return modulate;
}

void TileData::set_z_index(int p_z_index) {
	z_index = p_z_index;
	emit_signal("changed");
}
int TileData::get_z_index() const {
	return z_index;
}

void TileData::set_y_sort_origin(Vector2i p_y_sort_origin) {
	y_sort_origin = p_y_sort_origin;
	emit_signal("changed");
}
Vector2i TileData::get_y_sort_origin() const {
	return y_sort_origin;
}

void TileData::set_occluder(int p_layer_id, Ref<OccluderPolygon2D> p_occluder_polygon) {
	ERR_FAIL_INDEX(p_layer_id, occluders.size());
	occluders.write[p_layer_id] = p_occluder_polygon;
	emit_signal("changed");
}

Ref<OccluderPolygon2D> TileData::get_occluder(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, occluders.size(), Ref<OccluderPolygon2D>());
	return occluders[p_layer_id];
}

// Physics
int TileData::get_collision_shapes_count(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, physics.size(), 0);
	return physics[p_layer_id].shapes.size();
}

void TileData::set_collision_shapes_count(int p_layer_id, int p_shapes_count) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	ERR_FAIL_COND(p_shapes_count < 0);
	physics.write[p_layer_id].shapes.resize(p_shapes_count);
	notify_property_list_changed();
	emit_signal("changed");
}

void TileData::add_collision_shape(int p_layer_id) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	physics.write[p_layer_id].shapes.push_back(PhysicsLayerTileData::ShapeTileData());
	emit_signal("changed");
}

void TileData::remove_collision_shape(int p_layer_id, int p_shape_index) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	ERR_FAIL_INDEX(p_shape_index, physics[p_layer_id].shapes.size());
	physics.write[p_layer_id].shapes.remove(p_shape_index);
	emit_signal("changed");
}

void TileData::set_collision_shape_shape(int p_layer_id, int p_shape_index, Ref<Shape2D> p_shape) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	ERR_FAIL_INDEX(p_shape_index, physics[p_layer_id].shapes.size());
	physics.write[p_layer_id].shapes.write[p_shape_index].shape = p_shape;
	emit_signal("changed");
}

Ref<Shape2D> TileData::get_collision_shape_shape(int p_layer_id, int p_shape_index) const {
	ERR_FAIL_INDEX_V(p_layer_id, physics.size(), Ref<Shape2D>());
	ERR_FAIL_INDEX_V(p_shape_index, physics[p_layer_id].shapes.size(), Ref<Shape2D>());
	return physics[p_layer_id].shapes[p_shape_index].shape;
}

void TileData::set_collision_shape_one_way(int p_layer_id, int p_shape_index, bool p_one_way) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	ERR_FAIL_INDEX(p_shape_index, physics[p_layer_id].shapes.size());
	physics.write[p_layer_id].shapes.write[p_shape_index].one_way = p_one_way;
	emit_signal("changed");
}

bool TileData::is_collision_shape_one_way(int p_layer_id, int p_shape_index) const {
	ERR_FAIL_INDEX_V(p_layer_id, physics.size(), false);
	ERR_FAIL_INDEX_V(p_shape_index, physics[p_layer_id].shapes.size(), false);
	return physics[p_layer_id].shapes[p_shape_index].one_way;
}

void TileData::set_collision_shape_one_way_margin(int p_layer_id, int p_shape_index, float p_one_way_margin) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	ERR_FAIL_INDEX(p_shape_index, physics[p_layer_id].shapes.size());
	physics.write[p_layer_id].shapes.write[p_shape_index].one_way_margin = p_one_way_margin;
	emit_signal("changed");
}

float TileData::get_collision_shape_one_way_margin(int p_layer_id, int p_shape_index) const {
	ERR_FAIL_INDEX_V(p_layer_id, physics.size(), 0.0);
	ERR_FAIL_INDEX_V(p_shape_index, physics[p_layer_id].shapes.size(), 0.0);
	return physics[p_layer_id].shapes[p_shape_index].one_way_margin;
}

// Terrain
void TileData::set_terrain_set(int p_terrain_set) {
	ERR_FAIL_COND(p_terrain_set < -1);
	if (tile_set) {
		ERR_FAIL_COND(p_terrain_set >= tile_set->get_terrain_sets_count());
	}
	terrain_set = p_terrain_set;
	notify_property_list_changed();
	emit_signal("changed");
}

int TileData::get_terrain_set() const {
	return terrain_set;
}

void TileData::set_peering_bit_terrain(TileSet::CellNeighbor p_peering_bit, int p_terrain_index) {
	ERR_FAIL_COND(p_terrain_index < -1);
	if (tile_set) {
		ERR_FAIL_COND(p_terrain_index >= tile_set->get_terrains_count(terrain_set));
		ERR_FAIL_COND(!is_valid_peering_bit_terrain(p_peering_bit));
	}
	terrain_peering_bits[p_peering_bit] = p_terrain_index;
	emit_signal("changed");
}

int TileData::get_peering_bit_terrain(TileSet::CellNeighbor p_peering_bit) const {
	return terrain_peering_bits[p_peering_bit];
}

bool TileData::is_valid_peering_bit_terrain(TileSet::CellNeighbor p_peering_bit) const {
	ERR_FAIL_COND_V(!tile_set, false);

	return tile_set->is_valid_peering_bit_terrain(terrain_set, p_peering_bit);
}

// Navigation
void TileData::set_navigation_polygon(int p_layer_id, Ref<NavigationPolygon> p_navigation_polygon) {
	ERR_FAIL_INDEX(p_layer_id, navigation.size());
	navigation.write[p_layer_id] = p_navigation_polygon;
	emit_signal("changed");
}

Ref<NavigationPolygon> TileData::get_navigation_polygon(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, navigation.size(), Ref<NavigationPolygon>());
	return navigation[p_layer_id];
}

// Misc
void TileData::set_probability(float p_probability) {
	ERR_FAIL_COND(p_probability <= 0.0);
	probability = p_probability;
	emit_signal("changed");
}
float TileData::get_probability() const {
	return probability;
}

// Custom data
void TileData::set_custom_data(String p_layer_name, Variant p_value) {
	ERR_FAIL_COND(!tile_set);
	int p_layer_id = tile_set->get_custom_data_layer_by_name(p_layer_name);
	ERR_FAIL_COND_MSG(p_layer_id < 0, vformat("TileSet has no layer with name: %s", p_layer_name));
	set_custom_data_by_layer_id(p_layer_id, p_value);
}

Variant TileData::get_custom_data(String p_layer_name) const {
	ERR_FAIL_COND_V(!tile_set, Variant());
	int p_layer_id = tile_set->get_custom_data_layer_by_name(p_layer_name);
	ERR_FAIL_COND_V_MSG(p_layer_id < 0, Variant(), vformat("TileSet has no layer with name: %s", p_layer_name));
	return get_custom_data_by_layer_id(p_layer_id);
}

void TileData::set_custom_data_by_layer_id(int p_layer_id, Variant p_value) {
	ERR_FAIL_INDEX(p_layer_id, custom_data.size());
	custom_data.write[p_layer_id] = p_value;
	emit_signal("changed");
}

Variant TileData::get_custom_data_by_layer_id(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, custom_data.size(), Variant());
	return custom_data[p_layer_id];
}

bool TileData::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);

	if (components.size() == 2 && components[0].begins_with("occlusion_layer_") && components[0].trim_prefix("occlusion_layer_").is_valid_integer()) {
		// Occlusion layers.
		int layer_index = components[0].trim_prefix("occlusion_layer_").to_int();
		ERR_FAIL_COND_V(layer_index < 0, false);
		if (components[1] == "polygon") {
			Ref<OccluderPolygon2D> polygon = p_value;
			if (!polygon.is_valid()) {
				return false;
			}

			if (layer_index >= occluders.size()) {
				if (tile_set) {
					return false;
				} else {
					occluders.resize(layer_index + 1);
				}
			}
			set_occluder(layer_index, polygon);
			return true;
		}
	} else if (components.size() >= 2 && components[0].begins_with("physics_layer_") && components[0].trim_prefix("physics_layer_").is_valid_integer()) {
		// Physics layers.
		int layer_index = components[0].trim_prefix("physics_layer_").to_int();
		ERR_FAIL_COND_V(layer_index < 0, false);
		if (components.size() == 2 && components[1] == "shapes_count") {
			if (p_value.get_type() != Variant::INT) {
				return false;
			}

			if (layer_index >= physics.size()) {
				if (tile_set) {
					return false;
				} else {
					physics.resize(layer_index + 1);
				}
			}
			set_collision_shapes_count(layer_index, p_value);
			return true;
		} else if (components.size() == 3 && components[1].begins_with("shape_") && components[1].trim_prefix("shape_").is_valid_integer()) {
			int shape_index = components[1].trim_prefix("shape_").to_int();
			ERR_FAIL_COND_V(shape_index < 0, false);

			if (components[2] == "shape" || components[2] == "one_way" || components[2] == "one_way_margin") {
				if (layer_index >= physics.size()) {
					if (tile_set) {
						return false;
					} else {
						physics.resize(layer_index + 1);
					}
				}

				if (shape_index >= physics[layer_index].shapes.size()) {
					physics.write[layer_index].shapes.resize(shape_index + 1);
				}
			}
			if (components[2] == "shape") {
				Ref<Shape2D> shape = p_value;
				set_collision_shape_shape(layer_index, shape_index, shape);
				return true;
			} else if (components[2] == "one_way") {
				set_collision_shape_one_way(layer_index, shape_index, p_value);
				return true;
			} else if (components[2] == "one_way_margin") {
				set_collision_shape_one_way_margin(layer_index, shape_index, p_value);
				return true;
			}
		}
	} else if (components.size() == 2 && components[0].begins_with("navigation_layer_") && components[0].trim_prefix("navigation_layer_").is_valid_integer()) {
		// Navigation layers.
		int layer_index = components[0].trim_prefix("navigation_layer_").to_int();
		ERR_FAIL_COND_V(layer_index < 0, false);
		if (components[1] == "polygon") {
			Ref<NavigationPolygon> polygon = p_value;
			if (!polygon.is_valid()) {
				return false;
			}

			if (layer_index >= navigation.size()) {
				if (tile_set) {
					return false;
				} else {
					navigation.resize(layer_index + 1);
				}
			}
			set_navigation_polygon(layer_index, polygon);
			return true;
		}
	} else if (components.size() == 2 && components[0] == "terrains_peering_bit") {
		// Terrains.
		if (components[1] == "right_side") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_RIGHT_SIDE, p_value);
		} else if (components[1] == "right_corner") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_RIGHT_CORNER, p_value);
		} else if (components[1] == "bottom_right_side") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, p_value);
		} else if (components[1] == "bottom_right_corner") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER, p_value);
		} else if (components[1] == "bottom_side") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, p_value);
		} else if (components[1] == "bottom_corner") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, p_value);
		} else if (components[1] == "bottom_left_side") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, p_value);
		} else if (components[1] == "bottom_left_corner") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER, p_value);
		} else if (components[1] == "left_side") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_LEFT_SIDE, p_value);
		} else if (components[1] == "left_corner") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_LEFT_CORNER, p_value);
		} else if (components[1] == "top_left_side") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, p_value);
		} else if (components[1] == "top_left_corner") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER, p_value);
		} else if (components[1] == "top_side") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_SIDE, p_value);
		} else if (components[1] == "top_corner") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_CORNER, p_value);
		} else if (components[1] == "top_right_side") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, p_value);
		} else if (components[1] == "top_right_corner") {
			set_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER, p_value);
		} else {
			return false;
		}
		return true;
	} else if (components.size() == 1 && components[0].begins_with("custom_data_") && components[0].trim_prefix("custom_data_").is_valid_integer()) {
		// Custom data layers.
		int layer_index = components[0].trim_prefix("custom_data_").to_int();
		ERR_FAIL_COND_V(layer_index < 0, false);

		if (layer_index >= custom_data.size()) {
			if (tile_set) {
				return false;
			} else {
				custom_data.resize(layer_index + 1);
			}
		}
		set_custom_data_by_layer_id(layer_index, p_value);

		return true;
	}

	return false;
}

bool TileData::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> components = String(p_name).split("/", true, 2);

	if (tile_set) {
		if (components.size() == 2 && components[0].begins_with("occlusion_layer") && components[0].trim_prefix("occlusion_layer_").is_valid_integer()) {
			// Occlusion layers.
			int layer_index = components[0].trim_prefix("occlusion_layer_").to_int();
			ERR_FAIL_COND_V(layer_index < 0, false);
			if (layer_index >= occluders.size()) {
				return false;
			}
			if (components[1] == "polygon") {
				r_ret = get_occluder(layer_index);
				return true;
			}
		} else if (components.size() >= 2 && components[0].begins_with("physics_layer_") && components[0].trim_prefix("physics_layer_").is_valid_integer()) {
			// Physics layers.
			int layer_index = components[0].trim_prefix("physics_layer_").to_int();
			ERR_FAIL_COND_V(layer_index < 0, false);
			if (layer_index >= physics.size()) {
				return false;
			}
			if (components.size() == 2 && components[1] == "shapes_count") {
				r_ret = get_collision_shapes_count(layer_index);
				return true;
			} else if (components.size() == 3 && components[1].begins_with("shape_") && components[1].trim_prefix("shape_").is_valid_integer()) {
				int shape_index = components[1].trim_prefix("shape_").to_int();
				ERR_FAIL_COND_V(shape_index < 0, false);
				if (shape_index >= physics[layer_index].shapes.size()) {
					return false;
				}
				if (components[2] == "shape") {
					r_ret = get_collision_shape_shape(layer_index, shape_index);
					return true;
				} else if (components[2] == "one_way") {
					r_ret = is_collision_shape_one_way(layer_index, shape_index);
					return true;
				} else if (components[2] == "one_way_margin") {
					r_ret = get_collision_shape_one_way_margin(layer_index, shape_index);
					return true;
				}
			}
		} else if (components.size() == 2 && components[0] == "terrains_peering_bit") {
			// Terrains.
			if (components[1] == "right_side") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_RIGHT_SIDE];
			} else if (components[1] == "right_corner") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_RIGHT_CORNER];
			} else if (components[1] == "bottom_right_side") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE];
			} else if (components[1] == "bottom_right_corner") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER];
			} else if (components[1] == "bottom_side") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_BOTTOM_SIDE];
			} else if (components[1] == "bottom_corner") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_BOTTOM_CORNER];
			} else if (components[1] == "bottom_left_side") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE];
			} else if (components[1] == "bottom_left_corner") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER];
			} else if (components[1] == "left_side") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_LEFT_SIDE];
			} else if (components[1] == "left_corner") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_LEFT_CORNER];
			} else if (components[1] == "top_left_side") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE];
			} else if (components[1] == "top_left_corner") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER];
			} else if (components[1] == "top_side") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_TOP_SIDE];
			} else if (components[1] == "top_corner") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_TOP_CORNER];
			} else if (components[1] == "top_right_side") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE];
			} else if (components[1] == "top_right_corner") {
				r_ret = terrain_peering_bits[TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER];
			} else {
				return false;
			}
			return true;
		} else if (components.size() == 2 && components[0].begins_with("navigation_layer_") && components[0].trim_prefix("navigation_layer_").is_valid_integer()) {
			// Occlusion layers.
			int layer_index = components[0].trim_prefix("navigation_layer_").to_int();
			ERR_FAIL_COND_V(layer_index < 0, false);
			if (layer_index >= navigation.size()) {
				return false;
			}
			if (components[1] == "polygon") {
				r_ret = get_navigation_polygon(layer_index);
				return true;
			}
		} else if (components.size() == 1 && components[0].begins_with("custom_data_") && components[0].trim_prefix("custom_data_").is_valid_integer()) {
			// Custom data layers.
			int layer_index = components[0].trim_prefix("custom_data_").to_int();
			ERR_FAIL_COND_V(layer_index < 0, false);
			if (layer_index >= custom_data.size()) {
				return false;
			}
			r_ret = get_custom_data_by_layer_id(layer_index);
			return true;
		}
	}

	return false;
}

void TileData::_get_property_list(List<PropertyInfo> *p_list) const {
	// Add the groups manually.
	if (tile_set) {
		// Occlusion layers.
		p_list->push_back(PropertyInfo(Variant::NIL, "Rendering", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
		for (int i = 0; i < occluders.size(); i++) {
			p_list->push_back(PropertyInfo(Variant::OBJECT, vformat("occlusion_layer_%d/polygon", i), PROPERTY_HINT_RESOURCE_TYPE, "OccluderPolygon2D", PROPERTY_USAGE_DEFAULT));
		}

		// Physics layers.
		p_list->push_back(PropertyInfo(Variant::NIL, "Physics", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
		for (int i = 0; i < physics.size(); i++) {
			p_list->push_back(PropertyInfo(Variant::INT, vformat("physics_layer_%d/shapes_count", i), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
			for (int j = 0; j < physics[i].shapes.size(); j++) {
				p_list->push_back(PropertyInfo(Variant::OBJECT, vformat("physics_layer_%d/shape_%d/shape", i, j), PROPERTY_HINT_RESOURCE_TYPE, "Shape2D", PROPERTY_USAGE_DEFAULT));
				p_list->push_back(PropertyInfo(Variant::BOOL, vformat("physics_layer_%d/shape_%d/one_way", i, j)));
				p_list->push_back(PropertyInfo(Variant::FLOAT, vformat("physics_layer_%d/shape_%d/one_way_margin", i, j)));
			}
		}

		// Terrain data
		if (terrain_set >= 0) {
			p_list->push_back(PropertyInfo(Variant::NIL, "Terrains", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/right_side"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_RIGHT_CORNER)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/right_corner"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/bottom_right_side"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/bottom_right_corner"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/bottom_side"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_CORNER)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/bottom_corner"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/bottom_left_side"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/bottom_left_corner"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/left_side"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_LEFT_CORNER)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/left_corner"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/top_left_side"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/top_left_corner"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/top_side"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_CORNER)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/top_corner"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/top_right_side"));
			}
			if (is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER)) {
				p_list->push_back(PropertyInfo(Variant::INT, "terrains_peering_bit/top_right_corner"));
			}
		}

		// Navigation layers.
		p_list->push_back(PropertyInfo(Variant::NIL, "Navigation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
		for (int i = 0; i < navigation.size(); i++) {
			p_list->push_back(PropertyInfo(Variant::OBJECT, vformat("navigation_layer_%d/polygon", i), PROPERTY_HINT_RESOURCE_TYPE, "NavigationPolygon", PROPERTY_USAGE_DEFAULT));
		}

		// Custom data layers.
		p_list->push_back(PropertyInfo(Variant::NIL, "Custom data", PROPERTY_HINT_NONE, "custom_data_", PROPERTY_USAGE_GROUP));
		for (int i = 0; i < custom_data.size(); i++) {
			p_list->push_back(PropertyInfo(tile_set->get_custom_data_type(i), vformat("custom_data_%d", i), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT));
		}
	}
}

void TileData::_bind_methods() {
	// Rendering.
	ClassDB::bind_method(D_METHOD("set_flip_h", "flip_h"), &TileData::set_flip_h);
	ClassDB::bind_method(D_METHOD("get_flip_h"), &TileData::get_flip_h);
	ClassDB::bind_method(D_METHOD("set_flip_v", "flip_v"), &TileData::set_flip_v);
	ClassDB::bind_method(D_METHOD("get_flip_v"), &TileData::get_flip_v);
	ClassDB::bind_method(D_METHOD("set_transpose", "transpose"), &TileData::set_transpose);
	ClassDB::bind_method(D_METHOD("get_transpose"), &TileData::get_transpose);
	ClassDB::bind_method(D_METHOD("tile_set_material", "material"), &TileData::tile_set_material);
	ClassDB::bind_method(D_METHOD("tile_get_material"), &TileData::tile_get_material);
	ClassDB::bind_method(D_METHOD("set_texture_offset", "texture_offset"), &TileData::set_texture_offset);
	ClassDB::bind_method(D_METHOD("get_texture_offset"), &TileData::get_texture_offset);
	ClassDB::bind_method(D_METHOD("set_modulate", "modulate"), &TileData::set_modulate);
	ClassDB::bind_method(D_METHOD("get_modulate"), &TileData::get_modulate);
	ClassDB::bind_method(D_METHOD("set_z_index", "z_index"), &TileData::set_z_index);
	ClassDB::bind_method(D_METHOD("get_z_index"), &TileData::get_z_index);
	ClassDB::bind_method(D_METHOD("set_y_sort_origin", "y_sort_origin"), &TileData::set_y_sort_origin);
	ClassDB::bind_method(D_METHOD("get_y_sort_origin"), &TileData::get_y_sort_origin);

	ClassDB::bind_method(D_METHOD("set_occluder", "layer_id", "occluder_polygon"), &TileData::set_occluder);
	ClassDB::bind_method(D_METHOD("get_occluder", "layer_id"), &TileData::get_occluder);

	// Physics.
	ClassDB::bind_method(D_METHOD("get_collision_shapes_count", "layer_id"), &TileData::get_collision_shapes_count);
	ClassDB::bind_method(D_METHOD("set_collision_shapes_count", "layer_id", "shapes_count"), &TileData::set_collision_shapes_count);
	ClassDB::bind_method(D_METHOD("add_collision_shape", "layer_id"), &TileData::add_collision_shape);
	ClassDB::bind_method(D_METHOD("remove_collision_shape", "layer_id", "shape_index"), &TileData::remove_collision_shape);
	ClassDB::bind_method(D_METHOD("set_collision_shape_shape", "layer_id", "shape_index", "shape"), &TileData::set_collision_shape_shape);
	ClassDB::bind_method(D_METHOD("get_collision_shape_shape", "layer_id", "shape_index"), &TileData::get_collision_shape_shape);
	ClassDB::bind_method(D_METHOD("set_collision_shape_one_way", "layer_id", "shape_index", "one_way"), &TileData::set_collision_shape_one_way);
	ClassDB::bind_method(D_METHOD("is_collision_shape_one_way", "layer_id", "shape_index"), &TileData::is_collision_shape_one_way);
	ClassDB::bind_method(D_METHOD("set_collision_shape_one_way_margin", "layer_id", "shape_index", "one_way_margin"), &TileData::set_collision_shape_one_way_margin);
	ClassDB::bind_method(D_METHOD("get_collision_shape_one_way_margin", "layer_id", "shape_index"), &TileData::get_collision_shape_one_way_margin);

	// Terrain
	ClassDB::bind_method(D_METHOD("set_terrain_set", "terrain_set"), &TileData::set_terrain_set);
	ClassDB::bind_method(D_METHOD("get_terrain_set"), &TileData::get_terrain_set);
	ClassDB::bind_method(D_METHOD("set_peering_bit_terrain", "peering_bit", "terrain"), &TileData::set_peering_bit_terrain);
	ClassDB::bind_method(D_METHOD("get_peering_bit_terrain", "peering_bit"), &TileData::get_peering_bit_terrain);

	// Navigation
	ClassDB::bind_method(D_METHOD("set_navigation_polygon", "layer_id", "navigation_polygon"), &TileData::set_navigation_polygon);
	ClassDB::bind_method(D_METHOD("get_navigation_polygon", "layer_id"), &TileData::get_navigation_polygon);

	// Misc.
	ClassDB::bind_method(D_METHOD("set_probability", "probability"), &TileData::set_probability);
	ClassDB::bind_method(D_METHOD("get_probability"), &TileData::get_probability);

	// Custom data.
	ClassDB::bind_method(D_METHOD("set_custom_data", "layer_name", "value"), &TileData::set_custom_data);
	ClassDB::bind_method(D_METHOD("get_custom_data", "layer_name"), &TileData::get_custom_data);
	ClassDB::bind_method(D_METHOD("set_custom_data_by_layer_id", "layer_id", "value"), &TileData::set_custom_data_by_layer_id);
	ClassDB::bind_method(D_METHOD("get_custom_data_by_layer_id", "layer_id"), &TileData::get_custom_data_by_layer_id);

	ADD_GROUP("Rendering", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_h"), "set_flip_h", "get_flip_h");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_v"), "set_flip_v", "get_flip_v");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "transpose"), "set_transpose", "get_transpose");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "texture_offset"), "set_texture_offset", "get_texture_offset");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "modulate"), "set_modulate", "get_modulate");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "z_index"), "set_z_index", "get_z_index");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "y_sort_origin"), "set_y_sort_origin", "get_y_sort_origin");

	ADD_GROUP("Terrains", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "terrain_set"), "set_terrain_set", "get_terrain_set");

	ADD_GROUP("Miscellaneous", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "probability"), "set_probability", "get_probability");

	ADD_SIGNAL(MethodInfo("changed"));
}

void TileSetSource::set_tile_set(const TileSet *p_tile_set) {
	tile_set = p_tile_set;
}

void TileSetAtlasSource::set_tile_set(const TileSet *p_tile_set) {
	tile_set = p_tile_set;

	// Set the TileSet on all TileData.
	for (Map<Vector2i, TileAlternativesData>::Element *E_tile = tiles.front(); E_tile; E_tile = E_tile->next()) {
		for (Map<int, TileData *>::Element *E_alternative = E_tile->get().alternatives.front(); E_alternative; E_alternative = E_alternative->next()) {
			E_alternative->get()->set_tile_set(tile_set);
		}
	}
}

void TileSetAtlasSource::notify_tile_data_properties_should_change() {
	// Set the TileSet on all TileData.
	for (Map<Vector2i, TileAlternativesData>::Element *E_tile = tiles.front(); E_tile; E_tile = E_tile->next()) {
		for (Map<int, TileData *>::Element *E_alternative = E_tile->get().alternatives.front(); E_alternative; E_alternative = E_alternative->next()) {
			E_alternative->get()->notify_tile_data_properties_should_change();
		}
	}
}

void TileSetAtlasSource::reset_state() {
	// Reset all TileData.
	for (Map<Vector2i, TileAlternativesData>::Element *E_tile = tiles.front(); E_tile; E_tile = E_tile->next()) {
		for (Map<int, TileData *>::Element *E_alternative = E_tile->get().alternatives.front(); E_alternative; E_alternative = E_alternative->next()) {
			E_alternative->get()->reset_state();
		}
	}
}

void TileSetAtlasSource::set_texture(Ref<Texture2D> p_texture) {
	texture = p_texture;

	emit_changed();
}

Ref<Texture2D> TileSetAtlasSource::get_texture() const {
	return texture;
}

void TileSetAtlasSource::set_margins(Vector2i p_margins) {
	if (p_margins.x < 0 || p_margins.y < 0) {
		WARN_PRINT("Atlas source margins should be positive.");
		margins = Vector2i(MAX(0, p_margins.x), MAX(0, p_margins.y));
	} else {
		margins = p_margins;
	}

	emit_changed();
}
Vector2i TileSetAtlasSource::get_margins() const {
	return margins;
}

void TileSetAtlasSource::set_separation(Vector2i p_separation) {
	if (p_separation.x < 0 || p_separation.y < 0) {
		WARN_PRINT("Atlas source separation should be positive.");
		separation = Vector2i(MAX(0, p_separation.x), MAX(0, p_separation.y));
	} else {
		separation = p_separation;
	}

	emit_changed();
}
Vector2i TileSetAtlasSource::get_separation() const {
	return separation;
}

void TileSetAtlasSource::set_texture_region_size(Vector2i p_tile_size) {
	if (p_tile_size.x <= 0 || p_tile_size.y <= 0) {
		WARN_PRINT("Atlas source tile_size should be strictly positive.");
		texture_region_size = Vector2i(MAX(1, p_tile_size.x), MAX(1, p_tile_size.y));
	} else {
		texture_region_size = p_tile_size;
	}

	emit_changed();
}
Vector2i TileSetAtlasSource::get_texture_region_size() const {
	return texture_region_size;
}

void TileSetAtlasSource::set_base_texture_offset(Vector2i p_base_texture_offset) {
	base_texture_offset = p_base_texture_offset;

	emit_changed();
}
Vector2i TileSetAtlasSource::get_base_texture_offset() const {
	return base_texture_offset;
}

Vector2i TileSetAtlasSource::get_atlas_grid_size() const {
	Ref<Texture2D> texture = get_texture();
	if (!texture.is_valid()) {
		return Vector2i();
	}

	ERR_FAIL_COND_V(texture_region_size.x <= 0 || texture_region_size.y <= 0, Vector2i());

	Size2i valid_area = texture->get_size() - margins;

	// Compute the number of valid tiles in the tiles atlas
	Size2i grid_size = Size2i();
	if (valid_area.x >= texture_region_size.x && valid_area.y >= texture_region_size.y) {
		valid_area -= texture_region_size;
		grid_size = Size2i(1, 1) + valid_area / (texture_region_size + separation);
	}
	return grid_size;
}

bool TileSetAtlasSource::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);

	// Compute the vector2i if we have coordinates.
	Vector<String> coords_split = components[0].split(":");
	Vector2i coords = TileSetAtlasSource::INVALID_ATLAS_COORDS;
	if (coords_split.size() == 2 && coords_split[0].is_valid_integer() && coords_split[1].is_valid_integer()) {
		coords = Vector2i(coords_split[0].to_int(), coords_split[1].to_int());
	}

	// Properties.
	if (coords != TileSetAtlasSource::INVALID_ATLAS_COORDS) {
		// Create the tile if needed.
		if (!has_tile(coords)) {
			create_tile(coords);
		}
		if (components.size() >= 2) {
			// Properties.
			if (components[1] == "size_in_atlas") {
				move_tile_in_atlas(coords, coords, p_value);
			} else if (components[1] == "next_alternative_id") {
				tiles[coords].next_alternative_id = p_value;
			} else if (components[1].is_valid_integer()) {
				int alternative_id = components[1].to_int();
				if (alternative_id != TileSetAtlasSource::INVALID_TILE_ALTERNATIVE) {
					// Create the alternative if needed ?
					if (!has_alternative_tile(coords, alternative_id)) {
						create_alternative_tile(coords, alternative_id);
					}
					if (!tiles[coords].alternatives.has(alternative_id)) {
						tiles[coords].alternatives[alternative_id] = memnew(TileData);
						tiles[coords].alternatives[alternative_id]->set_tile_set(tile_set);
						tiles[coords].alternatives[alternative_id]->set_allow_transform(alternative_id > 0);
						tiles[coords].alternatives_ids.append(alternative_id);
					}
					if (components.size() >= 3) {
						bool valid;
						tiles[coords].alternatives[alternative_id]->set(components[2], p_value, &valid);
						return valid;
					} else {
						// Only create the alternative if it did not exist yet.
						return true;
					}
				}
			}
		}
	}

	return false;
}

bool TileSetAtlasSource::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> components = String(p_name).split("/", true, 2);

	// Properties.
	Vector<String> coords_split = components[0].split(":");
	if (coords_split.size() == 2 && coords_split[0].is_valid_integer() && coords_split[1].is_valid_integer()) {
		Vector2i coords = Vector2i(coords_split[0].to_int(), coords_split[1].to_int());
		if (tiles.has(coords)) {
			if (components.size() >= 2) {
				// Properties.
				if (components[1] == "size_in_atlas") {
					r_ret = tiles[coords].size_in_atlas;
					return true;
				} else if (components[1] == "next_alternative_id") {
					r_ret = tiles[coords].next_alternative_id;
					return true;
				} else if (components[1].is_valid_integer()) {
					int alternative_id = components[1].to_int();
					if (alternative_id != TileSetAtlasSource::INVALID_TILE_ALTERNATIVE && tiles[coords].alternatives.has(alternative_id)) {
						if (components.size() >= 3) {
							bool valid;
							r_ret = tiles[coords].alternatives[alternative_id]->get(components[2], &valid);
							return valid;
						} else {
							// Only to notify the tile alternative exists.
							r_ret = alternative_id;
							return true;
						}
					}
				}
			}
		}
	}

	return false;
}

void TileSetAtlasSource::_get_property_list(List<PropertyInfo> *p_list) const {
	// Atlases data.
	for (Map<Vector2i, TileAlternativesData>::Element *E_tile = tiles.front(); E_tile; E_tile = E_tile->next()) {
		List<PropertyInfo> tile_property_list;
		if (E_tile->get().size_in_atlas != Vector2i(1, 1)) {
			tile_property_list.push_back(PropertyInfo(Variant::VECTOR2I, "size_in_atlas", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		}
		if (E_tile->get().next_alternative_id != 1) {
			tile_property_list.push_back(PropertyInfo(Variant::INT, "next_alternative_id", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		}

		// Atlases data.
		for (Map<int, TileData *>::Element *E_alternative = E_tile->get().alternatives.front(); E_alternative; E_alternative = E_alternative->next()) {
			// Add a dummy property to show the tile exists.
			tile_property_list.push_back(PropertyInfo(Variant::INT, vformat("%d", E_alternative->key()), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));

			List<PropertyInfo> alternative_property_list;
			E_alternative->get()->get_property_list(&alternative_property_list);
			TileSet::_append_property_list_with_prefix(vformat("%d", E_alternative->key()), &alternative_property_list, &tile_property_list);
		}

		TileSet::_append_property_list_with_prefix(vformat("%d:%d", E_tile->key().x, E_tile->key().y), &tile_property_list, p_list);
	}
}

// --- TileSetAtlasSource ---
void TileSetAtlasSource::create_tile(const Vector2i p_atlas_coords, const Vector2i p_size) {
	// Create a tile if it does not exists.
	ERR_FAIL_COND(p_atlas_coords.x < 0 || p_atlas_coords.y < 0);
	ERR_FAIL_COND(p_size.x <= 0 || p_size.y <= 0);
	for (int x = 0; x < p_size.x; x++) {
		for (int y = 0; y < p_size.y; y++) {
			Vector2i coords = p_atlas_coords + Vector2i(x, y);
			ERR_FAIL_COND_MSG(tiles.has(coords), vformat("Cannot create tile at position %s with size %s. Already a tile present at %s.", p_atlas_coords, p_size, coords));
		}
	}

	// Create and resize the tile.
	tiles.insert(p_atlas_coords, TileSetAtlasSource::TileAlternativesData());
	tiles_ids.append(p_atlas_coords);
	tiles_ids.sort();

	tiles[p_atlas_coords].size_in_atlas = p_size;
	tiles[p_atlas_coords].alternatives[0] = memnew(TileData);
	tiles[p_atlas_coords].alternatives[0]->set_tile_set(tile_set);
	tiles[p_atlas_coords].alternatives[0]->set_allow_transform(false);
	tiles[p_atlas_coords].alternatives[0]->connect("changed", callable_mp((Resource *)this, &TileSetAtlasSource::emit_changed));
	tiles[p_atlas_coords].alternatives[0]->notify_property_list_changed();
	tiles[p_atlas_coords].alternatives_ids.append(0);

	// Add all covered positions to the mapping cache
	for (int x = 0; x < p_size.x; x++) {
		for (int y = 0; y < p_size.y; y++) {
			Vector2i coords = p_atlas_coords + Vector2i(x, y);
			_coords_mapping_cache[coords] = p_atlas_coords;
		}
	}

	emit_signal("changed");
}

void TileSetAtlasSource::remove_tile(Vector2i p_atlas_coords) {
	ERR_FAIL_COND_MSG(!tiles.has(p_atlas_coords), vformat("TileSetAtlasSource has no tile at %s.", String(p_atlas_coords)));

	// Remove all covered positions from the mapping cache
	Size2i size = tiles[p_atlas_coords].size_in_atlas;

	for (int x = 0; x < size.x; x++) {
		for (int y = 0; y < size.y; y++) {
			Vector2i coords = p_atlas_coords + Vector2i(x, y);
			_coords_mapping_cache.erase(coords);
		}
	}

	// Free tile data.
	for (Map<int, TileData *>::Element *E_tile_data = tiles[p_atlas_coords].alternatives.front(); E_tile_data; E_tile_data = E_tile_data->next()) {
		memdelete(E_tile_data->get());
	}

	// Delete the tile
	tiles.erase(p_atlas_coords);
	tiles_ids.erase(p_atlas_coords);
	tiles_ids.sort();

	emit_signal("changed");
}

bool TileSetAtlasSource::has_tile(Vector2i p_atlas_coords) const {
	return tiles.has(p_atlas_coords);
}

Vector2i TileSetAtlasSource::get_tile_at_coords(Vector2i p_atlas_coords) const {
	if (!_coords_mapping_cache.has(p_atlas_coords)) {
		return INVALID_ATLAS_COORDS;
	}

	return _coords_mapping_cache[p_atlas_coords];
}

Vector2i TileSetAtlasSource::get_tile_size_in_atlas(Vector2i p_atlas_coords) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), Vector2i(-1, -1), vformat("TileSetAtlasSource has no tile at %s.", String(p_atlas_coords)));

	return tiles[p_atlas_coords].size_in_atlas;
}

int TileSetAtlasSource::get_tiles_count() const {
	return tiles_ids.size();
}

Vector2i TileSetAtlasSource::get_tile_id(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, tiles_ids.size(), TileSetAtlasSource::INVALID_ATLAS_COORDS);
	return tiles_ids[p_index];
}

Rect2i TileSetAtlasSource::get_tile_texture_region(Vector2i p_atlas_coords) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), Rect2i(), vformat("TileSetAtlasSource has no tile at %s.", String(p_atlas_coords)));

	Vector2i size_in_atlas = tiles[p_atlas_coords].size_in_atlas;
	Vector2 region_size = texture_region_size * size_in_atlas + separation * (size_in_atlas - Vector2i(1, 1));

	Vector2 origin = margins + (p_atlas_coords * (texture_region_size + separation));

	return Rect2(origin, region_size);
	;
}

bool TileSetAtlasSource::can_move_tile_in_atlas(Vector2i p_atlas_coords, Vector2i p_new_atlas_coords, Vector2i p_new_size) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), false, vformat("TileSetAtlasSource has no tile at %s.", String(p_atlas_coords)));

	Vector2i new_atlas_coords = (p_new_atlas_coords != INVALID_ATLAS_COORDS) ? p_new_atlas_coords : p_atlas_coords;
	if (new_atlas_coords.x < 0 || new_atlas_coords.y < 0) {
		return false;
	}

	Vector2i size = (p_new_size != Vector2i(-1, -1)) ? p_new_size : tiles[p_atlas_coords].size_in_atlas;
	ERR_FAIL_COND_V(size.x <= 0 || size.y <= 0, false);

	Size2i grid_size = get_atlas_grid_size();
	if (new_atlas_coords.x + size.x > grid_size.x || new_atlas_coords.y + size.y > grid_size.y) {
		return false;
	}

	Rect2i new_rect = Rect2i(new_atlas_coords, size);
	// Check if the new tile can fit in the new rect.
	for (int x = new_rect.position.x; x < new_rect.get_end().x; x++) {
		for (int y = new_rect.position.y; y < new_rect.get_end().y; y++) {
			Vector2i coords = get_tile_at_coords(Vector2i(x, y));
			if (coords != p_atlas_coords && coords != TileSetAtlasSource::INVALID_ATLAS_COORDS) {
				return false;
			}
		}
	}

	return true;
}

void TileSetAtlasSource::move_tile_in_atlas(Vector2i p_atlas_coords, Vector2i p_new_atlas_coords, Vector2i p_new_size) {
	bool can_move = can_move_tile_in_atlas(p_atlas_coords, p_new_atlas_coords, p_new_size);
	ERR_FAIL_COND_MSG(!can_move, vformat("Cannot move tile at position %s with size %s. Tile already present.", p_new_atlas_coords, p_new_size));

	// Compute the actual new rect from arguments.
	Vector2i new_atlas_coords = (p_new_atlas_coords != INVALID_ATLAS_COORDS) ? p_new_atlas_coords : p_atlas_coords;
	Vector2i size = (p_new_size != Vector2i(-1, -1)) ? p_new_size : tiles[p_atlas_coords].size_in_atlas;

	if (new_atlas_coords == p_atlas_coords && size == tiles[p_atlas_coords].size_in_atlas) {
		return;
	}

	// Remove all covered positions from the mapping cache.
	Size2i old_size = tiles[p_atlas_coords].size_in_atlas;
	for (int x = 0; x < old_size.x; x++) {
		for (int y = 0; y < old_size.y; y++) {
			Vector2i coords = p_atlas_coords + Vector2i(x, y);
			_coords_mapping_cache.erase(coords);
		}
	}

	// Move the tile and update its size.
	if (new_atlas_coords != p_atlas_coords) {
		tiles[new_atlas_coords] = tiles[p_atlas_coords];
		tiles.erase(p_atlas_coords);

		tiles_ids.erase(p_atlas_coords);
		tiles_ids.append(new_atlas_coords);
		tiles_ids.sort();
	}
	tiles[new_atlas_coords].size_in_atlas = size;

	// Add all covered positions to the mapping cache again.
	for (int x = 0; x < size.x; x++) {
		for (int y = 0; y < size.y; y++) {
			Vector2i coords = new_atlas_coords + Vector2i(x, y);
			_coords_mapping_cache[coords] = new_atlas_coords;
		}
	}

	emit_signal("changed");
}

bool TileSetAtlasSource::has_tiles_outside_texture() {
	Vector2i grid_size = get_atlas_grid_size();
	Vector<Vector2i> to_remove;

	for (Map<Vector2i, TileSetAtlasSource::TileAlternativesData>::Element *E = tiles.front(); E; E = E->next()) {
		if (E->key().x >= grid_size.x || E->key().y >= grid_size.y) {
			return true;
		}
	}

	return false;
}

void TileSetAtlasSource::clear_tiles_outside_texture() {
	Vector2i grid_size = get_atlas_grid_size();
	Vector<Vector2i> to_remove;

	for (Map<Vector2i, TileSetAtlasSource::TileAlternativesData>::Element *E = tiles.front(); E; E = E->next()) {
		if (E->key().x >= grid_size.x || E->key().y >= grid_size.y) {
			to_remove.append(E->key());
		}
	}

	for (int i = 0; i < to_remove.size(); i++) {
		remove_tile(to_remove[i]);
	}
}

int TileSetAtlasSource::create_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_id_override) {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), -1, vformat("TileSetAtlasSource has no tile at %s.", String(p_atlas_coords)));
	ERR_FAIL_COND_V_MSG(p_alternative_id_override >= 0 && (tiles[p_atlas_coords].alternatives.has(p_alternative_id_override) || tiles[p_atlas_coords].alternatives.has(p_alternative_id_override)), -1, vformat("Cannot create alternative tile. Another alternative exists with id %d.", p_alternative_id_override));

	int new_alternative_id = p_alternative_id_override >= 0 ? p_alternative_id_override : tiles[p_atlas_coords].next_alternative_id;

	tiles[p_atlas_coords].alternatives[new_alternative_id] = memnew(TileData);
	tiles[p_atlas_coords].alternatives[new_alternative_id]->set_tile_set(tile_set);
	tiles[p_atlas_coords].alternatives[new_alternative_id]->set_allow_transform(true);
	tiles[p_atlas_coords].alternatives[new_alternative_id]->notify_property_list_changed();
	tiles[p_atlas_coords].alternatives_ids.append(new_alternative_id);
	tiles[p_atlas_coords].alternatives_ids.sort();
	_compute_next_alternative_id(p_atlas_coords);

	emit_signal("changed");

	return new_alternative_id;
}

void TileSetAtlasSource::remove_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_tile) {
	ERR_FAIL_COND_MSG(!tiles.has(p_atlas_coords), vformat("TileSetAtlasSource has no tile at %s.", String(p_atlas_coords)));
	ERR_FAIL_COND_MSG(!tiles[p_atlas_coords].alternatives.has(p_alternative_tile), vformat("TileSetAtlasSource has no alternative with id %d for tile coords %s.", p_alternative_tile, String(p_atlas_coords)));
	ERR_FAIL_COND_MSG(p_alternative_tile == 0, "Cannot remove the alternative with id 0, the base tile alternative cannot be removed.");

	memdelete(tiles[p_atlas_coords].alternatives[p_alternative_tile]);
	tiles[p_atlas_coords].alternatives.erase(p_alternative_tile);
	tiles[p_atlas_coords].alternatives_ids.erase(p_alternative_tile);
	tiles[p_atlas_coords].alternatives_ids.sort();

	emit_signal("changed");
}

void TileSetAtlasSource::set_alternative_tile_id(const Vector2i p_atlas_coords, int p_alternative_tile, int p_new_id) {
	ERR_FAIL_COND_MSG(!tiles.has(p_atlas_coords), vformat("TileSetAtlasSource has no tile at %s.", String(p_atlas_coords)));
	ERR_FAIL_COND_MSG(!tiles[p_atlas_coords].alternatives.has(p_alternative_tile), vformat("TileSetAtlasSource has no alternative with id %d for tile coords %s.", p_alternative_tile, String(p_atlas_coords)));
	ERR_FAIL_COND_MSG(p_alternative_tile == 0, "Cannot change the alternative with id 0, the base tile alternative cannot be modified.");

	ERR_FAIL_COND_MSG(tiles[p_atlas_coords].alternatives.has(p_new_id), vformat("TileSetAtlasSource has already an alternative with id %d at %s.", p_new_id, String(p_atlas_coords)));

	tiles[p_atlas_coords].alternatives[p_new_id] = tiles[p_atlas_coords].alternatives[p_alternative_tile];
	tiles[p_atlas_coords].alternatives_ids.append(p_new_id);

	tiles[p_atlas_coords].alternatives.erase(p_alternative_tile);
	tiles[p_atlas_coords].alternatives_ids.erase(p_alternative_tile);
	tiles[p_atlas_coords].alternatives_ids.sort();

	emit_signal("changed");
}

bool TileSetAtlasSource::has_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_tile) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), false, vformat("The TileSetAtlasSource atlas has no tile at %s.", String(p_atlas_coords)));
	return tiles[p_atlas_coords].alternatives.has(p_alternative_tile);
}

int TileSetAtlasSource::get_next_alternative_tile_id(const Vector2i p_atlas_coords) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), -1, vformat("The TileSetAtlasSource atlas has no tile at %s.", String(p_atlas_coords)));
	return tiles[p_atlas_coords].next_alternative_id;
}

int TileSetAtlasSource::get_alternative_tiles_count(const Vector2i p_atlas_coords) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), -1, vformat("The TileSetAtlasSource atlas has no tile at %s.", String(p_atlas_coords)));
	return tiles[p_atlas_coords].alternatives_ids.size();
}

int TileSetAtlasSource::get_alternative_tile_id(const Vector2i p_atlas_coords, int p_index) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), -1, vformat("The TileSetAtlasSource atlas has no tile at %s.", String(p_atlas_coords)));
	ERR_FAIL_INDEX_V(p_index, tiles[p_atlas_coords].alternatives_ids.size(), -1);

	return tiles[p_atlas_coords].alternatives_ids[p_index];
}

Object *TileSetAtlasSource::get_tile_data(const Vector2i p_atlas_coords, int p_alternative_tile) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), nullptr, vformat("The TileSetAtlasSource atlas has no tile at %s.", String(p_atlas_coords)));
	ERR_FAIL_COND_V_MSG(!tiles[p_atlas_coords].alternatives.has(p_alternative_tile), nullptr, vformat("TileSetAtlasSource has no alternative with id %d for tile coords %s.", p_alternative_tile, String(p_atlas_coords)));

	return tiles[p_atlas_coords].alternatives[p_alternative_tile];
}

void TileSetAtlasSource::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &TileSetAtlasSource::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &TileSetAtlasSource::get_texture);
	ClassDB::bind_method(D_METHOD("set_margins", "margins"), &TileSetAtlasSource::set_margins);
	ClassDB::bind_method(D_METHOD("get_margins"), &TileSetAtlasSource::get_margins);
	ClassDB::bind_method(D_METHOD("set_separation", "separation"), &TileSetAtlasSource::set_separation);
	ClassDB::bind_method(D_METHOD("get_separation"), &TileSetAtlasSource::get_separation);
	ClassDB::bind_method(D_METHOD("set_texture_region_size", "texture_region_size"), &TileSetAtlasSource::set_texture_region_size);
	ClassDB::bind_method(D_METHOD("get_texture_region_size"), &TileSetAtlasSource::get_texture_region_size);
	ClassDB::bind_method(D_METHOD("set_base_texture_offset", "base_texture_offset"), &TileSetAtlasSource::set_base_texture_offset);
	ClassDB::bind_method(D_METHOD("get_base_texture_offset"), &TileSetAtlasSource::get_base_texture_offset);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", PROPERTY_USAGE_NOEDITOR), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "margins", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_margins", "get_margins");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "separation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_separation", "get_separation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "tile_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_texture_region_size", "get_texture_region_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "base_texture_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_base_texture_offset", "get_base_texture_offset");

	// Base tiles
	ClassDB::bind_method(D_METHOD("create_tile", "atlas_coords", "size"), &TileSetAtlasSource::create_tile, DEFVAL(Vector2i(1, 1)));
	ClassDB::bind_method(D_METHOD("remove_tile", "atlas_coords"), &TileSetAtlasSource::remove_tile); // Remove a tile. If p_tile_key.alternative_tile if different from 0, remove the alternative
	ClassDB::bind_method(D_METHOD("has_tile", "atlas_coords"), &TileSetAtlasSource::has_tile);
	ClassDB::bind_method(D_METHOD("can_move_tile_in_atlas", "atlas_coords", "new_atlas_coords", "new_size"), &TileSetAtlasSource::can_move_tile_in_atlas, DEFVAL(INVALID_ATLAS_COORDS), DEFVAL(Vector2i(-1, -1)));
	ClassDB::bind_method(D_METHOD("move_tile_in_atlas", "atlas_coords", "new_atlas_coords", "new_size"), &TileSetAtlasSource::move_tile_in_atlas, DEFVAL(INVALID_ATLAS_COORDS), DEFVAL(Vector2i(-1, -1)));
	ClassDB::bind_method(D_METHOD("get_tile_size_in_atlas", "atlas_coords"), &TileSetAtlasSource::get_tile_size_in_atlas);

	ClassDB::bind_method(D_METHOD("get_tiles_count"), &TileSetAtlasSource::get_tiles_count);
	ClassDB::bind_method(D_METHOD("get_tile_id", "index"), &TileSetAtlasSource::get_tile_id);

	ClassDB::bind_method(D_METHOD("get_tile_at_coords", "atlas_coords"), &TileSetAtlasSource::get_tile_at_coords);

	// Alternative tiles
	ClassDB::bind_method(D_METHOD("create_alternative_tile", "atlas_coords", "alternative_id_override"), &TileSetAtlasSource::create_alternative_tile, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("remove_alternative_tile", "atlas_coords", "alternative_tile"), &TileSetAtlasSource::remove_alternative_tile);
	ClassDB::bind_method(D_METHOD("set_alternative_tile_id", "atlas_coords", "alternative_tile", "new_id"), &TileSetAtlasSource::set_alternative_tile_id);
	ClassDB::bind_method(D_METHOD("has_alternative_tile", "atlas_coords", "alternative_tile"), &TileSetAtlasSource::has_alternative_tile);
	ClassDB::bind_method(D_METHOD("get_next_alternative_tile_id", "atlas_coords"), &TileSetAtlasSource::get_next_alternative_tile_id);

	ClassDB::bind_method(D_METHOD("get_alternative_tiles_count", "atlas_coords"), &TileSetAtlasSource::get_alternative_tiles_count);
	ClassDB::bind_method(D_METHOD("get_alternative_tile_id", "atlas_coords", "index"), &TileSetAtlasSource::get_alternative_tile_id);

	ClassDB::bind_method(D_METHOD("get_tile_data", "atlas_coords", "index"), &TileSetAtlasSource::get_tile_data);

	// Helpers.
	ClassDB::bind_method(D_METHOD("get_atlas_grid_size"), &TileSetAtlasSource::get_atlas_grid_size);
	ClassDB::bind_method(D_METHOD("has_tiles_outside_texture"), &TileSetAtlasSource::has_tiles_outside_texture);
	ClassDB::bind_method(D_METHOD("clear_tiles_outside_texture"), &TileSetAtlasSource::clear_tiles_outside_texture);
	ClassDB::bind_method(D_METHOD("get_tile_texture_region", "atlas_coords"), &TileSetAtlasSource::get_tile_texture_region);
}

TileSetAtlasSource::~TileSetAtlasSource() {
	// Free everything needed.
	for (Map<Vector2i, TileAlternativesData>::Element *E_alternatives = tiles.front(); E_alternatives; E_alternatives = E_alternatives->next()) {
		for (Map<int, TileData *>::Element *E_tile_data = E_alternatives->get().alternatives.front(); E_tile_data; E_tile_data = E_tile_data->next()) {
			memdelete(E_tile_data->get());
		}
	}
}

TileData *TileSetAtlasSource::_get_atlas_tile_data(Vector2i p_atlas_coords, int p_alternative_tile) {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), nullptr, vformat("TileSetAtlasSource has no tile at %s.", String(p_atlas_coords)));
	ERR_FAIL_COND_V_MSG(!tiles[p_atlas_coords].alternatives.has(p_alternative_tile), nullptr, vformat("TileSetAtlasSource has no alternative with id %d for tile coords %s.", p_alternative_tile, String(p_atlas_coords)));

	return tiles[p_atlas_coords].alternatives[p_alternative_tile];
}

const TileData *TileSetAtlasSource::_get_atlas_tile_data(Vector2i p_atlas_coords, int p_alternative_tile) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), nullptr, vformat("TileSetAtlasSource has no tile at %s.", String(p_atlas_coords)));
	ERR_FAIL_COND_V_MSG(!tiles[p_atlas_coords].alternatives.has(p_alternative_tile), nullptr, vformat("TileSetAtlasSource has no alternative with id %d for tile coords %s.", p_alternative_tile, String(p_atlas_coords)));

	return tiles[p_atlas_coords].alternatives[p_alternative_tile];
}

void TileSetAtlasSource::_compute_next_alternative_id(const Vector2i p_atlas_coords) {
	ERR_FAIL_COND_MSG(!tiles.has(p_atlas_coords), vformat("TileSetAtlasSource has no tile at %s.", String(p_atlas_coords)));

	while (tiles[p_atlas_coords].alternatives.has(tiles[p_atlas_coords].next_alternative_id)) {
		tiles[p_atlas_coords].next_alternative_id = (tiles[p_atlas_coords].next_alternative_id % 1073741823) + 1; // 2 ** 30
	};
}

// --- Plugins ---
Vector<TileSetPlugin *> TileSet::get_tile_set_atlas_plugins() const {
	return tile_set_plugins_vector;
}

// -- Shape and layout --
void TileSet::set_tile_shape(TileSet::TileShape p_shape) {
	tile_shape = p_shape;

	for (Map<int, Ref<TileSetSource>>::Element *E_source = sources.front(); E_source; E_source = E_source->next()) {
		E_source->get()->notify_tile_data_properties_should_change();
	}

	emit_changed();
}
TileSet::TileShape TileSet::get_tile_shape() const {
	return tile_shape;
}

void TileSet::set_tile_layout(TileSet::TileLayout p_layout) {
	tile_layout = p_layout;
	emit_changed();
}
TileSet::TileLayout TileSet::get_tile_layout() const {
	return tile_layout;
}

void TileSet::set_tile_offset_axis(TileSet::TileOffsetAxis p_alignment) {
	tile_offset_axis = p_alignment;

	for (Map<int, Ref<TileSetSource>>::Element *E_source = sources.front(); E_source; E_source = E_source->next()) {
		E_source->get()->notify_tile_data_properties_should_change();
	}

	emit_changed();
}
TileSet::TileOffsetAxis TileSet::get_tile_offset_axis() const {
	return tile_offset_axis;
}

void TileSet::set_tile_size(Size2i p_size) {
	ERR_FAIL_COND(p_size.x < 1 || p_size.y < 1);
	tile_size = p_size;
	emit_changed();
}
Size2i TileSet::get_tile_size() const {
	return tile_size;
}

void TileSet::set_tile_skew(Vector2 p_skew) {
	emit_changed();
	tile_skew = p_skew;
}
Vector2 TileSet::get_tile_skew() const {
	return tile_skew;
}

int TileSet::get_next_source_id() const {
	return next_source_id;
}

void TileSet::_compute_next_source_id() {
	while (sources.has(next_source_id)) {
		next_source_id = (next_source_id + 1) % 1073741824; // 2 ** 30
	};
}

// Sources management
int TileSet::add_source(Ref<TileSetAtlasSource> p_tile_atlas_source, int p_atlas_source_id_override) {
	ERR_FAIL_COND_V(!p_tile_atlas_source.is_valid(), -1);
	ERR_FAIL_COND_V_MSG(p_atlas_source_id_override >= 0 && (sources.has(p_atlas_source_id_override)), -1, vformat("Cannot create TileSet atlas source. Another atlas source exists with id %d.", p_atlas_source_id_override));

	int new_source_id = p_atlas_source_id_override >= 0 ? p_atlas_source_id_override : next_source_id;
	sources[new_source_id] = p_tile_atlas_source;
	source_ids.append(new_source_id);
	source_ids.sort();
	p_tile_atlas_source->set_tile_set(this);
	_compute_next_source_id();

	sources[new_source_id]->connect("changed", callable_mp(this, &TileSet::_source_changed));

	emit_changed();

	return new_source_id;
}

void TileSet::remove_source(int p_source_id) {
	ERR_FAIL_COND_MSG(!sources.has(p_source_id), vformat("Cannot remove TileSet atlas source. No tileset atlas source with id %d.", p_source_id));

	sources[p_source_id]->disconnect("changed", callable_mp(this, &TileSet::_source_changed));

	sources[p_source_id]->set_tile_set(nullptr);
	sources.erase(p_source_id);
	source_ids.erase(p_source_id);
	source_ids.sort();

	emit_changed();
}

void TileSet::set_source_id(int p_source_id, int p_new_source_id) {
	ERR_FAIL_COND(p_new_source_id < 0);
	ERR_FAIL_COND_MSG(!sources.has(p_source_id), vformat("Cannot change TileSet atlas source ID. No tileset atlas source with id %d.", p_source_id));
	if (p_source_id == p_new_source_id) {
		return;
	}

	ERR_FAIL_COND_MSG(sources.has(p_new_source_id), vformat("Cannot change TileSet atlas source ID. Another atlas source exists with id %d.", p_new_source_id));

	sources[p_new_source_id] = sources[p_source_id];
	sources.erase(p_source_id);

	source_ids.erase(p_source_id);
	source_ids.append(p_new_source_id);
	source_ids.sort();

	emit_changed();
}

bool TileSet::has_source(int p_source_id) const {
	return sources.has(p_source_id);
}

Ref<TileSetSource> TileSet::get_source(int p_source_id) const {
	ERR_FAIL_COND_V_MSG(!sources.has(p_source_id), nullptr, vformat("No TileSet atlas source with id %d.", p_source_id));

	return sources[p_source_id];
}

int TileSet::get_source_count() const {
	return source_ids.size();
}

int TileSet::get_source_id(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, source_ids.size(), -1);
	return source_ids[p_index];
}

// Rendering
void TileSet::set_uv_clipping(bool p_uv_clipping) {
	if (uv_clipping == p_uv_clipping) {
		return;
	}
	uv_clipping = p_uv_clipping;
	emit_changed();
}
bool TileSet::is_uv_clipping() const {
	return uv_clipping;
};

void TileSet::set_y_sorting(bool p_y_sort) {
	if (y_sorting == p_y_sort) {
		return;
	}
	y_sorting = p_y_sort;
	emit_changed();
}
bool TileSet::is_y_sorting() const {
	return y_sorting;
};

void TileSet::set_occlusion_layers_count(int p_occlusion_layers_count) {
	ERR_FAIL_COND(p_occlusion_layers_count < 0);
	if (occlusion_layers.size() == p_occlusion_layers_count) {
		return;
	}

	occlusion_layers.resize(p_occlusion_layers_count);

	for (Map<int, Ref<TileSetSource>>::Element *E_source = sources.front(); E_source; E_source = E_source->next()) {
		E_source->get()->notify_tile_data_properties_should_change();
	}

	notify_property_list_changed();
	emit_changed();
}

int TileSet::get_occlusion_layers_count() const {
	return occlusion_layers.size();
};

void TileSet::set_occlusion_layer_light_mask(int p_layer_index, int p_light_mask) {
	ERR_FAIL_INDEX(p_layer_index, occlusion_layers.size());
	occlusion_layers.write[p_layer_index].light_mask = p_light_mask;
	emit_changed();
}

int TileSet::get_occlusion_layer_light_mask(int p_layer_index) const {
	ERR_FAIL_INDEX_V(p_layer_index, occlusion_layers.size(), 0);
	return occlusion_layers[p_layer_index].light_mask;
}

void TileSet::set_occlusion_layer_sdf_collision(int p_layer_index, int p_sdf_collision) {
	ERR_FAIL_INDEX(p_layer_index, occlusion_layers.size());
	occlusion_layers.write[p_layer_index].sdf_collision = p_sdf_collision;
	emit_changed();
}

bool TileSet::get_occlusion_layer_sdf_collision(int p_layer_index) const {
	ERR_FAIL_INDEX_V(p_layer_index, occlusion_layers.size(), false);
	return occlusion_layers[p_layer_index].sdf_collision;
}

void TileSet::draw_tile_shape(CanvasItem *p_canvas_item, Rect2 p_region, Color p_color, bool p_filled, Ref<Texture2D> p_texture) {
	// TODO: optimize this with 2D meshes when they work again.
	if (get_tile_shape() == TileSet::TILE_SHAPE_SQUARE) {
		if (p_filled && p_texture.is_valid()) {
			p_canvas_item->draw_texture_rect(p_texture, p_region, false, p_color);
		} else {
			p_canvas_item->draw_rect(p_region, p_color, p_filled);
		}
	} else {
		float overlap = 0.0;
		switch (get_tile_shape()) {
			case TileSet::TILE_SHAPE_ISOMETRIC:
				overlap = 0.5;
				break;
			case TileSet::TILE_SHAPE_HEXAGON:
				overlap = 0.25;
				break;
			case TileSet::TILE_SHAPE_HALF_OFFSET_SQUARE:
				overlap = 0.0;
				break;
			default:
				break;
		}

		Vector<Vector2> uvs;
		uvs.append(Vector2(0.5, 0.0));
		uvs.append(Vector2(0.0, overlap));
		uvs.append(Vector2(0.0, 1.0 - overlap));
		uvs.append(Vector2(0.5, 1.0));
		uvs.append(Vector2(1.0, 1.0 - overlap));
		uvs.append(Vector2(1.0, overlap));
		uvs.append(Vector2(0.5, 0.0));
		if (get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
			for (int i = 0; i < uvs.size(); i++) {
				uvs.write[i] = Vector2(uvs[i].y, uvs[i].x);
			}
		}

		Vector<Vector2> points;
		for (int i = 0; i < uvs.size(); i++) {
			points.append(p_region.position + uvs[i] * p_region.size);
		}

		if (p_filled) {
			// This does hurt performances a lot. We should use a mesh if possible instead.
			p_canvas_item->draw_colored_polygon(points, p_color, uvs, p_texture);

			// Should improve performances, but does not work as draw_primitive does not work with textures :/ :
			/*for (int i = 0; i < 6; i += 3) {
				Vector<Vector2> quad;
				quad.append(points[i]);
				quad.append(points[(i + 1) % points.size()]);
				quad.append(points[(i + 2) % points.size()]);
				quad.append(points[(i + 3) % points.size()]);

				Vector<Vector2> uv_quad;
				uv_quad.append(uvs[i]);
				uv_quad.append(uvs[(i + 1) % uvs.size()]);
				uv_quad.append(uvs[(i + 2) % uvs.size()]);
				uv_quad.append(uvs[(i + 3) % uvs.size()]);

				p_control->draw_primitive(quad, Vector<Color>(), uv_quad, p_texture);
			}*/

		} else {
			// This does hurt performances a lot. We should use a mesh if possible instead.
			// tile_shape_grid->draw_polyline(points, p_color);
			for (int i = 0; i < points.size() - 1; i++) {
				p_canvas_item->draw_line(points[i], points[i + 1], p_color);
			}
		}
	}
}

Vector2i TileSet::get_tile_effective_texture_offset(int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile) const {
	ERR_FAIL_COND_V_MSG(!sources.has(p_atlas_source_id), Vector2i(), vformat("The TileSet has no source with id %d", p_atlas_source_id));
	const TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(*(sources[p_atlas_source_id]));
	ERR_FAIL_COND_V_MSG(!atlas_source, Vector2i(), vformat("The TileSet source with %d is not a TileAtlasSource", p_atlas_source_id));
	ERR_FAIL_COND_V_MSG(!atlas_source->has_tile(p_atlas_coords), Vector2i(), vformat("The TileSet atlas source with id %d has no tile at %s.", p_atlas_source_id, String(p_atlas_coords)));
	ERR_FAIL_COND_V_MSG(!atlas_source->has_alternative_tile(p_atlas_coords, p_alternative_tile), Vector2i(), vformat("The TileSet atlas source with id %d has no alternative tile with id %d at %s.", p_atlas_source_id, p_alternative_tile, String(p_atlas_coords)));

	Vector2 margin = (atlas_source->get_tile_texture_region(p_atlas_coords).size - get_tile_size()) / 2;
	margin = Vector2i(MAX(0, margin.x), MAX(0, margin.y));
	Vector2i effective_texture_offset = atlas_source->get_base_texture_offset() + Object::cast_to<TileData>(atlas_source->get_tile_data(p_atlas_coords, p_alternative_tile))->get_texture_offset();
	if (ABS(effective_texture_offset.x) > margin.x || ABS(effective_texture_offset.y) > margin.y) {
		effective_texture_offset.x = CLAMP(effective_texture_offset.x, -margin.x, margin.x);
		effective_texture_offset.y = CLAMP(effective_texture_offset.y, -margin.y, margin.y);
	}

	return effective_texture_offset;
}

// Physics
void TileSet::set_physics_layers_count(int p_physics_layers_count) {
	ERR_FAIL_COND(p_physics_layers_count < 0);
	if (physics_layers.size() == p_physics_layers_count) {
		return;
	}

	physics_layers.resize(p_physics_layers_count);

	for (Map<int, Ref<TileSetSource>>::Element *E_source = sources.front(); E_source; E_source = E_source->next()) {
		E_source->get()->notify_tile_data_properties_should_change();
	}

	notify_property_list_changed();
	emit_changed();
}

int TileSet::get_physics_layers_count() const {
	return physics_layers.size();
}

void TileSet::set_physics_layer_collision_layer(int p_layer_index, uint32_t p_layer) {
	ERR_FAIL_INDEX(p_layer_index, physics_layers.size());
	physics_layers.write[p_layer_index].collision_layer = p_layer;
	emit_changed();
}

uint32_t TileSet::get_physics_layer_collision_layer(int p_layer_index) const {
	ERR_FAIL_INDEX_V(p_layer_index, physics_layers.size(), 0);
	return physics_layers[p_layer_index].collision_layer;
}

void TileSet::set_physics_layer_collision_mask(int p_layer_index, uint32_t p_mask) {
	ERR_FAIL_INDEX(p_layer_index, physics_layers.size());
	physics_layers.write[p_layer_index].collision_mask = p_mask;
	emit_changed();
}

uint32_t TileSet::get_physics_layer_collision_mask(int p_layer_index) const {
	ERR_FAIL_INDEX_V(p_layer_index, physics_layers.size(), 0);
	return physics_layers[p_layer_index].collision_mask;
}

void TileSet::set_physics_layer_physics_material(int p_layer_index, Ref<PhysicsMaterial> p_physics_material) {
	ERR_FAIL_INDEX(p_layer_index, physics_layers.size());
	physics_layers.write[p_layer_index].physics_material = p_physics_material;
}

Ref<PhysicsMaterial> TileSet::get_physics_layer_physics_material(int p_layer_index) const {
	ERR_FAIL_INDEX_V(p_layer_index, physics_layers.size(), Ref<PhysicsMaterial>());
	return physics_layers[p_layer_index].physics_material;
}

// Terrains
void TileSet::set_terrain_sets_count(int p_terrains_sets_count) {
	ERR_FAIL_COND(p_terrains_sets_count < 0);

	terrain_sets.resize(p_terrains_sets_count);

	notify_property_list_changed();
	emit_changed();
}

int TileSet::get_terrain_sets_count() const {
	return terrain_sets.size();
}

void TileSet::set_terrain_set_mode(int p_terrain_set, TerrainMode p_terrain_mode) {
	ERR_FAIL_INDEX(p_terrain_set, terrain_sets.size());
	terrain_sets.write[p_terrain_set].mode = p_terrain_mode;
	for (Map<int, Ref<TileSetSource>>::Element *E_source = sources.front(); E_source; E_source = E_source->next()) {
		E_source->get()->notify_tile_data_properties_should_change();
	}

	notify_property_list_changed();
	emit_changed();
}

TileSet::TerrainMode TileSet::get_terrain_set_mode(int p_terrain_set) const {
	ERR_FAIL_INDEX_V(p_terrain_set, terrain_sets.size(), TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES);
	return terrain_sets[p_terrain_set].mode;
}

void TileSet::set_terrains_count(int p_terrain_set, int p_terrains_layers_count) {
	ERR_FAIL_INDEX(p_terrain_set, terrain_sets.size());
	ERR_FAIL_COND(p_terrains_layers_count < 0);
	if (terrain_sets[p_terrain_set].terrains.size() == p_terrains_layers_count) {
		return;
	}

	int old_size = terrain_sets[p_terrain_set].terrains.size();
	terrain_sets.write[p_terrain_set].terrains.resize(p_terrains_layers_count);

	// Default name and color
	for (int i = old_size; i < terrain_sets.write[p_terrain_set].terrains.size(); i++) {
		float hue_rotate = (i * 2 % 16) / 16.0;
		Color c;
		c.set_hsv(Math::fmod(float(hue_rotate), float(1.0)), 0.5, 0.5);
		terrain_sets.write[p_terrain_set].terrains.write[i].color = c;
		terrain_sets.write[p_terrain_set].terrains.write[i].name = String(vformat("Terrain %d", i));
	}

	for (Map<int, Ref<TileSetSource>>::Element *E_source = sources.front(); E_source; E_source = E_source->next()) {
		E_source->get()->notify_tile_data_properties_should_change();
	}

	notify_property_list_changed();
	emit_changed();
}

int TileSet::get_terrains_count(int p_terrain_set) const {
	ERR_FAIL_INDEX_V(p_terrain_set, terrain_sets.size(), -1);
	return terrain_sets[p_terrain_set].terrains.size();
}

void TileSet::set_terrain_name(int p_terrain_set, int p_terrain_index, String p_name) {
	ERR_FAIL_INDEX(p_terrain_set, terrain_sets.size());
	ERR_FAIL_INDEX(p_terrain_index, terrain_sets[p_terrain_set].terrains.size());
	terrain_sets.write[p_terrain_set].terrains.write[p_terrain_index].name = p_name;
	emit_changed();
}

String TileSet::get_terrain_name(int p_terrain_set, int p_terrain_index) const {
	ERR_FAIL_INDEX_V(p_terrain_set, terrain_sets.size(), String());
	ERR_FAIL_INDEX_V(p_terrain_index, terrain_sets[p_terrain_set].terrains.size(), String());
	return terrain_sets[p_terrain_set].terrains[p_terrain_index].name;
}

void TileSet::set_terrain_color(int p_terrain_set, int p_terrain_index, Color p_color) {
	ERR_FAIL_INDEX(p_terrain_set, terrain_sets.size());
	ERR_FAIL_INDEX(p_terrain_index, terrain_sets[p_terrain_set].terrains.size());
	if (p_color.a != 1.0) {
		WARN_PRINT("Terrain color should have alpha == 1.0");
		p_color.a = 1.0;
	}
	terrain_sets.write[p_terrain_set].terrains.write[p_terrain_index].color = p_color;
	emit_changed();
}

Color TileSet::get_terrain_color(int p_terrain_set, int p_terrain_index) const {
	ERR_FAIL_INDEX_V(p_terrain_index, terrain_sets[p_terrain_set].terrains.size(), Color());
	return terrain_sets[p_terrain_set].terrains[p_terrain_index].color;
}

bool TileSet::is_valid_peering_bit_terrain(int p_terrain_set, TileSet::CellNeighbor p_peering_bit) const {
	if (p_terrain_set < 0 || p_terrain_set >= get_terrain_sets_count()) {
		return false;
	}

	TileSet::TerrainMode terrain_mode = get_terrain_set_mode(p_terrain_set);
	if (tile_shape == TileSet::TILE_SHAPE_SQUARE) {
		if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || terrain_mode == TileSet::TERRAIN_MODE_MATCH_SIDES) {
			if (p_peering_bit == TileSet::CELL_NEIGHBOR_RIGHT_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_LEFT_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_SIDE) {
				return true;
			}
		}
		if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
			if (p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER) {
				return true;
			}
		}
	} else if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || terrain_mode == TileSet::TERRAIN_MODE_MATCH_SIDES) {
			if (p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
				return true;
			}
		}
		if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
			if (p_peering_bit == TileSet::CELL_NEIGHBOR_RIGHT_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_LEFT_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
				return true;
			}
		}
	} else {
		if (get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || terrain_mode == TileSet::TERRAIN_MODE_MATCH_SIDES) {
				if (p_peering_bit == TileSet::CELL_NEIGHBOR_RIGHT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_LEFT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
					return true;
				}
			}
			if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
				if (p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_CORNER ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER) {
					return true;
				}
			}
		} else {
			if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || terrain_mode == TileSet::TERRAIN_MODE_MATCH_SIDES) {
				if (p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
					return true;
				}
			}
			if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
				if (p_peering_bit == TileSet::CELL_NEIGHBOR_RIGHT_CORNER ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_LEFT_CORNER ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER) {
					return true;
				}
			}
		}
	}
	return false;
}

// Navigation
void TileSet::set_navigation_layers_count(int p_navigation_layers_count) {
	ERR_FAIL_COND(p_navigation_layers_count < 0);
	if (navigation_layers.size() == p_navigation_layers_count) {
		return;
	}

	navigation_layers.resize(p_navigation_layers_count);

	for (Map<int, Ref<TileSetSource>>::Element *E_source = sources.front(); E_source; E_source = E_source->next()) {
		E_source->get()->notify_tile_data_properties_should_change();
	}

	notify_property_list_changed();
	emit_changed();
}

int TileSet::get_navigation_layers_count() const {
	return navigation_layers.size();
}

void TileSet::set_navigation_layer_layers(int p_layer_index, uint32_t p_layers) {
	ERR_FAIL_INDEX(p_layer_index, navigation_layers.size());
	navigation_layers.write[p_layer_index].layers = p_layers;
	emit_changed();
}

uint32_t TileSet::get_navigation_layer_layers(int p_layer_index) const {
	ERR_FAIL_INDEX_V(p_layer_index, navigation_layers.size(), 0);
	return navigation_layers[p_layer_index].layers;
}

// Custom data.
void TileSet::set_custom_data_layers_count(int p_custom_data_layers_count) {
	ERR_FAIL_COND(p_custom_data_layers_count < 0);
	if (custom_data_layers.size() == p_custom_data_layers_count) {
		return;
	}

	custom_data_layers.resize(p_custom_data_layers_count);

	for (Map<String, int>::Element *E = custom_data_layers_by_name.front(); E; E = E->next()) {
		if (E->get() >= custom_data_layers.size()) {
			custom_data_layers_by_name.erase(E);
		}
	}

	for (Map<int, Ref<TileSetSource>>::Element *E_source = sources.front(); E_source; E_source = E_source->next()) {
		E_source->get()->notify_tile_data_properties_should_change();
	}

	notify_property_list_changed();
	emit_changed();
}

int TileSet::get_custom_data_layers_count() const {
	return custom_data_layers.size();
}

int TileSet::get_custom_data_layer_by_name(String p_value) const {
	if (custom_data_layers_by_name.has(p_value)) {
		return custom_data_layers_by_name[p_value];
	} else {
		return -1;
	}
}

void TileSet::set_custom_data_name(int p_layer_id, String p_value) {
	ERR_FAIL_INDEX(p_layer_id, custom_data_layers.size());

	// Exit if another property has the same name.
	if (!p_value.is_empty()) {
		for (int other_layer_id = 0; other_layer_id < get_custom_data_layers_count(); other_layer_id++) {
			if (other_layer_id != p_layer_id && get_custom_data_name(other_layer_id) == p_value) {
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

String TileSet::get_custom_data_name(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, custom_data_layers.size(), "");
	return custom_data_layers[p_layer_id].name;
}

void TileSet::set_custom_data_type(int p_layer_id, Variant::Type p_value) {
	ERR_FAIL_INDEX(p_layer_id, custom_data_layers.size());
	custom_data_layers.write[p_layer_id].type = p_value;

	for (Map<int, Ref<TileSetSource>>::Element *E_source = sources.front(); E_source; E_source = E_source->next()) {
		E_source->get()->notify_tile_data_properties_should_change();
	}

	emit_changed();
}

Variant::Type TileSet::get_custom_data_type(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, custom_data_layers.size(), Variant::NIL);
	return custom_data_layers[p_layer_id].type;
}

void TileSet::_source_changed() {
	emit_changed();
	notify_property_list_changed();
}

void TileSet::reset_state() {
	occlusion_layers.clear();
	physics_layers.clear();
	custom_data_layers.clear();
}

const Vector2i TileSetAtlasSource::INVALID_ATLAS_COORDS = Vector2i(-1, -1);
const int TileSetAtlasSource::INVALID_TILE_ALTERNATIVE = -1;

#ifndef DISABLE_DEPRECATED
void TileSet::compatibility_conversion() {
	for (Map<int, CompatibilityTileData *>::Element *E = compatibility_data.front(); E; E = E->next()) {
		CompatibilityTileData *ctd = E->value();

		// Add the texture
		TileSetAtlasSource *atlas_source = memnew(TileSetAtlasSource);
		int source_id = add_source(Ref<TileSetSource>(atlas_source));

		atlas_source->set_texture(ctd->texture);

		// Handle each tile as a new source. Not optimal but at least it should stay compatible.
		switch (ctd->tile_mode) {
			case 0: // SINGLE_TILE
				// TODO
				break;
			case 1: // AUTO_TILE
				// TODO
				break;
			case 2: // ATLAS_TILE
				atlas_source->set_margins(ctd->region.get_position());
				atlas_source->set_separation(Vector2i(ctd->autotile_spacing, ctd->autotile_spacing));
				atlas_source->set_texture_region_size(ctd->autotile_tile_size);

				Size2i atlas_size = ctd->region.get_size() / (ctd->autotile_tile_size + atlas_source->get_separation());
				for (int i = 0; i < atlas_size.x; i++) {
					for (int j = 0; j < atlas_size.y; j++) {
						Vector2i coords = Vector2i(i, j);

						for (int flags = 0; flags < 8; flags++) {
							bool flip_h = flags & 1;
							bool flip_v = flags & 2;
							bool transpose = flags & 4;

							int alternative_tile = 0;
							if (!atlas_source->has_tile(coords)) {
								atlas_source->create_tile(coords);
							} else {
								alternative_tile = atlas_source->create_alternative_tile(coords);
							}
							TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(coords, alternative_tile));

							tile_data->set_flip_h(flip_h);
							tile_data->set_flip_v(flip_v);
							tile_data->set_transpose(transpose);
							tile_data->tile_set_material(ctd->material);
							tile_data->set_modulate(ctd->modulate);
							tile_data->set_z_index(ctd->z_index);
							if (ctd->autotile_z_index_map.has(coords)) {
								tile_data->set_z_index(ctd->autotile_z_index_map[coords]);
							}
							if (ctd->autotile_priority_map.has(coords)) {
								tile_data->set_probability(ctd->autotile_priority_map[coords]);
							}

							// -- TODO: handle --
							// texture offset
							// bitmask_mode
							// bitmask_flags
							// occluder_map
							// navpoly_map
							// "occluder"
							// "occluder_offset"
							// "navigation"
							// "navigation_offset"

							print_line(vformat("Created compatibility tile: source id=%d, coords=(%s), alternative=%d", source_id, String(coords), alternative_tile));
						}
					}
				}
				break;
		}

		// Add the mapping to the map
		compatibility_source_mapping.insert(E->key(), source_id);
	}

	// Reset compatibility data
	compatibility_data = Map<int, CompatibilityTileData *>();
}
#endif // DISABLE_DEPRECATED

bool TileSet::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);

#ifndef DISABLE_DEPRECATED
	// TODO: THIS IS HOW WE CHECK IF WE HAVE A DEPRECATED RESOURCE
	// This should be moved to a dedicated conversion system
	if (components[0].is_valid_integer()) {
		int id = components[0].to_int();

		// Get or create the compatibility object
		CompatibilityTileData *ctd;
		Map<int, CompatibilityTileData *>::Element *E = compatibility_data.find(id);
		if (!E) {
			ctd = memnew(CompatibilityTileData);
			compatibility_data.insert(id, ctd);
		} else {
			ctd = E->get();
		}

		String what = components[1];

		if (what == "name") {
			ctd->name = p_value;
		} else if (what == "texture") {
			ctd->texture = p_value;
		} else if (what == "tex_offset") {
			ctd->tex_offset = p_value;
		} else if (what == "material") {
			ctd->material = p_value;
		} else if (what == "modulate") {
			ctd->modulate = p_value;
		} else if (what == "region") {
			ctd->region = p_value;
		} else if (what == "tile_mode") {
			ctd->tile_mode = p_value;
		} else if (what.left(9) == "autotile") {
			what = what.right(9);
			if (what == "bitmask_mode") {
				ctd->autotile_bitmask_mode = p_value;
			} else if (what == "icon_coordinate") {
				ctd->autotile_icon_coordinate = p_value;
			} else if (what == "tile_size") {
				ctd->autotile_tile_size = p_value;
			} else if (what == "spacing") {
				ctd->autotile_spacing = p_value;
			} else if (what == "bitmask_flags") {
				if (p_value.is_array()) {
					Array p = p_value;
					Vector2i last_coord;
					while (p.size() > 0) {
						if (p[0].get_type() == Variant::VECTOR2) {
							last_coord = p[0];
						} else if (p[0].get_type() == Variant::INT) {
							ctd->autotile_bitmask_flags.insert(last_coord, p[0]);
						}
						p.pop_front();
					}
				}
			} else if (what == "occluder_map") {
				Array p = p_value;
				Vector2 last_coord;
				while (p.size() > 0) {
					if (p[0].get_type() == Variant::VECTOR2) {
						last_coord = p[0];
					} else if (p[0].get_type() == Variant::OBJECT) {
						ctd->autotile_occluder_map.insert(last_coord, p[0]);
					}
					p.pop_front();
				}
			} else if (what == "navpoly_map") {
				Array p = p_value;
				Vector2 last_coord;
				while (p.size() > 0) {
					if (p[0].get_type() == Variant::VECTOR2) {
						last_coord = p[0];
					} else if (p[0].get_type() == Variant::OBJECT) {
						ctd->autotile_navpoly_map.insert(last_coord, p[0]);
					}
					p.pop_front();
				}
			} else if (what == "priority_map") {
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
						ctd->autotile_priority_map.insert(v, priority);
					}
					p.pop_front();
				}
			} else if (what == "z_index_map") {
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
						ctd->autotile_z_index_map.insert(v, z_index);
					}
					p.pop_front();
				}
			}
			/*
		// INGORED FOR NOW, they seem duplicated data compared to the shapes array
		else if (what == "shape") {
			// TODO
		} else if (what == "shape_offset") {
			// TODO
		} else if (what == "shape_transform") {
			// TODO
		} else if (what == "shape_one_way") {
			// TODO
		} else if (what == "shape_one_way_margin") {
			// TODO
		}
		*/
		} else if (what == "shapes") {
			// TODO
		} else if (what == "occluder") {
			// TODO
		} else if (what == "occluder_offset") {
			// TODO
		} else if (what == "navigation") {
			// TODO
		} else if (what == "navigation_offset") {
			// TODO
		} else if (what == "z_index") {
			ctd->z_index = p_value;

			// TODO: remove the consersion from here, it's not where it should be done
			compatibility_conversion();
		} else {
			return false;
		}
	} else {
#endif // DISABLE_DEPRECATED

		// This is now a new property.
		if (components.size() == 2 && components[0].begins_with("occlusion_layer_") && components[0].trim_prefix("occlusion_layer_").is_valid_integer()) {
			// Occlusion layers.
			int index = components[0].trim_prefix("occlusion_layer_").to_int();
			ERR_FAIL_COND_V(index < 0, false);
			if (components[1] == "light_mask") {
				ERR_FAIL_COND_V(p_value.get_type() != Variant::INT, false);
				if (index >= occlusion_layers.size()) {
					set_occlusion_layers_count(index + 1);
				}
				set_occlusion_layer_light_mask(index, p_value);
				return true;
			} else if (components[1] == "sdf_collision") {
				ERR_FAIL_COND_V(p_value.get_type() != Variant::BOOL, false);
				if (index >= occlusion_layers.size()) {
					set_occlusion_layers_count(index + 1);
				}
				set_occlusion_layer_sdf_collision(index, p_value);
				return true;
			}
		} else if (components.size() == 2 && components[0].begins_with("physics_layer_") && components[0].trim_prefix("physics_layer_").is_valid_integer()) {
			// Physics layers.
			int index = components[0].trim_prefix("physics_layer_").to_int();
			ERR_FAIL_COND_V(index < 0, false);
			if (components[1] == "collision_layer") {
				ERR_FAIL_COND_V(p_value.get_type() != Variant::INT, false);
				if (index >= physics_layers.size()) {
					set_physics_layers_count(index + 1);
				}
				set_physics_layer_collision_layer(index, p_value);
				return true;
			} else if (components[1] == "collision_mask") {
				ERR_FAIL_COND_V(p_value.get_type() != Variant::INT, false);
				if (index >= physics_layers.size()) {
					set_physics_layers_count(index + 1);
				}
				set_physics_layer_collision_mask(index, p_value);
				return true;
			} else if (components[1] == "physics_material") {
				Ref<PhysicsMaterial> physics_material = p_value;
				ERR_FAIL_COND_V(!physics_material.is_valid(), false);
				if (index >= physics_layers.size()) {
					set_physics_layers_count(index + 1);
				}
				set_physics_layer_physics_material(index, physics_material);
				return true;
			}
		} else if (components.size() >= 2 && components[0].begins_with("terrain_set_") && components[0].trim_prefix("terrain_set_").is_valid_integer()) {
			// Terrains.
			int terrain_set_index = components[0].trim_prefix("terrain_set_").to_int();
			ERR_FAIL_COND_V(terrain_set_index < 0, false);
			if (components[1] == "mode") {
				ERR_FAIL_COND_V(p_value.get_type() != Variant::INT, false);
				if (terrain_set_index >= terrain_sets.size()) {
					set_terrain_sets_count(terrain_set_index + 1);
				}
				set_terrain_set_mode(terrain_set_index, TerrainMode(int(p_value)));
			} else if (components[1] == "terrains_count") {
				ERR_FAIL_COND_V(p_value.get_type() != Variant::INT, false);
				if (terrain_set_index >= terrain_sets.size()) {
					set_terrain_sets_count(terrain_set_index + 1);
				}
				set_terrains_count(terrain_set_index, p_value);
				return true;
			} else if (components.size() >= 3 && components[1].begins_with("terrain_") && components[1].trim_prefix("terrain_").is_valid_integer()) {
				int terrain_index = components[1].trim_prefix("terrain_").to_int();
				ERR_FAIL_COND_V(terrain_index < 0, false);
				if (components[2] == "name") {
					ERR_FAIL_COND_V(p_value.get_type() != Variant::STRING, false);
					if (terrain_set_index >= terrain_sets.size()) {
						set_terrain_sets_count(terrain_set_index + 1);
					}
					if (terrain_index >= terrain_sets[terrain_set_index].terrains.size()) {
						set_terrains_count(terrain_set_index, terrain_index + 1);
					}
					set_terrain_name(terrain_set_index, terrain_index, p_value);
					return true;
				} else if (components[2] == "color") {
					ERR_FAIL_COND_V(p_value.get_type() != Variant::COLOR, false);
					if (terrain_set_index >= terrain_sets.size()) {
						set_terrain_sets_count(terrain_set_index + 1);
					}
					if (terrain_index >= terrain_sets[terrain_set_index].terrains.size()) {
						set_terrains_count(terrain_set_index, terrain_index + 1);
					}
					set_terrain_color(terrain_set_index, terrain_index, p_value);
					return true;
				}
			}
		} else if (components.size() == 2 && components[0].begins_with("navigation_layer_") && components[0].trim_prefix("navigation_layer_").is_valid_integer()) {
			// Navigation layers.
			int index = components[0].trim_prefix("navigation_layer_").to_int();
			ERR_FAIL_COND_V(index < 0, false);
			if (components[1] == "layers") {
				ERR_FAIL_COND_V(p_value.get_type() != Variant::INT, false);
				if (index >= navigation_layers.size()) {
					set_navigation_layers_count(index + 1);
				}
				set_navigation_layer_layers(index, p_value);
				return true;
			}
		} else if (components.size() == 2 && components[0].begins_with("custom_data_layer_") && components[0].trim_prefix("custom_data_layer_").is_valid_integer()) {
			// Custom data layers.
			int index = components[0].trim_prefix("custom_data_layer_").to_int();
			ERR_FAIL_COND_V(index < 0, false);
			if (components[1] == "name") {
				ERR_FAIL_COND_V(p_value.get_type() != Variant::STRING, false);
				if (index >= custom_data_layers.size()) {
					set_custom_data_layers_count(index + 1);
				}
				set_custom_data_name(index, p_value);
				return true;
			} else if (components[1] == "type") {
				ERR_FAIL_COND_V(p_value.get_type() != Variant::INT, false);
				if (index >= custom_data_layers.size()) {
					set_custom_data_layers_count(index + 1);
				}
				set_custom_data_type(index, Variant::Type(int(p_value)));
				return true;
			}
		} else if (components.size() == 2 && components[0] == "sources" && components[1].is_valid_integer()) {
			// Create atlas if it does not exists.
			int source_id = components[1].to_int();

			if (!has_source(source_id)) {
				add_source(p_value, source_id);
			}
			return true;
		}

#ifndef DISABLE_DEPRECATED
	}
#endif // DISABLE_DEPRECATED

	return false;
}

bool TileSet::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> components = String(p_name).split("/", true, 2);

	if (components.size() == 2 && components[0].begins_with("occlusion_layer_") && components[0].trim_prefix("occlusion_layer_").is_valid_integer()) {
		// Occlusion layers.
		int index = components[0].trim_prefix("occlusion_layer_").to_int();
		if (index < 0 || index >= occlusion_layers.size()) {
			return false;
		}
		if (components[1] == "light_mask") {
			r_ret = get_occlusion_layer_light_mask(index);
			return true;
		} else if (components[1] == "sdf_collision") {
			r_ret = get_occlusion_layer_sdf_collision(index);
			return true;
		}
	} else if (components.size() == 2 && components[0].begins_with("physics_layer_") && components[0].trim_prefix("physics_layer_").is_valid_integer()) {
		// Physics layers.
		int index = components[0].trim_prefix("physics_layer_").to_int();
		if (index < 0 || index >= physics_layers.size()) {
			return false;
		}
		if (components[1] == "collision_layer") {
			r_ret = get_physics_layer_collision_layer(index);
			return true;
		} else if (components[1] == "collision_mask") {
			r_ret = get_physics_layer_collision_mask(index);
			return true;
		} else if (components[1] == "physics_material") {
			r_ret = get_physics_layer_physics_material(index);
			return true;
		}
	} else if (components.size() >= 2 && components[0].begins_with("terrain_set_") && components[0].trim_prefix("terrain_set_").is_valid_integer()) {
		// Terrains.
		int terrain_set_index = components[0].trim_prefix("terrain_set_").to_int();
		if (terrain_set_index < 0 || terrain_set_index >= terrain_sets.size()) {
			return false;
		}
		if (components[1] == "mode") {
			r_ret = get_terrain_set_mode(terrain_set_index);
			return true;
		} else if (components[1] == "terrains_count") {
			r_ret = get_terrains_count(terrain_set_index);
			return true;
		} else if (components.size() >= 3 && components[1].begins_with("terrain_") && components[1].trim_prefix("terrain_").is_valid_integer()) {
			int terrain_index = components[1].trim_prefix("terrain_").to_int();
			if (terrain_index < 0 || terrain_index >= terrain_sets[terrain_set_index].terrains.size()) {
				return false;
			}
			if (components[2] == "name") {
				r_ret = get_terrain_name(terrain_set_index, terrain_index);
				return true;
			} else if (components[2] == "color") {
				r_ret = get_terrain_color(terrain_set_index, terrain_index);
				return true;
			}
		}
	} else if (components.size() == 2 && components[0].begins_with("navigation_layer_") && components[0].trim_prefix("navigation_layer_").is_valid_integer()) {
		// navigation layers.
		int index = components[0].trim_prefix("navigation_layer_").to_int();
		if (index < 0 || index >= navigation_layers.size()) {
			return false;
		}
		if (components[1] == "layers") {
			r_ret = get_navigation_layer_layers(index);
			return true;
		}
	} else if (components.size() == 2 && components[0].begins_with("custom_data_layer_") && components[0].trim_prefix("custom_data_layer_").is_valid_integer()) {
		// Custom data layers.
		int index = components[0].trim_prefix("custom_data_layer_").to_int();
		if (index < 0 || index >= custom_data_layers.size()) {
			return false;
		}
		if (components[1] == "name") {
			r_ret = get_custom_data_name(index);
			return true;
		} else if (components[1] == "type") {
			r_ret = get_custom_data_type(index);
			return true;
		}
	} else if (components.size() == 2 && components[0] == "sources" && components[1].is_valid_integer()) {
		// Atlases data.
		int source_id = components[1].to_int();

		if (has_source(source_id)) {
			r_ret = get_source(source_id);
			return true;
		} else {
			return false;
		}
	}

	return false;
}

void TileSet::_append_property_list_with_prefix(const StringName &p_name, List<PropertyInfo> *p_to_prepend, List<PropertyInfo> *p_list) {
	for (List<PropertyInfo>::Element *E_property = p_to_prepend->front(); E_property; E_property = E_property->next()) {
		E_property->get().name = vformat("%s/%s", p_name, E_property->get().name);
		p_list->push_back(E_property->get());
	}
}

void TileSet::_get_property_list(List<PropertyInfo> *p_list) const {
	// Rendering.
	p_list->push_back(PropertyInfo(Variant::NIL, "Rendering", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
	for (int i = 0; i < occlusion_layers.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::INT, vformat("occlusion_layer_%d/light_mask", i), PROPERTY_HINT_LAYERS_2D_RENDER));
		p_list->push_back(PropertyInfo(Variant::BOOL, vformat("occlusion_layer_%d/sdf_collision", i)));
	}

	// Physics.
	p_list->push_back(PropertyInfo(Variant::NIL, "Physics", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
	for (int i = 0; i < physics_layers.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::INT, vformat("physics_layer_%d/collision_layer", i), PROPERTY_HINT_LAYERS_2D_PHYSICS));
		p_list->push_back(PropertyInfo(Variant::INT, vformat("physics_layer_%d/collision_mask", i), PROPERTY_HINT_LAYERS_2D_PHYSICS));
		p_list->push_back(PropertyInfo(Variant::OBJECT, vformat("physics_layer_%d/physics_material", i), PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"));
	}

	// Terrains.
	p_list->push_back(PropertyInfo(Variant::NIL, "Terrains", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
	for (int terrain_set_index = 0; terrain_set_index < terrain_sets.size(); terrain_set_index++) {
		p_list->push_back(PropertyInfo(Variant::INT, vformat("terrain_set_%d/mode", terrain_set_index), PROPERTY_HINT_ENUM, "Match corners and sides,Match corners,Match sides"));
		p_list->push_back(PropertyInfo(Variant::INT, vformat("terrain_set_%d/terrains_count", terrain_set_index), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		for (int terrain_index = 0; terrain_index < terrain_sets[terrain_set_index].terrains.size(); terrain_index++) {
			p_list->push_back(PropertyInfo(Variant::STRING, vformat("terrain_set_%d/terrain_%d/name", terrain_set_index, terrain_index)));
			p_list->push_back(PropertyInfo(Variant::COLOR, vformat("terrain_set_%d/terrain_%d/color", terrain_set_index, terrain_index)));
		}
	}

	// Navigation.
	p_list->push_back(PropertyInfo(Variant::NIL, "Navigation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
	for (int i = 0; i < navigation_layers.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::INT, vformat("navigation_layer_%d/layers", i), PROPERTY_HINT_LAYERS_2D_NAVIGATION));
	}

	// Custom data.
	String argt = "Any";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "Custom data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
	for (int i = 0; i < custom_data_layers.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::STRING, vformat("custom_data_layer_%d/name", i)));
		p_list->push_back(PropertyInfo(Variant::INT, vformat("custom_data_layer_%d/type", i), PROPERTY_HINT_ENUM, argt));
	}

	// Sources.
	// Note: sources have to be listed in at the end as some TileData rely on the TileSet properties being initialized first.
	for (Map<int, Ref<TileSetSource>>::Element *E_source = sources.front(); E_source; E_source = E_source->next()) {
		// Add a dummy property to show the tile exists.
		p_list->push_back(PropertyInfo(Variant::INT, vformat("sources/%d", E_source->key()), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	}
}

void TileSet::_bind_methods() {
	// Sources management.
	ClassDB::bind_method(D_METHOD("get_next_source_id"), &TileSet::get_next_source_id);
	ClassDB::bind_method(D_METHOD("add_source", "atlas_source_id_override"), &TileSet::add_source, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("remove_source", "source_id"), &TileSet::remove_source);
	ClassDB::bind_method(D_METHOD("set_source_id", "source_id"), &TileSet::set_source_id);
	ClassDB::bind_method(D_METHOD("get_source_count"), &TileSet::get_source_count);
	ClassDB::bind_method(D_METHOD("get_source_id", "index"), &TileSet::get_source_id);
	ClassDB::bind_method(D_METHOD("has_source", "index"), &TileSet::has_source);
	ClassDB::bind_method(D_METHOD("get_source", "index"), &TileSet::get_source);

	// Shape and layout.
	ClassDB::bind_method(D_METHOD("set_tile_shape", "shape"), &TileSet::set_tile_shape);
	ClassDB::bind_method(D_METHOD("get_tile_shape"), &TileSet::get_tile_shape);
	ClassDB::bind_method(D_METHOD("set_tile_layout", "layout"), &TileSet::set_tile_layout);
	ClassDB::bind_method(D_METHOD("get_tile_layout"), &TileSet::get_tile_layout);
	ClassDB::bind_method(D_METHOD("set_tile_offset_axis", "alignment"), &TileSet::set_tile_offset_axis);
	ClassDB::bind_method(D_METHOD("get_tile_offset_axis"), &TileSet::get_tile_offset_axis);
	ClassDB::bind_method(D_METHOD("set_tile_size", "size"), &TileSet::set_tile_size);
	ClassDB::bind_method(D_METHOD("get_tile_size"), &TileSet::get_tile_size);
	ClassDB::bind_method(D_METHOD("set_tile_skew", "skew"), &TileSet::set_tile_skew);
	ClassDB::bind_method(D_METHOD("get_tile_skew"), &TileSet::get_tile_skew);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "tile_shape", PROPERTY_HINT_ENUM, "Square,Isometric,Half-offset square,Hexagon"), "set_tile_shape", "get_tile_shape");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tile_layout", PROPERTY_HINT_ENUM, "Stacked,Stacked Offset,Stairs Right,Stairs Down,Diamond Right,Diamond Down"), "set_tile_layout", "get_tile_layout");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tile_offset_axis", PROPERTY_HINT_ENUM, "Horizontal Offset,Vertical Offset"), "set_tile_offset_axis", "get_tile_offset_axis");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "tile_size"), "set_tile_size", "get_tile_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "tile_skew"), "set_tile_skew", "get_tile_skew");

	// Rendering.
	ClassDB::bind_method(D_METHOD("set_uv_clipping", "uv_clipping"), &TileSet::set_uv_clipping);
	ClassDB::bind_method(D_METHOD("is_uv_clipping"), &TileSet::is_uv_clipping);
	ClassDB::bind_method(D_METHOD("set_y_sorting", "y_sorting"), &TileSet::set_y_sorting);
	ClassDB::bind_method(D_METHOD("is_y_sorting"), &TileSet::is_y_sorting);

	ClassDB::bind_method(D_METHOD("set_occlusion_layers_count", "occlusion_layers_count"), &TileSet::set_occlusion_layers_count);
	ClassDB::bind_method(D_METHOD("get_occlusion_layers_count"), &TileSet::get_occlusion_layers_count);
	ClassDB::bind_method(D_METHOD("set_occlusion_layer_light_mask", "layer_index", "light_mask"), &TileSet::set_occlusion_layer_light_mask);
	ClassDB::bind_method(D_METHOD("get_occlusion_layer_light_mask"), &TileSet::get_occlusion_layer_light_mask);
	ClassDB::bind_method(D_METHOD("set_occlusion_layer_sdf_collision", "layer_index", "sdf_collision"), &TileSet::set_occlusion_layer_sdf_collision);
	ClassDB::bind_method(D_METHOD("get_occlusion_layer_sdf_collision"), &TileSet::get_occlusion_layer_sdf_collision);

	// Physics
	ClassDB::bind_method(D_METHOD("set_physics_layers_count", "physics_layers_count"), &TileSet::set_physics_layers_count);
	ClassDB::bind_method(D_METHOD("get_physics_layers_count"), &TileSet::get_physics_layers_count);
	ClassDB::bind_method(D_METHOD("set_physics_layer_collision_layer", "layer_index", "layer"), &TileSet::set_physics_layer_collision_layer);
	ClassDB::bind_method(D_METHOD("get_physics_layer_collision_layer", "layer_index"), &TileSet::get_physics_layer_collision_layer);
	ClassDB::bind_method(D_METHOD("set_physics_layer_collision_mask", "layer_index", "mask"), &TileSet::set_physics_layer_collision_mask);
	ClassDB::bind_method(D_METHOD("get_physics_layer_collision_mask", "layer_index"), &TileSet::get_physics_layer_collision_mask);
	ClassDB::bind_method(D_METHOD("set_physics_layer_physics_material", "layer_index", "physics_material"), &TileSet::set_physics_layer_physics_material);
	ClassDB::bind_method(D_METHOD("get_physics_layer_physics_material", "layer_index"), &TileSet::get_physics_layer_physics_material);

	// Terrains
	ClassDB::bind_method(D_METHOD("set_terrain_sets_count", "terrain_sets_count"), &TileSet::set_terrain_sets_count);
	ClassDB::bind_method(D_METHOD("get_terrain_sets_count"), &TileSet::get_terrain_sets_count);
	ClassDB::bind_method(D_METHOD("set_terrain_set_mode", "terrain_set", "mode"), &TileSet::set_terrain_set_mode);
	ClassDB::bind_method(D_METHOD("get_terrain_set_mode", "terrain_set"), &TileSet::get_terrain_set_mode);

	ClassDB::bind_method(D_METHOD("set_terrains_count", "terrain_set", "terrains_count"), &TileSet::set_terrains_count);
	ClassDB::bind_method(D_METHOD("get_terrains_count", "terrain_set"), &TileSet::get_terrains_count);
	ClassDB::bind_method(D_METHOD("set_terrain_name", "terrain_set", "terrain_index", "name"), &TileSet::set_terrain_name);
	ClassDB::bind_method(D_METHOD("get_terrain_name", "terrain_set", "terrain_index"), &TileSet::get_terrain_name);
	ClassDB::bind_method(D_METHOD("set_terrain_color", "terrain_set", "terrain_index", "color"), &TileSet::set_terrain_color);
	ClassDB::bind_method(D_METHOD("get_terrain_color", "terrain_set", "terrain_index"), &TileSet::get_terrain_color);

	// Navigation
	ClassDB::bind_method(D_METHOD("set_navigation_layers_count", "navigation_layers_count"), &TileSet::set_navigation_layers_count);
	ClassDB::bind_method(D_METHOD("get_navigation_layers_count"), &TileSet::get_navigation_layers_count);
	ClassDB::bind_method(D_METHOD("set_navigation_layer_layers", "layer_index", "layers"), &TileSet::set_navigation_layer_layers);
	ClassDB::bind_method(D_METHOD("get_navigation_layer_layers", "layer_index"), &TileSet::get_navigation_layer_layers);

	// Custom data
	ClassDB::bind_method(D_METHOD("set_custom_data_layers_count", "custom_data_layers_count"), &TileSet::set_custom_data_layers_count);
	ClassDB::bind_method(D_METHOD("get_custom_data_layers_count"), &TileSet::get_custom_data_layers_count);

	ADD_GROUP("Rendering", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "uv_clipping"), "set_uv_clipping", "is_uv_clipping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "y_sorting"), "set_y_sorting", "is_y_sorting");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "occlusion_layers_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_occlusion_layers_count", "get_occlusion_layers_count");

	ADD_GROUP("Physics", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "physics_layers_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_physics_layers_count", "get_physics_layers_count");

	ADD_GROUP("Terrains", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "terrains_sets_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_terrain_sets_count", "get_terrain_sets_count");

	ADD_GROUP("Navigation", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_layers_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_navigation_layers_count", "get_navigation_layers_count");

	ADD_GROUP("Custom data", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "custom_data_layers_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_custom_data_layers_count", "get_custom_data_layers_count");

	// -- Enum binding --
	BIND_ENUM_CONSTANT(TILE_SHAPE_SQUARE);
	BIND_ENUM_CONSTANT(TILE_SHAPE_ISOMETRIC);
	BIND_ENUM_CONSTANT(TILE_SHAPE_HALF_OFFSET_SQUARE);
	BIND_ENUM_CONSTANT(TILE_SHAPE_HEXAGON);

	BIND_ENUM_CONSTANT(TILE_LAYOUT_STACKED);
	BIND_ENUM_CONSTANT(TILE_LAYOUT_STACKED_OFFSET);
	BIND_ENUM_CONSTANT(TILE_LAYOUT_STAIRS_RIGHT);
	BIND_ENUM_CONSTANT(TILE_LAYOUT_STAIRS_DOWN);
	BIND_ENUM_CONSTANT(TILE_LAYOUT_DIAMOND_RIGHT);
	BIND_ENUM_CONSTANT(TILE_LAYOUT_DIAMOND_DOWN);

	BIND_ENUM_CONSTANT(TILE_OFFSET_AXIS_HORIZONTAL);
	BIND_ENUM_CONSTANT(TILE_OFFSET_AXIS_VERTICAL);

	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_RIGHT_SIDE);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_RIGHT_CORNER);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_BOTTOM_SIDE);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_BOTTOM_CORNER);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_LEFT_SIDE);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_LEFT_CORNER);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_TOP_SIDE);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_TOP_CORNER);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
	BIND_ENUM_CONSTANT(TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER);

	BIND_ENUM_CONSTANT(TERRAIN_MODE_MATCH_CORNERS_AND_SIDES);
	BIND_ENUM_CONSTANT(TERRAIN_MODE_MATCH_CORNERS);
	BIND_ENUM_CONSTANT(TERRAIN_MODE_MATCH_SIDES);
}

TileSet::TileSet() {
	// Instanciatie and list all plugins.
	tile_set_plugins_vector.append(memnew(TileSetAtlasPluginRendering));
	tile_set_plugins_vector.append(memnew(TileSetAtlasPluginPhysics));
	tile_set_plugins_vector.append(memnew(TileSetAtlasPluginTerrain));
	tile_set_plugins_vector.append(memnew(TileSetAtlasPluginNavigation));
}

TileSet::~TileSet() {
	for (Map<int, CompatibilityTileData *>::Element *E = compatibility_data.front(); E; E = E->next()) {
		memdelete(E->get());
	}
	while (!source_ids.is_empty()) {
		remove_source(source_ids[0]);
	}
	for (int i = 0; i < tile_set_plugins_vector.size(); i++) {
		memdelete(tile_set_plugins_vector[i]);
	}
}
