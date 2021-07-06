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

#include "core/core_string_names.h"
#include "core/math/geometry_2d.h"
#include "core/templates/local_vector.h"

#include "scene/2d/navigation_region_2d.h"
#include "scene/gui/control.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "servers/navigation_server_2d.h"

/////////////////////////////// TileSet //////////////////////////////////////

const int TileSet::INVALID_SOURCE = -1;

const char *TileSet::CELL_NEIGHBOR_ENUM_TO_TEXT[] = {
	"right_side",
	"right_corner",
	"bottom_right_side",
	"bottom_right_corner",
	"bottom_side",
	"bottom_corner",
	"bottom_left_side",
	"bottom_left_corner",
	"left_side",
	"left_corner",
	"top_left_side",
	"top_left_corner",
	"top_side",
	"top_corner",
	"top_right_side",
	"top_right_corner"
};

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

	terrain_bits_meshes_dirty = true;
	tile_meshes_dirty = true;
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

	terrain_bits_meshes_dirty = true;
	tile_meshes_dirty = true;
	emit_changed();
}
TileSet::TileOffsetAxis TileSet::get_tile_offset_axis() const {
	return tile_offset_axis;
}

void TileSet::set_tile_size(Size2i p_size) {
	ERR_FAIL_COND(p_size.x < 1 || p_size.y < 1);
	tile_size = p_size;
	terrain_bits_meshes_dirty = true;
	tile_meshes_dirty = true;
	emit_changed();
}
Size2i TileSet::get_tile_size() const {
	return tile_size;
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
int TileSet::add_source(Ref<TileSetSource> p_tile_set_source, int p_atlas_source_id_override) {
	ERR_FAIL_COND_V(!p_tile_set_source.is_valid(), TileSet::INVALID_SOURCE);
	ERR_FAIL_COND_V_MSG(p_atlas_source_id_override >= 0 && (sources.has(p_atlas_source_id_override)), TileSet::INVALID_SOURCE, vformat("Cannot create TileSet atlas source. Another atlas source exists with id %d.", p_atlas_source_id_override));

	int new_source_id = p_atlas_source_id_override >= 0 ? p_atlas_source_id_override : next_source_id;
	sources[new_source_id] = p_tile_set_source;
	source_ids.append(new_source_id);
	source_ids.sort();
	p_tile_set_source->set_tile_set(this);
	_compute_next_source_id();

	sources[new_source_id]->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &TileSet::_source_changed));

	emit_changed();

	return new_source_id;
}

void TileSet::remove_source(int p_source_id) {
	ERR_FAIL_COND_MSG(!sources.has(p_source_id), vformat("Cannot remove TileSet atlas source. No tileset atlas source with id %d.", p_source_id));

	sources[p_source_id]->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &TileSet::_source_changed));

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
	ERR_FAIL_INDEX_V(p_index, source_ids.size(), TileSet::INVALID_SOURCE);
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
	ERR_FAIL_INDEX_V(p_terrain_set, terrain_sets.size(), Color());
	ERR_FAIL_INDEX_V(p_terrain_index, terrain_sets[p_terrain_set].terrains.size(), Color());
	return terrain_sets[p_terrain_set].terrains[p_terrain_index].color;
}

bool TileSet::is_valid_peering_bit_for_mode(TileSet::TerrainMode p_terrain_mode, TileSet::CellNeighbor p_peering_bit) const {
	if (tile_shape == TileSet::TILE_SHAPE_SQUARE) {
		if (p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_SIDES) {
			if (p_peering_bit == TileSet::CELL_NEIGHBOR_RIGHT_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_LEFT_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_SIDE) {
				return true;
			}
		}
		if (p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
			if (p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER) {
				return true;
			}
		}
	} else if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		if (p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_SIDES) {
			if (p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
				return true;
			}
		}
		if (p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
			if (p_peering_bit == TileSet::CELL_NEIGHBOR_RIGHT_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_LEFT_CORNER ||
					p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
				return true;
			}
		}
	} else {
		if (get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			if (p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_SIDES) {
				if (p_peering_bit == TileSet::CELL_NEIGHBOR_RIGHT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_LEFT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
					return true;
				}
			}
			if (p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
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
			if (p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_SIDES) {
				if (p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_SIDE ||
						p_peering_bit == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
					return true;
				}
			}
			if (p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES || p_terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
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

bool TileSet::is_valid_peering_bit_terrain(int p_terrain_set, TileSet::CellNeighbor p_peering_bit) const {
	if (p_terrain_set < 0 || p_terrain_set >= get_terrain_sets_count()) {
		return false;
	}

	TileSet::TerrainMode terrain_mode = get_terrain_set_mode(p_terrain_set);
	return is_valid_peering_bit_for_mode(terrain_mode, p_peering_bit);
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

void TileSet::set_source_level_tile_proxy(int p_source_from, int p_source_to) {
	ERR_FAIL_COND(p_source_from == TileSet::INVALID_SOURCE || p_source_to == TileSet::INVALID_SOURCE);

	source_level_proxies[p_source_from] = p_source_to;

	emit_changed();
}

int TileSet::get_source_level_tile_proxy(int p_source_from) {
	ERR_FAIL_COND_V(!source_level_proxies.has(p_source_from), TileSet::INVALID_SOURCE);

	return source_level_proxies[p_source_from];
}

bool TileSet::has_source_level_tile_proxy(int p_source_from) {
	return source_level_proxies.has(p_source_from);
}

void TileSet::remove_source_level_tile_proxy(int p_source_from) {
	ERR_FAIL_COND(!source_level_proxies.has(p_source_from));

	source_level_proxies.erase(p_source_from);

	emit_changed();
}

void TileSet::set_coords_level_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_source_to, Vector2i p_coords_to) {
	ERR_FAIL_COND(p_source_from == TileSet::INVALID_SOURCE || p_source_to == TileSet::INVALID_SOURCE);
	ERR_FAIL_COND(p_coords_from == TileSetSource::INVALID_ATLAS_COORDS || p_coords_to == TileSetSource::INVALID_ATLAS_COORDS);

	Array from;
	from.push_back(p_source_from);
	from.push_back(p_coords_from);

	Array to;
	to.push_back(p_source_to);
	to.push_back(p_coords_to);

	coords_level_proxies[from] = to;

	emit_changed();
}

Array TileSet::get_coords_level_tile_proxy(int p_source_from, Vector2i p_coords_from) {
	Array from;
	from.push_back(p_source_from);
	from.push_back(p_coords_from);

	ERR_FAIL_COND_V(!coords_level_proxies.has(from), Array());

	return coords_level_proxies[from];
}

bool TileSet::has_coords_level_tile_proxy(int p_source_from, Vector2i p_coords_from) {
	Array from;
	from.push_back(p_source_from);
	from.push_back(p_coords_from);

	return coords_level_proxies.has(from);
}

void TileSet::remove_coords_level_tile_proxy(int p_source_from, Vector2i p_coords_from) {
	Array from;
	from.push_back(p_source_from);
	from.push_back(p_coords_from);

	ERR_FAIL_COND(!coords_level_proxies.has(from));

	coords_level_proxies.erase(from);

	emit_changed();
}

void TileSet::set_alternative_level_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_alternative_from, int p_source_to, Vector2i p_coords_to, int p_alternative_to) {
	ERR_FAIL_COND(p_source_from == TileSet::INVALID_SOURCE || p_source_to == TileSet::INVALID_SOURCE);
	ERR_FAIL_COND(p_coords_from == TileSetSource::INVALID_ATLAS_COORDS || p_coords_to == TileSetSource::INVALID_ATLAS_COORDS);

	Array from;
	from.push_back(p_source_from);
	from.push_back(p_coords_from);
	from.push_back(p_alternative_from);

	Array to;
	to.push_back(p_source_to);
	to.push_back(p_coords_to);
	to.push_back(p_alternative_to);

	alternative_level_proxies[from] = to;

	emit_changed();
}

Array TileSet::get_alternative_level_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_alternative_from) {
	Array from;
	from.push_back(p_source_from);
	from.push_back(p_coords_from);
	from.push_back(p_alternative_from);

	ERR_FAIL_COND_V(!alternative_level_proxies.has(from), Array());

	return alternative_level_proxies[from];
}

bool TileSet::has_alternative_level_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_alternative_from) {
	Array from;
	from.push_back(p_source_from);
	from.push_back(p_coords_from);
	from.push_back(p_alternative_from);

	return alternative_level_proxies.has(from);
}

void TileSet::remove_alternative_level_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_alternative_from) {
	Array from;
	from.push_back(p_source_from);
	from.push_back(p_coords_from);
	from.push_back(p_alternative_from);

	ERR_FAIL_COND(!alternative_level_proxies.has(from));

	alternative_level_proxies.erase(from);

	emit_changed();
}

Array TileSet::get_source_level_tile_proxies() const {
	Array output;
	for (Map<int, int>::Element *E = source_level_proxies.front(); E; E = E->next()) {
		Array proxy;
		proxy.push_back(E->key());
		proxy.push_back(E->get());
		output.push_back(proxy);
	}
	return output;
}

Array TileSet::get_coords_level_tile_proxies() const {
	Array output;
	for (Map<Array, Array>::Element *E = coords_level_proxies.front(); E; E = E->next()) {
		Array proxy;
		proxy.append_array(E->key());
		proxy.append_array(E->get());
		output.push_back(proxy);
	}
	return output;
}

Array TileSet::get_alternative_level_tile_proxies() const {
	Array output;
	for (Map<Array, Array>::Element *E = alternative_level_proxies.front(); E; E = E->next()) {
		Array proxy;
		proxy.append_array(E->key());
		proxy.append_array(E->get());
		output.push_back(proxy);
	}
	return output;
}

Array TileSet::map_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_alternative_from) const {
	Array from;
	from.push_back(p_source_from);
	from.push_back(p_coords_from);
	from.push_back(p_alternative_from);

	// Check if the tile is valid, and if so, don't map the tile and return the input.
	if (has_source(p_source_from)) {
		Ref<TileSetSource> source = get_source(p_source_from);
		if (source->has_tile(p_coords_from) && source->has_alternative_tile(p_coords_from, p_alternative_from)) {
			return from;
		}
	}

	// Source, coords and alternative match.
	if (alternative_level_proxies.has(from)) {
		return alternative_level_proxies[from].duplicate();
	}

	// Source and coords match.
	from.pop_back();
	if (coords_level_proxies.has(from)) {
		Array output = coords_level_proxies[from].duplicate();
		output.push_back(p_alternative_from);
		return output;
	}

	// Source matches.
	if (source_level_proxies.has(p_source_from)) {
		Array output;
		output.push_back(source_level_proxies[p_source_from]);
		output.push_back(p_coords_from);
		output.push_back(p_alternative_from);
		return output;
	}

	Array output;
	output.push_back(p_source_from);
	output.push_back(p_coords_from);
	output.push_back(p_alternative_from);
	return output;
}

void TileSet::cleanup_invalid_tile_proxies() {
	// Source level.
	Vector<int> source_to_remove;
	for (Map<int, int>::Element *E = source_level_proxies.front(); E; E = E->next()) {
		if (has_source(E->key())) {
			source_to_remove.append(E->key());
		}
	}
	for (int i = 0; i < source_to_remove.size(); i++) {
		remove_source_level_tile_proxy(source_to_remove[i]);
	}

	// Coords level.
	Vector<Array> coords_to_remove;
	for (Map<Array, Array>::Element *E = coords_level_proxies.front(); E; E = E->next()) {
		Array a = E->key();
		if (has_source(a[0]) && get_source(a[0])->has_tile(a[1])) {
			coords_to_remove.append(a);
		}
	}
	for (int i = 0; i < coords_to_remove.size(); i++) {
		Array a = coords_to_remove[i];
		remove_coords_level_tile_proxy(a[0], a[1]);
	}

	// Alternative level.
	Vector<Array> alternative_to_remove;
	for (Map<Array, Array>::Element *E = alternative_level_proxies.front(); E; E = E->next()) {
		Array a = E->key();
		if (has_source(a[0]) && get_source(a[0])->has_tile(a[1]) && get_source(a[0])->has_alternative_tile(a[1], a[2])) {
			alternative_to_remove.append(a);
		}
	}
	for (int i = 0; i < alternative_to_remove.size(); i++) {
		Array a = alternative_to_remove[i];
		remove_alternative_level_tile_proxy(a[0], a[1], a[2]);
	}
}

void TileSet::clear_tile_proxies() {
	source_level_proxies.clear();
	coords_level_proxies.clear();
	alternative_level_proxies.clear();

	emit_changed();
}

Vector<Vector2> TileSet::get_tile_shape_polygon() {
	Vector<Vector2> points;
	if (tile_shape == TileSet::TILE_SHAPE_SQUARE) {
		points.append(Vector2(0.0, 0.0));
		points.append(Vector2(1.0, 0.0));
		points.append(Vector2(1.0, 1.0));
		points.append(Vector2(0.0, 1.0));
	} else {
		float overlap = 0.0;
		switch (tile_shape) {
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

		points.append(Vector2(0.5, 0.0));
		points.append(Vector2(0.0, overlap));
		points.append(Vector2(0.0, 1.0 - overlap));
		points.append(Vector2(0.5, 1.0));
		points.append(Vector2(1.0, 1.0 - overlap));
		points.append(Vector2(1.0, overlap));
		points.append(Vector2(0.5, 0.0));
		if (get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
			for (int i = 0; i < points.size(); i++) {
				points.write[i] = Vector2(points[i].y, points[i].x);
			}
		}
	}
	for (int i = 0; i < points.size(); i++) {
		points.write[i] = points[i] * tile_size - tile_size / 2;
	}
	return points;
}

void TileSet::draw_tile_shape(CanvasItem *p_canvas_item, Rect2 p_region, Color p_color, bool p_filled, Ref<Texture2D> p_texture) {
	if (tile_meshes_dirty) {
		Vector<Vector2> uvs = get_tile_shape_polygon();
		for (int i = 0; i < uvs.size(); i++) {
			uvs.write[i] = (uvs[i] + tile_size / 2) / tile_size;
		}

		Vector<Color> colors;
		colors.resize(uvs.size());
		colors.fill(Color(1.0, 1.0, 1.0, 1.0));

		// Filled mesh.
		tile_filled_mesh->clear_surfaces();
		Array a;
		a.resize(Mesh::ARRAY_MAX);
		a[Mesh::ARRAY_VERTEX] = uvs;
		a[Mesh::ARRAY_TEX_UV] = uvs;
		a[Mesh::ARRAY_COLOR] = colors;
		a[Mesh::ARRAY_INDEX] = Geometry2D::triangulate_polygon(uvs);
		tile_filled_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, a, Array(), Dictionary(), Mesh::ARRAY_FLAG_USE_2D_VERTICES);

		// Lines mesh.
		tile_lines_mesh->clear_surfaces();
		a.clear();
		a.resize(Mesh::ARRAY_MAX);
		// Add the first point again when drawing lines.
		uvs.push_back(uvs[0]);
		colors.push_back(colors[0]);
		a[Mesh::ARRAY_VERTEX] = uvs;
		a[Mesh::ARRAY_COLOR] = colors;
		tile_lines_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINE_STRIP, a, Array(), Dictionary(), Mesh::ARRAY_FLAG_USE_2D_VERTICES);

		tile_meshes_dirty = false;
	}

	Transform2D xform;
	xform.scale(p_region.size);
	xform.set_origin(p_region.get_position());
	if (p_filled) {
		p_canvas_item->draw_mesh(tile_filled_mesh, p_texture, xform, p_color);
	} else {
		p_canvas_item->draw_mesh(tile_lines_mesh, Ref<Texture2D>(), xform, p_color);
	}
}

Vector<Point2> TileSet::get_terrain_bit_polygon(int p_terrain_set, TileSet::CellNeighbor p_bit) {
	ERR_FAIL_COND_V(p_terrain_set < 0 || p_terrain_set >= get_terrain_sets_count(), Vector<Point2>());

	TileSet::TerrainMode terrain_mode = get_terrain_set_mode(p_terrain_set);

	if (tile_shape == TileSet::TILE_SHAPE_SQUARE) {
		if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
			return _get_square_corner_or_side_terrain_bit_polygon(tile_size, p_bit);
		} else if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
			return _get_square_corner_terrain_bit_polygon(tile_size, p_bit);
		} else { // TileData::TERRAIN_MODE_MATCH_SIDES
			return _get_square_side_terrain_bit_polygon(tile_size, p_bit);
		}
	} else if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
			return _get_isometric_corner_or_side_terrain_bit_polygon(tile_size, p_bit);
		} else if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
			return _get_isometric_corner_terrain_bit_polygon(tile_size, p_bit);
		} else { // TileData::TERRAIN_MODE_MATCH_SIDES
			return _get_isometric_side_terrain_bit_polygon(tile_size, p_bit);
		}
	} else {
		float overlap = 0.0;
		switch (tile_shape) {
			case TileSet::TILE_SHAPE_HEXAGON:
				overlap = 0.25;
				break;
			case TileSet::TILE_SHAPE_HALF_OFFSET_SQUARE:
				overlap = 0.0;
				break;
			default:
				break;
		}
		if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
			return _get_half_offset_corner_or_side_terrain_bit_polygon(tile_size, p_bit, overlap, tile_offset_axis);
		} else if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
			return _get_half_offset_corner_terrain_bit_polygon(tile_size, p_bit, overlap, tile_offset_axis);
		} else { // TileData::TERRAIN_MODE_MATCH_SIDES
			return _get_half_offset_side_terrain_bit_polygon(tile_size, p_bit, overlap, tile_offset_axis);
		}
	}
}

#define TERRAIN_ALPHA 0.6

void TileSet::draw_terrains(CanvasItem *p_canvas_item, Transform2D p_transform, const TileData *p_tile_data) {
	ERR_FAIL_COND(!p_tile_data);

	if (terrain_bits_meshes_dirty) {
		// Recompute the meshes.
		terrain_bits_meshes.clear();

		for (int terrain_mode_index = 0; terrain_mode_index < 3; terrain_mode_index++) {
			TerrainMode terrain_mode = TerrainMode(terrain_mode_index);
			for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
				CellNeighbor bit = CellNeighbor(i);

				if (is_valid_peering_bit_for_mode(terrain_mode, bit)) {
					Vector<Vector2> polygon;
					if (tile_shape == TileSet::TILE_SHAPE_SQUARE) {
						if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
							polygon = _get_square_corner_or_side_terrain_bit_polygon(tile_size, bit);
						} else if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
							polygon = _get_square_corner_terrain_bit_polygon(tile_size, bit);
						} else { // TileData::TERRAIN_MODE_MATCH_SIDES
							polygon = _get_square_side_terrain_bit_polygon(tile_size, bit);
						}
					} else if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
						if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
							polygon = _get_isometric_corner_or_side_terrain_bit_polygon(tile_size, bit);
						} else if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
							polygon = _get_isometric_corner_terrain_bit_polygon(tile_size, bit);
						} else { // TileData::TERRAIN_MODE_MATCH_SIDES
							polygon = _get_isometric_side_terrain_bit_polygon(tile_size, bit);
						}
					} else {
						float overlap = 0.0;
						switch (tile_shape) {
							case TileSet::TILE_SHAPE_HEXAGON:
								overlap = 0.25;
								break;
							case TileSet::TILE_SHAPE_HALF_OFFSET_SQUARE:
								overlap = 0.0;
								break;
							default:
								break;
						}
						if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
							polygon = _get_half_offset_corner_or_side_terrain_bit_polygon(tile_size, bit, overlap, tile_offset_axis);
						} else if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
							polygon = _get_half_offset_corner_terrain_bit_polygon(tile_size, bit, overlap, tile_offset_axis);
						} else { // TileData::TERRAIN_MODE_MATCH_SIDES
							polygon = _get_half_offset_side_terrain_bit_polygon(tile_size, bit, overlap, tile_offset_axis);
						}
					}

					Ref<ArrayMesh> mesh;
					mesh.instantiate();
					Vector<Vector2> uvs;
					uvs.resize(polygon.size());
					Vector<Color> colors;
					colors.resize(polygon.size());
					colors.fill(Color(1.0, 1.0, 1.0, 1.0));
					Array a;
					a.resize(Mesh::ARRAY_MAX);
					a[Mesh::ARRAY_VERTEX] = polygon;
					a[Mesh::ARRAY_TEX_UV] = uvs;
					a[Mesh::ARRAY_COLOR] = colors;
					a[Mesh::ARRAY_INDEX] = Geometry2D::triangulate_polygon(polygon);
					mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, a, Array(), Dictionary(), Mesh::ARRAY_FLAG_USE_2D_VERTICES);
					terrain_bits_meshes[terrain_mode][bit] = mesh;
				}
			}
		}
		terrain_bits_meshes_dirty = false;
	}

	int terrain_set = p_tile_data->get_terrain_set();
	if (terrain_set < 0) {
		return;
	}
	TileSet::TerrainMode terrain_mode = get_terrain_set_mode(terrain_set);

	RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), p_transform);
	for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
		CellNeighbor bit = CellNeighbor(i);
		if (is_valid_peering_bit_terrain(terrain_set, bit)) {
			int terrain_id = p_tile_data->get_peering_bit_terrain(bit);
			if (terrain_id >= 0) {
				Color color = get_terrain_color(terrain_set, terrain_id);
				color.a = TERRAIN_ALPHA;
				p_canvas_item->draw_mesh(terrain_bits_meshes[terrain_mode][bit], Ref<Texture2D>(), Transform2D(), color);
			}
		}
	}
	RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), Transform2D());
}

Vector<Vector<Ref<Texture2D>>> TileSet::generate_terrains_icons(Size2i p_size) {
	// Counts the number of matching terrain tiles and find the best matching icon.
	struct Count {
		int count = 0;
		float probability = 0.0;
		Ref<Texture2D> texture;
		Rect2i region;
	};
	Vector<Vector<Ref<Texture2D>>> output;
	LocalVector<LocalVector<Count>> counts;
	output.resize(get_terrain_sets_count());
	counts.resize(get_terrain_sets_count());
	for (int terrain_set = 0; terrain_set < get_terrain_sets_count(); terrain_set++) {
		output.write[terrain_set].resize(get_terrains_count(terrain_set));
		counts[terrain_set].resize(get_terrains_count(terrain_set));
	}

	for (int source_index = 0; source_index < get_source_count(); source_index++) {
		int source_id = get_source_id(source_index);
		Ref<TileSetSource> source = get_source(source_id);

		Ref<TileSetAtlasSource> atlas_source = source;
		if (atlas_source.is_valid()) {
			for (int tile_index = 0; tile_index < source->get_tiles_count(); tile_index++) {
				Vector2i tile_id = source->get_tile_id(tile_index);
				for (int alternative_index = 0; alternative_index < source->get_alternative_tiles_count(tile_id); alternative_index++) {
					int alternative_id = source->get_alternative_tile_id(tile_id, alternative_index);

					TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(tile_id, alternative_id));
					int terrain_set = tile_data->get_terrain_set();
					if (terrain_set >= 0) {
						ERR_FAIL_INDEX_V(terrain_set, get_terrain_sets_count(), Vector<Vector<Ref<Texture2D>>>());

						LocalVector<int> bit_counts;
						bit_counts.resize(get_terrains_count(terrain_set));
						for (int terrain = 0; terrain < get_terrains_count(terrain_set); terrain++) {
							bit_counts[terrain] = 0;
						}
						for (int terrain_bit = 0; terrain_bit < TileSet::CELL_NEIGHBOR_MAX; terrain_bit++) {
							TileSet::CellNeighbor cell_neighbor = TileSet::CellNeighbor(terrain_bit);
							if (is_valid_peering_bit_terrain(terrain_set, cell_neighbor)) {
								int terrain = tile_data->get_peering_bit_terrain(cell_neighbor);
								if (terrain >= 0) {
									bit_counts[terrain] += 1;
								}
							}
						}

						for (int terrain = 0; terrain < get_terrains_count(terrain_set); terrain++) {
							if ((bit_counts[terrain] > counts[terrain_set][terrain].count) || (bit_counts[terrain] == counts[terrain_set][terrain].count && tile_data->get_probability() > counts[terrain_set][terrain].probability)) {
								counts[terrain_set][terrain].count = bit_counts[terrain];
								counts[terrain_set][terrain].probability = tile_data->get_probability();
								counts[terrain_set][terrain].texture = atlas_source->get_texture();
								counts[terrain_set][terrain].region = atlas_source->get_tile_texture_region(tile_id);
							}
						}
					}
				}
			}
		}
	}

	// Generate the icons.
	for (int terrain_set = 0; terrain_set < get_terrain_sets_count(); terrain_set++) {
		for (int terrain = 0; terrain < get_terrains_count(terrain_set); terrain++) {
			Ref<Image> image;
			image.instantiate();
			if (counts[terrain_set][terrain].count > 0) {
				// Get the best tile.
				Ref<Texture2D> texture = counts[terrain_set][terrain].texture;
				Rect2 region = counts[terrain_set][terrain].region;
				image->create(region.size.x, region.size.y, false, Image::FORMAT_RGBA8);
				image->blit_rect(texture->get_image(), region, Point2());
				image->resize(p_size.x, p_size.y, Image::INTERPOLATE_NEAREST);
			} else {
				image->create(1, 1, false, Image::FORMAT_RGBA8);
				image->set_pixel(0, 0, get_terrain_color(terrain_set, terrain));
			}
			Ref<ImageTexture> icon;
			icon.instantiate();
			icon->create_from_image(image);
			icon->set_size_override(p_size);

			output.write[terrain_set].write[terrain] = icon;
		}
	}
	return output;
}

void TileSet::_source_changed() {
	emit_changed();
}

Vector<Point2> TileSet::_get_square_corner_or_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit) {
	Rect2 bit_rect;
	bit_rect.size = Vector2(p_size) / 3;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
			bit_rect.position = Vector2(1, -1);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
			bit_rect.position = Vector2(1, 1);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
			bit_rect.position = Vector2(-1, 1);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
			bit_rect.position = Vector2(-3, 1);
			break;
		case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
			bit_rect.position = Vector2(-3, -1);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
			bit_rect.position = Vector2(-3, -3);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_SIDE:
			bit_rect.position = Vector2(-1, -3);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
			bit_rect.position = Vector2(1, -3);
			break;
		default:
			break;
	}
	bit_rect.position *= Vector2(p_size) / 6.0;
	Vector<Vector2> polygon;
	polygon.push_back(bit_rect.position);
	polygon.push_back(Vector2(bit_rect.get_end().x, bit_rect.position.y));
	polygon.push_back(bit_rect.get_end());
	polygon.push_back(Vector2(bit_rect.position.x, bit_rect.get_end().y));
	return polygon;
}

Vector<Point2> TileSet::_get_square_corner_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit) {
	Vector2 unit = Vector2(p_size) / 6.0;
	Vector<Vector2> polygon;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(3, 3) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(1, 1) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(-3, 3) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-1, 1) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(-3, -3) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-1, -1) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(3, -3) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(1, -1) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		default:
			break;
	}
	return polygon;
}

Vector<Point2> TileSet::_get_square_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit) {
	Vector2 unit = Vector2(p_size) / 6.0;
	Vector<Vector2> polygon;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
			polygon.push_back(Vector2(1, -1) * unit);
			polygon.push_back(Vector2(3, -3) * unit);
			polygon.push_back(Vector2(3, 3) * unit);
			polygon.push_back(Vector2(1, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
			polygon.push_back(Vector2(-1, 1) * unit);
			polygon.push_back(Vector2(-3, 3) * unit);
			polygon.push_back(Vector2(3, 3) * unit);
			polygon.push_back(Vector2(1, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
			polygon.push_back(Vector2(-1, -1) * unit);
			polygon.push_back(Vector2(-3, -3) * unit);
			polygon.push_back(Vector2(-3, 3) * unit);
			polygon.push_back(Vector2(-1, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_SIDE:
			polygon.push_back(Vector2(-1, -1) * unit);
			polygon.push_back(Vector2(-3, -3) * unit);
			polygon.push_back(Vector2(3, -3) * unit);
			polygon.push_back(Vector2(1, -1) * unit);
			break;
		default:
			break;
	}
	return polygon;
}

Vector<Point2> TileSet::_get_isometric_corner_or_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit) {
	Vector2 unit = Vector2(p_size) / 6.0;
	Vector<Vector2> polygon;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(2, -1) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(2, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
			polygon.push_back(Vector2(0, 1) * unit);
			polygon.push_back(Vector2(1, 2) * unit);
			polygon.push_back(Vector2(2, 1) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
			polygon.push_back(Vector2(0, 1) * unit);
			polygon.push_back(Vector2(-1, 2) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(1, 2) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
			polygon.push_back(Vector2(0, 1) * unit);
			polygon.push_back(Vector2(-1, 2) * unit);
			polygon.push_back(Vector2(-2, 1) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-2, -1) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-2, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
			polygon.push_back(Vector2(0, -1) * unit);
			polygon.push_back(Vector2(-1, -2) * unit);
			polygon.push_back(Vector2(-2, -1) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_CORNER:
			polygon.push_back(Vector2(0, -1) * unit);
			polygon.push_back(Vector2(-1, -2) * unit);
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(1, -2) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
			polygon.push_back(Vector2(0, -1) * unit);
			polygon.push_back(Vector2(1, -2) * unit);
			polygon.push_back(Vector2(2, -1) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			break;
		default:
			break;
	}
	return polygon;
}

Vector<Point2> TileSet::_get_isometric_corner_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit) {
	Vector2 unit = Vector2(p_size) / 6.0;
	Vector<Vector2> polygon;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
			polygon.push_back(Vector2(0.5, -0.5) * unit);
			polygon.push_back(Vector2(1.5, -1.5) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(1.5, 1.5) * unit);
			polygon.push_back(Vector2(0.5, 0.5) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
			polygon.push_back(Vector2(-0.5, 0.5) * unit);
			polygon.push_back(Vector2(-1.5, 1.5) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(1.5, 1.5) * unit);
			polygon.push_back(Vector2(0.5, 0.5) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
			polygon.push_back(Vector2(-0.5, -0.5) * unit);
			polygon.push_back(Vector2(-1.5, -1.5) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-1.5, 1.5) * unit);
			polygon.push_back(Vector2(-0.5, 0.5) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_CORNER:
			polygon.push_back(Vector2(-0.5, -0.5) * unit);
			polygon.push_back(Vector2(-1.5, -1.5) * unit);
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(1.5, -1.5) * unit);
			polygon.push_back(Vector2(0.5, -0.5) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		default:
			break;
	}
	return polygon;
}

Vector<Point2> TileSet::_get_isometric_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit) {
	Vector2 unit = Vector2(p_size) / 6.0;
	Vector<Vector2> polygon;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		default:
			break;
	}
	return polygon;
}

Vector<Point2> TileSet::_get_half_offset_corner_or_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis) {
	Vector<Vector2> point_list;
	point_list.push_back(Vector2(3, (3.0 * (1.0 - p_overlap * 2.0)) / 2.0));
	point_list.push_back(Vector2(3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(2, 3.0 * (1.0 - (p_overlap * 2.0) * 2.0 / 3.0)));
	point_list.push_back(Vector2(1, 3.0 - p_overlap * 2.0));
	point_list.push_back(Vector2(0, 3));
	point_list.push_back(Vector2(-1, 3.0 - p_overlap * 2.0));
	point_list.push_back(Vector2(-2, 3.0 * (1.0 - (p_overlap * 2.0) * 2.0 / 3.0)));
	point_list.push_back(Vector2(-3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(-3, (3.0 * (1.0 - p_overlap * 2.0)) / 2.0));
	point_list.push_back(Vector2(-3, -(3.0 * (1.0 - p_overlap * 2.0)) / 2.0));
	point_list.push_back(Vector2(-3, -3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(-2, -3.0 * (1.0 - (p_overlap * 2.0) * 2.0 / 3.0)));
	point_list.push_back(Vector2(-1, -(3.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(0, -3));
	point_list.push_back(Vector2(1, -(3.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(2, -3.0 * (1.0 - (p_overlap * 2.0) * 2.0 / 3.0)));
	point_list.push_back(Vector2(3, -3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(3, -(3.0 * (1.0 - p_overlap * 2.0)) / 2.0));

	Vector2 unit = Vector2(p_size) / 6.0;
	for (int i = 0; i < point_list.size(); i++) {
		point_list.write[i] = point_list[i] * unit;
	}

	Vector<Vector2> polygon;
	if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
				polygon.push_back(point_list[17]);
				polygon.push_back(point_list[0]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[6]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				polygon.push_back(point_list[11]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
				polygon.push_back(point_list[11]);
				polygon.push_back(point_list[12]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_CORNER:
				polygon.push_back(point_list[12]);
				polygon.push_back(point_list[13]);
				polygon.push_back(point_list[14]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
				polygon.push_back(point_list[14]);
				polygon.push_back(point_list[15]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
				polygon.push_back(point_list[15]);
				polygon.push_back(point_list[16]);
				polygon.push_back(point_list[17]);
				break;
			default:
				break;
		}
	} else {
		if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
			for (int i = 0; i < point_list.size(); i++) {
				point_list.write[i] = Vector2(point_list[i].y, point_list[i].x);
			}
		}
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
				polygon.push_back(point_list[17]);
				polygon.push_back(point_list[0]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[15]);
				polygon.push_back(point_list[16]);
				polygon.push_back(point_list[17]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[14]);
				polygon.push_back(point_list[15]);
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
				polygon.push_back(point_list[12]);
				polygon.push_back(point_list[13]);
				polygon.push_back(point_list[14]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
				polygon.push_back(point_list[11]);
				polygon.push_back(point_list[12]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				polygon.push_back(point_list[11]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_SIDE:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[6]);
				break;
			default:
				break;
		}
	}

	int half_polygon_size = polygon.size();
	for (int i = 0; i < half_polygon_size; i++) {
		polygon.push_back(polygon[half_polygon_size - 1 - i] / 3.0);
	}

	return polygon;
}

Vector<Point2> TileSet::_get_half_offset_corner_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis) {
	Vector<Vector2> point_list;
	point_list.push_back(Vector2(3, 0));
	point_list.push_back(Vector2(3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(1.5, (3.0 * (1.0 - p_overlap * 2.0) + 3.0) / 2.0));
	point_list.push_back(Vector2(0, 3));
	point_list.push_back(Vector2(-1.5, (3.0 * (1.0 - p_overlap * 2.0) + 3.0) / 2.0));
	point_list.push_back(Vector2(-3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(-3, 0));
	point_list.push_back(Vector2(-3, -3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(-1.5, -(3.0 * (1.0 - p_overlap * 2.0) + 3.0) / 2.0));
	point_list.push_back(Vector2(0, -3));
	point_list.push_back(Vector2(1.5, -(3.0 * (1.0 - p_overlap * 2.0) + 3.0) / 2.0));
	point_list.push_back(Vector2(3, -3.0 * (1.0 - p_overlap * 2.0)));

	Vector2 unit = Vector2(p_size) / 6.0;
	for (int i = 0; i < point_list.size(); i++) {
		point_list.write[i] = point_list[i] * unit;
	}

	Vector<Vector2> polygon;
	if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[6]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_CORNER:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
				polygon.push_back(point_list[10]);
				polygon.push_back(point_list[11]);
				polygon.push_back(point_list[0]);
				break;
			default:
				break;
		}
	} else {
		if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
			for (int i = 0; i < point_list.size(); i++) {
				point_list.write[i] = Vector2(point_list[i].y, point_list[i].x);
			}
		}
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[10]);
				polygon.push_back(point_list[11]);
				polygon.push_back(point_list[0]);
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[6]);
				break;
			default:
				break;
		}
	}

	int half_polygon_size = polygon.size();
	for (int i = 0; i < half_polygon_size; i++) {
		polygon.push_back(polygon[half_polygon_size - 1 - i] / 3.0);
	}

	return polygon;
}

Vector<Point2> TileSet::_get_half_offset_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis) {
	Vector<Vector2> point_list;
	point_list.push_back(Vector2(3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(0, 3));
	point_list.push_back(Vector2(-3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(-3, -3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(0, -3));
	point_list.push_back(Vector2(3, -3.0 * (1.0 - p_overlap * 2.0)));

	Vector2 unit = Vector2(p_size) / 6.0;
	for (int i = 0; i < point_list.size(); i++) {
		point_list.write[i] = point_list[i] * unit;
	}

	Vector<Vector2> polygon;
	if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[0]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				break;
			default:
				break;
		}
	} else {
		if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
			for (int i = 0; i < point_list.size(); i++) {
				point_list.write[i] = Vector2(point_list[i].y, point_list[i].x);
			}
		}
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[0]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			default:
				break;
		}
	}

	int half_polygon_size = polygon.size();
	for (int i = 0; i < half_polygon_size; i++) {
		polygon.push_back(polygon[half_polygon_size - 1 - i] / 3.0);
	}

	return polygon;
}

void TileSet::reset_state() {
	occlusion_layers.clear();
	physics_layers.clear();
	custom_data_layers.clear();
}

const Vector2i TileSetSource::INVALID_ATLAS_COORDS = Vector2i(-1, -1);
const int TileSetSource::INVALID_TILE_ALTERNATIVE = -1;

#ifndef DISABLE_DEPRECATED
void TileSet::_compatibility_conversion() {
	for (Map<int, CompatibilityTileData *>::Element *E = compatibility_data.front(); E; E = E->next()) {
		CompatibilityTileData *ctd = E->value();

		// Add the texture
		TileSetAtlasSource *atlas_source = memnew(TileSetAtlasSource);
		int source_id = add_source(Ref<TileSetSource>(atlas_source));

		atlas_source->set_texture(ctd->texture);

		// Handle each tile as a new source. Not optimal but at least it should stay compatible.
		switch (ctd->tile_mode) {
			case COMPATIBILITY_TILE_MODE_SINGLE_TILE: {
				atlas_source->set_margins(ctd->region.get_position());
				atlas_source->set_texture_region_size(ctd->region.get_size());

				Vector2i coords;
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

					// Add to the mapping.
					Array key_array;
					key_array.push_back(flip_h);
					key_array.push_back(flip_v);
					key_array.push_back(transpose);

					Array value_array;
					value_array.push_back(source_id);
					value_array.push_back(coords);
					value_array.push_back(alternative_tile);

					if (!compatibility_tilemap_mapping.has(E->key())) {
						compatibility_tilemap_mapping[E->key()] = Map<Array, Array>();
					}
					compatibility_tilemap_mapping[E->key()][key_array] = value_array;
					compatibility_tilemap_mapping_tile_modes[E->key()] = COMPATIBILITY_TILE_MODE_SINGLE_TILE;

					TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(coords, alternative_tile));

					tile_data->set_flip_h(flip_h);
					tile_data->set_flip_v(flip_v);
					tile_data->set_transpose(transpose);
					tile_data->tile_set_material(ctd->material);
					tile_data->set_modulate(ctd->modulate);
					tile_data->set_z_index(ctd->z_index);

					if (ctd->occluder.is_valid()) {
						if (get_occlusion_layers_count() < 1) {
							set_occlusion_layers_count(1);
						}
						tile_data->set_occluder(0, ctd->occluder);
					}
					if (ctd->navigation.is_valid()) {
						if (get_navigation_layers_count() < 1) {
							set_navigation_layers_count(1);
						}
						tile_data->set_navigation_polygon(0, ctd->autotile_navpoly_map[coords]);
					}

					tile_data->set_z_index(ctd->z_index);

					// Add the shapes.
					if (ctd->shapes.size() > 0) {
						if (get_physics_layers_count() < 1) {
							set_physics_layers_count(1);
						}
					}
					for (int k = 0; k < ctd->shapes.size(); k++) {
						CompatibilityShapeData csd = ctd->shapes[k];
						if (csd.autotile_coords == coords) {
							Ref<ConvexPolygonShape2D> convex_shape = csd.shape; // Only ConvexPolygonShape2D are supported, which is the default type used by the 3.x editor
							if (convex_shape.is_valid()) {
								Vector<Vector2> polygon = convex_shape->get_points();
								for (int point_index = 0; point_index < polygon.size(); point_index++) {
									polygon.write[point_index] = csd.transform.xform(polygon[point_index]);
								}
								tile_data->set_collision_polygons_count(0, tile_data->get_collision_polygons_count(0) + 1);
								int index = tile_data->get_collision_polygons_count(0) - 1;
								tile_data->set_collision_polygon_one_way(0, index, csd.one_way);
								tile_data->set_collision_polygon_one_way_margin(0, index, csd.one_way_margin);
								tile_data->set_collision_polygon_points(0, index, polygon);
							}
						}
					}
				}
			} break;
			case COMPATIBILITY_TILE_MODE_AUTO_TILE: {
				// Not supported. It would need manual conversion.
			} break;
			case COMPATIBILITY_TILE_MODE_ATLAS_TILE: {
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

							// Add to the mapping.
							Array key_array;
							key_array.push_back(coords);
							key_array.push_back(flip_h);
							key_array.push_back(flip_v);
							key_array.push_back(transpose);

							Array value_array;
							value_array.push_back(source_id);
							value_array.push_back(coords);
							value_array.push_back(alternative_tile);

							if (!compatibility_tilemap_mapping.has(E->key())) {
								compatibility_tilemap_mapping[E->key()] = Map<Array, Array>();
							}
							compatibility_tilemap_mapping[E->key()][key_array] = value_array;
							compatibility_tilemap_mapping_tile_modes[E->key()] = COMPATIBILITY_TILE_MODE_ATLAS_TILE;

							TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(coords, alternative_tile));

							tile_data->set_flip_h(flip_h);
							tile_data->set_flip_v(flip_v);
							tile_data->set_transpose(transpose);
							tile_data->tile_set_material(ctd->material);
							tile_data->set_modulate(ctd->modulate);
							tile_data->set_z_index(ctd->z_index);
							if (ctd->autotile_occluder_map.has(coords)) {
								if (get_occlusion_layers_count() < 1) {
									set_occlusion_layers_count(1);
								}
								tile_data->set_occluder(0, ctd->autotile_occluder_map[coords]);
							}
							if (ctd->autotile_navpoly_map.has(coords)) {
								if (get_navigation_layers_count() < 1) {
									set_navigation_layers_count(1);
								}
								tile_data->set_navigation_polygon(0, ctd->autotile_navpoly_map[coords]);
							}
							if (ctd->autotile_priority_map.has(coords)) {
								tile_data->set_probability(ctd->autotile_priority_map[coords]);
							}
							if (ctd->autotile_z_index_map.has(coords)) {
								tile_data->set_z_index(ctd->autotile_z_index_map[coords]);
							}

							// Add the shapes.
							if (ctd->shapes.size() > 0) {
								if (get_physics_layers_count() < 1) {
									set_physics_layers_count(1);
								}
							}
							for (int k = 0; k < ctd->shapes.size(); k++) {
								CompatibilityShapeData csd = ctd->shapes[k];
								if (csd.autotile_coords == coords) {
									Ref<ConvexPolygonShape2D> convex_shape = csd.shape; // Only ConvexPolygonShape2D are supported, which is the default type used by the 3.x editor
									if (convex_shape.is_valid()) {
										Vector<Vector2> polygon = convex_shape->get_points();
										for (int point_index = 0; point_index < polygon.size(); point_index++) {
											polygon.write[point_index] = csd.transform.xform(polygon[point_index]);
										}
										tile_data->set_collision_polygons_count(0, tile_data->get_collision_polygons_count(0) + 1);
										int index = tile_data->get_collision_polygons_count(0) - 1;
										tile_data->set_collision_polygon_one_way(0, index, csd.one_way);
										tile_data->set_collision_polygon_one_way_margin(0, index, csd.one_way_margin);
										tile_data->set_collision_polygon_points(0, index, polygon);
									}
								}
							}

							// -- TODO: handle --
							// Those are offset for the whole atlas, they are likely useless for the atlases, but might make sense for single tiles.
							// texture offset
							// occluder_offset
							// navigation_offset

							// For terrains, ignored for now?
							// bitmask_mode
							// bitmask_flags
						}
					}
				}
			} break;
		}

		// Offset all shapes
		for (int k = 0; k < ctd->shapes.size(); k++) {
			Ref<ConvexPolygonShape2D> convex = ctd->shapes[k].shape;
			if (convex.is_valid()) {
				Vector<Vector2> points = convex->get_points();
				for (int i_point = 0; i_point < points.size(); i_point++) {
					points.write[i_point] = points[i_point] - get_tile_size() / 2;
				}
				convex->set_points(points);
			}
		}
	}

	// Reset compatibility data
	for (Map<int, CompatibilityTileData *>::Element *E = compatibility_data.front(); E; E = E->next()) {
		memdelete(E->get());
	}
	compatibility_data = Map<int, CompatibilityTileData *>();
}

Array TileSet::compatibility_tilemap_map(int p_tile_id, Vector2i p_coords, bool p_flip_h, bool p_flip_v, bool p_transpose) {
	Array cannot_convert_array;
	cannot_convert_array.push_back(TileSet::INVALID_SOURCE);
	cannot_convert_array.push_back(TileSetAtlasSource::INVALID_ATLAS_COORDS);
	cannot_convert_array.push_back(TileSetAtlasSource::INVALID_TILE_ALTERNATIVE);

	if (!compatibility_tilemap_mapping.has(p_tile_id)) {
		return cannot_convert_array;
	}

	int tile_mode = compatibility_tilemap_mapping_tile_modes[p_tile_id];
	switch (tile_mode) {
		case COMPATIBILITY_TILE_MODE_SINGLE_TILE: {
			Array a;
			a.push_back(p_flip_h);
			a.push_back(p_flip_v);
			a.push_back(p_transpose);
			return compatibility_tilemap_mapping[p_tile_id][a];
		}
		case COMPATIBILITY_TILE_MODE_AUTO_TILE:
			return cannot_convert_array;
			break;
		case COMPATIBILITY_TILE_MODE_ATLAS_TILE: {
			Array a;
			a.push_back(p_coords);
			a.push_back(p_flip_h);
			a.push_back(p_flip_v);
			a.push_back(p_transpose);
			return compatibility_tilemap_mapping[p_tile_id][a];
		}
		default:
			return cannot_convert_array;
			break;
	}
};

#endif // DISABLE_DEPRECATED

bool TileSet::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);

#ifndef DISABLE_DEPRECATED
	// TODO: This should be moved to a dedicated conversion system (see #50691)
	if (components.size() >= 1 && components[0].is_valid_int()) {
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

		if (components.size() < 2) {
			return false;
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
			what = what.substr(9);
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

		} else if (what == "shapes") {
			Array p = p_value;
			for (int i = 0; i < p.size(); i++) {
				CompatibilityShapeData csd;
				Dictionary d = p[i];
				for (int j = 0; j < d.size(); j++) {
					String key = d.get_key_at_index(j);
					if (key == "autotile_coord") {
						csd.autotile_coords = d[key];
					} else if (key == "one_way") {
						csd.one_way = d[key];
					} else if (key == "one_way_margin") {
						csd.one_way_margin = d[key];
					} else if (key == "shape") {
						csd.shape = d[key];
					} else if (key == "shape_transform") {
						csd.transform = d[key];
					}
				}
				ctd->shapes.push_back(csd);
			}

			/*
		// IGNORED FOR NOW, they seem duplicated data compared to the shapes array
		} else if (what == "shape") {
		} else if (what == "shape_offset") {
		} else if (what == "shape_transform") {
		} else if (what == "shape_one_way") {
		} else if (what == "shape_one_way_margin") {
		}
		// IGNORED FOR NOW, maybe useless ?
		else if (what == "occluder_offset") {
			// Not
		} else if (what == "navigation_offset") {
		}
		*/

		} else if (what == "z_index") {
			ctd->z_index = p_value;

			// TODO: remove the conversion from here, it's not where it should be done (see #50691)
			_compatibility_conversion();
		} else {
			return false;
		}
	} else {
#endif // DISABLE_DEPRECATED

		// This is now a new property.
		if (components.size() == 2 && components[0].begins_with("occlusion_layer_") && components[0].trim_prefix("occlusion_layer_").is_valid_int()) {
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
		} else if (components.size() == 2 && components[0].begins_with("physics_layer_") && components[0].trim_prefix("physics_layer_").is_valid_int()) {
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
		} else if (components.size() >= 2 && components[0].begins_with("terrain_set_") && components[0].trim_prefix("terrain_set_").is_valid_int()) {
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
			} else if (components.size() >= 3 && components[1].begins_with("terrain_") && components[1].trim_prefix("terrain_").is_valid_int()) {
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
		} else if (components.size() == 2 && components[0].begins_with("navigation_layer_") && components[0].trim_prefix("navigation_layer_").is_valid_int()) {
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
		} else if (components.size() == 2 && components[0].begins_with("custom_data_layer_") && components[0].trim_prefix("custom_data_layer_").is_valid_int()) {
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
		} else if (components.size() == 2 && components[0] == "sources" && components[1].is_valid_int()) {
			// Create source only if it does not exists.
			int source_id = components[1].to_int();

			if (!has_source(source_id)) {
				add_source(p_value, source_id);
			}
			return true;
		} else if (components.size() == 2 && components[0] == "tile_proxies") {
			ERR_FAIL_COND_V(p_value.get_type() != Variant::ARRAY, false);
			Array a = p_value;
			ERR_FAIL_COND_V(a.size() % 2 != 0, false);
			if (components[1] == "source_level") {
				for (int i = 0; i < a.size(); i += 2) {
					set_source_level_tile_proxy(a[i], a[i + 1]);
				}
				return true;
			} else if (components[1] == "coords_level") {
				for (int i = 0; i < a.size(); i += 2) {
					Array key = a[i];
					Array value = a[i + 1];
					set_coords_level_tile_proxy(key[0], key[1], value[0], value[1]);
				}
				return true;
			} else if (components[1] == "alternative_level") {
				for (int i = 0; i < a.size(); i += 2) {
					Array key = a[i];
					Array value = a[i + 1];
					set_alternative_level_tile_proxy(key[0], key[1], key[2], value[0], value[1], value[2]);
				}
				return true;
			}
			return false;
		}

#ifndef DISABLE_DEPRECATED
	}
#endif // DISABLE_DEPRECATED

	return false;
}

bool TileSet::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> components = String(p_name).split("/", true, 2);

	if (components.size() == 2 && components[0].begins_with("occlusion_layer_") && components[0].trim_prefix("occlusion_layer_").is_valid_int()) {
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
	} else if (components.size() == 2 && components[0].begins_with("physics_layer_") && components[0].trim_prefix("physics_layer_").is_valid_int()) {
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
	} else if (components.size() >= 2 && components[0].begins_with("terrain_set_") && components[0].trim_prefix("terrain_set_").is_valid_int()) {
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
		} else if (components.size() >= 3 && components[1].begins_with("terrain_") && components[1].trim_prefix("terrain_").is_valid_int()) {
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
	} else if (components.size() == 2 && components[0].begins_with("navigation_layer_") && components[0].trim_prefix("navigation_layer_").is_valid_int()) {
		// navigation layers.
		int index = components[0].trim_prefix("navigation_layer_").to_int();
		if (index < 0 || index >= navigation_layers.size()) {
			return false;
		}
		if (components[1] == "layers") {
			r_ret = get_navigation_layer_layers(index);
			return true;
		}
	} else if (components.size() == 2 && components[0].begins_with("custom_data_layer_") && components[0].trim_prefix("custom_data_layer_").is_valid_int()) {
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
	} else if (components.size() == 2 && components[0] == "sources" && components[1].is_valid_int()) {
		// Atlases data.
		int source_id = components[1].to_int();

		if (has_source(source_id)) {
			r_ret = get_source(source_id);
			return true;
		} else {
			return false;
		}
	} else if (components.size() == 2 && components[0] == "tile_proxies") {
		if (components[1] == "source_level") {
			Array a;
			for (Map<int, int>::Element *E = source_level_proxies.front(); E; E = E->next()) {
				a.push_back(E->key());
				a.push_back(E->get());
			}
			r_ret = a;
			return true;
		} else if (components[1] == "coords_level") {
			Array a;
			for (Map<Array, Array>::Element *E = coords_level_proxies.front(); E; E = E->next()) {
				a.push_back(E->key());
				a.push_back(E->get());
			}
			r_ret = a;
			return true;
		} else if (components[1] == "alternative_level") {
			Array a;
			for (Map<Array, Array>::Element *E = alternative_level_proxies.front(); E; E = E->next()) {
				a.push_back(E->key());
				a.push_back(E->get());
			}
			r_ret = a;
			return true;
		}
		return false;
	}

	return false;
}

void TileSet::_get_property_list(List<PropertyInfo> *p_list) const {
	PropertyInfo property_info;
	// Rendering.
	p_list->push_back(PropertyInfo(Variant::NIL, "Rendering", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
	for (int i = 0; i < occlusion_layers.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::INT, vformat("occlusion_layer_%d/light_mask", i), PROPERTY_HINT_LAYERS_2D_RENDER));

		// occlusion_layer_%d/sdf_collision
		property_info = PropertyInfo(Variant::BOOL, vformat("occlusion_layer_%d/sdf_collision", i));
		if (occlusion_layers[i].sdf_collision == false) {
			property_info.usage ^= PROPERTY_USAGE_STORAGE;
		}
		p_list->push_back(property_info);
	}

	// Physics.
	p_list->push_back(PropertyInfo(Variant::NIL, "Physics", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
	for (int i = 0; i < physics_layers.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::INT, vformat("physics_layer_%d/collision_layer", i), PROPERTY_HINT_LAYERS_2D_PHYSICS));

		// physics_layer_%d/collision_mask
		property_info = PropertyInfo(Variant::INT, vformat("physics_layer_%d/collision_mask", i), PROPERTY_HINT_LAYERS_2D_PHYSICS);
		if (physics_layers[i].collision_mask == 1) {
			property_info.usage ^= PROPERTY_USAGE_STORAGE;
		}
		p_list->push_back(property_info);

		// physics_layer_%d/physics_material
		property_info = PropertyInfo(Variant::OBJECT, vformat("physics_layer_%d/physics_material", i), PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial");
		if (!physics_layers[i].physics_material.is_valid()) {
			property_info.usage ^= PROPERTY_USAGE_STORAGE;
		}
		p_list->push_back(property_info);
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
		p_list->push_back(PropertyInfo(Variant::INT, vformat("sources/%d", E_source->key()), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	}

	// Tile Proxies.
	// Note: proxies need to be set after sources are set.
	p_list->push_back(PropertyInfo(Variant::NIL, "Tile Proxies", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
	p_list->push_back(PropertyInfo(Variant::ARRAY, "tile_proxies/source_level", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::ARRAY, "tile_proxies/coords_level", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::ARRAY, "tile_proxies/alternative_level", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
}

void TileSet::_bind_methods() {
	// Sources management.
	ClassDB::bind_method(D_METHOD("get_next_source_id"), &TileSet::get_next_source_id);
	ClassDB::bind_method(D_METHOD("add_source", "atlas_source_id_override"), &TileSet::add_source, DEFVAL(TileSet::INVALID_SOURCE));
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

	ADD_PROPERTY(PropertyInfo(Variant::INT, "tile_shape", PROPERTY_HINT_ENUM, "Square,Isometric,Half-Offset Square,Hexagon"), "set_tile_shape", "get_tile_shape");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tile_layout", PROPERTY_HINT_ENUM, "Stacked,Stacked Offset,Stairs Right,Stairs Down,Diamond Right,Diamond Down"), "set_tile_layout", "get_tile_layout");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tile_offset_axis", PROPERTY_HINT_ENUM, "Horizontal Offset,Vertical Offset"), "set_tile_offset_axis", "get_tile_offset_axis");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "tile_size"), "set_tile_size", "get_tile_size");

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

	// Tile proxies
	ClassDB::bind_method(D_METHOD("set_source_level_tile_proxy", "source_from", "source_to"), &TileSet::set_source_level_tile_proxy);
	ClassDB::bind_method(D_METHOD("get_source_level_tile_proxy", "source_from"), &TileSet::get_source_level_tile_proxy);
	ClassDB::bind_method(D_METHOD("has_source_level_tile_proxy", "source_from"), &TileSet::has_source_level_tile_proxy);
	ClassDB::bind_method(D_METHOD("remove_source_level_tile_proxy", "source_from"), &TileSet::remove_source_level_tile_proxy);

	ClassDB::bind_method(D_METHOD("set_coords_level_tile_proxy", "p_source_from", "coords_from", "source_to", "coords_to"), &TileSet::set_coords_level_tile_proxy);
	ClassDB::bind_method(D_METHOD("get_coords_level_tile_proxy", "source_from", "coords_from"), &TileSet::get_coords_level_tile_proxy);
	ClassDB::bind_method(D_METHOD("has_coords_level_tile_proxy", "source_from", "coords_from"), &TileSet::has_coords_level_tile_proxy);
	ClassDB::bind_method(D_METHOD("remove_coords_level_tile_proxy", "source_from", "coords_from"), &TileSet::remove_coords_level_tile_proxy);

	ClassDB::bind_method(D_METHOD("set_alternative_level_tile_proxy", "source_from", "coords_from", "alternative_from", "source_to", "coords_to", "alternative_to"), &TileSet::set_alternative_level_tile_proxy);
	ClassDB::bind_method(D_METHOD("get_alternative_level_tile_proxy", "source_from", "coords_from", "alternative_from"), &TileSet::get_alternative_level_tile_proxy);
	ClassDB::bind_method(D_METHOD("has_alternative_level_tile_proxy", "source_from", "coords_from", "alternative_from"), &TileSet::has_alternative_level_tile_proxy);
	ClassDB::bind_method(D_METHOD("remove_alternative_level_tile_proxy", "source_from", "coords_from", "alternative_from"), &TileSet::remove_alternative_level_tile_proxy);

	ClassDB::bind_method(D_METHOD("map_tile_proxy", "source_from", "coords_from", "alternative_from"), &TileSet::map_tile_proxy);

	ClassDB::bind_method(D_METHOD("cleanup_invalid_tile_proxies"), &TileSet::cleanup_invalid_tile_proxies);
	ClassDB::bind_method(D_METHOD("clear_tile_proxies"), &TileSet::clear_tile_proxies);

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

	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_RIGHT_SIDE);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_RIGHT_CORNER);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_BOTTOM_SIDE);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_BOTTOM_CORNER);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_BOTTOM_LEFT_SIDE);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_BOTTOM_LEFT_CORNER);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_LEFT_SIDE);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_LEFT_CORNER);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_TOP_LEFT_SIDE);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_TOP_LEFT_CORNER);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_TOP_SIDE);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_TOP_CORNER);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_TOP_RIGHT_SIDE);
	BIND_ENUM_CONSTANT(CELL_NEIGHBOR_TOP_RIGHT_CORNER);

	BIND_ENUM_CONSTANT(TERRAIN_MODE_MATCH_CORNERS_AND_SIDES);
	BIND_ENUM_CONSTANT(TERRAIN_MODE_MATCH_CORNERS);
	BIND_ENUM_CONSTANT(TERRAIN_MODE_MATCH_SIDES);
}

TileSet::TileSet() {
	// Instantiate the tile meshes.
	tile_lines_mesh.instantiate();
	tile_filled_mesh.instantiate();

	// Instanciate and list all plugins.
	tile_set_plugins_vector.append(memnew(TileSetPluginAtlasRendering));
	tile_set_plugins_vector.append(memnew(TileSetPluginAtlasPhysics));
	tile_set_plugins_vector.append(memnew(TileSetPluginAtlasNavigation));
	tile_set_plugins_vector.append(memnew(TileSetPluginScenesCollections));
}

TileSet::~TileSet() {
#ifndef DISABLE_DEPRECATED
	for (Map<int, CompatibilityTileData *>::Element *E = compatibility_data.front(); E; E = E->next()) {
		memdelete(E->get());
	}
#endif // DISABLE_DEPRECATED
	while (!source_ids.is_empty()) {
		remove_source(source_ids[0]);
	}
	for (int i = 0; i < tile_set_plugins_vector.size(); i++) {
		memdelete(tile_set_plugins_vector[i]);
	}
}

/////////////////////////////// TileSetSource //////////////////////////////////////

void TileSetSource::set_tile_set(const TileSet *p_tile_set) {
	tile_set = p_tile_set;
}

/////////////////////////////// TileSetAtlasSource //////////////////////////////////////

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
	Vector2i coords = TileSetSource::INVALID_ATLAS_COORDS;
	if (coords_split.size() == 2 && coords_split[0].is_valid_int() && coords_split[1].is_valid_int()) {
		coords = Vector2i(coords_split[0].to_int(), coords_split[1].to_int());
	}

	// Properties.
	if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
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
			} else if (components[1].is_valid_int()) {
				int alternative_id = components[1].to_int();
				if (alternative_id != TileSetSource::INVALID_TILE_ALTERNATIVE) {
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
	if (coords_split.size() == 2 && coords_split[0].is_valid_int() && coords_split[1].is_valid_int()) {
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
				} else if (components[1].is_valid_int()) {
					int alternative_id = components[1].to_int();
					if (alternative_id != TileSetSource::INVALID_TILE_ALTERNATIVE && tiles[coords].alternatives.has(alternative_id)) {
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
	PropertyInfo property_info;
	for (Map<Vector2i, TileAlternativesData>::Element *E_tile = tiles.front(); E_tile; E_tile = E_tile->next()) {
		List<PropertyInfo> tile_property_list;

		// size_in_atlas
		property_info = PropertyInfo(Variant::VECTOR2I, "size_in_atlas", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR);
		if (E_tile->get().size_in_atlas == Vector2i(1, 1)) {
			property_info.usage ^= PROPERTY_USAGE_STORAGE;
		}
		tile_property_list.push_back(property_info);

		// next_alternative_id
		property_info = PropertyInfo(Variant::INT, "next_alternative_id", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR);
		if (E_tile->get().next_alternative_id == 1) {
			property_info.usage ^= PROPERTY_USAGE_STORAGE;
		}
		tile_property_list.push_back(property_info);

		for (Map<int, TileData *>::Element *E_alternative = E_tile->get().alternatives.front(); E_alternative; E_alternative = E_alternative->next()) {
			// Add a dummy property to show the alternative exists.
			tile_property_list.push_back(PropertyInfo(Variant::INT, vformat("%d", E_alternative->key()), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));

			// Get the alternative tile's properties and append them to the list of properties.
			List<PropertyInfo> alternative_property_list;
			E_alternative->get()->get_property_list(&alternative_property_list);
			for (List<PropertyInfo>::Element *E_property = alternative_property_list.front(); E_property; E_property = E_property->next()) {
				property_info = E_property->get();
				bool valid;
				Variant default_value = ClassDB::class_get_default_property_value("TileData", property_info.name, &valid);
				Variant value = E_alternative->get()->get(property_info.name);
				if (valid && value == default_value) {
					property_info.usage ^= PROPERTY_USAGE_STORAGE;
				}
				property_info.name = vformat("%s/%s", vformat("%d", E_alternative->key()), property_info.name);
				tile_property_list.push_back(property_info);
			}
		}

		// Add all alternative.
		for (List<PropertyInfo>::Element *E_property = tile_property_list.front(); E_property; E_property = E_property->next()) {
			E_property->get().name = vformat("%s/%s", vformat("%d:%d", E_tile->key().x, E_tile->key().y), E_property->get().name);
			p_list->push_back(E_property->get());
		}
	}
}

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

	emit_signal(SNAME("changed"));
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

	emit_signal(SNAME("changed"));
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
	ERR_FAIL_INDEX_V(p_index, tiles_ids.size(), TileSetSource::INVALID_ATLAS_COORDS);
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

Vector2i TileSetAtlasSource::get_tile_effective_texture_offset(Vector2i p_atlas_coords, int p_alternative_tile) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), Vector2i(), vformat("TileSetAtlasSource has no tile at %s.", Vector2i(p_atlas_coords)));
	ERR_FAIL_COND_V_MSG(!has_alternative_tile(p_atlas_coords, p_alternative_tile), Vector2i(), vformat("TileSetAtlasSource has no alternative tile with id %d at %s.", p_alternative_tile, String(p_atlas_coords)));
	ERR_FAIL_COND_V(!tile_set, Vector2i());

	Vector2 margin = (get_tile_texture_region(p_atlas_coords).size - tile_set->get_tile_size()) / 2;
	margin = Vector2i(MAX(0, margin.x), MAX(0, margin.y));
	Vector2i effective_texture_offset = Object::cast_to<TileData>(get_tile_data(p_atlas_coords, p_alternative_tile))->get_texture_offset();
	if (ABS(effective_texture_offset.x) > margin.x || ABS(effective_texture_offset.y) > margin.y) {
		effective_texture_offset.x = CLAMP(effective_texture_offset.x, -margin.x, margin.x);
		effective_texture_offset.y = CLAMP(effective_texture_offset.y, -margin.y, margin.y);
	}

	return effective_texture_offset;
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
			if (coords != p_atlas_coords && coords != TileSetSource::INVALID_ATLAS_COORDS) {
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

	emit_signal(SNAME("changed"));
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
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), TileSetSource::INVALID_TILE_ALTERNATIVE, vformat("TileSetAtlasSource has no tile at %s.", String(p_atlas_coords)));
	ERR_FAIL_COND_V_MSG(p_alternative_id_override >= 0 && tiles[p_atlas_coords].alternatives.has(p_alternative_id_override), TileSetSource::INVALID_TILE_ALTERNATIVE, vformat("Cannot create alternative tile. Another alternative exists with id %d.", p_alternative_id_override));

	int new_alternative_id = p_alternative_id_override >= 0 ? p_alternative_id_override : tiles[p_atlas_coords].next_alternative_id;

	tiles[p_atlas_coords].alternatives[new_alternative_id] = memnew(TileData);
	tiles[p_atlas_coords].alternatives[new_alternative_id]->set_tile_set(tile_set);
	tiles[p_atlas_coords].alternatives[new_alternative_id]->set_allow_transform(true);
	tiles[p_atlas_coords].alternatives[new_alternative_id]->notify_property_list_changed();
	tiles[p_atlas_coords].alternatives_ids.append(new_alternative_id);
	tiles[p_atlas_coords].alternatives_ids.sort();
	_compute_next_alternative_id(p_atlas_coords);

	emit_signal(SNAME("changed"));

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

	emit_signal(SNAME("changed"));
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

	emit_signal(SNAME("changed"));
}

bool TileSetAtlasSource::has_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_tile) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), false, vformat("The TileSetAtlasSource atlas has no tile at %s.", String(p_atlas_coords)));
	return tiles[p_atlas_coords].alternatives.has(p_alternative_tile);
}

int TileSetAtlasSource::get_next_alternative_tile_id(const Vector2i p_atlas_coords) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), TileSetSource::INVALID_TILE_ALTERNATIVE, vformat("The TileSetAtlasSource atlas has no tile at %s.", String(p_atlas_coords)));
	return tiles[p_atlas_coords].next_alternative_id;
}

int TileSetAtlasSource::get_alternative_tiles_count(const Vector2i p_atlas_coords) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), -1, vformat("The TileSetAtlasSource atlas has no tile at %s.", String(p_atlas_coords)));
	return tiles[p_atlas_coords].alternatives_ids.size();
}

int TileSetAtlasSource::get_alternative_tile_id(const Vector2i p_atlas_coords, int p_index) const {
	ERR_FAIL_COND_V_MSG(!tiles.has(p_atlas_coords), TileSetSource::INVALID_TILE_ALTERNATIVE, vformat("The TileSetAtlasSource atlas has no tile at %s.", String(p_atlas_coords)));
	ERR_FAIL_INDEX_V(p_index, tiles[p_atlas_coords].alternatives_ids.size(), TileSetSource::INVALID_TILE_ALTERNATIVE);

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

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", PROPERTY_USAGE_NOEDITOR), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "margins", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_margins", "get_margins");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "separation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_separation", "get_separation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "tile_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_texture_region_size", "get_texture_region_size");

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
	ClassDB::bind_method(D_METHOD("create_alternative_tile", "atlas_coords", "alternative_id_override"), &TileSetAtlasSource::create_alternative_tile, DEFVAL(INVALID_TILE_ALTERNATIVE));
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

/////////////////////////////// TileSetScenesCollectionSource //////////////////////////////////////

void TileSetScenesCollectionSource::_compute_next_alternative_id() {
	while (scenes.has(next_scene_id)) {
		next_scene_id = (next_scene_id % 1073741823) + 1; // 2 ** 30
	};
}

int TileSetScenesCollectionSource::get_tiles_count() const {
	return 1;
}

Vector2i TileSetScenesCollectionSource::get_tile_id(int p_tile_index) const {
	ERR_FAIL_COND_V(p_tile_index != 0, TileSetSource::INVALID_ATLAS_COORDS);
	return Vector2i();
}

bool TileSetScenesCollectionSource::has_tile(Vector2i p_atlas_coords) const {
	return p_atlas_coords == Vector2i();
}

int TileSetScenesCollectionSource::get_alternative_tiles_count(const Vector2i p_atlas_coords) const {
	return scenes_ids.size();
}

int TileSetScenesCollectionSource::get_alternative_tile_id(const Vector2i p_atlas_coords, int p_index) const {
	ERR_FAIL_COND_V(p_atlas_coords != Vector2i(), TileSetSource::INVALID_TILE_ALTERNATIVE);
	ERR_FAIL_INDEX_V(p_index, scenes_ids.size(), TileSetSource::INVALID_TILE_ALTERNATIVE);

	return scenes_ids[p_index];
}

bool TileSetScenesCollectionSource::has_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_tile) const {
	ERR_FAIL_COND_V(p_atlas_coords != Vector2i(), false);
	return scenes.has(p_alternative_tile);
}

int TileSetScenesCollectionSource::create_scene_tile(Ref<PackedScene> p_packed_scene, int p_id_override) {
	ERR_FAIL_COND_V_MSG(p_id_override >= 0 && scenes.has(p_id_override), INVALID_TILE_ALTERNATIVE, vformat("Cannot create scene tile. Another scene tile exists with id %d.", p_id_override));

	int new_scene_id = p_id_override >= 0 ? p_id_override : next_scene_id;

	scenes[new_scene_id] = SceneData();
	scenes_ids.append(new_scene_id);
	scenes_ids.sort();
	set_scene_tile_scene(new_scene_id, p_packed_scene);
	_compute_next_alternative_id();

	emit_signal(SNAME("changed"));

	return new_scene_id;
}

void TileSetScenesCollectionSource::set_scene_tile_id(int p_id, int p_new_id) {
	ERR_FAIL_COND(p_new_id < 0);
	ERR_FAIL_COND(!has_scene_tile_id(p_id));
	ERR_FAIL_COND(has_scene_tile_id(p_new_id));

	scenes[p_new_id] = SceneData();
	scenes[p_new_id] = scenes[p_id];
	scenes_ids.append(p_new_id);
	scenes_ids.sort();

	_compute_next_alternative_id();

	scenes.erase(p_id);
	scenes_ids.erase(p_id);

	emit_signal(SNAME("changed"));
}

void TileSetScenesCollectionSource::set_scene_tile_scene(int p_id, Ref<PackedScene> p_packed_scene) {
	ERR_FAIL_COND(!scenes.has(p_id));
	if (p_packed_scene.is_valid()) {
		// Make sure we have a root node. Supposed to be at 0 index because find_node_by_path() does not seem to work.
		ERR_FAIL_COND(!p_packed_scene->get_state().is_valid());
		ERR_FAIL_COND(p_packed_scene->get_state()->get_node_count() < 1);

		// Check if it extends CanvasItem.
		String type = p_packed_scene->get_state()->get_node_type(0);
		bool extends_correct_class = ClassDB::is_parent_class(type, "Control") || ClassDB::is_parent_class(type, "Node2D");
		ERR_FAIL_COND_MSG(!extends_correct_class, vformat("Invalid PackedScene for TileSetScenesCollectionSource: %s. Root node should extend Control or Node2D.", p_packed_scene->get_path()));

		scenes[p_id].scene = p_packed_scene;
	} else {
		scenes[p_id].scene = Ref<PackedScene>();
	}
	emit_signal(SNAME("changed"));
}

Ref<PackedScene> TileSetScenesCollectionSource::get_scene_tile_scene(int p_id) const {
	ERR_FAIL_COND_V(!scenes.has(p_id), Ref<PackedScene>());
	return scenes[p_id].scene;
}

void TileSetScenesCollectionSource::set_scene_tile_display_placeholder(int p_id, bool p_display_placeholder) {
	ERR_FAIL_COND(!scenes.has(p_id));

	scenes[p_id].display_placeholder = p_display_placeholder;

	emit_signal(SNAME("changed"));
}

bool TileSetScenesCollectionSource::get_scene_tile_display_placeholder(int p_id) const {
	ERR_FAIL_COND_V(!scenes.has(p_id), false);
	return scenes[p_id].display_placeholder;
}

void TileSetScenesCollectionSource::remove_scene_tile(int p_id) {
	ERR_FAIL_COND(!scenes.has(p_id));

	scenes.erase(p_id);
	scenes_ids.erase(p_id);
	emit_signal(SNAME("changed"));
}

int TileSetScenesCollectionSource::get_next_scene_tile_id() const {
	return next_scene_id;
}

bool TileSetScenesCollectionSource::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);

	if (components.size() >= 2 && components[0] == "scenes" && components[1].is_valid_int()) {
		int scene_id = components[1].to_int();
		if (components.size() >= 3 && components[2] == "scene") {
			if (has_scene_tile_id(scene_id)) {
				set_scene_tile_scene(scene_id, p_value);
			} else {
				create_scene_tile(p_value, scene_id);
			}
			return true;
		} else if (components.size() >= 3 && components[2] == "display_placeholder") {
			if (!has_scene_tile_id(scene_id)) {
				create_scene_tile(p_value, scene_id);
			}

			return true;
		}
	}

	return false;
}

bool TileSetScenesCollectionSource::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> components = String(p_name).split("/", true, 2);

	if (components.size() >= 2 && components[0] == "scenes" && components[1].is_valid_int() && scenes.has(components[1].to_int())) {
		if (components.size() >= 3 && components[2] == "scene") {
			r_ret = scenes[components[1].to_int()].scene;
			return true;
		} else if (components.size() >= 3 && components[2] == "display_placeholder") {
			r_ret = scenes[components[1].to_int()].scene;
			return true;
		}
	}

	return false;
}

void TileSetScenesCollectionSource::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < scenes_ids.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, vformat("scenes/%d/scene", scenes_ids[i]), PROPERTY_HINT_RESOURCE_TYPE, "TileSetScenesCollectionSource"));

		PropertyInfo property_info = PropertyInfo(Variant::BOOL, vformat("scenes/%d/display_placeholder", scenes_ids[i]));
		if (scenes[scenes_ids[i]].display_placeholder == false) {
			property_info.usage ^= PROPERTY_USAGE_STORAGE;
		}
		p_list->push_back(property_info);
	}
}

void TileSetScenesCollectionSource::_bind_methods() {
	// Base tiles
	ClassDB::bind_method(D_METHOD("get_tiles_count"), &TileSetScenesCollectionSource::get_tiles_count);
	ClassDB::bind_method(D_METHOD("get_tile_id", "index"), &TileSetScenesCollectionSource::get_tile_id);
	ClassDB::bind_method(D_METHOD("has_tile", "atlas_coords"), &TileSetScenesCollectionSource::has_tile);

	// Alternative tiles
	ClassDB::bind_method(D_METHOD("get_alternative_tiles_count", "atlas_coords"), &TileSetScenesCollectionSource::get_alternative_tiles_count);
	ClassDB::bind_method(D_METHOD("get_alternative_tile_id", "atlas_coords", "index"), &TileSetScenesCollectionSource::get_alternative_tile_id);
	ClassDB::bind_method(D_METHOD("has_alternative_tile", "atlas_coords", "alternative_tile"), &TileSetScenesCollectionSource::has_alternative_tile);

	ClassDB::bind_method(D_METHOD("get_scene_tiles_count"), &TileSetScenesCollectionSource::get_scene_tiles_count);
	ClassDB::bind_method(D_METHOD("get_scene_tile_id", "index"), &TileSetScenesCollectionSource::get_scene_tile_id);
	ClassDB::bind_method(D_METHOD("has_scene_tile_id", "id"), &TileSetScenesCollectionSource::has_scene_tile_id);
	ClassDB::bind_method(D_METHOD("create_scene_tile", "packed_scene", "id_override"), &TileSetScenesCollectionSource::create_scene_tile, DEFVAL(INVALID_TILE_ALTERNATIVE));
	ClassDB::bind_method(D_METHOD("set_scene_tile_id", "id", "new_id"), &TileSetScenesCollectionSource::set_scene_tile_id);
	ClassDB::bind_method(D_METHOD("set_scene_tile_scene", "id", "packed_scene"), &TileSetScenesCollectionSource::set_scene_tile_scene);
	ClassDB::bind_method(D_METHOD("get_scene_tile_scene", "id"), &TileSetScenesCollectionSource::get_scene_tile_scene);
	ClassDB::bind_method(D_METHOD("set_scene_tile_display_placeholder", "id", "display_placeholder"), &TileSetScenesCollectionSource::set_scene_tile_display_placeholder);
	ClassDB::bind_method(D_METHOD("get_scene_tile_display_placeholder", "id"), &TileSetScenesCollectionSource::get_scene_tile_display_placeholder);
	ClassDB::bind_method(D_METHOD("remove_scene_tile", "id"), &TileSetScenesCollectionSource::remove_scene_tile);
	ClassDB::bind_method(D_METHOD("get_next_scene_tile_id"), &TileSetScenesCollectionSource::get_next_scene_tile_id);
}

/////////////////////////////// TileData //////////////////////////////////////

void TileData::set_tile_set(const TileSet *p_tile_set) {
	tile_set = p_tile_set;
	notify_tile_data_properties_should_change();
}

void TileData::notify_tile_data_properties_should_change() {
	if (!tile_set) {
		return;
	}

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
	emit_signal(SNAME("changed"));
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
	emit_signal(SNAME("changed"));
}
bool TileData::get_flip_h() const {
	return flip_h;
}

void TileData::set_flip_v(bool p_flip_v) {
	ERR_FAIL_COND_MSG(!allow_transform && p_flip_v, "Transform is only allowed for alternative tiles (with its alternative_id != 0)");
	flip_v = p_flip_v;
	emit_signal(SNAME("changed"));
}

bool TileData::get_flip_v() const {
	return flip_v;
}

void TileData::set_transpose(bool p_transpose) {
	ERR_FAIL_COND_MSG(!allow_transform && p_transpose, "Transform is only allowed for alternative tiles (with its alternative_id != 0)");
	transpose = p_transpose;
	emit_signal(SNAME("changed"));
}
bool TileData::get_transpose() const {
	return transpose;
}

void TileData::set_texture_offset(Vector2i p_texture_offset) {
	tex_offset = p_texture_offset;
	emit_signal(SNAME("changed"));
}

Vector2i TileData::get_texture_offset() const {
	return tex_offset;
}

void TileData::tile_set_material(Ref<ShaderMaterial> p_material) {
	material = p_material;
	emit_signal(SNAME("changed"));
}
Ref<ShaderMaterial> TileData::tile_get_material() const {
	return material;
}

void TileData::set_modulate(Color p_modulate) {
	modulate = p_modulate;
	emit_signal(SNAME("changed"));
}
Color TileData::get_modulate() const {
	return modulate;
}

void TileData::set_z_index(int p_z_index) {
	z_index = p_z_index;
	emit_signal(SNAME("changed"));
}
int TileData::get_z_index() const {
	return z_index;
}

void TileData::set_y_sort_origin(int p_y_sort_origin) {
	y_sort_origin = p_y_sort_origin;
	emit_signal(SNAME("changed"));
}
int TileData::get_y_sort_origin() const {
	return y_sort_origin;
}

void TileData::set_occluder(int p_layer_id, Ref<OccluderPolygon2D> p_occluder_polygon) {
	ERR_FAIL_INDEX(p_layer_id, occluders.size());
	occluders.write[p_layer_id] = p_occluder_polygon;
	emit_signal(SNAME("changed"));
}

Ref<OccluderPolygon2D> TileData::get_occluder(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, occluders.size(), Ref<OccluderPolygon2D>());
	return occluders[p_layer_id];
}

// Physics
int TileData::get_collision_polygons_count(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, physics.size(), 0);
	return physics[p_layer_id].polygons.size();
}

void TileData::set_collision_polygons_count(int p_layer_id, int p_polygons_count) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	ERR_FAIL_COND(p_polygons_count < 0);
	physics.write[p_layer_id].polygons.resize(p_polygons_count);
	notify_property_list_changed();
	emit_signal(SNAME("changed"));
}

void TileData::add_collision_polygon(int p_layer_id) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	physics.write[p_layer_id].polygons.push_back(PhysicsLayerTileData::PolygonShapeTileData());
	emit_signal(SNAME("changed"));
}

void TileData::remove_collision_polygon(int p_layer_id, int p_polygon_index) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	ERR_FAIL_INDEX(p_polygon_index, physics[p_layer_id].polygons.size());
	physics.write[p_layer_id].polygons.remove(p_polygon_index);
	emit_signal(SNAME("changed"));
}

void TileData::set_collision_polygon_points(int p_layer_id, int p_polygon_index, Vector<Vector2> p_polygon) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	ERR_FAIL_INDEX(p_polygon_index, physics[p_layer_id].polygons.size());
	ERR_FAIL_COND_MSG(p_polygon.size() != 0 && p_polygon.size() < 3, "Invalid polygon. Needs either 0 or more than 3 points.");

	if (p_polygon.is_empty()) {
		physics.write[p_layer_id].polygons.write[p_polygon_index].shapes.clear();
	} else {
		// Decompose into convex shapes.
		Vector<Vector<Vector2>> decomp = Geometry2D::decompose_polygon_in_convex(p_polygon);
		ERR_FAIL_COND_MSG(decomp.is_empty(), "Could not decompose the polygon into convex shapes.");

		physics.write[p_layer_id].polygons.write[p_polygon_index].shapes.resize(decomp.size());
		for (int i = 0; i < decomp.size(); i++) {
			Ref<ConvexPolygonShape2D> shape;
			shape.instantiate();
			shape->set_points(decomp[i]);
			physics.write[p_layer_id].polygons.write[p_polygon_index].shapes[i] = shape;
		}
	}
	physics.write[p_layer_id].polygons.write[p_polygon_index].polygon = p_polygon;
	emit_signal(SNAME("changed"));
}

Vector<Vector2> TileData::get_collision_polygon_points(int p_layer_id, int p_polygon_index) const {
	ERR_FAIL_INDEX_V(p_layer_id, physics.size(), Vector<Vector2>());
	ERR_FAIL_INDEX_V(p_polygon_index, physics[p_layer_id].polygons.size(), Vector<Vector2>());
	return physics[p_layer_id].polygons[p_polygon_index].polygon;
}

void TileData::set_collision_polygon_one_way(int p_layer_id, int p_polygon_index, bool p_one_way) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	ERR_FAIL_INDEX(p_polygon_index, physics[p_layer_id].polygons.size());
	physics.write[p_layer_id].polygons.write[p_polygon_index].one_way = p_one_way;
	emit_signal(SNAME("changed"));
}

bool TileData::is_collision_polygon_one_way(int p_layer_id, int p_polygon_index) const {
	ERR_FAIL_INDEX_V(p_layer_id, physics.size(), false);
	ERR_FAIL_INDEX_V(p_polygon_index, physics[p_layer_id].polygons.size(), false);
	return physics[p_layer_id].polygons[p_polygon_index].one_way;
}

void TileData::set_collision_polygon_one_way_margin(int p_layer_id, int p_polygon_index, float p_one_way_margin) {
	ERR_FAIL_INDEX(p_layer_id, physics.size());
	ERR_FAIL_INDEX(p_polygon_index, physics[p_layer_id].polygons.size());
	physics.write[p_layer_id].polygons.write[p_polygon_index].one_way_margin = p_one_way_margin;
	emit_signal(SNAME("changed"));
}

float TileData::get_collision_polygon_one_way_margin(int p_layer_id, int p_polygon_index) const {
	ERR_FAIL_INDEX_V(p_layer_id, physics.size(), 0.0);
	ERR_FAIL_INDEX_V(p_polygon_index, physics[p_layer_id].polygons.size(), 0.0);
	return physics[p_layer_id].polygons[p_polygon_index].one_way_margin;
}

int TileData::get_collision_polygon_shapes_count(int p_layer_id, int p_polygon_index) const {
	ERR_FAIL_INDEX_V(p_layer_id, physics.size(), 0);
	ERR_FAIL_INDEX_V(p_polygon_index, physics[p_layer_id].polygons.size(), 0);
	return physics[p_layer_id].polygons[p_polygon_index].shapes.size();
}

Ref<ConvexPolygonShape2D> TileData::get_collision_polygon_shape(int p_layer_id, int p_polygon_index, int shape_index) const {
	ERR_FAIL_INDEX_V(p_layer_id, physics.size(), 0);
	ERR_FAIL_INDEX_V(p_polygon_index, physics[p_layer_id].polygons.size(), Ref<ConvexPolygonShape2D>());
	ERR_FAIL_INDEX_V(shape_index, (int)physics[p_layer_id].polygons[shape_index].shapes.size(), Ref<ConvexPolygonShape2D>());
	return physics[p_layer_id].polygons[shape_index].shapes[shape_index];
}

// Terrain
void TileData::set_terrain_set(int p_terrain_set) {
	ERR_FAIL_COND(p_terrain_set < -1);
	if (p_terrain_set == terrain_set) {
		return;
	}
	if (tile_set) {
		ERR_FAIL_COND(p_terrain_set >= tile_set->get_terrain_sets_count());
		for (int i = 0; i < 16; i++) {
			terrain_peering_bits[i] = -1;
		}
	}
	terrain_set = p_terrain_set;
	notify_property_list_changed();
	emit_signal(SNAME("changed"));
}

int TileData::get_terrain_set() const {
	return terrain_set;
}

void TileData::set_peering_bit_terrain(TileSet::CellNeighbor p_peering_bit, int p_terrain_index) {
	ERR_FAIL_COND(terrain_set < 0);
	ERR_FAIL_COND(p_terrain_index < -1);
	if (tile_set) {
		ERR_FAIL_COND(p_terrain_index >= tile_set->get_terrains_count(terrain_set));
		ERR_FAIL_COND(!is_valid_peering_bit_terrain(p_peering_bit));
	}
	terrain_peering_bits[p_peering_bit] = p_terrain_index;
	emit_signal(SNAME("changed"));
}

int TileData::get_peering_bit_terrain(TileSet::CellNeighbor p_peering_bit) const {
	ERR_FAIL_COND_V(!is_valid_peering_bit_terrain(p_peering_bit), -1);
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
	emit_signal(SNAME("changed"));
}

Ref<NavigationPolygon> TileData::get_navigation_polygon(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, navigation.size(), Ref<NavigationPolygon>());
	return navigation[p_layer_id];
}

// Misc
void TileData::set_probability(float p_probability) {
	ERR_FAIL_COND(p_probability < 0.0);
	probability = p_probability;
	emit_signal(SNAME("changed"));
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
	emit_signal(SNAME("changed"));
}

Variant TileData::get_custom_data_by_layer_id(int p_layer_id) const {
	ERR_FAIL_INDEX_V(p_layer_id, custom_data.size(), Variant());
	return custom_data[p_layer_id];
}

bool TileData::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);

	if (components.size() == 2 && components[0].begins_with("occlusion_layer_") && components[0].trim_prefix("occlusion_layer_").is_valid_int()) {
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
	} else if (components.size() >= 2 && components[0].begins_with("physics_layer_") && components[0].trim_prefix("physics_layer_").is_valid_int()) {
		// Physics layers.
		int layer_index = components[0].trim_prefix("physics_layer_").to_int();
		ERR_FAIL_COND_V(layer_index < 0, false);
		if (components.size() == 2 && components[1] == "polygons_count") {
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
			set_collision_polygons_count(layer_index, p_value);
			return true;
		} else if (components.size() == 3 && components[1].begins_with("polygon_") && components[1].trim_prefix("polygon_").is_valid_int()) {
			int polygon_index = components[1].trim_prefix("polygon_").to_int();
			ERR_FAIL_COND_V(polygon_index < 0, false);

			if (components[2] == "points" || components[2] == "one_way" || components[2] == "one_way_margin") {
				if (layer_index >= physics.size()) {
					if (tile_set) {
						return false;
					} else {
						physics.resize(layer_index + 1);
					}
				}

				if (polygon_index >= physics[layer_index].polygons.size()) {
					physics.write[layer_index].polygons.resize(polygon_index + 1);
				}
			}
			if (components[2] == "points") {
				Vector<Vector2> polygon = p_value;
				set_collision_polygon_points(layer_index, polygon_index, polygon);
				return true;
			} else if (components[2] == "one_way") {
				set_collision_polygon_one_way(layer_index, polygon_index, p_value);
				return true;
			} else if (components[2] == "one_way_margin") {
				set_collision_polygon_one_way_margin(layer_index, polygon_index, p_value);
				return true;
			}
		}
	} else if (components.size() == 2 && components[0].begins_with("navigation_layer_") && components[0].trim_prefix("navigation_layer_").is_valid_int()) {
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
		for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(i);
			if (components[1] == TileSet::CELL_NEIGHBOR_ENUM_TO_TEXT[i]) {
				set_peering_bit_terrain(bit, p_value);
				return true;
			}
		}
		return false;
	} else if (components.size() == 1 && components[0].begins_with("custom_data_") && components[0].trim_prefix("custom_data_").is_valid_int()) {
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
		if (components.size() == 2 && components[0].begins_with("occlusion_layer") && components[0].trim_prefix("occlusion_layer_").is_valid_int()) {
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
		} else if (components.size() >= 2 && components[0].begins_with("physics_layer_") && components[0].trim_prefix("physics_layer_").is_valid_int()) {
			// Physics layers.
			int layer_index = components[0].trim_prefix("physics_layer_").to_int();
			ERR_FAIL_COND_V(layer_index < 0, false);
			if (layer_index >= physics.size()) {
				return false;
			}
			if (components.size() == 2 && components[1] == "polygons_count") {
				r_ret = get_collision_polygons_count(layer_index);
				return true;
			} else if (components.size() == 3 && components[1].begins_with("polygon_") && components[1].trim_prefix("polygon_").is_valid_int()) {
				int polygon_index = components[1].trim_prefix("polygon_").to_int();
				ERR_FAIL_COND_V(polygon_index < 0, false);
				if (polygon_index >= physics[layer_index].polygons.size()) {
					return false;
				}
				if (components[2] == "points") {
					r_ret = get_collision_polygon_points(layer_index, polygon_index);
					return true;
				} else if (components[2] == "one_way") {
					r_ret = is_collision_polygon_one_way(layer_index, polygon_index);
					return true;
				} else if (components[2] == "one_way_margin") {
					r_ret = get_collision_polygon_one_way_margin(layer_index, polygon_index);
					return true;
				}
			}
		} else if (components.size() == 2 && components[0] == "terrains_peering_bit") {
			// Terrains.
			for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
				if (components[1] == TileSet::CELL_NEIGHBOR_ENUM_TO_TEXT[i]) {
					r_ret = terrain_peering_bits[i];
					return true;
				}
			}
			return false;
		} else if (components.size() == 2 && components[0].begins_with("navigation_layer_") && components[0].trim_prefix("navigation_layer_").is_valid_int()) {
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
		} else if (components.size() == 1 && components[0].begins_with("custom_data_") && components[0].trim_prefix("custom_data_").is_valid_int()) {
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
	PropertyInfo property_info;
	// Add the groups manually.
	if (tile_set) {
		// Occlusion layers.
		p_list->push_back(PropertyInfo(Variant::NIL, "Rendering", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
		for (int i = 0; i < occluders.size(); i++) {
			// occlusion_layer_%d/polygon
			property_info = PropertyInfo(Variant::OBJECT, vformat("occlusion_layer_%d/polygon", i), PROPERTY_HINT_RESOURCE_TYPE, "OccluderPolygon2D", PROPERTY_USAGE_DEFAULT);
			if (!occluders[i].is_valid()) {
				property_info.usage ^= PROPERTY_USAGE_STORAGE;
			}
			p_list->push_back(property_info);
		}

		// Physics layers.
		p_list->push_back(PropertyInfo(Variant::NIL, "Physics", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
		for (int i = 0; i < physics.size(); i++) {
			p_list->push_back(PropertyInfo(Variant::INT, vformat("physics_layer_%d/polygons_count", i), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));

			for (int j = 0; j < physics[i].polygons.size(); j++) {
				// physics_layer_%d/points
				property_info = PropertyInfo(Variant::ARRAY, vformat("physics_layer_%d/polygon_%d/points", i, j), PROPERTY_HINT_ARRAY_TYPE, "Vector2", PROPERTY_USAGE_DEFAULT);
				if (physics[i].polygons[j].polygon.is_empty()) {
					property_info.usage ^= PROPERTY_USAGE_STORAGE;
				}
				p_list->push_back(property_info);

				// physics_layer_%d/polygon_%d/one_way
				property_info = PropertyInfo(Variant::BOOL, vformat("physics_layer_%d/polygon_%d/one_way", i, j));
				if (physics[i].polygons[j].one_way == false) {
					property_info.usage ^= PROPERTY_USAGE_STORAGE;
				}
				p_list->push_back(property_info);

				// physics_layer_%d/polygon_%d/one_way_margin
				property_info = PropertyInfo(Variant::FLOAT, vformat("physics_layer_%d/polygon_%d/one_way_margin", i, j));
				if (physics[i].polygons[j].one_way_margin == 1.0) {
					property_info.usage ^= PROPERTY_USAGE_STORAGE;
				}
				p_list->push_back(property_info);
			}
		}

		// Terrain data
		if (terrain_set >= 0) {
			p_list->push_back(PropertyInfo(Variant::NIL, "Terrains", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
			for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
				TileSet::CellNeighbor bit = TileSet::CellNeighbor(i);
				if (is_valid_peering_bit_terrain(bit)) {
					property_info = PropertyInfo(Variant::INT, "terrains_peering_bit/" + String(TileSet::CELL_NEIGHBOR_ENUM_TO_TEXT[i]));
					if (get_peering_bit_terrain(bit) == -1) {
						property_info.usage ^= PROPERTY_USAGE_STORAGE;
					}
					p_list->push_back(property_info);
				}
			}
		}

		// Navigation layers.
		p_list->push_back(PropertyInfo(Variant::NIL, "Navigation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
		for (int i = 0; i < navigation.size(); i++) {
			property_info = PropertyInfo(Variant::OBJECT, vformat("navigation_layer_%d/polygon", i), PROPERTY_HINT_RESOURCE_TYPE, "NavigationPolygon", PROPERTY_USAGE_DEFAULT);
			if (!navigation[i].is_valid()) {
				property_info.usage ^= PROPERTY_USAGE_STORAGE;
			}
			p_list->push_back(property_info);
		}

		// Custom data layers.
		p_list->push_back(PropertyInfo(Variant::NIL, "Custom data", PROPERTY_HINT_NONE, "custom_data_", PROPERTY_USAGE_GROUP));
		for (int i = 0; i < custom_data.size(); i++) {
			Variant default_val;
			Callable::CallError error;
			Variant::construct(custom_data[i].get_type(), default_val, nullptr, 0, error);
			property_info = PropertyInfo(tile_set->get_custom_data_type(i), vformat("custom_data_%d", i), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT);
			if (custom_data[i] == default_val) {
				property_info.usage ^= PROPERTY_USAGE_STORAGE;
			}
			p_list->push_back(property_info);
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
	ClassDB::bind_method(D_METHOD("get_collision_polygons_count", "layer_id"), &TileData::get_collision_polygons_count);
	ClassDB::bind_method(D_METHOD("set_collision_polygons_count", "layer_id", "polygons_count"), &TileData::set_collision_polygons_count);
	ClassDB::bind_method(D_METHOD("add_collision_polygon", "layer_id"), &TileData::add_collision_polygon);
	ClassDB::bind_method(D_METHOD("remove_collision_polygon", "layer_id", "polygon_index"), &TileData::remove_collision_polygon);
	ClassDB::bind_method(D_METHOD("set_collision_polygon_points", "layer_id", "polygon_index", "polygon"), &TileData::set_collision_polygon_points);
	ClassDB::bind_method(D_METHOD("get_collision_polygon_points", "layer_id", "polygon_index"), &TileData::get_collision_polygon_points);
	ClassDB::bind_method(D_METHOD("set_collision_polygon_one_way", "layer_id", "polygon_index", "one_way"), &TileData::set_collision_polygon_one_way);
	ClassDB::bind_method(D_METHOD("is_collision_polygon_one_way", "layer_id", "polygon_index"), &TileData::is_collision_polygon_one_way);
	ClassDB::bind_method(D_METHOD("set_collision_polygon_one_way_margin", "layer_id", "polygon_index", "one_way_margin"), &TileData::set_collision_polygon_one_way_margin);
	ClassDB::bind_method(D_METHOD("get_collision_polygon_one_way_margin", "layer_id", "polygon_index"), &TileData::get_collision_polygon_one_way_margin);

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
	ADD_PROPERTY(PropertyInfo(Variant::INT, "y_sort_origin"), "set_y_sort_origin", "get_y_sort_origin");

	ADD_GROUP("Terrains", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "terrain_set"), "set_terrain_set", "get_terrain_set");

	ADD_GROUP("Miscellaneous", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "probability"), "set_probability", "get_probability");

	ADD_SIGNAL(MethodInfo("changed"));
}
/////////////////////////////// TileSetPluginAtlasRendering //////////////////////////////////////

void TileSetPluginAtlasRendering::tilemap_notification(TileMap *p_tile_map, int p_what) {
	switch (p_what) {
		case CanvasItem::NOTIFICATION_VISIBILITY_CHANGED: {
			bool visible = p_tile_map->is_visible_in_tree();
			for (Map<Vector2i, TileMapQuadrant>::Element *E_quadrant = p_tile_map->get_quadrant_map().front(); E_quadrant; E_quadrant = E_quadrant->next()) {
				TileMapQuadrant &q = E_quadrant->get();

				// Update occluders transform.
				for (Map<Vector2i, Vector2i, TileMapQuadrant::CoordsWorldComparator>::Element *E_cell = q.world_to_map.front(); E_cell; E_cell = E_cell->next()) {
					Transform2D xform;
					xform.set_origin(E_cell->key());
					for (List<RID>::Element *E_occluder_id = q.occluders.front(); E_occluder_id; E_occluder_id = E_occluder_id->next()) {
						RS::get_singleton()->canvas_light_occluder_set_enabled(E_occluder_id->get(), visible);
					}
				}
			}
		} break;
		case CanvasItem::NOTIFICATION_TRANSFORM_CHANGED: {
			if (!p_tile_map->is_inside_tree()) {
				return;
			}

			for (Map<Vector2i, TileMapQuadrant>::Element *E_quadrant = p_tile_map->get_quadrant_map().front(); E_quadrant; E_quadrant = E_quadrant->next()) {
				TileMapQuadrant &q = E_quadrant->get();

				// Update occluders transform.
				for (Map<Vector2i, Vector2i, TileMapQuadrant::CoordsWorldComparator>::Element *E_cell = q.world_to_map.front(); E_cell; E_cell = E_cell->next()) {
					Transform2D xform;
					xform.set_origin(E_cell->key());
					for (List<RID>::Element *E_occluder_id = q.occluders.front(); E_occluder_id; E_occluder_id = E_occluder_id->next()) {
						RS::get_singleton()->canvas_light_occluder_set_transform(E_occluder_id->get(), p_tile_map->get_global_transform() * xform);
					}
				}
			}
		} break;
		case CanvasItem::NOTIFICATION_DRAW: {
			Ref<TileSet> tile_set = p_tile_map->get_tileset();
			if (tile_set.is_valid() || p_tile_map->is_y_sort_enabled()) {
				RenderingServer::get_singleton()->canvas_item_set_sort_children_by_y(p_tile_map->get_canvas_item(), tile_set->is_y_sorting() || p_tile_map->is_y_sort_enabled());
			}
		} break;
	}
}

void TileSetPluginAtlasRendering::draw_tile(RID p_canvas_item, Vector2i p_position, const Ref<TileSet> p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, Color p_modulation) {
	ERR_FAIL_COND(!p_tile_set.is_valid());
	ERR_FAIL_COND(!p_tile_set->has_source(p_atlas_source_id));
	ERR_FAIL_COND(!p_tile_set->get_source(p_atlas_source_id)->has_tile(p_atlas_coords));
	ERR_FAIL_COND(!p_tile_set->get_source(p_atlas_source_id)->has_alternative_tile(p_atlas_coords, p_alternative_tile));

	TileSetSource *source = *p_tile_set->get_source(p_atlas_source_id);
	TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
	if (atlas_source) {
		// Get the texture.
		Ref<Texture2D> tex = atlas_source->get_texture();
		if (!tex.is_valid()) {
			return;
		}

		// Check if we are in the texture, return otherwise.
		Vector2i grid_size = atlas_source->get_atlas_grid_size();
		if (p_atlas_coords.x >= grid_size.x || p_atlas_coords.y >= grid_size.y) {
			return;
		}

		// Get tile data.
		TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(p_atlas_coords, p_alternative_tile));

		// Compute the offset
		Rect2i source_rect = atlas_source->get_tile_texture_region(p_atlas_coords);
		Vector2i tile_offset = atlas_source->get_tile_effective_texture_offset(p_atlas_coords, p_alternative_tile);

		// Compute the destination rectangle in the CanvasItem.
		Rect2 dest_rect;
		dest_rect.size = source_rect.size;
		dest_rect.size.x += fp_adjust;
		dest_rect.size.y += fp_adjust;

		bool transpose = tile_data->get_transpose();
		if (transpose) {
			dest_rect.position = (p_position - Vector2(dest_rect.size.y, dest_rect.size.x) / 2 - tile_offset);
		} else {
			dest_rect.position = (p_position - dest_rect.size / 2 - tile_offset);
		}

		if (tile_data->get_flip_h()) {
			dest_rect.size.x = -dest_rect.size.x;
		}

		if (tile_data->get_flip_v()) {
			dest_rect.size.y = -dest_rect.size.y;
		}

		// Get the tile modulation.
		Color modulate = tile_data->get_modulate();
		modulate = Color(modulate.r * p_modulation.r, modulate.g * p_modulation.g, modulate.b * p_modulation.b, modulate.a * p_modulation.a);

		// Draw the tile.
		tex->draw_rect_region(p_canvas_item, dest_rect, source_rect, modulate, transpose, p_tile_set->is_uv_clipping());
	}
}

void TileSetPluginAtlasRendering::update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
	ERR_FAIL_COND(!p_tile_map);
	ERR_FAIL_COND(!p_tile_map->is_inside_tree());
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	bool visible = p_tile_map->is_visible_in_tree();

	SelfList<TileMapQuadrant> *q_list_element = r_dirty_quadrant_list.first();
	while (q_list_element) {
		TileMapQuadrant &q = *q_list_element->self();

		RenderingServer *rs = RenderingServer::get_singleton();

		// Free the canvas items.
		for (List<RID>::Element *E = q.canvas_items.front(); E; E = E->next()) {
			rs->free(E->get());
		}
		q.canvas_items.clear();

		// Free the occluders.
		for (List<RID>::Element *E = q.occluders.front(); E; E = E->next()) {
			rs->free(E->get());
		}
		q.occluders.clear();

		// Those allow to group cell per material or z-index.
		Ref<ShaderMaterial> prev_material;
		int prev_z_index = 0;
		RID prev_canvas_item;

		// Iterate over the cells of the quadrant.
		for (Map<Vector2i, Vector2i, TileMapQuadrant::CoordsWorldComparator>::Element *E_cell = q.world_to_map.front(); E_cell; E_cell = E_cell->next()) {
			TileMapCell c = p_tile_map->get_cell(E_cell->value(), true);

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					// Get the tile data.
					TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile));
					Ref<ShaderMaterial> mat = tile_data->tile_get_material();
					int z_index = tile_data->get_z_index();

					// Quandrant pos.
					Vector2 position = p_tile_map->map_to_world(q.coords * p_tile_map->get_effective_quadrant_size());
					if (tile_set->is_y_sorting()) {
						// When Y-sorting, the quandrant size is sure to be 1, we can thus offset the CanvasItem.
						position.y += tile_data->get_y_sort_origin();
					}

					// --- CanvasItems ---
					// Create two canvas items, for rendering and debug.
					RID canvas_item;

					// Check if the material or the z_index changed.
					if (prev_canvas_item == RID() || prev_material != mat || prev_z_index != z_index) {
						// If so, create a new CanvasItem.
						canvas_item = rs->canvas_item_create();
						if (mat.is_valid()) {
							rs->canvas_item_set_material(canvas_item, mat->get_rid());
						}
						rs->canvas_item_set_parent(canvas_item, p_tile_map->get_canvas_item());
						rs->canvas_item_set_use_parent_material(canvas_item, p_tile_map->get_use_parent_material() || p_tile_map->get_material().is_valid());

						Transform2D xform;
						xform.set_origin(position);
						rs->canvas_item_set_transform(canvas_item, xform);

						rs->canvas_item_set_light_mask(canvas_item, p_tile_map->get_light_mask());
						rs->canvas_item_set_z_index(canvas_item, z_index);

						rs->canvas_item_set_default_texture_filter(canvas_item, RS::CanvasItemTextureFilter(p_tile_map->CanvasItem::get_texture_filter()));
						rs->canvas_item_set_default_texture_repeat(canvas_item, RS::CanvasItemTextureRepeat(p_tile_map->CanvasItem::get_texture_repeat()));

						q.canvas_items.push_back(canvas_item);

						prev_canvas_item = canvas_item;
						prev_material = mat;
						prev_z_index = z_index;

					} else {
						// Keep the same canvas_item to draw on.
						canvas_item = prev_canvas_item;
					}

					// Drawing the tile in the canvas item.
					draw_tile(canvas_item, E_cell->key() - position, tile_set, c.source_id, c.get_atlas_coords(), c.alternative_tile, p_tile_map->get_self_modulate());

					// --- Occluders ---
					for (int i = 0; i < tile_set->get_occlusion_layers_count(); i++) {
						Transform2D xform;
						xform.set_origin(E_cell->key());
						if (tile_data->get_occluder(i).is_valid()) {
							RID occluder_id = rs->canvas_light_occluder_create();
							rs->canvas_light_occluder_set_enabled(occluder_id, visible);
							rs->canvas_light_occluder_set_transform(occluder_id, p_tile_map->get_global_transform() * xform);
							rs->canvas_light_occluder_set_polygon(occluder_id, tile_data->get_occluder(i)->get_rid());
							rs->canvas_light_occluder_attach_to_canvas(occluder_id, p_tile_map->get_canvas());
							rs->canvas_light_occluder_set_light_mask(occluder_id, tile_set->get_occlusion_layer_light_mask(i));
							q.occluders.push_back(occluder_id);
						}
					}
				}
			}
		}

		quadrant_order_dirty = true;
		q_list_element = q_list_element->next();
	}

	// Reset the drawing indices
	if (quadrant_order_dirty) {
		int index = -(int64_t)0x80000000; //always must be drawn below children.

		// Sort the quadrants coords per world coordinates
		Map<Vector2i, Vector2i, TileMapQuadrant::CoordsWorldComparator> world_to_map;
		Map<Vector2i, TileMapQuadrant> quadrant_map = p_tile_map->get_quadrant_map();
		for (Map<Vector2i, TileMapQuadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
			world_to_map[p_tile_map->map_to_world(E->key())] = E->key();
		}

		// Sort the quadrants
		for (Map<Vector2i, Vector2i, TileMapQuadrant::CoordsWorldComparator>::Element *E = world_to_map.front(); E; E = E->next()) {
			TileMapQuadrant &q = quadrant_map[E->value()];
			for (List<RID>::Element *F = q.canvas_items.front(); F; F = F->next()) {
				RS::get_singleton()->canvas_item_set_draw_index(F->get(), index++);
			}
		}

		quadrant_order_dirty = false;
	}
}

void TileSetPluginAtlasRendering::create_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	quadrant_order_dirty = true;
}

void TileSetPluginAtlasRendering::cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	// Free the canvas items.
	for (List<RID>::Element *E = p_quadrant->canvas_items.front(); E; E = E->next()) {
		RenderingServer::get_singleton()->free(E->get());
	}
	p_quadrant->canvas_items.clear();

	// Free the occluders.
	for (List<RID>::Element *E = p_quadrant->occluders.front(); E; E = E->next()) {
		RenderingServer::get_singleton()->free(E->get());
	}
	p_quadrant->occluders.clear();
}

void TileSetPluginAtlasRendering::draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	// Draw a placeholder for scenes needing one.
	RenderingServer *rs = RenderingServer::get_singleton();
	Vector2 quadrant_pos = p_tile_map->map_to_world(p_quadrant->coords * p_tile_map->get_effective_quadrant_size());
	for (Set<Vector2i>::Element *E_cell = p_quadrant->cells.front(); E_cell; E_cell = E_cell->next()) {
		const TileMapCell &c = p_tile_map->get_cell(E_cell->get(), true);

		TileSetSource *source;
		if (tile_set->has_source(c.source_id)) {
			source = *tile_set->get_source(c.source_id);

			if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
				continue;
			}

			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				Vector2i grid_size = atlas_source->get_atlas_grid_size();
				if (!atlas_source->get_texture().is_valid() || c.get_atlas_coords().x >= grid_size.x || c.get_atlas_coords().y >= grid_size.y) {
					// Generate a random color from the hashed values of the tiles.
					Array to_hash;
					to_hash.push_back(c.source_id);
					to_hash.push_back(c.get_atlas_coords());
					to_hash.push_back(c.alternative_tile);
					uint32_t hash = RandomPCG(to_hash.hash()).rand();

					Color color;
					color = color.from_hsv(
							(float)((hash >> 24) & 0xFF) / 256.0,
							Math::lerp(0.5, 1.0, (float)((hash >> 16) & 0xFF) / 256.0),
							Math::lerp(0.5, 1.0, (float)((hash >> 8) & 0xFF) / 256.0),
							0.8);

					// Draw a placeholder tile.
					Transform2D xform;
					xform.set_origin(p_tile_map->map_to_world(E_cell->get()) - quadrant_pos);
					rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, xform);
					rs->canvas_item_add_circle(p_quadrant->debug_canvas_item, Vector2(), MIN(tile_set->get_tile_size().x, tile_set->get_tile_size().y) / 4.0, color);
				}
			}
		}
	}
}

/////////////////////////////// TileSetPluginAtlasPhysics //////////////////////////////////////

void TileSetPluginAtlasPhysics::tilemap_notification(TileMap *p_tile_map, int p_what) {
	switch (p_what) {
		case CanvasItem::NOTIFICATION_TRANSFORM_CHANGED: {
			// Update the bodies transforms.
			if (p_tile_map->is_inside_tree()) {
				Map<Vector2i, TileMapQuadrant> quadrant_map = p_tile_map->get_quadrant_map();
				Transform2D global_transform = p_tile_map->get_global_transform();

				for (Map<Vector2i, TileMapQuadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
					TileMapQuadrant &q = E->get();

					Transform2D xform;
					xform.set_origin(p_tile_map->map_to_world(E->key() * p_tile_map->get_effective_quadrant_size()));
					xform = global_transform * xform;

					for (int body_index = 0; body_index < q.bodies.size(); body_index++) {
						PhysicsServer2D::get_singleton()->body_set_state(q.bodies[body_index], PhysicsServer2D::BODY_STATE_TRANSFORM, xform);
					}
				}
			}
		} break;
	}
}

void TileSetPluginAtlasPhysics::update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
	ERR_FAIL_COND(!p_tile_map);
	ERR_FAIL_COND(!p_tile_map->is_inside_tree());
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	Transform2D global_transform = p_tile_map->get_global_transform();
	PhysicsServer2D *ps = PhysicsServer2D::get_singleton();

	SelfList<TileMapQuadrant> *q_list_element = r_dirty_quadrant_list.first();
	while (q_list_element) {
		TileMapQuadrant &q = *q_list_element->self();

		Vector2 quadrant_pos = p_tile_map->map_to_world(q.coords * p_tile_map->get_effective_quadrant_size());

		// Clear shapes.
		for (int body_index = 0; body_index < q.bodies.size(); body_index++) {
			ps->body_clear_shapes(q.bodies[body_index]);

			// Position the bodies.
			Transform2D xform;
			xform.set_origin(quadrant_pos);
			xform = global_transform * xform;
			ps->body_set_state(q.bodies[body_index], PhysicsServer2D::BODY_STATE_TRANSFORM, xform);
		}

		for (Set<Vector2i>::Element *E_cell = q.cells.front(); E_cell; E_cell = E_cell->next()) {
			TileMapCell c = p_tile_map->get_cell(E_cell->get(), true);

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile));

					for (int body_index = 0; body_index < q.bodies.size(); body_index++) {
						// Add the shapes again.
						for (int polygon_index = 0; polygon_index < tile_data->get_collision_polygons_count(body_index); polygon_index++) {
							bool one_way_collision = tile_data->is_collision_polygon_one_way(body_index, polygon_index);
							float one_way_collision_margin = tile_data->get_collision_polygon_one_way_margin(body_index, polygon_index);

							int shapes_count = tile_data->get_collision_polygon_shapes_count(body_index, polygon_index);
							for (int shape_index = 0; shape_index < shapes_count; shape_index++) {
								Transform2D xform = Transform2D();
								xform.set_origin(p_tile_map->map_to_world(E_cell->get()) - quadrant_pos);

								// Add decomposed convex shapes.
								Ref<ConvexPolygonShape2D> shape = tile_data->get_collision_polygon_shape(body_index, polygon_index, shape_index);
								ps->body_add_shape(q.bodies[body_index], shape->get_rid(), xform);
								ps->body_set_shape_metadata(q.bodies[body_index], shape_index, E_cell->get());
								ps->body_set_shape_as_one_way_collision(q.bodies[body_index], shape_index, one_way_collision, one_way_collision_margin);
							}
						}
					}
				}
			}
		}

		q_list_element = q_list_element->next();
	}
}

void TileSetPluginAtlasPhysics::create_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	//Get the TileMap's gobla transform.
	Transform2D global_transform;
	if (p_tile_map->is_inside_tree()) {
		global_transform = p_tile_map->get_global_transform();
	}

	// Clear all bodies.
	p_quadrant->bodies.clear();

	// Create the body and set its parameters.
	for (int layer_index = 0; layer_index < tile_set->get_physics_layers_count(); layer_index++) {
		RID body = PhysicsServer2D::get_singleton()->body_create();
		PhysicsServer2D::get_singleton()->body_set_mode(body, PhysicsServer2D::BODY_MODE_STATIC);

		PhysicsServer2D::get_singleton()->body_attach_object_instance_id(body, p_tile_map->get_instance_id());
		PhysicsServer2D::get_singleton()->body_set_collision_layer(body, tile_set->get_physics_layer_collision_layer(layer_index));
		PhysicsServer2D::get_singleton()->body_set_collision_mask(body, tile_set->get_physics_layer_collision_mask(layer_index));

		Ref<PhysicsMaterial> physics_material = tile_set->get_physics_layer_physics_material(layer_index);
		if (!physics_material.is_valid()) {
			PhysicsServer2D::get_singleton()->body_set_param(body, PhysicsServer2D::BODY_PARAM_BOUNCE, 0);
			PhysicsServer2D::get_singleton()->body_set_param(body, PhysicsServer2D::BODY_PARAM_FRICTION, 1);
		} else {
			PhysicsServer2D::get_singleton()->body_set_param(body, PhysicsServer2D::BODY_PARAM_BOUNCE, physics_material->computed_bounce());
			PhysicsServer2D::get_singleton()->body_set_param(body, PhysicsServer2D::BODY_PARAM_FRICTION, physics_material->computed_friction());
		}

		if (p_tile_map->is_inside_tree()) {
			RID space = p_tile_map->get_world_2d()->get_space();
			PhysicsServer2D::get_singleton()->body_set_space(body, space);

			Transform2D xform;
			xform.set_origin(p_tile_map->map_to_world(p_quadrant->coords * p_tile_map->get_effective_quadrant_size()));
			xform = global_transform * xform;
			PhysicsServer2D::get_singleton()->body_set_state(body, PhysicsServer2D::BODY_STATE_TRANSFORM, xform);
		}

		p_quadrant->bodies.push_back(body);
	}
}

void TileSetPluginAtlasPhysics::cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	// Remove a quadrant.
	for (int body_index = 0; body_index < p_quadrant->bodies.size(); body_index++) {
		PhysicsServer2D::get_singleton()->free(p_quadrant->bodies[body_index]);
	}
	p_quadrant->bodies.clear();
}

void TileSetPluginAtlasPhysics::draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	// Draw the debug collision shapes.
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!p_tile_map->get_tree()) {
		return;
	}

	bool show_collision = false;
	switch (p_tile_map->get_collision_visibility_mode()) {
		case TileMap::VISIBILITY_MODE_DEFAULT:
			show_collision = !Engine::get_singleton()->is_editor_hint() && (p_tile_map->get_tree() && p_tile_map->get_tree()->is_debugging_navigation_hint());
			break;
		case TileMap::VISIBILITY_MODE_FORCE_HIDE:
			show_collision = false;
			break;
		case TileMap::VISIBILITY_MODE_FORCE_SHOW:
			show_collision = true;
			break;
	}
	if (!show_collision) {
		return;
	}

	RenderingServer *rs = RenderingServer::get_singleton();

	Vector2 quadrant_pos = p_tile_map->map_to_world(p_quadrant->coords * p_tile_map->get_effective_quadrant_size());

	Color debug_collision_color = p_tile_map->get_tree()->get_debug_collisions_color();
	for (Set<Vector2i>::Element *E_cell = p_quadrant->cells.front(); E_cell; E_cell = E_cell->next()) {
		TileMapCell c = p_tile_map->get_cell(E_cell->get(), true);

		Transform2D xform;
		xform.set_origin(p_tile_map->map_to_world(E_cell->get()) - quadrant_pos);
		rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, xform);

		if (tile_set->has_source(c.source_id)) {
			TileSetSource *source = *tile_set->get_source(c.source_id);

			if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
				continue;
			}

			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile));

				for (int body_index = 0; body_index < p_quadrant->bodies.size(); body_index++) {
					for (int polygon_index = 0; polygon_index < tile_data->get_collision_polygons_count(body_index); polygon_index++) {
						// Draw the debug polygon.
						Vector<Vector2> polygon = tile_data->get_collision_polygon_points(body_index, polygon_index);
						if (polygon.size() >= 3) {
							Vector<Color> color;
							color.push_back(debug_collision_color);
							rs->canvas_item_add_polygon(p_quadrant->debug_canvas_item, polygon, color);
						}
					}
				}
			}
		}
		rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, Transform2D());
	}
};

/////////////////////////////// TileSetPluginAtlasNavigation //////////////////////////////////////

void TileSetPluginAtlasNavigation::tilemap_notification(TileMap *p_tile_map, int p_what) {
	switch (p_what) {
		case CanvasItem::NOTIFICATION_TRANSFORM_CHANGED: {
			if (p_tile_map->is_inside_tree()) {
				Map<Vector2i, TileMapQuadrant> quadrant_map = p_tile_map->get_quadrant_map();
				Transform2D tilemap_xform = p_tile_map->get_global_transform();
				for (Map<Vector2i, TileMapQuadrant>::Element *E_quadrant = quadrant_map.front(); E_quadrant; E_quadrant = E_quadrant->next()) {
					TileMapQuadrant &q = E_quadrant->get();
					for (Map<Vector2i, Vector<RID>>::Element *E_region = q.navigation_regions.front(); E_region; E_region = E_region->next()) {
						for (int layer_index = 0; layer_index < E_region->get().size(); layer_index++) {
							RID region = E_region->get()[layer_index];
							if (!region.is_valid()) {
								continue;
							}
							Transform2D tile_transform;
							tile_transform.set_origin(p_tile_map->map_to_world(E_region->key()));
							NavigationServer2D::get_singleton()->region_set_transform(region, tilemap_xform * tile_transform);
						}
					}
				}
			}
		} break;
	}
}

void TileSetPluginAtlasNavigation::update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
	ERR_FAIL_COND(!p_tile_map);
	ERR_FAIL_COND(!p_tile_map->is_inside_tree());
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	// Get colors for debug.
	SceneTree *st = SceneTree::get_singleton();
	Color debug_navigation_color;
	bool debug_navigation = st && st->is_debugging_navigation_hint();
	if (debug_navigation) {
		debug_navigation_color = st->get_debug_navigation_color();
	}

	Transform2D tilemap_xform = p_tile_map->get_global_transform();
	SelfList<TileMapQuadrant> *q_list_element = r_dirty_quadrant_list.first();
	while (q_list_element) {
		TileMapQuadrant &q = *q_list_element->self();

		// Clear navigation shapes in the quadrant.
		for (Map<Vector2i, Vector<RID>>::Element *E = q.navigation_regions.front(); E; E = E->next()) {
			for (int i = 0; i < E->get().size(); i++) {
				RID region = E->get()[i];
				if (!region.is_valid()) {
					continue;
				}
				NavigationServer2D::get_singleton()->region_set_map(region, RID());
			}
		}
		q.navigation_regions.clear();

		// Get the navigation polygons and create regions.
		for (Set<Vector2i>::Element *E_cell = q.cells.front(); E_cell; E_cell = E_cell->next()) {
			TileMapCell c = p_tile_map->get_cell(E_cell->get(), true);

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile));
					q.navigation_regions[E_cell->get()].resize(tile_set->get_navigation_layers_count());

					for (int layer_index = 0; layer_index < tile_set->get_navigation_layers_count(); layer_index++) {
						Ref<NavigationPolygon> navpoly;
						navpoly = tile_data->get_navigation_polygon(layer_index);

						if (navpoly.is_valid()) {
							Transform2D tile_transform;
							tile_transform.set_origin(p_tile_map->map_to_world(E_cell->get()));

							RID region = NavigationServer2D::get_singleton()->region_create();
							NavigationServer2D::get_singleton()->region_set_map(region, p_tile_map->get_world_2d()->get_navigation_map());
							NavigationServer2D::get_singleton()->region_set_transform(region, tilemap_xform * tile_transform);
							NavigationServer2D::get_singleton()->region_set_navpoly(region, navpoly);
							q.navigation_regions[E_cell->get()].write[layer_index] = region;
						}
					}
				}
			}
		}

		q_list_element = q_list_element->next();
	}
}

void TileSetPluginAtlasNavigation::cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	// Clear navigation shapes in the quadrant.
	for (Map<Vector2i, Vector<RID>>::Element *E = p_quadrant->navigation_regions.front(); E; E = E->next()) {
		for (int i = 0; i < E->get().size(); i++) {
			RID region = E->get()[i];
			if (!region.is_valid()) {
				continue;
			}
			NavigationServer2D::get_singleton()->free(region);
		}
	}
	p_quadrant->navigation_regions.clear();
}

void TileSetPluginAtlasNavigation::draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	// Draw the debug collision shapes.
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!p_tile_map->get_tree()) {
		return;
	}

	bool show_navigation = false;
	switch (p_tile_map->get_navigation_visibility_mode()) {
		case TileMap::VISIBILITY_MODE_DEFAULT:
			show_navigation = !Engine::get_singleton()->is_editor_hint() && (p_tile_map->get_tree() && p_tile_map->get_tree()->is_debugging_navigation_hint());
			break;
		case TileMap::VISIBILITY_MODE_FORCE_HIDE:
			show_navigation = false;
			break;
		case TileMap::VISIBILITY_MODE_FORCE_SHOW:
			show_navigation = true;
			break;
	}
	if (!show_navigation) {
		return;
	}

	RenderingServer *rs = RenderingServer::get_singleton();

	Color color = p_tile_map->get_tree()->get_debug_navigation_color();
	RandomPCG rand;

	Vector2 quadrant_pos = p_tile_map->map_to_world(p_quadrant->coords * p_tile_map->get_effective_quadrant_size());

	for (Set<Vector2i>::Element *E_cell = p_quadrant->cells.front(); E_cell; E_cell = E_cell->next()) {
		TileMapCell c = p_tile_map->get_cell(E_cell->get(), true);

		TileSetSource *source;
		if (tile_set->has_source(c.source_id)) {
			source = *tile_set->get_source(c.source_id);

			if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
				continue;
			}

			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile));

				Transform2D xform;
				xform.set_origin(p_tile_map->map_to_world(E_cell->get()) - quadrant_pos);
				rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, xform);

				for (int layer_index = 0; layer_index < tile_set->get_navigation_layers_count(); layer_index++) {
					Ref<NavigationPolygon> navpoly = tile_data->get_navigation_polygon(layer_index);
					if (navpoly.is_valid()) {
						PackedVector2Array navigation_polygon_vertices = navpoly->get_vertices();

						for (int i = 0; i < navpoly->get_polygon_count(); i++) {
							// An array of vertices for this polygon.
							Vector<int> polygon = navpoly->get_polygon(i);
							Vector<Vector2> vertices;
							vertices.resize(polygon.size());
							for (int j = 0; j < polygon.size(); j++) {
								ERR_FAIL_INDEX(polygon[j], navigation_polygon_vertices.size());
								vertices.write[j] = navigation_polygon_vertices[polygon[j]];
							}

							// Generate the polygon color, slightly randomly modified from the settings one.
							Color random_variation_color;
							random_variation_color.set_hsv(color.get_h() + rand.random(-1.0, 1.0) * 0.05, color.get_s(), color.get_v() + rand.random(-1.0, 1.0) * 0.1);
							random_variation_color.a = color.a;
							Vector<Color> colors;
							colors.push_back(random_variation_color);

							rs->canvas_item_add_polygon(p_quadrant->debug_canvas_item, vertices, colors);
						}
					}
				}
			}
		}
	}
}

/////////////////////////////// TileSetPluginScenesCollections //////////////////////////////////////

void TileSetPluginScenesCollections::update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	SelfList<TileMapQuadrant> *q_list_element = r_dirty_quadrant_list.first();
	while (q_list_element) {
		TileMapQuadrant &q = *q_list_element->self();

		// Clear the scenes.
		for (Map<Vector2i, String>::Element *E = q.scenes.front(); E; E = E->next()) {
			Node *node = p_tile_map->get_node(E->get());
			if (node) {
				node->queue_delete();
			}
		}

		q.scenes.clear();

		// Recreate the scenes.
		for (Set<Vector2i>::Element *E_cell = q.cells.front(); E_cell; E_cell = E_cell->next()) {
			const TileMapCell &c = p_tile_map->get_cell(E_cell->get(), true);

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetScenesCollectionSource *scenes_collection_source = Object::cast_to<TileSetScenesCollectionSource>(source);
				if (scenes_collection_source) {
					Ref<PackedScene> packed_scene = scenes_collection_source->get_scene_tile_scene(c.alternative_tile);
					if (packed_scene.is_valid()) {
						Node *scene = packed_scene->instantiate();
						p_tile_map->add_child(scene);
						Control *scene_as_control = Object::cast_to<Control>(scene);
						Node2D *scene_as_node2d = Object::cast_to<Node2D>(scene);
						if (scene_as_control) {
							scene_as_control->set_position(p_tile_map->map_to_world(E_cell->get()) + scene_as_control->get_position());
						} else if (scene_as_node2d) {
							Transform2D xform;
							xform.set_origin(p_tile_map->map_to_world(E_cell->get()));
							scene_as_node2d->set_transform(xform * scene_as_node2d->get_transform());
						}
						q.scenes[E_cell->get()] = scene->get_name();
					}
				}
			}
		}

		q_list_element = q_list_element->next();
	}
}

void TileSetPluginScenesCollections::cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	// Clear the scenes.
	for (Map<Vector2i, String>::Element *E = p_quadrant->scenes.front(); E; E = E->next()) {
		Node *node = p_tile_map->get_node(E->get());
		if (node) {
			node->queue_delete();
		}
	}

	p_quadrant->scenes.clear();
}

void TileSetPluginScenesCollections::draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	// Draw a placeholder for scenes needing one.
	RenderingServer *rs = RenderingServer::get_singleton();
	Vector2 quadrant_pos = p_tile_map->map_to_world(p_quadrant->coords * p_tile_map->get_effective_quadrant_size());
	for (Set<Vector2i>::Element *E_cell = p_quadrant->cells.front(); E_cell; E_cell = E_cell->next()) {
		const TileMapCell &c = p_tile_map->get_cell(E_cell->get(), true);

		TileSetSource *source;
		if (tile_set->has_source(c.source_id)) {
			source = *tile_set->get_source(c.source_id);

			if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
				continue;
			}

			TileSetScenesCollectionSource *scenes_collection_source = Object::cast_to<TileSetScenesCollectionSource>(source);
			if (scenes_collection_source) {
				if (!scenes_collection_source->get_scene_tile_scene(c.alternative_tile).is_valid() || scenes_collection_source->get_scene_tile_display_placeholder(c.alternative_tile)) {
					// Generate a random color from the hashed values of the tiles.
					Array to_hash;
					to_hash.push_back(c.source_id);
					to_hash.push_back(c.alternative_tile);
					uint32_t hash = RandomPCG(to_hash.hash()).rand();

					Color color;
					color = color.from_hsv(
							(float)((hash >> 24) & 0xFF) / 256.0,
							Math::lerp(0.5, 1.0, (float)((hash >> 16) & 0xFF) / 256.0),
							Math::lerp(0.5, 1.0, (float)((hash >> 8) & 0xFF) / 256.0),
							0.8);

					// Draw a placeholder tile.
					Transform2D xform;
					xform.set_origin(p_tile_map->map_to_world(E_cell->get()) - quadrant_pos);
					rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, xform);
					rs->canvas_item_add_circle(p_quadrant->debug_canvas_item, Vector2(), MIN(tile_set->get_tile_size().x, tile_set->get_tile_size().y) / 4.0, color);
				}
			}
		}
	}
}
