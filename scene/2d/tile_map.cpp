/**************************************************************************/
/*  tile_map.cpp                                                          */
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

#include "tile_map.h"
#include "tile_map.compat.inc"

#include "core/core_string_names.h"
#include "core/io/marshalls.h"
#include "scene/gui/control.h"

#define TILEMAP_CALL_FOR_LAYER(layer, function, ...) \
	if (layer < 0) {                                 \
		layer = layers.size() + layer;               \
	};                                               \
	ERR_FAIL_INDEX(layer, (int)layers.size());       \
	layers[layer]->function(__VA_ARGS__);

#define TILEMAP_CALL_FOR_LAYER_V(layer, err_value, function, ...) \
	if (layer < 0) {                                              \
		layer = layers.size() + layer;                            \
	};                                                            \
	ERR_FAIL_INDEX_V(layer, (int)layers.size(), err_value);       \
	return layers[layer]->function(__VA_ARGS__);

void TileMap::_tile_set_changed() {
	update_configuration_warnings();
}

void TileMap::_emit_changed() {
	emit_signal(CoreStringNames::get_singleton()->changed);
}

void TileMap::_set_tile_map_data_using_compatibility_format(int p_layer, TileMapDataFormat p_format, const Vector<int> &p_data) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	ERR_FAIL_COND(p_format >= TileMapDataFormat::TILE_MAP_DATA_FORMAT_MAX);
#ifndef DISABLE_DEPRECATED
	ERR_FAIL_COND_MSG(p_format != (TileMapDataFormat)(TILE_MAP_DATA_FORMAT_MAX - 1), "Old TileMap data format detected despite DISABLE_DEPRECATED being set compilation time.");
#endif // DISABLE_DEPRECATED

						for (int i = 0; i < navigation_polygon->get_polygon_count(); i++) {
							// An array of vertices for this polygon.
							Vector<int> polygon = navigation_polygon->get_polygon(i);
							Vector<Vector2> debug_polygon_vertices;
							debug_polygon_vertices.resize(polygon.size());
							for (int j = 0; j < polygon.size(); j++) {
								ERR_FAIL_INDEX(polygon[j], navigation_polygon_vertices.size());
								debug_polygon_vertices.write[j] = navigation_polygon_vertices[polygon[j]];
							}

							// Generate the polygon color, slightly randomly modified from the settings one.
							Color random_variation_color = debug_face_color;
							if (enabled_geometry_face_random_color) {
								random_variation_color.set_hsv(
										debug_face_color.get_h() + rand.random(-1.0, 1.0) * 0.1,
										debug_face_color.get_s(),
										debug_face_color.get_v() + rand.random(-1.0, 1.0) * 0.2);
							}
							random_variation_color.a = debug_face_color.a;

							Vector<Color> debug_face_colors;
							debug_face_colors.push_back(random_variation_color);
							rs->canvas_item_add_polygon(p_canvas_item, debug_polygon_vertices, debug_face_colors);

							if (enabled_edge_lines) {
								Vector<Color> debug_edge_colors;
								debug_edge_colors.push_back(debug_edge_color);
								debug_polygon_vertices.push_back(debug_polygon_vertices[0]); // Add first again for closing polyline.
								rs->canvas_item_add_polyline(p_canvas_item, debug_polygon_vertices, debug_edge_colors);
							}
						}
					}
				}
			}
		}
	}
}
#endif // DEBUG_ENABLED

/////////////////////////////// Scenes //////////////////////////////////////

void TileMapLayer::_scenes_update() {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();

	// Check if we should cleanup everything.
	bool forced_cleanup = in_destructor || !enabled || !tile_map_node->is_inside_tree() || !tile_set.is_valid();

	if (forced_cleanup) {
		// Clean everything.
		for (KeyValue<Vector2i, CellData> &kv : tile_map) {
			_scenes_clear_cell(kv.value);
		}
	} else {
		if (_scenes_was_cleaned_up || dirty.flags[DIRTY_FLAGS_TILE_MAP_TILE_SET]) {
			// Update all cells.
			for (KeyValue<Vector2i, CellData> &kv : tile_map) {
				_scenes_update_cell(kv.value);
			}
		} else {
			// Update dirty cells.
			for (SelfList<CellData> *cell_data_list_element = dirty.cell_list.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
				CellData &cell_data = *cell_data_list_element->self();
				_scenes_update_cell(cell_data);
			}
		}
	}

	// -----------
	// Mark the scenes state as up to date.
	_scenes_was_cleaned_up = forced_cleanup;
}

void TileMapLayer::_scenes_clear_cell(CellData &r_cell_data) {
	// Cleanup existing scene.
	Node *node = tile_map_node->get_node_or_null(r_cell_data.scene);
	if (node) {
		node->queue_free();
	}
	r_cell_data.scene = "";
}

void TileMapLayer::_scenes_update_cell(CellData &r_cell_data) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();

	// Clear the scene in any case.
	_scenes_clear_cell(r_cell_data);

	// Create the scene.
	const TileMapCell &c = r_cell_data.cell;

	TileSetSource *source;
	if (tile_set->has_source(c.source_id)) {
		source = *tile_set->get_source(c.source_id);

		if (source->has_tile(c.get_atlas_coords()) && source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
			TileSetScenesCollectionSource *scenes_collection_source = Object::cast_to<TileSetScenesCollectionSource>(source);
			if (scenes_collection_source) {
				Ref<PackedScene> packed_scene = scenes_collection_source->get_scene_tile_scene(c.alternative_tile);
				if (packed_scene.is_valid()) {
					Node *scene = packed_scene->instantiate();
					Control *scene_as_control = Object::cast_to<Control>(scene);
					Node2D *scene_as_node2d = Object::cast_to<Node2D>(scene);
					if (scene_as_control) {
						scene_as_control->set_position(tile_map_node->map_to_local(r_cell_data.coords) + scene_as_control->get_position());
					} else if (scene_as_node2d) {
						Transform2D xform;
						xform.set_origin(tile_map_node->map_to_local(r_cell_data.coords));
						scene_as_node2d->set_transform(xform * scene_as_node2d->get_transform());
					}
					tile_map_node->add_child(scene);
					r_cell_data.scene = scene->get_name();
				}
			}
		}
	}
}

#ifdef DEBUG_ENABLED
void TileMapLayer::_scenes_draw_cell_debug(const RID &p_canvas_item, const Vector2i &p_quadrant_pos, const CellData &r_cell_data) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	// Draw a placeholder for scenes needing one.
	RenderingServer *rs = RenderingServer::get_singleton();

	const TileMapCell &c = r_cell_data.cell;

	TileSetSource *source;
	if (tile_set->has_source(c.source_id)) {
		source = *tile_set->get_source(c.source_id);

		if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
			return;
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
				Transform2D cell_to_quadrant;
				cell_to_quadrant.set_origin(tile_map_node->map_to_local(r_cell_data.coords) - p_quadrant_pos);
				rs->canvas_item_add_set_transform(p_canvas_item, cell_to_quadrant);
				rs->canvas_item_add_circle(p_canvas_item, Vector2(), MIN(tile_set->get_tile_size().x, tile_set->get_tile_size().y) / 4.0, color);
			}
		}
	}
}
#endif // DEBUG_ENABLED

/////////////////////////////////////////////////////////////////////

void TileMapLayer::_build_runtime_update_tile_data() {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();

	// Check if we should cleanup everything.
	bool forced_cleanup = in_destructor || !enabled || !tile_map_node->is_inside_tree() || !tile_set.is_valid() || !tile_map_node->is_visible_in_tree();
	if (!forced_cleanup) {
		if (tile_map_node->GDVIRTUAL_IS_OVERRIDDEN(_use_tile_data_runtime_update) && tile_map_node->GDVIRTUAL_IS_OVERRIDDEN(_tile_data_runtime_update)) {
			if (_runtime_update_tile_data_was_cleaned_up || dirty.flags[DIRTY_FLAGS_TILE_MAP_TILE_SET]) {
				for (KeyValue<Vector2i, CellData> &E : tile_map) {
					_build_runtime_update_tile_data_for_cell(E.value);
				}
			} else if (dirty.flags[DIRTY_FLAGS_TILE_MAP_RUNTIME_UPDATE]) {
				for (KeyValue<Vector2i, CellData> &E : tile_map) {
					_build_runtime_update_tile_data_for_cell(E.value, true);
				}
			} else {
				for (SelfList<CellData> *cell_data_list_element = dirty.cell_list.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
					CellData &cell_data = *cell_data_list_element->self();
					_build_runtime_update_tile_data_for_cell(cell_data);
				}
			}
		}
	}

	// -----------
	// Mark the navigation state as up to date.
	_runtime_update_tile_data_was_cleaned_up = forced_cleanup;
}

void TileMapLayer::_build_runtime_update_tile_data_for_cell(CellData &r_cell_data, bool p_auto_add_to_dirty_list) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();

	TileMapCell &c = r_cell_data.cell;
	TileSetSource *source;
	if (tile_set->has_source(c.source_id)) {
		source = *tile_set->get_source(c.source_id);

		if (source->has_tile(c.get_atlas_coords()) && source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				bool ret = false;
				if (tile_map_node->GDVIRTUAL_CALL(_use_tile_data_runtime_update, layer_index_in_tile_map_node, r_cell_data.coords, ret) && ret) {
					TileData *tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);

					// Create the runtime TileData.
					TileData *tile_data_runtime_use = tile_data->duplicate();
					tile_data_runtime_use->set_allow_transform(true);
					r_cell_data.runtime_tile_data_cache = tile_data_runtime_use;

					tile_map_node->GDVIRTUAL_CALL(_tile_data_runtime_update, layer_index_in_tile_map_node, r_cell_data.coords, tile_data_runtime_use);

					if (p_auto_add_to_dirty_list) {
						dirty.cell_list.add(&r_cell_data.dirty_list_element);
					}
				}
			}
		}
	}
}

void TileMapLayer::_clear_runtime_update_tile_data() {
	for (SelfList<CellData> *cell_data_list_element = dirty.cell_list.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
		CellData &cell_data = *cell_data_list_element->self();

		// Clear the runtime tile data.
		if (cell_data.runtime_tile_data_cache) {
			memdelete(cell_data.runtime_tile_data_cache);
			cell_data.runtime_tile_data_cache = nullptr;
		}
	}
}

TileSet::TerrainsPattern TileMapLayer::_get_best_terrain_pattern_for_constraints(int p_terrain_set, const Vector2i &p_position, const RBSet<TerrainConstraint> &p_constraints, TileSet::TerrainsPattern p_current_pattern) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	if (!tile_set.is_valid()) {
		return TileSet::TerrainsPattern();
	}
	// Returns all tiles compatible with the given constraints.
	RBMap<TileSet::TerrainsPattern, int> terrain_pattern_score;
	RBSet<TileSet::TerrainsPattern> pattern_set = tile_set->get_terrains_pattern_set(p_terrain_set);
	ERR_FAIL_COND_V(pattern_set.is_empty(), TileSet::TerrainsPattern());
	for (TileSet::TerrainsPattern &terrain_pattern : pattern_set) {
		int score = 0;

		// Check the center bit constraint.
		TerrainConstraint terrain_constraint = TerrainConstraint(tile_map_node, p_position, terrain_pattern.get_terrain());
		const RBSet<TerrainConstraint>::Element *in_set_constraint_element = p_constraints.find(terrain_constraint);
		if (in_set_constraint_element) {
			if (in_set_constraint_element->get().get_terrain() != terrain_constraint.get_terrain()) {
				score += in_set_constraint_element->get().get_priority();
			}
		} else if (p_current_pattern.get_terrain() != terrain_pattern.get_terrain()) {
			continue; // Ignore a pattern that cannot keep bits without constraints unmodified.
		}

		// Check the surrounding bits
		bool invalid_pattern = false;
		for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(i);
			if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, bit)) {
				// Check if the bit is compatible with the constraints.
				TerrainConstraint terrain_bit_constraint = TerrainConstraint(tile_map_node, p_position, bit, terrain_pattern.get_terrain_peering_bit(bit));
				in_set_constraint_element = p_constraints.find(terrain_bit_constraint);
				if (in_set_constraint_element) {
					if (in_set_constraint_element->get().get_terrain() != terrain_bit_constraint.get_terrain()) {
						score += in_set_constraint_element->get().get_priority();
					}
				} else if (p_current_pattern.get_terrain_peering_bit(bit) != terrain_pattern.get_terrain_peering_bit(bit)) {
					invalid_pattern = true; // Ignore a pattern that cannot keep bits without constraints unmodified.
					break;
				}
			}
		}
		if (invalid_pattern) {
			continue;
		}

		terrain_pattern_score[terrain_pattern] = score;
	}

	// Compute the minimum score.
	TileSet::TerrainsPattern min_score_pattern = p_current_pattern;
	int min_score = INT32_MAX;
	for (KeyValue<TileSet::TerrainsPattern, int> E : terrain_pattern_score) {
		if (E.value < min_score) {
			min_score_pattern = E.key;
			min_score = E.value;
		}
	}

	return min_score_pattern;
}

RBSet<TerrainConstraint> TileMapLayer::_get_terrain_constraints_from_added_pattern(const Vector2i &p_position, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern) const {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	if (!tile_set.is_valid()) {
		return RBSet<TerrainConstraint>();
	}

	// Compute the constraints needed from the surrounding tiles.
	RBSet<TerrainConstraint> output;
	output.insert(TerrainConstraint(tile_map_node, p_position, p_terrains_pattern.get_terrain()));

	for (uint32_t i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
		TileSet::CellNeighbor side = TileSet::CellNeighbor(i);
		if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, side)) {
			TerrainConstraint c = TerrainConstraint(tile_map_node, p_position, side, p_terrains_pattern.get_terrain_peering_bit(side));
			output.insert(c);
		}
	}

	return output;
}

RBSet<TerrainConstraint> TileMapLayer::_get_terrain_constraints_from_painted_cells_list(const RBSet<Vector2i> &p_painted, int p_terrain_set, bool p_ignore_empty_terrains) const {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	if (!tile_set.is_valid()) {
		return RBSet<TerrainConstraint>();
	}

	ERR_FAIL_INDEX_V(p_terrain_set, tile_set->get_terrain_sets_count(), RBSet<TerrainConstraint>());

	// Build a set of dummy constraints to get the constrained points.
	RBSet<TerrainConstraint> dummy_constraints;
	for (const Vector2i &E : p_painted) {
		for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) { // Iterates over neighbor bits.
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(i);
			if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, bit)) {
				dummy_constraints.insert(TerrainConstraint(tile_map_node, E, bit, -1));
			}
		}
	}

	// For each constrained point, we get all overlapping tiles, and select the most adequate terrain for it.
	RBSet<TerrainConstraint> constraints;
	for (const TerrainConstraint &E_constraint : dummy_constraints) {
		HashMap<int, int> terrain_count;

		// Count the number of occurrences per terrain.
		HashMap<Vector2i, TileSet::CellNeighbor> overlapping_terrain_bits = E_constraint.get_overlapping_coords_and_peering_bits();
		for (const KeyValue<Vector2i, TileSet::CellNeighbor> &E_overlapping : overlapping_terrain_bits) {
			TileData *neighbor_tile_data = nullptr;
			TileMapCell neighbor_cell = get_cell(E_overlapping.key);
			if (neighbor_cell.source_id != TileSet::INVALID_SOURCE) {
				Ref<TileSetSource> source = tile_set->get_source(neighbor_cell.source_id);
				Ref<TileSetAtlasSource> atlas_source = source;
				if (atlas_source.is_valid()) {
					TileData *tile_data = atlas_source->get_tile_data(neighbor_cell.get_atlas_coords(), neighbor_cell.alternative_tile);
					if (tile_data && tile_data->get_terrain_set() == p_terrain_set) {
						neighbor_tile_data = tile_data;
					}
				}
			}

			int terrain = neighbor_tile_data ? neighbor_tile_data->get_terrain_peering_bit(TileSet::CellNeighbor(E_overlapping.value)) : -1;
			if (!p_ignore_empty_terrains || terrain >= 0) {
				if (!terrain_count.has(terrain)) {
					terrain_count[terrain] = 0;
				}
				terrain_count[terrain] += 1;
			}
		}

		// Get the terrain with the max number of occurrences.
		int max = 0;
		int max_terrain = -1;
		for (const KeyValue<int, int> &E_terrain_count : terrain_count) {
			if (E_terrain_count.value > max) {
				max = E_terrain_count.value;
				max_terrain = E_terrain_count.key;
			}
		}

		// Set the adequate terrain.
		if (max > 0) {
			TerrainConstraint c = E_constraint;
			c.set_terrain(max_terrain);
			constraints.insert(c);
		}
	}

	// Add the centers as constraints.
	for (Vector2i E_coords : p_painted) {
		TileData *tile_data = nullptr;
		TileMapCell cell = get_cell(E_coords);
		if (cell.source_id != TileSet::INVALID_SOURCE) {
			Ref<TileSetSource> source = tile_set->get_source(cell.source_id);
			Ref<TileSetAtlasSource> atlas_source = source;
			if (atlas_source.is_valid()) {
				tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
			}
		}

		int terrain = (tile_data && tile_data->get_terrain_set() == p_terrain_set) ? tile_data->get_terrain() : -1;
		if (!p_ignore_empty_terrains || terrain >= 0) {
			constraints.insert(TerrainConstraint(tile_map_node, E_coords, terrain));
		}
	}

	return constraints;
}

void TileMapLayer::set_tile_map(TileMap *p_tile_map) {
	tile_map_node = p_tile_map;
}

void TileMapLayer::set_layer_index_in_tile_map_node(int p_index) {
	if (p_index == layer_index_in_tile_map_node) {
		return;
	}
	layer_index_in_tile_map_node = p_index;
	dirty.flags[DIRTY_FLAGS_LAYER_INDEX_IN_TILE_MAP_NODE] = true;
	tile_map_node->queue_internal_update();
}

Rect2 TileMapLayer::get_rect(bool &r_changed) const {
	// Compute the displayed area of the tilemap.
	r_changed = false;
#ifdef DEBUG_ENABLED

	if (rect_cache_dirty) {
		Rect2 r_total;
		bool first = true;
		for (const KeyValue<Vector2i, CellData> &E : tile_map) {
			Rect2 r;
			r.position = tile_map_node->map_to_local(E.key);
			r.size = Size2();
			if (first) {
				r_total = r;
				first = false;
			} else {
				r_total = r_total.merge(r);
			}
		}

		r_changed = rect_cache != r_total;

		rect_cache = r_total;
		rect_cache_dirty = false;
	}
#endif
	return rect_cache;
}

HashMap<Vector2i, TileSet::TerrainsPattern> TileMapLayer::terrain_fill_constraints(const Vector<Vector2i> &p_to_replace, int p_terrain_set, const RBSet<TerrainConstraint> &p_constraints) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	if (!tile_set.is_valid()) {
		return HashMap<Vector2i, TileSet::TerrainsPattern>();
	}

	// Copy the constraints set.
	RBSet<TerrainConstraint> constraints = p_constraints;

	// Output map.
	HashMap<Vector2i, TileSet::TerrainsPattern> output;

	// Add all positions to a set.
	for (int i = 0; i < p_to_replace.size(); i++) {
		const Vector2i &coords = p_to_replace[i];

		// Select the best pattern for the given constraints.
		TileSet::TerrainsPattern current_pattern = TileSet::TerrainsPattern(*tile_set, p_terrain_set);
		TileMapCell cell = get_cell(coords);
		if (cell.source_id != TileSet::INVALID_SOURCE) {
			TileSetSource *source = *tile_set->get_source(cell.source_id);
			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				// Get tile data.
				TileData *tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
				if (tile_data && tile_data->get_terrain_set() == p_terrain_set) {
					current_pattern = tile_data->get_terrains_pattern();
				}
			}
		}
		TileSet::TerrainsPattern pattern = _get_best_terrain_pattern_for_constraints(p_terrain_set, coords, constraints, current_pattern);

		// Update the constraint set with the new ones.
		RBSet<TerrainConstraint> new_constraints = _get_terrain_constraints_from_added_pattern(coords, p_terrain_set, pattern);
		for (const TerrainConstraint &E_constraint : new_constraints) {
			if (constraints.has(E_constraint)) {
				constraints.erase(E_constraint);
			}
			TerrainConstraint c = E_constraint;
			c.set_priority(5);
			constraints.insert(c);
		}

		output[coords] = pattern;
	}
	return output;
}

HashMap<Vector2i, TileSet::TerrainsPattern> TileMapLayer::terrain_fill_connect(const Vector<Vector2i> &p_coords_array, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	HashMap<Vector2i, TileSet::TerrainsPattern> output;
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	ERR_FAIL_COND_V(!tile_set.is_valid(), output);
	ERR_FAIL_INDEX_V(p_terrain_set, tile_set->get_terrain_sets_count(), output);

	// Build list and set of tiles that can be modified (painted and their surroundings).
	Vector<Vector2i> can_modify_list;
	RBSet<Vector2i> can_modify_set;
	RBSet<Vector2i> painted_set;
	for (int i = p_coords_array.size() - 1; i >= 0; i--) {
		const Vector2i &coords = p_coords_array[i];
		can_modify_list.push_back(coords);
		can_modify_set.insert(coords);
		painted_set.insert(coords);
	}
	for (Vector2i coords : p_coords_array) {
		// Find the adequate neighbor.
		for (int j = 0; j < TileSet::CELL_NEIGHBOR_MAX; j++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(j);
			if (tile_map_node->is_existing_neighbor(bit)) {
				Vector2i neighbor = tile_map_node->get_neighbor_cell(coords, bit);
				if (!can_modify_set.has(neighbor)) {
					can_modify_list.push_back(neighbor);
					can_modify_set.insert(neighbor);
				}
			}
		}
	}

	// Build a set, out of the possibly modified tiles, of the one with a center bit that is set (or will be) to the painted terrain.
	RBSet<Vector2i> cells_with_terrain_center_bit;
	for (Vector2i coords : can_modify_set) {
		bool connect = false;
		if (painted_set.has(coords)) {
			connect = true;
		} else {
			// Get the center bit of the cell.
			TileData *tile_data = nullptr;
			TileMapCell cell = get_cell(coords);
			if (cell.source_id != TileSet::INVALID_SOURCE) {
				Ref<TileSetSource> source = tile_set->get_source(cell.source_id);
				Ref<TileSetAtlasSource> atlas_source = source;
				if (atlas_source.is_valid()) {
					tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
				}
			}

			if (tile_data && tile_data->get_terrain_set() == p_terrain_set && tile_data->get_terrain() == p_terrain) {
				connect = true;
			}
		}
		if (connect) {
			cells_with_terrain_center_bit.insert(coords);
		}
	}

	RBSet<TerrainConstraint> constraints;

	// Add new constraints from the path drawn.
	for (Vector2i coords : p_coords_array) {
		// Constraints on the center bit.
		TerrainConstraint c = TerrainConstraint(tile_map_node, coords, p_terrain);
		c.set_priority(10);
		constraints.insert(c);

		// Constraints on the connecting bits.
		for (int j = 0; j < TileSet::CELL_NEIGHBOR_MAX; j++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(j);
			if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, bit)) {
				c = TerrainConstraint(tile_map_node, coords, bit, p_terrain);
				c.set_priority(10);
				if ((int(bit) % 2) == 0) {
					// Side peering bits: add the constraint if the center is of the same terrain.
					Vector2i neighbor = tile_map_node->get_neighbor_cell(coords, bit);
					if (cells_with_terrain_center_bit.has(neighbor)) {
						constraints.insert(c);
					}
				} else {
					// Corner peering bits: add the constraint if all tiles on the constraint has the same center bit.
					HashMap<Vector2i, TileSet::CellNeighbor> overlapping_terrain_bits = c.get_overlapping_coords_and_peering_bits();
					bool valid = true;
					for (KeyValue<Vector2i, TileSet::CellNeighbor> kv : overlapping_terrain_bits) {
						if (!cells_with_terrain_center_bit.has(kv.key)) {
							valid = false;
							break;
						}
					}
					if (valid) {
						constraints.insert(c);
					}
				}
			}
		}
	}

	// Fills in the constraint list from existing tiles.
	for (TerrainConstraint c : _get_terrain_constraints_from_painted_cells_list(painted_set, p_terrain_set, p_ignore_empty_terrains)) {
		constraints.insert(c);
	}

	// Fill the terrains.
	output = terrain_fill_constraints(can_modify_list, p_terrain_set, constraints);
	return output;
}

HashMap<Vector2i, TileSet::TerrainsPattern> TileMapLayer::terrain_fill_path(const Vector<Vector2i> &p_coords_array, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	HashMap<Vector2i, TileSet::TerrainsPattern> output;
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	ERR_FAIL_COND_V(!tile_set.is_valid(), output);
	ERR_FAIL_INDEX_V(p_terrain_set, tile_set->get_terrain_sets_count(), output);

	// Make sure the path is correct and build the peering bit list while doing it.
	Vector<TileSet::CellNeighbor> neighbor_list;
	for (int i = 0; i < p_coords_array.size() - 1; i++) {
		// Find the adequate neighbor.
		TileSet::CellNeighbor found_bit = TileSet::CELL_NEIGHBOR_MAX;
		for (int j = 0; j < TileSet::CELL_NEIGHBOR_MAX; j++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(j);
			if (tile_map_node->is_existing_neighbor(bit)) {
				if (tile_map_node->get_neighbor_cell(p_coords_array[i], bit) == p_coords_array[i + 1]) {
					found_bit = bit;
					break;
				}
			}
		}
		ERR_FAIL_COND_V_MSG(found_bit == TileSet::CELL_NEIGHBOR_MAX, output, vformat("Invalid terrain path, %s is not a neighboring tile of %s", p_coords_array[i + 1], p_coords_array[i]));
		neighbor_list.push_back(found_bit);
	}

	// Build list and set of tiles that can be modified (painted and their surroundings).
	Vector<Vector2i> can_modify_list;
	RBSet<Vector2i> can_modify_set;
	RBSet<Vector2i> painted_set;
	for (int i = p_coords_array.size() - 1; i >= 0; i--) {
		const Vector2i &coords = p_coords_array[i];
		can_modify_list.push_back(coords);
		can_modify_set.insert(coords);
		painted_set.insert(coords);
	}
	for (Vector2i coords : p_coords_array) {
		// Find the adequate neighbor.
		for (int j = 0; j < TileSet::CELL_NEIGHBOR_MAX; j++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(j);
			if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, bit)) {
				Vector2i neighbor = tile_map_node->get_neighbor_cell(coords, bit);
				if (!can_modify_set.has(neighbor)) {
					can_modify_list.push_back(neighbor);
					can_modify_set.insert(neighbor);
				}
			}
		}
	}

	RBSet<TerrainConstraint> constraints;

	// Add new constraints from the path drawn.
	for (Vector2i coords : p_coords_array) {
		// Constraints on the center bit.
		TerrainConstraint c = TerrainConstraint(tile_map_node, coords, p_terrain);
		c.set_priority(10);
		constraints.insert(c);
	}
	for (int i = 0; i < p_coords_array.size() - 1; i++) {
		// Constraints on the peering bits.
		TerrainConstraint c = TerrainConstraint(tile_map_node, p_coords_array[i], neighbor_list[i], p_terrain);
		c.set_priority(10);
		constraints.insert(c);
	}

	// Fills in the constraint list from existing tiles.
	for (TerrainConstraint c : _get_terrain_constraints_from_painted_cells_list(painted_set, p_terrain_set, p_ignore_empty_terrains)) {
		constraints.insert(c);
	}

	// Fill the terrains.
	output = terrain_fill_constraints(can_modify_list, p_terrain_set, constraints);
	return output;
}

HashMap<Vector2i, TileSet::TerrainsPattern> TileMapLayer::terrain_fill_pattern(const Vector<Vector2i> &p_coords_array, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern, bool p_ignore_empty_terrains) {
	HashMap<Vector2i, TileSet::TerrainsPattern> output;
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	ERR_FAIL_COND_V(!tile_set.is_valid(), output);
	ERR_FAIL_INDEX_V(p_terrain_set, tile_set->get_terrain_sets_count(), output);

	// Build list and set of tiles that can be modified (painted and their surroundings).
	Vector<Vector2i> can_modify_list;
	RBSet<Vector2i> can_modify_set;
	RBSet<Vector2i> painted_set;
	for (int i = p_coords_array.size() - 1; i >= 0; i--) {
		const Vector2i &coords = p_coords_array[i];
		can_modify_list.push_back(coords);
		can_modify_set.insert(coords);
		painted_set.insert(coords);
	}
	for (Vector2i coords : p_coords_array) {
		// Find the adequate neighbor.
		for (int j = 0; j < TileSet::CELL_NEIGHBOR_MAX; j++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(j);
			if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, bit)) {
				Vector2i neighbor = tile_map_node->get_neighbor_cell(coords, bit);
				if (!can_modify_set.has(neighbor)) {
					can_modify_list.push_back(neighbor);
					can_modify_set.insert(neighbor);
				}
			}
		}
	}

	// Add constraint by the new ones.
	RBSet<TerrainConstraint> constraints;

	// Add new constraints from the path drawn.
	for (Vector2i coords : p_coords_array) {
		// Constraints on the center bit.
		RBSet<TerrainConstraint> added_constraints = _get_terrain_constraints_from_added_pattern(coords, p_terrain_set, p_terrains_pattern);
		for (TerrainConstraint c : added_constraints) {
			c.set_priority(10);
			constraints.insert(c);
		}
	}

	// Fills in the constraint list from modified tiles border.
	for (TerrainConstraint c : _get_terrain_constraints_from_painted_cells_list(painted_set, p_terrain_set, p_ignore_empty_terrains)) {
		constraints.insert(c);
	}

	// Fill the terrains.
	output = terrain_fill_constraints(can_modify_list, p_terrain_set, constraints);
	return output;
}

TileMapCell TileMapLayer::get_cell(const Vector2i &p_coords, bool p_use_proxies) const {
	if (!tile_map.has(p_coords)) {
		return TileMapCell();
	} else {
		TileMapCell c = tile_map.find(p_coords)->value.cell;
		const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
		if (p_use_proxies && tile_set.is_valid()) {
			Array proxyed = tile_set->map_tile_proxy(c.source_id, c.get_atlas_coords(), c.alternative_tile);
			c.source_id = proxyed[0];
			c.set_atlas_coords(proxyed[1]);
			c.alternative_tile = proxyed[2];
		}
		return c;
	}
}

void TileMapLayer::set_tile_data(TileMapLayer::DataFormat p_format, const Vector<int> &p_data) {
	ERR_FAIL_COND(p_format > TileMapLayer::FORMAT_3);

	ERR_FAIL_COND(p_format > TileMapLayer::FORMAT_3);

	ERR_FAIL_COND(p_format > TileMapLayer::FORMAT_3);

	ERR_FAIL_COND(p_format > TileMapLayer::FORMAT_3);

	// Set data for a given tile from raw data.
	int c = p_data.size();
	const int *r = p_data.ptr();

	int offset = (p_format >= TileMapDataFormat::TILE_MAP_DATA_FORMAT_2) ? 3 : 2;
	ERR_FAIL_COND_MSG(c % offset != 0, vformat("Corrupted tile data. Got size: %d. Expected modulo: %d", c, offset));

	layers[p_layer]->clear();

	for (int i = 0; i < c; i += offset) {
		const uint8_t *ptr = (const uint8_t *)&r[i];
		uint8_t local[12];
		for (int j = 0; j < ((p_format >= TileMapDataFormat::TILE_MAP_DATA_FORMAT_2) ? 12 : 8); j++) {
			local[j] = ptr[j];
		}

#ifdef BIG_ENDIAN_ENABLED
		SWAP(local[0], local[3]);
		SWAP(local[1], local[2]);
		SWAP(local[4], local[7]);
		SWAP(local[5], local[6]);
		//TODO: ask someone to check this...
		if (FORMAT >= FORMAT_2) {
			SWAP(local[8], local[11]);
			SWAP(local[9], local[10]);
		}
#endif
		// Extracts position in TileMap.
		int16_t x = decode_uint16(&local[0]);
		int16_t y = decode_uint16(&local[2]);

		if (p_format == TileMapDataFormat::TILE_MAP_DATA_FORMAT_3) {
			uint16_t source_id = decode_uint16(&local[4]);
			uint16_t atlas_coords_x = decode_uint16(&local[6]);
			uint16_t atlas_coords_y = decode_uint16(&local[8]);
			uint16_t alternative_tile = decode_uint16(&local[10]);
			layers[p_layer]->set_cell(Vector2i(x, y), source_id, Vector2i(atlas_coords_x, atlas_coords_y), alternative_tile);
		} else {
#ifndef DISABLE_DEPRECATED
			// Previous decated format.
			uint32_t v = decode_uint32(&local[4]);
			// Extract the transform flags that used to be in the tilemap.
			bool flip_h = v & (1UL << 29);
			bool flip_v = v & (1UL << 30);
			bool transpose = v & (1UL << 31);
			v &= (1UL << 29) - 1;

			// Extract autotile/atlas coords.
			int16_t coord_x = 0;
			int16_t coord_y = 0;
			if (p_format == TileMapDataFormat::TILE_MAP_DATA_FORMAT_2) {
				coord_x = decode_uint16(&local[8]);
				coord_y = decode_uint16(&local[10]);
			}

			if (tile_set.is_valid()) {
				Array a = tile_set->compatibility_tilemap_map(v, Vector2i(coord_x, coord_y), flip_h, flip_v, transpose);
				if (a.size() == 3) {
					layers[p_layer]->set_cell(Vector2i(x, y), a[0], a[1], a[2]);
				} else {
					ERR_PRINT(vformat("No valid tile in Tileset for: tile:%s coords:%s flip_h:%s flip_v:%s transpose:%s", v, Vector2i(coord_x, coord_y), flip_h, flip_v, transpose));
				}
			} else {
				int compatibility_alternative_tile = ((int)flip_h) + ((int)flip_v << 1) + ((int)transpose << 2);
				layers[p_layer]->set_cell(Vector2i(x, y), v, Vector2i(coord_x, coord_y), compatibility_alternative_tile);
			}
#endif // DISABLE_DEPRECATED
		}
	}
}

Vector<int> TileMap::_get_tile_map_data_using_compatibility_format(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), Vector<int>());

	// Export tile data to raw format.
	const HashMap<Vector2i, CellData> tile_map_layer_data = layers[p_layer]->get_tile_map_layer_data();
	Vector<int> tile_data;
	tile_data.resize(tile_map_layer_data.size() * 3);
	int *w = tile_data.ptrw();

	// Save in highest format.

	int idx = 0;
	for (const KeyValue<Vector2i, CellData> &E : tile_map_layer_data) {
		uint8_t *ptr = (uint8_t *)&w[idx];
		encode_uint16((int16_t)(E.key.x), &ptr[0]);
		encode_uint16((int16_t)(E.key.y), &ptr[2]);
		encode_uint16(E.value.cell.source_id, &ptr[4]);
		encode_uint16(E.value.cell.coord_x, &ptr[6]);
		encode_uint16(E.value.cell.coord_y, &ptr[8]);
		encode_uint16(E.value.cell.alternative_tile, &ptr[10]);
void TileMapLayer::notify_tile_map_change(DirtyFlags p_what) {
	dirty.flags[p_what] = true;
	tile_map_node->queue_internal_update();
	_physics_notify_tilemap_change(p_what);
}

void TileMapLayer::internal_update() {
	// Find TileData that need a runtime modification.
	// This may add cells to the dirty list is a runtime modification has been notified.
	_build_runtime_update_tile_data();

	// Update all subsystems.
	_rendering_update();
	_physics_update();
	_navigation_update();
	_scenes_update();
#ifdef DEBUG_ENABLED
	_debug_update();
#endif // DEBUG_ENABLED

	_clear_runtime_update_tile_data();

	// Clear the "what is dirty" flags.
	for (int i = 0; i < DIRTY_FLAGS_MAX; i++) {
		dirty.flags[i] = false;
	}

	// List the cells to delete definitely.
	Vector<Vector2i> to_delete;
	for (SelfList<CellData> *cell_data_list_element = dirty.cell_list.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
		CellData &cell_data = *cell_data_list_element->self();
		// Select the the cell from tile_map if it is invalid.
		if (cell_data.cell.source_id == TileSet::INVALID_SOURCE) {
			to_delete.push_back(cell_data.coords);
		}
	}

	// Remove cells that are empty after the cleanup.
	for (const Vector2i &coords : to_delete) {
		tile_map.erase(coords);
	}

	// Clear the dirty cells list.
	dirty.cell_list.clear();
}

void TileMapLayer::set_cell(const Vector2i &p_coords, int p_source_id, const Vector2i p_atlas_coords, int p_alternative_tile) {
	// Set the current cell tile (using integer position).
	Vector2i pk(p_coords);
	HashMap<Vector2i, CellData>::Iterator E = tile_map.find(pk);

	int source_id = p_source_id;
	Vector2i atlas_coords = p_atlas_coords;
	int alternative_tile = p_alternative_tile;

	if ((source_id == TileSet::INVALID_SOURCE || atlas_coords == TileSetSource::INVALID_ATLAS_COORDS || alternative_tile == TileSetSource::INVALID_TILE_ALTERNATIVE) &&
			(source_id != TileSet::INVALID_SOURCE || atlas_coords != TileSetSource::INVALID_ATLAS_COORDS || alternative_tile != TileSetSource::INVALID_TILE_ALTERNATIVE)) {
		source_id = TileSet::INVALID_SOURCE;
		atlas_coords = TileSetSource::INVALID_ATLAS_COORDS;
		alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE;
	}

	if (!E) {
		if (source_id == TileSet::INVALID_SOURCE) {
			return; // Nothing to do, the tile is already empty.
		}

		// Insert a new cell in the tile map.
		CellData new_cell_data;
		new_cell_data.coords = pk;
		E = tile_map.insert(pk, new_cell_data);
	} else {
		if (E->value.cell.source_id == source_id && E->value.cell.get_atlas_coords() == atlas_coords && E->value.cell.alternative_tile == alternative_tile) {
			return; // Nothing changed.
		}
	}

	TileMapCell &c = E->value.cell;
	c.source_id = source_id;
	c.set_atlas_coords(atlas_coords);
	c.alternative_tile = alternative_tile;

	// Make the given cell dirty.
	if (!E->value.dirty_list_element.in_list()) {
		dirty.cell_list.add(&(E->value.dirty_list_element));
	}
	tile_map_node->queue_internal_update();

	used_rect_cache_dirty = true;
}

void TileMapLayer::erase_cell(const Vector2i &p_coords) {
	set_cell(p_coords, TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
}

int TileMapLayer::get_cell_source_id(const Vector2i &p_coords, bool p_use_proxies) const {
	// Get a cell source id from position.
	HashMap<Vector2i, CellData>::ConstIterator E = tile_map.find(p_coords);

	if (!E) {
		return TileSet::INVALID_SOURCE;
	}

	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	if (p_use_proxies && tile_set.is_valid()) {
		Array proxyed = tile_set->map_tile_proxy(E->value.cell.source_id, E->value.cell.get_atlas_coords(), E->value.cell.alternative_tile);
		return proxyed[0];
	}

	return E->value.cell.source_id;
}

Vector2i TileMapLayer::get_cell_atlas_coords(const Vector2i &p_coords, bool p_use_proxies) const {
	// Get a cell source id from position.
	HashMap<Vector2i, CellData>::ConstIterator E = tile_map.find(p_coords);

	if (!E) {
		return TileSetSource::INVALID_ATLAS_COORDS;
	}

	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	if (p_use_proxies && tile_set.is_valid()) {
		Array proxyed = tile_set->map_tile_proxy(E->value.cell.source_id, E->value.cell.get_atlas_coords(), E->value.cell.alternative_tile);
		return proxyed[1];
	}

	return E->value.cell.get_atlas_coords();
}

int TileMapLayer::get_cell_alternative_tile(const Vector2i &p_coords, bool p_use_proxies) const {
	// Get a cell source id from position.
	HashMap<Vector2i, CellData>::ConstIterator E = tile_map.find(p_coords);

	if (!E) {
		return TileSetSource::INVALID_TILE_ALTERNATIVE;
	}

	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	if (p_use_proxies && tile_set.is_valid()) {
		Array proxyed = tile_set->map_tile_proxy(E->value.cell.source_id, E->value.cell.get_atlas_coords(), E->value.cell.alternative_tile);
		return proxyed[2];
	}

	return E->value.cell.alternative_tile;
}

TileData *TileMapLayer::get_cell_tile_data(const Vector2i &p_coords, bool p_use_proxies) const {
	int source_id = get_cell_source_id(p_coords, p_use_proxies);
	if (source_id == TileSet::INVALID_SOURCE) {
		return nullptr;
	}

	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	Ref<TileSetAtlasSource> source = tile_set->get_source(source_id);
	if (source.is_valid()) {
		return source->get_tile_data(get_cell_atlas_coords(p_coords, p_use_proxies), get_cell_alternative_tile(p_coords, p_use_proxies));
	}

	return nullptr;
}

void TileMapLayer::clear() {
	// Remove all tiles.
	for (KeyValue<Vector2i, CellData> &kv : tile_map) {
		erase_cell(kv.key);
	}
	used_rect_cache_dirty = true;
}

Ref<TileMapPattern> TileMapLayer::get_pattern(TypedArray<Vector2i> p_coords_array) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	ERR_FAIL_COND_V(!tile_set.is_valid(), nullptr);

	Ref<TileMapPattern> output;
	output.instantiate();
	if (p_coords_array.is_empty()) {
		return output;
	}

	Vector2i min = Vector2i(p_coords_array[0]);
	for (int i = 1; i < p_coords_array.size(); i++) {
		min = min.min(p_coords_array[i]);
	}

	Vector<Vector2i> coords_in_pattern_array;
	coords_in_pattern_array.resize(p_coords_array.size());
	Vector2i ensure_positive_offset;
	for (int i = 0; i < p_coords_array.size(); i++) {
		Vector2i coords = p_coords_array[i];
		Vector2i coords_in_pattern = coords - min;
		if (tile_set->get_tile_shape() != TileSet::TILE_SHAPE_SQUARE) {
			if (tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_STACKED) {
				if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL && bool(min.y % 2) && bool(coords_in_pattern.y % 2)) {
					coords_in_pattern.x -= 1;
					if (coords_in_pattern.x < 0) {
						ensure_positive_offset.x = 1;
					}
				} else if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL && bool(min.x % 2) && bool(coords_in_pattern.x % 2)) {
					coords_in_pattern.y -= 1;
					if (coords_in_pattern.y < 0) {
						ensure_positive_offset.y = 1;
					}
				}
			} else if (tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_STACKED_OFFSET) {
				if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL && bool(min.y % 2) && bool(coords_in_pattern.y % 2)) {
					coords_in_pattern.x += 1;
				} else if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL && bool(min.x % 2) && bool(coords_in_pattern.x % 2)) {
					coords_in_pattern.y += 1;
				}
			}
		}
		coords_in_pattern_array.write[i] = coords_in_pattern;
	}

	for (int i = 0; i < coords_in_pattern_array.size(); i++) {
		Vector2i coords = p_coords_array[i];
		Vector2i coords_in_pattern = coords_in_pattern_array[i];
		output->set_cell(coords_in_pattern + ensure_positive_offset, get_cell_source_id(coords), get_cell_atlas_coords(coords), get_cell_alternative_tile(coords));
	}

	return output;
}

void TileMapLayer::set_pattern(const Vector2i &p_position, const Ref<TileMapPattern> p_pattern) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	ERR_FAIL_COND(tile_set.is_null());
	ERR_FAIL_COND(p_pattern.is_null());

	TypedArray<Vector2i> used_cells = p_pattern->get_used_cells();
	for (int i = 0; i < used_cells.size(); i++) {
		Vector2i coords = tile_map_node->map_pattern(p_position, used_cells[i], p_pattern);
		set_cell(coords, p_pattern->get_cell_source_id(used_cells[i]), p_pattern->get_cell_atlas_coords(used_cells[i]), p_pattern->get_cell_alternative_tile(used_cells[i]));
	}
}

void TileMapLayer::set_cells_terrain_connect(TypedArray<Vector2i> p_cells, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());
	ERR_FAIL_INDEX(p_terrain_set, tile_set->get_terrain_sets_count());

	Vector<Vector2i> cells_vector;
	HashSet<Vector2i> painted_set;
	for (int i = 0; i < p_cells.size(); i++) {
		cells_vector.push_back(p_cells[i]);
		painted_set.insert(p_cells[i]);
	}
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_output = terrain_fill_connect(cells_vector, p_terrain_set, p_terrain, p_ignore_empty_terrains);
	for (const KeyValue<Vector2i, TileSet::TerrainsPattern> &kv : terrain_fill_output) {
		if (painted_set.has(kv.key)) {
			// Paint a random tile with the correct terrain for the painted path.
			TileMapCell c = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
			set_cell(kv.key, c.source_id, c.get_atlas_coords(), c.alternative_tile);
		} else {
			// Avoids updating the painted path from the output if the new pattern is the same as before.
			TileSet::TerrainsPattern in_map_terrain_pattern = TileSet::TerrainsPattern(*tile_set, p_terrain_set);
			TileMapCell cell = get_cell(kv.key);
			if (cell.source_id != TileSet::INVALID_SOURCE) {
				TileSetSource *source = *tile_set->get_source(cell.source_id);
				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					// Get tile data.
					TileData *tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
					if (tile_data && tile_data->get_terrain_set() == p_terrain_set) {
						in_map_terrain_pattern = tile_data->get_terrains_pattern();
					}
				}
			}
			if (in_map_terrain_pattern != kv.value) {
				TileMapCell c = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
				set_cell(kv.key, c.source_id, c.get_atlas_coords(), c.alternative_tile);
			}
		}
	}
}

void TileMapLayer::set_cells_terrain_path(TypedArray<Vector2i> p_path, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());
	ERR_FAIL_INDEX(p_terrain_set, tile_set->get_terrain_sets_count());

	Vector<Vector2i> vector_path;
	HashSet<Vector2i> painted_set;
	for (int i = 0; i < p_path.size(); i++) {
		vector_path.push_back(p_path[i]);
		painted_set.insert(p_path[i]);
	}

	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_output = terrain_fill_path(vector_path, p_terrain_set, p_terrain, p_ignore_empty_terrains);
	for (const KeyValue<Vector2i, TileSet::TerrainsPattern> &kv : terrain_fill_output) {
		if (painted_set.has(kv.key)) {
			// Paint a random tile with the correct terrain for the painted path.
			TileMapCell c = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
			set_cell(kv.key, c.source_id, c.get_atlas_coords(), c.alternative_tile);
		} else {
			// Avoids updating the painted path from the output if the new pattern is the same as before.
			TileSet::TerrainsPattern in_map_terrain_pattern = TileSet::TerrainsPattern(*tile_set, p_terrain_set);
			TileMapCell cell = get_cell(kv.key);
			if (cell.source_id != TileSet::INVALID_SOURCE) {
				TileSetSource *source = *tile_set->get_source(cell.source_id);
				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					// Get tile data.
					TileData *tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
					if (tile_data && tile_data->get_terrain_set() == p_terrain_set) {
						in_map_terrain_pattern = tile_data->get_terrains_pattern();
					}
				}
			}
			if (in_map_terrain_pattern != kv.value) {
				TileMapCell c = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
				set_cell(kv.key, c.source_id, c.get_atlas_coords(), c.alternative_tile);
			}
		}
	}
}

TypedArray<Vector2i> TileMapLayer::get_used_cells() const {
	// Returns the cells used in the tilemap.
	TypedArray<Vector2i> a;
	for (const KeyValue<Vector2i, CellData> &E : tile_map) {
		const TileMapCell &c = E.value.cell;
		if (c.source_id == TileSet::INVALID_SOURCE) {
			continue;
		}
		a.push_back(E.key);
	}

	return a;
}

TypedArray<Vector2i> TileMapLayer::get_used_cells_by_id(int p_source_id, const Vector2i p_atlas_coords, int p_alternative_tile) const {
	// Returns the cells used in the tilemap.
	TypedArray<Vector2i> a;
	for (const KeyValue<Vector2i, CellData> &E : tile_map) {
		const TileMapCell &c = E.value.cell;
		if (c.source_id == TileSet::INVALID_SOURCE) {
			continue;
		}
		if ((p_source_id == TileSet::INVALID_SOURCE || p_source_id == c.source_id) &&
				(p_atlas_coords == TileSetSource::INVALID_ATLAS_COORDS || p_atlas_coords == c.get_atlas_coords()) &&
				(p_alternative_tile == TileSetSource::INVALID_TILE_ALTERNATIVE || p_alternative_tile == c.alternative_tile)) {
			a.push_back(E.key);
		}
	}

	return a;
}

Rect2i TileMapLayer::get_used_rect() const {
	// Return the rect of the currently used area.
	if (used_rect_cache_dirty) {
		used_rect_cache = Rect2i();

		bool first = true;
		for (const KeyValue<Vector2i, CellData> &E : tile_map) {
			const TileMapCell &c = E.value.cell;
			if (c.source_id == TileSet::INVALID_SOURCE) {
				continue;
			}
			if (first) {
				used_rect_cache = Rect2i(E.key.x, E.key.y, 0, 0);
				first = false;
			} else {
				used_rect_cache.expand_to(E.key);
			}
		}
		if (!first) {
			// Only if we have at least one cell.
			// The cache expands to top-left coordinate, so we add one full tile.
			used_rect_cache.size += Vector2i(1, 1);
		}
		used_rect_cache_dirty = false;
	}

	return used_rect_cache;
}

void TileMapLayer::set_name(String p_name) {
	if (name == p_name) {
		return;
	}
	name = p_name;
	tile_map_node->emit_signal(CoreStringNames::get_singleton()->changed);
}

String TileMapLayer::get_name() const {
	return name;
}

void TileMapLayer::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}
	enabled = p_enabled;
	dirty.flags[DIRTY_FLAGS_LAYER_ENABLED] = true;
	tile_map_node->queue_internal_update();
	tile_map_node->emit_signal(CoreStringNames::get_singleton()->changed);

	tile_map_node->update_configuration_warnings();
}

bool TileMapLayer::is_enabled() const {
	return enabled;
}

void TileMapLayer::set_modulate(Color p_modulate) {
	if (modulate == p_modulate) {
		return;
	}
	modulate = p_modulate;
	dirty.flags[DIRTY_FLAGS_LAYER_MODULATE] = true;
	tile_map_node->queue_internal_update();
	tile_map_node->emit_signal(CoreStringNames::get_singleton()->changed);
}

Color TileMapLayer::get_modulate() const {
	return modulate;
}

void TileMapLayer::set_y_sort_enabled(bool p_y_sort_enabled) {
	if (y_sort_enabled == p_y_sort_enabled) {
		return;
	}
	y_sort_enabled = p_y_sort_enabled;
	dirty.flags[DIRTY_FLAGS_LAYER_Y_SORT_ENABLED] = true;
	tile_map_node->queue_internal_update();
	tile_map_node->emit_signal(CoreStringNames::get_singleton()->changed);

	tile_map_node->update_configuration_warnings();
}

bool TileMapLayer::is_y_sort_enabled() const {
	return y_sort_enabled;
}

void TileMapLayer::set_y_sort_origin(int p_y_sort_origin) {
	if (y_sort_origin == p_y_sort_origin) {
		return;
	}
	y_sort_origin = p_y_sort_origin;
	dirty.flags[DIRTY_FLAGS_LAYER_Y_SORT_ORIGIN] = true;
	tile_map_node->queue_internal_update();
	tile_map_node->emit_signal(CoreStringNames::get_singleton()->changed);
}

int TileMapLayer::get_y_sort_origin() const {
	return y_sort_origin;
}

void TileMapLayer::set_z_index(int p_z_index) {
	if (z_index == p_z_index) {
		return;
	}
	z_index = p_z_index;
	dirty.flags[DIRTY_FLAGS_LAYER_Z_INDEX] = true;
	tile_map_node->queue_internal_update();
	tile_map_node->emit_signal(CoreStringNames::get_singleton()->changed);

	tile_map_node->update_configuration_warnings();
}

int TileMapLayer::get_z_index() const {
	return z_index;
}

void TileMapLayer::set_navigation_enabled(bool p_enabled) {
	if (navigation_enabled == p_enabled) {
		return;
	}
	navigation_enabled = p_enabled;
	dirty.flags[DIRTY_FLAGS_LAYER_NAVIGATION_ENABLED] = true;
	tile_map_node->queue_internal_update();
	tile_map_node->emit_signal(CoreStringNames::get_singleton()->changed);
}

bool TileMapLayer::is_navigation_enabled() const {
	return navigation_enabled;
}

void TileMapLayer::set_navigation_map(RID p_map) {
	ERR_FAIL_COND_MSG(!tile_map_node->is_inside_tree(), "A TileMap navigation map can only be changed while inside the SceneTree.");
	navigation_map = p_map;
	uses_world_navigation_map = p_map == tile_map_node->get_world_2d()->get_navigation_map();
}

RID TileMapLayer::get_navigation_map() const {
	if (navigation_map.is_valid()) {
		return navigation_map;
	}
	return RID();
}

void TileMapLayer::fix_invalid_tiles() {
	Ref<TileSet> tileset = tile_map_node->get_tileset();
	ERR_FAIL_COND_MSG(tileset.is_null(), "Cannot call fix_invalid_tiles() on a TileMap without a valid TileSet.");

	RBSet<Vector2i> coords;
	for (const KeyValue<Vector2i, CellData> &E : tile_map) {
		TileSetSource *source = *tileset->get_source(E.value.cell.source_id);
		if (!source || !source->has_tile(E.value.cell.get_atlas_coords()) || !source->has_alternative_tile(E.value.cell.get_atlas_coords(), E.value.cell.alternative_tile)) {
			coords.insert(E.key);
		}
	}
	for (const Vector2i &E : coords) {
		set_cell(E, TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
	}
}

bool TileMapLayer::has_body_rid(RID p_physics_body) const {
	return bodies_coords.has(p_physics_body);
}

Vector2i TileMapLayer::get_coords_for_body_rid(RID p_physics_body) const {
	return bodies_coords[p_physics_body];
}

TileMapLayer::~TileMapLayer() {
	if (!tile_map_node) {
		// Temporary layer.
		return;
	}

	in_destructor = true;
	clear();
	internal_update();
}

HashMap<Vector2i, TileSet::CellNeighbor> TerrainConstraint::get_overlapping_coords_and_peering_bits() const {
	HashMap<Vector2i, TileSet::CellNeighbor> output;

	ERR_FAIL_COND_V(is_center_bit(), output);

	Ref<TileSet> ts = tile_map->get_tileset();
	ERR_FAIL_COND_V(!ts.is_valid(), output);

	TileSet::TileShape shape = ts->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		switch (bit) {
			case 1:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_RIGHT_SIDE;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_LEFT_SIDE;
				break;
			case 2:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER;
				break;
			case 3:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_SIDE;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_SIDE;
				break;
			default:
				ERR_FAIL_V(output);
		}
	} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		switch (bit) {
			case 1:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE;
				break;
			case 2:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_LEFT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_CORNER)] = TileSet::CELL_NEIGHBOR_TOP_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_RIGHT_CORNER;
				break;
			case 3:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
				break;
			default:
				ERR_FAIL_V(output);
		}
	} else {
		// Half offset shapes.
		TileSet::TileOffsetAxis offset_axis = ts->get_tile_offset_axis();
		if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			switch (bit) {
				case 1:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_RIGHT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_LEFT_SIDE;
					break;
				case 2:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_CORNER;
					break;
				case 3:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE;
					break;
				case 4:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER;
					break;
				case 5:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
					break;
				default:
					ERR_FAIL_V(output);
			}
		} else {
			switch (bit) {
				case 1:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_RIGHT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
					break;
				case 2:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE;
					break;
				case 3:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
					break;
				case 4:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_SIDE;
					break;
				case 5:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
					break;
				default:
					ERR_FAIL_V(output);
			}
		}
	}
	return output;
}

TerrainConstraint::TerrainConstraint(const TileMap *p_tile_map, const Vector2i &p_position, int p_terrain) {
	tile_map = p_tile_map;

	Ref<TileSet> ts = tile_map->get_tileset();
	ERR_FAIL_COND(!ts.is_valid());

	bit = 0;
	base_cell_coords = p_position;
	terrain = p_terrain;
}

TerrainConstraint::TerrainConstraint(const TileMap *p_tile_map, const Vector2i &p_position, const TileSet::CellNeighbor &p_bit, int p_terrain) {
	// The way we build the constraint make it easy to detect conflicting constraints.
	tile_map = p_tile_map;

	Ref<TileSet> ts = tile_map->get_tileset();
	ERR_FAIL_COND(!ts.is_valid());

	TileSet::TileShape shape = ts->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
				bit = 1;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				bit = 2;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
				bit = 3;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
				bit = 1;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_SIDE:
				bit = 3;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
				break;
			default:
				ERR_FAIL();
				break;
		}
	} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
				bit = 1;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
				bit = 2;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
				bit = 3;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
				bit = 1;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_CORNER:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_CORNER);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
				bit = 3;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
				break;
			default:
				ERR_FAIL();
				break;
		}
	} else {
		// Half-offset shapes.
		TileSet::TileOffsetAxis offset_axis = ts->get_tile_offset_axis();
		if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			switch (p_bit) {
				case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
					bit = 1;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
					bit = 2;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
					bit = 3;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
					bit = 4;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
					bit = 5;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
					bit = 2;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
					bit = 1;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
					bit = 4;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
					bit = 3;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_CORNER:
					bit = 2;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
					bit = 5;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
					bit = 4;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
					break;
				default:
					ERR_FAIL();
					break;
			}
		} else {
			switch (p_bit) {
				case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
					bit = 1;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
					bit = 2;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
					bit = 3;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
					bit = 4;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
					bit = 1;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
					bit = 5;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
					bit = 3;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
					bit = 2;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
					bit = 1;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_SIDE:
					bit = 4;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
					bit = 3;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
					bit = 5;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
					break;
				default:
					ERR_FAIL();
					break;
			}
		}
	}
	terrain = p_terrain;
}

#define TILEMAP_CALL_FOR_LAYER(layer, function, ...) \
	if (layer < 0) {                                 \
		layer = layers.size() + layer;               \
	};                                               \
	ERR_FAIL_INDEX(layer, (int)layers.size());       \
	layers[layer]->function(__VA_ARGS__);

#define TILEMAP_CALL_FOR_LAYER_V(layer, err_value, function, ...) \
	if (layer < 0) {                                              \
		layer = layers.size() + layer;                            \
	};                                                            \
	ERR_FAIL_INDEX_V(layer, (int)layers.size(), err_value);       \
	return layers[layer]->function(__VA_ARGS__);

Vector2i TileMap::transform_coords_layout(const Vector2i &p_coords, TileSet::TileOffsetAxis p_offset_axis, TileSet::TileLayout p_from_layout, TileSet::TileLayout p_to_layout) {
	// Transform to stacked layout.
	Vector2i output = p_coords;
	if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
		SWAP(output.x, output.y);
	}
	switch (p_from_layout) {
		case TileSet::TILE_LAYOUT_STACKED:
			break;
		case TileSet::TILE_LAYOUT_STACKED_OFFSET:
			if (output.y % 2) {
				output.x -= 1;
			}
			break;
		case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
		case TileSet::TILE_LAYOUT_STAIRS_DOWN:
			if ((p_from_layout == TileSet::TILE_LAYOUT_STAIRS_RIGHT) ^ (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
				if (output.y < 0 && bool(output.y % 2)) {
					output = Vector2i(output.x + output.y / 2 - 1, output.y);
				} else {
					output = Vector2i(output.x + output.y / 2, output.y);
				}
			} else {
				if (output.x < 0 && bool(output.x % 2)) {
					output = Vector2i(output.x / 2 - 1, output.x + output.y * 2);
				} else {
					output = Vector2i(output.x / 2, output.x + output.y * 2);
				}
			}
			break;
		case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
		case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
			if ((p_from_layout == TileSet::TILE_LAYOUT_DIAMOND_RIGHT) ^ (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
				if ((output.x + output.y) < 0 && (output.x - output.y) % 2) {
					output = Vector2i((output.x + output.y) / 2 - 1, output.y - output.x);
				} else {
					output = Vector2i((output.x + output.y) / 2, -output.x + output.y);
				}
			} else {
				if ((output.x - output.y) < 0 && (output.x + output.y) % 2) {
					output = Vector2i((output.x - output.y) / 2 - 1, output.x + output.y);
				} else {
					output = Vector2i((output.x - output.y) / 2, output.x + output.y);
				}
			}
			break;
	}

	switch (p_to_layout) {
		case TileSet::TILE_LAYOUT_STACKED:
			break;
		case TileSet::TILE_LAYOUT_STACKED_OFFSET:
			if (output.y % 2) {
				output.x += 1;
			}
			break;
		case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
		case TileSet::TILE_LAYOUT_STAIRS_DOWN:
			if ((p_to_layout == TileSet::TILE_LAYOUT_STAIRS_RIGHT) ^ (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
				if (output.y < 0 && (output.y % 2)) {
					output = Vector2i(output.x - output.y / 2 + 1, output.y);
				} else {
					output = Vector2i(output.x - output.y / 2, output.y);
				}
			} else {
				if (output.y % 2) {
					if (output.y < 0) {
						output = Vector2i(2 * output.x + 1, -output.x + output.y / 2 - 1);
					} else {
						output = Vector2i(2 * output.x + 1, -output.x + output.y / 2);
					}
				} else {
					output = Vector2i(2 * output.x, -output.x + output.y / 2);
				}
			}
			break;
		case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
		case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
			if ((p_to_layout == TileSet::TILE_LAYOUT_DIAMOND_RIGHT) ^ (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
				if (output.y % 2) {
					if (output.y > 0) {
						output = Vector2i(output.x - output.y / 2, output.x + output.y / 2 + 1);
					} else {
						output = Vector2i(output.x - output.y / 2 + 1, output.x + output.y / 2);
					}
				} else {
					output = Vector2i(output.x - output.y / 2, output.x + output.y / 2);
				}
			} else {
				if (output.y % 2) {
					if (output.y < 0) {
						output = Vector2i(output.x + output.y / 2, -output.x + output.y / 2 - 1);
					} else {
						output = Vector2i(output.x + output.y / 2 + 1, -output.x + output.y / 2);
					}
				} else {
					output = Vector2i(output.x + output.y / 2, -output.x + output.y / 2);
				}
			}
			break;
	}

	if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
		SWAP(output.x, output.y);
	}

	return output;
}

void TileMap::set_selected_layer(int p_layer_id) {
	ERR_FAIL_COND(p_layer_id < -1 || p_layer_id >= (int)layers.size());
	if (selected_layer == p_layer_id) {
		return;
	}
	selected_layer = p_layer_id;
	emit_signal(CoreStringNames::get_singleton()->changed);

	// Update the layers modulation.
	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_SELECTED_LAYER);
	}
}

int TileMap::get_selected_layer() const {
	return selected_layer;
}

}

}

}

void TileMap::_notification(int p_what) {
	switch (p_what) {
		case TileMap::NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			// This is only executed when collision_animatable is enabled.

			bool in_editor = false;
#ifdef TOOLS_ENABLED
			in_editor = Engine::get_singleton()->is_editor_hint();
#endif
			if (is_inside_tree() && collision_animatable && !in_editor) {
				// Update transform on the physics tick when in animatable mode.
				last_valid_transform = new_transform;
				set_notify_local_transform(false);
				set_global_transform(new_transform);
				set_notify_local_transform(true);
			}
		} break;

		case TileMap::NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			// This is only executed when collision_animatable is enabled.

			bool in_editor = false;
#ifdef TOOLS_ENABLED
			in_editor = Engine::get_singleton()->is_editor_hint();
#endif

			if (is_inside_tree() && collision_animatable && !in_editor) {
				// Store last valid transform.
				new_transform = get_global_transform();

				// ... but then revert changes.
				set_notify_local_transform(false);
				set_global_transform(last_valid_transform);
				set_notify_local_transform(true);
			}
		} break;
	}
}

#ifndef DISABLE_DEPRECATED
// Deprecated methods.
void TileMap::force_update(int p_layer) {
	notify_runtime_tile_data_update(p_layer);
	update_internals();
}
#endif

void TileMap::set_rendering_quadrant_size(int p_size) {
	ERR_FAIL_COND_MSG(p_size < 1, "TileMapQuadrant size cannot be smaller than 1.");

	rendering_quadrant_size = p_size;
	for (TileMapLayer *layer : layers) {
		layer->set_rendering_quadrant_size(p_size);
	}
	_emit_changed();
}

int TileMap::get_rendering_quadrant_size() const {
	return rendering_quadrant_size;
}

void TileMap::draw_tile(RID p_canvas_item, const Vector2 &p_position, const Ref<TileSet> p_tile_set, int p_atlas_source_id, const Vector2i &p_atlas_coords, int p_alternative_tile, int p_frame, Color p_modulation, const TileData *p_tile_data_override, real_t p_normalized_animation_offset) {
	ERR_FAIL_COND(!p_tile_set.is_valid());
	ERR_FAIL_COND(!p_tile_set->has_source(p_atlas_source_id));
	ERR_FAIL_COND(!p_tile_set->get_source(p_atlas_source_id)->has_tile(p_atlas_coords));
	ERR_FAIL_COND(!p_tile_set->get_source(p_atlas_source_id)->has_alternative_tile(p_atlas_coords, p_alternative_tile));
	TileSetSource *source = *p_tile_set->get_source(p_atlas_source_id);
	TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
	if (atlas_source) {
		// Check for the frame.
		if (p_frame >= 0) {
			ERR_FAIL_INDEX(p_frame, atlas_source->get_tile_animation_frames_count(p_atlas_coords));
		}

		// Get the texture.
		Ref<Texture2D> tex = atlas_source->get_runtime_texture();
		if (!tex.is_valid()) {
			return;
		}

		// Check if we are in the texture, return otherwise.
		Vector2i grid_size = atlas_source->get_atlas_grid_size();
		if (p_atlas_coords.x >= grid_size.x || p_atlas_coords.y >= grid_size.y) {
			return;
		}

		// Get tile data.
		const TileData *tile_data = p_tile_data_override ? p_tile_data_override : atlas_source->get_tile_data(p_atlas_coords, p_alternative_tile);

		// Get the tile modulation.
		Color modulate = tile_data->get_modulate() * p_modulation;

		// Compute the offset.
		Vector2 tile_offset = tile_data->get_texture_origin();

		// Get destination rect.
		Rect2 dest_rect;
		dest_rect.size = atlas_source->get_runtime_tile_texture_region(p_atlas_coords).size;
		dest_rect.size.x += FP_ADJUST;
		dest_rect.size.y += FP_ADJUST;

		bool transpose = tile_data->get_transpose() ^ bool(p_alternative_tile & TileSetAtlasSource::TRANSFORM_TRANSPOSE);
		if (transpose) {
			dest_rect.position = (p_position - Vector2(dest_rect.size.y, dest_rect.size.x) / 2 - tile_offset);
		} else {
			dest_rect.position = (p_position - dest_rect.size / 2 - tile_offset);
		}

		if (tile_data->get_flip_h() ^ bool(p_alternative_tile & TileSetAtlasSource::TRANSFORM_FLIP_H)) {
			dest_rect.size.x = -dest_rect.size.x;
		}

		if (tile_data->get_flip_v() ^ bool(p_alternative_tile & TileSetAtlasSource::TRANSFORM_FLIP_V)) {
			dest_rect.size.y = -dest_rect.size.y;
		}

		// Draw the tile.
		if (p_frame >= 0) {
			Rect2i source_rect = atlas_source->get_runtime_tile_texture_region(p_atlas_coords, p_frame);
			tex->draw_rect_region(p_canvas_item, dest_rect, source_rect, modulate, transpose, p_tile_set->is_uv_clipping());
		} else if (atlas_source->get_tile_animation_frames_count(p_atlas_coords) == 1) {
			Rect2i source_rect = atlas_source->get_runtime_tile_texture_region(p_atlas_coords, 0);
			tex->draw_rect_region(p_canvas_item, dest_rect, source_rect, modulate, transpose, p_tile_set->is_uv_clipping());
		} else {
			real_t speed = atlas_source->get_tile_animation_speed(p_atlas_coords);
			real_t animation_duration = atlas_source->get_tile_animation_total_duration(p_atlas_coords) / speed;
			real_t animation_offset = p_normalized_animation_offset * animation_duration;
			// Accumulate durations unaffected by the speed to avoid accumulating floating point division errors.
			// Aka do `sum(duration[i]) / speed` instead of `sum(duration[i] / speed)`.
			real_t time_unscaled = 0.0;
			for (int frame = 0; frame < atlas_source->get_tile_animation_frames_count(p_atlas_coords); frame++) {
				real_t frame_duration_unscaled = atlas_source->get_tile_animation_frame_duration(p_atlas_coords, frame);
				real_t slice_start = time_unscaled / speed;
				real_t slice_end = (time_unscaled + frame_duration_unscaled) / speed;
				RenderingServer::get_singleton()->canvas_item_add_animation_slice(p_canvas_item, animation_duration, slice_start, slice_end, animation_offset);

				Rect2i source_rect = atlas_source->get_runtime_tile_texture_region(p_atlas_coords, frame);
				tex->draw_rect_region(p_canvas_item, dest_rect, source_rect, modulate, transpose, p_tile_set->is_uv_clipping());

				time_unscaled += frame_duration_unscaled;
			}
			RenderingServer::get_singleton()->canvas_item_add_animation_slice(p_canvas_item, 1.0, 0.0, 1.0, 0.0);
		}
	}
}

void TileMap::set_tileset(const Ref<TileSet> &p_tileset) {
	if (p_tileset == tile_set) {
		return;
	}

	// Set the tileset, registering to its changes.
	if (tile_set.is_valid()) {
		tile_set->disconnect_changed(callable_mp(this, &TileMap::_tile_set_changed));
	}

	tile_set = p_tileset;

	if (tile_set.is_valid()) {
		tile_set->connect_changed(callable_mp(this, &TileMap::_tile_set_changed));
	}

	for (int i = 0; i < get_child_count(); i++) {
		TileMapLayer *layer = Object::cast_to<TileMapLayer>(get_child(i));
		if (layer) {
			layer->set_tile_set(tile_set);
		}
	}
}

Ref<TileSet> TileMap::get_tileset() const {
	return tile_set;
}

int TileMap::get_layers_count() const {
	return layers.size();
}

void TileMap::add_layer(int p_to_pos) {
	if (p_to_pos < 0) {
		p_to_pos = layers.size() + p_to_pos + 1;
	}

	ERR_FAIL_INDEX(p_to_pos, (int)layers.size() + 1);

	// Must clear before adding the layer.
	TileMapLayer *new_layer = memnew(TileMapLayer);
	layers.insert(p_to_pos, new_layer);
	add_child(new_layer, false, INTERNAL_MODE_FRONT);
	new_layer->set_name(vformat("Layer%d", p_to_pos));
	new_layer->set_tile_set(tile_set);
	move_child(new_layer, p_to_pos);
	for (uint32_t i = 0; i < layers.size(); i++) {
		layers[i]->set_as_tile_map_internal_node(i);
	}
	new_layer->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &TileMap::_emit_changed));

	notify_property_list_changed();

	_emit_changed();

	update_configuration_warnings();
}

void TileMap::move_layer(int p_layer, int p_to_pos) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	ERR_FAIL_INDEX(p_to_pos, (int)layers.size() + 1);

	// Clear before shuffling layers.
	TileMapLayer *layer = layers[p_layer];
	layers.insert(p_to_pos, layer);
	layers.remove_at(p_to_pos < p_layer ? p_layer + 1 : p_layer);
	for (uint32_t i = 0; i < layers.size(); i++) {
		move_child(layers[i], i);
		layers[i]->set_as_tile_map_internal_node(i);
	}
	notify_property_list_changed();

	_emit_changed();

	update_configuration_warnings();
}

void TileMap::remove_layer(int p_layer) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	// Clear before removing the layer.
	TileMapLayer *removed = layers[p_layer];
	layers.remove_at(p_layer);
	remove_child(removed);
	removed->queue_free();

	for (uint32_t i = 0; i < layers.size(); i++) {
		layers[i]->set_as_tile_map_internal_node(i);
	}
	notify_property_list_changed();

	_emit_changed();

	update_configuration_warnings();
}

void TileMap::set_layer_name(int p_layer, String p_name) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_name, p_name);
}

String TileMap::get_layer_name(int p_layer) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, "", get_name);
}

void TileMap::set_layer_enabled(int p_layer, bool p_enabled) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_enabled, p_enabled);
}

bool TileMap::is_layer_enabled(int p_layer) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, false, is_enabled);
}

void TileMap::set_layer_modulate(int p_layer, Color p_modulate) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_modulate, p_modulate);
}

Color TileMap::get_layer_modulate(int p_layer) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, Color(), get_modulate);
}

void TileMap::set_layer_y_sort_enabled(int p_layer, bool p_y_sort_enabled) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_y_sort_enabled, p_y_sort_enabled);
}

bool TileMap::is_layer_y_sort_enabled(int p_layer) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, false, is_y_sort_enabled);
}

void TileMap::set_layer_y_sort_origin(int p_layer, int p_y_sort_origin) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_y_sort_origin, p_y_sort_origin);
}

int TileMap::get_layer_y_sort_origin(int p_layer) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, 0, get_y_sort_origin);
}

void TileMap::set_layer_z_index(int p_layer, int p_z_index) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_z_index, p_z_index);
}

int TileMap::get_layer_z_index(int p_layer) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, 0, get_z_index);
}

void TileMap::set_layer_navigation_enabled(int p_layer, bool p_enabled) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_navigation_enabled, p_enabled);
}

bool TileMap::is_layer_navigation_enabled(int p_layer) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, false, is_navigation_enabled);
}

void TileMap::set_layer_navigation_map(int p_layer, RID p_map) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_navigation_map, p_map);
}

RID TileMap::get_layer_navigation_map(int p_layer) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, RID(), get_navigation_map);
}

void TileMap::set_collision_animatable(bool p_collision_animatable) {
	if (collision_animatable == p_collision_animatable) {
		return;
	}
	collision_animatable = p_collision_animatable;
	set_notify_local_transform(p_collision_animatable);
	set_physics_process_internal(p_collision_animatable);
	for (TileMapLayer *layer : layers) {
		layer->set_use_kinematic_bodies(layer);
	}
}

bool TileMap::is_collision_animatable() const {
	return collision_animatable;
}

void TileMap::set_collision_visibility_mode(TileMap::VisibilityMode p_show_collision) {
	if (collision_visibility_mode == p_show_collision) {
		return;
	}
	collision_visibility_mode = p_show_collision;
	for (TileMapLayer *layer : layers) {
		layer->set_collision_visibility_mode(TileMapLayer::DebugVisibilityMode(p_show_collision));
	}
	_emit_changed();
}

TileMap::VisibilityMode TileMap::get_collision_visibility_mode() const {
	return collision_visibility_mode;
}

void TileMap::set_navigation_visibility_mode(TileMap::VisibilityMode p_show_navigation) {
	if (navigation_visibility_mode == p_show_navigation) {
		return;
	}
	navigation_visibility_mode = p_show_navigation;
	for (TileMapLayer *layer : layers) {
		layer->set_navigation_visibility_mode(TileMapLayer::DebugVisibilityMode(p_show_navigation));
	}
	_emit_changed();
}

TileMap::VisibilityMode TileMap::get_navigation_visibility_mode() const {
	return navigation_visibility_mode;
}

void TileMap::set_y_sort_enabled(bool p_enable) {
	if (is_y_sort_enabled() == p_enable) {
		return;
	}
	Node2D::set_y_sort_enabled(p_enable);
	for (TileMapLayer *layer : layers) {
		layer->set_y_sort_enabled(p_enable);
	}
	_emit_changed();
	update_configuration_warnings();
}

void TileMap::set_cell(int p_layer, const Vector2i &p_coords, int p_source_id, const Vector2i p_atlas_coords, int p_alternative_tile) {
	if (p_layer < 0) {
		p_layer = layers.size() + p_layer;
	};
	if ((p_layer) < 0 || (p_layer) >= ((int)layers.size())) {
		_err_print_index_error(__FUNCTION__, "C:\\Users\\LPC\\source\\repos\\Godot Extension Project\\Godot Engine Fork\\godot\\scene\\2d\\tile_map.cpp", 3237, p_layer, (int)layers.size(), "p_layer", "(int)layers.size()");
		return;
	} else
		((void)0);
	layers[p_layer]->set_cell(p_coords, p_source_id, p_atlas_coords, p_alternative_tile);
	;
}

void TileMap::erase_cell(int p_layer, const Vector2i &p_coords) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_cell, p_coords, TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
}

int TileMap::get_cell_source_id(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	if (p_use_proxies && tile_set.is_valid()) {
		if (p_layer < 0) {
			p_layer = layers.size() + p_layer;
		}
		ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), TileSet::INVALID_SOURCE);

		int source_id = layers[p_layer]->get_cell_source_id(p_coords);
		Vector2i atlas_coords = layers[p_layer]->get_cell_atlas_coords(p_coords);
		int alternative_id = layers[p_layer]->get_cell_alternative_tile(p_coords);

		Array arr = tile_set->map_tile_proxy(source_id, atlas_coords, alternative_id);
		ERR_FAIL_COND_V(arr.size() != 3, TileSet::INVALID_SOURCE);
		return arr[0];
	} else {
		TILEMAP_CALL_FOR_LAYER_V(p_layer, TileSet::INVALID_SOURCE, get_cell_source_id, p_coords);
	}
}

Vector2i TileMap::get_cell_atlas_coords(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	if (p_use_proxies && tile_set.is_valid()) {
		if (p_layer < 0) {
			p_layer = layers.size() + p_layer;
		}
		ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), TileSetAtlasSource::INVALID_ATLAS_COORDS);

		int source_id = layers[p_layer]->get_cell_source_id(p_coords);
		Vector2i atlas_coords = layers[p_layer]->get_cell_atlas_coords(p_coords);
		int alternative_id = layers[p_layer]->get_cell_alternative_tile(p_coords);

		Array arr = tile_set->map_tile_proxy(source_id, atlas_coords, alternative_id);
		ERR_FAIL_COND_V(arr.size() != 3, TileSetSource::INVALID_ATLAS_COORDS);
		return arr[1];
	} else {
		TILEMAP_CALL_FOR_LAYER_V(p_layer, TileSetSource::INVALID_ATLAS_COORDS, get_cell_atlas_coords, p_coords);
	}
}

int TileMap::get_cell_alternative_tile(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
		int source_id = layers[p_layer]->get_cell_source_id(p_coords);
		Vector2i atlas_coords = layers[p_layer]->get_cell_atlas_coords(p_coords);
		int alternative_id = layers[p_layer]->get_cell_alternative_tile(p_coords);

		Array arr = tile_set->map_tile_proxy(source_id, atlas_coords, alternative_id);
		ERR_FAIL_COND_V(arr.size() != 3, TileSetSource::INVALID_TILE_ALTERNATIVE);
		return arr[2];
	} else {
		TILEMAP_CALL_FOR_LAYER_V(p_layer, TileSetSource::INVALID_TILE_ALTERNATIVE, get_cell_alternative_tile, p_coords);
	}
}
	}
TileData *TileMap::get_cell_tile_data(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	if (p_use_proxies && tile_set.is_valid()) {
		if (p_layer < 0) {
			p_layer = layers.size() + p_layer;
		}
		ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), nullptr);
		}
		int source_id = layers[p_layer]->get_cell_source_id(p_coords);
		Vector2i atlas_coords = layers[p_layer]->get_cell_atlas_coords(p_coords);
		int alternative_id = layers[p_layer]->get_cell_alternative_tile(p_coords);

		Array arr = tile_set->map_tile_proxy(source_id, atlas_coords, alternative_id);
		ERR_FAIL_COND_V(arr.size() != 3, nullptr);

		Ref<TileSetAtlasSource> atlas_source = tile_set->get_source(arr[0]);
		if (atlas_source.is_valid()) {
			return atlas_source->get_tile_data(arr[1], arr[2]);
		} else {
			return nullptr;
		}
	} else {
		TILEMAP_CALL_FOR_LAYER_V(p_layer, nullptr, get_cell_tile_data, p_coords);
	}
}

Ref<TileMapPattern> TileMap::get_pattern(int p_layer, TypedArray<Vector2i> p_coords_array) {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, Ref<TileMapPattern>(), get_pattern, p_coords_array);
}
	TILEMAP_CALL_FOR_LAYER_V(p_layer, Ref<TileMapPattern>(), get_pattern, p_coords_array);
}
		}
	}
		}
	}

Vector2i TileMap::map_pattern(const Vector2i &p_position_in_tilemap, const Vector2i &p_coords_in_pattern, Ref<TileMapPattern> p_pattern) {
	ERR_FAIL_COND_V(!tile_set.is_valid(), Vector2i());
	return tile_set->map_pattern(p_position_in_tilemap, p_coords_in_pattern, p_pattern);
}

void TileMap::set_pattern(int p_layer, const Vector2i &p_position, const Ref<TileMapPattern> p_pattern) {
	
	if (p_pattern->get_is_single_layer()) {
		print_line("set_pattern single layer called");
		if (p_layer < 0) {
			p_layer = layers.size() + p_layer;
		};
		if ((p_layer) < 0 || (p_layer) >= ((int)layers.size())) {
			_err_print_index_error(__FUNCTION__, "C:\\Users\\LPC\\source\\repos\\Godot Extension Project\\Godot Engine Fork\\godot\\scene\\2d\\tile_map.cpp", 3259, p_layer, (int)layers.size(), "p_layer", "(int)layers.size()");
			return;
		} else
			((void)0);
		layers[p_layer]->set_pattern_layer(p_layer, p_position, p_pattern);
	}

	else {
		print_line("set_pattern multi layer called");
		for (int pattern_layer = 0; pattern_layer < p_pattern->get_number_of_layers(); pattern_layer++) {
			layers[pattern_layer]->set_pattern_layer(pattern_layer, p_position, p_pattern);
		}
	}
}


HashMap<Vector2i, TileSet::TerrainsPattern> TileMap::terrain_fill_constraints(int p_layer, const Vector<Vector2i> &p_to_replace, int p_terrain_set, const RBSet<TerrainConstraint> &p_constraints) {
	HashMap<Vector2i, TileSet::TerrainsPattern> err_value;
	TILEMAP_CALL_FOR_LAYER_V(p_layer, err_value, terrain_fill_constraints, p_to_replace, p_terrain_set, p_constraints);
}

HashMap<Vector2i, TileSet::TerrainsPattern> TileMap::terrain_fill_connect(int p_layer, const Vector<Vector2i> &p_coords_array, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	HashMap<Vector2i, TileSet::TerrainsPattern> err_value;
	TILEMAP_CALL_FOR_LAYER_V(p_layer, err_value, terrain_fill_connect, p_coords_array, p_terrain_set, p_terrain, p_ignore_empty_terrains);
}

HashMap<Vector2i, TileSet::TerrainsPattern> TileMap::terrain_fill_path(int p_layer, const Vector<Vector2i> &p_coords_array, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	HashMap<Vector2i, TileSet::TerrainsPattern> err_value;
	TILEMAP_CALL_FOR_LAYER_V(p_layer, err_value, terrain_fill_path, p_coords_array, p_terrain_set, p_terrain, p_ignore_empty_terrains);
}

HashMap<Vector2i, TileSet::TerrainsPattern> TileMap::terrain_fill_pattern(int p_layer, const Vector<Vector2i> &p_coords_array, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern, bool p_ignore_empty_terrains) {
	HashMap<Vector2i, TileSet::TerrainsPattern> err_value;
	TILEMAP_CALL_FOR_LAYER_V(p_layer, err_value, terrain_fill_pattern, p_coords_array, p_terrain_set, p_terrains_pattern, p_ignore_empty_terrains);
}

void TileMap::set_cells_terrain_connect(int p_layer, TypedArray<Vector2i> p_cells, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_cells_terrain_connect, p_cells, p_terrain_set, p_terrain, p_ignore_empty_terrains);
}

void TileMap::set_cells_terrain_path(int p_layer, TypedArray<Vector2i> p_path, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_cells_terrain_path, p_path, p_terrain_set, p_terrain, p_ignore_empty_terrains);
}

TileMapCell TileMap::get_cell(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	if (p_use_proxies) {
		WARN_DEPRECATED_MSG("use_proxies is deprecated.");
	}
	TILEMAP_CALL_FOR_LAYER_V(p_layer, TileMapCell(), get_cell, p_coords);
}

Vector2i TileMap::get_coords_for_body_rid(RID p_physics_body) {
	for (const TileMapLayer *layer : layers) {
		if (layer->has_body_rid(p_physics_body)) {
			return layer->get_coords_for_body_rid(p_physics_body);
		}
	}
	ERR_FAIL_V_MSG(Vector2i(), vformat("No tiles for the given body RID %d.", p_physics_body.get_id()));
}

int TileMap::get_layer_for_body_rid(RID p_physics_body) {
	for (uint32_t i = 0; i < layers.size(); i++) {
		if (layers[i]->has_body_rid(p_physics_body)) {
			return i;
		}
	}
	ERR_FAIL_V_MSG(-1, vformat("No tiles for the given body RID %d.", p_physics_body.get_id()));
}

void TileMap::fix_invalid_tiles() {
	for (TileMapLayer *layer : layers) {
		layer->fix_invalid_tiles();
	}
}

#ifdef TOOLS_ENABLED
TileMapLayer *TileMap::duplicate_layer_from_internal(int p_layer) {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), nullptr);
	return Object::cast_to<TileMapLayer>(layers[p_layer]->duplicate(DUPLICATE_USE_INSTANTIATION | DUPLICATE_FROM_EDITOR));
}
#endif // TOOLS_ENABLED

void TileMap::clear_layer(int p_layer) {
	TILEMAP_CALL_FOR_LAYER(p_layer, clear)
}

void TileMap::clear() {
	for (TileMapLayer *layer : layers) {
		layer->clear();
	}
}

void TileMap::update_internals() {
	for (TileMapLayer *layer : layers) {
		layer->update_internals();
	}
}

void TileMap::notify_runtime_tile_data_update(int p_layer) {
	if (p_layer >= 0) {
		TILEMAP_CALL_FOR_LAYER(p_layer, notify_runtime_tile_data_update);
	} else {
		for (TileMapLayer *layer : layers) {
			layer->notify_runtime_tile_data_update();
		}
	}
}

#ifdef TOOLS_ENABLED
Rect2 TileMap::_edit_get_rect() const {
	// Return the visible rect of the tilemap.
	if (layers.is_empty()) {
		return Rect2();
	}

	bool any_changed = false;
	bool changed = false;
	Rect2 rect = layers[0]->get_rect(changed);
	any_changed |= changed;
	for (uint32_t i = 1; i < layers.size(); i++) {
		rect = rect.merge(layers[i]->get_rect(changed));
		any_changed |= changed;
	}
	const_cast<TileMap *>(this)->item_rect_changed(any_changed);
	return rect;
}
#endif

bool TileMap::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (p_name == "format") {
		if (p_value.get_type() == Variant::INT) {
			format = (TileMapDataFormat)(p_value.operator int64_t()); // Set format used for loading.
			return true;
		}
	}
#ifndef DISABLE_DEPRECATED
	else if (p_name == "tile_data") { // Kept for compatibility reasons.
		if (p_value.is_array()) {
			if (layers.size() == 0) {
				TileMapLayer *new_layer = memnew(TileMapLayer);
				add_child(new_layer, false, INTERNAL_MODE_FRONT);
				new_layer->set_as_tile_map_internal_node(0);
				new_layer->set_name("Layer0");
				new_layer->set_tile_set(tile_set);
				new_layer->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &TileMap::_emit_changed));
				layers.push_back(new_layer);
			}
			_set_tile_map_data_using_compatibility_format(0, format, p_value);
			_emit_changed();
			return true;
		}
		return false;
	} else if (p_name == "cell_quadrant_size") {
		set_rendering_quadrant_size(p_value);
		return true;
	}
#endif // DISABLE_DEPRECATED
	else if (components.size() == 2 && components[0].begins_with("layer_") && components[0].trim_prefix("layer_").is_valid_int()) {
		int index = components[0].trim_prefix("layer_").to_int();
		if (index < 0) {
			return false;
		}

		if (index >= (int)layers.size()) {
			while (index >= (int)layers.size()) {
				TileMapLayer *new_layer = memnew(TileMapLayer);
				add_child(new_layer, false, INTERNAL_MODE_FRONT);
				new_layer->set_as_tile_map_internal_node(index);
				new_layer->set_name(vformat("Layer%d", index));
				new_layer->set_tile_set(tile_set);
				new_layer->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &TileMap::_emit_changed));
				layers.push_back(new_layer);
			}

			notify_property_list_changed();
			_emit_changed();
			update_configuration_warnings();
		}

		if (components[1] == "name") {
			set_layer_name(index, p_value);
			return true;
		} else if (components[1] == "enabled") {
			set_layer_enabled(index, p_value);
			return true;
		} else if (components[1] == "modulate") {
			set_layer_modulate(index, p_value);
			return true;
		} else if (components[1] == "y_sort_enabled") {
			set_layer_y_sort_enabled(index, p_value);
			return true;
		} else if (components[1] == "y_sort_origin") {
			set_layer_y_sort_origin(index, p_value);
			return true;
		} else if (components[1] == "z_index") {
			set_layer_z_index(index, p_value);
			return true;
		} else if (components[1] == "navigation_enabled") {
			set_layer_navigation_enabled(index, p_value);
			return true;
		} else if (components[1] == "tile_data") {
			_set_tile_map_data_using_compatibility_format(index, format, p_value);
			_emit_changed();
			return true;
		} else {
			return false;
		}
	}
	return false;
}

bool TileMap::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (p_name == "format") {
		r_ret = TileMapDataFormat::TILE_MAP_DATA_FORMAT_MAX - 1; // When saving, always save highest format.
		return true;
	}
#ifndef DISABLE_DEPRECATED
	else if (p_name == "cell_quadrant_size") { // Kept for compatibility reasons.
		r_ret = get_rendering_quadrant_size();
		return true;
	}
#endif
	else if (components.size() == 2 && components[0].begins_with("layer_") && components[0].trim_prefix("layer_").is_valid_int()) {
		int index = components[0].trim_prefix("layer_").to_int();
		if (index < 0 || index >= (int)layers.size()) {
			return false;
		}

		if (components[1] == "name") {
			r_ret = get_layer_name(index);
			return true;
		} else if (components[1] == "enabled") {
			r_ret = is_layer_enabled(index);
			return true;
		} else if (components[1] == "modulate") {
			r_ret = get_layer_modulate(index);
			return true;
		} else if (components[1] == "y_sort_enabled") {
			r_ret = is_layer_y_sort_enabled(index);
			return true;
		} else if (components[1] == "y_sort_origin") {
			r_ret = get_layer_y_sort_origin(index);
			return true;
		} else if (components[1] == "z_index") {
			r_ret = get_layer_z_index(index);
			return true;
		} else if (components[1] == "navigation_enabled") {
			r_ret = is_layer_navigation_enabled(index);
			return true;
		} else if (components[1] == "tile_data") {
			r_ret = _get_tile_map_data_using_compatibility_format(index);
			return true;
		} else {
			return false;
		}
	}
	return false;
}

void TileMap::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::INT, "format", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
	p_list->push_back(PropertyInfo(Variant::NIL, "Layers", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));

#define MAKE_LAYER_PROPERTY(m_type, m_name, m_hint)                                                                                                                                                      \
	{                                                                                                                                                                                                    \
		const String property_name = vformat("layer_%d/" m_name, i);                                                                                                                                     \
		p_list->push_back(PropertyInfo(m_type, property_name, PROPERTY_HINT_NONE, m_hint, (get(property_name) == property_get_revert(property_name)) ? PROPERTY_USAGE_EDITOR : PROPERTY_USAGE_DEFAULT)); \
	}

	for (uint32_t i = 0; i < layers.size(); i++) {
		MAKE_LAYER_PROPERTY(Variant::STRING, "name", "");
		MAKE_LAYER_PROPERTY(Variant::BOOL, "enabled", "");
		MAKE_LAYER_PROPERTY(Variant::COLOR, "modulate", "");
		MAKE_LAYER_PROPERTY(Variant::BOOL, "y_sort_enabled", "");
		MAKE_LAYER_PROPERTY(Variant::INT, "y_sort_origin", "suffix:px");
		MAKE_LAYER_PROPERTY(Variant::INT, "z_index", "");
		MAKE_LAYER_PROPERTY(Variant::BOOL, "navigation_enabled", "");
		p_list->push_back(PropertyInfo(Variant::OBJECT, vformat("layer_%d/tile_data", i), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	}

#undef MAKE_LAYER_PROPERTY
}

bool TileMap::_property_can_revert(const StringName &p_name) const {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (components.size() == 2 && components[0].begins_with("layer_")) {
		int index = components[0].trim_prefix("layer_").to_int();
		if (index <= 0 || index >= (int)layers.size()) {
			return false;
		}

		if (components[1] == "name") {
			return layers[index]->get_name() != default_layer->get_name();
		} else if (components[1] == "enabled") {
			return layers[index]->is_enabled() != default_layer->is_enabled();
		} else if (components[1] == "modulate") {
			return layers[index]->get_modulate() != default_layer->get_modulate();
		} else if (components[1] == "y_sort_enabled") {
			return layers[index]->is_y_sort_enabled() != default_layer->is_y_sort_enabled();
		} else if (components[1] == "y_sort_origin") {
			return layers[index]->get_y_sort_origin() != default_layer->get_y_sort_origin();
		} else if (components[1] == "z_index") {
			return layers[index]->get_z_index() != default_layer->get_z_index();
		} else if (components[1] == "navigation_enabled") {
			return layers[index]->is_navigation_enabled() != default_layer->is_navigation_enabled();
		}
	}

	return false;
}

bool TileMap::_property_get_revert(const StringName &p_name, Variant &r_property) const {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (components.size() == 2 && components[0].begins_with("layer_")) {
		int index = components[0].trim_prefix("layer_").to_int();
		if (index <= 0 || index >= (int)layers.size()) {
			return false;
		}

		if (components[1] == "name") {
			r_property = default_layer->get_name();
			return true;
		} else if (components[1] == "enabled") {
			r_property = default_layer->is_enabled();
			return true;
		} else if (components[1] == "modulate") {
			r_property = default_layer->get_modulate();
			return true;
		} else if (components[1] == "y_sort_enabled") {
			r_property = default_layer->is_y_sort_enabled();
			return true;
		} else if (components[1] == "y_sort_origin") {
			r_property = default_layer->get_y_sort_origin();
			return true;
		} else if (components[1] == "z_index") {
			r_property = default_layer->get_z_index();
			return true;
		} else if (components[1] == "navigation_enabled") {
			r_property = default_layer->is_navigation_enabled();
			return true;
		}
	}

	return false;
}

Vector2 TileMap::map_to_local(const Vector2i &p_pos) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), Vector2());
	return tile_set->map_to_local(p_pos);
}

Vector2i TileMap::local_to_map(const Vector2 &p_pos) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), Vector2i());
	return tile_set->local_to_map(p_pos);
}

bool TileMap::is_existing_neighbor(TileSet::CellNeighbor p_cell_neighbor) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), false);
	return tile_set->is_existing_neighbor(p_cell_neighbor);
}

Vector2i TileMap::get_neighbor_cell(const Vector2i &p_coords, TileSet::CellNeighbor p_cell_neighbor) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), Vector2i());
	return tile_set->get_neighbor_cell(p_coords, p_cell_neighbor);
}

TypedArray<Vector2i> TileMap::get_used_cells(int p_layer) const {
	if (p_layer < 0) {
		p_layer = layers.size() + p_layer;
	};
	if ((p_layer) < 0 || (p_layer) >= ((int)layers.size())) {
		_err_print_index_error(__FUNCTION__, "C:\\Users\\LPC\\source\\repos\\Godot Extension Project\\Godot Engine Fork\\godot\\scene\\2d\\tile_map.cpp", 4211, p_layer, (int)layers.size(), "p_layer", "(int)layers.size()");
		return TypedArray<Vector2i>();
	} else
		((void)0);
	return layers[p_layer]->get_used_cells();
	;
}

TypedArray<Vector2i> TileMap::get_used_cells_by_id(int p_layer, int p_source_id, const Vector2i p_atlas_coords, int p_alternative_tile) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, TypedArray<Vector2i>(), get_used_cells_by_id, p_source_id, p_atlas_coords, p_alternative_tile);
}

Rect2i TileMap::get_used_rect() const {
	// Return the visible rect of the tilemap.
	bool first = true;
	Rect2i rect = Rect2i();
	for (const TileMapLayer *layer : layers) {
		Rect2i layer_rect = layer->get_used_rect();
		if (layer_rect == Rect2i()) {
			continue;
		}
		if (first) {
			rect = layer_rect;
			first = false;
		} else {
			rect = rect.merge(layer_rect);
		}
	}
	return rect;
}

// --- Override some methods of the CanvasItem class to pass the changes to the quadrants CanvasItems ---

void TileMap::set_light_mask(int p_light_mask) {
	// Set light mask for occlusion and applies it to all layers too.
	CanvasItem::set_light_mask(p_light_mask);
	for (TileMapLayer *layer : layers) {
		layer->set_light_mask(p_light_mask);
	}
}

void TileMap::set_self_modulate(const Color &p_self_modulate) {
	// Set self_modulation and applies it to all layers too.
	CanvasItem::set_self_modulate(p_self_modulate);
	for (TileMapLayer *layer : layers) {
		layer->set_self_modulate(p_self_modulate);
	}
}

void TileMap::set_texture_filter(TextureFilter p_texture_filter) {
	// Set a default texture filter and applies it to all layers too.
	CanvasItem::set_texture_filter(p_texture_filter);
	for (TileMapLayer *layer : layers) {
		layer->set_texture_filter(p_texture_filter);
	}
}

void TileMap::set_texture_repeat(CanvasItem::TextureRepeat p_texture_repeat) {
	// Set a default texture repeat and applies it to all layers too.
	CanvasItem::set_texture_repeat(p_texture_repeat);
	for (TileMapLayer *layer : layers) {
		layer->set_texture_repeat(p_texture_repeat);
	}
}

TypedArray<Vector2i> TileMap::get_surrounding_cells(const Vector2i &p_coords) {
	if (!tile_set.is_valid()) {
		return TypedArray<Vector2i>();
	}

	return tile_set->get_surrounding_cells(p_coords);
}

PackedStringArray TileMap::get_configuration_warnings() const {
	PackedStringArray warnings = Node::get_configuration_warnings();

	// Retrieve the set of Z index values with a Y-sorted layer.
	RBSet<int> y_sorted_z_index;
	for (const TileMapLayer *layer : layers) {
		if (layer->is_y_sort_enabled()) {
			y_sorted_z_index.insert(layer->get_z_index());
		}
	}

	// Check if we have a non-sorted layer in a Z-index with a Y-sorted layer.
	for (const TileMapLayer *layer : layers) {
		if (!layer->is_y_sort_enabled() && y_sorted_z_index.has(layer->get_z_index())) {
			warnings.push_back(RTR("A Y-sorted layer has the same Z-index value as a not Y-sorted layer.\nThis may lead to unwanted behaviors, as a layer that is not Y-sorted will be Y-sorted as a whole with tiles from Y-sorted layers."));
			break;
		}
	}

	if (!is_y_sort_enabled()) {
		// Check if Y-sort is enabled on a layer but not on the node.
		for (const TileMapLayer *layer : layers) {
			if (layer->is_y_sort_enabled()) {
				warnings.push_back(RTR("A TileMap layer is set as Y-sorted, but Y-sort is not enabled on the TileMap node itself."));
				break;
			}
		}
	} else {
		// Check if Y-sort is enabled on the node, but not on any of the layers.
		bool need_warning = true;
		for (const TileMapLayer *layer : layers) {
			if (layer->is_y_sort_enabled()) {
				need_warning = false;
				break;
			}
		}
		if (need_warning) {
			warnings.push_back(RTR("The TileMap node is set as Y-sorted, but Y-sort is not enabled on any of the TileMap's layers.\nThis may lead to unwanted behaviors, as a layer that is not Y-sorted will be Y-sorted as a whole."));
		}
	}

	// Check if we are in isometric mode without Y-sort enabled.
	if (tile_set.is_valid() && tile_set->get_tile_shape() == TileSet::TILE_SHAPE_ISOMETRIC) {
		bool warn = !is_y_sort_enabled();
		if (!warn) {
			for (const TileMapLayer *layer : layers) {
				if (!layer->is_y_sort_enabled()) {
					warn = true;
					break;
				}
			}
		}

		if (warn) {
			warnings.push_back(RTR("Isometric TileSet will likely not look as intended without Y-sort enabled for the TileMap and all of its layers."));
		}
	}

	return warnings;
}

void TileMap::_bind_methods() {
#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("set_navigation_map", "layer", "map"), &TileMap::set_layer_navigation_map);
	ClassDB::bind_method(D_METHOD("get_navigation_map", "layer"), &TileMap::get_layer_navigation_map);
	ClassDB::bind_method(D_METHOD("force_update", "layer"), &TileMap::force_update, DEFVAL(-1));
#endif // DISABLE_DEPRECATED

	ClassDB::bind_method(D_METHOD("set_tileset", "tileset"), &TileMap::set_tileset);
	ClassDB::bind_method(D_METHOD("get_tileset"), &TileMap::get_tileset);

	ClassDB::bind_method(D_METHOD("set_rendering_quadrant_size", "size"), &TileMap::set_rendering_quadrant_size);
	ClassDB::bind_method(D_METHOD("get_rendering_quadrant_size"), &TileMap::get_rendering_quadrant_size);

	ClassDB::bind_method(D_METHOD("get_layers_count"), &TileMap::get_layers_count);
	ClassDB::bind_method(D_METHOD("add_layer", "to_position"), &TileMap::add_layer);
	ClassDB::bind_method(D_METHOD("move_layer", "layer", "to_position"), &TileMap::move_layer);
	ClassDB::bind_method(D_METHOD("remove_layer", "layer"), &TileMap::remove_layer);
	ClassDB::bind_method(D_METHOD("set_layer_name", "layer", "name"), &TileMap::set_layer_name);
	ClassDB::bind_method(D_METHOD("get_layer_name", "layer"), &TileMap::get_layer_name);
	ClassDB::bind_method(D_METHOD("set_layer_enabled", "layer", "enabled"), &TileMap::set_layer_enabled);
	ClassDB::bind_method(D_METHOD("is_layer_enabled", "layer"), &TileMap::is_layer_enabled);
	ClassDB::bind_method(D_METHOD("set_layer_modulate", "layer", "modulate"), &TileMap::set_layer_modulate);
	ClassDB::bind_method(D_METHOD("get_layer_modulate", "layer"), &TileMap::get_layer_modulate);
	ClassDB::bind_method(D_METHOD("set_layer_y_sort_enabled", "layer", "y_sort_enabled"), &TileMap::set_layer_y_sort_enabled);
	ClassDB::bind_method(D_METHOD("is_layer_y_sort_enabled", "layer"), &TileMap::is_layer_y_sort_enabled);
	ClassDB::bind_method(D_METHOD("set_layer_y_sort_origin", "layer", "y_sort_origin"), &TileMap::set_layer_y_sort_origin);
	ClassDB::bind_method(D_METHOD("get_layer_y_sort_origin", "layer"), &TileMap::get_layer_y_sort_origin);
	ClassDB::bind_method(D_METHOD("set_layer_z_index", "layer", "z_index"), &TileMap::set_layer_z_index);
	ClassDB::bind_method(D_METHOD("get_layer_z_index", "layer"), &TileMap::get_layer_z_index);
	ClassDB::bind_method(D_METHOD("set_layer_navigation_enabled", "layer", "enabled"), &TileMap::set_layer_navigation_enabled);
	ClassDB::bind_method(D_METHOD("is_layer_navigation_enabled", "layer"), &TileMap::is_layer_navigation_enabled);
	ClassDB::bind_method(D_METHOD("set_layer_navigation_map", "layer", "map"), &TileMap::set_layer_navigation_map);
	ClassDB::bind_method(D_METHOD("get_layer_navigation_map", "layer"), &TileMap::get_layer_navigation_map);

	ClassDB::bind_method(D_METHOD("set_collision_animatable", "enabled"), &TileMap::set_collision_animatable);
	ClassDB::bind_method(D_METHOD("is_collision_animatable"), &TileMap::is_collision_animatable);
	ClassDB::bind_method(D_METHOD("set_collision_visibility_mode", "collision_visibility_mode"), &TileMap::set_collision_visibility_mode);
	ClassDB::bind_method(D_METHOD("get_collision_visibility_mode"), &TileMap::get_collision_visibility_mode);

	ClassDB::bind_method(D_METHOD("set_navigation_visibility_mode", "navigation_visibility_mode"), &TileMap::set_navigation_visibility_mode);
	ClassDB::bind_method(D_METHOD("get_navigation_visibility_mode"), &TileMap::get_navigation_visibility_mode);

	ClassDB::bind_method(D_METHOD("set_cell", "layer", "coords", "source_id", "atlas_coords", "alternative_tile"), &TileMap::set_cell, DEFVAL(TileSet::INVALID_SOURCE), DEFVAL(TileSetSource::INVALID_ATLAS_COORDS), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("erase_cell", "layer", "coords"), &TileMap::erase_cell);
	ClassDB::bind_method(D_METHOD("get_cell_source_id", "layer", "coords", "use_proxies"), &TileMap::get_cell_source_id, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_cell_atlas_coords", "layer", "coords", "use_proxies"), &TileMap::get_cell_atlas_coords, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_cell_alternative_tile", "layer", "coords", "use_proxies"), &TileMap::get_cell_alternative_tile, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_cell_tile_data", "layer", "coords", "use_proxies"), &TileMap::get_cell_tile_data, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("get_coords_for_body_rid", "body"), &TileMap::get_coords_for_body_rid);
	ClassDB::bind_method(D_METHOD("get_layer_for_body_rid", "body"), &TileMap::get_layer_for_body_rid);

	ClassDB::bind_method(D_METHOD("get_pattern", "layer", "coords_array", "is_single_layer"), &TileMap::get_pattern);
	ClassDB::bind_method(D_METHOD("map_pattern", "position_in_tilemap", "coords_in_pattern", "pattern"), &TileMap::map_pattern);
	ClassDB::bind_method(D_METHOD("set_pattern", "layer", "position", "pattern"), &TileMap::set_pattern);

	ClassDB::bind_method(D_METHOD("set_cells_terrain_connect", "layer", "cells", "terrain_set", "terrain", "ignore_empty_terrains"), &TileMap::set_cells_terrain_connect, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("set_cells_terrain_path", "layer", "path", "terrain_set", "terrain", "ignore_empty_terrains"), &TileMap::set_cells_terrain_path, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("fix_invalid_tiles"), &TileMap::fix_invalid_tiles);
	ClassDB::bind_method(D_METHOD("clear_layer", "layer"), &TileMap::clear_layer);
	ClassDB::bind_method(D_METHOD("clear"), &TileMap::clear);

	ClassDB::bind_method(D_METHOD("update_internals"), &TileMap::update_internals);
	ClassDB::bind_method(D_METHOD("notify_runtime_tile_data_update", "layer"), &TileMap::notify_runtime_tile_data_update, DEFVAL(-1));

	ClassDB::bind_method(D_METHOD("get_surrounding_cells", "coords"), &TileMap::get_surrounding_cells);

	ClassDB::bind_method(D_METHOD("get_used_cells", "layer"), &TileMap::get_used_cells);
	ClassDB::bind_method(D_METHOD("get_used_cells_by_id", "layer", "source_id", "atlas_coords", "alternative_tile"), &TileMap::get_used_cells_by_id, DEFVAL(TileSet::INVALID_SOURCE), DEFVAL(TileSetSource::INVALID_ATLAS_COORDS), DEFVAL(TileSetSource::INVALID_TILE_ALTERNATIVE));
	ClassDB::bind_method(D_METHOD("get_used_rect"), &TileMap::get_used_rect);

	ClassDB::bind_method(D_METHOD("map_to_local", "map_position"), &TileMap::map_to_local);
	ClassDB::bind_method(D_METHOD("local_to_map", "local_position"), &TileMap::local_to_map);

	ClassDB::bind_method(D_METHOD("get_neighbor_cell", "coords", "neighbor"), &TileMap::get_neighbor_cell);

	GDVIRTUAL_BIND(_use_tile_data_runtime_update, "layer", "coords");
	GDVIRTUAL_BIND(_tile_data_runtime_update, "layer", "coords", "tile_data");

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "tile_set", PROPERTY_HINT_RESOURCE_TYPE, "TileSet"), "set_tileset", "get_tileset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rendering_quadrant_size", PROPERTY_HINT_RANGE, "1,128,1"), "set_rendering_quadrant_size", "get_rendering_quadrant_size");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collision_animatable"), "set_collision_animatable", "is_collision_animatable");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_visibility_mode", PROPERTY_HINT_ENUM, "Default,Force Show,Force Hide"), "set_collision_visibility_mode", "get_collision_visibility_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_visibility_mode", PROPERTY_HINT_ENUM, "Default,Force Show,Force Hide"), "set_navigation_visibility_mode", "get_navigation_visibility_mode");

	ADD_ARRAY("layers", "layer_");

	ADD_PROPERTY_DEFAULT("format", TileMapDataFormat::TILE_MAP_DATA_FORMAT_1);

	ADD_SIGNAL(MethodInfo(CoreStringNames::get_singleton()->changed));

	BIND_ENUM_CONSTANT(VISIBILITY_MODE_DEFAULT);
	BIND_ENUM_CONSTANT(VISIBILITY_MODE_FORCE_HIDE);
	BIND_ENUM_CONSTANT(VISIBILITY_MODE_FORCE_SHOW);
}

TileMap::TileMap() {
	TileMapLayer *new_layer = memnew(TileMapLayer);
	add_child(new_layer, false, INTERNAL_MODE_FRONT);
	new_layer->set_as_tile_map_internal_node(0);
	new_layer->set_name("Layer0");
	new_layer->set_tile_set(tile_set);
	new_layer->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &TileMap::_emit_changed));
	layers.push_back(new_layer);
	default_layer = memnew(TileMapLayer);
}

TileMap::~TileMap() {
	memdelete(default_layer);
}

#undef TILEMAP_CALL_FOR_LAYER
#undef TILEMAP_CALL_FOR_LAYER_V
