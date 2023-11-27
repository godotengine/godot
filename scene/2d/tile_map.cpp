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
#include "scene/resources/world_2d.h"
#include "servers/navigation_server_2d.h"

#ifdef DEBUG_ENABLED
#include "servers/navigation_server_3d.h"
#endif // DEBUG_ENABLED

#ifdef DEBUG_ENABLED
/////////////////////////////// Debug //////////////////////////////////////////
constexpr int TILE_MAP_DEBUG_QUADRANT_SIZE = 16;

Vector2i TileMapLayer::_coords_to_debug_quadrant_coords(const Vector2i &p_coords) const {
	return Vector2i(
			p_coords.x > 0 ? p_coords.x / TILE_MAP_DEBUG_QUADRANT_SIZE : (p_coords.x - (TILE_MAP_DEBUG_QUADRANT_SIZE - 1)) / TILE_MAP_DEBUG_QUADRANT_SIZE,
			p_coords.y > 0 ? p_coords.y / TILE_MAP_DEBUG_QUADRANT_SIZE : (p_coords.y - (TILE_MAP_DEBUG_QUADRANT_SIZE - 1)) / TILE_MAP_DEBUG_QUADRANT_SIZE);
}

void TileMapLayer::_debug_update() {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	RenderingServer *rs = RenderingServer::get_singleton();

	// Check if we should cleanup everything.
	bool forced_cleanup = in_destructor || !enabled || !tile_map_node->is_inside_tree() || !tile_set.is_valid() || !tile_map_node->is_visible_in_tree();

	if (forced_cleanup) {
		for (KeyValue<Vector2i, Ref<DebugQuadrant>> &kv : debug_quadrant_map) {
			// Free the quadrant.
			Ref<DebugQuadrant> &debug_quadrant = kv.value;
			if (debug_quadrant->canvas_item.is_valid()) {
				rs->free(debug_quadrant->canvas_item);
			}
		}
		debug_quadrant_map.clear();
		_debug_was_cleaned_up = true;
		return;
	}

	// Check if anything is dirty, in such a case, redraw debug.
	bool anything_changed = false;
	for (int i = 0; i < DIRTY_FLAGS_MAX; i++) {
		if (dirty.flags[i]) {
			anything_changed = true;
			break;
		}
	}

	// List all debug quadrants to update, creating new ones if needed.
	SelfList<DebugQuadrant>::List dirty_debug_quadrant_list;

	if (_debug_was_cleaned_up || anything_changed) {
		// Update all cells.
		for (KeyValue<Vector2i, CellData> &kv : tile_map) {
			CellData &cell_data = kv.value;
			_debug_quadrants_update_cell(cell_data, dirty_debug_quadrant_list);
		}
	} else {
		// Update dirty cells.
		for (SelfList<CellData> *cell_data_list_element = dirty.cell_list.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
			CellData &cell_data = *cell_data_list_element->self();
			_debug_quadrants_update_cell(cell_data, dirty_debug_quadrant_list);
		}
	}

	// Update those quadrants.
	for (SelfList<DebugQuadrant> *quadrant_list_element = dirty_debug_quadrant_list.first(); quadrant_list_element;) {
		SelfList<DebugQuadrant> *next_quadrant_list_element = quadrant_list_element->next(); // "Hack" to clear the list while iterating.

		DebugQuadrant &debug_quadrant = *quadrant_list_element->self();

		// Check if the quadrant has a tile.
		bool has_a_tile = false;
		RID &ci = debug_quadrant.canvas_item;
		for (SelfList<CellData> *cell_data_list_element = debug_quadrant.cells.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
			CellData &cell_data = *cell_data_list_element->self();
			if (cell_data.cell.source_id != TileSet::INVALID_SOURCE) {
				has_a_tile = true;
				break;
			}
		}

		if (has_a_tile) {
			// Update the quadrant.
			if (ci.is_valid()) {
				rs->canvas_item_clear(ci);
			} else {
				ci = rs->canvas_item_create();
				rs->canvas_item_set_z_index(ci, RS::CANVAS_ITEM_Z_MAX - 1);
				rs->canvas_item_set_parent(ci, tile_map_node->get_canvas_item());
			}

			const Vector2 quadrant_pos = tile_map_node->map_to_local(debug_quadrant.quadrant_coords * TILE_MAP_DEBUG_QUADRANT_SIZE);
			Transform2D xform(0, quadrant_pos);
			rs->canvas_item_set_transform(ci, xform);

			for (SelfList<CellData> *cell_data_list_element = debug_quadrant.cells.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
				CellData &cell_data = *cell_data_list_element->self();
				if (cell_data.cell.source_id != TileSet::INVALID_SOURCE) {
					_rendering_draw_cell_debug(ci, quadrant_pos, cell_data);
					_physics_draw_cell_debug(ci, quadrant_pos, cell_data);
					_navigation_draw_cell_debug(ci, quadrant_pos, cell_data);
					_scenes_draw_cell_debug(ci, quadrant_pos, cell_data);
				}
			}
		} else {
			// Free the quadrant.
			if (ci.is_valid()) {
				rs->free(ci);
			}
			quadrant_list_element->remove_from_list();
			debug_quadrant_map.erase(debug_quadrant.quadrant_coords);
		}

		quadrant_list_element = next_quadrant_list_element;
	}

	dirty_debug_quadrant_list.clear();

	_debug_was_cleaned_up = false;
}

void TileMapLayer::_debug_quadrants_update_cell(CellData &r_cell_data, SelfList<DebugQuadrant>::List &r_dirty_debug_quadrant_list) {
	Vector2i quadrant_coords = _coords_to_debug_quadrant_coords(r_cell_data.coords);

	if (!debug_quadrant_map.has(quadrant_coords)) {
		// Create a new quadrant and add it to the quadrant map.
		Ref<DebugQuadrant> new_quadrant;
		new_quadrant.instantiate();
		new_quadrant->quadrant_coords = quadrant_coords;
		debug_quadrant_map[quadrant_coords] = new_quadrant;
	}

	// Add the cell to its quadrant, if it is not already in there.
	Ref<DebugQuadrant> &debug_quadrant = debug_quadrant_map[quadrant_coords];
	if (!r_cell_data.debug_quadrant_list_element.in_list()) {
		debug_quadrant->cells.add(&r_cell_data.debug_quadrant_list_element);
	}

	// Mark the quadrant as dirty.
	if (!debug_quadrant->dirty_quadrant_list_element.in_list()) {
		r_dirty_debug_quadrant_list.add(&debug_quadrant->dirty_quadrant_list_element);
	}
}
#endif // DEBUG_ENABLED

/////////////////////////////// Rendering //////////////////////////////////////
void TileMapLayer::_rendering_update() {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	RenderingServer *rs = RenderingServer::get_singleton();

	// Check if we should cleanup everything.
	bool forced_cleanup = in_destructor || !enabled || !tile_map_node->is_inside_tree() || !tile_set.is_valid() || !tile_map_node->is_visible_in_tree();

	// ----------- Layer level processing -----------
	if (forced_cleanup) {
		// Cleanup.
		if (canvas_item.is_valid()) {
			rs->free(canvas_item);
			canvas_item = RID();
		}
	} else {
		// Create/Update the layer's CanvasItem.
		if (!canvas_item.is_valid()) {
			RID ci = rs->canvas_item_create();
			rs->canvas_item_set_parent(ci, tile_map_node->get_canvas_item());
			canvas_item = ci;
		}
		RID &ci = canvas_item;
		rs->canvas_item_set_draw_index(ci, layer_index_in_tile_map_node - (int64_t)0x80000000);
		rs->canvas_item_set_sort_children_by_y(ci, y_sort_enabled);
		rs->canvas_item_set_use_parent_material(ci, tile_map_node->get_use_parent_material() || tile_map_node->get_material().is_valid());
		rs->canvas_item_set_z_index(ci, z_index);
		rs->canvas_item_set_default_texture_filter(ci, RS::CanvasItemTextureFilter(tile_map_node->get_texture_filter_in_tree()));
		rs->canvas_item_set_default_texture_repeat(ci, RS::CanvasItemTextureRepeat(tile_map_node->get_texture_repeat_in_tree()));
		rs->canvas_item_set_light_mask(ci, tile_map_node->get_light_mask());

		// Modulate the layer.
		Color layer_modulate = modulate;
		int selected_layer = tile_map_node->get_selected_layer();
		if (selected_layer >= 0 && layer_index_in_tile_map_node != selected_layer) {
			int z_selected = tile_map_node->get_layer_z_index(selected_layer);
			if (z_index < z_selected || (z_index == z_selected && layer_index_in_tile_map_node < selected_layer)) {
				layer_modulate = layer_modulate.darkened(0.5);
			} else if (z_index > z_selected || (z_index == z_selected && layer_index_in_tile_map_node > selected_layer)) {
				layer_modulate = layer_modulate.darkened(0.5);
				layer_modulate.a *= 0.3;
			}
		}
		rs->canvas_item_set_modulate(ci, layer_modulate);
	}

	// ----------- Quadrants processing -----------

	// List all rendering quadrants to update, creating new ones if needed.
	SelfList<RenderingQuadrant>::List dirty_rendering_quadrant_list;

	// Check if anything changed that might change the quadrant shape.
	// If so, recreate everything.
	bool quandrant_shape_changed = dirty.flags[DIRTY_FLAGS_TILE_MAP_QUADRANT_SIZE] ||
			(tile_map_node->is_y_sort_enabled() && y_sort_enabled && (dirty.flags[DIRTY_FLAGS_LAYER_Y_SORT_ENABLED] || dirty.flags[DIRTY_FLAGS_LAYER_Y_SORT_ORIGIN] || dirty.flags[DIRTY_FLAGS_TILE_MAP_Y_SORT_ENABLED] || dirty.flags[DIRTY_FLAGS_TILE_MAP_LOCAL_XFORM] || dirty.flags[DIRTY_FLAGS_TILE_MAP_TILE_SET]));

	// Free all quadrants.
	if (forced_cleanup || quandrant_shape_changed) {
		for (const KeyValue<Vector2i, Ref<RenderingQuadrant>> &kv : rendering_quadrant_map) {
			for (int i = 0; i < kv.value->canvas_items.size(); i++) {
				const RID &ci = kv.value->canvas_items[i];
				if (ci.is_valid()) {
					rs->free(ci);
				}
			}
			kv.value->cells.clear();
		}
		rendering_quadrant_map.clear();
		_rendering_was_cleaned_up = true;
	}

	if (!forced_cleanup) {
		// List all quadrants to update, recreating them if needed.
		if (dirty.flags[DIRTY_FLAGS_TILE_MAP_TILE_SET] || _rendering_was_cleaned_up) {
			// Update all cells.
			for (KeyValue<Vector2i, CellData> &kv : tile_map) {
				CellData &cell_data = kv.value;
				_rendering_quadrants_update_cell(cell_data, dirty_rendering_quadrant_list);
			}
		} else {
			// Update dirty cells.
			for (SelfList<CellData> *cell_data_list_element = dirty.cell_list.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
				CellData &cell_data = *cell_data_list_element->self();
				_rendering_quadrants_update_cell(cell_data, dirty_rendering_quadrant_list);
			}
		}

		// Update all dirty quadrants.
		for (SelfList<RenderingQuadrant> *quadrant_list_element = dirty_rendering_quadrant_list.first(); quadrant_list_element;) {
			SelfList<RenderingQuadrant> *next_quadrant_list_element = quadrant_list_element->next(); // "Hack" to clear the list while iterating.

			const Ref<RenderingQuadrant> &rendering_quadrant = quadrant_list_element->self();

			// Check if the quadrant has a tile.
			bool has_a_tile = false;
			for (SelfList<CellData> *cell_data_list_element = rendering_quadrant->cells.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
				CellData &cell_data = *cell_data_list_element->self();
				if (cell_data.cell.source_id != TileSet::INVALID_SOURCE) {
					has_a_tile = true;
					break;
				}
			}

			if (has_a_tile) {
				// Process the quadrant.

				// First, clear the quadrant's canvas items.
				for (RID &ci : rendering_quadrant->canvas_items) {
					rs->free(ci);
				}
				rendering_quadrant->canvas_items.clear();

				// Sort the quadrant cells.
				if (tile_map_node->is_y_sort_enabled() && is_y_sort_enabled()) {
					// For compatibility reasons, we use another comparator for Y-sorted layers.
					rendering_quadrant->cells.sort_custom<CellDataYSortedComparator>();
				} else {
					rendering_quadrant->cells.sort();
				}

				// Those allow to group cell per material or z-index.
				Ref<Material> prev_material;
				int prev_z_index = 0;
				RID prev_ci;

				for (SelfList<CellData> *cell_data_quadrant_list_element = rendering_quadrant->cells.first(); cell_data_quadrant_list_element; cell_data_quadrant_list_element = cell_data_quadrant_list_element->next()) {
					CellData &cell_data = *cell_data_quadrant_list_element->self();

					TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(*tile_set->get_source(cell_data.cell.source_id));

					// Get the tile data.
					const TileData *tile_data;
					if (cell_data.runtime_tile_data_cache) {
						tile_data = cell_data.runtime_tile_data_cache;
					} else {
						tile_data = atlas_source->get_tile_data(cell_data.cell.get_atlas_coords(), cell_data.cell.alternative_tile);
					}

					Ref<Material> mat = tile_data->get_material();
					int tile_z_index = tile_data->get_z_index();

					// Quandrant pos.

					// --- CanvasItems ---
					RID ci;

					// Check if the material or the z_index changed.
					if (prev_ci == RID() || prev_material != mat || prev_z_index != tile_z_index) {
						// If so, create a new CanvasItem.
						ci = rs->canvas_item_create();
						if (mat.is_valid()) {
							rs->canvas_item_set_material(ci, mat->get_rid());
						}
						rs->canvas_item_set_parent(ci, canvas_item);
						rs->canvas_item_set_use_parent_material(ci, tile_map_node->get_use_parent_material() || tile_map_node->get_material().is_valid());

						Transform2D xform(0, rendering_quadrant->canvas_items_position);
						rs->canvas_item_set_transform(ci, xform);

						rs->canvas_item_set_light_mask(ci, tile_map_node->get_light_mask());
						rs->canvas_item_set_z_as_relative_to_parent(ci, true);
						rs->canvas_item_set_z_index(ci, tile_z_index);

						rs->canvas_item_set_default_texture_filter(ci, RS::CanvasItemTextureFilter(tile_map_node->get_texture_filter_in_tree()));
						rs->canvas_item_set_default_texture_repeat(ci, RS::CanvasItemTextureRepeat(tile_map_node->get_texture_repeat_in_tree()));

						rendering_quadrant->canvas_items.push_back(ci);

						prev_ci = ci;
						prev_material = mat;
						prev_z_index = tile_z_index;

					} else {
						// Keep the same canvas_item to draw on.
						ci = prev_ci;
					}

					const Vector2 local_tile_pos = tile_map_node->map_to_local(cell_data.coords);

					// Random animation offset.
					real_t random_animation_offset = 0.0;
					if (atlas_source->get_tile_animation_mode(cell_data.cell.get_atlas_coords()) != TileSetAtlasSource::TILE_ANIMATION_MODE_DEFAULT) {
						Array to_hash;
						to_hash.push_back(local_tile_pos);
						to_hash.push_back(get_instance_id()); // Use instance id as a random hash
						random_animation_offset = RandomPCG(to_hash.hash()).randf();
					}

					// Drawing the tile in the canvas item.
					tile_map_node->draw_tile(ci, local_tile_pos - rendering_quadrant->canvas_items_position, tile_set, cell_data.cell.source_id, cell_data.cell.get_atlas_coords(), cell_data.cell.alternative_tile, -1, tile_map_node->get_self_modulate(), tile_data, random_animation_offset);
				}
			} else {
				// Free the quadrant.
				for (int i = 0; i < rendering_quadrant->canvas_items.size(); i++) {
					const RID &ci = rendering_quadrant->canvas_items[i];
					if (ci.is_valid()) {
						rs->free(ci);
					}
				}
				rendering_quadrant->cells.clear();
				rendering_quadrant_map.erase(rendering_quadrant->quadrant_coords);
			}

			quadrant_list_element = next_quadrant_list_element;
		}

		dirty_rendering_quadrant_list.clear();

		// Reset the drawing indices.
		{
			int index = -(int64_t)0x80000000; // Always must be drawn below children.

			// Sort the quadrants coords per local coordinates.
			RBMap<Vector2, Ref<RenderingQuadrant>, RenderingQuadrant::CoordsWorldComparator> local_to_map;
			for (KeyValue<Vector2i, Ref<RenderingQuadrant>> &kv : rendering_quadrant_map) {
				Ref<RenderingQuadrant> &rendering_quadrant = kv.value;
				local_to_map[tile_map_node->map_to_local(rendering_quadrant->quadrant_coords)] = rendering_quadrant;
			}

			// Sort the quadrants.
			for (const KeyValue<Vector2, Ref<RenderingQuadrant>> &E : local_to_map) {
				for (const RID &ci : E.value->canvas_items) {
					RS::get_singleton()->canvas_item_set_draw_index(ci, index++);
				}
			}
		}

		// Updates on TileMap changes.
		if (dirty.flags[DIRTY_FLAGS_TILE_MAP_LIGHT_MASK] ||
				dirty.flags[DIRTY_FLAGS_TILE_MAP_USE_PARENT_MATERIAL] ||
				dirty.flags[DIRTY_FLAGS_TILE_MAP_MATERIAL] ||
				dirty.flags[DIRTY_FLAGS_TILE_MAP_TEXTURE_FILTER] ||
				dirty.flags[DIRTY_FLAGS_TILE_MAP_TEXTURE_REPEAT]) {
			for (KeyValue<Vector2i, Ref<RenderingQuadrant>> &kv : rendering_quadrant_map) {
				Ref<RenderingQuadrant> &rendering_quadrant = kv.value;
				for (const RID &ci : rendering_quadrant->canvas_items) {
					rs->canvas_item_set_light_mask(ci, tile_map_node->get_light_mask());
					rs->canvas_item_set_use_parent_material(ci, tile_map_node->get_use_parent_material() || tile_map_node->get_material().is_valid());
					rs->canvas_item_set_default_texture_filter(ci, RS::CanvasItemTextureFilter(tile_map_node->get_texture_filter_in_tree()));
					rs->canvas_item_set_default_texture_repeat(ci, RS::CanvasItemTextureRepeat(tile_map_node->get_texture_repeat_in_tree()));
				}
			}
		}
	}

	// ----------- Occluders processing -----------
	if (forced_cleanup) {
		// Clean everything.
		for (KeyValue<Vector2i, CellData> &kv : tile_map) {
			_rendering_occluders_clear_cell(kv.value);
		}
	} else {
		if (_rendering_was_cleaned_up || dirty.flags[DIRTY_FLAGS_TILE_MAP_TILE_SET]) {
			// Update all cells.
			for (KeyValue<Vector2i, CellData> &kv : tile_map) {
				_rendering_occluders_update_cell(kv.value);
			}
		} else {
			// Update dirty cells.
			for (SelfList<CellData> *cell_data_list_element = dirty.cell_list.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
				CellData &cell_data = *cell_data_list_element->self();
				_rendering_occluders_update_cell(cell_data);
			}
		}

		// Updates on TileMap changes.
		if (dirty.flags[DIRTY_FLAGS_TILE_MAP_IN_CANVAS] || dirty.flags[DIRTY_FLAGS_TILE_MAP_VISIBILITY]) {
			for (KeyValue<Vector2i, CellData> &kv : tile_map) {
				CellData &cell_data = kv.value;
				for (const RID &occluder : cell_data.occluders) {
					Transform2D xform(0, tile_map_node->map_to_local(kv.key));
					rs->canvas_light_occluder_attach_to_canvas(occluder, tile_map_node->get_canvas());
					rs->canvas_light_occluder_set_transform(occluder, tile_map_node->get_global_transform() * xform);
				}
			}
		}
	}

	// -----------
	// Mark the rendering state as up to date.
	_rendering_was_cleaned_up = forced_cleanup;
}

void TileMapLayer::_rendering_quadrants_update_cell(CellData &r_cell_data, SelfList<RenderingQuadrant>::List &r_dirty_rendering_quadrant_list) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();

	// Check if the cell is valid and retrieve its y_sort_origin.
	bool is_valid = false;
	int tile_y_sort_origin = 0;
	TileSetSource *source;
	if (tile_set->has_source(r_cell_data.cell.source_id)) {
		source = *tile_set->get_source(r_cell_data.cell.source_id);
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source && atlas_source->has_tile(r_cell_data.cell.get_atlas_coords()) && atlas_source->has_alternative_tile(r_cell_data.cell.get_atlas_coords(), r_cell_data.cell.alternative_tile)) {
			is_valid = true;
			const TileData *tile_data;
			if (r_cell_data.runtime_tile_data_cache) {
				tile_data = r_cell_data.runtime_tile_data_cache;
			} else {
				tile_data = atlas_source->get_tile_data(r_cell_data.cell.get_atlas_coords(), r_cell_data.cell.alternative_tile);
			}
			tile_y_sort_origin = tile_data->get_y_sort_origin();
		}
	}

	if (is_valid) {
		// Get the quadrant coords.
		Vector2 canvas_items_position;
		Vector2i quadrant_coords;
		if (tile_map_node->is_y_sort_enabled() && is_y_sort_enabled()) {
			canvas_items_position = Vector2(0, tile_map_node->map_to_local(r_cell_data.coords).y + tile_y_sort_origin + y_sort_origin);
			quadrant_coords = canvas_items_position * 100;
		} else {
			int quad_size = tile_map_node->get_rendering_quadrant_size();
			const Vector2i &coords = r_cell_data.coords;

			// Rounding down, instead of simply rounding towards zero (truncating).
			quadrant_coords = Vector2i(
					coords.x > 0 ? coords.x / quad_size : (coords.x - (quad_size - 1)) / quad_size,
					coords.y > 0 ? coords.y / quad_size : (coords.y - (quad_size - 1)) / quad_size);
			canvas_items_position = quad_size * quadrant_coords;
		}

		Ref<RenderingQuadrant> rendering_quadrant;
		if (rendering_quadrant_map.has(quadrant_coords)) {
			// Reuse existing rendering quadrant.
			rendering_quadrant = rendering_quadrant_map[quadrant_coords];
		} else {
			// Create a new rendering quadrant.
			rendering_quadrant.instantiate();
			rendering_quadrant->quadrant_coords = quadrant_coords;
			rendering_quadrant->canvas_items_position = canvas_items_position;
			rendering_quadrant_map[quadrant_coords] = rendering_quadrant;
		}

		// Mark the old quadrant as dirty (if it exists).
		if (r_cell_data.rendering_quadrant.is_valid()) {
			if (!r_cell_data.rendering_quadrant->dirty_quadrant_list_element.in_list()) {
				r_dirty_rendering_quadrant_list.add(&r_cell_data.rendering_quadrant->dirty_quadrant_list_element);
			}
		}

		// Remove the cell from that quadrant.
		if (r_cell_data.rendering_quadrant_list_element.in_list()) {
			r_cell_data.rendering_quadrant_list_element.remove_from_list();
		}

		// Add the cell to its new quadrant.
		r_cell_data.rendering_quadrant = rendering_quadrant;
		r_cell_data.rendering_quadrant->cells.add(&r_cell_data.rendering_quadrant_list_element);

		// Add the new quadrant to the dirty quadrant list.
		if (!rendering_quadrant->dirty_quadrant_list_element.in_list()) {
			r_dirty_rendering_quadrant_list.add(&rendering_quadrant->dirty_quadrant_list_element);
		}
	} else {
		Ref<RenderingQuadrant> rendering_quadrant = r_cell_data.rendering_quadrant;

		// Remove the cell from its quadrant.
		r_cell_data.rendering_quadrant = Ref<RenderingQuadrant>();
		if (r_cell_data.rendering_quadrant_list_element.in_list()) {
			rendering_quadrant->cells.remove(&r_cell_data.rendering_quadrant_list_element);
		}

		if (rendering_quadrant.is_valid()) {
			// Add the quadrant to the dirty quadrant list.
			if (!rendering_quadrant->dirty_quadrant_list_element.in_list()) {
				r_dirty_rendering_quadrant_list.add(&rendering_quadrant->dirty_quadrant_list_element);
			}
		}
	}
}

void TileMapLayer::_rendering_occluders_clear_cell(CellData &r_cell_data) {
	RenderingServer *rs = RenderingServer::get_singleton();

	// Free the occluders.
	for (const RID &rid : r_cell_data.occluders) {
		rs->free(rid);
	}
	r_cell_data.occluders.clear();
}

void TileMapLayer::_rendering_occluders_update_cell(CellData &r_cell_data) {
	bool node_visible = tile_map_node->is_visible_in_tree();
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	RenderingServer *rs = RenderingServer::get_singleton();

	TileSetSource *source;
	if (tile_set->has_source(r_cell_data.cell.source_id)) {
		source = *tile_set->get_source(r_cell_data.cell.source_id);

		if (source->has_tile(r_cell_data.cell.get_atlas_coords()) && source->has_alternative_tile(r_cell_data.cell.get_atlas_coords(), r_cell_data.cell.alternative_tile)) {
			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				// Get the tile data.
				const TileData *tile_data;
				if (r_cell_data.runtime_tile_data_cache) {
					tile_data = r_cell_data.runtime_tile_data_cache;
				} else {
					tile_data = atlas_source->get_tile_data(r_cell_data.cell.get_atlas_coords(), r_cell_data.cell.alternative_tile);
				}

				// Update/create occluders.
				for (int i = 0; i < tile_set->get_occlusion_layers_count(); i++) {
					Transform2D xform;
					xform.set_origin(tile_map_node->map_to_local(r_cell_data.coords));
					if (tile_data->get_occluder(i).is_valid()) {
						RID occluder_id = rs->canvas_light_occluder_create();
						rs->canvas_light_occluder_set_enabled(occluder_id, node_visible);
						rs->canvas_light_occluder_set_transform(occluder_id, tile_map_node->get_global_transform() * xform);
						rs->canvas_light_occluder_set_polygon(occluder_id, tile_map_node->get_transformed_polygon(Ref<Resource>(tile_data->get_occluder(i)), r_cell_data.cell.alternative_tile)->get_rid());
						rs->canvas_light_occluder_attach_to_canvas(occluder_id, tile_map_node->get_canvas());
						rs->canvas_light_occluder_set_light_mask(occluder_id, tile_set->get_occlusion_layer_light_mask(i));
						r_cell_data.occluders.push_back(occluder_id);
					}
				}

				return;
			}
		}
	}

	// If we did not return earlier, clear the cell.
	_rendering_occluders_clear_cell(r_cell_data);
}

#ifdef DEBUG_ENABLED
void TileMapLayer::_rendering_draw_cell_debug(const RID &p_canvas_item, const Vector2i &p_quadrant_pos, const CellData &r_cell_data) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	// Draw a placeholder for tiles needing one.
	RenderingServer *rs = RenderingServer::get_singleton();
	const TileMapCell &c = r_cell_data.cell;

	TileSetSource *source;
	if (tile_set->has_source(c.source_id)) {
		source = *tile_set->get_source(c.source_id);

		if (source->has_tile(c.get_atlas_coords()) && source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				Vector2i grid_size = atlas_source->get_atlas_grid_size();
				if (!atlas_source->get_runtime_texture().is_valid() || c.get_atlas_coords().x >= grid_size.x || c.get_atlas_coords().y >= grid_size.y) {
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
					Transform2D cell_to_quadrant;
					cell_to_quadrant.set_origin(tile_map_node->map_to_local(r_cell_data.coords) - p_quadrant_pos);
					rs->canvas_item_add_set_transform(p_canvas_item, cell_to_quadrant);
					rs->canvas_item_add_circle(p_canvas_item, Vector2(), MIN(tile_set->get_tile_size().x, tile_set->get_tile_size().y) / 4.0, color);
				}
			}
		}
	}
}
#endif // DEBUG_ENABLED

/////////////////////////////// Physics //////////////////////////////////////

void TileMapLayer::_physics_update() {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();

	// Check if we should cleanup everything.
	bool forced_cleanup = in_destructor || !enabled || !tile_map_node->is_inside_tree() || !tile_set.is_valid();
	if (forced_cleanup) {
		// Clean everything.
		for (KeyValue<Vector2i, CellData> &kv : tile_map) {
			_physics_clear_cell(kv.value);
		}
	} else {
		if (_physics_was_cleaned_up || dirty.flags[DIRTY_FLAGS_TILE_MAP_TILE_SET] || dirty.flags[DIRTY_FLAGS_TILE_MAP_COLLISION_ANIMATABLE]) {
			// Update all cells.
			for (KeyValue<Vector2i, CellData> &kv : tile_map) {
				_physics_update_cell(kv.value);
			}
		} else {
			// Update dirty cells.
			for (SelfList<CellData> *cell_data_list_element = dirty.cell_list.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
				CellData &cell_data = *cell_data_list_element->self();
				_physics_update_cell(cell_data);
			}
		}
	}

	// -----------
	// Mark the physics state as up to date.
	_physics_was_cleaned_up = forced_cleanup;
}

void TileMapLayer::_physics_notify_tilemap_change(TileMapLayer::DirtyFlags p_what) {
	Transform2D gl_transform = tile_map_node->get_global_transform();
	PhysicsServer2D *ps = PhysicsServer2D::get_singleton();

	bool in_editor = false;
#ifdef TOOLS_ENABLED
	in_editor = Engine::get_singleton()->is_editor_hint();
#endif

	if (p_what == DIRTY_FLAGS_TILE_MAP_XFORM) {
		if (tile_map_node->is_inside_tree() && (!tile_map_node->is_collision_animatable() || in_editor)) {
			// Move the collisison shapes along with the TileMap.
			for (KeyValue<Vector2i, CellData> &kv : tile_map) {
				const CellData &cell_data = kv.value;

				for (RID body : cell_data.bodies) {
					if (body.is_valid()) {
						Transform2D xform(0, tile_map_node->map_to_local(bodies_coords[body]));
						xform = gl_transform * xform;
						ps->body_set_state(body, PhysicsServer2D::BODY_STATE_TRANSFORM, xform);
					}
				}
			}
		}
	} else if (p_what == DIRTY_FLAGS_TILE_MAP_LOCAL_XFORM) {
		// With collisions animatable, move the collisison shapes along with the TileMap only on local xform change (they are synchornized on physics tick instead).
		if (tile_map_node->is_inside_tree() && tile_map_node->is_collision_animatable() && !in_editor) {
			for (KeyValue<Vector2i, CellData> &kv : tile_map) {
				const CellData &cell_data = kv.value;

				for (RID body : cell_data.bodies) {
					if (body.is_valid()) {
						Transform2D xform(0, tile_map_node->map_to_local(bodies_coords[body]));
						xform = gl_transform * xform;
						ps->body_set_state(body, PhysicsServer2D::BODY_STATE_TRANSFORM, xform);
					}
				}
			}
		}
	} else if (p_what == DIRTY_FLAGS_TILE_MAP_IN_TREE) {
		// Changes in the tree may cause the space to change (e.g. when reparenting to a SubViewport).
		if (tile_map_node->is_inside_tree()) {
			RID space = tile_map_node->get_world_2d()->get_space();

			for (KeyValue<Vector2i, CellData> &kv : tile_map) {
				const CellData &cell_data = kv.value;

				for (RID body : cell_data.bodies) {
					if (body.is_valid()) {
						ps->body_set_space(body, space);
					}
				}
			}
		}
	}
}

void TileMapLayer::_physics_clear_cell(CellData &r_cell_data) {
	PhysicsServer2D *ps = PhysicsServer2D::get_singleton();

	// Clear bodies.
	for (RID body : r_cell_data.bodies) {
		if (body.is_valid()) {
			bodies_coords.erase(body);
			ps->free(body);
		}
	}
	r_cell_data.bodies.clear();
}

void TileMapLayer::_physics_update_cell(CellData &r_cell_data) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	Transform2D gl_transform = tile_map_node->get_global_transform();
	RID space = tile_map_node->get_world_2d()->get_space();
	PhysicsServer2D *ps = PhysicsServer2D::get_singleton();

	// Recreate bodies and shapes.
	TileMapCell &c = r_cell_data.cell;

	TileSetSource *source;
	if (tile_set->has_source(c.source_id)) {
		source = *tile_set->get_source(c.source_id);

		if (source->has_tile(c.get_atlas_coords()) && source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				const TileData *tile_data;
				if (r_cell_data.runtime_tile_data_cache) {
					tile_data = r_cell_data.runtime_tile_data_cache;
				} else {
					tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);
				}

				// Free unused bodies then resize the bodies array.
				for (unsigned int i = tile_set->get_physics_layers_count(); i < r_cell_data.bodies.size(); i++) {
					RID body = r_cell_data.bodies[i];
					if (body.is_valid()) {
						bodies_coords.erase(body);
						ps->free(body);
					}
				}
				r_cell_data.bodies.resize(tile_set->get_physics_layers_count());

				for (int tile_set_physics_layer = 0; tile_set_physics_layer < tile_set->get_physics_layers_count(); tile_set_physics_layer++) {
					Ref<PhysicsMaterial> physics_material = tile_set->get_physics_layer_physics_material(tile_set_physics_layer);
					uint32_t physics_layer = tile_set->get_physics_layer_collision_layer(tile_set_physics_layer);
					uint32_t physics_mask = tile_set->get_physics_layer_collision_mask(tile_set_physics_layer);

					RID body = r_cell_data.bodies[tile_set_physics_layer];
					if (tile_data->get_collision_polygons_count(tile_set_physics_layer) == 0) {
						// No body needed, free it if it exists.
						if (body.is_valid()) {
							bodies_coords.erase(body);
							ps->free(body);
						}
						body = RID();
					} else {
						// Create or update the body.
						if (!body.is_valid()) {
							body = ps->body_create();
						}
						bodies_coords[body] = r_cell_data.coords;
						ps->body_set_mode(body, tile_map_node->is_collision_animatable() ? PhysicsServer2D::BODY_MODE_KINEMATIC : PhysicsServer2D::BODY_MODE_STATIC);
						ps->body_set_space(body, space);

						Transform2D xform;
						xform.set_origin(tile_map_node->map_to_local(r_cell_data.coords));
						xform = gl_transform * xform;
						ps->body_set_state(body, PhysicsServer2D::BODY_STATE_TRANSFORM, xform);

						ps->body_attach_object_instance_id(body, tile_map_node->get_instance_id());
						ps->body_set_collision_layer(body, physics_layer);
						ps->body_set_collision_mask(body, physics_mask);
						ps->body_set_pickable(body, false);
						ps->body_set_state(body, PhysicsServer2D::BODY_STATE_LINEAR_VELOCITY, tile_data->get_constant_linear_velocity(tile_set_physics_layer));
						ps->body_set_state(body, PhysicsServer2D::BODY_STATE_ANGULAR_VELOCITY, tile_data->get_constant_angular_velocity(tile_set_physics_layer));

						if (!physics_material.is_valid()) {
							ps->body_set_param(body, PhysicsServer2D::BODY_PARAM_BOUNCE, 0);
							ps->body_set_param(body, PhysicsServer2D::BODY_PARAM_FRICTION, 1);
						} else {
							ps->body_set_param(body, PhysicsServer2D::BODY_PARAM_BOUNCE, physics_material->computed_bounce());
							ps->body_set_param(body, PhysicsServer2D::BODY_PARAM_FRICTION, physics_material->computed_friction());
						}

						// Clear body's shape if needed.
						ps->body_clear_shapes(body);

						// Add the shapes to the body.
						int body_shape_index = 0;
						for (int polygon_index = 0; polygon_index < tile_data->get_collision_polygons_count(tile_set_physics_layer); polygon_index++) {
							// Iterate over the polygons.
							bool one_way_collision = tile_data->is_collision_polygon_one_way(tile_set_physics_layer, polygon_index);
							float one_way_collision_margin = tile_data->get_collision_polygon_one_way_margin(tile_set_physics_layer, polygon_index);
							int shapes_count = tile_data->get_collision_polygon_shapes_count(tile_set_physics_layer, polygon_index);
							for (int shape_index = 0; shape_index < shapes_count; shape_index++) {
								// Add decomposed convex shapes.
								Ref<ConvexPolygonShape2D> shape = tile_data->get_collision_polygon_shape(tile_set_physics_layer, polygon_index, shape_index);
								shape = tile_map_node->get_transformed_polygon(Ref<Resource>(shape), c.alternative_tile);
								ps->body_add_shape(body, shape->get_rid());
								ps->body_set_shape_as_one_way_collision(body, body_shape_index, one_way_collision, one_way_collision_margin);

								body_shape_index++;
							}
						}
					}

					// Set the body again.
					r_cell_data.bodies[tile_set_physics_layer] = body;
				}

				return;
			}
		}
	}

	// If we did not return earlier, clear the cell.
	_physics_clear_cell(r_cell_data);
}

#ifdef DEBUG_ENABLED
void TileMapLayer::_physics_draw_cell_debug(const RID &p_canvas_item, const Vector2i &p_quadrant_pos, const CellData &r_cell_data) {
	// Draw the debug collision shapes.
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!tile_map_node->get_tree()) {
		return;
	}

	bool show_collision = false;
	switch (tile_map_node->get_collision_visibility_mode()) {
		case TileMap::VISIBILITY_MODE_DEFAULT:
			show_collision = !Engine::get_singleton()->is_editor_hint() && tile_map_node->get_tree()->is_debugging_collisions_hint();
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
	PhysicsServer2D *ps = PhysicsServer2D::get_singleton();

	Color debug_collision_color = tile_map_node->get_tree()->get_debug_collisions_color();
	Vector<Color> color;
	color.push_back(debug_collision_color);

	Transform2D quadrant_to_local(0, p_quadrant_pos);
	Transform2D global_to_quadrant = (tile_map_node->get_global_transform() * quadrant_to_local).affine_inverse();

	for (RID body : r_cell_data.bodies) {
		if (body.is_valid()) {
			Transform2D body_to_quadrant = global_to_quadrant * Transform2D(ps->body_get_state(body, PhysicsServer2D::BODY_STATE_TRANSFORM));
			rs->canvas_item_add_set_transform(p_canvas_item, body_to_quadrant);
			for (int shape_index = 0; shape_index < ps->body_get_shape_count(body); shape_index++) {
				const RID &shape = ps->body_get_shape(body, shape_index);
				const PhysicsServer2D::ShapeType &type = ps->shape_get_type(shape);
				if (type == PhysicsServer2D::SHAPE_CONVEX_POLYGON) {
					rs->canvas_item_add_polygon(p_canvas_item, ps->shape_get_data(shape), color);
				} else {
					WARN_PRINT("Wrong shape type for a tile, should be SHAPE_CONVEX_POLYGON.");
				}
			}
			rs->canvas_item_add_set_transform(p_canvas_item, Transform2D());
		}
	}
};
#endif // DEBUG_ENABLED

/////////////////////////////// Navigation //////////////////////////////////////

void TileMapLayer::_navigation_update() {
	ERR_FAIL_NULL(NavigationServer2D::get_singleton());
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	NavigationServer2D *ns = NavigationServer2D::get_singleton();

	// Check if we should cleanup everything.
	bool forced_cleanup = in_destructor || !enabled || !navigation_enabled || !tile_map_node->is_inside_tree() || !tile_set.is_valid();

	// ----------- Layer level processing -----------
	if (forced_cleanup) {
		if (navigation_map.is_valid() && !uses_world_navigation_map) {
			ns->free(navigation_map);
			navigation_map = RID();
		}
	} else {
		// Update navigation maps.
		if (!navigation_map.is_valid()) {
			if (layer_index_in_tile_map_node == 0) {
				// Use the default World2D navigation map for the first layer when empty.
				navigation_map = tile_map_node->get_world_2d()->get_navigation_map();
				uses_world_navigation_map = true;
			} else {
				RID new_layer_map = ns->map_create();
				// Set the default NavigationPolygon cell_size on the new map as a mismatch causes an error.
				ns->map_set_cell_size(new_layer_map, 1.0);
				ns->map_set_active(new_layer_map, true);
				navigation_map = new_layer_map;
				uses_world_navigation_map = false;
			}
		}
	}

	// ----------- Navigation regions processing -----------
	if (forced_cleanup) {
		// Clean everything.
		for (KeyValue<Vector2i, CellData> &kv : tile_map) {
			_navigation_clear_cell(kv.value);
		}
	} else {
		if (_navigation_was_cleaned_up || dirty.flags[DIRTY_FLAGS_TILE_MAP_TILE_SET]) {
			// Update all cells.
			for (KeyValue<Vector2i, CellData> &kv : tile_map) {
				_navigation_update_cell(kv.value);
			}
		} else {
			// Update dirty cells.
			for (SelfList<CellData> *cell_data_list_element = dirty.cell_list.first(); cell_data_list_element; cell_data_list_element = cell_data_list_element->next()) {
				CellData &cell_data = *cell_data_list_element->self();
				_navigation_update_cell(cell_data);
			}
		}

		if (dirty.flags[DIRTY_FLAGS_TILE_MAP_XFORM]) {
			Transform2D tilemap_xform = tile_map_node->get_global_transform();
			for (KeyValue<Vector2i, CellData> &kv : tile_map) {
				const CellData &cell_data = kv.value;
				// Update navigation regions transform.
				for (const RID &region : cell_data.navigation_regions) {
					if (!region.is_valid()) {
						continue;
					}
					Transform2D tile_transform;
					tile_transform.set_origin(tile_map_node->map_to_local(kv.key));
					NavigationServer2D::get_singleton()->region_set_transform(region, tilemap_xform * tile_transform);
				}
			}
		}
	}

	// -----------
	// Mark the navigation state as up to date.
	_navigation_was_cleaned_up = forced_cleanup;
}

void TileMapLayer::_navigation_clear_cell(CellData &r_cell_data) {
	NavigationServer2D *ns = NavigationServer2D::get_singleton();
	// Clear navigation shapes.
	for (unsigned int i = 0; i < r_cell_data.navigation_regions.size(); i++) {
		const RID &region = r_cell_data.navigation_regions[i];
		if (region.is_valid()) {
			ns->region_set_map(region, RID());
			ns->free(region);
		}
	}
	r_cell_data.navigation_regions.clear();
}

void TileMapLayer::_navigation_update_cell(CellData &r_cell_data) {
	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
	NavigationServer2D *ns = NavigationServer2D::get_singleton();
	Transform2D tilemap_xform = tile_map_node->get_global_transform();

	// Get the navigation polygons and create regions.
	TileMapCell &c = r_cell_data.cell;

	TileSetSource *source;
	if (tile_set->has_source(c.source_id)) {
		source = *tile_set->get_source(c.source_id);

		if (source->has_tile(c.get_atlas_coords()) && source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				const TileData *tile_data;
				if (r_cell_data.runtime_tile_data_cache) {
					tile_data = r_cell_data.runtime_tile_data_cache;
				} else {
					tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);
				}

				// Free unused regions then resize the regions array.
				for (unsigned int i = tile_set->get_navigation_layers_count(); i < r_cell_data.navigation_regions.size(); i++) {
					RID &region = r_cell_data.navigation_regions[i];
					if (region.is_valid()) {
						ns->region_set_map(region, RID());
						ns->free(region);
						region = RID();
					}
				}
				r_cell_data.navigation_regions.resize(tile_set->get_navigation_layers_count());

				// Create, update or clear regions.
				for (unsigned int navigation_layer_index = 0; navigation_layer_index < r_cell_data.navigation_regions.size(); navigation_layer_index++) {
					Ref<NavigationPolygon> navigation_polygon;
					navigation_polygon = tile_data->get_navigation_polygon(navigation_layer_index);
					navigation_polygon = tile_map_node->get_transformed_polygon(Ref<Resource>(navigation_polygon), c.alternative_tile);

					RID &region = r_cell_data.navigation_regions[navigation_layer_index];

					if (navigation_polygon.is_valid() && (navigation_polygon->get_polygon_count() > 0 || navigation_polygon->get_outline_count() > 0)) {
						// Create or update regions.
						Transform2D tile_transform;
						tile_transform.set_origin(tile_map_node->map_to_local(r_cell_data.coords));
						if (!region.is_valid()) {
							region = ns->region_create();
						}
						ns->region_set_owner_id(region, tile_map_node->get_instance_id());
						ns->region_set_map(region, navigation_map);
						ns->region_set_transform(region, tilemap_xform * tile_transform);
						ns->region_set_navigation_layers(region, tile_set->get_navigation_layer_layers(navigation_layer_index));
						ns->region_set_navigation_polygon(region, navigation_polygon);
					} else {
						// Clear region.
						if (region.is_valid()) {
							ns->region_set_map(region, RID());
							ns->free(region);
							region = RID();
						}
					}
				}

				return;
			}
		}
	}

	// If we did not return earlier, clear the cell.
	_navigation_clear_cell(r_cell_data);
}

#ifdef DEBUG_ENABLED
void TileMapLayer::_navigation_draw_cell_debug(const RID &p_canvas_item, const Vector2i &p_quadrant_pos, const CellData &r_cell_data) {
	// Draw the debug collision shapes.
	bool show_navigation = false;
	switch (tile_map_node->get_navigation_visibility_mode()) {
		case TileMap::VISIBILITY_MODE_DEFAULT:
			show_navigation = !Engine::get_singleton()->is_editor_hint() && tile_map_node->get_tree()->is_debugging_navigation_hint();
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

	// Check if the navigation is used.
	if (r_cell_data.navigation_regions.is_empty()) {
		return;
	}

	const Ref<TileSet> &tile_set = tile_map_node->get_tileset();

	RenderingServer *rs = RenderingServer::get_singleton();
	const NavigationServer2D *ns2d = NavigationServer2D::get_singleton();

	bool enabled_geometry_face_random_color = ns2d->get_debug_navigation_enable_geometry_face_random_color();
	bool enabled_edge_lines = ns2d->get_debug_navigation_enable_edge_lines();

	Color debug_face_color = ns2d->get_debug_navigation_geometry_face_color();
	Color debug_edge_color = ns2d->get_debug_navigation_geometry_edge_color();

	RandomPCG rand;

	const TileMapCell &c = r_cell_data.cell;

	TileSetSource *source;
	if (tile_set->has_source(c.source_id)) {
		source = *tile_set->get_source(c.source_id);

		if (source->has_tile(c.get_atlas_coords()) && source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				const TileData *tile_data;
				if (r_cell_data.runtime_tile_data_cache) {
					tile_data = r_cell_data.runtime_tile_data_cache;
				} else {
					tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);
				}

				Transform2D cell_to_quadrant;
				cell_to_quadrant.set_origin(tile_map_node->map_to_local(r_cell_data.coords) - p_quadrant_pos);
				rs->canvas_item_add_set_transform(p_canvas_item, cell_to_quadrant);

				for (int layer_index = 0; layer_index < tile_set->get_navigation_layers_count(); layer_index++) {
					Ref<NavigationPolygon> navigation_polygon = tile_data->get_navigation_polygon(layer_index);
					if (navigation_polygon.is_valid()) {
						navigation_polygon = tile_map_node->get_transformed_polygon(Ref<Resource>(navigation_polygon), c.alternative_tile);
						Vector<Vector2> navigation_polygon_vertices = navigation_polygon->get_vertices();
						if (navigation_polygon_vertices.size() < 3) {
							continue;
						}

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
	bool forced_cleanup = in_destructor || !enabled || !tile_map_node->is_inside_tree() || !tile_set.is_valid() || !tile_map_node->is_visible_in_tree();

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

	// Set data for a given tile from raw data.

	int c = p_data.size();
	const int *r = p_data.ptr();

	int offset = (p_format >= TileMapLayer::FORMAT_2) ? 3 : 2;
	ERR_FAIL_COND_MSG(c % offset != 0, vformat("Corrupted tile data. Got size: %s. Expected modulo: %s", offset));

	clear();

#ifdef DISABLE_DEPRECATED
	ERR_FAIL_COND_MSG(p_format != TileMapLayer::FORMAT_3, vformat("Cannot handle deprecated TileMap data format version %d. This Godot version was compiled with no support for deprecated data.", p_format));
#endif

	for (int i = 0; i < c; i += offset) {
		const uint8_t *ptr = (const uint8_t *)&r[i];
		uint8_t local[12];
		for (int j = 0; j < ((p_format >= TileMapLayer::FORMAT_2) ? 12 : 8); j++) {
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

		if (p_format == TileMapLayer::FORMAT_3) {
			uint16_t source_id = decode_uint16(&local[4]);
			uint16_t atlas_coords_x = decode_uint16(&local[6]);
			uint16_t atlas_coords_y = decode_uint16(&local[8]);
			uint16_t alternative_tile = decode_uint16(&local[10]);
			set_cell(Vector2i(x, y), source_id, Vector2i(atlas_coords_x, atlas_coords_y), alternative_tile);
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
			if (p_format == TileMapLayer::FORMAT_2) {
				coord_x = decode_uint16(&local[8]);
				coord_y = decode_uint16(&local[10]);
			}

			const Ref<TileSet> &tile_set = tile_map_node->get_tileset();
			if (tile_set.is_valid()) {
				Array a = tile_set->compatibility_tilemap_map(v, Vector2i(coord_x, coord_y), flip_h, flip_v, transpose);
				if (a.size() == 3) {
					set_cell(Vector2i(x, y), a[0], a[1], a[2]);
				} else {
					ERR_PRINT(vformat("No valid tile in Tileset for: tile:%s coords:%s flip_h:%s flip_v:%s transpose:%s", v, Vector2i(coord_x, coord_y), flip_h, flip_v, transpose));
				}
			} else {
				int compatibility_alternative_tile = ((int)flip_h) + ((int)flip_v << 1) + ((int)transpose << 2);
				set_cell(Vector2i(x, y), v, Vector2i(coord_x, coord_y), compatibility_alternative_tile);
			}
#endif
		}
	}
}

Vector<int> TileMapLayer::get_tile_data() const {
	// Export tile data to raw format.
	Vector<int> tile_data;
	tile_data.resize(tile_map.size() * 3);
	int *w = tile_data.ptrw();

	// Save in highest format.

	int idx = 0;
	for (const KeyValue<Vector2i, CellData> &E : tile_map) {
		uint8_t *ptr = (uint8_t *)&w[idx];
		encode_uint16((int16_t)(E.key.x), &ptr[0]);
		encode_uint16((int16_t)(E.key.y), &ptr[2]);
		encode_uint16(E.value.cell.source_id, &ptr[4]);
		encode_uint16(E.value.cell.coord_x, &ptr[6]);
		encode_uint16(E.value.cell.coord_y, &ptr[8]);
		encode_uint16(E.value.cell.alternative_tile, &ptr[10]);
		idx += 3;
	}

	return tile_data;
}

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

void TileMap::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			for (Ref<TileMapLayer> &layer : layers) {
				layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_IN_TREE);
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			for (Ref<TileMapLayer> &layer : layers) {
				layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_IN_TREE);
			}
		} break;

		case TileMap::NOTIFICATION_ENTER_CANVAS: {
			for (Ref<TileMapLayer> &layer : layers) {
				layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_IN_CANVAS);
			}
		} break;

		case TileMap::NOTIFICATION_EXIT_CANVAS: {
			for (Ref<TileMapLayer> &layer : layers) {
				layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_IN_CANVAS);
			}
		} break;

		case NOTIFICATION_DRAW: {
			// Rendering.
			if (tile_set.is_valid()) {
				RenderingServer::get_singleton()->canvas_item_set_sort_children_by_y(get_canvas_item(), is_y_sort_enabled());
			}
		} break;

		case TileMap::NOTIFICATION_VISIBILITY_CHANGED: {
			for (Ref<TileMapLayer> &layer : layers) {
				layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_VISIBILITY);
			}
		} break;

		case TileMap::NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			// Physics.
			bool in_editor = false;
#ifdef TOOLS_ENABLED
			in_editor = Engine::get_singleton()->is_editor_hint();
#endif
			if (is_inside_tree() && collision_animatable && !in_editor) {
				// Update transform on the physics tick when in animatable mode.
				last_valid_transform = new_transform;
				set_notify_local_transform(false);
				set_global_transform(new_transform);
				_update_notify_local_transform();
			}
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			// Physics.
			for (Ref<TileMapLayer> &layer : layers) {
				layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_XFORM);
			}
		} break;

		case TileMap::NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			for (Ref<TileMapLayer> &layer : layers) {
				layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_LOCAL_XFORM);
			}

			// Physics.
			bool in_editor = false;
#ifdef TOOLS_ENABLED
			in_editor = Engine::get_singleton()->is_editor_hint();
#endif

			// Only active when animatable. Send the new transform to the physics...
			if (is_inside_tree() && collision_animatable && !in_editor) {
				// Store last valid transform.
				new_transform = get_global_transform();

				// ... but then revert changes.
				set_notify_local_transform(false);
				set_global_transform(last_valid_transform);
				_update_notify_local_transform();
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

void TileMap::queue_internal_update() {
	if (pending_update) {
		return;
	}
	pending_update = true;
	callable_mp(this, &TileMap::_internal_update).call_deferred();
}

void TileMap::_internal_update() {
	// Other updates.
	if (!pending_update) {
		return;
	}

	// FIXME: This should only clear polygons that are no longer going to be used, but since it's difficult to determine,
	// the cache is never cleared at runtime to prevent invalidating used polygons.
	if (Engine::get_singleton()->is_editor_hint()) {
		polygon_cache.clear();
	}
	// Update dirty quadrants on layers.
	for (Ref<TileMapLayer> &layer : layers) {
		layer->internal_update();
	}

	pending_update = false;
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

	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_TILE_SET);
	}

	emit_signal(CoreStringNames::get_singleton()->changed);
}

Ref<TileSet> TileMap::get_tileset() const {
	return tile_set;
}

void TileMap::set_rendering_quadrant_size(int p_size) {
	ERR_FAIL_COND_MSG(p_size < 1, "TileMapQuadrant size cannot be smaller than 1.");

	rendering_quadrant_size = p_size;
	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_QUADRANT_SIZE);
	}
	emit_signal(CoreStringNames::get_singleton()->changed);
}

int TileMap::get_rendering_quadrant_size() const {
	return rendering_quadrant_size;
}

void TileMap::draw_tile(RID p_canvas_item, const Vector2 &p_position, const Ref<TileSet> p_tile_set, int p_atlas_source_id, const Vector2i &p_atlas_coords, int p_alternative_tile, int p_frame, Color p_modulation, const TileData *p_tile_data_override, real_t p_animation_offset) {
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
			// Accumulate durations unaffected by the speed to avoid accumulating floating point division errors.
			// Aka do `sum(duration[i]) / speed` instead of `sum(duration[i] / speed)`.
			real_t time_unscaled = 0.0;
			for (int frame = 0; frame < atlas_source->get_tile_animation_frames_count(p_atlas_coords); frame++) {
				real_t frame_duration_unscaled = atlas_source->get_tile_animation_frame_duration(p_atlas_coords, frame);
				real_t slice_start = time_unscaled / speed;
				real_t slice_end = (time_unscaled + frame_duration_unscaled) / speed;
				RenderingServer::get_singleton()->canvas_item_add_animation_slice(p_canvas_item, animation_duration, slice_start, slice_end, p_animation_offset);

				Rect2i source_rect = atlas_source->get_runtime_tile_texture_region(p_atlas_coords, frame);
				tex->draw_rect_region(p_canvas_item, dest_rect, source_rect, modulate, transpose, p_tile_set->is_uv_clipping());

				time_unscaled += frame_duration_unscaled;
			}
			RenderingServer::get_singleton()->canvas_item_add_animation_slice(p_canvas_item, 1.0, 0.0, 1.0, 0.0);
		}
	}
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
	Ref<TileMapLayer> new_layer;
	new_layer.instantiate();
	new_layer->set_tile_map(this);
	layers.insert(p_to_pos, new_layer);
	for (unsigned int i = 0; i < layers.size(); i++) {
		layers[i]->set_layer_index_in_tile_map_node(i);
	}
	queue_internal_update();
	notify_property_list_changed();

	emit_signal(CoreStringNames::get_singleton()->changed);

	update_configuration_warnings();
}

void TileMap::move_layer(int p_layer, int p_to_pos) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	ERR_FAIL_INDEX(p_to_pos, (int)layers.size() + 1);

	// Clear before shuffling layers.
	Ref<TileMapLayer> layer = layers[p_layer];
	layers.insert(p_to_pos, layer);
	layers.remove_at(p_to_pos < p_layer ? p_layer + 1 : p_layer);
	for (unsigned int i = 0; i < layers.size(); i++) {
		layers[i]->set_layer_index_in_tile_map_node(i);
	}
	queue_internal_update();
	notify_property_list_changed();

	if (selected_layer == p_layer) {
		selected_layer = p_to_pos < p_layer ? p_to_pos - 1 : p_to_pos;
	}

	emit_signal(CoreStringNames::get_singleton()->changed);

	update_configuration_warnings();
}

void TileMap::remove_layer(int p_layer) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	// Clear before removing the layer.
	layers.remove_at(p_layer);
	for (unsigned int i = 0; i < layers.size(); i++) {
		layers[i]->set_layer_index_in_tile_map_node(i);
	}
	queue_internal_update();
	notify_property_list_changed();

	if (selected_layer >= p_layer) {
		selected_layer -= 1;
	}

	emit_signal(CoreStringNames::get_singleton()->changed);

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
	_update_notify_local_transform();
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

void TileMap::set_collision_animatable(bool p_enabled) {
	if (collision_animatable == p_enabled) {
		return;
	}
	collision_animatable = p_enabled;
	_update_notify_local_transform();
	set_physics_process_internal(p_enabled);
	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_COLLISION_ANIMATABLE);
	}
	emit_signal(CoreStringNames::get_singleton()->changed);
}

bool TileMap::is_collision_animatable() const {
	return collision_animatable;
}

void TileMap::set_collision_visibility_mode(TileMap::VisibilityMode p_show_collision) {
	if (collision_visibility_mode == p_show_collision) {
		return;
	}
	collision_visibility_mode = p_show_collision;
	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_COLLISION_VISIBILITY_MODE);
	}
	emit_signal(CoreStringNames::get_singleton()->changed);
}

TileMap::VisibilityMode TileMap::get_collision_visibility_mode() {
	return collision_visibility_mode;
}

void TileMap::set_navigation_visibility_mode(TileMap::VisibilityMode p_show_navigation) {
	if (navigation_visibility_mode == p_show_navigation) {
		return;
	}
	navigation_visibility_mode = p_show_navigation;
	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_NAVIGATION_VISIBILITY_MODE);
	}
	emit_signal(CoreStringNames::get_singleton()->changed);
}

TileMap::VisibilityMode TileMap::get_navigation_visibility_mode() {
	return navigation_visibility_mode;
}

void TileMap::set_y_sort_enabled(bool p_enable) {
	if (is_y_sort_enabled() == p_enable) {
		return;
	}
	Node2D::set_y_sort_enabled(p_enable);
	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_Y_SORT_ENABLED);
	}
	emit_signal(CoreStringNames::get_singleton()->changed);
	update_configuration_warnings();
}

void TileMap::set_cell(int p_layer, const Vector2i &p_coords, int p_source_id, const Vector2i p_atlas_coords, int p_alternative_tile) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_cell, p_coords, p_source_id, p_atlas_coords, p_alternative_tile);
}

void TileMap::erase_cell(int p_layer, const Vector2i &p_coords) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_cell, p_coords, TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
}

int TileMap::get_cell_source_id(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, TileSet::INVALID_SOURCE, get_cell_source_id, p_coords, p_use_proxies);
}

Vector2i TileMap::get_cell_atlas_coords(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, TileSetSource::INVALID_ATLAS_COORDS, get_cell_atlas_coords, p_coords, p_use_proxies);
}

int TileMap::get_cell_alternative_tile(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, TileSetSource::INVALID_TILE_ALTERNATIVE, get_cell_alternative_tile, p_coords, p_use_proxies);
}

TileData *TileMap::get_cell_tile_data(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, nullptr, get_cell_tile_data, p_coords, p_use_proxies);
}

Ref<TileMapPattern> TileMap::get_pattern(int p_layer, TypedArray<Vector2i> p_coords_array) {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, Ref<TileMapPattern>(), get_pattern, p_coords_array);
}

Vector2i TileMap::map_pattern(const Vector2i &p_position_in_tilemap, const Vector2i &p_coords_in_pattern, Ref<TileMapPattern> p_pattern) {
	ERR_FAIL_COND_V(p_pattern.is_null(), Vector2i());
	ERR_FAIL_COND_V(!p_pattern->has_cell(p_coords_in_pattern), Vector2i());

	Vector2i output = p_position_in_tilemap + p_coords_in_pattern;
	if (tile_set->get_tile_shape() != TileSet::TILE_SHAPE_SQUARE) {
		if (tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_STACKED) {
			if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL && bool(p_position_in_tilemap.y % 2) && bool(p_coords_in_pattern.y % 2)) {
				output.x += 1;
			} else if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL && bool(p_position_in_tilemap.x % 2) && bool(p_coords_in_pattern.x % 2)) {
				output.y += 1;
			}
		} else if (tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_STACKED_OFFSET) {
			if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL && bool(p_position_in_tilemap.y % 2) && bool(p_coords_in_pattern.y % 2)) {
				output.x -= 1;
			} else if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL && bool(p_position_in_tilemap.x % 2) && bool(p_coords_in_pattern.x % 2)) {
				output.y -= 1;
			}
		}
	}

	return output;
}

void TileMap::set_pattern(int p_layer, const Vector2i &p_position, const Ref<TileMapPattern> p_pattern) {
	TILEMAP_CALL_FOR_LAYER(p_layer, set_pattern, p_position, p_pattern);
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
	TILEMAP_CALL_FOR_LAYER_V(p_layer, TileMapCell(), get_cell, p_coords, p_use_proxies);
}

Vector2i TileMap::get_coords_for_body_rid(RID p_physics_body) {
	for (const Ref<TileMapLayer> &layer : layers) {
		if (layer->has_body_rid(p_physics_body)) {
			return layer->get_coords_for_body_rid(p_physics_body);
		}
	}
	ERR_FAIL_V_MSG(Vector2i(), vformat("No tiles for the given body RID %d.", p_physics_body.get_id()));
}

int TileMap::get_layer_for_body_rid(RID p_physics_body) {
	for (unsigned int i = 0; i < layers.size(); i++) {
		if (layers[i]->has_body_rid(p_physics_body)) {
			return i;
		}
	}
	ERR_FAIL_V_MSG(-1, vformat("No tiles for the given body RID %d.", p_physics_body.get_id()));
}

void TileMap::fix_invalid_tiles() {
	for (Ref<TileMapLayer> &layer : layers) {
		layer->fix_invalid_tiles();
	}
}

void TileMap::clear_layer(int p_layer) {
	TILEMAP_CALL_FOR_LAYER(p_layer, clear)
}

void TileMap::clear() {
	for (Ref<TileMapLayer> &layer : layers) {
		layer->clear();
	}
}

void TileMap::update_internals() {
	pending_update = true;
	_internal_update();
}

void TileMap::notify_runtime_tile_data_update(int p_layer) {
	if (p_layer >= 0) {
		TILEMAP_CALL_FOR_LAYER(p_layer, notify_tile_map_change, TileMapLayer::DIRTY_FLAGS_TILE_MAP_RUNTIME_UPDATE);
	} else {
		for (Ref<TileMapLayer> &layer : layers) {
			layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_RUNTIME_UPDATE);
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
	for (unsigned int i = 1; i < layers.size(); i++) {
		rect = rect.merge(layers[i]->get_rect(changed));
		any_changed |= changed;
	}
	const_cast<TileMap *>(this)->item_rect_changed(any_changed);
	return rect;
}
#endif

PackedVector2Array TileMap::_get_transformed_vertices(const PackedVector2Array &p_vertices, int p_alternative_id) {
	const Vector2 *r = p_vertices.ptr();
	int size = p_vertices.size();

	PackedVector2Array new_points;
	new_points.resize(size);
	Vector2 *w = new_points.ptrw();

	bool flip_h = (p_alternative_id & TileSetAtlasSource::TRANSFORM_FLIP_H);
	bool flip_v = (p_alternative_id & TileSetAtlasSource::TRANSFORM_FLIP_V);
	bool transpose = (p_alternative_id & TileSetAtlasSource::TRANSFORM_TRANSPOSE);

	for (int i = 0; i < size; i++) {
		Vector2 v;
		if (transpose) {
			v = Vector2(r[i].y, r[i].x);
		} else {
			v = r[i];
		}

		if (flip_h) {
			v.x *= -1;
		}
		if (flip_v) {
			v.y *= -1;
		}
		w[i] = v;
	}
	return new_points;
}

bool TileMap::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (p_name == "format") {
		if (p_value.get_type() == Variant::INT) {
			format = (TileMapLayer::DataFormat)(p_value.operator int64_t()); // Set format used for loading.
			return true;
		}
#ifndef DISABLE_DEPRECATED
	} else if (p_name == "tile_data") { // Kept for compatibility reasons.
		if (p_value.is_array()) {
			if (layers.size() == 0) {
				Ref<TileMapLayer> new_layer;
				new_layer.instantiate();
				new_layer->set_tile_map(this);
				new_layer->set_layer_index_in_tile_map_node(0);
				layers.push_back(new_layer);
			}
			layers[0]->set_tile_data(format, p_value);
			emit_signal(CoreStringNames::get_singleton()->changed);
			return true;
		}
		return false;
	} else if (p_name == "rendering_quadrant_size") {
		set_rendering_quadrant_size(p_value);
#endif // DISABLE_DEPRECATED
	} else if (components.size() == 2 && components[0].begins_with("layer_") && components[0].trim_prefix("layer_").is_valid_int()) {
		int index = components[0].trim_prefix("layer_").to_int();
		if (index < 0) {
			return false;
		}

		if (index >= (int)layers.size()) {
			while (index >= (int)layers.size()) {
				Ref<TileMapLayer> new_layer;
				new_layer.instantiate();
				new_layer->set_tile_map(this);
				new_layer->set_layer_index_in_tile_map_node(index);
				layers.push_back(new_layer);
			}

			notify_property_list_changed();
			emit_signal(CoreStringNames::get_singleton()->changed);
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
			layers[index]->set_tile_data(format, p_value);
			emit_signal(CoreStringNames::get_singleton()->changed);
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
		r_ret = TileMapLayer::FORMAT_MAX - 1; // When saving, always save highest format.
		return true;
	} else if (components.size() == 2 && components[0].begins_with("layer_") && components[0].trim_prefix("layer_").is_valid_int()) {
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
			r_ret = layers[index]->get_tile_data();
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

	for (unsigned int i = 0; i < layers.size(); i++) {
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
	// SHOULD RETURN THE CENTER OF THE CELL.
	ERR_FAIL_COND_V(!tile_set.is_valid(), Vector2());

	Vector2 ret = p_pos;
	TileSet::TileShape tile_shape = tile_set->get_tile_shape();
	TileSet::TileOffsetAxis tile_offset_axis = tile_set->get_tile_offset_axis();

	if (tile_shape == TileSet::TILE_SHAPE_HALF_OFFSET_SQUARE || tile_shape == TileSet::TILE_SHAPE_HEXAGON || tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		// Technically, those 3 shapes are equivalent, as they are basically half-offset, but with different levels or overlap.
		// square = no overlap, hexagon = 0.25 overlap, isometric = 0.5 overlap.
		if (tile_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			switch (tile_set->get_tile_layout()) {
				case TileSet::TILE_LAYOUT_STACKED:
					ret = Vector2(ret.x + (Math::posmod(ret.y, 2) == 0 ? 0.0 : 0.5), ret.y);
					break;
				case TileSet::TILE_LAYOUT_STACKED_OFFSET:
					ret = Vector2(ret.x + (Math::posmod(ret.y, 2) == 1 ? 0.0 : 0.5), ret.y);
					break;
				case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
					ret = Vector2(ret.x + ret.y / 2, ret.y);
					break;
				case TileSet::TILE_LAYOUT_STAIRS_DOWN:
					ret = Vector2(ret.x / 2, ret.y * 2 + ret.x);
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
					ret = Vector2((ret.x + ret.y) / 2, ret.y - ret.x);
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
					ret = Vector2((ret.x - ret.y) / 2, ret.y + ret.x);
					break;
			}
		} else { // TILE_OFFSET_AXIS_VERTICAL.
			switch (tile_set->get_tile_layout()) {
				case TileSet::TILE_LAYOUT_STACKED:
					ret = Vector2(ret.x, ret.y + (Math::posmod(ret.x, 2) == 0 ? 0.0 : 0.5));
					break;
				case TileSet::TILE_LAYOUT_STACKED_OFFSET:
					ret = Vector2(ret.x, ret.y + (Math::posmod(ret.x, 2) == 1 ? 0.0 : 0.5));
					break;
				case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
					ret = Vector2(ret.x * 2 + ret.y, ret.y / 2);
					break;
				case TileSet::TILE_LAYOUT_STAIRS_DOWN:
					ret = Vector2(ret.x, ret.y + ret.x / 2);
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
					ret = Vector2(ret.x + ret.y, (ret.y - ret.x) / 2);
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
					ret = Vector2(ret.x - ret.y, (ret.y + ret.x) / 2);
					break;
			}
		}
	}

	// Multiply by the overlapping ratio.
	double overlapping_ratio = 1.0;
	if (tile_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
		if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
			overlapping_ratio = 0.5;
		} else if (tile_shape == TileSet::TILE_SHAPE_HEXAGON) {
			overlapping_ratio = 0.75;
		}
		ret.y *= overlapping_ratio;
	} else { // TILE_OFFSET_AXIS_VERTICAL.
		if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
			overlapping_ratio = 0.5;
		} else if (tile_shape == TileSet::TILE_SHAPE_HEXAGON) {
			overlapping_ratio = 0.75;
		}
		ret.x *= overlapping_ratio;
	}

	return (ret + Vector2(0.5, 0.5)) * tile_set->get_tile_size();
}

Vector2i TileMap::local_to_map(const Vector2 &p_local_position) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), Vector2i());

	Vector2 ret = p_local_position;
	ret /= tile_set->get_tile_size();

	TileSet::TileShape tile_shape = tile_set->get_tile_shape();
	TileSet::TileOffsetAxis tile_offset_axis = tile_set->get_tile_offset_axis();
	TileSet::TileLayout tile_layout = tile_set->get_tile_layout();

	// Divide by the overlapping ratio.
	double overlapping_ratio = 1.0;
	if (tile_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
		if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
			overlapping_ratio = 0.5;
		} else if (tile_shape == TileSet::TILE_SHAPE_HEXAGON) {
			overlapping_ratio = 0.75;
		}
		ret.y /= overlapping_ratio;
	} else { // TILE_OFFSET_AXIS_VERTICAL.
		if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
			overlapping_ratio = 0.5;
		} else if (tile_shape == TileSet::TILE_SHAPE_HEXAGON) {
			overlapping_ratio = 0.75;
		}
		ret.x /= overlapping_ratio;
	}

	// For each half-offset shape, we check if we are in the corner of the tile, and thus should correct the local position accordingly.
	if (tile_shape == TileSet::TILE_SHAPE_HALF_OFFSET_SQUARE || tile_shape == TileSet::TILE_SHAPE_HEXAGON || tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		// Technically, those 3 shapes are equivalent, as they are basically half-offset, but with different levels or overlap.
		// square = no overlap, hexagon = 0.25 overlap, isometric = 0.5 overlap.
		if (tile_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			// Smart floor of the position
			Vector2 raw_pos = ret;
			if (Math::posmod(Math::floor(ret.y), 2) ^ (tile_layout == TileSet::TILE_LAYOUT_STACKED_OFFSET)) {
				ret = Vector2(Math::floor(ret.x + 0.5) - 0.5, Math::floor(ret.y));
			} else {
				ret = ret.floor();
			}

			// Compute the tile offset, and if we might the output for a neighbor top tile.
			Vector2 in_tile_pos = raw_pos - ret;
			bool in_top_left_triangle = (in_tile_pos - Vector2(0.5, 0.0)).cross(Vector2(-0.5, 1.0 / overlapping_ratio - 1)) <= 0;
			bool in_top_right_triangle = (in_tile_pos - Vector2(0.5, 0.0)).cross(Vector2(0.5, 1.0 / overlapping_ratio - 1)) > 0;

			switch (tile_layout) {
				case TileSet::TILE_LAYOUT_STACKED:
					ret = ret.floor();
					if (in_top_left_triangle) {
						ret += Vector2i(Math::posmod(Math::floor(ret.y), 2) ? 0 : -1, -1);
					} else if (in_top_right_triangle) {
						ret += Vector2i(Math::posmod(Math::floor(ret.y), 2) ? 1 : 0, -1);
					}
					break;
				case TileSet::TILE_LAYOUT_STACKED_OFFSET:
					ret = ret.floor();
					if (in_top_left_triangle) {
						ret += Vector2i(Math::posmod(Math::floor(ret.y), 2) ? -1 : 0, -1);
					} else if (in_top_right_triangle) {
						ret += Vector2i(Math::posmod(Math::floor(ret.y), 2) ? 0 : 1, -1);
					}
					break;
				case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
					ret = Vector2(ret.x - ret.y / 2, ret.y).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(0, -1);
					} else if (in_top_right_triangle) {
						ret += Vector2i(1, -1);
					}
					break;
				case TileSet::TILE_LAYOUT_STAIRS_DOWN:
					ret = Vector2(ret.x * 2, ret.y / 2 - ret.x).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, 0);
					} else if (in_top_right_triangle) {
						ret += Vector2i(1, -1);
					}
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
					ret = Vector2(ret.x - ret.y / 2, ret.y / 2 + ret.x).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(0, -1);
					} else if (in_top_right_triangle) {
						ret += Vector2i(1, 0);
					}
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
					ret = Vector2(ret.x + ret.y / 2, ret.y / 2 - ret.x).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, 0);
					} else if (in_top_right_triangle) {
						ret += Vector2i(0, -1);
					}
					break;
			}
		} else { // TILE_OFFSET_AXIS_VERTICAL.
			// Smart floor of the position.
			Vector2 raw_pos = ret;
			if (Math::posmod(Math::floor(ret.x), 2) ^ (tile_layout == TileSet::TILE_LAYOUT_STACKED_OFFSET)) {
				ret = Vector2(Math::floor(ret.x), Math::floor(ret.y + 0.5) - 0.5);
			} else {
				ret = ret.floor();
			}

			// Compute the tile offset, and if we might the output for a neighbor top tile.
			Vector2 in_tile_pos = raw_pos - ret;
			bool in_top_left_triangle = (in_tile_pos - Vector2(0.0, 0.5)).cross(Vector2(1.0 / overlapping_ratio - 1, -0.5)) > 0;
			bool in_bottom_left_triangle = (in_tile_pos - Vector2(0.0, 0.5)).cross(Vector2(1.0 / overlapping_ratio - 1, 0.5)) <= 0;

			switch (tile_layout) {
				case TileSet::TILE_LAYOUT_STACKED:
					ret = ret.floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, Math::posmod(Math::floor(ret.x), 2) ? 0 : -1);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(-1, Math::posmod(Math::floor(ret.x), 2) ? 1 : 0);
					}
					break;
				case TileSet::TILE_LAYOUT_STACKED_OFFSET:
					ret = ret.floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, Math::posmod(Math::floor(ret.x), 2) ? -1 : 0);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(-1, Math::posmod(Math::floor(ret.x), 2) ? 0 : 1);
					}
					break;
				case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
					ret = Vector2(ret.x / 2 - ret.y, ret.y * 2).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(0, -1);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(-1, 1);
					}
					break;
				case TileSet::TILE_LAYOUT_STAIRS_DOWN:
					ret = Vector2(ret.x, ret.y - ret.x / 2).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, 0);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(-1, 1);
					}
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
					ret = Vector2(ret.x / 2 - ret.y, ret.y + ret.x / 2).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(0, -1);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(-1, 0);
					}
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
					ret = Vector2(ret.x / 2 + ret.y, ret.y - ret.x / 2).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, 0);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(0, 1);
					}
					break;
			}
		}
	} else {
		ret = (ret + Vector2(0.00005, 0.00005)).floor();
	}
	return Vector2i(ret);
}

bool TileMap::is_existing_neighbor(TileSet::CellNeighbor p_cell_neighbor) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), false);

	TileSet::TileShape shape = tile_set->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		return p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER;

	} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		return p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER ||
				p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
	} else {
		if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			return p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE ||
					p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
					p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
					p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE ||
					p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
					p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
		} else {
			return p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
					p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE ||
					p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
					p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
					p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE ||
					p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
		}
	}
}

Vector2i TileMap::get_neighbor_cell(const Vector2i &p_coords, TileSet::CellNeighbor p_cell_neighbor) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), p_coords);

	TileSet::TileShape shape = tile_set->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		switch (p_cell_neighbor) {
			case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
				return p_coords + Vector2i(1, 0);
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				return p_coords + Vector2i(1, 1);
			case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
				return p_coords + Vector2i(0, 1);
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				return p_coords + Vector2i(-1, 1);
			case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
				return p_coords + Vector2i(-1, 0);
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				return p_coords + Vector2i(-1, -1);
			case TileSet::CELL_NEIGHBOR_TOP_SIDE:
				return p_coords + Vector2i(0, -1);
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
				return p_coords + Vector2i(1, -1);
			default:
				ERR_FAIL_V(p_coords);
		}
	} else { // Half-offset shapes (square and hexagon).
		switch (tile_set->get_tile_layout()) {
			case TileSet::TILE_LAYOUT_STACKED: {
				if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
					bool is_offset = p_coords.y % 2;
					if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
						return p_coords + Vector2i(1, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
						return p_coords + Vector2i(is_offset ? 1 : 0, 1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
						return p_coords + Vector2i(0, 2);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
						return p_coords + Vector2i(is_offset ? 0 : -1, 1);
					} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
						return p_coords + Vector2i(-1, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
						return p_coords + Vector2i(is_offset ? 0 : -1, -1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
						return p_coords + Vector2i(0, -2);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
						return p_coords + Vector2i(is_offset ? 1 : 0, -1);
					} else {
						ERR_FAIL_V(p_coords);
					}
				} else {
					bool is_offset = p_coords.x % 2;

					if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
						return p_coords + Vector2i(0, 1);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
						return p_coords + Vector2i(1, is_offset ? 1 : 0);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
						return p_coords + Vector2i(2, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
						return p_coords + Vector2i(1, is_offset ? 0 : -1);
					} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
						return p_coords + Vector2i(0, -1);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
						return p_coords + Vector2i(-1, is_offset ? 0 : -1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
						return p_coords + Vector2i(-2, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
						return p_coords + Vector2i(-1, is_offset ? 1 : 0);
					} else {
						ERR_FAIL_V(p_coords);
					}
				}
			} break;
			case TileSet::TILE_LAYOUT_STACKED_OFFSET: {
				if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
					bool is_offset = p_coords.y % 2;

					if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
						return p_coords + Vector2i(1, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
						return p_coords + Vector2i(is_offset ? 0 : 1, 1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
						return p_coords + Vector2i(0, 2);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
						return p_coords + Vector2i(is_offset ? -1 : 0, 1);
					} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
						return p_coords + Vector2i(-1, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
						return p_coords + Vector2i(is_offset ? -1 : 0, -1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
						return p_coords + Vector2i(0, -2);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
						return p_coords + Vector2i(is_offset ? 0 : 1, -1);
					} else {
						ERR_FAIL_V(p_coords);
					}
				} else {
					bool is_offset = p_coords.x % 2;

					if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
						return p_coords + Vector2i(0, 1);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
						return p_coords + Vector2i(1, is_offset ? 0 : 1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
						return p_coords + Vector2i(2, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
						return p_coords + Vector2i(1, is_offset ? -1 : 0);
					} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
						return p_coords + Vector2i(0, -1);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
						return p_coords + Vector2i(-1, is_offset ? -1 : 0);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
						return p_coords + Vector2i(-2, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
						return p_coords + Vector2i(-1, is_offset ? 0 : 1);
					} else {
						ERR_FAIL_V(p_coords);
					}
				}
			} break;
			case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
			case TileSet::TILE_LAYOUT_STAIRS_DOWN: {
				if ((tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_STAIRS_RIGHT) ^ (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
					if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
							return p_coords + Vector2i(1, 0);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
							return p_coords + Vector2i(-1, 2);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
							return p_coords + Vector2i(-1, 0);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
							return p_coords + Vector2i(1, -2);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, -1);
						} else {
							ERR_FAIL_V(p_coords);
						}

					} else {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
							return p_coords + Vector2i(0, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
							return p_coords + Vector2i(2, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, -1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
							return p_coords + Vector2i(0, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
							return p_coords + Vector2i(-2, 1);

						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 1);
						} else {
							ERR_FAIL_V(p_coords);
						}
					}
				} else {
					if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
							return p_coords + Vector2i(2, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
							return p_coords + Vector2i(0, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
							return p_coords + Vector2i(-2, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
							return p_coords + Vector2i(0, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, -1);
						} else {
							ERR_FAIL_V(p_coords);
						}

					} else {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
							return p_coords + Vector2i(-1, 2);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
							return p_coords + Vector2i(1, 0);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, -1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
							return p_coords + Vector2i(1, -2);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
							return p_coords + Vector2i(-1, 0);

						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 1);
						} else {
							ERR_FAIL_V(p_coords);
						}
					}
				}
			} break;
			case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
			case TileSet::TILE_LAYOUT_DIAMOND_DOWN: {
				if ((tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_DIAMOND_RIGHT) ^ (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
					if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
							return p_coords + Vector2i(1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
							return p_coords + Vector2i(-1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
							return p_coords + Vector2i(-1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
							return p_coords + Vector2i(1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else {
							ERR_FAIL_V(p_coords);
						}

					} else {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
							return p_coords + Vector2i(1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
							return p_coords + Vector2i(1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
							return p_coords + Vector2i(-1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
							return p_coords + Vector2i(-1, 1);

						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else {
							ERR_FAIL_V(p_coords);
						}
					}
				} else {
					if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
							return p_coords + Vector2i(1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
							return p_coords + Vector2i(1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
							return p_coords + Vector2i(-1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
							return p_coords + Vector2i(-1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else {
							ERR_FAIL_V(p_coords);
						}

					} else {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
							return p_coords + Vector2i(-1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
							return p_coords + Vector2i(1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
							return p_coords + Vector2i(1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
							return p_coords + Vector2i(-1, -1);

						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else {
							ERR_FAIL_V(p_coords);
						}
					}
				}
			} break;
			default:
				ERR_FAIL_V(p_coords);
		}
	}
}

TypedArray<Vector2i> TileMap::get_used_cells(int p_layer) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, TypedArray<Vector2i>(), get_used_cells);
}

TypedArray<Vector2i> TileMap::get_used_cells_by_id(int p_layer, int p_source_id, const Vector2i p_atlas_coords, int p_alternative_tile) const {
	TILEMAP_CALL_FOR_LAYER_V(p_layer, TypedArray<Vector2i>(), get_used_cells_by_id, p_source_id, p_atlas_coords, p_alternative_tile);
}

Rect2i TileMap::get_used_rect() const {
	// Return the visible rect of the tilemap.
	bool first = true;
	Rect2i rect = Rect2i();
	for (const Ref<TileMapLayer> &layer : layers) {
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
	// Occlusion: set light mask.
	CanvasItem::set_light_mask(p_light_mask);
	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_LIGHT_MASK);
	}
}

void TileMap::set_material(const Ref<Material> &p_material) {
	// Set material for the whole tilemap.
	CanvasItem::set_material(p_material);

	// Update material for the whole tilemap.
	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_MATERIAL);
	}
}

void TileMap::set_use_parent_material(bool p_use_parent_material) {
	// Set use_parent_material for the whole tilemap.
	CanvasItem::set_use_parent_material(p_use_parent_material);

	// Update use_parent_material for the whole tilemap.
	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_USE_PARENT_MATERIAL);
	}
}

void TileMap::set_texture_filter(TextureFilter p_texture_filter) {
	// Set a default texture filter for the whole tilemap.
	CanvasItem::set_texture_filter(p_texture_filter);
	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_TEXTURE_FILTER);
	}
}

void TileMap::set_texture_repeat(CanvasItem::TextureRepeat p_texture_repeat) {
	// Set a default texture repeat for the whole tilemap.
	CanvasItem::set_texture_repeat(p_texture_repeat);
	for (Ref<TileMapLayer> &layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_TEXTURE_REPEAT);
	}
}

TypedArray<Vector2i> TileMap::get_surrounding_cells(const Vector2i &coords) {
	if (!tile_set.is_valid()) {
		return TypedArray<Vector2i>();
	}

	TypedArray<Vector2i> around;
	TileSet::TileShape shape = tile_set->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_LEFT_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_SIDE));
	} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE));
	} else {
		if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_LEFT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE));
		} else {
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE));
		}
	}

	return around;
}

void TileMap::draw_cells_outline(Control *p_control, const RBSet<Vector2i> &p_cells, Color p_color, Transform2D p_transform) {
	if (!tile_set.is_valid()) {
		return;
	}

	// Create a set.
	Vector2i tile_size = tile_set->get_tile_size();
	Vector<Vector2> polygon = tile_set->get_tile_shape_polygon();
	TileSet::TileShape shape = tile_set->get_tile_shape();

	for (const Vector2i &E : p_cells) {
		Vector2 center = map_to_local(E);

#define DRAW_SIDE_IF_NEEDED(side, polygon_index_from, polygon_index_to)                     \
	if (!p_cells.has(get_neighbor_cell(E, side))) {                                         \
		Vector2 from = p_transform.xform(center + polygon[polygon_index_from] * tile_size); \
		Vector2 to = p_transform.xform(center + polygon[polygon_index_to] * tile_size);     \
		p_control->draw_line(from, to, p_color);                                            \
	}

		if (shape == TileSet::TILE_SHAPE_SQUARE) {
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_RIGHT_SIDE, 1, 2);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, 2, 3);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_LEFT_SIDE, 3, 0);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_SIDE, 0, 1);
		} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, 2, 3);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, 1, 2);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, 0, 1);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, 3, 0);
		} else {
			if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, 3, 4);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, 2, 3);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_LEFT_SIDE, 1, 2);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, 0, 1);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, 5, 0);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_RIGHT_SIDE, 4, 5);
			} else {
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, 3, 4);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, 4, 5);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, 5, 0);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, 0, 1);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_SIDE, 1, 2);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, 2, 3);
			}
		}
	}
#undef DRAW_SIDE_IF_NEEDED
}

Ref<Resource> TileMap::get_transformed_polygon(Ref<Resource> p_polygon, int p_alternative_id) {
	if (!bool(p_alternative_id & (TileSetAtlasSource::TRANSFORM_FLIP_H | TileSetAtlasSource::TRANSFORM_FLIP_V | TileSetAtlasSource::TRANSFORM_TRANSPOSE))) {
		return p_polygon;
	}

	{
		HashMap<Pair<Ref<Resource>, int>, Ref<Resource>, PairHash<Ref<Resource>, int>>::Iterator E = polygon_cache.find(Pair<Ref<Resource>, int>(p_polygon, p_alternative_id));
		if (E) {
			return E->value;
		}
	}

	Ref<ConvexPolygonShape2D> col = p_polygon;
	if (col.is_valid()) {
		Ref<ConvexPolygonShape2D> ret;
		ret.instantiate();
		ret->set_points(_get_transformed_vertices(col->get_points(), p_alternative_id));
		polygon_cache[Pair<Ref<Resource>, int>(p_polygon, p_alternative_id)] = ret;
		return ret;
	}

	Ref<NavigationPolygon> nav = p_polygon;
	if (nav.is_valid()) {
		PackedVector2Array new_points = _get_transformed_vertices(nav->get_vertices(), p_alternative_id);
		Ref<NavigationPolygon> ret;
		ret.instantiate();
		ret->set_vertices(new_points);

		PackedInt32Array indices;
		indices.resize(new_points.size());
		int *w = indices.ptrw();
		for (int i = 0; i < new_points.size(); i++) {
			w[i] = i;
		}
		ret->add_polygon(indices);
		polygon_cache[Pair<Ref<Resource>, int>(p_polygon, p_alternative_id)] = ret;
		return ret;
	}

	Ref<OccluderPolygon2D> ocd = p_polygon;
	if (ocd.is_valid()) {
		Ref<OccluderPolygon2D> ret;
		ret.instantiate();
		ret->set_polygon(_get_transformed_vertices(ocd->get_polygon(), p_alternative_id));
		polygon_cache[Pair<Ref<Resource>, int>(p_polygon, p_alternative_id)] = ret;
		return ret;
	}

	return p_polygon;
}

PackedStringArray TileMap::get_configuration_warnings() const {
	PackedStringArray warnings = Node::get_configuration_warnings();

	// Retrieve the set of Z index values with a Y-sorted layer.
	RBSet<int> y_sorted_z_index;
	for (const Ref<TileMapLayer> &layer : layers) {
		if (layer->is_y_sort_enabled()) {
			y_sorted_z_index.insert(layer->get_z_index());
		}
	}

	// Check if we have a non-sorted layer in a Z-index with a Y-sorted layer.
	for (const Ref<TileMapLayer> &layer : layers) {
		if (!layer->is_y_sort_enabled() && y_sorted_z_index.has(layer->get_z_index())) {
			warnings.push_back(RTR("A Y-sorted layer has the same Z-index value as a not Y-sorted layer.\nThis may lead to unwanted behaviors, as a layer that is not Y-sorted will be Y-sorted as a whole with tiles from Y-sorted layers."));
			break;
		}
	}

	if (!is_y_sort_enabled()) {
		// Check if Y-sort is enabled on a layer but not on the node.
		for (const Ref<TileMapLayer> &layer : layers) {
			if (layer->is_y_sort_enabled()) {
				warnings.push_back(RTR("A TileMap layer is set as Y-sorted, but Y-sort is not enabled on the TileMap node itself."));
				break;
			}
		}
	} else {
		// Check if Y-sort is enabled on the node, but not on any of the layers.
		bool need_warning = true;
		for (const Ref<TileMapLayer> &layer : layers) {
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
			for (const Ref<TileMapLayer> &layer : layers) {
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

	ClassDB::bind_method(D_METHOD("get_pattern", "layer", "coords_array"), &TileMap::get_pattern);
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

	ADD_PROPERTY_DEFAULT("format", TileMapLayer::FORMAT_1);

	ADD_SIGNAL(MethodInfo(CoreStringNames::get_singleton()->changed));

	BIND_ENUM_CONSTANT(VISIBILITY_MODE_DEFAULT);
	BIND_ENUM_CONSTANT(VISIBILITY_MODE_FORCE_HIDE);
	BIND_ENUM_CONSTANT(VISIBILITY_MODE_FORCE_SHOW);
}

void TileMap::_tile_set_changed() {
	emit_signal(CoreStringNames::get_singleton()->changed);
	for (Ref<TileMapLayer> layer : layers) {
		layer->notify_tile_map_change(TileMapLayer::DIRTY_FLAGS_TILE_MAP_TILE_SET);
	}
	update_configuration_warnings();
}

void TileMap::_update_notify_local_transform() {
	bool notify = collision_animatable || is_y_sort_enabled();
	if (!notify) {
		for (const Ref<TileMapLayer> &layer : layers) {
			if (layer->is_y_sort_enabled()) {
				notify = true;
				break;
			}
		}
	}
	set_notify_local_transform(notify);
}

TileMap::TileMap() {
	set_notify_transform(true);
	_update_notify_local_transform();

	Ref<TileMapLayer> new_layer;
	new_layer.instantiate();
	new_layer->set_tile_map(this);
	new_layer->set_layer_index_in_tile_map_node(0);
	layers.push_back(new_layer);

	default_layer.instantiate();
}

TileMap::~TileMap() {
	if (tile_set.is_valid()) {
		tile_set->disconnect_changed(callable_mp(this, &TileMap::_tile_set_changed));
	}
}

#undef TILEMAP_CALL_FOR_LAYER
#undef TILEMAP_CALL_FOR_LAYER_V
