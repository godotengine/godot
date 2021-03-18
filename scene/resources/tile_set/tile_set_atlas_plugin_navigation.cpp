/*************************************************************************/
/*  tile_set_atlas_plugin_navigation.cpp                                 */
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

#include "tile_set_atlas_plugin_navigation.h"

#include "scene/2d/navigation_region_2d.h"
#include "scene/2d/tile_map.h"
#include "servers/navigation_server_2d.h"

void TileSetAtlasPluginNavigation::tilemap_notification(TileMap *p_tile_map, int p_what) {
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

void TileSetAtlasPluginNavigation::update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
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
			TileMapCell c = p_tile_map->get_cell(E_cell->get());

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

void TileSetAtlasPluginNavigation::cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
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

void TileSetAtlasPluginNavigation::draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	// Draw the debug collision shapes.
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!p_tile_map->get_tree() || !(Engine::get_singleton()->is_editor_hint() || p_tile_map->get_tree()->is_debugging_navigation_hint())) {
		return;
	}

	RenderingServer *rs = RenderingServer::get_singleton();

	Color color = p_tile_map->get_tree()->get_debug_navigation_color();
	RandomPCG rand;

	Vector2 quadrant_pos = p_tile_map->map_to_world(p_quadrant->coords * p_tile_map->get_effective_quadrant_size());

	for (Set<Vector2i>::Element *E_cell = p_quadrant->cells.front(); E_cell; E_cell = E_cell->next()) {
		TileMapCell c = p_tile_map->get_cell(E_cell->get());

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

							RS::get_singleton()->canvas_item_add_polygon(p_quadrant->debug_canvas_item, vertices, colors);
						}
					}
				}
			}
		}
	}
}
