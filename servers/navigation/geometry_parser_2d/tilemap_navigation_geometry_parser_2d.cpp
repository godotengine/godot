/**************************************************************************/
/*  tilemap_navigation_geometry_parser_2d.cpp                             */
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

#include "tilemap_navigation_geometry_parser_2d.h"

#include "scene/2d/tile_map.h"

#ifdef CLIPPER_ENABLED
#include "thirdparty/clipper2/include/clipper2/clipper.h"
#endif // CLIPPER_ENABLED

bool TileMap2DNavigationGeometryParser2D::parses_node(Node *p_node) {
	return (Object::cast_to<TileMap>(p_node) != nullptr);
}

void TileMap2DNavigationGeometryParser2D::parse_geometry(Node *p_node, Ref<NavigationPolygon> p_navigation_polygon, Ref<NavigationMeshSourceGeometryData2D> p_source_geometry) {
#ifdef CLIPPER_ENABLED
	TileMap *tilemap = Object::cast_to<TileMap>(p_node);
	NavigationPolygon::ParsedGeometryType parsed_geometry_type = p_navigation_polygon->get_parsed_geometry_type();
	uint32_t navigation_polygon_collision_mask = p_navigation_polygon->get_collision_mask();

	if (tilemap) {
		if (tilemap->get_layers_count() <= 0) {
			return;
		}

		int tilemap_layer = 0; // only main tile map layer is supported

		Ref<TileSet> tile_set = tilemap->get_tileset();
		if (!tile_set.is_valid()) {
			return;
		}

		int physics_layers_count = tile_set->get_physics_layers_count();
		int navigation_layers_count = tile_set->get_navigation_layers_count();

		if (physics_layers_count <= 0 && navigation_layers_count <= 0) {
			return;
		}

		const Transform2D tilemap_xform = tilemap->get_transform();
		TypedArray<Vector2i> used_cells = tilemap->get_used_cells(tilemap_layer);

		for (int used_cell_index = 0; used_cell_index < used_cells.size(); used_cell_index++) {
			const Vector2i &cell = used_cells[used_cell_index];

			const TileData *tile_data = tilemap->get_cell_tile_data(tilemap_layer, cell, false);

			Transform2D tile_transform;
			tile_transform.set_origin(tilemap->map_to_local(cell));

			const Transform2D tile_transform_offset = tilemap_xform * tile_transform;

			if (navigation_layers_count > 0) {
				Ref<NavigationPolygon> navigation_polygon = tile_data->get_navigation_polygon(tilemap_layer);
				if (navigation_polygon.is_valid()) {
					for (int outline_index = 0; outline_index < navigation_polygon->get_outline_count(); outline_index++) {
						Vector<Vector2> traversable_outline = navigation_polygon->get_outline(outline_index);

						for (int traversable_outline_index = 0; traversable_outline_index < traversable_outline.size(); traversable_outline_index++) {
							traversable_outline.write[traversable_outline_index] = tile_transform_offset.xform(traversable_outline[traversable_outline_index]);
						}

						p_source_geometry->_add_traversable_outline(traversable_outline);
					}
				}
			}

			if (physics_layers_count > 0 && parsed_geometry_type != NavigationPolygon::PARSED_GEOMETRY_MESH_INSTANCES && (tile_set->get_physics_layer_collision_layer(tilemap_layer) & navigation_polygon_collision_mask)) {
				for (int collision_polygon_index = 0; collision_polygon_index < tile_data->get_collision_polygons_count(tilemap_layer); collision_polygon_index++) {
					Vector<Vector2> obstruction_outline = tile_data->get_collision_polygon_points(tilemap_layer, collision_polygon_index);

					for (int obstruction_outline_index = 0; obstruction_outline_index < obstruction_outline.size(); obstruction_outline_index++) {
						obstruction_outline.write[obstruction_outline_index] = tile_transform_offset.xform(obstruction_outline[obstruction_outline_index]);
					}

					p_source_geometry->_add_obstruction_outline(obstruction_outline);
				}
			}
		}
	}
#endif // CLIPPER_ENABLED
}
