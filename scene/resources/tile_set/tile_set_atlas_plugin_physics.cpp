/*************************************************************************/
/*  tile_set_atlas_plugin_physics.cpp                                    */
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

#include "tile_set_atlas_plugin_physics.h"

#include "scene/2d/tile_map.h"
#include "scene/resources/convex_polygon_shape_2d.h"

void TileSetAtlasPluginPhysics::tilemap_notification(TileMap *p_tile_map, int p_what) {
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

void TileSetAtlasPluginPhysics::update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
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
			TileMapCell c = p_tile_map->get_cell(E_cell->get());

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					TileData *tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);

					for (int body_index = 0; body_index < q.bodies.size(); body_index++) {
						// Add the shapes again.
						for (int shape_index = 0; shape_index < tile_data->get_collision_shapes_count(body_index); shape_index++) {
							bool one_way_collision = tile_data->is_collision_shape_one_way(body_index, shape_index);
							float one_way_collision_margin = tile_data->get_collision_shape_one_way_margin(body_index, shape_index);
							Ref<Shape2D> shape = tile_data->get_collision_shape_shape(body_index, shape_index);
							if (shape.is_valid()) {
								Transform2D xform = Transform2D();
								xform.set_origin(p_tile_map->map_to_world(E_cell->get()) - quadrant_pos);

								// Add decomposed convex shapes.
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

void TileSetAtlasPluginPhysics::create_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
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

void TileSetAtlasPluginPhysics::cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	// Remove a quadrant.
	for (int body_index = 0; body_index < p_quadrant->bodies.size(); body_index++) {
		PhysicsServer2D::get_singleton()->free(p_quadrant->bodies[body_index]);
	}
	p_quadrant->bodies.clear();
}

void TileSetAtlasPluginPhysics::draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	// Draw the debug collision shapes.
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!p_tile_map->get_tree() || !(Engine::get_singleton()->is_editor_hint() || p_tile_map->get_tree()->is_debugging_collisions_hint())) {
		return;
	}

	RenderingServer *rs = RenderingServer::get_singleton();

	Color debug_collision_color = p_tile_map->get_tree()->get_debug_collisions_color();
	for (Set<Vector2i>::Element *E_cell = p_quadrant->cells.front(); E_cell; E_cell = E_cell->next()) {
		TileMapCell c = p_tile_map->get_cell(E_cell->get());

		Transform2D xform;
		xform.set_origin(p_tile_map->map_to_world(E_cell->get()) - p_tile_map->map_to_world(p_quadrant->coords * p_tile_map->get_effective_quadrant_size()));
		rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, xform);

		if (tile_set->has_source(c.source_id)) {
			TileSetSource *source = *tile_set->get_source(c.source_id);

			if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
				continue;
			}

			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				TileData *tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);

				for (int body_index = 0; body_index < p_quadrant->bodies.size(); body_index++) {
					for (int shape_index = 0; shape_index < tile_data->get_collision_shapes_count(body_index); shape_index++) {
						// Draw the debug shape.
						Ref<Shape2D> shape = tile_data->get_collision_shape_shape(body_index, shape_index);
						if (shape.is_valid()) {
							shape->draw(p_quadrant->debug_canvas_item, debug_collision_color);
						}
					}
				}
			}
		}
		rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, Transform2D());
	}
};
