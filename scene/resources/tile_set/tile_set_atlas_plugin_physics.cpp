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

/*
// Get the list of tileset layers supported by a tileset plugin
virtual int get_layer_type_count() { return 0; };
virtual String get_layer_type_name(int p_id) { return ""; };
virtual String get_layer_type_icon(int p_id) { return ""; };
virtual bool get_layer_type_multiple_mode(int p_id) { return true; };



void TileSetAtlasPluginPhysics::_update_state(const TileMap * p_tile_map) {
    if (!p_tile_map->is_inside_tree()) {
        return;
    }

    Map<Vector2i, Quadrant> * quadrant_map = p_tile_map->get_quadrant_map();
    Transform2D global_transform = p_tile_map->get_global_transform();

    for (Map<Vector2i, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
        Quadrant &q = E->get();

        Transform2D xform;
        xform.set_origin(q.pos);
        xform = global_transform * xform;

        PhysicsServer2D::get_singleton()->body_set_state(q.body, PhysicsServer2D::BODY_STATE_TRANSFORM, xform);
    }
}

void TileSetAtlasPluginPhysics::tilemap_notification(TileMap * p_tile_map, int p_what) {
    Map<Vector2i, Quadrant> * quadrant_map = p_tile_map->get_quadrant_map();


	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
            RID space = p_tile_map->get_world_2d()->get_space();
            for (Map<Vector2i, Quadrant>::Element *E = quadrant_map->front(); E; E = E->next()) {
                Quadrant &q = E->get();
                PhysicsServer2D::get_singleton()->body_set_space(q.body, space);
            }
            _update_state(p_tile_map);
		} break;

		case NOTIFICATION_EXIT_TREE: {
            for (Map<Vector2i, Quadrant>::Element *E = quadrant_map->front(); E; E = E->next()) {
                Quadrant &q = E->get();
                PhysicsServer2D::get_singleton()->body_set_space(q.body, RID());
            }
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
            _update_state(p_tile_map);
		} break;
	}
}

void TileSetAtlasPluginPhysics::_add_shape(int &shape_idx, const Quadrant &p_q, const Ref<Shape2D> &p_shape, const TileSet::ShapeData &p_shape_data, const Transform2D &p_xform, const Vector2 &p_metadata) {
	// Collisions: add a collision shape
	PhysicsServer2D *ps = PhysicsServer2D::get_singleton();

	ps->body_add_shape(p_q.body, p_shape->get_rid(), p_xform);
	ps->body_set_shape_metadata(p_q.body, shape_idx, p_metadata);
	ps->body_set_shape_as_one_way_collision(p_q.body, shape_idx, p_shape_data.one_way_collision, p_shape_data.one_way_collision_margin);

	shape_idx++;
}

void TileSetAtlasPluginPhysics::update_dirty_quadrants(TileMap * p_tile_map) {
    // Collision: get the physics server.
    PhysicsServer2D *ps = PhysicsServer2D::get_singleton();

    // Collisions and Navigation: get the scene tree for next things.
    SceneTree *st = SceneTree::get_singleton();

    // Collisions: color for debug.
    Color debug_collision_color;
    bool debug_shapes = st && st->is_debugging_collisions_hint();
    if (debug_shapes) {
        debug_collision_color = st->get_debug_collisions_color();
    }

    while (dirty_quadrant_list.first()) { // Keep in tilemap
        // Collisions: Clear shapes in the quadrant.
        ps->body_clear_shapes(q.body);

        int shape_idx = 0;
        for (int i = 0; i < q.cells.size(); i++) {
            // Collisions: get the tileset collision shapes, and add the shape.
            Vector<TileSet::ShapeData> shapes = tile_set->tile_get_shapes(c.source_id);
            for (int j = 0; j < shapes.size(); j++) {
                Ref<Shape2D> shape = shapes[j].shape;
                if (shape.is_valid()) {
                    if (tile_set->tile_get_tile_mode(c.source_id) == TileSet::SINGLE_TILE || (shapes[j].autotile_coord.x == c.coord_x && shapes[j].autotile_coord.y == c.coord_y)) {
                        Transform2D xform;
                        xform.set_origin(offset.floor());

                        Vector2 shape_ofs = shapes[j].shape_transform.get_origin();

                        _fix_cell_transform(xform, c, shape_ofs, s);

                        xform *= shapes[j].shape_transform.untranslated();

                        // Draw the debug shape.
                        if (debug_canvas_item.is_valid()) {
                            vs->canvas_item_add_set_transform(debug_canvas_item, xform);
                            shape->draw(debug_canvas_item, debug_collision_color);
                        }

                        // Add the shape from the one in the tileset.
                        if (shape->has_meta("decomposed")) {
                            Array _shapes = shape->get_meta("decomposed");
                            for (int k = 0; k < _shapes.size(); k++) {
                                Ref<ConvexPolygonShape2D> convex = _shapes[k];
                                if (convex.is_valid()) {
                                    _add_shape(shape_idx, q, convex, shapes[j], xform, Vector2(E->key().x, E->key().y));
    #ifdef DEBUG_ENABLED
                                } else {
                                    print_error("The TileSet assigned to the TileMap " + get_name() + " has an invalid convex shape.");
    #endif
                                }
                            }
                        } else {
                            _add_shape(shape_idx, q, shape, shapes[j], xform, Vector2(E->key().x, E->key().y));
                        }
                    }
                }
            }

            // Create two canvas items, for rendering and debug.
            RID debug_canvas_item;

            // Check if the material or the z_index changed.
            if (prev_canvas_item == RID() || prev_material != mat || prev_z_index != z_index) {

                // Debug canvas item, drawn on top of the normal one.
                if (debug_shapes) {
                    debug_canvas_item = vs->canvas_item_create();
                    vs->canvas_item_set_parent(debug_canvas_item, canvas_item);
                    vs->canvas_item_set_z_as_relative_to_parent(debug_canvas_item, false);
                    vs->canvas_item_set_z_index(debug_canvas_item, RS::CANVAS_ITEM_Z_MAX - 1);
                    q.canvas_items.push_back(debug_canvas_item);
                    prev_debug_canvas_item = debug_canvas_item;
                }
                prev_canvas_item = canvas_item;
                prev_material = mat;
                prev_z_index = z_index;

            } else {
                // Keep the same canvas_item to draw on.
                canvas_item = prev_canvas_item;
                if (debug_shapes) {
                    debug_canvas_item = prev_debug_canvas_item;
                }
            }
        }
    }
}

void TileSetAtlasPluginPhysics::initialize_quadrant(TileMap * p_tile_map, Quadrant * p_quadrant) {
    // Create the body and set its parameters.
    p_quadrant.body = PhysicsServer2D::get_singleton()->body_create();
    PhysicsServer2D::get_singleton()->body_set_mode(p_quadrant.body, use_kinematic ? PhysicsServer2D::BODY_MODE_KINEMATIC : PhysicsServer2D::BODY_MODE_STATIC);

    PhysicsServer2D::get_singleton()->body_attach_object_instance_id(p_quadrant.body, get_instance_id());
    PhysicsServer2D::get_singleton()->body_set_collision_layer(p_quadrant.body, collision_layer);
    PhysicsServer2D::get_singleton()->body_set_collision_mask(p_quadrant.body, collision_mask);
    PhysicsServer2D::get_singleton()->body_set_param(p_quadrant.body, PhysicsServer2D::BODY_PARAM_FRICTION, friction);
    PhysicsServer2D::get_singleton()->body_set_param(p_quadrant.body, PhysicsServer2D::BODY_PARAM_BOUNCE, bounce);

    if (is_inside_tree()) {
        xform = get_global_transform() * xform;
        RID space = get_world_2d()->get_space();
        PhysicsServer2D::get_singleton()->body_set_space(p_quadrant.body, space);
    }

    PhysicsServer2D::get_singleton()->body_set_state(p_quadrant.body, PhysicsServer2D::BODY_STATE_TRANSFORM, xform);
}

void TileSetAtlasPluginPhysics::cleanup_quadrant(TileMap * p_tile_map, Quadrant * p_quadrant) {
    // Remove a quadrant.
    PhysicsServer2D::get_singleton()->free(p_quadrant.body);
}


















// --- Accessors ---
uint32_t TileMap::get_collision_layer() const {
	// Collisions: Get collision layer.
	return collision_layer;
}

void TileMap::set_collision_layer(uint32_t p_layer) {
	// Collisions: Set collision layer.
	collision_layer = p_layer;
	for (Map<Vector2i, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		Quadrant &q = E->get();
		PhysicsServer2D::get_singleton()->body_set_collision_layer(q.body, collision_layer);
	}
}

uint32_t TileMap::get_collision_mask() const {
	// Collisions: Get collision mask.
	return collision_mask;
}

void TileMap::set_collision_mask(uint32_t p_mask) {
	// Collisions: Set collision mask.
	collision_mask = p_mask;

	for (Map<Vector2i, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		Quadrant &q = E->get();
		PhysicsServer2D::get_singleton()->body_set_collision_mask(q.body, collision_mask);
	}
}
bool TileMap::get_collision_layer_bit(int p_bit) const {
	// Collisions: Get collision layer bit.
	return get_collision_layer() & (1 << p_bit);
}

void TileMap::set_collision_layer_bit(int p_bit, bool p_value) {
	// Collisions: Set collision layer bit.
	uint32_t layer = get_collision_layer();
	if (p_value) {
		layer |= 1 << p_bit;
	} else {
		layer &= ~(1 << p_bit);
	}
	set_collision_layer(layer);
}

bool TileMap::get_collision_mask_bit(int p_bit) const {
	// Collisions: Get collision mask bit.
	return get_collision_mask() & (1 << p_bit);
}

void TileMap::set_collision_mask_bit(int p_bit, bool p_value) {
	// Collisions: Set collision mask bit.
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << p_bit;
	} else {
		mask &= ~(1 << p_bit);
	}
	set_collision_mask(mask);
}

bool TileMap::get_collision_use_kinematic() const {
	// Collisions: get kinematic.
	return use_kinematic;
}

void TileMap::set_collision_use_kinematic(bool p_use_kinematic) {
	// Collisions: set kinematic.
	_clear_quadrants();
	use_kinematic = p_use_kinematic;
	_recreate_quadrants();
}

float TileMap::get_collision_friction() const {
	// Collisions: get friction.
	return friction;
}

void TileMap::set_collision_friction(float p_friction) {
	// Collisions: set friction.
	friction = p_friction;
	for (Map<Vector2i, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		Quadrant &q = E->get();
		PhysicsServer2D::get_singleton()->body_set_param(q.body, PhysicsServer2D::BODY_PARAM_FRICTION, p_friction);
	}
}

float TileMap::get_collision_bounce() const {
	// Collisions: Get bounce.
	return bounce;
}

void TileMap::set_collision_bounce(float p_bounce) {
	// Collisions: set bounce.
	bounce = p_bounce;
	for (Map<Vector2i, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		Quadrant &q = E->get();
		PhysicsServer2D::get_singleton()->body_set_param(q.body, PhysicsServer2D::BODY_PARAM_BOUNCE, p_bounce);
	}
}*/
