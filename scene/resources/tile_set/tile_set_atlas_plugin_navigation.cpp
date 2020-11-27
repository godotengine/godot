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

/*
void TileSetAtlasPluginNavigation::tilemap_notification(TileMap * p_tile_map, int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
            // Get the parent navigation node
            Node2D *c = this;
            while (c) {
                navigation = Object::cast_to<Navigation2D>(c);
                if (navigation) {
                    break;
                }

                c = Object::cast_to<Node2D>(c->get_parent());
            }

            _update_transform(p_tile_map);
		} break;

		case NOTIFICATION_EXIT_TREE: {
            for (Map<Vector2i, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
                Quadrant &q = E->get();

                if (navigation) {
                    for (Map<Vector2i, Quadrant::NavPoly>::Element *F = q.navpoly_ids.front(); F; F = F->next()) {
                        NavigationServer2D::get_singleton()->region_set_map(F->get().region, RID());
                    }
                    q.navpoly_ids.clear();
                }

                navigation = nullptr;
            }

            _update_transform(p_tile_map);
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
		} break;
	}
}

// --- Called in NOTIFICATION_ENTER_TREE and NOTIFICATION_TRANSFORM_CHANGED ---
void TileSetPLuginNavigation::_update_transform(const TileMap * p_tile_map) {
    if (!p_tile_map->is_inside_tree()) {
        return;
    }

    Transform2D nav_rel;
    if (navigation) {
        nav_rel = p_tile_map->get_relative_transform_to_parent(navigation);
    }

    for (Map<Vector2i, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
        Quadrant &q = E->get();
        if (navigation) {
            for (Map<Vector2i, Quadrant::NavPoly>::Element *F = q.navpoly_ids.front(); F; F = F->next()) {
                NavigationServer2D::get_singleton()->region_set_transform(F->get().region, nav_rel * F->get().xform);
            }
        }
    }
}


// --- Called in update_dirty_quadrants ---
// Compute the transform to the navigation node.
Transform2D nav_rel;
if (navigation) {
    nav_rel = get_relative_transform_to_parent(navigation);
}

// Navigation: get the scene tree for next things.
SceneTree *st = SceneTree::get_singleton();

// Navigation: color for debug.
Color debug_navigation_color;
bool debug_navigation = st && st->is_debugging_navigation_hint();
if (debug_navigation) {
    debug_navigation_color = st->get_debug_navigation_color();
}

while (dirty_quadrant_list.first()) {
    Quadrant &q = *dirty_quadrant_list.first()->self();
    // Navigation: Clear navigation shapes in the quadrant.
    if (navigation) {
        for (Map<Vector2i, Quadrant::NavPoly>::Element *E = q.navpoly_ids.front(); E; E = E->next()) {
            NavigationServer2D::get_singleton()->region_set_map(E->get().region, RID());
        }
        q.navpoly_ids.clear();
    }

    // Navigation: handle navigation shapes.
    if (navigation) {
        // Get the navigation polygon.
        Ref<NavigationPolygon> navpoly;
        Vector2 npoly_ofs;
        if (tile_set->tile_get_tile_mode(c.source_id) == TileSet::AUTO_TILE || tile_set->tile_get_tile_mode(c.source_id) == TileSet::ATLAS_TILE) {
            navpoly = tile_set->autotile_get_navigation_polygon(c.source_id, c.get_tileset_coords());
            npoly_ofs = Vector2();
        } else {
            navpoly = tile_set->tile_get_navigation_polygon(c.source_id);
            npoly_ofs = tile_set->tile_get_navigation_polygon_offset(c.source_id);
        }

        if (navpoly.is_valid()) {
            Transform2D xform;
            xform.set_origin(offset.floor() + q.pos);
            _fix_cell_transform(xform, c, npoly_ofs, s);

            RID region = NavigationServer2D::get_singleton()->region_create();
            NavigationServer2D::get_singleton()->region_set_map(region, navigation->get_rid());
            NavigationServer2D::get_singleton()->region_set_transform(region, nav_rel * xform);
            NavigationServer2D::get_singleton()->region_set_navpoly(region, navpoly);

            Quadrant::NavPoly np;
            np.region = region;
            np.xform = xform;
            q.navpoly_ids[E->key()] = np;

            // Diplay debug info.
            if (debug_navigation) {
                RID debug_navigation_item = vs->canvas_item_create();
                vs->canvas_item_set_parent(debug_navigation_item, canvas_item);
                vs->canvas_item_set_z_as_relative_to_parent(debug_navigation_item, false);
                vs->canvas_item_set_z_index(debug_navigation_item, RS::CANVAS_ITEM_Z_MAX - 2); // Display one below collision debug

                if (debug_navigation_item.is_valid()) {
                    Vector<Vector2> navigation_polygon_vertices = navpoly->get_vertices();
                    int vsize = navigation_polygon_vertices.size();

                    if (vsize > 2) {
                        Vector<Color> colors;
                        Vector<Vector2> vertices;
                        vertices.resize(vsize);
                        colors.resize(vsize);
                        {
                            const Vector2 *vr = navigation_polygon_vertices.ptr();
                            for (int j = 0; j < vsize; j++) {
                                vertices.write[j] = vr[j];
                                colors.write[j] = debug_navigation_color;
                            }
                        }

                        Vector<int> indices;

                        for (int j = 0; j < navpoly->get_polygon_count(); j++) {
                            Vector<int> polygon = navpoly->get_polygon(j);

                            for (int k = 2; k < polygon.size(); k++) {
                                int kofs[3] = { 0, k - 1, k };
                                for (int l = 0; l < 3; l++) {
                                    int idx = polygon[kofs[l]];
                                    ERR_FAIL_INDEX(idx, vsize);
                                    indices.push_back(idx);
                                }
                            }
                        }
                        Transform2D navxform;
                        navxform.set_origin(offset.floor());
                        _fix_cell_transform(navxform, c, npoly_ofs, s);

                        vs->canvas_item_set_transform(debug_navigation_item, navxform);
                        vs->canvas_item_add_triangle_array(debug_navigation_item, indices, vertices, colors);
                    }
                }
            }
        }
    }
}

// --- Called in _erase_quadrant ---
// Navigation: Free the navigation polygons.
if (navigation) {
    for (Map<Vector2i, Quadrant::NavPoly>::Element *E = q.navpoly_ids.front(); E; E = E->next()) {
        NavigationServer2D::get_singleton()->region_set_map(E->get().region, RID());
    }
    q.navpoly_ids.clear();
}
*/
