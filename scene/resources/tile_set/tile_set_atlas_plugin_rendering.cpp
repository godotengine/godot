/*************************************************************************/
/*  tile_set_atlas_plugin_rendering.cpp                                  */
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

#include "tile_set_atlas_plugin_rendering.h"

#include "scene/2d/tile_map.h"

// --- TileSet data ---
bool RenderingTileSetData::set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "y_sorting") {
		y_sorting = p_value;
	} else if (p_name == "uv_clipping") {
		uv_clipping = p_value;
	} /*else if (p_name == "occluder_light_masks") {
			occluder_light_masks = p_value;
	} */
	else {
		return false;
	}
	return true;
};

bool RenderingTileSetData::get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "y_sorting") {
		r_ret = y_sorting;
	} else if (p_name == "uv_clipping") {
		r_ret = uv_clipping;
	} /*else if (p_name == "occluder_light_masks") {
			r_ret = occluder_light_masks;
	} */
	else {
		return false;
	}
	return true;
};

void RenderingTileSetData::get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::BOOL, "y_sorting", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::BOOL, "uv_clipping", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
};

// --- TileData ---
bool RenderingTileData::set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "tex_offset") {
		tex_offset = p_value;
	} else if (p_name == "material") {
		material = p_value;
	} else if (p_name == "modulate") {
		modulate = p_value;
	} else if (p_name == "z_index") {
		z_index = p_value;
	} else if (p_name == "y_sort_origin") {
		y_sort_origin = p_value;
	} /*else if (p_name == "occluders") {
			occluders = p_value;
		} */
	else {
		return false;
	}
	return true;
}

bool RenderingTileData::get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "tex_offset") {
		r_ret = tex_offset;
	} else if (p_name == "material") {
		r_ret = material;
	} else if (p_name == "modulate") {
		r_ret = modulate;
	} else if (p_name == "z_index") {
		r_ret = z_index;
	} else if (p_name == "y_sort_origin") {
		r_ret = y_sort_origin;
	} /*else if (p_name == "occluders") {
			r_ret = occluders;
		} */
	else {
		return false;
	}
	return true;
};

void RenderingTileData::get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::VECTOR2I, "tex_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::COLOR, "modulate", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::INT, "z_index", PROPERTY_HINT_RANGE, itos(RS::CANVAS_ITEM_Z_MIN) + "," + itos(RS::CANVAS_ITEM_Z_MAX) + ",1", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::VECTOR2I, "y_sort_origin", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	//p_list->push_back(PropertyInfo(Variant::OBJECT, "occluders", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
};

// -----
const String TileSetAtlasPluginRendering::NAME = "Rendering";
const String TileSetAtlasPluginRendering::ID = "rendering";

void TileSetAtlasPluginRendering::tilemap_notification(TileMap *p_tile_map, int p_what) {
}

void TileSetAtlasPluginRendering::update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	SelfList<TileMapQuadrant> *q_list_element = r_dirty_quadrant_list.first();
	while (q_list_element) {
		TileMapQuadrant &q = *q_list_element->self();

		// Draw offset.
		RenderingServer *vs = RenderingServer::get_singleton();

		// Free all canvas items in the quadrant.
		for (List<RID>::Element *E = q.canvas_items.front(); E; E = E->next()) {
			vs->free(E->get());
		}

		q.canvas_items.clear();

		Ref<ShaderMaterial> prev_material;
		int prev_z_index = 0;
		RID prev_canvas_item;
		RID prev_debug_canvas_item;

		// Iterate over the cells of the quadrant.
		for (Map<Vector2i, Vector2i, TileMapQuadrant::CoordsWorldComparator>::Element *E = q.world_to_map.front(); E; E = E->next()) {
			TileMapCell &c = p_tile_map->get_cell(E->value());

			if (tile_set->get_source_type(c.source_id) == TileSet::SOURCE_TYPE_ATLAS) {
				TileAtlasSource *atlas = tile_set->get_atlas_source(c.source_id);

				// Check if the tileset has a tile with the given ID, otherwise, ignore it.
				if (!atlas->has_tile(c.get_atlas_coords()) || !atlas->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				// Get the texture.
				Ref<Texture2D> tex = atlas->get_texture();
				if (!tex.is_valid()) {
					continue;
				}

				// Get the material.
				Ref<ShaderMaterial> mat = atlas->get_tile_data(c.get_atlas_coords(), c.alternative_tile)->tile_get_material();

				// Get the Z-index.
				int z_index = atlas->get_tile_data(c.get_atlas_coords(), c.alternative_tile)->tile_get_z_index();

				// Create two canvas items, for rendering and debug.
				RID canvas_item;
				RID debug_canvas_item;

				// Check if the material or the z_index changed.
				if (prev_canvas_item == RID() || prev_material != mat || prev_z_index != z_index) {
					canvas_item = vs->canvas_item_create();
					if (mat.is_valid()) {
						vs->canvas_item_set_material(canvas_item, mat->get_rid());
					}
					vs->canvas_item_set_parent(canvas_item, p_tile_map->get_canvas_item());
					RS::get_singleton()->canvas_item_set_use_parent_material(canvas_item, p_tile_map->get_use_parent_material() || p_tile_map->get_material().is_valid());
					Transform2D xform;
					xform.set_origin(q.pos);
					vs->canvas_item_set_transform(canvas_item, xform);
					vs->canvas_item_set_light_mask(canvas_item, p_tile_map->get_light_mask());
					vs->canvas_item_set_z_index(canvas_item, z_index);

					vs->canvas_item_set_default_texture_filter(canvas_item, RS::CanvasItemTextureFilter(p_tile_map->CanvasItem::get_texture_filter()));
					vs->canvas_item_set_default_texture_repeat(canvas_item, RS::CanvasItemTextureRepeat(p_tile_map->CanvasItem::get_texture_repeat()));

					q.canvas_items.push_back(canvas_item);

					prev_canvas_item = canvas_item;
					prev_material = mat;
					prev_z_index = z_index;

				} else {
					// Keep the same canvas_item to draw on.
					canvas_item = prev_canvas_item;
				}

				// Get the tile region in the tileset, if it is defined.
				Rect2 r = atlas->get_tile_texture_region(c.get_atlas_coords());

				// Get the texture size.
				Size2 s;
				if (r == Rect2()) {
					s = tex->get_size(); // No region, use the full texture.
				} else {
					s = r.size; // Region, use the region size.
				}

				bool transpose = atlas->get_tile_data(c.get_atlas_coords(), c.alternative_tile)->tile_get_transpose();

				// Compute the offset
				Vector2i source_tile_size = atlas->get_tile_texture_region(c.get_atlas_coords()).size;
				Vector2i tile_offset = tile_set->get_tile_effective_texture_offset(c.source_id, c.get_atlas_coords(), c.alternative_tile);
				Vector2 offset = E->key() - source_tile_size / 2 - q.pos - tile_offset;

				// Compute the destination rectangle in the CanvasItem.
				Rect2 rect;
				rect.position = offset.floor();
				rect.size = s;
				rect.size.x += fp_adjust;
				rect.size.y += fp_adjust;
				/*
                if (transpose) {
                    SWAP(tile_ofs.x, tile_ofs.y);
                }

                if (tile_set->tile_get_flip_h(c.source_id, c.get_atlas_coords(), c.alternative_tile)) {
                    rect.size.x = -rect.size.x;
                    tile_ofs.x = -tile_ofs.x;
                }

                if (tile_set->tile_get_flip_v(c.source_id, c.get_atlas_coords(), c.alternative_tile)) {
                    rect.size.y = -rect.size.y;
                    tile_ofs.y = -tile_ofs.y;
                }

                rect.position += tile_ofs;
                */
				// Get the tile modulation.
				Color modulate = atlas->get_tile_data(c.get_atlas_coords(), c.alternative_tile)->tile_get_modulate();
				Color self_modulate = p_tile_map->get_self_modulate();
				modulate = Color(modulate.r * self_modulate.r, modulate.g * self_modulate.g, modulate.b * self_modulate.b, modulate.a * self_modulate.a);

				// Draw the tile.
				if (r == Rect2()) {
					tex->draw_rect(canvas_item, rect, false, modulate, transpose);
				} else {
					tex->draw_rect_region(canvas_item, rect, r, modulate, transpose, tile_set->is_uv_clipping());
				}

				// Change the debug_canvas_item transform ?
				if (debug_canvas_item.is_valid()) {
					vs->canvas_item_add_set_transform(debug_canvas_item, Transform2D());
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

void TileSetAtlasPluginRendering::initialize_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
}

void TileSetAtlasPluginRendering::create_quadrant(TileMap *p_tile_map, const Vector2i &p_quadrant_coords, TileMapQuadrant *p_quadrant) {
	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	p_quadrant->pos = p_tile_map->map_to_world(p_quadrant_coords * p_tile_map->get_effective_quadrant_size()) - tile_set->get_tile_size() / 2; // Quadrant's position in the TileMap's local coords
	quadrant_order_dirty = true;
}

void TileSetAtlasPluginRendering::cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) {
	// Free the canvas item..
	for (List<RID>::Element *E = p_quadrant->canvas_items.front(); E; E = E->next()) {
		RenderingServer::get_singleton()->free(E->get());
	}
	p_quadrant->canvas_items.clear();
}

/*
void TileSetPluginOccluders::tilemap_notification(TileMap * p_tile_map, int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
		} break;

		case NOTIFICATION_EXIT_TREE: {
            for (Map<Vector2i, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
                Quadrant &q = E->get();

                for (Map<Vector2i, Quadrant::Occluder>::Element *F = q.occluder_instances.front(); F; F = F->next()) {
                    RS::get_singleton()->free(F->get().id);
                }
                q.occluder_instances.clear();
            }
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
		} break;
	}
}

void TileSetPluginOccluders::_update_transform(const TileMap * p_tile_map) {
    if (!is_inside_tree()) {
		return;
	}

	Transform2D global_transform = get_global_transform();

    for (Map<Vector2i, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
        Quadrant &q = E->get();

        // Occlusion
        for (Map<Vector2i, Quadrant::Occluder>::Element *F = q.occluder_instances.front(); F; F = F->next()) {
            RS::get_singleton()->canvas_light_occluder_set_transform(F->get().id, global_transform * F->get().xform);
        }
    }
}






// --- Called in update_dirty_quadrants ---
// Occlusion: Clear occlusion shapes in the quadrant.
for (Map<Vector2i, Quadrant::Occluder>::Element *E = q.occluder_instances.front(); E; E = E->next()) {
    RS::get_singleton()->free(E->get().id);
}
q.occluder_instances.clear();

while (dirty_quadrant_list.first()) {
    Quadrant &q = *dirty_quadrant_list.first()->self();

    for (int i = 0; i < q.cells.size(); i++) {
        // Occlusion: handle occluder shape.
        Ref<OccluderPolygon2D> occluder;
        if (tile_set->tile_get_tile_mode(c.source_id) == TileSet::AUTO_TILE || tile_set->tile_get_tile_mode(c.source_id) == TileSet::ATLAS_TILE) {
            occluder = tile_set->autotile_get_light_occluder(c.source_id, c.get_tileset_coords());
        } else {
            occluder = tile_set->tile_get_light_occluder(c.source_id);
        }
        if (occluder.is_valid()) {
            Vector2 occluder_ofs = tile_set->tile_get_occluder_offset(c.source_id);
            Transform2D xform;
            xform.set_origin(offset.floor() + q.pos);
            _fix_cell_transform(xform, c, occluder_ofs, s);

            RID orid = RS::get_singleton()->canvas_light_occluder_create();
            RS::get_singleton()->canvas_light_occluder_set_transform(orid, get_global_transform() * xform);
            RS::get_singleton()->canvas_light_occluder_set_polygon(orid, occluder->get_rid());
            RS::get_singleton()->canvas_light_occluder_attach_to_canvas(orid, get_canvas());
            RS::get_singleton()->canvas_light_occluder_set_light_mask(orid, occluder_light_mask);
            Quadrant::Occluder oc;
            oc.xform = xform;
            oc.id = orid;
            q.occluder_instances[E->key()] = oc;
        }
    }
}



// --- Called in _erase_quadrant ---
// Occlusion: remove occluders
for (Map<Vector2i, Quadrant::Occluder>::Element *E = q.occluder_instances.front(); E; E = E->next()) {
    RS::get_singleton()->free(E->get().id);
}
q.occluder_instances.clear();


// --- Accessors ---
int TileMap::get_occluder_light_mask() const {
	// Occlusion: set light mask.
	return occluder_light_mask;
}

void TileMap::set_occluder_light_mask(int p_mask) {
	// Occlusion: set occluder light mask.
	occluder_light_mask = p_mask;
	for (Map<Vector2i, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		for (Map<Vector2i, Quadrant::Occluder>::Element *F = E->get().occluder_instances.front(); F; F = F->next()) {
			RenderingServer::get_singleton()->canvas_light_occluder_set_light_mask(F->get().id, occluder_light_mask);
		}
	}
}
*/
