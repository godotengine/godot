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

bool RenderingTileData::set(const StringName &p_name, const Variant &p_value) {
	return false;
}

bool RenderingTileData::get(const StringName &p_name, Variant &r_ret) const {
	return false;
}

void RenderingTileData::get_property_list(List<PropertyInfo> *p_list) const {
}

// -----
const String TileSetAtlasPluginRendering::NAME = "Rendering";
const String TileSetAtlasPluginRendering::ID = "rendering";

void TileSetAtlasPluginRendering::tilemap_notification(TileMap *p_tile_map, int p_what) {
}

void TileSetAtlasPluginRendering::draw_tile(RID p_canvas_item, Vector2i p_position, const Ref<TileSet> p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, Color p_modulation) {
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

		// Get tile data.
		TileData *tile_data = atlas_source->get_tile_data(p_atlas_coords, p_alternative_tile);

		// Compute the offset
		Rect2i source_rect = atlas_source->get_tile_texture_region(p_atlas_coords);
		Vector2i tile_offset = p_tile_set->get_tile_effective_texture_offset(p_atlas_source_id, p_atlas_coords, p_alternative_tile);

		// Compute the destination rectangle in the CanvasItem.
		Rect2 dest_rect;
		dest_rect.size = source_rect.size;
		dest_rect.size.x += fp_adjust;
		dest_rect.size.y += fp_adjust;

		bool transpose = tile_data->tile_get_transpose();
		if (transpose) {
			dest_rect.position = (p_position - Vector2(dest_rect.size.y, dest_rect.size.x) / 2 - tile_offset);
		} else {
			dest_rect.position = (p_position - dest_rect.size / 2 - tile_offset);
		}

		if (tile_data->tile_get_flip_h()) {
			dest_rect.size.x = -dest_rect.size.x;
		}

		if (tile_data->tile_get_flip_v()) {
			dest_rect.size.y = -dest_rect.size.y;
		}

		// Get the tile modulation.
		Color modulate = tile_data->tile_get_modulate();
		modulate = Color(modulate.r * p_modulation.r, modulate.g * p_modulation.g, modulate.b * p_modulation.b, modulate.a * p_modulation.a);

		// Draw the tile.
		tex->draw_rect_region(p_canvas_item, dest_rect, source_rect, modulate, transpose, p_tile_set->is_uv_clipping());
	}
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
			TileMapCell c = p_tile_map->get_cell(E->value());

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					// Get the tile data.
					TileData *tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);
					Ref<ShaderMaterial> mat = tile_data->tile_get_material();
					int z_index = tile_data->tile_get_z_index();

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

					// Drawing the tile in the canvas item.
					draw_tile(canvas_item, E->key() - q.pos, tile_set, c.source_id, c.get_atlas_coords(), c.alternative_tile, p_tile_map->get_self_modulate());

					// Change the debug_canvas_item transform ?
					if (debug_canvas_item.is_valid()) {
						vs->canvas_item_add_set_transform(debug_canvas_item, Transform2D());
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
