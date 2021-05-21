/*************************************************************************/
/*  tile_data_editors.cpp                                                */
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

#include "tile_data_editors.h"

#include "tile_set_editor.h"

TileData *TileDataEditor::_get_tile_data(TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile) {
	ERR_FAIL_COND_V(!p_tile_set, nullptr);
	ERR_FAIL_COND_V(!p_tile_set->has_source(p_atlas_source_id), nullptr);

	TileData *td = nullptr;
	TileSetSource *source = *p_tile_set->get_source(p_atlas_source_id);
	TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
	if (atlas_source) {
		ERR_FAIL_COND_V(!atlas_source->has_tile(p_atlas_coords), nullptr);
		ERR_FAIL_COND_V(!atlas_source->has_alternative_tile(p_atlas_coords, p_alternative_tile), nullptr);
		td = Object::cast_to<TileData>(atlas_source->get_tile_data(p_atlas_coords, p_alternative_tile));
	}

	return td;
}

void TileDataEditor::edit(TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) {
}

void TileDataTextureOffsetEditor::draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) {
	TileData *tile_data = _get_tile_data(p_tile_set, p_atlas_source_id, p_atlas_coords, p_alternative_tile);
	ERR_FAIL_COND(!tile_data);

	bool valid;
	Variant value = tile_data->get(p_property, &valid);
	if (!valid) {
		return;
	}
	ERR_FAIL_COND(value.get_type() != Variant::VECTOR2I);

	Vector2i tile_set_tile_size = p_tile_set->get_tile_size();
	Rect2i rect = Rect2i(-tile_set_tile_size / 2, tile_set_tile_size);
	p_tile_set->draw_tile_shape(p_canvas_item, p_transform.xform(rect), Color(1.0, 0.0, 0.0));
}

void TileDataIntegerEditor::draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) {
	TileData *tile_data = _get_tile_data(p_tile_set, p_atlas_source_id, p_atlas_coords, p_alternative_tile);
	ERR_FAIL_COND(!tile_data);

	bool valid;
	Variant value = tile_data->get(p_property, &valid);
	if (!valid) {
		return;
	}
	ERR_FAIL_COND(value.get_type() != Variant::INT);

	Ref<Font> font = TileSetEditor::get_singleton()->get_theme_font("bold", "EditorFonts");
	int height = font->get_height();
	int width = 200;
	p_canvas_item->draw_string(font, p_transform.get_origin() + Vector2i(-width / 2, height / 2), vformat("%d", value), HALIGN_CENTER, width, -1, Color(1, 1, 1), 1, Color(0, 0, 0, 1));
}

void TileDataFloatEditor::draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) {
	TileData *tile_data = _get_tile_data(p_tile_set, p_atlas_source_id, p_atlas_coords, p_alternative_tile);
	ERR_FAIL_COND(!tile_data);

	bool valid;
	Variant value = tile_data->get(p_property, &valid);
	if (!valid) {
		return;
	}
	ERR_FAIL_COND(value.get_type() != Variant::FLOAT);

	Ref<Font> font = TileSetEditor::get_singleton()->get_theme_font("bold", "EditorFonts");
	int height = font->get_height();
	int width = 200;
	p_canvas_item->draw_string(font, p_transform.get_origin() + Vector2i(-width / 2, height / 2), vformat("%.2f", value), HALIGN_CENTER, width, -1, Color(1, 1, 1), 1, Color(0, 0, 0, 1));
}

void TileDataPositionEditor::draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) {
	TileData *tile_data = _get_tile_data(p_tile_set, p_atlas_source_id, p_atlas_coords, p_alternative_tile);
	ERR_FAIL_COND(!tile_data);

	bool valid;
	Variant value = tile_data->get(p_property, &valid);
	if (!valid) {
		return;
	}
	ERR_FAIL_COND(value.get_type() != Variant::VECTOR2I && value.get_type() != Variant::VECTOR2);

	Ref<Texture2D> position_icon = TileSetEditor::get_singleton()->get_theme_icon("EditorPosition", "EditorIcons");
	p_canvas_item->draw_texture(position_icon, p_transform.xform(Vector2(value)) - position_icon->get_size() / 2);
}

void TileDataYSortEditor::draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) {
	TileData *tile_data = _get_tile_data(p_tile_set, p_atlas_source_id, p_atlas_coords, p_alternative_tile);
	ERR_FAIL_COND(!tile_data);

	bool valid;
	Variant value = tile_data->get(p_property, &valid);
	if (!valid) {
		return;
	}
	ERR_FAIL_COND(value.get_type() != Variant::INT);

	Ref<Texture2D> position_icon = TileSetEditor::get_singleton()->get_theme_icon("EditorPosition", "EditorIcons");
	p_canvas_item->draw_texture(position_icon, p_transform.xform(Vector2(0, value)) - position_icon->get_size() / 2);
}

void TileDataOcclusionShapeEditor::draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) {
	TileData *tile_data = _get_tile_data(p_tile_set, p_atlas_source_id, p_atlas_coords, p_alternative_tile);
	ERR_FAIL_COND(!tile_data);

	Vector<String> components = String(p_property).split("/", true);
	if (components[0].begins_with("occlusion_layer_") && components[0].trim_prefix("occlusion_layer_").is_valid_integer()) {
		int occlusion_layer = components[0].trim_prefix("occlusion_layer_").to_int();
		if (occlusion_layer >= 0 && occlusion_layer < p_tile_set->get_occlusion_layers_count()) {
			// Draw all shapes.
			Vector<Color> debug_occlusion_color;
			debug_occlusion_color.push_back(Color(0.5, 0, 0, 0.6));

			RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), p_transform);
			Ref<OccluderPolygon2D> occluder = tile_data->get_occluder(occlusion_layer);
			if (occluder.is_valid() && occluder->get_polygon().size() >= 3) {
				p_canvas_item->draw_polygon(Variant(occluder->get_polygon()), debug_occlusion_color);
			}
			RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), Transform2D());
		}
	}
}

void TileDataCollisionShapeEditor::draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) {
	TileData *tile_data = _get_tile_data(p_tile_set, p_atlas_source_id, p_atlas_coords, p_alternative_tile);
	ERR_FAIL_COND(!tile_data);

	Vector<String> components = String(p_property).split("/", true);
	if (components[0].begins_with("physics_layer_") && components[0].trim_prefix("physics_layer_").is_valid_integer()) {
		int physics_layer = components[0].trim_prefix("physics_layer_").to_int();
		if (physics_layer >= 0 && physics_layer < p_tile_set->get_physics_layers_count()) {
			// Draw all shapes.
			Color debug_collision_color = p_canvas_item->get_tree()->get_debug_collisions_color();
			RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), p_transform);
			for (int i = 0; i < tile_data->get_collision_shapes_count(physics_layer); i++) {
				Ref<Shape2D> shape = tile_data->get_collision_shape_shape(physics_layer, i);
				if (shape.is_valid()) {
					shape->draw(p_canvas_item->get_canvas_item(), debug_collision_color);
				}
			}
			RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), Transform2D());
		}
	}
}

void TileDataTerrainsEditor::draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) {
	TileData *tile_data = _get_tile_data(p_tile_set, p_atlas_source_id, p_atlas_coords, p_alternative_tile);
	ERR_FAIL_COND(!tile_data);

	Vector<String> components = String(p_property).split("/", true);
	if (components[0] == "terrain_mode" || components[0] == "terrain" || components[0] == "terrains_peering_bit") {
		TileSetPluginAtlasTerrain::draw_terrains(p_canvas_item, p_transform, p_tile_set, tile_data);
	}
}

void TileDataNavigationPolygonEditor::draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) {
	TileData *tile_data = _get_tile_data(p_tile_set, p_atlas_source_id, p_atlas_coords, p_alternative_tile);
	ERR_FAIL_COND(!tile_data);

	Vector<String> components = String(p_property).split("/", true);
	if (components[0].begins_with("navigation_layer_") && components[0].trim_prefix("navigation_layer_").is_valid_integer()) {
		int navigation_layer = components[0].trim_prefix("navigation_layer_").to_int();
		if (navigation_layer >= 0 && navigation_layer < p_tile_set->get_navigation_layers_count()) {
			// Draw all shapes.
			RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), p_transform);

			Ref<NavigationPolygon> navigation_polygon = tile_data->get_navigation_polygon(navigation_layer);
			if (navigation_polygon.is_valid()) {
				Vector<Vector2> verts = navigation_polygon->get_vertices();
				if (verts.size() < 3) {
					return;
				}

				Color color = p_canvas_item->get_tree()->get_debug_navigation_color();

				RandomPCG rand;
				for (int i = 0; i < navigation_polygon->get_polygon_count(); i++) {
					// An array of vertices for this polygon.
					Vector<int> polygon = navigation_polygon->get_polygon(i);
					Vector<Vector2> vertices;
					vertices.resize(polygon.size());
					for (int j = 0; j < polygon.size(); j++) {
						ERR_FAIL_INDEX(polygon[j], verts.size());
						vertices.write[j] = verts[polygon[j]];
					}

					// Generate the polygon color, slightly randomly modified from the settings one.
					Color random_variation_color;
					random_variation_color.set_hsv(color.get_h() + rand.random(-1.0, 1.0) * 0.05, color.get_s(), color.get_v() + rand.random(-1.0, 1.0) * 0.1);
					random_variation_color.a = color.a;
					Vector<Color> colors;
					colors.push_back(random_variation_color);

					RenderingServer::get_singleton()->canvas_item_add_polygon(p_canvas_item->get_canvas_item(), vertices, colors);
				}
			}

			RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), Transform2D());
		}
	}
}
