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
	p_canvas_item->draw_texture(position_icon, p_transform.get_origin() + Vector2(value) - position_icon->get_size() / 2);
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

void TileDataTerrainsEditor::_draw_square_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileData::TerrainPeeringBit p_bit) {
	Rect2 bit_rect;
	bit_rect.size = Vector2(p_size) / 3;
	switch (p_bit) {
		case TileData::TERRAIN_PEERING_BIT_RIGHT_SIDE:
			bit_rect.position = Vector2(1, -1);
			break;
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER:
			bit_rect.position = Vector2(1, 1);
			break;
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_SIDE:
			bit_rect.position = Vector2(-1, 1);
			break;
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER:
			bit_rect.position = Vector2(-3, 1);
			break;
		case TileData::TERRAIN_PEERING_BIT_LEFT_SIDE:
			bit_rect.position = Vector2(-3, -1);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER:
			bit_rect.position = Vector2(-3, -3);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_SIDE:
			bit_rect.position = Vector2(-1, -3);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER:
			bit_rect.position = Vector2(1, -3);
			break;
		default:
			break;
	}
	bit_rect.position *= Vector2(p_size) / 6.0;
	p_canvas_item->draw_rect(bit_rect, p_color);
}

void TileDataTerrainsEditor::_draw_square_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileData::TerrainPeeringBit p_bit) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	Vector2 unit = Vector2(p_size) / 6.0;
	PackedVector2Array polygon;
	switch (p_bit) {
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER:
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(3, 3) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(1, 1) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER:
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(-3, 3) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-1, 1) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER:
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(-3, -3) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-1, -1) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER:
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(3, -3) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(1, -1) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		default:
			break;
	}
	if (!polygon.is_empty()) {
		p_canvas_item->draw_polygon(polygon, color_array);
	}
}

void TileDataTerrainsEditor::_draw_square_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileData::TerrainPeeringBit p_bit) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	Vector2 unit = Vector2(p_size) / 6.0;
	PackedVector2Array polygon;
	switch (p_bit) {
		case TileData::TERRAIN_PEERING_BIT_RIGHT_SIDE:
			polygon.push_back(Vector2(1, -1) * unit);
			polygon.push_back(Vector2(3, -3) * unit);
			polygon.push_back(Vector2(3, 3) * unit);
			polygon.push_back(Vector2(1, 1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_SIDE:
			polygon.push_back(Vector2(-1, 1) * unit);
			polygon.push_back(Vector2(-3, 3) * unit);
			polygon.push_back(Vector2(3, 3) * unit);
			polygon.push_back(Vector2(1, 1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_LEFT_SIDE:
			polygon.push_back(Vector2(-1, -1) * unit);
			polygon.push_back(Vector2(-3, -3) * unit);
			polygon.push_back(Vector2(-3, 3) * unit);
			polygon.push_back(Vector2(-1, 1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_SIDE:
			polygon.push_back(Vector2(-1, -1) * unit);
			polygon.push_back(Vector2(-3, -3) * unit);
			polygon.push_back(Vector2(3, -3) * unit);
			polygon.push_back(Vector2(1, -1) * unit);
			break;
		default:
			break;
	}
	if (!polygon.is_empty()) {
		p_canvas_item->draw_polygon(polygon, color_array);
	}
}

void TileDataTerrainsEditor::_draw_isometric_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileData::TerrainPeeringBit p_bit) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	Vector2 unit = Vector2(p_size) / 6.0;
	PackedVector2Array polygon;
	switch (p_bit) {
		case TileData::TERRAIN_PEERING_BIT_RIGHT_CORNER:
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(2, -1) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(2, 1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE:
			polygon.push_back(Vector2(0, 1) * unit);
			polygon.push_back(Vector2(1, 2) * unit);
			polygon.push_back(Vector2(2, 1) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_CORNER:
			polygon.push_back(Vector2(0, 1) * unit);
			polygon.push_back(Vector2(-1, 2) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(1, 2) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE:
			polygon.push_back(Vector2(0, 1) * unit);
			polygon.push_back(Vector2(-1, 2) * unit);
			polygon.push_back(Vector2(-2, 1) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_LEFT_CORNER:
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-2, -1) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-2, 1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE:
			polygon.push_back(Vector2(0, -1) * unit);
			polygon.push_back(Vector2(-1, -2) * unit);
			polygon.push_back(Vector2(-2, -1) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_CORNER:
			polygon.push_back(Vector2(0, -1) * unit);
			polygon.push_back(Vector2(-1, -2) * unit);
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(1, -2) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE:
			polygon.push_back(Vector2(0, -1) * unit);
			polygon.push_back(Vector2(1, -2) * unit);
			polygon.push_back(Vector2(2, -1) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			break;
		default:
			break;
	}
	if (!polygon.is_empty()) {
		p_canvas_item->draw_polygon(polygon, color_array);
	}
}

void TileDataTerrainsEditor::_draw_isometric_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileData::TerrainPeeringBit p_bit) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	Vector2 unit = Vector2(p_size) / 6.0;
	PackedVector2Array polygon;
	switch (p_bit) {
		case TileData::TERRAIN_PEERING_BIT_RIGHT_CORNER:
			polygon.push_back(Vector2(0.5, -0.5) * unit);
			polygon.push_back(Vector2(1.5, -1.5) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(1.5, 1.5) * unit);
			polygon.push_back(Vector2(0.5, 0.5) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_CORNER:
			polygon.push_back(Vector2(-0.5, 0.5) * unit);
			polygon.push_back(Vector2(-1.5, 1.5) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(1.5, 1.5) * unit);
			polygon.push_back(Vector2(0.5, 0.5) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_LEFT_CORNER:
			polygon.push_back(Vector2(-0.5, -0.5) * unit);
			polygon.push_back(Vector2(-1.5, -1.5) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-1.5, 1.5) * unit);
			polygon.push_back(Vector2(-0.5, 0.5) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_CORNER:
			polygon.push_back(Vector2(-0.5, -0.5) * unit);
			polygon.push_back(Vector2(-1.5, -1.5) * unit);
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(1.5, -1.5) * unit);
			polygon.push_back(Vector2(0.5, -0.5) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		default:
			break;
	}
	if (!polygon.is_empty()) {
		p_canvas_item->draw_polygon(polygon, color_array);
	}
}

void TileDataTerrainsEditor::_draw_isometric_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileData::TerrainPeeringBit p_bit) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	Vector2 unit = Vector2(p_size) / 6.0;
	PackedVector2Array polygon;
	switch (p_bit) {
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE:
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE:
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE:
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE:
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		default:
			break;
	}
	if (!polygon.is_empty()) {
		p_canvas_item->draw_polygon(polygon, color_array);
	}
}

void TileDataTerrainsEditor::_draw_half_offset_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileData::TerrainPeeringBit p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	PackedVector2Array point_list;
	point_list.push_back(Vector2(3, (3.0 * (1.0 - p_overlap * 2.0)) / 2.0));
	point_list.push_back(Vector2(3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(2, 3.0 * (1.0 - (p_overlap * 2.0) * 2.0 / 3.0)));
	point_list.push_back(Vector2(1, 3.0 - p_overlap * 2.0));
	point_list.push_back(Vector2(0, 3));
	point_list.push_back(Vector2(-1, 3.0 - p_overlap * 2.0));
	point_list.push_back(Vector2(-2, 3.0 * (1.0 - (p_overlap * 2.0) * 2.0 / 3.0)));
	point_list.push_back(Vector2(-3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(-3, (3.0 * (1.0 - p_overlap * 2.0)) / 2.0));
	point_list.push_back(Vector2(-3, -(3.0 * (1.0 - p_overlap * 2.0)) / 2.0));
	point_list.push_back(Vector2(-3, -3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(-2, -3.0 * (1.0 - (p_overlap * 2.0) * 2.0 / 3.0)));
	point_list.push_back(Vector2(-1, -(3.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(0, -3));
	point_list.push_back(Vector2(1, -(3.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(2, -3.0 * (1.0 - (p_overlap * 2.0) * 2.0 / 3.0)));
	point_list.push_back(Vector2(3, -3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(3, -(3.0 * (1.0 - p_overlap * 2.0)) / 2.0));

	Vector2 unit = Vector2(p_size) / 6.0;
	for (int i = 0; i < point_list.size(); i++) {
		point_list.write[i] = point_list[i] * unit;
	}

	PackedVector2Array polygon;
	if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
		switch (p_bit) {
			case TileData::TERRAIN_PEERING_BIT_RIGHT_SIDE:
				polygon.push_back(point_list[17]);
				polygon.push_back(point_list[0]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_CORNER:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[6]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileData::TERRAIN_PEERING_BIT_LEFT_SIDE:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER:
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				polygon.push_back(point_list[11]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE:
				polygon.push_back(point_list[11]);
				polygon.push_back(point_list[12]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_CORNER:
				polygon.push_back(point_list[12]);
				polygon.push_back(point_list[13]);
				polygon.push_back(point_list[14]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE:
				polygon.push_back(point_list[14]);
				polygon.push_back(point_list[15]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER:
				polygon.push_back(point_list[15]);
				polygon.push_back(point_list[16]);
				polygon.push_back(point_list[17]);
				break;
			default:
				break;
		}
	} else {
		if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
			for (int i = 0; i < point_list.size(); i++) {
				point_list.write[i] = Vector2(point_list[i].y, point_list[i].x);
			}
		}
		switch (p_bit) {
			case TileData::TERRAIN_PEERING_BIT_RIGHT_CORNER:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_SIDE:
				polygon.push_back(point_list[17]);
				polygon.push_back(point_list[0]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[15]);
				polygon.push_back(point_list[16]);
				polygon.push_back(point_list[17]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[14]);
				polygon.push_back(point_list[15]);
				break;
			case TileData::TERRAIN_PEERING_BIT_LEFT_CORNER:
				polygon.push_back(point_list[12]);
				polygon.push_back(point_list[13]);
				polygon.push_back(point_list[14]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE:
				polygon.push_back(point_list[11]);
				polygon.push_back(point_list[12]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER:
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				polygon.push_back(point_list[11]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_SIDE:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE:
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[6]);
				break;
			default:
				break;
		}
	}

	int half_polygon_size = polygon.size();
	for (int i = 0; i < half_polygon_size; i++) {
		polygon.push_back(polygon[half_polygon_size - 1 - i] / 3.0);
	}

	if (!polygon.is_empty()) {
		p_canvas_item->draw_polygon(polygon, color_array);
	}
}

void TileDataTerrainsEditor::_draw_half_offset_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileData::TerrainPeeringBit p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	PackedVector2Array point_list;
	point_list.push_back(Vector2(3, 0));
	point_list.push_back(Vector2(3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(1.5, (3.0 * (1.0 - p_overlap * 2.0) + 3.0) / 2.0));
	point_list.push_back(Vector2(0, 3));
	point_list.push_back(Vector2(-1.5, (3.0 * (1.0 - p_overlap * 2.0) + 3.0) / 2.0));
	point_list.push_back(Vector2(-3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(-3, 0));
	point_list.push_back(Vector2(-3, -3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(-1.5, -(3.0 * (1.0 - p_overlap * 2.0) + 3.0) / 2.0));
	point_list.push_back(Vector2(0, -3));
	point_list.push_back(Vector2(1.5, -(3.0 * (1.0 - p_overlap * 2.0) + 3.0) / 2.0));
	point_list.push_back(Vector2(3, -3.0 * (1.0 - p_overlap * 2.0)));

	Vector2 unit = Vector2(p_size) / 6.0;
	for (int i = 0; i < point_list.size(); i++) {
		point_list.write[i] = point_list[i] * unit;
	}

	PackedVector2Array polygon;
	if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
		switch (p_bit) {
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_CORNER:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[6]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_CORNER:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER:
				polygon.push_back(point_list[10]);
				polygon.push_back(point_list[11]);
				polygon.push_back(point_list[0]);
				break;
			default:
				break;
		}
	} else {
		if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
			for (int i = 0; i < point_list.size(); i++) {
				point_list.write[i] = Vector2(point_list[i].y, point_list[i].x);
			}
		}
		switch (p_bit) {
			case TileData::TERRAIN_PEERING_BIT_RIGHT_CORNER:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[10]);
				polygon.push_back(point_list[11]);
				polygon.push_back(point_list[0]);
				break;
			case TileData::TERRAIN_PEERING_BIT_LEFT_CORNER:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER:
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[6]);
				break;
			default:
				break;
		}
	}

	int half_polygon_size = polygon.size();
	for (int i = 0; i < half_polygon_size; i++) {
		polygon.push_back(polygon[half_polygon_size - 1 - i] / 3.0);
	}

	if (!polygon.is_empty()) {
		p_canvas_item->draw_polygon(polygon, color_array);
	}
}

void TileDataTerrainsEditor::_draw_half_offset_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileData::TerrainPeeringBit p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	PackedVector2Array point_list;
	point_list.push_back(Vector2(3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(0, 3));
	point_list.push_back(Vector2(-3, 3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(-3, -3.0 * (1.0 - p_overlap * 2.0)));
	point_list.push_back(Vector2(0, -3));
	point_list.push_back(Vector2(3, -3.0 * (1.0 - p_overlap * 2.0)));

	Vector2 unit = Vector2(p_size) / 6.0;
	for (int i = 0; i < point_list.size(); i++) {
		point_list.write[i] = point_list[i] * unit;
	}

	PackedVector2Array polygon;
	if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
		switch (p_bit) {
			case TileData::TERRAIN_PEERING_BIT_RIGHT_SIDE:
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[0]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileData::TERRAIN_PEERING_BIT_LEFT_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE:
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				break;
			default:
				break;
		}
	} else {
		if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
			for (int i = 0; i < point_list.size(); i++) {
				point_list.write[i] = Vector2(point_list[i].y, point_list[i].x);
			}
		}
		switch (p_bit) {
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_SIDE:
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[0]);
				break;
			case TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE:
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			default:
				break;
		}
	}

	int half_polygon_size = polygon.size();
	for (int i = 0; i < half_polygon_size; i++) {
		polygon.push_back(polygon[half_polygon_size - 1 - i] / 3.0);
	}

	if (!polygon.is_empty()) {
		p_canvas_item->draw_polygon(polygon, color_array);
	}
}

#define TERRAIN_ALPHA 0.8

#define DRAW_TERRAIN_BIT(f, bit)                                     \
	{                                                                \
		int terrain_id = tile_data->get_peering_bit_terrain((bit));  \
		if (terrain_id >= 0) {                                       \
			Color color = p_tile_set->get_terrain_color(terrain_id); \
			color.a = TERRAIN_ALPHA;                                 \
			f(p_canvas_item, color, size, (bit));                    \
		}                                                            \
	}

#define DRAW_HALF_OFFSET_TERRAIN_BIT(f, bit, overlap, half_offset_axis)      \
	{                                                                        \
		int terrain_id = tile_data->get_peering_bit_terrain((bit));          \
		if (terrain_id >= 0) {                                               \
			Color color = p_tile_set->get_terrain_color(terrain_id);         \
			color.a = TERRAIN_ALPHA;                                         \
			f(p_canvas_item, color, size, (bit), overlap, half_offset_axis); \
		}                                                                    \
	}

void TileDataTerrainsEditor::draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) {
	TileData *tile_data = _get_tile_data(p_tile_set, p_atlas_source_id, p_atlas_coords, p_alternative_tile);
	ERR_FAIL_COND(!tile_data);

	TileSet::TileShape shape = p_tile_set->get_tile_shape();
	Vector2i size = p_tile_set->get_tile_size();

	Vector<String> components = String(p_property).split("/", true);
	if (components[0] == "terrain_mode" || components[0] == "terrain" || components[0] == "terrains_peering_bit") {
		RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), p_transform);
		if (shape == TileSet::TILE_SHAPE_SQUARE) {
			if (tile_data->get_terrain_mode() == TileData::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
				DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_RIGHT_SIDE);
				DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER);
				DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_SIDE);
				DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER);
				DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_LEFT_SIDE);
				DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER);
				DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_SIDE);
				DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER);
			} else if (tile_data->get_terrain_mode() == TileData::TERRAIN_MODE_MATCH_CORNERS) {
				DRAW_TERRAIN_BIT(_draw_square_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER);
				DRAW_TERRAIN_BIT(_draw_square_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER);
				DRAW_TERRAIN_BIT(_draw_square_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER);
				DRAW_TERRAIN_BIT(_draw_square_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER);
			} else { // TileData::TERRAIN_MODE_MATCH_SIDES
				DRAW_TERRAIN_BIT(_draw_square_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_RIGHT_SIDE);
				DRAW_TERRAIN_BIT(_draw_square_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_SIDE);
				DRAW_TERRAIN_BIT(_draw_square_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_LEFT_SIDE);
				DRAW_TERRAIN_BIT(_draw_square_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_SIDE);
			}

			// Draw the center terrain.
			int terrain_id = tile_data->get_terrain();
			if (terrain_id >= 0) {
				Color color = p_tile_set->get_terrain_color(terrain_id);
				color.a = TERRAIN_ALPHA;
				Rect2 rect = Rect2(-Vector2(p_tile_set->get_tile_size()) / 6.0, Vector2(p_tile_set->get_tile_size()) / 3.0);
				p_canvas_item->draw_rect(rect, color);
			}
		} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
			if (tile_data->get_terrain_mode() == TileData::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
				DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_RIGHT_CORNER);
				DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE);
				DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_CORNER);
				DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE);
				DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_LEFT_CORNER);
				DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE);
				DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_CORNER);
				DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE);
			} else if (tile_data->get_terrain_mode() == TileData::TERRAIN_MODE_MATCH_CORNERS) {
				DRAW_TERRAIN_BIT(_draw_isometric_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_RIGHT_CORNER);
				DRAW_TERRAIN_BIT(_draw_isometric_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_CORNER);
				DRAW_TERRAIN_BIT(_draw_isometric_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_LEFT_CORNER);
				DRAW_TERRAIN_BIT(_draw_isometric_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_CORNER);
			} else { // TileData::TERRAIN_MODE_MATCH_SIDES
				DRAW_TERRAIN_BIT(_draw_isometric_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE);
				DRAW_TERRAIN_BIT(_draw_isometric_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE);
				DRAW_TERRAIN_BIT(_draw_isometric_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE);
				DRAW_TERRAIN_BIT(_draw_isometric_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE);
			}

			// Draw the center terrain.
			int terrain_id = tile_data->get_terrain();
			if (terrain_id >= 0) {
				Color color = p_tile_set->get_terrain_color(terrain_id);
				color.a = TERRAIN_ALPHA;
				PackedColorArray color_array;
				color_array.push_back(color);

				Vector2 unit = Vector2(p_tile_set->get_tile_size()) / 6.0;
				PackedVector2Array polygon;
				polygon.push_back(Vector2(1, 0) * unit);
				polygon.push_back(Vector2(0, 1) * unit);
				polygon.push_back(Vector2(-1, 0) * unit);
				polygon.push_back(Vector2(0, -1) * unit);
				p_canvas_item->draw_polygon(polygon, color_array);
			}
		} else {
			TileSet::TileOffsetAxis offset_axis = p_tile_set->get_tile_offset_axis();
			float overlap = 0.0;
			switch (p_tile_set->get_tile_shape()) {
				case TileSet::TILE_SHAPE_HEXAGON:
					overlap = 0.25;
					break;
				case TileSet::TILE_SHAPE_HALF_OFFSET_SQUARE:
					overlap = 0.0;
					break;
				default:
					break;
			}
			if (tile_data->get_terrain_mode() == TileData::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
				if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_RIGHT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_LEFT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER, overlap, offset_axis);
				} else {
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_RIGHT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_LEFT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE, overlap, offset_axis);
				}
			} else if (tile_data->get_terrain_mode() == TileData::TERRAIN_MODE_MATCH_CORNERS) {
				if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER, overlap, offset_axis);
				} else {
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_RIGHT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_LEFT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_CORNER, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_CORNER, overlap, offset_axis);
				}
			} else { // TileData::TERRAIN_MODE_MATCH_SIDES
				if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_RIGHT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_LEFT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE, overlap, offset_axis);
				} else {
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_RIGHT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_BOTTOM_LEFT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_LEFT_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_SIDE, overlap, offset_axis);
					DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileData::TERRAIN_PEERING_BIT_TOP_RIGHT_SIDE, overlap, offset_axis);
				}
			}

			// Draw the center terrain.
			int terrain_id = tile_data->get_terrain();
			if (terrain_id >= 0) {
				Color color = p_tile_set->get_terrain_color(terrain_id);
				color.a = TERRAIN_ALPHA;
				PackedColorArray color_array;
				color_array.push_back(color);

				Vector2 unit = Vector2(p_tile_set->get_tile_size()) / 6.0;
				PackedVector2Array polygon;
				if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
					polygon.push_back(Vector2(1, (1.0 - overlap * 2.0)) * unit);
					polygon.push_back(Vector2(0, 1) * unit);
					polygon.push_back(Vector2(-1, (1.0 - overlap * 2.0)) * unit);
					polygon.push_back(Vector2(-1, -(1.0 - overlap * 2.0)) * unit);
					polygon.push_back(Vector2(0, -1) * unit);
					polygon.push_back(Vector2(1, -(1.0 - overlap * 2.0)) * unit);
				} else {
					polygon.push_back(Vector2((1.0 - overlap * 2.0), 1) * unit);
					polygon.push_back(Vector2(1, 0) * unit);
					polygon.push_back(Vector2((1.0 - overlap * 2.0), -1) * unit);
					polygon.push_back(Vector2(-(1.0 - overlap * 2.0), -1) * unit);
					polygon.push_back(Vector2(-1, 0) * unit);
					polygon.push_back(Vector2(-(1.0 - overlap * 2.0), 1) * unit);
				}
				p_canvas_item->draw_polygon(polygon, color_array);
			}
		}
		RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), Transform2D());
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
