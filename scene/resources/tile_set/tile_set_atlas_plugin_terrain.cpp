/*************************************************************************/
/*  tile_set_atlas_plugin_terrain.cpp                                    */
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

#include "tile_set_atlas_plugin_terrain.h"

#include "scene/2d/tile_map.h"
#include "scene/main/canvas_item.h"

void TileSetAtlasPluginTerrain::_draw_square_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit) {
	Rect2 bit_rect;
	bit_rect.size = Vector2(p_size) / 3;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
			bit_rect.position = Vector2(1, -1);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
			bit_rect.position = Vector2(1, 1);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
			bit_rect.position = Vector2(-1, 1);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
			bit_rect.position = Vector2(-3, 1);
			break;
		case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
			bit_rect.position = Vector2(-3, -1);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
			bit_rect.position = Vector2(-3, -3);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_SIDE:
			bit_rect.position = Vector2(-1, -3);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
			bit_rect.position = Vector2(1, -3);
			break;
		default:
			break;
	}
	bit_rect.position *= Vector2(p_size) / 6.0;
	p_canvas_item->draw_rect(bit_rect, p_color);
}

void TileSetAtlasPluginTerrain::_draw_square_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	Vector2 unit = Vector2(p_size) / 6.0;
	PackedVector2Array polygon;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(3, 3) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(1, 1) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(-3, 3) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-1, 1) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(-3, -3) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-1, -1) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
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

void TileSetAtlasPluginTerrain::_draw_square_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	Vector2 unit = Vector2(p_size) / 6.0;
	PackedVector2Array polygon;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
			polygon.push_back(Vector2(1, -1) * unit);
			polygon.push_back(Vector2(3, -3) * unit);
			polygon.push_back(Vector2(3, 3) * unit);
			polygon.push_back(Vector2(1, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
			polygon.push_back(Vector2(-1, 1) * unit);
			polygon.push_back(Vector2(-3, 3) * unit);
			polygon.push_back(Vector2(3, 3) * unit);
			polygon.push_back(Vector2(1, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
			polygon.push_back(Vector2(-1, -1) * unit);
			polygon.push_back(Vector2(-3, -3) * unit);
			polygon.push_back(Vector2(-3, 3) * unit);
			polygon.push_back(Vector2(-1, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_SIDE:
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

void TileSetAtlasPluginTerrain::_draw_isometric_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	Vector2 unit = Vector2(p_size) / 6.0;
	PackedVector2Array polygon;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(2, -1) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(2, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
			polygon.push_back(Vector2(0, 1) * unit);
			polygon.push_back(Vector2(1, 2) * unit);
			polygon.push_back(Vector2(2, 1) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
			polygon.push_back(Vector2(0, 1) * unit);
			polygon.push_back(Vector2(-1, 2) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(1, 2) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
			polygon.push_back(Vector2(0, 1) * unit);
			polygon.push_back(Vector2(-1, 2) * unit);
			polygon.push_back(Vector2(-2, 1) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-2, -1) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-2, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
			polygon.push_back(Vector2(0, -1) * unit);
			polygon.push_back(Vector2(-1, -2) * unit);
			polygon.push_back(Vector2(-2, -1) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_CORNER:
			polygon.push_back(Vector2(0, -1) * unit);
			polygon.push_back(Vector2(-1, -2) * unit);
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(1, -2) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
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

void TileSetAtlasPluginTerrain::_draw_isometric_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	Vector2 unit = Vector2(p_size) / 6.0;
	PackedVector2Array polygon;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
			polygon.push_back(Vector2(0.5, -0.5) * unit);
			polygon.push_back(Vector2(1.5, -1.5) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(1.5, 1.5) * unit);
			polygon.push_back(Vector2(0.5, 0.5) * unit);
			polygon.push_back(Vector2(1, 0) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
			polygon.push_back(Vector2(-0.5, 0.5) * unit);
			polygon.push_back(Vector2(-1.5, 1.5) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(1.5, 1.5) * unit);
			polygon.push_back(Vector2(0.5, 0.5) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
			polygon.push_back(Vector2(-0.5, -0.5) * unit);
			polygon.push_back(Vector2(-1.5, -1.5) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(-1.5, 1.5) * unit);
			polygon.push_back(Vector2(-0.5, 0.5) * unit);
			polygon.push_back(Vector2(-1, 0) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_CORNER:
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

void TileSetAtlasPluginTerrain::_draw_isometric_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit) {
	PackedColorArray color_array;
	color_array.push_back(p_color);

	Vector2 unit = Vector2(p_size) / 6.0;
	PackedVector2Array polygon;
	switch (p_bit) {
		case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
			polygon.push_back(Vector2(1, 0) * unit);
			polygon.push_back(Vector2(3, 0) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(0, 3) * unit);
			polygon.push_back(Vector2(0, 1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
			polygon.push_back(Vector2(-1, 0) * unit);
			polygon.push_back(Vector2(-3, 0) * unit);
			polygon.push_back(Vector2(0, -3) * unit);
			polygon.push_back(Vector2(0, -1) * unit);
			break;
		case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
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

void TileSetAtlasPluginTerrain::_draw_half_offset_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis) {
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
			case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
				polygon.push_back(point_list[17]);
				polygon.push_back(point_list[0]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[6]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				polygon.push_back(point_list[11]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
				polygon.push_back(point_list[11]);
				polygon.push_back(point_list[12]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_CORNER:
				polygon.push_back(point_list[12]);
				polygon.push_back(point_list[13]);
				polygon.push_back(point_list[14]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
				polygon.push_back(point_list[14]);
				polygon.push_back(point_list[15]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
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
			case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
				polygon.push_back(point_list[17]);
				polygon.push_back(point_list[0]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[15]);
				polygon.push_back(point_list[16]);
				polygon.push_back(point_list[17]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[14]);
				polygon.push_back(point_list[15]);
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
				polygon.push_back(point_list[12]);
				polygon.push_back(point_list[13]);
				polygon.push_back(point_list[14]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
				polygon.push_back(point_list[11]);
				polygon.push_back(point_list[12]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				polygon.push_back(point_list[11]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_SIDE:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
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

void TileSetAtlasPluginTerrain::_draw_half_offset_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis) {
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
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[6]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_CORNER:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
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
			case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				polygon.push_back(point_list[10]);
				polygon.push_back(point_list[11]);
				polygon.push_back(point_list[0]);
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
				polygon.push_back(point_list[8]);
				polygon.push_back(point_list[9]);
				polygon.push_back(point_list[10]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				polygon.push_back(point_list[6]);
				polygon.push_back(point_list[7]);
				polygon.push_back(point_list[8]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
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

void TileSetAtlasPluginTerrain::_draw_half_offset_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis) {
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
			case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[0]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[1]);
				polygon.push_back(point_list[2]);
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
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
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
				polygon.push_back(point_list[0]);
				polygon.push_back(point_list[1]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
				polygon.push_back(point_list[5]);
				polygon.push_back(point_list[0]);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
				polygon.push_back(point_list[4]);
				polygon.push_back(point_list[5]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
				polygon.push_back(point_list[3]);
				polygon.push_back(point_list[4]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_SIDE:
				polygon.push_back(point_list[2]);
				polygon.push_back(point_list[3]);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
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

#define DRAW_TERRAIN_BIT(f, bit)                                                  \
	{                                                                             \
		int terrain_id = p_tile_data->get_peering_bit_terrain((bit));             \
		if (terrain_id >= 0) {                                                    \
			Color color = p_tile_set->get_terrain_color(terrain_set, terrain_id); \
			color.a = TERRAIN_ALPHA;                                              \
			f(p_canvas_item, color, size, (bit));                                 \
		}                                                                         \
	}

#define DRAW_HALF_OFFSET_TERRAIN_BIT(f, bit, overlap, half_offset_axis)           \
	{                                                                             \
		int terrain_id = p_tile_data->get_peering_bit_terrain((bit));             \
		if (terrain_id >= 0) {                                                    \
			Color color = p_tile_set->get_terrain_color(terrain_set, terrain_id); \
			color.a = TERRAIN_ALPHA;                                              \
			f(p_canvas_item, color, size, (bit), overlap, half_offset_axis);      \
		}                                                                         \
	}

void TileSetAtlasPluginTerrain::draw_terrains(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, const TileData *p_tile_data) {
	ERR_FAIL_COND(!p_tile_set);
	ERR_FAIL_COND(!p_tile_data);

	int terrain_set = p_tile_data->get_terrain_set();
	if (terrain_set < 0) {
		return;
	}
	TileSet::TerrainMode terrain_mode = p_tile_set->get_terrain_set_mode(terrain_set);

	TileSet::TileShape shape = p_tile_set->get_tile_shape();
	Vector2i size = p_tile_set->get_tile_size();

	RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), p_transform);
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
			DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_RIGHT_SIDE);
			DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER);
			DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE);
			DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER);
			DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
			DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER);
			DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_SIDE);
			DRAW_TERRAIN_BIT(_draw_square_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER);
		} else if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
			DRAW_TERRAIN_BIT(_draw_square_corner_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER);
			DRAW_TERRAIN_BIT(_draw_square_corner_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER);
			DRAW_TERRAIN_BIT(_draw_square_corner_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER);
			DRAW_TERRAIN_BIT(_draw_square_corner_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER);
		} else { // TileData::TERRAIN_MODE_MATCH_SIDES
			DRAW_TERRAIN_BIT(_draw_square_side_terrain_bit, TileSet::CELL_NEIGHBOR_RIGHT_SIDE);
			DRAW_TERRAIN_BIT(_draw_square_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE);
			DRAW_TERRAIN_BIT(_draw_square_side_terrain_bit, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
			DRAW_TERRAIN_BIT(_draw_square_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_SIDE);
		}
	} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
			DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_RIGHT_CORNER);
			DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE);
			DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_CORNER);
			DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE);
			DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_LEFT_CORNER);
			DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
			DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_CORNER);
			DRAW_TERRAIN_BIT(_draw_isometric_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
		} else if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
			DRAW_TERRAIN_BIT(_draw_isometric_corner_terrain_bit, TileSet::CELL_NEIGHBOR_RIGHT_CORNER);
			DRAW_TERRAIN_BIT(_draw_isometric_corner_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_CORNER);
			DRAW_TERRAIN_BIT(_draw_isometric_corner_terrain_bit, TileSet::CELL_NEIGHBOR_LEFT_CORNER);
			DRAW_TERRAIN_BIT(_draw_isometric_corner_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_CORNER);
		} else { // TileData::TERRAIN_MODE_MATCH_SIDES
			DRAW_TERRAIN_BIT(_draw_isometric_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE);
			DRAW_TERRAIN_BIT(_draw_isometric_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE);
			DRAW_TERRAIN_BIT(_draw_isometric_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
			DRAW_TERRAIN_BIT(_draw_isometric_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
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
		if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
			if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_RIGHT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_LEFT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER, overlap, offset_axis);
			} else {
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_RIGHT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_LEFT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_or_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, overlap, offset_axis);
			}
		} else if (terrain_mode == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
			if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER, overlap, offset_axis);
			} else {
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_RIGHT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_LEFT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_corner_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER, overlap, offset_axis);
			}
		} else { // TileData::TERRAIN_MODE_MATCH_SIDES
			if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_RIGHT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_LEFT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, overlap, offset_axis);
			} else {
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_SIDE, overlap, offset_axis);
				DRAW_HALF_OFFSET_TERRAIN_BIT(_draw_half_offset_side_terrain_bit, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, overlap, offset_axis);
			}
		}
	}
	RenderingServer::get_singleton()->canvas_item_add_set_transform(p_canvas_item->get_canvas_item(), Transform2D());
}
