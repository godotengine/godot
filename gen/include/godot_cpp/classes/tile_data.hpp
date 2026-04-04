/**************************************************************************/
/*  tile_data.hpp                                                         */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/tile_set.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Material;
class NavigationPolygon;
class OccluderPolygon2D;
class String;

class TileData : public Object {
	GDEXTENSION_CLASS(TileData, Object)

public:
	void set_flip_h(bool p_flip_h);
	bool get_flip_h() const;
	void set_flip_v(bool p_flip_v);
	bool get_flip_v() const;
	void set_transpose(bool p_transpose);
	bool get_transpose() const;
	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;
	void set_texture_origin(const Vector2i &p_texture_origin);
	Vector2i get_texture_origin() const;
	void set_modulate(const Color &p_modulate);
	Color get_modulate() const;
	void set_z_index(int32_t p_z_index);
	int32_t get_z_index() const;
	void set_y_sort_origin(int32_t p_y_sort_origin);
	int32_t get_y_sort_origin() const;
	void set_occluder_polygons_count(int32_t p_layer_id, int32_t p_polygons_count);
	int32_t get_occluder_polygons_count(int32_t p_layer_id) const;
	void add_occluder_polygon(int32_t p_layer_id);
	void remove_occluder_polygon(int32_t p_layer_id, int32_t p_polygon_index);
	void set_occluder_polygon(int32_t p_layer_id, int32_t p_polygon_index, const Ref<OccluderPolygon2D> &p_polygon);
	Ref<OccluderPolygon2D> get_occluder_polygon(int32_t p_layer_id, int32_t p_polygon_index, bool p_flip_h = false, bool p_flip_v = false, bool p_transpose = false) const;
	void set_occluder(int32_t p_layer_id, const Ref<OccluderPolygon2D> &p_occluder_polygon);
	Ref<OccluderPolygon2D> get_occluder(int32_t p_layer_id, bool p_flip_h = false, bool p_flip_v = false, bool p_transpose = false) const;
	void set_constant_linear_velocity(int32_t p_layer_id, const Vector2 &p_velocity);
	Vector2 get_constant_linear_velocity(int32_t p_layer_id) const;
	void set_constant_angular_velocity(int32_t p_layer_id, float p_velocity);
	float get_constant_angular_velocity(int32_t p_layer_id) const;
	void set_collision_polygons_count(int32_t p_layer_id, int32_t p_polygons_count);
	int32_t get_collision_polygons_count(int32_t p_layer_id) const;
	void add_collision_polygon(int32_t p_layer_id);
	void remove_collision_polygon(int32_t p_layer_id, int32_t p_polygon_index);
	void set_collision_polygon_points(int32_t p_layer_id, int32_t p_polygon_index, const PackedVector2Array &p_polygon);
	PackedVector2Array get_collision_polygon_points(int32_t p_layer_id, int32_t p_polygon_index) const;
	void set_collision_polygon_one_way(int32_t p_layer_id, int32_t p_polygon_index, bool p_one_way);
	bool is_collision_polygon_one_way(int32_t p_layer_id, int32_t p_polygon_index) const;
	void set_collision_polygon_one_way_margin(int32_t p_layer_id, int32_t p_polygon_index, float p_one_way_margin);
	float get_collision_polygon_one_way_margin(int32_t p_layer_id, int32_t p_polygon_index) const;
	void set_terrain_set(int32_t p_terrain_set);
	int32_t get_terrain_set() const;
	void set_terrain(int32_t p_terrain);
	int32_t get_terrain() const;
	void set_terrain_peering_bit(TileSet::CellNeighbor p_peering_bit, int32_t p_terrain);
	int32_t get_terrain_peering_bit(TileSet::CellNeighbor p_peering_bit) const;
	bool is_valid_terrain_peering_bit(TileSet::CellNeighbor p_peering_bit) const;
	void set_navigation_polygon(int32_t p_layer_id, const Ref<NavigationPolygon> &p_navigation_polygon);
	Ref<NavigationPolygon> get_navigation_polygon(int32_t p_layer_id, bool p_flip_h = false, bool p_flip_v = false, bool p_transpose = false) const;
	void set_probability(float p_probability);
	float get_probability() const;
	void set_custom_data(const String &p_layer_name, const Variant &p_value);
	Variant get_custom_data(const String &p_layer_name) const;
	bool has_custom_data(const String &p_layer_name) const;
	void set_custom_data_by_layer_id(int32_t p_layer_id, const Variant &p_value);
	Variant get_custom_data_by_layer_id(int32_t p_layer_id) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

