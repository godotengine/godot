/**************************************************************************/
/*  tile_set.h                                                            */
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

#ifndef TILE_SET_H
#define TILE_SET_H

#include "core/array.h"
#include "core/resource.h"
#include "scene/2d/light_occluder_2d.h"
#include "scene/2d/navigation_polygon.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "scene/resources/shape_2d.h"
#include "scene/resources/texture.h"

class TileSet : public Resource {
	GDCLASS(TileSet, Resource);

public:
	struct ShapeData {
		Ref<Shape2D> shape;
		Transform2D shape_transform;
		Vector2 autotile_coord;
		bool one_way_collision;
		float one_way_collision_margin;

		ShapeData() {
			one_way_collision = false;
			one_way_collision_margin = 1.0;
		}
	};

	enum BitmaskMode {
		BITMASK_2X2,
		BITMASK_3X3_MINIMAL,
		BITMASK_3X3
	};

	enum AutotileBindings {
		BIND_TOPLEFT = 1,
		BIND_TOP = 2,
		BIND_TOPRIGHT = 4,
		BIND_LEFT = 8,
		BIND_CENTER = 16,
		BIND_RIGHT = 32,
		BIND_BOTTOMLEFT = 64,
		BIND_BOTTOM = 128,
		BIND_BOTTOMRIGHT = 256,

		BIND_IGNORE_TOPLEFT = 1 << 16,
		BIND_IGNORE_TOP = 1 << 17,
		BIND_IGNORE_TOPRIGHT = 1 << 18,
		BIND_IGNORE_LEFT = 1 << 19,
		BIND_IGNORE_CENTER = 1 << 20,
		BIND_IGNORE_RIGHT = 1 << 21,
		BIND_IGNORE_BOTTOMLEFT = 1 << 22,
		BIND_IGNORE_BOTTOM = 1 << 23,
		BIND_IGNORE_BOTTOMRIGHT = 1 << 24
	};

	enum TileMode {
		SINGLE_TILE,
		AUTO_TILE,
		ATLAS_TILE
	};

	enum FallbackMode {
		FALLBACK_AUTO,
		FALLBACK_ICON
	};

	struct AutotileData {
		BitmaskMode bitmask_mode;
		Size2 size;
		int spacing;
		Vector2 icon_coord;
		Map<Vector2, uint32_t> flags;
		Map<Vector2, Ref<OccluderPolygon2D>> occluder_map;
		Map<Vector2, Ref<NavigationPolygon>> navpoly_map;
		Map<Vector2, int> priority_map;
		Map<Vector2, int> z_index_map;
		FallbackMode fallback_mode;

		// Default size to prevent invalid value
		explicit AutotileData() :
				bitmask_mode(BITMASK_2X2),
				size(64, 64),
				spacing(0),
				icon_coord(0, 0),
				fallback_mode(FALLBACK_AUTO) {}
	};

private:
	struct TileData {
		String name;
		Ref<Texture> texture;
		Ref<Texture> normal_map;
		Vector2 offset;
		Rect2i region;
		Vector<ShapeData> shapes_data;
		Vector2 occluder_offset;
		Ref<OccluderPolygon2D> occluder;
		Vector2 navigation_polygon_offset;
		Ref<NavigationPolygon> navigation_polygon;
		Ref<ShaderMaterial> material;
		TileMode tile_mode;
		Color modulate;
		AutotileData autotile_data;
		int z_index;

		// Default modulate for back-compat
		explicit TileData() :
				tile_mode(SINGLE_TILE),
				modulate(1, 1, 1),
				z_index(0) {}
	};

	Map<int, TileData> tile_map;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _tile_set_shapes(int p_id, const Array &p_shapes);
	Array _tile_get_shapes(int p_id) const;
	Array _get_tiles_ids() const;
	void _decompose_convex_shape(Ref<Shape2D> p_shape);
	List<Vector2> _autotile_get_subtile_candidates_for_bitmask(int p_id, uint16_t p_bitmask) const;

	uint32_t _count_bitmask_bits(uint32_t p_bitmask);
	uint32_t _score_bitmask_difference(uint32_t p_bitmask, uint32_t p_ref_bitmask);

	static void _bind_methods();

public:
	void create_tile(int p_id);

	void autotile_set_bitmask_mode(int p_id, BitmaskMode p_mode);
	BitmaskMode autotile_get_bitmask_mode(int p_id) const;

	void tile_set_name(int p_id, const String &p_name);
	String tile_get_name(int p_id) const;

	void tile_set_texture(int p_id, const Ref<Texture> &p_texture);
	Ref<Texture> tile_get_texture(int p_id) const;

	void tile_set_normal_map(int p_id, const Ref<Texture> &p_normal_map);
	Ref<Texture> tile_get_normal_map(int p_id) const;

	void tile_set_texture_offset(int p_id, const Vector2 &p_offset);
	Vector2 tile_get_texture_offset(int p_id) const;

	void tile_set_region(int p_id, const Rect2 &p_region);
	Rect2 tile_get_region(int p_id) const;

	void tile_set_tile_mode(int p_id, TileMode p_tile_mode);
	TileMode tile_get_tile_mode(int p_id) const;

	void autotile_set_icon_coordinate(int p_id, const Vector2 &coord);
	Vector2 autotile_get_icon_coordinate(int p_id) const;

	void autotile_set_spacing(int p_id, int p_spacing);
	int autotile_get_spacing(int p_id) const;

	void autotile_set_size(int p_id, const Size2 &p_size);
	Size2 autotile_get_size(int p_id) const;

	void autotile_clear_bitmask_map(int p_id);
	void autotile_set_subtile_priority(int p_id, const Vector2 &p_coord, int p_priority);
	int autotile_get_subtile_priority(int p_id, const Vector2 &p_coord);
	const Map<Vector2, int> &autotile_get_priority_map(int p_id) const;

	void autotile_set_z_index(int p_id, const Vector2 &p_coord, int p_z_index);
	int autotile_get_z_index(int p_id, const Vector2 &p_coord);
	const Map<Vector2, int> &autotile_get_z_index_map(int p_id) const;

	void autotile_set_fallback_mode(int p_id, FallbackMode p_mode);
	FallbackMode autotile_get_fallback_mode(int p_id) const;

	void autotile_set_bitmask(int p_id, const Vector2 &p_coord, uint32_t p_flag);
	uint32_t autotile_get_bitmask(int p_id, const Vector2 &p_coord);
	const Map<Vector2, uint32_t> &autotile_get_bitmask_map(int p_id);
	Vector2 autotile_get_subtile_for_bitmask(int p_id, uint16_t p_bitmask, const Node *p_tilemap_node = nullptr, const Vector2 &p_tile_location = Vector2());
	Vector2 atlastile_get_subtile_by_priority(int p_id, const Node *p_tilemap_node = nullptr, const Vector2 &p_tile_location = Vector2());

	void tile_set_shape(int p_id, int p_shape_id, const Ref<Shape2D> &p_shape);
	Ref<Shape2D> tile_get_shape(int p_id, int p_shape_id) const;

	void tile_set_shape_transform(int p_id, int p_shape_id, const Transform2D &p_offset);
	Transform2D tile_get_shape_transform(int p_id, int p_shape_id) const;

	void tile_set_shape_offset(int p_id, int p_shape_id, const Vector2 &p_offset);
	Vector2 tile_get_shape_offset(int p_id, int p_shape_id) const;

	void tile_set_shape_one_way(int p_id, int p_shape_id, bool p_one_way);
	bool tile_get_shape_one_way(int p_id, int p_shape_id) const;

	void tile_set_shape_one_way_margin(int p_id, int p_shape_id, float p_margin);
	float tile_get_shape_one_way_margin(int p_id, int p_shape_id) const;

	void tile_clear_shapes(int p_id);
	void tile_add_shape(int p_id, const Ref<Shape2D> &p_shape, const Transform2D &p_transform, bool p_one_way = false, const Vector2 &p_autotile_coord = Vector2());
	int tile_get_shape_count(int p_id) const;

	void tile_set_shapes(int p_id, const Vector<ShapeData> &p_shapes);
	Vector<ShapeData> tile_get_shapes(int p_id) const;

	void tile_set_material(int p_id, const Ref<ShaderMaterial> &p_material);
	Ref<ShaderMaterial> tile_get_material(int p_id) const;

	void tile_set_modulate(int p_id, const Color &p_modulate);
	Color tile_get_modulate(int p_id) const;

	void tile_set_occluder_offset(int p_id, const Vector2 &p_offset);
	Vector2 tile_get_occluder_offset(int p_id) const;

	void tile_set_light_occluder(int p_id, const Ref<OccluderPolygon2D> &p_light_occluder);
	Ref<OccluderPolygon2D> tile_get_light_occluder(int p_id) const;

	void autotile_set_light_occluder(int p_id, const Ref<OccluderPolygon2D> &p_light_occluder, const Vector2 &p_coord);
	Ref<OccluderPolygon2D> autotile_get_light_occluder(int p_id, const Vector2 &p_coord) const;
	const Map<Vector2, Ref<OccluderPolygon2D>> &autotile_get_light_oclusion_map(int p_id) const;

	void tile_set_navigation_polygon_offset(int p_id, const Vector2 &p_offset);
	Vector2 tile_get_navigation_polygon_offset(int p_id) const;

	void tile_set_navigation_polygon(int p_id, const Ref<NavigationPolygon> &p_navigation_polygon);
	Ref<NavigationPolygon> tile_get_navigation_polygon(int p_id) const;

	void autotile_set_navigation_polygon(int p_id, const Ref<NavigationPolygon> &p_navigation_polygon, const Vector2 &p_coord);
	Ref<NavigationPolygon> autotile_get_navigation_polygon(int p_id, const Vector2 &p_coord) const;
	const Map<Vector2, Ref<NavigationPolygon>> &autotile_get_navigation_map(int p_id) const;

	void tile_set_z_index(int p_id, int p_z_index);
	int tile_get_z_index(int p_id) const;

	void remove_tile(int p_id);

	bool has_tile(int p_id) const;

	bool is_tile_bound(int p_drawn_id, int p_neighbor_id);

	int find_tile_by_name(const String &p_name) const;
	void get_tile_list(List<int> *p_tiles) const;

	void clear();

	int get_last_unused_tile_id() const;

	TileSet();
};

VARIANT_ENUM_CAST(TileSet::AutotileBindings);
VARIANT_ENUM_CAST(TileSet::BitmaskMode);
VARIANT_ENUM_CAST(TileSet::TileMode);
VARIANT_ENUM_CAST(TileSet::FallbackMode);

#endif // TILE_SET_H
