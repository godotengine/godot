/*************************************************************************/
/*  tile_set.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef TILE_SET_H
#define TILE_SET_H

#include "resource.h"
#include "scene/2d/light_occluder_2d.h"
#include "scene/2d/navigation_polygon.h"
#include "scene/resources/shape_2d.h"
#include "scene/resources/texture.h"

class TileSet : public Resource {

	OBJ_TYPE(TileSet, Resource);

	struct Data {

		String name;
		Ref<Texture> texture;
		Vector2 offset;
		Vector2 shape_offset;
		Rect2i region;
		Vector<Ref<Shape2D> > shapes;
		Vector2 one_way_collision_direction;
		float one_way_collision_max_depth;
		Vector2 occluder_offset;
		Ref<OccluderPolygon2D> occluder;
		Vector2 navigation_polygon_offset;
		Ref<NavigationPolygon> navigation_polygon;
		Ref<CanvasItemMaterial> material;
		Color modulate;

		explicit Data()
			: one_way_collision_max_depth(0.0f), modulate(1, 1, 1) {}
	};

	Map<int, Data> tile_map;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _tile_set_shapes(int p_id, const Array &p_shapes);
	Array _tile_get_shapes(int p_id) const;
	Array _get_tiles_ids() const;

	static void _bind_methods();

public:
	void create_tile(int p_id);

	void tile_set_name(int p_id, const String &p_name);
	String tile_get_name(int p_id) const;

	void tile_set_texture(int p_id, const Ref<Texture> &p_texture);
	Ref<Texture> tile_get_texture(int p_id) const;

	void tile_set_texture_offset(int p_id, const Vector2 &p_offset);
	Vector2 tile_get_texture_offset(int p_id) const;

	void tile_set_shape_offset(int p_id, const Vector2 &p_offset);
	Vector2 tile_get_shape_offset(int p_id) const;

	void tile_set_region(int p_id, const Rect2 &p_region);
	Rect2 tile_get_region(int p_id) const;

	void tile_set_shape(int p_id, const Ref<Shape2D> &p_shape);
	Ref<Shape2D> tile_get_shape(int p_id) const;

	void tile_set_material(int p_id, const Ref<CanvasItemMaterial> &p_material);
	Ref<CanvasItemMaterial> tile_get_material(int p_id) const;

	void tile_set_modulate(int p_id, const Color &p_color);
	Color tile_get_modulate(int p_id) const;

	void tile_set_occluder_offset(int p_id, const Vector2 &p_offset);
	Vector2 tile_get_occluder_offset(int p_id) const;

	void tile_set_light_occluder(int p_id, const Ref<OccluderPolygon2D> &p_light_occluder);
	Ref<OccluderPolygon2D> tile_get_light_occluder(int p_id) const;

	void tile_set_navigation_polygon_offset(int p_id, const Vector2 &p_offset);
	Vector2 tile_get_navigation_polygon_offset(int p_id) const;

	void tile_set_navigation_polygon(int p_id, const Ref<NavigationPolygon> &p_navigation_polygon);
	Ref<NavigationPolygon> tile_get_navigation_polygon(int p_id) const;

	void tile_set_shapes(int p_id, const Vector<Ref<Shape2D> > &p_shapes);
	Vector<Ref<Shape2D> > tile_get_shapes(int p_id) const;

	void tile_set_one_way_collision_direction(int p_id, Vector2 p_direction);
	Vector2 tile_get_one_way_collision_direction(int p_id) const;

	void tile_set_one_way_collision_max_depth(int p_id, float p_max_depth);
	float tile_get_one_way_collision_max_depth(int p_id) const;

	void remove_tile(int p_id);

	bool has_tile(int p_id) const;

	int find_tile_by_name(const String &p_name) const;
	void get_tile_list(List<int> *p_tiles) const;

	void clear();

	int get_last_unused_tile_id() const;

	TileSet();
};

#endif // TILE_SET_H
