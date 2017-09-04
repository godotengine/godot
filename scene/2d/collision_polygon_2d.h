/*************************************************************************/
/*  collision_polygon_2d.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef COLLISION_POLYGON_2D_H
#define COLLISION_POLYGON_2D_H

#include "scene/2d/node_2d.h"
#include "scene/resources/shape_2d.h"
#include "scene/2d/polygon_node_2d.h"

class CollisionObject2D;

class CollisionPolygon2D : public PolygonNode2D {

	GDCLASS(CollisionPolygon2D, PolygonNode2D);

public:
	enum BuildMode {
		BUILD_SOLIDS,
		BUILD_SEGMENTS,
	};

protected:
	Ref<Ring2D> ring;
	CollisionObject2D *parent;
	uint32_t owner_id;
	BuildMode build_mode;
	bool disabled;
	bool one_way_collision;

	void _build_polygon();
	void _polygon_changed();
	Vector<Vector<Vector2> > _decompose_in_convex();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_ring(Ref<Ring2D> p_polygon);
	Ref<Ring2D> get_ring() const;

	void set_build_mode(BuildMode p_mode);
	BuildMode get_build_mode() const;

	void set_disabled(bool p_disabled);
	bool is_disabled() const;

	void set_one_way_collision(bool p_enable);
	bool is_one_way_collision_enabled() const;

	virtual int get_polygon_count() const;
	virtual Ref<Resource> get_nth_polygon(int p_idx) const;
	virtual int get_ring_count(Ref<Resource> p_polygon) const;
	virtual Ref<Ring2D> get_nth_ring(Ref<Resource> p_polygon, int p_idx) const;

	virtual Ref<Resource> new_polygon(const Ref<Ring2D> &p_ring) const;
	virtual void add_polygon_at_index(Ref<Resource> p_polygon, int p_idx);
	virtual void remove_polygon(int p_idx);

	virtual String get_configuration_warning() const;
	virtual Rect2 get_item_rect() const;

	CollisionPolygon2D();
};

VARIANT_ENUM_CAST(CollisionPolygon2D::BuildMode);

#endif // COLLISION_POLYGON_2D_H
