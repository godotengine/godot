/*************************************************************************/
/*  collision_polygon_2d.cpp                                             */
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
#include "collision_polygon_2d.h"

#include "collision_object_2d.h"
#include "engine.h"
#include "scene/resources/concave_polygon_shape_2d.h"
#include "scene/resources/convex_polygon_shape_2d.h"

#include "thirdparty/misc/triangulator.h"

void CollisionPolygon2D::_build_polygon() {

	parent->shape_owner_clear_shapes(owner_id);

	if (polygon.size() == 0)
		return;

	bool solids = build_mode == BUILD_SOLIDS;

	if (solids) {

		//here comes the sun, lalalala
		//decompose concave into multiple convex polygons and add them
		Vector<Vector<Vector2> > decomp = _decompose_in_convex();
		for (int i = 0; i < decomp.size(); i++) {
			Ref<ConvexPolygonShape2D> convex = memnew(ConvexPolygonShape2D);
			convex->set_points(decomp[i]);
			parent->shape_owner_add_shape(owner_id, convex);
		}

	} else {

		Ref<ConcavePolygonShape2D> concave = memnew(ConcavePolygonShape2D);

		PoolVector<Vector2> segments;
		segments.resize(polygon.size() * 2);
		PoolVector<Vector2>::Write w = segments.write();

		for (int i = 0; i < polygon.size(); i++) {
			w[(i << 1) + 0] = polygon[i];
			w[(i << 1) + 1] = polygon[(i + 1) % polygon.size()];
		}

		w = PoolVector<Vector2>::Write();
		concave->set_segments(segments);

		parent->shape_owner_add_shape(owner_id, concave);
	}
}

Vector<Vector<Vector2> > CollisionPolygon2D::_decompose_in_convex() {

	Vector<Vector<Vector2> > decomp;
	List<TriangulatorPoly> in_poly, out_poly;

	TriangulatorPoly inp;
	inp.Init(polygon.size());
	for (int i = 0; i < polygon.size(); i++) {
		inp.GetPoint(i) = polygon[i];
	}
	inp.SetOrientation(TRIANGULATOR_CCW);
	in_poly.push_back(inp);
	TriangulatorPartition tpart;
	if (tpart.ConvexPartition_HM(&in_poly, &out_poly) == 0) { //failed!
		ERR_PRINT("Convex decomposing failed!");
		return decomp;
	}

	decomp.resize(out_poly.size());
	int idx = 0;

	for (List<TriangulatorPoly>::Element *I = out_poly.front(); I; I = I->next()) {

		TriangulatorPoly &tp = I->get();

		decomp[idx].resize(tp.GetNumPoints());

		for (int i = 0; i < tp.GetNumPoints(); i++) {

			decomp[idx][i] = tp.GetPoint(i);
		}

		idx++;
	}

	return decomp;
}

void CollisionPolygon2D::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_PARENTED: {

			parent = Object::cast_to<CollisionObject2D>(get_parent());
			if (parent) {
				owner_id = parent->create_shape_owner(this);
				_build_polygon();
				parent->shape_owner_set_transform(owner_id, get_transform());
				parent->shape_owner_set_disabled(owner_id, disabled);
				parent->shape_owner_set_one_way_collision(owner_id, one_way_collision);
			}

			/*if (Engine::get_singleton()->is_editor_hint()) {
				//display above all else
				set_z_as_relative(false);
				set_z(VS::CANVAS_ITEM_Z_MAX - 1);
			}*/

		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {

			if (parent) {
				parent->shape_owner_set_transform(owner_id, get_transform());
			}

		} break;
		case NOTIFICATION_UNPARENTED: {
			if (parent) {
				parent->remove_shape_owner(owner_id);
			}
			owner_id = 0;
			parent = NULL;
		} break;

		case NOTIFICATION_DRAW: {

			if (!Engine::get_singleton()->is_editor_hint() && !get_tree()->is_debugging_collisions_hint()) {
				break;
			}

			for (int i = 0; i < polygon.size(); i++) {

				Vector2 p = polygon[i];
				Vector2 n = polygon[(i + 1) % polygon.size()];
				draw_line(p, n, Color(0.9, 0.2, 0.0, 0.8), 3);
			}
#define DEBUG_DECOMPOSE
#if defined(TOOLS_ENABLED) && defined(DEBUG_DECOMPOSE)

			Vector<Vector<Vector2> > decomp = _decompose_in_convex();

			Color c(0.4, 0.9, 0.1);
			for (int i = 0; i < decomp.size(); i++) {

				c.set_hsv(Math::fmod(c.get_h() + 0.738, 1), c.get_s(), c.get_v(), 0.5);
				draw_colored_polygon(decomp[i], c);
			}
#else
			draw_colored_polygon(polygon, get_tree()->get_debug_collisions_color());
#endif

			if (one_way_collision) {
				Color dcol = get_tree()->get_debug_collisions_color(); //0.9,0.2,0.2,0.4);
				dcol.a = 1.0;
				Vector2 line_to(0, 20);
				draw_line(Vector2(), line_to, dcol, 3);
				Vector<Vector2> pts;
				float tsize = 8;
				pts.push_back(line_to + (Vector2(0, tsize)));
				pts.push_back(line_to + (Vector2(0.707 * tsize, 0)));
				pts.push_back(line_to + (Vector2(-0.707 * tsize, 0)));
				Vector<Color> cols;
				for (int i = 0; i < 3; i++)
					cols.push_back(dcol);

				draw_primitive(pts, cols, Vector<Vector2>()); //small arrow
			}
		} break;
	}
}

void CollisionPolygon2D::set_polygon(const Vector<Point2> &p_polygon) {

	polygon = p_polygon;

	{
		for (int i = 0; i < polygon.size(); i++) {
			if (i == 0)
				aabb = Rect2(polygon[i], Size2());
			else
				aabb.expand_to(polygon[i]);
		}
		if (aabb == Rect2()) {

			aabb = Rect2(-10, -10, 20, 20);
		} else {
			aabb.position -= aabb.size * 0.3;
			aabb.size += aabb.size * 0.6;
		}
	}

	if (parent) {
		_build_polygon();
	}
	update();
	update_configuration_warning();
}

Vector<Point2> CollisionPolygon2D::get_polygon() const {

	return polygon;
}

void CollisionPolygon2D::set_build_mode(BuildMode p_mode) {

	ERR_FAIL_INDEX(p_mode, 2);
	build_mode = p_mode;
	if (parent) {
		_build_polygon();
	}
}

CollisionPolygon2D::BuildMode CollisionPolygon2D::get_build_mode() const {

	return build_mode;
}

Rect2 CollisionPolygon2D::_edit_get_rect() const {

	return aabb;
}

String CollisionPolygon2D::get_configuration_warning() const {

	if (!Object::cast_to<CollisionObject2D>(get_parent())) {
		return TTR("CollisionPolygon2D only serves to provide a collision shape to a CollisionObject2D derived node. Please only use it as a child of Area2D, StaticBody2D, RigidBody2D, KinematicBody2D, etc. to give them a shape.");
	}

	if (polygon.empty()) {
		return TTR("An empty CollisionPolygon2D has no effect on collision.");
	}

	return String();
}

void CollisionPolygon2D::set_disabled(bool p_disabled) {
	disabled = p_disabled;
	update();
	if (parent) {
		parent->shape_owner_set_disabled(owner_id, p_disabled);
	}
}

bool CollisionPolygon2D::is_disabled() const {
	return disabled;
}

void CollisionPolygon2D::set_one_way_collision(bool p_enable) {
	one_way_collision = p_enable;
	update();
	if (parent) {
		parent->shape_owner_set_one_way_collision(owner_id, p_enable);
	}
}

bool CollisionPolygon2D::is_one_way_collision_enabled() const {

	return one_way_collision;
}

void CollisionPolygon2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_polygon", "polygon"), &CollisionPolygon2D::set_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon"), &CollisionPolygon2D::get_polygon);

	ClassDB::bind_method(D_METHOD("set_build_mode", "build_mode"), &CollisionPolygon2D::set_build_mode);
	ClassDB::bind_method(D_METHOD("get_build_mode"), &CollisionPolygon2D::get_build_mode);
	ClassDB::bind_method(D_METHOD("set_disabled", "disabled"), &CollisionPolygon2D::set_disabled);
	ClassDB::bind_method(D_METHOD("is_disabled"), &CollisionPolygon2D::is_disabled);
	ClassDB::bind_method(D_METHOD("set_one_way_collision", "enabled"), &CollisionPolygon2D::set_one_way_collision);
	ClassDB::bind_method(D_METHOD("is_one_way_collision_enabled"), &CollisionPolygon2D::is_one_way_collision_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "build_mode", PROPERTY_HINT_ENUM, "Solids,Segments"), "set_build_mode", "get_build_mode");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "polygon"), "set_polygon", "get_polygon");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "one_way_collision"), "set_one_way_collision", "is_one_way_collision_enabled");

	BIND_ENUM_CONSTANT(BUILD_SOLIDS);
	BIND_ENUM_CONSTANT(BUILD_SEGMENTS);
}

CollisionPolygon2D::CollisionPolygon2D() {

	aabb = Rect2(-10, -10, 20, 20);
	build_mode = BUILD_SOLIDS;
	set_notify_local_transform(true);
	parent = NULL;
	owner_id = 0;
	disabled = false;
	one_way_collision = false;
}
