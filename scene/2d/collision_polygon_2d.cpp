/*************************************************************************/
/*  collision_polygon_2d.cpp                                             */
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
#include "collision_polygon_2d.h"
#include "collision_object_2d.h"
#include "scene/resources/concave_polygon_shape_2d.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "triangulator.h"
void CollisionPolygon2D::_add_to_collision_object(Object *p_obj) {

	if (unparenting || !can_update_body)
		return;

	CollisionObject2D *co = p_obj->cast_to<CollisionObject2D>();
	ERR_FAIL_COND(!co);

	if (polygon.size() == 0)
		return;

	bool solids = build_mode == BUILD_SOLIDS;

	if (solids) {

		//here comes the sun, lalalala
		//decompose concave into multiple convex polygons and add them
		Vector<Vector<Vector2> > decomp = _decompose_in_convex();
		shape_from = co->get_shape_count();
		for (int i = 0; i < decomp.size(); i++) {
			Ref<ConvexPolygonShape2D> convex = memnew(ConvexPolygonShape2D);
			convex->set_points(decomp[i]);
			co->add_shape(convex, get_transform());
			if (trigger)
				co->set_shape_as_trigger(co->get_shape_count() - 1, true);
		}
		shape_to = co->get_shape_count() - 1;
		if (shape_to < shape_from) {
			shape_from = -1;
			shape_to = -1;
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

		co->add_shape(concave, get_transform());
		if (trigger)
			co->set_shape_as_trigger(co->get_shape_count() - 1, true);

		shape_from = co->get_shape_count() - 1;
		shape_to = co->get_shape_count() - 1;
	}

	//co->add_shape(shape,get_transform());
}

void CollisionPolygon2D::_update_parent() {

	if (!can_update_body)
		return;
	Node *parent = get_parent();
	if (!parent)
		return;
	CollisionObject2D *co = parent->cast_to<CollisionObject2D>();
	if (!co)
		return;
	co->_update_shapes_from_children();
}

Vector<Vector<Vector2> > CollisionPolygon2D::_decompose_in_convex() {

	Vector<Vector<Vector2> > decomp;
#if 0
	//fast but imprecise triangulator, gave us problems
	decomp = Geometry::decompose_polygon(polygon);
#else

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

#endif

	return decomp;
}

void CollisionPolygon2D::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			unparenting = false;
			can_update_body = get_tree()->is_editor_hint();
			if (!get_tree()->is_editor_hint()) {
				//display above all else
				set_z_as_relative(false);
				set_z(VS::CANVAS_ITEM_Z_MAX - 1);
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			can_update_body = false;
		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {

			if (!is_inside_tree())
				break;
			if (can_update_body) {
				_update_parent();
			} else if (shape_from >= 0 && shape_to >= 0) {
				CollisionObject2D *co = get_parent()->cast_to<CollisionObject2D>();
				for (int i = shape_from; i <= shape_to; i++) {
					co->set_shape_transform(i, get_transform());
				}
			}

		} break;

		case NOTIFICATION_DRAW: {

			if (!get_tree()->is_editor_hint() && !get_tree()->is_debugging_collisions_hint()) {
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

		} break;
		case NOTIFICATION_UNPARENTED: {
			unparenting = true;
			_update_parent();
		} break;
	}
}

void CollisionPolygon2D::set_polygon(const Vector<Point2> &p_polygon) {

	polygon = p_polygon;

	if (can_update_body) {
		for (int i = 0; i < polygon.size(); i++) {
			if (i == 0)
				aabb = Rect2(polygon[i], Size2());
			else
				aabb.expand_to(polygon[i]);
		}
		if (aabb == Rect2()) {

			aabb = Rect2(-10, -10, 20, 20);
		} else {
			aabb.pos -= aabb.size * 0.3;
			aabb.size += aabb.size * 0.6;
		}
		_update_parent();
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
	_update_parent();
}

CollisionPolygon2D::BuildMode CollisionPolygon2D::get_build_mode() const {

	return build_mode;
}

Rect2 CollisionPolygon2D::get_item_rect() const {

	return aabb;
}

void CollisionPolygon2D::set_trigger(bool p_trigger) {

	trigger = p_trigger;
	_update_parent();
	if (!can_update_body && is_inside_tree() && shape_from >= 0 && shape_to >= 0) {
		CollisionObject2D *co = get_parent()->cast_to<CollisionObject2D>();
		for (int i = shape_from; i <= shape_to; i++) {
			co->set_shape_as_trigger(i, p_trigger);
		}
	}
}

bool CollisionPolygon2D::is_trigger() const {

	return trigger;
}

void CollisionPolygon2D::_set_shape_range(const Vector2 &p_range) {

	shape_from = p_range.x;
	shape_to = p_range.y;
}

Vector2 CollisionPolygon2D::_get_shape_range() const {

	return Vector2(shape_from, shape_to);
}

String CollisionPolygon2D::get_configuration_warning() const {

	if (!get_parent()->cast_to<CollisionObject2D>()) {
		return TTR("CollisionPolygon2D only serves to provide a collision shape to a CollisionObject2D derived node. Please only use it as a child of Area2D, StaticBody2D, RigidBody2D, KinematicBody2D, etc. to give them a shape.");
	}

	if (polygon.empty()) {
		return TTR("An empty CollisionPolygon2D has no effect on collision.");
	}

	return String();
}

void CollisionPolygon2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_add_to_collision_object"), &CollisionPolygon2D::_add_to_collision_object);
	ClassDB::bind_method(D_METHOD("set_polygon", "polygon"), &CollisionPolygon2D::set_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon"), &CollisionPolygon2D::get_polygon);

	ClassDB::bind_method(D_METHOD("set_build_mode", "build_mode"), &CollisionPolygon2D::set_build_mode);
	ClassDB::bind_method(D_METHOD("get_build_mode"), &CollisionPolygon2D::get_build_mode);

	ClassDB::bind_method(D_METHOD("set_trigger", "trigger"), &CollisionPolygon2D::set_trigger);
	ClassDB::bind_method(D_METHOD("is_trigger"), &CollisionPolygon2D::is_trigger);

	ClassDB::bind_method(D_METHOD("_set_shape_range", "shape_range"), &CollisionPolygon2D::_set_shape_range);
	ClassDB::bind_method(D_METHOD("_get_shape_range"), &CollisionPolygon2D::_get_shape_range);

	ClassDB::bind_method(D_METHOD("get_collision_object_first_shape"), &CollisionPolygon2D::get_collision_object_first_shape);
	ClassDB::bind_method(D_METHOD("get_collision_object_last_shape"), &CollisionPolygon2D::get_collision_object_last_shape);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "build_mode", PROPERTY_HINT_ENUM, "Solids,Segments"), "set_build_mode", "get_build_mode");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "polygon"), "set_polygon", "get_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "shape_range", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_shape_range", "_get_shape_range");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "trigger"), "set_trigger", "is_trigger");
}

CollisionPolygon2D::CollisionPolygon2D() {

	aabb = Rect2(-10, -10, 20, 20);
	build_mode = BUILD_SOLIDS;
	trigger = false;
	unparenting = false;
	shape_from = -1;
	shape_to = -1;
	can_update_body = false;
	set_notify_local_transform(true);
}
