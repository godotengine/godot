/*************************************************************************/
/*  collision_polygon.cpp                                                */
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
#include "collision_polygon.h"

#include "collision_object.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"

void CollisionPolygon::_add_to_collision_object(Object *p_obj) {

	if (!can_update_body)
		return;

	CollisionObject *co = p_obj->cast_to<CollisionObject>();
	ERR_FAIL_COND(!co);

	if (polygon.size() == 0)
		return;

	bool solids = build_mode == BUILD_SOLIDS;

	Vector<Vector<Vector2> > decomp = Geometry::decompose_polygon(polygon);
	if (decomp.size() == 0)
		return;

	if (true || solids) {

		//here comes the sun, lalalala
		//decompose concave into multiple convex polygons and add them
		shape_from = co->get_shape_count();
		for (int i = 0; i < decomp.size(); i++) {
			Ref<ConvexPolygonShape> convex = memnew(ConvexPolygonShape);
			PoolVector<Vector3> cp;
			int cs = decomp[i].size();
			cp.resize(cs * 2);
			{
				PoolVector<Vector3>::Write w = cp.write();
				int idx = 0;
				for (int j = 0; j < cs; j++) {

					Vector2 d = decomp[i][j];
					w[idx++] = Vector3(d.x, d.y, depth * 0.5);
					w[idx++] = Vector3(d.x, d.y, -depth * 0.5);
				}
			}

			convex->set_points(cp);
			co->add_shape(convex, get_transform());
		}
		shape_to = co->get_shape_count() - 1;
		if (shape_to < shape_from) {
			shape_from = -1;
			shape_to = -1;
		}

	} else {
#if 0
		Ref<ConcavePolygonShape> concave = memnew( ConcavePolygonShape );

		PoolVector<Vector2> segments;
		segments.resize(polygon.size()*2);
		PoolVector<Vector2>::Write w=segments.write();

		for(int i=0;i<polygon.size();i++) {
			w[(i<<1)+0]=polygon[i];
			w[(i<<1)+1]=polygon[(i+1)%polygon.size()];
		}

		w=PoolVector<Vector2>::Write();
		concave->set_segments(segments);

		co->add_shape(concave,get_transform());
#endif
	}

	//co->add_shape(shape,get_transform());
}

void CollisionPolygon::_update_parent() {

	if (!can_update_body)
		return;

	Node *parent = get_parent();
	if (!parent)
		return;
	CollisionObject *co = parent->cast_to<CollisionObject>();
	if (!co)
		return;
	co->_update_shapes_from_children();
}

void CollisionPolygon::_set_shape_range(const Vector2 &p_range) {

	shape_from = p_range.x;
	shape_to = p_range.y;
}

Vector2 CollisionPolygon::_get_shape_range() const {

	return Vector2(shape_from, shape_to);
}

void CollisionPolygon::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			can_update_body = get_tree()->is_editor_hint();
			set_notify_local_transform(!can_update_body);

			//indicator_instance = VisualServer::get_singleton()->instance_create2(indicator,get_world()->get_scenario());
		} break;
		case NOTIFICATION_EXIT_TREE: {
			can_update_body = false;
			set_notify_local_transform(false);
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (!is_inside_tree())
				break;
			if (can_update_body) {
				_update_parent();
			}

		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			if (!can_update_body && shape_from >= 0 && shape_to >= 0) {

				CollisionObject *co = get_parent()->cast_to<CollisionObject>();
				if (co) {
					for (int i = shape_from; i <= shape_to; i++) {
						co->set_shape_transform(i, get_transform());
					}
				}
			}

		} break;
#if 0
		case NOTIFICATION_DRAW: {
			for(int i=0;i<polygon.size();i++) {

				Vector2 p = polygon[i];
				Vector2 n = polygon[(i+1)%polygon.size()];
				draw_line(p,n,Color(0,0.6,0.7,0.5),3);
			}

			Vector< Vector<Vector2> > decomp = Geometry::decompose_polygon(polygon);
#define DEBUG_DECOMPOSE
#ifdef DEBUG_DECOMPOSE
			Color c(0.4,0.9,0.1);
			for(int i=0;i<decomp.size();i++) {

				c.set_hsv( Math::fmod(c.get_h() + 0.738,1),c.get_s(),c.get_v(),0.5);
				draw_colored_polygon(decomp[i],c);
			}
#endif

		} break;
#endif
	}
}

void CollisionPolygon::set_polygon(const Vector<Point2> &p_polygon) {

	polygon = p_polygon;
	if (can_update_body) {

		for (int i = 0; i < polygon.size(); i++) {

			Vector3 p1(polygon[i].x, polygon[i].y, depth * 0.5);

			if (i == 0)
				aabb = Rect3(p1, Vector3());
			else
				aabb.expand_to(p1);

			Vector3 p2(polygon[i].x, polygon[i].y, -depth * 0.5);
			aabb.expand_to(p2);
		}
		if (aabb == Rect3()) {

			aabb = Rect3(Vector3(-1, -1, -1), Vector3(2, 2, 2));
		} else {
			aabb.pos -= aabb.size * 0.3;
			aabb.size += aabb.size * 0.6;
		}
		_update_parent();
	}
	update_gizmo();
}

Vector<Point2> CollisionPolygon::get_polygon() const {

	return polygon;
}

void CollisionPolygon::set_build_mode(BuildMode p_mode) {

	ERR_FAIL_INDEX(p_mode, 2);
	build_mode = p_mode;
	if (!can_update_body)
		return;
	_update_parent();
}

CollisionPolygon::BuildMode CollisionPolygon::get_build_mode() const {

	return build_mode;
}

Rect3 CollisionPolygon::get_item_rect() const {

	return aabb;
}

void CollisionPolygon::set_depth(float p_depth) {

	depth = p_depth;
	if (!can_update_body)
		return;
	_update_parent();
	update_gizmo();
}

float CollisionPolygon::get_depth() const {

	return depth;
}

String CollisionPolygon::get_configuration_warning() const {

	if (!get_parent()->cast_to<CollisionObject>()) {
		return TTR("CollisionPolygon only serves to provide a collision shape to a CollisionObject derived node. Please only use it as a child of Area, StaticBody, RigidBody, KinematicBody, etc. to give them a shape.");
	}

	if (polygon.empty()) {
		return TTR("An empty CollisionPolygon has no effect on collision.");
	}

	return String();
}

void CollisionPolygon::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_add_to_collision_object"), &CollisionPolygon::_add_to_collision_object);

	ClassDB::bind_method(D_METHOD("set_build_mode", "build_mode"), &CollisionPolygon::set_build_mode);
	ClassDB::bind_method(D_METHOD("get_build_mode"), &CollisionPolygon::get_build_mode);

	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &CollisionPolygon::set_depth);
	ClassDB::bind_method(D_METHOD("get_depth"), &CollisionPolygon::get_depth);

	ClassDB::bind_method(D_METHOD("set_polygon", "polygon"), &CollisionPolygon::set_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon"), &CollisionPolygon::get_polygon);

	ClassDB::bind_method(D_METHOD("_set_shape_range", "shape_range"), &CollisionPolygon::_set_shape_range);
	ClassDB::bind_method(D_METHOD("_get_shape_range"), &CollisionPolygon::_get_shape_range);

	ClassDB::bind_method(D_METHOD("get_collision_object_first_shape"), &CollisionPolygon::get_collision_object_first_shape);
	ClassDB::bind_method(D_METHOD("get_collision_object_last_shape"), &CollisionPolygon::get_collision_object_last_shape);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "build_mode", PROPERTY_HINT_ENUM, "Solids,Triangles"), "set_build_mode", "get_build_mode");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "depth"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "polygon"), "set_polygon", "get_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "shape_range", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_shape_range", "_get_shape_range");
}

CollisionPolygon::CollisionPolygon() {

	shape_from = -1;
	shape_to = -1;
	can_update_body = false;

	aabb = Rect3(Vector3(-1, -1, -1), Vector3(2, 2, 2));
	build_mode = BUILD_SOLIDS;
	depth = 1.0;
}
