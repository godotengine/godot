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

#include "core_string_names.h"

Vector<Vector<Vector2> > CollisionPolygon2D::_decompose_in_convex() {

	Vector<Vector<Vector2> > decomp;
	List<TriangulatorPoly> in_poly, out_poly;

	TriangulatorPoly inp;
	inp.Init(vertices.size());
	for (int i = 0; i < vertices.size(); i++) {
		inp.GetPoint(i) = vertices[i];
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

void CollisionPolygon2DInstance::_build_polygon() {

	parent->shape_owner_clear_shapes(owner_id);

	if (!polygon.is_valid())
		return;

	const Vector<Vector2> points = polygon->get_vertices();

	if (points.size() == 0)
		return;

	bool solids = build_mode == BUILD_SOLIDS;

	if (solids) {

		//here comes the sun, lalalala
		//decompose concave into multiple convex polygons and add them
		Vector<Vector<Vector2> > decomp = polygon->_decompose_in_convex();
		for (int i = 0; i < decomp.size(); i++) {
			Ref<ConvexPolygonShape2D> convex = memnew(ConvexPolygonShape2D);
			convex->set_points(decomp[i]);
			parent->shape_owner_add_shape(owner_id, convex);
		}

	} else {

		Ref<ConcavePolygonShape2D> concave = memnew(ConcavePolygonShape2D);

		PoolVector<Vector2> segments;
		segments.resize(points.size() * 2);
		PoolVector<Vector2>::Write w = segments.write();

		for (int i = 0; i < points.size(); i++) {
			w[(i << 1) + 0] = points[i];
			w[(i << 1) + 1] = points[(i + 1) % points.size()];
		}

		w = PoolVector<Vector2>::Write();
		concave->set_segments(segments);

		parent->shape_owner_add_shape(owner_id, concave);
	}
}

void CollisionPolygon2DInstance::_notification(int p_what) {

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

			if (!polygon.is_valid())
				break;

			const Vector<Vector2> points = polygon->get_vertices();

			for (int i = 0; i < points.size(); i++) {

				Vector2 p = points[i];
				Vector2 n = points[(i + 1) % points.size()];
				draw_line(p, n, Color(0.9, 0.2, 0.0, 0.8), 3);
			}
#define DEBUG_DECOMPOSE
#if defined(TOOLS_ENABLED) && defined(DEBUG_DECOMPOSE)

			Vector<Vector<Vector2> > decomp = polygon->_decompose_in_convex();

			Color c(0.4, 0.9, 0.1);
			for (int i = 0; i < decomp.size(); i++) {

				c.set_hsv(Math::fmod(c.get_h() + 0.738, 1), c.get_s(), c.get_v(), 0.5);
				draw_colored_polygon(decomp[i], c);
			}
#else
			draw_colored_polygon(points, get_tree()->get_debug_collisions_color());
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

void CollisionPolygon2DInstance::set_polygon(Ref<CollisionPolygon2D> p_polygon) {

	if (p_polygon == polygon)
		return;

	if (polygon.is_valid()) {
		polygon->disconnect(CoreStringNames::get_singleton()->changed, this, "_polygon_changed");
	}
	polygon = p_polygon;

	if (polygon.is_valid()) {
		polygon->connect(CoreStringNames::get_singleton()->changed, this, "_polygon_changed");
	}

	_polygon_changed();
	_change_notify("polygon");
	update_configuration_warning();
}

Ref<CollisionPolygon2D> CollisionPolygon2DInstance::get_polygon() const {

	return polygon;
}

void CollisionPolygon2DInstance::set_build_mode(BuildMode p_mode) {

	ERR_FAIL_INDEX(p_mode, 2);
	build_mode = p_mode;
	if (parent) {
		_build_polygon();
	}
}

CollisionPolygon2DInstance::BuildMode CollisionPolygon2DInstance::get_build_mode() const {

	return build_mode;
}

void CollisionPolygon2DInstance::set_disabled(bool p_disabled) {
	disabled = p_disabled;
	update();
	if (parent) {
		parent->shape_owner_set_disabled(owner_id, p_disabled);
	}
}

bool CollisionPolygon2DInstance::is_disabled() const {
	return disabled;
}

void CollisionPolygon2DInstance::set_one_way_collision(bool p_enable) {
	one_way_collision = p_enable;
	update();
	if (parent) {
		parent->shape_owner_set_one_way_collision(owner_id, p_enable);
	}
}

bool CollisionPolygon2DInstance::is_one_way_collision_enabled() const {

	return one_way_collision;
}

void CollisionPolygon2DInstance::_polygon_changed() {

	if (parent) {
		_build_polygon();
	}
	update();
	update_configuration_warning();
}

String CollisionPolygon2DInstance::get_configuration_warning() const {

	if (!is_visible_in_tree() || !is_inside_tree())
		return String();

	if (!Object::cast_to<CollisionObject2D>(get_parent())) {
		return TTR("CollisionPolygon2D only serves to provide a collision shape to a CollisionObject2D derived node. Please only use it as a child of Area2D, StaticBody2D, RigidBody2D, KinematicBody2D, etc. to give them a shape.");
	}

	if (!polygon.is_valid()) {
		return TTR("A CollisionPolygon2D resource must be set or created for this node to work. Please set a property or draw a polygon.");
	}

	if (polygon->is_empty()) {
		return TTR("An empty CollisionPolygon2D has no effect on collision.");
	}

	return String();
}

Rect2 CollisionPolygon2DInstance::get_item_rect() const {

	if (polygon.is_valid())
		return polygon->get_item_rect();
	else
		return Rect2();
}

void CollisionPolygon2DInstance::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_polygon", "polygon"), &CollisionPolygon2DInstance::set_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon"), &CollisionPolygon2DInstance::get_polygon);
	ClassDB::bind_method(D_METHOD("_polygon_changed"), &CollisionPolygon2DInstance::_polygon_changed);

	ClassDB::bind_method(D_METHOD("set_build_mode", "build_mode"), &CollisionPolygon2DInstance::set_build_mode);
	ClassDB::bind_method(D_METHOD("get_build_mode"), &CollisionPolygon2DInstance::get_build_mode);
	ClassDB::bind_method(D_METHOD("set_disabled", "disabled"), &CollisionPolygon2DInstance::set_disabled);
	ClassDB::bind_method(D_METHOD("is_disabled"), &CollisionPolygon2DInstance::is_disabled);
	ClassDB::bind_method(D_METHOD("set_one_way_collision", "enabled"), &CollisionPolygon2DInstance::set_one_way_collision);
	ClassDB::bind_method(D_METHOD("is_one_way_collision_enabled"), &CollisionPolygon2DInstance::is_one_way_collision_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "build_mode", PROPERTY_HINT_ENUM, "Solids,Segments"), "set_build_mode", "get_build_mode");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "one_way_collision"), "set_one_way_collision", "is_one_way_collision_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "polygon", PROPERTY_HINT_RESOURCE_TYPE, "CollisionPolygon2D"), "set_polygon", "get_polygon");
}

bool CollisionPolygon2DInstance::_has_resource() const {

	return polygon.is_valid();
}

void CollisionPolygon2DInstance::_create_resource(UndoRedo *undo_redo) {

	undo_redo->create_action(TTR("Create Collision Polygon"));
	undo_redo->add_do_method(this, "set_polygon", Ref<CollisionPolygon2D>(memnew(CollisionPolygon2D)));
	undo_redo->add_undo_method(this, "set_polygon", Variant(REF()));
	undo_redo->commit_action();
}

int CollisionPolygon2DInstance::get_polygon_count() const {

	return polygon.is_valid() ? 1 : 0;
}

Ref<AbstractPolygon2D> CollisionPolygon2DInstance::get_nth_polygon(int p_idx) const {

	return polygon;
}

void CollisionPolygon2DInstance::append_polygon(const Vector<Point2> &p_vertices) {

	Ref<CollisionPolygon2D> polygon = Ref<CollisionPolygon2D>(memnew(CollisionPolygon2D));
	polygon->set_vertices(p_vertices);
	set_polygon(polygon);
}

void CollisionPolygon2DInstance::add_polygon_at_index(int p_idx, Ref<AbstractPolygon2D> p_polygon) {

	set_polygon(p_polygon);
}

void CollisionPolygon2DInstance::set_vertices(int p_idx, const Vector<Point2> &p_vertices) {

	if (polygon.is_valid())
		polygon->set_vertices(p_vertices);
}

void CollisionPolygon2DInstance::remove_polygon(int p_idx) {

	set_polygon(Ref<CollisionPolygon2D>());
}

CollisionPolygon2DInstance::CollisionPolygon2DInstance() {

	set_notify_local_transform(true);
	parent = NULL;
	owner_id = 0;
	build_mode = BUILD_SOLIDS;
	disabled = false;
	one_way_collision = false;
}
