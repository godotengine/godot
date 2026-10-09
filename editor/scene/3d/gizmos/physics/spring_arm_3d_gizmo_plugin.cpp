/**************************************************************************/
/*  spring_arm_3d_gizmo_plugin.cpp                                        */
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

#include "spring_arm_3d_gizmo_plugin.h"

#include "core/math/convex_hull.h"
#include "core/math/geometry_3d.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/3d/gizmos/gizmo_3d_helper.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "scene/3d/physics/spring_arm_3d.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/3d/box_shape_3d.h"
#include "scene/resources/3d/capsule_shape_3d.h"
#include "scene/resources/3d/cylinder_shape_3d.h"
#include "scene/resources/3d/sphere_shape_3d.h"

SpringArm3DGizmoPlugin::SpringArm3DGizmoPlugin() {
	helper.instantiate();

	show_only_when_selected = EDITOR_GET("editors/3d_gizmos/gizmo_settings/show_collision_shapes_only_when_selected");

	Color gizmo_color = SceneTree::get_singleton()->get_debug_collisions_color();
	create_material("shape_material", gizmo_color);

	create_handle_material("handles");
}

bool SpringArm3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<SpringArm3D>(p_spatial) != nullptr;
}

String SpringArm3DGizmoPlugin::get_gizmo_name() const {
	return "SpringArm3D";
}

int SpringArm3DGizmoPlugin::get_priority() const {
	return -1;
}

String SpringArm3DGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	const SpringArm3D *spring_arm = Object::cast_to<SpringArm3D>(p_gizmo->get_node_3d());

	Ref<Shape3D> s = spring_arm->get_shape();
	if (s.is_null()) {
		return "";
	}

	if (Object::cast_to<SphereShape3D>(*s)) {
		return "Radius";
	}

	if (Object::cast_to<BoxShape3D>(*s)) {
		return helper->box_get_handle_name(p_id);
	}

	if (Object::cast_to<CapsuleShape3D>(*s)) {
		return helper->capsule_get_handle_name(p_id);
	}

	if (Object::cast_to<CylinderShape3D>(*s)) {
		return helper->cylinder_get_handle_name(p_id);
	}

	return "";
}

Variant SpringArm3DGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	SpringArm3D *spring_arm = Object::cast_to<SpringArm3D>(p_gizmo->get_node_3d());

	Ref<Shape3D> s = spring_arm->get_shape();
	if (s.is_null()) {
		return Variant();
	}

	if (Object::cast_to<SphereShape3D>(*s)) {
		Ref<SphereShape3D> ss = s;
		return ss->get_radius();
	}

	if (Object::cast_to<BoxShape3D>(*s)) {
		Ref<BoxShape3D> bs = s;
		return bs->get_size();
	}

	if (Object::cast_to<CapsuleShape3D>(*s)) {
		Ref<CapsuleShape3D> cs2 = s;
		return Vector2(cs2->get_radius(), cs2->get_height());
	}

	if (Object::cast_to<CylinderShape3D>(*s)) {
		Ref<CylinderShape3D> cs2 = s;
		return Vector2(cs2->get_radius(), cs2->get_height());
	}

	return Variant();
}

void SpringArm3DGizmoPlugin::begin_handle_action(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) {
	helper->initialize_handle_action(get_handle_value(p_gizmo, p_id, p_secondary), p_gizmo->get_node_3d()->get_global_transform());
}

void SpringArm3DGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	SpringArm3D *spring_arm = Object::cast_to<SpringArm3D>(p_gizmo->get_node_3d());

	Ref<Shape3D> s = spring_arm->get_shape();
	if (s.is_null()) {
		return;
	}

	Vector3 sg[2];
	helper->get_segment(p_camera, p_point, sg);

	Vector3 position_offset(0, 0, spring_arm->get_length());
	sg[0] = sg[0] - position_offset;
	sg[1] = sg[1] - position_offset;

	if (Object::cast_to<SphereShape3D>(*s)) {
		Ref<SphereShape3D> ss = s;
		Vector3 ra, rb;
		Geometry3D::get_closest_points_between_segments(Vector3(), Vector3(4096, 0, 0), sg[0], sg[1], ra, rb);
		float d = (ra - position_offset).x;
		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		ss->set_radius(d);
	}

	if (Object::cast_to<BoxShape3D>(*s)) {
		Ref<BoxShape3D> bs = s;
		Vector3 size = bs->get_size();
		Vector3 position;
		helper->box_set_handle(sg, p_id, size, position);
		bs->set_size(size);
		spring_arm->set_global_position(position);
	}

	if (Object::cast_to<CapsuleShape3D>(*s)) {
		Ref<CapsuleShape3D> cs2 = s;

		real_t height = cs2->get_height();
		real_t radius = cs2->get_radius();
		Vector3 position;
		helper->capsule_set_handle(sg, p_id, height, radius, position);
		cs2->set_height(height);
		cs2->set_radius(radius);
		spring_arm->set_global_position(position);
	}

	if (Object::cast_to<CylinderShape3D>(*s)) {
		Ref<CylinderShape3D> cs2 = s;

		real_t height = cs2->get_height();
		real_t radius = cs2->get_radius();
		Vector3 position;
		helper->cylinder_set_handle(sg, p_id, height, radius, position);
		cs2->set_height(height);
		cs2->set_radius(radius);
		spring_arm->set_global_position(position);
	}
}

void SpringArm3DGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	SpringArm3D *spring_arm = Object::cast_to<SpringArm3D>(p_gizmo->get_node_3d());

	Ref<Shape3D> s = spring_arm->get_shape();
	if (s.is_null()) {
		return;
	}

	if (Object::cast_to<SphereShape3D>(*s)) {
		Ref<SphereShape3D> ss = s;
		if (p_cancel) {
			ss->set_radius(p_restore);
			return;
		}

		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Change Sphere Shape Radius"));
		ur->add_do_method(ss.ptr(), "set_radius", ss->get_radius());
		ur->add_undo_method(ss.ptr(), "set_radius", p_restore);
		ur->commit_action();
	}

	if (Object::cast_to<BoxShape3D>(*s)) {
		helper->box_commit_handle(TTR("Change Box Shape Size"), p_cancel, spring_arm, s.ptr());
	}

	if (Object::cast_to<CapsuleShape3D>(*s)) {
		Ref<CapsuleShape3D> ss = s;
		helper->cylinder_commit_handle(p_id, TTR("Change Capsule Shape Radius"), TTR("Change Capsule Shape Height"), p_cancel, spring_arm, *ss, *ss);
	}

	if (Object::cast_to<CylinderShape3D>(*s)) {
		Ref<CylinderShape3D> ss = s;
		helper->cylinder_commit_handle(p_id, TTR("Change Cylinder Shape Radius"), TTR("Change Cylinder Shape Height"), p_cancel, spring_arm, *ss, *ss);
	}
}

void SpringArm3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	SpringArm3D *spring_arm = Object::cast_to<SpringArm3D>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	Vector<Vector3> spring_arm_lines = {
		Vector3(),
		Vector3(0, 0, 1.0) * spring_arm->get_length()
	};

	Ref<StandardMaterial3D> material = get_material("shape_material", p_gizmo);

	p_gizmo->add_lines(spring_arm_lines, material);
	p_gizmo->add_collision_segments(spring_arm_lines);

	Ref<Shape3D> s = spring_arm->get_shape();
	if (s.is_null()) {
		return;
	}

	if (show_only_when_selected && !p_gizmo->is_selected()) {
		return;
	}

	const Ref<Material> handles_material = get_material("handles");

	if (Object::cast_to<SphereShape3D>(*s)) {
		Ref<SphereShape3D> sp = s;
		float radius = sp->get_radius();
		Vector3 position_offset(0, 0, spring_arm->get_length());

#define PUSH_QUARTER(from_x, from_y, to_x, to_y, y) \
	points_ptrw[index++] = Vector3(from_x, y, from_y) + position_offset; \
	points_ptrw[index++] = Vector3(to_x, y, to_y) + position_offset; \
	points_ptrw[index++] = Vector3(from_x, y, -from_y) + position_offset; \
	points_ptrw[index++] = Vector3(to_x, y, -to_y) + position_offset; \
	points_ptrw[index++] = Vector3(-from_x, y, from_y) + position_offset; \
	points_ptrw[index++] = Vector3(-to_x, y, to_y) + position_offset; \
	points_ptrw[index++] = Vector3(-from_x, y, -from_y) + position_offset; \
	points_ptrw[index++] = Vector3(-to_x, y, -to_y) + position_offset;

#define PUSH_QUARTER_XY(from_x, from_y, to_x, to_y, y) \
	points_ptrw[index++] = Vector3(from_x, -from_y - y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(to_x, -to_y - y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(from_x, from_y + y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(to_x, to_y + y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(-from_x, -from_y - y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(-to_x, -to_y - y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(-from_x, from_y + y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(-to_x, to_y + y, 0) + position_offset;

#define PUSH_QUARTER_YZ(from_x, from_y, to_x, to_y, y) \
	points_ptrw[index++] = Vector3(0, -from_y - y, from_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, -to_y - y, to_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, from_y + y, from_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, to_y + y, to_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, -from_y - y, -from_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, -to_y - y, -to_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, from_y + y, -from_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, to_y + y, -to_x) + position_offset;

		// Number of points in an octant. So there will be 8 * points_in_octant * 2 points in total for one circle.
		// This Corresponds to the smoothness of the circle.
		const uint32_t points_in_octant = 16;
		const real_t inc = (Math::PI / (4 * points_in_octant));
		const real_t radius_squared = radius * radius;
		real_t r = 0;

		Vector<Vector3> points;
		uint32_t index = 0;
		// 3 full circles.
		points.resize(3 * 8 * points_in_octant * 2);
		Vector3 *points_ptrw = points.ptrw();

		float previous_x = radius;
		float previous_y = 0.f;

		for (uint32_t i = 0; i < points_in_octant; ++i) {
			r += inc;
			real_t x = Math::cos(r) * radius;
			real_t y = Math::sqrt(radius_squared - (x * x));

			PUSH_QUARTER(previous_x, previous_y, x, y, 0);
			PUSH_QUARTER(previous_y, previous_x, y, x, 0);

			PUSH_QUARTER_XY(previous_x, previous_y, x, y, 0);
			PUSH_QUARTER_XY(previous_y, previous_x, y, x, 0);

			PUSH_QUARTER_YZ(previous_x, previous_y, x, y, 0);
			PUSH_QUARTER_YZ(previous_y, previous_x, y, x, 0)

			previous_x = x;
			previous_y = y;
		}
#undef PUSH_QUARTER
#undef PUSH_QUARTER_XY
#undef PUSH_QUARTER_YZ

		p_gizmo->add_lines(points, material, false);
		p_gizmo->add_collision_segments(points);
		Vector<Vector3> handles;
		handles.push_back(Vector3(radius, 0, 0) + position_offset);
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<BoxShape3D>(*s)) {
		Ref<BoxShape3D> bs = s;
		Vector<Vector3> lines;
		AABB aabb;
		Vector3 position_offset(0, 0, spring_arm->get_length());
		aabb.position = (-bs->get_size() / 2) + position_offset;
		aabb.size = bs->get_size();

		for (int i = 0; i < 12; i++) {
			Vector3 a, b;
			aabb.get_edge(i, a, b);
			lines.push_back(a);
			lines.push_back(b);
		}

		Vector<Vector3> handles = helper->box_get_handles(bs->get_size());
		for (Vector3 &h : handles) {
			h += position_offset;
		}

		p_gizmo->add_lines(lines, material, false);
		p_gizmo->add_collision_segments(lines);
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<CapsuleShape3D>(*s)) {
		Ref<CapsuleShape3D> cs2 = s;
		float radius = cs2->get_radius();
		float height = cs2->get_height();
		Vector3 position_offset(0, 0, spring_arm->get_length());

		// Number of points in an octant. So there will be 8 * points_in_octant points in total.
		// This corresponds to the smoothness of the circle.
		const uint32_t points_in_octant = 16;
		const real_t octant_angle = Math::PI / 4;
		const real_t inc = (Math::PI / (4 * points_in_octant));
		const real_t radius_squared = radius * radius;
		real_t r = 0;

		Vector<Vector3> points;
		// 4 vertical lines and 4 full circles.
		points.resize(4 * 2 + 4 * 8 * points_in_octant * 2);
		Vector3 *points_ptrw = points.ptrw();

		uint32_t index = 0;
		float y_value = height * 0.5 - radius;

		// Vertical Lines.
		points_ptrw[index++] = Vector3(0.f, y_value, radius) + position_offset;
		points_ptrw[index++] = Vector3(0.f, -y_value, radius) + position_offset;
		points_ptrw[index++] = Vector3(0.f, y_value, -radius) + position_offset;
		points_ptrw[index++] = Vector3(0.f, -y_value, -radius) + position_offset;
		points_ptrw[index++] = Vector3(radius, y_value, 0.f) + position_offset;
		points_ptrw[index++] = Vector3(radius, -y_value, 0.f) + position_offset;
		points_ptrw[index++] = Vector3(-radius, y_value, 0.f) + position_offset;
		points_ptrw[index++] = Vector3(-radius, -y_value, 0.f) + position_offset;

#define PUSH_QUARTER(from_x, from_y, to_x, to_y, y) \
	points_ptrw[index++] = Vector3(from_x, y, from_y) + position_offset; \
	points_ptrw[index++] = Vector3(to_x, y, to_y) + position_offset; \
	points_ptrw[index++] = Vector3(from_x, y, -from_y) + position_offset; \
	points_ptrw[index++] = Vector3(to_x, y, -to_y) + position_offset; \
	points_ptrw[index++] = Vector3(-from_x, y, from_y) + position_offset; \
	points_ptrw[index++] = Vector3(-to_x, y, to_y) + position_offset; \
	points_ptrw[index++] = Vector3(-from_x, y, -from_y) + position_offset; \
	points_ptrw[index++] = Vector3(-to_x, y, -to_y) + position_offset;

#define PUSH_QUARTER_XY(from_x, from_y, to_x, to_y, y) \
	points_ptrw[index++] = Vector3(from_x, -from_y - y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(to_x, -to_y - y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(from_x, from_y + y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(to_x, to_y + y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(-from_x, -from_y - y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(-to_x, -to_y - y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(-from_x, from_y + y, 0) + position_offset; \
	points_ptrw[index++] = Vector3(-to_x, to_y + y, 0) + position_offset;

#define PUSH_QUARTER_YZ(from_x, from_y, to_x, to_y, y) \
	points_ptrw[index++] = Vector3(0, -from_y - y, from_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, -to_y - y, to_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, from_y + y, from_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, to_y + y, to_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, -from_y - y, -from_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, -to_y - y, -to_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, from_y + y, -from_x) + position_offset; \
	points_ptrw[index++] = Vector3(0, to_y + y, -to_x) + position_offset;

		float previous_x = radius;
		float previous_y = 0.f;

		for (uint32_t i = 0; i < points_in_octant; ++i) {
			r += inc;
			real_t x = Math::cos((i == points_in_octant - 1) ? octant_angle : r) * radius;
			real_t y = Math::sqrt(radius_squared - (x * x));

			// High circle ring.
			PUSH_QUARTER(previous_x, previous_y, x, y, y_value);
			PUSH_QUARTER(previous_y, previous_x, y, x, y_value);

			// Low circle ring.
			PUSH_QUARTER(previous_x, previous_y, x, y, -y_value);
			PUSH_QUARTER(previous_y, previous_x, y, x, -y_value);

			// Up and Low circle in X-Y plane.
			PUSH_QUARTER_XY(previous_x, previous_y, x, y, y_value);
			PUSH_QUARTER_XY(previous_y, previous_x, y, x, y_value);

			// Up and Low circle in Y-Z plane.
			PUSH_QUARTER_YZ(previous_x, previous_y, x, y, y_value);
			PUSH_QUARTER_YZ(previous_y, previous_x, y, x, y_value)

			previous_x = x;
			previous_y = y;
		}

#undef PUSH_QUARTER
#undef PUSH_QUARTER_XY
#undef PUSH_QUARTER_YZ

		p_gizmo->add_lines(points, material, false);
		p_gizmo->add_collision_segments(points);

		Vector<Vector3> handles = helper->capsule_get_handles(cs2->get_height(), cs2->get_radius());
		for (Vector3 &h : handles) {
			h += position_offset;
		}
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<CylinderShape3D>(*s)) {
		Ref<CylinderShape3D> cs2 = s;
		float radius = cs2->get_radius();
		float height = cs2->get_height();
		Vector3 position_offset(0, 0, spring_arm->get_length());

#define PUSH_QUARTER(from_x, from_y, to_x, to_y, y) \
	points_ptrw[index++] = Vector3(from_x, y, from_y) + position_offset; \
	points_ptrw[index++] = Vector3(to_x, y, to_y) + position_offset; \
	points_ptrw[index++] = Vector3(from_x, y, -from_y) + position_offset; \
	points_ptrw[index++] = Vector3(to_x, y, -to_y) + position_offset; \
	points_ptrw[index++] = Vector3(-from_x, y, from_y) + position_offset; \
	points_ptrw[index++] = Vector3(-to_x, y, to_y) + position_offset; \
	points_ptrw[index++] = Vector3(-from_x, y, -from_y) + position_offset; \
	points_ptrw[index++] = Vector3(-to_x, y, -to_y) + position_offset;

		// Number of points in an octant. So there will be 8 * points_in_octant * 2 points in total for one circle.
		// This corresponds to the smoothness of the circle.
		const uint32_t points_in_octant = 16;
		const real_t inc = (Math::PI / (4 * points_in_octant));
		const real_t radius_squared = radius * radius;
		real_t r = 0;

		Vector<Vector3> points;
		uint32_t index = 0;
		// 4 vertical lines and 2 full circles.
		points.resize(4 * 2 + 2 * 8 * points_in_octant * 2);
		Vector3 *points_ptrw = points.ptrw();
		float y_value = height * 0.5;

		// Vertical lines.
		points_ptrw[index++] = Vector3(0.f, y_value, radius) + position_offset;
		points_ptrw[index++] = Vector3(0.f, -y_value, radius) + position_offset;
		points_ptrw[index++] = Vector3(0.f, y_value, -radius) + position_offset;
		points_ptrw[index++] = Vector3(0.f, -y_value, -radius) + position_offset;
		points_ptrw[index++] = Vector3(radius, y_value, 0.f) + position_offset;
		points_ptrw[index++] = Vector3(radius, -y_value, 0.f) + position_offset;
		points_ptrw[index++] = Vector3(-radius, y_value, 0.f) + position_offset;
		points_ptrw[index++] = Vector3(-radius, -y_value, 0.f) + position_offset;

		float previous_x = radius;
		float previous_y = 0.f;

		for (uint32_t i = 0; i < points_in_octant; ++i) {
			r += inc;
			real_t x = Math::cos(r) * radius;
			real_t y = Math::sqrt(radius_squared - (x * x));

			// High circle ring.
			PUSH_QUARTER(previous_x, previous_y, x, y, y_value);
			PUSH_QUARTER(previous_y, previous_x, y, x, y_value);

			// Low circle ring.
			PUSH_QUARTER(previous_x, previous_y, x, y, -y_value);
			PUSH_QUARTER(previous_y, previous_x, y, x, -y_value);

			previous_x = x;
			previous_y = y;
		}
#undef PUSH_QUARTER

		p_gizmo->add_lines(points, material, false);
		p_gizmo->add_collision_segments(points);

		Vector<Vector3> handles = helper->cylinder_get_handles(cs2->get_height(), cs2->get_radius());
		for (Vector3 &h : handles) {
			h += position_offset;
		}
		p_gizmo->add_handles(handles, handles_material);
	}
}

void SpringArm3DGizmoPlugin::set_show_only_when_selected(bool p_enabled) {
	show_only_when_selected = p_enabled;
}
