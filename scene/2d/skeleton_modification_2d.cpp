/*************************************************************************/
/*  skeleton_modification_2d.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "skeleton_modification_2d.h"
#include "scene/scene_string_names.h"

#include "scene/2d/skeleton_2d.h"

#include "scene/2d/collision_object_2d.h"
#include "scene/2d/collision_shape_2d.h"
#include "scene/2d/physical_bone_2d.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif // TOOLS_ENABLED

///////////////////////////////////////
// Modification2D
///////////////////////////////////////

void SkeletonModification2D::_validate_property(PropertyInfo &p_property) const {
	if (is_property_hidden(p_property.name)) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

void SkeletonModification2D::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	if (is_inside_tree()) {
		set_process_internal(enabled && (!Engine::get_singleton()->is_editor_hint() || run_in_editor));
	}
}

bool SkeletonModification2D::get_enabled() const {
	return enabled;
}

void SkeletonModification2D::set_run_in_editor(bool p_enabled_in_editor) {
	run_in_editor = p_enabled_in_editor;
	if (Engine::get_singleton()->is_editor_hint() && is_inside_tree()) {
		set_process_internal(enabled && run_in_editor);
	}
}

bool SkeletonModification2D::get_run_in_editor() const {
	return run_in_editor;
}

float SkeletonModification2D::clamp_angle(float p_angle, float p_min_bound, float p_max_bound, bool p_invert) {
	// Map to the 0 to 360 range (in radians though) instead of the -180 to 180 range.
	if (p_angle < 0) {
		p_angle = Math_TAU + p_angle;
	}

	// Make min and max in the range of 0 to 360 (in radians), and make sure they are in the right order
	if (p_min_bound < 0) {
		p_min_bound = Math_TAU + p_min_bound;
	}
	if (p_max_bound < 0) {
		p_max_bound = Math_TAU + p_max_bound;
	}
	if (p_min_bound > p_max_bound) {
		SWAP(p_min_bound, p_max_bound);
	}

	bool is_beyond_bounds = (p_angle < p_min_bound || p_angle > p_max_bound);
	bool is_within_bounds = (p_angle > p_min_bound && p_angle < p_max_bound);

	// Note: May not be the most optimal way to clamp, but it always constraints to the nearest angle.
	if ((!p_invert && is_beyond_bounds) || (p_invert && is_within_bounds)) {
		Vector2 min_bound_vec = Vector2(Math::cos(p_min_bound), Math::sin(p_min_bound));
		Vector2 max_bound_vec = Vector2(Math::cos(p_max_bound), Math::sin(p_max_bound));
		Vector2 angle_vec = Vector2(Math::cos(p_angle), Math::sin(p_angle));

		if (angle_vec.distance_squared_to(min_bound_vec) <= angle_vec.distance_squared_to(max_bound_vec)) {
			p_angle = p_min_bound;
		} else {
			p_angle = p_max_bound;
		}
	}

	return p_angle;
}

NodePath SkeletonModification2D::get_skeleton_path() const {
	return skeleton_path;
}

void SkeletonModification2D::set_skeleton_path(NodePath p_path) {
	if (p_path.is_empty()) {
		p_path = NodePath("..");
	}
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && get_skeleton()) {
		get_skeleton()->disconnect(SceneStringNames::get_singleton()->draw, Callable(this, SNAME("draw_editor_gizmo")));
	}
#endif
	skeleton_path = p_path;
	skeleton_change_queued = true;
	cached_skeleton = Variant();
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && get_skeleton()) {
		get_skeleton()->connect(SceneStringNames::get_singleton()->draw, Callable(this, SNAME("draw_editor_gizmo")));
	}
	update_configuration_warnings();
#endif
}

Skeleton2D *SkeletonModification2D::get_skeleton() const {
	Skeleton2D *skeleton_node = cast_to<Skeleton2D>(cached_skeleton);
	if (skeleton_node == nullptr) {
		skeleton_node = cast_to<Skeleton2D>(get_node_or_null(skeleton_path));
		cached_skeleton = skeleton_node;
	}
	return skeleton_node;
}

void SkeletonModification2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			set_process_internal(enabled && (!Engine::get_singleton()->is_editor_hint() || run_in_editor));
			cached_skeleton = Variant();
#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				call_deferred(SNAME("update_configuration_warnings"));
			}
#endif
		} break;
		case NOTIFICATION_READY: {
#ifdef TOOLS_ENABLED
			Skeleton2D *skel = get_skeleton();
			if (Engine::get_singleton()->is_editor_hint() && skel) {
				skel->connect(SceneStringNames::get_singleton()->draw, Callable(this, SNAME("draw_editor_gizmo")));
			}
#endif
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			ERR_FAIL_COND(!enabled);
			execute(get_process_delta_time());
		} break;
	}
}

void SkeletonModification2D::_do_gizmo_draw() {
#ifdef TOOLS_ENABLED
	if (editor_gizmo_dirty && Engine::get_singleton()->is_editor_hint()) {
		draw_editor_gizmo();
		Skeleton2D *skel = get_skeleton();
		if (skel) {
			skel->draw_set_transform(Vector2(0, 0));
		}
		editor_gizmo_dirty = false;
	}
#endif // TOOLS_ENABLED
}

void SkeletonModification2D::draw_editor_gizmo() {
	GDVIRTUAL_CALL(_draw_editor_gizmo);
}

void SkeletonModification2D::set_editor_gizmo_dirty(bool p_dirty) {
#ifdef TOOLS_ENABLED
	if (!editor_gizmo_dirty && p_dirty && Engine::get_singleton()->is_editor_hint()) {
		editor_gizmo_dirty = p_dirty;
		Skeleton2D *skeleton = get_skeleton();
		if (skeleton) {
			skeleton->queue_redraw();
		}
	} else {
		editor_gizmo_dirty = p_dirty;
	}
#endif
}

Variant SkeletonModification2D::resolve_node(const NodePath &target_node_path) const {
	Node *resolved_node = get_node(target_node_path);
	if (cast_to<CanvasItem>(resolved_node)) {
		return Variant(resolved_node);
	}
	return Variant(false);
}

Variant SkeletonModification2D::resolve_bone(const NodePath &target_node_path) const {
	Node *resolved_node = get_node(target_node_path);
	if (cast_to<Bone2D>(resolved_node)) {
		return Variant(resolved_node);
	}
	return Variant(false);
}

bool SkeletonModification2D::_cache_node(Variant &cache, const NodePath &target_node_path) const {
	if (cache.get_type() == Variant::NIL) {
		cache = resolve_node(target_node_path);
	}
	return cache.get_type() == Variant::OBJECT;
}

Bone2D *SkeletonModification2D::_cache_bone(Variant &cache, const NodePath &target_node_path) const {
	if (cache.get_type() == Variant::NIL) {
		cache = resolve_node(target_node_path);
	}
	if (cache.get_type() == Variant::OBJECT) {
		return cast_to<Bone2D>((Object *)cache);
	}
	return nullptr;
}

Transform2D SkeletonModification2D::get_target_transform(Variant resolved_target) const {
	if (resolved_target.get_type() == Variant::OBJECT) {
		CanvasItem *resolved_node = cast_to<CanvasItem>((Object *)resolved_target);
		return resolved_node->get_global_transform();
	}
	ERR_FAIL_V_MSG(Transform2D(), "Looking up transform of unresolved target.");
}

real_t SkeletonModification2D::get_target_rotation(Variant resolved_target) const {
	if (resolved_target.get_type() == Variant::OBJECT) {
		CanvasItem *resolved_node = cast_to<CanvasItem>((Object *)resolved_target);
		return resolved_node->get_global_transform().get_rotation();
	}
	ERR_FAIL_V_MSG(0.0f, "Looking up quaternion of unresolved target.");
}

Vector2 SkeletonModification2D::get_target_position(Variant resolved_target) const {
	if (resolved_target.get_type() == Variant::OBJECT) {
		CanvasItem *resolved_node = cast_to<CanvasItem>((Object *)resolved_target);
		return resolved_node->get_global_transform().get_origin();
	}
	ERR_FAIL_V_MSG(Vector2(), "Looking up quaternion of unresolved target.");
}

void SkeletonModification2D::editor_draw_angle_constraints(Bone2D *p_operation_bone, float p_min_bound, float p_max_bound,
		bool p_constraint_enabled, bool p_constraint_in_localspace, bool p_constraint_inverted) {
	if (!p_operation_bone) {
		return;
	}
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	Color bone_ik_color = Color(1.0, 0.65, 0.0, 0.4);
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		bone_ik_color = EditorSettings::get_singleton()->get("editors/2d/bone_ik_color");
	}
#endif // TOOLS_ENABLED

	float arc_angle_min = p_min_bound;
	float arc_angle_max = p_max_bound;
	if (arc_angle_min < 0) {
		arc_angle_min = (Math_PI * 2) + arc_angle_min;
	}
	if (arc_angle_max < 0) {
		arc_angle_max = (Math_PI * 2) + arc_angle_max;
	}
	if (arc_angle_min > arc_angle_max) {
		SWAP(arc_angle_min, arc_angle_max);
	}
	arc_angle_min += p_operation_bone->get_bone_angle();
	arc_angle_max += p_operation_bone->get_bone_angle();

	if (p_constraint_enabled) {
		if (p_constraint_in_localspace) {
			Node *operation_bone_parent = p_operation_bone->get_parent();
			Bone2D *operation_bone_parent_bone = Object::cast_to<Bone2D>(operation_bone_parent);

			if (operation_bone_parent_bone) {
				skeleton->draw_set_transform(
						skeleton->to_local(p_operation_bone->get_global_position()),
						operation_bone_parent_bone->get_global_rotation() - skeleton->get_global_rotation());
			} else {
				skeleton->draw_set_transform(skeleton->to_local(p_operation_bone->get_global_position()));
			}
		} else {
			skeleton->draw_set_transform(skeleton->to_local(p_operation_bone->get_global_position()));
		}

		if (p_constraint_inverted) {
			skeleton->draw_arc(Vector2(0, 0), p_operation_bone->get_length(),
					arc_angle_min + (Math_PI * 2), arc_angle_max, 32, bone_ik_color, 1.0);
		} else {
			skeleton->draw_arc(Vector2(0, 0), p_operation_bone->get_length(),
					arc_angle_min, arc_angle_max, 32, bone_ik_color, 1.0);
		}
		skeleton->draw_line(Vector2(0, 0), Vector2(Math::cos(arc_angle_min), Math::sin(arc_angle_min)) * p_operation_bone->get_length(), bone_ik_color, 1.0);
		skeleton->draw_line(Vector2(0, 0), Vector2(Math::cos(arc_angle_max), Math::sin(arc_angle_max)) * p_operation_bone->get_length(), bone_ik_color, 1.0);

	} else {
		skeleton->draw_set_transform(skeleton->to_local(p_operation_bone->get_global_position()));
		skeleton->draw_arc(Vector2(0, 0), p_operation_bone->get_length(), 0, Math_PI * 2, 32, bone_ik_color, 1.0);
		skeleton->draw_line(Vector2(0, 0), Vector2(1, 0) * p_operation_bone->get_length(), bone_ik_color, 1.0);
	}
}

void SkeletonModification2D::_bind_methods() {
	GDVIRTUAL_BIND(_execute, "delta");
	GDVIRTUAL_BIND(_draw_editor_gizmo);
	GDVIRTUAL_BIND(_is_property_hidden, "property_name");

	ClassDB::bind_method(D_METHOD("set_skeleton_path", "path"), &SkeletonModification2D::set_skeleton_path);
	ClassDB::bind_method(D_METHOD("get_skeleton_path"), &SkeletonModification2D::get_skeleton_path);
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SkeletonModification2D::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &SkeletonModification2D::get_enabled);
	ClassDB::bind_method(D_METHOD("set_run_in_editor", "enabled_in_editor"), &SkeletonModification2D::set_run_in_editor);
	ClassDB::bind_method(D_METHOD("get_run_in_editor"), &SkeletonModification2D::get_run_in_editor);
	ClassDB::bind_method(D_METHOD("execute", "delta"), &SkeletonModification2D::execute);
	ClassDB::bind_method(D_METHOD("draw_editor_gizmo"), &SkeletonModification2D::_do_gizmo_draw);
	ClassDB::bind_method(D_METHOD("set_editor_gizmo_dirty", "is_dirty"), &SkeletonModification2D::set_editor_gizmo_dirty);

	ClassDB::bind_method(D_METHOD("clamp_angle", "angle", "min", "max", "invert"), &SkeletonModification2D::clamp_angle);
	ClassDB::bind_method(D_METHOD("editor_draw_angle_constraints", "p_operation_bone", "min_bound", "max_bound", "constraint_enabled", "constraint_in_localspace", "constraint_inverted"), &SkeletonModification2D::editor_draw_angle_constraints);

	ClassDB::bind_method(D_METHOD("resolve_node", "target_node_path"), &SkeletonModification2D::resolve_node);
	ClassDB::bind_method(D_METHOD("resolve_bone", "target_bone_path"), &SkeletonModification2D::resolve_bone);
	ClassDB::bind_method(D_METHOD("get_target_transform", "resolved_target"), &SkeletonModification2D::get_target_transform);
	ClassDB::bind_method(D_METHOD("get_target_rotation", "resolved_target"), &SkeletonModification2D::get_target_rotation);
	ClassDB::bind_method(D_METHOD("get_target_position", "resolved_target"), &SkeletonModification2D::get_target_position);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "run_in_editor"), "set_run_in_editor", "get_run_in_editor");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "skeleton_path"), "set_skeleton_path", "get_skeleton_path");
}

void SkeletonModification2D::execute(real_t p_delta) {
	GDVIRTUAL_CALL(_execute, p_delta);
}

bool SkeletonModification2D::is_property_hidden(String p_property_name) const {
	bool ret = false;
	const_cast<SkeletonModification2D *>(this)->GDVIRTUAL_CALL(_is_property_hidden, p_property_name, ret);
	return ret;
}

PackedStringArray SkeletonModification2D::get_configuration_warnings() const {
	PackedStringArray ret = Node::get_configuration_warnings();
	if (!get_skeleton()) {
		ret.push_back("Modification skeleton_path must point to a Skeleton2D node.");
	}
	return ret;
}
