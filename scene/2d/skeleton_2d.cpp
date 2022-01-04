/*************************************************************************/
/*  skeleton_2d.cpp                                                      */
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

#include "skeleton_2d.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#include "editor/plugins/canvas_item_editor_plugin.h"
#endif //TOOLS_ENABLED

bool Bone2D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("auto_calculate_length_and_angle")) {
		set_autocalculate_length_and_angle(p_value);
	} else if (path.begins_with("length")) {
		set_length(p_value);
	} else if (path.begins_with("bone_angle")) {
		set_bone_angle(Math::deg2rad(real_t(p_value)));
	} else if (path.begins_with("default_length")) {
		set_length(p_value);
	}

#ifdef TOOLS_ENABLED
	if (path.begins_with("editor_settings/show_bone_gizmo")) {
		_editor_set_show_bone_gizmo(p_value);
	}
#endif // TOOLS_ENABLED

	return true;
}

bool Bone2D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("auto_calculate_length_and_angle")) {
		r_ret = get_autocalculate_length_and_angle();
	} else if (path.begins_with("length")) {
		r_ret = get_length();
	} else if (path.begins_with("bone_angle")) {
		r_ret = Math::rad2deg(get_bone_angle());
	} else if (path.begins_with("default_length")) {
		r_ret = get_length();
	}

#ifdef TOOLS_ENABLED
	if (path.begins_with("editor_settings/show_bone_gizmo")) {
		r_ret = _editor_get_show_bone_gizmo();
	}
#endif // TOOLS_ENABLED

	return true;
}

void Bone2D::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::BOOL, "auto_calculate_length_and_angle", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	if (!autocalculate_length_and_angle) {
		p_list->push_back(PropertyInfo(Variant::FLOAT, "length", PROPERTY_HINT_RANGE, "1, 1024, 1", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "bone_angle", PROPERTY_HINT_RANGE, "-360, 360, 0.01", PROPERTY_USAGE_DEFAULT));
	}

#ifdef TOOLS_ENABLED
	p_list->push_back(PropertyInfo(Variant::BOOL, "editor_settings/show_bone_gizmo", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
#endif // TOOLS_ENABLED
}

void Bone2D::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		Node *parent = get_parent();
		parent_bone = Object::cast_to<Bone2D>(parent);
		skeleton = nullptr;
		while (parent) {
			skeleton = Object::cast_to<Skeleton2D>(parent);
			if (skeleton) {
				break;
			}
			if (!Object::cast_to<Bone2D>(parent)) {
				break; //skeletons must be chained to Bone2Ds.
			}

			parent = parent->get_parent();
		}

		if (skeleton) {
			Skeleton2D::Bone bone;
			bone.bone = this;
			skeleton->bones.push_back(bone);
			skeleton->_make_bone_setup_dirty();
		}

		cache_transform = get_transform();
		copy_transform_to_cache = true;

#ifdef TOOLS_ENABLED
		// Only draw the gizmo in the editor!
		if (Engine::get_singleton()->is_editor_hint() == false) {
			return;
		}

		update();
#endif // TOOLS_ENABLED
	}

	else if (p_what == NOTIFICATION_LOCAL_TRANSFORM_CHANGED) {
		if (skeleton) {
			skeleton->_make_transform_dirty();
		}
		if (copy_transform_to_cache) {
			cache_transform = get_transform();
		}
#ifdef TOOLS_ENABLED
		// Only draw the gizmo in the editor!
		if (Engine::get_singleton()->is_editor_hint() == false) {
			return;
		}

		update();

		if (get_parent()) {
			Bone2D *parent_bone = Object::cast_to<Bone2D>(get_parent());
			if (parent_bone) {
				parent_bone->update();
			}
		}
#endif // TOOLS_ENABLED
	}

	else if (p_what == NOTIFICATION_MOVED_IN_PARENT) {
		if (skeleton) {
			skeleton->_make_bone_setup_dirty();
		}
		if (copy_transform_to_cache) {
			cache_transform = get_transform();
		}
	}

	else if (p_what == NOTIFICATION_EXIT_TREE) {
		if (skeleton) {
			for (int i = 0; i < skeleton->bones.size(); i++) {
				if (skeleton->bones[i].bone == this) {
					skeleton->bones.remove_at(i);
					break;
				}
			}
			skeleton->_make_bone_setup_dirty();
			skeleton = nullptr;
		}
		parent_bone = nullptr;
		set_transform(cache_transform);
	}

	else if (p_what == NOTIFICATION_READY) {
		if (autocalculate_length_and_angle) {
			calculate_length_and_rotation();
		}
	}
#ifdef TOOLS_ENABLED
	else if (p_what == NOTIFICATION_EDITOR_PRE_SAVE || p_what == NOTIFICATION_EDITOR_POST_SAVE) {
		Transform2D tmp_trans = get_transform();
		set_transform(cache_transform);
		cache_transform = tmp_trans;
	}
	// Bone2D Editor gizmo drawing:
#ifndef _MSC_VER
#warning TODO Bone2D gizmo drawing needs to be moved to an editor plugin
#endif
	else if (p_what == NOTIFICATION_DRAW) {
		// Only draw the gizmo in the editor!
		if (Engine::get_singleton()->is_editor_hint() == false) {
			return;
		}

		if (editor_gizmo_rid.is_null()) {
			editor_gizmo_rid = RenderingServer::get_singleton()->canvas_item_create();
			RenderingServer::get_singleton()->canvas_item_set_parent(editor_gizmo_rid, get_canvas_item());
			RenderingServer::get_singleton()->canvas_item_set_z_as_relative_to_parent(editor_gizmo_rid, true);
			RenderingServer::get_singleton()->canvas_item_set_z_index(editor_gizmo_rid, 10);
		}
		RenderingServer::get_singleton()->canvas_item_clear(editor_gizmo_rid);

		if (!_editor_show_bone_gizmo) {
			return;
		}

		// Undo scaling
		Transform2D editor_gizmo_trans = Transform2D();
		editor_gizmo_trans.set_scale(Vector2(1, 1) / get_global_scale());
		RenderingServer::get_singleton()->canvas_item_set_transform(editor_gizmo_rid, editor_gizmo_trans);

		Color bone_color1 = EditorSettings::get_singleton()->get("editors/2d/bone_color1");
		Color bone_color2 = EditorSettings::get_singleton()->get("editors/2d/bone_color2");
		Color bone_ik_color = EditorSettings::get_singleton()->get("editors/2d/bone_ik_color");
		Color bone_outline_color = EditorSettings::get_singleton()->get("editors/2d/bone_outline_color");
		Color bone_selected_color = EditorSettings::get_singleton()->get("editors/2d/bone_selected_color");

		bool Bone2D_found = false;
		for (int i = 0; i < get_child_count(); i++) {
			Bone2D *child_node = nullptr;
			child_node = Object::cast_to<Bone2D>(get_child(i));
			if (!child_node) {
				continue;
			}
			Bone2D_found = true;

			Vector<Vector2> bone_shape;
			Vector<Vector2> bone_shape_outline;

			_editor_get_bone_shape(&bone_shape, &bone_shape_outline, child_node);

			Vector<Color> colors;
			if (has_meta("_local_pose_override_enabled_")) {
				colors.push_back(bone_ik_color);
				colors.push_back(bone_ik_color);
				colors.push_back(bone_ik_color);
				colors.push_back(bone_ik_color);
			} else {
				colors.push_back(bone_color1);
				colors.push_back(bone_color2);
				colors.push_back(bone_color1);
				colors.push_back(bone_color2);
			}

			Vector<Color> outline_colors;
			if (CanvasItemEditor::get_singleton()->editor_selection->is_selected(this)) {
				outline_colors.push_back(bone_selected_color);
				outline_colors.push_back(bone_selected_color);
				outline_colors.push_back(bone_selected_color);
				outline_colors.push_back(bone_selected_color);
				outline_colors.push_back(bone_selected_color);
				outline_colors.push_back(bone_selected_color);
			} else {
				outline_colors.push_back(bone_outline_color);
				outline_colors.push_back(bone_outline_color);
				outline_colors.push_back(bone_outline_color);
				outline_colors.push_back(bone_outline_color);
				outline_colors.push_back(bone_outline_color);
				outline_colors.push_back(bone_outline_color);
			}

			RenderingServer::get_singleton()->canvas_item_add_polygon(editor_gizmo_rid, bone_shape_outline, outline_colors);
			RenderingServer::get_singleton()->canvas_item_add_polygon(editor_gizmo_rid, bone_shape, colors);
		}

		if (!Bone2D_found) {
			Vector<Vector2> bone_shape;
			Vector<Vector2> bone_shape_outline;

			_editor_get_bone_shape(&bone_shape, &bone_shape_outline, nullptr);

			Vector<Color> colors;
			if (has_meta("_local_pose_override_enabled_")) {
				colors.push_back(bone_ik_color);
				colors.push_back(bone_ik_color);
				colors.push_back(bone_ik_color);
				colors.push_back(bone_ik_color);
			} else {
				colors.push_back(bone_color1);
				colors.push_back(bone_color2);
				colors.push_back(bone_color1);
				colors.push_back(bone_color2);
			}

			Vector<Color> outline_colors;
			if (CanvasItemEditor::get_singleton()->editor_selection->is_selected(this)) {
				outline_colors.push_back(bone_selected_color);
				outline_colors.push_back(bone_selected_color);
				outline_colors.push_back(bone_selected_color);
				outline_colors.push_back(bone_selected_color);
				outline_colors.push_back(bone_selected_color);
				outline_colors.push_back(bone_selected_color);
			} else {
				outline_colors.push_back(bone_outline_color);
				outline_colors.push_back(bone_outline_color);
				outline_colors.push_back(bone_outline_color);
				outline_colors.push_back(bone_outline_color);
				outline_colors.push_back(bone_outline_color);
				outline_colors.push_back(bone_outline_color);
			}

			RenderingServer::get_singleton()->canvas_item_add_polygon(editor_gizmo_rid, bone_shape_outline, outline_colors);
			RenderingServer::get_singleton()->canvas_item_add_polygon(editor_gizmo_rid, bone_shape, colors);
		}
	}
#endif // TOOLS_ENALBED
}

#ifdef TOOLS_ENABLED
bool Bone2D::_editor_get_bone_shape(Vector<Vector2> *p_shape, Vector<Vector2> *p_outline_shape, Bone2D *p_other_bone) {
	int bone_width = EditorSettings::get_singleton()->get("editors/2d/bone_width");
	int bone_outline_width = EditorSettings::get_singleton()->get("editors/2d/bone_outline_size");

	if (!is_inside_tree()) {
		return false; //may have been removed
	}
	if (!p_other_bone && length <= 0) {
		return false;
	}

	Vector2 rel;
	if (p_other_bone) {
		rel = (p_other_bone->get_global_position() - get_global_position());
		rel = rel.rotated(-get_global_rotation()); // Undo Bone2D node's rotation so its drawn correctly regardless of the node's rotation
	} else {
		real_t angle_to_use = get_rotation() + bone_angle;
		rel = Vector2(cos(angle_to_use), sin(angle_to_use)) * (length * MIN(get_global_scale().x, get_global_scale().y));
		rel = rel.rotated(-get_rotation()); // Undo Bone2D node's rotation so its drawn correctly regardless of the node's rotation
	}

	Vector2 relt = rel.rotated(Math_PI * 0.5).normalized() * bone_width;
	Vector2 reln = rel.normalized();
	Vector2 reltn = relt.normalized();

	if (p_shape) {
		p_shape->clear();
		p_shape->push_back(Vector2(0, 0));
		p_shape->push_back(rel * 0.2 + relt);
		p_shape->push_back(rel);
		p_shape->push_back(rel * 0.2 - relt);
	}

	if (p_outline_shape) {
		p_outline_shape->clear();
		p_outline_shape->push_back((-reln - reltn) * bone_outline_width);
		p_outline_shape->push_back((-reln + reltn) * bone_outline_width);
		p_outline_shape->push_back(rel * 0.2 + relt + reltn * bone_outline_width);
		p_outline_shape->push_back(rel + (reln + reltn) * bone_outline_width);
		p_outline_shape->push_back(rel + (reln - reltn) * bone_outline_width);
		p_outline_shape->push_back(rel * 0.2 - relt - reltn * bone_outline_width);
	}
	return true;
}

void Bone2D::_editor_set_show_bone_gizmo(bool p_show_gizmo) {
	_editor_show_bone_gizmo = p_show_gizmo;
	update();
}

bool Bone2D::_editor_get_show_bone_gizmo() const {
	return _editor_show_bone_gizmo;
}
#endif // TOOLS_ENABLED

void Bone2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_rest", "rest"), &Bone2D::set_rest);
	ClassDB::bind_method(D_METHOD("get_rest"), &Bone2D::get_rest);
	ClassDB::bind_method(D_METHOD("apply_rest"), &Bone2D::apply_rest);
	ClassDB::bind_method(D_METHOD("get_skeleton_rest"), &Bone2D::get_skeleton_rest);
	ClassDB::bind_method(D_METHOD("get_index_in_skeleton"), &Bone2D::get_index_in_skeleton);

	ClassDB::bind_method(D_METHOD("set_default_length", "default_length"), &Bone2D::set_default_length);
	ClassDB::bind_method(D_METHOD("get_default_length"), &Bone2D::get_default_length);

	ClassDB::bind_method(D_METHOD("set_autocalculate_length_and_angle", "auto_calculate"), &Bone2D::set_autocalculate_length_and_angle);
	ClassDB::bind_method(D_METHOD("get_autocalculate_length_and_angle"), &Bone2D::get_autocalculate_length_and_angle);
	ClassDB::bind_method(D_METHOD("set_length", "length"), &Bone2D::set_length);
	ClassDB::bind_method(D_METHOD("get_length"), &Bone2D::get_length);
	ClassDB::bind_method(D_METHOD("set_bone_angle", "angle"), &Bone2D::set_bone_angle);
	ClassDB::bind_method(D_METHOD("get_bone_angle"), &Bone2D::get_bone_angle);

	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "rest"), "set_rest", "get_rest");
}

void Bone2D::set_rest(const Transform2D &p_rest) {
	rest = p_rest;
	if (skeleton) {
		skeleton->_make_bone_setup_dirty();
	}

	update_configuration_warnings();
}

Transform2D Bone2D::get_rest() const {
	return rest;
}

Transform2D Bone2D::get_skeleton_rest() const {
	if (parent_bone) {
		return parent_bone->get_skeleton_rest() * rest;
	} else {
		return rest;
	}
}

void Bone2D::apply_rest() {
	set_transform(rest);
}

void Bone2D::set_default_length(real_t p_length) {
	WARN_DEPRECATED_MSG("set_default_length is deprecated. Please use set_length instead!");
	set_length(p_length);
}

real_t Bone2D::get_default_length() const {
	WARN_DEPRECATED_MSG("get_default_length is deprecated. Please use get_length instead!");
	return get_length();
}

int Bone2D::get_index_in_skeleton() const {
	ERR_FAIL_COND_V(!skeleton, -1);
	skeleton->_update_bone_setup();
	return skeleton_index;
}

TypedArray<String> Bone2D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();
	if (!skeleton) {
		if (parent_bone) {
			warnings.push_back(TTR("This Bone2D chain should end at a Skeleton2D node."));
		} else {
			warnings.push_back(TTR("A Bone2D only works with a Skeleton2D or another Bone2D as parent node."));
		}
	}

	if (rest == Transform2D(0, 0, 0, 0, 0, 0)) {
		warnings.push_back(TTR("This bone lacks a proper REST pose. Go to the Skeleton2D node and set one."));
	}

	return warnings;
}

void Bone2D::calculate_length_and_rotation() {
	// if there is at least a single child Bone2D node, we can calculate
	// the length and direction. We will always just use the first Bone2D for this.
	bool calculated = false;
	int child_count = get_child_count();
	if (child_count > 0) {
		for (int i = 0; i < child_count; i++) {
			Bone2D *child = Object::cast_to<Bone2D>(get_child(i));
			if (child) {
				Vector2 child_local_pos = to_local(child->get_global_position());
				length = child_local_pos.length();
				bone_angle = child_local_pos.normalized().angle();
				calculated = true;
				break;
			}
		}
	}
	if (calculated) {
		return; // Finished!
	} else {
		WARN_PRINT("No Bone2D children of node " + get_name() + ". Cannot calculate bone length or angle reliably.\nUsing transform rotation for bone angle");
		bone_angle = get_transform().get_rotation();
		return;
	}
}

void Bone2D::set_autocalculate_length_and_angle(bool p_autocalculate) {
	autocalculate_length_and_angle = p_autocalculate;
	if (autocalculate_length_and_angle) {
		calculate_length_and_rotation();
	}
	notify_property_list_changed();
}

bool Bone2D::get_autocalculate_length_and_angle() const {
	return autocalculate_length_and_angle;
}

void Bone2D::set_length(real_t p_length) {
	length = p_length;

#ifdef TOOLS_ENABLED
	update();
#endif // TOOLS_ENABLED
}

real_t Bone2D::get_length() const {
	return length;
}

void Bone2D::set_bone_angle(real_t p_angle) {
	bone_angle = p_angle;

#ifdef TOOLS_ENABLED
	update();
#endif // TOOLS_ENABLED
}

real_t Bone2D::get_bone_angle() const {
	return bone_angle;
}

Bone2D::Bone2D() {
	skeleton = nullptr;
	parent_bone = nullptr;
	skeleton_index = -1;
	length = 16;
	bone_angle = 0;
	autocalculate_length_and_angle = true;
	set_notify_local_transform(true);
	//this is a clever hack so the bone knows no rest has been set yet, allowing to show an error.
	for (int i = 0; i < 3; i++) {
		rest[i] = Vector2(0, 0);
	}
	copy_transform_to_cache = true;
}

Bone2D::~Bone2D() {
#ifdef TOOLS_ENABLED
	if (!editor_gizmo_rid.is_null()) {
		RenderingServer::get_singleton()->free(editor_gizmo_rid);
	}
#endif // TOOLS_ENABLED
}

//////////////////////////////////////

bool Skeleton2D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("modification_stack")) {
		set_modification_stack(p_value);
		return true;
	}
	return true;
}

bool Skeleton2D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("modification_stack")) {
		r_ret = get_modification_stack();
		return true;
	}
	return true;
}

void Skeleton2D::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(
			PropertyInfo(Variant::OBJECT, "modification_stack",
					PROPERTY_HINT_RESOURCE_TYPE,
					"SkeletonModificationStack2D",
					PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DEFERRED_SET_RESOURCE | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE));
}

void Skeleton2D::_make_bone_setup_dirty() {
	if (bone_setup_dirty) {
		return;
	}
	bone_setup_dirty = true;
	if (is_inside_tree()) {
		call_deferred(SNAME("_update_bone_setup"));
	}
}

void Skeleton2D::_update_bone_setup() {
	if (!bone_setup_dirty) {
		return;
	}

	bone_setup_dirty = false;
	RS::get_singleton()->skeleton_allocate_data(skeleton, bones.size(), true);

	bones.sort(); //sorting so that they are always in the same order/index

	for (int i = 0; i < bones.size(); i++) {
		bones.write[i].rest_inverse = bones[i].bone->get_skeleton_rest().affine_inverse(); //bind pose
		bones.write[i].bone->skeleton_index = i;
		Bone2D *parent_bone = Object::cast_to<Bone2D>(bones[i].bone->get_parent());
		if (parent_bone) {
			bones.write[i].parent_index = parent_bone->skeleton_index;
		} else {
			bones.write[i].parent_index = -1;
		}

		bones.write[i].local_pose_override = bones[i].bone->get_skeleton_rest();
	}

	transform_dirty = true;
	_update_transform();
	emit_signal(SNAME("bone_setup_changed"));
}

void Skeleton2D::_make_transform_dirty() {
	if (transform_dirty) {
		return;
	}
	transform_dirty = true;
	if (is_inside_tree()) {
		call_deferred(SNAME("_update_transform"));
	}
}

void Skeleton2D::_update_transform() {
	if (bone_setup_dirty) {
		_update_bone_setup();
		return; //above will update transform anyway
	}
	if (!transform_dirty) {
		return;
	}

	transform_dirty = false;

	for (int i = 0; i < bones.size(); i++) {
		ERR_CONTINUE(bones[i].parent_index >= i);
		if (bones[i].parent_index >= 0) {
			bones.write[i].accum_transform = bones[bones[i].parent_index].accum_transform * bones[i].bone->get_transform();
		} else {
			bones.write[i].accum_transform = bones[i].bone->get_transform();
		}
	}

	for (int i = 0; i < bones.size(); i++) {
		Transform2D final_xform = bones[i].accum_transform * bones[i].rest_inverse;
		RS::get_singleton()->skeleton_bone_set_transform_2d(skeleton, i, final_xform);
	}
}

int Skeleton2D::get_bone_count() const {
	ERR_FAIL_COND_V(!is_inside_tree(), 0);

	if (bone_setup_dirty) {
		const_cast<Skeleton2D *>(this)->_update_bone_setup();
	}

	return bones.size();
}

Bone2D *Skeleton2D::get_bone(int p_idx) {
	ERR_FAIL_COND_V(!is_inside_tree(), nullptr);
	ERR_FAIL_INDEX_V(p_idx, bones.size(), nullptr);

	return bones[p_idx].bone;
}

void Skeleton2D::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		if (bone_setup_dirty) {
			_update_bone_setup();
		}
		if (transform_dirty) {
			_update_transform();
		}
		request_ready();
	}

	if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {
		RS::get_singleton()->skeleton_set_base_transform_2d(skeleton, get_global_transform());
	} else if (p_what == NOTIFICATION_INTERNAL_PROCESS) {
		if (modification_stack.is_valid()) {
			execute_modifications(get_process_delta_time(), SkeletonModificationStack2D::EXECUTION_MODE::execution_mode_process);
		}
	} else if (p_what == NOTIFICATION_INTERNAL_PHYSICS_PROCESS) {
		if (modification_stack.is_valid()) {
			execute_modifications(get_physics_process_delta_time(), SkeletonModificationStack2D::EXECUTION_MODE::execution_mode_physics_process);
		}
	}
#ifdef TOOLS_ENABLED
	else if (p_what == NOTIFICATION_DRAW) {
		if (Engine::get_singleton()->is_editor_hint()) {
			if (modification_stack.is_valid()) {
				modification_stack->draw_editor_gizmos();
			}
		}
	}
#endif // TOOLS_ENABLED
}

RID Skeleton2D::get_skeleton() const {
	return skeleton;
}

void Skeleton2D::set_bone_local_pose_override(int p_bone_idx, Transform2D p_override, real_t p_amount, bool p_persistent) {
	ERR_FAIL_INDEX_MSG(p_bone_idx, bones.size(), "Bone index is out of range!");
	bones.write[p_bone_idx].local_pose_override = p_override;
	bones.write[p_bone_idx].local_pose_override_amount = p_amount;
	bones.write[p_bone_idx].local_pose_override_persistent = p_persistent;
}

Transform2D Skeleton2D::get_bone_local_pose_override(int p_bone_idx) {
	ERR_FAIL_INDEX_V_MSG(p_bone_idx, bones.size(), Transform2D(), "Bone index is out of range!");
	return bones[p_bone_idx].local_pose_override;
}

void Skeleton2D::set_modification_stack(Ref<SkeletonModificationStack2D> p_stack) {
	if (modification_stack.is_valid()) {
		modification_stack->is_setup = false;
		modification_stack->set_skeleton(nullptr);

		set_process_internal(false);
		set_physics_process_internal(false);
	}
	modification_stack = p_stack;
	if (modification_stack.is_valid()) {
		modification_stack->set_skeleton(this);
		modification_stack->setup();

		set_process_internal(true);
		set_physics_process_internal(true);

#ifdef TOOLS_ENABLED
		modification_stack->set_editor_gizmos_dirty(true);
#endif // TOOLS_ENABLED
	}
}

Ref<SkeletonModificationStack2D> Skeleton2D::get_modification_stack() const {
	return modification_stack;
}

void Skeleton2D::execute_modifications(real_t p_delta, int p_execution_mode) {
	if (!modification_stack.is_valid()) {
		return;
	}

	// Do not cache the transform changes caused by the modifications!
	for (int i = 0; i < bones.size(); i++) {
		bones[i].bone->copy_transform_to_cache = false;
	}

	if (modification_stack->skeleton != this) {
		modification_stack->set_skeleton(this);
	}

	modification_stack->execute(p_delta, p_execution_mode);

	// Only apply the local pose override on _process. Otherwise, just calculate the local_pose_override and reset the transform.
	if (p_execution_mode == SkeletonModificationStack2D::EXECUTION_MODE::execution_mode_process) {
		for (int i = 0; i < bones.size(); i++) {
			if (bones[i].local_pose_override_amount > 0) {
				bones[i].bone->set_meta("_local_pose_override_enabled_", true);

				Transform2D final_trans = bones[i].bone->cache_transform;
				final_trans = final_trans.interpolate_with(bones[i].local_pose_override, bones[i].local_pose_override_amount);
				bones[i].bone->set_transform(final_trans);
				bones[i].bone->propagate_call("force_update_transform");

				if (bones[i].local_pose_override_persistent) {
					bones.write[i].local_pose_override_amount = 0.0;
				}
			} else {
				// TODO: see if there is a way to undo the override without having to resort to setting every bone's transform.
				bones[i].bone->remove_meta("_local_pose_override_enabled_");
				bones[i].bone->set_transform(bones[i].bone->cache_transform);
			}
		}
	}

	// Cache any future transform changes
	for (int i = 0; i < bones.size(); i++) {
		bones[i].bone->copy_transform_to_cache = true;
	}

#ifdef TOOLS_ENABLED
	modification_stack->set_editor_gizmos_dirty(true);
#endif // TOOLS_ENABLED
}

void Skeleton2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_bone_setup"), &Skeleton2D::_update_bone_setup);
	ClassDB::bind_method(D_METHOD("_update_transform"), &Skeleton2D::_update_transform);

	ClassDB::bind_method(D_METHOD("get_bone_count"), &Skeleton2D::get_bone_count);
	ClassDB::bind_method(D_METHOD("get_bone", "idx"), &Skeleton2D::get_bone);

	ClassDB::bind_method(D_METHOD("get_skeleton"), &Skeleton2D::get_skeleton);

	ClassDB::bind_method(D_METHOD("set_modification_stack", "modification_stack"), &Skeleton2D::set_modification_stack);
	ClassDB::bind_method(D_METHOD("get_modification_stack"), &Skeleton2D::get_modification_stack);
	ClassDB::bind_method(D_METHOD("execute_modifications", "delta", "execution_mode"), &Skeleton2D::execute_modifications);

	ClassDB::bind_method(D_METHOD("set_bone_local_pose_override", "bone_idx", "override_pose", "strength", "persistent"), &Skeleton2D::set_bone_local_pose_override);
	ClassDB::bind_method(D_METHOD("get_bone_local_pose_override", "bone_idx"), &Skeleton2D::get_bone_local_pose_override);

	ADD_SIGNAL(MethodInfo("bone_setup_changed"));
}

Skeleton2D::Skeleton2D() {
	skeleton = RS::get_singleton()->skeleton_create();
	set_notify_transform(true);
}

Skeleton2D::~Skeleton2D() {
	RS::get_singleton()->free(skeleton);
}
