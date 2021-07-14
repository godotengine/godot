/*************************************************************************/
/*  skeleton_3d_editor_plugin.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "skeleton_3d_editor_plugin.h"

#include "core/io/resource_saver.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_properties.h"
#include "editor/editor_scale.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "node_3d_editor_plugin.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/3d/physics_joint_3d.h"
#include "scene/3d/visual_instance_3d.h"
#include "scene/resources/capsule_shape_3d.h"
#include "scene/resources/sphere_shape_3d.h"

#define DISTANCE_DEFAULT 4

#define GIZMO_ARROW_SIZE 0.35
#define GIZMO_RING_HALF_WIDTH 0.1
#define GIZMO_SCALE_DEFAULT 0.15
#define GIZMO_PLANE_SIZE 0.2
#define GIZMO_PLANE_DST 0.3
#define GIZMO_CIRCLE_SIZE 1.1
#define GIZMO_SCALE_OFFSET (GIZMO_CIRCLE_SIZE + 0.3)
#define GIZMO_ARROW_OFFSET (GIZMO_CIRCLE_SIZE + 0.3)

#define ZOOM_MIN_DISTANCE 0.001
#define ZOOM_MULTIPLIER 1.08
#define ZOOM_INDICATOR_DELAY_S 1.5

#define FREELOOK_MIN_SPEED 0.01
#define FREELOOK_SPEED_MULTIPLIER 1.08

#define MIN_Z 0.01
#define MAX_Z 1000000.0

#define MIN_FOV 0.01
#define MAX_FOV 179

void BoneTransformEditor::create_editors() {
	const Color section_color = get_theme_color("prop_subsection", "Editor");

	section = memnew(EditorInspectorSection);
	section->setup("trf_properties", label, this, section_color, true);
	add_child(section);

	key_button = memnew(Button);
	key_button->set_text(TTR("Key Transform"));
	key_button->set_visible(keyable);
	key_button->set_icon(get_theme_icon("Key", "EditorIcons"));
	key_button->set_flat(true);
	section->get_vbox()->add_child(key_button);

	enabled_checkbox = memnew(CheckBox(TTR("Pose Enabled")));
	enabled_checkbox->set_flat(true);
	enabled_checkbox->set_visible(toggle_enabled);
	section->get_vbox()->add_child(enabled_checkbox);

	// Translation property
	translation_property = memnew(EditorPropertyVector3());
	translation_property->setup(-10000, 10000, 0.001f, true);
	translation_property->set_label("Translation");
	translation_property->set_use_folding(true);
	translation_property->set_read_only(false);
	translation_property->connect("property_changed", callable_mp(this, &BoneTransformEditor::_value_changed_vector3));
	section->get_vbox()->add_child(translation_property);

	// Rotation property
	rotation_property = memnew(EditorPropertyVector3());
	rotation_property->setup(-10000, 10000, 0.001f, true);
	rotation_property->set_label("Rotation Degrees");
	rotation_property->set_use_folding(true);
	rotation_property->set_read_only(false);
	rotation_property->connect("property_changed", callable_mp(this, &BoneTransformEditor::_value_changed_vector3));
	section->get_vbox()->add_child(rotation_property);

	// Scale property
	scale_property = memnew(EditorPropertyVector3());
	scale_property->setup(-10000, 10000, 0.001f, true);
	scale_property->set_label("Scale");
	scale_property->set_use_folding(true);
	scale_property->set_read_only(false);
	scale_property->connect("property_changed", callable_mp(this, &BoneTransformEditor::_value_changed_vector3));
	section->get_vbox()->add_child(scale_property);

	// Transform/Matrix section
	transform_section = memnew(EditorInspectorSection);
	transform_section->setup("trf_properties_transform", "Matrix", this, section_color, true);
	section->get_vbox()->add_child(transform_section);

	// Transform/Matrix property
	transform_property = memnew(EditorPropertyTransform3D());
	transform_property->setup(-10000, 10000, 0.001f, true);
	transform_property->set_label("Transform");
	transform_property->set_use_folding(true);
	transform_property->set_read_only(false);
	transform_property->connect("property_changed", callable_mp(this, &BoneTransformEditor::_value_changed_transform));
	transform_section->get_vbox()->add_child(transform_property);
}

void BoneTransformEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editors();
			key_button->connect("pressed", callable_mp(this, &BoneTransformEditor::_key_button_pressed));
			enabled_checkbox->connect("toggled", callable_mp(this, &BoneTransformEditor::_checkbox_toggled));
			[[fallthrough]];
		}
		case NOTIFICATION_SORT_CHILDREN: {
			const Ref<Font> font = get_theme_font("font", "Tree");
			int font_size = get_theme_font_size("font_size", "Tree");

			Point2 buffer;
			buffer.x += get_theme_constant("inspector_margin", "Editor");
			buffer.y += font->get_height(font_size);
			buffer.y += get_theme_constant("vseparation", "Tree");

			const float vector_height = translation_property->get_size().y;
			const float transform_height = transform_property->get_size().y;
			const float button_height = key_button->get_size().y;

			const float width = get_size().x - get_theme_constant("inspector_margin", "Editor");
			Vector<Rect2> input_rects;
			if (keyable && section->get_vbox()->is_visible()) {
				input_rects.push_back(Rect2(key_button->get_position() + buffer, Size2(width, button_height)));
			} else {
				input_rects.push_back(Rect2(0, 0, 0, 0));
			}

			if (section->get_vbox()->is_visible()) {
				input_rects.push_back(Rect2(translation_property->get_position() + buffer, Size2(width, vector_height)));
				input_rects.push_back(Rect2(rotation_property->get_position() + buffer, Size2(width, vector_height)));
				input_rects.push_back(Rect2(scale_property->get_position() + buffer, Size2(width, vector_height)));
				input_rects.push_back(Rect2(transform_property->get_position() + buffer, Size2(width, transform_height)));
			} else {
				const int32_t start = input_rects.size();
				const int32_t empty_input_rect_elements = 4;
				const int32_t end = start + empty_input_rect_elements;
				for (int i = start; i < end; ++i) {
					input_rects.push_back(Rect2(0, 0, 0, 0));
				}
			}

			for (int32_t i = 0; i < input_rects.size(); i++) {
				background_rects[i] = input_rects[i];
			}

			update();
			break;
		}
		case NOTIFICATION_DRAW: {
			const Color dark_color = get_theme_color("dark_color_2", "Editor");

			for (int i = 0; i < 5; ++i) {
				draw_rect(background_rects[i], dark_color);
			}

			break;
		}
	}
}

void BoneTransformEditor::_value_changed(const double p_value) {
	if (updating) {
		return;
	}

	Transform3D tform = compute_transform_from_vector3s();
	_change_transform(tform);
}

void BoneTransformEditor::_value_changed_vector3(const String p_property_name, const Vector3 p_vector, const StringName p_edited_property_name, const bool p_boolean) {
	if (updating) {
		return;
	}
	Transform3D tform = compute_transform_from_vector3s();
	_change_transform(tform);
}

Transform3D BoneTransformEditor::compute_transform_from_vector3s() const {
	// Convert rotation from degrees to radians.
	Vector3 prop_rotation = rotation_property->get_vector();
	prop_rotation.x = Math::deg2rad(prop_rotation.x);
	prop_rotation.y = Math::deg2rad(prop_rotation.y);
	prop_rotation.z = Math::deg2rad(prop_rotation.z);

	return Transform3D(
			Basis(prop_rotation, scale_property->get_vector()),
			translation_property->get_vector());
}

void BoneTransformEditor::_value_changed_transform(const String p_property_name, const Transform3D p_transform, const StringName p_edited_property_name, const bool p_boolean) {
	if (updating) {
		return;
	}
	_change_transform(p_transform);
}

void BoneTransformEditor::_change_transform(Transform3D p_new_transform) {
	if (property.get_slicec('/', 0) == "bones" && property.get_slicec('/', 2) == "custom_pose") {
		undo_redo->create_action(TTR("Set Custom Bone Pose Transform"), UndoRedo::MERGE_ENDS);
		undo_redo->add_undo_method(skeleton, "set_bone_custom_pose", property.get_slicec('/', 1).to_int(), skeleton->get_bone_custom_pose(property.get_slicec('/', 1).to_int()));
		undo_redo->add_do_method(skeleton, "set_bone_custom_pose", property.get_slicec('/', 1).to_int(), p_new_transform);
		undo_redo->commit_action();
	} else if (property.get_slicec('/', 0) == "bones") {
		undo_redo->create_action(TTR("Set Bone Transform"), UndoRedo::MERGE_ENDS);
		undo_redo->add_undo_property(skeleton, property, skeleton->get(property));
		undo_redo->add_do_property(skeleton, property, p_new_transform);
		undo_redo->commit_action();
	}
}

void BoneTransformEditor::update_enabled_checkbox() {
	if (enabled_checkbox) {
		const String path = "bones/" + property.get_slicec('/', 1) + "/enabled";
		const bool is_enabled = skeleton->get(path);
		enabled_checkbox->set_pressed(is_enabled);
	}
}

void BoneTransformEditor::_update_properties() {
	if (updating) {
		return;
	}

	if (skeleton == nullptr) {
		return;
	}

	updating = true;

	Transform3D tform = skeleton->get(property);
	_update_transform_properties(tform);
}

void BoneTransformEditor::_update_custom_pose_properties() {
	if (updating) {
		return;
	}

	if (skeleton == nullptr) {
		return;
	}

	updating = true;

	Transform3D tform = skeleton->get_bone_custom_pose(property.to_int());
	_update_transform_properties(tform);
}

void BoneTransformEditor::_update_transform_properties(Transform3D tform) {
	Basis rotation_basis = tform.get_basis();
	Vector3 rotation_radians = rotation_basis.get_rotation_euler();
	Vector3 rotation_degrees = Vector3(Math::rad2deg(rotation_radians.x), Math::rad2deg(rotation_radians.y), Math::rad2deg(rotation_radians.z));
	Vector3 translation = tform.get_origin();
	Vector3 scale = tform.basis.get_scale();

	translation_property->update_using_vector(translation);
	rotation_property->update_using_vector(rotation_degrees);
	scale_property->update_using_vector(scale);
	transform_property->update_using_transform(tform);

	update_enabled_checkbox();
	updating = false;
}

BoneTransformEditor::BoneTransformEditor(Skeleton3D *p_skeleton) :
		skeleton(p_skeleton) {
	undo_redo = EditorNode::get_undo_redo();
}

void BoneTransformEditor::set_target(const String &p_prop) {
	property = p_prop;
}

void BoneTransformEditor::set_keyable(const bool p_keyable) {
	keyable = p_keyable;
	if (key_button) {
		key_button->set_visible(p_keyable);
	}
}

void BoneTransformEditor::set_toggle_enabled(const bool p_enabled) {
	toggle_enabled = p_enabled;
	if (enabled_checkbox) {
		enabled_checkbox->set_visible(p_enabled);
	}
}

void BoneTransformEditor::_key_button_pressed() {
	if (skeleton == nullptr) {
		return;
	}

	const BoneId bone_id = property.get_slicec('/', 1).to_int();
	const String name = skeleton->get_bone_name(bone_id);

	if (name.is_empty()) {
		return;
	}

	// Need to normalize the basis before you key it
	Transform3D tform = compute_transform_from_vector3s();
	tform.orthonormalize();
	AnimationPlayerEditor::singleton->get_track_editor()->insert_transform_key(skeleton, name, tform);
}

void BoneTransformEditor::_checkbox_toggled(const bool p_toggled) {
	if (enabled_checkbox) {
		const String path = "bones/" + property.get_slicec('/', 1) + "/enabled";
		skeleton->set(path, p_toggled);
	}
}

void BoneTransformEditor::set_read_only(const bool p_read_only) {
	translation_property->set_read_only(p_read_only);
	rotation_property->set_read_only(p_read_only);
	scale_property->set_read_only(p_read_only);
	transform_property->set_read_only(p_read_only);
}

void Skeleton3DEditor::set_keyable(const bool p_keyable) {
	keyable = p_keyable;
	options->get_popup()->set_item_disabled(MENU_OPTION_INSERT_KEYS, !p_keyable);
	options->get_popup()->set_item_disabled(MENU_OPTION_INSERT_KEYS_EXISTED, !p_keyable);
};

void Skeleton3DEditor::_on_click_option(int p_option) {
	if (!skeleton) {
		return;
	}

	switch (p_option) {
		case MENU_OPTION_CREATE_PHYSICAL_SKELETON: {
			create_physical_skeleton();
			break;
		}
		case MENU_OPTION_INIT_POSE: {
			init_pose();
			break;
		}
		case MENU_OPTION_INSERT_KEYS: {
			insert_keys(true);
			break;
		}
		case MENU_OPTION_INSERT_KEYS_EXISTED: {
			insert_keys(false);
			break;
		}
		case MENU_OPTION_POSE_TO_REST: {
			pose_to_rest();
			break;
		}
	}
}

void Skeleton3DEditor::init_pose() {
	const int bone_len = skeleton->get_bone_count();
	if (!bone_len) {
		return;
	}
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Bone Transform"), UndoRedo::MERGE_ENDS);
	for (int i = 0; i < bone_len; i++) {
		ur->add_do_method(skeleton, "set_bone_pose", i, Transform3D());
		ur->add_undo_method(skeleton, "set_bone_pose", i, skeleton->get_bone_pose(i));
	}
	ur->commit_action();
}

void Skeleton3DEditor::insert_keys(bool p_all_bones) {
	if (skeleton == nullptr) {
		return;
	}

	int bone_len = skeleton->get_bone_count();
	Node *root = EditorNode::get_singleton()->get_tree()->get_root();
	String path = root->get_path_to(skeleton);

	for (int i = 0; i < bone_len; i++) {
		const String name = skeleton->get_bone_name(i);

		if (name.is_empty()) {
			continue;
		}

		if (!p_all_bones && !AnimationPlayerEditor::singleton->get_track_editor()->has_transform_key(skeleton, name)) {
			continue;
		}

		// Need to normalize the basis before you key it
		Transform3D tform = skeleton->get_bone_pose(i);
		tform.orthonormalize();
		AnimationPlayerEditor::singleton->get_track_editor()->insert_transform_key(skeleton, name, tform);
	}
}

void Skeleton3DEditor::pose_to_rest() {
	const int bone_len = skeleton->get_bone_count();
	if (!bone_len) {
		return;
	}
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Bone Transform"), UndoRedo::MERGE_ENDS);
	for (int i = 0; i < bone_len; i++) {
		ur->add_do_method(skeleton, "set_bone_pose", i, Transform3D());
		ur->add_undo_method(skeleton, "set_bone_pose", i, skeleton->get_bone_pose(i));
		ur->add_do_method(skeleton, "set_bone_custom_pose", i, Transform3D());
		ur->add_undo_method(skeleton, "set_bone_custom_pose", i, skeleton->get_bone_custom_pose(i));
		ur->add_do_method(skeleton, "set_bone_rest", i, skeleton->get_bone_rest(i) * skeleton->get_bone_custom_pose(i) * skeleton->get_bone_pose(i));
		ur->add_undo_method(skeleton, "set_bone_rest", i, skeleton->get_bone_rest(i));
	}
	ur->commit_action();
}

void Skeleton3DEditor::create_physical_skeleton() {
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ERR_FAIL_COND(!get_tree());
	Node *owner = skeleton == get_tree()->get_edited_scene_root() ? skeleton : skeleton->get_owner();

	const int bc = skeleton->get_bone_count();

	if (!bc) {
		return;
	}

	Vector<BoneInfo> bones_infos;
	bones_infos.resize(bc);

	for (int bone_id = 0; bc > bone_id; ++bone_id) {
		const int parent = skeleton->get_bone_parent(bone_id);

		if (parent < 0) {
			bones_infos.write[bone_id].relative_rest = skeleton->get_bone_rest(bone_id);

		} else {
			const int parent_parent = skeleton->get_bone_parent(parent);

			bones_infos.write[bone_id].relative_rest = bones_infos[parent].relative_rest * skeleton->get_bone_rest(bone_id);

			/// create physical bone on parent
			if (!bones_infos[parent].physical_bone) {
				bones_infos.write[parent].physical_bone = create_physical_bone(parent, bone_id, bones_infos);

				ur->create_action(TTR("Create physical bones"));
				ur->add_do_method(skeleton, "add_child", bones_infos[parent].physical_bone);
				ur->add_do_reference(bones_infos[parent].physical_bone);
				ur->add_undo_method(skeleton, "remove_child", bones_infos[parent].physical_bone);
				ur->commit_action();

				bones_infos[parent].physical_bone->set_bone_name(skeleton->get_bone_name(parent));
				bones_infos[parent].physical_bone->set_owner(owner);
				bones_infos[parent].physical_bone->get_child(0)->set_owner(owner); // set shape owner

				/// Create joint between parent of parent
				if (-1 != parent_parent) {
					bones_infos[parent].physical_bone->set_joint_type(PhysicalBone3D::JOINT_TYPE_PIN);
				}
			}
		}
	}
}

PhysicalBone3D *Skeleton3DEditor::create_physical_bone(int bone_id, int bone_child_id, const Vector<BoneInfo> &bones_infos) {
	const Transform3D child_rest = skeleton->get_bone_rest(bone_child_id);

	const real_t half_height(child_rest.origin.length() * 0.5);
	const real_t radius(half_height * 0.2);

	CapsuleShape3D *bone_shape_capsule = memnew(CapsuleShape3D);
	bone_shape_capsule->set_height((half_height - radius) * 2);
	bone_shape_capsule->set_radius(radius);

	CollisionShape3D *bone_shape = memnew(CollisionShape3D);
	bone_shape->set_shape(bone_shape_capsule);

	Transform3D capsule_transform;
	capsule_transform.basis = Basis(Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(0, -1, 0));
	bone_shape->set_transform(capsule_transform);

	Transform3D body_transform;
	body_transform.set_look_at(Vector3(0, 0, 0), child_rest.origin);
	body_transform.origin = body_transform.basis.xform(Vector3(0, 0, -half_height));

	Transform3D joint_transform;
	joint_transform.origin = Vector3(0, 0, half_height);

	PhysicalBone3D *physical_bone = memnew(PhysicalBone3D);
	physical_bone->add_child(bone_shape);
	physical_bone->set_name("Physical Bone " + skeleton->get_bone_name(bone_id));
	physical_bone->set_body_offset(body_transform);
	physical_bone->set_joint_offset(joint_transform);
	return physical_bone;
}

Variant Skeleton3DEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	TreeItem *selected = joint_tree->get_selected();

	if (!selected) {
		return Variant();
	}

	Ref<Texture> icon = selected->get_icon(0);

	VBoxContainer *vb = memnew(VBoxContainer);
	HBoxContainer *hb = memnew(HBoxContainer);
	TextureRect *tf = memnew(TextureRect);
	tf->set_texture(icon);
	tf->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	hb->add_child(tf);
	Label *label = memnew(Label(selected->get_text(0)));
	hb->add_child(label);
	vb->add_child(hb);
	hb->set_modulate(Color(1, 1, 1, 1));

	set_drag_preview(vb);
	Dictionary drag_data;
	drag_data["type"] = "nodes";
	drag_data["node"] = selected;

	return drag_data;
}

bool Skeleton3DEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	TreeItem *target = joint_tree->get_item_at_position(p_point);
	if (!target) {
		return false;
	}

	const String path = target->get_metadata(0);
	if (!path.begins_with("bones/")) {
		return false;
	}

	TreeItem *selected = Object::cast_to<TreeItem>(Dictionary(p_data)["node"]);
	if (target == selected) {
		return false;
	}

	const String path2 = target->get_metadata(0);
	if (!path2.begins_with("bones/")) {
		return false;
	}

	return true;
}

void Skeleton3DEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from)) {
		return;
	}

	TreeItem *target = joint_tree->get_item_at_position(p_point);
	TreeItem *selected = Object::cast_to<TreeItem>(Dictionary(p_data)["node"]);

	const BoneId target_boneidx = String(target->get_metadata(0)).get_slicec('/', 1).to_int();
	const BoneId selected_boneidx = String(selected->get_metadata(0)).get_slicec('/', 1).to_int();

	move_skeleton_bone(skeleton->get_path(), selected_boneidx, target_boneidx);
}

void Skeleton3DEditor::move_skeleton_bone(NodePath p_skeleton_path, int32_t p_selected_boneidx, int32_t p_target_boneidx) {
	Node *node = get_node_or_null(p_skeleton_path);
	Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(node);
	ERR_FAIL_NULL(skeleton);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Bone Parentage"));
	// If the target is a child of ourselves, we move only *us* and not our children
	if (skeleton->is_bone_parent_of(p_target_boneidx, p_selected_boneidx)) {
		const BoneId parent_idx = skeleton->get_bone_parent(p_selected_boneidx);
		const int bone_count = skeleton->get_bone_count();
		for (BoneId i = 0; i < bone_count; ++i) {
			if (skeleton->get_bone_parent(i) == p_selected_boneidx) {
				ur->add_undo_method(skeleton, "set_bone_parent", i, skeleton->get_bone_parent(i));
				ur->add_do_method(skeleton, "set_bone_parent", i, parent_idx);
				skeleton->set_bone_parent(i, parent_idx);
			}
		}
	}
	ur->add_undo_method(skeleton, "set_bone_parent", p_selected_boneidx, skeleton->get_bone_parent(p_selected_boneidx));
	ur->add_do_method(skeleton, "set_bone_parent", p_selected_boneidx, p_target_boneidx);
	skeleton->set_bone_parent(p_selected_boneidx, p_target_boneidx);

	update_joint_tree();
	ur->commit_action();
}

void Skeleton3DEditor::_update_sub_gizmo() {
	Node3DEditor::get_singleton()->clear_externals();
	if (skeleton->get_selected_bone() >= 0) {
		Node3DEditor::get_singleton()->append_to_externals(skeleton->get_global_transform() * skeleton->get_bone_global_pose(skeleton->get_selected_bone()));
	}
	Node3DEditor::get_singleton()->update_transform_gizmo();
};

void Skeleton3DEditor::_joint_tree_selection_changed() {
	TreeItem *selected = joint_tree->get_selected();
	const String path = selected->get_metadata(0);

	if (path.begins_with("bones/")) {
		const int b_idx = path.get_slicec('/', 1).to_int();
		const String bone_path = "bones/" + itos(b_idx) + "/";

		pose_editor->set_target(bone_path + "pose");
		rest_editor->set_target(bone_path + "rest");
		custom_pose_editor->set_target(bone_path + "custom_pose");

		_update_properties();

		pose_editor->set_visible(true);
		rest_editor->set_visible(true);
		custom_pose_editor->set_visible(true);

		skeleton->set_selected_bone(b_idx);
	}
}

void Skeleton3DEditor::_joint_tree_rmb_select(const Vector2 &p_pos) {
	skeleton->set_selected_bone(-1);
	_update_sub_gizmo();
}

void Skeleton3DEditor::_update_properties() {
	if (rest_editor) {
		rest_editor->_update_properties();
	}
	if (pose_editor) {
		pose_editor->_update_properties();
	}
	if (custom_pose_editor) {
		custom_pose_editor->_update_custom_pose_properties();
	}
	_update_sub_gizmo();
}

void Skeleton3DEditor::update_joint_tree() {
	joint_tree->clear();

	if (skeleton == nullptr) {
		return;
	}

	TreeItem *root = joint_tree->create_item();

	Map<int, TreeItem *> items;

	items.insert(-1, root);

	const Vector<int> &joint_porder = skeleton->get_bone_process_orders();
	Ref<Texture> bone_icon = get_theme_icon("BoneAttachment3D", "EditorIcons");

	for (int i = 0; i < joint_porder.size(); ++i) {
		const int b_idx = joint_porder[i];

		const int p_idx = skeleton->get_bone_parent(b_idx);
		TreeItem *p_item = items.find(p_idx)->get();

		TreeItem *joint_item = joint_tree->create_item(p_item);
		items.insert(b_idx, joint_item);

		joint_item->set_text(0, skeleton->get_bone_name(b_idx));
		joint_item->set_icon(0, bone_icon);
		joint_item->set_selectable(0, true);
		joint_item->set_metadata(0, "bones/" + itos(b_idx));
	}
}

void Skeleton3DEditor::update_editors() {
}

void Skeleton3DEditor::create_editors() {
	set_h_size_flags(SIZE_EXPAND_FILL);
	add_theme_constant_override("separation", 0);

	set_focus_mode(FOCUS_ALL);

	// Create Top Menu Bar
	separators[0] = memnew(VSeparator);
	separators[1] = memnew(VSeparator);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(separators[0]);

	// Create Top Menu Bar
	options = memnew(MenuButton);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(options);

	options->set_text(TTR("Skeleton3D"));
	options->set_icon(EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Skeleton3D", "EditorIcons"));

	options->get_popup()->add_item(TTR("Init pose"), MENU_OPTION_INIT_POSE);
	options->get_popup()->add_item(TTR("Insert key of all bone poses"), MENU_OPTION_INSERT_KEYS);
	options->get_popup()->add_item(TTR("Insert key of bone poses already exist track"), MENU_OPTION_INSERT_KEYS_EXISTED);
	options->get_popup()->add_item(TTR("Apply current pose to rest"), MENU_OPTION_POSE_TO_REST);
	options->get_popup()->add_item(TTR("Create physical skeleton"), MENU_OPTION_CREATE_PHYSICAL_SKELETON);

	options->get_popup()->connect("id_pressed", callable_mp(this, &Skeleton3DEditor::_on_click_option));

	Vector<Variant> button_binds;
	button_binds.resize(1);

	tool_button[TOOL_MODE_BONE_SELECT] = memnew(Button);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(tool_button[TOOL_MODE_BONE_SELECT]);
	tool_button[TOOL_MODE_BONE_SELECT]->set_tooltip(TTR("Transform Bone Mode"));
	tool_button[TOOL_MODE_BONE_SELECT]->set_toggle_mode(true);
	tool_button[TOOL_MODE_BONE_SELECT]->set_flat(true);
	button_binds.write[0] = MENU_TOOL_BONE_SELECT;
	tool_button[TOOL_MODE_BONE_SELECT]->connect("pressed", callable_mp(this, &Skeleton3DEditor::_menu_tool_item_pressed), button_binds);

	tool_button[TOOL_MODE_BONE_MOVE] = memnew(Button);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(tool_button[TOOL_MODE_BONE_MOVE]);
	tool_button[TOOL_MODE_BONE_MOVE]->set_tooltip(TTR("Move Bone Mode"));
	tool_button[TOOL_MODE_BONE_MOVE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_BONE_MOVE]->set_flat(true);
	button_binds.write[0] = MENU_TOOL_BONE_MOVE;
	tool_button[TOOL_MODE_BONE_MOVE]->connect("pressed", callable_mp(this, &Skeleton3DEditor::_menu_tool_item_pressed), button_binds);

	tool_button[TOOL_MODE_BONE_ROTATE] = memnew(Button);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(tool_button[TOOL_MODE_BONE_ROTATE]);
	tool_button[TOOL_MODE_BONE_ROTATE]->set_tooltip(TTR("Rotate Bone Mode"));
	tool_button[TOOL_MODE_BONE_ROTATE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_BONE_ROTATE]->set_flat(true);
	button_binds.write[0] = MENU_TOOL_BONE_ROTATE;
	tool_button[TOOL_MODE_BONE_ROTATE]->connect("pressed", callable_mp(this, &Skeleton3DEditor::_menu_tool_item_pressed), button_binds);

	tool_button[TOOL_MODE_BONE_SCALE] = memnew(Button);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(tool_button[TOOL_MODE_BONE_SCALE]);
	tool_button[TOOL_MODE_BONE_SCALE]->set_tooltip(TTR("Scale Bone Mode"));
	tool_button[TOOL_MODE_BONE_SCALE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_BONE_SCALE]->set_flat(true);
	button_binds.write[0] = MENU_TOOL_BONE_SCALE;
	tool_button[TOOL_MODE_BONE_SCALE]->connect("pressed", callable_mp(this, &Skeleton3DEditor::_menu_tool_item_pressed), button_binds);

	tool_button[TOOL_MODE_BONE_NONE] = memnew(Button);
	button_binds.write[0] = MENU_TOOL_BONE_NONE;
	tool_button[TOOL_MODE_BONE_NONE]->connect("pressed", callable_mp(this, &Skeleton3DEditor::_menu_tool_item_pressed), button_binds);
	Node3DEditor::get_singleton()->connect("change_tool_mode", callable_mp(this, &Skeleton3DEditor::_menu_tool_item_pressed), button_binds);

	tool_mode = TOOL_MODE_BONE_NONE;

	Node3DEditor::get_singleton()->add_control_to_menu_panel(separators[1]);

	rest_mode_button = memnew(Button);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(rest_mode_button);
	rest_mode_button->set_tooltip(TTR("Rest Mode\nNote: Bone poses are disabled during Rest Mode."));
	rest_mode_button->set_toggle_mode(true);
	rest_mode_button->set_flat(true);
	rest_mode_button->connect("toggled", callable_mp(this, &Skeleton3DEditor::rest_mode_toggled));

	rest_mode = false;

	set_keyable(AnimationPlayerEditor::singleton->get_track_editor()->has_keying());

	if (skeleton) {
		skeleton->add_child(pointsm);
		pointsm->set_skeleton_path(NodePath(""));
		skeleton->connect("pose_updated", callable_mp(this, &Skeleton3DEditor::_draw_handles));
		skeleton->set_selected_bone(-1);
	}

	const Color section_color = get_theme_color("prop_subsection", "Editor");

	EditorInspectorSection *bones_section = memnew(EditorInspectorSection);
	bones_section->setup("bones", "Bones", skeleton, section_color, true);
	add_child(bones_section);
	bones_section->unfold();

	ScrollContainer *s_con = memnew(ScrollContainer);
	s_con->set_h_size_flags(SIZE_EXPAND_FILL);
	s_con->set_custom_minimum_size(Size2(1, 350) * EDSCALE);
	bones_section->get_vbox()->add_child(s_con);

	joint_tree = memnew(Tree);
	joint_tree->set_columns(1);
	joint_tree->set_focus_mode(Control::FocusMode::FOCUS_NONE);
	joint_tree->set_select_mode(Tree::SELECT_SINGLE);
	joint_tree->set_hide_root(true);
	joint_tree->set_v_size_flags(SIZE_EXPAND_FILL);
	joint_tree->set_h_size_flags(SIZE_EXPAND_FILL);
	joint_tree->set_allow_rmb_select(true);
	joint_tree->set_drag_forwarding(this);
	s_con->add_child(joint_tree);

	pose_editor = memnew(BoneTransformEditor(skeleton));
	pose_editor->set_label(TTR("Bone Pose"));
	pose_editor->set_keyable(AnimationPlayerEditor::singleton->get_track_editor()->has_keying());
	// pose_editor->set_toggle_enabled(true);
	pose_editor->set_visible(false);
	add_child(pose_editor);

	rest_editor = memnew(BoneTransformEditor(skeleton));
	rest_editor->set_label(TTR("Bone Rest"));
	rest_editor->set_visible(false);
	add_child(rest_editor);

	custom_pose_editor = memnew(BoneTransformEditor(skeleton));
	custom_pose_editor->set_label(TTR("Bone Custom Pose"));
	custom_pose_editor->set_visible(false);
	add_child(custom_pose_editor);
}

void Skeleton3DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			tool_button[TOOL_MODE_BONE_SELECT]->set_icon(get_theme_icon("ToolBoneSelect", "EditorIcons"));
			tool_button[TOOL_MODE_BONE_MOVE]->set_icon(get_theme_icon("ToolBoneMove", "EditorIcons"));
			tool_button[TOOL_MODE_BONE_ROTATE]->set_icon(get_theme_icon("ToolBoneRotate", "EditorIcons"));
			tool_button[TOOL_MODE_BONE_SCALE]->set_icon(get_theme_icon("ToolBoneScale", "EditorIcons"));
			rest_mode_button->set_icon(get_theme_icon("ToolBoneRest", "EditorIcons"));
			break;
		}
		case NOTIFICATION_ENTER_TREE: {
			create_editors();
			update_joint_tree();
			update_editors();

			get_tree()->connect("node_removed", callable_mp(this, &Skeleton3DEditor::_node_removed), Vector<Variant>(), Object::CONNECT_ONESHOT);
			joint_tree->connect("item_selected", callable_mp(this, &Skeleton3DEditor::_joint_tree_selection_changed));
			joint_tree->connect("item_rmb_selected", callable_mp(this, &Skeleton3DEditor::_joint_tree_rmb_select));
#ifdef TOOLS_ENABLED
			skeleton->connect("pose_updated", callable_mp(this, &Skeleton3DEditor::_update_properties));
#endif // TOOLS_ENABLED

			break;
		}
	}
}

void Skeleton3DEditor::_node_removed(Node *p_node) {
	if (skeleton && p_node == skeleton) {
		skeleton = nullptr;
		options->hide();
	}

	_update_properties();
}

void Skeleton3DEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_node_removed"), &Skeleton3DEditor::_node_removed);
	ClassDB::bind_method(D_METHOD("_joint_tree_selection_changed"), &Skeleton3DEditor::_joint_tree_selection_changed);
	ClassDB::bind_method(D_METHOD("_joint_tree_rmb_select"), &Skeleton3DEditor::_joint_tree_rmb_select);
	ClassDB::bind_method(D_METHOD("_update_properties"), &Skeleton3DEditor::_update_properties);
	ClassDB::bind_method(D_METHOD("_on_click_option"), &Skeleton3DEditor::_on_click_option);

	ClassDB::bind_method(D_METHOD("rest_mode_toggled"), &Skeleton3DEditor::rest_mode_toggled);
	ClassDB::bind_method(D_METHOD("set_rest_mode_toggled"), &Skeleton3DEditor::set_rest_mode_toggled);

	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &Skeleton3DEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &Skeleton3DEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &Skeleton3DEditor::drop_data_fw);

	ClassDB::bind_method(D_METHOD("move_skeleton_bone"), &Skeleton3DEditor::move_skeleton_bone);

	ClassDB::bind_method(D_METHOD("_draw_handles"), &Skeleton3DEditor::_draw_handles);
}

void Skeleton3DEditor::_menu_tool_item_pressed(int p_option) {
	if (p_option != TOOL_MODE_BONE_NONE && !Node3DEditor::get_singleton()->is_tool_external()) {
		Node3DEditor::get_singleton()->set_tool_mode(Node3DEditor::TOOL_MODE_EXTERNAL);
	}
	for (int i = 0; i < TOOL_MODE_BONE_MAX; i++) {
		tool_button[i]->set_pressed(i == p_option);
	}
	tool_mode = (ToolMode)p_option;
	if (skeleton) {
		if (p_option == TOOL_MODE_BONE_NONE) {
			_hide_handles();
		} else {
			_draw_handles();
			if (skeleton->get_selected_bone() >= 0) {
				Node3DEditor::get_singleton()->clear_externals();
				Node3DEditor::get_singleton()->append_to_externals(skeleton->get_global_transform() * skeleton->get_bone_global_pose(skeleton->get_selected_bone()));
			}
		}
	}

	switch (p_option) {
		case TOOL_MODE_BONE_SELECT: {
			Node3DEditor::get_singleton()->set_external_tool_mode(Node3DEditor::EX_TOOL_MODE_SELECT);
		} break;
		case TOOL_MODE_BONE_MOVE: {
			Node3DEditor::get_singleton()->set_external_tool_mode(Node3DEditor::EX_TOOL_MODE_MOVE);
		} break;
		case TOOL_MODE_BONE_ROTATE: {
			Node3DEditor::get_singleton()->set_external_tool_mode(Node3DEditor::EX_TOOL_MODE_ROTATE);
		} break;
		case TOOL_MODE_BONE_SCALE: {
			Node3DEditor::get_singleton()->set_external_tool_mode(Node3DEditor::EX_TOOL_MODE_SCALE);
		} break;
		case TOOL_MODE_BONE_NONE:
			break;
	}

	_update_sub_gizmo();
}

void Skeleton3DEditor::rest_mode_toggled(const bool pressed) {
	bool before_val = rest_mode;

	// Prevent that bone pose will be undo during rest mode.
	// However SkeletonEditor will be memdeleted,
	// so it need to record in SpatialEditor with calling method in
	// EditorInspectorPluginSkeleton and it will not be memdeleted.
	UndoRedo *ur = Node3DEditor::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Toggled Rest Mode"));
	set_rest_mode_toggled(pressed);
	ur->add_undo_method(editor_plugin, "set_rest_mode_toggled", before_val);
	ur->add_do_method(editor_plugin, "set_rest_mode_toggled", pressed);
	ur->commit_action();
}

void Skeleton3DEditor::set_rest_mode_toggled(const bool pressed, const bool destructing) {
	rest_mode_button->disconnect("toggled", callable_mp(this, &Skeleton3DEditor::rest_mode_toggled));
	rest_mode_button->set_pressed(pressed);
	rest_mode_button->connect("toggled", callable_mp(this, &Skeleton3DEditor::rest_mode_toggled));

	rest_mode = pressed;
	const int bone_len = skeleton->get_bone_count();
	for (int i = 0; i < bone_len; i++) {
		skeleton->set_bone_enabled(i, !rest_mode);
	}
	if (!destructing) {
		if (pose_editor) {
			pose_editor->set_read_only(rest_mode);
		}
		if (custom_pose_editor) {
			custom_pose_editor->set_read_only(rest_mode);
		}
	}

	set_keyable(AnimationPlayerEditor::singleton->get_track_editor()->has_keying() && !rest_mode);
}

Skeleton3DEditor::Skeleton3DEditor(EditorInspectorPluginSkeleton *e_plugin, EditorNode *p_editor, Skeleton3D *p_skeleton) :
		editor(p_editor),
		editor_plugin(e_plugin),
		skeleton(p_skeleton) {
	handle_material = Ref<ShaderMaterial>(memnew(ShaderMaterial));
	handle_shader = Ref<Shader>(memnew(Shader));
	handle_shader->set_code(" \
		shader_type spatial; \
		render_mode unshaded, shadows_disabled, depth_draw_always; \
		uniform vec4 albedo : hint_color = vec4(1,1,1,1); \
		uniform sampler2D texture_albedo : hint_albedo; \
		uniform float point_size : hint_range(0,128) = 32; \
		void vertex() { \
			if (!OUTPUT_IS_SRGB) { \
				COLOR.rgb = mix( pow((COLOR.rgb + vec3(0.055)) * (1.0 / (1.0 + 0.055)), vec3(2.4)), COLOR.rgb* (1.0 / 12.92), lessThan(COLOR.rgb,vec3(0.04045)) ); \
			} \
			VERTEX = VERTEX; \
			POSITION=PROJECTION_MATRIX*INV_CAMERA_MATRIX*WORLD_MATRIX*vec4(VERTEX.xyz,1.0); \
			POSITION.z = mix(POSITION.z, 0, 0.999); \
			POINT_SIZE = point_size; \
		} \
		void fragment() { \
			vec4 albedo_tex = texture(texture_albedo,POINT_COORD); \
			vec3 col = albedo_tex.rgb + COLOR.rgb; \
			col = vec3(min(col.r,1.0),min(col.g,1.0),min(col.b,1.0)); \
			ALBEDO = albedo.rgb * col; \
			if (albedo.a * albedo_tex.a < 0.5) { discard; } \
			ALPHA = albedo.a * albedo_tex.a; \
		} \
	");
	handle_material->set_shader(handle_shader);
	Ref<Texture2D> handle = editor->get_gui_base()->get_theme_icon("EditorBoneHandle", "EditorIcons");
	handle_material->set_shader_param("point_size", handle->get_width());
	handle_material->set_shader_param("texture_albedo", handle);

	pointsm = memnew(MeshInstance3D);
	pointsm->set_cast_shadows_setting(GeometryInstance3D::ShadowCastingSetting::SHADOW_CASTING_SETTING_OFF);
	am.instantiate();
	pointsm->set_mesh(am);
}

void Skeleton3DEditor::_hide_handles() {
	if (!skeleton) {
		return;
	}
	pointsm->hide();
}

void Skeleton3DEditor::_draw_handles() {
	if (!skeleton || tool_mode == TOOL_MODE_BONE_NONE) {
		return;
	}

	am->clear_surfaces();

	pointsm->show();

	Array a;
	a.resize(Mesh::ARRAY_MAX);
	Vector<Vector3> va;
	Vector<Color> ca;

	const int bone_len = skeleton->get_bone_count();
	va.resize(bone_len);
	ca.resize(bone_len);
	VectorWriteProxy<Vector3> &vaw = va.write;
	VectorWriteProxy<Color> &caw = ca.write;

	for (int i = 0; i < bone_len; i++) {
		Vector3 point = skeleton->get_bone_global_pose(i).origin;
		vaw[i] = point;
		Color c;
		if (i == skeleton->get_selected_bone()) {
			c = Color(1, 1, 0);
		} else {
			c = Color(0.1, 0.25, 0.8);
		}
		caw[i] = c;
	}

	a[Mesh::ARRAY_VERTEX] = va;
	a[Mesh::ARRAY_COLOR] = ca;
	am->add_surface_from_arrays(Mesh::PRIMITIVE_POINTS, a);
	am->surface_set_material(0, handle_material);
}

void Skeleton3DEditor::_compute_edit(int p_index, const Point2 &p_point) {
	Node3DEditor *se = Node3DEditor::get_singleton();
	Node3DEditorViewport *sev = se->get_editor_viewport(p_index);

	_edit.click_ray = sev->get_ray(Vector2(p_point.x, p_point.y));
	_edit.click_ray_pos = sev->get_ray_pos(Vector2(p_point.x, p_point.y));
	_edit.plane = Node3DEditorViewport::TRANSFORM_VIEW;
	_update_sub_gizmo();
	_edit.center = se->get_gizmo_transform().origin;

	if (skeleton->get_selected_bone() != -1) {
		original_global = skeleton->get_global_transform() * skeleton->get_bone_global_pose(skeleton->get_selected_bone());
		if (rest_mode) {
			original_local = skeleton->get_bone_rest(skeleton->get_selected_bone());
		} else {
			original_local = skeleton->get_bone_pose(skeleton->get_selected_bone());
		}
		original_to_local = skeleton->get_global_transform();
		int parent_idx = skeleton->get_bone_parent(skeleton->get_selected_bone());
		if (parent_idx >= 0) {
			original_to_local = original_to_local * skeleton->get_bone_global_pose(parent_idx);
		}
		if (!rest_mode) {
			original_to_local = original_to_local * skeleton->get_bone_rest(skeleton->get_selected_bone()) * skeleton->get_bone_custom_pose(skeleton->get_selected_bone());
		}
	}
}

bool Skeleton3DEditor::_gizmo_select(int p_index, const Vector2 &p_screenpos, bool p_highlight_only) {
	Node3DEditor *se = Node3DEditor::get_singleton();
	Node3DEditorViewport *sev = se->get_editor_viewport(p_index);

	if (!se->is_gizmo_visible()) {
		return false;
	}
	if (skeleton->get_selected_bone() == -1) {
		if (p_highlight_only) {
			se->select_gizmo_highlight_axis(-1);
		}
		return false;
	}

	Vector3 ray_pos = sev->get_ray_pos(Vector2(p_screenpos.x, p_screenpos.y));
	Vector3 ray = sev->get_ray(Vector2(p_screenpos.x, p_screenpos.y));

	Transform3D gt = se->get_gizmo_transform();
	float gs = sev->get_gizmo_scale();

	if (se->get_external_tool_mode() == Node3DEditor::EX_TOOL_MODE_SELECT || se->get_external_tool_mode() == Node3DEditor::EX_TOOL_MODE_MOVE) {
		int col_axis = -1;
		float col_d = 1e20;

		for (int i = 0; i < 3; i++) {
			Vector3 grabber_pos = gt.origin + gt.basis.get_axis(i) * gs * (GIZMO_ARROW_OFFSET + (GIZMO_ARROW_SIZE * 0.5));
			float grabber_radius = gs * GIZMO_ARROW_SIZE;

			Vector3 r;

			if (Geometry3D::segment_intersects_sphere(ray_pos, ray_pos + ray * MAX_Z, grabber_pos, grabber_radius, &r)) {
				float d = r.distance_to(ray_pos);
				if (d < col_d) {
					col_d = d;
					col_axis = i;
				}
			}
		}

		bool is_plane_translate = false;
		// plane select
		if (col_axis == -1) {
			col_d = 1e20;

			for (int i = 0; i < 3; i++) {
				Vector3 ivec2 = gt.basis.get_axis((i + 1) % 3).normalized();
				Vector3 ivec3 = gt.basis.get_axis((i + 2) % 3).normalized();

				Vector3 grabber_pos = gt.origin + (ivec2 + ivec3) * gs * (GIZMO_PLANE_SIZE + GIZMO_PLANE_DST);

				Vector3 r;
				Plane plane(gt.origin, gt.basis.get_axis(i).normalized());

				if (plane.intersects_ray(ray_pos, ray, &r)) {
					float dist = r.distance_to(grabber_pos);
					if (dist < (gs * GIZMO_PLANE_SIZE)) {
						float d = ray_pos.distance_to(r);
						if (d < col_d) {
							col_d = d;
							col_axis = i;
							is_plane_translate = true;
						}
					}
				}
			}
		}

		if (col_axis != -1) {
			if (p_highlight_only) {
				se->select_gizmo_highlight_axis(col_axis + (is_plane_translate ? 6 : 0));
			} else {
				//handle plane translate
				_edit.mode = Node3DEditorViewport::TRANSFORM_TRANSLATE;
				_compute_edit(p_index, Point2(p_screenpos.x, p_screenpos.y));
				_edit.plane = Node3DEditorViewport::TransformPlane(Node3DEditorViewport::TRANSFORM_X_AXIS + col_axis + (is_plane_translate ? 3 : 0));
			}
			return true;
		}
	}

	if (se->get_external_tool_mode() == Node3DEditor::EX_TOOL_MODE_SELECT || se->get_external_tool_mode() == Node3DEditor::EX_TOOL_MODE_ROTATE) {
		int col_axis = -1;
		float col_d = 1e20;

		for (int i = 0; i < 3; i++) {
			Plane plane(gt.origin, gt.basis.get_axis(i).normalized());
			Vector3 r;
			if (!plane.intersects_ray(ray_pos, ray, &r)) {
				continue;
			}
			float dist = r.distance_to(gt.origin);
			if (dist > gs * (GIZMO_CIRCLE_SIZE - GIZMO_RING_HALF_WIDTH) && dist < gs * (GIZMO_CIRCLE_SIZE + GIZMO_RING_HALF_WIDTH)) {
				float d = ray_pos.distance_to(r);
				if (d < col_d) {
					col_d = d;
					col_axis = i;
				}
			}
		}

		if (col_axis != -1) {
			if (p_highlight_only) {
				se->select_gizmo_highlight_axis(col_axis + 3);
			} else {
				//handle rotate
				_edit.mode = Node3DEditorViewport::TRANSFORM_ROTATE;
				_compute_edit(p_index, Point2(p_screenpos.x, p_screenpos.y));
				_edit.plane = Node3DEditorViewport::TransformPlane(Node3DEditorViewport::TRANSFORM_X_AXIS + col_axis);
			}
			return true;
		}
	}

	if (se->get_external_tool_mode() == Node3DEditor::EX_TOOL_MODE_SCALE) {
		int col_axis = -1;
		float col_d = 1e20;
		for (int i = 0; i < 3; i++) {
			Vector3 grabber_pos = gt.origin + gt.basis.get_axis(i) * gs * GIZMO_SCALE_OFFSET;
			float grabber_radius = gs * GIZMO_ARROW_SIZE;
			Vector3 r;
			if (Geometry3D::segment_intersects_sphere(ray_pos, ray_pos + ray * MAX_Z, grabber_pos, grabber_radius, &r)) {
				float d = r.distance_to(ray_pos);
				if (d < col_d) {
					col_d = d;
					col_axis = i;
				}
			}
		}

		bool is_plane_scale = false;
		// plane select
		if (col_axis == -1) {
			col_d = 1e20;
			for (int i = 0; i < 3; i++) {
				Vector3 ivec2 = gt.basis.get_axis((i + 1) % 3).normalized();
				Vector3 ivec3 = gt.basis.get_axis((i + 2) % 3).normalized();

				Vector3 grabber_pos = gt.origin + (ivec2 + ivec3) * gs * (GIZMO_PLANE_SIZE + GIZMO_PLANE_DST);

				Vector3 r;
				Plane plane(gt.origin, gt.basis.get_axis(i).normalized());

				if (plane.intersects_ray(ray_pos, ray, &r)) {
					float dist = r.distance_to(grabber_pos);
					if (dist < (gs * GIZMO_PLANE_SIZE)) {
						float d = ray_pos.distance_to(r);
						if (d < col_d) {
							col_d = d;
							col_axis = i;
							is_plane_scale = true;
						}
					}
				}
			}
		}

		if (col_axis != -1) {
			if (p_highlight_only) {
				se->select_gizmo_highlight_axis(col_axis + (is_plane_scale ? 12 : 9));
			} else {
				//handle scale
				_edit.mode = Node3DEditorViewport::TRANSFORM_SCALE;
				_compute_edit(p_index, Point2(p_screenpos.x, p_screenpos.y));
				_edit.plane = Node3DEditorViewport::TransformPlane(Node3DEditorViewport::TRANSFORM_X_AXIS + col_axis + (is_plane_scale ? 3 : 0));
			}
			return true;
		}
	}

	if (p_highlight_only) {
		se->select_gizmo_highlight_axis(-1);
	}
	return false;
}

TreeItem *Skeleton3DEditor::_find(TreeItem *p_node, const NodePath &p_path) {
	if (!p_node) {
		return nullptr;
	}

	NodePath np = p_node->get_metadata(0);
	if (np == p_path) {
		return p_node;
	}

	TreeItem *children = p_node->get_first_child();
	while (children) {
		TreeItem *n = _find(children, p_path);
		if (n) {
			return n;
		}
		children = children->get_next();
	}

	return nullptr;
}

bool Skeleton3DEditor::forward_spatial_gui_input(int p_index, Camera3D *p_camera, const Ref<InputEvent> &p_event) {
	if (!skeleton || tool_mode == TOOL_MODE_BONE_NONE) {
		return false;
	}

	Node3DEditor *se = Node3DEditor::get_singleton();
	Node3DEditorViewport *sev = se->get_editor_viewport(p_index);

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		Transform3D gt = skeleton->get_global_transform();
		Vector3 ray_from = p_camera->get_global_transform().origin;
		Vector2 gpoint = mb->get_position();
		real_t grab_threshold = 4 * EDSCALE;

		switch (mb->get_button_index()) {
			case MOUSE_BUTTON_LEFT: {
				if (mb->is_pressed()) {
					_edit.mouse_pos = mb->get_position();
					_edit.snap = se->is_snap_enabled();
					_edit.mode = Node3DEditorViewport::TRANSFORM_NONE;

					// check gizmo
					if (_gizmo_select(p_index, _edit.mouse_pos)) {
						return true;
					}

					// select bone
					int closest_idx = -1;
					real_t closest_dist = 1e10;
					const int bone_len = skeleton->get_bone_count();
					for (int i = 0; i < bone_len; i++) {
						Vector3 joint_pos_3d = gt.xform(skeleton->get_bone_global_pose(i).origin);
						Vector2 joint_pos_2d = p_camera->unproject_position(joint_pos_3d);
						real_t dist_3d = ray_from.distance_to(joint_pos_3d);
						real_t dist_2d = gpoint.distance_to(joint_pos_2d);
						if (dist_2d < grab_threshold && dist_3d < closest_dist) {
							closest_dist = dist_3d;
							closest_idx = i;
						}
					}
					if (closest_idx >= 0) {
						TreeItem *ti = _find(joint_tree->get_root(), "bones/" + itos(closest_idx));
						if (ti) {
							// make visible when it's collapsed
							TreeItem *node = ti->get_parent();
							while (node && node != joint_tree->get_root()) {
								node->set_collapsed(false);
								node = node->get_parent();
							}
							ti->select(0);
							joint_tree->scroll_to_item(ti);
						}
					} else {
						skeleton->set_selected_bone(-1);
						joint_tree->deselect_all();
					}

				} else {
					if (_edit.mode != Node3DEditorViewport::TRANSFORM_NONE) {
						if (skeleton && (skeleton->get_selected_bone() >= 0)) {
							UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
							ur->create_action(TTR("Set Bone Transform"), UndoRedo::MERGE_ENDS);
							if (rest_mode) {
								ur->add_do_method(skeleton, "set_bone_rest", skeleton->get_selected_bone(), skeleton->get_bone_rest(skeleton->get_selected_bone()));
								ur->add_undo_method(skeleton, "set_bone_rest", skeleton->get_selected_bone(), original_local);
							} else {
								ur->add_do_method(skeleton, "set_bone_pose", skeleton->get_selected_bone(), skeleton->get_bone_pose(skeleton->get_selected_bone()));
								ur->add_undo_method(skeleton, "set_bone_pose", skeleton->get_selected_bone(), original_local);
							}
							ur->commit_action();
							_edit.mode = Node3DEditorViewport::TRANSFORM_NONE;
						}
					}
				}
				return true;
			} break;
			default: {
				break;
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		_edit.mouse_pos = mm->get_position();
		if (!(mm->get_button_mask() & 1)) {
			_gizmo_select(p_index, _edit.mouse_pos, true);
		}
		if (mm->get_button_mask() & MOUSE_BUTTON_MASK_LEFT) {
			if (_edit.mode == Node3DEditorViewport::TRANSFORM_NONE) {
				return true;
			}

			Vector3 ray_pos = sev->get_ray_pos(mm->get_position());
			Vector3 ray = sev->get_ray(mm->get_position());
			float snap = EDITOR_GET("interface/inspector/default_float_step");

			switch (_edit.mode) {
				case Node3DEditorViewport::TRANSFORM_SCALE: {
					Vector3 motion_mask;
					Plane plane;
					bool plane_mv = false;
					switch (_edit.plane) {
						case Node3DEditorViewport::TRANSFORM_VIEW: {
							motion_mask = Vector3(0, 0, 0);
							plane = Plane(_edit.center, sev->get_camera_normal());
							break;
						}
						case Node3DEditorViewport::TRANSFORM_X_AXIS: {
							motion_mask = se->get_gizmo_transform().basis.get_axis(0);
							plane = Plane(_edit.center, motion_mask.cross(motion_mask.cross(sev->get_camera_normal())).normalized());
							break;
						}
						case Node3DEditorViewport::TRANSFORM_Y_AXIS: {
							motion_mask = se->get_gizmo_transform().basis.get_axis(1);
							plane = Plane(_edit.center, motion_mask.cross(motion_mask.cross(sev->get_camera_normal())).normalized());
							break;
						}
						case Node3DEditorViewport::TRANSFORM_Z_AXIS: {
							motion_mask = se->get_gizmo_transform().basis.get_axis(2);
							plane = Plane(_edit.center, motion_mask.cross(motion_mask.cross(sev->get_camera_normal())).normalized());
							break;
						}
						case Node3DEditorViewport::TRANSFORM_YZ: {
							motion_mask = se->get_gizmo_transform().basis.get_axis(2) + se->get_gizmo_transform().basis.get_axis(1);
							plane = Plane(_edit.center, se->get_gizmo_transform().basis.get_axis(0));
							plane_mv = true;
							break;
						}
						case Node3DEditorViewport::TRANSFORM_XZ: {
							motion_mask = se->get_gizmo_transform().basis.get_axis(2) + se->get_gizmo_transform().basis.get_axis(0);
							plane = Plane(_edit.center, se->get_gizmo_transform().basis.get_axis(1));
							plane_mv = true;
							break;
						}
						case Node3DEditorViewport::TRANSFORM_XY: {
							motion_mask = se->get_gizmo_transform().basis.get_axis(0) + se->get_gizmo_transform().basis.get_axis(1);
							plane = Plane(_edit.center, se->get_gizmo_transform().basis.get_axis(2));
							plane_mv = true;
							break;
						}
					}
					Vector3 intersection;
					if (!plane.intersects_ray(ray_pos, ray, &intersection)) {
						break;
					}
					Vector3 click;
					if (!plane.intersects_ray(_edit.click_ray_pos, _edit.click_ray, &click)) {
						break;
					}
					Vector3 motion = intersection - click;
					if (_edit.plane != Node3DEditorViewport::TRANSFORM_VIEW) {
						if (!plane_mv) {
							motion = motion_mask.dot(motion) * motion_mask;
						}
					} else {
						float center_click_dist = click.distance_to(_edit.center);
						float center_inters_dist = intersection.distance_to(_edit.center);
						if (center_click_dist == 0) {
							break;
						}
						float scale = center_inters_dist - center_click_dist;
						motion = Vector3(scale, scale, scale);
					}
					bool local_coords = (se->are_local_coords_enabled() && _edit.plane != Node3DEditorViewport::TRANSFORM_VIEW);
					if (_edit.snap || se->is_snap_enabled()) {
						snap = se->get_scale_snap() / 100;
					}
					Transform3D t;
					if (local_coords) {
						Basis g = original_global.basis;
						motion = g.inverse().xform(motion);
						if (_edit.snap || se->is_snap_enabled()) {
							motion.snap(Vector3(snap, snap, snap));
						}
						Vector3 local_scale = original_local.basis.get_scale() * (motion + Vector3(1, 1, 1));
						// Prevent scaling to 0 it would break the gizmo
						Basis check = original_local.basis;
						check.scale(local_scale);
						if (check.determinant() != 0) {
							t = original_local;
							t.basis = t.basis.scaled_local(motion + Vector3(1, 1, 1));
						}
					} else {
						if (_edit.snap || se->is_snap_enabled()) {
							motion.snap(Vector3(snap, snap, snap));
						}
						t = original_local;
						Transform3D r;
						r.basis.scale(motion + Vector3(1, 1, 1));
						Basis base = original_to_local.get_basis().orthonormalized().inverse();
						t.basis = base * (r.get_basis() * (base.inverse() * original_local.get_basis()));
					}
					// Apply scale
					if (rest_mode) {
						skeleton->set_bone_rest(skeleton->get_selected_bone(), t);
					} else {
						skeleton->set_bone_pose(skeleton->get_selected_bone(), t);
					}
					sev->update_surface();
				} break;
				case Node3DEditorViewport::TRANSFORM_TRANSLATE: {
					Vector3 motion_mask;
					Plane plane;
					bool plane_mv = false;
					switch (_edit.plane) {
						case Node3DEditorViewport::TRANSFORM_VIEW: {
							plane = Plane(_edit.center, sev->get_camera_normal());
							break;
						}
						case Node3DEditorViewport::TRANSFORM_X_AXIS: {
							motion_mask = se->get_gizmo_transform().basis.get_axis(0);
							plane = Plane(_edit.center, motion_mask.cross(motion_mask.cross(sev->get_camera_normal())).normalized());
							break;
						}
						case Node3DEditorViewport::TRANSFORM_Y_AXIS: {
							motion_mask = se->get_gizmo_transform().basis.get_axis(1);
							plane = Plane(_edit.center, motion_mask.cross(motion_mask.cross(sev->get_camera_normal())).normalized());
							break;
						}
						case Node3DEditorViewport::TRANSFORM_Z_AXIS: {
							motion_mask = se->get_gizmo_transform().basis.get_axis(2);
							plane = Plane(_edit.center, motion_mask.cross(motion_mask.cross(sev->get_camera_normal())).normalized());
							break;
						}
						case Node3DEditorViewport::TRANSFORM_YZ: {
							plane = Plane(_edit.center, se->get_gizmo_transform().basis.get_axis(0));
							plane_mv = true;
							break;
						}
						case Node3DEditorViewport::TRANSFORM_XZ: {
							plane = Plane(_edit.center, se->get_gizmo_transform().basis.get_axis(1));
							plane_mv = true;
							break;
						}
						case Node3DEditorViewport::TRANSFORM_XY: {
							plane = Plane(_edit.center, se->get_gizmo_transform().basis.get_axis(2));
							plane_mv = true;
							break;
						}
					}
					Vector3 intersection;
					if (!plane.intersects_ray(ray_pos, ray, &intersection)) {
						break;
					}
					Vector3 click;
					if (!plane.intersects_ray(_edit.click_ray_pos, _edit.click_ray, &click)) {
						break;
					}
					Vector3 motion = intersection - click;
					if (_edit.plane != Node3DEditorViewport::TRANSFORM_VIEW) {
						if (!plane_mv) {
							motion = motion_mask.dot(motion) * motion_mask;
						}
					}
					if (_edit.snap || se->is_snap_enabled()) {
						snap = se->get_translate_snap();
					}
					motion = original_to_local.basis.inverse().xform(motion);
					if (_edit.snap || se->is_snap_enabled()) {
						motion.snap(Vector3(snap, snap, snap));
					}
					Transform3D t;
					// Apply translation
					t = original_local;
					t.origin += motion;
					if (rest_mode) {
						skeleton->set_bone_rest(skeleton->get_selected_bone(), t);
					} else {
						skeleton->set_bone_pose(skeleton->get_selected_bone(), t);
					}
					sev->update_surface();
				} break;
				case Node3DEditorViewport::TRANSFORM_ROTATE: {
					Plane plane;
					Vector3 axis;
					switch (_edit.plane) {
						case Node3DEditorViewport::TRANSFORM_VIEW: {
							plane = Plane(_edit.center, sev->get_camera_normal());
							break;
						}
						case Node3DEditorViewport::TRANSFORM_X_AXIS: {
							plane = Plane(_edit.center, se->get_gizmo_transform().basis.get_axis(0));
							axis = Vector3(1, 0, 0);
							break;
						}
						case Node3DEditorViewport::TRANSFORM_Y_AXIS: {
							plane = Plane(_edit.center, se->get_gizmo_transform().basis.get_axis(1));
							axis = Vector3(0, 1, 0);
							break;
						}
						case Node3DEditorViewport::TRANSFORM_Z_AXIS: {
							plane = Plane(_edit.center, se->get_gizmo_transform().basis.get_axis(2));
							axis = Vector3(0, 0, 1);
							break;
						}
						case Node3DEditorViewport::TRANSFORM_YZ:
						case Node3DEditorViewport::TRANSFORM_XZ:
						case Node3DEditorViewport::TRANSFORM_XY: {
							break;
						}
					}
					Vector3 intersection;
					if (!plane.intersects_ray(ray_pos, ray, &intersection)) {
						break;
					}

					Vector3 click;
					if (!plane.intersects_ray(_edit.click_ray_pos, _edit.click_ray, &click)) {
						break;
					}

					Vector3 y_axis = (click - _edit.center).normalized();
					Vector3 x_axis = plane.normal.cross(y_axis).normalized();

					float angle = Math::atan2(x_axis.dot(intersection - _edit.center), y_axis.dot(intersection - _edit.center));

					if (_edit.snap || se->is_snap_enabled()) {
						snap = se->get_rotate_snap();
					}
					angle = Math::rad2deg(angle) + snap * 0.5; //else it won't reach +180
					angle -= Math::fmod(angle, snap);
					// set_message(vformat(TTR("Rotating %s degrees."), String::num(angle, snap_step_decimals)));
					angle = Math::deg2rad(angle);

					bool local_coords = (se->are_local_coords_enabled() && _edit.plane != Node3DEditorViewport::TRANSFORM_VIEW); // Disable local transformation for TRANSFORM_VIEW

					Transform3D t;

					if (local_coords) {
						Basis rot = Basis(axis, angle);
						t.basis = original_local.get_basis().orthonormalized() * rot;
						t.basis = t.basis.scaled_local(original_local.basis.get_scale());
						t.origin = original_local.origin;
					} else {
						Transform3D r;
						Basis base = original_to_local.get_basis().orthonormalized().inverse();
						r.basis.rotate(plane.normal, angle);
						t.basis = base * r.get_basis() * base.inverse() * original_local.get_basis();
						// t.basis = t.basis.scaled(original_local.basis.get_scale());
						t.origin = original_local.origin;
					}
					// Apply rotation
					if (rest_mode) {
						skeleton->set_bone_rest(skeleton->get_selected_bone(), t);
					} else {
						skeleton->set_bone_pose(skeleton->get_selected_bone(), t);
					}
					sev->update_surface();
				} break;
				default: {
					break;
				}
			}
			return true;
		}
	}
	return false;
}

Skeleton3DEditor::~Skeleton3DEditor() {
	set_rest_mode_toggled(false, true);
	Node3DEditor::get_singleton()->disconnect("change_tool_mode", callable_mp(this, &Skeleton3DEditor::_menu_tool_item_pressed));
	if (skeleton) {
		pointsm->get_parent()->remove_child(pointsm);
		skeleton->set_selected_bone(-1);
		skeleton->disconnect("pose_updated", callable_mp(this, &Skeleton3DEditor::_draw_handles));
		memdelete(pointsm);
	}
	for (int i = 0; i < 2; i++) {
		if (separators[i]) {
			Node3DEditor::get_singleton()->remove_control_from_menu_panel(separators[i]);
			memdelete(separators[i]);
		}
	}
	if (options) {
		Node3DEditor::get_singleton()->remove_control_from_menu_panel(options);
		memdelete(options);
	}
	Node3DEditor::get_singleton()->remove_control_from_menu_panel(tool_button[TOOL_MODE_BONE_SELECT]);
	Node3DEditor::get_singleton()->remove_control_from_menu_panel(tool_button[TOOL_MODE_BONE_MOVE]);
	Node3DEditor::get_singleton()->remove_control_from_menu_panel(tool_button[TOOL_MODE_BONE_ROTATE]);
	Node3DEditor::get_singleton()->remove_control_from_menu_panel(tool_button[TOOL_MODE_BONE_SCALE]);
	for (int i = 0; i < TOOL_MODE_BONE_MAX; i++) {
		if (tool_button[i]) {
			memdelete(tool_button[i]);
		}
	}
	Node3DEditor::get_singleton()->remove_control_from_menu_panel(rest_mode_button);
	if (rest_mode_button) {
		memdelete(rest_mode_button);
	}
	if (Node3DEditor::get_singleton()->is_tool_external()) {
		Node3DEditor::get_singleton()->set_tool_mode(Node3DEditor::TOOL_MODE_SELECT);
		Node3DEditor::get_singleton()->set_external_tool_mode(Node3DEditor::EX_TOOL_MODE_SELECT);
	}
}

void EditorInspectorPluginSkeleton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_rest_mode_toggled"), &EditorInspectorPluginSkeleton::set_rest_mode_toggled);
}

bool EditorInspectorPluginSkeleton::can_handle(Object *p_object) {
	return Object::cast_to<Skeleton3D>(p_object) != nullptr;
}

void EditorInspectorPluginSkeleton::parse_begin(Object *p_object) {
	Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(p_object);
	ERR_FAIL_COND(!skeleton);

	skel_editor = memnew(Skeleton3DEditor(this, editor, skeleton));
	add_custom_control(skel_editor);
}

void EditorInspectorPluginSkeleton::set_rest_mode_toggled(const bool p_pressed) {
	if (Node3DEditor::get_singleton()->get_selected()->get_class() == "Skeleton3D" && skel_editor) {
		skel_editor->set_rest_mode_toggled(p_pressed);
	}
}

Skeleton3DEditorPlugin::Skeleton3DEditorPlugin(EditorNode *p_node) {
	editor = p_node;

	skeleton_plugin = memnew(EditorInspectorPluginSkeleton);
	skeleton_plugin->editor = editor;

	EditorInspector::add_inspector_plugin(skeleton_plugin);
}

bool Skeleton3DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Skeleton3D");
}
