/*************************************************************************/
/*  skeleton_editor_plugin.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "skeleton_editor_plugin.h"

#include "core/io/resource_saver.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_properties.h"
#include "editor/editor_scale.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "scene/3d/collision_shape.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/physics_body.h"
#include "scene/3d/physics_joint.h"
#include "scene/resources/capsule_shape.h"
#include "scene/resources/sphere_shape.h"
#include "spatial_editor_plugin.h"

void BoneTransformEditor::create_editors() {
	const Color section_color = get_color("prop_subsection", "Editor");

	section = memnew(EditorInspectorSection);
	section->setup("trf_properties", label, this, section_color, true);
	add_child(section);

	key_button = memnew(Button);
	key_button->set_text(TTR("Key Transform"));
	key_button->set_visible(keyable);
	key_button->set_icon(get_icon("Key", "EditorIcons"));
	key_button->set_flat(true);
	section->get_vbox()->add_child(key_button);

	enabled_checkbox = memnew(CheckBox(TTR("Pose Enabled")));
	enabled_checkbox->set_flat(true);
	enabled_checkbox->set_visible(toggle_enabled);
	section->get_vbox()->add_child(enabled_checkbox);

	Label *l1 = memnew(Label(TTR("Translation")));
	section->get_vbox()->add_child(l1);

	translation_grid = memnew(GridContainer());
	translation_grid->set_columns(TRANSLATION_COMPONENTS);
	section->get_vbox()->add_child(translation_grid);

	Label *l2 = memnew(Label(TTR("Rotation Degrees")));
	section->get_vbox()->add_child(l2);

	rotation_grid = memnew(GridContainer());
	rotation_grid->set_columns(ROTATION_DEGREES_COMPONENTS);
	section->get_vbox()->add_child(rotation_grid);

	Label *l3 = memnew(Label(TTR("Scale")));
	section->get_vbox()->add_child(l3);

	scale_grid = memnew(GridContainer());
	scale_grid->set_columns(SCALE_COMPONENTS);
	section->get_vbox()->add_child(scale_grid);

	Label *l4 = memnew(Label(TTR("Transform")));
	section->get_vbox()->add_child(l4);

	transform_grid = memnew(GridContainer());
	transform_grid->set_columns(TRANSFORM_CONTROL_COMPONENTS);
	section->get_vbox()->add_child(transform_grid);

	static const char *desc[TRANSFORM_COMPONENTS] = { "x", "y", "z", "x", "y", "z", "x", "y", "z", "x", "y", "z" };

	for (int i = 0; i < TRANSFORM_CONTROL_COMPONENTS; ++i) {
		translation_slider[i] = memnew(EditorSpinSlider());
		translation_slider[i]->set_label(desc[i]);
		setup_spinner(translation_slider[i], false);
		translation_grid->add_child(translation_slider[i]);

		rotation_slider[i] = memnew(EditorSpinSlider());
		rotation_slider[i]->set_label(desc[i]);
		setup_spinner(rotation_slider[i], false);
		rotation_grid->add_child(rotation_slider[i]);

		scale_slider[i] = memnew(EditorSpinSlider());
		scale_slider[i]->set_label(desc[i]);
		setup_spinner(scale_slider[i], false);
		scale_grid->add_child(scale_slider[i]);
	}

	for (int i = 0; i < TRANSFORM_COMPONENTS; ++i) {
		transform_slider[i] = memnew(EditorSpinSlider());
		transform_slider[i]->set_label(desc[i]);
		setup_spinner(transform_slider[i], true);
		transform_grid->add_child(transform_slider[i]);
	}
}

void BoneTransformEditor::setup_spinner(EditorSpinSlider *spinner, const bool is_transform_spinner) {
	spinner->set_flat(true);
	spinner->set_min(-10000);
	spinner->set_max(10000);
	spinner->set_step(0.001f);
	spinner->set_hide_slider(true);
	spinner->set_allow_greater(true);
	spinner->set_allow_lesser(true);
	spinner->set_h_size_flags(SIZE_EXPAND_FILL);

	spinner->connect("value_changed", this, "_value_changed", varray(is_transform_spinner));
}

void BoneTransformEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editors();
			key_button->connect("pressed", this, "_key_button_pressed");
			enabled_checkbox->connect("toggled", this, "_checkbox_toggled");
			FALLTHROUGH;
		}
		case NOTIFICATION_THEME_CHANGED: {
			const Color base = get_color("accent_color", "Editor");
			const Color bg_color = get_color("property_color", "Editor");
			const Color bg_lbl_color(bg_color.r, bg_color.g, bg_color.b, 0.5);

			for (int i = 0; i < TRANSLATION_COMPONENTS; i++) {
				Color c = base;
				c.set_hsv(float(i % TRANSLATION_COMPONENTS) / TRANSLATION_COMPONENTS + 0.05, c.get_s() * 0.75, c.get_v());
				if (!translation_slider[i]) {
					continue;
				}
				translation_slider[i]->set_custom_label_color(true, c);
			}

			for (int i = 0; i < ROTATION_DEGREES_COMPONENTS; i++) {
				Color c = base;
				c.set_hsv(float(i % ROTATION_DEGREES_COMPONENTS) / ROTATION_DEGREES_COMPONENTS + 0.05, c.get_s() * 0.75, c.get_v());
				if (!rotation_slider[i]) {
					continue;
				}
				rotation_slider[i]->set_custom_label_color(true, c);
			}

			for (int i = 0; i < SCALE_COMPONENTS; i++) {
				Color c = base;
				c.set_hsv(float(i % SCALE_COMPONENTS) / SCALE_COMPONENTS + 0.05, c.get_s() * 0.75, c.get_v());
				if (!scale_slider[i]) {
					continue;
				}
				scale_slider[i]->set_custom_label_color(true, c);
			}

			for (int i = 0; i < TRANSFORM_COMPONENTS; i++) {
				Color c = base;
				c.set_hsv(float(i % TRANSFORM_COMPONENTS) / TRANSFORM_COMPONENTS + 0.05, c.get_s() * 0.75, c.get_v());
				if (!transform_slider[i]) {
					continue;
				}
				transform_slider[i]->set_custom_label_color(true, c);
			}

			break;
		}
		case NOTIFICATION_SORT_CHILDREN: {
			const Ref<Font> font = get_font("font", "Tree");

			Point2 buffer;
			buffer.x += get_constant("inspector_margin", "Editor");
			buffer.y += font->get_height();
			buffer.y += get_constant("vseparation", "Tree");

			const float vector_height = translation_grid->get_size().y;
			const float transform_height = transform_grid->get_size().y;
			const float button_height = key_button->get_size().y;

			const float width = get_size().x - get_constant("inspector_margin", "Editor");
			Vector<Rect2> input_rects;
			if (keyable && section->get_vbox()->is_visible()) {
				input_rects.push_back(Rect2(key_button->get_position() + buffer, Size2(width, button_height)));
			} else {
				input_rects.push_back(Rect2(0, 0, 0, 0));
			}

			if (section->get_vbox()->is_visible()) {
				input_rects.push_back(Rect2(translation_grid->get_position() + buffer, Size2(width, vector_height)));
				input_rects.push_back(Rect2(rotation_grid->get_position() + buffer, Size2(width, vector_height)));
				input_rects.push_back(Rect2(scale_grid->get_position() + buffer, Size2(width, vector_height)));
				input_rects.push_back(Rect2(transform_grid->get_position() + buffer, Size2(width, transform_height)));
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
			const Color dark_color = get_color("dark_color_2", "Editor");

			for (int i = 0; i < 5; ++i) {
				draw_rect(background_rects[i], dark_color);
			}

			break;
		}
	}
}

void BoneTransformEditor::_value_changed(const double p_value, const bool p_from_transform) {
	if (updating)
		return;

	if (property.get_slicec('/', 0) == "bones" && property.get_slicec('/', 2) == "custom_pose") {
		const Transform tform = compute_transform(p_from_transform);

		undo_redo->create_action(TTR("Set Custom Bone Pose Transform"), UndoRedo::MERGE_ENDS);
		undo_redo->add_undo_method(skeleton, "set_bone_custom_pose", property.get_slicec('/', 1).to_int(), skeleton->get_bone_custom_pose(property.get_slicec('/', 1).to_int()));
		undo_redo->add_do_method(skeleton, "set_bone_custom_pose", property.get_slicec('/', 1).to_int(), tform);
		undo_redo->commit_action();
	} else if (property.get_slicec('/', 0) == "bones") {
		const Transform tform = compute_transform(p_from_transform);

		undo_redo->create_action(TTR("Set Bone Transform"), UndoRedo::MERGE_ENDS);
		undo_redo->add_undo_property(skeleton, property, skeleton->get(property));
		undo_redo->add_do_property(skeleton, property, tform);
		undo_redo->commit_action();
	}
}

Transform BoneTransformEditor::compute_transform(const bool p_from_transform) const {
	// Last modified was a raw transform column...
	if (p_from_transform) {
		Transform tform;

		for (int i = 0; i < BASIS_COMPONENTS; ++i) {
			tform.basis[i / BASIS_SPLIT_COMPONENTS][i % BASIS_SPLIT_COMPONENTS] = transform_slider[i]->get_value();
		}

		for (int i = 0; i < TRANSLATION_COMPONENTS; ++i) {
			tform.origin[i] = transform_slider[i + BASIS_COMPONENTS]->get_value();
		}

		return tform;
	}

	return Transform(
			Basis(Vector3(Math::deg2rad(rotation_slider[0]->get_value()), Math::deg2rad(rotation_slider[1]->get_value()), Math::deg2rad(rotation_slider[2]->get_value())),
					Vector3(scale_slider[0]->get_value(), scale_slider[1]->get_value(), scale_slider[2]->get_value())),
			Vector3(translation_slider[0]->get_value(), translation_slider[1]->get_value(), translation_slider[2]->get_value()));
}

void BoneTransformEditor::update_enabled_checkbox() {
	if (enabled_checkbox) {
		const String path = "bones/" + property.get_slicec('/', 1) + "/enabled";
		const bool is_enabled = skeleton->get(path);
		enabled_checkbox->set_pressed(is_enabled);
	}
}

void BoneTransformEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_value_changed", "value"), &BoneTransformEditor::_value_changed);
	ClassDB::bind_method(D_METHOD("_key_button_pressed"), &BoneTransformEditor::_key_button_pressed);
	ClassDB::bind_method(D_METHOD("_checkbox_toggled", "toggled"), &BoneTransformEditor::_checkbox_toggled);
}

void BoneTransformEditor::_update_properties() {
	if (updating)
		return;

	if (skeleton == nullptr)
		return;

	updating = true;

	Transform tform = skeleton->get(property);
	_update_transform_properties(tform);
}

void BoneTransformEditor::_update_custom_pose_properties() {
	if (updating)
		return;

	if (skeleton == nullptr)
		return;

	updating = true;

	Transform tform = skeleton->get_bone_custom_pose(property.to_int());
	_update_transform_properties(tform);
}

void BoneTransformEditor::_update_transform_properties(Transform tform) {
	Quat rot = tform.get_basis();
	Vector3 rot_rad = rot.get_euler();
	Vector3 rot_degrees = Vector3(Math::rad2deg(rot_rad.x), Math::rad2deg(rot_rad.y), Math::rad2deg(rot_rad.z));
	Vector3 tr = tform.get_origin();
	Vector3 scale = tform.basis.get_scale();

	for (int i = 0; i < TRANSLATION_COMPONENTS; i++) {
		translation_slider[i]->set_value(tr[i]);
	}

	for (int i = 0; i < ROTATION_DEGREES_COMPONENTS; i++) {
		rotation_slider[i]->set_value(rot_degrees[i]);
	}

	for (int i = 0; i < SCALE_COMPONENTS; i++) {
		scale_slider[i]->set_value(scale[i]);
	}

	transform_slider[0]->set_value(tform.get_basis()[Vector3::AXIS_X].x);
	transform_slider[1]->set_value(tform.get_basis()[Vector3::AXIS_X].y);
	transform_slider[2]->set_value(tform.get_basis()[Vector3::AXIS_X].z);
	transform_slider[3]->set_value(tform.get_basis()[Vector3::AXIS_Y].x);
	transform_slider[4]->set_value(tform.get_basis()[Vector3::AXIS_Y].y);
	transform_slider[5]->set_value(tform.get_basis()[Vector3::AXIS_Y].z);
	transform_slider[6]->set_value(tform.get_basis()[Vector3::AXIS_Z].x);
	transform_slider[7]->set_value(tform.get_basis()[Vector3::AXIS_Z].y);
	transform_slider[8]->set_value(tform.get_basis()[Vector3::AXIS_Z].z);

	for (int i = 0; i < TRANSLATION_COMPONENTS; i++) {
		transform_slider[BASIS_COMPONENTS + i]->set_value(tform.get_origin()[i]);
	}

	update_enabled_checkbox();
	updating = false;
}

BoneTransformEditor::BoneTransformEditor(Skeleton *p_skeleton) :
		translation_slider(),
		rotation_slider(),
		scale_slider(),
		transform_slider(),
		skeleton(p_skeleton),
		key_button(nullptr),
		enabled_checkbox(nullptr),
		keyable(false),
		toggle_enabled(false),
		updating(false) {
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
	if (skeleton == nullptr)
		return;

	const BoneId bone_id = property.get_slicec('/', 1).to_int();
	const String name = skeleton->get_bone_name(bone_id);

	if (name.empty())
		return;

	// Need to normalize the basis before you key it
	Transform tform = compute_transform(true);
	tform.orthonormalize();
	AnimationPlayerEditor::singleton->get_track_editor()->insert_transform_key(skeleton, name, tform);
}

void BoneTransformEditor::_checkbox_toggled(const bool p_toggled) {
	if (enabled_checkbox) {
		const String path = "bones/" + property.get_slicec('/', 1) + "/enabled";
		skeleton->set(path, p_toggled);
	}
}

void SkeletonEditor::_on_click_option(int p_option) {
	if (!skeleton) {
		return;
	}

	switch (p_option) {
		case MENU_OPTION_CREATE_PHYSICAL_SKELETON: {
			create_physical_skeleton();
			break;
		}
	}
}

void SkeletonEditor::create_physical_skeleton() {
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
					bones_infos[parent].physical_bone->set_joint_type(PhysicalBone::JOINT_TYPE_PIN);
				}
			}
		}
	}
}

PhysicalBone *SkeletonEditor::create_physical_bone(int bone_id, int bone_child_id, const Vector<BoneInfo> &bones_infos) {
	const Transform child_rest = skeleton->get_bone_rest(bone_child_id);

	const real_t half_height(child_rest.origin.length() * 0.5);
	const real_t radius(half_height * 0.2);

	CapsuleShape *bone_shape_capsule = memnew(CapsuleShape);
	bone_shape_capsule->set_height((half_height - radius) * 2);
	bone_shape_capsule->set_radius(radius);

	CollisionShape *bone_shape = memnew(CollisionShape);
	bone_shape->set_shape(bone_shape_capsule);

	Transform body_transform;
	body_transform.set_look_at(Vector3(0, 0, 0), child_rest.origin, Vector3(0, 1, 0));
	body_transform.origin = body_transform.basis.xform(Vector3(0, 0, -half_height));

	Transform joint_transform;
	joint_transform.origin = Vector3(0, 0, half_height);

	PhysicalBone *physical_bone = memnew(PhysicalBone);
	physical_bone->add_child(bone_shape);
	physical_bone->set_name("Physical Bone " + skeleton->get_bone_name(bone_id));
	physical_bone->set_body_offset(body_transform);
	physical_bone->set_joint_offset(joint_transform);
	return physical_bone;
}

Variant SkeletonEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	TreeItem *selected = joint_tree->get_selected();

	if (!selected)
		return Variant();

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

bool SkeletonEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	TreeItem *target = joint_tree->get_item_at_position(p_point);
	if (!target)
		return false;

	const String path = target->get_metadata(0);
	if (!path.begins_with("bones/"))
		return false;

	TreeItem *selected = Object::cast_to<TreeItem>(Dictionary(p_data)["node"]);
	if (target == selected)
		return false;

	const String path2 = target->get_metadata(0);
	if (!path2.begins_with("bones/"))
		return false;

	return true;
}

void SkeletonEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from))
		return;

	TreeItem *target = joint_tree->get_item_at_position(p_point);
	TreeItem *selected = Object::cast_to<TreeItem>(Dictionary(p_data)["node"]);

	const BoneId target_boneidx = String(target->get_metadata(0)).get_slicec('/', 1).to_int();
	const BoneId selected_boneidx = String(selected->get_metadata(0)).get_slicec('/', 1).to_int();

	move_skeleton_bone(skeleton->get_path(), selected_boneidx, target_boneidx);
}

void SkeletonEditor::move_skeleton_bone(NodePath p_skeleton_path, int32_t p_selected_boneidx, int32_t p_target_boneidx) {
	Node *node = get_node_or_null(p_skeleton_path);
	Skeleton *skeleton = Object::cast_to<Skeleton>(node);
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

void SkeletonEditor::_joint_tree_selection_changed() {
	TreeItem *selected = joint_tree->get_selected();
	const String path = selected->get_metadata(0);

	if (path.begins_with("bones/")) {
		const int b_idx = path.get_slicec('/', 1).to_int();
		const String bone_path = "bones/" + itos(b_idx) + "/";

		pose_editor->set_target(bone_path + "pose");
		rest_editor->set_target(bone_path + "rest");
		custom_pose_editor->set_target(bone_path + "custom_pose");

		pose_editor->set_visible(true);
		rest_editor->set_visible(true);
		custom_pose_editor->set_visible(true);
	}
}

void SkeletonEditor::_joint_tree_rmb_select(const Vector2 &p_pos) {
}

void SkeletonEditor::_update_properties() {
	if (rest_editor)
		rest_editor->_update_properties();
	if (pose_editor)
		pose_editor->_update_properties();
	if (custom_pose_editor)
		custom_pose_editor->_update_custom_pose_properties();
}

void SkeletonEditor::update_joint_tree() {
	joint_tree->clear();

	if (skeleton == nullptr)
		return;

	TreeItem *root = joint_tree->create_item();

	Map<int, TreeItem *> items;

	items.insert(-1, root);

	const Vector<int> &joint_porder = skeleton->get_bone_process_orders();

	Ref<Texture> bone_icon = get_icon("Skeleton", "EditorIcons");

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

void SkeletonEditor::update_editors() {
}

void SkeletonEditor::create_editors() {
	set_h_size_flags(SIZE_EXPAND_FILL);
	add_constant_override("separation", 0);

	set_focus_mode(FOCUS_ALL);

	// Create Top Menu Bar
	options = memnew(MenuButton);
	SpatialEditor::get_singleton()->add_control_to_menu_panel(options);

	options->set_text(TTR("Skeleton"));
	options->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("Skeleton", "EditorIcons"));

	options->get_popup()->add_item(TTR("Create physical skeleton"), MENU_OPTION_CREATE_PHYSICAL_SKELETON);

	options->get_popup()->connect("id_pressed", this, "_on_click_option");
	options->hide();

	const Color section_color = get_color("prop_subsection", "Editor");

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
	pose_editor->set_toggle_enabled(true);
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

void SkeletonEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editors();
			update_joint_tree();
			update_editors();

			get_tree()->connect("node_removed", this, "_node_removed", Vector<Variant>(), Object::CONNECT_ONESHOT);
			joint_tree->connect("item_selected", this, "_joint_tree_selection_changed");
			joint_tree->connect("item_rmb_selected", this, "_joint_tree_rmb_select");
#ifdef TOOLS_ENABLED
			skeleton->connect("pose_updated", this, "_update_properties");
#endif // TOOLS_ENABLED

			break;
		}
	}
}

void SkeletonEditor::_node_removed(Node *p_node) {
	if (skeleton && p_node == skeleton) {
		skeleton = nullptr;
		options->hide();
	}

	_update_properties();
}

void SkeletonEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_node_removed"), &SkeletonEditor::_node_removed);
	ClassDB::bind_method(D_METHOD("_joint_tree_selection_changed"), &SkeletonEditor::_joint_tree_selection_changed);
	ClassDB::bind_method(D_METHOD("_joint_tree_rmb_select"), &SkeletonEditor::_joint_tree_rmb_select);
	ClassDB::bind_method(D_METHOD("_update_properties"), &SkeletonEditor::_update_properties);
	ClassDB::bind_method(D_METHOD("_on_click_option"), &SkeletonEditor::_on_click_option);

	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &SkeletonEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &SkeletonEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &SkeletonEditor::drop_data_fw);
	ClassDB::bind_method(D_METHOD("move_skeleton_bone"), &SkeletonEditor::move_skeleton_bone);
}

SkeletonEditor::SkeletonEditor(EditorInspectorPluginSkeleton *e_plugin, EditorNode *p_editor, Skeleton *p_skeleton) :
		editor(p_editor),
		editor_plugin(e_plugin),
		skeleton(p_skeleton) {
}

SkeletonEditor::~SkeletonEditor() {
	if (options) {
		SpatialEditor::get_singleton()->remove_control_from_menu_panel(options);
	}
}

bool EditorInspectorPluginSkeleton::can_handle(Object *p_object) {
	return Object::cast_to<Skeleton>(p_object) != nullptr;
}

void EditorInspectorPluginSkeleton::parse_begin(Object *p_object) {
	Skeleton *skeleton = Object::cast_to<Skeleton>(p_object);
	ERR_FAIL_COND(!skeleton);

	SkeletonEditor *skel_editor = memnew(SkeletonEditor(this, editor, skeleton));
	add_custom_control(skel_editor);
}

SkeletonEditorPlugin::SkeletonEditorPlugin(EditorNode *p_node) {
	editor = p_node;

	Ref<EditorInspectorPluginSkeleton> skeleton_plugin;
	skeleton_plugin.instance();
	skeleton_plugin->editor = editor;

	EditorInspector::add_inspector_plugin(skeleton_plugin);
}
