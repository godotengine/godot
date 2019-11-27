/*************************************************************************/
/*  skeleton_3d_editor_plugin.cpp                                        */
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

#include "skeleton_3d_editor_plugin.h"

#include "core/io/resource_saver.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_properties.h"
#include "editor/editor_scale.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/physics_bone_3d.h"
#include "scene/3d/physics_bone_joint_3d.h"
#include "scene/resources/capsule_shape_3d.h"
#include "scene/resources/sphere_shape_3d.h"

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
	transform_property = memnew(EditorPropertyTransform());
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

			Point2 buffer;
			buffer.x += get_theme_constant("inspector_margin", "Editor");
			buffer.y += font->get_height();
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
	if (updating)
		return;

	Transform tform = compute_transform_from_vector3s();
	_change_transform(tform);
}

void BoneTransformEditor::_value_changed_vector3(const String p_property_name, const Vector3 p_vector, const StringName p_edited_property_name, const bool p_boolean) {
	if (updating)
		return;
	Transform tform = compute_transform_from_vector3s();
	_change_transform(tform);
}

Transform BoneTransformEditor::compute_transform_from_vector3s() const {
	// Convert rotation from degrees to radians.
	Vector3 prop_rotation = rotation_property->get_vector();
	prop_rotation.x = Math::deg2rad(prop_rotation.x);
	prop_rotation.y = Math::deg2rad(prop_rotation.y);
	prop_rotation.z = Math::deg2rad(prop_rotation.z);

	return Transform(
			Basis(prop_rotation, scale_property->get_vector()),
			translation_property->get_vector());
}

void BoneTransformEditor::_value_changed_transform(const String p_property_name, const Transform p_transform, const StringName p_edited_property_name, const bool p_boolean) {
	if (updating)
		return;
	_change_transform(p_transform);
}

void BoneTransformEditor::_change_transform(Transform p_new_transform) {
	if (property.get_slicec('/', 0) == "bones") {
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
	if (updating)
		return;

	if (skeleton == nullptr)
		return;

	updating = true;

	Transform tform = skeleton->get(property);
	_update_transform_properties(tform);
}

void BoneTransformEditor::_update_transform_properties(Transform tform) {
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
	_update_properties();
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
	Transform tform = compute_transform_from_vector3s();
	tform.orthonormalize();
	AnimationPlayerEditor::singleton->get_track_editor()->insert_transform_key(skeleton, name, tform);
}

void BoneTransformEditor::_checkbox_toggled(const bool p_toggled) {
	if (enabled_checkbox) {
		const String path = "bones/" + property.get_slicec('/', 1) + "/enabled";
		skeleton->set(path, p_toggled);
	}
}

void Skeleton3DEditor::_on_click_option(int p_option) {
	if (!skeleton) {
		return;
	}

	switch (p_option) {
		case Menu::RESET_POSE: {
			skeleton->reset_bone_poses();
			break;
		}
	}
}

static void set_owner_recurisve(Node *p_node, Node *p_owner) {
	p_node->set_owner(p_owner);
	for (int i = 0; i < p_node->get_child_count(); ++i) {
		set_owner_recurisve(p_node->get_child(i), p_owner);
	}
}

void Skeleton3DEditor::create_single_bone(BoneId p_bone_id) const {
	PhysicsBone3D *new_bone = create_physics_bone(p_bone_id);

	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Create Physics Bone"));
	ur->add_do_method(skeleton, "add_child", new_bone);
	ur->add_do_reference(new_bone);
	ur->add_undo_method(skeleton, "remove_child", new_bone);
	ur->commit_action();

	Node *owner = skeleton == get_tree()->get_edited_scene_root() ? skeleton : skeleton->get_owner();
	set_owner_recurisve(new_bone, owner);
}

void Skeleton3DEditor::create_physics_bones_for_tree(const JointTree &p_joint_tree, const bool p_create_joints) const {
	using BoneMap = Map<BoneId, PhysicsBone3D *>;

	BoneMap bones;

	for (int i = 0; i < p_joint_tree.joints.size(); ++i) {
		const JointTree::Node &node = p_joint_tree.joints[i];

		BoneId bone_id = node.bone_id;
		BoneId parent_id = node.parent_id;

		// if this node is a parent node, it doesnt make a bone itself
		if (node.parent_id == -1) {
			continue;
		}

		// we create a bone between the PARENT and the child, since the parent is at the top of the tree going down to the child
		bones.insert(bone_id, create_physics_bone(parent_id, bone_id));
	}

	Vector<PhysicsBone3D *> roots;

	for (BoneMap::Element *el = bones.front(); el != nullptr; el = el->next()) {
		BoneId bone_id = el->key();

		PhysicsBone3D *child = el->value();

		const JointTree::Node &node = p_joint_tree.get(bone_id);

		BoneMap::Element *parent_el = bones.find(node.parent_id);
		// if we cant find the parent, we are a root node in the joint tree, add it to the skeleton and continue...
		if (parent_el == nullptr) {
			roots.push_back(child);
			continue;
		}

		PhysicsBone3D *parent = parent_el->value();

		// parent the child node to the parent
		parent->add_child(child);

		// make a physics joint node
		if (p_create_joints) {
			BoneConeTwistJoint3D *joint = memnew(BoneConeTwistJoint3D);
			const NodePath a("../../../" + (String)parent->get_name());
			const NodePath b("../../" + (String)child->get_name());
			joint->set_node_a(a);
			joint->set_node_b(b);
			child->add_child(joint);
		}
	}

	Node *owner = skeleton == get_tree()->get_edited_scene_root() ? skeleton : skeleton->get_owner();

	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

	for (int i = 0; i < roots.size(); ++i) {
		ur->create_action(TTR("Create Physics Bone Chain"));
		ur->add_do_method(skeleton, "add_child", roots[i]);
		ur->add_do_reference(roots[i]);
		ur->add_undo_method(skeleton, "remove_child", roots[i]);
		ur->commit_action();

		set_owner_recurisve(roots[i], owner);
	}
}

PhysicsBone3D *Skeleton3DEditor::create_physics_bone(BoneId p_bone_id) const {
	SphereShape3D *bone_shape_sphere = memnew(SphereShape3D);

	bone_shape_sphere->set_radius(0.25);

	CollisionShape3D *bone_shape = memnew(CollisionShape3D);
	bone_shape->set_shape(bone_shape_sphere);

	const String bone_name = skeleton->get_bone_name(p_bone_id);

	PhysicsBone3D *physics_bone = memnew(PhysicsBone3D);
	physics_bone->add_child(bone_shape);
	physics_bone->set_name("PhysicsBone_" + bone_name);
	physics_bone->set_bone_name(bone_name);
	physics_bone->set_bone_offset(Vector3());
	return physics_bone;
}

PhysicsBone3D *Skeleton3DEditor::create_physics_bone(BoneId p_target_bone_id, BoneId p_other_bone_id) const {
	const Transform rest = skeleton->get_bone_global_rest(p_target_bone_id);
	const Transform other_rest = skeleton->get_bone_global_rest(p_other_bone_id);

	const real_t half_height = ((rest.origin - other_rest.origin).length() / 2);
	const real_t radius = half_height * 0.34;

	CapsuleShape3D *bone_shape_capsule = memnew(CapsuleShape3D);
	// remember capsulse is the radius of the sphere + cylinder half height
	bone_shape_capsule->set_height((half_height - radius) * 2);
	bone_shape_capsule->set_radius(radius);

	CollisionShape3D *bone_shape = memnew(CollisionShape3D);
	bone_shape->set_shape(bone_shape_capsule);

	Vector3 bone_offset(0, half_height, 0);

	const String bone_name = skeleton->get_bone_name(p_target_bone_id);

	PhysicsBone3D *physics_bone = memnew(PhysicsBone3D);
	physics_bone->add_child(bone_shape);
	physics_bone->set_name("PhysicsBone_" + bone_name);
	physics_bone->set_bone_name(bone_name);
	physics_bone->set_bone_offset(bone_offset);
	return physics_bone;
}

Variant Skeleton3DEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
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

bool Skeleton3DEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
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

void Skeleton3DEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from))
		return;

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

void Skeleton3DEditor::_joint_tree_selection_changed() {
	TreeItem *selected = joint_tree->get_selected();
	const String path = selected->get_metadata(0);

	if (path.begins_with("bones/")) {
		const int b_idx = path.get_slicec('/', 1).to_int();
		const String bone_path = "bones/" + itos(b_idx) + "/";

		pose_editor->set_target(bone_path + "pose");
		rest_editor->set_target(bone_path + "rest");

		pose_editor->set_visible(true);
		rest_editor->set_visible(true);
	}
}

bool Skeleton3DEditor::construct_joint_tree(JointTree &p_out_tree, const Vector<BoneId> &p_bone_ids) const {
	int found_count = 0;

	const auto &increment_and_find_fn = [&](const BoneId p_bone_id) -> JointTree::Node * {
		if (p_out_tree.has(p_bone_id)) {
			return &p_out_tree.get(p_bone_id);
		}

		++found_count;
		p_out_tree.insert(p_bone_id);
		return &p_out_tree.get(p_bone_id);
	};

	for (int i = 0; i < p_bone_ids.size(); ++i) {
		const BoneId bone_id = p_bone_ids[i];

		// node added as a parent?
		BoneId parent_bone_id = bone_id;
		do {
			parent_bone_id = skeleton->get_bone_parent(parent_bone_id);
			if (p_bone_ids.find(parent_bone_id) != -1) {
				break;
			}
		} while (parent_bone_id != -1);

		// found parent in passed in vector?
		if (parent_bone_id != -1) {
			JointTree::Node *node = increment_and_find_fn(bone_id);
			node->parent_id = parent_bone_id;

			// mark the parent
			JointTree::Node *parent = increment_and_find_fn(parent_bone_id);
			parent->children.push_back(bone_id);
		}
	}

	return found_count == p_bone_ids.size();
}

void Skeleton3DEditor::_joint_tree_rmb_select(const Vector2 &p_pos) {
	joint_tree_popup->clear();
	joint_tree_popup->set_position(get_screen_position() + p_pos);

	Vector<BoneId> selected_bone_ids = joint_tree_get_selected_bones();

	int bone_count = selected_bone_ids.size();

	bool items_added = false;

	JointTree joint_tree;
	if (bone_count > 0 && construct_joint_tree(joint_tree, selected_bone_ids)) {
		joint_tree_popup->add_icon_item(get_theme_icon("PhysicalBone3D", "EditorIcons"), TTR("Create Physics Bones"), JointTreePopupMenu::CREATE_PHYSICS_BONES);
		joint_tree_popup->add_icon_item(get_theme_icon("PhysicalBone3D", "EditorIcons"), TTR("Create Physics Bones With Joints"), JointTreePopupMenu::CREATE_PHYSICS_BONES_WITH_JOINTS);
		items_added = true;
	}

	if (bone_count == 1) {
		joint_tree_popup->add_icon_item(get_theme_icon("PhysicalBone3D", "EditorIcons"), TTR("Create Physics Bone"), JointTreePopupMenu::CREATE_PHYSICS_BONE);
		items_added = true;
	}

	if (items_added) {
		joint_tree_popup->popup();
	}
}

void Skeleton3DEditor::_update_properties() {
	if (rest_editor) {
		rest_editor->_update_properties();
	}
	if (pose_editor) {
		pose_editor->_update_properties();
	}
}

void Skeleton3DEditor::update_joint_tree() {
	joint_tree->clear();

	if (skeleton == nullptr)
		return;

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

void Skeleton3DEditor::_reset_pose() {
	if (!skeleton)
		return;

	skeleton->reset_bone_poses();
	_update_properties();
}

Vector<BoneId> Skeleton3DEditor::joint_tree_get_selected_bones() const {
	Vector<BoneId> selected_bones;

	TreeItem *selected = nullptr;
	do {
		selected = joint_tree->get_next_selected(selected);
		if (!selected) {
			break;
		}

		String node_name = selected->get_metadata(0);
		if (!node_name.begins_with("bones/")) {
			continue;
		}

		const BoneId bone_idx = node_name.get_slicec('/', 1).to_int();
		selected_bones.push_back(bone_idx);
	} while (selected);

	return selected_bones;
}

void Skeleton3DEditor::_joint_tree_popup_selected(int p_option) {
	bool with_joints = false;

	switch (p_option) {
		case JointTreePopupMenu::CREATE_PHYSICS_BONE: {
			Vector<BoneId> selected_bones = joint_tree_get_selected_bones();
			create_single_bone(selected_bones[0]);
			break;
		}
		case JointTreePopupMenu::CREATE_PHYSICS_BONES_WITH_JOINTS: {
			with_joints = true;
			[[fallthrough]];
		}
		case JointTreePopupMenu::CREATE_PHYSICS_BONES: {
			Vector<BoneId> selected_bones = joint_tree_get_selected_bones();
			JointTree tree;
			if (construct_joint_tree(tree, selected_bones)) {
				create_physics_bones_for_tree(tree, with_joints);
			}
			break;
		}
	}
}

void Skeleton3DEditor::create_editors() {
	set_h_size_flags(SIZE_EXPAND_FILL);
	add_theme_constant_override("separation", 0);

	set_focus_mode(FOCUS_ALL);

	// Create Top Menu Bar
	options = memnew(MenuButton);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(options);

	options->set_text(TTR("Skeleton3D"));
	options->set_icon(EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Skeleton3D", "EditorIcons"));

	options->get_popup()->add_item(TTR("Reset Pose"), Menu::RESET_POSE);

	options->get_popup()->connect("id_pressed", callable_mp(this, &Skeleton3DEditor::_on_click_option));

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
	joint_tree->set_select_mode(Tree::SELECT_MULTI);
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

	joint_tree_popup = memnew(PopupMenu());
	joint_tree_popup->connect("id_pressed", callable_mp(this, &Skeleton3DEditor::_joint_tree_popup_selected));
	add_child(joint_tree_popup);
}

void Skeleton3DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editors();
			update_joint_tree();

			get_tree()->connect("node_removed", callable_mp(this, &Skeleton3DEditor::_node_removed), Vector<Variant>(), Object::CONNECT_ONESHOT);
			joint_tree->connect("cell_selected", callable_mp(this, &Skeleton3DEditor::_joint_tree_selection_changed));
			joint_tree->connect("item_rmb_selected", callable_mp(this, &Skeleton3DEditor::_joint_tree_rmb_select));
			skeleton->connect("pose_updated", callable_mp(this, &Skeleton3DEditor::_update_properties));

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

	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &Skeleton3DEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &Skeleton3DEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &Skeleton3DEditor::drop_data_fw);
	ClassDB::bind_method(D_METHOD("move_skeleton_bone"), &Skeleton3DEditor::move_skeleton_bone);
}

Skeleton3DEditor::Skeleton3DEditor(EditorInspectorPluginSkeleton *e_plugin, EditorNode *p_editor, Skeleton3D *p_skeleton) :
		editor(p_editor),
		editor_plugin(e_plugin),
		skeleton(p_skeleton) {
}

Skeleton3DEditor::~Skeleton3DEditor() {
	if (options) {
		Node3DEditor::get_singleton()->remove_control_from_menu_panel(options);
	}
}

bool EditorInspectorPluginSkeleton::can_handle(Object *p_object) {
	return Object::cast_to<Skeleton3D>(p_object) != nullptr;
}

void EditorInspectorPluginSkeleton::parse_begin(Object *p_object) {
	Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(p_object);
	ERR_FAIL_COND(!skeleton);

	Skeleton3DEditor *skel_editor = memnew(Skeleton3DEditor(this, editor, skeleton));
	add_custom_control(skel_editor);
}

Skeleton3DEditorPlugin::Skeleton3DEditorPlugin(EditorNode *p_node) {
	editor = p_node;

	Ref<EditorInspectorPluginSkeleton> skeleton_plugin;
	skeleton_plugin.instance();
	skeleton_plugin->editor = editor;

	EditorInspector::add_inspector_plugin(skeleton_plugin);
}
