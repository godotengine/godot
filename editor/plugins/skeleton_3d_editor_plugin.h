/*************************************************************************/
/*  skeleton_3d_editor_plugin.h                                          */
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

#ifndef SKELETON_3D_EDITOR_PLUGIN_H
#define SKELETON_3D_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "node_3d_editor_plugin.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/skeleton_3d.h"

class EditorInspectorPluginSkeleton;
class Joint;
class PhysicalBone3D;
class Skeleton3DEditorPlugin;
class Button;
class CheckBox;
class EditorPropertyTransform3D;
class EditorPropertyVector3;

class BoneTransformEditor : public VBoxContainer {
	GDCLASS(BoneTransformEditor, VBoxContainer);

	EditorInspectorSection *section = nullptr;

	EditorPropertyVector3 *translation_property = nullptr;
	EditorPropertyVector3 *rotation_property = nullptr;
	EditorPropertyVector3 *scale_property = nullptr;
	EditorInspectorSection *transform_section = nullptr;
	EditorPropertyTransform3D *transform_property = nullptr;

	Rect2 background_rects[5];

	Skeleton3D *skeleton;
	String property;

	UndoRedo *undo_redo;

	Button *key_button = nullptr;
	CheckBox *enabled_checkbox = nullptr;

	bool keyable = false;
	bool toggle_enabled = false;
	bool updating = false;

	String label;

	void create_editors();

	// Called when one of the EditorSpinSliders are changed.
	void _value_changed(const double p_value);
	// Called when the one of the EditorPropertyVector3 are updated.
	void _value_changed_vector3(const String p_property_name, const Vector3 p_vector, const StringName p_edited_property_name, const bool p_boolean);
	// Called when the transform_property is updated.
	void _value_changed_transform(const String p_property_name, const Transform3D p_transform, const StringName p_edited_property_name, const bool p_boolean);
	// Changes the transform to the given transform and updates the UI accordingly.
	void _change_transform(Transform3D p_new_transform);
	// Creates a Transform using the EditorPropertyVector3 properties.
	Transform3D compute_transform_from_vector3s() const;

	void update_enabled_checkbox();

protected:
	void _notification(int p_what);

public:
	BoneTransformEditor(Skeleton3D *p_skeleton);

	// Which transform target to modify
	void set_target(const String &p_prop);
	void set_label(const String &p_label) { label = p_label; }

	void _update_properties();
	void _update_custom_pose_properties();
	void _update_transform_properties(Transform3D p_transform);

	// Can/cannot modify the spinner values for the Transform
	void set_read_only(const bool p_read_only);

	// Transform can be keyed, whether or not to show the button
	void set_keyable(const bool p_keyable);

	// Bone can be toggled enabled or disabled, whether or not to show the checkbox
	void set_toggle_enabled(const bool p_enabled);

	// Key Transform Button pressed
	void _key_button_pressed();

	// Bone Enabled Checkbox toggled
	void _checkbox_toggled(const bool p_toggled);
};

class Skeleton3DEditor : public VBoxContainer {
	GDCLASS(Skeleton3DEditor, VBoxContainer);

	friend class Skeleton3DEditorPlugin;

	enum Menu {
		MENU_OPTION_INIT_POSE,
		MENU_OPTION_INSERT_KEYS,
		MENU_OPTION_INSERT_KEYS_EXISTED,
		MENU_OPTION_POSE_TO_REST,
		MENU_OPTION_CREATE_PHYSICAL_SKELETON
	};

	enum ToolMode {
		TOOL_MODE_BONE_SELECT,
		TOOL_MODE_BONE_MOVE,
		TOOL_MODE_BONE_ROTATE,
		TOOL_MODE_BONE_SCALE,
		TOOL_MODE_BONE_NONE,
		TOOL_MODE_BONE_MAX
	};

	enum MenuToolOption {
		MENU_TOOL_BONE_SELECT,
		MENU_TOOL_BONE_MOVE,
		MENU_TOOL_BONE_ROTATE,
		MENU_TOOL_BONE_SCALE,
		MENU_TOOL_BONE_NONE,
		MENU_TOOL_BONE_MAX
	};

	struct BoneInfo {
		PhysicalBone3D *physical_bone = nullptr;
		Transform3D relative_rest; // Relative to skeleton node
	};

	EditorNode *editor;
	EditorInspectorPluginSkeleton *editor_plugin;

	Skeleton3D *skeleton;

	Tree *joint_tree = nullptr;
	BoneTransformEditor *rest_editor = nullptr;
	BoneTransformEditor *pose_editor = nullptr;
	BoneTransformEditor *custom_pose_editor = nullptr;

	VSeparator *separators[2];
	MenuButton *options = nullptr;
	Button *tool_button[TOOL_MODE_BONE_MAX];
	Button *rest_mode_button;

	ToolMode tool_mode = TOOL_MODE_BONE_NONE;
	bool rest_mode = false;

	EditorFileDialog *file_dialog = nullptr;

	UndoRedo *undo_redo = nullptr;

	bool keyable;

	void _on_click_option(int p_option);
	void _file_selected(const String &p_file);
	void _menu_tool_item_pressed(int p_option);
	TreeItem *_find(TreeItem *p_node, const NodePath &p_path);
	void rest_mode_toggled(const bool pressed);

	EditorFileDialog *file_export_lib = nullptr;

	void update_joint_tree();
	void update_editors();

	void create_editors();

	void init_pose();
	void insert_keys(bool p_all_bones);
	void pose_to_rest();

	void create_physical_skeleton();
	PhysicalBone3D *create_physical_bone(int bone_id, int bone_child_id, const Vector<BoneInfo> &bones_infos);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	Ref<ShaderMaterial> handle_material;
	Ref<Shader> handle_shader;
	MeshInstance3D *pointsm;
	Ref<ArrayMesh> am;
	void _hide_handles();
	void _draw_handles();

	Node3DEditorViewport::EditData _edit;
	void _compute_edit(int p_index, const Point2 &p_point);
	bool _gizmo_select(int p_index, const Vector2 &p_screenpos, bool p_highlight_only = false);

	Transform3D original_local;
	Transform3D original_global;
	Transform3D original_to_local;

	void _update_sub_gizmo();

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	virtual bool forward_spatial_gui_input(int p_index, Camera3D *p_camera, const Ref<InputEvent> &p_event);
	void move_skeleton_bone(NodePath p_skeleton_path, int32_t p_selected_boneidx, int32_t p_target_boneidx);

	void set_keyable(const bool p_keyable);

	Skeleton3D *get_skeleton() const { return skeleton; };

	void set_rest_mode_toggled(const bool pressed, const bool destructing = false);

	void _joint_tree_selection_changed();
	void _joint_tree_rmb_select(const Vector2 &p_pos);

	void _update_properties();

	Skeleton3DEditor(EditorInspectorPluginSkeleton *e_plugin, EditorNode *p_editor, Skeleton3D *skeleton);
	~Skeleton3DEditor();
};

class EditorInspectorPluginSkeleton : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginSkeleton, EditorInspectorPlugin);

	friend class Skeleton3DEditorPlugin;

	Skeleton3DEditor *skel_editor;
	EditorNode *editor;
	UndoRedo *undo_redo;

	void set_rest_mode_toggled(const bool p_pressed);

protected:
	static void _bind_methods();

public:
	virtual bool forward_spatial_gui_input(int p_index, Camera3D *p_camera, const Ref<InputEvent> &p_event) { return skel_editor->forward_spatial_gui_input(p_index, p_camera, p_event); }
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
	UndoRedo *get_undo_redo() { return undo_redo; }
};

class Skeleton3DEditorPlugin : public EditorPlugin {
	GDCLASS(Skeleton3DEditorPlugin, EditorPlugin);

	EditorInspectorPluginSkeleton *skeleton_plugin;
	EditorNode *editor;

public:
	virtual bool forward_spatial_gui_input(int p_index, Camera3D *p_camera, const Ref<InputEvent> &p_event) override {
		if (Node3DEditor::get_singleton()->get_tool_mode() != Node3DEditor::TOOL_MODE_EXTERNAL) {
			return false;
		}
		return skeleton_plugin->forward_spatial_gui_input(p_index, p_camera, p_event);
	}
	bool has_main_screen() const override { return false; }
	virtual bool handles(Object *p_object) const override;

	virtual String get_name() const override { return "Skeleton3D"; }

	Skeleton3DEditorPlugin(EditorNode *p_node);
};

#endif // SKELETON_3D_EDITOR_PLUGIN_H
