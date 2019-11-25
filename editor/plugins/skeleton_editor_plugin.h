/*************************************************************************/
/*  skeleton_editor_plugin.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef SKELETON_EDITOR_PLUGIN_H
#define SKELETON_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/skeleton.h"

class EditorInspectorPluginSkeleton;
class Joint;
class PhysicalBone;
class SkeletonEditorPlugin;
class Button;
class CheckBox;

class BoneTransformEditor : public VBoxContainer {
	GDCLASS(BoneTransformEditor, VBoxContainer);

	EditorInspectorSection *section;

	GridContainer *translation_grid;
	GridContainer *rotation_grid;
	GridContainer *scale_grid;
	GridContainer *transform_grid;

	EditorSpinSlider *translation[3];
	EditorSpinSlider *rotation[3];
	EditorSpinSlider *scale[3];
	EditorSpinSlider *transform[12];

	Rect2 background_rects[5];

	Skeleton *skeleton;
	String property;

	UndoRedo *undo_redo;

	Button *key_button;
	CheckBox *enabled_checkbox;

	bool keyable;
	bool read_only;
	bool toggle_enabled;
	bool updating;

	String label;

	void create_editors();
	void setup_spinner(EditorSpinSlider *spinner, const bool is_transform_spinner);

	void _value_changed(const double p_value, const bool p_from_transform);

	Transform compute_transform(const bool p_from_transform) const;

	void update_enabled_checkbox();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	BoneTransformEditor(Skeleton *p_skeleton);

	// Which transform target to modify
	void set_target(const String &p_prop);
	void set_label(const String &p_label) { label = p_label; }

	void _update_properties();

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

class SkeletonEditor : public VBoxContainer {

	GDCLASS(SkeletonEditor, VBoxContainer);

	friend class SkeletonEditorPlugin;

	enum Menu {
		MENU_OPTION_CREATE_PHYSICAL_SKELETON,
		MENU_OPTION_SAVE_DEFINITION
	};

	struct BoneInfo {
		PhysicalBone *physical_bone;
		Transform relative_rest; // Relative to skeleton node
		BoneInfo() :
				physical_bone(NULL) {}
	};

	EditorNode *editor;
	EditorInspectorPluginSkeleton *editor_plugin;

	Skeleton *skeleton;

	Tree *joint_tree;
	BoneTransformEditor *rest_editor;
	BoneTransformEditor *pose_editor;

	MenuButton *options;
	EditorFileDialog *file_dialog;

	VBoxContainer *read_only_box;
	Rect2 read_only_box_rect;

	UndoRedo *undo_redo;

	bool read_only;

	void _on_click_option(int p_option);
	void save_skeleton_definition();
	void _file_selected(const String &p_file);

	void update_joint_tree();
	void update_editors();

	void create_editors();

	void create_physical_skeleton();
	PhysicalBone *create_physical_bone(int bone_id, int bone_child_id, const Vector<BoneInfo> &bones_infos);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	Skeleton *get_skeleton() const { return skeleton; };

	void _joint_tree_selection_changed();
	void _joint_tree_rmb_select(const Vector2 &p_pos);

	void _update_properties();

	void _skeleton_definition_changed();

	SkeletonEditor(EditorInspectorPluginSkeleton *e_plugin, EditorNode *p_editor, Skeleton *skeleton);
	~SkeletonEditor();
};

class EditorInspectorPluginSkeleton : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginSkeleton, EditorInspectorPlugin);

	friend class SkeletonEditorPlugin;

	EditorNode *editor;

public:
	virtual bool can_handle(Object *p_object);
	virtual void parse_begin(Object *p_object);
};

class SkeletonEditorPlugin : public EditorPlugin {
	GDCLASS(SkeletonEditorPlugin, EditorPlugin);

	EditorNode *editor;

public:
	SkeletonEditorPlugin(EditorNode *p_node);

	virtual String get_name() const { return "Skeleton"; }
};

#endif // SKELETON_EDITOR_PLUGIN_H
