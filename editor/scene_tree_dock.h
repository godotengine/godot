/*************************************************************************/
/*  scene_tree_dock.h                                                    */
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

#ifndef SCENE_TREE_DOCK_H
#define SCENE_TREE_DOCK_H

#include "editor/connections_dialog.h"
#include "editor/create_dialog.h"
#include "editor/editor_data.h"
#include "editor/editor_sub_scene.h"
#include "editor/groups_editor.h"
#include "editor/quick_open.h"
#include "editor/rename_dialog.h"
#include "editor/reparent_dialog.h"
#include "editor/script_create_dialog.h"
#include "scene/animation/animation_player.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "scene/gui/label.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/tree.h"
#include "scene_tree_editor.h"

class EditorNode;

class SceneTreeDock : public VBoxContainer {
	GDCLASS(SceneTreeDock, VBoxContainer);

	enum Tool {

		TOOL_NEW,
		TOOL_INSTANCE,
		TOOL_EXPAND_COLLAPSE,
		TOOL_RENAME,
		TOOL_BATCH_RENAME,
		TOOL_REPLACE,
		TOOL_EXTEND_SCRIPT,
		TOOL_ATTACH_SCRIPT,
		TOOL_DETACH_SCRIPT,
		TOOL_MOVE_UP,
		TOOL_MOVE_DOWN,
		TOOL_DUPLICATE,
		TOOL_REPARENT,
		TOOL_REPARENT_TO_NEW_NODE,
		TOOL_MAKE_ROOT,
		TOOL_NEW_SCENE_FROM,
		TOOL_MERGE_FROM_SCENE,
		TOOL_MULTI_EDIT,
		TOOL_ERASE,
		TOOL_COPY_NODE_PATH,
		TOOL_BUTTON_MAX,
		TOOL_OPEN_DOCUMENTATION,
		TOOL_SCENE_EDITABLE_CHILDREN,
		TOOL_SCENE_USE_PLACEHOLDER,
		TOOL_SCENE_MAKE_LOCAL,
		TOOL_SCENE_OPEN,
		TOOL_SCENE_CLEAR_INHERITANCE,
		TOOL_SCENE_CLEAR_INHERITANCE_CONFIRM,
		TOOL_SCENE_OPEN_INHERITED,

		TOOL_CREATE_2D_SCENE,
		TOOL_CREATE_3D_SCENE,
		TOOL_CREATE_USER_INTERFACE,
		TOOL_CREATE_FAVORITE,

	};

	enum {
		EDIT_SUBRESOURCE_BASE = 100
	};

	Vector<ObjectID> subresources;

	bool restore_script_editor_on_drag;

	int current_option;
	CreateDialog *create_dialog;
	RenameDialog *rename_dialog;

	Button *button_add;
	Button *button_instance;
	Button *button_create_script;
	Button *button_detach_script;

	Button *button_2d;
	Button *button_3d;
	Button *button_ui;
	Button *button_custom;

	HBoxContainer *button_hb;
	Button *edit_local, *edit_remote;
	SceneTreeEditor *scene_tree;
	Control *remote_tree;

	HBoxContainer *tool_hbc;
	void _tool_selected(int p_tool, bool p_confirm_override = false);
	void _node_collapsed(Object *p_obj);

	EditorData *editor_data;
	EditorSelection *editor_selection;

	ScriptCreateDialog *script_create_dialog;
	AcceptDialog *accept;
	ConfirmationDialog *delete_dialog;
	ConfirmationDialog *editable_instance_remove_dialog;
	ConfirmationDialog *placeholder_editable_instance_remove_dialog;

	ReparentDialog *reparent_dialog;
	EditorQuickOpen *quick_open;
	EditorSubScene *import_subscene_dialog;
	EditorFileDialog *new_scene_from_dialog;

	LineEdit *filter;
	TextureRect *filter_icon;

	PopupMenu *menu;
	PopupMenu *menu_subresources;
	ConfirmationDialog *clear_inherit_confirm;

	bool first_enter;

	void _create();
	void _do_create(Node *p_parent);
	Node *scene_root;
	Node *edited_scene;
	EditorNode *editor;

	VBoxContainer *create_root_dialog;
	String selected_favorite_root;

	void _add_children_to_popup(Object *p_obj, int p_depth);

	void _node_reparent(NodePath p_path, bool p_keep_global_xform);
	void _do_reparent(Node *p_new_parent, int p_position_in_parent, Vector<Node *> p_nodes, bool p_keep_global_xform);

	bool _is_collapsed_recursive(TreeItem *p_item) const;
	void _set_collapsed_recursive(TreeItem *p_item, bool p_collapsed);

	void _set_owners(Node *p_owner, const Array &p_nodes);

	enum ReplaceOwnerMode {
		MODE_BIDI,
		MODE_DO,
		MODE_UNDO
	};

	void _node_replace_owner(Node *p_base, Node *p_node, Node *p_root, ReplaceOwnerMode p_mode = MODE_BIDI);
	void _load_request(const String &p_path);
	void _script_open_request(const Ref<Script> &p_script);

	bool _cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node);
	bool _track_inherit(const String &p_target_scene_path, Node *p_desired_node);

	void _node_selected();
	void _node_renamed();
	void _script_created(Ref<Script> p_script);
	void _script_creation_closed();

	void _delete_confirm();

	void _toggle_editable_children_from_selection();
	void _toggle_editable_children(Node *p_node);

	void _toggle_placeholder_from_selection();

	void _node_prerenamed(Node *p_node, const String &p_new_name);

	void _nodes_drag_begin();
	void _input(Ref<InputEvent> p_event);
	void _unhandled_key_input(Ref<InputEvent> p_event);

	void _import_subscene();

	void _new_scene_from(String p_file);

	bool _validate_no_foreign();
	void _selection_changed();
	void _update_script_button();
	Node *_get_selection_group_tail(Node *p_node, List<Node *> p_list);

	void _fill_path_renames(Vector<StringName> base_path, Vector<StringName> new_base_path, Node *p_node, List<Pair<NodePath, NodePath>> *p_renames);

	void _normalize_drop(Node *&to_node, int &to_pos, int p_type);

	void _nodes_dragged(Array p_nodes, NodePath p_to, int p_type);
	void _files_dropped(Vector<String> p_files, NodePath p_to, int p_type);
	void _script_dropped(String p_file, NodePath p_to);
	void _quick_open();

	void _tree_rmb(const Vector2 &p_menu_pos);

	void _filter_changed(const String &p_filter);

	void _perform_instance_scenes(const Vector<String> &p_files, Node *parent, int p_pos);
	void _replace_with_branch_scene(const String &p_file, Node *base);

	void _file_selected(String p_file);

	void _remote_tree_selected();
	void _local_tree_selected();

	void _update_create_root_dialog();
	void _favorite_root_selected(const String &p_class);

	void _feature_profile_changed();

	bool profile_allow_editing;
	bool profile_allow_script_editing;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	String get_filter();
	void set_filter(const String &p_filter);

	void _focus_node();

	void import_subscene();
	void set_edited_scene(Node *p_scene);
	void instance(const String &p_file);
	void instance_scenes(const Vector<String> &p_files, Node *p_parent = nullptr);
	void set_selected(Node *p_node, bool p_emit_selected = false);
	void fill_path_renames(Node *p_node, Node *p_new_parent, List<Pair<NodePath, NodePath>> *p_renames);
	void perform_node_renames(Node *p_base, List<Pair<NodePath, NodePath>> *p_renames, Map<Ref<Animation>, Set<int>> *r_rem_anims = nullptr);
	SceneTreeEditor *get_tree_editor() { return scene_tree; }
	EditorData *get_editor_data() { return editor_data; }

	void add_remote_tree_editor(Control *p_remote);
	void show_remote_tree();
	void hide_remote_tree();
	void show_tab_buttons();
	void hide_tab_buttons();

	void replace_node(Node *p_node, Node *p_by_node, bool p_keep_properties = true, bool p_remove_old = true);

	void attach_script_to_selected(bool p_extend);
	void open_script_dialog(Node *p_for_node, bool p_extend);

	ScriptCreateDialog *get_script_create_dialog() { return script_create_dialog; }

	SceneTreeDock(EditorNode *p_editor, Node *p_scene_root, EditorSelection *p_editor_selection, EditorData &p_editor_data);
};

#endif // SCENE_TREE_DOCK_H
