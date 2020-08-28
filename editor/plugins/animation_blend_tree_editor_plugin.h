/*************************************************************************/
/*  animation_blend_tree_editor_plugin.h                                 */
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

#ifndef ANIMATION_BLEND_TREE_EDITOR_PLUGIN_H
#define ANIMATION_BLEND_TREE_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/plugins/animation_tree_editor_plugin.h"
#include "editor/property_editor.h"
#include "scene/animation/animation_blend_tree.h"
#include "scene/gui/button.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/popup.h"
#include "scene/gui/tree.h"

class ProgressBar;

class AnimationNodeBlendTreeEditor : public AnimationTreeNodeEditorPlugin {
	GDCLASS(AnimationNodeBlendTreeEditor, AnimationTreeNodeEditorPlugin);

	Ref<AnimationNodeBlendTree> blend_tree;
	GraphEdit *graph;
	MenuButton *add_node;
	Vector2 popup_menu_position;
	bool use_popup_menu_position;

	PanelContainer *error_panel;
	Label *error_label;

	UndoRedo *undo_redo;

	AcceptDialog *filter_dialog;
	Tree *filters;
	CheckBox *filter_enabled;

	Map<StringName, ProgressBar *> animations;
	Vector<EditorProperty *> visible_properties;

	void _update_graph();

	struct AddOption {
		String name;
		String type;
		Ref<Script> script;
		AddOption(const String &p_name = String(), const String &p_type = String()) :
				name(p_name),
				type(p_type) {
		}
	};

	Vector<AddOption> add_options;

	void _add_node(int p_idx);
	void _update_options_menu();

	static AnimationNodeBlendTreeEditor *singleton;

	void _node_dragged(const Vector2 &p_from, const Vector2 &p_to, const StringName &p_which);
	void _node_renamed(const String &p_text, Ref<AnimationNode> p_node);
	void _node_renamed_focus_out(Node *le, Ref<AnimationNode> p_node);

	bool updating;

	void _connection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index);
	void _disconnection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index);

	void _scroll_changed(const Vector2 &p_scroll);
	void _node_selected(Object *p_node);
	void _open_in_editor(const String &p_which);
	void _anim_selected(int p_index, Array p_options, const String &p_node);
	void _delete_request(const String &p_which);
	void _delete_nodes_request();
	void _popup_request(const Vector2 &p_position);

	bool _update_filters(const Ref<AnimationNode> &anode);
	void _edit_filters(const String &p_which);
	void _filter_edited();
	void _filter_toggled();
	Ref<AnimationNode> _filter_edit;

	void _property_changed(const StringName &p_property, const Variant &p_value, const String &p_field, bool p_changing);
	void _removed_from_graph();

	EditorFileDialog *open_file;
	Ref<AnimationNode> file_loaded;
	void _file_opened(const String &p_file);

	enum {
		MENU_LOAD_FILE = 1000,
		MENU_PASTE = 1001,
		MENU_LOAD_FILE_CONFIRM = 1002
	};

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static AnimationNodeBlendTreeEditor *get_singleton() { return singleton; }

	void add_custom_type(const String &p_name, const Ref<Script> &p_script);
	void remove_custom_type(const Ref<Script> &p_script);

	virtual Size2 get_minimum_size() const override;

	virtual bool can_edit(const Ref<AnimationNode> &p_node) override;
	virtual void edit(const Ref<AnimationNode> &p_node) override;

	AnimationNodeBlendTreeEditor();
};

#endif // ANIMATION_BLEND_TREE_EDITOR_PLUGIN_H
