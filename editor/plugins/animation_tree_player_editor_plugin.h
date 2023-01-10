/**************************************************************************/
/*  animation_tree_player_editor_plugin.h                                 */
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

#ifndef ANIMATION_TREE_PLAYER_EDITOR_PLUGIN_H
#define ANIMATION_TREE_PLAYER_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/property_editor.h"
#include "scene/animation/animation_tree_player.h"
#include "scene/gui/button.h"
#include "scene/gui/popup.h"
#include "scene/gui/tree.h"

class AnimationTreePlayerEditor : public Control {
	GDCLASS(AnimationTreePlayerEditor, Control);

	static const char *_node_type_names[];

	enum ClickType {
		CLICK_NONE,
		CLICK_NAME,
		CLICK_NODE,
		CLICK_INPUT_SLOT,
		CLICK_OUTPUT_SLOT,
		CLICK_PARAMETER
	};

	enum {

		MENU_GRAPH_CLEAR = 100,
		MENU_IMPORT_ANIMATIONS = 101,
		NODE_DISCONNECT,
		NODE_RENAME,
		NODE_ERASE,
		NODE_ADD_INPUT,
		NODE_DELETE_INPUT,
		NODE_SET_AUTOADVANCE,
		NODE_CLEAR_AUTOADVANCE
	};

	bool renaming_edit;
	StringName edited_node;
	bool updating_edit;
	Popup *edit_dialog;
	HSlider *edit_scroll[2];
	LineEdit *edit_line[4];
	OptionButton *edit_option;
	Label *edit_label[4];
	Button *edit_button;
	Button *filter_button;
	CheckButton *edit_check;
	EditorFileDialog *file_dialog;
	int file_op;

	void _popup_edit_dialog();

	PopupMenu *master_anim_popup;
	PopupMenu *node_popup;
	PopupMenu *add_popup;
	HScrollBar *h_scroll;
	VScrollBar *v_scroll;
	MenuButton *add_menu;

	CustomPropertyEditor *property_editor;

	AnimationTreePlayer *anim_tree;
	List<StringName> order;
	Set<StringName> active_nodes;

	int last_x, last_y;

	Point2 offset;
	ClickType click_type;
	Point2 click_pos;
	StringName click_node;
	int click_slot;
	Point2 click_motion;
	ClickType rclick_type;
	StringName rclick_node;
	int rclick_slot;

	Button *play_button;

	Size2 _get_maximum_size();
	Size2 get_node_size(const StringName &p_node) const;
	void _draw_node(const StringName &p_node);

	AcceptDialog *filter_dialog;
	Tree *filter;

	void _draw_cos_line(const Vector2 &p_from, const Vector2 &p_to, const Color &p_color);
	void _update_scrollbars();
	void _scroll_moved(float);
	void _play_toggled();
	/*
	void _node_param_changed();
	void _node_add_callback();
	void _node_add(VisualServer::AnimationTreeNodeType p_type);
	void _node_edit_property(const StringName& p_node);
*/

	void _master_anim_menu_item(int p_item);
	void _node_menu_item(int p_item);
	void _add_menu_item(int p_item);

	void _filter_edited();
	void _find_paths_for_filter(const StringName &p_node, Set<String> &paths);
	void _edit_filters();

	void _edit_oneshot_start();
	void _edit_dialog_animation_changed();
	void _edit_dialog_edit_animation();
	void _edit_dialog_changeds(String);
	void _edit_dialog_changede(String);
	void _edit_dialog_changedf(float);
	void _edit_dialog_changed();
	ClickType _locate_click(const Point2 &p_click, StringName *p_node_id, int *p_slot_index) const;
	Point2 _get_slot_pos(const StringName &p_node_id, bool p_input, int p_slot);

	StringName _add_node(int p_item);
	void _file_dialog_selected(String p_path);

protected:
	void _notification(int p_what);
	void _gui_input(Ref<InputEvent> p_event);
	static void _bind_methods();

public:
	virtual Size2 get_minimum_size() const;
	void edit(AnimationTreePlayer *p_anim_tree);
	AnimationTreePlayerEditor();
};

class AnimationTreePlayerEditorPlugin : public EditorPlugin {
	GDCLASS(AnimationTreePlayerEditorPlugin, EditorPlugin);

	AnimationTreePlayerEditor *anim_tree_editor;
	EditorNode *editor;
	Button *button;

public:
	virtual String get_name() const { return "AnimTree"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	AnimationTreePlayerEditorPlugin(EditorNode *p_node);
	~AnimationTreePlayerEditorPlugin();
};

#endif // ANIMATION_TREE_PLAYER_EDITOR_PLUGIN_H
