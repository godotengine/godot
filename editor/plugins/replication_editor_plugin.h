/*************************************************************************/
/*  replication_editor_plugin.h                                          */
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

#ifndef REPLICATION_EDITOR_PLUGIN_H
#define REPLICATION_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "scene/resources/scene_replication_config.h"

class ConfirmationDialog;
class MultiplayerSynchronizer;
class Tree;

class ReplicationEditor : public VBoxContainer {
	GDCLASS(ReplicationEditor, VBoxContainer);

private:
	MultiplayerSynchronizer *current = nullptr;

	AcceptDialog *error_dialog = nullptr;
	ConfirmationDialog *delete_dialog = nullptr;
	Button *add_button = nullptr;
	LineEdit *np_line_edit = nullptr;

	Ref<SceneReplicationConfig> config;
	NodePath deleting;
	Tree *tree;
	bool keying = false;

	Ref<Texture2D> _get_class_icon(const Node *p_node);

	void _add_pressed();
	void _tree_item_edited();
	void _tree_button_pressed(Object *p_item, int p_column, int p_id);
	void _update_checked(const NodePath &p_prop, int p_column, bool p_checked);
	void _update_config();
	void _dialog_closed(bool p_confirmed);
	void _add_property(const NodePath &p_property, bool p_spawn = true, bool p_sync = true);

protected:
	static void _bind_methods();

	void _notification(int p_what);

public:
	void update_keying();
	void edit(MultiplayerSynchronizer *p_object);
	bool has_keying() const { return keying; }
	MultiplayerSynchronizer *get_current() const { return current; }
	void property_keyed(const String &p_property);

	ReplicationEditor();
	~ReplicationEditor() {}
};

class ReplicationEditorPlugin : public EditorPlugin {
	GDCLASS(ReplicationEditorPlugin, EditorPlugin);

private:
	ReplicationEditor *repl_editor;

	void _node_removed(Node *p_node);
	void _keying_changed();
	void _property_keyed(const String &p_keyed, const Variant &p_value, bool p_advance);

protected:
	void _notification(int p_what);

public:
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	ReplicationEditorPlugin();
	~ReplicationEditorPlugin();
};

#endif // REPLICATION_EDITOR_PLUGIN_H
