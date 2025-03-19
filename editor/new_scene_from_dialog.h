/**************************************************************************/
/*  new_scene_from_dialog.h                                               */
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

#pragma once

#include "editor/gui/editor_file_dialog.h"
#include "scene/gui/check_box.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel_container.h"

class NewSceneFromDialog : public ConfirmationDialog {
	GDCLASS(NewSceneFromDialog, ConfirmationDialog);

private:
	List<String> extensions;
	// ItemList *ancestor_list = nullptr;
	OptionButton *ancestor_options = nullptr;
	LineEdit *root_name_edit = nullptr;
	LineEdit *file_path_edit = nullptr;
	// VBoxContainer *sidemenu = nullptr;
	// VBoxContainer *ancestor_sidemenu = nullptr;
	EditorFileDialog *file_browser = nullptr;
	Button *path_button;

	CheckBox *reset_position_cb;
	CheckBox *reset_rotation_cb;
	CheckBox *reset_scale_cb;
	CheckBox *remove_script_cb;

	AcceptDialog *accept = nullptr;

	Node *selected_node;

	void _browse_file();
	void _file_selected(const String &p_file);
	void _create_new_node();
	void _set_node_owner_recursive(Node *p_node, Node *p_owner, const HashMap<const Node *, Node *> &p_inverse_duplimap);
	virtual void ok_pressed() override;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	NewSceneFromDialog();
	void config(Node *p_selected_node);
	Ref<SceneState> get_selected_scene_state() const;
	String get_new_node_name() const;
};
