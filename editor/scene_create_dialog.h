/**************************************************************************/
/*  scene_create_dialog.h                                                 */
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

#ifndef SCENE_CREATE_DIALOG_H
#define SCENE_CREATE_DIALOG_H

#include "scene/gui/dialogs.h"

class ButtonGroup;
class CheckBox;
class CreateDialog;
class EditorFileDialog;
class Label;
class LineEdit;
class OptionButton;
class PanelContainer;

class SceneCreateDialog : public ConfirmationDialog {
	GDCLASS(SceneCreateDialog, ConfirmationDialog);

	enum MsgType {
		MSG_OK,
		MSG_ERROR,
	};

	const StringName type_meta = StringName("type");

public:
	enum RootType {
		ROOT_2D_SCENE,
		ROOT_3D_SCENE,
		ROOT_USER_INTERFACE,
		ROOT_OTHER,
	};

private:
	String directory;
	String scene_name;
	String root_name;

	Ref<ButtonGroup> node_type_group;
	CheckBox *node_type_2d = nullptr;
	CheckBox *node_type_3d = nullptr;
	CheckBox *node_type_gui = nullptr;
	CheckBox *node_type_other = nullptr;

	LineEdit *other_type_display = nullptr;
	Button *select_node_button = nullptr;
	CreateDialog *select_node_dialog = nullptr;

	LineEdit *scene_name_edit = nullptr;
	OptionButton *scene_extension_picker = nullptr;
	LineEdit *root_name_edit = nullptr;

	PanelContainer *status_panel = nullptr;
	Label *file_error_label = nullptr;
	Label *node_error_label = nullptr;

	void accept_create(Variant p_discard = Variant()); // Extra unused argument, because unbind() doesn't exist in 3.x.
	void browse_types();
	void on_type_picked();
	void update_dialog(Variant p_discard = Variant());
	void update_error(Label *p_label, MsgType p_type, const String &p_msg);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void config(const String &p_dir);

	String get_scene_path() const;
	Node *create_scene_root();

	SceneCreateDialog();
};

#endif // SCENE_CREATE_DIALOG_H
