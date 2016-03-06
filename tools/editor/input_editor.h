/*************************************************************************/
/*  input_editor.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#ifndef INPUT_EDITOR_H
#define INPUT_EDITOR_H

#include "scene/gui/dialogs.h"
#include "property_editor.h"
#include "optimized_save_dialog.h"
#include "undo_redo.h"
#include "editor_data.h"
#include "editor_plugin_settings.h"

class InputEditor : public Control {

	OBJ_TYPE(InputEditor, Control);

	InputEvent::Type add_type;
	String add_at;

	UndoRedo *undo_redo;
	ConfirmationDialog *message;

	PopupMenu *popup_add_key;
	ConfirmationDialog *press_a_key_dialog;
	Label*press_a_key_label;
	ConfirmationDialog *device_input_dialog;
	SpinBox *device_id;
	OptionButton *device_index;
	Label *device_index_label;
	LineEdit *action_name;
	Tree *input_tree;

	bool setting;

	InputEvent last_wait_for_key;

	void _update_actions() const;
	void _add_item(int p_item);

	void _action_adds(String);
	void _action_add();
	void _device_input_add();

	void _action_selected();
	void _action_edited();
	void _action_button_pressed(Object* p_obj, int p_column, int p_id);
	void _wait_for_key(const InputEvent& p_event);
	void _press_a_key_confirm();

	void _settings_changed();

protected:

	void _notification(int p_what);
	static void _bind_methods();

public:

	InputEditor(UndoRedo *p_undoredo = 0);

};

#endif // INPUT_EDITOR_H
