/**************************************************************************/
/*  editor_create_dialog.h                                                */
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

#ifndef EDITOR_CREATE_DIALOG_H
#define EDITOR_CREATE_DIALOG_H

#include "editor/create_dialog.h"
#include "editor/editor_node.h"

class CreateDialog;

class EditorCreateDialog : public Node {
	GDCLASS(EditorCreateDialog, Node);

	CreateDialog *create_dialog = nullptr;

#define CHECK_IF_NO_BOUND_CREATE_DIALOG \
	ERR_FAIL_COND_MSG(create_dialog == nullptr, "Null create dialog is bound. It's a bug and please report it.")
#define CHECK_IF_NO_BOUND_CREATE_DIALOG_V(m_retval) \
	ERR_FAIL_COND_V_MSG(create_dialog == nullptr, (m_retval), "Null create dialog is bound. It's a bug and please report it.")

	void _notify_visibility_changed();
	void _notify_created();
	void _notify_favourites_updated();

protected:
	static void _bind_methods();

public:
	void add_type_to_blacklist(const StringName &p_type_name);
	void remove_type_from_blacklist(const StringName &p_type_name);

	void set_type_custom_suffix(const StringName &p_type_name, const String &p_custom_suffix);
	String get_type_custom_suffix(const StringName &p_type_name) const;
	void clear_all_type_custom_suffixes();

	Tree *get_search_options() const;

	EditorCreateDialog();
	EditorCreateDialog(CreateDialog *p_create_dialog);
};

#endif EDITOR_CREATE_DIALOG_H
