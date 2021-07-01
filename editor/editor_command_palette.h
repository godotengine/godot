/*************************************************************************/
/*  editor_command_palette.h                                             */
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

#ifndef EDITOR_COMMAND_PALETTE_H
#define EDITOR_COMMAND_PALETTE_H

#include "core/os/thread_safe.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/shortcut.h"
#include "scene/gui/tree.h"

class EditorCommandPalette : public ConfirmationDialog {
	GDCLASS(EditorCommandPalette, ConfirmationDialog);

	static EditorCommandPalette *singleton;
	LineEdit *command_search_box;
	Tree *search_options;
	StringName base_type;
	bool allow_multi_select;
	String selected_command;

	HashMap<String, Callable> callables;
	Vector<String> unregisterd_shortcuts;

	void _text_changed(const String &p_newtext);
	void _update_search();
	void _update_command_search();
	void _search_file();
	void _search_action();
	void _confirmed();
	void _cleanup();
	void _sbox_input(const Ref<InputEvent> &p_ie);
	void _hide_command_palette();
	void _text_confirmed(const String &p_text);
	EditorCommandPalette();

protected:
	String get_command_text() const;
	static void _bind_methods();

public:
	static EditorCommandPalette *get_singleton();
	Ref<Shortcut> create_shortcut_and_command(const String &p_path, const String &p_name, uint32_t p_keycode);
	void register_shortcuts_as_command();
	void open_popup();
	void set_selected_commmad(String);
	String get_selected_command();
	void get_actions_list(List<String> *p_list) const;
	void add_command(String p_command_name, Callable p_action, Vector<Variant> arguments);
	void execute_command(String p_command_name);
};

Ref<Shortcut> ED_SHORTCUT_AS_COMMAND(String p_command, Ref<Shortcut> p_shortcut);

#endif //EDITOR_COMMAND_PALETTE_H
