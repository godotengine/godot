/**************************************************************************/
/*  visual_script_property_selector.h                                     */
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

#ifndef VISUAL_SCRIPT_PROPERTY_SELECTOR_H
#define VISUAL_SCRIPT_PROPERTY_SELECTOR_H

#include "editor/editor_help.h"
#include "editor/property_editor.h"
#include "scene/gui/rich_text_label.h"

class VisualScriptPropertySelector : public ConfirmationDialog {
	GDCLASS(VisualScriptPropertySelector, ConfirmationDialog);

	LineEdit *search_box;
	Tree *search_options;

	void _update_search();

	void create_visualscript_item(const String &name, TreeItem *const root, const String &search_input, const String &text);

	void get_visual_node_names(const String &root_filter, const Set<String> &p_modifiers, bool &found, TreeItem *const root, LineEdit *const search_box);

	void _sbox_input(const Ref<InputEvent> &p_ie);

	void _confirmed();
	void _text_changed(const String &p_newtext);

	EditorHelpBit *help_bit;

	bool properties;
	bool visual_script_generic;
	bool connecting;
	String selected;
	Variant::Type type;
	String base_type;
	ObjectID script;
	Object *instance;
	bool virtuals_only;
	bool seq_connect;

	void _item_selected();

	Vector<Variant::Type> type_filter;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void select_method_from_base_type(const String &p_base, const String &p_current = "", const bool p_virtuals_only = false, const bool p_connecting = true, bool clear_text = true);
	void select_from_base_type(const String &p_base, const String &p_current = "", bool p_virtuals_only = false, bool p_seq_connect = false, const bool p_connecting = true, bool clear_text = true);
	void select_from_script(const Ref<Script> &p_script, const String &p_current = "", const bool p_connecting = true, bool clear_text = true);
	void select_from_basic_type(Variant::Type p_type, const String &p_current = "", const bool p_connecting = true, bool clear_text = true);
	void select_from_action(const String &p_type, const String &p_current = "", const bool p_connecting = true, bool clear_text = true);
	void select_from_instance(Object *p_instance, const String &p_current = "", const bool p_connecting = true, const String &p_basetype = "", bool clear_text = true);
	void select_from_visual_script(const String &p_base, const bool p_connecting = true, bool clear_text = true);

	void show_window(float p_screen_ratio);

	void set_type_filter(const Vector<Variant::Type> &p_type_filter);

	VisualScriptPropertySelector();
};

#endif // VISUAL_SCRIPT_PROPERTY_SELECTOR_H
