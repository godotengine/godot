/*************************************************************************/
/*  editor_help.h                                                        */
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

#ifndef EDITOR_HELP_H
#define EDITOR_HELP_H

#include "editor/code_editor.h"
#include "editor/doc_tools.h"
#include "editor/editor_plugin.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/main/timer.h"

class FindBar : public HBoxContainer {
	GDCLASS(FindBar, HBoxContainer);

	LineEdit *search_text;
	Button *find_prev;
	Button *find_next;
	Label *matches_label;
	TextureButton *hide_button;
	String prev_search;

	RichTextLabel *rich_text_label;

	int results_count;

	void _hide_bar();

	void _search_text_changed(const String &p_text);
	void _search_text_submitted(const String &p_text);

	void _update_results_count();
	void _update_matches_label();

protected:
	void _notification(int p_what);
	virtual void unhandled_input(const Ref<InputEvent> &p_event) override;

	bool _search(bool p_search_previous = false);

	static void _bind_methods();

public:
	void set_rich_text_label(RichTextLabel *p_rich_text_label);

	void popup_search();

	bool search_prev();
	bool search_next();

	FindBar();
};

class EditorHelp : public VBoxContainer {
	GDCLASS(EditorHelp, VBoxContainer);

	enum Page {
		PAGE_CLASS_LIST,
		PAGE_CLASS_DESC,
		PAGE_CLASS_PREV,
		PAGE_CLASS_NEXT,
		PAGE_SEARCH,
		CLASS_SEARCH,

	};

	bool select_locked;

	String prev_search;

	String edited_class;

	Vector<Pair<String, int>> section_line;
	Map<String, int> method_line;
	Map<String, int> signal_line;
	Map<String, int> property_line;
	Map<String, int> theme_property_line;
	Map<String, int> constant_line;
	Map<String, int> enum_line;
	Map<String, Map<String, int>> enum_values_line;
	int description_line;

	RichTextLabel *class_desc;
	HSplitContainer *h_split;
	static DocTools *doc;

	ConfirmationDialog *search_dialog;
	LineEdit *search;
	FindBar *find_bar;
	HBoxContainer *status_bar;
	Button *toggle_scripts_button;

	String base_path;

	Color title_color;
	Color text_color;
	Color headline_color;
	Color base_type_color;
	Color type_color;
	Color comment_color;
	Color symbol_color;
	Color value_color;
	Color qualifier_color;

	void _init_colors();
	void _help_callback(const String &p_topic);

	void _add_text(const String &p_bbcode);
	bool scroll_locked;

	//void _button_pressed(int p_idx);
	void _add_type(const String &p_type, const String &p_enum = String());
	void _add_method(const DocData::MethodDoc &p_method, bool p_overview = true);

	void _class_list_select(const String &p_select);
	void _class_desc_select(const String &p_select);
	void _class_desc_input(const Ref<InputEvent> &p_input);
	void _class_desc_resized();

	Error _goto_desc(const String &p_class, int p_vscr = -1);
	//void _update_history_buttons();
	void _update_doc();

	void _request_help(const String &p_string);
	void _search(bool p_search_previous = false);

	String _fix_constant(const String &p_constant) const;
	void _toggle_scripts_pressed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static void generate_doc();
	static DocTools *get_doc_data() { return doc; }

	void go_to_help(const String &p_help);
	void go_to_class(const String &p_class, int p_scroll = 0);
	void update_doc();

	Vector<Pair<String, int>> get_sections();
	void scroll_to_section(int p_section_index);

	void popup_search();
	void search_again(bool p_search_previous = false);

	String get_class();

	void set_focused() { class_desc->grab_focus(); }

	int get_scroll() const;
	void set_scroll(int p_scroll);

	void update_toggle_scripts_button();

	EditorHelp();
	~EditorHelp();
};

class EditorHelpBit : public MarginContainer {
	GDCLASS(EditorHelpBit, MarginContainer);

	RichTextLabel *rich_text;
	void _go_to_help(String p_what);
	void _meta_clicked(String p_select);

	String text;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	RichTextLabel *get_rich_text() { return rich_text; }
	void set_text(const String &p_text);
	EditorHelpBit();
};

#endif // EDITOR_HELP_H
