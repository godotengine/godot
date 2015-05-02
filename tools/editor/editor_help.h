/*************************************************************************/
/*  editor_help.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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

#include "tools/editor/editor_plugin.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/split_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/tree.h"

#include "scene/main/timer.h"
#include "tools/editor/code_editor.h"
#include "tools/doc/doc_data.h"


class EditorNode;

class EditorHelpSearch : public ConfirmationDialog {

	OBJ_TYPE(EditorHelpSearch,ConfirmationDialog )

	EditorNode *editor;
	LineEdit *search_box;
	Tree *search_options;
	String base_type;

	void _update_search();

	void _sbox_input(const InputEvent& p_ie);

	void _confirmed();
	void _text_changed(const String& p_newtext);


protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	void popup(const String& p_term="");

	EditorHelpSearch(EditorNode *p_editor);
};



class EditorHelp : public VBoxContainer {
	OBJ_TYPE( EditorHelp, VBoxContainer );


	enum Page {

		PAGE_CLASS_LIST,
		PAGE_CLASS_DESC,
		PAGE_CLASS_PREV,
		PAGE_CLASS_NEXT,
		PAGE_SEARCH,
		CLASS_SEARCH,

	};


	struct History {
		String c;
		int scroll;
	};

	Vector<History> history;
	int history_pos;
	bool select_locked;

	String prev_search;
	String prev_search_page;

	EditorNode *editor;
	Map<String,int> method_line;
	Map<String,int> signal_line;
	Map<String,int> property_line;
	Map<String,int> theme_property_line;
	Map<String,int> constant_line;
	int description_line;

	Tree *class_list;

	RichTextLabel *class_desc;
	HSplitContainer *h_split;
	static DocData *doc;

	Button *class_list_button;
	Button *edited_class;
	Button *back;
	Button *forward;
	LineEdit *search;

	String base_path;

	HashMap<String,TreeItem*> tree_item_map;


	void _help_callback(const String& p_topic);

	void _add_text(const String& p_text);
	bool scroll_locked;

	void _button_pressed(int p_idx);
	void _add_type(const String& p_type);

	void _scroll_changed(double p_scroll);
	void _class_list_select(const String& p_select);
	void _class_desc_select(const String& p_select);

	Error _goto_desc(const String& p_class,bool p_update_history=true,int p_vscr=-1);
	void _update_history_buttons();
	void _update_doc();

	void _request_help(const String& p_string);
	void _search(const String& p_str);

	void _unhandled_key_input(const InputEvent& p_ev);
	void add_type(const String& p_type,HashMap<String,TreeItem*>& p_types,TreeItem *p_root);
	void _tree_item_selected();

	EditorHelpSearch *class_search;

protected:


	void _notification(int p_what);
	static void _bind_methods();
public:

	static void generate_doc();
	static DocData *get_doc_data() { return doc; }

	EditorHelp(EditorNode *p_editor=NULL);
	~EditorHelp();
};



class EditorHelpPlugin : public EditorPlugin {

	OBJ_TYPE( EditorHelpPlugin, EditorPlugin );

	EditorHelp *editor_help;
	EditorNode *editor;
public:

	virtual String get_name() const { return "Help"; }
	bool has_main_screen() const { return true; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);
	virtual void selected_notify();

	Dictionary get_state() const;
	virtual void set_state(const Dictionary& p_state);
	virtual void clear();

	virtual void save_external_data();
	virtual void apply_changes();

	virtual void restore_global_state();
	virtual void save_global_state();

	EditorHelpPlugin(EditorNode *p_node);
	~EditorHelpPlugin();

};


#endif // EDITOR_HELP_H
