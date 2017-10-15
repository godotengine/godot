/*************************************************************************/
/*  editor_help.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor_help.h"

#include "doc_data_compressed.gen.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "os/keyboard.h"

#define CONTRIBUTE_URL "http://docs.godotengine.org/en/latest/community/contributing/updating_the_class_reference.html"

void EditorHelpSearch::popup() {

	popup_centered(Size2(700, 600) * EDSCALE);
	if (search_box->get_text() != "") {
		search_box->select_all();
		_update_search();
	}
	search_box->grab_focus();
}

void EditorHelpSearch::popup(const String &p_term) {

	popup_centered(Size2(700, 600) * EDSCALE);
	if (p_term != "") {
		search_box->set_text(p_term);
		search_box->select_all();
		_update_search();
	} else
		search_box->clear();
	search_box->grab_focus();
}

void EditorHelpSearch::_text_changed(const String &p_newtext) {

	_update_search();
}

void EditorHelpSearch::_sbox_input(const Ref<InputEvent> &p_ie) {

	Ref<InputEventKey> k = p_ie;

	if (k.is_valid() && (k->get_scancode() == KEY_UP ||
								k->get_scancode() == KEY_DOWN ||
								k->get_scancode() == KEY_PAGEUP ||
								k->get_scancode() == KEY_PAGEDOWN)) {

		search_options->call("_gui_input", k);
		search_box->accept_event();
	}
}

void EditorHelpSearch::_update_search() {

	search_options->clear();
	search_options->set_hide_root(true);

	/*
	TreeItem *root = search_options->create_item();
	_parse_fs(EditorFileSystem::get_singleton()->get_filesystem());
*/

	List<StringName> type_list;
	ClassDB::get_class_list(&type_list);

	DocData *doc = EditorHelp::get_doc_data();
	String term = search_box->get_text();
	if (term.length() < 2)
		return;

	TreeItem *root = search_options->create_item();

	Ref<Texture> def_icon = get_icon("Node", "EditorIcons");
	//classes first
	for (Map<String, DocData::ClassDoc>::Element *E = doc->class_list.front(); E; E = E->next()) {

		if (E->key().findn(term) != -1) {

			TreeItem *item = search_options->create_item(root);
			item->set_metadata(0, "class_name:" + E->key());
			item->set_text(0, E->key() + " (Class)");
			if (has_icon(E->key(), "EditorIcons"))
				item->set_icon(0, get_icon(E->key(), "EditorIcons"));
			else
				item->set_icon(0, def_icon);
		}
	}

	//class methods, etc second
	for (Map<String, DocData::ClassDoc>::Element *E = doc->class_list.front(); E; E = E->next()) {

		DocData::ClassDoc &c = E->get();

		Ref<Texture> cicon;
		if (has_icon(E->key(), "EditorIcons"))
			cicon = get_icon(E->key(), "EditorIcons");
		else
			cicon = def_icon;

		for (int i = 0; i < c.methods.size(); i++) {
			if ((term.begins_with(".") && c.methods[i].name.begins_with(term.right(1))) || (term.ends_with("(") && c.methods[i].name.ends_with(term.left(term.length() - 1).strip_edges())) || (term.begins_with(".") && term.ends_with("(") && c.methods[i].name == term.substr(1, term.length() - 2).strip_edges()) || c.methods[i].name.findn(term) != -1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0, "class_method:" + E->key() + ":" + c.methods[i].name);
				item->set_text(0, E->key() + "." + c.methods[i].name + " (Method)");
				item->set_icon(0, cicon);
			}
		}

		for (int i = 0; i < c.signals.size(); i++) {

			if (c.signals[i].name.findn(term) != -1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0, "class_signal:" + E->key() + ":" + c.signals[i].name);
				item->set_text(0, E->key() + "." + c.signals[i].name + " (Signal)");
				item->set_icon(0, cicon);
			}
		}

		for (int i = 0; i < c.constants.size(); i++) {

			if (c.constants[i].name.findn(term) != -1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0, "class_constant:" + E->key() + ":" + c.constants[i].name);
				item->set_text(0, E->key() + "." + c.constants[i].name + " (Constant)");
				item->set_icon(0, cicon);
			}
		}

		for (int i = 0; i < c.properties.size(); i++) {

			if (c.properties[i].name.findn(term) != -1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0, "class_property:" + E->key() + ":" + c.properties[i].name);
				item->set_text(0, E->key() + "." + c.properties[i].name + " (Property)");
				item->set_icon(0, cicon);
			}
		}

		for (int i = 0; i < c.theme_properties.size(); i++) {

			if (c.theme_properties[i].name.findn(term) != -1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0, "class_theme_item:" + E->key() + ":" + c.theme_properties[i].name);
				item->set_text(0, E->key() + "." + c.theme_properties[i].name + " (Theme Item)");
				item->set_icon(0, cicon);
			}
		}
	}

	//same but descriptions

	for (Map<String, DocData::ClassDoc>::Element *E = doc->class_list.front(); E; E = E->next()) {

		DocData::ClassDoc &c = E->get();

		Ref<Texture> cicon;
		if (has_icon(E->key(), "EditorIcons"))
			cicon = get_icon(E->key(), "EditorIcons");
		else
			cicon = def_icon;

		if (c.description.findn(term) != -1) {

			TreeItem *item = search_options->create_item(root);
			item->set_metadata(0, "class_desc:" + E->key());
			item->set_text(0, E->key() + " (Class Description)");
			item->set_icon(0, cicon);
		}

		for (int i = 0; i < c.methods.size(); i++) {

			if (c.methods[i].description.findn(term) != -1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0, "class_method_desc:" + E->key() + ":" + c.methods[i].name);
				item->set_text(0, E->key() + "." + c.methods[i].name + " (Method Description)");
				item->set_icon(0, cicon);
			}
		}

		for (int i = 0; i < c.signals.size(); i++) {

			if (c.signals[i].description.findn(term) != -1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0, "class_signal:" + E->key() + ":" + c.signals[i].name);
				item->set_text(0, E->key() + "." + c.signals[i].name + " (Signal Description)");
				item->set_icon(0, cicon);
			}
		}

		for (int i = 0; i < c.constants.size(); i++) {

			if (c.constants[i].description.findn(term) != -1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0, "class_constant:" + E->key() + ":" + c.constants[i].name);
				item->set_text(0, E->key() + "." + c.constants[i].name + " (Constant Description)");
				item->set_icon(0, cicon);
			}
		}

		for (int i = 0; i < c.properties.size(); i++) {

			if (c.properties[i].description.findn(term) != -1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0, "class_property_desc:" + E->key() + ":" + c.properties[i].name);
				item->set_text(0, E->key() + "." + c.properties[i].name + " (Property Description)");
				item->set_icon(0, cicon);
			}
		}
	}

	get_ok()->set_disabled(root->get_children() == NULL);
}

void EditorHelpSearch::_confirmed() {

	TreeItem *ti = search_options->get_selected();
	if (!ti)
		return;

	String mdata = ti->get_metadata(0);
	EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
	emit_signal("go_to_help", mdata);
	// go to that
	hide();
}

void EditorHelpSearch::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		//_update_icons
		search_box->add_icon_override("right_icon", get_icon("Search", "EditorIcons"));

		connect("confirmed", this, "_confirmed");
		_update_search();
	} else if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {

		if (is_visible_in_tree()) {

			search_box->call_deferred("grab_focus"); // still not visible
			search_box->select_all();
		}
	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {

		//_update_icons
		search_box->add_icon_override("right_icon", get_icon("Search", "EditorIcons"));
	}
}

void EditorHelpSearch::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_text_changed"), &EditorHelpSearch::_text_changed);
	ClassDB::bind_method(D_METHOD("_confirmed"), &EditorHelpSearch::_confirmed);
	ClassDB::bind_method(D_METHOD("_sbox_input"), &EditorHelpSearch::_sbox_input);
	ClassDB::bind_method(D_METHOD("_update_search"), &EditorHelpSearch::_update_search);

	ADD_SIGNAL(MethodInfo("go_to_help"));
}

EditorHelpSearch::EditorHelpSearch() {

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	search_box = memnew(LineEdit);
	vbc->add_child(search_box);
	search_box->connect("text_changed", this, "_text_changed");
	search_box->connect("gui_input", this, "_sbox_input");
	search_options = memnew(Tree);
	vbc->add_margin_child(TTR("Matches:"), search_options, true);
	get_ok()->set_text(TTR("Open"));
	get_ok()->set_disabled(true);
	register_text_enter(search_box);
	set_hide_on_ok(false);
	search_options->connect("item_activated", this, "_confirmed");
	set_title(TTR("Search Help"));

	//search_options->set_hide_root(true);
}

/////////////////////////////////

////////////////////////////////////
/// /////////////////////////////////

void EditorHelpIndex::add_type(const String &p_type, HashMap<String, TreeItem *> &p_types, TreeItem *p_root) {

	if (p_types.has(p_type))
		return;
	/*
	if (!ClassDB::is_type(p_type,base) || p_type==base)
		return;
	*/

	String inherits = EditorHelp::get_doc_data()->class_list[p_type].inherits;

	TreeItem *parent = p_root;

	if (inherits.length()) {

		if (!p_types.has(inherits)) {

			add_type(inherits, p_types, p_root);
		}

		if (p_types.has(inherits))
			parent = p_types[inherits];
	}

	TreeItem *item = class_list->create_item(parent);
	item->set_metadata(0, p_type);
	item->set_tooltip(0, EditorHelp::get_doc_data()->class_list[p_type].brief_description);
	item->set_text(0, p_type);

	if (has_icon(p_type, "EditorIcons")) {

		item->set_icon(0, get_icon(p_type, "EditorIcons"));
	}

	p_types[p_type] = item;
}

void EditorHelpIndex::_tree_item_selected() {

	TreeItem *s = class_list->get_selected();
	if (!s)
		return;

	EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
	emit_signal("open_class", s->get_text(0));
	hide();

	//_goto_desc(s->get_text(0));
}

void EditorHelpIndex::select_class(const String &p_class) {

	if (!tree_item_map.has(p_class))
		return;
	tree_item_map[p_class]->select(0);
	class_list->ensure_cursor_is_visible();
}

void EditorHelpIndex::popup() {

	popup_centered(Size2(500, 600) * EDSCALE);

	search_box->set_text("");
	_update_class_list();
}

void EditorHelpIndex::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		//_update_icons
		search_box->add_icon_override("right_icon", get_icon("Search", "EditorIcons"));
		_update_class_list();

		connect("confirmed", this, "_tree_item_selected");

	} else if (p_what == NOTIFICATION_POST_POPUP) {

		search_box->call_deferred("grab_focus");
	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {

		//_update_icons
		search_box->add_icon_override("right_icon", get_icon("Search", "EditorIcons"));
	}
}

void EditorHelpIndex::_text_changed(const String &p_text) {

	_update_class_list();
}

void EditorHelpIndex::_update_class_list() {

	class_list->clear();
	tree_item_map.clear();
	TreeItem *root = class_list->create_item();
	class_list->set_hide_root(true);

	String filter = search_box->get_text().strip_edges();
	String to_select = "";

	for (Map<String, DocData::ClassDoc>::Element *E = EditorHelp::get_doc_data()->class_list.front(); E; E = E->next()) {

		if (filter == "") {
			add_type(E->key(), tree_item_map, root);
		} else {

			bool found = false;
			String type = E->key();

			while (type != "") {
				if (filter.is_subsequence_ofi(type)) {

					if (to_select.empty()) {
						to_select = type;
					}

					found = true;
					break;
				}

				type = EditorHelp::get_doc_data()->class_list[type].inherits;
			}

			if (found) {
				add_type(E->key(), tree_item_map, root);
			}
		}
	}

	if (tree_item_map.has(filter)) {
		select_class(filter);
	} else if (to_select != "") {
		select_class(to_select);
	}
}

void EditorHelpIndex::_sbox_input(const Ref<InputEvent> &p_ie) {

	Ref<InputEventKey> k = p_ie;

	if (k.is_valid() && (k->get_scancode() == KEY_UP ||
								k->get_scancode() == KEY_DOWN ||
								k->get_scancode() == KEY_PAGEUP ||
								k->get_scancode() == KEY_PAGEDOWN)) {

		class_list->call("_gui_input", k);
		search_box->accept_event();
	}
}

void EditorHelpIndex::_bind_methods() {

	ClassDB::bind_method("_tree_item_selected", &EditorHelpIndex::_tree_item_selected);
	ClassDB::bind_method("_text_changed", &EditorHelpIndex::_text_changed);
	ClassDB::bind_method("_sbox_input", &EditorHelpIndex::_sbox_input);
	ClassDB::bind_method("select_class", &EditorHelpIndex::select_class);
	ADD_SIGNAL(MethodInfo("open_class"));
}

EditorHelpIndex::EditorHelpIndex() {

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	search_box = memnew(LineEdit);
	vbc->add_child(search_box);
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);

	register_text_enter(search_box);

	search_box->connect("text_changed", this, "_text_changed");
	search_box->connect("gui_input", this, "_sbox_input");

	class_list = memnew(Tree);
	vbc->add_margin_child(TTR("Class List:") + " ", class_list, true);
	class_list->set_v_size_flags(SIZE_EXPAND_FILL);

	class_list->connect("item_activated", this, "_tree_item_selected");

	get_ok()->set_text(TTR("Open"));
	set_title(TTR("Search Classes"));
}

/////////////////////////////////

////////////////////////////////////
/// /////////////////////////////////
DocData *EditorHelp::doc = NULL;

void EditorHelp::_unhandled_key_input(const Ref<InputEvent> &p_ev) {

	if (!is_visible_in_tree())
		return;

	Ref<InputEventKey> k = p_ev;

	if (k.is_valid() && k->get_control() && k->get_scancode() == KEY_F) {

		search->grab_focus();
		search->select_all();
	}
}

void EditorHelp::_search(const String &) {

	if (search->get_text() == "")
		return;

	String stext = search->get_text();
	bool keep = prev_search == stext;

	bool ret = class_desc->search(stext, keep);
	if (!ret) {
		class_desc->search(stext, false);
	}

	prev_search = stext;
}

void EditorHelp::_class_list_select(const String &p_select) {

	_goto_desc(p_select);
}

void EditorHelp::_class_desc_select(const String &p_select) {

	//print_line("LINK: "+p_select);
	if (p_select.begins_with("$")) { //enum
		//_goto_desc(p_select.substr(1,p_select.length()));
		String select = p_select.substr(1, p_select.length());
		String class_name;
		if (select.find(".") != -1) {
			class_name = select.get_slice(".", 0);
		} else {
			class_name = "@Global Scope";
		}
		emit_signal("go_to_help", "class_enum:" + class_name + ":" + select);
		return;
	} else if (p_select.begins_with("#")) {
		//_goto_desc(p_select.substr(1,p_select.length()));
		emit_signal("go_to_help", "class_name:" + p_select.substr(1, p_select.length()));
		return;
	} else if (p_select.begins_with("@")) {

		String m = p_select.substr(1, p_select.length());

		if (m.find(".") != -1) {
			//must go somewhere else

			emit_signal("go_to_help", "class_method:" + m.get_slice(".", 0) + ":" + m.get_slice(".", 0));
		} else {

			if (!method_line.has(m))
				return;
			class_desc->scroll_to_line(method_line[m]);
		}
	} else if (p_select.begins_with("http")) {
		OS::get_singleton()->shell_open(p_select);
	}
}

void EditorHelp::_class_desc_input(const Ref<InputEvent> &p_input) {

	Ref<InputEventMouseButton> mb = p_input;

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == 1) {
		class_desc->set_selection_enabled(false);
		class_desc->set_selection_enabled(true);
	}
	set_focused();
}

void EditorHelp::_add_type(const String &p_type, const String &p_enum) {

	String t = p_type;
	if (t == "")
		t = "void";
	bool can_ref = (t != "int" && t != "real" && t != "bool" && t != "void") || p_enum != String();

	if (p_enum != String()) {
		if (p_enum.get_slice_count(".") > 1) {
			t = p_enum.get_slice(".", 1);
		} else {
			t = p_enum.get_slice(".", 0);
		}
	}
	const Color text_color = get_color("default_color", "RichTextLabel");
	const Color type_color = get_color("accent_color", "Editor").linear_interpolate(text_color, 0.5);
	class_desc->push_color(type_color);
	if (can_ref) {
		if (p_enum == "") {
			class_desc->push_meta("#" + t); //class
		} else {
			class_desc->push_meta("$" + p_enum); //class
		}
	}
	class_desc->add_text(t);
	if (can_ref)
		class_desc->pop();
	class_desc->pop();
}

Error EditorHelp::_goto_desc(const String &p_class, int p_vscr) {

	//ERR_FAIL_COND(!doc->class_list.has(p_class));
	if (!doc->class_list.has(p_class))
		return ERR_DOES_NOT_EXIST;

	//if (tree_item_map.has(p_class)) {
	select_locked = true;
	//}

	class_desc->show();
	//tabs->set_current_tab(PAGE_CLASS_DESC);
	description_line = 0;

	if (p_class == edited_class)
		return OK; //already there

	scroll_locked = true;

	class_desc->clear();
	method_line.clear();
	section_line.clear();
	edited_class = p_class;
	//edited_class->show();

	// Colors
	const Color title_color = get_color("accent_color", "Editor");
	const Color text_color = get_color("default_color", "RichTextLabel");
	const Color headline_color = get_color("headline_color", "EditorHelp");
	const Color base_type_color = title_color.linear_interpolate(text_color, 0.5);
	const Color comment_color = Color(text_color.r, text_color.g, text_color.b, 0.6);
	const Color symbol_color = comment_color;
	const Color value_color = Color(text_color.r, text_color.g, text_color.b, 0.4);
	const Color qualifier_color = Color(text_color.r, text_color.g, text_color.b, 0.8);

	DocData::ClassDoc cd = doc->class_list[p_class]; //make a copy, so we can sort without worrying

	Ref<Font> doc_font = get_font("doc", "EditorFonts");
	Ref<Font> doc_title_font = get_font("doc_title", "EditorFonts");
	Ref<Font> doc_code_font = get_font("doc_source", "EditorFonts");
	String link_color_text = title_color.to_html(false);

	section_line.push_back(Pair<String, int>(TTR("Top"), 0));
	class_desc->push_font(doc_title_font);
	class_desc->push_color(title_color);
	class_desc->add_text(TTR("Class:") + " ");
	class_desc->push_color(headline_color);
	_add_text(p_class);
	class_desc->pop();
	class_desc->pop();
	class_desc->pop();
	class_desc->add_newline();

	if (cd.inherits != "") {

		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Inherits:") + " ");
		class_desc->pop();
		class_desc->pop();

		String inherits = cd.inherits;

		class_desc->push_font(doc_font);

		while (inherits != "") {
			_add_type(inherits);

			inherits = doc->class_list[inherits].inherits;

			if (inherits != "") {
				class_desc->add_text(" < ");
			}
		}

		class_desc->pop();
		class_desc->add_newline();
	}

	if (ClassDB::class_exists(cd.name)) {

		bool found = false;
		bool prev = false;

		for (Map<String, DocData::ClassDoc>::Element *E = doc->class_list.front(); E; E = E->next()) {

			if (E->get().inherits == cd.name) {

				if (!found) {
					class_desc->push_color(title_color);
					class_desc->push_font(doc_title_font);
					class_desc->add_text(TTR("Inherited by:") + " ");
					class_desc->pop();
					class_desc->pop();

					found = true;
					class_desc->push_font(doc_font);
				}

				if (prev) {

					class_desc->add_text(" , ");
					prev = false;
				}

				_add_type(E->get().name);
				prev = true;
			}
		}

		if (found)
			class_desc->pop();

		class_desc->add_newline();
	}

	class_desc->add_newline();
	class_desc->add_newline();

	if (cd.brief_description != "") {

		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Brief Description:"));
		class_desc->pop();
		class_desc->pop();

		//class_desc->add_newline();
		class_desc->add_newline();
		class_desc->push_color(text_color);
		class_desc->push_font(doc_font);
		class_desc->push_indent(1);
		_add_text(cd.brief_description);
		class_desc->pop();
		class_desc->pop();
		class_desc->pop();
		class_desc->add_newline();
		class_desc->add_newline();
		class_desc->add_newline();
	}

	Set<String> skip_methods;
	bool property_descr = false;

	if (cd.properties.size()) {

		section_line.push_back(Pair<String, int>(TTR("Members"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Members:"));
		class_desc->pop();
		class_desc->pop();
		//class_desc->add_newline();

		class_desc->push_indent(1);
		class_desc->push_table(2);
		class_desc->set_table_column_expand(1, 1);

		for (int i = 0; i < cd.properties.size(); i++) {
			property_line[cd.properties[i].name] = class_desc->get_line_count() - 2; //gets overridden if description

			class_desc->push_cell();
			class_desc->push_align(RichTextLabel::ALIGN_RIGHT);
			class_desc->push_font(doc_code_font);
			_add_type(cd.properties[i].type, cd.properties[i].enumeration);
			class_desc->pop();
			class_desc->pop();
			class_desc->pop();

			bool describe = false;

			if (cd.properties[i].setter != "") {
				skip_methods.insert(cd.properties[i].setter);
				describe = true;
			}
			if (cd.properties[i].getter != "") {
				skip_methods.insert(cd.properties[i].getter);
				describe = true;
			}

			if (cd.properties[i].description != "") {
				describe = true;
			}
			class_desc->push_cell();
			if (describe) {
				class_desc->push_meta("@" + cd.properties[i].name);
			}

			class_desc->push_font(doc_code_font);
			class_desc->push_color(headline_color);
			_add_text(cd.properties[i].name);

			if (describe) {
				class_desc->pop();
				property_descr = true;
			}

			class_desc->pop();
			class_desc->pop();
			class_desc->pop();
		}

		class_desc->pop(); //table
		class_desc->pop();
		class_desc->add_newline();
		class_desc->add_newline();
	}

	bool method_descr = false;
	bool sort_methods = EditorSettings::get_singleton()->get("text_editor/help/sort_functions_alphabetically");

	Vector<DocData::MethodDoc> methods;

	for (int i = 0; i < cd.methods.size(); i++) {
		if (skip_methods.has(cd.methods[i].name))
			continue;
		methods.push_back(cd.methods[i]);
	}

	if (methods.size()) {

		if (sort_methods)
			methods.sort();

		section_line.push_back(Pair<String, int>(TTR("Public Methods"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Public Methods:"));
		class_desc->pop();
		class_desc->pop();

		//class_desc->add_newline();
		//class_desc->add_newline();

		class_desc->push_indent(1);
		class_desc->push_table(2);
		class_desc->set_table_column_expand(1, 1);

		for (int i = 0; i < methods.size(); i++) {

			bool is_vararg = methods[i].qualifiers.find("vararg") != -1;

			method_line[methods[i].name] = class_desc->get_line_count() - 2; //gets overridden if description

			class_desc->push_cell();
			class_desc->push_align(RichTextLabel::ALIGN_RIGHT);
			class_desc->push_font(doc_code_font);
			_add_type(methods[i].return_type, methods[i].return_enum);
			//class_desc->add_text(" ");
			class_desc->pop(); //align
			class_desc->pop(); //font
			class_desc->pop(); //cell

			class_desc->push_cell();
			class_desc->push_font(doc_code_font);

			if (methods[i].description != "") {
				method_descr = true;
				class_desc->push_meta("@" + methods[i].name);
			}
			class_desc->push_color(headline_color);
			_add_text(methods[i].name);
			class_desc->pop();
			if (methods[i].description != "")
				class_desc->pop(); // pop meta
			class_desc->push_color(symbol_color);
			class_desc->add_text(methods[i].arguments.size() || is_vararg ? "( " : "(");
			class_desc->pop();
			for (int j = 0; j < methods[i].arguments.size(); j++) {
				class_desc->push_color(text_color);
				if (j > 0)
					class_desc->add_text(", ");
				_add_type(methods[i].arguments[j].type, methods[i].arguments[j].enumeration);
				class_desc->add_text(" ");
				_add_text(methods[i].arguments[j].name);
				if (methods[i].arguments[j].default_value != "") {

					class_desc->push_color(symbol_color);
					class_desc->add_text("=");
					class_desc->pop();
					_add_text(methods[i].arguments[j].default_value);
				}

				class_desc->pop();
			}

			if (is_vararg) {
				class_desc->push_color(text_color);
				if (methods[i].arguments.size())
					class_desc->add_text(", ");
				class_desc->push_color(symbol_color);
				class_desc->add_text("...");
				class_desc->pop();
				class_desc->pop();
			}

			class_desc->push_color(symbol_color);
			class_desc->add_text(methods[i].arguments.size() || is_vararg ? " )" : ")");
			class_desc->pop();
			if (methods[i].qualifiers != "") {

				class_desc->push_color(qualifier_color);
				class_desc->add_text(" ");
				_add_text(methods[i].qualifiers);
				class_desc->pop();
			}
			class_desc->pop(); //monofont
			//class_desc->add_newline();
			class_desc->pop(); //cell
		}
		class_desc->pop(); //table
		class_desc->pop();
		class_desc->add_newline();
		class_desc->add_newline();
	}

	if (cd.theme_properties.size()) {

		section_line.push_back(Pair<String, int>(TTR("GUI Theme Items"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("GUI Theme Items:"));
		class_desc->pop();
		class_desc->pop();
		// class_desc->add_newline();

		class_desc->push_indent(1);
		class_desc->push_table(2);
		class_desc->set_table_column_expand(1, 1);

		//class_desc->add_newline();

		for (int i = 0; i < cd.theme_properties.size(); i++) {

			theme_property_line[cd.theme_properties[i].name] = class_desc->get_line_count() - 2; //gets overridden if description

			class_desc->push_cell();
			class_desc->push_align(RichTextLabel::ALIGN_RIGHT);
			class_desc->push_font(doc_code_font);
			_add_type(cd.theme_properties[i].type);
			class_desc->pop();
			class_desc->pop();
			class_desc->pop();

			class_desc->push_cell();
			class_desc->push_font(doc_code_font);
			class_desc->push_color(headline_color);
			_add_text(cd.theme_properties[i].name);
			class_desc->pop();
			class_desc->pop();

			if (cd.theme_properties[i].description != "") {
				class_desc->push_font(doc_font);
				class_desc->add_text("  ");
				class_desc->push_color(comment_color);
				_add_text(cd.theme_properties[i].description);
				class_desc->pop();
				class_desc->pop();
			}
			class_desc->pop(); // cell
		}

		class_desc->pop(); // table
		class_desc->pop();
		class_desc->add_newline();
		class_desc->add_newline();
	}

	if (cd.signals.size()) {

		if (sort_methods) {
			cd.signals.sort();
		}

		section_line.push_back(Pair<String, int>(TTR("Signals"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Signals:"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		//class_desc->add_newline();

		class_desc->push_indent(1);

		for (int i = 0; i < cd.signals.size(); i++) {

			signal_line[cd.signals[i].name] = class_desc->get_line_count() - 2; //gets overridden if description
			class_desc->push_font(doc_code_font); // monofont
			//_add_type("void");
			//class_desc->add_text(" ");
			class_desc->push_color(headline_color);
			_add_text(cd.signals[i].name);
			class_desc->pop();
			class_desc->push_color(symbol_color);
			class_desc->add_text(cd.signals[i].arguments.size() ? "( " : "(");
			class_desc->pop();
			for (int j = 0; j < cd.signals[i].arguments.size(); j++) {
				class_desc->push_color(text_color);
				if (j > 0)
					class_desc->add_text(", ");
				_add_type(cd.signals[i].arguments[j].type);
				class_desc->add_text(" ");
				_add_text(cd.signals[i].arguments[j].name);
				if (cd.signals[i].arguments[j].default_value != "") {

					class_desc->push_color(symbol_color);
					class_desc->add_text("=");
					class_desc->pop();
					_add_text(cd.signals[i].arguments[j].default_value);
				}

				class_desc->pop();
			}

			class_desc->push_color(symbol_color);
			class_desc->add_text(cd.signals[i].arguments.size() ? " )" : ")");
			class_desc->pop();
			class_desc->pop(); // end monofont
			if (cd.signals[i].description != "") {

				class_desc->push_color(comment_color);
				class_desc->add_text(" ");
				_add_text(cd.signals[i].description);
				class_desc->pop();
			}
			class_desc->add_newline();
		}

		class_desc->pop();
		class_desc->add_newline();
	}

	if (cd.constants.size()) {

		Map<String, Vector<DocData::ConstantDoc> > enums;
		Vector<DocData::ConstantDoc> constants;

		for (int i = 0; i < cd.constants.size(); i++) {

			if (cd.constants[i].enumeration != String()) {
				if (!enums.has(cd.constants[i].enumeration)) {
					enums[cd.constants[i].enumeration] = Vector<DocData::ConstantDoc>();
				}

				enums[cd.constants[i].enumeration].push_back(cd.constants[i]);
			} else {

				constants.push_back(cd.constants[i]);
			}
		}

		if (enums.size()) {

			section_line.push_back(Pair<String, int>(TTR("Enumerations"), class_desc->get_line_count() - 2));
			class_desc->push_color(title_color);
			class_desc->push_font(doc_title_font);
			class_desc->add_text(TTR("Enumerations:"));
			class_desc->pop();
			class_desc->pop();
			class_desc->push_indent(1);

			class_desc->add_newline();
			//class_desc->add_newline();

			for (Map<String, Vector<DocData::ConstantDoc> >::Element *E = enums.front(); E; E = E->next()) {

				enum_line[E->key()] = class_desc->get_line_count() - 2;

				class_desc->push_color(title_color);
				class_desc->add_text(TTR("enum  "));
				class_desc->pop();
				class_desc->push_font(doc_code_font);
				String e = E->key();
				if (e.get_slice_count(".")) {
					e = e.get_slice(".", 1);
				}

				class_desc->push_color(headline_color);
				class_desc->add_text(e);
				class_desc->pop();
				class_desc->pop();
				class_desc->push_color(symbol_color);
				class_desc->add_text(":");
				class_desc->pop();
				class_desc->add_newline();

				class_desc->push_indent(1);
				Vector<DocData::ConstantDoc> enum_list = E->get();

				for (int i = 0; i < enum_list.size(); i++) {

					class_desc->push_font(doc_code_font);
					class_desc->push_color(headline_color);
					_add_text(enum_list[i].name);
					class_desc->pop();
					class_desc->push_color(symbol_color);
					class_desc->add_text(" = ");
					class_desc->pop();
					class_desc->push_color(value_color);
					_add_text(enum_list[i].value);
					class_desc->pop();
					class_desc->pop();
					if (enum_list[i].description != "") {
						class_desc->push_font(doc_font);
						class_desc->add_text("  ");
						class_desc->push_color(comment_color);
						_add_text(enum_list[i].description);
						class_desc->pop();
						class_desc->pop();
					}

					class_desc->add_newline();
				}

				class_desc->pop();

				class_desc->add_newline();
			}

			class_desc->pop();
			class_desc->add_newline();
		}

		if (constants.size()) {

			section_line.push_back(Pair<String, int>(TTR("Constants"), class_desc->get_line_count() - 2));
			class_desc->push_color(title_color);
			class_desc->push_font(doc_title_font);
			class_desc->add_text(TTR("Constants:"));
			class_desc->pop();
			class_desc->pop();
			class_desc->push_indent(1);

			class_desc->add_newline();
			//class_desc->add_newline();

			for (int i = 0; i < constants.size(); i++) {

				constant_line[constants[i].name] = class_desc->get_line_count() - 2;
				class_desc->push_font(doc_code_font);
				class_desc->push_color(headline_color);
				_add_text(constants[i].name);
				class_desc->pop();
				class_desc->push_color(symbol_color);
				class_desc->add_text(" = ");
				class_desc->pop();
				class_desc->push_color(value_color);
				_add_text(constants[i].value);
				class_desc->pop();
				class_desc->pop();
				if (constants[i].description != "") {
					class_desc->push_font(doc_font);
					class_desc->add_text("  ");
					class_desc->push_color(comment_color);
					_add_text(constants[i].description);
					class_desc->pop();
					class_desc->pop();
				}

				class_desc->add_newline();
			}

			class_desc->pop();
			class_desc->add_newline();
		}
	}

	if (cd.description != "") {

		section_line.push_back(Pair<String, int>(TTR("Description"), class_desc->get_line_count() - 2));
		description_line = class_desc->get_line_count() - 2;
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Description:"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->push_color(text_color);
		class_desc->push_font(doc_font);
		class_desc->push_indent(1);
		_add_text(cd.description);
		class_desc->pop();
		class_desc->pop();
		class_desc->pop();
		class_desc->add_newline();
		class_desc->add_newline();
		class_desc->add_newline();
	}

	if (property_descr) {

		section_line.push_back(Pair<String, int>(TTR("Properties"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Property Description:"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();

		for (int i = 0; i < cd.properties.size(); i++) {

			method_line[cd.properties[i].name] = class_desc->get_line_count() - 2;

			class_desc->push_font(doc_code_font);
			_add_type(cd.properties[i].type, cd.properties[i].enumeration);

			class_desc->add_text(" ");
			class_desc->push_color(headline_color);
			_add_text(cd.properties[i].name);
			class_desc->pop(); //color

			class_desc->add_text(" ");

			class_desc->pop(); //font

			if (cd.properties[i].setter != "") {

				class_desc->push_font(doc_font);

				class_desc->push_indent(2);
				class_desc->push_color(comment_color);
				class_desc->add_text("Setter: ");
				class_desc->pop();

				class_desc->push_color(text_color);
				class_desc->add_text(cd.properties[i].setter + "(value)");
				class_desc->pop(); //color

				class_desc->pop(); //indent

				class_desc->pop(); //font
			}

			if (cd.properties[i].getter != "") {

				class_desc->push_font(doc_font);

				class_desc->push_indent(2);
				class_desc->push_color(comment_color);
				class_desc->add_text("Getter: ");
				class_desc->pop();

				class_desc->push_color(text_color);
				class_desc->add_text(cd.properties[i].getter + "()");
				class_desc->pop(); //color

				class_desc->pop(); //indent

				class_desc->pop(); //font
			}

			class_desc->add_newline();

			class_desc->push_color(text_color);
			class_desc->push_font(doc_font);
			class_desc->push_indent(1);
			if (cd.properties[i].description.strip_edges() != String()) {
				_add_text(cd.properties[i].description);
			} else {
				class_desc->add_image(get_icon("Error", "EditorIcons"));
				class_desc->add_text(" ");
				class_desc->push_color(comment_color);
				class_desc->append_bbcode(TTR("There is currently no description for this property. Please help us by [color=$color][url=$url]contributing one[/url][/color]!").replace("$url", CONTRIBUTE_URL).replace("$color", link_color_text));
				class_desc->pop();
			}
			class_desc->pop();
			class_desc->pop();
			class_desc->pop();
			class_desc->add_newline();
			class_desc->add_newline();
			class_desc->add_newline();
		}
	}

	if (method_descr) {

		section_line.push_back(Pair<String, int>(TTR("Methods"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Method Description:"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();

		for (int i = 0; i < methods.size(); i++) {

			bool is_vararg = methods[i].qualifiers.find("vararg") != -1;

			method_line[methods[i].name] = class_desc->get_line_count() - 2;

			class_desc->push_font(doc_code_font);
			_add_type(methods[i].return_type, methods[i].return_enum);

			class_desc->add_text(" ");
			class_desc->push_color(headline_color);
			_add_text(methods[i].name);
			class_desc->pop();
			class_desc->push_color(symbol_color);
			class_desc->add_text(methods[i].arguments.size() || is_vararg ? "( " : "(");
			class_desc->pop();
			for (int j = 0; j < methods[i].arguments.size(); j++) {
				class_desc->push_color(text_color);
				if (j > 0)
					class_desc->add_text(", ");
				_add_type(methods[i].arguments[j].type, methods[i].arguments[j].enumeration);
				class_desc->add_text(" ");
				_add_text(methods[i].arguments[j].name);
				if (methods[i].arguments[j].default_value != "") {

					class_desc->push_color(symbol_color);
					class_desc->add_text("=");
					class_desc->pop();
					_add_text(methods[i].arguments[j].default_value);
				}

				class_desc->pop();
			}

			if (is_vararg) {
				class_desc->push_color(text_color);
				if (methods[i].arguments.size())
					class_desc->add_text(", ");
				class_desc->push_color(symbol_color);
				class_desc->add_text("...");
				class_desc->pop();
				class_desc->pop();
			}

			class_desc->push_color(symbol_color);
			class_desc->add_text(methods[i].arguments.size() || is_vararg ? " )" : ")");
			class_desc->pop();
			if (methods[i].qualifiers != "") {

				class_desc->push_color(qualifier_color);
				class_desc->add_text(" ");
				_add_text(methods[i].qualifiers);
				class_desc->pop();
			}

			class_desc->pop();

			class_desc->add_newline();
			class_desc->push_color(text_color);
			class_desc->push_font(doc_font);
			class_desc->push_indent(1);
			if (methods[i].description.strip_edges() != String()) {
				_add_text(methods[i].description);
			} else {
				class_desc->add_image(get_icon("Error", "EditorIcons"));
				class_desc->add_text(" ");
				class_desc->push_color(comment_color);
				class_desc->append_bbcode(TTR("There is currently no description for this method. Please help us by [color=$color][url=$url]contributing one[/url][/color]!").replace("$url", CONTRIBUTE_URL).replace("$color", link_color_text));
				class_desc->pop();
			}

			class_desc->pop();
			class_desc->pop();
			class_desc->pop();
			class_desc->add_newline();
			class_desc->add_newline();
			class_desc->add_newline();
		}
	}

	scroll_locked = false;

	return OK;
}

void EditorHelp::_request_help(const String &p_string) {
	Error err = _goto_desc(p_string);
	if (err == OK) {
		EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
	}
	//100 palabras
}

void EditorHelp::_help_callback(const String &p_topic) {

	String what = p_topic.get_slice(":", 0);
	String clss = p_topic.get_slice(":", 1);
	String name;
	if (p_topic.get_slice_count(":") == 3)
		name = p_topic.get_slice(":", 2);

	_request_help(clss); //first go to class

	int line = 0;

	if (what == "class_desc") {
		line = description_line;
	} else if (what == "class_signal") {
		if (signal_line.has(name))
			line = signal_line[name];
	} else if (what == "class_method" || what == "class_method_desc") {
		if (method_line.has(name))
			line = method_line[name];
	} else if (what == "class_property") {

		if (property_line.has(name))
			line = property_line[name];
	} else if (what == "class_enum") {

		print_line("go to enum:");
		if (enum_line.has(name))
			line = enum_line[name];
	} else if (what == "class_theme_item") {

		if (theme_property_line.has(name))
			line = theme_property_line[name];
	} else if (what == "class_constant") {

		if (constant_line.has(name))
			line = constant_line[name];
	}

	class_desc->call_deferred("scroll_to_line", line);
}

static void _add_text_to_rt(const String &p_bbcode, RichTextLabel *p_rt) {

	DocData *doc = EditorHelp::get_doc_data();
	String base_path;

	Ref<Font> doc_font = p_rt->get_font("doc", "EditorFonts");
	Ref<Font> doc_code_font = p_rt->get_font("doc_source", "EditorFonts");
	Color font_color_hl = p_rt->get_color("headline_color", "EditorHelp");
	Color link_color = p_rt->get_color("accent_color", "Editor").linear_interpolate(font_color_hl, 0.8);

	String bbcode = p_bbcode.replace("\t", " ").replace("\r", " ").strip_edges();

	List<String> tag_stack;

	int pos = 0;
	while (pos < bbcode.length()) {

		int brk_pos = bbcode.find("[", pos);

		if (brk_pos < 0)
			brk_pos = bbcode.length();

		if (brk_pos > pos) {
			p_rt->add_text(bbcode.substr(pos, brk_pos - pos));
		}

		if (brk_pos == bbcode.length())
			break; //nothing else o add

		int brk_end = bbcode.find("]", brk_pos + 1);

		if (brk_end == -1) {
			p_rt->add_text(bbcode.substr(brk_pos, bbcode.length() - brk_pos));

			break;
		}

		String tag = bbcode.substr(brk_pos + 1, brk_end - brk_pos - 1);

		if (tag.begins_with("/")) {
			bool tag_ok = tag_stack.size() && tag_stack.front()->get() == tag.substr(1, tag.length());
			if (tag_stack.size()) {
			}

			if (!tag_ok) {

				p_rt->add_text("[");
				pos++;
				continue;
			}

			tag_stack.pop_front();
			pos = brk_end + 1;
			if (tag != "/img")
				p_rt->pop();

		} else if (tag.begins_with("method ")) {

			String m = tag.substr(7, tag.length());
			p_rt->push_color(link_color);
			p_rt->push_meta("@" + m);
			p_rt->add_text(m + "()");
			p_rt->pop();
			p_rt->pop();
			pos = brk_end + 1;

		} else if (doc->class_list.has(tag)) {

			p_rt->push_color(link_color);
			p_rt->push_meta("#" + tag);
			p_rt->add_text(tag);
			p_rt->pop();
			p_rt->pop();
			pos = brk_end + 1;

		} else if (tag == "b") {

			//use bold font
			p_rt->push_font(doc_code_font);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "i") {

			//use italics font
			p_rt->push_color(font_color_hl);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "code" || tag == "codeblock") {

			//use monospace font
			p_rt->push_font(doc_code_font);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "center") {

			//use monospace font
			p_rt->push_align(RichTextLabel::ALIGN_CENTER);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "br") {

			//use monospace font
			p_rt->add_newline();
			pos = brk_end + 1;
		} else if (tag == "u") {

			//use underline
			p_rt->push_underline();
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "s") {

			//use strikethrough (not supported underline instead)
			p_rt->push_underline();
			pos = brk_end + 1;
			tag_stack.push_front(tag);

		} else if (tag == "url") {

			//use strikethrough (not supported underline instead)
			int end = bbcode.find("[", brk_end);
			if (end == -1)
				end = bbcode.length();
			String url = bbcode.substr(brk_end + 1, end - brk_end - 1);
			p_rt->push_meta(url);

			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("url=")) {

			String url = tag.substr(4, tag.length());
			p_rt->push_meta(url);
			pos = brk_end + 1;
			tag_stack.push_front("url");
		} else if (tag == "img") {

			//use strikethrough (not supported underline instead)
			int end = bbcode.find("[", brk_end);
			if (end == -1)
				end = bbcode.length();
			String image = bbcode.substr(brk_end + 1, end - brk_end - 1);

			Ref<Texture> texture = ResourceLoader::load(base_path + "/" + image, "Texture");
			if (texture.is_valid())
				p_rt->add_image(texture);

			pos = end;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("color=")) {

			String col = tag.substr(6, tag.length());
			Color color;

			if (col.begins_with("#"))
				color = Color::html(col);
			else if (col == "aqua")
				color = Color::html("#00FFFF");
			else if (col == "black")
				color = Color::html("#000000");
			else if (col == "blue")
				color = Color::html("#0000FF");
			else if (col == "fuchsia")
				color = Color::html("#FF00FF");
			else if (col == "gray" || col == "grey")
				color = Color::html("#808080");
			else if (col == "green")
				color = Color::html("#008000");
			else if (col == "lime")
				color = Color::html("#00FF00");
			else if (col == "maroon")
				color = Color::html("#800000");
			else if (col == "navy")
				color = Color::html("#000080");
			else if (col == "olive")
				color = Color::html("#808000");
			else if (col == "purple")
				color = Color::html("#800080");
			else if (col == "red")
				color = Color::html("#FF0000");
			else if (col == "silver")
				color = Color::html("#C0C0C0");
			else if (col == "teal")
				color = Color::html("#008008");
			else if (col == "white")
				color = Color::html("#FFFFFF");
			else if (col == "yellow")
				color = Color::html("#FFFF00");
			else
				color = Color(0, 0, 0, 1); //base_color;

			p_rt->push_color(color);
			pos = brk_end + 1;
			tag_stack.push_front("color");

		} else if (tag.begins_with("font=")) {

			String fnt = tag.substr(5, tag.length());

			Ref<Font> font = ResourceLoader::load(base_path + "/" + fnt, "Font");
			if (font.is_valid())
				p_rt->push_font(font);
			else {
				p_rt->push_font(doc_font);
			}

			pos = brk_end + 1;
			tag_stack.push_front("font");

		} else {

			p_rt->add_text("["); //ignore
			pos = brk_pos + 1;
		}
	}

	/*p_rt->pop();
	p_rt->pop();
	p_rt->pop();*/
}

void EditorHelp::_add_text(const String &p_bbcode) {

	_add_text_to_rt(p_bbcode, class_desc);
}

void EditorHelp::_update_doc() {
}

void EditorHelp::generate_doc() {

	doc = memnew(DocData);
	doc->generate(true);
	DocData compdoc;
	compdoc.load_compressed(_doc_data_compressed, _doc_data_compressed_size, _doc_data_uncompressed_size);
	doc->merge_from(compdoc); //ensure all is up to date
}

void EditorHelp::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {
			//forward->set_icon(get_icon("Forward","EditorIcons"));
			//back->set_icon(get_icon("Back","EditorIcons"));
			_update_doc();

		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {

			class_desc->add_color_override("selection_color", get_color("text_editor/theme/selection_color", "Editor"));

		} break;

		default: break;
	}
}

void EditorHelp::go_to_help(const String &p_help) {

	_help_callback(p_help);
}

void EditorHelp::go_to_class(const String &p_class, int p_scroll) {

	_goto_desc(p_class, p_scroll);
}

Vector<Pair<String, int> > EditorHelp::get_sections() {
	Vector<Pair<String, int> > sections;

	for (int i = 0; i < section_line.size(); i++) {
		sections.push_back(Pair<String, int>(section_line[i].first, i));
	}
	return sections;
}

void EditorHelp::scroll_to_section(int p_section_index) {
	int line = section_line[p_section_index].second;
	class_desc->scroll_to_line(line);
}

void EditorHelp::popup_search() {

	search_dialog->popup_centered(Size2(250, 80) * EDSCALE);
	search->grab_focus();
}

void EditorHelp::_search_cbk() {

	_search(search->get_text());
}

String EditorHelp::get_class() {

	return edited_class;
}

void EditorHelp::search_again() {
	_search(prev_search);
}

int EditorHelp::get_scroll() const {

	return class_desc->get_v_scroll()->get_value();
}
void EditorHelp::set_scroll(int p_scroll) {

	class_desc->get_v_scroll()->set_value(p_scroll);
}

void EditorHelp::_bind_methods() {

	ClassDB::bind_method("_class_list_select", &EditorHelp::_class_list_select);
	ClassDB::bind_method("_class_desc_select", &EditorHelp::_class_desc_select);
	ClassDB::bind_method("_class_desc_input", &EditorHelp::_class_desc_input);
	//ClassDB::bind_method("_button_pressed",&EditorHelp::_button_pressed);
	ClassDB::bind_method("_request_help", &EditorHelp::_request_help);
	ClassDB::bind_method("_unhandled_key_input", &EditorHelp::_unhandled_key_input);
	ClassDB::bind_method("_search", &EditorHelp::_search);
	ClassDB::bind_method("_search_cbk", &EditorHelp::_search_cbk);
	ClassDB::bind_method("_help_callback", &EditorHelp::_help_callback);

	ADD_SIGNAL(MethodInfo("go_to_help"));
}

EditorHelp::EditorHelp() {

	VBoxContainer *vbc = this;

	EDITOR_DEF("text_editor/help/sort_functions_alphabetically", true);

	//class_list->connect("meta_clicked",this,"_class_list_select");
	//class_list->set_selection_enabled(true);

	{
		class_desc = memnew(RichTextLabel);
		vbc->add_child(class_desc);
		class_desc->set_v_size_flags(SIZE_EXPAND_FILL);
		class_desc->add_color_override("selection_color", get_color("text_editor/theme/selection_color", "Editor"));
		class_desc->connect("meta_clicked", this, "_class_desc_select");
		class_desc->connect("gui_input", this, "_class_desc_input");
	}

	class_desc->set_selection_enabled(true);

	scroll_locked = false;
	select_locked = false;
	set_process_unhandled_key_input(true);
	class_desc->hide();

	search_dialog = memnew(ConfirmationDialog);
	add_child(search_dialog);
	VBoxContainer *search_vb = memnew(VBoxContainer);
	search_dialog->add_child(search_vb);

	search = memnew(LineEdit);
	search_dialog->register_text_enter(search);
	search_vb->add_margin_child(TTR("Search Text"), search);
	search_dialog->get_ok()->set_text(TTR("Find"));
	search_dialog->connect("confirmed", this, "_search_cbk");
	search_dialog->set_hide_on_ok(false);

	/*class_search = memnew( EditorHelpSearch(editor) );
	editor->get_gui_base()->add_child(class_search);
	class_search->connect("go_to_help",this,"_help_callback");*/

	//prev_search_page=-1;
}

EditorHelp::~EditorHelp() {
}

/////////////

void EditorHelpBit::_go_to_help(String p_what) {

	EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
	ScriptEditor::get_singleton()->goto_help(p_what);
	emit_signal("request_hide");
}

void EditorHelpBit::_meta_clicked(String p_select) {

	print_line("got meta " + p_select);
	//print_line("LINK: "+p_select);
	if (p_select.begins_with("$")) { //enum
		//_goto_desc(p_select.substr(1,p_select.length()));
		String select = p_select.substr(1, p_select.length());
		String class_name;
		if (select.find(".") != -1) {
			class_name = select.get_slice(".", 0);
		} else {
			class_name = "@Global";
		}
		_go_to_help("class_enum:" + class_name + ":" + select);
		return;
	} else if (p_select.begins_with("#")) {
		//_goto_desc(p_select.substr(1,p_select.length()));
		_go_to_help("class_name:" + p_select.substr(1, p_select.length()));
		return;
	} else if (p_select.begins_with("@")) {

		String m = p_select.substr(1, p_select.length());

		if (m.find(".") != -1) {
			//must go somewhere else

			_go_to_help("class_method:" + m.get_slice(".", 0) + ":" + m.get_slice(".", 0));
		} else {
			/*
			if (!method_line.has(m))
				return;
			class_desc->scroll_to_line(method_line[m]);
			*/
		}
	}
}

void EditorHelpBit::_bind_methods() {

	ClassDB::bind_method("_meta_clicked", &EditorHelpBit::_meta_clicked);
	ADD_SIGNAL(MethodInfo("request_hide"));
}

void EditorHelpBit::_notification(int p_what) {

	switch (p_what) {
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {

			rich_text->add_color_override("selection_color", get_color("text_editor/theme/selection_color", "Editor"));
		} break;

		default: break;
	}
}

void EditorHelpBit::set_text(const String &p_text) {

	rich_text->clear();
	_add_text_to_rt(p_text, rich_text);
}

EditorHelpBit::EditorHelpBit() {

	rich_text = memnew(RichTextLabel);
	add_child(rich_text);
	rich_text->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	rich_text->connect("meta_clicked", this, "_meta_clicked");
	rich_text->add_color_override("selection_color", get_color("text_editor/theme/selection_color", "Editor"));
	rich_text->set_override_selected_font_color(false);
	set_custom_minimum_size(Size2(0, 70 * EDSCALE));
}
