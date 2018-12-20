/*************************************************************************/
/*  editor_help.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/os/keyboard.h"
#include "doc_data_compressed.gen.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor_node.h"
#include "editor_settings.h"

#define CONTRIBUTE_URL "https://docs.godotengine.org/en/latest/community/contributing/updating_the_class_reference.html"
#define CONTRIBUTE2_URL "https://github.com/godotengine/godot-docs"
#define REQUEST_URL "https://github.com/godotengine/godot-docs/issues/new"

void EditorHelpSearch::popup_dialog() {

	popup_centered(Size2(700, 600) * EDSCALE);
	if (search_box->get_text() != "") {
		search_box->select_all();
		_update_search();
	}
	search_box->grab_focus();
}

void EditorHelpSearch::popup_dialog(const String &p_term) {

	popup_centered(Size2(700, 600) * EDSCALE);
	if (p_term != "") {
		search_box->set_text(p_term);
		search_box->select_all();
		_update_search();
	} else {
		search_box->clear();
	}
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

void EditorHelpSearch::IncrementalSearch::phase1(Map<String, DocData::ClassDoc>::Element *E) {

	if (E->key().findn(term) != -1) {

		TreeItem *item = search_options->create_item(root);
		item->set_metadata(0, "class_name:" + E->key());
		item->set_text(0, E->key() + " (Class)");
		Ref<Texture> icon = EditorNode::get_singleton()->get_class_icon(E->key(), "Node");
		item->set_icon(0, icon);
	}
}

void EditorHelpSearch::IncrementalSearch::phase2(Map<String, DocData::ClassDoc>::Element *E) {

	DocData::ClassDoc &c = E->get();

	Ref<Texture> cicon = EditorNode::get_singleton()->get_class_icon(E->key(), "Node");

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

bool EditorHelpSearch::IncrementalSearch::slice() {

	if (phase > 2)
		return true;

	if (iterator) {

		switch (phase) {

			case 1: {
				phase1(iterator);
			} break;
			case 2: {
				phase2(iterator);
			} break;
			default: {
				WARN_PRINT("illegal phase in IncrementalSearch");
				return true;
			}
		}

		iterator = iterator->next();
	} else {

		phase += 1;
		iterator = doc->class_list.front();
	}

	return false;
}

EditorHelpSearch::IncrementalSearch::IncrementalSearch(EditorHelpSearch *p_search, Tree *p_search_options, const String &p_term) :
		search(p_search),
		search_options(p_search_options) {

	def_icon = search->get_icon("Node", "EditorIcons");
	doc = EditorHelp::get_doc_data();

	term = p_term;

	root = search_options->create_item();
	phase = 0;
	iterator = 0;
}

bool EditorHelpSearch::IncrementalSearch::empty() const {

	return root->get_children() == NULL;
}

bool EditorHelpSearch::IncrementalSearch::work(uint64_t slot) {

	const uint64_t until = OS::get_singleton()->get_ticks_usec() + slot;

	while (!slice()) {

		if (OS::get_singleton()->get_ticks_usec() > until)
			return false;
	}

	return true;
}

void EditorHelpSearch::_update_search() {
	search_options->clear();

	String term = search_box->get_text();
	if (term.length() < 2)
		return;

	search = Ref<IncrementalSearch>(memnew(IncrementalSearch(this, search_options, term)));
	set_process(true);
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
		search_box->set_right_icon(get_icon("Search", "EditorIcons"));
		search_box->set_clear_button_enabled(true);

		connect("confirmed", this, "_confirmed");
		_update_search();
	} else if (p_what == NOTIFICATION_EXIT_TREE) {
		disconnect("confirmed", this, "_confirmed");
	} else if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {

		if (is_visible_in_tree()) {

			search_box->call_deferred("grab_focus"); // still not visible
			search_box->select_all();
		}
	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {

		//_update_icons
		search_box->set_right_icon(get_icon("Search", "EditorIcons"));
		search_box->set_clear_button_enabled(true);
	} else if (p_what == NOTIFICATION_PROCESS) {

		if (search.is_valid()) {

			if (search->work()) {

				get_ok()->set_disabled(search->empty());
				search = Ref<IncrementalSearch>();
				set_process(false);
			}
		} else {

			set_process(false);
		}
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
	search_options->set_hide_root(true);
	vbc->add_margin_child(TTR("Matches:"), search_options, true);
	get_ok()->set_text(TTR("Open"));
	get_ok()->set_disabled(true);
	register_text_enter(search_box);
	set_hide_on_ok(false);
	search_options->connect("item_activated", this, "_confirmed");
	set_title(TTR("Search Help"));
}

/////////////////////////////////

void EditorHelpIndex::add_type(const String &p_type, HashMap<String, TreeItem *> &p_types, TreeItem *p_root) {

	if (p_types.has(p_type))
		return;

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

	Ref<Texture> icon = EditorNode::get_singleton()->get_class_icon(p_type);
	item->set_icon(0, icon);

	p_types[p_type] = item;
}

void EditorHelpIndex::_tree_item_selected() {

	TreeItem *s = class_list->get_selected();
	if (!s)
		return;

	EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
	emit_signal("open_class", s->get_text(0));
	hide();
}

void EditorHelpIndex::select_class(const String &p_class) {

	if (!tree_item_map.has(p_class))
		return;
	tree_item_map[p_class]->select(0);
	class_list->ensure_cursor_is_visible();
}

void EditorHelpIndex::popup_dialog() {

	popup_centered(Size2(500, 600) * EDSCALE);

	search_box->set_text("");
	_update_class_list();
}

void EditorHelpIndex::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		//_update_icons
		search_box->set_right_icon(get_icon("Search", "EditorIcons"));
		search_box->set_clear_button_enabled(true);
		_update_class_list();

		connect("confirmed", this, "_tree_item_selected");

	} else if (p_what == NOTIFICATION_POST_POPUP) {

		search_box->call_deferred("grab_focus");
	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {

		//_update_icons
		search_box->set_right_icon(get_icon("Search", "EditorIcons"));
		search_box->set_clear_button_enabled(true);

		bool enable_rl = EditorSettings::get_singleton()->get("docks/scene_tree/draw_relationship_lines");
		Color rl_color = EditorSettings::get_singleton()->get("docks/scene_tree/relationship_line_color");

		if (enable_rl) {
			class_list->add_constant_override("draw_relationship_lines", 1);
			class_list->add_color_override("relationship_line_color", rl_color);
		} else {
			class_list->add_constant_override("draw_relationship_lines", 0);
		}
	}
}

void EditorHelpIndex::_text_changed(const String &p_text) {

	_update_class_list();
}

void EditorHelpIndex::_update_class_list() {

	class_list->clear();
	tree_item_map.clear();
	TreeItem *root = class_list->create_item();

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

					if (to_select.empty() || type.length() < to_select.length()) {
						to_select = type;
					}

					found = true;
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
	class_list->set_hide_root(true);
	class_list->set_v_size_flags(SIZE_EXPAND_FILL);

	class_list->connect("item_activated", this, "_tree_item_selected");

	bool enable_rl = EditorSettings::get_singleton()->get("docks/scene_tree/draw_relationship_lines");
	Color rl_color = EditorSettings::get_singleton()->get("docks/scene_tree/relationship_line_color");

	if (enable_rl) {
		class_list->add_constant_override("draw_relationship_lines", 1);
		class_list->add_color_override("relationship_line_color", rl_color);
	} else {
		class_list->add_constant_override("draw_relationship_lines", 0);
	}

	get_ok()->set_text(TTR("Open"));
	set_title(TTR("Search Classes"));
}

/////////////////////////////////

DocData *EditorHelp::doc = NULL;

void EditorHelp::_init_colors() {

	title_color = get_color("accent_color", "Editor");
	text_color = get_color("default_color", "RichTextLabel");
	headline_color = get_color("headline_color", "EditorHelp");
	base_type_color = title_color.linear_interpolate(text_color, 0.5);
	comment_color = Color(text_color.r, text_color.g, text_color.b, 0.6);
	symbol_color = comment_color;
	value_color = Color(text_color.r, text_color.g, text_color.b, 0.4);
	qualifier_color = Color(text_color.r, text_color.g, text_color.b, 0.8);
	type_color = get_color("accent_color", "Editor").linear_interpolate(text_color, 0.5);
}

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

	find_bar->search_next();
}

void EditorHelp::_class_list_select(const String &p_select) {

	_goto_desc(p_select);
}

void EditorHelp::_class_desc_select(const String &p_select) {

	if (p_select.begins_with("$")) { //enum
		String select = p_select.substr(1, p_select.length());
		String class_name;
		if (select.find(".") != -1) {
			class_name = select.get_slice(".", 0);
			select = select.get_slice(".", 1);
		} else {
			class_name = "@GlobalScope";
		}
		emit_signal("go_to_help", "class_enum:" + class_name + ":" + select);
		return;
	} else if (p_select.begins_with("#")) {
		emit_signal("go_to_help", "class_name:" + p_select.substr(1, p_select.length()));
		return;
	} else if (p_select.begins_with("@")) {
		String tag = p_select.substr(1, 6);
		String link = p_select.substr(7, p_select.length());

		String topic;
		Map<String, int> *table = NULL;

		if (tag == "method") {
			topic = "class_method";
			table = &this->method_line;
		} else if (tag == "member") {
			topic = "class_property";
			table = &this->property_line;
		} else if (tag == "enum  ") {
			topic = "class_enum";
			table = &this->enum_line;
		} else if (tag == "signal") {
			topic = "class_signal";
			table = &this->signal_line;
		} else {
			return;
		}

		if (link.find(".") != -1) {

			emit_signal("go_to_help", topic + ":" + link.get_slice(".", 0) + ":" + link.get_slice(".", 1));
		} else {

			if (!table->has(link))
				return;
			class_desc->scroll_to_line((*table)[link]);
		}
	} else if (p_select.begins_with("http")) {
		OS::get_singleton()->shell_open(p_select);
	}
}

void EditorHelp::_class_desc_input(const Ref<InputEvent> &p_input) {
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

String EditorHelp::_fix_constant(const String &p_constant) const {

	if (p_constant.strip_edges() == "4294967295") {
		return "0xFFFFFFFF";
	}

	if (p_constant.strip_edges() == "2147483647") {
		return "0x7FFFFFFF";
	}
	if (p_constant.strip_edges() == "1048575") {
		return "0xfffff";
	}

	return p_constant;
}

void EditorHelp::_add_method(const DocData::MethodDoc &p_method, bool p_overview) {

	method_line[p_method.name] = class_desc->get_line_count() - 2; //gets overridden if description

	const bool is_vararg = p_method.qualifiers.find("vararg") != -1;

	if (p_overview) {
		class_desc->push_cell();
		class_desc->push_align(RichTextLabel::ALIGN_RIGHT);
	}

	_add_type(p_method.return_type, p_method.return_enum);

	if (p_overview) {
		class_desc->pop(); //align
		class_desc->pop(); //cell
		class_desc->push_cell();
	} else {
		class_desc->add_text(" ");
	}

	if (p_overview && p_method.description != "") {
		class_desc->push_meta("@method" + p_method.name);
	}

	class_desc->push_color(headline_color);
	_add_text(p_method.name);
	class_desc->pop();

	if (p_overview && p_method.description != "") {
		class_desc->pop(); //meta
	}

	class_desc->push_color(symbol_color);
	class_desc->add_text(p_method.arguments.size() || is_vararg ? "( " : "(");
	class_desc->pop();

	for (int j = 0; j < p_method.arguments.size(); j++) {
		class_desc->push_color(text_color);
		if (j > 0)
			class_desc->add_text(", ");
		_add_type(p_method.arguments[j].type, p_method.arguments[j].enumeration);
		class_desc->add_text(" ");
		_add_text(p_method.arguments[j].name);
		if (p_method.arguments[j].default_value != "") {

			class_desc->push_color(symbol_color);
			class_desc->add_text("=");
			class_desc->pop();
			_add_text(_fix_constant(p_method.arguments[j].default_value));
		}

		class_desc->pop();
	}

	if (is_vararg) {
		class_desc->push_color(text_color);
		if (p_method.arguments.size())
			class_desc->add_text(", ");
		class_desc->push_color(symbol_color);
		class_desc->add_text("...");
		class_desc->pop();
		class_desc->pop();
	}

	class_desc->push_color(symbol_color);
	class_desc->add_text(p_method.arguments.size() || is_vararg ? " )" : ")");
	class_desc->pop();
	if (p_method.qualifiers != "") {

		class_desc->push_color(qualifier_color);
		class_desc->add_text(" ");
		_add_text(p_method.qualifiers);
		class_desc->pop();
	}

	if (p_overview)
		class_desc->pop(); //cell
}

Error EditorHelp::_goto_desc(const String &p_class, int p_vscr) {

	if (!doc->class_list.has(p_class))
		return ERR_DOES_NOT_EXIST;

	select_locked = true;

	class_desc->show();

	description_line = 0;

	if (p_class == edited_class)
		return OK; //already there

	edited_class = p_class;
	_update_doc();
	return OK;
}

void EditorHelp::_update_doc() {
	if (!doc->class_list.has(edited_class))
		return;

	scroll_locked = true;

	class_desc->clear();
	method_line.clear();
	section_line.clear();

	_init_colors();

	DocData::ClassDoc cd = doc->class_list[edited_class]; //make a copy, so we can sort without worrying

	Ref<Font> doc_font = get_font("doc", "EditorFonts");
	Ref<Font> doc_title_font = get_font("doc_title", "EditorFonts");
	Ref<Font> doc_code_font = get_font("doc_source", "EditorFonts");
	String link_color_text = title_color.to_html(false);

	// Class name
	section_line.push_back(Pair<String, int>(TTR("Top"), 0));
	class_desc->push_font(doc_title_font);
	class_desc->push_color(title_color);
	class_desc->add_text(TTR("Class:") + " ");
	class_desc->push_color(headline_color);
	_add_text(edited_class);
	class_desc->pop();
	class_desc->pop();
	class_desc->pop();
	class_desc->add_newline();

	// Inheritance tree

	// Ascendents
	if (cd.inherits != "") {

		class_desc->push_color(title_color);
		class_desc->push_font(doc_font);
		class_desc->add_text(TTR("Inherits:") + " ");
		class_desc->pop();

		String inherits = cd.inherits;

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

	// Descendents
	if (ClassDB::class_exists(cd.name)) {

		bool found = false;
		bool prev = false;

		for (Map<String, DocData::ClassDoc>::Element *E = doc->class_list.front(); E; E = E->next()) {

			if (E->get().inherits == cd.name) {

				if (!found) {
					class_desc->push_color(title_color);
					class_desc->push_font(doc_font);
					class_desc->add_text(TTR("Inherited by:") + " ");
					class_desc->pop();
					found = true;
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

	// Brief description
	if (cd.brief_description != "") {

		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Brief Description:"));
		class_desc->pop();
		class_desc->pop();

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

	// Properties overview
	Set<String> skip_methods;
	bool property_descr = false;

	if (cd.properties.size()) {

		section_line.push_back(Pair<String, int>(TTR("Properties"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Properties:"));
		class_desc->pop();
		class_desc->pop();

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
				class_desc->push_meta("@member" + cd.properties[i].name);
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

	// Methods overview
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

		section_line.push_back(Pair<String, int>(TTR("Methods"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Methods:"));
		class_desc->pop();
		class_desc->pop();

		class_desc->push_font(doc_code_font);
		class_desc->push_indent(1);
		class_desc->push_table(2);
		class_desc->set_table_column_expand(1, 1);

		bool any_previous = false;
		for (int pass = 0; pass < 2; pass++) {
			Vector<DocData::MethodDoc> m;

			for (int i = 0; i < methods.size(); i++) {
				const String &q = methods[i].qualifiers;
				if ((pass == 0 && q.find("virtual") != -1) || (pass == 1 && q.find("virtual") == -1)) {
					m.push_back(methods[i]);
				}
			}

			if (any_previous && !m.empty()) {
				class_desc->push_cell();
				class_desc->pop(); //cell
				class_desc->push_cell();
				class_desc->pop(); //cell
				any_previous = false;
			}

			String group_prefix;
			for (int i = 0; i < m.size(); i++) {
				const String new_prefix = m[i].name.substr(0, 3);
				bool is_new_group = false;

				if (i < m.size() - 1 && new_prefix == m[i + 1].name.substr(0, 3) && new_prefix != group_prefix) {
					is_new_group = i > 0;
					group_prefix = new_prefix;
				} else if (group_prefix != "" && new_prefix != group_prefix) {
					is_new_group = true;
					group_prefix = "";
				}

				if (is_new_group && pass == 1) {
					class_desc->push_cell();
					class_desc->pop(); //cell
					class_desc->push_cell();
					class_desc->pop(); //cell
				}

				if (m[i].description != "") {
					method_descr = true;
				}

				_add_method(m[i], true);
			}

			any_previous = !m.empty();
		}

		class_desc->pop(); //table
		class_desc->pop();
		class_desc->pop(); // font
		class_desc->add_newline();
		class_desc->add_newline();
	}

	// Theme properties
	if (cd.theme_properties.size()) {

		section_line.push_back(Pair<String, int>(TTR("Theme Properties"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Theme Properties:"));
		class_desc->pop();
		class_desc->pop();

		class_desc->push_indent(1);
		class_desc->push_table(2);
		class_desc->set_table_column_expand(1, 1);

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

	// Signals
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
		class_desc->add_newline();

		class_desc->push_indent(1);

		for (int i = 0; i < cd.signals.size(); i++) {

			signal_line[cd.signals[i].name] = class_desc->get_line_count() - 2; //gets overridden if description
			class_desc->push_font(doc_code_font); // monofont
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

				class_desc->push_font(doc_font);
				class_desc->push_color(comment_color);
				class_desc->push_indent(1);
				_add_text(cd.signals[i].description);
				class_desc->pop(); // indent
				class_desc->pop();
				class_desc->pop(); // font
			}
			class_desc->add_newline();
			class_desc->add_newline();
		}

		class_desc->pop();
		class_desc->add_newline();
	}

	// Constants and enums
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

		// Enums
		if (enums.size()) {

			section_line.push_back(Pair<String, int>(TTR("Enumerations"), class_desc->get_line_count() - 2));
			class_desc->push_color(title_color);
			class_desc->push_font(doc_title_font);
			class_desc->add_text(TTR("Enumerations:"));
			class_desc->pop();
			class_desc->pop();
			class_desc->push_indent(1);

			class_desc->add_newline();

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

				Map<String, int> enumValuesContainer;
				int enumStartingLine = enum_line[E->key()];

				for (int i = 0; i < enum_list.size(); i++) {
					if (cd.name == "@GlobalScope")
						enumValuesContainer[enum_list[i].name] = enumStartingLine;

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
						//class_desc->add_text("  ");
						class_desc->push_indent(1);
						class_desc->push_color(comment_color);
						_add_text(enum_list[i].description);
						class_desc->pop();
						class_desc->pop();
						class_desc->pop(); // indent
						class_desc->add_newline();
					}

					class_desc->add_newline();
				}

				if (cd.name == "@GlobalScope")
					enum_values_line[E->key()] = enumValuesContainer;

				class_desc->pop();

				class_desc->add_newline();
			}

			class_desc->pop();
			class_desc->add_newline();
		}

		// Constants
		if (constants.size()) {

			section_line.push_back(Pair<String, int>(TTR("Constants"), class_desc->get_line_count() - 2));
			class_desc->push_color(title_color);
			class_desc->push_font(doc_title_font);
			class_desc->add_text(TTR("Constants:"));
			class_desc->pop();
			class_desc->pop();
			class_desc->push_indent(1);

			class_desc->add_newline();

			for (int i = 0; i < constants.size(); i++) {

				constant_line[constants[i].name] = class_desc->get_line_count() - 2;
				class_desc->push_font(doc_code_font);

				if (constants[i].value.begins_with("Color(") && constants[i].value.ends_with(")")) {
					String stripped = constants[i].value.replace(" ", "").replace("Color(", "").replace(")", "");
					Vector<float> color = stripped.split_floats(",");
					if (color.size() >= 3) {
						class_desc->push_color(Color(color[0], color[1], color[2]));
						static const CharType prefix[3] = { 0x25CF /* filled circle */, ' ', 0 };
						class_desc->add_text(String(prefix));
						class_desc->pop();
					}
				}

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
					class_desc->push_indent(1);
					class_desc->push_color(comment_color);
					_add_text(constants[i].description);
					class_desc->pop();
					class_desc->pop();
					class_desc->pop(); // indent
					class_desc->add_newline();
				}

				class_desc->add_newline();
			}

			class_desc->pop();
			class_desc->add_newline();
		}
	}

	// Class description
	if (cd.description != "") {

		section_line.push_back(Pair<String, int>(TTR("Class Description"), class_desc->get_line_count() - 2));
		description_line = class_desc->get_line_count() - 2;
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Class Description:"));
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

	// Online tutorials
	{
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Online Tutorials:"));
		class_desc->pop();
		class_desc->pop();
		class_desc->push_indent(1);

		class_desc->push_font(doc_code_font);

		class_desc->add_newline();
		//	class_desc->add_newline();

		if (cd.tutorials.size() != 0) {

			for (int i = 0; i < cd.tutorials.size(); i++) {
				String link = cd.tutorials[i];
				String linktxt = link;
				int seppos = linktxt.find("//");
				if (seppos != -1) {
					linktxt = link.right(seppos + 2);
				}

				class_desc->push_color(symbol_color);
				class_desc->append_bbcode("[url=" + link + "]" + linktxt + "[/url]");
				class_desc->pop();
				class_desc->add_newline();
			}
		} else {
			class_desc->push_color(comment_color);
			class_desc->append_bbcode(TTR("There are currently no tutorials for this class, you can [color=$color][url=$url]contribute one[/url][/color] or [color=$color][url=$url2]request one[/url][/color].").replace("$url2", REQUEST_URL).replace("$url", CONTRIBUTE2_URL).replace("$color", link_color_text));
			class_desc->pop();
		}
		class_desc->pop();
		class_desc->pop();
		class_desc->add_newline();
		class_desc->add_newline();
	}

	// Property descriptions
	if (property_descr) {

		section_line.push_back(Pair<String, int>(TTR("Property Descriptions"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Property Descriptions:"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();

		for (int i = 0; i < cd.properties.size(); i++) {

			property_line[cd.properties[i].name] = class_desc->get_line_count() - 2;

			class_desc->push_table(2);
			class_desc->set_table_column_expand(1, 1);

			class_desc->push_cell();
			class_desc->push_font(doc_code_font);
			_add_type(cd.properties[i].type, cd.properties[i].enumeration);
			class_desc->add_text(" ");
			class_desc->pop(); // font
			class_desc->pop(); // cell

			class_desc->push_cell();
			class_desc->push_font(doc_code_font);
			class_desc->push_color(headline_color);
			_add_text(cd.properties[i].name);
			class_desc->pop(); // color
			class_desc->pop(); // font
			class_desc->pop(); // cell

			if (cd.properties[i].setter != "") {

				class_desc->push_cell();
				class_desc->pop(); // cell

				class_desc->push_cell();
				class_desc->push_font(doc_code_font);
				class_desc->push_color(text_color);
				class_desc->add_text(cd.properties[i].setter + "(value)");
				class_desc->pop(); // color
				class_desc->push_color(comment_color);
				class_desc->add_text(" setter");
				class_desc->pop(); // color
				class_desc->pop(); // font
				class_desc->pop(); // cell
			}

			if (cd.properties[i].getter != "") {

				class_desc->push_cell();
				class_desc->pop(); // cell

				class_desc->push_cell();
				class_desc->push_font(doc_code_font);
				class_desc->push_color(text_color);
				class_desc->add_text(cd.properties[i].getter + "()");
				class_desc->pop(); //color
				class_desc->push_color(comment_color);
				class_desc->add_text(" getter");
				class_desc->pop(); //color
				class_desc->pop(); //font
				class_desc->pop(); //cell
			}

			class_desc->pop(); // table

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

	// Method descriptions
	if (method_descr) {

		section_line.push_back(Pair<String, int>(TTR("Method Descriptions"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Method Descriptions:"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();

		for (int i = 0; i < methods.size(); i++) {

			class_desc->push_font(doc_code_font);
			_add_method(methods[i], false);
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
		if (enum_line.has(name))
			line = enum_line[name];
	} else if (what == "class_theme_item") {
		if (theme_property_line.has(name))
			line = theme_property_line[name];
	} else if (what == "class_constant") {
		if (constant_line.has(name))
			line = constant_line[name];
	} else if (what == "class_global") {
		if (constant_line.has(name))
			line = constant_line[name];
		else {
			Map<String, Map<String, int> >::Element *iter = enum_values_line.front();
			while (true) {
				if (iter->value().has(name)) {
					line = iter->value()[name];
					break;
				} else if (iter == enum_values_line.back())
					break;
				else
					iter = iter->next();
			}
		}
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

	String bbcode = p_bbcode.dedent().replace("\t", "").replace("\r", "").strip_edges();

	List<String> tag_stack;
	bool code_tag = false;

	int pos = 0;
	while (pos < bbcode.length()) {

		int brk_pos = bbcode.find("[", pos);

		if (brk_pos < 0)
			brk_pos = bbcode.length();

		if (brk_pos > pos) {
			String text = bbcode.substr(pos, brk_pos - pos);
			if (!code_tag)
				text = text.replace("\n", "\n\n");
			p_rt->add_text(text);
		}

		if (brk_pos == bbcode.length())
			break; //nothing else to add

		int brk_end = bbcode.find("]", brk_pos + 1);

		if (brk_end == -1) {

			String text = bbcode.substr(brk_pos, bbcode.length() - brk_pos);
			if (!code_tag)
				text = text.replace("\n", "\n\n");
			p_rt->add_text(text);

			break;
		}

		String tag = bbcode.substr(brk_pos + 1, brk_end - brk_pos - 1);

		if (tag.begins_with("/")) {
			bool tag_ok = tag_stack.size() && tag_stack.front()->get() == tag.substr(1, tag.length());

			if (!tag_ok) {

				p_rt->add_text("[");
				pos = brk_pos + 1;
				continue;
			}

			tag_stack.pop_front();
			pos = brk_end + 1;
			code_tag = false;
			if (tag != "/img")
				p_rt->pop();
		} else if (code_tag) {

			p_rt->add_text("[");
			pos = brk_pos + 1;

		} else if (tag.begins_with("method ") || tag.begins_with("member ") || tag.begins_with("signal ") || tag.begins_with("enum ")) {

			String link_target = tag.substr(tag.find(" ") + 1, tag.length());
			String link_tag = tag.substr(0, tag.find(" ")).rpad(6);
			p_rt->push_color(link_color);
			p_rt->push_meta("@" + link_tag + link_target);
			p_rt->add_text(link_target + (tag.begins_with("method ") ? "()" : ""));
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
			code_tag = true;
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
}

void EditorHelp::_add_text(const String &p_bbcode) {

	_add_text_to_rt(p_bbcode, class_desc);
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

			_update_doc();

		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {

			class_desc->add_color_override("selection_color", EditorSettings::get_singleton()->get("text_editor/theme/selection_color"));
			_update_doc();

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

	find_bar->popup_search();
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
	ClassDB::bind_method("_request_help", &EditorHelp::_request_help);
	ClassDB::bind_method("_unhandled_key_input", &EditorHelp::_unhandled_key_input);
	ClassDB::bind_method("_search", &EditorHelp::_search);
	ClassDB::bind_method("_help_callback", &EditorHelp::_help_callback);

	ADD_SIGNAL(MethodInfo("go_to_help"));
}

EditorHelp::EditorHelp() {

	set_custom_minimum_size(Size2(150 * EDSCALE, 0));

	EDITOR_DEF("text_editor/help/sort_functions_alphabetically", true);

	find_bar = memnew(FindBar);
	add_child(find_bar);
	find_bar->hide();

	class_desc = memnew(RichTextLabel);
	add_child(class_desc);
	class_desc->set_v_size_flags(SIZE_EXPAND_FILL);
	class_desc->add_color_override("selection_color", EditorSettings::get_singleton()->get("text_editor/theme/selection_color"));
	class_desc->connect("meta_clicked", this, "_class_desc_select");
	class_desc->connect("gui_input", this, "_class_desc_input");

	find_bar->set_rich_text_label(class_desc);

	class_desc->set_selection_enabled(true);

	scroll_locked = false;
	select_locked = false;
	//set_process_unhandled_key_input(true);
	class_desc->hide();
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

	if (p_select.begins_with("$")) { //enum

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

		_go_to_help("class_name:" + p_select.substr(1, p_select.length()));
		return;
	} else if (p_select.begins_with("@")) {

		String m = p_select.substr(1, p_select.length());

		if (m.find(".") != -1)
			_go_to_help("class_method:" + m.get_slice(".", 0) + ":" + m.get_slice(".", 0)); //must go somewhere else
	}
}

void EditorHelpBit::_bind_methods() {

	ClassDB::bind_method("_meta_clicked", &EditorHelpBit::_meta_clicked);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &EditorHelpBit::set_text);
	ADD_SIGNAL(MethodInfo("request_hide"));
}

void EditorHelpBit::_notification(int p_what) {

	switch (p_what) {
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {

			rich_text->add_color_override("selection_color", EditorSettings::get_singleton()->get("text_editor/theme/selection_color"));
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
	//rich_text->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	rich_text->connect("meta_clicked", this, "_meta_clicked");
	rich_text->add_color_override("selection_color", EditorSettings::get_singleton()->get("text_editor/theme/selection_color"));
	rich_text->set_override_selected_font_color(false);
	set_custom_minimum_size(Size2(0, 70 * EDSCALE));
}

FindBar::FindBar() {

	container = memnew(Control);
	add_child(container);

	container->set_clip_contents(true);
	container->set_h_size_flags(SIZE_EXPAND_FILL);

	hbc = memnew(HBoxContainer);
	container->add_child(hbc);

	vbc_search_text = memnew(VBoxContainer);
	hbc->add_child(vbc_search_text);
	vbc_search_text->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->set_anchor_and_margin(MARGIN_RIGHT, 1, 0);

	search_text = memnew(LineEdit);
	vbc_search_text->add_child(search_text);
	search_text->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	search_text->connect("text_changed", this, "_search_text_changed");
	search_text->connect("text_entered", this, "_search_text_entered");

	find_prev = memnew(ToolButton);
	hbc->add_child(find_prev);
	find_prev->set_focus_mode(FOCUS_NONE);
	find_prev->connect("pressed", this, "_search_prev");

	find_next = memnew(ToolButton);
	hbc->add_child(find_next);
	find_next->set_focus_mode(FOCUS_NONE);
	find_next->connect("pressed", this, "_search_next");

	error_label = memnew(Label);
	hbc->add_child(error_label);
	error_label->add_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_color("error_color", "Editor"));

	hide_button = memnew(TextureButton);
	add_child(hide_button);
	hide_button->set_focus_mode(FOCUS_NONE);
	hide_button->set_expand(true);
	hide_button->set_stretch_mode(TextureButton::STRETCH_KEEP_CENTERED);
	hide_button->connect("pressed", this, "_hide_pressed");
}

void FindBar::popup_search() {

	show();
	bool grabbed_focus = false;
	if (!search_text->has_focus()) {
		search_text->grab_focus();
		grabbed_focus = true;
	}

	if (!search_text->get_text().empty()) {
		search_text->select_all();
		search_text->set_cursor_position(search_text->get_text().length());
		if (grabbed_focus) {
			_search();
		}
	}

	call_deferred("_update_size");
}

void FindBar::_update_size() {

	container->set_custom_minimum_size(Size2(0, hbc->get_size().height));
}

void FindBar::_notification(int p_what) {

	if (p_what == NOTIFICATION_READY) {

		find_prev->set_icon(get_icon("MoveUp", "EditorIcons"));
		find_next->set_icon(get_icon("MoveDown", "EditorIcons"));
		hide_button->set_normal_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_hover_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_pressed_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_custom_minimum_size(hide_button->get_normal_texture()->get_size());
	} else if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {

		set_process_unhandled_input(is_visible_in_tree());
	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {

		find_prev->set_icon(get_icon("MoveUp", "EditorIcons"));
		find_next->set_icon(get_icon("MoveDown", "EditorIcons"));
		hide_button->set_normal_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_hover_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_pressed_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_custom_minimum_size(hide_button->get_normal_texture()->get_size());
	}
}

void FindBar::_bind_methods() {

	ClassDB::bind_method("_unhandled_input", &FindBar::_unhandled_input);

	ClassDB::bind_method("_search_text_changed", &FindBar::_search_text_changed);
	ClassDB::bind_method("_search_text_entered", &FindBar::_search_text_entered);
	ClassDB::bind_method("_search_next", &FindBar::search_next);
	ClassDB::bind_method("_search_prev", &FindBar::search_prev);
	ClassDB::bind_method("_hide_pressed", &FindBar::_hide_bar);
	ClassDB::bind_method("_update_size", &FindBar::_update_size);

	ADD_SIGNAL(MethodInfo("search"));
}

void FindBar::set_rich_text_label(RichTextLabel *p_rich_text_label) {

	rich_text_label = p_rich_text_label;
}

bool FindBar::search_next() {

	return _search();
}

bool FindBar::search_prev() {

	return _search(true);
}

bool FindBar::_search(bool p_search_previous) {

	String stext = search_text->get_text();
	bool keep = prev_search == stext;

	bool ret = rich_text_label->search(stext, keep, p_search_previous);
	if (!ret) {
		ret = rich_text_label->search(stext, false, p_search_previous);
	}

	prev_search = stext;

	if (ret) {
		set_error("");
	} else {
		set_error(stext.empty() ? "" : TTR("No Matches"));
	}

	return ret;
}

void FindBar::set_error(const String &p_label) {

	error_label->set_text(p_label);
}

void FindBar::_hide_bar() {

	if (search_text->has_focus())
		rich_text_label->grab_focus();

	hide();
}

void FindBar::_unhandled_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventKey> k = p_event;
	if (k.is_valid()) {

		if (k->is_pressed() && (rich_text_label->has_focus() || hbc->is_a_parent_of(get_focus_owner()))) {

			bool accepted = true;

			switch (k->get_scancode()) {

				case KEY_ESCAPE: {

					_hide_bar();
				} break;
				default: {

					accepted = false;
				} break;
			}

			if (accepted) {
				accept_event();
			}
		}
	}
}

void FindBar::_search_text_changed(const String &p_text) {

	search_next();
}

void FindBar::_search_text_entered(const String &p_text) {

	search_next();
}
