/*************************************************************************/
/*  editor_help.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "core/version_generated.gen.h"
#include "doc_data_compressed.gen.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "editor_settings.h"

#define CONTRIBUTE_URL vformat("%s/community/contributing/updating_the_class_reference.html", VERSION_DOCS_URL)

DocData *EditorHelp::doc = nullptr;

void EditorHelp::_init_colors() {
	title_color = get_color("accent_color", "Editor");
	text_color = get_color("default_color", "RichTextLabel");
	headline_color = get_color("headline_color", "EditorHelp");
	base_type_color = title_color.linear_interpolate(text_color, 0.5);
	comment_color = text_color * Color(1, 1, 1, 0.6);
	symbol_color = comment_color;
	value_color = text_color * Color(1, 1, 1, 0.6);
	qualifier_color = text_color * Color(1, 1, 1, 0.8);
	type_color = get_color("accent_color", "Editor").linear_interpolate(text_color, 0.5);
	class_desc->add_color_override("selection_color", get_color("accent_color", "Editor") * Color(1, 1, 1, 0.4));
	class_desc->add_constant_override("line_separation", Math::round(5 * EDSCALE));
}

void EditorHelp::_unhandled_key_input(const Ref<InputEvent> &p_ev) {
	if (!is_visible_in_tree()) {
		return;
	}

	Ref<InputEventKey> k = p_ev;

	if (k.is_valid() && k->get_control() && k->get_scancode() == KEY_F) {
		search->grab_focus();
		search->select_all();
	}
}

void EditorHelp::_search(bool p_search_previous) {
	if (p_search_previous) {
		find_bar->search_prev();
	} else {
		find_bar->search_next();
	}
}

void EditorHelp::_class_list_select(const String &p_select) {
	_goto_desc(p_select);
}

void EditorHelp::_class_desc_select(const String &p_select) {
	if (p_select.begins_with("$")) { // enum
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
		int tag_end = p_select.find(" ");

		String tag = p_select.substr(1, tag_end - 1);
		String link = p_select.substr(tag_end + 1, p_select.length()).lstrip(" ");

		String topic;
		Map<String, int> *table = nullptr;

		if (tag == "method") {
			topic = "class_method";
			table = &this->method_line;
		} else if (tag == "member") {
			topic = "class_property";
			table = &this->property_line;
		} else if (tag == "enum") {
			topic = "class_enum";
			table = &this->enum_line;
		} else if (tag == "signal") {
			topic = "class_signal";
			table = &this->signal_line;
		} else if (tag == "constant") {
			topic = "class_constant";
			table = &this->constant_line;
		} else {
			return;
		}

		// Case order is important here to correctly handle edge cases like Variant.Type in @GlobalScope.
		if (table->has(link)) {
			// Found in the current page.
			class_desc->scroll_to_line((*table)[link]);
		} else {
			// Look for link in @GlobalScope
			// Note that a link like @GlobalScope.enum_name won't be found in this section, only enum_name will be
			if (topic == "class_enum") {
				const DocData::ClassDoc &cd = doc->class_list["@GlobalScope"];

				for (int i = 0; i < cd.constants.size(); i++) {
					if (cd.constants[i].enumeration == link) {
						// Found in @GlobalScope.
						emit_signal("go_to_help", topic + ":@GlobalScope:" + link);
						return;
					}
				}
			} else if (topic == "class_constant") {
				const DocData::ClassDoc &cd = doc->class_list["@GlobalScope"];

				for (int i = 0; i < cd.constants.size(); i++) {
					if (cd.constants[i].name == link) {
						// Found in @GlobalScope.
						emit_signal("go_to_help", topic + ":@GlobalScope:" + link);
						return;
					}
				}
			}

			if (link.find(".") != -1) {
				// Parse the link as Class.X.
				int class_end = link.find(".");
				emit_signal("go_to_help", topic + ":" + link.substr(0, class_end) + ":" + link.substr(class_end + 1, link.length()));
			}
		}
	} else if (p_select.begins_with("http")) {
		OS::get_singleton()->shell_open(p_select);
	}
}

void EditorHelp::_class_desc_input(const Ref<InputEvent> &p_input) {
}

void EditorHelp::_class_desc_resized() {
	// Add extra horizontal margins for better readability.
	// The margins increase as the width of the editor help container increases.
	Ref<Font> doc_code_font = get_font("doc_source", "EditorFonts");
	real_t char_width = doc_code_font->get_char_size('x').width;
	const int display_margin = MAX(30 * EDSCALE, get_parent_anchorable_rect().size.width - char_width * 120 * EDSCALE) * 0.5;

	Ref<StyleBox> class_desc_stylebox = EditorNode::get_singleton()->get_theme_base()->get_stylebox("normal", "RichTextLabel")->duplicate();
	class_desc_stylebox->set_default_margin(MARGIN_LEFT, display_margin);
	class_desc_stylebox->set_default_margin(MARGIN_RIGHT, display_margin);
	class_desc->add_style_override("normal", class_desc_stylebox);
}

void EditorHelp::_add_type(const String &p_type, const String &p_enum) {
	String t = p_type;
	if (t.empty()) {
		t = "void";
	}
	bool can_ref = (t != "void") || !p_enum.empty();

	if (!p_enum.empty()) {
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
		if (p_enum.empty()) {
			class_desc->push_meta("#" + t); // class
		} else {
			class_desc->push_meta("$" + p_enum); // class
		}
	}
	class_desc->add_text(t);
	if (can_ref) {
		class_desc->pop();
	}
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
		return "0xFFFFF";
	}

	return p_constant;
}

void EditorHelp::_add_method(const DocData::MethodDoc &p_method, bool p_overview) {
	method_line[p_method.name] = class_desc->get_line_count() - 2; // Gets overridden if description.

	const bool is_vararg = p_method.qualifiers.find("vararg") != -1;

	if (p_overview) {
		class_desc->push_cell();
		class_desc->push_align(RichTextLabel::ALIGN_RIGHT);
	} else {
		_add_bulletpoint();
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
		class_desc->push_meta("@method " + p_method.name);
	}

	class_desc->push_color(headline_color);
	_add_text(p_method.name);
	class_desc->pop();

	if (p_overview && p_method.description != "") {
		class_desc->pop(); //meta
	}

	class_desc->push_color(symbol_color);
	class_desc->add_text("(");
	class_desc->pop();

	for (int j = 0; j < p_method.arguments.size(); j++) {
		class_desc->push_color(text_color);
		if (j > 0) {
			class_desc->add_text(", ");
		}

		_add_text(p_method.arguments[j].name);
		class_desc->add_text(": ");
		_add_type(p_method.arguments[j].type, p_method.arguments[j].enumeration);
		if (p_method.arguments[j].default_value != "") {
			class_desc->push_color(symbol_color);
			class_desc->add_text(" = ");
			class_desc->pop();
			class_desc->push_color(value_color);
			_add_text(_fix_constant(p_method.arguments[j].default_value));
			class_desc->pop();
		}

		class_desc->pop();
	}

	if (is_vararg) {
		class_desc->push_color(text_color);
		if (p_method.arguments.size()) {
			class_desc->add_text(", ");
		}
		class_desc->push_color(symbol_color);
		class_desc->add_text("...");
		class_desc->pop();
		class_desc->pop();
	}

	class_desc->push_color(symbol_color);
	class_desc->add_text(")");
	class_desc->pop();
	if (p_method.qualifiers != "") {
		class_desc->push_color(qualifier_color);
		class_desc->add_text(" ");
		_add_text(p_method.qualifiers);
		class_desc->pop();
	}

	if (p_overview) {
		class_desc->pop(); //cell
	}
}

void EditorHelp::_add_bulletpoint() {
	static const CharType prefix[3] = { 0x25CF /* filled circle */, ' ', 0 };
	class_desc->add_text(String(prefix));
}

Error EditorHelp::_goto_desc(const String &p_class, int p_vscr) {
	if (!doc->class_list.has(p_class)) {
		return ERR_DOES_NOT_EXIST;
	}

	select_locked = true;

	class_desc->show();

	description_line = 0;

	if (p_class == edited_class) {
		return OK; // Already there.
	}

	edited_class = p_class;
	_update_doc();
	return OK;
}

void EditorHelp::_update_doc() {
	if (!doc->class_list.has(edited_class)) {
		return;
	}

	scroll_locked = true;

	class_desc->clear();
	method_line.clear();
	section_line.clear();

	_init_colors();

	DocData::ClassDoc cd = doc->class_list[edited_class]; // Make a copy, so we can sort without worrying.

	Ref<Font> doc_font = get_font("doc", "EditorFonts");
	Ref<Font> doc_bold_font = get_font("doc_bold", "EditorFonts");
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

		String inherits = cd.inherits;

		while (inherits != "") {
			_add_type(inherits);

			inherits = doc->class_list[inherits].inherits;

			if (inherits != "") {
				class_desc->add_text(" < ");
			}
		}

		class_desc->pop();
		class_desc->pop();
		class_desc->add_newline();
	}

	// Descendents
	if (ClassDB::class_exists(cd.name)) {
		bool found = false;
		bool prev = false;

		class_desc->push_font(doc_font);
		for (Map<String, DocData::ClassDoc>::Element *E = doc->class_list.front(); E; E = E->next()) {
			if (E->get().inherits == cd.name) {
				if (!found) {
					class_desc->push_color(title_color);
					class_desc->add_text(TTR("Inherited by:") + " ");
					found = true;
				}

				if (prev) {
					class_desc->add_text(" , ");
				}

				_add_type(E->get().name);
				prev = true;
			}
		}
		class_desc->pop();

		if (found) {
			class_desc->pop();
			class_desc->add_newline();
		}
	}

	class_desc->add_newline();
	class_desc->add_newline();

	// Brief description
	if (cd.brief_description != "") {
		class_desc->push_color(text_color);
		class_desc->push_font(doc_bold_font);
		class_desc->push_indent(1);
		_add_text(DTR(cd.brief_description));
		class_desc->pop();
		class_desc->pop();
		class_desc->pop();
		class_desc->add_newline();
		class_desc->add_newline();
		class_desc->add_newline();
	}

	// Class description
	if (cd.description != "") {
		section_line.push_back(Pair<String, int>(TTR("Description"), class_desc->get_line_count() - 2));
		description_line = class_desc->get_line_count() - 2;
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Description"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();
		class_desc->push_color(text_color);
		class_desc->push_font(doc_font);
		class_desc->push_indent(1);
		_add_text(DTR(cd.description));
		class_desc->pop();
		class_desc->pop();
		class_desc->pop();
		class_desc->add_newline();
		class_desc->add_newline();
		class_desc->add_newline();
	}

	// Online tutorials
	if (cd.tutorials.size()) {
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Online Tutorials"));
		class_desc->pop();
		class_desc->pop();

		class_desc->push_indent(1);
		class_desc->push_font(doc_code_font);
		class_desc->add_newline();

		for (int i = 0; i < cd.tutorials.size(); i++) {
			const String link = DTR(cd.tutorials[i].link);
			String linktxt = (cd.tutorials[i].title.empty()) ? link : DTR(cd.tutorials[i].title);
			const int seppos = linktxt.find("//");
			if (seppos != -1) {
				linktxt = link.right(seppos + 2);
			}

			class_desc->push_color(symbol_color);
			class_desc->append_bbcode("[url=" + link + "]" + linktxt + "[/url]");
			class_desc->pop();
			class_desc->add_newline();
		}

		class_desc->pop();
		class_desc->pop();
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
		class_desc->add_text(TTR("Properties"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->push_font(doc_code_font);
		class_desc->push_indent(1);
		class_desc->push_table(3);
		class_desc->set_table_column_expand(1, true);

		for (int i = 0; i < cd.properties.size(); i++) {
			property_line[cd.properties[i].name] = class_desc->get_line_count() - 2; // Gets overridden if description.

			// Property type.
			class_desc->push_cell();
			class_desc->push_align(RichTextLabel::ALIGN_RIGHT);
			class_desc->push_font(doc_code_font);
			_add_type(cd.properties[i].type, cd.properties[i].enumeration);
			class_desc->pop();
			class_desc->pop();
			class_desc->pop(); // Cell

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

			if (cd.properties[i].overridden) {
				describe = false;
			}

			// Property name.
			class_desc->push_cell();
			class_desc->push_font(doc_code_font);
			class_desc->push_color(headline_color);

			if (describe) {
				class_desc->push_meta("@member " + cd.properties[i].name);
			}

			_add_text(cd.properties[i].name);

			if (describe) {
				class_desc->pop();
				property_descr = true;
			}

			class_desc->pop();
			class_desc->pop();
			class_desc->pop(); // Cell

			// Property value.
			class_desc->push_cell();
			class_desc->push_font(doc_code_font);

			if (cd.properties[i].default_value != "") {
				class_desc->push_color(symbol_color);
				if (cd.properties[i].overridden) {
					class_desc->add_text(" [");
					class_desc->push_meta("@member " + cd.properties[i].overrides + "." + cd.properties[i].name);
					_add_text(vformat(TTR("overrides %s:"), cd.properties[i].overrides));
					class_desc->pop();
					class_desc->add_text(" ");
				} else {
					class_desc->add_text(" [" + TTR("default:") + " ");
				}
				class_desc->pop();

				class_desc->push_color(value_color);
				_add_text(_fix_constant(cd.properties[i].default_value));
				class_desc->pop();

				class_desc->push_color(symbol_color);
				class_desc->add_text("]");
				class_desc->pop();
			}

			class_desc->pop();
			class_desc->pop(); // Cell
		}

		class_desc->pop(); // Table
		class_desc->pop();
		class_desc->pop(); // Font
		class_desc->add_newline();
		class_desc->add_newline();
	}

	// Methods overview
	bool method_descr = false;
	bool sort_methods = EditorSettings::get_singleton()->get("text_editor/help/sort_functions_alphabetically");

	Vector<DocData::MethodDoc> methods;

	for (int i = 0; i < cd.methods.size(); i++) {
		if (skip_methods.has(cd.methods[i].name)) {
			if (cd.methods[i].arguments.size() == 0 /* getter */ || (cd.methods[i].arguments.size() == 1 && cd.methods[i].return_type == "void" /* setter */)) {
				continue;
			}
		}
		methods.push_back(cd.methods[i]);
	}

	if (methods.size()) {
		if (sort_methods) {
			methods.sort();
		}

		section_line.push_back(Pair<String, int>(TTR("Methods"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Methods"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->push_font(doc_code_font);
		class_desc->push_indent(1);
		class_desc->push_table(2);
		class_desc->set_table_column_expand(1, true);

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
				class_desc->pop(); // Cell
				class_desc->push_cell();
				class_desc->pop(); // Cell
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
		class_desc->add_text(TTR("Theme Properties"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();

		class_desc->push_indent(1);

		String theme_data_type;
		Map<String, String> data_type_names;
		data_type_names["color"] = TTR("Colors");
		data_type_names["constant"] = TTR("Constants");
		data_type_names["font"] = TTR("Fonts");
		data_type_names["icon"] = TTR("Icons");
		data_type_names["style"] = TTR("Styles");

		for (int i = 0; i < cd.theme_properties.size(); i++) {
			theme_property_line[cd.theme_properties[i].name] = class_desc->get_line_count() - 2; // Gets overridden if description.

			if (theme_data_type != cd.theme_properties[i].data_type) {
				theme_data_type = cd.theme_properties[i].data_type;

				class_desc->push_color(title_color);
				class_desc->push_font(doc_bold_font);
				if (data_type_names.has(theme_data_type)) {
					class_desc->add_text(data_type_names[theme_data_type]);
				} else {
					class_desc->add_text("");
				}
				class_desc->pop();
				class_desc->pop();

				class_desc->add_newline();
				class_desc->add_newline();
			}

			// Theme item header.
			class_desc->push_font(doc_code_font);
			_add_bulletpoint();

			// Theme item object type.
			_add_type(cd.theme_properties[i].type);

			// Theme item name.
			class_desc->push_color(headline_color);
			class_desc->add_text(" ");
			_add_text(cd.theme_properties[i].name);
			class_desc->pop();

			// Theme item default value.
			if (cd.theme_properties[i].default_value != "") {
				class_desc->push_color(symbol_color);
				class_desc->add_text(" [" + TTR("default:") + " ");
				class_desc->pop();
				class_desc->push_color(value_color);
				_add_text(_fix_constant(cd.theme_properties[i].default_value));
				class_desc->pop();
				class_desc->push_color(symbol_color);
				class_desc->add_text("]");
				class_desc->pop();
			}

			class_desc->pop(); // monofont

			// Theme item description.
			if (cd.theme_properties[i].description != "") {
				class_desc->push_font(doc_font);
				class_desc->push_color(comment_color);
				class_desc->push_indent(1);
				_add_text(DTR(cd.theme_properties[i].description));
				class_desc->pop(); // indent
				class_desc->pop(); // color
				class_desc->pop(); // font
			}

			class_desc->add_newline();
			class_desc->add_newline();
		}

		class_desc->pop();
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
		class_desc->add_text(TTR("Signals"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();

		class_desc->push_indent(1);

		for (int i = 0; i < cd.signals.size(); i++) {
			signal_line[cd.signals[i].name] = class_desc->get_line_count() - 2; // Gets overridden if description.

			class_desc->push_font(doc_code_font); // monofont
			class_desc->push_color(headline_color);
			_add_bulletpoint();
			_add_text(cd.signals[i].name);
			class_desc->pop();
			class_desc->push_color(symbol_color);
			class_desc->add_text("(");
			class_desc->pop();
			for (int j = 0; j < cd.signals[i].arguments.size(); j++) {
				class_desc->push_color(text_color);
				if (j > 0) {
					class_desc->add_text(", ");
				}

				_add_text(cd.signals[i].arguments[j].name);
				class_desc->add_text(": ");
				_add_type(cd.signals[i].arguments[j].type);
				if (cd.signals[i].arguments[j].default_value != "") {
					class_desc->push_color(symbol_color);
					class_desc->add_text(" = ");
					class_desc->pop();
					_add_text(cd.signals[i].arguments[j].default_value);
				}

				class_desc->pop();
			}

			class_desc->push_color(symbol_color);
			class_desc->add_text(")");
			class_desc->pop();
			class_desc->pop(); // End monofont
			if (cd.signals[i].description != "") {
				class_desc->push_font(doc_font);
				class_desc->push_color(comment_color);
				class_desc->push_indent(1);
				_add_text(DTR(cd.signals[i].description));
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
		Map<String, Vector<DocData::ConstantDoc>> enums;
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
			class_desc->add_text(TTR("Enumerations"));
			class_desc->pop();
			class_desc->pop();
			class_desc->push_indent(1);

			class_desc->add_newline();

			for (Map<String, Vector<DocData::ConstantDoc>>::Element *E = enums.front(); E; E = E->next()) {
				enum_line[E->key()] = class_desc->get_line_count() - 2;

				class_desc->push_font(doc_code_font);
				class_desc->push_color(title_color);
				class_desc->add_text("enum  ");
				class_desc->pop();
				String e = E->key();
				if ((e.get_slice_count(".") > 1) && (e.get_slice(".", 0) == edited_class)) {
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
				class_desc->add_newline();

				class_desc->push_indent(1);
				Vector<DocData::ConstantDoc> enum_list = E->get();

				Map<String, int> enumValuesContainer;
				int enumStartingLine = enum_line[E->key()];

				for (int i = 0; i < enum_list.size(); i++) {
					if (cd.name == "@GlobalScope") {
						enumValuesContainer[enum_list[i].name] = enumStartingLine;
					}

					// Add the enum constant line to the constant_line map so we can locate it as a constant.
					constant_line[enum_list[i].name] = class_desc->get_line_count() - 2;

					class_desc->push_font(doc_code_font);
					class_desc->push_color(headline_color);
					_add_bulletpoint();
					_add_text(enum_list[i].name);
					class_desc->pop();
					class_desc->push_color(symbol_color);
					class_desc->add_text(" = ");
					class_desc->pop();
					class_desc->push_color(value_color);
					_add_text(_fix_constant(enum_list[i].value));
					class_desc->pop();
					class_desc->pop();

					class_desc->add_newline();

					if (enum_list[i].description.strip_edges() != "") {
						class_desc->push_font(doc_font);
						class_desc->push_color(comment_color);
						_add_text(DTR(enum_list[i].description));
						class_desc->pop();
						class_desc->pop();
						if (DTR(enum_list[i].description).find("\n") > 0) {
							class_desc->add_newline();
						}
					}

					class_desc->add_newline();
				}

				if (cd.name == "@GlobalScope") {
					enum_values_line[E->key()] = enumValuesContainer;
				}

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
			class_desc->add_text(TTR("Constants"));
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
						_add_bulletpoint();
						class_desc->pop();
					}
				} else {
					_add_bulletpoint();
				}

				class_desc->push_color(headline_color);
				_add_text(constants[i].name);
				class_desc->pop();
				class_desc->push_color(symbol_color);
				class_desc->add_text(" = ");
				class_desc->pop();
				class_desc->push_color(value_color);
				_add_text(_fix_constant(constants[i].value));
				class_desc->pop();

				class_desc->pop();

				class_desc->add_newline();

				if (constants[i].description != "") {
					class_desc->push_font(doc_font);
					class_desc->push_color(comment_color);
					_add_text(DTR(constants[i].description));
					class_desc->pop();
					class_desc->pop();
					if (DTR(constants[i].description).find("\n") > 0) {
						class_desc->add_newline();
					}
				}

				class_desc->add_newline();
			}

			class_desc->pop();
			class_desc->add_newline();
		}
	}

	// Property descriptions
	if (property_descr) {
		section_line.push_back(Pair<String, int>(TTR("Property Descriptions"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Property Descriptions"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();

		for (int i = 0; i < cd.properties.size(); i++) {
			if (cd.properties[i].overridden) {
				continue;
			}

			property_line[cd.properties[i].name] = class_desc->get_line_count() - 2;

			class_desc->push_table(2);
			class_desc->set_table_column_expand(1, true);

			class_desc->push_cell();
			class_desc->push_font(doc_code_font);
			_add_bulletpoint();

			_add_type(cd.properties[i].type, cd.properties[i].enumeration);
			class_desc->add_text(" ");
			class_desc->pop(); // font
			class_desc->pop(); // cell

			class_desc->push_cell();
			class_desc->push_font(doc_code_font);
			class_desc->push_color(headline_color);
			_add_text(cd.properties[i].name);
			class_desc->pop(); // color

			if (cd.properties[i].default_value != "") {
				class_desc->push_color(symbol_color);
				class_desc->add_text(" [" + TTR("default:") + " ");
				class_desc->pop(); // color

				class_desc->push_color(value_color);
				_add_text(_fix_constant(cd.properties[i].default_value));
				class_desc->pop(); // color

				class_desc->push_color(symbol_color);
				class_desc->add_text("]");
				class_desc->pop(); // color
			}

			class_desc->pop(); // font
			class_desc->pop(); // cell

			Map<String, DocData::MethodDoc> method_map;
			for (int j = 0; j < methods.size(); j++) {
				method_map[methods[j].name] = methods[j];
			}

			if (cd.properties[i].setter != "") {
				class_desc->push_cell();
				class_desc->pop(); // cell

				class_desc->push_cell();
				class_desc->push_font(doc_code_font);
				class_desc->push_color(text_color);
				if (method_map[cd.properties[i].setter].arguments.size() > 1) {
					// Setters with additional arguments are exposed in the method list, so we link them here for quick access.
					class_desc->push_meta("@method " + cd.properties[i].setter);
					class_desc->add_text(cd.properties[i].setter + TTR("(value)"));
					class_desc->pop();
				} else {
					class_desc->add_text(cd.properties[i].setter + TTR("(value)"));
				}
				class_desc->pop(); // color
				class_desc->push_color(comment_color);
				class_desc->add_text(" setter");
				class_desc->pop(); // color
				class_desc->pop(); // font
				class_desc->pop(); // cell
				method_line[cd.properties[i].setter] = property_line[cd.properties[i].name];
			}

			if (cd.properties[i].getter != "") {
				class_desc->push_cell();
				class_desc->pop(); // cell

				class_desc->push_cell();
				class_desc->push_font(doc_code_font);
				class_desc->push_color(text_color);
				if (method_map[cd.properties[i].getter].arguments.size() > 0) {
					// Getters with additional arguments are exposed in the method list, so we link them here for quick access.
					class_desc->push_meta("@method " + cd.properties[i].getter);
					class_desc->add_text(cd.properties[i].getter + "()");
					class_desc->pop();
				} else {
					class_desc->add_text(cd.properties[i].getter + "()");
				}
				class_desc->pop(); //color
				class_desc->push_color(comment_color);
				class_desc->add_text(" getter");
				class_desc->pop(); //color
				class_desc->pop(); //font
				class_desc->pop(); //cell
				method_line[cd.properties[i].getter] = property_line[cd.properties[i].name];
			}

			class_desc->pop(); // table

			class_desc->add_newline();
			class_desc->add_newline();

			class_desc->push_color(text_color);
			class_desc->push_font(doc_font);
			class_desc->push_indent(1);
			if (cd.properties[i].description.strip_edges() != String()) {
				_add_text(DTR(cd.properties[i].description));
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
		class_desc->add_text(TTR("Method Descriptions"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();

		for (int pass = 0; pass < 2; pass++) {
			Vector<DocData::MethodDoc> methods_filtered;

			for (int i = 0; i < methods.size(); i++) {
				const String &q = methods[i].qualifiers;
				if ((pass == 0 && q.find("virtual") != -1) || (pass == 1 && q.find("virtual") == -1)) {
					methods_filtered.push_back(methods[i]);
				}
			}

			for (int i = 0; i < methods_filtered.size(); i++) {
				class_desc->push_font(doc_code_font);
				_add_method(methods_filtered[i], false);
				class_desc->pop();

				class_desc->add_newline();
				class_desc->add_newline();

				class_desc->push_color(text_color);
				class_desc->push_font(doc_font);
				class_desc->push_indent(1);
				if (methods_filtered[i].description.strip_edges() != String()) {
					_add_text(DTR(methods_filtered[i].description));
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
	}
	scroll_locked = false;
}

void EditorHelp::_request_help(const String &p_string) {
	Error err = _goto_desc(p_string);
	if (err == OK) {
		EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
	}
	// 100 palabras
}

void EditorHelp::_help_callback(const String &p_topic) {
	String what = p_topic.get_slice(":", 0);
	String clss = p_topic.get_slice(":", 1);
	String name;
	if (p_topic.get_slice_count(":") == 3) {
		name = p_topic.get_slice(":", 2);
	}

	_request_help(clss); // First go to class.

	int line = 0;

	if (what == "class_desc") {
		line = description_line;
	} else if (what == "class_signal") {
		if (signal_line.has(name)) {
			line = signal_line[name];
		}
	} else if (what == "class_method" || what == "class_method_desc") {
		if (method_line.has(name)) {
			line = method_line[name];
		}
	} else if (what == "class_property") {
		if (property_line.has(name)) {
			line = property_line[name];
		}
	} else if (what == "class_enum") {
		if (enum_line.has(name)) {
			line = enum_line[name];
		}
	} else if (what == "class_theme_item") {
		if (theme_property_line.has(name)) {
			line = theme_property_line[name];
		}
	} else if (what == "class_constant") {
		if (constant_line.has(name)) {
			line = constant_line[name];
		}
	} else if (what == "class_global") {
		if (constant_line.has(name)) {
			line = constant_line[name];
		} else {
			Map<String, Map<String, int>>::Element *iter = enum_values_line.front();
			while (true) {
				if (iter->value().has(name)) {
					line = iter->value()[name];
					break;
				} else if (iter == enum_values_line.back()) {
					break;
				} else {
					iter = iter->next();
				}
			}
		}
	}

	class_desc->call_deferred("scroll_to_line", line);
}

static void _add_text_to_rt(const String &p_bbcode, RichTextLabel *p_rt) {
	DocData *doc = EditorHelp::get_doc_data();
	String base_path;

	Ref<Font> doc_font = p_rt->get_font("doc", "EditorFonts");
	Ref<Font> doc_bold_font = p_rt->get_font("doc_bold", "EditorFonts");
	Ref<Font> doc_code_font = p_rt->get_font("doc_source", "EditorFonts");
	Ref<Font> doc_kbd_font = p_rt->get_font("doc_keyboard", "EditorFonts");

	Color font_color_hl = p_rt->get_color("headline_color", "EditorHelp");
	Color accent_color = p_rt->get_color("accent_color", "Editor");
	Color property_color = p_rt->get_color("property_color", "Editor");
	Color link_color = accent_color.linear_interpolate(font_color_hl, 0.8);
	Color code_color = accent_color.linear_interpolate(font_color_hl, 0.6);
	Color kbd_color = accent_color.linear_interpolate(property_color, 0.6);

	String bbcode = p_bbcode.dedent().replace("\t", "").replace("\r", "").strip_edges();

	// Remove extra new lines around code blocks.
	bbcode = bbcode.replace("[codeblock]\n", "[codeblock]");
	bbcode = bbcode.replace("\n[/codeblock]", "[/codeblock]");

	List<String> tag_stack;
	bool code_tag = false;

	int pos = 0;
	while (pos < bbcode.length()) {
		int brk_pos = bbcode.find("[", pos);

		if (brk_pos < 0) {
			brk_pos = bbcode.length();
		}

		if (brk_pos > pos) {
			String text = bbcode.substr(pos, brk_pos - pos);
			if (!code_tag) {
				text = text.replace("\n", "\n\n");
			}
			p_rt->add_text(text);
		}

		if (brk_pos == bbcode.length()) {
			break; // Nothing else to add.
		}

		int brk_end = bbcode.find("]", brk_pos + 1);

		if (brk_end == -1) {
			String text = bbcode.substr(brk_pos, bbcode.length() - brk_pos);
			if (!code_tag) {
				text = text.replace("\n", "\n\n");
			}
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
			if (tag != "/img") {
				p_rt->pop();
				if (code_tag) {
					p_rt->pop();
				}
			}
			code_tag = false;

		} else if (code_tag) {
			p_rt->add_text("[");
			pos = brk_pos + 1;

		} else if (tag.begins_with("method ") || tag.begins_with("member ") || tag.begins_with("signal ") || tag.begins_with("enum ") || tag.begins_with("constant ")) {
			const int tag_end = tag.find(" ");
			const String link_tag = tag.substr(0, tag_end);
			const String link_target = tag.substr(tag_end + 1, tag.length()).lstrip(" ");

			p_rt->push_font(doc_code_font);
			p_rt->push_color(link_color);
			p_rt->push_meta("@" + link_tag + " " + link_target);
			p_rt->add_text(link_target + (tag.begins_with("method ") ? "()" : ""));
			p_rt->pop();
			p_rt->pop();
			p_rt->pop();
			pos = brk_end + 1;

		} else if (doc->class_list.has(tag)) {
			// Class reference tag such as [Node2D] or [SceneTree].
			p_rt->push_font(doc_code_font);
			p_rt->push_color(link_color);
			p_rt->push_meta("#" + tag);
			p_rt->add_text(tag);
			p_rt->pop();
			p_rt->pop();
			p_rt->pop();
			pos = brk_end + 1;

		} else if (tag == "b") {
			// Use bold font.
			p_rt->push_font(doc_bold_font);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "i") {
			// Use italics font.
			p_rt->push_color(font_color_hl);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "code" || tag == "codeblock") {
			// Use monospace font.
			p_rt->push_font(doc_code_font);
			p_rt->push_color(code_color);
			code_tag = true;
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "kbd") {
			// Use keyboard font with custom color.
			p_rt->push_font(doc_kbd_font);
			p_rt->push_color(kbd_color);
			code_tag = true; // Though not strictly a code tag, logic is similar.
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "center") {
			// Align to center.
			p_rt->push_align(RichTextLabel::ALIGN_CENTER);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "br") {
			// Force a line break.
			p_rt->add_newline();
			pos = brk_end + 1;
		} else if (tag == "u") {
			// Use underline.
			p_rt->push_underline();
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "s") {
			// Use strikethrough.
			p_rt->push_strikethrough();
			pos = brk_end + 1;
			tag_stack.push_front(tag);

		} else if (tag == "url") {
			int end = bbcode.find("[", brk_end);
			if (end == -1) {
				end = bbcode.length();
			}
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
			int end = bbcode.find("[", brk_end);
			if (end == -1) {
				end = bbcode.length();
			}
			String image = bbcode.substr(brk_end + 1, end - brk_end - 1);

			Ref<Texture> texture = ResourceLoader::load(base_path.plus_file(image), "Texture");
			if (texture.is_valid()) {
				p_rt->add_image(texture);
			}

			pos = end;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("color=")) {
			String col = tag.substr(6, tag.length());
			Color color;

			if (col.begins_with("#")) {
				color = Color::html(col);
			} else if (col == "aqua") {
				color = Color(0, 1, 1);
			} else if (col == "black") {
				color = Color(0, 0, 0);
			} else if (col == "blue") {
				color = Color(0, 0, 1);
			} else if (col == "fuchsia") {
				color = Color(1, 0, 1);
			} else if (col == "gray" || col == "grey") {
				color = Color(0.5, 0.5, 0.5);
			} else if (col == "green") {
				color = Color(0, 0.5, 0);
			} else if (col == "lime") {
				color = Color(0, 1, 0);
			} else if (col == "maroon") {
				color = Color(0.5, 0, 0);
			} else if (col == "navy") {
				color = Color(0, 0, 0.5);
			} else if (col == "olive") {
				color = Color(0.5, 0.5, 0);
			} else if (col == "purple") {
				color = Color(0.5, 0, 0.5);
			} else if (col == "red") {
				color = Color(1, 0, 0);
			} else if (col == "silver") {
				color = Color(0.75, 0.75, 0.75);
			} else if (col == "teal") {
				color = Color(0, 0.5, 0.5);
			} else if (col == "white") {
				color = Color(1, 1, 1);
			} else if (col == "yellow") {
				color = Color(1, 1, 0);
			} else {
				color = Color(0, 0, 0); // base_color;
			}

			p_rt->push_color(color);
			pos = brk_end + 1;
			tag_stack.push_front("color");

		} else if (tag.begins_with("font=")) {
			String fnt = tag.substr(5, tag.length());

			Ref<Font> font = ResourceLoader::load(base_path.plus_file(fnt), "Font");
			if (font.is_valid()) {
				p_rt->push_font(font);
			} else {
				p_rt->push_font(doc_font);
			}

			pos = brk_end + 1;
			tag_stack.push_front("font");

		} else {
			p_rt->add_text("["); // ignore
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
	doc->merge_from(compdoc); // Ensure all is up to date.
}

void EditorHelp::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY:
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			_update_doc();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			if (is_inside_tree()) {
				_class_desc_resized();
			}
		} break;
		default:
			break;
	}
}

void EditorHelp::go_to_help(const String &p_help) {
	_help_callback(p_help);
}

void EditorHelp::go_to_class(const String &p_class, int p_scroll) {
	_goto_desc(p_class, p_scroll);
}

Vector<Pair<String, int>> EditorHelp::get_sections() {
	Vector<Pair<String, int>> sections;

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

void EditorHelp::search_again(bool p_search_previous) {
	_search(p_search_previous);
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
	ClassDB::bind_method("_class_desc_resized", &EditorHelp::_class_desc_resized);
	ClassDB::bind_method("_request_help", &EditorHelp::_request_help);
	ClassDB::bind_method("_unhandled_key_input", &EditorHelp::_unhandled_key_input);
	ClassDB::bind_method("_search", &EditorHelp::_search);
	ClassDB::bind_method("_help_callback", &EditorHelp::_help_callback);

	ADD_SIGNAL(MethodInfo("go_to_help"));
}

EditorHelp::EditorHelp() {
	set_custom_minimum_size(Size2(150 * EDSCALE, 0));

	EDITOR_DEF("text_editor/help/sort_functions_alphabetically", true);

	class_desc = memnew(RichTextLabel);
	add_child(class_desc);
	class_desc->set_v_size_flags(SIZE_EXPAND_FILL);
	class_desc->add_color_override("selection_color", get_color("accent_color", "Editor") * Color(1, 1, 1, 0.4));

	class_desc->connect("meta_clicked", this, "_class_desc_select");
	class_desc->connect("gui_input", this, "_class_desc_input");
	class_desc->connect("resized", this, "_class_desc_resized");
	_class_desc_resized();

	// Added second so it opens at the bottom so it won't offset the entire widget.
	find_bar = memnew(FindBar);
	add_child(find_bar);
	find_bar->hide();
	find_bar->set_rich_text_label(class_desc);

	class_desc->set_selection_enabled(true);

	scroll_locked = false;
	select_locked = false;
	class_desc->hide();
}

EditorHelp::~EditorHelp() {
}

void EditorHelpBit::_go_to_help(String p_what) {
	EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
	ScriptEditor::get_singleton()->goto_help(p_what);
	emit_signal("request_hide");
}

void EditorHelpBit::_meta_clicked(String p_select) {
	if (p_select.begins_with("$")) { // enum

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

		if (m.find(".") != -1) {
			_go_to_help("class_method:" + m.get_slice(".", 0) + ":" + m.get_slice(".", 0)); // Must go somewhere else.
		}
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
			rich_text->add_color_override("selection_color", get_color("accent_color", "Editor") * Color(1, 1, 1, 0.4));
		} break;
		default:
			break;
	}
}

void EditorHelpBit::set_text(const String &p_text) {
	rich_text->clear();
	_add_text_to_rt(p_text, rich_text);
}

EditorHelpBit::EditorHelpBit() {
	rich_text = memnew(RichTextLabel);
	add_child(rich_text);
	rich_text->connect("meta_clicked", this, "_meta_clicked");
	rich_text->add_color_override("selection_color", get_color("accent_color", "Editor") * Color(1, 1, 1, 0.4));
	rich_text->set_override_selected_font_color(false);
	set_custom_minimum_size(Size2(0, 70 * EDSCALE));
}

FindBar::FindBar() {
	results_count = 0;

	search_text = memnew(LineEdit);
	add_child(search_text);
	search_text->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	search_text->set_h_size_flags(SIZE_EXPAND_FILL);
	search_text->connect("text_changed", this, "_search_text_changed");
	search_text->connect("text_entered", this, "_search_text_entered");

	matches_label = memnew(Label);
	add_child(matches_label);
	matches_label->hide();

	find_prev = memnew(ToolButton);
	add_child(find_prev);
	find_prev->set_focus_mode(FOCUS_NONE);
	find_prev->connect("pressed", this, "_search_prev");

	find_next = memnew(ToolButton);
	add_child(find_next);
	find_next->set_focus_mode(FOCUS_NONE);
	find_next->connect("pressed", this, "_search_next");

	Control *space = memnew(Control);
	add_child(space);
	space->set_custom_minimum_size(Size2(4, 0) * EDSCALE);

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
}

void FindBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			find_prev->set_icon(get_icon("MoveUp", "EditorIcons"));
			find_next->set_icon(get_icon("MoveDown", "EditorIcons"));
			hide_button->set_normal_texture(get_icon("Close", "EditorIcons"));
			hide_button->set_hover_texture(get_icon("Close", "EditorIcons"));
			hide_button->set_pressed_texture(get_icon("Close", "EditorIcons"));
			hide_button->set_custom_minimum_size(hide_button->get_normal_texture()->get_size());
			matches_label->add_color_override("font_color", results_count > 0 ? get_color("font_color", "Label") : get_color("error_color", "Editor"));
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			set_process_unhandled_input(is_visible_in_tree());
		} break;
	}
}

void FindBar::_bind_methods() {
	ClassDB::bind_method("_unhandled_input", &FindBar::_unhandled_input);

	ClassDB::bind_method("_search_text_changed", &FindBar::_search_text_changed);
	ClassDB::bind_method("_search_text_entered", &FindBar::_search_text_entered);
	ClassDB::bind_method("_search_next", &FindBar::search_next);
	ClassDB::bind_method("_search_prev", &FindBar::search_prev);
	ClassDB::bind_method("_hide_pressed", &FindBar::_hide_bar);

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
		_update_results_count();
	} else {
		results_count = 0;
	}
	_update_matches_label();

	return ret;
}

void FindBar::_update_results_count() {
	results_count = 0;

	String searched = search_text->get_text();
	if (searched.empty()) {
		return;
	}

	String full_text = rich_text_label->get_text();

	int from_pos = 0;

	while (true) {
		int pos = full_text.findn(searched, from_pos);
		if (pos == -1) {
			break;
		}

		results_count++;
		from_pos = pos + searched.length();
	}
}

void FindBar::_update_matches_label() {
	if (search_text->get_text().empty() || results_count == -1) {
		matches_label->hide();
	} else {
		matches_label->show();

		matches_label->add_color_override("font_color", results_count > 0 ? get_color("font_color", "Label") : get_color("error_color", "Editor"));
		matches_label->set_text(vformat(results_count == 1 ? TTR("%d match.") : TTR("%d matches."), results_count));
	}
}

void FindBar::_hide_bar() {
	if (search_text->has_focus()) {
		rich_text_label->grab_focus();
	}

	hide();
}

void FindBar::_unhandled_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;
	if (k.is_valid()) {
		if (k->is_pressed() && (rich_text_label->has_focus() || is_a_parent_of(get_focus_owner()))) {
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
	if (Input::get_singleton()->is_key_pressed(KEY_SHIFT)) {
		search_prev();
	} else {
		search_next();
	}
}
