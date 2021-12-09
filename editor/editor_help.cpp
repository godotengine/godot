/*************************************************************************/
/*  editor_help.cpp                                                      */
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

#include "editor_help.h"

#include "core/core_constants.h"
#include "core/input/input.h"
#include "core/os/keyboard.h"
#include "core/version_generated.gen.h"
#include "doc_data_compressed.gen.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "editor_settings.h"

#define CONTRIBUTE_URL vformat("%s/community/contributing/updating_the_class_reference.html", VERSION_DOCS_URL)

DocTools *EditorHelp::doc = nullptr;

void EditorHelp::_init_colors() {
	title_color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
	text_color = get_theme_color(SNAME("default_color"), SNAME("RichTextLabel"));
	headline_color = get_theme_color(SNAME("headline_color"), SNAME("EditorHelp"));
	base_type_color = title_color.lerp(text_color, 0.5);
	comment_color = text_color * Color(1, 1, 1, 0.6);
	symbol_color = comment_color;
	value_color = text_color * Color(1, 1, 1, 0.6);
	qualifier_color = text_color * Color(1, 1, 1, 0.8);
	type_color = get_theme_color(SNAME("accent_color"), SNAME("Editor")).lerp(text_color, 0.5);
	class_desc->add_theme_color_override("selection_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.4));
	class_desc->add_theme_constant_override("line_separation", Math::round(5 * EDSCALE));
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
	if (p_select.begins_with("$")) { //enum
		String select = p_select.substr(1, p_select.length());
		String class_name;
		if (select.find(".") != -1) {
			class_name = select.get_slice(".", 0);
			select = select.get_slice(".", 1);
		} else {
			class_name = "@GlobalScope";
		}
		emit_signal(SNAME("go_to_help"), "class_enum:" + class_name + ":" + select);
		return;
	} else if (p_select.begins_with("#")) {
		emit_signal(SNAME("go_to_help"), "class_name:" + p_select.substr(1, p_select.length()));
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
		} else if (tag == "theme_item") {
			topic = "theme_item";
			table = &this->theme_property_line;
		} else {
			return;
		}

		if (link.find(".") != -1) {
			emit_signal(SNAME("go_to_help"), topic + ":" + link.get_slice(".", 0) + ":" + link.get_slice(".", 1));
		} else {
			if (table->has(link)) {
				// Found in the current page
				class_desc->scroll_to_paragraph((*table)[link]);
			} else {
				if (topic == "class_enum") {
					// Try to find the enum in @GlobalScope
					const DocData::ClassDoc &cd = doc->class_list["@GlobalScope"];

					for (int i = 0; i < cd.constants.size(); i++) {
						if (cd.constants[i].enumeration == link) {
							// Found in @GlobalScope
							emit_signal(SNAME("go_to_help"), topic + ":@GlobalScope:" + link);
							break;
						}
					}
				} else if (topic == "class_constant") {
					// Try to find the constant in @GlobalScope
					const DocData::ClassDoc &cd = doc->class_list["@GlobalScope"];

					for (int i = 0; i < cd.constants.size(); i++) {
						if (cd.constants[i].name == link) {
							// Found in @GlobalScope
							emit_signal(SNAME("go_to_help"), topic + ":@GlobalScope:" + link);
							break;
						}
					}
				}
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
	Ref<Font> doc_code_font = get_theme_font(SNAME("doc_source"), SNAME("EditorFonts"));
	int font_size = get_theme_font_size(SNAME("doc_source_size"), SNAME("EditorFonts"));
	real_t char_width = doc_code_font->get_char_size('x', 0, font_size).width;
	const int display_margin = MAX(30 * EDSCALE, get_parent_anchorable_rect().size.width - char_width * 120 * EDSCALE) * 0.5;

	Ref<StyleBox> class_desc_stylebox = EditorNode::get_singleton()->get_theme_base()->get_theme_stylebox(SNAME("normal"), SNAME("RichTextLabel"))->duplicate();
	class_desc_stylebox->set_default_margin(SIDE_LEFT, display_margin);
	class_desc_stylebox->set_default_margin(SIDE_RIGHT, display_margin);
	class_desc->add_theme_style_override("normal", class_desc_stylebox);
}

void EditorHelp::_add_type(const String &p_type, const String &p_enum) {
	String t = p_type;
	if (t.is_empty()) {
		t = "void";
	}
	bool can_ref = (t != "void" && t.find("*") == -1) || !p_enum.is_empty();

	if (!p_enum.is_empty()) {
		if (p_enum.get_slice_count(".") > 1) {
			t = p_enum.get_slice(".", 1);
		} else {
			t = p_enum.get_slice(".", 0);
		}
	}
	const Color text_color = get_theme_color(SNAME("default_color"), SNAME("RichTextLabel"));
	const Color type_color = get_theme_color(SNAME("accent_color"), SNAME("Editor")).lerp(text_color, 0.5);
	class_desc->push_color(type_color);
	bool add_array = false;
	if (can_ref) {
		if (t.ends_with("[]")) {
			add_array = true;
			t = t.replace("[]", "");
		}
		if (p_enum.is_empty()) {
			class_desc->push_meta("#" + t); //class
		} else {
			class_desc->push_meta("$" + p_enum); //class
		}
	}
	class_desc->add_text(t);
	if (can_ref) {
		class_desc->pop();
		if (add_array) {
			class_desc->add_text(" ");
			class_desc->push_meta("#Array"); //class
			class_desc->add_text("[]");
			class_desc->pop();
		}
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
	method_line[p_method.name] = class_desc->get_line_count() - 2; //gets overridden if description

	const bool is_vararg = p_method.qualifiers.find("vararg") != -1;

	if (p_overview) {
		class_desc->push_cell();
		class_desc->push_paragraph(HORIZONTAL_ALIGNMENT_RIGHT, Control::TEXT_DIRECTION_AUTO, "");
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
	static const char32_t prefix[3] = { 0x25CF /* filled circle */, ' ', 0 };
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
		return OK; //already there
	}

	edited_class = p_class;
	_update_doc();
	return OK;
}

void EditorHelp::_update_method_list(const Vector<DocData::MethodDoc> p_methods, bool &r_method_descrpitons) {
	Ref<Font> doc_code_font = get_theme_font(SNAME("doc_source"), SNAME("EditorFonts"));
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

		for (int i = 0; i < p_methods.size(); i++) {
			const String &q = p_methods[i].qualifiers;
			if ((pass == 0 && q.find("virtual") != -1) || (pass == 1 && q.find("virtual") == -1)) {
				m.push_back(p_methods[i]);
			}
		}

		if (any_previous && !m.is_empty()) {
			class_desc->push_cell();
			class_desc->pop(); //cell
			class_desc->push_cell();
			class_desc->pop(); //cell
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

			if (m[i].description != "" || m[i].errors_returned.size() > 0) {
				r_method_descrpitons = true;
			}

			_add_method(m[i], true);
		}

		any_previous = !m.is_empty();
	}

	class_desc->pop(); //table
	class_desc->pop();
	class_desc->pop(); // font
	class_desc->add_newline();
	class_desc->add_newline();
}

void EditorHelp::_update_method_descriptions(const DocData::ClassDoc p_classdoc, const Vector<DocData::MethodDoc> p_methods, const String &p_method_type) {
	Ref<Font> doc_font = get_theme_font(SNAME("doc"), SNAME("EditorFonts"));
	Ref<Font> doc_bold_font = get_theme_font(SNAME("doc_bold"), SNAME("EditorFonts"));
	Ref<Font> doc_code_font = get_theme_font(SNAME("doc_source"), SNAME("EditorFonts"));
	String link_color_text = title_color.to_html(false);
	class_desc->pop();
	class_desc->pop();

	class_desc->add_newline();
	class_desc->add_newline();

	for (int pass = 0; pass < 2; pass++) {
		Vector<DocData::MethodDoc> methods_filtered;

		for (int i = 0; i < p_methods.size(); i++) {
			const String &q = p_methods[i].qualifiers;
			if ((pass == 0 && q.find("virtual") != -1) || (pass == 1 && q.find("virtual") == -1)) {
				methods_filtered.push_back(p_methods[i]);
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
			if (methods_filtered[i].errors_returned.size()) {
				class_desc->append_text(TTR("Error codes returned:"));
				class_desc->add_newline();
				class_desc->push_list(0, RichTextLabel::LIST_DOTS, false);
				for (int j = 0; j < methods_filtered[i].errors_returned.size(); j++) {
					if (j > 0) {
						class_desc->add_newline();
					}
					int val = methods_filtered[i].errors_returned[j];
					String text = itos(val);
					for (int k = 0; k < CoreConstants::get_global_constant_count(); k++) {
						if (CoreConstants::get_global_constant_value(k) == val && CoreConstants::get_global_constant_enum(k) == SNAME("Error")) {
							text = CoreConstants::get_global_constant_name(k);
							break;
						}
					}

					class_desc->push_bold();
					class_desc->append_text(text);
					class_desc->pop();
				}
				class_desc->pop();
				class_desc->add_newline();
				class_desc->add_newline();
			}
			if (!methods_filtered[i].description.strip_edges().is_empty()) {
				_add_text(DTR(methods_filtered[i].description));
			} else {
				class_desc->add_image(get_theme_icon(SNAME("Error"), SNAME("EditorIcons")));
				class_desc->add_text(" ");
				class_desc->push_color(comment_color);
				if (p_classdoc.is_script_doc) {
					class_desc->append_text(TTR("There is currently no description for this " + p_method_type + "."));
				} else {
					class_desc->append_text(TTR("There is currently no description for this " + p_method_type + ". Please help us by [color=$color][url=$url]contributing one[/url][/color]!").replace("$url", CONTRIBUTE_URL).replace("$color", link_color_text));
				}
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

void EditorHelp::_update_doc() {
	if (!doc->class_list.has(edited_class)) {
		return;
	}

	scroll_locked = true;

	class_desc->clear();
	method_line.clear();
	section_line.clear();

	_init_colors();

	DocData::ClassDoc cd = doc->class_list[edited_class]; //make a copy, so we can sort without worrying

	Ref<Font> doc_font = get_theme_font(SNAME("doc"), SNAME("EditorFonts"));
	Ref<Font> doc_bold_font = get_theme_font(SNAME("doc_bold"), SNAME("EditorFonts"));
	Ref<Font> doc_title_font = get_theme_font(SNAME("doc_title"), SNAME("EditorFonts"));
	Ref<Font> doc_code_font = get_theme_font(SNAME("doc_source"), SNAME("EditorFonts"));
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
	if (cd.is_script_doc || ClassDB::class_exists(cd.name)) {
		bool found = false;
		bool prev = false;

		class_desc->push_font(doc_font);
		for (const KeyValue<String, DocData::ClassDoc> &E : doc->class_list) {
			if (E.value.inherits == cd.name) {
				if (!found) {
					class_desc->push_color(title_color);
					class_desc->add_text(TTR("Inherited by:") + " ");
					found = true;
				}

				if (prev) {
					class_desc->add_text(" , ");
				}

				_add_type(E.value.name);
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
			String linktxt = (cd.tutorials[i].title.is_empty()) ? link : DTR(cd.tutorials[i].title);
			const int seppos = linktxt.find("//");
			if (seppos != -1) {
				linktxt = link.substr(seppos + 2);
			}

			class_desc->push_color(symbol_color);
			class_desc->append_text("[url=" + link + "]" + linktxt + "[/url]");
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

	bool has_properties = cd.properties.size() != 0;
	if (cd.is_script_doc) {
		has_properties = false;
		for (int i = 0; i < cd.properties.size(); i++) {
			if (cd.properties[i].name.begins_with("_") && cd.properties[i].description.is_empty()) {
				continue;
			}
			has_properties = true;
			break;
		}
	}

	if (has_properties) {
		section_line.push_back(Pair<String, int>(TTR("Properties"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Properties"));
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->push_font(doc_code_font);
		class_desc->push_indent(1);
		class_desc->push_table(4);
		class_desc->set_table_column_expand(1, true);

		for (int i = 0; i < cd.properties.size(); i++) {
			// Ignore undocumented private.
			if (cd.properties[i].name.begins_with("_") && cd.properties[i].description.is_empty()) {
				continue;
			}
			property_line[cd.properties[i].name] = class_desc->get_line_count() - 2; //gets overridden if description

			// Property type.
			class_desc->push_cell();
			class_desc->push_paragraph(HORIZONTAL_ALIGNMENT_RIGHT, Control::TEXT_DIRECTION_AUTO, "");
			class_desc->push_font(doc_code_font);
			_add_type(cd.properties[i].type, cd.properties[i].enumeration);
			class_desc->pop();
			class_desc->pop();
			class_desc->pop(); // cell

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
			class_desc->pop(); // cell

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
			class_desc->pop(); // cell

			// Property setters and getters.
			class_desc->push_cell();
			class_desc->push_font(doc_code_font);

			if (cd.is_script_doc && (cd.properties[i].setter != "" || cd.properties[i].getter != "")) {
				class_desc->push_color(symbol_color);
				class_desc->add_text(" [" + TTR("property:") + " ");
				class_desc->pop(); // color

				if (cd.properties[i].setter != "") {
					class_desc->push_color(value_color);
					class_desc->add_text("setter");
					class_desc->pop(); // color
				}
				if (cd.properties[i].getter != "") {
					if (cd.properties[i].setter != "") {
						class_desc->push_color(symbol_color);
						class_desc->add_text(", ");
						class_desc->pop(); // color
					}
					class_desc->push_color(value_color);
					class_desc->add_text("getter");
					class_desc->pop(); // color
				}

				class_desc->push_color(symbol_color);
				class_desc->add_text("]");
				class_desc->pop(); // color
			}

			class_desc->pop();
			class_desc->pop(); // cell
		}

		class_desc->pop(); // table
		class_desc->pop();
		class_desc->pop(); // font
		class_desc->add_newline();
		class_desc->add_newline();
	}

	// Methods overview
	bool constructor_descriptions = false;
	bool method_descriptions = false;
	bool operator_descriptions = false;
	bool sort_methods = EditorSettings::get_singleton()->get("text_editor/help/sort_functions_alphabetically");

	Vector<DocData::MethodDoc> methods;

	for (int i = 0; i < cd.methods.size(); i++) {
		if (skip_methods.has(cd.methods[i].name)) {
			if (cd.methods[i].arguments.size() == 0 /* getter */ || (cd.methods[i].arguments.size() == 1 && cd.methods[i].return_type == "void" /* setter */)) {
				continue;
			}
		}
		// Ignore undocumented non virtual private.
		if (cd.methods[i].name.begins_with("_") && cd.methods[i].description.is_empty() && cd.methods[i].qualifiers.find("virtual") == -1) {
			continue;
		}
		methods.push_back(cd.methods[i]);
	}

	if (!cd.constructors.is_empty()) {
		if (sort_methods) {
			cd.constructors.sort();
		}

		section_line.push_back(Pair<String, int>(TTR("Constructors"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Constructors"));
		_update_method_list(cd.constructors, constructor_descriptions);
	}

	if (!methods.is_empty()) {
		if (sort_methods) {
			methods.sort();
		}
		section_line.push_back(Pair<String, int>(TTR("Methods"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Methods"));
		_update_method_list(methods, method_descriptions);
	}

	if (!cd.operators.is_empty()) {
		if (sort_methods) {
			cd.operators.sort();
		}

		section_line.push_back(Pair<String, int>(TTR("Operators"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Operators"));
		_update_method_list(cd.operators, operator_descriptions);
	}

	// Theme properties
	if (!cd.theme_properties.is_empty()) {
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
		data_type_names["font_size"] = TTR("Font Sizes");
		data_type_names["icon"] = TTR("Icons");
		data_type_names["style"] = TTR("Styles");

		for (int i = 0; i < cd.theme_properties.size(); i++) {
			theme_property_line[cd.theme_properties[i].name] = class_desc->get_line_count() - 2; //gets overridden if description

			if (theme_data_type != cd.theme_properties[i].data_type) {
				theme_data_type = cd.theme_properties[i].data_type;

				class_desc->push_color(title_color);
				class_desc->push_font(doc_title_font);
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
	if (!cd.signals.is_empty()) {
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
			signal_line[cd.signals[i].name] = class_desc->get_line_count() - 2; //gets overridden if description

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
			class_desc->pop(); // end monofont
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
	if (!cd.constants.is_empty()) {
		Map<String, Vector<DocData::ConstantDoc>> enums;
		Vector<DocData::ConstantDoc> constants;

		for (int i = 0; i < cd.constants.size(); i++) {
			if (!cd.constants[i].enumeration.is_empty()) {
				if (!enums.has(cd.constants[i].enumeration)) {
					enums[cd.constants[i].enumeration] = Vector<DocData::ConstantDoc>();
				}

				enums[cd.constants[i].enumeration].push_back(cd.constants[i]);
			} else {
				// Ignore undocumented private.
				if (cd.constants[i].name.begins_with("_") && cd.constants[i].description.is_empty()) {
					continue;
				}
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

			for (KeyValue<String, Vector<DocData::ConstantDoc>> &E : enums) {
				enum_line[E.key] = class_desc->get_line_count() - 2;

				class_desc->push_font(doc_code_font);
				class_desc->push_color(title_color);
				class_desc->add_text("enum  ");
				class_desc->pop();
				String e = E.key;
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

				// Enum description.
				if (e != "@unnamed_enums" && cd.enums.has(e)) {
					class_desc->push_color(text_color);
					class_desc->push_font(doc_font);
					class_desc->push_indent(1);
					_add_text(cd.enums[e]);
					class_desc->pop();
					class_desc->pop();
					class_desc->pop();
					class_desc->add_newline();
					class_desc->add_newline();
				}

				class_desc->push_indent(1);
				Vector<DocData::ConstantDoc> enum_list = E.value;

				Map<String, int> enumValuesContainer;
				int enumStartingLine = enum_line[E.key];

				for (int i = 0; i < enum_list.size(); i++) {
					if (cd.name == "@GlobalScope") {
						enumValuesContainer[enum_list[i].name] = enumStartingLine;
					}

					// Add the enum constant line to the constant_line map so we can locate it as a constant
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
					enum_values_line[E.key] = enumValuesContainer;
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

			if (cd.is_script_doc && (cd.properties[i].setter != "" || cd.properties[i].getter != "")) {
				class_desc->push_color(symbol_color);
				class_desc->add_text(" [" + TTR("property:") + " ");
				class_desc->pop(); // color

				if (cd.properties[i].setter != "") {
					class_desc->push_color(value_color);
					class_desc->add_text("setter");
					class_desc->pop(); // color
				}
				if (cd.properties[i].getter != "") {
					if (cd.properties[i].setter != "") {
						class_desc->push_color(symbol_color);
						class_desc->add_text(", ");
						class_desc->pop(); // color
					}
					class_desc->push_color(value_color);
					class_desc->add_text("getter");
					class_desc->pop(); // color
				}

				class_desc->push_color(symbol_color);
				class_desc->add_text("]");
				class_desc->pop(); // color
			}

			class_desc->pop(); // font
			class_desc->pop(); // cell

			// Script doc doesn't have setter, getter.
			if (!cd.is_script_doc) {
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
			}

			class_desc->pop(); // table

			class_desc->add_newline();
			class_desc->add_newline();

			class_desc->push_color(text_color);
			class_desc->push_font(doc_font);
			class_desc->push_indent(1);
			if (!cd.properties[i].description.strip_edges().is_empty()) {
				_add_text(DTR(cd.properties[i].description));
			} else {
				class_desc->add_image(get_theme_icon(SNAME("Error"), SNAME("EditorIcons")));
				class_desc->add_text(" ");
				class_desc->push_color(comment_color);
				if (cd.is_script_doc) {
					class_desc->append_text(TTR("There is currently no description for this property."));
				} else {
					class_desc->append_text(TTR("There is currently no description for this property. Please help us by [color=$color][url=$url]contributing one[/url][/color]!").replace("$url", CONTRIBUTE_URL).replace("$color", link_color_text));
				}
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

	// Constructor descriptions
	if (constructor_descriptions) {
		section_line.push_back(Pair<String, int>(TTR("Constructor Descriptions"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Constructor Descriptions"));
		_update_method_descriptions(cd, cd.constructors, "constructor");
	}

	// Method descriptions
	if (method_descriptions) {
		section_line.push_back(Pair<String, int>(TTR("Method Descriptions"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Method Descriptions"));
		_update_method_descriptions(cd, methods, "method");
	}

	// Operator descriptions
	if (operator_descriptions) {
		section_line.push_back(Pair<String, int>(TTR("Operator Descriptions"), class_desc->get_line_count() - 2));
		class_desc->push_color(title_color);
		class_desc->push_font(doc_title_font);
		class_desc->add_text(TTR("Operator Descriptions"));
		_update_method_descriptions(cd, cd.operators, "operator");
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
	if (p_topic.get_slice_count(":") == 3) {
		name = p_topic.get_slice(":", 2);
	}

	_request_help(clss); //first go to class

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
		} else if (method_line.has(name)) {
			line = method_line[name];
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

	class_desc->call_deferred(SNAME("scroll_to_paragraph"), line);
}

static void _add_text_to_rt(const String &p_bbcode, RichTextLabel *p_rt) {
	DocTools *doc = EditorHelp::get_doc_data();
	String base_path;

	Ref<Font> doc_font = p_rt->get_theme_font(SNAME("doc"), SNAME("EditorFonts"));
	Ref<Font> doc_bold_font = p_rt->get_theme_font(SNAME("doc_bold"), SNAME("EditorFonts"));
	Ref<Font> doc_code_font = p_rt->get_theme_font(SNAME("doc_source"), SNAME("EditorFonts"));
	Ref<Font> doc_kbd_font = p_rt->get_theme_font(SNAME("doc_keyboard"), SNAME("EditorFonts"));

	Color headline_color = p_rt->get_theme_color(SNAME("headline_color"), SNAME("EditorHelp"));
	Color accent_color = p_rt->get_theme_color(SNAME("accent_color"), SNAME("Editor"));
	Color property_color = p_rt->get_theme_color(SNAME("property_color"), SNAME("Editor"));
	Color link_color = accent_color.lerp(headline_color, 0.8);
	Color code_color = accent_color.lerp(headline_color, 0.6);
	Color kbd_color = accent_color.lerp(property_color, 0.6);

	String bbcode = p_bbcode.dedent().replace("\t", "").replace("\r", "").strip_edges();

	// Select the correct code examples
	switch ((int)EDITOR_GET("text_editor/help/class_reference_examples")) {
		case 0: // GDScript
			bbcode = bbcode.replace("[gdscript]", "[codeblock]");
			bbcode = bbcode.replace("[/gdscript]", "[/codeblock]");

			for (int pos = bbcode.find("[csharp]"); pos != -1; pos = bbcode.find("[csharp]")) {
				int end_pos = bbcode.find("[/csharp]");
				if (end_pos == -1) {
					WARN_PRINT("Unclosed [csharp] block or parse fail in code (search for tag errors)");
					break;
				}

				bbcode = bbcode.left(pos) + bbcode.substr(end_pos + 9); // 9 is length of "[/csharp]".
				while (bbcode[pos] == '\n') {
					bbcode = bbcode.left(pos) + bbcode.substr(pos + 1);
				}
			}
			break;
		case 1: // C#
			bbcode = bbcode.replace("[csharp]", "[codeblock]");
			bbcode = bbcode.replace("[/csharp]", "[/codeblock]");

			for (int pos = bbcode.find("[gdscript]"); pos != -1; pos = bbcode.find("[gdscript]")) {
				int end_pos = bbcode.find("[/gdscript]");
				if (end_pos == -1) {
					WARN_PRINT("Unclosed [gdscript] block or parse fail in code (search for tag errors)");
					break;
				}

				bbcode = bbcode.left(pos) + bbcode.substr(end_pos + 11); // 11 is length of "[/gdscript]".
				while (bbcode[pos] == '\n') {
					bbcode = bbcode.left(pos) + bbcode.substr(pos + 1);
				}
			}
			break;
		case 2: // GDScript and C#
			bbcode = bbcode.replace("[csharp]", "[b]C#:[/b]\n[codeblock]");
			bbcode = bbcode.replace("[gdscript]", "[b]GDScript:[/b]\n[codeblock]");

			bbcode = bbcode.replace("[/csharp]", "[/codeblock]");
			bbcode = bbcode.replace("[/gdscript]", "[/codeblock]");
			break;
	}

	// Remove codeblocks (they would be printed otherwise)
	bbcode = bbcode.replace("[codeblocks]\n", "");
	bbcode = bbcode.replace("\n[/codeblocks]", "");
	bbcode = bbcode.replace("[codeblocks]", "");
	bbcode = bbcode.replace("[/codeblocks]", "");

	// remove extra new lines around code blocks
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
			break; //nothing else to add
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

		} else if (tag.begins_with("method ") || tag.begins_with("member ") || tag.begins_with("signal ") || tag.begins_with("enum ") || tag.begins_with("constant ") || tag.begins_with("theme_item ")) {
			int tag_end = tag.find(" ");

			String link_tag = tag.substr(0, tag_end);
			String link_target = tag.substr(tag_end + 1, tag.length()).lstrip(" ");

			p_rt->push_color(link_color);
			p_rt->push_meta("@" + link_tag + " " + link_target);
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
			p_rt->push_font(doc_bold_font);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "i") {
			//use italics font
			p_rt->push_color(headline_color);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "code" || tag == "codeblock") {
			//use monospace font
			p_rt->push_font(doc_code_font);
			p_rt->push_color(code_color);
			code_tag = true;
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "kbd") {
			//use keyboard font with custom color
			p_rt->push_font(doc_kbd_font);
			p_rt->push_color(kbd_color);
			code_tag = true; // though not strictly a code tag, logic is similar
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "center") {
			//align to center
			p_rt->push_paragraph(HORIZONTAL_ALIGNMENT_CENTER, Control::TEXT_DIRECTION_AUTO, "");
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "br") {
			//force a line break
			p_rt->add_newline();
			pos = brk_end + 1;
		} else if (tag == "u") {
			//use underline
			p_rt->push_underline();
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "s") {
			//use strikethrough
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

			Ref<Texture2D> texture = ResourceLoader::load(base_path.plus_file(image), "Texture2D");
			if (texture.is_valid()) {
				p_rt->add_image(texture);
			}

			pos = end;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("color=")) {
			String col = tag.substr(6, tag.length());
			Color color = Color::from_string(col, Color());
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
			p_rt->add_text("["); //ignore
			pos = brk_pos + 1;
		}
	}
}

void EditorHelp::_add_text(const String &p_bbcode) {
	_add_text_to_rt(p_bbcode, class_desc);
}

void EditorHelp::generate_doc() {
	doc = memnew(DocTools);
	doc->generate(true);
	DocTools compdoc;
	compdoc.load_compressed(_doc_data_compressed, _doc_data_compressed_size, _doc_data_uncompressed_size);
	doc->merge_from(compdoc); //ensure all is up to date
}

void EditorHelp::_toggle_scripts_pressed() {
	ScriptEditor::get_singleton()->toggle_scripts_panel();
	update_toggle_scripts_button();
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
			update_toggle_scripts_button();
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED:
			update_toggle_scripts_button();
			break;
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

void EditorHelp::update_doc() {
	ERR_FAIL_COND(!doc->class_list.has(edited_class));
	ERR_FAIL_COND(!doc->class_list[edited_class].is_script_doc);
	_update_doc();
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
	class_desc->scroll_to_paragraph(line);
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

void EditorHelp::update_toggle_scripts_button() {
	if (is_layout_rtl()) {
		toggle_scripts_button->set_icon(get_theme_icon(ScriptEditor::get_singleton()->is_scripts_panel_toggled() ? SNAME("Forward") : SNAME("Back"), SNAME("EditorIcons")));
	} else {
		toggle_scripts_button->set_icon(get_theme_icon(ScriptEditor::get_singleton()->is_scripts_panel_toggled() ? SNAME("Back") : SNAME("Forward"), SNAME("EditorIcons")));
	}
	toggle_scripts_button->set_tooltip(vformat("%s (%s)", TTR("Toggle Scripts Panel"), ED_GET_SHORTCUT("script_editor/toggle_scripts_panel")->get_as_text()));
}

void EditorHelp::_bind_methods() {
	ClassDB::bind_method("_class_list_select", &EditorHelp::_class_list_select);
	ClassDB::bind_method("_request_help", &EditorHelp::_request_help);
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
	class_desc->add_theme_color_override("selection_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.4));

	class_desc->connect("meta_clicked", callable_mp(this, &EditorHelp::_class_desc_select));
	class_desc->connect("gui_input", callable_mp(this, &EditorHelp::_class_desc_input));
	class_desc->connect("resized", callable_mp(this, &EditorHelp::_class_desc_resized));
	_class_desc_resized();

	// Added second so it opens at the bottom so it won't offset the entire widget.
	find_bar = memnew(FindBar);
	add_child(find_bar);
	find_bar->hide();
	find_bar->set_rich_text_label(class_desc);

	status_bar = memnew(HBoxContainer);
	add_child(status_bar);
	status_bar->set_h_size_flags(SIZE_EXPAND_FILL);
	status_bar->set_custom_minimum_size(Size2(0, 24 * EDSCALE));

	toggle_scripts_button = memnew(Button);
	toggle_scripts_button->set_flat(true);
	toggle_scripts_button->connect("pressed", callable_mp(this, &EditorHelp::_toggle_scripts_pressed));
	status_bar->add_child(toggle_scripts_button);

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
	emit_signal(SNAME("request_hide"));
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

		if (m.find(".") != -1) {
			_go_to_help("class_method:" + m.get_slice(".", 0) + ":" + m.get_slice(".", 0)); //must go somewhere else
		}
	}
}

void EditorHelpBit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_text", "text"), &EditorHelpBit::set_text);
	ADD_SIGNAL(MethodInfo("request_hide"));
}

void EditorHelpBit::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			rich_text->clear();
			_add_text_to_rt(text, rich_text);

		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			rich_text->add_theme_color_override("selection_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.4));
		} break;
		default:
			break;
	}
}

void EditorHelpBit::set_text(const String &p_text) {
	text = p_text;
	rich_text->clear();
	_add_text_to_rt(text, rich_text);
}

EditorHelpBit::EditorHelpBit() {
	rich_text = memnew(RichTextLabel);
	add_child(rich_text);
	rich_text->connect("meta_clicked", callable_mp(this, &EditorHelpBit::_meta_clicked));
	rich_text->add_theme_color_override("selection_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.4));
	rich_text->set_override_selected_font_color(false);
	set_custom_minimum_size(Size2(0, 70 * EDSCALE));
}

FindBar::FindBar() {
	search_text = memnew(LineEdit);
	add_child(search_text);
	search_text->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	search_text->set_h_size_flags(SIZE_EXPAND_FILL);
	search_text->connect("text_changed", callable_mp(this, &FindBar::_search_text_changed));
	search_text->connect("text_submitted", callable_mp(this, &FindBar::_search_text_submitted));

	matches_label = memnew(Label);
	add_child(matches_label);
	matches_label->hide();

	find_prev = memnew(Button);
	find_prev->set_flat(true);
	add_child(find_prev);
	find_prev->set_focus_mode(FOCUS_NONE);
	find_prev->connect("pressed", callable_mp(this, &FindBar::search_prev));

	find_next = memnew(Button);
	find_next->set_flat(true);
	add_child(find_next);
	find_next->set_focus_mode(FOCUS_NONE);
	find_next->connect("pressed", callable_mp(this, &FindBar::search_next));

	Control *space = memnew(Control);
	add_child(space);
	space->set_custom_minimum_size(Size2(4, 0) * EDSCALE);

	hide_button = memnew(TextureButton);
	add_child(hide_button);
	hide_button->set_focus_mode(FOCUS_NONE);
	hide_button->set_expand(true);
	hide_button->set_stretch_mode(TextureButton::STRETCH_KEEP_CENTERED);
	hide_button->connect("pressed", callable_mp(this, &FindBar::_hide_bar));
}

void FindBar::popup_search() {
	show();
	bool grabbed_focus = false;
	if (!search_text->has_focus()) {
		search_text->grab_focus();
		grabbed_focus = true;
	}

	if (!search_text->get_text().is_empty()) {
		search_text->select_all();
		search_text->set_caret_column(search_text->get_text().length());
		if (grabbed_focus) {
			_search();
		}
	}
}

void FindBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			find_prev->set_icon(get_theme_icon(SNAME("MoveUp"), SNAME("EditorIcons")));
			find_next->set_icon(get_theme_icon(SNAME("MoveDown"), SNAME("EditorIcons")));
			hide_button->set_normal_texture(get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
			hide_button->set_hover_texture(get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
			hide_button->set_pressed_texture(get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
			hide_button->set_custom_minimum_size(hide_button->get_normal_texture()->get_size());
			matches_label->add_theme_color_override("font_color", results_count > 0 ? get_theme_color(SNAME("font_color"), SNAME("Label")) : get_theme_color(SNAME("error_color"), SNAME("Editor")));
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			set_process_unhandled_input(is_visible_in_tree());
		} break;
	}
}

void FindBar::_bind_methods() {
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
	if (searched.is_empty()) {
		return;
	}

	String full_text = rich_text_label->get_parsed_text();

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
	if (search_text->get_text().is_empty() || results_count == -1) {
		matches_label->hide();
	} else {
		matches_label->show();

		matches_label->add_theme_color_override("font_color", results_count > 0 ? get_theme_color(SNAME("font_color"), SNAME("Label")) : get_theme_color(SNAME("error_color"), SNAME("Editor")));
		matches_label->set_text(vformat(results_count == 1 ? TTR("%d match.") : TTR("%d matches."), results_count));
	}
}

void FindBar::_hide_bar() {
	if (search_text->has_focus()) {
		rich_text_label->grab_focus();
	}

	hide();
}

void FindBar::unhandled_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventKey> k = p_event;
	if (k.is_valid()) {
		if (k->is_pressed() && (rich_text_label->has_focus() || is_ancestor_of(get_focus_owner()))) {
			bool accepted = true;

			switch (k->get_keycode()) {
				case Key::ESCAPE: {
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

void FindBar::_search_text_submitted(const String &p_text) {
	if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		search_prev();
	} else {
		search_next();
	}
}
