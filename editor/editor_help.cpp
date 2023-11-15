/**************************************************************************/
/*  editor_help.cpp                                                       */
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

#include "editor_help.h"

#include "core/core_constants.h"
#include "core/input/input.h"
#include "core/object/script_language.h"
#include "core/os/keyboard.h"
#include "core/version.h"
#include "doc_data_compressed.gen.h"
#include "editor/editor_node.h"
#include "editor/editor_paths.h"
#include "editor/editor_property_name_processor.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/plugins/script_editor_plugin.h"
#include "scene/gui/line_edit.h"

#define CONTRIBUTE_URL vformat("%s/contributing/documentation/updating_the_class_reference.html", VERSION_DOCS_URL)

#ifdef MODULE_MONO_ENABLED
// Sync with the types mentioned in https://docs.godotengine.org/en/stable/tutorials/scripting/c_sharp/c_sharp_differences.html
const Vector<String> classes_with_csharp_differences = {
	"@GlobalScope",
	"String",
	"NodePath",
	"Signal",
	"Callable",
	"RID",
	"Basis",
	"Transform2D",
	"Transform3D",
	"Rect2",
	"Rect2i",
	"AABB",
	"Quaternion",
	"Projection",
	"Color",
	"Array",
	"Dictionary",
	"PackedByteArray",
	"PackedColorArray",
	"PackedFloat32Array",
	"PackedFloat64Array",
	"PackedInt32Array",
	"PackedInt64Array",
	"PackedStringArray",
	"PackedVector2Array",
	"PackedVector3Array",
	"Variant",
};
#endif

// TODO: this is sometimes used directly as doc->something, other times as EditorHelp::get_doc_data(), which is thread-safe.
// Might this be a problem?
DocTools *EditorHelp::doc = nullptr;

static bool _attempt_doc_load(const String &p_class) {
	// Docgen always happens in the outer-most class: it also generates docs for inner classes.
	String outer_class = p_class.get_slice(".", 0);
	if (!ScriptServer::is_global_class(outer_class)) {
		return false;
	}

	// ResourceLoader is used in order to have a script-agnostic way to load scripts.
	// This forces GDScript to compile the code, which is unnecessary for docgen, but it's a good compromise right now.
	Ref<Script> script = ResourceLoader::load(ScriptServer::get_global_class_path(outer_class), outer_class);
	if (script.is_valid()) {
		Vector<DocData::ClassDoc> docs = script->get_documentation();
		for (int j = 0; j < docs.size(); j++) {
			const DocData::ClassDoc &doc = docs.get(j);
			EditorHelp::get_doc_data()->add_doc(doc);
		}
		return true;
	}

	return false;
}

// Removes unnecessary prefix from p_class_specifier when within the p_edited_class context
static String _contextualize_class_specifier(const String &p_class_specifier, const String &p_edited_class) {
	// If this is a completely different context than the current class, then keep full path
	if (!p_class_specifier.begins_with(p_edited_class)) {
		return p_class_specifier;
	}

	// Here equal length + begins_with from above implies p_class_specifier == p_edited_class :)
	if (p_class_specifier.length() == p_edited_class.length()) {
		int rfind = p_class_specifier.rfind(".");
		if (rfind == -1) { // Single identifier
			return p_class_specifier;
		}
		// Multiple specifiers: keep last one only
		return p_class_specifier.substr(rfind + 1);
	}

	// They share a _name_ prefix but not a _class specifier_ prefix, e.g. Tree & TreeItem
	// begins_with + lengths being different implies p_class_specifier.length() > p_edited_class.length() so this is safe
	if (p_class_specifier[p_edited_class.length()] != '.') {
		return p_class_specifier;
	}

	// Remove class specifier prefix
	return p_class_specifier.substr(p_edited_class.length() + 1);
}

void EditorHelp::_update_theme_item_cache() {
	VBoxContainer::_update_theme_item_cache();

	theme_cache.text_color = get_theme_color(SNAME("text_color"), SNAME("EditorHelp"));
	theme_cache.title_color = get_theme_color(SNAME("title_color"), SNAME("EditorHelp"));
	theme_cache.headline_color = get_theme_color(SNAME("headline_color"), SNAME("EditorHelp"));
	theme_cache.comment_color = get_theme_color(SNAME("comment_color"), SNAME("EditorHelp"));
	theme_cache.symbol_color = get_theme_color(SNAME("symbol_color"), SNAME("EditorHelp"));
	theme_cache.value_color = get_theme_color(SNAME("value_color"), SNAME("EditorHelp"));
	theme_cache.qualifier_color = get_theme_color(SNAME("qualifier_color"), SNAME("EditorHelp"));
	theme_cache.type_color = get_theme_color(SNAME("type_color"), SNAME("EditorHelp"));

	theme_cache.doc_font = get_theme_font(SNAME("doc"), EditorStringName(EditorFonts));
	theme_cache.doc_bold_font = get_theme_font(SNAME("doc_bold"), EditorStringName(EditorFonts));
	theme_cache.doc_italic_font = get_theme_font(SNAME("doc_italic"), EditorStringName(EditorFonts));
	theme_cache.doc_title_font = get_theme_font(SNAME("doc_title"), EditorStringName(EditorFonts));
	theme_cache.doc_code_font = get_theme_font(SNAME("doc_source"), EditorStringName(EditorFonts));
	theme_cache.doc_kbd_font = get_theme_font(SNAME("doc_keyboard"), EditorStringName(EditorFonts));

	theme_cache.doc_font_size = get_theme_font_size(SNAME("doc_size"), EditorStringName(EditorFonts));
	theme_cache.doc_title_font_size = get_theme_font_size(SNAME("doc_title_size"), EditorStringName(EditorFonts));
	theme_cache.doc_code_font_size = get_theme_font_size(SNAME("doc_source_size"), EditorStringName(EditorFonts));
	theme_cache.doc_kbd_font_size = get_theme_font_size(SNAME("doc_keyboard_size"), EditorStringName(EditorFonts));

	theme_cache.background_style = get_theme_stylebox(SNAME("background"), SNAME("EditorHelp"));

	class_desc->begin_bulk_theme_override();
	class_desc->add_theme_font_override("normal_font", theme_cache.doc_font);
	class_desc->add_theme_font_size_override("normal_font_size", theme_cache.doc_font_size);

	class_desc->add_theme_color_override("selection_color", get_theme_color(SNAME("selection_color"), SNAME("EditorHelp")));
	class_desc->add_theme_constant_override("line_separation", get_theme_constant(SNAME("line_separation"), SNAME("EditorHelp")));
	class_desc->add_theme_constant_override("table_h_separation", get_theme_constant(SNAME("table_h_separation"), SNAME("EditorHelp")));
	class_desc->add_theme_constant_override("table_v_separation", get_theme_constant(SNAME("table_v_separation"), SNAME("EditorHelp")));
	class_desc->add_theme_constant_override("text_highlight_h_padding", get_theme_constant(SNAME("text_highlight_h_padding"), SNAME("EditorHelp")));
	class_desc->add_theme_constant_override("text_highlight_v_padding", get_theme_constant(SNAME("text_highlight_v_padding"), SNAME("EditorHelp")));
	class_desc->end_bulk_theme_override();
}

void EditorHelp::_search(bool p_search_previous) {
	if (p_search_previous) {
		find_bar->search_prev();
	} else {
		find_bar->search_next();
	}
}

void EditorHelp::_class_desc_finished() {
	if (scroll_to >= 0) {
		class_desc->scroll_to_paragraph(scroll_to);
	}
	scroll_to = -1;
}

void EditorHelp::_class_list_select(const String &p_select) {
	_goto_desc(p_select);
}

void EditorHelp::_class_desc_select(const String &p_select) {
	if (p_select.begins_with("$")) { // enum
		String select = p_select.substr(1, p_select.length());
		String class_name;
		int rfind = select.rfind(".");
		if (rfind != -1) {
			class_name = select.substr(0, rfind);
			select = select.substr(rfind + 1);
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
		HashMap<String, int> *table = nullptr;

		if (tag == "method") {
			topic = "class_method";
			table = &this->method_line;
		} else if (tag == "constructor") {
			topic = "class_method";
			table = &this->method_line;
		} else if (tag == "operator") {
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
		} else if (tag == "annotation") {
			topic = "class_annotation";
			table = &this->annotation_line;
		} else if (tag == "theme_item") {
			topic = "theme_item";
			table = &this->theme_property_line;
		} else {
			return;
		}

		// Case order is important here to correctly handle edge cases like Variant.Type in @GlobalScope.
		if (table->has(link)) {
			// Found in the current page.
			if (class_desc->is_ready()) {
				class_desc->scroll_to_paragraph((*table)[link]);
			} else {
				scroll_to = (*table)[link];
			}
		} else {
			// Look for link in @GlobalScope.
			// Note that a link like @GlobalScope.enum_name will not be found in this section, only enum_name will be.
			if (topic == "class_enum") {
				const DocData::ClassDoc &cd = doc->class_list["@GlobalScope"];

				for (int i = 0; i < cd.constants.size(); i++) {
					if (cd.constants[i].enumeration == link) {
						// Found in @GlobalScope.
						emit_signal(SNAME("go_to_help"), topic + ":@GlobalScope:" + link);
						return;
					}
				}
			} else if (topic == "class_constant") {
				const DocData::ClassDoc &cd = doc->class_list["@GlobalScope"];

				for (int i = 0; i < cd.constants.size(); i++) {
					if (cd.constants[i].name == link) {
						// Found in @GlobalScope.
						emit_signal(SNAME("go_to_help"), topic + ":@GlobalScope:" + link);
						return;
					}
				}
			}

			if (link.contains(".")) {
				int class_end = link.find(".");
				emit_signal(SNAME("go_to_help"), topic + ":" + link.substr(0, class_end) + ":" + link.substr(class_end + 1, link.length()));
			}
		}
	} else if (p_select.begins_with("http")) {
		OS::get_singleton()->shell_open(p_select);
	}
}

void EditorHelp::_class_desc_input(const Ref<InputEvent> &p_input) {
}

void EditorHelp::_class_desc_resized(bool p_force_update_theme) {
	// Add extra horizontal margins for better readability.
	// The margins increase as the width of the editor help container increases.
	real_t char_width = theme_cache.doc_code_font->get_char_size('x', theme_cache.doc_code_font_size).width;
	const int new_display_margin = MAX(30 * EDSCALE, get_parent_anchorable_rect().size.width - char_width * 120 * EDSCALE) * 0.5;
	if (display_margin != new_display_margin || p_force_update_theme) {
		display_margin = new_display_margin;

		Ref<StyleBox> class_desc_stylebox = theme_cache.background_style->duplicate();
		class_desc_stylebox->set_content_margin(SIDE_LEFT, display_margin);
		class_desc_stylebox->set_content_margin(SIDE_RIGHT, display_margin);
		class_desc->add_theme_style_override("normal", class_desc_stylebox);
		class_desc->add_theme_style_override("focused", class_desc_stylebox);
	}
}

void EditorHelp::_add_type(const String &p_type, const String &p_enum, bool p_is_bitfield) {
	if (p_type.is_empty() || p_type == "void") {
		class_desc->push_color(Color(theme_cache.type_color, 0.5));
		class_desc->push_hint(TTR("No return value."));
		class_desc->add_text("void");
		class_desc->pop();
		class_desc->pop();
		return;
	}

	bool is_enum_type = !p_enum.is_empty();
	bool is_bitfield = p_is_bitfield && is_enum_type;
	bool can_ref = !p_type.contains("*") || is_enum_type;

	String link_t = p_type; // For links in metadata
	String display_t; // For display purposes.
	if (is_enum_type) {
		link_t = p_enum; // The link for enums is always the full enum description
		display_t = _contextualize_class_specifier(p_enum, edited_class);
	} else {
		display_t = _contextualize_class_specifier(p_type, edited_class);
	}

	class_desc->push_color(theme_cache.type_color);
	bool add_array = false;
	if (can_ref) {
		if (link_t.ends_with("[]")) {
			add_array = true;
			link_t = link_t.trim_suffix("[]");
			display_t = display_t.trim_suffix("[]");

			class_desc->push_meta("#Array"); // class
			class_desc->add_text("Array");
			class_desc->pop();
			class_desc->add_text("[");
		} else if (is_bitfield) {
			class_desc->push_color(Color(theme_cache.type_color, 0.5));
			class_desc->push_hint(TTR("This value is an integer composed as a bitmask of the following flags."));
			class_desc->add_text("BitField");
			class_desc->pop();
			class_desc->add_text("[");
			class_desc->pop();
		}

		if (is_enum_type) {
			class_desc->push_meta("$" + link_t); // enum
		} else {
			class_desc->push_meta("#" + link_t); // class
		}
	}
	class_desc->add_text(display_t);
	if (can_ref) {
		class_desc->pop(); // Pushed meta above.
		if (add_array) {
			class_desc->add_text("]");
		} else if (is_bitfield) {
			class_desc->push_color(Color(theme_cache.type_color, 0.5));
			class_desc->add_text("]");
			class_desc->pop();
		}
	}
	class_desc->pop();
}

void EditorHelp::_add_type_icon(const String &p_type, int p_size, const String &p_fallback) {
	Ref<Texture2D> icon = EditorNode::get_singleton()->get_class_icon(p_type, p_fallback);
	Vector2i size = Vector2i(icon->get_width(), icon->get_height());
	if (p_size > 0) {
		// Ensures icon scales proportionally on both axes, based on icon height.
		float ratio = p_size / float(size.height);
		size.width *= ratio;
		size.height *= ratio;
	}

	class_desc->add_image(icon, size.width, size.height);
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

// Macros for assigning the deprecation/experimental information to class members
#define DEPRECATED_DOC_TAG                                                                   \
	class_desc->push_color(get_theme_color(SNAME("error_color"), EditorStringName(Editor))); \
	Ref<Texture2D> error_icon = get_editor_theme_icon(SNAME("StatusError"));                 \
	class_desc->add_text(" ");                                                               \
	class_desc->add_image(error_icon, error_icon->get_width(), error_icon->get_height());    \
	class_desc->add_text(" (" + TTR("Deprecated") + ")");                                    \
	class_desc->pop();

#define EXPERIMENTAL_DOC_TAG                                                                    \
	class_desc->push_color(get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));  \
	Ref<Texture2D> warning_icon = get_editor_theme_icon(SNAME("NodeWarning"));                  \
	class_desc->add_text(" ");                                                                  \
	class_desc->add_image(warning_icon, warning_icon->get_width(), warning_icon->get_height()); \
	class_desc->add_text(" (" + TTR("Experimental") + ")");                                     \
	class_desc->pop();

void EditorHelp::_add_method(const DocData::MethodDoc &p_method, bool p_overview, bool p_override) {
	if (p_override) {
		method_line[p_method.name] = class_desc->get_paragraph_count() - 2; // Gets overridden if description.
	}

	const bool is_vararg = p_method.qualifiers.contains("vararg");

	if (p_overview) {
		class_desc->push_cell();
		class_desc->push_paragraph(HORIZONTAL_ALIGNMENT_RIGHT, Control::TEXT_DIRECTION_AUTO, "");
	} else {
		_add_bulletpoint();
	}

	_add_type(p_method.return_type, p_method.return_enum, p_method.return_is_bitfield);

	if (p_overview) {
		class_desc->pop(); // align
		class_desc->pop(); // cell
		class_desc->push_cell();
	} else {
		class_desc->add_text(" ");
	}

	if (p_overview && !p_method.description.strip_edges().is_empty()) {
		class_desc->push_meta("@method " + p_method.name);
	}

	class_desc->push_color(theme_cache.headline_color);
	_add_text(p_method.name);
	class_desc->pop();

	if (p_overview && !p_method.description.strip_edges().is_empty()) {
		class_desc->pop(); // meta
	}

	class_desc->push_color(theme_cache.symbol_color);
	class_desc->add_text("(");
	class_desc->pop();

	for (int j = 0; j < p_method.arguments.size(); j++) {
		class_desc->push_color(theme_cache.text_color);
		if (j > 0) {
			class_desc->add_text(", ");
		}

		_add_text(p_method.arguments[j].name);
		class_desc->add_text(": ");
		_add_type(p_method.arguments[j].type, p_method.arguments[j].enumeration, p_method.arguments[j].is_bitfield);
		if (!p_method.arguments[j].default_value.is_empty()) {
			class_desc->push_color(theme_cache.symbol_color);
			class_desc->add_text(" = ");
			class_desc->pop();
			class_desc->push_color(theme_cache.value_color);
			class_desc->add_text(_fix_constant(p_method.arguments[j].default_value));
			class_desc->pop();
		}

		class_desc->pop();
	}

	if (is_vararg) {
		class_desc->push_color(theme_cache.text_color);
		if (p_method.arguments.size()) {
			class_desc->add_text(", ");
		}
		class_desc->push_color(theme_cache.symbol_color);
		class_desc->add_text("...");
		class_desc->pop();
		class_desc->pop();
	}

	class_desc->push_color(theme_cache.symbol_color);
	class_desc->add_text(")");
	class_desc->pop();
	if (!p_method.qualifiers.is_empty()) {
		class_desc->push_color(theme_cache.qualifier_color);

		PackedStringArray qualifiers = p_method.qualifiers.split_spaces();
		for (const String &qualifier : qualifiers) {
			String hint;
			if (qualifier == "vararg") {
				hint = TTR("This method supports a variable number of arguments.");
			} else if (qualifier == "virtual") {
				hint = TTR("This method is called by the engine.\nIt can be overridden to customize built-in behavior.");
			} else if (qualifier == "const") {
				hint = TTR("This method has no side effects.\nIt does not modify the object in any way.");
			} else if (qualifier == "static") {
				hint = TTR("This method does not need an instance to be called.\nIt can be called directly using the class name.");
			}

			class_desc->add_text(" ");
			if (!hint.is_empty()) {
				class_desc->push_hint(hint);
				class_desc->add_text(qualifier);
				class_desc->pop();
			} else {
				class_desc->add_text(qualifier);
			}
		}
		class_desc->pop();
	}

	if (p_method.is_deprecated) {
		DEPRECATED_DOC_TAG;
	}

	if (p_method.is_experimental) {
		EXPERIMENTAL_DOC_TAG;
	}

	if (p_overview) {
		class_desc->pop(); // cell
	}
}

void EditorHelp::_add_bulletpoint() {
	static const char32_t prefix[3] = { 0x25CF /* filled circle */, ' ', 0 };
	class_desc->add_text(String(prefix));
}

void EditorHelp::_push_normal_font() {
	class_desc->push_font(theme_cache.doc_font);
	class_desc->push_font_size(theme_cache.doc_font_size);
}

void EditorHelp::_pop_normal_font() {
	class_desc->pop();
	class_desc->pop();
}

void EditorHelp::_push_title_font() {
	class_desc->push_color(theme_cache.title_color);
	class_desc->push_font(theme_cache.doc_title_font);
	class_desc->push_font_size(theme_cache.doc_title_font_size);
}

void EditorHelp::_pop_title_font() {
	class_desc->pop();
	class_desc->pop();
	class_desc->pop();
}

void EditorHelp::_push_code_font() {
	class_desc->push_font(theme_cache.doc_code_font);
	class_desc->push_font_size(theme_cache.doc_code_font_size);
}

void EditorHelp::_pop_code_font() {
	class_desc->pop();
	class_desc->pop();
}

Error EditorHelp::_goto_desc(const String &p_class) {
	// If class doesn't have docs listed, attempt on-demand docgen
	if (!doc->class_list.has(p_class) && !_attempt_doc_load(p_class)) {
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

void EditorHelp::_update_method_list(const Vector<DocData::MethodDoc> p_methods, MethodType p_method_type) {
	class_desc->add_newline();

	_push_code_font();
	class_desc->push_indent(1);
	class_desc->push_table(2);
	class_desc->set_table_column_expand(1, true);

	bool any_previous = false;
	for (int pass = 0; pass < 2; pass++) {
		Vector<DocData::MethodDoc> m;

		for (int i = 0; i < p_methods.size(); i++) {
			const String &q = p_methods[i].qualifiers;
			if ((pass == 0 && q.contains("virtual")) || (pass == 1 && !q.contains("virtual"))) {
				m.push_back(p_methods[i]);
			}
		}

		if (any_previous && !m.is_empty()) {
			class_desc->push_cell();
			class_desc->pop(); // cell
			class_desc->push_cell();
			class_desc->pop(); // cell
		}

		String group_prefix;
		for (int i = 0; i < m.size(); i++) {
			const String new_prefix = m[i].name.substr(0, 3);
			bool is_new_group = false;

			if (i < m.size() - 1 && new_prefix == m[i + 1].name.substr(0, 3) && new_prefix != group_prefix) {
				is_new_group = i > 0;
				group_prefix = new_prefix;
			} else if (!group_prefix.is_empty() && new_prefix != group_prefix) {
				is_new_group = true;
				group_prefix = "";
			}

			if (is_new_group && pass == 1) {
				class_desc->push_cell();
				class_desc->pop(); // cell
				class_desc->push_cell();
				class_desc->pop(); // cell
			}

			// For constructors always point to the first one.
			_add_method(m[i], true, (p_method_type != METHOD_TYPE_CONSTRUCTOR || i == 0));
		}

		any_previous = !m.is_empty();
	}

	class_desc->pop(); // table
	class_desc->pop();
	_pop_code_font();

	class_desc->add_newline();
	class_desc->add_newline();
}

void EditorHelp::_update_method_descriptions(const DocData::ClassDoc p_classdoc, const Vector<DocData::MethodDoc> p_methods, MethodType p_method_type) {
	String link_color_text = theme_cache.title_color.to_html(false);

	class_desc->add_newline();
	class_desc->add_newline();

	for (int pass = 0; pass < 2; pass++) {
		Vector<DocData::MethodDoc> methods_filtered;

		for (int i = 0; i < p_methods.size(); i++) {
			const String &q = p_methods[i].qualifiers;
			if ((pass == 0 && q.contains("virtual")) || (pass == 1 && !q.contains("virtual"))) {
				methods_filtered.push_back(p_methods[i]);
			}
		}

		for (int i = 0; i < methods_filtered.size(); i++) {
			_push_code_font();
			// For constructors always point to the first one.
			_add_method(methods_filtered[i], false, (p_method_type != METHOD_TYPE_CONSTRUCTOR || i == 0));
			_pop_code_font();

			class_desc->add_newline();
			class_desc->add_newline();

			class_desc->push_color(theme_cache.text_color);
			_push_normal_font();
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
				class_desc->add_image(get_editor_theme_icon(SNAME("Error")));
				class_desc->add_text(" ");
				class_desc->push_color(theme_cache.comment_color);

				String message;
				if (p_classdoc.is_script_doc) {
					static const char *messages_by_type[METHOD_TYPE_MAX] = {
						TTRC("There is currently no description for this method."),
						TTRC("There is currently no description for this constructor."),
						TTRC("There is currently no description for this operator."),
					};
					message = TTRGET(messages_by_type[p_method_type]);
				} else {
					static const char *messages_by_type[METHOD_TYPE_MAX] = {
						TTRC("There is currently no description for this method. Please help us by [color=$color][url=$url]contributing one[/url][/color]!"),
						TTRC("There is currently no description for this constructor. Please help us by [color=$color][url=$url]contributing one[/url][/color]!"),
						TTRC("There is currently no description for this operator. Please help us by [color=$color][url=$url]contributing one[/url][/color]!"),
					};
					message = TTRGET(messages_by_type[p_method_type]).replace("$url", CONTRIBUTE_URL).replace("$color", link_color_text);
				}
				class_desc->append_text(message);
				class_desc->pop();
			}

			class_desc->pop();
			_pop_normal_font();
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

	String link_color_text = theme_cache.title_color.to_html(false);

	DocData::ClassDoc cd = doc->class_list[edited_class]; // Make a copy, so we can sort without worrying.

	// Class name
	section_line.push_back(Pair<String, int>(TTR("Top"), 0));
	_push_title_font();
	class_desc->add_text(TTR("Class:") + " ");
	_add_type_icon(edited_class, theme_cache.doc_title_font_size, "Object");
	class_desc->add_text(" ");
	class_desc->push_color(theme_cache.headline_color);
	_add_text(edited_class);
	class_desc->pop(); // color
	_pop_title_font();

	if (cd.is_deprecated) {
		class_desc->add_text(" ");
		Ref<Texture2D> error_icon = get_editor_theme_icon(SNAME("StatusError"));
		class_desc->add_image(error_icon, error_icon->get_width(), error_icon->get_height());
	}
	if (cd.is_experimental) {
		class_desc->add_text(" ");
		Ref<Texture2D> warning_icon = get_editor_theme_icon(SNAME("NodeWarning"));
		class_desc->add_image(warning_icon, warning_icon->get_width(), warning_icon->get_height());
	}
	class_desc->add_newline();

	const String non_breaking_space = String::chr(160);

	// Inheritance tree

	// Ascendents
	if (!cd.inherits.is_empty()) {
		class_desc->push_color(theme_cache.title_color);
		_push_normal_font();
		class_desc->add_text(TTR("Inherits:") + " ");

		String inherits = cd.inherits;

		while (!inherits.is_empty()) {
			_add_type_icon(inherits, theme_cache.doc_font_size, "ArrowRight");
			class_desc->add_text(non_breaking_space); // Otherwise icon borrows hyperlink from _add_type().
			_add_type(inherits);

			inherits = doc->class_list[inherits].inherits;

			if (!inherits.is_empty()) {
				class_desc->add_text(" < ");
			}
		}

		_pop_normal_font();
		class_desc->pop();
		class_desc->add_newline();
	}

	// Descendents
	if (cd.is_script_doc || ClassDB::class_exists(cd.name)) {
		bool found = false;
		bool prev = false;

		_push_normal_font();
		for (const KeyValue<String, DocData::ClassDoc> &E : doc->class_list) {
			if (E.value.inherits == cd.name) {
				if (!found) {
					class_desc->push_color(theme_cache.title_color);
					class_desc->add_text(TTR("Inherited by:") + " ");
					found = true;
				}

				if (prev) {
					class_desc->add_text(" , ");
				}
				_add_type_icon(E.value.name, theme_cache.doc_font_size, "ArrowRight");
				class_desc->add_text(non_breaking_space); // Otherwise icon borrows hyperlink from _add_type().
				_add_type(E.value.name);
				prev = true;
			}
		}
		_pop_normal_font();

		if (found) {
			class_desc->pop();
			class_desc->add_newline();
		}
	}

	// Note if deprecated.
	if (cd.is_deprecated) {
		Ref<Texture2D> error_icon = get_editor_theme_icon(SNAME("StatusError"));
		class_desc->push_color(get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
		class_desc->add_image(error_icon, error_icon->get_width(), error_icon->get_height());
		class_desc->add_text(String(" ") + TTR("This class is marked as deprecated. It will be removed in future versions."));
		class_desc->pop();
		class_desc->add_newline();
	}

	// Note if experimental.
	if (cd.is_experimental) {
		Ref<Texture2D> warning_icon = get_editor_theme_icon(SNAME("NodeWarning"));
		class_desc->push_color(get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
		class_desc->add_image(warning_icon, warning_icon->get_width(), warning_icon->get_height());
		class_desc->add_text(String(" ") + TTR("This class is marked as experimental. It is subject to likely change or possible removal in future versions. Use at your own discretion."));
		class_desc->pop();
		class_desc->add_newline();
	}

	bool has_description = false;

	class_desc->add_newline();
	class_desc->add_newline();

	// Brief description
	if (!cd.brief_description.strip_edges().is_empty()) {
		has_description = true;

		class_desc->push_color(theme_cache.text_color);
		class_desc->push_font(theme_cache.doc_bold_font);
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
	if (!cd.description.strip_edges().is_empty()) {
		has_description = true;

		section_line.push_back(Pair<String, int>(TTR("Description"), class_desc->get_paragraph_count() - 2));
		description_line = class_desc->get_paragraph_count() - 2;
		_push_title_font();
		class_desc->add_text(TTR("Description"));
		_pop_title_font();

		class_desc->add_newline();
		class_desc->add_newline();
		class_desc->push_color(theme_cache.text_color);
		_push_normal_font();
		class_desc->push_indent(1);
		_add_text(DTR(cd.description));
		class_desc->pop();
		_pop_normal_font();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();
		class_desc->add_newline();
	}

	if (!has_description) {
		class_desc->add_image(get_editor_theme_icon(SNAME("Error")));
		class_desc->add_text(" ");
		class_desc->push_color(theme_cache.comment_color);

		if (cd.is_script_doc) {
			class_desc->append_text(TTR("There is currently no description for this class."));
		} else {
			class_desc->append_text(TTR("There is currently no description for this class. Please help us by [color=$color][url=$url]contributing one[/url][/color]!").replace("$url", CONTRIBUTE_URL).replace("$color", link_color_text));
		}

		class_desc->add_newline();
		class_desc->add_newline();
	}

#ifdef MODULE_MONO_ENABLED
	if (classes_with_csharp_differences.has(cd.name)) {
		const String &csharp_differences_url = vformat("%s/tutorials/scripting/c_sharp/c_sharp_differences.html", VERSION_DOCS_URL);

		class_desc->push_color(theme_cache.text_color);
		_push_normal_font();
		class_desc->push_indent(1);
		_add_text("[b]" + TTR("Note:") + "[/b] " + vformat(TTR("There are notable differences when using this API with C#. See [url=%s]C# API differences to GDScript[/url] for more information."), csharp_differences_url));
		class_desc->pop();
		_pop_normal_font();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();
	}
#endif

	// Online tutorials
	if (cd.tutorials.size()) {
		_push_title_font();
		class_desc->add_text(TTR("Online Tutorials"));
		_pop_title_font();

		class_desc->add_newline();

		class_desc->push_indent(1);
		_push_code_font();

		for (int i = 0; i < cd.tutorials.size(); i++) {
			const String link = DTR(cd.tutorials[i].link);
			String linktxt = (cd.tutorials[i].title.is_empty()) ? link : DTR(cd.tutorials[i].title);
			const int seppos = linktxt.find("//");
			if (seppos != -1) {
				linktxt = link.substr(seppos + 2);
			}

			class_desc->push_color(theme_cache.symbol_color);
			class_desc->append_text("[url=" + link + "]" + linktxt + "[/url]");
			class_desc->pop();
			class_desc->add_newline();
		}

		_pop_code_font();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();
	}

	// Properties overview
	HashSet<String> skip_methods;

	bool has_properties = false;
	bool has_property_descriptions = false;
	for (const DocData::PropertyDoc &prop : cd.properties) {
		if (cd.is_script_doc && prop.name.begins_with("_") && prop.description.strip_edges().is_empty()) {
			continue;
		}
		has_properties = true;
		if (!prop.overridden) {
			has_property_descriptions = true;
			break;
		}
	}

	if (has_properties) {
		section_line.push_back(Pair<String, int>(TTR("Properties"), class_desc->get_paragraph_count() - 2));
		_push_title_font();
		class_desc->add_text(TTR("Properties"));
		_pop_title_font();

		class_desc->add_newline();

		_push_code_font();
		class_desc->push_indent(1);
		class_desc->push_table(4);
		class_desc->set_table_column_expand(1, true);

		for (int i = 0; i < cd.properties.size(); i++) {
			// Ignore undocumented private.
			if (cd.properties[i].name.begins_with("_") && cd.properties[i].description.strip_edges().is_empty()) {
				continue;
			}
			property_line[cd.properties[i].name] = class_desc->get_paragraph_count() - 2; //gets overridden if description

			// Property type.
			class_desc->push_cell();
			class_desc->push_paragraph(HORIZONTAL_ALIGNMENT_RIGHT, Control::TEXT_DIRECTION_AUTO, "");
			_push_code_font();
			_add_type(cd.properties[i].type, cd.properties[i].enumeration, cd.properties[i].is_bitfield);
			_pop_code_font();
			class_desc->pop();
			class_desc->pop(); // cell

			bool describe = false;

			if (!cd.properties[i].setter.is_empty()) {
				skip_methods.insert(cd.properties[i].setter);
				describe = true;
			}
			if (!cd.properties[i].getter.is_empty()) {
				skip_methods.insert(cd.properties[i].getter);
				describe = true;
			}

			if (!cd.properties[i].description.strip_edges().is_empty()) {
				describe = true;
			}

			if (cd.properties[i].overridden) {
				describe = false;
			}

			// Property name.
			class_desc->push_cell();
			_push_code_font();
			class_desc->push_color(theme_cache.headline_color);

			if (describe) {
				class_desc->push_meta("@member " + cd.properties[i].name);
			}

			_add_text(cd.properties[i].name);

			if (describe) {
				class_desc->pop();
			}

			class_desc->pop();
			_pop_code_font();
			class_desc->pop(); // cell

			// Property value.
			class_desc->push_cell();
			_push_code_font();

			if (!cd.properties[i].default_value.is_empty()) {
				class_desc->push_color(theme_cache.symbol_color);
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

				class_desc->push_color(theme_cache.value_color);
				class_desc->add_text(_fix_constant(cd.properties[i].default_value));
				class_desc->pop();

				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text("]");
				class_desc->pop();
			}

			if (cd.properties[i].is_deprecated) {
				DEPRECATED_DOC_TAG;
			}

			if (cd.properties[i].is_experimental) {
				EXPERIMENTAL_DOC_TAG;
			}

			_pop_code_font();
			class_desc->pop(); // cell

			// Property setters and getters.
			class_desc->push_cell();
			_push_code_font();

			if (cd.is_script_doc && (!cd.properties[i].setter.is_empty() || !cd.properties[i].getter.is_empty())) {
				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text(" [" + TTR("property:") + " ");
				class_desc->pop(); // color

				if (!cd.properties[i].setter.is_empty()) {
					class_desc->push_color(theme_cache.value_color);
					class_desc->add_text("setter");
					class_desc->pop(); // color
				}
				if (!cd.properties[i].getter.is_empty()) {
					if (!cd.properties[i].setter.is_empty()) {
						class_desc->push_color(theme_cache.symbol_color);
						class_desc->add_text(", ");
						class_desc->pop(); // color
					}
					class_desc->push_color(theme_cache.value_color);
					class_desc->add_text("getter");
					class_desc->pop(); // color
				}

				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text("]");
				class_desc->pop(); // color
			}

			_pop_code_font();
			class_desc->pop(); // cell
		}

		class_desc->pop(); // table
		class_desc->pop();
		_pop_code_font();

		class_desc->add_newline();
		class_desc->add_newline();
	}

	// Methods overview
	bool sort_methods = EDITOR_GET("text_editor/help/sort_functions_alphabetically");

	Vector<DocData::MethodDoc> methods;

	for (int i = 0; i < cd.methods.size(); i++) {
		if (skip_methods.has(cd.methods[i].name)) {
			if (cd.methods[i].arguments.size() == 0 /* getter */ || (cd.methods[i].arguments.size() == 1 && cd.methods[i].return_type == "void" /* setter */)) {
				continue;
			}
		}
		// Ignore undocumented non virtual private.
		if (cd.methods[i].name.begins_with("_") && cd.methods[i].description.strip_edges().is_empty() && !cd.methods[i].qualifiers.contains("virtual")) {
			continue;
		}
		methods.push_back(cd.methods[i]);
	}

	if (!cd.constructors.is_empty()) {
		if (sort_methods) {
			cd.constructors.sort();
		}

		section_line.push_back(Pair<String, int>(TTR("Constructors"), class_desc->get_paragraph_count() - 2));
		_push_title_font();
		class_desc->add_text(TTR("Constructors"));
		_pop_title_font();

		_update_method_list(cd.constructors, METHOD_TYPE_CONSTRUCTOR);
	}

	if (!methods.is_empty()) {
		if (sort_methods) {
			methods.sort();
		}

		section_line.push_back(Pair<String, int>(TTR("Methods"), class_desc->get_paragraph_count() - 2));
		_push_title_font();
		class_desc->add_text(TTR("Methods"));
		_pop_title_font();

		_update_method_list(methods, METHOD_TYPE_METHOD);
	}

	if (!cd.operators.is_empty()) {
		if (sort_methods) {
			cd.operators.sort();
		}

		section_line.push_back(Pair<String, int>(TTR("Operators"), class_desc->get_paragraph_count() - 2));
		_push_title_font();
		class_desc->add_text(TTR("Operators"));
		_pop_title_font();

		_update_method_list(cd.operators, METHOD_TYPE_OPERATOR);
	}

	// Theme properties
	if (!cd.theme_properties.is_empty()) {
		section_line.push_back(Pair<String, int>(TTR("Theme Properties"), class_desc->get_paragraph_count() - 2));
		_push_title_font();
		class_desc->add_text(TTR("Theme Properties"));
		_pop_title_font();

		class_desc->add_newline();
		class_desc->add_newline();

		class_desc->push_indent(1);

		String theme_data_type;
		HashMap<String, String> data_type_names;
		data_type_names["color"] = TTR("Colors");
		data_type_names["constant"] = TTR("Constants");
		data_type_names["font"] = TTR("Fonts");
		data_type_names["font_size"] = TTR("Font Sizes");
		data_type_names["icon"] = TTR("Icons");
		data_type_names["style"] = TTR("Styles");

		for (int i = 0; i < cd.theme_properties.size(); i++) {
			theme_property_line[cd.theme_properties[i].name] = class_desc->get_paragraph_count() - 2; // Gets overridden if description.

			if (theme_data_type != cd.theme_properties[i].data_type) {
				theme_data_type = cd.theme_properties[i].data_type;

				_push_title_font();
				if (data_type_names.has(theme_data_type)) {
					class_desc->add_text(data_type_names[theme_data_type]);
				} else {
					class_desc->add_text("");
				}
				_pop_title_font();

				class_desc->add_newline();
				class_desc->add_newline();
			}

			// Theme item header.
			_push_code_font();
			_add_bulletpoint();

			// Theme item object type.
			_add_type(cd.theme_properties[i].type);

			// Theme item name.
			class_desc->push_color(theme_cache.headline_color);
			class_desc->add_text(" ");
			_add_text(cd.theme_properties[i].name);
			class_desc->pop();

			// Theme item default value.
			if (!cd.theme_properties[i].default_value.is_empty()) {
				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text(" [" + TTR("default:") + " ");
				class_desc->pop();
				class_desc->push_color(theme_cache.value_color);
				class_desc->add_text(_fix_constant(cd.theme_properties[i].default_value));
				class_desc->pop();
				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text("]");
				class_desc->pop();
			}

			_pop_code_font();

			// Theme item description.
			if (!cd.theme_properties[i].description.strip_edges().is_empty()) {
				class_desc->push_color(theme_cache.comment_color);
				_push_normal_font();
				class_desc->push_indent(1);
				_add_text(DTR(cd.theme_properties[i].description));
				class_desc->pop(); // indent
				_pop_normal_font();
				class_desc->pop(); // color
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

		section_line.push_back(Pair<String, int>(TTR("Signals"), class_desc->get_paragraph_count() - 2));
		_push_title_font();
		class_desc->add_text(TTR("Signals"));
		_pop_title_font();

		class_desc->add_newline();
		class_desc->add_newline();

		class_desc->push_indent(1);

		for (int i = 0; i < cd.signals.size(); i++) {
			signal_line[cd.signals[i].name] = class_desc->get_paragraph_count() - 2; // Gets overridden if description.

			_push_code_font();
			_add_bulletpoint();
			class_desc->push_color(theme_cache.headline_color);
			_add_text(cd.signals[i].name);
			class_desc->pop();
			class_desc->push_color(theme_cache.symbol_color);
			class_desc->add_text("(");
			class_desc->pop();
			for (int j = 0; j < cd.signals[i].arguments.size(); j++) {
				class_desc->push_color(theme_cache.text_color);
				if (j > 0) {
					class_desc->add_text(", ");
				}

				_add_text(cd.signals[i].arguments[j].name);
				class_desc->add_text(": ");
				_add_type(cd.signals[i].arguments[j].type, cd.signals[i].arguments[j].enumeration, cd.signals[i].arguments[j].is_bitfield);
				if (!cd.signals[i].arguments[j].default_value.is_empty()) {
					class_desc->push_color(theme_cache.symbol_color);
					class_desc->add_text(" = ");
					class_desc->pop();
					_add_text(cd.signals[i].arguments[j].default_value);
				}

				class_desc->pop();
			}

			class_desc->push_color(theme_cache.symbol_color);
			class_desc->add_text(")");

			if (cd.signals[i].is_deprecated) {
				DEPRECATED_DOC_TAG;
			}

			if (cd.signals[i].is_experimental) {
				EXPERIMENTAL_DOC_TAG;
			}

			class_desc->pop();
			_pop_code_font();

			if (!cd.signals[i].description.strip_edges().is_empty()) {
				class_desc->push_color(theme_cache.comment_color);
				_push_normal_font();
				class_desc->push_indent(1);
				_add_text(DTR(cd.signals[i].description));
				class_desc->pop(); // indent
				_pop_normal_font();
				class_desc->pop(); // color
			}

			class_desc->add_newline();
			class_desc->add_newline();
		}

		class_desc->pop();
		class_desc->add_newline();
	}

	// Constants and enums
	if (!cd.constants.is_empty()) {
		HashMap<String, Vector<DocData::ConstantDoc>> enums;
		Vector<DocData::ConstantDoc> constants;

		for (int i = 0; i < cd.constants.size(); i++) {
			if (!cd.constants[i].enumeration.is_empty()) {
				if (!enums.has(cd.constants[i].enumeration)) {
					enums[cd.constants[i].enumeration] = Vector<DocData::ConstantDoc>();
				}

				enums[cd.constants[i].enumeration].push_back(cd.constants[i]);
			} else {
				// Ignore undocumented private.
				if (cd.constants[i].name.begins_with("_") && cd.constants[i].description.strip_edges().is_empty()) {
					continue;
				}
				constants.push_back(cd.constants[i]);
			}
		}

		// Enums
		if (enums.size()) {
			section_line.push_back(Pair<String, int>(TTR("Enumerations"), class_desc->get_paragraph_count() - 2));
			_push_title_font();
			class_desc->add_text(TTR("Enumerations"));
			_pop_title_font();
			class_desc->push_indent(1);

			class_desc->add_newline();

			for (KeyValue<String, Vector<DocData::ConstantDoc>> &E : enums) {
				enum_line[E.key] = class_desc->get_paragraph_count() - 2;

				_push_code_font();

				class_desc->push_color(theme_cache.title_color);
				if (E.value.size() && E.value[0].is_bitfield) {
					class_desc->add_text("flags  ");
				} else {
					class_desc->add_text("enum  ");
				}
				class_desc->pop();

				String e = E.key;
				if ((e.get_slice_count(".") > 1) && (e.get_slice(".", 0) == edited_class)) {
					e = e.get_slice(".", 1);
				}

				class_desc->push_color(theme_cache.headline_color);
				class_desc->add_text(e);
				class_desc->pop();

				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text(":");
				class_desc->pop();

				if (cd.enums.has(e)) {
					if (cd.enums[e].is_deprecated) {
						DEPRECATED_DOC_TAG;
					}

					if (cd.enums[e].is_experimental) {
						EXPERIMENTAL_DOC_TAG;
					}
				}

				_pop_code_font();

				class_desc->add_newline();
				class_desc->add_newline();

				// Enum description.
				if (e != "@unnamed_enums" && cd.enums.has(e) && !cd.enums[e].description.strip_edges().is_empty()) {
					class_desc->push_color(theme_cache.text_color);
					_push_normal_font();
					class_desc->push_indent(1);
					_add_text(cd.enums[e].description);
					class_desc->pop();
					_pop_normal_font();
					class_desc->pop();

					class_desc->add_newline();
					class_desc->add_newline();
				}

				class_desc->push_indent(1);
				Vector<DocData::ConstantDoc> enum_list = E.value;

				HashMap<String, int> enumValuesContainer;
				int enumStartingLine = enum_line[E.key];

				for (int i = 0; i < enum_list.size(); i++) {
					if (cd.name == "@GlobalScope") {
						enumValuesContainer[enum_list[i].name] = enumStartingLine;
					}

					// Add the enum constant line to the constant_line map so we can locate it as a constant.
					constant_line[enum_list[i].name] = class_desc->get_paragraph_count() - 2;

					_push_code_font();

					_add_bulletpoint();
					class_desc->push_color(theme_cache.headline_color);
					_add_text(enum_list[i].name);
					class_desc->pop();
					class_desc->push_color(theme_cache.symbol_color);
					class_desc->add_text(" = ");
					class_desc->pop();
					class_desc->push_color(theme_cache.value_color);
					class_desc->add_text(_fix_constant(enum_list[i].value));
					class_desc->pop();

					if (enum_list[i].is_deprecated) {
						DEPRECATED_DOC_TAG;
					}

					if (enum_list[i].is_experimental) {
						EXPERIMENTAL_DOC_TAG;
					}

					_pop_code_font();

					class_desc->add_newline();

					if (!enum_list[i].description.strip_edges().is_empty()) {
						class_desc->push_color(theme_cache.comment_color);
						_push_normal_font();
						_add_text(DTR(enum_list[i].description));
						_pop_normal_font();
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
			section_line.push_back(Pair<String, int>(TTR("Constants"), class_desc->get_paragraph_count() - 2));
			_push_title_font();
			class_desc->add_text(TTR("Constants"));
			_pop_title_font();
			class_desc->push_indent(1);

			class_desc->add_newline();

			for (int i = 0; i < constants.size(); i++) {
				constant_line[constants[i].name] = class_desc->get_paragraph_count() - 2;

				_push_code_font();

				if (constants[i].value.begins_with("Color(") && constants[i].value.ends_with(")")) {
					String stripped = constants[i].value.replace(" ", "").replace("Color(", "").replace(")", "");
					PackedFloat64Array color = stripped.split_floats(",");
					if (color.size() >= 3) {
						class_desc->push_color(Color(color[0], color[1], color[2]));
						_add_bulletpoint();
						class_desc->pop();
					}
				} else {
					_add_bulletpoint();
				}

				class_desc->push_color(theme_cache.headline_color);
				_add_text(constants[i].name);
				class_desc->pop();
				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text(" = ");
				class_desc->pop();
				class_desc->push_color(theme_cache.value_color);
				class_desc->add_text(_fix_constant(constants[i].value));
				class_desc->pop();

				if (constants[i].is_deprecated) {
					DEPRECATED_DOC_TAG;
				}

				if (constants[i].is_experimental) {
					EXPERIMENTAL_DOC_TAG;
				}

				_pop_code_font();

				class_desc->add_newline();

				if (!constants[i].description.strip_edges().is_empty()) {
					class_desc->push_color(theme_cache.comment_color);
					_push_normal_font();
					_add_text(DTR(constants[i].description));
					_pop_normal_font();
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

	// Annotations
	if (!cd.annotations.is_empty()) {
		if (sort_methods) {
			cd.annotations.sort();
		}

		section_line.push_back(Pair<String, int>(TTR("Annotations"), class_desc->get_paragraph_count() - 2));
		_push_title_font();
		class_desc->add_text(TTR("Annotations"));
		_pop_title_font();

		class_desc->add_newline();
		class_desc->add_newline();

		class_desc->push_indent(1);

		for (int i = 0; i < cd.annotations.size(); i++) {
			annotation_line[cd.annotations[i].name] = class_desc->get_paragraph_count() - 2; // Gets overridden if description.

			_push_code_font();
			_add_bulletpoint();
			class_desc->push_color(theme_cache.headline_color);
			_add_text(cd.annotations[i].name);
			class_desc->pop();

			if (cd.annotations[i].arguments.size() > 0) {
				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text("(");
				class_desc->pop();
				for (int j = 0; j < cd.annotations[i].arguments.size(); j++) {
					class_desc->push_color(theme_cache.text_color);
					if (j > 0) {
						class_desc->add_text(", ");
					}

					_add_text(cd.annotations[i].arguments[j].name);
					class_desc->add_text(": ");
					_add_type(cd.annotations[i].arguments[j].type);
					if (!cd.annotations[i].arguments[j].default_value.is_empty()) {
						class_desc->push_color(theme_cache.symbol_color);
						class_desc->add_text(" = ");
						class_desc->pop();
						_add_text(cd.annotations[i].arguments[j].default_value);
					}

					class_desc->pop();
				}

				if (cd.annotations[i].qualifiers.contains("vararg")) {
					class_desc->push_color(theme_cache.text_color);
					if (cd.annotations[i].arguments.size()) {
						class_desc->add_text(", ");
					}
					class_desc->push_color(theme_cache.symbol_color);
					class_desc->add_text("...");
					class_desc->pop();
					class_desc->pop();
				}

				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text(")");
				class_desc->pop();
			}

			if (!cd.annotations[i].qualifiers.is_empty()) {
				class_desc->push_color(theme_cache.qualifier_color);
				class_desc->add_text(" ");
				_add_text(cd.annotations[i].qualifiers);
				class_desc->pop();
			}

			_pop_code_font();

			if (!cd.annotations[i].description.strip_edges().is_empty()) {
				class_desc->push_color(theme_cache.comment_color);
				_push_normal_font();
				class_desc->push_indent(1);
				_add_text(DTR(cd.annotations[i].description));
				class_desc->pop(); // indent
				_pop_normal_font();
				class_desc->pop(); // color
			} else {
				class_desc->push_indent(1);
				class_desc->add_image(get_editor_theme_icon(SNAME("Error")));
				class_desc->add_text(" ");
				class_desc->push_color(theme_cache.comment_color);
				if (cd.is_script_doc) {
					class_desc->append_text(TTR("There is currently no description for this annotation."));
				} else {
					class_desc->append_text(TTR("There is currently no description for this annotation. Please help us by [color=$color][url=$url]contributing one[/url][/color]!").replace("$url", CONTRIBUTE_URL).replace("$color", link_color_text));
				}
				class_desc->pop();
				class_desc->pop(); // indent
			}
			class_desc->add_newline();
			class_desc->add_newline();
		}

		class_desc->pop();
		class_desc->add_newline();
	}

	// Property descriptions
	if (has_property_descriptions) {
		section_line.push_back(Pair<String, int>(TTR("Property Descriptions"), class_desc->get_paragraph_count() - 2));
		_push_title_font();
		class_desc->add_text(TTR("Property Descriptions"));
		_pop_title_font();

		class_desc->add_newline();
		class_desc->add_newline();

		for (int i = 0; i < cd.properties.size(); i++) {
			if (cd.properties[i].overridden) {
				continue;
			}
			// Ignore undocumented private.
			if (cd.properties[i].name.begins_with("_") && cd.properties[i].description.strip_edges().is_empty()) {
				continue;
			}

			property_line[cd.properties[i].name] = class_desc->get_paragraph_count() - 2;

			class_desc->push_table(2);
			class_desc->set_table_column_expand(1, true);

			class_desc->push_cell();
			_push_code_font();
			_add_bulletpoint();

			_add_type(cd.properties[i].type, cd.properties[i].enumeration, cd.properties[i].is_bitfield);
			class_desc->add_text(" ");
			_pop_code_font();
			class_desc->pop(); // cell

			class_desc->push_cell();
			_push_code_font();
			class_desc->push_color(theme_cache.headline_color);
			_add_text(cd.properties[i].name);
			class_desc->pop(); // color

			if (!cd.properties[i].default_value.is_empty()) {
				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text(" [" + TTR("default:") + " ");
				class_desc->pop(); // color

				class_desc->push_color(theme_cache.value_color);
				class_desc->add_text(_fix_constant(cd.properties[i].default_value));
				class_desc->pop(); // color

				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text("]");
				class_desc->pop(); // color
			}

			if (cd.properties[i].is_deprecated) {
				DEPRECATED_DOC_TAG;
			}

			if (cd.properties[i].is_experimental) {
				EXPERIMENTAL_DOC_TAG;
			}

			if (cd.is_script_doc && (!cd.properties[i].setter.is_empty() || !cd.properties[i].getter.is_empty())) {
				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text(" [" + TTR("property:") + " ");
				class_desc->pop(); // color

				if (!cd.properties[i].setter.is_empty()) {
					class_desc->push_color(theme_cache.value_color);
					class_desc->add_text("setter");
					class_desc->pop(); // color
				}
				if (!cd.properties[i].getter.is_empty()) {
					if (!cd.properties[i].setter.is_empty()) {
						class_desc->push_color(theme_cache.symbol_color);
						class_desc->add_text(", ");
						class_desc->pop(); // color
					}
					class_desc->push_color(theme_cache.value_color);
					class_desc->add_text("getter");
					class_desc->pop(); // color
				}

				class_desc->push_color(theme_cache.symbol_color);
				class_desc->add_text("]");
				class_desc->pop(); // color
			}

			_pop_code_font();
			class_desc->pop(); // cell

			// Script doc doesn't have setter, getter.
			if (!cd.is_script_doc) {
				HashMap<String, DocData::MethodDoc> method_map;
				for (int j = 0; j < methods.size(); j++) {
					method_map[methods[j].name] = methods[j];
				}

				if (!cd.properties[i].setter.is_empty()) {
					class_desc->push_cell();
					class_desc->pop(); // cell

					class_desc->push_cell();
					_push_code_font();
					class_desc->push_color(theme_cache.text_color);

					if (method_map[cd.properties[i].setter].arguments.size() > 1) {
						// Setters with additional arguments are exposed in the method list, so we link them here for quick access.
						class_desc->push_meta("@method " + cd.properties[i].setter);
						class_desc->add_text(cd.properties[i].setter + TTR("(value)"));
						class_desc->pop();
					} else {
						class_desc->add_text(cd.properties[i].setter + TTR("(value)"));
					}

					class_desc->pop(); // color
					class_desc->push_color(theme_cache.comment_color);
					class_desc->add_text(" setter");
					class_desc->pop(); // color
					_pop_code_font();
					class_desc->pop(); // cell

					method_line[cd.properties[i].setter] = property_line[cd.properties[i].name];
				}

				if (!cd.properties[i].getter.is_empty()) {
					class_desc->push_cell();
					class_desc->pop(); // cell

					class_desc->push_cell();
					_push_code_font();
					class_desc->push_color(theme_cache.text_color);

					if (method_map[cd.properties[i].getter].arguments.size() > 0) {
						// Getters with additional arguments are exposed in the method list, so we link them here for quick access.
						class_desc->push_meta("@method " + cd.properties[i].getter);
						class_desc->add_text(cd.properties[i].getter + "()");
						class_desc->pop();
					} else {
						class_desc->add_text(cd.properties[i].getter + "()");
					}

					class_desc->pop(); // color
					class_desc->push_color(theme_cache.comment_color);
					class_desc->add_text(" getter");
					class_desc->pop(); // color
					_pop_code_font();
					class_desc->pop(); // cell

					method_line[cd.properties[i].getter] = property_line[cd.properties[i].name];
				}
			}

			class_desc->pop(); // table

			class_desc->add_newline();
			class_desc->add_newline();

			class_desc->push_color(theme_cache.text_color);
			_push_normal_font();
			class_desc->push_indent(1);
			if (!cd.properties[i].description.strip_edges().is_empty()) {
				_add_text(DTR(cd.properties[i].description));
			} else {
				class_desc->add_image(get_editor_theme_icon(SNAME("Error")));
				class_desc->add_text(" ");
				class_desc->push_color(theme_cache.comment_color);
				if (cd.is_script_doc) {
					class_desc->append_text(TTR("There is currently no description for this property."));
				} else {
					class_desc->append_text(TTR("There is currently no description for this property. Please help us by [color=$color][url=$url]contributing one[/url][/color]!").replace("$url", CONTRIBUTE_URL).replace("$color", link_color_text));
				}
				class_desc->pop();
			}
			class_desc->pop();
			_pop_normal_font();
			class_desc->pop();

			class_desc->add_newline();
			class_desc->add_newline();
			class_desc->add_newline();
		}
	}

	// Constructor descriptions
	if (!cd.constructors.is_empty()) {
		section_line.push_back(Pair<String, int>(TTR("Constructor Descriptions"), class_desc->get_paragraph_count() - 2));
		_push_title_font();
		class_desc->add_text(TTR("Constructor Descriptions"));
		_pop_title_font();

		_update_method_descriptions(cd, cd.constructors, METHOD_TYPE_CONSTRUCTOR);
	}

	// Method descriptions
	if (!methods.is_empty()) {
		section_line.push_back(Pair<String, int>(TTR("Method Descriptions"), class_desc->get_paragraph_count() - 2));
		_push_title_font();
		class_desc->add_text(TTR("Method Descriptions"));
		_pop_title_font();

		_update_method_descriptions(cd, methods, METHOD_TYPE_METHOD);
	}

	// Operator descriptions
	if (!cd.operators.is_empty()) {
		section_line.push_back(Pair<String, int>(TTR("Operator Descriptions"), class_desc->get_paragraph_count() - 2));
		_push_title_font();
		class_desc->add_text(TTR("Operator Descriptions"));
		_pop_title_font();

		_update_method_descriptions(cd, cd.operators, METHOD_TYPE_OPERATOR);
	}

	// Free the scroll.
	scroll_locked = false;
}

void EditorHelp::_request_help(const String &p_string) {
	Error err = _goto_desc(p_string);
	if (err == OK) {
		EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
	}
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
	} else if (what == "class_annotation") {
		if (annotation_line.has(name)) {
			line = annotation_line[name];
		}
	} else if (what == "class_global") {
		if (constant_line.has(name)) {
			line = constant_line[name];
		} else if (method_line.has(name)) {
			line = method_line[name];
		} else {
			HashMap<String, HashMap<String, int>>::Iterator iter = enum_values_line.begin();
			while (true) {
				if (iter->value.has(name)) {
					line = iter->value[name];
					break;
				} else if (iter == enum_values_line.last()) {
					break;
				} else {
					++iter;
				}
			}
		}
	}

	if (class_desc->is_ready()) {
		class_desc->call_deferred(SNAME("scroll_to_paragraph"), line);
	} else {
		scroll_to = line;
	}
}

static void _add_text_to_rt(const String &p_bbcode, RichTextLabel *p_rt, Control *p_owner_node, const String &p_class = "") {
	DocTools *doc = EditorHelp::get_doc_data();
	String base_path;

	Ref<Font> doc_font = p_owner_node->get_theme_font(SNAME("doc"), EditorStringName(EditorFonts));
	Ref<Font> doc_bold_font = p_owner_node->get_theme_font(SNAME("doc_bold"), EditorStringName(EditorFonts));
	Ref<Font> doc_italic_font = p_owner_node->get_theme_font(SNAME("doc_italic"), EditorStringName(EditorFonts));
	Ref<Font> doc_code_font = p_owner_node->get_theme_font(SNAME("doc_source"), EditorStringName(EditorFonts));
	Ref<Font> doc_kbd_font = p_owner_node->get_theme_font(SNAME("doc_keyboard"), EditorStringName(EditorFonts));

	int doc_code_font_size = p_owner_node->get_theme_font_size(SNAME("doc_source_size"), EditorStringName(EditorFonts));
	int doc_kbd_font_size = p_owner_node->get_theme_font_size(SNAME("doc_keyboard_size"), EditorStringName(EditorFonts));

	const Color type_color = p_owner_node->get_theme_color(SNAME("type_color"), SNAME("EditorHelp"));
	const Color code_color = p_owner_node->get_theme_color(SNAME("code_color"), SNAME("EditorHelp"));
	const Color kbd_color = p_owner_node->get_theme_color(SNAME("kbd_color"), SNAME("EditorHelp"));
	const Color code_dark_color = Color(code_color, 0.8);

	const Color link_color = p_owner_node->get_theme_color(SNAME("link_color"), SNAME("EditorHelp"));
	const Color link_method_color = p_owner_node->get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
	const Color link_property_color = link_color.lerp(p_owner_node->get_theme_color(SNAME("accent_color"), EditorStringName(Editor)), 0.25);
	const Color link_annotation_color = link_color.lerp(p_owner_node->get_theme_color(SNAME("accent_color"), EditorStringName(Editor)), 0.5);

	const Color code_bg_color = p_owner_node->get_theme_color(SNAME("code_bg_color"), SNAME("EditorHelp"));
	const Color kbd_bg_color = p_owner_node->get_theme_color(SNAME("kbd_bg_color"), SNAME("EditorHelp"));
	const Color param_bg_color = p_owner_node->get_theme_color(SNAME("param_bg_color"), SNAME("EditorHelp"));

	String bbcode = p_bbcode.dedent().replace("\t", "").replace("\r", "").strip_edges();

	// Select the correct code examples.
	switch ((int)EDITOR_GET("text_editor/help/class_reference_examples")) {
		case 0: // GDScript
			bbcode = bbcode.replace("[gdscript", "[codeblock"); // Tag can have extra arguments.
			bbcode = bbcode.replace("[/gdscript]", "[/codeblock]");

			for (int pos = bbcode.find("[csharp"); pos != -1; pos = bbcode.find("[csharp")) {
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
			bbcode = bbcode.replace("[csharp", "[codeblock"); // Tag can have extra arguments.
			bbcode = bbcode.replace("[/csharp]", "[/codeblock]");

			for (int pos = bbcode.find("[gdscript"); pos != -1; pos = bbcode.find("[gdscript")) {
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
			bbcode = bbcode.replace("[csharp", "[b]C#:[/b]\n[codeblock"); // Tag can have extra arguments.
			bbcode = bbcode.replace("[gdscript", "[b]GDScript:[/b]\n[codeblock"); // Tag can have extra arguments.

			bbcode = bbcode.replace("[/csharp]", "[/codeblock]");
			bbcode = bbcode.replace("[/gdscript]", "[/codeblock]");
			break;
	}

	// Remove codeblocks (they would be printed otherwise).
	bbcode = bbcode.replace("[codeblocks]\n", "");
	bbcode = bbcode.replace("\n[/codeblocks]", "");
	bbcode = bbcode.replace("[codeblocks]", "");
	bbcode = bbcode.replace("[/codeblocks]", "");

	// Remove extra new lines around code blocks.
	bbcode = bbcode.replace("[codeblock]\n", "[codeblock]");
	bbcode = bbcode.replace("[codeblock skip-lint]\n", "[codeblock skip-lint]"); // Extra argument to silence validation warnings.
	bbcode = bbcode.replace("\n[/codeblock]", "[/codeblock]");

	List<String> tag_stack;
	bool code_tag = false;
	bool codeblock_tag = false;

	int pos = 0;
	while (pos < bbcode.length()) {
		int brk_pos = bbcode.find("[", pos);

		if (brk_pos < 0) {
			brk_pos = bbcode.length();
		}

		if (brk_pos > pos) {
			String text = bbcode.substr(pos, brk_pos - pos);
			if (!code_tag && !codeblock_tag) {
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
			if (!code_tag && !codeblock_tag) {
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
					p_rt->pop(); // font size
					// Pop both color and background color.
					p_rt->pop();
					p_rt->pop();
				} else if (codeblock_tag) {
					p_rt->pop(); // font size
					// Pop color, cell and table.
					p_rt->pop();
					p_rt->pop();
					p_rt->pop();
				}
			}
			code_tag = false;
			codeblock_tag = false;

		} else if (code_tag || codeblock_tag) {
			p_rt->add_text("[");
			pos = brk_pos + 1;

		} else if (tag.begins_with("method ") || tag.begins_with("constructor ") || tag.begins_with("operator ") || tag.begins_with("member ") || tag.begins_with("signal ") || tag.begins_with("enum ") || tag.begins_with("constant ") || tag.begins_with("annotation ") || tag.begins_with("theme_item ")) {
			const int tag_end = tag.find(" ");
			const String link_tag = tag.substr(0, tag_end);
			const String link_target = tag.substr(tag_end + 1, tag.length()).lstrip(" ");

			// Use monospace font to make clickable references
			// easier to distinguish from inline code and other text.
			p_rt->push_font(doc_code_font);
			p_rt->push_font_size(doc_code_font_size);

			Color target_color = link_color;
			if (link_tag == "method" || link_tag == "constructor" || link_tag == "operator") {
				target_color = link_method_color;
			} else if (link_tag == "member" || link_tag == "signal" || link_tag == "theme property") {
				target_color = link_property_color;
			} else if (link_tag == "annotation") {
				target_color = link_annotation_color;
			}
			p_rt->push_color(target_color);
			p_rt->push_meta("@" + link_tag + " " + link_target);
			p_rt->add_text(link_target + (link_tag == "method" ? "()" : ""));
			p_rt->pop();
			p_rt->pop();

			p_rt->pop(); // font size
			p_rt->pop(); // font
			pos = brk_end + 1;

		} else if (tag.begins_with("param ")) {
			const int tag_end = tag.find(" ");
			const String param_name = tag.substr(tag_end + 1, tag.length()).lstrip(" ");

			// Use monospace font with translucent background color to make code easier to distinguish from other text.
			p_rt->push_font(doc_code_font);
			p_rt->push_font_size(doc_code_font_size);

			p_rt->push_bgcolor(param_bg_color);
			p_rt->push_color(code_color);
			p_rt->add_text(param_name);
			p_rt->pop();
			p_rt->pop();

			p_rt->pop(); // font size
			p_rt->pop(); // font
			pos = brk_end + 1;

		} else if (tag == p_class) {
			// Use a bold font when class reference tags are in their own page.
			p_rt->push_font(doc_bold_font);
			p_rt->add_text(tag);
			p_rt->pop();

			pos = brk_end + 1;

		} else if (doc->class_list.has(tag)) {
			// Use a monospace font for class reference tags such as [Node2D] or [SceneTree].

			p_rt->push_font(doc_code_font);
			p_rt->push_font_size(doc_code_font_size);
			p_rt->push_color(type_color);
			p_rt->push_meta("#" + tag);
			p_rt->add_text(tag);

			p_rt->pop();
			p_rt->pop();
			p_rt->pop(); // Font size
			p_rt->pop(); // Font

			pos = brk_end + 1;

		} else if (tag == "b") {
			// Use bold font.
			p_rt->push_font(doc_bold_font);

			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "i") {
			// Use italics font.
			p_rt->push_font(doc_italic_font);

			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "code" || tag.begins_with("code ")) {
			// Use monospace font with darkened background color to make code easier to distinguish from other text.
			p_rt->push_font(doc_code_font);
			p_rt->push_font_size(doc_code_font_size);
			p_rt->push_bgcolor(code_bg_color);
			p_rt->push_color(code_color.lerp(p_owner_node->get_theme_color(SNAME("error_color"), EditorStringName(Editor)), 0.6));

			code_tag = true;
			pos = brk_end + 1;
			tag_stack.push_front("code");
		} else if (tag == "codeblock" || tag.begins_with("codeblock ")) {
			// Use monospace font with darkened background color to make code easier to distinguish from other text.
			// Use a single-column table with cell row background color instead of `[bgcolor]`.
			// This makes the background color highlight cover the entire block, rather than individual lines.
			p_rt->push_font(doc_code_font);
			p_rt->push_font_size(doc_code_font_size);

			p_rt->push_table(1);
			p_rt->push_cell();
			p_rt->set_cell_row_background_color(code_bg_color, Color(code_bg_color, 0.99));
			p_rt->set_cell_padding(Rect2(10 * EDSCALE, 10 * EDSCALE, 10 * EDSCALE, 10 * EDSCALE));
			p_rt->push_color(code_dark_color);

			codeblock_tag = true;
			pos = brk_end + 1;
			tag_stack.push_front("codeblock");
		} else if (tag == "kbd") {
			// Use keyboard font with custom color and background color.
			p_rt->push_font(doc_kbd_font);
			p_rt->push_font_size(doc_kbd_font_size);
			p_rt->push_bgcolor(kbd_bg_color);
			p_rt->push_color(kbd_color);

			code_tag = true; // Though not strictly a code tag, logic is similar.
			pos = brk_end + 1;
			tag_stack.push_front(tag);

		} else if (tag == "center") {
			// Align to center.
			p_rt->push_paragraph(HORIZONTAL_ALIGNMENT_CENTER, Control::TEXT_DIRECTION_AUTO, "");
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
		} else if (tag == "lb") {
			p_rt->add_text("[");
			pos = brk_end + 1;
		} else if (tag == "rb") {
			p_rt->add_text("]");
			pos = brk_end + 1;
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
		} else if (tag.begins_with("img")) {
			int width = 0;
			int height = 0;
			bool size_in_percent = false;
			if (tag.length() > 4) {
				Vector<String> subtags = tag.substr(4, tag.length()).split(" ");
				HashMap<String, String> bbcode_options;
				for (int i = 0; i < subtags.size(); i++) {
					const String &expr = subtags[i];
					int value_pos = expr.find("=");
					if (value_pos > -1) {
						bbcode_options[expr.substr(0, value_pos)] = expr.substr(value_pos + 1).unquote();
					}
				}
				HashMap<String, String>::Iterator width_option = bbcode_options.find("width");
				if (width_option) {
					width = width_option->value.to_int();
					if (width_option->value.ends_with("%")) {
						size_in_percent = true;
					}
				}

				HashMap<String, String>::Iterator height_option = bbcode_options.find("height");
				if (height_option) {
					height = height_option->value.to_int();
					if (height_option->value.ends_with("%")) {
						size_in_percent = true;
					}
				}
			}
			int end = bbcode.find("[", brk_end);
			if (end == -1) {
				end = bbcode.length();
			}
			String image = bbcode.substr(brk_end + 1, end - brk_end - 1);

			p_rt->add_image(ResourceLoader::load(base_path.path_join(image), "Texture2D"), width, height, Color(1, 1, 1), INLINE_ALIGNMENT_CENTER, Rect2(), Variant(), false, String(), size_in_percent);

			pos = end;
			tag_stack.push_front("img");
		} else if (tag.begins_with("color=")) {
			String col = tag.substr(6, tag.length());
			Color color = Color::from_string(col, Color());
			p_rt->push_color(color);

			pos = brk_end + 1;
			tag_stack.push_front("color");

		} else if (tag.begins_with("font=")) {
			String fnt = tag.substr(5, tag.length());

			Ref<Font> font = ResourceLoader::load(base_path.path_join(fnt), "Font");
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
	_add_text_to_rt(p_bbcode, class_desc, this, edited_class);
}

String EditorHelp::doc_version_hash;
Thread EditorHelp::worker_thread;

void EditorHelp::_wait_for_thread() {
	if (worker_thread.is_started()) {
		worker_thread.wait_to_finish();
	}
}

void EditorHelp::_compute_doc_version_hash() {
	uint32_t version_hash = Engine::get_singleton()->get_version_info().hash();
	doc_version_hash = vformat("%d/%d/%d/%s", version_hash, ClassDB::get_api_hash(ClassDB::API_CORE), ClassDB::get_api_hash(ClassDB::API_EDITOR), _doc_data_hash);
}

String EditorHelp::get_cache_full_path() {
	return EditorPaths::get_singleton()->get_cache_dir().path_join("editor_doc_cache.res");
}

void EditorHelp::_load_doc_thread(void *p_udata) {
	Ref<Resource> cache_res = ResourceLoader::load(get_cache_full_path());
	if (cache_res.is_valid() && cache_res->get_meta("version_hash", "") == doc_version_hash) {
		Array classes = cache_res->get_meta("classes", Array());
		for (int i = 0; i < classes.size(); i++) {
			doc->add_doc(DocData::ClassDoc::from_dict(classes[i]));
		}

		// Extensions' docs are not cached. Generate them now (on the main thread).
		callable_mp_static(&EditorHelp::_gen_extensions_docs).call_deferred();
	} else {
		// We have to go back to the main thread to start from scratch, bypassing any possibly existing cache.
		callable_mp_static(&EditorHelp::generate_doc).bind(false).call_deferred();
	}
}

void EditorHelp::_gen_doc_thread(void *p_udata) {
	DocTools compdoc;
	compdoc.load_compressed(_doc_data_compressed, _doc_data_compressed_size, _doc_data_uncompressed_size);
	doc->merge_from(compdoc); // Ensure all is up to date.

	Ref<Resource> cache_res;
	cache_res.instantiate();
	cache_res->set_meta("version_hash", doc_version_hash);
	Array classes;
	for (const KeyValue<String, DocData::ClassDoc> &E : doc->class_list) {
		if (ClassDB::class_exists(E.value.name)) {
			ClassDB::APIType api = ClassDB::get_api_type(E.value.name);
			if (api == ClassDB::API_EXTENSION || api == ClassDB::API_EDITOR_EXTENSION) {
				continue;
			}
		}
		classes.push_back(DocData::ClassDoc::to_dict(E.value));
	}
	cache_res->set_meta("classes", classes);
	Error err = ResourceSaver::save(cache_res, get_cache_full_path(), ResourceSaver::FLAG_COMPRESS);
	if (err) {
		ERR_PRINT("Cannot save editor help cache (" + get_cache_full_path() + ").");
	}
}

void EditorHelp::_gen_extensions_docs() {
	doc->generate((DocTools::GENERATE_FLAG_SKIP_BASIC_TYPES | DocTools::GENERATE_FLAG_EXTENSION_CLASSES_ONLY));
}

void EditorHelp::generate_doc(bool p_use_cache) {
	OS::get_singleton()->benchmark_begin_measure("EditorHelp::generate_doc");

	// In case not the first attempt.
	_wait_for_thread();

	if (!doc) {
		doc = memnew(DocTools);
	}

	if (doc_version_hash.is_empty()) {
		_compute_doc_version_hash();
	}

	if (p_use_cache && FileAccess::exists(get_cache_full_path())) {
		worker_thread.start(_load_doc_thread, nullptr);
	} else {
		print_verbose("Regenerating editor help cache");
		doc->generate();
		worker_thread.start(_gen_doc_thread, nullptr);
	}

	OS::get_singleton()->benchmark_end_measure("EditorHelp::generate_doc");
}

void EditorHelp::_toggle_scripts_pressed() {
	ScriptEditor::get_singleton()->toggle_scripts_panel();
	update_toggle_scripts_button();
}

void EditorHelp::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			// Requires theme to be up to date.
			_class_desc_resized(false);
		} break;

		case NOTIFICATION_READY:
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			_wait_for_thread();
			_update_doc();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			if (is_inside_tree()) {
				_class_desc_resized(true);
			}
			update_toggle_scripts_button();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			update_toggle_scripts_button();
		} break;
	}
}

void EditorHelp::go_to_help(const String &p_help) {
	_wait_for_thread();
	_help_callback(p_help);
}

void EditorHelp::go_to_class(const String &p_class) {
	_wait_for_thread();
	_goto_desc(p_class);
}

void EditorHelp::update_doc() {
	_wait_for_thread();
	ERR_FAIL_COND(!doc->class_list.has(edited_class));
	ERR_FAIL_COND(!doc->class_list[edited_class].is_script_doc);
	_update_doc();
}

void EditorHelp::cleanup_doc() {
	_wait_for_thread();
	memdelete(doc);
}

Vector<Pair<String, int>> EditorHelp::get_sections() {
	_wait_for_thread();
	Vector<Pair<String, int>> sections;

	for (int i = 0; i < section_line.size(); i++) {
		sections.push_back(Pair<String, int>(section_line[i].first, i));
	}
	return sections;
}

void EditorHelp::scroll_to_section(int p_section_index) {
	_wait_for_thread();
	int line = section_line[p_section_index].second;
	if (class_desc->is_ready()) {
		class_desc->scroll_to_paragraph(line);
	} else {
		scroll_to = line;
	}
}

void EditorHelp::popup_search() {
	_wait_for_thread();
	find_bar->popup_search();
}

String EditorHelp::get_class() {
	return edited_class;
}

void EditorHelp::search_again(bool p_search_previous) {
	_search(p_search_previous);
}

int EditorHelp::get_scroll() const {
	return class_desc->get_v_scroll_bar()->get_value();
}

void EditorHelp::set_scroll(int p_scroll) {
	class_desc->get_v_scroll_bar()->set_value(p_scroll);
}

void EditorHelp::update_toggle_scripts_button() {
	if (is_layout_rtl()) {
		toggle_scripts_button->set_icon(get_editor_theme_icon(ScriptEditor::get_singleton()->is_scripts_panel_toggled() ? SNAME("Forward") : SNAME("Back")));
	} else {
		toggle_scripts_button->set_icon(get_editor_theme_icon(ScriptEditor::get_singleton()->is_scripts_panel_toggled() ? SNAME("Back") : SNAME("Forward")));
	}
	toggle_scripts_button->set_tooltip_text(vformat("%s (%s)", TTR("Toggle Scripts Panel"), ED_GET_SHORTCUT("script_editor/toggle_scripts_panel")->get_as_text()));
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
	class_desc->set_threaded(true);
	class_desc->set_v_size_flags(SIZE_EXPAND_FILL);

	class_desc->connect("finished", callable_mp(this, &EditorHelp::_class_desc_finished));
	class_desc->connect("meta_clicked", callable_mp(this, &EditorHelp::_class_desc_select));
	class_desc->connect("gui_input", callable_mp(this, &EditorHelp::_class_desc_input));
	class_desc->connect("resized", callable_mp(this, &EditorHelp::_class_desc_resized).bind(false));

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
	class_desc->set_context_menu_enabled(true);

	class_desc->hide();
}

EditorHelp::~EditorHelp() {
}

DocTools *EditorHelp::get_doc_data() {
	_wait_for_thread();
	return doc;
}

/// EditorHelpBit ///

void EditorHelpBit::_go_to_help(String p_what) {
	EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
	ScriptEditor::get_singleton()->goto_help(p_what);
	emit_signal(SNAME("request_hide"));
}

void EditorHelpBit::_meta_clicked(String p_select) {
	if (p_select.begins_with("$")) { // enum
		String select = p_select.substr(1, p_select.length());
		String class_name;
		int rfind = select.rfind(".");
		if (rfind != -1) {
			class_name = select.substr(0, rfind);
			select = select.substr(rfind + 1);
		} else {
			class_name = "@GlobalScope";
		}
		_go_to_help("class_enum:" + class_name + ":" + select);
		return;
	} else if (p_select.begins_with("#")) {
		_go_to_help("class_name:" + p_select.substr(1, p_select.length()));
		return;
	} else if (p_select.begins_with("@")) {
		String m = p_select.substr(1, p_select.length());

		if (m.contains(".")) {
			_go_to_help("class_method:" + m.get_slice(".", 0) + ":" + m.get_slice(".", 0)); // Must go somewhere else.
		}
	}
}

String EditorHelpBit::get_class_description(const StringName &p_class_name) const {
	if (doc_class_cache.has(p_class_name)) {
		return doc_class_cache[p_class_name];
	}

	String description;
	HashMap<String, DocData::ClassDoc>::ConstIterator E = EditorHelp::get_doc_data()->class_list.find(p_class_name);
	if (E) {
		// Non-native class shouldn't be cached, nor translated.
		bool is_native = ClassDB::class_exists(p_class_name);
		description = is_native ? DTR(E->value.brief_description) : E->value.brief_description;

		if (is_native) {
			doc_class_cache[p_class_name] = description;
		}
	}

	return description;
}

String EditorHelpBit::get_property_description(const StringName &p_class_name, const StringName &p_property_name) const {
	if (doc_property_cache.has(p_class_name) && doc_property_cache[p_class_name].has(p_property_name)) {
		return doc_property_cache[p_class_name][p_property_name];
	}

	String description;
	// Non-native properties shouldn't be cached, nor translated.
	bool is_native = ClassDB::class_exists(p_class_name);
	DocTools *dd = EditorHelp::get_doc_data();
	HashMap<String, DocData::ClassDoc>::ConstIterator E = dd->class_list.find(p_class_name);
	if (E) {
		for (int i = 0; i < E->value.properties.size(); i++) {
			String description_current = is_native ? DTR(E->value.properties[i].description) : E->value.properties[i].description;

			const Vector<String> class_enum = E->value.properties[i].enumeration.split(".");
			const String enum_name = class_enum.size() >= 2 ? class_enum[1] : "";
			if (!enum_name.is_empty()) {
				// Classes can use enums from other classes, so check from which it came.
				HashMap<String, DocData::ClassDoc>::ConstIterator enum_class = dd->class_list.find(class_enum[0]);
				if (enum_class) {
					for (DocData::ConstantDoc val : enum_class->value.constants) {
						// Don't display `_MAX` enum value descriptions, as these are never exposed in the inspector.
						if (val.enumeration == enum_name && !val.name.ends_with("_MAX")) {
							const String enum_value = EditorPropertyNameProcessor::get_singleton()->process_name(val.name, EditorPropertyNameProcessor::STYLE_CAPITALIZED);
							const String enum_prefix = EditorPropertyNameProcessor::get_singleton()->process_name(enum_name, EditorPropertyNameProcessor::STYLE_CAPITALIZED) + " ";
							const String enum_description = is_native ? DTR(val.description) : val.description;

							// Prettify the enum value display, so that "<ENUM NAME>_<VALUE>" becomes "Value".
							description_current = description_current.trim_prefix("\n") + vformat("\n[b]%s:[/b] %s", enum_value.trim_prefix(enum_prefix), enum_description.is_empty() ? ("[i]" + DTR("No description available.") + "[/i]") : enum_description);
						}
					}
				}
			}

			if (E->value.properties[i].name == p_property_name) {
				description = description_current;

				if (!is_native) {
					break;
				}
			}

			if (is_native) {
				doc_property_cache[p_class_name][E->value.properties[i].name] = description_current;
			}
		}
	}

	return description;
}

String EditorHelpBit::get_method_description(const StringName &p_class_name, const StringName &p_method_name) const {
	if (doc_method_cache.has(p_class_name) && doc_method_cache[p_class_name].has(p_method_name)) {
		return doc_method_cache[p_class_name][p_method_name];
	}

	String description;
	HashMap<String, DocData::ClassDoc>::ConstIterator E = EditorHelp::get_doc_data()->class_list.find(p_class_name);
	if (E) {
		// Non-native methods shouldn't be cached, nor translated.
		bool is_native = ClassDB::class_exists(p_class_name);

		for (int i = 0; i < E->value.methods.size(); i++) {
			String description_current = is_native ? DTR(E->value.methods[i].description) : E->value.methods[i].description;

			if (E->value.methods[i].name == p_method_name) {
				description = description_current;

				if (!is_native) {
					break;
				}
			}

			if (is_native) {
				doc_method_cache[p_class_name][E->value.methods[i].name] = description_current;
			}
		}
	}

	return description;
}

String EditorHelpBit::get_signal_description(const StringName &p_class_name, const StringName &p_signal_name) const {
	if (doc_signal_cache.has(p_class_name) && doc_signal_cache[p_class_name].has(p_signal_name)) {
		return doc_signal_cache[p_class_name][p_signal_name];
	}

	String description;
	HashMap<String, DocData::ClassDoc>::ConstIterator E = EditorHelp::get_doc_data()->class_list.find(p_class_name);
	if (E) {
		// Non-native signals shouldn't be cached, nor translated.
		bool is_native = ClassDB::class_exists(p_class_name);

		for (int i = 0; i < E->value.signals.size(); i++) {
			String description_current = is_native ? DTR(E->value.signals[i].description) : E->value.signals[i].description;

			if (E->value.signals[i].name == p_signal_name) {
				description = description_current;

				if (!is_native) {
					break;
				}
			}

			if (is_native) {
				doc_signal_cache[p_class_name][E->value.signals[i].name] = description_current;
			}
		}
	}

	return description;
}

String EditorHelpBit::get_theme_item_description(const StringName &p_class_name, const StringName &p_theme_item_name) const {
	if (doc_theme_item_cache.has(p_class_name) && doc_theme_item_cache[p_class_name].has(p_theme_item_name)) {
		return doc_theme_item_cache[p_class_name][p_theme_item_name];
	}

	String description;
	bool found = false;
	DocTools *dd = EditorHelp::get_doc_data();
	HashMap<String, DocData::ClassDoc>::ConstIterator E = dd->class_list.find(p_class_name);
	while (E) {
		// Non-native theme items shouldn't be cached, nor translated.
		bool is_native = ClassDB::class_exists(p_class_name);

		for (int i = 0; i < E->value.theme_properties.size(); i++) {
			String description_current = is_native ? DTR(E->value.theme_properties[i].description) : E->value.theme_properties[i].description;

			if (E->value.theme_properties[i].name == p_theme_item_name) {
				description = description_current;
				found = true;

				if (!is_native) {
					break;
				}
			}

			if (is_native) {
				doc_theme_item_cache[p_class_name][E->value.theme_properties[i].name] = description_current;
			}
		}

		if (found || E->value.inherits.is_empty()) {
			break;
		}
		// Check for inherited theme items.
		E = dd->class_list.find(E->value.inherits);
	}

	return description;
}

void EditorHelpBit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_text", "text"), &EditorHelpBit::set_text);
	ADD_SIGNAL(MethodInfo("request_hide"));
}

void EditorHelpBit::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			rich_text->add_theme_color_override("selection_color", get_theme_color(SNAME("selection_color"), SNAME("EditorHelp")));
			rich_text->clear();
			_add_text_to_rt(text, rich_text, this);
			rich_text->reset_size(); // Force recalculating size after parsing bbcode.
		} break;
	}
}

void EditorHelpBit::set_text(const String &p_text) {
	text = p_text;
	rich_text->clear();
	_add_text_to_rt(text, rich_text, this);
}

EditorHelpBit::EditorHelpBit() {
	rich_text = memnew(RichTextLabel);
	add_child(rich_text);
	rich_text->connect("meta_clicked", callable_mp(this, &EditorHelpBit::_meta_clicked));
	rich_text->set_fit_content(true);
	set_custom_minimum_size(Size2(0, 50 * EDSCALE));
}

/// EditorHelpTooltip ///

void EditorHelpTooltip::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			if (!tooltip_text.is_empty()) {
				parse_tooltip(tooltip_text);
			}
		} break;
	}
}

// `p_text` is expected to be something like these:
// - `class|Control||`;
// - `property|Control|size|`;
// - `signal|Control|gui_input|(event: InputEvent)`.
void EditorHelpTooltip::parse_tooltip(const String &p_text) {
	tooltip_text = p_text;

	PackedStringArray slices = p_text.split("|", true, 3);
	ERR_FAIL_COND_MSG(slices.size() < 4, "Invalid tooltip formatting. The expect string should be formatted as 'type|class|property|args'.");

	String type = slices[0];
	String class_name = slices[1];
	String property_name = slices[2];
	String property_args = slices[3];

	String title;
	String description;
	String formatted_text;

	if (type == "class") {
		title = class_name;
		description = get_class_description(class_name);
		formatted_text = TTR("Class:");
	} else {
		title = property_name;

		if (type == "property") {
			description = get_property_description(class_name, property_name);
			if (property_name.begins_with("metadata/")) {
				formatted_text = TTR("Metadata:");
			} else {
				formatted_text = TTR("Property:");
			}
		} else if (type == "method") {
			description = get_method_description(class_name, property_name);
			formatted_text = TTR("Method:");
		} else if (type == "signal") {
			description = get_signal_description(class_name, property_name);
			formatted_text = TTR("Signal:");
		} else if (type == "theme_item") {
			description = get_theme_item_description(class_name, property_name);
			formatted_text = TTR("Theme Item:");
		} else {
			ERR_FAIL_MSG("Invalid tooltip type '" + type + "'. Valid types are 'class', 'property', 'method', 'signal', and 'theme_item'.");
		}
	}

	// Metadata special handling replaces "Property:" with "Metadata": above.
	formatted_text += " [u][b]" + title.trim_prefix("metadata/") + "[/b][/u]" + property_args.replace("[", "[lb]") + "\n";
	formatted_text += description.is_empty() ? "[i]" + TTR("No description available.") + "[/i]" : description;
	set_text(formatted_text);
}

EditorHelpTooltip::EditorHelpTooltip(const String &p_text) {
	tooltip_text = p_text;

	get_rich_text()->set_custom_minimum_size(Size2(360 * EDSCALE, 0));
}

/// FindBar ///

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
	hide_button->set_ignore_texture_size(true);
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
		case NOTIFICATION_THEME_CHANGED: {
			find_prev->set_icon(get_editor_theme_icon(SNAME("MoveUp")));
			find_next->set_icon(get_editor_theme_icon(SNAME("MoveDown")));
			hide_button->set_texture_normal(get_editor_theme_icon(SNAME("Close")));
			hide_button->set_texture_hover(get_editor_theme_icon(SNAME("Close")));
			hide_button->set_texture_pressed(get_editor_theme_icon(SNAME("Close")));
			hide_button->set_custom_minimum_size(hide_button->get_texture_normal()->get_size());
			matches_label->add_theme_color_override("font_color", results_count > 0 ? get_theme_color(SNAME("font_color"), SNAME("Label")) : get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			set_process_unhandled_input(is_visible_in_tree());
		} break;
	}
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

		matches_label->add_theme_color_override("font_color", results_count > 0 ? get_theme_color(SNAME("font_color"), SNAME("Label")) : get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
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
	if (k.is_valid() && k->is_action_pressed(SNAME("ui_cancel"), false, true)) {
		if (rich_text_label->has_focus() || is_ancestor_of(get_viewport()->gui_get_focus_owner())) {
			_hide_bar();
			accept_event();
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
