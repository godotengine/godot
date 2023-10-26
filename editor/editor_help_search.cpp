/**************************************************************************/
/*  editor_help_search.cpp                                                */
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

#include "editor_help_search.h"

#include "core/os/keyboard.h"
#include "editor/editor_feature_profile.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"

void EditorHelpSearch::_update_results() {
	String term = search_box->get_text();

	int search_flags = filter_combo->get_selected_id();
	if (case_sensitive_button->is_pressed()) {
		search_flags |= SEARCH_CASE_SENSITIVE;
	}
	if (hierarchy_button->is_pressed()) {
		search_flags |= SEARCH_SHOW_HIERARCHY;
	}

	search = Ref<Runner>(memnew(Runner(results_tree, results_tree, term, search_flags)));
	set_process(true);
}

void EditorHelpSearch::_search_box_gui_input(const Ref<InputEvent> &p_event) {
	// Redirect up and down navigational key events to the results list.
	Ref<InputEventKey> key = p_event;
	if (key.is_valid()) {
		switch (key->get_keycode()) {
			case Key::UP:
			case Key::DOWN:
			case Key::PAGEUP:
			case Key::PAGEDOWN: {
				results_tree->gui_input(key);
				search_box->accept_event();
			} break;
			default:
				break;
		}
	}
}

void EditorHelpSearch::_search_box_text_changed(const String &p_text) {
	_update_results();
}

void EditorHelpSearch::_filter_combo_item_selected(int p_option) {
	_update_results();
}

void EditorHelpSearch::_confirmed() {
	TreeItem *item = results_tree->get_selected();
	if (!item) {
		return;
	}

	// Activate the script editor and emit the signal with the documentation link to display.
	EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);

	emit_signal(SNAME("go_to_help"), item->get_metadata(0));

	hide();
}

void EditorHelpSearch::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				results_tree->call_deferred(SNAME("clear")); // Wait for the Tree's mouse event propagation.
				get_ok_button()->set_disabled(true);
				EditorSettings::get_singleton()->set_project_metadata("dialog_bounds", "search_help", Rect2(get_position(), get_size()));
			}
		} break;

		case NOTIFICATION_READY: {
			connect("confirmed", callable_mp(this, &EditorHelpSearch::_confirmed));
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			const int icon_width = get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor));
			results_tree->add_theme_constant_override("icon_max_width", icon_width);

			search_box->set_right_icon(get_editor_theme_icon(SNAME("Search")));
			search_box->add_theme_icon_override("right_icon", get_editor_theme_icon(SNAME("Search")));

			case_sensitive_button->set_icon(get_editor_theme_icon(SNAME("MatchCase")));
			hierarchy_button->set_icon(get_editor_theme_icon(SNAME("ClassList")));

			if (is_visible()) {
				_update_results();
			}
		} break;

		case NOTIFICATION_PROCESS: {
			// Update background search.
			if (search.is_valid()) {
				if (search->work()) {
					// Search done.

					// Only point to the match if it's a new search, and not just reopening a old one.
					if (!old_search) {
						results_tree->ensure_cursor_is_visible();
					} else {
						old_search = false;
					}

					get_ok_button()->set_disabled(!results_tree->get_selected());

					search = Ref<Runner>();
					set_process(false);
				}
			} else {
				set_process(false);
			}
		} break;
	}
}

void EditorHelpSearch::_bind_methods() {
	ADD_SIGNAL(MethodInfo("go_to_help"));
}

void EditorHelpSearch::popup_dialog() {
	popup_dialog(search_box->get_text());
}

void EditorHelpSearch::popup_dialog(const String &p_term) {
	// Restore valid window bounds or pop up at default size.
	Rect2 saved_size = EditorSettings::get_singleton()->get_project_metadata("dialog_bounds", "search_help", Rect2());
	if (saved_size != Rect2()) {
		popup(saved_size);
	} else {
		popup_centered_ratio(0.5F);
	}

	if (p_term.is_empty()) {
		search_box->clear();
	} else {
		if (old_term == p_term) {
			old_search = true;
		} else {
			old_term = p_term;
		}

		search_box->set_text(p_term);
		search_box->select_all();
	}
	search_box->grab_focus();
	_update_results();
}

EditorHelpSearch::EditorHelpSearch() {
	set_hide_on_ok(false);
	set_clamp_to_embedder(true);

	set_title(TTR("Search Help"));

	get_ok_button()->set_disabled(true);
	set_ok_button_text(TTR("Open"));

	// Split search and results area.
	VBoxContainer *vbox = memnew(VBoxContainer);
	add_child(vbox);

	// Create the search box and filter controls (at the top).
	HBoxContainer *hbox = memnew(HBoxContainer);
	vbox->add_child(hbox);

	search_box = memnew(LineEdit);
	search_box->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_box->set_clear_button_enabled(true);
	search_box->connect("gui_input", callable_mp(this, &EditorHelpSearch::_search_box_gui_input));
	search_box->connect("text_changed", callable_mp(this, &EditorHelpSearch::_search_box_text_changed));
	register_text_enter(search_box);
	hbox->add_child(search_box);

	case_sensitive_button = memnew(Button);
	case_sensitive_button->set_theme_type_variation("FlatButton");
	case_sensitive_button->set_tooltip_text(TTR("Case Sensitive"));
	case_sensitive_button->connect("pressed", callable_mp(this, &EditorHelpSearch::_update_results));
	case_sensitive_button->set_toggle_mode(true);
	case_sensitive_button->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(case_sensitive_button);

	hierarchy_button = memnew(Button);
	hierarchy_button->set_theme_type_variation("FlatButton");
	hierarchy_button->set_tooltip_text(TTR("Show Hierarchy"));
	hierarchy_button->connect("pressed", callable_mp(this, &EditorHelpSearch::_update_results));
	hierarchy_button->set_toggle_mode(true);
	hierarchy_button->set_pressed(true);
	hierarchy_button->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(hierarchy_button);

	filter_combo = memnew(OptionButton);
	filter_combo->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	filter_combo->set_stretch_ratio(0); // Fixed width.
	filter_combo->add_item(TTR("Display All"), SEARCH_ALL);
	filter_combo->add_separator();
	filter_combo->add_item(TTR("Classes Only"), SEARCH_CLASSES);
	filter_combo->add_item(TTR("Constructors Only"), SEARCH_CONSTRUCTORS);
	filter_combo->add_item(TTR("Methods Only"), SEARCH_METHODS);
	filter_combo->add_item(TTR("Operators Only"), SEARCH_OPERATORS);
	filter_combo->add_item(TTR("Signals Only"), SEARCH_SIGNALS);
	filter_combo->add_item(TTR("Annotations Only"), SEARCH_ANNOTATIONS);
	filter_combo->add_item(TTR("Constants Only"), SEARCH_CONSTANTS);
	filter_combo->add_item(TTR("Properties Only"), SEARCH_PROPERTIES);
	filter_combo->add_item(TTR("Theme Properties Only"), SEARCH_THEME_ITEMS);
	filter_combo->connect("item_selected", callable_mp(this, &EditorHelpSearch::_filter_combo_item_selected));
	hbox->add_child(filter_combo);

	// Create the results tree.
	results_tree = memnew(Tree);
	results_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	results_tree->set_columns(2);
	results_tree->set_column_title(0, TTR("Name"));
	results_tree->set_column_clip_content(0, true);
	results_tree->set_column_title(1, TTR("Member Type"));
	results_tree->set_column_expand(1, false);
	results_tree->set_column_custom_minimum_width(1, 150 * EDSCALE);
	results_tree->set_column_clip_content(1, true);
	results_tree->set_custom_minimum_size(Size2(0, 100) * EDSCALE);
	results_tree->set_hide_root(true);
	results_tree->set_select_mode(Tree::SELECT_ROW);
	results_tree->connect("item_activated", callable_mp(this, &EditorHelpSearch::_confirmed));
	results_tree->connect("item_selected", callable_mp((BaseButton *)get_ok_button(), &BaseButton::set_disabled).bind(false));
	vbox->add_child(results_tree, true);
}

bool EditorHelpSearch::Runner::_is_class_disabled_by_feature_profile(const StringName &p_class) {
	Ref<EditorFeatureProfile> profile = EditorFeatureProfileManager::get_singleton()->get_current_profile();
	if (profile.is_null()) {
		return false;
	}

	StringName class_name = p_class;
	while (class_name != StringName()) {
		if (!ClassDB::class_exists(class_name)) {
			return false;
		}

		if (profile->is_class_disabled(class_name)) {
			return true;
		}
		class_name = ClassDB::get_parent_class(class_name);
	}

	return false;
}

bool EditorHelpSearch::Runner::_slice() {
	bool phase_done = false;
	switch (phase) {
		case PHASE_MATCH_CLASSES_INIT:
			phase_done = _phase_match_classes_init();
			break;
		case PHASE_MATCH_CLASSES:
			phase_done = _phase_match_classes();
			break;
		case PHASE_CLASS_ITEMS_INIT:
			phase_done = _phase_class_items_init();
			break;
		case PHASE_CLASS_ITEMS:
			phase_done = _phase_class_items();
			break;
		case PHASE_MEMBER_ITEMS_INIT:
			phase_done = _phase_member_items_init();
			break;
		case PHASE_MEMBER_ITEMS:
			phase_done = _phase_member_items();
			break;
		case PHASE_SELECT_MATCH:
			phase_done = _phase_select_match();
			break;
		case PHASE_MAX:
			return true;
		default:
			WARN_PRINT("Invalid or unhandled phase in EditorHelpSearch::Runner, aborting search.");
			return true;
	};

	if (phase_done) {
		phase++;
	}
	return false;
}

bool EditorHelpSearch::Runner::_phase_match_classes_init() {
	iterator_doc = EditorHelp::get_doc_data()->class_list.begin();
	matches.clear();
	matched_item = nullptr;
	match_highest_score = 0;

	terms = term.split_spaces();
	if (terms.is_empty()) {
		terms.append(term);
	}

	return true;
}

bool EditorHelpSearch::Runner::_phase_match_classes() {
	if (!iterator_doc) {
		return true;
	}

	DocData::ClassDoc &class_doc = iterator_doc->value;
	if (class_doc.name.is_empty()) {
		++iterator_doc;
		return false;
	}

	if (!_is_class_disabled_by_feature_profile(class_doc.name)) {
		ClassMatch match;
		match.doc = &class_doc;

		// Match class name.
		if (search_flags & SEARCH_CLASSES) {
			// If the search term is empty, add any classes which are not script docs or which don't start with
			// a double-quotation. This will ensure that only C++ classes and explicitly named classes will
			// be added.
			match.name = (term.is_empty() && (!class_doc.is_script_doc || class_doc.name[0] != '\"')) || _match_string(term, class_doc.name);
		}

		// Match members only if the term is long enough, to avoid slow performance from building a large tree.
		// Make an exception for annotations, since there are not that many of them.
		if (term.length() > 1 || term == "@") {
			if (search_flags & SEARCH_CONSTRUCTORS) {
				_match_method_name_and_push_back(class_doc.constructors, &match.constructors);
			}
			if (search_flags & SEARCH_METHODS) {
				_match_method_name_and_push_back(class_doc.methods, &match.methods);
			}
			if (search_flags & SEARCH_OPERATORS) {
				_match_method_name_and_push_back(class_doc.operators, &match.operators);
			}
			if (search_flags & SEARCH_SIGNALS) {
				for (int i = 0; i < class_doc.signals.size(); i++) {
					if (_all_terms_in_name(class_doc.signals[i].name)) {
						match.signals.push_back(const_cast<DocData::MethodDoc *>(&class_doc.signals[i]));
					}
				}
			}
			if (search_flags & SEARCH_CONSTANTS) {
				for (int i = 0; i < class_doc.constants.size(); i++) {
					if (_all_terms_in_name(class_doc.constants[i].name)) {
						match.constants.push_back(const_cast<DocData::ConstantDoc *>(&class_doc.constants[i]));
					}
				}
			}
			if (search_flags & SEARCH_PROPERTIES) {
				for (int i = 0; i < class_doc.properties.size(); i++) {
					if (_all_terms_in_name(class_doc.properties[i].name)) {
						match.properties.push_back(const_cast<DocData::PropertyDoc *>(&class_doc.properties[i]));
					}
				}
			}
			if (search_flags & SEARCH_THEME_ITEMS) {
				for (int i = 0; i < class_doc.theme_properties.size(); i++) {
					if (_all_terms_in_name(class_doc.theme_properties[i].name)) {
						match.theme_properties.push_back(const_cast<DocData::ThemeItemDoc *>(&class_doc.theme_properties[i]));
					}
				}
			}
			if (search_flags & SEARCH_ANNOTATIONS) {
				for (int i = 0; i < class_doc.annotations.size(); i++) {
					if (_match_string(term, class_doc.annotations[i].name)) {
						match.annotations.push_back(const_cast<DocData::MethodDoc *>(&class_doc.annotations[i]));
					}
				}
			}
		}
		matches[class_doc.name] = match;
	}

	++iterator_doc;
	return !iterator_doc;
}

bool EditorHelpSearch::Runner::_phase_class_items_init() {
	iterator_match = matches.begin();

	results_tree->clear();
	root_item = results_tree->create_item();
	class_items.clear();

	return true;
}

bool EditorHelpSearch::Runner::_phase_class_items() {
	if (!iterator_match) {
		return true;
	}
	ClassMatch &match = iterator_match->value;

	if (search_flags & SEARCH_SHOW_HIERARCHY) {
		if (match.required()) {
			_create_class_hierarchy(match);
		}
	} else {
		if (match.name) {
			_create_class_item(root_item, match.doc, false);
		}
	}

	++iterator_match;
	return !iterator_match;
}

bool EditorHelpSearch::Runner::_phase_member_items_init() {
	iterator_match = matches.begin();

	return true;
}

bool EditorHelpSearch::Runner::_phase_member_items() {
	if (!iterator_match) {
		return true;
	}

	ClassMatch &match = iterator_match->value;

	if (!match.doc || match.doc->name.is_empty()) {
		++iterator_match;
		return false;
	}

	TreeItem *parent_item = (search_flags & SEARCH_SHOW_HIERARCHY) ? class_items[match.doc->name] : root_item;
	bool constructor_created = false;
	for (int i = 0; i < match.methods.size(); i++) {
		String text = match.methods[i]->name;
		if (!constructor_created) {
			if (match.doc->name == match.methods[i]->name) {
				text += " " + TTR("(constructors)");
				constructor_created = true;
			}
		} else {
			if (match.doc->name == match.methods[i]->name) {
				continue;
			}
		}
		_create_method_item(parent_item, match.doc, text, match.methods[i]);
	}
	for (int i = 0; i < match.signals.size(); i++) {
		_create_signal_item(parent_item, match.doc, match.signals[i]);
	}
	for (int i = 0; i < match.constants.size(); i++) {
		_create_constant_item(parent_item, match.doc, match.constants[i]);
	}
	for (int i = 0; i < match.properties.size(); i++) {
		_create_property_item(parent_item, match.doc, match.properties[i]);
	}
	for (int i = 0; i < match.theme_properties.size(); i++) {
		_create_theme_property_item(parent_item, match.doc, match.theme_properties[i]);
	}
	for (int i = 0; i < match.annotations.size(); i++) {
		// Hide the redundant leading @ symbol.
		_create_annotation_item(parent_item, match.doc, match.annotations[i]->name.substr(1), match.annotations[i]);
	}

	++iterator_match;
	return !iterator_match;
}

bool EditorHelpSearch::Runner::_phase_select_match() {
	if (matched_item) {
		matched_item->select(0);
	}
	return true;
}

void EditorHelpSearch::Runner::_match_method_name_and_push_back(Vector<DocData::MethodDoc> &p_methods, Vector<DocData::MethodDoc *> *r_match_methods) {
	// Constructors, Methods, Operators...
	for (int i = 0; i < p_methods.size(); i++) {
		String method_name = (search_flags & SEARCH_CASE_SENSITIVE) ? p_methods[i].name : p_methods[i].name.to_lower();
		if (_all_terms_in_name(method_name) ||
				(term.begins_with(".") && method_name.begins_with(term.substr(1))) ||
				(term.ends_with("(") && method_name.ends_with(term.left(term.length() - 1).strip_edges())) ||
				(term.begins_with(".") && term.ends_with("(") && method_name == term.substr(1, term.length() - 2).strip_edges())) {
			r_match_methods->push_back(const_cast<DocData::MethodDoc *>(&p_methods[i]));
		}
	}
}

bool EditorHelpSearch::Runner::_all_terms_in_name(String name) {
	for (int i = 0; i < terms.size(); i++) {
		if (!_match_string(terms[i], name)) {
			return false;
		}
	}
	return true;
}

bool EditorHelpSearch::Runner::_match_string(const String &p_term, const String &p_string) const {
	if (search_flags & SEARCH_CASE_SENSITIVE) {
		return p_string.find(p_term) > -1;
	} else {
		return p_string.findn(p_term) > -1;
	}
}

void EditorHelpSearch::Runner::_match_item(TreeItem *p_item, const String &p_text) {
	float inverse_length = 1.f / float(p_text.length());

	// Favor types where search term is a substring close to the start of the type.
	float w = 0.5f;
	int pos = p_text.findn(term);
	float score = (pos > -1) ? 1.0f - w * MIN(1, 3 * pos * inverse_length) : MAX(0.f, .9f - w);

	// Favor shorter items: they resemble the search term more.
	w = 0.1f;
	score *= (1 - w) + w * (term.length() * inverse_length);

	if (match_highest_score == 0 || score > match_highest_score) {
		matched_item = p_item;
		match_highest_score = score;
	}
}

String EditorHelpSearch::Runner::_build_method_tooltip(const DocData::ClassDoc *p_class_doc, const DocData::MethodDoc *p_doc) const {
	String tooltip = p_doc->return_type + " " + p_class_doc->name + "." + p_doc->name + "(";
	for (int i = 0; i < p_doc->arguments.size(); i++) {
		const DocData::ArgumentDoc &arg = p_doc->arguments[i];
		tooltip += arg.type + " " + arg.name;
		if (!arg.default_value.is_empty()) {
			tooltip += " = " + arg.default_value;
		}
		if (i < p_doc->arguments.size() - 1) {
			tooltip += ", ";
		}
	}
	tooltip += ")";
	return tooltip;
}

TreeItem *EditorHelpSearch::Runner::_create_class_hierarchy(const ClassMatch &p_match) {
	if (p_match.doc->name.is_empty()) {
		return nullptr;
	}
	if (class_items.has(p_match.doc->name)) {
		return class_items[p_match.doc->name];
	}

	// Ensure parent nodes are created first.
	TreeItem *parent_item = root_item;
	if (!p_match.doc->inherits.is_empty()) {
		if (class_items.has(p_match.doc->inherits)) {
			parent_item = class_items[p_match.doc->inherits];
		} else {
			ClassMatch &base_match = matches[p_match.doc->inherits];
			if (base_match.doc) {
				parent_item = _create_class_hierarchy(base_match);
			}
		}
	}

	TreeItem *class_item = _create_class_item(parent_item, p_match.doc, !p_match.name);
	class_items[p_match.doc->name] = class_item;
	return class_item;
}

TreeItem *EditorHelpSearch::Runner::_create_class_item(TreeItem *p_parent, const DocData::ClassDoc *p_doc, bool p_gray) {
	String tooltip = DTR(p_doc->brief_description.strip_edges());

	TreeItem *item = results_tree->create_item(p_parent);
	item->set_icon(0, EditorNode::get_singleton()->get_class_icon(p_doc->name));
	item->set_text(0, p_doc->name);
	item->set_text(1, TTR("Class"));
	item->set_tooltip_text(0, tooltip);
	item->set_tooltip_text(1, tooltip);
	item->set_metadata(0, "class_name:" + p_doc->name);
	if (p_gray) {
		item->set_custom_color(0, disabled_color);
		item->set_custom_color(1, disabled_color);
	}

	if (p_doc->is_deprecated) {
		Ref<Texture2D> error_icon = ui_service->get_editor_theme_icon("StatusError");
		item->add_button(0, error_icon, 0, false, TTR("This class is marked as deprecated."));
	} else if (p_doc->is_experimental) {
		Ref<Texture2D> warning_icon = ui_service->get_editor_theme_icon("NodeWarning");
		item->add_button(0, warning_icon, 0, false, TTR("This class is marked as experimental."));
	}

	_match_item(item, p_doc->name);

	return item;
}

TreeItem *EditorHelpSearch::Runner::_create_method_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const String &p_text, const DocData::MethodDoc *p_doc) {
	String tooltip = _build_method_tooltip(p_class_doc, p_doc);
	return _create_member_item(p_parent, p_class_doc->name, "MemberMethod", p_doc->name, p_text, TTRC("Method"), "method", tooltip, p_doc->is_deprecated, p_doc->is_experimental);
}

TreeItem *EditorHelpSearch::Runner::_create_signal_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::MethodDoc *p_doc) {
	String tooltip = _build_method_tooltip(p_class_doc, p_doc);
	return _create_member_item(p_parent, p_class_doc->name, "MemberSignal", p_doc->name, p_doc->name, TTRC("Signal"), "signal", tooltip, p_doc->is_deprecated, p_doc->is_experimental);
}

TreeItem *EditorHelpSearch::Runner::_create_annotation_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const String &p_text, const DocData::MethodDoc *p_doc) {
	String tooltip = _build_method_tooltip(p_class_doc, p_doc);
	return _create_member_item(p_parent, p_class_doc->name, "MemberAnnotation", p_doc->name, p_text, TTRC("Annotation"), "annotation", tooltip, p_doc->is_deprecated, p_doc->is_experimental);
}

TreeItem *EditorHelpSearch::Runner::_create_constant_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::ConstantDoc *p_doc) {
	String tooltip = p_class_doc->name + "." + p_doc->name;
	return _create_member_item(p_parent, p_class_doc->name, "MemberConstant", p_doc->name, p_doc->name, TTRC("Constant"), "constant", tooltip, p_doc->is_deprecated, p_doc->is_experimental);
}

TreeItem *EditorHelpSearch::Runner::_create_property_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::PropertyDoc *p_doc) {
	String tooltip = p_doc->type + " " + p_class_doc->name + "." + p_doc->name;
	tooltip += "\n    " + p_class_doc->name + "." + p_doc->setter + "(value) setter";
	tooltip += "\n    " + p_class_doc->name + "." + p_doc->getter + "() getter";
	return _create_member_item(p_parent, p_class_doc->name, "MemberProperty", p_doc->name, p_doc->name, TTRC("Property"), "property", tooltip, p_doc->is_deprecated, p_doc->is_experimental);
}

TreeItem *EditorHelpSearch::Runner::_create_theme_property_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::ThemeItemDoc *p_doc) {
	String tooltip = p_doc->type + " " + p_class_doc->name + "." + p_doc->name;
	return _create_member_item(p_parent, p_class_doc->name, "MemberTheme", p_doc->name, p_doc->name, TTRC("Theme Property"), "theme_item", tooltip, false, false);
}

TreeItem *EditorHelpSearch::Runner::_create_member_item(TreeItem *p_parent, const String &p_class_name, const String &p_icon, const String &p_name, const String &p_text, const String &p_type, const String &p_metatype, const String &p_tooltip, bool is_deprecated, bool is_experimental) {
	Ref<Texture2D> icon;
	String text;
	if (search_flags & SEARCH_SHOW_HIERARCHY) {
		icon = ui_service->get_editor_theme_icon(p_icon);
		text = p_text;
	} else {
		icon = ui_service->get_editor_theme_icon(p_icon);
		text = p_class_name + "." + p_text;
	}

	TreeItem *item = results_tree->create_item(p_parent);
	item->set_icon(0, icon);
	item->set_text(0, text);
	item->set_text(1, TTRGET(p_type));
	item->set_tooltip_text(0, p_tooltip);
	item->set_tooltip_text(1, p_tooltip);
	item->set_metadata(0, "class_" + p_metatype + ":" + p_class_name + ":" + p_name);

	if (is_deprecated) {
		Ref<Texture2D> error_icon = ui_service->get_editor_theme_icon("StatusError");
		item->add_button(0, error_icon, 0, false, TTR("This member is marked as deprecated."));
	} else if (is_experimental) {
		Ref<Texture2D> warning_icon = ui_service->get_editor_theme_icon("NodeWarning");
		item->add_button(0, warning_icon, 0, false, TTR("This member is marked as experimental."));
	}

	_match_item(item, p_name);

	return item;
}

bool EditorHelpSearch::Runner::work(uint64_t slot) {
	// Return true when the search has been completed, otherwise false.
	const uint64_t until = OS::get_singleton()->get_ticks_usec() + slot;
	while (!_slice()) {
		if (OS::get_singleton()->get_ticks_usec() > until) {
			return false;
		}
	}
	return true;
}

EditorHelpSearch::Runner::Runner(Control *p_icon_service, Tree *p_results_tree, const String &p_term, int p_search_flags) :
		ui_service(p_icon_service),
		results_tree(p_results_tree),
		term((p_search_flags & SEARCH_CASE_SENSITIVE) == 0 ? p_term.strip_edges().to_lower() : p_term.strip_edges()),
		search_flags(p_search_flags),
		disabled_color(ui_service->get_theme_color(SNAME("disabled_font_color"), EditorStringName(Editor))) {
}
