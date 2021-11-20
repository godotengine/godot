/*************************************************************************/
/*  visual_script_property_selector.cpp                                  */
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

#include "visual_script_property_selector.h"

#include "../visual_script.h"
#include "../visual_script_builtin_funcs.h"
#include "../visual_script_flow_control.h"
#include "../visual_script_func_nodes.h"
#include "../visual_script_nodes.h"
#include "core/os/keyboard.h"
#include "editor/doc_tools.h"
#include "editor/editor_feature_profile.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "scene/main/node.h"
#include "scene/main/window.h"

void VisualScriptPropertySelector::_update_icons() {
	search_box->set_right_icon(results_tree->get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
	search_box->set_clear_button_enabled(true);
	search_box->add_theme_icon_override("right_icon", results_tree->get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));

	search_visual_script_nodes->set_icon(results_tree->get_theme_icon(SNAME("VisualScript"), SNAME("EditorIcons")));
	search_classes->set_icon(results_tree->get_theme_icon(SNAME("Object"), SNAME("EditorIcons")));
	search_methods->set_icon(results_tree->get_theme_icon(SNAME("MemberMethod"), SNAME("EditorIcons")));
	search_operators->set_icon(results_tree->get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
	search_signals->set_icon(results_tree->get_theme_icon(SNAME("MemberSignal"), SNAME("EditorIcons")));
	search_constants->set_icon(results_tree->get_theme_icon(SNAME("MemberConstant"), SNAME("EditorIcons")));
	search_properties->set_icon(results_tree->get_theme_icon(SNAME("MemberProperty"), SNAME("EditorIcons")));
	search_theme_items->set_icon(results_tree->get_theme_icon(SNAME("MemberTheme"), SNAME("EditorIcons")));

	case_sensitive_button->set_icon(results_tree->get_theme_icon(SNAME("MatchCase"), SNAME("EditorIcons")));
	hierarchy_button->set_icon(results_tree->get_theme_icon(SNAME("ClassList"), SNAME("EditorIcons")));
}

void VisualScriptPropertySelector::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;

	if (k.is_valid()) {
		switch (k->get_keycode()) {
			case Key::UP:
			case Key::DOWN:
			case Key::PAGEUP:
			case Key::PAGEDOWN: {
				results_tree->gui_input(k);
				search_box->accept_event();
			} break;
			default:
				break;
		}
	}
}

void VisualScriptPropertySelector::_update_results_i(int p_int) {
	_update_results();
}

void VisualScriptPropertySelector::_update_results_s(String p_string) {
	_update_results();
}

void VisualScriptPropertySelector::_update_results() {
	_update_icons();
	search_runner = Ref<SearchRunner>(memnew(SearchRunner(this, results_tree)));
	set_process(true);
}

void VisualScriptPropertySelector::_confirmed() {
	TreeItem *ti = results_tree->get_selected();
	if (!ti) {
		return;
	}
	emit_signal(SNAME("selected"), ti->get_metadata(0), ti->get_metadata(1), connecting);
	set_visible(false);
}

void VisualScriptPropertySelector::_item_selected() {
	if (results_tree->get_selected()->has_meta("description")) {
		help_bit->set_text(results_tree->get_selected()->get_meta("description"));
	} else {
		help_bit->set_text("No description available");
	}
}

void VisualScriptPropertySelector::_hide_requested() {
	_cancel_pressed(); // From AcceptDialog.
}

void VisualScriptPropertySelector::_notification(int p_what) {
	switch (p_what) {
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			_update_icons();
		} break;
		case NOTIFICATION_ENTER_TREE: {
			connect("confirmed", callable_mp(this, &VisualScriptPropertySelector::_confirmed));
		} break;
		case NOTIFICATION_PROCESS: {
			// Update background search.
			if (search_runner.is_valid()) {
				if (search_runner->work()) {
					// Search done.
					get_ok_button()->set_disabled(!results_tree->get_selected());

					search_runner = Ref<SearchRunner>();
					set_process(false);
				}
			} else {
				// if one is valid
				set_process(false);
			}
		} break;
	}
}

void VisualScriptPropertySelector::select_method_from_base_type(const String &p_base, const bool p_virtuals_only, const bool p_connecting, bool clear_text) {
	set_title(TTR("Select method from base type"));
	base_type = p_base;
	base_script = "";
	type = Variant::NIL;
	connecting = p_connecting;

	if (clear_text) {
		if (p_virtuals_only) {
			search_box->set_text("._"); // show all _methods
			search_box->set_caret_column(2);
		} else {
			search_box->set_text("."); // show all methods
			search_box->set_caret_column(1);
		}
	}

	search_visual_script_nodes->set_pressed(false);
	search_classes->set_pressed(false);
	search_methods->set_pressed(true);
	search_operators->set_pressed(false);
	search_signals->set_pressed(false);
	search_constants->set_pressed(false);
	search_properties->set_pressed(false);
	search_theme_items->set_pressed(false);

	scope_combo->select(2); //id0 = "Search Related" //id2 = "Search Base" //id3 = "Search Inheriters" //id4 = "Search Unrelated"

	results_tree->clear();
	show_window(.5f);
	search_box->grab_focus();

	_update_results();
}

void VisualScriptPropertySelector::select_from_base_type(const String &p_base, const String &p_base_script, bool p_virtuals_only, const bool p_connecting, bool clear_text) {
	set_title(TTR("Select from base type"));
	base_type = p_base;
	base_script = p_base_script.lstrip("res://").quote(); // filepath to EditorHelp::get_doc_data().name
	type = Variant::NIL;
	connecting = p_connecting;

	if (clear_text) {
		if (p_virtuals_only) {
			search_box->set_text("_");
		} else {
			search_box->set_text(" ");
		}
	}
	search_box->select_all();

	search_visual_script_nodes->set_pressed(false);
	search_classes->set_pressed(false);
	search_methods->set_pressed(true);
	search_operators->set_pressed(false);
	search_signals->set_pressed(true);
	search_constants->set_pressed(false);
	search_properties->set_pressed(true);
	search_theme_items->set_pressed(false);

	// When class is Input only show inheritors
	scope_combo->select(0); //id0 = "Search Related" //id2 = "Search Base" //id3 = "Search Inheriters" //id4 = "Search Unrelated"

	results_tree->clear();
	show_window(.5f);
	search_box->grab_focus();
	_update_results();
}

void VisualScriptPropertySelector::select_from_script(const Ref<Script> &p_script, const bool p_connecting, bool clear_text) {
	set_title(TTR("Select from script"));
	ERR_FAIL_COND(p_script.is_null());

	base_type = p_script->get_instance_base_type();
	base_script = p_script->get_path().lstrip("res://").quote(); // filepath to EditorHelp::get_doc_data().name
	type = Variant::NIL;
	script = p_script->get_instance_id();
	connecting = p_connecting;

	if (clear_text) {
		search_box->set_text("");
	}
	search_box->select_all();

	search_visual_script_nodes->set_pressed(false);
	search_classes->set_pressed(true);
	search_methods->set_pressed(true);
	search_operators->set_pressed(true);
	search_signals->set_pressed(true);
	search_constants->set_pressed(true);
	search_properties->set_pressed(true);
	search_theme_items->set_pressed(false);

	scope_combo->select(2); //id0 = "Search Related" //id2 = "Search Base" //id3 = "Search Inheriters" //id4 = "Search Unrelated"

	results_tree->clear();
	show_window(.5f);
	search_box->grab_focus();
	_update_results();
}

void VisualScriptPropertySelector::select_from_basic_type(Variant::Type p_type, const bool p_connecting, bool clear_text) {
	set_title(TTR("Select from basic type"));
	ERR_FAIL_COND(p_type == Variant::NIL);
	base_type = Variant::get_type_name(p_type);
	base_script = "";
	type = p_type;
	connecting = p_connecting;

	if (clear_text) {
		search_box->set_text(" ");
	}
	search_box->select_all();

	search_visual_script_nodes->set_pressed(false);
	search_classes->set_pressed(false);
	search_methods->set_pressed(true);
	search_operators->set_pressed(true);
	search_signals->set_pressed(false);
	search_constants->set_pressed(true);
	search_properties->set_pressed(true);
	search_theme_items->set_pressed(false);

	scope_combo->select(2); //id0 = "Search Related" //id2 = "Search Base" //id3 = "Search Inheriters" //id4 = "Search Unrelated" //id5 "Search All"

	results_tree->clear();
	show_window(.5f);
	search_box->grab_focus();

	_update_results();
}

void VisualScriptPropertySelector::select_from_action(const String &p_type, const bool p_connecting, bool clear_text) {
	set_title(TTR("Select from action"));
	base_type = p_type;
	base_script = "";
	type = Variant::NIL;
	connecting = p_connecting;

	if (clear_text) {
		search_box->set_text("");
	}
	search_box->select_all();

	search_visual_script_nodes->set_pressed(true);
	search_classes->set_pressed(false);
	search_methods->set_pressed(false);
	search_operators->set_pressed(false);
	search_signals->set_pressed(false);
	search_constants->set_pressed(false);
	search_properties->set_pressed(false);
	search_theme_items->set_pressed(false);

	scope_combo->select(0); //id0 = "Search Related" //id2 = "Search Base" //id3 = "Search Inheriters" //id4 = "Search Unrelated" //id5 "Search All"

	results_tree->clear();
	show_window(.5f);
	search_box->grab_focus();
	_update_results();
}

void VisualScriptPropertySelector::select_from_instance(Object *p_instance, const bool p_connecting, bool clear_text) {
	set_title(TTR("Select from instance"));
	base_type = p_instance->get_class();

	const Ref<Script> &p_script = p_instance->get_script();
	if (p_script == nullptr) {
		base_script = "";
	} else {
		base_script = p_script->get_path().lstrip("res://").quote(); // filepath to EditorHelp::get_doc_data().name
	}

	type = Variant::NIL;
	connecting = p_connecting;

	if (clear_text) {
		search_box->set_text(" ");
	}
	search_box->select_all();

	search_visual_script_nodes->set_pressed(false);
	search_classes->set_pressed(false);
	search_methods->set_pressed(true);
	search_operators->set_pressed(false);
	search_signals->set_pressed(true);
	search_constants->set_pressed(true);
	search_properties->set_pressed(true);
	search_theme_items->set_pressed(false);

	scope_combo->select(2); //id0 = "Search Related" //id2 = "Search Base" //id3 = "Search Inheriters" //id4 = "Search Unrelated" //id5 "Search All"

	results_tree->clear();
	show_window(.5f);
	search_box->grab_focus();
	_update_results();
}

void VisualScriptPropertySelector::select_from_visual_script(const Ref<Script> &p_script, bool clear_text) {
	set_title(TTR("Select from visual script"));
	base_type = p_script->get_instance_base_type();
	if (p_script == nullptr) {
		base_script = "";
	} else {
		base_script = p_script->get_path().lstrip("res://").quote(); // filepath to EditorHelp::get_doc_data().name
	}
	type = Variant::NIL;
	connecting = false;

	if (clear_text) {
		search_box->set_text(" ");
	}
	search_box->select_all();

	search_visual_script_nodes->set_pressed(true);
	search_classes->set_pressed(false);
	search_methods->set_pressed(true);
	search_operators->set_pressed(false);
	search_signals->set_pressed(true);
	search_constants->set_pressed(true);
	search_properties->set_pressed(true);
	search_theme_items->set_pressed(false);

	scope_combo->select(2); //id0 = "Search Related" //id2 = "Search Base" //id3 = "Search Inheriters" //id4 = "Search Unrelated" //id5 "Search All"

	results_tree->clear();
	show_window(.5f);
	search_box->grab_focus();
	_update_results();
}

void VisualScriptPropertySelector::show_window(float p_screen_ratio) {
	popup_centered_ratio(p_screen_ratio);
}

void VisualScriptPropertySelector::_bind_methods() {
	ADD_SIGNAL(MethodInfo("selected", PropertyInfo(Variant::STRING, "name"), PropertyInfo(Variant::STRING, "category"), PropertyInfo(Variant::BOOL, "connecting")));
}

VisualScriptPropertySelector::VisualScriptPropertySelector() {
	virtuals_only = false;

	vbox = memnew(VBoxContainer);
	add_child(vbox);

	HBoxContainer *hbox = memnew(HBoxContainer);
	hbox->set_alignment(hbox->ALIGNMENT_CENTER);
	vbox->add_child(hbox);

	case_sensitive_button = memnew(Button);
	case_sensitive_button->set_flat(true);
	case_sensitive_button->set_tooltip(TTR("Case Sensitive"));
	case_sensitive_button->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
	case_sensitive_button->set_toggle_mode(true);
	case_sensitive_button->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(case_sensitive_button);

	hierarchy_button = memnew(Button);
	hierarchy_button->set_flat(true);
	hierarchy_button->set_tooltip(TTR("Show Hierarchy"));
	hierarchy_button->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
	hierarchy_button->set_toggle_mode(true);
	hierarchy_button->set_pressed(true);
	hierarchy_button->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(hierarchy_button);

	hbox->add_child(memnew(VSeparator));

	search_visual_script_nodes = memnew(Button);
	search_visual_script_nodes->set_flat(true);
	search_visual_script_nodes->set_tooltip(TTR("Search Visual Script Nodes"));
	search_visual_script_nodes->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
	search_visual_script_nodes->set_toggle_mode(true);
	search_visual_script_nodes->set_pressed(true);
	search_visual_script_nodes->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(search_visual_script_nodes);

	search_classes = memnew(Button);
	search_classes->set_flat(true);
	search_classes->set_tooltip(TTR("Search Classes"));
	search_classes->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
	search_classes->set_toggle_mode(true);
	search_classes->set_pressed(true);
	search_classes->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(search_classes);

	search_operators = memnew(Button);
	search_operators->set_flat(true);
	search_operators->set_tooltip(TTR("Search Operators"));
	search_operators->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
	search_operators->set_toggle_mode(true);
	search_operators->set_pressed(true);
	search_operators->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(search_operators);

	hbox->add_child(memnew(VSeparator));

	search_methods = memnew(Button);
	search_methods->set_flat(true);
	search_methods->set_tooltip(TTR("Search Methods"));
	search_methods->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
	search_methods->set_toggle_mode(true);
	search_methods->set_pressed(true);
	search_methods->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(search_methods);

	search_signals = memnew(Button);
	search_signals->set_flat(true);
	search_signals->set_tooltip(TTR("Search Signals"));
	search_signals->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
	search_signals->set_toggle_mode(true);
	search_signals->set_pressed(true);
	search_signals->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(search_signals);

	search_constants = memnew(Button);
	search_constants->set_flat(true);
	search_constants->set_tooltip(TTR("Search Constants"));
	search_constants->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
	search_constants->set_toggle_mode(true);
	search_constants->set_pressed(true);
	search_constants->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(search_constants);

	search_properties = memnew(Button);
	search_properties->set_flat(true);
	search_properties->set_tooltip(TTR("Search Properties"));
	search_properties->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
	search_properties->set_toggle_mode(true);
	search_properties->set_pressed(true);
	search_properties->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(search_properties);

	search_theme_items = memnew(Button);
	search_theme_items->set_flat(true);
	search_theme_items->set_tooltip(TTR("Search Theme Items"));
	search_theme_items->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
	search_theme_items->set_toggle_mode(true);
	search_theme_items->set_pressed(true);
	search_theme_items->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(search_theme_items);

	scope_combo = memnew(OptionButton);
	scope_combo->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	scope_combo->set_tooltip(TTR("Select the search limits"));
	scope_combo->set_stretch_ratio(0); // Fixed width.
	scope_combo->add_item(TTR("Search Related"), SCOPE_RELATED);
	scope_combo->add_separator();
	scope_combo->add_item(TTR("Search Base"), SCOPE_BASE);
	scope_combo->add_item(TTR("Search Inheriters"), SCOPE_INHERITERS);
	scope_combo->add_item(TTR("Search Unrelated"), SCOPE_UNRELATED);
	scope_combo->add_item(TTR("Search All"), SCOPE_ALL);
	scope_combo->connect("item_selected", callable_mp(this, &VisualScriptPropertySelector::_update_results_i));
	hbox->add_child(scope_combo);

	search_box = memnew(LineEdit);
	search_box->set_tooltip(TTR("Enter \" \" to show all filterd options\nEnter \".\" to show all filterd methods, operators and constructors\nUse CTRL_KEY to drop property setters"));
	search_box->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_box->connect("text_changed", callable_mp(this, &VisualScriptPropertySelector::_update_results_s));
	search_box->connect("gui_input", callable_mp(this, &VisualScriptPropertySelector::_sbox_input));
	register_text_enter(search_box);
	vbox->add_child(search_box);

	results_tree = memnew(Tree);
	results_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	results_tree->set_hide_root(true);
	results_tree->set_hide_folding(false);
	results_tree->set_columns(2);
	results_tree->set_column_title(0, TTR("Name"));
	results_tree->set_column_clip_content(0, true);
	results_tree->set_column_title(1, TTR("Member Type"));
	results_tree->set_column_expand(1, false);
	results_tree->set_column_custom_minimum_width(1, 150 * EDSCALE);
	results_tree->set_column_clip_content(1, true);
	results_tree->set_custom_minimum_size(Size2(0, 100) * EDSCALE);
	results_tree->set_select_mode(Tree::SELECT_ROW);
	results_tree->connect("item_activated", callable_mp(this, &VisualScriptPropertySelector::_confirmed));
	results_tree->connect("item_selected", callable_mp(this, &VisualScriptPropertySelector::_item_selected));
	vbox->add_child(results_tree);

	help_bit = memnew(EditorHelpBit);
	vbox->add_child(help_bit);
	help_bit->connect("request_hide", callable_mp(this, &VisualScriptPropertySelector::_hide_requested));
	get_ok_button()->set_text(TTR("Open"));
	get_ok_button()->set_disabled(true);
	set_hide_on_ok(false);
}

bool VisualScriptPropertySelector::SearchRunner::_is_class_disabled_by_feature_profile(const StringName &p_class) {
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

bool VisualScriptPropertySelector::SearchRunner::_is_class_disabled_by_scope(const StringName &p_class) {
	bool is_base_script = false;
	if (p_class == selector_ui->base_script) {
		is_base_script = true;
	}
	bool is_base = false;
	if (selector_ui->base_type == p_class) {
		is_base = true;
	}
	bool is_parent = false;
	if ((ClassDB::is_parent_class(selector_ui->base_type, p_class)) && !is_base) {
		is_parent = true;
	}

	bool is_inheriter = false;
	List<StringName> inheriters;
	ClassDB::get_inheriters_from_class(selector_ui->base_type, &inheriters);
	if (inheriters.find(p_class)) {
		is_inheriter = true;
	}

	if (scope_flags & SCOPE_BASE) {
		if (is_base_script || is_base || is_parent) {
			return false;
		}
	}
	if (scope_flags & SCOPE_INHERITERS) {
		if (is_base_script || is_base || is_inheriter) {
			return false;
		}
	}
	//	if (scope_flags & SCOPE_RELATED) {
	//		/* code */
	//	}
	if (scope_flags & SCOPE_UNRELATED) {
		if (!is_base_script && !is_base && !is_inheriter) {
			return false;
		}
	}
	return true;
}

bool VisualScriptPropertySelector::SearchRunner::_slice() {
	bool phase_done = false;
	switch (phase) {
		case PHASE_INIT:
			phase_done = _phase_init();
			break;
		case PHASE_MATCH_CLASSES_INIT:
			phase_done = _phase_match_classes_init();
			break;
		case PHASE_NODE_CLASSES_INIT:
			phase_done = _phase_node_classes_init();
			break;
		case PHASE_NODE_CLASSES_BUILD:
			phase_done = _phase_node_classes_build();
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

bool VisualScriptPropertySelector::SearchRunner::_phase_init() {
	search_flags = 0; // selector_ui->filter_combo->get_selected_id();
	if (selector_ui->search_visual_script_nodes->is_pressed()) {
		search_flags |= SEARCH_VISUAL_SCRIPT_NODES;
	}
	if (selector_ui->search_classes->is_pressed()) {
		search_flags |= SEARCH_CLASSES;
	}
	//	if (selector_ui->search_constructors->is_pressed()) {
	search_flags |= SEARCH_CONSTRUCTORS;
	//	}
	if (selector_ui->search_methods->is_pressed()) {
		search_flags |= SEARCH_METHODS;
	}
	if (selector_ui->search_operators->is_pressed()) {
		search_flags |= SEARCH_OPERATORS;
	}
	if (selector_ui->search_signals->is_pressed()) {
		search_flags |= SEARCH_SIGNALS;
	}
	if (selector_ui->search_constants->is_pressed()) {
		search_flags |= SEARCH_CONSTANTS;
	}
	if (selector_ui->search_properties->is_pressed()) {
		search_flags |= SEARCH_PROPERTIES;
	}
	if (selector_ui->search_theme_items->is_pressed()) {
		search_flags |= SEARCH_THEME_ITEMS;
	}
	if (selector_ui->case_sensitive_button->is_pressed()) {
		search_flags |= SEARCH_CASE_SENSITIVE;
	}
	if (selector_ui->hierarchy_button->is_pressed()) {
		search_flags |= SEARCH_SHOW_HIERARCHY;
	}
	scope_flags = selector_ui->scope_combo->get_selected_id();

	return true;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_match_classes_init() {
	combined_docs = EditorHelp::get_doc_data()->class_list;
	matches.clear();
	matched_item = nullptr;
	match_highest_score = 0;

	if (
			(selector_ui->base_script.unquote() != "") &&
			(selector_ui->base_script.unquote() != ".") &&
			!combined_docs.has(selector_ui->base_script)) {
		String file_path = "res://" + selector_ui->base_script.unquote(); // EditorHelp::get_doc_data().name to filepath
		Ref<Script> script;
		script = ResourceLoader::load(file_path);
		if (!script.is_null()) {
			DocData::ClassDoc class_doc = DocData::ClassDoc();

			class_doc.name = selector_ui->base_script;

			class_doc.inherits = script->get_instance_base_type();
			class_doc.brief_description = ".vs files not suported by EditorHelp::get_doc_data()";
			class_doc.description = "";

			Object *obj = ObjectDB::get_instance(script->get_instance_id());
			if (Object::cast_to<Script>(obj)) {
				List<MethodInfo> methods;
				Object::cast_to<Script>(obj)->get_script_method_list(&methods);
				for (List<MethodInfo>::Element *M = methods.front(); M; M = M->next()) {
					class_doc.methods.push_back(_get_method_doc(M->get()));
				}

				List<MethodInfo> signals;
				Object::cast_to<Script>(obj)->get_script_signal_list(&signals);
				for (List<MethodInfo>::Element *S = signals.front(); S; S = S->next()) {
					class_doc.signals.push_back(_get_method_doc(S->get()));
				}

				List<PropertyInfo> propertys;
				Object::cast_to<Script>(obj)->get_script_property_list(&propertys);
				for (List<PropertyInfo>::Element *P = propertys.front(); P; P = P->next()) {
					DocData::PropertyDoc pd = DocData::PropertyDoc();
					pd.name = P->get().name;
					pd.type = Variant::get_type_name(P->get().type);
					class_doc.properties.push_back(pd);
				}
			}
			combined_docs.insert(class_doc.name, class_doc);
		}
	}
	iterator_doc = combined_docs.front();
	return true;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_node_classes_init() {
	VisualScriptLanguage::singleton->get_registered_node_names(&vs_nodes);
	_add_class_doc("functions", "", "");
	_add_class_doc("operators", "", "");
	return true;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_node_classes_build() {
	if (vs_nodes.is_empty()) {
		return true;
	}
	String registerd_node_name = vs_nodes[0];
	vs_nodes.pop_front();

	Vector<String> path = registerd_node_name.split("/");
	if (path[0] == "constants") {
		_add_class_doc(registerd_node_name, "", "constants");
	} else if (path[0] == "custom") {
		_add_class_doc(registerd_node_name, "", "custom");
	} else if (path[0] == "data") {
		_add_class_doc(registerd_node_name, "", "data");
	} else if (path[0] == "flow_control") {
		_add_class_doc(registerd_node_name, "", "flow_control");
	} else if (path[0] == "functions") {
		if (path[1] == "built_in") {
			_add_class_doc(registerd_node_name, "functions", "built_in");
		} else if (path[1] == "by_type") {
			if (search_flags & SEARCH_CLASSES) {
				_add_class_doc(registerd_node_name, path[2], "by_type_class");
			}
		} else if (path[1] == "constructors") {
			if (search_flags & SEARCH_CLASSES) {
				_add_class_doc(registerd_node_name, path[2].substr(0, path[2].find_char('(')), "constructors_class");
			}
		} else if (path[1] == "deconstruct") {
			_add_class_doc(registerd_node_name, "", "deconstruct");
		} else if (path[1] == "wait") {
			_add_class_doc(registerd_node_name, "functions", "yield");
		} else {
			_add_class_doc(registerd_node_name, "functions", "");
		}
	} else if (path[0] == "index") {
		_add_class_doc(registerd_node_name, "", "index");
	} else if (path[0] == "operators") {
		if (path[1] == "bitwise") {
			_add_class_doc(registerd_node_name, "operators", "bitwise");
		} else if (path[1] == "compare") {
			_add_class_doc(registerd_node_name, "operators", "compare");
		} else if (path[1] == "logic") {
			_add_class_doc(registerd_node_name, "operators", "logic");
		} else if (path[1] == "math") {
			_add_class_doc(registerd_node_name, "operators", "math");
		} else {
			_add_class_doc(registerd_node_name, "operators", "");
		}
	}
	return false;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_match_classes() {
	DocData::ClassDoc &class_doc = iterator_doc->value();
	if (
			(!_is_class_disabled_by_feature_profile(class_doc.name) && !_is_class_disabled_by_scope(class_doc.name)) ||
			_match_visual_script(class_doc)) {
		if (class_doc.inherits == "VisualScriptCustomNode") {
			class_doc.script_path = "res://" + class_doc.name.unquote();
			Ref<Script> script = ResourceLoader::load(class_doc.script_path);
			Ref<VisualScriptCustomNode> vsn;
			vsn.instantiate();
			vsn->set_script(script);
			class_doc.name = vsn->get_caption();
			if (combined_docs.has(vsn->get_category())) {
				class_doc.inherits = vsn->get_category();
			} else if (combined_docs.has("VisualScriptNode/" + vsn->get_category())) {
				class_doc.inherits = "VisualScriptNode/" + vsn->get_category();
			} else if (combined_docs.has("VisualScriptCustomNode/" + vsn->get_category())) {
				class_doc.inherits = "VisualScriptCustomNode/" + vsn->get_category();
			} else {
				class_doc.inherits = "";
			}
			class_doc.category = "VisualScriptCustomNode/" + vsn->get_category();
			class_doc.brief_description = "";
			class_doc.constructors.clear();
			class_doc.methods.clear();
			class_doc.operators.clear();
			class_doc.signals.clear();
			class_doc.constants.clear();
			class_doc.enums.clear();
			class_doc.properties.clear();
			class_doc.theme_properties.clear();
		}

		matches[class_doc.name] = ClassMatch();
		ClassMatch &match = matches[class_doc.name];

		match.category = class_doc.category;
		match.doc = &class_doc;
		// Match class name.
		if (search_flags & SEARCH_CLASSES || _match_visual_script(class_doc)) {
			if (term == "") {
				match.name = !_match_is_hidden(class_doc);
			} else {
				match.name = _match_string(term, class_doc.name);
			}
			//	match.name = term == "" || _match_string(term, class_doc.name);
		}

		// Match members if the term is long enough.
		if (term.length() >= 0) {
			if (search_flags & SEARCH_CONSTRUCTORS) {
				for (int i = 0; i < class_doc.constructors.size(); i++) {
					String method_name = (search_flags & SEARCH_CASE_SENSITIVE) ? class_doc.constructors[i].name : class_doc.constructors[i].name.to_lower();
					if (method_name.find(term) > -1 ||
							term == " " ||
							(term.begins_with(".") && method_name.begins_with(term.substr(1))) ||
							(term.ends_with("(") && method_name.ends_with(term.left(term.length() - 1).strip_edges())) ||
							(term.begins_with(".") && term.ends_with("(") && method_name == term.substr(1, term.length() - 2).strip_edges())) {
						match.constructors.push_back(const_cast<DocData::MethodDoc *>(&class_doc.constructors[i]));
					}
				}
			}
			if (search_flags & SEARCH_METHODS) {
				for (int i = 0; i < class_doc.methods.size(); i++) {
					String method_name = (search_flags & SEARCH_CASE_SENSITIVE) ? class_doc.methods[i].name : class_doc.methods[i].name.to_lower();
					if (method_name.find(term) > -1 ||
							term == " " ||
							(term.begins_with(".") && method_name.begins_with(term.substr(1))) ||
							(term.ends_with("(") && method_name.ends_with(term.left(term.length() - 1).strip_edges())) ||
							(term.begins_with(".") && term.ends_with("(") && method_name == term.substr(1, term.length() - 2).strip_edges())) {
						match.methods.push_back(const_cast<DocData::MethodDoc *>(&class_doc.methods[i]));
					}
				}
			}
			if (search_flags & SEARCH_OPERATORS) {
				for (int i = 0; i < class_doc.operators.size(); i++) {
					String method_name = (search_flags & SEARCH_CASE_SENSITIVE) ? class_doc.operators[i].name : class_doc.operators[i].name.to_lower();
					if (method_name.find(term) > -1 ||
							term == " " ||
							(term.begins_with(".") && method_name.begins_with(term.substr(1))) ||
							(term.ends_with("(") && method_name.ends_with(term.left(term.length() - 1).strip_edges())) ||
							(term.begins_with(".") && term.ends_with("(") && method_name == term.substr(1, term.length() - 2).strip_edges())) {
						match.operators.push_back(const_cast<DocData::MethodDoc *>(&class_doc.operators[i]));
					}
				}
			}
			if (search_flags & SEARCH_SIGNALS) {
				for (int i = 0; i < class_doc.signals.size(); i++) {
					if (_match_string(term, class_doc.signals[i].name) ||
							term == " ") {
						match.signals.push_back(const_cast<DocData::MethodDoc *>(&class_doc.signals[i]));
					}
				}
			}
			if (search_flags & SEARCH_CONSTANTS) {
				for (int i = 0; i < class_doc.constants.size(); i++) {
					if (_match_string(term, class_doc.constants[i].name) ||
							term == " ") {
						match.constants.push_back(const_cast<DocData::ConstantDoc *>(&class_doc.constants[i]));
					}
				}
			}
			if (search_flags & SEARCH_PROPERTIES) {
				for (int i = 0; i < class_doc.properties.size(); i++) {
					if (_match_string(term, class_doc.properties[i].name) ||
							term == " " ||
							_match_string(term, class_doc.properties[i].getter) ||
							_match_string(term, class_doc.properties[i].setter)) {
						match.properties.push_back(const_cast<DocData::PropertyDoc *>(&class_doc.properties[i]));
					}
				}
			}
			if (search_flags & SEARCH_THEME_ITEMS) {
				for (int i = 0; i < class_doc.theme_properties.size(); i++) {
					if (_match_string(term, class_doc.theme_properties[i].name) ||
							term == " ") {
						match.theme_properties.push_back(const_cast<DocData::ThemeItemDoc *>(&class_doc.theme_properties[i]));
					}
				}
			}
		}
	}

	iterator_doc = iterator_doc->next();
	return !iterator_doc;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_class_items_init() {
	results_tree->clear();
	iterator_match = matches.front();

	root_item = results_tree->create_item();
	class_items.clear();

	return true;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_class_items() {
	if (!iterator_match) {
		return true;
	}

	ClassMatch &match = iterator_match->value();

	if (search_flags & SEARCH_SHOW_HIERARCHY) {
		if (match.required()) {
			_create_class_hierarchy(match);
		}
	} else {
		if (match.name) {
			_create_class_item(root_item, match.doc, true);
		}
	}

	iterator_match = iterator_match->next();
	return !iterator_match;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_member_items_init() {
	iterator_match = matches.front();

	return true;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_member_items() {
	if (!iterator_match) {
		return true;
	}

	ClassMatch &match = iterator_match->value();

	TreeItem *parent = (search_flags & SEARCH_SHOW_HIERARCHY) ? class_items[match.doc->name] : root_item;
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
		_create_method_item(parent, match.doc, text, match.methods[i]);
	}
	for (int i = 0; i < match.signals.size(); i++) {
		_create_signal_item(parent, match.doc, match.signals[i]);
	}
	for (int i = 0; i < match.constants.size(); i++) {
		_create_constant_item(parent, match.doc, match.constants[i]);
	}
	for (int i = 0; i < match.properties.size(); i++) {
		_create_property_item(parent, match.doc, match.properties[i]);
	}
	for (int i = 0; i < match.theme_properties.size(); i++) {
		_create_theme_property_item(parent, match.doc, match.theme_properties[i]);
	}

	iterator_match = iterator_match->next();
	return !iterator_match;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_select_match() {
	if (matched_item) {
		matched_item->select(0);
	}
	return true;
}

bool VisualScriptPropertySelector::SearchRunner::_match_string(const String &p_term, const String &p_string) const {
	if (search_flags & SEARCH_CASE_SENSITIVE) {
		return p_string.find(p_term) > -1;
	} else {
		return p_string.findn(p_term) > -1;
	}
}

bool VisualScriptPropertySelector::SearchRunner::_match_visual_script(DocData::ClassDoc &class_doc) {
	if (class_doc.category.ends_with("_class")) {
		if (class_doc.category.begins_with("VisualScript") && search_flags & SEARCH_CLASSES) {
			if (matches.has(class_doc.inherits)) {
				return true;
			}
		}
		return false;
	}
	if (class_doc.category.begins_with("VisualScript") && search_flags & SEARCH_VISUAL_SCRIPT_NODES) {
		return true;
	}
	if (class_doc.name.begins_with("operators") && search_flags & SEARCH_OPERATORS) {
		return true;
	}
	if (class_doc.category.begins_with("VisualScriptNode/deconstruct")) {
		if (class_doc.name.find(selector_ui->base_type, 0) > -1) {
			return true;
		}
	}

	return false;
}

bool VisualScriptPropertySelector::SearchRunner::_match_is_hidden(DocData::ClassDoc &class_doc) {
	if (class_doc.category.begins_with("VisualScript")) {
		if (class_doc.name.begins_with("flow_control")) {
			return false;
		} else if (class_doc.name.begins_with("operators")) {
			return !(search_flags & SEARCH_OPERATORS);
		} else if (class_doc.name.begins_with("functions/built_in/print")) {
			return false;
		}
		return true;
	}
	return false;
}

void VisualScriptPropertySelector::SearchRunner::_match_item(TreeItem *p_item, const String &p_text) {
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

void VisualScriptPropertySelector::SearchRunner::_add_class_doc(String class_name, String inherits, String category) {
	DocData::ClassDoc class_doc = DocData::ClassDoc();
	class_doc.name = class_name;
	class_doc.inherits = inherits;
	class_doc.category = "VisualScriptNode/" + category;
	class_doc.brief_description = category;
	combined_docs.insert(class_doc.name, class_doc);
}

DocData::MethodDoc VisualScriptPropertySelector::SearchRunner::_get_method_doc(MethodInfo method_info) {
	DocData::MethodDoc method_doc = DocData::MethodDoc();
	method_doc.name = method_info.name;
	method_doc.return_type = Variant::get_type_name(method_info.return_val.type);
	method_doc.description = "No description available";
	for (List<PropertyInfo>::Element *P = method_info.arguments.front(); P; P = P->next()) {
		DocData::ArgumentDoc argument_doc = DocData::ArgumentDoc();
		argument_doc.name = P->get().name;
		argument_doc.type = Variant::get_type_name(P->get().type);
		method_doc.arguments.push_back(argument_doc);
	}
	return method_doc;
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_class_hierarchy(const ClassMatch &p_match) {
	if (class_items.has(p_match.doc->name)) {
		return class_items[p_match.doc->name];
	}

	// Ensure parent nodes are created first.
	TreeItem *parent = root_item;
	if (p_match.doc->inherits != "") {
		if (class_items.has(p_match.doc->inherits)) {
			parent = class_items[p_match.doc->inherits];
		} else if (matches.has(p_match.doc->inherits)) {
			ClassMatch &base_match = matches[p_match.doc->inherits];
			parent = _create_class_hierarchy(base_match);
		}
	}

	TreeItem *class_item = _create_class_item(parent, p_match.doc, !p_match.name);
	class_items[p_match.doc->name] = class_item;
	return class_item;
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_class_item(TreeItem *p_parent, const DocData::ClassDoc *p_doc, bool p_gray) {
	Ref<Texture2D> icon = empty_icon;
	String text_0 = p_doc->name;
	String text_1 = "Class";

	String what = "Class";
	String details = p_doc->name;
	if (p_doc->category.begins_with("VisualScriptCustomNode/")) {
		Vector<String> path = p_doc->name.split("/");
		icon = ui_service->get_theme_icon("VisualScript", "EditorIcons");
		text_0 = path[path.size() - 1];
		text_1 = "VisualScriptCustomNode";
		what = "VisualScriptCustomNode";
		details = "CustomNode";
	} else if (p_doc->category.begins_with("VisualScriptNode/")) {
		Vector<String> path = p_doc->name.split("/");
		icon = ui_service->get_theme_icon("VisualScript", "EditorIcons");
		text_0 = path[path.size() - 1];
		if (p_doc->category.begins_with("VisualScriptNode/deconstruct")) {
			text_0 = "deconstruct " + text_0;
		}
		text_1 = "VisualScriptNode";
		what = "VisualScriptNode";
		details = p_doc->name;

		if (path.size() == 1) {
			if (path[0] == "functions" || path[0] == "operators") {
				text_1 = "VisualScript";
				p_gray = true;
				what = "no_result";
				details = "";
			}
		}

	} else {
		if (p_doc->name.is_quoted()) {
			text_0 = p_doc->name.unquote().get_file();
			if (ui_service->has_theme_icon(p_doc->inherits, "EditorIcons")) {
				icon = ui_service->get_theme_icon(p_doc->inherits, "EditorIcons");
			}
		} else if (ui_service->has_theme_icon(p_doc->name, "EditorIcons")) {
			icon = ui_service->get_theme_icon(p_doc->name, "EditorIcons");
		} else if (ClassDB::class_exists(p_doc->name) && ClassDB::is_parent_class(p_doc->name, "Object")) {
			icon = ui_service->get_theme_icon(SNAME("Object"), SNAME("EditorIcons"));
		}
	}
	String tooltip = p_doc->brief_description.strip_edges();

	TreeItem *item = results_tree->create_item(p_parent);
	item->set_icon(0, icon);
	item->set_text(0, text_0);
	item->set_text(1, TTR(text_1));
	item->set_tooltip(0, tooltip);
	item->set_tooltip(1, tooltip);
	item->set_metadata(0, details);
	item->set_metadata(1, what);
	if (p_gray) {
		item->set_custom_color(0, disabled_color);
		item->set_custom_color(1, disabled_color);
	}

	_match_item(item, p_doc->name);

	return item;
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_method_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const String &p_text, const DocData::MethodDoc *p_doc) {
	String tooltip = p_doc->return_type + " " + p_class_doc->name + "." + p_doc->name + "(";
	for (int i = 0; i < p_doc->arguments.size(); i++) {
		const DocData::ArgumentDoc &arg = p_doc->arguments[i];
		tooltip += arg.type + " " + arg.name;
		if (arg.default_value != "") {
			tooltip += " = " + arg.default_value;
		}
		if (i < p_doc->arguments.size() - 1) {
			tooltip += ", ";
		}
	}
	tooltip += ")";
	return _create_member_item(p_parent, p_class_doc->name, "MemberMethod", p_doc->name, p_text, TTRC("Method"), "method", tooltip, p_doc->description);
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_signal_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::MethodDoc *p_doc) {
	String tooltip = p_doc->return_type + " " + p_class_doc->name + "." + p_doc->name + "(";
	for (int i = 0; i < p_doc->arguments.size(); i++) {
		const DocData::ArgumentDoc &arg = p_doc->arguments[i];
		tooltip += arg.type + " " + arg.name;
		if (arg.default_value != "") {
			tooltip += " = " + arg.default_value;
		}
		if (i < p_doc->arguments.size() - 1) {
			tooltip += ", ";
		}
	}
	tooltip += ")";
	return _create_member_item(p_parent, p_class_doc->name, "MemberSignal", p_doc->name, p_doc->name, TTRC("Signal"), "signal", tooltip, p_doc->description);
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_constant_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::ConstantDoc *p_doc) {
	String tooltip = p_class_doc->name + "." + p_doc->name;
	return _create_member_item(p_parent, p_class_doc->name, "MemberConstant", p_doc->name, p_doc->name, TTRC("Constant"), "constant", tooltip, p_doc->description);
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_property_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::PropertyDoc *p_doc) {
	String tooltip = p_doc->type + " " + p_class_doc->name + "." + p_doc->name;
	tooltip += "\n    " + p_class_doc->name + "." + p_doc->setter + "(value) setter";
	tooltip += "\n    " + p_class_doc->name + "." + p_doc->getter + "() getter";
	return _create_member_item(p_parent, p_class_doc->name, "MemberProperty", p_doc->name, p_doc->name, TTRC("Property"), "property", tooltip, p_doc->description);
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_theme_property_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::ThemeItemDoc *p_doc) {
	String tooltip = p_doc->type + " " + p_class_doc->name + "." + p_doc->name;
	return _create_member_item(p_parent, p_class_doc->name, "MemberTheme", p_doc->name, p_doc->name, TTRC("Theme Property"), "theme_item", tooltip, p_doc->description);
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_member_item(TreeItem *p_parent, const String &p_class_name, const String &p_icon, const String &p_name, const String &p_text, const String &p_type, const String &p_metatype, const String &p_tooltip, const String &p_description) {
	Ref<Texture2D> icon;
	String text;
	if (search_flags & SEARCH_SHOW_HIERARCHY) {
		icon = ui_service->get_theme_icon(p_icon, SNAME("EditorIcons"));
		text = p_text;
	} else {
		icon = ui_service->get_theme_icon(p_icon, SNAME("EditorIcons"));
		text = p_class_name + "." + p_text;
	}

	TreeItem *item = results_tree->create_item(p_parent);
	item->set_icon(0, icon);
	item->set_text(0, text);
	item->set_text(1, TTRGET(p_type));
	item->set_tooltip(0, p_tooltip);
	item->set_tooltip(1, p_tooltip);
	item->set_metadata(0, p_class_name + ":" + p_name);
	item->set_metadata(1, "class_" + p_metatype);
	item->set_meta("description", p_description);

	_match_item(item, p_name);

	return item;
}

bool VisualScriptPropertySelector::SearchRunner::work(uint64_t slot) {
	// Return true when the search has been completed, otherwise false.
	const uint64_t until = OS::get_singleton()->get_ticks_usec() + slot;
	while (!_slice()) {
		if (OS::get_singleton()->get_ticks_usec() > until) {
			return false;
		}
	}
	return true;
}

VisualScriptPropertySelector::SearchRunner::SearchRunner(VisualScriptPropertySelector *p_selector_ui, Tree *p_results_tree) :
		selector_ui(p_selector_ui),
		ui_service(p_selector_ui->vbox),
		results_tree(p_results_tree),
		term(p_selector_ui->search_box->get_text()),
		empty_icon(ui_service->get_theme_icon(SNAME("ArrowRight"), SNAME("EditorIcons"))),
		disabled_color(ui_service->get_theme_color(SNAME("disabled_font_color"), SNAME("Editor"))) {
}
