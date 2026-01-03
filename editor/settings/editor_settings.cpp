/**************************************************************************/
/*  editor_settings.cpp                                                   */
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

#include "editor_settings.h"

#include "core/config/project_settings.h"
#include "core/input/input_event.h"
#include "core/input/input_map.h"
#include "core/input/shortcut.h"
#include "core/io/certs_compressed.gen.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/ip.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/object/class_db.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/translation_server.h"
#include "core/version.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_paths.h"
#include "editor/inspector/editor_property_name_processor.h"
#include "editor/project_manager/engine_update_label.h"
#include "editor/themes/editor_theme_manager.h"
#include "editor/translations/editor_translation.h"
#include "main/main.h"
#include "modules/regex/regex.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/file_dialog.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/animation.h"

// PRIVATE METHODS

Ref<EditorSettings> EditorSettings::singleton = nullptr;

// Properties

bool EditorSettings::_set(const StringName &p_name, const Variant &p_value) {
	_THREAD_SAFE_METHOD_

	bool changed = _set_only(p_name, p_value);
	if (changed && initialized) {
		changed_settings.insert(p_name);
		if (p_name == SNAME("text_editor/external/exec_path")) {
			const StringName exec_args_name = "text_editor/external/exec_flags";
			const String exec_args_value = _guess_exec_args_for_extenal_editor(p_value);
			if (!exec_args_value.is_empty() && _set_only(exec_args_name, exec_args_value)) {
				changed_settings.insert(exec_args_name);
			}
		}
		emit_signal(SNAME("settings_changed"));

		if (p_name == SNAME("interface/editor/editor_language")) {
			setup_language(false);
		}
	}
	return true;
}

bool EditorSettings::_set_only(const StringName &p_name, const Variant &p_value) {
	_THREAD_SAFE_METHOD_

	if (p_name == "shortcuts") {
		Array arr = p_value;
		for (int i = 0; i < arr.size(); i++) {
			Dictionary dict = arr[i];
			String shortcut_name = dict["name"];

			Array shortcut_events = dict["shortcuts"];

			Ref<Shortcut> sc;
			sc.instantiate();
			sc->set_events(shortcut_events);
			_add_shortcut_default(shortcut_name, sc);
		}

		return false;
	} else if (p_name == "builtin_action_overrides") {
		Array actions_arr = p_value;
		for (int i = 0; i < actions_arr.size(); i++) {
			Dictionary action_dict = actions_arr[i];

			String action_name = action_dict["name"];
			Array events = action_dict["events"];

			InputMap *im = InputMap::get_singleton();
			im->action_erase_events(action_name);

			builtin_action_overrides[action_name].clear();
			for (int ev_idx = 0; ev_idx < events.size(); ev_idx++) {
				im->action_add_event(action_name, events[ev_idx]);
				builtin_action_overrides[action_name].push_back(events[ev_idx]);
			}
		}
		return false;
	}

	bool changed = false;

	if (p_value.get_type() == Variant::NIL) {
		if (props.has(p_name)) {
			props.erase(p_name);
			changed = true;
		}
	} else {
		if (props.has(p_name)) {
			if (p_value != props[p_name].variant) {
				props[p_name].variant = p_value;
				changed = true;
			}
		} else {
			props[p_name] = VariantContainer(p_value, last_order++);
			changed = true;
		}

		if (save_changed_setting) {
			if (!props[p_name].save) {
				props[p_name].save = true;
				changed = true;
			}
		}
	}

	return changed;
}

bool EditorSettings::_get(const StringName &p_name, Variant &r_ret) const {
	_THREAD_SAFE_METHOD_

	if (p_name == "shortcuts") {
		Array save_array;
		const HashMap<String, List<Ref<InputEvent>>> &builtin_list = InputMap::get_singleton()->get_builtins();
		for (const KeyValue<String, Ref<Shortcut>> &shortcut_definition : shortcuts) {
			Ref<Shortcut> sc = shortcut_definition.value;

			if (builtin_list.has(shortcut_definition.key)) {
				// This shortcut was auto-generated from built in actions: don't save.
				// If the builtin is overridden, it will be saved in the "builtin_action_overrides" section below.
				continue;
			}

			Array shortcut_events = sc->get_events();

			Dictionary dict;
			dict["name"] = shortcut_definition.key;
			dict["shortcuts"] = shortcut_events;

			if (!sc->has_meta("original")) {
				// Getting the meta when it doesn't exist will return an empty array. If the 'shortcut_events' have been cleared,
				// we still want save the shortcut in this case so that shortcuts that the user has customized are not reset,
				// even if the 'original' has not been populated yet. This can happen when calling save() from the Project Manager.
				save_array.push_back(dict);
				continue;
			}

			Array original_events = sc->get_meta("original");

			bool is_same = Shortcut::is_event_array_equal(original_events, shortcut_events);
			if (is_same) {
				continue; // Not changed from default; don't save.
			}

			save_array.push_back(dict);
		}
		r_ret = save_array;
		return true;
	} else if (p_name == "builtin_action_overrides") {
		Array actions_arr;
		for (const KeyValue<String, List<Ref<InputEvent>>> &action_override : builtin_action_overrides) {
			const List<Ref<InputEvent>> *defaults = InputMap::get_singleton()->get_builtins().getptr(action_override.key);
			if (!defaults) {
				continue;
			}

			List<Ref<InputEvent>> events = action_override.value;

			Dictionary action_dict;
			action_dict["name"] = action_override.key;

			// Convert the list to an array, and only keep key events as this is for the editor.
			Array events_arr;
			for (const Ref<InputEvent> &ie : events) {
				Ref<InputEventKey> iek = ie;
				if (iek.is_valid()) {
					events_arr.append(iek);
				}
			}

			Array defaults_arr;
			for (const Ref<InputEvent> &default_input_event : *defaults) {
				if (default_input_event.is_valid()) {
					defaults_arr.append(default_input_event);
				}
			}

			bool same = Shortcut::is_event_array_equal(events_arr, defaults_arr);

			// Don't save if same as default.
			if (same) {
				continue;
			}

			action_dict["events"] = events_arr;
			actions_arr.push_back(action_dict);
		}

		r_ret = actions_arr;
		return true;
	}

	const VariantContainer *v = props.getptr(p_name);
	if (!v) {
		return false;
	}
	r_ret = v->variant;
	return true;
}

void EditorSettings::_initial_set(const StringName &p_name, const Variant &p_value, bool p_basic) {
	set(p_name, p_value);
	props[p_name].initial = p_value;
	props[p_name].has_default_value = true;
	props[p_name].basic = p_basic;
}

struct _EVCSort {
	String name;
	Variant::Type type = Variant::Type::NIL;
	int order = 0;
	bool basic = false;
	bool save = false;
	bool restart_if_changed = false;

	bool operator<(const _EVCSort &p_vcs) const { return order < p_vcs.order; }
};

void EditorSettings::_get_property_list(List<PropertyInfo> *p_list) const {
	_THREAD_SAFE_METHOD_

	RBSet<_EVCSort> vclist;

	for (const KeyValue<String, VariantContainer> &E : props) {
		const VariantContainer *v = &E.value;

		if (v->hide_from_editor) {
			continue;
		}

		_EVCSort vc;
		vc.name = E.key;
		vc.order = v->order;
		vc.type = v->variant.get_type();
		vc.basic = v->basic;
		vc.save = v->save;
		if (vc.save) {
			if (v->initial.get_type() != Variant::NIL && v->initial == v->variant) {
				vc.save = false;
			}
		}
		vc.restart_if_changed = v->restart_if_changed;

		vclist.insert(vc);
	}

	for (const _EVCSort &E : vclist) {
		uint32_t pusage = PROPERTY_USAGE_NONE;
		if (E.save || !optimize_save) {
			pusage |= PROPERTY_USAGE_STORAGE;
		}

		if (!E.name.begins_with("_") && !E.name.begins_with("projects/")) {
			pusage |= PROPERTY_USAGE_EDITOR;
		} else {
			pusage |= PROPERTY_USAGE_STORAGE; //hiddens must always be saved
		}

		PropertyInfo pi(E.type, E.name);
		pi.usage = pusage;
		if (hints.has(E.name)) {
			pi = hints[E.name];
		}

		if (E.basic) {
			pi.usage |= PROPERTY_USAGE_EDITOR_BASIC_SETTING;
		}

		if (E.restart_if_changed) {
			pi.usage |= PROPERTY_USAGE_RESTART_IF_CHANGED;
		}

		p_list->push_back(pi);
	}

	p_list->push_back(PropertyInfo(Variant::ARRAY, "shortcuts", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL)); //do not edit
	p_list->push_back(PropertyInfo(Variant::ARRAY, "builtin_action_overrides", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
}

void EditorSettings::_add_property_info_bind(const Dictionary &p_info) {
	ERR_FAIL_COND_MSG(!p_info.has("name"), "Property info is missing \"name\" field.");
	ERR_FAIL_COND_MSG(!p_info.has("type"), "Property info is missing \"type\" field.");

	if (p_info.has("usage")) {
		WARN_PRINT("\"usage\" is not supported in add_property_info().");
	}

	PropertyInfo pinfo;
	pinfo.name = p_info["name"];
	ERR_FAIL_COND(!props.has(pinfo.name));
	pinfo.type = Variant::Type(p_info["type"].operator int());
	ERR_FAIL_INDEX(pinfo.type, Variant::VARIANT_MAX);

	if (p_info.has("hint")) {
		pinfo.hint = PropertyHint(p_info["hint"].operator int());
	}
	if (p_info.has("hint_string")) {
		pinfo.hint_string = p_info["hint_string"];
	}

	add_property_hint(pinfo);
}

// Default configs
bool EditorSettings::has_default_value(const String &p_setting) const {
	_THREAD_SAFE_METHOD_

	if (!props.has(p_setting)) {
		return false;
	}
	return props[p_setting].has_default_value;
}

void EditorSettings::_set_initialized() {
	initialized = true;
}

static LocalVector<String> _get_skipped_locales() {
	// Skip locales if Text server lack required features.
	LocalVector<String> locales_to_skip;
	if (!TS->has_feature(TextServer::FEATURE_BIDI_LAYOUT) || !TS->has_feature(TextServer::FEATURE_SHAPING)) {
		locales_to_skip.push_back("ar"); // Arabic.
		locales_to_skip.push_back("fa"); // Persian.
		locales_to_skip.push_back("ur"); // Urdu.
	}
	if (!TS->has_feature(TextServer::FEATURE_BIDI_LAYOUT)) {
		locales_to_skip.push_back("he"); // Hebrew.
	}
	if (!TS->has_feature(TextServer::FEATURE_SHAPING)) {
		locales_to_skip.push_back("bn"); // Bengali.
		locales_to_skip.push_back("hi"); // Hindi.
		locales_to_skip.push_back("ml"); // Malayalam.
		locales_to_skip.push_back("si"); // Sinhala.
		locales_to_skip.push_back("ta"); // Tamil.
		locales_to_skip.push_back("te"); // Telugu.
	}
	return locales_to_skip;
}

void EditorSettings::_load_defaults(Ref<ConfigFile> p_extra_config) {
	_THREAD_SAFE_METHOD_
// Sets up the editor setting with a default value and hint PropertyInfo.
#define EDITOR_SETTING(m_type, m_property_hint, m_name, m_default_value, m_hint_string) \
	_initial_set(m_name, m_default_value);                                              \
	hints[m_name] = PropertyInfo(m_type, m_name, m_property_hint, m_hint_string);

#define EDITOR_SETTING_BASIC(m_type, m_property_hint, m_name, m_default_value, m_hint_string) \
	_initial_set(m_name, m_default_value, true);                                              \
	hints[m_name] = PropertyInfo(m_type, m_name, m_property_hint, m_hint_string);

#define EDITOR_SETTING_USAGE(m_type, m_property_hint, m_name, m_default_value, m_hint_string, m_usage) \
	_initial_set(m_name, m_default_value);                                                             \
	hints[m_name] = PropertyInfo(m_type, m_name, m_property_hint, m_hint_string, m_usage);

	/* Languages */

	{
		String lang_hint;
		const String host_lang = OS::get_singleton()->get_locale();

		// Skip locales which we can't render properly.
		const LocalVector<String> locales_to_skip = _get_skipped_locales();
		if (!locales_to_skip.is_empty()) {
			WARN_PRINT("Some locales are not properly supported by selected Text Server and are disabled.");
		}

		String best = "en";
		int best_score = 0;
		for (const String &locale : get_editor_locales()) {
			// Test against language code without regional variants (e.g. ur_PK).
			String lang_code = locale.get_slicec('_', 0);
			if (locales_to_skip.has(lang_code)) {
				continue;
			}

			lang_hint += ";";
			const String lang_name = TranslationServer::get_singleton()->get_locale_name(locale);
			lang_hint += vformat("%s/[%s] %s", locale, locale, lang_name);

			int score = TranslationServer::get_singleton()->compare_locales(host_lang, locale);
			if (score > 0 && score >= best_score) {
				best = locale;
				best_score = score;
			}
		}
		lang_hint = vformat(";auto/Auto (%s);en/[en] English", TranslationServer::get_singleton()->get_locale_name(best)) + lang_hint;

		EDITOR_SETTING_USAGE(Variant::STRING, PROPERTY_HINT_ENUM, "interface/editor/editor_language", "auto", lang_hint, PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED | PROPERTY_USAGE_EDITOR_BASIC_SETTING);
	}

	// Asset library
	_initial_set("asset_library/use_threads", true);

	/* Interface */

	// Editor
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/editor/localize_settings", true, "")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/dock_tab_style", 0, "Text Only,Icon Only,Text and Icon")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/bottom_dock_tab_style", 0, "Text Only,Icon Only,Text and Icon")
	EDITOR_SETTING_USAGE(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/ui_layout_direction", 0, "Based on Application Locale,Left-to-Right,Right-to-Left,Based on System Locale", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)

	// Display what the Auto display scale setting effectively corresponds to.
	const String display_scale_hint_string = vformat("Auto (%d%%),75%%,100%%,125%%,150%%,175%%,200%%,Custom", Math::round(get_auto_display_scale() * 100));
	EDITOR_SETTING_USAGE(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/display_scale", 0, display_scale_hint_string, PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED | PROPERTY_USAGE_EDITOR_BASIC_SETTING)
	EDITOR_SETTING_USAGE(Variant::FLOAT, PROPERTY_HINT_RANGE, "interface/editor/custom_display_scale", 1.0, "0.5,3,0.01", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED | PROPERTY_USAGE_EDITOR_BASIC_SETTING)

	String ed_screen_hints = "Auto (Remembers last position):-5,Screen With Mouse Pointer:-4,Screen With Keyboard Focus:-3,Primary Screen:-2";
	for (int i = 0; i < DisplayServer::get_singleton()->get_screen_count(); i++) {
		ed_screen_hints += ",Screen " + itos(i + 1) + ":" + itos(i);
	}
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/editor_screen", EditorSettings::InitialScreen::INITIAL_SCREEN_AUTO, ed_screen_hints)

#ifdef WINDOWS_ENABLED
	String tablet_hints = "Use Project Settings:-1";
	for (int i = 0; i < DisplayServer::get_singleton()->tablet_get_driver_count(); i++) {
		String drv_name = DisplayServer::get_singleton()->tablet_get_driver_name(i);
		if (EditorPropertyNameProcessor::get_singleton()) {
			drv_name = EditorPropertyNameProcessor::get_singleton()->process_name(drv_name, EditorPropertyNameProcessor::STYLE_CAPITALIZED); // Note: EditorPropertyNameProcessor is not available when doctool is used, but this value is not part of docs.
		}
		tablet_hints += vformat(",%s:%d", drv_name, i);
	}
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/tablet_driver", -1, tablet_hints);
#else
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/tablet_driver", -1, "Default:-1");
#endif

	String project_manager_screen_hints = "Screen With Mouse Pointer:-4,Screen With Keyboard Focus:-3,Primary Screen:-2";
	for (int i = 0; i < DisplayServer::get_singleton()->get_screen_count(); i++) {
		project_manager_screen_hints += ",Screen " + itos(i + 1) + ":" + itos(i);
	}
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/project_manager_screen", EditorSettings::InitialScreen::INITIAL_SCREEN_PRIMARY, project_manager_screen_hints)

	{
		EngineUpdateLabel::UpdateMode default_update_mode = EngineUpdateLabel::UpdateMode::NEWEST_UNSTABLE;
		if (String(GODOT_VERSION_STATUS) == String("stable")) {
			default_update_mode = EngineUpdateLabel::UpdateMode::NEWEST_STABLE;
		}
		EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "network/connection/check_for_updates", int(default_update_mode), "Disable Update Checks,Check Newest Preview,Check Newest Stable,Check Newest Patch"); // Uses EngineUpdateLabel::UpdateMode.
	}

	EDITOR_SETTING_USAGE(Variant::BOOL, PROPERTY_HINT_NONE, "interface/editor/use_embedded_menu", false, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_BASIC_SETTING)
	EDITOR_SETTING_USAGE(Variant::BOOL, PROPERTY_HINT_NONE, "interface/editor/use_native_file_dialogs", false, "", PROPERTY_USAGE_DEFAULT)
	EDITOR_SETTING_USAGE(Variant::BOOL, PROPERTY_HINT_NONE, "interface/editor/expand_to_title", true, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED | PROPERTY_USAGE_EDITOR_BASIC_SETTING)

	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_RANGE, "interface/editor/main_font_size", 14, "8,48,1")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_RANGE, "interface/editor/code_font_size", 14, "8,48,1")
	_initial_set("interface/editor/main_font_custom_opentype_features", "");
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/code_font_contextual_ligatures", 1, "Enabled,Disable Contextual Alternates (Coding Ligatures),Use Custom OpenType Feature Set")
	_initial_set("interface/editor/code_font_custom_opentype_features", "");
	_initial_set("interface/editor/code_font_custom_variations", "");
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/font_antialiasing", 1, "None,Grayscale,LCD Subpixel")
#ifdef MACOS_ENABLED
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/font_hinting", 0, "Auto (None),None,Light,Normal")
#else
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/font_hinting", 0, "Auto (Light),None,Light,Normal")
#endif
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/font_subpixel_positioning", 1, "Disabled,Auto,One Half of a Pixel,One Quarter of a Pixel")
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/editor/font_disable_embedded_bitmaps", true, "");
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/editor/font_allow_msdf", true, "")

	EDITOR_SETTING(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "interface/editor/main_font", "", "*.ttf,*.otf,*.woff,*.woff2,*.pfb,*.pfm")
	EDITOR_SETTING(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "interface/editor/main_font_bold", "", "*.ttf,*.otf,*.woff,*.woff2,*.pfb,*.pfm")
	EDITOR_SETTING(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "interface/editor/code_font", "", "*.ttf,*.otf,*.woff,*.woff2,*.pfb,*.pfm")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "interface/editor/dragging_hover_wait_seconds", 0.5, "0.01,10,0.01,or_greater,suffix:s");
	_initial_set("interface/editor/separate_distraction_mode", false, true);
	_initial_set("interface/editor/automatically_open_screenshots", true, true);
	EDITOR_SETTING_USAGE(Variant::BOOL, PROPERTY_HINT_NONE, "interface/editor/single_window_mode", false, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED | PROPERTY_USAGE_EDITOR_BASIC_SETTING)
	_initial_set("interface/editor/mouse_extra_buttons_navigate_history", true);
	_initial_set("interface/editor/save_each_scene_on_quit", true, true); // Regression
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "interface/editor/save_on_focus_loss", false, "")
	EDITOR_SETTING_USAGE(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/accept_dialog_cancel_ok_buttons", 0,
			vformat("Auto (%s),Cancel First,OK First", DisplayServer::get_singleton()->get_swap_cancel_ok() ? "OK First" : "Cancel First"),
			PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
#ifdef DEV_ENABLED
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/show_internal_errors_in_toast_notifications", 0, "Auto (Enabled),Enabled,Disabled")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/show_update_spinner", 0, "Auto (Enabled),Enabled,Disabled")
#else
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/show_internal_errors_in_toast_notifications", 0, "Auto (Disabled),Enabled,Disabled")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/show_update_spinner", 0, "Auto (Disabled),Enabled,Disabled")
#endif

	_initial_set("interface/editor/keep_screen_on", false, true);
	EDITOR_SETTING_USAGE(Variant::INT, PROPERTY_HINT_RANGE, "interface/editor/low_processor_mode_sleep_usec", 6900, "1,100000,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	// Default unfocused usec sleep is for 10 FPS. Allow an unfocused FPS limit
	// as low as 1 FPS for those who really need low power usage (but don't need
	// to preview particles or shaders while the editor is unfocused). With very
	// low FPS limits, the editor can take a small while to become usable after
	// being focused again, so this should be used at the user's discretion.
	EDITOR_SETTING_USAGE(Variant::INT, PROPERTY_HINT_RANGE, "interface/editor/unfocused_low_processor_mode_sleep_usec", 100000, "1,1000000,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)

	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "interface/editor/import_resources_when_unfocused", false, "")

	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "interface/editor/vsync_mode", 1, "Disabled,Enabled,Adaptive,Mailbox")
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/editor/update_continuously", false, "")

	bool is_android_editor = false;
#ifdef ANDROID_ENABLED
	if (!OS::get_singleton()->has_feature("xr_editor")) {
		is_android_editor = true;
	}
#endif
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/editor/collapse_main_menu", is_android_editor, "")

	_initial_set("interface/editors/show_scene_tree_root_selection", true);
	_initial_set("interface/editors/derive_script_globals_by_name", true);
	_initial_set("docks/scene_tree/ask_before_revoking_unique_name", true);

	// Inspector
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_RANGE, "interface/inspector/max_array_dictionary_items_per_page", 20, "10,100,1")
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/inspector/show_low_level_opentype_features", false, "")
	EDITOR_SETTING_BASIC(Variant::FLOAT, PROPERTY_HINT_RANGE, "interface/inspector/float_drag_speed", 5.0, "0.1,100,0.01")
	EDITOR_SETTING_BASIC(Variant::FLOAT, PROPERTY_HINT_RANGE, "interface/inspector/integer_drag_speed", 0.5, "0.1,10,0.01")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "interface/inspector/nested_color_mode", 0, "Containers & Resources,Resources,External Resources")
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/inspector/delimitate_all_container_and_resources", true, "")
	EDITOR_SETTING_USAGE(Variant::INT, PROPERTY_HINT_ENUM, "interface/inspector/default_property_name_style", EditorPropertyNameProcessor::STYLE_CAPITALIZED, "Raw (e.g. \"z_index\"),Capitalized (e.g. \"Z Index\"),Localized (e.g. \"Z Index\")", PROPERTY_USAGE_DEFAULT);
	// The lowest value is equal to the minimum float step for 32-bit floats.
	// The step must be set manually, as changing this setting should not change the step here.
	EDITOR_SETTING_USAGE(Variant::FLOAT, PROPERTY_HINT_RANGE, "interface/inspector/default_float_step", 0.001, "0.0000001,1,0.0000001", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_BASIC_SETTING);
	EDITOR_SETTING_USAGE(Variant::BOOL, PROPERTY_HINT_NONE, "interface/inspector/disable_folding", false, "", PROPERTY_USAGE_DEFAULT);
	EDITOR_SETTING_USAGE(Variant::BOOL, PROPERTY_HINT_NONE, "interface/inspector/auto_unfold_foreign_scenes", true, "", PROPERTY_USAGE_DEFAULT)
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/inspector/horizontal_vector2_editing", false, "")
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/inspector/horizontal_vector_types_editing", true, "")
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/inspector/open_resources_in_current_inspector", true, "")

	PackedStringArray open_in_new_inspector_defaults;
	// Required for the script editor to work.
	open_in_new_inspector_defaults.push_back("Script");
	// Required for the GridMap editor to work.
	open_in_new_inspector_defaults.push_back("MeshLibrary");
	_initial_set("interface/inspector/resources_to_open_in_new_inspector", open_in_new_inspector_defaults);

	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "interface/accessibility/accessibility_support", 0, "Auto (When Screen Reader is Running),Always Active,Disabled")
	set_restart_if_changed("interface/accessibility/accessibility_support", true);

	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "interface/inspector/default_color_picker_mode", (int32_t)ColorPicker::MODE_RGB, "RGB,HSV,RAW,OKHSL")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "interface/inspector/default_color_picker_shape", (int32_t)ColorPicker::SHAPE_VHS_CIRCLE, "HSV Rectangle,HSV Rectangle Wheel,VHS Circle,OKHSL Circle,OK HS Rectangle:5,OK HL Rectangle") // `SHAPE_NONE` is 4.
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "interface/inspector/color_picker_show_intensity", true, "");

	// Theme
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_ENUM, "interface/theme/follow_system_theme", false, "")
	EDITOR_SETTING_BASIC(Variant::STRING, PROPERTY_HINT_ENUM, "interface/theme/style", "Modern", "Modern,Classic")
	EDITOR_SETTING_BASIC(Variant::STRING, PROPERTY_HINT_ENUM, "interface/theme/color_preset", "Default", "Default,Breeze Dark,Godot 2,Godot 3,Gray,Light,Solarized (Dark),Solarized (Light),Black (OLED),Custom")
	EDITOR_SETTING_BASIC(Variant::STRING, PROPERTY_HINT_ENUM, "interface/theme/spacing_preset", "Default", "Compact,Default,Spacious,Custom")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/theme/icon_and_font_color", 0, "Auto,Dark,Light")
	EDITOR_SETTING_BASIC(Variant::COLOR, PROPERTY_HINT_NONE, "interface/theme/base_color", Color(0.14, 0.14, 0.14), "")
	EDITOR_SETTING_BASIC(Variant::COLOR, PROPERTY_HINT_NONE, "interface/theme/accent_color", Color(0.34, 0.62, 1.0), "")
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "interface/theme/use_system_accent_color", false, "")
	EDITOR_SETTING_BASIC(Variant::FLOAT, PROPERTY_HINT_RANGE, "interface/theme/contrast", 0.3, "-1,1,0.01")
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/theme/draw_extra_borders", false, "")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "interface/theme/icon_saturation", 2.0, "0,2,0.01")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/theme/draw_relationship_lines", (int32_t)EditorThemeManager::RELATIONSHIP_SELECTED_ONLY, "None,Selected Only,All")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "interface/theme/relationship_line_opacity", 0.1, "0.00,1,0.01")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "interface/theme/border_size", 0, "0,2,1")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "interface/theme/corner_radius", 4, "0,6,1")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "interface/theme/base_spacing", 4, "0,8,1")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "interface/theme/additional_spacing", 0, "0,8,1")
	EDITOR_SETTING_USAGE(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "interface/theme/custom_theme", "", "*.res,*.tres,*.theme", PROPERTY_USAGE_DEFAULT)

	// Touchscreen
	bool has_touchscreen_ui = DisplayServer::get_singleton()->is_touchscreen_available();
	bool is_native_touchscreen = has_touchscreen_ui && !OS::get_singleton()->has_feature("xr_editor"); // Disable some touchscreen settings by default for the XR Editor.

	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/touchscreen/enable_touch_optimizations", is_native_touchscreen, "")
	set_restart_if_changed("interface/touchscreen/enable_touch_optimizations", true);
	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/touchscreen/enable_long_press_as_right_click", is_native_touchscreen, "")
	set_restart_if_changed("interface/touchscreen/enable_long_press_as_right_click", true);

	EDITOR_SETTING(Variant::BOOL, PROPERTY_HINT_NONE, "interface/touchscreen/enable_pan_and_scale_gestures", has_touchscreen_ui, "")
	set_restart_if_changed("interface/touchscreen/enable_pan_and_scale_gestures", true);
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "interface/touchscreen/scale_gizmo_handles", has_touchscreen_ui ? 2 : 1, "1,5,1")
	set_restart_if_changed("interface/touchscreen/scale_gizmo_handles", true);

	// Only available in the Android/XR editor.
	String touch_actions_panel_hints = "Disabled:0,Embedded Panel:1,Floating Panel:2";
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "interface/touchscreen/touch_actions_panel", 1, touch_actions_panel_hints)

	// Scene tabs
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "interface/scene_tabs/display_close_button", 1, "Never,If Tab Active,Always"); // TabBar::CloseButtonDisplayPolicy
	_initial_set("interface/scene_tabs/show_thumbnail_on_hover", true);
	EDITOR_SETTING_USAGE(Variant::INT, PROPERTY_HINT_RANGE, "interface/scene_tabs/maximum_width", 350, "0,9999,1", PROPERTY_USAGE_DEFAULT)
	_initial_set("interface/scene_tabs/show_script_button", false, true);
	_initial_set("interface/scene_tabs/restore_scenes_on_load", true, true);
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "interface/scene_tabs/auto_select_current_scene_file", false, "");

	// Multi Window
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "interface/multi_window/enable", true, "");
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "interface/multi_window/restore_windows_on_load", true, "");
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "interface/multi_window/maximize_window", false, "");
	set_restart_if_changed("interface/multi_window/enable", true);

	/* Filesystem */

	// External Programs
	EDITOR_SETTING_BASIC(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "filesystem/external_programs/raster_image_editor", "", "")
	EDITOR_SETTING_BASIC(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "filesystem/external_programs/vector_image_editor", "", "")
	EDITOR_SETTING_BASIC(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "filesystem/external_programs/audio_editor", "", "")
	EDITOR_SETTING_BASIC(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "filesystem/external_programs/3d_model_editor", "", "")
	EDITOR_SETTING_BASIC(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "filesystem/external_programs/terminal_emulator", "", "")
	EDITOR_SETTING(Variant::STRING, PROPERTY_HINT_PLACEHOLDER_TEXT, "filesystem/external_programs/terminal_emulator_flags", "", "Call flags with placeholder: {directory}.");

	// Directories
	EDITOR_SETTING(Variant::STRING, PROPERTY_HINT_GLOBAL_DIR, "filesystem/directories/autoscan_project_path", "", "")
	const String fs_dir_default_project_path = OS::get_singleton()->has_environment("HOME") ? OS::get_singleton()->get_environment("HOME") : OS::get_singleton()->get_system_dir(OS::SYSTEM_DIR_DOCUMENTS);
	EDITOR_SETTING_BASIC(Variant::STRING, PROPERTY_HINT_GLOBAL_DIR, "filesystem/directories/default_project_path", fs_dir_default_project_path, "")

	// On save
	_initial_set("filesystem/on_save/compress_binary_resources", true);
	_initial_set("filesystem/on_save/safe_save_on_backup_then_rename", true);
	_initial_set("filesystem/on_save/warn_on_saving_large_text_resources", true);

	// EditorFileServer
	_initial_set("filesystem/file_server/port", 6010);
	_initial_set("filesystem/file_server/password", "");

	// File dialog
	_initial_set("filesystem/file_dialog/show_hidden_files", false);
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "filesystem/file_dialog/display_mode", 0, "Thumbnails,List")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "filesystem/file_dialog/thumbnail_size", 64, "32,128,16")

	// Quick Open dialog
	EDITOR_SETTING_USAGE(Variant::INT, PROPERTY_HINT_RANGE, "filesystem/quick_open_dialog/max_results", 100, "0,10000,1", PROPERTY_USAGE_DEFAULT)
	_initial_set("filesystem/quick_open_dialog/instant_preview", false);
	_initial_set("filesystem/quick_open_dialog/show_search_highlight", true);
	_initial_set("filesystem/quick_open_dialog/enable_fuzzy_matching", true);
	EDITOR_SETTING_USAGE(Variant::INT, PROPERTY_HINT_RANGE, "filesystem/quick_open_dialog/max_fuzzy_misses", 2, "0,10,1", PROPERTY_USAGE_DEFAULT)
	_initial_set("filesystem/quick_open_dialog/include_addons", false);
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "filesystem/quick_open_dialog/default_display_mode", 0, "Adaptive,Last Used")

	// Import (for glft module)
	EDITOR_SETTING_USAGE(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "filesystem/import/blender/blender_path", "", "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED | PROPERTY_USAGE_EDITOR_BASIC_SETTING)
	EDITOR_SETTING_USAGE(Variant::INT, PROPERTY_HINT_RANGE, "filesystem/import/blender/rpc_port", 6011, "0,65535,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::FLOAT, PROPERTY_HINT_RANGE, "filesystem/import/blender/rpc_server_uptime", 5, "0,300,1,or_greater,suffix:s", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "filesystem/import/fbx/fbx2gltf_path", "", "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)

	// Tools (denoise)
	EDITOR_SETTING_USAGE(Variant::STRING, PROPERTY_HINT_GLOBAL_DIR, "filesystem/tools/oidn/oidn_denoise_path", "", "", PROPERTY_USAGE_DEFAULT)

	/* Docks */

	// SceneTree
	_initial_set("docks/scene_tree/ask_before_deleting_related_animation_tracks", true);
	_initial_set("docks/scene_tree/start_create_dialog_fully_expanded", false);
	_initial_set("docks/scene_tree/auto_expand_to_selected", true);
	_initial_set("docks/scene_tree/center_node_on_reparent", false);
	_initial_set("docks/scene_tree/hide_filtered_out_parents", true);
	_initial_set("docks/scene_tree/accessibility_warnings", false);

	// FileSystem
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "docks/filesystem/thumbnail_size", 64, "32,128,16")
	_initial_set("docks/filesystem/always_show_folders", true);
	_initial_set("docks/filesystem/textfile_extensions", "txt,md,cfg,ini,log,json,yml,yaml,toml,xml");
	_initial_set("docks/filesystem/other_file_extensions", "ico,icns");

	// Property editor
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "docks/property_editor/auto_refresh_interval", 0.2, "0.01,1,0.001"); // Update 5 times per second by default.
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "docks/property_editor/subresource_hue_tint", 0.75, "0,1,0.01")

	/* Text editor */

	// Theme
	EDITOR_SETTING_BASIC(Variant::STRING, PROPERTY_HINT_ENUM, "text_editor/theme/color_theme", "Default", "Default,Godot 2,Custom")

	// Theme: Highlighting
	const LocalVector<StringName> basic_text_editor_settings = {
		"text_editor/theme/highlighting/symbol_color",
		"text_editor/theme/highlighting/keyword_color",
		"text_editor/theme/highlighting/control_flow_keyword_color",
		"text_editor/theme/highlighting/base_type_color",
		"text_editor/theme/highlighting/engine_type_color",
		"text_editor/theme/highlighting/user_type_color",
		"text_editor/theme/highlighting/comment_color",
		"text_editor/theme/highlighting/doc_comment_color",
		"text_editor/theme/highlighting/string_color",
		"text_editor/theme/highlighting/string_placeholder_color",
		"text_editor/theme/highlighting/background_color",
		"text_editor/theme/highlighting/text_color",
		"text_editor/theme/highlighting/line_number_color",
		"text_editor/theme/highlighting/safe_line_number_color",
		"text_editor/theme/highlighting/caret_color",
		"text_editor/theme/highlighting/caret_background_color",
		"text_editor/theme/highlighting/text_selected_color",
		"text_editor/theme/highlighting/selection_color",
		"text_editor/theme/highlighting/brace_mismatch_color",
		"text_editor/theme/highlighting/current_line_color",
		"text_editor/theme/highlighting/line_length_guideline_color",
		"text_editor/theme/highlighting/word_highlighted_color",
		"text_editor/theme/highlighting/number_color",
		"text_editor/theme/highlighting/function_color",
		"text_editor/theme/highlighting/member_variable_color",
		"text_editor/theme/highlighting/mark_color",
	};
	// These values will be overwritten by EditorThemeManager, but can still be seen in some edge cases.
	const HashMap<StringName, Color> text_colors = get_godot2_text_editor_theme();
	for (const KeyValue<StringName, Color> &text_color : text_colors) {
		if (basic_text_editor_settings.has(text_color.key)) {
			EDITOR_SETTING_BASIC(Variant::COLOR, PROPERTY_HINT_NONE, text_color.key, text_color.value, "")
		} else {
			EDITOR_SETTING(Variant::COLOR, PROPERTY_HINT_NONE, text_color.key, text_color.value, "")
		}
	}

	// The list is based on <https://github.com/KDE/syntax-highlighting/blob/master/data/syntax/alert.xml>.
	_initial_set("text_editor/theme/highlighting/comment_markers/critical_list", "ALERT,ATTENTION,CAUTION,CRITICAL,DANGER,SECURITY");
	_initial_set("text_editor/theme/highlighting/comment_markers/warning_list", "BUG,DEPRECATED,FIXME,HACK,TASK,TBD,TODO,WARNING");
	_initial_set("text_editor/theme/highlighting/comment_markers/notice_list", "INFO,NOTE,NOTICE,TEST,TESTING");

	// Appearance
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "text_editor/appearance/enable_inline_color_picker", true, "");

	// Appearance: Caret
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "text_editor/appearance/caret/type", 0, "Line,Block")
	_initial_set("text_editor/appearance/caret/caret_blink", true, true);
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "text_editor/appearance/caret/caret_blink_interval", 0.5, "0.1,10,0.01")
	_initial_set("text_editor/appearance/caret/highlight_current_line", true, true);
	_initial_set("text_editor/appearance/caret/highlight_all_occurrences", true, true);

	// Appearance: Guidelines
	_initial_set("text_editor/appearance/guidelines/show_line_length_guidelines", true, true);
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "text_editor/appearance/guidelines/line_length_guideline_soft_column", 80, "20,160,1")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "text_editor/appearance/guidelines/line_length_guideline_hard_column", 100, "20,160,1")

	// Appearance: Gutters
	_initial_set("text_editor/appearance/gutters/show_line_numbers", true, true);
	_initial_set("text_editor/appearance/gutters/line_numbers_zero_padded", false, true);
	_initial_set("text_editor/appearance/gutters/highlight_type_safe_lines", true, true);
	_initial_set("text_editor/appearance/gutters/show_info_gutter", true, true);

	// Appearance: Minimap
	_initial_set("text_editor/appearance/minimap/show_minimap", true, true);
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "text_editor/appearance/minimap/minimap_width", 80, "50,250,1")

	// Appearance: Lines
	_initial_set("text_editor/appearance/lines/code_folding", true, true);
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "text_editor/appearance/lines/word_wrap", 0, "None,Boundary")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "text_editor/appearance/lines/autowrap_mode", 3, "Arbitrary:1,Word:2,Word (Smart):3")

	// Appearance: Whitespace
	_initial_set("text_editor/appearance/whitespace/draw_tabs", true, true);
	_initial_set("text_editor/appearance/whitespace/draw_spaces", false, true);
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "text_editor/appearance/whitespace/line_spacing", 4, "0,50,1")

	// Behavior
	// Behavior: General
	_initial_set("text_editor/behavior/general/empty_selection_clipboard", true);

	// Behavior: Navigation
	_initial_set("text_editor/behavior/navigation/move_caret_on_right_click", true, true);
	_initial_set("text_editor/behavior/navigation/scroll_past_end_of_file", false, true);
	_initial_set("text_editor/behavior/navigation/smooth_scrolling", true, true);
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "text_editor/behavior/navigation/v_scroll_speed", 80, "1,10000,1")
	_initial_set("text_editor/behavior/navigation/drag_and_drop_selection", true, true);
	_initial_set("text_editor/behavior/navigation/stay_in_script_editor_on_node_selected", true, true);
	_initial_set("text_editor/behavior/navigation/open_script_when_connecting_signal_to_existing_method", true, true);
	_initial_set("text_editor/behavior/navigation/use_default_word_separators", true); // Includes ´`~$^=+|<> General punctuation and CJK punctuation.
	_initial_set("text_editor/behavior/navigation/use_custom_word_separators", false);
	_initial_set("text_editor/behavior/navigation/custom_word_separators", ""); // Custom word separators.

	// Behavior: Indent
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "text_editor/behavior/indent/type", 0, "Tabs,Spaces")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_RANGE, "text_editor/behavior/indent/size", 4, "1,64,1") // size of 0 crashes.
	_initial_set("text_editor/behavior/indent/auto_indent", true);
	_initial_set("text_editor/behavior/indent/indent_wrapped_lines", true);

	// Behavior: Files
	_initial_set("text_editor/behavior/files/trim_trailing_whitespace_on_save", false);
	_initial_set("text_editor/behavior/files/trim_final_newlines_on_save", true);
	_initial_set("text_editor/behavior/files/autosave_interval_secs", 0);
	_initial_set("text_editor/behavior/files/restore_scripts_on_load", true);
	_initial_set("text_editor/behavior/files/convert_indent_on_save", true);
	_initial_set("text_editor/behavior/files/auto_reload_scripts_on_external_change", true);
	_initial_set("text_editor/behavior/files/auto_reload_and_parse_scripts_on_save", true);
	_initial_set("text_editor/behavior/files/open_dominant_script_on_scene_change", false, true);
	_initial_set("text_editor/behavior/files/drop_preload_resources_as_uid", true, true);

	// Behavior: Documentation
	_initial_set("text_editor/behavior/documentation/enable_tooltips", true, true);

	// Script list
	_initial_set("text_editor/script_list/show_members_overview", true, true);
	_initial_set("text_editor/script_list/sort_members_outline_alphabetically", false, true);
	_initial_set("text_editor/script_list/script_temperature_enabled", true);
	_initial_set("text_editor/script_list/script_temperature_history_size", 15);
	_initial_set("text_editor/script_list/highlight_scene_scripts", true);
	_initial_set("text_editor/script_list/group_help_pages", true);
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "text_editor/script_list/sort_scripts_by", 0, "None:2,Name:0,Path:1");
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "text_editor/script_list/list_script_names_as", 0, "Name,Parent Directory And Name,Full Path");
	EDITOR_SETTING(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "text_editor/external/exec_path", "", "");
	EDITOR_SETTING(Variant::STRING, PROPERTY_HINT_PLACEHOLDER_TEXT, "text_editor/external/exec_flags", "{file}", "Call flags with placeholders: {project}, {file}, {col}, {line}.");

	// Completion
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "text_editor/completion/idle_parse_delay", 1.5, "0.1,10,0.01")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "text_editor/completion/idle_parse_delay_with_errors_found", 0.5, "0.1,5,0.01")
	_initial_set("text_editor/completion/auto_brace_complete", true, true);
	_initial_set("text_editor/completion/code_complete_enabled", true, true);
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "text_editor/completion/code_complete_delay", 0.3, "0.01,5,0.01,or_greater")
	_initial_set("text_editor/completion/put_callhint_tooltip_below_current_line", true);
	_initial_set("text_editor/completion/complete_file_paths", true);
	_initial_set("text_editor/completion/add_type_hints", true, true);
	_initial_set("text_editor/completion/add_string_name_literals", false, true);
	_initial_set("text_editor/completion/add_node_path_literals", false, true);
	_initial_set("text_editor/completion/use_single_quotes", false, true);
	_initial_set("text_editor/completion/colorize_suggestions", true);

	// External editor (ScriptEditorPlugin)
	_initial_set("text_editor/external/use_external_editor", false, true);
	_initial_set("text_editor/external/exec_path", "");

	// Help
	_initial_set("text_editor/help/show_help_index", true);
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_RANGE, "text_editor/help/help_font_size", 16, "8,48,1")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_RANGE, "text_editor/help/help_source_font_size", 15, "8,48,1")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_RANGE, "text_editor/help/help_title_font_size", 23, "8,64,1")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "text_editor/help/class_reference_examples", 0, "GDScript,C#,GDScript and C#")
	_initial_set("text_editor/help/sort_functions_alphabetically", true);

	/* Editors */

	// GridMap
	// GridMapEditor
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/grid_map/pick_distance", 5000.0, "1,8192,0.1,or_greater");
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_RANGE, "editors/grid_map/preview_size", 64, "16,128,1")

	// 3D
	EDITOR_SETTING_BASIC(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d/primary_grid_color", Color(0.56, 0.56, 0.56, 0.5), "")
	EDITOR_SETTING_BASIC(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d/secondary_grid_color", Color(0.38, 0.38, 0.38, 0.5), "")

	// Use a similar color to the 2D editor selection.
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d/selection_box_color", Color(1.0, 0.5, 0), "", PROPERTY_USAGE_DEFAULT)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d/active_selection_box_color", Color(1.5, 0.75, 0, 1.0), "", PROPERTY_USAGE_DEFAULT)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/instantiated", Color(0.7, 0.7, 0.7, 0.6), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/joint", Color(0.5, 0.8, 1), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/aabb", Color(0.28, 0.8, 0.82), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/stream_player_3d", Color(0.4, 0.8, 1), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/camera", Color(0.8, 0.4, 0.8), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/decal", Color(0.6, 0.5, 1.0), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/fog_volume", Color(0.5, 0.7, 1), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/particles", Color(0.8, 0.7, 0.4), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/particle_attractor", Color(1, 0.7, 0.5), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/particle_collision", Color(0.5, 0.7, 1), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/joint_body_a", Color(0.6, 0.8, 1), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/joint_body_b", Color(0.6, 0.9, 1), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/lightmap_lines", Color(0.5, 0.6, 1), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/lightprobe_lines", Color(0.5, 0.6, 1), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/occluder", Color(0.8, 0.5, 1), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/reflection_probe", Color(0.6, 1, 0.5), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/visibility_notifier", Color(0.8, 0.5, 0.7), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/voxel_gi", Color(0.5, 1, 0.6), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/path_tilt", Color(1.0, 1.0, 0.4, 0.9), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/skeleton", Color(1, 0.8, 0.4), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/selected_bone", Color(0.8, 0.3, 0.0), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/csg", Color(0.0, 0.4, 1, 0.15), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/gridmap_grid", Color(0.8, 0.5, 0.1), "")
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/spring_bone_joint", Color(0.8, 0.9, 0.6), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/spring_bone_collision", Color(0.6, 0.8, 0.9), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/spring_bone_inside_collision", Color(0.9, 0.6, 0.8), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	EDITOR_SETTING_USAGE(Variant::COLOR, PROPERTY_HINT_NONE, "editors/3d_gizmos/gizmo_colors/ik_chain", Color(0.6, 0.9, 0.8), "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)
	_initial_set("editors/3d_gizmos/gizmo_settings/bone_axis_length", (float)0.1);
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "editors/3d_gizmos/gizmo_settings/bone_shape", 1, "Wire,Octahedron");
	EDITOR_SETTING_USAGE(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d_gizmos/gizmo_settings/path3d_tilt_disk_size", 0.8, "0.01,4.0,0.001,or_greater", PROPERTY_USAGE_DEFAULT)
	EDITOR_SETTING_USAGE(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d_gizmos/gizmo_settings/lightmap_gi_probe_size", 0.4, "0.0,1.0,0.001,or_greater", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED)

	// If a line is a multiple of this, it uses the primary grid color.
	// Use a power of 2 value by default as it's more common to use powers of 2 in level design.
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_RANGE, "editors/3d/primary_grid_steps", 8, "1,100,1")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_RANGE, "editors/3d/grid_size", 200, "1,2000,1")
	// Higher values produce graphical artifacts when far away unless View Z-Far
	// is increased significantly more than it really should need to be.
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "editors/3d/grid_division_level_max", 2, "-1,3,1")
	// Lower values produce graphical artifacts regardless of view clipping planes, so limit to -2 as a lower bound.
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "editors/3d/grid_division_level_min", 0, "-2,2,1")
	// -0.2 seems like a sensible default. -1.0 gives Blender-like behavior, 0.5 gives huge grids.
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/grid_division_level_bias", -0.2, "-1.0,0.5,0.1")

	_initial_set("editors/3d/grid_xz_plane", true);
	_initial_set("editors/3d/grid_xy_plane", false);
	_initial_set("editors/3d/grid_yz_plane", false);

	// Use a lower default FOV for the 3D camera compared to the
	// Camera3D node as the 3D viewport doesn't span the whole screen.
	// This means it's technically viewed from a further distance, which warrants a narrower FOV.
	EDITOR_SETTING_BASIC(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/default_fov", 70.0, "1,179,0.1,degrees")
	EDITOR_SETTING_BASIC(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/default_z_near", 0.05, "0.01,10,0.01,or_greater,suffix:m")
	EDITOR_SETTING_BASIC(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/default_z_far", 4000.0, "0.1,4000,0.1,or_greater,suffix:m")

	// 3D: Navigation
	_initial_set("editors/3d/navigation/invert_x_axis", false, true);
	_initial_set("editors/3d/navigation/invert_y_axis", false, true);
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "editors/3d/navigation/navigation_scheme", 0, "Godot:0,Maya:1,Modo:2,Tablet/Trackpad:4,Custom:3")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "editors/3d/navigation/orbit_mouse_button", 1, "Left Mouse,Middle Mouse,Right Mouse,Mouse Button 4,Mouse Button 5")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "editors/3d/navigation/pan_mouse_button", 1, "Left Mouse,Middle Mouse,Right Mouse,Mouse Button 4,Mouse Button 5")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "editors/3d/navigation/zoom_mouse_button", 1, "Left Mouse,Middle Mouse,Right Mouse,Mouse Button 4,Mouse Button 5")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "editors/3d/navigation/zoom_style", 0, "Vertical,Horizontal")

	_initial_set("editors/3d/navigation/emulate_numpad", true, true);
	_initial_set("editors/3d/navigation/emulate_3_button_mouse", false, true);
	_initial_set("editors/3d/navigation/warped_mouse_panning", true, true);

	// 3D: Navigation feel
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/navigation_feel/orbit_sensitivity", 0.25, "0.01,20,0.001")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/navigation_feel/translation_sensitivity", 1.0, "0.01,20,0.001")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/navigation_feel/orbit_inertia", 0.0, "0,1,0.001")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/navigation_feel/translation_inertia", 0.05, "0,1,0.001")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/navigation_feel/zoom_inertia", 0.05, "0,1,0.001")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/navigation_feel/angle_snap_threshold", 10.0, "1,20,0.1,degrees")
	_initial_set("editors/3d/navigation/show_viewport_rotation_gizmo", true);
	_initial_set("editors/3d/navigation/show_viewport_navigation_gizmo", DisplayServer::get_singleton()->is_touchscreen_available());

	// 3D: Freelook
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "editors/3d/freelook/freelook_navigation_scheme", 0, "Default,Partially Axis-Locked (id Tech),Fully Axis-Locked (Minecraft)")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/freelook/freelook_sensitivity", 0.25, "0.01,2,0.001")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/freelook/freelook_inertia", 0.0, "0,1,0.001")
	EDITOR_SETTING_BASIC(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/freelook/freelook_base_speed", 5.0, "0,10,0.01,or_greater")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "editors/3d/freelook/freelook_activation_modifier", 0, "None,Shift,Alt,Meta,Ctrl")
	_initial_set("editors/3d/freelook/freelook_speed_zoom_link", false);

	// 3D: Manipulator
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "editors/3d/manipulator_gizmo_size", 80, "16,160,1");
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/3d/manipulator_gizmo_opacity", 0.9, "0,1,0.01")

	// 2D
	_initial_set("editors/2d/grid_color", Color(1.0, 1.0, 1.0, 0.07), true);
	_initial_set("editors/2d/guides_color", Color(0.6, 0.0, 0.8), true);
	_initial_set("editors/2d/smart_snapping_line_color", Color(0.9, 0.1, 0.1), true);
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/2d/bone_width", 5.0, "0.01,20,0.01,or_greater")
	_initial_set("editors/2d/bone_color1", Color(1.0, 1.0, 1.0, 0.7));
	_initial_set("editors/2d/bone_color2", Color(0.6, 0.6, 0.6, 0.7));
	_initial_set("editors/2d/bone_selected_color", Color(0.9, 0.45, 0.45, 0.7));
	_initial_set("editors/2d/bone_ik_color", Color(0.9, 0.9, 0.45, 0.7));
	_initial_set("editors/2d/bone_outline_color", Color(0.35, 0.35, 0.35, 0.5));
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/2d/bone_outline_size", 2.0, "0.01,8,0.01,or_greater")
	_initial_set("editors/2d/viewport_border_color", Color(0.4, 0.4, 1.0, 0.4), true);
	_initial_set("editors/2d/use_integer_zoom_by_default", false, true);
	EDITOR_SETTING_BASIC(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/2d/zoom_speed_factor", 1.1, "1.01,2,0.01")
	EDITOR_SETTING_BASIC(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/2d/ruler_width", 16.0, "12.0,30.0,1.0")

	// Bone mapper (BoneMapEditorPlugin)
	_initial_set("editors/bone_mapper/handle_colors/unset", Color(0.3, 0.3, 0.3));
	_initial_set("editors/bone_mapper/handle_colors/set", Color(0.1, 0.6, 0.25));
	_initial_set("editors/bone_mapper/handle_colors/missing", Color(0.8, 0.2, 0.8));
	_initial_set("editors/bone_mapper/handle_colors/error", Color(0.8, 0.2, 0.2));

	// Panning
	// Enum should be in sync with ControlScheme in ViewPanner.
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "editors/panning/2d_editor_panning_scheme", 0, "Scroll Zooms,Scroll Pans");
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "editors/panning/sub_editors_panning_scheme", 0, "Scroll Zooms,Scroll Pans");
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "editors/panning/animation_editors_panning_scheme", 1, "Scroll Zooms,Scroll Pans");
	_initial_set("editors/panning/simple_panning", false);
	_initial_set("editors/panning/warped_mouse_panning", true);
	_initial_set("editors/panning/2d_editor_pan_speed", 20, true);
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "editors/panning/zoom_style", 0, "Vertical,Horizontal");

	// Tiles editor
	_initial_set("editors/tiles_editor/display_grid", true);
	_initial_set("editors/tiles_editor/highlight_selected_layer", true);
	_initial_set("editors/tiles_editor/grid_color", Color(1.0, 0.5, 0.2, 0.5));

	// Polygon editor
	_initial_set("editors/polygon_editor/point_grab_radius", has_touchscreen_ui ? 32 : 8);
	_initial_set("editors/polygon_editor/show_previous_outline", true);
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/polygon_editor/auto_bake_delay", 1.5, "-1.0,10.0,0.01");

	// Animation
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/animation/default_animation_step", Animation::DEFAULT_STEP, "0.0,10.0,0.00000001");
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "editors/animation/default_fps_mode", 0, "Seconds,FPS");
	_initial_set("editors/animation/default_fps_compatibility", true);
	_initial_set("editors/animation/autorename_animation_tracks", true);
	_initial_set("editors/animation/confirm_insert_track", true, true);
	_initial_set("editors/animation/default_create_bezier_tracks", false, true);
	_initial_set("editors/animation/default_create_reset_tracks", true, true);
	_initial_set("editors/animation/insert_at_current_time", false, true);
	_initial_set("editors/animation/onion_layers_past_color", Color(1, 0, 0));
	_initial_set("editors/animation/onion_layers_future_color", Color(0, 1, 0));

	// Shader editor
	_initial_set("editors/shader_editor/behavior/files/restore_shaders_on_load", true, true);

	// Visual editors
	EDITOR_SETTING(Variant::STRING, PROPERTY_HINT_ENUM, "editors/visual_editors/color_theme", "Default", "Default,Legacy,Custom")

	_load_default_visual_shader_editor_theme();

	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/visual_editors/minimap_opacity", 0.85, "0.0,1.0,0.01")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "editors/visual_editors/lines_curvature", 0.5, "0.0,1.0,0.01")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "editors/visual_editors/grid_pattern", 1, "Lines,Dots")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "editors/visual_editors/visual_shader/port_preview_size", 160, "100,400,0.01")

	// Export (EditorExportPlugin)
	_initial_set("export/ssh/ssh", "");
	_initial_set("export/ssh/scp", "");

	/* Run */

	// Window placement
#ifndef ANDROID_ENABLED
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "run/window_placement/rect", 1, "Top Left,Centered,Custom Position,Force Maximized,Force Fullscreen")
	// Keep the enum values in sync with the `DisplayServer::SCREEN_` enum.
	String screen_hints = "Same as Editor:-5,Previous Screen:-4,Next Screen:-3,Primary Screen:-2"; // Note: Main Window Screen:-1 is not used for the main window.
	for (int i = 0; i < DisplayServer::get_singleton()->get_screen_count(); i++) {
		screen_hints += ",Screen " + itos(i + 1) + ":" + itos(i);
	}
	_initial_set("run/window_placement/rect_custom_position", Vector2());
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "run/window_placement/screen", -5, screen_hints)
#endif
	// Should match the ANDROID_WINDOW_* constants in 'platform/android/java/editor/src/main/java/org/godotengine/editor/BaseGodotEditor.kt'.
	String android_window_hints = "Auto (based on screen size):0,Same as Editor:1,Side-by-side with Editor:2";
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "run/window_placement/android_window", 0, android_window_hints)

	String game_embed_mode_hints = "Disabled:-1,Use Per-Project Configuration:0,Embed Game:1,Make Game Workspace Floating:2";
#ifdef ANDROID_ENABLED
	if (OS::get_singleton()->has_feature("xr_editor")) {
		game_embed_mode_hints = "Disabled:-1";
	} else {
		game_embed_mode_hints = "Disabled:-1,Auto (based on screen size):0,Enabled:1";
	}
#endif
	int default_game_embed_mode = OS::get_singleton()->has_feature("xr_editor") ? -1 : 0;
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "run/window_placement/game_embed_mode", default_game_embed_mode, game_embed_mode_hints);

	// Auto save
	_initial_set("run/auto_save/save_before_running", true, true);

	// Bottom panel
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "run/bottom_panel/action_on_play", EditorNode::ACTION_ON_PLAY_OPEN_OUTPUT, "Do Nothing,Open Output,Open Debugger")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "run/bottom_panel/action_on_stop", EditorNode::ACTION_ON_STOP_DO_NOTHING, "Do Nothing,Close Bottom Panel")

	// Output
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_RANGE, "run/output/font_size", 13, "8,48,1")
	_initial_set("run/output/always_clear_output_on_play", true, true);

	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "run/output/max_lines", 10000, "100,100000,1")

	// Platform
	_initial_set("run/platforms/linuxbsd/prefer_wayland", false, true);
	set_restart_if_changed("run/platforms/linuxbsd/prefer_wayland", true);

	/* Network */

	// General
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "network/connection/network_mode", 0, "Offline,Online");

	// HTTP Proxy
	_initial_set("network/http_proxy/host", "");
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "network/http_proxy/port", 8080, "1,65535,1")

	// SSL
	EDITOR_SETTING_USAGE(Variant::STRING, PROPERTY_HINT_GLOBAL_FILE, "network/tls/editor_tls_certificates", _SYSTEM_CERTS_PATH, "*.crt,*.pem", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "network/tls/enable_tls_v1.3", true, "")

	// Debug
	_initial_set("network/debug/remote_host", "127.0.0.1"); // Hints provided in setup_network
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "network/debug/remote_port", 6007, "1,65535,1")

	/* Debugger/profiler */

	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "debugger/auto_switch_to_remote_scene_tree", false, "")
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "debugger/auto_switch_to_stack_trace", true, "")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "debugger/max_node_selection", 20, "1,100,1")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "debugger/profiler_frame_history_size", 3600, "60,10000,1")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "debugger/profiler_frame_max_functions", 64, "16,512,1")
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_RANGE, "debugger/profiler_target_fps", 60, "1,1000,1")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "debugger/remote_scene_tree_refresh_interval", 1.0, "0.1,10,0.01,or_greater")
	EDITOR_SETTING(Variant::FLOAT, PROPERTY_HINT_RANGE, "debugger/remote_inspect_refresh_interval", 0.2, "0.02,10,0.01,or_greater")
	EDITOR_SETTING_BASIC(Variant::BOOL, PROPERTY_HINT_NONE, "debugger/profile_native_calls", false, "")

	// Version control (VersionControlEditorPlugin)
	_initial_set("version_control/username", "", true);
	_initial_set("version_control/ssh_public_key_path", "");
	_initial_set("version_control/ssh_private_key_path", "");

	/* Extra config */

	EDITOR_SETTING_USAGE(Variant::BOOL, PROPERTY_HINT_NONE, "input/buffering/agile_event_flushing", false, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED | PROPERTY_USAGE_EDITOR_BASIC_SETTING)
	EDITOR_SETTING_USAGE(Variant::BOOL, PROPERTY_HINT_NONE, "input/buffering/use_accumulated_input", true, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED | PROPERTY_USAGE_EDITOR_BASIC_SETTING)

	// TRANSLATORS: Project Manager here refers to the tool used to create/manage Godot projects.
	EDITOR_SETTING(Variant::INT, PROPERTY_HINT_ENUM, "project_manager/sorting_order", 0, "Last Edited,Name,Path")
	EDITOR_SETTING_BASIC(Variant::INT, PROPERTY_HINT_ENUM, "project_manager/directory_naming_convention", 1, "No Convention,kebab-case,snake_case,camelCase,PascalCase,Title Case")

#if defined(WEB_ENABLED)
	// Web platform only supports `gl_compatibility`.
	const String default_renderer = "gl_compatibility";
#elif defined(ANDROID_ENABLED)
	// Use more suitable rendering method by default.
	const String default_renderer = "mobile";
#else
	const String default_renderer = "forward_plus";
#endif
	EDITOR_SETTING_BASIC(Variant::STRING, PROPERTY_HINT_ENUM, "project_manager/default_renderer", default_renderer, "forward_plus,mobile,gl_compatibility")

#undef EDITOR_SETTING
#undef EDITOR_SETTING_BASIC
#undef EDITOR_SETTING_USAGE

	if (p_extra_config.is_valid()) {
		if (p_extra_config->has_section("init_projects") && p_extra_config->has_section_key("init_projects", "list")) {
			Vector<String> list = p_extra_config->get_value("init_projects", "list");
			for (int i = 0; i < list.size(); i++) {
				String proj_name = list[i].replace("/", "::");
				set("projects/" + proj_name, list[i]);
			}
		}

		if (p_extra_config->has_section("presets")) {
			Vector<String> keys = p_extra_config->get_section_keys("presets");

			for (const String &key : keys) {
				Variant val = p_extra_config->get_value("presets", key);
				set(key, val);
			}
		}
	}
}

void EditorSettings::_load_default_visual_shader_editor_theme() {
	// Connection type colors
	_initial_set("editors/visual_editors/connection_colors/scalar_color", Color(0.55, 0.55, 0.55));
	_initial_set("editors/visual_editors/connection_colors/vector2_color", Color(0.44, 0.43, 0.64));
	_initial_set("editors/visual_editors/connection_colors/vector3_color", Color(0.337, 0.314, 0.71));
	_initial_set("editors/visual_editors/connection_colors/vector4_color", Color(0.7, 0.65, 0.147));
	_initial_set("editors/visual_editors/connection_colors/boolean_color", Color(0.243, 0.612, 0.349));
	_initial_set("editors/visual_editors/connection_colors/transform_color", Color(0.71, 0.357, 0.64));
	_initial_set("editors/visual_editors/connection_colors/sampler_color", Color(0.659, 0.4, 0.137));

	// Node category colors (used for the node headers)
	_initial_set("editors/visual_editors/category_colors/output_color", Color(0.26, 0.10, 0.15));
	_initial_set("editors/visual_editors/category_colors/color_color", Color(0.5, 0.5, 0.1));
	_initial_set("editors/visual_editors/category_colors/conditional_color", Color(0.208, 0.522, 0.298));
	_initial_set("editors/visual_editors/category_colors/input_color", Color(0.502, 0.2, 0.204));
	_initial_set("editors/visual_editors/category_colors/scalar_color", Color(0.1, 0.5, 0.6));
	_initial_set("editors/visual_editors/category_colors/textures_color", Color(0.5, 0.3, 0.1));
	_initial_set("editors/visual_editors/category_colors/transform_color", Color(0.5, 0.3, 0.5));
	_initial_set("editors/visual_editors/category_colors/utility_color", Color(0.2, 0.2, 0.2));
	_initial_set("editors/visual_editors/category_colors/vector_color", Color(0.2, 0.2, 0.5));
	_initial_set("editors/visual_editors/category_colors/special_color", Color(0.098, 0.361, 0.294));
	_initial_set("editors/visual_editors/category_colors/particle_color", Color(0.12, 0.358, 0.8));
}

String EditorSettings::_guess_exec_args_for_extenal_editor(const String &p_path) {
	Ref<RegEx> regex = RegEx::create_from_string(R"((?:jetbrains\s*)?rider(?:\s*(eap|\d{4}\.\d+|\d{4}\.\d+\s*dev)?)?|visual\s*studio\s*code|subl(ime\s*text)?|sublime_text|zed(it(or)?)?|(g)?vim|emacs|atom|geany|kate|code|(vs)?codium)");
	Ref<RegExMatch> editor_match = regex->search(p_path.to_lower().get_file().get_basename());

	if (editor_match.is_null()) {
		return String();
	}

	const String editor = editor_match->get_string(0).to_lower();
	String new_exec_flags = "{file}";

	if (editor.begins_with("rider")) {
		new_exec_flags = "{project} --line {line} {file}";
	} else if (editor == "subl" || editor == "sublime text" || editor == "sublime_text" || editor == "zed" || editor == "zedit" || editor == "zeditor") {
		new_exec_flags = "{project} {file}:{line}:{col}";
	} else if (editor == "vim" || editor == "gvim") {
		new_exec_flags = "\"+call cursor({line}, {col})\" {file}";
	} else if (editor == "emacs") {
		new_exec_flags = "emacs +{line}:{col} {file}";
	} else if (editor == "atom") {
		new_exec_flags = "{file}:{line}";
	} else if (editor == "geany" || editor == "kate") {
		new_exec_flags = "{file} --line {line} --column {col}";
	} else if (editor == "code" || editor == "visual studio code" || editor == "codium" || editor == "vscodium") {
		new_exec_flags = "{project} --goto {file}:{line}:{col}";
	}

	return new_exec_flags;
}

const String EditorSettings::_get_project_metadata_path() const {
	return EditorPaths::get_singleton()->get_project_settings_dir().path_join("project_metadata.cfg");
}

#ifndef DISABLE_DEPRECATED
void EditorSettings::_remove_deprecated_settings() {
	erase("interface/theme/preset");
	erase("network/connection/engine_version_update_mode");
	erase("run/output/always_open_output_on_play");
	erase("run/output/always_close_output_on_stop");
	erase("text_editor/theme/line_spacing"); // See GH-106137.
}
#endif

// PUBLIC METHODS

EditorSettings *EditorSettings::get_singleton() {
	return singleton.ptr();
}

String EditorSettings::get_existing_settings_path() {
	const String config_dir = EditorPaths::get_singleton()->get_config_dir();
	int minor = GODOT_VERSION_MINOR;
	String filename;

	do {
		if (GODOT_VERSION_MAJOR == 4 && minor < 3) {
			// Minor version is used since 4.3, so special case to load older settings.
			filename = vformat("editor_settings-%d.tres", GODOT_VERSION_MAJOR);
			minor = -1;
		} else {
			filename = vformat("editor_settings-%d.%d.tres", GODOT_VERSION_MAJOR, minor);
			minor--;
		}
	} while (minor >= 0 && !FileAccess::exists(config_dir.path_join(filename)));
	return config_dir.path_join(filename);
}

String EditorSettings::get_newest_settings_path() {
	const String config_file_name = vformat("editor_settings-%d.%d.tres", GODOT_VERSION_MAJOR, GODOT_VERSION_MINOR);
	return EditorPaths::get_singleton()->get_config_dir().path_join(config_file_name);
}

void EditorSettings::create() {
	// IMPORTANT: create() *must* create a valid EditorSettings singleton,
	// as the rest of the engine code will assume it. As such, it should never
	// return (incl. via ERR_FAIL) without initializing the singleton member.

	if (singleton.ptr()) {
		ERR_PRINT("Can't recreate EditorSettings as it already exists.");
		return;
	}

	String config_file_path;
	Ref<ConfigFile> extra_config = memnew(ConfigFile);

	if (!EditorPaths::get_singleton()) {
		ERR_PRINT("Bug (please report): EditorPaths haven't been initialized, EditorSettings cannot be created properly.");
		goto fail;
	}

	if (EditorPaths::get_singleton()->is_self_contained()) {
		Error err = extra_config->load(EditorPaths::get_singleton()->get_self_contained_file());
		if (err != OK) {
			ERR_PRINT("Can't load extra config from path: " + EditorPaths::get_singleton()->get_self_contained_file());
		}
	}

	if (EditorPaths::get_singleton()->are_paths_valid()) {
		// Validate editor config file.
		ERR_FAIL_COND(!DirAccess::dir_exists_absolute(EditorPaths::get_singleton()->get_config_dir()));

		config_file_path = get_existing_settings_path();
		if (!FileAccess::exists(config_file_path)) {
			config_file_path = get_newest_settings_path();
			goto fail;
		}

		singleton = ResourceLoader::load(config_file_path, "EditorSettings");
		if (singleton.is_null()) {
			ERR_PRINT("Could not load editor settings from path: " + config_file_path);
			config_file_path = get_newest_settings_path();
			goto fail;
		}

		singleton->set_path(get_newest_settings_path()); // Settings can be loaded from older version file, so make sure it's newest.
		singleton->save_changed_setting = true;

		print_verbose("EditorSettings: Load OK!");

		singleton->setup_language(true);
		singleton->setup_network();
		singleton->load_favorites_and_recent_dirs();
		singleton->update_text_editor_themes_list();
#ifndef DISABLE_DEPRECATED
		singleton->_remove_deprecated_settings();
#endif

		return;
	}

fail:
	// patch init projects
	String exe_path = OS::get_singleton()->get_executable_path().get_base_dir();

	if (extra_config->has_section("init_projects")) {
		Vector<String> list = extra_config->get_value("init_projects", "list");
		for (int i = 0; i < list.size(); i++) {
			list.write[i] = exe_path.path_join(list[i]);
		}
		extra_config->set_value("init_projects", "list", list);
	}

	singleton.instantiate();
	singleton->set_path(config_file_path, true);
	singleton->save_changed_setting = true;
	singleton->_load_defaults(extra_config);
	singleton->setup_language(true);
	singleton->setup_network();
	singleton->update_text_editor_themes_list();
}

void EditorSettings::setup_language(bool p_initial_setup) {
	String lang = get_language();
	if (p_initial_setup) {
		String lang_ov = Main::get_locale_override();
		if (!lang_ov.is_empty()) {
			lang = lang_ov;
		}
	}

	if (lang == "en") {
		TranslationServer::get_singleton()->set_locale(lang);
		return; // Default, nothing to do.
	}

	load_editor_translations(lang);
	load_doc_translations(lang);

	TranslationServer::get_singleton()->set_locale(lang);
}

void EditorSettings::setup_network() {
	List<IPAddress> local_ip;
	IP::get_singleton()->get_local_addresses(&local_ip);
	String hint;
	String current = has_setting("network/debug/remote_host") ? get("network/debug/remote_host") : "";
	String selected = "127.0.0.1";

	// Check that current remote_host is a valid interface address and populate hints.
	for (const IPAddress &ip : local_ip) {
		// link-local IPv6 addresses don't work, skipping them
		if (String(ip).begins_with("fe80:0:0:0:")) { // fe80::/64
			continue;
		}
		// Same goes for IPv4 link-local (APIPA) addresses.
		if (String(ip).begins_with("169.254.")) { // 169.254.0.0/16
			continue;
		}
		// Select current IP (found)
		if (ip == current) {
			selected = String(ip);
		}
		if (!hint.is_empty()) {
			hint += ",";
		}
		hint += String(ip);
	}

	// Add hints with valid IP addresses to remote_host property.
	add_property_hint(PropertyInfo(Variant::STRING, "network/debug/remote_host", PROPERTY_HINT_ENUM, hint));
	// Fix potentially invalid remote_host due to network change.
	set("network/debug/remote_host", selected);
}

void EditorSettings::save() {
	//_THREAD_SAFE_METHOD_

	if (!singleton.ptr()) {
		return;
	}

	Error err = ResourceSaver::save(singleton);

	if (err != OK) {
		ERR_PRINT("Error saving editor settings to " + singleton->get_path());
	} else {
		singleton->changed_settings.clear();
		print_verbose("EditorSettings: Save OK!");
	}
}

PackedStringArray EditorSettings::get_changed_settings() const {
	PackedStringArray arr;
	for (const String &setting : changed_settings) {
		arr.push_back(setting);
	}

	return arr;
}

bool EditorSettings::check_changed_settings_in_group(const String &p_setting_prefix) const {
	for (const String &setting : changed_settings) {
		if (setting.begins_with(p_setting_prefix)) {
			return true;
		}
	}

	return false;
}

void EditorSettings::mark_setting_changed(const String &p_setting) {
	changed_settings.insert(p_setting);
}

void EditorSettings::destroy() {
	if (!singleton.ptr()) {
		return;
	}
	save();
	singleton = Ref<EditorSettings>();
}

void EditorSettings::set_optimize_save(bool p_optimize) {
	optimize_save = p_optimize;
}

// Properties

void EditorSettings::set_setting(const String &p_setting, const Variant &p_value) {
	_THREAD_SAFE_METHOD_
	set(p_setting, p_value);
}

Variant EditorSettings::get_setting(const String &p_setting) const {
	_THREAD_SAFE_METHOD_
	if (ProjectSettings::get_singleton()->has_editor_setting_override(p_setting)) {
		return ProjectSettings::get_singleton()->get_editor_setting_override(p_setting);
	}
	return get(p_setting);
}

bool EditorSettings::has_setting(const String &p_setting) const {
	_THREAD_SAFE_METHOD_

	return props.has(p_setting);
}

void EditorSettings::erase(const String &p_setting) {
	_THREAD_SAFE_METHOD_

	props.erase(p_setting);
}

void EditorSettings::raise_order(const String &p_setting) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!props.has(p_setting));
	props[p_setting].order = ++last_order;
}

void EditorSettings::set_restart_if_changed(const StringName &p_setting, bool p_restart) {
	_THREAD_SAFE_METHOD_

	if (!props.has(p_setting)) {
		return;
	}
	props[p_setting].restart_if_changed = p_restart;
}

void EditorSettings::set_basic(const StringName &p_setting, bool p_basic) {
	_THREAD_SAFE_METHOD_

	if (!props.has(p_setting)) {
		return;
	}
	props[p_setting].basic = p_basic;
}

void EditorSettings::set_initial_value(const StringName &p_setting, const Variant &p_value, bool p_update_current) {
	_THREAD_SAFE_METHOD_

	if (!props.has(p_setting)) {
		return;
	}
	props[p_setting].initial = p_value;
	props[p_setting].has_default_value = true;
	if (p_update_current) {
		set(p_setting, p_value);
	}
}

Variant _EDITOR_DEF(const String &p_setting, const Variant &p_default, bool p_restart_if_changed, bool p_basic) {
	ERR_FAIL_NULL_V_MSG(EditorSettings::get_singleton(), p_default, "EditorSettings not instantiated yet.");

	Variant ret = p_default;
	if (EditorSettings::get_singleton()->has_setting(p_setting)) {
		ret = EDITOR_GET(p_setting);
	} else {
		EditorSettings::get_singleton()->set_manually(p_setting, p_default);
	}
	EditorSettings::get_singleton()->set_restart_if_changed(p_setting, p_restart_if_changed);
	EditorSettings::get_singleton()->set_basic(p_setting, p_basic);

	if (!EditorSettings::get_singleton()->has_default_value(p_setting)) {
		EditorSettings::get_singleton()->set_initial_value(p_setting, p_default);
	}

	return ret;
}

Variant _EDITOR_GET(const String &p_setting) {
	ERR_FAIL_NULL_V_MSG(EditorSettings::get_singleton(), Variant(), vformat(R"(EditorSettings not instantiated yet when getting setting "%s".)", p_setting));
	ERR_FAIL_COND_V_MSG(!EditorSettings::get_singleton()->has_setting(p_setting), Variant(), vformat(R"(Editor setting "%s" does not exist.)", p_setting));
	return EditorSettings::get_singleton()->get_setting(p_setting);
}

bool EditorSettings::_property_can_revert(const StringName &p_name) const {
	const VariantContainer *property = props.getptr(p_name);
	if (property) {
		return property->has_default_value;
	}
	return false;
}

bool EditorSettings::_property_get_revert(const StringName &p_name, Variant &r_property) const {
	const VariantContainer *value = props.getptr(p_name);
	if (value && value->has_default_value) {
		r_property = value->initial;
		return true;
	}
	return false;
}

void EditorSettings::add_property_hint(const PropertyInfo &p_hint) {
	_THREAD_SAFE_METHOD_

	hints[p_hint.name] = p_hint;
}

// Metadata

void EditorSettings::set_project_metadata(const String &p_section, const String &p_key, const Variant &p_data) {
	const String path = _get_project_metadata_path();

	if (project_metadata.is_null()) {
		project_metadata.instantiate();

		Error err = project_metadata->load(path);
		if (err != OK && err != ERR_FILE_NOT_FOUND) {
			ERR_PRINT("Cannot load project metadata from file '" + path + "'.");
		}
	}
	project_metadata->set_value(p_section, p_key, p_data);
	project_metadata_dirty = true;
}

Variant EditorSettings::get_project_metadata(const String &p_section, const String &p_key, const Variant &p_default) const {
	if (project_metadata.is_null()) {
		project_metadata.instantiate();

		const String path = _get_project_metadata_path();
		Error err = project_metadata->load(path);
		ERR_FAIL_COND_V_MSG(err != OK && err != ERR_FILE_NOT_FOUND, p_default, "Cannot load project metadata from file '" + path + "'.");
	}
	return project_metadata->get_value(p_section, p_key, p_default);
}

void EditorSettings::save_project_metadata() {
	if (!project_metadata_dirty) {
		return;
	}
	const String path = _get_project_metadata_path();
	Error err = project_metadata->save(path);
	ERR_FAIL_COND_MSG(err != OK, "Cannot save project metadata to file '" + path + "'.");
	project_metadata_dirty = false;
}

void EditorSettings::set_favorites(const Vector<String> &p_favorites, bool p_update_file_dialog) {
	if (p_update_file_dialog) {
		FileDialog::set_favorite_list(p_favorites);
	} else if (p_favorites == favorites) {
		// If the list came from EditorFileDialog, it may be the same as before.
		return;
	}
	set_favorites_bind(p_favorites);
}

void EditorSettings::set_favorites_bind(const Vector<String> &p_favorites) {
	favorites = p_favorites;
	String favorites_file;
	if (Engine::get_singleton()->is_project_manager_hint()) {
		favorites_file = EditorPaths::get_singleton()->get_config_dir().path_join("favorite_dirs");
	} else {
		favorites_file = EditorPaths::get_singleton()->get_project_settings_dir().path_join("favorites");
	}
	Ref<FileAccess> f = FileAccess::open(favorites_file, FileAccess::WRITE);
	if (f.is_valid()) {
		for (int i = 0; i < favorites.size(); i++) {
			f->store_line(favorites[i]);
		}
	}
}

void EditorSettings::set_favorite_properties(const HashMap<String, PackedStringArray> &p_favorite_properties) {
	favorite_properties = p_favorite_properties;
	String favorite_properties_file = EditorPaths::get_singleton()->get_project_settings_dir().path_join("favorite_properties");

	Ref<ConfigFile> cf;
	cf.instantiate();
	for (const KeyValue<String, PackedStringArray> &kv : p_favorite_properties) {
		cf->set_value(kv.key, "properties", kv.value);
	}
	cf->save(favorite_properties_file);
}

Vector<String> EditorSettings::get_favorites() const {
	return favorites;
}

HashMap<String, PackedStringArray> EditorSettings::get_favorite_properties() const {
	return favorite_properties;
}

void EditorSettings::set_recent_dirs(const Vector<String> &p_recent_dirs, bool p_update_file_dialog) {
	if (p_update_file_dialog) {
		FileDialog::set_recent_list(p_recent_dirs);
	} else if (p_recent_dirs == recent_dirs) {
		// If the list came from EditorFileDialog, it may be the same as before.
		return;
	}
	set_recent_dirs_bind(p_recent_dirs);
}

void EditorSettings::set_recent_dirs_bind(const Vector<String> &p_recent_dirs) {
	recent_dirs = p_recent_dirs;
	String recent_dirs_file;
	if (Engine::get_singleton()->is_project_manager_hint()) {
		recent_dirs_file = EditorPaths::get_singleton()->get_config_dir().path_join("recent_dirs");
	} else {
		recent_dirs_file = EditorPaths::get_singleton()->get_project_settings_dir().path_join("recent_dirs");
	}
	Ref<FileAccess> f = FileAccess::open(recent_dirs_file, FileAccess::WRITE);
	if (f.is_valid()) {
		for (int i = 0; i < recent_dirs.size(); i++) {
			f->store_line(recent_dirs[i]);
		}
	}
}

Vector<String> EditorSettings::get_recent_dirs() const {
	return recent_dirs;
}

void EditorSettings::load_favorites_and_recent_dirs() {
	String favorites_file;
	String favorite_properties_file;
	String recent_dirs_file;
	if (Engine::get_singleton()->is_project_manager_hint()) {
		favorites_file = EditorPaths::get_singleton()->get_config_dir().path_join("favorite_dirs");
		favorite_properties_file = EditorPaths::get_singleton()->get_config_dir().path_join("favorite_properties");
		recent_dirs_file = EditorPaths::get_singleton()->get_config_dir().path_join("recent_dirs");
	} else {
		favorites_file = EditorPaths::get_singleton()->get_project_settings_dir().path_join("favorites");
		favorite_properties_file = EditorPaths::get_singleton()->get_project_settings_dir().path_join("favorite_properties");
		recent_dirs_file = EditorPaths::get_singleton()->get_project_settings_dir().path_join("recent_dirs");
	}

	/// File Favorites

	Ref<FileAccess> f = FileAccess::open(favorites_file, FileAccess::READ);
	if (f.is_valid()) {
		String line = f->get_line().strip_edges();
		while (!line.is_empty()) {
			favorites.append(line);
			line = f->get_line().strip_edges();
		}
	}
	FileDialog::set_favorite_list(favorites);

	/// Inspector Favorites

	Ref<ConfigFile> cf;
	cf.instantiate();
	if (cf->load(favorite_properties_file) == OK) {
		Vector<String> secs = cf->get_sections();

		for (String &E : secs) {
			PackedStringArray properties = PackedStringArray(cf->get_value(E, "properties"));
			if (EditorNode::get_editor_data().is_type_recognized(E) || ResourceLoader::exists(E, "Script")) {
				for (const String &property : properties) {
					if (!favorite_properties[E].has(property)) {
						favorite_properties[E].push_back(property);
					}
				}
			}
		}
	}

	/// Recent Directories

	f = FileAccess::open(recent_dirs_file, FileAccess::READ);
	if (f.is_valid()) {
		String line = f->get_line().strip_edges();
		while (!line.is_empty()) {
			recent_dirs.push_back(line);
			line = f->get_line().strip_edges();
		}
	}
	FileDialog::set_recent_list(recent_dirs);
}

HashMap<StringName, Color> EditorSettings::get_godot2_text_editor_theme() {
	// Godot 2 is only a dark theme; it doesn't have a light theme counterpart.
	HashMap<StringName, Color> colors;
	colors["text_editor/theme/highlighting/symbol_color"] = Color(0.73, 0.87, 1.0);
	colors["text_editor/theme/highlighting/keyword_color"] = Color(1.0, 1.0, 0.7);
	colors["text_editor/theme/highlighting/control_flow_keyword_color"] = Color(1.0, 0.85, 0.7);
	colors["text_editor/theme/highlighting/base_type_color"] = Color(0.64, 1.0, 0.83);
	colors["text_editor/theme/highlighting/engine_type_color"] = Color(0.51, 0.83, 1.0);
	colors["text_editor/theme/highlighting/user_type_color"] = Color(0.42, 0.67, 0.93);
	colors["text_editor/theme/highlighting/comment_color"] = Color(0.4, 0.4, 0.4);
	colors["text_editor/theme/highlighting/doc_comment_color"] = Color(0.5, 0.6, 0.7);
	colors["text_editor/theme/highlighting/string_color"] = Color(0.94, 0.43, 0.75);
	colors["text_editor/theme/highlighting/string_placeholder_color"] = Color(1, 0.75, 0.4);
	colors["text_editor/theme/highlighting/background_color"] = Color(0.13, 0.12, 0.15);
	colors["text_editor/theme/highlighting/completion_background_color"] = Color(0.17, 0.16, 0.2);
	colors["text_editor/theme/highlighting/completion_selected_color"] = Color(0.26, 0.26, 0.27);
	colors["text_editor/theme/highlighting/completion_existing_color"] = Color(0.87, 0.87, 0.87, 0.13);
	colors["text_editor/theme/highlighting/completion_scroll_color"] = Color(1, 1, 1, 0.29);
	colors["text_editor/theme/highlighting/completion_scroll_hovered_color"] = Color(1, 1, 1, 0.4);
	colors["text_editor/theme/highlighting/completion_font_color"] = Color(0.67, 0.67, 0.67);
	colors["text_editor/theme/highlighting/text_color"] = Color(0.67, 0.67, 0.67);
	colors["text_editor/theme/highlighting/line_number_color"] = Color(0.67, 0.67, 0.67, 0.4);
	colors["text_editor/theme/highlighting/safe_line_number_color"] = Color(0.67, 0.78, 0.67, 0.6);
	colors["text_editor/theme/highlighting/caret_color"] = Color(0.67, 0.67, 0.67);
	colors["text_editor/theme/highlighting/caret_background_color"] = Color(0, 0, 0);
	colors["text_editor/theme/highlighting/text_selected_color"] = Color(0, 0, 0, 0);
	colors["text_editor/theme/highlighting/selection_color"] = Color(0.41, 0.61, 0.91, 0.35);
	colors["text_editor/theme/highlighting/brace_mismatch_color"] = Color(1, 0.2, 0.2);
	colors["text_editor/theme/highlighting/current_line_color"] = Color(0.3, 0.5, 0.8, 0.15);
	colors["text_editor/theme/highlighting/line_length_guideline_color"] = Color(0.3, 0.5, 0.8, 0.1);
	colors["text_editor/theme/highlighting/word_highlighted_color"] = Color(0.8, 0.9, 0.9, 0.15);
	colors["text_editor/theme/highlighting/number_color"] = Color(0.92, 0.58, 0.2);
	colors["text_editor/theme/highlighting/function_color"] = Color(0.4, 0.64, 0.81);
	colors["text_editor/theme/highlighting/member_variable_color"] = Color(0.9, 0.31, 0.35);
	colors["text_editor/theme/highlighting/mark_color"] = Color(1.0, 0.4, 0.4, 0.4);
	colors["text_editor/theme/highlighting/warning_color"] = Color(1.0, 0.8, 0.4, 0.1);
	colors["text_editor/theme/highlighting/bookmark_color"] = Color(0.08, 0.49, 0.98);
	colors["text_editor/theme/highlighting/breakpoint_color"] = Color(0.9, 0.29, 0.3);
	colors["text_editor/theme/highlighting/executing_line_color"] = Color(0.98, 0.89, 0.27);
	colors["text_editor/theme/highlighting/code_folding_color"] = Color(0.8, 0.8, 0.8, 0.8);
	colors["text_editor/theme/highlighting/folded_code_region_color"] = Color(0.68, 0.46, 0.77, 0.2);
	colors["text_editor/theme/highlighting/search_result_color"] = Color(0.05, 0.25, 0.05, 1);
	colors["text_editor/theme/highlighting/search_result_border_color"] = Color(0.41, 0.61, 0.91, 0.38);
	colors["text_editor/theme/highlighting/gdscript/function_definition_color"] = Color(0.4, 0.9, 1.0);

	colors["text_editor/theme/highlighting/gdscript/global_function_color"] = Color(0.64, 0.64, 0.96);
	colors["text_editor/theme/highlighting/gdscript/node_path_color"] = Color(0.72, 0.77, 0.49);
	colors["text_editor/theme/highlighting/gdscript/node_reference_color"] = Color(0.39, 0.76, 0.35);
	colors["text_editor/theme/highlighting/gdscript/annotation_color"] = Color(1.0, 0.7, 0.45);
	colors["text_editor/theme/highlighting/gdscript/string_name_color"] = Color(1.0, 0.76, 0.65);
	colors["text_editor/theme/highlighting/comment_markers/critical_color"] = Color(0.77, 0.35, 0.35);
	colors["text_editor/theme/highlighting/comment_markers/warning_color"] = Color(0.72, 0.61, 0.48);
	colors["text_editor/theme/highlighting/comment_markers/notice_color"] = Color(0.56, 0.67, 0.51);
	return colors;
}

bool EditorSettings::is_default_text_editor_theme(const String &p_theme_name) {
	return p_theme_name == "default" || p_theme_name == "godot 2" || p_theme_name == "custom";
}

void EditorSettings::update_text_editor_themes_list() {
	String themes = "Default,Godot 2,Custom";

	Ref<DirAccess> d = DirAccess::open(EditorPaths::get_singleton()->get_text_editor_themes_dir());
	if (d.is_null()) {
		return;
	}

	PackedStringArray custom_themes;
	d->list_dir_begin();
	String file = d->get_next();
	while (!file.is_empty()) {
		if (file.get_extension() == "tet" && !is_default_text_editor_theme(file.get_basename().to_lower())) {
			custom_themes.push_back(file.get_basename());
		}
		file = d->get_next();
	}
	d->list_dir_end();

	if (!custom_themes.is_empty()) {
		custom_themes.sort();
		themes += "," + String(",").join(custom_themes);
	}
	add_property_hint(PropertyInfo(Variant::STRING, "text_editor/theme/color_theme", PROPERTY_HINT_ENUM, themes));
}

Vector<String> EditorSettings::get_script_templates(const String &p_extension, const String &p_custom_path) {
	Vector<String> templates;
	String template_dir = EditorPaths::get_singleton()->get_script_templates_dir();
	if (!p_custom_path.is_empty()) {
		template_dir = p_custom_path;
	}
	Ref<DirAccess> d = DirAccess::open(template_dir);
	if (d.is_valid()) {
		d->list_dir_begin();
		String file = d->get_next();
		while (!file.is_empty()) {
			if (file.get_extension() == p_extension) {
				templates.push_back(file.get_basename());
			}
			file = d->get_next();
		}
		d->list_dir_end();
	}
	return templates;
}

String EditorSettings::get_editor_layouts_config() const {
	return EditorPaths::get_singleton()->get_config_dir().path_join("editor_layouts.cfg");
}

float EditorSettings::get_auto_display_scale() {
#ifdef LINUXBSD_ENABLED
	if (DisplayServer::get_singleton()->get_name() == "Wayland") {
		float main_window_scale = DisplayServer::get_singleton()->screen_get_scale(DisplayServer::SCREEN_OF_MAIN_WINDOW);

		if (DisplayServer::get_singleton()->get_screen_count() == 1 || Math::fract(main_window_scale) != 0) {
			// If we have a single screen or the screen of the window is fractional, all
			// bets are off. At this point, let's just return the current's window scale,
			// which is special-cased to the scale of `SCREEN_OF_MAIN_WINDOW`.
			return main_window_scale;
		}

		// If the above branch didn't fire, fractional scaling isn't going to work
		// properly anyways (we're need the ability to change the UI scale at runtime).
		// At this point it's more convenient to "supersample" like we do with other
		// platforms, hoping that the user is only using integer-scaled screens.
		return DisplayServer::get_singleton()->screen_get_max_scale();
	}
#endif

#if defined(MACOS_ENABLED) || defined(ANDROID_ENABLED)
	return DisplayServer::get_singleton()->screen_get_max_scale();
#else
	const int screen = DisplayServer::get_singleton()->window_get_current_screen();

	if (DisplayServer::get_singleton()->screen_get_size(screen) == Vector2i()) {
		// Invalid screen size, skip.
		return 1.0;
	}

#if defined(WINDOWS_ENABLED)
	return DisplayServer::get_singleton()->screen_get_dpi(screen) / 96.0;
#else
	// Use the smallest dimension to use a correct display scale on portrait displays.
	const int smallest_dimension = MIN(DisplayServer::get_singleton()->screen_get_size(screen).x, DisplayServer::get_singleton()->screen_get_size(screen).y);
	if (DisplayServer::get_singleton()->screen_get_dpi(screen) >= 192 && smallest_dimension >= 1400) {
		// hiDPI display.
		return 2.0;
	} else if (smallest_dimension >= 1700) {
		// Likely a hiDPI display, but we aren't certain due to the returned DPI.
		// Use an intermediate scale to handle this situation.
		return 1.5;
	} else if (smallest_dimension <= 800) {
		// Small loDPI display. Use a smaller display scale so that editor elements fit more easily.
		// Icons won't look great, but this is better than having editor elements overflow from its window.
		return 0.75;
	}
	return 1.0;
#endif // defined(WINDOWS_ENABLED)

#endif // defined(MACOS_ENABLED) || defined(ANDROID_ENABLED)
}

String EditorSettings::get_language() const {
	const String language = has_setting("interface/editor/editor_language") ? get("interface/editor/editor_language") : "auto";
	if (language != "auto") {
		return language;
	}

	if (auto_language.is_empty()) {
		// Skip locales which we can't render properly.
		const LocalVector<String> locales_to_skip = _get_skipped_locales();
		const String host_lang = OS::get_singleton()->get_locale();

		String best = "en";
		int best_score = 0;
		for (const String &locale : get_editor_locales()) {
			// Test against language code without regional variants (e.g. ur_PK).
			String lang_code = locale.get_slicec('_', 0);
			if (locales_to_skip.has(lang_code)) {
				continue;
			}

			int score = TranslationServer::get_singleton()->compare_locales(host_lang, locale);
			if (score > 0 && score >= best_score) {
				best = locale;
				best_score = score;
			}
		}
		auto_language = best;
	}
	return auto_language;
}

// Shortcuts

void EditorSettings::_add_shortcut_default(const String &p_path, const Ref<Shortcut> &p_shortcut) {
	shortcuts[p_path] = p_shortcut;
}

void EditorSettings::add_shortcut(const String &p_path, const Ref<Shortcut> &p_shortcut) {
	Array use_events = p_shortcut->get_events();
	if (shortcuts.has(p_path)) {
		Ref<Shortcut> existing = shortcuts.get(p_path);
		if (!existing->has_meta("original")) {
			// Loaded from editor settings, but plugin not loaded yet.
			// Keep the events from editor settings but still override the shortcut in the shortcuts map
			use_events = existing->get_events();
		} else if (!Shortcut::is_event_array_equal(existing->get_events(), existing->get_meta("original"))) {
			// Shortcut exists and is customized - don't override with default.
			return;
		}
	}

	p_shortcut->set_meta("original", p_shortcut->get_events());
	p_shortcut->set_events(use_events);
	if (p_shortcut->get_name().is_empty()) {
		String shortcut_name = p_path.get_slicec('/', 1);
		if (shortcut_name.is_empty()) {
			shortcut_name = p_path;
		}
		p_shortcut->set_name(shortcut_name);
	}
	shortcuts[p_path] = p_shortcut;
	shortcuts[p_path]->set_meta("customized", true);
}

void EditorSettings::remove_shortcut(const String &p_path) {
	shortcuts.erase(p_path);
}

bool EditorSettings::is_shortcut(const String &p_path, const Ref<InputEvent> &p_event) const {
	HashMap<String, Ref<Shortcut>>::ConstIterator E = shortcuts.find(p_path);
	ERR_FAIL_COND_V_MSG(!E, false, "Unknown Shortcut: " + p_path + ".");

	return E->value->matches_event(p_event);
}

bool EditorSettings::has_shortcut(const String &p_path) const {
	return get_shortcut(p_path).is_valid();
}

Ref<Shortcut> EditorSettings::get_shortcut(const String &p_path) const {
	HashMap<String, Ref<Shortcut>>::ConstIterator SC = shortcuts.find(p_path);
	if (SC) {
		return SC->value;
	}

	// If no shortcut with the provided name is found in the list, check the built-in shortcuts.
	// Use the first item in the action list for the shortcut event, since a shortcut can only have 1 linked event.

	Ref<Shortcut> sc;
	HashMap<String, List<Ref<InputEvent>>>::ConstIterator builtin_override = builtin_action_overrides.find(p_path);
	if (builtin_override) {
		sc.instantiate();
		sc->set_events_list(&builtin_override->value);
		sc->set_name(InputMap::get_singleton()->get_builtin_display_name(p_path));
	}

	// If there was no override, check the default builtins to see if it has an InputEvent for the provided name.
	if (sc.is_null()) {
		HashMap<String, List<Ref<InputEvent>>>::ConstIterator builtin_default = InputMap::get_singleton()->get_builtins_with_feature_overrides_applied().find(p_path);
		if (builtin_default) {
			sc.instantiate();
			sc->set_events_list(&builtin_default->value);
			sc->set_name(InputMap::get_singleton()->get_builtin_display_name(p_path));
		}
	}

	if (sc.is_valid()) {
		// Add the shortcut to the list.
		shortcuts[p_path] = sc;
		return sc;
	}

	return Ref<Shortcut>();
}

Vector<String> EditorSettings::_get_shortcut_list() {
	List<String> shortcut_list;
	get_shortcut_list(&shortcut_list);
	Vector<String> ret;
	for (const String &shortcut : shortcut_list) {
		ret.push_back(shortcut);
	}
	return ret;
}

void EditorSettings::get_shortcut_list(List<String> *r_shortcuts) {
	for (const KeyValue<String, Ref<Shortcut>> &E : shortcuts) {
		r_shortcuts->push_back(E.key);
	}
}

Ref<Shortcut> ED_GET_SHORTCUT(const String &p_path) {
	ERR_FAIL_NULL_V_MSG(EditorSettings::get_singleton(), nullptr, "EditorSettings not instantiated yet.");

	Ref<Shortcut> sc = EditorSettings::get_singleton()->get_shortcut(p_path);

	ERR_FAIL_COND_V_MSG(sc.is_null(), sc, "Used ED_GET_SHORTCUT with invalid shortcut: " + p_path);

	return sc;
}

void ED_SHORTCUT_OVERRIDE(const String &p_path, const String &p_feature, Key p_keycode, bool p_physical) {
	if (!EditorSettings::get_singleton()) {
		return;
	}

	Ref<Shortcut> sc = EditorSettings::get_singleton()->get_shortcut(p_path);
	ERR_FAIL_COND_MSG(sc.is_null(), "Used ED_SHORTCUT_OVERRIDE with invalid shortcut: " + p_path);

	PackedInt32Array arr;
	arr.push_back((int32_t)p_keycode);

	ED_SHORTCUT_OVERRIDE_ARRAY(p_path, p_feature, arr, p_physical);
}

void ED_SHORTCUT_OVERRIDE_ARRAY(const String &p_path, const String &p_feature, const PackedInt32Array &p_keycodes, bool p_physical) {
	if (!EditorSettings::get_singleton()) {
		return;
	}

	Ref<Shortcut> sc = EditorSettings::get_singleton()->get_shortcut(p_path);
	ERR_FAIL_COND_MSG(sc.is_null(), "Used ED_SHORTCUT_OVERRIDE_ARRAY with invalid shortcut: " + p_path);

	// Only add the override if the OS supports the provided feature.
	if (!OS::get_singleton()->has_feature(p_feature)) {
		if (!(p_feature == "macos" && (OS::get_singleton()->has_feature("web_macos") || OS::get_singleton()->has_feature("web_ios")))) {
			return;
		}
	}

	Array events;

	for (int i = 0; i < p_keycodes.size(); i++) {
		Key keycode = (Key)p_keycodes[i];

		if (OS::prefer_meta_over_ctrl()) {
			// Use Cmd+Backspace as a general replacement for Delete shortcuts on macOS
			if (keycode == Key::KEY_DELETE) {
				keycode = KeyModifierMask::META | Key::BACKSPACE;
			}
		}

		Ref<InputEventKey> ie;
		if (keycode != Key::NONE) {
			ie = InputEventKey::create_reference(keycode, p_physical);
			events.push_back(ie);
		}
	}

	// Override the existing shortcut only if it wasn't customized by the user.
	if (!sc->has_meta("customized")) {
		sc->set_events(events);
	}

	sc->set_meta("original", events.duplicate(true));
}

Ref<Shortcut> ED_SHORTCUT(const String &p_path, const String &p_name, Key p_keycode, bool p_physical) {
	PackedInt32Array arr;
	arr.push_back((int32_t)p_keycode);
	return ED_SHORTCUT_ARRAY(p_path, p_name, arr, p_physical);
}

Ref<Shortcut> ED_SHORTCUT_ARRAY(const String &p_path, const String &p_name, const PackedInt32Array &p_keycodes, bool p_physical) {
	Array events;

	for (int i = 0; i < p_keycodes.size(); i++) {
		Key keycode = (Key)p_keycodes[i];

		if (OS::prefer_meta_over_ctrl()) {
			// Use Cmd+Backspace as a general replacement for Delete shortcuts on macOS
			if (keycode == Key::KEY_DELETE) {
				keycode = KeyModifierMask::META | Key::BACKSPACE;
			}
		}

		Ref<InputEventKey> ie;
		if (keycode != Key::NONE) {
			ie = InputEventKey::create_reference(keycode, p_physical);
			events.push_back(ie);
		}
	}

	if (!EditorSettings::get_singleton()) {
		Ref<Shortcut> sc;
		sc.instantiate();
		sc->set_name(p_name);
		sc->set_events(events);
		sc->set_meta("original", events.duplicate(true));
		return sc;
	}

	Ref<Shortcut> sc = EditorSettings::get_singleton()->get_shortcut(p_path);
	if (sc.is_valid()) {
		sc->set_name(p_name); //keep name (the ones that come from disk have no name)
		sc->set_meta("original", events.duplicate(true)); //to compare against changes
		return sc;
	}

	sc.instantiate();
	sc->set_name(p_name);
	sc->set_events(events);
	sc->set_meta("original", events.duplicate(true)); //to compare against changes
	EditorSettings::get_singleton()->_add_shortcut_default(p_path, sc);

	return sc;
}

void EditorSettings::set_builtin_action_override(const String &p_name, const TypedArray<InputEvent> &p_events) {
	List<Ref<InputEvent>> event_list;

	// Override the whole list, since events may have their order changed or be added, removed or edited.
	InputMap::get_singleton()->action_erase_events(p_name);
	for (int i = 0; i < p_events.size(); i++) {
		event_list.push_back(p_events[i]);
		InputMap::get_singleton()->action_add_event(p_name, p_events[i]);
	}

	// Check if the provided event array is same as built-in. If it is, it does not need to be added to the overrides.
	// Note that event order must also be the same.
	bool same_as_builtin = true;
	HashMap<String, List<Ref<InputEvent>>>::ConstIterator builtin_default = InputMap::get_singleton()->get_builtins_with_feature_overrides_applied().find(p_name);
	if (builtin_default) {
		const List<Ref<InputEvent>> &builtin_events = builtin_default->value;

		// In the editor we only care about key events.
		List<Ref<InputEventKey>> builtin_key_events;
		for (Ref<InputEventKey> iek : builtin_events) {
			if (iek.is_valid()) {
				builtin_key_events.push_back(iek);
			}
		}

		if (p_events.size() == builtin_key_events.size()) {
			int event_idx = 0;

			// Check equality of each event.
			for (const Ref<InputEventKey> &E : builtin_key_events) {
				if (!E->is_match(p_events[event_idx])) {
					same_as_builtin = false;
					break;
				}
				event_idx++;
			}
		} else {
			same_as_builtin = false;
		}
	}

	if (same_as_builtin && builtin_action_overrides.has(p_name)) {
		builtin_action_overrides.erase(p_name);
	} else {
		builtin_action_overrides[p_name] = event_list;
	}

	// Update the shortcut (if it is used somewhere in the editor) to be the first event of the new list.
	if (shortcuts.has(p_name)) {
		shortcuts[p_name]->set_events_list(&event_list);
	}
}

const Array EditorSettings::get_builtin_action_overrides(const String &p_name) const {
	HashMap<String, List<Ref<InputEvent>>>::ConstIterator AO = builtin_action_overrides.find(p_name);
	if (AO) {
		Array event_array;

		List<Ref<InputEvent>> events_list = AO->value;
		for (const Ref<InputEvent> &E : events_list) {
			event_array.push_back(E);
		}
		return event_array;
	}

	return Array();
}

void EditorSettings::notify_changes() {
	_THREAD_SAFE_METHOD_

	SceneTree *sml = Object::cast_to<SceneTree>(OS::get_singleton()->get_main_loop());

	if (!sml) {
		return;
	}

	Node *root = sml->get_root()->get_child(0);

	if (!root) {
		return;
	}
	root->propagate_notification(NOTIFICATION_EDITOR_SETTINGS_CHANGED);
}

void EditorSettings::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
	const String pf = p_function;
	if (p_idx == 0) {
		if (pf == "has_setting" || pf == "set_setting" || pf == "get_setting" || pf == "erase" ||
				pf == "set_initial_value" || pf == "set_as_basic" || pf == "mark_setting_changed") {
			for (const KeyValue<String, VariantContainer> &E : props) {
				if (E.value.hide_from_editor) {
					continue;
				}

				r_options->push_back(E.key.quote());
			}
		} else if (pf == "get_project_metadata" && project_metadata.is_valid()) {
			Vector<String> sections = project_metadata->get_sections();
			for (const String &section : sections) {
				r_options->push_back(section.quote());
			}
		} else if (pf == "set_builtin_action_override") {
			for (const Variant &action : InputMap::get_singleton()->get_actions()) {
				r_options->push_back(String(action).quote());
			}
		}
	}
	Object::get_argument_options(p_function, p_idx, r_options);
}

void EditorSettings::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_setting", "name"), &EditorSettings::has_setting);
	ClassDB::bind_method(D_METHOD("set_setting", "name", "value"), &EditorSettings::set_setting);
	ClassDB::bind_method(D_METHOD("get_setting", "name"), &EditorSettings::get_setting);
	ClassDB::bind_method(D_METHOD("erase", "property"), &EditorSettings::erase);
	ClassDB::bind_method(D_METHOD("set_initial_value", "name", "value", "update_current"), &EditorSettings::set_initial_value);
	ClassDB::bind_method(D_METHOD("add_property_info", "info"), &EditorSettings::_add_property_info_bind);

	ClassDB::bind_method(D_METHOD("set_project_metadata", "section", "key", "data"), &EditorSettings::set_project_metadata);
	ClassDB::bind_method(D_METHOD("get_project_metadata", "section", "key", "default"), &EditorSettings::get_project_metadata, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("set_favorites", "dirs"), &EditorSettings::set_favorites_bind);
	ClassDB::bind_method(D_METHOD("get_favorites"), &EditorSettings::get_favorites);
	ClassDB::bind_method(D_METHOD("set_recent_dirs", "dirs"), &EditorSettings::set_recent_dirs_bind);
	ClassDB::bind_method(D_METHOD("get_recent_dirs"), &EditorSettings::get_recent_dirs);

	ClassDB::bind_method(D_METHOD("set_builtin_action_override", "name", "actions_list"), &EditorSettings::set_builtin_action_override);

	ClassDB::bind_method(D_METHOD("add_shortcut", "path", "shortcut"), &EditorSettings::add_shortcut);
	ClassDB::bind_method(D_METHOD("remove_shortcut", "path"), &EditorSettings::remove_shortcut);
	ClassDB::bind_method(D_METHOD("is_shortcut", "path", "event"), &EditorSettings::is_shortcut);
	ClassDB::bind_method(D_METHOD("has_shortcut", "path"), &EditorSettings::has_shortcut);
	ClassDB::bind_method(D_METHOD("get_shortcut", "path"), &EditorSettings::get_shortcut);
	ClassDB::bind_method(D_METHOD("get_shortcut_list"), &EditorSettings::_get_shortcut_list);

	ClassDB::bind_method(D_METHOD("check_changed_settings_in_group", "setting_prefix"), &EditorSettings::check_changed_settings_in_group);
	ClassDB::bind_method(D_METHOD("get_changed_settings"), &EditorSettings::get_changed_settings);
	ClassDB::bind_method(D_METHOD("mark_setting_changed", "setting"), &EditorSettings::mark_setting_changed);

	ADD_SIGNAL(MethodInfo("settings_changed"));

	BIND_CONSTANT(NOTIFICATION_EDITOR_SETTINGS_CHANGED);
}

EditorSettings::EditorSettings() {
	last_order = 0;

	_load_defaults();
	callable_mp(this, &EditorSettings::_set_initialized).call_deferred();
}
