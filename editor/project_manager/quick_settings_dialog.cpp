/**************************************************************************/
/*  quick_settings_dialog.cpp                                             */
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

#include "quick_settings_dialog.h"

#include "core/string/translation_server.h"
#include "editor/doc/editor_help.h"
#include "editor/editor_string_names.h"
#include "editor/inspector/editor_properties.h"
#include "editor/settings/editor_settings.h"
#include "editor/settings/editor_settings_dialog.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel_container.h"

void QuickSettingsDialog::_fetch_setting_values() {
#ifndef ANDROID_ENABLED
	editor_languages.clear();
#endif
	editor_styles.clear();
	editor_themes.clear();
	editor_scales.clear();
	editor_network_modes.clear();
	editor_check_for_updates.clear();
	editor_directory_naming_conventions.clear();

	{
		List<PropertyInfo> editor_settings_properties;
		EditorSettings::get_singleton()->get_property_list(&editor_settings_properties);

		for (const PropertyInfo &pi : editor_settings_properties) {
			if (pi.name == "interface/editor/editor_language") {
#ifndef ANDROID_ENABLED
				editor_languages = pi.hint_string.split(";", false);
#endif
			} else if (pi.name == "interface/theme/style") {
				editor_styles = pi.hint_string.split(",");
			} else if (pi.name == "interface/theme/color_preset") {
				editor_themes = pi.hint_string.split(",");
			} else if (pi.name == "interface/editor/display_scale") {
				editor_scales = pi.hint_string.split(",");
			} else if (pi.name == "network/connection/network_mode") {
				editor_network_modes = pi.hint_string.split(",");
			} else if (pi.name == "network/connection/check_for_updates") {
				editor_check_for_updates = pi.hint_string.split(",");
			} else if (pi.name == "project_manager/directory_naming_convention") {
				editor_directory_naming_conventions = pi.hint_string.split(",");
			}
		}
	}
}

void QuickSettingsDialog::_update_current_values() {
#ifndef ANDROID_ENABLED
	// Language options.
	{
		const String current_lang = EDITOR_GET("interface/editor/editor_language");

		for (int i = 0; i < editor_languages.size(); i++) {
			const String &lang_value = editor_languages[i].get_slicec('/', 0);
			if (current_lang == lang_value) {
				language_option_button->set_text(editor_languages[i].get_slicec('/', 1));
				language_option_button->select(i);
				break;
			}
		}
	}
#endif
	// Style options.
	{
		const String current_style = EDITOR_GET("interface/theme/style");

		for (int i = 0; i < editor_styles.size(); i++) {
			const String &style_value = editor_styles[i];
			if (current_style == style_value) {
				style_option_button->set_text(current_style);
				style_option_button->select(i);
				style_option_button->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
			}
		}
	}

	// Theme options.
	{
		const String current_theme = EDITOR_GET("interface/theme/color_preset");

		for (int i = 0; i < editor_themes.size(); i++) {
			const String &theme_value = editor_themes[i];
			if (current_theme == theme_value) {
				theme_option_button->set_text(current_theme);
				theme_option_button->select(i);
				theme_option_button->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);

				custom_theme_label->set_visible(current_theme == "Custom");
			}
		}
	}

	// Scale options.
	{
		const int current_scale = EDITOR_GET("interface/editor/display_scale");

		for (int i = 0; i < editor_scales.size(); i++) {
			const String &scale_value = editor_scales[i];
			if (current_scale == i) {
				scale_option_button->set_text(scale_value);
				scale_option_button->select(i);
			}
		}
	}

	// Network mode options.
	{
		const int current_network_mode = EDITOR_GET("network/connection/network_mode");

		for (int i = 0; i < editor_network_modes.size(); i++) {
			const String &network_mode_value = editor_network_modes[i];
			if (current_network_mode == i) {
				network_mode_option_button->set_text(network_mode_value);
				network_mode_option_button->select(i);
			}
		}
	}

	// Check for updates options.
	{
		const int current_update_mode = EDITOR_GET("network/connection/check_for_updates");

		for (int i = 0; i < editor_check_for_updates.size(); i++) {
			const String &check_for_update_value = editor_check_for_updates[i];
			if (current_update_mode == i) {
				check_for_update_button->set_text(check_for_update_value);
				check_for_update_button->select(i);

				// Disables Check for Updates selection if Network mode is set to Offline.
				check_for_update_button->set_disabled(!EDITOR_GET("network/connection/network_mode"));
			}
		}
	}

	// Project directory naming options.
	{
		const int current_directory_naming = EDITOR_GET("project_manager/directory_naming_convention");

		for (int i = 0; i < editor_directory_naming_conventions.size(); i++) {
			const String &directory_naming_value = editor_directory_naming_conventions[i];
			if (current_directory_naming == i) {
				directory_naming_convention_button->set_text(directory_naming_value);
				directory_naming_convention_button->select(i);
			}
		}
	}
}

void QuickSettingsDialog::_add_setting_control(const String &p_text, Control *p_control) {
	HBoxContainer *container = memnew(HBoxContainer);
	settings_list->add_child(container);

	Label *label = memnew(Label(p_text));
	label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	container->add_child(label);

	p_control->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	container->add_child(p_control);
}

#ifndef ANDROID_ENABLED
void QuickSettingsDialog::_language_selected(int p_id) {
	const String selected_language = language_option_button->get_item_metadata(p_id);
	_set_setting_value("interface/editor/editor_language", selected_language);
}
#endif

void QuickSettingsDialog::_style_selected(int p_id) {
	const String selected_style = style_option_button->get_item_text(p_id);
	_set_setting_value("interface/theme/style", selected_style);
}

void QuickSettingsDialog::_theme_selected(int p_id) {
	const String selected_theme = theme_option_button->get_item_text(p_id);
	_set_setting_value("interface/theme/color_preset", selected_theme);

	custom_theme_label->set_visible(selected_theme == "Custom");
}

void QuickSettingsDialog::_scale_selected(int p_id) {
	_set_setting_value("interface/editor/display_scale", p_id, true);
}

void QuickSettingsDialog::_network_mode_selected(int p_id) {
	_set_setting_value("network/connection/network_mode", p_id);

	// Disables Check for Updates selection if Network mode is set to Offline.
	check_for_update_button->set_disabled(!p_id);
}

void QuickSettingsDialog::_check_for_update_selected(int p_id) {
	_set_setting_value("network/connection/check_for_updates", p_id);
}

void QuickSettingsDialog::_directory_naming_convention_selected(int p_id) {
	_set_setting_value("project_manager/directory_naming_convention", p_id);
}

void QuickSettingsDialog::_set_setting_value(const String &p_setting, const Variant &p_value, bool p_restart_required) {
	EditorSettings::get_singleton()->set(p_setting, p_value);
	EditorSettings::get_singleton()->notify_changes();
	EditorSettings::get_singleton()->save();

	if (p_restart_required) {
		restart_required_label->show();

		if (!restart_required_button) {
			int ed_swap_cancel_ok = EDITOR_GET("interface/editor/accept_dialog_cancel_ok_buttons");
			if (ed_swap_cancel_ok == 0) {
				ed_swap_cancel_ok = DisplayServer::get_singleton()->get_swap_cancel_ok() ? 2 : 1;
			}
			restart_required_button = add_button(TTRC("Restart Now"), ed_swap_cancel_ok != 2);
			restart_required_button->connect(SceneStringName(pressed), callable_mp(this, &QuickSettingsDialog::_request_restart));
		}
	}
}

void QuickSettingsDialog::_show_full_settings() {
	if (!editor_settings_dialog) {
		EditorHelp::generate_doc();

		Ref<EditorInspectorDefaultPlugin> eidp;
		eidp.instantiate();
		EditorInspector::add_inspector_plugin(eidp);

		EditorPropertyNameProcessor *epnp = memnew(EditorPropertyNameProcessor);
		add_child(epnp);

		editor_settings_dialog = memnew(EditorSettingsDialog);
		get_parent()->add_child(editor_settings_dialog);
		editor_settings_dialog->connect("restart_requested", callable_mp(this, &QuickSettingsDialog::_request_restart));
	}
	hide();
	editor_settings_dialog->popup_edit_settings();
}

void QuickSettingsDialog::_request_restart() {
	emit_signal("restart_required");
}

void QuickSettingsDialog::update_size_limits(const Size2 &p_max_popup_size) {
#ifndef ANDROID_ENABLED
	language_option_button->get_popup()->set_max_size(p_max_popup_size);
#endif
}

void QuickSettingsDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			settings_list_panel->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("quick_settings_panel"), SNAME("ProjectManager")));

			restart_required_label->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
			custom_theme_label->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("font_placeholder_color"), EditorStringName(Editor)));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				_update_current_values();
			}
		} break;
	}
}

void QuickSettingsDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("restart_required"));
}

QuickSettingsDialog::QuickSettingsDialog() {
	set_title(TTRC("Quick Settings"));
	set_ok_button_text(TTRC("Close"));

	VBoxContainer *main_vbox = memnew(VBoxContainer);
	add_child(main_vbox);
	main_vbox->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

	// Settings grid.
	{
		_fetch_setting_values();

		settings_list_panel = memnew(PanelContainer);
		main_vbox->add_child(settings_list_panel);

		settings_list = memnew(VBoxContainer);
		settings_list_panel->add_child(settings_list);

#ifndef ANDROID_ENABLED
		// Language options.
		{
			language_option_button = memnew(OptionButton);
			language_option_button->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
			language_option_button->set_fit_to_longest_item(false);
			language_option_button->connect(SceneStringName(item_selected), callable_mp(this, &QuickSettingsDialog::_language_selected));

			for (int i = 0; i < editor_languages.size(); i++) {
				const String &lang_code = editor_languages[i].get_slicec('/', 0);
				const String &lang_name = editor_languages[i].get_slicec('/', 1);
				language_option_button->add_item(lang_name, i);
				language_option_button->set_item_metadata(i, lang_code);
			}

			_add_setting_control(TTRC("Language"), language_option_button);
		}
#endif
		// Style options.
		{
			style_option_button = memnew(OptionButton);
			style_option_button->set_fit_to_longest_item(false);
			style_option_button->connect(SceneStringName(item_selected), callable_mp(this, &QuickSettingsDialog::_style_selected));

			for (int i = 0; i < editor_styles.size(); i++) {
				const String &style_value = editor_styles[i];
				style_option_button->add_item(style_value, i);
			}

			_add_setting_control(TTRC("Style"), style_option_button);
		}

		// Theme options.
		{
			theme_option_button = memnew(OptionButton);
			theme_option_button->set_fit_to_longest_item(false);
			theme_option_button->connect(SceneStringName(item_selected), callable_mp(this, &QuickSettingsDialog::_theme_selected));

			for (int i = 0; i < editor_themes.size(); i++) {
				const String &theme_value = editor_themes[i];
				theme_option_button->add_item(theme_value, i);
			}

			_add_setting_control(TTRC("Color Preset"), theme_option_button);

			custom_theme_label = memnew(Label(TTRC("Custom preset can be further configured in the editor.")));
			custom_theme_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_RIGHT);
			custom_theme_label->set_custom_minimum_size(Size2(220, 0) * EDSCALE);
			custom_theme_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
			custom_theme_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			custom_theme_label->set_stretch_ratio(2.0);
			custom_theme_label->hide();
			settings_list->add_child(custom_theme_label);
		}

		// Scale options.
		{
			scale_option_button = memnew(OptionButton);
			scale_option_button->set_fit_to_longest_item(false);
			scale_option_button->connect(SceneStringName(item_selected), callable_mp(this, &QuickSettingsDialog::_scale_selected));

			for (int i = 0; i < editor_scales.size(); i++) {
				const String &scale_value = editor_scales[i];
				scale_option_button->add_item(scale_value, i);
			}

			_add_setting_control(TTRC("Display Scale"), scale_option_button);
		}

		// Network mode options.
		{
			network_mode_option_button = memnew(OptionButton);
			network_mode_option_button->set_fit_to_longest_item(false);
			network_mode_option_button->connect(SceneStringName(item_selected), callable_mp(this, &QuickSettingsDialog::_network_mode_selected));

			for (int i = 0; i < editor_network_modes.size(); i++) {
				const String &network_mode_value = editor_network_modes[i];
				network_mode_option_button->add_item(network_mode_value, i);
			}

			_add_setting_control(TTRC("Network Mode"), network_mode_option_button);
		}

		// Check for updates options.
		{
			check_for_update_button = memnew(OptionButton);
			check_for_update_button->set_fit_to_longest_item(false);
			check_for_update_button->connect(SceneStringName(item_selected), callable_mp(this, &QuickSettingsDialog::_check_for_update_selected));

			for (int i = 0; i < editor_check_for_updates.size(); i++) {
				const String &check_for_update_value = editor_check_for_updates[i];
				check_for_update_button->add_item(check_for_update_value, i);
			}

			_add_setting_control(TTRC("Check for Updates"), check_for_update_button);
		}

		// Project directory naming options.
		{
			directory_naming_convention_button = memnew(OptionButton);
			directory_naming_convention_button->set_fit_to_longest_item(false);
			directory_naming_convention_button->connect(SceneStringName(item_selected), callable_mp(this, &QuickSettingsDialog::_directory_naming_convention_selected));

			for (int i = 0; i < editor_directory_naming_conventions.size(); i++) {
				const String &directory_naming_convention = editor_directory_naming_conventions[i];
				directory_naming_convention_button->add_item(directory_naming_convention, i);
			}

			_add_setting_control(TTRC("Directory Naming Convention"), directory_naming_convention_button);
		}

		_update_current_values();
	}

	// Full settings button.
	{
		Button *open_full_settings = memnew(Button);
		open_full_settings->set_text(TTRC("Edit All Settings"));
		open_full_settings->set_h_size_flags(Control::SIZE_SHRINK_END);
		settings_list->add_child(open_full_settings);
		open_full_settings->connect(SceneStringName(pressed), callable_mp(this, &QuickSettingsDialog::_show_full_settings));
	}

	// Restart required panel.
	{
		restart_required_label = memnew(Label(TTRC("Settings changed! The project manager must be restarted for changes to take effect.")));
		restart_required_label->set_custom_minimum_size(Size2(560, 0) * EDSCALE);
		restart_required_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
		restart_required_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		restart_required_label->hide();
		main_vbox->add_child(restart_required_label);
	}
}
