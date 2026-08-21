/**************************************************************************/
/*  setting_preset_editor.cpp                                             */
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

#include "setting_preset_editor.h"

#include "editor/themes/editor_scale.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/tree.h"
#include "scene/main/viewport.h"

String SettingPresetEditor::get_setting_preset_name(SettingPreset p_preset) {
	switch (p_preset) {
		case SETTING_PRESET_NONE:
			return TTRC("(none)");
		case SETTING_PRESET_2D_PIXEL_ART:
			return vformat("%s - %s", TTRC("2D"), TTRC("Pixel Art"));
		case SETTING_PRESET_2D_HIGH_DEFINITION:
			return vformat("%s - %s", TTRC("2D"), TTRC("High Definition"));
		case SETTING_PRESET_3D_PIXEL_ART:
			return vformat("%s - %s", TTRC("3D"), TTRC("Pixel Art"));
		case SETTING_PRESET_3D_STYLIZED:
			return vformat("%s - %s", TTRC("3D"), TTRC("Stylized"));
		case SETTING_PRESET_3D_HIGH_END:
			return vformat("%s - %s", TTRC("3D"), TTRC("High End"));
		case SETTING_PRESET_MISC_APPLICATION:
			return vformat("%s - %s", TTRC("Misc"), TTRC("Application"));
		case SETTING_PRESET_MAX:
			// Internal value, skip.
			return "";
	};

	return "";
}

String SettingPresetEditor::get_setting_preset_description(SettingPreset p_preset) {
	// Some descriptions are wrapped by hand to fit on two lines with good appearance.
	switch (p_preset) {
		case SETTING_PRESET_NONE:
			return TTRC("Use the default project settings, not tailored for a specific use case.");
		case SETTING_PRESET_2D_PIXEL_ART:
			return TTRC("Optimize project settings for a 2D pixel art project.");
		case SETTING_PRESET_2D_HIGH_DEFINITION:
			return TTRC("Optimize project settings for a high-resolution 2D project.");
		case SETTING_PRESET_3D_PIXEL_ART:
			return TTRC("Optimize project settings for a 3D pixel art project.");
		case SETTING_PRESET_3D_STYLIZED:
			return TTRC("Optimize project settings for a 3D stylized project\nwith simplified, clean visuals.");
		case SETTING_PRESET_3D_HIGH_END:
			return TTRC("Optimize project settings for a 3D project with high-end graphics.\nA dedicated GPU is recommended for this preset.");
		case SETTING_PRESET_MISC_APPLICATION:
			return TTRC("Optimize project settings for a non-game application. This aims to improve system integration and reduce power consumption.");
		case SETTING_PRESET_MAX:
			// Internal value, skip.
			return "";
	};

	return "";
}

StringName SettingPresetEditor::get_setting_preset_icon_name(SettingPreset p_preset) {
	// Some descriptions are wrapped by hand to fit on two lines with good appearance.
	switch (p_preset) {
		case SETTING_PRESET_NONE:
			return SNAME("Node");
		case SETTING_PRESET_2D_PIXEL_ART:
		case SETTING_PRESET_2D_HIGH_DEFINITION:
			return SNAME("Node2D");
		case SETTING_PRESET_3D_PIXEL_ART:
		case SETTING_PRESET_3D_STYLIZED:
		case SETTING_PRESET_3D_HIGH_END:
			return SNAME("Node3D");
		case SETTING_PRESET_MISC_APPLICATION:
			return SNAME("Control");
		case SETTING_PRESET_MAX:
			// Internal value, skip.
			return "";
	};

	return "";
}

HashMap<String, Variant> SettingPresetEditor::get_setting_preset_values(SettingPreset p_preset) {
	// Keep lists of settings alphabetically sorted (with the exception of width/height values).
	HashMap<String, Variant> settings;
	switch (p_preset) {
		case SETTING_PRESET_NONE: {
			// No settings to override.
			break;
		}

		case SETTING_PRESET_2D_PIXEL_ART: {
			settings["display/window/size/viewport_width"] = 640;
			settings["display/window/size/viewport_height"] = 360;
			settings["display/window/size/window_width_override"] = 1280;
			settings["display/window/size/window_height_override"] = 720;
			settings["display/window/stretch/mode"] = "viewport";
			settings["display/window/stretch/scale_mode"] = "integer";
			settings["rendering/2d/snap/snap_2d_transforms_to_pixel"] = true;
			settings["rendering/textures/canvas_textures/default_texture_filter"] = CanvasItem::TEXTURE_FILTER_NEAREST;
		} break;

		case SETTING_PRESET_2D_HIGH_DEFINITION: {
			// Enable high quality VRAM compression in case the user opts into VRAM compression.
			// Lossless compression is still the default for textures used in 2D, but some HD 2D games
			// may wish to use VRAM compression for large backgrounds or complex spritesheets.
			Dictionary dictionary;
			dictionary[StringName("compress/high_quality")] = true;
			settings["importer_defaults/texture"] = dictionary;
		} break;

		case SETTING_PRESET_3D_PIXEL_ART: {
			settings["display/window/size/viewport_width"] = 640;
			settings["display/window/size/viewport_height"] = 360;
			settings["display/window/size/window_width_override"] = 1280;
			settings["display/window/size/window_height_override"] = 720;
			settings["display/window/stretch/scale_mode"] = "integer";
			settings["rendering/2d/snap/snap_2d_transforms_to_pixel"] = true;
			settings["rendering/textures/canvas_textures/default_texture_filter"] = CanvasItem::TEXTURE_FILTER_NEAREST;
			settings["rendering/textures/decals/filter"] = RenderingServerEnums::DECAL_FILTER_NEAREST;
			settings["rendering/textures/light_projectors/filter"] = RenderingServerEnums::LIGHT_PROJECTOR_FILTER_NEAREST;

			// Disable automatic VRAM compression for textures used in 3D, as it looks bad on pixel art.
			// High quality VRAM compression isn't as bad, but we want a lossless result for pixel art
			// since it doesn't use much VRAM in the first place.
			Dictionary dictionary;
			dictionary[StringName("detect_3d/compress_to")] = 0; // Disabled
			settings["importer_defaults/texture"] = dictionary;
		} break;

		case SETTING_PRESET_3D_STYLIZED: {
			settings["rendering/anti_aliasing/quality/msaa_3d"] = Viewport::MSAA_4X;
			settings["rendering/anti_aliasing/quality/use_debanding"] = true;
		} break;

		case SETTING_PRESET_3D_HIGH_END: {
			// Disable FSR sharpening, as we are using FSR2 at native resolution as a high-quality TAA solution.
			settings["rendering/scaling_3d/fsr_sharpness"] = 2.0;
			settings["rendering/scaling_3d/mode"] = Viewport::SCALING_3D_MODE_FSR2;
			settings["rendering/textures/decals/filter"] = RenderingServerEnums::DECAL_FILTER_LINEAR_MIPMAPS_ANISOTROPIC;
			settings["rendering/textures/light_projectors/filter"] = RenderingServerEnums::LIGHT_PROJECTOR_FILTER_LINEAR_MIPMAPS_ANISOTROPIC;
			// Make the project ready for enabling HDR output, without further adjustments needed down the line.
			settings["rendering/viewport/hdr_2d"] = true;

			// Enable high quality VRAM compression.
			Dictionary dictionary;
			dictionary[StringName("compress/high_quality")] = true;
			settings["importer_defaults/texture"] = dictionary;
		} break;

		case SETTING_PRESET_MISC_APPLICATION: {
			settings["application/run/low_processor_mode"] = true;
			settings["display/window/energy_saving/keep_screen_on"] = false;
			settings["display/window/ios/hide_home_indicator"] = false;
			settings["display/window/ios/hide_status_bar"] = false;
			settings["display/window/ios/suppress_ui_gesture"] = false;
			settings["display/window/subwindows/embed_subwindows"] = false;
			// Reduce input lag by disabling V-Sync.
			// Framerate is still capped by low processor mode to avoid excessive CPU/GPU utilization.
			settings["display/window/vsync/vsync_mode"] = DisplayServerEnums::VSYNC_DISABLED;
			settings["input_devices/pointing/android/enable_long_press_as_right_click"] = true;
			settings["input_devices/pointing/android/enable_pan_and_scale_gestures"] = true;
		} break;

		case SETTING_PRESET_MAX: {
			// Internal value, skip.
			break;
		}
	}
	return settings;
}

void SettingPresetEditor::_setting_preset_selected(int p_index) {
	setting_preset_description->set_text(get_setting_preset_description(SettingPreset(setting_preset_selection->get_selected())));

	// Update preview tree.
	setting_preset_preview->clear();
	// Create hidden root item (as all items are displayed at the top level).
	setting_preset_preview->create_item();

	const HashMap<String, Variant> preset_settings = get_setting_preset_values(SettingPreset(p_index));
	for (const KeyValue<String, Variant> &preset_setting : preset_settings) {
		TreeItem *item = setting_preset_preview->create_item();
		// FIXME: EditorPropertyNameProcessor doesn't seem usable here (crashes in the project manager).
		String key = preset_setting.key.replace("/", " > ").capitalize();
		item->set_text(0, key);
		if (preset_setting.value.get_type() == Variant::BOOL) {
			// Always display "On" text even if unchecked, as the icon determines the actual state.
			// This is similar to the Project Settings dialog.
			item->set_text(1, TTRC("On"));
			item->set_icon(1, get_editor_theme_icon(preset_setting.value ? SNAME("GuiCheckedDisabled") : SNAME("GuiUncheckedDisabled")));
		} else {
			item->set_text(1, String(preset_setting.value));
		}
	}

	setting_preset_preview->set_visible(setting_preset_selection->get_selected() != SettingPreset::SETTING_PRESET_NONE);
}

SettingPresetEditor::SettingPreset SettingPresetEditor::get_selected_preset() const {
	return SettingPreset(setting_preset_selection->get_selected());
}

void SettingPresetEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			for (int i = 0; i < SettingPreset::SETTING_PRESET_MAX; i++) {
				setting_preset_selection->set_item_icon(i, get_editor_theme_icon(get_setting_preset_icon_name(SettingPreset(i))));
			}
		} break;
	}
}

SettingPresetEditor::SettingPresetEditor() {
	option_button_container = memnew(HBoxContainer);
	add_child(option_button_container);

	setting_preset_label = memnew(Label);
	setting_preset_label->set_text(TTRC("Optimize For:"));
	option_button_container->add_child(setting_preset_label);

	setting_preset_selection = memnew(OptionButton);
	setting_preset_selection->set_custom_minimum_size(Size2(100, 20));
	for (int i = 0; i < SettingPreset::SETTING_PRESET_MAX; i++) {
		setting_preset_selection->add_item(get_setting_preset_name(SettingPreset(i)), i);
		setting_preset_selection->set_item_tooltip(i, get_setting_preset_description(SettingPreset(i)));
	}
	setting_preset_selection->set_accessibility_name(TTRC("Optimize For:"));
	option_button_container->add_child(setting_preset_selection);

	setting_preset_description = memnew(Label);
	setting_preset_description->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	setting_preset_description->set_text(get_setting_preset_description(SettingPreset(setting_preset_selection->get_selected())));
	setting_preset_description->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	// Set minimum height to roughly two lines of text to avoid reflows.
	setting_preset_description->set_custom_minimum_size(Size2(200, 54) * EDSCALE);
	setting_preset_description->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	setting_preset_description->set_modulate(Color(1, 1, 1, 0.7));
	setting_preset_description->set_text(get_setting_preset_description(SettingPreset(setting_preset_selection->get_selected())));
	add_child(setting_preset_description);
	setting_preset_selection->connect(SceneStringName(item_selected), callable_mp(this, &SettingPresetEditor::_setting_preset_selected));

	setting_preset_preview = memnew(Tree);
	setting_preset_preview->set_columns(2);
	setting_preset_preview->set_column_titles_visible(true);
	setting_preset_preview->set_hide_root(true);
	setting_preset_preview->set_column_title(0, TTRC("Setting"));
	setting_preset_preview->set_column_title(1, TTRC("Value"));
	setting_preset_preview->set_column_expand_ratio(0, 6);
	setting_preset_preview->set_column_expand_ratio(1, 1);
	setting_preset_preview->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_BOTTOM);
	setting_preset_preview->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	setting_preset_preview->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	setting_preset_preview->set_custom_minimum_size(Size2(200, 150) * EDSCALE);
	// Start hidden, and show when a setting preset other than "(none)" is selected.
	setting_preset_preview->hide();

	add_child(setting_preset_preview);
}
