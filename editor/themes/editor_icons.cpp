/**************************************************************************/
/*  editor_icons.cpp                                                      */
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

#include "editor_icons.h"

#include "editor/editor_string_names.h"
#include "editor/themes/editor_color_map.h"
#include "editor/themes/editor_icons.gen.h"
#include "editor/themes/editor_scale.h"
#include "scene/resources/dpi_texture.h"
#include "scene/resources/image_texture.h"

#include "modules/svg/image_loader_svg.h"

void editor_configure_icons(bool p_dark_icon_and_font) {
	if (p_dark_icon_and_font) {
		ImageLoaderSVG::set_forced_color_map(HashMap<Color, Color>());
	} else {
		ImageLoaderSVG::set_forced_color_map(EditorColorMap::get_color_conversion_map());
	}
}

// See also `generate_icon()` in `scene/theme/default_theme.cpp`.
Ref<DPITexture> editor_generate_icon(int p_index, float p_scale, float p_saturation, const Dictionary &p_convert_colors = Dictionary()) {
	return DPITexture::create_from_string(editor_icons_sources[p_index], p_scale, p_saturation, p_convert_colors);
}

float get_gizmo_handle_scale(const String &p_gizmo_handle_name, float p_gizmo_handle_scale) {
	if (p_gizmo_handle_scale > 1.0f) {
		// The names of the icons that require additional scaling.
		static HashSet<StringName> gizmo_to_scale;
		if (gizmo_to_scale.is_empty()) {
			gizmo_to_scale.insert("EditorHandle");
			gizmo_to_scale.insert("EditorHandleAdd");
			gizmo_to_scale.insert("EditorHandleDisabled");
			gizmo_to_scale.insert("EditorCurveHandle");
			gizmo_to_scale.insert("EditorPathSharpHandle");
			gizmo_to_scale.insert("EditorPathSmoothHandle");
			gizmo_to_scale.insert("EditorControlAnchor");
		}

		if (gizmo_to_scale.has(p_gizmo_handle_name)) {
			return EDSCALE * p_gizmo_handle_scale;
		}
	}

	return EDSCALE;
}

void editor_register_icons(const Ref<Theme> &p_theme, bool p_dark_theme, float p_icon_saturation, int p_thumb_size, float p_gizmo_handle_scale) {
	// Before we register the icons, we adjust their colors and saturation.
	// Most icons follow the standard rules for color conversion to follow the editor
	// theme's polarity (dark/light). We also adjust the saturation for most icons,
	// following the editor setting.
	// Some icons are excluded from this conversion, and instead use the configured
	// accent color to replace their innate accent color to match the editor theme.
	// And then some icons are completely excluded from the conversion.

	// Standard color conversion map.
	Dictionary color_conversion_map_light;
	Dictionary color_conversion_map_dark;
	// Icons by default are set up for the dark theme, so if the theme is light,
	// we apply the dark-to-light color conversion map.
	for (KeyValue<Color, Color> &E : EditorColorMap::get_color_conversion_map()) {
		color_conversion_map_light[E.key] = E.value;
	}
	// These colors should be converted even if we are using a dark theme.
	const Color error_color = p_theme->get_color(SNAME("error_color"), EditorStringName(Editor));
	const Color success_color = p_theme->get_color(SNAME("success_color"), EditorStringName(Editor));
	const Color warning_color = p_theme->get_color(SNAME("warning_color"), EditorStringName(Editor));
	color_conversion_map_dark[Color::html("#ff5f5f")] = error_color;
	color_conversion_map_dark[Color::html("#5fff97")] = success_color;
	color_conversion_map_dark[Color::html("#ffdd65")] = warning_color;
	color_conversion_map_light[Color::html("#ff5f5f")] = error_color;
	color_conversion_map_light[Color::html("#5fff97")] = success_color;
	color_conversion_map_light[Color::html("#ffdd65")] = warning_color;

	Dictionary color_conversion_map = p_dark_theme ? color_conversion_map_dark : color_conversion_map_light;

	// The names of the icons to exclude from the standard color conversion.
	const HashSet<StringName> &conversion_exceptions = EditorColorMap::get_color_conversion_exceptions();

	// The names of the icons to exclude when adjusting for saturation.
	HashSet<StringName> saturation_exceptions;
	saturation_exceptions.insert("DefaultProjectIcon");
	saturation_exceptions.insert("Godot");
	saturation_exceptions.insert("Logo");

	// Accent color conversion map.
	// It is used on some icons (checkbox, radio, toggle, etc.), regardless of the dark
	// or light mode.
	Dictionary accent_color_map;
	HashSet<StringName> accent_color_icons;

	const Color accent_color = p_theme->get_color(SNAME("accent_color"), EditorStringName(Editor));
	accent_color_map[Color::html("699ce8")] = accent_color;
	if (accent_color.get_luminance() > 0.75) {
		accent_color_map[Color::html("ffffff")] = Color(0.2, 0.2, 0.2);
	}

	accent_color_icons.insert("GuiChecked");
	accent_color_icons.insert("GuiRadioChecked");
	accent_color_icons.insert("GuiIndeterminate");
	accent_color_icons.insert("GuiToggleOn");
	accent_color_icons.insert("GuiToggleOnMirrored");
	accent_color_icons.insert("PlayOverlay");

	// Generate icons.
	{
		for (int i = 0; i < editor_icons_count; i++) {
			const String &editor_icon_name = editor_icons_names[i];
			Ref<DPITexture> icon;
			if (accent_color_icons.has(editor_icon_name)) {
				icon = editor_generate_icon(i, get_gizmo_handle_scale(editor_icon_name, p_gizmo_handle_scale), 1.0, accent_color_map);
			} else {
				float saturation = p_icon_saturation;
				if (saturation_exceptions.has(editor_icon_name)) {
					saturation = 1.0;
				}

				if (conversion_exceptions.has(editor_icon_name)) {
					icon = editor_generate_icon(i, get_gizmo_handle_scale(editor_icon_name, p_gizmo_handle_scale), saturation);
				} else {
					icon = editor_generate_icon(i, get_gizmo_handle_scale(editor_icon_name, p_gizmo_handle_scale), saturation, color_conversion_map);
				}
			}

			p_theme->set_icon(editor_icon_name, EditorStringName(EditorIcons), icon);
		}
	}

	// Generate thumbnail icons with the given thumbnail size.
	// See editor\icons\editor_icons_builders.py for the code that determines which icons are thumbnails.
	if (p_thumb_size >= 64) {
		const float scale = (float)p_thumb_size / 64.0 * EDSCALE;
		for (int i = 0; i < editor_bg_thumbs_count; i++) {
			const int index = editor_bg_thumbs_indices[i];
			Ref<DPITexture> icon;

			if (accent_color_icons.has(editor_icons_names[index])) {
				icon = editor_generate_icon(index, scale, 1.0, accent_color_map);
			} else {
				float saturation = p_icon_saturation;
				if (saturation_exceptions.has(editor_icons_names[index])) {
					saturation = 1.0;
				}

				if (conversion_exceptions.has(editor_icons_names[index])) {
					icon = editor_generate_icon(index, scale, saturation);
				} else {
					icon = editor_generate_icon(index, scale, saturation, color_conversion_map);
				}
			}

			p_theme->set_icon(editor_icons_names[index], EditorStringName(EditorIcons), icon);
		}
	} else {
		const float scale = (float)p_thumb_size / 32.0 * EDSCALE;
		for (int i = 0; i < editor_md_thumbs_count; i++) {
			const int index = editor_md_thumbs_indices[i];
			Ref<DPITexture> icon;

			if (accent_color_icons.has(editor_icons_names[index])) {
				icon = editor_generate_icon(index, scale, 1.0, accent_color_map);
			} else {
				float saturation = p_icon_saturation;
				if (saturation_exceptions.has(editor_icons_names[index])) {
					saturation = 1.0;
				}

				if (conversion_exceptions.has(editor_icons_names[index])) {
					icon = editor_generate_icon(index, scale, saturation);
				} else {
					icon = editor_generate_icon(index, scale, saturation, color_conversion_map);
				}
			}

			p_theme->set_icon(editor_icons_names[index], EditorStringName(EditorIcons), icon);
		}
	}
}

void editor_copy_icons(const Ref<Theme> &p_theme, const Ref<Theme> &p_old_theme) {
	for (int i = 0; i < editor_icons_count; i++) {
		p_theme->set_icon(editor_icons_names[i], EditorStringName(EditorIcons), p_old_theme->get_icon(editor_icons_names[i], EditorStringName(EditorIcons)));
	}
}

// Returns the SVG code for the default project icon.
String get_default_project_icon() {
	// FIXME: This icon can probably be predefined in editor_icons.gen.h so we don't have to look up.
	for (int i = 0; i < editor_icons_count; i++) {
		if (strcmp(editor_icons_names[i], "DefaultProjectIcon") == 0) {
			return String(editor_icons_sources[i]);
		}
	}
	return String();
}
