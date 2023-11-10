/**************************************************************************/
/*  editor_themes.cpp                                                     */
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

#include "editor_themes.h"

#include "core/error/error_macros.h"
#include "core/io/resource_loader.h"
#include "editor/editor_fonts.h"
#include "editor/editor_icons.gen.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/style_box_flat.h"
#include "scene/resources/style_box_line.h"
#include "scene/resources/style_box_texture.h"
#include "scene/theme/theme_db.h"

#include "modules/modules_enabled.gen.h" // For svg.
#ifdef MODULE_SVG_ENABLED
#include "modules/svg/image_loader_svg.h"
#endif

HashMap<Color, Color> EditorColorMap::color_conversion_map;
HashSet<StringName> EditorColorMap::color_conversion_exceptions;

void EditorColorMap::add_conversion_color_pair(const String p_from_color, const String p_to_color) {
	color_conversion_map[Color::html(p_from_color)] = Color::html(p_to_color);
}

void EditorColorMap::add_conversion_exception(const StringName &p_icon_name) {
	color_conversion_exceptions.insert(p_icon_name);
}

void EditorColorMap::create() {
	// Some of the colors below are listed for completeness sake.
	// This can be a basis for proper palette validation later.

	// Convert:               FROM       TO
	add_conversion_color_pair("#478cbf", "#478cbf"); // Godot Blue
	add_conversion_color_pair("#414042", "#414042"); // Godot Gray

	add_conversion_color_pair("#ffffff", "#414141"); // Pure white
	add_conversion_color_pair("#fefefe", "#fefefe"); // Forced light color
	add_conversion_color_pair("#000000", "#bfbfbf"); // Pure black
	add_conversion_color_pair("#010101", "#010101"); // Forced dark color

	// Keep pure RGB colors as is, but list them for explicitness.
	add_conversion_color_pair("#ff0000", "#ff0000"); // Pure red
	add_conversion_color_pair("#00ff00", "#00ff00"); // Pure green
	add_conversion_color_pair("#0000ff", "#0000ff"); // Pure blue

	// GUI Colors
	add_conversion_color_pair("#e0e0e0", "#5a5a5a"); // Common icon color
	add_conversion_color_pair("#808080", "#808080"); // GUI disabled color
	add_conversion_color_pair("#b3b3b3", "#363636"); // GUI disabled light color
	add_conversion_color_pair("#699ce8", "#699ce8"); // GUI highlight color
	add_conversion_color_pair("#f9f9f9", "#606060"); // Scrollbar grabber highlight color

	add_conversion_color_pair("#c38ef1", "#a85de9"); // Animation
	add_conversion_color_pair("#8da5f3", "#3d64dd"); // 2D
	add_conversion_color_pair("#7582a8", "#6d83c8"); // 2D Abstract
	add_conversion_color_pair("#fc7f7f", "#cd3838"); // 3D
	add_conversion_color_pair("#b56d6d", "#be6a6a"); // 3D Abstract
	add_conversion_color_pair("#8eef97", "#2fa139"); // GUI Control
	add_conversion_color_pair("#76ad7b", "#64a66a"); // GUI Control Abstract

	add_conversion_color_pair("#5fb2ff", "#0079f0"); // Selection (blue)
	add_conversion_color_pair("#003e7a", "#2b74bb"); // Selection (darker blue)
	add_conversion_color_pair("#f7f5cf", "#615f3a"); // Gizmo (yellow)

	// Rainbow
	add_conversion_color_pair("#ff4545", "#ff2929"); // Red
	add_conversion_color_pair("#ffe345", "#ffe337"); // Yellow
	add_conversion_color_pair("#80ff45", "#74ff34"); // Green
	add_conversion_color_pair("#45ffa2", "#2cff98"); // Aqua
	add_conversion_color_pair("#45d7ff", "#22ccff"); // Blue
	add_conversion_color_pair("#8045ff", "#702aff"); // Purple
	add_conversion_color_pair("#ff4596", "#ff2781"); // Pink

	// Audio gradients
	add_conversion_color_pair("#e1da5b", "#d6cf4b"); // Yellow

	add_conversion_color_pair("#62aeff", "#1678e0"); // Frozen gradient top
	add_conversion_color_pair("#75d1e6", "#41acc5"); // Frozen gradient middle
	add_conversion_color_pair("#84ffee", "#49ccba"); // Frozen gradient bottom

	add_conversion_color_pair("#f70000", "#c91616"); // Color track red
	add_conversion_color_pair("#eec315", "#d58c0b"); // Color track orange
	add_conversion_color_pair("#dbee15", "#b7d10a"); // Color track yellow
	add_conversion_color_pair("#288027", "#218309"); // Color track green

	// Other objects
	add_conversion_color_pair("#ffca5f", "#fea900"); // Mesh resource (orange)
	add_conversion_color_pair("#2998ff", "#68b6ff"); // Shape resource (blue)
	add_conversion_color_pair("#a2d2ff", "#4998e3"); // Shape resource (light blue)
	add_conversion_color_pair("#69c4d4", "#29a3cc"); // Input event highlight (light blue)

	// Animation editor tracks
	// The property track icon color is set by the common icon color.
	add_conversion_color_pair("#ea7940", "#bd5e2c"); // 3D Position track
	add_conversion_color_pair("#ff2b88", "#bd165f"); // 3D Rotation track
	add_conversion_color_pair("#eac840", "#bd9d1f"); // 3D Scale track
	add_conversion_color_pair("#3cf34e", "#16a827"); // Call Method track
	add_conversion_color_pair("#2877f6", "#236be6"); // Bezier Curve track
	add_conversion_color_pair("#eae440", "#9f9722"); // Audio Playback track
	add_conversion_color_pair("#a448f0", "#9853ce"); // Animation Playback track
	add_conversion_color_pair("#5ad5c4", "#0a9c88"); // Blend Shape track

	// Control layouts
	add_conversion_color_pair("#d6d6d6", "#474747"); // Highlighted part
	add_conversion_color_pair("#474747", "#d6d6d6"); // Background part
	add_conversion_color_pair("#919191", "#6e6e6e"); // Border part

	// TileSet editor icons
	add_conversion_color_pair("#fce00e", "#aa8d24"); // New Single Tile
	add_conversion_color_pair("#0e71fc", "#0350bd"); // New Autotile
	add_conversion_color_pair("#c6ced4", "#828f9b"); // New Atlas

	// Variant types
	add_conversion_color_pair("#41ecad", "#25e3a0"); // Variant
	add_conversion_color_pair("#6f91f0", "#6d8eeb"); // bool
	add_conversion_color_pair("#5abbef", "#4fb2e9"); // int/uint
	add_conversion_color_pair("#35d4f4", "#27ccf0"); // float
	add_conversion_color_pair("#4593ec", "#4690e7"); // String
	add_conversion_color_pair("#ee5677", "#ee7991"); // AABB
	add_conversion_color_pair("#e0e0e0", "#5a5a5a"); // Array
	add_conversion_color_pair("#e1ec41", "#b2bb19"); // Basis
	add_conversion_color_pair("#54ed9e", "#57e99f"); // Dictionary
	add_conversion_color_pair("#417aec", "#6993ec"); // NodePath
	add_conversion_color_pair("#55f3e3", "#12d5c3"); // Object
	add_conversion_color_pair("#f74949", "#f77070"); // Plane
	add_conversion_color_pair("#44bd44", "#46b946"); // Projection
	add_conversion_color_pair("#ec418e", "#ec69a3"); // Quaternion
	add_conversion_color_pair("#f1738f", "#ee758e"); // Rect2
	add_conversion_color_pair("#41ec80", "#2ce573"); // RID
	add_conversion_color_pair("#b9ec41", "#96ce1a"); // Transform2D
	add_conversion_color_pair("#f68f45", "#f49047"); // Transform3D
	add_conversion_color_pair("#ac73f1", "#ad76ee"); // Vector2
	add_conversion_color_pair("#de66f0", "#dc6aed"); // Vector3
	add_conversion_color_pair("#f066bd", "#ed6abd"); // Vector4

	// Visual shaders
	add_conversion_color_pair("#77ce57", "#67c046"); // Vector funcs
	add_conversion_color_pair("#ea686c", "#d95256"); // Vector transforms
	add_conversion_color_pair("#eac968", "#d9b64f"); // Textures and cubemaps
	add_conversion_color_pair("#cf68ea", "#c050dd"); // Functions and expressions

	// These icons should not be converted.
	add_conversion_exception("EditorPivot");
	add_conversion_exception("EditorHandle");
	add_conversion_exception("Editor3DHandle");
	add_conversion_exception("EditorBoneHandle");
	add_conversion_exception("Godot");
	add_conversion_exception("Sky");
	add_conversion_exception("EditorControlAnchor");
	add_conversion_exception("DefaultProjectIcon");
	add_conversion_exception("ZoomMore");
	add_conversion_exception("ZoomLess");
	add_conversion_exception("ZoomReset");
	add_conversion_exception("LockViewport");
	add_conversion_exception("GroupViewport");
	add_conversion_exception("StatusError");
	add_conversion_exception("StatusSuccess");
	add_conversion_exception("StatusWarning");
	add_conversion_exception("OverbrightIndicator");
	add_conversion_exception("MaterialPreviewCube");
	add_conversion_exception("MaterialPreviewSphere");
	add_conversion_exception("MaterialPreviewLight1");
	add_conversion_exception("MaterialPreviewLight2");

	// GUI
	add_conversion_exception("GuiChecked");
	add_conversion_exception("GuiRadioChecked");
	add_conversion_exception("GuiIndeterminate");
	add_conversion_exception("GuiCloseCustomizable");
	add_conversion_exception("GuiGraphNodePort");
	add_conversion_exception("GuiResizer");
	add_conversion_exception("GuiMiniCheckerboard");

	/// Code Editor.
	add_conversion_exception("GuiTab");
	add_conversion_exception("GuiSpace");
	add_conversion_exception("CodeFoldedRightArrow");
	add_conversion_exception("CodeFoldDownArrow");
	add_conversion_exception("CodeRegionFoldedRightArrow");
	add_conversion_exception("CodeRegionFoldDownArrow");
	add_conversion_exception("TextEditorPlay");
	add_conversion_exception("Breakpoint");
}

void EditorColorMap::finish() {
	color_conversion_map.clear();
	color_conversion_exceptions.clear();
}

Vector<StringName> EditorTheme::editor_theme_types;

// TODO: Refactor these and corresponding Theme methods to use the bool get_xxx(r_value) pattern internally.

// Keep in sync with Theme::get_color.
Color EditorTheme::get_color(const StringName &p_name, const StringName &p_theme_type) const {
	if (color_map.has(p_theme_type) && color_map[p_theme_type].has(p_name)) {
		return color_map[p_theme_type][p_name];
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme color '%s' in '%s'.", p_name, p_theme_type));
		}
		return Color();
	}
}

// Keep in sync with Theme::get_constant.
int EditorTheme::get_constant(const StringName &p_name, const StringName &p_theme_type) const {
	if (constant_map.has(p_theme_type) && constant_map[p_theme_type].has(p_name)) {
		return constant_map[p_theme_type][p_name];
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme constant '%s' in '%s'.", p_name, p_theme_type));
		}
		return 0;
	}
}

// Keep in sync with Theme::get_font.
Ref<Font> EditorTheme::get_font(const StringName &p_name, const StringName &p_theme_type) const {
	if (font_map.has(p_theme_type) && font_map[p_theme_type].has(p_name) && font_map[p_theme_type][p_name].is_valid()) {
		return font_map[p_theme_type][p_name];
	} else if (has_default_font()) {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme font '%s' in '%s'.", p_name, p_theme_type));
		}
		return default_font;
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme font '%s' in '%s'.", p_name, p_theme_type));
		}
		return ThemeDB::get_singleton()->get_fallback_font();
	}
}

// Keep in sync with Theme::get_font_size.
int EditorTheme::get_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	if (font_size_map.has(p_theme_type) && font_size_map[p_theme_type].has(p_name) && (font_size_map[p_theme_type][p_name] > 0)) {
		return font_size_map[p_theme_type][p_name];
	} else if (has_default_font_size()) {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme font size '%s' in '%s'.", p_name, p_theme_type));
		}
		return default_font_size;
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme font size '%s' in '%s'.", p_name, p_theme_type));
		}
		return ThemeDB::get_singleton()->get_fallback_font_size();
	}
}

// Keep in sync with Theme::get_icon.
Ref<Texture2D> EditorTheme::get_icon(const StringName &p_name, const StringName &p_theme_type) const {
	if (icon_map.has(p_theme_type) && icon_map[p_theme_type].has(p_name) && icon_map[p_theme_type][p_name].is_valid()) {
		return icon_map[p_theme_type][p_name];
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme icon '%s' in '%s'.", p_name, p_theme_type));
		}
		return ThemeDB::get_singleton()->get_fallback_icon();
	}
}

// Keep in sync with Theme::get_stylebox.
Ref<StyleBox> EditorTheme::get_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	if (style_map.has(p_theme_type) && style_map[p_theme_type].has(p_name) && style_map[p_theme_type][p_name].is_valid()) {
		return style_map[p_theme_type][p_name];
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme stylebox '%s' in '%s'.", p_name, p_theme_type));
		}
		return ThemeDB::get_singleton()->get_fallback_stylebox();
	}
}

void EditorTheme::initialize() {
	editor_theme_types.append(EditorStringName(Editor));
	editor_theme_types.append(EditorStringName(EditorFonts));
	editor_theme_types.append(EditorStringName(EditorIcons));
	editor_theme_types.append(EditorStringName(EditorStyles));
}

void EditorTheme::finalize() {
	editor_theme_types.clear();
}

// Editor theme generatior.

static Ref<StyleBoxTexture> make_stylebox(Ref<Texture2D> p_texture, float p_left, float p_top, float p_right, float p_bottom, float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1, bool p_draw_center = true) {
	Ref<StyleBoxTexture> style(memnew(StyleBoxTexture));
	style->set_texture(p_texture);
	style->set_texture_margin_individual(p_left * EDSCALE, p_top * EDSCALE, p_right * EDSCALE, p_bottom * EDSCALE);
	style->set_content_margin_individual((p_left + p_margin_left) * EDSCALE, (p_top + p_margin_top) * EDSCALE, (p_right + p_margin_right) * EDSCALE, (p_bottom + p_margin_bottom) * EDSCALE);
	style->set_draw_center(p_draw_center);
	return style;
}

static Ref<StyleBoxEmpty> make_empty_stylebox(float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1) {
	Ref<StyleBoxEmpty> style(memnew(StyleBoxEmpty));
	style->set_content_margin_individual(p_margin_left * EDSCALE, p_margin_top * EDSCALE, p_margin_right * EDSCALE, p_margin_bottom * EDSCALE);
	return style;
}

static Ref<StyleBoxFlat> make_flat_stylebox(Color p_color, float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1, int p_corner_width = 0) {
	Ref<StyleBoxFlat> style(memnew(StyleBoxFlat));
	style->set_bg_color(p_color);
	// Adjust level of detail based on the corners' effective sizes.
	style->set_corner_detail(Math::ceil(0.8 * p_corner_width * EDSCALE));
	style->set_corner_radius_all(p_corner_width * EDSCALE);
	style->set_content_margin_individual(p_margin_left * EDSCALE, p_margin_top * EDSCALE, p_margin_right * EDSCALE, p_margin_bottom * EDSCALE);
	// Work around issue about antialiased edges being blurrier (GH-35279).
	style->set_anti_aliased(false);
	return style;
}

static Ref<StyleBoxLine> make_line_stylebox(Color p_color, int p_thickness = 1, float p_grow_begin = 1, float p_grow_end = 1, bool p_vertical = false) {
	Ref<StyleBoxLine> style(memnew(StyleBoxLine));
	style->set_color(p_color);
	style->set_grow_begin(p_grow_begin);
	style->set_grow_end(p_grow_end);
	style->set_thickness(p_thickness);
	style->set_vertical(p_vertical);
	return style;
}

// See also `generate_icon()` in `scene/theme/default_theme.cpp`.
static Ref<ImageTexture> editor_generate_icon(int p_index, float p_scale, float p_saturation, const HashMap<Color, Color> &p_convert_colors = HashMap<Color, Color>()) {
	Ref<Image> img = memnew(Image);

#ifdef MODULE_SVG_ENABLED
	// Upsample icon generation only if the editor scale isn't an integer multiplier.
	// Generating upsampled icons is slower, and the benefit is hardly visible
	// with integer editor scales.
	const bool upsample = !Math::is_equal_approx(Math::round(p_scale), p_scale);
	Error err = ImageLoaderSVG::create_image_from_string(img, editor_icons_sources[p_index], p_scale, upsample, p_convert_colors);
	ERR_FAIL_COND_V_MSG(err != OK, Ref<ImageTexture>(), "Failed generating icon, unsupported or invalid SVG data in editor theme.");
	if (p_saturation != 1.0) {
		img->adjust_bcs(1.0, 1.0, p_saturation);
	}
#else
	// If the SVG module is disabled, we can't really display the UI well, but at least we won't crash.
	// 16 pixels is used as it's the most common base size for Godot icons.
	img = Image::create_empty(16 * p_scale, 16 * p_scale, false, Image::FORMAT_RGBA8);
#endif

	return ImageTexture::create_from_image(img);
}

float get_gizmo_handle_scale(const String &gizmo_handle_name = "") {
	const float scale_gizmo_handles_for_touch = EDITOR_GET("interface/touchscreen/scale_gizmo_handles");
	if (scale_gizmo_handles_for_touch > 1.0f) {
		// The names of the icons that require additional scaling.
		static HashSet<StringName> gizmo_to_scale;
		if (gizmo_to_scale.is_empty()) {
			gizmo_to_scale.insert("EditorHandle");
			gizmo_to_scale.insert("EditorHandleAdd");
			gizmo_to_scale.insert("EditorHandleDisabled");
			gizmo_to_scale.insert("EditorCurveHandle");
			gizmo_to_scale.insert("EditorPathSharpHandle");
			gizmo_to_scale.insert("EditorPathSmoothHandle");
		}

		if (gizmo_to_scale.has(gizmo_handle_name)) {
			return EDSCALE * scale_gizmo_handles_for_touch;
		}
	}

	return EDSCALE;
}

void editor_register_and_generate_icons(Ref<Theme> p_theme, bool p_dark_theme, float p_icon_saturation, int p_thumb_size, bool p_only_thumbs = false) {
	OS::get_singleton()->benchmark_begin_measure("editor_register_and_generate_icons_" + String((p_only_thumbs ? "with_only_thumbs" : "all")));
	// Before we register the icons, we adjust their colors and saturation.
	// Most icons follow the standard rules for color conversion to follow the editor
	// theme's polarity (dark/light). We also adjust the saturation for most icons,
	// following the editor setting.
	// Some icons are excluded from this conversion, and instead use the configured
	// accent color to replace their innate accent color to match the editor theme.
	// And then some icons are completely excluded from the conversion.

	// Standard color conversion map.
	HashMap<Color, Color> color_conversion_map;
	// Icons by default are set up for the dark theme, so if the theme is light,
	// we apply the dark-to-light color conversion map.
	if (!p_dark_theme) {
		for (KeyValue<Color, Color> &E : EditorColorMap::get_color_conversion_map()) {
			color_conversion_map[E.key] = E.value;
		}
	}
	// These colors should be converted even if we are using a dark theme.
	const Color error_color = p_theme->get_color(SNAME("error_color"), EditorStringName(Editor));
	const Color success_color = p_theme->get_color(SNAME("success_color"), EditorStringName(Editor));
	const Color warning_color = p_theme->get_color(SNAME("warning_color"), EditorStringName(Editor));
	color_conversion_map[Color::html("#ff5f5f")] = error_color;
	color_conversion_map[Color::html("#5fff97")] = success_color;
	color_conversion_map[Color::html("#ffdd65")] = warning_color;

	// The names of the icons to exclude from the standard color conversion.
	HashSet<StringName> conversion_exceptions = EditorColorMap::get_color_conversion_exceptions();

	// The names of the icons to exclude when adjusting for saturation.
	HashSet<StringName> saturation_exceptions;
	saturation_exceptions.insert("DefaultProjectIcon");
	saturation_exceptions.insert("Godot");
	saturation_exceptions.insert("Logo");

	// Accent color conversion map.
	// It is used on some icons (checkbox, radio, toggle, etc.), regardless of the dark
	// or light mode.
	HashMap<Color, Color> accent_color_map;
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
	if (!p_only_thumbs) {
		for (int i = 0; i < editor_icons_count; i++) {
			Ref<ImageTexture> icon;

			const String &editor_icon_name = editor_icons_names[i];
			if (accent_color_icons.has(editor_icon_name)) {
				icon = editor_generate_icon(i, get_gizmo_handle_scale(editor_icon_name), 1.0, accent_color_map);
			} else {
				float saturation = p_icon_saturation;
				if (saturation_exceptions.has(editor_icon_name)) {
					saturation = 1.0;
				}

				if (conversion_exceptions.has(editor_icon_name)) {
					icon = editor_generate_icon(i, get_gizmo_handle_scale(editor_icon_name), saturation);
				} else {
					icon = editor_generate_icon(i, get_gizmo_handle_scale(editor_icon_name), saturation, color_conversion_map);
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
			Ref<ImageTexture> icon;

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
			Ref<ImageTexture> icon;

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
	OS::get_singleton()->benchmark_end_measure("editor_register_and_generate_icons_" + String((p_only_thumbs ? "with_only_thumbs" : "all")));
}

Ref<Theme> create_editor_theme(const Ref<Theme> p_theme) {
	OS::get_singleton()->benchmark_begin_measure("create_editor_theme");
	Ref<EditorTheme> theme = memnew(EditorTheme);

	// Controls may rely on the scale for their internal drawing logic.
	theme->set_default_base_scale(EDSCALE);

	// Theme settings
	Color accent_color = EDITOR_GET("interface/theme/accent_color");
	Color base_color = EDITOR_GET("interface/theme/base_color");
	float contrast = EDITOR_GET("interface/theme/contrast");
	bool increase_scrollbar_touch_area = EDITOR_GET("interface/touchscreen/increase_scrollbar_touch_area");
	const float gizmo_handle_scale = EDITOR_GET("interface/touchscreen/scale_gizmo_handles");
	bool draw_extra_borders = EDITOR_GET("interface/theme/draw_extra_borders");
	float icon_saturation = EDITOR_GET("interface/theme/icon_saturation");
	float relationship_line_opacity = EDITOR_GET("interface/theme/relationship_line_opacity");

	String preset = EDITOR_GET("interface/theme/preset");

	int border_size = EDITOR_GET("interface/theme/border_size");
	int corner_radius = EDITOR_GET("interface/theme/corner_radius");

	Color preset_accent_color;
	Color preset_base_color;
	float preset_contrast = 0;
	bool preset_draw_extra_borders = false;

	const float default_contrast = 0.3;

	// Please use alphabetical order if you're adding a new theme here
	// (after "Custom")

	if (preset == "Custom") {
		accent_color = EDITOR_GET("interface/theme/accent_color");
		base_color = EDITOR_GET("interface/theme/base_color");
		contrast = EDITOR_GET("interface/theme/contrast");
	} else if (preset == "Breeze Dark") {
		preset_accent_color = Color(0.26, 0.76, 1.00);
		preset_base_color = Color(0.24, 0.26, 0.28);
		preset_contrast = default_contrast;
	} else if (preset == "Godot 2") {
		preset_accent_color = Color(0.53, 0.67, 0.89);
		preset_base_color = Color(0.24, 0.23, 0.27);
		preset_contrast = default_contrast;
	} else if (preset == "Gray") {
		preset_accent_color = Color(0.44, 0.73, 0.98);
		preset_base_color = Color(0.24, 0.24, 0.24);
		preset_contrast = default_contrast;
	} else if (preset == "Light") {
		preset_accent_color = Color(0.18, 0.50, 1.00);
		preset_base_color = Color(0.9, 0.9, 0.9);
		// A negative contrast rate looks better for light themes, since it better follows the natural order of UI "elevation".
		preset_contrast = -0.06;
	} else if (preset == "Solarized (Dark)") {
		preset_accent_color = Color(0.15, 0.55, 0.82);
		preset_base_color = Color(0.04, 0.23, 0.27);
		preset_contrast = default_contrast;
	} else if (preset == "Solarized (Light)") {
		preset_accent_color = Color(0.15, 0.55, 0.82);
		preset_base_color = Color(0.89, 0.86, 0.79);
		// A negative contrast rate looks better for light themes, since it better follows the natural order of UI "elevation".
		preset_contrast = -0.06;
	} else if (preset == "Black (OLED)") {
		preset_accent_color = Color(0.45, 0.75, 1.0);
		preset_base_color = Color(0, 0, 0);
		// The contrast rate value is irrelevant on a fully black theme.
		preset_contrast = 0.0;
		preset_draw_extra_borders = true;
	} else { // Default
		preset_accent_color = Color(0.44, 0.73, 0.98);
		preset_base_color = Color(0.21, 0.24, 0.29);
		preset_contrast = default_contrast;
	}

	if (preset != "Custom") {
		accent_color = preset_accent_color;
		base_color = preset_base_color;
		contrast = preset_contrast;
		draw_extra_borders = preset_draw_extra_borders;
		EditorSettings::get_singleton()->set_initial_value("interface/theme/accent_color", accent_color);
		EditorSettings::get_singleton()->set_initial_value("interface/theme/base_color", base_color);
		EditorSettings::get_singleton()->set_initial_value("interface/theme/contrast", contrast);
		EditorSettings::get_singleton()->set_initial_value("interface/theme/draw_extra_borders", draw_extra_borders);
	}

	EditorSettings::get_singleton()->set_manually("interface/theme/preset", preset);
	EditorSettings::get_singleton()->set_manually("interface/theme/accent_color", accent_color);
	EditorSettings::get_singleton()->set_manually("interface/theme/base_color", base_color);
	EditorSettings::get_singleton()->set_manually("interface/theme/contrast", contrast);
	EditorSettings::get_singleton()->set_manually("interface/theme/draw_extra_borders", draw_extra_borders);

	// Colors
	bool dark_theme = EditorSettings::get_singleton()->is_dark_theme();

#ifdef MODULE_SVG_ENABLED
	if (dark_theme) {
		ImageLoaderSVG::set_forced_color_map(HashMap<Color, Color>());
	} else {
		ImageLoaderSVG::set_forced_color_map(EditorColorMap::get_color_conversion_map());
	}
#endif

	// Ensure base colors are in the 0..1 luminance range to avoid 8-bit integer overflow or text rendering issues.
	// Some places in the editor use 8-bit integer colors.
	const Color dark_color_1 = base_color.lerp(Color(0, 0, 0, 1), contrast).clamp();
	const Color dark_color_2 = base_color.lerp(Color(0, 0, 0, 1), contrast * 1.5).clamp();
	const Color dark_color_3 = base_color.lerp(Color(0, 0, 0, 1), contrast * 2).clamp();

	// Only used when the Draw Extra Borders editor setting is enabled.
	const Color extra_border_color_1 = Color(0.5, 0.5, 0.5);
	const Color extra_border_color_2 = dark_theme ? Color(0.3, 0.3, 0.3) : Color(0.7, 0.7, 0.7);

	const Color background_color = dark_color_2;

	// White (dark theme) or black (light theme), will be used to generate the rest of the colors
	const Color mono_color = dark_theme ? Color(1, 1, 1) : Color(0, 0, 0);

	const Color contrast_color_1 = base_color.lerp(mono_color, MAX(contrast, default_contrast));
	const Color contrast_color_2 = base_color.lerp(mono_color, MAX(contrast * 1.5, default_contrast * 1.5));

	const Color font_color = mono_color.lerp(base_color, 0.25);
	const Color font_hover_color = mono_color.lerp(base_color, 0.125);
	const Color font_focus_color = mono_color.lerp(base_color, 0.125);
	const Color font_hover_pressed_color = font_hover_color.lerp(accent_color, 0.74);
	const Color font_disabled_color = Color(mono_color.r, mono_color.g, mono_color.b, 0.35);
	const Color font_readonly_color = Color(mono_color.r, mono_color.g, mono_color.b, 0.65);
	const Color font_placeholder_color = Color(mono_color.r, mono_color.g, mono_color.b, 0.6);
	const Color font_outline_color = Color(0, 0, 0, 0);
	const Color selection_color = accent_color * Color(1, 1, 1, 0.4);
	const Color disabled_color = mono_color.inverted().lerp(base_color, 0.7);
	const Color disabled_bg_color = mono_color.inverted().lerp(base_color, 0.9);

	const Color icon_normal_color = Color(1, 1, 1);
	Color icon_hover_color = icon_normal_color * (dark_theme ? 1.15 : 1.45);
	icon_hover_color.a = 1.0;
	Color icon_focus_color = icon_hover_color;
	Color icon_disabled_color = Color(icon_normal_color, 0.4);
	// Make the pressed icon color overbright because icons are not completely white on a dark theme.
	// On a light theme, icons are dark, so we need to modulate them with an even brighter color.
	Color icon_pressed_color = accent_color * (dark_theme ? 1.15 : 3.5);
	icon_pressed_color.a = 1.0;

	const Color separator_color = Color(mono_color.r, mono_color.g, mono_color.b, 0.1);
	const Color highlight_color = Color(accent_color.r, accent_color.g, accent_color.b, 0.275);
	const Color disabled_highlight_color = highlight_color.lerp(dark_theme ? Color(0, 0, 0) : Color(1, 1, 1), 0.5);

	// Can't save single float in theme, so using Color.
	theme->set_color("icon_saturation", EditorStringName(Editor), Color(icon_saturation, icon_saturation, icon_saturation));
	theme->set_color("accent_color", EditorStringName(Editor), accent_color);
	theme->set_color("highlight_color", EditorStringName(Editor), highlight_color);
	theme->set_color("disabled_highlight_color", EditorStringName(Editor), disabled_highlight_color);
	theme->set_color("base_color", EditorStringName(Editor), base_color);
	theme->set_color("dark_color_1", EditorStringName(Editor), dark_color_1);
	theme->set_color("dark_color_2", EditorStringName(Editor), dark_color_2);
	theme->set_color("dark_color_3", EditorStringName(Editor), dark_color_3);
	theme->set_color("contrast_color_1", EditorStringName(Editor), contrast_color_1);
	theme->set_color("contrast_color_2", EditorStringName(Editor), contrast_color_2);
	theme->set_color("box_selection_fill_color", EditorStringName(Editor), accent_color * Color(1, 1, 1, 0.3));
	theme->set_color("box_selection_stroke_color", EditorStringName(Editor), accent_color * Color(1, 1, 1, 0.8));

	theme->set_color("axis_x_color", EditorStringName(Editor), Color(0.96, 0.20, 0.32));
	theme->set_color("axis_y_color", EditorStringName(Editor), Color(0.53, 0.84, 0.01));
	theme->set_color("axis_z_color", EditorStringName(Editor), Color(0.16, 0.55, 0.96));
	theme->set_color("axis_w_color", EditorStringName(Editor), Color(0.55, 0.55, 0.55));

	const float prop_color_saturation = accent_color.get_s() * 0.75;
	const float prop_color_value = accent_color.get_v();

	theme->set_color("property_color_x", EditorStringName(Editor), Color().from_hsv(0.0 / 3.0 + 0.05, prop_color_saturation, prop_color_value));
	theme->set_color("property_color_y", EditorStringName(Editor), Color().from_hsv(1.0 / 3.0 + 0.05, prop_color_saturation, prop_color_value));
	theme->set_color("property_color_z", EditorStringName(Editor), Color().from_hsv(2.0 / 3.0 + 0.05, prop_color_saturation, prop_color_value));
	theme->set_color("property_color_w", EditorStringName(Editor), Color().from_hsv(1.5 / 3.0 + 0.05, prop_color_saturation, prop_color_value));

	theme->set_color("font_color", EditorStringName(Editor), font_color);
	theme->set_color("highlighted_font_color", EditorStringName(Editor), font_hover_color);
	theme->set_color("disabled_font_color", EditorStringName(Editor), font_disabled_color);
	theme->set_color("readonly_font_color", EditorStringName(Editor), font_readonly_color);

	theme->set_color("mono_color", EditorStringName(Editor), mono_color);

	Color success_color = Color(0.45, 0.95, 0.5);
	Color warning_color = Color(1, 0.87, 0.4);
	Color error_color = Color(1, 0.47, 0.42);
	Color property_color = font_color.lerp(Color(0.5, 0.5, 0.5), 0.5);
	Color readonly_color = property_color.lerp(dark_theme ? Color(0, 0, 0) : Color(1, 1, 1), 0.25);
	Color readonly_warning_color = error_color.lerp(dark_theme ? Color(0, 0, 0) : Color(1, 1, 1), 0.25);

	if (!dark_theme) {
		// Darken some colors to be readable on a light background.
		success_color = success_color.lerp(mono_color, 0.35);
		warning_color = warning_color.lerp(mono_color, 0.35);
		error_color = error_color.lerp(mono_color, 0.25);
	}

	theme->set_color("success_color", EditorStringName(Editor), success_color);
	theme->set_color("warning_color", EditorStringName(Editor), warning_color);
	theme->set_color("error_color", EditorStringName(Editor), error_color);
	theme->set_color("property_color", EditorStringName(Editor), property_color);
	theme->set_color("readonly_color", EditorStringName(Editor), readonly_color);

	if (!dark_theme) {
		theme->set_color("highend_color", EditorStringName(Editor), Color::hex(0xad1128ff));
	} else {
		theme->set_color("highend_color", EditorStringName(Editor), Color(1.0, 0.0, 0.0));
	}

	const int thumb_size = EDITOR_GET("filesystem/file_dialog/thumbnail_size");
	theme->set_constant("scale", EditorStringName(Editor), EDSCALE);
	theme->set_constant("thumb_size", EditorStringName(Editor), thumb_size);
	theme->set_constant("class_icon_size", EditorStringName(Editor), 16 * EDSCALE);
	theme->set_constant("dark_theme", EditorStringName(Editor), dark_theme);
	theme->set_constant("color_picker_button_height", EditorStringName(Editor), 28 * EDSCALE);
	theme->set_constant("gizmo_handle_scale", EditorStringName(Editor), gizmo_handle_scale);
	theme->set_constant("window_border_margin", EditorStringName(Editor), 8);
	theme->set_constant("top_bar_separation", EditorStringName(Editor), 8 * EDSCALE);

	// Register editor icons.
	// If the settings are comparable to the old theme, then just copy them over.
	// Otherwise, regenerate them. Also check if we need to regenerate "thumb" icons.
	bool keep_old_icons = false;
	bool regenerate_thumb_icons = true;
	if (p_theme != nullptr) {
		// We check editor scale, theme dark/light mode, icon saturation, and accent color.

		// That doesn't really work as expected, since theme constants are integers, and scales are floats.
		// So this check will never work when changing between 100-199% values.
		const float prev_scale = (float)p_theme->get_constant(SNAME("scale"), EditorStringName(Editor));
		const bool prev_dark_theme = (bool)p_theme->get_constant(SNAME("dark_theme"), EditorStringName(Editor));
		const Color prev_accent_color = p_theme->get_color(SNAME("accent_color"), EditorStringName(Editor));
		const float prev_icon_saturation = p_theme->get_color(SNAME("icon_saturation"), EditorStringName(Editor)).r;
		const float prev_gizmo_handle_scale = (float)p_theme->get_constant(SNAME("gizmo_handle_scale"), EditorStringName(Editor));

		keep_old_icons = (Math::is_equal_approx(prev_scale, EDSCALE) &&
				Math::is_equal_approx(prev_gizmo_handle_scale, gizmo_handle_scale) &&
				prev_dark_theme == dark_theme &&
				prev_accent_color == accent_color &&
				prev_icon_saturation == icon_saturation);

		const double prev_thumb_size = (double)p_theme->get_constant(SNAME("thumb_size"), EditorStringName(Editor));

		regenerate_thumb_icons = !Math::is_equal_approx(prev_thumb_size, thumb_size);
	}

#ifndef MODULE_SVG_ENABLED
	WARN_PRINT("SVG support disabled, editor icons won't be rendered.");
#endif

	if (keep_old_icons) {
		for (int i = 0; i < editor_icons_count; i++) {
			theme->set_icon(editor_icons_names[i], EditorStringName(EditorIcons), p_theme->get_icon(editor_icons_names[i], EditorStringName(EditorIcons)));
		}
	} else {
		editor_register_and_generate_icons(theme, dark_theme, icon_saturation, thumb_size, false);
	}
	if (regenerate_thumb_icons) {
		editor_register_and_generate_icons(theme, dark_theme, icon_saturation, thumb_size, true);
	}

	// Register editor fonts.
	editor_register_fonts(theme);

	// Ensure borders are visible when using an editor scale below 100%.
	const int border_width = CLAMP(border_size, 0, 2) * MAX(1, EDSCALE);
	const int corner_width = CLAMP(corner_radius, 0, 6);
	const int default_margin_size = 4;
	const int margin_size_extra = default_margin_size + CLAMP(border_size, 0, 2);

	// Styleboxes
	// This is the most commonly used stylebox, variations should be made as duplicate of this
	Ref<StyleBoxFlat> style_default = make_flat_stylebox(base_color, default_margin_size, default_margin_size, default_margin_size, default_margin_size, corner_width);
	style_default->set_border_width_all(border_width);
	style_default->set_border_color(base_color);

	// Button and widgets
	const float extra_spacing = EDITOR_GET("interface/theme/additional_spacing");

	const Vector2 widget_default_margin = Vector2(extra_spacing + 6, extra_spacing + default_margin_size + 1) * EDSCALE;

	Ref<StyleBoxFlat> style_widget = style_default->duplicate();
	style_widget->set_content_margin_individual(widget_default_margin.x, widget_default_margin.y, widget_default_margin.x, widget_default_margin.y);
	style_widget->set_bg_color(dark_color_1);
	if (draw_extra_borders) {
		style_widget->set_border_width_all(Math::round(EDSCALE));
		style_widget->set_border_color(extra_border_color_1);
	} else {
		style_widget->set_border_color(dark_color_2);
	}

	Ref<StyleBoxFlat> style_widget_disabled = style_widget->duplicate();
	if (draw_extra_borders) {
		style_widget_disabled->set_border_color(extra_border_color_2);
	} else {
		style_widget_disabled->set_border_color(disabled_color);
	}
	style_widget_disabled->set_bg_color(disabled_bg_color);

	Ref<StyleBoxFlat> style_widget_focus = style_widget->duplicate();
	style_widget_focus->set_draw_center(false);
	style_widget_focus->set_border_width_all(Math::round(2 * MAX(1, EDSCALE)));
	style_widget_focus->set_border_color(accent_color);

	Ref<StyleBoxFlat> style_widget_pressed = style_widget->duplicate();
	style_widget_pressed->set_bg_color(dark_color_1.darkened(0.125));

	Ref<StyleBoxFlat> style_widget_hover = style_widget->duplicate();
	style_widget_hover->set_bg_color(mono_color * Color(1, 1, 1, 0.11));
	if (draw_extra_borders) {
		style_widget_hover->set_border_color(extra_border_color_1);
	} else {
		style_widget_hover->set_border_color(mono_color * Color(1, 1, 1, 0.05));
	}

	// Style for windows, popups, etc..
	Ref<StyleBoxFlat> style_popup = style_default->duplicate();
	const int popup_margin_size = default_margin_size * EDSCALE * 3;
	style_popup->set_content_margin_all(popup_margin_size);
	style_popup->set_border_color(contrast_color_1);
	const Color shadow_color = Color(0, 0, 0, dark_theme ? 0.3 : 0.1);
	style_popup->set_shadow_color(shadow_color);
	style_popup->set_shadow_size(4 * EDSCALE);
	// Popups are separate windows by default in the editor. Windows currently don't support per-pixel transparency
	// in 4.0, and even if it was, it may not always work in practice (e.g. running with compositing disabled).
	style_popup->set_corner_radius_all(0);

	Ref<StyleBoxLine> style_popup_separator(memnew(StyleBoxLine));
	style_popup_separator->set_color(separator_color);
	style_popup_separator->set_grow_begin(popup_margin_size - MAX(Math::round(EDSCALE), border_width));
	style_popup_separator->set_grow_end(popup_margin_size - MAX(Math::round(EDSCALE), border_width));
	style_popup_separator->set_thickness(MAX(Math::round(EDSCALE), border_width));

	Ref<StyleBoxLine> style_popup_labeled_separator_left(memnew(StyleBoxLine));
	style_popup_labeled_separator_left->set_grow_begin(popup_margin_size - MAX(Math::round(EDSCALE), border_width));
	style_popup_labeled_separator_left->set_color(separator_color);
	style_popup_labeled_separator_left->set_thickness(MAX(Math::round(EDSCALE), border_width));

	Ref<StyleBoxLine> style_popup_labeled_separator_right(memnew(StyleBoxLine));
	style_popup_labeled_separator_right->set_grow_end(popup_margin_size - MAX(Math::round(EDSCALE), border_width));
	style_popup_labeled_separator_right->set_color(separator_color);
	style_popup_labeled_separator_right->set_thickness(MAX(Math::round(EDSCALE), border_width));

	Ref<StyleBoxEmpty> style_empty = make_empty_stylebox(default_margin_size, default_margin_size, default_margin_size, default_margin_size);

	// TabBar

	Ref<StyleBoxFlat> style_tab_base = style_widget->duplicate();

	style_tab_base->set_border_width_all(0);
	// Don't round the top corners to avoid creating a small blank space between the tabs and the main panel.
	// This also makes the top highlight look better.
	style_tab_base->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
	style_tab_base->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);

	// When using a border width greater than 0, visually line up the left of the selected tab with the underlying panel.
	style_tab_base->set_expand_margin(SIDE_LEFT, -border_width);

	style_tab_base->set_content_margin(SIDE_LEFT, widget_default_margin.x + 5 * EDSCALE);
	style_tab_base->set_content_margin(SIDE_RIGHT, widget_default_margin.x + 5 * EDSCALE);
	style_tab_base->set_content_margin(SIDE_BOTTOM, widget_default_margin.y);
	style_tab_base->set_content_margin(SIDE_TOP, widget_default_margin.y);

	Ref<StyleBoxFlat> style_tab_selected = style_tab_base->duplicate();

	style_tab_selected->set_bg_color(base_color);
	// Add a highlight line at the top of the selected tab.
	style_tab_selected->set_border_width(SIDE_TOP, Math::round(2 * EDSCALE));
	// Make the highlight line prominent, but not too prominent as to not be distracting.
	Color tab_highlight = dark_color_2.lerp(accent_color, 0.75);
	style_tab_selected->set_border_color(tab_highlight);
	style_tab_selected->set_corner_radius_all(0);

	Ref<StyleBoxFlat> style_tab_hovered = style_tab_base->duplicate();

	style_tab_hovered->set_bg_color(dark_color_1.lerp(base_color, 0.4));
	// Hovered tab has a subtle highlight between normal and selected states.
	style_tab_hovered->set_corner_radius_all(0);

	Ref<StyleBoxFlat> style_tab_unselected = style_tab_base->duplicate();
	style_tab_unselected->set_expand_margin(SIDE_BOTTOM, 0);
	style_tab_unselected->set_bg_color(dark_color_1);
	// Add some spacing between unselected tabs to make them easier to distinguish from each other
	style_tab_unselected->set_border_color(Color(0, 0, 0, 0));

	Ref<StyleBoxFlat> style_tab_disabled = style_tab_base->duplicate();
	style_tab_disabled->set_expand_margin(SIDE_BOTTOM, 0);
	style_tab_disabled->set_bg_color(disabled_bg_color);
	style_tab_disabled->set_border_color(disabled_bg_color);

	Ref<StyleBoxFlat> style_tab_focus = style_widget_focus->duplicate();

	// Editor background
	Color background_color_opaque = background_color;
	background_color_opaque.a = 1.0;
	theme->set_color("background", EditorStringName(Editor), background_color_opaque);
	theme->set_stylebox("Background", EditorStringName(EditorStyles), make_flat_stylebox(background_color_opaque, default_margin_size, default_margin_size, default_margin_size, default_margin_size));

	// Focus
	theme->set_stylebox("Focus", EditorStringName(EditorStyles), style_widget_focus);
	// Use a less opaque color to be less distracting for the 2D and 3D editor viewports.
	Ref<StyleBoxFlat> style_widget_focus_viewport = style_widget_focus->duplicate();
	style_widget_focus_viewport->set_border_color(accent_color * Color(1, 1, 1, 0.5));
	theme->set_stylebox("FocusViewport", EditorStringName(EditorStyles), style_widget_focus_viewport);

	// Menu
	Ref<StyleBoxFlat> style_menu = style_widget->duplicate();
	style_menu->set_draw_center(false);
	style_menu->set_border_width_all(0);
	theme->set_stylebox("panel", "PanelContainer", style_menu);
	theme->set_stylebox("MenuPanel", EditorStringName(EditorStyles), style_menu);

	// CanvasItem Editor
	Ref<StyleBoxFlat> style_canvas_editor_info = make_flat_stylebox(Color(0.0, 0.0, 0.0, 0.2));
	style_canvas_editor_info->set_expand_margin_all(4 * EDSCALE);
	theme->set_stylebox("CanvasItemInfoOverlay", EditorStringName(EditorStyles), style_canvas_editor_info);

	// 2D and 3D contextual toolbar.
	// Use a custom stylebox to make contextual menu items stand out from the rest.
	// This helps with editor usability as contextual menu items change when selecting nodes,
	// even though it may not be immediately obvious at first.
	Ref<StyleBoxFlat> toolbar_stylebox = memnew(StyleBoxFlat);
	toolbar_stylebox->set_bg_color(accent_color * Color(1, 1, 1, 0.1));
	toolbar_stylebox->set_anti_aliased(false);
	// Add an underline to the StyleBox, but prevent its minimum vertical size from changing.
	toolbar_stylebox->set_border_color(accent_color);
	toolbar_stylebox->set_border_width(SIDE_BOTTOM, Math::round(2 * EDSCALE));
	toolbar_stylebox->set_content_margin(SIDE_BOTTOM, 0);
	toolbar_stylebox->set_expand_margin_individual(4 * EDSCALE, 2 * EDSCALE, 4 * EDSCALE, 4 * EDSCALE);
	theme->set_stylebox("ContextualToolbar", EditorStringName(EditorStyles), toolbar_stylebox);

	// Script Editor
	theme->set_stylebox("ScriptEditorPanel", EditorStringName(EditorStyles), make_empty_stylebox(default_margin_size, 0, default_margin_size, default_margin_size));
	theme->set_stylebox("ScriptEditorPanelFloating", EditorStringName(EditorStyles), make_empty_stylebox(0, 0, 0, 0));

	theme->set_stylebox("ScriptEditor", EditorStringName(EditorStyles), make_empty_stylebox(0, 0, 0, 0));

	// Launch Pad and Play buttons
	Ref<StyleBoxFlat> style_launch_pad = make_flat_stylebox(dark_color_1, 2 * EDSCALE, 0, 2 * EDSCALE, 0, corner_width);
	style_launch_pad->set_corner_radius_all(corner_radius * EDSCALE);
	theme->set_stylebox("LaunchPadNormal", EditorStringName(EditorStyles), style_launch_pad);
	Ref<StyleBoxFlat> style_launch_pad_movie = style_launch_pad->duplicate();
	style_launch_pad_movie->set_bg_color(accent_color * Color(1, 1, 1, 0.1));
	style_launch_pad_movie->set_border_color(accent_color);
	style_launch_pad_movie->set_border_width_all(Math::round(2 * EDSCALE));
	theme->set_stylebox("LaunchPadMovieMode", EditorStringName(EditorStyles), style_launch_pad_movie);

	theme->set_stylebox("MovieWriterButtonNormal", EditorStringName(EditorStyles), make_empty_stylebox(0, 0, 0, 0));
	Ref<StyleBoxFlat> style_write_movie_button = style_widget_pressed->duplicate();
	style_write_movie_button->set_bg_color(accent_color);
	style_write_movie_button->set_corner_radius_all(corner_radius * EDSCALE);
	style_write_movie_button->set_content_margin(SIDE_TOP, 0);
	style_write_movie_button->set_content_margin(SIDE_BOTTOM, 0);
	style_write_movie_button->set_content_margin(SIDE_LEFT, 0);
	style_write_movie_button->set_content_margin(SIDE_RIGHT, 0);
	style_write_movie_button->set_expand_margin(SIDE_RIGHT, 2 * EDSCALE);
	theme->set_stylebox("MovieWriterButtonPressed", EditorStringName(EditorStyles), style_write_movie_button);

	// MenuButton
	theme->set_stylebox("normal", "MenuButton", style_menu);
	theme->set_stylebox("hover", "MenuButton", style_widget_hover);
	theme->set_stylebox("pressed", "MenuButton", style_menu);
	theme->set_stylebox("focus", "MenuButton", style_menu);
	theme->set_stylebox("disabled", "MenuButton", style_menu);

	theme->set_color("font_color", "MenuButton", font_color);
	theme->set_color("font_hover_color", "MenuButton", font_hover_color);
	theme->set_color("font_hover_pressed_color", "MenuButton", font_hover_pressed_color);
	theme->set_color("font_focus_color", "MenuButton", font_focus_color);
	theme->set_color("font_outline_color", "MenuButton", font_outline_color);

	theme->set_constant("outline_size", "MenuButton", 0);

	theme->set_stylebox("MenuHover", EditorStringName(EditorStyles), style_widget_hover);

	// Buttons
	theme->set_stylebox("normal", "Button", style_widget);
	theme->set_stylebox("hover", "Button", style_widget_hover);
	theme->set_stylebox("pressed", "Button", style_widget_pressed);
	theme->set_stylebox("focus", "Button", style_widget_focus);
	theme->set_stylebox("disabled", "Button", style_widget_disabled);

	theme->set_color("font_color", "Button", font_color);
	theme->set_color("font_hover_color", "Button", font_hover_color);
	theme->set_color("font_hover_pressed_color", "Button", font_hover_pressed_color);
	theme->set_color("font_focus_color", "Button", font_focus_color);
	theme->set_color("font_pressed_color", "Button", accent_color);
	theme->set_color("font_disabled_color", "Button", font_disabled_color);
	theme->set_color("font_outline_color", "Button", font_outline_color);

	theme->set_color("icon_normal_color", "Button", icon_normal_color);
	theme->set_color("icon_hover_color", "Button", icon_hover_color);
	theme->set_color("icon_focus_color", "Button", icon_focus_color);
	theme->set_color("icon_pressed_color", "Button", icon_pressed_color);
	theme->set_color("icon_disabled_color", "Button", icon_disabled_color);

	theme->set_constant("h_separation", "Button", 4 * EDSCALE);
	theme->set_constant("outline_size", "Button", 0);

	// Flat button variations.

	Ref<StyleBoxEmpty> style_flat_button = make_empty_stylebox();
	for (int i = 0; i < 4; i++) {
		style_flat_button->set_content_margin((Side)i, style_widget->get_margin((Side)i) + style_widget->get_border_width((Side)i));
	}

	Ref<StyleBoxFlat> style_flat_button_pressed = style_widget_pressed->duplicate();
	Color flat_pressed_color = dark_color_1.lightened(0.24).lerp(accent_color, 0.2) * Color(0.8, 0.8, 0.8, 0.85);
	if (dark_theme) {
		flat_pressed_color = dark_color_1.lerp(accent_color, 0.12) * Color(0.6, 0.6, 0.6, 0.85);
	}
	style_flat_button_pressed->set_bg_color(flat_pressed_color);

	theme->set_stylebox("normal", "FlatButton", style_flat_button);
	theme->set_stylebox("hover", "FlatButton", style_flat_button);
	theme->set_stylebox("pressed", "FlatButton", style_flat_button_pressed);
	theme->set_stylebox("disabled", "FlatButton", style_flat_button);

	theme->set_stylebox("normal", "FlatMenuButton", style_flat_button);
	theme->set_stylebox("hover", "FlatMenuButton", style_flat_button);
	theme->set_stylebox("pressed", "FlatMenuButton", style_flat_button_pressed);
	theme->set_stylebox("disabled", "FlatMenuButton", style_flat_button);

	const float ACTION_BUTTON_EXTRA_MARGIN = 32 * EDSCALE;

	theme->set_type_variation("InspectorActionButton", "Button");
	Color color_inspector_action = dark_color_1.lerp(mono_color, 0.12);
	color_inspector_action.a = 0.5;
	Ref<StyleBoxFlat> style_inspector_action = style_widget->duplicate();
	style_inspector_action->set_bg_color(color_inspector_action);
	style_inspector_action->set_content_margin(SIDE_RIGHT, ACTION_BUTTON_EXTRA_MARGIN);
	theme->set_stylebox("normal", "InspectorActionButton", style_inspector_action);
	style_inspector_action = style_widget_hover->duplicate();
	style_inspector_action->set_content_margin(SIDE_RIGHT, ACTION_BUTTON_EXTRA_MARGIN);
	theme->set_stylebox("hover", "InspectorActionButton", style_inspector_action);
	style_inspector_action = style_widget_pressed->duplicate();
	style_inspector_action->set_content_margin(SIDE_RIGHT, ACTION_BUTTON_EXTRA_MARGIN);
	theme->set_stylebox("pressed", "InspectorActionButton", style_inspector_action);
	style_inspector_action = style_widget_disabled->duplicate();
	style_inspector_action->set_content_margin(SIDE_RIGHT, ACTION_BUTTON_EXTRA_MARGIN);
	theme->set_stylebox("disabled", "InspectorActionButton", style_inspector_action);
	theme->set_constant("h_separation", "InspectorActionButton", ACTION_BUTTON_EXTRA_MARGIN);

	// Variation for Editor Log filter buttons
	theme->set_type_variation("EditorLogFilterButton", "Button");
	// When pressed, don't tint the icons with the accent color, just leave them normal.
	theme->set_color("icon_pressed_color", "EditorLogFilterButton", icon_normal_color);
	// When unpressed, dim the icons.
	theme->set_color("icon_normal_color", "EditorLogFilterButton", icon_disabled_color);
	// When pressed, add a small bottom border to the buttons to better show their active state,
	// similar to active tabs.

	Ref<StyleBoxFlat> editor_log_button_pressed = style_flat_button_pressed->duplicate();
	editor_log_button_pressed->set_border_width(SIDE_BOTTOM, 2 * EDSCALE);
	editor_log_button_pressed->set_border_color(accent_color);
	theme->set_stylebox("pressed", "EditorLogFilterButton", editor_log_button_pressed);

	// Buttons in material previews
	const Color dim_light_color = icon_normal_color.darkened(0.24);
	const Color dim_light_highlighted_color = icon_normal_color.darkened(0.18);
	Ref<StyleBox> sb_empty_borderless = make_empty_stylebox();

	theme->set_type_variation("PreviewLightButton", "Button");
	// When pressed, don't use the accent color tint. When unpressed, dim the icon.
	theme->set_color("icon_normal_color", "PreviewLightButton", dim_light_color);
	theme->set_color("icon_focus_color", "PreviewLightButton", dim_light_color);
	theme->set_color("icon_pressed_color", "PreviewLightButton", icon_normal_color);
	theme->set_color("icon_hover_pressed_color", "PreviewLightButton", icon_normal_color);
	// Unpressed icon is dim, so use a dim highlight.
	theme->set_color("icon_hover_color", "PreviewLightButton", dim_light_highlighted_color);

	theme->set_stylebox("normal", "PreviewLightButton", sb_empty_borderless);
	theme->set_stylebox("hover", "PreviewLightButton", sb_empty_borderless);
	theme->set_stylebox("focus", "PreviewLightButton", sb_empty_borderless);
	theme->set_stylebox("pressed", "PreviewLightButton", sb_empty_borderless);

	// ProjectTag
	{
		theme->set_type_variation("ProjectTag", "Button");

		Ref<StyleBoxFlat> tag = style_widget->duplicate();
		tag->set_bg_color(dark_theme ? tag->get_bg_color().lightened(0.2) : tag->get_bg_color().darkened(0.2));
		tag->set_corner_radius(CORNER_TOP_LEFT, 0);
		tag->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
		tag->set_corner_radius(CORNER_TOP_RIGHT, 4);
		tag->set_corner_radius(CORNER_BOTTOM_RIGHT, 4);
		theme->set_stylebox("normal", "ProjectTag", tag);

		tag = style_widget_hover->duplicate();
		tag->set_corner_radius(CORNER_TOP_LEFT, 0);
		tag->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
		tag->set_corner_radius(CORNER_TOP_RIGHT, 4);
		tag->set_corner_radius(CORNER_BOTTOM_RIGHT, 4);
		theme->set_stylebox("hover", "ProjectTag", tag);

		tag = style_widget_pressed->duplicate();
		tag->set_corner_radius(CORNER_TOP_LEFT, 0);
		tag->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
		tag->set_corner_radius(CORNER_TOP_RIGHT, 4);
		tag->set_corner_radius(CORNER_BOTTOM_RIGHT, 4);
		theme->set_stylebox("pressed", "ProjectTag", tag);
	}

	// MenuBar
	theme->set_stylebox("normal", "MenuBar", style_widget);
	theme->set_stylebox("hover", "MenuBar", style_widget_hover);
	theme->set_stylebox("pressed", "MenuBar", style_widget_pressed);
	theme->set_stylebox("disabled", "MenuBar", style_widget_disabled);

	theme->set_color("font_color", "MenuBar", font_color);
	theme->set_color("font_hover_color", "MenuBar", font_hover_color);
	theme->set_color("font_hover_pressed_color", "MenuBar", font_hover_pressed_color);
	theme->set_color("font_focus_color", "MenuBar", font_focus_color);
	theme->set_color("font_pressed_color", "MenuBar", accent_color);
	theme->set_color("font_disabled_color", "MenuBar", font_disabled_color);
	theme->set_color("font_outline_color", "MenuBar", font_outline_color);

	theme->set_color("icon_normal_color", "MenuBar", icon_normal_color);
	theme->set_color("icon_hover_color", "MenuBar", icon_hover_color);
	theme->set_color("icon_focus_color", "MenuBar", icon_focus_color);
	theme->set_color("icon_pressed_color", "MenuBar", icon_pressed_color);
	theme->set_color("icon_disabled_color", "MenuBar", icon_disabled_color);

	theme->set_constant("h_separation", "MenuBar", 4 * EDSCALE);
	theme->set_constant("outline_size", "MenuBar", 0);

	// OptionButton
	Ref<StyleBoxFlat> style_option_button_focus = style_widget_focus->duplicate();
	Ref<StyleBoxFlat> style_option_button_normal = style_widget->duplicate();
	Ref<StyleBoxFlat> style_option_button_hover = style_widget_hover->duplicate();
	Ref<StyleBoxFlat> style_option_button_pressed = style_widget_pressed->duplicate();
	Ref<StyleBoxFlat> style_option_button_disabled = style_widget_disabled->duplicate();

	style_option_button_focus->set_content_margin(SIDE_RIGHT, 4 * EDSCALE);
	style_option_button_normal->set_content_margin(SIDE_RIGHT, 4 * EDSCALE);
	style_option_button_hover->set_content_margin(SIDE_RIGHT, 4 * EDSCALE);
	style_option_button_pressed->set_content_margin(SIDE_RIGHT, 4 * EDSCALE);
	style_option_button_disabled->set_content_margin(SIDE_RIGHT, 4 * EDSCALE);

	theme->set_stylebox("focus", "OptionButton", style_option_button_focus);
	theme->set_stylebox("normal", "OptionButton", style_widget);
	theme->set_stylebox("hover", "OptionButton", style_widget_hover);
	theme->set_stylebox("pressed", "OptionButton", style_widget_pressed);
	theme->set_stylebox("disabled", "OptionButton", style_widget_disabled);

	theme->set_stylebox("normal_mirrored", "OptionButton", style_option_button_normal);
	theme->set_stylebox("hover_mirrored", "OptionButton", style_option_button_hover);
	theme->set_stylebox("pressed_mirrored", "OptionButton", style_option_button_pressed);
	theme->set_stylebox("disabled_mirrored", "OptionButton", style_option_button_disabled);

	theme->set_color("font_color", "OptionButton", font_color);
	theme->set_color("font_hover_color", "OptionButton", font_hover_color);
	theme->set_color("font_hover_pressed_color", "OptionButton", font_hover_pressed_color);
	theme->set_color("font_focus_color", "OptionButton", font_focus_color);
	theme->set_color("font_pressed_color", "OptionButton", accent_color);
	theme->set_color("font_disabled_color", "OptionButton", font_disabled_color);
	theme->set_color("font_outline_color", "OptionButton", font_outline_color);

	theme->set_color("icon_normal_color", "OptionButton", icon_normal_color);
	theme->set_color("icon_hover_color", "OptionButton", icon_hover_color);
	theme->set_color("icon_focus_color", "OptionButton", icon_focus_color);
	theme->set_color("icon_pressed_color", "OptionButton", icon_pressed_color);
	theme->set_color("icon_disabled_color", "OptionButton", icon_disabled_color);

	theme->set_icon("arrow", "OptionButton", theme->get_icon(SNAME("GuiOptionArrow"), EditorStringName(EditorIcons)));
	theme->set_constant("arrow_margin", "OptionButton", widget_default_margin.x - 2 * EDSCALE);
	theme->set_constant("modulate_arrow", "OptionButton", true);
	theme->set_constant("h_separation", "OptionButton", 4 * EDSCALE);
	theme->set_constant("outline_size", "OptionButton", 0);

	// CheckButton
	theme->set_stylebox("normal", "CheckButton", style_menu);
	theme->set_stylebox("pressed", "CheckButton", style_menu);
	theme->set_stylebox("disabled", "CheckButton", style_menu);
	theme->set_stylebox("hover", "CheckButton", style_menu);
	theme->set_stylebox("hover_pressed", "CheckButton", style_menu);

	theme->set_icon("checked", "CheckButton", theme->get_icon(SNAME("GuiToggleOn"), EditorStringName(EditorIcons)));
	theme->set_icon("checked_disabled", "CheckButton", theme->get_icon(SNAME("GuiToggleOnDisabled"), EditorStringName(EditorIcons)));
	theme->set_icon("unchecked", "CheckButton", theme->get_icon(SNAME("GuiToggleOff"), EditorStringName(EditorIcons)));
	theme->set_icon("unchecked_disabled", "CheckButton", theme->get_icon(SNAME("GuiToggleOffDisabled"), EditorStringName(EditorIcons)));

	theme->set_icon("checked_mirrored", "CheckButton", theme->get_icon(SNAME("GuiToggleOnMirrored"), EditorStringName(EditorIcons)));
	theme->set_icon("checked_disabled_mirrored", "CheckButton", theme->get_icon(SNAME("GuiToggleOnDisabledMirrored"), EditorStringName(EditorIcons)));
	theme->set_icon("unchecked_mirrored", "CheckButton", theme->get_icon(SNAME("GuiToggleOffMirrored"), EditorStringName(EditorIcons)));
	theme->set_icon("unchecked_disabled_mirrored", "CheckButton", theme->get_icon(SNAME("GuiToggleOffDisabledMirrored"), EditorStringName(EditorIcons)));

	theme->set_color("font_color", "CheckButton", font_color);
	theme->set_color("font_hover_color", "CheckButton", font_hover_color);
	theme->set_color("font_hover_pressed_color", "CheckButton", font_hover_pressed_color);
	theme->set_color("font_focus_color", "CheckButton", font_focus_color);
	theme->set_color("font_pressed_color", "CheckButton", accent_color);
	theme->set_color("font_disabled_color", "CheckButton", font_disabled_color);
	theme->set_color("font_outline_color", "CheckButton", font_outline_color);

	theme->set_color("icon_normal_color", "CheckButton", icon_normal_color);
	theme->set_color("icon_hover_color", "CheckButton", icon_hover_color);
	theme->set_color("icon_focus_color", "CheckButton", icon_focus_color);
	theme->set_color("icon_pressed_color", "CheckButton", icon_pressed_color);
	theme->set_color("icon_disabled_color", "CheckButton", icon_disabled_color);

	theme->set_constant("h_separation", "CheckButton", 8 * EDSCALE);
	theme->set_constant("check_v_offset", "CheckButton", 0);
	theme->set_constant("outline_size", "CheckButton", 0);

	// Checkbox
	Ref<StyleBoxFlat> sb_checkbox = style_menu->duplicate();
	sb_checkbox->set_content_margin_all(default_margin_size * EDSCALE);

	theme->set_stylebox("normal", "CheckBox", sb_checkbox);
	theme->set_stylebox("pressed", "CheckBox", sb_checkbox);
	theme->set_stylebox("disabled", "CheckBox", sb_checkbox);
	theme->set_stylebox("hover", "CheckBox", sb_checkbox);
	theme->set_stylebox("hover_pressed", "CheckBox", sb_checkbox);
	theme->set_icon("checked", "CheckBox", theme->get_icon(SNAME("GuiChecked"), EditorStringName(EditorIcons)));
	theme->set_icon("unchecked", "CheckBox", theme->get_icon(SNAME("GuiUnchecked"), EditorStringName(EditorIcons)));
	theme->set_icon("radio_checked", "CheckBox", theme->get_icon(SNAME("GuiRadioChecked"), EditorStringName(EditorIcons)));
	theme->set_icon("radio_unchecked", "CheckBox", theme->get_icon(SNAME("GuiRadioUnchecked"), EditorStringName(EditorIcons)));
	theme->set_icon("checked_disabled", "CheckBox", theme->get_icon(SNAME("GuiCheckedDisabled"), EditorStringName(EditorIcons)));
	theme->set_icon("unchecked_disabled", "CheckBox", theme->get_icon(SNAME("GuiUncheckedDisabled"), EditorStringName(EditorIcons)));
	theme->set_icon("radio_checked_disabled", "CheckBox", theme->get_icon(SNAME("GuiRadioCheckedDisabled"), EditorStringName(EditorIcons)));
	theme->set_icon("radio_unchecked_disabled", "CheckBox", theme->get_icon(SNAME("GuiRadioUncheckedDisabled"), EditorStringName(EditorIcons)));

	theme->set_color("font_color", "CheckBox", font_color);
	theme->set_color("font_hover_color", "CheckBox", font_hover_color);
	theme->set_color("font_hover_pressed_color", "CheckBox", font_hover_pressed_color);
	theme->set_color("font_focus_color", "CheckBox", font_focus_color);
	theme->set_color("font_pressed_color", "CheckBox", accent_color);
	theme->set_color("font_disabled_color", "CheckBox", font_disabled_color);
	theme->set_color("font_outline_color", "CheckBox", font_outline_color);

	theme->set_color("icon_normal_color", "CheckBox", icon_normal_color);
	theme->set_color("icon_hover_color", "CheckBox", icon_hover_color);
	theme->set_color("icon_focus_color", "CheckBox", icon_focus_color);
	theme->set_color("icon_pressed_color", "CheckBox", icon_pressed_color);
	theme->set_color("icon_disabled_color", "CheckBox", icon_disabled_color);

	theme->set_constant("h_separation", "CheckBox", 8 * EDSCALE);
	theme->set_constant("check_v_offset", "CheckBox", 0);
	theme->set_constant("outline_size", "CheckBox", 0);

	// PopupDialog
	theme->set_stylebox("panel", "PopupDialog", style_popup);

	// PopupMenu
	Ref<StyleBoxFlat> style_popup_menu = style_popup->duplicate();
	// Use 1 pixel for the sides, since if 0 is used, the highlight of hovered items is drawn
	// on top of the popup border. This causes a 'gap' in the panel border when an item is highlighted,
	// and it looks weird. 1px solves this.
	style_popup_menu->set_content_margin_individual(EDSCALE, 2 * EDSCALE, EDSCALE, 2 * EDSCALE);
	// Always display a border for PopupMenus so they can be distinguished from their background.
	style_popup_menu->set_border_width_all(EDSCALE);
	if (draw_extra_borders) {
		style_popup_menu->set_border_color(extra_border_color_2);
	} else {
		style_popup_menu->set_border_color(dark_color_2);
	}
	theme->set_stylebox("panel", "PopupMenu", style_popup_menu);

	Ref<StyleBoxFlat> style_menu_hover = style_widget_hover->duplicate();
	// Don't use rounded corners for hover highlights since the StyleBox touches the PopupMenu's edges.
	style_menu_hover->set_corner_radius_all(0);
	theme->set_stylebox("hover", "PopupMenu", style_menu_hover);

	theme->set_stylebox("separator", "PopupMenu", style_popup_separator);
	theme->set_stylebox("labeled_separator_left", "PopupMenu", style_popup_labeled_separator_left);
	theme->set_stylebox("labeled_separator_right", "PopupMenu", style_popup_labeled_separator_right);

	theme->set_color("font_color", "PopupMenu", font_color);
	theme->set_color("font_hover_color", "PopupMenu", font_hover_color);
	theme->set_color("font_accelerator_color", "PopupMenu", font_disabled_color);
	theme->set_color("font_disabled_color", "PopupMenu", font_disabled_color);
	theme->set_color("font_separator_color", "PopupMenu", font_disabled_color);
	theme->set_color("font_outline_color", "PopupMenu", font_outline_color);
	theme->set_icon("checked", "PopupMenu", theme->get_icon(SNAME("GuiChecked"), EditorStringName(EditorIcons)));
	theme->set_icon("unchecked", "PopupMenu", theme->get_icon(SNAME("GuiUnchecked"), EditorStringName(EditorIcons)));
	theme->set_icon("radio_checked", "PopupMenu", theme->get_icon(SNAME("GuiRadioChecked"), EditorStringName(EditorIcons)));
	theme->set_icon("radio_unchecked", "PopupMenu", theme->get_icon(SNAME("GuiRadioUnchecked"), EditorStringName(EditorIcons)));
	theme->set_icon("checked_disabled", "PopupMenu", theme->get_icon(SNAME("GuiCheckedDisabled"), EditorStringName(EditorIcons)));
	theme->set_icon("unchecked_disabled", "PopupMenu", theme->get_icon(SNAME("GuiUncheckedDisabled"), EditorStringName(EditorIcons)));
	theme->set_icon("radio_checked_disabled", "PopupMenu", theme->get_icon(SNAME("GuiRadioCheckedDisabled"), EditorStringName(EditorIcons)));
	theme->set_icon("radio_unchecked_disabled", "PopupMenu", theme->get_icon(SNAME("GuiRadioUncheckedDisabled"), EditorStringName(EditorIcons)));
	theme->set_icon("submenu", "PopupMenu", theme->get_icon(SNAME("ArrowRight"), EditorStringName(EditorIcons)));
	theme->set_icon("submenu_mirrored", "PopupMenu", theme->get_icon(SNAME("ArrowLeft"), EditorStringName(EditorIcons)));
	theme->set_icon("visibility_hidden", "PopupMenu", theme->get_icon(SNAME("GuiVisibilityHidden"), EditorStringName(EditorIcons)));
	theme->set_icon("visibility_visible", "PopupMenu", theme->get_icon(SNAME("GuiVisibilityVisible"), EditorStringName(EditorIcons)));
	theme->set_icon("visibility_xray", "PopupMenu", theme->get_icon(SNAME("GuiVisibilityXray"), EditorStringName(EditorIcons)));

	// Force the v_separation to be even so that the spacing on top and bottom is even.
	// If the vsep is odd and cannot be split into 2 even groups (of pixels), then it will be lopsided.
	// We add 2 to the vsep to give it some extra spacing which looks a bit more modern (see Windows, for example).
	const int vsep_base = extra_spacing + default_margin_size + 6;
	const int force_even_vsep = vsep_base + (vsep_base % 2);
	theme->set_constant("v_separation", "PopupMenu", force_even_vsep * EDSCALE);
	theme->set_constant("outline_size", "PopupMenu", 0);
	theme->set_constant("item_start_padding", "PopupMenu", default_margin_size * 1.5 * EDSCALE);
	theme->set_constant("item_end_padding", "PopupMenu", default_margin_size * 1.5 * EDSCALE);

	// Sub-inspectors
	for (int i = 0; i < 16; i++) {
		Color si_base_color = accent_color;

		float hue_rotate = (i * 2 % 16) / 16.0;
		si_base_color.set_hsv(Math::fmod(float(si_base_color.get_h() + hue_rotate), float(1.0)), si_base_color.get_s(), si_base_color.get_v());
		si_base_color = accent_color.lerp(si_base_color, float(EDITOR_GET("docks/property_editor/subresource_hue_tint")));

		// Sub-inspector background.
		Ref<StyleBoxFlat> sub_inspector_bg = style_default->duplicate();
		sub_inspector_bg->set_bg_color(dark_color_1.lerp(si_base_color, 0.08));
		sub_inspector_bg->set_border_width_all(2 * EDSCALE);
		sub_inspector_bg->set_border_color(si_base_color * Color(0.7, 0.7, 0.7, 0.8));
		sub_inspector_bg->set_content_margin_all(4 * EDSCALE);
		sub_inspector_bg->set_corner_radius(CORNER_TOP_LEFT, 0);
		sub_inspector_bg->set_corner_radius(CORNER_TOP_RIGHT, 0);

		theme->set_stylebox("sub_inspector_bg" + itos(i), EditorStringName(Editor), sub_inspector_bg);

		// EditorProperty background while it has a sub-inspector open.
		Ref<StyleBoxFlat> bg_color = make_flat_stylebox(si_base_color * Color(0.7, 0.7, 0.7, 0.8), 0, 0, 0, 0, corner_radius);
		bg_color->set_anti_aliased(false);
		bg_color->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
		bg_color->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);

		theme->set_stylebox("sub_inspector_property_bg" + itos(i), EditorStringName(Editor), bg_color);
	}

	theme->set_color("sub_inspector_property_color", EditorStringName(Editor), dark_theme ? Color(1, 1, 1, 1) : Color(0, 0, 0, 1));

	// EditorSpinSlider.
	theme->set_color("label_color", "EditorSpinSlider", font_color);
	theme->set_color("read_only_label_color", "EditorSpinSlider", font_readonly_color);

	Ref<StyleBoxFlat> editor_spin_label_bg = style_default->duplicate();
	editor_spin_label_bg->set_bg_color(dark_color_3);
	editor_spin_label_bg->set_border_width_all(0);
	theme->set_stylebox("label_bg", "EditorSpinSlider", editor_spin_label_bg);

	// EditorProperty
	Ref<StyleBoxFlat> style_property_bg = style_default->duplicate();
	style_property_bg->set_bg_color(highlight_color);
	style_property_bg->set_border_width_all(0);

	Ref<StyleBoxFlat> style_property_child_bg = style_default->duplicate();
	style_property_child_bg->set_bg_color(dark_color_2);
	style_property_child_bg->set_border_width_all(0);

	theme->set_constant("font_offset", "EditorProperty", 8 * EDSCALE);
	theme->set_stylebox("bg_selected", "EditorProperty", style_property_bg);
	theme->set_stylebox("bg", "EditorProperty", Ref<StyleBoxEmpty>(memnew(StyleBoxEmpty)));
	theme->set_stylebox("child_bg", "EditorProperty", style_property_child_bg);
	theme->set_constant("v_separation", "EditorProperty", (extra_spacing + default_margin_size) * EDSCALE);
	theme->set_color("warning_color", "EditorProperty", warning_color);
	theme->set_color("property_color", "EditorProperty", property_color);
	theme->set_color("readonly_color", "EditorProperty", readonly_color);
	theme->set_color("readonly_warning_color", "EditorProperty", readonly_warning_color);

	Ref<StyleBoxFlat> style_property_group_note = style_default->duplicate();
	Color property_group_note_color = accent_color;
	property_group_note_color.a = 0.1;
	style_property_group_note->set_bg_color(property_group_note_color);
	theme->set_stylebox("bg_group_note", "EditorProperty", style_property_group_note);

	// EditorInspectorSection
	Color inspector_section_color = font_color.lerp(Color(0.5, 0.5, 0.5), 0.35);
	theme->set_color("font_color", "EditorInspectorSection", inspector_section_color);

	Color inspector_indent_color = accent_color;
	inspector_indent_color.a = 0.2;
	Ref<StyleBoxFlat> inspector_indent_style = make_flat_stylebox(inspector_indent_color, 2.0 * EDSCALE, 0, 2.0 * EDSCALE, 0);
	theme->set_stylebox("indent_box", "EditorInspectorSection", inspector_indent_style);
	theme->set_constant("indent_size", "EditorInspectorSection", 6.0 * EDSCALE);

	theme->set_constant("inspector_margin", EditorStringName(Editor), 12 * EDSCALE);

	// Tree & ItemList background
	Ref<StyleBoxFlat> style_tree_bg = style_default->duplicate();
	// Make Trees easier to distinguish from other controls by using a darker background color.
	style_tree_bg->set_bg_color(dark_color_1.lerp(dark_color_2, 0.5));
	if (draw_extra_borders) {
		style_tree_bg->set_border_width_all(Math::round(EDSCALE));
		style_tree_bg->set_border_color(extra_border_color_2);
	} else {
		style_tree_bg->set_border_color(dark_color_3);
	}

	theme->set_stylebox("panel", "Tree", style_tree_bg);
	theme->set_stylebox("panel", "EditorValidationPanel", style_tree_bg);

	// Tree
	theme->set_icon("checked", "Tree", theme->get_icon(SNAME("GuiChecked"), EditorStringName(EditorIcons)));
	theme->set_icon("indeterminate", "Tree", theme->get_icon(SNAME("GuiIndeterminate"), EditorStringName(EditorIcons)));
	theme->set_icon("unchecked", "Tree", theme->get_icon(SNAME("GuiUnchecked"), EditorStringName(EditorIcons)));
	theme->set_icon("arrow", "Tree", theme->get_icon(SNAME("GuiTreeArrowDown"), EditorStringName(EditorIcons)));
	theme->set_icon("arrow_collapsed", "Tree", theme->get_icon(SNAME("GuiTreeArrowRight"), EditorStringName(EditorIcons)));
	theme->set_icon("arrow_collapsed_mirrored", "Tree", theme->get_icon(SNAME("GuiTreeArrowLeft"), EditorStringName(EditorIcons)));
	theme->set_icon("updown", "Tree", theme->get_icon(SNAME("GuiTreeUpdown"), EditorStringName(EditorIcons)));
	theme->set_icon("select_arrow", "Tree", theme->get_icon(SNAME("GuiDropdown"), EditorStringName(EditorIcons)));
	theme->set_stylebox("focus", "Tree", style_widget_focus);
	theme->set_stylebox("custom_button", "Tree", make_empty_stylebox());
	theme->set_stylebox("custom_button_pressed", "Tree", make_empty_stylebox());
	theme->set_stylebox("custom_button_hover", "Tree", style_widget);
	theme->set_color("custom_button_font_highlight", "Tree", font_hover_color);
	theme->set_color("font_color", "Tree", font_color);
	theme->set_color("font_selected_color", "Tree", mono_color);
	theme->set_color("font_outline_color", "Tree", font_outline_color);
	theme->set_color("title_button_color", "Tree", font_color);
	theme->set_color("drop_position_color", "Tree", accent_color);
	theme->set_constant("v_separation", "Tree", widget_default_margin.y - EDSCALE);
	theme->set_constant("h_separation", "Tree", 6 * EDSCALE);
	theme->set_constant("guide_width", "Tree", border_width);
	theme->set_constant("item_margin", "Tree", 3 * default_margin_size * EDSCALE);
	theme->set_constant("inner_item_margin_bottom", "Tree", (default_margin_size + extra_spacing) * EDSCALE);
	theme->set_constant("inner_item_margin_left", "Tree", (default_margin_size + extra_spacing) * EDSCALE);
	theme->set_constant("inner_item_margin_right", "Tree", (default_margin_size + extra_spacing) * EDSCALE);
	theme->set_constant("inner_item_margin_top", "Tree", (default_margin_size + extra_spacing) * EDSCALE);
	theme->set_constant("button_margin", "Tree", default_margin_size * EDSCALE);
	theme->set_constant("scroll_border", "Tree", 40 * EDSCALE);
	theme->set_constant("scroll_speed", "Tree", 12);
	theme->set_constant("outline_size", "Tree", 0);
	theme->set_constant("scrollbar_margin_left", "Tree", 0);
	theme->set_constant("scrollbar_margin_top", "Tree", 0);
	theme->set_constant("scrollbar_margin_right", "Tree", 0);
	theme->set_constant("scrollbar_margin_bottom", "Tree", 0);
	theme->set_constant("scrollbar_h_separation", "Tree", 1 * EDSCALE);
	theme->set_constant("scrollbar_v_separation", "Tree", 1 * EDSCALE);

	const Color guide_color = mono_color * Color(1, 1, 1, 0.05);
	Color relationship_line_color = mono_color * Color(1, 1, 1, relationship_line_opacity);

	theme->set_constant("draw_guides", "Tree", relationship_line_opacity < 0.01);
	theme->set_color("guide_color", "Tree", guide_color);

	int relationship_line_width = 1;
	Color parent_line_color = mono_color * Color(1, 1, 1, CLAMP(relationship_line_opacity + 0.45, 0.0, 1.0));
	Color children_line_color = mono_color * Color(1, 1, 1, CLAMP(relationship_line_opacity + 0.25, 0.0, 1.0));
	theme->set_constant("draw_relationship_lines", "Tree", relationship_line_opacity >= 0.01);
	theme->set_constant("relationship_line_width", "Tree", relationship_line_width);
	theme->set_constant("parent_hl_line_width", "Tree", relationship_line_width * 2);
	theme->set_constant("children_hl_line_width", "Tree", relationship_line_width);
	theme->set_constant("parent_hl_line_margin", "Tree", relationship_line_width * 3);
	theme->set_color("relationship_line_color", "Tree", relationship_line_color);
	theme->set_color("parent_hl_line_color", "Tree", parent_line_color);
	theme->set_color("children_hl_line_color", "Tree", children_line_color);

	Ref<StyleBoxFlat> style_tree_btn = style_default->duplicate();
	style_tree_btn->set_bg_color(highlight_color);
	style_tree_btn->set_border_width_all(0);
	theme->set_stylebox("button_pressed", "Tree", style_tree_btn);

	Ref<StyleBoxFlat> style_tree_hover = style_default->duplicate();
	style_tree_hover->set_bg_color(highlight_color * Color(1, 1, 1, 0.4));
	style_tree_hover->set_border_width_all(0);
	theme->set_stylebox("hover", "Tree", style_tree_hover);

	Ref<StyleBoxFlat> style_tree_focus = style_default->duplicate();
	style_tree_focus->set_bg_color(highlight_color);
	style_tree_focus->set_border_width_all(0);
	theme->set_stylebox("selected_focus", "Tree", style_tree_focus);

	Ref<StyleBoxFlat> style_tree_selected = style_tree_focus->duplicate();
	theme->set_stylebox("selected", "Tree", style_tree_selected);

	Ref<StyleBoxFlat> style_tree_cursor = style_default->duplicate();
	style_tree_cursor->set_draw_center(false);
	style_tree_cursor->set_border_width_all(MAX(1, border_width));
	style_tree_cursor->set_border_color(contrast_color_1);

	Ref<StyleBoxFlat> style_tree_title = style_default->duplicate();
	style_tree_title->set_bg_color(dark_color_3);
	style_tree_title->set_border_width_all(0);
	theme->set_stylebox("cursor", "Tree", style_tree_cursor);
	theme->set_stylebox("cursor_unfocused", "Tree", style_tree_cursor);
	theme->set_stylebox("title_button_normal", "Tree", style_tree_title);
	theme->set_stylebox("title_button_hover", "Tree", style_tree_title);
	theme->set_stylebox("title_button_pressed", "Tree", style_tree_title);

	Color prop_category_color = dark_color_1.lerp(mono_color, 0.12);
	Color prop_section_color = dark_color_1.lerp(mono_color, 0.09);
	Color prop_subsection_color = dark_color_1.lerp(mono_color, 0.06);
	theme->set_color("prop_category", EditorStringName(Editor), prop_category_color);
	theme->set_color("prop_section", EditorStringName(Editor), prop_section_color);
	theme->set_color("prop_subsection", EditorStringName(Editor), prop_subsection_color);
	theme->set_color("drop_position_color", "Tree", accent_color);

	// EditorInspectorCategory
	Ref<StyleBoxFlat> category_bg = style_default->duplicate();
	category_bg->set_bg_color(prop_category_color);
	category_bg->set_border_color(prop_category_color);
	theme->set_stylebox("bg", "EditorInspectorCategory", category_bg);

	// ItemList
	Ref<StyleBoxFlat> style_itemlist_bg = style_default->duplicate();
	style_itemlist_bg->set_bg_color(dark_color_1);

	if (draw_extra_borders) {
		style_itemlist_bg->set_border_width_all(Math::round(EDSCALE));
		style_itemlist_bg->set_border_color(extra_border_color_2);
	} else {
		style_itemlist_bg->set_border_width_all(border_width);
		style_itemlist_bg->set_border_color(dark_color_3);
	}

	Ref<StyleBoxFlat> style_itemlist_cursor = style_default->duplicate();
	style_itemlist_cursor->set_draw_center(false);
	style_itemlist_cursor->set_border_width_all(border_width);
	style_itemlist_cursor->set_border_color(highlight_color);

	Ref<StyleBoxFlat> style_itemlist_hover = style_tree_selected->duplicate();
	style_itemlist_hover->set_bg_color(highlight_color * Color(1, 1, 1, 0.3));
	style_itemlist_hover->set_border_width_all(0);

	theme->set_stylebox("panel", "ItemList", style_itemlist_bg);
	theme->set_stylebox("focus", "ItemList", style_widget_focus);
	theme->set_stylebox("cursor", "ItemList", style_itemlist_cursor);
	theme->set_stylebox("cursor_unfocused", "ItemList", style_itemlist_cursor);
	theme->set_stylebox("selected_focus", "ItemList", style_tree_focus);
	theme->set_stylebox("selected", "ItemList", style_tree_selected);
	theme->set_stylebox("hovered", "ItemList", style_itemlist_hover);
	theme->set_color("font_color", "ItemList", font_color);
	theme->set_color("font_hovered_color", "ItemList", mono_color);
	theme->set_color("font_selected_color", "ItemList", mono_color);
	theme->set_color("font_outline_color", "ItemList", font_outline_color);
	theme->set_color("guide_color", "ItemList", guide_color);
	theme->set_constant("v_separation", "ItemList", force_even_vsep * 0.5 * EDSCALE);
	theme->set_constant("h_separation", "ItemList", 6 * EDSCALE);
	theme->set_constant("icon_margin", "ItemList", 6 * EDSCALE);
	theme->set_constant("line_separation", "ItemList", 3 * EDSCALE);
	theme->set_constant("outline_size", "ItemList", 0);

	// TabBar & TabContainer
	Ref<StyleBoxFlat> style_tabbar_background = make_flat_stylebox(dark_color_1, 0, 0, 0, 0, corner_radius * EDSCALE);
	style_tabbar_background->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
	style_tabbar_background->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);
	theme->set_stylebox("tabbar_background", "TabContainer", style_tabbar_background);

	theme->set_stylebox("tab_selected", "TabContainer", style_tab_selected);
	theme->set_stylebox("tab_hovered", "TabContainer", style_tab_hovered);
	theme->set_stylebox("tab_unselected", "TabContainer", style_tab_unselected);
	theme->set_stylebox("tab_disabled", "TabContainer", style_tab_disabled);
	theme->set_stylebox("tab_focus", "TabContainer", style_tab_focus);
	theme->set_stylebox("tab_selected", "TabBar", style_tab_selected);
	theme->set_stylebox("tab_hovered", "TabBar", style_tab_hovered);
	theme->set_stylebox("tab_unselected", "TabBar", style_tab_unselected);
	theme->set_stylebox("tab_disabled", "TabBar", style_tab_disabled);
	theme->set_stylebox("tab_focus", "TabBar", style_tab_focus);
	theme->set_stylebox("button_pressed", "TabBar", style_menu);
	theme->set_stylebox("button_highlight", "TabBar", style_menu);
	theme->set_color("font_selected_color", "TabContainer", font_color);
	theme->set_color("font_hovered_color", "TabContainer", font_color);
	theme->set_color("font_unselected_color", "TabContainer", font_disabled_color);
	theme->set_color("font_outline_color", "TabContainer", font_outline_color);
	theme->set_color("font_selected_color", "TabBar", font_color);
	theme->set_color("font_hovered_color", "TabBar", font_color);
	theme->set_color("font_unselected_color", "TabBar", font_disabled_color);
	theme->set_color("font_outline_color", "TabBar", font_outline_color);
	theme->set_color("drop_mark_color", "TabContainer", tab_highlight);
	theme->set_color("drop_mark_color", "TabBar", tab_highlight);
	theme->set_icon("menu", "TabContainer", theme->get_icon(SNAME("GuiTabMenu"), EditorStringName(EditorIcons)));
	theme->set_icon("menu_highlight", "TabContainer", theme->get_icon(SNAME("GuiTabMenuHl"), EditorStringName(EditorIcons)));
	theme->set_icon("close", "TabBar", theme->get_icon(SNAME("GuiClose"), EditorStringName(EditorIcons)));
	theme->set_icon("increment", "TabContainer", theme->get_icon(SNAME("GuiScrollArrowRight"), EditorStringName(EditorIcons)));
	theme->set_icon("decrement", "TabContainer", theme->get_icon(SNAME("GuiScrollArrowLeft"), EditorStringName(EditorIcons)));
	theme->set_icon("increment", "TabBar", theme->get_icon(SNAME("GuiScrollArrowRight"), EditorStringName(EditorIcons)));
	theme->set_icon("decrement", "TabBar", theme->get_icon(SNAME("GuiScrollArrowLeft"), EditorStringName(EditorIcons)));
	theme->set_icon("increment_highlight", "TabBar", theme->get_icon(SNAME("GuiScrollArrowRightHl"), EditorStringName(EditorIcons)));
	theme->set_icon("decrement_highlight", "TabBar", theme->get_icon(SNAME("GuiScrollArrowLeftHl"), EditorStringName(EditorIcons)));
	theme->set_icon("increment_highlight", "TabContainer", theme->get_icon(SNAME("GuiScrollArrowRightHl"), EditorStringName(EditorIcons)));
	theme->set_icon("decrement_highlight", "TabContainer", theme->get_icon(SNAME("GuiScrollArrowLeftHl"), EditorStringName(EditorIcons)));
	theme->set_icon("drop_mark", "TabContainer", theme->get_icon(SNAME("GuiTabDropMark"), EditorStringName(EditorIcons)));
	theme->set_icon("drop_mark", "TabBar", theme->get_icon(SNAME("GuiTabDropMark"), EditorStringName(EditorIcons)));
	theme->set_constant("side_margin", "TabContainer", 0);
	theme->set_constant("outline_size", "TabContainer", 0);
	theme->set_constant("h_separation", "TabBar", 4 * EDSCALE);
	theme->set_constant("outline_size", "TabBar", 0);

	// Content of each tab.
	Ref<StyleBoxFlat> style_content_panel = style_default->duplicate();
	style_content_panel->set_border_color(dark_color_3);
	style_content_panel->set_border_width_all(border_width);
	style_content_panel->set_border_width(Side::SIDE_TOP, 0);
	style_content_panel->set_corner_radius(CORNER_TOP_LEFT, 0);
	style_content_panel->set_corner_radius(CORNER_TOP_RIGHT, 0);
	// Compensate for the border.
	style_content_panel->set_content_margin_individual(margin_size_extra * EDSCALE, (2 + margin_size_extra) * EDSCALE, margin_size_extra * EDSCALE, margin_size_extra * EDSCALE);
	theme->set_stylebox("panel", "TabContainer", style_content_panel);

	// Bottom panel.
	Ref<StyleBoxFlat> style_bottom_panel = style_content_panel->duplicate();
	style_bottom_panel->set_corner_radius_all(corner_radius * EDSCALE);
	theme->set_stylebox("BottomPanel", EditorStringName(EditorStyles), style_bottom_panel);

	// TabContainerOdd can be used on tabs against the base color background (e.g. nested tabs).
	theme->set_type_variation("TabContainerOdd", "TabContainer");

	Ref<StyleBoxFlat> style_tab_selected_odd = style_tab_selected->duplicate();
	style_tab_selected_odd->set_bg_color(disabled_bg_color);
	theme->set_stylebox("tab_selected", "TabContainerOdd", style_tab_selected_odd);

	Ref<StyleBoxFlat> style_content_panel_odd = style_content_panel->duplicate();
	style_content_panel_odd->set_bg_color(disabled_bg_color);
	theme->set_stylebox("panel", "TabContainerOdd", style_content_panel_odd);

	// This stylebox is used in 3d and 2d viewports (no borders).
	Ref<StyleBoxFlat> style_content_panel_vp = style_content_panel->duplicate();
	style_content_panel_vp->set_content_margin_individual(border_width * 2, default_margin_size * EDSCALE, border_width * 2, border_width * 2);
	theme->set_stylebox("Content", EditorStringName(EditorStyles), style_content_panel_vp);

	// This stylebox is used by preview tabs in the Theme Editor.
	Ref<StyleBoxFlat> style_theme_preview_tab = style_tab_selected_odd->duplicate();
	style_theme_preview_tab->set_expand_margin(SIDE_BOTTOM, 5 * EDSCALE);
	theme->set_stylebox("ThemeEditorPreviewFG", EditorStringName(EditorStyles), style_theme_preview_tab);
	Ref<StyleBoxFlat> style_theme_preview_bg_tab = style_tab_unselected->duplicate();
	style_theme_preview_bg_tab->set_expand_margin(SIDE_BOTTOM, 2 * EDSCALE);
	theme->set_stylebox("ThemeEditorPreviewBG", EditorStringName(EditorStyles), style_theme_preview_bg_tab);

	Ref<StyleBoxFlat> style_texture_region_bg = style_tree_bg->duplicate();
	style_texture_region_bg->set_content_margin_all(0);
	theme->set_stylebox("TextureRegionPreviewBG", EditorStringName(EditorStyles), style_texture_region_bg);
	theme->set_stylebox("TextureRegionPreviewFG", EditorStringName(EditorStyles), make_empty_stylebox(0, 0, 0, 0));

	// Separators
	theme->set_stylebox("separator", "HSeparator", make_line_stylebox(separator_color, MAX(Math::round(EDSCALE), border_width)));
	theme->set_stylebox("separator", "VSeparator", make_line_stylebox(separator_color, MAX(Math::round(EDSCALE), border_width), 0, 0, true));

	// Debugger

	Ref<StyleBoxFlat> style_panel_debugger = style_content_panel->duplicate();
	style_panel_debugger->set_border_width(SIDE_BOTTOM, 0);
	theme->set_stylebox("DebuggerPanel", EditorStringName(EditorStyles), style_panel_debugger);

	Ref<StyleBoxFlat> style_panel_invisible_top = style_content_panel->duplicate();
	int stylebox_offset = theme->get_font(SNAME("tab_selected"), SNAME("TabContainer"))->get_height(theme->get_font_size(SNAME("tab_selected"), SNAME("TabContainer"))) + theme->get_stylebox(SNAME("tab_selected"), SNAME("TabContainer"))->get_minimum_size().height + theme->get_stylebox(SNAME("panel"), SNAME("TabContainer"))->get_content_margin(SIDE_TOP);
	style_panel_invisible_top->set_expand_margin(SIDE_TOP, -stylebox_offset);
	style_panel_invisible_top->set_content_margin(SIDE_TOP, 0);
	theme->set_stylebox("BottomPanelDebuggerOverride", EditorStringName(EditorStyles), style_panel_invisible_top);

	// LineEdit

	Ref<StyleBoxFlat> style_line_edit = style_widget->duplicate();
	// The original style_widget style has an extra 1 pixel offset that makes LineEdits not align with Buttons,
	// so this compensates for that.
	style_line_edit->set_content_margin(SIDE_TOP, style_line_edit->get_content_margin(SIDE_TOP) - 1 * EDSCALE);

	// Don't round the bottom corners to make the line look sharper.
	style_line_edit->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
	style_line_edit->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);

	if (draw_extra_borders) {
		style_line_edit->set_border_width_all(Math::round(EDSCALE));
		style_line_edit->set_border_color(extra_border_color_1);
	} else {
		// Add a bottom line to make LineEdits more visible, especially in sectioned inspectors
		// such as the Project Settings.
		style_line_edit->set_border_width(SIDE_BOTTOM, Math::round(2 * EDSCALE));
		style_line_edit->set_border_color(dark_color_2);
	}

	Ref<StyleBoxFlat> style_line_edit_disabled = style_line_edit->duplicate();
	style_line_edit_disabled->set_border_color(disabled_color);
	style_line_edit_disabled->set_bg_color(disabled_bg_color);

	theme->set_stylebox("normal", "LineEdit", style_line_edit);
	theme->set_stylebox("focus", "LineEdit", style_widget_focus);
	theme->set_stylebox("read_only", "LineEdit", style_line_edit_disabled);
	theme->set_icon("clear", "LineEdit", theme->get_icon(SNAME("GuiClose"), EditorStringName(EditorIcons)));
	theme->set_color("font_color", "LineEdit", font_color);
	theme->set_color("font_selected_color", "LineEdit", mono_color);
	theme->set_color("font_uneditable_color", "LineEdit", font_readonly_color);
	theme->set_color("font_placeholder_color", "LineEdit", font_placeholder_color);
	theme->set_color("font_outline_color", "LineEdit", font_outline_color);
	theme->set_color("caret_color", "LineEdit", font_color);
	theme->set_color("selection_color", "LineEdit", selection_color);
	theme->set_color("clear_button_color", "LineEdit", font_color);
	theme->set_color("clear_button_color_pressed", "LineEdit", accent_color);

	theme->set_constant("minimum_character_width", "LineEdit", 4);
	theme->set_constant("outline_size", "LineEdit", 0);
	theme->set_constant("caret_width", "LineEdit", 1);

	// TextEdit
	theme->set_stylebox("normal", "TextEdit", style_line_edit);
	theme->set_stylebox("focus", "TextEdit", style_widget_focus);
	theme->set_stylebox("read_only", "TextEdit", style_line_edit_disabled);
	theme->set_icon("tab", "TextEdit", theme->get_icon(SNAME("GuiTab"), EditorStringName(EditorIcons)));
	theme->set_icon("space", "TextEdit", theme->get_icon(SNAME("GuiSpace"), EditorStringName(EditorIcons)));
	theme->set_color("font_color", "TextEdit", font_color);
	theme->set_color("font_readonly_color", "TextEdit", font_readonly_color);
	theme->set_color("font_placeholder_color", "TextEdit", font_placeholder_color);
	theme->set_color("font_outline_color", "TextEdit", font_outline_color);
	theme->set_color("caret_color", "TextEdit", font_color);
	theme->set_color("selection_color", "TextEdit", selection_color);
	theme->set_color("background_color", "TextEdit", Color(0, 0, 0, 0));

	theme->set_constant("line_spacing", "TextEdit", 4 * EDSCALE);
	theme->set_constant("outline_size", "TextEdit", 0);
	theme->set_constant("caret_width", "TextEdit", 1);

	theme->set_icon("h_grabber", "SplitContainer", theme->get_icon(SNAME("GuiHsplitter"), EditorStringName(EditorIcons)));
	theme->set_icon("v_grabber", "SplitContainer", theme->get_icon(SNAME("GuiVsplitter"), EditorStringName(EditorIcons)));
	theme->set_icon("grabber", "VSplitContainer", theme->get_icon(SNAME("GuiVsplitter"), EditorStringName(EditorIcons)));
	theme->set_icon("grabber", "HSplitContainer", theme->get_icon(SNAME("GuiHsplitter"), EditorStringName(EditorIcons)));

	theme->set_constant("separation", "SplitContainer", default_margin_size * 2 * EDSCALE);
	theme->set_constant("separation", "HSplitContainer", default_margin_size * 2 * EDSCALE);
	theme->set_constant("separation", "VSplitContainer", default_margin_size * 2 * EDSCALE);

	theme->set_constant("minimum_grab_thickness", "SplitContainer", 6 * EDSCALE);
	theme->set_constant("minimum_grab_thickness", "HSplitContainer", 6 * EDSCALE);
	theme->set_constant("minimum_grab_thickness", "VSplitContainer", 6 * EDSCALE);

	// Containers
	theme->set_constant("separation", "BoxContainer", default_margin_size * EDSCALE);
	theme->set_constant("separation", "HBoxContainer", default_margin_size * EDSCALE);
	theme->set_constant("separation", "VBoxContainer", default_margin_size * EDSCALE);
	theme->set_constant("margin_left", "MarginContainer", 0);
	theme->set_constant("margin_top", "MarginContainer", 0);
	theme->set_constant("margin_right", "MarginContainer", 0);
	theme->set_constant("margin_bottom", "MarginContainer", 0);
	theme->set_constant("h_separation", "GridContainer", default_margin_size * EDSCALE);
	theme->set_constant("v_separation", "GridContainer", default_margin_size * EDSCALE);
	theme->set_constant("h_separation", "FlowContainer", default_margin_size * EDSCALE);
	theme->set_constant("v_separation", "FlowContainer", default_margin_size * EDSCALE);
	theme->set_constant("h_separation", "HFlowContainer", default_margin_size * EDSCALE);
	theme->set_constant("v_separation", "HFlowContainer", default_margin_size * EDSCALE);
	theme->set_constant("h_separation", "VFlowContainer", default_margin_size * EDSCALE);
	theme->set_constant("v_separation", "VFlowContainer", default_margin_size * EDSCALE);

	// Custom theme type for MarginContainer with 4px margins.
	theme->set_type_variation("MarginContainer4px", "MarginContainer");
	theme->set_constant("margin_left", "MarginContainer4px", 4 * EDSCALE);
	theme->set_constant("margin_top", "MarginContainer4px", 4 * EDSCALE);
	theme->set_constant("margin_right", "MarginContainer4px", 4 * EDSCALE);
	theme->set_constant("margin_bottom", "MarginContainer4px", 4 * EDSCALE);

	// Window

	// Prevent corner artifacts between window title and body.
	Ref<StyleBoxFlat> style_window_title = style_default->duplicate();
	style_window_title->set_corner_radius(CORNER_TOP_LEFT, 0);
	style_window_title->set_corner_radius(CORNER_TOP_RIGHT, 0);
	// Prevent visible line between window title and body.
	style_window_title->set_expand_margin(SIDE_BOTTOM, 2 * EDSCALE);

	Ref<StyleBoxFlat> style_window = style_popup->duplicate();
	style_window->set_border_color(base_color);
	style_window->set_border_width(SIDE_TOP, 24 * EDSCALE);
	style_window->set_expand_margin(SIDE_TOP, 24 * EDSCALE);
	theme->set_stylebox("embedded_border", "Window", style_window);
	theme->set_stylebox("embedded_unfocused_border", "Window", style_window);

	theme->set_color("title_color", "Window", font_color);
	theme->set_icon("close", "Window", theme->get_icon(SNAME("GuiClose"), EditorStringName(EditorIcons)));
	theme->set_icon("close_pressed", "Window", theme->get_icon(SNAME("GuiClose"), EditorStringName(EditorIcons)));
	theme->set_constant("close_h_offset", "Window", 22 * EDSCALE);
	theme->set_constant("close_v_offset", "Window", 20 * EDSCALE);
	theme->set_constant("title_height", "Window", 24 * EDSCALE);
	theme->set_constant("resize_margin", "Window", 4 * EDSCALE);
	theme->set_font("title_font", "Window", theme->get_font(SNAME("title"), EditorStringName(EditorFonts)));
	theme->set_font_size("title_font_size", "Window", theme->get_font_size(SNAME("title_size"), EditorStringName(EditorFonts)));

	// Complex window (currently only Editor Settings and Project Settings)
	Ref<StyleBoxFlat> style_complex_window = style_window->duplicate();
	style_complex_window->set_bg_color(dark_color_2);
	style_complex_window->set_border_color(dark_color_2);
	theme->set_stylebox("panel", "EditorSettingsDialog", style_complex_window);
	theme->set_stylebox("panel", "ProjectSettingsEditor", style_complex_window);
	theme->set_stylebox("panel", "EditorAbout", style_complex_window);

	// AcceptDialog
	theme->set_stylebox("panel", "AcceptDialog", style_window_title);
	theme->set_constant("buttons_separation", "AcceptDialog", 8 * EDSCALE);

	// HScrollBar
	Ref<Texture2D> empty_icon = memnew(ImageTexture);

	if (increase_scrollbar_touch_area) {
		theme->set_stylebox("scroll", "HScrollBar", make_line_stylebox(separator_color, 50));
	} else {
		theme->set_stylebox("scroll", "HScrollBar", make_stylebox(theme->get_icon(SNAME("GuiScrollBg"), EditorStringName(EditorIcons)), 5, 5, 5, 5, -5, 1, -5, 1));
	}
	theme->set_stylebox("scroll_focus", "HScrollBar", make_stylebox(theme->get_icon(SNAME("GuiScrollBg"), EditorStringName(EditorIcons)), 5, 5, 5, 5, 1, 1, 1, 1));
	theme->set_stylebox("grabber", "HScrollBar", make_stylebox(theme->get_icon(SNAME("GuiScrollGrabber"), EditorStringName(EditorIcons)), 6, 6, 6, 6, 1, 1, 1, 1));
	theme->set_stylebox("grabber_highlight", "HScrollBar", make_stylebox(theme->get_icon(SNAME("GuiScrollGrabberHl"), EditorStringName(EditorIcons)), 5, 5, 5, 5, 1, 1, 1, 1));
	theme->set_stylebox("grabber_pressed", "HScrollBar", make_stylebox(theme->get_icon(SNAME("GuiScrollGrabberPressed"), EditorStringName(EditorIcons)), 6, 6, 6, 6, 1, 1, 1, 1));

	theme->set_icon("increment", "HScrollBar", empty_icon);
	theme->set_icon("increment_highlight", "HScrollBar", empty_icon);
	theme->set_icon("increment_pressed", "HScrollBar", empty_icon);
	theme->set_icon("decrement", "HScrollBar", empty_icon);
	theme->set_icon("decrement_highlight", "HScrollBar", empty_icon);
	theme->set_icon("decrement_pressed", "HScrollBar", empty_icon);

	// VScrollBar
	if (increase_scrollbar_touch_area) {
		theme->set_stylebox("scroll", "VScrollBar", make_line_stylebox(separator_color, 50, 1, 1, true));
	} else {
		theme->set_stylebox("scroll", "VScrollBar", make_stylebox(theme->get_icon(SNAME("GuiScrollBg"), EditorStringName(EditorIcons)), 5, 5, 5, 5, 1, -5, 1, -5));
	}
	theme->set_stylebox("scroll_focus", "VScrollBar", make_stylebox(theme->get_icon(SNAME("GuiScrollBg"), EditorStringName(EditorIcons)), 5, 5, 5, 5, 1, 1, 1, 1));
	theme->set_stylebox("grabber", "VScrollBar", make_stylebox(theme->get_icon(SNAME("GuiScrollGrabber"), EditorStringName(EditorIcons)), 6, 6, 6, 6, 1, 1, 1, 1));
	theme->set_stylebox("grabber_highlight", "VScrollBar", make_stylebox(theme->get_icon(SNAME("GuiScrollGrabberHl"), EditorStringName(EditorIcons)), 5, 5, 5, 5, 1, 1, 1, 1));
	theme->set_stylebox("grabber_pressed", "VScrollBar", make_stylebox(theme->get_icon(SNAME("GuiScrollGrabberPressed"), EditorStringName(EditorIcons)), 6, 6, 6, 6, 1, 1, 1, 1));

	theme->set_icon("increment", "VScrollBar", empty_icon);
	theme->set_icon("increment_highlight", "VScrollBar", empty_icon);
	theme->set_icon("increment_pressed", "VScrollBar", empty_icon);
	theme->set_icon("decrement", "VScrollBar", empty_icon);
	theme->set_icon("decrement_highlight", "VScrollBar", empty_icon);
	theme->set_icon("decrement_pressed", "VScrollBar", empty_icon);

	// HSlider
	theme->set_icon("grabber_highlight", "HSlider", theme->get_icon(SNAME("GuiSliderGrabberHl"), EditorStringName(EditorIcons)));
	theme->set_icon("grabber", "HSlider", theme->get_icon(SNAME("GuiSliderGrabber"), EditorStringName(EditorIcons)));
	theme->set_stylebox("slider", "HSlider", make_flat_stylebox(dark_color_3, 0, default_margin_size / 2, 0, default_margin_size / 2, corner_width));
	theme->set_stylebox("grabber_area", "HSlider", make_flat_stylebox(contrast_color_1, 0, default_margin_size / 2, 0, default_margin_size / 2, corner_width));
	theme->set_stylebox("grabber_area_highlight", "HSlider", make_flat_stylebox(contrast_color_1, 0, default_margin_size / 2, 0, default_margin_size / 2));
	theme->set_constant("center_grabber", "HSlider", 0);
	theme->set_constant("grabber_offset", "HSlider", 0);

	// VSlider
	theme->set_icon("grabber", "VSlider", theme->get_icon(SNAME("GuiSliderGrabber"), EditorStringName(EditorIcons)));
	theme->set_icon("grabber_highlight", "VSlider", theme->get_icon(SNAME("GuiSliderGrabberHl"), EditorStringName(EditorIcons)));
	theme->set_stylebox("slider", "VSlider", make_flat_stylebox(dark_color_3, default_margin_size / 2, 0, default_margin_size / 2, 0, corner_width));
	theme->set_stylebox("grabber_area", "VSlider", make_flat_stylebox(contrast_color_1, default_margin_size / 2, 0, default_margin_size / 2, 0, corner_width));
	theme->set_stylebox("grabber_area_highlight", "VSlider", make_flat_stylebox(contrast_color_1, default_margin_size / 2, 0, default_margin_size / 2, 0));
	theme->set_constant("center_grabber", "VSlider", 0);
	theme->set_constant("grabber_offset", "VSlider", 0);

	// RichTextLabel
	theme->set_color("default_color", "RichTextLabel", font_color);
	theme->set_color("font_shadow_color", "RichTextLabel", Color(0, 0, 0, 0));
	theme->set_color("font_outline_color", "RichTextLabel", font_outline_color);
	theme->set_color("selection_color", "RichTextLabel", selection_color);
	theme->set_constant("shadow_offset_x", "RichTextLabel", 1 * EDSCALE);
	theme->set_constant("shadow_offset_y", "RichTextLabel", 1 * EDSCALE);
	theme->set_constant("shadow_outline_size", "RichTextLabel", 1 * EDSCALE);
	theme->set_constant("outline_size", "RichTextLabel", 0);
	theme->set_stylebox("focus", "RichTextLabel", make_empty_stylebox());
	theme->set_stylebox("normal", "RichTextLabel", style_tree_bg);

	// Editor help.
	Ref<StyleBoxFlat> style_editor_help = style_default->duplicate();
	style_editor_help->set_bg_color(dark_color_2);
	style_editor_help->set_border_color(dark_color_3);
	theme->set_stylebox("background", "EditorHelp", style_editor_help);

	theme->set_color("title_color", "EditorHelp", accent_color);
	theme->set_color("headline_color", "EditorHelp", mono_color);
	theme->set_color("text_color", "EditorHelp", font_color);
	theme->set_color("comment_color", "EditorHelp", font_color * Color(1, 1, 1, 0.6));
	theme->set_color("symbol_color", "EditorHelp", font_color * Color(1, 1, 1, 0.6));
	theme->set_color("value_color", "EditorHelp", font_color * Color(1, 1, 1, 0.6));
	theme->set_color("qualifier_color", "EditorHelp", font_color * Color(1, 1, 1, 0.8));
	theme->set_color("type_color", "EditorHelp", accent_color.lerp(font_color, 0.5));
	theme->set_color("selection_color", "EditorHelp", selection_color);
	theme->set_color("link_color", "EditorHelp", accent_color.lerp(mono_color, 0.8));
	theme->set_color("code_color", "EditorHelp", accent_color.lerp(mono_color, 0.6));
	theme->set_color("kbd_color", "EditorHelp", accent_color.lerp(property_color, 0.6));
	theme->set_color("code_bg_color", "EditorHelp", dark_color_3);
	theme->set_color("kbd_bg_color", "EditorHelp", dark_color_1);
	theme->set_color("param_bg_color", "EditorHelp", dark_color_1);
	theme->set_constant("line_separation", "EditorHelp", Math::round(6 * EDSCALE));
	theme->set_constant("table_h_separation", "EditorHelp", 16 * EDSCALE);
	theme->set_constant("table_v_separation", "EditorHelp", 6 * EDSCALE);
	theme->set_constant("text_highlight_h_padding", "EditorHelp", 1 * EDSCALE);
	theme->set_constant("text_highlight_v_padding", "EditorHelp", 2 * EDSCALE);

	// Panel
	theme->set_stylebox("panel", "Panel", make_flat_stylebox(dark_color_1, 6, 4, 6, 4, corner_width));
	theme->set_stylebox("PanelForeground", EditorStringName(EditorStyles), style_default);

	// Label
	theme->set_stylebox("normal", "Label", style_empty);
	theme->set_color("font_color", "Label", font_color);
	theme->set_color("font_shadow_color", "Label", Color(0, 0, 0, 0));
	theme->set_color("font_outline_color", "Label", font_outline_color);
	theme->set_constant("shadow_offset_x", "Label", 1 * EDSCALE);
	theme->set_constant("shadow_offset_y", "Label", 1 * EDSCALE);
	theme->set_constant("shadow_outline_size", "Label", 1 * EDSCALE);
	theme->set_constant("line_spacing", "Label", 3 * EDSCALE);
	theme->set_constant("outline_size", "Label", 0);

	// LinkButton
	theme->set_stylebox("focus", "LinkButton", style_empty);
	theme->set_color("font_color", "LinkButton", font_color);
	theme->set_color("font_hover_color", "LinkButton", font_hover_color);
	theme->set_color("font_hover_pressed_color", "LinkButton", font_hover_pressed_color);
	theme->set_color("font_focus_color", "LinkButton", font_focus_color);
	theme->set_color("font_pressed_color", "LinkButton", accent_color);
	theme->set_color("font_disabled_color", "LinkButton", font_disabled_color);
	theme->set_color("font_outline_color", "LinkButton", font_outline_color);

	theme->set_constant("outline_size", "LinkButton", 0);

	theme->set_type_variation("HeaderSmallLink", "LinkButton");
	theme->set_font("font", "HeaderSmallLink", theme->get_font(SNAME("font"), SNAME("HeaderSmall")));
	theme->set_font_size("font_size", "HeaderSmallLink", theme->get_font_size(SNAME("font_size"), SNAME("HeaderSmall")));

	// TooltipPanel + TooltipLabel
	// TooltipPanel is also used for custom tooltips, while TooltipLabel
	// is only relevant for default tooltips.
	Ref<StyleBoxFlat> style_tooltip = style_popup->duplicate();
	style_tooltip->set_shadow_size(0);
	style_tooltip->set_content_margin_all(default_margin_size * EDSCALE * 0.5);
	style_tooltip->set_bg_color(dark_color_3 * Color(0.8, 0.8, 0.8, 0.9));
	style_tooltip->set_border_width_all(0);
	theme->set_color("font_color", "TooltipLabel", font_hover_color);
	theme->set_color("font_shadow_color", "TooltipLabel", Color(0, 0, 0, 0));
	theme->set_stylebox("panel", "TooltipPanel", style_tooltip);

	// PopupPanel
	theme->set_stylebox("panel", "PopupPanel", style_popup);

	Ref<StyleBoxFlat> control_editor_popup_style = style_popup->duplicate();
	control_editor_popup_style->set_shadow_size(0);
	control_editor_popup_style->set_content_margin(SIDE_LEFT, default_margin_size * EDSCALE);
	control_editor_popup_style->set_content_margin(SIDE_TOP, default_margin_size * EDSCALE);
	control_editor_popup_style->set_content_margin(SIDE_RIGHT, default_margin_size * EDSCALE);
	control_editor_popup_style->set_content_margin(SIDE_BOTTOM, default_margin_size * EDSCALE);
	control_editor_popup_style->set_border_width_all(0);

	theme->set_stylebox("panel", "ControlEditorPopupPanel", control_editor_popup_style);
	theme->set_type_variation("ControlEditorPopupPanel", "PopupPanel");

	// SpinBox
	theme->set_icon("updown", "SpinBox", theme->get_icon(SNAME("GuiSpinboxUpdown"), EditorStringName(EditorIcons)));
	theme->set_icon("updown_disabled", "SpinBox", theme->get_icon(SNAME("GuiSpinboxUpdownDisabled"), EditorStringName(EditorIcons)));

	// ProgressBar
	theme->set_stylebox("background", "ProgressBar", make_stylebox(theme->get_icon(SNAME("GuiProgressBar"), EditorStringName(EditorIcons)), 4, 4, 4, 4, 0, 0, 0, 0));
	theme->set_stylebox("fill", "ProgressBar", make_stylebox(theme->get_icon(SNAME("GuiProgressFill"), EditorStringName(EditorIcons)), 6, 6, 6, 6, 2, 1, 2, 1));
	theme->set_color("font_color", "ProgressBar", font_color);
	theme->set_color("font_outline_color", "ProgressBar", font_outline_color);
	theme->set_constant("outline_size", "ProgressBar", 0);

	// GraphEdit
	theme->set_stylebox("panel", "GraphEdit", style_tree_bg);
	Ref<StyleBoxFlat> graph_toolbar_style = make_flat_stylebox(dark_color_1 * Color(1, 1, 1, 0.6), 4, 2, 4, 2, 3);
	theme->set_stylebox("menu_panel", "GraphEdit", graph_toolbar_style);

	if (dark_theme) {
		theme->set_color("grid_major", "GraphEdit", Color(1.0, 1.0, 1.0, 0.1));
		theme->set_color("grid_minor", "GraphEdit", Color(1.0, 1.0, 1.0, 0.05));
	} else {
		theme->set_color("grid_major", "GraphEdit", Color(0.0, 0.0, 0.0, 0.15));
		theme->set_color("grid_minor", "GraphEdit", Color(0.0, 0.0, 0.0, 0.07));
	}
	theme->set_color("selection_fill", "GraphEdit", theme->get_color(SNAME("box_selection_fill_color"), EditorStringName(Editor)));
	theme->set_color("selection_stroke", "GraphEdit", theme->get_color(SNAME("box_selection_stroke_color"), EditorStringName(Editor)));
	theme->set_color("activity", "GraphEdit", accent_color);

	theme->set_icon("zoom_out", "GraphEdit", theme->get_icon(SNAME("ZoomLess"), EditorStringName(EditorIcons)));
	theme->set_icon("zoom_in", "GraphEdit", theme->get_icon(SNAME("ZoomMore"), EditorStringName(EditorIcons)));
	theme->set_icon("zoom_reset", "GraphEdit", theme->get_icon(SNAME("ZoomReset"), EditorStringName(EditorIcons)));
	theme->set_icon("grid_toggle", "GraphEdit", theme->get_icon(SNAME("GridToggle"), EditorStringName(EditorIcons)));
	theme->set_icon("minimap_toggle", "GraphEdit", theme->get_icon(SNAME("GridMinimap"), EditorStringName(EditorIcons)));
	theme->set_icon("snapping_toggle", "GraphEdit", theme->get_icon(SNAME("SnapGrid"), EditorStringName(EditorIcons)));
	theme->set_icon("layout", "GraphEdit", theme->get_icon(SNAME("GridLayout"), EditorStringName(EditorIcons)));

	// GraphEditMinimap
	Ref<StyleBoxFlat> style_minimap_bg = make_flat_stylebox(dark_color_1, 0, 0, 0, 0);
	style_minimap_bg->set_border_color(dark_color_3);
	style_minimap_bg->set_border_width_all(1);
	theme->set_stylebox("panel", "GraphEditMinimap", style_minimap_bg);

	Ref<StyleBoxFlat> style_minimap_camera;
	Ref<StyleBoxFlat> style_minimap_node;
	if (dark_theme) {
		style_minimap_camera = make_flat_stylebox(Color(0.65, 0.65, 0.65, 0.2), 0, 0, 0, 0);
		style_minimap_camera->set_border_color(Color(0.65, 0.65, 0.65, 0.45));
		style_minimap_node = make_flat_stylebox(Color(1, 1, 1), 0, 0, 0, 0);
	} else {
		style_minimap_camera = make_flat_stylebox(Color(0.38, 0.38, 0.38, 0.2), 0, 0, 0, 0);
		style_minimap_camera->set_border_color(Color(0.38, 0.38, 0.38, 0.45));
		style_minimap_node = make_flat_stylebox(Color(0, 0, 0), 0, 0, 0, 0);
	}
	style_minimap_camera->set_border_width_all(1);
	style_minimap_node->set_anti_aliased(false);
	theme->set_stylebox("camera", "GraphEditMinimap", style_minimap_camera);
	theme->set_stylebox("node", "GraphEditMinimap", style_minimap_node);

	Color minimap_resizer_color;
	if (dark_theme) {
		minimap_resizer_color = Color(1, 1, 1, 0.65);
	} else {
		minimap_resizer_color = Color(0, 0, 0, 0.65);
	}
	theme->set_icon("resizer", "GraphEditMinimap", theme->get_icon(SNAME("GuiResizerTopLeft"), EditorStringName(EditorIcons)));
	theme->set_color("resizer_color", "GraphEditMinimap", minimap_resizer_color);

	// GraphNode

	const int gn_margin_top = 2;
	const int gn_margin_side = 2;
	const int gn_margin_bottom = 2;

	Color graphnode_bg = dark_color_3;
	if (!dark_theme) {
		graphnode_bg = prop_section_color;
	}
	const Color graph_node_selected_border_color = graphnode_bg.lerp(accent_color, 0.275);

	const Color graphnode_frame_bg = graphnode_bg.lerp(style_tree_bg->get_bg_color(), 0.3);

	Ref<StyleBoxFlat> graphn_sb_panel = make_flat_stylebox(graphnode_frame_bg, gn_margin_side, gn_margin_top, gn_margin_side, gn_margin_bottom, corner_width);
	graphn_sb_panel->set_border_width_all(border_width);
	graphn_sb_panel->set_border_color(graphnode_bg);
	graphn_sb_panel->set_corner_radius_individual(0, 0, corner_radius * EDSCALE, corner_radius * EDSCALE);
	graphn_sb_panel->set_expand_margin(SIDE_TOP, 17 * EDSCALE);

	Ref<StyleBoxFlat> graphn_sb_panel_selected = make_flat_stylebox(graphnode_frame_bg, gn_margin_side, gn_margin_top, gn_margin_side, gn_margin_bottom, corner_width);
	graphn_sb_panel_selected->set_border_width_all(2 * EDSCALE + border_width);
	graphn_sb_panel_selected->set_border_color(graph_node_selected_border_color);
	graphn_sb_panel_selected->set_corner_radius_individual(0, 0, corner_radius * EDSCALE, corner_radius * EDSCALE);
	graphn_sb_panel_selected->set_expand_margin(SIDE_TOP, 17 * EDSCALE);

	const int gn_titlebar_margin_left = 12;
	const int gn_titlebar_margin_right = 4; // The rest is for the close button.
	Ref<StyleBoxFlat> graphn_sb_titlebar = make_flat_stylebox(graphnode_bg, gn_titlebar_margin_left, gn_margin_top, gn_titlebar_margin_right, 0, corner_width);
	graphn_sb_titlebar->set_expand_margin(SIDE_TOP, 2 * EDSCALE);
	graphn_sb_titlebar->set_corner_radius_individual(corner_radius * EDSCALE, corner_radius * EDSCALE, 0, 0);

	Ref<StyleBoxFlat> graphn_sb_titlebar_selected = make_flat_stylebox(graph_node_selected_border_color, gn_titlebar_margin_left, gn_margin_top, gn_titlebar_margin_right, 0, corner_width);
	graphn_sb_titlebar_selected->set_corner_radius_individual(corner_radius * EDSCALE, corner_radius * EDSCALE, 0, 0);
	graphn_sb_titlebar_selected->set_expand_margin(SIDE_TOP, 2 * EDSCALE);
	Ref<StyleBoxEmpty> graphn_sb_slot = make_empty_stylebox(12, 0, 12, 0);

	theme->set_stylebox("panel", "GraphElement", graphn_sb_panel);
	theme->set_stylebox("panel_selected", "GraphElement", graphn_sb_panel_selected);
	theme->set_stylebox("titlebar", "GraphElement", graphn_sb_titlebar);
	theme->set_stylebox("titlebar_selected", "GraphElement", graphn_sb_titlebar_selected);

	// GraphNode's title Label.
	theme->set_type_variation("GraphNodeTitleLabel", "Label");

	theme->set_stylebox("normal", "GraphNodeTitleLabel", make_empty_stylebox(0, 0, 0, 0));
	theme->set_color("font_color", "GraphNodeTitleLabel", font_color);
	theme->set_constant("line_spacing", "GraphNodeTitleLabel", 3 * EDSCALE);

	Color graphnode_decoration_color = dark_color_1.inverted();

	theme->set_color("resizer_color", "GraphElement", graphnode_decoration_color);
	theme->set_icon("resizer", "GraphElement", theme->get_icon(SNAME("GuiResizer"), EditorStringName(EditorIcons)));

	// GraphNode.
	theme->set_stylebox("panel", "GraphNode", graphn_sb_panel);
	theme->set_stylebox("panel_selected", "GraphNode", graphn_sb_panel_selected);
	theme->set_stylebox("titlebar", "GraphNode", graphn_sb_titlebar);
	theme->set_stylebox("titlebar_selected", "GraphNode", graphn_sb_titlebar_selected);
	theme->set_stylebox("slot", "GraphNode", graphn_sb_slot);

	theme->set_color("resizer_color", "GraphNode", graphnode_decoration_color);

	theme->set_constant("port_h_offset", "GraphNode", 0);
	theme->set_constant("separation", "GraphNode", 1 * EDSCALE);

	Ref<ImageTexture> port_icon = theme->get_icon(SNAME("GuiGraphNodePort"), EditorStringName(EditorIcons));
	// The true size is 24x24 This is necessary for sharp port icons at high zoom levels in GraphEdit (up to ~200%).
	port_icon->set_size_override(Size2(12, 12));
	theme->set_icon("port", "GraphNode", port_icon);

	// StateMachine graph
	theme->set_stylebox("panel", "GraphStateMachine", style_tree_bg);
	theme->set_stylebox("error_panel", "GraphStateMachine", style_tree_bg);
	theme->set_color("error_color", "GraphStateMachine", error_color);

	const int sm_margin_side = 10 * EDSCALE;

	Ref<StyleBoxFlat> sm_node_style = make_flat_stylebox(dark_color_3 * Color(1, 1, 1, 0.7), sm_margin_side, 24 * EDSCALE, sm_margin_side, gn_margin_bottom, corner_width);
	sm_node_style->set_border_width_all(border_width);
	sm_node_style->set_border_color(graphnode_bg);

	Ref<StyleBoxFlat> sm_node_selected_style = make_flat_stylebox(graphnode_bg * Color(1, 1, 1, 0.9), sm_margin_side, 24 * EDSCALE, sm_margin_side, gn_margin_bottom, corner_width);
	sm_node_selected_style->set_border_width_all(2 * EDSCALE + border_width);
	sm_node_selected_style->set_border_color(accent_color * Color(1, 1, 1, 0.9));
	sm_node_selected_style->set_shadow_size(8 * EDSCALE);
	sm_node_selected_style->set_shadow_color(shadow_color);

	Ref<StyleBoxFlat> sm_node_playing_style = sm_node_selected_style->duplicate();
	sm_node_playing_style->set_border_color(warning_color);
	sm_node_playing_style->set_shadow_color(warning_color * Color(1, 1, 1, 0.2));

	theme->set_stylebox("node_frame", "GraphStateMachine", sm_node_style);
	theme->set_stylebox("node_frame_selected", "GraphStateMachine", sm_node_selected_style);
	theme->set_stylebox("node_frame_playing", "GraphStateMachine", sm_node_playing_style);

	Ref<StyleBoxFlat> sm_node_start_style = sm_node_style->duplicate();
	sm_node_start_style->set_border_width_all(1 * EDSCALE);
	sm_node_start_style->set_border_color(success_color.lightened(0.24));
	theme->set_stylebox("node_frame_start", "GraphStateMachine", sm_node_start_style);

	Ref<StyleBoxFlat> sm_node_end_style = sm_node_style->duplicate();
	sm_node_end_style->set_border_width_all(1 * EDSCALE);
	sm_node_end_style->set_border_color(error_color);
	theme->set_stylebox("node_frame_end", "GraphStateMachine", sm_node_end_style);

	theme->set_font("node_title_font", "GraphStateMachine", theme->get_font(SNAME("font"), SNAME("Label")));
	theme->set_font_size("node_title_font_size", "GraphStateMachine", theme->get_font_size(SNAME("font_size"), SNAME("Label")));
	theme->set_color("node_title_font_color", "GraphStateMachine", font_color);

	theme->set_color("transition_color", "GraphStateMachine", font_color);
	theme->set_color("transition_disabled_color", "GraphStateMachine", font_color * Color(1, 1, 1, 0.2));
	theme->set_color("transition_icon_color", "GraphStateMachine", Color(1, 1, 1));
	theme->set_color("transition_icon_disabled_color", "GraphStateMachine", Color(1, 1, 1, 0.2));
	theme->set_color("highlight_color", "GraphStateMachine", accent_color);
	theme->set_color("highlight_disabled_color", "GraphStateMachine", accent_color * Color(1, 1, 1, 0.6));
	theme->set_color("guideline_color", "GraphStateMachine", font_color * Color(1, 1, 1, 0.3));

	theme->set_color("playback_color", "GraphStateMachine", font_color);
	theme->set_color("playback_background_color", "GraphStateMachine", font_color * Color(1, 1, 1, 0.3));

	// GridContainer
	theme->set_constant("v_separation", "GridContainer", Math::round(widget_default_margin.y - 2 * EDSCALE));

	// FileDialog
	theme->set_icon("folder", "FileDialog", theme->get_icon(SNAME("Folder"), EditorStringName(EditorIcons)));
	theme->set_icon("parent_folder", "FileDialog", theme->get_icon(SNAME("ArrowUp"), EditorStringName(EditorIcons)));
	theme->set_icon("back_folder", "FileDialog", theme->get_icon(SNAME("Back"), EditorStringName(EditorIcons)));
	theme->set_icon("forward_folder", "FileDialog", theme->get_icon(SNAME("Forward"), EditorStringName(EditorIcons)));
	theme->set_icon("reload", "FileDialog", theme->get_icon(SNAME("Reload"), EditorStringName(EditorIcons)));
	theme->set_icon("toggle_hidden", "FileDialog", theme->get_icon(SNAME("GuiVisibilityVisible"), EditorStringName(EditorIcons)));
	// Use a different color for folder icons to make them easier to distinguish from files.
	// On a light theme, the icon will be dark, so we need to lighten it before blending it with the accent color.
	theme->set_color("folder_icon_color", "FileDialog", (dark_theme ? Color(1, 1, 1) : Color(4.25, 4.25, 4.25)).lerp(accent_color, 0.7));
	theme->set_color("files_disabled", "FileDialog", font_disabled_color);

	// ColorPicker
	theme->set_constant("margin", "ColorPicker", default_margin_size);
	theme->set_constant("sv_width", "ColorPicker", 256 * EDSCALE);
	theme->set_constant("sv_height", "ColorPicker", 256 * EDSCALE);
	theme->set_constant("h_width", "ColorPicker", 30 * EDSCALE);
	theme->set_constant("label_width", "ColorPicker", 10 * EDSCALE);
	theme->set_constant("center_slider_grabbers", "ColorPicker", 1);
	theme->set_icon("screen_picker", "ColorPicker", theme->get_icon(SNAME("ColorPick"), EditorStringName(EditorIcons)));
	theme->set_icon("shape_circle", "ColorPicker", theme->get_icon(SNAME("PickerShapeCircle"), EditorStringName(EditorIcons)));
	theme->set_icon("shape_rect", "ColorPicker", theme->get_icon(SNAME("PickerShapeRectangle"), EditorStringName(EditorIcons)));
	theme->set_icon("shape_rect_wheel", "ColorPicker", theme->get_icon(SNAME("PickerShapeRectangleWheel"), EditorStringName(EditorIcons)));
	theme->set_icon("add_preset", "ColorPicker", theme->get_icon(SNAME("Add"), EditorStringName(EditorIcons)));
	theme->set_icon("sample_bg", "ColorPicker", theme->get_icon(SNAME("GuiMiniCheckerboard"), EditorStringName(EditorIcons)));
	theme->set_icon("overbright_indicator", "ColorPicker", theme->get_icon(SNAME("OverbrightIndicator"), EditorStringName(EditorIcons)));
	theme->set_icon("bar_arrow", "ColorPicker", theme->get_icon(SNAME("ColorPickerBarArrow"), EditorStringName(EditorIcons)));
	theme->set_icon("picker_cursor", "ColorPicker", theme->get_icon(SNAME("PickerCursor"), EditorStringName(EditorIcons)));

	// ColorPickerButton
	theme->set_icon("bg", "ColorPickerButton", theme->get_icon(SNAME("GuiMiniCheckerboard"), EditorStringName(EditorIcons)));

	// ColorPresetButton
	Ref<StyleBoxFlat> preset_sb = make_flat_stylebox(Color(1, 1, 1), 2, 2, 2, 2, 2);
	theme->set_stylebox("preset_fg", "ColorPresetButton", preset_sb);
	theme->set_icon("preset_bg", "ColorPresetButton", theme->get_icon(SNAME("GuiMiniCheckerboard"), EditorStringName(EditorIcons)));
	theme->set_icon("overbright_indicator", "ColorPresetButton", theme->get_icon(SNAME("OverbrightIndicator"), EditorStringName(EditorIcons)));

	// Information on 3D viewport
	Ref<StyleBoxFlat> style_info_3d_viewport = style_default->duplicate();
	style_info_3d_viewport->set_bg_color(style_info_3d_viewport->get_bg_color() * Color(1, 1, 1, 0.5));
	style_info_3d_viewport->set_border_width_all(0);
	theme->set_stylebox("Information3dViewport", EditorStringName(EditorStyles), style_info_3d_viewport);

	// Asset Library.
	theme->set_stylebox("bg", "AssetLib", style_empty);
	theme->set_stylebox("panel", "AssetLib", style_content_panel);
	theme->set_color("status_color", "AssetLib", Color(0.5, 0.5, 0.5));
	theme->set_icon("dismiss", "AssetLib", theme->get_icon(SNAME("Close"), EditorStringName(EditorIcons)));

	// Theme editor.
	theme->set_color("preview_picker_overlay_color", "ThemeEditor", Color(0.1, 0.1, 0.1, 0.25));
	Color theme_preview_picker_bg_color = accent_color;
	theme_preview_picker_bg_color.a = 0.2;
	Ref<StyleBoxFlat> theme_preview_picker_sb = make_flat_stylebox(theme_preview_picker_bg_color, 0, 0, 0, 0);
	theme_preview_picker_sb->set_border_color(accent_color);
	theme_preview_picker_sb->set_border_width_all(1.0 * EDSCALE);
	theme->set_stylebox("preview_picker_overlay", "ThemeEditor", theme_preview_picker_sb);
	Color theme_preview_picker_label_bg_color = accent_color;
	theme_preview_picker_label_bg_color.set_v(0.5);
	Ref<StyleBoxFlat> theme_preview_picker_label_sb = make_flat_stylebox(theme_preview_picker_label_bg_color, 4.0, 1.0, 4.0, 3.0);
	theme->set_stylebox("preview_picker_label", "ThemeEditor", theme_preview_picker_label_sb);

	// Dictionary editor add item.
	// Expand to the left and right by 4px to compensate for the dictionary editor margins.
	Ref<StyleBoxFlat> style_dictionary_add_item = make_flat_stylebox(prop_subsection_color, 0, 4, 0, 4, corner_radius);
	style_dictionary_add_item->set_expand_margin(SIDE_LEFT, 4 * EDSCALE);
	style_dictionary_add_item->set_expand_margin(SIDE_RIGHT, 4 * EDSCALE);
	theme->set_stylebox("DictionaryAddItem", EditorStringName(EditorStyles), style_dictionary_add_item);

	Ref<StyleBoxEmpty> vshader_label_style = make_empty_stylebox(2, 1, 2, 1);
	theme->set_stylebox("label_style", "VShaderEditor", vshader_label_style);

	// Project manager.
	theme->set_stylebox("search_panel", "ProjectManager", style_tree_bg);
	theme->set_constant("sidebar_button_icon_separation", "ProjectManager", int(6 * EDSCALE));

	// adaptive script theme constants
	// for comments and elements with lower relevance
	const Color dim_color = Color(font_color, 0.5);

	const float mono_value = mono_color.r;
	const Color alpha1 = Color(mono_value, mono_value, mono_value, 0.07);
	const Color alpha2 = Color(mono_value, mono_value, mono_value, 0.14);
	const Color alpha3 = Color(mono_value, mono_value, mono_value, 0.27);

	const Color symbol_color = dark_theme ? Color(0.67, 0.79, 1) : Color(0, 0, 0.61);
	const Color keyword_color = dark_theme ? Color(1.0, 0.44, 0.52) : Color(0.9, 0.135, 0.51);
	const Color control_flow_keyword_color = dark_theme ? Color(1.0, 0.55, 0.8) : Color(0.743, 0.12, 0.8);
	const Color base_type_color = dark_theme ? Color(0.26, 1.0, 0.76) : Color(0, 0.6, 0.2);
	const Color engine_type_color = dark_theme ? Color(0.56, 1, 0.86) : Color(0.11, 0.55, 0.4);
	const Color user_type_color = dark_theme ? Color(0.78, 1, 0.93) : Color(0.18, 0.45, 0.4);
	const Color comment_color = dark_theme ? dim_color : Color(0.08, 0.08, 0.08, 0.5);
	const Color doc_comment_color = dark_theme ? Color(0.6, 0.7, 0.8, 0.8) : Color(0.15, 0.15, 0.4, 0.7);
	const Color string_color = dark_theme ? Color(1, 0.93, 0.63) : Color(0.6, 0.42, 0);

	// Use the brightest background color on a light theme (which generally uses a negative contrast rate).
	const Color te_background_color = dark_theme ? background_color : dark_color_3;
	const Color completion_background_color = dark_theme ? base_color : background_color;
	const Color completion_selected_color = alpha1;
	const Color completion_existing_color = alpha2;
	// Same opacity as the scroll grabber editor icon.
	const Color completion_scroll_color = Color(mono_value, mono_value, mono_value, 0.29);
	const Color completion_scroll_hovered_color = Color(mono_value, mono_value, mono_value, 0.4);
	const Color completion_font_color = font_color;
	const Color text_color = font_color;
	const Color line_number_color = dim_color;
	const Color safe_line_number_color = dark_theme ? (dim_color * Color(1, 1.2, 1, 1.5)) : Color(0, 0.4, 0, 0.75);
	const Color caret_color = mono_color;
	const Color caret_background_color = mono_color.inverted();
	const Color text_selected_color = Color(0, 0, 0, 0);
	const Color brace_mismatch_color = dark_theme ? error_color : Color(1, 0.08, 0, 1);
	const Color current_line_color = alpha1;
	const Color line_length_guideline_color = dark_theme ? base_color : background_color;
	const Color word_highlighted_color = alpha1;
	const Color number_color = dark_theme ? Color(0.63, 1, 0.88) : Color(0, 0.55, 0.28, 1);
	const Color function_color = dark_theme ? Color(0.34, 0.7, 1.0) : Color(0, 0.225, 0.9, 1);
	const Color member_variable_color = dark_theme ? Color(0.34, 0.7, 1.0).lerp(mono_color, 0.6) : Color(0, 0.4, 0.68, 1);
	const Color mark_color = Color(error_color.r, error_color.g, error_color.b, 0.3);
	const Color bookmark_color = Color(0.08, 0.49, 0.98);
	const Color breakpoint_color = dark_theme ? error_color : Color(1, 0.27, 0.2, 1);
	const Color executing_line_color = Color(0.98, 0.89, 0.27);
	const Color code_folding_color = alpha3;
	const Color folded_code_region_color = Color(0.68, 0.46, 0.77, 0.2);
	const Color search_result_color = alpha1;
	const Color search_result_border_color = dark_theme ? Color(0.41, 0.61, 0.91, 0.38) : Color(0, 0.4, 1, 0.38);

	EditorSettings *setting = EditorSettings::get_singleton();
	String text_editor_color_theme = setting->get("text_editor/theme/color_theme");
	if (text_editor_color_theme == "Default") {
		setting->set_initial_value("text_editor/theme/highlighting/symbol_color", symbol_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/keyword_color", keyword_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/control_flow_keyword_color", control_flow_keyword_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/base_type_color", base_type_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/engine_type_color", engine_type_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/user_type_color", user_type_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/comment_color", comment_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/doc_comment_color", doc_comment_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/string_color", string_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/background_color", te_background_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/completion_background_color", completion_background_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/completion_selected_color", completion_selected_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/completion_existing_color", completion_existing_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/completion_scroll_color", completion_scroll_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/completion_scroll_hovered_color", completion_scroll_hovered_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/completion_font_color", completion_font_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/text_color", text_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/line_number_color", line_number_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/safe_line_number_color", safe_line_number_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/caret_color", caret_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/caret_background_color", caret_background_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/text_selected_color", text_selected_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/selection_color", selection_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/brace_mismatch_color", brace_mismatch_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/current_line_color", current_line_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/line_length_guideline_color", line_length_guideline_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/word_highlighted_color", word_highlighted_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/number_color", number_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/function_color", function_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/member_variable_color", member_variable_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/mark_color", mark_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/bookmark_color", bookmark_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/breakpoint_color", breakpoint_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/executing_line_color", executing_line_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/code_folding_color", code_folding_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/folded_code_region_color", folded_code_region_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/search_result_color", search_result_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/search_result_border_color", search_result_border_color, true);
	} else if (text_editor_color_theme == "Godot 2") {
		setting->load_text_editor_theme();
	}

	// Now theme is loaded, apply it to CodeEdit.
	theme->set_font("font", "CodeEdit", theme->get_font(SNAME("source"), EditorStringName(EditorFonts)));
	theme->set_font_size("font_size", "CodeEdit", theme->get_font_size(SNAME("source_size"), EditorStringName(EditorFonts)));

	Ref<StyleBoxFlat> code_edit_stylebox = make_flat_stylebox(EDITOR_GET("text_editor/theme/highlighting/background_color"), widget_default_margin.x, widget_default_margin.y, widget_default_margin.x, widget_default_margin.y, corner_radius);
	theme->set_stylebox("normal", "CodeEdit", code_edit_stylebox);
	theme->set_stylebox("read_only", "CodeEdit", code_edit_stylebox);
	theme->set_stylebox("focus", "CodeEdit", Ref<StyleBoxEmpty>(memnew(StyleBoxEmpty)));

	theme->set_icon("tab", "CodeEdit", theme->get_icon(SNAME("GuiTab"), EditorStringName(EditorIcons)));
	theme->set_icon("space", "CodeEdit", theme->get_icon(SNAME("GuiSpace"), EditorStringName(EditorIcons)));
	theme->set_icon("folded", "CodeEdit", theme->get_icon(SNAME("CodeFoldedRightArrow"), EditorStringName(EditorIcons)));
	theme->set_icon("can_fold", "CodeEdit", theme->get_icon(SNAME("CodeFoldDownArrow"), EditorStringName(EditorIcons)));
	theme->set_icon("folded_code_region", "CodeEdit", theme->get_icon(SNAME("CodeRegionFoldedRightArrow"), EditorStringName(EditorIcons)));
	theme->set_icon("can_fold_code_region", "CodeEdit", theme->get_icon(SNAME("CodeRegionFoldDownArrow"), EditorStringName(EditorIcons)));
	theme->set_icon("executing_line", "CodeEdit", theme->get_icon(SNAME("TextEditorPlay"), EditorStringName(EditorIcons)));
	theme->set_icon("breakpoint", "CodeEdit", theme->get_icon(SNAME("Breakpoint"), EditorStringName(EditorIcons)));

	theme->set_constant("line_spacing", "CodeEdit", EDITOR_GET("text_editor/appearance/whitespace/line_spacing"));

	theme->set_color("background_color", "CodeEdit", Color(0, 0, 0, 0));
	theme->set_color("completion_background_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/completion_background_color"));
	theme->set_color("completion_selected_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/completion_selected_color"));
	theme->set_color("completion_existing_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/completion_existing_color"));
	theme->set_color("completion_scroll_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/completion_scroll_color"));
	theme->set_color("completion_scroll_hovered_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/completion_scroll_hovered_color"));
	theme->set_color("font_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/text_color"));
	theme->set_color("line_number_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/line_number_color"));
	theme->set_color("caret_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/caret_color"));
	theme->set_color("font_selected_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/text_selected_color"));
	theme->set_color("selection_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/selection_color"));
	theme->set_color("brace_mismatch_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/brace_mismatch_color"));
	theme->set_color("current_line_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/current_line_color"));
	theme->set_color("line_length_guideline_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/line_length_guideline_color"));
	theme->set_color("word_highlighted_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/word_highlighted_color"));
	theme->set_color("bookmark_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/bookmark_color"));
	theme->set_color("breakpoint_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/breakpoint_color"));
	theme->set_color("executing_line_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/executing_line_color"));
	theme->set_color("code_folding_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/code_folding_color"));
	theme->set_color("folded_code_region_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/folded_code_region_color"));
	theme->set_color("search_result_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/search_result_color"));
	theme->set_color("search_result_border_color", "CodeEdit", EDITOR_GET("text_editor/theme/highlighting/search_result_border_color"));

	OS::get_singleton()->benchmark_end_measure("create_editor_theme");

	return theme;
}

Ref<Theme> create_custom_theme(const Ref<Theme> p_theme) {
	OS::get_singleton()->benchmark_begin_measure("create_custom_theme");
	Ref<Theme> theme = create_editor_theme(p_theme);

	const String custom_theme_path = EDITOR_GET("interface/theme/custom_theme");
	if (!custom_theme_path.is_empty()) {
		Ref<Theme> custom_theme = ResourceLoader::load(custom_theme_path);
		if (custom_theme.is_valid()) {
			theme->merge_with(custom_theme);
		}
	}

	OS::get_singleton()->benchmark_end_measure("create_custom_theme");
	return theme;
}

/**
 * Returns the SVG code for the default project icon.
 */
String get_default_project_icon() {
	for (int i = 0; i < editor_icons_count; i++) {
		if (strcmp(editor_icons_names[i], "DefaultProjectIcon") == 0) {
			return String(editor_icons_sources[i]);
		}
	}
	return String();
}
