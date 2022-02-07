/*************************************************************************/
/*  editor_themes.cpp                                                    */
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

#include "editor_themes.h"

#include "core/error/error_macros.h"
#include "core/io/resource_loader.h"
#include "core/variant/dictionary.h"
#include "editor_fonts.h"
#include "editor_icons.gen.h"
#include "editor_scale.h"
#include "editor_settings.h"

#include "modules/modules_enabled.gen.h" // For svg.
#ifdef MODULE_SVG_ENABLED
#include "modules/svg/image_loader_svg.h"
#endif

static Ref<StyleBoxTexture> make_stylebox(Ref<Texture2D> p_texture, float p_left, float p_top, float p_right, float p_bottom, float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1, bool p_draw_center = true) {
	Ref<StyleBoxTexture> style(memnew(StyleBoxTexture));
	style->set_texture(p_texture);
	style->set_margin_size(SIDE_LEFT, p_left * EDSCALE);
	style->set_margin_size(SIDE_RIGHT, p_right * EDSCALE);
	style->set_margin_size(SIDE_BOTTOM, p_bottom * EDSCALE);
	style->set_margin_size(SIDE_TOP, p_top * EDSCALE);
	style->set_default_margin(SIDE_LEFT, p_margin_left * EDSCALE);
	style->set_default_margin(SIDE_RIGHT, p_margin_right * EDSCALE);
	style->set_default_margin(SIDE_BOTTOM, p_margin_bottom * EDSCALE);
	style->set_default_margin(SIDE_TOP, p_margin_top * EDSCALE);
	style->set_draw_center(p_draw_center);
	return style;
}

static Ref<StyleBoxEmpty> make_empty_stylebox(float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1) {
	Ref<StyleBoxEmpty> style(memnew(StyleBoxEmpty));
	style->set_default_margin(SIDE_LEFT, p_margin_left * EDSCALE);
	style->set_default_margin(SIDE_RIGHT, p_margin_right * EDSCALE);
	style->set_default_margin(SIDE_BOTTOM, p_margin_bottom * EDSCALE);
	style->set_default_margin(SIDE_TOP, p_margin_top * EDSCALE);
	return style;
}

static Ref<StyleBoxFlat> make_flat_stylebox(Color p_color, float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1, int p_corner_width = 0) {
	Ref<StyleBoxFlat> style(memnew(StyleBoxFlat));
	style->set_bg_color(p_color);
	// Adjust level of detail based on the corners' effective sizes.
	style->set_corner_detail(Math::ceil(1.5 * p_corner_width * EDSCALE));
	style->set_corner_radius_all(p_corner_width);
	style->set_default_margin(SIDE_LEFT, p_margin_left * EDSCALE);
	style->set_default_margin(SIDE_RIGHT, p_margin_right * EDSCALE);
	style->set_default_margin(SIDE_BOTTOM, p_margin_bottom * EDSCALE);
	style->set_default_margin(SIDE_TOP, p_margin_top * EDSCALE);
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

static Ref<Texture2D> flip_icon(Ref<Texture2D> p_texture, bool p_flip_y = false, bool p_flip_x = false) {
	if (!p_flip_y && !p_flip_x) {
		return p_texture;
	}

	Ref<ImageTexture> texture(memnew(ImageTexture));
	Ref<Image> img = p_texture->get_image();
	ERR_FAIL_NULL_V(img, Ref<Texture2D>());
	img = img->duplicate();

	if (p_flip_y) {
		img->flip_y();
	}
	if (p_flip_x) {
		img->flip_x();
	}

	texture->create_from_image(img);
	return texture;
}

#ifdef MODULE_SVG_ENABLED
// See also `generate_icon()` in `scene/resources/default_theme.cpp`.
static Ref<ImageTexture> editor_generate_icon(int p_index, bool p_convert_color, float p_scale = EDSCALE, float p_saturation = 1.0, Dictionary p_convert_colors = Dictionary()) {
	Ref<ImageTexture> icon = memnew(ImageTexture);
	Ref<Image> img = memnew(Image);

	// Upsample icon generation only if the editor scale isn't an integer multiplier.
	// Generating upsampled icons is slower, and the benefit is hardly visible
	// with integer editor scales.
	const bool upsample = !Math::is_equal_approx(Math::round(p_scale), p_scale);
	ImageLoaderSVG img_loader;
	img_loader.set_replace_colors(p_convert_colors);
	img_loader.create_image_from_string(img, editor_icons_sources[p_index], p_scale, upsample, p_convert_color);
	if (p_saturation != 1.0) {
		img->adjust_bcs(1.0, 1.0, p_saturation);
	}
	icon->create_from_image(img); // in this case filter really helps

	return icon;
}
#endif

#ifndef ADD_CONVERT_COLOR
#define ADD_CONVERT_COLOR(dictionary, old_color, new_color) dictionary[Color::html(old_color)] = Color::html(new_color)
#endif

void editor_register_and_generate_icons(Ref<Theme> p_theme, bool p_dark_theme = true, int p_thumb_size = 32, bool p_only_thumbs = false, float p_icon_saturation = 1.0) {
#ifdef MODULE_SVG_ENABLED
	// The default icon theme is designed to be used for a dark theme.
	// This dictionary stores Color values to convert to other colors
	// for better readability on a light theme.
	// Godot Color values are used to avoid the ambiguity of strings
	// (where "#ffffff", "fff", and "white" are all equivalent).
	Dictionary dark_icon_color_dictionary;

	// The names of the icons to never convert, even if one of their colors
	// are contained in the dictionary above.
	Set<StringName> exceptions;

	// Some of the colors below are listed for completeness sake.
	// This can be a basis for proper palette validation later.
	if (!p_dark_theme) {
		// Convert color:                             FROM       TO
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#478cbf", "#478cbf"); // Godot Blue
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#414042", "#414042"); // Godot Gray

		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ffffff", "#414141"); // Pure white
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#000000", "#bfbfbf"); // Pure black
		// Keep pure RGB colors as is, but list them for explicity.
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ff0000", "#ff0000"); // Pure red
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#00ff00", "#00ff00"); // Pure green
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#0000ff", "#0000ff"); // Pure blue

		// GUI Colors
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#e0e0e0", "#5a5a5a"); // Common icon color
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#fefefe", "#fefefe"); // Forced light color
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#808080", "#808080"); // GUI disabled color
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#b3b3b3", "#363636"); // GUI disabled light color
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#699ce8", "#699ce8"); // GUI highlight color
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#f9f9f9", "#606060"); // Scrollbar grabber highlight color

		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#c38ef1", "#a85de9"); // Animation
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#fc7f7f", "#cd3838"); // Spatial
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#8da5f3", "#3d64dd"); // 2D
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#4b70ea", "#1a3eac"); // 2D Dark
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#8eef97", "#2fa139"); // Control

		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#5fb2ff", "#0079f0"); // Selection (blue)
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#003e7a", "#2b74bb"); // Selection (darker blue)
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#f7f5cf", "#615f3a"); // Gizmo (yellow)

		// Rainbow
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ff4545", "#ff2929"); // Red
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ffe345", "#ffe337"); // Yellow
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#80ff45", "#74ff34"); // Green
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#45ffa2", "#2cff98"); // Aqua
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#45d7ff", "#22ccff"); // Blue
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#8045ff", "#702aff"); // Purple
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ff4596", "#ff2781"); // Pink

		// Audio gradients
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#e1da5b", "#d6cf4b"); // Yellow

		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#62aeff", "#1678e0"); // Frozen gradient top
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#75d1e6", "#41acc5"); // Frozen gradient middle
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#84ffee", "#49ccba"); // Frozen gradient bottom

		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#f70000", "#c91616"); // Color track red
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#eec315", "#d58c0b"); // Color track orange
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#dbee15", "#b7d10a"); // Color track yellow
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#288027", "#218309"); // Color track green

		// Resource groups
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ffca5f", "#fea900"); // Mesh resource (orange)
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#2998ff", "#68b6ff"); // Shape resource (blue)
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#a2d2ff", "#4998e3"); // Shape resource (light blue)

		// Animation editor tracks
		// The property track icon color is set by the common icon color.
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ea7940", "#bd5e2c"); // 3D Position track
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ff2b88", "#bd165f"); // 3D Rotation track
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#eac840", "#bd9d1f"); // 3D Scale track
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#3cf34e", "#16a827"); // Call Method track
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#2877f6", "#236be6"); // Bezier Curve track
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#eae440", "#9f9722"); // Audio Playback track
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#a448f0", "#9853ce"); // Animation Playback track
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#5ad5c4", "#0a9c88"); // Blend Shape track

		// Control layouts
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#d6d6d6", "#474747"); // Highlighted part
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#474747", "#d6d6d6"); // Background part
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#919191", "#6e6e6e"); // Border part

		// TileSet editor icons
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#fce00e", "#aa8d24"); // New Single Tile
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#0e71fc", "#0350bd"); // New Autotile
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#c6ced4", "#828f9b"); // New Atlas

		// Visual script
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#41ecad", "#25e3a0"); // VisualScript variant
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#6f91f0", "#6d8eeb"); // VisualScript bool
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#5abbef", "#4fb2e9"); // VisualScript int
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#35d4f4", "#27ccf0"); // VisualScript float
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#4593ec", "#4690e7"); // VisualScript String
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ac73f1", "#ad76ee"); // VisualScript Vector2
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#f1738f", "#ee758e"); // VisualScript Rect2
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#de66f0", "#dc6aed"); // VisualScript Vector3
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#b9ec41", "#96ce1a"); // VisualScript Transform2D
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#f74949", "#f77070"); // VisualScript Plane
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ec418e", "#ec69a3"); // VisualScript Quat
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ee5677", "#ee7991"); // VisualScript AABB
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#e1ec41", "#b2bb19"); // VisualScript Basis
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#f68f45", "#f49047"); // VisualScript Transform
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#417aec", "#6993ec"); // VisualScript NodePath
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#41ec80", "#2ce573"); // VisualScript RID
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#55f3e3", "#12d5c3"); // VisualScript Object
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#54ed9e", "#57e99f"); // VisualScript Dictionary
		// Visual shaders
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#77ce57", "#67c046"); // Vector funcs
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ea686c", "#d95256"); // Vector transforms
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#eac968", "#d9b64f"); // Textures and cubemaps
		ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#cf68ea", "#c050dd"); // Functions and expressions

		exceptions.insert("EditorPivot");
		exceptions.insert("EditorHandle");
		exceptions.insert("Editor3DHandle");
		exceptions.insert("EditorBoneHandle");
		exceptions.insert("Godot");
		exceptions.insert("Sky");
		exceptions.insert("EditorControlAnchor");
		exceptions.insert("DefaultProjectIcon");
		exceptions.insert("GuiChecked");
		exceptions.insert("GuiRadioChecked");
		exceptions.insert("GuiIndeterminate");
		exceptions.insert("GuiCloseCustomizable");
		exceptions.insert("GuiGraphNodePort");
		exceptions.insert("GuiResizer");
		exceptions.insert("ZoomMore");
		exceptions.insert("ZoomLess");
		exceptions.insert("ZoomReset");
		exceptions.insert("LockViewport");
		exceptions.insert("GroupViewport");
		exceptions.insert("StatusError");
		exceptions.insert("StatusSuccess");
		exceptions.insert("StatusWarning");
		exceptions.insert("OverbrightIndicator");
	}

	// These ones should be converted even if we are using a dark theme.
	const Color error_color = p_theme->get_color(SNAME("error_color"), SNAME("Editor"));
	const Color success_color = p_theme->get_color(SNAME("success_color"), SNAME("Editor"));
	const Color warning_color = p_theme->get_color(SNAME("warning_color"), SNAME("Editor"));
	dark_icon_color_dictionary[Color::html("#ff5f5f")] = error_color;
	dark_icon_color_dictionary[Color::html("#5fff97")] = success_color;
	dark_icon_color_dictionary[Color::html("#ffdd65")] = warning_color;

	// Generate icons.
	if (!p_only_thumbs) {
		for (int i = 0; i < editor_icons_count; i++) {
			float saturation = p_icon_saturation;

			if (strcmp(editor_icons_names[i], "DefaultProjectIcon") == 0 || strcmp(editor_icons_names[i], "Godot") == 0 || strcmp(editor_icons_names[i], "Logo") == 0) {
				saturation = 1.0;
			}

			const int is_exception = exceptions.has(editor_icons_names[i]);
			const Ref<ImageTexture> icon = editor_generate_icon(i, !is_exception, EDSCALE, saturation, dark_icon_color_dictionary);

			p_theme->set_icon(editor_icons_names[i], SNAME("EditorIcons"), icon);
		}
	}

	// Generate thumbnail icons with the given thumbnail size.
	// We don't need filtering when generating at one of the default resolutions.
	const bool force_filter = p_thumb_size != 64 && p_thumb_size != 32;
	if (p_thumb_size >= 64) {
		const float scale = (float)p_thumb_size / 64.0 * EDSCALE;
		for (int i = 0; i < editor_bg_thumbs_count; i++) {
			const int index = editor_bg_thumbs_indices[i];
			const int is_exception = exceptions.has(editor_icons_names[index]);
			const Ref<ImageTexture> icon = editor_generate_icon(index, !p_dark_theme && !is_exception, scale, force_filter, dark_icon_color_dictionary);

			p_theme->set_icon(editor_icons_names[index], SNAME("EditorIcons"), icon);
		}
	} else {
		const float scale = (float)p_thumb_size / 32.0 * EDSCALE;
		for (int i = 0; i < editor_md_thumbs_count; i++) {
			const int index = editor_md_thumbs_indices[i];
			const bool is_exception = exceptions.has(editor_icons_names[index]);
			const Ref<ImageTexture> icon = editor_generate_icon(index, !p_dark_theme && !is_exception, scale, force_filter, dark_icon_color_dictionary);

			p_theme->set_icon(editor_icons_names[index], SNAME("EditorIcons"), icon);
		}
	}
#else
	WARN_PRINT("SVG support disabled, editor icons won't be rendered.");
#endif
}

Ref<Theme> create_editor_theme(const Ref<Theme> p_theme) {
	Ref<Theme> theme = Ref<Theme>(memnew(Theme));

	// Controls may rely on the scale for their internal drawing logic.
	theme->set_default_base_scale(EDSCALE);

	// Theme settings
	Color accent_color = EDITOR_GET("interface/theme/accent_color");
	Color base_color = EDITOR_GET("interface/theme/base_color");
	float contrast = EDITOR_GET("interface/theme/contrast");
	float icon_saturation = EDITOR_GET("interface/theme/icon_saturation");
	float relationship_line_opacity = EDITOR_GET("interface/theme/relationship_line_opacity");

	String preset = EDITOR_GET("interface/theme/preset");

	int border_size = EDITOR_GET("interface/theme/border_size");
	int corner_radius = EDITOR_GET("interface/theme/corner_radius");

	Color preset_accent_color;
	Color preset_base_color;
	float preset_contrast = 0;

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
	} else if (preset == "Grey") {
		preset_accent_color = Color(0.72, 0.89, 1.00);
		preset_base_color = Color(0.24, 0.24, 0.24);
		preset_contrast = default_contrast;
	} else if (preset == "Light") {
		preset_accent_color = Color(0.18, 0.50, 1.00);
		preset_base_color = Color(0.9, 0.9, 0.9);
		// A negative contrast rate looks better for light themes, since it better follows the natural order of UI "elevation".
		preset_contrast = -0.08;
	} else if (preset == "Solarized (Dark)") {
		preset_accent_color = Color(0.15, 0.55, 0.82);
		preset_base_color = Color(0.04, 0.23, 0.27);
		preset_contrast = default_contrast;
	} else if (preset == "Solarized (Light)") {
		preset_accent_color = Color(0.15, 0.55, 0.82);
		preset_base_color = Color(0.89, 0.86, 0.79);
		// A negative contrast rate looks better for light themes, since it better follows the natural order of UI "elevation".
		preset_contrast = -0.08;
	} else { // Default
		preset_accent_color = Color(0.44, 0.73, 0.98);
		preset_base_color = Color(0.21, 0.24, 0.29);
		preset_contrast = default_contrast;
	}

	if (preset != "Custom") {
		accent_color = preset_accent_color;
		base_color = preset_base_color;
		contrast = preset_contrast;
		EditorSettings::get_singleton()->set_initial_value("interface/theme/accent_color", accent_color);
		EditorSettings::get_singleton()->set_initial_value("interface/theme/base_color", base_color);
		EditorSettings::get_singleton()->set_initial_value("interface/theme/contrast", contrast);
	}

	EditorSettings::get_singleton()->set_manually("interface/theme/preset", preset);
	EditorSettings::get_singleton()->set_manually("interface/theme/accent_color", accent_color);
	EditorSettings::get_singleton()->set_manually("interface/theme/base_color", base_color);
	EditorSettings::get_singleton()->set_manually("interface/theme/contrast", contrast);

	// Colors
	bool dark_theme = EditorSettings::get_singleton()->is_dark_theme();

	const Color dark_color_1 = base_color.lerp(Color(0, 0, 0, 1), contrast);
	const Color dark_color_2 = base_color.lerp(Color(0, 0, 0, 1), contrast * 1.5);
	const Color dark_color_3 = base_color.lerp(Color(0, 0, 0, 1), contrast * 2);

	const Color background_color = dark_color_2;

	// White (dark theme) or black (light theme), will be used to generate the rest of the colors
	const Color mono_color = dark_theme ? Color(1, 1, 1) : Color(0, 0, 0);

	const Color contrast_color_1 = base_color.lerp(mono_color, MAX(contrast, default_contrast));
	const Color contrast_color_2 = base_color.lerp(mono_color, MAX(contrast * 1.5, default_contrast * 1.5));

	const Color font_color = mono_color.lerp(base_color, 0.25);
	const Color font_hover_color = mono_color.lerp(base_color, 0.125);
	const Color font_focus_color = mono_color.lerp(base_color, 0.125);
	const Color font_disabled_color = Color(mono_color.r, mono_color.g, mono_color.b, 0.3);
	const Color font_readonly_color = Color(mono_color.r, mono_color.g, mono_color.b, 0.65);
	const Color font_placeholder_color = Color(mono_color.r, mono_color.g, mono_color.b, 0.6);
	const Color selection_color = accent_color * Color(1, 1, 1, 0.4);
	const Color disabled_color = mono_color.inverted().lerp(base_color, 0.7);
	const Color disabled_bg_color = mono_color.inverted().lerp(base_color, 0.9);

	Color icon_hover_color = Color(1, 1, 1) * (dark_theme ? 1.15 : 1.45);
	icon_hover_color.a = 1.0;
	Color icon_focus_color = icon_hover_color;
	// Make the pressed icon color overbright because icons are not completely white on a dark theme.
	// On a light theme, icons are dark, so we need to modulate them with an even brighter color.
	Color icon_pressed_color = accent_color * (dark_theme ? 1.15 : 3.5);
	icon_pressed_color.a = 1.0;

	const Color separator_color = Color(mono_color.r, mono_color.g, mono_color.b, 0.1);
	const Color highlight_color = Color(accent_color.r, accent_color.g, accent_color.b, 0.275);
	const Color disabled_highlight_color = highlight_color.lerp(dark_theme ? Color(0, 0, 0) : Color(1, 1, 1), 0.5);

	float prev_icon_saturation = theme->has_color(SNAME("icon_saturation"), SNAME("Editor")) ? theme->get_color(SNAME("icon_saturation"), SNAME("Editor")).r : 1.0;

	theme->set_color(SNAME("icon_saturation"), SNAME("Editor"), Color(icon_saturation, icon_saturation, icon_saturation)); // can't save single float in theme, so using color
	theme->set_color(SNAME("accent_color"), SNAME("Editor"), accent_color);
	theme->set_color(SNAME("highlight_color"), SNAME("Editor"), highlight_color);
	theme->set_color(SNAME("disabled_highlight_color"), SNAME("Editor"), disabled_highlight_color);
	theme->set_color(SNAME("base_color"), SNAME("Editor"), base_color);
	theme->set_color(SNAME("dark_color_1"), SNAME("Editor"), dark_color_1);
	theme->set_color(SNAME("dark_color_2"), SNAME("Editor"), dark_color_2);
	theme->set_color(SNAME("dark_color_3"), SNAME("Editor"), dark_color_3);
	theme->set_color(SNAME("contrast_color_1"), SNAME("Editor"), contrast_color_1);
	theme->set_color(SNAME("contrast_color_2"), SNAME("Editor"), contrast_color_2);
	theme->set_color(SNAME("box_selection_fill_color"), SNAME("Editor"), accent_color * Color(1, 1, 1, 0.3));
	theme->set_color(SNAME("box_selection_stroke_color"), SNAME("Editor"), accent_color * Color(1, 1, 1, 0.8));

	theme->set_color(SNAME("axis_x_color"), SNAME("Editor"), Color(0.96, 0.20, 0.32));
	theme->set_color(SNAME("axis_y_color"), SNAME("Editor"), Color(0.53, 0.84, 0.01));
	theme->set_color(SNAME("axis_z_color"), SNAME("Editor"), Color(0.16, 0.55, 0.96));

	theme->set_color(SNAME("font_color"), SNAME("Editor"), font_color);
	theme->set_color(SNAME("highlighted_font_color"), SNAME("Editor"), font_hover_color);
	theme->set_color(SNAME("disabled_font_color"), SNAME("Editor"), font_disabled_color);

	theme->set_color(SNAME("mono_color"), SNAME("Editor"), mono_color);

	Color success_color = Color(0.45, 0.95, 0.5);
	Color warning_color = Color(1, 0.87, 0.4);
	Color error_color = Color(1, 0.47, 0.42);
	Color property_color = font_color.lerp(Color(0.5, 0.5, 0.5), 0.5);
	Color readonly_color = property_color.lerp(dark_theme ? Color(0, 0, 0) : Color(1, 1, 1), 0.5);
	Color readonly_warning_color = error_color.lerp(dark_theme ? Color(0, 0, 0) : Color(1, 1, 1), 0.5);

	if (!dark_theme) {
		// Darken some colors to be readable on a light background
		success_color = success_color.lerp(mono_color, 0.35);
		warning_color = warning_color.lerp(mono_color, 0.35);
		error_color = error_color.lerp(mono_color, 0.25);
	}

	theme->set_color(SNAME("success_color"), SNAME("Editor"), success_color);
	theme->set_color(SNAME("warning_color"), SNAME("Editor"), warning_color);
	theme->set_color(SNAME("error_color"), SNAME("Editor"), error_color);
	theme->set_color(SNAME("property_color"), SNAME("Editor"), property_color);
	theme->set_color(SNAME("readonly_color"), SNAME("Editor"), readonly_color);

	if (!dark_theme) {
		theme->set_color(SNAME("vulkan_color"), SNAME("Editor"), Color::hex(0xad1128ff));
	} else {
		theme->set_color(SNAME("vulkan_color"), SNAME("Editor"), Color(1.0, 0.0, 0.0));
	}
	const int thumb_size = EDITOR_GET("filesystem/file_dialog/thumbnail_size");
	theme->set_constant(SNAME("scale"), SNAME("Editor"), EDSCALE);
	theme->set_constant(SNAME("thumb_size"), SNAME("Editor"), thumb_size);
	theme->set_constant(SNAME("dark_theme"), SNAME("Editor"), dark_theme);

	// Register icons + font

	// The resolution and the icon color (dark_theme bool) has not changed, so we do not regenerate the icons.
	if (p_theme != nullptr && fabs(p_theme->get_constant(SNAME("scale"), SNAME("Editor")) - EDSCALE) < 0.00001 && (bool)p_theme->get_constant(SNAME("dark_theme"), SNAME("Editor")) == dark_theme && prev_icon_saturation == icon_saturation) {
		// Register already generated icons.
		for (int i = 0; i < editor_icons_count; i++) {
			theme->set_icon(editor_icons_names[i], SNAME("EditorIcons"), p_theme->get_icon(editor_icons_names[i], SNAME("EditorIcons")));
		}
	} else {
		editor_register_and_generate_icons(theme, dark_theme, thumb_size, false, icon_saturation);
	}
	// Thumbnail size has changed, so we regenerate the medium sizes
	if (p_theme != nullptr && fabs((double)p_theme->get_constant(SNAME("thumb_size"), SNAME("Editor")) - thumb_size) > 0.00001) {
		editor_register_and_generate_icons(p_theme, dark_theme, thumb_size, true);
	}

	editor_register_fonts(theme);

	// Ensure borders are visible when using an editor scale below 100%.
	const int border_width = CLAMP(border_size, 0, 2) * MAX(1, EDSCALE);
	const int corner_width = CLAMP(corner_radius, 0, 6);
	const int default_margin_size = 4;
	const int margin_size_extra = default_margin_size + CLAMP(border_size, 0, 2);

	// Styleboxes
	// This is the most commonly used stylebox, variations should be made as duplicate of this
	Ref<StyleBoxFlat> style_default = make_flat_stylebox(base_color, default_margin_size, default_margin_size, default_margin_size, default_margin_size, corner_width);
	// Work around issue about antialiased edges being blurrier (GH-35279).
	style_default->set_anti_aliased(false);
	style_default->set_border_width_all(border_width);
	style_default->set_border_color(base_color);
	style_default->set_draw_center(true);

	// Button and widgets
	const float extra_spacing = EDITOR_GET("interface/theme/additional_spacing");

	const Vector2 widget_default_margin = Vector2(extra_spacing + 6, extra_spacing + default_margin_size + 1) * EDSCALE;

	Ref<StyleBoxFlat> style_widget = style_default->duplicate();
	style_widget->set_default_margin(SIDE_LEFT, widget_default_margin.x);
	style_widget->set_default_margin(SIDE_TOP, widget_default_margin.y);
	style_widget->set_default_margin(SIDE_RIGHT, widget_default_margin.x);
	style_widget->set_default_margin(SIDE_BOTTOM, widget_default_margin.y);
	style_widget->set_bg_color(dark_color_1);
	style_widget->set_border_color(dark_color_2);

	Ref<StyleBoxFlat> style_widget_disabled = style_widget->duplicate();
	style_widget_disabled->set_border_color(disabled_color);
	style_widget_disabled->set_bg_color(disabled_bg_color);

	Ref<StyleBoxFlat> style_widget_focus = style_widget->duplicate();
	style_widget_focus->set_draw_center(false);
	style_widget_focus->set_border_width_all(Math::round(2 * MAX(1, EDSCALE)));
	style_widget_focus->set_border_color(accent_color);

	Ref<StyleBoxFlat> style_widget_pressed = style_widget->duplicate();
	style_widget_pressed->set_bg_color(dark_color_1.darkened(0.125));

	Ref<StyleBoxFlat> style_widget_hover = style_widget->duplicate();
	style_widget_hover->set_bg_color(mono_color * Color(1, 1, 1, 0.11));
	style_widget_hover->set_border_color(mono_color * Color(1, 1, 1, 0.05));

	// Style for windows, popups, etc..
	Ref<StyleBoxFlat> style_popup = style_default->duplicate();
	const int popup_margin_size = default_margin_size * EDSCALE * 3;
	style_popup->set_default_margin(SIDE_LEFT, popup_margin_size);
	style_popup->set_default_margin(SIDE_TOP, popup_margin_size);
	style_popup->set_default_margin(SIDE_RIGHT, popup_margin_size);
	style_popup->set_default_margin(SIDE_BOTTOM, popup_margin_size);
	style_popup->set_border_color(contrast_color_1);
	const Color shadow_color = Color(0, 0, 0, dark_theme ? 0.3 : 0.1);
	style_popup->set_shadow_color(shadow_color);
	style_popup->set_shadow_size(4 * EDSCALE);

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

	Ref<StyleBoxFlat> style_tab_selected = style_widget->duplicate();

	// Add a highlight line at the top of the selected tab.
	style_tab_selected->set_border_width_all(0);
	style_tab_selected->set_border_width(SIDE_TOP, Math::round(2 * EDSCALE));
	// Make the highlight line prominent, but not too prominent as to not be distracting.
	style_tab_selected->set_border_color(dark_color_2.lerp(accent_color, 0.75));
	// Don't round the top corners to avoid creating a small blank space between the tabs and the main panel.
	// This also makes the top highlight look better.
	style_tab_selected->set_corner_radius_all(0);

	// Prevent visible artifacts and cover the top-left rounded corner of the panel below the tab if selected
	// We can't prevent them with both rounded corners and non-zero border width, though
	style_tab_selected->set_expand_margin_size(SIDE_BOTTOM, corner_width > 0 ? corner_width : border_width);

	style_tab_selected->set_default_margin(SIDE_LEFT, widget_default_margin.x + 2 * EDSCALE);
	style_tab_selected->set_default_margin(SIDE_RIGHT, widget_default_margin.x + 2 * EDSCALE);
	style_tab_selected->set_default_margin(SIDE_BOTTOM, widget_default_margin.y);
	style_tab_selected->set_default_margin(SIDE_TOP, widget_default_margin.y);
	style_tab_selected->set_bg_color(base_color);

	Ref<StyleBoxFlat> style_tab_unselected = style_tab_selected->duplicate();
	style_tab_unselected->set_bg_color(dark_color_1);
	style_tab_unselected->set_expand_margin_size(SIDE_BOTTOM, 0);
	// Add some spacing between unselected tabs to make them easier to distinguish from each other
	style_tab_unselected->set_border_color(Color(0, 0, 0, 0));
	style_tab_unselected->set_border_width(SIDE_LEFT, Math::round(1 * EDSCALE));
	style_tab_unselected->set_border_width(SIDE_RIGHT, Math::round(1 * EDSCALE));
	style_tab_unselected->set_default_margin(SIDE_LEFT, widget_default_margin.x + 2 * EDSCALE);
	style_tab_unselected->set_default_margin(SIDE_RIGHT, widget_default_margin.x + 2 * EDSCALE);

	Ref<StyleBoxFlat> style_tab_disabled = style_tab_selected->duplicate();
	style_tab_disabled->set_bg_color(disabled_bg_color);
	style_tab_disabled->set_expand_margin_size(SIDE_BOTTOM, 0);
	style_tab_disabled->set_border_color(disabled_bg_color);

	// Editor background
	Color background_color_opaque = background_color;
	background_color_opaque.a = 1.0;
	theme->set_stylebox(SNAME("Background"), SNAME("EditorStyles"), make_flat_stylebox(background_color_opaque, default_margin_size, default_margin_size, default_margin_size, default_margin_size));

	// Focus
	theme->set_stylebox(SNAME("Focus"), SNAME("EditorStyles"), style_widget_focus);
	// Use a less opaque color to be less distracting for the 2D and 3D editor viewports.
	Ref<StyleBoxFlat> style_widget_focus_viewport = style_widget_focus->duplicate();
	style_widget_focus_viewport->set_border_color(accent_color * Color(1, 1, 1, 0.5));
	theme->set_stylebox(SNAME("FocusViewport"), SNAME("EditorStyles"), style_widget_focus_viewport);

	// Menu
	Ref<StyleBoxFlat> style_menu = style_widget->duplicate();
	style_menu->set_draw_center(false);
	style_menu->set_border_width_all(0);
	theme->set_stylebox(SNAME("panel"), SNAME("PanelContainer"), style_menu);
	theme->set_stylebox(SNAME("MenuPanel"), SNAME("EditorStyles"), style_menu);

	// CanvasItem Editor
	Ref<StyleBoxFlat> style_canvas_editor_info = make_flat_stylebox(Color(0.0, 0.0, 0.0, 0.2));
	style_canvas_editor_info->set_expand_margin_size_all(4 * EDSCALE);
	theme->set_stylebox(SNAME("CanvasItemInfoOverlay"), SNAME("EditorStyles"), style_canvas_editor_info);

	// Script Editor
	theme->set_stylebox(SNAME("ScriptEditorPanel"), SNAME("EditorStyles"), make_empty_stylebox(default_margin_size, 0, default_margin_size, default_margin_size));
	theme->set_stylebox(SNAME("ScriptEditor"), SNAME("EditorStyles"), make_empty_stylebox(0, 0, 0, 0));

	// Play button group
	theme->set_stylebox(SNAME("PlayButtonPanel"), SNAME("EditorStyles"), style_empty);

	theme->set_stylebox(SNAME("normal"), SNAME("MenuButton"), style_menu);
	theme->set_stylebox(SNAME("hover"), SNAME("MenuButton"), style_widget_hover);
	theme->set_stylebox(SNAME("pressed"), SNAME("MenuButton"), style_menu);
	theme->set_stylebox(SNAME("focus"), SNAME("MenuButton"), style_menu);
	theme->set_stylebox(SNAME("disabled"), SNAME("MenuButton"), style_menu);

	theme->set_color(SNAME("font_color"), SNAME("MenuButton"), font_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("MenuButton"), font_hover_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("MenuButton"), font_focus_color);

	theme->set_stylebox(SNAME("MenuHover"), SNAME("EditorStyles"), style_widget_hover);

	// Buttons
	theme->set_stylebox(SNAME("normal"), SNAME("Button"), style_widget);
	theme->set_stylebox(SNAME("hover"), SNAME("Button"), style_widget_hover);
	theme->set_stylebox(SNAME("pressed"), SNAME("Button"), style_widget_pressed);
	theme->set_stylebox(SNAME("focus"), SNAME("Button"), style_widget_focus);
	theme->set_stylebox(SNAME("disabled"), SNAME("Button"), style_widget_disabled);

	theme->set_color(SNAME("font_color"), SNAME("Button"), font_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("Button"), font_hover_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("Button"), font_focus_color);
	theme->set_color(SNAME("font_pressed_color"), SNAME("Button"), accent_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("Button"), font_disabled_color);
	theme->set_color(SNAME("icon_hover_color"), SNAME("Button"), icon_hover_color);
	theme->set_color(SNAME("icon_focus_color"), SNAME("Button"), icon_focus_color);
	theme->set_color(SNAME("icon_pressed_color"), SNAME("Button"), icon_pressed_color);

	// OptionButton
	theme->set_stylebox(SNAME("focus"), SNAME("OptionButton"), style_widget_focus);

	theme->set_stylebox(SNAME("normal"), SNAME("OptionButton"), style_widget);
	theme->set_stylebox(SNAME("hover"), SNAME("OptionButton"), style_widget_hover);
	theme->set_stylebox(SNAME("pressed"), SNAME("OptionButton"), style_widget_pressed);
	theme->set_stylebox(SNAME("disabled"), SNAME("OptionButton"), style_widget_disabled);

	theme->set_stylebox(SNAME("normal_mirrored"), SNAME("OptionButton"), style_widget);
	theme->set_stylebox(SNAME("hover_mirrored"), SNAME("OptionButton"), style_widget_hover);
	theme->set_stylebox(SNAME("pressed_mirrored"), SNAME("OptionButton"), style_widget_pressed);
	theme->set_stylebox(SNAME("disabled_mirrored"), SNAME("OptionButton"), style_widget_disabled);

	theme->set_color(SNAME("font_color"), SNAME("OptionButton"), font_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("OptionButton"), font_hover_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("OptionButton"), font_focus_color);
	theme->set_color(SNAME("font_pressed_color"), SNAME("OptionButton"), accent_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("OptionButton"), font_disabled_color);
	theme->set_color(SNAME("icon_hover_color"), SNAME("OptionButton"), icon_hover_color);
	theme->set_color(SNAME("icon_focus_color"), SNAME("OptionButton"), icon_focus_color);
	theme->set_icon(SNAME("arrow"), SNAME("OptionButton"), theme->get_icon(SNAME("GuiOptionArrow"), SNAME("EditorIcons")));
	theme->set_constant(SNAME("arrow_margin"), SNAME("OptionButton"), widget_default_margin.x - 2 * EDSCALE);
	theme->set_constant(SNAME("modulate_arrow"), SNAME("OptionButton"), true);
	theme->set_constant(SNAME("hseparation"), SNAME("OptionButton"), 4 * EDSCALE);

	// CheckButton
	theme->set_stylebox(SNAME("normal"), SNAME("CheckButton"), style_menu);
	theme->set_stylebox(SNAME("pressed"), SNAME("CheckButton"), style_menu);
	theme->set_stylebox(SNAME("disabled"), SNAME("CheckButton"), style_menu);
	theme->set_stylebox(SNAME("hover"), SNAME("CheckButton"), style_menu);

	theme->set_icon(SNAME("on"), SNAME("CheckButton"), theme->get_icon(SNAME("GuiToggleOn"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("on_disabled"), SNAME("CheckButton"), theme->get_icon(SNAME("GuiToggleOnDisabled"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("off"), SNAME("CheckButton"), theme->get_icon(SNAME("GuiToggleOff"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("off_disabled"), SNAME("CheckButton"), theme->get_icon(SNAME("GuiToggleOffDisabled"), SNAME("EditorIcons")));

	theme->set_icon(SNAME("on_mirrored"), SNAME("CheckButton"), theme->get_icon(SNAME("GuiToggleOnMirrored"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("on_disabled_mirrored"), SNAME("CheckButton"), theme->get_icon(SNAME("GuiToggleOnDisabledMirrored"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("off_mirrored"), SNAME("CheckButton"), theme->get_icon(SNAME("GuiToggleOffMirrored"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("off_disabled_mirrored"), SNAME("CheckButton"), theme->get_icon(SNAME("GuiToggleOffDisabledMirrored"), SNAME("EditorIcons")));

	theme->set_color(SNAME("font_color"), SNAME("CheckButton"), font_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("CheckButton"), font_hover_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("CheckButton"), font_focus_color);
	theme->set_color(SNAME("font_pressed_color"), SNAME("CheckButton"), accent_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("CheckButton"), font_disabled_color);
	theme->set_color(SNAME("icon_hover_color"), SNAME("CheckButton"), icon_hover_color);
	theme->set_color(SNAME("icon_focus_color"), SNAME("CheckButton"), icon_focus_color);

	theme->set_constant(SNAME("hseparation"), SNAME("CheckButton"), 8 * EDSCALE);
	theme->set_constant(SNAME("check_vadjust"), SNAME("CheckButton"), 0 * EDSCALE);

	// Checkbox
	Ref<StyleBoxFlat> sb_checkbox = style_menu->duplicate();
	sb_checkbox->set_default_margin(SIDE_LEFT, default_margin_size * EDSCALE);
	sb_checkbox->set_default_margin(SIDE_RIGHT, default_margin_size * EDSCALE);
	sb_checkbox->set_default_margin(SIDE_TOP, default_margin_size * EDSCALE);
	sb_checkbox->set_default_margin(SIDE_BOTTOM, default_margin_size * EDSCALE);

	theme->set_stylebox(SNAME("normal"), SNAME("CheckBox"), sb_checkbox);
	theme->set_stylebox(SNAME("pressed"), SNAME("CheckBox"), sb_checkbox);
	theme->set_stylebox(SNAME("disabled"), SNAME("CheckBox"), sb_checkbox);
	theme->set_stylebox(SNAME("hover"), SNAME("CheckBox"), sb_checkbox);
	theme->set_icon(SNAME("checked"), SNAME("CheckBox"), theme->get_icon(SNAME("GuiChecked"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("unchecked"), SNAME("CheckBox"), theme->get_icon(SNAME("GuiUnchecked"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("radio_checked"), SNAME("CheckBox"), theme->get_icon(SNAME("GuiRadioChecked"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("radio_unchecked"), SNAME("CheckBox"), theme->get_icon(SNAME("GuiRadioUnchecked"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("checked_disabled"), SNAME("CheckBox"), theme->get_icon(SNAME("GuiCheckedDisabled"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("unchecked_disabled"), SNAME("CheckBox"), theme->get_icon(SNAME("GuiUncheckedDisabled"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("radio_checked_disabled"), SNAME("CheckBox"), theme->get_icon(SNAME("GuiRadioCheckedDisabled"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("radio_unchecked_disabled"), SNAME("CheckBox"), theme->get_icon(SNAME("GuiRadioUncheckedDisabled"), SNAME("EditorIcons")));

	theme->set_color(SNAME("font_color"), SNAME("CheckBox"), font_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("CheckBox"), font_hover_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("CheckBox"), font_focus_color);
	theme->set_color(SNAME("font_pressed_color"), SNAME("CheckBox"), accent_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("CheckBox"), font_disabled_color);
	theme->set_color(SNAME("icon_hover_color"), SNAME("CheckBox"), icon_hover_color);
	theme->set_color(SNAME("icon_focus_color"), SNAME("CheckBox"), icon_focus_color);

	theme->set_constant(SNAME("hseparation"), SNAME("CheckBox"), 8 * EDSCALE);
	theme->set_constant(SNAME("check_vadjust"), SNAME("CheckBox"), 0 * EDSCALE);

	// PopupDialog
	theme->set_stylebox(SNAME("panel"), SNAME("PopupDialog"), style_popup);

	// PopupMenu
	const int popup_menu_margin_size = default_margin_size * 1.5 * EDSCALE;
	Ref<StyleBoxFlat> style_popup_menu = style_popup->duplicate();
	// Use 1 pixel for the sides, since if 0 is used, the highlight of hovered items is drawn
	// on top of the popup border. This causes a 'gap' in the panel border when an item is highlighted,
	// and it looks weird. 1px solves this.
	style_popup_menu->set_default_margin(SIDE_LEFT, 1 * EDSCALE);
	style_popup_menu->set_default_margin(SIDE_TOP, popup_menu_margin_size);
	style_popup_menu->set_default_margin(SIDE_RIGHT, 1 * EDSCALE);
	style_popup_menu->set_default_margin(SIDE_BOTTOM, popup_menu_margin_size);
	// Always display a border for PopupMenus so they can be distinguished from their background.
	style_popup_menu->set_border_width_all(1 * EDSCALE);
	style_popup_menu->set_border_color(dark_color_2);
	theme->set_stylebox(SNAME("panel"), SNAME("PopupMenu"), style_popup_menu);

	Ref<StyleBoxFlat> style_menu_hover = style_widget_hover->duplicate();
	// Don't use rounded corners for hover highlights since the StyleBox touches the PopupMenu's edges.
	style_menu_hover->set_corner_radius_all(0);
	theme->set_stylebox(SNAME("hover"), SNAME("PopupMenu"), style_menu_hover);

	theme->set_stylebox(SNAME("separator"), SNAME("PopupMenu"), style_popup_separator);
	theme->set_stylebox(SNAME("labeled_separator_left"), SNAME("PopupMenu"), style_popup_labeled_separator_left);
	theme->set_stylebox(SNAME("labeled_separator_right"), SNAME("PopupMenu"), style_popup_labeled_separator_right);

	theme->set_color(SNAME("font_color"), SNAME("PopupMenu"), font_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("PopupMenu"), font_hover_color);
	theme->set_color(SNAME("font_accelerator_color"), SNAME("PopupMenu"), font_disabled_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("PopupMenu"), font_disabled_color);
	theme->set_color(SNAME("font_separator_color"), SNAME("PopupMenu"), font_disabled_color);
	theme->set_icon(SNAME("checked"), SNAME("PopupMenu"), theme->get_icon(SNAME("GuiChecked"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("unchecked"), SNAME("PopupMenu"), theme->get_icon(SNAME("GuiUnchecked"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("radio_checked"), SNAME("PopupMenu"), theme->get_icon(SNAME("GuiRadioChecked"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("radio_unchecked"), SNAME("PopupMenu"), theme->get_icon(SNAME("GuiRadioUnchecked"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("checked_disabled"), SNAME("PopupMenu"), theme->get_icon(SNAME("GuiCheckedDisabled"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("unchecked_disabled"), SNAME("PopupMenu"), theme->get_icon(SNAME("GuiUncheckedDisabled"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("radio_checked_disabled"), SNAME("PopupMenu"), theme->get_icon(SNAME("GuiRadioCheckedDisabled"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("radio_unchecked_disabled"), SNAME("PopupMenu"), theme->get_icon(SNAME("GuiRadioUncheckedDisabled"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("submenu"), SNAME("PopupMenu"), theme->get_icon(SNAME("ArrowRight"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("submenu_mirrored"), SNAME("PopupMenu"), theme->get_icon(SNAME("ArrowLeft"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("visibility_hidden"), SNAME("PopupMenu"), theme->get_icon(SNAME("GuiVisibilityHidden"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("visibility_visible"), SNAME("PopupMenu"), theme->get_icon(SNAME("GuiVisibilityVisible"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("visibility_xray"), SNAME("PopupMenu"), theme->get_icon(SNAME("GuiVisibilityXray"), SNAME("EditorIcons")));

	theme->set_constant(SNAME("vseparation"), SNAME("PopupMenu"), (extra_spacing + default_margin_size + 1) * EDSCALE);
	theme->set_constant(SNAME("item_start_padding"), SNAME("PopupMenu"), popup_menu_margin_size * EDSCALE);
	theme->set_constant(SNAME("item_end_padding"), SNAME("PopupMenu"), popup_menu_margin_size * EDSCALE);

	for (int i = 0; i < 16; i++) {
		Color si_base_color = accent_color;

		float hue_rotate = (i * 2 % 16) / 16.0;
		si_base_color.set_hsv(Math::fmod(float(si_base_color.get_h() + hue_rotate), float(1.0)), si_base_color.get_s(), si_base_color.get_v());
		si_base_color = accent_color.lerp(si_base_color, float(EDITOR_GET("docks/property_editor/subresource_hue_tint")));

		Ref<StyleBoxFlat> sub_inspector_bg;

		sub_inspector_bg = make_flat_stylebox(dark_color_1.lerp(si_base_color, 0.08), 2, 0, 2, 2);

		sub_inspector_bg->set_border_width(SIDE_LEFT, 2);
		sub_inspector_bg->set_border_width(SIDE_RIGHT, 2);
		sub_inspector_bg->set_border_width(SIDE_BOTTOM, 2);
		sub_inspector_bg->set_border_width(SIDE_TOP, 2);
		sub_inspector_bg->set_default_margin(SIDE_LEFT, 3);
		sub_inspector_bg->set_default_margin(SIDE_RIGHT, 3);
		sub_inspector_bg->set_default_margin(SIDE_BOTTOM, 10);
		sub_inspector_bg->set_default_margin(SIDE_TOP, 5);
		sub_inspector_bg->set_border_color(si_base_color * Color(0.7, 0.7, 0.7, 0.8));
		sub_inspector_bg->set_draw_center(true);

		theme->set_stylebox("sub_inspector_bg" + itos(i), SNAME("Editor"), sub_inspector_bg);

		Ref<StyleBoxFlat> bg_color;
		bg_color.instantiate();
		bg_color->set_bg_color(si_base_color * Color(0.7, 0.7, 0.7, 0.8));
		bg_color->set_border_width_all(0);

		Ref<StyleBoxFlat> bg_color_selected;
		bg_color_selected.instantiate();
		bg_color_selected->set_border_width_all(0);
		bg_color_selected->set_bg_color(si_base_color * Color(0.8, 0.8, 0.8, 0.8));

		theme->set_stylebox("sub_inspector_property_bg" + itos(i), SNAME("Editor"), bg_color);
		theme->set_stylebox("sub_inspector_property_bg_selected" + itos(i), SNAME("Editor"), bg_color_selected);
	}

	theme->set_color(SNAME("sub_inspector_property_color"), SNAME("Editor"), dark_theme ? Color(1, 1, 1, 1) : Color(0, 0, 0, 1));
	theme->set_constant(SNAME("sub_inspector_font_offset"), SNAME("Editor"), 4 * EDSCALE);

	Ref<StyleBoxFlat> style_property_bg = style_default->duplicate();
	style_property_bg->set_bg_color(highlight_color);
	style_property_bg->set_border_width_all(0);

	theme->set_constant(SNAME("font_offset"), SNAME("EditorProperty"), 8 * EDSCALE);
	theme->set_stylebox(SNAME("bg_selected"), SNAME("EditorProperty"), style_property_bg);
	theme->set_stylebox(SNAME("bg"), SNAME("EditorProperty"), Ref<StyleBoxEmpty>(memnew(StyleBoxEmpty)));
	theme->set_constant(SNAME("vseparation"), SNAME("EditorProperty"), (extra_spacing + default_margin_size) * EDSCALE);
	theme->set_color(SNAME("warning_color"), SNAME("EditorProperty"), warning_color);
	theme->set_color(SNAME("property_color"), SNAME("EditorProperty"), property_color);
	theme->set_color(SNAME("readonly_color"), SNAME("EditorProperty"), readonly_color);
	theme->set_color(SNAME("readonly_warning_color"), SNAME("EditorProperty"), readonly_warning_color);

	Color inspector_section_color = font_color.lerp(Color(0.5, 0.5, 0.5), 0.35);
	theme->set_color(SNAME("font_color"), SNAME("EditorInspectorSection"), inspector_section_color);

	theme->set_constant(SNAME("inspector_margin"), SNAME("Editor"), 12 * EDSCALE);

	// Tree & ItemList background
	Ref<StyleBoxFlat> style_tree_bg = style_default->duplicate();
	// Make Trees easier to distinguish from other controls by using a darker background color.
	style_tree_bg->set_bg_color(dark_color_1.lerp(dark_color_2, 0.5));
	style_tree_bg->set_border_color(dark_color_3);
	theme->set_stylebox(SNAME("bg"), SNAME("Tree"), style_tree_bg);

	// Tree
	theme->set_icon(SNAME("checked"), SNAME("Tree"), theme->get_icon(SNAME("GuiChecked"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("indeterminate"), SNAME("Tree"), theme->get_icon(SNAME("GuiIndeterminate"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("unchecked"), SNAME("Tree"), theme->get_icon(SNAME("GuiUnchecked"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("arrow"), SNAME("Tree"), theme->get_icon(SNAME("GuiTreeArrowDown"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("arrow_collapsed"), SNAME("Tree"), theme->get_icon(SNAME("GuiTreeArrowRight"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("arrow_collapsed_mirrored"), SNAME("Tree"), theme->get_icon(SNAME("GuiTreeArrowLeft"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("updown"), SNAME("Tree"), theme->get_icon(SNAME("GuiTreeUpdown"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("select_arrow"), SNAME("Tree"), theme->get_icon(SNAME("GuiDropdown"), SNAME("EditorIcons")));
	theme->set_stylebox(SNAME("bg_focus"), SNAME("Tree"), style_widget_focus);
	theme->set_stylebox(SNAME("custom_button"), SNAME("Tree"), make_empty_stylebox());
	theme->set_stylebox(SNAME("custom_button_pressed"), SNAME("Tree"), make_empty_stylebox());
	theme->set_stylebox(SNAME("custom_button_hover"), SNAME("Tree"), style_widget);
	theme->set_color(SNAME("custom_button_font_highlight"), SNAME("Tree"), font_hover_color);
	theme->set_color(SNAME("font_color"), SNAME("Tree"), font_color);
	theme->set_color(SNAME("font_selected_color"), SNAME("Tree"), mono_color);
	theme->set_color(SNAME("title_button_color"), SNAME("Tree"), font_color);
	theme->set_color(SNAME("drop_position_color"), SNAME("Tree"), accent_color);
	theme->set_constant(SNAME("vseparation"), SNAME("Tree"), widget_default_margin.y - EDSCALE);
	theme->set_constant(SNAME("hseparation"), SNAME("Tree"), 6 * EDSCALE);
	theme->set_constant(SNAME("guide_width"), SNAME("Tree"), border_width);
	theme->set_constant(SNAME("item_margin"), SNAME("Tree"), 3 * default_margin_size * EDSCALE);
	theme->set_constant(SNAME("button_margin"), SNAME("Tree"), default_margin_size * EDSCALE);
	theme->set_constant(SNAME("scroll_border"), SNAME("Tree"), 40 * EDSCALE);
	theme->set_constant(SNAME("scroll_speed"), SNAME("Tree"), 12);

	const Color guide_color = mono_color * Color(1, 1, 1, 0.05);
	Color relationship_line_color = mono_color * Color(1, 1, 1, relationship_line_opacity);

	theme->set_constant(SNAME("draw_guides"), SNAME("Tree"), relationship_line_opacity < 0.01);
	theme->set_color(SNAME("guide_color"), SNAME("Tree"), guide_color);

	int relationship_line_width = 1;
	Color parent_line_color = mono_color * Color(1, 1, 1, CLAMP(relationship_line_opacity + 0.45, 0.0, 1.0));
	Color children_line_color = mono_color * Color(1, 1, 1, CLAMP(relationship_line_opacity + 0.25, 0.0, 1.0));
	theme->set_constant(SNAME("draw_relationship_lines"), SNAME("Tree"), relationship_line_opacity >= 0.01);
	theme->set_constant(SNAME("relationship_line_width"), SNAME("Tree"), relationship_line_width);
	theme->set_constant(SNAME("parent_hl_line_width"), SNAME("Tree"), relationship_line_width * 2);
	theme->set_constant(SNAME("children_hl_line_width"), SNAME("Tree"), relationship_line_width);
	theme->set_constant(SNAME("parent_hl_line_margin"), SNAME("Tree"), relationship_line_width * 3);
	theme->set_color(SNAME("relationship_line_color"), SNAME("Tree"), relationship_line_color);
	theme->set_color(SNAME("parent_hl_line_color"), SNAME("Tree"), parent_line_color);
	theme->set_color(SNAME("children_hl_line_color"), SNAME("Tree"), children_line_color);

	Ref<StyleBoxFlat> style_tree_btn = style_default->duplicate();
	style_tree_btn->set_bg_color(highlight_color);
	style_tree_btn->set_border_width_all(0);
	theme->set_stylebox(SNAME("button_pressed"), SNAME("Tree"), style_tree_btn);

	Ref<StyleBoxFlat> style_tree_hover = style_default->duplicate();
	style_tree_hover->set_bg_color(highlight_color * Color(1, 1, 1, 0.4));
	style_tree_hover->set_border_width_all(0);
	theme->set_stylebox(SNAME("hover"), SNAME("Tree"), style_tree_hover);

	Ref<StyleBoxFlat> style_tree_focus = style_default->duplicate();
	style_tree_focus->set_bg_color(highlight_color);
	style_tree_focus->set_border_width_all(0);
	theme->set_stylebox(SNAME("selected_focus"), SNAME("Tree"), style_tree_focus);

	Ref<StyleBoxFlat> style_tree_selected = style_tree_focus->duplicate();
	theme->set_stylebox(SNAME("selected"), SNAME("Tree"), style_tree_selected);

	Ref<StyleBoxFlat> style_tree_cursor = style_default->duplicate();
	style_tree_cursor->set_draw_center(false);
	style_tree_cursor->set_border_width_all(MAX(1, border_width));
	style_tree_cursor->set_border_color(contrast_color_1);

	Ref<StyleBoxFlat> style_tree_title = style_default->duplicate();
	style_tree_title->set_bg_color(dark_color_3);
	style_tree_title->set_border_width_all(0);
	theme->set_stylebox(SNAME("cursor"), SNAME("Tree"), style_tree_cursor);
	theme->set_stylebox(SNAME("cursor_unfocused"), SNAME("Tree"), style_tree_cursor);
	theme->set_stylebox(SNAME("title_button_normal"), SNAME("Tree"), style_tree_title);
	theme->set_stylebox(SNAME("title_button_hover"), SNAME("Tree"), style_tree_title);
	theme->set_stylebox(SNAME("title_button_pressed"), SNAME("Tree"), style_tree_title);

	Color prop_category_color = dark_color_1.lerp(mono_color, 0.12);
	Color prop_section_color = dark_color_1.lerp(mono_color, 0.09);
	Color prop_subsection_color = dark_color_1.lerp(mono_color, 0.06);
	theme->set_color(SNAME("prop_category"), SNAME("Editor"), prop_category_color);
	theme->set_color(SNAME("prop_section"), SNAME("Editor"), prop_section_color);
	theme->set_color(SNAME("prop_subsection"), SNAME("Editor"), prop_subsection_color);
	theme->set_color(SNAME("drop_position_color"), SNAME("Tree"), accent_color);

	Ref<StyleBoxFlat> category_bg = style_default->duplicate();
	// Make Trees easier to distinguish from other controls by using a darker background color.
	category_bg->set_bg_color(prop_category_color);
	category_bg->set_border_color(prop_category_color);
	theme->set_stylebox(SNAME("prop_category_style"), SNAME("Editor"), category_bg);

	// ItemList
	Ref<StyleBoxFlat> style_itemlist_bg = style_default->duplicate();
	style_itemlist_bg->set_bg_color(dark_color_1);
	style_itemlist_bg->set_border_width_all(border_width);
	style_itemlist_bg->set_border_color(dark_color_3);

	Ref<StyleBoxFlat> style_itemlist_cursor = style_default->duplicate();
	style_itemlist_cursor->set_draw_center(false);
	style_itemlist_cursor->set_border_width_all(border_width);
	style_itemlist_cursor->set_border_color(highlight_color);
	theme->set_stylebox(SNAME("cursor"), SNAME("ItemList"), style_itemlist_cursor);
	theme->set_stylebox(SNAME("cursor_unfocused"), SNAME("ItemList"), style_itemlist_cursor);
	theme->set_stylebox(SNAME("selected_focus"), SNAME("ItemList"), style_tree_focus);
	theme->set_stylebox(SNAME("selected"), SNAME("ItemList"), style_tree_selected);
	theme->set_stylebox(SNAME("bg_focus"), SNAME("ItemList"), style_widget_focus);
	theme->set_stylebox(SNAME("bg"), SNAME("ItemList"), style_itemlist_bg);
	theme->set_color(SNAME("font_color"), SNAME("ItemList"), font_color);
	theme->set_color(SNAME("font_selected_color"), SNAME("ItemList"), mono_color);
	theme->set_color(SNAME("guide_color"), SNAME("ItemList"), guide_color);
	theme->set_constant(SNAME("vseparation"), SNAME("ItemList"), widget_default_margin.y - EDSCALE);
	theme->set_constant(SNAME("hseparation"), SNAME("ItemList"), 6 * EDSCALE);
	theme->set_constant(SNAME("icon_margin"), SNAME("ItemList"), 6 * EDSCALE);
	theme->set_constant(SNAME("line_separation"), SNAME("ItemList"), 3 * EDSCALE);

	// TabBar & TabContainer
	theme->set_stylebox(SNAME("tab_selected"), SNAME("TabContainer"), style_tab_selected);
	theme->set_stylebox(SNAME("tab_unselected"), SNAME("TabContainer"), style_tab_unselected);
	theme->set_stylebox(SNAME("tab_disabled"), SNAME("TabContainer"), style_tab_disabled);
	theme->set_stylebox(SNAME("tab_selected"), SNAME("TabBar"), style_tab_selected);
	theme->set_stylebox(SNAME("tab_unselected"), SNAME("TabBar"), style_tab_unselected);
	theme->set_stylebox(SNAME("tab_disabled"), SNAME("TabBar"), style_tab_disabled);
	theme->set_color(SNAME("font_selected_color"), SNAME("TabContainer"), font_color);
	theme->set_color(SNAME("font_unselected_color"), SNAME("TabContainer"), font_disabled_color);
	theme->set_color(SNAME("font_selected_color"), SNAME("TabBar"), font_color);
	theme->set_color(SNAME("font_unselected_color"), SNAME("TabBar"), font_disabled_color);
	theme->set_icon(SNAME("menu"), SNAME("TabContainer"), theme->get_icon(SNAME("GuiTabMenu"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("menu_highlight"), SNAME("TabContainer"), theme->get_icon(SNAME("GuiTabMenuHl"), SNAME("EditorIcons")));
	theme->set_stylebox(SNAME("SceneTabFG"), SNAME("EditorStyles"), style_tab_selected);
	theme->set_stylebox(SNAME("SceneTabBG"), SNAME("EditorStyles"), style_tab_unselected);
	theme->set_icon(SNAME("close"), SNAME("TabBar"), theme->get_icon(SNAME("GuiClose"), SNAME("EditorIcons")));
	theme->set_stylebox(SNAME("button_pressed"), SNAME("TabBar"), style_menu);
	theme->set_stylebox(SNAME("button_highlight"), SNAME("TabBar"), style_menu);
	theme->set_icon(SNAME("increment"), SNAME("TabContainer"), theme->get_icon(SNAME("GuiScrollArrowRight"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("decrement"), SNAME("TabContainer"), theme->get_icon(SNAME("GuiScrollArrowLeft"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("increment"), SNAME("TabBar"), theme->get_icon(SNAME("GuiScrollArrowRight"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("decrement"), SNAME("TabBar"), theme->get_icon(SNAME("GuiScrollArrowLeft"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("increment_highlight"), SNAME("TabBar"), theme->get_icon(SNAME("GuiScrollArrowRightHl"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("decrement_highlight"), SNAME("TabBar"), theme->get_icon(SNAME("GuiScrollArrowLeftHl"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("increment_highlight"), SNAME("TabContainer"), theme->get_icon(SNAME("GuiScrollArrowRightHl"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("decrement_highlight"), SNAME("TabContainer"), theme->get_icon(SNAME("GuiScrollArrowLeftHl"), SNAME("EditorIcons")));
	theme->set_constant(SNAME("hseparation"), SNAME("TabBar"), 4 * EDSCALE);

	// Content of each tab
	Ref<StyleBoxFlat> style_content_panel = style_default->duplicate();
	style_content_panel->set_border_color(dark_color_3);
	style_content_panel->set_border_width_all(border_width);
	// compensate the border
	style_content_panel->set_default_margin(SIDE_TOP, (2 + margin_size_extra) * EDSCALE);
	style_content_panel->set_default_margin(SIDE_RIGHT, margin_size_extra * EDSCALE);
	style_content_panel->set_default_margin(SIDE_BOTTOM, margin_size_extra * EDSCALE);
	style_content_panel->set_default_margin(SIDE_LEFT, margin_size_extra * EDSCALE);
	// Display border to visually split the body of the container from its possible backgrounds.
	style_content_panel->set_border_width(Side::SIDE_TOP, Math::round(2 * EDSCALE));
	style_content_panel->set_border_color(dark_color_2);
	theme->set_stylebox(SNAME("panel"), SNAME("TabContainer"), style_content_panel);

	// These styleboxes can be used on tabs against the base color background (e.g. nested tabs).
	Ref<StyleBoxFlat> style_tab_selected_odd = style_tab_selected->duplicate();
	style_tab_selected_odd->set_bg_color(disabled_bg_color);
	theme->set_stylebox(SNAME("tab_selected_odd"), SNAME("TabContainer"), style_tab_selected_odd);

	Ref<StyleBoxFlat> style_content_panel_odd = style_content_panel->duplicate();
	style_content_panel_odd->set_bg_color(disabled_bg_color);
	theme->set_stylebox(SNAME("panel_odd"), SNAME("TabContainer"), style_content_panel_odd);

	// This stylebox is used in 3d and 2d viewports (no borders).
	Ref<StyleBoxFlat> style_content_panel_vp = style_content_panel->duplicate();
	style_content_panel_vp->set_default_margin(SIDE_LEFT, border_width * 2);
	style_content_panel_vp->set_default_margin(SIDE_TOP, default_margin_size * EDSCALE);
	style_content_panel_vp->set_default_margin(SIDE_RIGHT, border_width * 2);
	style_content_panel_vp->set_default_margin(SIDE_BOTTOM, border_width * 2);
	theme->set_stylebox(SNAME("Content"), SNAME("EditorStyles"), style_content_panel_vp);

	// This stylebox is used by preview tabs in the Theme Editor.
	Ref<StyleBoxFlat> style_theme_preview_tab = style_tab_selected_odd->duplicate();
	style_theme_preview_tab->set_expand_margin_size(SIDE_BOTTOM, 5 * EDSCALE);
	theme->set_stylebox(SNAME("ThemeEditorPreviewFG"), SNAME("EditorStyles"), style_theme_preview_tab);
	Ref<StyleBoxFlat> style_theme_preview_bg_tab = style_tab_unselected->duplicate();
	style_theme_preview_bg_tab->set_expand_margin_size(SIDE_BOTTOM, 2 * EDSCALE);
	theme->set_stylebox(SNAME("ThemeEditorPreviewBG"), SNAME("EditorStyles"), style_theme_preview_bg_tab);

	// Separators
	theme->set_stylebox(SNAME("separator"), SNAME("HSeparator"), make_line_stylebox(separator_color, MAX(Math::round(EDSCALE), border_width)));
	theme->set_stylebox(SNAME("separator"), SNAME("VSeparator"), make_line_stylebox(separator_color, MAX(Math::round(EDSCALE), border_width), 0, 0, true));

	// Debugger

	Ref<StyleBoxFlat> style_panel_debugger = style_content_panel->duplicate();
	style_panel_debugger->set_border_width(SIDE_BOTTOM, 0);
	theme->set_stylebox(SNAME("DebuggerPanel"), SNAME("EditorStyles"), style_panel_debugger);

	Ref<StyleBoxFlat> style_panel_invisible_top = style_content_panel->duplicate();
	int stylebox_offset = theme->get_font(SNAME("tab_selected"), SNAME("TabContainer"))->get_height(theme->get_font_size(SNAME("tab_selected"), SNAME("TabContainer"))) + theme->get_stylebox(SNAME("tab_selected"), SNAME("TabContainer"))->get_minimum_size().height + theme->get_stylebox(SNAME("panel"), SNAME("TabContainer"))->get_default_margin(SIDE_TOP);
	style_panel_invisible_top->set_expand_margin_size(SIDE_TOP, -stylebox_offset);
	style_panel_invisible_top->set_default_margin(SIDE_TOP, 0);
	theme->set_stylebox(SNAME("BottomPanelDebuggerOverride"), SNAME("EditorStyles"), style_panel_invisible_top);

	// LineEdit

	Ref<StyleBoxFlat> style_line_edit = style_widget->duplicate();
	// The original style_widget style has an extra 1 pixel offset that makes LineEdits not align with Buttons,
	// so this compensates for that.
	style_line_edit->set_default_margin(SIDE_TOP, style_line_edit->get_default_margin(SIDE_TOP) - 1 * EDSCALE);
	// Add a bottom line to make LineEdits more visible, especially in sectioned inspectors
	// such as the Project Settings.
	style_line_edit->set_border_width(SIDE_BOTTOM, Math::round(2 * EDSCALE));
	style_line_edit->set_border_color(dark_color_2);
	// Don't round the bottom corner to make the line look sharper.
	style_tab_selected->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
	style_tab_selected->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);

	Ref<StyleBoxFlat> style_line_edit_disabled = style_line_edit->duplicate();
	style_line_edit_disabled->set_border_color(disabled_color);
	style_line_edit_disabled->set_bg_color(disabled_bg_color);

	theme->set_stylebox(SNAME("normal"), SNAME("LineEdit"), style_line_edit);
	theme->set_stylebox(SNAME("focus"), SNAME("LineEdit"), style_widget_focus);
	theme->set_stylebox(SNAME("read_only"), SNAME("LineEdit"), style_line_edit_disabled);
	theme->set_icon(SNAME("clear"), SNAME("LineEdit"), theme->get_icon(SNAME("GuiClose"), SNAME("EditorIcons")));
	theme->set_color(SNAME("read_only"), SNAME("LineEdit"), font_disabled_color);
	theme->set_color(SNAME("font_color"), SNAME("LineEdit"), font_color);
	theme->set_color(SNAME("font_selected_color"), SNAME("LineEdit"), mono_color);
	theme->set_color(SNAME("font_uneditable_color"), SNAME("LineEdit"), font_readonly_color);
	theme->set_color(SNAME("font_placeholder_color"), SNAME("LineEdit"), font_placeholder_color);
	theme->set_color(SNAME("caret_color"), SNAME("LineEdit"), font_color);
	theme->set_color(SNAME("selection_color"), SNAME("LineEdit"), selection_color);
	theme->set_color(SNAME("clear_button_color"), SNAME("LineEdit"), font_color);
	theme->set_color(SNAME("clear_button_color_pressed"), SNAME("LineEdit"), accent_color);

	// TextEdit
	theme->set_stylebox(SNAME("normal"), SNAME("TextEdit"), style_line_edit);
	theme->set_stylebox(SNAME("focus"), SNAME("TextEdit"), style_widget_focus);
	theme->set_stylebox(SNAME("read_only"), SNAME("TextEdit"), style_line_edit_disabled);
	theme->set_constant(SNAME("side_margin"), SNAME("TabContainer"), 0);
	theme->set_icon(SNAME("tab"), SNAME("TextEdit"), theme->get_icon(SNAME("GuiTab"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("space"), SNAME("TextEdit"), theme->get_icon(SNAME("GuiSpace"), SNAME("EditorIcons")));
	theme->set_color(SNAME("font_color"), SNAME("TextEdit"), font_color);
	theme->set_color(SNAME("font_readonly_color"), SNAME("TextEdit"), font_readonly_color);
	theme->set_color(SNAME("font_placeholder_color"), SNAME("TextEdit"), font_placeholder_color);
	theme->set_color(SNAME("caret_color"), SNAME("TextEdit"), font_color);
	theme->set_color(SNAME("selection_color"), SNAME("TextEdit"), selection_color);
	theme->set_constant(SNAME("line_spacing"), SNAME("TextEdit"), 4 * EDSCALE);

	// CodeEdit
	theme->set_font(SNAME("font"), SNAME("CodeEdit"), theme->get_font(SNAME("source"), SNAME("EditorFonts")));
	theme->set_font_size(SNAME("font_size"), SNAME("CodeEdit"), theme->get_font_size(SNAME("source_size"), SNAME("EditorFonts")));
	theme->set_stylebox(SNAME("normal"), SNAME("CodeEdit"), style_widget);
	theme->set_stylebox(SNAME("focus"), SNAME("CodeEdit"), style_widget_hover);
	theme->set_stylebox(SNAME("read_only"), SNAME("CodeEdit"), style_widget_disabled);
	theme->set_icon(SNAME("tab"), SNAME("CodeEdit"), theme->get_icon(SNAME("GuiTab"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("space"), SNAME("CodeEdit"), theme->get_icon(SNAME("GuiSpace"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("folded"), SNAME("CodeEdit"), theme->get_icon(SNAME("GuiTreeArrowRight"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("can_fold"), SNAME("CodeEdit"), theme->get_icon(SNAME("GuiTreeArrowDown"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("executing_line"), SNAME("CodeEdit"), theme->get_icon(SNAME("MainPlay"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("breakpoint"), SNAME("CodeEdit"), theme->get_icon(SNAME("Breakpoint"), SNAME("EditorIcons")));
	theme->set_constant(SNAME("line_spacing"), SNAME("CodeEdit"), EDITOR_DEF("text_editor/appearance/whitespace/line_spacing", 6));

	// H/VSplitContainer
	theme->set_stylebox(SNAME("bg"), SNAME("VSplitContainer"), make_stylebox(theme->get_icon(SNAME("GuiVsplitBg"), SNAME("EditorIcons")), 1, 1, 1, 1));
	theme->set_stylebox(SNAME("bg"), SNAME("HSplitContainer"), make_stylebox(theme->get_icon(SNAME("GuiHsplitBg"), SNAME("EditorIcons")), 1, 1, 1, 1));

	theme->set_icon(SNAME("grabber"), SNAME("VSplitContainer"), theme->get_icon(SNAME("GuiVsplitter"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("grabber"), SNAME("HSplitContainer"), theme->get_icon(SNAME("GuiHsplitter"), SNAME("EditorIcons")));

	theme->set_constant(SNAME("separation"), SNAME("HSplitContainer"), default_margin_size * 2 * EDSCALE);
	theme->set_constant(SNAME("separation"), SNAME("VSplitContainer"), default_margin_size * 2 * EDSCALE);

	// Containers
	theme->set_constant(SNAME("separation"), SNAME("BoxContainer"), default_margin_size * EDSCALE);
	theme->set_constant(SNAME("separation"), SNAME("HBoxContainer"), default_margin_size * EDSCALE);
	theme->set_constant(SNAME("separation"), SNAME("VBoxContainer"), default_margin_size * EDSCALE);
	theme->set_constant(SNAME("margin_left"), SNAME("MarginContainer"), 0);
	theme->set_constant(SNAME("margin_top"), SNAME("MarginContainer"), 0);
	theme->set_constant(SNAME("margin_right"), SNAME("MarginContainer"), 0);
	theme->set_constant(SNAME("margin_bottom"), SNAME("MarginContainer"), 0);
	theme->set_constant(SNAME("hseparation"), SNAME("GridContainer"), default_margin_size * EDSCALE);
	theme->set_constant(SNAME("vseparation"), SNAME("GridContainer"), default_margin_size * EDSCALE);
	theme->set_constant(SNAME("hseparation"), SNAME("FlowContainer"), default_margin_size * EDSCALE);
	theme->set_constant(SNAME("vseparation"), SNAME("FlowContainer"), default_margin_size * EDSCALE);
	theme->set_constant(SNAME("hseparation"), SNAME("HFlowContainer"), default_margin_size * EDSCALE);
	theme->set_constant(SNAME("vseparation"), SNAME("HFlowContainer"), default_margin_size * EDSCALE);
	theme->set_constant(SNAME("hseparation"), SNAME("VFlowContainer"), default_margin_size * EDSCALE);
	theme->set_constant(SNAME("vseparation"), SNAME("VFlowContainer"), default_margin_size * EDSCALE);

	// Window

	// Prevent corner artifacts between window title and body.
	Ref<StyleBoxFlat> style_window_title = style_default->duplicate();
	style_window_title->set_corner_radius(CORNER_TOP_LEFT, 0);
	style_window_title->set_corner_radius(CORNER_TOP_RIGHT, 0);
	// Prevent visible line between window title and body.
	style_window_title->set_expand_margin_size(SIDE_BOTTOM, 2 * EDSCALE);

	Ref<StyleBoxFlat> style_window = style_popup->duplicate();
	style_window->set_border_color(base_color);
	style_window->set_border_width(SIDE_TOP, 24 * EDSCALE);
	style_window->set_expand_margin_size(SIDE_TOP, 24 * EDSCALE);
	theme->set_stylebox(SNAME("embedded_border"), SNAME("Window"), style_window);

	theme->set_color(SNAME("title_color"), SNAME("Window"), font_color);
	theme->set_icon(SNAME("close"), SNAME("Window"), theme->get_icon(SNAME("GuiClose"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("close_pressed"), SNAME("Window"), theme->get_icon(SNAME("GuiClose"), SNAME("EditorIcons")));
	theme->set_constant(SNAME("close_h_ofs"), SNAME("Window"), 22 * EDSCALE);
	theme->set_constant(SNAME("close_v_ofs"), SNAME("Window"), 20 * EDSCALE);
	theme->set_constant(SNAME("title_height"), SNAME("Window"), 24 * EDSCALE);
	theme->set_constant(SNAME("resize_margin"), SNAME("Window"), 4 * EDSCALE);
	theme->set_font(SNAME("title_font"), SNAME("Window"), theme->get_font(SNAME("title"), SNAME("EditorFonts")));
	theme->set_font_size(SNAME("title_font_size"), SNAME("Window"), theme->get_font_size(SNAME("title_size"), SNAME("EditorFonts")));

	// Complex window (currently only Editor Settings and Project Settings)
	Ref<StyleBoxFlat> style_complex_window = style_window->duplicate();
	style_complex_window->set_bg_color(dark_color_2);
	style_complex_window->set_border_color(dark_color_2);
	theme->set_stylebox(SNAME("panel"), SNAME("EditorSettingsDialog"), style_complex_window);
	theme->set_stylebox(SNAME("panel"), SNAME("ProjectSettingsEditor"), style_complex_window);
	theme->set_stylebox(SNAME("panel"), SNAME("EditorAbout"), style_complex_window);

	// AcceptDialog
	theme->set_stylebox(SNAME("panel"), SNAME("AcceptDialog"), style_window_title);

	// HScrollBar
	Ref<Texture2D> empty_icon = memnew(ImageTexture);

	theme->set_stylebox(SNAME("scroll"), SNAME("HScrollBar"), make_stylebox(theme->get_icon(SNAME("GuiScrollBg"), SNAME("EditorIcons")), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox(SNAME("scroll_focus"), SNAME("HScrollBar"), make_stylebox(theme->get_icon(SNAME("GuiScrollBg"), SNAME("EditorIcons")), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox(SNAME("grabber"), SNAME("HScrollBar"), make_stylebox(theme->get_icon(SNAME("GuiScrollGrabber"), SNAME("EditorIcons")), 6, 6, 6, 6, 2, 2, 2, 2));
	theme->set_stylebox(SNAME("grabber_highlight"), SNAME("HScrollBar"), make_stylebox(theme->get_icon(SNAME("GuiScrollGrabberHl"), SNAME("EditorIcons")), 5, 5, 5, 5, 2, 2, 2, 2));
	theme->set_stylebox(SNAME("grabber_pressed"), SNAME("HScrollBar"), make_stylebox(theme->get_icon(SNAME("GuiScrollGrabberPressed"), SNAME("EditorIcons")), 6, 6, 6, 6, 2, 2, 2, 2));

	theme->set_icon(SNAME("increment"), SNAME("HScrollBar"), empty_icon);
	theme->set_icon(SNAME("increment_highlight"), SNAME("HScrollBar"), empty_icon);
	theme->set_icon(SNAME("increment_pressed"), SNAME("HScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement"), SNAME("HScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement_highlight"), SNAME("HScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement_pressed"), SNAME("HScrollBar"), empty_icon);

	// VScrollBar
	theme->set_stylebox(SNAME("scroll"), SNAME("VScrollBar"), make_stylebox(theme->get_icon(SNAME("GuiScrollBg"), SNAME("EditorIcons")), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox(SNAME("scroll_focus"), SNAME("VScrollBar"), make_stylebox(theme->get_icon(SNAME("GuiScrollBg"), SNAME("EditorIcons")), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox(SNAME("grabber"), SNAME("VScrollBar"), make_stylebox(theme->get_icon(SNAME("GuiScrollGrabber"), SNAME("EditorIcons")), 6, 6, 6, 6, 2, 2, 2, 2));
	theme->set_stylebox(SNAME("grabber_highlight"), SNAME("VScrollBar"), make_stylebox(theme->get_icon(SNAME("GuiScrollGrabberHl"), SNAME("EditorIcons")), 5, 5, 5, 5, 2, 2, 2, 2));
	theme->set_stylebox(SNAME("grabber_pressed"), SNAME("VScrollBar"), make_stylebox(theme->get_icon(SNAME("GuiScrollGrabberPressed"), SNAME("EditorIcons")), 6, 6, 6, 6, 2, 2, 2, 2));

	theme->set_icon(SNAME("increment"), SNAME("VScrollBar"), empty_icon);
	theme->set_icon(SNAME("increment_highlight"), SNAME("VScrollBar"), empty_icon);
	theme->set_icon(SNAME("increment_pressed"), SNAME("VScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement"), SNAME("VScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement_highlight"), SNAME("VScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement_pressed"), SNAME("VScrollBar"), empty_icon);

	// HSlider
	theme->set_icon(SNAME("grabber_highlight"), SNAME("HSlider"), theme->get_icon(SNAME("GuiSliderGrabberHl"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("grabber"), SNAME("HSlider"), theme->get_icon(SNAME("GuiSliderGrabber"), SNAME("EditorIcons")));
	theme->set_stylebox(SNAME("slider"), SNAME("HSlider"), make_flat_stylebox(dark_color_3, 0, default_margin_size / 2, 0, default_margin_size / 2, corner_width));
	theme->set_stylebox(SNAME("grabber_area"), SNAME("HSlider"), make_flat_stylebox(contrast_color_1, 0, default_margin_size / 2, 0, default_margin_size / 2, corner_width));
	theme->set_stylebox(SNAME("grabber_area_highlight"), SNAME("HSlider"), make_flat_stylebox(contrast_color_1, 0, default_margin_size / 2, 0, default_margin_size / 2));

	// VSlider
	theme->set_icon(SNAME("grabber"), SNAME("VSlider"), theme->get_icon(SNAME("GuiSliderGrabber"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("grabber_highlight"), SNAME("VSlider"), theme->get_icon(SNAME("GuiSliderGrabberHl"), SNAME("EditorIcons")));
	theme->set_stylebox(SNAME("slider"), SNAME("VSlider"), make_flat_stylebox(dark_color_3, default_margin_size / 2, 0, default_margin_size / 2, 0, corner_width));
	theme->set_stylebox(SNAME("grabber_area"), SNAME("VSlider"), make_flat_stylebox(contrast_color_1, default_margin_size / 2, 0, default_margin_size / 2, 0, corner_width));
	theme->set_stylebox(SNAME("grabber_area_highlight"), SNAME("VSlider"), make_flat_stylebox(contrast_color_1, default_margin_size / 2, 0, default_margin_size / 2, 0));

	// RichTextLabel
	theme->set_color(SNAME("default_color"), SNAME("RichTextLabel"), font_color);
	theme->set_color(SNAME("font_shadow_color"), SNAME("RichTextLabel"), Color(0, 0, 0, 0));
	theme->set_constant(SNAME("shadow_offset_x"), SNAME("RichTextLabel"), 1 * EDSCALE);
	theme->set_constant(SNAME("shadow_offset_y"), SNAME("RichTextLabel"), 1 * EDSCALE);
	theme->set_constant(SNAME("shadow_outline_size"), SNAME("RichTextLabel"), 1 * EDSCALE);
	theme->set_stylebox(SNAME("focus"), SNAME("RichTextLabel"), make_empty_stylebox());
	theme->set_stylebox(SNAME("normal"), SNAME("RichTextLabel"), style_tree_bg);

	// Editor help.
	theme->set_color(SNAME("title_color"), SNAME("EditorHelp"), accent_color);
	theme->set_color(SNAME("headline_color"), SNAME("EditorHelp"), mono_color);
	theme->set_color(SNAME("text_color"), SNAME("EditorHelp"), font_color);
	theme->set_color(SNAME("comment_color"), SNAME("EditorHelp"), font_color * Color(1, 1, 1, 0.6));
	theme->set_color(SNAME("symbol_color"), SNAME("EditorHelp"), font_color * Color(1, 1, 1, 0.6));
	theme->set_color(SNAME("value_color"), SNAME("EditorHelp"), font_color * Color(1, 1, 1, 0.6));
	theme->set_color(SNAME("qualifier_color"), SNAME("EditorHelp"), font_color * Color(1, 1, 1, 0.8));
	theme->set_color(SNAME("type_color"), SNAME("EditorHelp"), accent_color.lerp(font_color, 0.5));
	theme->set_color(SNAME("selection_color"), SNAME("EditorHelp"), accent_color * Color(1, 1, 1, 0.4));
	theme->set_color(SNAME("link_color"), SNAME("EditorHelp"), accent_color.lerp(mono_color, 0.8));
	theme->set_color(SNAME("code_color"), SNAME("EditorHelp"), accent_color.lerp(mono_color, 0.6));
	theme->set_color(SNAME("kbd_color"), SNAME("EditorHelp"), accent_color.lerp(property_color, 0.6));
	theme->set_constant(SNAME("line_separation"), SNAME("EditorHelp"), Math::round(6 * EDSCALE));
	theme->set_constant(SNAME("table_hseparation"), SNAME("EditorHelp"), 16 * EDSCALE);
	theme->set_constant(SNAME("table_vseparation"), SNAME("EditorHelp"), 6 * EDSCALE);

	// Panel
	theme->set_stylebox(SNAME("panel"), SNAME("Panel"), make_flat_stylebox(dark_color_1, 6, 4, 6, 4, corner_width));
	theme->set_stylebox(SNAME("PanelForeground"), SNAME("EditorStyles"), style_default);

	// Label
	theme->set_stylebox(SNAME("normal"), SNAME("Label"), style_empty);
	theme->set_color(SNAME("font_color"), SNAME("Label"), font_color);
	theme->set_color(SNAME("font_shadow_color"), SNAME("Label"), Color(0, 0, 0, 0));
	theme->set_constant(SNAME("shadow_offset_x"), SNAME("Label"), 1 * EDSCALE);
	theme->set_constant(SNAME("shadow_offset_y"), SNAME("Label"), 1 * EDSCALE);
	theme->set_constant(SNAME("shadow_outline_size"), SNAME("Label"), 1 * EDSCALE);
	theme->set_constant(SNAME("line_spacing"), SNAME("Label"), 3 * EDSCALE);

	// LinkButton
	theme->set_stylebox(SNAME("focus"), SNAME("LinkButton"), style_empty);
	theme->set_color(SNAME("font_color"), SNAME("LinkButton"), font_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("LinkButton"), font_hover_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("LinkButton"), font_focus_color);
	theme->set_color(SNAME("font_pressed_color"), SNAME("LinkButton"), accent_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("LinkButton"), font_disabled_color);

	// TooltipPanel
	Ref<StyleBoxFlat> style_tooltip = style_popup->duplicate();
	style_tooltip->set_shadow_size(0);
	style_tooltip->set_default_margin(SIDE_LEFT, default_margin_size * EDSCALE * 0.5);
	style_tooltip->set_default_margin(SIDE_TOP, default_margin_size * EDSCALE * 0.5);
	style_tooltip->set_default_margin(SIDE_RIGHT, default_margin_size * EDSCALE * 0.5);
	style_tooltip->set_default_margin(SIDE_BOTTOM, default_margin_size * EDSCALE * 0.5);
	style_tooltip->set_bg_color(dark_color_3 * Color(0.8, 0.8, 0.8, 0.9));
	style_tooltip->set_border_width_all(0);
	theme->set_color(SNAME("font_color"), SNAME("TooltipLabel"), font_hover_color);
	theme->set_color(SNAME("font_color_shadow"), SNAME("TooltipLabel"), Color(0, 0, 0, 0));
	theme->set_stylebox(SNAME("panel"), SNAME("TooltipPanel"), style_tooltip);

	// PopupPanel
	theme->set_stylebox(SNAME("panel"), SNAME("PopupPanel"), style_popup);

	// SpinBox
	theme->set_icon(SNAME("updown"), SNAME("SpinBox"), theme->get_icon(SNAME("GuiSpinboxUpdown"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("updown_disabled"), SNAME("SpinBox"), theme->get_icon(SNAME("GuiSpinboxUpdownDisabled"), SNAME("EditorIcons")));

	// ProgressBar
	theme->set_stylebox(SNAME("bg"), SNAME("ProgressBar"), make_stylebox(theme->get_icon(SNAME("GuiProgressBar"), SNAME("EditorIcons")), 4, 4, 4, 4, 0, 0, 0, 0));
	theme->set_stylebox(SNAME("fg"), SNAME("ProgressBar"), make_stylebox(theme->get_icon(SNAME("GuiProgressFill"), SNAME("EditorIcons")), 6, 6, 6, 6, 2, 1, 2, 1));
	theme->set_color(SNAME("font_color"), SNAME("ProgressBar"), font_color);

	// GraphEdit
	theme->set_stylebox(SNAME("bg"), "GraphEdit", style_tree_bg);
	if (dark_theme) {
		theme->set_color(SNAME("grid_major"), SNAME("GraphEdit"), Color(1.0, 1.0, 1.0, 0.15));
		theme->set_color(SNAME("grid_minor"), SNAME("GraphEdit"), Color(1.0, 1.0, 1.0, 0.07));
	} else {
		theme->set_color(SNAME("grid_major"), SNAME("GraphEdit"), Color(0.0, 0.0, 0.0, 0.15));
		theme->set_color(SNAME("grid_minor"), SNAME("GraphEdit"), Color(0.0, 0.0, 0.0, 0.07));
	}
	theme->set_color(SNAME("selection_fill"), SNAME("GraphEdit"), theme->get_color(SNAME("box_selection_fill_color"), SNAME("Editor")));
	theme->set_color(SNAME("selection_stroke"), SNAME("GraphEdit"), theme->get_color(SNAME("box_selection_stroke_color"), SNAME("Editor")));
	theme->set_color(SNAME("activity"), SNAME("GraphEdit"), accent_color);
	theme->set_icon(SNAME("minus"), SNAME("GraphEdit"), theme->get_icon(SNAME("ZoomLess"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("more"), SNAME("GraphEdit"), theme->get_icon(SNAME("ZoomMore"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("reset"), SNAME("GraphEdit"), theme->get_icon(SNAME("ZoomReset"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("snap"), SNAME("GraphEdit"), theme->get_icon(SNAME("SnapGrid"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("minimap"), SNAME("GraphEdit"), theme->get_icon(SNAME("GridMinimap"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("layout"), SNAME("GraphEdit"), theme->get_icon(SNAME("GridLayout"), SNAME("EditorIcons")));
	theme->set_constant(SNAME("bezier_len_pos"), SNAME("GraphEdit"), 80 * EDSCALE);
	theme->set_constant(SNAME("bezier_len_neg"), SNAME("GraphEdit"), 160 * EDSCALE);

	// GraphEditMinimap
	Ref<StyleBoxFlat> style_minimap_bg = make_flat_stylebox(dark_color_1, 0, 0, 0, 0);
	style_minimap_bg->set_border_color(dark_color_3);
	style_minimap_bg->set_border_width_all(1);
	theme->set_stylebox(SNAME("bg"), SNAME("GraphEditMinimap"), style_minimap_bg);

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
	style_minimap_node->set_corner_radius_all(1);
	theme->set_stylebox(SNAME("camera"), SNAME("GraphEditMinimap"), style_minimap_camera);
	theme->set_stylebox(SNAME("node"), SNAME("GraphEditMinimap"), style_minimap_node);

	Ref<Texture2D> minimap_resizer_icon = theme->get_icon(SNAME("GuiResizer"), SNAME("EditorIcons"));
	Color minimap_resizer_color;
	if (dark_theme) {
		minimap_resizer_color = Color(1, 1, 1, 0.65);
	} else {
		minimap_resizer_color = Color(0, 0, 0, 0.65);
	}
	theme->set_icon(SNAME("resizer"), SNAME("GraphEditMinimap"), flip_icon(minimap_resizer_icon, true, true));
	theme->set_color(SNAME("resizer_color"), SNAME("GraphEditMinimap"), minimap_resizer_color);

	// GraphNode
	const int gn_margin_side = 28;

	Ref<StyleBoxFlat> graphsb = make_flat_stylebox(dark_color_3 * Color(1, 1, 1, 0.7), gn_margin_side, 24, gn_margin_side, 5, corner_width);
	graphsb->set_border_width_all(border_width);
	graphsb->set_border_color(dark_color_3);
	Ref<StyleBoxFlat> graphsbselected = make_flat_stylebox(dark_color_3 * Color(1, 1, 1, 0.9), gn_margin_side, 24, gn_margin_side, 5, corner_width);
	graphsbselected->set_border_width_all(2 * EDSCALE + border_width);
	graphsbselected->set_border_color(Color(accent_color.r, accent_color.g, accent_color.b, 0.9));
	Ref<StyleBoxFlat> graphsbcomment = make_flat_stylebox(dark_color_3 * Color(1, 1, 1, 0.3), gn_margin_side, 24, gn_margin_side, 5, corner_width);
	graphsbcomment->set_border_width_all(border_width);
	graphsbcomment->set_border_color(dark_color_3);
	Ref<StyleBoxFlat> graphsbcommentselected = make_flat_stylebox(dark_color_3 * Color(1, 1, 1, 0.4), gn_margin_side, 24, gn_margin_side, 5, corner_width);
	graphsbcommentselected->set_border_width_all(border_width);
	graphsbcommentselected->set_border_color(dark_color_3);
	Ref<StyleBoxFlat> graphsbbreakpoint = graphsbselected->duplicate();
	graphsbbreakpoint->set_draw_center(false);
	graphsbbreakpoint->set_border_color(warning_color);
	graphsbbreakpoint->set_shadow_color(warning_color * Color(1.0, 1.0, 1.0, 0.1));
	Ref<StyleBoxFlat> graphsbposition = graphsbselected->duplicate();
	graphsbposition->set_draw_center(false);
	graphsbposition->set_border_color(error_color);
	graphsbposition->set_shadow_color(error_color * Color(1.0, 1.0, 1.0, 0.2));
	Ref<StyleBoxFlat> smgraphsb = make_flat_stylebox(dark_color_3 * Color(1, 1, 1, 0.7), gn_margin_side, 24, gn_margin_side, 5, corner_width);
	smgraphsb->set_border_width_all(border_width);
	smgraphsb->set_border_color(dark_color_3);
	Ref<StyleBoxFlat> smgraphsbselected = make_flat_stylebox(dark_color_3 * Color(1, 1, 1, 0.9), gn_margin_side, 24, gn_margin_side, 5, corner_width);
	smgraphsbselected->set_border_width_all(2 * EDSCALE + border_width);
	smgraphsbselected->set_border_color(Color(accent_color.r, accent_color.g, accent_color.b, 0.9));
	smgraphsbselected->set_shadow_size(8 * EDSCALE);
	smgraphsbselected->set_shadow_color(shadow_color);

	graphsb->set_border_width(SIDE_TOP, 24 * EDSCALE);
	graphsbselected->set_border_width(SIDE_TOP, 24 * EDSCALE);
	graphsbcomment->set_border_width(SIDE_TOP, 24 * EDSCALE);
	graphsbcommentselected->set_border_width(SIDE_TOP, 24 * EDSCALE);

	theme->set_stylebox(SNAME("frame"), SNAME("GraphNode"), graphsb);
	theme->set_stylebox(SNAME("selectedframe"), SNAME("GraphNode"), graphsbselected);
	theme->set_stylebox(SNAME("comment"), SNAME("GraphNode"), graphsbcomment);
	theme->set_stylebox(SNAME("commentfocus"), SNAME("GraphNode"), graphsbcommentselected);
	theme->set_stylebox(SNAME("breakpoint"), SNAME("GraphNode"), graphsbbreakpoint);
	theme->set_stylebox(SNAME("position"), SNAME("GraphNode"), graphsbposition);
	theme->set_stylebox(SNAME("state_machine_frame"), SNAME("GraphNode"), smgraphsb);
	theme->set_stylebox(SNAME("state_machine_selectedframe"), SNAME("GraphNode"), smgraphsbselected);

	Color default_node_color = dark_color_1.inverted();
	theme->set_color(SNAME("title_color"), SNAME("GraphNode"), default_node_color);
	default_node_color.a = 0.7;
	theme->set_color(SNAME("close_color"), SNAME("GraphNode"), default_node_color);
	theme->set_color(SNAME("resizer_color"), SNAME("GraphNode"), default_node_color);

	theme->set_constant(SNAME("port_offset"), SNAME("GraphNode"), 14 * EDSCALE);
	theme->set_constant(SNAME("title_h_offset"), SNAME("GraphNode"), -16 * EDSCALE);
	theme->set_constant(SNAME("title_offset"), SNAME("GraphNode"), 20 * EDSCALE);
	theme->set_constant(SNAME("close_h_offset"), SNAME("GraphNode"), 20 * EDSCALE);
	theme->set_constant(SNAME("close_offset"), SNAME("GraphNode"), 20 * EDSCALE);
	theme->set_constant(SNAME("separation"), SNAME("GraphNode"), 1 * EDSCALE);

	theme->set_icon(SNAME("close"), SNAME("GraphNode"), theme->get_icon(SNAME("GuiCloseCustomizable"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("resizer"), SNAME("GraphNode"), theme->get_icon(SNAME("GuiResizer"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("port"), SNAME("GraphNode"), theme->get_icon(SNAME("GuiGraphNodePort"), SNAME("EditorIcons")));

	// GridContainer
	theme->set_constant(SNAME("vseparation"), SNAME("GridContainer"), Math::round(widget_default_margin.y - 2 * EDSCALE));

	// FileDialog
	theme->set_icon(SNAME("folder"), SNAME("FileDialog"), theme->get_icon(SNAME("Folder"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("parent_folder"), SNAME("FileDialog"), theme->get_icon(SNAME("ArrowUp"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("back_folder"), SNAME("FileDialog"), theme->get_icon(SNAME("Back"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("forward_folder"), SNAME("FileDialog"), theme->get_icon(SNAME("Forward"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("reload"), SNAME("FileDialog"), theme->get_icon(SNAME("Reload"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("toggle_hidden"), SNAME("FileDialog"), theme->get_icon(SNAME("GuiVisibilityVisible"), SNAME("EditorIcons")));
	// Use a different color for folder icons to make them easier to distinguish from files.
	// On a light theme, the icon will be dark, so we need to lighten it before blending it with the accent color.
	theme->set_color(SNAME("folder_icon_modulate"), SNAME("FileDialog"), (dark_theme ? Color(1, 1, 1) : Color(4.25, 4.25, 4.25)).lerp(accent_color, 0.7));
	theme->set_color(SNAME("files_disabled"), SNAME("FileDialog"), font_disabled_color);

	// ColorPicker
	theme->set_constant(SNAME("margin"), SNAME("ColorPicker"), popup_margin_size);
	theme->set_constant(SNAME("sv_width"), SNAME("ColorPicker"), 256 * EDSCALE);
	theme->set_constant(SNAME("sv_height"), SNAME("ColorPicker"), 256 * EDSCALE);
	theme->set_constant(SNAME("h_width"), SNAME("ColorPicker"), 30 * EDSCALE);
	theme->set_constant(SNAME("label_width"), SNAME("ColorPicker"), 10 * EDSCALE);
	theme->set_icon(SNAME("screen_picker"), SNAME("ColorPicker"), theme->get_icon(SNAME("ColorPick"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("add_preset"), SNAME("ColorPicker"), theme->get_icon(SNAME("Add"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("sample_bg"), SNAME("ColorPicker"), theme->get_icon(SNAME("GuiMiniCheckerboard"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("overbright_indicator"), SNAME("ColorPicker"), theme->get_icon(SNAME("OverbrightIndicator"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("bar_arrow"), SNAME("ColorPicker"), theme->get_icon(SNAME("ColorPickerBarArrow"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("picker_cursor"), SNAME("ColorPicker"), theme->get_icon(SNAME("PickerCursor"), SNAME("EditorIcons")));

	// ColorPickerButton
	theme->set_icon(SNAME("bg"), SNAME("ColorPickerButton"), theme->get_icon(SNAME("GuiMiniCheckerboard"), SNAME("EditorIcons")));

	// ColorPresetButton
	Ref<StyleBoxFlat> preset_sb = make_flat_stylebox(Color(1, 1, 1), 2, 2, 2, 2, 2);
	preset_sb->set_anti_aliased(false);
	theme->set_stylebox(SNAME("preset_fg"), SNAME("ColorPresetButton"), preset_sb);
	theme->set_icon(SNAME("preset_bg"), SNAME("ColorPresetButton"), theme->get_icon(SNAME("GuiMiniCheckerboard"), SNAME("EditorIcons")));
	theme->set_icon(SNAME("overbright_indicator"), SNAME("ColorPresetButton"), theme->get_icon(SNAME("OverbrightIndicator"), SNAME("EditorIcons")));

	// Information on 3D viewport
	Ref<StyleBoxFlat> style_info_3d_viewport = style_default->duplicate();
	style_info_3d_viewport->set_bg_color(style_info_3d_viewport->get_bg_color() * Color(1, 1, 1, 0.5));
	style_info_3d_viewport->set_border_width_all(0);
	theme->set_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles"), style_info_3d_viewport);

	// Asset Library.
	theme->set_stylebox(SNAME("panel"), SNAME("AssetLib"), style_content_panel);
	theme->set_color(SNAME("status_color"), SNAME("AssetLib"), Color(0.5, 0.5, 0.5));
	theme->set_icon(SNAME("dismiss"), SNAME("AssetLib"), theme->get_icon(SNAME("Close"), SNAME("EditorIcons")));

	// Theme editor.
	theme->set_color(SNAME("preview_picker_overlay_color"), SNAME("ThemeEditor"), Color(0.1, 0.1, 0.1, 0.25));
	Color theme_preview_picker_bg_color = accent_color;
	theme_preview_picker_bg_color.a = 0.2;
	Ref<StyleBoxFlat> theme_preview_picker_sb = make_flat_stylebox(theme_preview_picker_bg_color, 0, 0, 0, 0);
	theme_preview_picker_sb->set_border_color(accent_color);
	theme_preview_picker_sb->set_border_width_all(1.0 * EDSCALE);
	theme->set_stylebox(SNAME("preview_picker_overlay"), SNAME("ThemeEditor"), theme_preview_picker_sb);
	Color theme_preview_picker_label_bg_color = accent_color;
	theme_preview_picker_label_bg_color.set_v(0.5);
	Ref<StyleBoxFlat> theme_preview_picker_label_sb = make_flat_stylebox(theme_preview_picker_label_bg_color, 4.0, 1.0, 4.0, 3.0);
	theme->set_stylebox(SNAME("preview_picker_label"), SNAME("ThemeEditor"), theme_preview_picker_label_sb);

	// adaptive script theme constants
	// for comments and elements with lower relevance
	const Color dim_color = Color(font_color.r, font_color.g, font_color.b, 0.5);

	const float mono_value = mono_color.r;
	const Color alpha1 = Color(mono_value, mono_value, mono_value, 0.07);
	const Color alpha2 = Color(mono_value, mono_value, mono_value, 0.14);
	const Color alpha3 = Color(mono_value, mono_value, mono_value, 0.7);

	// editor main color
	const Color main_color = dark_theme ? Color(0.34, 0.7, 1.0) : Color(0.02, 0.5, 1.0);

	const Color symbol_color = Color(0.34, 0.57, 1.0).lerp(mono_color, dark_theme ? 0.5 : 0.3);
	const Color keyword_color = Color(1.0, 0.44, 0.52);
	const Color control_flow_keyword_color = dark_theme ? Color(1.0, 0.55, 0.8) : Color(0.8, 0.4, 0.6);
	const Color basetype_color = dark_theme ? Color(0.26, 1.0, 0.76) : Color(0.0, 0.76, 0.38);
	const Color type_color = basetype_color.lerp(mono_color, dark_theme ? 0.4 : 0.3);
	const Color usertype_color = basetype_color.lerp(mono_color, dark_theme ? 0.7 : 0.5);
	const Color comment_color = dim_color;
	const Color string_color = (dark_theme ? Color(1.0, 0.85, 0.26) : Color(1.0, 0.82, 0.09)).lerp(mono_color, dark_theme ? 0.5 : 0.3);

	// Use the brightest background color on a light theme (which generally uses a negative contrast rate).
	const Color te_background_color = dark_theme ? background_color : dark_color_3;
	const Color completion_background_color = dark_theme ? base_color : background_color;
	const Color completion_selected_color = alpha1;
	const Color completion_existing_color = alpha2;
	const Color completion_scroll_color = alpha1;
	const Color completion_font_color = font_color;
	const Color text_color = font_color;
	const Color line_number_color = dim_color;
	const Color safe_line_number_color = dim_color * Color(1, 1.2, 1, 1.5);
	const Color caret_color = mono_color;
	const Color caret_background_color = mono_color.inverted();
	const Color text_selected_color = dark_color_3;
	const Color brace_mismatch_color = error_color;
	const Color current_line_color = alpha1;
	const Color line_length_guideline_color = dark_theme ? base_color : background_color;
	const Color word_highlighted_color = alpha1;
	const Color number_color = basetype_color.lerp(mono_color, dark_theme ? 0.5 : 0.3);
	const Color function_color = main_color;
	const Color member_variable_color = main_color.lerp(mono_color, 0.6);
	const Color mark_color = Color(error_color.r, error_color.g, error_color.b, 0.3);
	const Color bookmark_color = Color(0.08, 0.49, 0.98);
	const Color breakpoint_color = error_color;
	const Color executing_line_color = Color(0.98, 0.89, 0.27);
	const Color code_folding_color = alpha3;
	const Color search_result_color = alpha1;
	const Color search_result_border_color = Color(0.41, 0.61, 0.91, 0.38);

	EditorSettings *setting = EditorSettings::get_singleton();
	String text_editor_color_theme = setting->get("text_editor/theme/color_theme");
	if (text_editor_color_theme == "Default") {
		setting->set_initial_value("text_editor/theme/highlighting/symbol_color", symbol_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/keyword_color", keyword_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/control_flow_keyword_color", control_flow_keyword_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/base_type_color", basetype_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/engine_type_color", type_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/user_type_color", usertype_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/comment_color", comment_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/string_color", string_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/background_color", te_background_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/completion_background_color", completion_background_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/completion_selected_color", completion_selected_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/completion_existing_color", completion_existing_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/completion_scroll_color", completion_scroll_color, true);
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
		setting->set_initial_value("text_editor/theme/highlighting/search_result_color", search_result_color, true);
		setting->set_initial_value("text_editor/theme/highlighting/search_result_border_color", search_result_border_color, true);
	} else if (text_editor_color_theme == "Godot 2") {
		setting->load_text_editor_theme();
	}

	// Now theme is loaded, apply it to CodeEdit.
	theme->set_color(SNAME("background_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/background_color"));
	theme->set_color(SNAME("completion_background_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/completion_background_color"));
	theme->set_color(SNAME("completion_selected_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/completion_selected_color"));
	theme->set_color(SNAME("completion_existing_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/completion_existing_color"));
	theme->set_color(SNAME("completion_scroll_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/completion_scroll_color"));
	theme->set_color(SNAME("completion_font_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/completion_font_color"));
	theme->set_color(SNAME("font_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/text_color"));
	theme->set_color(SNAME("line_number_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/line_number_color"));
	theme->set_color(SNAME("caret_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/caret_color"));
	theme->set_color(SNAME("font_selected_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/text_selected_color"));
	theme->set_color(SNAME("selection_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/selection_color"));
	theme->set_color(SNAME("brace_mismatch_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/brace_mismatch_color"));
	theme->set_color(SNAME("current_line_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/current_line_color"));
	theme->set_color(SNAME("line_length_guideline_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/line_length_guideline_color"));
	theme->set_color(SNAME("word_highlighted_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/word_highlighted_color"));
	theme->set_color(SNAME("bookmark_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/bookmark_color"));
	theme->set_color(SNAME("breakpoint_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/breakpoint_color"));
	theme->set_color(SNAME("executing_line_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/executing_line_color"));
	theme->set_color(SNAME("code_folding_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/code_folding_color"));
	theme->set_color(SNAME("search_result_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/search_result_color"));
	theme->set_color(SNAME("search_result_border_color"), SNAME("CodeEdit"), EDITOR_GET("text_editor/theme/highlighting/search_result_border_color"));

	return theme;
}

Ref<Theme> create_custom_theme(const Ref<Theme> p_theme) {
	Ref<Theme> theme = create_editor_theme(p_theme);

	const String custom_theme_path = EditorSettings::get_singleton()->get("interface/theme/custom_theme");
	if (!custom_theme_path.is_empty()) {
		Ref<Theme> custom_theme = ResourceLoader::load(custom_theme_path);
		if (custom_theme.is_valid()) {
			theme->merge_with(custom_theme);
		}
	}

	return theme;
}

Ref<ImageTexture> create_unscaled_default_project_icon() {
#ifdef MODULE_SVG_ENABLED
	for (int i = 0; i < editor_icons_count; i++) {
		// ESCALE should never affect size of the icon
		if (strcmp(editor_icons_names[i], "DefaultProjectIcon") == 0) {
			return editor_generate_icon(i, false, 1.0);
		}
	}
#endif
	return Ref<ImageTexture>(memnew(ImageTexture));
}
