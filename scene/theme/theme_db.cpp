/*************************************************************************/
/*  theme_db.cpp                                                         */
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

#include "theme_db.h"

#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "scene/resources/default_theme/default_theme.h"
#include "scene/resources/font.h"
#include "scene/resources/style_box.h"
#include "scene/resources/texture.h"
#include "scene/resources/theme.h"
#include "servers/text_server.h"

// Default engine theme creation and configuration.
void ThemeDB::initialize_theme() {
	// Allow creating the default theme at a different scale to suit higher/lower base resolutions.
	float default_theme_scale = GLOBAL_DEF("gui/theme/default_theme_scale", 1.0);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/default_theme_scale", PropertyInfo(Variant::FLOAT, "gui/theme/default_theme_scale", PROPERTY_HINT_RANGE, "0.5,8,0.01", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));

	String theme_path = GLOBAL_DEF_RST("gui/theme/custom", "");
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/custom", PropertyInfo(Variant::STRING, "gui/theme/custom", PROPERTY_HINT_FILE, "*.tres,*.res,*.theme", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));

	String font_path = GLOBAL_DEF_RST("gui/theme/custom_font", "");
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/custom_font", PropertyInfo(Variant::STRING, "gui/theme/custom_font", PROPERTY_HINT_FILE, "*.tres,*.res,*.otf,*.ttf,*.woff,*.woff2,*.fnt,*.font,*.pfb,*.pfm", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));

	TextServer::FontAntialiasing font_antialiasing = (TextServer::FontAntialiasing)(int)GLOBAL_DEF_RST("gui/theme/default_font_antialiasing", 1);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/default_font_antialiasing", PropertyInfo(Variant::INT, "gui/theme/default_font_antialiasing", PROPERTY_HINT_ENUM, "None,Grayscale,LCD Subpixel", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));

	TextServer::Hinting font_hinting = (TextServer::Hinting)(int)GLOBAL_DEF_RST("gui/theme/default_font_hinting", TextServer::HINTING_LIGHT);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/default_font_hinting", PropertyInfo(Variant::INT, "gui/theme/default_font_hinting", PROPERTY_HINT_ENUM, "None,Light,Normal", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));

	TextServer::SubpixelPositioning font_subpixel_positioning = (TextServer::SubpixelPositioning)(int)GLOBAL_DEF_RST("gui/theme/default_font_subpixel_positioning", TextServer::SUBPIXEL_POSITIONING_AUTO);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/default_font_subpixel_positioning", PropertyInfo(Variant::INT, "gui/theme/default_font_subpixel_positioning", PROPERTY_HINT_ENUM, "Disabled,Auto,One Half of a Pixel,One Quarter of a Pixel", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));

	const bool font_msdf = GLOBAL_DEF_RST("gui/theme/default_font_multichannel_signed_distance_field", false);
	const bool font_generate_mipmaps = GLOBAL_DEF_RST("gui/theme/default_font_generate_mipmaps", false);

	GLOBAL_DEF_RST("gui/theme/lcd_subpixel_layout", 1);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/lcd_subpixel_layout", PropertyInfo(Variant::INT, "gui/theme/lcd_subpixel_layout", PROPERTY_HINT_ENUM, "Disabled,Horizontal RGB,Horizontal BGR,Vertical RGB,Vertical BGR"));
	ProjectSettings::get_singleton()->set_restart_if_changed("gui/theme/lcd_subpixel_layout", false);

	Ref<Font> font;
	if (!font_path.is_empty()) {
		font = ResourceLoader::load(font_path);
		if (font.is_valid()) {
			set_fallback_font(font);
		} else {
			ERR_PRINT("Error loading custom font '" + font_path + "'");
		}
	}

	// Always make the default theme to avoid invalid default font/icon/style in the given theme.
	if (RenderingServer::get_singleton()) {
		make_default_theme(default_theme_scale, font, font_subpixel_positioning, font_hinting, font_antialiasing, font_msdf, font_generate_mipmaps);
	}

	if (!theme_path.is_empty()) {
		Ref<Theme> theme = ResourceLoader::load(theme_path);
		if (theme.is_valid()) {
			set_project_theme(theme);
		} else {
			ERR_PRINT("Error loading custom theme '" + theme_path + "'");
		}
	}
}

void ThemeDB::initialize_theme_noproject() {
	if (RenderingServer::get_singleton()) {
		make_default_theme(1.0, Ref<Font>());
	}
}

// Universal fallback Theme resources.

void ThemeDB::set_default_theme(const Ref<Theme> &p_default) {
	default_theme = p_default;
}

Ref<Theme> ThemeDB::get_default_theme() {
	return default_theme;
}

void ThemeDB::set_project_theme(const Ref<Theme> &p_project_default) {
	project_theme = p_project_default;
}

Ref<Theme> ThemeDB::get_project_theme() {
	return project_theme;
}

// Universal fallback values for theme item types.

void ThemeDB::set_fallback_base_scale(float p_base_scale) {
	if (fallback_base_scale == p_base_scale) {
		return;
	}

	fallback_base_scale = p_base_scale;
	emit_signal(SNAME("fallback_changed"));
}

float ThemeDB::get_fallback_base_scale() {
	return fallback_base_scale;
}

void ThemeDB::set_fallback_font(const Ref<Font> &p_font) {
	if (fallback_font == p_font) {
		return;
	}

	fallback_font = p_font;
	emit_signal(SNAME("fallback_changed"));
}

Ref<Font> ThemeDB::get_fallback_font() {
	return fallback_font;
}

void ThemeDB::set_fallback_font_size(int p_font_size) {
	if (fallback_font_size == p_font_size) {
		return;
	}

	fallback_font_size = p_font_size;
	emit_signal(SNAME("fallback_changed"));
}

int ThemeDB::get_fallback_font_size() {
	return fallback_font_size;
}

void ThemeDB::set_fallback_icon(const Ref<Texture2D> &p_icon) {
	if (fallback_icon == p_icon) {
		return;
	}

	fallback_icon = p_icon;
	emit_signal(SNAME("fallback_changed"));
}

Ref<Texture2D> ThemeDB::get_fallback_icon() {
	return fallback_icon;
}

void ThemeDB::set_fallback_stylebox(const Ref<StyleBox> &p_stylebox) {
	if (fallback_stylebox == p_stylebox) {
		return;
	}

	fallback_stylebox = p_stylebox;
	emit_signal(SNAME("fallback_changed"));
}

Ref<StyleBox> ThemeDB::get_fallback_stylebox() {
	return fallback_stylebox;
}

// Object methods.
void ThemeDB::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_default_theme"), &ThemeDB::get_default_theme);
	ClassDB::bind_method(D_METHOD("get_project_theme"), &ThemeDB::get_project_theme);

	ClassDB::bind_method(D_METHOD("set_fallback_base_scale", "base_scale"), &ThemeDB::set_fallback_base_scale);
	ClassDB::bind_method(D_METHOD("get_fallback_base_scale"), &ThemeDB::get_fallback_base_scale);
	ClassDB::bind_method(D_METHOD("set_fallback_font", "font"), &ThemeDB::set_fallback_font);
	ClassDB::bind_method(D_METHOD("get_fallback_font"), &ThemeDB::get_fallback_font);
	ClassDB::bind_method(D_METHOD("set_fallback_font_size", "font_size"), &ThemeDB::set_fallback_font_size);
	ClassDB::bind_method(D_METHOD("get_fallback_font_size"), &ThemeDB::get_fallback_font_size);
	ClassDB::bind_method(D_METHOD("set_fallback_icon", "icon"), &ThemeDB::set_fallback_icon);
	ClassDB::bind_method(D_METHOD("get_fallback_icon"), &ThemeDB::get_fallback_icon);
	ClassDB::bind_method(D_METHOD("set_fallback_stylebox", "stylebox"), &ThemeDB::set_fallback_stylebox);
	ClassDB::bind_method(D_METHOD("get_fallback_stylebox"), &ThemeDB::get_fallback_stylebox);

	ADD_GROUP("Fallback values", "fallback_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fallback_base_scale", PROPERTY_HINT_RANGE, "0.0,2.0,0.01,or_greater"), "set_fallback_base_scale", "get_fallback_base_scale");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "fallback_font", PROPERTY_HINT_RESOURCE_TYPE, "Font", PROPERTY_USAGE_NONE), "set_fallback_font", "get_fallback_font");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fallback_font_size", PROPERTY_HINT_RANGE, "0,256,1,or_greater,suffix:px"), "set_fallback_font_size", "get_fallback_font_size");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "fallback_icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", PROPERTY_USAGE_NONE), "set_fallback_icon", "get_fallback_icon");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "fallback_stylebox", PROPERTY_HINT_RESOURCE_TYPE, "StyleBox", PROPERTY_USAGE_NONE), "set_fallback_stylebox", "get_fallback_stylebox");

	ADD_SIGNAL(MethodInfo("fallback_changed"));
}

// Memory management, reference, and initialization
ThemeDB *ThemeDB::singleton = nullptr;

ThemeDB *ThemeDB::get_singleton() {
	return singleton;
}

ThemeDB::ThemeDB() {
	singleton = this;

	// Universal default values, final fallback for every theme.
	fallback_base_scale = 1.0;
	fallback_font_size = 16;
}

ThemeDB::~ThemeDB() {
	default_theme.unref();
	project_theme.unref();

	fallback_font.unref();
	fallback_icon.unref();
	fallback_stylebox.unref();

	singleton = nullptr;
}
