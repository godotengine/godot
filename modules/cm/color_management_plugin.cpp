/*************************************************************************/
/*  color_profile.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "color_management_plugin.h"
#include "color_profile.h"
#include "color_transform.h"
#include "editor/editor_settings.h"
#include "servers/visual_server.h"

void ColorManagementPlugin::_clear_lut() {
	VS::get_singleton()->set_screen_lut(Ref<Image>(), 0, 0);
}

void ColorManagementPlugin::_on_settings_change() {
	String profile = EDITOR_GET("interface/color_management/screen_profile");
	if (profile.empty()) {
		return _clear_lut();
	}

	Ref<ColorProfile> screen_profile = memnew(ColorProfile(profile));
	if (screen_profile.is_null() || !screen_profile->is_valid()) {
		ERR_PRINTS("Failed to load profile from: " + profile + ".");
		return _clear_lut();
	}

	Ref<ColorTransform> transform = memnew(ColorTransform());
	if (transform.is_null()) {
		ERR_FAIL_V(_clear_lut());
	}

	transform->set_src_profile(memnew(ColorProfile(ColorProfile::PREDEF_SRGB)));
	transform->set_dst_profile(screen_profile);

	String intent = EDITOR_GET("interface/color_management/intent");
	if (intent == "Perceptual") {
		transform->set_intent(ColorTransform::CM_INTENT_PERCEPTUAL);
	} else if (intent == "Relative Colorimetric") {
		transform->set_intent(ColorTransform::CM_INTENT_RELATIVE);
	} else if (intent == "Saturation") {
		transform->set_intent(ColorTransform::CM_INTENT_SATURATION);
	} else if (intent == "Absolute Colorimetric") {
		transform->set_intent(ColorTransform::CM_INTENT_ABSOLUTE);
	} else {
		WARN_PRINTS("Unexpected color management intent: " + intent + ". Falling back to Perceptual.");
		transform->set_intent(ColorTransform::CM_INTENT_PERCEPTUAL);
	}

	transform->set_black_point_compensation(EDITOR_GET("interface/color_management/black_point_compensation"));

	if (!transform->apply_screen_lut()) {
		ERR_PRINTS("Failed to apply screen LUT, falling back to identity");
		_clear_lut();
	}
}

void ColorManagementPlugin::_register_settings() {
	EditorSettings *es = EditorSettings::get_singleton();

	EDITOR_DEF("interface/color_management/screen_profile", "");
	es->add_property_hint(PropertyInfo(Variant::STRING, "interface/color_management/screen_profile", PROPERTY_HINT_GLOBAL_FILE, "*.icc,*.icm", PROPERTY_USAGE_DEFAULT));

	EDITOR_DEF("interface/color_management/intent", "Perceptual");
	es->add_property_hint(PropertyInfo(Variant::STRING, "interface/color_management/intent", PROPERTY_HINT_ENUM,
			"Perceptual"
			","
			"Relative Colorimetric"
			","
			"Saturation"
			","
			"Absolute Colorimetric"));

	EDITOR_DEF("interface/color_management/black_point_compensation", true);
	es->add_property_hint(PropertyInfo(Variant::BOOL, "interface/color_management/black_point_compensation"));
}

void ColorManagementPlugin::_bind_methods() {
	ClassDB::bind_method("_editor_settings_changed", &ColorManagementPlugin::_on_settings_change);
}

ColorManagementPlugin::ColorManagementPlugin() {
	EditorSettings::get_singleton()->connect("settings_changed", this, "_editor_settings_changed");
	_register_settings();
	_on_settings_change();
}