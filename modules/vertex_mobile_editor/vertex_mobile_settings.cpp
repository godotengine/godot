/**************************************************************************/
/*  vertex_mobile_settings.cpp                                            */
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

#include "vertex_mobile_settings.h"

#include "core/object/class_db.h"

Dictionary VertexMobileSettings::to_dictionary() const {
	Dictionary d;
	d["touch_target_size"] = touch_target_size;
	d["compact_toolbar_height"] = compact_toolbar_height;
	d["panel_toggle_size"] = panel_toggle_size;
	d["pinch_zoom_min"] = pinch_zoom_min;
	d["pinch_zoom_max"] = pinch_zoom_max;
	d["gesture_pan_threshold"] = gesture_pan_threshold;
	d["compact_mode"] = compact_mode;
	d["panels_collapsed"] = panels_collapsed;
	d["virtual_keyboard_assist"] = virtual_keyboard_assist;
	d["screen_density_dp"] = screen_density_dp;
	return d;
}

void VertexMobileSettings::from_dictionary(const Dictionary &p_dict) {
	if (p_dict.has("touch_target_size")) {
		touch_target_size = p_dict["touch_target_size"];
	}
	if (p_dict.has("compact_toolbar_height")) {
		compact_toolbar_height = p_dict["compact_toolbar_height"];
	}
	if (p_dict.has("panel_toggle_size")) {
		panel_toggle_size = p_dict["panel_toggle_size"];
	}
	if (p_dict.has("pinch_zoom_min")) {
		pinch_zoom_min = p_dict["pinch_zoom_min"];
	}
	if (p_dict.has("pinch_zoom_max")) {
		pinch_zoom_max = p_dict["pinch_zoom_max"];
	}
	if (p_dict.has("gesture_pan_threshold")) {
		gesture_pan_threshold = p_dict["gesture_pan_threshold"];
	}
	if (p_dict.has("compact_mode")) {
		compact_mode = p_dict["compact_mode"];
	}
	if (p_dict.has("panels_collapsed")) {
		panels_collapsed = p_dict["panels_collapsed"];
	}
	if (p_dict.has("virtual_keyboard_assist")) {
		virtual_keyboard_assist = p_dict["virtual_keyboard_assist"];
	}
	if (p_dict.has("screen_density_dp")) {
		screen_density_dp = p_dict["screen_density_dp"];
	}
}

void VertexMobileSettings::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_touch_target_size", "size"), &VertexMobileSettings::set_touch_target_size);
	ClassDB::bind_method(D_METHOD("get_touch_target_size"), &VertexMobileSettings::get_touch_target_size);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "touch_target_size", PROPERTY_HINT_RANGE, "32,96,1,suffix:dp"), "set_touch_target_size", "get_touch_target_size");

	ClassDB::bind_method(D_METHOD("set_compact_toolbar_height", "height"), &VertexMobileSettings::set_compact_toolbar_height);
	ClassDB::bind_method(D_METHOD("get_compact_toolbar_height"), &VertexMobileSettings::get_compact_toolbar_height);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "compact_toolbar_height", PROPERTY_HINT_RANGE, "32,72,1,suffix:dp"), "set_compact_toolbar_height", "get_compact_toolbar_height");

	ClassDB::bind_method(D_METHOD("set_panel_toggle_size", "size"), &VertexMobileSettings::set_panel_toggle_size);
	ClassDB::bind_method(D_METHOD("get_panel_toggle_size"), &VertexMobileSettings::get_panel_toggle_size);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "panel_toggle_size", PROPERTY_HINT_RANGE, "32,72,1,suffix:dp"), "set_panel_toggle_size", "get_panel_toggle_size");

	ClassDB::bind_method(D_METHOD("set_pinch_zoom_min", "min"), &VertexMobileSettings::set_pinch_zoom_min);
	ClassDB::bind_method(D_METHOD("get_pinch_zoom_min"), &VertexMobileSettings::get_pinch_zoom_min);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pinch_zoom_min", PROPERTY_HINT_RANGE, "0.1,1,0.05"), "set_pinch_zoom_min", "get_pinch_zoom_min");

	ClassDB::bind_method(D_METHOD("set_pinch_zoom_max", "max"), &VertexMobileSettings::set_pinch_zoom_max);
	ClassDB::bind_method(D_METHOD("get_pinch_zoom_max"), &VertexMobileSettings::get_pinch_zoom_max);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pinch_zoom_max", PROPERTY_HINT_RANGE, "1,8,0.1"), "set_pinch_zoom_max", "get_pinch_zoom_max");

	ClassDB::bind_method(D_METHOD("set_gesture_pan_threshold", "threshold"), &VertexMobileSettings::set_gesture_pan_threshold);
	ClassDB::bind_method(D_METHOD("get_gesture_pan_threshold"), &VertexMobileSettings::get_gesture_pan_threshold);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gesture_pan_threshold", PROPERTY_HINT_RANGE, "1,32,1,suffix:dp"), "set_gesture_pan_threshold", "get_gesture_pan_threshold");

	ClassDB::bind_method(D_METHOD("set_compact_mode", "enabled"), &VertexMobileSettings::set_compact_mode);
	ClassDB::bind_method(D_METHOD("get_compact_mode"), &VertexMobileSettings::get_compact_mode);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "compact_mode"), "set_compact_mode", "get_compact_mode");

	ClassDB::bind_method(D_METHOD("set_panels_collapsed", "collapsed"), &VertexMobileSettings::set_panels_collapsed);
	ClassDB::bind_method(D_METHOD("get_panels_collapsed"), &VertexMobileSettings::get_panels_collapsed);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "panels_collapsed"), "set_panels_collapsed", "get_panels_collapsed");

	ClassDB::bind_method(D_METHOD("set_virtual_keyboard_assist", "enabled"), &VertexMobileSettings::set_virtual_keyboard_assist);
	ClassDB::bind_method(D_METHOD("get_virtual_keyboard_assist"), &VertexMobileSettings::get_virtual_keyboard_assist);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "virtual_keyboard_assist"), "set_virtual_keyboard_assist", "get_virtual_keyboard_assist");

	ClassDB::bind_method(D_METHOD("set_screen_density_dp", "dp"), &VertexMobileSettings::set_screen_density_dp);
	ClassDB::bind_method(D_METHOD("get_screen_density_dp"), &VertexMobileSettings::get_screen_density_dp);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "screen_density_dp", PROPERTY_HINT_RANGE, "0,640,1,suffix:dp"), "set_screen_density_dp", "get_screen_density_dp");

	ClassDB::bind_method(D_METHOD("to_dictionary"), &VertexMobileSettings::to_dictionary);
	ClassDB::bind_method(D_METHOD("from_dictionary", "dict"), &VertexMobileSettings::from_dictionary);
}
