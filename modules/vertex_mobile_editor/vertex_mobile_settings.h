/**************************************************************************/
/*  vertex_mobile_settings.h                                              */
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

#pragma once

#include "core/object/ref_counted.h"

// Touch-friendly editor settings shared by the Vertex mobile editor layout.
// Centralizes sizes/gesture thresholds so the desktop UI is not merely
// shrunk but re-laid-out for small touchscreens and virtual keyboards.
class VertexMobileSettings : public RefCounted {
	GDCLASS(VertexMobileSettings, RefCounted)

private:
	float touch_target_size = 48.0f; // dp; meets minimum touch-target guidance.
	float compact_toolbar_height = 44.0f;
	float panel_toggle_size = 40.0f;
	float pinch_zoom_min = 0.25f;
	float pinch_zoom_max = 4.0f;
	float gesture_pan_threshold = 8.0f;
	bool compact_mode = false;
	bool panels_collapsed = false;
	bool virtual_keyboard_assist = true;
	int screen_density_dp = 0; // 0 = auto.

public:
	float get_touch_target_size() const { return touch_target_size; }
	void set_touch_target_size(float p_v) { touch_target_size = p_v; }
	float get_compact_toolbar_height() const { return compact_toolbar_height; }
	void set_compact_toolbar_height(float p_v) { compact_toolbar_height = p_v; }
	float get_panel_toggle_size() const { return panel_toggle_size; }
	void set_panel_toggle_size(float p_v) { panel_toggle_size = p_v; }
	float get_pinch_zoom_min() const { return pinch_zoom_min; }
	void set_pinch_zoom_min(float p_v) { pinch_zoom_min = p_v; }
	float get_pinch_zoom_max() const { return pinch_zoom_max; }
	void set_pinch_zoom_max(float p_v) { pinch_zoom_max = p_v; }
	float get_gesture_pan_threshold() const { return gesture_pan_threshold; }
	void set_gesture_pan_threshold(float p_v) { gesture_pan_threshold = p_v; }
	bool get_compact_mode() const { return compact_mode; }
	void set_compact_mode(bool p_v) { compact_mode = p_v; }
	bool get_panels_collapsed() const { return panels_collapsed; }
	void set_panels_collapsed(bool p_v) { panels_collapsed = p_v; }
	bool get_virtual_keyboard_assist() const { return virtual_keyboard_assist; }
	void set_virtual_keyboard_assist(bool p_v) { virtual_keyboard_assist = p_v; }
	int get_screen_density_dp() const { return screen_density_dp; }
	void set_screen_density_dp(int p_v) { screen_density_dp = p_v; }

	Dictionary to_dictionary() const;
	void from_dictionary(const Dictionary &p_dict);

protected:
	static void _bind_methods();
};
