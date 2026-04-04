/**************************************************************************/
/*  web_xr_interface.hpp                                                  */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/xr_interface.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class XRControllerTracker;

class WebXRInterface : public XRInterface {
	GDEXTENSION_CLASS(WebXRInterface, XRInterface)

public:
	enum TargetRayMode {
		TARGET_RAY_MODE_UNKNOWN = 0,
		TARGET_RAY_MODE_GAZE = 1,
		TARGET_RAY_MODE_TRACKED_POINTER = 2,
		TARGET_RAY_MODE_SCREEN = 3,
	};

	void is_session_supported(const String &p_session_mode);
	void set_session_mode(const String &p_session_mode);
	String get_session_mode() const;
	void set_required_features(const String &p_required_features);
	String get_required_features() const;
	void set_optional_features(const String &p_optional_features);
	String get_optional_features() const;
	String get_reference_space_type() const;
	String get_enabled_features() const;
	void set_requested_reference_space_types(const String &p_requested_reference_space_types);
	String get_requested_reference_space_types() const;
	bool is_input_source_active(int32_t p_input_source_id) const;
	Ref<XRControllerTracker> get_input_source_tracker(int32_t p_input_source_id) const;
	WebXRInterface::TargetRayMode get_input_source_target_ray_mode(int32_t p_input_source_id) const;
	String get_visibility_state() const;
	float get_display_refresh_rate() const;
	void set_display_refresh_rate(float p_refresh_rate);
	Array get_available_display_refresh_rates() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		XRInterface::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(WebXRInterface::TargetRayMode);

