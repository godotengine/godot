/**************************************************************************/
/*  xr_server.hpp                                                         */
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
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class String;
class StringName;
class XRInterface;
class XRTracker;

class XRServer : public Object {
	GDEXTENSION_CLASS(XRServer, Object)

	static XRServer *singleton;

public:
	enum TrackerType {
		TRACKER_HEAD = 1,
		TRACKER_CONTROLLER = 2,
		TRACKER_BASESTATION = 4,
		TRACKER_ANCHOR = 8,
		TRACKER_HAND = 16,
		TRACKER_BODY = 32,
		TRACKER_FACE = 64,
		TRACKER_ANY_KNOWN = 127,
		TRACKER_UNKNOWN = 128,
		TRACKER_ANY = 255,
	};

	enum RotationMode {
		RESET_FULL_ROTATION = 0,
		RESET_BUT_KEEP_TILT = 1,
		DONT_RESET_ROTATION = 2,
	};

	static XRServer *get_singleton();

	double get_world_scale() const;
	void set_world_scale(double p_scale);
	Transform3D get_world_origin() const;
	void set_world_origin(const Transform3D &p_world_origin);
	Transform3D get_reference_frame() const;
	void clear_reference_frame();
	void center_on_hmd(XRServer::RotationMode p_rotation_mode, bool p_keep_height);
	Transform3D get_hmd_transform();
	void set_camera_locked_to_origin(bool p_enabled);
	bool is_camera_locked_to_origin() const;
	void add_interface(const Ref<XRInterface> &p_interface);
	int32_t get_interface_count() const;
	void remove_interface(const Ref<XRInterface> &p_interface);
	Ref<XRInterface> get_interface(int32_t p_idx) const;
	TypedArray<Dictionary> get_interfaces() const;
	Ref<XRInterface> find_interface(const String &p_name) const;
	void add_tracker(const Ref<XRTracker> &p_tracker);
	void remove_tracker(const Ref<XRTracker> &p_tracker);
	Dictionary get_trackers(int32_t p_tracker_types);
	Ref<XRTracker> get_tracker(const StringName &p_tracker_name) const;
	Ref<XRInterface> get_primary_interface() const;
	void set_primary_interface(const Ref<XRInterface> &p_interface);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~XRServer();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(XRServer::TrackerType);
VARIANT_ENUM_CAST(XRServer::RotationMode);

