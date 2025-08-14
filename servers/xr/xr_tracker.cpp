/**************************************************************************/
/*  xr_tracker.cpp                                                        */
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

#include "xr_tracker.h"

void XRTracker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_tracker_type"), &XRTracker::get_tracker_type);
	ClassDB::bind_method(D_METHOD("set_tracker_type", "type"), &XRTracker::set_tracker_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type"), "set_tracker_type", "get_tracker_type");

	ClassDB::bind_method(D_METHOD("get_tracker_name"), &XRTracker::get_tracker_name);
	ClassDB::bind_method(D_METHOD("set_tracker_name", "name"), &XRTracker::set_tracker_name);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "name"), "set_tracker_name", "get_tracker_name");

	ClassDB::bind_method(D_METHOD("get_tracker_desc"), &XRTracker::get_tracker_desc);
	ClassDB::bind_method(D_METHOD("set_tracker_desc", "description"), &XRTracker::set_tracker_desc);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description"), "set_tracker_desc", "get_tracker_desc");
}

void XRTracker::set_tracker_type(XRServer::TrackerType p_type) {
	type = p_type;
}

XRServer::TrackerType XRTracker::get_tracker_type() const {
	return type;
}

void XRTracker::set_tracker_name(const StringName &p_name) {
	// Note: this should not be changed after the tracker is registered with the XRServer!
	name = p_name;
}

StringName XRTracker::get_tracker_name() const {
	return name;
}

void XRTracker::set_tracker_desc(const String &p_desc) {
	description = p_desc;
}

String XRTracker::get_tracker_desc() const {
	return description;
}
