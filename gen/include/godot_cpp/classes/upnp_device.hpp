/**************************************************************************/
/*  upnp_device.hpp                                                       */
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
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class UPNPDevice : public RefCounted {
	GDEXTENSION_CLASS(UPNPDevice, RefCounted)

public:
	enum IGDStatus {
		IGD_STATUS_OK = 0,
		IGD_STATUS_HTTP_ERROR = 1,
		IGD_STATUS_HTTP_EMPTY = 2,
		IGD_STATUS_NO_URLS = 3,
		IGD_STATUS_NO_IGD = 4,
		IGD_STATUS_DISCONNECTED = 5,
		IGD_STATUS_UNKNOWN_DEVICE = 6,
		IGD_STATUS_INVALID_CONTROL = 7,
		IGD_STATUS_MALLOC_ERROR = 8,
		IGD_STATUS_UNKNOWN_ERROR = 9,
	};

	bool is_valid_gateway() const;
	String query_external_address() const;
	int32_t add_port_mapping(int32_t p_port, int32_t p_port_internal = 0, const String &p_desc = String(), const String &p_proto = "UDP", int32_t p_duration = 0) const;
	int32_t delete_port_mapping(int32_t p_port, const String &p_proto = "UDP") const;
	void set_description_url(const String &p_url);
	String get_description_url() const;
	void set_service_type(const String &p_type);
	String get_service_type() const;
	void set_igd_control_url(const String &p_url);
	String get_igd_control_url() const;
	void set_igd_service_type(const String &p_type);
	String get_igd_service_type() const;
	void set_igd_our_addr(const String &p_addr);
	String get_igd_our_addr() const;
	void set_igd_status(UPNPDevice::IGDStatus p_status);
	UPNPDevice::IGDStatus get_igd_status() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(UPNPDevice::IGDStatus);

