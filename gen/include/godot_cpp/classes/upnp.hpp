/**************************************************************************/
/*  upnp.hpp                                                              */
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

class UPNPDevice;

class UPNP : public RefCounted {
	GDEXTENSION_CLASS(UPNP, RefCounted)

public:
	enum UPNPResult {
		UPNP_RESULT_SUCCESS = 0,
		UPNP_RESULT_NOT_AUTHORIZED = 1,
		UPNP_RESULT_PORT_MAPPING_NOT_FOUND = 2,
		UPNP_RESULT_INCONSISTENT_PARAMETERS = 3,
		UPNP_RESULT_NO_SUCH_ENTRY_IN_ARRAY = 4,
		UPNP_RESULT_ACTION_FAILED = 5,
		UPNP_RESULT_SRC_IP_WILDCARD_NOT_PERMITTED = 6,
		UPNP_RESULT_EXT_PORT_WILDCARD_NOT_PERMITTED = 7,
		UPNP_RESULT_INT_PORT_WILDCARD_NOT_PERMITTED = 8,
		UPNP_RESULT_REMOTE_HOST_MUST_BE_WILDCARD = 9,
		UPNP_RESULT_EXT_PORT_MUST_BE_WILDCARD = 10,
		UPNP_RESULT_NO_PORT_MAPS_AVAILABLE = 11,
		UPNP_RESULT_CONFLICT_WITH_OTHER_MECHANISM = 12,
		UPNP_RESULT_CONFLICT_WITH_OTHER_MAPPING = 13,
		UPNP_RESULT_SAME_PORT_VALUES_REQUIRED = 14,
		UPNP_RESULT_ONLY_PERMANENT_LEASE_SUPPORTED = 15,
		UPNP_RESULT_INVALID_GATEWAY = 16,
		UPNP_RESULT_INVALID_PORT = 17,
		UPNP_RESULT_INVALID_PROTOCOL = 18,
		UPNP_RESULT_INVALID_DURATION = 19,
		UPNP_RESULT_INVALID_ARGS = 20,
		UPNP_RESULT_INVALID_RESPONSE = 21,
		UPNP_RESULT_INVALID_PARAM = 22,
		UPNP_RESULT_HTTP_ERROR = 23,
		UPNP_RESULT_SOCKET_ERROR = 24,
		UPNP_RESULT_MEM_ALLOC_ERROR = 25,
		UPNP_RESULT_NO_GATEWAY = 26,
		UPNP_RESULT_NO_DEVICES = 27,
		UPNP_RESULT_UNKNOWN_ERROR = 28,
	};

	int32_t get_device_count() const;
	Ref<UPNPDevice> get_device(int32_t p_index) const;
	void add_device(const Ref<UPNPDevice> &p_device);
	void set_device(int32_t p_index, const Ref<UPNPDevice> &p_device);
	void remove_device(int32_t p_index);
	void clear_devices();
	Ref<UPNPDevice> get_gateway() const;
	int32_t discover(int32_t p_timeout = 2000, int32_t p_ttl = 2, const String &p_device_filter = "InternetGatewayDevice");
	String query_external_address() const;
	int32_t add_port_mapping(int32_t p_port, int32_t p_port_internal = 0, const String &p_desc = String(), const String &p_proto = "UDP", int32_t p_duration = 0) const;
	int32_t delete_port_mapping(int32_t p_port, const String &p_proto = "UDP") const;
	void set_discover_multicast_if(const String &p_m_if);
	String get_discover_multicast_if() const;
	void set_discover_local_port(int32_t p_port);
	int32_t get_discover_local_port() const;
	void set_discover_ipv6(bool p_ipv6);
	bool is_discover_ipv6() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(UPNP::UPNPResult);

