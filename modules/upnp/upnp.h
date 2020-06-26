/*************************************************************************/
/*  upnp.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GODOT_UPNP_H
#define GODOT_UPNP_H

#include "core/reference.h"

#include "upnp_device.h"

#include <miniupnpc/miniupnpc.h>

class UPNP : public Reference {
	GDCLASS(UPNP, Reference);

private:
	String discover_multicast_if;
	int discover_local_port;
	bool discover_ipv6;

	Vector<Ref<UPNPDevice>> devices;

	bool is_common_device(const String &dev) const;
	void add_device_to_list(UPNPDev *dev, UPNPDev *devlist);
	void parse_igd(Ref<UPNPDevice> dev, UPNPDev *devlist);
	char *load_description(const String &url, int *size, int *status_code) const;

protected:
	static void _bind_methods();

public:
	enum UPNPResult {

		UPNP_RESULT_SUCCESS,
		UPNP_RESULT_NOT_AUTHORIZED,
		UPNP_RESULT_PORT_MAPPING_NOT_FOUND,
		UPNP_RESULT_INCONSISTENT_PARAMETERS,
		UPNP_RESULT_NO_SUCH_ENTRY_IN_ARRAY,
		UPNP_RESULT_ACTION_FAILED,
		UPNP_RESULT_SRC_IP_WILDCARD_NOT_PERMITTED,
		UPNP_RESULT_EXT_PORT_WILDCARD_NOT_PERMITTED,
		UPNP_RESULT_INT_PORT_WILDCARD_NOT_PERMITTED,
		UPNP_RESULT_REMOTE_HOST_MUST_BE_WILDCARD,
		UPNP_RESULT_EXT_PORT_MUST_BE_WILDCARD,
		UPNP_RESULT_NO_PORT_MAPS_AVAILABLE,
		UPNP_RESULT_CONFLICT_WITH_OTHER_MECHANISM,
		UPNP_RESULT_CONFLICT_WITH_OTHER_MAPPING,
		UPNP_RESULT_SAME_PORT_VALUES_REQUIRED,
		UPNP_RESULT_ONLY_PERMANENT_LEASE_SUPPORTED,
		UPNP_RESULT_INVALID_GATEWAY,
		UPNP_RESULT_INVALID_PORT,
		UPNP_RESULT_INVALID_PROTOCOL,
		UPNP_RESULT_INVALID_DURATION,
		UPNP_RESULT_INVALID_ARGS,
		UPNP_RESULT_INVALID_RESPONSE,
		UPNP_RESULT_INVALID_PARAM,
		UPNP_RESULT_HTTP_ERROR,
		UPNP_RESULT_SOCKET_ERROR,
		UPNP_RESULT_MEM_ALLOC_ERROR,
		UPNP_RESULT_NO_GATEWAY,
		UPNP_RESULT_NO_DEVICES,
		UPNP_RESULT_UNKNOWN_ERROR,
	};

	static int upnp_result(int in);

	int get_device_count() const;
	Ref<UPNPDevice> get_device(int index) const;
	void add_device(Ref<UPNPDevice> device);
	void set_device(int index, Ref<UPNPDevice> device);
	void remove_device(int index);
	void clear_devices();

	Ref<UPNPDevice> get_gateway() const;

	int discover(int timeout = 2000, int ttl = 2, const String &device_filter = "InternetGatewayDevice");

	String query_external_address() const;

	int add_port_mapping(int port, int port_internal = 0, String desc = "", String proto = "UDP", int duration = 0) const;
	int delete_port_mapping(int port, String proto = "UDP") const;

	void set_discover_multicast_if(const String &m_if);
	String get_discover_multicast_if() const;

	void set_discover_local_port(int port);
	int get_discover_local_port() const;

	void set_discover_ipv6(bool ipv6);
	bool is_discover_ipv6() const;

	UPNP();
	~UPNP();
};

VARIANT_ENUM_CAST(UPNP::UPNPResult)

#endif // GODOT_UPNP_H
