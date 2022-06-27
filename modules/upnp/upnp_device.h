/*************************************************************************/
/*  upnp_device.h                                                        */
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

#ifndef GODOT_UPNP_DEVICE_H
#define GODOT_UPNP_DEVICE_H

#include "core/object/ref_counted.h"

class UPNPDevice : public RefCounted {
	GDCLASS(UPNPDevice, RefCounted);

public:
	enum IGDStatus {
		IGD_STATUS_OK,
		IGD_STATUS_HTTP_ERROR,
		IGD_STATUS_HTTP_EMPTY,
		IGD_STATUS_NO_URLS,
		IGD_STATUS_NO_IGD,
		IGD_STATUS_DISCONNECTED,
		IGD_STATUS_UNKNOWN_DEVICE,
		IGD_STATUS_INVALID_CONTROL,
		IGD_STATUS_MALLOC_ERROR,
		IGD_STATUS_UNKNOWN_ERROR,
	};

	void set_description_url(const String &url);
	String get_description_url() const;

	void set_service_type(const String &type);
	String get_service_type() const;

	void set_igd_control_url(const String &url);
	String get_igd_control_url() const;

	void set_igd_service_type(const String &type);
	String get_igd_service_type() const;

	void set_igd_our_addr(const String &addr);
	String get_igd_our_addr() const;

	void set_igd_status(IGDStatus status);
	IGDStatus get_igd_status() const;

	bool is_valid_gateway() const;
	String query_external_address() const;
	int add_port_mapping(int port, int port_internal = 0, String desc = "", String proto = "UDP", int duration = 0) const;
	int delete_port_mapping(int port, String proto = "UDP") const;

	UPNPDevice();
	~UPNPDevice();

protected:
	static void _bind_methods();

private:
	String description_url;
	String service_type;
	String igd_control_url;
	String igd_service_type;
	String igd_our_addr;
	IGDStatus igd_status;
};

VARIANT_ENUM_CAST(UPNPDevice::IGDStatus)

#endif // GODOT_UPNP_DEVICE_H
