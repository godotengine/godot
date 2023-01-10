/**************************************************************************/
/*  upnp_device.cpp                                                       */
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

#include "upnp_device.h"

#include "upnp.h"

#include <upnpcommands.h>

String UPNPDevice::query_external_address() const {
	ERR_FAIL_COND_V(!is_valid_gateway(), "");

	char addr[16];
	int i = UPNP_GetExternalIPAddress(
			igd_control_url.utf8().get_data(),
			igd_service_type.utf8().get_data(),
			(char *)&addr);

	ERR_FAIL_COND_V(i != UPNPCOMMAND_SUCCESS, "");

	return String(addr);
}

int UPNPDevice::add_port_mapping(int port, int port_internal, String desc, String proto, int duration) const {
	ERR_FAIL_COND_V(!is_valid_gateway(), UPNP::UPNP_RESULT_INVALID_GATEWAY);
	ERR_FAIL_COND_V(port < 1 || port > 65535, UPNP::UPNP_RESULT_INVALID_PORT);
	ERR_FAIL_COND_V(port_internal < 0 || port_internal > 65535, UPNP::UPNP_RESULT_INVALID_PORT); // Needs to allow 0 because 0 signifies "use external port as internal port"
	ERR_FAIL_COND_V(proto != "UDP" && proto != "TCP", UPNP::UPNP_RESULT_INVALID_PROTOCOL);
	ERR_FAIL_COND_V(duration < 0, UPNP::UPNP_RESULT_INVALID_DURATION);

	if (port_internal < 1) {
		port_internal = port;
	}

	int i = UPNP_AddPortMapping(
			igd_control_url.utf8().get_data(),
			igd_service_type.utf8().get_data(),
			itos(port).utf8().get_data(),
			itos(port_internal).utf8().get_data(),
			igd_our_addr.utf8().get_data(),
			desc.empty() ? nullptr : desc.utf8().get_data(),
			proto.utf8().get_data(),
			nullptr, // Remote host, always NULL as IGDs don't support it
			duration > 0 ? itos(duration).utf8().get_data() : nullptr);

	ERR_FAIL_COND_V(i != UPNPCOMMAND_SUCCESS, UPNP::upnp_result(i));

	return UPNP::UPNP_RESULT_SUCCESS;
}

int UPNPDevice::delete_port_mapping(int port, String proto) const {
	ERR_FAIL_COND_V(port < 1 || port > 65535, UPNP::UPNP_RESULT_INVALID_PORT);
	ERR_FAIL_COND_V(proto != "UDP" && proto != "TCP", UPNP::UPNP_RESULT_INVALID_PROTOCOL);

	int i = UPNP_DeletePortMapping(
			igd_control_url.utf8().get_data(),
			igd_service_type.utf8().get_data(),
			itos(port).utf8().get_data(),
			proto.utf8().get_data(),
			nullptr // Remote host, always NULL as IGDs don't support it
	);

	ERR_FAIL_COND_V(i != UPNPCOMMAND_SUCCESS, UPNP::upnp_result(i));

	return UPNP::UPNP_RESULT_SUCCESS;
}

void UPNPDevice::set_description_url(const String &url) {
	description_url = url;
}

String UPNPDevice::get_description_url() const {
	return description_url;
}

void UPNPDevice::set_service_type(const String &type) {
	service_type = type;
}

String UPNPDevice::get_service_type() const {
	return service_type;
}

void UPNPDevice::set_igd_control_url(const String &url) {
	igd_control_url = url;
}

String UPNPDevice::get_igd_control_url() const {
	return igd_control_url;
}

void UPNPDevice::set_igd_service_type(const String &type) {
	igd_service_type = type;
}

String UPNPDevice::get_igd_service_type() const {
	return igd_service_type;
}

void UPNPDevice::set_igd_our_addr(const String &addr) {
	igd_our_addr = addr;
}

String UPNPDevice::get_igd_our_addr() const {
	return igd_our_addr;
}

void UPNPDevice::set_igd_status(IGDStatus status) {
	igd_status = status;
}

UPNPDevice::IGDStatus UPNPDevice::get_igd_status() const {
	return igd_status;
}

bool UPNPDevice::is_valid_gateway() const {
	return igd_status == IGD_STATUS_OK;
}

void UPNPDevice::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_valid_gateway"), &UPNPDevice::is_valid_gateway);
	ClassDB::bind_method(D_METHOD("query_external_address"), &UPNPDevice::query_external_address);
	ClassDB::bind_method(D_METHOD("add_port_mapping", "port", "port_internal", "desc", "proto", "duration"), &UPNPDevice::add_port_mapping, DEFVAL(0), DEFVAL(""), DEFVAL("UDP"), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("delete_port_mapping", "port", "proto"), &UPNPDevice::delete_port_mapping, DEFVAL("UDP"));

	ClassDB::bind_method(D_METHOD("set_description_url", "url"), &UPNPDevice::set_description_url);
	ClassDB::bind_method(D_METHOD("get_description_url"), &UPNPDevice::get_description_url);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description_url"), "set_description_url", "get_description_url");

	ClassDB::bind_method(D_METHOD("set_service_type", "type"), &UPNPDevice::set_service_type);
	ClassDB::bind_method(D_METHOD("get_service_type"), &UPNPDevice::get_service_type);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "service_type"), "set_service_type", "get_service_type");

	ClassDB::bind_method(D_METHOD("set_igd_control_url", "url"), &UPNPDevice::set_igd_control_url);
	ClassDB::bind_method(D_METHOD("get_igd_control_url"), &UPNPDevice::get_igd_control_url);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "igd_control_url"), "set_igd_control_url", "get_igd_control_url");

	ClassDB::bind_method(D_METHOD("set_igd_service_type", "type"), &UPNPDevice::set_igd_service_type);
	ClassDB::bind_method(D_METHOD("get_igd_service_type"), &UPNPDevice::get_igd_service_type);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "igd_service_type"), "set_igd_service_type", "get_igd_service_type");

	ClassDB::bind_method(D_METHOD("set_igd_our_addr", "addr"), &UPNPDevice::set_igd_our_addr);
	ClassDB::bind_method(D_METHOD("get_igd_our_addr"), &UPNPDevice::get_igd_our_addr);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "igd_our_addr"), "set_igd_our_addr", "get_igd_our_addr");

	ClassDB::bind_method(D_METHOD("set_igd_status", "status"), &UPNPDevice::set_igd_status);
	ClassDB::bind_method(D_METHOD("get_igd_status"), &UPNPDevice::get_igd_status);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "igd_status", PROPERTY_HINT_ENUM), "set_igd_status", "get_igd_status");

	BIND_ENUM_CONSTANT(IGD_STATUS_OK);
	BIND_ENUM_CONSTANT(IGD_STATUS_HTTP_ERROR);
	BIND_ENUM_CONSTANT(IGD_STATUS_HTTP_EMPTY);
	BIND_ENUM_CONSTANT(IGD_STATUS_NO_URLS);
	BIND_ENUM_CONSTANT(IGD_STATUS_NO_IGD);
	BIND_ENUM_CONSTANT(IGD_STATUS_DISCONNECTED);
	BIND_ENUM_CONSTANT(IGD_STATUS_UNKNOWN_DEVICE);
	BIND_ENUM_CONSTANT(IGD_STATUS_INVALID_CONTROL);
	BIND_ENUM_CONSTANT(IGD_STATUS_MALLOC_ERROR);
	BIND_ENUM_CONSTANT(IGD_STATUS_UNKNOWN_ERROR);
}

UPNPDevice::UPNPDevice() {
	description_url = "";
	service_type = "";
	igd_control_url = "";
	igd_service_type = "";
	igd_our_addr = "";
	igd_status = IGD_STATUS_UNKNOWN_ERROR;
}

UPNPDevice::~UPNPDevice() {
}
