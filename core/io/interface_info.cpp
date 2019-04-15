/*************************************************************************/
/*  interface_info.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "interface_info.h"

#include <stdio.h>
#include <string.h>

Interface_Info::operator String() const {

	/*
	if (wildcard)
		return "*";

	if (!valid)
		return "";

	if (is_ipv4())
		// IPv4 address mapped to IPv6
		return itos(field8[12]) + "." + itos(field8[13]) + "." + itos(field8[14]) + "." + itos(field8[15]);
	String ret;
	for (int i = 0; i < 8; i++) {
		if (i > 0)
			ret = ret + ":";
		uint16_t num = (field8[i * 2] << 8) + field8[i * 2 + 1];
		ret = ret + String::num_int64(num, 16);
	};
	
	return ret;
	*/

	return name;
}

IP_Address Interface_Info::get_ipv4() const {
	return ipv4;
}

void Interface_Info::set_ipv4(const IP_Address &p_ip) {
	ERR_FAIL_COND(!p_ip.is_valid());

	ipv4 = p_ip;
}

IP_Address Interface_Info::get_ipv6() const {
	return ipv6;
}

void Interface_Info::set_ipv6(const IP_Address &p_ip) {
	ERR_FAIL_COND(!p_ip.is_valid());

	ipv6 = p_ip;
}

void Interface_Info::set_name(const String &p_name) {

	name = p_name;
}
String Interface_Info::get_name() const {

	return name;
}

void Interface_Info::set_name_friendly(const String &p_name) {
	name_friendly = p_name;
}
String Interface_Info::get_name_friendly() const {

	return name_friendly;
}

Interface_Info::Interface_Info(const String &p_name, IP_Address p_ipv4, IP_Address p_ipv6) {
	name = p_name;
	ipv4 = p_ipv4;
	ipv4 = p_ipv6;
}

Interface_Info::Interface_Info(const String &p_name, const String &p_name_friendly, IP_Address p_ipv4, IP_Address p_ipv6) {
	name = p_name;
	name_friendly = p_name_friendly;
	ipv4 = p_ipv4;
	ipv4 = p_ipv6;
}