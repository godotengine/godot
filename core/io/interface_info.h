/*************************************************************************/
/*  ip_address.h                                                         */
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

#ifndef INTERFACE_INFO_H
#define INTERFACE_INFO_H

#include "core/io/ip_address.h"
#include "core/ustring.h"

struct Interface_Info {

private:
	String name;
	String name_friendly;
	IP_Address ipv4;
	IP_Address ipv6;

protected:

public:
	void set_name(const String &p_adapter);
	String get_name() const;

	void set_name_friendly(const String &p_adapter);
	String get_name_friendly() const;

	void set_ipv4(const IP_Address &p_ip);
	IP_Address get_ipv4() const;
	void set_ipv6(const IP_Address &p_ip);
	IP_Address get_ipv6() const;

	operator String() const;
	Interface_Info(const String &p_name, IP_Address p_ipv4, IP_Address p_ipv6);
	Interface_Info(const String &p_name, const String &p_name_friendly, IP_Address p_ipv4, IP_Address p_ipv6);
	Interface_Info() { }
};

#endif // IP_ADDRESS_H
