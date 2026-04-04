/**************************************************************************/
/*  ip.hpp                                                                */
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

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class IP : public Object {
	GDEXTENSION_CLASS(IP, Object)

	static IP *singleton;

public:
	enum ResolverStatus {
		RESOLVER_STATUS_NONE = 0,
		RESOLVER_STATUS_WAITING = 1,
		RESOLVER_STATUS_DONE = 2,
		RESOLVER_STATUS_ERROR = 3,
	};

	enum Type {
		TYPE_NONE = 0,
		TYPE_IPV4 = 1,
		TYPE_IPV6 = 2,
		TYPE_ANY = 3,
	};

	static const int RESOLVER_MAX_QUERIES = 256;
	static const int RESOLVER_INVALID_ID = -1;

	static IP *get_singleton();

	String resolve_hostname(const String &p_host, IP::Type p_ip_type = (IP::Type)3);
	PackedStringArray resolve_hostname_addresses(const String &p_host, IP::Type p_ip_type = (IP::Type)3);
	int32_t resolve_hostname_queue_item(const String &p_host, IP::Type p_ip_type = (IP::Type)3);
	IP::ResolverStatus get_resolve_item_status(int32_t p_id) const;
	String get_resolve_item_address(int32_t p_id) const;
	Array get_resolve_item_addresses(int32_t p_id) const;
	void erase_resolve_item(int32_t p_id);
	PackedStringArray get_local_addresses() const;
	TypedArray<Dictionary> get_local_interfaces() const;
	void clear_cache(const String &p_hostname = String());

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~IP();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(IP::ResolverStatus);
VARIANT_ENUM_CAST(IP::Type);

