/**************************************************************************/
/*  connection.h                                                          */
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

#ifndef CONNECTION_H
#define CONNECTION_H

#include "core/object/method_info.h"
#include "core/templates/hash_map.h"
#include "core/variant/struct_generator.h"
#include "core/variant/type_info.h"
#include "core/variant/variant.h"

enum ConnectFlags {
	CONNECT_DEFERRED = 1,
	CONNECT_PERSIST = 2, // hint for scene to save this connection
	CONNECT_ONE_SHOT = 4,
	CONNECT_REFERENCE_COUNTED = 8,
	CONNECT_INHERITED = 16, // Used in editor builds.
};

struct Connection {
	STRUCT_DECLARE(Connection);
	STRUCT_MEMBER_PRIMITIVE(::Signal, signal, ::Signal());
	STRUCT_MEMBER_PRIMITIVE(Callable, callable, Callable());
	STRUCT_MEMBER_PRIMITIVE(uint32_t, flags, 0);
	STRUCT_LAYOUT_OWNER(Object, Connection, struct signal, struct callable, struct flags);

	bool operator<(const Connection &p_conn) const;

	operator Variant() const;

	Connection() {}
	Connection(const Variant &p_variant);
};

struct SignalData {
	struct Slot {
		int reference_count = 0;
		Connection conn;
		List<Connection>::Element *cE = nullptr;
	};

	MethodInfo user;
	HashMap<Callable, Slot, HashableHasher<Callable>> slot_map;
};

#endif //CONNECTION_H
