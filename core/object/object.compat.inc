/**************************************************************************/
/*  object.compat.inc                                                     */
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

#ifndef DISABLE_DEPRECATED

#include "core/object/class_db.h"
#include "core/variant/typed_array.h"

TypedArray<Dictionary> Object::_get_property_list_bind_compat_82198() const {
	List<PropertyInfo> lpi;
	get_property_list(&lpi);
	return convert_property_list(&lpi);
}

TypedArray<Dictionary> Object::_get_method_list_bind_compat_82198() const {
	List<MethodInfo> ml;
	get_method_list(&ml);
	TypedArray<Dictionary> ret;

	for (List<MethodInfo>::Element *E = ml.front(); E; E = E->next()) {
		Dictionary d = E->get();
		//va.push_back(d);
		ret.push_back(d);
	}

	return ret;
}

void Object::_add_user_signal_compat_82198(const String &p_name, const Array &p_args) {
	// this version of add_user_signal is meant to be used from scripts or external apis
	// without access to ADD_SIGNAL in bind_methods
	// added events are per instance, as opposed to the other ones, which are global

	MethodInfo mi;
	mi.name = p_name;

	for (int i = 0; i < p_args.size(); i++) {
		Dictionary d = p_args[i];
		PropertyInfo param;

		if (d.has("name")) {
			param.name = d["name"];
		}
		if (d.has("type")) {
			param.type = (Variant::Type)(int)d["type"];
		}

		mi.arguments.push_back(param);
	}

	add_user_signal(mi);
}

TypedArray<Dictionary> Object::_get_signal_list_compat_82198() const {
	List<MethodInfo> signal_list;
	get_signal_list(&signal_list);

	TypedArray<Dictionary> ret;
	for (const MethodInfo &E : signal_list) {
		ret.push_back(Dictionary(E));
	}

	return ret;
}

TypedArray<Dictionary> Object::_get_signal_connection_list_compat_82198(const StringName &p_signal) const {
	List<Connection> conns;
	get_all_signal_connections(&conns);

	TypedArray<Dictionary> ret;

	for (const Connection &c : conns) {
		if (c.signal.get_name() == p_signal) {
			ret.push_back(c);
		}
	}

	return ret;
}

TypedArray<Dictionary> Object::_get_incoming_connections_compat_82198() const {
	TypedArray<Dictionary> ret;
	int connections_amount = connections.size();
	for (int idx_conn = 0; idx_conn < connections_amount; idx_conn++) {
		ret.push_back(connections[idx_conn]);
	}

	return ret;
}

void Object::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("tr", "message", "context"), &Object::tr, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("tr_n", "message", "plural_message", "n", "context"), &Object::tr_n, DEFVAL(""));

	ClassDB::bind_compatibility_method(D_METHOD("get_property_list"), &Object::_get_property_list_bind_compat_82198);
	ClassDB::bind_compatibility_method(D_METHOD("get_method_list"), &Object::_get_method_list_bind_compat_82198);
	ClassDB::bind_compatibility_method(D_METHOD("add_user_signal"), &Object::_add_user_signal_compat_82198);
	ClassDB::bind_compatibility_method(D_METHOD("get_signal_list"), &Object::_get_signal_list_compat_82198);
	ClassDB::bind_compatibility_method(D_METHOD("get_signal_connection_list", "signal"), &Object::_get_signal_connection_list_compat_82198);
	ClassDB::bind_compatibility_method(D_METHOD("get_incoming_connections"), &Object::_get_incoming_connections_compat_82198);
}

#endif
