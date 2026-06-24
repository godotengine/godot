/**************************************************************************/
/*  uwp_host.cpp                                                          */
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
/* included in all copies or substantial portions of the Software.       */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifdef GODOT_UWP_EMBED_ENABLED

#include "uwp_host.h"

#include "godot_uwp_embed.h"

#include "core/io/json.h"

// Defined in godot_uwp_embed.cpp.
namespace GodotUwpEmbedBus {
extern GodotUwpHostMsgCallback host_msg_callback;
extern String pending_call_return;
extern bool pending_call_return_set;
} //namespace GodotUwpEmbedBus

UWPHost *UWPHost::singleton = nullptr;

void UWPHost::_bind_methods() {
	ClassDB::bind_method(D_METHOD("send_to_host", "method", "args"), &UWPHost::send_to_host, DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("register_handler", "method", "callable"), &UWPHost::register_handler);
	ClassDB::bind_method(D_METHOD("unregister_handler", "method"), &UWPHost::unregister_handler);
	ClassDB::bind_method(D_METHOD("has_host"), &UWPHost::has_host);
}

Variant UWPHost::send_to_host(const String &p_method, const Variant &p_args) {
	GodotUwpHostMsgCallback cb = GodotUwpEmbedBus::host_msg_callback;
	if (cb == nullptr) {
		return Variant();
	}

	// Always hand the host a JSON ARRAY of arguments ("[]" if none).
	String args_json;
	if (p_args.get_type() == Variant::NIL) {
		args_json = "[]";
	} else if (p_args.get_type() == Variant::ARRAY) {
		args_json = JSON::stringify(p_args);
	} else {
		Array wrapped;
		wrapped.push_back(p_args);
		args_json = JSON::stringify(wrapped);
	}

	GodotUwpEmbedBus::pending_call_return = String();
	GodotUwpEmbedBus::pending_call_return_set = false;

	CharString method_utf8 = p_method.utf8();
	CharString args_utf8 = args_json.utf8();
	cb(method_utf8.get_data(), args_utf8.get_data());

	// The host may have replied synchronously via godot_uwp_set_call_return.
	if (!GodotUwpEmbedBus::pending_call_return_set) {
		return Variant();
	}
	return JSON::parse_string(GodotUwpEmbedBus::pending_call_return);
}

void UWPHost::register_handler(const String &p_method, const Callable &p_callable) {
	ERR_FAIL_COND_MSG(p_method.is_empty(), "Handler method name must not be empty.");
	handlers[p_method] = p_callable;
}

void UWPHost::unregister_handler(const String &p_method) {
	handlers.erase(p_method);
}

bool UWPHost::has_host() const {
	return GodotUwpEmbedBus::host_msg_callback != nullptr;
}

bool UWPHost::call_handler(const String &p_method, const String &p_args_json, String &r_ret_json) {
	r_ret_json = String();

	const Callable *handler = handlers.getptr(p_method);
	if (handler == nullptr || !handler->is_valid()) {
		WARN_PRINT(vformat("UWPHost: no handler registered for '%s'.", p_method));
		return false;
	}

	// Decode the argument payload: a JSON array maps to the callable's
	// argument list; any other JSON value is passed as a single argument.
	Array call_args;
	if (!p_args_json.is_empty()) {
		Variant parsed = JSON::parse_string(p_args_json);
		if (parsed.get_type() == Variant::ARRAY) {
			call_args = parsed;
		} else if (parsed.get_type() != Variant::NIL) {
			call_args.push_back(parsed);
		}
	}

	Variant result = handler->callv(call_args);
	if (result.get_type() != Variant::NIL) {
		r_ret_json = JSON::stringify(result);
	}
	return true;
}

UWPHost::UWPHost() {
	singleton = this;
}

UWPHost::~UWPHost() {
	if (singleton == this) {
		singleton = nullptr;
	}
}

#endif // GODOT_UWP_EMBED_ENABLED
