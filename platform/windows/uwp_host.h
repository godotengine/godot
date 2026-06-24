/**************************************************************************/
/*  uwp_host.h                                                            */
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

#pragma once

#ifdef GODOT_UWP_EMBED_ENABLED

// GDScript-visible singleton for the host<->engine JSON message bus, exposed
// to projects as "UWPHost":
//
//   var host = Engine.get_singleton("UWPHost")
//
//   host.send_to_host(method: String, args) -> Variant
//       Serializes args to JSON, invokes the host message callback on the
//       engine thread, and returns the host's optional JSON reply (parsed).
//   host.register_handler(method: String, callable: Callable)
//       Registers a handler the host can invoke via godot_uwp_call_engine.
//   host.has_host() -> bool
//       True when a host message callback is installed.
//
// All methods are engine-thread affine (GDScript and the host's call_engine
// both already run there).

#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/templates/hash_map.h"
#include "core/variant/callable.h"

class UWPHost : public Object {
	GDCLASS(UWPHost, Object);

	HashMap<String, Callable> handlers;

	static UWPHost *singleton;

protected:
	static void _bind_methods();

public:
	static UWPHost *get_singleton() { return singleton; }

	// GDScript API.
	Variant send_to_host(const String &p_method, const Variant &p_args);
	void register_handler(const String &p_method, const Callable &p_callable);
	void unregister_handler(const String &p_method);
	bool has_host() const;

	// Engine-internal: dispatch a host call to a registered handler.
	// Returns false when no handler is registered for p_method.
	// r_ret_json is empty when the handler returned null.
	bool call_handler(const String &p_method, const String &p_args_json, String &r_ret_json);

	UWPHost();
	~UWPHost();
};

#endif // GODOT_UWP_EMBED_ENABLED
