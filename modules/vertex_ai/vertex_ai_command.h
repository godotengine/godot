/**************************************************************************/
/*  vertex_ai_command.h                                                  */
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

#pragma once

#include "core/object/ref_counted.h"
#include "core/variant/callable.h"

// A named, documented action the Vertex AI assistant can perform. Destructive
// commands set [member is_destructive] and are only executed after an
// explicit confirmation. The handler is a Callable so backends can be
// implemented in C++ or scripting.
class VertexAICommand : public RefCounted {
	GDCLASS(VertexAICommand, RefCounted)

private:
	String name;
	String description;
	bool is_destructive = false;
	Callable handler;

public:
	void set_name(const String &p_name) { name = p_name; }
	String get_name() const { return name; }
	void set_description(const String &p_description) { description = p_description; }
	String get_description() const { return description; }
	void set_is_destructive(bool p_d) { is_destructive = p_d; }
	bool get_is_destructive() const { return is_destructive; }
	void set_handler(const Callable &p_h) { handler = p_h; }
	Callable get_handler() const { return handler; }

	Dictionary to_dictionary() const;

protected:
	static void _bind_methods();
};
