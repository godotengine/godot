/**************************************************************************/
/*  vertex_ai_context.h                                                  */
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

// Aggregated context the Vertex AI assistant reasons over. Built from the
// project filesystem (structure, scripts, scenes) plus runtime signals
// (errors, logs, profiler/perf results). Stored as a nested Dictionary so a
// pluggable LLM backend (Callable) can consume it generically.
class VertexAIContext : public RefCounted {
	GDCLASS(VertexAIContext, RefCounted)

private:
	Dictionary data;

public:
	Dictionary get_data() const { return data; }
	void set_data(const Dictionary &p_data) { data = p_data; }

	void set_section(const String &p_key, const Dictionary &p_section) { data[p_key] = p_section; }
	Dictionary get_section(const String &p_key) const { return data.has(p_key) ? Dictionary(data[p_key]) : Dictionary(); }

	void build_from_project(const String &p_root = "res://");
	Dictionary to_dictionary() const { return data; }

protected:
	static void _bind_methods();
};
