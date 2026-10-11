/**************************************************************************/
/*  debug_variable_store_v2.h                                             */
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
/* "Software"), to deal in the Software without restriction, including  */
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

#ifndef DEBUG_VARIABLE_STORE_V2_H
#define DEBUG_VARIABLE_STORE_V2_H

#include "core/templates/list.h"
#include "core/templates/rb_map.h"
#include "core/variant/variant.h"

class DebugVariableStoreV2 {
public:
	enum ScopeKind {
		SCOPE_LOCALS,
		SCOPE_MEMBERS,
		SCOPE_GLOBALS,
	};

private:
	enum HandleKind {
		HANDLE_VALUE,
		HANDLE_SCOPE,
	};

	struct HandleEntry {
		HandleKind kind = HANDLE_VALUE;
		uint64_t stop_id = 0;
		Variant value;
		int frame = 0;
		ScopeKind scope = SCOPE_LOCALS;
	};

	RBMap<uint64_t, HandleEntry> handles;
	uint64_t active_stop_id = 0;
	uint64_t next_handle = 1;
	int page_size = 100;
	int preview_length = 256;

	uint64_t _store_value(const Variant &p_value);
	String _preview(const Variant &p_value) const;
	Array _describe(const String &p_name, const Variant &p_value);
	int _child_count(const Variant &p_value) const;

public:
	void start_stop(uint64_t p_stop_id, int p_page_size);
	void invalidate();
	uint64_t create_scope_handle(int p_frame, ScopeKind p_scope);
	bool get_scope(uint64_t p_handle, int &r_frame, ScopeKind &r_scope) const;

	Error describe_named(const List<String> &p_names, const List<Variant> &p_values, int p_start, int p_count, Array &r_items, int &r_total, bool &r_has_more);
	Error describe_handle(uint64_t p_handle, int p_start, int p_count, Array &r_items, int &r_total, bool &r_has_more);

	uint64_t get_active_stop_id() const { return active_stop_id; }
	int get_page_size() const { return page_size; }
};

#endif // DEBUG_VARIABLE_STORE_V2_H
