/**************************************************************************/
/*  xml_parser.hpp                                                        */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PackedByteArray;

class XMLParser : public RefCounted {
	GDEXTENSION_CLASS(XMLParser, RefCounted)

public:
	enum NodeType {
		NODE_NONE = 0,
		NODE_ELEMENT = 1,
		NODE_ELEMENT_END = 2,
		NODE_TEXT = 3,
		NODE_COMMENT = 4,
		NODE_CDATA = 5,
		NODE_UNKNOWN = 6,
	};

	Error read();
	XMLParser::NodeType get_node_type();
	String get_node_name() const;
	String get_node_data() const;
	uint64_t get_node_offset() const;
	int32_t get_attribute_count() const;
	String get_attribute_name(int32_t p_idx) const;
	String get_attribute_value(int32_t p_idx) const;
	bool has_attribute(const String &p_name) const;
	String get_named_attribute_value(const String &p_name) const;
	String get_named_attribute_value_safe(const String &p_name) const;
	bool is_empty() const;
	int32_t get_current_line() const;
	void skip_section();
	Error seek(uint64_t p_position);
	Error open(const String &p_file);
	Error open_buffer(const PackedByteArray &p_buffer);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
	Error _open_buffer(const uint8_t *p_buffer, size_t p_size);
};

} // namespace godot

VARIANT_ENUM_CAST(XMLParser::NodeType);

