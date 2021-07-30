/*************************************************************************/
/*  plist.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

// Property list file format (application/x-plist) parser, property list ASN-1 serialization.

#ifndef PLIST_H
#define PLIST_H

#include "core/crypto/crypto_core.h"
#include "core/io/file_access.h"
#include "modules/modules_enabled.gen.h" // For regex.

#ifdef MODULE_REGEX_ENABLED

class PListNode;

class PList : public RefCounted {
	friend class PListNode;

public:
	enum PLNodeType {
		PL_NODE_TYPE_NIL,
		PL_NODE_TYPE_STRING,
		PL_NODE_TYPE_ARRAY,
		PL_NODE_TYPE_DICT,
		PL_NODE_TYPE_BOOLEAN,
		PL_NODE_TYPE_INTEGER,
		PL_NODE_TYPE_REAL,
		PL_NODE_TYPE_DATA,
		PL_NODE_TYPE_DATE,
	};

private:
	Ref<PListNode> root;

public:
	PList();
	PList(const String &p_string);

	bool load_file(const String &p_filename);
	bool load_string(const String &p_string);

	PackedByteArray save_asn1() const;
	String save_text() const;

	Ref<PListNode> get_root();
};

/*************************************************************************/

class PListNode : public RefCounted {
	static int _asn1_size_len(uint8_t p_len_octets);

public:
	PList::PLNodeType data_type = PList::PLNodeType::PL_NODE_TYPE_NIL;

	CharString data_string;
	Vector<Ref<PListNode>> data_array;
	Map<String, Ref<PListNode>> data_dict;
	union {
		int32_t data_int;
		bool data_bool;
		float data_real;
	};

	static Ref<PListNode> new_array();
	static Ref<PListNode> new_dict();
	static Ref<PListNode> new_string(const String &p_string);
	static Ref<PListNode> new_data(const String &p_string);
	static Ref<PListNode> new_date(const String &p_string);
	static Ref<PListNode> new_bool(bool p_bool);
	static Ref<PListNode> new_int(int32_t p_int);
	static Ref<PListNode> new_real(float p_real);

	bool push_subnode(const Ref<PListNode> &p_node, const String &p_key = "");

	size_t get_asn1_size(uint8_t p_len_octets) const;

	void store_asn1_size(PackedByteArray &p_stream, uint8_t p_len_octets) const;
	bool store_asn1(PackedByteArray &p_stream, uint8_t p_len_octets) const;
	void store_text(String &p_stream, uint8_t p_indent) const;

	PListNode() {}
	~PListNode() {}
};

#endif // MODULE_REGEX_ENABLED

#endif // PLIST_H
