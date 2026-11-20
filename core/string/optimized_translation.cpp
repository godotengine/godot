/**************************************************************************/
/*  optimized_translation.cpp                                             */
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

#include "optimized_translation.h"
#include "optimized_translation.compat.inc"

#include "core/object/class_db.h"
#include "core/string/plural_rules.h"
#include "core/templates/pair.h"

#ifndef DISABLE_DEPRECATED
extern "C" {
#include <thirdparty/misc/smaz.h>
}
#endif

bool OptimizedTranslation::generate(const Ref<Translation> &p_from) {
	// This method compresses a Translation instance.
#ifdef TOOLS_ENABLED
	ERR_FAIL_COND_V(p_from.is_null(), false);

	List<StringName> keys;
	p_from->get_message_list(&keys);

	int size = Math::larger_prime(keys.size());

	Vector<Vector<Pair<int, CharString>>> buckets;
	Vector<HashMap<uint32_t, int>> table;
	Vector<uint32_t> hfunc_table;

	version = 1;
	table.resize(size);
	hfunc_table.resize(size);
	buckets.resize(size);
	offsets.clear();
	strings.clear();

	int total_str_size = 0;
	for (const StringName &E : keys) {
		int off_start = offsets.size();

		// Hash key.
		{
			CharString cs = E.string().utf8();
			uint32_t h = hash(0, cs.get_data());
			Pair<int, CharString> p;
			p.first = off_start;
			p.second = cs;
			buckets.write[h % size].push_back(p);
		}

		String msgctxt;
		String msgid = E.string();
		int sep = msgid.find_char(0x04);
		if (sep != -1) {
			msgctxt = msgid.substr(0, sep);
			msgid = msgid.substr(sep + 1);
		}

		List<StringName> src_s;
		p_from->get_plural_message_list(msgid, msgctxt, &src_s);
		int str_idx = 0;
		offsets.resize(off_start + 1 + src_s.size() * 2);
		offsets.write[off_start] = src_s.size();

		for (const StringName &str : src_s) {
			CharString cs = str.string().utf8();
			strings.resize(total_str_size + cs.length());
			memcpy(&strings.write[total_str_size], cs.get_data(), cs.length());
			offsets.write[off_start + 1 + str_idx * 2] = total_str_size;
			offsets.write[off_start + 1 + str_idx * 2 + 1] = cs.length();
			total_str_size += cs.length();
			str_idx++;
		}
	}

	int bucket_table_size = 0;

	for (int i = 0; i < size; i++) {
		const Vector<Pair<int, CharString>> &b = buckets[i];
		HashMap<uint32_t, int> &t = table.write[i];

		if (b.is_empty()) {
			continue;
		}

		int d = 1;
		int item = 0;

		while (item < b.size()) {
			uint32_t slot = hash(d, b[item].second.get_data());
			if (t.has(slot)) {
				item = 0;
				d++;
				t.clear();
			} else {
				t[slot] = b[item].first;
				item++;
			}
		}

		hfunc_table.write[i] = d;
		bucket_table_size += 2 + b.size() * 4;
	}

	ERR_FAIL_COND_V(bucket_table_size == 0, false);

	hash_table.resize(size);
	bucket_table.resize(bucket_table_size);

	int *htwb = hash_table.ptrw();
	int *btwb = bucket_table.ptrw();

	uint32_t *htw = (uint32_t *)&htwb[0];
	uint32_t *btw = (uint32_t *)&btwb[0];

	int btindex = 0;

	for (int i = 0; i < size; i++) {
		const HashMap<uint32_t, int> &t = table[i];
		if (t.is_empty()) {
			htw[i] = 0xFFFFFFFF; //nothing
			continue;
		}

		htw[i] = btindex;
		btw[btindex++] = t.size();
		btw[btindex++] = hfunc_table[i];

		for (const KeyValue<uint32_t, int> &E : t) {
			btw[btindex++] = E.key;
			btw[btindex++] = E.value;
			btw[btindex++] = 0;
			btw[btindex++] = 0;
		}
	}

	ERR_FAIL_COND_V(btindex != bucket_table_size, false);
	set_locale(p_from->get_locale());
	set_plural_rules_override(p_from->get_plural_rules_override());

	return true;
#else
	return false;
#endif
}

bool OptimizedTranslation::_set(const StringName &p_name, const Variant &p_value) {
	String prop_name = p_name.string();
	if (prop_name == "version") {
		version = p_value;
	} else if (prop_name == "hash_table") {
		hash_table = p_value;
	} else if (prop_name == "bucket_table") {
		bucket_table = p_value;
	} else if (prop_name == "offset_table") {
		offsets = p_value;
	} else if (prop_name == "strings") {
		strings = p_value;
	} else if (prop_name == "load_from") {
		generate(p_value);
	} else {
		return false;
	}

	return true;
}

bool OptimizedTranslation::_get(const StringName &p_name, Variant &r_ret) const {
	String prop_name = p_name.string();
	if (prop_name == "version") {
		r_ret = version;
	} else if (prop_name == "hash_table") {
		r_ret = hash_table;
	} else if (prop_name == "bucket_table") {
		r_ret = bucket_table;
	} else if (prop_name == "offset_table") {
		r_ret = offsets;
	} else if (prop_name == "strings") {
		r_ret = strings;
	} else {
		return false;
	}

	return true;
}

const OptimizedTranslation::Bucket *OptimizedTranslation::_get_bucket(uint32_t p_offset) const {
	const uint64_t bucket_table_size = bucket_table.size();
	ERR_FAIL_COND_V_MSG((uint64_t)p_offset + 2 > bucket_table_size, nullptr, "Invalid bucket offset, corrupt translation.");

	const Bucket *bucket = (const Bucket *)&bucket_table.ptr()[p_offset];
	ERR_FAIL_COND_V_MSG(bucket->size < 0 || (uint64_t)p_offset + 2 + (uint64_t)bucket->size * 4 > bucket_table_size, nullptr, "Invalid bucket size, corrupt translation.");
	return bucket;
}

#ifndef DISABLE_DEPRECATED
String OptimizedTranslation::_get_string_v0(const Bucket::Elem &p_elem) const {
	ERR_FAIL_COND_V_MSG((uint64_t)p_elem.offset + p_elem.comp_size > (uint64_t)strings.size(), String(), "Invalid string offset, corrupt translation.");
	const char *sptr = (const char *)strings.ptr();

	if (p_elem.comp_size == p_elem.uncomp_size) {
		return String::utf8(&sptr[p_elem.offset], p_elem.uncomp_size);
	}

	ERR_FAIL_COND_V_MSG(p_elem.uncomp_size > INT32_MAX, String(), "Invalid string size, corrupt translation.");
	CharString uncomp;
	Error err = uncomp.resize_uninitialized(p_elem.uncomp_size);
	ERR_FAIL_COND_V(err != OK, String());
	int len = smaz_decompress(&sptr[p_elem.offset], p_elem.comp_size, uncomp.ptrw(), p_elem.uncomp_size);
	ERR_FAIL_COND_V_MSG(len > (int)p_elem.uncomp_size, String(), "Invalid compressed string, corrupt translation.");
	return String::utf8(uncomp.get_data(), len);
}
#endif

String OptimizedTranslation::_get_string(const Bucket::Elem &p_elem, int p_n) const {
	ERR_FAIL_COND_V_MSG((uint64_t)p_elem.offset + 1 > (uint64_t)offsets.size(), String(), "Invalid offset, corrupt translation.");
	int count = offsets[p_elem.offset];
	ERR_FAIL_COND_V_MSG((uint64_t)p_elem.offset + 1 + count * 2 > (uint64_t)offsets.size(), String(), "Invalid string count, corrupt translation.");

	int index = 0;
	if (p_n > 0) {
		index = _get_plural_rules()->evaluate(p_n);
	}
	ERR_FAIL_INDEX_V_MSG(index, count, String(), "Invalid plural index or number of plural translations.");

	const char *sptr = (const char *)strings.ptr();
	int offset = offsets[p_elem.offset + 1 + index * 2];
	int len = offsets[p_elem.offset + 1 + index * 2 + 1];
	ERR_FAIL_COND_V_MSG((uint64_t)offset + len > (uint64_t)strings.size(), String(), "Invalid string offset, corrupt translation.");

	return String::utf8(&sptr[offset], len);
}

Vector<String> OptimizedTranslation::_get_strings(const Bucket::Elem &p_elem) const {
	ERR_FAIL_COND_V_MSG((uint64_t)p_elem.offset + 1 > (uint64_t)offsets.size(), Vector<String>(), "Invalid offset, corrupt translation.");
	int count = offsets[p_elem.offset];
	ERR_FAIL_COND_V_MSG((uint64_t)p_elem.offset + 1 + count * 2 > (uint64_t)offsets.size(), Vector<String>(), "Invalid string count, corrupt translation.");

	const char *sptr = (const char *)strings.ptr();

	Vector<String> ret;
	for (int i = 0; i < count; i++) {
		int offset = offsets[p_elem.offset + 1 + i * 2];
		int len = offsets[p_elem.offset + 1 + i * 2 + 1];
		ERR_FAIL_COND_V_MSG((uint64_t)offset + len > (uint64_t)strings.size(), Vector<String>(), "Invalid string offset, corrupt translation.");
		ret.push_back(String::utf8(&sptr[offset], len));
	}

	return ret;
}

StringName OptimizedTranslation::get_message(const StringName &p_src_text, const StringName &p_context) const {
	int htsize = hash_table.size();

	if (htsize == 0) {
		return StringName();
	}

	String key;
#ifdef DISABLE_DEPRECATED
	ERR_FAIL_COND_V_MSG(version != 1, StringName(), vformat("Unsupported data version %d.", version));
#else
	ERR_FAIL_COND_V_MSG(version != 1 && version != 0, StringName(), vformat("Unsupported data version %d.", version));
	if (version == 0) {
		key = p_src_text.string(); // Context not supported.
	}
#endif
	if (version == 1) {
		if (!p_context.is_empty()) {
			key = vformat("%s\x04%s", p_context, p_src_text);
		} else {
			key = p_src_text.string();
		}
	}
	CharString str = key.utf8();
	uint32_t h = hash(0, str.get_data());

	const int *htr = hash_table.ptr();
	const uint32_t *htptr = (const uint32_t *)&htr[0];

	uint32_t p = htptr[h % htsize];

	if (p == 0xFFFFFFFF) {
		return StringName(); //nothing
	}

	const Bucket *bucket = _get_bucket(p);
	if (unlikely(!bucket)) {
		return StringName(); // Helper already prints an error.
	}

	h = hash(bucket->func, str.get_data());

	int idx = -1;

	for (int i = 0; i < bucket->size; i++) {
		if (bucket->elem[i].key == h) {
			idx = i;
			break;
		}
	}

	if (idx == -1) {
		return StringName();
	}

#ifndef DISABLE_DEPRECATED
	if (version == 0) {
		return _get_string_v0(bucket->elem[idx]);
	}
#endif
	if (version == 1) {
		return _get_string(bucket->elem[idx], 0);
	}
	return StringName();
}

Vector<String> OptimizedTranslation::get_translated_message_list() const {
#ifdef DISABLE_DEPRECATED
	ERR_FAIL_COND_V_MSG(version != 1, Vector<String>(), vformat("Unsupported data version %d.", version));
#else
	ERR_FAIL_COND_V_MSG(version != 1 && version != 0, Vector<String>(), vformat("Unsupported data version %d.", version));
#endif
	Vector<String> msgs;

	const int *htr = hash_table.ptr();
	const uint32_t *htptr = (const uint32_t *)&htr[0];

	for (int i = 0; i < hash_table.size(); i++) {
		uint32_t p = htptr[i];
		if (p != 0xFFFFFFFF) {
			const Bucket *bucket = _get_bucket(p);
			if (unlikely(!bucket)) {
				return Vector<String>(); // Helper already prints an error.
			}
			for (int j = 0; j < bucket->size; j++) {
#ifndef DISABLE_DEPRECATED
				if (version == 0) {
					msgs.push_back(_get_string_v0(bucket->elem[j]));
				}
#endif
				if (version == 1) {
					msgs.append_array(_get_strings(bucket->elem[j]));
				}
			}
		}
	}
	return msgs;
}

StringName OptimizedTranslation::get_plural_message(const StringName &p_src_text, const StringName &p_plural_text, int p_n, const StringName &p_context) const {
#ifdef DISABLE_DEPRECATED
	ERR_FAIL_COND_V_MSG(version != 1, StringName(), vformat("Unsupported data version %d.", version));
#else
	ERR_FAIL_COND_V_MSG(version != 1 && version != 0, StringName(), vformat("Unsupported data version %d.", version));
	if (version == 0) {
		return get_message(p_src_text, p_context);
	}
#endif
	int htsize = hash_table.size();

	if (htsize == 0) {
		return StringName();
	}

	String key;
	if (!p_context.is_empty()) {
		key = vformat("%s\x04%s", p_context, p_src_text);
	} else {
		key = p_src_text.string();
	}
	CharString str = key.utf8();
	uint32_t h = hash(0, str.get_data());

	const int *htr = hash_table.ptr();
	const uint32_t *htptr = (const uint32_t *)&htr[0];

	uint32_t p = htptr[h % htsize];

	if (p == 0xFFFFFFFF) {
		return StringName(); //nothing
	}

	const Bucket *bucket = _get_bucket(p);
	if (unlikely(!bucket)) {
		return StringName(); // Helper already prints an error.
	}

	h = hash(bucket->func, str.get_data());

	int idx = -1;

	for (int i = 0; i < bucket->size; i++) {
		if (bucket->elem[i].key == h) {
			idx = i;
			break;
		}
	}

	if (idx == -1) {
		return StringName();
	}

	return _get_string(bucket->elem[idx], p_n);
}

Vector<String> OptimizedTranslation::_get_message_list() const {
	WARN_PRINT_ONCE("OptimizedTranslation does not store the message texts to be translated.");
	return {};
}

void OptimizedTranslation::get_message_list(List<StringName> *r_messages) const {
	WARN_PRINT_ONCE("OptimizedTranslation does not store the message texts to be translated.");
}

int OptimizedTranslation::get_message_count() const {
	WARN_PRINT_ONCE("OptimizedTranslation does not store the message texts to be translated.");
	return 0;
}

void OptimizedTranslation::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::INT, "version", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	p_list->push_back(PropertyInfo(Variant::PACKED_INT32_ARRAY, "hash_table", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	p_list->push_back(PropertyInfo(Variant::PACKED_INT32_ARRAY, "bucket_table", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	p_list->push_back(PropertyInfo(Variant::PACKED_INT32_ARRAY, "offset_table", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	p_list->push_back(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "strings", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "load_from", PROPERTY_HINT_RESOURCE_TYPE, Translation::get_class_static(), PROPERTY_USAGE_EDITOR));
}

void OptimizedTranslation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("generate", "from"), &OptimizedTranslation::generate);
}
