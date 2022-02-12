/*************************************************************************/
/*  dictionary.h                                                         */
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

#ifndef DICTIONARY_H
#define DICTIONARY_H

#include "core/string/ustring.h"
#include "core/templates/list.h"
#include "core/variant/array.h"

class Variant;

struct DictionaryPrivate;

class Dictionary {
	mutable DictionaryPrivate *_p;

	void _ref(const Dictionary &p_from) const;
	void _unref() const;

public:
	void get_key_list(List<Variant> *p_keys) const;
	Variant get_key_at_index(int p_index) const;
	Variant get_value_at_index(int p_index) const;

	Variant &operator[](const Variant &p_key);
	const Variant &operator[](const Variant &p_key) const;

	const Variant *getptr(const Variant &p_key) const;
	Variant *getptr(const Variant &p_key);

	Variant get_valid(const Variant &p_key) const;
	Variant get(const Variant &p_key, const Variant &p_default) const;

	int size() const;
	bool is_empty() const;
	void clear();

	bool has(const Variant &p_key) const;
	bool has_all(const Array &p_keys) const;

	bool erase(const Variant &p_key);

	bool operator==(const Dictionary &p_dictionary) const;
	bool operator!=(const Dictionary &p_dictionary) const;
	bool recursive_equal(const Dictionary &p_dictionary, int recursion_count) const;

	uint32_t hash() const;
	uint32_t recursive_hash(int recursion_count) const;
	void operator=(const Dictionary &p_dictionary);

	const Variant *next(const Variant *p_key = nullptr) const;

	Array keys() const;
	Array values() const;

	Dictionary duplicate(bool p_deep = false) const;
	Dictionary recursive_duplicate(bool p_deep, int recursion_count) const;

	const void *id() const;

	Dictionary(const Dictionary &p_from);
	Dictionary();
	~Dictionary();
};

#endif // DICTIONARY_H
