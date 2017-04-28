/*************************************************************************/
/*  string_db.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef STRING_DB_H
#define STRING_DB_H

#include "hash_map.h"
#include "os/mutex.h"
#include "safe_refcount.h"
#include "ustring.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

struct StaticCString {

	const char *ptr;
	static StaticCString create(const char *p_ptr);
};

class StringName {

	enum {

		STRING_TABLE_BITS = 12,
		STRING_TABLE_LEN = 1 << STRING_TABLE_BITS,
		STRING_TABLE_MASK = STRING_TABLE_LEN - 1
	};

	struct _Data {
		SafeRefCount refcount;
		const char *cname;
		String name;

		String get_name() const { return cname ? String(cname) : name; }
		int idx;
		uint32_t hash;
		_Data *prev;
		_Data *next;
		_Data() {
			cname = NULL;
			next = prev = NULL;
			hash = 0;
		}
	};

	static _Data *_table[STRING_TABLE_LEN];

	_Data *_data;

	union _HashUnion {

		_Data *ptr;
		uint32_t hash;
	};

	void unref();
	friend void register_core_types();
	friend void unregister_core_types();

	static Mutex *lock;
	static void setup();
	static void cleanup();
	static bool configured;

	StringName(_Data *p_data) { _data = p_data; }

public:
	operator const void *() const { return (_data && (_data->cname || !_data->name.empty())) ? (void *)1 : 0; }

	bool operator==(const String &p_name) const;
	bool operator==(const char *p_name) const;
	bool operator!=(const String &p_name) const;
	_FORCE_INLINE_ bool operator<(const StringName &p_name) const {

		return _data < p_name._data;
	}
	_FORCE_INLINE_ bool operator==(const StringName &p_name) const {
		// the real magic of all this mess happens here.
		// this is why path comparisons are very fast
		return _data == p_name._data;
	}
	_FORCE_INLINE_ uint32_t hash() const {

		if (_data)
			return _data->hash;
		else
			return 0;
	}
	bool operator!=(const StringName &p_name) const;

	_FORCE_INLINE_ operator String() const {

		if (_data) {
			if (_data->cname)
				return String(_data->cname);
			else
				return _data->name;
		}

		return String();
	}

	static StringName search(const char *p_name);
	static StringName search(const CharType *p_name);
	static StringName search(const String &p_name);

	struct AlphCompare {

		_FORCE_INLINE_ bool operator()(const StringName &l, const StringName &r) const {

			return l.operator String() < r.operator String();
		}
	};

	void operator=(const StringName &p_name);
	StringName(const char *p_name);
	StringName(const StringName &p_name);
	StringName(const String &p_name);
	StringName(const StaticCString &p_static_string);
	StringName();
	~StringName();
};

struct StringNameHasher {

	static _FORCE_INLINE_ uint32_t hash(const StringName &p_string) { return p_string.hash(); }
};

StringName _scs_create(const char *p_chr);

#endif
