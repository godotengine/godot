/**************************************************************************/
/*  string_name.h                                                         */
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

#include "core/string/ustring.h"
#include "core/templates/safe_refcount.h"

#define UNIQUE_NODE_PREFIX "%"

class Main;

class [[nodiscard]] StringName {
	template <CowBuffer buf>
	friend struct ComptimeStringName;
	struct Table;

	struct _Data {
		SafeRefCount refcount;
		String name;
#ifdef DEBUG_ENABLED
		uint32_t debug_references;
#endif
		bool is_static;
		uint32_t hash;
		_Data *prev;
		_Data *next;
	};

	_Data *_data = nullptr;

	struct Register {
		Register(_Data &p_data);
	};

	void unref();
	friend void register_core_types();
	friend void unregister_core_types();
	friend class Main;
	static void cleanup();
	static uint32_t get_empty_hash();
	static inline bool configured = true;
#ifdef DEBUG_ENABLED
	struct DebugSortReferences {
		bool operator()(const _Data *p_left, const _Data *p_right) const {
			return p_left->debug_references > p_right->debug_references;
		}
	};

	static inline bool debug_stringname = false;
#endif

	constexpr StringName(_Data *p_data) { _data = p_data; }

public:
	_FORCE_INLINE_ explicit operator bool() const { return _data; }

	bool operator==(const String &p_name) const;
	bool operator==(const char *p_name) const;

	const char32_t *get_data() const { return _data ? _data->name.ptr() : U""; }
	char32_t operator[](int p_index) const;
	int length() const;
	_FORCE_INLINE_ bool is_empty() const { return !_data; }

	_FORCE_INLINE_ bool is_node_unique_name() const {
		if (!_data) {
			return false;
		}
		return (char32_t)_data->name[0] == (char32_t)UNIQUE_NODE_PREFIX[0];
	}
	_FORCE_INLINE_ bool operator<(const StringName &p_name) const {
		return _data < p_name._data;
	}
	_FORCE_INLINE_ bool operator<=(const StringName &p_name) const {
		return _data <= p_name._data;
	}
	_FORCE_INLINE_ bool operator>(const StringName &p_name) const {
		return _data > p_name._data;
	}
	_FORCE_INLINE_ bool operator>=(const StringName &p_name) const {
		return _data >= p_name._data;
	}
	_FORCE_INLINE_ bool operator==(const StringName &p_name) const {
		// The real magic of all this mess happens here.
		// This is why path comparisons are very fast.
		return _data == p_name._data;
	}
	_FORCE_INLINE_ uint32_t hash() const {
		if (_data) {
			return _data->hash;
		} else {
			return get_empty_hash();
		}
	}
	_FORCE_INLINE_ const void *data_unique_pointer() const {
		return (void *)_data;
	}

	_FORCE_INLINE_ operator String() const {
		if (_data) {
			return _data->name;
		}

		return String();
	}

	struct AlphCompare {
		template <typename LT, typename RT>
		_FORCE_INLINE_ bool operator()(const LT &l, const RT &r) const {
			return compare(l, r);
		}
		_FORCE_INLINE_ static bool compare(const StringName &l, const StringName &r) {
			return str_compare(l.get_data(), r.get_data()) < 0;
		}
		_FORCE_INLINE_ static bool compare(const String &l, const StringName &r) {
			return str_compare(l.get_data(), r.get_data()) < 0;
		}
		_FORCE_INLINE_ static bool compare(const StringName &l, const String &r) {
			return str_compare(l.get_data(), r.get_data()) < 0;
		}
		_FORCE_INLINE_ static bool compare(const String &l, const String &r) {
			return str_compare(l.get_data(), r.get_data()) < 0;
		}
	};

	StringName &operator=(const StringName &p_name);
	StringName &operator=(StringName &&p_name) {
		if (_data == p_name._data) {
			return *this;
		}

		unref();
		_data = p_name._data;
		p_name._data = nullptr;
		return *this;
	}
	StringName(const char *p_name, bool p_static = false);
	constexpr StringName(const StringName &p_name) {
		_data = nullptr;

		if (std::is_constant_evaluated() || (p_name._data && p_name._data->refcount.ref())) {
			_data = p_name._data;
		}
	}
	StringName(StringName &&p_name) {
		_data = p_name._data;
		p_name._data = nullptr;
	}
	StringName(const String &p_name, bool p_static = false);
	constexpr StringName() = default;

#ifdef SIZE_EXTRA
	_NO_INLINE_
#else
	_FORCE_INLINE_
#endif
	constexpr ~StringName() {
		if (!std::is_constant_evaluated() && _data) {
			unref();
		}
	}

#ifdef DEBUG_ENABLED
	static void set_debug_stringnames(bool p_enable) { debug_stringname = p_enable; }
#endif
};

// Zero-constructing StringName initializes _data to nullptr (and thus empty).
template <>
struct is_zero_constructible<StringName> : std::true_type {};

/*
 * The SNAME macro is used to speed up StringName creation, as it allows caching it after the first usage in a very efficient way.
 * It should NOT be used everywhere, but instead in places where high performance is required and the creation of a StringName
 * can be costly. Places where it should be used are:
 * - Control::get_theme_*(<name> and Window::get_theme_*(<name> functions.
 * - emit_signal(<name>,..) function
 * - call_deferred(<name>,..) function
 * - Comparisons to a StringName in overridden _set and _get methods.
 *
 * Use in places that can be called hundreds of times per frame (or more) is recommended, but this situation is very rare. If in doubt, do not use.
 */

#define SNAME(m_arg) ComptimeStringName<m_arg>().value

template <CowBuffer buf>
struct ComptimeStringName {
private:
	inline static constinit StringName::_Data data = {
		.refcount = SafeRefCount(2),
		.name = ComptimeString<buf>().value,
#ifdef DEBUG_ENABLED
		.debug_references = 0,
#endif
		.is_static = true,
		.hash = buf.hash(),
		.prev = nullptr,
		.next = nullptr
	};

	inline static StringName::Register _reg{ data };

public:
	// TODO: Once we can constexpr `String::is_empty()` this should be `data.name.is_empty() ? nullptr : &data`.
	// For now we can only watch out not to pass empty string in.
#if defined(_MSC_VER) && !defined(__clang__)
	// MSVC is more strict, but it can handle initialization order properly.
	inline static constexpr StringName value{ &data };
#else
	// Force register data before use for other compilers.
	inline static constexpr StringName value{ (_reg, &data) };
#endif
};
