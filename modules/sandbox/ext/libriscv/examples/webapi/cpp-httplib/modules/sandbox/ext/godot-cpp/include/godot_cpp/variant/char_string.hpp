/**************************************************************************/
/*  char_string.hpp                                                       */
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

#include <godot_cpp/templates/cowdata.hpp>

#include <cstddef>
#include <cstdint>

namespace godot {

template <typename T>
class CharStringT;

template <typename T>
class CharProxy {
	template <typename TS>
	friend class CharStringT;

	const int64_t _index;
	CowData<T> &_cowdata;
	static inline const T _null = 0;

	_FORCE_INLINE_ CharProxy(const int64_t &p_index, CowData<T> &p_cowdata) :
			_index(p_index),
			_cowdata(p_cowdata) {}

public:
	_FORCE_INLINE_ CharProxy(const CharProxy<T> &p_other) :
			_index(p_other._index),
			_cowdata(p_other._cowdata) {}

	_FORCE_INLINE_ operator T() const {
		if (unlikely(_index == _cowdata.size())) {
			return _null;
		}

		return _cowdata.get(_index);
	}

	_FORCE_INLINE_ const T *operator&() const {
		return _cowdata.ptr() + _index;
	}

	_FORCE_INLINE_ void operator=(const T &p_other) const {
		_cowdata.set(_index, p_other);
	}

	_FORCE_INLINE_ void operator=(const CharProxy<T> &p_other) const {
		_cowdata.set(_index, p_other.operator T());
	}
};

template <typename T>
class CharStringT {
	friend class String;

	CowData<T> _cowdata;
	static inline const T _null = 0;

public:
	_FORCE_INLINE_ T *ptrw() { return _cowdata.ptrw(); }
	_FORCE_INLINE_ const T *ptr() const { return _cowdata.ptr(); }
	_FORCE_INLINE_ int64_t size() const { return _cowdata.size(); }
	Error resize(int64_t p_size) { return _cowdata.resize(p_size); }

	_FORCE_INLINE_ T get(int64_t p_index) const { return _cowdata.get(p_index); }
	_FORCE_INLINE_ void set(int64_t p_index, const T &p_elem) { _cowdata.set(p_index, p_elem); }
	_FORCE_INLINE_ const T &operator[](int64_t p_index) const {
		if (unlikely(p_index == _cowdata.size())) {
			return _null;
		}

		return _cowdata.get(p_index);
	}
	_FORCE_INLINE_ CharProxy<T> operator[](int64_t p_index) { return CharProxy<T>(p_index, _cowdata); }

	_FORCE_INLINE_ CharStringT() {}
	_FORCE_INLINE_ CharStringT(const CharStringT<T> &p_str) { _cowdata._ref(p_str._cowdata); }
	_FORCE_INLINE_ void operator=(const CharStringT<T> &p_str) { _cowdata._ref(p_str._cowdata); }
	_FORCE_INLINE_ CharStringT(const T *p_cstr) { copy_from(p_cstr); }

	void operator=(const T *p_cstr);
	bool operator<(const CharStringT<T> &p_right) const;
	CharStringT<T> &operator+=(T p_char);
	int64_t length() const { return size() ? size() - 1 : 0; }
	const T *get_data() const;
	operator const T *() const { return get_data(); }

protected:
	void copy_from(const T *p_cstr);
};

template <>
const char *CharStringT<char>::get_data() const;

template <>
const char16_t *CharStringT<char16_t>::get_data() const;

template <>
const char32_t *CharStringT<char32_t>::get_data() const;

template <>
const wchar_t *CharStringT<wchar_t>::get_data() const;

typedef CharStringT<char> CharString;
typedef CharStringT<char16_t> Char16String;
typedef CharStringT<char32_t> Char32String;
typedef CharStringT<wchar_t> CharWideString;

} // namespace godot
