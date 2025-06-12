/**************************************************************************/
/*  distributed_string_view.h                                             */
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
#include "core/templates/span.h"

class DistributedStringView {
	template <size_t N>
	static constexpr Span<char> to_span(const char (&p_cstr)[N]) { return Span(p_cstr); }
	template <size_t N>
	static constexpr Span<wchar_t> to_span(const wchar_t (&p_cstr)[N]) { return Span(p_cstr); }
	static Span<char32_t> to_span(const String &p_str) { return p_str.span(); }
	template <typename T>
	static constexpr Span<T> to_span(const Span<T> &p_span) { return p_span; }

	template <typename T>
	static constexpr uint32_t _increment_hash(uint32_t hashv, const Span<T> &part) {
		using unsigned_char = std::conditional_t<sizeof(T) == 1, uint8_t,
				std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;

		for (const T &c : part) {
			hashv = ((hashv << 5) + hashv) + static_cast<unsigned_char>(c);
		}
		return hashv;
	}

	template <typename T>
	static constexpr bool _increment_equal(const Span<char32_t> &ref, const Span<T> &part) {
		if (ref.size() < part.size()) {
			return false;
		}
		for (uint64_t i = 0; i < part.size(); ++i) {
			if (ref[i] != static_cast<char32_t>(part[i])) {
				return false;
			}
		}
		return true;
	}

public:
	template <typename... Parts>
	static constexpr uint32_t size(Parts... parts) {
		uint32_t total_size = 0;
		((total_size += to_span(parts).size()), ...);
		return total_size;
	}
	template <typename... Parts>
	static constexpr uint32_t hash(Parts... parts) {
		uint32_t hashv = 5381;
		((hashv = _increment_hash(hashv, to_span(parts))), ...);
		return hashv;
	}

	template <typename... Parts>
	static constexpr bool equal(const String &ref, Parts... parts) {
		Span<char32_t> ref_span = ref.span();
		if (ref_span.size() != size(parts...)) {
			return false;
		}
		bool result = true;
		uint32_t index = 0;
		((result = result && _increment_equal(Span(ref_span.ptr() + index, ref_span.size() - index), to_span(parts)) && (index += to_span(parts).size())), ...);
		return result;
	}
};
