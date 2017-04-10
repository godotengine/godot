/*************************************************************************/
/*  simple_type.h                                                        */
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
#ifndef SIMPLE_TYPE_H
#define SIMPLE_TYPE_H

/* Batch of specializations to obtain the actual simple type */

template <class T>
struct GetSimpleType {

	T type;
};

template <class T>
struct GetSimpleTypeT {

	typedef T type_t;
};

template <class T>
struct GetSimpleType<T &> {

	T type;
};

template <class T>
struct GetSimpleTypeT<T &> {

	typedef T type_t;
};

template <class T>
struct GetSimpleType<T const> {

	T type;
	_FORCE_INLINE_ GetSimpleType() {}
};

template <class T>
struct GetSimpleTypeT<T const> {

	typedef T type_t;
};

template <class T>
struct GetSimpleType<const T &> {

	T type;
	_FORCE_INLINE_ GetSimpleType() {}
};

template <class T>
struct GetSimpleType<T *> {

	T *type;
	_FORCE_INLINE_ GetSimpleType() { type = NULL; }
};

template <class T>
struct GetSimpleType<const T *> {

	T *type;
	_FORCE_INLINE_ GetSimpleType() { type = NULL; }
};

#define SIMPLE_NUMERIC_TYPE(m_type)                          \
	template <>                                              \
	struct GetSimpleType<m_type> {                           \
		m_type type;                                         \
		_FORCE_INLINE_ GetSimpleType() { type = (m_type)0; } \
	};                                                       \
	template <>                                              \
	struct GetSimpleType<m_type const> {                     \
		m_type type;                                         \
		_FORCE_INLINE_ GetSimpleType() { type = (m_type)0; } \
	};                                                       \
	template <>                                              \
	struct GetSimpleType<m_type &> {                         \
		m_type type;                                         \
		_FORCE_INLINE_ GetSimpleType() { type = (m_type)0; } \
	};                                                       \
	template <>                                              \
	struct GetSimpleType<const m_type &> {                   \
		m_type type;                                         \
		_FORCE_INLINE_ GetSimpleType() { type = (m_type)0; } \
	};

SIMPLE_NUMERIC_TYPE(bool);
SIMPLE_NUMERIC_TYPE(uint8_t);
SIMPLE_NUMERIC_TYPE(int8_t);
SIMPLE_NUMERIC_TYPE(uint16_t);
SIMPLE_NUMERIC_TYPE(int16_t);
SIMPLE_NUMERIC_TYPE(uint32_t);
SIMPLE_NUMERIC_TYPE(int32_t);
SIMPLE_NUMERIC_TYPE(int64_t);
SIMPLE_NUMERIC_TYPE(uint64_t);
SIMPLE_NUMERIC_TYPE(float);
SIMPLE_NUMERIC_TYPE(double);

#endif
