/**************************************************************************/
/*  variant_struct_native.h                                               */
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

#include "core/variant/variant_struct.h"

#define STRINGIFY_MACRO(s) #s

#define VARIANT_STRUCT_DEFINITION(m_parent_class, m_struct_class, ...)                                                        \
	template <>                                                                                                               \
	StructDefinition *NativeStructDefinition<m_parent_class::m_struct_class>::build_definition() {                            \
		using T = m_parent_class::m_struct_class;                                                                             \
		StructDefinition *sd = StructDefinition::create(                                                                      \
				{##__VA_ARGS__##},                                                                                            \
				STRINGIFY_MACRO(m_parent_class.m_struct_class),                                                               \
				sizeof(m_parent_class::m_struct_class),                                                                       \
				(!std::is_trivially_constructible_v<T> ? &init_struct : &StructDefinition::generic_constructor),              \
				(!std::is_trivially_copy_constructible_v<T> ? &copy_construct : &StructDefinition::generic_copy_constructor), \
				(!std::is_trivially_destructible_v<T> ? &deinit_struct : &StructDefinition::trivial_destructor));             \
		return sd;                                                                                                            \
	}

#define VARIANT_STRUCT_PROPERTY(m_property_name) \
	StructDefinition::build_native_property(#m_property_name, &T::##m_property_name)

#define REGISTER_INBUILT_STRUCT(m_parent_class, m_struct_class)                                                                          \
	{                                                                                                                                    \
		ClassDB::bind_struct(#m_parent_class, #m_struct_class, &NativeStructDefinition<m_parent_class::m_struct_class>::get_definition); \
	}
