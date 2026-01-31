/**************************************************************************/
/*  gdextension_interface_conv.h                                          */
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

#include "gdextension_interface.gen.h"

class GDExtension;
class MethodBind;
class Object;
class StringName;
class String;
class Variant;

template <typename T>
class Ref;

template <typename T>
struct GDExtensionPtrTraits;

#define GDEXTENSION_PTR_CONV(m_type, m_opaque_ptr)                 \
	template <>                                                    \
	struct GDExtensionPtrTraits<m_type> {                          \
		using OpaquePtr = m_opaque_ptr;                            \
                                                                   \
		static inline OpaquePtr to(m_type *p_godot_type) {         \
			return reinterpret_cast<OpaquePtr>(p_godot_type);      \
		}                                                          \
		static inline m_type *from(OpaquePtr p_gdextension_type) { \
			return reinterpret_cast<m_type *>(p_gdextension_type); \
		}                                                          \
	};

GDEXTENSION_PTR_CONV(Variant, GDExtensionVariantPtr);
GDEXTENSION_PTR_CONV(const Variant, GDExtensionConstVariantPtr);
GDEXTENSION_PTR_CONV(const Variant *, GDExtensionConstVariantPtr *);
GDEXTENSION_PTR_CONV(StringName, GDExtensionStringNamePtr);
GDEXTENSION_PTR_CONV(const StringName, GDExtensionConstStringNamePtr);
GDEXTENSION_PTR_CONV(String, GDExtensionStringPtr);
GDEXTENSION_PTR_CONV(const String, GDExtensionConstStringPtr);
GDEXTENSION_PTR_CONV(GDExtension, GDExtensionClassLibraryPtr);
GDEXTENSION_PTR_CONV(Object, GDExtensionObjectPtr);
GDEXTENSION_PTR_CONV(const Object, GDExtensionConstObjectPtr);
GDEXTENSION_PTR_CONV(const MethodBind, GDExtensionMethodBindPtr);

template <typename T>
inline typename GDExtensionPtrTraits<T>::OpaquePtr to_gdextension(T *p_godot_type) {
	return GDExtensionPtrTraits<T>::to(p_godot_type);
}

template <typename T>
inline T *from_gdextension(typename GDExtensionPtrTraits<T>::OpaquePtr p_gdextension_type) {
	return GDExtensionPtrTraits<T>::from(p_gdextension_type);
}

// Any encoded type (ie `PtrToArg<T>::EncodedT`) can be converted to `GDExtensionTypePtr`.
template <typename T>
inline GDExtensionTypePtr to_gdextension_type_ptr(typename PtrToArg<T>::EncodeT *p_encoded_value) {
	return reinterpret_cast<GDExtensionTypePtr>(p_encoded_value);
}
