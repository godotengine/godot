/**************************************************************************/
/*  variant_cache.h                                                       */
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

class GDScriptFunction;
class MethodBind;

struct VariantCallCache {
	enum class Type {
		INVALID,
		GDSCRIPT_FUNCTION,
		METHOD_BIND,
		VARIANT_BUILTIN_METHOD,
	};

	Type type;

	struct VariantBuiltInMethod {
		void (*call)(Variant *, const Variant **, int, Variant &, const Vector<Variant> &, Callable::CallError &);
		const Vector<Variant> *default_values;
	};

	union {
		GDScriptFunction *gdscript_function;
		const MethodBind *method_bind;
		VariantBuiltInMethod variant_builtin_method;
	};

	VariantCallCache() : type(Type::INVALID) {}
	VariantCallCache(GDScriptFunction *p_gdscript_function) : type(Type::GDSCRIPT_FUNCTION), gdscript_function(p_gdscript_function) {}
	VariantCallCache(const MethodBind *p_method_bind) : type(Type::METHOD_BIND), method_bind(p_method_bind) {}
	VariantCallCache(
			void (*p_call)(Variant *, const Variant **, int, Variant &, const Vector<Variant> &, Callable::CallError &),
			const Vector<Variant> *p_default_values) :
			type(Type::VARIANT_BUILTIN_METHOD), variant_builtin_method{ p_call, p_default_values } {}
};
