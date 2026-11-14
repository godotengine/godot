/**************************************************************************/
/*  native_class.h                                                        */
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

#include "core/object/gdtype.h"
#include "core/object/ref_counted.h"

/**
 * First-class representation of native (`ClassDB`) types.
 *
 * This is the backing value when doing things like `var a = Node` or when creating callables from static methods.
 *
 * For a concrete implementation use `CompanionObject` which creates a dynamic companion `GDType` which reflects
 * static members. This is needed to support callables which rely on things like `has_method` which we can't override
 * like `callp`.
 */
class GDScriptNativeClass : public RefCounted {
	GDCLASS(GDScriptNativeClass, RefCounted);

protected:
	StringName name;
	bool _get(const StringName &p_name, Variant &r_ret) const;
	static void _bind_methods();

	GDScriptNativeClass(const StringName &p_name) : name(p_name) {}

public:
	_FORCE_INLINE_ const StringName &get_name() const { return name; }
	Variant _new();
	Object *instantiate();
	virtual Variant callp(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) override;
	virtual ~GDScriptNativeClass() = default;
};

/// See `GDScriptNativeClass`.
class CompanionObject final : public GDScriptNativeClass {
	GDSOFTCLASS(CompanionObject, GDScriptNativeClass);

	GDType companion_gdtype;

protected:
	virtual const GDType &_get_typev() const override {
		return companion_gdtype;
	}
	void _notification(int p_what);

public:
	CompanionObject(const StringName &p_name);
	virtual ~CompanionObject() = default;
};
