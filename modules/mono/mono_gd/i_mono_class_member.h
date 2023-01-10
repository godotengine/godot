/**************************************************************************/
/*  i_mono_class_member.h                                                 */
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

#ifndef I_MONO_CLASS_MEMBER_H
#define I_MONO_CLASS_MEMBER_H

#include "gd_mono_header.h"

#include <mono/metadata/object.h>

class IMonoClassMember {
public:
	enum Visibility {
		PRIVATE,
		PROTECTED_AND_INTERNAL, // FAM_AND_ASSEM
		INTERNAL, // ASSEMBLY
		PROTECTED, // FAMILY
		PUBLIC
	};

	enum MemberType {
		MEMBER_TYPE_FIELD,
		MEMBER_TYPE_PROPERTY,
		MEMBER_TYPE_METHOD
	};

	virtual ~IMonoClassMember() {}

	virtual GDMonoClass *get_enclosing_class() const = 0;

	virtual MemberType get_member_type() const = 0;

	virtual StringName get_name() const = 0;

	virtual bool is_static() = 0;

	virtual Visibility get_visibility() = 0;

	virtual bool has_attribute(GDMonoClass *p_attr_class) = 0;
	virtual MonoObject *get_attribute(GDMonoClass *p_attr_class) = 0;
};

#endif // I_MONO_CLASS_MEMBER_H
