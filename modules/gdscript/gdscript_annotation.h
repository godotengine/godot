/**************************************************************************/
/*  gdscript_annotation.h                                                 */
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

#ifndef GDSCRIPT_ANNOTATION_H
#define GDSCRIPT_ANNOTATION_H

#include "core/object/gdvirtual.gen.inc"
#include "core/object/ref_counted.h"

class GDScriptAnnotation : public RefCounted {
	GDCLASS(GDScriptAnnotation, RefCounted);

	friend class GDScriptAnalyzer;

public:
	enum TargetFlags : int {
		TARGET_NONE = 0,
		TARGET_VARIABLE = 1 << 0,
		TARGET_FUNCTION = 1 << 1,
		TARGET_SIGNAL = 1 << 2,
		TARGET_CLASS = 1 << 3,
	};

protected:
	StringName name;
	String error_message;

	static void _bind_methods();

public:
	StringName get_name() const;

	void set_error_message(const String &p_error_message);
	String get_error_message() const;
	_FORCE_INLINE_ bool has_error_message() const { return !error_message.is_empty(); }

	virtual TargetFlags get_target_mask() = 0;

	virtual bool get_allow_multiple() = 0;

	static _FORCE_INLINE_ constexpr const char *target_to_name(TargetFlags p_target) {
		switch (p_target) {
			case TARGET_VARIABLE:
				return "Variable";
			case TARGET_FUNCTION:
				return "Function";
			case TARGET_SIGNAL:
				return "Signal";
			case TARGET_CLASS:
				return "Class";
			default:
				return "Unknown";
		}
	}

	static void find_user_annotations(List<MethodInfo> *r_annotations);
	static void find_native_user_annotations(List<MethodInfo> *r_annotations);
};

#endif // GDSCRIPT_ANNOTATION_H
