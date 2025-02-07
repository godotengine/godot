/**************************************************************************/
/*  gdscript_member_annotations.h                                         */
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

#ifndef GDSCRIPT_MEMBER_ANNOTATIONS_H
#define GDSCRIPT_MEMBER_ANNOTATIONS_H

#include "gdscript_annotation.h"
#include "gdscript_parser.h"

class GDScriptVariableAnnotation : public GDScriptAnnotation {
	GDCLASS(GDScriptVariableAnnotation, GDScriptAnnotation);

protected:
	static void _bind_methods();

public:
	GDVIRTUAL0RC(bool, _get_allow_multiple)
	virtual bool get_allow_multiple() override final {
		if (GDVIRTUAL_IS_OVERRIDDEN(_get_allow_multiple)) {
			bool ret = false;
			GDVIRTUAL_CALL(_get_allow_multiple, ret);
			return ret;
		}
		return false;
	}

	virtual TargetFlags get_target_mask() override final {
		return TARGET_VARIABLE;
	}

	GDVIRTUAL4(_analyze, StringName, StringName, Variant::Type, bool)
	virtual void analyze(const StringName &p_name, const StringName &p_type_name, Variant::Type p_builtin_type, bool p_is_static) {
		if (GDVIRTUAL_IS_OVERRIDDEN(_analyze)) {
			GDVIRTUAL_CALL(_analyze, p_name, p_type_name, p_builtin_type, p_is_static);
		}
	}

	GDVIRTUAL0RC(bool, _is_export_annotation)
	virtual bool is_export_annotation() const {
		if (GDVIRTUAL_IS_OVERRIDDEN(_is_export_annotation)) {
			bool ret = false;
			GDVIRTUAL_CALL(_is_export_annotation, ret);
			return ret;
		} else {
			return false;
		}
	}

	GDVIRTUAL0RC(int, _get_property_hint)
	virtual PropertyHint get_property_hint() const {
		if (GDVIRTUAL_IS_OVERRIDDEN(_get_property_hint)) {
			int ret = PROPERTY_HINT_NONE;
			GDVIRTUAL_CALL(_get_property_hint, ret);
			return (PropertyHint)ret;
		} else {
			return PROPERTY_HINT_NONE;
		}
	}

	GDVIRTUAL0RC(String, _get_property_hint_string)
	virtual String get_property_hint_string() const {
		if (GDVIRTUAL_IS_OVERRIDDEN(_get_property_hint_string)) {
			String ret;
			GDVIRTUAL_CALL(_get_property_hint_string, ret);
			return ret;
		} else {
			return String();
		}
	}

	GDVIRTUAL0RC(int, _get_property_usage)
	virtual PropertyUsageFlags get_property_usage() const {
		if (GDVIRTUAL_IS_OVERRIDDEN(_get_property_usage)) {
			int ret = PROPERTY_USAGE_DEFAULT;
			GDVIRTUAL_CALL(_get_property_usage, ret);
			return (PropertyUsageFlags)ret;
		} else {
			return PROPERTY_USAGE_DEFAULT;
		}
	}

	// Default implementation is roughly equivalent to using @export_custom.
	// This means no validation is performed on the hint string. The user is responsible for validation in _init.
	virtual bool apply(GDScriptParser::VariableNode *p_target, GDScriptParser::ClassNode *p_class);
};

class GDScriptFunctionAnnotation : public GDScriptAnnotation {
	GDCLASS(GDScriptFunctionAnnotation, GDScriptAnnotation);

protected:
	static void _bind_methods();

public:
	GDVIRTUAL0RC(bool, _get_allow_multiple)
	virtual bool get_allow_multiple() override final {
		if (GDVIRTUAL_IS_OVERRIDDEN(_get_allow_multiple)) {
			bool ret = false;
			GDVIRTUAL_CALL(_get_allow_multiple, ret);
			return ret;
		}
		return false;
	}

	virtual TargetFlags get_target_mask() override final {
		return TARGET_FUNCTION;
	}

	GDVIRTUAL9(_analyze, StringName, PackedStringArray, PackedStringArray, PackedInt32Array, StringName, Variant::Type, Array, bool, bool)
	virtual void analyze(const StringName &p_name, const PackedStringArray &p_parameter_names, const PackedStringArray &p_parameter_type_names, const PackedInt32Array &p_parameter_builtin_types, const StringName &p_return_type_name, Variant::Type p_return_builtin_type, const Array &p_default_arguments, bool p_is_static, bool p_is_coroutine) {
		if (GDVIRTUAL_IS_OVERRIDDEN(_analyze)) {
			GDVIRTUAL_CALL(_analyze, p_name, p_parameter_names, p_parameter_type_names, p_parameter_builtin_types, p_return_type_name, p_return_builtin_type, p_default_arguments, p_is_static, p_is_coroutine);
		}
	}
};

class GDScriptSignalAnnotation : public GDScriptAnnotation {
	GDCLASS(GDScriptSignalAnnotation, GDScriptAnnotation);

protected:
	static void _bind_methods();

public:
	GDVIRTUAL0RC(bool, _get_allow_multiple)
	virtual bool get_allow_multiple() override final {
		if (GDVIRTUAL_IS_OVERRIDDEN(_get_allow_multiple)) {
			bool ret = false;
			GDVIRTUAL_CALL(_get_allow_multiple, ret);
			return ret;
		}
		return false;
	}

	virtual TargetFlags get_target_mask() override final {
		return TARGET_SIGNAL;
	}

	GDVIRTUAL4(_analyze, StringName, PackedStringArray, PackedStringArray, PackedInt32Array)
	virtual void analyze(const StringName &p_name, const PackedStringArray &p_parameter_names, const PackedStringArray &p_parameter_type_names, const PackedInt32Array &p_parameter_builtin_types) {
		if (GDVIRTUAL_IS_OVERRIDDEN(_analyze)) {
			GDVIRTUAL_CALL(_analyze, p_name, p_parameter_names, p_parameter_type_names, p_parameter_builtin_types);
		}
	}
};

class GDScriptClassAnnotation : public GDScriptAnnotation {
	GDCLASS(GDScriptClassAnnotation, GDScriptAnnotation);

protected:
	static void _bind_methods();

public:
	GDVIRTUAL0RC(bool, _get_allow_multiple)
	virtual bool get_allow_multiple() override final {
		if (GDVIRTUAL_IS_OVERRIDDEN(_get_allow_multiple)) {
			bool ret = false;
			GDVIRTUAL_CALL(_get_allow_multiple, ret);
			return ret;
		}
		return false;
	}

	virtual TargetFlags get_target_mask() override final {
		return TARGET_CLASS;
	}

	GDVIRTUAL1(_analyze, StringName)
	virtual void analyze(const StringName &p_name) {
		if (GDVIRTUAL_IS_OVERRIDDEN(_analyze)) {
			GDVIRTUAL_CALL(_analyze, p_name);
		}
	}
};

#endif // GDSCRIPT_MEMBER_ANNOTATIONS_H
