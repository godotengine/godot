/**************************************************************************/
/*  gdscript_member_annotations.cpp                                       */
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

#include "gdscript_member_annotations.h"

void GDScriptVariableAnnotation::_bind_methods() {
	GDVIRTUAL_BIND(_analyze, "name", "type_name", "type", "is_static");
	GDVIRTUAL_BIND(_get_allow_multiple);
	GDVIRTUAL_BIND(_is_export_annotation);
	GDVIRTUAL_BIND(_get_property_hint);
	GDVIRTUAL_BIND(_get_property_hint_string);
	GDVIRTUAL_BIND(_get_property_usage);
}

bool GDScriptVariableAnnotation::apply(GDScriptParser::VariableNode *p_target, GDScriptParser::ClassNode *p_class) {
	if (is_export_annotation()) {
		if (p_target->is_static) {
			error_message = vformat(R"(Annotation "%s" cannot be applied to a static variable.)", name);
			return false;
		}
		if (p_target->exported) {
			error_message = vformat(R"(Annotation "%s" cannot be used with another export annotation.)", name);
			return false;
		}

		p_target->exported = true;

		GDScriptParser::DataType export_type = p_target->get_datatype();

		p_target->export_info.type = export_type.builtin_type;
		p_target->export_info.hint = get_property_hint();
		p_target->export_info.hint_string = get_property_hint_string();
		p_target->export_info.usage = get_property_usage();
	}

	return true;
}

void GDScriptFunctionAnnotation::_bind_methods() {
	GDVIRTUAL_BIND(_analyze, "name", "parameter_names", "parameter_type_names", "parameter_builtin_types", "return_type_name", "return_builtin_type", "default_arguments", "is_static", "is_coroutine");
	GDVIRTUAL_BIND(_get_allow_multiple);
}

void GDScriptSignalAnnotation::_bind_methods() {
	GDVIRTUAL_BIND(_analyze, "name", "parameter_names", "parameter_type_names", "parameter_builtin_types");
	GDVIRTUAL_BIND(_get_allow_multiple);
}

void GDScriptClassAnnotation::_bind_methods() {
	GDVIRTUAL_BIND(_analyze, "name");
	GDVIRTUAL_BIND(_get_allow_multiple);
}
