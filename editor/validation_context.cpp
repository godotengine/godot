/**************************************************************************/
/*  validation_context.cpp                                                */
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

#include "validation_context.h"

void ValidationContext::set_current_scope(const String &p_scope) {
	current_scope = p_scope;
}

String ValidationContext::get_current_scope() const {
	return current_scope;
}

bool ValidationContext::has_errors() const {
	for (const ValidationInfo &validation_info : validations) {
		if (validation_info.severity >= VALIDATION_SEVERITY_ERROR) {
			return true;
		}
	}
	return false;
}

Vector<ValidationContext::ValidationInfo> ValidationContext::get_validations() const {
	return validations;
}

Vector<ValidationContext::ValidationInfo> ValidationContext::get_validations_for_scope(const String &p_scope) const {
	String scope = p_scope;
	if (scope.is_empty()) {
		scope = current_scope;
	}

	Vector<ValidationContext::ValidationInfo> scoped_validations;
	for (const ValidationInfo &validation_info : validations) {
		if (validation_info.scope == scope || validation_info.scope.begins_with(scope + "/")) {
			scoped_validations.push_back(validation_info);
		}
	}
	return scoped_validations;
}

void ValidationContext::add_validation(const ValidationSeverity &p_severity, const String &p_message) {
	ValidationInfo validation_info;
	validation_info.severity = p_severity;
	validation_info.message = p_message;
	validation_info.scope = current_scope;
	validations.push_back(validation_info);
}

void ValidationContext::clear() {
	validations.clear();
}

void ValidationContext::clear_scope(const String &p_scope) {
	String scope = p_scope;
	if (scope.is_empty()) {
		scope = current_scope;
	}

	for (int i = validations.size() - 1; i >= 0; i--) {
		if (validations[i].scope == scope || validations[i].scope.begins_with(scope + "/")) {
			validations.remove_at(i);
		}
	}
}

void ValidationContext::_bind_methods() {
	BIND_ENUM_CONSTANT(VALIDATION_SEVERITY_INFO);
	BIND_ENUM_CONSTANT(VALIDATION_SEVERITY_WARNING);
	BIND_ENUM_CONSTANT(VALIDATION_SEVERITY_ERROR);

	ClassDB::bind_method(D_METHOD("add_validation", "severity", "message"), &ValidationContext::add_validation);
}
