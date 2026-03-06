/**************************************************************************/
/*  validation_context.h                                                  */
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

#include "core/object/class_db.h"
#include "core/object/object.h"

class ValidationContext : public Object {
	GDCLASS(ValidationContext, Object);

public:
	enum ValidationSeverity {
		VALIDATION_SEVERITY_INFO,
		VALIDATION_SEVERITY_WARNING,
		VALIDATION_SEVERITY_ERROR,
	};

	struct ValidationInfo {
		ValidationSeverity severity = (ValidationSeverity)-1;
		String message;
		String scope;
	};

private:
	String current_scope;
	Vector<ValidationInfo> validations;

protected:
	static void _bind_methods();

public:
	void set_current_scope(const String &p_scope);
	String get_current_scope() const;
	bool has_errors() const;
	Vector<ValidationInfo> get_validations() const;
	Vector<ValidationInfo> get_validations_for_scope(const String &p_scope) const;
	void add_validation(const ValidationSeverity &p_severity, const String &p_message);
	void clear();
	void clear_scope(const String &p_scope = "");
};

VARIANT_ENUM_CAST(ValidationContext::ValidationSeverity);
