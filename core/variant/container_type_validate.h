/**************************************************************************/
/*  container_type_validate.h                                             */
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
#include "core/object/script_language.h"
#include "core/variant/variant.h"

struct ContainerType {
	Variant::Type variant_type = Variant::NIL;
	StringName class_name;
	Ref<Script> script;

	_FORCE_INLINE_ bool operator==(const ContainerType &p_type) const {
		return variant_type == p_type.variant_type && class_name == p_type.class_name && script == p_type.script;
	}
	_FORCE_INLINE_ bool operator!=(const ContainerType &p_type) const {
		return variant_type != p_type.variant_type || class_name != p_type.class_name || script != p_type.script;
	}
};

struct ContainerTypeValidate : ContainerType {
	const char *where = "container";

private:
	const Variant *_internal_convert_variant(const Variant &p_variant, Variant &r_tmp_variant, const char *p_operation, bool p_output_errors) const;
	_FORCE_INLINE_ const Variant *_internal_validate(const Variant &p_variant, Variant &r_tmp_variant, const char *p_operation, bool p_output_errors) const {
		if (variant_type == Variant::NIL) {
			return &p_variant;
		}
		if (p_variant.get_type() != variant_type) {
			if (p_variant.get_type() == Variant::NIL && variant_type == Variant::OBJECT) {
				return &p_variant;
			}
			return _internal_convert_variant(p_variant, r_tmp_variant, p_operation, p_output_errors);
		}
		if (variant_type != Variant::OBJECT) {
			return &p_variant;
		}
		return _internal_validate_object(p_variant, p_operation, p_output_errors) ? &p_variant : nullptr;
	}
	bool _internal_validate_object(const Variant &p_variant, const char *p_operation, bool p_output_errors) const;

public:
	// Returns a pointer to a Variant holding a compatible value.
	// Modifies and uses r_tmp_variant if conversions are needed.
	_FORCE_INLINE_ const Variant *validate(const Variant &p_variant, Variant &r_tmp_variant, const char *p_operation = "use") const {
		return _internal_validate(p_variant, r_tmp_variant, p_operation, true);
	}

	_FORCE_INLINE_ bool validate_object(const Variant &p_variant, const char *p_operation = "use") const {
		return _internal_validate_object(p_variant, p_operation, true);
	}

	_FORCE_INLINE_ bool test_validate(const Variant &p_variant) const {
		Variant tmp;
		return _internal_validate(p_variant, tmp, "", false) != nullptr;
	}

	_FORCE_INLINE_ bool can_reference(const ContainerTypeValidate &p_type) const {
		if (variant_type != p_type.variant_type) {
			return false;
		} else if (variant_type != Variant::OBJECT) {
			return true;
		}

		if (class_name == StringName()) {
			return true;
		} else if (p_type.class_name == StringName()) {
			return false;
		} else if (class_name != p_type.class_name && !ClassDB::is_parent_class(p_type.class_name, class_name)) {
			return false;
		}

		if (script.is_null()) {
			return true;
		} else if (p_type.script.is_null()) {
			return false;
		} else if (script != p_type.script && !p_type.script->inherits_script(script)) {
			return false;
		}

		return true;
	}
};
