/**
 * bb_variant.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bb_variant.h"

void BBVariant::set_type(Variant::Type p_type) {
	if (type != p_type) {
		type = p_type;
		if (get_saved_value().get_type() != p_type) {
			_assign_default_value();
		}
		emit_changed();
		notify_property_list_changed();
	}
}

Variant::Type BBVariant::get_type() const {
	return type;
}

void BBVariant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_type", "type"), &BBVariant::set_type);

	String vtypes;
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (i > 0) {
			vtypes += ",";
		}
		vtypes += Variant::get_type_name(Variant::Type(i));
	}
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, vtypes), "set_type", "get_type");
}

BBVariant::BBVariant(const Variant &p_value) {
	set_type(p_value.get_type());
	set_saved_value(p_value);
}

BBVariant::BBVariant() {
}
