/**************************************************************************/
/*  gltf_item.cpp                                                         */
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

#include "gltf_item.h"
#include "gltf_item.compat.inc"

#include "../gltf_state.h"

#include "core/object/class_db.h"

Variant GLTFItem::get_additional_data(const StringName &p_extension_name) const {
	return _additional_data.get(p_extension_name, Variant());
}

bool GLTFItem::has_additional_data(const StringName &p_extension_name) const {
	return _additional_data.has(p_extension_name);
}

void GLTFItem::set_additional_data(const StringName &p_extension_name, Variant p_additional_data) {
	_additional_data[p_extension_name] = p_additional_data;
}

Error GLTFItem::read_from_dictionary(const Dictionary &p_dict) {
	if (p_dict.has("name")) {
		set_name(p_dict["name"]);
	}
	if (p_dict.has("extras")) {
		set_meta("extras", p_dict["extras"]);
	}
	Error err = OK;
	GDVIRTUAL_CALL(_read_from_dictionary, p_dict, err);
	return err;
}

Dictionary GLTFItem::write_to_dictionary() const {
	Dictionary dict;
	GDVIRTUAL_CALL(_write_to_dictionary, dict);
	// It is possible for the virtual call to give us read-only Arrays or Dictionaries.
	// `dict.is_read_only()` could check for surface-level read-only status,
	// but we need to duplicate deep in case nested data is read-only too.
	dict = dict.duplicate_deep(RESOURCE_DEEP_DUPLICATE_NONE);
	// It is also possible for user code to give us a TypedDictionary, so we need to convert it.
	if (dict.is_typed()) {
		Dictionary untyped;
		untyped.assign(dict);
		dict = untyped;
	}
	// Write common GLTFItem data.
	if (has_meta("extras")) {
		dict["extras"] = get_meta("extras");
	}
	if (!get_name().is_empty()) {
		dict["name"] = get_name();
	}
	return dict;
}

void GLTFItem::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_additional_data", "extension_name"), &GLTFItem::get_additional_data);
	ClassDB::bind_method(D_METHOD("has_additional_data", "extension_name"), &GLTFItem::has_additional_data);
	ClassDB::bind_method(D_METHOD("set_additional_data", "extension_name", "additional_data"), &GLTFItem::set_additional_data);

	ClassDB::bind_method(D_METHOD("read_from_dictionary", "dictionary"), &GLTFItem::read_from_dictionary);
	ClassDB::bind_method(D_METHOD("write_to_dictionary"), &GLTFItem::write_to_dictionary);
	GDVIRTUAL_BIND(_read_from_dictionary, "dictionary");
	GDVIRTUAL_BIND(_write_to_dictionary);
}
