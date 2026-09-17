/**************************************************************************/
/*  editor_language_metadata.cpp                                          */
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

#include "editor_language_metadata.h"

#include "core/object/class_db.h"

HashMap<StringName, EditorLanguageMetadata::StaticArgOptionGetter> EditorLanguageMetadata::static_arg_options_getters;

void EditorLanguageMetadata::get_argument_options_getters(const StringName &p_class, List<EditorLanguageMetadata::StaticArgOptionGetter> *r_arg_option_getters) {
	const GDType* gdtype = ClassDB::get_gdtype(p_class);
	EditorLanguageMetadata::StaticArgOptionGetter *arg_option_getter_ptr = nullptr;
	if (!gdtype) {
		return;
	}
	const StringName class_name = gdtype->get_name();
	arg_option_getter_ptr = EditorLanguageMetadata::static_arg_options_getters.getptr(class_name);
	if (arg_option_getter_ptr) {
		r_arg_option_getters->push_back(*arg_option_getter_ptr);
	}
	get_argument_options_getters(ClassDB::get_parent_class(class_name), r_arg_option_getters);
}
