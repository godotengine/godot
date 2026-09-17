/**************************************************************************/
/*  editor_language_metadata.h                                            */
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

#include <functional>
#include "core/templates/vector.h"
#include "core/string/string_name.h"
#include "core/object/object.h"

class EditorLanguageMetadata {
public:
	// Contains static methods registered by each class to provide argument options for auto completion
	using StaticArgOptionGetter = std::function<Vector<String>(const Object * /*p_instance*/, const StringName & /*p_function*/, int /*p_idx*/)>;

	static void register_argument_options_getters(const StringName &p_class, const StaticArgOptionGetter &p_arg_option_getter) {
		static_arg_options_getters.insert(p_class, p_arg_option_getter);
	}
	// return static argument options getters for the class and its parent classes
	static void get_argument_options_getters(const StringName &p_class, List<EditorLanguageMetadata::StaticArgOptionGetter> *r_arg_option_getters);

private:
	static HashMap<StringName, StaticArgOptionGetter> static_arg_options_getters;
};
