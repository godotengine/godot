/**************************************************************************/
/*  shader_include.cpp                                                    */
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

#include "shader_include.h"

#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "servers/rendering/shader_preprocessor.h"

void ShaderInclude::_dependency_changed() {
	emit_changed();
}

void ShaderInclude::set_code(const String &p_code) {
	code = p_code;

	for (const Ref<ShaderInclude> &E : dependencies) {
		E->disconnect_changed(callable_mp(this, &ShaderInclude::_dependency_changed));
	}

	{
		String path = get_path();
		if (path.is_empty()) {
			path = include_path;
		}

		String pp_code;
		HashSet<Ref<ShaderInclude>> new_dependencies;
		ShaderPreprocessor preprocessor;
		Error result = preprocessor.preprocess(p_code, path, pp_code, nullptr, nullptr, nullptr, &new_dependencies);
		if (result == OK) {
			// This ensures previous include resources are not freed and then re-loaded during parse (which would make compiling slower)
			dependencies = new_dependencies;
		}
	}

	for (const Ref<ShaderInclude> &E : dependencies) {
		E->connect_changed(callable_mp(this, &ShaderInclude::_dependency_changed));
	}

	emit_changed();
}

String ShaderInclude::get_code() const {
	return code;
}

void ShaderInclude::set_include_path(const String &p_path) {
	include_path = p_path;
}

void ShaderInclude::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_code", "code"), &ShaderInclude::set_code);
	ClassDB::bind_method(D_METHOD("get_code"), &ShaderInclude::get_code);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "code", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_code", "get_code");
}
