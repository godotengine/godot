/**************************************************************************/
/*  gdscript_annotation.cpp                                               */
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

#include "gdscript_annotation.h"
#include "core/object/script_language.h"
#include "gdscript.h"

void GDScriptAnnotation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_name"), &GDScriptAnnotation::get_name);

	ClassDB::bind_method(D_METHOD("set_error_message", "error_message"), &GDScriptAnnotation::set_error_message);
	ClassDB::bind_method(D_METHOD("get_error_message"), &GDScriptAnnotation::get_error_message);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "error_message"), "set_error_message", "get_error_message");
}

StringName GDScriptAnnotation::get_name() const {
	return name;
}

void GDScriptAnnotation::set_error_message(const String &p_error_message) {
	error_message = p_error_message;
}

String GDScriptAnnotation::get_error_message() const {
	return error_message;
}

void GDScriptAnnotation::find_user_annotations(List<MethodInfo> *r_annotations) {
	List<StringName> global_classes;
	ScriptServer::get_global_class_list(&global_classes);
	for (const StringName &global_class : global_classes) {
		if (ScriptServer::get_global_class_language(global_class) == GDScript::get_class_static()) {
			if (ClassDB::is_parent_class(ScriptServer::get_global_class_native_base(global_class), get_class_static())) {
				const String path = ScriptServer::get_global_class_path(global_class);
				Ref<GDScript> script = ResourceLoader::load(path, GDScript::get_class_static());
				if (script->has_method("_init")) {
					MethodInfo mi = script->get_method_info("_init");
					mi.name = "@@" + global_class;
					r_annotations->push_back(mi);
				}
			}
		}
	}
}

void GDScriptAnnotation::find_native_user_annotations(List<MethodInfo> *r_annotations) {
	List<StringName> annotation_classes;
	ClassDB::get_inheriters_from_class(get_class_static(), &annotation_classes);
	for (const StringName &annotation_class : annotation_classes) {
		if (!ClassDB::is_abstract(annotation_class) && !ClassDB::is_virtual(annotation_class)) {
			MethodInfo mi;
			if (ClassDB::get_method_info(annotation_class, GDScriptLanguage::get_singleton()->strings._init, &mi)) {
				mi.name = "@@" + annotation_class;
				r_annotations->push_back(mi);
			}
		}
	}
}
