/**************************************************************************/
/*  csharp_language_server.cpp                                            */
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

#include "csharp_language_server.h"

#include "editor/editor_node.h"
#include "editor/editor_settings.h"

void CSharpLanguageServer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			EditorNode *editor_node = EditorNode::get_singleton();
			editor_node->connect("script_add_function_request", callable_mp(this, &CSharpLanguageServer::add_method_in_external_editor));
		} break;
	}
}

void CSharpLanguageServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_method_in_external_editor", "obj", "method", "args"), &CSharpLanguageServer::add_method_in_external_editor);
}

void CSharpLanguageServer::add_method_in_external_editor(Object *p_obj, const String &p_method, const PackedStringArray &p_args) {
	Ref<Script> scr = p_obj->get_script();
	bool use_external_editor =
			EDITOR_GET("text_editor/external/use_external_editor") ||
			(scr.is_valid() && scr->get_language()->overrides_external_editor());
	if (use_external_editor && scr->get_language()->get_name() == "C#") {
		CSharpLanguage::get_singleton()->get_godotsharp_editor()->call("AddMethodInExternalEditor", scr, p_method, p_args);
	}
}
