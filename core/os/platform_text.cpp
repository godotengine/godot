/**************************************************************************/
/*  platform_text.cpp                                                     */
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

#include "platform_text.h"

String PlatformTextImplementation::get_string(PlatformTextID p_id) const {
	// switch (p_id) {
	// 	default:
	// 		return "";
	// }

	// TODO: remove me when the previously commented switch will be used.
	return "";
}

String PlatformTextImplementation::get_editor_string(PlatformTextEditorID p_id) const {
#ifdef TOOLS_ENABLED
	if (!_is_editor()) {
		return "";
	}

	// switch (p_id) {
	// 	default:
	// 		return "";
	// }

	// TODO: remove me when the previously commented switch will be used.
	return "";
#else
	return "";
#endif
}

bool PlatformTextImplementation::_is_editor() const {
	return Engine::get_singleton()->is_editor_hint();
}

//////////////////////////

PlatformText *PlatformText::singleton = nullptr;

String PlatformText::get_string(PlatformTextID p_id) const {
	ERR_FAIL_COND_V(implementation == nullptr, "");
	return implementation->get_string(p_id);
}

String PlatformText::get_editor_string(PlatformTextEditorID p_id) const {
	ERR_FAIL_COND_V(implementation == nullptr, "");
	return implementation->get_editor_string(p_id);
}

void PlatformText::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_string", "id"), &PlatformText::get_string);
	ClassDB::bind_method(D_METHOD("get_editor_string", "id"), &PlatformText::get_editor_string);
}
