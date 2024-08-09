/**************************************************************************/
/*  platform_text_linuxbsd.cpp                                            */
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

#include "platform_text_linuxbsd.h"

String PlatformTextLinuxBSD::get_string(PlatformTextID p_id) const {
	// switch (p_id) {
	// 	default:
	// 		return PlatformTextImplementation::get_string(p_id);
	// }

	// TODO: remove me when the previously commented switch will be used.
	return PlatformTextImplementation::get_string(p_id);
}

String PlatformTextLinuxBSD::get_editor_string(PlatformTextEditorID p_id) const {
#ifdef TOOLS_ENABLED
	if (!_is_editor()) {
		return PlatformTextImplementation::get_editor_string(p_id);
	}

	// switch (p_id) {
	// 	default:
	// 		return PlatformTextImplementation::get_editor_string(p_id);
	// }

	// TODO: remove me when the previously commented switch will be used.
	return PlatformTextImplementation::get_editor_string(p_id);
#else
	return PlatformTextImplementation::get_editor_string(p_id);
#endif
}

PlatformTextLinuxBSD::PlatformTextLinuxBSD() {}
PlatformTextLinuxBSD::~PlatformTextLinuxBSD() {}
