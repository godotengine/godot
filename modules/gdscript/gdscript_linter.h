/**************************************************************************/
/*  gdscript_linter.h                                                     */
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

#ifdef DEBUG_ENABLED

#include "modules/gdscript/gdscript_parser.h"

/**
 * The `GDScriptLinter` emits warnings based on a completely analyzed AST.
 *
 * The linter pass can be skipped if it is not needed, e.g. in release builds
 * or when analyzing for editor features like autocompletion. As such the linter
 * must not have an influence on compilation i.e. it can't modify the AST.
 *
 * Especially expensive warnings should be implemented through this pass.
 */
class GDScriptLinter final {
	using CallbackWithValidation = void(const GDScriptParser::Node *, GDScriptParser &);

	GDScriptParser *const tree;
	static CallbackWithValidation *const checks[];

public:
	template <typename T>
	using Callback = void(const T *, GDScriptParser &);

public:
	Error lint();
	GDScriptLinter(GDScriptParser &p_tree) : tree(&p_tree) {}
};

#endif // DEBUG_ENABLED
