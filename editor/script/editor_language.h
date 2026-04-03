/**************************************************************************/
/*  editor_language.h                                                     */
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

#include "core/object/script_language.h"

/**
 * Interface for supporting semantic language features in the builtin script editor.
 *
 * API design should keep the LSP spec in mind: https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/ since the GDScript LSP is also using this API.
 *
 * Note: This class is unstable and work-in-progress, stability is only ensured for the things currently exposed through `ScriptLanguageExtension`.
 * The goal is to move all editor functionality out of `ScriptLanguage` over time.
 */
class EditorLanguage {
public:
	/**
	 * Called by the editor to request a list of `CodeCompletionOptions` from the language.
	 *
	 * The language MAY also calculate a hint for the call signature at the given position. See `r_call_hint` and `r_force` for more details.
	 *
	 * @param p_code The current content of the source fragment. Will always contain the special sentinel char 0xFFFF at the cursor position. The content might differ from the state on disk, the exact behavior in that case is up to the implementation.
	 * @param p_path The path which identifies the source fragment. Implementations MAY support paths to builtin resources, or return an error for those.
	 * @param p_owner If the source fragment is somehow tied to an object (e.g. the currently open scene) the editor will pass it to the language. Implementations MAY respect this in their analysis, e.g. for GDScript get node literals.
	 * @param r_options The returned options. Can be empty.
	 * @param r_force If `false` a non-empty call hint will take priority over completion. If `true` completion will take priority. Might be removed in the future in favor of showing both.
	 * @param r_call_hint The returned signature hint. Might be empty.
	 */
	virtual Error complete_code(const String &p_code, const String &p_path, Object *p_owner, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_force, String &r_call_hint) { return ERR_UNAVAILABLE; }

	virtual ~EditorLanguage() = default;
};
