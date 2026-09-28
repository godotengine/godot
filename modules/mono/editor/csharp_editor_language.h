/**************************************************************************/
/*  csharp_editor_language.h                                              */
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

#include "core/object/editor_language.h"

class CSharpEditorLanguage final : public EditorLanguage {
	static CSharpEditorLanguage *singleton;

public:
	_FORCE_INLINE_ static CSharpEditorLanguage *get_singleton() { return singleton; }

	virtual const Vector<String> get_reserved_words() const override;
	virtual bool is_control_flow_keyword(const String &p_keyword) const override;
	virtual const Vector<String> get_comment_delimiters() const override;
	virtual const Vector<String> get_doc_comment_delimiters() const override;
	virtual const Vector<String> get_string_delimiters() const override;
	virtual NameCasing get_preferred_file_name_casing() const override;

	CSharpEditorLanguage() {
		ERR_FAIL_COND(singleton != nullptr);
		singleton = this;
	}
	~CSharpEditorLanguage() {
		if (singleton == this) {
			singleton = nullptr;
		}
	}
};
