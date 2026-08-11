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

// TODO: This file resides in `core/` while migrating `ScriptLanguage` features to prevent `editor/` includes in core. It should be moved to `editor/` when those includes are not needed anymore.
#ifdef TOOLS_ENABLED

#include "core/object/script_language.h"

/**
 * Interface for supporting semantic language features in the builtin script editor.
 *
 * API design should keep the LSP spec in mind: https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/ since the GDScript language server is also using this API.
 *
 * Note: This class is unstable and work-in-progress, stability is only ensured for the things currently exposed through `ScriptLanguageExtension`.
 * The goal is to move all editor functionality out of `ScriptLanguage` over time.
 */
class EditorLanguage {
public:
	// Keep enums in sync with:
	// scene/gui/code_edit.h - CodeEdit::CodeCompletionKind
	enum class CompletionKind {
		CLASS,
		FUNCTION,
		SIGNAL,
		VARIABLE,
		MEMBER_VARIABLE,
		ENUM,
		CONSTANT,
		NODE_PATH,
		FILE_PATH,
		PLAIN_TEXT,
		KEYWORD,
	};

	// scene/gui/code_edit.h - CodeEdit::CodeCompletionLocation
	// Can't bundle in namespace since namespaces are not allowed inside of class :/
	struct CompletionLocation {
		static constexpr int LOCAL = 0;
		static constexpr int PARENT_MASK = 1 << 8;
		static constexpr int OTHER_USER_CODE = 1 << 9;
		static constexpr int OTHER = 1 << 10;

		CompletionLocation() = delete;
	};

	struct TextEdit {
		String new_text;
		int start_line = -1;
		int start_column = -1;
		int end_line = -1;
		int end_column = -1;

		_FORCE_INLINE_ bool is_set() const { return start_line != -1; }
	};

	struct CompletionOption {
		EditorLanguage::CompletionKind kind = EditorLanguage::CompletionKind::PLAIN_TEXT;
		String display;
		String insert_text;
		/**
		 * Optional server side calculated insertion.
		 *
		 * In contrast to `insert_text`, the editor must not do matching of preexisting text on `text_edit`.
		 * Note: This is used by the language server, there is no support in the builtin editor for this property at the moment.
		 */
		TextEdit text_edit;
		Variant default_value;
		int location = CompletionLocation::OTHER;
		String theme_color_name;

		CompletionOption() = default;

		CompletionOption(const String &p_text, EditorLanguage::CompletionKind p_kind, int p_location = CompletionLocation::OTHER, const String &p_theme_color_name = String()) {
			display = p_text;
			insert_text = p_text;
			kind = p_kind;
			location = p_location;
			theme_color_name = p_theme_color_name;
		}
	};

	/**
	 * Called by the editor to request a list of `CodeCompletionOptions` from the language.
	 *
	 * The language MAY also calculate a hint for the call signature at the given position. See `r_call_hint` and `r_force` for more details.
	 *
	 * @param p_code The current content of the source fragment. Will always contain the special sentinel char 0xFFFF at the cursor position. The content might differ from the state on disk, the exact behavior in that case is up to the implementation.
	 * @param p_path The path which identifies the source fragment. Implementations MAY support paths to builtin resources, or return an error for those.
	 * @param p_owner If the source fragment is somehow tied to an object (e.g. the currently open scene) the editor will pass it to the language. Implementations MAY respect this in their analysis, e.g. for GDScript get node literals.
	 * @param r_options The returned options. Can be empty.
	 * @param r_force If `false` a non-empty signature hint will take priority over completion. If `true` completion will take priority. Might be removed in the future in favor of showing both.
	 * @param r_call_hint The returned signature hint. Can be empty.
	 */
	virtual Error complete_code(const String &p_code, const String &p_path, Object *p_owner, List<CompletionOption> *r_options, bool &r_force, String &r_call_hint) { return ERR_UNAVAILABLE; }

	// Keep ScriptLanguageExtension::LookupResultType a subset of this.
	struct LookupResult {
		enum class Type {
			SCRIPT_LOCATION, // Use if none of the options below apply.
			CLASS,
			CLASS_CONSTANT,
			CLASS_PROPERTY,
			CLASS_METHOD,
			CLASS_SIGNAL,
			CLASS_ENUM,
			CLASS_TBD_GLOBALSCOPE [[deprecated]], // Don't bind to ClassDB.
			CLASS_ANNOTATION,
			LOCAL_CONSTANT,
			LOCAL_VARIABLE,
		};

		Type type;

		// For `CLASS_*`.
		String class_name;
		String class_member;

		// For `LOCAL_*`.
		String description;
		bool is_deprecated = false;
		String deprecated_message;
		bool is_experimental = false;
		String experimental_message;

		// For `LOCAL_*`.
		String doc_type;
		String enumeration;
		bool is_bitfield = false;

		// For `LOCAL_*`.
		String value;

		// `SCRIPT_LOCATION` and `LOCAL_*` must have, `CLASS_*` can have.
		String script_path;
		int location = -1;
	};

	/**
	 * Called by the editor to lookup code at a given position.
	 *
	 * @param p_code The current content of the source fragment. Will always contain the special sentinel char 0xFFFF at the requested position. The content might differ from the state on disk, the exact behavior in that case is up to the implementation.
	 * @param p_symbol The symbol under the cursor position. This parameter is unreliable (sometimes uses the user configured word separators) and will be removed in the future. New implementations should not use it and determine the symbol under the cursor on their own.
	 * @param p_path The path which identifies the source fragment. Implementations MAY support paths to builtin resources, or return an error for those.
	 * @param p_owner If the source fragment is somehow tied to an object (e.g. the currently open scene) the editor will pass it to the language. Implementations MAY respect this in their analysis, e.g. for GDScript get node literals.
	 * @param r_result The returned `LookupResult`.
	 */
	virtual Error lookup_code(const String &p_code, const String &p_symbol, const String &p_path, Object *p_owner, LookupResult &r_result) { return ERR_UNAVAILABLE; }

	/**
	 * Called by the editor to find a top-level function in the given source code.
	 *
	 * @return The zero-based line number of the function or `-1` if not found.
	 */
	virtual int32_t find_function(const String &p_function, const String &p_code) const { return -1; }

	/**
	 * Called by the editor to reformat a section of code.
	 *
	 * In the case of GDScript the only supported kind of formatting is auto-indentation.
	 *
	 * @param r_code The current content of the source fragment. Implementations should also use this to return the formatted code. Either by formatting in place or through assignment.
	 * @param p_from_line First line to be formatted (inclusive).
	 * @param p_to_line Last line to be formatted (inclusive).
	 */
	virtual void format_code(String &r_code, uint32_t p_from_line, uint32_t p_to_line) const {}

	struct Warning {
		/// One-based.
		int start_line = 0;
		int start_column = -1;

		/// One-based.
		int end_line = 0;
		int end_column = -1;

		String string_code;
		String message;
	};

	struct ScriptError {
		String path;
		/// All one-based.
		int start_line = -1;
		int start_column = -1;
		int end_line = -1;
		int end_column = -1;
		String message;
	};

	/**
	 * Called by the editor to check a source fragment for diagnostics (warnings and errors).
	 *
	 * The language should also return a list of functions for the editor to show in the outline. It can also specify safe lines for highlighting. In the future those features might be separated.
	 *
	 * @param p_code The current content of the source fragment.
	 * @param p_path The path which identifies the source fragment. Implementations MAY support paths to builtin resources, or return an error for those.
	 * @param r_errors The returned errors of the script. Might be `nullptr` if the caller does not care.
	 * @param r_warnings The returned warnings of the script. Might be `nullptr` if the caller does not care.
	 * @param r_functions The returned functions for the outline. Might be `nullptr` if the caller does not care.
	 * @param r_safe_lines The returned safe-lines, this is a GDScript specific concept. Might be `nullptr` if the caller does not care.
	 * @return `true` if the script has no errors, otherwise `false`.
	 */
	virtual bool validate(const String &p_code, const String &p_path, List<ScriptError> *r_errors, List<Warning> *r_warnings, List<String> *r_functions, HashSet<int> *r_safe_lines) const { return true; }

	virtual ~EditorLanguage() = default;
};

#endif // TOOLS_ENABLED
