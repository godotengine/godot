/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "modules/gdscript/gdscript.h"
#include "src/gdscript_language_wrapper.h"
#include "src/gdscript_wrapper.h"

// Access the original GDScriptLanguage through its singleton
// GDScriptLanguage::get_singleton() provides access to the original instance

// Our wrapper instance
static GDScriptLanguageWrapper *script_language_wrapper = nullptr;

void initialize_gdscript_elf_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		// At this point, gdscript module should already be initialized
		// (modules initialize in alphabetical order, so "gdscript" comes before "gdscript_elf")
		// Access the original GDScriptLanguage through its singleton
		GDScriptLanguage *original_language = GDScriptLanguage::get_singleton();

		ERR_FAIL_NULL_MSG(original_language, "GDScript module must be initialized before gdscript_elf module");

		// Unregister the original GDScriptLanguage
		ScriptServer::unregister_language(original_language);

		// Create our wrapper
		script_language_wrapper = memnew(GDScriptLanguageWrapper);
		script_language_wrapper->set_original_language(original_language);

		// Register the wrapper instead
		ScriptServer::register_language(script_language_wrapper);

		GDREGISTER_CLASS(GDScriptLanguageWrapper);
		GDREGISTER_CLASS(GDScriptWrapper);
	}
}

void uninitialize_gdscript_elf_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		if (script_language_wrapper) {
			// Get the original language from the wrapper before deleting it
			GDScriptLanguage *original_language = GDScriptLanguage::get_singleton();

			// Unregister our wrapper
			ScriptServer::unregister_language(script_language_wrapper);

			// Delete the wrapper
			memdelete(script_language_wrapper);
			script_language_wrapper = nullptr;

			// Re-register original for proper cleanup
			// Note: This may cause issues if gdscript module uninitializes after us
			// The original will be cleaned up by gdscript module's uninitialize
			if (original_language) {
				ScriptServer::register_language(original_language);
			}
		}
	}
}
