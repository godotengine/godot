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

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/object/class_db.h"
#include "core/object/script_language.h"
#include "core/string/print_string.h"
#include "src/cpp/resource_saver_cpp.h"
#include "src/cpp/script_cpp.h"
#include "src/cpp/script_language_cpp.h"
#include "src/elf/resource_loader_elf.h"
#include "src/elf/script_elf.h"
#include "src/elf/script_language_elf.h"
#include "src/sandbox.h"

static Ref<ResourceFormatLoaderELF> resource_loader_elf;
static Ref<ResourceFormatSaverCPP> resource_saver_cpp;

void initialize_sandbox_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		// Register the Sandbox class
		GDREGISTER_CLASS(Sandbox);
		GDREGISTER_CLASS(SandboxBase);

		// Register ELF script classes
		GDREGISTER_CLASS(ELFScript);
		GDREGISTER_CLASS(ELFScriptLanguage);

		// Register CPP script classes
		GDREGISTER_CLASS(CPPScript);
		GDREGISTER_CLASS(CPPScriptLanguage);

		// Register script languages
		ScriptServer::register_language(memnew(ELFScriptLanguage));
		CPPScriptLanguage::init_language();
		ScriptServer::register_language(CPPScriptLanguage::get_singleton());

		// Register resource loaders/savers
		resource_loader_elf.instantiate();
		ResourceLoader::add_resource_format_loader(resource_loader_elf);

		resource_saver_cpp.instantiate();
		ResourceSaver::add_resource_format_saver(resource_saver_cpp);
	}
}

void uninitialize_sandbox_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		// Unregister resource loaders/savers
		if (resource_loader_elf.is_valid()) {
			ResourceLoader::remove_resource_format_loader(resource_loader_elf);
			resource_loader_elf.unref();
		}

		if (resource_saver_cpp.is_valid()) {
			ResourceSaver::remove_resource_format_saver(resource_saver_cpp);
			resource_saver_cpp.unref();
		}

		// Cleanup script languages
		CPPScriptLanguage::deinit();

		// Note: ELFScriptLanguage cleanup will be handled by ScriptServer
		// when it shuts down, as it was created with memnew()

		// Cleanup static dummy machine to prevent ObjectDB leaks
		Sandbox::cleanup_static_resources();
	}
}
