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

#include "gdscript.h"
#include "gdscript_bytecode_serializer.h"
#include "gdscript_cache.h"
#include "gdscript_parser.h"
#include "gdscript_resource_format.h"
#include "gdscript_tokenizer_buffer.h"
#include "gdscript_utility_functions.h"

#ifdef TOOLS_ENABLED
#include "editor/gdscript_highlighter.h"
#include "editor/gdscript_translation_parser_plugin.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/file_system/editor_paths.h"

#ifndef GDSCRIPT_NO_LSP
#include "language_server/gdscript_language_protocol.h"
#include "language_server/gdscript_language_server.h"
#endif
#endif // TOOLS_ENABLED

#ifdef TESTS_ENABLED
#include "tests/test_gdscript.h"
#endif

#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/object/class_db.h"
#include "core/os/os.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/export/editor_export.h"
#include "editor/translations/editor_translation_parser.h"

#ifndef GDSCRIPT_NO_LSP
#include "core/config/engine.h"
#endif
#endif // TOOLS_ENABLED

#ifdef TESTS_ENABLED
#include "tests/test_macros.h"
#endif

GDScriptLanguage *script_language_gd = nullptr;
Ref<ResourceFormatLoaderGDScript> resource_loader_gd;
Ref<ResourceFormatSaverGDScript> resource_saver_gd;
GDScriptCache *gdscript_cache = nullptr;

#ifdef TOOLS_ENABLED

Ref<GDScriptEditorTranslationParserPlugin> gdscript_translation_parser_plugin;

class EditorExportGDScript : public EditorExportPlugin {
	GDCLASS(EditorExportGDScript, EditorExportPlugin);

	static constexpr EditorExportPreset::ScriptExportMode DEFAULT_SCRIPT_MODE = EditorExportPreset::MODE_SCRIPT_BINARY_TOKENS_COMPRESSED;
	EditorExportPreset::ScriptExportMode script_mode = DEFAULT_SCRIPT_MODE;
	String dump_dir;

	String _get_safe_dump_dir(const String &p_output_path) const {
		String dump_root;
		if (EditorPaths::get_singleton()) {
			dump_root = EditorPaths::get_singleton()->get_cache_dir();
		}
		if (dump_root.is_empty()) {
			dump_root = OS::get_singleton()->get_temp_path();
		}

		String dump_name = p_output_path.get_file().get_basename().validate_filename();
		if (dump_name.is_empty()) {
			dump_name = "export";
		}

		return dump_root.path_join("gdscript_bytecode_dump").path_join(dump_name + "-" + p_output_path.md5_text().substr(0, 12));
	}

	void _dump_bytecode(const String &p_path, const Vector<uint8_t> &p_data, const GDScript *p_script) {
		if (dump_dir.is_empty()) {
			return;
		}

		String rel_path = p_path;
		if (rel_path.begins_with("res://")) {
			rel_path = rel_path.substr(6);
		}

		String file_dir = dump_dir.path_join(rel_path.get_base_dir());
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		da->make_dir_recursive(file_dir);

		String gdb_path = dump_dir.path_join(rel_path.get_basename() + ".gdb");
		{
			Ref<FileAccess> f = FileAccess::open(gdb_path, FileAccess::WRITE);
			if (f.is_valid()) {
				f->store_buffer(p_data.ptr(), p_data.size());
				f->flush();
			}
		}

		String txt_path = dump_dir.path_join(rel_path.get_basename() + ".gdb.txt");
		{
			String text = GDScriptBytecodeSerializer::dump_script_text(p_script);
			Ref<FileAccess> f = FileAccess::open(txt_path, FileAccess::WRITE);
			if (f.is_valid()) {
				f->store_string(text);
				f->flush();
			}
		}

		print_line(vformat("[Bytecode Dump] %s -> %s", p_path, txt_path));
	}

protected:
	virtual void _export_begin(const HashSet<String> &p_features, bool p_debug, const String &p_path, int p_flags) override {
		script_mode = DEFAULT_SCRIPT_MODE;

		const Ref<EditorExportPreset> &preset = get_export_preset();
		if (preset.is_valid()) {
			script_mode = preset->get_script_export_mode();
		}

		dump_dir = "";
		if (script_mode == EditorExportPreset::MODE_SCRIPT_COMPILED_BYTECODE && !p_path.is_empty()) {
			dump_dir = _get_safe_dump_dir(p_path);
			Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			da->make_dir_recursive(dump_dir);
			print_line(vformat("[Bytecode Dump] Output directory: %s", dump_dir));
		}
	}

	virtual void _export_end() override {
		dump_dir = "";
	}

	virtual void _export_file(const String &p_path, const String &p_type, const HashSet<String> &p_features) override {
		if (p_path.get_extension() != "gd" || script_mode == EditorExportPreset::MODE_SCRIPT_TEXT) {
			return;
		}

		if (script_mode == EditorExportPreset::MODE_SCRIPT_COMPILED_BYTECODE) {
			// Compiled bytecode mode: serialize VM bytecode instead of tokens.
			Error err;
			Ref<GDScript> script = ResourceLoader::load(p_path, "GDScript", ResourceFormatLoader::CACHE_MODE_REUSE, &err);
			print_line(vformat("[Bytecode Export] Processing: %s", p_path));
			if (err != OK || script.is_null() || !script->is_valid()) {
				print_line(vformat("[Bytecode Export] FAILED to load/compile, falling back to .gdc: %s (err=%d, null=%d, valid=%d)", p_path, (int)err, script.is_null(), script.is_null() ? 0 : (int)script->is_valid()));
				// Fall through to .gdc tokenizer path below.
			} else {
				Vector<uint8_t> data = GDScriptBytecodeSerializer::serialize_script(script.ptr());
				if (data.is_empty()) {
					print_line(vformat("[Bytecode Export] Serialization returned empty, falling back to .gdc: %s", p_path));
					// Fall through to .gdc tokenizer path below.
				} else {
					print_line(vformat("[Bytecode Export] OK: %s -> .gdb (%d bytes)", p_path, data.size()));
					add_file(p_path.get_basename() + ".gdb", data, true);
					_dump_bytecode(p_path, data, script.ptr());
					return;
				}
			}
		}

		Vector<uint8_t> file = FileAccess::get_file_as_bytes(p_path);
		if (file.is_empty()) {
			return;
		}

		String source = String::utf8(reinterpret_cast<const char *>(file.ptr()), file.size());
		GDScriptTokenizerBuffer::CompressMode compress_mode = script_mode == EditorExportPreset::MODE_SCRIPT_BINARY_TOKENS_COMPRESSED ? GDScriptTokenizerBuffer::COMPRESS_ZSTD : GDScriptTokenizerBuffer::COMPRESS_NONE;
		file = GDScriptTokenizerBuffer::parse_code_string(source, compress_mode);
		if (file.is_empty()) {
			return;
		}

		add_file(p_path.get_basename() + ".gdc", file, true);
	}

public:
	virtual String get_name() const override { return "GDScript"; }
};

static void _editor_init() {
	Ref<EditorExportGDScript> gd_export;
	gd_export.instantiate();
	EditorExport::get_singleton()->add_export_plugin(gd_export);

#ifdef TOOLS_ENABLED
	Ref<GDScriptSyntaxHighlighter> gdscript_syntax_highlighter;
	gdscript_syntax_highlighter.instantiate();
	ScriptEditor::get_singleton()->register_syntax_highlighter(gdscript_syntax_highlighter);
#endif
}

#endif // TOOLS_ENABLED

void initialize_gdscript_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		GDREGISTER_CLASS(GDScript);

		script_language_gd = memnew(GDScriptLanguage);
		ScriptServer::register_language(script_language_gd);

		resource_loader_gd.instantiate();
		ResourceLoader::add_resource_format_loader(resource_loader_gd);

		resource_saver_gd.instantiate();
		ResourceSaver::add_resource_format_saver(resource_saver_gd);

		gdscript_cache = memnew(GDScriptCache);

		GDScriptUtilityFunctions::register_functions();
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		EditorNode::add_init_callback(_editor_init);

		gdscript_translation_parser_plugin.instantiate();
		EditorTranslationParser::get_singleton()->add_parser(gdscript_translation_parser_plugin, EditorTranslationParser::STANDARD);
	} else if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		GDREGISTER_CLASS(GDScriptSyntaxHighlighter);
#ifndef GDSCRIPT_NO_LSP
		register_lsp_types();
		memnew(GDScriptLanguageProtocol);
		EditorPlugins::add_by_type<GDScriptLanguageServer>();

		Engine::Singleton singleton("GDScriptLanguageProtocol", GDScriptLanguageProtocol::get_singleton());
		singleton.editor_only = true;
		Engine::get_singleton()->add_singleton(singleton);
#endif // !GDSCRIPT_NO_LSP
	}
#endif // TOOLS_ENABLED
}

void uninitialize_gdscript_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		ScriptServer::unregister_language(script_language_gd);

		if (gdscript_cache) {
			memdelete(gdscript_cache);
		}

		if (script_language_gd) {
			memdelete(script_language_gd);
		}

		ResourceLoader::remove_resource_format_loader(resource_loader_gd);
		resource_loader_gd.unref();

		ResourceSaver::remove_resource_format_saver(resource_saver_gd);
		resource_saver_gd.unref();

		GDScriptParser::cleanup();
		GDScriptUtilityFunctions::unregister_functions();
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		EditorTranslationParser::get_singleton()->remove_parser(gdscript_translation_parser_plugin, EditorTranslationParser::STANDARD);
		gdscript_translation_parser_plugin.unref();
#ifndef GDSCRIPT_NO_LSP
		memdelete(GDScriptLanguageProtocol::get_singleton());
#endif // GDSCRIPT_NO_LSP
	}
#endif // TOOLS_ENABLED
}

#ifdef TESTS_ENABLED
void test_tokenizer() {
	GDScriptTests::test(GDScriptTests::TestType::TEST_TOKENIZER);
}

void test_tokenizer_buffer() {
	GDScriptTests::test(GDScriptTests::TestType::TEST_TOKENIZER_BUFFER);
}

void test_parser() {
	GDScriptTests::test(GDScriptTests::TestType::TEST_PARSER);
}

void test_compiler() {
	GDScriptTests::test(GDScriptTests::TestType::TEST_COMPILER);
}

void test_bytecode() {
	GDScriptTests::test(GDScriptTests::TestType::TEST_BYTECODE);
}

REGISTER_TEST_COMMAND("gdscript-tokenizer", &test_tokenizer);
REGISTER_TEST_COMMAND("gdscript-tokenizer-buffer", &test_tokenizer_buffer);
REGISTER_TEST_COMMAND("gdscript-parser", &test_parser);
REGISTER_TEST_COMMAND("gdscript-compiler", &test_compiler);
REGISTER_TEST_COMMAND("gdscript-bytecode", &test_bytecode);
#endif
