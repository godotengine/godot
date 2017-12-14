/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "register_types.h"

#include "gdscript.h"
#include "gdscript_tokenizer.h"
#include "io/file_access_encrypted.h"
#include "io/resource_loader.h"
#include "os/file_access.h"

GDScriptLanguage *script_language_gd = NULL;
ResourceFormatLoaderGDScript *resource_loader_gd = NULL;
ResourceFormatSaverGDScript *resource_saver_gd = NULL;

#ifdef TOOLS_ENABLED

#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"

class EditorExportGDScript : public EditorExportPlugin {

	GDCLASS(EditorExportGDScript, EditorExportPlugin);

public:
	virtual void _export_file(const String &p_path, const String &p_type, const Set<String> &p_features) {

		if (!p_path.ends_with(".gd"))
			return;

		Vector<uint8_t> file = FileAccess::get_file_as_array(p_path);
		if (file.empty())
			return;
		String txt;
		txt.parse_utf8((const char *)file.ptr(), file.size());
		file = GDScriptTokenizerBuffer::parse_code_string(txt);

		if (file.empty())
			return;

		add_file(p_path.get_basename() + ".gdc", file, true);
	}
};

static void _editor_init() {

	Ref<EditorExportGDScript> gd_export;
	gd_export.instance();
	EditorExport::get_singleton()->add_export_plugin(gd_export);
}

#endif

void register_gdscript_types() {

	ClassDB::register_class<GDScript>();
	ClassDB::register_virtual_class<GDScriptFunctionState>();

	script_language_gd = memnew(GDScriptLanguage);
	ScriptServer::register_language(script_language_gd);
	resource_loader_gd = memnew(ResourceFormatLoaderGDScript);
	ResourceLoader::add_resource_format_loader(resource_loader_gd);
	resource_saver_gd = memnew(ResourceFormatSaverGDScript);
	ResourceSaver::add_resource_format_saver(resource_saver_gd);

#ifdef TOOLS_ENABLED
	EditorNode::add_init_callback(_editor_init);
#endif
}

void unregister_gdscript_types() {

	ScriptServer::unregister_language(script_language_gd);

	if (script_language_gd)
		memdelete(script_language_gd);
	if (resource_loader_gd)
		memdelete(resource_loader_gd);
	if (resource_saver_gd)
		memdelete(resource_saver_gd);
}
