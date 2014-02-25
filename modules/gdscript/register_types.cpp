/*************************************************/
/*  register_script_types.cpp                    */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2010 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#include "register_types.h"

#include "gd_script.h"
#include "io/resource_loader.h"
#include "os/file_access.h"


GDScriptLanguage *script_language_gd=NULL;
ResourceFormatLoaderGDScript *resource_loader_gd=NULL;
ResourceFormatSaverGDScript *resource_saver_gd=NULL;

#ifdef TOOLS_ENABLED

#include "tools/editor/editor_import_export.h"
#include "gd_tokenizer.h"
#include "tools/editor/editor_node.h"

class EditorExportGDScript : public EditorExportPlugin {

	OBJ_TYPE(EditorExportGDScript,EditorExportPlugin);

public:

	virtual Vector<uint8_t> custom_export(String& p_path,const Ref<EditorExportPlatform> &p_platform) {
		//compile gdscript to bytecode
		if (p_path.ends_with(".gd")) {
			Vector<uint8_t> file = FileAccess::get_file_as_array(p_path);
			if (file.empty())
				return file;
			String txt;
			txt.parse_utf8((const char*)file.ptr(),file.size());
			file = GDTokenizerBuffer::parse_code_string(txt);
			if (!file.empty()) {
				print_line("PREV: "+p_path);
				p_path=p_path.basename()+".gdc";
				print_line("NOW: "+p_path);
				return file;
			}

		}

		return Vector<uint8_t>();
	}


	EditorExportGDScript(){}

};

static void register_editor_plugin() {

	Ref<EditorExportGDScript> egd = memnew( EditorExportGDScript );
	EditorImportExport::get_singleton()->add_export_plugin(egd);
}


#endif

void register_gdscript_types() {


	script_language_gd=memnew( GDScriptLanguage );
	script_language_gd->init();
	ScriptServer::register_language(script_language_gd);
	ObjectTypeDB::register_type<GDScript>();
	resource_loader_gd=memnew( ResourceFormatLoaderGDScript );
	ResourceLoader::add_resource_format_loader(resource_loader_gd);
	resource_saver_gd=memnew( ResourceFormatSaverGDScript );
	ResourceSaver::add_resource_format_saver(resource_saver_gd);

#ifdef TOOLS_ENABLED

	EditorNode::add_init_callback(register_editor_plugin);
#endif

}
void unregister_gdscript_types() {




	if (script_language_gd)
		memdelete( script_language_gd );
	if (resource_loader_gd)
		memdelete( resource_loader_gd );
	if (resource_saver_gd)
		memdelete( resource_saver_gd );

}
