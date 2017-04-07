/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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

#include "gd_script.h"
#include "io/file_access_encrypted.h"
#include "io/resource_loader.h"
#include "os/file_access.h"

GDScriptLanguage *script_language_gd = NULL;
ResourceFormatLoaderGDScript *resource_loader_gd = NULL;
ResourceFormatSaverGDScript *resource_saver_gd = NULL;
#if 0
#ifdef TOOLS_ENABLED

#include "editor/editor_import_export.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "gd_tokenizer.h"

class EditorExportGDScript : public EditorExportPlugin {

	GDCLASS(EditorExportGDScript,EditorExportPlugin);

public:

	virtual Vector<uint8_t> custom_export(String& p_path,const Ref<EditorExportPlatform> &p_platform) {
		//compile gdscript to bytecode

		if (EditorImportExport::get_singleton()->script_get_action()!=EditorImportExport::SCRIPT_ACTION_NONE) {

			if (p_path.ends_with(".gd")) {
				Vector<uint8_t> file = FileAccess::get_file_as_array(p_path);
				if (file.empty())
					return file;
				String txt;
				txt.parse_utf8((const char*)file.ptr(),file.size());
				file = GDTokenizerBuffer::parse_code_string(txt);

				if (!file.empty()) {

					if (EditorImportExport::get_singleton()->script_get_action()==EditorImportExport::SCRIPT_ACTION_ENCRYPT) {

						String tmp_path=EditorSettings::get_singleton()->get_settings_path().plus_file("tmp/script.gde");
						FileAccess *fa = FileAccess::open(tmp_path,FileAccess::WRITE);
						String skey=EditorImportExport::get_singleton()->script_get_encryption_key().to_lower();
						Vector<uint8_t> key;
						key.resize(32);
						for(int i=0;i<32;i++) {
							int v=0;
							if (i*2<skey.length()) {
								CharType ct = skey[i*2];
								if (ct>='0' && ct<='9')
									ct=ct-'0';
								else if (ct>='a' && ct<='f')
									ct=10+ct-'a';
								v|=ct<<4;
							}

							if (i*2+1<skey.length()) {
								CharType ct = skey[i*2+1];
								if (ct>='0' && ct<='9')
									ct=ct-'0';
								else if (ct>='a' && ct<='f')
									ct=10+ct-'a';
								v|=ct;
							}
							key[i]=v;
						}
						FileAccessEncrypted *fae=memnew(FileAccessEncrypted);
						Error err = fae->open_and_parse(fa,key,FileAccessEncrypted::MODE_WRITE_AES256);
						if (err==OK) {

							fae->store_buffer(file.ptr(),file.size());
							p_path=p_path.get_basename()+".gde";
						}

						memdelete(fae);

						file=FileAccess::get_file_as_array(tmp_path);
						return file;


					} else {

						p_path=p_path.get_basename()+".gdc";
						return file;
					}
				}

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
#endif
void register_gdscript_types() {

	ClassDB::register_class<GDScript>();
	ClassDB::register_virtual_class<GDFunctionState>();

	script_language_gd = memnew(GDScriptLanguage);
	//script_language_gd->init();
	ScriptServer::register_language(script_language_gd);
	resource_loader_gd = memnew(ResourceFormatLoaderGDScript);
	ResourceLoader::add_resource_format_loader(resource_loader_gd);
	resource_saver_gd = memnew(ResourceFormatSaverGDScript);
	ResourceSaver::add_resource_format_saver(resource_saver_gd);
#if 0
#ifdef TOOLS_ENABLED

	EditorNode::add_init_callback(register_editor_plugin);
#endif
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
