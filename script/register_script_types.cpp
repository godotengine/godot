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

#include "register_script_types.h"

#include "script/gdscript/gd_script.h"
#include "script/multiscript/multi_script.h"
#include "io/resource_loader.h"



#ifdef GDSCRIPT_ENABLED
GDScriptLanguage *script_language_gd=NULL;
ResourceFormatLoaderGDScript *resource_loader_gd=NULL;
ResourceFormatSaverGDScript *resource_saver_gd=NULL;
#endif

static MultiScriptLanguage *script_multi_script=NULL;

void register_script_types() {

#ifdef GDSCRIPT_ENABLED

	script_language_gd=memnew( GDScriptLanguage );
	script_language_gd->init();
	ScriptServer::register_language(script_language_gd);
	ObjectTypeDB::register_type<GDScript>();
	resource_loader_gd=memnew( ResourceFormatLoaderGDScript );
	ResourceLoader::add_resource_format_loader(resource_loader_gd);
	resource_saver_gd=memnew( ResourceFormatSaverGDScript );
	ResourceSaver::add_resource_format_saver(resource_saver_gd);
#endif


	script_multi_script = memnew( MultiScriptLanguage );
	ScriptServer::register_language(script_multi_script);
	ObjectTypeDB::register_type<MultiScript>();


}
void unregister_script_types() {




#ifdef GDSCRIPT_ENABLED
	if (script_language_gd)
		memdelete( script_language_gd );
	if (resource_loader_gd)
		memdelete( resource_loader_gd );
	if (resource_saver_gd)
		memdelete( resource_saver_gd );

#endif

	if (script_multi_script);
		memdelete(script_multi_script);
}
