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

GDScriptLanguage *script_language_gd=NULL;
ResourceFormatLoaderGDScript *resource_loader_gd=NULL;
ResourceFormatSaverGDScript *resource_saver_gd=NULL;

void register_gdscript_types() {


	script_language_gd=memnew( GDScriptLanguage );
	script_language_gd->init();
	ScriptServer::register_language(script_language_gd);
	ObjectTypeDB::register_type<GDScript>();
	resource_loader_gd=memnew( ResourceFormatLoaderGDScript );
	ResourceLoader::add_resource_format_loader(resource_loader_gd);
	resource_saver_gd=memnew( ResourceFormatSaverGDScript );
	ResourceSaver::add_resource_format_saver(resource_saver_gd);

}
void unregister_gdscript_types() {




	if (script_language_gd)
		memdelete( script_language_gd );
	if (resource_loader_gd)
		memdelete( resource_loader_gd );
	if (resource_saver_gd)
		memdelete( resource_saver_gd );

}