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

#include "multi_script.h"
#include "io/resource_loader.h"

static MultiScriptLanguage *script_multi_script=NULL;

void register_multiscript_types() {


	script_multi_script = memnew( MultiScriptLanguage );
	ScriptServer::register_language(script_multi_script);
	ObjectTypeDB::register_type<MultiScript>();


}
void unregister_multiscript_types() {

	if (script_multi_script);
		memdelete(script_multi_script);
}
