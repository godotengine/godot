/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#include "visual_script.h"
#include "visual_script_editor.h"
#include "io/resource_loader.h"
#include "visual_script_nodes.h"
#include "visual_script_func_nodes.h"
#include "visual_script_builtin_funcs.h"
#include "visual_script_flow_control.h"
#include "visual_script_yield_nodes.h"


VisualScriptLanguage *visual_script_language=NULL;


void register_visual_script_types() {

	ObjectTypeDB::register_type<VisualScript>();
	ObjectTypeDB::register_virtual_type<VisualScriptNode>();
	ObjectTypeDB::register_virtual_type<VisualScriptFunctionState>();
	ObjectTypeDB::register_type<VisualScriptFunction>();
	ObjectTypeDB::register_type<VisualScriptOperator>();
	ObjectTypeDB::register_type<VisualScriptVariableSet>();
	ObjectTypeDB::register_type<VisualScriptVariableGet>();
	ObjectTypeDB::register_type<VisualScriptConstant>();
	ObjectTypeDB::register_type<VisualScriptIndexGet>();
	ObjectTypeDB::register_type<VisualScriptIndexSet>();
	ObjectTypeDB::register_type<VisualScriptGlobalConstant>();
	ObjectTypeDB::register_type<VisualScriptMathConstant>();
	ObjectTypeDB::register_type<VisualScriptEngineSingleton>();
	ObjectTypeDB::register_type<VisualScriptSceneNode>();
	ObjectTypeDB::register_type<VisualScriptSceneTree>();
	ObjectTypeDB::register_type<VisualScriptResourcePath>();
	ObjectTypeDB::register_type<VisualScriptSelf>();
	ObjectTypeDB::register_type<VisualScriptCustomNode>();
	ObjectTypeDB::register_type<VisualScriptSubCall>();

	ObjectTypeDB::register_type<VisualScriptFunctionCall>();
	ObjectTypeDB::register_type<VisualScriptPropertySet>();
	ObjectTypeDB::register_type<VisualScriptPropertyGet>();
	ObjectTypeDB::register_type<VisualScriptScriptCall>();
	ObjectTypeDB::register_type<VisualScriptEmitSignal>();

	ObjectTypeDB::register_type<VisualScriptReturn>();
	ObjectTypeDB::register_type<VisualScriptCondition>();
	ObjectTypeDB::register_type<VisualScriptWhile>();
	ObjectTypeDB::register_type<VisualScriptIterator>();
	ObjectTypeDB::register_type<VisualScriptSequence>();
	ObjectTypeDB::register_type<VisualScriptInputFilter>();
	ObjectTypeDB::register_type<VisualScriptInputSelector>();

	ObjectTypeDB::register_type<VisualScriptYield>();
	ObjectTypeDB::register_type<VisualScriptYieldSignal>();

	ObjectTypeDB::register_type<VisualScriptBuiltinFunc>();

	visual_script_language=memnew( VisualScriptLanguage );
	//script_language_gd->init();
	ScriptServer::register_language(visual_script_language);

	register_visual_script_nodes();
	register_visual_script_func_nodes();
	register_visual_script_builtin_func_node();
	register_visual_script_flow_control_nodes();
	register_visual_script_yield_nodes();

#ifdef TOOLS_ENABLED
	VisualScriptEditor::register_editor();
#endif


}

void unregister_visual_script_types() {


	ScriptServer::unregister_language(visual_script_language);

	if (visual_script_language)
		memdelete( visual_script_language );

}
