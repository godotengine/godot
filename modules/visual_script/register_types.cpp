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

#include "core/engine.h"
#include "io/resource_loader.h"
#include "visual_script.h"
#include "visual_script_builtin_funcs.h"
#include "visual_script_editor.h"
#include "visual_script_expression.h"
#include "visual_script_flow_control.h"
#include "visual_script_func_nodes.h"
#include "visual_script_nodes.h"
#include "visual_script_yield_nodes.h"

VisualScriptLanguage *visual_script_language = NULL;
#ifdef TOOLS_ENABLED
static _VisualScriptEditor *vs_editor_singleton = NULL;
#endif

void register_visual_script_types() {

	visual_script_language = memnew(VisualScriptLanguage);
	//script_language_gd->init();
	ScriptServer::register_language(visual_script_language);

	ClassDB::register_class<VisualScript>();
	ClassDB::register_virtual_class<VisualScriptNode>();
	ClassDB::register_class<VisualScriptFunctionState>();
	ClassDB::register_class<VisualScriptFunction>();
	ClassDB::register_class<VisualScriptOperator>();
	ClassDB::register_class<VisualScriptVariableSet>();
	ClassDB::register_class<VisualScriptVariableGet>();
	ClassDB::register_class<VisualScriptConstant>();
	ClassDB::register_class<VisualScriptIndexGet>();
	ClassDB::register_class<VisualScriptIndexSet>();
	ClassDB::register_class<VisualScriptGlobalConstant>();
	ClassDB::register_class<VisualScriptClassConstant>();
	ClassDB::register_class<VisualScriptMathConstant>();
	ClassDB::register_class<VisualScriptBasicTypeConstant>();
	ClassDB::register_class<VisualScriptEngineSingleton>();
	ClassDB::register_class<VisualScriptSceneNode>();
	ClassDB::register_class<VisualScriptSceneTree>();
	ClassDB::register_class<VisualScriptResourcePath>();
	ClassDB::register_class<VisualScriptSelf>();
	ClassDB::register_class<VisualScriptCustomNode>();
	ClassDB::register_class<VisualScriptSubCall>();
	ClassDB::register_class<VisualScriptComment>();
	ClassDB::register_class<VisualScriptConstructor>();
	ClassDB::register_class<VisualScriptLocalVar>();
	ClassDB::register_class<VisualScriptLocalVarSet>();
	ClassDB::register_class<VisualScriptInputAction>();
	ClassDB::register_class<VisualScriptDeconstruct>();
	ClassDB::register_class<VisualScriptPreload>();
	ClassDB::register_class<VisualScriptTypeCast>();

	ClassDB::register_class<VisualScriptFunctionCall>();
	ClassDB::register_class<VisualScriptPropertySet>();
	ClassDB::register_class<VisualScriptPropertyGet>();
	//ClassDB::register_type<VisualScriptScriptCall>();
	ClassDB::register_class<VisualScriptEmitSignal>();

	ClassDB::register_class<VisualScriptReturn>();
	ClassDB::register_class<VisualScriptCondition>();
	ClassDB::register_class<VisualScriptWhile>();
	ClassDB::register_class<VisualScriptIterator>();
	ClassDB::register_class<VisualScriptSequence>();
	//ClassDB::register_class<VisualScriptInputFilter>();
	ClassDB::register_class<VisualScriptSwitch>();
	ClassDB::register_class<VisualScriptSelect>();

	ClassDB::register_class<VisualScriptYield>();
	ClassDB::register_class<VisualScriptYieldSignal>();

	ClassDB::register_class<VisualScriptBuiltinFunc>();

	ClassDB::register_class<VisualScriptExpression>();

	register_visual_script_nodes();
	register_visual_script_func_nodes();
	register_visual_script_builtin_func_node();
	register_visual_script_flow_control_nodes();
	register_visual_script_yield_nodes();
	register_visual_script_expression_node();

#ifdef TOOLS_ENABLED
	ClassDB::register_class<_VisualScriptEditor>();
	vs_editor_singleton = memnew(_VisualScriptEditor);
	Engine::get_singleton()->add_singleton(Engine::Singleton("VisualScriptEditor", _VisualScriptEditor::get_singleton()));

	VisualScriptEditor::register_editor();
#endif
}

void unregister_visual_script_types() {

	unregister_visual_script_nodes();

	ScriptServer::unregister_language(visual_script_language);

#ifdef TOOLS_ENABLED
	VisualScriptEditor::free_clipboard();
	if (vs_editor_singleton) {
		memdelete(vs_editor_singleton);
	}
#endif
	if (visual_script_language)
		memdelete(visual_script_language);
}
