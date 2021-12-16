/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/config/engine.h"
#include "core/io/resource_loader.h"
#include "visual_script.h"
#include "visual_script_builtin_funcs.h"
#include "visual_script_expression.h"
#include "visual_script_flow_control.h"
#include "visual_script_func_nodes.h"
#include "visual_script_nodes.h"
#include "visual_script_yield_nodes.h"

VisualScriptLanguage *visual_script_language = nullptr;

#ifdef TOOLS_ENABLED
#include "editor/visual_script_editor.h"
static VisualScriptCustomNodes *vs_custom_nodes_singleton = nullptr;
#endif

void register_visual_script_types() {
	visual_script_language = memnew(VisualScriptLanguage);
	//script_language_gd->init();
	ScriptServer::register_language(visual_script_language);

	GDREGISTER_CLASS(VisualScript);
	GDREGISTER_VIRTUAL_CLASS(VisualScriptNode);
	GDREGISTER_HIDDEN_CLASS(VisualScriptFunctionState);
	GDREGISTER_HIDDEN_CLASS(VisualScriptFunction);
	GDREGISTER_VIRTUAL_HIDDEN_CLASS(VisualScriptLists);
	GDREGISTER_HIDDEN_CLASS(VisualScriptComposeArray);
	GDREGISTER_HIDDEN_CLASS(VisualScriptOperator);
	GDREGISTER_HIDDEN_CLASS(VisualScriptVariableSet);
	GDREGISTER_HIDDEN_CLASS(VisualScriptVariableGet);
	GDREGISTER_HIDDEN_CLASS(VisualScriptConstant);
	GDREGISTER_HIDDEN_CLASS(VisualScriptIndexGet);
	GDREGISTER_HIDDEN_CLASS(VisualScriptIndexSet);
	GDREGISTER_HIDDEN_CLASS(VisualScriptGlobalConstant);
	GDREGISTER_HIDDEN_CLASS(VisualScriptClassConstant);
	GDREGISTER_HIDDEN_CLASS(VisualScriptMathConstant);
	GDREGISTER_HIDDEN_CLASS(VisualScriptBasicTypeConstant);
	GDREGISTER_HIDDEN_CLASS(VisualScriptEngineSingleton);
	GDREGISTER_HIDDEN_CLASS(VisualScriptSceneNode);
	GDREGISTER_HIDDEN_CLASS(VisualScriptSceneTree);
	GDREGISTER_HIDDEN_CLASS(VisualScriptResourcePath);
	GDREGISTER_HIDDEN_CLASS(VisualScriptSelf);
	GDREGISTER_HIDDEN_CLASS(VisualScriptCustomNode);
	GDREGISTER_HIDDEN_CLASS(VisualScriptSubCall);
	GDREGISTER_HIDDEN_CLASS(VisualScriptComment);
	GDREGISTER_HIDDEN_CLASS(VisualScriptConstructor);
	GDREGISTER_HIDDEN_CLASS(VisualScriptLocalVar);
	GDREGISTER_HIDDEN_CLASS(VisualScriptLocalVarSet);
	GDREGISTER_HIDDEN_CLASS(VisualScriptInputAction);
	GDREGISTER_HIDDEN_CLASS(VisualScriptDeconstruct);
	GDREGISTER_HIDDEN_CLASS(VisualScriptPreload);
	GDREGISTER_HIDDEN_CLASS(VisualScriptTypeCast);

	GDREGISTER_HIDDEN_CLASS(VisualScriptFunctionCall);
	GDREGISTER_HIDDEN_CLASS(VisualScriptPropertySet);
	GDREGISTER_HIDDEN_CLASS(VisualScriptPropertyGet);
	//ClassDB::register_type<VisualScriptScriptCall>();
	GDREGISTER_HIDDEN_CLASS(VisualScriptEmitSignal);

	GDREGISTER_HIDDEN_CLASS(VisualScriptReturn);
	GDREGISTER_HIDDEN_CLASS(VisualScriptCondition);
	GDREGISTER_HIDDEN_CLASS(VisualScriptWhile);
	GDREGISTER_HIDDEN_CLASS(VisualScriptIterator);
	GDREGISTER_HIDDEN_CLASS(VisualScriptSequence);
	//GDREGISTER_CLASS(VisualScriptInputFilter);
	GDREGISTER_HIDDEN_CLASS(VisualScriptSwitch);
	GDREGISTER_HIDDEN_CLASS(VisualScriptSelect);

	GDREGISTER_HIDDEN_CLASS(VisualScriptYield);
	GDREGISTER_HIDDEN_CLASS(VisualScriptYieldSignal);

	GDREGISTER_HIDDEN_CLASS(VisualScriptBuiltinFunc);

	GDREGISTER_HIDDEN_CLASS(VisualScriptExpression);

	register_visual_script_nodes();
	register_visual_script_func_nodes();
	register_visual_script_builtin_func_node();
	register_visual_script_flow_control_nodes();
	register_visual_script_yield_nodes();
	register_visual_script_expression_node();

#ifdef TOOLS_ENABLED
	ClassDB::set_current_api(ClassDB::API_EDITOR);
	GDREGISTER_CLASS(VisualScriptCustomNodes);
	ClassDB::set_current_api(ClassDB::API_CORE);
	vs_custom_nodes_singleton = memnew(VisualScriptCustomNodes);
	Engine::get_singleton()->add_singleton(Engine::Singleton("VisualScriptCustomNodes", VisualScriptCustomNodes::get_singleton()));

	VisualScriptEditor::register_editor();
#endif
}

void unregister_visual_script_types() {
	unregister_visual_script_nodes();

	ScriptServer::unregister_language(visual_script_language);

#ifdef TOOLS_ENABLED
	VisualScriptEditor::free_clipboard();
	if (vs_custom_nodes_singleton) {
		memdelete(vs_custom_nodes_singleton);
	}
#endif
	if (visual_script_language) {
		memdelete(visual_script_language);
	}
}
