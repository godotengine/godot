/*************************************************************************/
/*  gd_editor.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include <stdlib.h>
#include <string.h>
#include "lua_script.h"

void LuaScriptLanguage::get_comment_delimiters(List<String> *p_delimiters) const {

	p_delimiters->push_back("--");

}
void LuaScriptLanguage::get_string_delimiters(List<String> *p_delimiters) const {

	p_delimiters->push_back("\" \"");
	p_delimiters->push_back("' '");


}
String LuaScriptLanguage::get_template(const String& p_class_name, const String& p_base_class_name) const {

	String _template = String()+
	"\nlocal node = extends '%BASE%'\n\n"+
	"# member variables here, example:\n"+
	"# local a=2\n"+
	"# b=\"textvar\"\n\n"+
    "function node:_ready():\n"+
	"\t# Initalization here\n"+
	"\t-- pass\n"+
	"end\n"+
	"\n"
	"\n";

	return _template.replace("%BASE%",p_base_class_name);
}

bool LuaScriptLanguage::validate(const String& p_script, int &r_line_error,int &r_col_error,String& r_test_error, const String& p_path,List<String> *r_functions) const {

    CharString code = p_script.utf8();

    lua_State *L = LuaScriptLanguage::get_singleton()->get_state();
    int top = lua_gettop(L);
    if(luaL_loadbuffer(L, code.get_data(), code.length(), p_path.utf8().get_data()))
    {
        const char *err = lua_tostring(L, -1);

        int error_line = -1;
        const char *ls = err;
        while(ls != NULL && ls[0] != '\0')
        {
            ls = strchr(ls, ':');
            if(ls == NULL)
                break;
            error_line = atoi(ls + 1);
            if(error_line > 0)
                break;
            ls ++;
        }
        r_line_error = error_line;
        r_col_error = 0;
        r_test_error = err;

        return false;
    }
    lua_pop(L, 1);

	return true;
}

bool LuaScriptLanguage::has_named_classes() const {

	return false;
}

int LuaScriptLanguage::find_function(const String& p_function,const String& p_code) const {

//	GDTokenizerText tokenizer;
//	tokenizer.set_code(p_code);
//	int indent=0;
//	while(tokenizer.get_token()!=GDTokenizer::TK_EOF && tokenizer.get_token()!=GDTokenizer::TK_ERROR) {
//
//		if (tokenizer.get_token()==GDTokenizer::TK_NEWLINE) {
//			indent=tokenizer.get_token_line_indent();
//		}
//		if (indent==0 && tokenizer.get_token()==GDTokenizer::TK_PR_FUNCTION && tokenizer.get_token(1)==GDTokenizer::TK_IDENTIFIER) {
//
//			String identifier = tokenizer.get_token_identifier(1);
//			if (identifier==p_function) {
//				return tokenizer.get_token_line();
//			}
//		}
//		tokenizer.advance();
//	}
	return -1;
}

Script *LuaScriptLanguage::create_script() const {

	return memnew( LuaScript );
}

/* DEBUGGER FUNCTIONS */


bool LuaScriptLanguage::debug_break_parse(const String& p_file, int p_line,const String& p_error) {
	//break because of parse error

    if (ScriptDebugger::get_singleton() && Thread::get_caller_ID()==Thread::get_main_ID()) {

	_debug_parse_err_line=p_line;
	_debug_parse_err_file=p_file;
	_debug_error=p_error;
	ScriptDebugger::get_singleton()->debug(this,false);
	return true;
    } else {
	return false;
    }

}

bool LuaScriptLanguage::debug_break(const String& p_error,bool p_allow_continue) {

    if (ScriptDebugger::get_singleton() && Thread::get_caller_ID()==Thread::get_main_ID()) {

	_debug_parse_err_line=-1;
	_debug_parse_err_file="";
	_debug_error=p_error;
	ScriptDebugger::get_singleton()->debug(this,p_allow_continue);
	return true;
    } else {
	return false;
    }

}

String LuaScriptLanguage::debug_get_error() const {

    return _debug_error;
}

int LuaScriptLanguage::debug_get_stack_level_count() const {

	if (_debug_parse_err_line>=0)
		return 1;

    lua_Debug ld;
    int depth = 0;
    for(; lua_getstack(L, depth, &ld); ++depth);
    return depth;
}

bool printFrame(lua_State *L, unsigned int level)
{
    lua_Debug ld;
    if(!lua_getstack(L, level, &ld))
        return false;

    lua_getinfo(L, "n", &ld);
    lua_getinfo(L, "S", &ld);
    lua_getinfo(L, "l", &ld);

    printf("#%d  %s %s", level, ld.name ? ld.name : "(trunk)", ld.short_src); //  ld.linedefined ? ld.source : "[string]");
    if(ld.source[0] == '@')
        printf(":%d", ld.currentline);
    printf("\n");
    return true;
}


int LuaScriptLanguage::debug_get_stack_level_line(int p_level) const {

	if (_debug_parse_err_line>=0)
		return _debug_parse_err_line;

    lua_Debug ar;
    if(lua_getstack(L, p_level, &ar))
    {
        lua_getinfo(L, "l", &ar);
        return ar.currentline;
    }
    return -1;
}

String LuaScriptLanguage::debug_get_stack_level_function(int p_level) const {

	if (_debug_parse_err_line>=0)
		return "";

    lua_Debug ar;
    if(lua_getstack(L, p_level, &ar))
    {
        lua_getinfo(L, "Snl", &ar);
        return ar.name ? ar.name : "(trunk)";
    }
    return "";
}
String LuaScriptLanguage::debug_get_stack_level_source(int p_level) const {

	if (_debug_parse_err_line>=0)
		return _debug_parse_err_file;

    lua_Debug ar;
    if(lua_getstack(L, p_level, &ar))
    {
        lua_getinfo(L, "S", &ar);
        return ar.source ? ar.source : ar.short_src;
    }
    return "";
}

static bool is_filter_locals(const char *name)
{
    if(!strcmp(name, "(*temporary)"))
        return true;
    if(!strcmp(name, "(for index)"))
        return true;
    if(!strcmp(name, "(for limit)"))
        return true;
    if(!strcmp(name, "(for step)"))
        return true;
    return false;
}

void LuaScriptLanguage::debug_get_stack_level_locals(int p_level,List<String> *p_locals, List<Variant> *p_values, int p_max_subitems,int p_max_depth) {

	if (_debug_parse_err_line>=0)
		return;

    int level = p_level;

    lua_Debug _ar;
    for(int depth = 0; lua_getstack(L, depth, &_ar); ++depth)
    {
        if((level != -1) && (level != depth))
            continue;

        int n = 1;
        const char *name = NULL;
        // get local values first
        while((name = lua_getlocal(L, &_ar, n++)) != NULL)
        {
            if(!is_filter_locals(name))
            {
                p_locals->push_back(String("[local] ") + name);
                Variant var;
                LuaInstance::l_get_variant(L, -1, var);
                if(var.get_type() == Variant::NIL || var.get_type() == Variant::OBJECT)
                {
                    lua_getglobal(L, "tostring");
                    lua_insert(L, -2);
                    lua_call(L, 1, 1);
                    var = lua_tostring(L, -1);
                }
                p_values->push_back(var);
            }
            lua_pop(L, 1);
        }
        // get upvalues
        n = 1;
        lua_getinfo(L, "f", &_ar);
        while((name = lua_getupvalue(L, -1, n++)) != NULL)
        {
            if(!is_filter_locals(name))
            {
                p_locals->push_back(String("[upvalue] ") + name);
                Variant var;
                LuaInstance::l_get_variant(L, -1, var);
                if(var.get_type() == Variant::NIL || var.get_type() == Variant::OBJECT)
                {
                    lua_getglobal(L, "tostring");
                    lua_insert(L, -2);
                    lua_call(L, 1, 1);
                    var = lua_tostring(L, -1);
                }
                p_values->push_back(var);
            }
            lua_pop(L, 1);
        }
    }
}
void LuaScriptLanguage::debug_get_stack_level_members(int p_level,List<String> *p_members, List<Variant> *p_values, int p_max_subitems,int p_max_depth) {

	if (_debug_parse_err_line>=0)
		return;

    ERR_FAIL_INDEX(p_level,_debug_call_stack_pos);
    int l = _debug_call_stack_pos - p_level -1;


    const LuaInstance *instance = _call_stack[l].instance;
    if (!instance)
    	return;

    lua_rawgeti(L, LUA_REGISTRYINDEX, instance->ref);
    if(lua_istable(L, -1))
    {
        lua_pushnil(L);
        while(lua_next(L, -2))
        {
            Variant key, value;
            LuaInstance::l_get_variant(L, -2, key);
            if(key.get_type() == Variant::NIL || key.get_type() == Variant::OBJECT)
            {
                lua_getglobal(L, "tostring");
                lua_insert(L, -2);
                lua_call(L, 1, 1);
                key = lua_tostring(L, -1);
            }
            LuaInstance::l_get_variant(L, -1, value);
            if(value.get_type() == Variant::NIL || value.get_type() == Variant::OBJECT)
            {
                lua_getglobal(L, "tostring");
                lua_insert(L, -2);
                lua_call(L, 1, 1);
                value = lua_tostring(L, -1);
            }
            if((String) key != String(".c_instance"))
            {
                p_members->push_back(key);
                p_values->push_back(value);
            }

            lua_pop(L, 1);
        }
    }
    lua_pop(L, 1);

    //Ref<LuaScript> script = instance->get_script();
    //ERR_FAIL_COND( script.is_null() );

    //for(Map<StringName,PropertyInfo>::Element *E=script->member_info.front(); E; E=E->next())
    //{
    //    const StringName& key = E->key();
    //    Variant value;
    //    if(instance->get(key, value))
    //    {
    //        p_members->push_back(key);
    //        p_values->push_back(value);
    //    }
    //}
}

void LuaScriptLanguage::debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems,int p_max_depth) {

    //no globals are really reachable in lua script
}

String LuaScriptLanguage::debug_parse_stack_level_expression(int p_level,const String& p_expression,int p_max_subitems,int p_max_depth) {

	if (_debug_parse_err_line>=0)
		return "";

    char script[4096];
#if defined(_MSC_VER)
    _snprintf(script, sizeof(script), "return %s", p_expression.utf8().get_data());
#else
    snprintf(script, sizeof(script), "return %s", p_expression.utf8().get_data());
#endif
    if(luaL_loadbuffer(L, script, strlen(script), "=(debug command)") == 0)
        lua_pcall(L, 0, 1, 0);
    const char *ret = lua_tostring(L, -1);
    if(ret == NULL)
    {
        lua_getglobal(L, "tostring");
        lua_insert(L, -2);
        lua_call(L, 1, 1);
        ret = lua_tostring(L, -1);
    }
    lua_pop(L, 1);
    return ret ? ret : "";
}

bool LuaScriptLanguage::hitBreakPoint(lua_State *L, lua_Debug *ar)
{
    ScriptDebugger *deb = ScriptDebugger::get_singleton();
    if(deb == NULL || !deb->has_breakpoint())
        return false;

    lua_getinfo(L, "S", ar);
    lua_getinfo(L, "l", ar);

    return deb->is_breakpoint(ar->currentline, ar->source);
}

void LuaScriptLanguage::onHook(lua_State *L, lua_Debug *ar)
{
    ScriptDebugger *deb = ScriptDebugger::get_singleton();
    //mAr = ar;
    bool hitbp = hitBreakPoint(L, ar);
    if(L != this->L)
    {
        if(_debug_in_coroutine)
            return;

        switch(ar->event)
        {
        case LUA_HOOKCALL:
            if(_debug_break_level <= _debug_running_level)
            {
                _debug_in_coroutine = true;
                return;
            }
        case LUA_HOOKRET:
            {
                this->L = L;
                // step in
                lua_sethook(L, hookRoutine, LUA_MASKLINE | LUA_MASKCALL | LUA_MASKRET, 0);
                _debug_running_level = debug_get_stack_level_count();
                _debug_break_level = INT_MAX;
            }
            break;
        default:
            printf("Invalid hook event %d when switching coroutine.\n", ar->event);
            return;
        }
        if(!hitbp)
            return;
    }
    else
    {
        if(_debug_break_level != -1)
        {
            switch(ar->event)
            {
            case LUA_HOOKCALL:
                //_debug_running_level = lua_get_stackdepth(L);
                //if(_debug_break_level < _debug_running_level)
                //    //lua_sethook(L, HookRoutine, LUA_MASKCALL | LUA_MASKRET | (hasBreakPoint() ? LUA_MASKLINE : 0), 0);
                //    lua_sethook(L, HookRoutine, LUA_MASKCALL | LUA_MASKRET | LUA_MASKLINE, 0);
                return;
            case LUA_HOOKRET:
                if(_debug_in_coroutine)
                    _debug_in_coroutine = false;
                //_debug_running_level = lua_get_stackdepth(L) - 1;
                //if(_debug_break_level >= _debug_running_level)
                //    lua_sethook(L, HookRoutine, LUA_MASKCALL | LUA_MASKRET | LUA_MASKLINE, 0);
                return;
            case LUA_HOOKTAILRET:
            case LUA_HOOKLINE:
                deb->line_poll();
                break;
            default:
                printf("Invalid event %d when hook line.\n", ar->event);
                return;
            }
            _debug_running_level = debug_get_stack_level_count();

            if(!hitbp && (_debug_break_level >= 0) && (_debug_break_level < _debug_running_level))
                return;

            _debug_running_level = _debug_break_level = -1;
        }
        else
        {
            if(!hitbp || ar->event != LUA_HOOKLINE)
                return;
        }
    }

    if(!deb->has_breakpoint())
        lua_sethook(L, hookRoutine, 0, 0);

    debug_break("Breakpoint",true);
}

void LuaScriptLanguage::hookRoutine(lua_State *L, lua_Debug *ar)
{
    LuaScriptLanguage::get_singleton()->onHook(L, ar);
}

void LuaScriptLanguage::debug_status_changed()
{
    ScriptDebugger *deb = ScriptDebugger::get_singleton();
    if(deb == NULL)
        return;

    if(deb->get_lines_left() == 1)
    {
        if(deb->get_depth() == -1)
        {
            // step in
            lua_sethook(L, hookRoutine, LUA_MASKLINE | LUA_MASKCALL | LUA_MASKRET, 0);
            _debug_running_level = debug_get_stack_level_count();
            _debug_break_level = INT_MAX;
        }
        else if(deb->get_depth() == 0)
        {
            // next
            lua_sethook(L, hookRoutine, LUA_MASKLINE | LUA_MASKCALL | LUA_MASKRET, 0);
            _debug_running_level = debug_get_stack_level_count();
            _debug_break_level = _debug_running_level;
        }
    }
    else if(deb->has_breakpoint())
    {
        lua_sethook(L, hookRoutine, LUA_MASKLINE | LUA_MASKRET | LUA_MASKLINE, 0);
    }
    else
    {
        lua_sethook(L, hookRoutine, 0, 0);

        _debug_in_coroutine = false;
        _debug_running_level = -1;
        _debug_break_level = -1;
    }
}

void LuaScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("lua");
}


void LuaScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {


//	for(int i=0;i<GDFunctions::FUNC_MAX;i++) {
//
//		p_functions->push_back(GDFunctions::get_info(GDFunctions::Function(i)));
//	}
}

void LuaScriptLanguage::get_public_constants(List<Pair<String,Variant> > *p_constants) const {

//	Pair<String,Variant> pi;
//	pi.first="PI";
//	pi.second=Math_PI;
//	p_constants->push_back(pi);
}

String LuaScriptLanguage::make_function(const String& p_class,const String& p_name,const StringArray& p_args) const {

    String s="function node:"+p_name+"(";
	if (p_args.size()) {
		s+=" ";
		for(int i=0;i<p_args.size();i++) {
			if (i>0)
				s+=", ";
			s+=p_args[i];
		}
		s+=" ";
	}
	s+=")\n\t-- replace with function body\nend\n";

	return s;

}

//static void _parse_native_symbols(const StringName& p_native,bool p_static,List<String>* r_options) {
//
//	if (!p_static) {
//		List<MethodInfo> methods;
//		ObjectTypeDB::get_method_list(p_native,&methods);
//		for(List<MethodInfo>::Element *E=methods.front();E;E=E->next()) {
//			if (!E->get().name.begins_with("_")) {
//				r_options->push_back(E->get().name);
//			}
//		}
//	}
//
//	List<String> constants;
//	ObjectTypeDB::get_integer_constant_list(p_native,&constants);
//
//	for(List<String>::Element *E=constants.front();E;E=E->next()) {
//		r_options->push_back(E->get());
//	}
//
//}
//
//
//static bool _parse_script_symbols(const Ref<GDScript>& p_script,bool p_static,List<String>* r_options,List<String>::Element *p_indices);
//
//
//static bool _parse_completion_variant(const Variant& p_var,List<String>* r_options,List<String>::Element *p_indices) {
//
//	if (p_indices) {
//
//		bool ok;
//		Variant si = p_var.get(p_indices->get(),&ok);
//		if (!ok)
//			return false;
//		return _parse_completion_variant(si,r_options,p_indices->next());
//	} else {
//
//		switch(p_var.get_type()) {
//
//
//			case Variant::DICTIONARY: {
//
//				Dictionary d=p_var;
//				List<Variant> vl;
//				d.get_key_list(&vl);
//				for (List<Variant>::Element *E=vl.front();E;E=E->next()) {
//
//					if (E->get().get_type()==Variant::STRING)
//						r_options->push_back(E->get());
//				}
//
//
//				List<MethodInfo> ml;
//				p_var.get_method_list(&ml);
//				for(List<MethodInfo>::Element *E=ml.front();E;E=E->next()) {
//					r_options->push_back(E->get().name);
//				}
//
//			} break;
//			case Variant::OBJECT: {
//
//
//				Object *o=p_var;
//				if (o) {
//					print_line("OBJECT: "+o->get_type());
//					if (p_var.is_ref() && o->cast_to<GDScript>()) {
//
//						Ref<GDScript> gds = p_var;
//						_parse_script_symbols(gds,true,r_options,NULL);
//					} else if (o->is_type("GDNativeClass")){
//
//						GDNativeClass *gnc = o->cast_to<GDNativeClass>();
//						_parse_native_symbols(gnc->get_name(),false,r_options);
//					} else {
//
//						print_line("REGULAR BLEND");
//						_parse_native_symbols(o->get_type(),false,r_options);
//					}
//				}
//
//			} break;
//			default: {
//
//				List<PropertyInfo> pi;
//				p_var.get_property_list(&pi);
//				for(List<PropertyInfo>::Element *E=pi.front();E;E=E->next()) {
//					r_options->push_back(E->get().name);
//				}
//				List<StringName> cl;
//
//				p_var.get_numeric_constants_for_type(p_var.get_type(),&cl);
//				for(List<StringName>::Element *E=cl.front();E;E=E->next()) {
//					r_options->push_back(E->get());
//				}
//
//				List<MethodInfo> ml;
//				p_var.get_method_list(&ml);
//				for(List<MethodInfo>::Element *E=ml.front();E;E=E->next()) {
//					r_options->push_back(E->get().name);
//				}
//
//			} break;
//		}
//
//		return true;
//	}
//
//
//}
//
//
//static void _parse_expression_node(const GDParser::Node *p_node,List<String>* r_options,List<String>::Element *p_indices) {
//
//
//
//	if (p_node->type==GDParser::Node::TYPE_CONSTANT) {
//
//		const GDParser::ConstantNode *cn=static_cast<const GDParser::ConstantNode *>(p_node);
//		_parse_completion_variant(cn->value,r_options,p_indices?p_indices->next():NULL);
//	} else if (p_node->type==GDParser::Node::TYPE_DICTIONARY) {
//
//		const GDParser::DictionaryNode *dn=static_cast<const GDParser::DictionaryNode*>(p_node);
//		for(int i=0;i<dn->elements.size();i++) {
//
//			if (dn->elements[i].key->type==GDParser::Node::TYPE_CONSTANT) {
//
//				const GDParser::ConstantNode *cn=static_cast<const GDParser::ConstantNode *>(dn->elements[i].key);
//				if (cn->value.get_type()==Variant::STRING) {
//
//					String str=cn->value;
//					if (p_indices) {
//
//						if (str==p_indices->get()) {
//							_parse_expression_node(dn->elements[i].value,r_options,p_indices->next());
//							return;
//						}
//
//					} else {
//						r_options->push_back(str);
//					}
//				}
//			}
//		}
//	}
//}
//
//static bool _parse_completion_block(const GDParser::BlockNode *p_block,int p_line,List<String>* r_options,List<String>::Element *p_indices) {
//
//	for(int i=0;i<p_block->sub_blocks.size();i++) {
//		//parse inner first
//		if (p_line>=p_block->sub_blocks[i]->line && (p_line<=p_block->sub_blocks[i]->end_line || p_block->sub_blocks[i]->end_line==-1)) {
//			if (_parse_completion_block(p_block->sub_blocks[i],p_line,r_options,p_indices))
//				return true;
//		}
//	}
//
//	if (p_indices) {
//
//		//parse indices in expressions :|
//		for (int i=0;i<p_block->statements.size();i++) {
//
//			if (p_block->statements[i]->line>p_line)
//				break;
//
//			if (p_block->statements[i]->type==GDParser::BlockNode::TYPE_LOCAL_VAR) {
//
//				const GDParser::LocalVarNode *lv=static_cast<const GDParser::LocalVarNode *>(p_block->statements[i]);
//				if (lv->assign && String(lv->name)==p_indices->get()) {
//
//					_parse_expression_node(lv->assign,r_options,p_indices->next());
//					return true;
//				}
//			}
//		}
//
//	} else {
//		for(int i=0;i<p_block->variables.size();i++) {
//			//parse variables second
//			if (p_line>=p_block->variable_lines[i]) {
//				r_options->push_back(p_block->variables[i]);
//			}
//			else break;
//
//		}
//	}
//
//	return false;
//}
//
//
//static bool _parse_script_symbols(const Ref<GDScript>& p_script,bool p_static,List<String>* r_options,List<String>::Element *p_indices) {
//
//	//for (Map<StringName,Ref<GDScript> >::Element ?
//
//	if (!p_static && !p_indices) {
//		for(const Set<StringName>::Element *E=p_script->get_members().front();E;E=E->next()) {
//
//			r_options->push_back(E->get());
//		}
//	}
//
//	for (const Map<StringName,Variant >::Element *E=p_script->get_constants().front();E;E=E->next()) {
//
//		if( p_indices) {
//			if (p_indices->get()==String(E->get())) {
//				_parse_completion_variant(E->get(),r_options,p_indices->next());
//				return true;
//			}
//		} else {
//			r_options->push_back(E->key());
//		}
//	}
//
//	if (!p_indices){
//		for (const Map<StringName,GDFunction>::Element *E=p_script->get_member_functions().front();E;E=E->next()) {
//
//			if (E->get().is_static() || !p_static)
//				r_options->push_back(E->key());
//		}
//	}
//
//	if (p_script->get_base().is_valid()){
//		if (_parse_script_symbols(p_script->get_base(),p_static,r_options,p_indices))
//			return true;
//	} else if (p_script->get_native().is_valid() && !p_indices) {
//		_parse_native_symbols(p_script->get_native()->get_name(),p_static,r_options);
//	}
//
//	return false;
//}
//
//
//static bool _parse_completion_class(const String& p_base_path,const GDParser::ClassNode *p_class,int p_line,List<String>* r_options,List<String>::Element *p_indices) {
//
//
//	static const char*_type_names[Variant::VARIANT_MAX]={
//		"null","bool","int","float","String","Vector2","Rect2","Vector3","Matrix32","Plane","Quat","AABB","Matrix3","Trasnform",
//		"Color","Image","NodePath","RID","Object","InputEvent","Dictionary","Array","RawArray","IntArray","FloatArray","StringArray",
//		"Vector2Array","Vector3Array","ColorArray"};
//
//	if (p_indices && !p_indices->next()) {
//		for(int i=0;i<Variant::VARIANT_MAX;i++) {
//
//			if (p_indices->get()==_type_names[i]) {
//
//				List<StringName> ic;
//
//				Variant::get_numeric_constants_for_type(Variant::Type(i),&ic);
//				for(List<StringName>::Element *E=ic.front();E;E=E->next()) {
//					r_options->push_back(E->get());
//				}
//				return true;
//			}
//		}
//	}
//
//
//
//	for(int i=0;i<p_class->subclasses.size();i++) {
//
//		if (p_line>=p_class->subclasses[i]->line && (p_line<=p_class->subclasses[i]->end_line || p_class->subclasses[i]->end_line==-1)) {
//
//			if (_parse_completion_class(p_base_path,p_class->subclasses[i],p_line,r_options,p_indices))
//				return true;
//		}
//	}
//
//	bool in_static_func=false;
//
//	for(int i=0;i<p_class->functions.size();i++) {
//
//		const GDParser::FunctionNode *fu = p_class->functions[i];
//
//		if (p_line>=fu->body->line && (p_line<=fu->body->end_line || fu->body->end_line==-1)) {
//			//if in function, first block stuff from outer to inner
//			if (_parse_completion_block(fu->body,p_line,r_options,p_indices))
//				return true;
//			//then function arguments
//			if (!p_indices) {
//				for(int j=0;j<fu->arguments.size();j++) {
//
//					r_options->push_back(fu->arguments[j]);
//				}
//			}
//		}
//
//	}
//
//	for(int i=0;i<p_class->static_functions.size();i++) {
//
//		const GDParser::FunctionNode *fu = p_class->static_functions[i];
//
//		if (p_line>=fu->body->line && (p_line<=fu->body->end_line || fu->body->end_line==-1)) {
//
//			//if in function, first block stuff from outer to inne
//			if (_parse_completion_block(fu->body,p_line,r_options,p_indices))
//				return true;
//			//then function arguments
//			if (!p_indices) {
//				for(int j=0;j<fu->arguments.size();j++) {
//
//					r_options->push_back(fu->arguments[j]);
//				}
//			}
//
//			in_static_func=true;
//		}
//
//	}
//
//
//	//add all local names
//	if (!p_indices) {
//
//		if (!in_static_func) {
//
//			for(int i=0;i<p_class->variables.size();i++) {
//
//				r_options->push_back(p_class->variables[i].identifier);
//			}
//		}
//
//		for(int i=0;i<p_class->constant_expressions.size();i++) {
//
//			r_options->push_back(p_class->constant_expressions[i].identifier);
//		}
//
//		if (!in_static_func) {
//			for(int i=0;i<p_class->functions.size();i++) {
//
//				r_options->push_back(p_class->functions[i]->name);
//			}
//		}
//
//		for(int i=0;i<p_class->static_functions.size();i++) {
//
//			r_options->push_back(p_class->static_functions[i]->name);
//		}
//	}
//
//
//	if (p_class->extends_used) {
//		//do inheritance
//		String path = p_class->extends_file;
//
//		Ref<GDScript> script;
//		Ref<GDNativeClass> native;
//
//		if (path!="") {
//			//path (and optionally subclasses)
//
//			script = ResourceLoader::load(path);
//			if (script.is_null()) {
//				return false;
//			}
//
//			if (p_class->extends_class.size()) {
//
//				for(int i=0;i<p_class->extends_class.size();i++) {
//
//					String sub = p_class->extends_class[i];
//					if (script->get_subclasses().has(sub)) {
//
//						script=script->get_subclasses()[sub];
//					} else {
//
//						return false;
//					}
//				}
//			}
//
//		} else {
//
//			ERR_FAIL_COND_V(p_class->extends_class.size()==0,false);
//			//look around for the subclasses
//
//			String base=p_class->extends_class[0];
//			Ref<GDScript> base_class;
//#if 0
//			while(p) {
//
//				if (p->subclasses.has(base)) {
//
//					base_class=p->subclasses[base];
//					break;
//				}
//				p=p->_owner;
//			}
//
//			if (base_class.is_valid()) {
//
//				for(int i=1;i<p_class->extends_class.size();i++) {
//
//					String subclass=p_class->extends_class[i];
//
//					if (base_class->subclasses.has(subclass)) {
//
//						base_class=base_class->subclasses[subclass];
//					} else {
//
//						_set_error("Could not find subclass: "+subclass,p_class);
//						return ERR_FILE_NOT_FOUND;
//					}
//				}
//
//
//			} else {
//#endif
//				if (p_class->extends_class.size()>1) {
//
//					return false;
//
//				}
//				//if not found, try engine classes
//				if (!LuaScriptLanguage::get_singleton()->get_global_map().has(base)) {
//					return false;
//				}
//
//				int base_idx = LuaScriptLanguage::get_singleton()->get_global_map()[base];
//				native = LuaScriptLanguage::get_singleton()->get_global_array()[base_idx];
//				if (!native.is_valid()) {
//					return false;
//				}
//#if 0
//			}
//#endif
//
//		}
//
//		if (script.is_valid()) {
//			if (_parse_script_symbols(script,in_static_func,r_options,p_indices))
//				return true;
//
//		} else if (native.is_valid() && !p_indices) {
//
//			_parse_native_symbols(native->get_name(),in_static_func,r_options);
//		}
//	}
//
//	return false;
//
//}
//
//
Error LuaScriptLanguage::complete_keyword(const String& p_code, int p_line, const String& p_base_path, const String& p_base, List<String>* r_options) {

//	GDParser p;
//	Error err = p.parse(p_code,p_base_path);
//	// don't care much about error I guess
//	const GDParser::Node* root = p.get_parse_tree();
//	ERR_FAIL_COND_V(root->type!=GDParser::Node::TYPE_CLASS,ERR_INVALID_DATA);
//
//	const GDParser::ClassNode *cl = static_cast<const GDParser::ClassNode*>(root);
//
//	List<String> indices;
//	Vector<String> spl = p_base.split(".");
//
//	for(int i=0;i<spl.size()-1;i++) {
//		indices.push_back(spl[i]);
//	}
//
//	if (_parse_completion_class(p_base,cl,p_line,r_options,indices.front()))
//		return OK;
//	//and the globals x_x?
//	for(Map<StringName,int>::Element *E=globals.front();E;E=E->next()) {
//		if (!indices.empty()) {
//			if (String(E->key())==indices.front()->get()) {
//
//				_parse_completion_variant(global_array[E->get()],r_options,indices.front()->next());
//
//				return OK;
//			}
//		} else {
//			r_options->push_back(E->key());
//		}
//	}

	return OK;
}

