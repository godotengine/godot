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
#include "gd_script.h"
#include "gd_compiler.h"


void GDScriptLanguage::get_comment_delimiters(List<String> *p_delimiters) const {

	p_delimiters->push_back("#");
	p_delimiters->push_back("\"\"\"");

}
void GDScriptLanguage::get_string_delimiters(List<String> *p_delimiters) const {

	p_delimiters->push_back("\" \"");
	p_delimiters->push_back("' '");


}
String GDScriptLanguage::get_template(const String& p_class_name, const String& p_base_class_name) const {

	String _template = String()+
	"\nextends %BASE%\n\n"+
	"# member variables here, example:\n"+
	"# var a=2\n"+
	"# var b=\"textvar\"\n\n"+
	"func _ready():\n"+
	"\t# Initalization here\n"+
	"\tpass\n"+
	"\n"+
	"\n";

	return _template.replace("%BASE%",p_base_class_name);
}




bool GDScriptLanguage::validate(const String& p_script, int &r_line_error,int &r_col_error,String& r_test_error, const String& p_path,List<String> *r_functions) const {

	GDParser parser;

	Error err = parser.parse(p_script,p_path.get_base_dir(),true,p_path);
	if (err) {
		r_line_error=parser.get_error_line();
		r_col_error=parser.get_error_column();
		r_test_error=parser.get_error();
		return false;
	} else {

		const GDParser::Node *root = parser.get_parse_tree();
		ERR_FAIL_COND_V(root->type!=GDParser::Node::TYPE_CLASS,false);

		const GDParser::ClassNode *cl = static_cast<const GDParser::ClassNode*>(root);
		Map<int,String> funcs;
		for(int i=0;i<cl->functions.size();i++) {

			funcs[cl->functions[i]->line]=cl->functions[i]->name;
		}

		for(int i=0;i<cl->static_functions.size();i++) {

			funcs[cl->static_functions[i]->line]=cl->static_functions[i]->name;
		}

		for (Map<int,String>::Element *E=funcs.front();E;E=E->next()) {

			r_functions->push_back(E->get()+":"+itos(E->key()));
		}


	}

	return true;
}

bool GDScriptLanguage::has_named_classes() const {

	return false;
}

int GDScriptLanguage::find_function(const String& p_function,const String& p_code) const {

	GDTokenizerText tokenizer;
	tokenizer.set_code(p_code);
	int indent=0;
	while(tokenizer.get_token()!=GDTokenizer::TK_EOF && tokenizer.get_token()!=GDTokenizer::TK_ERROR) {

		if (tokenizer.get_token()==GDTokenizer::TK_NEWLINE) {
			indent=tokenizer.get_token_line_indent();
		}
		//print_line("TOKEN: "+String(GDTokenizer::get_token_name(tokenizer.get_token())));
		if (indent==0 && tokenizer.get_token()==GDTokenizer::TK_PR_FUNCTION && tokenizer.get_token(1)==GDTokenizer::TK_IDENTIFIER) {

			String identifier = tokenizer.get_token_identifier(1);
			if (identifier==p_function) {
				return tokenizer.get_token_line();
			}
		}
		tokenizer.advance();
		//print_line("NEXT: "+String(GDTokenizer::get_token_name(tokenizer.get_token())));

	}
	return -1;
}

Script *GDScriptLanguage::create_script() const {

	return memnew( GDScript );
}

/* DEBUGGER FUNCTIONS */


bool GDScriptLanguage::debug_break_parse(const String& p_file, int p_line,const String& p_error) {
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

bool GDScriptLanguage::debug_break(const String& p_error,bool p_allow_continue) {

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

String GDScriptLanguage::debug_get_error() const {

    return _debug_error;
}

int GDScriptLanguage::debug_get_stack_level_count() const {

	if (_debug_parse_err_line>=0)
		return 1;


	return _debug_call_stack_pos;
}
int GDScriptLanguage::debug_get_stack_level_line(int p_level) const {

	if (_debug_parse_err_line>=0)
		return _debug_parse_err_line;

    ERR_FAIL_INDEX_V(p_level,_debug_call_stack_pos,-1);

    int l = _debug_call_stack_pos - p_level -1;

    return *(_call_stack[l].line);

}
String GDScriptLanguage::debug_get_stack_level_function(int p_level) const {

	if (_debug_parse_err_line>=0)
		return "";

    ERR_FAIL_INDEX_V(p_level,_debug_call_stack_pos,"");
    int l = _debug_call_stack_pos - p_level -1;
    return _call_stack[l].function->get_name();
}
String GDScriptLanguage::debug_get_stack_level_source(int p_level) const {

	if (_debug_parse_err_line>=0)
		return _debug_parse_err_file;

    ERR_FAIL_INDEX_V(p_level,_debug_call_stack_pos,"");
    int l = _debug_call_stack_pos - p_level -1;
    return _call_stack[l].function->get_script()->get_path();

}
void GDScriptLanguage::debug_get_stack_level_locals(int p_level,List<String> *p_locals, List<Variant> *p_values, int p_max_subitems,int p_max_depth) {

	if (_debug_parse_err_line>=0)
		return;

    ERR_FAIL_INDEX(p_level,_debug_call_stack_pos);
    int l = _debug_call_stack_pos - p_level -1;

    GDFunction *f = _call_stack[l].function;

    List<Pair<StringName,int> > locals;

    f->debug_get_stack_member_state(*_call_stack[l].line,&locals);
    for( List<Pair<StringName,int> >::Element *E = locals.front();E;E=E->next() ) {

	p_locals->push_back(E->get().first);
	p_values->push_back(_call_stack[l].stack[E->get().second]);
    }

}
void GDScriptLanguage::debug_get_stack_level_members(int p_level,List<String> *p_members, List<Variant> *p_values, int p_max_subitems,int p_max_depth) {

	if (_debug_parse_err_line>=0)
		return;

    ERR_FAIL_INDEX(p_level,_debug_call_stack_pos);
    int l = _debug_call_stack_pos - p_level -1;


    GDInstance *instance = _call_stack[l].instance;

    if (!instance)
	return;

    Ref<GDScript> script = instance->get_script();
    ERR_FAIL_COND( script.is_null() );


    const Map<StringName,GDScript::MemberInfo>& mi = script->debug_get_member_indices();

    for(const Map<StringName,GDScript::MemberInfo>::Element *E=mi.front();E;E=E->next()) {

	p_members->push_back(E->key());
	p_values->push_back( instance->debug_get_member_by_index(E->get().index));
    }

}
void GDScriptLanguage::debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems,int p_max_depth) {

    //no globals are really reachable in gdscript
}
String GDScriptLanguage::debug_parse_stack_level_expression(int p_level,const String& p_expression,int p_max_subitems,int p_max_depth) {

	if (_debug_parse_err_line>=0)
		return "";
	return "";
}

void GDScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("gd");
}


void GDScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {


	for(int i=0;i<GDFunctions::FUNC_MAX;i++) {

		p_functions->push_back(GDFunctions::get_info(GDFunctions::Function(i)));
	}
}

void GDScriptLanguage::get_public_constants(List<Pair<String,Variant> > *p_constants) const {

	Pair<String,Variant> pi;
	pi.first="PI";
	pi.second=Math_PI;
	p_constants->push_back(pi);
}

String GDScriptLanguage::make_function(const String& p_class,const String& p_name,const StringArray& p_args) const {

	String s="func "+p_name+"(";
	if (p_args.size()) {
		s+=" ";
		for(int i=0;i<p_args.size();i++) {
			if (i>0)
				s+=", ";
			s+=p_args[i];
		}
		s+=" ";
	}
	s+="):\n\tpass # replace with function body\n";

	return s;

}

static void _parse_native_symbols(const StringName& p_native,bool p_static,List<String>* r_options) {

	if (!p_static) {
		List<MethodInfo> methods;
		ObjectTypeDB::get_method_list(p_native,&methods);
		for(List<MethodInfo>::Element *E=methods.front();E;E=E->next()) {
			if (!E->get().name.begins_with("_")) {
				r_options->push_back(E->get().name);
			}
		}
	}

	List<String> constants;
	ObjectTypeDB::get_integer_constant_list(p_native,&constants);

	for(List<String>::Element *E=constants.front();E;E=E->next()) {
		r_options->push_back(E->get());
	}

}


static bool _parse_script_symbols(const Ref<GDScript>& p_script,bool p_static,List<String>* r_options,List<String>::Element *p_indices);


static bool _parse_completion_variant(const Variant& p_var,List<String>* r_options,List<String>::Element *p_indices) {

	if (p_indices) {

		bool ok;
		Variant si = p_var.get(p_indices->get(),&ok);
		if (!ok)
			return false;
		return _parse_completion_variant(si,r_options,p_indices->next());
	} else {

		switch(p_var.get_type()) {


			case Variant::DICTIONARY: {

				Dictionary d=p_var;
				List<Variant> vl;
				d.get_key_list(&vl);
				for (List<Variant>::Element *E=vl.front();E;E=E->next()) {

					if (E->get().get_type()==Variant::STRING)
						r_options->push_back(E->get());
				}


				List<MethodInfo> ml;
				p_var.get_method_list(&ml);
				for(List<MethodInfo>::Element *E=ml.front();E;E=E->next()) {
					r_options->push_back(E->get().name);
				}

			} break;
			case Variant::OBJECT: {


				Object *o=p_var;
				if (o) {
					print_line("OBJECT: "+o->get_type());
					if (p_var.is_ref() && o->cast_to<GDScript>()) {

						Ref<GDScript> gds = p_var;
						_parse_script_symbols(gds,true,r_options,NULL);
					} else if (o->is_type("GDNativeClass")){

						GDNativeClass *gnc = o->cast_to<GDNativeClass>();
						_parse_native_symbols(gnc->get_name(),false,r_options);
					} else {

						print_line("REGULAR BLEND");
						_parse_native_symbols(o->get_type(),false,r_options);
					}
				}

			} break;
			default: {

				List<PropertyInfo> pi;
				p_var.get_property_list(&pi);
				for(List<PropertyInfo>::Element *E=pi.front();E;E=E->next()) {
					r_options->push_back(E->get().name);
				}
				List<StringName> cl;

				p_var.get_numeric_constants_for_type(p_var.get_type(),&cl);
				for(List<StringName>::Element *E=cl.front();E;E=E->next()) {
					r_options->push_back(E->get());
				}

				List<MethodInfo> ml;
				p_var.get_method_list(&ml);
				for(List<MethodInfo>::Element *E=ml.front();E;E=E->next()) {
					r_options->push_back(E->get().name);
				}

			} break;
		}

		return true;
	}


}

struct GDCompletionIdentifier {

	StringName obj_type;
	Variant::Type type;
};


static GDCompletionIdentifier _guess_identifier_type(const GDParser::ClassNode *p_class,int p_line,const StringName& p_identifier);


static bool _guess_identifier_type_in_expression(const GDParser::ClassNode *p_class,const GDParser::Node *p_node,int p_line,GDCompletionIdentifier &r_type) {


	if (p_node->type==GDParser::Node::TYPE_CONSTANT) {

		const GDParser::ConstantNode *cn=static_cast<const GDParser::ConstantNode *>(p_node);

		r_type.type=cn->value.get_type();
		if (r_type.type==Variant::OBJECT) {
			Object *obj = cn->value;
			if (obj) {
				r_type.obj_type=obj->get_type();
			}
		}

		return true;
	} else if (p_node->type==GDParser::Node::TYPE_DICTIONARY) {

		r_type.type=Variant::DICTIONARY;
		return true;
	} else if (p_node->type==GDParser::Node::TYPE_ARRAY) {

		r_type.type=Variant::ARRAY;
		return true;

	} else if (p_node->type==GDParser::Node::TYPE_BUILT_IN_FUNCTION) {

		MethodInfo mi = GDFunctions::get_info(static_cast<const GDParser::BuiltInFunctionNode*>(p_node)->function);
		r_type.type=mi.return_val.type;
		if (mi.return_val.hint==PROPERTY_HINT_RESOURCE_TYPE) {
			r_type.obj_type=mi.return_val.hint_string;
		}
		return true;
	} else if (p_node->type==GDParser::Node::TYPE_IDENTIFIER) {


		r_type=_guess_identifier_type(p_class,p_line,static_cast<const GDParser::IdentifierNode *>(p_node)->name);
		return true;
	} else if (p_node->type==GDParser::Node::TYPE_SELF) {
		//eeh...
		return false;

	} else if (p_node->type==GDParser::Node::TYPE_OPERATOR) {
		const GDParser::OperatorNode *op = static_cast<const GDParser::OperatorNode *>(p_node);
		if (op->op==GDParser::OperatorNode::OP_CALL) {

			if (op->arguments[0]->type==GDParser::Node::TYPE_TYPE) {

				const GDParser::TypeNode *tn = static_cast<const GDParser::TypeNode *>(op->arguments[0]);
				r_type.type=tn->vtype;
				return true;
			} else if (op->arguments[0]->type==GDParser::Node::TYPE_BUILT_IN_FUNCTION) {


				const GDParser::BuiltInFunctionNode *bin = static_cast<const GDParser::BuiltInFunctionNode *>(op->arguments[0]);
				return _guess_identifier_type_in_expression(p_class,bin,p_line,r_type);

			} else if (op->arguments.size()>1 && op->arguments[1]->type==GDParser::Node::TYPE_IDENTIFIER) {

				GDCompletionIdentifier base;
				if (!_guess_identifier_type_in_expression(p_class,op->arguments[0],p_line,base))
					return false;
				StringName id = static_cast<const GDParser::IdentifierNode *>(p_node)->name;
				if (base.type==Variant::OBJECT) {

					if (ObjectTypeDB::has_method(base.obj_type,id)) {
#ifdef TOOLS_ENABLED
						MethodBind *mb = ObjectTypeDB::get_method(base.obj_type,id);
						PropertyInfo pi = mb->get_argument_info(-1);

						r_type.type=pi.type;
						if (pi.hint==PROPERTY_HINT_RESOURCE_TYPE) {
							r_type.obj_type=pi.hint_string;
						}
#else
						return false;
#endif
					} else {
						return false;
					}
				} else {
					//method for some variant..
					Variant::CallError ce;
					Variant v = Variant::construct(base.type,NULL,0,ce);
					List<MethodInfo> mi;
					v.get_method_list(&mi);
					for (List<MethodInfo>::Element *E=mi.front();E;E=E->next()) {

						if (E->get().name==id.operator String()) {

							MethodInfo mi = E->get();
							r_type.type=mi.return_val.type;
							if (mi.return_val.hint==PROPERTY_HINT_RESOURCE_TYPE) {
								r_type.obj_type=mi.return_val.hint_string;
							}
							return true;
						}
					}

				}


			}
		} else {

			Variant::Operator vop = Variant::OP_MAX;
			switch(op->op) {
				case GDParser::OperatorNode::OP_ASSIGN_ADD: vop=Variant::OP_ADD; break;
				case GDParser::OperatorNode::OP_ASSIGN_SUB: vop=Variant::OP_SUBSTRACT; break;
				case GDParser::OperatorNode::OP_ASSIGN_MUL: vop=Variant::OP_MULTIPLY; break;
				case GDParser::OperatorNode::OP_ASSIGN_DIV: vop=Variant::OP_DIVIDE; break;
				case GDParser::OperatorNode::OP_ASSIGN_MOD: vop=Variant::OP_MODULE; break;
				case GDParser::OperatorNode::OP_ASSIGN_SHIFT_LEFT: vop=Variant::OP_SHIFT_LEFT; break;
				case GDParser::OperatorNode::OP_ASSIGN_SHIFT_RIGHT: vop=Variant::OP_SHIFT_RIGHT; break;
				case GDParser::OperatorNode::OP_ASSIGN_BIT_AND: vop=Variant::OP_BIT_AND; break;
				case GDParser::OperatorNode::OP_ASSIGN_BIT_OR: vop=Variant::OP_BIT_OR; break;
				case GDParser::OperatorNode::OP_ASSIGN_BIT_XOR: vop=Variant::OP_BIT_XOR; break;
				default:{}

			}

			if (vop==Variant::OP_MAX)
				return false;

			GDCompletionIdentifier p1;
			GDCompletionIdentifier p2;

			if (op->arguments[0]) {
				if (!_guess_identifier_type_in_expression(p_class,op->arguments[0],p_line,p1))
					return false;
			}

			if (op->arguments.size()>1) {
				if (!_guess_identifier_type_in_expression(p_class,op->arguments[1],p_line,p2))
					return false;
			}

			Variant::CallError ce;
			Variant v1 = Variant::construct(p1.type,NULL,0,ce);
			Variant v2 = Variant::construct(p2.type,NULL,0,ce);
			// avoid potential invalid ops
			if ((vop==Variant::OP_DIVIDE || vop==Variant::OP_MODULE) && v2.get_type()==Variant::INT) {
				v2=1;
			}
			if (vop==Variant::OP_DIVIDE && v2.get_type()==Variant::REAL) {
				v2=1.0;
			}

			Variant r;
			bool valid;
			Variant::evaluate(vop,v1,v2,r,valid);
			if (!valid)
				return false;
			r_type.type=r.get_type();
			return true;

		}

	}

	return false;
}

static bool _guess_identifier_type_in_block(const GDParser::ClassNode *p_class,const GDParser::BlockNode *p_block,int p_line,const StringName& p_identifier,GDCompletionIdentifier &r_type) {


	for(int i=0;i<p_block->sub_blocks.size();i++) {
		//parse inner first
		if (p_line>=p_block->sub_blocks[i]->line && (p_line<=p_block->sub_blocks[i]->end_line || p_block->sub_blocks[i]->end_line==-1)) {
			if (_guess_identifier_type_in_block(p_class,p_block->sub_blocks[i],p_line,p_identifier,r_type))
				return true;
		}
	}



	const GDParser::Node *last_assign=NULL;
	int last_assign_line=-1;

	for (int i=0;i<p_block->statements.size();i++) {

		if (p_block->statements[i]->line>p_line)
			break;


		if (p_block->statements[i]->type==GDParser::BlockNode::TYPE_LOCAL_VAR) {

			const GDParser::LocalVarNode *lv=static_cast<const GDParser::LocalVarNode *>(p_block->statements[i]);
			if (lv->assign && lv->name==p_identifier) {
				last_assign=lv->assign;
				last_assign_line=p_block->statements[i]->line;
			}
		}

		if (p_block->statements[i]->type==GDParser::BlockNode::TYPE_OPERATOR) {
			const GDParser::OperatorNode *op = static_cast<const GDParser::OperatorNode *>(p_block->statements[i]);
			if (op->op==GDParser::OperatorNode::OP_ASSIGN) {

				if (op->arguments.size() && op->arguments[0]->type==GDParser::Node::TYPE_IDENTIFIER) {
					const GDParser::IdentifierNode *id = static_cast<const GDParser::IdentifierNode *>(op->arguments[0]);
					if (id->name==p_identifier) {
						last_assign=op->arguments[1];
						last_assign_line=p_block->statements[i]->line;
					}
				}
			}
		}
	}

	//use the last assignment, (then backwards?)
	if (last_assign) {
		return _guess_identifier_type_in_expression(p_class,last_assign,last_assign_line-1,r_type);
	}

	return false;
}

static GDCompletionIdentifier _guess_identifier_type(const GDParser::ClassNode *p_class,int p_line,const StringName& p_identifier) {


	return GDCompletionIdentifier();
}

static void _parse_expression_node(const GDParser::ClassNode *p_class,const GDParser::Node *p_node,int p_line,List<String>* r_options,List<String>::Element *p_indices) {



	if (p_node->type==GDParser::Node::TYPE_CONSTANT) {

		const GDParser::ConstantNode *cn=static_cast<const GDParser::ConstantNode *>(p_node);
		_parse_completion_variant(cn->value,r_options,p_indices?p_indices->next():NULL);
	} else if (p_node->type==GDParser::Node::TYPE_DICTIONARY) {

		const GDParser::DictionaryNode *dn=static_cast<const GDParser::DictionaryNode*>(p_node);
		for (int i=0;i<dn->elements.size();i++) {

			if (dn->elements[i].key->type==GDParser::Node::TYPE_CONSTANT) {

				const GDParser::ConstantNode *cn=static_cast<const GDParser::ConstantNode *>(dn->elements[i].key);
				if (cn->value.get_type()==Variant::STRING) {

					String str=cn->value;
					if (p_indices) {

						if (str==p_indices->get()) {
							_parse_expression_node(p_class,dn->elements[i].value,p_line,r_options,p_indices->next());
							return;
						}

					} else {
						r_options->push_back(str);
					}
				}
			}
		}
	} else if (p_node->type==GDParser::Node::TYPE_BUILT_IN_FUNCTION) {

		MethodInfo mi = GDFunctions::get_info(static_cast<const GDParser::BuiltInFunctionNode*>(p_node)->function);

		Variant::CallError ce;
		_parse_completion_variant(Variant::construct(mi.return_val.type,NULL,0,ce),r_options,p_indices?p_indices->next():NULL);
	} else if (p_node->type==GDParser::Node::TYPE_IDENTIFIER) {

		//GDCompletionIdentifier idt = _guess_identifier_type(p_class,p_line-1,static_cast<const GDParser::IdentifierNode *>(p_node)->name);
		//Variant::CallError ce;
		//_parse_completion_variant(Variant::construct(mi.return_val.type,NULL,0,ce),r_options,p_indices?p_indices->next():NULL);
	}
}

static bool _parse_completion_block(const GDParser::ClassNode *p_class,const GDParser::BlockNode *p_block,int p_line,List<String>* r_options,List<String>::Element *p_indices) {

	print_line("COMPLETION BLOCK "+itos(p_block->line)+" -> "+itos(p_block->end_line));

	for(int i=0;i<p_block->sub_blocks.size();i++) {
		//parse inner first
		if (p_line>=p_block->sub_blocks[i]->line && (p_line<=p_block->sub_blocks[i]->end_line || p_block->sub_blocks[i]->end_line==-1)) {
			if (_parse_completion_block(p_class,p_block->sub_blocks[i],p_line,r_options,p_indices))
				return true;
		}
	}

	if (p_indices) {

		//parse indices in expressions :|

		const GDParser::Node *last_assign=NULL;
		int last_assign_line=-1;

		for (int i=0;i<p_block->statements.size();i++) {

			if (p_block->statements[i]->line>p_line)
				break;


			if (p_block->statements[i]->type==GDParser::BlockNode::TYPE_LOCAL_VAR) {

				const GDParser::LocalVarNode *lv=static_cast<const GDParser::LocalVarNode *>(p_block->statements[i]);				
				if (lv->assign && String(lv->name)==p_indices->get()) {
					last_assign=lv->assign;
					last_assign_line=p_block->statements[i]->line;
				}
			}
		}

		//use the last assignment, (then backwards?)
		if (last_assign) {
			_parse_expression_node(p_class,last_assign,last_assign_line,r_options,p_indices->next());
			return true;
		}

	} else {
		//no indices, just add all variables and continue
		for(int i=0;i<p_block->variables.size();i++) {
			//parse variables second
			if (p_line>=p_block->variable_lines[i]) {
				r_options->push_back(p_block->variables[i]);
			} else break;

		}
	}

	return false;
}


static bool _parse_script_symbols(const Ref<GDScript>& p_script,bool p_static,List<String>* r_options,List<String>::Element *p_indices) {

	//for (Map<StringName,Ref<GDScript> >::Element ?

	if (!p_static && !p_indices) {
		for(const Set<StringName>::Element *E=p_script->get_members().front();E;E=E->next()) {

			r_options->push_back(E->get());
		}
	}

	for (const Map<StringName,Variant >::Element *E=p_script->get_constants().front();E;E=E->next()) {

		if( p_indices) {
			if (p_indices->get()==String(E->get())) {
				_parse_completion_variant(E->get(),r_options,p_indices->next());
				return true;
			}
		} else {
			r_options->push_back(E->key());
		}
	}

	if (!p_indices){
		for (const Map<StringName,GDFunction>::Element *E=p_script->get_member_functions().front();E;E=E->next()) {

			if (E->get().is_static() || !p_static)
				r_options->push_back(E->key());
		}
	}

	if (p_script->get_base().is_valid()){
		if (_parse_script_symbols(p_script->get_base(),p_static,r_options,p_indices))
			return true;
	} else if (p_script->get_native().is_valid() && !p_indices) {
		_parse_native_symbols(p_script->get_native()->get_name(),p_static,r_options);
	}

	return false;
}


static bool _parse_completion_class(const String& p_base_path,const GDParser::ClassNode *p_class,int p_line,List<String>* r_options,List<String>::Element *p_indices) {

	//checks known classes or built-in types for completion

	if (p_indices && !p_indices->next()) {
		//built-in types do not have sub-classes, try these first if no sub-indices exist.
		static const char*_type_names[Variant::VARIANT_MAX]={
			"null","bool","int","float","String","Vector2","Rect2","Vector3","Matrix32","Plane","Quat","AABB","Matrix3","Trasnform",
			"Color","Image","NodePath","RID","Object","InputEvent","Dictionary","Array","RawArray","IntArray","FloatArray","StringArray",
			"Vector2Array","Vector3Array","ColorArray"};

		for(int i=0;i<Variant::VARIANT_MAX;i++) {

			if (p_indices->get()==_type_names[i]) {

				List<StringName> ic;

				Variant::get_numeric_constants_for_type(Variant::Type(i),&ic);
				for(List<StringName>::Element *E=ic.front();E;E=E->next()) {
					r_options->push_back(E->get());
				}
				return true;
			}
		}
	}

	// check the sub-classes of current class

	for(int i=0;i<p_class->subclasses.size();i++) {

		if (p_line>=p_class->subclasses[i]->line && (p_line<=p_class->subclasses[i]->end_line || p_class->subclasses[i]->end_line==-1)) {
			// if OK in sub-classes, try completing the sub-class
			if (_parse_completion_class(p_base_path,p_class->subclasses[i],p_line,r_options,p_indices))
				return true;
		}
	}

	bool in_static_func=false;

	for(int i=0;i<p_class->functions.size();i++) {

		const GDParser::FunctionNode *fu = p_class->functions[i];

		if (p_line>=fu->body->line && (p_line<=fu->body->end_line || fu->body->end_line==-1)) {
			//if in function, first block stuff from outer to inner
			if (_parse_completion_block(p_class,fu->body,p_line,r_options,p_indices))
				return true;
			//then function arguments
			if (!p_indices) {
				for(int j=0;j<fu->arguments.size();j++) {

					r_options->push_back(fu->arguments[j]);
				}
			}
		}

	}

	for(int i=0;i<p_class->static_functions.size();i++) {

		const GDParser::FunctionNode *fu = p_class->static_functions[i];

		if (p_line>=fu->body->line && (p_line<=fu->body->end_line || fu->body->end_line==-1)) {

			//if in function, first block stuff from outer to inne
			if (_parse_completion_block(p_class,fu->body,p_line,r_options,p_indices))
				return true;
			//then function arguments
			if (!p_indices) {
				for(int j=0;j<fu->arguments.size();j++) {

					r_options->push_back(fu->arguments[j]);
				}
			}

			in_static_func=true;
		}

	}


	//add all local names
	if (!p_indices) {

		if (!in_static_func) {

			for(int i=0;i<p_class->variables.size();i++) {

				r_options->push_back(p_class->variables[i].identifier);
			}
		}

		for(int i=0;i<p_class->constant_expressions.size();i++) {

			r_options->push_back(p_class->constant_expressions[i].identifier);
		}

		if (!in_static_func) {
			for(int i=0;i<p_class->functions.size();i++) {

				r_options->push_back(p_class->functions[i]->name);
			}
		}

		for(int i=0;i<p_class->static_functions.size();i++) {

			r_options->push_back(p_class->static_functions[i]->name);
		}
	}


	if (p_class->extends_used) {
		//do inheritance
		String path = p_class->extends_file;

		Ref<GDScript> script;
		Ref<GDNativeClass> native;

		if (path!="") {
			//path (and optionally subclasses)

			script = ResourceLoader::load(path);
			if (script.is_null()) {
				return false;
			}

			if (p_class->extends_class.size()) {

				for(int i=0;i<p_class->extends_class.size();i++) {

					String sub = p_class->extends_class[i];
					if (script->get_subclasses().has(sub)) {

						script=script->get_subclasses()[sub];
					} else {

						return false;
					}
				}
			}

		} else {

			ERR_FAIL_COND_V(p_class->extends_class.size()==0,false);
			//look around for the subclasses

			String base=p_class->extends_class[0];
			Ref<GDScript> base_class;
#if 0
			while(p) {

				if (p->subclasses.has(base)) {

					base_class=p->subclasses[base];
					break;
				}
				p=p->_owner;
			}

			if (base_class.is_valid()) {

				for(int i=1;i<p_class->extends_class.size();i++) {

					String subclass=p_class->extends_class[i];

					if (base_class->subclasses.has(subclass)) {

						base_class=base_class->subclasses[subclass];
					} else {

						_set_error("Could not find subclass: "+subclass,p_class);
						return ERR_FILE_NOT_FOUND;
					}
				}


			} else {
#endif
				if (p_class->extends_class.size()>1) {

					return false;

				}
				//if not found, try engine classes
				if (!GDScriptLanguage::get_singleton()->get_global_map().has(base)) {
					return false;
				}

				int base_idx = GDScriptLanguage::get_singleton()->get_global_map()[base];
				native = GDScriptLanguage::get_singleton()->get_global_array()[base_idx];
				if (!native.is_valid()) {
					return false;
				}
#if 0
			}
#endif

		}

		if (script.is_valid()) {
			if (_parse_script_symbols(script,in_static_func,r_options,p_indices))
				return true;

		} else if (native.is_valid() && !p_indices) {

			_parse_native_symbols(native->get_name(),in_static_func,r_options);
		}
	}

	return false;

}


Error GDScriptLanguage::complete_keyword(const String& p_code, int p_line, const String& p_base_path, const String& p_base, List<String>* r_options) {



	GDParser p;
	Error err = p.parse(p_code,p_base_path);
	// don't care much about error I guess
	const GDParser::Node* root = p.get_parse_tree();
	ERR_FAIL_COND_V(root->type!=GDParser::Node::TYPE_CLASS,ERR_INVALID_DATA);

	print_line("BASE: "+p_base);
	const GDParser::ClassNode *cl = static_cast<const GDParser::ClassNode*>(root);

	List<String> indices;
	Vector<String> spl = p_base.split(".");

	for(int i=0;i<spl.size()-1;i++) {
		print_line("INDEX "+itos(i)+":  "+spl[i]);
		indices.push_back(spl[i]);
	}

	//parse completion inside of the class first
	if (_parse_completion_class(p_base,cl,p_line,r_options,indices.front()))
		return OK;

	//if parsing completion inside of the class fails (none found), try using globals for completion
	for(Map<StringName,int>::Element *E=globals.front();E;E=E->next()) {
		if (!indices.empty()) {
			if (String(E->key())==indices.front()->get()) {

				_parse_completion_variant(global_array[E->get()],r_options,indices.front()->next());

				return OK;
			}
		} else {
			r_options->push_back(E->key());
		}
	}

	return OK;
}

void GDScriptLanguage::auto_indent_code(String& p_code,int p_from_line,int p_to_line) const {


	Vector<String> lines = p_code.split("\n");
	List<int> indent_stack;

	for(int i=0;i<lines.size();i++) {

		String l = lines[i];
		int tc=0;
		for(int j=0;j<l.length();j++) {
			if (l[j]==' ' || l[j]=='\t') {

				tc++;
			} else {
				break;
			}
		}


		String st = l.substr(tc,l.length()).strip_edges();
		if (st=="" || st.begins_with("#"))
			continue; //ignore!

		int ilevel=0;
		if (indent_stack.size()) {
			ilevel=indent_stack.back()->get();
		}

		if (tc>ilevel) {
			indent_stack.push_back(tc);
		} else if (tc<ilevel) {
			while(indent_stack.size() && indent_stack.back()->get()>tc) {
				indent_stack.pop_back();
			}

			if (indent_stack.size() && indent_stack.back()->get()!=tc)
				indent_stack.push_back(tc); //this is not right but gets the job done
		}

		if (i>=p_from_line) {

			l="";
			for(int j=0;j<indent_stack.size();j++)
				l+="\t";
			l+=st;


		} else if (i>p_to_line) {
			break;
		}

		//print_line(itos(indent_stack.size())+","+itos(tc)+": "+l);
		lines[i]=l;
	}

	p_code="";
	for(int i=0;i<lines.size();i++) {
		if (i>0)
			p_code+="\n";
		p_code+=lines[i];
	}

}
