/*************************************************************************/
/*  shader_compiler_gles2.cpp                                            */
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
#include "shader_compiler_gles2.h"
#include "print_string.h"

#include "stdio.h"

//#define DEBUG_SHADER_ENABLED

typedef ShaderLanguage SL;

struct CodeGLSL2 {

	String code;

};

static String _mktab(int p_level) {

	String tb;
	for(int i=0;i<p_level;i++) {
		tb+="\t";
	}

	return tb;
}

static String _typestr(SL::DataType p_type) {

	switch(p_type) {

		case SL::TYPE_VOID: return "void";
		case SL::TYPE_BOOL: return "bool";
		case SL::TYPE_FLOAT: return "float";
		case SL::TYPE_VEC2: return "vec2";
		case SL::TYPE_VEC3: return "vec3";
		case SL::TYPE_VEC4: return "vec4";
		case SL::TYPE_MAT3: return "mat3";
		case SL::TYPE_MAT4: return "mat4";
		case SL::TYPE_TEXTURE: return "sampler2D";
		case SL::TYPE_CUBEMAP: return "samplerCube";
	}

	return "";
}

static String _mknum(float p_num) {
	return String::num_real(p_num);
}

static String _opstr(SL::Operator p_op) {

	switch(p_op) {
		case SL::OP_ASSIGN: return "=";
		case SL::OP_ADD: return "+";
		case SL::OP_SUB: return "-";
		case SL::OP_MUL: return "*";
		case SL::OP_DIV: return "/";
		case SL::OP_ASSIGN_ADD: return "+=";
		case SL::OP_ASSIGN_SUB: return "-=";
		case SL::OP_ASSIGN_MUL: return "*=";
		case SL::OP_ASSIGN_DIV: return "/=";
		case SL::OP_NEG: return "-";
		case SL::OP_NOT: return "!";
		case SL::OP_CMP_EQ: return "==";
		case SL::OP_CMP_NEQ: return "!=";
		case SL::OP_CMP_LEQ: return "<=";
		case SL::OP_CMP_GEQ: return ">=";
		case SL::OP_CMP_LESS: return "<";
		case SL::OP_CMP_GREATER: return ">";
		case SL::OP_CMP_OR: return "||";
		case SL::OP_CMP_AND: return "&&";
		default: return "";
	}

	return "";
}


//#ifdef DEBUG_SHADER_ENABLED
#if 1
#define ENDL "\n"
#else
#define ENDL ""
#endif



String ShaderCompilerGLES2::dump_node_code(SL::Node *p_node,int p_level,bool p_assign_left) {

	String code;

	switch(p_node->type) {

		case SL::Node::TYPE_PROGRAM: {

			SL::ProgramNode *pnode=(SL::ProgramNode*)p_node;

			code+=dump_node_code(pnode->body,p_level);
		} break;
		case SL::Node::TYPE_FUNCTION: {

		} break;
		case SL::Node::TYPE_BLOCK: {
			SL::BlockNode *bnode=(SL::BlockNode*)p_node;

			//variables
			code+="{"ENDL;
			for(Map<StringName,SL::DataType>::Element *E=bnode->variables.front();E;E=E->next()) {

				code+=_mktab(p_level)+_typestr(E->value())+" "+replace_string(E->key())+";"ENDL;
			}

			for(int i=0;i<bnode->statements.size();i++) {

				code+=_mktab(p_level)+dump_node_code(bnode->statements[i],p_level)+";"ENDL;
			}

			code+="}"ENDL;

		} break;
		case SL::Node::TYPE_VARIABLE: {
			SL::VariableNode *vnode=(SL::VariableNode*)p_node;

			if (type==ShaderLanguage::SHADER_MATERIAL_VERTEX) {

				if (vnode->name==vname_vertex && p_assign_left) {
					vertex_code_writes_vertex=true;
				}
				if (vnode->name==vname_color_interp) {
					flags->use_color_interp=true;
				}
				if (vnode->name==vname_uv_interp) {
					flags->use_uv_interp=true;
				}
				if (vnode->name==vname_uv2_interp) {
					flags->use_uv2_interp=true;
				}
				if (vnode->name==vname_var1_interp) {
					flags->use_var1_interp=true;
				}
				if (vnode->name==vname_var2_interp) {
					flags->use_var2_interp=true;
				}
				if (vnode->name==vname_tangent_interp || vnode->name==vname_binormal_interp) {
					flags->use_tangent_interp=true;
				}


			}



			if (type==ShaderLanguage::SHADER_MATERIAL_FRAGMENT) {

				if (vnode->name==vname_discard) {
					uses_discard=true;
				}
				if (vnode->name==vname_normalmap) {
					uses_normalmap=true;
				}
				if (vnode->name==vname_screen_uv) {
					uses_screen_uv=true;
				}
				if (vnode->name==vname_diffuse_alpha && p_assign_left) {
					uses_alpha=true;
				}
				if (vnode->name==vname_color_interp) {
					flags->use_color_interp=true;
				}
				if (vnode->name==vname_uv_interp) {
					flags->use_uv_interp=true;
				}
				if (vnode->name==vname_uv2_interp) {
					flags->use_uv2_interp=true;
				}
				if (vnode->name==vname_var1_interp) {
					flags->use_var1_interp=true;
				}
				if (vnode->name==vname_var2_interp) {
					flags->use_var2_interp=true;
				}
				if (vnode->name==vname_tangent_interp || vnode->name==vname_binormal_interp) {
					flags->use_tangent_interp=true;
				}

			}
			if (type==ShaderLanguage::SHADER_MATERIAL_LIGHT) {

				if (vnode->name==vname_light) {
					uses_light=true;
				}

			}
			if (type==ShaderLanguage::SHADER_CANVAS_ITEM_VERTEX) {

				if (vnode->name==vname_var1_interp) {
					flags->use_var1_interp=true;
				}
				if (vnode->name==vname_var2_interp) {
					flags->use_var2_interp=true;
				}
				if (vnode->name==vname_world_vec) {
					uses_worldvec=true;
				}

			}


			if (type==ShaderLanguage::SHADER_CANVAS_ITEM_FRAGMENT) {


				if (vnode->name==vname_texpixel_size) {
					uses_texpixel_size=true;
				}
				if (vnode->name==vname_normal) {
					uses_normal=true;
				}

				if (vnode->name==vname_screen_uv) {
					uses_screen_uv=true;
				}

				if (vnode->name==vname_var1_interp) {
					flags->use_var1_interp=true;
				}
				if (vnode->name==vname_var2_interp) {
					flags->use_var2_interp=true;
				}
			}

			if (type==ShaderLanguage::SHADER_CANVAS_ITEM_LIGHT) {

				if (vnode->name==vname_light) {
					uses_light=true;
				}

			}

			if (vnode->name==vname_time) {
				uses_time=true;
			}
			code=replace_string(vnode->name);

		} break;
		case SL::Node::TYPE_CONSTANT: {
			SL::ConstantNode *cnode=(SL::ConstantNode*)p_node;
			switch(cnode->datatype) {


				case SL::TYPE_BOOL: code=cnode->value.operator bool()?"true":"false"; break;
				case SL::TYPE_FLOAT: code=_mknum(cnode->value); break; //force zeros, so GLSL doesn't confuse with integer.
				case SL::TYPE_VEC2: { Vector2 v = cnode->value; code="vec2("+_mknum(v.x)+", "+_mknum(v.y)+")"; } break;
				case SL::TYPE_VEC3: { Vector3 v = cnode->value; code="vec3("+_mknum(v.x)+", "+_mknum(v.y)+", "+_mknum(v.z)+")"; } break;
				case SL::TYPE_VEC4: { Plane v = cnode->value; code="vec4("+_mknum(v.normal.x)+", "+_mknum(v.normal.y)+", "+_mknum(v.normal.z)+", "+_mknum(v.d)+")"; } break;
				case SL::TYPE_MAT3: { Matrix3 x = cnode->value; code="mat3( vec3("+_mknum(x.get_axis(0).x)+", "+_mknum(x.get_axis(0).y)+", "+_mknum(x.get_axis(0).z)+"), vec3("+_mknum(x.get_axis(1).x)+", "+_mknum(x.get_axis(1).y)+", "+_mknum(x.get_axis(1).z)+"), vec3("+_mknum(x.get_axis(2).x)+", "+_mknum(x.get_axis(2).y)+", "+_mknum(x.get_axis(2).z)+"))"; } break;
				case SL::TYPE_MAT4: { Transform x = cnode->value; code="mat4( vec4("+_mknum(x.basis.get_axis(0).x)+", "+_mknum(x.basis.get_axis(0).y)+", "+_mknum(x.basis.get_axis(0).z)+",0.0), vec4("+_mknum(x.basis.get_axis(1).x)+", "+_mknum(x.basis.get_axis(1).y)+", "+_mknum(x.basis.get_axis(1).z)+",0.0), vec4("+_mknum(x.basis.get_axis(2).x)+", "+_mknum(x.basis.get_axis(2).y)+", "+_mknum(x.basis.get_axis(2).z)+",0.0), vec4("+_mknum(x.origin.x)+", "+_mknum(x.origin.y)+", "+_mknum(x.origin.z)+",1.0))"; } break;
				default: code="<error: "+Variant::get_type_name(cnode->value.get_type())+" ("+itos(cnode->datatype)+">";
			}

		} break;
		case SL::Node::TYPE_OPERATOR: {
			SL::OperatorNode *onode=(SL::OperatorNode*)p_node;


			switch(onode->op) {

				case SL::OP_ASSIGN_MUL: {


					if (onode->arguments[0]->get_datatype()==SL::TYPE_VEC3 && onode->arguments[1]->get_datatype()==SL::TYPE_MAT4) {

						String mul_l=dump_node_code(onode->arguments[0],p_level,true);
						String mul_r=dump_node_code(onode->arguments[1],p_level);
						code=mul_l+"=(vec4("+mul_l+",1.0)*("+mul_r+")).xyz";
						break;
					} else if (onode->arguments[0]->get_datatype()==SL::TYPE_MAT4 && onode->arguments[1]->get_datatype()==SL::TYPE_VEC3) {

						String mul_l=dump_node_code(onode->arguments[0],p_level,true);
						String mul_r=dump_node_code(onode->arguments[1],p_level);
						code=mul_l+"=(("+mul_l+")*vec4("+mul_r+",1.0)).xyz";
						break;
					} else if (onode->arguments[0]->get_datatype()==SL::TYPE_VEC2 && onode->arguments[1]->get_datatype()==SL::TYPE_MAT4) {

						String mul_l=dump_node_code(onode->arguments[0],p_level,true);
						String mul_r=dump_node_code(onode->arguments[1],p_level);
						code=mul_l+"=(vec4("+mul_l+",0.0,1.0)*("+mul_r+")).xy";
						break;
					} else if (onode->arguments[0]->get_datatype()==SL::TYPE_MAT4 && onode->arguments[1]->get_datatype()==SL::TYPE_VEC2) {

						String mul_l=dump_node_code(onode->arguments[0],p_level,true);
						String mul_r=dump_node_code(onode->arguments[1],p_level);
						code=mul_l+"=(("+mul_l+")*vec4("+mul_r+",0.0,1.0)).xy";
						break;
					} else if (onode->arguments[0]->get_datatype()==SL::TYPE_VEC2 && onode->arguments[1]->get_datatype()==SL::TYPE_MAT3) {
						String mul_l=dump_node_code(onode->arguments[0],p_level,true);
						String mul_r=dump_node_code(onode->arguments[1],p_level);
						code=mul_l+"=(("+mul_l+")*vec3("+mul_r+",1.0)).xy";
						break;
					}


				};
				case SL::OP_ASSIGN:
				case SL::OP_ASSIGN_ADD:
				case SL::OP_ASSIGN_SUB:
				case SL::OP_ASSIGN_DIV:
					code="("+dump_node_code(onode->arguments[0],p_level,true)+_opstr(onode->op)+dump_node_code(onode->arguments[1],p_level)+")";
					break;

				case SL::OP_MUL:

					if (onode->arguments[0]->get_datatype()==SL::TYPE_MAT4 && onode->arguments[1]->get_datatype()==SL::TYPE_VEC3) {

						code="("+dump_node_code(onode->arguments[0],p_level)+"*vec4("+dump_node_code(onode->arguments[1],p_level)+",1.0)).xyz";
						break;
					} else if (onode->arguments[0]->get_datatype()==SL::TYPE_VEC3 && onode->arguments[1]->get_datatype()==SL::TYPE_MAT4) {

						code="(vec4("+dump_node_code(onode->arguments[0],p_level)+",1.0)*"+dump_node_code(onode->arguments[1],p_level)+").xyz";
						break;
					} else if (onode->arguments[0]->get_datatype()==SL::TYPE_MAT4 && onode->arguments[1]->get_datatype()==SL::TYPE_VEC2) {

						code="("+dump_node_code(onode->arguments[0],p_level)+"*vec4("+dump_node_code(onode->arguments[1],p_level)+",0.0,1.0)).xy";
						break;
					} else if (onode->arguments[0]->get_datatype()==SL::TYPE_VEC2 && onode->arguments[1]->get_datatype()==SL::TYPE_MAT4) {

						code="(vec4("+dump_node_code(onode->arguments[0],p_level)+",0.0,1.0)*"+dump_node_code(onode->arguments[1],p_level)+").xy";
						break;
					} else if (onode->arguments[0]->get_datatype()==SL::TYPE_MAT3 && onode->arguments[1]->get_datatype()==SL::TYPE_VEC2) {

						code="("+dump_node_code(onode->arguments[0],p_level)+"*vec3("+dump_node_code(onode->arguments[1],p_level)+",1.0)).xy";
						break;
					} else if (onode->arguments[0]->get_datatype()==SL::TYPE_VEC2 && onode->arguments[1]->get_datatype()==SL::TYPE_MAT3) {

						code="(vec3("+dump_node_code(onode->arguments[0],p_level)+",1.0)*"+dump_node_code(onode->arguments[1],p_level)+").xy";
						break;
					}

				case SL::OP_ADD:
				case SL::OP_SUB:
				case SL::OP_DIV:
				case SL::OP_CMP_EQ:
				case SL::OP_CMP_NEQ:
				case SL::OP_CMP_LEQ:
				case SL::OP_CMP_GEQ:
				case SL::OP_CMP_LESS:
				case SL::OP_CMP_GREATER:
				case SL::OP_CMP_OR:
				case SL::OP_CMP_AND:
					//handle binary
					code="("+dump_node_code(onode->arguments[0],p_level)+_opstr(onode->op)+dump_node_code(onode->arguments[1],p_level)+")";
					break;
				case SL::OP_NEG:
				case SL::OP_NOT:
					//handle unary
					code=_opstr(onode->op)+dump_node_code(onode->arguments[0],p_level);
					break;
				case SL::OP_CONSTRUCT:
				case SL::OP_CALL: {
					String callfunc=dump_node_code(onode->arguments[0],p_level);


					code=callfunc+"(";
					/*if (callfunc=="mat4") {
						//fix constructor for mat4
						for(int i=1;i<onode->arguments.size();i++) {
							if (i>1)
								code+=", ";
								//transform
							code+="vec4( "+dump_node_code(onode->arguments[i],p_level)+(i==4?",1.0)":",0.0)");

						}
					} else*/ if (callfunc=="tex") {

						code="texture2D( "+dump_node_code(onode->arguments[1],p_level)+","+dump_node_code(onode->arguments[2],p_level)+")";
						break;
					} else if (callfunc=="texcube") {

						code="(textureCube( "+dump_node_code(onode->arguments[1],p_level)+",("+dump_node_code(onode->arguments[2],p_level)+")).xyz";
						break;
					} else if (callfunc=="texscreen") {
						//create the call to sample the screen, and clamp it
						uses_texscreen=true;
						code="(texture2D( texscreen_tex, clamp(("+dump_node_code(onode->arguments[1],p_level)+").xy*texscreen_screen_mult,texscreen_screen_clamp.xy,texscreen_screen_clamp.zw))).rgb";
						//code="(texture2D( screen_texture, ("+dump_node_code(onode->arguments[1],p_level)+").xy).rgb";
						break;
					} else if (callfunc=="texpos") {
						//create the call to sample the screen, and clamp it
						uses_texpos=true;
						code="get_texpos("+dump_node_code(onode->arguments[1],p_level)+"";
//						code="get_texpos(gl_ProjectionMatrixInverse * texture2D( depth_texture, clamp(("+dump_node_code(onode->arguments[1],p_level)+").xy,vec2(0.0),vec2(1.0))*gl_LightSource[5].specular.zw+gl_LightSource[5].specular.xy)";
						//code="(texture2D( screen_texture, ("+dump_node_code(onode->arguments[1],p_level)+").xy).rgb";
						break;

					} else {

						for(int i=1;i<onode->arguments.size();i++) {
							if (i>1)
								code+=", ";
								//transform
							code+=dump_node_code(onode->arguments[i],p_level);

						}
					}
					code+=")";
					break;
				} break;
				default: {}
			}

		} break;
		case SL::Node::TYPE_CONTROL_FLOW: {
			SL::ControlFlowNode *cfnode=(SL::ControlFlowNode*)p_node;
			if (cfnode->flow_op==SL::FLOW_OP_IF) {

				code+="if ("+dump_node_code(cfnode->statements[0],p_level)+") {"ENDL;
				code+=dump_node_code(cfnode->statements[1],p_level+1);
				if (cfnode->statements.size()==3) {

					code+="} else {"ENDL;
					code+=dump_node_code(cfnode->statements[2],p_level+1);
				}

				code+="}"ENDL;

			} else if (cfnode->flow_op==SL::FLOW_OP_RETURN) {

				if (cfnode->statements.size()) {
					code="return "+dump_node_code(cfnode->statements[0],p_level);
				} else {
					code="return";
				}
			}

		} break;
		case SL::Node::TYPE_MEMBER: {
			SL::MemberNode *mnode=(SL::MemberNode*)p_node;
			String m;
			if (mnode->basetype==SL::TYPE_MAT4) {
				if (mnode->name=="x")
					m="[0]";
				else if (mnode->name=="y")
					m="[1]";
				else if (mnode->name=="z")
					m="[2]";
				else if (mnode->name=="w")
					m="[3]";

			} else if (mnode->basetype==SL::TYPE_MAT3) {
				if (mnode->name=="x")
					m="[0]";
				else if (mnode->name=="y")
					m="[1]";
				else if (mnode->name=="z")
					m="[2]";

			} else {
				m="."+mnode->name;
			}
			code=dump_node_code(mnode->owner,p_level)+m;

		} break;
	}

	return code;

}


Error ShaderCompilerGLES2::compile_node(SL::ProgramNode *p_program) {

	// feed the local replace table and global code
	global_code="";

	// uniforms first!

	int ubase=0;
	if (uniforms)
		ubase=uniforms->size();
	for(Map<StringName,SL::Uniform>::Element *E=p_program->uniforms.front();E;E=E->next()) {

		String uline="uniform "+_typestr(E->get().type)+" _"+E->key().operator String()+";"ENDL;

		global_code+=uline;
		if (uniforms) {
			//if (uniforms->has(E->key())) {
			//	//repeated uniform, error
		//		ERR_EXPLAIN("Uniform already exists from other shader: "+String(E->key()));
		//		ERR_FAIL_COND_V(uniforms->has(E->key()),ERR_ALREADY_EXISTS);
//
//			}
			SL::Uniform u = E->get();
			u.order+=ubase;
			uniforms->insert(E->key(),u);
		}
	}

	for(int i=0;i<p_program->functions.size();i++) {


		SL::FunctionNode *fnode=p_program->functions[i].function;

		StringName funcname=fnode->name;
		String newfuncname=replace_string(funcname);

		String header;
		header=_typestr(fnode->return_type)+" "+newfuncname+"(";
		for(int i=0;i<fnode->arguments.size();i++) {

			if (i>0)
				header+=", ";
			header+=_typestr(fnode->arguments[i].type)+" "+replace_string(fnode->arguments[i].name);
		}

		header+=") {"ENDL;
		String fcode=header;
		fcode+=dump_node_code(fnode->body,1);
		fcode+="}"ENDL;
		global_code+=fcode;

	}

/*	for(Map<StringName,SL::DataType>::Element *E=p_program->preexisting_variables.front();E;E=E->next()) {

		StringName varname=E->key();
		String newvarname=replace_string(varname);
		global_code+="uniform "+_typestr(E->get())+" "+newvarname+";"ENDL;
	}*/

	code=dump_node_code(p_program,0);

#ifdef DEBUG_SHADER_ENABLED

	print_line("GLOBAL CODE:\n\n");
	print_line(global_code);
	global_code=global_code.replace("\n","");
	print_line("CODE:\n\n");
	print_line(code);
	code=code.replace("\n","");
#endif

	return OK;
}

Error ShaderCompilerGLES2::create_glsl_120_code(void *p_str,SL::ProgramNode *p_program) {

	ShaderCompilerGLES2 *compiler=(ShaderCompilerGLES2*)p_str;
	return compiler->compile_node(p_program);
}


String ShaderCompilerGLES2::replace_string(const StringName& p_string) {

	Map<StringName,StringName>::Element *E=NULL;
	E=replace_table.find(p_string);
	if (E)
		return E->get();

	E=mode_replace_table[type].find(p_string);
	if (E)
		return E->get();


	return "_"+p_string.operator String();
}

Error ShaderCompilerGLES2::compile(const String& p_code, ShaderLanguage::ShaderType p_type, String& r_code_line, String& r_globals_line, Flags& r_flags, Map<StringName,ShaderLanguage::Uniform> *r_uniforms) {

	uses_texscreen=false;
	uses_texpos=false;
	uses_alpha=false;
	uses_discard=false;
	uses_screen_uv=false;
	uses_light=false;
	uses_time=false;
	uses_normalmap=false;
	uses_normal=false;
	uses_texpixel_size=false;
	uses_worldvec=false;
	vertex_code_writes_vertex=false;
	uniforms=r_uniforms;
	flags=&r_flags;
	r_flags.use_color_interp=false;
	r_flags.use_uv_interp=false;
	r_flags.use_uv2_interp=false;
	r_flags.use_tangent_interp=false;
	r_flags.use_var1_interp=false;
	r_flags.use_var2_interp=false;
	r_flags.uses_normalmap=false;
	r_flags.uses_normal=false;

	String error;
	int errline,errcol;

	type=p_type;
	Error err = SL::compile(p_code,p_type,create_glsl_120_code,this,&error,&errline,&errcol);

	if (err) {
		print_line("***Error precompiling shader: "+error);
		print_line("error "+itos(errline)+":"+itos(errcol));
		return err;
	}

	r_flags.uses_alpha=uses_alpha;
	r_flags.uses_texscreen=uses_texscreen;
	r_flags.uses_texpos=uses_texpos;
	r_flags.vertex_code_writes_vertex=vertex_code_writes_vertex;
	r_flags.uses_discard=uses_discard;
	r_flags.uses_screen_uv=uses_screen_uv;
	r_flags.uses_light=uses_light;
	r_flags.uses_time=uses_time;
	r_flags.uses_normalmap=uses_normalmap;
	r_flags.uses_normal=uses_normalmap;
	r_flags.uses_texpixel_size=uses_texpixel_size;
	r_flags.uses_worldvec=uses_worldvec;
	r_code_line=code;
	r_globals_line=global_code;

	return OK;
}

ShaderCompilerGLES2::ShaderCompilerGLES2() {

	replace_table["bool"]= "bool";
	replace_table["float" ]=  "float";
	replace_table["vec2"  ]= "vec2";
	replace_table["vec3"  ]= "vec3";
	replace_table["vec4"  ]= "vec4";
	replace_table["mat3" ]= "mat3";
	replace_table["mat4" ]= "mat4";
	replace_table["texture" ]= "sampler2D";
	replace_table["cubemap" ]= "samplerCube";

	replace_table["sin"]= "sin";
	replace_table["cos" ]=  "cos";
	replace_table["tan"  ]= "tan";
	replace_table["asin" ]= "asin";
	replace_table["acos" ]= "acos";
	replace_table["atan" ]= "atan";
	replace_table["atan2"]= "atan";
	replace_table["sinh" ]= "sinh";
	replace_table["cosh" ]= "cosh";
	replace_table["tanh" ]= "tanh";
	replace_table["pow"  ]= "pow";
	replace_table["exp" ]=  "exp";
	replace_table["log" ]=  "log";
	replace_table["sqrt"]=  "sqrt";
	replace_table["abs" ]=  "abs";
	replace_table["sign"]=  "sign";
	replace_table["floor"]= "floor";
	replace_table["trunc"]= "trunc";
#ifdef GLEW_ENABLED
	replace_table["round"]= "roundfix";
#else
	replace_table["round"]= "round";
#endif
	replace_table["ceil" ]= "ceil";
	replace_table["fract"]= "fract";
	replace_table["mod"  ]= "mod";
	replace_table["min"  ]= "min";
	replace_table["max"]= "max";
	replace_table["clamp"]= "clamp";
	replace_table["mix"  ]= "mix";
	replace_table["step" ]= "step";
	replace_table["smoothstep" ]= "smoothstep";
	replace_table["length"]= "length";
	replace_table["distance"]= "distance";
	replace_table["dot" ]=  "dot";
	replace_table["cross" ]="cross";
	replace_table["normalize"]= "normalize";
	replace_table["reflect"]= "reflect";
	replace_table["refract"]= "refract";
	replace_table["tex"]= "tex";
	replace_table["texa"]= "texa";
	replace_table["tex2"]= "tex2";
	replace_table["texcube"]= "textureCube";
	replace_table["texscreen"]= "texscreen";
	replace_table["texpos"]= "texpos";

	mode_replace_table[0]["SRC_VERTEX"]="vertex_in.xyz";
	mode_replace_table[0]["SRC_NORMAL"]="normal_in";
	mode_replace_table[0]["SRC_TANGENT"]="tangent_in";
	mode_replace_table[0]["SRC_BINORMALF"]="binormalf";

	mode_replace_table[0]["VERTEX"]="vertex_interp";
	mode_replace_table[0]["NORMAL"]="normal_interp";
	mode_replace_table[0]["TANGENT"]="tangent_interp";
	mode_replace_table[0]["BINORMAL"]="binormal_interp";
	mode_replace_table[0]["UV"]="uv_interp.xy";
	mode_replace_table[0]["UV2"]="uv_interp.zw";
	mode_replace_table[0]["COLOR"]="color_interp";
	//@TODO convert to glsl stuff
	mode_replace_table[0]["SPEC_EXP"]="vertex_specular_exp";
	mode_replace_table[0]["WORLD_MATRIX"]="world_transform";
	mode_replace_table[0]["INV_CAMERA_MATRIX"]="camera_inverse_transform";
	mode_replace_table[0]["PROJECTION_MATRIX"]="projection_transform";
	mode_replace_table[0]["MODELVIEW_MATRIX"]="modelview";
	mode_replace_table[0]["POINT_SIZE"]="gl_PointSize";
	mode_replace_table[0]["VAR1"]="var1_interp";
	mode_replace_table[0]["VAR2"]="var2_interp";

//	mode_replace_table[0]["SCREEN_POS"]="SCREEN_POS";
//	mode_replace_table[0]["SCREEN_SIZE"]="SCREEN_SIZE";
	mode_replace_table[0]["INSTANCE_ID"]="instance_id";
	mode_replace_table[0]["TIME"]="time";

	mode_replace_table[1]["VERTEX"]="vertex";
	//mode_replace_table[1]["POSITION"]="IN_POSITION";
	mode_replace_table[1]["NORMAL"]="normal";
	mode_replace_table[1]["TANGENT"]="tangent";
	mode_replace_table[1]["POSITION"]="gl_Position";
	mode_replace_table[1]["BINORMAL"]="binormal";
	mode_replace_table[1]["NORMALMAP"]="normalmap";
	mode_replace_table[1]["NORMALMAP_DEPTH"]="normaldepth";
	mode_replace_table[1]["VAR1"]="var1_interp";
	mode_replace_table[1]["VAR2"]="var2_interp";
	mode_replace_table[1]["UV"]="uv";
	mode_replace_table[1]["UV2"]="uv2";
	mode_replace_table[1]["SCREEN_UV"]="screen_uv";
	mode_replace_table[1]["VAR1"]="var1_interp";
	mode_replace_table[1]["VAR2"]="var2_interp";
	mode_replace_table[1]["COLOR"]="color";
	mode_replace_table[1]["DIFFUSE"]="diffuse.rgb";
	mode_replace_table[1]["DIFFUSE_ALPHA"]="diffuse";
	mode_replace_table[1]["SPECULAR"]="specular";
	mode_replace_table[1]["EMISSION"]="emission";
	mode_replace_table[1]["SHADE_PARAM"]="shade_param";
	mode_replace_table[1]["SPEC_EXP"]="specular_exp";
	mode_replace_table[1]["GLOW"]="glow";
	mode_replace_table[1]["DISCARD"]="discard_";
	mode_replace_table[1]["POINT_COORD"]="gl_PointCoord";
	mode_replace_table[1]["INV_CAMERA_MATRIX"]="camera_inverse_transform";

	//mode_replace_table[1]["SCREEN_POS"]="SCREEN_POS";
	//mode_replace_table[1]["SCREEN_TEXEL_SIZE"]="SCREEN_TEXEL_SIZE";
	mode_replace_table[1]["TIME"]="time";

	//////////////

	mode_replace_table[2]["NORMAL"]="normal";
	//mode_replace_table[2]["POSITION"]="IN_POSITION";
	mode_replace_table[2]["LIGHT_DIR"]="light_dir";
	mode_replace_table[2]["LIGHT_DIFFUSE"]="light_diffuse";
	mode_replace_table[2]["LIGHT_SPECULAR"]="light_specular";
	mode_replace_table[2]["EYE_VEC"]="eye_vec";
	mode_replace_table[2]["DIFFUSE"]="mdiffuse";
	mode_replace_table[2]["SPECULAR"]="specular";
	mode_replace_table[2]["SPECULAR_EXP"]="specular_exp";
	mode_replace_table[2]["SHADE_PARAM"]="shade_param";
	mode_replace_table[2]["LIGHT"]="light";
	mode_replace_table[2]["POINT_COORD"]="gl_PointCoord";
	mode_replace_table[2]["TIME"]="time";

	mode_replace_table[3]["SRC_VERTEX"]="src_vtx";
	mode_replace_table[3]["VERTEX"]="outvec.xy";
	mode_replace_table[3]["WORLD_VERTEX"]="outvec.xy";
	mode_replace_table[3]["UV"]="uv_interp";
	mode_replace_table[3]["COLOR"]="color_interp";
	mode_replace_table[3]["VAR1"]="var1_interp";
	mode_replace_table[3]["VAR2"]="var2_interp";
	mode_replace_table[3]["POINT_SIZE"]="gl_PointSize";
	mode_replace_table[3]["WORLD_MATRIX"]="modelview_matrix";
	mode_replace_table[3]["PROJECTION_MATRIX"]="projection_matrix";
	mode_replace_table[3]["EXTRA_MATRIX"]="extra_matrix";
	mode_replace_table[3]["TIME"]="time";

	mode_replace_table[4]["POSITION"]="gl_Position";
	mode_replace_table[4]["NORMAL"]="normal";
	mode_replace_table[4]["UV"]="uv_interp";
	mode_replace_table[4]["SRC_COLOR"]="color_interp";
	mode_replace_table[4]["COLOR"]="color";
	mode_replace_table[4]["TEXTURE"]="texture";
	mode_replace_table[4]["TEXTURE_PIXEL_SIZE"]="texpixel_size";
	mode_replace_table[4]["VAR1"]="var1_interp";
	mode_replace_table[4]["VAR2"]="var2_interp";
	mode_replace_table[4]["SCREEN_UV"]="screen_uv";
	mode_replace_table[4]["POINT_COORD"]="gl_PointCoord";
	mode_replace_table[4]["TIME"]="time";

	mode_replace_table[5]["SRC_COLOR"]="color";
	mode_replace_table[5]["COLOR"]="color";
	mode_replace_table[5]["NORMAL"]="normal";
	mode_replace_table[5]["LIGHT_DIR"]="light_dir";
	mode_replace_table[5]["LIGHT_DISTANCE"]="light_distance";
	mode_replace_table[5]["LIGHT"]="light";
	mode_replace_table[5]["POINT_COORD"]="gl_PointCoord";
	mode_replace_table[5]["TIME"]="time";



	//mode_replace_table[2]["SCREEN_POS"]="SCREEN_POS";
	//mode_replace_table[2]["SCREEN_TEXEL_SIZE"]="SCREEN_TEXEL_SIZE";


	out_vertex_name="VERTEX";

	vname_discard="DISCARD";
	vname_screen_uv="SCREEN_UV";
	vname_diffuse_alpha="DIFFUSE_ALPHA";
	vname_color_interp="COLOR";
	vname_uv_interp="UV";
	vname_uv2_interp="UV2";
	vname_tangent_interp="TANGENT";
	vname_binormal_interp="BINORMAL";
	vname_var1_interp="VAR1";
	vname_var2_interp="VAR2";
	vname_vertex="VERTEX";
	vname_light="LIGHT";
	vname_time="TIME";
	vname_normalmap="NORMALMAP";
	vname_normal="NORMAL";
	vname_texpixel_size="TEXTURE_PIXEL_SIZE";
	vname_world_vec="WORLD_VERTEX";

}
