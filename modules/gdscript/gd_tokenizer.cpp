/*************************************************************************/
/*  gd_tokenizer.cpp                                                     */
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
#include "gd_tokenizer.h"
#include "print_string.h"
#include "gd_functions.h"
const char* GDTokenizer::token_names[TK_MAX]={
"Empty",
"Identifier",
"Constant",
"Self",
"Built-In Type",
"Built-In Func",
"In",
"'=='",
"'!='",
"'<'",
"'<='",
"'>'",
"'>='",
"'and'",
"'or'",
"'not'",
"'+'",
"'-'",
"'*'",
"'/'",
"'%'",
"'<<'",
"'>>'",
"'='",
"'+='",
"'-='",
"'*='",
"'/='",
"'%='",
"'<<='",
"'>>='",
"'&='",
"'|='",
"'^='",
"'&'",
"'|'",
"'^'",
"'~'",
//"Plus Plus",
//"Minus Minus",
"if",
"elif",
"else",
"for",
"do",
"while",
"switch",
"case",
"break",
"continue",
"pass",
"return",
"func",
"class",
"extends",
"tool",
"static",
"export",
"const",
"var",
"preload",
"assert",
"'['",
"']'",
"'{'",
"'}'",
"'('",
"')'",
"','",
"';'",
"'.'",
"'?'",
"':'",
"'\\n'",
"Error",
"EOF"};

const char *GDTokenizer::get_token_name(Token p_token) {

	ERR_FAIL_INDEX_V(p_token,TK_MAX,"<error>");
	return token_names[p_token];
}

static bool _is_text_char(CharType c) {

	return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='_';
}

static bool _is_number(CharType c) {

	return (c>='0' && c<='9');
}

static bool _is_hex(CharType c) {

	return (c>='0' && c<='9') || (c>='a' && c<='f') || (c>='A' && c<='F');
}

void GDTokenizer::_make_token(Token p_type)  {

	TokenData &tk=tk_rb[tk_rb_pos];

	tk.type=p_type;
	tk.line=line;
	tk.col=column;

	tk_rb_pos=(tk_rb_pos+1)%TK_RB_SIZE;
}
void GDTokenizer::_make_identifier(const StringName& p_identifier)  {

	TokenData &tk=tk_rb[tk_rb_pos];

	tk.type=TK_IDENTIFIER;
	tk.identifier=p_identifier;
	tk.line=line;
	tk.col=column;

	tk_rb_pos=(tk_rb_pos+1)%TK_RB_SIZE;

}

void GDTokenizer::_make_built_in_func(GDFunctions::Function p_func)  {

	TokenData &tk=tk_rb[tk_rb_pos];

	tk.type=TK_BUILT_IN_FUNC;
	tk.func=p_func;
	tk.line=line;
	tk.col=column;

	tk_rb_pos=(tk_rb_pos+1)%TK_RB_SIZE;

}
void GDTokenizer::_make_constant(const Variant& p_constant)  {

	TokenData &tk=tk_rb[tk_rb_pos];

	tk.type=TK_CONSTANT;
	tk.constant=p_constant;
	tk.line=line;
	tk.col=column;

	tk_rb_pos=(tk_rb_pos+1)%TK_RB_SIZE;

}

void GDTokenizer::_make_type(const Variant::Type& p_type) {


	TokenData &tk=tk_rb[tk_rb_pos];

	tk.type=TK_BUILT_IN_TYPE;
	tk.vtype=p_type;
	tk.line=line;
	tk.col=column;

	tk_rb_pos=(tk_rb_pos+1)%TK_RB_SIZE;

}


void GDTokenizer::_make_error(const String& p_error) {

	error_flag=true;
	last_error=p_error;

	TokenData &tk=tk_rb[tk_rb_pos];
	tk.type=TK_ERROR;
	tk.constant=p_error;
	tk.line=line;
	tk.col=column;
	tk_rb_pos=(tk_rb_pos+1)%TK_RB_SIZE;

}


void GDTokenizer::_make_newline(int p_spaces) {

	TokenData &tk=tk_rb[tk_rb_pos];
	tk.type=TK_NEWLINE;
	tk.constant=p_spaces;
	tk.line=line;
	tk.col=column;
	tk_rb_pos=(tk_rb_pos+1)%TK_RB_SIZE;
}

void GDTokenizer::_advance() {

	if (error_flag) {
		//parser broke
		_make_error(last_error);
		return;
	}

	if (code_pos>=len) {
		_make_token(TK_EOF);
		return;
	}
#define GETCHAR(m_ofs) ((m_ofs+code_pos)>=len?0:_code[m_ofs+code_pos])
#define INCPOS(m_amount) { code_pos+=m_amount; column+=m_amount; }
	while (true) {


		bool is_node_path=false;

		switch(GETCHAR(0)) {
			case 0:
				_make_token(TK_EOF);
				break;
			case '\t':
			case '\r':
			case ' ':
				INCPOS(1);
				continue;
			case '\n': {
				line++;
				INCPOS(1);
				column=0;
				int i=0;
				while(GETCHAR(i)==' ' || GETCHAR(i)=='\t') {
					i++;
				}

				_make_newline(i);
				return;
			}
#if 1 //py style tokenizer
			case '#': { // line comment skip

				while(GETCHAR(0)!='\n') {
					code_pos++;
					if (GETCHAR(0)==0) { //end of file
						_make_error("Unterminated Comment");
						return;
					}
				}
				INCPOS(1);
				column=0;
				line++;
				int i=0;
				while(GETCHAR(i)==' ' || GETCHAR(i)=='\t') {
					i++;
				}
				_make_newline(i);
				return;

			} break;
#endif
			case '/': {

				switch(GETCHAR(1)) {
#if 0 // c style tokenizer
					case '*': { // block comment
						int pos = code_pos+2;
						int new_line=line;
						int new_col=column+2;

						while(true) {
							if (_code[pos]=='0') {
								_make_error("Unterminated Comment");
								code_pos=pos;
								return;
							}
							if (_code[pos]=='*' && _code[pos+1]=='/') {
								new_col+=2;
								pos+=2; //compensate
								break;
							} else if (_code[pos]=='\n') {
								new_line++;
								new_col=0;
							} else {
								new_col++;
							}
							pos++;
						}

						column=new_col;
						line=new_line;
						code_pos=pos;
						continue;

					} break;
					case '/': { // line comment skip

						while(GETCHAR(0)!='\n') {
							code_pos++;
							if (GETCHAR(0)==0) { //end of file
								_make_error("Unterminated Comment");
								return;
							}
						}
						INCPOS(1);
						column=0;
						line++;
						continue;

					} break;
#endif
					case '=': { // diveq

						_make_token(TK_OP_ASSIGN_DIV);
						INCPOS(1);

					} break;
					default:
						_make_token(TK_OP_DIV);

				}
			} break;
			case '=': {
				if (GETCHAR(1)=='=') {
					_make_token(TK_OP_EQUAL);
					INCPOS(1);

				} else
					_make_token(TK_OP_ASSIGN);

			} break;
			case '<': {
				if (GETCHAR(1)=='=') {

					_make_token(TK_OP_LESS_EQUAL);
					INCPOS(1);
				} else if (GETCHAR(1)=='<') {
					if (GETCHAR(2)=='=') {
						_make_token(TK_OP_ASSIGN_SHIFT_LEFT);
						INCPOS(1);
					} else {
						_make_token(TK_OP_SHIFT_LEFT);
					}
					INCPOS(1);
				} else
					_make_token(TK_OP_LESS);

			} break;
			case '>': {
				if (GETCHAR(1)=='=') {
					_make_token(TK_OP_GREATER_EQUAL);
					INCPOS(1);
				} else if (GETCHAR(1)=='>') {
					if (GETCHAR(2)=='=') {
						_make_token(TK_OP_ASSIGN_SHIFT_RIGHT);
						INCPOS(1);

					} else {
						_make_token(TK_OP_SHIFT_RIGHT);
					}
					INCPOS(1);
				} else {
					_make_token(TK_OP_GREATER);
				}

			} break;
			case '!': {
				if (GETCHAR(1)=='=') {
					_make_token(TK_OP_NOT_EQUAL);
					INCPOS(1);
				} else {
					_make_token(TK_OP_NOT);
				}

			} break;
			//case '"' //string - no strings in shader
			//case '\'' //string - no strings in shader
			case '{':
				_make_token(TK_CURLY_BRACKET_OPEN);
				break;
			case '}':
				_make_token(TK_CURLY_BRACKET_CLOSE);
				break;
			case '[':
				_make_token(TK_BRACKET_OPEN);
				break;
			case ']':
				_make_token(TK_BRACKET_CLOSE);
				break;
			case '(':
				_make_token(TK_PARENTHESIS_OPEN);
				break;
			case ')':
				_make_token(TK_PARENTHESIS_CLOSE);
				break;
			case ',':
				_make_token(TK_COMMA);
				break;
			case ';':
				_make_token(TK_SEMICOLON);
				break;
			case '?':
				_make_token(TK_QUESTION_MARK);
				break;
			case ':':
				_make_token(TK_COLON); //for methods maybe but now useless.
				break;
			case '^': {
				if (GETCHAR(1)=='=') {
					_make_token(TK_OP_ASSIGN_BIT_XOR);
					INCPOS(1);
				} else {
					_make_token(TK_OP_BIT_XOR);
				}

			} break;
			case '~':
				_make_token(TK_OP_BIT_INVERT);
				break;
			case '&': {
				if (GETCHAR(1)=='&') {

					_make_token(TK_OP_AND);
					INCPOS(1);
				} else if (GETCHAR(1)=='=') {
					_make_token(TK_OP_ASSIGN_BIT_AND);
					INCPOS(1);
				} else {
					_make_token(TK_OP_BIT_AND);
				}
			} break;
			case '|': {
				if (GETCHAR(1)=='|') {

					_make_token(TK_OP_OR);
					INCPOS(1);
				} else if (GETCHAR(1)=='=') {
					_make_token(TK_OP_ASSIGN_BIT_OR);
					INCPOS(1);
				} else {
					_make_token(TK_OP_BIT_OR);
				}
			} break;
			case '*': {

				if (GETCHAR(1)=='=') {
					_make_token(TK_OP_ASSIGN_MUL);
					INCPOS(1);
				} else {
					_make_token(TK_OP_MUL);
				}
			} break;
			case '+': {

				if (GETCHAR(1)=='=') {
					_make_token(TK_OP_ASSIGN_ADD);
					INCPOS(1);
				//}  else if (GETCHAR(1)=='+') {
				//	_make_token(TK_OP_PLUS_PLUS);
				//	INCPOS(1);
				} else {
					_make_token(TK_OP_ADD);
				}

			} break;
			case '-': {

				if (GETCHAR(1)=='=') {
					_make_token(TK_OP_ASSIGN_SUB);
					INCPOS(1);
				//}  else if (GETCHAR(1)=='-') {
				//	_make_token(TK_OP_MINUS_MINUS);
				//	INCPOS(1);
				} else {
					_make_token(TK_OP_SUB);
				}
			} break;
			case '%': {

				if (GETCHAR(1)=='=') {
					_make_token(TK_OP_ASSIGN_MOD);
					INCPOS(1);
				} else {
					_make_token(TK_OP_MOD);
				}
			} break;
			case '@':
				if (CharType(GETCHAR(1))!='"') {
					_make_error("Unexpected '@'");
					return;
				}
				INCPOS(1);
				is_node_path=true;
			case '"': {

				int i=1;
				String str;
				while(true) {
					if (CharType(GETCHAR(i)==0)) {

						_make_error("Unterminated String");
						return;
					} else if (CharType(GETCHAR(i)=='"')) {
						break;
					} else if (CharType(GETCHAR(i)=='\\')) {
						//escaped characters...
						i++;
						CharType next = GETCHAR(i);
						if (next==0) {
							_make_error("Unterminated String");
							return;
						}
						CharType res=0;

						switch(next) {

							case 'a': res=7; break;
							case 'b': res=8; break;
							case 't': res=9; break;
							case 'n': res=10; break;
							case 'v': res=11; break;
							case 'f': res=12; break;
							case 'r': res=13; break;
							case '\'': res='\''; break;
							case '\"': res='\"'; break;
							case '\\': res='\\'; break;
							case 'x': {
								//hexnumbarh - oct is deprecated

								int read=0;
								for(int j=0;j<4;j++) {
									CharType c = GETCHAR(i+j);
									if (c==0) {
										_make_error("Unterminated String");
										return;
									}
									if (!_is_hex(c)) {
										if (j==0 || !(j&1)) {
											_make_error("Malformed hex constant in string");
											return;
										} else
											break;
									}
									CharType v;
									if (c>='0' && c<='9') {
										v=c-'0';
									} else if (c>='a' && c<='f') {
										v=c-'a';
										v+=10;
									} else if (c>='A' && c<='F') {
										v=c-'A';
										v+=10;
									} else {
										ERR_PRINT("BUG");
										v=0;
									}

									res<<=4;
									res|=v;

									read++;
								}
								i+=read-1;


							} break;
							default: {

								_make_error("Invalid escape sequence");
								return;
							} break;
						}

						str+=res;

					} else {
						str+=CharType(GETCHAR(i));
					}
					i++;
				}
				INCPOS(i);

				if (is_node_path) {
					_make_constant(NodePath(str));
				} else {
					_make_constant(str);
				}

			} break;
			default: {

				if (_is_number(GETCHAR(0)) || (GETCHAR(0)=='.' && _is_number(GETCHAR(1)))) {
					// parse number
					bool period_found=false;
					bool exponent_found=false;
					bool hexa_found=false;
					bool sign_found=false;

					String str;
					int i=0;

					while(true) {
						if (GETCHAR(i)=='.') {
							if (period_found || exponent_found) {
                                _make_error("Invalid numeric constant at '.'");
								return;
							}
							period_found=true;
						} else if (GETCHAR(i)=='x') {
                            if (hexa_found || str.length()!=1 || !( (i==1 && str[0]=='0') || (i==2 && str[1]=='0' && str[0]=='-') ) ) {
                                _make_error("Invalid numeric constant at 'x'");
								return;
							}
							hexa_found=true;
                        } else if (!hexa_found && GETCHAR(i)=='e') {
							if (hexa_found || exponent_found) {
                                _make_error("Invalid numeric constant at 'e'");
								return;
							}
							exponent_found=true;
						} else if (_is_number(GETCHAR(i))) {
							//all ok
						} else if (hexa_found && _is_hex(GETCHAR(i))) {

						} else if ((GETCHAR(i)=='-' || GETCHAR(i)=='+') && exponent_found) {
							if (sign_found) {
                                _make_error("Invalid numeric constant at '-'");
								return;
							}
							sign_found=true;
						} else
							break;

						str+=CharType(GETCHAR(i));
						i++;
					}

                    if (!( _is_number(str[str.length()-1]) || (hexa_found && _is_hex(str[str.length()-1])))) {
                        _make_error("Invalid numeric constant: "+str);
						return;
					}

					INCPOS(str.length());
                    if (hexa_found) {
                        int val = str.hex_to_int();
                        _make_constant(val);
                    } else if (period_found) {
						real_t val = str.to_double();
						//print_line("*%*%*%*% to convert: "+str+" result: "+rtos(val));
						_make_constant(val);
                    } else {
						int val = str.to_int();
						_make_constant(val);

					}

					return;
				}

				if (GETCHAR(0)=='.') {
					//parse period
					_make_token(TK_PERIOD);
					break;
				}

				if (_is_text_char(GETCHAR(0))) {
					// parse identifier
					String str;
					str+=CharType(GETCHAR(0));

					int i=1;
					while(_is_text_char(GETCHAR(i))) {
						str+=CharType(GETCHAR(i));
						i++;
					}

					bool identifier=false;

					if (str=="null") {
						_make_constant(Variant());

					} else if (str=="true") {
						_make_constant(true);

					} else if (str=="false") {
						_make_constant(false);
					} else {

						bool found=false;

						struct _bit { Variant::Type type; const char *text;};
						//built in types

						static const  _bit type_list[]={
							//types
							{Variant::BOOL,"bool"},
							{Variant::INT,"int"},
							{Variant::REAL,"float"},
							{Variant::STRING,"String"},
							{Variant::VECTOR2,"vec2"},
							{Variant::VECTOR2,"Vector2"},
							{Variant::RECT2,"Rect2"},
							{Variant::MATRIX32,"Matrix32"},
							{Variant::MATRIX32,"mat32"},
							{Variant::VECTOR3,"vec3"},
							{Variant::VECTOR3,"Vector3"},
							{Variant::_AABB,"AABB"},
							{Variant::_AABB,"Rect3"},
							{Variant::PLANE,"Plane"},
							{Variant::QUAT,"Quat"},
							{Variant::MATRIX3,"mat3"},
							{Variant::MATRIX3,"Matrix3"},
							{Variant::TRANSFORM,"trn"},
							{Variant::TRANSFORM,"Transform"},
							{Variant::COLOR,"Color"},
							{Variant::IMAGE,"Image"},
							{Variant::_RID,"RID"},
							{Variant::OBJECT,"Object"},
							{Variant::INPUT_EVENT,"InputEvent"},
							{Variant::DICTIONARY,"dict"},
							{Variant::DICTIONARY,"Dictionary"},
							{Variant::ARRAY,"Array"},
							{Variant::RAW_ARRAY,"RawArray"},
							{Variant::INT_ARRAY,"IntArray"},
							{Variant::REAL_ARRAY,"FloatArray"},
							{Variant::STRING_ARRAY,"StringArray"},
							{Variant::VECTOR2_ARRAY,"Vector2Array"},
							{Variant::VECTOR3_ARRAY,"Vector3Array"},
							{Variant::COLOR_ARRAY,"ColorArray"},
							{Variant::VARIANT_MAX,NULL},
						};

						{


							int idx=0;

							while(type_list[idx].text) {

								if (str==type_list[idx].text) {
									_make_type(type_list[idx].type);
									found=true;
									break;
								}
								idx++;
							}
						}

						if (!found) {

							//built in func?

							for(int i=0;i<GDFunctions::FUNC_MAX;i++) {

								if (str==GDFunctions::get_func_name(GDFunctions::Function(i))) {

									_make_built_in_func(GDFunctions::Function(i));
									found=true;
									 break;
								}
							}

							//keywor
						}

						if (!found) {


							struct _kws { Token token; const char *text;};

							static const  _kws keyword_list[]={
								//ops
								{TK_OP_IN,"in"},
								{TK_OP_NOT,"not"},
								{TK_OP_OR,"or"},
								{TK_OP_AND,"and"},
								//func
								{TK_PR_FUNCTION,"func"},
								{TK_PR_FUNCTION,"function"},
								{TK_PR_CLASS,"class"},
								{TK_PR_EXTENDS,"extends"},
								{TK_PR_TOOL,"tool"},
								{TK_PR_STATIC,"static"},
								{TK_PR_EXPORT,"export"},
								{TK_PR_VAR,"var"},
								{TK_PR_PRELOAD,"preload"},
								{TK_PR_ASSERT,"assert"},
								{TK_PR_CONST,"const"},
								//controlflow
								{TK_CF_IF,"if"},
								{TK_CF_ELIF,"elif"},
								{TK_CF_ELSE,"else"},
								{TK_CF_FOR,"for"},
								{TK_CF_WHILE,"while"},
								{TK_CF_DO,"do"},
								{TK_CF_SWITCH,"switch"},
								{TK_CF_BREAK,"break"},
								{TK_CF_CONTINUE,"continue"},
								{TK_CF_RETURN,"return"},
								{TK_CF_PASS,"pass"},
								{TK_SELF,"self"},
								{TK_ERROR,NULL}
							};

							int idx=0;
							found=false;

							while(keyword_list[idx].text) {

								if (str==keyword_list[idx].text) {
									_make_token(keyword_list[idx].token);
									found=true;
									break;
								}
								idx++;
							}
						}

						if (!found)
							identifier=true;
					}


					if (identifier) {
						_make_identifier(str);
					}
					INCPOS(str.length());
					return;
				}

				_make_error("Unknown character");
				return;

			} break;
		}

		INCPOS(1);
		break;
	}

}

void GDTokenizer::set_code(const String& p_code) {

	code=p_code;
	len = p_code.length();
	if (len) {
		_code=&code[0];
	} else {
		_code=NULL;
	}
	code_pos=0;
	line=1; //it is stand-ar-ized that lines begin in 1 in code..
	column=0;
	tk_rb_pos=0;
	error_flag=false;
	last_error="";
	for(int i=0;i<MAX_LOOKAHEAD+1;i++)
		_advance();
}

GDTokenizer::Token GDTokenizer::get_token(int p_offset) const {
	ERR_FAIL_COND_V( p_offset <= -MAX_LOOKAHEAD, TK_ERROR);
	ERR_FAIL_COND_V( p_offset >= MAX_LOOKAHEAD, TK_ERROR);

	int ofs = (TK_RB_SIZE + tk_rb_pos + p_offset - MAX_LOOKAHEAD -1)%TK_RB_SIZE;
	return tk_rb[ofs].type;
}

int GDTokenizer::get_token_line(int p_offset) const {
	ERR_FAIL_COND_V( p_offset <= -MAX_LOOKAHEAD, -1);
	ERR_FAIL_COND_V( p_offset >= MAX_LOOKAHEAD, -1);

	int ofs = (TK_RB_SIZE + tk_rb_pos + p_offset - MAX_LOOKAHEAD -1)%TK_RB_SIZE;
	return tk_rb[ofs].line;
}

int GDTokenizer::get_token_column(int p_offset) const {
	ERR_FAIL_COND_V( p_offset <= -MAX_LOOKAHEAD, -1);
	ERR_FAIL_COND_V( p_offset >= MAX_LOOKAHEAD, -1);

	int ofs = (TK_RB_SIZE + tk_rb_pos + p_offset - MAX_LOOKAHEAD -1)%TK_RB_SIZE;
	return tk_rb[ofs].col;
}

const Variant& GDTokenizer::get_token_constant(int p_offset) const {
	ERR_FAIL_COND_V( p_offset <= -MAX_LOOKAHEAD, tk_rb[0].constant);
	ERR_FAIL_COND_V( p_offset >= MAX_LOOKAHEAD, tk_rb[0].constant);

	int ofs = (TK_RB_SIZE + tk_rb_pos + p_offset - MAX_LOOKAHEAD -1)%TK_RB_SIZE;
	ERR_FAIL_COND_V(tk_rb[ofs].type!=TK_CONSTANT,tk_rb[0].constant);
	return tk_rb[ofs].constant;
}
StringName GDTokenizer::get_token_identifier(int p_offset) const {

	ERR_FAIL_COND_V( p_offset <= -MAX_LOOKAHEAD, StringName());
	ERR_FAIL_COND_V( p_offset >= MAX_LOOKAHEAD, StringName());

	int ofs = (TK_RB_SIZE + tk_rb_pos + p_offset - MAX_LOOKAHEAD -1)%TK_RB_SIZE;
	ERR_FAIL_COND_V(tk_rb[ofs].type!=TK_IDENTIFIER,StringName());
	return tk_rb[ofs].identifier;

}

GDFunctions::Function GDTokenizer::get_token_built_in_func(int p_offset) const {

	ERR_FAIL_COND_V( p_offset <= -MAX_LOOKAHEAD, GDFunctions::FUNC_MAX);
	ERR_FAIL_COND_V( p_offset >= MAX_LOOKAHEAD, GDFunctions::FUNC_MAX);

	int ofs = (TK_RB_SIZE + tk_rb_pos + p_offset - MAX_LOOKAHEAD -1)%TK_RB_SIZE;
	ERR_FAIL_COND_V(tk_rb[ofs].type!=TK_BUILT_IN_FUNC,GDFunctions::FUNC_MAX);
	return tk_rb[ofs].func;

}

Variant::Type GDTokenizer::get_token_type(int p_offset) const {

	ERR_FAIL_COND_V( p_offset <= -MAX_LOOKAHEAD, Variant::NIL);
	ERR_FAIL_COND_V( p_offset >= MAX_LOOKAHEAD, Variant::NIL);

	int ofs = (TK_RB_SIZE + tk_rb_pos + p_offset - MAX_LOOKAHEAD -1)%TK_RB_SIZE;
	ERR_FAIL_COND_V(tk_rb[ofs].type!=TK_BUILT_IN_TYPE,Variant::NIL);
	return tk_rb[ofs].vtype;

}


int GDTokenizer::get_token_line_indent(int p_offset) const {

	ERR_FAIL_COND_V( p_offset <= -MAX_LOOKAHEAD, 0);
	ERR_FAIL_COND_V( p_offset >= MAX_LOOKAHEAD, 0);

	int ofs = (TK_RB_SIZE + tk_rb_pos + p_offset - MAX_LOOKAHEAD -1)%TK_RB_SIZE;
	ERR_FAIL_COND_V(tk_rb[ofs].type!=TK_NEWLINE,0);
	return tk_rb[ofs].constant;

}

String GDTokenizer::get_token_error(int p_offset) const {

	ERR_FAIL_COND_V( p_offset <= -MAX_LOOKAHEAD, String());
	ERR_FAIL_COND_V( p_offset >= MAX_LOOKAHEAD, String());

	int ofs = (TK_RB_SIZE + tk_rb_pos + p_offset - MAX_LOOKAHEAD -1)%TK_RB_SIZE;
	ERR_FAIL_COND_V(tk_rb[ofs].type!=TK_ERROR,String());
	return tk_rb[ofs].constant;
}

void GDTokenizer::advance(int p_amount) {

	ERR_FAIL_COND( p_amount <=0 );
	for(int i=0;i<p_amount;i++)
		_advance();
}
