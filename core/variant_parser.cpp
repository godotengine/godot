#include "variant_parser.h"
#include "io/resource_loader.h"
#include "os/keyboard.h"



CharType VariantParser::StreamFile::get_char() {

	return f->get_8();
}

bool VariantParser::StreamFile::is_utf8() const {

	return true;
}
bool VariantParser::StreamFile::is_eof() const {

	return f->eof_reached();
}




/////////////////////////////////////////////////////////////////////////////////////////////////



const char * VariantParser::tk_name[TK_MAX] = {
	"'{'",
	"'}'",
	"'['",
	"']'",
	"'('",
	"')'",
	"identifier",
	"string",
	"number",
	"':'",
	"','",
	"'='",
	"EOF",
	"ERROR"
};



Error VariantParser::get_token(Stream *p_stream, Token& r_token, int &line, String &r_err_str) {

	while (true) {

		CharType cchar;
		if (p_stream->saved) {
			cchar=p_stream->saved;
			p_stream->saved=0;
		} else {
			cchar=p_stream->get_char();
		}

		switch(cchar) {

			case '\n': {

				line++;
				break;
			};
			case 0: {
				r_token.type=TK_EOF;
				return OK;
			} break;
			case '{': {

				r_token.type=TK_CURLY_BRACKET_OPEN;
				return OK;
			};
			case '}': {

				r_token.type=TK_CURLY_BRACKET_CLOSE;
				return OK;
			};
			case '[': {

				r_token.type=TK_BRACKET_OPEN;
				return OK;
			};
			case ']': {

				r_token.type=TK_BRACKET_CLOSE;
				return OK;
			};
			case '(': {

				r_token.type=TK_PARENTHESIS_OPEN;
				return OK;
			};
			case ')': {

				r_token.type=TK_PARENTHESIS_CLOSE;
				return OK;
			};
			case ':': {

				r_token.type=TK_COLON;
				return OK;
			};
			case ',': {

				r_token.type=TK_COMMA;
				return OK;
			};
			case '=': {

				r_token.type=TK_EQUAL;
				return OK;
			};
			case '"': {


				String str;
				while(true) {

					CharType ch=p_stream->get_char();

					if (ch==0) {
						r_err_str="Unterminated String";
						r_token.type=TK_ERROR;
						return ERR_PARSE_ERROR;
					} else if (ch=='"') {
						break;
					} else if (ch=='\\') {
						//escaped characters...
						CharType next = p_stream->get_char();
						if (next==0) {
							r_err_str="Unterminated String";
							r_token.type=TK_ERROR;
							return  ERR_PARSE_ERROR;
						}
						CharType res=0;

						switch(next) {

							case 'b': res=8; break;
							case 't': res=9; break;
							case 'n': res=10; break;
							case 'f': res=12; break;
							case 'r': res=13; break;
							case 'u': {
								//hexnumbarh - oct is deprecated


								for(int j=0;j<4;j++) {
									CharType c = p_stream->get_char();
									if (c==0) {
										r_err_str="Unterminated String";
										r_token.type=TK_ERROR;
										return ERR_PARSE_ERROR;
									}
									if (!((c>='0' && c<='9') || (c>='a' && c<='f') || (c>='A' && c<='F'))) {

										r_err_str="Malformed hex constant in string";
										r_token.type=TK_ERROR;
										return ERR_PARSE_ERROR;
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


								}



							} break;
							//case '\"': res='\"'; break;
							//case '\\': res='\\'; break;
							//case '/': res='/'; break;
							default: {
								res = next;
								//r_err_str="Invalid escape sequence";
								//return ERR_PARSE_ERROR;
							} break;
						}

						str+=res;

					} else {
						if (ch=='\n')
							line++;
						str+=ch;
					}
				}

				r_token.type=TK_STRING;
				r_token.value=str;
				return OK;

			} break;
			default: {

				if (cchar<=32) {
					break;
				}

				if (cchar=='-' || (cchar>='0' && cchar<='9')) {
					//a number
					print_line("a numbar");

					String num;
#define READING_SIGN 0
#define READING_INT 1
#define READING_DEC 2
#define READING_EXP 3
#define READING_DONE 4
					int reading=READING_INT;

					if (cchar=='-') {
						num+='-';
						cchar=p_stream->get_char();
						print_line("isnegative");

					}



					CharType c = cchar;
					bool exp_sign=false;
					bool exp_beg=false;
					bool is_float=false;

					while(true) {

						switch(reading) {
							case READING_INT: {

								if (c>='0' && c<='9') {
									//pass
									print_line("num: regular");
								} else if (c=='.') {
									reading=READING_DEC;
									print_line("num: decimal");
									is_float=true;
								} else if (c=='e') {
									reading=READING_EXP;
									print_line("num: exp");
								} else {
									reading=READING_DONE;
								}

							 } break;
							case READING_DEC: {

								if (c>='0' && c<='9') {
									print_line("dec: exp");

								} else if (c=='e') {
									reading=READING_EXP;
									print_line("dec: expe");
								} else {
									reading=READING_DONE;
								}

							 } break;
							case READING_EXP: {

								if (c>='0' && c<='9') {
									exp_beg=true;
									print_line("exp: num");
								} else if ((c=='-' || c=='+') && !exp_sign && !exp_beg) {
									exp_sign=true;
									print_line("exp: sgn");
								} else {
									reading=READING_DONE;
								}
							 } break;
						}

						if (reading==READING_DONE)
							break;
						num+=String::chr(c);
						c = p_stream->get_char();
						print_line("add to c");

					}

					p_stream->saved=c;

					print_line("num was: "+num);
					r_token.type=TK_NUMBER;
					if (is_float)
						r_token.value=num.to_double();
					else
						r_token.value=num.to_int();
					return OK;

				} else if ((cchar>='A' && cchar<='Z') || (cchar>='a' && cchar<='z') || cchar=='_') {

					String id;

					while((cchar>='A' && cchar<='Z') || (cchar>='a' && cchar<='z') || cchar=='_') {

						id+=String::chr(cchar);
						cchar=p_stream->get_char();
					}

					p_stream->saved=cchar;

					r_token.type=TK_IDENTIFIER;
					r_token.value=id;
					return OK;
				} else {
					r_err_str="Unexpected character.";
					r_token.type=TK_ERROR;
					return ERR_PARSE_ERROR;
				}
			}
		}
	}

	r_token.type=TK_ERROR;
	return ERR_PARSE_ERROR;
}


Error VariantParser::_parse_construct(Stream *p_stream,Vector<float>& r_construct,int &line,String &r_err_str) {


	Token token;
	get_token(p_stream,token,line,r_err_str);
	if (token.type!=TK_PARENTHESIS_OPEN) {
		r_err_str="Expected '('";
		return ERR_PARSE_ERROR;
	}


	bool first=true;
	while(true) {

		if (!first) {
			get_token(p_stream,token,line,r_err_str);
			if (token.type==TK_COMMA) {
				//do none
			} else if (token.type!=TK_PARENTHESIS_CLOSE) {
				break;
			} else {
				r_err_str="Expected ',' or ')'";
				return ERR_PARSE_ERROR;

			}
		}
		get_token(p_stream,token,line,r_err_str);
		if (token.type!=TK_NUMBER) {
			r_err_str="Expected float";
			return ERR_PARSE_ERROR;
		}

		r_construct.push_back(token.value);
	}

	return OK;

}

Error VariantParser::parse_value(Token& token,Variant &value,Stream *p_stream,int &line,String &r_err_str,ResourceParser *p_res_parser) {



/*	{
		Error err = get_token(p_stream,token,line,r_err_str);
		if (err)
			return err;
	}*/


	if (token.type==TK_CURLY_BRACKET_OPEN) {

		Dictionary d;
		Error err = _parse_dictionary(d,p_stream,line,r_err_str,p_res_parser);
		if (err)
			return err;
		value=d;
		return OK;
	} else if (token.type==TK_BRACKET_OPEN) {

		Array a;
		Error err = _parse_array(a,p_stream,line,r_err_str,p_res_parser);
		if (err)
			return err;
		value=a;
		return OK;

	} else if (token.type==TK_IDENTIFIER) {
/*
		VECTOR2,		// 5
		RECT2,
		VECTOR3,
		MATRIX32,
		PLANE,
		QUAT,			// 10
		_AABB, //sorry naming convention fail :( not like it's used often
		MATRIX3,
		TRANSFORM,

		// misc types
		COLOR,
		IMAGE,			// 15
		NODE_PATH,
		_RID,
		OBJECT,
		INPUT_EVENT,
		DICTIONARY,		// 20
		ARRAY,

		// arrays
		RAW_ARRAY,
		INT_ARRAY,
		REAL_ARRAY,
		STRING_ARRAY,	// 25
		VECTOR2_ARRAY,
		VECTOR3_ARRAY,
		COLOR_ARRAY,

		VARIANT_MAX

*/
		String id = token.value;
		if (id=="true")
			value=true;
		else if (id=="false")
			value=false;
		else if (id=="null")
			value=Variant();
		else if (id=="Vector2"){

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			if (args.size()!=2) {
				r_err_str="Expected 2 arguments for constructor";
			}

			value=Vector2(args[0],args[1]);
			return OK;
		} else if (id=="Vector3"){

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			if (args.size()!=3) {
				r_err_str="Expected 3 arguments for constructor";
			}

			value=Vector3(args[0],args[1],args[2]);
			return OK;
		} else if (id=="Matrix32"){

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			if (args.size()!=6) {
				r_err_str="Expected 6 arguments for constructor";
			}
			Matrix32 m;
			m[0]=Vector2(args[0],args[1]);
			m[1]=Vector2(args[2],args[3]);
			m[2]=Vector2(args[4],args[5]);
			value=m;
			return OK;
		} else if (id=="Plane") {

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			if (args.size()!=4) {
				r_err_str="Expected 4 arguments for constructor";
			}

			value=Plane(args[0],args[1],args[2],args[3]);
			return OK;
		} else if (id=="Quat") {

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			if (args.size()!=4) {
				r_err_str="Expected 4 arguments for constructor";
			}

			value=Quat(args[0],args[1],args[2],args[3]);
			return OK;

		} else if (id=="AABB"){

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			if (args.size()!=6) {
				r_err_str="Expected 6 arguments for constructor";
			}

			value=AABB(Vector3(args[0],args[1],args[2]),Vector3(args[3],args[4],args[5]));
			return OK;

		} else if (id=="Matrix3"){

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			if (args.size()!=9) {
				r_err_str="Expected 9 arguments for constructor";
			}

			value=Matrix3(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8]);
			return OK;
		} else if (id=="Transform"){

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			if (args.size()!=12) {
				r_err_str="Expected 12 arguments for constructor";
			}

			value=Transform(Matrix3(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8]),Vector3(args[9],args[10],args[11]));
			return OK;

		} else if (id=="Color") {

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			if (args.size()!=4) {
				r_err_str="Expected 4 arguments for constructor";
			}

			value=Color(args[0],args[1],args[2],args[3]);
			return OK;

		} else if (id=="Image") {

			//:|

			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_PARENTHESIS_OPEN) {
				r_err_str="Expected '('";
				return ERR_PARSE_ERROR;
			}


			get_token(p_stream,token,line,r_err_str);
			if (token.type==TK_PARENTHESIS_CLOSE) {
				value=Image(); // just an Image()
				return OK;
			} else if (token.type!=TK_NUMBER) {
				r_err_str="Expected number (width)";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream,token,line,r_err_str);

			int width=token.value;
			if (token.type!=TK_COMMA) {
				r_err_str="Expected ','";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_NUMBER) {
				r_err_str="Expected number (height)";
				return ERR_PARSE_ERROR;
			}

			int height=token.value;
			if (token.type!=TK_COMMA) {
				r_err_str="Expected ','";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_NUMBER) {
				r_err_str="Expected number (mipmaps)";
				return ERR_PARSE_ERROR;
			}

			int mipmaps=token.value;
			if (token.type!=TK_COMMA) {
				r_err_str="Expected ','";
				return ERR_PARSE_ERROR;
			}


			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_IDENTIFIER) {
				r_err_str="Expected identifier (format)";
				return ERR_PARSE_ERROR;
			}

			String sformat=token.value;

			Image::Format format;

			if (sformat=="GRAYSCALE") format=Image::FORMAT_GRAYSCALE;
			else if (sformat=="INTENSITY") format=Image::FORMAT_INTENSITY;
			else if (sformat=="GRAYSCALE_ALPHA") format=Image::FORMAT_GRAYSCALE_ALPHA;
			else if (sformat=="RGB") format=Image::FORMAT_RGB;
			else if (sformat=="RGBA") format=Image::FORMAT_RGBA;
			else if (sformat=="INDEXED") format=Image::FORMAT_INDEXED;
			else if (sformat=="INDEXED_ALPHA") format=Image::FORMAT_INDEXED_ALPHA;
			else if (sformat=="BC1") format=Image::FORMAT_BC1;
			else if (sformat=="BC2") format=Image::FORMAT_BC2;
			else if (sformat=="BC3") format=Image::FORMAT_BC3;
			else if (sformat=="BC4") format=Image::FORMAT_BC4;
			else if (sformat=="BC5") format=Image::FORMAT_BC5;
			else if (sformat=="PVRTC2") format=Image::FORMAT_PVRTC2;
			else if (sformat=="PVRTC2_ALPHA") format=Image::FORMAT_PVRTC2_ALPHA;
			else if (sformat=="PVRTC4") format=Image::FORMAT_PVRTC4;
			else if (sformat=="PVRTC4_ALPHA") format=Image::FORMAT_PVRTC4_ALPHA;
			else if (sformat=="ATC") format=Image::FORMAT_ATC;
			else if (sformat=="ATC_ALPHA_EXPLICIT") format=Image::FORMAT_ATC_ALPHA_EXPLICIT;
			else if (sformat=="ATC_ALPHA_INTERPOLATED") format=Image::FORMAT_ATC_ALPHA_INTERPOLATED;
			else if (sformat=="CUSTOM") format=Image::FORMAT_CUSTOM;
			else {
				r_err_str="Invalid image format: '"+sformat+"'";
				return ERR_PARSE_ERROR;
			};

			int len = Image::get_image_data_size(width,height,format,mipmaps);

			DVector<uint8_t> buffer;
			buffer.resize(len);

			if (buffer.size()!=len) {
				r_err_str="Couldn't allocate image buffer of size: "+itos(len);
			}

			{
				DVector<uint8_t>::Write w=buffer.write();

				for(int i=0;i<len;i++) {

					if (token.type!=TK_COMMA) {
						r_err_str="Expected ','";
						return ERR_PARSE_ERROR;
					}

					if (token.type!=TK_NUMBER) {
						r_err_str="Expected number";
						return ERR_PARSE_ERROR;
					}

					w[i]=int(token.value);

				}
			}


			Image img(width,height,mipmaps,format,buffer);

			value=img;

			return OK;


		} else if (id=="NodePath") {



			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_PARENTHESIS_OPEN) {
				r_err_str="Expected '('";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_STRING) {
				r_err_str="Expected string as argument";
				return ERR_PARSE_ERROR;
			}

			value=NodePath(String(token.value));

			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_PARENTHESIS_CLOSE) {
				r_err_str="Expected ')'";
				return ERR_PARSE_ERROR;
			}

		} else if (id=="RID") {



			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_PARENTHESIS_OPEN) {
				r_err_str="Expected '('";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_NUMBER) {
				r_err_str="Expected number as argument";
				return ERR_PARSE_ERROR;
			}

			value=token.value;

			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_PARENTHESIS_CLOSE) {
				r_err_str="Expected ')'";
				return ERR_PARSE_ERROR;
			}


			return OK;

		} else if (id=="Resource") {



			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_PARENTHESIS_OPEN) {
				r_err_str="Expected '('";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream,token,line,r_err_str);
			if (token.type==TK_STRING) {
				String path=token.value;
				RES res = ResourceLoader::load(path);
				if (res.is_null()) {
					r_err_str="Can't load resource at path: '"+path+"'.";
					return ERR_PARSE_ERROR;
				}

				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_PARENTHESIS_CLOSE) {
					r_err_str="Expected ')'";
					return ERR_PARSE_ERROR;
				}

				value=res;
				return OK;

			} else if (p_res_parser && p_res_parser->func){

				RES res;
				Error err = p_res_parser->func(p_res_parser->userdata,p_stream,res,line,r_err_str);
				if (err)
					return err;

				value=res;

				return OK;
			} else {

				r_err_str="Expected string as argument.";
				return ERR_PARSE_ERROR;
			}

			return OK;

		} else if (id=="InputEvent") {



			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_PARENTHESIS_OPEN) {
				r_err_str="Expected '('";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream,token,line,r_err_str);

			if (token.type!=TK_IDENTIFIER) {
				r_err_str="Expected identifier";
				return ERR_PARSE_ERROR;
			}


			String id = token.value;

			InputEvent ie;

			if (id=="KEY") {

				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_COMMA) {
					r_err_str="Expected ','";
					return ERR_PARSE_ERROR;
				}

				ie.type=InputEvent::KEY;


				get_token(p_stream,token,line,r_err_str);
				if (token.type==TK_IDENTIFIER) {
					String name=token.value;
					ie.key.scancode=find_keycode(name);
				} else if (token.type==TK_NUMBER) {

					ie.key.scancode=token.value;
				} else {

					r_err_str="Expected string or integer for keycode";
					return ERR_PARSE_ERROR;
				}

				get_token(p_stream,token,line,r_err_str);

				if (token.type==TK_COMMA) {

					get_token(p_stream,token,line,r_err_str);

					if (token.type!=TK_IDENTIFIER) {
						r_err_str="Expected identifier with modifier flas";
						return ERR_PARSE_ERROR;
					}

					String mods=token.value;

					if (mods.findn("C")!=-1)
						ie.key.mod.control=true;
					if (mods.findn("A")!=-1)
						ie.key.mod.alt=true;
					if (mods.findn("S")!=-1)
						ie.key.mod.shift=true;
					if (mods.findn("M")!=-1)
						ie.key.mod.meta=true;

					get_token(p_stream,token,line,r_err_str);
					if (token.type!=TK_PARENTHESIS_CLOSE) {
						r_err_str="Expected ')'";
						return ERR_PARSE_ERROR;
					}

				} else if (token.type!=TK_PARENTHESIS_CLOSE) {

					r_err_str="Expected ')' or modifier flags.";
					return ERR_PARSE_ERROR;
				}


			} else if (id=="MBUTTON") {

				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_COMMA) {
					r_err_str="Expected ','";
					return ERR_PARSE_ERROR;
				}

				ie.type=InputEvent::MOUSE_BUTTON;

				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_NUMBER) {
					r_err_str="Expected button index";
					return ERR_PARSE_ERROR;
				}

				ie.mouse_button.button_index = token.value;

				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_PARENTHESIS_CLOSE) {
					r_err_str="Expected ')'";
					return ERR_PARSE_ERROR;
				}

			} else if (id=="JBUTTON") {

				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_COMMA) {
					r_err_str="Expected ','";
					return ERR_PARSE_ERROR;
				}

				ie.type=InputEvent::JOYSTICK_BUTTON;

				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_NUMBER) {
					r_err_str="Expected button index";
					return ERR_PARSE_ERROR;
				}

				ie.joy_button.button_index = token.value;

				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_PARENTHESIS_CLOSE) {
					r_err_str="Expected ')'";
					return ERR_PARSE_ERROR;
				}

			} else if (id=="JAXIS") {

				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_COMMA) {
					r_err_str="Expected ','";
					return ERR_PARSE_ERROR;
				}

				ie.type=InputEvent::JOYSTICK_MOTION;

				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_NUMBER) {
					r_err_str="Expected axis index";
					return ERR_PARSE_ERROR;
				}

				ie.joy_motion.axis = token.value;

				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_PARENTHESIS_CLOSE) {
					r_err_str="Expected ')'";
					return ERR_PARSE_ERROR;
				}

			} else {

				r_err_str="Invalid input event type.";
				return ERR_PARSE_ERROR;
			}

			value=ie;

			return OK;

		} else if (id=="ByteArray") {

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			DVector<uint8_t> arr;
			{
				int len=args.size();
				arr.resize(len);
				DVector<uint8_t>::Write w = arr.write();
				for(int i=0;i<len;i++) {
					w[i]=args[i];
				}
			}

			value=arr;

			return OK;

		} else if (id=="IntArray") {

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			DVector<int32_t> arr;
			{
				int len=args.size();
				arr.resize(len);
				DVector<int32_t>::Write w = arr.write();
				for(int i=0;i<len;i++) {
					w[i]=Math::fast_ftoi(args[i]);
				}
			}

			value=arr;

			return OK;

		} else if (id=="FloatArray") {

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			DVector<float> arr;
			{
				int len=args.size();
				arr.resize(len);
				DVector<float>::Write w = arr.write();
				for(int i=0;i<len;i++) {
					w[i]=args[i];
				}
			}

			value=arr;

			return OK;
		} else if (id=="StringArray") {


			get_token(p_stream,token,line,r_err_str);
			if (token.type!=TK_PARENTHESIS_OPEN) {
				r_err_str="Expected '('";
				return ERR_PARSE_ERROR;
			}

			Vector<String> cs;

			bool first=true;
			while(true) {

				if (!first) {
					get_token(p_stream,token,line,r_err_str);
					if (token.type==TK_COMMA) {
						//do none
					} else if (token.type!=TK_PARENTHESIS_CLOSE) {
						break;
					} else {
						r_err_str="Expected ',' or ')'";
						return ERR_PARSE_ERROR;

					}
				}
				get_token(p_stream,token,line,r_err_str);
				if (token.type!=TK_STRING) {
					r_err_str="Expected string";
					return ERR_PARSE_ERROR;
				}

				cs.push_back(token.value);
			}


			DVector<String> arr;
			{
				int len=cs.size();
				arr.resize(len);
				DVector<String>::Write w = arr.write();
				for(int i=0;i<len;i++) {
					w[i]=cs[i];
				}
			}

			value=arr;

			return OK;


		} else if (id=="Vector2Array") {

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			DVector<Vector2> arr;
			{
				int len=args.size()/2;
				arr.resize(len);
				DVector<Vector2>::Write w = arr.write();
				for(int i=0;i<len;i++) {
					w[i]=Vector2(args[i*2+0],args[i*2+1]);
				}
			}

			value=arr;

			return OK;

		} else if (id=="Vector3Array") {

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			DVector<Vector3> arr;
			{
				int len=args.size()/3;
				arr.resize(len);
				DVector<Vector3>::Write w = arr.write();
				for(int i=0;i<len;i++) {
					w[i]=Vector3(args[i*3+0],args[i*3+1],args[i*3+2]);
				}
			}

			value=arr;

			return OK;

		} else if (id=="ColorArray") {

			Vector<float> args;
			Error err = _parse_construct(p_stream,args,line,r_err_str);
			if (err)
				return err;

			DVector<Color> arr;
			{
				int len=args.size()/4;
				arr.resize(len);
				DVector<Color>::Write w = arr.write();
				for(int i=0;i<len;i++) {
					w[i]=Color(args[i*3+0],args[i*3+1],args[i*3+2],args[i*3+3]);
				}
			}

			value=arr;

			return OK;

		} else {
			r_err_str="Unexpected identifier: '"+id+"'.";
			return ERR_PARSE_ERROR;
		}


		/*
				VECTOR2,		// 5
				RECT2,
				VECTOR3,
				MATRIX32,
				PLANE,
				QUAT,			// 10
				_AABB, //sorry naming convention fail :( not like it's used often
				MATRIX3,
				TRANSFORM,

				// misc types
				COLOR,
				IMAGE,			// 15
				NODE_PATH,
				_RID,
				OBJECT,
				INPUT_EVENT,
				DICTIONARY,		// 20
				ARRAY,

				// arrays
				RAW_ARRAY,
				INT_ARRAY,
				REAL_ARRAY,
				STRING_ARRAY,	// 25
				VECTOR2_ARRAY,
				VECTOR3_ARRAY,
				COLOR_ARRAY,

				VARIANT_MAX

		*/

		return OK;

	} else if (token.type==TK_NUMBER) {

		value=token.value;
		return OK;
	} else if (token.type==TK_STRING) {

		value=token.value;
		return OK;
	} else {
		r_err_str="Expected value, got "+String(tk_name[token.type])+".";
		return ERR_PARSE_ERROR;
	}

	return ERR_PARSE_ERROR;
}


Error VariantParser::_parse_array(Array &array, Stream *p_stream, int &line, String &r_err_str, ResourceParser *p_res_parser) {

	Token token;
	bool need_comma=false;


	while(!p_stream->is_eof()) {

		Error err = get_token(p_stream,token,line,r_err_str);
		if (err!=OK)
			return err;

		if (token.type==TK_BRACKET_CLOSE) {

			return OK;
		}

		if (need_comma) {

			if (token.type!=TK_COMMA) {

				r_err_str="Expected ','";
				return ERR_PARSE_ERROR;
			} else {
				need_comma=false;
				continue;
			}
		}

		Variant v;
		err = parse_value(token,v,p_stream,line,r_err_str,p_res_parser);
		if (err)
			return err;

		array.push_back(v);
		need_comma=true;

	}

	return OK;

}

Error VariantParser::_parse_dictionary(Dictionary &object, Stream *p_stream, int &line, String &r_err_str, ResourceParser *p_res_parser) {

	bool at_key=true;
	Variant key;
	Token token;
	bool need_comma=false;


	while(!p_stream->is_eof()) {


		if (at_key) {

			Error err = get_token(p_stream,token,line,r_err_str);
			if (err!=OK)
				return err;

			if (token.type==TK_CURLY_BRACKET_CLOSE) {

				return OK;
			}

			if (need_comma) {

				if (token.type!=TK_COMMA) {

					r_err_str="Expected '}' or ','";
					return ERR_PARSE_ERROR;
				} else {
					need_comma=false;
					continue;
				}
			}




			err = parse_value(token,key,p_stream,line,r_err_str,p_res_parser);

			if (err)
				return err;

			err = get_token(p_stream,token,line,r_err_str);

			if (err!=OK)
				return err;
			if (token.type!=TK_COLON) {

				r_err_str="Expected ':'";
				return ERR_PARSE_ERROR;
			}
			at_key=false;
		} else {


			Error err = get_token(p_stream,token,line,r_err_str);
			if (err!=OK)
				return err;

			Variant v;
			err = parse_value(token,v,p_stream,line,r_err_str,p_res_parser);
			if (err)
				return err;
			object[key]=v;
			need_comma=true;
			at_key=true;
		}
	}

	return OK;
}


Error VariantParser::_parse_tag(Token& token,Stream *p_stream, int &line, String &r_err_str,Tag& r_tag) {

	r_tag.fields.clear();

	if (token.type!=TK_BRACKET_OPEN) {
		r_err_str="Expected '['";
		return ERR_PARSE_ERROR;
	}


	get_token(p_stream,token,line,r_err_str);


	if (token.type!=TK_IDENTIFIER) {
		r_err_str="Expected identifier (tag name)";
		return ERR_PARSE_ERROR;
	}

	r_tag.name=token.value;

	print_line("tag name: "+r_tag.name);

	while(true) {

		get_token(p_stream,token,line,r_err_str);
		if (token.type==TK_BRACKET_CLOSE)
			break;

		if (token.type!=TK_IDENTIFIER) {
			r_err_str="Expected Identifier";
			return ERR_PARSE_ERROR;
		}

		String id=token.value;

		print_line("got ID: "+id);

		get_token(p_stream,token,line,r_err_str);
		if (token.type!=TK_EQUAL) {
			r_err_str="Expected '='";
			return ERR_PARSE_ERROR;
		}

		print_line("got tk: "+String(tk_name[token.type]));

		get_token(p_stream,token,line,r_err_str);
		Variant value;
		Error err = parse_value(token,value,p_stream,line,r_err_str);
		if (err)
			return err;

		print_line("id: "+id+" value: "+String(value));

		r_tag.fields[id]=value;

	}


	return OK;

}

Error VariantParser::parse_tag(Stream *p_stream, int &line, String &r_err_str,Tag& r_tag) {

	Token token;
	get_token(p_stream,token,line,r_err_str);
	if (token.type!=TK_BRACKET_OPEN) {
		r_err_str="Expected '['";
		return ERR_PARSE_ERROR;
	}

	return _parse_tag(token,p_stream,line,r_err_str,r_tag);

}

Error VariantParser::parse_tag_assign_eof(Stream *p_stream, int &line, String &r_err_str,Tag& r_tag,String &r_assign) {

	r_tag.name.clear();
	r_assign=String();

	return OK;
}

Error VariantParser::parse(Stream *p_stream, Variant& r_ret, String &r_err_str, int &r_err_line, ResourceParser *p_res_parser) {


	Token token;
	Error err = get_token(p_stream,token,r_err_line,r_err_str);
	if (err)
		return err;
	return parse_value(token,r_ret,p_stream,r_err_line,r_err_str,p_res_parser);

}


