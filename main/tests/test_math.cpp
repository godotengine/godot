/*************************************************************************/
/*  test_math.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "test_math.h"

#include "camera_matrix.h"
#include "math_funcs.h"
#include "matrix3.h"
#include "os/file_access.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "print_string.h"
#include "scene/main/node.h"
#include "scene/resources/texture.h"
#include "servers/visual/shader_language.h"
#include "transform.h"
#include "ustring.h"
#include "variant.h"
#include "vmap.h"

#include "method_ptrcall.h"

namespace TestMath {

class GetClassAndNamespace {

	String code;
	int idx;
	int line;
	String error_str;
	bool error;
	Variant value;

	String class_name;

	enum Token {
		TK_BRACKET_OPEN,
		TK_BRACKET_CLOSE,
		TK_CURLY_BRACKET_OPEN,
		TK_CURLY_BRACKET_CLOSE,
		TK_PERIOD,
		TK_COLON,
		TK_COMMA,
		TK_SYMBOL,
		TK_IDENTIFIER,
		TK_STRING,
		TK_NUMBER,
		TK_EOF,
		TK_ERROR
	};

	Token get_token() {

		while (true) {
			switch (code[idx]) {

				case '\n': {

					line++;
					idx++;
					break;
				};
				case 0: {
					return TK_EOF;

				} break;
				case '{': {

					idx++;
					return TK_CURLY_BRACKET_OPEN;
				};
				case '}': {

					idx++;
					return TK_CURLY_BRACKET_CLOSE;
				};
				case '[': {

					idx++;
					return TK_BRACKET_OPEN;
				};
				case ']': {

					idx++;
					return TK_BRACKET_CLOSE;
				};
				case ':': {

					idx++;
					return TK_COLON;
				};
				case ',': {

					idx++;
					return TK_COMMA;
				};
				case '.': {

					idx++;
					return TK_PERIOD;
				};
				case '#': {
					//compiler directive
					while (code[idx] != '\n' && code[idx] != 0) {
						idx++;
					}
					continue;
				} break;
				case '/': {

					switch (code[idx + 1]) {
						case '*': { // block comment

							idx += 2;
							while (true) {
								if (code[idx] == 0) {
									error_str = "Unterminated comment";
									error = true;
									return TK_ERROR;
								} else if (code[idx] == '*' && code[idx + 1] == '/') {

									idx += 2;
									break;
								} else if (code[idx] == '\n') {
									line++;
								}

								idx++;
							}

						} break;
						case '/': { // line comment skip

							while (code[idx] != '\n' && code[idx] != 0) {
								idx++;
							}

						} break;
						default: {
							value = "/";
							idx++;
							return TK_SYMBOL;
						}
					}

					continue; // a comment
				} break;
				case '\'':
				case '"': {

					CharType begin_str = code[idx];
					idx++;
					String tk_string = String();
					while (true) {
						if (code[idx] == 0) {
							error_str = "Unterminated String";
							error = true;
							return TK_ERROR;
						} else if (code[idx] == begin_str) {
							idx++;
							break;
						} else if (code[idx] == '\\') {
							//escaped characters...
							idx++;
							CharType next = code[idx];
							if (next == 0) {
								error_str = "Unterminated String";
								error = true;
								return TK_ERROR;
							}
							CharType res = 0;

							switch (next) {

								case 'b': res = 8; break;
								case 't': res = 9; break;
								case 'n': res = 10; break;
								case 'f': res = 12; break;
								case 'r':
									res = 13;
									break;
								/* too much, not needed for now
								case 'u': {
									//hexnumbarh - oct is deprecated


									for(int j=0;j<4;j++) {
										CharType c = code[idx+j+1];
										if (c==0) {
											r_err_str="Unterminated String";
											return ERR_PARSE_ERROR;
										}
										if (!((c>='0' && c<='9') || (c>='a' && c<='f') || (c>='A' && c<='F'))) {

											r_err_str="Malformed hex constant in string";
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
									idx+=4; //will add at the end anyway


								} break;*/
								case '\"': res = '\"'; break;
								case '\\':
									res = '\\';
									break;
								//case '/': res='/'; break;
								default: {
									res = next;
									//r_err_str="Invalid escape sequence";
									//return ERR_PARSE_ERROR;
								} break;
							}

							tk_string += res;

						} else {
							if (code[idx] == '\n')
								line++;
							tk_string += code[idx];
						}
						idx++;
					}

					value = tk_string;

					return TK_STRING;

				} break;
				default: {

					if (code[idx] <= 32) {
						idx++;
						break;
					}

					if ((code[idx] >= 33 && code[idx] <= 47) || (code[idx] >= 58 && code[idx] <= 64) || (code[idx] >= 91 && code[idx] <= 96) || (code[idx] >= 123 && code[idx] <= 127)) {
						value = String::chr(code[idx]);
						idx++;
						return TK_SYMBOL;
					}

					if (code[idx] == '-' || (code[idx] >= '0' && code[idx] <= '9')) {
						//a number
						const CharType *rptr;
						double number = String::to_double(&code[idx], &rptr);
						idx += (rptr - &code[idx]);
						value = number;
						return TK_NUMBER;

					} else if ((code[idx] >= 'A' && code[idx] <= 'Z') || (code[idx] >= 'a' && code[idx] <= 'z') || code[idx] > 127) {

						String id;

						while ((code[idx] >= 'A' && code[idx] <= 'Z') || (code[idx] >= 'a' && code[idx] <= 'z') || code[idx] > 127) {

							id += code[idx];
							idx++;
						}

						value = id;
						return TK_IDENTIFIER;
					} else {
						error_str = "Unexpected character.";
						error = true;
						return TK_ERROR;
					}
				}
			}
		}
	}

public:
	Error parse(const String &p_code, const String &p_known_class_name = String()) {

		code = p_code;
		idx = 0;
		line = 0;
		error_str = String();
		error = false;
		value = Variant();
		class_name = String();

		bool use_next_class = false;
		Token tk = get_token();

		Map<int, String> namespace_stack;
		int curly_stack = 0;

		while (!error || tk != TK_EOF) {

			if (tk == TK_BRACKET_OPEN) {
				tk = get_token();
				if (tk == TK_IDENTIFIER && String(value) == "ScriptClass") {
					if (get_token() == TK_BRACKET_CLOSE) {
						use_next_class = true;
					}
				}
			} else if (tk == TK_IDENTIFIER && String(value) == "class") {
				tk = get_token();
				if (tk == TK_IDENTIFIER) {
					String name = value;
					if (use_next_class || p_known_class_name == name) {
						for (Map<int, String>::Element *E = namespace_stack.front(); E; E = E->next()) {
							class_name += E->get() + ".";
						}
						class_name += String(value);
						break;
					}
				}

			} else if (tk == TK_IDENTIFIER && String(value) == "namespace") {
				String name;
				int at_level = curly_stack;
				while (true) {
					tk = get_token();
					if (tk == TK_IDENTIFIER) {
						name += String(value);
					}

					tk = get_token();
					if (tk == TK_PERIOD) {
						name += ".";
					} else if (tk == TK_CURLY_BRACKET_OPEN) {
						curly_stack++;
						break;
					} else {
						break; //whathever else
					}
				}

				if (name != String()) {
					namespace_stack[at_level] = name;
				}

			} else if (tk == TK_CURLY_BRACKET_OPEN) {
				curly_stack++;
			} else if (tk == TK_CURLY_BRACKET_CLOSE) {
				curly_stack--;
				if (namespace_stack.has(curly_stack)) {
					namespace_stack.erase(curly_stack);
				}
			}

			tk = get_token();
		}

		if (error)
			return ERR_PARSE_ERROR;

		return OK;
	}

	String get_error() {
		return error_str;
	}

	String get_class() {
		return class_name;
	}
};

void test_vec(Plane p_vec) {

	CameraMatrix cm;
	cm.set_perspective(45, 1, 0, 100);
	Plane v0 = cm.xform4(p_vec);

	print_line("out: " + v0);
	v0.normal.z = (v0.d / 100.0 * 2.0 - 1.0) * v0.d;
	print_line("out_F: " + v0);

	/*v0: 0, 0, -0.1, 0.1
v1: 0, 0, 0, 0.1
fix: 0, 0, 0, 0.1
v0: 0, 0, 1.302803, 1.5
v1: 0, 0, 1.401401, 1.5
fix: 0, 0, 1.401401, 1.5
v0: 0, 0, 25.851850, 26
v1: 0, 0, 25.925926, 26
fix: 0, 0, 25.925924, 26
v0: 0, 0, 49.899902, 50
v1: 0, 0, 49.949947, 50
fix: 0, 0, 49.949951, 50
v0: 0, 0, 100, 100
v1: 0, 0, 100, 100
fix: 0, 0, 100, 100
*/
}

uint32_t ihash(uint32_t a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

uint32_t ihash2(uint32_t a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

uint32_t ihash3(uint32_t a) {
	a = (a + 0x479ab41d) + (a << 8);
	a = (a ^ 0xe4aa10ce) ^ (a >> 5);
	a = (a + 0x9942f0a6) - (a << 14);
	a = (a ^ 0x5aedd67d) ^ (a >> 3);
	a = (a + 0x17bea992) + (a << 7);
	return a;
}

MainLoop *test() {

	print_line("Dvectors: " + itos(MemoryPool::allocs_used));
	print_line("Mem used: " + itos(MemoryPool::total_memory));
	print_line("MAx mem used: " + itos(MemoryPool::max_memory));

	PoolVector<int> ints;
	ints.resize(20);

	{
		PoolVector<int>::Write w;
		w = ints.write();
		for (int i = 0; i < ints.size(); i++) {
			w[i] = i;
		}
	}

	PoolVector<int> posho = ints;

	{
		PoolVector<int>::Read r = posho.read();
		for (int i = 0; i < posho.size(); i++) {
			print_line(itos(i) + " : " + itos(r[i]));
		}
	}

	print_line("later Dvectors: " + itos(MemoryPool::allocs_used));
	print_line("later Mem used: " + itos(MemoryPool::total_memory));
	print_line("Mlater Ax mem used: " + itos(MemoryPool::max_memory));

	return NULL;

	List<String> cmdlargs = OS::get_singleton()->get_cmdline_args();

	if (cmdlargs.empty()) {
		//try editor!
		return NULL;
	}

	String test = cmdlargs.back()->get();

	FileAccess *fa = FileAccess::open(test, FileAccess::READ);

	if (!fa) {
		ERR_EXPLAIN("Could not open file: " + test);
		ERR_FAIL_V(NULL);
	}

	Vector<uint8_t> buf;
	int flen = fa->get_len();
	buf.resize(fa->get_len() + 1);
	fa->get_buffer(&buf[0], flen);
	buf[flen] = 0;

	String code;
	code.parse_utf8((const char *)&buf[0]);

	GetClassAndNamespace getclass;
	if (getclass.parse(code)) {
		print_line("Parse error: " + getclass.get_error());
	} else {
		print_line("Found class: " + getclass.get_class());
	}

	return NULL;

	{

		Vector<int> hashes;
		List<StringName> tl;
		ClassDB::get_class_list(&tl);

		for (List<StringName>::Element *E = tl.front(); E; E = E->next()) {

			Vector<uint8_t> m5b = E->get().operator String().md5_buffer();
			hashes.push_back(hashes.size());
		}

		//hashes.resize(50);

		for (int i = nearest_shift(hashes.size()); i < 20; i++) {

			bool success = true;
			for (int s = 0; s < 10000; s++) {
				Set<uint32_t> existing;
				success = true;

				for (int j = 0; j < hashes.size(); j++) {

					uint32_t eh = ihash2(ihash3(hashes[j] + ihash(s) + s)) & ((1 << i) - 1);
					if (existing.has(eh)) {
						success = false;
						break;
					}
					existing.insert(eh);
				}

				if (success) {
					print_line("success at " + itos(i) + "/" + itos(nearest_shift(hashes.size())) + " shift " + itos(s));
					break;
				}
			}
			if (success)
				break;
		}

		print_line("DONE");

		return NULL;
	}
	{

		//print_line("NUM: "+itos(237641278346127));
		print_line("NUM: " + itos(-128));
		return NULL;
	}

	{
		Vector3 v(1, 2, 3);
		v.normalize();
		float a = 0.3;

		//Quat q(v,a);
		Basis m(v, a);

		Vector3 v2(7, 3, 1);
		v2.normalize();
		float a2 = 0.8;

		//Quat q(v,a);
		Basis m2(v2, a2);

		Quat q = m;
		Quat q2 = m2;

		Basis m3 = m.inverse() * m2;
		Quat q3 = (q.inverse() * q2); //.normalized();

		print_line(Quat(m3));
		print_line(q3);

		print_line("before v: " + v + " a: " + rtos(a));
		q.get_axis_and_angle(v, a);
		print_line("after v: " + v + " a: " + rtos(a));
	}

	return NULL;
	String ret;

	List<String> args;
	args.push_back("-l");
	Error err = OS::get_singleton()->execute("/bin/ls", args, true, NULL, &ret);
	print_line("error: " + itos(err));
	print_line(ret);

	return NULL;
	Basis m3;
	m3.rotate(Vector3(1, 0, 0), 0.2);
	m3.rotate(Vector3(0, 1, 0), 1.77);
	m3.rotate(Vector3(0, 0, 1), 212);
	Basis m32;
	m32.set_euler(m3.get_euler());
	print_line("ELEULEEEEEEEEEEEEEEEEEER: " + m3.get_euler() + " vs " + m32.get_euler());

	return NULL;

	{

		Dictionary d;
		d["momo"] = 1;
		Dictionary b = d;
		b["44"] = 4;
	}

	return NULL;
	print_line("inters: " + rtos(Geometry::segment_intersects_circle(Vector2(-5, 0), Vector2(-2, 0), Vector2(), 1.0)));

	print_line("cross: " + Vector3(1, 2, 3).cross(Vector3(4, 5, 7)));
	print_line("dot: " + rtos(Vector3(1, 2, 3).dot(Vector3(4, 5, 7))));
	print_line("abs: " + Vector3(-1, 2, -3).abs());
	print_line("distance_to: " + rtos(Vector3(1, 2, 3).distance_to(Vector3(4, 5, 7))));
	print_line("distance_squared_to: " + rtos(Vector3(1, 2, 3).distance_squared_to(Vector3(4, 5, 7))));
	print_line("plus: " + (Vector3(1, 2, 3) + Vector3(Vector3(4, 5, 7))));
	print_line("minus: " + (Vector3(1, 2, 3) - Vector3(Vector3(4, 5, 7))));
	print_line("mul: " + (Vector3(1, 2, 3) * Vector3(Vector3(4, 5, 7))));
	print_line("div: " + (Vector3(1, 2, 3) / Vector3(Vector3(4, 5, 7))));
	print_line("mul scalar: " + (Vector3(1, 2, 3) * 2));
	print_line("premul scalar: " + (2 * Vector3(1, 2, 3)));
	print_line("div scalar: " + (Vector3(1, 2, 3) / 3.0));
	print_line("length: " + rtos(Vector3(1, 2, 3).length()));
	print_line("length squared: " + rtos(Vector3(1, 2, 3).length_squared()));
	print_line("normalized: " + Vector3(1, 2, 3).normalized());
	print_line("inverse: " + Vector3(1, 2, 3).inverse());

	{
		Vector3 v(4, 5, 7);
		v.normalize();
		print_line("normalize: " + v);
	}

	{
		Vector3 v(4, 5, 7);
		v += Vector3(1, 2, 3);
		print_line("+=: " + v);
	}

	{
		Vector3 v(4, 5, 7);
		v -= Vector3(1, 2, 3);
		print_line("-=: " + v);
	}

	{
		Vector3 v(4, 5, 7);
		v *= Vector3(1, 2, 3);
		print_line("*=: " + v);
	}

	{
		Vector3 v(4, 5, 7);
		v /= Vector3(1, 2, 3);
		print_line("/=: " + v);
	}

	{
		Vector3 v(4, 5, 7);
		v *= 2.0;
		print_line("scalar *=: " + v);
	}

	{
		Vector3 v(4, 5, 7);
		v /= 2.0;
		print_line("scalar /=: " + v);
	}

#if 0
	print_line(String("C:\\momo\\.\\popo\\..\\gongo").simplify_path());
	print_line(String("res://../popo/..//gongo").simplify_path());
	print_line(String("res://..").simplify_path());


	PoolVector<uint8_t> a;
	PoolVector<uint8_t> b;

	a.resize(20);
	b=a;
	b.resize(30);
	a=b;
#endif

#if 0
	String za = String::utf8("á");
	printf("unicode: %x\n",za[0]);
	CharString cs=za.utf8();
	for(int i=0;i<cs.size();i++) {
		uint32_t v = uint8_t(cs[i]);
		printf("%i - %x\n",i,v);
	}
	return NULL;

	print_line(String("C:\\window\\system\\momo").path_to("C:\\window\\momonga"));
	print_line(String("res://momo/sampler").path_to("res://pindonga"));
	print_line(String("/margarito/terere").path_to("/margarito/pilates"));
	print_line(String("/algo").path_to("/algo"));
	print_line(String("c:").path_to("c:\\"));
	print_line(String("/").path_to("/"));


	print_line(itos(sizeof(Variant)));
	return NULL;

	Vector<StringName> path;
	path.push_back("three");
	path.push_back("two");
	path.push_back("one");
	path.push_back("comeon");
	path.revert();

	NodePath np(path,true);

	print_line(np);


	return NULL;

	bool a=2;

	print_line(Variant(a));


	Transform2D mat2_1;
	mat2_1.rotate(0.5);
	Transform2D mat2_2;
	mat2_2.translate(Vector2(1,2));
	Transform2D mat2_3 = mat2_1 * mat2_2;
	mat2_3.affine_invert();

	print_line(mat2_3.elements[0]);
	print_line(mat2_3.elements[1]);
	print_line(mat2_3.elements[2]);



	Transform mat3_1;
	mat3_1.basis.rotate(Vector3(0,0,1),0.5);
	Transform mat3_2;
	mat3_2.translate(Vector3(1,2,0));
	Transform mat3_3 = mat3_1 * mat3_2;
	mat3_3.affine_invert();

	print_line(mat3_3.basis.get_axis(0));
	print_line(mat3_3.basis.get_axis(1));
	print_line(mat3_3.origin);

#endif
	return NULL;
}
}
