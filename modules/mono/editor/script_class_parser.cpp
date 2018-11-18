#include "script_class_parser.h"

#include "core/map.h"
#include "core/os/os.h"

#include "../utils/string_utils.h"

const char *ScriptClassParser::token_names[ScriptClassParser::TK_MAX] = {
	"[",
	"]",
	"{",
	"}",
	".",
	":",
	",",
	"Symbol",
	"Identifier",
	"String",
	"Number",
	"<",
	">",
	"EOF",
	"Error"
};

String ScriptClassParser::get_token_name(ScriptClassParser::Token p_token) {

	ERR_FAIL_INDEX_V(p_token, TK_MAX, "<error>");
	return token_names[p_token];
}

ScriptClassParser::Token ScriptClassParser::get_token() {

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
			case '<': {
				idx++;
				return TK_OP_LESS;
			};
			case '>': {
				idx++;
				return TK_OP_GREATER;
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
				bool verbatim = idx != 0 && code[idx - 1] == '@';

				CharType begin_str = code[idx];
				idx++;
				String tk_string = String();
				while (true) {
					if (code[idx] == 0) {
						error_str = "Unterminated String";
						error = true;
						return TK_ERROR;
					} else if (code[idx] == begin_str) {
						if (verbatim && code[idx + 1] == '"') { // `""` is verbatim string's `\"`
							idx += 2; // skip next `"` as well
							continue;
						}

						idx += 1;
						break;
					} else if (code[idx] == '\\' && !verbatim) {
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
							case '\"': res = '\"'; break;
							case '\\':
								res = '\\';
								break;
							default: {
								res = next;
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

				if ((code[idx] >= 33 && code[idx] <= 47) || (code[idx] >= 58 && code[idx] <= 63) || (code[idx] >= 91 && code[idx] <= 94) || code[idx] == 96 || (code[idx] >= 123 && code[idx] <= 127)) {
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

				} else if ((code[idx] == '@' && code[idx + 1] != '"') || code[idx] == '_' || (code[idx] >= 'A' && code[idx] <= 'Z') || (code[idx] >= 'a' && code[idx] <= 'z') || code[idx] > 127) {
					String id;

					id += code[idx];
					idx++;

					while (code[idx] == '_' || (code[idx] >= 'A' && code[idx] <= 'Z') || (code[idx] >= 'a' && code[idx] <= 'z') || (code[idx] >= '0' && code[idx] <= '9') || code[idx] > 127) {
						id += code[idx];
						idx++;
					}

					value = id;
					return TK_IDENTIFIER;
				} else if (code[idx] == '@' && code[idx + 1] == '"') {
					// begin of verbatim string
					idx++;
				} else {
					error_str = "Unexpected character.";
					error = true;
					return TK_ERROR;
				}
			}
		}
	}
}

Error ScriptClassParser::_skip_generic_type_params() {

	Token tk;

	while (true) {
		tk = get_token();

		if (tk == TK_IDENTIFIER) {
			tk = get_token();

			if (tk == TK_PERIOD) {
				while (true) {
					tk = get_token();

					if (tk != TK_IDENTIFIER) {
						error_str = "Expected " + get_token_name(TK_IDENTIFIER) + ", found: " + get_token_name(tk);
						error = true;
						return ERR_PARSE_ERROR;
					}

					tk = get_token();

					if (tk != TK_PERIOD)
						break;
				}
			}

			if (tk == TK_OP_LESS) {
				Error err = _skip_generic_type_params();
				if (err)
					return err;
				continue;
			} else if (tk != TK_COMMA) {
				error_str = "Unexpected token: " + get_token_name(tk);
				error = true;
				return ERR_PARSE_ERROR;
			}
		} else if (tk == TK_OP_LESS) {
			error_str = "Expected " + get_token_name(TK_IDENTIFIER) + ", found " + get_token_name(TK_OP_LESS);
			error = true;
			return ERR_PARSE_ERROR;
		} else if (tk == TK_OP_GREATER) {
			return OK;
		} else {
			error_str = "Unexpected token: " + get_token_name(tk);
			error = true;
			return ERR_PARSE_ERROR;
		}
	}
}

Error ScriptClassParser::_parse_type_full_name(String &r_full_name) {

	Token tk = get_token();

	if (tk != TK_IDENTIFIER) {
		error_str = "Expected " + get_token_name(TK_IDENTIFIER) + ", found: " + get_token_name(tk);
		error = true;
		return ERR_PARSE_ERROR;
	}

	r_full_name += String(value);

	if (code[idx] != '.') // We only want to take the next token if it's a period
		return OK;

	tk = get_token();

	CRASH_COND(tk != TK_PERIOD); // Assertion

	r_full_name += ".";

	return _parse_type_full_name(r_full_name);
}

Error ScriptClassParser::_parse_class_base(Vector<String> &r_base) {

	String name;

	Error err = _parse_type_full_name(name);
	if (err)
		return err;

	Token tk = get_token();

	if (tk == TK_OP_LESS) {
		// We don't add it to the base list if it's generic
		Error err = _skip_generic_type_params();
		if (err)
			return err;
	} else if (tk == TK_COMMA) {
		Error err = _parse_class_base(r_base);
		if (err)
			return err;
		r_base.push_back(name);
	} else if (tk == TK_CURLY_BRACKET_OPEN) {
		r_base.push_back(name);
	} else {
		error_str = "Unexpected token: " + get_token_name(tk);
		error = true;
		return ERR_PARSE_ERROR;
	}

	return OK;
}

Error ScriptClassParser::_parse_namespace_name(String &r_name, int &r_curly_stack) {

	Token tk = get_token();

	if (tk == TK_IDENTIFIER) {
		r_name += String(value);
	} else {
		error_str = "Unexpected token: " + get_token_name(tk);
		error = true;
		return ERR_PARSE_ERROR;
	}

	tk = get_token();

	if (tk == TK_PERIOD) {
		r_name += ".";
		return _parse_namespace_name(r_name, r_curly_stack);
	} else if (tk == TK_CURLY_BRACKET_OPEN) {
		r_curly_stack++;
		return OK;
	} else {
		error_str = "Unexpected token: " + get_token_name(tk);
		error = true;
		return ERR_PARSE_ERROR;
	}
}

Error ScriptClassParser::parse(const String &p_code) {

	code = p_code;
	idx = 0;
	line = 0;
	error_str = String();
	error = false;
	value = Variant();
	classes.clear();

	Token tk = get_token();

	Map<int, NameDecl> name_stack;
	int curly_stack = 0;
	int type_curly_stack = 0;

	while (!error && tk != TK_EOF) {
		if (tk == TK_IDENTIFIER && String(value) == "class") {
			tk = get_token();

			if (tk == TK_IDENTIFIER) {
				String name = value;
				int at_level = type_curly_stack;

				ClassDecl class_decl;

				for (Map<int, NameDecl>::Element *E = name_stack.front(); E; E = E->next()) {
					const NameDecl &name_decl = E->value();

					if (name_decl.type == NameDecl::NAMESPACE_DECL) {
						if (E != name_stack.front())
							class_decl.namespace_ += ".";
						class_decl.namespace_ += name_decl.name;
					} else {
						class_decl.name += name_decl.name + ".";
					}
				}

				class_decl.name += name;
				class_decl.nested = type_curly_stack > 0;

				bool generic = false;

				while (true) {
					tk = get_token();

					if (tk == TK_COLON) {
						Error err = _parse_class_base(class_decl.base);
						if (err)
							return err;

						curly_stack++;
						type_curly_stack++;

						break;
					} else if (tk == TK_CURLY_BRACKET_OPEN) {
						curly_stack++;
						type_curly_stack++;
						break;
					} else if (tk == TK_OP_LESS && !generic) {
						generic = true;

						Error err = _skip_generic_type_params();
						if (err)
							return err;
					} else {
						error_str = "Unexpected token: " + get_token_name(tk);
						error = true;
						return ERR_PARSE_ERROR;
					}
				}

				NameDecl name_decl;
				name_decl.name = name;
				name_decl.type = NameDecl::CLASS_DECL;
				name_stack[at_level] = name_decl;

				if (!generic) { // no generics, thanks
					classes.push_back(class_decl);
				} else if (OS::get_singleton()->is_stdout_verbose()) {
					String full_name = class_decl.namespace_;
					if (full_name.length())
						full_name += ".";
					full_name += class_decl.name;
					OS::get_singleton()->print(String("Ignoring generic class declaration: " + class_decl.name).utf8());
				}
			}
		} else if (tk == TK_IDENTIFIER && String(value) == "struct") {
			String name;
			int at_level = type_curly_stack;
			while (true) {
				tk = get_token();
				if (tk == TK_IDENTIFIER && name.empty()) {
					name = String(value);
				} else if (tk == TK_CURLY_BRACKET_OPEN) {
					if (name.empty()) {
						error_str = "Expected " + get_token_name(TK_IDENTIFIER) + " after keyword `struct`, found " + get_token_name(TK_CURLY_BRACKET_OPEN);
						error = true;
						return ERR_PARSE_ERROR;
					}

					curly_stack++;
					type_curly_stack++;
					break;
				} else if (tk == TK_EOF) {
					error_str = "Expected " + get_token_name(TK_CURLY_BRACKET_OPEN) + " after struct decl, found " + get_token_name(TK_EOF);
					error = true;
					return ERR_PARSE_ERROR;
				}
			}

			NameDecl name_decl;
			name_decl.name = name;
			name_decl.type = NameDecl::STRUCT_DECL;
			name_stack[at_level] = name_decl;
		} else if (tk == TK_IDENTIFIER && String(value) == "namespace") {
			if (type_curly_stack > 0) {
				error_str = "Found namespace nested inside type.";
				error = true;
				return ERR_PARSE_ERROR;
			}

			String name;
			int at_level = curly_stack;

			Error err = _parse_namespace_name(name, curly_stack);
			if (err)
				return err;

			NameDecl name_decl;
			name_decl.name = name;
			name_decl.type = NameDecl::NAMESPACE_DECL;
			name_stack[at_level] = name_decl;
		} else if (tk == TK_CURLY_BRACKET_OPEN) {
			curly_stack++;
		} else if (tk == TK_CURLY_BRACKET_CLOSE) {
			curly_stack--;
			if (name_stack.has(curly_stack)) {
				if (name_stack[curly_stack].type != NameDecl::NAMESPACE_DECL)
					type_curly_stack--;
				name_stack.erase(curly_stack);
			}
		}

		tk = get_token();
	}

	if (!error && tk == TK_EOF && curly_stack > 0) {
		error_str = "Reached EOF with missing close curly brackets.";
		error = true;
	}

	if (error)
		return ERR_PARSE_ERROR;

	return OK;
}

Error ScriptClassParser::parse_file(const String &p_filepath) {

	String source;

	Error ferr = read_all_file_utf8(p_filepath, source);
	if (ferr != OK) {
		if (ferr == ERR_INVALID_DATA) {
			ERR_EXPLAIN("File '" + p_filepath + "' contains invalid unicode (utf-8), so it was not loaded. Please ensure that scripts are saved in valid utf-8 unicode.");
		}
		ERR_FAIL_V(ferr);
	}

	return parse(source);
}

String ScriptClassParser::get_error() {
	return error_str;
}

Vector<ScriptClassParser::ClassDecl> ScriptClassParser::get_classes() {
	return classes;
}
