/*************************************************************************/
/*  gd_parser.cpp                                                        */
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
#include "gd_parser.h"
#include "gd_script.h"
#include "io/resource_loader.h"
#include "os/file_access.h"
#include "print_string.h"
#include "script_language.h"

template <class T>
T *GDParser::alloc_node() {

	T *t = memnew(T);

	t->next = list;
	list = t;

	if (!head)
		head = t;

	t->line = tokenizer->get_token_line();
	t->column = tokenizer->get_token_column();
	return t;
}

bool GDParser::_end_statement() {

	if (tokenizer->get_token() == GDTokenizer::TK_SEMICOLON) {
		tokenizer->advance();
		return true; //handle next
	} else if (tokenizer->get_token() == GDTokenizer::TK_NEWLINE || tokenizer->get_token() == GDTokenizer::TK_EOF) {
		return true; //will be handled properly
	}

	return false;
}

bool GDParser::_enter_indent_block(BlockNode *p_block) {

	if (tokenizer->get_token() != GDTokenizer::TK_COLON) {
		// report location at the previous token (on the previous line)
		int error_line = tokenizer->get_token_line(-1);
		int error_column = tokenizer->get_token_column(-1);
		_set_error("':' expected at end of line.", error_line, error_column);
		return false;
	}
	tokenizer->advance();

	if (tokenizer->get_token() != GDTokenizer::TK_NEWLINE) {

		// be more python-like
		int current = tab_level.back()->get();
		tab_level.push_back(current);
		return true;
		//_set_error("newline expected after ':'.");
		//return false;
	}

	while (true) {

		if (tokenizer->get_token() != GDTokenizer::TK_NEWLINE) {

			return false; //wtf
		} else if (tokenizer->get_token(1) != GDTokenizer::TK_NEWLINE) {

			int indent = tokenizer->get_token_line_indent();
			int current = tab_level.back()->get();
			if (indent <= current) {
				print_line("current: " + itos(current) + " indent: " + itos(indent));
				print_line("less than current");
				return false;
			}

			tab_level.push_back(indent);
			tokenizer->advance();
			return true;

		} else if (p_block) {

			NewLineNode *nl = alloc_node<NewLineNode>();
			nl->line = tokenizer->get_token_line();
			p_block->statements.push_back(nl);
		}

		tokenizer->advance(); // go to next newline
	}
}

bool GDParser::_parse_arguments(Node *p_parent, Vector<Node *> &p_args, bool p_static, bool p_can_codecomplete) {

	if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE) {
		tokenizer->advance();
	} else {

		parenthesis++;
		int argidx = 0;

		while (true) {

			if (tokenizer->get_token() == GDTokenizer::TK_CURSOR) {
				_make_completable_call(argidx);
				completion_node = p_parent;
			} else if (tokenizer->get_token() == GDTokenizer::TK_CONSTANT && tokenizer->get_token_constant().get_type() == Variant::STRING && tokenizer->get_token(1) == GDTokenizer::TK_CURSOR) {
				//completing a string argument..
				completion_cursor = tokenizer->get_token_constant();

				_make_completable_call(argidx);
				completion_node = p_parent;
				tokenizer->advance(1);
				return false;
			}

			Node *arg = _parse_expression(p_parent, p_static);
			if (!arg)
				return false;

			p_args.push_back(arg);

			if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE) {
				tokenizer->advance();
				break;

			} else if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {

				if (tokenizer->get_token(1) == GDTokenizer::TK_PARENTHESIS_CLOSE) {

					_set_error("Expression expected");
					return false;
				}

				tokenizer->advance();
				argidx++;
			} else {
				// something is broken
				_set_error("Expected ',' or ')'");
				return false;
			}
		}
		parenthesis--;
	}

	return true;
}

void GDParser::_make_completable_call(int p_arg) {

	completion_cursor = StringName();
	completion_type = COMPLETION_CALL_ARGUMENTS;
	completion_class = current_class;
	completion_function = current_function;
	completion_line = tokenizer->get_token_line();
	completion_argument = p_arg;
	completion_block = current_block;
	completion_found = true;
	tokenizer->advance();
}

bool GDParser::_get_completable_identifier(CompletionType p_type, StringName &identifier) {

	identifier = StringName();
	if (tokenizer->is_token_literal()) {
		identifier = tokenizer->get_token_literal();
		tokenizer->advance();
	}
	if (tokenizer->get_token() == GDTokenizer::TK_CURSOR) {

		completion_cursor = identifier;
		completion_type = p_type;
		completion_class = current_class;
		completion_function = current_function;
		completion_line = tokenizer->get_token_line();
		completion_block = current_block;
		completion_found = true;
		completion_ident_is_call = false;
		tokenizer->advance();

		if (tokenizer->is_token_literal()) {
			identifier = identifier.operator String() + tokenizer->get_token_literal().operator String();
			tokenizer->advance();
		}

		if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_OPEN) {
			completion_ident_is_call = true;
		}
		return true;
	}

	return false;
}

GDParser::Node *GDParser::_parse_expression(Node *p_parent, bool p_static, bool p_allow_assign, bool p_parsing_constant) {

	//Vector<Node*> expressions;
	//Vector<OperatorNode::Operator> operators;

	Vector<Expression> expression;

	Node *expr = NULL;

	int op_line = tokenizer->get_token_line(); // when operators are created at the bottom, the line might have been changed (\n found)

	while (true) {

		/*****************/
		/* Parse Operand */
		/*****************/

		if (parenthesis > 0) {
			//remove empty space (only allowed if inside parenthesis
			while (tokenizer->get_token() == GDTokenizer::TK_NEWLINE) {
				tokenizer->advance();
			}
		}

		if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_OPEN) {
			//subexpression ()
			tokenizer->advance();
			parenthesis++;
			Node *subexpr = _parse_expression(p_parent, p_static, p_allow_assign, p_parsing_constant);
			parenthesis--;
			if (!subexpr)
				return NULL;

			if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {

				_set_error("Expected ')' in expression");
				return NULL;
			}

			tokenizer->advance();
			expr = subexpr;
		} else if (tokenizer->get_token() == GDTokenizer::TK_DOLLAR) {
			tokenizer->advance();

			String path;

			bool need_identifier = true;
			bool done = false;

			while (!done) {

				switch (tokenizer->get_token()) {
					case GDTokenizer::TK_CURSOR: {
						completion_cursor = StringName();
						completion_type = COMPLETION_GET_NODE;
						completion_class = current_class;
						completion_function = current_function;
						completion_line = tokenizer->get_token_line();
						completion_cursor = path;
						completion_argument = 0;
						completion_block = current_block;
						completion_found = true;
						tokenizer->advance();
					} break;
					case GDTokenizer::TK_CONSTANT: {

						if (!need_identifier) {
							done = true;
							break;
						}

						if (tokenizer->get_token_constant().get_type() != Variant::STRING) {
							_set_error("Expected string constant or identifier after '$' or '/'.");
							return NULL;
						}

						path += String(tokenizer->get_token_constant());
						tokenizer->advance();
						need_identifier = false;

					} break;
					case GDTokenizer::TK_OP_DIV: {

						if (need_identifier) {
							done = true;
							break;
						}

						path += "/";
						tokenizer->advance();
						need_identifier = true;

					} break;
					default: {
						// Instead of checking for TK_IDENTIFIER, we check with is_token_literal, as this allows us to use match/sync/etc. as a name
						if (need_identifier && tokenizer->is_token_literal()) {
							path += String(tokenizer->get_token_literal());
							tokenizer->advance();
							need_identifier = false;
						} else {
							done = true;
						}

						break;
					}
				}
			}

			if (path == "") {
				_set_error("Path expected after $.");
				return NULL;
			}

			OperatorNode *op = alloc_node<OperatorNode>();
			op->op = OperatorNode::OP_CALL;

			op->arguments.push_back(alloc_node<SelfNode>());

			IdentifierNode *funcname = alloc_node<IdentifierNode>();
			funcname->name = "get_node";

			op->arguments.push_back(funcname);

			ConstantNode *nodepath = alloc_node<ConstantNode>();
			nodepath->value = NodePath(StringName(path));
			op->arguments.push_back(nodepath);

			expr = op;

		} else if (tokenizer->get_token() == GDTokenizer::TK_CURSOR) {
			tokenizer->advance();
			continue; //no point in cursor in the middle of expression

		} else if (tokenizer->get_token() == GDTokenizer::TK_CONSTANT) {

			//constant defined by tokenizer
			ConstantNode *constant = alloc_node<ConstantNode>();
			constant->value = tokenizer->get_token_constant();
			tokenizer->advance();
			expr = constant;
		} else if (tokenizer->get_token() == GDTokenizer::TK_CONST_PI) {

			//constant defined by tokenizer
			ConstantNode *constant = alloc_node<ConstantNode>();
			constant->value = Math_PI;
			tokenizer->advance();
			expr = constant;
		} else if (tokenizer->get_token() == GDTokenizer::TK_CONST_INF) {

			//constant defined by tokenizer
			ConstantNode *constant = alloc_node<ConstantNode>();
			constant->value = Math_INF;
			tokenizer->advance();
			expr = constant;
		} else if (tokenizer->get_token() == GDTokenizer::TK_CONST_NAN) {

			//constant defined by tokenizer
			ConstantNode *constant = alloc_node<ConstantNode>();
			constant->value = Math_NAN;
			tokenizer->advance();
			expr = constant;
		} else if (tokenizer->get_token() == GDTokenizer::TK_PR_PRELOAD) {

			//constant defined by tokenizer
			tokenizer->advance();
			if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_OPEN) {
				_set_error("Expected '(' after 'preload'");
				return NULL;
			}
			tokenizer->advance();

			if (tokenizer->get_token() == GDTokenizer::TK_CURSOR) {
				completion_cursor = StringName();
				completion_node = p_parent;
				completion_type = COMPLETION_RESOURCE_PATH;
				completion_class = current_class;
				completion_function = current_function;
				completion_line = tokenizer->get_token_line();
				completion_block = current_block;
				completion_argument = 0;
				completion_found = true;
				tokenizer->advance();
			}

			String path;
			bool found_constant = false;
			bool valid = false;
			ConstantNode *cn;

			Node *subexpr = _parse_and_reduce_expression(p_parent, p_static);
			if (subexpr) {
				if (subexpr->type == Node::TYPE_CONSTANT) {
					cn = static_cast<ConstantNode *>(subexpr);
					found_constant = true;
				}
				if (subexpr->type == Node::TYPE_IDENTIFIER) {
					IdentifierNode *in = static_cast<IdentifierNode *>(subexpr);
					Vector<ClassNode::Constant> ce = current_class->constant_expressions;

					// Try to find the constant expression by the identifier
					for (int i = 0; i < ce.size(); ++i) {
						if (ce[i].identifier == in->name) {
							if (ce[i].expression->type == Node::TYPE_CONSTANT) {
								cn = static_cast<ConstantNode *>(ce[i].expression);
								found_constant = true;
							}
						}
					}
				}

				if (found_constant && cn->value.get_type() == Variant::STRING) {
					valid = true;
					path = (String)cn->value;
				}
			}

			if (!valid) {
				_set_error("expected string constant as 'preload' argument.");
				return NULL;
			}

			if (!path.is_abs_path() && base_path != "")
				path = base_path + "/" + path;
			path = path.replace("///", "//").simplify_path();
			if (path == self_path) {

				_set_error("Can't preload itself (use 'get_script()').");
				return NULL;
			}

			Ref<Resource> res;
			if (!validating) {

				//this can be too slow for just validating code
				if (for_completion && ScriptCodeCompletionCache::get_sigleton()) {
					res = ScriptCodeCompletionCache::get_sigleton()->get_cached_resource(path);
				} else {
					res = ResourceLoader::load(path);
				}
				if (!res.is_valid()) {
					_set_error("Can't preload resource at path: " + path);
					return NULL;
				}
			} else {

				if (!FileAccess::exists(path)) {
					_set_error("Can't preload resource at path: " + path);
					return NULL;
				}
			}

			if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {
				_set_error("Expected ')' after 'preload' path");
				return NULL;
			}
			tokenizer->advance();

			ConstantNode *constant = alloc_node<ConstantNode>();
			constant->value = res;

			expr = constant;
		} else if (tokenizer->get_token() == GDTokenizer::TK_PR_YIELD) {

			//constant defined by tokenizer

			tokenizer->advance();
			if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_OPEN) {
				_set_error("Expected '(' after 'yield'");
				return NULL;
			}

			tokenizer->advance();

			OperatorNode *yield = alloc_node<OperatorNode>();
			yield->op = OperatorNode::OP_YIELD;

			while (tokenizer->get_token() == GDTokenizer::TK_NEWLINE) {
				tokenizer->advance();
			}

			if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE) {
				expr = yield;
				tokenizer->advance();
			} else {

				parenthesis++;

				Node *object = _parse_and_reduce_expression(p_parent, p_static);
				if (!object)
					return NULL;
				yield->arguments.push_back(object);

				if (tokenizer->get_token() != GDTokenizer::TK_COMMA) {
					_set_error("Expected ',' after first argument of 'yield'");
					return NULL;
				}

				tokenizer->advance();

				if (tokenizer->get_token() == GDTokenizer::TK_CURSOR) {

					completion_cursor = StringName();
					completion_node = object;
					completion_type = COMPLETION_YIELD;
					completion_class = current_class;
					completion_function = current_function;
					completion_line = tokenizer->get_token_line();
					completion_argument = 0;
					completion_block = current_block;
					completion_found = true;
					tokenizer->advance();
				}

				Node *signal = _parse_and_reduce_expression(p_parent, p_static);
				if (!signal)
					return NULL;
				yield->arguments.push_back(signal);

				if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {
					_set_error("Expected ')' after second argument of 'yield'");
					return NULL;
				}

				parenthesis--;

				tokenizer->advance();

				expr = yield;
			}

		} else if (tokenizer->get_token() == GDTokenizer::TK_SELF) {

			if (p_static) {
				_set_error("'self'' not allowed in static function or constant expression");
				return NULL;
			}
			//constant defined by tokenizer
			SelfNode *self = alloc_node<SelfNode>();
			tokenizer->advance();
			expr = self;
		} else if (tokenizer->get_token() == GDTokenizer::TK_BUILT_IN_TYPE && tokenizer->get_token(1) == GDTokenizer::TK_PERIOD) {

			Variant::Type bi_type = tokenizer->get_token_type();
			tokenizer->advance(2);

			StringName identifier;

			if (_get_completable_identifier(COMPLETION_BUILT_IN_TYPE_CONSTANT, identifier)) {

				completion_built_in_constant = bi_type;
			}

			if (identifier == StringName()) {

				_set_error("Built-in type constant expected after '.'");
				return NULL;
			}
			if (!Variant::has_numeric_constant(bi_type, identifier)) {

				_set_error("Static constant  '" + identifier.operator String() + "' not present in built-in type " + Variant::get_type_name(bi_type) + ".");
				return NULL;
			}

			ConstantNode *cn = alloc_node<ConstantNode>();
			cn->value = Variant::get_numeric_constant_value(bi_type, identifier);
			expr = cn;

		} else if (tokenizer->get_token(1) == GDTokenizer::TK_PARENTHESIS_OPEN && tokenizer->is_token_literal()) {
			// We check with is_token_literal, as this allows us to use match/sync/etc. as a name
			//function or constructor

			OperatorNode *op = alloc_node<OperatorNode>();
			op->op = OperatorNode::OP_CALL;

			if (tokenizer->get_token() == GDTokenizer::TK_BUILT_IN_TYPE) {

				TypeNode *tn = alloc_node<TypeNode>();
				tn->vtype = tokenizer->get_token_type();
				op->arguments.push_back(tn);
				tokenizer->advance(2);
			} else if (tokenizer->get_token() == GDTokenizer::TK_BUILT_IN_FUNC) {

				BuiltInFunctionNode *bn = alloc_node<BuiltInFunctionNode>();
				bn->function = tokenizer->get_token_built_in_func();
				op->arguments.push_back(bn);
				tokenizer->advance(2);
			} else {

				SelfNode *self = alloc_node<SelfNode>();
				op->arguments.push_back(self);

				StringName identifier;
				if (_get_completable_identifier(COMPLETION_FUNCTION, identifier)) {
				}

				IdentifierNode *id = alloc_node<IdentifierNode>();
				id->name = identifier;
				op->arguments.push_back(id);
				tokenizer->advance(1);
			}

			if (tokenizer->get_token() == GDTokenizer::TK_CURSOR) {
				_make_completable_call(0);
				completion_node = op;
			}
			if (!_parse_arguments(op, op->arguments, p_static, true))
				return NULL;

			expr = op;

		} else if (tokenizer->is_token_literal(0, true)) {
			// We check with is_token_literal, as this allows us to use match/sync/etc. as a name
			//identifier (reference)

			const ClassNode *cln = current_class;
			bool bfn = false;
			StringName identifier;
			if (_get_completable_identifier(COMPLETION_IDENTIFIER, identifier)) {
			}

			if (p_parsing_constant) {
				for (int i = 0; i < cln->constant_expressions.size(); ++i) {

					if (cln->constant_expressions[i].identifier == identifier) {

						expr = cln->constant_expressions[i].expression;
						bfn = true;
						break;
					}
				}

				if (GDScriptLanguage::get_singleton()->get_global_map().has(identifier)) {
					//check from constants
					ConstantNode *constant = alloc_node<ConstantNode>();
					constant->value = GDScriptLanguage::get_singleton()->get_global_array()[GDScriptLanguage::get_singleton()->get_global_map()[identifier]];
					expr = constant;
					bfn = true;
				}
			}

			if (!bfn) {
				IdentifierNode *id = alloc_node<IdentifierNode>();
				id->name = identifier;
				expr = id;
			}

		} else if (tokenizer->get_token() == GDTokenizer::TK_OP_ADD || tokenizer->get_token() == GDTokenizer::TK_OP_SUB || tokenizer->get_token() == GDTokenizer::TK_OP_NOT || tokenizer->get_token() == GDTokenizer::TK_OP_BIT_INVERT) {

			//single prefix operators like !expr +expr -expr ++expr --expr
			alloc_node<OperatorNode>();
			Expression e;
			e.is_op = true;

			switch (tokenizer->get_token()) {
				case GDTokenizer::TK_OP_ADD: e.op = OperatorNode::OP_POS; break;
				case GDTokenizer::TK_OP_SUB: e.op = OperatorNode::OP_NEG; break;
				case GDTokenizer::TK_OP_NOT: e.op = OperatorNode::OP_NOT; break;
				case GDTokenizer::TK_OP_BIT_INVERT: e.op = OperatorNode::OP_BIT_INVERT; break;
				default: {}
			}

			tokenizer->advance();

			if (e.op != OperatorNode::OP_NOT && tokenizer->get_token() == GDTokenizer::TK_OP_NOT) {
				_set_error("Misplaced 'not'.");
				return NULL;
			}

			expression.push_back(e);
			continue; //only exception, must continue...

			/*
			Node *subexpr=_parse_expression(op,p_static);
			if (!subexpr)
				return NULL;
			op->arguments.push_back(subexpr);
			expr=op;*/

		} else if (tokenizer->get_token() == GDTokenizer::TK_BRACKET_OPEN) {
			// array
			tokenizer->advance();

			ArrayNode *arr = alloc_node<ArrayNode>();
			bool expecting_comma = false;

			while (true) {

				if (tokenizer->get_token() == GDTokenizer::TK_EOF) {

					_set_error("Unterminated array");
					return NULL;

				} else if (tokenizer->get_token() == GDTokenizer::TK_BRACKET_CLOSE) {
					tokenizer->advance();
					break;
				} else if (tokenizer->get_token() == GDTokenizer::TK_NEWLINE) {

					tokenizer->advance(); //ignore newline
				} else if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {
					if (!expecting_comma) {
						_set_error("expression or ']' expected");
						return NULL;
					}

					expecting_comma = false;
					tokenizer->advance(); //ignore newline
				} else {
					//parse expression
					if (expecting_comma) {
						_set_error("',' or ']' expected");
						return NULL;
					}
					Node *n = _parse_expression(arr, p_static, p_allow_assign, p_parsing_constant);
					if (!n)
						return NULL;
					arr->elements.push_back(n);
					expecting_comma = true;
				}
			}

			expr = arr;
		} else if (tokenizer->get_token() == GDTokenizer::TK_CURLY_BRACKET_OPEN) {
			// array
			tokenizer->advance();

			DictionaryNode *dict = alloc_node<DictionaryNode>();

			enum DictExpect {

				DICT_EXPECT_KEY,
				DICT_EXPECT_COLON,
				DICT_EXPECT_VALUE,
				DICT_EXPECT_COMMA

			};

			Node *key = NULL;
			Set<Variant> keys;

			DictExpect expecting = DICT_EXPECT_KEY;

			while (true) {

				if (tokenizer->get_token() == GDTokenizer::TK_EOF) {

					_set_error("Unterminated dictionary");
					return NULL;

				} else if (tokenizer->get_token() == GDTokenizer::TK_CURLY_BRACKET_CLOSE) {

					if (expecting == DICT_EXPECT_COLON) {
						_set_error("':' expected");
						return NULL;
					}
					if (expecting == DICT_EXPECT_VALUE) {
						_set_error("value expected");
						return NULL;
					}
					tokenizer->advance();
					break;
				} else if (tokenizer->get_token() == GDTokenizer::TK_NEWLINE) {

					tokenizer->advance(); //ignore newline
				} else if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {

					if (expecting == DICT_EXPECT_KEY) {
						_set_error("key or '}' expected");
						return NULL;
					}
					if (expecting == DICT_EXPECT_VALUE) {
						_set_error("value expected");
						return NULL;
					}
					if (expecting == DICT_EXPECT_COLON) {
						_set_error("':' expected");
						return NULL;
					}

					expecting = DICT_EXPECT_KEY;
					tokenizer->advance(); //ignore newline

				} else if (tokenizer->get_token() == GDTokenizer::TK_COLON) {

					if (expecting == DICT_EXPECT_KEY) {
						_set_error("key or '}' expected");
						return NULL;
					}
					if (expecting == DICT_EXPECT_VALUE) {
						_set_error("value expected");
						return NULL;
					}
					if (expecting == DICT_EXPECT_COMMA) {
						_set_error("',' or '}' expected");
						return NULL;
					}

					expecting = DICT_EXPECT_VALUE;
					tokenizer->advance(); //ignore newline
				} else {

					if (expecting == DICT_EXPECT_COMMA) {
						_set_error("',' or '}' expected");
						return NULL;
					}
					if (expecting == DICT_EXPECT_COLON) {
						_set_error("':' expected");
						return NULL;
					}

					if (expecting == DICT_EXPECT_KEY) {

						if (tokenizer->is_token_literal() && tokenizer->get_token(1) == GDTokenizer::TK_OP_ASSIGN) {
							// We check with is_token_literal, as this allows us to use match/sync/etc. as a name
							//lua style identifier, easier to write
							ConstantNode *cn = alloc_node<ConstantNode>();
							cn->value = tokenizer->get_token_literal();
							key = cn;
							tokenizer->advance(2);
							expecting = DICT_EXPECT_VALUE;
						} else {
							//python/js style more flexible
							key = _parse_expression(dict, p_static, p_allow_assign, p_parsing_constant);
							if (!key)
								return NULL;
							expecting = DICT_EXPECT_COLON;
						}
					}

					if (expecting == DICT_EXPECT_VALUE) {
						Node *value = _parse_expression(dict, p_static, p_allow_assign, p_parsing_constant);
						if (!value)
							return NULL;
						expecting = DICT_EXPECT_COMMA;

						if (key->type == GDParser::Node::TYPE_CONSTANT) {
							Variant const &keyName = static_cast<const GDParser::ConstantNode *>(key)->value;

							if (keys.has(keyName)) {
								_set_error("Duplicate key found in Dictionary literal");
								return NULL;
							}
							keys.insert(keyName);
						}

						DictionaryNode::Pair pair;
						pair.key = key;
						pair.value = value;
						dict->elements.push_back(pair);
						key = NULL;
					}
				}
			}

			expr = dict;

		} else if (tokenizer->get_token() == GDTokenizer::TK_PERIOD && (tokenizer->is_token_literal(1) || tokenizer->get_token(1) == GDTokenizer::TK_CURSOR) && tokenizer->get_token(2) == GDTokenizer::TK_PARENTHESIS_OPEN) {
			// We check with is_token_literal, as this allows us to use match/sync/etc. as a name
			// parent call

			tokenizer->advance(); //goto identifier
			OperatorNode *op = alloc_node<OperatorNode>();
			op->op = OperatorNode::OP_PARENT_CALL;

			/*SelfNode *self = alloc_node<SelfNode>();
			op->arguments.push_back(self);
			forbidden for now */
			StringName identifier;
			if (_get_completable_identifier(COMPLETION_PARENT_FUNCTION, identifier)) {
				//indexing stuff
			}

			IdentifierNode *id = alloc_node<IdentifierNode>();
			id->name = identifier;
			op->arguments.push_back(id);

			tokenizer->advance(1);
			if (!_parse_arguments(op, op->arguments, p_static))
				return NULL;

			expr = op;

		} else {

			//find list [ or find dictionary {

			//print_line("found bug?");

			_set_error("Error parsing expression, misplaced: " + String(tokenizer->get_token_name(tokenizer->get_token())));
			return NULL; //nothing
		}

		if (!expr) {
			ERR_EXPLAIN("GDParser bug, couldn't figure out what expression is..");
			ERR_FAIL_COND_V(!expr, NULL);
		}

		/******************/
		/* Parse Indexing */
		/******************/

		while (true) {

			//expressions can be indexed any number of times

			if (tokenizer->get_token() == GDTokenizer::TK_PERIOD) {

				//indexing using "."

				if (tokenizer->get_token(1) != GDTokenizer::TK_CURSOR && !tokenizer->is_token_literal(1)) {
					// We check with is_token_literal, as this allows us to use match/sync/etc. as a name
					_set_error("Expected identifier as member");
					return NULL;
				} else if (tokenizer->get_token(2) == GDTokenizer::TK_PARENTHESIS_OPEN) {
					//call!!
					OperatorNode *op = alloc_node<OperatorNode>();
					op->op = OperatorNode::OP_CALL;

					tokenizer->advance();

					IdentifierNode *id = alloc_node<IdentifierNode>();
					if (tokenizer->get_token() == GDTokenizer::TK_BUILT_IN_FUNC) {
						//small hack so built in funcs don't obfuscate methods

						id->name = GDFunctions::get_func_name(tokenizer->get_token_built_in_func());
						tokenizer->advance();

					} else {
						StringName identifier;
						if (_get_completable_identifier(COMPLETION_METHOD, identifier)) {
							completion_node = op;
							//indexing stuff
						}

						id->name = identifier;
					}

					op->arguments.push_back(expr); // call what
					op->arguments.push_back(id); // call func
					//get arguments
					tokenizer->advance(1);
					if (tokenizer->get_token() == GDTokenizer::TK_CURSOR) {
						_make_completable_call(0);
						completion_node = op;
					}
					if (!_parse_arguments(op, op->arguments, p_static, true))
						return NULL;
					expr = op;

				} else {
					//simple indexing!

					OperatorNode *op = alloc_node<OperatorNode>();
					op->op = OperatorNode::OP_INDEX_NAMED;
					tokenizer->advance();

					StringName identifier;
					if (_get_completable_identifier(COMPLETION_INDEX, identifier)) {

						if (identifier == StringName()) {
							identifier = "@temp"; //so it parses allright
						}
						completion_node = op;

						//indexing stuff
					}

					IdentifierNode *id = alloc_node<IdentifierNode>();
					id->name = identifier;

					op->arguments.push_back(expr);
					op->arguments.push_back(id);

					expr = op;
				}

			} else if (tokenizer->get_token() == GDTokenizer::TK_BRACKET_OPEN) {
				//indexing using "[]"
				OperatorNode *op = alloc_node<OperatorNode>();
				op->op = OperatorNode::OP_INDEX;

				tokenizer->advance(1);

				Node *subexpr = _parse_expression(op, p_static, p_allow_assign, p_parsing_constant);
				if (!subexpr) {
					return NULL;
				}

				if (tokenizer->get_token() != GDTokenizer::TK_BRACKET_CLOSE) {
					_set_error("Expected ']'");
					return NULL;
				}

				op->arguments.push_back(expr);
				op->arguments.push_back(subexpr);
				tokenizer->advance(1);
				expr = op;

			} else
				break;
		}

		/******************/
		/* Parse Operator */
		/******************/

		if (parenthesis > 0) {
			//remove empty space (only allowed if inside parenthesis
			while (tokenizer->get_token() == GDTokenizer::TK_NEWLINE) {
				tokenizer->advance();
			}
		}

		Expression e;
		e.is_op = false;
		e.node = expr;
		expression.push_back(e);

		// determine which operator is next

		OperatorNode::Operator op;
		bool valid = true;

//assign, if allowed is only allowed on the first operator
#define _VALIDATE_ASSIGN                  \
	if (!p_allow_assign) {                \
		_set_error("Unexpected assign."); \
		return NULL;                      \
	}                                     \
	p_allow_assign = false;

		switch (tokenizer->get_token()) { //see operator

			case GDTokenizer::TK_OP_IN: op = OperatorNode::OP_IN; break;
			case GDTokenizer::TK_OP_EQUAL: op = OperatorNode::OP_EQUAL; break;
			case GDTokenizer::TK_OP_NOT_EQUAL: op = OperatorNode::OP_NOT_EQUAL; break;
			case GDTokenizer::TK_OP_LESS: op = OperatorNode::OP_LESS; break;
			case GDTokenizer::TK_OP_LESS_EQUAL: op = OperatorNode::OP_LESS_EQUAL; break;
			case GDTokenizer::TK_OP_GREATER: op = OperatorNode::OP_GREATER; break;
			case GDTokenizer::TK_OP_GREATER_EQUAL: op = OperatorNode::OP_GREATER_EQUAL; break;
			case GDTokenizer::TK_OP_AND: op = OperatorNode::OP_AND; break;
			case GDTokenizer::TK_OP_OR: op = OperatorNode::OP_OR; break;
			case GDTokenizer::TK_OP_ADD: op = OperatorNode::OP_ADD; break;
			case GDTokenizer::TK_OP_SUB: op = OperatorNode::OP_SUB; break;
			case GDTokenizer::TK_OP_MUL: op = OperatorNode::OP_MUL; break;
			case GDTokenizer::TK_OP_DIV: op = OperatorNode::OP_DIV; break;
			case GDTokenizer::TK_OP_MOD:
				op = OperatorNode::OP_MOD;
				break;
			//case GDTokenizer::TK_OP_NEG: op=OperatorNode::OP_NEG ; break;
			case GDTokenizer::TK_OP_SHIFT_LEFT: op = OperatorNode::OP_SHIFT_LEFT; break;
			case GDTokenizer::TK_OP_SHIFT_RIGHT: op = OperatorNode::OP_SHIFT_RIGHT; break;
			case GDTokenizer::TK_OP_ASSIGN: {
				_VALIDATE_ASSIGN op = OperatorNode::OP_ASSIGN;

				if (tokenizer->get_token(1) == GDTokenizer::TK_CURSOR) {
					//code complete assignment
					completion_type = COMPLETION_ASSIGN;
					completion_node = expr;
					completion_class = current_class;
					completion_function = current_function;
					completion_line = tokenizer->get_token_line();
					completion_block = current_block;
					completion_found = true;
					tokenizer->advance();
				}

			} break;
			case GDTokenizer::TK_OP_ASSIGN_ADD: _VALIDATE_ASSIGN op = OperatorNode::OP_ASSIGN_ADD; break;
			case GDTokenizer::TK_OP_ASSIGN_SUB: _VALIDATE_ASSIGN op = OperatorNode::OP_ASSIGN_SUB; break;
			case GDTokenizer::TK_OP_ASSIGN_MUL: _VALIDATE_ASSIGN op = OperatorNode::OP_ASSIGN_MUL; break;
			case GDTokenizer::TK_OP_ASSIGN_DIV: _VALIDATE_ASSIGN op = OperatorNode::OP_ASSIGN_DIV; break;
			case GDTokenizer::TK_OP_ASSIGN_MOD: _VALIDATE_ASSIGN op = OperatorNode::OP_ASSIGN_MOD; break;
			case GDTokenizer::TK_OP_ASSIGN_SHIFT_LEFT: _VALIDATE_ASSIGN op = OperatorNode::OP_ASSIGN_SHIFT_LEFT; break;
			case GDTokenizer::TK_OP_ASSIGN_SHIFT_RIGHT: _VALIDATE_ASSIGN op = OperatorNode::OP_ASSIGN_SHIFT_RIGHT; break;
			case GDTokenizer::TK_OP_ASSIGN_BIT_AND: _VALIDATE_ASSIGN op = OperatorNode::OP_ASSIGN_BIT_AND; break;
			case GDTokenizer::TK_OP_ASSIGN_BIT_OR: _VALIDATE_ASSIGN op = OperatorNode::OP_ASSIGN_BIT_OR; break;
			case GDTokenizer::TK_OP_ASSIGN_BIT_XOR: _VALIDATE_ASSIGN op = OperatorNode::OP_ASSIGN_BIT_XOR; break;
			case GDTokenizer::TK_OP_BIT_AND: op = OperatorNode::OP_BIT_AND; break;
			case GDTokenizer::TK_OP_BIT_OR: op = OperatorNode::OP_BIT_OR; break;
			case GDTokenizer::TK_OP_BIT_XOR: op = OperatorNode::OP_BIT_XOR; break;
			case GDTokenizer::TK_PR_IS: op = OperatorNode::OP_IS; break;
			case GDTokenizer::TK_CF_IF: op = OperatorNode::OP_TERNARY_IF; break;
			case GDTokenizer::TK_CF_ELSE: op = OperatorNode::OP_TERNARY_ELSE; break;
			default: valid = false; break;
		}

		if (valid) {
			e.is_op = true;
			e.op = op;
			expression.push_back(e);
			tokenizer->advance();
		} else {
			break;
		}
	}

	/* Reduce the set set of expressions and place them in an operator tree, respecting precedence */

	while (expression.size() > 1) {

		int next_op = -1;
		int min_priority = 0xFFFFF;
		bool is_unary = false;
		bool is_ternary = false;

		for (int i = 0; i < expression.size(); i++) {

			if (!expression[i].is_op) {

				continue;
			}

			int priority;

			bool unary = false;
			bool ternary = false;
			bool error = false;

			switch (expression[i].op) {

				case OperatorNode::OP_IS:
					priority = -1;
					break; //before anything

				case OperatorNode::OP_BIT_INVERT:
					priority = 0;
					unary = true;
					break;
				case OperatorNode::OP_NEG:
					priority = 1;
					unary = true;
					break;
				case OperatorNode::OP_POS:
					priority = 1;
					unary = true;
					break;

				case OperatorNode::OP_MUL: priority = 2; break;
				case OperatorNode::OP_DIV: priority = 2; break;
				case OperatorNode::OP_MOD: priority = 2; break;

				case OperatorNode::OP_ADD: priority = 3; break;
				case OperatorNode::OP_SUB: priority = 3; break;

				case OperatorNode::OP_SHIFT_LEFT: priority = 4; break;
				case OperatorNode::OP_SHIFT_RIGHT: priority = 4; break;

				case OperatorNode::OP_BIT_AND: priority = 5; break;
				case OperatorNode::OP_BIT_XOR: priority = 6; break;
				case OperatorNode::OP_BIT_OR: priority = 7; break;

				case OperatorNode::OP_LESS: priority = 8; break;
				case OperatorNode::OP_LESS_EQUAL: priority = 8; break;
				case OperatorNode::OP_GREATER: priority = 8; break;
				case OperatorNode::OP_GREATER_EQUAL: priority = 8; break;

				case OperatorNode::OP_EQUAL: priority = 8; break;
				case OperatorNode::OP_NOT_EQUAL: priority = 8; break;

				case OperatorNode::OP_IN: priority = 10; break;

				case OperatorNode::OP_NOT:
					priority = 11;
					unary = true;
					break;
				case OperatorNode::OP_AND: priority = 12; break;
				case OperatorNode::OP_OR: priority = 13; break;

				case OperatorNode::OP_TERNARY_IF:
					priority = 14;
					ternary = true;
					break;
				case OperatorNode::OP_TERNARY_ELSE:
					priority = 14;
					error = true;
					break; // Errors out when found without IF (since IF would consume it)

				case OperatorNode::OP_ASSIGN: priority = 15; break;
				case OperatorNode::OP_ASSIGN_ADD: priority = 15; break;
				case OperatorNode::OP_ASSIGN_SUB: priority = 15; break;
				case OperatorNode::OP_ASSIGN_MUL: priority = 15; break;
				case OperatorNode::OP_ASSIGN_DIV: priority = 15; break;
				case OperatorNode::OP_ASSIGN_MOD: priority = 15; break;
				case OperatorNode::OP_ASSIGN_SHIFT_LEFT: priority = 15; break;
				case OperatorNode::OP_ASSIGN_SHIFT_RIGHT: priority = 15; break;
				case OperatorNode::OP_ASSIGN_BIT_AND: priority = 15; break;
				case OperatorNode::OP_ASSIGN_BIT_OR: priority = 15; break;
				case OperatorNode::OP_ASSIGN_BIT_XOR: priority = 15; break;

				default: {
					_set_error("GDParser bug, invalid operator in expression: " + itos(expression[i].op));
					return NULL;
				}
			}

			if (priority < min_priority) {
				if (error) {
					_set_error("Unexpected operator");
					return NULL;
				}
				// < is used for left to right (default)
				// <= is used for right to left
				next_op = i;
				min_priority = priority;
				is_unary = unary;
				is_ternary = ternary;
			}
		}

		if (next_op == -1) {

			_set_error("Yet another parser bug....");
			ERR_FAIL_COND_V(next_op == -1, NULL);
		}

		// OK! create operator..
		if (is_unary) {

			int expr_pos = next_op;
			while (expression[expr_pos].is_op) {

				expr_pos++;
				if (expr_pos == expression.size()) {
					//can happen..
					_set_error("Unexpected end of expression..");
					return NULL;
				}
			}

			//consecutively do unary opeators
			for (int i = expr_pos - 1; i >= next_op; i--) {

				OperatorNode *op = alloc_node<OperatorNode>();
				op->op = expression[i].op;
				op->arguments.push_back(expression[i + 1].node);
				op->line = op_line; //line might have been changed from a \n
				expression[i].is_op = false;
				expression[i].node = op;
				expression.remove(i + 1);
			}

		} else if (is_ternary) {
			if (next_op < 1 || next_op >= (expression.size() - 1)) {
				_set_error("Parser bug..");
				ERR_FAIL_V(NULL);
			}

			if (next_op >= (expression.size() - 2) || expression[next_op + 2].op != OperatorNode::OP_TERNARY_ELSE) {
				_set_error("Expected else after ternary if.");
				ERR_FAIL_V(NULL);
			}
			if (next_op >= (expression.size() - 3)) {
				_set_error("Expected value after ternary else.");
				ERR_FAIL_V(NULL);
			}

			OperatorNode *op = alloc_node<OperatorNode>();
			op->op = expression[next_op].op;
			op->line = op_line; //line might have been changed from a \n

			if (expression[next_op - 1].is_op) {

				_set_error("Parser bug..");
				ERR_FAIL_V(NULL);
			}

			if (expression[next_op + 1].is_op) {
				// this is not invalid and can really appear
				// but it becomes invalid anyway because no binary op
				// can be followed by a unary op in a valid combination,
				// due to how precedence works, unaries will always disappear first

				_set_error("Unexpected two consecutive operators after ternary if.");
				return NULL;
			}

			if (expression[next_op + 3].is_op) {
				// this is not invalid and can really appear
				// but it becomes invalid anyway because no binary op
				// can be followed by a unary op in a valid combination,
				// due to how precedence works, unaries will always disappear first

				_set_error("Unexpected two consecutive operators after ternary else.");
				return NULL;
			}

			op->arguments.push_back(expression[next_op + 1].node); //next expression goes as first
			op->arguments.push_back(expression[next_op - 1].node); //left expression goes as when-true
			op->arguments.push_back(expression[next_op + 3].node); //expression after next goes as when-false

			//replace all 3 nodes by this operator and make it an expression
			expression[next_op - 1].node = op;
			expression.remove(next_op);
			expression.remove(next_op);
			expression.remove(next_op);
			expression.remove(next_op);
		} else {

			if (next_op < 1 || next_op >= (expression.size() - 1)) {
				_set_error("Parser bug..");
				ERR_FAIL_V(NULL);
			}

			OperatorNode *op = alloc_node<OperatorNode>();
			op->op = expression[next_op].op;
			op->line = op_line; //line might have been changed from a \n

			if (expression[next_op - 1].is_op) {

				_set_error("Parser bug..");
				ERR_FAIL_V(NULL);
			}

			if (expression[next_op + 1].is_op) {
				// this is not invalid and can really appear
				// but it becomes invalid anyway because no binary op
				// can be followed by a unary op in a valid combination,
				// due to how precedence works, unaries will always disappear first

				_set_error("Unexpected two consecutive operators.");
				return NULL;
			}

			op->arguments.push_back(expression[next_op - 1].node); //expression goes as left
			op->arguments.push_back(expression[next_op + 1].node); //next expression goes as right

			//replace all 3 nodes by this operator and make it an expression
			expression[next_op - 1].node = op;
			expression.remove(next_op);
			expression.remove(next_op);
		}
	}

	return expression[0].node;
}

GDParser::Node *GDParser::_reduce_expression(Node *p_node, bool p_to_const) {

	switch (p_node->type) {

		case Node::TYPE_BUILT_IN_FUNCTION: {
			//many may probably be optimizable
			return p_node;
		} break;
		case Node::TYPE_ARRAY: {

			ArrayNode *an = static_cast<ArrayNode *>(p_node);
			bool all_constants = true;

			for (int i = 0; i < an->elements.size(); i++) {

				an->elements[i] = _reduce_expression(an->elements[i], p_to_const);
				if (an->elements[i]->type != Node::TYPE_CONSTANT)
					all_constants = false;
			}

			if (all_constants && p_to_const) {
				//reduce constant array expression

				ConstantNode *cn = alloc_node<ConstantNode>();
				Array arr;
				//print_line("mk array "+itos(!p_to_const));
				arr.resize(an->elements.size());
				for (int i = 0; i < an->elements.size(); i++) {
					ConstantNode *acn = static_cast<ConstantNode *>(an->elements[i]);
					arr[i] = acn->value;
				}
				cn->value = arr;
				return cn;
			}

			return an;

		} break;
		case Node::TYPE_DICTIONARY: {

			DictionaryNode *dn = static_cast<DictionaryNode *>(p_node);
			bool all_constants = true;

			for (int i = 0; i < dn->elements.size(); i++) {

				dn->elements[i].key = _reduce_expression(dn->elements[i].key, p_to_const);
				if (dn->elements[i].key->type != Node::TYPE_CONSTANT)
					all_constants = false;
				dn->elements[i].value = _reduce_expression(dn->elements[i].value, p_to_const);
				if (dn->elements[i].value->type != Node::TYPE_CONSTANT)
					all_constants = false;
			}

			if (all_constants && p_to_const) {
				//reduce constant array expression

				ConstantNode *cn = alloc_node<ConstantNode>();
				Dictionary dict;
				for (int i = 0; i < dn->elements.size(); i++) {
					ConstantNode *key_c = static_cast<ConstantNode *>(dn->elements[i].key);
					ConstantNode *value_c = static_cast<ConstantNode *>(dn->elements[i].value);

					dict[key_c->value] = value_c->value;
				}
				cn->value = dict;
				return cn;
			}

			return dn;

		} break;
		case Node::TYPE_OPERATOR: {

			OperatorNode *op = static_cast<OperatorNode *>(p_node);

			bool all_constants = true;
			int last_not_constant = -1;

			for (int i = 0; i < op->arguments.size(); i++) {

				op->arguments[i] = _reduce_expression(op->arguments[i], p_to_const);
				if (op->arguments[i]->type != Node::TYPE_CONSTANT) {
					all_constants = false;
					last_not_constant = i;
				}
			}

			if (op->op == OperatorNode::OP_IS) {
				//nothing much
				return op;
			}
			if (op->op == OperatorNode::OP_PARENT_CALL) {
				//nothing much
				return op;

			} else if (op->op == OperatorNode::OP_CALL) {
				//can reduce base type constructors
				if ((op->arguments[0]->type == Node::TYPE_TYPE || (op->arguments[0]->type == Node::TYPE_BUILT_IN_FUNCTION && GDFunctions::is_deterministic(static_cast<BuiltInFunctionNode *>(op->arguments[0])->function))) && last_not_constant == 0) {

					//native type constructor or intrinsic function
					const Variant **vptr = NULL;
					Vector<Variant *> ptrs;
					if (op->arguments.size() > 1) {

						ptrs.resize(op->arguments.size() - 1);
						for (int i = 0; i < ptrs.size(); i++) {

							ConstantNode *cn = static_cast<ConstantNode *>(op->arguments[i + 1]);
							ptrs[i] = &cn->value;
						}

						vptr = (const Variant **)&ptrs[0];
					}

					Variant::CallError ce;
					Variant v;

					if (op->arguments[0]->type == Node::TYPE_TYPE) {
						TypeNode *tn = static_cast<TypeNode *>(op->arguments[0]);
						v = Variant::construct(tn->vtype, vptr, ptrs.size(), ce);

					} else {
						GDFunctions::Function func = static_cast<BuiltInFunctionNode *>(op->arguments[0])->function;
						GDFunctions::call(func, vptr, ptrs.size(), v, ce);
					}

					if (ce.error != Variant::CallError::CALL_OK) {

						String errwhere;
						if (op->arguments[0]->type == Node::TYPE_TYPE) {
							TypeNode *tn = static_cast<TypeNode *>(op->arguments[0]);
							errwhere = "'" + Variant::get_type_name(tn->vtype) + "'' constructor";

						} else {
							GDFunctions::Function func = static_cast<BuiltInFunctionNode *>(op->arguments[0])->function;
							errwhere = String("'") + GDFunctions::get_func_name(func) + "'' intrinsic function";
						}

						switch (ce.error) {

							case Variant::CallError::CALL_ERROR_INVALID_ARGUMENT: {

								_set_error("Invalid argument (#" + itos(ce.argument + 1) + ") for " + errwhere + ".");

							} break;
							case Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS: {

								_set_error("Too many arguments for " + errwhere + ".");
							} break;
							case Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS: {

								_set_error("Too few arguments for " + errwhere + ".");
							} break;
							default: {
								_set_error("Invalid arguments for " + errwhere + ".");

							} break;
						}

						error_line = op->line;

						return p_node;
					}

					ConstantNode *cn = alloc_node<ConstantNode>();
					cn->value = v;
					return cn;

				} else if (op->arguments[0]->type == Node::TYPE_BUILT_IN_FUNCTION && last_not_constant == 0) {
				}

				return op; //don't reduce yet

			} else if (op->op == OperatorNode::OP_YIELD) {
				return op;

			} else if (op->op == OperatorNode::OP_INDEX) {
				//can reduce indices into constant arrays or dictionaries

				if (all_constants) {

					ConstantNode *ca = static_cast<ConstantNode *>(op->arguments[0]);
					ConstantNode *cb = static_cast<ConstantNode *>(op->arguments[1]);

					bool valid;

					Variant v = ca->value.get(cb->value, &valid);
					if (!valid) {
						_set_error("invalid index in constant expression");
						error_line = op->line;
						return op;
					}

					ConstantNode *cn = alloc_node<ConstantNode>();
					cn->value = v;
					return cn;

				} /*else if (op->arguments[0]->type==Node::TYPE_CONSTANT && op->arguments[1]->type==Node::TYPE_IDENTIFIER) {

					ConstantNode *ca = static_cast<ConstantNode*>(op->arguments[0]);
					IdentifierNode *ib = static_cast<IdentifierNode*>(op->arguments[1]);

					bool valid;
					Variant v = ca->value.get_named(ib->name,&valid);
					if (!valid) {
						_set_error("invalid index '"+String(ib->name)+"' in constant expression");
						return op;
					}

					ConstantNode *cn = alloc_node<ConstantNode>();
					cn->value=v;
					return cn;
				}*/

				return op;

			} else if (op->op == OperatorNode::OP_INDEX_NAMED) {

				if (op->arguments[0]->type == Node::TYPE_CONSTANT && op->arguments[1]->type == Node::TYPE_IDENTIFIER) {

					ConstantNode *ca = static_cast<ConstantNode *>(op->arguments[0]);
					IdentifierNode *ib = static_cast<IdentifierNode *>(op->arguments[1]);

					bool valid;
					Variant v = ca->value.get_named(ib->name, &valid);
					if (!valid) {
						_set_error("invalid index '" + String(ib->name) + "' in constant expression");
						error_line = op->line;
						return op;
					}

					ConstantNode *cn = alloc_node<ConstantNode>();
					cn->value = v;
					return cn;
				}

				return op;
			}

			//validate assignment (don't assign to cosntant expression
			switch (op->op) {

				case OperatorNode::OP_ASSIGN:
				case OperatorNode::OP_ASSIGN_ADD:
				case OperatorNode::OP_ASSIGN_SUB:
				case OperatorNode::OP_ASSIGN_MUL:
				case OperatorNode::OP_ASSIGN_DIV:
				case OperatorNode::OP_ASSIGN_MOD:
				case OperatorNode::OP_ASSIGN_SHIFT_LEFT:
				case OperatorNode::OP_ASSIGN_SHIFT_RIGHT:
				case OperatorNode::OP_ASSIGN_BIT_AND:
				case OperatorNode::OP_ASSIGN_BIT_OR:
				case OperatorNode::OP_ASSIGN_BIT_XOR: {

					if (op->arguments[0]->type == Node::TYPE_CONSTANT) {
						_set_error("Can't assign to constant", tokenizer->get_token_line() - 1);
						error_line = op->line;
						return op;
					}

					if (op->arguments[0]->type == Node::TYPE_OPERATOR) {
						OperatorNode *on = static_cast<OperatorNode *>(op->arguments[0]);
						if (on->op != OperatorNode::OP_INDEX && on->op != OperatorNode::OP_INDEX_NAMED) {
							_set_error("Can't assign to an expression", tokenizer->get_token_line() - 1);
							error_line = op->line;
							return op;
						}
					}

				} break;
				default: { break; }
			}
			//now se if all are constants
			if (!all_constants)
				return op; //nothing to reduce from here on
#define _REDUCE_UNARY(m_vop)                                                                               \
	bool valid = false;                                                                                    \
	Variant res;                                                                                           \
	Variant::evaluate(m_vop, static_cast<ConstantNode *>(op->arguments[0])->value, Variant(), res, valid); \
	if (!valid) {                                                                                          \
		_set_error("Invalid operand for unary operator");                                                  \
		error_line = op->line;                                                                             \
		return p_node;                                                                                     \
	}                                                                                                      \
	ConstantNode *cn = alloc_node<ConstantNode>();                                                         \
	cn->value = res;                                                                                       \
	return cn;

#define _REDUCE_BINARY(m_vop)                                                                                                                         \
	bool valid = false;                                                                                                                               \
	Variant res;                                                                                                                                      \
	Variant::evaluate(m_vop, static_cast<ConstantNode *>(op->arguments[0])->value, static_cast<ConstantNode *>(op->arguments[1])->value, res, valid); \
	if (!valid) {                                                                                                                                     \
		_set_error("Invalid operands for operator");                                                                                                  \
		error_line = op->line;                                                                                                                        \
		return p_node;                                                                                                                                \
	}                                                                                                                                                 \
	ConstantNode *cn = alloc_node<ConstantNode>();                                                                                                    \
	cn->value = res;                                                                                                                                  \
	return cn;

			switch (op->op) {

				//unary operators
				case OperatorNode::OP_NEG: {
					_REDUCE_UNARY(Variant::OP_NEGATE);
				} break;
				case OperatorNode::OP_POS: {
					_REDUCE_UNARY(Variant::OP_POSITIVE);
				} break;
				case OperatorNode::OP_NOT: {
					_REDUCE_UNARY(Variant::OP_NOT);
				} break;
				case OperatorNode::OP_BIT_INVERT: {
					_REDUCE_UNARY(Variant::OP_BIT_NEGATE);
				} break;
				//binary operators (in precedence order)
				case OperatorNode::OP_IN: {
					_REDUCE_BINARY(Variant::OP_IN);
				} break;
				case OperatorNode::OP_EQUAL: {
					_REDUCE_BINARY(Variant::OP_EQUAL);
				} break;
				case OperatorNode::OP_NOT_EQUAL: {
					_REDUCE_BINARY(Variant::OP_NOT_EQUAL);
				} break;
				case OperatorNode::OP_LESS: {
					_REDUCE_BINARY(Variant::OP_LESS);
				} break;
				case OperatorNode::OP_LESS_EQUAL: {
					_REDUCE_BINARY(Variant::OP_LESS_EQUAL);
				} break;
				case OperatorNode::OP_GREATER: {
					_REDUCE_BINARY(Variant::OP_GREATER);
				} break;
				case OperatorNode::OP_GREATER_EQUAL: {
					_REDUCE_BINARY(Variant::OP_GREATER_EQUAL);
				} break;
				case OperatorNode::OP_AND: {
					_REDUCE_BINARY(Variant::OP_AND);
				} break;
				case OperatorNode::OP_OR: {
					_REDUCE_BINARY(Variant::OP_OR);
				} break;
				case OperatorNode::OP_ADD: {
					_REDUCE_BINARY(Variant::OP_ADD);
				} break;
				case OperatorNode::OP_SUB: {
					_REDUCE_BINARY(Variant::OP_SUBTRACT);
				} break;
				case OperatorNode::OP_MUL: {
					_REDUCE_BINARY(Variant::OP_MULTIPLY);
				} break;
				case OperatorNode::OP_DIV: {
					_REDUCE_BINARY(Variant::OP_DIVIDE);
				} break;
				case OperatorNode::OP_MOD: {
					_REDUCE_BINARY(Variant::OP_MODULE);
				} break;
				case OperatorNode::OP_SHIFT_LEFT: {
					_REDUCE_BINARY(Variant::OP_SHIFT_LEFT);
				} break;
				case OperatorNode::OP_SHIFT_RIGHT: {
					_REDUCE_BINARY(Variant::OP_SHIFT_RIGHT);
				} break;
				case OperatorNode::OP_BIT_AND: {
					_REDUCE_BINARY(Variant::OP_BIT_AND);
				} break;
				case OperatorNode::OP_BIT_OR: {
					_REDUCE_BINARY(Variant::OP_BIT_OR);
				} break;
				case OperatorNode::OP_BIT_XOR: {
					_REDUCE_BINARY(Variant::OP_BIT_XOR);
				} break;
				default: { ERR_FAIL_V(op); }
			}

			ERR_FAIL_V(op);
		} break;
		default: {
			return p_node;
		} break;
	}
}

GDParser::Node *GDParser::_parse_and_reduce_expression(Node *p_parent, bool p_static, bool p_reduce_const, bool p_allow_assign) {

	Node *expr = _parse_expression(p_parent, p_static, p_allow_assign, p_reduce_const);
	if (!expr || error_set)
		return NULL;
	expr = _reduce_expression(expr, p_reduce_const);
	if (!expr || error_set)
		return NULL;
	return expr;
}

bool GDParser::_recover_from_completion() {

	if (!completion_found) {
		return false; //can't recover if no completion
	}
	//skip stuff until newline
	while (tokenizer->get_token() != GDTokenizer::TK_NEWLINE && tokenizer->get_token() != GDTokenizer::TK_EOF && tokenizer->get_token() != GDTokenizer::TK_ERROR) {
		tokenizer->advance();
	}
	completion_found = false;
	error_set = false;
	if (tokenizer->get_token() == GDTokenizer::TK_ERROR) {
		error_set = true;
	}

	return true;
}

GDParser::PatternNode *GDParser::_parse_pattern(bool p_static) {

	PatternNode *pattern = alloc_node<PatternNode>();

	GDTokenizer::Token token = tokenizer->get_token();
	if (error_set)
		return NULL;

	if (token == GDTokenizer::TK_EOF) {
		return NULL;
	}

	switch (token) {
		// array
		case GDTokenizer::TK_BRACKET_OPEN: {
			tokenizer->advance();
			pattern->pt_type = GDParser::PatternNode::PT_ARRAY;
			while (true) {

				if (tokenizer->get_token() == GDTokenizer::TK_BRACKET_CLOSE) {
					tokenizer->advance();
					break;
				}

				if (tokenizer->get_token() == GDTokenizer::TK_PERIOD && tokenizer->get_token(1) == GDTokenizer::TK_PERIOD) {
					// match everything
					tokenizer->advance(2);
					PatternNode *sub_pattern = alloc_node<PatternNode>();
					sub_pattern->pt_type = GDParser::PatternNode::PT_IGNORE_REST;
					pattern->array.push_back(sub_pattern);
					if (tokenizer->get_token() == GDTokenizer::TK_COMMA && tokenizer->get_token(1) == GDTokenizer::TK_BRACKET_CLOSE) {
						tokenizer->advance(2);
						break;
					} else if (tokenizer->get_token() == GDTokenizer::TK_BRACKET_CLOSE) {
						tokenizer->advance(1);
						break;
					} else {
						_set_error("'..' pattern only allowed at the end of an array pattern");
						return NULL;
					}
				}

				PatternNode *sub_pattern = _parse_pattern(p_static);
				if (!sub_pattern) {
					return NULL;
				}

				pattern->array.push_back(sub_pattern);

				if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {
					tokenizer->advance();
					continue;
				} else if (tokenizer->get_token() == GDTokenizer::TK_BRACKET_CLOSE) {
					tokenizer->advance();
					break;
				} else {
					_set_error("Not a valid pattern");
					return NULL;
				}
			}
		} break;
		// bind
		case GDTokenizer::TK_PR_VAR: {
			tokenizer->advance();
			pattern->pt_type = GDParser::PatternNode::PT_BIND;
			pattern->bind = tokenizer->get_token_identifier();
			tokenizer->advance();
		} break;
		// dictionary
		case GDTokenizer::TK_CURLY_BRACKET_OPEN: {
			tokenizer->advance();
			pattern->pt_type = GDParser::PatternNode::PT_DICTIONARY;
			while (true) {

				if (tokenizer->get_token() == GDTokenizer::TK_CURLY_BRACKET_CLOSE) {
					tokenizer->advance();
					break;
				}

				if (tokenizer->get_token() == GDTokenizer::TK_PERIOD && tokenizer->get_token(1) == GDTokenizer::TK_PERIOD) {
					// match everything
					tokenizer->advance(2);
					PatternNode *sub_pattern = alloc_node<PatternNode>();
					sub_pattern->pt_type = PatternNode::PT_IGNORE_REST;
					pattern->array.push_back(sub_pattern);
					if (tokenizer->get_token() == GDTokenizer::TK_COMMA && tokenizer->get_token(1) == GDTokenizer::TK_CURLY_BRACKET_CLOSE) {
						tokenizer->advance(2);
						break;
					} else if (tokenizer->get_token() == GDTokenizer::TK_CURLY_BRACKET_CLOSE) {
						tokenizer->advance(1);
						break;
					} else {
						_set_error("'..' pattern only allowed at the end of a dictionary pattern");
						return NULL;
					}
				}

				Node *key = _parse_and_reduce_expression(pattern, p_static);
				if (!key) {
					_set_error("Not a valid key in pattern");
					return NULL;
				}

				if (key->type != GDParser::Node::TYPE_CONSTANT) {
					_set_error("Not a constant expression as key");
					return NULL;
				}

				if (tokenizer->get_token() == GDTokenizer::TK_COLON) {
					tokenizer->advance();

					PatternNode *value = _parse_pattern(p_static);
					if (!value) {
						_set_error("Expected pattern in dictionary value");
						return NULL;
					}

					pattern->dictionary.insert(static_cast<ConstantNode *>(key), value);
				} else {
					pattern->dictionary.insert(static_cast<ConstantNode *>(key), NULL);
				}

				if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {
					tokenizer->advance();
					continue;
				} else if (tokenizer->get_token() == GDTokenizer::TK_CURLY_BRACKET_CLOSE) {
					tokenizer->advance();
					break;
				} else {
					_set_error("Not a valid pattern");
					return NULL;
				}
			}
		} break;
		case GDTokenizer::TK_WILDCARD: {
			tokenizer->advance();
			pattern->pt_type = PatternNode::PT_WILDCARD;
		} break;
		// all the constants like strings and numbers
		default: {
			Node *value = _parse_and_reduce_expression(pattern, p_static);
			if (!value) {
				_set_error("Expect constant expression or variables in a pattern");
				return NULL;
			}

			if (value->type == Node::TYPE_OPERATOR) {
				// Maybe it's SomeEnum.VALUE
				Node *current_value = value;

				while (current_value->type == Node::TYPE_OPERATOR) {
					OperatorNode *op_node = static_cast<OperatorNode *>(current_value);

					if (op_node->op != OperatorNode::OP_INDEX_NAMED) {
						_set_error("Invalid operator in pattern. Only index (`A.B`) is allowed");
						return NULL;
					}
					current_value = op_node->arguments[0];
				}

				if (current_value->type != Node::TYPE_IDENTIFIER) {
					_set_error("Only constant expression or variables allowed in a pattern");
					return NULL;
				}

			} else if (value->type != Node::TYPE_IDENTIFIER && value->type != Node::TYPE_CONSTANT) {
				_set_error("Only constant expressions or variables allowed in a pattern");
				return NULL;
			}

			pattern->pt_type = PatternNode::PT_CONSTANT;
			pattern->constant = value;
		} break;
	}

	return pattern;
}

void GDParser::_parse_pattern_block(BlockNode *p_block, Vector<PatternBranchNode *> &p_branches, bool p_static) {
	int indent_level = tab_level.back()->get();

	while (true) {

		while (tokenizer->get_token() == GDTokenizer::TK_NEWLINE && _parse_newline())
			;

		// GDTokenizer::Token token = tokenizer->get_token();
		if (error_set)
			return;

		if (indent_level > tab_level.back()->get()) {
			return; // go back a level
		}

		if (pending_newline != -1) {
			pending_newline = -1;
		}

		PatternBranchNode *branch = alloc_node<PatternBranchNode>();

		branch->patterns.push_back(_parse_pattern(p_static));
		if (!branch->patterns[0]) {
			return;
		}

		while (tokenizer->get_token() == GDTokenizer::TK_COMMA) {
			tokenizer->advance();
			branch->patterns.push_back(_parse_pattern(p_static));
			if (!branch->patterns[branch->patterns.size() - 1]) {
				return;
			}
		}

		if (!_enter_indent_block()) {
			_set_error("Expected block in pattern branch");
			return;
		}

		branch->body = alloc_node<BlockNode>();
		branch->body->parent_block = p_block;
		p_block->sub_blocks.push_back(branch->body);
		current_block = branch->body;

		_parse_block(branch->body, p_static);

		current_block = p_block;

		p_branches.push_back(branch);
	}
}

void GDParser::_generate_pattern(PatternNode *p_pattern, Node *p_node_to_match, Node *&p_resulting_node, Map<StringName, Node *> &p_bindings) {
	switch (p_pattern->pt_type) {
		case PatternNode::PT_CONSTANT: {

			// typecheck
			BuiltInFunctionNode *typeof_node = alloc_node<BuiltInFunctionNode>();
			typeof_node->function = GDFunctions::TYPE_OF;

			OperatorNode *typeof_match_value = alloc_node<OperatorNode>();
			typeof_match_value->op = OperatorNode::OP_CALL;
			typeof_match_value->arguments.push_back(typeof_node);
			typeof_match_value->arguments.push_back(p_node_to_match);

			OperatorNode *typeof_pattern_value = alloc_node<OperatorNode>();
			typeof_pattern_value->op = OperatorNode::OP_CALL;
			typeof_pattern_value->arguments.push_back(typeof_node);
			typeof_pattern_value->arguments.push_back(p_pattern->constant);

			OperatorNode *type_comp = alloc_node<OperatorNode>();
			type_comp->op = OperatorNode::OP_EQUAL;
			type_comp->arguments.push_back(typeof_match_value);
			type_comp->arguments.push_back(typeof_pattern_value);

			// comare the actual values
			OperatorNode *value_comp = alloc_node<OperatorNode>();
			value_comp->op = OperatorNode::OP_EQUAL;
			value_comp->arguments.push_back(p_pattern->constant);
			value_comp->arguments.push_back(p_node_to_match);

			OperatorNode *comparison = alloc_node<OperatorNode>();
			comparison->op = OperatorNode::OP_AND;
			comparison->arguments.push_back(type_comp);
			comparison->arguments.push_back(value_comp);

			p_resulting_node = comparison;

		} break;
		case PatternNode::PT_BIND: {
			p_bindings[p_pattern->bind] = p_node_to_match;

			// a bind always matches
			ConstantNode *true_value = alloc_node<ConstantNode>();
			true_value->value = Variant(true);
			p_resulting_node = true_value;
		} break;
		case PatternNode::PT_ARRAY: {

			bool open_ended = false;

			if (p_pattern->array.size() > 0) {
				if (p_pattern->array[p_pattern->array.size() - 1]->pt_type == PatternNode::PT_IGNORE_REST) {
					open_ended = true;
				}
			}

			// typeof(value_to_match) == TYPE_ARRAY && value_to_match.size() >= length
			// typeof(value_to_match) == TYPE_ARRAY && value_to_match.size() == length

			{
				// typecheck
				BuiltInFunctionNode *typeof_node = alloc_node<BuiltInFunctionNode>();
				typeof_node->function = GDFunctions::TYPE_OF;

				OperatorNode *typeof_match_value = alloc_node<OperatorNode>();
				typeof_match_value->op = OperatorNode::OP_CALL;
				typeof_match_value->arguments.push_back(typeof_node);
				typeof_match_value->arguments.push_back(p_node_to_match);

				IdentifierNode *typeof_array = alloc_node<IdentifierNode>();
				typeof_array->name = "TYPE_ARRAY";

				OperatorNode *type_comp = alloc_node<OperatorNode>();
				type_comp->op = OperatorNode::OP_EQUAL;
				type_comp->arguments.push_back(typeof_match_value);
				type_comp->arguments.push_back(typeof_array);

				// size
				ConstantNode *length = alloc_node<ConstantNode>();
				length->value = Variant(open_ended ? p_pattern->array.size() - 1 : p_pattern->array.size());

				OperatorNode *call = alloc_node<OperatorNode>();
				call->op = OperatorNode::OP_CALL;
				call->arguments.push_back(p_node_to_match);

				IdentifierNode *size = alloc_node<IdentifierNode>();
				size->name = "size";
				call->arguments.push_back(size);

				OperatorNode *length_comparison = alloc_node<OperatorNode>();
				length_comparison->op = open_ended ? OperatorNode::OP_GREATER_EQUAL : OperatorNode::OP_EQUAL;
				length_comparison->arguments.push_back(call);
				length_comparison->arguments.push_back(length);

				OperatorNode *type_and_length_comparison = alloc_node<OperatorNode>();
				type_and_length_comparison->op = OperatorNode::OP_AND;
				type_and_length_comparison->arguments.push_back(type_comp);
				type_and_length_comparison->arguments.push_back(length_comparison);

				p_resulting_node = type_and_length_comparison;
			}

			for (int i = 0; i < p_pattern->array.size(); i++) {
				PatternNode *pattern = p_pattern->array[i];

				Node *condition = NULL;

				ConstantNode *index = alloc_node<ConstantNode>();
				index->value = Variant(i);

				OperatorNode *indexed_value = alloc_node<OperatorNode>();
				indexed_value->op = OperatorNode::OP_INDEX;
				indexed_value->arguments.push_back(p_node_to_match);
				indexed_value->arguments.push_back(index);

				_generate_pattern(pattern, indexed_value, condition, p_bindings);

				// concatenate all the patterns with &&
				OperatorNode *and_node = alloc_node<OperatorNode>();
				and_node->op = OperatorNode::OP_AND;
				and_node->arguments.push_back(p_resulting_node);
				and_node->arguments.push_back(condition);

				p_resulting_node = and_node;
			}

		} break;
		case PatternNode::PT_DICTIONARY: {

			bool open_ended = false;

			if (p_pattern->array.size() > 0) {
				open_ended = true;
			}

			// typeof(value_to_match) == TYPE_DICTIONARY && value_to_match.size() >= length
			// typeof(value_to_match) == TYPE_DICTIONARY && value_to_match.size() == length

			{
				// typecheck
				BuiltInFunctionNode *typeof_node = alloc_node<BuiltInFunctionNode>();
				typeof_node->function = GDFunctions::TYPE_OF;

				OperatorNode *typeof_match_value = alloc_node<OperatorNode>();
				typeof_match_value->op = OperatorNode::OP_CALL;
				typeof_match_value->arguments.push_back(typeof_node);
				typeof_match_value->arguments.push_back(p_node_to_match);

				IdentifierNode *typeof_dictionary = alloc_node<IdentifierNode>();
				typeof_dictionary->name = "TYPE_DICTIONARY";

				OperatorNode *type_comp = alloc_node<OperatorNode>();
				type_comp->op = OperatorNode::OP_EQUAL;
				type_comp->arguments.push_back(typeof_match_value);
				type_comp->arguments.push_back(typeof_dictionary);

				// size
				ConstantNode *length = alloc_node<ConstantNode>();
				length->value = Variant(open_ended ? p_pattern->dictionary.size() - 1 : p_pattern->dictionary.size());

				OperatorNode *call = alloc_node<OperatorNode>();
				call->op = OperatorNode::OP_CALL;
				call->arguments.push_back(p_node_to_match);

				IdentifierNode *size = alloc_node<IdentifierNode>();
				size->name = "size";
				call->arguments.push_back(size);

				OperatorNode *length_comparison = alloc_node<OperatorNode>();
				length_comparison->op = open_ended ? OperatorNode::OP_GREATER_EQUAL : OperatorNode::OP_EQUAL;
				length_comparison->arguments.push_back(call);
				length_comparison->arguments.push_back(length);

				OperatorNode *type_and_length_comparison = alloc_node<OperatorNode>();
				type_and_length_comparison->op = OperatorNode::OP_AND;
				type_and_length_comparison->arguments.push_back(type_comp);
				type_and_length_comparison->arguments.push_back(length_comparison);

				p_resulting_node = type_and_length_comparison;
			}

			for (Map<ConstantNode *, PatternNode *>::Element *e = p_pattern->dictionary.front(); e; e = e->next()) {

				Node *condition = NULL;

				// chech for has, then for pattern

				IdentifierNode *has = alloc_node<IdentifierNode>();
				has->name = "has";

				OperatorNode *has_call = alloc_node<OperatorNode>();
				has_call->op = OperatorNode::OP_CALL;
				has_call->arguments.push_back(p_node_to_match);
				has_call->arguments.push_back(has);
				has_call->arguments.push_back(e->key());

				if (e->value()) {

					OperatorNode *indexed_value = alloc_node<OperatorNode>();
					indexed_value->op = OperatorNode::OP_INDEX;
					indexed_value->arguments.push_back(p_node_to_match);
					indexed_value->arguments.push_back(e->key());

					_generate_pattern(e->value(), indexed_value, condition, p_bindings);

					OperatorNode *has_and_pattern = alloc_node<OperatorNode>();
					has_and_pattern->op = OperatorNode::OP_AND;
					has_and_pattern->arguments.push_back(has_call);
					has_and_pattern->arguments.push_back(condition);

					condition = has_and_pattern;

				} else {
					condition = has_call;
				}

				// concatenate all the patterns with &&
				OperatorNode *and_node = alloc_node<OperatorNode>();
				and_node->op = OperatorNode::OP_AND;
				and_node->arguments.push_back(p_resulting_node);
				and_node->arguments.push_back(condition);

				p_resulting_node = and_node;
			}

		} break;
		case PatternNode::PT_IGNORE_REST:
		case PatternNode::PT_WILDCARD: {
			// simply generate a `true`
			ConstantNode *true_value = alloc_node<ConstantNode>();
			true_value->value = Variant(true);
			p_resulting_node = true_value;
		} break;
		default: {

		} break;
	}
}

void GDParser::_transform_match_statment(BlockNode *p_block, MatchNode *p_match_statement) {
	IdentifierNode *id = alloc_node<IdentifierNode>();
	id->name = "#match_value";

	for (int i = 0; i < p_match_statement->branches.size(); i++) {

		PatternBranchNode *branch = p_match_statement->branches[i];

		MatchNode::CompiledPatternBranch compiled_branch;
		compiled_branch.compiled_pattern = NULL;

		Map<StringName, Node *> binding;

		for (int j = 0; j < branch->patterns.size(); j++) {
			PatternNode *pattern = branch->patterns[j];

			Map<StringName, Node *> bindings;
			Node *resulting_node;
			_generate_pattern(pattern, id, resulting_node, bindings);

			if (!binding.empty() && !bindings.empty()) {
				_set_error("Multipatterns can't contain bindings");
				return;
			} else {
				binding = bindings;
			}

			if (compiled_branch.compiled_pattern) {
				OperatorNode *or_node = alloc_node<OperatorNode>();
				or_node->op = OperatorNode::OP_OR;
				or_node->arguments.push_back(compiled_branch.compiled_pattern);
				or_node->arguments.push_back(resulting_node);

				compiled_branch.compiled_pattern = or_node;
			} else {
				// single pattern | first one
				compiled_branch.compiled_pattern = resulting_node;
			}
		}

		// prepare the body ...hehe
		for (Map<StringName, Node *>::Element *e = binding.front(); e; e = e->next()) {
			LocalVarNode *local_var = alloc_node<LocalVarNode>();
			local_var->name = e->key();
			local_var->assign = e->value();

			IdentifierNode *id = alloc_node<IdentifierNode>();
			id->name = local_var->name;

			OperatorNode *op = alloc_node<OperatorNode>();
			op->op = OperatorNode::OP_ASSIGN;
			op->arguments.push_back(id);
			op->arguments.push_back(local_var->assign);

			branch->body->statements.push_front(op);
			branch->body->statements.push_front(local_var);
		}

		compiled_branch.body = branch->body;

		p_match_statement->compiled_pattern_branches.push_back(compiled_branch);
	}
}

void GDParser::_parse_block(BlockNode *p_block, bool p_static) {

	int indent_level = tab_level.back()->get();

#ifdef DEBUG_ENABLED

	NewLineNode *nl = alloc_node<NewLineNode>();

	nl->line = tokenizer->get_token_line();
	p_block->statements.push_back(nl);
#endif

	bool is_first_line = true;

	while (true) {
		if (!is_first_line && tab_level.back()->prev() && tab_level.back()->prev()->get() == indent_level) {
			// pythonic single-line expression, don't parse future lines
			tab_level.pop_back();
			p_block->end_line = tokenizer->get_token_line();
			return;
		}
		is_first_line = false;

		GDTokenizer::Token token = tokenizer->get_token();
		if (error_set)
			return;

		if (indent_level > tab_level.back()->get()) {
			p_block->end_line = tokenizer->get_token_line();
			return; //go back a level
		}

		if (pending_newline != -1) {

			NewLineNode *nl = alloc_node<NewLineNode>();
			nl->line = pending_newline;
			p_block->statements.push_back(nl);
			pending_newline = -1;
		}

		switch (token) {

			case GDTokenizer::TK_EOF:
				p_block->end_line = tokenizer->get_token_line();
			case GDTokenizer::TK_ERROR: {
				return; //go back

				//end of file!

			} break;
			case GDTokenizer::TK_NEWLINE: {

				if (!_parse_newline()) {
					if (!error_set) {
						p_block->end_line = tokenizer->get_token_line();
						pending_newline = p_block->end_line;
					}
					return;
				}

				NewLineNode *nl = alloc_node<NewLineNode>();
				nl->line = tokenizer->get_token_line();
				p_block->statements.push_back(nl);

			} break;
			case GDTokenizer::TK_CF_PASS: {
				if (tokenizer->get_token(1) != GDTokenizer::TK_SEMICOLON && tokenizer->get_token(1) != GDTokenizer::TK_NEWLINE && tokenizer->get_token(1) != GDTokenizer::TK_EOF) {

					_set_error("Expected ';' or <NewLine>.");
					return;
				}
				tokenizer->advance();
				if (tokenizer->get_token() == GDTokenizer::TK_SEMICOLON) {
					// Ignore semicolon after 'pass'
					tokenizer->advance();
				}
			} break;
			case GDTokenizer::TK_PR_VAR: {
				//variale declaration and (eventual) initialization

				tokenizer->advance();
				if (!tokenizer->is_token_literal(0, true)) {

					_set_error("Expected identifier for local variable name.");
					return;
				}
				StringName n = tokenizer->get_token_literal();
				tokenizer->advance();
				if (current_function) {
					for (int i = 0; i < current_function->arguments.size(); i++) {
						if (n == current_function->arguments[i]) {
							_set_error("Variable '" + String(n) + "' already defined in the scope (at line: " + itos(current_function->line) + ").");
							return;
						}
					}
				}
				BlockNode *check_block = p_block;
				while (check_block) {
					for (int i = 0; i < check_block->variables.size(); i++) {
						if (n == check_block->variables[i]) {
							_set_error("Variable '" + String(n) + "' already defined in the scope (at line: " + itos(check_block->variable_lines[i]) + ").");
							return;
						}
					}
					check_block = check_block->parent_block;
				}

				int var_line = tokenizer->get_token_line();

				//must know when the local variable is declared
				LocalVarNode *lv = alloc_node<LocalVarNode>();
				lv->name = n;
				p_block->statements.push_back(lv);

				Node *assigned = NULL;

				if (tokenizer->get_token() == GDTokenizer::TK_OP_ASSIGN) {

					tokenizer->advance();
					Node *subexpr = _parse_and_reduce_expression(p_block, p_static);
					if (!subexpr) {
						if (_recover_from_completion()) {
							break;
						}
						return;
					}

					lv->assign = subexpr;
					assigned = subexpr;
				} else {

					ConstantNode *c = alloc_node<ConstantNode>();
					c->value = Variant();
					assigned = c;
				}
				//must be added later, to avoid self-referencing.
				p_block->variables.push_back(n); //line?
				p_block->variable_lines.push_back(var_line);

				IdentifierNode *id = alloc_node<IdentifierNode>();
				id->name = n;

				OperatorNode *op = alloc_node<OperatorNode>();
				op->op = OperatorNode::OP_ASSIGN;
				op->arguments.push_back(id);
				op->arguments.push_back(assigned);
				p_block->statements.push_back(op);

				if (!_end_statement()) {
					_set_error("Expected end of statement (var)");
					return;
				}

			} break;
			case GDTokenizer::TK_CF_IF: {

				tokenizer->advance();

				Node *condition = _parse_and_reduce_expression(p_block, p_static);
				if (!condition) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}

				ControlFlowNode *cf_if = alloc_node<ControlFlowNode>();

				cf_if->cf_type = ControlFlowNode::CF_IF;
				cf_if->arguments.push_back(condition);

				cf_if->body = alloc_node<BlockNode>();
				cf_if->body->parent_block = p_block;
				cf_if->body->if_condition = condition; //helps code completion

				p_block->sub_blocks.push_back(cf_if->body);

				if (!_enter_indent_block(cf_if->body)) {
					_set_error("Expected indented block after 'if'");
					p_block->end_line = tokenizer->get_token_line();
					return;
				}

				current_block = cf_if->body;
				_parse_block(cf_if->body, p_static);
				current_block = p_block;

				if (error_set)
					return;
				p_block->statements.push_back(cf_if);

				while (true) {

					while (tokenizer->get_token() == GDTokenizer::TK_NEWLINE && _parse_newline())
						;

					if (tab_level.back()->get() < indent_level) { //not at current indent level
						p_block->end_line = tokenizer->get_token_line();
						return;
					}

					if (tokenizer->get_token() == GDTokenizer::TK_CF_ELIF) {

						if (tab_level.back()->get() > indent_level) {

							_set_error("Invalid indent");
							return;
						}

						tokenizer->advance();

						cf_if->body_else = alloc_node<BlockNode>();
						cf_if->body_else->parent_block = p_block;
						p_block->sub_blocks.push_back(cf_if->body_else);

						ControlFlowNode *cf_else = alloc_node<ControlFlowNode>();
						cf_else->cf_type = ControlFlowNode::CF_IF;

						//condition
						Node *condition = _parse_and_reduce_expression(p_block, p_static);
						if (!condition) {
							if (_recover_from_completion()) {
								break;
							}
							return;
						}
						cf_else->arguments.push_back(condition);
						cf_else->cf_type = ControlFlowNode::CF_IF;

						cf_if->body_else->statements.push_back(cf_else);
						cf_if = cf_else;
						cf_if->body = alloc_node<BlockNode>();
						cf_if->body->parent_block = p_block;
						p_block->sub_blocks.push_back(cf_if->body);

						if (!_enter_indent_block(cf_if->body)) {
							_set_error("Expected indented block after 'elif'");
							p_block->end_line = tokenizer->get_token_line();
							return;
						}

						current_block = cf_else->body;
						_parse_block(cf_else->body, p_static);
						current_block = p_block;
						if (error_set)
							return;

					} else if (tokenizer->get_token() == GDTokenizer::TK_CF_ELSE) {

						if (tab_level.back()->get() > indent_level) {
							_set_error("Invalid indent");
							return;
						}

						tokenizer->advance();
						cf_if->body_else = alloc_node<BlockNode>();
						cf_if->body_else->parent_block = p_block;
						p_block->sub_blocks.push_back(cf_if->body_else);

						if (!_enter_indent_block(cf_if->body_else)) {
							_set_error("Expected indented block after 'else'");
							p_block->end_line = tokenizer->get_token_line();
							return;
						}
						current_block = cf_if->body_else;
						_parse_block(cf_if->body_else, p_static);
						current_block = p_block;
						if (error_set)
							return;

						break; //after else, exit

					} else
						break;
				}

			} break;
			case GDTokenizer::TK_CF_WHILE: {

				tokenizer->advance();
				Node *condition = _parse_and_reduce_expression(p_block, p_static);
				if (!condition) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}

				ControlFlowNode *cf_while = alloc_node<ControlFlowNode>();

				cf_while->cf_type = ControlFlowNode::CF_WHILE;
				cf_while->arguments.push_back(condition);

				cf_while->body = alloc_node<BlockNode>();
				cf_while->body->parent_block = p_block;
				p_block->sub_blocks.push_back(cf_while->body);

				if (!_enter_indent_block(cf_while->body)) {
					_set_error("Expected indented block after 'while'");
					p_block->end_line = tokenizer->get_token_line();
					return;
				}

				current_block = cf_while->body;
				_parse_block(cf_while->body, p_static);
				current_block = p_block;
				if (error_set)
					return;
				p_block->statements.push_back(cf_while);
			} break;
			case GDTokenizer::TK_CF_FOR: {

				tokenizer->advance();

				if (!tokenizer->is_token_literal(0, true)) {

					_set_error("identifier expected after 'for'");
				}

				IdentifierNode *id = alloc_node<IdentifierNode>();
				id->name = tokenizer->get_token_identifier();

				tokenizer->advance();

				if (tokenizer->get_token() != GDTokenizer::TK_OP_IN) {
					_set_error("'in' expected after identifier");
					return;
				}

				tokenizer->advance();

				Node *container = _parse_and_reduce_expression(p_block, p_static);
				if (!container) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}

				if (container->type == Node::TYPE_OPERATOR) {

					OperatorNode *op = static_cast<OperatorNode *>(container);
					if (op->op == OperatorNode::OP_CALL && op->arguments[0]->type == Node::TYPE_BUILT_IN_FUNCTION && static_cast<BuiltInFunctionNode *>(op->arguments[0])->function == GDFunctions::GEN_RANGE) {
						//iterating a range, so see if range() can be optimized without allocating memory, by replacing it by vectors (which can work as iterable too!)

						Vector<Node *> args;
						Vector<double> constants;

						bool constant = false;

						for (int i = 1; i < op->arguments.size(); i++) {
							args.push_back(op->arguments[i]);
							if (constant && op->arguments[i]->type == Node::TYPE_CONSTANT) {
								ConstantNode *c = static_cast<ConstantNode *>(op->arguments[i]);
								if (c->value.get_type() == Variant::REAL || c->value.get_type() == Variant::INT) {
									constants.push_back(c->value);
									constant = true;
								}
							} else {
								constant = false;
							}
						}

						if (args.size() > 0 && args.size() < 4) {

							if (constant) {

								ConstantNode *cn = alloc_node<ConstantNode>();
								switch (args.size()) {
									case 1: cn->value = (int)constants[0]; break;
									case 2: cn->value = Vector2(constants[0], constants[1]); break;
									case 3: cn->value = Vector3(constants[0], constants[1], constants[2]); break;
								}
								container = cn;
							} else {
								OperatorNode *on = alloc_node<OperatorNode>();
								on->op = OperatorNode::OP_CALL;

								TypeNode *tn = alloc_node<TypeNode>();
								on->arguments.push_back(tn);

								switch (args.size()) {
									case 1: tn->vtype = Variant::INT; break;
									case 2: tn->vtype = Variant::VECTOR2; break;
									case 3: tn->vtype = Variant::VECTOR3; break;
								}

								for (int i = 0; i < args.size(); i++) {
									on->arguments.push_back(args[i]);
								}

								container = on;
							}
						}
					}
				}

				ControlFlowNode *cf_for = alloc_node<ControlFlowNode>();

				cf_for->cf_type = ControlFlowNode::CF_FOR;
				cf_for->arguments.push_back(id);
				cf_for->arguments.push_back(container);

				cf_for->body = alloc_node<BlockNode>();
				cf_for->body->parent_block = p_block;
				p_block->sub_blocks.push_back(cf_for->body);

				if (!_enter_indent_block(cf_for->body)) {
					_set_error("Expected indented block after 'for'");
					p_block->end_line = tokenizer->get_token_line();
					return;
				}

				current_block = cf_for->body;

				// this is for checking variable for redefining
				// inside this _parse_block
				cf_for->body->variables.push_back(id->name);
				cf_for->body->variable_lines.push_back(id->line);
				_parse_block(cf_for->body, p_static);
				cf_for->body->variables.remove(0);
				cf_for->body->variable_lines.remove(0);
				current_block = p_block;

				if (error_set)
					return;
				p_block->statements.push_back(cf_for);
			} break;
			case GDTokenizer::TK_CF_CONTINUE: {

				tokenizer->advance();
				ControlFlowNode *cf_continue = alloc_node<ControlFlowNode>();
				cf_continue->cf_type = ControlFlowNode::CF_CONTINUE;
				p_block->statements.push_back(cf_continue);
				if (!_end_statement()) {
					_set_error("Expected end of statement (continue)");
					return;
				}
			} break;
			case GDTokenizer::TK_CF_BREAK: {

				tokenizer->advance();
				ControlFlowNode *cf_break = alloc_node<ControlFlowNode>();
				cf_break->cf_type = ControlFlowNode::CF_BREAK;
				p_block->statements.push_back(cf_break);
				if (!_end_statement()) {
					_set_error("Expected end of statement (break)");
					return;
				}
			} break;
			case GDTokenizer::TK_CF_RETURN: {

				tokenizer->advance();
				ControlFlowNode *cf_return = alloc_node<ControlFlowNode>();
				cf_return->cf_type = ControlFlowNode::CF_RETURN;

				if (tokenizer->get_token() == GDTokenizer::TK_SEMICOLON || tokenizer->get_token() == GDTokenizer::TK_NEWLINE || tokenizer->get_token() == GDTokenizer::TK_EOF) {
					//expect end of statement
					p_block->statements.push_back(cf_return);
					if (!_end_statement()) {
						return;
					}
				} else {
					//expect expression
					Node *retexpr = _parse_and_reduce_expression(p_block, p_static);
					if (!retexpr) {
						if (_recover_from_completion()) {
							break;
						}
						return;
					}
					cf_return->arguments.push_back(retexpr);
					p_block->statements.push_back(cf_return);
					if (!_end_statement()) {
						_set_error("Expected end of statement after return expression.");
						return;
					}
				}

			} break;
			case GDTokenizer::TK_CF_MATCH: {

				tokenizer->advance();

				MatchNode *match_node = alloc_node<MatchNode>();

				Node *val_to_match = _parse_and_reduce_expression(p_block, p_static);

				if (!val_to_match) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}

				match_node->val_to_match = val_to_match;

				if (!_enter_indent_block()) {
					_set_error("Expected indented pattern matching block after 'match'");
					return;
				}

				BlockNode *compiled_branches = alloc_node<BlockNode>();
				compiled_branches->parent_block = p_block;
				compiled_branches->parent_class = p_block->parent_class;

				p_block->sub_blocks.push_back(compiled_branches);

				_parse_pattern_block(compiled_branches, match_node->branches, p_static);

				_transform_match_statment(compiled_branches, match_node);

				ControlFlowNode *match_cf_node = alloc_node<ControlFlowNode>();
				match_cf_node->cf_type = ControlFlowNode::CF_MATCH;
				match_cf_node->match = match_node;

				p_block->statements.push_back(match_cf_node);

				_end_statement();
			} break;
			case GDTokenizer::TK_PR_ASSERT: {

				tokenizer->advance();
				Node *condition = _parse_and_reduce_expression(p_block, p_static);
				if (!condition) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}
				AssertNode *an = alloc_node<AssertNode>();
				an->condition = condition;
				p_block->statements.push_back(an);

				if (!_end_statement()) {
					_set_error("Expected end of statement after assert.");
					return;
				}
			} break;
			case GDTokenizer::TK_PR_BREAKPOINT: {

				tokenizer->advance();
				BreakpointNode *bn = alloc_node<BreakpointNode>();
				p_block->statements.push_back(bn);

				if (!_end_statement()) {
					_set_error("Expected end of statement after breakpoint.");
					return;
				}
			} break;
			default: {

				Node *expression = _parse_and_reduce_expression(p_block, p_static, false, true);
				if (!expression) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}
				p_block->statements.push_back(expression);
				if (!_end_statement()) {
					_set_error("Expected end of statement after expression.");
					return;
				}

			} break;
				/*
			case GDTokenizer::TK_CF_LOCAL: {

				if (tokenizer->get_token(1)!=GDTokenizer::TK_SEMICOLON && tokenizer->get_token(1)!=GDTokenizer::TK_NEWLINE ) {

					_set_error("Expected ';' or <NewLine>.");
				}
				tokenizer->advance();
			} break;
			*/
		}
	}
}

bool GDParser::_parse_newline() {

	if (tokenizer->get_token(1) != GDTokenizer::TK_EOF && tokenizer->get_token(1) != GDTokenizer::TK_NEWLINE) {

		int indent = tokenizer->get_token_line_indent();
		int current_indent = tab_level.back()->get();

		if (indent > current_indent) {
			_set_error("Unexpected indent.");
			return false;
		}

		if (indent < current_indent) {

			while (indent < current_indent) {

				//exit block
				if (tab_level.size() == 1) {
					_set_error("Invalid indent. BUG?");
					return false;
				}

				tab_level.pop_back();

				if (tab_level.back()->get() < indent) {

					_set_error("Unindent does not match any outer indentation level.");
					return false;
				}
				current_indent = tab_level.back()->get();
			}

			tokenizer->advance();
			return false;
		}
	}

	tokenizer->advance();
	return true;
}

void GDParser::_parse_extends(ClassNode *p_class) {

	if (p_class->extends_used) {

		_set_error("'extends' already used for this class.");
		return;
	}

	if (!p_class->constant_expressions.empty() || !p_class->subclasses.empty() || !p_class->functions.empty() || !p_class->variables.empty()) {

		_set_error("'extends' must be used before anything else.");
		return;
	}

	p_class->extends_used = true;

	tokenizer->advance();

	if (tokenizer->get_token() == GDTokenizer::TK_BUILT_IN_TYPE && tokenizer->get_token_type() == Variant::OBJECT) {
		p_class->extends_class.push_back(Variant::get_type_name(Variant::OBJECT));
		tokenizer->advance();
		return;
	}

	// see if inheritance happens from a file
	if (tokenizer->get_token() == GDTokenizer::TK_CONSTANT) {

		Variant constant = tokenizer->get_token_constant();
		if (constant.get_type() != Variant::STRING) {

			_set_error("'extends' constant must be a string.");
			return;
		}

		p_class->extends_file = constant;
		tokenizer->advance();

		if (tokenizer->get_token() != GDTokenizer::TK_PERIOD) {
			return;
		} else
			tokenizer->advance();
	}

	while (true) {
		if (tokenizer->get_token() != GDTokenizer::TK_IDENTIFIER) {

			_set_error("Invalid 'extends' syntax, expected string constant (path) and/or identifier (parent class).");
			return;
		}

		StringName identifier = tokenizer->get_token_identifier();
		p_class->extends_class.push_back(identifier);

		tokenizer->advance(1);
		if (tokenizer->get_token() != GDTokenizer::TK_PERIOD)
			return;
	}
}

void GDParser::_parse_class(ClassNode *p_class) {

	int indent_level = tab_level.back()->get();

	while (true) {

		GDTokenizer::Token token = tokenizer->get_token();
		if (error_set)
			return;

		if (indent_level > tab_level.back()->get()) {
			p_class->end_line = tokenizer->get_token_line();
			return; //go back a level
		}

		switch (token) {

			case GDTokenizer::TK_EOF:
				p_class->end_line = tokenizer->get_token_line();
			case GDTokenizer::TK_ERROR: {
				return; //go back
				//end of file!
			} break;
			case GDTokenizer::TK_NEWLINE: {
				if (!_parse_newline()) {
					if (!error_set) {
						p_class->end_line = tokenizer->get_token_line();
					}
					return;
				}
			} break;
			case GDTokenizer::TK_PR_EXTENDS: {

				_parse_extends(p_class);
				if (error_set)
					return;
				if (!_end_statement()) {
					_set_error("Expected end of statement after extends");
					return;
				}

			} break;
			case GDTokenizer::TK_PR_TOOL: {

				if (p_class->tool) {

					_set_error("tool used more than once");
					return;
				}

				p_class->tool = true;
				tokenizer->advance();

			} break;
			case GDTokenizer::TK_PR_CLASS: {
				//class inside class :D

				StringName name;
				StringName extends;

				if (tokenizer->get_token(1) != GDTokenizer::TK_IDENTIFIER) {

					_set_error("'class' syntax: 'class <Name>:' or 'class <Name> extends <BaseClass>:'");
					return;
				}
				name = tokenizer->get_token_identifier(1);
				tokenizer->advance(2);

				ClassNode *newclass = alloc_node<ClassNode>();
				newclass->initializer = alloc_node<BlockNode>();
				newclass->initializer->parent_class = newclass;
				newclass->ready = alloc_node<BlockNode>();
				newclass->ready->parent_class = newclass;
				newclass->name = name;
				newclass->owner = p_class;

				p_class->subclasses.push_back(newclass);

				if (tokenizer->get_token() == GDTokenizer::TK_PR_EXTENDS) {

					_parse_extends(newclass);
					if (error_set)
						return;
				}

				if (!_enter_indent_block()) {

					_set_error("Indented block expected.");
					return;
				}
				current_class = newclass;
				_parse_class(newclass);
				current_class = p_class;

			} break;
			/* this is for functions....
			case GDTokenizer::TK_CF_PASS: {

				tokenizer->advance(1);
			} break;
			*/
			case GDTokenizer::TK_PR_STATIC: {
				tokenizer->advance();
				if (tokenizer->get_token() != GDTokenizer::TK_PR_FUNCTION) {

					_set_error("Expected 'func'.");
					return;
				}

			}; //fallthrough to function
			case GDTokenizer::TK_PR_FUNCTION: {

				bool _static = false;
				pending_newline = -1;

				if (tokenizer->get_token(-1) == GDTokenizer::TK_PR_STATIC) {

					_static = true;
				}

				tokenizer->advance();
				StringName name;

				if (_get_completable_identifier(COMPLETION_VIRTUAL_FUNC, name)) {
				}

				if (name == StringName()) {

					_set_error("Expected identifier after 'func' (syntax: 'func <identifier>([arguments]):' ).");
					return;
				}

				for (int i = 0; i < p_class->functions.size(); i++) {
					if (p_class->functions[i]->name == name) {
						_set_error("Function '" + String(name) + "' already exists in this class (at line: " + itos(p_class->functions[i]->line) + ").");
					}
				}
				for (int i = 0; i < p_class->static_functions.size(); i++) {
					if (p_class->static_functions[i]->name == name) {
						_set_error("Function '" + String(name) + "' already exists in this class (at line: " + itos(p_class->static_functions[i]->line) + ").");
					}
				}

				if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_OPEN) {

					_set_error("Expected '(' after identifier (syntax: 'func <identifier>([arguments]):' ).");
					return;
				}

				tokenizer->advance();

				Vector<StringName> arguments;
				Vector<Node *> default_values;

				int fnline = tokenizer->get_token_line();

				if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {
					//has arguments
					bool defaulting = false;
					while (true) {

						if (tokenizer->get_token() == GDTokenizer::TK_NEWLINE) {
							tokenizer->advance();
							continue;
						}

						if (tokenizer->get_token() == GDTokenizer::TK_PR_VAR) {

							tokenizer->advance(); //var before the identifier is allowed
						}

						if (!tokenizer->is_token_literal(0, true)) {

							_set_error("Expected identifier for argument.");
							return;
						}

						StringName argname = tokenizer->get_token_identifier();
						arguments.push_back(argname);

						tokenizer->advance();

						if (defaulting && tokenizer->get_token() != GDTokenizer::TK_OP_ASSIGN) {

							_set_error("Default parameter expected.");
							return;
						}

						//tokenizer->advance();

						if (tokenizer->get_token() == GDTokenizer::TK_OP_ASSIGN) {
							defaulting = true;
							tokenizer->advance(1);
							Node *defval = _parse_and_reduce_expression(p_class, _static);
							if (!defval || error_set)
								return;

							OperatorNode *on = alloc_node<OperatorNode>();
							on->op = OperatorNode::OP_ASSIGN;

							IdentifierNode *in = alloc_node<IdentifierNode>();
							in->name = argname;

							on->arguments.push_back(in);
							on->arguments.push_back(defval);
							/* no ..
							if (defval->type!=Node::TYPE_CONSTANT) {

								_set_error("default argument must be constant");
							}
							*/
							default_values.push_back(on);
						}

						while (tokenizer->get_token() == GDTokenizer::TK_NEWLINE) {
							tokenizer->advance();
						}

						if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {
							tokenizer->advance();
							continue;
						} else if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {

							_set_error("Expected ',' or ')'.");
							return;
						}

						break;
					}
				}

				tokenizer->advance();

				BlockNode *block = alloc_node<BlockNode>();
				block->parent_class = p_class;

				if (name == "_init") {

					if (p_class->extends_used) {

						OperatorNode *cparent = alloc_node<OperatorNode>();
						cparent->op = OperatorNode::OP_PARENT_CALL;
						block->statements.push_back(cparent);

						IdentifierNode *id = alloc_node<IdentifierNode>();
						id->name = "_init";
						cparent->arguments.push_back(id);

						if (tokenizer->get_token() == GDTokenizer::TK_PERIOD) {
							tokenizer->advance();
							if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_OPEN) {
								_set_error("expected '(' for parent constructor arguments.");
							}
							tokenizer->advance();

							if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {
								//has arguments
								parenthesis++;
								while (true) {

									Node *arg = _parse_and_reduce_expression(p_class, _static);
									cparent->arguments.push_back(arg);

									if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {
										tokenizer->advance();
										continue;
									} else if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {

										_set_error("Expected ',' or ')'.");
										return;
									}

									break;
								}
								parenthesis--;
							}

							tokenizer->advance();
						}
					} else {

						if (tokenizer->get_token() == GDTokenizer::TK_PERIOD) {

							_set_error("Parent constructor call found for a class without inheritance.");
							return;
						}
					}
				}

				if (!_enter_indent_block(block)) {

					_set_error("Indented block expected.");
					return;
				}

				FunctionNode *function = alloc_node<FunctionNode>();
				function->name = name;
				function->arguments = arguments;
				function->default_values = default_values;
				function->_static = _static;
				function->line = fnline;

				function->rpc_mode = rpc_mode;
				rpc_mode = ScriptInstance::RPC_MODE_DISABLED;

				if (_static)
					p_class->static_functions.push_back(function);
				else
					p_class->functions.push_back(function);

				current_function = function;
				function->body = block;
				current_block = block;
				_parse_block(block, _static);
				current_block = NULL;

				//arguments
			} break;
			case GDTokenizer::TK_PR_SIGNAL: {
				tokenizer->advance();

				if (!tokenizer->is_token_literal()) {
					_set_error("Expected identifier after 'signal'.");
					return;
				}

				ClassNode::Signal sig;
				sig.name = tokenizer->get_token_identifier();
				tokenizer->advance();

				if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_OPEN) {
					tokenizer->advance();
					while (true) {
						if (tokenizer->get_token() == GDTokenizer::TK_NEWLINE) {
							tokenizer->advance();
							continue;
						}

						if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE) {
							tokenizer->advance();
							break;
						}

						if (!tokenizer->is_token_literal(0, true)) {
							_set_error("Expected identifier in signal argument.");
							return;
						}

						sig.arguments.push_back(tokenizer->get_token_identifier());
						tokenizer->advance();

						while (tokenizer->get_token() == GDTokenizer::TK_NEWLINE) {
							tokenizer->advance();
						}

						if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {
							tokenizer->advance();
						} else if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {
							_set_error("Expected ',' or ')' after signal parameter identifier.");
							return;
						}
					}
				}

				p_class->_signals.push_back(sig);

				if (!_end_statement()) {
					_set_error("Expected end of statement (signal)");
					return;
				}
			} break;
			case GDTokenizer::TK_PR_EXPORT: {

				tokenizer->advance();

				if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_OPEN) {

					tokenizer->advance();
					if (tokenizer->get_token() == GDTokenizer::TK_BUILT_IN_TYPE) {

						Variant::Type type = tokenizer->get_token_type();
						if (type == Variant::NIL) {
							_set_error("Can't export null type.");
							return;
						}
						current_export.type = type;
						current_export.usage |= PROPERTY_USAGE_SCRIPT_VARIABLE;
						tokenizer->advance();

						String hint_prefix = "";

						if (type == Variant::ARRAY && tokenizer->get_token() == GDTokenizer::TK_COMMA) {
							tokenizer->advance();

							while (tokenizer->get_token() == GDTokenizer::TK_BUILT_IN_TYPE) {
								type = tokenizer->get_token_type();

								tokenizer->advance();

								if (type == Variant::ARRAY) {
									hint_prefix += itos(Variant::ARRAY) + ":";
									if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {
										tokenizer->advance();
									}
								} else {
									hint_prefix += itos(type);
									break;
								}
							}
						}

						if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {
							// hint expected next!
							tokenizer->advance();

							switch (type) {

								case Variant::INT: {

									if (tokenizer->get_token() == GDTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "FLAGS") {

										//current_export.hint=PROPERTY_HINT_ALL_FLAGS;
										tokenizer->advance();

										if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE) {
											break;
										}
										if (tokenizer->get_token() != GDTokenizer::TK_COMMA) {
											_set_error("Expected ')' or ',' in bit flags hint.");
											return;
										}

										current_export.hint = PROPERTY_HINT_FLAGS;
										tokenizer->advance();

										bool first = true;
										while (true) {

											if (tokenizer->get_token() != GDTokenizer::TK_CONSTANT || tokenizer->get_token_constant().get_type() != Variant::STRING) {
												current_export = PropertyInfo();
												_set_error("Expected a string constant in named bit flags hint.");
												return;
											}

											String c = tokenizer->get_token_constant();
											if (!first)
												current_export.hint_string += ",";
											else
												first = false;

											current_export.hint_string += c.xml_escape();

											tokenizer->advance();
											if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE)
												break;

											if (tokenizer->get_token() != GDTokenizer::TK_COMMA) {
												current_export = PropertyInfo();
												_set_error("Expected ')' or ',' in named bit flags hint.");
												return;
											}
											tokenizer->advance();
										}

										break;
									}

									if (tokenizer->get_token() == GDTokenizer::TK_CONSTANT && tokenizer->get_token_constant().get_type() == Variant::STRING) {
										//enumeration
										current_export.hint = PROPERTY_HINT_ENUM;
										bool first = true;
										while (true) {

											if (tokenizer->get_token() != GDTokenizer::TK_CONSTANT || tokenizer->get_token_constant().get_type() != Variant::STRING) {

												current_export = PropertyInfo();
												_set_error("Expected a string constant in enumeration hint.");
												return;
											}

											String c = tokenizer->get_token_constant();
											if (!first)
												current_export.hint_string += ",";
											else
												first = false;

											current_export.hint_string += c.xml_escape();

											tokenizer->advance();
											if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE)
												break;

											if (tokenizer->get_token() != GDTokenizer::TK_COMMA) {
												current_export = PropertyInfo();
												_set_error("Expected ')' or ',' in enumeration hint.");
												return;
											}

											tokenizer->advance();
										}

										break;
									}

								}; //fallthrough to use the same
								case Variant::REAL: {

									if (tokenizer->get_token() == GDTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "EASE") {
										current_export.hint = PROPERTY_HINT_EXP_EASING;
										tokenizer->advance();
										if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {
											_set_error("Expected ')' in hint.");
											return;
										}
										break;
									}

									// range
									if (tokenizer->get_token() == GDTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "EXP") {

										current_export.hint = PROPERTY_HINT_EXP_RANGE;
										tokenizer->advance();

										if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE)
											break;
										else if (tokenizer->get_token() != GDTokenizer::TK_COMMA) {
											_set_error("Expected ')' or ',' in exponential range hint.");
											return;
										}
										tokenizer->advance();
									} else
										current_export.hint = PROPERTY_HINT_RANGE;

									float sign = 1.0;

									if (tokenizer->get_token() == GDTokenizer::TK_OP_SUB) {
										sign = -1;
										tokenizer->advance();
									}
									if (tokenizer->get_token() != GDTokenizer::TK_CONSTANT || !tokenizer->get_token_constant().is_num()) {

										current_export = PropertyInfo();
										_set_error("Expected a range in numeric hint.");
										return;
									}

									current_export.hint_string = rtos(sign * double(tokenizer->get_token_constant()));
									tokenizer->advance();

									if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE) {
										current_export.hint_string = "0," + current_export.hint_string;
										break;
									}

									if (tokenizer->get_token() != GDTokenizer::TK_COMMA) {

										current_export = PropertyInfo();
										_set_error("Expected ',' or ')' in numeric range hint.");
										return;
									}

									tokenizer->advance();

									sign = 1.0;
									if (tokenizer->get_token() == GDTokenizer::TK_OP_SUB) {
										sign = -1;
										tokenizer->advance();
									}

									if (tokenizer->get_token() != GDTokenizer::TK_CONSTANT || !tokenizer->get_token_constant().is_num()) {

										current_export = PropertyInfo();
										_set_error("Expected a number as upper bound in numeric range hint.");
										return;
									}

									current_export.hint_string += "," + rtos(sign * double(tokenizer->get_token_constant()));
									tokenizer->advance();

									if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE)
										break;

									if (tokenizer->get_token() != GDTokenizer::TK_COMMA) {

										current_export = PropertyInfo();
										_set_error("Expected ',' or ')' in numeric range hint.");
										return;
									}

									tokenizer->advance();
									sign = 1.0;
									if (tokenizer->get_token() == GDTokenizer::TK_OP_SUB) {
										sign = -1;
										tokenizer->advance();
									}

									if (tokenizer->get_token() != GDTokenizer::TK_CONSTANT || !tokenizer->get_token_constant().is_num()) {

										current_export = PropertyInfo();
										_set_error("Expected a number as step in numeric range hint.");
										return;
									}

									current_export.hint_string += "," + rtos(sign * double(tokenizer->get_token_constant()));
									tokenizer->advance();

								} break;
								case Variant::STRING: {

									if (tokenizer->get_token() == GDTokenizer::TK_CONSTANT && tokenizer->get_token_constant().get_type() == Variant::STRING) {
										//enumeration
										current_export.hint = PROPERTY_HINT_ENUM;
										bool first = true;
										while (true) {

											if (tokenizer->get_token() != GDTokenizer::TK_CONSTANT || tokenizer->get_token_constant().get_type() != Variant::STRING) {

												current_export = PropertyInfo();
												_set_error("Expected a string constant in enumeration hint.");
												return;
											}

											String c = tokenizer->get_token_constant();
											if (!first)
												current_export.hint_string += ",";
											else
												first = false;

											current_export.hint_string += c.xml_escape();
											tokenizer->advance();
											if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE)
												break;

											if (tokenizer->get_token() != GDTokenizer::TK_COMMA) {
												current_export = PropertyInfo();
												_set_error("Expected ')' or ',' in enumeration hint.");
												return;
											}
											tokenizer->advance();
										}

										break;
									}

									if (tokenizer->get_token() == GDTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "DIR") {

										tokenizer->advance();

										if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE)
											current_export.hint = PROPERTY_HINT_DIR;
										else if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {

											tokenizer->advance();

											if (tokenizer->get_token() != GDTokenizer::TK_IDENTIFIER || !(tokenizer->get_token_identifier() == "GLOBAL")) {
												_set_error("Expected 'GLOBAL' after comma in directory hint.");
												return;
											}
											if (!p_class->tool) {
												_set_error("Global filesystem hints may only be used in tool scripts.");
												return;
											}
											current_export.hint = PROPERTY_HINT_GLOBAL_DIR;
											tokenizer->advance();

											if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {
												_set_error("Expected ')' in hint.");
												return;
											}
										} else {
											_set_error("Expected ')' or ',' in hint.");
											return;
										}
										break;
									}

									if (tokenizer->get_token() == GDTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "FILE") {

										current_export.hint = PROPERTY_HINT_FILE;
										tokenizer->advance();

										if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {

											tokenizer->advance();

											if (tokenizer->get_token() == GDTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "GLOBAL") {

												if (!p_class->tool) {
													_set_error("Global filesystem hints may only be used in tool scripts.");
													return;
												}
												current_export.hint = PROPERTY_HINT_GLOBAL_FILE;
												tokenizer->advance();

												if (tokenizer->get_token() == GDTokenizer::TK_PARENTHESIS_CLOSE)
													break;
												else if (tokenizer->get_token() == GDTokenizer::TK_COMMA)
													tokenizer->advance();
												else {
													_set_error("Expected ')' or ',' in hint.");
													return;
												}
											}

											if (tokenizer->get_token() != GDTokenizer::TK_CONSTANT || tokenizer->get_token_constant().get_type() != Variant::STRING) {

												if (current_export.hint == PROPERTY_HINT_GLOBAL_FILE)
													_set_error("Expected string constant with filter");
												else
													_set_error("Expected 'GLOBAL' or string constant with filter");
												return;
											}
											current_export.hint_string = tokenizer->get_token_constant();
											tokenizer->advance();
										}

										if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {
											_set_error("Expected ')' in hint.");
											return;
										}
										break;
									}

									if (tokenizer->get_token() == GDTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "MULTILINE") {

										current_export.hint = PROPERTY_HINT_MULTILINE_TEXT;
										tokenizer->advance();
										if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {
											_set_error("Expected ')' in hint.");
											return;
										}
										break;
									}
								} break;
								case Variant::COLOR: {

									if (tokenizer->get_token() != GDTokenizer::TK_IDENTIFIER) {

										current_export = PropertyInfo();
										_set_error("Color type hint expects RGB or RGBA as hints");
										return;
									}

									String identifier = tokenizer->get_token_identifier();
									if (identifier == "RGB") {
										current_export.hint = PROPERTY_HINT_COLOR_NO_ALPHA;
									} else if (identifier == "RGBA") {
										//none
									} else {
										current_export = PropertyInfo();
										_set_error("Color type hint expects RGB or RGBA as hints");
										return;
									}
									tokenizer->advance();

								} break;
								default: {

									current_export = PropertyInfo();
									_set_error("Type '" + Variant::get_type_name(type) + "' can't take hints.");
									return;
								} break;
							}
						}
						if (current_export.type == Variant::ARRAY && !hint_prefix.empty()) {
							if (current_export.hint) {
								hint_prefix += "/" + itos(current_export.hint);
							}
							current_export.hint_string = hint_prefix + ":" + current_export.hint_string;
							current_export.hint = PROPERTY_HINT_NONE;
						}

					} else if (tokenizer->get_token() == GDTokenizer::TK_IDENTIFIER) {

						String identifier = tokenizer->get_token_identifier();
						if (!ClassDB::is_parent_class(identifier, "Resource")) {

							current_export = PropertyInfo();
							_set_error("Export hint not a type or resource.");
						}

						current_export.type = Variant::OBJECT;
						current_export.hint = PROPERTY_HINT_RESOURCE_TYPE;
						current_export.usage |= PROPERTY_USAGE_SCRIPT_VARIABLE;

						current_export.hint_string = identifier;

						tokenizer->advance();
					}

					if (tokenizer->get_token() != GDTokenizer::TK_PARENTHESIS_CLOSE) {

						current_export = PropertyInfo();
						_set_error("Expected ')' or ',' after export hint.");
						return;
					}

					tokenizer->advance();
				}

				if (tokenizer->get_token() != GDTokenizer::TK_PR_VAR && tokenizer->get_token() != GDTokenizer::TK_PR_ONREADY && tokenizer->get_token() != GDTokenizer::TK_PR_REMOTE && tokenizer->get_token() != GDTokenizer::TK_PR_MASTER && tokenizer->get_token() != GDTokenizer::TK_PR_SLAVE && tokenizer->get_token() != GDTokenizer::TK_PR_SYNC) {

					current_export = PropertyInfo();
					_set_error("Expected 'var', 'onready', 'remote', 'master', 'slave' or 'sync'.");
					return;
				}

				continue;
			} break;
			case GDTokenizer::TK_PR_ONREADY: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (tokenizer->get_token() != GDTokenizer::TK_PR_VAR) {
					_set_error("Expected 'var'.");
					return;
				}

				continue;
			} break;
			case GDTokenizer::TK_PR_REMOTE: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (current_export.type) {
					if (tokenizer->get_token() != GDTokenizer::TK_PR_VAR) {
						_set_error("Expected 'var'.");
						return;
					}

				} else {
					if (tokenizer->get_token() != GDTokenizer::TK_PR_VAR && tokenizer->get_token() != GDTokenizer::TK_PR_FUNCTION) {
						_set_error("Expected 'var' or 'func'.");
						return;
					}
				}
				rpc_mode = ScriptInstance::RPC_MODE_REMOTE;

				continue;
			} break;
			case GDTokenizer::TK_PR_MASTER: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (current_export.type) {
					if (tokenizer->get_token() != GDTokenizer::TK_PR_VAR) {
						_set_error("Expected 'var'.");
						return;
					}

				} else {
					if (tokenizer->get_token() != GDTokenizer::TK_PR_VAR && tokenizer->get_token() != GDTokenizer::TK_PR_FUNCTION) {
						_set_error("Expected 'var' or 'func'.");
						return;
					}
				}

				rpc_mode = ScriptInstance::RPC_MODE_MASTER;
				continue;
			} break;
			case GDTokenizer::TK_PR_SLAVE: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (current_export.type) {
					if (tokenizer->get_token() != GDTokenizer::TK_PR_VAR) {
						_set_error("Expected 'var'.");
						return;
					}

				} else {
					if (tokenizer->get_token() != GDTokenizer::TK_PR_VAR && tokenizer->get_token() != GDTokenizer::TK_PR_FUNCTION) {
						_set_error("Expected 'var' or 'func'.");
						return;
					}
				}

				rpc_mode = ScriptInstance::RPC_MODE_SLAVE;
				continue;
			} break;
			case GDTokenizer::TK_PR_SYNC: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (tokenizer->get_token() != GDTokenizer::TK_PR_VAR && tokenizer->get_token() != GDTokenizer::TK_PR_FUNCTION) {
					if (current_export.type)
						_set_error("Expected 'var'.");
					else
						_set_error("Expected 'var' or 'func'.");
					return;
				}

				rpc_mode = ScriptInstance::RPC_MODE_SYNC;
				continue;
			} break;
			case GDTokenizer::TK_PR_VAR: {
				//variale declaration and (eventual) initialization

				ClassNode::Member member;
				bool autoexport = tokenizer->get_token(-1) == GDTokenizer::TK_PR_EXPORT;
				if (current_export.type != Variant::NIL) {
					member._export = current_export;
					current_export = PropertyInfo();
				}

				bool onready = tokenizer->get_token(-1) == GDTokenizer::TK_PR_ONREADY;

				tokenizer->advance();
				if (!tokenizer->is_token_literal(0, true)) {

					_set_error("Expected identifier for member variable name.");
					return;
				}

				member.identifier = tokenizer->get_token_literal();
				member.expression = NULL;
				member._export.name = member.identifier;
				member.line = tokenizer->get_token_line();
				member.rpc_mode = rpc_mode;

				tokenizer->advance();

				rpc_mode = ScriptInstance::RPC_MODE_DISABLED;

				if (tokenizer->get_token() == GDTokenizer::TK_OP_ASSIGN) {

#ifdef DEBUG_ENABLED
					int line = tokenizer->get_token_line();
#endif
					tokenizer->advance();

					Node *subexpr = _parse_and_reduce_expression(p_class, false, autoexport);
					if (!subexpr) {
						if (_recover_from_completion()) {
							break;
						}
						return;
					}

					//discourage common error
					if (!onready && subexpr->type == Node::TYPE_OPERATOR) {

						OperatorNode *op = static_cast<OperatorNode *>(subexpr);
						if (op->op == OperatorNode::OP_CALL && op->arguments[0]->type == Node::TYPE_SELF && op->arguments[1]->type == Node::TYPE_IDENTIFIER) {
							IdentifierNode *id = static_cast<IdentifierNode *>(op->arguments[1]);
							if (id->name == "get_node") {

								_set_error("Use 'onready var " + String(member.identifier) + " = get_node(..)' instead");
								return;
							}
						}
					}

					member.expression = subexpr;

					if (autoexport) {
						if (1) /*(subexpr->type==Node::TYPE_ARRAY) {

							member._export.type=Variant::ARRAY;

						} else if (subexpr->type==Node::TYPE_DICTIONARY) {

							member._export.type=Variant::DICTIONARY;

						} else*/ {

							if (subexpr->type != Node::TYPE_CONSTANT) {

								_set_error("Type-less export needs a constant expression assigned to infer type.");
								return;
							}

							ConstantNode *cn = static_cast<ConstantNode *>(subexpr);
							if (cn->value.get_type() == Variant::NIL) {

								_set_error("Can't accept a null constant expression for infering export type.");
								return;
							}
							member._export.type = cn->value.get_type();
							member._export.usage |= PROPERTY_USAGE_SCRIPT_VARIABLE;
							if (cn->value.get_type() == Variant::OBJECT) {
								Object *obj = cn->value;
								Resource *res = Object::cast_to<Resource>(obj);
								if (res == NULL) {
									_set_error("Exported constant not a type or resource.");
									return;
								}
								member._export.hint = PROPERTY_HINT_RESOURCE_TYPE;
								member._export.hint_string = res->get_class();
							}
						}
					}
#ifdef TOOLS_ENABLED
					if (subexpr->type == Node::TYPE_CONSTANT && member._export.type != Variant::NIL) {

						ConstantNode *cn = static_cast<ConstantNode *>(subexpr);
						if (cn->value.get_type() != Variant::NIL) {
							member.default_value = cn->value;
						}
					}
#endif

					IdentifierNode *id = alloc_node<IdentifierNode>();
					id->name = member.identifier;

					OperatorNode *op = alloc_node<OperatorNode>();
					op->op = OperatorNode::OP_INIT_ASSIGN;
					op->arguments.push_back(id);
					op->arguments.push_back(subexpr);

#ifdef DEBUG_ENABLED
					NewLineNode *nl = alloc_node<NewLineNode>();
					nl->line = line;
					if (onready)
						p_class->ready->statements.push_back(nl);
					else
						p_class->initializer->statements.push_back(nl);
#endif
					if (onready)
						p_class->ready->statements.push_back(op);
					else
						p_class->initializer->statements.push_back(op);

				} else {

					if (autoexport) {

						_set_error("Type-less export needs a constant expression assigned to infer type.");
						return;
					}
				}

				if (tokenizer->get_token() == GDTokenizer::TK_PR_SETGET) {

					tokenizer->advance();

					if (tokenizer->get_token() != GDTokenizer::TK_COMMA) {
						//just comma means using only getter
						if (!tokenizer->is_token_literal()) {
							_set_error("Expected identifier for setter function after 'setget'.");
						}

						member.setter = tokenizer->get_token_literal();

						tokenizer->advance();
					}

					if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {
						//there is a getter
						tokenizer->advance();

						if (!tokenizer->is_token_literal()) {
							_set_error("Expected identifier for getter function after ','.");
						}

						member.getter = tokenizer->get_token_literal();
						tokenizer->advance();
					}
				}

				p_class->variables.push_back(member);

				if (!_end_statement()) {
					_set_error("Expected end of statement (continue)");
					return;
				}
			} break;
			case GDTokenizer::TK_PR_CONST: {
				//variale declaration and (eventual) initialization

				ClassNode::Constant constant;

				tokenizer->advance();
				if (!tokenizer->is_token_literal(0, true)) {

					_set_error("Expected name (identifier) for constant.");
					return;
				}

				constant.identifier = tokenizer->get_token_literal();
				tokenizer->advance();

				if (tokenizer->get_token() != GDTokenizer::TK_OP_ASSIGN) {
					_set_error("Constant expects assignment.");
					return;
				}

				tokenizer->advance();

				Node *subexpr = _parse_and_reduce_expression(p_class, true, true);
				if (!subexpr) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}

				if (subexpr->type != Node::TYPE_CONSTANT) {
					_set_error("Expected constant expression");
				}
				constant.expression = subexpr;

				p_class->constant_expressions.push_back(constant);

				if (!_end_statement()) {
					_set_error("Expected end of statement (constant)");
					return;
				}

			} break;
			case GDTokenizer::TK_PR_ENUM: {
				//mutiple constant declarations..

				int last_assign = -1; // Incremented by 1 right before the assingment.
				String enum_name;
				Dictionary enum_dict;

				tokenizer->advance();
				if (tokenizer->is_token_literal(0, true)) {
					enum_name = tokenizer->get_token_literal();
					tokenizer->advance();
				}
				if (tokenizer->get_token() != GDTokenizer::TK_CURLY_BRACKET_OPEN) {
					_set_error("Expected '{' in enum declaration");
					return;
				}
				tokenizer->advance();

				while (true) {
					if (tokenizer->get_token() == GDTokenizer::TK_NEWLINE) {

						tokenizer->advance(); // Ignore newlines
					} else if (tokenizer->get_token() == GDTokenizer::TK_CURLY_BRACKET_CLOSE) {

						tokenizer->advance();
						break; // End of enum
					} else if (!tokenizer->is_token_literal(0, true)) {

						if (tokenizer->get_token() == GDTokenizer::TK_EOF) {
							_set_error("Unexpected end of file.");
						} else {
							_set_error(String("Unexpected ") + GDTokenizer::get_token_name(tokenizer->get_token()) + ", expected identifier");
						}

						return;
					} else { // tokenizer->is_token_literal(0, true)
						ClassNode::Constant constant;

						constant.identifier = tokenizer->get_token_literal();

						tokenizer->advance();

						if (tokenizer->get_token() == GDTokenizer::TK_OP_ASSIGN) {
							tokenizer->advance();

							Node *subexpr = _parse_and_reduce_expression(p_class, true, true);
							if (!subexpr) {
								if (_recover_from_completion()) {
									break;
								}
								return;
							}

							if (subexpr->type != Node::TYPE_CONSTANT) {
								_set_error("Expected constant expression");
							}

							const ConstantNode *subexpr_const = static_cast<const ConstantNode *>(subexpr);

							if (subexpr_const->value.get_type() != Variant::INT) {
								_set_error("Expected an int value for enum");
							}

							last_assign = subexpr_const->value;

							constant.expression = subexpr;

						} else {
							last_assign = last_assign + 1;
							ConstantNode *cn = alloc_node<ConstantNode>();
							cn->value = last_assign;
							constant.expression = cn;
						}

						if (tokenizer->get_token() == GDTokenizer::TK_COMMA) {
							tokenizer->advance();
						}

						if (enum_name != "") {
							const ConstantNode *cn = static_cast<const ConstantNode *>(constant.expression);
							enum_dict[constant.identifier] = cn->value;
						}

						p_class->constant_expressions.push_back(constant);
					}
				}

				if (enum_name != "") {
					ClassNode::Constant enum_constant;
					enum_constant.identifier = enum_name;
					ConstantNode *cn = alloc_node<ConstantNode>();
					cn->value = enum_dict;
					enum_constant.expression = cn;
					p_class->constant_expressions.push_back(enum_constant);
				}

				if (!_end_statement()) {
					_set_error("Expected end of statement (enum)");
					return;
				}

			} break;

			case GDTokenizer::TK_CONSTANT: {
				if (tokenizer->get_token_constant().get_type() == Variant::STRING) {
					tokenizer->advance();
					// Ignore
				} else {
					_set_error(String() + "Unexpected constant of type: " + Variant::get_type_name(tokenizer->get_token_constant().get_type()));
					return;
				}
			} break;

			default: {

				_set_error(String() + "Unexpected token: " + tokenizer->get_token_name(tokenizer->get_token()) + ":" + tokenizer->get_token_identifier());
				return;

			} break;
		}
	}
}

void GDParser::_set_error(const String &p_error, int p_line, int p_column) {

	if (error_set)
		return; //allow no further errors

	error = p_error;
	error_line = p_line < 0 ? tokenizer->get_token_line() : p_line;
	error_column = p_column < 0 ? tokenizer->get_token_column() : p_column;
	error_set = true;
}

String GDParser::get_error() const {

	return error;
}

int GDParser::get_error_line() const {

	return error_line;
}
int GDParser::get_error_column() const {

	return error_column;
}

Error GDParser::_parse(const String &p_base_path) {

	base_path = p_base_path;

	clear();

	//assume class
	ClassNode *main_class = alloc_node<ClassNode>();
	main_class->initializer = alloc_node<BlockNode>();
	main_class->initializer->parent_class = main_class;
	main_class->ready = alloc_node<BlockNode>();
	main_class->ready->parent_class = main_class;
	current_class = main_class;

	_parse_class(main_class);

	if (tokenizer->get_token() == GDTokenizer::TK_ERROR) {
		error_set = false;
		_set_error("Parse Error: " + tokenizer->get_token_error());
	}

	if (error_set) {

		return ERR_PARSE_ERROR;
	}
	return OK;
}

Error GDParser::parse_bytecode(const Vector<uint8_t> &p_bytecode, const String &p_base_path, const String &p_self_path) {

	for_completion = false;
	validating = false;
	completion_type = COMPLETION_NONE;
	completion_node = NULL;
	completion_class = NULL;
	completion_function = NULL;
	completion_block = NULL;
	completion_found = false;
	current_block = NULL;
	current_class = NULL;
	current_function = NULL;

	self_path = p_self_path;
	GDTokenizerBuffer *tb = memnew(GDTokenizerBuffer);
	tb->set_code_buffer(p_bytecode);
	tokenizer = tb;
	Error ret = _parse(p_base_path);
	memdelete(tb);
	tokenizer = NULL;
	return ret;
}

Error GDParser::parse(const String &p_code, const String &p_base_path, bool p_just_validate, const String &p_self_path, bool p_for_completion) {

	completion_type = COMPLETION_NONE;
	completion_node = NULL;
	completion_class = NULL;
	completion_function = NULL;
	completion_block = NULL;
	completion_found = false;
	current_block = NULL;
	current_class = NULL;

	current_function = NULL;

	self_path = p_self_path;
	GDTokenizerText *tt = memnew(GDTokenizerText);
	tt->set_code(p_code);

	validating = p_just_validate;
	for_completion = p_for_completion;
	tokenizer = tt;
	Error ret = _parse(p_base_path);
	memdelete(tt);
	tokenizer = NULL;
	return ret;
}

bool GDParser::is_tool_script() const {

	return (head && head->type == Node::TYPE_CLASS && static_cast<const ClassNode *>(head)->tool);
}

const GDParser::Node *GDParser::get_parse_tree() const {

	return head;
}

void GDParser::clear() {

	while (list) {

		Node *l = list;
		list = list->next;
		memdelete(l);
	}

	head = NULL;
	list = NULL;

	completion_type = COMPLETION_NONE;
	completion_node = NULL;
	completion_class = NULL;
	completion_function = NULL;
	completion_block = NULL;
	current_block = NULL;
	current_class = NULL;

	completion_found = false;
	rpc_mode = ScriptInstance::RPC_MODE_DISABLED;

	current_function = NULL;

	validating = false;
	for_completion = false;
	error_set = false;
	tab_level.clear();
	tab_level.push_back(0);
	error_line = 0;
	error_column = 0;
	pending_newline = -1;
	parenthesis = 0;
	current_export.type = Variant::NIL;
	error = "";
}

GDParser::CompletionType GDParser::get_completion_type() {

	return completion_type;
}

StringName GDParser::get_completion_cursor() {

	return completion_cursor;
}

int GDParser::get_completion_line() {

	return completion_line;
}

Variant::Type GDParser::get_completion_built_in_constant() {

	return completion_built_in_constant;
}

GDParser::Node *GDParser::get_completion_node() {

	return completion_node;
}

GDParser::BlockNode *GDParser::get_completion_block() {

	return completion_block;
}

GDParser::ClassNode *GDParser::get_completion_class() {

	return completion_class;
}

GDParser::FunctionNode *GDParser::get_completion_function() {

	return completion_function;
}

int GDParser::get_completion_argument_index() {

	return completion_argument;
}

int GDParser::get_completion_identifier_is_function() {

	return completion_ident_is_call;
}

GDParser::GDParser() {

	head = NULL;
	list = NULL;
	tokenizer = NULL;
	pending_newline = -1;
	clear();
}

GDParser::~GDParser() {

	clear();
}
