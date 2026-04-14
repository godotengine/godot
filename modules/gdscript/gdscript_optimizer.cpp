/**************************************************************************/
/*  gdscript_optimizer.cpp                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "gdscript_optimizer.h"
#include "gdscript_reconstructor.h"

struct GDScriptOptimizer::LocalOptions GDScriptOptimizer::local_options;
struct GDScriptOptimizer::GlobalOptions GDScriptOptimizer::global_options;
bool GDScriptOptimizer::Data::log_script_name_pending = true;

// Whether the optimizer will only optimize a file if the keyword `inline` or `unroll` are present.
//#define GDSCRIPT_OPTIMIZER_REQUIRE_KEYWORDS

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#ifdef TOOLS_ENABLED
#if 1
#define GDOPT_LOG(LEVEL, TEXT)                                                            \
	if (global_options.logging_level >= LEVEL) {                                          \
		helper_flush_pending_script_name_log();                                           \
		print_line(GDScriptReconstructor::draw_highlight("GDScriptOptimizer : ") + TEXT); \
	}
#else
#define GDOPT_LOG(LEVEL, TEXT)
#endif
#else
#define GDOPT_LOG(LEVEL, TEXT)
#endif

// This define enables us to use less verbose  forms of the GDScriptParser nodes,
// without the GDScriptParser:: scope.
// We do however need to disable warnings for unused local typedefs to prevent
// useless compiler warnings.
#define USING_GDSCRIPTPARSER                                 \
	using Node = GDScriptParser::Node;                       \
	using FunctionNode = GDScriptParser::FunctionNode;       \
	using BlockNode = GDScriptParser::BlockNode;             \
	using ControlFlowNode = GDScriptParser::ControlFlowNode; \
	using OperatorNode = GDScriptParser::OperatorNode;       \
	using LocalVarNode = GDScriptParser::LocalVarNode;       \
	using IdentifierNode = GDScriptParser::IdentifierNode;   \
	using NewLineNode = GDScriptParser::NewLineNode;         \
	using ClassNode = GDScriptParser::ClassNode;             \
	using ConstantNode = GDScriptParser::ConstantNode;       \
	using ArrayNode = GDScriptParser::ArrayNode;             \
	using InlineBlockNode = GDScriptParser::InlineBlockNode;

#define GDOPT_ALLOC_NODE(TYPE, NAME, LOCATION_COPY_NODE) GDScriptParser::TYPE *NAME = alloc_node<GDScriptParser::TYPE>(LOCATION_COPY_NODE)

template <class T>
T *GDScriptOptimizer::alloc_node(const GDScriptParser::Node *p_location) const {
	T *t = memnew(T);

	t->next = data.parser->list;
	data.parser->list = t;

	if (!data.parser->head) {
		data.parser->head = t;
	}

	if (p_location) {
		t->line = p_location->line;
		t->column = p_location->column;
	} else {
		t->line = 0;
		t->column = 0;
	}
	return t;
}

GDScriptOptimizer::ASTNode::~ASTNode() {
	for (uint32_t n = 0; n < children.size(); n++) {
		memdelete(children[n]);
	}
}

uint32_t GDScriptOptimizer::ASTNode::count_nodes() const {
	uint32_t count = 1;
	for (uint32_t n = 0; n < children.size(); n++) {
		count += children[n]->count_nodes();
	}
	return count;
}

void GDScriptOptimizer::ASTNode::debug_print(uint32_t p_depth) const {
	print_line(String("\t").repeat(p_depth) + node_to_string(node));
	if (node->type == GDScriptParser::Node::TYPE_BLOCK) {
		print_line("");
	}
	for (uint32_t n = 0; n < children.size(); n++) {
		children[n]->debug_print(p_depth + 1);
	}
}

uint32_t GDScriptOptimizer::ASTNode::find_child_id(GDScriptParser::Node *p_node) const {
	for (uint32_t n = 0; n < children.size(); n++) {
		if (children[n]->node == p_node) {
			return n;
		}
	}
	return UINT32_MAX;
}

GDScriptOptimizer::ASTNode::ASTNode(GDScriptParser::Node *p_node, ASTNode *p_parent) {
	USING_GDSCRIPTPARSER

	node = p_node;
	parent = p_parent;

	DEV_ASSERT(node);

	switch (node->type) {
		default: {
		} break;
		// Note that for TYPE_LOCAL_VAR,
		// the assignment operator is handled separately as a separate node following in the statements,
		// NOT as part of the LocalVarNode.
		// This is a peculiarity of the parse tree.
		case Node::TYPE_ARRAY: {
			ArrayNode *arr = (ArrayNode *)p_node;
			for (int n = 0; n < arr->elements.size(); n++) {
				children.push_back(memnew(ASTNode(arr->elements[n], this)));
			}
		} break;
		case Node::TYPE_CLASS: {
			ClassNode *cl = (ClassNode *)p_node;
			for (int n = 0; n < cl->subclasses.size(); n++) {
				children.push_back(memnew(ASTNode(cl->subclasses[n], this)));
			}
			for (int n = 0; n < cl->functions.size(); n++) {
				children.push_back(memnew(ASTNode(cl->functions[n], this)));
			}
			for (int n = 0; n < cl->static_functions.size(); n++) {
				children.push_back(memnew(ASTNode(cl->static_functions[n], this)));
			}
		} break;
		case Node::TYPE_FUNCTION: {
			FunctionNode *func = (FunctionNode *)p_node;
			if (func->body) {
				children.push_back(memnew(ASTNode(func->body, this)));
			}
		} break;
		case Node::TYPE_BLOCK: {
			BlockNode *block = (BlockNode *)p_node;
			for (int n = 0; n < block->statements.size(); n++) {
				children.push_back(memnew(ASTNode(block->statements[n], this)));
			}

		} break;
		case Node::TYPE_INLINE_BLOCK: {
			InlineBlockNode *block = (InlineBlockNode *)p_node;
			for (int n = 0; n < block->statements.size(); n++) {
				children.push_back(memnew(ASTNode(block->statements[n], this)));
			}

		} break;
		case Node::TYPE_CONTROL_FLOW: {
			const ControlFlowNode *cf = (const ControlFlowNode *)p_node;
			for (int n = 0; n < cf->arguments.size(); n++) {
				DEV_ASSERT(cf->arguments[n]);
				children.push_back(memnew(ASTNode(cf->arguments[n], this)));
			}
			if (cf->body) {
				children.push_back(memnew(ASTNode(cf->body, this)));
			}
			if (cf->body_else) {
				children.push_back(memnew(ASTNode(cf->body_else, this)));
			}
			if (cf->match) {
				children.push_back(memnew(ASTNode(cf->match->val_to_match, this)));

				for (int n = 0; n < cf->match->branches.size(); n++) {
					GDScriptParser::PatternBranchNode *branch = cf->match->branches[n];

					for (int p = 0; p < branch->patterns.size(); p++) {
						GDScriptParser::PatternNode *pat = branch->patterns[p];

						if (pat->pt_type == GDScriptParser::PatternNode::PT_CONSTANT) {
							children.push_back(memnew(ASTNode(pat->constant, this)));
						}
					}

					children.push_back(memnew(ASTNode(branch->body, this)));
				}
			}
		} break;
		case Node::TYPE_OPERATOR: {
			const OperatorNode *op = (const OperatorNode *)p_node;
			for (int n = 0; n < op->arguments.size(); n++) {
				children.push_back(memnew(ASTNode(op->arguments[n], this)));
			}
		}
	}
}

GDScriptParser::Node *GDScriptOptimizer::Chain::find_call_parent() const {
	if (_count < 2) {
		return nullptr;
	}
	return nodes[_count - 2];
}

GDScriptParser::BlockNode *GDScriptOptimizer::Chain::find_enclosing_block(GDScriptParser::Node *&r_first_child) const {
	USING_GDSCRIPTPARSER

	if (!_count) {
		return nullptr;
	}

	Node *prev = nullptr;

	for (int32_t i = _count - 1; i >= 0; i--) {
		Node *node = nodes[i];
		if (node->type == Node::TYPE_BLOCK) {
			if (prev) {
				BlockNode *block = (BlockNode *)node;
				r_first_child = prev;
				return block;
			} else {
				return nullptr;
			}
		}

		prev = node;
	}

	return nullptr;
}

bool GDScriptOptimizer::inline_search(LocalVector<Call> &r_calls, GDScriptParser::Node *p_node, Chain p_chain, GDScriptParser::FunctionNode *p_parent_func, bool p_caller_requires_return, bool p_const_inlines_only) {
	USING_GDSCRIPTPARSER

	if (!p_node) {
		return false;
	}

#if 0
	print_line(String("\t").repeat(p_chain.length()) + String("inline_search ") + (p_const_inlines_only ? "CONST " : "") + node_to_string(p_node));
#endif

	p_chain.push_back(p_node);
	bool found_any = false;

	switch (p_node->type) {
		default: {
			return false;
		} break;
		case Node::TYPE_LOCAL_VAR: {
			// For local var, the assign expression will be checked in the following statement,
			// so no need to check the assign operator here directly.
#if 0
			LocalVarNode *var = (LocalVarNode *)p_node;
			inline_search(r_calls, var->assign, p_chain, p_parent_func, true);
#endif
			return false;
		} break;
		case Node::TYPE_BLOCK: {
			BlockNode *block = (BlockNode *)p_node;
			Chain chain(p_node);

			for (int n = 0; n < block->statements.size(); n++) {
				// Statements are the only containers that allow inlines with no return value.
				// Any other case is a USER SCRIPT ERROR.
				if (inline_search(r_calls, block->statements[n], chain, p_parent_func, false)) {
					found_any = true;
				}
			}
			return found_any;
		} break;
		case Node::TYPE_CONTROL_FLOW: {
			const ControlFlowNode *cf = (const ControlFlowNode *)p_node;

			// Note that with e.g. IF statements, we can't evaluate later arguments BEFORE
			// earlier arguments have been checked.
			// We handle this in the TYPE_OPERATOR though rather than TYPE_CONTROL_FLOW.
			for (int n = 0; n < cf->arguments.size(); n++) {
				if (inline_search(r_calls, cf->arguments[n], p_chain, p_parent_func, true)) {
					found_any = true;
				}
			}
			if (inline_search(r_calls, cf->body, p_chain, p_parent_func, true)) {
				found_any = true;
			}
			if (inline_search(r_calls, cf->body_else, p_chain, p_parent_func, true)) {
				found_any = true;
			}
			return found_any;
		} break;
		case Node::TYPE_OPERATOR: {
			const OperatorNode *op = (const OperatorNode *)p_node;

			// We can safetly inline any function for the first function in an operator chain, but we *cannot*
			// safely inline anything but const functions after this, because the first function may have side effects
			// that affect later function calls, which will be a potential problem if the inlines are evalulated before
			// the start of the operators.

			// Some care here, with e.g. AND operators, we cannot evaluate the second operand
			// BEFORE the first operand has been shown to be true, because it might be an expression like:
			// if (pointer && pointer->is_true())
			switch (op->op) {
				default: {
					for (int n = 0; n < op->arguments.size(); n++) {
						if (inline_search(r_calls, op->arguments[n], p_chain, p_parent_func, true, found_any || p_const_inlines_only)) {
							found_any = true;
						}
					}
				} break;
				case OperatorNode::OP_AND: {
					DEV_ASSERT(op->arguments.size());
					if (inline_search(r_calls, op->arguments[0], p_chain, p_parent_func, true)) {
						found_any = true;
					}
				} break;
				// Not supported.
				case OperatorNode::OP_TERNARY_IF:
				case OperatorNode::OP_TERNARY_ELSE:
				case OperatorNode::OP_IN: {
				} break;
			}
		} break;
	}

	// Only interested in operator calls.
	OperatorNode *op = static_cast<OperatorNode *>(p_node);
	if (op->op != OperatorNode::OP_CALL) {
		return found_any;
	}

	// Only interested in self calls and strongly typed classes.
	GDScriptParser::Node *arg0 = op->arguments[0];

	Call call;

	if ((op->arguments.size() < 2) || op->arguments[1]->type != Node::TYPE_IDENTIFIER) {
		return found_any;
	}

	const IdentifierNode *func_identifier = (const IdentifierNode *)op->arguments[1];
	const StringName &func_name = func_identifier->name;

	if (arg0->type != Node::TYPE_SELF) {
		if ((arg0->get_datatype().kind == GDScriptParser::DataType::CLASS) && arg0->get_datatype().class_type) {
			call.gdclass_id = data.find_class_by_name(arg0->get_datatype().class_type->name);
			if (call.gdclass_id == UINT32_MAX) {
				// Class not found in this file, not supported.
				return found_any;
			}

			call.function_id = data.classes[call.gdclass_id].find_function_by_name(func_name);
			call.self = arg0;
		} else {
			return found_any;
		}
	} else {
		// Only add if there is such an inline function available.
		call.function_id = data.get_current_class().find_function_by_name(func_name);
		call.gdclass_id = 0;
	}

	if (call.function_id == UINT32_MAX) {
		return found_any;
	}

	// Determine whether the inlined function needs to be const.
	if (p_const_inlines_only) {
		const Function &func = data.classes[call.gdclass_id].functions[call.function_id];
		if (!func.is_const_function) {
			goto failed;
		}
	}

	// Don't allow recursive inlines.
	if (func_name == p_parent_func->name) {
		goto failed;
	} else {
		// Check for USER script errors.
		// Don't allow the system to attempt to inline functions with no return value into operators that require
		// a return value, this would cause complications to error handling later on in optimization,
		// best to avoid and let the standard compiler report the error.
		if (p_caller_requires_return && (data.classes[call.gdclass_id].functions[call.function_id].num_return_values == 0)) {
			GDOPT_LOG(1, "SCRIPT ERROR : " + node_to_string(op) + " to \"" + func_name + "\" requires return value. ");
			goto failed;
		}

		// Don't allow the same call node to feature in the call list more than once.
		for (uint32_t n = 0; n < r_calls.size(); n++) {
			Call &dcall = r_calls[n];
			if (dcall.call_node == op) {
				// Duplicate. Not an error, may be referenced in e.g. an operator and also a statement.
				dcall.call_parent = p_chain.find_call_parent();
				// GDOPT_LOG(2, "Identified duplicate call " + name);
				return found_any;
			}
		}

		call.call_node = op;
		call.parent_block = p_chain.find_enclosing_block(call.parent_block_first_child);
		call.call_parent = p_chain.find_call_parent();
		call.call_identifier = func_identifier;
		call.call_container_function = p_parent_func;

		if (call.parent_block) {
			r_calls.push_back(call);
		}
	}

	return true;
failed:
	// Keep a record that we couldn't inline, we mustn't remove the inline from the source code.
	data.classes[call.gdclass_id].functions[call.function_id].non_inline_calls += 1;
	return found_any;
}

void GDScriptOptimizer::inline_search(LocalVector<Call> &r_calls, GDScriptParser::FunctionNode *p_func) {
	DEV_ASSERT(p_func);
	GDScriptParser::BlockNode *body = p_func->body;
	ERR_FAIL_NULL(body);

	for (int n = 0; n < body->statements.size(); n++) {
		inline_search(r_calls, body->statements[n], body, p_func, false);
	}
}

void GDScriptOptimizer::inline_setup_class_functions() {
	GDClass &cl = data.get_current_class();

	cl.functions.resize(cl.root->functions.size());

	for (int n = 0; n < cl.root->functions.size(); n++) {
		cl.functions[n].node = cl.root->functions[n];
		function_register_and_check_inline_support(n);
	}

	// Delete any that that are unsupported.
	for (uint32_t n = 0; n < cl.functions.size(); n++) {
		if (!cl.functions[n].supported) {
			cl.functions.remove(n);
			n--;
		}
	}
}

void GDScriptOptimizer::inline_search(LocalVector<Call> &r_calls) {
	USING_GDSCRIPTPARSER

	GDClass &cl = data.get_current_class();

	for (int n = 0; n < cl.root->functions.size(); n++) {
		inline_search(r_calls, cl.root->functions[n]);
	}
}

bool GDScriptOptimizer::_helper_statements_contain_return(const Vector<GDScriptParser::Node *> &p_statements, bool p_within_control_flow) {
	USING_GDSCRIPTPARSER

	for (int n = 0; n < p_statements.size(); n++) {
		if (p_statements[n]->type == Node::TYPE_INLINE_BLOCK) {
			if (_helper_statements_contain_return(((InlineBlockNode *)p_statements[n])->statements, p_within_control_flow)) {
				return true;
			}
		}

		if (p_statements[n]->type == Node::TYPE_CONTROL_FLOW) {
			ControlFlowNode *cf = (ControlFlowNode *)p_statements[n];

			if (cf->cf_type == ControlFlowNode::CF_RETURN) {
				return true;
			}

			// Call recursive for IFs.
			if (!p_within_control_flow && cf->cf_type == ControlFlowNode::CF_IF) {
				int child_count = 0;
				int return_count = 0;

				if (cf->body) {
					child_count++;
					if (helper_block_contains_return(cf->body)) {
						return_count++;
					}
				}

				if (cf->body_else) {
					child_count++;
					if (helper_block_contains_return(cf->body_else)) {
						return_count++;
					}
				}

				if ((child_count == 2) && (return_count == 2)) {
					return true;
				}
				return false;
			} else {
				// Always check children for returns for non-if control flows,
				// as this is not yet supported.

				if (helper_block_contains_return(cf->body, true)) {
					return true;
				}

				if (helper_block_contains_return(cf->body_else, true)) {
					return true;
				}
			}
		}
	}

	return false;
}

bool GDScriptOptimizer::helper_block_contains_return(const GDScriptParser::BlockNode *p_block, bool p_within_control_flow) {
	if (!p_block) {
		return false;
	}

	return _helper_statements_contain_return(p_block->statements, p_within_control_flow);
}

void GDScriptOptimizer::function_param_mark_use(GDScriptOptimizer::Function &p_func_decl, const GDScriptParser::Node *p_node, bool p_assigned) {
	if (!p_node || p_node->type != GDScriptParser::Node::TYPE_IDENTIFIER) {
		return;
	}

	uint32_t param_id = p_func_decl.find_param_by_name(((GDScriptParser::IdentifierNode *)p_node)->name);
	if (param_id != UINT32_MAX) {
		p_func_decl.params[param_id].is_used = true;
		if (p_assigned) {
			p_func_decl.params[param_id].is_written = true;
		}
	}
}

void GDScriptOptimizer::function_register_and_check_inline_support(int p_func_id) {
	USING_GDSCRIPTPARSER

	Function &func_decl = data.get_current_class().functions[p_func_id];

	// Build a local AST tree to analyze this function.
	ASTNode *tree = memnew(ASTNode(func_decl.node, nullptr));

	// Traverse the AST tree using a lambda to extract useful info about the function.
	tree->traverse([&func_decl](ASTNode *p_node, uint32_t p_depth, bool &p_user_flag) {
		switch (p_node->node->type) {
			default:
				break;
			case Node::TYPE_FUNCTION: {
				GDScriptParser::FunctionNode *func = (FunctionNode *)p_node->node;

				// Functions not declared with the inline keyword are not inlined.
				if (!func->_inline_func && global_options.require_inline_keyword) {
					func_decl.supported = false;
					return false; // Stop traversing.
				}

				for (int n = 0; n < func->arguments.size(); n++) {
					FunctionParam fp;
					fp.name = func->arguments[n];
					func_decl.params.push_back(fp);
				}

				for (int n = 0; n < func->default_values.size(); n++) {
					if (func->default_values[n]->type == Node::TYPE_OPERATOR) {
						GDScriptParser::OperatorNode *op = (OperatorNode *)func->default_values[n];
						if (op->arguments.size() >= 2) {
							const StringName *name = helper_get_identifier_node_name(op->arguments[0]);
							DEV_ASSERT(name);
							String def_name = *name;

							for (uint32_t i = 0; i < func_decl.params.size(); i++) {
								if (func_decl.params[i].name == def_name) {
									func_decl.params[i].default_value = op->arguments[1];
								}
							}
						}
					}
				}

			} break;
			case Node::TYPE_OPERATOR: {
				const OperatorNode *op = (const OperatorNode *)p_node->node;
				if (op->arguments.size()) {
					if (helper_is_operator_assign(op->op)) {
						function_param_mark_use(func_decl, op->arguments[0], true);

						// Any function that contains an assign is non-const, and faces
						// restrictions for inlining.
						// ToDo: strictly speaking this only has to be for non-local vars.
						func_decl.is_const_function = false;
					}
					if (op->op == OperatorNode::OP_CALL || op->op == OperatorNode::OP_PARENT_CALL) {
						func_decl.is_const_function = false;
					}
				}

			} break;
			case Node::TYPE_IDENTIFIER: {
				// If this identifier is one of the function params, mark the param as used
				// (so it won't be optimized out).
				function_param_mark_use(func_decl, p_node->node, false);
			} break;
			case Node::TYPE_DICTIONARY: {
				func_decl.supported = false;
				warning_inline_unsupported(func_decl.node, "dictionary is unsupported.");
				return false;
			} break;
			case Node::TYPE_CONTROL_FLOW: {
				const GDScriptParser::ControlFlowNode *cf = (const ControlFlowNode *)p_node->node;

				// With if statements, if one body but not the else contains a return, we can't use the inline.
				if (cf->cf_type == ControlFlowNode::CF_IF) {
					bool body_return = false;
					bool else_return = false;

					if (cf->body) {
						body_return = helper_block_contains_return(cf->body);
					}
					if (cf->body_else) {
						else_return = helper_block_contains_return(cf->body_else);
					}

					if (cf->body) {
						// If there is an if with no else, we only allow ONE LAYER for this situation,
						// because otherwise the logic gets fiendishly complex (because there is no goto
						// in GDScript!).
						if (!cf->body_else) {
							if (!p_user_flag) {
								p_user_flag = true;
							} else {
								func_decl.supported = false;
								warning_inline_unsupported(func_decl.node, "if with return and no else cannot be nested.");
								return false;
							}
						}
					}
					if (cf->body_else) {
						if (else_return) {
							if (!body_return) {
								func_decl.supported = false;
								warning_inline_unsupported(func_decl.node, "if_else containing mixed returns.");
								return false;
							}

						} else {
							if (body_return) {
								func_decl.supported = false;
								warning_inline_unsupported(func_decl.node, "if_else containing mixed returns.");
								return false;
							}
						}
					}
				} else {
					// For non-if control flows, return statements anywhere below are not supported.
					if (helper_block_contains_return(cf->body, true) || helper_block_contains_return(cf->body_else, true)) {
						func_decl.supported = false;
						warning_inline_unsupported(func_decl.node, "non-if control flow containing return.");
						return false;
					}
				}

				// Matches with return are not yet supported.
				if (cf->match) {
					for (int n = 0; n < cf->match->branches.size(); n++) {
						const GDScriptParser::PatternBranchNode *branch = cf->match->branches[n];
						if (helper_block_contains_return(branch->body)) {
							func_decl.supported = false;
							warning_inline_unsupported(func_decl.node, "match containing return.");
							return false;
						}
					}
				}

				if (cf->cf_type == ControlFlowNode::CF_RETURN) {
					if (cf->arguments.size()) {
						// We are using a trick here.
						// If the return statement is within e.g. an If statement,
						// then removing it entirely can cause compile errors,
						// so we will only substitute if the return statement
						// has a depth immediately off of the function root.

						// By setting this to over 1, the later logic will prevent
						// substitution.
						func_decl.num_return_values += p_depth != 2 ? 2 : 1;
					}
				}
			} break;
		}
		return true;
	},
			0);

	// Finished with the tree.
	memdelete(tree);
}

void GDScriptOptimizer::duplicate_statements_recursive(const Vector<GDScriptParser::Node *> &p_source, int32_t p_source_pos, Vector<GDScriptParser::Node *> &r_dest, int32_t p_dest_pos, InlineInfo &p_changes, DuplicateNodeResult *r_result) {
	USING_GDSCRIPTPARSER

	int dest_n = p_dest_pos;

	for (int n = p_source_pos; n < p_source.size(); n++) {
		DuplicateNodeResult res;
		Node *dup = duplicate_node_recursive(*p_source[n], p_changes, &res);
		if (dup) {
			if (dest_n < r_dest.size()) {
				r_dest.set(dest_n, dup);
			} else {
				r_dest.push_back(dup);
			}
		} else {
			if (dest_n < r_dest.size()) {
				r_dest.remove(dest_n);
			}
			dest_n--;
		}
		dest_n++;

		// If we hit a return value, we either want to:
		// * remove all following statements (if both if and else have return)
		// * add all following statements to an else branch (if the if had no else)
		if (res.if_return_needs_else) {
			DEV_ASSERT(dup->type == Node::TYPE_CONTROL_FLOW);
			GDScriptParser::ControlFlowNode *cf = (GDScriptParser::ControlFlowNode *)dup;
			DEV_ASSERT(cf->cf_type == GDScriptParser::ControlFlowNode::CF_IF);
			DEV_ASSERT(!cf->body_else);
			GDOPT_ALLOC_NODE(BlockNode, else_block, cf);
			cf->body_else = else_block;

			// Add remaining statements to the else block.
			DuplicateNodeResult res2;
			duplicate_statements_recursive(p_source, n + 1, else_block->statements, 0, p_changes, &res2);
		}

		if (res.has_return) {
			// Allow one newline.
			if ((dest_n < r_dest.size()) && (r_dest[dest_n]->type == Node::TYPE_NEWLINE)) {
				dest_n++;
			}
			for (int i = r_dest.size() - 1; i >= dest_n; i--) {
				r_dest.remove(i);
			}

			if (r_result) {
				r_result->has_return = true;
			}
			break;
		}
	}
}

GDScriptParser::Node *GDScriptOptimizer::duplicate_node_recursive(const GDScriptParser::Node &p_source, InlineInfo &p_changes, DuplicateNodeResult *r_result) {
	USING_GDSCRIPTPARSER

	if (r_result) {
		r_result->if_return_needs_else = false;
	}

	// Special treatment for return instructions,
	// but only when inlining (we can test for this because returned expression is only set when inlining).
	if (p_changes.returned_expression && (p_source.type == Node::TYPE_CONTROL_FLOW)) {
		const ControlFlowNode *source_cf = (const ControlFlowNode *)&p_source;
		if (source_cf->cf_type == ControlFlowNode::CF_RETURN) {
			// If there are arguments...
			if (source_cf->arguments.size()) {
				// Make sure the arguments to the return value are duplicated.
				Node *returned_expression = duplicate_node_recursive(*source_cf->arguments[0], p_changes);

				// If we hit a return value, the calling node should delete any statements after this.
				if (r_result) {
					r_result->has_return = true;
				}

				if (p_changes.return_intermediate) {
					// If using an intermediate,
					// assign the expression instead of returning it.
					OperatorNode *assignment = inline_make_declare_assignment(p_changes.return_intermediate, returned_expression);
					return assignment;
				}

				// Blank out return values if we are passing directly to the calling code.
				if (p_changes.returned_expression) {
					*p_changes.returned_expression = returned_expression;
				}

				return nullptr;
			} // if there are arguments
			else {
				// If we hit a return value, the calling node should delete any statements after this.
				if (r_result) {
					r_result->has_return = true;
				}
				return nullptr;
			}
		}
	}

	GDScriptParser::Node *dest = duplicate_node(p_source);

	// Duplicate any children.
	switch (p_source.type) {
		default:
			break;

		case Node::TYPE_ARRAY: {
			const ArrayNode *source_arr = (ArrayNode *)&p_source;
			ArrayNode *dest_arr = (ArrayNode *)dest;
			for (int n = 0; n < source_arr->elements.size(); n++) {
				dest_arr->elements.set(n, duplicate_node_recursive(*source_arr->elements[n], p_changes));
			}
		} break;

		case Node::TYPE_LOCAL_VAR: {
			const LocalVarNode *source_var = (const LocalVarNode *)&p_source;
			LocalVarNode *dest_var = (LocalVarNode *)dest;

			String unique_var_name = helper_make_unique_name(source_var->name, UniqueNameType::VARIABLE);
			dest_var->name = unique_var_name;

			InlineChange change;
			change.identifier_from = source_var->name;
			change.identifier_to = unique_var_name;
			p_changes.changes.push_back(change);

			if (source_var->assign) {
				dest_var->assign = duplicate_node_recursive(*source_var->assign, p_changes);
			}

		} break;
		case Node::TYPE_IDENTIFIER: {
			const IdentifierNode *source_ident = (const IdentifierNode *)&p_source;
			IdentifierNode *dest_ident = (IdentifierNode *)dest;

			// Make any changes.
			// Note go backwards through the changes, as the most recently added scoped
			// variables take priority.
			bool changed = false;

			for (int32_t n = ((int32_t)p_changes.changes.size()) - 1; n >= 0; n--) {
				const InlineChange &change = p_changes.changes[n];
				if (source_ident->name == change.identifier_from) {
					if (change.node_to) {
						// Replace the returned node with the change constant.
						return change.node_to;
					} else {
						dest_ident->name = change.identifier_to;
						changed = true;
						break;
					}
				}
			}

			// Special case .. if we an identifier in a non-self class, we may need to add self scope to it.
			if (!changed && p_changes.gdclass && p_changes.self) {
				for (int n = 0; n < p_changes.gdclass->variables.size(); n++) {
					if (p_changes.gdclass->variables[n].identifier == dest_ident->name) {
						// Add a new change to a scoped variable.
						GDOPT_ALLOC_NODE(OperatorNode, scoped_variable, source_ident);
						scoped_variable->op = OperatorNode::OP_INDEX_NAMED;
						scoped_variable->arguments.push_back(p_changes.self);

						GDOPT_ALLOC_NODE(IdentifierNode, ident_copy, source_ident);
						ident_copy->name = source_ident->name;

						scoped_variable->arguments.push_back(ident_copy);

						InlineChange change;
						change.identifier_from = source_ident->name;
						change.node_to = scoped_variable;

						p_changes.changes.push_back(change);

						// Replace the current node with the changed constant.
						return scoped_variable;
					}
				}
			}

		} break;
		case Node::TYPE_BLOCK: {
			const BlockNode *source_block = (const BlockNode *)&p_source;
			BlockNode *dest_block = (BlockNode *)dest;

			// When we hit a block, we want to duplicate the changes, because we are starting a new scope.
			// Variables declared within this scope should not be preserved to outer scopes.
			InlineInfo new_scope = p_changes;

			DuplicateNodeResult res;
			duplicate_statements_recursive(source_block->statements, 0, dest_block->statements, 0, new_scope, &res);

			if (r_result && res.has_return) {
				r_result->has_return = true;
			}

		} break;
		case Node::TYPE_INLINE_BLOCK: {
			const InlineBlockNode *source_block = (const InlineBlockNode *)&p_source;
			InlineBlockNode *dest_block = (InlineBlockNode *)dest;

			DuplicateNodeResult res;
			duplicate_statements_recursive(source_block->statements, 0, dest_block->statements, 0, p_changes, &res);

			if (r_result && res.has_return) {
				r_result->has_return = true;
			}

		} break;
		case Node::TYPE_OPERATOR: {
			const OperatorNode *source_op = (const OperatorNode *)&p_source;
			OperatorNode *dest_op = (OperatorNode *)dest;

			// May need a special exemption for OP_INDEX_NAMED, to prevent infinite recursion loop
			// renaming these.
			// However, this cannot be used as is, because sometimes you genuinely do need to
			// do renames within OP_INDEX_NAMED when unrolling loops.
			switch (source_op->op) {
				default: {
					for (int n = 0; n < source_op->arguments.size(); n++) {
						dest_op->arguments.set(n, duplicate_node_recursive(*source_op->arguments[n], p_changes));
					}
				} break;
				case OperatorNode::OP_INDEX_NAMED: {
					DEV_ASSERT(source_op->arguments.size() == 2);
					// Change the scope, but NOT the element identifier.
					dest_op->arguments.set(0, duplicate_node_recursive(*source_op->arguments[0], p_changes));
				} break;
				case OperatorNode::OP_CALL: {
					DEV_ASSERT(source_op->arguments.size());

					// For built in functions, we can translate all args.
					// For non-built in, we don't want to translate scoped function names.
					switch (source_op->arguments[0]->type) {
						default: {
							dest_op->arguments.set(0, duplicate_node_recursive(*source_op->arguments[0], p_changes));

							// Do NOT modify the function name, as it may shadow a local variable.

							// Params
							for (int n = 2; n < source_op->arguments.size(); n++) {
								dest_op->arguments.set(n, duplicate_node_recursive(*source_op->arguments[n], p_changes));
							}
						} break;
						case Node::TYPE_TYPE:
						case Node::TYPE_BUILT_IN_FUNCTION: {
							for (int n = 0; n < source_op->arguments.size(); n++) {
								dest_op->arguments.set(n, duplicate_node_recursive(*source_op->arguments[n], p_changes));
							}

						} break;
					}
				} break;
			}
		} break;
		case Node::TYPE_CONTROL_FLOW: {
			const ControlFlowNode *source_cf = (const ControlFlowNode *)&p_source;
			ControlFlowNode *dest_cf = (ControlFlowNode *)dest;
			DEV_ASSERT(source_cf->arguments.size() == dest_cf->arguments.size());

			for (int n = 0; n < source_cf->arguments.size(); n++) {
				dest_cf->arguments.set(n, duplicate_node_recursive(*source_cf->arguments[n], p_changes));
			}

			// Special case for for loops, the first argument is a variable declaration.
			if (source_cf->cf_type == ControlFlowNode::CF_FOR) {
				DEV_ASSERT(source_cf->arguments.size() > 1);
				DEV_ASSERT(source_cf->arguments[0]->type == Node::TYPE_IDENTIFIER);
				IdentifierNode *source_var = (IdentifierNode *)source_cf->arguments[0];
				IdentifierNode *dest_var = (IdentifierNode *)dest_cf->arguments[0];

				String unique_var_name = helper_make_unique_name(source_var->name, UniqueNameType::VARIABLE);
				dest_var->name = unique_var_name;

				InlineChange change;
				change.identifier_from = source_var->name;
				change.identifier_to = unique_var_name;
				p_changes.changes.push_back(change);
			}

			DuplicateNodeResult res;
			if (source_cf->body) {
				dest_cf->body = (BlockNode *)duplicate_node_recursive(*source_cf->body, p_changes, &res);
				if (res.has_return) {
					res.if_return_needs_else = true;
				}
			}
			if (source_cf->body_else) {
				DuplicateNodeResult res2;
				dest_cf->body_else = (BlockNode *)duplicate_node_recursive(*source_cf->body_else, p_changes, &res2);
				if (res.has_return && !res2.has_return) {
					res.has_return = false;
				}
				res.if_return_needs_else = false;
			}

			if (source_cf->match) {
				// We need a duplicate match
				// (otherwise the data will be shared with the original).
				dest_cf->match = alloc_node<GDScriptParser::MatchNode>(source_cf->match);
				*dest_cf->match = *source_cf->match;

				dest_cf->match->val_to_match = duplicate_node_recursive(*source_cf->match->val_to_match, p_changes);

				for (int n = 0; n < source_cf->match->branches.size(); n++) {
					const GDScriptParser::PatternBranchNode *source_branch = source_cf->match->branches[n];
					// We need a duplicate pattern branch node
					// (otherwise the body will be shared with the original).
					GDScriptParser::PatternBranchNode *new_branch = alloc_node<GDScriptParser::PatternBranchNode>(source_branch);
					*new_branch = *source_branch;
					dest_cf->match->branches.set(n, new_branch);

					GDScriptParser::PatternBranchNode *dest_branch = dest_cf->match->branches[n];

					dest_branch->body = (BlockNode *)duplicate_node_recursive(*source_branch->body, p_changes);
				}

				for (int n = 0; n < source_cf->match->compiled_pattern_branches.size(); n++) {
					GDScriptParser::MatchNode::CompiledPatternBranch cpb = source_cf->match->compiled_pattern_branches[n];

					cpb.compiled_pattern = duplicate_node_recursive(*cpb.compiled_pattern, p_changes);
					cpb.body = (BlockNode *)duplicate_node_recursive(*cpb.body, p_changes);
					dest_cf->match->compiled_pattern_branches.set(n, cpb);
				}
			}

			if (r_result) {
				*r_result = res;
			}
		} break;
	}

	return dest;
}

GDScriptParser::Node *GDScriptOptimizer::duplicate_node(const GDScriptParser::Node &p_source) const {
	GDScriptParser::Node *res = nullptr;

	GDScriptParser::Node *source_next = p_source.next;

#define GDSCRIPTOPTIMIZER_COPY_CASE(TYPE, CLASS)                                                     \
	case GDScriptParser::Node::TYPE: {                                                               \
		const GDScriptParser::CLASS *source = static_cast<const GDScriptParser::CLASS *>(&p_source); \
		GDScriptParser::CLASS *node = alloc_node<GDScriptParser::CLASS>(&p_source);                  \
		*node = *source;                                                                             \
		node->next = source_next;                                                                    \
		res = node;                                                                                  \
	} break;

	switch (p_source.type) {
		default: {
			DEV_ASSERT(0);
		} break;
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_CLASS, ClassNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_FUNCTION, FunctionNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_BUILT_IN_FUNCTION, BuiltInFunctionNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_BLOCK, BlockNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_IDENTIFIER, IdentifierNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_TYPE, TypeNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_CONSTANT, ConstantNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_ARRAY, ArrayNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_DICTIONARY, DictionaryNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_SELF, SelfNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_OPERATOR, OperatorNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_CONTROL_FLOW, ControlFlowNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_LOCAL_VAR, LocalVarNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_CAST, CastNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_ASSERT, AssertNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_BREAKPOINT, BreakpointNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_NEWLINE, NewLineNode)
			GDSCRIPTOPTIMIZER_COPY_CASE(TYPE_INLINE_BLOCK, InlineBlockNode)
	}

	return res;
}

bool GDScriptOptimizer::inline_make_params(const Call &p_call, const Function &p_source_func, InlineInfo &r_info, int &r_insert_statement_id) {
	USING_GDSCRIPTPARSER

	// Some local aliases.
	Vector<Node *> &statements = p_call.parent_block->statements;
	FunctionNode *source_node = p_source_func.node;

	// If there is more than 1 return value, we need to create a return value intermediate local var to store these in.
	if (p_source_func.num_return_values > 1) // 1
	{
		String unique_return_name = helper_make_unique_name("return", UniqueNameType::RETURN);
		r_info.return_intermediate = helper_declare_local_var(unique_return_name, nullptr, statements, r_insert_statement_id);
	}

	// A change for each argument.
	r_info.changes.resize(source_node->arguments.size());
	DEV_ASSERT(r_info.changes.size() == p_source_func.params.size());

	// Parameters.
	for (int p = 0; p < source_node->arguments.size(); p++) {
		const StringName &source_param_name = source_node->arguments[p];
		const FunctionParam &function_param = p_source_func.params[p];

		if (!function_param.is_used) {
			GDOPT_LOG(1, "Inlining " + node_to_string(p_source_func.node) + ", param \"" + source_param_name + "\" is unused, optimizing out.");
			continue;
		}

		// Find the passed argument from the call node.
		//DEV_ASSERT(p_call.call_node->arguments.size() == (source_node->arguments.size() + 2));

		// Allow for default arguments.
		Node *passed_argument = nullptr;
		if (p_call.call_node->arguments.size() > p + 2) {
			passed_argument = p_call.call_node->arguments[p + 2];
		} else {
			passed_argument = function_param.default_value;
			ERR_FAIL_NULL_V(passed_argument, false);
		}

		// If the argument is not written, we can directly use it in the inlined function body,
		// but ONLY if it is an identifier (otherwise it may be a complex expression, and we are better off
		// evaluating it once as a local variable.
		bool substitute_directly = !function_param.is_written;
		switch (passed_argument->type) {
			default: {
				substitute_directly = false;
			} break;
			case Node::TYPE_IDENTIFIER:
			case Node::TYPE_CONSTANT:
				break;
		}

		// Create a change for each argument.
		InlineChange &change = r_info.changes[p];
		change.identifier_from = source_param_name;

		if (substitute_directly) {
			// If the argument is not written, we can directly use it in the inlined function body,
			// but ONLY if it is an identifier (otherwise it may be a complex expression, and we are better off
			// evaluating it once as a local variable.
			if (passed_argument->type == Node::TYPE_IDENTIFIER) {
				change.identifier_to = ((IdentifierNode *)passed_argument)->name;
			} else {
				// Will be a constant to be substituted directly.
				change.node_to = passed_argument;
			}
		} else {
			String unique_param_name = helper_make_unique_name(source_param_name, UniqueNameType::VARIABLE);
			change.identifier_to = unique_param_name;

			const GDScriptParser::DataType *arg_data_type = source_node->argument_types.size() ? &source_node->argument_types[p] : nullptr;
			helper_declare_local_var(unique_param_name, passed_argument, statements, r_insert_statement_id, source_node, arg_data_type);
		}
	}
	return true;
}

// Returns the local var name identifier.
GDScriptParser::IdentifierNode *GDScriptOptimizer::helper_declare_local_var(const String &p_local_var_name, GDScriptParser::Node *p_assigned_node, Vector<GDScriptParser::Node *> &r_statements, int &r_insert_statement_id, GDScriptParser::Node *p_source_location, const GDScriptParser::DataType *p_source_data_type) {
	USING_GDSCRIPTPARSER

	// If no source location supplied, use the assigned node for the location.
	if (!p_source_location) {
		p_source_location = p_assigned_node;
	}

	// Write out the variable declaration statement.
	LocalVarNode *local_var = alloc_node<LocalVarNode>(p_source_location);
	local_var->name = p_local_var_name;

	IdentifierNode *param_identifier_node = alloc_node<IdentifierNode>(p_source_location);
	param_identifier_node->name = p_local_var_name;

	if (p_source_data_type) {
		local_var->datatype = *p_source_data_type;
		param_identifier_node->datatype = *p_source_data_type;
	}

	r_statements.insert(r_insert_statement_id++, local_var);

	local_var->assign_op = inline_make_declare_assignment(param_identifier_node, p_assigned_node);

	local_var->assign = local_var->assign_op->arguments[1];
	r_statements.insert(r_insert_statement_id++, local_var->assign_op);
	r_statements.insert(r_insert_statement_id++, alloc_node<NewLineNode>(p_source_location));

	return param_identifier_node;
}

GDScriptParser::OperatorNode *GDScriptOptimizer::inline_make_declare_assignment(GDScriptParser::IdentifierNode *p_var_name, GDScriptParser::Node *p_assigned_node) {
	// Assign constant nil if not specified.
	if (!p_assigned_node) {
		p_assigned_node = alloc_node<GDScriptParser::ConstantNode>(p_var_name);
	}

	// Assign
	GDScriptParser::OperatorNode *assign_op = alloc_node<GDScriptParser::OperatorNode>(p_var_name); // p_assigned_node?
	assign_op->op = GDScriptParser::OperatorNode::OP_ASSIGN;
	assign_op->arguments.push_back(p_var_name);
	assign_op->arguments.push_back(p_assigned_node);

	return assign_op;
}

int GDScriptOptimizer::helper_find_insert_statement(const Vector<GDScriptParser::Node *> &p_statements, GDScriptParser::Node *p_search_node) const {
	for (int n = 0; n < p_statements.size(); n++) {
		GDScriptParser::Node *s = p_statements[n];

		if (s == p_search_node) {
			return n;
		}
	}

	return -1;
}

uint32_t GDScriptOptimizer::Data::find_class_by_name(const String &p_name) const {
	for (uint32_t n = 1; n < classes.size(); n++) {
		if (classes[n].class_name == p_name) {
			return n;
		}
	}

	return UINT32_MAX;
}

uint32_t GDScriptOptimizer::GDClass::find_function_by_name(const StringName &p_name) const {
	for (uint32_t n = 0; n < functions.size(); n++) {
		if (functions[n].node->name == p_name) {
			return n;
		}
	}
	return UINT32_MAX;
}

void GDScriptOptimizer::inline_make(const Call &p_call) {
	USING_GDSCRIPTPARSER

	// Find which function is being inlined.
	GDClass &cl = data.classes[p_call.gdclass_id];

	uint32_t inlined_func_id = cl.find_function_by_name(p_call.call_identifier->name);
	DEV_ASSERT(inlined_func_id == p_call.function_id);

	if (inlined_func_id == UINT32_MAX) {
		GDOPT_LOG(2, "Failed to inline " + p_call.call_identifier->name);
		return;
	}

	const Function &func = cl.functions[inlined_func_id];
	FunctionNode *source_node = func.node;
	ERR_FAIL_NULL(source_node);

	if (global_options.logging_level >= 1) {
		GDOPT_LOG(1, "Inlining " + node_to_string(source_node) + " into " + node_to_string(p_call.call_container_function));
	}

	if (global_options.logging_level >= 3) {
		String text = "\ninline original function\n************************\n";
		GDScriptReconstructor rec;
		rec.output_branch(p_call.parent_block, text, p_call.call_node);
		print_line(text);
		text = "\ninline source\n*************\n";
		rec.output_branch(source_node, text);
		print_line(text);
	}

	BlockNode *source_body = source_node->body;
	ERR_FAIL_NULL(source_body);

	int insert_statement_id = helper_find_insert_statement(p_call.parent_block->statements, p_call.parent_block_first_child);
	ERR_FAIL_COND(insert_statement_id == -1);

	InlineInfo info;
	info.gdclass = cl.root;
	info.self = p_call.self;

	GDScriptParser::Node *returned_expression = nullptr;
	info.returned_expression = &returned_expression;

	bool made_inline_params = inline_make_params(p_call, func, info, insert_statement_id);
	ERR_FAIL_COND(!made_inline_params);

	// We will use a special trick here.
	// We want to put the statements into an `InlineBlock`, but the source is a regular block.
	// So we will duplicate a regular block, then copy the statements into the inline block.
	GDOPT_ALLOC_NODE(InlineBlockNode, inline_block, source_body);
	BlockNode *inline_body = (BlockNode *)duplicate_node_recursive(*source_body, info);
	inline_block->statements = inline_body->statements;

	p_call.parent_block->statements.insert(insert_statement_id++, inline_block);

	// Finally change the call expression itself to be the return expression or return identifier..
	// or remove completely if there is no return value.
	if (returned_expression) {
		helper_node_exchange_child(*p_call.call_parent, p_call.call_node, returned_expression);
	} else if (info.return_intermediate) {
		helper_node_exchange_child(*p_call.call_parent, p_call.call_node, info.return_intermediate);
	} else {
		DEV_ASSERT(p_call.call_parent->type != Node::TYPE_OPERATOR);
		helper_node_exchange_child(*p_call.call_parent, p_call.call_node, nullptr);
	}

	// The InlineBlocks are nice for sorting, but they seem to confuse the standard compiler,
	// so we will remove them when no longer required.
	remove_inline_blocks(p_call.parent_block);
	remove_excessive_newlines(p_call.parent_block->statements, false);

	if (global_options.logging_level >= 3) {
		String text;
		text = "\ninline result\n*************\n";
		GDScriptReconstructor rec;
		rec.output_branch(p_call.parent_block, text, inline_body);
		print_line(text);
		print_line("\n");
	}
}

bool GDScriptOptimizer::helper_node_exchange_child(GDScriptParser::Node &r_parent, const GDScriptParser::Node *p_old_child, GDScriptParser::Node *p_new_child) {
	USING_GDSCRIPTPARSER

	switch (r_parent.type) {
		default: {
			return false;
		} break;
		case Node::TYPE_CONTROL_FLOW: {
			GDScriptParser::ControlFlowNode *cf = (ControlFlowNode *)&r_parent;
			for (int n = 0; n < cf->arguments.size(); n++) {
				if (cf->arguments[n] == p_old_child) {
					if (p_new_child) {
						cf->arguments.set(n, p_new_child);
					} else {
						cf->arguments.remove(n);
					}
					return true;
				}
			}
		} break;

		case Node::TYPE_BLOCK: {
			BlockNode *block = (BlockNode *)&r_parent;
			for (int n = 0; n < block->statements.size(); n++) {
				if (block->statements[n] == p_old_child) {
					if (p_new_child) {
						block->statements.set(n, p_new_child);
					} else {
						block->statements.remove(n);
					}
					return true;
				}
			}
		} break;
		case Node::TYPE_OPERATOR: {
			OperatorNode *op = (OperatorNode *)&r_parent;
			for (int n = 0; n < op->arguments.size(); n++) {
				if (op->arguments[n] == p_old_child) {
					op->arguments.set(n, p_new_child);
					return true;
				}
			}

		} break;
	}

	ERR_FAIL_V(false);
}

const StringName *GDScriptOptimizer::helper_get_identifier_node_name(const GDScriptParser::Node *p_node) {
	if (p_node->type == GDScriptParser::Node::TYPE_IDENTIFIER) {
		return &((const GDScriptParser::IdentifierNode *)p_node)->name;
	}
	return nullptr;
}

Vector<GDScriptParser::Node *> *GDScriptOptimizer::_helper_find_ancestor_statements(ASTNode *p_ast_node, ASTNode **p_first_child, bool p_highest_block) const {
	USING_GDSCRIPTPARSER

	// Find the parent block and insert point.
	DEV_ASSERT(p_ast_node);
	ASTNode *curr_child = nullptr;
	ASTNode *ast_node = p_ast_node;

	DEV_ASSERT(p_first_child);
	*p_first_child = nullptr;
	Vector<Node *> *statements = nullptr;

	bool always_accept_next_block = false;

	// Only let us out through the first for loop, so we can do LICM, but not create regressions.
	// Sometimes this will miss out on some possible performance.
	while (ast_node) {
		switch (ast_node->node->type) {
			default:
				break;
			case Node::TYPE_CONTROL_FLOW: {
				ControlFlowNode *cf = (ControlFlowNode *)ast_node->node;
				if (cf->cf_type == ControlFlowNode::CF_FOR) {
					always_accept_next_block = true;
				}
			} break;
			case Node::TYPE_BLOCK: {
				statements = &((BlockNode *)ast_node->node)->statements;
				*p_first_child = curr_child;

				if (!p_highest_block || always_accept_next_block) {
					return statements;
				}
			} break;
			case Node::TYPE_INLINE_BLOCK: {
				statements = &((InlineBlockNode *)ast_node->node)->statements;
				*p_first_child = curr_child;

				if (!p_highest_block || always_accept_next_block) {
					return statements;
				}
			} break;
		}

		curr_child = ast_node;
		ast_node = ast_node->parent;
	}

	return statements;
}

int GDScriptOptimizer::helper_find_ancestor_insert_statement(ASTNode *p_ast_node, Vector<GDScriptParser::Node *> **r_statements, ASTNode **r_statement_holder, bool p_highest_block) const {
	ASTNode *first_child = nullptr;
	Vector<GDScriptParser::Node *> *statements = _helper_find_ancestor_statements(p_ast_node, &first_child, p_highest_block);
	if (!statements) {
		return -1;
	}
	*r_statements = statements;
	*r_statement_holder = first_child->parent;

	return helper_find_insert_statement(*statements, first_child->node);
}

bool GDScriptOptimizer::licm_remove_if_statement_holder_empty(ASTNode *p_holder, Vector<GDScriptParser::Node *> &r_statements) {
	USING_GDSCRIPTPARSER

	// Are the statements empty or all newlines?
	for (int n = 0; n < r_statements.size(); n++) {
		if (r_statements[n]->type != Node::TYPE_NEWLINE) {
			return false;
		}
	}

	// If we got to here, we want to remove the holder entirely.
	r_statements.clear();
	ASTNode *parent = p_holder->parent;
	DEV_ASSERT(parent);

	switch (parent->node->type) {
		default: {
			return false;
		} break;
		case Node::TYPE_INLINE_BLOCK: {
			InlineBlockNode *block = (InlineBlockNode *)parent->node;
			int pos = helper_find_insert_statement(block->statements, p_holder->node);
			block->statements.remove(pos);

			// Recurse to parent...
			return licm_remove_if_statement_holder_empty(parent, block->statements);
		} break;
		case Node::TYPE_BLOCK: {
			BlockNode *block = (BlockNode *)parent->node;
			int pos = helper_find_insert_statement(block->statements, p_holder->node);
			block->statements.remove(pos);

			// Recurse to parent...
			return licm_remove_if_statement_holder_empty(parent, block->statements);
		} break;
	}

	return true;
}

void GDScriptOptimizer::licm_shift_variable_declaration(ASTNode *p_ast_for, const LICMVarInfo &p_vi) {
	USING_GDSCRIPTPARSER
	GDOPT_LOG(2, "shifting variable declaration " + file_location_to_string(p_ast_for->node) + " : \"" + p_vi.name + "\"");
	ERR_FAIL_NULL(p_vi.local_var_node);

	Vector<Node *> *extract_statements = nullptr;
	ASTNode *extract_statement_holder = nullptr;
	int extract_pos = helper_find_ancestor_insert_statement(p_vi.local_var_node, &extract_statements, &extract_statement_holder);

	if (global_options.logging_level >= 3) {
		print_line("\nlicm shift variable declaration before\n*************************************\n");
		GDScriptReconstructor rec;
		String text;
		rec.output_branch(extract_statement_holder->node, text, p_vi.local_var_node->node);
		print_line(text);
	}

	// Mark as done for this run...
	p_vi.local_var_node->flood_fill(1);

	// Move the "var" statement.
	// In the insert statements, it still needs to be followed by an assignment.
	// In the extract statements, without the var, it becomes an assignment only,
	// which is what we want to achieve.
	LICMInsertLocation &loc = data.licm_location;

	loc.insert_statements->insert(loc.insert_pos++, p_vi.local_var_node->node);
	extract_statements->remove(extract_pos);

	// Make a *copy* of the assignment (so we can change the assigned value to null).
	InlineInfo changes;
	Node *new_decl = duplicate_node_recursive(*extract_statements->get(extract_pos), changes);
	loc.insert_statements->insert(loc.insert_pos++, new_decl);

	ERR_FAIL_COND(loc.insert_statements->get(loc.insert_pos - 1)->type != Node::TYPE_OPERATOR);
	OperatorNode *op = (OperatorNode *)loc.insert_statements->get(loc.insert_pos - 1);
	ERR_FAIL_COND(op->arguments.size() != 2);

	op->arguments.set(1, helper_get_default_value_for_type(op->arguments[0]->get_datatype()));

	helper_statements_add_newline(*loc.insert_statements, loc.insert_pos);

	if (global_options.logging_level >= 3) {
		print_line("\nlicm shift variable declaration result\n*************************************\n");
		GDScriptReconstructor rec;
		String text;
		rec.output_branch(loc.insert_statement_holder->node, text, new_decl);
		print_line(text);
	}
}

GDScriptParser::Node *GDScriptOptimizer::helper_get_default_value_for_type(const GDScriptParser::DataType &p_type, int p_line) const {
	GDScriptParser::Node *result;

	if (p_type.has_type && p_type.kind == GDScriptParser::DataType::BUILTIN && p_type.builtin_type != Variant::NIL && p_type.builtin_type != Variant::OBJECT) {
		if (p_type.builtin_type == Variant::ARRAY) {
			result = alloc_node<GDScriptParser::ArrayNode>(nullptr);
		} else if (p_type.builtin_type == Variant::DICTIONARY) {
			result = alloc_node<GDScriptParser::DictionaryNode>(nullptr);
		} else {
			GDScriptParser::ConstantNode *c = alloc_node<GDScriptParser::ConstantNode>(nullptr);
			Variant::CallError err;
			c->value = Variant::construct(p_type.builtin_type, nullptr, 0, err);
			result = c;
		}
	} else {
		GDScriptParser::ConstantNode *c = alloc_node<GDScriptParser::ConstantNode>(nullptr);
		c->value = Variant();
		result = c;
	}

	result->line = p_line;

	return result;
}

void GDScriptOptimizer::licm_shift_variable(ASTNode *p_ast_for, const LICMVarInfo &p_vi) {
	USING_GDSCRIPTPARSER

	// BugFix.
	// Only allow shifting variables that are local variables in the loop.
	// This is because variables declared outside the loop may be changed within the loop,
	// and thus become non-invariant.
	// ToDo: Alternatively we can disallow them if they are changed within the loop - Investigate.
	if (!p_vi.local_var_node) {
		return;
	}

	GDOPT_LOG(2, "shifting variable " + file_location_to_string(p_ast_for->node) + " : \"" + p_vi.name + "\"");

	LICMInsertLocation &loc = data.licm_location;

	// Move assigns.
	if (p_vi.local_var_node) {
		if (global_options.logging_level >= 3) {
			print_line("\nlicm shift variable assign before\n*********************************\n");
			GDScriptReconstructor rec;
			String text;
			rec.output_branch(p_vi.local_var_node->parent->node, text, p_vi.local_var_node->node);
			print_line(text);
		}

		Vector<Node *> *extract_statements = nullptr;
		ASTNode *extract_statement_holder = nullptr;
		int extract_pos = helper_find_ancestor_insert_statement(p_vi.local_var_node, &extract_statements, &extract_statement_holder);

		// Mark as done for this run...
		// Note that involves marking TWO AST node branches,
		// The assignment operator should be marked in the assignment mentions loop below.
		p_vi.local_var_node->flood_fill(1);

		loc.insert_statements->insert(loc.insert_pos++, p_vi.local_var_node->node);
		extract_statements->remove(extract_pos);

		helper_exchange_statements(*extract_statements, extract_pos, *loc.insert_statements, loc.insert_pos);
		helper_exchange_statements(*extract_statements, extract_pos, *loc.insert_statements, loc.insert_pos);

		licm_remove_if_statement_holder_empty(extract_statement_holder, *extract_statements);
	}

	// Move all mentions.
	// The assignment mentions are expressions that depend on the invariant variable:
	// e.g. myfunc(chicken + 1)
	// where chicken is invariant, (chicken+1) can be evaluated outside the loop.
	for (uint32_t n = 0; n < p_vi.assignment_mentions.size(); n++) {
		Node *op_to_move = p_vi.assignment_mentions[n].op->node;

		Vector<Node *> *extract_statements = nullptr;
		ASTNode *extract_statement_holder = nullptr;
		int extract_pos = helper_find_ancestor_insert_statement(p_vi.assignment_mentions[n].op, &extract_statements, &extract_statement_holder);

		// Mark it as moved (even if part of the local var assignment).
		p_vi.assignment_mentions[n].op->flood_fill(1);

		// We might be trying to copy an operator that has already been copied as part of a local var assignment.
		if (extract_pos == -1) {
			continue;
		}

		if (global_options.logging_level >= 3) {
			print_line("\nlicm shift variable mention before\n**********************************\n");
			GDScriptReconstructor rec;
			String text;
			rec.output_branch(extract_statement_holder->node, text, op_to_move);
			print_line(text);
		}

		loc.insert_statements->insert(loc.insert_pos++, op_to_move);
		extract_statements->remove(extract_pos);

		// Operator parameters and newline.
		helper_exchange_statements(*extract_statements, extract_pos, *loc.insert_statements, loc.insert_pos);
	}

	if (global_options.logging_level >= 3) {
		print_line("\nlicm shift variable result\n**************************\n");
		GDScriptReconstructor rec;
		String text;
		rec.output_branch(loc.insert_statement_holder->node, text, p_vi.local_var_node ? p_vi.local_var_node->node : nullptr);
		print_line(text);
	}
}

// The InlineBlocks are nice for sorting, but they seem to confuse the standard compiler,
// so we will remove them when no longer required.
void GDScriptOptimizer::remove_inline_blocks(GDScriptParser::Node *p_branch) {
	USING_GDSCRIPTPARSER

	ASTNode *tree = memnew(ASTNode(p_branch, nullptr));

	LocalVector<GDScriptParser::BlockNode *> blocks;

	tree->traverse([&blocks](ASTNode *p_node, uint32_t p_depth, bool p_user_flag) {
		switch (p_node->node->type) {
			default:
				break;
			case GDScriptParser::Node::TYPE_BLOCK: {
				blocks.push_back((BlockNode *)p_node->node);
			} break;
		}
		return true;
	},
			0);

	memdelete(tree);

	for (uint32_t b = 0; b < blocks.size(); b++) {
		BlockNode *block = blocks[b];

		for (int n = 0; n < block->statements.size(); n++) {
			if (block->statements[n]->type == Node::TYPE_INLINE_BLOCK) {
				InlineBlockNode *inblock = (InlineBlockNode *)block->statements[n];
				block->statements.remove(n);

				// Add each statement from the inline black to the statements individually.
				for (int i = 0; i < inblock->statements.size(); i++) {
					block->statements.insert(n++, inblock->statements[i]);
				}

				n--; // This will get incremented on next for loop.
			}
		}
	}
}

void GDScriptOptimizer::_remove_excessive_newlines_recursive(GDScriptParser::Node *p_branch, bool p_remove_from_front) {
	ASTNode *tree = memnew(ASTNode(p_branch, nullptr));

	tree->traverse([p_remove_from_front](ASTNode *p_node, uint32_t p_depth, bool p_user_flag) {
		switch (p_node->node->type) {
			default:
				break;
			case GDScriptParser::Node::TYPE_BLOCK: {
				remove_excessive_newlines(((GDScriptParser::BlockNode *)p_node->node)->statements, p_remove_from_front);
			} break;
			case GDScriptParser::Node::TYPE_INLINE_BLOCK: {
				remove_excessive_newlines(((GDScriptParser::InlineBlockNode *)p_node->node)->statements, p_remove_from_front);
			} break;
		}
		return true;
	},
			0);

	memdelete(tree);
}

void GDScriptOptimizer::remove_excessive_newlines_recursive() {
	_remove_excessive_newlines_recursive(data.get_root_class().root, false);
}

void GDScriptOptimizer::remove_excessive_newlines(Vector<GDScriptParser::Node *> &r_statements, bool p_remove_from_front, bool *p_leading_newline) {
	bool prev_newline = p_leading_newline ? *p_leading_newline : false;

	if (p_remove_from_front) {
		while (r_statements.size() && r_statements[0]->type == GDScriptParser::Node::TYPE_NEWLINE) {
			r_statements.remove(0);
		}
	}

	for (int n = 0; n < r_statements.size(); n++) {
		switch (r_statements[n]->type) {
			default: {
				prev_newline = false;
			} break;
			case GDScriptParser::Node::TYPE_NEWLINE: {
				if (prev_newline) {
					r_statements.remove(n);
					n--;
				} else {
					prev_newline = true;
				}
			} break;
			case GDScriptParser::Node::TYPE_INLINE_BLOCK: {
				GDScriptParser::InlineBlockNode *block = (GDScriptParser::InlineBlockNode *)r_statements[n];
				remove_excessive_newlines(block->statements, false, &prev_newline);
			} break;
			case GDScriptParser::Node::TYPE_BLOCK: {
				GDScriptParser::BlockNode *block = (GDScriptParser::BlockNode *)r_statements[n];
				remove_excessive_newlines(block->statements, false, &prev_newline);
			} break;
		}
	}
	if (p_leading_newline) {
		*p_leading_newline = prev_newline;
	}
}

bool GDScriptOptimizer::helper_exchange_statements(Vector<GDScriptParser::Node *> &r_extract_statements, int p_extract_pos, Vector<GDScriptParser::Node *> &r_insert_statements, int &r_insert_pos) const {
	DEV_ASSERT(p_extract_pos >= 0);
	DEV_ASSERT(r_insert_pos >= 0);
	ERR_FAIL_COND_V(p_extract_pos >= r_extract_statements.size(), false);

	r_insert_statements.insert(r_insert_pos++, r_extract_statements[p_extract_pos]);
	r_extract_statements.remove(p_extract_pos);
	return true;
}

bool GDScriptOptimizer::licm_is_simple_type(const GDScriptParser::DataType &p_dt) {
	if (p_dt.kind != GDScriptParser::DataType::BUILTIN) {
		return false;
	}
	switch (p_dt.builtin_type) {
		default: {
			return false;
		} break;
		case Variant::BOOL:
		case Variant::REAL:
		case Variant::INT: {
		} break;
	}

	return true;
}

void GDScriptOptimizer::licm_process_for_loop(GDScriptParser::ControlFlowNode *p_cf_for) {
	USING_GDSCRIPTPARSER

	// We need to create local ASTTrees for each for loop separately because the details may change,
	// and the AST may become out of date for each for loop if we reused the same one.
	ASTNode *ast_for = memnew(ASTNode(p_cf_for, nullptr));

	struct LICMVars info;
	info.licm_root = ast_for;

#if 0
	print_line("licm_process_for_loop : AST for loop.....");
	p_ast_for->debug_print();
#endif

	// Traverse the AST tree using a lambda to extract useful info about the function.
	ast_for->traverse([&info](ASTNode *p_node, uint32_t p_depth, bool &r_kill_flag) {
		switch (p_node->node->type) {
			default:
				break;
			case Node::TYPE_LOCAL_VAR: {
				GDScriptParser::LocalVarNode *var = (LocalVarNode *)p_node->node;

				LICMVarInfo *vi = info.find_or_create(var->name);
				vi->assigned = true;

				// Non simple types (e.g. arrays, Vector3)
				// may be altered by changing member variables
				// (e.g. vec.x = 3, arr.append(5))
				// This is hard to detect, so we don't support anything but simple types for now.
				if (!var->assign || !licm_is_simple_type(var->assign->get_datatype())) {
					vi->disallow = true;
				}

				if (r_kill_flag) {
					vi->disallow = true;
					vi->shift_declaration = false;
				}
				// If not at a lower level that the loop, then we can shift the variable
				// declaration outside the loop.
				else if (!vi->disallow) {
					vi->shift_declaration = true;
				}

				switch (var->assign->type) {
					default: {
						vi->expression_writes++;
					} break;
					case Node::TYPE_CONSTANT: {
						vi->constant_writes++;
					} break;
				}

				bool multiple_declarations = vi->local_var_node;
				vi->local_var_node = p_node;

				GDOPT_LOG(4, "assessing variable \"" + vi->name + "\" for loop optimization.");
				if (multiple_declarations) {
					// Same variable name reused in multiple places,
					// don't change at all.
					GDOPT_LOG(2, "multiple declarations detected \"" + vi->name + "\" during loop optimization.");
					vi->disallow = true;
					vi->shift_declaration = false;
				}
			} break;
			case Node::TYPE_CONTROL_FLOW: {
				GDScriptParser::ControlFlowNode *cf = (ControlFlowNode *)p_node->node;

				// Once we hit a control flow node, and it isn't the starting e.g. for loop, then we enter
				// a "kill" node when any variable touched becomes disallowed.
				// This is because the variable state is conditional, so we can't guarantee its state easily.
				if (p_node != info.licm_root) {
					r_kill_flag = true;
				}

				// Special. In for loops, the argument is the same as a local variable declaration.
				if (cf->cf_type == ControlFlowNode::CF_FOR) {
					for (int n = 0; n < cf->arguments.size(); n++) {
						if (cf->arguments[n]->type == Node::TYPE_IDENTIFIER) {
							GDScriptParser::IdentifierNode *ident = (GDScriptParser::IdentifierNode *)cf->arguments[n];

							LICMVarInfo *vi = info.find_or_create(ident->name);
							vi->assigned = true;
							vi->expression_writes++;
							vi->disallow = true;
							vi->shift_declaration = false;
						}
					}
				}
			} break;
			case Node::TYPE_OPERATOR: {
				GDScriptParser::OperatorNode *op = (OperatorNode *)p_node->node;

				if (!op->arguments.size() || op->arguments.size() > 2) {
					// Not interested.. keeping traversing.
					return true;
				}

				// First deal with assignments.
				if (op->arguments[0]->type == OperatorNode::TYPE_IDENTIFIER) {
					const IdentifierNode *ident = (const IdentifierNode *)op->arguments[0];

					switch (op->op) {
						default:
							break;

						// All these alter the value in a loop,
						// so are disallowed.
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
							LICMVarInfo *vi = info.find_or_create(ident->name);
							vi->disallow = true;

							DEV_ASSERT(op->arguments.size() == 2);
							LICMMention mention;
							mention.expression = p_node->children[p_node->find_child_id(op->arguments[1])];
							mention.op = mention.expression; // p_node?

							info.free_expression_mentions.push_back(mention);
						} break;

						case OperatorNode::OP_INIT_ASSIGN:
						case OperatorNode::OP_ASSIGN: {
							LICMVarInfo *vi = info.find_or_create(ident->name);

							if (r_kill_flag) {
								vi->disallow = true;
							}
							DEV_ASSERT(op->arguments.size() == 2);

							switch (op->arguments[1]->type) {
								default: {
									vi->expression_writes++;
								} break;
								case Node::TYPE_CONSTANT: {
									vi->constant_writes++;
								} break;
							}

							LICMMention mention;
							mention.op = p_node;
							mention.expression = p_node->children[p_node->find_child_id(op->arguments[1])];

							vi->assignment_mentions.push_back(mention);
						} break;
					}
				}

				// Potential const expressions to move out of the loop.
				// Now deal with all non-assignment expressions we are interested in.
				if (helper_is_operator_unary(op->op)) {
					DEV_ASSERT(op->arguments.size() == 1);
					if (op->arguments[0]->type == Node::TYPE_IDENTIFIER) {
						const IdentifierNode *id = (const IdentifierNode *)op->arguments[0];
						LICMVarInfo *vi = info.find_or_create(id->name);

						LICMMention mention;
						mention.op = p_node;
						mention.expression = p_node->children[p_node->find_child_id(op->arguments[0])];
						vi->expression_mentions.push_back(mention);
					}
				} else if (helper_is_operator_binary(op->op)) {
					DEV_ASSERT(op->arguments.size() == 2);
					for (int a = 0; a < 2; a++) {
						if (op->arguments[a]->type == Node::TYPE_IDENTIFIER) {
							const IdentifierNode *id = (const IdentifierNode *)op->arguments[a];
							LICMVarInfo *vi = info.find_or_create(id->name);

							LICMMention mention;
							mention.op = p_node;

							uint32_t child_id = p_node->find_child_id(op->arguments[a]);
							mention.expression = p_node->children[child_id];
							vi->expression_mentions.push_back(mention);
						}
					}
				}
			}
		}
		return true;
	},
			0);

	// Once we have declared all the variables and decided whether they are invariant, we can mark all of the AST variable nodes,
	// in order to later decide which expressions are invariant.
	ast_for->traverse([info](ASTNode *p_node, uint32_t p_depth, bool &r_kill_flag) {
		switch (p_node->node->type) {
			default:
				break;
#if 0
			case Node::TYPE_LOCAL_VAR: {
				GDScriptParser::LocalVarNode *var = (LocalVarNode *)p_node->node;
			} break;
			case Node::TYPE_CONTROL_FLOW: {
				GDScriptParser::ControlFlowNode *cf = (ControlFlowNode *)p_node->node;
			} break;
#endif
			case Node::TYPE_OPERATOR: {
				GDScriptParser::OperatorNode *op = (OperatorNode *)p_node->node;

				int num_args = op->arguments.size();

				if (num_args > 2) {
					// Not interested.. keeping traversing.
					return true;
				}

				for (int a = 0; a < num_args; a++) {
					GDScriptParser::Node *arg = op->arguments[a];
					if (arg->type == Node::TYPE_CONSTANT) {
						p_node->get_argument(a)->invariant_state = ASTNode::ASTInvariant::INVARIANT;
					}
					if (arg->type == Node::TYPE_IDENTIFIER) {
						const IdentifierNode *id = (const IdentifierNode *)arg;
						const LICMVarInfo *var = info.find(id->name);
						if (!(var && (var->expression_writes || var->disallow))) {
							p_node->get_argument(a)->invariant_state = ASTNode::ASTInvariant::INVARIANT;
						}
					}
				}
			} break;
		}
		return true;
	},
			0);

	for (uint32_t n = 0; n < info.vars.size(); n++) {
		LICMVarInfo &vi = info.vars[n];

		// Even if disallowed, if there is a variable declaration,
		// we can push the declaration outside the loop, and
		// only do assignment in the loop.
		if (vi.disallow) {
			if (vi.shift_declaration) {
				licm_shift_variable_declaration(ast_for, vi);
			}

			continue;
		}

		if (!vi.expression_writes) {
			vi.invariant = true;
			licm_shift_variable(ast_for, vi);
		} else {
			// See if the expression is invariant in the loop.
			bool invariant = true;

			for (uint32_t m = 0; m < vi.assignment_mentions.size(); m++) {
				// If any operation is not invariant, we can't move it.
				if (!licm_is_op_invariant(vi.name, vi.assignment_mentions[m].expression, info)) {
					invariant = false;
					break;
				}
			}

			if (invariant) {
				vi.invariant = true;
				licm_shift_variable(ast_for, vi);
			}
		}
	}

	// Once we have the variable information, we know which are constants, so we can precalculate any
	// expressions that use only constants.
	// These potential operators are already stored as "mentions".
	for (uint32_t n = 0; n < info.vars.size(); n++) {
		const LICMVarInfo &vi = info.vars[n];
		if (vi.invariant) {
			for (uint32_t m = 0; m < vi.expression_mentions.size(); m++) {
				licm_try_optimize_expression_mention(ast_for, vi.expression_mentions[m]);
			}
		}
	}

	// Free expression mentions (not associated with any particular variable)
	for (uint32_t n = 0; n < info.free_expression_mentions.size(); n++) {
		licm_try_optimize_expression_mention(ast_for, info.free_expression_mentions[n]);
	}

	// Finished with the tree.
	memdelete(ast_for);
}

bool GDScriptOptimizer::licm_is_expression_constant(ASTNode *p_op) const {
	USING_GDSCRIPTPARSER

	uint32_t num_args = p_op->children.size();

	// Not dealt with. May be a call, we are only interested in double operators.
	if (!num_args || num_args > 2) {
		return false;
	}

	if (p_op->invariant_state != ASTNode::ASTInvariant::UNPROCESSED) {
		return p_op->invariant_state == ASTNode::ASTInvariant::INVARIANT;
	}

	for (uint32_t n = 0; n < num_args; n++) {
		ASTNode *arg = p_op->get_argument(n);
		switch (arg->invariant_state) {
			case ASTNode::ASTInvariant::INVARIANT:
				// Fine...
				break;
			case ASTNode::ASTInvariant::VARIANT: {
				p_op->invariant_state = ASTNode::ASTInvariant::VARIANT;
				return false;
			} break;
			case ASTNode::ASTInvariant::UNPROCESSED: {
				// We need to evaluate this child.
				if (!licm_is_expression_constant(arg)) {
					p_op->invariant_state = ASTNode::ASTInvariant::VARIANT;
					return false;
				}
			} break;
		}
	}

	// Cache the state for next time hit.
	p_op->invariant_state = ASTNode::ASTInvariant::INVARIANT;
	return true;
}

bool GDScriptOptimizer::licm_try_optimize_expression_mention(ASTNode *p_ast_for, const LICMMention &p_mention) {
	USING_GDSCRIPTPARSER

#if 0
	print_line("licm_try_optimize_expression : mention " + node_to_string(mention.op->node) + ", flood is " + itos(mention.op->flood_id));
#endif

	// The condition for the op to be shiftable, is if both sides of the expression are constant.
	// Either constants, or invariant variables.

	ASTNode *ast_op = p_mention.op;

	// Don't do if we've hit it already...
	if (ast_op->flood_id != 0) {
		return false;
	}

	ASTNode *biggest_const_op = nullptr;

	while (ast_op) {
		if (ast_op->node->type != Node::TYPE_OPERATOR) {
			break;
		}
		if (!licm_is_expression_constant(ast_op)) {
			break;
		}

		biggest_const_op = ast_op;

		// Now try the parent.
		ast_op = ast_op->parent;
	}

	// If no const expr found...
	if (!biggest_const_op) {
		return false;
	}

	// Now we can shift the expression.
	licm_shift_const_expression(p_ast_for, biggest_const_op);
	return true;
}

void GDScriptOptimizer::licm_shift_const_expression(ASTNode *p_ast_for, ASTNode *p_ast_expr) {
	USING_GDSCRIPTPARSER

	GDOPT_LOG(2, "shifting const expression " + node_to_string(p_ast_expr->node) + " to " + node_to_string(p_ast_for->node));

	if (global_options.logging_level >= 3) {
		print_line("\nlicm shift constexpr before\n***************************\n");
		GDScriptReconstructor rec;
		String text;
		rec.output_branch(p_ast_expr->parent->node, text, p_ast_expr->node);
		print_line(text);
	}

	// Mark all the operators recursively as done for this run...
	p_ast_expr->flood_fill(1);

	LICMInsertLocation &loc = data.licm_location;

	// Create the temporary declared variable before the loop.
	String unique_temp_name = helper_make_unique_name("temp", UniqueNameType::TEMP);
	GDScriptParser::IdentifierNode *temp_ident = helper_declare_local_var(unique_temp_name, p_ast_expr->node, *loc.insert_statements, loc.insert_pos);

	// Replace the expression with the temp.
	// This is somewhat complex as it depends on the parent type.
	helper_node_exchange_child(*p_ast_expr->parent->node, p_ast_expr->node, temp_ident);

	if (global_options.logging_level >= 3) {
		print_line("\nlicm shift constexpr result\n***************************\n");
		GDScriptReconstructor rec;
		String text;
		rec.output_branch(loc.insert_statement_holder->node, text, temp_ident);
		print_line(text);
	}
}

bool GDScriptOptimizer::licm_is_op_invariant(const String &p_variable_name, ASTNode *p_ast_variable, const LICMVars &p_vars) const {
	USING_GDSCRIPTPARSER

	bool invariant = true;

	p_ast_variable->traverse([&p_vars, &invariant, p_variable_name](ASTNode *p_node, uint32_t p_depth, bool p_user_flag) {
		switch (p_node->node->type) {
			default:
				break;
			case Node::TYPE_OPERATOR: {
				const GDScriptParser::OperatorNode *op = (const OperatorNode *)p_node->node;
				switch (op->op) {
					default:
						break;
					// A lot of operators are disallowed for invariant, because they could have side effects.
					case OperatorNode::OP_CALL: {
						invariant = false;
					} break;
				}

			} break;
			case Node::TYPE_IDENTIFIER: {
				const GDScriptParser::IdentifierNode *ident = (const IdentifierNode *)p_node->node;
				const LICMVarInfo *var = p_vars.find(ident->name);
				if (var && (var->expression_writes || var->disallow)) {
					// It is not invariant.
					invariant = false;
				}
			} break;
		}
		return true;
	},
			0);

	return invariant;
}

void GDScriptOptimizer::unroll_loops() {
	USING_GDSCRIPTPARSER
	ASTNode *tree = memnew(ASTNode(data.get_root_class().root, nullptr));

	LocalVector<ASTNode *> loops;
	helper_find_all_types(loops, tree, Node::TYPE_CONTROL_FLOW, ControlFlowNode::CF_FOR);

	for (uint32_t n = 0; n < loops.size(); n++) {
		unroll_loop(loops[n]);
	}

	memdelete(tree);
}

bool GDScriptOptimizer::unroll_are_counters_invariant(ASTNode *p_control_flow, GDScriptParser::BlockNode *p_body, bool &r_is_read) const {
	USING_GDSCRIPTPARSER

	// Could be an else section with nothing in.
	if (!p_body) {
		return true;
	}

	ControlFlowNode *cf = (ControlFlowNode *)p_control_flow->node;
	DEV_ASSERT(cf->arguments.size() >= 2);

	Node *a0 = cf->arguments[0];
	Node *a1 = cf->arguments[1];

	// Counter must always be identifier.
	if (a0->type != Node::TYPE_IDENTIFIER) {
		warning_unroll_unsupported(p_control_flow->node, "loop counter is not identifier.");
		return false;
	}
	IdentifierNode *counter = (IdentifierNode *)a0;

	// Limit can be constant or identifier.
	IdentifierNode *limit = nullptr;
	switch (a1->type) {
		default: {
			warning_unroll_unsupported(p_control_flow->node, "loop limit is unsupported.");
			return false;
		} break;
		case Node::TYPE_IDENTIFIER: {
			limit = (IdentifierNode *)a1;
		} break;
		case Node::TYPE_CONSTANT: {
			GDScriptParser::ConstantNode *con = (ConstantNode *)a1;
			if (con->value.get_type() != Variant::INT) {
				warning_unroll_unsupported(p_control_flow->node, "loop constant limit must be INT.");
				return false;
			}
		} break;
		case Node::TYPE_OPERATOR: {
			// If an operator, a simple call is allowed with
			// arg 0 is TYPE_TYPE Variant::INT
			// arg 1 is identifier
			OperatorNode *op = (OperatorNode *)a1;
			if (op->op != OperatorNode::OP_CALL) {
				goto op_not_supported;
			}
			if (op->arguments.size() != 2) {
				goto op_not_supported;
			}
			if (op->arguments[0]->type != Node::TYPE_TYPE) {
				goto op_not_supported;
			}
			if (((GDScriptParser::TypeNode *)op->arguments[0])->vtype != Variant::INT) {
				goto op_not_supported;
			}
			if (op->arguments[1]->type != Node::TYPE_IDENTIFIER) {
				goto op_not_supported;
			}
			limit = (IdentifierNode *)op->arguments[1];
			break;
		op_not_supported:
			warning_unroll_unsupported(p_control_flow->node, "loop limit operator not supported.");
			return false;
		} break;
	}

	bool invariant = true;

	ASTNode *ast_body = p_control_flow->find_child(p_body);
	ERR_FAIL_NULL_V(ast_body, false);

	ast_body->traverse([counter, limit, &invariant, &r_is_read, p_control_flow](ASTNode *p_node, uint32_t p_depth, bool p_user_flag) {
		switch (p_node->node->type) {
			default:
				break;
			case Node::TYPE_OPERATOR: {
				const GDScriptParser::OperatorNode *op = (const OperatorNode *)p_node->node;
				switch (op->op) {
					default:
						break;
					// A lot of operators are disallowed for invariant, because they could have side effects.
					case OperatorNode::OP_CALL: {
						for (int n = 0; n < op->arguments.size(); n++) {
							if (helper_identifiers_match_by_name(limit, op->arguments[n])) {
								invariant = false;
								warning_unroll_unsupported(p_control_flow->node, "call using counter or limit.", op);
								return false;
							}
						}
					} break;
				}

				if (helper_is_operator_assign(op->op)) {
					if (op->arguments.size()) {
						Node *target = op->arguments[0];
						if (helper_identifiers_match_by_name(counter, target)) {
							invariant = false;
							warning_unroll_unsupported(p_control_flow->node, "assigning to counter.", op);
							return false;
						}
						if (helper_identifiers_match_by_name(limit, target)) {
							invariant = false;
							warning_unroll_unsupported(p_control_flow->node, "assigning to limit.", op);
							return false;
						}
					}
				}

				// If any of the arguments is the counter, don't allow.
				for (int n = 0; n < op->arguments.size(); n++) {
					if (helper_identifiers_match_by_name(counter, op->arguments[n])) {
						r_is_read = true;
					}
				}

			} break;
			case Node::TYPE_IDENTIFIER: {
				//const GDScriptParser::IdentifierNode *ident = (const IdentifierNode *)p_node->node;
				// If the counter is used within the loop, we would then have to add gdscript to keep the counter
				// up to date, and the unrolling would less of a performance win, so we won't unroll.
				if (helper_identifiers_match_by_name(counter, p_node->node)) {
					r_is_read = true;
				}
			} break;
		}
		return true;
	},
			0);

	return invariant;
}

bool GDScriptOptimizer::helper_identifiers_match_by_name(GDScriptParser::IdentifierNode *p_a, GDScriptParser::Node *p_b) {
	if (!p_a) {
		return false;
	}

	if (p_b->type != GDScriptParser::Node::TYPE_IDENTIFIER) {
		return false;
	}

	return (((GDScriptParser::IdentifierNode *)p_b)->name == p_a->name);
}

bool GDScriptOptimizer::helper_is_operator_unary(const GDScriptParser::OperatorNode::Operator p_op) {
	USING_GDSCRIPTPARSER

	switch (p_op) {
		default:
			break;
		case OperatorNode::OP_NEG:
		case OperatorNode::OP_POS:
		case OperatorNode::OP_NOT:
		case OperatorNode::OP_BIT_INVERT: {
			return true;
		} break;
	}
	return false;
}

bool GDScriptOptimizer::helper_is_operator_binary(const GDScriptParser::OperatorNode::Operator p_op) {
	USING_GDSCRIPTPARSER

	switch (p_op) {
		default:
			break;
		case OperatorNode::OP_SHIFT_LEFT:
		case OperatorNode::OP_SHIFT_RIGHT:
		case OperatorNode::OP_ADD:
		case OperatorNode::OP_SUB:
		case OperatorNode::OP_MUL:
		case OperatorNode::OP_DIV:
		case OperatorNode::OP_MOD:
		case OperatorNode::OP_BIT_AND:
		case OperatorNode::OP_BIT_OR:
		case OperatorNode::OP_BIT_XOR: {
			return true;
		} break;
	}
	return false;
}

bool GDScriptOptimizer::helper_is_operator_assign(const GDScriptParser::OperatorNode::Operator p_op) {
	USING_GDSCRIPTPARSER

	switch (p_op) {
		default:
			break;
		case OperatorNode::OP_INIT_ASSIGN:
		case OperatorNode::OP_ASSIGN_ADD:
		case OperatorNode::OP_ASSIGN_SUB:
		case OperatorNode::OP_ASSIGN_MUL:
		case OperatorNode::OP_ASSIGN_DIV:
		case OperatorNode::OP_ASSIGN_MOD:
		case OperatorNode::OP_ASSIGN_SHIFT_LEFT:
		case OperatorNode::OP_ASSIGN_SHIFT_RIGHT:
		case OperatorNode::OP_ASSIGN_BIT_AND:
		case OperatorNode::OP_ASSIGN_BIT_OR:
		case OperatorNode::OP_ASSIGN_BIT_XOR:
		case OperatorNode::OP_ASSIGN: {
			return true;
		} break;
	}
	return false;
}

void GDScriptOptimizer::helper_flush_pending_script_name_log() {
	if (Data::log_script_name_pending && Data::script_path) {
		Data::log_script_name_pending = false;
		GDOPT_LOG(1, String("script ") + *Data::script_path);
	}
}

String GDScriptOptimizer::helper_make_unique_name(String p_name, UniqueNameType p_type) {
	uint32_t counter = data.unique_identifier_counts[(uint32_t)p_type]++;

	// Special case for already underscored...
	if (p_name.begins_with("__")) {
		return p_name + "_" + itos(counter);
	}
	return String("__") + p_name + "_" + itos(counter);
}

void GDScriptOptimizer::unroll_loop(ASTNode *p_loop) {
	USING_GDSCRIPTPARSER

	ControlFlowNode *cf = (ControlFlowNode *)p_loop->node;

	// Only interested in loops marked as unroll in the script.
	if (!cf->_unroll && global_options.require_unroll_keyword) {
		return;
	}

	ASTNode *loop_parent = p_loop->parent;

	// The smaller the loop, the greater the benefit to unrolling,
	// and the more times we will unroll it.
	int32_t num_nodes = p_loop->count_nodes();

	const float max_nodes_per_unit = 200.0f;
	float f = max_nodes_per_unit / num_nodes;

	int32_t unit_size = Math::round(f);
	if (unit_size <= 1) {
		// If the loop is too big to be worth unrolling.
		warning_unroll_unsupported(loop_parent->node, "loop too long.");
		return;
	}
	// The unit_size is the number of unrolls in the large "unit".
	unit_size = MIN(unit_size, 8);

	///////////////////////////////////////

	// Find loop expression.
	if (cf->arguments.size() < 2) {
		// Not sure this ever happens...
		return;
	}

	bool counters_read = false;

	if (!unroll_are_counters_invariant(p_loop, cf->body, counters_read)) {
		return;
	}
	if (!unroll_are_counters_invariant(p_loop, cf->body_else, counters_read)) {
		return;
	}

	IdentifierNode *orig_counter = (IdentifierNode *)cf->arguments[0];
	Node *limit = cf->arguments[1];

	if (global_options.logging_level >= 2) {
		helper_flush_pending_script_name_log();
		print_line("\nunroll before\n*************\n");
		GDScriptReconstructor rec;
		String text;
		rec.output_branch(loop_parent->node, text);
		print_line(text);
	}

	Vector<Node *> *old_statements = nullptr;
	ASTNode *old_statement_holder = nullptr;
	int old_insert_pos = helper_find_ancestor_insert_statement(p_loop, &old_statements, &old_statement_holder);
	ERR_FAIL_COND(old_insert_pos == -1);

	// Remove the original for loop.
	old_statements->remove(old_insert_pos);

	String unroll_limit_name = helper_make_unique_name("unroll_limit", UniqueNameType::UNROLL_LIMIT);
	GDScriptParser::IdentifierNode *limit_ident = helper_declare_local_var(unroll_limit_name, limit, *old_statements, old_insert_pos);

	// Assign constant nil if not specified.
	GDOPT_ALLOC_NODE(ConstantNode, divisor, cf);
	divisor->value = Variant(unit_size);
	GDOPT_ALLOC_NODE(ConstantNode, constant_zero, cf);
	constant_zero->value = Variant(0);
	GDOPT_ALLOC_NODE(ConstantNode, constant_one, cf);
	constant_one->value = Variant(1);

	// If condition small loop limits...
	///////////////////////////////////////
	// If statement to take care of small loop limits.
	GDOPT_ALLOC_NODE(ControlFlowNode, if_small_limit, cf);

	GDOPT_ALLOC_NODE(BlockNode, if_small_limit_body, cf);
	GDOPT_ALLOC_NODE(BlockNode, main_block, cf);
	if_small_limit_body->statements.push_back(alloc_node<GDScriptParser::NewLineNode>(limit));
	if_small_limit_body->statements.push_back(cf);
	if_small_limit->body = main_block;
	if_small_limit->body_else = if_small_limit_body;

	Vector<Node *> *new_statements = &main_block->statements;
	new_statements->push_back(alloc_node<GDScriptParser::NewLineNode>(cf));
	int new_insert_pos = 1;

	// If limit condition to not use the unrolled version.
	GDOPT_ALLOC_NODE(OperatorNode, greater_than_limit, limit);
	greater_than_limit->op = OperatorNode::OP_GREATER_EQUAL;
	greater_than_limit->arguments.push_back(limit_ident);
	greater_than_limit->arguments.push_back(divisor);

	GDOPT_ALLOC_NODE(OperatorNode, is_an_int, limit);
	is_an_int->op = OperatorNode::OP_IS_BUILTIN;
	is_an_int->arguments.push_back(limit_ident);
	GDOPT_ALLOC_NODE(GDScriptParser::TypeNode, type_int, limit);
	type_int->vtype = Variant::INT;
	is_an_int->arguments.push_back(type_int);

	GDOPT_ALLOC_NODE(OperatorNode, if_small_limit_condition, cf);
	if_small_limit_condition->op = OperatorNode::OP_AND;
	if_small_limit_condition->arguments.push_back(is_an_int);
	if_small_limit_condition->arguments.push_back(greater_than_limit);

	if_small_limit->arguments.push_back(if_small_limit_condition);

	old_statements->insert(old_insert_pos++, if_small_limit);

	///////////////////////////////////////

	GDOPT_ALLOC_NODE(OperatorNode, divide_op, cf);
	divide_op->op = GDScriptParser::OperatorNode::OP_DIV;
	divide_op->arguments.push_back(limit_ident);
	divide_op->arguments.push_back(divisor);

	String unroll_quads_name = helper_make_unique_name("unroll_units", UniqueNameType::UNROLL_UNITS);
	GDScriptParser::IdentifierNode *quads_ident = helper_declare_local_var(unroll_quads_name, divide_op, *new_statements, new_insert_pos);

	GDOPT_ALLOC_NODE(OperatorNode, modulus_op, cf);
	modulus_op->op = GDScriptParser::OperatorNode::OP_MOD;
	modulus_op->arguments.push_back(limit_ident);
	modulus_op->arguments.push_back(divisor);

	String unroll_left_name = helper_make_unique_name("unroll_leftovers", UniqueNameType::UNROLL_LEFTOVERS);
	GDScriptParser::IdentifierNode *left_ident = helper_declare_local_var(unroll_left_name, modulus_op, *new_statements, new_insert_pos);

	// Counter, if needed (if the counter is read),
	// and a counter increment operation.
	GDScriptParser::IdentifierNode *counter_ident = nullptr;
	GDScriptParser::OperatorNode *op_counter_increment = nullptr;
	if (counters_read) {
		String counter_name = helper_make_unique_name("counter", UniqueNameType::COUNTER);
		counter_ident = helper_declare_local_var(counter_name, constant_zero, *new_statements, new_insert_pos);

		op_counter_increment = alloc_node<GDScriptParser::OperatorNode>(cf);
		op_counter_increment->op = OperatorNode::OP_ASSIGN_ADD;
		op_counter_increment->arguments.push_back(counter_ident);
		op_counter_increment->arguments.push_back(constant_one);
	}

	String unroll_counter_name = helper_make_unique_name("unroll_counter", UniqueNameType::UNROLL_COUNTER);
	GDOPT_ALLOC_NODE(IdentifierNode, unroll_counter, cf);
	unroll_counter->name = unroll_counter_name;
	///////////////////////////////////////

	GDOPT_ALLOC_NODE(ControlFlowNode, unit_for, cf);
	unit_for->cf_type = ControlFlowNode::CF_FOR;
	unit_for->arguments.push_back(unroll_counter);
	unit_for->arguments.push_back(quads_ident);

	new_statements->insert(new_insert_pos++, unit_for);

	GDOPT_ALLOC_NODE(BlockNode, unit_for_block, cf);

	unit_for->body = unit_for_block;

	// Changes are just changing the existing counter to the special counter.
	InlineInfo changes;
	InlineChange change;
	change.identifier_from = orig_counter->name;
	if (counter_ident) {
		change.identifier_to = counter_ident->name;
	}
	changes.changes.push_back(change);

	// Clone the whole for block in advance, and change any iterators to the new counters.
	BlockNode *cloned_for_block = (BlockNode *)duplicate_node_recursive(*cf->body, changes);
	if (op_counter_increment) {
		helper_statements_add_newline(cloned_for_block->statements);
		cloned_for_block->statements.push_back(op_counter_increment);
	}
	_remove_excessive_newlines_recursive(cloned_for_block);

	// Copy all the statements from the original for loop to unit for.
	const Vector<Node *> &source_statements = cloned_for_block->statements;

	for (int32_t u = 0; u < unit_size; u++) {
		for (int n = 0; n < source_statements.size(); n++) {
			unit_for_block->statements.push_back(source_statements[n]);
		}
	}

	// Make sure there is a newline.
	helper_statements_add_newline(unit_for_block->statements);

	GDOPT_ALLOC_NODE(ControlFlowNode, unit_for2, cf);
	unit_for2->cf_type = ControlFlowNode::CF_FOR;
	unit_for2->arguments.push_back(unroll_counter);
	unit_for2->arguments.push_back(left_ident);

	new_statements->insert(new_insert_pos++, unit_for2);
	unit_for2->body = cloned_for_block;

	_remove_excessive_newlines_recursive(loop_parent->node);
	if (global_options.logging_level >= 2) {
		print_line("\nunroll result\n*************\n");
		GDScriptReconstructor rec;
		String text;
		rec.output_branch(loop_parent->node, text);
		print_line(text);
	}

#if 0
	if (global_options.logging_level >= 4) {
		print_line("\nunroll result AST\n*****************\n");

		ASTNode *debug_tree = memnew(ASTNode(loop_parent->node, nullptr));
		debug_tree->debug_print();
		memdelete(debug_tree);
	}
#endif
}

void GDScriptOptimizer::helper_statements_add_newline(Vector<GDScriptParser::Node *> &r_statements, int32_t p_pos) {
	USING_GDSCRIPTPARSER

	if (p_pos == -1) {
		Node *prev = nullptr;
		if (r_statements.size()) {
			prev = r_statements[r_statements.size() - 1];
			if (prev->type == Node::TYPE_NEWLINE) {
				return;
			}
		}
		GDOPT_ALLOC_NODE(NewLineNode, newline, prev);
		r_statements.push_back(newline);
	} else {
		ERR_FAIL_COND(p_pos > r_statements.size());

		const Node *location_copy = (p_pos > 0) ? r_statements[p_pos - 1] : nullptr;
		GDOPT_ALLOC_NODE(NewLineNode, newline, location_copy);
		r_statements.insert(p_pos, newline);
	}
}

void GDScriptOptimizer::helper_find_all_types(LocalVector<ASTNode *> &r_found, ASTNode *p_root, GDScriptParser::Node::Type p_type, GDScriptParser::ControlFlowNode::CFType p_cf_type) {
	p_root->traverse([&r_found, p_type, p_cf_type](ASTNode *p_node, uint32_t p_depth, bool p_user_flag) {
		if (p_node->node->type == p_type) {
			// Special case for control flows...
			if (p_type == GDScriptParser::Node::TYPE_CONTROL_FLOW) {
				if (((GDScriptParser::ControlFlowNode *)p_node->node)->cf_type == p_cf_type) {
					r_found.push_back(p_node);
				}
			} else {
				r_found.push_back(p_node);
			}
		}
		return true;
	},
			0);
}

bool GDScriptOptimizer::helper_is_static_control_flow_true(const GDScriptParser::ClassNode *p_curr_class, const GDScriptParser::ControlFlowNode *p_cf, bool &r_is_true) {
	USING_GDSCRIPTPARSER

	if (p_cf->arguments.size() == 1) {
		GDScriptParser::Node *arg = p_cf->arguments[0];

		if (helper_get_constant_bool(p_curr_class, arg, r_is_true)) {
			return true;
		}

		if (p_cf->arguments[0]->type == Node::TYPE_OPERATOR) {
			GDScriptParser::OperatorNode *op = (OperatorNode *)arg;

			if (op->arguments.size() == 1) {
				switch (op->op) {
					default:
						break;
					case OperatorNode::OP_NOT: {
						if (helper_get_constant_bool(p_curr_class, op->arguments[0], r_is_true)) {
							return true;
						}

					} break;
				}

			} else if (op->arguments.size() == 2) {
				Variant val0;
				Variant val1;
				if (!helper_get_constant_value(p_curr_class, op->arguments[0], val0) || !helper_get_constant_value(p_curr_class, op->arguments[1], val1)) {
					return false;
				}

				if (helper_do_variant_comparison(op->op, val0, val1, r_is_true)) {
					return true;
				}
			}
		}

		return false;
	}

	return false;
}

bool GDScriptOptimizer::helper_get_constant_value(const GDScriptParser::ClassNode *p_curr_class, const GDScriptParser::Node *p_node, Variant &r_value) {
	USING_GDSCRIPTPARSER
	switch (p_node->type) {
		default:
			break;
		case Node::TYPE_CONSTANT: {
			GDScriptParser::ConstantNode *const_ = (GDScriptParser::ConstantNode *)p_node;
			r_value = const_->value;
			return true;
		} break;
		case Node::TYPE_IDENTIFIER: {
			GDScriptParser::IdentifierNode *ident = (GDScriptParser::IdentifierNode *)p_node;

			// Progressively check for the constant in this class, and any enclosing classes
			// (as constants will still be accessible in owning classes).
			const GDScriptParser::ClassNode *_class = p_curr_class;

			while (_class) {
				if (_class->constant_expressions.has(ident->name)) {
					Node *cn_exp = _class->constant_expressions[ident->name].expression;
					if (cn_exp->type == Node::TYPE_CONSTANT) {
						ConstantNode *const_ = static_cast<ConstantNode *>(cn_exp);
						r_value = const_->value;
						return true;
					} else {
						return false;
					}
				}

				_class = _class->owner;
			}

		} break;
	}

	return false;
}

bool GDScriptOptimizer::helper_get_constant_bool(const GDScriptParser::ClassNode *p_curr_class, const GDScriptParser::Node *p_node, bool &r_is_true) {
	Variant val;
	if (helper_get_constant_value(p_curr_class, p_node, val)) {
		r_is_true = val.booleanize();
		return true;
	}
	return false;
}

bool GDScriptOptimizer::contract_is_binary_expression_contractable(GDScriptParser::ClassNode *p_curr_class, GDScriptParser::Node *p_node, bool p_add_or_multiply, const GDScriptParser::ConstantNode **r_child_constant, GDScriptParser::Node **r_child_non_constant) {
	USING_GDSCRIPTPARSER
	if (p_node->type != Node::TYPE_OPERATOR) {
		return false;
	}
	GDScriptParser::OperatorNode *op = (GDScriptParser::OperatorNode *)p_node;
	if (op->arguments.size() != 2) {
		return false;
	}

	// Make sure this math expression is compatible with the parent for reduction.
	switch (op->op) {
		default: {
			return false;
		} break;
		case OperatorNode::OP_SUB:
		case OperatorNode::OP_ADD: {
			if (!p_add_or_multiply) {
				return false;
			}
		} break;
		case OperatorNode::OP_MUL:
		case OperatorNode::OP_DIV: {
			if (p_add_or_multiply) {
				return false;
			}
		} break;
	}

	for (int a = 0; a < op->arguments.size(); a++) {
		Node *arg = op->arguments[a];

		if (arg->type == Node::TYPE_CONSTANT) {
			*r_child_constant = (const ConstantNode *)arg;
			*r_child_non_constant = op->arguments[1 - a];
			return true;
		}
	}

	return false;
}

bool GDScriptOptimizer::contract_try_expression_two_constants(ASTNode *p_ast, const GDScriptParser::ConstantNode *p_const_a, const GDScriptParser::ConstantNode *p_const_b) {
	USING_GDSCRIPTPARSER
	GDScriptParser::OperatorNode *op = (GDScriptParser::OperatorNode *)p_ast->node;

	Variant var_result;

	if (!helper_do_variant_math(op->op, p_const_a->value, p_const_b->value, var_result)) {
		return false;
	}

	// Replace our constant with the new constant...
	GDOPT_ALLOC_NODE(ConstantNode, new_constant, op);
	new_constant->value = var_result;
	new_constant->datatype.kind = GDScriptParser::DataType::BUILTIN;
	new_constant->datatype.builtin_type = var_result.get_type();

	GDOPT_LOG(2, "constant pair folding " + node_to_string(op));
	String log_text;
	if (global_options.logging_level >= 3) {
		GDScriptReconstructor rec;
		rec.output_branch(p_ast->parent->node, log_text);
	}

	helper_node_exchange_child(*p_ast->parent->node, op, new_constant);

	if (global_options.logging_level >= 3) {
		GDScriptReconstructor rec;
		log_text += GDScriptReconstructor::draw_bold("   >>>   ");
		rec.output_branch(p_ast->parent->node, log_text);
		print_line("\t" + log_text);
	}

	// Mark operator as done.
	p_ast->flood_fill(1);

	return true;
}

bool GDScriptOptimizer::contract_try_paired_expression_binary(ASTNode *p_ast, GDScriptParser::ClassNode *p_curr_class, GDScriptParser::OperatorNode *child_op_node_a, GDScriptParser::OperatorNode *child_op_node_b) {
	USING_GDSCRIPTPARSER
	GDScriptParser::OperatorNode *parent_op_node = (GDScriptParser::OperatorNode *)p_ast->node;
	OperatorNode::Operator parent_op = parent_op_node->op;
	OperatorNode::Operator child_op_a = child_op_node_a->op;
	OperatorNode::Operator child_op_b = child_op_node_b->op;

	bool add_or_multiply_parent = (parent_op == OperatorNode::OP_ADD) || (parent_op == OperatorNode::OP_SUB);
	bool add_or_multiply_a = (child_op_a == OperatorNode::OP_ADD) || (child_op_a == OperatorNode::OP_SUB);
	bool add_or_multiply_b = (child_op_b == OperatorNode::OP_ADD) || (child_op_b == OperatorNode::OP_SUB);

	// Must be the same type of math expression throughout.
	if ((add_or_multiply_parent != add_or_multiply_a) || (add_or_multiply_parent != add_or_multiply_b)) {
		return false;
	}

	if (!add_or_multiply_parent) {
		// Multiply and divide not supported, only addition.
		return false;
	}

	const GDScriptParser::ConstantNode *const_a = nullptr;
	const GDScriptParser::ConstantNode *const_b = nullptr;
	GDScriptParser::Node *non_const_a = nullptr;
	GDScriptParser::Node *non_const_b = nullptr;

	if (!contract_is_binary_expression_contractable(p_curr_class, child_op_node_a, add_or_multiply_a, &const_a, &non_const_a)) {
		ERR_FAIL_V(false);
	}
	if (!contract_is_binary_expression_contractable(p_curr_class, child_op_node_b, add_or_multiply_b, &const_b, &non_const_b)) {
		ERR_FAIL_V(false);
	}

	bool const_a_is_first_arg = child_op_node_a->arguments[0] == const_a;
	bool const_b_is_first_arg = child_op_node_b->arguments[0] == const_b;

	// If we got to here, we can do the contraction.
	Variant var_result;
	String log_text;

	bool a_is_add = child_op_a == OperatorNode::OP_ADD;
	bool b_is_add = child_op_b == OperatorNode::OP_ADD;
	bool parent_is_add = parent_op == OperatorNode::OP_ADD;

	// Polarity of A.
	bool const_a_positive = true;
	bool non_const_a_positive = true;
	if (!a_is_add) {
		const_a_positive = const_a_is_first_arg;
		non_const_a_positive = !const_a_is_first_arg;
	}

	// Polarity of B.
	bool const_b_positive = true;
	bool non_const_b_positive = true;
	if (!b_is_add) {
		const_b_positive = const_b_is_first_arg;
		non_const_b_positive = !const_b_is_first_arg;
	}

	// Polarity of parent.
	if (!parent_is_add) {
		const_b_positive = !const_b_positive;
		non_const_b_positive = !non_const_b_positive;
	}

	OperatorNode::Operator math_op = const_a_positive == const_b_positive ? OperatorNode::OP_ADD : OperatorNode::OP_SUB;

	if (!helper_do_variant_math(math_op, const_a->value, const_b->value, var_result)) {
		return false;
	}

	// Mark child operator as done.
	ASTNode *ast_child = p_ast->find_child(child_op_node_b);
	if (ast_child) {
		ast_child->flood_id = 1;
	}

	if (!const_a_positive) {
		helper_flip_variant_polarity(var_result);
	}

	// Replace our constant with the new constant...
	GDOPT_ALLOC_NODE(ConstantNode, new_constant, const_a);
	new_constant->value = var_result;
	new_constant->datatype = const_a->datatype;

	GDOPT_LOG(2, "constant folding " + node_to_string(parent_op_node));

	if (global_options.logging_level >= 3) {
		GDScriptReconstructor rec;
		rec.output_branch(parent_op_node, log_text);
	}

	// The new operation will be:
	// (new_constant += non_const_a ) +- non_const_b
	// The new_constant will always be positive, but both non-consts can potentially be negative.
	parent_op_node->op = non_const_b_positive ? OperatorNode::OP_ADD : OperatorNode::OP_SUB;
	parent_op_node->arguments.set(1, non_const_b);

	child_op_node_a->op = non_const_a_positive ? OperatorNode::OP_ADD : OperatorNode::OP_SUB;

	child_op_node_a->arguments.set(0, new_constant);
	child_op_node_a->arguments.set(1, non_const_a);

	if (global_options.logging_level >= 3) {
		GDScriptReconstructor rec;
		log_text += GDScriptReconstructor::draw_bold("   >>>   ");
		rec.output_branch(parent_op_node, log_text);
		print_line("\t" + log_text);
	}

	return true;
}

bool GDScriptOptimizer::contract_try_expression_binary(ASTNode *p_ast, GDScriptParser::ClassNode *p_curr_class) {
	USING_GDSCRIPTPARSER
	GDScriptParser::OperatorNode *op = (GDScriptParser::OperatorNode *)p_ast->node;
	DEV_ASSERT(op->arguments.size() == 2);

	bool add_or_multiply = false;

	switch (op->op) {
		default: {
			return false;
		} break;
		case OperatorNode::OP_SUB:
		case OperatorNode::OP_ADD: {
			add_or_multiply = true;
		} break;
		case OperatorNode::OP_MUL:
		case OperatorNode::OP_DIV: {
			add_or_multiply = false;
		} break;
	}

	// If either of the children are e.g. IDENT + CONSTANT, and the other child is + CONSTANT,
	// then we can reduce.
	int const_arg = -1;
	OperatorNode *child_op_node_a = nullptr;
	OperatorNode::Operator child_op_a = OperatorNode::OP_CALL;

	// In this routine we will usually contract e.g. a + (func() + b),
	// but we also encounter the situation (func() + a) + (func() + b)
	// and want to recognise two contractable child operators.
	OperatorNode *child_op_node_b = nullptr;

	const ConstantNode *const_b = nullptr;
	Node *var_b = nullptr;
	bool const_b_is_first_arg = false;

	const ConstantNode *const_a = nullptr;
	bool const_a_is_first_arg = false;

	for (int a = 0; a < op->arguments.size(); a++) {
		Node *arg = op->arguments[a];

		const ConstantNode *arg_constant = nullptr;
		if (contract_is_node_const_identifier_or_constant(p_curr_class, arg, &arg_constant)) {
			// Special case, combining two constants...
			if (const_arg != -1) {
				return contract_try_expression_two_constants(p_ast, const_a, arg_constant);
			}

			const_arg = a;
			const_a = arg_constant;

			const_a_is_first_arg = a == 0;
		} else {
			if (!contract_is_binary_expression_contractable(p_curr_class, arg, add_or_multiply, &const_b, &var_b)) {
				return false;
			} else {
				if (!child_op_node_a) {
					child_op_node_a = (OperatorNode *)arg;
					child_op_a = child_op_node_a->op;
					const_b_is_first_arg = child_op_node_a->arguments[0] == const_b;
				} else {
					child_op_node_b = (OperatorNode *)arg;
				}
			}
		}
	}

	// A contractable child may be paired with a non-const argument,
	// we are not attempting to deal with this
	// (and hence are missing some cases, for instance (a+2) + (a+2) will not contract,
	// whereas 2 + (a+2) will).
	if (!const_a || !const_b) {
		return contract_try_paired_expression_binary(p_ast, p_curr_class, child_op_node_a, child_op_node_b);
	}

	// If we got to here, we can do the contraction.
	Variant var_result;
	OperatorNode::Operator new_op = op->op;
	int new_const_pos = 0;

#define GDFOLD_BIN(OP, CONST0, CONST1, NEW_OP, NEW_CONST_POS)                                  \
	if (!helper_do_variant_math(OperatorNode::OP, CONST0->value, CONST1->value, var_result)) { \
		return false;                                                                          \
	}                                                                                          \
	new_op = OperatorNode::NEW_OP;                                                             \
	new_const_pos = NEW_CONST_POS;

	if (add_or_multiply) {
		bool outer_add = op->op == OperatorNode::OP_ADD;
		bool inner_add = child_op_a == OperatorNode::OP_ADD;

		// Polarity of outer.
		bool const_a_positive = true;
		if (!outer_add && !const_a_is_first_arg) {
			const_a_positive = false;
		}

		// Polarity of inner.
		bool const_b_positive = true;
		bool var_b_positive = true;
		if (!inner_add) {
			const_b_positive = const_b_is_first_arg;
			var_b_positive = !const_b_is_first_arg;
		}

		// Now reverse polarity of inner if we are subtracting and second arg.
		if (!outer_add && const_a_is_first_arg) {
			const_b_positive = !const_b_positive;
			var_b_positive = !var_b_positive;
		}

		OperatorNode::Operator math_op = const_a_positive == const_b_positive ? OperatorNode::OP_ADD : OperatorNode::OP_SUB;

		if (!helper_do_variant_math(math_op, const_a->value, const_b->value, var_result)) {
			return false;
		}

		new_op = var_b_positive ? OperatorNode::OP_ADD : OperatorNode::OP_SUB;

		if (!const_a_positive) {
			helper_flip_variant_polarity(var_result);
		}

#if 0
		// For easy reading, if the result is A - -B, change to A + B
		// A + -B becomes A - B
		if (!helper_make_variant_positive(var_result)) {
			// If we had to reverse the polarity of the variant...
			new_op = new_op == OperatorNode::OP_ADD ? OperatorNode::OP_SUB : OperatorNode::OP_ADD;
		}
#endif
	} else {
		// Simplest case, two multiplies.
		if (op->op == OperatorNode::OP_MUL && child_op_a == OperatorNode::OP_MUL) {
			// Multiply twice is the same as multiplying the operands.
			GDFOLD_BIN(OP_MUL, const_a, const_b, OP_MUL, 0);
		} else {
			// 16 possible cases of A OP B OP C (with brackets).

			// First most important branch on whether "(COMPLEX) OP A" or "A OP (COMPLEX)"
			if (const_a_is_first_arg) {
				// "A OP (COMPLEX)"
				if (op->op == OperatorNode::OP_MUL) {
					// COMPLEX MUST BE DIVIDE
					DEV_ASSERT(child_op_a == OperatorNode::OP_DIV);

					// Branch on whether the complex is "B OP CONST" or "CONST OP B"
					if (!const_b_is_first_arg) {
						// A * (VAR / B)
						GDFOLD_BIN(OP_DIV, const_a, const_b, OP_MUL, 0);
					} else {
						// A * (B / VAR)
						GDFOLD_BIN(OP_MUL, const_a, const_b, OP_DIV, 0);
					}
				} else {
					// A is DIVIDE, COMPLEX CAN BE MULT OR DIVIDE.
					DEV_ASSERT(op->op == OperatorNode::OP_DIV);

					if (!const_b_is_first_arg) {
						if (child_op_a == OperatorNode::OP_MUL) {
							// A / (VAR * B)
							GDFOLD_BIN(OP_DIV, const_a, const_b, OP_DIV, 0);
						} else {
							// A / (VAR / B)
							GDFOLD_BIN(OP_MUL, const_a, const_b, OP_DIV, 0);
						}
					} else {
						if (child_op_a == OperatorNode::OP_MUL) {
							// A / (B * VAR)
							GDFOLD_BIN(OP_DIV, const_a, const_b, OP_DIV, 0);
						} else {
							// A / (B / VAR)
							GDFOLD_BIN(OP_DIV, const_a, const_b, OP_MUL, 0);
						}
					}
				}
			} else {
				// (COMPLEX) OP A
				if (op->op == OperatorNode::OP_MUL) {
					// COMPLEX MUST BE DIVIDE
					DEV_ASSERT(child_op_a == OperatorNode::OP_DIV);

					// Branch on whether the complex is "B OP CONST" or "CONST OP B"
					if (!const_b_is_first_arg) {
						// (VAR / B) * A
						GDFOLD_BIN(OP_DIV, const_a, const_b, OP_MUL, 0);
					} else {
						// (B / VAR) * A
						GDFOLD_BIN(OP_MUL, const_a, const_b, OP_DIV, 0);
					}
				} else {
					// A is DIVIDE, COMPLEX CAN BE MULT OR DIVIDE.
					DEV_ASSERT(op->op == OperatorNode::OP_DIV);

					if (!const_b_is_first_arg) {
						if (child_op_a == OperatorNode::OP_MUL) {
							// (VAR * B) / A
							GDFOLD_BIN(OP_DIV, const_b, const_a, OP_MUL, 0);
						} else {
							// (VAR / B) / A
							GDFOLD_BIN(OP_MUL, const_a, const_b, OP_DIV, 1);
						}
					} else {
						if (child_op_a == OperatorNode::OP_MUL) {
							// (B * VAR) / A
							GDFOLD_BIN(OP_DIV, const_b, const_a, OP_MUL, 0);

						} else {
							// (B / VAR) / A
							GDFOLD_BIN(OP_DIV, const_b, const_a, OP_DIV, 0);
						}
					}
				}
			}
		}
	}

#undef GDFOLD_BIN

	// Replace our constant with the new constant...
	GDOPT_ALLOC_NODE(ConstantNode, new_constant, const_a);
	new_constant->value = var_result;
	new_constant->datatype = const_a->datatype;

	GDOPT_LOG(2, "constant folding " + node_to_string(op));

	String log_text;
	if (global_options.logging_level >= 3) {
		GDScriptReconstructor rec;
		rec.output_branch(op, log_text);
	}

	// We may have reversed the operation.
	op->op = new_op;

	if (add_or_multiply) {
		op->arguments.set(0, new_constant);
		op->arguments.set(1, var_b);
	} else {
		op->arguments.set(new_const_pos, new_constant);
		op->arguments.set(1 - new_const_pos, var_b);
	}

	if (global_options.logging_level >= 3) {
		GDScriptReconstructor rec;
		log_text += GDScriptReconstructor::draw_bold("   >>>   ");
		rec.output_branch(op, log_text);
		print_line("\t" + log_text);
	}

	// Mark child operator as done.
	ASTNode *ast_child = p_ast->find_child(child_op_node_a);
	if (ast_child) {
		ast_child->flood_id = 1;
	}

	return true;
}

bool GDScriptOptimizer::helper_flip_variant_polarity(Variant &r_var) const {
	switch (r_var.get_type()) {
		default:
			break;
		case Variant::INT: {
			r_var = -(int64_t)r_var;
			return true;
		} break;
		case Variant::REAL: {
			r_var = -(double)r_var;
			return true;
		} break;
	}
	return false;
}

bool GDScriptOptimizer::helper_make_variant_positive(Variant &r_var) const {
	switch (r_var.get_type()) {
		default:
			break;
		case Variant::INT: {
			if (((int64_t)r_var) >= 0) {
				return true;
			}
			r_var = -(int64_t)r_var;
			return false;
		} break;
		case Variant::REAL: {
			if (((double)r_var) >= 0) {
				return true;
			}
			r_var = -(double)r_var;
			return false;
		} break;
	}
	return true;
}

// Returns false if the types are not compatible, or the bool result if supported.
bool GDScriptOptimizer::helper_do_variant_comparison(GDScriptParser::OperatorNode::Operator p_op, const Variant &p_a, const Variant &p_b, bool &r_is_true) {
	USING_GDSCRIPTPARSER

	Variant::Type promoted_type = p_a.get_type();
	if ((promoted_type != Variant::BOOL) && (promoted_type != Variant::INT) && (promoted_type != Variant::REAL)) {
		return false;
	}

	bool use_real = false;
	if (promoted_type == Variant::REAL) {
		use_real = true;
	}

	if (p_b.get_type() == Variant::REAL) {
		use_real = true;
	}

	double r0 = (double)p_a;
	double r1 = (double)p_b;
	int64_t i0 = (int64_t)p_a;
	int64_t i1 = (int64_t)p_b;

#define GODOT_VARIANT_COMPARISON_OP_CASE(OP_CODE, OPERATOR) \
	case OP_CODE: {                                         \
		if (use_real) {                                     \
			r_is_true = r0 OPERATOR r1;                     \
		} else {                                            \
			r_is_true = i0 OPERATOR i1;                     \
		}                                                   \
		return true;                                        \
	} break;

	switch (p_op) {
		default:
			break;
			GODOT_VARIANT_COMPARISON_OP_CASE(OperatorNode::OP_EQUAL, ==)
			GODOT_VARIANT_COMPARISON_OP_CASE(OperatorNode::OP_NOT_EQUAL, !=)
			GODOT_VARIANT_COMPARISON_OP_CASE(OperatorNode::OP_GREATER, >)
			GODOT_VARIANT_COMPARISON_OP_CASE(OperatorNode::OP_GREATER_EQUAL, >=)
			GODOT_VARIANT_COMPARISON_OP_CASE(OperatorNode::OP_LESS, <)
			GODOT_VARIANT_COMPARISON_OP_CASE(OperatorNode::OP_LESS_EQUAL, <=)
		case OperatorNode::OP_AND: {
			r_is_true = p_a.booleanize() && p_b.booleanize();
			return true;
		} break;
		case OperatorNode::OP_OR: {
			r_is_true = p_a.booleanize() || p_b.booleanize();
			return true;
		} break;
	}

#undef GODOT_VARIANT_COMPARISON_OP_CASE

	return false;
}

// Returns false if the types are not compatible, or the math expression not supported.
bool GDScriptOptimizer::helper_do_variant_math(GDScriptParser::OperatorNode::Operator p_op, const Variant &p_a, const Variant &p_b, Variant &r_result) const {
	USING_GDSCRIPTPARSER

	Variant::Type promoted_type = p_a.get_type();
	if ((promoted_type != Variant::INT) && (promoted_type != Variant::REAL)) {
		// NYI
		return false;
	}

	if (p_b.get_type() == Variant::REAL) {
		promoted_type = Variant::REAL;
	}

	//	if (p_a.get_type() != p_b.get_type()) {
	//		return false;
	//	}

#define GODOT_OPT_VARIANT_MATH_OP_CASE(OP_TYPE, OPERATOR)      \
	case OperatorNode::OP_TYPE: {                              \
		switch (promoted_type) {                               \
			default: {                                         \
				return false;                                  \
			} break;                                           \
			case Variant::INT: {                               \
				r_result = (int64_t)p_a OPERATOR(int64_t) p_b; \
				return true;                                   \
			} break;                                           \
			case Variant::REAL: {                              \
				r_result = (double)p_a OPERATOR(double) p_b;   \
				return true;                                   \
			} break;                                           \
		}                                                      \
	} break;

	// Special check for divide by zero.
	if (p_op == OperatorNode::OP_DIV) {
		if ((p_b.get_type() == Variant::INT) && ((int64_t)p_b == 0)) {
			WARN_PRINT("Divide by zero in script");
			return false;
		}
		if ((p_b.get_type() == Variant::REAL) && ((double)p_b == 0)) {
			WARN_PRINT("Divide by zero in script");
			return false;
		}

		// Do not allow integer divisions if the result would not be exact.
		if (promoted_type == Variant::INT) {
			if (((int64_t)p_a % (int64_t)p_b) != 0) {
				return false;
			}
		}
	}

	switch (p_op) {
		default: {
			return false;
		} break;
			GODOT_OPT_VARIANT_MATH_OP_CASE(OP_ADD, +)
			GODOT_OPT_VARIANT_MATH_OP_CASE(OP_SUB, -)
			GODOT_OPT_VARIANT_MATH_OP_CASE(OP_MUL, *)
			GODOT_OPT_VARIANT_MATH_OP_CASE(OP_DIV, /)
	}

#undef GODOT_OPT_VARIANT_MATH_OP_CASE
	return true;
}

bool GDScriptOptimizer::contract_try_expression_unary(ASTNode *p_ast, GDScriptParser::ClassNode *p_curr_class) {
	USING_GDSCRIPTPARSER
	GDScriptParser::OperatorNode *op = (GDScriptParser::OperatorNode *)p_ast->node;
	DEV_ASSERT(op->arguments.size() == 1);

	// Unary operators only need checking once.
	p_ast->flood_fill(1);

	Node *arg = op->arguments[0];
#if 0
	if (arg->type == Node::TYPE_CONSTANT) {
		GDOPT_ALLOC_NODE(ConstantNode, result, arg);

		// Mark this ast node as done, whether the exchange works or not.
		p_ast->flood_fill(1);
		ERR_FAIL_NULL_V(p_ast->parent->node, false);
		GDOPT_LOG(3, "contracting operator " + node_to_string(p_ast->node) + ".");
		return helper_node_exchange_child(*p_ast->parent->node, arg, result);
	}
#endif
	if (arg->type == Node::TYPE_IDENTIFIER) {
		GDScriptParser::IdentifierNode *ident = (GDScriptParser::IdentifierNode *)arg;
		if (p_curr_class && p_curr_class->constant_expressions.has(ident->name)) {
			Node *cn_exp = p_curr_class->constant_expressions[ident->name].expression;
			if (cn_exp->type == Node::TYPE_CONSTANT) {
				ConstantNode *con = static_cast<ConstantNode *>(cn_exp);
				GDScriptParser::DataType dt = con->get_datatype();

				Variant result_var;
				if (dt.has_type && dt.kind == GDScriptParser::DataType::BUILTIN) {
					switch (dt.builtin_type) {
						default: {
							return false;
						} break;
						case Variant::REAL: {
							switch (op->op) {
								default: {
									return false;
								} break;
								case OperatorNode::OP_NEG: {
									result_var = -(double)con->value;
								} break;
								case OperatorNode::OP_POS: {
									result_var = con->value;
								} break;
							}
						} break;
						case Variant::INT: {
							switch (op->op) {
								default: {
									return false;
								} break;
								case OperatorNode::OP_NEG: {
									result_var = -(int64_t)con->value;
								} break;
								case OperatorNode::OP_POS: {
									result_var = con->value;
								} break;
								case OperatorNode::OP_NOT: {
									result_var = !con->value.booleanize();
								} break;
							}
						} break;
						case Variant::VECTOR3: {
							switch (op->op) {
								default: {
									return false;
								} break;
								case OperatorNode::OP_NEG: {
									result_var = -(Vector3)con->value;
								} break;
							}
						} break;
						case Variant::VECTOR2: {
							switch (op->op) {
								default: {
									return false;
								} break;
								case OperatorNode::OP_NEG: {
									result_var = -(Vector2)con->value;
								} break;
							}
						} break;
					}
				}

				GDOPT_ALLOC_NODE(ConstantNode, result, arg);
				result->value = result_var;
				result->datatype = dt;
				result->datatype.builtin_type = result_var.get_type();
				GDOPT_LOG(3, "constant unary folding " + node_to_string(p_ast->node));
				return helper_node_exchange_child(*p_ast->parent->node, p_ast->node, result);
			}
		}
	}
	return false;
}

bool GDScriptOptimizer::contract_try_expression(ASTNode *p_ast, GDScriptParser::ClassNode *p_curr_class) {
	USING_GDSCRIPTPARSER
	GDScriptParser::OperatorNode *op = (GDScriptParser::OperatorNode *)p_ast->node;
	if (helper_is_operator_unary(op->op)) {
		return contract_try_expression_unary(p_ast, p_curr_class);
	} else if (helper_is_operator_binary(op->op)) {
		return contract_try_expression_binary(p_ast, p_curr_class);
	}

	return false;
}

bool GDScriptOptimizer::contract_is_node_const_identifier_or_constant(const GDScriptParser::ClassNode *p_curr_class, const GDScriptParser::Node *p_node, const GDScriptParser::ConstantNode **r_constant_node) const {
	USING_GDSCRIPTPARSER

	*r_constant_node = nullptr;

	if (p_node->type == Node::TYPE_CONSTANT) {
		*r_constant_node = static_cast<const ConstantNode *>(p_node);
		return true;
	}

	if (p_node->type == Node::TYPE_IDENTIFIER) {
		GDScriptParser::IdentifierNode *ident = (GDScriptParser::IdentifierNode *)p_node;
		if (p_curr_class && p_curr_class->constant_expressions.has(ident->name)) {
			Node *cn_exp = p_curr_class->constant_expressions[ident->name].expression;
			if (cn_exp->type == Node::TYPE_CONSTANT) {
				*r_constant_node = static_cast<const ConstantNode *>(cn_exp);
				return true;
			}
		}
	}

	return false;
}

void GDScriptOptimizer::constant_fold_expressions() {
	USING_GDSCRIPTPARSER
	ASTNode *tree = memnew(ASTNode(data.get_root_class().root, nullptr));

	struct Expr {
		ASTNode *ast_op = nullptr;
		ClassNode *curr_class = nullptr;
	};

	LocalVector<Expr> found;

	tree->traverse([&found](ASTNode *p_node, uint32_t p_depth, bool p_user_flag) {
		static ClassNode *curr_class = nullptr;

		switch (p_node->node->type) {
			default:
				break;
			case Node::TYPE_CLASS: {
				curr_class = (ClassNode *)p_node->node;
			};
			case Node::TYPE_OPERATOR: {
				Expr e;
				e.ast_op = p_node;
				e.curr_class = curr_class;
				found.push_back(e);
			} break;
		}
		return true;
	},
			0);

	bool any_contracted = true;
	while (any_contracted) {
		any_contracted = false;
		for (uint32_t n = 0; n < found.size(); n++) {
			if (!found[n].ast_op->flood_id) {
				if (contract_try_expression(found[n].ast_op, found[n].curr_class)) {
					any_contracted = true;
				}
			}
		}
	}

	memdelete(tree);
}

void GDScriptOptimizer::unused_remove_expressions() {
	USING_GDSCRIPTPARSER
	ASTNode *tree = memnew(ASTNode(data.get_root_class().root, nullptr));

	struct IfStatement {
		ASTNode *cf = nullptr;
		const ClassNode *class_node = nullptr;
	};

	LocalVector<IfStatement> if_statements;

	tree->traverse([&if_statements](ASTNode *p_node, uint32_t p_depth, bool p_user_flag) {
		static const GDScriptParser::ClassNode *curr_class = nullptr;

		switch (p_node->node->type) {
			default:
				break;
			case Node::TYPE_CLASS: {
				curr_class = (const ClassNode *)p_node->node;
			} break;
			case Node::TYPE_CONTROL_FLOW: {
				GDScriptParser::ControlFlowNode *cf = (ControlFlowNode *)p_node->node;
				if (cf->cf_type == ControlFlowNode::CF_IF) {
					IfStatement ifs;
					ifs.cf = p_node;
					ifs.class_node = curr_class;
					if_statements.push_back(ifs);
				}
			} break;
		}
		return true;
	},
			0);

	// Work out which if statements are constant.
	LocalVector<ASTNode *> if_const_true;
	LocalVector<ASTNode *> if_const_false;

	for (uint32_t n = 0; n < if_statements.size(); n++) {
		IfStatement &ifs = if_statements[n];

		GDScriptParser::ControlFlowNode *cf = (GDScriptParser::ControlFlowNode *)ifs.cf->node;

		bool is_true = false;
		if (helper_is_static_control_flow_true(ifs.class_node, cf, is_true)) {
			if (is_true) {
				if_const_true.push_back(ifs.cf);
			} else {
				if_const_false.push_back(ifs.cf);
			}
		}
	}

	bool removed_any_if_clauses = false;

	for (uint32_t n = 0; n < if_const_false.size(); n++) {
		ASTNode *ast_if = if_const_false[n];
		removed_any_if_clauses = true;

		GDScriptParser::ControlFlowNode *cf = (ControlFlowNode *)ast_if->node;

		if (global_options.logging_level >= 2) {
			helper_flush_pending_script_name_log();
			String text = "\nif const false before\n*********************\n";
			GDScriptReconstructor rec;
			rec.output_branch(cf, text, cf->arguments[0] ? cf->arguments[0] : nullptr);
			print_line(text + "\n");
		}

		if (cf->body) {
			// We can just clear the body.
			// This will get removed later.
			cf->body->statements.clear();
		}

		// Move any else clause to an inline block.
		if (cf->body_else) {
			// Move everything into the statements above.
			GDOPT_ALLOC_NODE(InlineBlockNode, inline_block, cf->body_else);
			inline_block->statements = cf->body_else->statements;

			// Clear original statements so the if will get deleted.
			cf->body_else->statements.clear();

			Vector<Node *> *insert_statements = nullptr;
			ASTNode *insert_statement_holder = nullptr;
			int insert_pos = helper_find_ancestor_insert_statement(ast_if, &insert_statements, &insert_statement_holder);
			ERR_FAIL_COND(insert_pos == -1);

			insert_statements->insert(insert_pos, inline_block);

			if (global_options.logging_level >= 2) {
				String text = "\nif const false after\n********************\n";
				GDScriptReconstructor rec;
				rec.output_branch(inline_block, text);
				print_line(text + "\n");
			}
		}
	}
	for (uint32_t n = 0; n < if_const_true.size(); n++) {
		ASTNode *ast_if = if_const_true[n];
		removed_any_if_clauses = true;

		GDScriptParser::ControlFlowNode *cf = (ControlFlowNode *)ast_if->node;

		if (global_options.logging_level >= 2) {
			helper_flush_pending_script_name_log();
			String text = "\nif const true before\n********************\n";
			GDScriptReconstructor rec;
			rec.output_branch(cf, text, cf->arguments[0] ? cf->arguments[0] : nullptr);
			print_line(text + "\n");
		}

		if (cf->body_else) {
			cf->body_else->statements.clear();
			cf->body_else = nullptr;
		}

		// Move everything into the statements above.
		//BlockNode *inline_body = (BlockNode *)duplicate_node_recursive(*source_body, info);
		GDOPT_ALLOC_NODE(InlineBlockNode, inline_block, cf->body);
		inline_block->statements = cf->body->statements;

		// Clear original statements so the if will get deleted.
		cf->body->statements.clear();

		Vector<Node *> *insert_statements = nullptr;
		ASTNode *insert_statement_holder = nullptr;
		int insert_pos = helper_find_ancestor_insert_statement(ast_if, &insert_statements, &insert_statement_holder);
		ERR_FAIL_COND(insert_pos == -1);

		insert_statements->insert(insert_pos, inline_block);

		if (global_options.logging_level >= 2) {
			String text = "\nif const true after\n*******************\n";
			GDScriptReconstructor rec;
			rec.output_branch(inline_block, text);
			print_line(text + "\n");
		}
	}

	// The AST tree is invalidated by deleting potentially if clauses, so we need to build another.
	if (removed_any_if_clauses) {
		memdelete(tree);
		tree = memnew(ASTNode(data.get_root_class().root, nullptr));
	}

	bool repeat = true;
	bool first_run = true;
	while (repeat) {
		repeat = false;

		LocalVector<ASTNode *> unused;
		LocalVector<ASTNode *> blocks;

		tree->traverse([&unused, &blocks](ASTNode *p_node, uint32_t p_depth, bool p_user_flag) {
			switch (p_node->node->type) {
				default:
					break;
				case Node::TYPE_BLOCK: {
					blocks.push_back(p_node);
				} break;
				case Node::TYPE_CONSTANT: {
					DEV_ASSERT(p_node->parent);
					if (p_node->parent->node->type == Node::TYPE_BLOCK) {
						unused.push_back(p_node);
					}
				} break;
			}
			return true;
		},
				0);

		if (first_run) {
			first_run = false;
			for (uint32_t n = 0; n < unused.size(); n++) {
				ASTNode *del_node = unused[n];
				BlockNode *block = (BlockNode *)del_node->parent->node;
				int pos = helper_find_insert_statement(block->statements, del_node->node);
				DEV_ASSERT(pos != -1);

				GDOPT_LOG(2, "removing dead code : " + node_to_string(del_node->node));
				block->statements.remove(pos);
			}
		}

		// Remove empty loops.
		for (uint32_t n = 0; n < blocks.size(); n++) {
			ASTNode *ast = blocks[n];
			BlockNode *block = (BlockNode *)ast->node;

			if (helper_statements_are_empty(block->statements)) {
				if (unused_try_remove_child(ast)) {
					repeat = true;
					break;
				}
			}
		}

		if (repeat) {
			// Recreate AST tree for safety (can be optimized).
			memdelete(tree);
			tree = memnew(ASTNode(data.get_root_class().root, nullptr));
		}
	} // while repeat

	memdelete(tree);
}

bool GDScriptOptimizer::helper_statements_are_empty(const Vector<GDScriptParser::Node *> &p_statements) const {
	USING_GDSCRIPTPARSER
	int num_statements = p_statements.size();

	for (int n = 0; n < num_statements; n++) {
		switch (p_statements[n]->type) {
			default: {
				return false;
			} break;
			case Node::TYPE_NEWLINE: {
			} break;
			case Node::TYPE_INLINE_BLOCK: {
				InlineBlockNode *block = (InlineBlockNode *)p_statements[n];
				if (!helper_statements_are_empty(block->statements)) {
					return false;
				}
			} break;
			case Node::TYPE_BLOCK: {
				BlockNode *block = (BlockNode *)p_statements[n];
				if (!helper_statements_are_empty(block->statements)) {
					return false;
				}
			} break;
		}
	}

	return true;
}

bool GDScriptOptimizer::unused_try_remove_child(ASTNode *p_ast) {
	USING_GDSCRIPTPARSER

	if (!p_ast->parent) {
		return false;
	}
	Node *parent = p_ast->parent->node;

	switch (parent->type) {
		default:
			break;
		case Node::TYPE_BLOCK: {
			BlockNode *block = (BlockNode *)parent;
			int pos = helper_find_insert_statement(block->statements, p_ast->node);
			DEV_ASSERT(pos != -1);
			GDOPT_LOG(2, "removing dead code " + node_to_string(p_ast->node));
			block->statements.remove(pos);

			// Add a newline
			helper_statements_add_newline(block->statements, pos);
			return true;
		} break;
		case Node::TYPE_CONTROL_FLOW: {
			ControlFlowNode *cf = (ControlFlowNode *)parent;
			if (cf->match) {
				// We aren't able to remove match statements for now,
				// too complex.
				return false;
			}
			if (cf->body_else && !helper_statements_are_empty(cf->body_else->statements)) {
				return false;
			}
			if (cf->body == p_ast->node) {
				// We can delete this recursively.
				return unused_try_remove_child(p_ast->parent);
			}
		} break;
	}
	return false;
}

void GDScriptOptimizer::warning_inline_unsupported(const GDScriptParser::Node *p_node, const String &p_message) {
	if (global_options.logging_level >= 2) {
		WARN_PRINT("inlining unsupported " + node_to_string(p_node, Data::script_path) + " : " + p_message);
	}
}

void GDScriptOptimizer::warning_unroll_unsupported(const GDScriptParser::Node *p_node, const String &p_message, const GDScriptParser::Node *p_extra) {
	if (global_options.logging_level >= 2) {
		WARN_PRINT("loop unroll unsupported " + node_to_string(p_node, Data::script_path) + (p_extra ? node_to_string(p_extra) : "") + " : " + p_message);
	}
}

String GDScriptOptimizer::file_location_to_string(const GDScriptParser::Node *p_node, const String *p_path) {
	String sz_line;
	if (!p_node->line && !p_node->column) {
		sz_line = "unknown";
	} else {
		sz_line = String(" line ") + itos(p_node->line) + ", col " + itos(p_node->column);
	}

	return String("(") + (p_path ? (" \"" + *p_path + "\"") : "") + sz_line + " )";
}

String GDScriptOptimizer::node_to_string(const GDScriptParser::Node *p_node, const String *p_path) {
	USING_GDSCRIPTPARSER

	bool no_location_info = !p_node->line && !p_node->column && !p_path;
	String text = GDScriptReconstructor::draw_location(no_location_info ? "" : file_location_to_string(p_node, p_path) + " ");

	text += GDScriptReconstructor::draw_bold(GDScriptParser::_node_type_strings[p_node->type]);

	String sz;
	switch (p_node->type) {
		default: {
		} break;
		case Node::TYPE_OPERATOR: {
			sz = String(" ") + GDScriptReconstructor::OPStrings[((OperatorNode *)p_node)->op];
		} break;
		case Node::TYPE_IDENTIFIER: {
			sz = String(" \"") + ((IdentifierNode *)p_node)->name + "\"";
		} break;
		case Node::TYPE_LOCAL_VAR: {
			sz = String(" \"") + ((LocalVarNode *)p_node)->name + "\"";
		} break;
		case Node::TYPE_CONSTANT: {
			sz = String(" \"") + GDScriptReconstructor::variant_to_string(((ConstantNode *)p_node)->value, false) + "\"";
		} break;
		case Node::TYPE_FUNCTION: {
			sz = String(" \"") + ((FunctionNode *)p_node)->name + "\"";
		} break;
		case Node::TYPE_CONTROL_FLOW: {
			GDScriptParser::ControlFlowNode *cf = (ControlFlowNode *)p_node;
			sz = String(" ( ") + GDScriptParser::_control_flow_type_strings[cf->cf_type] + " )";
		} break;
	}

	text += GDScriptReconstructor::draw_node_info(sz);
	text += " " + GDScriptReconstructor::draw_location(p_node->get_datatype().to_string());

	return text;
}

void GDScriptOptimizer::licm_optimize() {
	USING_GDSCRIPTPARSER

	// First build AST tree and find all the for loops.
	ASTNode *tree = memnew(ASTNode(data.get_root_class().root, nullptr));

	LocalVector<ASTNode *> for_loops;

	// Traverse the AST tree using a lambda to extract useful info about the function.
	tree->traverse([&for_loops](ASTNode *p_node, uint32_t p_depth, bool p_user_flag) {
		switch (p_node->node->type) {
			default:
				break;
			case Node::TYPE_CONTROL_FLOW: {
				ControlFlowNode *cf = (ControlFlowNode *)p_node->node;
				if (cf->cf_type == ControlFlowNode::CF_FOR) {
					if (local_options.licm) {
						for_loops.push_back(p_node);
					} else {
						// Only add when the for loop is unrolled.
						// We always need to do LICM for unrolled loops,
						// because there may be duplicate var names.
						if (cf->_unroll) {
							for_loops.push_back(p_node);
						}
					}
				}
			} break;
		}
		return true;
	},
			0);

	// Traverse just from the for loops.
	for (uint32_t l = 0; l < for_loops.size(); l++) {
		// Find the insert location for LICM ahead of time for each for loop, while
		// we have access to the whole tree.
		data.licm_location.insert_pos = helper_find_ancestor_insert_statement(for_loops[l], &data.licm_location.insert_statements, &data.licm_location.insert_statement_holder, true);
		ERR_CONTINUE(data.licm_location.insert_pos == -1);

		licm_process_for_loop((ControlFlowNode *)for_loops[l]->node);
	}

	// Finished with the tree.
	memdelete(tree);
}

void GDScriptOptimizer::generate_error_report(GDScriptParser &r_parser, bool p_file_requests_optimization) {
#ifdef GDSCRIPT_OPTIMIZER_REQUIRE_KEYWORDS
	if (!p_file_requests_optimization) {
		return;
	}
#endif

	if (!global_options.optimization) {
		return;
	}

	if (global_options.logging_level >= 4) {
		// Already will have printed source.
		return;
	}
	helper_flush_pending_script_name_log();
	GDScriptReconstructor rc;
	String text = String("\nSCRIPT ERROR: Compile Error in optimized script : ") + *data.script_path + " :\n";
	rc.output(r_parser, text);
	print_line(text);
}

void GDScriptOptimizer::inline_setup_class_functions_recursive(GDScriptParser::ClassNode *p_class, bool p_root_class) {
	if (!p_root_class) {
		data.curr_class_id = data.classes.size();
		data.classes.resize(data.classes.size() + 1);
		data.get_current_class().class_name = p_class->name;
	}

	data.get_current_class().root = p_class;
	// Make sure any previous functions are cleared.
	data.get_current_class().functions.clear();

	inline_setup_class_functions();

	// Subclasses.
	for (int n = 0; n < p_class->subclasses.size(); n++) {
		inline_setup_class_functions_recursive(p_class->subclasses[n], false);
	}
}

void GDScriptOptimizer::inline_process_class_recursive(GDScriptParser::ClassNode *p_class, bool p_root_class) {
	static uint32_t curr_class_id = 0;

	if (!p_root_class) {
		data.curr_class_id = ++curr_class_id;
		DEV_ASSERT(data.get_current_class().class_name == p_class->name);
	} else {
		data.curr_class_id = 0;
	}

	LocalVector<Call> calls;
	inline_search(calls);

	for (uint32_t n = 0; n < calls.size(); n++) {
		inline_make(calls[n]);
	}

	// Remove any unused inline funcs from the source code.
	// (But only if the inline keyword is being used, because otherwise,
	// we might remove functions that need to be called from other classes etc.)
	// N.B. Turned off for now, as the user might want to e.g. call inline funcs via strings,
	// rather than directly.
	// Have to decide in future whether removal is worth pursuing.
#if 0
	if (require_inline_keyword) {
		for (uint32_t n = 0; n < data.get_current_class().functions.size(); n++) {
			const Function &func = data.get_current_class().functions[n];

			if (!func.non_inline_calls) {
				p_class->functions.erase(func.node);
				p_class->static_functions.erase(func.node);
			}
		}
	}
#endif

	// Subclasses.
	for (int n = 0; n < p_class->subclasses.size(); n++) {
		inline_process_class_recursive(p_class->subclasses[n], false);
	}
}

Error GDScriptOptimizer::optimize(GDScriptParser &r_parser, const String &p_path, bool p_file_requests_optimization) {
	if (!global_options.optimization) {
		return OK;
	}

#ifdef GDSCRIPT_OPTIMIZER_REQUIRE_KEYWORDS
	if (!p_file_requests_optimization && global_options.require_inline_keyword && global_options.require_unroll_keyword) {
		return OK;
	}
#endif

	// Modify the local options with the global option on / offs.
	local_options.constant_folding = local_options.constant_folding && global_options.constant_folding;
	local_options.inlining = local_options.inlining && global_options.inlining;
	local_options.licm = local_options.licm && global_options.licm;
	local_options.remove_unused = local_options.remove_unused && global_options.remove_unused;
	local_options.unrolling = local_options.unrolling && global_options.unrolling;

	////////////////////////////////////////////////////
	// Debugging and testing.
#if 0
	// Necessary to get Wrought Flesh to run for testing when always inline is set,
	// because this script uses virtual functions and is not compatible.
	if (p_path.ends_with("NPC.gd")) {
		return OK;
	}
#endif

// #define GODOT_SCRIPT_OPTIMIZER_ISOLATE_SCRIPT "GridContainer.gd"
#ifdef GODOT_SCRIPT_OPTIMIZER_ISOLATE_SCRIPT
	if (p_path.find(GODOT_SCRIPT_OPTIMIZER_ISOLATE_SCRIPT) == -1) {
		return OK;
	}
#endif
	////////////////////////////////////////////////////

#ifndef TOOLS_ENABLED
	// Turn off logging in exports.
	global_options.logging_level = 0;
#endif

	static bool printed_mode = false;
	if (!printed_mode) {
		GDOPT_LOG(1, String("inlining ") + helper_bool_to_on_off(local_options.inlining) + ", licm " + helper_bool_to_on_off(local_options.licm) + ", dead code elimination " + helper_bool_to_on_off(local_options.remove_unused) + ", unrolling " + helper_bool_to_on_off(local_options.unrolling) + ", constant folding " + helper_bool_to_on_off(local_options.unrolling));
		printed_mode = true;
	}

	data.log_script_name_pending = true;
	data.script_path = &p_path;

	if (local_options.all_off()) {
		return OK;
	}

	GDScriptParser::Node *root = r_parser.head;
	ERR_FAIL_COND_V(root->type != GDScriptParser::Node::TYPE_CLASS, ERR_INVALID_DATA);

	data.parser = &r_parser;
	data.get_root_class().root = static_cast<GDScriptParser::ClassNode *>(root);

	GDScriptReconstructor rc;
	String text;

	remove_excessive_newlines_recursive();

	if (local_options.constant_folding) {
		constant_fold_expressions();
	}

	if (local_options.remove_unused) {
		unused_remove_expressions();
	}

	if (local_options.inlining) {
		GDScriptParser::ClassNode *orig_root = data.get_root_class().root;

		inline_setup_class_functions_recursive(orig_root, true);
		inline_process_class_recursive(orig_root, true);

		// Reset original root class.
		data.get_root_class().root = orig_root;
	}

	if (local_options.constant_folding) {
		constant_fold_expressions();
	}

	licm_optimize();

	if (local_options.remove_unused) {
		unused_remove_expressions();
	}

	if (local_options.unrolling) {
		unroll_loops();
	}

	remove_inline_blocks(data.get_root_class().root);
	remove_excessive_newlines_recursive();

	if (global_options.logging_level >= 4) {
		helper_flush_pending_script_name_log();
		if (global_options.logging_level >= 5) {
			ASTNode *tree = memnew(ASTNode(data.get_root_class().root, nullptr));
			print_line("\nfinal AST tree\n**************\n");
			tree->debug_print();
			memdelete(tree);
		}

		text.clear();
		text = "\nfinal script\n************\n";
		rc.output(r_parser, text);
		print_line(text);
	}

	return OK;
}

#undef GDOPT_ALLOC_NODE
#undef GDOPT_LOG
#undef USING_GDSCRIPTPARSER

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
