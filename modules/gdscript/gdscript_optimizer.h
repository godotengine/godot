/**************************************************************************/
/*  gdscript_optimizer.h                                                  */
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

#ifndef GDSCRIPT_OPTIMIZER_H
#define GDSCRIPT_OPTIMIZER_H

#include "gdscript_parser.h"

class GDScriptOptimizer {
	// Wrapper for allocating GDScriptParser nodes.
	template <class T>
	T *alloc_node(const GDScriptParser::Node *p_location) const;

	// The existing GDScriptParser nodes are complex to traverse,
	// so instead we create simpler Abstract Syntax Tree structure mirroring the original
	// nodes, with a simple child list and parent pointer.
	struct ASTNode {
		GDScriptParser::Node *node = nullptr;
		ASTNode *parent;

		// All possible children, arguments, statements, etc all as one easily iteratable list.
		LocalVector<ASTNode *> children;

		// Note that for Operators, the first children will always be the arguments by convention, so they are
		// quickly accessible. We enforce this with a wrapper here.
		ASTNode *get_argument(uint32_t p_arg) { return children[p_arg]; }
		const ASTNode *get_argument(uint32_t p_arg) const { return children[p_arg]; }

		// In order to prevent processing a node more than once on a "run" of e.g. LICM shifting,
		// it can be useful to mark it with a flood_id.
		uint32_t flood_id = 0;

		// For LICM, we can mark branches as invariant, which allows them to be shifted outside the loop.
		enum class ASTInvariant {
			UNPROCESSED,
			VARIANT,
			INVARIANT,
		};
		ASTInvariant invariant_state = ASTInvariant::UNPROCESSED;

		uint32_t find_child_id(GDScriptParser::Node *p_node) const;
		ASTNode *find_child(GDScriptParser::Node *p_node) const {
			uint32_t id = find_child_id(p_node);
			if (id != UINT32_MAX) {
				return children[id];
			}
			return nullptr;
		}

		void flood_fill(uint32_t p_flood_id) {
			flood_id = p_flood_id;
			// print_line("flood filling " + node_to_string(node));
			for (uint32_t n = 0; n < children.size(); n++) {
				children[n]->flood_fill(p_flood_id);
			}
		}

		template <typename Func>
		void traverse(Func p_callback, uint32_t p_depth, bool p_user_flag = false) {
			// Callback returns whether to continue traversing tree.
			if (p_callback(this, p_depth, p_user_flag) == false) {
				return;
			}
			for (uint32_t n = 0; n < children.size(); n++) {
				children[n]->traverse(p_callback, p_depth + 1, p_user_flag);
			}
		}

		uint32_t count_nodes() const;
		void debug_print(uint32_t p_depth = 0) const;
		ASTNode(GDScriptParser::Node *p_node, ASTNode *p_parent);
		~ASTNode();
	};

	// Call site of a function, used to select calls for inlining.
	struct Call {
		GDScriptParser::OperatorNode *call_node = nullptr;
		GDScriptParser::Node *call_parent = nullptr;
		const GDScriptParser::IdentifierNode *call_identifier = nullptr;
		GDScriptParser::BlockNode *parent_block = nullptr;
		GDScriptParser::Node *parent_block_first_child = nullptr;

		// The function that is making this inline call.
		GDScriptParser::FunctionNode *call_container_function = nullptr;

		// The identified function to be inlined.
		uint32_t gdclass_id = UINT32_MAX;
		uint32_t function_id = UINT32_MAX;

		// Object pointer for non root classes.
		GDScriptParser::Node *self = nullptr;
	};

	// Function declaration parameter.
	struct FunctionParam {
		String name;
		GDScriptParser::Node *default_value = nullptr;
		bool is_written = false;
		bool is_used = false;
	};

	// Function declaration.
	struct Function {
		GDScriptParser::FunctionNode *node = nullptr;
		LocalVector<FunctionParam> params;

		// If there is a single return value, we can substitute it
		// directly in the calling code.
		// If there are multiple return values, we need an intermediate.
		uint32_t num_return_values = 0;

		// Whether this function can be supported as inlineable.
		bool supported = true;

		//If the function assigns anything but a local variable,
		// or makes any function calls, we can't guarantee is it const.
		// Non-const functions cannot be chained in multi-expressions,
		// because the first function could alter the result of the second function.
		// This is a painful restriction, but is necessary to ensure correctness.
		bool is_const_function = true;

		// If the function is inlined but has no non_inlined_calls,
		// we can remove it from the final script.
		uint32_t non_inline_calls = 0;

		uint32_t find_param_by_name(String p_name) const {
			for (uint32_t n = 0; n < params.size(); n++) {
				if (params[n].name == p_name) {
					return n;
				}
			}
			return UINT32_MAX;
		}
	};

	// When duplicating nodes, we often need to change identifiers
	// to prevent name conflicts.
	// The Inline changes enables us to do this, by replacing identifiers as we go.
	struct InlineChange {
		String identifier_from;
		String identifier_to;

		// If this is set, takes precedence over an identifier to.
		GDScriptParser::Node *node_to = nullptr;
	};

	struct InlineInfo {
		LocalVector<InlineChange> changes;
		GDScriptParser::IdentifierNode *return_intermediate = nullptr;
		GDScriptParser::Node **returned_expression = nullptr;

		// If the inline is called on another class, we need to pass the this (self)
		// so that the correct object variables can be accessed.
		GDScriptParser::Node *self = nullptr;
		GDScriptParser::ClassNode *gdclass = nullptr;
	};

	// GDScriptParser nodes don't maintain a parent.
	// We manually keep a chain in order to be able to move up through parents,
	// without modifying GDScriptParser or ensuring that the parents are up to date.
	class Chain {
		static inline const uint32_t MAX_CHAIN_LENGTH = 8;
		GDScriptParser::Node *nodes[MAX_CHAIN_LENGTH] = {};
		uint32_t _count = 0;

	public:
		bool push_back(GDScriptParser::Node *p_node) {
			if (_count >= MAX_CHAIN_LENGTH) {
				return false;
			}
			nodes[_count++] = p_node;
			return true;
		}
		void clear() { _count = 0; }
		uint32_t length() const { return _count; }
		GDScriptParser::BlockNode *find_enclosing_block(GDScriptParser::Node *&r_first_child) const;
		GDScriptParser::Node *find_call_parent() const;
		Chain(GDScriptParser::Node *p_start) { push_back(p_start); }
	};

	struct LICMInsertLocation {
		Vector<GDScriptParser::Node *> *insert_statements = nullptr;
		ASTNode *insert_statement_holder = nullptr;
		int insert_pos = -1;
	};

	struct LICMMention {
		ASTNode *op = nullptr;
		ASTNode *expression = nullptr;
	};

	// For Loop invariant code motion, we need to keep track of variables.
	struct LICMVarInfo {
		String name;
		bool assigned = false;

		// Disallow shifting, or shifting any expression that contains this variable.
		// Probably a for loop counter.
		bool disallow = false;

		// Variable declarations that are in the first
		// layer inside a loop can normally be shifted outside the loop
		// (although take care for name clashes).
		bool shift_declaration = false;

		// This is determined by checking all mentions
		// to check none of them
		bool invariant = false;

		ASTNode *local_var_node = nullptr;

		uint32_t constant_writes = 0;
		uint32_t expression_writes = 0;

		// Assignment mentions are operators where this variable was assigned.
		LocalVector<LICMMention> assignment_mentions;

		// Expressions are either side of operators that do not result in an assignment to the variable.
		// These can sometimes be optimized to consts outside the loop.
		LocalVector<LICMMention> expression_mentions;
	};

	struct LICMVars {
		const LICMVarInfo *find(String p_name) const {
			uint32_t id = find_id(p_name);
			if (id != UINT32_MAX) {
				return &vars[id];
			}
			return nullptr;
		}
		uint32_t find_id(String p_name) const {
			for (uint32_t n = 0; n < vars.size(); n++) {
				if (vars[n].name == p_name) {
					return n;
				}
			}
			return UINT32_MAX;
		}
		LICMVarInfo *find_or_create(String p_name) {
			uint32_t id = find_id(p_name);
			if (id != UINT32_MAX) {
				return &vars[id];
			}
			LICMVarInfo vi;
			vi.name = p_name;
			vars.push_back(vi);
			return &vars[vars.size() - 1];
		}
		LocalVector<LICMVarInfo> vars;

		// Not associated with any variable, can freely be moved outside the loop
		// if const expressions.
		LocalVector<LICMMention> free_expression_mentions;

		ASTNode *licm_root = nullptr;
	};

	enum class UniqueNameType : uint32_t {
		VARIABLE = 0,
		RETURN,
		COUNTER,
		UNROLL_LIMIT,
		UNROLL_UNITS,
		UNROLL_LEFTOVERS,
		UNROLL_COUNTER,
		TEMP,
		MAX,
	};

	struct GDClass {
		String class_name;
		LocalVector<Function> functions;
		uint32_t find_function_by_name(const StringName &p_name) const;
		GDScriptParser::ClassNode *root = nullptr;
	};

	// Global data used by the optimizer.
	struct Data {
		GDScriptParser *parser = nullptr;

		uint32_t unique_identifier_counts[(uint32_t)UniqueNameType::MAX] = {};

		LocalVector<GDClass> classes;
		uint32_t curr_class_id = 0;
		GDClass &get_root_class() { return classes[0]; }
		GDClass &get_current_class() { return classes[curr_class_id]; }

		uint32_t find_class_by_name(const String &p_name) const;

		inline static const String *script_path = nullptr;

		LICMInsertLocation licm_location;

		// Defer logging the name of the script being optimized until we have something to report.
		// This helps prevent log spam.
		static bool log_script_name_pending;

		Data() {
			classes.resize(1);
		}
	} data;

	// Inlining
	void inline_setup_class_functions();
	void inline_search(LocalVector<Call> &r_calls);
	void inline_search(LocalVector<Call> &r_calls, GDScriptParser::FunctionNode *p_func);
	bool inline_search(LocalVector<Call> &r_calls, GDScriptParser::Node *p_node, Chain p_chain, GDScriptParser::FunctionNode *p_parent_func, bool p_caller_requires_return, bool p_const_inlines_only = false);

	GDScriptParser::OperatorNode *inline_make_declare_assignment(GDScriptParser::IdentifierNode *p_var_name, GDScriptParser::Node *p_assigned_node);
	bool inline_make_params(const Call &p_call, const Function &p_source_func, InlineInfo &r_info, int &r_insert_statement_id);
	void inline_make(const Call &p_call);
	void inline_process_class_recursive(GDScriptParser::ClassNode *p_class, bool p_root_class);
	void inline_setup_class_functions_recursive(GDScriptParser::ClassNode *p_class, bool p_root_class);

	void function_register_and_check_inline_support(int p_func_id);
	static void function_param_mark_use(GDScriptOptimizer::Function &p_func_decl, const GDScriptParser::Node *p_node, bool p_assigned);

	// Duplicating AST tree branches
	struct DuplicateNodeResult {
		// If the duplicated node would always return, we don't need any more statements
		// from the calling block.
		bool has_return = false;

		// An if without else, that has a return, can still work by changing the following
		// statements to an else block in the calling code.
		bool if_return_needs_else = false;
	};

	GDScriptParser::Node *duplicate_node(const GDScriptParser::Node &p_source) const;
	GDScriptParser::Node *duplicate_node_recursive(const GDScriptParser::Node &p_source, InlineInfo &p_changes, DuplicateNodeResult *r_result = nullptr);
	void duplicate_statements_recursive(const Vector<GDScriptParser::Node *> &p_source, int32_t p_source_pos, Vector<GDScriptParser::Node *> &r_dest, int32_t p_dest_pos, InlineInfo &p_changes, DuplicateNodeResult *r_result);

	// Newlines
	void remove_inline_blocks(GDScriptParser::Node *p_branch);
	void remove_excessive_newlines_recursive();
	void _remove_excessive_newlines_recursive(GDScriptParser::Node *p_branch, bool p_remove_from_front = false);
	static void remove_excessive_newlines(Vector<GDScriptParser::Node *> &r_statements, bool p_remove_from_front, bool *p_leading_newline = nullptr);

	static String file_location_to_string(const GDScriptParser::Node *p_node, const String *p_path = nullptr);
	static String node_to_string(const GDScriptParser::Node *p_node, const String *p_path = nullptr);

	static void warning_inline_unsupported(const GDScriptParser::Node *p_node, const String &p_message);
	static void warning_unroll_unsupported(const GDScriptParser::Node *p_node, const String &p_message, const GDScriptParser::Node *p_extra = nullptr);

	// Loop invariant code motion (LICM).
	void licm_optimize();
	void licm_process_for_loop(GDScriptParser::ControlFlowNode *p_cf_for);
	static bool licm_is_simple_type(const GDScriptParser::DataType &p_dt);

	void licm_shift_variable_declaration(ASTNode *p_ast_for, const LICMVarInfo &p_vi);
	void licm_shift_variable(ASTNode *p_ast_for, const LICMVarInfo &p_vi);
	Vector<GDScriptParser::Node *> *_helper_find_ancestor_statements(ASTNode *p_ast_node, ASTNode **p_first_child, bool p_highest_block) const;
	bool licm_is_op_invariant(const String &p_variable_name, ASTNode *p_ast_variable, const LICMVars &p_vars) const;
	bool licm_remove_if_statement_holder_empty(ASTNode *p_holder, Vector<GDScriptParser::Node *> &r_statements);

	bool licm_try_optimize_expression_mention(ASTNode *p_ast_for, const LICMMention &p_mention);
	bool licm_is_expression_constant(ASTNode *p_op) const;
	void licm_shift_const_expression(ASTNode *p_ast_for, ASTNode *p_ast_expr);

	// Unused expressions / constants.
	void unused_remove_expressions();
	bool unused_try_remove_child(ASTNode *p_ast);

	// Constant foldering / contracting math expressions.
	void constant_fold_expressions();
	bool contract_try_expression(ASTNode *p_ast, GDScriptParser::ClassNode *p_curr_class);
	bool contract_try_expression_unary(ASTNode *p_ast, GDScriptParser::ClassNode *p_curr_class);
	bool contract_try_expression_binary(ASTNode *p_ast, GDScriptParser::ClassNode *p_curr_class);
	bool contract_try_paired_expression_binary(ASTNode *p_ast, GDScriptParser::ClassNode *p_curr_class, GDScriptParser::OperatorNode *child_op_node_a, GDScriptParser::OperatorNode *child_op_node_b);

	bool contract_try_expression_two_constants(ASTNode *p_ast, const GDScriptParser::ConstantNode *p_const_a, const GDScriptParser::ConstantNode *p_const_b);

	bool contract_is_binary_expression_contractable(GDScriptParser::ClassNode *p_curr_class, GDScriptParser::Node *p_node, bool p_add_or_multiply, const GDScriptParser::ConstantNode **r_child_constant, GDScriptParser::Node **r_child_non_constant);
	bool contract_is_node_const_identifier_or_constant(const GDScriptParser::ClassNode *p_curr_class, const GDScriptParser::Node *p_node, const GDScriptParser::ConstantNode **r_constant_node) const;

	bool helper_do_variant_math(GDScriptParser::OperatorNode::Operator p_op, const Variant &p_a, const Variant &p_b, Variant &r_result) const;
	static bool helper_do_variant_comparison(GDScriptParser::OperatorNode::Operator p_op, const Variant &p_a, const Variant &p_b, bool &r_is_true);
	bool helper_make_variant_positive(Variant &r_var) const;
	bool helper_flip_variant_polarity(Variant &r_var) const;

	// Loop unrolling.
	void unroll_loops();
	void unroll_loop(ASTNode *p_loop);
	bool unroll_are_counters_invariant(ASTNode *p_control_flow, GDScriptParser::BlockNode *p_body, bool &r_is_read) const;

	// Helpers.
	GDScriptParser::IdentifierNode *helper_declare_local_var(const String &p_local_var_name, GDScriptParser::Node *p_assigned_node, Vector<GDScriptParser::Node *> &r_statements, int &r_insert_statement_id, GDScriptParser::Node *p_source_location = nullptr, const GDScriptParser::DataType *p_source_data_type = nullptr);

	void helper_statements_add_newline(Vector<GDScriptParser::Node *> &r_statements, int32_t p_pos = -1);
	int helper_find_ancestor_insert_statement(ASTNode *p_ast_node, Vector<GDScriptParser::Node *> **r_statements, ASTNode **r_statement_holder, bool p_highest_block = false) const;

	bool helper_node_exchange_child(GDScriptParser::Node &r_parent, const GDScriptParser::Node *p_old_child, GDScriptParser::Node *p_new_child);
	int helper_find_insert_statement(const Vector<GDScriptParser::Node *> &p_statements, GDScriptParser::Node *p_search_node) const;

	bool helper_statements_are_empty(const Vector<GDScriptParser::Node *> &p_statements) const;
	bool helper_exchange_statements(Vector<GDScriptParser::Node *> &r_extract_statements, int p_extract_pos, Vector<GDScriptParser::Node *> &r_insert_statements, int &r_insert_pos) const;
	void helper_find_all_types(LocalVector<ASTNode *> &r_found, ASTNode *p_root, GDScriptParser::Node::Type p_type, GDScriptParser::ControlFlowNode::CFType p_cf_type = GDScriptParser::ControlFlowNode::CF_FOR);
	GDScriptParser::Node *helper_get_default_value_for_type(const GDScriptParser::DataType &p_type, int p_line = -1) const;

	static bool helper_identifiers_match_by_name(GDScriptParser::IdentifierNode *p_a, GDScriptParser::Node *p_b);
	static bool helper_is_operator_assign(const GDScriptParser::OperatorNode::Operator p_op);
	static bool helper_is_operator_unary(const GDScriptParser::OperatorNode::Operator p_op);
	static bool helper_is_operator_binary(const GDScriptParser::OperatorNode::Operator p_op);

	static bool helper_is_static_control_flow_true(const GDScriptParser::ClassNode *p_curr_class, const GDScriptParser::ControlFlowNode *p_cf, bool &r_is_true);
	static bool helper_get_constant_bool(const GDScriptParser::ClassNode *p_curr_class, const GDScriptParser::Node *p_node, bool &r_is_true);
	static bool helper_get_constant_value(const GDScriptParser::ClassNode *p_curr_class, const GDScriptParser::Node *p_node, Variant &r_value);

	static bool helper_block_contains_return(const GDScriptParser::BlockNode *p_block, bool p_within_control_flow = false);
	static bool _helper_statements_contain_return(const Vector<GDScriptParser::Node *> &p_statements, bool p_within_control_flow);
	static const StringName *helper_get_identifier_node_name(const GDScriptParser::Node *p_node);
	static const char *helper_bool_to_on_off(bool p_on) { return p_on ? "ON" : "OFF"; }

	String helper_make_unique_name(String p_name, UniqueNameType p_type);
	static void helper_flush_pending_script_name_log();

public:
	static struct GlobalOptions {
		// Global on / off switch.
		bool optimization = true;

		// Optimization options (set in project settings).
		bool inlining = true;
		bool licm = true;
		bool remove_unused = true;
		bool unrolling = true;
		bool constant_folding = true;

		// Debug logging options.
#ifdef TOOLS_ENABLED
		uint32_t logging_level = 2; // 0 to 5, 2 default.
#else
		uint32_t logging_level = 0;
#endif

		// These two are really for debugging / testing.
		// They are always required in production.
		bool require_inline_keyword = true;
		bool require_unroll_keyword = true;
	} global_options;

	static struct LocalOptions {
		bool constant_folding = true;
		bool remove_unused = true;
		bool inlining = true;
		bool licm = true;
		bool unrolling = true;

		bool all_off() const { return !constant_folding && !remove_unused && !inlining && !licm && !unrolling; }
		void reset() {
			constant_folding = true;
			remove_unused = true;
			inlining = true;
			licm = true;
			unrolling = true;
		}
	} local_options;

	Error optimize(GDScriptParser &r_parser, const String &p_path, bool p_file_requests_optimization);
	void generate_error_report(GDScriptParser &r_parser, bool p_file_requests_optimization);

	GDScriptOptimizer() { local_options.reset(); }
};

#endif // GDSCRIPT_OPTIMIZER_H
