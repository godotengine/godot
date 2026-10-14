/**************************************************************************/
/*  gdscript_linter.cpp                                                   */
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

#include "gdscript_linter.h"

#ifdef DEBUG_ENABLED

#include "core/typedefs.h"
#include "servers/text/text_server.h"

#include "modules/gdscript/gdscript_parser.h"

// Checks ===========================================================================

struct ConfusableIdentifier final {
private:
	static void check_identifier(const GDScriptParser::IdentifierNode *p_identifier, GDScriptParser &p_sink) {
		const Ref<TextServer> &ts = TextServerManager::get_singleton()->get_primary_interface();
		// Check for synthetic nodes (`start_line != -1`) manually, since this is not called directly by `GDScriptLinter`.
		if (p_identifier && p_identifier->start_line != -1 && ts->has_feature(TextServer::FEATURE_UNICODE_SECURITY) && ts->spoof_check(p_identifier->name)) {
			p_sink.push_warning(p_identifier, GDScriptWarning::CONFUSABLE_IDENTIFIER, p_identifier->name.string());
		}
	}

public:
	static void check_class(const GDScriptParser::ClassNode *p_class, GDScriptParser &p_sink) { check_identifier(p_class->identifier, p_sink); }
	static void check_assignable(const GDScriptParser::AssignableNode *p_var, GDScriptParser &p_sink) { check_identifier(p_var->identifier, p_sink); }
	static void check_function(const GDScriptParser::FunctionNode *p_func, GDScriptParser &p_sink) { check_identifier(p_func->identifier, p_sink); }
	static void check_signal(const GDScriptParser::SignalNode *p_sig, GDScriptParser &p_sink) { check_identifier(p_sig->identifier, p_sink); }
	static void check_for(const GDScriptParser::ForNode *p_for, GDScriptParser &p_sink) { check_identifier(p_for->variable, p_sink); }
	static void check_enum(const GDScriptParser::EnumNode *p_enum, GDScriptParser &p_sink) {
		check_identifier(p_enum->identifier, p_sink);
		for (const GDScriptParser::EnumNode::Value &value : p_enum->values) {
			check_identifier(value.identifier, p_sink);
		}
	}
	static void check_pattern(const GDScriptParser::PatternNode *p_pattern, GDScriptParser &p_sink) {
		if (p_pattern->pattern_type == GDScriptParser::PatternNode::PT_BIND) {
			check_identifier(p_pattern->bind, p_sink);
		}
	}
};

struct OnreadyWithCast final {
	static void check_variable(const GDScriptParser::VariableNode *p_variable, GDScriptParser &p_sink) {
		if (!p_variable->onready || !p_variable->initializer || p_variable->initializer->type != GDScriptParser::Node::CAST) {
			return;
		}

		const GDScriptParser::CastNode *cast = static_cast<const GDScriptParser::CastNode *>(p_variable->initializer);
		if (!cast->operand || cast->operand->type != GDScriptParser::Node::GET_NODE) {
			return;
		}

		const GDScriptParser::GetNodeNode *get_node = static_cast<const GDScriptParser::GetNodeNode *>(cast->operand);
		p_sink.push_warning(cast, GDScriptWarning::ONREADY_WITH_CAST, (get_node->use_dollar ? "$" : "%") + get_node->full_path);
	}
};

// Infrastructure ===================================================================

template <typename T>
void _FORCE_INLINE_ validated(const GDScriptParser::Node *p_node, GDScriptParser &p_sink, GDScriptLinter::Callback<T> *p_callback) {
	const T *casted = dynamic_cast<const T *>(p_node);
	if (casted) {
		p_callback(casted, p_sink);
	}
}

#define LINTER_CHECK(m_method) [](const GDScriptParser::Node *p_node, GDScriptParser &p_sink) { validated(p_node, p_sink, &m_method); }

constexpr GDScriptLinter::CallbackWithValidation *const GDScriptLinter::checks[] = {
	LINTER_CHECK(ConfusableIdentifier::check_assignable),
	LINTER_CHECK(ConfusableIdentifier::check_class),
	LINTER_CHECK(ConfusableIdentifier::check_enum),
	LINTER_CHECK(ConfusableIdentifier::check_for),
	LINTER_CHECK(ConfusableIdentifier::check_function),
	LINTER_CHECK(ConfusableIdentifier::check_pattern),
	LINTER_CHECK(ConfusableIdentifier::check_signal),

	LINTER_CHECK(OnreadyWithCast::check_variable),
};

#undef LINTER_CHECK

Error GDScriptLinter::lint() {
	// We use the fact that all nodes form a linked list for memory management purposes to iterate all nodes without actually walking the tree.
	// The iteration order is undefined, this should not matter since the tree is immutable for the linter.
	GDScriptParser::Node *curr = tree->list;
	for (; curr != nullptr; curr = curr->next) {
		if (curr->start_line == -1) {
			// Sometimes we allocate synthetic nodes with no correspondence in code e.g. identifiers containing the synthetic names of getter functions.
			// We shouldn't analyze those nodes directly, since suppressing them with `@warning_ignore` would be impossible.
			continue;
		}
		for (CallbackWithValidation *check : checks) {
			check(curr, *tree);
		}
	}

	tree->apply_pending_warnings();
	return tree->get_errors().is_empty() ? OK : ERR_PARSE_ERROR;
}

#endif // DEBUG_ENABLED
