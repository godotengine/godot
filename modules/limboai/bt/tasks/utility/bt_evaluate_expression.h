/**
 * bt_evaluate_expression.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 * Copyright 2024 Wilson E. Alvarez
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_EVALUATE_EXPRESSION_H
#define BT_EVALUATE_EXPRESSION_H

#include "../bt_action.h"

#ifdef LIMBOAI_MODULE
#include "core/math/expression.h"
#endif

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/expression.hpp>
#endif

#include "../../../blackboard/bb_param/bb_node.h"
#include "../../../blackboard/bb_param/bb_variant.h"

class BTEvaluateExpression : public BTAction {
	GDCLASS(BTEvaluateExpression, BTAction);
	TASK_CATEGORY(Utility);

private:
	Ref<Expression> expression;
	Error is_parsed = FAILED;
	Ref<BBNode> node_param;
	String expression_string;
	PackedStringArray input_names;
	TypedArray<BBVariant> input_values;
	bool input_include_delta = false;
	Array processed_input_values;
	StringName result_var;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual void _setup() override;
	virtual Status _tick(double p_delta) override;

public:
	Error parse();

	void set_expression_string(const String &p_expression_string);
	String get_expression_string() const { return expression_string; }

	void set_node_param(Ref<BBNode> p_object);
	Ref<BBNode> get_node_param() const { return node_param; }

	void set_input_names(const PackedStringArray &p_input_names);
	PackedStringArray get_input_names() const { return input_names; }

	void set_input_values(const TypedArray<BBVariant> &p_input_values);
	TypedArray<BBVariant> get_input_values() const { return input_values; }

	void set_input_include_delta(bool p_input_include_delta);
	bool is_input_delta_included() const { return input_include_delta; }

	void set_result_var(const StringName &p_result_var);
	StringName get_result_var() const { return result_var; }

	virtual PackedStringArray get_configuration_warnings() override;

	BTEvaluateExpression();
};

#endif // BT_EVALUATE_EXPRESSION_H
