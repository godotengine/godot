/**
 * test_evaluate_expression.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 * Copyright 2024 Wilson E. Alvarez
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_EVALUATE_EXPRESSION_H
#define TEST_EVALUATE_EXPRESSION_H

#include "limbo_test.h"

#include "modules/limboai/blackboard/bb_param/bb_node.h"
#include "modules/limboai/blackboard/blackboard.h"
#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/utility/bt_evaluate_expression.h"

#include "core/os/memory.h"
#include "core/variant/array.h"

namespace TestEvaluateExpression {

TEST_CASE("[Modules][LimboAI] BTEvaluateExpression") {
	Ref<BTEvaluateExpression> ee = memnew(BTEvaluateExpression);

	SUBCASE("When node parameter is null") {
		ee->set_node_param(nullptr);
		ERR_PRINT_OFF;
		CHECK(ee->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	SUBCASE("With object on the blackboard") {
		Node *dummy = memnew(Node);
		Ref<Blackboard> bb = memnew(Blackboard);

		Ref<BBNode> node_param = memnew(BBNode);
		ee->set_node_param(node_param);
		Ref<CallbackCounter> callback_counter = memnew(CallbackCounter);
		bb->set_var("object", callback_counter);
		node_param->set_value_source(BBParam::BLACKBOARD_VAR);
		node_param->set_variable("object");
		ee->set_expression_string("callback()");

		ee->initialize(dummy, bb, dummy);

		SUBCASE("When expression string is empty") {
			ee->set_expression_string("");
			CHECK(ee->parse() == ERR_INVALID_PARAMETER);
			ERR_PRINT_OFF;
			CHECK(ee->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
		}
		SUBCASE("When expression string calls non-existent function") {
			ee->set_expression_string("not_found()");
			CHECK(ee->parse() == OK);
			ERR_PRINT_OFF;
			CHECK(ee->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
		}
		SUBCASE("When expression string accesses a non-existent property") {
			ee->set_expression_string("not_found");
			CHECK(ee->parse() == OK);
			ERR_PRINT_OFF;
			CHECK(ee->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
		}
		SUBCASE("When expression string can't be parsed") {
			ee->set_expression_string("assignment = failure");
			CHECK(ee->parse() == ERR_INVALID_PARAMETER);
			ERR_PRINT_OFF;
			CHECK(ee->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
		}
		SUBCASE("When expression is valid") {
			ee->set_expression_string("callback()");
			CHECK(ee->parse() == OK);
			ERR_PRINT_OFF;
			CHECK(ee->execute(0.01666) == BTTask::SUCCESS);
			ERR_PRINT_ON;
			CHECK(callback_counter->num_callbacks == 1);
		}
		SUBCASE("With inputs") {
			ee->set_expression_string("callback_delta(delta)");

			SUBCASE("Should fail with 0 inputs") {
				ee->set_input_include_delta(false);
				ee->set_input_names(PackedStringArray());
				CHECK(ee->parse() == OK);
				ee->set_input_values(TypedArray<BBVariant>());
				ERR_PRINT_OFF;
				CHECK(ee->execute(0.01666) == BTTask::FAILURE);
				ERR_PRINT_ON;
				CHECK(callback_counter->num_callbacks == 0);
			}
			SUBCASE("Should succeed with too many inputs") {
				ee->set_input_include_delta(true);
				PackedStringArray input_names;
				input_names.push_back("point_two");
				ee->set_input_names(input_names);
				CHECK(ee->parse() == OK);
				TypedArray<BBVariant> input_values;
				input_values.push_back(memnew(BBVariant(0.2)));
				ee->set_input_values(input_values);
				ERR_PRINT_OFF;
				CHECK(ee->execute(0.01666) == BTTask::SUCCESS);
				ERR_PRINT_ON;
				CHECK(callback_counter->num_callbacks == 1);
			}
			SUBCASE("Should fail with a wrong type arg") {
				ee->set_input_include_delta(false);
				PackedStringArray input_names;
				input_names.push_back("delta");
				ee->set_input_names(input_names);
				CHECK(ee->parse() == OK);
				TypedArray<BBVariant> input_values;
				input_values.push_back(memnew(BBVariant("wrong data type")));
				ee->set_input_values(input_values);
				ERR_PRINT_OFF;
				CHECK(ee->execute(0.01666) == BTTask::FAILURE);
				ERR_PRINT_ON;
				CHECK(callback_counter->num_callbacks == 1);
			}
			SUBCASE("Should succeed with delta included") {
				ee->set_input_include_delta(true);
				ee->set_input_names(PackedStringArray());
				CHECK(ee->parse() == OK);
				ee->set_input_values(TypedArray<BBVariant>());
				ERR_PRINT_OFF;
				CHECK(ee->execute(0.01666) == BTTask::SUCCESS);
				ERR_PRINT_ON;
				CHECK(callback_counter->num_callbacks == 1);
			}
			SUBCASE("Should succeed with one float arg") {
				ee->set_input_include_delta(false);
				PackedStringArray input_names;
				input_names.push_back("delta");
				ee->set_input_names(input_names);
				CHECK(ee->parse() == OK);
				TypedArray<BBVariant> input_values;
				input_values.push_back(memnew(BBVariant(0.2)));
				ee->set_input_values(input_values);
				ERR_PRINT_OFF;
				CHECK(ee->execute(0.01666) == BTTask::SUCCESS);
				ERR_PRINT_ON;
				CHECK(callback_counter->num_callbacks == 1);
			}
		}

		SUBCASE("Should fail with too many method arguments") {
			ee->set_expression_string("callback_delta(delta, extra)");
			ee->set_input_include_delta(true);
			PackedStringArray input_names;
			input_names.push_back("point_two");
			ee->set_input_names(input_names);
			CHECK(ee->parse() == OK);
			TypedArray<BBVariant> input_values;
			input_values.push_back(memnew(BBVariant(0.2)));
			ee->set_input_values(input_values);
			ERR_PRINT_OFF;
			CHECK(ee->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
			CHECK(callback_counter->num_callbacks == 0);
		}

		SUBCASE("When toggling input_include_delta") {
			ee->set_expression_string("delta + extra");
			ee->set_result_var("sum_result");

			SUBCASE("Sum should be greater than 1 with input_include_delta set to false") {
				ee->set_input_include_delta(false);
				PackedStringArray input_names;
				input_names.push_back("delta");
				input_names.push_back("extra");
				ee->set_input_names(input_names);
				CHECK(ee->parse() == OK);
				TypedArray<BBVariant> input_values;
				input_values.push_back(memnew(BBVariant(0.01666)));
				input_values.push_back(memnew(BBVariant(1)));
				ee->set_input_values(input_values);
				ERR_PRINT_OFF;
				CHECK(ee->execute(0.01666) == BTTask::SUCCESS);
				CHECK(float(ee->get_blackboard()->get_var("sum_result", 0)) > 1);
				ERR_PRINT_ON;
				CHECK(callback_counter->num_callbacks == 0);
			}

			SUBCASE("Sum should be greater than 1 with input_include_delta set to true") {
				ee->set_input_include_delta(true);
				PackedStringArray input_names;
				input_names.push_back("extra");
				ee->set_input_names(input_names);
				CHECK(ee->parse() == OK);
				TypedArray<BBVariant> input_values;
				input_values.push_back(memnew(BBVariant(1)));
				ee->set_input_values(input_values);
				ERR_PRINT_OFF;
				CHECK(ee->execute(0.01666) == BTTask::SUCCESS);
				CHECK(float(ee->get_blackboard()->get_var("sum_result", 0)) > 1);
				ERR_PRINT_ON;
				CHECK(callback_counter->num_callbacks == 0);
			}
		}

		memdelete(dummy);
	}
}

} //namespace TestEvaluateExpression

#endif // TEST_EVALUATE_EXPRESSION_H
