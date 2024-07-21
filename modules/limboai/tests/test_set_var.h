/**
 * test_set_var.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_SET_VAR_H
#define TEST_SET_VAR_H

#include "core/variant/variant.h"
#include "limbo_test.h"

#include "modules/limboai/blackboard/bb_param/bb_param.h"
#include "modules/limboai/blackboard/bb_param/bb_variant.h"
#include "modules/limboai/blackboard/blackboard.h"
#include "modules/limboai/bt/tasks/blackboard/bt_set_var.h"
#include "modules/limboai/bt/tasks/bt_task.h"
#include "tests/test_macros.h"

namespace TestSetVar {

TEST_CASE("[Modules][LimboAI] BTSetVar") {
	Ref<BTSetVar> sv = memnew(BTSetVar);
	Ref<Blackboard> bb = memnew(Blackboard);
	Node *dummy = memnew(Node);

	sv->initialize(dummy, bb, dummy);

	SUBCASE("When variable is not set") {
		ERR_PRINT_OFF;
		sv->set_variable("");
		CHECK(sv->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	SUBCASE("With variable set") {
		Ref<BBVariant> value = memnew(BBVariant);
		sv->set_value(value);
		sv->set_variable("var");

		SUBCASE("When assigning a raw value") {
			value->set_value_source(BBParam::SAVED_VALUE);
			value->set_saved_value(123);
			CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
			CHECK(bb->get_var("var", 0) == Variant(123));
		}
		SUBCASE("When assigning value of another blackboard variable") {
			value->set_value_source(BBParam::BLACKBOARD_VAR);

			SUBCASE("BB variable is empty") {
				ERR_PRINT_OFF;
				value->set_variable("");
				CHECK(sv->execute(0.01666) == BTTask::FAILURE);
				ERR_PRINT_ON;
			}
			SUBCASE("BB variable doesn't exist") {
				ERR_PRINT_OFF;
				Variant initial_value = Variant(777);
				bb->set_var("var", initial_value);
				value->set_variable("not_found");
				CHECK(sv->execute(0.01666) == BTTask::FAILURE);
				CHECK(bb->get_var("var", 0) == initial_value); // * Check initial value is intact.
				ERR_PRINT_ON;
			}
			SUBCASE("BB variable exists") {
				value->set_variable("compare_var");
				bb->set_var("compare_var", 123);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(123));
			}
		}
		SUBCASE("When performing an operation") {
			bb->set_var("var", 8);
			value->set_value_source(BBParam::SAVED_VALUE);
			value->set_saved_value(3);

			SUBCASE("Addition") {
				sv->set_operation(LimboUtility::OPERATION_ADDITION);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(11));
			}
			SUBCASE("Subtraction") {
				sv->set_operation(LimboUtility::OPERATION_SUBTRACTION);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(5));
			}
			SUBCASE("Multiplication") {
				sv->set_operation(LimboUtility::OPERATION_MULTIPLICATION);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(24));
			}
			SUBCASE("Division") {
				sv->set_operation(LimboUtility::OPERATION_DIVISION);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(2));
			}
			SUBCASE("Modulo") {
				sv->set_operation(LimboUtility::OPERATION_MODULO);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(2));
			}
			SUBCASE("Power") {
				sv->set_operation(LimboUtility::OPERATION_POWER);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(512));
			}
			SUBCASE("Bitwise shift left") {
				sv->set_operation(LimboUtility::OPERATION_BIT_SHIFT_LEFT);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(64));
			}
			SUBCASE("Bitwise shift right") {
				sv->set_operation(LimboUtility::OPERATION_BIT_SHIFT_RIGHT);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(1));
			}
			SUBCASE("Bitwise AND") {
				bb->set_var("var", 6);
				sv->set_operation(LimboUtility::OPERATION_BIT_AND);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(2));
			}
			SUBCASE("Bitwise OR") {
				bb->set_var("var", 6);
				sv->set_operation(LimboUtility::OPERATION_BIT_OR);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(7));
			}
			SUBCASE("Bitwise XOR") {
				bb->set_var("var", 6);
				sv->set_operation(LimboUtility::OPERATION_BIT_XOR);
				CHECK(sv->execute(0.01666) == BTTask::SUCCESS);
				CHECK(bb->get_var("var", 0) == Variant(5));
			}
		}
		SUBCASE("Performing an operation when assigned variable doesn't exist.") {
			value->set_value_source(BBParam::SAVED_VALUE);
			value->set_saved_value(3);
			sv->set_operation(LimboUtility::OPERATION_ADDITION);

			ERR_PRINT_OFF;
			CHECK(sv->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
			CHECK_FALSE(bb->has_var("var"));
		}
		SUBCASE("Performing an operation with incompatible operand types.") {
			bb->set_var("var", 2); // int
			value->set_value_source(BBParam::SAVED_VALUE);
			value->set_saved_value("3"); // String
			sv->set_operation(LimboUtility::OPERATION_ADDITION);

			ERR_PRINT_OFF;
			CHECK(sv->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
			CHECK(bb->get_var("var", 0) == Variant(2));
		}
	}
}

} //namespace TestSetVar

#endif // TEST_SET_VAR_H
