/**
 * test_set_agent_property.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_SET_AGENT_PROPERTY_H
#define TEST_SET_AGENT_PROPERTY_H

#include "limbo_test.h"

#include "modules/limboai/blackboard/bb_param/bb_param.h"
#include "modules/limboai/blackboard/bb_param/bb_variant.h"
#include "modules/limboai/blackboard/blackboard.h"
#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/scene/bt_set_agent_property.h"

#include "core/os/memory.h"

namespace TestSetAgentProperty {

TEST_CASE("[Modules][LimboAI] BTSetAgentProperty") {
	Ref<BTSetAgentProperty> sap = memnew(BTSetAgentProperty);
	Node *agent = memnew(Node);
	Ref<Blackboard> bb = memnew(Blackboard);
	sap->initialize(agent, bb, agent);

	sap->set_property("process_priority"); // * property that will be set by the task
	Ref<BBVariant> value = memnew(BBVariant);
	value->set_value_source(BBParam::SAVED_VALUE);
	value->set_saved_value(7);
	sap->set_value(value);

	SUBCASE("With integer") {
		CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
		CHECK(agent->get_process_priority() == 7);
	}
	SUBCASE("When value is not set") {
		sap->set_value(nullptr);
		ERR_PRINT_OFF;
		CHECK(sap->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}
	SUBCASE("When property is empty") {
		sap->set_property("");
		ERR_PRINT_OFF;
		CHECK(sap->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}
	SUBCASE("When property doesn't exist") {
		sap->set_property("not_found");
		ERR_PRINT_OFF;
		CHECK(sap->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}
	SUBCASE("With StringName and String") {
		value->set_saved_value("TestName");
		sap->set_property("name");
		CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
		CHECK(agent->get_name() == "TestName");
	}
	SUBCASE("With blackboard variable") {
		value->set_value_source(BBParam::BLACKBOARD_VAR);
		value->set_variable("priority");

		SUBCASE("With proper BB variable") {
			bb->set_var("priority", 8);
			CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
			CHECK(agent->get_process_priority() == 8);
		}
		SUBCASE("With BB variable of wrong type") {
			bb->set_var("priority", "high");
			ERR_PRINT_OFF;
			CHECK(sap->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
			CHECK(agent->get_process_priority() == 0);
		}
		SUBCASE("When BB variable doesn't exist") {
			value->set_variable("not_found");
			ERR_PRINT_OFF;
			CHECK(sap->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
			CHECK(agent->get_process_priority() == 0);
		}
		SUBCASE("When BB variable isn't set") {
			value->set_variable("");
			ERR_PRINT_OFF;
			CHECK(sap->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
			CHECK(agent->get_process_priority() == 0);
		}
		SUBCASE("When performing an operation") {
			agent->set_process_priority(8);
			value->set_value_source(BBParam::SAVED_VALUE);
			value->set_saved_value(3);

			SUBCASE("Addition") {
				sap->set_operation(LimboUtility::OPERATION_ADDITION);
				CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
				CHECK(agent->get_process_priority() == 11);
			}
			SUBCASE("Subtraction") {
				sap->set_operation(LimboUtility::OPERATION_SUBTRACTION);
				CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
				CHECK(agent->get_process_priority() == 5);
			}
			SUBCASE("Multiplication") {
				sap->set_operation(LimboUtility::OPERATION_MULTIPLICATION);
				CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
				CHECK(agent->get_process_priority() == 24);
			}
			SUBCASE("Division") {
				sap->set_operation(LimboUtility::OPERATION_DIVISION);
				CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
				CHECK(agent->get_process_priority() == 2);
			}
			SUBCASE("Modulo") {
				sap->set_operation(LimboUtility::OPERATION_MODULO);
				CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
				CHECK(agent->get_process_priority() == 2);
			}
			SUBCASE("Power") {
				sap->set_operation(LimboUtility::OPERATION_POWER);
				CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
				CHECK(agent->get_process_priority() == 512);
			}
			SUBCASE("Bitwise shift left") {
				sap->set_operation(LimboUtility::OPERATION_BIT_SHIFT_LEFT);
				CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
				CHECK(agent->get_process_priority() == 64);
			}
			SUBCASE("Bitwise shift right") {
				sap->set_operation(LimboUtility::OPERATION_BIT_SHIFT_RIGHT);
				CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
				CHECK(agent->get_process_priority() == 1);
			}
			SUBCASE("Bitwise AND") {
				agent->set_process_priority(6);
				sap->set_operation(LimboUtility::OPERATION_BIT_AND);
				CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
				CHECK(agent->get_process_priority() == 2);
			}
			SUBCASE("Bitwise OR") {
				agent->set_process_priority(6);
				sap->set_operation(LimboUtility::OPERATION_BIT_OR);
				CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
				CHECK(agent->get_process_priority() == 7);
			}
			SUBCASE("Bitwise XOR") {
				agent->set_process_priority(6);
				sap->set_operation(LimboUtility::OPERATION_BIT_XOR);
				CHECK(sap->execute(0.01666) == BTTask::SUCCESS);
				CHECK(agent->get_process_priority() == 5);
			}
		}
		SUBCASE("Performing an operation when assigned variable doesn't exist.") {
			sap->set_property("not_found");
			value->set_value_source(BBParam::SAVED_VALUE);
			value->set_saved_value(3);
			sap->set_operation(LimboUtility::OPERATION_ADDITION);

			ERR_PRINT_OFF;
			CHECK(sap->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
		}
		SUBCASE("Performing an operation with incompatible operand types.") {
			agent->set_process_priority(2);
			value->set_value_source(BBParam::SAVED_VALUE);
			value->set_saved_value("3"); // String
			sap->set_operation(LimboUtility::OPERATION_ADDITION);

			ERR_PRINT_OFF;
			CHECK(sap->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
			CHECK(agent->get_process_priority() == 2);
		}
	}

	memdelete(agent);
}

} //namespace TestSetAgentProperty

#endif // TEST_SET_AGENT_PROPERTY_H
