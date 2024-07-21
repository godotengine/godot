/**
 * test_check_var.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
#ifndef TEST_CHECK_VAR_H
#define TEST_CHECK_VAR_H

#include "limbo_test.h"

#include "modules/limboai/blackboard/bb_param/bb_param.h"
#include "modules/limboai/bt/tasks/blackboard/bt_check_var.h"
#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/util/limbo_utility.h"
#include "tests/test_macros.h"

namespace TestCheckVar {

// Compare m_correct, m_incorrect and m_invalid to m_value based using m_check_type.
#define TC_CHECK_VALUES(m_task, m_correct, m_incorrect, m_invalid, m_check_type, m_value) \
	m_task->get_value()->set_saved_value(m_value);                                        \
	m_task->set_check_type(m_check_type);                                                 \
	m_task->get_blackboard()->set_var("var", m_correct);                                  \
	CHECK(m_task->execute(0.01666) == BTTask::SUCCESS);                                   \
	m_task->get_blackboard()->set_var("var", m_incorrect);                                \
	CHECK(m_task->execute(0.01666) == BTTask::FAILURE);                                   \
	m_task->get_blackboard()->set_var("var", m_invalid);                                  \
	CHECK(m_task->execute(0.01666) == BTTask::FAILURE);

TEST_CASE("[Modules][LimboAI] BTCheckVar") {
	Ref<BTCheckVar> cv = memnew(BTCheckVar);
	Ref<Blackboard> bb = memnew(Blackboard);
	Node *dummy = memnew(Node);
	cv->initialize(dummy, bb, dummy);

	SUBCASE("Check with empty variable and value") {
		cv->set_variable("");
		cv->set_value(nullptr);
		ERR_PRINT_OFF;
		CHECK(cv->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}
	SUBCASE("With variable and value set") {
		cv->set_variable("var");
		Ref<BBVariant> value = memnew(BBVariant);
		cv->set_value(value);

		SUBCASE("When checking against another variable") {
			cv->set_check_type(LimboUtility::CHECK_EQUAL);
			value->set_value_source(BBParam::BLACKBOARD_VAR);
			bb->set_var("var", 123);
			SUBCASE("When variable exists") {
				value->set_variable("compare_var");
				bb->set_var("compare_var", 123);
				CHECK(cv->execute(0.01666) == BTTask::SUCCESS);
				bb->set_var("compare_var", 567);
				CHECK(cv->execute(0.01666) == BTTask::FAILURE);
			}
			SUBCASE("When variable doesn't exist") {
				value->set_variable("not_found");
				ERR_PRINT_OFF;
				CHECK(cv->execute(0.01666) == BTTask::FAILURE);
				ERR_PRINT_ON;
			}
		}

		value->set_value_source(BBParam::SAVED_VALUE);

		SUBCASE("With integer") {
			TC_CHECK_VALUES(cv, 5, 4, "5", LimboUtility::CHECK_EQUAL, 5);
			TC_CHECK_VALUES(cv, 5, 4, "5", LimboUtility::CHECK_GREATER_THAN_OR_EQUAL, 5);
			TC_CHECK_VALUES(cv, 6, 4, "6", LimboUtility::CHECK_GREATER_THAN, 5);
			TC_CHECK_VALUES(cv, 5, 6, "5", LimboUtility::CHECK_LESS_THAN_OR_EQUAL, 5);
			TC_CHECK_VALUES(cv, 4, 6, "4", LimboUtility::CHECK_LESS_THAN, 5);
			TC_CHECK_VALUES(cv, 4, 5, "4", LimboUtility::CHECK_NOT_EQUAL, 5);
		}
		SUBCASE("With bool") {
			TC_CHECK_VALUES(cv, true, false, "true", LimboUtility::CHECK_EQUAL, true);
			TC_CHECK_VALUES(cv, true, false, "true", LimboUtility::CHECK_NOT_EQUAL, false);
		}
		SUBCASE("With float") {
			TC_CHECK_VALUES(cv, 3.14, 3.0, "3.14", LimboUtility::CHECK_EQUAL, 3.14);
			TC_CHECK_VALUES(cv, 3.14, 3.0, "3.14", LimboUtility::CHECK_GREATER_THAN_OR_EQUAL, 3.14);
			TC_CHECK_VALUES(cv, 4.0, 3.0, "4.0", LimboUtility::CHECK_GREATER_THAN, 3.14);
			TC_CHECK_VALUES(cv, 3.14, 4.0, "3.14", LimboUtility::CHECK_LESS_THAN_OR_EQUAL, 3.14);
			TC_CHECK_VALUES(cv, 3.0, 4.0, "3.0", LimboUtility::CHECK_LESS_THAN, 3.14);
			TC_CHECK_VALUES(cv, 3.0, 3.14, "3.0", LimboUtility::CHECK_NOT_EQUAL, 3.14);
		}
		SUBCASE("With string") {
			TC_CHECK_VALUES(cv, "AAA", "AAC", 123, LimboUtility::CHECK_EQUAL, "AAA");
			TC_CHECK_VALUES(cv, "AAC", "AAA", 123, LimboUtility::CHECK_GREATER_THAN_OR_EQUAL, "AAB");
			TC_CHECK_VALUES(cv, "AAC", "AAA", 123, LimboUtility::CHECK_GREATER_THAN, "AAB");
			TC_CHECK_VALUES(cv, "AAA", "AAC", 123, LimboUtility::CHECK_LESS_THAN_OR_EQUAL, "AAB");
			TC_CHECK_VALUES(cv, "AAA", "AAC", 123, LimboUtility::CHECK_LESS_THAN, "AAB");
			TC_CHECK_VALUES(cv, "AAA", "AAB", 123, LimboUtility::CHECK_NOT_EQUAL, "AAB");
		}
	}

	memdelete(dummy);
}

} //namespace TestCheckVar

#endif // TEST_CHECK_VAR_H
