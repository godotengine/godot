/**
 * test_check_agent_property.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_CHECK_AGENT_PROPERTY_H
#define TEST_CHECK_AGENT_PROPERTY_H

#include "limbo_test.h"

#include "modules/limboai/blackboard/bb_param/bb_variant.h"
#include "modules/limboai/blackboard/blackboard.h"
#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/scene/bt_check_agent_property.h"
#include "modules/limboai/util/limbo_utility.h"

#include "core/os/memory.h"
#include "core/variant/variant.h"

namespace TestCheckAgentProperty {

// Check with m_correct, m_incorrect and m_invalid values using m_check_type.
#define TC_CHECK_AGENT_PROP(m_task, m_check_type, m_correct, m_incorrect, m_invalid) \
	m_task->set_check_type(m_check_type);                                            \
	m_task->get_value()->set_saved_value(m_correct);                                 \
	CHECK(m_task->execute(0.01666) == BTTask::SUCCESS);                              \
	m_task->get_value()->set_saved_value(m_incorrect);                               \
	CHECK(m_task->execute(0.01666) == BTTask::FAILURE);                              \
	m_task->get_value()->set_saved_value(m_invalid);                                 \
	CHECK(m_task->execute(0.01666) == BTTask::FAILURE);

TEST_CASE("[Modules][LimboAI] BTCheckAgentProperty") {
	Ref<BTCheckAgentProperty> cap = memnew(BTCheckAgentProperty);
	Node *agent = memnew(Node);
	Ref<Blackboard> bb = memnew(Blackboard);
	cap->initialize(agent, bb, agent);
	StringName agent_name = "SimpleNode";
	agent->set_name(agent_name);

	// * Defaults that should produce successful check:
	cap->set_property("name");
	cap->set_check_type(LimboUtility::CHECK_EQUAL);
	Ref<BBVariant> value = memnew(BBVariant);
	cap->set_value(value);
	value->set_saved_value(agent_name);
	REQUIRE(cap->execute(0.01666) == BTTask::SUCCESS);

	SUBCASE("When property is not set") {
		cap->set_property("");
		ERR_PRINT_OFF;
		CHECK(cap->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}
	SUBCASE("When property is not found") {
		cap->set_property("not_found");
		ERR_PRINT_OFF;
		CHECK(cap->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}
	SUBCASE("When value is not set") {
		cap->set_value(nullptr);

		ERR_PRINT_OFF;
		CHECK(cap->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}
	SUBCASE("With StringName") {
		StringName other_name = "OtherName";
		TC_CHECK_AGENT_PROP(cap, LimboUtility::CHECK_EQUAL, agent_name, other_name, 123);
		TC_CHECK_AGENT_PROP(cap, LimboUtility::CHECK_NOT_EQUAL, other_name, agent_name, 123);
	}
	SUBCASE("With integer") {
		cap->set_property("process_priority");
		TC_CHECK_AGENT_PROP(cap, LimboUtility::CHECK_EQUAL, 0, -1, "invalid");
		TC_CHECK_AGENT_PROP(cap, LimboUtility::CHECK_GREATER_THAN_OR_EQUAL, 0, 1, "invalid");
		TC_CHECK_AGENT_PROP(cap, LimboUtility::CHECK_GREATER_THAN, -1, 1, "invalid");
		TC_CHECK_AGENT_PROP(cap, LimboUtility::CHECK_LESS_THAN_OR_EQUAL, 0, -1, "invalid");
		TC_CHECK_AGENT_PROP(cap, LimboUtility::CHECK_LESS_THAN, 1, 0, "invalid");
		TC_CHECK_AGENT_PROP(cap, LimboUtility::CHECK_NOT_EQUAL, 1, 0, "invalid");
	}

	memdelete(agent);
}

} //namespace TestCheckAgentProperty

#endif // TEST_CHECK_AGENT_PROPERTY_H
