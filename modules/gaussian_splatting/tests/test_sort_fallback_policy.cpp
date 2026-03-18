#include "test_macros.h"

#include "../renderer/sort_fallback_policy.h"

using namespace GaussianSplatting;

TEST_CASE("[GaussianSplatting][SortFallback] force_cpu_sort policy stays deterministic across domains") {
	const SortFallbackPolicyDecision instance_policy =
			build_sort_fallback_policy(SortFallbackScenario::FORCE_CPU_OVERRIDE, true);
	CHECK(instance_policy.action_count == 3);
	CHECK(instance_policy.actions[0] == SortFallbackAction::PUBLISH_INSTANCE_IDENTITY);
	CHECK(instance_policy.actions[1] == SortFallbackAction::REUSE_PREVIOUS_SORT);
	CHECK(instance_policy.actions[2] == SortFallbackAction::FAIL);
	CHECK(!instance_policy.cpu_sort_forced);

	const SortFallbackPolicyDecision global_policy =
			build_sort_fallback_policy(SortFallbackScenario::FORCE_CPU_OVERRIDE, false);
	CHECK(global_policy.action_count == 2);
	CHECK(global_policy.actions[0] == SortFallbackAction::RUN_CPU_SORT);
	CHECK(global_policy.actions[1] == SortFallbackAction::FAIL);
	CHECK(global_policy.cpu_sort_forced);
}

TEST_CASE("[GaussianSplatting][SortFallback] GPU fallback policy prefers reuse then identity/CPU") {
	for (SortFallbackScenario scenario : { SortFallbackScenario::SORTER_UNAVAILABLE, SortFallbackScenario::GPU_SORT_FAILED }) {
		SUBCASE(scenario == SortFallbackScenario::SORTER_UNAVAILABLE ? "sorter unavailable" : "gpu sort failed") {
			const SortFallbackPolicyDecision instance_policy = build_sort_fallback_policy(scenario, true);
			CHECK(instance_policy.action_count == 3);
			CHECK(instance_policy.actions[0] == SortFallbackAction::REUSE_PREVIOUS_SORT);
			CHECK(instance_policy.actions[1] == SortFallbackAction::PUBLISH_INSTANCE_IDENTITY);
			CHECK(instance_policy.actions[2] == SortFallbackAction::FAIL);
			CHECK(!instance_policy.cpu_sort_forced);

			const SortFallbackPolicyDecision global_policy = build_sort_fallback_policy(scenario, false);
			CHECK(global_policy.action_count == 3);
			CHECK(global_policy.actions[0] == SortFallbackAction::REUSE_PREVIOUS_SORT);
			CHECK(global_policy.actions[1] == SortFallbackAction::RUN_CPU_SORT);
			CHECK(global_policy.actions[2] == SortFallbackAction::FAIL);
			CHECK(!global_policy.cpu_sort_forced);
		}
	}
}
