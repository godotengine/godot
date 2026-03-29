#include "test_macros.h"

#include "../renderer/sort_fallback_policy.h"

using namespace GaussianSplatting;

TEST_CASE("[GaussianSplatting][SortFallback] force_cpu_sort policy stays deterministic across domains") {
	const SortFallbackPolicyDecision instance_policy =
			build_sort_fallback_policy(SortFallbackScenario::FORCE_CPU_OVERRIDE, true);
	CHECK(instance_policy.action_count == 3);
	CHECK(instance_policy.actions[0] == SortFallbackAction::REUSE_PREVIOUS_SORT);
	CHECK(instance_policy.actions[1] == SortFallbackAction::PUBLISH_INSTANCE_IDENTITY);
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

TEST_CASE("[GaussianSplatting][SortFallback] strict mode suppresses unsorted instance fallback publication") {
	CHECK(allow_unsorted_fallback_publication(false));
	CHECK_FALSE(allow_unsorted_fallback_publication(true));

	const SortFallbackPolicyDecision strict_force_cpu_instance_policy =
			build_sort_fallback_policy(SortFallbackScenario::FORCE_CPU_OVERRIDE, true, true);
	CHECK(strict_force_cpu_instance_policy.action_count == 2);
	CHECK(strict_force_cpu_instance_policy.actions[0] == SortFallbackAction::REUSE_PREVIOUS_SORT);
	CHECK(strict_force_cpu_instance_policy.actions[1] == SortFallbackAction::FAIL);
	CHECK(!strict_force_cpu_instance_policy.cpu_sort_forced);

	const SortFallbackPolicyDecision strict_gpu_failure_instance_policy =
			build_sort_fallback_policy(SortFallbackScenario::GPU_SORT_FAILED, true, true);
	CHECK(strict_gpu_failure_instance_policy.action_count == 2);
	CHECK(strict_gpu_failure_instance_policy.actions[0] == SortFallbackAction::REUSE_PREVIOUS_SORT);
	CHECK(strict_gpu_failure_instance_policy.actions[1] == SortFallbackAction::FAIL);
	CHECK(!strict_gpu_failure_instance_policy.cpu_sort_forced);
}

TEST_CASE("[GaussianSplatting][SortFallback] orchestrator guards block unsorted strict-mode fallbacks") {
	CHECK(allow_instance_identity_fallback_in_orchestrator(false, true, true, 1));
	CHECK_FALSE(allow_instance_identity_fallback_in_orchestrator(true, true, true, 1));
	CHECK_FALSE(allow_instance_identity_fallback_in_orchestrator(false, false, true, 1));
	CHECK_FALSE(allow_instance_identity_fallback_in_orchestrator(false, true, false, 1));
	CHECK_FALSE(allow_instance_identity_fallback_in_orchestrator(false, true, true, 0));

	CHECK(allow_camera_stable_cull_order_bootstrap_in_orchestrator(false, false, true, true));
	CHECK_FALSE(allow_camera_stable_cull_order_bootstrap_in_orchestrator(true, false, true, true));
	CHECK_FALSE(allow_camera_stable_cull_order_bootstrap_in_orchestrator(false, true, true, true));
	CHECK_FALSE(allow_camera_stable_cull_order_bootstrap_in_orchestrator(false, false, false, true));
	CHECK_FALSE(allow_camera_stable_cull_order_bootstrap_in_orchestrator(false, false, true, false));

	CHECK(allow_unsorted_cpu_fallback_in_orchestrator(false, false));
	CHECK(allow_unsorted_cpu_fallback_in_orchestrator(true, true));
	CHECK_FALSE(allow_unsorted_cpu_fallback_in_orchestrator(true, false));
}
