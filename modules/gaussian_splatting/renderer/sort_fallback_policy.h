#ifndef GAUSSIAN_SORT_FALLBACK_POLICY_H
#define GAUSSIAN_SORT_FALLBACK_POLICY_H

#include <stdint.h>

namespace GaussianSplatting {

enum class SortFallbackAction : uint8_t {
	REUSE_PREVIOUS_SORT = 0,
	PUBLISH_INSTANCE_IDENTITY = 1,
	RUN_CPU_SORT = 2,
	FAIL = 3,
};

enum class SortFallbackScenario : uint8_t {
	FORCE_CPU_OVERRIDE = 0,
	SORTER_UNAVAILABLE = 1,
	GPU_SORT_FAILED = 2,
};

struct SortFallbackPolicyDecision {
	SortFallbackAction actions[4] = {
		SortFallbackAction::FAIL,
		SortFallbackAction::FAIL,
		SortFallbackAction::FAIL,
		SortFallbackAction::FAIL,
	};
	uint32_t action_count = 0;
	bool cpu_sort_forced = false;
};

static inline bool allow_unsorted_fallback_publication(bool p_strict_global_sort) {
	return !p_strict_global_sort;
}

static inline bool allow_instance_identity_fallback_in_orchestrator(
		bool p_strict_global_sort, bool p_instance_pipeline_active, bool p_has_sorting_pipeline, uint32_t p_instance_visible_splats) {
	return !p_strict_global_sort && p_instance_pipeline_active && p_has_sorting_pipeline && p_instance_visible_splats > 0;
}

static inline bool allow_camera_stable_cull_order_bootstrap_in_orchestrator(
		bool p_strict_global_sort, bool p_instance_pipeline_active, bool p_has_culled_indices, bool p_has_sorting_pipeline) {
	return !p_strict_global_sort && !p_instance_pipeline_active && p_has_culled_indices && p_has_sorting_pipeline;
}

static inline bool allow_unsorted_cpu_fallback_in_orchestrator(bool p_strict_global_sort, bool p_positions_ready) {
	return p_positions_ready || !p_strict_global_sort;
}

// Keep fallback behavior deterministic across force_cpu and GPU failure paths.
static inline SortFallbackPolicyDecision build_sort_fallback_policy(
		SortFallbackScenario p_scenario, bool p_instance_pipeline_active, bool p_strict_global_sort = false) {
	SortFallbackPolicyDecision decision;
	auto push_action = [&](SortFallbackAction p_action) {
		if (decision.action_count < 4) {
			decision.actions[decision.action_count++] = p_action;
		}
	};

	switch (p_scenario) {
		case SortFallbackScenario::FORCE_CPU_OVERRIDE: {
			if (p_instance_pipeline_active) {
				push_action(SortFallbackAction::REUSE_PREVIOUS_SORT);
				if (allow_unsorted_fallback_publication(p_strict_global_sort)) {
					push_action(SortFallbackAction::PUBLISH_INSTANCE_IDENTITY);
				}
				push_action(SortFallbackAction::FAIL);
			} else {
				decision.cpu_sort_forced = true;
				push_action(SortFallbackAction::RUN_CPU_SORT);
				push_action(SortFallbackAction::FAIL);
			}
		} break;
		case SortFallbackScenario::SORTER_UNAVAILABLE:
		case SortFallbackScenario::GPU_SORT_FAILED: {
			push_action(SortFallbackAction::REUSE_PREVIOUS_SORT);
			if (p_instance_pipeline_active) {
				if (allow_unsorted_fallback_publication(p_strict_global_sort)) {
					push_action(SortFallbackAction::PUBLISH_INSTANCE_IDENTITY);
				}
			} else {
				push_action(SortFallbackAction::RUN_CPU_SORT);
			}
			push_action(SortFallbackAction::FAIL);
		} break;
	}

	return decision;
}

} // namespace GaussianSplatting

#endif // GAUSSIAN_SORT_FALLBACK_POLICY_H
