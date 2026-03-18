#include "residency_budget_controller.h"

#include "core/error/error_macros.h"

namespace {

static bool _admission_gate_invariant_holds(const ResidencyBudgetController::AdmissionContext &p_context,
        ResidencyBudgetController::AdmissionDecision p_decision) {
    const bool at_or_over_capacity = p_context.loaded_chunks >= p_context.effective_max;
    const bool can_evict_for_load = p_context.has_eviction_budget && !p_context.eviction_blocked;
    const bool vram_regulator_denies_load =
            p_context.enforce_vram_regulator_gate && !p_context.vram_regulator_allows_load;

    if (!at_or_over_capacity && !vram_regulator_denies_load) {
        return p_decision == ResidencyBudgetController::AdmissionDecision::LoadDirect;
    }
    if (at_or_over_capacity && p_context.can_replace_without_eviction && !vram_regulator_denies_load) {
        return p_decision == ResidencyBudgetController::AdmissionDecision::LoadDirect;
    }
    if (can_evict_for_load) {
        return p_decision == ResidencyBudgetController::AdmissionDecision::EvictThenLoad;
    }
    return p_decision == ResidencyBudgetController::AdmissionDecision::Skip;
}

} // namespace

ResidencyBudgetController::AdmissionFrameBudget ResidencyBudgetController::make_frame_budget(
        uint32_t p_effective_max, uint32_t p_evictions_left, bool p_eviction_blocked) {
    AdmissionFrameBudget budget;
    budget.effective_max = p_effective_max;
    budget.evictions_left = p_evictions_left;
    budget.eviction_blocked = p_eviction_blocked;
    return budget;
}

ResidencyBudgetController::AdmissionGate ResidencyBudgetController::compute_admission_gate(
        uint32_t p_loaded_chunks,
        const AdmissionFrameBudget &p_frame_budget,
        const AdmissionPolicy &p_policy) {
    AdmissionGate gate;
    gate.context.loaded_chunks = p_loaded_chunks;
    gate.context.effective_max = p_frame_budget.effective_max;
    gate.context.can_replace_without_eviction = p_policy.can_replace_without_eviction;
    gate.context.has_eviction_budget = p_frame_budget.evictions_left > 0;
    gate.context.eviction_blocked = p_frame_budget.eviction_blocked;
    gate.context.enforce_vram_regulator_gate = p_policy.enforce_vram_regulator_gate;
    gate.context.vram_regulator_allows_load = p_policy.vram_regulator_allows_load;
    gate.decision = decide_admission(gate.context);
    if (!_admission_gate_invariant_holds(gate.context, gate.decision)) {
        ERR_PRINT("[Streaming][Invariant] Residency admission gate produced invalid decision; forcing Skip.");
        gate.decision = AdmissionDecision::Skip;
    }
    return gate;
}

void ResidencyBudgetController::note_successful_eviction(AdmissionFrameBudget &p_frame_budget) {
    ERR_FAIL_COND_MSG(p_frame_budget.evictions_left == 0,
            "[Streaming][Invariant] Residency eviction budget underflow.");
    p_frame_budget.evictions_left--;
}

void ResidencyBudgetController::note_blocked_eviction(AdmissionFrameBudget &p_frame_budget) {
    p_frame_budget.eviction_blocked = true;
}

bool ResidencyBudgetController::should_attempt_visible_evict_fallback(const AdmissionGate &p_gate) {
    if (p_gate.decision != AdmissionDecision::EvictThenLoad) {
        return false;
    }

    const AdmissionContext &context = p_gate.context;
    if (!context.has_eviction_budget || context.eviction_blocked) {
        return false;
    }

    const bool at_or_over_capacity = context.loaded_chunks >= context.effective_max;
    const bool regulator_gated =
            context.enforce_vram_regulator_gate && !context.vram_regulator_allows_load;
    return at_or_over_capacity || regulator_gated;
}

ResidencyBudgetController::AdmissionDecision ResidencyBudgetController::decide_admission(
        const AdmissionContext &p_context) {
    const bool at_or_over_capacity = p_context.loaded_chunks >= p_context.effective_max;
    const bool can_evict_for_load = p_context.has_eviction_budget && !p_context.eviction_blocked;
    const bool vram_regulator_denies_load =
            p_context.enforce_vram_regulator_gate && !p_context.vram_regulator_allows_load;

    if (at_or_over_capacity) {
        if (p_context.can_replace_without_eviction && !vram_regulator_denies_load) {
            return AdmissionDecision::LoadDirect;
        }
        if (can_evict_for_load) {
            return AdmissionDecision::EvictThenLoad;
        }
        return AdmissionDecision::Skip;
    }

    if (vram_regulator_denies_load) {
        if (can_evict_for_load) {
            return AdmissionDecision::EvictThenLoad;
        }
        return AdmissionDecision::Skip;
    }

    return AdmissionDecision::LoadDirect;
}
