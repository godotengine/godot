#ifndef RESIDENCY_BUDGET_CONTROLLER_H
#define RESIDENCY_BUDGET_CONTROLLER_H

#include <cstdint>

class ResidencyBudgetController {
public:
    enum class AdmissionDecision : uint8_t {
        LoadDirect = 0,
        EvictThenLoad = 1,
        Skip = 2,
    };

    struct AdmissionFrameBudget {
        uint32_t effective_max = 0;
        uint32_t evictions_left = 0;
        bool eviction_blocked = false;
    };

    struct AdmissionPolicy {
        bool can_replace_without_eviction = false;
        bool enforce_vram_regulator_gate = false;
        bool vram_regulator_allows_load = true;
    };

    struct AdmissionContext {
        uint32_t loaded_chunks = 0;
        uint32_t effective_max = 0;
        bool can_replace_without_eviction = false;
        bool has_eviction_budget = false;
        bool eviction_blocked = false;
        bool enforce_vram_regulator_gate = false;
        bool vram_regulator_allows_load = true;
    };

    struct AdmissionGate {
        AdmissionContext context;
        AdmissionDecision decision = AdmissionDecision::Skip;
    };

    // Invariants:
    // - Decision must be deterministic from AdmissionContext.
    // - note_successful_eviction() must not underflow evictions_left.
    static AdmissionFrameBudget make_frame_budget(uint32_t p_effective_max, uint32_t p_evictions_left, bool p_eviction_blocked);
    static AdmissionGate compute_admission_gate(
            uint32_t p_loaded_chunks,
            const AdmissionFrameBudget &p_frame_budget,
            const AdmissionPolicy &p_policy);
    static void note_successful_eviction(AdmissionFrameBudget &p_frame_budget);
    static void note_blocked_eviction(AdmissionFrameBudget &p_frame_budget);
    // Visible-eviction fallback is valid only when admission required eviction due
    // to hard chunk-cap pressure or regulator-gated pressure.
    static bool should_attempt_visible_evict_fallback(const AdmissionGate &p_gate);
    static AdmissionDecision decide_admission(const AdmissionContext &p_context);
};

#endif // RESIDENCY_BUDGET_CONTROLLER_H
