#ifndef SORTING_CONFIG_H
#define SORTING_CONFIG_H

#include <cstdint>

#include "core/string/string_name.h"
#include "core/string/ustring.h"

class ProjectSettings;

struct SortingStrategyConfig {
    enum class ForcedAlgorithm : uint8_t {
        AUTO = 0,
        RADIX = 1,
        BITONIC = 2,
        ONESWEEP = 3
    };

    uint32_t bitonic_max_elements = 131072;
    uint32_t radix_max_elements = 5000000;
    uint32_t onesweep_max_elements = 10000000;
    uint32_t hybrid_trigger_elements = 10000000;
    uint32_t hybrid_batch_size = 5000000;
    uint32_t history_size = 120;
    uint32_t log_interval_frames = 60;
    float target_sort_time_ms = 2.0f;
    bool log_metrics = true;
    ForcedAlgorithm force_algorithm = ForcedAlgorithm::AUTO;
    bool force_cpu_sort = false;

    void sanitize();
    String describe_thresholds() const;
    bool is_algorithm_forced() const;
    String get_forced_algorithm_name() const;

    static SortingStrategyConfig load_from_project_settings();
};

#endif // SORTING_CONFIG_H
