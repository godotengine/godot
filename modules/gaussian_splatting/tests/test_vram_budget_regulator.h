#pragma once

#include "test_macros.h"

#include "../core/gaussian_streaming.h"
#include "core/math/math_funcs.h"

TEST_CASE("[GaussianSplatting][VRAMBudgetRegulator] Unknown device capacity remains explicit") {
    Ref<VRAMBudgetRegulator> regulator;
    regulator.instantiate();
    REQUIRE(regulator.is_valid());

    VRAMBudgetConfig override_config;
    override_config.budget_mb = 1234;
    override_config.min_chunks = 3;
    override_config.max_chunks = 9;
    override_config.auto_regulate_enabled = false;

    regulator->set_config_override(override_config);
    regulator->initialize(nullptr);

    Dictionary stats = regulator->get_debug_stats_dictionary();
    CHECK_FALSE(bool(stats.get("device_memory_queryable", true)));
    CHECK_FALSE(bool(stats.get("device_total_known", true)));
    CHECK(uint64_t(stats.get("device_total_bytes", uint64_t(1))) == uint64_t(0));
    CHECK(uint64_t(stats.get("device_reported_bytes", uint64_t(1))) == uint64_t(0));

    const float budget_mb = float(stats.get("budget_mb", -1.0f));
    CHECK(Math::is_equal_approx(budget_mb, 1234.0f));
    CHECK(regulator->get_current_max_chunks() == 9u);
}
