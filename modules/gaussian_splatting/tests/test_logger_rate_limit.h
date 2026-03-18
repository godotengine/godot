#pragma once

#include "test_macros.h"
#include "../logger/gs_logger.h"

TEST_CASE("[Gaussian Logger] Rate limiter keys low-severity logs by category level and fingerprint") {
    using namespace gs_logger;

    test::reset_rate_limiter();

    const uint64_t t0 = 1'000'000;
    const uint64_t window = 500'000;
    const String message = "shared-fingerprint";

    CHECK(test::check_rate_limit(Category::STREAMING, Level::INFO, message, t0, window));
    CHECK_FALSE(test::check_rate_limit(Category::STREAMING, Level::INFO, message, t0 + 1000, window));

    // WARN/ERROR are a separate severity path and should not be blocked by INFO chatter.
    CHECK(test::check_rate_limit(Category::STREAMING, Level::WARN, message, t0 + 2000, window));
    CHECK(test::check_rate_limit(Category::STREAMING, Level::ERROR, message, t0 + 3000, window));

    // Different message fingerprint should not be suppressed either.
    CHECK(test::check_rate_limit(Category::STREAMING, Level::INFO, "different-fingerprint", t0 + 4000, window));
}

TEST_CASE("[Gaussian Logger] Rate limiter does not suppress high-severity repeats") {
    using namespace gs_logger;

    test::reset_rate_limiter();

    const uint64_t t0 = 2'000'000;
    const uint64_t window = 500'000;
    const String message = "high-severity";

    CHECK(test::check_rate_limit(Category::RENDERER, Level::WARN, message, t0, window));
    CHECK(test::check_rate_limit(Category::RENDERER, Level::WARN, message, t0 + 1000, window));
    CHECK(test::check_rate_limit(Category::RENDERER, Level::ERROR, message, t0 + 2000, window));
    CHECK(test::check_rate_limit(Category::RENDERER, Level::ERROR, message, t0 + 3000, window));
}
