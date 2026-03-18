#pragma once

#include "test_macros.h"
#include "../renderer/rendering_diagnostics.h"
#include "../renderer/rendering_error.h"

TEST_CASE("[Gaussian Diagnostics] Singleton initialization is idempotent") {
    GaussianRenderingDiagnostics::ensure_singleton();
    GaussianRenderingDiagnostics *first = GaussianRenderingDiagnostics::get_singleton();
    REQUIRE(first != nullptr);

    GaussianRenderingDiagnostics::ensure_singleton();
    GaussianRenderingDiagnostics *second = GaussianRenderingDiagnostics::get_singleton();
    CHECK(second == first);
}

TEST_CASE("[Gaussian Diagnostics] Null renderer notifications are safe no-ops") {
    GaussianRenderingDiagnostics::ensure_singleton();
    GaussianRenderingDiagnostics *diagnostics = GaussianRenderingDiagnostics::get_singleton();
    REQUIRE(diagnostics != nullptr);

    RenderingError error;
    diagnostics->register_renderer(nullptr);
    diagnostics->unregister_renderer(nullptr);
    diagnostics->notify_error(nullptr, error);
    diagnostics->notify_recovery(nullptr, error);
    diagnostics->notify_frame_completed(nullptr);
    diagnostics->request_runtime_report();

    CHECK(true);
}
