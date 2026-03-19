#include "../core/gaussian_streaming.h"

#include "test_macros.h"

TEST_CASE("[Streaming Pipeline] stop_pack_threads clears partial lifecycle state") {
    GaussianStreamingSystem system;
    GaussianStreamingSystem::UploadQueueState &uploads = system._internal_get_upload_state();

    uploads.pack_thread_running.store(false, std::memory_order_release);
    uploads.pack_thread_exit.store(true, std::memory_order_release);
    uploads.pack_threads.resize(2);
    uploads.pack_thread_contexts.resize(2);
    uploads.pack_threads[0] = nullptr;
    uploads.pack_threads[1] = nullptr;

    uploads.stop_pack_threads(system);

    CHECK(uploads.pack_threads.is_empty());
    CHECK(uploads.pack_thread_contexts.is_empty());
    CHECK_FALSE(uploads.pack_thread_running.load(std::memory_order_acquire));
    CHECK_FALSE(uploads.pack_thread_exit.load(std::memory_order_acquire));
}
