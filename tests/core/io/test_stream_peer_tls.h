#pragma once

#include "core/io/stream_peer_tls.h"
#include "tests/test_macros.h"

namespace TestStreamPeerTLS {

TEST_CASE("[StreamPeerTLS] Availability and creation") {
    // Factory should report available
    CHECK(StreamPeerTLS::is_available() == true);

    // Create a backend instance
    StreamPeerTLS *tls = StreamPeerTLS::create();

    // Check that we got a valid object
    CHECK(tls != nullptr);
    CHECK_MESSAGE(tls != nullptr, "TLS backend should be registered and create() should return a valid object.");

    // Clean up the object to prevent ObjectDB leak
    memdelete(tls);
}

TEST_CASE("[StreamPeerTLS] Enum values") {
    CHECK(StreamPeerTLS::STATUS_DISCONNECTED == 0);
    CHECK(StreamPeerTLS::STATUS_HANDSHAKING == 1);
    CHECK(StreamPeerTLS::STATUS_CONNECTED == 2);
    CHECK(StreamPeerTLS::STATUS_ERROR == 3);
    CHECK(StreamPeerTLS::STATUS_ERROR_HOSTNAME_MISMATCH == 4);
}

} // namespace TestStreamPeerTLS
