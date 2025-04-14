#include "modules/websocket/websocket_peer.h"
#include "tests/test_macros.h"

namespace TestWebSocketPeer {

TEST_CASE("[WebSocketPeer] after creation state is closed") {
	WebSocketPeer *peer = WebSocketPeer::create();
	CHECK_MESSAGE(peer != nullptr, "WebSocketPeer::create() returned nullptr");

	if (peer) {
		CHECK(peer->get_ready_state() == WebSocketPeer::STATE_CLOSED);
		memdelete(peer);
	}

}

} // namespace TestWebSocketPeer

