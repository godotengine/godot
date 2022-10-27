#include "future.h"

#define MAX_EXPECTED_FPS double(1000.0)
#define WAIT_TIME uint64_t((1.0 / MAX_EXPECTED_FPS) * 1000.0 * 1000.0)

void Future::_bind_methods(){
	ClassDB::bind_method(D_METHOD("get_value"), &Future::get_value);
	ClassDB::bind_method(D_METHOD("await"), &Future::await);
	ClassDB::bind_method(D_METHOD("is_available"), &Future::is_available);
	ClassDB::bind_method(D_METHOD("is_legit"), &Future::is_legit);
}

Variant Future::await() const {
	while (!available) {
		std::this_thread::sleep_for(std::chrono::microseconds(WAIT_TIME));
	}
	return get_value();
}
