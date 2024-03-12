#pragma once

#include "tests/test_macros.h"

#include "modules/unknoter/unknoterNode.h"

namespace TestUnknoter {

TEST_CASE("[Modules][Unknoter] Adding numbers") {
    UnknoterNode unknoter;
	CHECK(unknoter.add(15, 27) == 42);
	CHECK(unknoter.add(27, 15) == 42);
}

} // namespace TestUnknoter