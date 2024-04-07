#pragma once

#include "tests/test_macros.h"

#include "modules/unknoter/unknoterImpl.h"

namespace TestUnknoter {

TEST_CASE("[Modules][Unknoter] Test UnknoterImpl behavior") {
    UnknoterImpl unknoter;

    unknoter.reset(2, 5, 5);

    CHECK(unknoter.get_width() == 5);
    CHECK(unknoter.get_height() == 5);

    CHECK(unknoter.get_edge_player(1, 1) == -1);

    CHECK_FALSE(unknoter.can_player_shift_edges(0, 1, 1, 1, 0));

    unknoter._set_field({
        {0, -1, 0, -1, 0},
        {-1, -1, -1, -1, -1},
        {0, -1, 0, -1, 0},
        {-1, -1, -1, -1, -1},
        {0, -1, 0, -1, 0}});

    CHECK(unknoter.can_player_shift_edges(0, 1, 1, 1, 0));

    CHECK_FALSE(unknoter.can_player_shift_edges(1, 1, 1, 1, 0));

    CHECK(unknoter.get_edge_player(1, 2) == -1);
}

} // namespace TestUnknoter

