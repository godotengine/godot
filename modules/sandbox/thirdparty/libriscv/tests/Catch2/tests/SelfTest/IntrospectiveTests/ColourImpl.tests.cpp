
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_console_colour.hpp>
#include <catch2/internal/catch_istream.hpp>

#include <sstream>

namespace {
    class TestColourImpl : public Catch::ColourImpl {
        using Catch::ColourImpl::ColourImpl;
        // Inherited via ColourImpl
        void use( Catch::Colour::Code colourCode ) const override {
            m_stream->stream() << "Using code: " << colourCode << '\n';
        }
    };

    class TestStringStream : public Catch::IStream {
        std::stringstream m_stream;
    public:
        std::ostream& stream() override {
            return m_stream;
        }

        std::string str() const { return m_stream.str(); }
    };
}

TEST_CASE("ColourGuard behaviour", "[console-colours]") {
    TestStringStream streamWrapper;
    TestColourImpl colourImpl( &streamWrapper );
    auto& stream = streamWrapper.stream();

    SECTION("ColourGuard is disengaged by default") {
        { auto guard = colourImpl.guardColour( Catch::Colour::Red ); }

        REQUIRE( streamWrapper.str().empty() );
    }

    SECTION("ColourGuard is engaged by op<<") {
        stream << "1\n" << colourImpl.guardColour( Catch::Colour::Red ) << "2\n";
        stream << "3\n";

        REQUIRE( streamWrapper.str() == "1\nUsing code: 2\n2\nUsing code: 0\n3\n" );
    }

    SECTION("ColourGuard can be engaged explicitly") {
        {
            auto guard =
                colourImpl.guardColour( Catch::Colour::Red ).engage( stream );
            stream << "A\n"
                   << "B\n";
        }
        stream << "C\n";
        REQUIRE( streamWrapper.str() ==
                 "Using code: 2\nA\nB\nUsing code: 0\nC\n" );
    }
}
