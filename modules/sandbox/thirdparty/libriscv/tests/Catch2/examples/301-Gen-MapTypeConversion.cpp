
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

// 301-Gen-MapTypeConversion.cpp
// Shows how to use map to modify generator's return type.

// Specifically we wrap a std::string returning generator with a generator
// that converts the strings using stoi, so the returned type is actually
// an int.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include <string>
#include <sstream>

namespace {

// Returns a line from a stream. You could have it e.g. read lines from
// a file, but to avoid problems with paths in examples, we will use
// a fixed stringstream.
class LineGenerator final : public Catch::Generators::IGenerator<std::string> {
    std::string m_line;
    std::stringstream m_stream;
public:
    explicit LineGenerator( std::string const& lines ) {
        m_stream.str( lines );
        if (!next()) {
            Catch::Generators::Detail::throw_generator_exception("Couldn't read a single line");
        }
    }

    std::string const& get() const override;

    bool next() override {
        return !!std::getline(m_stream, m_line);
    }
};

std::string const& LineGenerator::get() const {
    return m_line;
}

// This helper function provides a nicer UX when instantiating the generator
// Notice that it returns an instance of GeneratorWrapper<std::string>, which
// is a value-wrapper around std::unique_ptr<IGenerator<std::string>>.
Catch::Generators::GeneratorWrapper<std::string>
lines( std::string const& lines ) {
    return Catch::Generators::GeneratorWrapper<std::string>(
        new LineGenerator( lines ) );
}

} // end anonymous namespace


TEST_CASE("filter can convert types inside the generator expression", "[example][generator]") {
    auto num = GENERATE(
        map<int>( []( std::string const& line ) { return std::stoi( line ); },
                  lines( "1\n2\n3\n4\n" ) ) );

    REQUIRE(num > 0);
}

// Compiling and running this file will result in 4 successful assertions
