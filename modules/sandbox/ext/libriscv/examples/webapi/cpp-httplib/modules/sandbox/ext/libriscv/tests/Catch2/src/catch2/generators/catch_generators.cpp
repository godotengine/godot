
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/generators/catch_generators.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/generators/catch_generator_exception.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>

namespace Catch {

    IGeneratorTracker::~IGeneratorTracker() = default;

namespace Generators {

namespace Detail {

    [[noreturn]]
    void throw_generator_exception(char const* msg) {
        Catch::throw_exception(GeneratorException{ msg });
    }
} // end namespace Detail

    GeneratorUntypedBase::~GeneratorUntypedBase() = default;

    IGeneratorTracker* acquireGeneratorTracker(StringRef generatorName, SourceLineInfo const& lineInfo ) {
        return getResultCapture().acquireGeneratorTracker( generatorName, lineInfo );
    }

    IGeneratorTracker* createGeneratorTracker( StringRef generatorName,
                                 SourceLineInfo lineInfo,
                                 GeneratorBasePtr&& generator ) {
        return getResultCapture().createGeneratorTracker(
            generatorName, lineInfo, CATCH_MOVE( generator ) );
    }

} // namespace Generators
} // namespace Catch
