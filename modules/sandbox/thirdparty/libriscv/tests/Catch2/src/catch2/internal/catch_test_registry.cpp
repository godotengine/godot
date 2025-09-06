
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/internal/catch_test_registry.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/interfaces/catch_interfaces_registry_hub.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

#include <algorithm>
#include <iterator>

namespace Catch {
    void ITestInvoker::prepareTestCase() {}
    void ITestInvoker::tearDownTestCase() {}
    ITestInvoker::~ITestInvoker() = default;

    namespace {
        static StringRef extractClassName( StringRef classOrMethodName ) {
            if ( !startsWith( classOrMethodName, '&' ) ) {
                return classOrMethodName;
            }

            // Remove the leading '&' to avoid having to special case it later
            const auto methodName =
                classOrMethodName.substr( 1, classOrMethodName.size() );

            auto reverseStart = std::make_reverse_iterator( methodName.end() );
            auto reverseEnd = std::make_reverse_iterator( methodName.begin() );

            // We make a simplifying assumption that ":" is only present
            // in the input as part of "::" from C++ typenames (this is
            // relatively safe assumption because the input is generated
            // as stringification of type through preprocessor).
            auto lastColons = std::find( reverseStart, reverseEnd, ':' ) + 1;
            auto secondLastColons =
                std::find( lastColons + 1, reverseEnd, ':' );

            auto const startIdx = reverseEnd - secondLastColons;
            auto const classNameSize = secondLastColons - lastColons - 1;

            return methodName.substr(
                static_cast<std::size_t>( startIdx ),
                static_cast<std::size_t>( classNameSize ) );
        }

        class TestInvokerAsFunction final : public ITestInvoker {
            using TestType = void ( * )();
            TestType m_testAsFunction;

        public:
            constexpr TestInvokerAsFunction( TestType testAsFunction ) noexcept:
                m_testAsFunction( testAsFunction ) {}

            void invoke() const override { m_testAsFunction(); }
        };

    } // namespace

    Detail::unique_ptr<ITestInvoker> makeTestInvoker( void(*testAsFunction)() ) {
        return Detail::make_unique<TestInvokerAsFunction>( testAsFunction );
    }

    AutoReg::AutoReg( Detail::unique_ptr<ITestInvoker> invoker, SourceLineInfo const& lineInfo, StringRef classOrMethod, NameAndTags const& nameAndTags ) noexcept {
        CATCH_TRY {
            getMutableRegistryHub()
                    .registerTest(
                        makeTestCaseInfo(
                            extractClassName( classOrMethod ),
                            nameAndTags,
                            lineInfo),
                        CATCH_MOVE(invoker)
                    );
        } CATCH_CATCH_ALL {
            // Do not throw when constructing global objects, instead register the exception to be processed later
            getMutableRegistryHub().registerStartupException();
        }
    }
}
