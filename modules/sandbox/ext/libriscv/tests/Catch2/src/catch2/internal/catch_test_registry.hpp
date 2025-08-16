
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TEST_REGISTRY_HPP_INCLUDED
#define CATCH_TEST_REGISTRY_HPP_INCLUDED

#include <catch2/internal/catch_config_static_analysis_support.hpp>
#include <catch2/internal/catch_source_line_info.hpp>
#include <catch2/internal/catch_noncopyable.hpp>
#include <catch2/interfaces/catch_interfaces_test_invoker.hpp>
#include <catch2/internal/catch_stringref.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_unique_name.hpp>
#include <catch2/internal/catch_preprocessor_remove_parens.hpp>

// GCC 5 and older do not properly handle disabling unused-variable warning
// with a _Pragma. This means that we have to leak the suppression to the
// user code as well :-(
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ <= 5
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif



namespace Catch {

template<typename C>
class TestInvokerAsMethod : public ITestInvoker {
    void (C::*m_testAsMethod)();
public:
    constexpr TestInvokerAsMethod( void ( C::*testAsMethod )() ) noexcept:
        m_testAsMethod( testAsMethod ) {}

    void invoke() const override {
        C obj;
        (obj.*m_testAsMethod)();
    }
};

Detail::unique_ptr<ITestInvoker> makeTestInvoker( void(*testAsFunction)() );

template<typename C>
Detail::unique_ptr<ITestInvoker> makeTestInvoker( void (C::*testAsMethod)() ) {
    return Detail::make_unique<TestInvokerAsMethod<C>>( testAsMethod );
}

template <typename C>
class TestInvokerFixture : public ITestInvoker {
    void ( C::*m_testAsMethod )() const;
    Detail::unique_ptr<C> m_fixture = nullptr;

public:
    constexpr TestInvokerFixture( void ( C::*testAsMethod )() const ) noexcept:
        m_testAsMethod( testAsMethod ) {}

    void prepareTestCase() override {
        m_fixture = Detail::make_unique<C>();
    }

    void tearDownTestCase() override {
        m_fixture.reset();
    }

    void invoke() const override {
        auto* f = m_fixture.get();
        ( f->*m_testAsMethod )();
    }
};

template<typename C>
Detail::unique_ptr<ITestInvoker> makeTestInvokerFixture( void ( C::*testAsMethod )() const ) {
    return Detail::make_unique<TestInvokerFixture<C>>( testAsMethod );
}

struct NameAndTags {
    constexpr NameAndTags( StringRef name_ = StringRef(),
                           StringRef tags_ = StringRef() ) noexcept:
        name( name_ ), tags( tags_ ) {}
    StringRef name;
    StringRef tags;
};

struct AutoReg : Detail::NonCopyable {
    AutoReg( Detail::unique_ptr<ITestInvoker> invoker, SourceLineInfo const& lineInfo, StringRef classOrMethod, NameAndTags const& nameAndTags ) noexcept;
};

} // end namespace Catch

#if defined(CATCH_CONFIG_DISABLE)
    #define INTERNAL_CATCH_TESTCASE_NO_REGISTRATION( TestName, ... ) \
        static inline void TestName()
    #define INTERNAL_CATCH_TESTCASE_METHOD_NO_REGISTRATION( TestName, ClassName, ... ) \
        namespace{                        \
            struct TestName : INTERNAL_CATCH_REMOVE_PARENS(ClassName) { \
                void test();              \
            };                            \
        }                                 \
        void TestName::test()
#endif


#if !defined(CATCH_CONFIG_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT)

    ///////////////////////////////////////////////////////////////////////////////
    #define INTERNAL_CATCH_TESTCASE2( TestName, ... ) \
        static void TestName(); \
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
        namespace{ const Catch::AutoReg INTERNAL_CATCH_UNIQUE_NAME( autoRegistrar )( Catch::makeTestInvoker( &TestName ), CATCH_INTERNAL_LINEINFO, Catch::StringRef(), Catch::NameAndTags{ __VA_ARGS__ } ); } /* NOLINT */ \
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
        static void TestName()
    #define INTERNAL_CATCH_TESTCASE( ... ) \
        INTERNAL_CATCH_TESTCASE2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), __VA_ARGS__ )

#else  // ^^ !CATCH_CONFIG_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT | vv CATCH_CONFIG_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT


// Dummy registrator for the dumy test case macros
namespace Catch {
    namespace Detail {
        struct DummyUse {
            DummyUse( void ( * )( int ), Catch::NameAndTags const& );
        };
    } // namespace Detail
} // namespace Catch

// Note that both the presence of the argument and its exact name are
// necessary for the section support.

// We provide a shadowed variable so that a `SECTION` inside non-`TEST_CASE`
// tests can compile. The redefined `TEST_CASE` shadows this with param.
static int catchInternalSectionHint = 0;

#    define INTERNAL_CATCH_TESTCASE2( fname, ... )                         \
        static void fname( int );                                          \
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION                          \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS                           \
        CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS                   \
        static const Catch::Detail::DummyUse INTERNAL_CATCH_UNIQUE_NAME(   \
            dummyUser )( &(fname), Catch::NameAndTags{ __VA_ARGS__ } );    \
        CATCH_INTERNAL_SUPPRESS_SHADOW_WARNINGS                            \
        static void fname( [[maybe_unused]] int catchInternalSectionHint ) \
            CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#    define INTERNAL_CATCH_TESTCASE( ... ) \
        INTERNAL_CATCH_TESTCASE2( INTERNAL_CATCH_UNIQUE_NAME( dummyFunction ), __VA_ARGS__ )


#endif // CATCH_CONFIG_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT

    ///////////////////////////////////////////////////////////////////////////////
    #define INTERNAL_CATCH_TEST_CASE_METHOD2( TestName, ClassName, ... )\
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
        namespace{ \
            struct TestName : INTERNAL_CATCH_REMOVE_PARENS(ClassName) { \
                void test(); \
            }; \
            const Catch::AutoReg INTERNAL_CATCH_UNIQUE_NAME( autoRegistrar )( \
            Catch::makeTestInvoker( &TestName::test ),                    \
            CATCH_INTERNAL_LINEINFO,                                      \
            #ClassName##_catch_sr,                                        \
            Catch::NameAndTags{ __VA_ARGS__ } ); /* NOLINT */ \
        } \
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
        void TestName::test()
    #define INTERNAL_CATCH_TEST_CASE_METHOD( ClassName, ... ) \
        INTERNAL_CATCH_TEST_CASE_METHOD2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), ClassName, __VA_ARGS__ )

    ///////////////////////////////////////////////////////////////////////////////
    #define INTERNAL_CATCH_TEST_CASE_PERSISTENT_FIXTURE2( TestName, ClassName, ... )      \
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION                             \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS                              \
        CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS                      \
        namespace {                                                           \
            struct TestName : INTERNAL_CATCH_REMOVE_PARENS( ClassName ) {     \
                void test() const;                                            \
            };                                                                \
            const Catch::AutoReg INTERNAL_CATCH_UNIQUE_NAME( autoRegistrar )( \
                Catch::makeTestInvokerFixture( &TestName::test ),                    \
                CATCH_INTERNAL_LINEINFO,                                      \
                #ClassName##_catch_sr,                                        \
                Catch::NameAndTags{ __VA_ARGS__ } ); /* NOLINT */             \
        }                                                                     \
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION                              \
        void TestName::test() const
    #define INTERNAL_CATCH_TEST_CASE_PERSISTENT_FIXTURE( ClassName, ... )    \
        INTERNAL_CATCH_TEST_CASE_PERSISTENT_FIXTURE2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), ClassName, __VA_ARGS__ )


    ///////////////////////////////////////////////////////////////////////////////
    #define INTERNAL_CATCH_METHOD_AS_TEST_CASE( QualifiedMethod, ... ) \
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
        namespace {                                                           \
        const Catch::AutoReg INTERNAL_CATCH_UNIQUE_NAME( autoRegistrar )( \
            Catch::makeTestInvoker( &QualifiedMethod ),                   \
            CATCH_INTERNAL_LINEINFO,                                      \
            "&" #QualifiedMethod##_catch_sr,                              \
            Catch::NameAndTags{ __VA_ARGS__ } );                          \
    } /* NOLINT */ \
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION


    ///////////////////////////////////////////////////////////////////////////////
    #define INTERNAL_CATCH_REGISTER_TESTCASE( Function, ... ) \
        do { \
            CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
            CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
            CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
            Catch::AutoReg INTERNAL_CATCH_UNIQUE_NAME( autoRegistrar )( Catch::makeTestInvoker( Function ), CATCH_INTERNAL_LINEINFO, Catch::StringRef(), Catch::NameAndTags{ __VA_ARGS__ } ); /* NOLINT */ \
            CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
        } while(false)


#endif // CATCH_TEST_REGISTRY_HPP_INCLUDED
