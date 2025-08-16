
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TEMPLATE_TEST_REGISTRY_HPP_INCLUDED
#define CATCH_TEMPLATE_TEST_REGISTRY_HPP_INCLUDED

#include <catch2/internal/catch_test_registry.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>
#include <catch2/internal/catch_preprocessor.hpp>
#include <catch2/internal/catch_meta.hpp>
#include <catch2/internal/catch_unique_name.hpp>


// GCC 5 and older do not properly handle disabling unused-variable warning
// with a _Pragma. This means that we have to leak the suppression to the
// user code as well :-(
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ <= 5
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#if defined(CATCH_CONFIG_DISABLE)
    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION_2( TestName, TestFunc, Name, Tags, Signature, ... )  \
        INTERNAL_CATCH_DEFINE_SIG_TEST(TestFunc, INTERNAL_CATCH_REMOVE_PARENS(Signature))
    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION_2( TestNameClass, TestName, ClassName, Name, Tags, Signature, ... )    \
        namespace{                                                                                  \
            namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName) {                                      \
            INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD(TestName, ClassName, INTERNAL_CATCH_REMOVE_PARENS(Signature));\
        }                                                                                           \
        }                                                                                           \
        INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD(TestName, INTERNAL_CATCH_REMOVE_PARENS(Signature))

    #ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
        #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION(Name, Tags, ...) \
            INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename TestType, __VA_ARGS__ )
    #else
        #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION(Name, Tags, ...) \
            INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename TestType, __VA_ARGS__ ) )
    #endif

    #ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
        #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG_NO_REGISTRATION(Name, Tags, Signature, ...) \
            INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__ )
    #else
        #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG_NO_REGISTRATION(Name, Tags, Signature, ...) \
            INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__ ) )
    #endif

    #ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
        #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION( ClassName, Name, Tags,... ) \
            INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, typename T, __VA_ARGS__ )
    #else
        #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION( ClassName, Name, Tags,... ) \
            INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, typename T, __VA_ARGS__ ) )
    #endif

    #ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
        #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG_NO_REGISTRATION( ClassName, Name, Tags, Signature, ... ) \
            INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, Signature, __VA_ARGS__ )
    #else
        #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG_NO_REGISTRATION( ClassName, Name, Tags, Signature, ... ) \
            INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, Signature, __VA_ARGS__ ) )
    #endif
#endif


    ///////////////////////////////////////////////////////////////////////////////
    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_2(TestName, TestFunc, Name, Tags, Signature, ... )\
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS \
        INTERNAL_CATCH_DECLARE_SIG_TEST(TestFunc, INTERNAL_CATCH_REMOVE_PARENS(Signature));\
        namespace {\
        namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName){\
            INTERNAL_CATCH_TYPE_GEN\
            INTERNAL_CATCH_NTTP_GEN(INTERNAL_CATCH_REMOVE_PARENS(Signature))\
            INTERNAL_CATCH_NTTP_REG_GEN(TestFunc,INTERNAL_CATCH_REMOVE_PARENS(Signature))\
            template<typename...Types> \
            struct TestName{\
                TestName(){\
                    size_t index = 0;                                    \
                    constexpr char const* tmpl_types[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, __VA_ARGS__)}; /* NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,hicpp-avoid-c-arrays) */\
                    using expander = size_t[]; /* NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,hicpp-avoid-c-arrays) */\
                    (void)expander{(reg_test(Types{}, Catch::NameAndTags{ Name " - " + std::string(tmpl_types[index]), Tags } ), index++)... };/* NOLINT */ \
                }\
            };\
            static const int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){\
            TestName<INTERNAL_CATCH_MAKE_TYPE_LISTS_FROM_TYPES(__VA_ARGS__)>();\
            return 0;\
        }();\
        }\
        }\
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
        INTERNAL_CATCH_DEFINE_SIG_TEST(TestFunc,INTERNAL_CATCH_REMOVE_PARENS(Signature))

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE(Name, Tags, ...) \
        INTERNAL_CATCH_TEMPLATE_TEST_CASE_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename TestType, __VA_ARGS__ )
#else
    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE(Name, Tags, ...) \
        INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename TestType, __VA_ARGS__ ) )
#endif

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG(Name, Tags, Signature, ...) \
        INTERNAL_CATCH_TEMPLATE_TEST_CASE_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__ )
#else
    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG(Name, Tags, Signature, ...) \
        INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__ ) )
#endif

    #define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE2(TestName, TestFuncName, Name, Tags, Signature, TmplTypes, TypesList) \
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION                      \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS                      \
        CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS                \
        CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS       \
        CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS \
        template<typename TestType> static void TestFuncName();       \
        namespace {\
        namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName) {                                     \
            INTERNAL_CATCH_TYPE_GEN                                                  \
            INTERNAL_CATCH_NTTP_GEN(INTERNAL_CATCH_REMOVE_PARENS(Signature))         \
            template<typename... Types>                               \
            struct TestName {                                         \
                void reg_tests() {                                          \
                    size_t index = 0;                                    \
                    using expander = size_t[];                           \
                    constexpr char const* tmpl_types[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, INTERNAL_CATCH_REMOVE_PARENS(TmplTypes))};\
                    constexpr char const* types_list[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, INTERNAL_CATCH_REMOVE_PARENS(TypesList))};\
                    constexpr auto num_types = sizeof(types_list) / sizeof(types_list[0]);\
                    (void)expander{(Catch::AutoReg( Catch::makeTestInvoker( &TestFuncName<Types> ), CATCH_INTERNAL_LINEINFO, Catch::StringRef(), Catch::NameAndTags{ Name " - " + std::string(tmpl_types[index / num_types]) + '<' + types_list[index % num_types] + '>', Tags } ), index++)... };/* NOLINT */\
                }                                                     \
            };                                                        \
            static const int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){ \
                using TestInit = typename create<TestName, decltype(get_wrapper<INTERNAL_CATCH_REMOVE_PARENS(TmplTypes)>(Catch::Detail::priority_tag<1>{})), TypeList<INTERNAL_CATCH_MAKE_TYPE_LISTS_FROM_TYPES(INTERNAL_CATCH_REMOVE_PARENS(TypesList))>>::type; \
                TestInit t;                                           \
                t.reg_tests();                                        \
                return 0;                                             \
            }();                                                      \
        }                                                             \
        }                                                             \
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION                       \
        template<typename TestType>                                   \
        static void TestFuncName()

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
    #define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE(Name, Tags, ...)\
        INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE2(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename T,__VA_ARGS__)
#else
    #define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE(Name, Tags, ...)\
        INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename T, __VA_ARGS__ ) )
#endif

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
    #define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG(Name, Tags, Signature, ...)\
        INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE2(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__)
#else
    #define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG(Name, Tags, Signature, ...)\
        INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__ ) )
#endif

    #define INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_2(TestName, TestFunc, Name, Tags, TmplList)\
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS \
        template<typename TestType> static void TestFunc();       \
        namespace {\
        namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName){\
        INTERNAL_CATCH_TYPE_GEN\
        template<typename... Types>                               \
        struct TestName {                                         \
            void reg_tests() {                                          \
                size_t index = 0;                                    \
                using expander = size_t[];                           \
                (void)expander{(Catch::AutoReg( Catch::makeTestInvoker( &TestFunc<Types> ), CATCH_INTERNAL_LINEINFO, Catch::StringRef(), Catch::NameAndTags{ Name " - " INTERNAL_CATCH_STRINGIZE(TmplList) " - " + std::to_string(index), Tags } ), index++)... };/* NOLINT */\
            }                                                     \
        };\
        static const int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){ \
                using TestInit = typename convert<TestName, TmplList>::type; \
                TestInit t;                                           \
                t.reg_tests();                                        \
                return 0;                                             \
            }();                                                      \
        }}\
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION                       \
        template<typename TestType>                                   \
        static void TestFunc()

    #define INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE(Name, Tags, TmplList) \
        INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, TmplList )


    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_2( TestNameClass, TestName, ClassName, Name, Tags, Signature, ... ) \
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
        namespace {\
        namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName){ \
            INTERNAL_CATCH_TYPE_GEN\
            INTERNAL_CATCH_NTTP_GEN(INTERNAL_CATCH_REMOVE_PARENS(Signature))\
            INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD(TestName, ClassName, INTERNAL_CATCH_REMOVE_PARENS(Signature));\
            INTERNAL_CATCH_NTTP_REG_METHOD_GEN(TestName, INTERNAL_CATCH_REMOVE_PARENS(Signature))\
            template<typename...Types> \
            struct TestNameClass{\
                TestNameClass(){\
                    size_t index = 0;                                    \
                    constexpr char const* tmpl_types[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, __VA_ARGS__)};\
                    using expander = size_t[];\
                    (void)expander{(reg_test(Types{}, #ClassName, Catch::NameAndTags{ Name " - " + std::string(tmpl_types[index]), Tags } ), index++)... };/* NOLINT */ \
                }\
            };\
            static const int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){\
                TestNameClass<INTERNAL_CATCH_MAKE_TYPE_LISTS_FROM_TYPES(__VA_ARGS__)>();\
                return 0;\
        }();\
        }\
        }\
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
        INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD(TestName, INTERNAL_CATCH_REMOVE_PARENS(Signature))

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD( ClassName, Name, Tags,... ) \
        INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, typename T, __VA_ARGS__ )
#else
    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD( ClassName, Name, Tags,... ) \
        INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, typename T, __VA_ARGS__ ) )
#endif

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( ClassName, Name, Tags, Signature, ... ) \
        INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, Signature, __VA_ARGS__ )
#else
    #define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( ClassName, Name, Tags, Signature, ... ) \
        INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, Signature, __VA_ARGS__ ) )
#endif

    #define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_2(TestNameClass, TestName, ClassName, Name, Tags, Signature, TmplTypes, TypesList)\
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
        template<typename TestType> \
            struct TestName : INTERNAL_CATCH_REMOVE_PARENS(ClassName <TestType>) { \
                void test();\
            };\
        namespace {\
        namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestNameClass) {\
            INTERNAL_CATCH_TYPE_GEN                  \
            INTERNAL_CATCH_NTTP_GEN(INTERNAL_CATCH_REMOVE_PARENS(Signature))\
            template<typename...Types>\
            struct TestNameClass{\
                void reg_tests(){\
                    std::size_t index = 0;\
                    using expander = std::size_t[];\
                    constexpr char const* tmpl_types[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, INTERNAL_CATCH_REMOVE_PARENS(TmplTypes))};\
                    constexpr char const* types_list[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, INTERNAL_CATCH_REMOVE_PARENS(TypesList))};\
                    constexpr auto num_types = sizeof(types_list) / sizeof(types_list[0]);\
                    (void)expander{(Catch::AutoReg( Catch::makeTestInvoker( &TestName<Types>::test ), CATCH_INTERNAL_LINEINFO, #ClassName, Catch::NameAndTags{ Name " - " + std::string(tmpl_types[index / num_types]) + '<' + types_list[index % num_types] + '>', Tags } ), index++)... };/* NOLINT */ \
                }\
            };\
            static const int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){\
                using TestInit = typename create<TestNameClass, decltype(get_wrapper<INTERNAL_CATCH_REMOVE_PARENS(TmplTypes)>(Catch::Detail::priority_tag<1>{})), TypeList<INTERNAL_CATCH_MAKE_TYPE_LISTS_FROM_TYPES(INTERNAL_CATCH_REMOVE_PARENS(TypesList))>>::type;\
                TestInit t;\
                t.reg_tests();\
                return 0;\
            }(); \
        }\
        }\
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
        template<typename TestType> \
        void TestName<TestType>::test()

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
    #define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD( ClassName, Name, Tags, ... )\
        INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), ClassName, Name, Tags, typename T, __VA_ARGS__ )
#else
    #define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD( ClassName, Name, Tags, ... )\
        INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), ClassName, Name, Tags, typename T,__VA_ARGS__ ) )
#endif

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
    #define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( ClassName, Name, Tags, Signature, ... )\
        INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), ClassName, Name, Tags, Signature, __VA_ARGS__ )
#else
    #define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( ClassName, Name, Tags, Signature, ... )\
        INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), ClassName, Name, Tags, Signature,__VA_ARGS__ ) )
#endif

    #define INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD_2( TestNameClass, TestName, ClassName, Name, Tags, TmplList) \
        CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
        CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
        CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS \
        template<typename TestType> \
        struct TestName : INTERNAL_CATCH_REMOVE_PARENS(ClassName <TestType>) { \
            void test();\
        };\
        namespace {\
        namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName){ \
            INTERNAL_CATCH_TYPE_GEN\
            template<typename...Types>\
            struct TestNameClass{\
                void reg_tests(){\
                    size_t index = 0;\
                    using expander = size_t[];\
                    (void)expander{(Catch::AutoReg( Catch::makeTestInvoker( &TestName<Types>::test ), CATCH_INTERNAL_LINEINFO, #ClassName##_catch_sr, Catch::NameAndTags{ Name " - " INTERNAL_CATCH_STRINGIZE(TmplList) " - " + std::to_string(index), Tags } ), index++)... };/* NOLINT */ \
                }\
            };\
            static const int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){\
                using TestInit = typename convert<TestNameClass, TmplList>::type;\
                TestInit t;\
                t.reg_tests();\
                return 0;\
            }(); \
        }}\
        CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
        template<typename TestType> \
        void TestName<TestType>::test()

#define INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD(ClassName, Name, Tags, TmplList) \
        INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), ClassName, Name, Tags, TmplList )


#endif // CATCH_TEMPLATE_TEST_REGISTRY_HPP_INCLUDED
