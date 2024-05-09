//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef JSMATERIALX_HELPERS_H
#define JSMATERIALX_HELPERS_H

#include <vector>

using stRef = const std::string&;

// Binding helpers

#define UNPACK(...) __VA_ARGS__

/**
 * Use this macro to conveniently create bindings for class member functions with optional parameters.
 * If this is the last member on a class binding, finalize the invocation with a semicolon, e.g. BIND_MEMBER_FUNC(...);
 * NOTE: The macro expects the MaterialX scope to be available as 'mx', and the emscripten scope as 'ems'!
 * @param JSNAME The name of the function in JavaScript, as a double-quoted string (e.g. "addNodeGraph").
 * @param CLASSNAME The name (and scope) of the class that the member functions belongs to (e.g. mx::Document).
 * @param FUNCNAME The name of the function to bind (e.g. addNodeGraph).
 * @param MINARGS The minimal number of parameters that need to provided when calling the function (a.k.a # of required parameters).
 * @param MAXARGS The total number of parameters that the function takes, including optional ones.
 * @param ... The types of all parameters, as a comma-separated list (e.g. const std::string&, float, bool)
 */
#define BIND_MEMBER_FUNC(JSNAME, CLASSNAME, FUNCNAME, MINARGS, MAXARGS, ...) \
BIND_ ##MINARGS ## _ ##MAXARGS(.function, JSNAME, CLASSNAME &self, (,), self., FUNCNAME, ( ), , __VA_ARGS__)

/**
 * Use this macro to conveniently create bindings for class member functions with optional parameters,
 * where some of the parameters are raw pointers.
 * If this is the last member on a class binding, finalize the invocation with a semicolon, e.g. BIND_MEMBER_FUNC_RAW_PTR(...);
 * NOTE: The macro expects the MaterialX scope to be available as 'mx', and the emscripten scope as 'ems'!
 * @param JSNAME The name of the function in JavaScript, as a double-quoted string (e.g. "addNodeGraph").
 * @param CLASSNAME The name (and scope) of the class that the member functions belongs to (e.g. mx::Document).
 * @param FUNCNAME The name of the function to bind (e.g. addNodeGraph).
 * @param MINARGS The minimal number of parameters that need to provided when calling the function (a.k.a # of required parameters).
 * @param MAXARGS The total number of parameters that the function takes, including optional ones.
 * @param ... The types of all parameters, as a comma-separated list (e.g. const std::string&, float, bool)
 */
#define BIND_MEMBER_FUNC_RAW_PTR(JSNAME, CLASSNAME, FUNCNAME, MINARGS, MAXARGS, ...) \
BIND_ ##MINARGS ## _ ##MAXARGS(.function, JSNAME, CLASSNAME &self, (,), self., FUNCNAME, (, ems::allow_raw_pointers()), , __VA_ARGS__)


/**
 * Use this macro to conveniently create bindings for static class functions with optional parameters.
 * If this is the last member on a class binding, finalize the invocation with a semicolon, e.g. BIND_CLASS_FUNC(...);
 * NOTE: The macro expects the MaterialX scope to be available as 'mx', and the emscripten scope as 'ems'!
 * @param JSNAME The name of the function in JavaScript, as a double-quoted string (e.g. "addNodeGraph").
 * @param CLASSNAME The name (and scope) of the class that the member functions belongs to (e.g. mx::Document).
 * @param FUNCNAME The name of the function to bind (e.g. addNodeGraph).
 * @param MINARGS The minimal number of parameters that need to provided when calling the function (a.k.a # of required parameters).
 * @param MAXARGS The total number of parameters that the function takes, including optional ones.
 * @param ... The types of all parameters, as a comma-separated list (e.g. const std::string&, float, bool)
 */
#define BIND_CLASS_FUNC(JSNAME, CLASSNAME, FUNCNAME, MINARGS, MAXARGS, ...) \
BIND_ ##MINARGS ## _ ##MAXARGS(.class_function, JSNAME, , ( ), CLASSNAME ::, FUNCNAME, ( ), , __VA_ARGS__)

/**
 * Use this macro to conveniently create bindings for static class functions with optional parameters,
 * where some of the parameters are raw pointers.
 * If this is the last member on a class binding, finalize the invocation with a semicolon, e.g. BIND_CLASS_FUNC_RAW_PTR(...);
 * NOTE: The macro expects the MaterialX scope to be available as 'mx', and the emscripten scope as 'ems'!
 * @param JSNAME The name of the function in JavaScript, as a double-quoted string (e.g. "addNodeGraph").
 * @param CLASSNAME The name (and scope) of the class that the member functions belongs to (e.g. mx::Document).
 * @param FUNCNAME The name of the function to bind (e.g. addNodeGraph).
 * @param MINARGS The minimal number of parameters that need to provided when calling the function (a.k.a # of required parameters).
 * @param MAXARGS The total number of parameters that the function takes, including optional ones.
 * @param ... The types of all parameters, as a comma-separated list (e.g. const std::string&, float, bool)
 */
#define BIND_CLASS_FUNC_RAW_PTR(JSNAME, CLASSNAME, FUNCNAME, MINARGS, MAXARGS, ...) \
BIND_ ##MINARGS ## _ ##MAXARGS(.class_function, JSNAME, , ( ), CLASSNAME ::, FUNCNAME, (, ems::allow_raw_pointers()), , __VA_ARGS__)


/**
 * Use this macro to conveniently create bindings for global (utility) functions with optional parameters.
 * Do not finalize the invocation with a semicolon, e.g. simply use it as BIND_FUNC(...)
 * NOTE: The macro expects the MaterialX scope to be available as 'mx', and the emscripten scope as 'ems'!
 * @param JSNAME The name of the function in JavaScript, as a double-quoted string (e.g. "createDocument").
 * @param FUNCNAME The name of the function to bind (e.g. createDocument).
 * @param MINARGS The minimal number of parameters that need to provided when calling the function (a.k.a # of required parameters).
 * @param MAXARGS The total number of parameters that the function takes, including optional ones.
 * @param ... The types of all parameters, as a comma-separated list (e.g. const std::string&, float, bool)
 */
#define BIND_FUNC(JSNAME, FUNCNAME, MINARGS, MAXARGS, ...) \
BIND_ ##MINARGS ## _ ##MAXARGS(ems::function, JSNAME, , ( ), , FUNCNAME, ( ), ;, __VA_ARGS__)

/**
 * Use this macro to conveniently create bindings for global (utility) functions with optional parameters,
 * where some of the parameters are raw pointers.
 * Do not finalize the invocation with a semicolon, e.g. simply use it as BIND_FUNC_RAW_PTR(...).
 * NOTE: The macro expects the MaterialX scope to be available as 'mx', and the emscripten scope as 'ems'!
 * @param JSNAME The name of the function in JavaScript, as a double-quoted string (e.g. "createDocument").
 * @param FUNCNAME The name of the function to bind (e.g. createDocument).
 * @param MINARGS The minimal number of parameters that need to provided when calling the function (a.k.a # of required parameters).
 * @param MAXARGS The total number of parameters that the function takes, including optional ones.
 * @param ... The types of all parameters, as a comma-separated list (e.g. const std::string&, float, bool)
 */
#define BIND_FUNC_RAW_PTR(JSNAME, FUNCNAME, MINARGS, MAXARGS, ...) \
BIND_ ##MINARGS ## _ ##MAXARGS(ems::function, JSNAME, , ( ), , FUNCNAME, (, ems::allow_raw_pointers()), ;, __VA_ARGS__)


#define BIND_8(API, JSNAME, SELF, SEP, SCOPE, FUNCNAME, OPTIONS, SEMICOLON, \
  TYPE1, TYPE2, TYPE3, TYPE4, TYPE5, TYPE6, TYPE7, TYPE8, ...) \
API(JSNAME, ems::optional_override([](SELF UNPACK SEP \
  TYPE1 p1, TYPE2 p2, TYPE3 p3, TYPE4 p4, TYPE5 p5, TYPE6 p6, TYPE7 p7, TYPE8 p8) { \
    return SCOPE FUNCNAME(p1, p2, p3, p4, p5, p6, p7, p8); \
})UNPACK OPTIONS)SEMICOLON

// 9 Macros for MAXARGS = 8
#define BIND_8_8(...) \
BIND_8(__VA_ARGS__)

#define BIND_7_8(...) \
BIND_8_8(__VA_ARGS__) \
BIND_7(__VA_ARGS__)

#define BIND_6_8(...) \
BIND_7_8(__VA_ARGS__) \
BIND_6(__VA_ARGS__)

#define BIND_5_8(...) \
BIND_6_8(__VA_ARGS__) \
BIND_5(__VA_ARGS__)

#define BIND_4_8(...) \
BIND_5_8(__VA_ARGS__) \
BIND_4(__VA_ARGS__)

#define BIND_3_8(...) \
BIND_4_8(__VA_ARGS__) \
BIND_3(__VA_ARGS__)

#define BIND_2_8(...) \
BIND_3_8(__VA_ARGS__) \
BIND_2(__VA_ARGS__)

#define BIND_1_8(...) \
BIND_2_8(__VA_ARGS__) \
BIND_1(__VA_ARGS__)

#define BIND_0_8(...) \
BIND_1_8(__VA_ARGS__) \
BIND_0(__VA_ARGS__)


#define BIND_7(API, JSNAME, SELF, SEP, SCOPE, FUNCNAME, OPTIONS, SEMICOLON, \
  TYPE1, TYPE2, TYPE3, TYPE4, TYPE5, TYPE6, TYPE7, ...) \
API(JSNAME, ems::optional_override([](SELF UNPACK SEP \
  TYPE1 p1, TYPE2 p2, TYPE3 p3, TYPE4 p4, TYPE5 p5, TYPE6 p6, TYPE7 p7) { \
    return SCOPE FUNCNAME(p1, p2, p3, p4, p5, p6, p7); \
}) UNPACK OPTIONS)SEMICOLON

// 8 Macros for MAXARGS = 7
#define BIND_7_7(...) \
BIND_7(__VA_ARGS__)

#define BIND_6_7(...) \
BIND_7_7(__VA_ARGS__) \
BIND_6(__VA_ARGS__)

#define BIND_5_7(...) \
BIND_6_7(__VA_ARGS__) \
BIND_5(__VA_ARGS__)

#define BIND_4_7(...) \
BIND_5_7(__VA_ARGS__) \
BIND_4(__VA_ARGS__)

#define BIND_3_7(...) \
BIND_4_7(__VA_ARGS__) \
BIND_3(__VA_ARGS__)

#define BIND_2_7(...) \
BIND_3_7(__VA_ARGS__) \
BIND_2(__VA_ARGS__)

#define BIND_1_7(...) \
BIND_2_7(__VA_ARGS__) \
BIND_1(__VA_ARGS__)

#define BIND_0_7(...) \
BIND_1_7(__VA_ARGS__) \
BIND_0(__VA_ARGS__)


#define BIND_6(API, JSNAME, SELF, SEP, SCOPE, FUNCNAME, OPTIONS, SEMICOLON, \
  TYPE1, TYPE2, TYPE3, TYPE4, TYPE5, TYPE6, ...) \
API(JSNAME, ems::optional_override([](SELF UNPACK SEP \
  TYPE1 p1, TYPE2 p2, TYPE3 p3, TYPE4 p4, TYPE5 p5, TYPE6 p6) { \
    return SCOPE FUNCNAME(p1, p2, p3, p4, p5, p6); \
}) UNPACK OPTIONS)SEMICOLON

// 7 Macros for MAXARGS = 6
#define BIND_6_6(...) \
BIND_6(__VA_ARGS__)

#define BIND_5_6(...) \
BIND_6_6(__VA_ARGS__) \
BIND_5(__VA_ARGS__)

#define BIND_4_6(...) \
BIND_5_6(__VA_ARGS__) \
BIND_4(__VA_ARGS__)

#define BIND_3_6(...) \
BIND_4_6(__VA_ARGS__) \
BIND_3(__VA_ARGS__)

#define BIND_2_6(...) \
BIND_3_6(__VA_ARGS__) \
BIND_2(__VA_ARGS__)

#define BIND_1_6(...) \
BIND_2_6(__VA_ARGS__) \
BIND_1(__VA_ARGS__)

#define BIND_0_6(...) \
BIND_1_6(__VA_ARGS__) \
BIND_0(__VA_ARGS__)


#define BIND_5(API, JSNAME, SELF, SEP, SCOPE, FUNCNAME, OPTIONS, SEMICOLON, TYPE1, TYPE2, TYPE3, TYPE4, TYPE5, ...) \
API(JSNAME, ems::optional_override([](SELF UNPACK SEP \
  TYPE1 p1, TYPE2 p2, TYPE3 p3, TYPE4 p4, TYPE5 p5) { \
    return SCOPE FUNCNAME(p1, p2, p3, p4, p5); \
}) UNPACK OPTIONS)SEMICOLON

// 6 Macros for MAXARGS = 5
#define BIND_5_5(...) \
BIND_5(__VA_ARGS__)

#define BIND_4_5(...) \
BIND_5_5(__VA_ARGS__) \
BIND_4(__VA_ARGS__)

#define BIND_3_5(...) \
BIND_4_5(__VA_ARGS__) \
BIND_3(__VA_ARGS__)

#define BIND_2_5(...) \
BIND_3_5(__VA_ARGS__) \
BIND_2(__VA_ARGS__)

#define BIND_1_5(...) \
BIND_2_5(__VA_ARGS__) \
BIND_1(__VA_ARGS__)

#define BIND_0_5(...) \
BIND_1_5(__VA_ARGS__) \
BIND_0(__VA_ARGS__)


#define BIND_4(API, JSNAME, SELF, SEP, SCOPE, FUNCNAME, OPTIONS, SEMICOLON, TYPE1, TYPE2, TYPE3, TYPE4, ...) \
API(JSNAME, ems::optional_override([](SELF UNPACK SEP \
  TYPE1 p1, TYPE2 p2, TYPE3 p3, TYPE4 p4) { \
    return SCOPE FUNCNAME(p1, p2, p3, p4); \
}) UNPACK OPTIONS)SEMICOLON

// 5 Macros for MAXARGS = 4
#define BIND_4_4(...) \
BIND_4(__VA_ARGS__)

#define BIND_3_4(...) \
BIND_4_4(__VA_ARGS__) \
BIND_3(__VA_ARGS__)

#define BIND_2_4(...) \
BIND_3_4(__VA_ARGS__) \
BIND_2(__VA_ARGS__)

#define BIND_1_4(...) \
BIND_2_4(__VA_ARGS__) \
BIND_1(__VA_ARGS__)

#define BIND_0_4(...) \
BIND_1_4(__VA_ARGS__) \
BIND_0(__VA_ARGS__)


#define BIND_3(API, JSNAME, SELF, SEP, SCOPE, FUNCNAME, OPTIONS, SEMICOLON, TYPE1, TYPE2, TYPE3, ...) \
API(JSNAME, ems::optional_override([](SELF UNPACK SEP \
  TYPE1 p1, TYPE2 p2, TYPE3 p3) { \
    return SCOPE FUNCNAME(p1, p2, p3); \
}) UNPACK OPTIONS)SEMICOLON

// 4 Macros for MAXARGS = 3
#define BIND_3_3(...) \
BIND_3(__VA_ARGS__)

#define BIND_2_3(...) \
BIND_3_3(__VA_ARGS__) \
BIND_2(__VA_ARGS__)

#define BIND_1_3(...) \
BIND_2_3(__VA_ARGS__) \
BIND_1(__VA_ARGS__)

#define BIND_0_3(...) \
BIND_1_3(__VA_ARGS__) \
BIND_0(__VA_ARGS__)


#define BIND_2(API, JSNAME, SELF, SEP, SCOPE, FUNCNAME, OPTIONS, SEMICOLON, TYPE1, TYPE2, ...) \
API(JSNAME, ems::optional_override([](SELF UNPACK SEP \
  TYPE1 p1, TYPE2 p2) { \
    return SCOPE FUNCNAME(p1, p2); \
}) UNPACK OPTIONS)SEMICOLON

// 3 Macros for MAXARGS = 2
#define BIND_2_2(...) \
BIND_2(__VA_ARGS__)

#define BIND_1_2(...) \
BIND_2_2(__VA_ARGS__) \
BIND_1(__VA_ARGS__)

#define BIND_0_2(...) \
BIND_1_2(__VA_ARGS__) \
BIND_0(__VA_ARGS__)


#define BIND_1(API, JSNAME, SELF, SEP, SCOPE, FUNCNAME, OPTIONS, SEMICOLON, TYPE1, ...) \
API(JSNAME, ems::optional_override([](SELF UNPACK SEP \
  TYPE1 p1) { \
    return SCOPE FUNCNAME(p1); \
}) UNPACK OPTIONS)SEMICOLON

// 2 Macros for MAXARGS = 1
#define BIND_1_1(...) \
BIND_1(__VA_ARGS__)

#define BIND_0_1(...) \
BIND_1_1(__VA_ARGS__) \
BIND_0(__VA_ARGS__)


#define BIND_0(API, JSNAME, SELF, SEP, SCOPE, FUNCNAME, OPTIONS, SEMICOLON, ...) \
API(JSNAME, ems::optional_override([](SELF) { \
    return SCOPE FUNCNAME(); \
}) UNPACK OPTIONS)SEMICOLON

// 1 Macros for MAXARGS = 0
#define BIND_0_0(...) \
BIND_0(__VA_ARGS__)


/* Creates a wrapper converting a member function returning a reference to one returning
   a pointer. Emscripten converts references to copies and the wrapper emulates
   the expected behavior */
template<typename RetTypeRef, RetTypeRef method> struct ptrReturnHelper;
template<
    typename Class,
    typename RetType,
    typename... ArgType,
    RetType &(Class::*method)(ArgType...)
> struct ptrReturnHelper<RetType &(Class::*)(ArgType...), method> {
    auto getWrapper()->auto(*)(Class &, ArgType...)->RetType * {
        return [](Class &obj, ArgType ...arg) -> RetType * {
            return &(obj.*method)(arg...);
        };
    }
};

/**
 * Creates a wrapper converting a non-overloaded member function returning a reference to one returning a pointer.
 * @param METHOD Function pointer to the member function
 */
#define PTR_RETURN(METHOD) \
    (ptrReturnHelper<decltype(METHOD),METHOD>().getWrapper())

/**
 * Creates a wrapper converting an overloaded member function returning a reference to one returning a pointer.
 * @param DECL   Declaration of the function to wrap
 * @param METHOD Function pointer to the member function
 */
#define PTR_RETURN_OVERLOAD(DECL, METHOD) \
    (ptrReturnHelper<DECL,METHOD>().getWrapper())

#endif // JSMATERIALX_HELPERS_H
