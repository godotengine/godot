#pragma once



#ifndef PP_CAT
#define _PP_CAT(a, b)    a##b
#define PP_CAT(a, b)     _PP_CAT(a, b)
#endif

#define MATH_FORCEINLINE        inline

#define MATH_NOINLINE           inline
#define MATH_INLINE             inline
#define MATH_EMPTYINLINE        inline


#if defined(_MSC_VER)

    #if !defined(_DEBUG)
    #define META_PEEPHOLE
    #endif

    #define explicit_typename           typename
    #define template_decl(name, type)   template<type> name
    #define template_spec(name, val)    template<> name<val>
    #define template_inst(name, val)    name<val>
    #define explicit_operator           explicit operator

    #define vec_attr            const
    #define rhs_attr
    #define lhs_attr
    #define scalar_attr     mutable

#else

    #if defined(__OPTIMIZE__)
    #define META_PEEPHOLE
    #endif

    #define explicit_typename           typename
    #define template_decl(name, type)   template<type, int foo> name
    #define template_spec(name, val)    template<int foo> name<val, foo>
    #define template_inst(name, val)    name<val, 0>

    #if HAS_CLANG_FEATURE(cxx_explicit_conversions)
        #define explicit_operator           explicit operator
    #else
        #define explicit_operator           operator
    #endif

    #if (defined  __has_extension)
        #define SUPPORT_VECTOR_EXTENSION __has_extension(attribute_ext_vector_type)
    #else
        #define SUPPORT_VECTOR_EXTENSION 0
    #endif

    #if defined(MATH_HAS_NATIVE_SIMD) && MATH_HAS_NATIVE_SIMD == 0
        #undef MATH_HAS_NATIVE_SIMD
    #elif defined(__clang__) && SUPPORT_VECTOR_EXTENSION && (defined(__SSE__) || defined(__ARM_NEON)) && !defined(EMSCRIPTEN)
        #define MATH_HAS_NATIVE_SIMD
    #endif

    #define vec_attr            const
    #define rhs_attr
    #define lhs_attr
    #define scalar_attr     mutable

#endif
