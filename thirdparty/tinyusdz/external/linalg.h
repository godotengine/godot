// linalg.h - 2.2-beta - Single-header public domain linear algebra library
//
// The intent of this library is to provide the bulk of the functionality
// you need to write programs that frequently use small, fixed-size vectors
// and matrices, in domains such as computational geometry or computer
// graphics. It strives for terse, readable source code.
//
// The original author of this software is Sterling Orsten, and its permanent
// home is <http://github.com/sgorsten/linalg/>. If you find this software
// useful, an acknowledgement in your source text and/or product documentation
// is appreciated, but not required.
//
// The author acknowledges significant insights and contributions by:
//     Stan Melax <http://github.com/melax/>
//     Dimitri Diakopoulos <http://github.com/ddiakopoulos/>
//
// Some features are deprecated. Define LINALG_FORWARD_COMPATIBLE to remove them.



// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.
//
// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// For more information, please refer to <http://unlicense.org/>



#pragma once
#ifndef LINALG_H
#define LINALG_H

#include <cmath>        // For various unary math functions, such as std::sqrt
#include <cstdlib>      // To resolve std::abs ambiguity on clang
#include <cstdint>      // For implementing namespace linalg::aliases
#include <array>        // For std::array
#include <iosfwd>       // For forward definitions of std::ostream
#include <type_traits>  // For std::enable_if, std::is_same, std::declval
#include <functional>   // For std::hash declaration

// In Visual Studio 2015, `constexpr` applied to a member function implies `const`, which causes ambiguous overload resolution
#if _MSC_VER <= 1900
#define LINALG_CONSTEXPR14
#else
#define LINALG_CONSTEXPR14 constexpr
#endif

namespace linalg
{
    // Small, fixed-length vector type, consisting of exactly M elements of type T, and presumed to be a column-vector unless otherwise noted.
    template<class T, int M> struct vec;

    // Small, fixed-size matrix type, consisting of exactly M rows and N columns of type T, stored in column-major order.
    template<class T, int M, int N> struct mat;

    // Specialize converter<T,U> with a function application operator that converts type U to type T to enable implicit conversions
    template<class T, class U> struct converter {};
    namespace detail
    {
        template<class T, class U> using conv_t = typename std::enable_if<!std::is_same<T,U>::value, decltype(converter<T,U>{}(std::declval<U>()))>::type;

        // Trait for retrieving scalar type of any linear algebra object
        template<class A> struct scalar_type {};
        template<class T, int M       > struct scalar_type<vec<T,M  >> { using type = T; };
        template<class T, int M, int N> struct scalar_type<mat<T,M,N>> { using type = T; };

        // Type returned by the compare(...) function which supports all six comparison operators against 0
        template<class T> struct ord { T a,b; };
        template<class T> constexpr bool operator == (const ord<T> & o, std::nullptr_t) { return o.a == o.b; }
        template<class T> constexpr bool operator != (const ord<T> & o, std::nullptr_t) { return !(o.a == o.b); }
        template<class T> constexpr bool operator < (const ord<T> & o, std::nullptr_t) { return o.a < o.b; }
        template<class T> constexpr bool operator > (const ord<T> & o, std::nullptr_t) { return o.b < o.a; }
        template<class T> constexpr bool operator <= (const ord<T> & o, std::nullptr_t) { return !(o.b < o.a); }
        template<class T> constexpr bool operator >= (const ord<T> & o, std::nullptr_t) { return !(o.a < o.b); }

        // Patterns which can be used with the compare(...) function
        template<class A, class B> struct any_compare {};
        template<class T> struct any_compare<vec<T,1>,vec<T,1>> { using type=ord<T>; constexpr ord<T> operator() (const vec<T,1> & a, const vec<T,1> & b) const { return ord<T>{a.x,b.x}; } };
        template<class T> struct any_compare<vec<T,2>,vec<T,2>> { using type=ord<T>; constexpr ord<T> operator() (const vec<T,2> & a, const vec<T,2> & b) const { return !(a.x==b.x) ? ord<T>{a.x,b.x} : ord<T>{a.y,b.y}; } };
        template<class T> struct any_compare<vec<T,3>,vec<T,3>> { using type=ord<T>; constexpr ord<T> operator() (const vec<T,3> & a, const vec<T,3> & b) const { return !(a.x==b.x) ? ord<T>{a.x,b.x} : !(a.y==b.y) ? ord<T>{a.y,b.y} : ord<T>{a.z,b.z}; } };
        template<class T> struct any_compare<vec<T,4>,vec<T,4>> { using type=ord<T>; constexpr ord<T> operator() (const vec<T,4> & a, const vec<T,4> & b) const { return !(a.x==b.x) ? ord<T>{a.x,b.x} : !(a.y==b.y) ? ord<T>{a.y,b.y} : !(a.z==b.z) ? ord<T>{a.z,b.z} : ord<T>{a.w,b.w}; } };
        template<class T, int M> struct any_compare<mat<T,M,1>,mat<T,M,1>> { using type=ord<T>; constexpr ord<T> operator() (const mat<T,M,1> & a, const mat<T,M,1> & b) const { return compare(a.x,b.x); } };
        template<class T, int M> struct any_compare<mat<T,M,2>,mat<T,M,2>> { using type=ord<T>; constexpr ord<T> operator() (const mat<T,M,2> & a, const mat<T,M,2> & b) const { return a.x!=b.x ? compare(a.x,b.x) : compare(a.y,b.y); } };
        template<class T, int M> struct any_compare<mat<T,M,3>,mat<T,M,3>> { using type=ord<T>; constexpr ord<T> operator() (const mat<T,M,3> & a, const mat<T,M,3> & b) const { return a.x!=b.x ? compare(a.x,b.x) : a.y!=b.y ? compare(a.y,b.y) : compare(a.z,b.z); } };
        template<class T, int M> struct any_compare<mat<T,M,4>,mat<T,M,4>> { using type=ord<T>; constexpr ord<T> operator() (const mat<T,M,4> & a, const mat<T,M,4> & b) const { return a.x!=b.x ? compare(a.x,b.x) : a.y!=b.y ? compare(a.y,b.y) : a.z!=b.z ? compare(a.z,b.z) : compare(a.w,b.w); } }; 

        // Helper for compile-time index-based access to members of vector and matrix types
        template<int I> struct getter;
        template<> struct getter<0> { template<class A> constexpr auto operator() (A & a) const -> decltype(a.x) { return a.x; } };
        template<> struct getter<1> { template<class A> constexpr auto operator() (A & a) const -> decltype(a.y) { return a.y; } };
        template<> struct getter<2> { template<class A> constexpr auto operator() (A & a) const -> decltype(a.z) { return a.z; } };
        template<> struct getter<3> { template<class A> constexpr auto operator() (A & a) const -> decltype(a.w) { return a.w; } };

        // Stand-in for std::integer_sequence/std::make_integer_sequence
        template<int... I> struct seq {};
        template<int A, int N> struct make_seq_impl;
        template<int A> struct make_seq_impl<A,0> { using type=seq<>; };
        template<int A> struct make_seq_impl<A,1> { using type=seq<A+0>; };
        template<int A> struct make_seq_impl<A,2> { using type=seq<A+0,A+1>; };
        template<int A> struct make_seq_impl<A,3> { using type=seq<A+0,A+1,A+2>; };
        template<int A> struct make_seq_impl<A,4> { using type=seq<A+0,A+1,A+2,A+3>; };
        template<int A, int B> using make_seq = typename make_seq_impl<A,B-A>::type;
        template<class T, int M, int... I> vec<T,sizeof...(I)> constexpr swizzle(const vec<T,M> & v, seq<I...> i) { return {getter<I>{}(v)...}; }
        template<class T, int M, int N, int... I, int... J> mat<T,sizeof...(I),sizeof...(J)> constexpr swizzle(const mat<T,M,N> & m, seq<I...> i, seq<J...> j) { return {swizzle(getter<J>{}(m),i)...}; }

        // SFINAE helpers to determine result of function application
        template<class F, class... T> using ret_t = decltype(std::declval<F>()(std::declval<T>()...));

        // SFINAE helper which is defined if all provided types are scalars
        struct empty {};
        template<class... T> struct scalars;
        template<> struct scalars<> { using type=void; };
        template<class T, class... U> struct scalars<T,U...> : std::conditional<std::is_arithmetic<T>::value, scalars<U...>, empty>::type {};
        template<class... T> using scalars_t = typename scalars<T...>::type;

        // Helpers which indicate how apply(F, ...) should be called for various arguments
        template<class F, class Void, class... T> struct apply {}; // Patterns which contain only vectors or scalars
        template<class F, int M, class A                  > struct apply<F, scalars_t<   >, vec<A,M>                    > { using type=vec<ret_t<F,A    >,M>; enum {size=M, mm=0}; template<int... I> static constexpr type impl(seq<I...>, F f, const vec<A,M> & a                                        ) { return {f(getter<I>{}(a)                                )...}; } };
        template<class F, int M, class A, class B         > struct apply<F, scalars_t<   >, vec<A,M>, vec<B,M>          > { using type=vec<ret_t<F,A,B  >,M>; enum {size=M, mm=0}; template<int... I> static constexpr type impl(seq<I...>, F f, const vec<A,M> & a, const vec<B,M> & b                    ) { return {f(getter<I>{}(a), getter<I>{}(b)                )...}; } };
        template<class F, int M, class A, class B         > struct apply<F, scalars_t<B  >, vec<A,M>, B                 > { using type=vec<ret_t<F,A,B  >,M>; enum {size=M, mm=0}; template<int... I> static constexpr type impl(seq<I...>, F f, const vec<A,M> & a, B                b                    ) { return {f(getter<I>{}(a), b                             )...}; } };
        template<class F, int M, class A, class B         > struct apply<F, scalars_t<A  >, A,        vec<B,M>          > { using type=vec<ret_t<F,A,B  >,M>; enum {size=M, mm=0}; template<int... I> static constexpr type impl(seq<I...>, F f, A                a, const vec<B,M> & b                    ) { return {f(a,              getter<I>{}(b)                )...}; } };
        template<class F, int M, class A, class B, class C> struct apply<F, scalars_t<   >, vec<A,M>, vec<B,M>, vec<C,M>> { using type=vec<ret_t<F,A,B,C>,M>; enum {size=M, mm=0}; template<int... I> static constexpr type impl(seq<I...>, F f, const vec<A,M> & a, const vec<B,M> & b, const vec<C,M> & c) { return {f(getter<I>{}(a), getter<I>{}(b), getter<I>{}(c))...}; } };
        template<class F, int M, class A, class B, class C> struct apply<F, scalars_t<C  >, vec<A,M>, vec<B,M>, C       > { using type=vec<ret_t<F,A,B,C>,M>; enum {size=M, mm=0}; template<int... I> static constexpr type impl(seq<I...>, F f, const vec<A,M> & a, const vec<B,M> & b, C                c) { return {f(getter<I>{}(a), getter<I>{}(b), c             )...}; } };
        template<class F, int M, class A, class B, class C> struct apply<F, scalars_t<B  >, vec<A,M>, B,        vec<C,M>> { using type=vec<ret_t<F,A,B,C>,M>; enum {size=M, mm=0}; template<int... I> static constexpr type impl(seq<I...>, F f, const vec<A,M> & a, B                b, const vec<C,M> & c) { return {f(getter<I>{}(a), b,              getter<I>{}(c))...}; } };
        template<class F, int M, class A, class B, class C> struct apply<F, scalars_t<B,C>, vec<A,M>, B,        C       > { using type=vec<ret_t<F,A,B,C>,M>; enum {size=M, mm=0}; template<int... I> static constexpr type impl(seq<I...>, F f, const vec<A,M> & a, B                b, C                c) { return {f(getter<I>{}(a), b,              c             )...}; } };
        template<class F, int M, class A, class B, class C> struct apply<F, scalars_t<A  >, A,        vec<B,M>, vec<C,M>> { using type=vec<ret_t<F,A,B,C>,M>; enum {size=M, mm=0}; template<int... I> static constexpr type impl(seq<I...>, F f, A                a, const vec<B,M> & b, const vec<C,M> & c) { return {f(a,              getter<I>{}(b), getter<I>{}(c))...}; } };
        template<class F, int M, class A, class B, class C> struct apply<F, scalars_t<A,C>, A,        vec<B,M>, C       > { using type=vec<ret_t<F,A,B,C>,M>; enum {size=M, mm=0}; template<int... I> static constexpr type impl(seq<I...>, F f, A                a, const vec<B,M> & b, C                c) { return {f(a,              getter<I>{}(b), c             )...}; } };
        template<class F, int M, class A, class B, class C> struct apply<F, scalars_t<A,B>, A,        B,        vec<C,M>> { using type=vec<ret_t<F,A,B,C>,M>; enum {size=M, mm=0}; template<int... I> static constexpr type impl(seq<I...>, F f, A                a, B                b, const vec<C,M> & c) { return {f(a,              b,              getter<I>{}(c))...}; } };
        template<class F, int M, int N, class A         > struct apply<F, scalars_t< >, mat<A,M,N>            > { using type=mat<ret_t<F,A  >,M,N>; enum {size=N, mm=0}; template<int... J> static constexpr type impl(seq<J...>, F f, const mat<A,M,N> & a                      ) { return {apply<F, void, vec<A,M>          >::impl(make_seq<0,M>{}, f, getter<J>{}(a)                )...}; } };
        template<class F, int M, int N, class A, class B> struct apply<F, scalars_t< >, mat<A,M,N>, mat<B,M,N>> { using type=mat<ret_t<F,A,B>,M,N>; enum {size=N, mm=1}; template<int... J> static constexpr type impl(seq<J...>, F f, const mat<A,M,N> & a, const mat<B,M,N> & b) { return {apply<F, void, vec<A,M>, vec<B,M>>::impl(make_seq<0,M>{}, f, getter<J>{}(a), getter<J>{}(b))...}; } };
        template<class F, int M, int N, class A, class B> struct apply<F, scalars_t<B>, mat<A,M,N>, B         > { using type=mat<ret_t<F,A,B>,M,N>; enum {size=N, mm=0}; template<int... J> static constexpr type impl(seq<J...>, F f, const mat<A,M,N> & a, B                  b) { return {apply<F, void, vec<A,M>, B       >::impl(make_seq<0,M>{}, f, getter<J>{}(a), b             )...}; } };
        template<class F, int M, int N, class A, class B> struct apply<F, scalars_t<A>, A,          mat<B,M,N>> { using type=mat<ret_t<F,A,B>,M,N>; enum {size=N, mm=0}; template<int... J> static constexpr type impl(seq<J...>, F f, A                  a, const mat<B,M,N> & b) { return {apply<F, void, A,        vec<B,M>>::impl(make_seq<0,M>{}, f, a,              getter<J>{}(b))...}; } };
        template<class F, class... A> struct apply<F, scalars_t<A...>, A...> { using type = ret_t<F,A...>; enum {size=0, mm=0}; static constexpr type impl(seq<>, F f, A... a) { return f(a...); } };

        // Function objects for selecting between alternatives
        struct min    { template<class A, class B> constexpr auto operator() (A a, B b) const -> typename std::remove_reference<decltype(a<b ? a : b)>::type { return a<b ? a : b; } };
        struct max    { template<class A, class B> constexpr auto operator() (A a, B b) const -> typename std::remove_reference<decltype(a<b ? b : a)>::type { return a<b ? b : a; } };
        struct clamp  { template<class A, class B, class C> constexpr auto operator() (A a, B b, C c) const -> typename std::remove_reference<decltype(a<b ? b : a<c ? a : c)>::type { return a<b ? b : a<c ? a : c; } };
        struct select { template<class A, class B, class C> constexpr auto operator() (A a, B b, C c) const -> typename std::remove_reference<decltype(a ? b : c)>::type             { return a ? b : c; } };
        struct lerp   { template<class A, class B, class C> constexpr auto operator() (A a, B b, C c) const -> decltype(a*(1-c) + b*c)                                               { return a*(1-c) + b*c; } };

        // Function objects for applying operators
        struct op_pos { template<class A> constexpr auto operator() (A a) const -> decltype(+a) { return +a; } };
        struct op_neg { template<class A> constexpr auto operator() (A a) const -> decltype(-a) { return -a; } };
        struct op_not { template<class A> constexpr auto operator() (A a) const -> decltype(!a) { return !a; } };
        struct op_cmp { template<class A> constexpr auto operator() (A a) const -> decltype(~(a)) { return ~a; } };
        struct op_mul { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a * b)  { return a * b; } };
        struct op_div { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a / b)  { return a / b; } };
        struct op_mod { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a % b)  { return a % b; } };
        struct op_add { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a + b)  { return a + b; } };
        struct op_sub { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a - b)  { return a - b; } };
        struct op_lsh { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a << b) { return a << b; } };
        struct op_rsh { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a >> b) { return a >> b; } };
        struct op_lt  { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a < b)  { return a < b; } };
        struct op_gt  { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a > b)  { return a > b; } };
        struct op_le  { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a <= b) { return a <= b; } };
        struct op_ge  { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a >= b) { return a >= b; } };
        struct op_eq  { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a == b) { return a == b; } };
        struct op_ne  { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a != b) { return a != b; } };
        struct op_int { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a & b)  { return a & b; } };
        struct op_xor { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a ^ b)  { return a ^ b; } };
        struct op_un  { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a | b)  { return a | b; } };
        struct op_and { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a && b) { return a && b; } };
        struct op_or  { template<class A, class B> constexpr auto operator() (A a, B b) const -> decltype(a || b) { return a || b; } };

        // Function objects for applying standard library math functions
        struct std_abs      { template<class A> auto operator() (A a) const -> decltype(std::abs  (a)) { return std::abs  (a); } };
        struct std_floor    { template<class A> auto operator() (A a) const -> decltype(std::floor(a)) { return std::floor(a); } };
        struct std_ceil     { template<class A> auto operator() (A a) const -> decltype(std::ceil (a)) { return std::ceil (a); } };
        struct std_exp      { template<class A> auto operator() (A a) const -> decltype(std::exp  (a)) { return std::exp  (a); } };
        struct std_log      { template<class A> auto operator() (A a) const -> decltype(std::log  (a)) { return std::log  (a); } };
        struct std_log10    { template<class A> auto operator() (A a) const -> decltype(std::log10(a)) { return std::log10(a); } };
        struct std_sqrt     { template<class A> auto operator() (A a) const -> decltype(std::sqrt (a)) { return std::sqrt (a); } };
        struct std_sin      { template<class A> auto operator() (A a) const -> decltype(std::sin  (a)) { return std::sin  (a); } };
        struct std_cos      { template<class A> auto operator() (A a) const -> decltype(std::cos  (a)) { return std::cos  (a); } };
        struct std_tan      { template<class A> auto operator() (A a) const -> decltype(std::tan  (a)) { return std::tan  (a); } };
        struct std_asin     { template<class A> auto operator() (A a) const -> decltype(std::asin (a)) { return std::asin (a); } };
        struct std_acos     { template<class A> auto operator() (A a) const -> decltype(std::acos (a)) { return std::acos (a); } };
        struct std_atan     { template<class A> auto operator() (A a) const -> decltype(std::atan (a)) { return std::atan (a); } };
        struct std_sinh     { template<class A> auto operator() (A a) const -> decltype(std::sinh (a)) { return std::sinh (a); } };
        struct std_cosh     { template<class A> auto operator() (A a) const -> decltype(std::cosh (a)) { return std::cosh (a); } };
        struct std_tanh     { template<class A> auto operator() (A a) const -> decltype(std::tanh (a)) { return std::tanh (a); } };
        struct std_round    { template<class A> auto operator() (A a) const -> decltype(std::round(a)) { return std::round(a); } };
        struct std_fmod     { template<class A, class B> auto operator() (A a, B b) const -> decltype(std::fmod    (a, b)) { return std::fmod    (a, b); } };
        struct std_pow      { template<class A, class B> auto operator() (A a, B b) const -> decltype(std::pow     (a, b)) { return std::pow     (a, b); } };
        struct std_atan2    { template<class A, class B> auto operator() (A a, B b) const -> decltype(std::atan2   (a, b)) { return std::atan2   (a, b); } };
        struct std_copysign { template<class A, class B> auto operator() (A a, B b) const -> decltype(std::copysign(a, b)) { return std::copysign(a, b); } };
    }

    // Small, fixed-length vector type, consisting of exactly M elements of type T, and presumed to be a column-vector unless otherwise noted
    template<class T> struct vec<T,1>
    {
        T                           x;
        constexpr                   vec()                               : x() {}
        constexpr                   vec(const T & x_)                   : x(x_) {}
        // NOTE: vec<T,1> does NOT have a constructor from pointer, this can conflict with initializing its single element from zero
        template<class U>
        constexpr explicit          vec(const vec<U,1> & v)             : vec(static_cast<T>(v.x)) {}
        constexpr const T &         operator[] (int i) const            { return x; }
        LINALG_CONSTEXPR14 T &      operator[] (int i)                  { return x; }

        template<class U, class=detail::conv_t<vec,U>> constexpr vec(const U & u) : vec(converter<vec,U>{}(u)) {}
        template<class U, class=detail::conv_t<U,vec>> constexpr operator U () const { return converter<U,vec>{}(*this); }
    };
    template<class T> struct vec<T,2>
    {
        T                           x,y;
        constexpr                   vec()                               : x(), y() {}
        constexpr                   vec(const T & x_, const T & y_)     : x(x_), y(y_) {}
        constexpr explicit          vec(const T & s)                    : vec(s, s) {}
        constexpr explicit          vec(const T * p)                    : vec(p[0], p[1]) {}
        template<class U>
        constexpr explicit          vec(const vec<U,2> & v)             : vec(static_cast<T>(v.x), static_cast<T>(v.y)) {}
        constexpr const T &         operator[] (int i) const            { return i==0?x:y; }
        LINALG_CONSTEXPR14 T &      operator[] (int i)                  { return i==0?x:y; }

        template<class U, class=detail::conv_t<vec,U>> constexpr vec(const U & u) : vec(converter<vec,U>{}(u)) {}
        template<class U, class=detail::conv_t<U,vec>> constexpr operator U () const { return converter<U,vec>{}(*this); }
    };
    template<class T> struct vec<T,3>
    {
        T                           x,y,z;
        constexpr                   vec()                               : x(), y(), z() {}
        constexpr                   vec(const T & x_, const T & y_, 
                                        const T & z_)                   : x(x_), y(y_), z(z_) {}
        constexpr                   vec(const vec<T,2> & xy,
                                        const T & z_)                   : vec(xy.x, xy.y, z_) {}
        constexpr explicit          vec(const T & s)                    : vec(s, s, s) {}
        constexpr explicit          vec(const T * p)                    : vec(p[0], p[1], p[2]) {}
        template<class U>
        constexpr explicit          vec(const vec<U,3> & v)             : vec(static_cast<T>(v.x), static_cast<T>(v.y), static_cast<T>(v.z)) {}
        constexpr const T &         operator[] (int i) const            { return i==0?x:i==1?y:z; }
        LINALG_CONSTEXPR14 T &      operator[] (int i)                  { return i==0?x:i==1?y:z; }
        constexpr const vec<T,2> &  xy() const                          { return *reinterpret_cast<const vec<T,2> *>(this); }
        vec<T,2> &                  xy()                                { return *reinterpret_cast<vec<T,2> *>(this); }

        template<class U, class=detail::conv_t<vec,U>> constexpr vec(const U & u) : vec(converter<vec,U>{}(u)) {}
        template<class U, class=detail::conv_t<U,vec>> constexpr operator U () const { return converter<U,vec>{}(*this); }
    };
    template<class T> struct vec<T,4>
    {
        T                           x,y,z,w;
        constexpr                   vec()                               : x(), y(), z(), w() {}
        constexpr                   vec(const T & x_, const T & y_,
                                        const T & z_, const T & w_)     : x(x_), y(y_), z(z_), w(w_) {}
        constexpr                   vec(const vec<T,2> & xy, 
                                        const T & z_, const T & w_)     : vec(xy.x, xy.y, z_, w_) {}
        constexpr                   vec(const vec<T,3> & xyz,
                                        const T & w_)                   : vec(xyz.x, xyz.y, xyz.z, w_) {}
        constexpr explicit          vec(const T & s)                    : vec(s, s, s, s) {}
        constexpr explicit          vec(const T * p)                    : vec(p[0], p[1], p[2], p[3]) {}
        template<class U> 
        constexpr explicit          vec(const vec<U,4> & v)             : vec(static_cast<T>(v.x), static_cast<T>(v.y), static_cast<T>(v.z), static_cast<T>(v.w)) {}
        constexpr const T &         operator[] (int i) const            { return i==0?x:i==1?y:i==2?z:w; }
        LINALG_CONSTEXPR14 T &      operator[] (int i)                  { return i==0?x:i==1?y:i==2?z:w; }
        constexpr const vec<T,2> &  xy() const                          { return *reinterpret_cast<const vec<T,2> *>(this); }
        constexpr const vec<T,3> &  xyz() const                         { return *reinterpret_cast<const vec<T,3> *>(this); }
        vec<T,2> &                  xy()                                { return *reinterpret_cast<vec<T,2> *>(this); }                
        vec<T,3> &                  xyz()                               { return *reinterpret_cast<vec<T,3> *>(this); }

        template<class U, class=detail::conv_t<vec,U>> constexpr vec(const U & u) : vec(converter<vec,U>{}(u)) {}
        template<class U, class=detail::conv_t<U,vec>> constexpr operator U () const { return converter<U,vec>{}(*this); }
    };

    // Small, fixed-size matrix type, consisting of exactly M rows and N columns of type T, stored in column-major order.
    template<class T, int M> struct mat<T,M,1>
    {
        typedef vec<T,M>            V;
        V                           x;
        constexpr                   mat()                               : x() {}
        constexpr                   mat(const V & x_)                   : x(x_) {}
        constexpr explicit          mat(const T & s)                    : x(s) {}
        constexpr explicit          mat(const T * p)                    : x(p+M*0) {}
        template<class U> 
        constexpr explicit          mat(const mat<U,M,1> & m)           : mat(V(m.x)) {}
        constexpr vec<T,1>          row(int i) const                    { return {x[i]}; }
        constexpr const V &         operator[] (int j) const            { return x; }
        LINALG_CONSTEXPR14 V &      operator[] (int j)                  { return x; }

        template<class U, class=detail::conv_t<mat,U>> constexpr mat(const U & u) : mat(converter<mat,U>{}(u)) {}
        template<class U, class=detail::conv_t<U,mat>> constexpr operator U () const { return converter<U,mat>{}(*this); }
    };
    template<class T, int M> struct mat<T,M,2>
    {
        typedef vec<T,M>            V;
        V                           x,y;
        constexpr                   mat()                               : x(), y() {}
        constexpr                   mat(const V & x_, const V & y_)     : x(x_), y(y_) {}
        constexpr explicit          mat(const T & s)                    : x(s), y(s) {}
        constexpr explicit          mat(const T * p)                    : x(p+M*0), y(p+M*1) {}
        template<class U> 
        constexpr explicit          mat(const mat<U,M,2> & m)           : mat(V(m.x), V(m.y)) {}
        constexpr vec<T,2>          row(int i) const                    { return {x[i], y[i]}; }
        constexpr const V &         operator[] (int j) const            { return j==0?x:y; }
        LINALG_CONSTEXPR14 V &      operator[] (int j)                  { return j==0?x:y; }

        template<class U, class=detail::conv_t<mat,U>> constexpr mat(const U & u) : mat(converter<mat,U>{}(u)) {}
        template<class U, class=detail::conv_t<U,mat>> constexpr operator U () const { return converter<U,mat>{}(*this); }
    };
    template<class T, int M> struct mat<T,M,3>
    {
        typedef vec<T,M>            V;
        V                           x,y,z;
        constexpr                   mat()                               : x(), y(), z() {}
        constexpr                   mat(const V & x_, const V & y_, 
                                        const V & z_)                   : x(x_), y(y_), z(z_) {}
        constexpr explicit          mat(const T & s)                    : x(s), y(s), z(s) {}
        constexpr explicit          mat(const T * p)                    : x(p+M*0), y(p+M*1), z(p+M*2) {}
        template<class U> 
        constexpr explicit          mat(const mat<U,M,3> & m)           : mat(V(m.x), V(m.y), V(m.z)) {}
        constexpr vec<T,3>          row(int i) const                    { return {x[i], y[i], z[i]}; }
        constexpr const V &         operator[] (int j) const            { return j==0?x:j==1?y:z; }
        LINALG_CONSTEXPR14 V &      operator[] (int j)                  { return j==0?x:j==1?y:z; }

        template<class U, class=detail::conv_t<mat,U>> constexpr mat(const U & u) : mat(converter<mat,U>{}(u)) {}
        template<class U, class=detail::conv_t<U,mat>> constexpr operator U () const { return converter<U,mat>{}(*this); }
    };
    template<class T, int M> struct mat<T,M,4>
    {
        typedef vec<T,M>            V;
        V                           x,y,z,w;
        constexpr                   mat()                               : x(), y(), z(), w() {}
        constexpr                   mat(const V & x_, const V & y_,
                                        const V & z_, const V & w_)     : x(x_), y(y_), z(z_), w(w_) {}
        constexpr explicit          mat(const T & s)                    : x(s), y(s), z(s), w(s) {}
        constexpr explicit          mat(const T * p)                    : x(p+M*0), y(p+M*1), z(p+M*2), w(p+M*3) {}
        template<class U> 
        constexpr explicit          mat(const mat<U,M,4> & m)           : mat(V(m.x), V(m.y), V(m.z), V(m.w)) {}
        constexpr vec<T,4>          row(int i) const                    { return {x[i], y[i], z[i], w[i]}; }
        constexpr const V &         operator[] (int j) const            { return j==0?x:j==1?y:j==2?z:w; }
        LINALG_CONSTEXPR14 V &      operator[] (int j)                  { return j==0?x:j==1?y:j==2?z:w; }

        template<class U, class=detail::conv_t<mat,U>> constexpr mat(const U & u) : mat(converter<mat,U>{}(u)) {}
        template<class U, class=detail::conv_t<U,mat>> constexpr operator U () const { return converter<U,mat>{}(*this); }
    };

    // Define a type which will convert to the multiplicative identity of any square matrix
    struct identity_t { constexpr explicit identity_t(int) {} };
    template<class T> struct converter<mat<T,1,1>, identity_t> { constexpr mat<T,1,1> operator() (identity_t) const { return {vec<T,1>{1}}; } };
    template<class T> struct converter<mat<T,2,2>, identity_t> { constexpr mat<T,2,2> operator() (identity_t) const { return {{1,0},{0,1}}; } };
    template<class T> struct converter<mat<T,3,3>, identity_t> { constexpr mat<T,3,3> operator() (identity_t) const { return {{1,0,0},{0,1,0},{0,0,1}}; } };
    template<class T> struct converter<mat<T,4,4>, identity_t> { constexpr mat<T,4,4> operator() (identity_t) const { return {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}; } };
    constexpr identity_t identity {1};

    // Produce a scalar by applying f(A,B) -> A to adjacent pairs of elements from a vec/mat in left-to-right/column-major order (matching the associativity of arithmetic and logical operators)
    template<class F, class A, class B> constexpr A fold(F f, A a, const vec<B,1> & b) { return f(a, b.x); }
    template<class F, class A, class B> constexpr A fold(F f, A a, const vec<B,2> & b) { return f(f(a, b.x), b.y); }
    template<class F, class A, class B> constexpr A fold(F f, A a, const vec<B,3> & b) { return f(f(f(a, b.x), b.y), b.z); }
    template<class F, class A, class B> constexpr A fold(F f, A a, const vec<B,4> & b) { return f(f(f(f(a, b.x), b.y), b.z), b.w); }
    template<class F, class A, class B, int M> constexpr A fold(F f, A a, const mat<B,M,1> & b) { return fold(f, a, b.x); }
    template<class F, class A, class B, int M> constexpr A fold(F f, A a, const mat<B,M,2> & b) { return fold(f, fold(f, a, b.x), b.y); }
    template<class F, class A, class B, int M> constexpr A fold(F f, A a, const mat<B,M,3> & b) { return fold(f, fold(f, fold(f, a, b.x), b.y), b.z); }
    template<class F, class A, class B, int M> constexpr A fold(F f, A a, const mat<B,M,4> & b) { return fold(f, fold(f, fold(f, fold(f, a, b.x), b.y), b.z), b.w); }

    // Type aliases for the result of calling apply(...) with various arguments, can be used with return type SFINAE to constrian overload sets
    template<class F, class... A> using apply_t = typename detail::apply<F,void,A...>::type;
    template<class F, class... A> using mm_apply_t = typename std::enable_if<detail::apply<F,void,A...>::mm, apply_t<F,A...>>::type;
    template<class F, class... A> using no_mm_apply_t = typename std::enable_if<!detail::apply<F,void,A...>::mm, apply_t<F,A...>>::type;
    template<class A> using scalar_t = typename detail::scalar_type<A>::type; // Underlying scalar type when performing elementwise operations

    // apply(f,...) applies the provided function in an elementwise fashion to its arguments, producing an object of the same dimensions
    template<class F, class... A> constexpr apply_t<F,A...> apply(F func, const A & ... args) { return detail::apply<F,void,A...>::impl(detail::make_seq<0,detail::apply<F,void,A...>::size>{}, func, args...); }

    // map(a,f) is equivalent to apply(f,a)
    template<class A, class F> constexpr apply_t<F,A> map(const A & a, F func) { return apply(func, a); }

    // zip(a,b,f) is equivalent to apply(f,a,b)
    template<class A, class B, class F> constexpr apply_t<F,A,B> zip(const A & a, const B & b, F func) { return apply(func, a, b); }

    // Relational operators are defined to compare the elements of two vectors or matrices lexicographically, in column-major order
    template<class A, class B> constexpr typename detail::any_compare<A,B>::type compare(const A & a, const B & b) { return detail::any_compare<A,B>()(a,b); }
    template<class A, class B> constexpr auto operator == (const A & a, const B & b) -> decltype(compare(a,b) == 0) { return compare(a,b) == 0; }
    template<class A, class B> constexpr auto operator != (const A & a, const B & b) -> decltype(compare(a,b) != 0) { return compare(a,b) != 0; }
    template<class A, class B> constexpr auto operator <  (const A & a, const B & b) -> decltype(compare(a,b) <  0) { return compare(a,b) <  0; }
    template<class A, class B> constexpr auto operator >  (const A & a, const B & b) -> decltype(compare(a,b) >  0) { return compare(a,b) >  0; }
    template<class A, class B> constexpr auto operator <= (const A & a, const B & b) -> decltype(compare(a,b) <= 0) { return compare(a,b) <= 0; }
    template<class A, class B> constexpr auto operator >= (const A & a, const B & b) -> decltype(compare(a,b) >= 0) { return compare(a,b) >= 0; }

    // Functions for coalescing scalar values
    template<class A> constexpr bool any (const A & a) { return fold(detail::op_or{}, false, a); }
    template<class A> constexpr bool all (const A & a) { return fold(detail::op_and{}, true, a); }
    template<class A> constexpr scalar_t<A> sum    (const A & a) { return fold(detail::op_add{}, scalar_t<A>(0), a); }
    template<class A> constexpr scalar_t<A> product(const A & a) { return fold(detail::op_mul{}, scalar_t<A>(1), a); }
    template<class A> constexpr scalar_t<A> minelem(const A & a) { return fold(detail::min{}, a.x, a); }
    template<class A> constexpr scalar_t<A> maxelem(const A & a) { return fold(detail::max{}, a.x, a); }
    template<class T, int M> int argmin(const vec<T,M> & a) { int j=0; for(int i=1; i<M; ++i) if(a[i] < a[j]) j = i; return j; }
    template<class T, int M> int argmax(const vec<T,M> & a) { int j=0; for(int i=1; i<M; ++i) if(a[i] > a[j]) j = i; return j; }

    // Unary operators are defined component-wise for linalg types
    template<class A> constexpr apply_t<detail::op_pos, A> operator + (const A & a) { return apply(detail::op_pos{}, a); }
    template<class A> constexpr apply_t<detail::op_neg, A> operator - (const A & a) { return apply(detail::op_neg{}, a); }
    template<class A> constexpr apply_t<detail::op_cmp, A> operator ~ (const A & a) { return apply(detail::op_cmp{}, a); }
    template<class A> constexpr apply_t<detail::op_not, A> operator ! (const A & a) { return apply(detail::op_not{}, a); }

    // Binary operators are defined component-wise for linalg types, EXCEPT for `operator *`
    template<class A, class B> constexpr apply_t<detail::op_add, A, B> operator +  (const A & a, const B & b) { return apply(detail::op_add{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_sub, A, B> operator -  (const A & a, const B & b) { return apply(detail::op_sub{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_mul, A, B> cmul        (const A & a, const B & b) { return apply(detail::op_mul{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_div, A, B> operator /  (const A & a, const B & b) { return apply(detail::op_div{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_mod, A, B> operator %  (const A & a, const B & b) { return apply(detail::op_mod{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_un,  A, B> operator |  (const A & a, const B & b) { return apply(detail::op_un{},  a, b); }
    template<class A, class B> constexpr apply_t<detail::op_xor, A, B> operator ^  (const A & a, const B & b) { return apply(detail::op_xor{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_int, A, B> operator &  (const A & a, const B & b) { return apply(detail::op_int{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_lsh, A, B> operator << (const A & a, const B & b) { return apply(detail::op_lsh{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_rsh, A, B> operator >> (const A & a, const B & b) { return apply(detail::op_rsh{}, a, b); }

    // Binary `operator *` was originally defined component-wise for all patterns, in a fashion consistent with the other operators. However,
    // this was one of the most frequent sources of confusion among new users of this library, with the binary `operator *` being used accidentally
    // by users who INTENDED the semantics of the algebraic matrix product, but RECEIVED the semantics of the Hadamard product. While there is
    // precedent within the HLSL, Fortran, R, APL, J, and Mathematica programming languages for `operator *` having the semantics of the Hadamard
    // product between matrices, it is counterintuitive to users of GLSL, Eigen, and many other languages and libraries that chose matrix product
    // semantics for `operator *`.
    //
    // For these reasons, binary `operator *` is now DEPRECATED between pairs of matrices. Users may call `cmul(...)` for component-wise multiplication,
    // or `mul(...)` for matrix multiplication. Binary `operator *` continues to be available for vector * vector, vector * scalar, matrix * scalar, etc.
    template<class A, class B> constexpr no_mm_apply_t<detail::op_mul, A, B> operator * (const A & a, const B & b) { return cmul(a,b); }
    #ifndef LINALG_FORWARD_COMPATIBLE
    template<class A, class B> [[deprecated("`operator *` between pairs of matrices is deprecated. See the source text for details.")]] constexpr mm_apply_t<detail::op_mul, A, B> operator * (const A & a, const B & b) { return cmul(a,b); }
    #endif

    // Binary assignment operators a $= b is always defined as though it were explicitly written a = a $ b
    template<class A, class B> constexpr auto operator +=  (A & a, const B & b) -> decltype(a = a + b) { return a = a + b; }
    template<class A, class B> constexpr auto operator -=  (A & a, const B & b) -> decltype(a = a - b) { return a = a - b; }
    template<class A, class B> constexpr auto operator *=  (A & a, const B & b) -> decltype(a = a * b) { return a = a * b; }
    template<class A, class B> constexpr auto operator /=  (A & a, const B & b) -> decltype(a = a / b) { return a = a / b; }
    template<class A, class B> constexpr auto operator %=  (A & a, const B & b) -> decltype(a = a % b) { return a = a % b; }
    template<class A, class B> constexpr auto operator |=  (A & a, const B & b) -> decltype(a = a | b) { return a = a | b; }
    template<class A, class B> constexpr auto operator ^=  (A & a, const B & b) -> decltype(a = a ^ b) { return a = a ^ b; }
    template<class A, class B> constexpr auto operator &=  (A & a, const B & b) -> decltype(a = a & b) { return a = a & b; }
    template<class A, class B> constexpr auto operator <<= (A & a, const B & b) -> decltype(a = a << b) { return a = a << b; }
    template<class A, class B> constexpr auto operator >>= (A & a, const B & b) -> decltype(a = a >> b) { return a = a >> b; }

    // Swizzles and subobjects
    template<int... I, class T, int M>                              constexpr vec<T,sizeof...(I)>   swizzle(const vec<T,M> & a)   { return {detail::getter<I>{}(a)...}; }
    template<int I0, int I1, class T, int M>                        constexpr vec<T,I1-I0>          subvec (const vec<T,M> & a)   { return detail::swizzle(a, detail::make_seq<I0,I1>{}); }
    template<int I0, int J0, int I1, int J1, class T, int M, int N> constexpr mat<T,I1-I0,J1-J0>    submat (const mat<T,M,N> & a) { return detail::swizzle(a, detail::make_seq<I0,I1>{}, detail::make_seq<J0,J1>{}); }

    // Component-wise standard library math functions
    template<class A> apply_t<detail::std_abs,   A> abs  (const A & a) { return apply(detail::std_abs{},   a); }
    template<class A> apply_t<detail::std_floor, A> floor(const A & a) { return apply(detail::std_floor{}, a); }
    template<class A> apply_t<detail::std_ceil,  A> ceil (const A & a) { return apply(detail::std_ceil{},  a); }
    template<class A> apply_t<detail::std_exp,   A> exp  (const A & a) { return apply(detail::std_exp{},   a); }
    template<class A> apply_t<detail::std_log,   A> log  (const A & a) { return apply(detail::std_log{},   a); }
    template<class A> apply_t<detail::std_log10, A> log10(const A & a) { return apply(detail::std_log10{}, a); }
    template<class A> apply_t<detail::std_sqrt,  A> sqrt (const A & a) { return apply(detail::std_sqrt{},  a); }
    template<class A> apply_t<detail::std_sin,   A> sin  (const A & a) { return apply(detail::std_sin{},   a); }
    template<class A> apply_t<detail::std_cos,   A> cos  (const A & a) { return apply(detail::std_cos{},   a); }
    template<class A> apply_t<detail::std_tan,   A> tan  (const A & a) { return apply(detail::std_tan{},   a); }
    template<class A> apply_t<detail::std_asin,  A> asin (const A & a) { return apply(detail::std_asin{},  a); }
    template<class A> apply_t<detail::std_acos,  A> acos (const A & a) { return apply(detail::std_acos{},  a); }
    template<class A> apply_t<detail::std_atan,  A> atan (const A & a) { return apply(detail::std_atan{},  a); }
    template<class A> apply_t<detail::std_sinh,  A> sinh (const A & a) { return apply(detail::std_sinh{},  a); }
    template<class A> apply_t<detail::std_cosh,  A> cosh (const A & a) { return apply(detail::std_cosh{},  a); }
    template<class A> apply_t<detail::std_tanh,  A> tanh (const A & a) { return apply(detail::std_tanh{},  a); }
    template<class A> apply_t<detail::std_round, A> round(const A & a) { return apply(detail::std_round{}, a); }

    template<class A, class B> apply_t<detail::std_fmod,     A, B> fmod    (const A & a, const B & b) { return apply(detail::std_fmod{},     a, b); }
    template<class A, class B> apply_t<detail::std_pow,      A, B> pow     (const A & a, const B & b) { return apply(detail::std_pow{},      a, b); }
    template<class A, class B> apply_t<detail::std_atan2,    A, B> atan2   (const A & a, const B & b) { return apply(detail::std_atan2{},    a, b); }
    template<class A, class B> apply_t<detail::std_copysign, A, B> copysign(const A & a, const B & b) { return apply(detail::std_copysign{}, a, b); }

    // Component-wise relational functions on vectors
    template<class A, class B> constexpr apply_t<detail::op_eq, A, B> equal  (const A & a, const B & b) { return apply(detail::op_eq{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_ne, A, B> nequal (const A & a, const B & b) { return apply(detail::op_ne{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_lt, A, B> less   (const A & a, const B & b) { return apply(detail::op_lt{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_gt, A, B> greater(const A & a, const B & b) { return apply(detail::op_gt{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_le, A, B> lequal (const A & a, const B & b) { return apply(detail::op_le{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::op_ge, A, B> gequal (const A & a, const B & b) { return apply(detail::op_ge{}, a, b); }

    // Component-wise selection functions on vectors
    template<class A, class B> constexpr apply_t<detail::min, A, B> min(const A & a, const B & b) { return apply(detail::min{}, a, b); }
    template<class A, class B> constexpr apply_t<detail::max, A, B> max(const A & a, const B & b) { return apply(detail::max{}, a, b); }
    template<class X, class L, class H> constexpr apply_t<detail::clamp,  X, L, H> clamp (const X & x, const L & l, const H & h) { return apply(detail::clamp{},  x, l, h); }
    template<class P, class A, class B> constexpr apply_t<detail::select, P, A, B> select(const P & p, const A & a, const B & b) { return apply(detail::select{}, p, a, b); }
    template<class A, class B, class T> constexpr apply_t<detail::lerp,   A, B, T> lerp  (const A & a, const B & b, const T & t) { return apply(detail::lerp{},   a, b, t); }

    // Support for vector algebra
    template<class T> constexpr T        cross    (const vec<T,2> & a, const vec<T,2> & b)      { return a.x*b.y-a.y*b.x; }
    template<class T> constexpr vec<T,2> cross    (T a, const vec<T,2> & b)                     { return {-a*b.y, a*b.x}; }
    template<class T> constexpr vec<T,2> cross    (const vec<T,2> & a, T b)                     { return {a.y*b, -a.x*b}; }
    template<class T> constexpr vec<T,3> cross    (const vec<T,3> & a, const vec<T,3> & b)      { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
    template<class T, int M> constexpr T dot      (const vec<T,M> & a, const vec<T,M> & b)      { return sum(a*b); }
    template<class T, int M> constexpr T length2  (const vec<T,M> & a)                          { return dot(a,a); }
    template<class T, int M> T           length   (const vec<T,M> & a)                          { return std::sqrt(length2(a)); }
    template<class T, int M> vec<T,M>    normalize(const vec<T,M> & a)                          { return a / length(a); }
    template<class T, int M> constexpr T distance2(const vec<T,M> & a, const vec<T,M> & b)      { return length2(b-a); }
    template<class T, int M> T           distance (const vec<T,M> & a, const vec<T,M> & b)      { return length(b-a); }
    template<class T, int M> T           uangle   (const vec<T,M> & a, const vec<T,M> & b)      { T d=dot(a,b); return d > 1 ? 0 : std::acos(d < -1 ? -1 : d); }
    template<class T, int M> T           angle    (const vec<T,M> & a, const vec<T,M> & b)      { return uangle(normalize(a), normalize(b)); }
    template<class T> vec<T,2>           rot      (T a, const vec<T,2> & v)                     { const T s = std::sin(a), c = std::cos(a); return {v.x*c - v.y*s, v.x*s + v.y*c}; }
    template<class T, int M> vec<T,M>    nlerp    (const vec<T,M> & a, const vec<T,M> & b, T t) { return normalize(lerp(a,b,t)); }
    template<class T, int M> vec<T,M>    slerp    (const vec<T,M> & a, const vec<T,M> & b, T t) { T th=uangle(a,b); return th == 0 ? a : a*(std::sin(th*(1-t))/std::sin(th)) + b*(std::sin(th*t)/std::sin(th)); }

    // Support for quaternion algebra using 4D vectors, representing xi + yj + zk + w
    template<class T> constexpr vec<T,4> qconj(const vec<T,4> & q)                     { return {-q.x,-q.y,-q.z,q.w}; }
    template<class T> vec<T,4>           qinv (const vec<T,4> & q)                     { return qconj(q)/length2(q); }
    template<class T> vec<T,4>           qexp (const vec<T,4> & q)                     { const auto v = q.xyz(); const auto vv = length(v); return std::exp(q.w) * vec<T,4>{v * (vv > 0 ? std::sin(vv)/vv : 0), std::cos(vv)}; }
    template<class T> vec<T,4>           qlog (const vec<T,4> & q)                     { const auto v = q.xyz(); const auto vv = length(v), qq = length(q); return {v * (vv > 0 ? std::acos(q.w/qq)/vv : 0), std::log(qq)}; }
    template<class T> vec<T,4>           qpow (const vec<T,4> & q, const T & p)        { const auto v = q.xyz(); const auto vv = length(v), qq = length(q), th = std::acos(q.w/qq); return std::pow(qq,p)*vec<T,4>{v * (vv > 0 ? std::sin(p*th)/vv : 0), std::cos(p*th)}; }
    template<class T> constexpr vec<T,4> qmul (const vec<T,4> & a, const vec<T,4> & b) { return {a.x*b.w+a.w*b.x+a.y*b.z-a.z*b.y, a.y*b.w+a.w*b.y+a.z*b.x-a.x*b.z, a.z*b.w+a.w*b.z+a.x*b.y-a.y*b.x, a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z}; }
    template<class T, class... R> constexpr vec<T,4> qmul(const vec<T,4> & a, R... r)  { return qmul(a, qmul(r...)); }

    // Support for 3D spatial rotations using quaternions, via qmul(qmul(q, v), qconj(q))
    template<class T> constexpr vec<T,3>   qxdir (const vec<T,4> & q)                          { return {q.w*q.w+q.x*q.x-q.y*q.y-q.z*q.z, (q.x*q.y+q.z*q.w)*2, (q.z*q.x-q.y*q.w)*2}; }
    template<class T> constexpr vec<T,3>   qydir (const vec<T,4> & q)                          { return {(q.x*q.y-q.z*q.w)*2, q.w*q.w-q.x*q.x+q.y*q.y-q.z*q.z, (q.y*q.z+q.x*q.w)*2}; }
    template<class T> constexpr vec<T,3>   qzdir (const vec<T,4> & q)                          { return {(q.z*q.x+q.y*q.w)*2, (q.y*q.z-q.x*q.w)*2, q.w*q.w-q.x*q.x-q.y*q.y+q.z*q.z}; }
    template<class T> constexpr mat<T,3,3> qmat  (const vec<T,4> & q)                          { return {qxdir(q), qydir(q), qzdir(q)}; }
    template<class T> constexpr vec<T,3>   qrot  (const vec<T,4> & q, const vec<T,3> & v)      { return qxdir(q)*v.x + qydir(q)*v.y + qzdir(q)*v.z; }
    template<class T> T                    qangle(const vec<T,4> & q)                          { return std::atan2(length(q.xyz()), q.w)*2; }
    template<class T> vec<T,3>             qaxis (const vec<T,4> & q)                          { return normalize(q.xyz()); }
    template<class T> vec<T,4>             qnlerp(const vec<T,4> & a, const vec<T,4> & b, T t) { return nlerp(a, dot(a,b) < 0 ? -b : b, t); }
    template<class T> vec<T,4>             qslerp(const vec<T,4> & a, const vec<T,4> & b, T t) { return slerp(a, dot(a,b) < 0 ? -b : b, t); }

    // Support for matrix algebra
    template<class T, int M> constexpr vec<T,M> mul(const mat<T,M,1> & a, const vec<T,1> & b) { return a.x*b.x; }
    template<class T, int M> constexpr vec<T,M> mul(const mat<T,M,2> & a, const vec<T,2> & b) { return a.x*b.x + a.y*b.y; }
    template<class T, int M> constexpr vec<T,M> mul(const mat<T,M,3> & a, const vec<T,3> & b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
    template<class T, int M> constexpr vec<T,M> mul(const mat<T,M,4> & a, const vec<T,4> & b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
    template<class T, int M, int N> constexpr mat<T,M,1> mul(const mat<T,M,N> & a, const mat<T,N,1> & b) { return {mul(a,b.x)}; }
    template<class T, int M, int N> constexpr mat<T,M,2> mul(const mat<T,M,N> & a, const mat<T,N,2> & b) { return {mul(a,b.x), mul(a,b.y)}; }
    template<class T, int M, int N> constexpr mat<T,M,3> mul(const mat<T,M,N> & a, const mat<T,N,3> & b) { return {mul(a,b.x), mul(a,b.y), mul(a,b.z)}; }
    template<class T, int M, int N> constexpr mat<T,M,4> mul(const mat<T,M,N> & a, const mat<T,N,4> & b) { return {mul(a,b.x), mul(a,b.y), mul(a,b.z), mul(a,b.w)}; }
    template<class T, int M, int N, int P> constexpr vec<T,M> mul(const mat<T,M,N> & a, const mat<T,N,P> & b, const vec<T,P> & c) { return mul(mul(a,b),c); }
    template<class T, int M, int N, int P, int Q> constexpr mat<T,M,Q> mul(const mat<T,M,N> & a, const mat<T,N,P> & b, const mat<T,P,Q> & c) { return mul(mul(a,b),c); }
    template<class T, int M, int N, int P, int Q> constexpr vec<T,M> mul(const mat<T,M,N> & a, const mat<T,N,P> & b, const mat<T,P,Q> & c, const vec<T,Q> & d) { return mul(mul(a,b,c),d); }
    template<class T, int M, int N, int P, int Q, int R> constexpr mat<T,M,R> mul(const mat<T,M,N> & a, const mat<T,N,P> & b, const mat<T,P,Q> & c, const mat<T,Q,R> & d) { return mul(mul(a,b,c),d); }
    template<class T, int M> constexpr mat<T,M,1> outerprod(const vec<T,M> & a, const vec<T,1> & b) { return {a*b.x}; }
    template<class T, int M> constexpr mat<T,M,2> outerprod(const vec<T,M> & a, const vec<T,2> & b) { return {a*b.x, a*b.y}; }
    template<class T, int M> constexpr mat<T,M,3> outerprod(const vec<T,M> & a, const vec<T,3> & b) { return {a*b.x, a*b.y, a*b.z}; }
    template<class T, int M> constexpr mat<T,M,4> outerprod(const vec<T,M> & a, const vec<T,4> & b) { return {a*b.x, a*b.y, a*b.z, a*b.w}; }
    template<class T> constexpr vec<T,1> diagonal(const mat<T,1,1> & a) { return {a.x.x}; }
    template<class T> constexpr vec<T,2> diagonal(const mat<T,2,2> & a) { return {a.x.x, a.y.y}; }
    template<class T> constexpr vec<T,3> diagonal(const mat<T,3,3> & a) { return {a.x.x, a.y.y, a.z.z}; }
    template<class T> constexpr vec<T,4> diagonal(const mat<T,4,4> & a) { return {a.x.x, a.y.y, a.z.z, a.w.w}; }
    template<class T, int N> constexpr T trace(const mat<T,N,N> & a) { return sum(diagonal(a)); }
    template<class T, int M> constexpr mat<T,M,1> transpose(const mat<T,1,M> & m) { return {m.row(0)}; }
    template<class T, int M> constexpr mat<T,M,2> transpose(const mat<T,2,M> & m) { return {m.row(0), m.row(1)}; }
    template<class T, int M> constexpr mat<T,M,3> transpose(const mat<T,3,M> & m) { return {m.row(0), m.row(1), m.row(2)}; }
    template<class T, int M> constexpr mat<T,M,4> transpose(const mat<T,4,M> & m) { return {m.row(0), m.row(1), m.row(2), m.row(3)}; }
    template<class T, int M> constexpr mat<T,1,M> transpose(const vec<T,M> & m) { return transpose(mat<T,M,1>(m)); }
    template<class T> constexpr mat<T,1,1> adjugate(const mat<T,1,1> & a) { return {vec<T,1>{1}}; }
    template<class T> constexpr mat<T,2,2> adjugate(const mat<T,2,2> & a) { return {{a.y.y, -a.x.y}, {-a.y.x, a.x.x}}; }
    template<class T> constexpr mat<T,3,3> adjugate(const mat<T,3,3> & a);
    template<class T> constexpr mat<T,4,4> adjugate(const mat<T,4,4> & a);
    template<class T, int N> constexpr mat<T,N,N> comatrix(const mat<T,N,N> & a) { return transpose(adjugate(a)); }
    template<class T> constexpr T determinant(const mat<T,1,1> & a) { return a.x.x; }
    template<class T> constexpr T determinant(const mat<T,2,2> & a) { return a.x.x*a.y.y - a.x.y*a.y.x; }
    template<class T> constexpr T determinant(const mat<T,3,3> & a) { return a.x.x*(a.y.y*a.z.z - a.z.y*a.y.z) + a.x.y*(a.y.z*a.z.x - a.z.z*a.y.x) + a.x.z*(a.y.x*a.z.y - a.z.x*a.y.y); }
    template<class T> constexpr T determinant(const mat<T,4,4> & a);
    template<class T, int N> constexpr mat<T,N,N> inverse(const mat<T,N,N> & a) { return adjugate(a)/determinant(a); }

    // Vectors and matrices can be used as ranges
    template<class T, int M>       T * begin(      vec<T,M> & a) { return &a.x; }
    template<class T, int M> const T * begin(const vec<T,M> & a) { return &a.x; }
    template<class T, int M>       T * end  (      vec<T,M> & a) { return begin(a) + M; }
    template<class T, int M> const T * end  (const vec<T,M> & a) { return begin(a) + M; }
    template<class T, int M, int N>       vec<T,M> * begin(      mat<T,M,N> & a) { return &a.x; }
    template<class T, int M, int N> const vec<T,M> * begin(const mat<T,M,N> & a) { return &a.x; }
    template<class T, int M, int N>       vec<T,M> * end  (      mat<T,M,N> & a) { return begin(a) + N; }
    template<class T, int M, int N> const vec<T,M> * end  (const mat<T,M,N> & a) { return begin(a) + N; }

    // Factory functions for 3D spatial transformations (will possibly be removed or changed in a future version)
    enum fwd_axis { neg_z, pos_z };                 // Should projection matrices be generated assuming forward is {0,0,-1} or {0,0,1}
    enum z_range { neg_one_to_one, zero_to_one };   // Should projection matrices map z into the range of [-1,1] or [0,1]?
    template<class T> vec<T,4>   rotation_quat     (const vec<T,3> & axis, T angle)         { return {axis*std::sin(angle/2), std::cos(angle/2)}; }
    template<class T> vec<T,4>   rotation_quat     (const mat<T,3,3> & m);
    template<class T> mat<T,4,4> translation_matrix(const vec<T,3> & translation)           { return {{1,0,0,0},{0,1,0,0},{0,0,1,0},{translation,1}}; }
    template<class T> mat<T,4,4> rotation_matrix   (const vec<T,4> & rotation)              { return {{qxdir(rotation),0}, {qydir(rotation),0}, {qzdir(rotation),0}, {0,0,0,1}}; }
    template<class T> mat<T,4,4> scaling_matrix    (const vec<T,3> & scaling)               { return {{scaling.x,0,0,0}, {0,scaling.y,0,0}, {0,0,scaling.z,0}, {0,0,0,1}}; }
    template<class T> mat<T,4,4> pose_matrix       (const vec<T,4> & q, const vec<T,3> & p) { return {{qxdir(q),0}, {qydir(q),0}, {qzdir(q),0}, {p,1}}; }
    template<class T> mat<T,4,4> lookat_matrix     (const vec<T,3> & eye, const vec<T,3> & center, const vec<T,3> & view_y_dir, fwd_axis fwd = neg_z);
    template<class T> mat<T,4,4> frustum_matrix    (T x0, T x1, T y0, T y1, T n, T f, fwd_axis a = neg_z, z_range z = neg_one_to_one);
    template<class T> mat<T,4,4> perspective_matrix(T fovy, T aspect, T n, T f, fwd_axis a = neg_z, z_range z = neg_one_to_one) { T y = n*std::tan(fovy / 2), x = y*aspect; return frustum_matrix(-x, x, -y, y, n, f, a, z); }

    // Provide implicit conversion between linalg::vec<T,M> and std::array<T,M>
    template<class T> struct converter<vec<T,1>, std::array<T,1>> { vec<T,1> operator() (const std::array<T,1> & a) const { return {a[0]}; } };
    template<class T> struct converter<vec<T,2>, std::array<T,2>> { vec<T,2> operator() (const std::array<T,2> & a) const { return {a[0], a[1]}; } };
    template<class T> struct converter<vec<T,3>, std::array<T,3>> { vec<T,3> operator() (const std::array<T,3> & a) const { return {a[0], a[1], a[2]}; } };
    template<class T> struct converter<vec<T,4>, std::array<T,4>> { vec<T,4> operator() (const std::array<T,4> & a) const { return {a[0], a[1], a[2], a[3]}; } };

    template<class T> struct converter<std::array<T,1>, vec<T,1>> { std::array<T,1> operator() (const vec<T,1> & a) const { return {a[0]}; } };
    template<class T> struct converter<std::array<T,2>, vec<T,2>> { std::array<T,2> operator() (const vec<T,2> & a) const { return {a[0], a[1]}; } };
    template<class T> struct converter<std::array<T,3>, vec<T,3>> { std::array<T,3> operator() (const vec<T,3> & a) const { return {a[0], a[1], a[2]}; } };
    template<class T> struct converter<std::array<T,4>, vec<T,4>> { std::array<T,4> operator() (const vec<T,4> & a) const { return {a[0], a[1], a[2], a[3]}; } };

    // Provide typedefs for common element types and vector/matrix sizes
    namespace aliases
    {
        typedef vec<bool,1> bool1; typedef vec<uint8_t,1> byte1; typedef vec<int16_t,1> short1; typedef vec<uint16_t,1> ushort1;
        typedef vec<bool,2> bool2; typedef vec<uint8_t,2> byte2; typedef vec<int16_t,2> short2; typedef vec<uint16_t,2> ushort2; 
        typedef vec<bool,3> bool3; typedef vec<uint8_t,3> byte3; typedef vec<int16_t,3> short3; typedef vec<uint16_t,3> ushort3; 
        typedef vec<bool,4> bool4; typedef vec<uint8_t,4> byte4; typedef vec<int16_t,4> short4; typedef vec<uint16_t,4> ushort4;
        typedef vec<int,1> int1; typedef vec<unsigned,1> uint1; typedef vec<float,1> float1; typedef vec<double,1> double1;
        typedef vec<int,2> int2; typedef vec<unsigned,2> uint2; typedef vec<float,2> float2; typedef vec<double,2> double2;
        typedef vec<int,3> int3; typedef vec<unsigned,3> uint3; typedef vec<float,3> float3; typedef vec<double,3> double3;
        typedef vec<int,4> int4; typedef vec<unsigned,4> uint4; typedef vec<float,4> float4; typedef vec<double,4> double4;
        typedef mat<bool,1,1> bool1x1; typedef mat<int,1,1> int1x1; typedef mat<float,1,1> float1x1; typedef mat<double,1,1> double1x1;
        typedef mat<bool,1,2> bool1x2; typedef mat<int,1,2> int1x2; typedef mat<float,1,2> float1x2; typedef mat<double,1,2> double1x2;
        typedef mat<bool,1,3> bool1x3; typedef mat<int,1,3> int1x3; typedef mat<float,1,3> float1x3; typedef mat<double,1,3> double1x3;
        typedef mat<bool,1,4> bool1x4; typedef mat<int,1,4> int1x4; typedef mat<float,1,4> float1x4; typedef mat<double,1,4> double1x4;
        typedef mat<bool,2,1> bool2x1; typedef mat<int,2,1> int2x1; typedef mat<float,2,1> float2x1; typedef mat<double,2,1> double2x1;
        typedef mat<bool,2,2> bool2x2; typedef mat<int,2,2> int2x2; typedef mat<float,2,2> float2x2; typedef mat<double,2,2> double2x2;
        typedef mat<bool,2,3> bool2x3; typedef mat<int,2,3> int2x3; typedef mat<float,2,3> float2x3; typedef mat<double,2,3> double2x3;
        typedef mat<bool,2,4> bool2x4; typedef mat<int,2,4> int2x4; typedef mat<float,2,4> float2x4; typedef mat<double,2,4> double2x4;
        typedef mat<bool,3,1> bool3x1; typedef mat<int,3,1> int3x1; typedef mat<float,3,1> float3x1; typedef mat<double,3,1> double3x1;
        typedef mat<bool,3,2> bool3x2; typedef mat<int,3,2> int3x2; typedef mat<float,3,2> float3x2; typedef mat<double,3,2> double3x2;
        typedef mat<bool,3,3> bool3x3; typedef mat<int,3,3> int3x3; typedef mat<float,3,3> float3x3; typedef mat<double,3,3> double3x3;
        typedef mat<bool,3,4> bool3x4; typedef mat<int,3,4> int3x4; typedef mat<float,3,4> float3x4; typedef mat<double,3,4> double3x4;
        typedef mat<bool,4,1> bool4x1; typedef mat<int,4,1> int4x1; typedef mat<float,4,1> float4x1; typedef mat<double,4,1> double4x1;
        typedef mat<bool,4,2> bool4x2; typedef mat<int,4,2> int4x2; typedef mat<float,4,2> float4x2; typedef mat<double,4,2> double4x2;
        typedef mat<bool,4,3> bool4x3; typedef mat<int,4,3> int4x3; typedef mat<float,4,3> float4x3; typedef mat<double,4,3> double4x3;
        typedef mat<bool,4,4> bool4x4; typedef mat<int,4,4> int4x4; typedef mat<float,4,4> float4x4; typedef mat<double,4,4> double4x4;
    }

    // Provide output streaming operators, writing something that resembles an aggregate literal that could be used to construct the specified value
    namespace ostream_overloads
    {
        template<class C, class T> std::basic_ostream<C> & operator << (std::basic_ostream<C> & out, const vec<T,1> & v) { return out << '{' << v[0] << '}'; }
        template<class C, class T> std::basic_ostream<C> & operator << (std::basic_ostream<C> & out, const vec<T,2> & v) { return out << '{' << v[0] << ',' << v[1] << '}'; }
        template<class C, class T> std::basic_ostream<C> & operator << (std::basic_ostream<C> & out, const vec<T,3> & v) { return out << '{' << v[0] << ',' << v[1] << ',' << v[2] << '}'; }
        template<class C, class T> std::basic_ostream<C> & operator << (std::basic_ostream<C> & out, const vec<T,4> & v) { return out << '{' << v[0] << ',' << v[1] << ',' << v[2] << ',' << v[3] << '}'; }

        template<class C, class T, int M> std::basic_ostream<C> & operator << (std::basic_ostream<C> & out, const mat<T,M,1> & m) { return out << '{' << m[0] << '}'; }
        template<class C, class T, int M> std::basic_ostream<C> & operator << (std::basic_ostream<C> & out, const mat<T,M,2> & m) { return out << '{' << m[0] << ',' << m[1] << '}'; }
        template<class C, class T, int M> std::basic_ostream<C> & operator << (std::basic_ostream<C> & out, const mat<T,M,3> & m) { return out << '{' << m[0] << ',' << m[1] << ',' << m[2] << '}'; }
        template<class C, class T, int M> std::basic_ostream<C> & operator << (std::basic_ostream<C> & out, const mat<T,M,4> & m) { return out << '{' << m[0] << ',' << m[1] << ',' << m[2] << ',' << m[3] << '}'; }
    }
}

namespace std 
{
    // Provide specializations for std::hash<...> with linalg types
    template<class T> struct hash<linalg::vec<T,1>> { std::size_t operator()(const linalg::vec<T,1> & v) const { std::hash<T> h; return h(v.x); } };
    template<class T> struct hash<linalg::vec<T,2>> { std::size_t operator()(const linalg::vec<T,2> & v) const { std::hash<T> h; return h(v.x) ^ (h(v.y) << 1); } };
    template<class T> struct hash<linalg::vec<T,3>> { std::size_t operator()(const linalg::vec<T,3> & v) const { std::hash<T> h; return h(v.x) ^ (h(v.y) << 1) ^ (h(v.z) << 2); } };
    template<class T> struct hash<linalg::vec<T,4>> { std::size_t operator()(const linalg::vec<T,4> & v) const { std::hash<T> h; return h(v.x) ^ (h(v.y) << 1) ^ (h(v.z) << 2) ^ (h(v.w) << 3); } };

    template<class T, int M> struct hash<linalg::mat<T,M,1>> { std::size_t operator()(const linalg::mat<T,M,1> & v) const { std::hash<linalg::vec<T,M>> h; return h(v.x); } };
    template<class T, int M> struct hash<linalg::mat<T,M,2>> { std::size_t operator()(const linalg::mat<T,M,2> & v) const { std::hash<linalg::vec<T,M>> h; return h(v.x) ^ (h(v.y) << M); } };
    template<class T, int M> struct hash<linalg::mat<T,M,3>> { std::size_t operator()(const linalg::mat<T,M,3> & v) const { std::hash<linalg::vec<T,M>> h; return h(v.x) ^ (h(v.y) << M) ^ (h(v.z) << (M*2)); } };
    template<class T, int M> struct hash<linalg::mat<T,M,4>> { std::size_t operator()(const linalg::mat<T,M,4> & v) const { std::hash<linalg::vec<T,M>> h; return h(v.x) ^ (h(v.y) << M) ^ (h(v.z) << (M*2)) ^ (h(v.w) << (M*3)); } };
}

// Definitions of functions too long to be defined inline
template<class T> constexpr linalg::mat<T,3,3> linalg::adjugate(const mat<T,3,3> & a) 
{
    return {{a.y.y*a.z.z - a.z.y*a.y.z, a.z.y*a.x.z - a.x.y*a.z.z, a.x.y*a.y.z - a.y.y*a.x.z},
            {a.y.z*a.z.x - a.z.z*a.y.x, a.z.z*a.x.x - a.x.z*a.z.x, a.x.z*a.y.x - a.y.z*a.x.x},
            {a.y.x*a.z.y - a.z.x*a.y.y, a.z.x*a.x.y - a.x.x*a.z.y, a.x.x*a.y.y - a.y.x*a.x.y}}; 
}

template<class T> constexpr linalg::mat<T,4,4> linalg::adjugate(const mat<T,4,4> & a) 
{
    return {{a.y.y*a.z.z*a.w.w + a.w.y*a.y.z*a.z.w + a.z.y*a.w.z*a.y.w - a.y.y*a.w.z*a.z.w - a.z.y*a.y.z*a.w.w - a.w.y*a.z.z*a.y.w,
             a.x.y*a.w.z*a.z.w + a.z.y*a.x.z*a.w.w + a.w.y*a.z.z*a.x.w - a.w.y*a.x.z*a.z.w - a.z.y*a.w.z*a.x.w - a.x.y*a.z.z*a.w.w,
             a.x.y*a.y.z*a.w.w + a.w.y*a.x.z*a.y.w + a.y.y*a.w.z*a.x.w - a.x.y*a.w.z*a.y.w - a.y.y*a.x.z*a.w.w - a.w.y*a.y.z*a.x.w,
             a.x.y*a.z.z*a.y.w + a.y.y*a.x.z*a.z.w + a.z.y*a.y.z*a.x.w - a.x.y*a.y.z*a.z.w - a.z.y*a.x.z*a.y.w - a.y.y*a.z.z*a.x.w},
            {a.y.z*a.w.w*a.z.x + a.z.z*a.y.w*a.w.x + a.w.z*a.z.w*a.y.x - a.y.z*a.z.w*a.w.x - a.w.z*a.y.w*a.z.x - a.z.z*a.w.w*a.y.x,
             a.x.z*a.z.w*a.w.x + a.w.z*a.x.w*a.z.x + a.z.z*a.w.w*a.x.x - a.x.z*a.w.w*a.z.x - a.z.z*a.x.w*a.w.x - a.w.z*a.z.w*a.x.x,
             a.x.z*a.w.w*a.y.x + a.y.z*a.x.w*a.w.x + a.w.z*a.y.w*a.x.x - a.x.z*a.y.w*a.w.x - a.w.z*a.x.w*a.y.x - a.y.z*a.w.w*a.x.x,
             a.x.z*a.y.w*a.z.x + a.z.z*a.x.w*a.y.x + a.y.z*a.z.w*a.x.x - a.x.z*a.z.w*a.y.x - a.y.z*a.x.w*a.z.x - a.z.z*a.y.w*a.x.x},
            {a.y.w*a.z.x*a.w.y + a.w.w*a.y.x*a.z.y + a.z.w*a.w.x*a.y.y - a.y.w*a.w.x*a.z.y - a.z.w*a.y.x*a.w.y - a.w.w*a.z.x*a.y.y,
             a.x.w*a.w.x*a.z.y + a.z.w*a.x.x*a.w.y + a.w.w*a.z.x*a.x.y - a.x.w*a.z.x*a.w.y - a.w.w*a.x.x*a.z.y - a.z.w*a.w.x*a.x.y,
             a.x.w*a.y.x*a.w.y + a.w.w*a.x.x*a.y.y + a.y.w*a.w.x*a.x.y - a.x.w*a.w.x*a.y.y - a.y.w*a.x.x*a.w.y - a.w.w*a.y.x*a.x.y,
             a.x.w*a.z.x*a.y.y + a.y.w*a.x.x*a.z.y + a.z.w*a.y.x*a.x.y - a.x.w*a.y.x*a.z.y - a.z.w*a.x.x*a.y.y - a.y.w*a.z.x*a.x.y},
            {a.y.x*a.w.y*a.z.z + a.z.x*a.y.y*a.w.z + a.w.x*a.z.y*a.y.z - a.y.x*a.z.y*a.w.z - a.w.x*a.y.y*a.z.z - a.z.x*a.w.y*a.y.z,
             a.x.x*a.z.y*a.w.z + a.w.x*a.x.y*a.z.z + a.z.x*a.w.y*a.x.z - a.x.x*a.w.y*a.z.z - a.z.x*a.x.y*a.w.z - a.w.x*a.z.y*a.x.z,
             a.x.x*a.w.y*a.y.z + a.y.x*a.x.y*a.w.z + a.w.x*a.y.y*a.x.z - a.x.x*a.y.y*a.w.z - a.w.x*a.x.y*a.y.z - a.y.x*a.w.y*a.x.z,
             a.x.x*a.y.y*a.z.z + a.z.x*a.x.y*a.y.z + a.y.x*a.z.y*a.x.z - a.x.x*a.z.y*a.y.z - a.y.x*a.x.y*a.z.z - a.z.x*a.y.y*a.x.z}}; 
}

template<class T> constexpr T linalg::determinant(const mat<T,4,4> & a) 
{
    return a.x.x*(a.y.y*a.z.z*a.w.w + a.w.y*a.y.z*a.z.w + a.z.y*a.w.z*a.y.w - a.y.y*a.w.z*a.z.w - a.z.y*a.y.z*a.w.w - a.w.y*a.z.z*a.y.w)
         + a.x.y*(a.y.z*a.w.w*a.z.x + a.z.z*a.y.w*a.w.x + a.w.z*a.z.w*a.y.x - a.y.z*a.z.w*a.w.x - a.w.z*a.y.w*a.z.x - a.z.z*a.w.w*a.y.x)
         + a.x.z*(a.y.w*a.z.x*a.w.y + a.w.w*a.y.x*a.z.y + a.z.w*a.w.x*a.y.y - a.y.w*a.w.x*a.z.y - a.z.w*a.y.x*a.w.y - a.w.w*a.z.x*a.y.y)
         + a.x.w*(a.y.x*a.w.y*a.z.z + a.z.x*a.y.y*a.w.z + a.w.x*a.z.y*a.y.z - a.y.x*a.z.y*a.w.z - a.w.x*a.y.y*a.z.z - a.z.x*a.w.y*a.y.z); 
}

template<class T> linalg::vec<T,4> linalg::rotation_quat(const mat<T,3,3> & m)
{
    const vec<T,4> q {m.x.x-m.y.y-m.z.z, m.y.y-m.x.x-m.z.z, m.z.z-m.x.x-m.y.y, m.x.x+m.y.y+m.z.z}, s[] {
        {1, m.x.y + m.y.x, m.z.x + m.x.z, m.y.z - m.z.y}, 
        {m.x.y + m.y.x, 1, m.y.z + m.z.y, m.z.x - m.x.z},
        {m.x.z + m.z.x, m.y.z + m.z.y, 1, m.x.y - m.y.x},
        {m.y.z - m.z.y, m.z.x - m.x.z, m.x.y - m.y.x, 1}};
    return copysign(normalize(sqrt(max(T(0), T(1)+q))), s[argmax(q)]);
}

template<class T> linalg::mat<T,4,4> linalg::lookat_matrix(const vec<T,3> & eye, const vec<T,3> & center, const vec<T,3> & view_y_dir, fwd_axis a)
{
    const vec<T,3> f = normalize(center - eye), z = a == pos_z ? f : -f, x = normalize(cross(view_y_dir, z)), y = cross(z, x);
    return inverse(mat<T,4,4>{{x,0},{y,0},{z,0},{eye,1}});
}

template<class T> linalg::mat<T,4,4> linalg::frustum_matrix(T x0, T x1, T y0, T y1, T n, T f, fwd_axis a, z_range z) 
{
    const T s = a == pos_z ? T(1) : T(-1), o = z == neg_one_to_one ? n : 0;
    return {{2*n/(x1-x0),0,0,0}, {0,2*n/(y1-y0),0,0}, {-s*(x0+x1)/(x1-x0),-s*(y0+y1)/(y1-y0),s*(f+o)/(f-n),s}, {0,0,-(n+o)*f/(f-n),0}};
}

#endif
