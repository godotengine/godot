#ifndef _C4_TYPES_HPP_
#define _C4_TYPES_HPP_

#include <stdint.h>
#include <stddef.h>
#include <type_traits>

#if __cplusplus >= 201103L
#include <utility>  // for integer_sequence and friends
#endif

#include "preprocessor.hpp"
#include "language.hpp"

/** @file types.hpp basic types, and utility macros and traits for types.
 * @ingroup basic_headers */

/** @defgroup types Type utilities */

namespace c4 {

/** @defgroup intrinsic_types Intrinsic types
 * @ingroup types
 * @{ */

using cbyte = const char; /**< a constant byte */
using  byte =       char; /**< a mutable byte */

using  i8 =   int8_t;
using i16 =  int16_t;
using i32 =  int32_t;
using i64 =  int64_t;
using  u8 =  uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using f32 =  float;
using f64 = double;

using ssize_t = typename std::make_signed<size_t>::type;

/** @} */

//--------------------------------------------------

/** @defgroup utility_types Utility types
 * @ingroup types
 * @{ */

// some tag types

#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#endif
#endif

/** a tag type for initializing the containers with variadic arguments a la
 * initializer_list, minus the initializer_list overload problems.
 */
struct aggregate_t {};
/** @see aggregate_t */
constexpr const aggregate_t aggregate{};

/** a tag type for specifying the initial capacity of allocatable contiguous storage */
struct with_capacity_t {};
/** @see with_capacity_t */
constexpr const with_capacity_t with_capacity{};

/** a tag type for disambiguating template parameter packs in variadic template overloads */
struct varargs_t {};
/** @see with_capacity_t */
constexpr const varargs_t varargs{};

#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif


//--------------------------------------------------

/** whether a value should be used in place of a const-reference in argument passing. */
template<class T>
struct cref_uses_val
{
    enum { value = (
    std::is_scalar<T>::value
    ||
    (
#if C4_CPP >= 20
        (std::is_trivially_copyable<T>::value && std::is_standard_layout<T>::value)
#else
        std::is_pod<T>::value
#endif
        &&
        sizeof(T) <= sizeof(size_t))) };
};
/** utility macro to override the default behaviour for c4::fastcref<T>
 @see fastcref */
#define C4_CREF_USES_VAL(T) \
template<>                  \
struct cref_uses_val<T>     \
{                           \
    enum { value = true };  \
};

/** Whether to use pass-by-value or pass-by-const-reference in a function argument
 * or return type. */
template<class T>
using fastcref = typename std::conditional<c4::cref_uses_val<T>::value, T, T const&>::type;

//--------------------------------------------------

/** Just what its name says. Useful sometimes as a default empty policy class. */
struct EmptyStruct
{
    template<class... T> EmptyStruct(T && ...){}
};

/** Just what its name says. Useful sometimes as a default policy class to
 * be inherited from. */
struct EmptyStructVirtual
{
    virtual ~EmptyStructVirtual() = default;
    template<class... T> EmptyStructVirtual(T && ...){}
};


/** */
template<class T>
struct inheritfrom : public T {};

//--------------------------------------------------
// Utilities to make a class obey size restrictions (eg, min size or size multiple of).
// DirectX usually makes this restriction with uniform buffers.
// This is also useful for padding to prevent false-sharing.

/** how many bytes must be added to size such that the result is at least minsize? */
C4_ALWAYS_INLINE constexpr size_t min_remainder(size_t size, size_t minsize) noexcept
{
    return size < minsize ? minsize-size : 0;
}

/** how many bytes must be added to size such that the result is a multiple of multipleof?  */
C4_ALWAYS_INLINE constexpr size_t mult_remainder(size_t size, size_t multipleof) noexcept
{
    return (((size % multipleof) != 0) ? (multipleof-(size % multipleof)) : 0);
}

/* force the following class to be tightly packed. */
#pragma pack(push, 1)
/** pad a class with more bytes at the end.
 * @see http://stackoverflow.com/questions/21092415/force-c-structure-to-pack-tightly */
template<class T, size_t BytesToPadAtEnd>
struct Padded : public T
{
    using T::T;
    using T::operator=;
    Padded(T const& val) : T(val) {}
    Padded(T && val) : T(val) {}
    char ___c4padspace___[BytesToPadAtEnd];
};
#pragma pack(pop)
/** When the padding argument is 0, we cannot declare the char[] array. */
template<class T>
struct Padded<T, 0> : public T
{
    using T::T;
    using T::operator=;
    Padded(T const& val) : T(val) {}
    Padded(T && val) : T(val) {}
};

/** make T have a size which is at least Min bytes */
template<class T, size_t Min>
using MinSized = Padded<T, min_remainder(sizeof(T), Min)>;

/** make T have a size which is a multiple of Mult bytes */
template<class T, size_t Mult>
using MultSized = Padded<T, mult_remainder(sizeof(T), Mult)>;

/** make T have a size which is simultaneously:
 *  -bigger or equal than Min
 *  -a multiple of Mult */
template<class T, size_t Min, size_t Mult>
using MinMultSized = MultSized<MinSized<T, Min>, Mult>;

/** make T be suitable for use as a uniform buffer. (at least with DirectX). */
template<class T>
using UbufSized = MinMultSized<T, 64, 16>;


//-----------------------------------------------------------------------------

#define C4_NO_COPY_CTOR(ty) ty(ty const&) = delete
#define C4_NO_MOVE_CTOR(ty) ty(ty     &&) = delete
#define C4_NO_COPY_ASSIGN(ty) ty& operator=(ty const&) = delete
#define C4_NO_MOVE_ASSIGN(ty) ty& operator=(ty     &&) = delete
#define C4_DEFAULT_COPY_CTOR(ty) ty(ty const&) noexcept = default
#define C4_DEFAULT_MOVE_CTOR(ty) ty(ty     &&) noexcept = default
#define C4_DEFAULT_COPY_ASSIGN(ty) ty& operator=(ty const&) noexcept = default
#define C4_DEFAULT_MOVE_ASSIGN(ty) ty& operator=(ty     &&) noexcept = default

#define C4_NO_COPY_OR_MOVE_CTOR(ty) \
    C4_NO_COPY_CTOR(ty); \
    C4_NO_MOVE_CTOR(ty)

#define C4_NO_COPY_OR_MOVE_ASSIGN(ty) \
    C4_NO_COPY_ASSIGN(ty); \
    C4_NO_MOVE_ASSIGN(ty)

#define C4_NO_COPY_OR_MOVE(ty) \
    C4_NO_COPY_OR_MOVE_CTOR(ty); \
    C4_NO_COPY_OR_MOVE_ASSIGN(ty)

#define C4_DEFAULT_COPY_AND_MOVE_CTOR(ty) \
    C4_DEFAULT_COPY_CTOR(ty); \
    C4_DEFAULT_MOVE_CTOR(ty)

#define C4_DEFAULT_COPY_AND_MOVE_ASSIGN(ty) \
    C4_DEFAULT_COPY_ASSIGN(ty); \
    C4_DEFAULT_MOVE_ASSIGN(ty)

#define C4_DEFAULT_COPY_AND_MOVE(ty) \
    C4_DEFAULT_COPY_AND_MOVE_CTOR(ty); \
    C4_DEFAULT_COPY_AND_MOVE_ASSIGN(ty)

/** @see https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable */
#define C4_MUST_BE_TRIVIAL_COPY(ty) \
    static_assert(std::is_trivially_copyable<ty>::value, #ty " must be trivially copyable")

/** @} */


//-----------------------------------------------------------------------------

/** @defgroup traits_types Type traits utilities
 * @ingroup types
 * @{ */

// http://stackoverflow.com/questions/10821380/is-t-an-instance-of-a-template-in-c
template<template<typename...> class X, typename    T> struct is_instance_of_tpl             : std::false_type {};
template<template<typename...> class X, typename... Y> struct is_instance_of_tpl<X, X<Y...>> : std::true_type {};

//-----------------------------------------------------------------------------

/** SFINAE. use this macro to enable a template function overload
based on a compile-time condition.
@code
// define an overload for a non-pod type
template<class T, C4_REQUIRE_T(std::is_pod<T>::value)>
void foo() { std::cout << "pod type\n"; }

// define an overload for a non-pod type
template<class T, C4_REQUIRE_T(!std::is_pod<T>::value)>
void foo() { std::cout << "nonpod type\n"; }

struct non_pod
{
    non_pod() : name("asdfkjhasdkjh") {}
    const char *name;
};

int main()
{
    foo<float>(); // prints "pod type"
    foo<non_pod>(); // prints "nonpod type"
}
@endcode */
#define C4_REQUIRE_T(cond) typename std::enable_if<cond, bool>::type* = nullptr

/** enable_if for a return type
 * @see C4_REQUIRE_T */
#define C4_REQUIRE_R(cond, type_) typename std::enable_if<cond, type_>::type

//-----------------------------------------------------------------------------
/** define a traits class reporting whether a type provides a member typedef */
#define C4_DEFINE_HAS_TYPEDEF(member_typedef)               \
template<typename T>                                        \
struct has_##stype                                          \
{                                                           \
private:                                                    \
                                                            \
    typedef char                      yes;                  \
    typedef struct { char array[2]; } no;                   \
                                                            \
    template<typename C>                                    \
    static yes _test(typename C::member_typedef*);          \
                                                            \
    template<typename C>                                    \
    static no  _test(...);                                  \
                                                            \
public:                                                     \
                                                            \
    enum { value = (sizeof(_test<T>(0)) == sizeof(yes)) };  \
                                                            \
}


/** @} */


//-----------------------------------------------------------------------------


/** @defgroup type_declarations Type declaration utilities
 * @ingroup types
 * @{ */

#define _c4_DEFINE_ARRAY_TYPES_WITHOUT_ITERATOR(T, I)           \
                                                                \
    using size_type = I;                                        \
    using ssize_type = typename std::make_signed<I>::type;      \
    using difference_type = typename std::make_signed<I>::type; \
                                                                \
    using value_type = T;                                       \
    using pointer = T*;                                         \
    using const_pointer = T const*;                             \
    using reference = T&;                                       \
    using const_reference = T const&

#define _c4_DEFINE_TUPLE_ARRAY_TYPES_WITHOUT_ITERATOR(interior_types, I) \
                                                                        \
    using size_type = I;                                                \
    using ssize_type = typename std::make_signed<I>::type;              \
    using difference_type = typename std::make_signed<I>::type;         \
                                                                        \
    template<I n> using value_type = typename std::tuple_element< n, std::tuple<interior_types...>>::type; \
    template<I n> using pointer = value_type<n>*;                       \
    template<I n> using const_pointer = value_type<n> const*;           \
    template<I n> using reference = value_type<n>&;                     \
    template<I n> using const_reference = value_type<n> const&


#define _c4_DEFINE_ARRAY_TYPES(T, I)                                \
                                                                    \
    _c4_DEFINE_ARRAY_TYPES_WITHOUT_ITERATOR(T, I);                  \
                                                                    \
    using iterator = T*;                                            \
    using const_iterator = T const*;                                \
    using reverse_iterator = std::reverse_iterator<T*>;             \
    using const_reverse_iterator = std::reverse_iterator<T const*>


#define _c4_DEFINE_TUPLE_ARRAY_TYPES(interior_types, I)                 \
                                                                        \
    _c4_DEFINE_TUPLE_ARRAY_TYPES_WITHOUT_ITERATOR(interior_types, I);   \
                                                                        \
    template<I n> using iterator = value_type<n>*;                      \
    template<I n> using const_iterator = value_type<n> const*;          \
    template<I n> using reverse_iterator = std::reverse_iterator< value_type<n>*>; \
    template<I n> using const_reverse_iterator = std::reverse_iterator< value_type<n> const*>



/** @} */


//-----------------------------------------------------------------------------


/** @defgroup compatility_utilities Backport implementation of some Modern C++ utilities
 * @ingroup types
 * @{ */

//-----------------------------------------------------------------------------
// index_sequence and friends are available only for C++14 and later.
// A C++11 implementation is provided here.
// This implementation was copied over from clang.
// see http://llvm.org/viewvc/llvm-project/libcxx/trunk/include/utility?revision=211563&view=markup#l687

#if __cplusplus > 201103L

using std::integer_sequence;
using std::index_sequence;
using std::make_integer_sequence;
using std::make_index_sequence;
using std::index_sequence_for;

#else

/** C++11 implementation of integer sequence
 * @see https://en.cppreference.com/w/cpp/utility/integer_sequence
 * @see taken from clang: http://llvm.org/viewvc/llvm-project/libcxx/trunk/include/utility?revision=211563&view=markup#l687 */
template<class _Tp, _Tp... _Ip>
struct integer_sequence
{
    static_assert(std::is_integral<_Tp>::value,
                  "std::integer_sequence can only be instantiated with an integral type" );
    using value_type = _Tp;
    static constexpr size_t size() noexcept { return sizeof...(_Ip); }
};

/** C++11 implementation of index sequence
 * @see https://en.cppreference.com/w/cpp/utility/integer_sequence
 * @see taken from clang: http://llvm.org/viewvc/llvm-project/libcxx/trunk/include/utility?revision=211563&view=markup#l687 */
template<size_t... _Ip>
using index_sequence = integer_sequence<size_t, _Ip...>;

/** @cond DONT_DOCUMENT_THIS */
namespace __detail {

template<typename _Tp, size_t ..._Extra>
struct __repeat;

template<typename _Tp, _Tp ..._Np, size_t ..._Extra>
struct __repeat<integer_sequence<_Tp, _Np...>, _Extra...>
{
    using type = integer_sequence<_Tp,
                            _Np...,
                            sizeof...(_Np) + _Np...,
                            2 * sizeof...(_Np) + _Np...,
                            3 * sizeof...(_Np) + _Np...,
                            4 * sizeof...(_Np) + _Np...,
                            5 * sizeof...(_Np) + _Np...,
                            6 * sizeof...(_Np) + _Np...,
                            7 * sizeof...(_Np) + _Np...,
                            _Extra...>;
};

template<size_t _Np> struct __parity;
template<size_t _Np> struct __make : __parity<_Np % 8>::template __pmake<_Np> {};

template<> struct __make<0> { using type = integer_sequence<size_t>; };
template<> struct __make<1> { using type = integer_sequence<size_t, 0>; };
template<> struct __make<2> { using type = integer_sequence<size_t, 0, 1>; };
template<> struct __make<3> { using type = integer_sequence<size_t, 0, 1, 2>; };
template<> struct __make<4> { using type = integer_sequence<size_t, 0, 1, 2, 3>; };
template<> struct __make<5> { using type = integer_sequence<size_t, 0, 1, 2, 3, 4>; };
template<> struct __make<6> { using type = integer_sequence<size_t, 0, 1, 2, 3, 4, 5>; };
template<> struct __make<7> { using type = integer_sequence<size_t, 0, 1, 2, 3, 4, 5, 6>; };

template<> struct __parity<0> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type> {}; };
template<> struct __parity<1> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 1> {}; };
template<> struct __parity<2> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 2, _Np - 1> {}; };
template<> struct __parity<3> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 3, _Np - 2, _Np - 1> {}; };
template<> struct __parity<4> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 4, _Np - 3, _Np - 2, _Np - 1> {}; };
template<> struct __parity<5> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 5, _Np - 4, _Np - 3, _Np - 2, _Np - 1> {}; };
template<> struct __parity<6> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 6, _Np - 5, _Np - 4, _Np - 3, _Np - 2, _Np - 1> {}; };
template<> struct __parity<7> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 7, _Np - 6, _Np - 5, _Np - 4, _Np - 3, _Np - 2, _Np - 1> {}; };

template<typename _Tp, typename _Up>
struct __convert
{
    template<typename> struct __result;
    template<_Tp ..._Np> struct __result<integer_sequence<_Tp, _Np...>>
    {
        using type = integer_sequence<_Up, _Np...>;
    };
};

template<typename _Tp>
struct __convert<_Tp, _Tp>
{
    template<typename _Up> struct __result
    {
         using type = _Up;
    };
};

template<typename _Tp, _Tp _Np>
using __make_integer_sequence_unchecked = typename __detail::__convert<size_t, _Tp>::template __result<typename __detail::__make<_Np>::type>::type;

template<class _Tp, _Tp _Ep>
struct __make_integer_sequence
{
    static_assert(std::is_integral<_Tp>::value,
                  "std::make_integer_sequence can only be instantiated with an integral type" );
    static_assert(0 <= _Ep, "std::make_integer_sequence input shall not be negative");
    typedef __make_integer_sequence_unchecked<_Tp, _Ep> type;
};

} // namespace __detail
/** @endcond */


/** C++11 implementation of index sequence
 * @see https://en.cppreference.com/w/cpp/utility/integer_sequence
 * @see taken from clang: http://llvm.org/viewvc/llvm-project/libcxx/trunk/include/utility?revision=211563&view=markup#l687 */
template<class _Tp, _Tp _Np>
using make_integer_sequence = typename __detail::__make_integer_sequence<_Tp, _Np>::type;

/** C++11 implementation of index sequence
 * @see https://en.cppreference.com/w/cpp/utility/integer_sequence
 * @see taken from clang: http://llvm.org/viewvc/llvm-project/libcxx/trunk/include/utility?revision=211563&view=markup#l687 */
template<size_t _Np>
using make_index_sequence = make_integer_sequence<size_t, _Np>;

/** C++11 implementation of index sequence
 * @see https://en.cppreference.com/w/cpp/utility/integer_sequence
 * @see taken from clang: http://llvm.org/viewvc/llvm-project/libcxx/trunk/include/utility?revision=211563&view=markup#l687 */
template<class... _Tp>
using index_sequence_for = make_index_sequence<sizeof...(_Tp)>;
#endif

/** @} */


} // namespace c4

#endif /* _C4_TYPES_HPP_ */
