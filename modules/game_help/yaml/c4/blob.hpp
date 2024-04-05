#ifndef _C4_BLOB_HPP_
#define _C4_BLOB_HPP_

#include "types.hpp"
#include "error.hpp"

/** @file blob.hpp Mutable and immutable binary data blobs.
*/

namespace c4 {

template<class T>
struct blob_;

namespace detail {
template<class T> struct is_blob_type : std::integral_constant<bool, false> {};
template<class T> struct is_blob_type<blob_<T>> : std::integral_constant<bool, true> {};
template<class T> struct is_blob_value_type : std::integral_constant<bool, (std::is_fundamental<T>::value || std::is_trivially_copyable<T>::value)> {};
} // namespace

template<class T>
struct blob_
{
    static_assert(std::is_same<T, byte>::value || std::is_same<T, cbyte>::value, "must be either byte or cbyte");
    static_assert(sizeof(T) == 1u, "must be either byte or cbyte");

public:

    T *    buf;
    size_t len;

public:

    C4_ALWAYS_INLINE blob_() noexcept = default;
    C4_ALWAYS_INLINE blob_(blob_ const& that) noexcept = default;
    C4_ALWAYS_INLINE blob_(blob_     && that) noexcept = default;
    C4_ALWAYS_INLINE blob_& operator=(blob_     && that) noexcept = default;
    C4_ALWAYS_INLINE blob_& operator=(blob_ const& that) noexcept = default;

    template<class U, class=typename std::enable_if<std::is_const<T>::value && std::is_same<typename std::add_const<U>::type, T>::value, U>::type> C4_ALWAYS_INLINE blob_(blob_<U> const& that) noexcept : buf(that.buf), len(that.len) {}
    template<class U, class=typename std::enable_if<std::is_const<T>::value && std::is_same<typename std::add_const<U>::type, T>::value, U>::type> C4_ALWAYS_INLINE blob_(blob_<U>     && that) noexcept : buf(that.buf), len(that.len) {}
    template<class U, class=typename std::enable_if<std::is_const<T>::value && std::is_same<typename std::add_const<U>::type, T>::value, U>::type> C4_ALWAYS_INLINE blob_& operator=(blob_<U>     && that) noexcept { buf = that.buf; len = that.len; }
    template<class U, class=typename std::enable_if<std::is_const<T>::value && std::is_same<typename std::add_const<U>::type, T>::value, U>::type> C4_ALWAYS_INLINE blob_& operator=(blob_<U> const& that) noexcept { buf = that.buf; len = that.len; }

    C4_ALWAYS_INLINE blob_(void       *ptr, size_t n) noexcept : buf(reinterpret_cast<T*>(ptr)), len(n) {}
    C4_ALWAYS_INLINE blob_(void const *ptr, size_t n) noexcept : buf(reinterpret_cast<T*>(ptr)), len(n) {}

    #define _C4_REQUIRE_BLOBTYPE(ty) class=typename std::enable_if<((!detail::is_blob_type<ty>::value) && (detail::is_blob_value_type<ty>::value)), T>::type
    template<class U, _C4_REQUIRE_BLOBTYPE(U)> C4_ALWAYS_INLINE blob_(U &var) noexcept : buf(reinterpret_cast<T*>(&var)), len(sizeof(U)) {}
    template<class U, _C4_REQUIRE_BLOBTYPE(U)> C4_ALWAYS_INLINE blob_(U *ptr, size_t n) noexcept : buf(reinterpret_cast<T*>(ptr)), len(sizeof(U) * n) { C4_ASSERT(is_aligned(ptr)); }
    template<class U, _C4_REQUIRE_BLOBTYPE(U)> C4_ALWAYS_INLINE blob_& operator= (U &var) noexcept { buf = reinterpret_cast<T*>(&var); len = sizeof(U); return *this; }
    template<class U, size_t N, _C4_REQUIRE_BLOBTYPE(U)> C4_ALWAYS_INLINE blob_(U (&arr)[N]) noexcept : buf(reinterpret_cast<T*>(arr)), len(sizeof(U) * N) {}
    template<class U, size_t N, _C4_REQUIRE_BLOBTYPE(U)> C4_ALWAYS_INLINE blob_& operator= (U (&arr)[N]) noexcept { buf = reinterpret_cast<T*>(arr); len = sizeof(U) * N; return *this; }
    #undef _C4_REQUIRE_BLOBTYPE
};

/** an immutable binary blob */
using cblob = blob_<cbyte>;
/** a mutable binary blob */
using  blob = blob_< byte>;

C4_MUST_BE_TRIVIAL_COPY(blob);
C4_MUST_BE_TRIVIAL_COPY(cblob);

} // namespace c4

#endif // _C4_BLOB_HPP_
