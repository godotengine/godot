#ifndef _C4_CTOR_DTOR_HPP_
#define _C4_CTOR_DTOR_HPP_

#include "preprocessor.hpp"
#include "language.hpp"
#include "memory_util.hpp"
#include "error.hpp"

#include <type_traits>
#include <utility> // std::forward

/** @file ctor_dtor.hpp object construction and destruction facilities.
 * Some of these are not yet available in C++11. */

namespace c4 {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

/** default-construct an object, trivial version */
template <class U> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_default_constructible<U>::value, void>::type
construct(U *ptr) noexcept
{
    memset(ptr, 0, sizeof(U));
}
/** default-construct an object, non-trivial version */
template<class U> C4_ALWAYS_INLINE typename std ::enable_if< ! std::is_trivially_default_constructible<U>::value, void>::type
construct(U* ptr) noexcept
{
    new ((void*)ptr) U();
}

/** default-construct n objects, trivial version */
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_default_constructible<U>::value, void>::type
construct_n(U* ptr, I n) noexcept
{
    memset(ptr, 0, n * sizeof(U));
}
/** default-construct n objects, non-trivial version */
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_default_constructible<U>::value, void>::type
construct_n(U* ptr, I n) noexcept
{
    for(I i = 0; i < n; ++i)
    {
        new ((void*)(ptr + i)) U();
    }
}

#ifdef __clang__
#   pragma clang diagnostic push
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   if __GNUC__ >= 6
#       pragma GCC diagnostic ignored "-Wnull-dereference"
#   endif
#endif

template<class U, class ...Args>
inline void construct(U* ptr, Args&&... args)
{
    new ((void*)ptr) U(std::forward<Args>(args)...);
}
template<class U, class I, class ...Args>
inline void construct_n(U* ptr, I n, Args&&... args)
{
    for(I i = 0; i < n; ++i)
    {
        new ((void*)(ptr + i)) U(args...);
    }
}

#ifdef __clang__
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#endif


//-----------------------------------------------------------------------------
// copy-construct

template<class U> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_copy_constructible<U>::value, void>::type
copy_construct(U* dst, U const* src) noexcept
{
    C4_ASSERT(dst != src);
    memcpy(dst, src, sizeof(U));
}
template<class U> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_copy_constructible<U>::value, void>::type
copy_construct(U* dst, U const* src)
{
    C4_ASSERT(dst != src);
    new ((void*)dst) U(*src);
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_copy_constructible<U>::value, void>::type
copy_construct_n(U* dst, U const* src, I n) noexcept
{
    C4_ASSERT(dst != src);
    memcpy(dst, src, n * sizeof(U));
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_copy_constructible<U>::value, void>::type
copy_construct_n(U* dst, U const* src, I n)
{
    C4_ASSERT(dst != src);
    for(I i = 0; i < n; ++i)
    {
        new ((void*)(dst + i)) U(*(src + i));
    }
}

template<class U> C4_ALWAYS_INLINE typename std::enable_if<std::is_scalar<U>::value, void>::type
copy_construct(U* dst, U src) noexcept // pass by value for scalar types
{
    *dst = src;
}
template<class U> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_scalar<U>::value, void>::type
copy_construct(U* dst, U const& src) // pass by reference for non-scalar types
{
    C4_ASSERT(dst != &src);
    new ((void*)dst) U(src);
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_scalar<U>::value, void>::type
copy_construct_n(U* dst, U src, I n) noexcept // pass by value for scalar types
{
    for(I i = 0; i < n; ++i)
    {
        dst[i] = src;
    }
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_scalar<U>::value, void>::type
copy_construct_n(U* dst, U const& src, I n) // pass by reference for non-scalar types
{
    C4_ASSERT(dst != &src);
    for(I i = 0; i < n; ++i)
    {
        new ((void*)(dst + i)) U(src);
    }
}

template<class U, size_t N>
C4_ALWAYS_INLINE void copy_construct(U (&dst)[N], U const (&src)[N]) noexcept
{
    copy_construct_n(dst, src, N);
}

//-----------------------------------------------------------------------------
// copy-assign

template<class U> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_copy_assignable<U>::value, void>::type
copy_assign(U* dst, U const* src) noexcept
{
    C4_ASSERT(dst != src);
    memcpy(dst, src, sizeof(U));
}
template<class U> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_copy_assignable<U>::value, void>::type
copy_assign(U* dst, U const* src) noexcept
{
    C4_ASSERT(dst != src);
    *dst = *src;
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_copy_assignable<U>::value, void>::type
copy_assign_n(U* dst, U const* src, I n) noexcept
{
    C4_ASSERT(dst != src);
    memcpy(dst, src, n * sizeof(U));
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_copy_assignable<U>::value, void>::type
copy_assign_n(U* dst, U const* src, I n) noexcept
{
    C4_ASSERT(dst != src);
    for(I i = 0; i < n; ++i)
    {
        dst[i] = src[i];
    }
}

template<class U> C4_ALWAYS_INLINE typename std::enable_if<std::is_scalar<U>::value, void>::type
copy_assign(U* dst, U src) noexcept // pass by value for scalar types
{
    *dst = src;
}
template<class U> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_scalar<U>::value, void>::type
copy_assign(U* dst, U const& src) noexcept // pass by reference for non-scalar types
{
    C4_ASSERT(dst != &src);
    *dst = src;
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_scalar<U>::value, void>::type
copy_assign_n(U* dst, U src, I n) noexcept // pass by value for scalar types
{
    for(I i = 0; i < n; ++i)
    {
        dst[i] = src;
    }
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_scalar<U>::value, void>::type
copy_assign_n(U* dst, U const& src, I n) noexcept // pass by reference for non-scalar types
{
    C4_ASSERT(dst != &src);
    for(I i = 0; i < n; ++i)
    {
        dst[i] = src;
    }
}

template<class U, size_t N>
C4_ALWAYS_INLINE void copy_assign(U (&dst)[N], U const (&src)[N]) noexcept
{
    copy_assign_n(dst, src, N);
}

//-----------------------------------------------------------------------------
// move-construct

template<class U> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_move_constructible<U>::value, void>::type
move_construct(U* dst, U* src) noexcept
{
    C4_ASSERT(dst != src);
    memcpy(dst, src, sizeof(U));
}
template<class U> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_move_constructible<U>::value, void>::type
move_construct(U* dst, U* src) noexcept
{
    C4_ASSERT(dst != src);
    new ((void*)dst) U(std::move(*src));
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_move_constructible<U>::value, void>::type
move_construct_n(U* dst, U* src, I n) noexcept
{
    C4_ASSERT(dst != src);
    memcpy(dst, src, n * sizeof(U));
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_move_constructible<U>::value, void>::type
move_construct_n(U* dst, U* src, I n) noexcept
{
    C4_ASSERT(dst != src);
    for(I i = 0; i < n; ++i)
    {
        new ((void*)(dst + i)) U(std::move(src[i]));
    }
}

//-----------------------------------------------------------------------------
// move-assign

template<class U> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_move_assignable<U>::value, void>::type
move_assign(U* dst, U* src) noexcept
{
    C4_ASSERT(dst != src);
    memcpy(dst, src, sizeof(U));
}
template<class U> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_move_assignable<U>::value, void>::type
move_assign(U* dst, U* src) noexcept
{
    C4_ASSERT(dst != src);
    *dst = std::move(*src);
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_move_assignable<U>::value, void>::type
move_assign_n(U* dst, U* src, I n) noexcept
{
    C4_ASSERT(dst != src);
    memcpy(dst, src, n * sizeof(U));
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_move_assignable<U>::value, void>::type
move_assign_n(U* dst, U* src, I n) noexcept
{
    C4_ASSERT(dst != src);
    for(I i = 0; i < n; ++i)
    {
        *(dst + i) = std::move(*(src + i));
    }
}

//-----------------------------------------------------------------------------
// destroy

template<class U> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_destructible<U>::value, void>::type
destroy(U* ptr) noexcept
{
    C4_UNUSED(ptr); // nothing to do
}
template<class U> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_destructible<U>::value, void>::type
destroy(U* ptr) noexcept
{
    ptr->~U();
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_destructible<U>::value, void>::type
destroy_n(U* ptr, I n) noexcept
{
    C4_UNUSED(ptr);
    C4_UNUSED(n); // nothing to do
}
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_destructible<U>::value, void>::type
destroy_n(U* ptr, I n) noexcept
{
    for(I i = 0; i <n; ++i)
    {
        ptr[i].~U();
    }
}

//-----------------------------------------------------------------------------

/** makes room at the beginning of buf, which has a current size of n */
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_move_constructible<U>::value, void>::type
make_room(U *buf, I bufsz, I room) C4_NOEXCEPT_A
{
    C4_ASSERT(bufsz >= 0 && room >= 0);
    if(room >= bufsz)
    {
        memcpy (buf + room, buf, bufsz * sizeof(U));
    }
    else
    {
        memmove(buf + room, buf, bufsz * sizeof(U));
    }
}
/** makes room at the beginning of buf, which has a current size of bufsz */
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_move_constructible<U>::value, void>::type
make_room(U *buf, I bufsz, I room) C4_NOEXCEPT_A
{
    C4_ASSERT(bufsz >= 0 && room >= 0);
    if(room >= bufsz)
    {
        for(I i = 0; i < bufsz; ++i)
        {
            new ((void*)(buf + (i + room))) U(std::move(buf[i]));
        }
    }
    else
    {
        for(I i = 0; i < bufsz; ++i)
        {
            I w = bufsz-1 - i; // do a backwards loop
            new ((void*)(buf + (w + room))) U(std::move(buf[w]));
        }
    }
}

/** make room to the right of pos */
template<class U, class I>
C4_ALWAYS_INLINE void make_room(U *buf, I bufsz, I currsz, I pos, I room)
{
    C4_ASSERT(pos >= 0 && pos <= currsz);
    C4_ASSERT(currsz <= bufsz);
    C4_ASSERT(room + currsz <= bufsz);
    C4_UNUSED(bufsz);
    make_room(buf + pos, currsz - pos, room);
}


/** make room to the right of pos, copying to the beginning of a different buffer */
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_move_constructible<U>::value, void>::type
make_room(U *dst, U const* src, I srcsz, I room, I pos) C4_NOEXCEPT_A
{
    C4_ASSERT(srcsz >= 0 && room >= 0 && pos >= 0);
    C4_ASSERT(pos < srcsz || (pos == 0 && srcsz == 0));
    memcpy(dst             , src      , pos           * sizeof(U));
    memcpy(dst + room + pos, src + pos, (srcsz - pos) * sizeof(U));
}
/** make room to the right of pos, copying to the beginning of a different buffer */
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_move_constructible<U>::value, void>::type
make_room(U *dst, U const* src, I srcsz, I room, I pos)
{
    C4_ASSERT(srcsz >= 0 && room >= 0 && pos >= 0);
    C4_ASSERT(pos < srcsz || (pos == 0 && srcsz == 0));
    for(I i = 0; i < pos; ++i)
    {
        new ((void*)(dst + i)) U(std::move(src[i]));
    }
    src += pos;
    dst += room + pos;
    for(I i = 0, e = srcsz - pos; i < e; ++i)
    {
        new ((void*)(dst + i)) U(std::move(src[i]));
    }
}

template<class U, class I>
C4_ALWAYS_INLINE void make_room
(
    U      * dst, I dstsz,
    U const* src, I srcsz,
    I room, I pos
)
{
    C4_ASSERT(pos >= 0 && pos < srcsz || (srcsz == 0 && pos == 0));
    C4_ASSERT(pos >= 0 && pos < dstsz || (dstsz == 0 && pos == 0));
    C4_ASSERT(srcsz+room <= dstsz);
    C4_UNUSED(dstsz);
    make_room(dst, src, srcsz, room, pos);
}


//-----------------------------------------------------------------------------
/** destroy room at the beginning of buf, which has a current size of n */
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_scalar<U>::value || (std::is_standard_layout<U>::value && std::is_trivial<U>::value), void>::type
destroy_room(U *buf, I n, I room) C4_NOEXCEPT_A
{
    C4_ASSERT(n >= 0 && room >= 0);
    C4_ASSERT(room <= n);
    if(room < n)
    {
        memmove(buf, buf + room, (n - room) * sizeof(U));
    }
    else
    {
        // nothing to do - no need to destroy scalar types
    }
}
/** destroy room at the beginning of buf, which has a current size of n */
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! (std::is_scalar<U>::value || (std::is_standard_layout<U>::value && std::is_trivial<U>::value)), void>::type
destroy_room(U *buf, I n, I room)
{
    C4_ASSERT(n >= 0 && room >= 0);
    C4_ASSERT(room <= n);
    if(room < n)
    {
        for(I i = 0, e = n - room; i < e; ++i)
        {
            buf[i] = std::move(buf[i + room]);
        }
    }
    else
    {
        for(I i = 0; i < n; ++i)
        {
            buf[i].~U();
        }
    }
}

/** destroy room to the right of pos, copying to a different buffer */
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if<std::is_trivially_move_constructible<U>::value, void>::type
destroy_room(U *dst, U const* src, I n, I room, I pos) C4_NOEXCEPT_A
{
    C4_ASSERT(n >= 0 && room >= 0 && pos >= 0);
    C4_ASSERT(pos <n);
    C4_ASSERT(pos + room <= n);
    memcpy(dst, src, pos * sizeof(U));
    memcpy(dst + pos, src + room + pos, (n - pos - room) * sizeof(U));
}
/** destroy room to the right of pos, copying to a different buffer */
template<class U, class I> C4_ALWAYS_INLINE typename std::enable_if< ! std::is_trivially_move_constructible<U>::value, void>::type
destroy_room(U *dst, U const* src, I n, I room, I pos)
{
    C4_ASSERT(n >= 0 && room >= 0 && pos >= 0);
    C4_ASSERT(pos < n);
    C4_ASSERT(pos + room <= n);
    for(I i = 0; i < pos; ++i)
    {
        new ((void*)(dst + i)) U(std::move(src[i]));
    }
    src += room + pos;
    dst += pos;
    for(I i = 0, e = n - pos - room; i < e; ++i)
    {
        new ((void*)(dst + i)) U(std::move(src[i]));
    }
}

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace c4

#endif /* _C4_CTOR_DTOR_HPP_ */
