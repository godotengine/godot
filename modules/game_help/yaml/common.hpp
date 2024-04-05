#ifndef _C4_YML_COMMON_HPP_
#define _C4_YML_COMMON_HPP_

#include <cstddef>
#include "./c4/substr.hpp"
#include "export.hpp"


#ifndef RYML_USE_ASSERT
#   define RYML_USE_ASSERT C4_USE_ASSERT
#endif


#if RYML_USE_ASSERT
#   define RYML_ASSERT(cond) RYML_CHECK(cond)
#   define RYML_ASSERT_MSG(cond, msg) RYML_CHECK_MSG(cond, msg)
#else
#   define RYML_ASSERT(cond)
#   define RYML_ASSERT_MSG(cond, msg)
#endif


#if defined(NDEBUG) || defined(C4_NO_DEBUG_BREAK)
#   define RYML_DEBUG_BREAK()
#else
#   define RYML_DEBUG_BREAK()                               \
    {                                                       \
        if(c4::get_error_flags() & c4::ON_ERROR_DEBUGBREAK) \
        {                                                   \
            C4_DEBUG_BREAK();                               \
        }                                                   \
    }
#endif


#define RYML_CHECK(cond)                                                \
    do {                                                                \
        if(!(cond))                                                     \
        {                                                               \
            RYML_DEBUG_BREAK()                                          \
            c4::yml::error("check failed: " #cond, c4::yml::Location(__FILE__, __LINE__, 0)); \
        }                                                               \
    } while(0)

#define RYML_CHECK_MSG(cond, msg)                                       \
    do                                                                  \
    {                                                                   \
        if(!(cond))                                                     \
        {                                                               \
            RYML_DEBUG_BREAK()                                          \
            c4::yml::error(msg ": check failed: " #cond, c4::yml::Location(__FILE__, __LINE__, 0)); \
        }                                                               \
    } while(0)


#if C4_CPP >= 14
#   define RYML_DEPRECATED(msg) [[deprecated(msg)]]
#else
#   if defined(_MSC_VER)
#       define RYML_DEPRECATED(msg) __declspec(deprecated(msg))
#   else // defined(__GNUC__) || defined(__clang__)
#       define RYML_DEPRECATED(msg) __attribute__((deprecated(msg)))
#   endif
#endif


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

namespace c4 {
namespace yml {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

enum : size_t {
    /** a null position */
    npos = size_t(-1),
    /** an index to none */
    NONE = size_t(-1)
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//! holds a position into a source buffer
struct RYML_EXPORT LineCol
{
    //! number of bytes from the beginning of the source buffer
    size_t offset;
    //! line
    size_t line;
    //! column
    size_t col;

    LineCol() : offset(), line(), col() {}
    //! construct from line and column
    LineCol(size_t l, size_t c) : offset(0), line(l), col(c) {}
    //! construct from offset, line and column
    LineCol(size_t o, size_t l, size_t c) : offset(o), line(l), col(c) {}
};


//! a source file position
struct RYML_EXPORT Location : public LineCol
{
    csubstr name;

    operator bool () const { return !name.empty() || line != 0 || offset != 0; }

    Location() : LineCol(), name() {}
    Location(                         size_t l, size_t c) : LineCol{   l, c}, name( ) {}
    Location(    csubstr n,           size_t l, size_t c) : LineCol{   l, c}, name(n) {}
    Location(    csubstr n, size_t b, size_t l, size_t c) : LineCol{b, l, c}, name(n) {}
    Location(const char *n,           size_t l, size_t c) : LineCol{   l, c}, name(to_csubstr(n)) {}
    Location(const char *n, size_t b, size_t l, size_t c) : LineCol{b, l, c}, name(to_csubstr(n)) {}
};


//-----------------------------------------------------------------------------

/** the type of the function used to report errors. This function must
 * interrupt execution, either by raising an exception or calling
 * std::abort().
 *
 * @warning the error callback must never return: it must either abort
 * or throw an exception. Otherwise, the parser will enter into an
 * infinite loop, or the program may crash. */
using pfn_error = void (*)(const char* msg, size_t msg_len, Location location, void *user_data);
/** the type of the function used to allocate memory */
using pfn_allocate = void* (*)(size_t len, void* hint, void *user_data);
/** the type of the function used to free memory */
using pfn_free = void (*)(void* mem, size_t size, void *user_data);

/** trigger an error: call the current error callback. */
RYML_EXPORT void error(const char *msg, size_t msg_len, Location loc);
/** @overload error */
inline void error(const char *msg, size_t msg_len)
{
    error(msg, msg_len, Location{});
}
/** @overload error */
template<size_t N>
inline void error(const char (&msg)[N], Location loc)
{
    error(msg, N-1, loc);
}
/** @overload error */
template<size_t N>
inline void error(const char (&msg)[N])
{
    error(msg, N-1, Location{});
}

//-----------------------------------------------------------------------------

/** a c-style callbacks class
 *
 * @warning the error callback must never return: it must either abort
 * or throw an exception. Otherwise, the parser will enter into an
 * infinite loop, or the program may crash. */
struct RYML_EXPORT Callbacks
{
    void *       m_user_data;
    pfn_allocate m_allocate;
    pfn_free     m_free;
    pfn_error    m_error;

    Callbacks();
    Callbacks(void *user_data, pfn_allocate alloc, pfn_free free, pfn_error error_);

    bool operator!= (Callbacks const& that) const { return !operator==(that); }
    bool operator== (Callbacks const& that) const
    {
        return (m_user_data == that.m_user_data &&
                m_allocate == that.m_allocate &&
                m_free == that.m_free &&
                m_error == that.m_error);
    }
};

/** set the global callbacks.
 *
 * @warning the error callback must never return: it must either abort
 * or throw an exception. Otherwise, the parser will enter into an
 * infinite loop, or the program may crash. */
RYML_EXPORT void set_callbacks(Callbacks const& c);
/// get the global callbacks
RYML_EXPORT Callbacks const& get_callbacks();
/// set the global callbacks back to their defaults
RYML_EXPORT void reset_callbacks();

/// @cond dev
#define _RYML_CB_ERR(cb, msg_literal)                                   \
do                                                                      \
{                                                                       \
    const char msg[] = msg_literal;                                     \
    RYML_DEBUG_BREAK()                                                  \
    (cb).m_error(msg, sizeof(msg), c4::yml::Location(__FILE__, 0, __LINE__, 0), (cb).m_user_data); \
} while(0)
#define _RYML_CB_CHECK(cb, cond)                                        \
    do                                                                  \
    {                                                                   \
        if(!(cond))                                                     \
        {                                                               \
            const char msg[] = "check failed: " #cond;                  \
            RYML_DEBUG_BREAK()                                          \
            (cb).m_error(msg, sizeof(msg), c4::yml::Location(__FILE__, 0, __LINE__, 0), (cb).m_user_data); \
        }                                                               \
    } while(0)
#ifdef RYML_USE_ASSERT
#define _RYML_CB_ASSERT(cb, cond) _RYML_CB_CHECK((cb), (cond))
#else
#define _RYML_CB_ASSERT(cb, cond) do {} while(0)
#endif
#define _RYML_CB_ALLOC_HINT(cb, T, num, hint) (T*) (cb).m_allocate((num) * sizeof(T), (hint), (cb).m_user_data)
#define _RYML_CB_ALLOC(cb, T, num) _RYML_CB_ALLOC_HINT((cb), (T), (num), nullptr)
#define _RYML_CB_FREE(cb, buf, T, num)                              \
    do {                                                            \
        (cb).m_free((buf), (num) * sizeof(T), (cb).m_user_data);    \
        (buf) = nullptr;                                            \
    } while(0)



namespace detail {
template<int8_t signedval, uint8_t unsignedval>
struct _charconstant_t
    : public std::conditional<std::is_signed<char>::value,
                              std::integral_constant<int8_t, signedval>,
                              std::integral_constant<uint8_t, unsignedval>>::type
{};
#define _RYML_CHCONST(signedval, unsignedval) ::c4::yml::detail::_charconstant_t<INT8_C(signedval), UINT8_C(unsignedval)>::value
} // namespace detail


namespace detail {
struct _SubstrWriter
{
    substr buf;
    size_t pos;
    _SubstrWriter(substr buf_, size_t pos_=0) : buf(buf_), pos(pos_) {}
    void append(csubstr s)
    {
        C4_ASSERT(!s.overlaps(buf));
        if(pos + s.len <= buf.len)
            memcpy(buf.str + pos, s.str, s.len);
        pos += s.len;
    }
    void append(char c)
    {
        if(pos < buf.len)
            buf.str[pos] = c;
        ++pos;
    }
    void append_n(char c, size_t numtimes)
    {
        if(pos + numtimes < buf.len)
            memset(buf.str + pos, c, numtimes);
        pos += numtimes;
    }
    size_t slack() const { return pos <= buf.len ? buf.len - pos : 0; }
    size_t excess() const { return pos > buf.len ? pos - buf.len : 0; }
    //! get the part written so far
    csubstr curr() const { return pos <= buf.len ? buf.first(pos) : buf; }
    //! get the part that is still free to write to (the remainder)
    substr rem() { return pos < buf.len ? buf.sub(pos) : buf.last(0); }

    size_t advance(size_t more) { pos += more; return pos; }
};
} // namespace detail

/// @endcond

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace yml
} // namespace c4

#endif /* _C4_YML_COMMON_HPP_ */
