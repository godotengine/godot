#ifndef C4_DUMP_HPP_
#define C4_DUMP_HPP_

#include "substr.hpp"

namespace c4 {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** type of the function to dump characters */
using DumperPfn = void (*)(csubstr buf);


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

template<DumperPfn dumpfn, class Arg>
inline size_t dump(substr buf, Arg const& a)
{
    size_t sz = to_chars(buf, a); // need to serialize to the buffer
    if(C4_LIKELY(sz <= buf.len))
        dumpfn(buf.first(sz));
    return sz;
}

template<class DumperFn, class Arg>
inline size_t dump(DumperFn &&dumpfn, substr buf, Arg const& a)
{
    size_t sz = to_chars(buf, a); // need to serialize to the buffer
    if(C4_LIKELY(sz <= buf.len))
        dumpfn(buf.first(sz));
    return sz;
}

template<DumperPfn dumpfn>
inline size_t dump(substr buf, csubstr a)
{
    if(buf.len)
        dumpfn(a); // dump directly, no need to serialize to the buffer
    return 0; // no space was used in the buffer
}

template<class DumperFn>
inline size_t dump(DumperFn &&dumpfn, substr buf, csubstr a)
{
    if(buf.len)
        dumpfn(a); // dump directly, no need to serialize to the buffer
    return 0; // no space was used in the buffer
}

template<DumperPfn dumpfn, size_t N>
inline size_t dump(substr buf, const char (&a)[N])
{
    if(buf.len)
        dumpfn(csubstr(a)); // dump directly, no need to serialize to the buffer
    return 0; // no space was used in the buffer
}

template<class DumperFn, size_t N>
inline size_t dump(DumperFn &&dumpfn, substr buf, const char (&a)[N])
{
    if(buf.len)
        dumpfn(csubstr(a)); // dump directly, no need to serialize to the buffer
    return 0; // no space was used in the buffer
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** */
struct DumpResults
{
    enum : size_t { noarg = (size_t)-1 };
    size_t bufsize = 0;
    size_t lastok = noarg;
    bool success_until(size_t expected) const { return lastok == noarg ? false : lastok >= expected; }
    bool write_arg(size_t arg) const { return lastok == noarg || arg > lastok; }
    size_t argfail() const { return lastok + 1; }
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/// @cond dev
// terminates the variadic recursion
template<class DumperFn>
size_t cat_dump(DumperFn &&, substr)
{
    return 0;
}

// terminates the variadic recursion
template<DumperPfn dumpfn>
size_t cat_dump(substr)
{
    return 0;
}
/// @endcond

/** take the function pointer as a function argument */
template<class DumperFn, class Arg, class... Args>
size_t cat_dump(DumperFn &&dumpfn, substr buf, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    size_t size_for_a = dump(dumpfn, buf, a);
    if(C4_UNLIKELY(size_for_a > buf.len))
        buf = buf.first(0); // ensure no more calls
    size_t size_for_more = cat_dump(dumpfn, buf, more...);
    return size_for_more > size_for_a ? size_for_more : size_for_a;
}

/** take the function pointer as a template argument */
template<DumperPfn dumpfn,class Arg, class... Args>
size_t cat_dump(substr buf, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    size_t size_for_a = dump<dumpfn>(buf, a);
    if(C4_LIKELY(size_for_a > buf.len))
        buf = buf.first(0); // ensure no more calls
    size_t size_for_more = cat_dump<dumpfn>(buf, more...);
    return size_for_more > size_for_a ? size_for_more : size_for_a;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/// @cond dev
namespace detail {

// terminates the variadic recursion
template<DumperPfn dumpfn, class Arg>
DumpResults cat_dump_resume(size_t currarg, DumpResults results, substr buf, Arg const& C4_RESTRICT a)
{
    if(C4_LIKELY(results.write_arg(currarg)))
    {
        size_t sz = dump<dumpfn>(buf, a);  // yield to the specialized function
        if(currarg == results.lastok + 1 && sz <= buf.len)
            results.lastok = currarg;
        results.bufsize = sz > results.bufsize ? sz : results.bufsize;
    }
    return results;
}

// terminates the variadic recursion
template<class DumperFn, class Arg>
DumpResults cat_dump_resume(size_t currarg, DumperFn &&dumpfn, DumpResults results, substr buf, Arg const& C4_RESTRICT a)
{
    if(C4_LIKELY(results.write_arg(currarg)))
    {
        size_t sz = dump(dumpfn, buf, a);  // yield to the specialized function
        if(currarg == results.lastok + 1 && sz <= buf.len)
            results.lastok = currarg;
        results.bufsize = sz > results.bufsize ? sz : results.bufsize;
    }
    return results;
}

template<DumperPfn dumpfn, class Arg, class... Args>
DumpResults cat_dump_resume(size_t currarg, DumpResults results, substr buf, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    results = detail::cat_dump_resume<dumpfn>(currarg, results, buf, a);
    return detail::cat_dump_resume<dumpfn>(currarg + 1u, results, buf, more...);
}

template<class DumperFn, class Arg, class... Args>
DumpResults cat_dump_resume(size_t currarg, DumperFn &&dumpfn, DumpResults results, substr buf, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    results = detail::cat_dump_resume(currarg, dumpfn, results, buf, a);
    return detail::cat_dump_resume(currarg + 1u, dumpfn, results, buf, more...);
}
} // namespace detail
/// @endcond


template<DumperPfn dumpfn, class Arg, class... Args>
C4_ALWAYS_INLINE DumpResults cat_dump_resume(DumpResults results, substr buf, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    if(results.bufsize > buf.len)
        return results;
    return detail::cat_dump_resume<dumpfn>(0u, results, buf, a, more...);
}

template<class DumperFn, class Arg, class... Args>
C4_ALWAYS_INLINE DumpResults cat_dump_resume(DumperFn &&dumpfn, DumpResults results, substr buf, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    if(results.bufsize > buf.len)
        return results;
    return detail::cat_dump_resume(0u, dumpfn, results, buf, a, more...);
}

template<DumperPfn dumpfn, class Arg, class... Args>
C4_ALWAYS_INLINE DumpResults cat_dump_resume(substr buf, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    return detail::cat_dump_resume<dumpfn>(0u, DumpResults{}, buf, a, more...);
}

template<class DumperFn, class Arg, class... Args>
C4_ALWAYS_INLINE DumpResults cat_dump_resume(DumperFn &&dumpfn, substr buf, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    return detail::cat_dump_resume(0u, dumpfn, DumpResults{}, buf, a, more...);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/// @cond dev
// terminate the recursion
template<class DumperFn, class Sep>
size_t catsep_dump(DumperFn &&, substr, Sep const& C4_RESTRICT)
{
    return 0;
}

// terminate the recursion
template<DumperPfn dumpfn, class Sep>
size_t catsep_dump(substr, Sep const& C4_RESTRICT)
{
    return 0;
}
/// @endcond

/** take the function pointer as a function argument */
template<class DumperFn, class Sep, class Arg, class... Args>
size_t catsep_dump(DumperFn &&dumpfn, substr buf, Sep const& C4_RESTRICT sep, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    size_t sz = dump(dumpfn, buf, a);
    if(C4_UNLIKELY(sz > buf.len))
        buf = buf.first(0); // ensure no more calls
    if C4_IF_CONSTEXPR (sizeof...(more) > 0)
    {
        size_t szsep = dump(dumpfn, buf, sep);
        if(C4_UNLIKELY(szsep > buf.len))
            buf = buf.first(0); // ensure no more calls
        sz = sz > szsep ? sz : szsep;
    }
    size_t size_for_more = catsep_dump(dumpfn, buf, sep, more...);
    return size_for_more > sz ? size_for_more : sz;
}

/** take the function pointer as a template argument */
template<DumperPfn dumpfn, class Sep, class Arg, class... Args>
size_t catsep_dump(substr buf, Sep const& C4_RESTRICT sep, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    size_t sz = dump<dumpfn>(buf, a);
    if(C4_UNLIKELY(sz > buf.len))
        buf = buf.first(0); // ensure no more calls
    if C4_IF_CONSTEXPR (sizeof...(more) > 0)
    {
        size_t szsep = dump<dumpfn>(buf, sep);
        if(C4_UNLIKELY(szsep > buf.len))
            buf = buf.first(0); // ensure no more calls
        sz = sz > szsep ? sz : szsep;
    }
    size_t size_for_more = catsep_dump<dumpfn>(buf, sep, more...);
    return size_for_more > sz ? size_for_more : sz;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/// @cond dev
namespace detail {
template<DumperPfn dumpfn, class Arg>
void catsep_dump_resume_(size_t currarg, DumpResults *C4_RESTRICT results, substr *C4_RESTRICT buf, Arg const& C4_RESTRICT a)
{
    if(C4_LIKELY(results->write_arg(currarg)))
    {
        size_t sz = dump<dumpfn>(*buf, a);
        results->bufsize = sz > results->bufsize ? sz : results->bufsize;
        if(C4_LIKELY(sz <= buf->len))
            results->lastok = currarg;
        else
            buf->len = 0;
    }
}

template<class DumperFn, class Arg>
void catsep_dump_resume_(size_t currarg, DumperFn &&dumpfn, DumpResults *C4_RESTRICT results, substr *C4_RESTRICT buf, Arg const& C4_RESTRICT a)
{
    if(C4_LIKELY(results->write_arg(currarg)))
    {
        size_t sz = dump(dumpfn, *buf, a);
        results->bufsize = sz > results->bufsize ? sz : results->bufsize;
        if(C4_LIKELY(sz <= buf->len))
            results->lastok = currarg;
        else
            buf->len = 0;
    }
}

template<DumperPfn dumpfn, class Sep, class Arg>
C4_ALWAYS_INLINE void catsep_dump_resume(size_t currarg, DumpResults *C4_RESTRICT results, substr *C4_RESTRICT buf, Sep const& C4_RESTRICT, Arg const& C4_RESTRICT a)
{
    detail::catsep_dump_resume_<dumpfn>(currarg, results, buf, a);
}

template<class DumperFn, class Sep, class Arg>
C4_ALWAYS_INLINE void catsep_dump_resume(size_t currarg, DumperFn &&dumpfn, DumpResults *C4_RESTRICT results, substr *C4_RESTRICT buf, Sep const& C4_RESTRICT, Arg const& C4_RESTRICT a)
{
    detail::catsep_dump_resume_(currarg, dumpfn, results, buf, a);
}

template<DumperPfn dumpfn, class Sep, class Arg, class... Args>
C4_ALWAYS_INLINE void catsep_dump_resume(size_t currarg, DumpResults *C4_RESTRICT results, substr *C4_RESTRICT buf, Sep const& C4_RESTRICT sep, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    detail::catsep_dump_resume_<dumpfn>(currarg     , results, buf, a);
    detail::catsep_dump_resume_<dumpfn>(currarg + 1u, results, buf, sep);
    detail::catsep_dump_resume <dumpfn>(currarg + 2u, results, buf, sep, more...);
}

template<class DumperFn, class Sep, class Arg, class... Args>
C4_ALWAYS_INLINE void catsep_dump_resume(size_t currarg, DumperFn &&dumpfn, DumpResults *C4_RESTRICT results, substr *C4_RESTRICT buf, Sep const& C4_RESTRICT sep, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    detail::catsep_dump_resume_(currarg     , dumpfn, results, buf, a);
    detail::catsep_dump_resume_(currarg + 1u, dumpfn, results, buf, sep);
    detail::catsep_dump_resume (currarg + 2u, dumpfn, results, buf, sep, more...);
}
} // namespace detail
/// @endcond


template<DumperPfn dumpfn, class Sep, class... Args>
C4_ALWAYS_INLINE DumpResults catsep_dump_resume(DumpResults results, substr buf, Sep const& C4_RESTRICT sep, Args const& C4_RESTRICT ...more)
{
    detail::catsep_dump_resume<dumpfn>(0u, &results, &buf, sep, more...);
    return results;
}

template<class DumperFn, class Sep, class... Args>
C4_ALWAYS_INLINE DumpResults catsep_dump_resume(DumperFn &&dumpfn, DumpResults results, substr buf, Sep const& C4_RESTRICT sep, Args const& C4_RESTRICT ...more)
{
    detail::catsep_dump_resume(0u, dumpfn, &results, &buf, sep, more...);
    return results;
}

template<DumperPfn dumpfn, class Sep, class... Args>
C4_ALWAYS_INLINE DumpResults catsep_dump_resume(substr buf, Sep const& C4_RESTRICT sep, Args const& C4_RESTRICT ...more)
{
    DumpResults results;
    detail::catsep_dump_resume<dumpfn>(0u, &results, &buf, sep, more...);
    return results;
}

template<class DumperFn, class Sep, class... Args>
C4_ALWAYS_INLINE DumpResults catsep_dump_resume(DumperFn &&dumpfn, substr buf, Sep const& C4_RESTRICT sep, Args const& C4_RESTRICT ...more)
{
    DumpResults results;
    detail::catsep_dump_resume(0u, dumpfn, &results, &buf, sep, more...);
    return results;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** take the function pointer as a function argument */
template<class DumperFn>
C4_ALWAYS_INLINE size_t format_dump(DumperFn &&dumpfn, substr buf, csubstr fmt)
{
    // we can dump without using buf
    // but we'll only dump if the buffer is ok
    if(C4_LIKELY(buf.len > 0 && fmt.len))
        dumpfn(fmt);
    return 0u;
}

/** take the function pointer as a function argument */
template<DumperPfn dumpfn>
C4_ALWAYS_INLINE size_t format_dump(substr buf, csubstr fmt)
{
    // we can dump without using buf
    // but we'll only dump if the buffer is ok
    if(C4_LIKELY(buf.len > 0 && fmt.len > 0))
        dumpfn(fmt);
    return 0u;
}

/** take the function pointer as a function argument */
template<class DumperFn, class Arg, class... Args>
size_t format_dump(DumperFn &&dumpfn, substr buf, csubstr fmt, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    // we can dump without using buf
    // but we'll only dump if the buffer is ok
    size_t pos = fmt.find("{}"); // @todo use _find_fmt()
    if(C4_UNLIKELY(pos == csubstr::npos))
    {
        if(C4_LIKELY(buf.len > 0 && fmt.len > 0))
            dumpfn(fmt);
        return 0u;
    }
    if(C4_LIKELY(buf.len > 0 && pos > 0))
        dumpfn(fmt.first(pos)); // we can dump without using buf
    fmt = fmt.sub(pos + 2); // skip {} do this before assigning to pos again
    pos = dump(dumpfn, buf, a);
    if(C4_UNLIKELY(pos > buf.len))
        buf.len = 0; // ensure no more calls to dump
    size_t size_for_more = format_dump(dumpfn, buf, fmt, more...);
    return size_for_more > pos ? size_for_more : pos;
}

/** take the function pointer as a template argument */
template<DumperPfn dumpfn, class Arg, class... Args>
size_t format_dump(substr buf, csubstr fmt, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    // we can dump without using buf
    // but we'll only dump if the buffer is ok
    size_t pos = fmt.find("{}"); // @todo use _find_fmt()
    if(C4_UNLIKELY(pos == csubstr::npos))
    {
        if(C4_LIKELY(buf.len > 0 && fmt.len > 0))
            dumpfn(fmt);
        return 0u;
    }
    if(C4_LIKELY(buf.len > 0 && pos > 0))
        dumpfn(fmt.first(pos)); // we can dump without using buf
    fmt = fmt.sub(pos + 2); // skip {} do this before assigning to pos again
    pos = dump<dumpfn>(buf, a);
    if(C4_UNLIKELY(pos > buf.len))
        buf.len = 0; // ensure no more calls to dump
    size_t size_for_more = format_dump<dumpfn>(buf, fmt, more...);
    return size_for_more > pos ? size_for_more : pos;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/// @cond dev
namespace detail {

template<DumperPfn dumpfn>
DumpResults format_dump_resume(size_t currarg, DumpResults results, substr buf, csubstr fmt)
{
    // we can dump without using buf
    // but we'll only dump if the buffer is ok
    if(C4_LIKELY(buf.len > 0))
    {
        dumpfn(fmt);
        results.lastok = currarg;
    }
    return results;
}

template<class DumperFn>
DumpResults format_dump_resume(size_t currarg, DumperFn &&dumpfn, DumpResults results, substr buf, csubstr fmt)
{
    // we can dump without using buf
    // but we'll only dump if the buffer is ok
    if(C4_LIKELY(buf.len > 0))
    {
        dumpfn(fmt);
        results.lastok = currarg;
    }
    return results;
}

template<DumperPfn dumpfn, class Arg, class... Args>
DumpResults format_dump_resume(size_t currarg, DumpResults results, substr buf, csubstr fmt, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    // we need to process the format even if we're not
    // going to print the first arguments because we're resuming
    size_t pos = fmt.find("{}"); // @todo use _find_fmt()
    // we can dump without using buf
    // but we'll only dump if the buffer is ok
    if(C4_LIKELY(results.write_arg(currarg)))
    {
        if(C4_UNLIKELY(pos == csubstr::npos))
        {
            if(C4_LIKELY(buf.len > 0))
            {
                results.lastok = currarg;
                dumpfn(fmt);
            }
            return results;
        }
        if(C4_LIKELY(buf.len > 0))
        {
            results.lastok = currarg;
            dumpfn(fmt.first(pos));
        }
    }
    fmt = fmt.sub(pos + 2);
    if(C4_LIKELY(results.write_arg(currarg + 1)))
    {
        pos = dump<dumpfn>(buf, a);
        results.bufsize = pos > results.bufsize ? pos : results.bufsize;
        if(C4_LIKELY(pos <= buf.len))
            results.lastok = currarg + 1;
        else
            buf.len = 0;
    }
    return detail::format_dump_resume<dumpfn>(currarg + 2u, results, buf, fmt, more...);
}
/// @endcond


template<class DumperFn, class Arg, class... Args>
DumpResults format_dump_resume(size_t currarg, DumperFn &&dumpfn, DumpResults results, substr buf, csubstr fmt, Arg const& C4_RESTRICT a, Args const& C4_RESTRICT ...more)
{
    // we need to process the format even if we're not
    // going to print the first arguments because we're resuming
    size_t pos = fmt.find("{}"); // @todo use _find_fmt()
    // we can dump without using buf
    // but we'll only dump if the buffer is ok
    if(C4_LIKELY(results.write_arg(currarg)))
    {
        if(C4_UNLIKELY(pos == csubstr::npos))
        {
            if(C4_LIKELY(buf.len > 0))
            {
                results.lastok = currarg;
                dumpfn(fmt);
            }
            return results;
        }
        if(C4_LIKELY(buf.len > 0))
        {
            results.lastok = currarg;
            dumpfn(fmt.first(pos));
        }
    }
    fmt = fmt.sub(pos + 2);
    if(C4_LIKELY(results.write_arg(currarg + 1)))
    {
        pos = dump(dumpfn, buf, a);
        results.bufsize = pos > results.bufsize ? pos : results.bufsize;
        if(C4_LIKELY(pos <= buf.len))
            results.lastok = currarg + 1;
        else
            buf.len = 0;
    }
    return detail::format_dump_resume(currarg + 2u, dumpfn, results, buf, fmt, more...);
}
} // namespace detail


template<DumperPfn dumpfn, class... Args>
C4_ALWAYS_INLINE DumpResults format_dump_resume(DumpResults results, substr buf, csubstr fmt, Args const& C4_RESTRICT ...more)
{
    return detail::format_dump_resume<dumpfn>(0u, results, buf, fmt, more...);
}

template<class DumperFn, class... Args>
C4_ALWAYS_INLINE DumpResults format_dump_resume(DumperFn &&dumpfn, DumpResults results, substr buf, csubstr fmt, Args const& C4_RESTRICT ...more)
{
    return detail::format_dump_resume(0u, dumpfn, results, buf, fmt, more...);
}


template<DumperPfn dumpfn, class... Args>
C4_ALWAYS_INLINE DumpResults format_dump_resume(substr buf, csubstr fmt, Args const& C4_RESTRICT ...more)
{
    return detail::format_dump_resume<dumpfn>(0u, DumpResults{}, buf, fmt, more...);
}

template<class DumperFn, class... Args>
C4_ALWAYS_INLINE DumpResults format_dump_resume(DumperFn &&dumpfn, substr buf, csubstr fmt, Args const& C4_RESTRICT ...more)
{
    return detail::format_dump_resume(0u, dumpfn, DumpResults{}, buf, fmt, more...);
}

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace c4


#endif /* C4_DUMP_HPP_ */
