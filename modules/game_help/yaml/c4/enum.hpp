#ifndef _C4_ENUM_HPP_
#define _C4_ENUM_HPP_

#include "error.hpp"
#include <string.h>

/** @file enum.hpp utilities for enums: convert to/from string
 */


namespace c4 {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

//! taken from http://stackoverflow.com/questions/15586163/c11-type-trait-to-differentiate-between-enum-class-and-regular-enum
template<typename Enum>
using is_scoped_enum = std::integral_constant<bool, std::is_enum<Enum>::value && !std::is_convertible<Enum, int>::value>;


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

typedef enum {
    EOFFS_NONE = 0,  ///< no offset
    EOFFS_CLS = 1,   ///< get the enum offset for the class name. @see eoffs_cls()
    EOFFS_PFX = 2,   ///< get the enum offset for the enum prefix. @see eoffs_pfx()
    _EOFFS_LAST      ///< reserved
} EnumOffsetType;


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** A simple (proxy) container for the value-name pairs of an enum type.
 * Uses linear search for finds; this could be improved for time-critical
 * code. */
template<class Enum>
class EnumSymbols
{
public:

    struct Sym
    {
        Enum value;
        const char *name;

        bool cmp(const char *s) const;
        bool cmp(const char *s, size_t len) const;

        const char *name_offs(EnumOffsetType t) const;
    };

    using const_iterator = Sym const*;

public:

    template<size_t N>
    EnumSymbols(Sym const (&p)[N]) : m_symbols(p), m_num(N) {}

    size_t size() const { return m_num; }
    bool empty() const { return m_num == 0; }

    Sym const* get(Enum v) const { auto p = find(v); C4_CHECK_MSG(p != nullptr, "could not find symbol=%zd", (std::ptrdiff_t)v); return p; }
    Sym const* get(const char *s) const { auto p = find(s); C4_CHECK_MSG(p != nullptr, "could not find symbol \"%s\"", s); return p; }
    Sym const* get(const char *s, size_t len) const { auto p = find(s, len); C4_CHECK_MSG(p != nullptr, "could not find symbol \"%.*s\"", len, s); return p; }

    Sym const* find(Enum v) const;
    Sym const* find(const char *s) const;
    Sym const* find(const char *s, size_t len) const;

    Sym const& operator[] (size_t i) const { C4_CHECK(i < m_num); return m_symbols[i]; }

    Sym const* begin() const { return m_symbols; }
    Sym const* end  () const { return m_symbols + m_num; }

private:

    Sym const* m_symbols;
    size_t const m_num;

};

//-----------------------------------------------------------------------------
/** return an EnumSymbols object for the enum type T
 *
 * @warning SPECIALIZE! This needs to be specialized for each enum
 * type. Failure to provide a specialization will cause a linker
 * error. */
template<class Enum>
EnumSymbols<Enum> const esyms();


/** return the offset for an enum symbol class. For example,
 * eoffs_cls<MyEnumClass>() would be 13=strlen("MyEnumClass::").
 *
 * With this function you can announce that the full prefix (including
 * an eventual enclosing class or C++11 enum class) is of a certain
 * length.
 *
 * @warning Needs to be specialized for each enum class type that
 * wants to use this. When no specialization is given, will return
 * 0. */
template<class Enum>
size_t eoffs_cls()
{
    return 0;
}


/** return the offset for an enum symbol prefix. This includes
 * eoffs_cls().  With this function you can announce that the full
 * prefix (including an eventual enclosing class or C++11 enum class
 * plus the string prefix) is of a certain length.
 *
 * @warning Needs to be specialized for each enum class type that
 * wants to use this. When no specialization is given, will return
 * 0. */
template<class Enum>
size_t eoffs_pfx()
{
    return 0;
}


template<class Enum>
size_t eoffs(EnumOffsetType which)
{
    switch(which)
    {
    case EOFFS_NONE:
        return 0;
    case EOFFS_CLS:
        return eoffs_cls<Enum>();
    case EOFFS_PFX:
    {
        size_t pfx = eoffs_pfx<Enum>();
        return pfx > 0 ? pfx : eoffs_cls<Enum>();
    }
    default:
        C4_ERROR("unknown offset type %d", (int)which);
        return 0;
    }
}


//-----------------------------------------------------------------------------
/** get the enum value corresponding to a c-string */

#ifdef __clang__
#   pragma clang diagnostic push
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   if __GNUC__ >= 6
#       pragma GCC diagnostic ignored "-Wnull-dereference"
#   endif
#endif

template<class Enum>
Enum str2e(const char* str)
{
    auto pairs = esyms<Enum>();
    auto *p = pairs.get(str);
    C4_CHECK_MSG(p != nullptr, "no valid enum pair name for '%s'", str);
    return p->value;
}

/** get the c-string corresponding to an enum value */
template<class Enum>
const char* e2str(Enum e)
{
    auto es = esyms<Enum>();
    auto *p = es.get(e);
    C4_CHECK_MSG(p != nullptr, "no valid enum pair name");
    return p->name;
}

/** like e2str(), but add an offset. */
template<class Enum>
const char* e2stroffs(Enum e, EnumOffsetType ot=EOFFS_PFX)
{
    const char *s = e2str<Enum>(e) + eoffs<Enum>(ot);
    return s;
}

#ifdef __clang__
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#endif

//-----------------------------------------------------------------------------
/** Find a symbol by value. Returns nullptr when none is found */
template<class Enum>
typename EnumSymbols<Enum>::Sym const* EnumSymbols<Enum>::find(Enum v) const
{
    for(Sym const* p = this->m_symbols, *e = p+this->m_num; p < e; ++p)
        if(p->value == v)
            return p;
    return nullptr;
}

/** Find a symbol by name. Returns nullptr when none is found */
template<class Enum>
typename EnumSymbols<Enum>::Sym const* EnumSymbols<Enum>::find(const char *s) const
{
    for(Sym const* p = this->m_symbols, *e = p+this->m_num; p < e; ++p)
        if(p->cmp(s))
            return p;
    return nullptr;
}

/** Find a symbol by name. Returns nullptr when none is found */
template<class Enum>
typename EnumSymbols<Enum>::Sym const* EnumSymbols<Enum>::find(const char *s, size_t len) const
{
    for(Sym const* p = this->m_symbols, *e = p+this->m_num; p < e; ++p)
        if(p->cmp(s, len))
            return p;
    return nullptr;
}

//-----------------------------------------------------------------------------
template<class Enum>
bool EnumSymbols<Enum>::Sym::cmp(const char *s) const
{
    if(strcmp(name, s) == 0)
        return true;

    for(int i = 1; i < _EOFFS_LAST; ++i)
    {
        auto o = eoffs<Enum>((EnumOffsetType)i);
        if(o > 0)
            if(strcmp(name + o, s) == 0)
                return true;
    }

    return false;
}

template<class Enum>
bool EnumSymbols<Enum>::Sym::cmp(const char *s, size_t len) const
{
    if(strncmp(name, s, len) == 0)
        return true;

    size_t nlen = 0;
    for(int i = 1; i <_EOFFS_LAST; ++i)
    {
        auto o = eoffs<Enum>((EnumOffsetType)i);
        if(o > 0)
        {
            if(!nlen)
            {
                nlen = strlen(name);
            }
            C4_ASSERT(o < nlen);
            size_t rem = nlen - o;
            auto m = len > rem ? len : rem;
            if(len >= m && strncmp(name + o, s, m) == 0)
                return true;
        }
    }

    return false;
}

//-----------------------------------------------------------------------------
template<class Enum>
const char* EnumSymbols<Enum>::Sym::name_offs(EnumOffsetType t) const
{
    C4_ASSERT(eoffs<Enum>(t) < strlen(name));
    return name + eoffs<Enum>(t);
}

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace c4

#endif // _C4_ENUM_HPP_
