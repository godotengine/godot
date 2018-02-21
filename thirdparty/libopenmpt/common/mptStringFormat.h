/*
 * mptStringFormat.h
 * -----------------
 * Purpose: Convert other types to strings.
 * Notes  : Currently none.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


#include "mptString.h"


OPENMPT_NAMESPACE_BEGIN



// The following section demands a rationale.
//  1. mpt::fmt::val(), mpt::wfmt::val() and mpt::ufmt::val() mimic the semantics of c++11 std::to_string() and std::to_wstring().
//     There is an important difference though. The c++11 versions are specified in terms of sprintf formatting which in turn
//     depends on the current C locale. This renders these functions unusable in a library context because the current
//     C locale is set by the library-using application and could be anything. There is no way a library can get reliable semantics
//     out of these functions. It is thus better to just avoid them.
//     ToString() and ToWString() are based on iostream internally, but the the locale of the stream is forced to std::locale::classic(),
//     which results in "C" ASCII locale behavior.
//  2. The full suite of printf-like or iostream like number formatting is generally not required. Instead, a sane subset functionality
//     is provided here.
//     For convenience, mpt::fmt::f(const char *, float) allows formatting a single floating point value with a
//     standard printf-like format string. This itself relies on iostream with classic() locale internally and is thus current locale
//     agnostic.
//     When formatting integers, it is recommended to use mpt::fmt::dec or mpt::fmt::hex. Appending a template argument '<n>' sets the width,
//     the same way as '%nd' would do. Appending a '0' to the function name causes zero-filling as print-like '%0nd' would do. Spelling 'HEX'
//     in upper-case generates upper-case hex digits. If these are not known at compile-time, a more verbose FormatVal(int, format) can be
//     used.
//  3. mpt::format(format)(...) provides simplified and type-safe message and localization string formatting.
//     The only specifier allowed is '%' followed by a single digit n. It references to n-th parameter after the format string (1-based).
//     This mimics the behaviour of QString::arg() in QT4/5 or MFC AfxFormatString2(). C printf-like functions offer similar functionality
//     with a '%n$TYPE' syntax. In .NET, the syntax is '{n}'. This is useful to support localization strings that can change the parameter
//     ordering.
//     There are macro verions (MPT_FORMAT and variants) which properly use wide string literals for the format parameter.
//  4. Every function is available for std::string, std::wstring and mpt::ustring. std::string makes no assumption about the encoding, which
//     basically means, it should work for any 7-bit or 8-bit encoding, including for example ASCII, UTF8 or the current locale encoding.
//     std::string        std::wstring       mpt::ustring                           CString
//     mpt::fmt::val      mpt::wfmt::val     mpt::ufmt::val                         mpt::tfmt::val
//     mpt::FormatVal     mpt::FormatValW    mpt::FormatValTFunctor<mpt::ustring>() mpt::FormatValTFunctor<Cstring>()
//     mpt::fmt           mpt::wfmt          mpt::ufmt                              mpt::tfmt
//     mpt::format        mpt::format        mpt::format                            mpt::format
//  5. All functionality here delegates real work outside of the header file so that <sstream> and <locale> do not need to be included when
//     using this functionality.
//     Advantages:
//      - Avoids binary code bloat when too much of iostream operator << gets inlined at every usage site.
//      - Faster compile times because <sstream> and <locale> (2 very complex headers) are not included everywhere.
//     Disadvantages:
//      - Slightly more c++ code is required for delegating work.
//      - As the header does not use iostreams, custom types need to overload mpt::String, mpt::ToWstring and mpt::UString instead of
//        iostream operator << to allow for custom type formatting.
//      - std::string, std::wstring and mpt::ustring are returned from somewhat deep cascades of helper functions. Where possible, code is
//        written in such a way that return-value-optimization (RVO) or named-return-value-optimization (NRVO) should be able to eliminate
//        almost all these copies. This should not be a problem for any decent modern compiler (and even less so for a c++11 compiler where
//        move-semantics will kick in if RVO/NRVO fails).

namespace mpt
{

// ToString() converts various built-in types to a well-defined, locale-independent string representation.
// This is also used as a type-tunnel pattern for mpt::format.
// Custom types that need to be converted to strings are encouraged to overload ToString() and ToWString().

// fallback to member function ToString()
template <typename T> auto ToString(const T & x) -> decltype(x.ToString()) { return x.ToString(); }

static inline std::string ToString(const std::string & x) { return x; }
static inline std::string ToString(const char * const & x) { return x; }
MPT_DEPRECATED static inline std::string ToString(const char & x) { return std::string(1, x); } // deprecated to catch potential API mis-use, use std::string(1, x) instead
#if MPT_WSTRING_FORMAT
MPT_DEPRECATED std::string ToString(const std::wstring & x); // Unknown encoding.
MPT_DEPRECATED std::string ToString(const wchar_t * const & x); // Unknown encoding.
MPT_DEPRECATED std::string ToString(const wchar_t & x); // deprecated to catch potential API mis-use, use std::wstring(1, x) instead
#endif
#if MPT_USTRING_MODE_UTF8
MPT_DEPRECATED std::string ToString(const mpt::ustring & x); // Unknown encoding.
#endif
#if defined(_MFC_VER)
MPT_DEPRECATED std::string ToString(const mpt::ustring & x); // Unknown encoding.
#endif
#if defined(_MFC_VER)
MPT_DEPRECATED std::string ToString(const CString & x);
#endif
std::string ToString(const bool & x);
std::string ToString(const signed char & x);
std::string ToString(const unsigned char & x);
std::string ToString(const signed short & x);
std::string ToString(const unsigned short & x);
std::string ToString(const signed int & x);
std::string ToString(const unsigned int & x);
std::string ToString(const signed long & x);
std::string ToString(const unsigned long & x);
std::string ToString(const signed long long & x);
std::string ToString(const unsigned long long & x);
std::string ToString(const float & x);
std::string ToString(const double & x);
std::string ToString(const long double & x);

// fallback to member function ToUString()
template <typename T> auto ToUString(const T & x) -> decltype(x.ToUString()) { return x.ToUString(); }

static inline mpt::ustring ToUString(const mpt::ustring & x) { return x; }
MPT_DEPRECATED mpt::ustring ToUString(const std::string & x); // Unknown encoding.
MPT_DEPRECATED mpt::ustring ToUString(const char * const & x); // Unknown encoding. Note that this also applies to TCHAR in !UNICODE builds as the type is indistinguishable from char. Wrap with CString or FromTcharStr in this case.
MPT_DEPRECATED mpt::ustring ToUString(const char & x); // deprecated to catch potential API mis-use, use std::string(1, x) instead
#if MPT_WSTRING_FORMAT
#if MPT_USTRING_MODE_UTF8
mpt::ustring ToUString(const std::wstring & x);
#endif
mpt::ustring ToUString(const wchar_t * const & x);
MPT_DEPRECATED mpt::ustring ToUString(const wchar_t & x); // deprecated to catch potential API mis-use, use std::wstring(1, x) instead
#endif
#if defined(_MFC_VER)
mpt::ustring ToUString(const CString & x);
#endif
mpt::ustring ToUString(const bool & x);
mpt::ustring ToUString(const signed char & x);
mpt::ustring ToUString(const unsigned char & x);
mpt::ustring ToUString(const signed short & x);
mpt::ustring ToUString(const unsigned short & x);
mpt::ustring ToUString(const signed int & x);
mpt::ustring ToUString(const unsigned int & x);
mpt::ustring ToUString(const signed long & x);
mpt::ustring ToUString(const unsigned long & x);
mpt::ustring ToUString(const signed long long & x);
mpt::ustring ToUString(const unsigned long long & x);
mpt::ustring ToUString(const float & x);
mpt::ustring ToUString(const double & x);
mpt::ustring ToUString(const long double & x);

#if MPT_WSTRING_FORMAT
MPT_DEPRECATED std::wstring ToWString(const std::string & x); // Unknown encoding.
MPT_DEPRECATED std::wstring ToWString(const char * const & x); // Unknown encoding. Note that this also applies to TCHAR in !UNICODE builds as the type is indistinguishable from char. Wrap with CString or FromTcharStr in this case.
MPT_DEPRECATED std::wstring ToWString(const char & x); // deprecated to catch potential API mis-use, use std::string(1, x) instead
static inline std::wstring ToWString(const std::wstring & x) { return x; }
static inline std::wstring ToWString(const wchar_t * const & x) { return x; }
MPT_DEPRECATED static inline std::wstring ToWString(const wchar_t & x) { return std::wstring(1, x); } // deprecated to catch potential API mis-use, use std::wstring(1, x) instead
#if MPT_USTRING_MODE_UTF8
std::wstring ToWString(const mpt::ustring & x);
#endif
#if defined(_MFC_VER)
std::wstring ToWString(const CString & x);
#endif
std::wstring ToWString(const bool & x);
std::wstring ToWString(const signed char & x);
std::wstring ToWString(const unsigned char & x);
std::wstring ToWString(const signed short & x);
std::wstring ToWString(const unsigned short & x);
std::wstring ToWString(const signed int & x);
std::wstring ToWString(const unsigned int & x);
std::wstring ToWString(const signed long & x);
std::wstring ToWString(const unsigned long & x);
std::wstring ToWString(const signed long long & x);
std::wstring ToWString(const unsigned long long & x);
std::wstring ToWString(const float & x);
std::wstring ToWString(const double & x);
std::wstring ToWString(const long double & x);
#endif

#if defined(_MFC_VER)
#ifdef UNICODE
#if MPT_WSTRING_FORMAT
template <typename T> static inline CString ToCStringHelper(const T & x) { return mpt::ToCString(ToWString(x)); }
#else
template <typename T> static inline CString ToCStringHelper(const T & x) { return mpt::ToCString(ToUString(x)); }
#endif
#else
namespace detail {
template <typename T> struct CstringToStdStringImpl { CString operator () (const T & v) { return mpt::ToCString(mpt::CharsetLocale, ToString(v)); } };
template <> struct CstringToStdStringImpl<CString> { CString operator () (const CString & v) { return v; } };
}
template <typename T> static inline CString ToCStringHelper(const T & x) { return mpt::detail::CstringToStdStringImpl<T>()(x); }
#endif
#endif

template <typename Tstring> struct ToStringTFunctor {};
template <> struct ToStringTFunctor<std::string> { template <typename T> inline std::string operator() (const T & x) { return ToString(x); } };
template <> struct ToStringTFunctor<mpt::ustring> { template <typename T> inline mpt::ustring operator() (const T & x) { return ToUString(x); } };
#if MPT_WSTRING_FORMAT && MPT_USTRING_MODE_UTF8
template <> struct ToStringTFunctor<std::wstring> { template <typename T> inline std::wstring operator() (const T & x) { return ToWString(x); } };
#endif
#if defined(_MFC_VER)
template <> struct ToStringTFunctor<CString> { template <typename T> inline CString operator() (const T & x) { return mpt::ToCStringHelper(x); } };
#endif

template<typename Tstring, typename T> inline Tstring ToStringT(const T & x) { return ToStringTFunctor<Tstring>()(x); }


struct fmt_base
{

enum FormatFlagsEnum
{
	BaseDec = 0x0001, // base 10 (integers only)
	BaseHex = 0x0002, // base 16 (integers only)
	CaseLow = 0x0010, // lower case hex digits
	CaseUpp = 0x0020, // upper case hex digits
	FillOff = 0x0100, // do not fill up width
	FillSpc = 0x0200, // fill up width with spaces
	FillNul = 0x0400, // fill up width with zeros
	NotaNrm = 0x1000, // float: normal/default notation
	NotaFix = 0x2000, // float: fixed point notation
	NotaSci = 0x4000, // float: scientific notation
};

}; // struct fmt_base

typedef unsigned int FormatFlags;

STATIC_ASSERT(sizeof(FormatFlags) >= sizeof(fmt_base::FormatFlagsEnum));

class FormatSpec;

MPT_DEPRECATED std::string FormatVal(const char & x, const FormatSpec & f); // deprecated to catch potential API mis-use, use std::string(1, x) instead
MPT_DEPRECATED std::string FormatVal(const wchar_t & x, const FormatSpec & f); // deprecated to catch potential API mis-use, use std::wstring(1, x) instead
std::string FormatVal(const bool & x, const FormatSpec & f);
std::string FormatVal(const signed char & x, const FormatSpec & f);
std::string FormatVal(const unsigned char & x, const FormatSpec & f);
std::string FormatVal(const signed short & x, const FormatSpec & f);
std::string FormatVal(const unsigned short & x, const FormatSpec & f);
std::string FormatVal(const signed int & x, const FormatSpec & f);
std::string FormatVal(const unsigned int & x, const FormatSpec & f);
std::string FormatVal(const signed long & x, const FormatSpec & f);
std::string FormatVal(const unsigned long & x, const FormatSpec & f);
std::string FormatVal(const signed long long & x, const FormatSpec & f);
std::string FormatVal(const unsigned long long & x, const FormatSpec & f);
std::string FormatVal(const float & x, const FormatSpec & f);
std::string FormatVal(const double & x, const FormatSpec & f);
std::string FormatVal(const long double & x, const FormatSpec & f);

#if MPT_WSTRING_FORMAT
MPT_DEPRECATED std::wstring FormatValW(const char & x, const FormatSpec & f); // deprecated to catch potential API mis-use, use std::string(1, x) instead
MPT_DEPRECATED std::wstring FormatValW(const wchar_t & x, const FormatSpec & f); // deprecated to catch potential API mis-use, use std::wstring(1, x) instead
std::wstring FormatValW(const bool & x, const FormatSpec & f);
std::wstring FormatValW(const signed char & x, const FormatSpec & f);
std::wstring FormatValW(const unsigned char & x, const FormatSpec & f);
std::wstring FormatValW(const signed short & x, const FormatSpec & f);
std::wstring FormatValW(const unsigned short & x, const FormatSpec & f);
std::wstring FormatValW(const signed int & x, const FormatSpec & f);
std::wstring FormatValW(const unsigned int & x, const FormatSpec & f);
std::wstring FormatValW(const signed long & x, const FormatSpec & f);
std::wstring FormatValW(const unsigned long & x, const FormatSpec & f);
std::wstring FormatValW(const signed long long & x, const FormatSpec & f);
std::wstring FormatValW(const unsigned long long & x, const FormatSpec & f);
std::wstring FormatValW(const float & x, const FormatSpec & f);
std::wstring FormatValW(const double & x, const FormatSpec & f);
std::wstring FormatValW(const long double & x, const FormatSpec & f);
#endif

template <typename Tstring> struct FormatValTFunctor {};
template <> struct FormatValTFunctor<std::string> { template <typename T> inline std::string operator() (const T & x, const FormatSpec & f) { return FormatVal(x, f); } };
#if MPT_WSTRING_FORMAT
template <> struct FormatValTFunctor<std::wstring> { template <typename T> inline std::wstring operator() (const T & x, const FormatSpec & f) { return FormatValW(x, f); } };
#endif
#if MPT_USTRING_MODE_UTF8
template <> struct FormatValTFunctor<mpt::ustring> { template <typename T> inline mpt::ustring operator() (const T & x, const FormatSpec & f) { return mpt::ToUnicode(mpt::CharsetUTF8, FormatVal(x, f)); } };
#endif
#if defined(_MFC_VER)
#ifdef UNICODE
template <> struct FormatValTFunctor<CString> { template <typename T> inline CString operator() (const T & x, const FormatSpec & f) { return mpt::ToCString(FormatValW(x, f)); } };
#else
template <> struct FormatValTFunctor<CString> { template <typename T> inline CString operator() (const T & x, const FormatSpec & f) { return mpt::ToCString(mpt::CharsetLocale, FormatVal(x, f)); } };
#endif
#endif


class FormatSpec
{
private:
	FormatFlags flags;
	std::size_t width;
	int precision;
public:
	FormatSpec() : flags(0), width(0), precision(-1) {}
	FormatFlags GetFlags() const { return flags; }
	std::size_t GetWidth() const { return width; }
	int GetPrecision() const { return precision; }
	FormatSpec & SetFlags(FormatFlags f) { flags = f; return *this; }
	FormatSpec & SetWidth(std::size_t w) { width = w; return *this; }
	FormatSpec & SetPrecision(int p) { precision = p; return *this; }
public:
	// short-hand construction
	explicit FormatSpec(FormatFlags f, std::size_t w = 0, int p = -1) : flags(f), width(w), precision(p) {}
	explicit FormatSpec(const char * format) : flags(0), width(0), precision(-1) { ParsePrintf(format); }
	explicit FormatSpec(const wchar_t * format) : flags(0), width(0), precision(-1) { ParsePrintf(format); }
	explicit FormatSpec(const std::string & format) : flags(0), width(0), precision(-1) { ParsePrintf(format); }
	explicit FormatSpec(const std::wstring & format) : flags(0), width(0), precision(-1) { ParsePrintf(format); }
public:
	// only for floating point formats
	FormatSpec & ParsePrintf(const char * format);
	FormatSpec & ParsePrintf(const wchar_t * format);
	FormatSpec & ParsePrintf(const std::string & format);
	FormatSpec & ParsePrintf(const std::wstring & format);
public:
	FormatSpec & BaseDec() { flags &= ~(fmt_base::BaseDec|fmt_base::BaseHex); flags |= fmt_base::BaseDec; return *this; }
	FormatSpec & BaseHex() { flags &= ~(fmt_base::BaseDec|fmt_base::BaseHex); flags |= fmt_base::BaseHex; return *this; }
	FormatSpec & CaseLow() { flags &= ~(fmt_base::CaseLow|fmt_base::CaseUpp); flags |= fmt_base::CaseLow; return *this; }
	FormatSpec & CaseUpp() { flags &= ~(fmt_base::CaseLow|fmt_base::CaseUpp); flags |= fmt_base::CaseUpp; return *this; }
	FormatSpec & FillOff() { flags &= ~(fmt_base::FillOff|fmt_base::FillSpc|fmt_base::FillNul); flags |= fmt_base::FillOff; return *this; }
	FormatSpec & FillSpc() { flags &= ~(fmt_base::FillOff|fmt_base::FillSpc|fmt_base::FillNul); flags |= fmt_base::FillSpc; return *this; }
	FormatSpec & FillNul() { flags &= ~(fmt_base::FillOff|fmt_base::FillSpc|fmt_base::FillNul); flags |= fmt_base::FillNul; return *this; }
	FormatSpec & NotaNrm() { flags &= ~(fmt_base::NotaNrm|fmt_base::NotaFix|fmt_base::NotaSci); flags |= fmt_base::NotaNrm; return *this; }
	FormatSpec & NotaFix() { flags &= ~(fmt_base::NotaNrm|fmt_base::NotaFix|fmt_base::NotaSci); flags |= fmt_base::NotaFix; return *this; }
	FormatSpec & NotaSci() { flags &= ~(fmt_base::NotaNrm|fmt_base::NotaFix|fmt_base::NotaSci); flags |= fmt_base::NotaSci; return *this; }
	FormatSpec & Width(std::size_t w) { width = w; return *this; }
	FormatSpec & Prec(int p) { precision = p; return *this; }
public:
	FormatSpec & Dec() { return BaseDec(); }
	FormatSpec & Hex() { return BaseHex(); }
	FormatSpec & Low() { return CaseLow(); }
	FormatSpec & Upp() { return CaseUpp(); }
	FormatSpec & Off() { return FillOff(); }
	FormatSpec & Spc() { return FillSpc(); }
	FormatSpec & Nul() { return FillNul(); }
	FormatSpec & Nrm() { return NotaNrm(); }
	FormatSpec & Fix() { return NotaFix(); }
	FormatSpec & Sci() { return NotaSci(); }
public:
	FormatSpec & Decimal() { return BaseDec(); }
	FormatSpec & Hexadecimal() { return BaseHex(); }
	FormatSpec & Lower() { return CaseLow(); }
	FormatSpec & Upper() { return CaseUpp(); }
	FormatSpec & FillNone() { return FillOff(); }
	FormatSpec & FillSpace() { return FillSpc(); }
	FormatSpec & FillZero() { return FillNul(); }
	FormatSpec & FloatNormal() { return NotaNrm(); }
	FormatSpec & FloatFixed() { return NotaFix(); }
	FormatSpec & FloatScientific() { return NotaSci(); }
	FormatSpec & Precision(int p) { return Prec(p); }
	template<typename Tstring, typename T>
	inline Tstring ToStringT(const T & x) const
	{
		return FormatValTFunctor<Tstring>()(x, *this);
	}
	template<typename T>
	inline std::string ToString(const T & x) const
	{
		return FormatVal(x, *this);
	}
#if MPT_WSTRING_FORMAT
	template<typename T>
	inline std::wstring ToWString(const T & x) const
	{
		return FormatValW(x, *this);
	}
#endif
};


template <typename Tstring>
struct fmtT : fmt_base
{

template<typename T>
static inline Tstring val(const T& x)
{
	return ToStringTFunctor<Tstring>()(x);
}

template<typename T>
static inline Tstring dec(const T& x)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	return FormatValTFunctor<Tstring>()(x, FormatSpec().BaseDec().FillOff());
}
template<int width, typename T>
static inline Tstring dec(const T& x)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	return FormatValTFunctor<Tstring>()(x, FormatSpec().BaseDec().FillSpc().Width(width));
}
template<int width, typename T>
static inline Tstring dec0(const T& x)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	return FormatValTFunctor<Tstring>()(x, FormatSpec().BaseDec().FillNul().Width(width));
}

template<typename T>
static inline Tstring hex(const T& x)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	return FormatValTFunctor<Tstring>()(x, FormatSpec().BaseHex().CaseLow().FillOff());
}
template<typename T>
static inline Tstring HEX(const T& x)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	return FormatValTFunctor<Tstring>()(x, FormatSpec().BaseHex().CaseUpp().FillOff());
}
template<int width, typename T>
static inline Tstring hex(const T& x)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	return FormatValTFunctor<Tstring>()(x, FormatSpec().BaseHex().CaseLow().FillSpc().Width(width));
}
template<int width, typename T>
static inline Tstring HEX(const T& x)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	return FormatValTFunctor<Tstring>()(x, FormatSpec().BaseHex().CaseUpp().FillSpc().Width(width));
}
template<int width, typename T>
static inline Tstring hex0(const T& x)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	return FormatValTFunctor<Tstring>()(x, FormatSpec().BaseHex().CaseLow().FillNul().Width(width));
}
template<int width, typename T>
static inline Tstring HEX0(const T& x)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	return FormatValTFunctor<Tstring>()(x, FormatSpec().BaseHex().CaseUpp().FillNul().Width(width));
}

template<typename T>
static inline Tstring flt(const T& x, std::size_t width = 0, int precision = -1)
{
	STATIC_ASSERT(std::is_floating_point<T>::value);
	if(width == 0)
	{
		return FormatValTFunctor<Tstring>()(x, FormatSpec().NotaNrm().FillOff().Precision(precision));
	} else
	{
		return FormatValTFunctor<Tstring>()(x, FormatSpec().NotaNrm().FillSpc().Width(width).Precision(precision));
	}
}
template<typename T>
static inline Tstring fix(const T& x, std::size_t width = 0, int precision = -1)
{
	STATIC_ASSERT(std::is_floating_point<T>::value);
	if(width == 0)
	{
		return FormatValTFunctor<Tstring>()(x, FormatSpec().NotaFix().FillOff().Precision(precision));
	} else
	{
		return FormatValTFunctor<Tstring>()(x, FormatSpec().NotaFix().FillSpc().Width(width).Precision(precision));
	}
}
template<typename T>
static inline Tstring sci(const T& x, std::size_t width = 0, int precision = -1)
{
	STATIC_ASSERT(std::is_floating_point<T>::value);
	if(width == 0)
	{
		return FormatValTFunctor<Tstring>()(x, FormatSpec().NotaSci().FillOff().Precision(precision));
	} else
	{
		return FormatValTFunctor<Tstring>()(x, FormatSpec().NotaSci().FillSpc().Width(width).Precision(precision));
	}
}

template <typename T, typename Tformat>
static inline Tstring f(const Tformat & format, const T& x)
{
	STATIC_ASSERT(std::is_floating_point<T>::value);
	return FormatValTFunctor<Tstring>()(x, FormatSpec().ParsePrintf(format));
}

}; // struct fmtT

typedef fmtT<std::string> fmt;
#if MPT_WSTRING_FORMAT
typedef fmtT<std::wstring> wfmt;
#endif
#if MPT_USTRING_MODE_WIDE
typedef fmtT<std::wstring> ufmt;
#else
typedef fmtT<mpt::ustring> ufmt;
#endif
#if defined(_MFC_VER)
typedef fmtT<CString> tfmt;
#endif

} // namespace mpt

namespace mpt {

namespace String {

namespace detail
{

template <typename T> struct to_string_type { };
template <> struct to_string_type<std::string    > { typedef std::string  type; };
template <> struct to_string_type<char           > { typedef std::string  type; };
template <> struct to_string_type<char *         > { typedef std::string  type; };
template <> struct to_string_type<const char     > { typedef std::string  type; };
template <> struct to_string_type<const char *   > { typedef std::string  type; };
template <> struct to_string_type<std::wstring   > { typedef std::wstring type; };
template <> struct to_string_type<wchar_t        > { typedef std::wstring type; };
template <> struct to_string_type<wchar_t *      > { typedef std::wstring type; };
template <> struct to_string_type<const wchar_t  > { typedef std::wstring type; };
template <> struct to_string_type<const wchar_t *> { typedef std::wstring type; };
#if MPT_USTRING_MODE_UTF8
template <> struct to_string_type<mpt::ustring   > { typedef mpt::ustring type; };
#endif
#if defined(_MFC_VER)
template <> struct to_string_type<CString        > { typedef CString      type; };
#endif
template <typename T, std::size_t N> struct to_string_type<T [N]> { typedef typename to_string_type<T>::type type; };

std::string PrintImpl(const std::string & format
	, const std::string & x1 = std::string()
	, const std::string & x2 = std::string()
	, const std::string & x3 = std::string()
	, const std::string & x4 = std::string()
	, const std::string & x5 = std::string()
	, const std::string & x6 = std::string()
	, const std::string & x7 = std::string()
	, const std::string & x8 = std::string()
	);

#if MPT_WSTRING_FORMAT
std::wstring PrintImpl(const std::wstring & format
	, const std::wstring & x1 = std::wstring()
	, const std::wstring & x2 = std::wstring()
	, const std::wstring & x3 = std::wstring()
	, const std::wstring & x4 = std::wstring()
	, const std::wstring & x5 = std::wstring()
	, const std::wstring & x6 = std::wstring()
	, const std::wstring & x7 = std::wstring()
	, const std::wstring & x8 = std::wstring()
	);
#endif

#if MPT_USTRING_MODE_UTF8
mpt::ustring PrintImpl(const mpt::ustring & format
	, const mpt::ustring & x1 = mpt::ustring()
	, const mpt::ustring & x2 = mpt::ustring()
	, const mpt::ustring & x3 = mpt::ustring()
	, const mpt::ustring & x4 = mpt::ustring()
	, const mpt::ustring & x5 = mpt::ustring()
	, const mpt::ustring & x6 = mpt::ustring()
	, const mpt::ustring & x7 = mpt::ustring()
	, const mpt::ustring & x8 = mpt::ustring()
	);
#endif

#if defined(_MFC_VER)
CString PrintImpl(const CString & format
	, const CString & x1 = CString()
	, const CString & x2 = CString()
	, const CString & x3 = CString()
	, const CString & x4 = CString()
	, const CString & x5 = CString()
	, const CString & x6 = CString()
	, const CString & x7 = CString()
	, const CString & x8 = CString()
	);
#endif

} // namespace detail

} // namespace String

template<typename Tformat>
struct message_formatter
{
typedef typename mpt::String::detail::to_string_type<Tformat>::type Tstring;
Tstring format;
message_formatter(const Tstring & format) : format(format) {}

Tstring operator() (
	) const
{
	return mpt::String::detail::PrintImpl(format
	);
}

template<
	  typename T1
>
Tstring operator() (
	  const T1& x1
	) const {
	return mpt::String::detail::PrintImpl(format
		, ToStringTFunctor<Tstring>()(x1)
	);
}

template<
	  typename T1
	, typename T2
>
Tstring operator() (
	  const T1& x1
	, const T2& x2
	) const {
	return mpt::String::detail::PrintImpl(format
		, ToStringTFunctor<Tstring>()(x1)
		, ToStringTFunctor<Tstring>()(x2)
	);
}

template<
	  typename T1
	, typename T2
	, typename T3
>
Tstring operator() (
	  const T1& x1
	, const T2& x2
	, const T3& x3
	) const {
	return mpt::String::detail::PrintImpl(format
		, ToStringTFunctor<Tstring>()(x1)
		, ToStringTFunctor<Tstring>()(x2)
		, ToStringTFunctor<Tstring>()(x3)
	);
}

template<
	  typename T1
	, typename T2
	, typename T3
	, typename T4
>
Tstring operator() (
	  const T1& x1
	, const T2& x2
	, const T3& x3
	, const T4& x4
	) const {
	return mpt::String::detail::PrintImpl(format
		, ToStringTFunctor<Tstring>()(x1)
		, ToStringTFunctor<Tstring>()(x2)
		, ToStringTFunctor<Tstring>()(x3)
		, ToStringTFunctor<Tstring>()(x4)
	);
}

template<
	  typename T1
	, typename T2
	, typename T3
	, typename T4
	, typename T5
>
Tstring operator() (
	  const T1& x1
	, const T2& x2
	, const T3& x3
	, const T4& x4
	, const T5& x5
	) const {
	return mpt::String::detail::PrintImpl(format
		, ToStringTFunctor<Tstring>()(x1)
		, ToStringTFunctor<Tstring>()(x2)
		, ToStringTFunctor<Tstring>()(x3)
		, ToStringTFunctor<Tstring>()(x4)
		, ToStringTFunctor<Tstring>()(x5)
	);
}

template<
	  typename T1
	, typename T2
	, typename T3
	, typename T4
	, typename T5
	, typename T6
>
Tstring operator() (
	  const T1& x1
	, const T2& x2
	, const T3& x3
	, const T4& x4
	, const T5& x5
	, const T6& x6
	) const {
	return mpt::String::detail::PrintImpl(format
		, ToStringTFunctor<Tstring>()(x1)
		, ToStringTFunctor<Tstring>()(x2)
		, ToStringTFunctor<Tstring>()(x3)
		, ToStringTFunctor<Tstring>()(x4)
		, ToStringTFunctor<Tstring>()(x5)
		, ToStringTFunctor<Tstring>()(x6)
	);
}

template<
	  typename T1
	, typename T2
	, typename T3
	, typename T4
	, typename T5
	, typename T6
	, typename T7
>
Tstring operator() (
	  const T1& x1
	, const T2& x2
	, const T3& x3
	, const T4& x4
	, const T5& x5
	, const T6& x6
	, const T7& x7
	) const {
	return mpt::String::detail::PrintImpl(format
		, ToStringTFunctor<Tstring>()(x1)
		, ToStringTFunctor<Tstring>()(x2)
		, ToStringTFunctor<Tstring>()(x3)
		, ToStringTFunctor<Tstring>()(x4)
		, ToStringTFunctor<Tstring>()(x5)
		, ToStringTFunctor<Tstring>()(x6)
		, ToStringTFunctor<Tstring>()(x7)
	);
}

template<
	  typename T1
	, typename T2
	, typename T3
	, typename T4
	, typename T5
	, typename T6
	, typename T7
	, typename T8
>
Tstring operator() (
	  const T1& x1
	, const T2& x2
	, const T3& x3
	, const T4& x4
	, const T5& x5
	, const T6& x6
	, const T7& x7
	, const T8& x8
	) const {
	return mpt::String::detail::PrintImpl(format
		, ToStringTFunctor<Tstring>()(x1)
		, ToStringTFunctor<Tstring>()(x2)
		, ToStringTFunctor<Tstring>()(x3)
		, ToStringTFunctor<Tstring>()(x4)
		, ToStringTFunctor<Tstring>()(x5)
		, ToStringTFunctor<Tstring>()(x6)
		, ToStringTFunctor<Tstring>()(x7)
		, ToStringTFunctor<Tstring>()(x8)
	);
}

}; // struct message_formatter<Tformat>

template<typename Tformat>
message_formatter<typename mpt::String::detail::to_string_type<Tformat>::type> format(const Tformat &format)
{
	typedef typename mpt::String::detail::to_string_type<Tformat>::type Tstring;
	return message_formatter<Tstring>(Tstring(format));
}

#if MPT_WSTRING_FORMAT
static inline message_formatter<std::wstring> wformat(const std::wstring &format)
{
	return message_formatter<std::wstring>(format);
}
#endif

static inline message_formatter<mpt::ustring> uformat(const mpt::ustring &format)
{
	return message_formatter<mpt::ustring>(format);
}

#if defined(_MFC_VER)
static inline message_formatter<CString> tformat(const CString &format)
{
	return message_formatter<CString>(format);
}
#endif

} // namespace mpt


OPENMPT_NAMESPACE_END
