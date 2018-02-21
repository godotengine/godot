/*
 * mptStringFormat.cpp
 * -------------------
 * Purpose: Convert other types to strings.
 * Notes  : Currently none.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#include "stdafx.h"
#include "mptStringFormat.h"

#include <iomanip>
#include <locale>
#include <sstream>
#include <string>


OPENMPT_NAMESPACE_BEGIN



namespace mpt
{


template<typename Tstream, typename T> inline void SaneInsert(Tstream & s, const T & x) { s << x; }
// do the right thing for signed/unsigned char and bool
template<typename Tstream> void SaneInsert(Tstream & s, const bool & x) { s << static_cast<int>(x); }
template<typename Tstream> void SaneInsert(Tstream & s, const signed char & x) { s << static_cast<signed int>(x); }
template<typename Tstream> void SaneInsert(Tstream & s, const unsigned char & x) { s << static_cast<unsigned int>(x); }
 
template<typename T>
inline std::string ToStringHelper(const T & x)
{
	std::ostringstream o;
	o.imbue(std::locale::classic());
	SaneInsert(o, x);
	return o.str();
}

#if MPT_WSTRING_FORMAT
template<typename T>
inline std::wstring ToWStringHelper(const T & x)
{
	std::wostringstream o;
	o.imbue(std::locale::classic());
	SaneInsert(o, x);
	return o.str();
}
#endif

#if MPT_WSTRING_CONVERT
std::string ToString(const std::wstring & x) { return mpt::ToCharset(mpt::CharsetLocaleOrUTF8, x); }
std::string ToString(const wchar_t * const & x) { return mpt::ToCharset(mpt::CharsetLocaleOrUTF8, x); }
std::string ToString(const wchar_t & x) { return mpt::ToCharset(mpt::CharsetLocaleOrUTF8, std::wstring(1, x)); }
#endif
#if MPT_USTRING_MODE_UTF8
std::string ToString(const mpt::ustring & x) { return mpt::ToCharset(mpt::CharsetLocaleOrUTF8, x); }
#endif
#if defined(_MFC_VER)
std::string ToString(const CString & x) { return mpt::ToCharset(mpt::CharsetLocaleOrUTF8, x); }
#endif
std::string ToString(const bool & x) { return ToStringHelper(x); }
std::string ToString(const signed char & x) { return ToStringHelper(x); }
std::string ToString(const unsigned char & x) { return ToStringHelper(x); }
std::string ToString(const signed short & x) { return ToStringHelper(x); }
std::string ToString(const unsigned short & x) { return ToStringHelper(x); }
std::string ToString(const signed int & x) { return ToStringHelper(x); }
std::string ToString(const unsigned int & x) { return ToStringHelper(x); }
std::string ToString(const signed long & x) { return ToStringHelper(x); }
std::string ToString(const unsigned long & x) { return ToStringHelper(x); }
std::string ToString(const signed long long & x) { return ToStringHelper(x); }
std::string ToString(const unsigned long long & x) { return ToStringHelper(x); }
std::string ToString(const float & x) { return ToStringHelper(x); }
std::string ToString(const double & x) { return ToStringHelper(x); }
std::string ToString(const long double & x) { return ToStringHelper(x); }

mpt::ustring ToUString(const std::string & x) { return mpt::ToUnicode(mpt::CharsetLocaleOrUTF8, x); }
mpt::ustring ToUString(const char * const & x) { return mpt::ToUnicode(mpt::CharsetLocaleOrUTF8, x); }
mpt::ustring ToUString(const char & x) { return mpt::ToUnicode(mpt::CharsetLocaleOrUTF8, std::string(1, x)); }
#if MPT_WSTRING_FORMAT
#if MPT_USTRING_MODE_UTF8
mpt::ustring ToUString(const std::wstring & x) { return mpt::ToUnicode(x); }
#endif
mpt::ustring ToUString(const wchar_t * const & x) { return mpt::ToUnicode(x); }
mpt::ustring ToUString(const wchar_t & x) { return mpt::ToUnicode(std::wstring(1, x)); }
#endif
#if defined(_MFC_VER)
mpt::ustring ToUString(const CString & x)  { return mpt::ToUnicode(x); }
#endif
#if MPT_USTRING_MODE_WIDE
mpt::ustring ToUString(const bool & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const signed char & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const unsigned char & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const signed short & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const unsigned short & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const signed int & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const unsigned int & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const signed long & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const unsigned long & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const signed long long & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const unsigned long long & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const float & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const double & x) { return ToWStringHelper(x); }
mpt::ustring ToUString(const long double & x) { return ToWStringHelper(x); }
#endif
#if MPT_USTRING_MODE_UTF8
mpt::ustring ToUString(const bool & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const signed char & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const unsigned char & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const signed short & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const unsigned short & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const signed int & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const unsigned int & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const signed long & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const unsigned long & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const signed long long & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const unsigned long long & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const float & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const double & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
mpt::ustring ToUString(const long double & x) { return mpt::ToUnicode(mpt::CharsetUTF8, ToStringHelper(x)); }
#endif

#if MPT_WSTRING_FORMAT
std::wstring ToWString(const std::string & x) { return mpt::ToWide(mpt::CharsetLocaleOrUTF8, x); }
std::wstring ToWString(const char * const & x) { return mpt::ToWide(mpt::CharsetLocaleOrUTF8, x); }
std::wstring ToWString(const char & x) { return mpt::ToWide(mpt::CharsetLocaleOrUTF8, std::string(1, x)); }
#if MPT_USTRING_MODE_UTF8
std::wstring ToWString(const mpt::ustring & x) { return mpt::ToWide(x); }
#endif
#if defined(_MFC_VER)
std::wstring ToWString(const CString & x) { return mpt::ToWide(x); }
#endif
std::wstring ToWString(const bool & x) { return ToWStringHelper(x); }
std::wstring ToWString(const signed char & x) { return ToWStringHelper(x); }
std::wstring ToWString(const unsigned char & x) { return ToWStringHelper(x); }
std::wstring ToWString(const signed short & x) { return ToWStringHelper(x); }
std::wstring ToWString(const unsigned short & x) { return ToWStringHelper(x); }
std::wstring ToWString(const signed int & x) { return ToWStringHelper(x); }
std::wstring ToWString(const unsigned int & x) { return ToWStringHelper(x); }
std::wstring ToWString(const signed long & x) { return ToWStringHelper(x); }
std::wstring ToWString(const unsigned long & x) { return ToWStringHelper(x); }
std::wstring ToWString(const signed long long & x) { return ToWStringHelper(x); }
std::wstring ToWString(const unsigned long long & x) { return ToWStringHelper(x); }
std::wstring ToWString(const float & x) { return ToWStringHelper(x); }
std::wstring ToWString(const double & x) { return ToWStringHelper(x); }
std::wstring ToWString(const long double & x) { return ToWStringHelper(x); }
#endif


template<typename Tostream>
inline void ApplyFormat(Tostream & o, const FormatSpec & format)
{
	FormatFlags f = format.GetFlags();
	std::size_t width = format.GetWidth();
	int precision = format.GetPrecision();
	if(precision != -1 && width != 0 && !(f & fmt_base::NotaFix) && !(f & fmt_base::NotaSci))
	{
		// fixup:
		// precision behaves differently from .#
		// avoid default format when precision and width are set
		f &= ~fmt_base::NotaNrm;
		f |= fmt_base::NotaFix;
	}
	if(f & fmt_base::BaseDec) { o << std::dec; }
	else if(f & fmt_base::BaseHex) { o << std::hex; }
	if(f & fmt_base::NotaNrm ) { /*nothing*/ }
	else if(f & fmt_base::NotaFix ) { o << std::setiosflags(std::ios::fixed); }
	else if(f & fmt_base::NotaSci ) { o << std::setiosflags(std::ios::scientific); }
	if(f & fmt_base::CaseLow) { o << std::nouppercase; }
	else if(f & fmt_base::CaseUpp) { o << std::uppercase; }
	if(f & fmt_base::FillOff) { /* nothing */ }
	else if(f & fmt_base::FillNul) { o << std::setw(width) << std::setfill(typename Tostream::char_type('0')); }
	else if(f & fmt_base::FillSpc) { o << std::setw(width) << std::setfill(typename Tostream::char_type(' ')); }
	if(precision != -1) { o << std::setprecision(precision); }
}


template<typename T>
inline std::string FormatValHelper(const T & x, const FormatSpec & f)
{
	std::ostringstream o;
	o.imbue(std::locale::classic());
	ApplyFormat(o, f);
	SaneInsert(o, x);
	return o.str();
}

#if MPT_WSTRING_FORMAT
template<typename T>
inline std::wstring FormatValWHelper(const T & x, const FormatSpec & f)
{
	std::wostringstream o;
	o.imbue(std::locale::classic());
	ApplyFormat(o, f);
	SaneInsert(o, x);
	return o.str();
}
#endif

// Parses a useful subset of standard sprintf syntax for specifying floating point formatting.
template<typename Tchar>
static inline FormatSpec ParseFormatStringFloat(const Tchar * str)
{
	MPT_ASSERT(str);
	FormatFlags f = FormatFlags();
	std::size_t width = 0;
	int precision = -1;
	if(!str)
	{
		return FormatSpec();
	}
	const Tchar * p = str;
	while(*p && *p != Tchar('%'))
	{
		++p;
	}
	++p;
	while(*p && (*p == Tchar(' ') || *p == Tchar('0')))
	{
		if(*p == Tchar(' ')) f |= mpt::fmt_base::FillSpc;
		if(*p == Tchar('0')) f |= mpt::fmt_base::FillNul;
		++p;
	}
	if(!(f & mpt::fmt_base::FillSpc) && !(f & mpt::fmt_base::FillNul))
	{
		f |= mpt::fmt_base::FillOff;
	}
	while(*p && (Tchar('0') <= *p && *p <= Tchar('9')))
	{
		if(f & mpt::fmt_base::FillOff)
		{
			f &= ~mpt::fmt_base::FillOff;
			f |= mpt::fmt_base::FillSpc;
		}
		width *= 10;
		width += *p - Tchar('0');
		++p;
	}
	if(*p && *p == Tchar('.'))
	{
		++p;
		precision = 0;
		while(*p && (Tchar('0') <= *p && *p <= Tchar('9')))
		{
			precision *= 10;
			precision += *p - Tchar('0');
			++p;
		}
	}
	if(*p && (*p == Tchar('g') || *p == Tchar('G') || *p == Tchar('f') || *p == Tchar('F') || *p == Tchar('e') || *p == Tchar('E')))
	{
		if(*p == Tchar('g')) f |= mpt::fmt_base::NotaNrm | mpt::fmt_base::CaseLow;
		if(*p == Tchar('G')) f |= mpt::fmt_base::NotaNrm | mpt::fmt_base::CaseUpp;
		if(*p == Tchar('f')) f |= mpt::fmt_base::NotaFix | mpt::fmt_base::CaseLow;
		if(*p == Tchar('F')) f |= mpt::fmt_base::NotaFix | mpt::fmt_base::CaseUpp;
		if(*p == Tchar('e')) f |= mpt::fmt_base::NotaSci | mpt::fmt_base::CaseLow;
		if(*p == Tchar('E')) f |= mpt::fmt_base::NotaSci | mpt::fmt_base::CaseUpp;
		++p;
	}
	return FormatSpec().SetFlags(f).SetWidth(width).SetPrecision(precision);
}

FormatSpec & FormatSpec::ParsePrintf(const char * format)
{
	*this = ParseFormatStringFloat(format);
	return *this;
}
FormatSpec & FormatSpec::ParsePrintf(const wchar_t * format)
{
	*this = ParseFormatStringFloat(format);
	return *this;
}
FormatSpec & FormatSpec::ParsePrintf(const std::string & format)
{
	*this = ParseFormatStringFloat(format.c_str());
	return *this;
}
FormatSpec & FormatSpec::ParsePrintf(const std::wstring & format)
{
	*this = ParseFormatStringFloat(format.c_str());
	return *this;
}


std::string FormatVal(const char & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const wchar_t & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const bool & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const signed char & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const unsigned char & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const signed short & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const unsigned short & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const signed int & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const unsigned int & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const signed long & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const unsigned long & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const signed long long & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const unsigned long long & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const float & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const double & x, const FormatSpec & f) { return FormatValHelper(x, f); }
std::string FormatVal(const long double & x, const FormatSpec & f) { return FormatValHelper(x, f); }

#if MPT_WSTRING_FORMAT
std::wstring FormatValW(const char & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const wchar_t & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const bool & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const signed char & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const unsigned char & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const signed short & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const unsigned short & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const signed int & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const unsigned int & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const signed long & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const unsigned long & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const signed long long & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const unsigned long long & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const float & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const double & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
std::wstring FormatValW(const long double & x, const FormatSpec & f) { return FormatValWHelper(x, f); }
#endif


namespace String
{


namespace detail
{

template<typename Tstring>
Tstring PrintImplTemplate(const Tstring & format
	, const Tstring & x1
	, const Tstring & x2
	, const Tstring & x3
	, const Tstring & x4
	, const Tstring & x5
	, const Tstring & x6
	, const Tstring & x7
	, const Tstring & x8
	)
{
	Tstring result;
	const std::size_t len = format.length();
	result.reserve(len);
	for(std::size_t pos = 0; pos != len; ++pos)
	{
		typename Tstring::value_type c = format[pos];
		if(pos + 1 != len && c == '%')
		{
			pos++;
			c = format[pos];
			if('1' <= c && c <= '9')
			{
				const std::size_t n = c - '0';
				switch(n)
				{
					case 1: result.append(x1); break;
					case 2: result.append(x2); break;
					case 3: result.append(x3); break;
					case 4: result.append(x4); break;
					case 5: result.append(x5); break;
					case 6: result.append(x6); break;
					case 7: result.append(x7); break;
					case 8: result.append(x8); break;
				}
				continue;
			} else if(c != '%')
			{
				result.append(1, '%');
			}
		}
		result.append(1, c);
	}
	return result;
}

#if defined(_MFC_VER)
template<>
CString PrintImplTemplate<CString>(const CString & format
	, const CString & x1
	, const CString & x2
	, const CString & x3
	, const CString & x4
	, const CString & x5
	, const CString & x6
	, const CString & x7
	, const CString & x8
	)
{
	CString result;
	const int len = format.GetLength();
	result.Preallocate(len);
	for(int pos = 0; pos != len; ++pos)
	{
		CString::XCHAR c = format[pos];
		if(pos + 1 != len && c == _T('%'))
		{
			pos++;
			c = format[pos];
			if(_T('1') <= c && c <= _T('9'))
			{
				const std::size_t n = c - _T('0');
				switch(n)
				{
					case 1: result += x1; break;
					case 2: result += x2; break;
					case 3: result += x3; break;
					case 4: result += x4; break;
					case 5: result += x5; break;
					case 6: result += x6; break;
					case 7: result += x7; break;
					case 8: result += x8; break;
				}
				continue;
			} else if(c != _T('%'))
			{
				result.AppendChar(_T('%'));
			}
		}
		result.AppendChar(c);
	}
	return result;
}
#endif

std::string PrintImpl(const std::string & format
	, const std::string & x1
	, const std::string & x2
	, const std::string & x3
	, const std::string & x4
	, const std::string & x5
	, const std::string & x6
	, const std::string & x7
	, const std::string & x8
	)
{
	return PrintImplTemplate<std::string>(format, x1,x2,x3,x4,x5,x6,x7,x8);
}

#if MPT_WSTRING_FORMAT
std::wstring PrintImpl(const std::wstring & format
	, const std::wstring & x1
	, const std::wstring & x2
	, const std::wstring & x3
	, const std::wstring & x4
	, const std::wstring & x5
	, const std::wstring & x6
	, const std::wstring & x7
	, const std::wstring & x8
	)
{
	return PrintImplTemplate<std::wstring>(format, x1,x2,x3,x4,x5,x6,x7,x8);
}
#endif

#if MPT_USTRING_MODE_UTF8
mpt::ustring PrintImpl(const mpt::ustring & format
	, const mpt::ustring & x1
	, const mpt::ustring & x2
	, const mpt::ustring & x3
	, const mpt::ustring & x4
	, const mpt::ustring & x5
	, const mpt::ustring & x6
	, const mpt::ustring & x7
	, const mpt::ustring & x8
	)
{
	return PrintImplTemplate<mpt::ustring>(format, x1,x2,x3,x4,x5,x6,x7,x8);
}
#endif

#if defined(_MFC_VER)
CString PrintImpl(const CString & format
	, const CString & x1
	, const CString & x2
	, const CString & x3
	, const CString & x4
	, const CString & x5
	, const CString & x6
	, const CString & x7
	, const CString & x8
	)
{
	return PrintImplTemplate<CString>(format, x1,x2,x3,x4,x5,x6,x7,x8);
}
#endif

} // namespace detail


} // namespace String


} // namespace mpt


OPENMPT_NAMESPACE_END
