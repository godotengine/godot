/*
 * mptStringParse.h
 * ----------------
 * Purpose: Convert strings to other types.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


OPENMPT_NAMESPACE_BEGIN


bool ConvertStrToBool(const std::string &str);
signed char ConvertStrToSignedChar(const std::string &str);
unsigned char ConvertStrToUnsignedChar(const std::string &str);
signed short ConvertStrToSignedShort(const std::string &str);
unsigned short ConvertStrToUnsignedShort(const std::string &str);
signed int ConvertStrToSignedInt(const std::string &str);
unsigned int ConvertStrToUnsignedInt(const std::string &str);
signed long ConvertStrToSignedLong(const std::string &str);
unsigned long ConvertStrToUnsignedLong(const std::string &str);
signed long long ConvertStrToSignedLongLong(const std::string &str);
unsigned long long ConvertStrToUnsignedLongLong(const std::string &str);
float ConvertStrToFloat(const std::string &str);
double ConvertStrToDouble(const std::string &str);
long double ConvertStrToLongDouble(const std::string &str);
template<typename T> inline T ConvertStrTo(const std::string &str); // not defined, generates compiler error for non-specialized types
template<> inline std::string ConvertStrTo(const std::string &str) { return str; }
template<> inline bool ConvertStrTo(const std::string &str) { return ConvertStrToBool(str); }
template<> inline signed char ConvertStrTo(const std::string &str) { return ConvertStrToSignedChar(str); }
template<> inline unsigned char ConvertStrTo(const std::string &str) { return ConvertStrToUnsignedChar(str); }
template<> inline signed short ConvertStrTo(const std::string &str) { return ConvertStrToSignedShort(str); }
template<> inline unsigned short ConvertStrTo(const std::string &str) { return ConvertStrToUnsignedShort(str); }
template<> inline signed int ConvertStrTo(const std::string &str) { return ConvertStrToSignedInt(str); }
template<> inline unsigned int ConvertStrTo(const std::string &str) { return ConvertStrToUnsignedInt(str); }
template<> inline signed long ConvertStrTo(const std::string &str) { return ConvertStrToSignedLong(str); }
template<> inline unsigned long ConvertStrTo(const std::string &str) { return ConvertStrToUnsignedLong(str); }
template<> inline signed long long ConvertStrTo(const std::string &str) { return ConvertStrToSignedLongLong(str); }
template<> inline unsigned long long ConvertStrTo(const std::string &str) { return ConvertStrToUnsignedLongLong(str); }
template<> inline float ConvertStrTo(const std::string &str) { return ConvertStrToFloat(str); }
template<> inline double ConvertStrTo(const std::string &str) { return ConvertStrToDouble(str); }
template<> inline long double ConvertStrTo(const std::string &str) { return ConvertStrToLongDouble(str); }

#if MPT_WSTRING_FORMAT
bool ConvertStrToBool(const std::wstring &str);
signed char ConvertStrToSignedChar(const std::wstring &str);
unsigned char ConvertStrToUnsignedChar(const std::wstring &str);
signed short ConvertStrToSignedShort(const std::wstring &str);
unsigned short ConvertStrToUnsignedShort(const std::wstring &str);
signed int ConvertStrToSignedInt(const std::wstring &str);
unsigned int ConvertStrToUnsignedInt(const std::wstring &str);
signed long ConvertStrToSignedLong(const std::wstring &str);
unsigned long ConvertStrToUnsignedLong(const std::wstring &str);
signed long long ConvertStrToSignedLongLong(const std::wstring &str);
unsigned long long ConvertStrToUnsignedLongLong(const std::wstring &str);
float ConvertStrToFloat(const std::wstring &str);
double ConvertStrToDouble(const std::wstring &str);
long double ConvertStrToLongDouble(const std::wstring &str);
template<typename T> inline T ConvertStrTo(const std::wstring &str); // not defined, generates compiler error for non-specialized types
template<> inline std::wstring ConvertStrTo(const std::wstring &str) { return str; }
template<> inline bool ConvertStrTo(const std::wstring &str) { return ConvertStrToBool(str); }
template<> inline signed char ConvertStrTo(const std::wstring &str) { return ConvertStrToSignedChar(str); }
template<> inline unsigned char ConvertStrTo(const std::wstring &str) { return ConvertStrToUnsignedChar(str); }
template<> inline signed short ConvertStrTo(const std::wstring &str) { return ConvertStrToSignedShort(str); }
template<> inline unsigned short ConvertStrTo(const std::wstring &str) { return ConvertStrToUnsignedShort(str); }
template<> inline signed int ConvertStrTo(const std::wstring &str) { return ConvertStrToSignedInt(str); }
template<> inline unsigned int ConvertStrTo(const std::wstring &str) { return ConvertStrToUnsignedInt(str); }
template<> inline signed long ConvertStrTo(const std::wstring &str) { return ConvertStrToSignedLong(str); }
template<> inline unsigned long ConvertStrTo(const std::wstring &str) { return ConvertStrToUnsignedLong(str); }
template<> inline signed long long ConvertStrTo(const std::wstring &str) { return ConvertStrToSignedLongLong(str); }
template<> inline unsigned long long ConvertStrTo(const std::wstring &str) { return ConvertStrToUnsignedLongLong(str); }
template<> inline float ConvertStrTo(const std::wstring &str) { return ConvertStrToFloat(str); }
template<> inline double ConvertStrTo(const std::wstring &str) { return ConvertStrToDouble(str); }
template<> inline long double ConvertStrTo(const std::wstring &str) { return ConvertStrToLongDouble(str); }
#endif

#if defined(_MFC_VER)
template<typename T>
inline T ConvertStrTo(const CString &str)
{
	#if defined(UNICODE) && MPT_WSTRING_FORMAT
		return ConvertStrTo<T>(mpt::ToWide(str));
	#elif defined(UNICODE)
		return ConvertStrTo<T>(mpt::ToCharset(mpt::CharsetUTF8, str));
	#else // !UNICODE
		return ConvertStrTo<T>(mpt::ToCharset(mpt::CharsetLocale, str));
	#endif // UNICODE
}
#endif // _MFC_VER

template<typename T>
inline T ConvertStrTo(const char *str)
{
	if(!str)
	{
		return T();
	}
	return ConvertStrTo<T>(std::string(str));
}

#if MPT_WSTRING_FORMAT
#if MPT_USTRING_MODE_UTF8
template<> inline mpt::ustring ConvertStrTo(const std::wstring &str) { return mpt::ToUnicode(str); }
#endif
template<typename T>
inline T ConvertStrTo(const wchar_t *str)
{
	if(!str)
	{
		return T();
	}
	return ConvertStrTo<T>(std::wstring(str));
}
#endif

#if MPT_USTRING_MODE_UTF8
template<typename T>
inline T ConvertStrTo(const mpt::ustring &str)
{
	return ConvertStrTo<T>(mpt::ToCharset(mpt::CharsetUTF8, str));
}
template<> inline mpt::ustring ConvertStrTo(const mpt::ustring &str) { return str; }
#if MPT_WSTRING_CONVERT
template<> inline std::wstring ConvertStrTo(const mpt::ustring &str) { return mpt::ToWide(str); }
#endif
#endif


namespace mpt
{
namespace String
{
namespace Parse
{

unsigned char HexToUnsignedChar(const std::string &str);
unsigned short HexToUnsignedShort(const std::string &str);
unsigned int HexToUnsignedInt(const std::string &str);
unsigned long HexToUnsignedLong(const std::string &str);
unsigned long long HexToUnsignedLongLong(const std::string &str);

template<typename T> inline T Hex(const std::string &str); // not defined, generates compiler error for non-specialized types
template<> inline unsigned char Hex(const std::string &str) { return HexToUnsignedChar(str); }
template<> inline unsigned short Hex(const std::string &str) { return HexToUnsignedShort(str); }
template<> inline unsigned int Hex(const std::string &str) { return HexToUnsignedInt(str); }
template<> inline unsigned long Hex(const std::string &str) { return HexToUnsignedLong(str); }
template<> inline unsigned long long Hex(const std::string &str) { return HexToUnsignedLongLong(str); }

template<typename T>
inline T Hex(const char *str)
{
	if(!str)
	{
		return T();
	}
	return Hex<T>(std::string(str));
}

#if MPT_WSTRING_FORMAT

template<typename T>
inline T Hex(const std::wstring &str)
{
	return Hex<T>(mpt::ToCharset(mpt::CharsetUTF8, str));
}

template<typename T>
inline T Hex(const wchar_t *str)
{
	if(!str)
	{
		return T();
	}
	return Hex<T>(std::wstring(str));
}

#endif

#if MPT_USTRING_MODE_UTF8
template<typename T>
inline T Hex(const mpt::ustring &str)
{
	return Hex<T>(mpt::ToCharset(mpt::CharsetUTF8, str));
}
#endif

} // namespace Parse
} // namespace String
} // namespace mpt


OPENMPT_NAMESPACE_END
