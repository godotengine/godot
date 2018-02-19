/*
 * mptStringParse.cpp
 * ------------------
 * Purpose: Convert strings to other types.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "mptStringParse.h"

#include <locale>
#include <sstream>


OPENMPT_NAMESPACE_BEGIN


template<typename T>
inline T ConvertStrToHelper(const std::string &str)
{
	std::istringstream i(str);
	i.imbue(std::locale::classic());
	T x;
	if(!(i >> x))
	{
		return T();
	}
	return x;
}
template<> inline bool ConvertStrToHelper(const std::string &str) { return ConvertStrToHelper<int>(str)?true:false; }
template<> inline signed char ConvertStrToHelper(const std::string &str) { return static_cast<signed char>(ConvertStrToHelper<signed int>(str)); }
template<> inline unsigned char ConvertStrToHelper(const std::string &str) { return static_cast<unsigned char>(ConvertStrToHelper<unsigned int>(str)); }

#if MPT_WSTRING_FORMAT
template<typename T>
inline T ConvertStrToHelper(const std::wstring &str)
{
	std::wistringstream i(str);
	i.imbue(std::locale::classic());
	T x;
	if(!(i >> x))
	{
		return T();
	}
	return x;
}
template<> inline bool ConvertStrToHelper(const std::wstring &str) { return ConvertStrToHelper<int>(str)?true:false; }
template<> inline signed char ConvertStrToHelper(const std::wstring &str) { return static_cast<signed char>(ConvertStrToHelper<signed int>(str)); }
template<> inline unsigned char ConvertStrToHelper(const std::wstring &str) { return static_cast<unsigned char>(ConvertStrToHelper<unsigned int>(str)); }
#endif

bool ConvertStrToBool(const std::string &str) { return ConvertStrToHelper<bool>(str); }
signed char ConvertStrToSignedChar(const std::string &str) { return ConvertStrToHelper<signed char>(str); }
unsigned char ConvertStrToUnsignedChar(const std::string &str) { return ConvertStrToHelper<unsigned char>(str); }
signed short ConvertStrToSignedShort(const std::string &str) { return ConvertStrToHelper<signed short>(str); }
unsigned short ConvertStrToUnsignedShort(const std::string &str) { return ConvertStrToHelper<unsigned short>(str); }
signed int ConvertStrToSignedInt(const std::string &str) { return ConvertStrToHelper<signed int>(str); }
unsigned int ConvertStrToUnsignedInt(const std::string &str) { return ConvertStrToHelper<unsigned int>(str); }
signed long ConvertStrToSignedLong(const std::string &str) { return ConvertStrToHelper<signed long>(str); }
unsigned long ConvertStrToUnsignedLong(const std::string &str) { return ConvertStrToHelper<unsigned long>(str); }
signed long long ConvertStrToSignedLongLong(const std::string &str) { return ConvertStrToHelper<signed long long>(str); }
unsigned long long ConvertStrToUnsignedLongLong(const std::string &str) { return ConvertStrToHelper<unsigned long long>(str); }
float ConvertStrToFloat(const std::string &str) { return ConvertStrToHelper<float>(str); }
double ConvertStrToDouble(const std::string &str) { return ConvertStrToHelper<double>(str); }
long double ConvertStrToLongDouble(const std::string &str) { return ConvertStrToHelper<long double>(str); }

#if MPT_WSTRING_FORMAT
bool ConvertStrToBool(const std::wstring &str) { return ConvertStrToHelper<bool>(str); }
signed char ConvertStrToSignedChar(const std::wstring &str) { return ConvertStrToHelper<signed char>(str); }
unsigned char ConvertStrToUnsignedChar(const std::wstring &str) { return ConvertStrToHelper<unsigned char>(str); }
signed short ConvertStrToSignedShort(const std::wstring &str) { return ConvertStrToHelper<signed short>(str); }
unsigned short ConvertStrToUnsignedShort(const std::wstring &str) { return ConvertStrToHelper<unsigned short>(str); }
signed int ConvertStrToSignedInt(const std::wstring &str) { return ConvertStrToHelper<signed int>(str); }
unsigned int ConvertStrToUnsignedInt(const std::wstring &str) { return ConvertStrToHelper<unsigned int>(str); }
signed long ConvertStrToSignedLong(const std::wstring &str) { return ConvertStrToHelper<signed long>(str); }
unsigned long ConvertStrToUnsignedLong(const std::wstring &str) { return ConvertStrToHelper<unsigned long>(str); }
signed long long ConvertStrToSignedLongLong(const std::wstring &str) { return ConvertStrToHelper<signed long long>(str); }
unsigned long long ConvertStrToUnsignedLongLong(const std::wstring &str) { return ConvertStrToHelper<unsigned long long>(str); }
float ConvertStrToFloat(const std::wstring &str) { return ConvertStrToHelper<float>(str); }
double ConvertStrToDouble(const std::wstring &str) { return ConvertStrToHelper<double>(str); }
long double ConvertStrToLongDouble(const std::wstring &str) { return ConvertStrToHelper<long double>(str); }
#endif


namespace mpt
{
namespace String
{
namespace Parse
{

template<typename T>
T HexToHelper(const std::string &str)
{
	std::istringstream i(str);
	i.imbue(std::locale::classic());
	T x;
	if(!(i >> std::hex >> x))
	{
		return T();
	}
	return x;
}
template<> unsigned char HexToHelper(const std::string &str) { return static_cast<unsigned char>(HexToHelper<unsigned int>(str)); }

unsigned char HexToUnsignedChar(const std::string &str) { return HexToHelper<unsigned char>(str); }
unsigned short HexToUnsignedShort(const std::string &str) { return HexToHelper<unsigned short>(str); }
unsigned int HexToUnsignedInt(const std::string &str) { return HexToHelper<unsigned int>(str); }
unsigned long HexToUnsignedLong(const std::string &str) { return HexToHelper<unsigned long>(str); }
unsigned long long HexToUnsignedLongLong(const std::string &str) { return HexToHelper<unsigned long long>(str); }

} // namespace Parse
} // namespace String
} // namespace mpt


OPENMPT_NAMESPACE_END
