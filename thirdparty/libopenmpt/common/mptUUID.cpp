/*
 * mptUUID.cpp
 * -----------
 * Purpose: UUID utility functions.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "mptUUID.h"

#include "mptRandom.h"
#include "mptStringFormat.h"
#include "Endianness.h"

#include <cstdlib>

#if MPT_OS_WINDOWS
#include <windows.h>
#include <rpc.h>
#if defined(MODPLUG_TRACKER) || !defined(NO_DMO) || MPT_OS_WINDOWS_WINRT
#include <objbase.h>
#endif // MODPLUG_TRACKER || !NO_DMO || MPT_OS_WINDOWS_WINRT
#endif // MPT_OS_WINDOWS


OPENMPT_NAMESPACE_BEGIN


#if MPT_OS_WINDOWS


namespace Util
{


#if defined(MODPLUG_TRACKER) || !defined(NO_DMO)


std::wstring CLSIDToString(CLSID clsid)
{
	std::wstring str;
	LPOLESTR tmp = nullptr;
	switch(::StringFromCLSID(clsid, &tmp))
	{
	case S_OK:
		break;
	case E_OUTOFMEMORY:
		if(tmp)
		{
			::CoTaskMemFree(tmp);
			tmp = nullptr;
		}
		MPT_EXCEPTION_THROW_OUT_OF_MEMORY();
		break;
	default:
		if(tmp)
		{
			::CoTaskMemFree(tmp);
			tmp = nullptr;
		}
		throw std::logic_error("StringFromCLSID() failed.");
		break;
	}
	if(!tmp)
	{
		throw std::logic_error("StringFromCLSID() failed.");
	}
	try
	{
		str = tmp;
	} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
	{
		::CoTaskMemFree(tmp);
		tmp = nullptr;
		MPT_UNUSED_VARIABLE(e);
		MPT_EXCEPTION_RETHROW_OUT_OF_MEMORY();
	}
	::CoTaskMemFree(tmp);
	tmp = nullptr;
	return str;
}


CLSID StringToCLSID(const std::wstring &str)
{
	CLSID clsid = CLSID();
	std::vector<OLECHAR> tmp(str.c_str(), str.c_str() + str.length() + 1);
	switch(::CLSIDFromString(tmp.data(), &clsid))
	{
	case NOERROR:
		// nothing
		break;
	case E_INVALIDARG:
		clsid = CLSID();
		break;
	case CO_E_CLASSSTRING:
		clsid = CLSID();
		break;
	case REGDB_E_CLASSNOTREG:
		clsid = CLSID();
		break;
	case REGDB_E_READREGDB:
		clsid = CLSID();
		throw std::runtime_error("CLSIDFromString() failed: REGDB_E_READREGDB.");
		break;
	default:
		clsid = CLSID();
		throw std::logic_error("CLSIDFromString() failed.");
		break;
	}
	return clsid;
}


bool VerifyStringToCLSID(const std::wstring &str, CLSID &clsid)
{
	bool result = false;
	clsid = CLSID();
	std::vector<OLECHAR> tmp(str.c_str(), str.c_str() + str.length() + 1);
	switch(::CLSIDFromString(tmp.data(), &clsid))
	{
	case NOERROR:
		result = true;
		break;
	case E_INVALIDARG:
		result = false;
		break;
	case CO_E_CLASSSTRING:
		result = false;
		break;
	case REGDB_E_CLASSNOTREG:
		result = false;
		break;
	case REGDB_E_READREGDB:
		throw std::runtime_error("CLSIDFromString() failed: REGDB_E_READREGDB.");
		break;
	default:
		throw std::logic_error("CLSIDFromString() failed.");
		break;
	}
	return result;
}


bool IsCLSID(const std::wstring &str)
{
	bool result = false;
	CLSID clsid = CLSID();
	std::vector<OLECHAR> tmp(str.c_str(), str.c_str() + str.length() + 1);
	switch(::CLSIDFromString(tmp.data(), &clsid))
	{
	case NOERROR:
		result = true;
		break;
	case E_INVALIDARG:
		result = false;
		break;
	case CO_E_CLASSSTRING:
		result = false;
		break;
	case REGDB_E_CLASSNOTREG:
		result = false;
		break;
	case REGDB_E_READREGDB:
		result = false;
		throw std::runtime_error("CLSIDFromString() failed: REGDB_E_READREGDB.");
		break;
	default:
		result = false;
		throw std::logic_error("CLSIDFromString() failed.");
		break;
	}
	return result;
}


std::wstring IIDToString(IID iid)
{
	std::wstring str;
	LPOLESTR tmp = nullptr;
	switch(::StringFromIID(iid, &tmp))
	{
	case S_OK:
		break;
	case E_OUTOFMEMORY:
		if(tmp)
		{
			::CoTaskMemFree(tmp);
			tmp = nullptr;
		}
		MPT_EXCEPTION_THROW_OUT_OF_MEMORY();
		break;
	default:
		if(tmp)
		{
			::CoTaskMemFree(tmp);
			tmp = nullptr;
		}
		throw std::logic_error("StringFromIID() failed.");
		break;
	}
	if(!tmp)
	{
		throw std::logic_error("StringFromIID() failed.");
	}
	try
	{
		str = tmp;
	} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
	{
		::CoTaskMemFree(tmp);
		tmp = nullptr;
		MPT_UNUSED_VARIABLE(e);
		MPT_EXCEPTION_RETHROW_OUT_OF_MEMORY();
	}
	return str;
}


IID StringToIID(const std::wstring &str)
{
	IID iid = IID();
	std::vector<OLECHAR> tmp(str.c_str(), str.c_str() + str.length() + 1);
	switch(::IIDFromString(tmp.data(), &iid))
	{
	case S_OK:
		// nothing
		break;
	case E_OUTOFMEMORY:
		iid = IID();
		MPT_EXCEPTION_THROW_OUT_OF_MEMORY();
		break;
	case E_INVALIDARG:
		iid = IID();
		break;
	default:
		iid = IID();
		throw std::logic_error("IIDFromString() failed.");
		break;
	}
	return iid;
}


std::wstring GUIDToString(GUID guid)
{
	std::vector<OLECHAR> tmp(256);
	if(::StringFromGUID2(guid, tmp.data(), static_cast<int>(tmp.size())) <= 0)
	{
		throw std::logic_error("StringFromGUID2() failed.");
	}
	return tmp.data();
}


GUID StringToGUID(const std::wstring &str)
{
	return StringToIID(str);
}


GUID CreateGUID()
{
	GUID guid = GUID();
	switch(::CoCreateGuid(&guid))
	{
	case S_OK:
		// nothing
		break;
	default:
		guid = GUID();
		throw std::runtime_error("CoCreateGuid() failed.");
	}
	return guid;
}


#if !MPT_OS_WINDOWS_WINRT

UUID StringToUUID(const mpt::ustring &str)
{
	UUID uuid = UUID();
	std::wstring wstr = mpt::ToWide(str);
	std::vector<wchar_t> tmp(wstr.c_str(), wstr.c_str() + wstr.length() + 1);
	switch(::UuidFromStringW((RPC_WSTR)(&(tmp[0])), &uuid))
	{
	case RPC_S_OK:
		// nothing
		break;
	case RPC_S_INVALID_STRING_UUID:
		uuid = UUID();
		break;
	default:
		throw std::logic_error("UuidFromStringW() failed.");
		break;
	}
	return uuid;
}


mpt::ustring UUIDToString(UUID uuid)
{
	std::wstring wstr;
	RPC_WSTR tmp = nullptr;
	switch(::UuidToStringW(&uuid, &tmp))
	{
	case RPC_S_OK:
		// nothing
		break;
	case RPC_S_OUT_OF_MEMORY:
		if(tmp)
		{
			::RpcStringFreeW(&tmp);
			tmp = nullptr;
		}
		MPT_EXCEPTION_THROW_OUT_OF_MEMORY();
		break;
	default:
		throw std::logic_error("UuidToStringW() failed.");
		break;
	}
	try
	{
		std::size_t len = 0;
		for(len = 0; tmp[len] != 0; ++len)
		{
			// nothing
		}
		wstr = std::wstring(tmp, tmp + len);
	} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
	{
		::RpcStringFreeW(&tmp);
		tmp = nullptr;
		MPT_UNUSED_VARIABLE(e);
		MPT_EXCEPTION_RETHROW_OUT_OF_MEMORY();
	}
	return mpt::ToUnicode(wstr);
}

#endif // !MPT_OS_WINDOWS_WINRT


bool IsValid(UUID uuid)
{
	return false
		|| uuid.Data1 != 0
		|| uuid.Data2 != 0
		|| uuid.Data3 != 0
		|| uuid.Data4[0] != 0
		|| uuid.Data4[1] != 0
		|| uuid.Data4[2] != 0
		|| uuid.Data4[3] != 0
		|| uuid.Data4[4] != 0
		|| uuid.Data4[5] != 0
		|| uuid.Data4[6] != 0
		|| uuid.Data4[7] != 0
		;
}


#endif // MODPLUG_TRACKER || !NO_DMO


} // namespace Util


#endif // MPT_OS_WINDOWS


namespace mpt
{

#if MPT_OS_WINDOWS

mpt::UUID UUIDFromWin32(::UUID uuid)
{
	return mpt::UUID
		( uuid.Data1
		, uuid.Data2
		, uuid.Data3
		, (static_cast<uint64>(0)
			| (static_cast<uint64>(uuid.Data4[0]) << 56)
			| (static_cast<uint64>(uuid.Data4[1]) << 48)
			| (static_cast<uint64>(uuid.Data4[2]) << 40)
			| (static_cast<uint64>(uuid.Data4[3]) << 32)
			| (static_cast<uint64>(uuid.Data4[4]) << 24)
			| (static_cast<uint64>(uuid.Data4[5]) << 16)
			| (static_cast<uint64>(uuid.Data4[6]) <<  8)
			| (static_cast<uint64>(uuid.Data4[7]) <<  0)
			)
		);
}

::UUID UUIDToWin32(mpt::UUID uuid)
{
	::UUID result = ::UUID();
	result.Data1 = uuid.GetData1();
	result.Data2 = uuid.GetData2();
	result.Data3 = uuid.GetData3();
	result.Data4[0] = static_cast<uint8>(uuid.GetData4() >> 56);
	result.Data4[1] = static_cast<uint8>(uuid.GetData4() >> 48);
	result.Data4[2] = static_cast<uint8>(uuid.GetData4() >> 40);
	result.Data4[3] = static_cast<uint8>(uuid.GetData4() >> 32);
	result.Data4[4] = static_cast<uint8>(uuid.GetData4() >> 24);
	result.Data4[5] = static_cast<uint8>(uuid.GetData4() >> 16);
	result.Data4[6] = static_cast<uint8>(uuid.GetData4() >>  8);
	result.Data4[7] = static_cast<uint8>(uuid.GetData4() >>  0);
	return result;
}

#if defined(MODPLUG_TRACKER) || !defined(NO_DMO)

UUID::UUID(::UUID uuid)
{
	*this = UUIDFromWin32(uuid);
}

UUID::operator ::UUID () const
{
	return UUIDToWin32(*this);
}

mpt::UUID UUID::FromGroups(uint32 group1, uint16 group2, uint16 group3, uint16 group4, uint64 group5)
{
	MPT_ASSERT((group5 & 0xffff000000000000ull) == 0ull);
	return mpt::UUID
		( group1
		, group2
		, group3
		, (static_cast<uint64>(group4) << 48) | group5
		);
}

#endif // MODPLUG_TRACKER || !NO_DMO

#endif // MPT_OS_WINDOWS

UUID UUID::Generate()
{
	#if MPT_OS_WINDOWS && MPT_OS_WINDOWS_WINRT
		#if (_WIN32_WINNT >= 0x0602)
			::GUID guid = ::GUID();
			HRESULT result = CoCreateGuid(&guid);
			if(result != S_OK)
			{
				return mpt::UUID::RFC4122Random();
			}
			return mpt::UUIDFromWin32(guid);
		#else
			return mpt::UUID::RFC4122Random();
		#endif
	#elif MPT_OS_WINDOWS && !MPT_OS_WINDOWS_WINRT
		::UUID uuid = ::UUID();
		RPC_STATUS status = ::UuidCreate(&uuid);
		if(status != RPC_S_OK && status != RPC_S_UUID_LOCAL_ONLY)
		{
			return mpt::UUID::RFC4122Random();
		}
		status = RPC_S_OK;
		if(UuidIsNil(&uuid, &status) != FALSE)
		{
			return mpt::UUID::RFC4122Random();
		}
		if(status != RPC_S_OK)
		{
			return mpt::UUID::RFC4122Random();
		}
		return mpt::UUIDFromWin32(uuid);
	#else
		return RFC4122Random();
	#endif
}

UUID UUID::GenerateLocalUseOnly()
{
	#if MPT_OS_WINDOWS && MPT_OS_WINDOWS_WINRT
		#if (_WIN32_WINNT >= 0x0602)
			::GUID guid = ::GUID();
			HRESULT result = CoCreateGuid(&guid);
			if(result != S_OK)
			{
				return mpt::UUID::RFC4122Random();
			}
			return mpt::UUIDFromWin32(guid);
		#else
			return mpt::UUID::RFC4122Random();
		#endif
	#elif MPT_OS_WINDOWS && !MPT_OS_WINDOWS_WINRT
		#if _WIN32_WINNT >= 0x0501
			// Available since Win2000, but we check for WinXP in order to not use this
			// function in Win32old builds. It is not available on some non-fully
			// patched Win98SE installs in the wild.
			::UUID uuid = ::UUID();
			RPC_STATUS status = ::UuidCreateSequential(&uuid);
			if(status != RPC_S_OK && status != RPC_S_UUID_LOCAL_ONLY)
			{
				return Generate();
			}
			status = RPC_S_OK;
			if(UuidIsNil(&uuid, &status) != FALSE)
			{
				return mpt::UUID::RFC4122Random();
			}
			if(status != RPC_S_OK)
			{
				return mpt::UUID::RFC4122Random();
			}
			return mpt::UUIDFromWin32(uuid);
		#else
			// Fallback to ::UuidCreate is safe as ::UuidCreateSequential is only a
			// tiny performance optimization.
			return Generate();
		#endif
	#else
		return RFC4122Random();
	#endif
}

UUID UUID::RFC4122Random()
{
	UUID result;
	mpt::thread_safe_prng<mpt::best_prng> & prng = mpt::global_prng();
	result.Data1 = mpt::random<uint32>(prng);
	result.Data2 = mpt::random<uint16>(prng);
	result.Data3 = mpt::random<uint16>(prng);
	result.Data4 = mpt::random<uint64>(prng);
	result.MakeRFC4122(4);
	return result;
}

uint32 UUID::GetData1() const
{
	return Data1;
}

uint16 UUID::GetData2() const
{
	return Data2;
}

uint16 UUID::GetData3() const
{
	return Data3;
}

uint64 UUID::GetData4() const
{
	return Data4;
}

bool UUID::IsNil() const
{
	return (Data1 == 0) && (Data2 == 0) && (Data3 == 0) && (Data4 == 0);
}

bool UUID::IsValid() const
{
	return (Data1 != 0) || (Data2 != 0) || (Data3 != 0) || (Data4 != 0);
}

uint8 UUID::Mm() const
{
	return static_cast<uint8>((Data3 >> 8) & 0xffu);
}

uint8 UUID::Nn() const
{
	return static_cast<uint8>((Data4 >> 56) & 0xffu);
}

uint8 UUID::Variant() const
{
	return Nn() >> 4u;
}

uint8 UUID::Version() const
{
	return Mm() >> 4u;
}

bool UUID::IsRFC4122() const
{
	return (Variant() & 0xcu) == 0x8u;
}

void UUID::MakeRFC4122(uint8 version)
{
	// variant
	uint8 Nn = static_cast<uint8>((Data4 >> 56) & 0xffu);
	Data4 &= 0x00ffffffffffffffull;
	Nn &= ~(0xc0u);
	Nn |= 0x80u;
	Data4 |= static_cast<uint64>(Nn) << 56;
	// version
	version &= 0x0fu;
	uint8 Mm = static_cast<uint8>((Data3 >> 8) & 0xffu);
	Data3 &= 0x00ffu;
	Mm &= ~(0xf0u);
	Mm |= (version << 4u);
	Data3 |= static_cast<uint16>(Mm) << 8;
}

UUID::UUID()
{
	Data1 = 0;
	Data2 = 0;
	Data3 = 0;
	Data4 = 0;
}

UUID::UUID(uint32 Data1, uint16 Data2, uint16 Data3, uint64 Data4)
{
	this->Data1 = Data1;
	this->Data2 = Data2;
	this->Data3 = Data3;
	this->Data4 = Data4;
}

bool operator==(const mpt::UUID & a, const mpt::UUID & b)
{
	return (a.Data1 == b.Data1) && (a.Data2 == b.Data2) && (a.Data3 == b.Data3) && (a.Data4 == b.Data4);
}

bool operator!=(const mpt::UUID & a, const mpt::UUID & b)
{
	return (a.Data1 != b.Data1) || (a.Data2 != b.Data2) || (a.Data3 != b.Data3) || (a.Data4 != b.Data4);
}

UUID UUID::FromString(const std::string &str)
{
	std::vector<std::string> segments = mpt::String::Split<std::string>(str, std::string("-"));
	if(segments.size() != 5)
	{
		return UUID();
	}
	if(segments[0].length() != 8)
	{
		return UUID();
	}
	if(segments[1].length() != 4)
	{
		return UUID();
	}
	if(segments[2].length() != 4)
	{
		return UUID();
	}
	if(segments[3].length() != 4)
	{
		return UUID();
	}
	if(segments[4].length() != 12)
	{
		return UUID();
	}
	UUID result;
	result.Data1 = mpt::String::Parse::Hex<uint32>(segments[0]);
	result.Data2 = mpt::String::Parse::Hex<uint16>(segments[1]);
	result.Data3 = mpt::String::Parse::Hex<uint16>(segments[2]);
	result.Data4 = mpt::String::Parse::Hex<uint64>(segments[3] + segments[4]);
	return result;
}

UUID UUID::FromString(const mpt::ustring &str)
{
	std::vector<mpt::ustring> segments = mpt::String::Split<mpt::ustring>(str, MPT_USTRING("-"));
	if(segments.size() != 5)
	{
		return UUID();
	}
	if(segments[0].length() != 8)
	{
		return UUID();
	}
	if(segments[1].length() != 4)
	{
		return UUID();
	}
	if(segments[2].length() != 4)
	{
		return UUID();
	}
	if(segments[3].length() != 4)
	{
		return UUID();
	}
	if(segments[4].length() != 12)
	{
		return UUID();
	}
	UUID result;
	result.Data1 = mpt::String::Parse::Hex<uint32>(segments[0]);
	result.Data2 = mpt::String::Parse::Hex<uint16>(segments[1]);
	result.Data3 = mpt::String::Parse::Hex<uint16>(segments[2]);
	result.Data4 = mpt::String::Parse::Hex<uint64>(segments[3] + segments[4]);
	return result;
}

std::string UUID::ToString() const
{
	return std::string()
		+ mpt::fmt::hex0<8>(GetData1())
		+ std::string("-")
		+ mpt::fmt::hex0<4>(GetData2())
		+ std::string("-")
		+ mpt::fmt::hex0<4>(GetData3())
		+ std::string("-")
		+ mpt::fmt::hex0<4>(static_cast<uint16>(GetData4() >> 48))
		+ std::string("-")
		+ mpt::fmt::hex0<4>(static_cast<uint16>(GetData4() >> 32))
		+ mpt::fmt::hex0<8>(static_cast<uint32>(GetData4() >>  0))
		;
}

mpt::ustring UUID::ToUString() const
{
	return mpt::ustring()
		+ mpt::ufmt::hex0<8>(GetData1())
		+ MPT_USTRING("-")
		+ mpt::ufmt::hex0<4>(GetData2())
		+ MPT_USTRING("-")
		+ mpt::ufmt::hex0<4>(GetData3())
		+ MPT_USTRING("-")
		+ mpt::ufmt::hex0<4>(static_cast<uint16>(GetData4() >> 48))
		+ MPT_USTRING("-")
		+ mpt::ufmt::hex0<4>(static_cast<uint16>(GetData4() >> 32))
		+ mpt::ufmt::hex0<8>(static_cast<uint32>(GetData4() >>  0))
		;
}

UUID::UUID(GUIDms guid)
{
	Data1 = guid.Data1.get();
	Data2 = guid.Data2.get();
	Data3 = guid.Data3.get();
	Data4 = guid.Data4.get();
}

UUID::operator GUIDms() const
{
	GUIDms result;
	result.Data1 = GetData1();
	result.Data2 = GetData2();
	result.Data3 = GetData3();
	result.Data4 = GetData4();
	return result;
}


} // namespace mpt


OPENMPT_NAMESPACE_END
