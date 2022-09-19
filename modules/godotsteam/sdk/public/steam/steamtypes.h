//========= Copyright © 1996-2008, Valve LLC, All rights reserved. ============
//
// Purpose:
//
//=============================================================================

#ifndef STEAMTYPES_H
#define STEAMTYPES_H
#ifdef _WIN32
#pragma once
#endif

#define S_CALLTYPE __cdecl

// Steam-specific types. Defined here so this header file can be included in other code bases.
#ifndef WCHARTYPES_H
typedef unsigned char uint8;
#endif

#if defined( __GNUC__ ) && !defined(_WIN32) && !defined(POSIX)
	#if __GNUC__ < 4
		#error "Steamworks requires GCC 4.X (4.2 or 4.4 have been tested)"
	#endif
	#define POSIX 1
#endif

#if defined(__LP64__) || defined(__x86_64__) || defined(_WIN64) || defined(__aarch64__) || defined(__s390x__)
#define X64BITS
#endif

#if !defined(VALVE_BIG_ENDIAN)
#if defined(_PS3)
// Make sure VALVE_BIG_ENDIAN gets set on PS3, may already be set previously in Valve internal code.
#define VALVE_BIG_ENDIAN 1
#endif
#if defined( __GNUC__ ) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define VALVE_BIG_ENDIAN 1
#endif
#endif

typedef unsigned char uint8;
typedef signed char int8;

#if defined( _WIN32 ) && !defined( __GNUC__ )

typedef __int16 int16;
typedef unsigned __int16 uint16;
typedef __int32 int32;
typedef unsigned __int32 uint32;
typedef __int64 int64;
typedef unsigned __int64 uint64;

typedef int64 lint64;
typedef uint64 ulint64;

#ifdef X64BITS
typedef __int64 intp;				// intp is an integer that can accomodate a pointer
typedef unsigned __int64 uintp;		// (ie, sizeof(intp) >= sizeof(int) && sizeof(intp) >= sizeof(void *)
#else
typedef __int32 intp;
typedef unsigned __int32 uintp;
#endif

#else // _WIN32

typedef short int16;
typedef unsigned short uint16;
typedef int int32;
typedef unsigned int uint32;
typedef long long int64;
typedef unsigned long long uint64;

// [u]int64 are actually defined as 'long long' and gcc 64-bit
// doesn't automatically consider them the same as 'long int'.
// Changing the types for [u]int64 is complicated by
// there being many definitions, so we just
// define a 'long int' here and use it in places that would
// otherwise confuse the compiler.
typedef long int lint64;
typedef unsigned long int ulint64;

#ifdef X64BITS
typedef long long intp;
typedef unsigned long long uintp;
#else
typedef int intp;
typedef unsigned int uintp;
#endif

#endif // else _WIN32

typedef uint32 AppId_t;
const AppId_t k_uAppIdInvalid = 0x0;

// AppIds and DepotIDs also presently share the same namespace
typedef uint32 DepotId_t;
const DepotId_t k_uDepotIdInvalid = 0x0;

// RTime32.  Seconds elapsed since Jan 1 1970, i.e. unix timestamp.
// It's the same as time_t, but it is always 32-bit and unsigned.  
typedef uint32 RTime32;

// handle to a Steam API call
typedef uint64 SteamAPICall_t;
const SteamAPICall_t k_uAPICallInvalid = 0x0;

typedef uint32 AccountID_t;

// Party Beacon ID
typedef uint64 PartyBeaconID_t;
const PartyBeaconID_t k_ulPartyBeaconIdInvalid = 0;

enum ESteamIPType
{
	k_ESteamIPTypeIPv4 = 0,
	k_ESteamIPTypeIPv6 = 1,
};

#pragma pack( push, 1 )

struct SteamIPAddress_t
{
	union {

		uint32			m_unIPv4;		// Host order
		uint8			m_rgubIPv6[16];		// Network order! Same as inaddr_in6.  (0011:2233:4455:6677:8899:aabb:ccdd:eeff)

		// Internal use only
		uint64			m_ipv6Qword[2];	// big endian
	};

	ESteamIPType m_eType;

	bool IsSet() const 
	{ 
		if ( k_ESteamIPTypeIPv4 == m_eType )
		{
			return m_unIPv4 != 0;
		}
		else 
		{
			return m_ipv6Qword[0] !=0 || m_ipv6Qword[1] != 0; 
		}
	}

	static SteamIPAddress_t IPv4Any()
	{
		SteamIPAddress_t ipOut;
		ipOut.m_eType = k_ESteamIPTypeIPv4;
		ipOut.m_unIPv4 = 0;

		return ipOut;
	}

	static SteamIPAddress_t IPv6Any()
	{
		SteamIPAddress_t ipOut;
		ipOut.m_eType = k_ESteamIPTypeIPv6;
		ipOut.m_ipv6Qword[0] = 0;
		ipOut.m_ipv6Qword[1] = 0;

		return ipOut;
	}

	static SteamIPAddress_t IPv4Loopback()
	{
		SteamIPAddress_t ipOut;
		ipOut.m_eType = k_ESteamIPTypeIPv4;
		ipOut.m_unIPv4 = 0x7f000001;

		return ipOut;
	}

	static SteamIPAddress_t IPv6Loopback()
	{
		SteamIPAddress_t ipOut;
		ipOut.m_eType = k_ESteamIPTypeIPv6;
		ipOut.m_ipv6Qword[0] = 0;
		ipOut.m_ipv6Qword[1] = 0;
		ipOut.m_rgubIPv6[15] = 1;

		return ipOut;
	}
};

#pragma pack( pop )

#endif // STEAMTYPES_H
