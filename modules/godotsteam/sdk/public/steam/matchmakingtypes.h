//========= Copyright � 1996-2008, Valve LLC, All rights reserved. ============
//
// Purpose: 
//
// $NoKeywords: $
//=============================================================================

#ifndef MATCHMAKINGTYPES_H
#define MATCHMAKINGTYPES_H

#ifdef _WIN32
#pragma once
#endif

#ifdef POSIX
#ifndef _snprintf
#define _snprintf snprintf
#endif
#endif

#include <stdio.h>
#include <string.h>

//
// Max size (in bytes of UTF-8 data, not in characters) of server fields, including null terminator.
// WARNING: These cannot be changed easily, without breaking clients using old interfaces.
//
const int k_cbMaxGameServerGameDir = 32;
const int k_cbMaxGameServerMapName = 32;
const int k_cbMaxGameServerGameDescription = 64;
const int k_cbMaxGameServerName = 64;
const int k_cbMaxGameServerTags = 128;
const int k_cbMaxGameServerGameData = 2048;

/// Store key/value pair used in matchmaking queries.
///
/// Actually, the name Key/Value is a bit misleading.  The "key" is better
/// understood as "filter operation code" and the "value" is the operand to this
/// filter operation.  The meaning of the operand depends upon the filter.
struct MatchMakingKeyValuePair_t
{
	MatchMakingKeyValuePair_t() { m_szKey[0] = m_szValue[0] = 0; }
	MatchMakingKeyValuePair_t( const char *pchKey, const char *pchValue )
	{
		strncpy( m_szKey, pchKey, sizeof(m_szKey) ); // this is a public header, use basic c library string funcs only!
		m_szKey[ sizeof( m_szKey ) - 1 ] = '\0';
		strncpy( m_szValue, pchValue, sizeof(m_szValue) );
		m_szValue[ sizeof( m_szValue ) - 1 ] = '\0';
	}
	char m_szKey[ 256 ];
	char m_szValue[ 256 ];
};


enum EMatchMakingServerResponse
{
	eServerResponded = 0,
	eServerFailedToRespond,
	eNoServersListedOnMasterServer // for the Internet query type, returned in response callback if no servers of this type match
};

// servernetadr_t is all the addressing info the serverbrowser needs to know about a game server,
// namely: its IP, its connection port, and its query port.
class servernetadr_t 
{
public:

	servernetadr_t() : m_usConnectionPort( 0 ), m_usQueryPort( 0 ), m_unIP( 0 ) {}
	
	void	Init( unsigned int ip, uint16 usQueryPort, uint16 usConnectionPort );
#ifdef NETADR_H
	netadr_t	GetIPAndQueryPort();
#endif
	
	// Access the query port.
	uint16	GetQueryPort() const;
	void	SetQueryPort( uint16 usPort );

	// Access the connection port.
	uint16	GetConnectionPort() const;
	void	SetConnectionPort( uint16 usPort );

	// Access the IP
	uint32 GetIP() const;
	void SetIP( uint32 unIP );

	// This gets the 'a.b.c.d:port' string with the connection port (instead of the query port).
	const char *GetConnectionAddressString() const;
	const char *GetQueryAddressString() const;

	// Comparison operators and functions.
	bool	operator<(const servernetadr_t &netadr) const;
	void operator=( const servernetadr_t &that )
	{
		m_usConnectionPort = that.m_usConnectionPort;
		m_usQueryPort = that.m_usQueryPort;
		m_unIP = that.m_unIP;
	}

	
private:
	const char *ToString( uint32 unIP, uint16 usPort ) const;
	uint16	m_usConnectionPort;	// (in HOST byte order)
	uint16	m_usQueryPort;
	uint32  m_unIP;
};


inline void	servernetadr_t::Init( unsigned int ip, uint16 usQueryPort, uint16 usConnectionPort )
{
	m_unIP = ip;
	m_usQueryPort = usQueryPort;
	m_usConnectionPort = usConnectionPort;
}

#ifdef NETADR_H
inline netadr_t servernetadr_t::GetIPAndQueryPort()
{
	return netadr_t( m_unIP, m_usQueryPort );
}
#endif

inline uint16 servernetadr_t::GetQueryPort() const
{
	return m_usQueryPort;
}

inline void	servernetadr_t::SetQueryPort( uint16 usPort )
{
	m_usQueryPort = usPort;
}

inline uint16 servernetadr_t::GetConnectionPort() const
{
	return m_usConnectionPort;
}

inline void	servernetadr_t::SetConnectionPort( uint16 usPort )
{
	m_usConnectionPort = usPort;
}

inline uint32 servernetadr_t::GetIP() const
{
	return m_unIP;
}

inline void	servernetadr_t::SetIP( uint32 unIP )
{
	m_unIP = unIP;
}

inline const char *servernetadr_t::ToString( uint32 unIP, uint16 usPort ) const
{
	static char s[4][64];
	static int nBuf = 0;
	unsigned char *ipByte = (unsigned char *)&unIP;
#ifdef VALVE_BIG_ENDIAN
	_snprintf (s[nBuf], sizeof( s[nBuf] ), "%u.%u.%u.%u:%i", (int)(ipByte[0]), (int)(ipByte[1]), (int)(ipByte[2]), (int)(ipByte[3]), usPort );
#else
	_snprintf (s[nBuf], sizeof( s[nBuf] ), "%u.%u.%u.%u:%i", (int)(ipByte[3]), (int)(ipByte[2]), (int)(ipByte[1]), (int)(ipByte[0]), usPort );
#endif
	const char *pchRet = s[nBuf];
	++nBuf;
	nBuf %= ( (sizeof(s)/sizeof(s[0])) );
	return pchRet;
}

inline const char* servernetadr_t::GetConnectionAddressString() const
{
	return ToString( m_unIP, m_usConnectionPort );
}

inline const char* servernetadr_t::GetQueryAddressString() const
{
	return ToString( m_unIP, m_usQueryPort );	
}

inline bool servernetadr_t::operator<(const servernetadr_t &netadr) const
{
	return ( m_unIP < netadr.m_unIP ) || ( m_unIP == netadr.m_unIP && m_usQueryPort < netadr.m_usQueryPort );
}

//-----------------------------------------------------------------------------
// Purpose: Data describing a single server
//-----------------------------------------------------------------------------
class gameserveritem_t
{
public:
	gameserveritem_t();

	const char* GetName() const;
	void SetName( const char *pName );

public:
	servernetadr_t m_NetAdr;									///< IP/Query Port/Connection Port for this server
	int m_nPing;												///< current ping time in milliseconds
	bool m_bHadSuccessfulResponse;								///< server has responded successfully in the past
	bool m_bDoNotRefresh;										///< server is marked as not responding and should no longer be refreshed
	char m_szGameDir[k_cbMaxGameServerGameDir];					///< current game directory
	char m_szMap[k_cbMaxGameServerMapName];						///< current map
	char m_szGameDescription[k_cbMaxGameServerGameDescription];	///< game description
	uint32 m_nAppID;											///< Steam App ID of this server
	int m_nPlayers;												///< total number of players currently on the server.  INCLUDES BOTS!!
	int m_nMaxPlayers;											///< Maximum players that can join this server
	int m_nBotPlayers;											///< Number of bots (i.e simulated players) on this server
	bool m_bPassword;											///< true if this server needs a password to join
	bool m_bSecure;												///< Is this server protected by VAC
	uint32 m_ulTimeLastPlayed;									///< time (in unix time) when this server was last played on (for favorite/history servers)
	int	m_nServerVersion;										///< server version as reported to Steam

private:

	/// Game server name
	char m_szServerName[k_cbMaxGameServerName];

	// For data added after SteamMatchMaking001 add it here
public:
	/// the tags this server exposes
	char m_szGameTags[k_cbMaxGameServerTags];

	/// steamID of the game server - invalid if it's doesn't have one (old server, or not connected to Steam)
	CSteamID m_steamID;
};


inline gameserveritem_t::gameserveritem_t()
{
	m_szGameDir[0] = m_szMap[0] = m_szGameDescription[0] = m_szServerName[0] = 0;
	m_bHadSuccessfulResponse = m_bDoNotRefresh = m_bPassword = m_bSecure = false;
	m_nPing = m_nAppID = m_nPlayers = m_nMaxPlayers = m_nBotPlayers = m_ulTimeLastPlayed = m_nServerVersion = 0;
	m_szGameTags[0] = 0;
}

inline const char* gameserveritem_t::GetName() const
{
	// Use the IP address as the name if nothing is set yet.
	if ( m_szServerName[0] == 0 )
		return m_NetAdr.GetConnectionAddressString();
	else
		return m_szServerName;
}

inline void gameserveritem_t::SetName( const char *pName )
{
	strncpy( m_szServerName, pName, sizeof( m_szServerName ) );
	m_szServerName[ sizeof( m_szServerName ) - 1 ] = '\0';
}


#endif // MATCHMAKINGTYPES_H
