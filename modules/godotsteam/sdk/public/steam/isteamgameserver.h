//====== Copyright (c) 1996-2008, Valve Corporation, All rights reserved. =======
//
// Purpose: interface to steam for game servers
//
//=============================================================================

#ifndef ISTEAMGAMESERVER_H
#define ISTEAMGAMESERVER_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"

//-----------------------------------------------------------------------------
// Purpose: Functions for authenticating users via Steam to play on a game server
//-----------------------------------------------------------------------------
class ISteamGameServer
{
public:

//
// Basic server data.  These properties, if set, must be set before before calling LogOn.  They
// may not be changed after logged in.
//

	/// This is called by SteamGameServer_Init, and you will usually not need to call it directly
	STEAM_PRIVATE_API( virtual bool InitGameServer( uint32 unIP, uint16 usGamePort, uint16 usQueryPort, uint32 unFlags, AppId_t nGameAppId, const char *pchVersionString ) = 0; )

	/// Game product identifier.  This is currently used by the master server for version checking purposes.
	/// It's a required field, but will eventually will go away, and the AppID will be used for this purpose.
	virtual void SetProduct( const char *pszProduct ) = 0;

	/// Description of the game.  This is a required field and is displayed in the steam server browser....for now.
	/// This is a required field, but it will go away eventually, as the data should be determined from the AppID.
	virtual void SetGameDescription( const char *pszGameDescription ) = 0;

	/// If your game is a "mod," pass the string that identifies it.  The default is an empty string, meaning
	/// this application is the original game, not a mod.
	///
	/// @see k_cbMaxGameServerGameDir
	virtual void SetModDir( const char *pszModDir ) = 0;

	/// Is this is a dedicated server?  The default value is false.
	virtual void SetDedicatedServer( bool bDedicated ) = 0;

//
// Login
//

	/// Begin process to login to a persistent game server account
	///
	/// You need to register for callbacks to determine the result of this operation.
	/// @see SteamServersConnected_t
	/// @see SteamServerConnectFailure_t
	/// @see SteamServersDisconnected_t
	virtual void LogOn( const char *pszToken ) = 0;

	/// Login to a generic, anonymous account.
	///
	/// Note: in previous versions of the SDK, this was automatically called within SteamGameServer_Init,
	/// but this is no longer the case.
	virtual void LogOnAnonymous() = 0;

	/// Begin process of logging game server out of steam
	virtual void LogOff() = 0;
	
	// status functions
	virtual bool BLoggedOn() = 0;
	virtual bool BSecure() = 0; 
	virtual CSteamID GetSteamID() = 0;

	/// Returns true if the master server has requested a restart.
	/// Only returns true once per request.
	virtual bool WasRestartRequested() = 0;

//
// Server state.  These properties may be changed at any time.
//

	/// Max player count that will be reported to server browser and client queries
	virtual void SetMaxPlayerCount( int cPlayersMax ) = 0;

	/// Number of bots.  Default value is zero
	virtual void SetBotPlayerCount( int cBotplayers ) = 0;

	/// Set the name of server as it will appear in the server browser
	///
	/// @see k_cbMaxGameServerName
	virtual void SetServerName( const char *pszServerName ) = 0;

	/// Set name of map to report in the server browser
	///
	/// @see k_cbMaxGameServerMapName
	virtual void SetMapName( const char *pszMapName ) = 0;

	/// Let people know if your server will require a password
	virtual void SetPasswordProtected( bool bPasswordProtected ) = 0;

	/// Spectator server port to advertise.  The default value is zero, meaning the
	/// service is not used.  If your server receives any info requests on the LAN,
	/// this is the value that will be placed into the reply for such local queries.
	///
	/// This is also the value that will be advertised by the master server.
	/// The only exception is if your server is using a FakeIP.  Then then the second
	/// fake port number (index 1) assigned to your server will be listed on the master
	/// server as the spectator port, if you set this value to any nonzero value.
	///
	/// This function merely controls the values that are advertised -- it's up to you to
	/// configure the server to actually listen on this port and handle any spectator traffic
	virtual void SetSpectatorPort( uint16 unSpectatorPort ) = 0;

	/// Name of the spectator server.  (Only used if spectator port is nonzero.)
	///
	/// @see k_cbMaxGameServerMapName
	virtual void SetSpectatorServerName( const char *pszSpectatorServerName ) = 0;

	/// Call this to clear the whole list of key/values that are sent in rules queries.
	virtual void ClearAllKeyValues() = 0;
	
	/// Call this to add/update a key/value pair.
	virtual void SetKeyValue( const char *pKey, const char *pValue ) = 0;

	/// Sets a string defining the "gametags" for this server, this is optional, but if it is set
	/// it allows users to filter in the matchmaking/server-browser interfaces based on the value
	///
	/// @see k_cbMaxGameServerTags
	virtual void SetGameTags( const char *pchGameTags ) = 0;

	/// Sets a string defining the "gamedata" for this server, this is optional, but if it is set
	/// it allows users to filter in the matchmaking/server-browser interfaces based on the value
	///
	/// @see k_cbMaxGameServerGameData
	virtual void SetGameData( const char *pchGameData ) = 0;

	/// Region identifier.  This is an optional field, the default value is empty, meaning the "world" region
	virtual void SetRegion( const char *pszRegion ) = 0;

	/// Indicate whether you wish to be listed on the master server list
	/// and/or respond to server browser / LAN discovery packets.
	/// The server starts with this value set to false.  You should set all
	/// relevant server parameters before enabling advertisement on the server.
	///
	/// (This function used to be named EnableHeartbeats, so if you are wondering
	/// where that function went, it's right here.  It does the same thing as before,
	/// the old name was just confusing.)
	virtual void SetAdvertiseServerActive( bool bActive ) = 0;

//
// Player list management / authentication.
//

	// Retrieve ticket to be sent to the entity who wishes to authenticate you ( using BeginAuthSession API ). 
	// pcbTicket retrieves the length of the actual ticket.
	virtual HAuthTicket GetAuthSessionTicket( void *pTicket, int cbMaxTicket, uint32 *pcbTicket ) = 0;

	// Authenticate ticket ( from GetAuthSessionTicket ) from entity steamID to be sure it is valid and isnt reused
	// Registers for callbacks if the entity goes offline or cancels the ticket ( see ValidateAuthTicketResponse_t callback and EAuthSessionResponse )
	virtual EBeginAuthSessionResult BeginAuthSession( const void *pAuthTicket, int cbAuthTicket, CSteamID steamID ) = 0;

	// Stop tracking started by BeginAuthSession - called when no longer playing game with this entity
	virtual void EndAuthSession( CSteamID steamID ) = 0;

	// Cancel auth ticket from GetAuthSessionTicket, called when no longer playing game with the entity you gave the ticket to
	virtual void CancelAuthTicket( HAuthTicket hAuthTicket ) = 0;

	// After receiving a user's authentication data, and passing it to SendUserConnectAndAuthenticate, use this function
	// to determine if the user owns downloadable content specified by the provided AppID.
	virtual EUserHasLicenseForAppResult UserHasLicenseForApp( CSteamID steamID, AppId_t appID ) = 0;

	// Ask if a user in in the specified group, results returns async by GSUserGroupStatus_t
	// returns false if we're not connected to the steam servers and thus cannot ask
	virtual bool RequestUserGroupStatus( CSteamID steamIDUser, CSteamID steamIDGroup ) = 0;


	// these two functions s are deprecated, and will not return results
	// they will be removed in a future version of the SDK
	virtual void GetGameplayStats( ) = 0;
	STEAM_CALL_RESULT( GSReputation_t )
	virtual SteamAPICall_t GetServerReputation() = 0;

	// Returns the public IP of the server according to Steam, useful when the server is 
	// behind NAT and you want to advertise its IP in a lobby for other clients to directly
	// connect to
	virtual SteamIPAddress_t GetPublicIP() = 0;

// Server browser related query packet processing for shared socket mode.  These are used
// when you pass STEAMGAMESERVER_QUERY_PORT_SHARED as the query port to SteamGameServer_Init.
// IP address and port are in host order, i.e 127.0.0.1 == 0x7f000001

	// These are used when you've elected to multiplex the game server's UDP socket
	// rather than having the master server updater use its own sockets.
	// 
	// Source games use this to simplify the job of the server admins, so they 
	// don't have to open up more ports on their firewalls.
	
	// Call this when a packet that starts with 0xFFFFFFFF comes in. That means
	// it's for us.
	virtual bool HandleIncomingPacket( const void *pData, int cbData, uint32 srcIP, uint16 srcPort ) = 0;

	// AFTER calling HandleIncomingPacket for any packets that came in that frame, call this.
	// This gets a packet that the master server updater needs to send out on UDP.
	// It returns the length of the packet it wants to send, or 0 if there are no more packets to send.
	// Call this each frame until it returns 0.
	virtual int GetNextOutgoingPacket( void *pOut, int cbMaxOut, uint32 *pNetAdr, uint16 *pPort ) = 0;

//
// Server clan association
//

	// associate this game server with this clan for the purposes of computing player compat
	STEAM_CALL_RESULT( AssociateWithClanResult_t )
	virtual SteamAPICall_t AssociateWithClan( CSteamID steamIDClan ) = 0;
	
	// ask if any of the current players dont want to play with this new player - or vice versa
	STEAM_CALL_RESULT( ComputeNewPlayerCompatibilityResult_t )
	virtual SteamAPICall_t ComputeNewPlayerCompatibility( CSteamID steamIDNewPlayer ) = 0;




	// Handles receiving a new connection from a Steam user.  This call will ask the Steam
	// servers to validate the users identity, app ownership, and VAC status.  If the Steam servers 
	// are off-line, then it will validate the cached ticket itself which will validate app ownership 
	// and identity.  The AuthBlob here should be acquired on the game client using SteamUser()->InitiateGameConnection()
	// and must then be sent up to the game server for authentication.
	//
	// Return Value: returns true if the users ticket passes basic checks. pSteamIDUser will contain the Steam ID of this user. pSteamIDUser must NOT be NULL
	// If the call succeeds then you should expect a GSClientApprove_t or GSClientDeny_t callback which will tell you whether authentication
	// for the user has succeeded or failed (the steamid in the callback will match the one returned by this call)
	//
	// DEPRECATED!  This function will be removed from the SDK in an upcoming version.
	//              Please migrate to BeginAuthSession and related functions.
	virtual bool SendUserConnectAndAuthenticate_DEPRECATED( uint32 unIPClient, const void *pvAuthBlob, uint32 cubAuthBlobSize, CSteamID *pSteamIDUser ) = 0;

	// Creates a fake user (ie, a bot) which will be listed as playing on the server, but skips validation.  
	// 
	// Return Value: Returns a SteamID for the user to be tracked with, you should call EndAuthSession()
	// when this user leaves the server just like you would for a real user.
	virtual CSteamID CreateUnauthenticatedUserConnection() = 0;

	// Should be called whenever a user leaves our game server, this lets Steam internally
	// track which users are currently on which servers for the purposes of preventing a single
	// account being logged into multiple servers, showing who is currently on a server, etc.
	//
	// DEPRECATED!  This function will be removed from the SDK in an upcoming version.
	//              Please migrate to BeginAuthSession and related functions.
	virtual void SendUserDisconnect_DEPRECATED( CSteamID steamIDUser ) = 0;

	// Update the data to be displayed in the server browser and matchmaking interfaces for a user
	// currently connected to the server.  For regular users you must call this after you receive a
	// GSUserValidationSuccess callback.
	// 
	// Return Value: true if successful, false if failure (ie, steamIDUser wasn't for an active player)
	virtual bool BUpdateUserData( CSteamID steamIDUser, const char *pchPlayerName, uint32 uScore ) = 0;

// Deprecated functions.  These will be removed in a future version of the SDK.
// If you really need these, please contact us and help us understand what you are
// using them for.

	STEAM_PRIVATE_API(
		virtual void SetMasterServerHeartbeatInterval_DEPRECATED( int iHeartbeatInterval ) = 0;
		virtual void ForceMasterServerHeartbeat_DEPRECATED() = 0;
	)
};

#define STEAMGAMESERVER_INTERFACE_VERSION "SteamGameServer014"

// Global accessor
inline ISteamGameServer *SteamGameServer();
STEAM_DEFINE_GAMESERVER_INTERFACE_ACCESSOR( ISteamGameServer *, SteamGameServer, STEAMGAMESERVER_INTERFACE_VERSION );

// callbacks
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 


// client has been approved to connect to this game server
struct GSClientApprove_t
{
	enum { k_iCallback = k_iSteamGameServerCallbacks + 1 };
	CSteamID m_SteamID;			// SteamID of approved player
	CSteamID m_OwnerSteamID;	// SteamID of original owner for game license
};


// client has been denied to connection to this game server
struct GSClientDeny_t
{
	enum { k_iCallback = k_iSteamGameServerCallbacks + 2 };
	CSteamID m_SteamID;
	EDenyReason m_eDenyReason;
	char m_rgchOptionalText[128];
};


// request the game server should kick the user
struct GSClientKick_t
{
	enum { k_iCallback = k_iSteamGameServerCallbacks + 3 };
	CSteamID m_SteamID;
	EDenyReason m_eDenyReason;
};

// NOTE: callback values 4 and 5 are skipped because they are used for old deprecated callbacks, 
// do not reuse them here.


// client achievement info
struct GSClientAchievementStatus_t
{
	enum { k_iCallback = k_iSteamGameServerCallbacks + 6 };
	uint64 m_SteamID;
	char m_pchAchievement[128];
	bool m_bUnlocked;
};

// received when the game server requests to be displayed as secure (VAC protected)
// m_bSecure is true if the game server should display itself as secure to users, false otherwise
struct GSPolicyResponse_t
{
	enum { k_iCallback = k_iSteamUserCallbacks + 15 };
	uint8 m_bSecure;
};

// GS gameplay stats info
struct GSGameplayStats_t
{
	enum { k_iCallback = k_iSteamGameServerCallbacks + 7 };
	EResult m_eResult;					// Result of the call
	int32	m_nRank;					// Overall rank of the server (0-based)
	uint32	m_unTotalConnects;			// Total number of clients who have ever connected to the server
	uint32	m_unTotalMinutesPlayed;		// Total number of minutes ever played on the server
};

// send as a reply to RequestUserGroupStatus()
struct GSClientGroupStatus_t
{
	enum { k_iCallback = k_iSteamGameServerCallbacks + 8 };
	CSteamID m_SteamIDUser;
	CSteamID m_SteamIDGroup;
	bool m_bMember;
	bool m_bOfficer;
};

// Sent as a reply to GetServerReputation()
struct GSReputation_t
{
	enum { k_iCallback = k_iSteamGameServerCallbacks + 9 };
	EResult	m_eResult;				// Result of the call;
	uint32	m_unReputationScore;	// The reputation score for the game server
	bool	m_bBanned;				// True if the server is banned from the Steam
									// master servers

	// The following members are only filled out if m_bBanned is true. They will all 
	// be set to zero otherwise. Master server bans are by IP so it is possible to be
	// banned even when the score is good high if there is a bad server on another port.
	// This information can be used to determine which server is bad.

	uint32	m_unBannedIP;		// The IP of the banned server
	uint16	m_usBannedPort;		// The port of the banned server
	uint64	m_ulBannedGameID;	// The game ID the banned server is serving
	uint32	m_unBanExpires;		// Time the ban expires, expressed in the Unix epoch (seconds since 1/1/1970)
};

// Sent as a reply to AssociateWithClan()
struct AssociateWithClanResult_t
{
	enum { k_iCallback = k_iSteamGameServerCallbacks + 10 };
	EResult	m_eResult;				// Result of the call;
};

// Sent as a reply to ComputeNewPlayerCompatibility()
struct ComputeNewPlayerCompatibilityResult_t
{
	enum { k_iCallback = k_iSteamGameServerCallbacks + 11 };
	EResult	m_eResult;				// Result of the call;
	int m_cPlayersThatDontLikeCandidate;
	int m_cPlayersThatCandidateDoesntLike;
	int m_cClanPlayersThatDontLikeCandidate;
	CSteamID m_SteamIDCandidate;
};


#pragma pack( pop )

#endif // ISTEAMGAMESERVER_H
