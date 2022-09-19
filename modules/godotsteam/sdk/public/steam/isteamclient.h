//====== Copyright Valve Corporation, All rights reserved. ====================
//
// Internal low-level access to Steamworks interfaces.
//
// Most users of the Steamworks SDK do not need to include this file.
// You should only include this if you are doing something special.
//=============================================================================

#ifndef ISTEAMCLIENT_H
#define ISTEAMCLIENT_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"

//-----------------------------------------------------------------------------
// Purpose: Interface to creating a new steam instance, or to
//			connect to an existing steam instance, whether it's in a
//			different process or is local.
//
//			For most scenarios this is all handled automatically via SteamAPI_Init().
//			You'll only need these APIs if you have a more complex versioning scheme,
//			or if you want to implement a multiplexed gameserver where a single process
//			is handling multiple games at once with independent gameserver SteamIDs.
//-----------------------------------------------------------------------------
class ISteamClient
{
public:
	// Creates a communication pipe to the Steam client.
	// NOT THREADSAFE - ensure that no other threads are accessing Steamworks API when calling
	virtual HSteamPipe CreateSteamPipe() = 0;

	// Releases a previously created communications pipe
	// NOT THREADSAFE - ensure that no other threads are accessing Steamworks API when calling
	virtual bool BReleaseSteamPipe( HSteamPipe hSteamPipe ) = 0;

	// connects to an existing global user, failing if none exists
	// used by the game to coordinate with the steamUI
	// NOT THREADSAFE - ensure that no other threads are accessing Steamworks API when calling
	virtual HSteamUser ConnectToGlobalUser( HSteamPipe hSteamPipe ) = 0;

	// used by game servers, create a steam user that won't be shared with anyone else
	// NOT THREADSAFE - ensure that no other threads are accessing Steamworks API when calling
	virtual HSteamUser CreateLocalUser( HSteamPipe *phSteamPipe, EAccountType eAccountType ) = 0;

	// removes an allocated user
	// NOT THREADSAFE - ensure that no other threads are accessing Steamworks API when calling
	virtual void ReleaseUser( HSteamPipe hSteamPipe, HSteamUser hUser ) = 0;

	// retrieves the ISteamUser interface associated with the handle
	virtual ISteamUser *GetISteamUser( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// retrieves the ISteamGameServer interface associated with the handle
	virtual ISteamGameServer *GetISteamGameServer( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// set the local IP and Port to bind to
	// this must be set before CreateLocalUser()
	virtual void SetLocalIPBinding( const SteamIPAddress_t &unIP, uint16 usPort ) = 0; 

	// returns the ISteamFriends interface
	virtual ISteamFriends *GetISteamFriends( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// returns the ISteamUtils interface
	virtual ISteamUtils *GetISteamUtils( HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// returns the ISteamMatchmaking interface
	virtual ISteamMatchmaking *GetISteamMatchmaking( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// returns the ISteamMatchmakingServers interface
	virtual ISteamMatchmakingServers *GetISteamMatchmakingServers( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// returns the a generic interface
	virtual void *GetISteamGenericInterface( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// returns the ISteamUserStats interface
	virtual ISteamUserStats *GetISteamUserStats( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// returns the ISteamGameServerStats interface
	virtual ISteamGameServerStats *GetISteamGameServerStats( HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// returns apps interface
	virtual ISteamApps *GetISteamApps( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// networking
	virtual ISteamNetworking *GetISteamNetworking( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// remote storage
	virtual ISteamRemoteStorage *GetISteamRemoteStorage( HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// user screenshots
	virtual ISteamScreenshots *GetISteamScreenshots( HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// game search
	virtual ISteamGameSearch *GetISteamGameSearch( HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// Deprecated. Applications should use SteamAPI_RunCallbacks() or SteamGameServer_RunCallbacks() instead.
	STEAM_PRIVATE_API( virtual void RunFrame() = 0; )

	// returns the number of IPC calls made since the last time this function was called
	// Used for perf debugging so you can understand how many IPC calls your game makes per frame
	// Every IPC call is at minimum a thread context switch if not a process one so you want to rate
	// control how often you do them.
	virtual uint32 GetIPCCallCount() = 0;

	// API warning handling
	// 'int' is the severity; 0 for msg, 1 for warning
	// 'const char *' is the text of the message
	// callbacks will occur directly after the API function is called that generated the warning or message.
	virtual void SetWarningMessageHook( SteamAPIWarningMessageHook_t pFunction ) = 0;

	// Trigger global shutdown for the DLL
	virtual bool BShutdownIfAllPipesClosed() = 0;

	// Expose HTTP interface
	virtual ISteamHTTP *GetISteamHTTP( HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// Deprecated - the ISteamUnifiedMessages interface is no longer intended for public consumption.
	STEAM_PRIVATE_API( virtual void *DEPRECATED_GetISteamUnifiedMessages( HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0 ; )

	// Exposes the ISteamController interface - deprecated in favor of Steam Input
	virtual ISteamController *GetISteamController( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// Exposes the ISteamUGC interface
	virtual ISteamUGC *GetISteamUGC( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// returns app list interface, only available on specially registered apps
	virtual ISteamAppList *GetISteamAppList( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;
	
	// Music Player
	virtual ISteamMusic *GetISteamMusic( HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// Music Player Remote
	virtual ISteamMusicRemote *GetISteamMusicRemote(HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion) = 0;

	// html page display
	virtual ISteamHTMLSurface *GetISteamHTMLSurface(HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion) = 0;

	// Helper functions for internal Steam usage
	STEAM_PRIVATE_API( virtual void DEPRECATED_Set_SteamAPI_CPostAPIResultInProcess( void (*)() ) = 0; )
	STEAM_PRIVATE_API( virtual void DEPRECATED_Remove_SteamAPI_CPostAPIResultInProcess( void (*)() ) = 0; )
	STEAM_PRIVATE_API( virtual void Set_SteamAPI_CCheckCallbackRegisteredInProcess( SteamAPI_CheckCallbackRegistered_t func ) = 0; )

	// inventory
	virtual ISteamInventory *GetISteamInventory( HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// Video
	virtual ISteamVideo *GetISteamVideo( HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// Parental controls
	virtual ISteamParentalSettings *GetISteamParentalSettings( HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// Exposes the Steam Input interface for controller support
	virtual ISteamInput *GetISteamInput( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// Steam Parties interface
	virtual ISteamParties *GetISteamParties( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	// Steam Remote Play interface
	virtual ISteamRemotePlay *GetISteamRemotePlay( HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char *pchVersion ) = 0;

	STEAM_PRIVATE_API( virtual void DestroyAllInterfaces() = 0; )

};
#define STEAMCLIENT_INTERFACE_VERSION		"SteamClient020"

#ifndef STEAM_API_EXPORTS

// Global ISteamClient interface accessor
inline ISteamClient *SteamClient();
STEAM_DEFINE_INTERFACE_ACCESSOR( ISteamClient *, SteamClient, SteamInternal_CreateInterface( STEAMCLIENT_INTERFACE_VERSION ), "global", STEAMCLIENT_INTERFACE_VERSION );

// The internal ISteamClient used for the gameserver interface.
// (This is actually the same thing.  You really shouldn't need to access any of this stuff directly.)
inline ISteamClient *SteamGameServerClient() { return SteamClient(); }

#endif

#endif // ISTEAMCLIENT_H
