//====== Copyright © 1996-2014 Valve Corporation, All rights reserved. =======
//
// Purpose: interface to Steam Video
//
//=============================================================================

#ifndef ISTEAMVIDEO_H
#define ISTEAMVIDEO_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"

// callbacks
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif




//-----------------------------------------------------------------------------
// Purpose: Steam Video API
//-----------------------------------------------------------------------------
class ISteamVideo
{
public:

	// Get a URL suitable for streaming the given Video app ID's video
	virtual void GetVideoURL( AppId_t unVideoAppID ) = 0;

	// returns true if user is uploading a live broadcast
	virtual bool IsBroadcasting( int *pnNumViewers ) = 0;

	// Get the OPF Details for 360 Video Playback
	STEAM_CALL_BACK( GetOPFSettingsResult_t )
	virtual void GetOPFSettings( AppId_t unVideoAppID ) = 0;
	virtual bool GetOPFStringForApp( AppId_t unVideoAppID, char *pchBuffer, int32 *pnBufferSize ) = 0;
};

#define STEAMVIDEO_INTERFACE_VERSION "STEAMVIDEO_INTERFACE_V002"

// Global interface accessor
inline ISteamVideo *SteamVideo();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamVideo *, SteamVideo, STEAMVIDEO_INTERFACE_VERSION );

STEAM_CALLBACK_BEGIN( GetVideoURLResult_t, k_iSteamVideoCallbacks + 11 )
	STEAM_CALLBACK_MEMBER( 0, EResult, m_eResult )
	STEAM_CALLBACK_MEMBER( 1, AppId_t, m_unVideoAppID )
	STEAM_CALLBACK_MEMBER( 2, char, m_rgchURL[256] )
STEAM_CALLBACK_END(3)


STEAM_CALLBACK_BEGIN( GetOPFSettingsResult_t, k_iSteamVideoCallbacks + 24 )
	STEAM_CALLBACK_MEMBER( 0, EResult, m_eResult )
	STEAM_CALLBACK_MEMBER( 1, AppId_t, m_unVideoAppID )
STEAM_CALLBACK_END(2)


#pragma pack( pop )


#endif // ISTEAMVIDEO_H
