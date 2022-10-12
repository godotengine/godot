//====== Copyright © 1996-2008, Valve Corporation, All rights reserved. =======
//
// Purpose: interface to app data in Steam
//
//=============================================================================

#ifndef ISTEAMAPPS_H
#define ISTEAMAPPS_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"

const int k_cubAppProofOfPurchaseKeyMax = 240;			// max supported length of a legacy cd key 


//-----------------------------------------------------------------------------
// Purpose: interface to app data
//-----------------------------------------------------------------------------
class ISteamApps
{
public:
	virtual bool BIsSubscribed() = 0;
	virtual bool BIsLowViolence() = 0;
	virtual bool BIsCybercafe() = 0;
	virtual bool BIsVACBanned() = 0;
	virtual const char *GetCurrentGameLanguage() = 0;
	virtual const char *GetAvailableGameLanguages() = 0;

	// only use this member if you need to check ownership of another game related to yours, a demo for example
	virtual bool BIsSubscribedApp( AppId_t appID ) = 0;

	// Takes AppID of DLC and checks if the user owns the DLC & if the DLC is installed
	virtual bool BIsDlcInstalled( AppId_t appID ) = 0;

	// returns the Unix time of the purchase of the app
	virtual uint32 GetEarliestPurchaseUnixTime( AppId_t nAppID ) = 0;

	// Checks if the user is subscribed to the current app through a free weekend
	// This function will return false for users who have a retail or other type of license
	// Before using, please ask your Valve technical contact how to package and secure your free weekened
	virtual bool BIsSubscribedFromFreeWeekend() = 0;

	// Returns the number of DLC pieces for the running app
	virtual int GetDLCCount() = 0;

	// Returns metadata for DLC by index, of range [0, GetDLCCount()]
	virtual bool BGetDLCDataByIndex( int iDLC, AppId_t *pAppID, bool *pbAvailable, char *pchName, int cchNameBufferSize ) = 0;

	// Install/Uninstall control for optional DLC
	virtual void InstallDLC( AppId_t nAppID ) = 0;
	virtual void UninstallDLC( AppId_t nAppID ) = 0;
	
	// Request legacy cd-key for yourself or owned DLC. If you are interested in this
	// data then make sure you provide us with a list of valid keys to be distributed
	// to users when they purchase the game, before the game ships.
	// You'll receive an AppProofOfPurchaseKeyResponse_t callback when
	// the key is available (which may be immediately).
	virtual void RequestAppProofOfPurchaseKey( AppId_t nAppID ) = 0;

	virtual bool GetCurrentBetaName( char *pchName, int cchNameBufferSize ) = 0; // returns current beta branch name, 'public' is the default branch
	virtual bool MarkContentCorrupt( bool bMissingFilesOnly ) = 0; // signal Steam that game files seems corrupt or missing
	virtual uint32 GetInstalledDepots( AppId_t appID, DepotId_t *pvecDepots, uint32 cMaxDepots ) = 0; // return installed depots in mount order

	// returns current app install folder for AppID, returns folder name length
	virtual uint32 GetAppInstallDir( AppId_t appID, char *pchFolder, uint32 cchFolderBufferSize ) = 0;
	virtual bool BIsAppInstalled( AppId_t appID ) = 0; // returns true if that app is installed (not necessarily owned)
	
	// returns the SteamID of the original owner. If this CSteamID is different from ISteamUser::GetSteamID(),
	// the user has a temporary license borrowed via Family Sharing
	virtual CSteamID GetAppOwner() = 0; 

	// Returns the associated launch param if the game is run via steam://run/<appid>//?param1=value1&param2=value2&param3=value3 etc.
	// Parameter names starting with the character '@' are reserved for internal use and will always return and empty string.
	// Parameter names starting with an underscore '_' are reserved for steam features -- they can be queried by the game,
	// but it is advised that you not param names beginning with an underscore for your own features.
	// Check for new launch parameters on callback NewUrlLaunchParameters_t
	virtual const char *GetLaunchQueryParam( const char *pchKey ) = 0; 

	// get download progress for optional DLC
	virtual bool GetDlcDownloadProgress( AppId_t nAppID, uint64 *punBytesDownloaded, uint64 *punBytesTotal ) = 0; 

	// return the buildid of this app, may change at any time based on backend updates to the game
	virtual int GetAppBuildId() = 0;

	// Request all proof of purchase keys for the calling appid and asociated DLC.
	// A series of AppProofOfPurchaseKeyResponse_t callbacks will be sent with
	// appropriate appid values, ending with a final callback where the m_nAppId
	// member is k_uAppIdInvalid (zero).
	virtual void RequestAllProofOfPurchaseKeys() = 0;

	STEAM_CALL_RESULT( FileDetailsResult_t )
	virtual SteamAPICall_t GetFileDetails( const char* pszFileName ) = 0;

	// Get command line if game was launched via Steam URL, e.g. steam://run/<appid>//<command line>/.
	// This method of passing a connect string (used when joining via rich presence, accepting an
	// invite, etc) is preferable to passing the connect string on the operating system command
	// line, which is a security risk.  In order for rich presence joins to go through this
	// path and not be placed on the OS command line, you must set a value in your app's
	// configuration on Steam.  Ask Valve for help with this.
	//
	// If game was already running and launched again, the NewUrlLaunchParameters_t will be fired.
	virtual int GetLaunchCommandLine( char *pszCommandLine, int cubCommandLine ) = 0;

	// Check if user borrowed this game via Family Sharing, If true, call GetAppOwner() to get the lender SteamID
	virtual bool BIsSubscribedFromFamilySharing() = 0;

	// check if game is a timed trial with limited playtime
	virtual bool BIsTimedTrial( uint32* punSecondsAllowed, uint32* punSecondsPlayed ) = 0; 
};

#define STEAMAPPS_INTERFACE_VERSION "STEAMAPPS_INTERFACE_VERSION008"

// Global interface accessor
inline ISteamApps *SteamApps();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamApps *, SteamApps, STEAMAPPS_INTERFACE_VERSION );

// callbacks
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 
//-----------------------------------------------------------------------------
// Purpose: posted after the user gains ownership of DLC & that DLC is installed
//-----------------------------------------------------------------------------
struct DlcInstalled_t
{
	enum { k_iCallback = k_iSteamAppsCallbacks + 5 };
	AppId_t m_nAppID;		// AppID of the DLC
};


//-----------------------------------------------------------------------------
// Purpose: possible results when registering an activation code
//-----------------------------------------------------------------------------
enum ERegisterActivationCodeResult
{
	k_ERegisterActivationCodeResultOK = 0,
	k_ERegisterActivationCodeResultFail = 1,
	k_ERegisterActivationCodeResultAlreadyRegistered = 2,
	k_ERegisterActivationCodeResultTimeout = 3,
	k_ERegisterActivationCodeAlreadyOwned = 4,
};


//-----------------------------------------------------------------------------
// Purpose: response to RegisterActivationCode()
//-----------------------------------------------------------------------------
struct RegisterActivationCodeResponse_t
{
	enum { k_iCallback = k_iSteamAppsCallbacks + 8 };
	ERegisterActivationCodeResult m_eResult;
	uint32 m_unPackageRegistered;						// package that was registered. Only set on success
};


//---------------------------------------------------------------------------------
// Purpose: posted after the user gains executes a Steam URL with command line or query parameters
// such as steam://run/<appid>//-commandline/?param1=value1&param2=value2&param3=value3 etc
// while the game is already running.  The new params can be queried
// with GetLaunchQueryParam and GetLaunchCommandLine
//---------------------------------------------------------------------------------
struct NewUrlLaunchParameters_t
{
	enum { k_iCallback = k_iSteamAppsCallbacks + 14 };
};


//-----------------------------------------------------------------------------
// Purpose: response to RequestAppProofOfPurchaseKey/RequestAllProofOfPurchaseKeys
// for supporting third-party CD keys, or other proof-of-purchase systems.
//-----------------------------------------------------------------------------
struct AppProofOfPurchaseKeyResponse_t
{
	enum { k_iCallback = k_iSteamAppsCallbacks + 21 };
	EResult m_eResult;
	uint32	m_nAppID;
	uint32	m_cchKeyLength;
	char	m_rgchKey[k_cubAppProofOfPurchaseKeyMax];
};


//-----------------------------------------------------------------------------
// Purpose: response to GetFileDetails
//-----------------------------------------------------------------------------
struct FileDetailsResult_t
{
	enum { k_iCallback = k_iSteamAppsCallbacks + 23 };
	EResult		m_eResult;
	uint64		m_ulFileSize;	// original file size in bytes
	uint8		m_FileSHA[20];	// original file SHA1 hash
	uint32		m_unFlags;		// 
};


//-----------------------------------------------------------------------------
// Purpose: called for games in Timed Trial mode
//-----------------------------------------------------------------------------
struct TimedTrialStatus_t
{
	enum { k_iCallback = k_iSteamAppsCallbacks + 30 };
	AppId_t		m_unAppID;			// appID
	bool		m_bIsOffline;		// if true, time allowed / played refers to offline time, not total time
	uint32		m_unSecondsAllowed;	// how many seconds the app can be played in total 
	uint32		m_unSecondsPlayed;	// how many seconds the app was already played
};

#pragma pack( pop )
#endif // ISTEAMAPPS_H
