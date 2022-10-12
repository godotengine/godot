//====== Copyright � 1996-2008, Valve Corporation, All rights reserved. =======
//
// Purpose: interface to utility functions in Steam
//
//=============================================================================

#ifndef ISTEAMUTILS_H
#define ISTEAMUTILS_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"


// Steam API call failure results
enum ESteamAPICallFailure
{
	k_ESteamAPICallFailureNone = -1,			// no failure
	k_ESteamAPICallFailureSteamGone = 0,		// the local Steam process has gone away
	k_ESteamAPICallFailureNetworkFailure = 1,	// the network connection to Steam has been broken, or was already broken
	// SteamServersDisconnected_t callback will be sent around the same time
	// SteamServersConnected_t will be sent when the client is able to talk to the Steam servers again
	k_ESteamAPICallFailureInvalidHandle = 2,	// the SteamAPICall_t handle passed in no longer exists
	k_ESteamAPICallFailureMismatchedCallback = 3,// GetAPICallResult() was called with the wrong callback type for this API call
};


// Input modes for the Big Picture gamepad text entry
enum EGamepadTextInputMode
{
	k_EGamepadTextInputModeNormal = 0,
	k_EGamepadTextInputModePassword = 1
};


// Controls number of allowed lines for the Big Picture gamepad text entry
enum EGamepadTextInputLineMode
{
	k_EGamepadTextInputLineModeSingleLine = 0,
	k_EGamepadTextInputLineModeMultipleLines = 1
};

enum EFloatingGamepadTextInputMode
{
	k_EFloatingGamepadTextInputModeModeSingleLine = 0,		// Enter dismisses the keyboard
	k_EFloatingGamepadTextInputModeModeMultipleLines = 1,	// User needs to explictly close the keyboard
	k_EFloatingGamepadTextInputModeModeEmail = 2,			// Keyboard layout is email, enter dismisses the keyboard
	k_EFloatingGamepadTextInputModeModeNumeric = 3,			// Keyboard layout is numeric, enter dismisses the keyboard

};

// The context where text filtering is being done
enum ETextFilteringContext
{
	k_ETextFilteringContextUnknown = 0,	// Unknown context
	k_ETextFilteringContextGameContent = 1,	// Game content, only legally required filtering is performed
	k_ETextFilteringContextChat = 2,	// Chat from another player
	k_ETextFilteringContextName = 3,	// Character or item name
};


// function prototype for warning message hook
#if defined( POSIX )
#define __cdecl
#endif
extern "C" typedef void (__cdecl *SteamAPIWarningMessageHook_t)(int, const char *);

//-----------------------------------------------------------------------------
// Purpose: interface to user independent utility functions
//-----------------------------------------------------------------------------
class ISteamUtils
{
public:
	// return the number of seconds since the user 
	virtual uint32 GetSecondsSinceAppActive() = 0;
	virtual uint32 GetSecondsSinceComputerActive() = 0;

	// the universe this client is connecting to
	virtual EUniverse GetConnectedUniverse() = 0;

	// Steam server time.  Number of seconds since January 1, 1970, GMT (i.e unix time)
	virtual uint32 GetServerRealTime() = 0;

	// returns the 2 digit ISO 3166-1-alpha-2 format country code this client is running in (as looked up via an IP-to-location database)
	// e.g "US" or "UK".
	virtual const char *GetIPCountry() = 0;

	// returns true if the image exists, and valid sizes were filled out
	virtual bool GetImageSize( int iImage, uint32 *pnWidth, uint32 *pnHeight ) = 0;

	// returns true if the image exists, and the buffer was successfully filled out
	// results are returned in RGBA format
	// the destination buffer size should be 4 * height * width * sizeof(char)
	virtual bool GetImageRGBA( int iImage, uint8 *pubDest, int nDestBufferSize ) = 0;

	// Deprecated.  Do not call this.
	STEAM_PRIVATE_API( virtual bool GetCSERIPPort( uint32 *unIP, uint16 *usPort ) = 0; )

	// return the amount of battery power left in the current system in % [0..100], 255 for being on AC power
	virtual uint8 GetCurrentBatteryPower() = 0;

	// returns the appID of the current process
	virtual uint32 GetAppID() = 0;

	// Sets the position where the overlay instance for the currently calling game should show notifications.
	// This position is per-game and if this function is called from outside of a game context it will do nothing.
	virtual void SetOverlayNotificationPosition( ENotificationPosition eNotificationPosition ) = 0;

	// API asynchronous call results
	// can be used directly, but more commonly used via the callback dispatch API (see steam_api.h)
	virtual bool IsAPICallCompleted( SteamAPICall_t hSteamAPICall, bool *pbFailed ) = 0;
	virtual ESteamAPICallFailure GetAPICallFailureReason( SteamAPICall_t hSteamAPICall ) = 0;
	virtual bool GetAPICallResult( SteamAPICall_t hSteamAPICall, void *pCallback, int cubCallback, int iCallbackExpected, bool *pbFailed ) = 0;

	// Deprecated. Applications should use SteamAPI_RunCallbacks() instead. Game servers do not need to call this function.
	STEAM_PRIVATE_API( virtual void RunFrame() = 0; )

	// returns the number of IPC calls made since the last time this function was called
	// Used for perf debugging so you can understand how many IPC calls your game makes per frame
	// Every IPC call is at minimum a thread context switch if not a process one so you want to rate
	// control how often you do them.
	virtual uint32 GetIPCCallCount() = 0;

	// API warning handling
	// 'int' is the severity; 0 for msg, 1 for warning
	// 'const char *' is the text of the message
	// callbacks will occur directly after the API function is called that generated the warning or message
	virtual void SetWarningMessageHook( SteamAPIWarningMessageHook_t pFunction ) = 0;

	// Returns true if the overlay is running & the user can access it. The overlay process could take a few seconds to
	// start & hook the game process, so this function will initially return false while the overlay is loading.
	virtual bool IsOverlayEnabled() = 0;

	// Normally this call is unneeded if your game has a constantly running frame loop that calls the 
	// D3D Present API, or OGL SwapBuffers API every frame.
	//
	// However, if you have a game that only refreshes the screen on an event driven basis then that can break 
	// the overlay, as it uses your Present/SwapBuffers calls to drive it's internal frame loop and it may also
	// need to Present() to the screen any time an even needing a notification happens or when the overlay is
	// brought up over the game by a user.  You can use this API to ask the overlay if it currently need a present
	// in that case, and then you can check for this periodically (roughly 33hz is desirable) and make sure you
	// refresh the screen with Present or SwapBuffers to allow the overlay to do it's work.
	virtual bool BOverlayNeedsPresent() = 0;

	// Asynchronous call to check if an executable file has been signed using the public key set on the signing tab
	// of the partner site, for example to refuse to load modified executable files.  
	// The result is returned in CheckFileSignature_t.
	//   k_ECheckFileSignatureNoSignaturesFoundForThisApp - This app has not been configured on the signing tab of the partner site to enable this function.
	//   k_ECheckFileSignatureNoSignaturesFoundForThisFile - This file is not listed on the signing tab for the partner site.
	//   k_ECheckFileSignatureFileNotFound - The file does not exist on disk.
	//   k_ECheckFileSignatureInvalidSignature - The file exists, and the signing tab has been set for this file, but the file is either not signed or the signature does not match.
	//   k_ECheckFileSignatureValidSignature - The file is signed and the signature is valid.
	STEAM_CALL_RESULT( CheckFileSignature_t )
	virtual SteamAPICall_t CheckFileSignature( const char *szFileName ) = 0;

	// Activates the full-screen text input dialog which takes a initial text string and returns the text the user has typed
	virtual bool ShowGamepadTextInput( EGamepadTextInputMode eInputMode, EGamepadTextInputLineMode eLineInputMode, const char *pchDescription, uint32 unCharMax, const char *pchExistingText ) = 0;

	// Returns previously entered text & length
	virtual uint32 GetEnteredGamepadTextLength() = 0;
	virtual bool GetEnteredGamepadTextInput( char *pchText, uint32 cchText ) = 0;

	// returns the language the steam client is running in, you probably want ISteamApps::GetCurrentGameLanguage instead, this is for very special usage cases
	virtual const char *GetSteamUILanguage() = 0;

	// returns true if Steam itself is running in VR mode
	virtual bool IsSteamRunningInVR() = 0;
	
	// Sets the inset of the overlay notification from the corner specified by SetOverlayNotificationPosition.
	virtual void SetOverlayNotificationInset( int nHorizontalInset, int nVerticalInset ) = 0;

	// returns true if Steam & the Steam Overlay are running in Big Picture mode
	// Games much be launched through the Steam client to enable the Big Picture overlay. During development,
	// a game can be added as a non-steam game to the developers library to test this feature
	virtual bool IsSteamInBigPictureMode() = 0;

	// ask SteamUI to create and render its OpenVR dashboard
	virtual void StartVRDashboard() = 0;

	// Returns true if the HMD content will be streamed via Steam Remote Play
	virtual bool IsVRHeadsetStreamingEnabled() = 0;

	// Set whether the HMD content will be streamed via Steam Remote Play
	// If this is set to true, then the scene in the HMD headset will be streamed, and remote input will not be allowed.
	// If this is set to false, then the application window will be streamed instead, and remote input will be allowed.
	// The default is true unless "VRHeadsetStreaming" "0" is in the extended appinfo for a game.
	// (this is useful for games that have asymmetric multiplayer gameplay)
	virtual void SetVRHeadsetStreamingEnabled( bool bEnabled ) = 0;

	// Returns whether this steam client is a Steam China specific client, vs the global client.
	virtual bool IsSteamChinaLauncher() = 0;

	// Initializes text filtering, loading dictionaries for the language the game is running in.
	//   unFilterOptions are reserved for future use and should be set to 0
	// Returns false if filtering is unavailable for the game's language, in which case FilterText() will act as a passthrough.
	//
	// Users can customize the text filter behavior in their Steam Account preferences:
	// https://store.steampowered.com/account/preferences#CommunityContentPreferences
	virtual bool InitFilterText( uint32 unFilterOptions = 0 ) = 0;

	// Filters the provided input message and places the filtered result into pchOutFilteredText, using legally required filtering and additional filtering based on the context and user settings
	//   eContext is the type of content in the input string
	//   sourceSteamID is the Steam ID that is the source of the input string (e.g. the player with the name, or who said the chat text)
	//   pchInputText is the input string that should be filtered, which can be ASCII or UTF-8
	//   pchOutFilteredText is where the output will be placed, even if no filtering is performed
	//   nByteSizeOutFilteredText is the size (in bytes) of pchOutFilteredText, should be at least strlen(pchInputText)+1
	// Returns the number of characters (not bytes) filtered
	virtual int FilterText( ETextFilteringContext eContext, CSteamID sourceSteamID, const char *pchInputMessage, char *pchOutFilteredText, uint32 nByteSizeOutFilteredText ) = 0;

	// Return what we believe your current ipv6 connectivity to "the internet" is on the specified protocol.
	// This does NOT tell you if the Steam client is currently connected to Steam via ipv6.
	virtual ESteamIPv6ConnectivityState GetIPv6ConnectivityState( ESteamIPv6ConnectivityProtocol eProtocol ) = 0;

	// returns true if currently running on the Steam Deck device
	virtual bool IsSteamRunningOnSteamDeck() = 0;

	// Opens a floating keyboard over the game content and sends OS keyboard keys directly to the game.
	// The text field position is specified in pixels relative the origin of the game window and is used to position the floating keyboard in a way that doesn't cover the text field
	virtual bool ShowFloatingGamepadTextInput( EFloatingGamepadTextInputMode eKeyboardMode, int nTextFieldXPosition, int nTextFieldYPosition, int nTextFieldWidth, int nTextFieldHeight ) = 0;

	// In game launchers that don't have controller support you can call this to have Steam Input translate the controller input into mouse/kb to navigate the launcher
	virtual void SetGameLauncherMode( bool bLauncherMode ) = 0;

	// Dismisses the floating keyboard.
	virtual bool DismissFloatingGamepadTextInput() = 0;
};

#define STEAMUTILS_INTERFACE_VERSION "SteamUtils010"

// Global interface accessor
inline ISteamUtils *SteamUtils();
STEAM_DEFINE_INTERFACE_ACCESSOR( ISteamUtils *, SteamUtils, SteamInternal_FindOrCreateUserInterface( 0, STEAMUTILS_INTERFACE_VERSION ), "user", STEAMUTILS_INTERFACE_VERSION );

// Global accessor for the gameserver client
inline ISteamUtils *SteamGameServerUtils();
STEAM_DEFINE_INTERFACE_ACCESSOR( ISteamUtils *, SteamGameServerUtils, SteamInternal_FindOrCreateGameServerInterface( 0, STEAMUTILS_INTERFACE_VERSION ), "gameserver", STEAMUTILS_INTERFACE_VERSION );

// callbacks
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 

//-----------------------------------------------------------------------------
// Purpose: The country of the user changed
//-----------------------------------------------------------------------------
struct IPCountry_t
{
	enum { k_iCallback = k_iSteamUtilsCallbacks + 1 };
};


//-----------------------------------------------------------------------------
// Purpose: Fired when running on a laptop and less than 10 minutes of battery is left, fires then every minute
//-----------------------------------------------------------------------------
struct LowBatteryPower_t
{
	enum { k_iCallback = k_iSteamUtilsCallbacks + 2 };
	uint8 m_nMinutesBatteryLeft;
};


//-----------------------------------------------------------------------------
// Purpose: called when a SteamAsyncCall_t has completed (or failed)
//-----------------------------------------------------------------------------
struct SteamAPICallCompleted_t
{
	enum { k_iCallback = k_iSteamUtilsCallbacks + 3 };
	SteamAPICall_t m_hAsyncCall;
	int m_iCallback;
	uint32 m_cubParam;
};


//-----------------------------------------------------------------------------
// called when Steam wants to shutdown
//-----------------------------------------------------------------------------
struct SteamShutdown_t
{
	enum { k_iCallback = k_iSteamUtilsCallbacks + 4 };
};

//-----------------------------------------------------------------------------
// results for CheckFileSignature
//-----------------------------------------------------------------------------
enum ECheckFileSignature
{
	k_ECheckFileSignatureInvalidSignature = 0,
	k_ECheckFileSignatureValidSignature = 1,
	k_ECheckFileSignatureFileNotFound = 2,
	k_ECheckFileSignatureNoSignaturesFoundForThisApp = 3,
	k_ECheckFileSignatureNoSignaturesFoundForThisFile = 4,
};

//-----------------------------------------------------------------------------
// callback for CheckFileSignature
//-----------------------------------------------------------------------------
struct CheckFileSignature_t
{
	enum { k_iCallback = k_iSteamUtilsCallbacks + 5 };
	ECheckFileSignature m_eCheckFileSignature;
};


// k_iSteamUtilsCallbacks + 13 is taken


//-----------------------------------------------------------------------------
// Full Screen gamepad text input has been closed
//-----------------------------------------------------------------------------
struct GamepadTextInputDismissed_t
{
	enum { k_iCallback = k_iSteamUtilsCallbacks + 14 };
	bool m_bSubmitted;										// true if user entered & accepted text (Call ISteamUtils::GetEnteredGamepadTextInput() for text), false if canceled input
	uint32 m_unSubmittedText;
};

// k_iSteamUtilsCallbacks + 15 through 35 are taken

STEAM_CALLBACK_BEGIN( AppResumingFromSuspend_t, k_iSteamUtilsCallbacks + 36 )
STEAM_CALLBACK_END(0)

// k_iSteamUtilsCallbacks + 37 is taken

//-----------------------------------------------------------------------------
// The floating on-screen keyboard has been closed
//-----------------------------------------------------------------------------
struct FloatingGamepadTextInputDismissed_t
{
	enum { k_iCallback = k_iSteamUtilsCallbacks + 38 };
};

#pragma pack( pop )

#endif // ISTEAMUTILS_H
