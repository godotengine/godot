//====== Copyright 1996-2013, Valve Corporation, All rights reserved. =======
//
// Purpose: interface to display html pages in a texture
//
//=============================================================================

#ifndef ISTEAMHTMLSURFACE_H
#define ISTEAMHTMLSURFACE_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"

typedef uint32 HHTMLBrowser;
const uint32 INVALID_HTMLBROWSER = 0;

//-----------------------------------------------------------------------------
// Purpose: Functions for displaying HTML pages and interacting with them
//-----------------------------------------------------------------------------
class ISteamHTMLSurface
{
public:
	virtual ~ISteamHTMLSurface() {}

	// Must call init and shutdown when starting/ending use of the interface
	virtual bool Init() = 0;
	virtual bool Shutdown() = 0;

	// Create a browser object for display of a html page, when creation is complete the call handle
	// will return a HTML_BrowserReady_t callback for the HHTMLBrowser of your new browser.
	//   The user agent string is a substring to be added to the general user agent string so you can
	// identify your client on web servers.
	//   The userCSS string lets you apply a CSS style sheet to every displayed page, leave null if
	// you do not require this functionality.
	//
	// YOU MUST HAVE IMPLEMENTED HANDLERS FOR HTML_BrowserReady_t, HTML_StartRequest_t,
	// HTML_JSAlert_t, HTML_JSConfirm_t, and HTML_FileOpenDialog_t! See the CALLBACKS
	// section of this interface (AllowStartRequest, etc) for more details. If you do
	// not implement these callback handlers, the browser may appear to hang instead of
	// navigating to new pages or triggering javascript popups.
	//
	STEAM_CALL_RESULT( HTML_BrowserReady_t )
	virtual SteamAPICall_t CreateBrowser( const char *pchUserAgent, const char *pchUserCSS ) = 0;

	// Call this when you are done with a html surface, this lets us free the resources being used by it
	virtual void RemoveBrowser( HHTMLBrowser unBrowserHandle ) = 0;

	// Navigate to this URL, results in a HTML_StartRequest_t as the request commences 
	virtual void LoadURL( HHTMLBrowser unBrowserHandle, const char *pchURL, const char *pchPostData ) = 0;

	// Tells the surface the size in pixels to display the surface
	virtual void SetSize( HHTMLBrowser unBrowserHandle, uint32 unWidth, uint32 unHeight ) = 0;

	// Stop the load of the current html page
	virtual void StopLoad( HHTMLBrowser unBrowserHandle ) = 0;
	// Reload (most likely from local cache) the current page
	virtual void Reload( HHTMLBrowser unBrowserHandle ) = 0;
	// navigate back in the page history
	virtual void GoBack( HHTMLBrowser unBrowserHandle ) = 0;
	// navigate forward in the page history
	virtual void GoForward( HHTMLBrowser unBrowserHandle ) = 0;

	// add this header to any url requests from this browser
	virtual void AddHeader( HHTMLBrowser unBrowserHandle, const char *pchKey, const char *pchValue ) = 0;
	// run this javascript script in the currently loaded page
	virtual void ExecuteJavascript( HHTMLBrowser unBrowserHandle, const char *pchScript ) = 0;

	enum EHTMLMouseButton
	{
		eHTMLMouseButton_Left = 0,
		eHTMLMouseButton_Right = 1,
		eHTMLMouseButton_Middle = 2,
	};

	// Mouse click and mouse movement commands
	virtual void MouseUp( HHTMLBrowser unBrowserHandle, EHTMLMouseButton eMouseButton ) = 0;
	virtual void MouseDown( HHTMLBrowser unBrowserHandle, EHTMLMouseButton eMouseButton ) = 0;
	virtual void MouseDoubleClick( HHTMLBrowser unBrowserHandle, EHTMLMouseButton eMouseButton ) = 0;
	// x and y are relative to the HTML bounds
	virtual void MouseMove( HHTMLBrowser unBrowserHandle, int x, int y ) = 0;
	// nDelta is pixels of scroll
	virtual void MouseWheel( HHTMLBrowser unBrowserHandle, int32 nDelta ) = 0;

	enum EMouseCursor
	{
		dc_user = 0,
		dc_none,
		dc_arrow,
		dc_ibeam,
		dc_hourglass,
		dc_waitarrow,
		dc_crosshair,
		dc_up,
		dc_sizenw,
		dc_sizese,
		dc_sizene,
		dc_sizesw,
		dc_sizew,
		dc_sizee,
		dc_sizen,
		dc_sizes,
		dc_sizewe,
		dc_sizens,
		dc_sizeall,
		dc_no,
		dc_hand,
		dc_blank, // don't show any custom cursor, just use your default
		dc_middle_pan,
		dc_north_pan,
		dc_north_east_pan,
		dc_east_pan,
		dc_south_east_pan,
		dc_south_pan,
		dc_south_west_pan,
		dc_west_pan,
		dc_north_west_pan,
		dc_alias,
		dc_cell,
		dc_colresize,
		dc_copycur,
		dc_verticaltext,
		dc_rowresize,
		dc_zoomin,
		dc_zoomout,
		dc_help,
		dc_custom,

		dc_last, // custom cursors start from this value and up
	};

	enum EHTMLKeyModifiers
	{
		k_eHTMLKeyModifier_None = 0,
		k_eHTMLKeyModifier_AltDown = 1 << 0,
		k_eHTMLKeyModifier_CtrlDown = 1 << 1,
		k_eHTMLKeyModifier_ShiftDown = 1 << 2,
	};

	// keyboard interactions, native keycode is the virtual key code value from your OS, system key flags the key to not
	// be sent as a typed character as well as a key down
	virtual void KeyDown( HHTMLBrowser unBrowserHandle, uint32 nNativeKeyCode, EHTMLKeyModifiers eHTMLKeyModifiers, bool bIsSystemKey = false ) = 0;
	virtual void KeyUp( HHTMLBrowser unBrowserHandle, uint32 nNativeKeyCode, EHTMLKeyModifiers eHTMLKeyModifiers ) = 0;
	// cUnicodeChar is the unicode character point for this keypress (and potentially multiple chars per press)
	virtual void KeyChar( HHTMLBrowser unBrowserHandle, uint32 cUnicodeChar, EHTMLKeyModifiers eHTMLKeyModifiers ) = 0;

	// programmatically scroll this many pixels on the page
	virtual void SetHorizontalScroll( HHTMLBrowser unBrowserHandle, uint32 nAbsolutePixelScroll ) = 0;
	virtual void SetVerticalScroll( HHTMLBrowser unBrowserHandle, uint32 nAbsolutePixelScroll ) = 0;

	// tell the html control if it has key focus currently, controls showing the I-beam cursor in text controls amongst other things
	virtual void SetKeyFocus( HHTMLBrowser unBrowserHandle, bool bHasKeyFocus ) = 0;

	// open the current pages html code in the local editor of choice, used for debugging
	virtual void ViewSource( HHTMLBrowser unBrowserHandle ) = 0;
	// copy the currently selected text on the html page to the local clipboard
	virtual void CopyToClipboard( HHTMLBrowser unBrowserHandle ) = 0;
	// paste from the local clipboard to the current html page
	virtual void PasteFromClipboard( HHTMLBrowser unBrowserHandle ) = 0;

	// find this string in the browser, if bCurrentlyInFind is true then instead cycle to the next matching element
	virtual void Find( HHTMLBrowser unBrowserHandle, const char *pchSearchStr, bool bCurrentlyInFind, bool bReverse ) = 0;
	// cancel a currently running find
	virtual void StopFind( HHTMLBrowser unBrowserHandle ) = 0;

	// return details about the link at position x,y on the current page
	virtual void GetLinkAtPosition(  HHTMLBrowser unBrowserHandle, int x, int y ) = 0;

	// set a webcookie for the hostname in question
	virtual void SetCookie( const char *pchHostname, const char *pchKey, const char *pchValue, const char *pchPath = "/", RTime32 nExpires = 0, bool bSecure = false, bool bHTTPOnly = false ) = 0;

	// Zoom the current page by flZoom ( from 0.0 to 2.0, so to zoom to 120% use 1.2 ), zooming around point X,Y in the page (use 0,0 if you don't care)
	virtual void SetPageScaleFactor( HHTMLBrowser unBrowserHandle, float flZoom, int nPointX, int nPointY ) = 0;

	// Enable/disable low-resource background mode, where javascript and repaint timers are throttled, resources are
	// more aggressively purged from memory, and audio/video elements are paused. When background mode is enabled,
	// all HTML5 video and audio objects will execute ".pause()" and gain the property "._steam_background_paused = 1".
	// When background mode is disabled, any video or audio objects with that property will resume with ".play()".
	virtual void SetBackgroundMode( HHTMLBrowser unBrowserHandle, bool bBackgroundMode ) = 0;

	// Scale the output display space by this factor, this is useful when displaying content on high dpi devices.
	// Specifies the ratio between physical and logical pixels.
	virtual void SetDPIScalingFactor( HHTMLBrowser unBrowserHandle, float flDPIScaling ) = 0;

	// Open HTML/JS developer tools
	virtual void OpenDeveloperTools( HHTMLBrowser unBrowserHandle ) = 0;

	// CALLBACKS
	//
	//  These set of functions are used as responses to callback requests
	//

	// You MUST call this in response to a HTML_StartRequest_t callback
	//  Set bAllowed to true to allow this navigation, false to cancel it and stay 
	// on the current page. You can use this feature to limit the valid pages
	// allowed in your HTML surface.
	virtual void AllowStartRequest( HHTMLBrowser unBrowserHandle, bool bAllowed ) = 0;

	// You MUST call this in response to a HTML_JSAlert_t or HTML_JSConfirm_t callback
	//  Set bResult to true for the OK option of a confirm, use false otherwise
	virtual void JSDialogResponse( HHTMLBrowser unBrowserHandle, bool bResult ) = 0;

	// You MUST call this in response to a HTML_FileOpenDialog_t callback
	virtual void FileLoadDialogResponse( HHTMLBrowser unBrowserHandle, const char **pchSelectedFiles ) = 0;
};

#define STEAMHTMLSURFACE_INTERFACE_VERSION "STEAMHTMLSURFACE_INTERFACE_VERSION_005"

// Global interface accessor
inline ISteamHTMLSurface *SteamHTMLSurface();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamHTMLSurface *, SteamHTMLSurface, STEAMHTMLSURFACE_INTERFACE_VERSION );

// callbacks
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 


//-----------------------------------------------------------------------------
// Purpose: The browser is ready for use
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_BrowserReady_t, k_iSteamHTMLSurfaceCallbacks + 1 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // this browser is now fully created and ready to navigate to pages
STEAM_CALLBACK_END(1)


//-----------------------------------------------------------------------------
// Purpose: the browser has a pending paint
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN(HTML_NeedsPaint_t, k_iSteamHTMLSurfaceCallbacks + 2)
STEAM_CALLBACK_MEMBER(0, HHTMLBrowser, unBrowserHandle) // the browser that needs the paint
STEAM_CALLBACK_MEMBER(1, const char *, pBGRA ) // a pointer to the B8G8R8A8 data for this surface, valid until SteamAPI_RunCallbacks is next called
STEAM_CALLBACK_MEMBER(2, uint32, unWide) // the total width of the pBGRA texture
STEAM_CALLBACK_MEMBER(3, uint32, unTall) // the total height of the pBGRA texture
STEAM_CALLBACK_MEMBER(4, uint32, unUpdateX) // the offset in X for the damage rect for this update
STEAM_CALLBACK_MEMBER(5, uint32, unUpdateY) // the offset in Y for the damage rect for this update
STEAM_CALLBACK_MEMBER(6, uint32, unUpdateWide) // the width of the damage rect for this update
STEAM_CALLBACK_MEMBER(7, uint32, unUpdateTall) // the height of the damage rect for this update
STEAM_CALLBACK_MEMBER(8, uint32, unScrollX) // the page scroll the browser was at when this texture was rendered
STEAM_CALLBACK_MEMBER(9, uint32, unScrollY) // the page scroll the browser was at when this texture was rendered
STEAM_CALLBACK_MEMBER(10, float, flPageScale) // the page scale factor on this page when rendered
STEAM_CALLBACK_MEMBER(11, uint32, unPageSerial) // incremented on each new page load, you can use this to reject draws while navigating to new pages
STEAM_CALLBACK_END(12)


//-----------------------------------------------------------------------------
// Purpose: The browser wanted to navigate to a new page
//   NOTE - you MUST call AllowStartRequest in response to this callback
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN(HTML_StartRequest_t, k_iSteamHTMLSurfaceCallbacks + 3)
STEAM_CALLBACK_MEMBER(0, HHTMLBrowser, unBrowserHandle) // the handle of the surface navigating
STEAM_CALLBACK_MEMBER(1, const char *, pchURL) // the url they wish to navigate to 
STEAM_CALLBACK_MEMBER(2, const char *, pchTarget) // the html link target type  (i.e _blank, _self, _parent, _top )
STEAM_CALLBACK_MEMBER(3, const char *, pchPostData ) // any posted data for the request
STEAM_CALLBACK_MEMBER(4, bool, bIsRedirect) // true if this was a http/html redirect from the last load request
STEAM_CALLBACK_END(5)


//-----------------------------------------------------------------------------
// Purpose: The browser has been requested to close due to user interaction (usually from a javascript window.close() call)
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN(HTML_CloseBrowser_t, k_iSteamHTMLSurfaceCallbacks + 4)
STEAM_CALLBACK_MEMBER(0, HHTMLBrowser, unBrowserHandle) // the handle of the surface 
STEAM_CALLBACK_END(1)


//-----------------------------------------------------------------------------
// Purpose: the browser is navigating to a new url
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_URLChanged_t, k_iSteamHTMLSurfaceCallbacks + 5 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface navigating
STEAM_CALLBACK_MEMBER( 1, const char *, pchURL ) // the url they wish to navigate to 
STEAM_CALLBACK_MEMBER( 2, const char *, pchPostData ) // any posted data for the request
STEAM_CALLBACK_MEMBER( 3, bool, bIsRedirect ) // true if this was a http/html redirect from the last load request
STEAM_CALLBACK_MEMBER( 4, const char *, pchPageTitle ) // the title of the page
STEAM_CALLBACK_MEMBER( 5, bool, bNewNavigation ) // true if this was from a fresh tab and not a click on an existing page
STEAM_CALLBACK_END(6)


//-----------------------------------------------------------------------------
// Purpose: A page is finished loading
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_FinishedRequest_t, k_iSteamHTMLSurfaceCallbacks + 6 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, const char *, pchURL ) // 
STEAM_CALLBACK_MEMBER( 2, const char *, pchPageTitle ) // 
STEAM_CALLBACK_END(3)


//-----------------------------------------------------------------------------
// Purpose: a request to load this url in a new tab
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_OpenLinkInNewTab_t, k_iSteamHTMLSurfaceCallbacks + 7 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, const char *, pchURL ) // 
STEAM_CALLBACK_END(2)


//-----------------------------------------------------------------------------
// Purpose: the page has a new title now
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_ChangedTitle_t, k_iSteamHTMLSurfaceCallbacks + 8 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, const char *, pchTitle ) // 
STEAM_CALLBACK_END(2)


//-----------------------------------------------------------------------------
// Purpose: results from a search
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_SearchResults_t, k_iSteamHTMLSurfaceCallbacks + 9 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, uint32, unResults ) // 
STEAM_CALLBACK_MEMBER( 2, uint32, unCurrentMatch ) // 
STEAM_CALLBACK_END(3)


//-----------------------------------------------------------------------------
// Purpose: page history status changed on the ability to go backwards and forward
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_CanGoBackAndForward_t, k_iSteamHTMLSurfaceCallbacks + 10 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, bool, bCanGoBack ) // 
STEAM_CALLBACK_MEMBER( 2, bool, bCanGoForward ) // 
STEAM_CALLBACK_END(3)


//-----------------------------------------------------------------------------
// Purpose: details on the visibility and size of the horizontal scrollbar
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_HorizontalScroll_t, k_iSteamHTMLSurfaceCallbacks + 11 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, uint32, unScrollMax ) // 
STEAM_CALLBACK_MEMBER( 2, uint32, unScrollCurrent ) // 
STEAM_CALLBACK_MEMBER( 3, float, flPageScale ) // 
STEAM_CALLBACK_MEMBER( 4, bool , bVisible ) // 
STEAM_CALLBACK_MEMBER( 5, uint32, unPageSize ) // 
STEAM_CALLBACK_END(6)


//-----------------------------------------------------------------------------
// Purpose: details on the visibility and size of the vertical scrollbar
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_VerticalScroll_t, k_iSteamHTMLSurfaceCallbacks + 12 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, uint32, unScrollMax ) // 
STEAM_CALLBACK_MEMBER( 2, uint32, unScrollCurrent ) // 
STEAM_CALLBACK_MEMBER( 3, float, flPageScale ) // 
STEAM_CALLBACK_MEMBER( 4, bool, bVisible ) // 
STEAM_CALLBACK_MEMBER( 5, uint32, unPageSize ) // 
STEAM_CALLBACK_END(6)


//-----------------------------------------------------------------------------
// Purpose: response to GetLinkAtPosition call 
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_LinkAtPosition_t, k_iSteamHTMLSurfaceCallbacks + 13 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, uint32, x ) // NOTE - Not currently set
STEAM_CALLBACK_MEMBER( 2, uint32, y ) // NOTE - Not currently set
STEAM_CALLBACK_MEMBER( 3, const char *, pchURL ) // 
STEAM_CALLBACK_MEMBER( 4, bool, bInput ) // 
STEAM_CALLBACK_MEMBER( 5, bool, bLiveLink ) // 
STEAM_CALLBACK_END(6)



//-----------------------------------------------------------------------------
// Purpose: show a Javascript alert dialog, call JSDialogResponse 
//   when the user dismisses this dialog (or right away to ignore it)
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_JSAlert_t, k_iSteamHTMLSurfaceCallbacks + 14 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, const char *, pchMessage ) // 
STEAM_CALLBACK_END(2)


//-----------------------------------------------------------------------------
// Purpose: show a Javascript confirmation dialog, call JSDialogResponse 
//   when the user dismisses this dialog (or right away to ignore it)
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_JSConfirm_t, k_iSteamHTMLSurfaceCallbacks + 15 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, const char *, pchMessage ) // 
STEAM_CALLBACK_END(2)


//-----------------------------------------------------------------------------
// Purpose: when received show a file open dialog
//   then call FileLoadDialogResponse with the file(s) the user selected.
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_FileOpenDialog_t, k_iSteamHTMLSurfaceCallbacks + 16 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, const char *, pchTitle ) // 
STEAM_CALLBACK_MEMBER( 2, const char *, pchInitialFile ) // 
STEAM_CALLBACK_END(3)


//-----------------------------------------------------------------------------
// Purpose: a new html window is being created.
//
// IMPORTANT NOTE: at this time, the API does not allow you to acknowledge or
// render the contents of this new window, so the new window is always destroyed
// immediately. The URL and other parameters of the new window are passed here
// to give your application the opportunity to call CreateBrowser and set up
// a new browser in response to the attempted popup, if you wish to do so.
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_NewWindow_t, k_iSteamHTMLSurfaceCallbacks + 21 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the current surface 
STEAM_CALLBACK_MEMBER( 1, const char *, pchURL ) // the page to load
STEAM_CALLBACK_MEMBER( 2, uint32, unX ) // the x pos into the page to display the popup
STEAM_CALLBACK_MEMBER( 3, uint32, unY ) // the y pos into the page to display the popup
STEAM_CALLBACK_MEMBER( 4, uint32, unWide ) // the total width of the pBGRA texture
STEAM_CALLBACK_MEMBER( 5, uint32, unTall ) // the total height of the pBGRA texture
STEAM_CALLBACK_MEMBER( 6, HHTMLBrowser, unNewWindow_BrowserHandle_IGNORE )
STEAM_CALLBACK_END(7)


//-----------------------------------------------------------------------------
// Purpose: change the cursor to display
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_SetCursor_t, k_iSteamHTMLSurfaceCallbacks + 22 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, uint32, eMouseCursor ) // the EMouseCursor to display
STEAM_CALLBACK_END(2)


//-----------------------------------------------------------------------------
// Purpose: informational message from the browser
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_StatusText_t, k_iSteamHTMLSurfaceCallbacks + 23 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, const char *, pchMsg ) // the EMouseCursor to display
STEAM_CALLBACK_END(2)


//-----------------------------------------------------------------------------
// Purpose: show a tooltip
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_ShowToolTip_t, k_iSteamHTMLSurfaceCallbacks + 24 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, const char *, pchMsg ) // the EMouseCursor to display
STEAM_CALLBACK_END(2)


//-----------------------------------------------------------------------------
// Purpose: update the text of an existing tooltip
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_UpdateToolTip_t, k_iSteamHTMLSurfaceCallbacks + 25 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_MEMBER( 1, const char *, pchMsg ) // the EMouseCursor to display
STEAM_CALLBACK_END(2)


//-----------------------------------------------------------------------------
// Purpose: hide the tooltip you are showing
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_HideToolTip_t, k_iSteamHTMLSurfaceCallbacks + 26 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // the handle of the surface 
STEAM_CALLBACK_END(1)


//-----------------------------------------------------------------------------
// Purpose: The browser has restarted due to an internal failure, use this new handle value
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( HTML_BrowserRestarted_t, k_iSteamHTMLSurfaceCallbacks + 27 )
STEAM_CALLBACK_MEMBER( 0, HHTMLBrowser, unBrowserHandle ) // this is the new browser handle after the restart
STEAM_CALLBACK_MEMBER( 1, HHTMLBrowser, unOldBrowserHandle ) // the handle for the browser before the restart, if your handle was this then switch to using unBrowserHandle for API calls
STEAM_CALLBACK_END(2)


#pragma pack( pop )


#endif // ISTEAMHTMLSURFACE_H
