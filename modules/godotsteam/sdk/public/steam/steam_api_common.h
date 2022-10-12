//====== Copyright Valve Corporation, All rights reserved. ====================
//
// Steamworks SDK minimal include
//
// Defines the minimal set of things we need to use any single interface
// or register for any callback.
//
//=============================================================================

#ifndef STEAM_API_COMMON_H
#define STEAM_API_COMMON_H
#ifdef _WIN32
#pragma once
#endif

#include "steamtypes.h"
#include "steamclientpublic.h"

// S_API defines the linkage and calling conventions for steam_api.dll exports
#if defined( _WIN32 ) && !defined( _X360 )
	#if defined( STEAM_API_EXPORTS )
	#define S_API extern "C" __declspec( dllexport ) 
	#elif defined( STEAM_API_NODLL )
	#define S_API extern "C"
	#else
	#define S_API extern "C" __declspec( dllimport ) 
	#endif // STEAM_API_EXPORTS
#elif defined( GNUC )
	#if defined( STEAM_API_EXPORTS )
	#define S_API extern "C" __attribute__ ((visibility("default"))) 
	#else
	#define S_API extern "C" 
	#endif // STEAM_API_EXPORTS
#else // !WIN32
	#if defined( STEAM_API_EXPORTS )
	#define S_API extern "C"  
	#else
	#define S_API extern "C" 
	#endif // STEAM_API_EXPORTS
#endif

#if ( defined(STEAM_API_EXPORTS) || defined(STEAM_API_NODLL) ) && !defined(API_GEN)
#define STEAM_PRIVATE_API( ... ) __VA_ARGS__
#elif defined(STEAM_API_EXPORTS) && defined(API_GEN)
#define STEAM_PRIVATE_API( ... )
#else
#define STEAM_PRIVATE_API( ... ) protected: __VA_ARGS__ public:
#endif

// handle to a communication pipe to the Steam client
typedef int32 HSteamPipe;
// handle to single instance of a steam user
typedef int32 HSteamUser;
// function prototype
#if defined( POSIX )
#define __cdecl
#endif
extern "C" typedef void (__cdecl *SteamAPIWarningMessageHook_t)(int, const char *);
extern "C" typedef uint32 ( *SteamAPI_CheckCallbackRegistered_t )( int iCallbackNum );
#if defined( __SNC__ )
	#pragma diag_suppress=1700	   // warning 1700: class "%s" has virtual functions but non-virtual destructor
#endif

//----------------------------------------------------------------------------------------------------------------------------------------------------------//
//	steam callback and call-result helpers
//
//	The following macros and classes are used to register your application for
//	callbacks and call-results, which are delivered in a predictable manner.
//
//	STEAM_CALLBACK macros are meant for use inside of a C++ class definition.
//	They map a Steam notification callback directly to a class member function
//	which is automatically prototyped as "void func( callback_type *pParam )".
//
//	CCallResult is used with specific Steam APIs that return "result handles".
//	The handle can be passed to a CCallResult object's Set function, along with
//	an object pointer and member-function pointer. The member function will
//	be executed once the results of the Steam API call are available.
//
//	CCallback and CCallbackManual classes can be used instead of STEAM_CALLBACK
//	macros if you require finer control over registration and unregistration.
//
//	Callbacks and call-results are queued automatically and are only
//	delivered/executed when your application calls SteamAPI_RunCallbacks().
//
//  Note that there is an alternative, lower level callback dispatch mechanism.
//  See SteamAPI_ManualDispatch_Init
//----------------------------------------------------------------------------------------------------------------------------------------------------------//

// Dispatch all queued Steamworks callbacks.
//
// This is safe to call from multiple threads simultaneously,
// but if you choose to do this, callback code could be executed on any thread.
// One alternative is to call SteamAPI_RunCallbacks from the main thread only,
// and call SteamAPI_ReleaseCurrentThreadMemory regularly on other threads.
S_API void S_CALLTYPE SteamAPI_RunCallbacks();

// Declares a callback member function plus a helper member variable which
// registers the callback on object creation and unregisters on destruction.
// The optional fourth 'var' param exists only for backwards-compatibility
// and can be ignored.
#define STEAM_CALLBACK( thisclass, func, .../*callback_type, [deprecated] var*/ ) \
	_STEAM_CALLBACK_SELECT( ( __VA_ARGS__, 4, 3 ), ( /**/, thisclass, func, __VA_ARGS__ ) )

// Declares a callback function and a named CCallbackManual variable which
// has Register and Unregister functions instead of automatic registration.
#define STEAM_CALLBACK_MANUAL( thisclass, func, callback_type, var )	\
	CCallbackManual< thisclass, callback_type > var; void func( callback_type *pParam )

// Dispatch callbacks relevant to the gameserver client and interfaces.
// To register for these, you need to use STEAM_GAMESERVER_CALLBACK.
// (Or call SetGameserverFlag on your CCallbackBase object.)
S_API void S_CALLTYPE SteamGameServer_RunCallbacks();

// Same as STEAM_CALLBACK, but for callbacks on the gameserver interface.
// These will be dispatched during SteamGameServer_RunCallbacks
#define STEAM_GAMESERVER_CALLBACK( thisclass, func, /*callback_type, [deprecated] var*/... ) \
	_STEAM_CALLBACK_SELECT( ( __VA_ARGS__, GS, 3 ), ( this->SetGameserverFlag();, thisclass, func, __VA_ARGS__ ) )
#define STEAM_GAMESERVER_CALLBACK_MANUAL( thisclass, func, callback_type, var ) \
	CCallbackManual< thisclass, callback_type, true > var; void func( callback_type *pParam )

//-----------------------------------------------------------------------------
// Purpose: base for callbacks and call results - internal implementation detail
//-----------------------------------------------------------------------------
class CCallbackBase
{
public:
	CCallbackBase() { m_nCallbackFlags = 0; m_iCallback = 0; }
	// don't add a virtual destructor because we export this binary interface across dll's
	virtual void Run( void *pvParam ) = 0;
	virtual void Run( void *pvParam, bool bIOFailure, SteamAPICall_t hSteamAPICall ) = 0;
	int GetICallback() { return m_iCallback; }
	virtual int GetCallbackSizeBytes() = 0;

protected:
	enum { k_ECallbackFlagsRegistered = 0x01, k_ECallbackFlagsGameServer = 0x02 };
	uint8 m_nCallbackFlags;
	int m_iCallback;
	friend class CCallbackMgr;

private:
	CCallbackBase( const CCallbackBase& );
	CCallbackBase& operator=( const CCallbackBase& );
};

//-----------------------------------------------------------------------------
// Purpose: templated base for callbacks - internal implementation detail
//-----------------------------------------------------------------------------
template< int sizeof_P >
class CCallbackImpl : protected CCallbackBase
{
public:
	virtual ~CCallbackImpl() { if ( m_nCallbackFlags & k_ECallbackFlagsRegistered ) SteamAPI_UnregisterCallback( this ); }
	void SetGameserverFlag() { m_nCallbackFlags |= k_ECallbackFlagsGameServer; }

protected:
	friend class CCallbackMgr;
	virtual void Run( void *pvParam ) = 0;
	virtual void Run( void *pvParam, bool /*bIOFailure*/, SteamAPICall_t /*hSteamAPICall*/ ) { Run( pvParam ); }
	virtual int GetCallbackSizeBytes() { return sizeof_P; }
};


//-----------------------------------------------------------------------------
// Purpose: maps a steam async call result to a class member function
//			template params: T = local class, P = parameter struct
//-----------------------------------------------------------------------------
template< class T, class P >
class CCallResult : private CCallbackBase
{
public:
	typedef void (T::*func_t)( P*, bool );

	CCallResult();
	~CCallResult();
	
	void Set( SteamAPICall_t hAPICall, T *p, func_t func );
	bool IsActive() const;
	void Cancel();

	void SetGameserverFlag() { m_nCallbackFlags |= k_ECallbackFlagsGameServer; }
private:
	virtual void Run( void *pvParam );
	virtual void Run( void *pvParam, bool bIOFailure, SteamAPICall_t hSteamAPICall );
	virtual int GetCallbackSizeBytes() { return sizeof( P ); }

	SteamAPICall_t m_hAPICall;
	T *m_pObj;
	func_t m_Func;
};



//-----------------------------------------------------------------------------
// Purpose: maps a steam callback to a class member function
//			template params: T = local class, P = parameter struct,
//			bGameserver = listen for gameserver callbacks instead of client callbacks
//-----------------------------------------------------------------------------
template< class T, class P, bool bGameserver = false >
class CCallback : public CCallbackImpl< sizeof( P ) >
{
public:
	typedef void (T::*func_t)(P*);

	// NOTE: If you can't provide the correct parameters at construction time, you should
	// use the CCallbackManual callback object (STEAM_CALLBACK_MANUAL macro) instead.
	CCallback( T *pObj, func_t func );

	void Register( T *pObj, func_t func );
	void Unregister();

protected:
	virtual void Run( void *pvParam );
	
	T *m_pObj;
	func_t m_Func;
};


//-----------------------------------------------------------------------------
// Purpose: subclass of CCallback which allows default-construction in
//			an unregistered state; you must call Register manually
//-----------------------------------------------------------------------------
template< class T, class P, bool bGameServer = false >
class CCallbackManual : public CCallback< T, P, bGameServer >
{
public:
	CCallbackManual() : CCallback< T, P, bGameServer >( nullptr, nullptr ) {}

	// Inherits public Register and Unregister functions from base class
};

// Internal implementation details for all of the above
#include "steam_api_internal.h"

#endif // STEAM_API_COMMON_H
