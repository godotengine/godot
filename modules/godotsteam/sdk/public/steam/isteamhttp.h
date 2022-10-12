//====== Copyright © 1996-2009, Valve Corporation, All rights reserved. =======
//
// Purpose: interface to http client 
//
//=============================================================================

#ifndef ISTEAMHTTP_H
#define ISTEAMHTTP_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"
#include "steamhttpenums.h"

// Handle to a HTTP Request handle
typedef uint32 HTTPRequestHandle;
#define INVALID_HTTPREQUEST_HANDLE		0

typedef uint32 HTTPCookieContainerHandle;
#define INVALID_HTTPCOOKIE_HANDLE		0

//-----------------------------------------------------------------------------
// Purpose: interface to http client
//-----------------------------------------------------------------------------
class ISteamHTTP
{
public:

	// Initializes a new HTTP request, returning a handle to use in further operations on it.  Requires
	// the method (GET or POST) and the absolute URL for the request.  Both http and https are supported,
	// so this string must start with http:// or https:// and should look like http://store.steampowered.com/app/250/ 
	// or such.
	virtual HTTPRequestHandle CreateHTTPRequest( EHTTPMethod eHTTPRequestMethod, const char *pchAbsoluteURL ) = 0;

	// Set a context value for the request, which will be returned in the HTTPRequestCompleted_t callback after
	// sending the request.  This is just so the caller can easily keep track of which callbacks go with which request data.
	virtual bool SetHTTPRequestContextValue( HTTPRequestHandle hRequest, uint64 ulContextValue ) = 0;

	// Set a timeout in seconds for the HTTP request, must be called prior to sending the request.  Default
	// timeout is 60 seconds if you don't call this.  Returns false if the handle is invalid, or the request
	// has already been sent.
	virtual bool SetHTTPRequestNetworkActivityTimeout( HTTPRequestHandle hRequest, uint32 unTimeoutSeconds ) = 0;

	// Set a request header value for the request, must be called prior to sending the request.  Will 
	// return false if the handle is invalid or the request is already sent.
	virtual bool SetHTTPRequestHeaderValue( HTTPRequestHandle hRequest, const char *pchHeaderName, const char *pchHeaderValue ) = 0;

	// Set a GET or POST parameter value on the request, which is set will depend on the EHTTPMethod specified
	// when creating the request.  Must be called prior to sending the request.  Will return false if the 
	// handle is invalid or the request is already sent.
	virtual bool SetHTTPRequestGetOrPostParameter( HTTPRequestHandle hRequest, const char *pchParamName, const char *pchParamValue ) = 0;

	// Sends the HTTP request, will return false on a bad handle, otherwise use SteamCallHandle to wait on
	// asynchronous response via callback.
	//
	// Note: If the user is in offline mode in Steam, then this will add a only-if-cached cache-control 
	// header and only do a local cache lookup rather than sending any actual remote request.
	virtual bool SendHTTPRequest( HTTPRequestHandle hRequest, SteamAPICall_t *pCallHandle ) = 0;

	// Sends the HTTP request, will return false on a bad handle, otherwise use SteamCallHandle to wait on
	// asynchronous response via callback for completion, and listen for HTTPRequestHeadersReceived_t and 
	// HTTPRequestDataReceived_t callbacks while streaming.
	virtual bool SendHTTPRequestAndStreamResponse( HTTPRequestHandle hRequest, SteamAPICall_t *pCallHandle ) = 0;

	// Defers a request you have sent, the actual HTTP client code may have many requests queued, and this will move
	// the specified request to the tail of the queue.  Returns false on invalid handle, or if the request is not yet sent.
	virtual bool DeferHTTPRequest( HTTPRequestHandle hRequest ) = 0;

	// Prioritizes a request you have sent, the actual HTTP client code may have many requests queued, and this will move
	// the specified request to the head of the queue.  Returns false on invalid handle, or if the request is not yet sent.
	virtual bool PrioritizeHTTPRequest( HTTPRequestHandle hRequest ) = 0;

	// Checks if a response header is present in a HTTP response given a handle from HTTPRequestCompleted_t, also 
	// returns the size of the header value if present so the caller and allocate a correctly sized buffer for
	// GetHTTPResponseHeaderValue.
	virtual bool GetHTTPResponseHeaderSize( HTTPRequestHandle hRequest, const char *pchHeaderName, uint32 *unResponseHeaderSize ) = 0;

	// Gets header values from a HTTP response given a handle from HTTPRequestCompleted_t, will return false if the
	// header is not present or if your buffer is too small to contain it's value.  You should first call 
	// BGetHTTPResponseHeaderSize to check for the presence of the header and to find out the size buffer needed.
	virtual bool GetHTTPResponseHeaderValue( HTTPRequestHandle hRequest, const char *pchHeaderName, uint8 *pHeaderValueBuffer, uint32 unBufferSize ) = 0;

	// Gets the size of the body data from a HTTP response given a handle from HTTPRequestCompleted_t, will return false if the 
	// handle is invalid.
	virtual bool GetHTTPResponseBodySize( HTTPRequestHandle hRequest, uint32 *unBodySize ) = 0;

	// Gets the body data from a HTTP response given a handle from HTTPRequestCompleted_t, will return false if the 
	// handle is invalid or is to a streaming response, or if the provided buffer is not the correct size.  Use BGetHTTPResponseBodySize first to find out
	// the correct buffer size to use.
	virtual bool GetHTTPResponseBodyData( HTTPRequestHandle hRequest, uint8 *pBodyDataBuffer, uint32 unBufferSize ) = 0;

	// Gets the body data from a streaming HTTP response given a handle from HTTPRequestDataReceived_t. Will return false if the 
	// handle is invalid or is to a non-streaming response (meaning it wasn't sent with SendHTTPRequestAndStreamResponse), or if the buffer size and offset 
	// do not match the size and offset sent in HTTPRequestDataReceived_t.
	virtual bool GetHTTPStreamingResponseBodyData( HTTPRequestHandle hRequest, uint32 cOffset, uint8 *pBodyDataBuffer, uint32 unBufferSize ) = 0;

	// Releases an HTTP response handle, should always be called to free resources after receiving a HTTPRequestCompleted_t
	// callback and finishing using the response.
	virtual bool ReleaseHTTPRequest( HTTPRequestHandle hRequest ) = 0;

	// Gets progress on downloading the body for the request.  This will be zero unless a response header has already been
	// received which included a content-length field.  For responses that contain no content-length it will report
	// zero for the duration of the request as the size is unknown until the connection closes.
	virtual bool GetHTTPDownloadProgressPct( HTTPRequestHandle hRequest, float *pflPercentOut ) = 0;

	// Sets the body for an HTTP Post request.  Will fail and return false on a GET request, and will fail if POST params
	// have already been set for the request.  Setting this raw body makes it the only contents for the post, the pchContentType
	// parameter will set the content-type header for the request so the server may know how to interpret the body.
	virtual bool SetHTTPRequestRawPostBody( HTTPRequestHandle hRequest, const char *pchContentType, uint8 *pubBody, uint32 unBodyLen ) = 0;

	// Creates a cookie container handle which you must later free with ReleaseCookieContainer().  If bAllowResponsesToModify=true
	// than any response to your requests using this cookie container may add new cookies which may be transmitted with
	// future requests.  If bAllowResponsesToModify=false than only cookies you explicitly set will be sent.  This API is just for
	// during process lifetime, after steam restarts no cookies are persisted and you have no way to access the cookie container across
	// repeat executions of your process.
	virtual HTTPCookieContainerHandle CreateCookieContainer( bool bAllowResponsesToModify ) = 0;

	// Release a cookie container you are finished using, freeing it's memory
	virtual bool ReleaseCookieContainer( HTTPCookieContainerHandle hCookieContainer ) = 0;

	// Adds a cookie to the specified cookie container that will be used with future requests.
	virtual bool SetCookie( HTTPCookieContainerHandle hCookieContainer, const char *pchHost, const char *pchUrl, const char *pchCookie ) = 0;

	// Set the cookie container to use for a HTTP request
	virtual bool SetHTTPRequestCookieContainer( HTTPRequestHandle hRequest, HTTPCookieContainerHandle hCookieContainer ) = 0;

	// Set the extra user agent info for a request, this doesn't clobber the normal user agent, it just adds the extra info on the end
	virtual bool SetHTTPRequestUserAgentInfo( HTTPRequestHandle hRequest, const char *pchUserAgentInfo ) = 0;

	// Disable or re-enable verification of SSL/TLS certificates.
	// By default, certificates are checked for all HTTPS requests.
	virtual bool SetHTTPRequestRequiresVerifiedCertificate( HTTPRequestHandle hRequest, bool bRequireVerifiedCertificate ) = 0;

	// Set an absolute timeout on the HTTP request, this is just a total time timeout different than the network activity timeout
	// which can bump everytime we get more data
	virtual bool SetHTTPRequestAbsoluteTimeoutMS( HTTPRequestHandle hRequest, uint32 unMilliseconds ) = 0;

	// Check if the reason the request failed was because we timed it out (rather than some harder failure)
	virtual bool GetHTTPRequestWasTimedOut( HTTPRequestHandle hRequest, bool *pbWasTimedOut ) = 0;
};

#define STEAMHTTP_INTERFACE_VERSION "STEAMHTTP_INTERFACE_VERSION003"

// Global interface accessor
inline ISteamHTTP *SteamHTTP();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamHTTP *, SteamHTTP, STEAMHTTP_INTERFACE_VERSION );

// Global accessor for the gameserver client
inline ISteamHTTP *SteamGameServerHTTP();
STEAM_DEFINE_GAMESERVER_INTERFACE_ACCESSOR( ISteamHTTP *, SteamGameServerHTTP, STEAMHTTP_INTERFACE_VERSION );

// callbacks
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 

struct HTTPRequestCompleted_t
{
	enum { k_iCallback = k_iSteamHTTPCallbacks + 1 };

	// Handle value for the request that has completed.
	HTTPRequestHandle m_hRequest;

	// Context value that the user defined on the request that this callback is associated with, 0 if
	// no context value was set.
	uint64 m_ulContextValue;

	// This will be true if we actually got any sort of response from the server (even an error).  
	// It will be false if we failed due to an internal error or client side network failure.
	bool m_bRequestSuccessful;

	// Will be the HTTP status code value returned by the server, k_EHTTPStatusCode200OK is the normal
	// OK response, if you get something else you probably need to treat it as a failure.
	EHTTPStatusCode m_eStatusCode;

	uint32 m_unBodySize; // Same as GetHTTPResponseBodySize()
};


struct HTTPRequestHeadersReceived_t
{
	enum { k_iCallback = k_iSteamHTTPCallbacks + 2 };

	// Handle value for the request that has received headers.
	HTTPRequestHandle m_hRequest;

	// Context value that the user defined on the request that this callback is associated with, 0 if
	// no context value was set.
	uint64 m_ulContextValue;
};

struct HTTPRequestDataReceived_t
{
	enum { k_iCallback = k_iSteamHTTPCallbacks + 3 };

	// Handle value for the request that has received data.
	HTTPRequestHandle m_hRequest;

	// Context value that the user defined on the request that this callback is associated with, 0 if
	// no context value was set.
	uint64 m_ulContextValue;


	// Offset to provide to GetHTTPStreamingResponseBodyData to get this chunk of data
	uint32 m_cOffset;

	// Size to provide to GetHTTPStreamingResponseBodyData to get this chunk of data
	uint32 m_cBytesReceived;
};


#pragma pack( pop )

#endif // ISTEAMHTTP_H