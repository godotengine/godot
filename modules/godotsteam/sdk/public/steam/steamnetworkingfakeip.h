//====== Copyright Valve Corporation, All rights reserved. ====================

#ifndef STEAMNETWORKINGFAKEIP_H
#define STEAMNETWORKINGFAKEIP_H
#pragma once

#include "steamnetworkingtypes.h"
#include "steam_api_common.h"

// It is HIGHLY recommended to limit messages sent via Fake UDP port to this
// value.  The purpose of a Fake UDP port is to make porting ordinary ad-hoc UDP
// code easier.  Although the real MTU might be higher than this, this particular
// conservative value is chosen so that fragmentation won't be occurring and
// hiding performance problems from you.
constexpr int k_cbSteamNetworkingSocketsFakeUDPPortRecommendedMTU = 1200;

// Messages larger than this size are not allowed and cannot be sent
// via Fake UDP port.
constexpr int k_cbSteamNetworkingSocketsFakeUDPPortMaxMessageSize = 4096;

//-----------------------------------------------------------------------------
/// ISteamNetworkingFakeUDPPort
///
/// Acts like a UDP port, sending and receiving datagrams addressed using
/// FakeIP addresses.
/// 
/// See: ISteamNetworkingSockets::CreateFakeUDPPort

class ISteamNetworkingFakeUDPPort
{
public:
	/// Destroy the object and cleanup any internal connections.
	/// Note that this function call is not threadsafe with respect
	/// to any other method of this interface.  (However, in general
	/// all other operations are threadsafe with respect to each other.)
	virtual void DestroyFakeUDPPort() = 0;

	/// Send a datagram to the specified FakeIP.
	/// 
	/// See ISteamNetworkingSockets::SendMessageToConnection for the meaning of
	/// nSendFlags and possible return codes.
	/// 
	/// Notes:
	/// - datagrams larger than the underlying MTU are supported, but
	///   reliable messages (k_nSteamNetworkingSend_Reliable) are not supported.
	/// - You will usually want to use k_nSteamNetworkingSend_NoNagle
	/// - k_EResultBusy is returned if this is a "server" port and the global
	///   allocation has not yet completed.
	/// - k_EResultIPNotFound will be returned if the address is a local/ephemeral
	///   address and no existing connection can be found.  This can happen if
	///   the remote host contacted us without having a global address, and we
	///   assigned them a random local address, and then the session with
	///   that host timed out.
	/// - When initiating communications, the first messages may be sent
	///   via backend signaling, or otherwise delayed, while a route is found.
	///   Expect the ping time to fluctuate during this period, and it's possible
	///   that messages will be delivered out of order (which is also possible with
	///   ordinary UDP).
	virtual EResult SendMessageToFakeIP( const SteamNetworkingIPAddr &remoteAddress, const void *pData, uint32 cbData, int nSendFlags ) = 0;

	/// Receive messages on the port.
	/// 
	/// Returns the number of messages returned into your array, up to nMaxMessages.
	/// 
	/// SteamNetworkingMessage_t::m_identity in the returned message(s) will always contain
	/// a FakeIP.  See ISteamNetworkingUtils::GetRealIdentityForFakeIP.
	virtual int ReceiveMessages( SteamNetworkingMessage_t **ppOutMessages, int nMaxMessages ) = 0;

	/// Schedule the internal connection for a given peer to be cleaned up in a few seconds.
	///
	/// Idle connections automatically time out, and so this is not strictly *necessary*,
	/// but if you have reason to believe that you are done talking to a given peer for
	/// a while, you can call this to speed up the timeout.  If any remaining packets are
	/// sent or received from the peer, the cleanup is canceled and the usual timeout
	/// value is restored.  Thus you will usually call this immediately after sending
	/// or receiving application-layer "close connection" packets.
	virtual void ScheduleCleanup( const SteamNetworkingIPAddr &remoteAddress ) = 0;
};

/// Callback struct used to notify when a connection has changed state
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error "Must define VALVE_CALLBACK_PACK_SMALL or VALVE_CALLBACK_PACK_LARGE"
#endif

/// A struct used to describe a "fake IP" we have been assigned to
/// use as an identifier.  This callback is posted when
/// ISteamNetworkingSoockets::BeginAsyncRequestFakeIP completes.
/// See also ISteamNetworkingSockets::GetFakeIP
struct SteamNetworkingFakeIPResult_t
{
	enum { k_iCallback = k_iSteamNetworkingSocketsCallbacks + 3 };

	/// Status/result of the allocation request.  Possible failure values are:
	/// - k_EResultBusy - you called GetFakeIP but the request has not completed.
	/// - k_EResultInvalidParam - you called GetFakeIP with an invalid port index
	/// - k_EResultLimitExceeded - You asked for too many ports, or made an
	///   additional request after one had already succeeded
	/// - k_EResultNoMatch - GetFakeIP was called, but no request has been made
	///
	/// Note that, with the exception of k_EResultBusy (if you are polling),
	/// it is highly recommended to treat all failures as fatal.
	EResult m_eResult;

	/// Local identity of the ISteamNetworkingSockets object that made
	/// this request and is assigned the IP.  This is needed in the callback
	/// in the case where there are multiple ISteamNetworkingSockets objects.
	/// (E.g. one for the user, and another for the local gameserver).
	SteamNetworkingIdentity m_identity;

	/// Fake IPv4 IP address that we have been assigned.  NOTE: this
	/// IP address is not exclusively ours!  Steam tries to avoid sharing
	/// IP addresses, but this may not always be possible.  The IP address
	/// may be currently in use by another host, but with different port(s).
	/// The exact same IP:port address may have been used previously.
	/// Steam tries to avoid reusing ports until they have not been in use for
	/// some time, but this may not always be possible.
	uint32 m_unIP;

	/// Port number(s) assigned to us.  Only the first entries will contain
	/// nonzero values.  Entries corresponding to ports beyond what was
	/// allocated for you will be zero.
	///
	/// (NOTE: At the time of this writing, the maximum number of ports you may
	/// request is 4.)
	enum { k_nMaxReturnPorts = 8 };
	uint16 m_unPorts[k_nMaxReturnPorts];
};

#pragma pack( pop )

#endif // _H
