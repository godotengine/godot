//====== Copyright © 1996-2008, Valve Corporation, All rights reserved. =======
//
// Purpose: interface to steam managing network connections between game clients & servers
//
//=============================================================================

#ifndef ISTEAMNETWORKING
#define ISTEAMNETWORKING
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"

// list of possible errors returned by SendP2PPacket() API
// these will be posted in the P2PSessionConnectFail_t callback
enum EP2PSessionError
{
	k_EP2PSessionErrorNone = 0,
	k_EP2PSessionErrorNoRightsToApp = 2,			// local user doesn't own the app that is running
	k_EP2PSessionErrorTimeout = 4,					// target isn't responding, perhaps not calling AcceptP2PSessionWithUser()
													// corporate firewalls can also block this (NAT traversal is not firewall traversal)
													// make sure that UDP ports 3478, 4379, and 4380 are open in an outbound direction

	// The following error codes were removed and will never be sent.
	// For privacy reasons, there is no reply if the user is offline or playing another game.
	k_EP2PSessionErrorNotRunningApp_DELETED = 1,
	k_EP2PSessionErrorDestinationNotLoggedIn_DELETED = 3,

	k_EP2PSessionErrorMax = 5
};

// SendP2PPacket() send types
// Typically k_EP2PSendUnreliable is what you want for UDP-like packets, k_EP2PSendReliable for TCP-like packets
enum EP2PSend
{
	// Basic UDP send. Packets can't be bigger than 1200 bytes (your typical MTU size). Can be lost, or arrive out of order (rare).
	// The sending API does have some knowledge of the underlying connection, so if there is no NAT-traversal accomplished or
	// there is a recognized adjustment happening on the connection, the packet will be batched until the connection is open again.
	k_EP2PSendUnreliable = 0,

	// As above, but if the underlying p2p connection isn't yet established the packet will just be thrown away. Using this on the first
	// packet sent to a remote host almost guarantees the packet will be dropped.
	// This is only really useful for kinds of data that should never buffer up, i.e. voice payload packets
	k_EP2PSendUnreliableNoDelay = 1,

	// Reliable message send. Can send up to 1MB of data in a single message. 
	// Does fragmentation/re-assembly of messages under the hood, as well as a sliding window for efficient sends of large chunks of data. 
	k_EP2PSendReliable = 2,

	// As above, but applies the Nagle algorithm to the send - sends will accumulate 
	// until the current MTU size (typically ~1200 bytes, but can change) or ~200ms has passed (Nagle algorithm). 
	// Useful if you want to send a set of smaller messages but have the coalesced into a single packet
	// Since the reliable stream is all ordered, you can do several small message sends with k_EP2PSendReliableWithBuffering and then
	// do a normal k_EP2PSendReliable to force all the buffered data to be sent.
	k_EP2PSendReliableWithBuffering = 3,

};


// connection state to a specified user, returned by GetP2PSessionState()
// this is under-the-hood info about what's going on with a SendP2PPacket(), shouldn't be needed except for debuggin
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 
struct P2PSessionState_t
{
	uint8 m_bConnectionActive;		// true if we've got an active open connection
	uint8 m_bConnecting;			// true if we're currently trying to establish a connection
	uint8 m_eP2PSessionError;		// last error recorded (see enum above)
	uint8 m_bUsingRelay;			// true if it's going through a relay server (TURN)
	int32 m_nBytesQueuedForSend;
	int32 m_nPacketsQueuedForSend;
	uint32 m_nRemoteIP;				// potential IP:Port of remote host. Could be TURN server. 
	uint16 m_nRemotePort;			// Only exists for compatibility with older authentication api's
};
#pragma pack( pop )


// handle to a socket
typedef uint32 SNetSocket_t;		// CreateP2PConnectionSocket()
typedef uint32 SNetListenSocket_t;	// CreateListenSocket()

// connection progress indicators, used by CreateP2PConnectionSocket()
enum ESNetSocketState
{
	k_ESNetSocketStateInvalid = 0,						

	// communication is valid
	k_ESNetSocketStateConnected = 1,				
	
	// states while establishing a connection
	k_ESNetSocketStateInitiated = 10,				// the connection state machine has started

	// p2p connections
	k_ESNetSocketStateLocalCandidatesFound = 11,	// we've found our local IP info
	k_ESNetSocketStateReceivedRemoteCandidates = 12,// we've received information from the remote machine, via the Steam back-end, about their IP info

	// direct connections
	k_ESNetSocketStateChallengeHandshake = 15,		// we've received a challenge packet from the server

	// failure states
	k_ESNetSocketStateDisconnecting = 21,			// the API shut it down, and we're in the process of telling the other end	
	k_ESNetSocketStateLocalDisconnect = 22,			// the API shut it down, and we've completed shutdown
	k_ESNetSocketStateTimeoutDuringConnect = 23,	// we timed out while trying to creating the connection
	k_ESNetSocketStateRemoteEndDisconnected = 24,	// the remote end has disconnected from us
	k_ESNetSocketStateConnectionBroken = 25,		// connection has been broken; either the other end has disappeared or our local network connection has broke

};

// describes how the socket is currently connected
enum ESNetSocketConnectionType
{
	k_ESNetSocketConnectionTypeNotConnected = 0,
	k_ESNetSocketConnectionTypeUDP = 1,
	k_ESNetSocketConnectionTypeUDPRelay = 2,
};


//-----------------------------------------------------------------------------
// Purpose: Functions for making connections and sending data between clients,
//			traversing NAT's where possible
//
// NOTE: This interface is deprecated and may be removed in a future release of
///      the Steamworks SDK.  Please see ISteamNetworkingSockets and
///      ISteamNetworkingMessages
//-----------------------------------------------------------------------------
class ISteamNetworking
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////
	//
	// UDP-style (connectionless) networking interface.  These functions send messages using
	// an API organized around the destination.  Reliable and unreliable messages are supported.
	//
	// For a more TCP-style interface (meaning you have a connection handle), see the functions below.
	// Both interface styles can send both reliable and unreliable messages.
	//
	// Automatically establishes NAT-traversing or Relay server connections
	//
	// These APIs are deprecated, and may be removed in a future version of the Steamworks
	// SDK.  See ISteamNetworkingMessages.

	// Sends a P2P packet to the specified user
	// UDP-like, unreliable and a max packet size of 1200 bytes
	// the first packet send may be delayed as the NAT-traversal code runs
	// if we can't get through to the user, an error will be posted via the callback P2PSessionConnectFail_t
	// see EP2PSend enum above for the descriptions of the different ways of sending packets
	//
	// nChannel is a routing number you can use to help route message to different systems 	- you'll have to call ReadP2PPacket() 
	// with the same channel number in order to retrieve the data on the other end
	// using different channels to talk to the same user will still use the same underlying p2p connection, saving on resources
	virtual bool SendP2PPacket( CSteamID steamIDRemote, const void *pubData, uint32 cubData, EP2PSend eP2PSendType, int nChannel = 0 ) = 0;

	// returns true if any data is available for read, and the amount of data that will need to be read
	virtual bool IsP2PPacketAvailable( uint32 *pcubMsgSize, int nChannel = 0 ) = 0;

	// reads in a packet that has been sent from another user via SendP2PPacket()
	// returns the size of the message and the steamID of the user who sent it in the last two parameters
	// if the buffer passed in is too small, the message will be truncated
	// this call is not blocking, and will return false if no data is available
	virtual bool ReadP2PPacket( void *pubDest, uint32 cubDest, uint32 *pcubMsgSize, CSteamID *psteamIDRemote, int nChannel = 0 ) = 0;

	// AcceptP2PSessionWithUser() should only be called in response to a P2PSessionRequest_t callback
	// P2PSessionRequest_t will be posted if another user tries to send you a packet that you haven't talked to yet
	// if you don't want to talk to the user, just ignore the request
	// if the user continues to send you packets, another P2PSessionRequest_t will be posted periodically
	// this may be called multiple times for a single user
	// (if you've called SendP2PPacket() on the other user, this implicitly accepts the session request)
	virtual bool AcceptP2PSessionWithUser( CSteamID steamIDRemote ) = 0;

	// call CloseP2PSessionWithUser() when you're done talking to a user, will free up resources under-the-hood
	// if the remote user tries to send data to you again, another P2PSessionRequest_t callback will be posted
	virtual bool CloseP2PSessionWithUser( CSteamID steamIDRemote ) = 0;

	// call CloseP2PChannelWithUser() when you're done talking to a user on a specific channel. Once all channels
	// open channels to a user have been closed, the open session to the user will be closed and new data from this
	// user will trigger a P2PSessionRequest_t callback
	virtual bool CloseP2PChannelWithUser( CSteamID steamIDRemote, int nChannel ) = 0;

	// fills out P2PSessionState_t structure with details about the underlying connection to the user
	// should only needed for debugging purposes
	// returns false if no connection exists to the specified user
	virtual bool GetP2PSessionState( CSteamID steamIDRemote, P2PSessionState_t *pConnectionState ) = 0;

	// Allow P2P connections to fall back to being relayed through the Steam servers if a direct connection
	// or NAT-traversal cannot be established. Only applies to connections created after setting this value,
	// or to existing connections that need to automatically reconnect after this value is set.
	//
	// P2P packet relay is allowed by default
	//
	// NOTE: This function is deprecated and may be removed in a future version of the SDK.  For
	// security purposes, we may decide to relay the traffic to certain peers, even if you pass false
	// to this function, to prevent revealing the client's IP address top another peer.
	virtual bool AllowP2PPacketRelay( bool bAllow ) = 0;


	////////////////////////////////////////////////////////////////////////////////////////////
	//
	// LISTEN / CONNECT connection-oriented interface functions
	//
	// These functions are more like a client-server TCP API.  One side is the "server"
	// and "listens" for incoming connections, which then must be "accepted."  The "client"
	// initiates a connection by "connecting."  Sending and receiving is done through a
	// connection handle.
	//
	// For a more UDP-style interface, where you do not track connection handles but
	// simply send messages to a SteamID, use the UDP-style functions above.
	//
	// Both methods can send both reliable and unreliable methods.
	//
	// These APIs are deprecated, and may be removed in a future version of the Steamworks
	// SDK.  See ISteamNetworkingSockets.
	//
	////////////////////////////////////////////////////////////////////////////////////////////


	// creates a socket and listens others to connect
	// will trigger a SocketStatusCallback_t callback on another client connecting
	// nVirtualP2PPort is the unique ID that the client will connect to, in case you have multiple ports
	//		this can usually just be 0 unless you want multiple sets of connections
	// unIP is the local IP address to bind to
	//		pass in 0 if you just want the default local IP
	// unPort is the port to use
	//		pass in 0 if you don't want users to be able to connect via IP/Port, but expect to be always peer-to-peer connections only
	virtual SNetListenSocket_t CreateListenSocket( int nVirtualP2PPort, SteamIPAddress_t nIP, uint16 nPort, bool bAllowUseOfPacketRelay ) = 0;

	// creates a socket and begin connection to a remote destination
	// can connect via a known steamID (client or game server), or directly to an IP
	// on success will trigger a SocketStatusCallback_t callback
	// on failure or timeout will trigger a SocketStatusCallback_t callback with a failure code in m_eSNetSocketState
	virtual SNetSocket_t CreateP2PConnectionSocket( CSteamID steamIDTarget, int nVirtualPort, int nTimeoutSec, bool bAllowUseOfPacketRelay ) = 0;
	virtual SNetSocket_t CreateConnectionSocket( SteamIPAddress_t nIP, uint16 nPort, int nTimeoutSec ) = 0;

	// disconnects the connection to the socket, if any, and invalidates the handle
	// any unread data on the socket will be thrown away
	// if bNotifyRemoteEnd is set, socket will not be completely destroyed until the remote end acknowledges the disconnect
	virtual bool DestroySocket( SNetSocket_t hSocket, bool bNotifyRemoteEnd ) = 0;
	// destroying a listen socket will automatically kill all the regular sockets generated from it
	virtual bool DestroyListenSocket( SNetListenSocket_t hSocket, bool bNotifyRemoteEnd ) = 0;

	// sending data
	// must be a handle to a connected socket
	// data is all sent via UDP, and thus send sizes are limited to 1200 bytes; after this, many routers will start dropping packets
	// use the reliable flag with caution; although the resend rate is pretty aggressive,
	// it can still cause stalls in receiving data (like TCP)
	virtual bool SendDataOnSocket( SNetSocket_t hSocket, void *pubData, uint32 cubData, bool bReliable ) = 0;

	// receiving data
	// returns false if there is no data remaining
	// fills out *pcubMsgSize with the size of the next message, in bytes
	virtual bool IsDataAvailableOnSocket( SNetSocket_t hSocket, uint32 *pcubMsgSize ) = 0; 

	// fills in pubDest with the contents of the message
	// messages are always complete, of the same size as was sent (i.e. packetized, not streaming)
	// if *pcubMsgSize < cubDest, only partial data is written
	// returns false if no data is available
	virtual bool RetrieveDataFromSocket( SNetSocket_t hSocket, void *pubDest, uint32 cubDest, uint32 *pcubMsgSize ) = 0; 

	// checks for data from any socket that has been connected off this listen socket
	// returns false if there is no data remaining
	// fills out *pcubMsgSize with the size of the next message, in bytes
	// fills out *phSocket with the socket that data is available on
	virtual bool IsDataAvailable( SNetListenSocket_t hListenSocket, uint32 *pcubMsgSize, SNetSocket_t *phSocket ) = 0;

	// retrieves data from any socket that has been connected off this listen socket
	// fills in pubDest with the contents of the message
	// messages are always complete, of the same size as was sent (i.e. packetized, not streaming)
	// if *pcubMsgSize < cubDest, only partial data is written
	// returns false if no data is available
	// fills out *phSocket with the socket that data is available on
	virtual bool RetrieveData( SNetListenSocket_t hListenSocket, void *pubDest, uint32 cubDest, uint32 *pcubMsgSize, SNetSocket_t *phSocket ) = 0;

	// returns information about the specified socket, filling out the contents of the pointers
	virtual bool GetSocketInfo( SNetSocket_t hSocket, CSteamID *pSteamIDRemote, int *peSocketStatus, SteamIPAddress_t *punIPRemote, uint16 *punPortRemote ) = 0;

	// returns which local port the listen socket is bound to
	// *pnIP and *pnPort will be 0 if the socket is set to listen for P2P connections only
	virtual bool GetListenSocketInfo( SNetListenSocket_t hListenSocket, SteamIPAddress_t *pnIP, uint16 *pnPort ) = 0;

	// returns true to describe how the socket ended up connecting
	virtual ESNetSocketConnectionType GetSocketConnectionType( SNetSocket_t hSocket ) = 0;

	// max packet size, in bytes
	virtual int GetMaxPacketSize( SNetSocket_t hSocket ) = 0;
};
#define STEAMNETWORKING_INTERFACE_VERSION "SteamNetworking006"

// Global interface accessor
inline ISteamNetworking *SteamNetworking();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamNetworking *, SteamNetworking, STEAMNETWORKING_INTERFACE_VERSION );

// Global accessor for the gameserver client
inline ISteamNetworking *SteamGameServerNetworking();
STEAM_DEFINE_GAMESERVER_INTERFACE_ACCESSOR( ISteamNetworking *, SteamGameServerNetworking, STEAMNETWORKING_INTERFACE_VERSION );

// callbacks
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 

// callback notification - a user wants to talk to us over the P2P channel via the SendP2PPacket() API
// in response, a call to AcceptP2PPacketsFromUser() needs to be made, if you want to talk with them
struct P2PSessionRequest_t
{ 
	enum { k_iCallback = k_iSteamNetworkingCallbacks + 2 };
	CSteamID m_steamIDRemote;			// user who wants to talk to us
};


// callback notification - packets can't get through to the specified user via the SendP2PPacket() API
// all packets queued packets unsent at this point will be dropped
// further attempts to send will retry making the connection (but will be dropped if we fail again)
struct P2PSessionConnectFail_t
{ 
	enum { k_iCallback = k_iSteamNetworkingCallbacks + 3 };
	CSteamID m_steamIDRemote;			// user we were sending packets to
	uint8 m_eP2PSessionError;			// EP2PSessionError indicating why we're having trouble
};


// callback notification - status of a socket has changed
// used as part of the CreateListenSocket() / CreateP2PConnectionSocket() 
struct SocketStatusCallback_t
{ 
	enum { k_iCallback = k_iSteamNetworkingCallbacks + 1 };
	SNetSocket_t m_hSocket;				// the socket used to send/receive data to the remote host
	SNetListenSocket_t m_hListenSocket;	// this is the server socket that we were listening on; NULL if this was an outgoing connection
	CSteamID m_steamIDRemote;			// remote steamID we have connected to, if it has one
	int m_eSNetSocketState;				// socket state, ESNetSocketState
};

#pragma pack( pop )

#endif // ISTEAMNETWORKING
