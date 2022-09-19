//====== Copyright © 1996-2008, Valve Corporation, All rights reserved. =======
//
// Purpose: interface to steam managing game server/client match making
//
//=============================================================================

#ifndef ISTEAMMATCHMAKING
#define ISTEAMMATCHMAKING
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"
#include "matchmakingtypes.h" 
#include "isteamfriends.h"

// lobby type description
enum ELobbyType
{
	k_ELobbyTypePrivate = 0,		// only way to join the lobby is to invite to someone else
	k_ELobbyTypeFriendsOnly = 1,	// shows for friends or invitees, but not in lobby list
	k_ELobbyTypePublic = 2,			// visible for friends and in lobby list
	k_ELobbyTypeInvisible = 3,		// returned by search, but not visible to other friends 
									//    useful if you want a user in two lobbies, for example matching groups together
									//	  a user can be in only one regular lobby, and up to two invisible lobbies
	k_ELobbyTypePrivateUnique = 4,	// private, unique and does not delete when empty - only one of these may exist per unique keypair set
									// can only create from webapi
};

// lobby search filter tools
enum ELobbyComparison
{
	k_ELobbyComparisonEqualToOrLessThan = -2,
	k_ELobbyComparisonLessThan = -1,
	k_ELobbyComparisonEqual = 0,
	k_ELobbyComparisonGreaterThan = 1,
	k_ELobbyComparisonEqualToOrGreaterThan = 2,
	k_ELobbyComparisonNotEqual = 3,
};

// lobby search distance. Lobby results are sorted from closest to farthest.
enum ELobbyDistanceFilter
{
	k_ELobbyDistanceFilterClose,		// only lobbies in the same immediate region will be returned
	k_ELobbyDistanceFilterDefault,		// only lobbies in the same region or near by regions
	k_ELobbyDistanceFilterFar,			// for games that don't have many latency requirements, will return lobbies about half-way around the globe
	k_ELobbyDistanceFilterWorldwide,	// no filtering, will match lobbies as far as India to NY (not recommended, expect multiple seconds of latency between the clients)
};

// maximum number of characters a lobby metadata key can be
#define k_nMaxLobbyKeyLength 255

//-----------------------------------------------------------------------------
// Purpose: Functions for match making services for clients to get to favorites
//			and to operate on game lobbies.
//-----------------------------------------------------------------------------
class ISteamMatchmaking
{
public:
	// game server favorites storage
	// saves basic details about a multiplayer game server locally

	// returns the number of favorites servers the user has stored
	virtual int GetFavoriteGameCount() = 0;
	
	// returns the details of the game server
	// iGame is of range [0,GetFavoriteGameCount())
	// *pnIP, *pnConnPort are filled in the with IP:port of the game server
	// *punFlags specify whether the game server was stored as an explicit favorite or in the history of connections
	// *pRTime32LastPlayedOnServer is filled in the with the Unix time the favorite was added
	virtual bool GetFavoriteGame( int iGame, AppId_t *pnAppID, uint32 *pnIP, uint16 *pnConnPort, uint16 *pnQueryPort, uint32 *punFlags, uint32 *pRTime32LastPlayedOnServer ) = 0;

	// adds the game server to the local list; updates the time played of the server if it already exists in the list
	virtual int AddFavoriteGame( AppId_t nAppID, uint32 nIP, uint16 nConnPort, uint16 nQueryPort, uint32 unFlags, uint32 rTime32LastPlayedOnServer ) = 0;
	
	// removes the game server from the local storage; returns true if one was removed
	virtual bool RemoveFavoriteGame( AppId_t nAppID, uint32 nIP, uint16 nConnPort, uint16 nQueryPort, uint32 unFlags ) = 0;

	///////
	// Game lobby functions

	// Get a list of relevant lobbies
	// this is an asynchronous request
	// results will be returned by LobbyMatchList_t callback & call result, with the number of lobbies found
	// this will never return lobbies that are full
	// to add more filter, the filter calls below need to be call before each and every RequestLobbyList() call
	// use the CCallResult<> object in steam_api.h to match the SteamAPICall_t call result to a function in an object, e.g.
	/*
		class CMyLobbyListManager
		{
			CCallResult<CMyLobbyListManager, LobbyMatchList_t> m_CallResultLobbyMatchList;
			void FindLobbies()
			{
				// SteamMatchmaking()->AddRequestLobbyListFilter*() functions would be called here, before RequestLobbyList()
				SteamAPICall_t hSteamAPICall = SteamMatchmaking()->RequestLobbyList();
				m_CallResultLobbyMatchList.Set( hSteamAPICall, this, &CMyLobbyListManager::OnLobbyMatchList );
			}

			void OnLobbyMatchList( LobbyMatchList_t *pLobbyMatchList, bool bIOFailure )
			{
				// lobby list has be retrieved from Steam back-end, use results
			}
		}
	*/
	// 
	STEAM_CALL_RESULT( LobbyMatchList_t )
	virtual SteamAPICall_t RequestLobbyList() = 0;
	// filters for lobbies
	// this needs to be called before RequestLobbyList() to take effect
	// these are cleared on each call to RequestLobbyList()
	virtual void AddRequestLobbyListStringFilter( const char *pchKeyToMatch, const char *pchValueToMatch, ELobbyComparison eComparisonType ) = 0;
	// numerical comparison
	virtual void AddRequestLobbyListNumericalFilter( const char *pchKeyToMatch, int nValueToMatch, ELobbyComparison eComparisonType ) = 0;
	// returns results closest to the specified value. Multiple near filters can be added, with early filters taking precedence
	virtual void AddRequestLobbyListNearValueFilter( const char *pchKeyToMatch, int nValueToBeCloseTo ) = 0;
	// returns only lobbies with the specified number of slots available
	virtual void AddRequestLobbyListFilterSlotsAvailable( int nSlotsAvailable ) = 0;
	// sets the distance for which we should search for lobbies (based on users IP address to location map on the Steam backed)
	virtual void AddRequestLobbyListDistanceFilter( ELobbyDistanceFilter eLobbyDistanceFilter ) = 0;
	// sets how many results to return, the lower the count the faster it is to download the lobby results & details to the client
	virtual void AddRequestLobbyListResultCountFilter( int cMaxResults ) = 0;

	virtual void AddRequestLobbyListCompatibleMembersFilter( CSteamID steamIDLobby ) = 0;

	// returns the CSteamID of a lobby, as retrieved by a RequestLobbyList call
	// should only be called after a LobbyMatchList_t callback is received
	// iLobby is of the range [0, LobbyMatchList_t::m_nLobbiesMatching)
	// the returned CSteamID::IsValid() will be false if iLobby is out of range
	virtual CSteamID GetLobbyByIndex( int iLobby ) = 0;

	// Create a lobby on the Steam servers.
	// If private, then the lobby will not be returned by any RequestLobbyList() call; the CSteamID
	// of the lobby will need to be communicated via game channels or via InviteUserToLobby()
	// this is an asynchronous request
	// results will be returned by LobbyCreated_t callback and call result; lobby is joined & ready to use at this point
	// a LobbyEnter_t callback will also be received (since the local user is joining their own lobby)
	STEAM_CALL_RESULT( LobbyCreated_t )
	virtual SteamAPICall_t CreateLobby( ELobbyType eLobbyType, int cMaxMembers ) = 0;

	// Joins an existing lobby
	// this is an asynchronous request
	// results will be returned by LobbyEnter_t callback & call result, check m_EChatRoomEnterResponse to see if was successful
	// lobby metadata is available to use immediately on this call completing
	STEAM_CALL_RESULT( LobbyEnter_t )
	virtual SteamAPICall_t JoinLobby( CSteamID steamIDLobby ) = 0;

	// Leave a lobby; this will take effect immediately on the client side
	// other users in the lobby will be notified by a LobbyChatUpdate_t callback
	virtual void LeaveLobby( CSteamID steamIDLobby ) = 0;

	// Invite another user to the lobby
	// the target user will receive a LobbyInvite_t callback
	// will return true if the invite is successfully sent, whether or not the target responds
	// returns false if the local user is not connected to the Steam servers
	// if the other user clicks the join link, a GameLobbyJoinRequested_t will be posted if the user is in-game,
	// or if the game isn't running yet the game will be launched with the parameter +connect_lobby <64-bit lobby id>
	virtual bool InviteUserToLobby( CSteamID steamIDLobby, CSteamID steamIDInvitee ) = 0;

	// Lobby iteration, for viewing details of users in a lobby
	// only accessible if the lobby user is a member of the specified lobby
	// persona information for other lobby members (name, avatar, etc.) will be asynchronously received
	// and accessible via ISteamFriends interface
	
	// returns the number of users in the specified lobby
	virtual int GetNumLobbyMembers( CSteamID steamIDLobby ) = 0;
	// returns the CSteamID of a user in the lobby
	// iMember is of range [0,GetNumLobbyMembers())
	// note that the current user must be in a lobby to retrieve CSteamIDs of other users in that lobby
	virtual CSteamID GetLobbyMemberByIndex( CSteamID steamIDLobby, int iMember ) = 0;

	// Get data associated with this lobby
	// takes a simple key, and returns the string associated with it
	// "" will be returned if no value is set, or if steamIDLobby is invalid
	virtual const char *GetLobbyData( CSteamID steamIDLobby, const char *pchKey ) = 0;
	// Sets a key/value pair in the lobby metadata
	// each user in the lobby will be broadcast this new value, and any new users joining will receive any existing data
	// this can be used to set lobby names, map, etc.
	// to reset a key, just set it to ""
	// other users in the lobby will receive notification of the lobby data change via a LobbyDataUpdate_t callback
	virtual bool SetLobbyData( CSteamID steamIDLobby, const char *pchKey, const char *pchValue ) = 0;

	// returns the number of metadata keys set on the specified lobby
	virtual int GetLobbyDataCount( CSteamID steamIDLobby ) = 0;

	// returns a lobby metadata key/values pair by index, of range [0, GetLobbyDataCount())
	virtual bool GetLobbyDataByIndex( CSteamID steamIDLobby, int iLobbyData, char *pchKey, int cchKeyBufferSize, char *pchValue, int cchValueBufferSize ) = 0;

	// removes a metadata key from the lobby
	virtual bool DeleteLobbyData( CSteamID steamIDLobby, const char *pchKey ) = 0;

	// Gets per-user metadata for someone in this lobby
	virtual const char *GetLobbyMemberData( CSteamID steamIDLobby, CSteamID steamIDUser, const char *pchKey ) = 0;
	// Sets per-user metadata (for the local user implicitly)
	virtual void SetLobbyMemberData( CSteamID steamIDLobby, const char *pchKey, const char *pchValue ) = 0;
	
	// Broadcasts a chat message to the all the users in the lobby
	// users in the lobby (including the local user) will receive a LobbyChatMsg_t callback
	// returns true if the message is successfully sent
	// pvMsgBody can be binary or text data, up to 4k
	// if pvMsgBody is text, cubMsgBody should be strlen( text ) + 1, to include the null terminator
	virtual bool SendLobbyChatMsg( CSteamID steamIDLobby, const void *pvMsgBody, int cubMsgBody ) = 0;
	// Get a chat message as specified in a LobbyChatMsg_t callback
	// iChatID is the LobbyChatMsg_t::m_iChatID value in the callback
	// *pSteamIDUser is filled in with the CSteamID of the member
	// *pvData is filled in with the message itself
	// return value is the number of bytes written into the buffer
	virtual int GetLobbyChatEntry( CSteamID steamIDLobby, int iChatID, STEAM_OUT_STRUCT() CSteamID *pSteamIDUser, void *pvData, int cubData, EChatEntryType *peChatEntryType ) = 0;

	// Refreshes metadata for a lobby you're not necessarily in right now
	// you never do this for lobbies you're a member of, only if your
	// this will send down all the metadata associated with a lobby
	// this is an asynchronous call
	// returns false if the local user is not connected to the Steam servers
	// results will be returned by a LobbyDataUpdate_t callback
	// if the specified lobby doesn't exist, LobbyDataUpdate_t::m_bSuccess will be set to false
	virtual bool RequestLobbyData( CSteamID steamIDLobby ) = 0;
	
	// sets the game server associated with the lobby
	// usually at this point, the users will join the specified game server
	// either the IP/Port or the steamID of the game server has to be valid, depending on how you want the clients to be able to connect
	virtual void SetLobbyGameServer( CSteamID steamIDLobby, uint32 unGameServerIP, uint16 unGameServerPort, CSteamID steamIDGameServer ) = 0;
	// returns the details of a game server set in a lobby - returns false if there is no game server set, or that lobby doesn't exist
	virtual bool GetLobbyGameServer( CSteamID steamIDLobby, uint32 *punGameServerIP, uint16 *punGameServerPort, STEAM_OUT_STRUCT() CSteamID *psteamIDGameServer ) = 0;

	// set the limit on the # of users who can join the lobby
	virtual bool SetLobbyMemberLimit( CSteamID steamIDLobby, int cMaxMembers ) = 0;
	// returns the current limit on the # of users who can join the lobby; returns 0 if no limit is defined
	virtual int GetLobbyMemberLimit( CSteamID steamIDLobby ) = 0;

	// updates which type of lobby it is
	// only lobbies that are k_ELobbyTypePublic or k_ELobbyTypeInvisible, and are set to joinable, will be returned by RequestLobbyList() calls
	virtual bool SetLobbyType( CSteamID steamIDLobby, ELobbyType eLobbyType ) = 0;

	// sets whether or not a lobby is joinable - defaults to true for a new lobby
	// if set to false, no user can join, even if they are a friend or have been invited
	virtual bool SetLobbyJoinable( CSteamID steamIDLobby, bool bLobbyJoinable ) = 0;

	// returns the current lobby owner
	// you must be a member of the lobby to access this
	// there always one lobby owner - if the current owner leaves, another user will become the owner
	// it is possible (bur rare) to join a lobby just as the owner is leaving, thus entering a lobby with self as the owner
	virtual CSteamID GetLobbyOwner( CSteamID steamIDLobby ) = 0;

	// changes who the lobby owner is
	// you must be the lobby owner for this to succeed, and steamIDNewOwner must be in the lobby
	// after completion, the local user will no longer be the owner
	virtual bool SetLobbyOwner( CSteamID steamIDLobby, CSteamID steamIDNewOwner ) = 0;

	// link two lobbies for the purposes of checking player compatibility
	// you must be the lobby owner of both lobbies
	virtual bool SetLinkedLobby( CSteamID steamIDLobby, CSteamID steamIDLobbyDependent ) = 0;

#ifdef _PS3
	// changes who the lobby owner is
	// you must be the lobby owner for this to succeed, and steamIDNewOwner must be in the lobby
	// after completion, the local user will no longer be the owner
	virtual void CheckForPSNGameBootInvite( unsigned int iGameBootAttributes  ) = 0;
#endif
};
#define STEAMMATCHMAKING_INTERFACE_VERSION "SteamMatchMaking009"

// Global interface accessor
inline ISteamMatchmaking *SteamMatchmaking();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamMatchmaking *, SteamMatchmaking, STEAMMATCHMAKING_INTERFACE_VERSION );

//-----------------------------------------------------------------------------
// Callback interfaces for server list functions (see ISteamMatchmakingServers below)
//
// The idea here is that your game code implements objects that implement these
// interfaces to receive callback notifications after calling asynchronous functions
// inside the ISteamMatchmakingServers() interface below.
//
// This is different than normal Steam callback handling due to the potentially
// large size of server lists.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Typedef for handle type you will receive when requesting server list.
//-----------------------------------------------------------------------------
typedef void* HServerListRequest;

//-----------------------------------------------------------------------------
// Purpose: Callback interface for receiving responses after a server list refresh
// or an individual server update.
//
// Since you get these callbacks after requesting full list refreshes you will
// usually implement this interface inside an object like CServerBrowser.  If that
// object is getting destructed you should use ISteamMatchMakingServers()->CancelQuery()
// to cancel any in-progress queries so you don't get a callback into the destructed
// object and crash.
//-----------------------------------------------------------------------------
class ISteamMatchmakingServerListResponse
{
public:
	// Server has responded ok with updated data
	virtual void ServerResponded( HServerListRequest hRequest, int iServer ) = 0; 

	// Server has failed to respond
	virtual void ServerFailedToRespond( HServerListRequest hRequest, int iServer ) = 0; 

	// A list refresh you had initiated is now 100% completed
	virtual void RefreshComplete( HServerListRequest hRequest, EMatchMakingServerResponse response ) = 0; 
};


//-----------------------------------------------------------------------------
// Purpose: Callback interface for receiving responses after pinging an individual server 
//
// These callbacks all occur in response to querying an individual server
// via the ISteamMatchmakingServers()->PingServer() call below.  If you are 
// destructing an object that implements this interface then you should call 
// ISteamMatchmakingServers()->CancelServerQuery() passing in the handle to the query
// which is in progress.  Failure to cancel in progress queries when destructing
// a callback handler may result in a crash when a callback later occurs.
//-----------------------------------------------------------------------------
class ISteamMatchmakingPingResponse
{
public:
	// Server has responded successfully and has updated data
	virtual void ServerResponded( gameserveritem_t &server ) = 0;

	// Server failed to respond to the ping request
	virtual void ServerFailedToRespond() = 0;
};


//-----------------------------------------------------------------------------
// Purpose: Callback interface for receiving responses after requesting details on
// who is playing on a particular server.
//
// These callbacks all occur in response to querying an individual server
// via the ISteamMatchmakingServers()->PlayerDetails() call below.  If you are 
// destructing an object that implements this interface then you should call 
// ISteamMatchmakingServers()->CancelServerQuery() passing in the handle to the query
// which is in progress.  Failure to cancel in progress queries when destructing
// a callback handler may result in a crash when a callback later occurs.
//-----------------------------------------------------------------------------
class ISteamMatchmakingPlayersResponse
{
public:
	// Got data on a new player on the server -- you'll get this callback once per player
	// on the server which you have requested player data on.
	virtual void AddPlayerToList( const char *pchName, int nScore, float flTimePlayed ) = 0;

	// The server failed to respond to the request for player details
	virtual void PlayersFailedToRespond() = 0;

	// The server has finished responding to the player details request 
	// (ie, you won't get anymore AddPlayerToList callbacks)
	virtual void PlayersRefreshComplete() = 0;
};


//-----------------------------------------------------------------------------
// Purpose: Callback interface for receiving responses after requesting rules
// details on a particular server.
//
// These callbacks all occur in response to querying an individual server
// via the ISteamMatchmakingServers()->ServerRules() call below.  If you are 
// destructing an object that implements this interface then you should call 
// ISteamMatchmakingServers()->CancelServerQuery() passing in the handle to the query
// which is in progress.  Failure to cancel in progress queries when destructing
// a callback handler may result in a crash when a callback later occurs.
//-----------------------------------------------------------------------------
class ISteamMatchmakingRulesResponse
{
public:
	// Got data on a rule on the server -- you'll get one of these per rule defined on
	// the server you are querying
	virtual void RulesResponded( const char *pchRule, const char *pchValue ) = 0;

	// The server failed to respond to the request for rule details
	virtual void RulesFailedToRespond() = 0;

	// The server has finished responding to the rule details request 
	// (ie, you won't get anymore RulesResponded callbacks)
	virtual void RulesRefreshComplete() = 0;
};


//-----------------------------------------------------------------------------
// Typedef for handle type you will receive when querying details on an individual server.
//-----------------------------------------------------------------------------
typedef int HServerQuery;
const int HSERVERQUERY_INVALID = 0xffffffff;

//-----------------------------------------------------------------------------
// Purpose: Functions for match making services for clients to get to game lists and details
//-----------------------------------------------------------------------------
class ISteamMatchmakingServers
{
public:
	// Request a new list of servers of a particular type.  These calls each correspond to one of the EMatchMakingType values.
	// Each call allocates a new asynchronous request object.
	// Request object must be released by calling ReleaseRequest( hServerListRequest )
	virtual HServerListRequest RequestInternetServerList( AppId_t iApp, STEAM_ARRAY_COUNT(nFilters) MatchMakingKeyValuePair_t **ppchFilters, uint32 nFilters, ISteamMatchmakingServerListResponse *pRequestServersResponse ) = 0;
	virtual HServerListRequest RequestLANServerList( AppId_t iApp, ISteamMatchmakingServerListResponse *pRequestServersResponse ) = 0;
	virtual HServerListRequest RequestFriendsServerList( AppId_t iApp, STEAM_ARRAY_COUNT(nFilters) MatchMakingKeyValuePair_t **ppchFilters, uint32 nFilters, ISteamMatchmakingServerListResponse *pRequestServersResponse ) = 0;
	virtual HServerListRequest RequestFavoritesServerList( AppId_t iApp, STEAM_ARRAY_COUNT(nFilters) MatchMakingKeyValuePair_t **ppchFilters, uint32 nFilters, ISteamMatchmakingServerListResponse *pRequestServersResponse ) = 0;
	virtual HServerListRequest RequestHistoryServerList( AppId_t iApp, STEAM_ARRAY_COUNT(nFilters) MatchMakingKeyValuePair_t **ppchFilters, uint32 nFilters, ISteamMatchmakingServerListResponse *pRequestServersResponse ) = 0;
	virtual HServerListRequest RequestSpectatorServerList( AppId_t iApp, STEAM_ARRAY_COUNT(nFilters) MatchMakingKeyValuePair_t **ppchFilters, uint32 nFilters, ISteamMatchmakingServerListResponse *pRequestServersResponse ) = 0;

	// Releases the asynchronous request object and cancels any pending query on it if there's a pending query in progress.
	// RefreshComplete callback is not posted when request is released.
	virtual void ReleaseRequest( HServerListRequest hServerListRequest ) = 0;

	/* the filter operation codes that go in the key part of MatchMakingKeyValuePair_t should be one of these:

		"map"
			- Server passes the filter if the server is playing the specified map.
		"gamedataand"
			- Server passes the filter if the server's game data (ISteamGameServer::SetGameData) contains all of the
			specified strings.  The value field is a comma-delimited list of strings to match.
		"gamedataor"
			- Server passes the filter if the server's game data (ISteamGameServer::SetGameData) contains at least one of the
			specified strings.  The value field is a comma-delimited list of strings to match.
		"gamedatanor"
			- Server passes the filter if the server's game data (ISteamGameServer::SetGameData) does not contain any
			of the specified strings.  The value field is a comma-delimited list of strings to check.
		"gametagsand"
			- Server passes the filter if the server's game tags (ISteamGameServer::SetGameTags) contains all
			of the specified strings.  The value field is a comma-delimited list of strings to check.
		"gametagsnor"
			- Server passes the filter if the server's game tags (ISteamGameServer::SetGameTags) does not contain any
			of the specified strings.  The value field is a comma-delimited list of strings to check.
		"and" (x1 && x2 && ... && xn)
		"or" (x1 || x2 || ... || xn)
		"nand" !(x1 && x2 && ... && xn)
		"nor" !(x1 || x2 || ... || xn)
			- Performs Boolean operation on the following filters.  The operand to this filter specifies
			the "size" of the Boolean inputs to the operation, in Key/value pairs.  (The keyvalue
			pairs must immediately follow, i.e. this is a prefix logical operator notation.)
			In the simplest case where Boolean expressions are not nested, this is simply
			the number of operands.

			For example, to match servers on a particular map or with a particular tag, would would
			use these filters.

				( server.map == "cp_dustbowl" || server.gametags.contains("payload") )
				"or", "2"
				"map", "cp_dustbowl"
				"gametagsand", "payload"

			If logical inputs are nested, then the operand specifies the size of the entire
			"length" of its operands, not the number of immediate children.

				( server.map == "cp_dustbowl" || ( server.gametags.contains("payload") && !server.gametags.contains("payloadrace") ) )
				"or", "4"
				"map", "cp_dustbowl"
				"and", "2"
				"gametagsand", "payload"
				"gametagsnor", "payloadrace"

			Unary NOT can be achieved using either "nand" or "nor" with a single operand.

		"addr"
			- Server passes the filter if the server's query address matches the specified IP or IP:port.
		"gameaddr"
			- Server passes the filter if the server's game address matches the specified IP or IP:port.

		The following filter operations ignore the "value" part of MatchMakingKeyValuePair_t

		"dedicated"
			- Server passes the filter if it passed true to SetDedicatedServer.
		"secure"
			- Server passes the filter if the server is VAC-enabled.
		"notfull"
			- Server passes the filter if the player count is less than the reported max player count.
		"hasplayers"
			- Server passes the filter if the player count is greater than zero.
		"noplayers"
			- Server passes the filter if it doesn't have any players.
		"linux"
			- Server passes the filter if it's a linux server
	*/

	// Get details on a given server in the list, you can get the valid range of index
	// values by calling GetServerCount().  You will also receive index values in 
	// ISteamMatchmakingServerListResponse::ServerResponded() callbacks
	virtual gameserveritem_t *GetServerDetails( HServerListRequest hRequest, int iServer ) = 0; 

	// Cancel an request which is operation on the given list type.  You should call this to cancel
	// any in-progress requests before destructing a callback object that may have been passed 
	// to one of the above list request calls.  Not doing so may result in a crash when a callback
	// occurs on the destructed object.
	// Canceling a query does not release the allocated request handle.
	// The request handle must be released using ReleaseRequest( hRequest )
	virtual void CancelQuery( HServerListRequest hRequest ) = 0; 

	// Ping every server in your list again but don't update the list of servers
	// Query callback installed when the server list was requested will be used
	// again to post notifications and RefreshComplete, so the callback must remain
	// valid until another RefreshComplete is called on it or the request
	// is released with ReleaseRequest( hRequest )
	virtual void RefreshQuery( HServerListRequest hRequest ) = 0; 

	// Returns true if the list is currently refreshing its server list
	virtual bool IsRefreshing( HServerListRequest hRequest ) = 0; 

	// How many servers in the given list, GetServerDetails above takes 0... GetServerCount() - 1
	virtual int GetServerCount( HServerListRequest hRequest ) = 0; 

	// Refresh a single server inside of a query (rather than all the servers )
	virtual void RefreshServer( HServerListRequest hRequest, int iServer ) = 0; 


	//-----------------------------------------------------------------------------
	// Queries to individual servers directly via IP/Port
	//-----------------------------------------------------------------------------

	// Request updated ping time and other details from a single server
	virtual HServerQuery PingServer( uint32 unIP, uint16 usPort, ISteamMatchmakingPingResponse *pRequestServersResponse ) = 0; 

	// Request the list of players currently playing on a server
	virtual HServerQuery PlayerDetails( uint32 unIP, uint16 usPort, ISteamMatchmakingPlayersResponse *pRequestServersResponse ) = 0;

	// Request the list of rules that the server is running (See ISteamGameServer::SetKeyValue() to set the rules server side)
	virtual HServerQuery ServerRules( uint32 unIP, uint16 usPort, ISteamMatchmakingRulesResponse *pRequestServersResponse ) = 0; 

	// Cancel an outstanding Ping/Players/Rules query from above.  You should call this to cancel
	// any in-progress requests before destructing a callback object that may have been passed 
	// to one of the above calls to avoid crashing when callbacks occur.
	virtual void CancelServerQuery( HServerQuery hServerQuery ) = 0; 
};
#define STEAMMATCHMAKINGSERVERS_INTERFACE_VERSION "SteamMatchMakingServers002"

// Global interface accessor
inline ISteamMatchmakingServers *SteamMatchmakingServers();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamMatchmakingServers *, SteamMatchmakingServers, STEAMMATCHMAKINGSERVERS_INTERFACE_VERSION );

// game server flags
const uint32 k_unFavoriteFlagNone			= 0x00;
const uint32 k_unFavoriteFlagFavorite		= 0x01; // this game favorite entry is for the favorites list
const uint32 k_unFavoriteFlagHistory		= 0x02; // this game favorite entry is for the history list


//-----------------------------------------------------------------------------
// Purpose: Used in ChatInfo messages - fields specific to a chat member - must fit in a uint32
//-----------------------------------------------------------------------------
enum EChatMemberStateChange
{
	// Specific to joining / leaving the chatroom
	k_EChatMemberStateChangeEntered			= 0x0001,		// This user has joined or is joining the chat room
	k_EChatMemberStateChangeLeft			= 0x0002,		// This user has left or is leaving the chat room
	k_EChatMemberStateChangeDisconnected	= 0x0004,		// User disconnected without leaving the chat first
	k_EChatMemberStateChangeKicked			= 0x0008,		// User kicked
	k_EChatMemberStateChangeBanned			= 0x0010,		// User kicked and banned
};

// returns true of the flags indicate that a user has been removed from the chat
#define BChatMemberStateChangeRemoved( rgfChatMemberStateChangeFlags ) ( rgfChatMemberStateChangeFlags & ( k_EChatMemberStateChangeDisconnected | k_EChatMemberStateChangeLeft | k_EChatMemberStateChangeKicked | k_EChatMemberStateChangeBanned ) )



//-----------------------------------------------------------------------------
// Purpose: Functions for match making services for clients to get to favorites
//			and to operate on game lobbies.
//-----------------------------------------------------------------------------
class ISteamGameSearch
{
public:
	// =============================================================================================
	// Game Player APIs

	// a keyname and a list of comma separated values: one of which is must be found in order for the match to qualify
	// fails if a search is currently in progress
	virtual EGameSearchErrorCode_t AddGameSearchParams( const char *pchKeyToFind, const char *pchValuesToFind ) = 0;

	// all players in lobby enter the queue and await a SearchForGameNotificationCallback_t callback. fails if another search is currently in progress
	// if not the owner of the lobby or search already in progress this call fails
	// periodic callbacks will be sent as queue time estimates change
	virtual EGameSearchErrorCode_t SearchForGameWithLobby( CSteamID steamIDLobby, int nPlayerMin, int nPlayerMax ) = 0;

	// user enter the queue and await a SearchForGameNotificationCallback_t callback. fails if another search is currently in progress
	// periodic callbacks will be sent as queue time estimates change
	virtual EGameSearchErrorCode_t SearchForGameSolo( int nPlayerMin, int nPlayerMax ) = 0;

	// after receiving SearchForGameResultCallback_t, accept or decline the game
	// multiple SearchForGameResultCallback_t will follow as players accept game until the host starts or cancels the game
	virtual EGameSearchErrorCode_t AcceptGame() = 0;
	virtual EGameSearchErrorCode_t DeclineGame() = 0;

	// after receiving GameStartedByHostCallback_t get connection details to server
	virtual EGameSearchErrorCode_t RetrieveConnectionDetails( CSteamID steamIDHost, char *pchConnectionDetails, int cubConnectionDetails ) = 0;

	// leaves queue if still waiting
	virtual EGameSearchErrorCode_t EndGameSearch() = 0;

	// =============================================================================================
	// Game Host APIs

	// a keyname and a list of comma separated values: all the values you allow
	virtual EGameSearchErrorCode_t SetGameHostParams( const char *pchKey, const char *pchValue ) = 0;

	// set connection details for players once game is found so they can connect to this server
	virtual EGameSearchErrorCode_t SetConnectionDetails( const char *pchConnectionDetails, int cubConnectionDetails ) = 0;

	// mark server as available for more players with nPlayerMin,nPlayerMax desired
	// accept no lobbies with playercount greater than nMaxTeamSize
	// the set of lobbies returned must be partitionable into teams of no more than nMaxTeamSize
	// RequestPlayersForGameNotificationCallback_t callback will be sent when the search has started
	// multple RequestPlayersForGameResultCallback_t callbacks will follow when players are found
	virtual EGameSearchErrorCode_t RequestPlayersForGame( int nPlayerMin, int nPlayerMax, int nMaxTeamSize ) = 0;

	// accept the player list and release connection details to players
	// players will only be given connection details and host steamid when this is called
	// ( allows host to accept after all players confirm, some confirm, or none confirm. decision is entirely up to the host )
	virtual EGameSearchErrorCode_t HostConfirmGameStart( uint64 ullUniqueGameID ) = 0;

	// cancel request and leave the pool of game hosts looking for players
	// if a set of players has already been sent to host, all players will receive SearchForGameHostFailedToConfirm_t
	virtual EGameSearchErrorCode_t CancelRequestPlayersForGame() = 0;

	// submit a result for one player. does not end the game. ullUniqueGameID continues to describe this game
	virtual EGameSearchErrorCode_t SubmitPlayerResult( uint64 ullUniqueGameID, CSteamID steamIDPlayer, EPlayerResult_t EPlayerResult ) = 0;

	// ends the game. no further SubmitPlayerResults for ullUniqueGameID will be accepted
	// any future requests will provide a new ullUniqueGameID
	virtual EGameSearchErrorCode_t EndGame( uint64 ullUniqueGameID ) = 0;

};
#define STEAMGAMESEARCH_INTERFACE_VERSION "SteamMatchGameSearch001"

// Global interface accessor
inline ISteamGameSearch *SteamGameSearch();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamGameSearch *, SteamGameSearch, STEAMGAMESEARCH_INTERFACE_VERSION );


//-----------------------------------------------------------------------------
// Purpose: Functions for quickly creating a Party with friends or acquaintances,
//			EG from chat rooms.
//-----------------------------------------------------------------------------
enum ESteamPartyBeaconLocationType
{
	k_ESteamPartyBeaconLocationType_Invalid = 0,
	k_ESteamPartyBeaconLocationType_ChatGroup = 1,

	k_ESteamPartyBeaconLocationType_Max,
};


#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 


struct SteamPartyBeaconLocation_t
{
	ESteamPartyBeaconLocationType m_eType;
	uint64 m_ulLocationID;
};

enum ESteamPartyBeaconLocationData
{
	k_ESteamPartyBeaconLocationDataInvalid = 0,
	k_ESteamPartyBeaconLocationDataName = 1,
	k_ESteamPartyBeaconLocationDataIconURLSmall = 2,
	k_ESteamPartyBeaconLocationDataIconURLMedium = 3,
	k_ESteamPartyBeaconLocationDataIconURLLarge = 4,
};

class ISteamParties
{
public:

	// =============================================================================================
	// Party Client APIs
	
	// Enumerate any active beacons for parties you may wish to join
	virtual uint32 GetNumActiveBeacons() = 0;
	virtual PartyBeaconID_t GetBeaconByIndex( uint32 unIndex ) = 0;
	virtual bool GetBeaconDetails( PartyBeaconID_t ulBeaconID, CSteamID *pSteamIDBeaconOwner, STEAM_OUT_STRUCT() SteamPartyBeaconLocation_t *pLocation, STEAM_OUT_STRING_COUNT(cchMetadata) char *pchMetadata, int cchMetadata ) = 0;

	// Join an open party. Steam will reserve one beacon slot for your SteamID,
	// and return the necessary JoinGame string for you to use to connect
	STEAM_CALL_RESULT( JoinPartyCallback_t )
	virtual SteamAPICall_t JoinParty( PartyBeaconID_t ulBeaconID ) = 0;

	// =============================================================================================
	// Party Host APIs

	// Get a list of possible beacon locations
	virtual bool GetNumAvailableBeaconLocations( uint32 *puNumLocations ) = 0;
	virtual bool GetAvailableBeaconLocations( SteamPartyBeaconLocation_t *pLocationList, uint32 uMaxNumLocations ) = 0;

	// Create a new party beacon and activate it in the selected location.
	// unOpenSlots is the maximum number of users that Steam will send to you.
	// When people begin responding to your beacon, Steam will send you
	// PartyReservationCallback_t callbacks to let you know who is on the way.
	STEAM_CALL_RESULT( CreateBeaconCallback_t )
	virtual SteamAPICall_t CreateBeacon( uint32 unOpenSlots, SteamPartyBeaconLocation_t *pBeaconLocation, const char *pchConnectString, const char *pchMetadata ) = 0;

	// Call this function when a user that had a reservation (see callback below) 
	// has successfully joined your party.
	// Steam will manage the remaining open slots automatically.
	virtual void OnReservationCompleted( PartyBeaconID_t ulBeacon, CSteamID steamIDUser ) = 0;

	// To cancel a reservation (due to timeout or user input), call this.
	// Steam will open a new reservation slot.
	// Note: The user may already be in-flight to your game, so it's possible they will still connect and try to join your party.
	virtual void CancelReservation( PartyBeaconID_t ulBeacon, CSteamID steamIDUser ) = 0;

	// Change the number of open beacon reservation slots.
	// Call this if, for example, someone without a reservation joins your party (eg a friend, or via your own matchmaking system).
	STEAM_CALL_RESULT( ChangeNumOpenSlotsCallback_t )
	virtual SteamAPICall_t ChangeNumOpenSlots( PartyBeaconID_t ulBeacon, uint32 unOpenSlots ) = 0;

	// Turn off the beacon. 
	virtual bool DestroyBeacon( PartyBeaconID_t ulBeacon ) = 0;

	// Utils
	virtual bool GetBeaconLocationData( SteamPartyBeaconLocation_t BeaconLocation, ESteamPartyBeaconLocationData eData, STEAM_OUT_STRING_COUNT(cchDataStringOut) char *pchDataStringOut, int cchDataStringOut ) = 0;

};
#define STEAMPARTIES_INTERFACE_VERSION "SteamParties002"

// Global interface accessor
inline ISteamParties *SteamParties();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamParties *, SteamParties, STEAMPARTIES_INTERFACE_VERSION );


//-----------------------------------------------------------------------------
// Callbacks for ISteamMatchmaking (which go through the regular Steam callback registration system)

//-----------------------------------------------------------------------------
// Purpose: a server was added/removed from the favorites list, you should refresh now
//-----------------------------------------------------------------------------
struct FavoritesListChanged_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 2 };
	uint32 m_nIP; // an IP of 0 means reload the whole list, any other value means just one server
	uint32 m_nQueryPort;
	uint32 m_nConnPort;
	uint32 m_nAppID;
	uint32 m_nFlags;
	bool m_bAdd; // true if this is adding the entry, otherwise it is a remove
	AccountID_t m_unAccountId;
};


//-----------------------------------------------------------------------------
// Purpose: Someone has invited you to join a Lobby
//			normally you don't need to do anything with this, since
//			the Steam UI will also display a '<user> has invited you to the lobby, join?' dialog
//
//			if the user outside a game chooses to join, your game will be launched with the parameter "+connect_lobby <64-bit lobby id>",
//			or with the callback GameLobbyJoinRequested_t if they're already in-game
//-----------------------------------------------------------------------------
struct LobbyInvite_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 3 };

	uint64 m_ulSteamIDUser;		// Steam ID of the person making the invite
	uint64 m_ulSteamIDLobby;	// Steam ID of the Lobby
	uint64 m_ulGameID;			// GameID of the Lobby
};


//-----------------------------------------------------------------------------
// Purpose: Sent on entering a lobby, or on failing to enter
//			m_EChatRoomEnterResponse will be set to k_EChatRoomEnterResponseSuccess on success,
//			or a higher value on failure (see enum EChatRoomEnterResponse)
//-----------------------------------------------------------------------------
struct LobbyEnter_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 4 };

	uint64 m_ulSteamIDLobby;							// SteamID of the Lobby you have entered
	uint32 m_rgfChatPermissions;						// Permissions of the current user
	bool m_bLocked;										// If true, then only invited users may join
	uint32 m_EChatRoomEnterResponse;	// EChatRoomEnterResponse
};


//-----------------------------------------------------------------------------
// Purpose: The lobby metadata has changed
//			if m_ulSteamIDMember is the steamID of a lobby member, use GetLobbyMemberData() to access per-user details
//			if m_ulSteamIDMember == m_ulSteamIDLobby, use GetLobbyData() to access lobby metadata
//-----------------------------------------------------------------------------
struct LobbyDataUpdate_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 5 };

	uint64 m_ulSteamIDLobby;		// steamID of the Lobby
	uint64 m_ulSteamIDMember;		// steamID of the member whose data changed, or the room itself
	uint8 m_bSuccess;				// true if we lobby data was successfully changed; 
									// will only be false if RequestLobbyData() was called on a lobby that no longer exists
};


//-----------------------------------------------------------------------------
// Purpose: The lobby chat room state has changed
//			this is usually sent when a user has joined or left the lobby
//-----------------------------------------------------------------------------
struct LobbyChatUpdate_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 6 };

	uint64 m_ulSteamIDLobby;			// Lobby ID
	uint64 m_ulSteamIDUserChanged;		// user who's status in the lobby just changed - can be recipient
	uint64 m_ulSteamIDMakingChange;		// Chat member who made the change (different from SteamIDUserChange if kicking, muting, etc.)
										// for example, if one user kicks another from the lobby, this will be set to the id of the user who initiated the kick
	uint32 m_rgfChatMemberStateChange;	// bitfield of EChatMemberStateChange values
};


//-----------------------------------------------------------------------------
// Purpose: A chat message for this lobby has been sent
//			use GetLobbyChatEntry( m_iChatID ) to retrieve the contents of this message
//-----------------------------------------------------------------------------
struct LobbyChatMsg_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 7 };

	uint64 m_ulSteamIDLobby;			// the lobby id this is in
	uint64 m_ulSteamIDUser;			// steamID of the user who has sent this message
	uint8 m_eChatEntryType;			// type of message
	uint32 m_iChatID;				// index of the chat entry to lookup
};


//-----------------------------------------------------------------------------
// Purpose: A game created a game for all the members of the lobby to join,
//			as triggered by a SetLobbyGameServer()
//			it's up to the individual clients to take action on this; the usual
//			game behavior is to leave the lobby and connect to the specified game server
//-----------------------------------------------------------------------------
struct LobbyGameCreated_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 9 };

	uint64 m_ulSteamIDLobby;		// the lobby we were in
	uint64 m_ulSteamIDGameServer;	// the new game server that has been created or found for the lobby members
	uint32 m_unIP;					// IP & Port of the game server (if any)
	uint16 m_usPort;
};


//-----------------------------------------------------------------------------
// Purpose: Number of matching lobbies found
//			iterate the returned lobbies with GetLobbyByIndex(), from values 0 to m_nLobbiesMatching-1
//-----------------------------------------------------------------------------
struct LobbyMatchList_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 10 };
	uint32 m_nLobbiesMatching;		// Number of lobbies that matched search criteria and we have SteamIDs for
};


//-----------------------------------------------------------------------------
// Purpose: posted if a user is forcefully removed from a lobby
//			can occur if a user loses connection to Steam
//-----------------------------------------------------------------------------
struct LobbyKicked_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 12 };
	uint64 m_ulSteamIDLobby;			// Lobby
	uint64 m_ulSteamIDAdmin;			// User who kicked you - possibly the ID of the lobby itself
	uint8 m_bKickedDueToDisconnect;		// true if you were kicked from the lobby due to the user losing connection to Steam (currently always true)
};


//-----------------------------------------------------------------------------
// Purpose: Result of our request to create a Lobby
//			m_eResult == k_EResultOK on success
//			at this point, the lobby has been joined and is ready for use
//			a LobbyEnter_t callback will also be received (since the local user is joining their own lobby)
//-----------------------------------------------------------------------------
struct LobbyCreated_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 13 };
	
	EResult m_eResult;		// k_EResultOK - the lobby was successfully created
							// k_EResultNoConnection - your Steam client doesn't have a connection to the back-end
							// k_EResultTimeout - you the message to the Steam servers, but it didn't respond
							// k_EResultFail - the server responded, but with an unknown internal error
							// k_EResultAccessDenied - your game isn't set to allow lobbies, or your client does haven't rights to play the game
							// k_EResultLimitExceeded - your game client has created too many lobbies

	uint64 m_ulSteamIDLobby;		// chat room, zero if failed
};

// used by now obsolete RequestFriendsLobbiesResponse_t
// enum { k_iCallback = k_iSteamMatchmakingCallbacks + 14 };


//-----------------------------------------------------------------------------
// Purpose: Result of CheckForPSNGameBootInvite
//			m_eResult == k_EResultOK on success
//			at this point, the local user may not have finishing joining this lobby;
//			game code should wait until the subsequent LobbyEnter_t callback is received
//-----------------------------------------------------------------------------
struct PSNGameBootInviteResult_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 15 };

	bool m_bGameBootInviteExists;
	CSteamID m_steamIDLobby;		// Should be valid if m_bGameBootInviteExists == true
};


//-----------------------------------------------------------------------------
// Purpose: Result of our request to create a Lobby
//			m_eResult == k_EResultOK on success
//			at this point, the lobby has been joined and is ready for use
//			a LobbyEnter_t callback will also be received (since the local user is joining their own lobby)
//-----------------------------------------------------------------------------
struct FavoritesListAccountsUpdated_t
{
	enum { k_iCallback = k_iSteamMatchmakingCallbacks + 16 };
	
	EResult m_eResult;
};



//-----------------------------------------------------------------------------
// Callbacks for ISteamGameSearch (which go through the regular Steam callback registration system)

struct SearchForGameProgressCallback_t
{
	enum { k_iCallback = k_iSteamGameSearchCallbacks + 1 };

	uint64  m_ullSearchID;	// all future callbacks referencing this search will include this Search ID

	EResult m_eResult; // if search has started this result will be k_EResultOK, any other value indicates search has failed to start or has terminated
	CSteamID m_lobbyID; // lobby ID if lobby search, invalid steamID otherwise
	CSteamID m_steamIDEndedSearch; // if search was terminated, steamID that terminated search

	int32	m_nSecondsRemainingEstimate;
	int32	m_cPlayersSearching;
};

// notification to all players searching that a game has been found
struct SearchForGameResultCallback_t
{
	enum { k_iCallback = k_iSteamGameSearchCallbacks + 2 };

	uint64  m_ullSearchID;

	EResult m_eResult; // if game/host was lost this will be an error value

	// if m_bGameFound is true the following are non-zero
	int32 m_nCountPlayersInGame;
	int32 m_nCountAcceptedGame;
	// if m_steamIDHost is valid the host has started the game
	CSteamID m_steamIDHost;
	bool m_bFinalCallback;
};


//-----------------------------------------------------------------------------
// ISteamGameSearch : Game Host API callbacks

// callback from RequestPlayersForGame when the matchmaking service has started or ended search
// callback will also follow a call from CancelRequestPlayersForGame - m_bSearchInProgress will be false
struct RequestPlayersForGameProgressCallback_t
{
	enum { k_iCallback = k_iSteamGameSearchCallbacks + 11 };

	EResult m_eResult;		// m_ullSearchID will be non-zero if this is k_EResultOK
	uint64  m_ullSearchID; 	// all future callbacks referencing this search will include this Search ID
};

// callback from RequestPlayersForGame
// one of these will be sent per player 
// followed by additional callbacks when players accept or decline the game
struct RequestPlayersForGameResultCallback_t
{
	enum { k_iCallback = k_iSteamGameSearchCallbacks + 12 };

	EResult m_eResult;		// m_ullSearchID will be non-zero if this is k_EResultOK
	uint64  m_ullSearchID;

	CSteamID m_SteamIDPlayerFound; // player steamID
	CSteamID m_SteamIDLobby;	// if the player is in a lobby, the lobby ID
	enum PlayerAcceptState_t
	{
		k_EStateUnknown = 0,
		k_EStatePlayerAccepted = 1,
		k_EStatePlayerDeclined = 2,
	};
	PlayerAcceptState_t m_ePlayerAcceptState;
	int32 m_nPlayerIndex;
	int32 m_nTotalPlayersFound;		// expect this many callbacks at minimum
	int32 m_nTotalPlayersAcceptedGame;
	int32 m_nSuggestedTeamIndex;
	uint64 m_ullUniqueGameID;
};


struct RequestPlayersForGameFinalResultCallback_t
{
	enum { k_iCallback = k_iSteamGameSearchCallbacks + 13 };

	EResult m_eResult;
	uint64  m_ullSearchID;
	uint64 m_ullUniqueGameID;
};



// this callback confirms that results were received by the matchmaking service for this player
struct SubmitPlayerResultResultCallback_t
{
	enum { k_iCallback = k_iSteamGameSearchCallbacks + 14 };

	EResult m_eResult;
	uint64 ullUniqueGameID;
	CSteamID steamIDPlayer;
};


// this callback confirms that the game is recorded as complete on the matchmaking service
// the next call to RequestPlayersForGame will generate a new unique game ID
struct EndGameResultCallback_t
{
	enum { k_iCallback = k_iSteamGameSearchCallbacks + 15 };

	EResult m_eResult;
	uint64 ullUniqueGameID;
};


// Steam has responded to the user request to join a party via the given Beacon ID.
// If successful, the connect string contains game-specific instructions to connect
// to the game with that party.
struct JoinPartyCallback_t
{
	enum { k_iCallback = k_iSteamPartiesCallbacks + 1 };

	EResult m_eResult;
	PartyBeaconID_t m_ulBeaconID;
	CSteamID m_SteamIDBeaconOwner;
	char m_rgchConnectString[256];
};

// Response to CreateBeacon request. If successful, the beacon ID is provided.
struct CreateBeaconCallback_t
{
	enum { k_iCallback = k_iSteamPartiesCallbacks + 2 };

	EResult m_eResult;
	PartyBeaconID_t m_ulBeaconID;
};

// Someone has used the beacon to join your party - they are in-flight now
// and we've reserved one of the open slots for them.
// You should confirm when they join your party by calling OnReservationCompleted().
// Otherwise, Steam may timeout their reservation eventually.
struct ReservationNotificationCallback_t
{
	enum { k_iCallback = k_iSteamPartiesCallbacks + 3 };

	PartyBeaconID_t m_ulBeaconID;
	CSteamID m_steamIDJoiner;
};
 
// Response to ChangeNumOpenSlots call
struct ChangeNumOpenSlotsCallback_t
{
	enum { k_iCallback = k_iSteamPartiesCallbacks + 4 };

	EResult m_eResult;
};

// The list of possible Party beacon locations has changed
struct AvailableBeaconLocationsUpdated_t
{
	enum { k_iCallback = k_iSteamPartiesCallbacks + 5 };
};

// The list of active beacons may have changed
struct ActiveBeaconsUpdated_t
{
	enum { k_iCallback = k_iSteamPartiesCallbacks + 6 };
};


#pragma pack( pop )


#endif // ISTEAMMATCHMAKING
