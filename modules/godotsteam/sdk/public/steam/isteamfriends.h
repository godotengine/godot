//====== Copyright Valve Corporation, All rights reserved. ====================
//
// Purpose: interface to both friends list data and general information about users
//
//=============================================================================

#ifndef ISTEAMFRIENDS_H
#define ISTEAMFRIENDS_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"

//-----------------------------------------------------------------------------
// Purpose: set of relationships to other users
//-----------------------------------------------------------------------------
enum EFriendRelationship
{
	k_EFriendRelationshipNone = 0,
	k_EFriendRelationshipBlocked = 1,			// this doesn't get stored; the user has just done an Ignore on an friendship invite
	k_EFriendRelationshipRequestRecipient = 2,
	k_EFriendRelationshipFriend = 3,
	k_EFriendRelationshipRequestInitiator = 4,
	k_EFriendRelationshipIgnored = 5,			// this is stored; the user has explicit blocked this other user from comments/chat/etc
	k_EFriendRelationshipIgnoredFriend = 6,
	k_EFriendRelationshipSuggested_DEPRECATED = 7,		// was used by the original implementation of the facebook linking feature, but now unused.

	// keep this updated
	k_EFriendRelationshipMax = 8,
};

// maximum length of friend group name (not including terminating nul!)
const int k_cchMaxFriendsGroupName = 64;

// maximum number of groups a single user is allowed
const int k_cFriendsGroupLimit = 100;

// friends group identifier type
typedef int16 FriendsGroupID_t;

// invalid friends group identifier constant
const FriendsGroupID_t k_FriendsGroupID_Invalid = -1;

const int k_cEnumerateFollowersMax = 50;


//-----------------------------------------------------------------------------
// Purpose: list of states a friend can be in
//-----------------------------------------------------------------------------
enum EPersonaState
{
	k_EPersonaStateOffline = 0,			// friend is not currently logged on
	k_EPersonaStateOnline = 1,			// friend is logged on
	k_EPersonaStateBusy = 2,			// user is on, but busy
	k_EPersonaStateAway = 3,			// auto-away feature
	k_EPersonaStateSnooze = 4,			// auto-away for a long time
	k_EPersonaStateLookingToTrade = 5,	// Online, trading
	k_EPersonaStateLookingToPlay = 6,	// Online, wanting to play
	k_EPersonaStateInvisible = 7,		// Online, but appears offline to friends.  This status is never published to clients.
	k_EPersonaStateMax,
};


//-----------------------------------------------------------------------------
// Purpose: flags for enumerating friends list, or quickly checking a the relationship between users
//-----------------------------------------------------------------------------
enum EFriendFlags
{
	k_EFriendFlagNone			= 0x00,
	k_EFriendFlagBlocked		= 0x01,
	k_EFriendFlagFriendshipRequested	= 0x02,
	k_EFriendFlagImmediate		= 0x04,			// "regular" friend
	k_EFriendFlagClanMember		= 0x08,
	k_EFriendFlagOnGameServer	= 0x10,	
	// k_EFriendFlagHasPlayedWith	= 0x20,	// not currently used
	// k_EFriendFlagFriendOfFriend	= 0x40, // not currently used
	k_EFriendFlagRequestingFriendship = 0x80,
	k_EFriendFlagRequestingInfo = 0x100,
	k_EFriendFlagIgnored		= 0x200,
	k_EFriendFlagIgnoredFriend	= 0x400,
	// k_EFriendFlagSuggested		= 0x800,	// not used
	k_EFriendFlagChatMember		= 0x1000,
	k_EFriendFlagAll			= 0xFFFF,
};


// friend game played information
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 
struct FriendGameInfo_t
{
	CGameID m_gameID;
	uint32 m_unGameIP;
	uint16 m_usGamePort;
	uint16 m_usQueryPort;
	CSteamID m_steamIDLobby;
};
#pragma pack( pop )

// maximum number of characters in a user's name. Two flavors; one for UTF-8 and one for UTF-16.
// The UTF-8 version has to be very generous to accomodate characters that get large when encoded
// in UTF-8.
enum
{
	k_cchPersonaNameMax = 128,
	k_cwchPersonaNameMax = 32,
};

//-----------------------------------------------------------------------------
// Purpose: user restriction flags
//-----------------------------------------------------------------------------
enum EUserRestriction
{
	k_nUserRestrictionNone		= 0,	// no known chat/content restriction
	k_nUserRestrictionUnknown	= 1,	// we don't know yet (user offline)
	k_nUserRestrictionAnyChat	= 2,	// user is not allowed to (or can't) send/recv any chat
	k_nUserRestrictionVoiceChat	= 4,	// user is not allowed to (or can't) send/recv voice chat
	k_nUserRestrictionGroupChat	= 8,	// user is not allowed to (or can't) send/recv group chat
	k_nUserRestrictionRating	= 16,	// user is too young according to rating in current region
	k_nUserRestrictionGameInvites	= 32,	// user cannot send or recv game invites (e.g. mobile)
	k_nUserRestrictionTrading	= 64,	// user cannot participate in trading (console, mobile)
};

// size limit on chat room or member metadata
const uint32 k_cubChatMetadataMax = 8192;

// size limits on Rich Presence data
enum { k_cchMaxRichPresenceKeys = 30 };
enum { k_cchMaxRichPresenceKeyLength = 64 };
enum { k_cchMaxRichPresenceValueLength = 256 };

// These values are passed as parameters to the store
enum EOverlayToStoreFlag
{
	k_EOverlayToStoreFlag_None = 0,
	k_EOverlayToStoreFlag_AddToCart = 1,
	k_EOverlayToStoreFlag_AddToCartAndShow = 2,
};


//-----------------------------------------------------------------------------
// Purpose: Tells Steam where to place the browser window inside the overlay
//-----------------------------------------------------------------------------
enum EActivateGameOverlayToWebPageMode
{
	k_EActivateGameOverlayToWebPageMode_Default = 0,		// Browser will open next to all other windows that the user has open in the overlay.
															// The window will remain open, even if the user closes then re-opens the overlay.

	k_EActivateGameOverlayToWebPageMode_Modal = 1			// Browser will be opened in a special overlay configuration which hides all other windows
															// that the user has open in the overlay. When the user closes the overlay, the browser window
															// will also close. When the user closes the browser window, the overlay will automatically close.
};

//-----------------------------------------------------------------------------
// Purpose: See GetProfileItemPropertyString and GetProfileItemPropertyUint
//-----------------------------------------------------------------------------
enum ECommunityProfileItemType
{
	k_ECommunityProfileItemType_AnimatedAvatar		 = 0,
	k_ECommunityProfileItemType_AvatarFrame			 = 1,
	k_ECommunityProfileItemType_ProfileModifier		 = 2,
	k_ECommunityProfileItemType_ProfileBackground	 = 3,
	k_ECommunityProfileItemType_MiniProfileBackground = 4,
};
enum ECommunityProfileItemProperty
{
	k_ECommunityProfileItemProperty_ImageSmall	   = 0, // string
	k_ECommunityProfileItemProperty_ImageLarge	   = 1, // string
	k_ECommunityProfileItemProperty_InternalName   = 2, // string
	k_ECommunityProfileItemProperty_Title		   = 3, // string
	k_ECommunityProfileItemProperty_Description	   = 4, // string
	k_ECommunityProfileItemProperty_AppID		   = 5, // uint32
	k_ECommunityProfileItemProperty_TypeID		   = 6, // uint32
	k_ECommunityProfileItemProperty_Class		   = 7, // uint32
	k_ECommunityProfileItemProperty_MovieWebM	   = 8, // string
	k_ECommunityProfileItemProperty_MovieMP4	   = 9, // string
	k_ECommunityProfileItemProperty_MovieWebMSmall = 10, // string
	k_ECommunityProfileItemProperty_MovieMP4Small  = 11, // string
};

//-----------------------------------------------------------------------------
// Purpose: interface to accessing information about individual users,
//			that can be a friend, in a group, on a game server or in a lobby with the local user
//-----------------------------------------------------------------------------
class ISteamFriends
{
public:
	// returns the local players name - guaranteed to not be NULL.
	// this is the same name as on the users community profile page
	// this is stored in UTF-8 format
	// like all the other interface functions that return a char *, it's important that this pointer is not saved
	// off; it will eventually be free'd or re-allocated
	virtual const char *GetPersonaName() = 0;

	// Sets the player name, stores it on the server and publishes the changes to all friends who are online.
	// Changes take place locally immediately, and a PersonaStateChange_t is posted, presuming success.
	//
	// The final results are available through the return value SteamAPICall_t, using SetPersonaNameResponse_t.
	//
	// If the name change fails to happen on the server, then an additional global PersonaStateChange_t will be posted
	// to change the name back, in addition to the SetPersonaNameResponse_t callback.
	STEAM_CALL_RESULT( SetPersonaNameResponse_t )
	virtual SteamAPICall_t SetPersonaName( const char *pchPersonaName ) = 0;

	// gets the status of the current user
	virtual EPersonaState GetPersonaState() = 0;

	// friend iteration
	// takes a set of k_EFriendFlags, and returns the number of users the client knows about who meet that criteria
	// then GetFriendByIndex() can then be used to return the id's of each of those users
	virtual int GetFriendCount( int iFriendFlags ) = 0;

	// returns the steamID of a user
	// iFriend is a index of range [0, GetFriendCount())
	// iFriendsFlags must be the same value as used in GetFriendCount()
	// the returned CSteamID can then be used by all the functions below to access details about the user
	virtual CSteamID GetFriendByIndex( int iFriend, int iFriendFlags ) = 0;

	// returns a relationship to a user
	virtual EFriendRelationship GetFriendRelationship( CSteamID steamIDFriend ) = 0;

	// returns the current status of the specified user
	// this will only be known by the local user if steamIDFriend is in their friends list; on the same game server; in a chat room or lobby; or in a small group with the local user
	virtual EPersonaState GetFriendPersonaState( CSteamID steamIDFriend ) = 0;

	// returns the name another user - guaranteed to not be NULL.
	// same rules as GetFriendPersonaState() apply as to whether or not the user knowns the name of the other user
	// note that on first joining a lobby, chat room or game server the local user will not known the name of the other users automatically; that information will arrive asyncronously
	// 
	virtual const char *GetFriendPersonaName( CSteamID steamIDFriend ) = 0;

	// returns true if the friend is actually in a game, and fills in pFriendGameInfo with an extra details 
	virtual bool GetFriendGamePlayed( CSteamID steamIDFriend, STEAM_OUT_STRUCT() FriendGameInfo_t *pFriendGameInfo ) = 0;
	// accesses old friends names - returns an empty string when their are no more items in the history
	virtual const char *GetFriendPersonaNameHistory( CSteamID steamIDFriend, int iPersonaName ) = 0;
	// friends steam level
	virtual int GetFriendSteamLevel( CSteamID steamIDFriend ) = 0;

	// Returns nickname the current user has set for the specified player. Returns NULL if the no nickname has been set for that player.
	// DEPRECATED: GetPersonaName follows the Steam nickname preferences, so apps shouldn't need to care about nicknames explicitly.
	virtual const char *GetPlayerNickname( CSteamID steamIDPlayer ) = 0;

	// friend grouping (tag) apis
	// returns the number of friends groups
	virtual int GetFriendsGroupCount() = 0;
	// returns the friends group ID for the given index (invalid indices return k_FriendsGroupID_Invalid)
	virtual FriendsGroupID_t GetFriendsGroupIDByIndex( int iFG ) = 0;
	// returns the name for the given friends group (NULL in the case of invalid friends group IDs)
	virtual const char *GetFriendsGroupName( FriendsGroupID_t friendsGroupID ) = 0;
	// returns the number of members in a given friends group
	virtual int GetFriendsGroupMembersCount( FriendsGroupID_t friendsGroupID ) = 0;
	// gets up to nMembersCount members of the given friends group, if fewer exist than requested those positions' SteamIDs will be invalid
	virtual void GetFriendsGroupMembersList( FriendsGroupID_t friendsGroupID, STEAM_OUT_ARRAY_CALL(nMembersCount, GetFriendsGroupMembersCount, friendsGroupID ) CSteamID *pOutSteamIDMembers, int nMembersCount ) = 0;

	// returns true if the specified user meets any of the criteria specified in iFriendFlags
	// iFriendFlags can be the union (binary or, |) of one or more k_EFriendFlags values
	virtual bool HasFriend( CSteamID steamIDFriend, int iFriendFlags ) = 0;

	// clan (group) iteration and access functions
	virtual int GetClanCount() = 0;
	virtual CSteamID GetClanByIndex( int iClan ) = 0;
	virtual const char *GetClanName( CSteamID steamIDClan ) = 0;
	virtual const char *GetClanTag( CSteamID steamIDClan ) = 0;
	// returns the most recent information we have about what's happening in a clan
	virtual bool GetClanActivityCounts( CSteamID steamIDClan, int *pnOnline, int *pnInGame, int *pnChatting ) = 0;

	// for clans a user is a member of, they will have reasonably up-to-date information, but for others you'll have to download the info to have the latest
	STEAM_CALL_RESULT( DownloadClanActivityCountsResult_t )
	virtual SteamAPICall_t DownloadClanActivityCounts( STEAM_ARRAY_COUNT(cClansToRequest) CSteamID *psteamIDClans, int cClansToRequest ) = 0;

	// iterators for getting users in a chat room, lobby, game server or clan
	// note that large clans that cannot be iterated by the local user
	// note that the current user must be in a lobby to retrieve CSteamIDs of other users in that lobby
	// steamIDSource can be the steamID of a group, game server, lobby or chat room
	virtual int GetFriendCountFromSource( CSteamID steamIDSource ) = 0;
	virtual CSteamID GetFriendFromSourceByIndex( CSteamID steamIDSource, int iFriend ) = 0;

	// returns true if the local user can see that steamIDUser is a member or in steamIDSource
	virtual bool IsUserInSource( CSteamID steamIDUser, CSteamID steamIDSource ) = 0;

	// User is in a game pressing the talk button (will suppress the microphone for all voice comms from the Steam friends UI)
	virtual void SetInGameVoiceSpeaking( CSteamID steamIDUser, bool bSpeaking ) = 0;

	// activates the game overlay, with an optional dialog to open 
	// valid options include "Friends", "Community", "Players", "Settings", "OfficialGameGroup", "Stats", "Achievements",
	// "chatroomgroup/nnnn"
	virtual void ActivateGameOverlay( const char *pchDialog ) = 0;

	// activates game overlay to a specific place
	// valid options are
	//		"steamid" - opens the overlay web browser to the specified user or groups profile
	//		"chat" - opens a chat window to the specified user, or joins the group chat 
	//		"jointrade" - opens a window to a Steam Trading session that was started with the ISteamEconomy/StartTrade Web API
	//		"stats" - opens the overlay web browser to the specified user's stats
	//		"achievements" - opens the overlay web browser to the specified user's achievements
	//		"friendadd" - opens the overlay in minimal mode prompting the user to add the target user as a friend
	//		"friendremove" - opens the overlay in minimal mode prompting the user to remove the target friend
	//		"friendrequestaccept" - opens the overlay in minimal mode prompting the user to accept an incoming friend invite
	//		"friendrequestignore" - opens the overlay in minimal mode prompting the user to ignore an incoming friend invite
	virtual void ActivateGameOverlayToUser( const char *pchDialog, CSteamID steamID ) = 0;

	// activates game overlay web browser directly to the specified URL
	// full address with protocol type is required, e.g. http://www.steamgames.com/
	virtual void ActivateGameOverlayToWebPage( const char *pchURL, EActivateGameOverlayToWebPageMode eMode = k_EActivateGameOverlayToWebPageMode_Default ) = 0;

	// activates game overlay to store page for app
	virtual void ActivateGameOverlayToStore( AppId_t nAppID, EOverlayToStoreFlag eFlag ) = 0;

	// Mark a target user as 'played with'. This is a client-side only feature that requires that the calling user is 
	// in game 
	virtual void SetPlayedWith( CSteamID steamIDUserPlayedWith ) = 0;

	// activates game overlay to open the invite dialog. Invitations will be sent for the provided lobby.
	virtual void ActivateGameOverlayInviteDialog( CSteamID steamIDLobby ) = 0;

	// gets the small (32x32) avatar of the current user, which is a handle to be used in IClientUtils::GetImageRGBA(), or 0 if none set
	virtual int GetSmallFriendAvatar( CSteamID steamIDFriend ) = 0;

	// gets the medium (64x64) avatar of the current user, which is a handle to be used in IClientUtils::GetImageRGBA(), or 0 if none set
	virtual int GetMediumFriendAvatar( CSteamID steamIDFriend ) = 0;

	// gets the large (184x184) avatar of the current user, which is a handle to be used in IClientUtils::GetImageRGBA(), or 0 if none set
	// returns -1 if this image has yet to be loaded, in this case wait for a AvatarImageLoaded_t callback and then call this again
	virtual int GetLargeFriendAvatar( CSteamID steamIDFriend ) = 0;

	// requests information about a user - persona name & avatar
	// if bRequireNameOnly is set, then the avatar of a user isn't downloaded 
	// - it's a lot slower to download avatars and churns the local cache, so if you don't need avatars, don't request them
	// if returns true, it means that data is being requested, and a PersonaStateChanged_t callback will be posted when it's retrieved
	// if returns false, it means that we already have all the details about that user, and functions can be called immediately
	virtual bool RequestUserInformation( CSteamID steamIDUser, bool bRequireNameOnly ) = 0;

	// requests information about a clan officer list
	// when complete, data is returned in ClanOfficerListResponse_t call result
	// this makes available the calls below
	// you can only ask about clans that a user is a member of
	// note that this won't download avatars automatically; if you get an officer,
	// and no avatar image is available, call RequestUserInformation( steamID, false ) to download the avatar
	STEAM_CALL_RESULT( ClanOfficerListResponse_t )
	virtual SteamAPICall_t RequestClanOfficerList( CSteamID steamIDClan ) = 0;

	// iteration of clan officers - can only be done when a RequestClanOfficerList() call has completed
	
	// returns the steamID of the clan owner
	virtual CSteamID GetClanOwner( CSteamID steamIDClan ) = 0;
	// returns the number of officers in a clan (including the owner)
	virtual int GetClanOfficerCount( CSteamID steamIDClan ) = 0;
	// returns the steamID of a clan officer, by index, of range [0,GetClanOfficerCount)
	virtual CSteamID GetClanOfficerByIndex( CSteamID steamIDClan, int iOfficer ) = 0;
	// if current user is chat restricted, he can't send or receive any text/voice chat messages.
	// the user can't see custom avatars. But the user can be online and send/recv game invites.
	// a chat restricted user can't add friends or join any groups.
	virtual uint32 GetUserRestrictions() = 0;

	// Rich Presence data is automatically shared between friends who are in the same game
	// Each user has a set of Key/Value pairs
	// Note the following limits: k_cchMaxRichPresenceKeys, k_cchMaxRichPresenceKeyLength, k_cchMaxRichPresenceValueLength
	// There are five magic keys:
	//		"status"  - a UTF-8 string that will show up in the 'view game info' dialog in the Steam friends list
	//		"connect" - a UTF-8 string that contains the command-line for how a friend can connect to a game
	//		"steam_display"				- Names a rich presence localization token that will be displayed in the viewing user's selected language
	//									  in the Steam client UI. For more info: https://partner.steamgames.com/doc/api/ISteamFriends#richpresencelocalization
	//		"steam_player_group"		- When set, indicates to the Steam client that the player is a member of a particular group. Players in the same group
	//									  may be organized together in various places in the Steam UI.
	//		"steam_player_group_size"	- When set, indicates the total number of players in the steam_player_group. The Steam client may use this number to
	//									  display additional information about a group when all of the members are not part of a user's friends list.
	// GetFriendRichPresence() returns an empty string "" if no value is set
	// SetRichPresence() to a NULL or an empty string deletes the key
	// You can iterate the current set of keys for a friend with GetFriendRichPresenceKeyCount()
	// and GetFriendRichPresenceKeyByIndex() (typically only used for debugging)
	virtual bool SetRichPresence( const char *pchKey, const char *pchValue ) = 0;
	virtual void ClearRichPresence() = 0;
	virtual const char *GetFriendRichPresence( CSteamID steamIDFriend, const char *pchKey ) = 0;
	virtual int GetFriendRichPresenceKeyCount( CSteamID steamIDFriend ) = 0;
	virtual const char *GetFriendRichPresenceKeyByIndex( CSteamID steamIDFriend, int iKey ) = 0;
	// Requests rich presence for a specific user.
	virtual void RequestFriendRichPresence( CSteamID steamIDFriend ) = 0;

	// Rich invite support.
	// If the target accepts the invite, a GameRichPresenceJoinRequested_t callback is posted containing the connect string.
	// (Or you can configure your game so that it is passed on the command line instead.  This is a deprecated path; ask us if you really need this.)
	virtual bool InviteUserToGame( CSteamID steamIDFriend, const char *pchConnectString ) = 0;

	// recently-played-with friends iteration
	// this iterates the entire list of users recently played with, across games
	// GetFriendCoplayTime() returns as a unix time
	virtual int GetCoplayFriendCount() = 0;
	virtual CSteamID GetCoplayFriend( int iCoplayFriend ) = 0;
	virtual int GetFriendCoplayTime( CSteamID steamIDFriend ) = 0;
	virtual AppId_t GetFriendCoplayGame( CSteamID steamIDFriend ) = 0;

	// chat interface for games
	// this allows in-game access to group (clan) chats from in the game
	// the behavior is somewhat sophisticated, because the user may or may not be already in the group chat from outside the game or in the overlay
	// use ActivateGameOverlayToUser( "chat", steamIDClan ) to open the in-game overlay version of the chat
	STEAM_CALL_RESULT( JoinClanChatRoomCompletionResult_t )
	virtual SteamAPICall_t JoinClanChatRoom( CSteamID steamIDClan ) = 0;
	virtual bool LeaveClanChatRoom( CSteamID steamIDClan ) = 0;
	virtual int GetClanChatMemberCount( CSteamID steamIDClan ) = 0;
	virtual CSteamID GetChatMemberByIndex( CSteamID steamIDClan, int iUser ) = 0;
	virtual bool SendClanChatMessage( CSteamID steamIDClanChat, const char *pchText ) = 0;
	virtual int GetClanChatMessage( CSteamID steamIDClanChat, int iMessage, void *prgchText, int cchTextMax, EChatEntryType *peChatEntryType, STEAM_OUT_STRUCT() CSteamID *psteamidChatter ) = 0;
	virtual bool IsClanChatAdmin( CSteamID steamIDClanChat, CSteamID steamIDUser ) = 0;

	// interact with the Steam (game overlay / desktop)
	virtual bool IsClanChatWindowOpenInSteam( CSteamID steamIDClanChat ) = 0;
	virtual bool OpenClanChatWindowInSteam( CSteamID steamIDClanChat ) = 0;
	virtual bool CloseClanChatWindowInSteam( CSteamID steamIDClanChat ) = 0;

	// peer-to-peer chat interception
	// this is so you can show P2P chats inline in the game
	virtual bool SetListenForFriendsMessages( bool bInterceptEnabled ) = 0;
	virtual bool ReplyToFriendMessage( CSteamID steamIDFriend, const char *pchMsgToSend ) = 0;
	virtual int GetFriendMessage( CSteamID steamIDFriend, int iMessageID, void *pvData, int cubData, EChatEntryType *peChatEntryType ) = 0;

	// following apis
	STEAM_CALL_RESULT( FriendsGetFollowerCount_t )
	virtual SteamAPICall_t GetFollowerCount( CSteamID steamID ) = 0;
	STEAM_CALL_RESULT( FriendsIsFollowing_t )
	virtual SteamAPICall_t IsFollowing( CSteamID steamID ) = 0;
	STEAM_CALL_RESULT( FriendsEnumerateFollowingList_t )
	virtual SteamAPICall_t EnumerateFollowingList( uint32 unStartIndex ) = 0;

	virtual bool IsClanPublic( CSteamID steamIDClan ) = 0;
	virtual bool IsClanOfficialGameGroup( CSteamID steamIDClan ) = 0;

	/// Return the number of chats (friends or chat rooms) with unread messages.
	/// A "priority" message is one that would generate some sort of toast or
	/// notification, and depends on user settings.
	///
	/// You can register for UnreadChatMessagesChanged_t callbacks to know when this
	/// has potentially changed.
	virtual int GetNumChatsWithUnreadPriorityMessages() = 0;

	// activates game overlay to open the remote play together invite dialog. Invitations will be sent for remote play together
	virtual void ActivateGameOverlayRemotePlayTogetherInviteDialog( CSteamID steamIDLobby ) = 0;

	// Call this before calling ActivateGameOverlayToWebPage() to have the Steam Overlay Browser block navigations
	// to your specified protocol (scheme) uris and instead dispatch a OverlayBrowserProtocolNavigation_t callback to your game.
	// ActivateGameOverlayToWebPage() must have been called with k_EActivateGameOverlayToWebPageMode_Modal
	virtual bool RegisterProtocolInOverlayBrowser( const char *pchProtocol ) = 0;

	// Activates the game overlay to open an invite dialog that will send the provided Rich Presence connect string to selected friends
	virtual void ActivateGameOverlayInviteDialogConnectString( const char *pchConnectString ) = 0;

	// Steam Community items equipped by a user on their profile
	// You can register for EquippedProfileItemsChanged_t to know when a friend has changed their equipped profile items
	STEAM_CALL_RESULT( EquippedProfileItems_t )
	virtual SteamAPICall_t RequestEquippedProfileItems( CSteamID steamID ) = 0;
	virtual bool BHasEquippedProfileItem( CSteamID steamID, ECommunityProfileItemType itemType ) = 0;
	virtual const char *GetProfileItemPropertyString( CSteamID steamID, ECommunityProfileItemType itemType, ECommunityProfileItemProperty prop ) = 0;
	virtual uint32 GetProfileItemPropertyUint( CSteamID steamID, ECommunityProfileItemType itemType, ECommunityProfileItemProperty prop ) = 0;
};

#define STEAMFRIENDS_INTERFACE_VERSION "SteamFriends017"

// Global interface accessor
inline ISteamFriends *SteamFriends();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamFriends *, SteamFriends, STEAMFRIENDS_INTERFACE_VERSION );

// callbacks
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 

//-----------------------------------------------------------------------------
// Purpose: called when a friends' status changes
//-----------------------------------------------------------------------------
struct PersonaStateChange_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 4 };
	
	uint64 m_ulSteamID;		// steamID of the friend who changed
	int m_nChangeFlags;		// what's changed
};


// used in PersonaStateChange_t::m_nChangeFlags to describe what's changed about a user
// these flags describe what the client has learned has changed recently, so on startup you'll see a name, avatar & relationship change for every friend
enum EPersonaChange
{
	k_EPersonaChangeName		= 0x0001,
	k_EPersonaChangeStatus		= 0x0002,
	k_EPersonaChangeComeOnline	= 0x0004,
	k_EPersonaChangeGoneOffline	= 0x0008,
	k_EPersonaChangeGamePlayed	= 0x0010,
	k_EPersonaChangeGameServer	= 0x0020,
	k_EPersonaChangeAvatar		= 0x0040,
	k_EPersonaChangeJoinedSource= 0x0080,
	k_EPersonaChangeLeftSource	= 0x0100,
	k_EPersonaChangeRelationshipChanged = 0x0200,
	k_EPersonaChangeNameFirstSet = 0x0400,
	k_EPersonaChangeBroadcast = 0x0800,
	k_EPersonaChangeNickname =	0x1000,
	k_EPersonaChangeSteamLevel = 0x2000,
	k_EPersonaChangeRichPresence = 0x4000,
};


//-----------------------------------------------------------------------------
// Purpose: posted when game overlay activates or deactivates
//			the game can use this to be pause or resume single player games
//-----------------------------------------------------------------------------
struct GameOverlayActivated_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 31 };
	uint8 m_bActive;	// true if it's just been activated, false otherwise
};


//-----------------------------------------------------------------------------
// Purpose: called when the user tries to join a different game server from their friends list
//			game client should attempt to connect to specified server when this is received
//-----------------------------------------------------------------------------
struct GameServerChangeRequested_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 32 };
	char m_rgchServer[64];		// server address ("127.0.0.1:27015", "tf2.valvesoftware.com")
	char m_rgchPassword[64];	// server password, if any
};


//-----------------------------------------------------------------------------
// Purpose: called when the user tries to join a lobby from their friends list
//			game client should attempt to connect to specified lobby when this is received
//-----------------------------------------------------------------------------
struct GameLobbyJoinRequested_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 33 };
	CSteamID m_steamIDLobby;

	// The friend they did the join via (will be invalid if not directly via a friend)
	//
	// On PS3, the friend will be invalid if this was triggered by a PSN invite via the XMB, but
	// the account type will be console user so you can tell at least that this was from a PSN friend
	// rather than a Steam friend.
	CSteamID m_steamIDFriend;		
};


//-----------------------------------------------------------------------------
// Purpose: called when an avatar is loaded in from a previous GetLargeFriendAvatar() call
//			if the image wasn't already available
//-----------------------------------------------------------------------------
struct AvatarImageLoaded_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 34 };
	CSteamID m_steamID; // steamid the avatar has been loaded for
	int m_iImage; // the image index of the now loaded image
	int m_iWide; // width of the loaded image
	int m_iTall; // height of the loaded image
};


//-----------------------------------------------------------------------------
// Purpose: marks the return of a request officer list call
//-----------------------------------------------------------------------------
struct ClanOfficerListResponse_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 35 };
	CSteamID m_steamIDClan;
	int m_cOfficers;
	uint8 m_bSuccess;
};


//-----------------------------------------------------------------------------
// Purpose: callback indicating updated data about friends rich presence information
//-----------------------------------------------------------------------------
struct FriendRichPresenceUpdate_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 36 };
	CSteamID m_steamIDFriend;	// friend who's rich presence has changed
	AppId_t m_nAppID;			// the appID of the game (should always be the current game)
};


//-----------------------------------------------------------------------------
// Purpose: called when the user tries to join a game from their friends list
//			rich presence will have been set with the "connect" key which is set here
//-----------------------------------------------------------------------------
struct GameRichPresenceJoinRequested_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 37 };
	CSteamID m_steamIDFriend;		// the friend they did the join via (will be invalid if not directly via a friend)
	char m_rgchConnect[k_cchMaxRichPresenceValueLength];
};


//-----------------------------------------------------------------------------
// Purpose: a chat message has been received for a clan chat the game has joined
//-----------------------------------------------------------------------------
struct GameConnectedClanChatMsg_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 38 };
	CSteamID m_steamIDClanChat;
	CSteamID m_steamIDUser;
	int m_iMessageID;
};


//-----------------------------------------------------------------------------
// Purpose: a user has joined a clan chat
//-----------------------------------------------------------------------------
struct GameConnectedChatJoin_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 39 };
	CSteamID m_steamIDClanChat;
	CSteamID m_steamIDUser;
};


//-----------------------------------------------------------------------------
// Purpose: a user has left the chat we're in
//-----------------------------------------------------------------------------
struct GameConnectedChatLeave_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 40 };
	CSteamID m_steamIDClanChat;
	CSteamID m_steamIDUser;
	bool m_bKicked;		// true if admin kicked
	bool m_bDropped;	// true if Steam connection dropped
};


//-----------------------------------------------------------------------------
// Purpose: a DownloadClanActivityCounts() call has finished
//-----------------------------------------------------------------------------
struct DownloadClanActivityCountsResult_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 41 };
	bool m_bSuccess;
};


//-----------------------------------------------------------------------------
// Purpose: a JoinClanChatRoom() call has finished
//-----------------------------------------------------------------------------
struct JoinClanChatRoomCompletionResult_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 42 };
	CSteamID m_steamIDClanChat;
	EChatRoomEnterResponse m_eChatRoomEnterResponse;
};

//-----------------------------------------------------------------------------
// Purpose: a chat message has been received from a user
//-----------------------------------------------------------------------------
struct GameConnectedFriendChatMsg_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 43 };
	CSteamID m_steamIDUser;
	int m_iMessageID;
};


struct FriendsGetFollowerCount_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 44 };
	EResult m_eResult;
	CSteamID m_steamID;
	int m_nCount;
};


struct FriendsIsFollowing_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 45 };
	EResult m_eResult;
	CSteamID m_steamID;
	bool m_bIsFollowing;
};


struct FriendsEnumerateFollowingList_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 46 };
	EResult m_eResult;
	CSteamID m_rgSteamID[ k_cEnumerateFollowersMax ];
	int32 m_nResultsReturned;
	int32 m_nTotalResultCount;
};

//-----------------------------------------------------------------------------
// Purpose: reports the result of an attempt to change the user's persona name
//-----------------------------------------------------------------------------
struct SetPersonaNameResponse_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 47 };

	bool m_bSuccess; // true if name change succeeded completely.
	bool m_bLocalSuccess; // true if name change was retained locally.  (We might not have been able to communicate with Steam)
	EResult m_result; // detailed result code
};

//-----------------------------------------------------------------------------
// Purpose: Invoked when the status of unread messages changes
//-----------------------------------------------------------------------------
struct UnreadChatMessagesChanged_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 48 };
};


//-----------------------------------------------------------------------------
// Purpose: Dispatched when an overlay browser instance is navigated to a protocol/scheme registered by RegisterProtocolInOverlayBrowser()
//-----------------------------------------------------------------------------
struct OverlayBrowserProtocolNavigation_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 49 };
	char rgchURI[ 1024 ];
};

//-----------------------------------------------------------------------------
// Purpose: A user's equipped profile items have changed
//-----------------------------------------------------------------------------
struct EquippedProfileItemsChanged_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 50 };
	CSteamID m_steamID;
};

//-----------------------------------------------------------------------------
// Purpose: 
//-----------------------------------------------------------------------------
struct EquippedProfileItems_t
{
	enum { k_iCallback = k_iSteamFriendsCallbacks + 51 };
	EResult m_eResult;
	CSteamID m_steamID;
	bool m_bHasAnimatedAvatar;
	bool m_bHasAvatarFrame;
	bool m_bHasProfileModifier;
	bool m_bHasProfileBackground;
	bool m_bHasMiniProfileBackground;
};

#pragma pack( pop )

#endif // ISTEAMFRIENDS_H
