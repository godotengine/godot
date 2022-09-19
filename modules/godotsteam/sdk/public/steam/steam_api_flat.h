//====== Copyright Valve Corporation, All rights reserved. ====================
//
// Purpose: Header for "flat" SteamAPI. Use this for binding to other languages.
// This file is auto-generated, do not edit it.
//
//=============================================================================

#ifndef STEAMAPIFLAT_H
#define STEAMAPIFLAT_H

#include "steam/steam_api.h"
#include "steam/isteamgameserver.h"
#include "steam/isteamgameserverstats.h"

typedef uint64 uint64_steamid; // Used when passing or returning CSteamID
typedef uint64 uint64_gameid; // Used when passing or return CGameID



// ISteamClient
S_API HSteamPipe SteamAPI_ISteamClient_CreateSteamPipe( ISteamClient* self );
S_API bool SteamAPI_ISteamClient_BReleaseSteamPipe( ISteamClient* self, HSteamPipe hSteamPipe );
S_API HSteamUser SteamAPI_ISteamClient_ConnectToGlobalUser( ISteamClient* self, HSteamPipe hSteamPipe );
S_API HSteamUser SteamAPI_ISteamClient_CreateLocalUser( ISteamClient* self, HSteamPipe * phSteamPipe, EAccountType eAccountType );
S_API void SteamAPI_ISteamClient_ReleaseUser( ISteamClient* self, HSteamPipe hSteamPipe, HSteamUser hUser );
S_API ISteamUser * SteamAPI_ISteamClient_GetISteamUser( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamGameServer * SteamAPI_ISteamClient_GetISteamGameServer( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API void SteamAPI_ISteamClient_SetLocalIPBinding( ISteamClient* self, const SteamIPAddress_t & unIP, uint16 usPort );
S_API ISteamFriends * SteamAPI_ISteamClient_GetISteamFriends( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamUtils * SteamAPI_ISteamClient_GetISteamUtils( ISteamClient* self, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamMatchmaking * SteamAPI_ISteamClient_GetISteamMatchmaking( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamMatchmakingServers * SteamAPI_ISteamClient_GetISteamMatchmakingServers( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API void * SteamAPI_ISteamClient_GetISteamGenericInterface( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamUserStats * SteamAPI_ISteamClient_GetISteamUserStats( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamGameServerStats * SteamAPI_ISteamClient_GetISteamGameServerStats( ISteamClient* self, HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamApps * SteamAPI_ISteamClient_GetISteamApps( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamNetworking * SteamAPI_ISteamClient_GetISteamNetworking( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamRemoteStorage * SteamAPI_ISteamClient_GetISteamRemoteStorage( ISteamClient* self, HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamScreenshots * SteamAPI_ISteamClient_GetISteamScreenshots( ISteamClient* self, HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamGameSearch * SteamAPI_ISteamClient_GetISteamGameSearch( ISteamClient* self, HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API uint32 SteamAPI_ISteamClient_GetIPCCallCount( ISteamClient* self );
S_API void SteamAPI_ISteamClient_SetWarningMessageHook( ISteamClient* self, SteamAPIWarningMessageHook_t pFunction );
S_API bool SteamAPI_ISteamClient_BShutdownIfAllPipesClosed( ISteamClient* self );
S_API ISteamHTTP * SteamAPI_ISteamClient_GetISteamHTTP( ISteamClient* self, HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamController * SteamAPI_ISteamClient_GetISteamController( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamUGC * SteamAPI_ISteamClient_GetISteamUGC( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamAppList * SteamAPI_ISteamClient_GetISteamAppList( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamMusic * SteamAPI_ISteamClient_GetISteamMusic( ISteamClient* self, HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamMusicRemote * SteamAPI_ISteamClient_GetISteamMusicRemote( ISteamClient* self, HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamHTMLSurface * SteamAPI_ISteamClient_GetISteamHTMLSurface( ISteamClient* self, HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamInventory * SteamAPI_ISteamClient_GetISteamInventory( ISteamClient* self, HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamVideo * SteamAPI_ISteamClient_GetISteamVideo( ISteamClient* self, HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamParentalSettings * SteamAPI_ISteamClient_GetISteamParentalSettings( ISteamClient* self, HSteamUser hSteamuser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamInput * SteamAPI_ISteamClient_GetISteamInput( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamParties * SteamAPI_ISteamClient_GetISteamParties( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );
S_API ISteamRemotePlay * SteamAPI_ISteamClient_GetISteamRemotePlay( ISteamClient* self, HSteamUser hSteamUser, HSteamPipe hSteamPipe, const char * pchVersion );

// ISteamUser

// A versioned accessor is exported by the library
S_API ISteamUser *SteamAPI_SteamUser_v021();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamUser(), but using this ensures that you are using a matching library.
inline ISteamUser *SteamAPI_SteamUser() { return SteamAPI_SteamUser_v021(); }
S_API HSteamUser SteamAPI_ISteamUser_GetHSteamUser( ISteamUser* self );
S_API bool SteamAPI_ISteamUser_BLoggedOn( ISteamUser* self );
S_API uint64_steamid SteamAPI_ISteamUser_GetSteamID( ISteamUser* self );
S_API int SteamAPI_ISteamUser_InitiateGameConnection_DEPRECATED( ISteamUser* self, void * pAuthBlob, int cbMaxAuthBlob, uint64_steamid steamIDGameServer, uint32 unIPServer, uint16 usPortServer, bool bSecure );
S_API void SteamAPI_ISteamUser_TerminateGameConnection_DEPRECATED( ISteamUser* self, uint32 unIPServer, uint16 usPortServer );
S_API void SteamAPI_ISteamUser_TrackAppUsageEvent( ISteamUser* self, uint64_gameid gameID, int eAppUsageEvent, const char * pchExtraInfo );
S_API bool SteamAPI_ISteamUser_GetUserDataFolder( ISteamUser* self, char * pchBuffer, int cubBuffer );
S_API void SteamAPI_ISteamUser_StartVoiceRecording( ISteamUser* self );
S_API void SteamAPI_ISteamUser_StopVoiceRecording( ISteamUser* self );
S_API EVoiceResult SteamAPI_ISteamUser_GetAvailableVoice( ISteamUser* self, uint32 * pcbCompressed, uint32 * pcbUncompressed_Deprecated, uint32 nUncompressedVoiceDesiredSampleRate_Deprecated );
S_API EVoiceResult SteamAPI_ISteamUser_GetVoice( ISteamUser* self, bool bWantCompressed, void * pDestBuffer, uint32 cbDestBufferSize, uint32 * nBytesWritten, bool bWantUncompressed_Deprecated, void * pUncompressedDestBuffer_Deprecated, uint32 cbUncompressedDestBufferSize_Deprecated, uint32 * nUncompressBytesWritten_Deprecated, uint32 nUncompressedVoiceDesiredSampleRate_Deprecated );
S_API EVoiceResult SteamAPI_ISteamUser_DecompressVoice( ISteamUser* self, const void * pCompressed, uint32 cbCompressed, void * pDestBuffer, uint32 cbDestBufferSize, uint32 * nBytesWritten, uint32 nDesiredSampleRate );
S_API uint32 SteamAPI_ISteamUser_GetVoiceOptimalSampleRate( ISteamUser* self );
S_API HAuthTicket SteamAPI_ISteamUser_GetAuthSessionTicket( ISteamUser* self, void * pTicket, int cbMaxTicket, uint32 * pcbTicket );
S_API EBeginAuthSessionResult SteamAPI_ISteamUser_BeginAuthSession( ISteamUser* self, const void * pAuthTicket, int cbAuthTicket, uint64_steamid steamID );
S_API void SteamAPI_ISteamUser_EndAuthSession( ISteamUser* self, uint64_steamid steamID );
S_API void SteamAPI_ISteamUser_CancelAuthTicket( ISteamUser* self, HAuthTicket hAuthTicket );
S_API EUserHasLicenseForAppResult SteamAPI_ISteamUser_UserHasLicenseForApp( ISteamUser* self, uint64_steamid steamID, AppId_t appID );
S_API bool SteamAPI_ISteamUser_BIsBehindNAT( ISteamUser* self );
S_API void SteamAPI_ISteamUser_AdvertiseGame( ISteamUser* self, uint64_steamid steamIDGameServer, uint32 unIPServer, uint16 usPortServer );
S_API SteamAPICall_t SteamAPI_ISteamUser_RequestEncryptedAppTicket( ISteamUser* self, void * pDataToInclude, int cbDataToInclude );
S_API bool SteamAPI_ISteamUser_GetEncryptedAppTicket( ISteamUser* self, void * pTicket, int cbMaxTicket, uint32 * pcbTicket );
S_API int SteamAPI_ISteamUser_GetGameBadgeLevel( ISteamUser* self, int nSeries, bool bFoil );
S_API int SteamAPI_ISteamUser_GetPlayerSteamLevel( ISteamUser* self );
S_API SteamAPICall_t SteamAPI_ISteamUser_RequestStoreAuthURL( ISteamUser* self, const char * pchRedirectURL );
S_API bool SteamAPI_ISteamUser_BIsPhoneVerified( ISteamUser* self );
S_API bool SteamAPI_ISteamUser_BIsTwoFactorEnabled( ISteamUser* self );
S_API bool SteamAPI_ISteamUser_BIsPhoneIdentifying( ISteamUser* self );
S_API bool SteamAPI_ISteamUser_BIsPhoneRequiringVerification( ISteamUser* self );
S_API SteamAPICall_t SteamAPI_ISteamUser_GetMarketEligibility( ISteamUser* self );
S_API SteamAPICall_t SteamAPI_ISteamUser_GetDurationControl( ISteamUser* self );
S_API bool SteamAPI_ISteamUser_BSetDurationControlOnlineState( ISteamUser* self, EDurationControlOnlineState eNewState );

// ISteamFriends

// A versioned accessor is exported by the library
S_API ISteamFriends *SteamAPI_SteamFriends_v017();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamFriends(), but using this ensures that you are using a matching library.
inline ISteamFriends *SteamAPI_SteamFriends() { return SteamAPI_SteamFriends_v017(); }
S_API const char * SteamAPI_ISteamFriends_GetPersonaName( ISteamFriends* self );
S_API SteamAPICall_t SteamAPI_ISteamFriends_SetPersonaName( ISteamFriends* self, const char * pchPersonaName );
S_API EPersonaState SteamAPI_ISteamFriends_GetPersonaState( ISteamFriends* self );
S_API int SteamAPI_ISteamFriends_GetFriendCount( ISteamFriends* self, int iFriendFlags );
S_API uint64_steamid SteamAPI_ISteamFriends_GetFriendByIndex( ISteamFriends* self, int iFriend, int iFriendFlags );
S_API EFriendRelationship SteamAPI_ISteamFriends_GetFriendRelationship( ISteamFriends* self, uint64_steamid steamIDFriend );
S_API EPersonaState SteamAPI_ISteamFriends_GetFriendPersonaState( ISteamFriends* self, uint64_steamid steamIDFriend );
S_API const char * SteamAPI_ISteamFriends_GetFriendPersonaName( ISteamFriends* self, uint64_steamid steamIDFriend );
S_API bool SteamAPI_ISteamFriends_GetFriendGamePlayed( ISteamFriends* self, uint64_steamid steamIDFriend, FriendGameInfo_t * pFriendGameInfo );
S_API const char * SteamAPI_ISteamFriends_GetFriendPersonaNameHistory( ISteamFriends* self, uint64_steamid steamIDFriend, int iPersonaName );
S_API int SteamAPI_ISteamFriends_GetFriendSteamLevel( ISteamFriends* self, uint64_steamid steamIDFriend );
S_API const char * SteamAPI_ISteamFriends_GetPlayerNickname( ISteamFriends* self, uint64_steamid steamIDPlayer );
S_API int SteamAPI_ISteamFriends_GetFriendsGroupCount( ISteamFriends* self );
S_API FriendsGroupID_t SteamAPI_ISteamFriends_GetFriendsGroupIDByIndex( ISteamFriends* self, int iFG );
S_API const char * SteamAPI_ISteamFriends_GetFriendsGroupName( ISteamFriends* self, FriendsGroupID_t friendsGroupID );
S_API int SteamAPI_ISteamFriends_GetFriendsGroupMembersCount( ISteamFriends* self, FriendsGroupID_t friendsGroupID );
S_API void SteamAPI_ISteamFriends_GetFriendsGroupMembersList( ISteamFriends* self, FriendsGroupID_t friendsGroupID, CSteamID * pOutSteamIDMembers, int nMembersCount );
S_API bool SteamAPI_ISteamFriends_HasFriend( ISteamFriends* self, uint64_steamid steamIDFriend, int iFriendFlags );
S_API int SteamAPI_ISteamFriends_GetClanCount( ISteamFriends* self );
S_API uint64_steamid SteamAPI_ISteamFriends_GetClanByIndex( ISteamFriends* self, int iClan );
S_API const char * SteamAPI_ISteamFriends_GetClanName( ISteamFriends* self, uint64_steamid steamIDClan );
S_API const char * SteamAPI_ISteamFriends_GetClanTag( ISteamFriends* self, uint64_steamid steamIDClan );
S_API bool SteamAPI_ISteamFriends_GetClanActivityCounts( ISteamFriends* self, uint64_steamid steamIDClan, int * pnOnline, int * pnInGame, int * pnChatting );
S_API SteamAPICall_t SteamAPI_ISteamFriends_DownloadClanActivityCounts( ISteamFriends* self, CSteamID * psteamIDClans, int cClansToRequest );
S_API int SteamAPI_ISteamFriends_GetFriendCountFromSource( ISteamFriends* self, uint64_steamid steamIDSource );
S_API uint64_steamid SteamAPI_ISteamFriends_GetFriendFromSourceByIndex( ISteamFriends* self, uint64_steamid steamIDSource, int iFriend );
S_API bool SteamAPI_ISteamFriends_IsUserInSource( ISteamFriends* self, uint64_steamid steamIDUser, uint64_steamid steamIDSource );
S_API void SteamAPI_ISteamFriends_SetInGameVoiceSpeaking( ISteamFriends* self, uint64_steamid steamIDUser, bool bSpeaking );
S_API void SteamAPI_ISteamFriends_ActivateGameOverlay( ISteamFriends* self, const char * pchDialog );
S_API void SteamAPI_ISteamFriends_ActivateGameOverlayToUser( ISteamFriends* self, const char * pchDialog, uint64_steamid steamID );
S_API void SteamAPI_ISteamFriends_ActivateGameOverlayToWebPage( ISteamFriends* self, const char * pchURL, EActivateGameOverlayToWebPageMode eMode );
S_API void SteamAPI_ISteamFriends_ActivateGameOverlayToStore( ISteamFriends* self, AppId_t nAppID, EOverlayToStoreFlag eFlag );
S_API void SteamAPI_ISteamFriends_SetPlayedWith( ISteamFriends* self, uint64_steamid steamIDUserPlayedWith );
S_API void SteamAPI_ISteamFriends_ActivateGameOverlayInviteDialog( ISteamFriends* self, uint64_steamid steamIDLobby );
S_API int SteamAPI_ISteamFriends_GetSmallFriendAvatar( ISteamFriends* self, uint64_steamid steamIDFriend );
S_API int SteamAPI_ISteamFriends_GetMediumFriendAvatar( ISteamFriends* self, uint64_steamid steamIDFriend );
S_API int SteamAPI_ISteamFriends_GetLargeFriendAvatar( ISteamFriends* self, uint64_steamid steamIDFriend );
S_API bool SteamAPI_ISteamFriends_RequestUserInformation( ISteamFriends* self, uint64_steamid steamIDUser, bool bRequireNameOnly );
S_API SteamAPICall_t SteamAPI_ISteamFriends_RequestClanOfficerList( ISteamFriends* self, uint64_steamid steamIDClan );
S_API uint64_steamid SteamAPI_ISteamFriends_GetClanOwner( ISteamFriends* self, uint64_steamid steamIDClan );
S_API int SteamAPI_ISteamFriends_GetClanOfficerCount( ISteamFriends* self, uint64_steamid steamIDClan );
S_API uint64_steamid SteamAPI_ISteamFriends_GetClanOfficerByIndex( ISteamFriends* self, uint64_steamid steamIDClan, int iOfficer );
S_API uint32 SteamAPI_ISteamFriends_GetUserRestrictions( ISteamFriends* self );
S_API bool SteamAPI_ISteamFriends_SetRichPresence( ISteamFriends* self, const char * pchKey, const char * pchValue );
S_API void SteamAPI_ISteamFriends_ClearRichPresence( ISteamFriends* self );
S_API const char * SteamAPI_ISteamFriends_GetFriendRichPresence( ISteamFriends* self, uint64_steamid steamIDFriend, const char * pchKey );
S_API int SteamAPI_ISteamFriends_GetFriendRichPresenceKeyCount( ISteamFriends* self, uint64_steamid steamIDFriend );
S_API const char * SteamAPI_ISteamFriends_GetFriendRichPresenceKeyByIndex( ISteamFriends* self, uint64_steamid steamIDFriend, int iKey );
S_API void SteamAPI_ISteamFriends_RequestFriendRichPresence( ISteamFriends* self, uint64_steamid steamIDFriend );
S_API bool SteamAPI_ISteamFriends_InviteUserToGame( ISteamFriends* self, uint64_steamid steamIDFriend, const char * pchConnectString );
S_API int SteamAPI_ISteamFriends_GetCoplayFriendCount( ISteamFriends* self );
S_API uint64_steamid SteamAPI_ISteamFriends_GetCoplayFriend( ISteamFriends* self, int iCoplayFriend );
S_API int SteamAPI_ISteamFriends_GetFriendCoplayTime( ISteamFriends* self, uint64_steamid steamIDFriend );
S_API AppId_t SteamAPI_ISteamFriends_GetFriendCoplayGame( ISteamFriends* self, uint64_steamid steamIDFriend );
S_API SteamAPICall_t SteamAPI_ISteamFriends_JoinClanChatRoom( ISteamFriends* self, uint64_steamid steamIDClan );
S_API bool SteamAPI_ISteamFriends_LeaveClanChatRoom( ISteamFriends* self, uint64_steamid steamIDClan );
S_API int SteamAPI_ISteamFriends_GetClanChatMemberCount( ISteamFriends* self, uint64_steamid steamIDClan );
S_API uint64_steamid SteamAPI_ISteamFriends_GetChatMemberByIndex( ISteamFriends* self, uint64_steamid steamIDClan, int iUser );
S_API bool SteamAPI_ISteamFriends_SendClanChatMessage( ISteamFriends* self, uint64_steamid steamIDClanChat, const char * pchText );
S_API int SteamAPI_ISteamFriends_GetClanChatMessage( ISteamFriends* self, uint64_steamid steamIDClanChat, int iMessage, void * prgchText, int cchTextMax, EChatEntryType * peChatEntryType, CSteamID * psteamidChatter );
S_API bool SteamAPI_ISteamFriends_IsClanChatAdmin( ISteamFriends* self, uint64_steamid steamIDClanChat, uint64_steamid steamIDUser );
S_API bool SteamAPI_ISteamFriends_IsClanChatWindowOpenInSteam( ISteamFriends* self, uint64_steamid steamIDClanChat );
S_API bool SteamAPI_ISteamFriends_OpenClanChatWindowInSteam( ISteamFriends* self, uint64_steamid steamIDClanChat );
S_API bool SteamAPI_ISteamFriends_CloseClanChatWindowInSteam( ISteamFriends* self, uint64_steamid steamIDClanChat );
S_API bool SteamAPI_ISteamFriends_SetListenForFriendsMessages( ISteamFriends* self, bool bInterceptEnabled );
S_API bool SteamAPI_ISteamFriends_ReplyToFriendMessage( ISteamFriends* self, uint64_steamid steamIDFriend, const char * pchMsgToSend );
S_API int SteamAPI_ISteamFriends_GetFriendMessage( ISteamFriends* self, uint64_steamid steamIDFriend, int iMessageID, void * pvData, int cubData, EChatEntryType * peChatEntryType );
S_API SteamAPICall_t SteamAPI_ISteamFriends_GetFollowerCount( ISteamFriends* self, uint64_steamid steamID );
S_API SteamAPICall_t SteamAPI_ISteamFriends_IsFollowing( ISteamFriends* self, uint64_steamid steamID );
S_API SteamAPICall_t SteamAPI_ISteamFriends_EnumerateFollowingList( ISteamFriends* self, uint32 unStartIndex );
S_API bool SteamAPI_ISteamFriends_IsClanPublic( ISteamFriends* self, uint64_steamid steamIDClan );
S_API bool SteamAPI_ISteamFriends_IsClanOfficialGameGroup( ISteamFriends* self, uint64_steamid steamIDClan );
S_API int SteamAPI_ISteamFriends_GetNumChatsWithUnreadPriorityMessages( ISteamFriends* self );
S_API void SteamAPI_ISteamFriends_ActivateGameOverlayRemotePlayTogetherInviteDialog( ISteamFriends* self, uint64_steamid steamIDLobby );
S_API bool SteamAPI_ISteamFriends_RegisterProtocolInOverlayBrowser( ISteamFriends* self, const char * pchProtocol );
S_API void SteamAPI_ISteamFriends_ActivateGameOverlayInviteDialogConnectString( ISteamFriends* self, const char * pchConnectString );
S_API SteamAPICall_t SteamAPI_ISteamFriends_RequestEquippedProfileItems( ISteamFriends* self, uint64_steamid steamID );
S_API bool SteamAPI_ISteamFriends_BHasEquippedProfileItem( ISteamFriends* self, uint64_steamid steamID, ECommunityProfileItemType itemType );
S_API const char * SteamAPI_ISteamFriends_GetProfileItemPropertyString( ISteamFriends* self, uint64_steamid steamID, ECommunityProfileItemType itemType, ECommunityProfileItemProperty prop );
S_API uint32 SteamAPI_ISteamFriends_GetProfileItemPropertyUint( ISteamFriends* self, uint64_steamid steamID, ECommunityProfileItemType itemType, ECommunityProfileItemProperty prop );

// ISteamUtils

// A versioned accessor is exported by the library
S_API ISteamUtils *SteamAPI_SteamUtils_v010();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamUtils(), but using this ensures that you are using a matching library.
inline ISteamUtils *SteamAPI_SteamUtils() { return SteamAPI_SteamUtils_v010(); }

// A versioned accessor is exported by the library
S_API ISteamUtils *SteamAPI_SteamGameServerUtils_v010();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamGameServerUtils(), but using this ensures that you are using a matching library.
inline ISteamUtils *SteamAPI_SteamGameServerUtils() { return SteamAPI_SteamGameServerUtils_v010(); }
S_API uint32 SteamAPI_ISteamUtils_GetSecondsSinceAppActive( ISteamUtils* self );
S_API uint32 SteamAPI_ISteamUtils_GetSecondsSinceComputerActive( ISteamUtils* self );
S_API EUniverse SteamAPI_ISteamUtils_GetConnectedUniverse( ISteamUtils* self );
S_API uint32 SteamAPI_ISteamUtils_GetServerRealTime( ISteamUtils* self );
S_API const char * SteamAPI_ISteamUtils_GetIPCountry( ISteamUtils* self );
S_API bool SteamAPI_ISteamUtils_GetImageSize( ISteamUtils* self, int iImage, uint32 * pnWidth, uint32 * pnHeight );
S_API bool SteamAPI_ISteamUtils_GetImageRGBA( ISteamUtils* self, int iImage, uint8 * pubDest, int nDestBufferSize );
S_API uint8 SteamAPI_ISteamUtils_GetCurrentBatteryPower( ISteamUtils* self );
S_API uint32 SteamAPI_ISteamUtils_GetAppID( ISteamUtils* self );
S_API void SteamAPI_ISteamUtils_SetOverlayNotificationPosition( ISteamUtils* self, ENotificationPosition eNotificationPosition );
S_API bool SteamAPI_ISteamUtils_IsAPICallCompleted( ISteamUtils* self, SteamAPICall_t hSteamAPICall, bool * pbFailed );
S_API ESteamAPICallFailure SteamAPI_ISteamUtils_GetAPICallFailureReason( ISteamUtils* self, SteamAPICall_t hSteamAPICall );
S_API bool SteamAPI_ISteamUtils_GetAPICallResult( ISteamUtils* self, SteamAPICall_t hSteamAPICall, void * pCallback, int cubCallback, int iCallbackExpected, bool * pbFailed );
S_API uint32 SteamAPI_ISteamUtils_GetIPCCallCount( ISteamUtils* self );
S_API void SteamAPI_ISteamUtils_SetWarningMessageHook( ISteamUtils* self, SteamAPIWarningMessageHook_t pFunction );
S_API bool SteamAPI_ISteamUtils_IsOverlayEnabled( ISteamUtils* self );
S_API bool SteamAPI_ISteamUtils_BOverlayNeedsPresent( ISteamUtils* self );
S_API SteamAPICall_t SteamAPI_ISteamUtils_CheckFileSignature( ISteamUtils* self, const char * szFileName );
S_API bool SteamAPI_ISteamUtils_ShowGamepadTextInput( ISteamUtils* self, EGamepadTextInputMode eInputMode, EGamepadTextInputLineMode eLineInputMode, const char * pchDescription, uint32 unCharMax, const char * pchExistingText );
S_API uint32 SteamAPI_ISteamUtils_GetEnteredGamepadTextLength( ISteamUtils* self );
S_API bool SteamAPI_ISteamUtils_GetEnteredGamepadTextInput( ISteamUtils* self, char * pchText, uint32 cchText );
S_API const char * SteamAPI_ISteamUtils_GetSteamUILanguage( ISteamUtils* self );
S_API bool SteamAPI_ISteamUtils_IsSteamRunningInVR( ISteamUtils* self );
S_API void SteamAPI_ISteamUtils_SetOverlayNotificationInset( ISteamUtils* self, int nHorizontalInset, int nVerticalInset );
S_API bool SteamAPI_ISteamUtils_IsSteamInBigPictureMode( ISteamUtils* self );
S_API void SteamAPI_ISteamUtils_StartVRDashboard( ISteamUtils* self );
S_API bool SteamAPI_ISteamUtils_IsVRHeadsetStreamingEnabled( ISteamUtils* self );
S_API void SteamAPI_ISteamUtils_SetVRHeadsetStreamingEnabled( ISteamUtils* self, bool bEnabled );
S_API bool SteamAPI_ISteamUtils_IsSteamChinaLauncher( ISteamUtils* self );
S_API bool SteamAPI_ISteamUtils_InitFilterText( ISteamUtils* self, uint32 unFilterOptions );
S_API int SteamAPI_ISteamUtils_FilterText( ISteamUtils* self, ETextFilteringContext eContext, uint64_steamid sourceSteamID, const char * pchInputMessage, char * pchOutFilteredText, uint32 nByteSizeOutFilteredText );
S_API ESteamIPv6ConnectivityState SteamAPI_ISteamUtils_GetIPv6ConnectivityState( ISteamUtils* self, ESteamIPv6ConnectivityProtocol eProtocol );
S_API bool SteamAPI_ISteamUtils_IsSteamRunningOnSteamDeck( ISteamUtils* self );
S_API bool SteamAPI_ISteamUtils_ShowFloatingGamepadTextInput( ISteamUtils* self, EFloatingGamepadTextInputMode eKeyboardMode, int nTextFieldXPosition, int nTextFieldYPosition, int nTextFieldWidth, int nTextFieldHeight );
S_API void SteamAPI_ISteamUtils_SetGameLauncherMode( ISteamUtils* self, bool bLauncherMode );
S_API bool SteamAPI_ISteamUtils_DismissFloatingGamepadTextInput( ISteamUtils* self );

// ISteamMatchmaking

// A versioned accessor is exported by the library
S_API ISteamMatchmaking *SteamAPI_SteamMatchmaking_v009();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamMatchmaking(), but using this ensures that you are using a matching library.
inline ISteamMatchmaking *SteamAPI_SteamMatchmaking() { return SteamAPI_SteamMatchmaking_v009(); }
S_API int SteamAPI_ISteamMatchmaking_GetFavoriteGameCount( ISteamMatchmaking* self );
S_API bool SteamAPI_ISteamMatchmaking_GetFavoriteGame( ISteamMatchmaking* self, int iGame, AppId_t * pnAppID, uint32 * pnIP, uint16 * pnConnPort, uint16 * pnQueryPort, uint32 * punFlags, uint32 * pRTime32LastPlayedOnServer );
S_API int SteamAPI_ISteamMatchmaking_AddFavoriteGame( ISteamMatchmaking* self, AppId_t nAppID, uint32 nIP, uint16 nConnPort, uint16 nQueryPort, uint32 unFlags, uint32 rTime32LastPlayedOnServer );
S_API bool SteamAPI_ISteamMatchmaking_RemoveFavoriteGame( ISteamMatchmaking* self, AppId_t nAppID, uint32 nIP, uint16 nConnPort, uint16 nQueryPort, uint32 unFlags );
S_API SteamAPICall_t SteamAPI_ISteamMatchmaking_RequestLobbyList( ISteamMatchmaking* self );
S_API void SteamAPI_ISteamMatchmaking_AddRequestLobbyListStringFilter( ISteamMatchmaking* self, const char * pchKeyToMatch, const char * pchValueToMatch, ELobbyComparison eComparisonType );
S_API void SteamAPI_ISteamMatchmaking_AddRequestLobbyListNumericalFilter( ISteamMatchmaking* self, const char * pchKeyToMatch, int nValueToMatch, ELobbyComparison eComparisonType );
S_API void SteamAPI_ISteamMatchmaking_AddRequestLobbyListNearValueFilter( ISteamMatchmaking* self, const char * pchKeyToMatch, int nValueToBeCloseTo );
S_API void SteamAPI_ISteamMatchmaking_AddRequestLobbyListFilterSlotsAvailable( ISteamMatchmaking* self, int nSlotsAvailable );
S_API void SteamAPI_ISteamMatchmaking_AddRequestLobbyListDistanceFilter( ISteamMatchmaking* self, ELobbyDistanceFilter eLobbyDistanceFilter );
S_API void SteamAPI_ISteamMatchmaking_AddRequestLobbyListResultCountFilter( ISteamMatchmaking* self, int cMaxResults );
S_API void SteamAPI_ISteamMatchmaking_AddRequestLobbyListCompatibleMembersFilter( ISteamMatchmaking* self, uint64_steamid steamIDLobby );
S_API uint64_steamid SteamAPI_ISteamMatchmaking_GetLobbyByIndex( ISteamMatchmaking* self, int iLobby );
S_API SteamAPICall_t SteamAPI_ISteamMatchmaking_CreateLobby( ISteamMatchmaking* self, ELobbyType eLobbyType, int cMaxMembers );
S_API SteamAPICall_t SteamAPI_ISteamMatchmaking_JoinLobby( ISteamMatchmaking* self, uint64_steamid steamIDLobby );
S_API void SteamAPI_ISteamMatchmaking_LeaveLobby( ISteamMatchmaking* self, uint64_steamid steamIDLobby );
S_API bool SteamAPI_ISteamMatchmaking_InviteUserToLobby( ISteamMatchmaking* self, uint64_steamid steamIDLobby, uint64_steamid steamIDInvitee );
S_API int SteamAPI_ISteamMatchmaking_GetNumLobbyMembers( ISteamMatchmaking* self, uint64_steamid steamIDLobby );
S_API uint64_steamid SteamAPI_ISteamMatchmaking_GetLobbyMemberByIndex( ISteamMatchmaking* self, uint64_steamid steamIDLobby, int iMember );
S_API const char * SteamAPI_ISteamMatchmaking_GetLobbyData( ISteamMatchmaking* self, uint64_steamid steamIDLobby, const char * pchKey );
S_API bool SteamAPI_ISteamMatchmaking_SetLobbyData( ISteamMatchmaking* self, uint64_steamid steamIDLobby, const char * pchKey, const char * pchValue );
S_API int SteamAPI_ISteamMatchmaking_GetLobbyDataCount( ISteamMatchmaking* self, uint64_steamid steamIDLobby );
S_API bool SteamAPI_ISteamMatchmaking_GetLobbyDataByIndex( ISteamMatchmaking* self, uint64_steamid steamIDLobby, int iLobbyData, char * pchKey, int cchKeyBufferSize, char * pchValue, int cchValueBufferSize );
S_API bool SteamAPI_ISteamMatchmaking_DeleteLobbyData( ISteamMatchmaking* self, uint64_steamid steamIDLobby, const char * pchKey );
S_API const char * SteamAPI_ISteamMatchmaking_GetLobbyMemberData( ISteamMatchmaking* self, uint64_steamid steamIDLobby, uint64_steamid steamIDUser, const char * pchKey );
S_API void SteamAPI_ISteamMatchmaking_SetLobbyMemberData( ISteamMatchmaking* self, uint64_steamid steamIDLobby, const char * pchKey, const char * pchValue );
S_API bool SteamAPI_ISteamMatchmaking_SendLobbyChatMsg( ISteamMatchmaking* self, uint64_steamid steamIDLobby, const void * pvMsgBody, int cubMsgBody );
S_API int SteamAPI_ISteamMatchmaking_GetLobbyChatEntry( ISteamMatchmaking* self, uint64_steamid steamIDLobby, int iChatID, CSteamID * pSteamIDUser, void * pvData, int cubData, EChatEntryType * peChatEntryType );
S_API bool SteamAPI_ISteamMatchmaking_RequestLobbyData( ISteamMatchmaking* self, uint64_steamid steamIDLobby );
S_API void SteamAPI_ISteamMatchmaking_SetLobbyGameServer( ISteamMatchmaking* self, uint64_steamid steamIDLobby, uint32 unGameServerIP, uint16 unGameServerPort, uint64_steamid steamIDGameServer );
S_API bool SteamAPI_ISteamMatchmaking_GetLobbyGameServer( ISteamMatchmaking* self, uint64_steamid steamIDLobby, uint32 * punGameServerIP, uint16 * punGameServerPort, CSteamID * psteamIDGameServer );
S_API bool SteamAPI_ISteamMatchmaking_SetLobbyMemberLimit( ISteamMatchmaking* self, uint64_steamid steamIDLobby, int cMaxMembers );
S_API int SteamAPI_ISteamMatchmaking_GetLobbyMemberLimit( ISteamMatchmaking* self, uint64_steamid steamIDLobby );
S_API bool SteamAPI_ISteamMatchmaking_SetLobbyType( ISteamMatchmaking* self, uint64_steamid steamIDLobby, ELobbyType eLobbyType );
S_API bool SteamAPI_ISteamMatchmaking_SetLobbyJoinable( ISteamMatchmaking* self, uint64_steamid steamIDLobby, bool bLobbyJoinable );
S_API uint64_steamid SteamAPI_ISteamMatchmaking_GetLobbyOwner( ISteamMatchmaking* self, uint64_steamid steamIDLobby );
S_API bool SteamAPI_ISteamMatchmaking_SetLobbyOwner( ISteamMatchmaking* self, uint64_steamid steamIDLobby, uint64_steamid steamIDNewOwner );
S_API bool SteamAPI_ISteamMatchmaking_SetLinkedLobby( ISteamMatchmaking* self, uint64_steamid steamIDLobby, uint64_steamid steamIDLobbyDependent );

// ISteamMatchmakingServerListResponse
S_API void SteamAPI_ISteamMatchmakingServerListResponse_ServerResponded( ISteamMatchmakingServerListResponse* self, HServerListRequest hRequest, int iServer );
S_API void SteamAPI_ISteamMatchmakingServerListResponse_ServerFailedToRespond( ISteamMatchmakingServerListResponse* self, HServerListRequest hRequest, int iServer );
S_API void SteamAPI_ISteamMatchmakingServerListResponse_RefreshComplete( ISteamMatchmakingServerListResponse* self, HServerListRequest hRequest, EMatchMakingServerResponse response );

// ISteamMatchmakingPingResponse
S_API void SteamAPI_ISteamMatchmakingPingResponse_ServerResponded( ISteamMatchmakingPingResponse* self, gameserveritem_t & server );
S_API void SteamAPI_ISteamMatchmakingPingResponse_ServerFailedToRespond( ISteamMatchmakingPingResponse* self );

// ISteamMatchmakingPlayersResponse
S_API void SteamAPI_ISteamMatchmakingPlayersResponse_AddPlayerToList( ISteamMatchmakingPlayersResponse* self, const char * pchName, int nScore, float flTimePlayed );
S_API void SteamAPI_ISteamMatchmakingPlayersResponse_PlayersFailedToRespond( ISteamMatchmakingPlayersResponse* self );
S_API void SteamAPI_ISteamMatchmakingPlayersResponse_PlayersRefreshComplete( ISteamMatchmakingPlayersResponse* self );

// ISteamMatchmakingRulesResponse
S_API void SteamAPI_ISteamMatchmakingRulesResponse_RulesResponded( ISteamMatchmakingRulesResponse* self, const char * pchRule, const char * pchValue );
S_API void SteamAPI_ISteamMatchmakingRulesResponse_RulesFailedToRespond( ISteamMatchmakingRulesResponse* self );
S_API void SteamAPI_ISteamMatchmakingRulesResponse_RulesRefreshComplete( ISteamMatchmakingRulesResponse* self );

// ISteamMatchmakingServers

// A versioned accessor is exported by the library
S_API ISteamMatchmakingServers *SteamAPI_SteamMatchmakingServers_v002();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamMatchmakingServers(), but using this ensures that you are using a matching library.
inline ISteamMatchmakingServers *SteamAPI_SteamMatchmakingServers() { return SteamAPI_SteamMatchmakingServers_v002(); }
S_API HServerListRequest SteamAPI_ISteamMatchmakingServers_RequestInternetServerList( ISteamMatchmakingServers* self, AppId_t iApp, MatchMakingKeyValuePair_t ** ppchFilters, uint32 nFilters, ISteamMatchmakingServerListResponse * pRequestServersResponse );
S_API HServerListRequest SteamAPI_ISteamMatchmakingServers_RequestLANServerList( ISteamMatchmakingServers* self, AppId_t iApp, ISteamMatchmakingServerListResponse * pRequestServersResponse );
S_API HServerListRequest SteamAPI_ISteamMatchmakingServers_RequestFriendsServerList( ISteamMatchmakingServers* self, AppId_t iApp, MatchMakingKeyValuePair_t ** ppchFilters, uint32 nFilters, ISteamMatchmakingServerListResponse * pRequestServersResponse );
S_API HServerListRequest SteamAPI_ISteamMatchmakingServers_RequestFavoritesServerList( ISteamMatchmakingServers* self, AppId_t iApp, MatchMakingKeyValuePair_t ** ppchFilters, uint32 nFilters, ISteamMatchmakingServerListResponse * pRequestServersResponse );
S_API HServerListRequest SteamAPI_ISteamMatchmakingServers_RequestHistoryServerList( ISteamMatchmakingServers* self, AppId_t iApp, MatchMakingKeyValuePair_t ** ppchFilters, uint32 nFilters, ISteamMatchmakingServerListResponse * pRequestServersResponse );
S_API HServerListRequest SteamAPI_ISteamMatchmakingServers_RequestSpectatorServerList( ISteamMatchmakingServers* self, AppId_t iApp, MatchMakingKeyValuePair_t ** ppchFilters, uint32 nFilters, ISteamMatchmakingServerListResponse * pRequestServersResponse );
S_API void SteamAPI_ISteamMatchmakingServers_ReleaseRequest( ISteamMatchmakingServers* self, HServerListRequest hServerListRequest );
S_API gameserveritem_t * SteamAPI_ISteamMatchmakingServers_GetServerDetails( ISteamMatchmakingServers* self, HServerListRequest hRequest, int iServer );
S_API void SteamAPI_ISteamMatchmakingServers_CancelQuery( ISteamMatchmakingServers* self, HServerListRequest hRequest );
S_API void SteamAPI_ISteamMatchmakingServers_RefreshQuery( ISteamMatchmakingServers* self, HServerListRequest hRequest );
S_API bool SteamAPI_ISteamMatchmakingServers_IsRefreshing( ISteamMatchmakingServers* self, HServerListRequest hRequest );
S_API int SteamAPI_ISteamMatchmakingServers_GetServerCount( ISteamMatchmakingServers* self, HServerListRequest hRequest );
S_API void SteamAPI_ISteamMatchmakingServers_RefreshServer( ISteamMatchmakingServers* self, HServerListRequest hRequest, int iServer );
S_API HServerQuery SteamAPI_ISteamMatchmakingServers_PingServer( ISteamMatchmakingServers* self, uint32 unIP, uint16 usPort, ISteamMatchmakingPingResponse * pRequestServersResponse );
S_API HServerQuery SteamAPI_ISteamMatchmakingServers_PlayerDetails( ISteamMatchmakingServers* self, uint32 unIP, uint16 usPort, ISteamMatchmakingPlayersResponse * pRequestServersResponse );
S_API HServerQuery SteamAPI_ISteamMatchmakingServers_ServerRules( ISteamMatchmakingServers* self, uint32 unIP, uint16 usPort, ISteamMatchmakingRulesResponse * pRequestServersResponse );
S_API void SteamAPI_ISteamMatchmakingServers_CancelServerQuery( ISteamMatchmakingServers* self, HServerQuery hServerQuery );

// ISteamGameSearch

// A versioned accessor is exported by the library
S_API ISteamGameSearch *SteamAPI_SteamGameSearch_v001();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamGameSearch(), but using this ensures that you are using a matching library.
inline ISteamGameSearch *SteamAPI_SteamGameSearch() { return SteamAPI_SteamGameSearch_v001(); }
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_AddGameSearchParams( ISteamGameSearch* self, const char * pchKeyToFind, const char * pchValuesToFind );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_SearchForGameWithLobby( ISteamGameSearch* self, uint64_steamid steamIDLobby, int nPlayerMin, int nPlayerMax );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_SearchForGameSolo( ISteamGameSearch* self, int nPlayerMin, int nPlayerMax );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_AcceptGame( ISteamGameSearch* self );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_DeclineGame( ISteamGameSearch* self );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_RetrieveConnectionDetails( ISteamGameSearch* self, uint64_steamid steamIDHost, char * pchConnectionDetails, int cubConnectionDetails );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_EndGameSearch( ISteamGameSearch* self );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_SetGameHostParams( ISteamGameSearch* self, const char * pchKey, const char * pchValue );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_SetConnectionDetails( ISteamGameSearch* self, const char * pchConnectionDetails, int cubConnectionDetails );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_RequestPlayersForGame( ISteamGameSearch* self, int nPlayerMin, int nPlayerMax, int nMaxTeamSize );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_HostConfirmGameStart( ISteamGameSearch* self, uint64 ullUniqueGameID );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_CancelRequestPlayersForGame( ISteamGameSearch* self );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_SubmitPlayerResult( ISteamGameSearch* self, uint64 ullUniqueGameID, uint64_steamid steamIDPlayer, EPlayerResult_t EPlayerResult );
S_API EGameSearchErrorCode_t SteamAPI_ISteamGameSearch_EndGame( ISteamGameSearch* self, uint64 ullUniqueGameID );

// ISteamParties

// A versioned accessor is exported by the library
S_API ISteamParties *SteamAPI_SteamParties_v002();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamParties(), but using this ensures that you are using a matching library.
inline ISteamParties *SteamAPI_SteamParties() { return SteamAPI_SteamParties_v002(); }
S_API uint32 SteamAPI_ISteamParties_GetNumActiveBeacons( ISteamParties* self );
S_API PartyBeaconID_t SteamAPI_ISteamParties_GetBeaconByIndex( ISteamParties* self, uint32 unIndex );
S_API bool SteamAPI_ISteamParties_GetBeaconDetails( ISteamParties* self, PartyBeaconID_t ulBeaconID, CSteamID * pSteamIDBeaconOwner, SteamPartyBeaconLocation_t * pLocation, char * pchMetadata, int cchMetadata );
S_API SteamAPICall_t SteamAPI_ISteamParties_JoinParty( ISteamParties* self, PartyBeaconID_t ulBeaconID );
S_API bool SteamAPI_ISteamParties_GetNumAvailableBeaconLocations( ISteamParties* self, uint32 * puNumLocations );
S_API bool SteamAPI_ISteamParties_GetAvailableBeaconLocations( ISteamParties* self, SteamPartyBeaconLocation_t * pLocationList, uint32 uMaxNumLocations );
S_API SteamAPICall_t SteamAPI_ISteamParties_CreateBeacon( ISteamParties* self, uint32 unOpenSlots, SteamPartyBeaconLocation_t * pBeaconLocation, const char * pchConnectString, const char * pchMetadata );
S_API void SteamAPI_ISteamParties_OnReservationCompleted( ISteamParties* self, PartyBeaconID_t ulBeacon, uint64_steamid steamIDUser );
S_API void SteamAPI_ISteamParties_CancelReservation( ISteamParties* self, PartyBeaconID_t ulBeacon, uint64_steamid steamIDUser );
S_API SteamAPICall_t SteamAPI_ISteamParties_ChangeNumOpenSlots( ISteamParties* self, PartyBeaconID_t ulBeacon, uint32 unOpenSlots );
S_API bool SteamAPI_ISteamParties_DestroyBeacon( ISteamParties* self, PartyBeaconID_t ulBeacon );
S_API bool SteamAPI_ISteamParties_GetBeaconLocationData( ISteamParties* self, SteamPartyBeaconLocation_t BeaconLocation, ESteamPartyBeaconLocationData eData, char * pchDataStringOut, int cchDataStringOut );

// ISteamRemoteStorage

// A versioned accessor is exported by the library
S_API ISteamRemoteStorage *SteamAPI_SteamRemoteStorage_v016();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamRemoteStorage(), but using this ensures that you are using a matching library.
inline ISteamRemoteStorage *SteamAPI_SteamRemoteStorage() { return SteamAPI_SteamRemoteStorage_v016(); }
S_API bool SteamAPI_ISteamRemoteStorage_FileWrite( ISteamRemoteStorage* self, const char * pchFile, const void * pvData, int32 cubData );
S_API int32 SteamAPI_ISteamRemoteStorage_FileRead( ISteamRemoteStorage* self, const char * pchFile, void * pvData, int32 cubDataToRead );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_FileWriteAsync( ISteamRemoteStorage* self, const char * pchFile, const void * pvData, uint32 cubData );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_FileReadAsync( ISteamRemoteStorage* self, const char * pchFile, uint32 nOffset, uint32 cubToRead );
S_API bool SteamAPI_ISteamRemoteStorage_FileReadAsyncComplete( ISteamRemoteStorage* self, SteamAPICall_t hReadCall, void * pvBuffer, uint32 cubToRead );
S_API bool SteamAPI_ISteamRemoteStorage_FileForget( ISteamRemoteStorage* self, const char * pchFile );
S_API bool SteamAPI_ISteamRemoteStorage_FileDelete( ISteamRemoteStorage* self, const char * pchFile );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_FileShare( ISteamRemoteStorage* self, const char * pchFile );
S_API bool SteamAPI_ISteamRemoteStorage_SetSyncPlatforms( ISteamRemoteStorage* self, const char * pchFile, ERemoteStoragePlatform eRemoteStoragePlatform );
S_API UGCFileWriteStreamHandle_t SteamAPI_ISteamRemoteStorage_FileWriteStreamOpen( ISteamRemoteStorage* self, const char * pchFile );
S_API bool SteamAPI_ISteamRemoteStorage_FileWriteStreamWriteChunk( ISteamRemoteStorage* self, UGCFileWriteStreamHandle_t writeHandle, const void * pvData, int32 cubData );
S_API bool SteamAPI_ISteamRemoteStorage_FileWriteStreamClose( ISteamRemoteStorage* self, UGCFileWriteStreamHandle_t writeHandle );
S_API bool SteamAPI_ISteamRemoteStorage_FileWriteStreamCancel( ISteamRemoteStorage* self, UGCFileWriteStreamHandle_t writeHandle );
S_API bool SteamAPI_ISteamRemoteStorage_FileExists( ISteamRemoteStorage* self, const char * pchFile );
S_API bool SteamAPI_ISteamRemoteStorage_FilePersisted( ISteamRemoteStorage* self, const char * pchFile );
S_API int32 SteamAPI_ISteamRemoteStorage_GetFileSize( ISteamRemoteStorage* self, const char * pchFile );
S_API int64 SteamAPI_ISteamRemoteStorage_GetFileTimestamp( ISteamRemoteStorage* self, const char * pchFile );
S_API ERemoteStoragePlatform SteamAPI_ISteamRemoteStorage_GetSyncPlatforms( ISteamRemoteStorage* self, const char * pchFile );
S_API int32 SteamAPI_ISteamRemoteStorage_GetFileCount( ISteamRemoteStorage* self );
S_API const char * SteamAPI_ISteamRemoteStorage_GetFileNameAndSize( ISteamRemoteStorage* self, int iFile, int32 * pnFileSizeInBytes );
S_API bool SteamAPI_ISteamRemoteStorage_GetQuota( ISteamRemoteStorage* self, uint64 * pnTotalBytes, uint64 * puAvailableBytes );
S_API bool SteamAPI_ISteamRemoteStorage_IsCloudEnabledForAccount( ISteamRemoteStorage* self );
S_API bool SteamAPI_ISteamRemoteStorage_IsCloudEnabledForApp( ISteamRemoteStorage* self );
S_API void SteamAPI_ISteamRemoteStorage_SetCloudEnabledForApp( ISteamRemoteStorage* self, bool bEnabled );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_UGCDownload( ISteamRemoteStorage* self, UGCHandle_t hContent, uint32 unPriority );
S_API bool SteamAPI_ISteamRemoteStorage_GetUGCDownloadProgress( ISteamRemoteStorage* self, UGCHandle_t hContent, int32 * pnBytesDownloaded, int32 * pnBytesExpected );
S_API bool SteamAPI_ISteamRemoteStorage_GetUGCDetails( ISteamRemoteStorage* self, UGCHandle_t hContent, AppId_t * pnAppID, char ** ppchName, int32 * pnFileSizeInBytes, CSteamID * pSteamIDOwner );
S_API int32 SteamAPI_ISteamRemoteStorage_UGCRead( ISteamRemoteStorage* self, UGCHandle_t hContent, void * pvData, int32 cubDataToRead, uint32 cOffset, EUGCReadAction eAction );
S_API int32 SteamAPI_ISteamRemoteStorage_GetCachedUGCCount( ISteamRemoteStorage* self );
S_API UGCHandle_t SteamAPI_ISteamRemoteStorage_GetCachedUGCHandle( ISteamRemoteStorage* self, int32 iCachedContent );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_PublishWorkshopFile( ISteamRemoteStorage* self, const char * pchFile, const char * pchPreviewFile, AppId_t nConsumerAppId, const char * pchTitle, const char * pchDescription, ERemoteStoragePublishedFileVisibility eVisibility, SteamParamStringArray_t * pTags, EWorkshopFileType eWorkshopFileType );
S_API PublishedFileUpdateHandle_t SteamAPI_ISteamRemoteStorage_CreatePublishedFileUpdateRequest( ISteamRemoteStorage* self, PublishedFileId_t unPublishedFileId );
S_API bool SteamAPI_ISteamRemoteStorage_UpdatePublishedFileFile( ISteamRemoteStorage* self, PublishedFileUpdateHandle_t updateHandle, const char * pchFile );
S_API bool SteamAPI_ISteamRemoteStorage_UpdatePublishedFilePreviewFile( ISteamRemoteStorage* self, PublishedFileUpdateHandle_t updateHandle, const char * pchPreviewFile );
S_API bool SteamAPI_ISteamRemoteStorage_UpdatePublishedFileTitle( ISteamRemoteStorage* self, PublishedFileUpdateHandle_t updateHandle, const char * pchTitle );
S_API bool SteamAPI_ISteamRemoteStorage_UpdatePublishedFileDescription( ISteamRemoteStorage* self, PublishedFileUpdateHandle_t updateHandle, const char * pchDescription );
S_API bool SteamAPI_ISteamRemoteStorage_UpdatePublishedFileVisibility( ISteamRemoteStorage* self, PublishedFileUpdateHandle_t updateHandle, ERemoteStoragePublishedFileVisibility eVisibility );
S_API bool SteamAPI_ISteamRemoteStorage_UpdatePublishedFileTags( ISteamRemoteStorage* self, PublishedFileUpdateHandle_t updateHandle, SteamParamStringArray_t * pTags );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_CommitPublishedFileUpdate( ISteamRemoteStorage* self, PublishedFileUpdateHandle_t updateHandle );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_GetPublishedFileDetails( ISteamRemoteStorage* self, PublishedFileId_t unPublishedFileId, uint32 unMaxSecondsOld );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_DeletePublishedFile( ISteamRemoteStorage* self, PublishedFileId_t unPublishedFileId );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_EnumerateUserPublishedFiles( ISteamRemoteStorage* self, uint32 unStartIndex );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_SubscribePublishedFile( ISteamRemoteStorage* self, PublishedFileId_t unPublishedFileId );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_EnumerateUserSubscribedFiles( ISteamRemoteStorage* self, uint32 unStartIndex );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_UnsubscribePublishedFile( ISteamRemoteStorage* self, PublishedFileId_t unPublishedFileId );
S_API bool SteamAPI_ISteamRemoteStorage_UpdatePublishedFileSetChangeDescription( ISteamRemoteStorage* self, PublishedFileUpdateHandle_t updateHandle, const char * pchChangeDescription );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_GetPublishedItemVoteDetails( ISteamRemoteStorage* self, PublishedFileId_t unPublishedFileId );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_UpdateUserPublishedItemVote( ISteamRemoteStorage* self, PublishedFileId_t unPublishedFileId, bool bVoteUp );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_GetUserPublishedItemVoteDetails( ISteamRemoteStorage* self, PublishedFileId_t unPublishedFileId );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_EnumerateUserSharedWorkshopFiles( ISteamRemoteStorage* self, uint64_steamid steamId, uint32 unStartIndex, SteamParamStringArray_t * pRequiredTags, SteamParamStringArray_t * pExcludedTags );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_PublishVideo( ISteamRemoteStorage* self, EWorkshopVideoProvider eVideoProvider, const char * pchVideoAccount, const char * pchVideoIdentifier, const char * pchPreviewFile, AppId_t nConsumerAppId, const char * pchTitle, const char * pchDescription, ERemoteStoragePublishedFileVisibility eVisibility, SteamParamStringArray_t * pTags );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_SetUserPublishedFileAction( ISteamRemoteStorage* self, PublishedFileId_t unPublishedFileId, EWorkshopFileAction eAction );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_EnumeratePublishedFilesByUserAction( ISteamRemoteStorage* self, EWorkshopFileAction eAction, uint32 unStartIndex );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_EnumeratePublishedWorkshopFiles( ISteamRemoteStorage* self, EWorkshopEnumerationType eEnumerationType, uint32 unStartIndex, uint32 unCount, uint32 unDays, SteamParamStringArray_t * pTags, SteamParamStringArray_t * pUserTags );
S_API SteamAPICall_t SteamAPI_ISteamRemoteStorage_UGCDownloadToLocation( ISteamRemoteStorage* self, UGCHandle_t hContent, const char * pchLocation, uint32 unPriority );
S_API int32 SteamAPI_ISteamRemoteStorage_GetLocalFileChangeCount( ISteamRemoteStorage* self );
S_API const char * SteamAPI_ISteamRemoteStorage_GetLocalFileChange( ISteamRemoteStorage* self, int iFile, ERemoteStorageLocalFileChange * pEChangeType, ERemoteStorageFilePathType * pEFilePathType );
S_API bool SteamAPI_ISteamRemoteStorage_BeginFileWriteBatch( ISteamRemoteStorage* self );
S_API bool SteamAPI_ISteamRemoteStorage_EndFileWriteBatch( ISteamRemoteStorage* self );

// ISteamUserStats

// A versioned accessor is exported by the library
S_API ISteamUserStats *SteamAPI_SteamUserStats_v012();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamUserStats(), but using this ensures that you are using a matching library.
inline ISteamUserStats *SteamAPI_SteamUserStats() { return SteamAPI_SteamUserStats_v012(); }
S_API bool SteamAPI_ISteamUserStats_RequestCurrentStats( ISteamUserStats* self );
S_API bool SteamAPI_ISteamUserStats_GetStatInt32( ISteamUserStats* self, const char * pchName, int32 * pData );
S_API bool SteamAPI_ISteamUserStats_GetStatFloat( ISteamUserStats* self, const char * pchName, float * pData );
S_API bool SteamAPI_ISteamUserStats_SetStatInt32( ISteamUserStats* self, const char * pchName, int32 nData );
S_API bool SteamAPI_ISteamUserStats_SetStatFloat( ISteamUserStats* self, const char * pchName, float fData );
S_API bool SteamAPI_ISteamUserStats_UpdateAvgRateStat( ISteamUserStats* self, const char * pchName, float flCountThisSession, double dSessionLength );
S_API bool SteamAPI_ISteamUserStats_GetAchievement( ISteamUserStats* self, const char * pchName, bool * pbAchieved );
S_API bool SteamAPI_ISteamUserStats_SetAchievement( ISteamUserStats* self, const char * pchName );
S_API bool SteamAPI_ISteamUserStats_ClearAchievement( ISteamUserStats* self, const char * pchName );
S_API bool SteamAPI_ISteamUserStats_GetAchievementAndUnlockTime( ISteamUserStats* self, const char * pchName, bool * pbAchieved, uint32 * punUnlockTime );
S_API bool SteamAPI_ISteamUserStats_StoreStats( ISteamUserStats* self );
S_API int SteamAPI_ISteamUserStats_GetAchievementIcon( ISteamUserStats* self, const char * pchName );
S_API const char * SteamAPI_ISteamUserStats_GetAchievementDisplayAttribute( ISteamUserStats* self, const char * pchName, const char * pchKey );
S_API bool SteamAPI_ISteamUserStats_IndicateAchievementProgress( ISteamUserStats* self, const char * pchName, uint32 nCurProgress, uint32 nMaxProgress );
S_API uint32 SteamAPI_ISteamUserStats_GetNumAchievements( ISteamUserStats* self );
S_API const char * SteamAPI_ISteamUserStats_GetAchievementName( ISteamUserStats* self, uint32 iAchievement );
S_API SteamAPICall_t SteamAPI_ISteamUserStats_RequestUserStats( ISteamUserStats* self, uint64_steamid steamIDUser );
S_API bool SteamAPI_ISteamUserStats_GetUserStatInt32( ISteamUserStats* self, uint64_steamid steamIDUser, const char * pchName, int32 * pData );
S_API bool SteamAPI_ISteamUserStats_GetUserStatFloat( ISteamUserStats* self, uint64_steamid steamIDUser, const char * pchName, float * pData );
S_API bool SteamAPI_ISteamUserStats_GetUserAchievement( ISteamUserStats* self, uint64_steamid steamIDUser, const char * pchName, bool * pbAchieved );
S_API bool SteamAPI_ISteamUserStats_GetUserAchievementAndUnlockTime( ISteamUserStats* self, uint64_steamid steamIDUser, const char * pchName, bool * pbAchieved, uint32 * punUnlockTime );
S_API bool SteamAPI_ISteamUserStats_ResetAllStats( ISteamUserStats* self, bool bAchievementsToo );
S_API SteamAPICall_t SteamAPI_ISteamUserStats_FindOrCreateLeaderboard( ISteamUserStats* self, const char * pchLeaderboardName, ELeaderboardSortMethod eLeaderboardSortMethod, ELeaderboardDisplayType eLeaderboardDisplayType );
S_API SteamAPICall_t SteamAPI_ISteamUserStats_FindLeaderboard( ISteamUserStats* self, const char * pchLeaderboardName );
S_API const char * SteamAPI_ISteamUserStats_GetLeaderboardName( ISteamUserStats* self, SteamLeaderboard_t hSteamLeaderboard );
S_API int SteamAPI_ISteamUserStats_GetLeaderboardEntryCount( ISteamUserStats* self, SteamLeaderboard_t hSteamLeaderboard );
S_API ELeaderboardSortMethod SteamAPI_ISteamUserStats_GetLeaderboardSortMethod( ISteamUserStats* self, SteamLeaderboard_t hSteamLeaderboard );
S_API ELeaderboardDisplayType SteamAPI_ISteamUserStats_GetLeaderboardDisplayType( ISteamUserStats* self, SteamLeaderboard_t hSteamLeaderboard );
S_API SteamAPICall_t SteamAPI_ISteamUserStats_DownloadLeaderboardEntries( ISteamUserStats* self, SteamLeaderboard_t hSteamLeaderboard, ELeaderboardDataRequest eLeaderboardDataRequest, int nRangeStart, int nRangeEnd );
S_API SteamAPICall_t SteamAPI_ISteamUserStats_DownloadLeaderboardEntriesForUsers( ISteamUserStats* self, SteamLeaderboard_t hSteamLeaderboard, CSteamID * prgUsers, int cUsers );
S_API bool SteamAPI_ISteamUserStats_GetDownloadedLeaderboardEntry( ISteamUserStats* self, SteamLeaderboardEntries_t hSteamLeaderboardEntries, int index, LeaderboardEntry_t * pLeaderboardEntry, int32 * pDetails, int cDetailsMax );
S_API SteamAPICall_t SteamAPI_ISteamUserStats_UploadLeaderboardScore( ISteamUserStats* self, SteamLeaderboard_t hSteamLeaderboard, ELeaderboardUploadScoreMethod eLeaderboardUploadScoreMethod, int32 nScore, const int32 * pScoreDetails, int cScoreDetailsCount );
S_API SteamAPICall_t SteamAPI_ISteamUserStats_AttachLeaderboardUGC( ISteamUserStats* self, SteamLeaderboard_t hSteamLeaderboard, UGCHandle_t hUGC );
S_API SteamAPICall_t SteamAPI_ISteamUserStats_GetNumberOfCurrentPlayers( ISteamUserStats* self );
S_API SteamAPICall_t SteamAPI_ISteamUserStats_RequestGlobalAchievementPercentages( ISteamUserStats* self );
S_API int SteamAPI_ISteamUserStats_GetMostAchievedAchievementInfo( ISteamUserStats* self, char * pchName, uint32 unNameBufLen, float * pflPercent, bool * pbAchieved );
S_API int SteamAPI_ISteamUserStats_GetNextMostAchievedAchievementInfo( ISteamUserStats* self, int iIteratorPrevious, char * pchName, uint32 unNameBufLen, float * pflPercent, bool * pbAchieved );
S_API bool SteamAPI_ISteamUserStats_GetAchievementAchievedPercent( ISteamUserStats* self, const char * pchName, float * pflPercent );
S_API SteamAPICall_t SteamAPI_ISteamUserStats_RequestGlobalStats( ISteamUserStats* self, int nHistoryDays );
S_API bool SteamAPI_ISteamUserStats_GetGlobalStatInt64( ISteamUserStats* self, const char * pchStatName, int64 * pData );
S_API bool SteamAPI_ISteamUserStats_GetGlobalStatDouble( ISteamUserStats* self, const char * pchStatName, double * pData );
S_API int32 SteamAPI_ISteamUserStats_GetGlobalStatHistoryInt64( ISteamUserStats* self, const char * pchStatName, int64 * pData, uint32 cubData );
S_API int32 SteamAPI_ISteamUserStats_GetGlobalStatHistoryDouble( ISteamUserStats* self, const char * pchStatName, double * pData, uint32 cubData );
S_API bool SteamAPI_ISteamUserStats_GetAchievementProgressLimitsInt32( ISteamUserStats* self, const char * pchName, int32 * pnMinProgress, int32 * pnMaxProgress );
S_API bool SteamAPI_ISteamUserStats_GetAchievementProgressLimitsFloat( ISteamUserStats* self, const char * pchName, float * pfMinProgress, float * pfMaxProgress );

// ISteamApps

// A versioned accessor is exported by the library
S_API ISteamApps *SteamAPI_SteamApps_v008();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamApps(), but using this ensures that you are using a matching library.
inline ISteamApps *SteamAPI_SteamApps() { return SteamAPI_SteamApps_v008(); }
S_API bool SteamAPI_ISteamApps_BIsSubscribed( ISteamApps* self );
S_API bool SteamAPI_ISteamApps_BIsLowViolence( ISteamApps* self );
S_API bool SteamAPI_ISteamApps_BIsCybercafe( ISteamApps* self );
S_API bool SteamAPI_ISteamApps_BIsVACBanned( ISteamApps* self );
S_API const char * SteamAPI_ISteamApps_GetCurrentGameLanguage( ISteamApps* self );
S_API const char * SteamAPI_ISteamApps_GetAvailableGameLanguages( ISteamApps* self );
S_API bool SteamAPI_ISteamApps_BIsSubscribedApp( ISteamApps* self, AppId_t appID );
S_API bool SteamAPI_ISteamApps_BIsDlcInstalled( ISteamApps* self, AppId_t appID );
S_API uint32 SteamAPI_ISteamApps_GetEarliestPurchaseUnixTime( ISteamApps* self, AppId_t nAppID );
S_API bool SteamAPI_ISteamApps_BIsSubscribedFromFreeWeekend( ISteamApps* self );
S_API int SteamAPI_ISteamApps_GetDLCCount( ISteamApps* self );
S_API bool SteamAPI_ISteamApps_BGetDLCDataByIndex( ISteamApps* self, int iDLC, AppId_t * pAppID, bool * pbAvailable, char * pchName, int cchNameBufferSize );
S_API void SteamAPI_ISteamApps_InstallDLC( ISteamApps* self, AppId_t nAppID );
S_API void SteamAPI_ISteamApps_UninstallDLC( ISteamApps* self, AppId_t nAppID );
S_API void SteamAPI_ISteamApps_RequestAppProofOfPurchaseKey( ISteamApps* self, AppId_t nAppID );
S_API bool SteamAPI_ISteamApps_GetCurrentBetaName( ISteamApps* self, char * pchName, int cchNameBufferSize );
S_API bool SteamAPI_ISteamApps_MarkContentCorrupt( ISteamApps* self, bool bMissingFilesOnly );
S_API uint32 SteamAPI_ISteamApps_GetInstalledDepots( ISteamApps* self, AppId_t appID, DepotId_t * pvecDepots, uint32 cMaxDepots );
S_API uint32 SteamAPI_ISteamApps_GetAppInstallDir( ISteamApps* self, AppId_t appID, char * pchFolder, uint32 cchFolderBufferSize );
S_API bool SteamAPI_ISteamApps_BIsAppInstalled( ISteamApps* self, AppId_t appID );
S_API uint64_steamid SteamAPI_ISteamApps_GetAppOwner( ISteamApps* self );
S_API const char * SteamAPI_ISteamApps_GetLaunchQueryParam( ISteamApps* self, const char * pchKey );
S_API bool SteamAPI_ISteamApps_GetDlcDownloadProgress( ISteamApps* self, AppId_t nAppID, uint64 * punBytesDownloaded, uint64 * punBytesTotal );
S_API int SteamAPI_ISteamApps_GetAppBuildId( ISteamApps* self );
S_API void SteamAPI_ISteamApps_RequestAllProofOfPurchaseKeys( ISteamApps* self );
S_API SteamAPICall_t SteamAPI_ISteamApps_GetFileDetails( ISteamApps* self, const char * pszFileName );
S_API int SteamAPI_ISteamApps_GetLaunchCommandLine( ISteamApps* self, char * pszCommandLine, int cubCommandLine );
S_API bool SteamAPI_ISteamApps_BIsSubscribedFromFamilySharing( ISteamApps* self );
S_API bool SteamAPI_ISteamApps_BIsTimedTrial( ISteamApps* self, uint32 * punSecondsAllowed, uint32 * punSecondsPlayed );

// ISteamNetworking

// A versioned accessor is exported by the library
S_API ISteamNetworking *SteamAPI_SteamNetworking_v006();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamNetworking(), but using this ensures that you are using a matching library.
inline ISteamNetworking *SteamAPI_SteamNetworking() { return SteamAPI_SteamNetworking_v006(); }

// A versioned accessor is exported by the library
S_API ISteamNetworking *SteamAPI_SteamGameServerNetworking_v006();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamGameServerNetworking(), but using this ensures that you are using a matching library.
inline ISteamNetworking *SteamAPI_SteamGameServerNetworking() { return SteamAPI_SteamGameServerNetworking_v006(); }
S_API bool SteamAPI_ISteamNetworking_SendP2PPacket( ISteamNetworking* self, uint64_steamid steamIDRemote, const void * pubData, uint32 cubData, EP2PSend eP2PSendType, int nChannel );
S_API bool SteamAPI_ISteamNetworking_IsP2PPacketAvailable( ISteamNetworking* self, uint32 * pcubMsgSize, int nChannel );
S_API bool SteamAPI_ISteamNetworking_ReadP2PPacket( ISteamNetworking* self, void * pubDest, uint32 cubDest, uint32 * pcubMsgSize, CSteamID * psteamIDRemote, int nChannel );
S_API bool SteamAPI_ISteamNetworking_AcceptP2PSessionWithUser( ISteamNetworking* self, uint64_steamid steamIDRemote );
S_API bool SteamAPI_ISteamNetworking_CloseP2PSessionWithUser( ISteamNetworking* self, uint64_steamid steamIDRemote );
S_API bool SteamAPI_ISteamNetworking_CloseP2PChannelWithUser( ISteamNetworking* self, uint64_steamid steamIDRemote, int nChannel );
S_API bool SteamAPI_ISteamNetworking_GetP2PSessionState( ISteamNetworking* self, uint64_steamid steamIDRemote, P2PSessionState_t * pConnectionState );
S_API bool SteamAPI_ISteamNetworking_AllowP2PPacketRelay( ISteamNetworking* self, bool bAllow );
S_API SNetListenSocket_t SteamAPI_ISteamNetworking_CreateListenSocket( ISteamNetworking* self, int nVirtualP2PPort, SteamIPAddress_t nIP, uint16 nPort, bool bAllowUseOfPacketRelay );
S_API SNetSocket_t SteamAPI_ISteamNetworking_CreateP2PConnectionSocket( ISteamNetworking* self, uint64_steamid steamIDTarget, int nVirtualPort, int nTimeoutSec, bool bAllowUseOfPacketRelay );
S_API SNetSocket_t SteamAPI_ISteamNetworking_CreateConnectionSocket( ISteamNetworking* self, SteamIPAddress_t nIP, uint16 nPort, int nTimeoutSec );
S_API bool SteamAPI_ISteamNetworking_DestroySocket( ISteamNetworking* self, SNetSocket_t hSocket, bool bNotifyRemoteEnd );
S_API bool SteamAPI_ISteamNetworking_DestroyListenSocket( ISteamNetworking* self, SNetListenSocket_t hSocket, bool bNotifyRemoteEnd );
S_API bool SteamAPI_ISteamNetworking_SendDataOnSocket( ISteamNetworking* self, SNetSocket_t hSocket, void * pubData, uint32 cubData, bool bReliable );
S_API bool SteamAPI_ISteamNetworking_IsDataAvailableOnSocket( ISteamNetworking* self, SNetSocket_t hSocket, uint32 * pcubMsgSize );
S_API bool SteamAPI_ISteamNetworking_RetrieveDataFromSocket( ISteamNetworking* self, SNetSocket_t hSocket, void * pubDest, uint32 cubDest, uint32 * pcubMsgSize );
S_API bool SteamAPI_ISteamNetworking_IsDataAvailable( ISteamNetworking* self, SNetListenSocket_t hListenSocket, uint32 * pcubMsgSize, SNetSocket_t * phSocket );
S_API bool SteamAPI_ISteamNetworking_RetrieveData( ISteamNetworking* self, SNetListenSocket_t hListenSocket, void * pubDest, uint32 cubDest, uint32 * pcubMsgSize, SNetSocket_t * phSocket );
S_API bool SteamAPI_ISteamNetworking_GetSocketInfo( ISteamNetworking* self, SNetSocket_t hSocket, CSteamID * pSteamIDRemote, int * peSocketStatus, SteamIPAddress_t * punIPRemote, uint16 * punPortRemote );
S_API bool SteamAPI_ISteamNetworking_GetListenSocketInfo( ISteamNetworking* self, SNetListenSocket_t hListenSocket, SteamIPAddress_t * pnIP, uint16 * pnPort );
S_API ESNetSocketConnectionType SteamAPI_ISteamNetworking_GetSocketConnectionType( ISteamNetworking* self, SNetSocket_t hSocket );
S_API int SteamAPI_ISteamNetworking_GetMaxPacketSize( ISteamNetworking* self, SNetSocket_t hSocket );

// ISteamScreenshots

// A versioned accessor is exported by the library
S_API ISteamScreenshots *SteamAPI_SteamScreenshots_v003();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamScreenshots(), but using this ensures that you are using a matching library.
inline ISteamScreenshots *SteamAPI_SteamScreenshots() { return SteamAPI_SteamScreenshots_v003(); }
S_API ScreenshotHandle SteamAPI_ISteamScreenshots_WriteScreenshot( ISteamScreenshots* self, void * pubRGB, uint32 cubRGB, int nWidth, int nHeight );
S_API ScreenshotHandle SteamAPI_ISteamScreenshots_AddScreenshotToLibrary( ISteamScreenshots* self, const char * pchFilename, const char * pchThumbnailFilename, int nWidth, int nHeight );
S_API void SteamAPI_ISteamScreenshots_TriggerScreenshot( ISteamScreenshots* self );
S_API void SteamAPI_ISteamScreenshots_HookScreenshots( ISteamScreenshots* self, bool bHook );
S_API bool SteamAPI_ISteamScreenshots_SetLocation( ISteamScreenshots* self, ScreenshotHandle hScreenshot, const char * pchLocation );
S_API bool SteamAPI_ISteamScreenshots_TagUser( ISteamScreenshots* self, ScreenshotHandle hScreenshot, uint64_steamid steamID );
S_API bool SteamAPI_ISteamScreenshots_TagPublishedFile( ISteamScreenshots* self, ScreenshotHandle hScreenshot, PublishedFileId_t unPublishedFileID );
S_API bool SteamAPI_ISteamScreenshots_IsScreenshotsHooked( ISteamScreenshots* self );
S_API ScreenshotHandle SteamAPI_ISteamScreenshots_AddVRScreenshotToLibrary( ISteamScreenshots* self, EVRScreenshotType eType, const char * pchFilename, const char * pchVRFilename );

// ISteamMusic

// A versioned accessor is exported by the library
S_API ISteamMusic *SteamAPI_SteamMusic_v001();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamMusic(), but using this ensures that you are using a matching library.
inline ISteamMusic *SteamAPI_SteamMusic() { return SteamAPI_SteamMusic_v001(); }
S_API bool SteamAPI_ISteamMusic_BIsEnabled( ISteamMusic* self );
S_API bool SteamAPI_ISteamMusic_BIsPlaying( ISteamMusic* self );
S_API AudioPlayback_Status SteamAPI_ISteamMusic_GetPlaybackStatus( ISteamMusic* self );
S_API void SteamAPI_ISteamMusic_Play( ISteamMusic* self );
S_API void SteamAPI_ISteamMusic_Pause( ISteamMusic* self );
S_API void SteamAPI_ISteamMusic_PlayPrevious( ISteamMusic* self );
S_API void SteamAPI_ISteamMusic_PlayNext( ISteamMusic* self );
S_API void SteamAPI_ISteamMusic_SetVolume( ISteamMusic* self, float flVolume );
S_API float SteamAPI_ISteamMusic_GetVolume( ISteamMusic* self );

// ISteamMusicRemote

// A versioned accessor is exported by the library
S_API ISteamMusicRemote *SteamAPI_SteamMusicRemote_v001();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamMusicRemote(), but using this ensures that you are using a matching library.
inline ISteamMusicRemote *SteamAPI_SteamMusicRemote() { return SteamAPI_SteamMusicRemote_v001(); }
S_API bool SteamAPI_ISteamMusicRemote_RegisterSteamMusicRemote( ISteamMusicRemote* self, const char * pchName );
S_API bool SteamAPI_ISteamMusicRemote_DeregisterSteamMusicRemote( ISteamMusicRemote* self );
S_API bool SteamAPI_ISteamMusicRemote_BIsCurrentMusicRemote( ISteamMusicRemote* self );
S_API bool SteamAPI_ISteamMusicRemote_BActivationSuccess( ISteamMusicRemote* self, bool bValue );
S_API bool SteamAPI_ISteamMusicRemote_SetDisplayName( ISteamMusicRemote* self, const char * pchDisplayName );
S_API bool SteamAPI_ISteamMusicRemote_SetPNGIcon_64x64( ISteamMusicRemote* self, void * pvBuffer, uint32 cbBufferLength );
S_API bool SteamAPI_ISteamMusicRemote_EnablePlayPrevious( ISteamMusicRemote* self, bool bValue );
S_API bool SteamAPI_ISteamMusicRemote_EnablePlayNext( ISteamMusicRemote* self, bool bValue );
S_API bool SteamAPI_ISteamMusicRemote_EnableShuffled( ISteamMusicRemote* self, bool bValue );
S_API bool SteamAPI_ISteamMusicRemote_EnableLooped( ISteamMusicRemote* self, bool bValue );
S_API bool SteamAPI_ISteamMusicRemote_EnableQueue( ISteamMusicRemote* self, bool bValue );
S_API bool SteamAPI_ISteamMusicRemote_EnablePlaylists( ISteamMusicRemote* self, bool bValue );
S_API bool SteamAPI_ISteamMusicRemote_UpdatePlaybackStatus( ISteamMusicRemote* self, AudioPlayback_Status nStatus );
S_API bool SteamAPI_ISteamMusicRemote_UpdateShuffled( ISteamMusicRemote* self, bool bValue );
S_API bool SteamAPI_ISteamMusicRemote_UpdateLooped( ISteamMusicRemote* self, bool bValue );
S_API bool SteamAPI_ISteamMusicRemote_UpdateVolume( ISteamMusicRemote* self, float flValue );
S_API bool SteamAPI_ISteamMusicRemote_CurrentEntryWillChange( ISteamMusicRemote* self );
S_API bool SteamAPI_ISteamMusicRemote_CurrentEntryIsAvailable( ISteamMusicRemote* self, bool bAvailable );
S_API bool SteamAPI_ISteamMusicRemote_UpdateCurrentEntryText( ISteamMusicRemote* self, const char * pchText );
S_API bool SteamAPI_ISteamMusicRemote_UpdateCurrentEntryElapsedSeconds( ISteamMusicRemote* self, int nValue );
S_API bool SteamAPI_ISteamMusicRemote_UpdateCurrentEntryCoverArt( ISteamMusicRemote* self, void * pvBuffer, uint32 cbBufferLength );
S_API bool SteamAPI_ISteamMusicRemote_CurrentEntryDidChange( ISteamMusicRemote* self );
S_API bool SteamAPI_ISteamMusicRemote_QueueWillChange( ISteamMusicRemote* self );
S_API bool SteamAPI_ISteamMusicRemote_ResetQueueEntries( ISteamMusicRemote* self );
S_API bool SteamAPI_ISteamMusicRemote_SetQueueEntry( ISteamMusicRemote* self, int nID, int nPosition, const char * pchEntryText );
S_API bool SteamAPI_ISteamMusicRemote_SetCurrentQueueEntry( ISteamMusicRemote* self, int nID );
S_API bool SteamAPI_ISteamMusicRemote_QueueDidChange( ISteamMusicRemote* self );
S_API bool SteamAPI_ISteamMusicRemote_PlaylistWillChange( ISteamMusicRemote* self );
S_API bool SteamAPI_ISteamMusicRemote_ResetPlaylistEntries( ISteamMusicRemote* self );
S_API bool SteamAPI_ISteamMusicRemote_SetPlaylistEntry( ISteamMusicRemote* self, int nID, int nPosition, const char * pchEntryText );
S_API bool SteamAPI_ISteamMusicRemote_SetCurrentPlaylistEntry( ISteamMusicRemote* self, int nID );
S_API bool SteamAPI_ISteamMusicRemote_PlaylistDidChange( ISteamMusicRemote* self );

// ISteamHTTP

// A versioned accessor is exported by the library
S_API ISteamHTTP *SteamAPI_SteamHTTP_v003();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamHTTP(), but using this ensures that you are using a matching library.
inline ISteamHTTP *SteamAPI_SteamHTTP() { return SteamAPI_SteamHTTP_v003(); }

// A versioned accessor is exported by the library
S_API ISteamHTTP *SteamAPI_SteamGameServerHTTP_v003();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamGameServerHTTP(), but using this ensures that you are using a matching library.
inline ISteamHTTP *SteamAPI_SteamGameServerHTTP() { return SteamAPI_SteamGameServerHTTP_v003(); }
S_API HTTPRequestHandle SteamAPI_ISteamHTTP_CreateHTTPRequest( ISteamHTTP* self, EHTTPMethod eHTTPRequestMethod, const char * pchAbsoluteURL );
S_API bool SteamAPI_ISteamHTTP_SetHTTPRequestContextValue( ISteamHTTP* self, HTTPRequestHandle hRequest, uint64 ulContextValue );
S_API bool SteamAPI_ISteamHTTP_SetHTTPRequestNetworkActivityTimeout( ISteamHTTP* self, HTTPRequestHandle hRequest, uint32 unTimeoutSeconds );
S_API bool SteamAPI_ISteamHTTP_SetHTTPRequestHeaderValue( ISteamHTTP* self, HTTPRequestHandle hRequest, const char * pchHeaderName, const char * pchHeaderValue );
S_API bool SteamAPI_ISteamHTTP_SetHTTPRequestGetOrPostParameter( ISteamHTTP* self, HTTPRequestHandle hRequest, const char * pchParamName, const char * pchParamValue );
S_API bool SteamAPI_ISteamHTTP_SendHTTPRequest( ISteamHTTP* self, HTTPRequestHandle hRequest, SteamAPICall_t * pCallHandle );
S_API bool SteamAPI_ISteamHTTP_SendHTTPRequestAndStreamResponse( ISteamHTTP* self, HTTPRequestHandle hRequest, SteamAPICall_t * pCallHandle );
S_API bool SteamAPI_ISteamHTTP_DeferHTTPRequest( ISteamHTTP* self, HTTPRequestHandle hRequest );
S_API bool SteamAPI_ISteamHTTP_PrioritizeHTTPRequest( ISteamHTTP* self, HTTPRequestHandle hRequest );
S_API bool SteamAPI_ISteamHTTP_GetHTTPResponseHeaderSize( ISteamHTTP* self, HTTPRequestHandle hRequest, const char * pchHeaderName, uint32 * unResponseHeaderSize );
S_API bool SteamAPI_ISteamHTTP_GetHTTPResponseHeaderValue( ISteamHTTP* self, HTTPRequestHandle hRequest, const char * pchHeaderName, uint8 * pHeaderValueBuffer, uint32 unBufferSize );
S_API bool SteamAPI_ISteamHTTP_GetHTTPResponseBodySize( ISteamHTTP* self, HTTPRequestHandle hRequest, uint32 * unBodySize );
S_API bool SteamAPI_ISteamHTTP_GetHTTPResponseBodyData( ISteamHTTP* self, HTTPRequestHandle hRequest, uint8 * pBodyDataBuffer, uint32 unBufferSize );
S_API bool SteamAPI_ISteamHTTP_GetHTTPStreamingResponseBodyData( ISteamHTTP* self, HTTPRequestHandle hRequest, uint32 cOffset, uint8 * pBodyDataBuffer, uint32 unBufferSize );
S_API bool SteamAPI_ISteamHTTP_ReleaseHTTPRequest( ISteamHTTP* self, HTTPRequestHandle hRequest );
S_API bool SteamAPI_ISteamHTTP_GetHTTPDownloadProgressPct( ISteamHTTP* self, HTTPRequestHandle hRequest, float * pflPercentOut );
S_API bool SteamAPI_ISteamHTTP_SetHTTPRequestRawPostBody( ISteamHTTP* self, HTTPRequestHandle hRequest, const char * pchContentType, uint8 * pubBody, uint32 unBodyLen );
S_API HTTPCookieContainerHandle SteamAPI_ISteamHTTP_CreateCookieContainer( ISteamHTTP* self, bool bAllowResponsesToModify );
S_API bool SteamAPI_ISteamHTTP_ReleaseCookieContainer( ISteamHTTP* self, HTTPCookieContainerHandle hCookieContainer );
S_API bool SteamAPI_ISteamHTTP_SetCookie( ISteamHTTP* self, HTTPCookieContainerHandle hCookieContainer, const char * pchHost, const char * pchUrl, const char * pchCookie );
S_API bool SteamAPI_ISteamHTTP_SetHTTPRequestCookieContainer( ISteamHTTP* self, HTTPRequestHandle hRequest, HTTPCookieContainerHandle hCookieContainer );
S_API bool SteamAPI_ISteamHTTP_SetHTTPRequestUserAgentInfo( ISteamHTTP* self, HTTPRequestHandle hRequest, const char * pchUserAgentInfo );
S_API bool SteamAPI_ISteamHTTP_SetHTTPRequestRequiresVerifiedCertificate( ISteamHTTP* self, HTTPRequestHandle hRequest, bool bRequireVerifiedCertificate );
S_API bool SteamAPI_ISteamHTTP_SetHTTPRequestAbsoluteTimeoutMS( ISteamHTTP* self, HTTPRequestHandle hRequest, uint32 unMilliseconds );
S_API bool SteamAPI_ISteamHTTP_GetHTTPRequestWasTimedOut( ISteamHTTP* self, HTTPRequestHandle hRequest, bool * pbWasTimedOut );

// ISteamInput

// A versioned accessor is exported by the library
S_API ISteamInput *SteamAPI_SteamInput_v006();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamInput(), but using this ensures that you are using a matching library.
inline ISteamInput *SteamAPI_SteamInput() { return SteamAPI_SteamInput_v006(); }
S_API bool SteamAPI_ISteamInput_Init( ISteamInput* self, bool bExplicitlyCallRunFrame );
S_API bool SteamAPI_ISteamInput_Shutdown( ISteamInput* self );
S_API bool SteamAPI_ISteamInput_SetInputActionManifestFilePath( ISteamInput* self, const char * pchInputActionManifestAbsolutePath );
S_API void SteamAPI_ISteamInput_RunFrame( ISteamInput* self, bool bReservedValue );
S_API bool SteamAPI_ISteamInput_BWaitForData( ISteamInput* self, bool bWaitForever, uint32 unTimeout );
S_API bool SteamAPI_ISteamInput_BNewDataAvailable( ISteamInput* self );
S_API int SteamAPI_ISteamInput_GetConnectedControllers( ISteamInput* self, InputHandle_t * handlesOut );
S_API void SteamAPI_ISteamInput_EnableDeviceCallbacks( ISteamInput* self );
S_API void SteamAPI_ISteamInput_EnableActionEventCallbacks( ISteamInput* self, SteamInputActionEventCallbackPointer pCallback );
S_API InputActionSetHandle_t SteamAPI_ISteamInput_GetActionSetHandle( ISteamInput* self, const char * pszActionSetName );
S_API void SteamAPI_ISteamInput_ActivateActionSet( ISteamInput* self, InputHandle_t inputHandle, InputActionSetHandle_t actionSetHandle );
S_API InputActionSetHandle_t SteamAPI_ISteamInput_GetCurrentActionSet( ISteamInput* self, InputHandle_t inputHandle );
S_API void SteamAPI_ISteamInput_ActivateActionSetLayer( ISteamInput* self, InputHandle_t inputHandle, InputActionSetHandle_t actionSetLayerHandle );
S_API void SteamAPI_ISteamInput_DeactivateActionSetLayer( ISteamInput* self, InputHandle_t inputHandle, InputActionSetHandle_t actionSetLayerHandle );
S_API void SteamAPI_ISteamInput_DeactivateAllActionSetLayers( ISteamInput* self, InputHandle_t inputHandle );
S_API int SteamAPI_ISteamInput_GetActiveActionSetLayers( ISteamInput* self, InputHandle_t inputHandle, InputActionSetHandle_t * handlesOut );
S_API InputDigitalActionHandle_t SteamAPI_ISteamInput_GetDigitalActionHandle( ISteamInput* self, const char * pszActionName );
S_API InputDigitalActionData_t SteamAPI_ISteamInput_GetDigitalActionData( ISteamInput* self, InputHandle_t inputHandle, InputDigitalActionHandle_t digitalActionHandle );
S_API int SteamAPI_ISteamInput_GetDigitalActionOrigins( ISteamInput* self, InputHandle_t inputHandle, InputActionSetHandle_t actionSetHandle, InputDigitalActionHandle_t digitalActionHandle, EInputActionOrigin * originsOut );
S_API const char * SteamAPI_ISteamInput_GetStringForDigitalActionName( ISteamInput* self, InputDigitalActionHandle_t eActionHandle );
S_API InputAnalogActionHandle_t SteamAPI_ISteamInput_GetAnalogActionHandle( ISteamInput* self, const char * pszActionName );
S_API InputAnalogActionData_t SteamAPI_ISteamInput_GetAnalogActionData( ISteamInput* self, InputHandle_t inputHandle, InputAnalogActionHandle_t analogActionHandle );
S_API int SteamAPI_ISteamInput_GetAnalogActionOrigins( ISteamInput* self, InputHandle_t inputHandle, InputActionSetHandle_t actionSetHandle, InputAnalogActionHandle_t analogActionHandle, EInputActionOrigin * originsOut );
S_API const char * SteamAPI_ISteamInput_GetGlyphPNGForActionOrigin( ISteamInput* self, EInputActionOrigin eOrigin, ESteamInputGlyphSize eSize, uint32 unFlags );
S_API const char * SteamAPI_ISteamInput_GetGlyphSVGForActionOrigin( ISteamInput* self, EInputActionOrigin eOrigin, uint32 unFlags );
S_API const char * SteamAPI_ISteamInput_GetGlyphForActionOrigin_Legacy( ISteamInput* self, EInputActionOrigin eOrigin );
S_API const char * SteamAPI_ISteamInput_GetStringForActionOrigin( ISteamInput* self, EInputActionOrigin eOrigin );
S_API const char * SteamAPI_ISteamInput_GetStringForAnalogActionName( ISteamInput* self, InputAnalogActionHandle_t eActionHandle );
S_API void SteamAPI_ISteamInput_StopAnalogActionMomentum( ISteamInput* self, InputHandle_t inputHandle, InputAnalogActionHandle_t eAction );
S_API InputMotionData_t SteamAPI_ISteamInput_GetMotionData( ISteamInput* self, InputHandle_t inputHandle );
S_API void SteamAPI_ISteamInput_TriggerVibration( ISteamInput* self, InputHandle_t inputHandle, unsigned short usLeftSpeed, unsigned short usRightSpeed );
S_API void SteamAPI_ISteamInput_TriggerVibrationExtended( ISteamInput* self, InputHandle_t inputHandle, unsigned short usLeftSpeed, unsigned short usRightSpeed, unsigned short usLeftTriggerSpeed, unsigned short usRightTriggerSpeed );
S_API void SteamAPI_ISteamInput_TriggerSimpleHapticEvent( ISteamInput* self, InputHandle_t inputHandle, EControllerHapticLocation eHapticLocation, uint8 nIntensity, char nGainDB, uint8 nOtherIntensity, char nOtherGainDB );
S_API void SteamAPI_ISteamInput_SetLEDColor( ISteamInput* self, InputHandle_t inputHandle, uint8 nColorR, uint8 nColorG, uint8 nColorB, unsigned int nFlags );
S_API void SteamAPI_ISteamInput_Legacy_TriggerHapticPulse( ISteamInput* self, InputHandle_t inputHandle, ESteamControllerPad eTargetPad, unsigned short usDurationMicroSec );
S_API void SteamAPI_ISteamInput_Legacy_TriggerRepeatedHapticPulse( ISteamInput* self, InputHandle_t inputHandle, ESteamControllerPad eTargetPad, unsigned short usDurationMicroSec, unsigned short usOffMicroSec, unsigned short unRepeat, unsigned int nFlags );
S_API bool SteamAPI_ISteamInput_ShowBindingPanel( ISteamInput* self, InputHandle_t inputHandle );
S_API ESteamInputType SteamAPI_ISteamInput_GetInputTypeForHandle( ISteamInput* self, InputHandle_t inputHandle );
S_API InputHandle_t SteamAPI_ISteamInput_GetControllerForGamepadIndex( ISteamInput* self, int nIndex );
S_API int SteamAPI_ISteamInput_GetGamepadIndexForController( ISteamInput* self, InputHandle_t ulinputHandle );
S_API const char * SteamAPI_ISteamInput_GetStringForXboxOrigin( ISteamInput* self, EXboxOrigin eOrigin );
S_API const char * SteamAPI_ISteamInput_GetGlyphForXboxOrigin( ISteamInput* self, EXboxOrigin eOrigin );
S_API EInputActionOrigin SteamAPI_ISteamInput_GetActionOriginFromXboxOrigin( ISteamInput* self, InputHandle_t inputHandle, EXboxOrigin eOrigin );
S_API EInputActionOrigin SteamAPI_ISteamInput_TranslateActionOrigin( ISteamInput* self, ESteamInputType eDestinationInputType, EInputActionOrigin eSourceOrigin );
S_API bool SteamAPI_ISteamInput_GetDeviceBindingRevision( ISteamInput* self, InputHandle_t inputHandle, int * pMajor, int * pMinor );
S_API uint32 SteamAPI_ISteamInput_GetRemotePlaySessionID( ISteamInput* self, InputHandle_t inputHandle );
S_API uint16 SteamAPI_ISteamInput_GetSessionInputConfigurationSettings( ISteamInput* self );

// ISteamController

// A versioned accessor is exported by the library
S_API ISteamController *SteamAPI_SteamController_v008();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamController(), but using this ensures that you are using a matching library.
inline ISteamController *SteamAPI_SteamController() { return SteamAPI_SteamController_v008(); }
S_API bool SteamAPI_ISteamController_Init( ISteamController* self );
S_API bool SteamAPI_ISteamController_Shutdown( ISteamController* self );
S_API void SteamAPI_ISteamController_RunFrame( ISteamController* self );
S_API int SteamAPI_ISteamController_GetConnectedControllers( ISteamController* self, ControllerHandle_t * handlesOut );
S_API ControllerActionSetHandle_t SteamAPI_ISteamController_GetActionSetHandle( ISteamController* self, const char * pszActionSetName );
S_API void SteamAPI_ISteamController_ActivateActionSet( ISteamController* self, ControllerHandle_t controllerHandle, ControllerActionSetHandle_t actionSetHandle );
S_API ControllerActionSetHandle_t SteamAPI_ISteamController_GetCurrentActionSet( ISteamController* self, ControllerHandle_t controllerHandle );
S_API void SteamAPI_ISteamController_ActivateActionSetLayer( ISteamController* self, ControllerHandle_t controllerHandle, ControllerActionSetHandle_t actionSetLayerHandle );
S_API void SteamAPI_ISteamController_DeactivateActionSetLayer( ISteamController* self, ControllerHandle_t controllerHandle, ControllerActionSetHandle_t actionSetLayerHandle );
S_API void SteamAPI_ISteamController_DeactivateAllActionSetLayers( ISteamController* self, ControllerHandle_t controllerHandle );
S_API int SteamAPI_ISteamController_GetActiveActionSetLayers( ISteamController* self, ControllerHandle_t controllerHandle, ControllerActionSetHandle_t * handlesOut );
S_API ControllerDigitalActionHandle_t SteamAPI_ISteamController_GetDigitalActionHandle( ISteamController* self, const char * pszActionName );
S_API InputDigitalActionData_t SteamAPI_ISteamController_GetDigitalActionData( ISteamController* self, ControllerHandle_t controllerHandle, ControllerDigitalActionHandle_t digitalActionHandle );
S_API int SteamAPI_ISteamController_GetDigitalActionOrigins( ISteamController* self, ControllerHandle_t controllerHandle, ControllerActionSetHandle_t actionSetHandle, ControllerDigitalActionHandle_t digitalActionHandle, EControllerActionOrigin * originsOut );
S_API ControllerAnalogActionHandle_t SteamAPI_ISteamController_GetAnalogActionHandle( ISteamController* self, const char * pszActionName );
S_API InputAnalogActionData_t SteamAPI_ISteamController_GetAnalogActionData( ISteamController* self, ControllerHandle_t controllerHandle, ControllerAnalogActionHandle_t analogActionHandle );
S_API int SteamAPI_ISteamController_GetAnalogActionOrigins( ISteamController* self, ControllerHandle_t controllerHandle, ControllerActionSetHandle_t actionSetHandle, ControllerAnalogActionHandle_t analogActionHandle, EControllerActionOrigin * originsOut );
S_API const char * SteamAPI_ISteamController_GetGlyphForActionOrigin( ISteamController* self, EControllerActionOrigin eOrigin );
S_API const char * SteamAPI_ISteamController_GetStringForActionOrigin( ISteamController* self, EControllerActionOrigin eOrigin );
S_API void SteamAPI_ISteamController_StopAnalogActionMomentum( ISteamController* self, ControllerHandle_t controllerHandle, ControllerAnalogActionHandle_t eAction );
S_API InputMotionData_t SteamAPI_ISteamController_GetMotionData( ISteamController* self, ControllerHandle_t controllerHandle );
S_API void SteamAPI_ISteamController_TriggerHapticPulse( ISteamController* self, ControllerHandle_t controllerHandle, ESteamControllerPad eTargetPad, unsigned short usDurationMicroSec );
S_API void SteamAPI_ISteamController_TriggerRepeatedHapticPulse( ISteamController* self, ControllerHandle_t controllerHandle, ESteamControllerPad eTargetPad, unsigned short usDurationMicroSec, unsigned short usOffMicroSec, unsigned short unRepeat, unsigned int nFlags );
S_API void SteamAPI_ISteamController_TriggerVibration( ISteamController* self, ControllerHandle_t controllerHandle, unsigned short usLeftSpeed, unsigned short usRightSpeed );
S_API void SteamAPI_ISteamController_SetLEDColor( ISteamController* self, ControllerHandle_t controllerHandle, uint8 nColorR, uint8 nColorG, uint8 nColorB, unsigned int nFlags );
S_API bool SteamAPI_ISteamController_ShowBindingPanel( ISteamController* self, ControllerHandle_t controllerHandle );
S_API ESteamInputType SteamAPI_ISteamController_GetInputTypeForHandle( ISteamController* self, ControllerHandle_t controllerHandle );
S_API ControllerHandle_t SteamAPI_ISteamController_GetControllerForGamepadIndex( ISteamController* self, int nIndex );
S_API int SteamAPI_ISteamController_GetGamepadIndexForController( ISteamController* self, ControllerHandle_t ulControllerHandle );
S_API const char * SteamAPI_ISteamController_GetStringForXboxOrigin( ISteamController* self, EXboxOrigin eOrigin );
S_API const char * SteamAPI_ISteamController_GetGlyphForXboxOrigin( ISteamController* self, EXboxOrigin eOrigin );
S_API EControllerActionOrigin SteamAPI_ISteamController_GetActionOriginFromXboxOrigin( ISteamController* self, ControllerHandle_t controllerHandle, EXboxOrigin eOrigin );
S_API EControllerActionOrigin SteamAPI_ISteamController_TranslateActionOrigin( ISteamController* self, ESteamInputType eDestinationInputType, EControllerActionOrigin eSourceOrigin );
S_API bool SteamAPI_ISteamController_GetControllerBindingRevision( ISteamController* self, ControllerHandle_t controllerHandle, int * pMajor, int * pMinor );

// ISteamUGC

// A versioned accessor is exported by the library
S_API ISteamUGC *SteamAPI_SteamUGC_v016();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamUGC(), but using this ensures that you are using a matching library.
inline ISteamUGC *SteamAPI_SteamUGC() { return SteamAPI_SteamUGC_v016(); }

// A versioned accessor is exported by the library
S_API ISteamUGC *SteamAPI_SteamGameServerUGC_v016();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamGameServerUGC(), but using this ensures that you are using a matching library.
inline ISteamUGC *SteamAPI_SteamGameServerUGC() { return SteamAPI_SteamGameServerUGC_v016(); }
S_API UGCQueryHandle_t SteamAPI_ISteamUGC_CreateQueryUserUGCRequest( ISteamUGC* self, AccountID_t unAccountID, EUserUGCList eListType, EUGCMatchingUGCType eMatchingUGCType, EUserUGCListSortOrder eSortOrder, AppId_t nCreatorAppID, AppId_t nConsumerAppID, uint32 unPage );
S_API UGCQueryHandle_t SteamAPI_ISteamUGC_CreateQueryAllUGCRequestPage( ISteamUGC* self, EUGCQuery eQueryType, EUGCMatchingUGCType eMatchingeMatchingUGCTypeFileType, AppId_t nCreatorAppID, AppId_t nConsumerAppID, uint32 unPage );
S_API UGCQueryHandle_t SteamAPI_ISteamUGC_CreateQueryAllUGCRequestCursor( ISteamUGC* self, EUGCQuery eQueryType, EUGCMatchingUGCType eMatchingeMatchingUGCTypeFileType, AppId_t nCreatorAppID, AppId_t nConsumerAppID, const char * pchCursor );
S_API UGCQueryHandle_t SteamAPI_ISteamUGC_CreateQueryUGCDetailsRequest( ISteamUGC* self, PublishedFileId_t * pvecPublishedFileID, uint32 unNumPublishedFileIDs );
S_API SteamAPICall_t SteamAPI_ISteamUGC_SendQueryUGCRequest( ISteamUGC* self, UGCQueryHandle_t handle );
S_API bool SteamAPI_ISteamUGC_GetQueryUGCResult( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index, SteamUGCDetails_t * pDetails );
S_API uint32 SteamAPI_ISteamUGC_GetQueryUGCNumTags( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index );
S_API bool SteamAPI_ISteamUGC_GetQueryUGCTag( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index, uint32 indexTag, char * pchValue, uint32 cchValueSize );
S_API bool SteamAPI_ISteamUGC_GetQueryUGCTagDisplayName( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index, uint32 indexTag, char * pchValue, uint32 cchValueSize );
S_API bool SteamAPI_ISteamUGC_GetQueryUGCPreviewURL( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index, char * pchURL, uint32 cchURLSize );
S_API bool SteamAPI_ISteamUGC_GetQueryUGCMetadata( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index, char * pchMetadata, uint32 cchMetadatasize );
S_API bool SteamAPI_ISteamUGC_GetQueryUGCChildren( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index, PublishedFileId_t * pvecPublishedFileID, uint32 cMaxEntries );
S_API bool SteamAPI_ISteamUGC_GetQueryUGCStatistic( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index, EItemStatistic eStatType, uint64 * pStatValue );
S_API uint32 SteamAPI_ISteamUGC_GetQueryUGCNumAdditionalPreviews( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index );
S_API bool SteamAPI_ISteamUGC_GetQueryUGCAdditionalPreview( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index, uint32 previewIndex, char * pchURLOrVideoID, uint32 cchURLSize, char * pchOriginalFileName, uint32 cchOriginalFileNameSize, EItemPreviewType * pPreviewType );
S_API uint32 SteamAPI_ISteamUGC_GetQueryUGCNumKeyValueTags( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index );
S_API bool SteamAPI_ISteamUGC_GetQueryUGCKeyValueTag( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index, uint32 keyValueTagIndex, char * pchKey, uint32 cchKeySize, char * pchValue, uint32 cchValueSize );
S_API bool SteamAPI_ISteamUGC_GetQueryFirstUGCKeyValueTag( ISteamUGC* self, UGCQueryHandle_t handle, uint32 index, const char * pchKey, char * pchValue, uint32 cchValueSize );
S_API bool SteamAPI_ISteamUGC_ReleaseQueryUGCRequest( ISteamUGC* self, UGCQueryHandle_t handle );
S_API bool SteamAPI_ISteamUGC_AddRequiredTag( ISteamUGC* self, UGCQueryHandle_t handle, const char * pTagName );
S_API bool SteamAPI_ISteamUGC_AddRequiredTagGroup( ISteamUGC* self, UGCQueryHandle_t handle, const SteamParamStringArray_t * pTagGroups );
S_API bool SteamAPI_ISteamUGC_AddExcludedTag( ISteamUGC* self, UGCQueryHandle_t handle, const char * pTagName );
S_API bool SteamAPI_ISteamUGC_SetReturnOnlyIDs( ISteamUGC* self, UGCQueryHandle_t handle, bool bReturnOnlyIDs );
S_API bool SteamAPI_ISteamUGC_SetReturnKeyValueTags( ISteamUGC* self, UGCQueryHandle_t handle, bool bReturnKeyValueTags );
S_API bool SteamAPI_ISteamUGC_SetReturnLongDescription( ISteamUGC* self, UGCQueryHandle_t handle, bool bReturnLongDescription );
S_API bool SteamAPI_ISteamUGC_SetReturnMetadata( ISteamUGC* self, UGCQueryHandle_t handle, bool bReturnMetadata );
S_API bool SteamAPI_ISteamUGC_SetReturnChildren( ISteamUGC* self, UGCQueryHandle_t handle, bool bReturnChildren );
S_API bool SteamAPI_ISteamUGC_SetReturnAdditionalPreviews( ISteamUGC* self, UGCQueryHandle_t handle, bool bReturnAdditionalPreviews );
S_API bool SteamAPI_ISteamUGC_SetReturnTotalOnly( ISteamUGC* self, UGCQueryHandle_t handle, bool bReturnTotalOnly );
S_API bool SteamAPI_ISteamUGC_SetReturnPlaytimeStats( ISteamUGC* self, UGCQueryHandle_t handle, uint32 unDays );
S_API bool SteamAPI_ISteamUGC_SetLanguage( ISteamUGC* self, UGCQueryHandle_t handle, const char * pchLanguage );
S_API bool SteamAPI_ISteamUGC_SetAllowCachedResponse( ISteamUGC* self, UGCQueryHandle_t handle, uint32 unMaxAgeSeconds );
S_API bool SteamAPI_ISteamUGC_SetCloudFileNameFilter( ISteamUGC* self, UGCQueryHandle_t handle, const char * pMatchCloudFileName );
S_API bool SteamAPI_ISteamUGC_SetMatchAnyTag( ISteamUGC* self, UGCQueryHandle_t handle, bool bMatchAnyTag );
S_API bool SteamAPI_ISteamUGC_SetSearchText( ISteamUGC* self, UGCQueryHandle_t handle, const char * pSearchText );
S_API bool SteamAPI_ISteamUGC_SetRankedByTrendDays( ISteamUGC* self, UGCQueryHandle_t handle, uint32 unDays );
S_API bool SteamAPI_ISteamUGC_SetTimeCreatedDateRange( ISteamUGC* self, UGCQueryHandle_t handle, RTime32 rtStart, RTime32 rtEnd );
S_API bool SteamAPI_ISteamUGC_SetTimeUpdatedDateRange( ISteamUGC* self, UGCQueryHandle_t handle, RTime32 rtStart, RTime32 rtEnd );
S_API bool SteamAPI_ISteamUGC_AddRequiredKeyValueTag( ISteamUGC* self, UGCQueryHandle_t handle, const char * pKey, const char * pValue );
S_API SteamAPICall_t SteamAPI_ISteamUGC_RequestUGCDetails( ISteamUGC* self, PublishedFileId_t nPublishedFileID, uint32 unMaxAgeSeconds );
S_API SteamAPICall_t SteamAPI_ISteamUGC_CreateItem( ISteamUGC* self, AppId_t nConsumerAppId, EWorkshopFileType eFileType );
S_API UGCUpdateHandle_t SteamAPI_ISteamUGC_StartItemUpdate( ISteamUGC* self, AppId_t nConsumerAppId, PublishedFileId_t nPublishedFileID );
S_API bool SteamAPI_ISteamUGC_SetItemTitle( ISteamUGC* self, UGCUpdateHandle_t handle, const char * pchTitle );
S_API bool SteamAPI_ISteamUGC_SetItemDescription( ISteamUGC* self, UGCUpdateHandle_t handle, const char * pchDescription );
S_API bool SteamAPI_ISteamUGC_SetItemUpdateLanguage( ISteamUGC* self, UGCUpdateHandle_t handle, const char * pchLanguage );
S_API bool SteamAPI_ISteamUGC_SetItemMetadata( ISteamUGC* self, UGCUpdateHandle_t handle, const char * pchMetaData );
S_API bool SteamAPI_ISteamUGC_SetItemVisibility( ISteamUGC* self, UGCUpdateHandle_t handle, ERemoteStoragePublishedFileVisibility eVisibility );
S_API bool SteamAPI_ISteamUGC_SetItemTags( ISteamUGC* self, UGCUpdateHandle_t updateHandle, const SteamParamStringArray_t * pTags );
S_API bool SteamAPI_ISteamUGC_SetItemContent( ISteamUGC* self, UGCUpdateHandle_t handle, const char * pszContentFolder );
S_API bool SteamAPI_ISteamUGC_SetItemPreview( ISteamUGC* self, UGCUpdateHandle_t handle, const char * pszPreviewFile );
S_API bool SteamAPI_ISteamUGC_SetAllowLegacyUpload( ISteamUGC* self, UGCUpdateHandle_t handle, bool bAllowLegacyUpload );
S_API bool SteamAPI_ISteamUGC_RemoveAllItemKeyValueTags( ISteamUGC* self, UGCUpdateHandle_t handle );
S_API bool SteamAPI_ISteamUGC_RemoveItemKeyValueTags( ISteamUGC* self, UGCUpdateHandle_t handle, const char * pchKey );
S_API bool SteamAPI_ISteamUGC_AddItemKeyValueTag( ISteamUGC* self, UGCUpdateHandle_t handle, const char * pchKey, const char * pchValue );
S_API bool SteamAPI_ISteamUGC_AddItemPreviewFile( ISteamUGC* self, UGCUpdateHandle_t handle, const char * pszPreviewFile, EItemPreviewType type );
S_API bool SteamAPI_ISteamUGC_AddItemPreviewVideo( ISteamUGC* self, UGCUpdateHandle_t handle, const char * pszVideoID );
S_API bool SteamAPI_ISteamUGC_UpdateItemPreviewFile( ISteamUGC* self, UGCUpdateHandle_t handle, uint32 index, const char * pszPreviewFile );
S_API bool SteamAPI_ISteamUGC_UpdateItemPreviewVideo( ISteamUGC* self, UGCUpdateHandle_t handle, uint32 index, const char * pszVideoID );
S_API bool SteamAPI_ISteamUGC_RemoveItemPreview( ISteamUGC* self, UGCUpdateHandle_t handle, uint32 index );
S_API SteamAPICall_t SteamAPI_ISteamUGC_SubmitItemUpdate( ISteamUGC* self, UGCUpdateHandle_t handle, const char * pchChangeNote );
S_API EItemUpdateStatus SteamAPI_ISteamUGC_GetItemUpdateProgress( ISteamUGC* self, UGCUpdateHandle_t handle, uint64 * punBytesProcessed, uint64 * punBytesTotal );
S_API SteamAPICall_t SteamAPI_ISteamUGC_SetUserItemVote( ISteamUGC* self, PublishedFileId_t nPublishedFileID, bool bVoteUp );
S_API SteamAPICall_t SteamAPI_ISteamUGC_GetUserItemVote( ISteamUGC* self, PublishedFileId_t nPublishedFileID );
S_API SteamAPICall_t SteamAPI_ISteamUGC_AddItemToFavorites( ISteamUGC* self, AppId_t nAppId, PublishedFileId_t nPublishedFileID );
S_API SteamAPICall_t SteamAPI_ISteamUGC_RemoveItemFromFavorites( ISteamUGC* self, AppId_t nAppId, PublishedFileId_t nPublishedFileID );
S_API SteamAPICall_t SteamAPI_ISteamUGC_SubscribeItem( ISteamUGC* self, PublishedFileId_t nPublishedFileID );
S_API SteamAPICall_t SteamAPI_ISteamUGC_UnsubscribeItem( ISteamUGC* self, PublishedFileId_t nPublishedFileID );
S_API uint32 SteamAPI_ISteamUGC_GetNumSubscribedItems( ISteamUGC* self );
S_API uint32 SteamAPI_ISteamUGC_GetSubscribedItems( ISteamUGC* self, PublishedFileId_t * pvecPublishedFileID, uint32 cMaxEntries );
S_API uint32 SteamAPI_ISteamUGC_GetItemState( ISteamUGC* self, PublishedFileId_t nPublishedFileID );
S_API bool SteamAPI_ISteamUGC_GetItemInstallInfo( ISteamUGC* self, PublishedFileId_t nPublishedFileID, uint64 * punSizeOnDisk, char * pchFolder, uint32 cchFolderSize, uint32 * punTimeStamp );
S_API bool SteamAPI_ISteamUGC_GetItemDownloadInfo( ISteamUGC* self, PublishedFileId_t nPublishedFileID, uint64 * punBytesDownloaded, uint64 * punBytesTotal );
S_API bool SteamAPI_ISteamUGC_DownloadItem( ISteamUGC* self, PublishedFileId_t nPublishedFileID, bool bHighPriority );
S_API bool SteamAPI_ISteamUGC_BInitWorkshopForGameServer( ISteamUGC* self, DepotId_t unWorkshopDepotID, const char * pszFolder );
S_API void SteamAPI_ISteamUGC_SuspendDownloads( ISteamUGC* self, bool bSuspend );
S_API SteamAPICall_t SteamAPI_ISteamUGC_StartPlaytimeTracking( ISteamUGC* self, PublishedFileId_t * pvecPublishedFileID, uint32 unNumPublishedFileIDs );
S_API SteamAPICall_t SteamAPI_ISteamUGC_StopPlaytimeTracking( ISteamUGC* self, PublishedFileId_t * pvecPublishedFileID, uint32 unNumPublishedFileIDs );
S_API SteamAPICall_t SteamAPI_ISteamUGC_StopPlaytimeTrackingForAllItems( ISteamUGC* self );
S_API SteamAPICall_t SteamAPI_ISteamUGC_AddDependency( ISteamUGC* self, PublishedFileId_t nParentPublishedFileID, PublishedFileId_t nChildPublishedFileID );
S_API SteamAPICall_t SteamAPI_ISteamUGC_RemoveDependency( ISteamUGC* self, PublishedFileId_t nParentPublishedFileID, PublishedFileId_t nChildPublishedFileID );
S_API SteamAPICall_t SteamAPI_ISteamUGC_AddAppDependency( ISteamUGC* self, PublishedFileId_t nPublishedFileID, AppId_t nAppID );
S_API SteamAPICall_t SteamAPI_ISteamUGC_RemoveAppDependency( ISteamUGC* self, PublishedFileId_t nPublishedFileID, AppId_t nAppID );
S_API SteamAPICall_t SteamAPI_ISteamUGC_GetAppDependencies( ISteamUGC* self, PublishedFileId_t nPublishedFileID );
S_API SteamAPICall_t SteamAPI_ISteamUGC_DeleteItem( ISteamUGC* self, PublishedFileId_t nPublishedFileID );
S_API bool SteamAPI_ISteamUGC_ShowWorkshopEULA( ISteamUGC* self );
S_API SteamAPICall_t SteamAPI_ISteamUGC_GetWorkshopEULAStatus( ISteamUGC* self );

// ISteamAppList

// A versioned accessor is exported by the library
S_API ISteamAppList *SteamAPI_SteamAppList_v001();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamAppList(), but using this ensures that you are using a matching library.
inline ISteamAppList *SteamAPI_SteamAppList() { return SteamAPI_SteamAppList_v001(); }
S_API uint32 SteamAPI_ISteamAppList_GetNumInstalledApps( ISteamAppList* self );
S_API uint32 SteamAPI_ISteamAppList_GetInstalledApps( ISteamAppList* self, AppId_t * pvecAppID, uint32 unMaxAppIDs );
S_API int SteamAPI_ISteamAppList_GetAppName( ISteamAppList* self, AppId_t nAppID, char * pchName, int cchNameMax );
S_API int SteamAPI_ISteamAppList_GetAppInstallDir( ISteamAppList* self, AppId_t nAppID, char * pchDirectory, int cchNameMax );
S_API int SteamAPI_ISteamAppList_GetAppBuildId( ISteamAppList* self, AppId_t nAppID );

// ISteamHTMLSurface

// A versioned accessor is exported by the library
S_API ISteamHTMLSurface *SteamAPI_SteamHTMLSurface_v005();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamHTMLSurface(), but using this ensures that you are using a matching library.
inline ISteamHTMLSurface *SteamAPI_SteamHTMLSurface() { return SteamAPI_SteamHTMLSurface_v005(); }
S_API bool SteamAPI_ISteamHTMLSurface_Init( ISteamHTMLSurface* self );
S_API bool SteamAPI_ISteamHTMLSurface_Shutdown( ISteamHTMLSurface* self );
S_API SteamAPICall_t SteamAPI_ISteamHTMLSurface_CreateBrowser( ISteamHTMLSurface* self, const char * pchUserAgent, const char * pchUserCSS );
S_API void SteamAPI_ISteamHTMLSurface_RemoveBrowser( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle );
S_API void SteamAPI_ISteamHTMLSurface_LoadURL( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, const char * pchURL, const char * pchPostData );
S_API void SteamAPI_ISteamHTMLSurface_SetSize( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, uint32 unWidth, uint32 unHeight );
S_API void SteamAPI_ISteamHTMLSurface_StopLoad( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle );
S_API void SteamAPI_ISteamHTMLSurface_Reload( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle );
S_API void SteamAPI_ISteamHTMLSurface_GoBack( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle );
S_API void SteamAPI_ISteamHTMLSurface_GoForward( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle );
S_API void SteamAPI_ISteamHTMLSurface_AddHeader( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, const char * pchKey, const char * pchValue );
S_API void SteamAPI_ISteamHTMLSurface_ExecuteJavascript( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, const char * pchScript );
S_API void SteamAPI_ISteamHTMLSurface_MouseUp( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, ISteamHTMLSurface::EHTMLMouseButton eMouseButton );
S_API void SteamAPI_ISteamHTMLSurface_MouseDown( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, ISteamHTMLSurface::EHTMLMouseButton eMouseButton );
S_API void SteamAPI_ISteamHTMLSurface_MouseDoubleClick( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, ISteamHTMLSurface::EHTMLMouseButton eMouseButton );
S_API void SteamAPI_ISteamHTMLSurface_MouseMove( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, int x, int y );
S_API void SteamAPI_ISteamHTMLSurface_MouseWheel( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, int32 nDelta );
S_API void SteamAPI_ISteamHTMLSurface_KeyDown( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, uint32 nNativeKeyCode, ISteamHTMLSurface::EHTMLKeyModifiers eHTMLKeyModifiers, bool bIsSystemKey );
S_API void SteamAPI_ISteamHTMLSurface_KeyUp( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, uint32 nNativeKeyCode, ISteamHTMLSurface::EHTMLKeyModifiers eHTMLKeyModifiers );
S_API void SteamAPI_ISteamHTMLSurface_KeyChar( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, uint32 cUnicodeChar, ISteamHTMLSurface::EHTMLKeyModifiers eHTMLKeyModifiers );
S_API void SteamAPI_ISteamHTMLSurface_SetHorizontalScroll( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, uint32 nAbsolutePixelScroll );
S_API void SteamAPI_ISteamHTMLSurface_SetVerticalScroll( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, uint32 nAbsolutePixelScroll );
S_API void SteamAPI_ISteamHTMLSurface_SetKeyFocus( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, bool bHasKeyFocus );
S_API void SteamAPI_ISteamHTMLSurface_ViewSource( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle );
S_API void SteamAPI_ISteamHTMLSurface_CopyToClipboard( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle );
S_API void SteamAPI_ISteamHTMLSurface_PasteFromClipboard( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle );
S_API void SteamAPI_ISteamHTMLSurface_Find( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, const char * pchSearchStr, bool bCurrentlyInFind, bool bReverse );
S_API void SteamAPI_ISteamHTMLSurface_StopFind( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle );
S_API void SteamAPI_ISteamHTMLSurface_GetLinkAtPosition( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, int x, int y );
S_API void SteamAPI_ISteamHTMLSurface_SetCookie( ISteamHTMLSurface* self, const char * pchHostname, const char * pchKey, const char * pchValue, const char * pchPath, RTime32 nExpires, bool bSecure, bool bHTTPOnly );
S_API void SteamAPI_ISteamHTMLSurface_SetPageScaleFactor( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, float flZoom, int nPointX, int nPointY );
S_API void SteamAPI_ISteamHTMLSurface_SetBackgroundMode( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, bool bBackgroundMode );
S_API void SteamAPI_ISteamHTMLSurface_SetDPIScalingFactor( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, float flDPIScaling );
S_API void SteamAPI_ISteamHTMLSurface_OpenDeveloperTools( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle );
S_API void SteamAPI_ISteamHTMLSurface_AllowStartRequest( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, bool bAllowed );
S_API void SteamAPI_ISteamHTMLSurface_JSDialogResponse( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, bool bResult );
S_API void SteamAPI_ISteamHTMLSurface_FileLoadDialogResponse( ISteamHTMLSurface* self, HHTMLBrowser unBrowserHandle, const char ** pchSelectedFiles );

// ISteamInventory

// A versioned accessor is exported by the library
S_API ISteamInventory *SteamAPI_SteamInventory_v003();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamInventory(), but using this ensures that you are using a matching library.
inline ISteamInventory *SteamAPI_SteamInventory() { return SteamAPI_SteamInventory_v003(); }

// A versioned accessor is exported by the library
S_API ISteamInventory *SteamAPI_SteamGameServerInventory_v003();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamGameServerInventory(), but using this ensures that you are using a matching library.
inline ISteamInventory *SteamAPI_SteamGameServerInventory() { return SteamAPI_SteamGameServerInventory_v003(); }
S_API EResult SteamAPI_ISteamInventory_GetResultStatus( ISteamInventory* self, SteamInventoryResult_t resultHandle );
S_API bool SteamAPI_ISteamInventory_GetResultItems( ISteamInventory* self, SteamInventoryResult_t resultHandle, SteamItemDetails_t * pOutItemsArray, uint32 * punOutItemsArraySize );
S_API bool SteamAPI_ISteamInventory_GetResultItemProperty( ISteamInventory* self, SteamInventoryResult_t resultHandle, uint32 unItemIndex, const char * pchPropertyName, char * pchValueBuffer, uint32 * punValueBufferSizeOut );
S_API uint32 SteamAPI_ISteamInventory_GetResultTimestamp( ISteamInventory* self, SteamInventoryResult_t resultHandle );
S_API bool SteamAPI_ISteamInventory_CheckResultSteamID( ISteamInventory* self, SteamInventoryResult_t resultHandle, uint64_steamid steamIDExpected );
S_API void SteamAPI_ISteamInventory_DestroyResult( ISteamInventory* self, SteamInventoryResult_t resultHandle );
S_API bool SteamAPI_ISteamInventory_GetAllItems( ISteamInventory* self, SteamInventoryResult_t * pResultHandle );
S_API bool SteamAPI_ISteamInventory_GetItemsByID( ISteamInventory* self, SteamInventoryResult_t * pResultHandle, const SteamItemInstanceID_t * pInstanceIDs, uint32 unCountInstanceIDs );
S_API bool SteamAPI_ISteamInventory_SerializeResult( ISteamInventory* self, SteamInventoryResult_t resultHandle, void * pOutBuffer, uint32 * punOutBufferSize );
S_API bool SteamAPI_ISteamInventory_DeserializeResult( ISteamInventory* self, SteamInventoryResult_t * pOutResultHandle, const void * pBuffer, uint32 unBufferSize, bool bRESERVED_MUST_BE_FALSE );
S_API bool SteamAPI_ISteamInventory_GenerateItems( ISteamInventory* self, SteamInventoryResult_t * pResultHandle, const SteamItemDef_t * pArrayItemDefs, const uint32 * punArrayQuantity, uint32 unArrayLength );
S_API bool SteamAPI_ISteamInventory_GrantPromoItems( ISteamInventory* self, SteamInventoryResult_t * pResultHandle );
S_API bool SteamAPI_ISteamInventory_AddPromoItem( ISteamInventory* self, SteamInventoryResult_t * pResultHandle, SteamItemDef_t itemDef );
S_API bool SteamAPI_ISteamInventory_AddPromoItems( ISteamInventory* self, SteamInventoryResult_t * pResultHandle, const SteamItemDef_t * pArrayItemDefs, uint32 unArrayLength );
S_API bool SteamAPI_ISteamInventory_ConsumeItem( ISteamInventory* self, SteamInventoryResult_t * pResultHandle, SteamItemInstanceID_t itemConsume, uint32 unQuantity );
S_API bool SteamAPI_ISteamInventory_ExchangeItems( ISteamInventory* self, SteamInventoryResult_t * pResultHandle, const SteamItemDef_t * pArrayGenerate, const uint32 * punArrayGenerateQuantity, uint32 unArrayGenerateLength, const SteamItemInstanceID_t * pArrayDestroy, const uint32 * punArrayDestroyQuantity, uint32 unArrayDestroyLength );
S_API bool SteamAPI_ISteamInventory_TransferItemQuantity( ISteamInventory* self, SteamInventoryResult_t * pResultHandle, SteamItemInstanceID_t itemIdSource, uint32 unQuantity, SteamItemInstanceID_t itemIdDest );
S_API void SteamAPI_ISteamInventory_SendItemDropHeartbeat( ISteamInventory* self );
S_API bool SteamAPI_ISteamInventory_TriggerItemDrop( ISteamInventory* self, SteamInventoryResult_t * pResultHandle, SteamItemDef_t dropListDefinition );
S_API bool SteamAPI_ISteamInventory_TradeItems( ISteamInventory* self, SteamInventoryResult_t * pResultHandle, uint64_steamid steamIDTradePartner, const SteamItemInstanceID_t * pArrayGive, const uint32 * pArrayGiveQuantity, uint32 nArrayGiveLength, const SteamItemInstanceID_t * pArrayGet, const uint32 * pArrayGetQuantity, uint32 nArrayGetLength );
S_API bool SteamAPI_ISteamInventory_LoadItemDefinitions( ISteamInventory* self );
S_API bool SteamAPI_ISteamInventory_GetItemDefinitionIDs( ISteamInventory* self, SteamItemDef_t * pItemDefIDs, uint32 * punItemDefIDsArraySize );
S_API bool SteamAPI_ISteamInventory_GetItemDefinitionProperty( ISteamInventory* self, SteamItemDef_t iDefinition, const char * pchPropertyName, char * pchValueBuffer, uint32 * punValueBufferSizeOut );
S_API SteamAPICall_t SteamAPI_ISteamInventory_RequestEligiblePromoItemDefinitionsIDs( ISteamInventory* self, uint64_steamid steamID );
S_API bool SteamAPI_ISteamInventory_GetEligiblePromoItemDefinitionIDs( ISteamInventory* self, uint64_steamid steamID, SteamItemDef_t * pItemDefIDs, uint32 * punItemDefIDsArraySize );
S_API SteamAPICall_t SteamAPI_ISteamInventory_StartPurchase( ISteamInventory* self, const SteamItemDef_t * pArrayItemDefs, const uint32 * punArrayQuantity, uint32 unArrayLength );
S_API SteamAPICall_t SteamAPI_ISteamInventory_RequestPrices( ISteamInventory* self );
S_API uint32 SteamAPI_ISteamInventory_GetNumItemsWithPrices( ISteamInventory* self );
S_API bool SteamAPI_ISteamInventory_GetItemsWithPrices( ISteamInventory* self, SteamItemDef_t * pArrayItemDefs, uint64 * pCurrentPrices, uint64 * pBasePrices, uint32 unArrayLength );
S_API bool SteamAPI_ISteamInventory_GetItemPrice( ISteamInventory* self, SteamItemDef_t iDefinition, uint64 * pCurrentPrice, uint64 * pBasePrice );
S_API SteamInventoryUpdateHandle_t SteamAPI_ISteamInventory_StartUpdateProperties( ISteamInventory* self );
S_API bool SteamAPI_ISteamInventory_RemoveProperty( ISteamInventory* self, SteamInventoryUpdateHandle_t handle, SteamItemInstanceID_t nItemID, const char * pchPropertyName );
S_API bool SteamAPI_ISteamInventory_SetPropertyString( ISteamInventory* self, SteamInventoryUpdateHandle_t handle, SteamItemInstanceID_t nItemID, const char * pchPropertyName, const char * pchPropertyValue );
S_API bool SteamAPI_ISteamInventory_SetPropertyBool( ISteamInventory* self, SteamInventoryUpdateHandle_t handle, SteamItemInstanceID_t nItemID, const char * pchPropertyName, bool bValue );
S_API bool SteamAPI_ISteamInventory_SetPropertyInt64( ISteamInventory* self, SteamInventoryUpdateHandle_t handle, SteamItemInstanceID_t nItemID, const char * pchPropertyName, int64 nValue );
S_API bool SteamAPI_ISteamInventory_SetPropertyFloat( ISteamInventory* self, SteamInventoryUpdateHandle_t handle, SteamItemInstanceID_t nItemID, const char * pchPropertyName, float flValue );
S_API bool SteamAPI_ISteamInventory_SubmitUpdateProperties( ISteamInventory* self, SteamInventoryUpdateHandle_t handle, SteamInventoryResult_t * pResultHandle );
S_API bool SteamAPI_ISteamInventory_InspectItem( ISteamInventory* self, SteamInventoryResult_t * pResultHandle, const char * pchItemToken );

// ISteamVideo

// A versioned accessor is exported by the library
S_API ISteamVideo *SteamAPI_SteamVideo_v002();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamVideo(), but using this ensures that you are using a matching library.
inline ISteamVideo *SteamAPI_SteamVideo() { return SteamAPI_SteamVideo_v002(); }
S_API void SteamAPI_ISteamVideo_GetVideoURL( ISteamVideo* self, AppId_t unVideoAppID );
S_API bool SteamAPI_ISteamVideo_IsBroadcasting( ISteamVideo* self, int * pnNumViewers );
S_API void SteamAPI_ISteamVideo_GetOPFSettings( ISteamVideo* self, AppId_t unVideoAppID );
S_API bool SteamAPI_ISteamVideo_GetOPFStringForApp( ISteamVideo* self, AppId_t unVideoAppID, char * pchBuffer, int32 * pnBufferSize );

// ISteamParentalSettings

// A versioned accessor is exported by the library
S_API ISteamParentalSettings *SteamAPI_SteamParentalSettings_v001();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamParentalSettings(), but using this ensures that you are using a matching library.
inline ISteamParentalSettings *SteamAPI_SteamParentalSettings() { return SteamAPI_SteamParentalSettings_v001(); }
S_API bool SteamAPI_ISteamParentalSettings_BIsParentalLockEnabled( ISteamParentalSettings* self );
S_API bool SteamAPI_ISteamParentalSettings_BIsParentalLockLocked( ISteamParentalSettings* self );
S_API bool SteamAPI_ISteamParentalSettings_BIsAppBlocked( ISteamParentalSettings* self, AppId_t nAppID );
S_API bool SteamAPI_ISteamParentalSettings_BIsAppInBlockList( ISteamParentalSettings* self, AppId_t nAppID );
S_API bool SteamAPI_ISteamParentalSettings_BIsFeatureBlocked( ISteamParentalSettings* self, EParentalFeature eFeature );
S_API bool SteamAPI_ISteamParentalSettings_BIsFeatureInBlockList( ISteamParentalSettings* self, EParentalFeature eFeature );

// ISteamRemotePlay

// A versioned accessor is exported by the library
S_API ISteamRemotePlay *SteamAPI_SteamRemotePlay_v001();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamRemotePlay(), but using this ensures that you are using a matching library.
inline ISteamRemotePlay *SteamAPI_SteamRemotePlay() { return SteamAPI_SteamRemotePlay_v001(); }
S_API uint32 SteamAPI_ISteamRemotePlay_GetSessionCount( ISteamRemotePlay* self );
S_API RemotePlaySessionID_t SteamAPI_ISteamRemotePlay_GetSessionID( ISteamRemotePlay* self, int iSessionIndex );
S_API uint64_steamid SteamAPI_ISteamRemotePlay_GetSessionSteamID( ISteamRemotePlay* self, RemotePlaySessionID_t unSessionID );
S_API const char * SteamAPI_ISteamRemotePlay_GetSessionClientName( ISteamRemotePlay* self, RemotePlaySessionID_t unSessionID );
S_API ESteamDeviceFormFactor SteamAPI_ISteamRemotePlay_GetSessionClientFormFactor( ISteamRemotePlay* self, RemotePlaySessionID_t unSessionID );
S_API bool SteamAPI_ISteamRemotePlay_BGetSessionClientResolution( ISteamRemotePlay* self, RemotePlaySessionID_t unSessionID, int * pnResolutionX, int * pnResolutionY );
S_API bool SteamAPI_ISteamRemotePlay_BSendRemotePlayTogetherInvite( ISteamRemotePlay* self, uint64_steamid steamIDFriend );

// ISteamNetworkingMessages

// A versioned accessor is exported by the library
S_API ISteamNetworkingMessages *SteamAPI_SteamNetworkingMessages_SteamAPI_v002();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamNetworkingMessages_SteamAPI(), but using this ensures that you are using a matching library.
inline ISteamNetworkingMessages *SteamAPI_SteamNetworkingMessages_SteamAPI() { return SteamAPI_SteamNetworkingMessages_SteamAPI_v002(); }

// A versioned accessor is exported by the library
S_API ISteamNetworkingMessages *SteamAPI_SteamGameServerNetworkingMessages_SteamAPI_v002();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamGameServerNetworkingMessages_SteamAPI(), but using this ensures that you are using a matching library.
inline ISteamNetworkingMessages *SteamAPI_SteamGameServerNetworkingMessages_SteamAPI() { return SteamAPI_SteamGameServerNetworkingMessages_SteamAPI_v002(); }
S_API EResult SteamAPI_ISteamNetworkingMessages_SendMessageToUser( ISteamNetworkingMessages* self, const SteamNetworkingIdentity & identityRemote, const void * pubData, uint32 cubData, int nSendFlags, int nRemoteChannel );
S_API int SteamAPI_ISteamNetworkingMessages_ReceiveMessagesOnChannel( ISteamNetworkingMessages* self, int nLocalChannel, SteamNetworkingMessage_t ** ppOutMessages, int nMaxMessages );
S_API bool SteamAPI_ISteamNetworkingMessages_AcceptSessionWithUser( ISteamNetworkingMessages* self, const SteamNetworkingIdentity & identityRemote );
S_API bool SteamAPI_ISteamNetworkingMessages_CloseSessionWithUser( ISteamNetworkingMessages* self, const SteamNetworkingIdentity & identityRemote );
S_API bool SteamAPI_ISteamNetworkingMessages_CloseChannelWithUser( ISteamNetworkingMessages* self, const SteamNetworkingIdentity & identityRemote, int nLocalChannel );
S_API ESteamNetworkingConnectionState SteamAPI_ISteamNetworkingMessages_GetSessionConnectionInfo( ISteamNetworkingMessages* self, const SteamNetworkingIdentity & identityRemote, SteamNetConnectionInfo_t * pConnectionInfo, SteamNetConnectionRealTimeStatus_t * pQuickStatus );

// ISteamNetworkingSockets

// A versioned accessor is exported by the library
S_API ISteamNetworkingSockets *SteamAPI_SteamNetworkingSockets_SteamAPI_v012();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamNetworkingSockets_SteamAPI(), but using this ensures that you are using a matching library.
inline ISteamNetworkingSockets *SteamAPI_SteamNetworkingSockets_SteamAPI() { return SteamAPI_SteamNetworkingSockets_SteamAPI_v012(); }

// A versioned accessor is exported by the library
S_API ISteamNetworkingSockets *SteamAPI_SteamGameServerNetworkingSockets_SteamAPI_v012();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamGameServerNetworkingSockets_SteamAPI(), but using this ensures that you are using a matching library.
inline ISteamNetworkingSockets *SteamAPI_SteamGameServerNetworkingSockets_SteamAPI() { return SteamAPI_SteamGameServerNetworkingSockets_SteamAPI_v012(); }
S_API HSteamListenSocket SteamAPI_ISteamNetworkingSockets_CreateListenSocketIP( ISteamNetworkingSockets* self, const SteamNetworkingIPAddr & localAddress, int nOptions, const SteamNetworkingConfigValue_t * pOptions );
S_API HSteamNetConnection SteamAPI_ISteamNetworkingSockets_ConnectByIPAddress( ISteamNetworkingSockets* self, const SteamNetworkingIPAddr & address, int nOptions, const SteamNetworkingConfigValue_t * pOptions );
S_API HSteamListenSocket SteamAPI_ISteamNetworkingSockets_CreateListenSocketP2P( ISteamNetworkingSockets* self, int nLocalVirtualPort, int nOptions, const SteamNetworkingConfigValue_t * pOptions );
S_API HSteamNetConnection SteamAPI_ISteamNetworkingSockets_ConnectP2P( ISteamNetworkingSockets* self, const SteamNetworkingIdentity & identityRemote, int nRemoteVirtualPort, int nOptions, const SteamNetworkingConfigValue_t * pOptions );
S_API EResult SteamAPI_ISteamNetworkingSockets_AcceptConnection( ISteamNetworkingSockets* self, HSteamNetConnection hConn );
S_API bool SteamAPI_ISteamNetworkingSockets_CloseConnection( ISteamNetworkingSockets* self, HSteamNetConnection hPeer, int nReason, const char * pszDebug, bool bEnableLinger );
S_API bool SteamAPI_ISteamNetworkingSockets_CloseListenSocket( ISteamNetworkingSockets* self, HSteamListenSocket hSocket );
S_API bool SteamAPI_ISteamNetworkingSockets_SetConnectionUserData( ISteamNetworkingSockets* self, HSteamNetConnection hPeer, int64 nUserData );
S_API int64 SteamAPI_ISteamNetworkingSockets_GetConnectionUserData( ISteamNetworkingSockets* self, HSteamNetConnection hPeer );
S_API void SteamAPI_ISteamNetworkingSockets_SetConnectionName( ISteamNetworkingSockets* self, HSteamNetConnection hPeer, const char * pszName );
S_API bool SteamAPI_ISteamNetworkingSockets_GetConnectionName( ISteamNetworkingSockets* self, HSteamNetConnection hPeer, char * pszName, int nMaxLen );
S_API EResult SteamAPI_ISteamNetworkingSockets_SendMessageToConnection( ISteamNetworkingSockets* self, HSteamNetConnection hConn, const void * pData, uint32 cbData, int nSendFlags, int64 * pOutMessageNumber );
S_API void SteamAPI_ISteamNetworkingSockets_SendMessages( ISteamNetworkingSockets* self, int nMessages, SteamNetworkingMessage_t *const * pMessages, int64 * pOutMessageNumberOrResult );
S_API EResult SteamAPI_ISteamNetworkingSockets_FlushMessagesOnConnection( ISteamNetworkingSockets* self, HSteamNetConnection hConn );
S_API int SteamAPI_ISteamNetworkingSockets_ReceiveMessagesOnConnection( ISteamNetworkingSockets* self, HSteamNetConnection hConn, SteamNetworkingMessage_t ** ppOutMessages, int nMaxMessages );
S_API bool SteamAPI_ISteamNetworkingSockets_GetConnectionInfo( ISteamNetworkingSockets* self, HSteamNetConnection hConn, SteamNetConnectionInfo_t * pInfo );
S_API EResult SteamAPI_ISteamNetworkingSockets_GetConnectionRealTimeStatus( ISteamNetworkingSockets* self, HSteamNetConnection hConn, SteamNetConnectionRealTimeStatus_t * pStatus, int nLanes, SteamNetConnectionRealTimeLaneStatus_t * pLanes );
S_API int SteamAPI_ISteamNetworkingSockets_GetDetailedConnectionStatus( ISteamNetworkingSockets* self, HSteamNetConnection hConn, char * pszBuf, int cbBuf );
S_API bool SteamAPI_ISteamNetworkingSockets_GetListenSocketAddress( ISteamNetworkingSockets* self, HSteamListenSocket hSocket, SteamNetworkingIPAddr * address );
S_API bool SteamAPI_ISteamNetworkingSockets_CreateSocketPair( ISteamNetworkingSockets* self, HSteamNetConnection * pOutConnection1, HSteamNetConnection * pOutConnection2, bool bUseNetworkLoopback, const SteamNetworkingIdentity * pIdentity1, const SteamNetworkingIdentity * pIdentity2 );
S_API EResult SteamAPI_ISteamNetworkingSockets_ConfigureConnectionLanes( ISteamNetworkingSockets* self, HSteamNetConnection hConn, int nNumLanes, const int * pLanePriorities, const uint16 * pLaneWeights );
S_API bool SteamAPI_ISteamNetworkingSockets_GetIdentity( ISteamNetworkingSockets* self, SteamNetworkingIdentity * pIdentity );
S_API ESteamNetworkingAvailability SteamAPI_ISteamNetworkingSockets_InitAuthentication( ISteamNetworkingSockets* self );
S_API ESteamNetworkingAvailability SteamAPI_ISteamNetworkingSockets_GetAuthenticationStatus( ISteamNetworkingSockets* self, SteamNetAuthenticationStatus_t * pDetails );
S_API HSteamNetPollGroup SteamAPI_ISteamNetworkingSockets_CreatePollGroup( ISteamNetworkingSockets* self );
S_API bool SteamAPI_ISteamNetworkingSockets_DestroyPollGroup( ISteamNetworkingSockets* self, HSteamNetPollGroup hPollGroup );
S_API bool SteamAPI_ISteamNetworkingSockets_SetConnectionPollGroup( ISteamNetworkingSockets* self, HSteamNetConnection hConn, HSteamNetPollGroup hPollGroup );
S_API int SteamAPI_ISteamNetworkingSockets_ReceiveMessagesOnPollGroup( ISteamNetworkingSockets* self, HSteamNetPollGroup hPollGroup, SteamNetworkingMessage_t ** ppOutMessages, int nMaxMessages );
S_API bool SteamAPI_ISteamNetworkingSockets_ReceivedRelayAuthTicket( ISteamNetworkingSockets* self, const void * pvTicket, int cbTicket, SteamDatagramRelayAuthTicket * pOutParsedTicket );
S_API int SteamAPI_ISteamNetworkingSockets_FindRelayAuthTicketForServer( ISteamNetworkingSockets* self, const SteamNetworkingIdentity & identityGameServer, int nRemoteVirtualPort, SteamDatagramRelayAuthTicket * pOutParsedTicket );
S_API HSteamNetConnection SteamAPI_ISteamNetworkingSockets_ConnectToHostedDedicatedServer( ISteamNetworkingSockets* self, const SteamNetworkingIdentity & identityTarget, int nRemoteVirtualPort, int nOptions, const SteamNetworkingConfigValue_t * pOptions );
S_API uint16 SteamAPI_ISteamNetworkingSockets_GetHostedDedicatedServerPort( ISteamNetworkingSockets* self );
S_API SteamNetworkingPOPID SteamAPI_ISteamNetworkingSockets_GetHostedDedicatedServerPOPID( ISteamNetworkingSockets* self );
S_API EResult SteamAPI_ISteamNetworkingSockets_GetHostedDedicatedServerAddress( ISteamNetworkingSockets* self, SteamDatagramHostedAddress * pRouting );
S_API HSteamListenSocket SteamAPI_ISteamNetworkingSockets_CreateHostedDedicatedServerListenSocket( ISteamNetworkingSockets* self, int nLocalVirtualPort, int nOptions, const SteamNetworkingConfigValue_t * pOptions );
S_API EResult SteamAPI_ISteamNetworkingSockets_GetGameCoordinatorServerLogin( ISteamNetworkingSockets* self, SteamDatagramGameCoordinatorServerLogin * pLoginInfo, int * pcbSignedBlob, void * pBlob );
S_API HSteamNetConnection SteamAPI_ISteamNetworkingSockets_ConnectP2PCustomSignaling( ISteamNetworkingSockets* self, ISteamNetworkingConnectionSignaling * pSignaling, const SteamNetworkingIdentity * pPeerIdentity, int nRemoteVirtualPort, int nOptions, const SteamNetworkingConfigValue_t * pOptions );
S_API bool SteamAPI_ISteamNetworkingSockets_ReceivedP2PCustomSignal( ISteamNetworkingSockets* self, const void * pMsg, int cbMsg, ISteamNetworkingSignalingRecvContext * pContext );
S_API bool SteamAPI_ISteamNetworkingSockets_GetCertificateRequest( ISteamNetworkingSockets* self, int * pcbBlob, void * pBlob, SteamNetworkingErrMsg & errMsg );
S_API bool SteamAPI_ISteamNetworkingSockets_SetCertificate( ISteamNetworkingSockets* self, const void * pCertificate, int cbCertificate, SteamNetworkingErrMsg & errMsg );
S_API void SteamAPI_ISteamNetworkingSockets_ResetIdentity( ISteamNetworkingSockets* self, const SteamNetworkingIdentity * pIdentity );
S_API void SteamAPI_ISteamNetworkingSockets_RunCallbacks( ISteamNetworkingSockets* self );
S_API bool SteamAPI_ISteamNetworkingSockets_BeginAsyncRequestFakeIP( ISteamNetworkingSockets* self, int nNumPorts );
S_API void SteamAPI_ISteamNetworkingSockets_GetFakeIP( ISteamNetworkingSockets* self, int idxFirstPort, SteamNetworkingFakeIPResult_t * pInfo );
S_API HSteamListenSocket SteamAPI_ISteamNetworkingSockets_CreateListenSocketP2PFakeIP( ISteamNetworkingSockets* self, int idxFakePort, int nOptions, const SteamNetworkingConfigValue_t * pOptions );
S_API EResult SteamAPI_ISteamNetworkingSockets_GetRemoteFakeIPForConnection( ISteamNetworkingSockets* self, HSteamNetConnection hConn, SteamNetworkingIPAddr * pOutAddr );
S_API ISteamNetworkingFakeUDPPort * SteamAPI_ISteamNetworkingSockets_CreateFakeUDPPort( ISteamNetworkingSockets* self, int idxFakeServerPort );

// ISteamNetworkingUtils

// A versioned accessor is exported by the library
S_API ISteamNetworkingUtils *SteamAPI_SteamNetworkingUtils_SteamAPI_v004();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamNetworkingUtils_SteamAPI(), but using this ensures that you are using a matching library.
inline ISteamNetworkingUtils *SteamAPI_SteamNetworkingUtils_SteamAPI() { return SteamAPI_SteamNetworkingUtils_SteamAPI_v004(); }
S_API SteamNetworkingMessage_t * SteamAPI_ISteamNetworkingUtils_AllocateMessage( ISteamNetworkingUtils* self, int cbAllocateBuffer );
S_API void SteamAPI_ISteamNetworkingUtils_InitRelayNetworkAccess( ISteamNetworkingUtils* self );
S_API ESteamNetworkingAvailability SteamAPI_ISteamNetworkingUtils_GetRelayNetworkStatus( ISteamNetworkingUtils* self, SteamRelayNetworkStatus_t * pDetails );
S_API float SteamAPI_ISteamNetworkingUtils_GetLocalPingLocation( ISteamNetworkingUtils* self, SteamNetworkPingLocation_t & result );
S_API int SteamAPI_ISteamNetworkingUtils_EstimatePingTimeBetweenTwoLocations( ISteamNetworkingUtils* self, const SteamNetworkPingLocation_t & location1, const SteamNetworkPingLocation_t & location2 );
S_API int SteamAPI_ISteamNetworkingUtils_EstimatePingTimeFromLocalHost( ISteamNetworkingUtils* self, const SteamNetworkPingLocation_t & remoteLocation );
S_API void SteamAPI_ISteamNetworkingUtils_ConvertPingLocationToString( ISteamNetworkingUtils* self, const SteamNetworkPingLocation_t & location, char * pszBuf, int cchBufSize );
S_API bool SteamAPI_ISteamNetworkingUtils_ParsePingLocationString( ISteamNetworkingUtils* self, const char * pszString, SteamNetworkPingLocation_t & result );
S_API bool SteamAPI_ISteamNetworkingUtils_CheckPingDataUpToDate( ISteamNetworkingUtils* self, float flMaxAgeSeconds );
S_API int SteamAPI_ISteamNetworkingUtils_GetPingToDataCenter( ISteamNetworkingUtils* self, SteamNetworkingPOPID popID, SteamNetworkingPOPID * pViaRelayPoP );
S_API int SteamAPI_ISteamNetworkingUtils_GetDirectPingToPOP( ISteamNetworkingUtils* self, SteamNetworkingPOPID popID );
S_API int SteamAPI_ISteamNetworkingUtils_GetPOPCount( ISteamNetworkingUtils* self );
S_API int SteamAPI_ISteamNetworkingUtils_GetPOPList( ISteamNetworkingUtils* self, SteamNetworkingPOPID * list, int nListSz );
S_API SteamNetworkingMicroseconds SteamAPI_ISteamNetworkingUtils_GetLocalTimestamp( ISteamNetworkingUtils* self );
S_API void SteamAPI_ISteamNetworkingUtils_SetDebugOutputFunction( ISteamNetworkingUtils* self, ESteamNetworkingSocketsDebugOutputType eDetailLevel, FSteamNetworkingSocketsDebugOutput pfnFunc );
S_API bool SteamAPI_ISteamNetworkingUtils_IsFakeIPv4( ISteamNetworkingUtils* self, uint32 nIPv4 );
S_API ESteamNetworkingFakeIPType SteamAPI_ISteamNetworkingUtils_GetIPv4FakeIPType( ISteamNetworkingUtils* self, uint32 nIPv4 );
S_API EResult SteamAPI_ISteamNetworkingUtils_GetRealIdentityForFakeIP( ISteamNetworkingUtils* self, const SteamNetworkingIPAddr & fakeIP, SteamNetworkingIdentity * pOutRealIdentity );
S_API bool SteamAPI_ISteamNetworkingUtils_SetGlobalConfigValueInt32( ISteamNetworkingUtils* self, ESteamNetworkingConfigValue eValue, int32 val );
S_API bool SteamAPI_ISteamNetworkingUtils_SetGlobalConfigValueFloat( ISteamNetworkingUtils* self, ESteamNetworkingConfigValue eValue, float val );
S_API bool SteamAPI_ISteamNetworkingUtils_SetGlobalConfigValueString( ISteamNetworkingUtils* self, ESteamNetworkingConfigValue eValue, const char * val );
S_API bool SteamAPI_ISteamNetworkingUtils_SetGlobalConfigValuePtr( ISteamNetworkingUtils* self, ESteamNetworkingConfigValue eValue, void * val );
S_API bool SteamAPI_ISteamNetworkingUtils_SetConnectionConfigValueInt32( ISteamNetworkingUtils* self, HSteamNetConnection hConn, ESteamNetworkingConfigValue eValue, int32 val );
S_API bool SteamAPI_ISteamNetworkingUtils_SetConnectionConfigValueFloat( ISteamNetworkingUtils* self, HSteamNetConnection hConn, ESteamNetworkingConfigValue eValue, float val );
S_API bool SteamAPI_ISteamNetworkingUtils_SetConnectionConfigValueString( ISteamNetworkingUtils* self, HSteamNetConnection hConn, ESteamNetworkingConfigValue eValue, const char * val );
S_API bool SteamAPI_ISteamNetworkingUtils_SetGlobalCallback_SteamNetConnectionStatusChanged( ISteamNetworkingUtils* self, FnSteamNetConnectionStatusChanged fnCallback );
S_API bool SteamAPI_ISteamNetworkingUtils_SetGlobalCallback_SteamNetAuthenticationStatusChanged( ISteamNetworkingUtils* self, FnSteamNetAuthenticationStatusChanged fnCallback );
S_API bool SteamAPI_ISteamNetworkingUtils_SetGlobalCallback_SteamRelayNetworkStatusChanged( ISteamNetworkingUtils* self, FnSteamRelayNetworkStatusChanged fnCallback );
S_API bool SteamAPI_ISteamNetworkingUtils_SetGlobalCallback_FakeIPResult( ISteamNetworkingUtils* self, FnSteamNetworkingFakeIPResult fnCallback );
S_API bool SteamAPI_ISteamNetworkingUtils_SetGlobalCallback_MessagesSessionRequest( ISteamNetworkingUtils* self, FnSteamNetworkingMessagesSessionRequest fnCallback );
S_API bool SteamAPI_ISteamNetworkingUtils_SetGlobalCallback_MessagesSessionFailed( ISteamNetworkingUtils* self, FnSteamNetworkingMessagesSessionFailed fnCallback );
S_API bool SteamAPI_ISteamNetworkingUtils_SetConfigValue( ISteamNetworkingUtils* self, ESteamNetworkingConfigValue eValue, ESteamNetworkingConfigScope eScopeType, intptr_t scopeObj, ESteamNetworkingConfigDataType eDataType, const void * pArg );
S_API bool SteamAPI_ISteamNetworkingUtils_SetConfigValueStruct( ISteamNetworkingUtils* self, const SteamNetworkingConfigValue_t & opt, ESteamNetworkingConfigScope eScopeType, intptr_t scopeObj );
S_API ESteamNetworkingGetConfigValueResult SteamAPI_ISteamNetworkingUtils_GetConfigValue( ISteamNetworkingUtils* self, ESteamNetworkingConfigValue eValue, ESteamNetworkingConfigScope eScopeType, intptr_t scopeObj, ESteamNetworkingConfigDataType * pOutDataType, void * pResult, size_t * cbResult );
S_API const char * SteamAPI_ISteamNetworkingUtils_GetConfigValueInfo( ISteamNetworkingUtils* self, ESteamNetworkingConfigValue eValue, ESteamNetworkingConfigDataType * pOutDataType, ESteamNetworkingConfigScope * pOutScope );
S_API ESteamNetworkingConfigValue SteamAPI_ISteamNetworkingUtils_IterateGenericEditableConfigValues( ISteamNetworkingUtils* self, ESteamNetworkingConfigValue eCurrent, bool bEnumerateDevVars );
S_API void SteamAPI_ISteamNetworkingUtils_SteamNetworkingIPAddr_ToString( ISteamNetworkingUtils* self, const SteamNetworkingIPAddr & addr, char * buf, uint32 cbBuf, bool bWithPort );
S_API bool SteamAPI_ISteamNetworkingUtils_SteamNetworkingIPAddr_ParseString( ISteamNetworkingUtils* self, SteamNetworkingIPAddr * pAddr, const char * pszStr );
S_API ESteamNetworkingFakeIPType SteamAPI_ISteamNetworkingUtils_SteamNetworkingIPAddr_GetFakeIPType( ISteamNetworkingUtils* self, const SteamNetworkingIPAddr & addr );
S_API void SteamAPI_ISteamNetworkingUtils_SteamNetworkingIdentity_ToString( ISteamNetworkingUtils* self, const SteamNetworkingIdentity & identity, char * buf, uint32 cbBuf );
S_API bool SteamAPI_ISteamNetworkingUtils_SteamNetworkingIdentity_ParseString( ISteamNetworkingUtils* self, SteamNetworkingIdentity * pIdentity, const char * pszStr );

// ISteamGameServer

// A versioned accessor is exported by the library
S_API ISteamGameServer *SteamAPI_SteamGameServer_v014();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamGameServer(), but using this ensures that you are using a matching library.
inline ISteamGameServer *SteamAPI_SteamGameServer() { return SteamAPI_SteamGameServer_v014(); }
S_API void SteamAPI_ISteamGameServer_SetProduct( ISteamGameServer* self, const char * pszProduct );
S_API void SteamAPI_ISteamGameServer_SetGameDescription( ISteamGameServer* self, const char * pszGameDescription );
S_API void SteamAPI_ISteamGameServer_SetModDir( ISteamGameServer* self, const char * pszModDir );
S_API void SteamAPI_ISteamGameServer_SetDedicatedServer( ISteamGameServer* self, bool bDedicated );
S_API void SteamAPI_ISteamGameServer_LogOn( ISteamGameServer* self, const char * pszToken );
S_API void SteamAPI_ISteamGameServer_LogOnAnonymous( ISteamGameServer* self );
S_API void SteamAPI_ISteamGameServer_LogOff( ISteamGameServer* self );
S_API bool SteamAPI_ISteamGameServer_BLoggedOn( ISteamGameServer* self );
S_API bool SteamAPI_ISteamGameServer_BSecure( ISteamGameServer* self );
S_API uint64_steamid SteamAPI_ISteamGameServer_GetSteamID( ISteamGameServer* self );
S_API bool SteamAPI_ISteamGameServer_WasRestartRequested( ISteamGameServer* self );
S_API void SteamAPI_ISteamGameServer_SetMaxPlayerCount( ISteamGameServer* self, int cPlayersMax );
S_API void SteamAPI_ISteamGameServer_SetBotPlayerCount( ISteamGameServer* self, int cBotplayers );
S_API void SteamAPI_ISteamGameServer_SetServerName( ISteamGameServer* self, const char * pszServerName );
S_API void SteamAPI_ISteamGameServer_SetMapName( ISteamGameServer* self, const char * pszMapName );
S_API void SteamAPI_ISteamGameServer_SetPasswordProtected( ISteamGameServer* self, bool bPasswordProtected );
S_API void SteamAPI_ISteamGameServer_SetSpectatorPort( ISteamGameServer* self, uint16 unSpectatorPort );
S_API void SteamAPI_ISteamGameServer_SetSpectatorServerName( ISteamGameServer* self, const char * pszSpectatorServerName );
S_API void SteamAPI_ISteamGameServer_ClearAllKeyValues( ISteamGameServer* self );
S_API void SteamAPI_ISteamGameServer_SetKeyValue( ISteamGameServer* self, const char * pKey, const char * pValue );
S_API void SteamAPI_ISteamGameServer_SetGameTags( ISteamGameServer* self, const char * pchGameTags );
S_API void SteamAPI_ISteamGameServer_SetGameData( ISteamGameServer* self, const char * pchGameData );
S_API void SteamAPI_ISteamGameServer_SetRegion( ISteamGameServer* self, const char * pszRegion );
S_API void SteamAPI_ISteamGameServer_SetAdvertiseServerActive( ISteamGameServer* self, bool bActive );
S_API HAuthTicket SteamAPI_ISteamGameServer_GetAuthSessionTicket( ISteamGameServer* self, void * pTicket, int cbMaxTicket, uint32 * pcbTicket );
S_API EBeginAuthSessionResult SteamAPI_ISteamGameServer_BeginAuthSession( ISteamGameServer* self, const void * pAuthTicket, int cbAuthTicket, uint64_steamid steamID );
S_API void SteamAPI_ISteamGameServer_EndAuthSession( ISteamGameServer* self, uint64_steamid steamID );
S_API void SteamAPI_ISteamGameServer_CancelAuthTicket( ISteamGameServer* self, HAuthTicket hAuthTicket );
S_API EUserHasLicenseForAppResult SteamAPI_ISteamGameServer_UserHasLicenseForApp( ISteamGameServer* self, uint64_steamid steamID, AppId_t appID );
S_API bool SteamAPI_ISteamGameServer_RequestUserGroupStatus( ISteamGameServer* self, uint64_steamid steamIDUser, uint64_steamid steamIDGroup );
S_API void SteamAPI_ISteamGameServer_GetGameplayStats( ISteamGameServer* self );
S_API SteamAPICall_t SteamAPI_ISteamGameServer_GetServerReputation( ISteamGameServer* self );
S_API SteamIPAddress_t SteamAPI_ISteamGameServer_GetPublicIP( ISteamGameServer* self );
S_API bool SteamAPI_ISteamGameServer_HandleIncomingPacket( ISteamGameServer* self, const void * pData, int cbData, uint32 srcIP, uint16 srcPort );
S_API int SteamAPI_ISteamGameServer_GetNextOutgoingPacket( ISteamGameServer* self, void * pOut, int cbMaxOut, uint32 * pNetAdr, uint16 * pPort );
S_API SteamAPICall_t SteamAPI_ISteamGameServer_AssociateWithClan( ISteamGameServer* self, uint64_steamid steamIDClan );
S_API SteamAPICall_t SteamAPI_ISteamGameServer_ComputeNewPlayerCompatibility( ISteamGameServer* self, uint64_steamid steamIDNewPlayer );
S_API bool SteamAPI_ISteamGameServer_SendUserConnectAndAuthenticate_DEPRECATED( ISteamGameServer* self, uint32 unIPClient, const void * pvAuthBlob, uint32 cubAuthBlobSize, CSteamID * pSteamIDUser );
S_API uint64_steamid SteamAPI_ISteamGameServer_CreateUnauthenticatedUserConnection( ISteamGameServer* self );
S_API void SteamAPI_ISteamGameServer_SendUserDisconnect_DEPRECATED( ISteamGameServer* self, uint64_steamid steamIDUser );
S_API bool SteamAPI_ISteamGameServer_BUpdateUserData( ISteamGameServer* self, uint64_steamid steamIDUser, const char * pchPlayerName, uint32 uScore );

// ISteamGameServerStats

// A versioned accessor is exported by the library
S_API ISteamGameServerStats *SteamAPI_SteamGameServerStats_v001();
// Inline, unversioned accessor to get the current version.  Essentially the same as SteamGameServerStats(), but using this ensures that you are using a matching library.
inline ISteamGameServerStats *SteamAPI_SteamGameServerStats() { return SteamAPI_SteamGameServerStats_v001(); }
S_API SteamAPICall_t SteamAPI_ISteamGameServerStats_RequestUserStats( ISteamGameServerStats* self, uint64_steamid steamIDUser );
S_API bool SteamAPI_ISteamGameServerStats_GetUserStatInt32( ISteamGameServerStats* self, uint64_steamid steamIDUser, const char * pchName, int32 * pData );
S_API bool SteamAPI_ISteamGameServerStats_GetUserStatFloat( ISteamGameServerStats* self, uint64_steamid steamIDUser, const char * pchName, float * pData );
S_API bool SteamAPI_ISteamGameServerStats_GetUserAchievement( ISteamGameServerStats* self, uint64_steamid steamIDUser, const char * pchName, bool * pbAchieved );
S_API bool SteamAPI_ISteamGameServerStats_SetUserStatInt32( ISteamGameServerStats* self, uint64_steamid steamIDUser, const char * pchName, int32 nData );
S_API bool SteamAPI_ISteamGameServerStats_SetUserStatFloat( ISteamGameServerStats* self, uint64_steamid steamIDUser, const char * pchName, float fData );
S_API bool SteamAPI_ISteamGameServerStats_UpdateUserAvgRateStat( ISteamGameServerStats* self, uint64_steamid steamIDUser, const char * pchName, float flCountThisSession, double dSessionLength );
S_API bool SteamAPI_ISteamGameServerStats_SetUserAchievement( ISteamGameServerStats* self, uint64_steamid steamIDUser, const char * pchName );
S_API bool SteamAPI_ISteamGameServerStats_ClearUserAchievement( ISteamGameServerStats* self, uint64_steamid steamIDUser, const char * pchName );
S_API SteamAPICall_t SteamAPI_ISteamGameServerStats_StoreUserStats( ISteamGameServerStats* self, uint64_steamid steamIDUser );

// ISteamNetworkingFakeUDPPort
S_API void SteamAPI_ISteamNetworkingFakeUDPPort_DestroyFakeUDPPort( ISteamNetworkingFakeUDPPort* self );
S_API EResult SteamAPI_ISteamNetworkingFakeUDPPort_SendMessageToFakeIP( ISteamNetworkingFakeUDPPort* self, const SteamNetworkingIPAddr & remoteAddress, const void * pData, uint32 cbData, int nSendFlags );
S_API int SteamAPI_ISteamNetworkingFakeUDPPort_ReceiveMessages( ISteamNetworkingFakeUDPPort* self, SteamNetworkingMessage_t ** ppOutMessages, int nMaxMessages );
S_API void SteamAPI_ISteamNetworkingFakeUDPPort_ScheduleCleanup( ISteamNetworkingFakeUDPPort* self, const SteamNetworkingIPAddr & remoteAddress );

// SteamIPAddress_t
S_API bool SteamAPI_SteamIPAddress_t_IsSet( SteamIPAddress_t* self );

// MatchMakingKeyValuePair_t
S_API void SteamAPI_MatchMakingKeyValuePair_t_Construct( MatchMakingKeyValuePair_t* self );

// servernetadr_t
S_API void SteamAPI_servernetadr_t_Construct( servernetadr_t* self );
S_API void SteamAPI_servernetadr_t_Init( servernetadr_t* self, unsigned int ip, uint16 usQueryPort, uint16 usConnectionPort );
S_API uint16 SteamAPI_servernetadr_t_GetQueryPort( servernetadr_t* self );
S_API void SteamAPI_servernetadr_t_SetQueryPort( servernetadr_t* self, uint16 usPort );
S_API uint16 SteamAPI_servernetadr_t_GetConnectionPort( servernetadr_t* self );
S_API void SteamAPI_servernetadr_t_SetConnectionPort( servernetadr_t* self, uint16 usPort );
S_API uint32 SteamAPI_servernetadr_t_GetIP( servernetadr_t* self );
S_API void SteamAPI_servernetadr_t_SetIP( servernetadr_t* self, uint32 unIP );
S_API const char * SteamAPI_servernetadr_t_GetConnectionAddressString( servernetadr_t* self );
S_API const char * SteamAPI_servernetadr_t_GetQueryAddressString( servernetadr_t* self );
S_API bool SteamAPI_servernetadr_t_IsLessThan( servernetadr_t* self, const servernetadr_t & netadr );
S_API void SteamAPI_servernetadr_t_Assign( servernetadr_t* self, const servernetadr_t & that );

// gameserveritem_t
S_API void SteamAPI_gameserveritem_t_Construct( gameserveritem_t* self );
S_API const char * SteamAPI_gameserveritem_t_GetName( gameserveritem_t* self );
S_API void SteamAPI_gameserveritem_t_SetName( gameserveritem_t* self, const char * pName );

// SteamNetworkingIPAddr
S_API void SteamAPI_SteamNetworkingIPAddr_Clear( SteamNetworkingIPAddr* self );
S_API bool SteamAPI_SteamNetworkingIPAddr_IsIPv6AllZeros( SteamNetworkingIPAddr* self );
S_API void SteamAPI_SteamNetworkingIPAddr_SetIPv6( SteamNetworkingIPAddr* self, const uint8 * ipv6, uint16 nPort );
S_API void SteamAPI_SteamNetworkingIPAddr_SetIPv4( SteamNetworkingIPAddr* self, uint32 nIP, uint16 nPort );
S_API bool SteamAPI_SteamNetworkingIPAddr_IsIPv4( SteamNetworkingIPAddr* self );
S_API uint32 SteamAPI_SteamNetworkingIPAddr_GetIPv4( SteamNetworkingIPAddr* self );
S_API void SteamAPI_SteamNetworkingIPAddr_SetIPv6LocalHost( SteamNetworkingIPAddr* self, uint16 nPort );
S_API bool SteamAPI_SteamNetworkingIPAddr_IsLocalHost( SteamNetworkingIPAddr* self );
S_API void SteamAPI_SteamNetworkingIPAddr_ToString( SteamNetworkingIPAddr* self, char * buf, uint32 cbBuf, bool bWithPort );
S_API bool SteamAPI_SteamNetworkingIPAddr_ParseString( SteamNetworkingIPAddr* self, const char * pszStr );
S_API bool SteamAPI_SteamNetworkingIPAddr_IsEqualTo( SteamNetworkingIPAddr* self, const SteamNetworkingIPAddr & x );
S_API ESteamNetworkingFakeIPType SteamAPI_SteamNetworkingIPAddr_GetFakeIPType( SteamNetworkingIPAddr* self );
S_API bool SteamAPI_SteamNetworkingIPAddr_IsFakeIP( SteamNetworkingIPAddr* self );

// SteamNetworkingIdentity
S_API void SteamAPI_SteamNetworkingIdentity_Clear( SteamNetworkingIdentity* self );
S_API bool SteamAPI_SteamNetworkingIdentity_IsInvalid( SteamNetworkingIdentity* self );
S_API void SteamAPI_SteamNetworkingIdentity_SetSteamID( SteamNetworkingIdentity* self, uint64_steamid steamID );
S_API uint64_steamid SteamAPI_SteamNetworkingIdentity_GetSteamID( SteamNetworkingIdentity* self );
S_API void SteamAPI_SteamNetworkingIdentity_SetSteamID64( SteamNetworkingIdentity* self, uint64 steamID );
S_API uint64 SteamAPI_SteamNetworkingIdentity_GetSteamID64( SteamNetworkingIdentity* self );
S_API bool SteamAPI_SteamNetworkingIdentity_SetXboxPairwiseID( SteamNetworkingIdentity* self, const char * pszString );
S_API const char * SteamAPI_SteamNetworkingIdentity_GetXboxPairwiseID( SteamNetworkingIdentity* self );
S_API void SteamAPI_SteamNetworkingIdentity_SetPSNID( SteamNetworkingIdentity* self, uint64 id );
S_API uint64 SteamAPI_SteamNetworkingIdentity_GetPSNID( SteamNetworkingIdentity* self );
S_API void SteamAPI_SteamNetworkingIdentity_SetStadiaID( SteamNetworkingIdentity* self, uint64 id );
S_API uint64 SteamAPI_SteamNetworkingIdentity_GetStadiaID( SteamNetworkingIdentity* self );
S_API void SteamAPI_SteamNetworkingIdentity_SetIPAddr( SteamNetworkingIdentity* self, const SteamNetworkingIPAddr & addr );
S_API const SteamNetworkingIPAddr * SteamAPI_SteamNetworkingIdentity_GetIPAddr( SteamNetworkingIdentity* self );
S_API void SteamAPI_SteamNetworkingIdentity_SetIPv4Addr( SteamNetworkingIdentity* self, uint32 nIPv4, uint16 nPort );
S_API uint32 SteamAPI_SteamNetworkingIdentity_GetIPv4( SteamNetworkingIdentity* self );
S_API ESteamNetworkingFakeIPType SteamAPI_SteamNetworkingIdentity_GetFakeIPType( SteamNetworkingIdentity* self );
S_API bool SteamAPI_SteamNetworkingIdentity_IsFakeIP( SteamNetworkingIdentity* self );
S_API void SteamAPI_SteamNetworkingIdentity_SetLocalHost( SteamNetworkingIdentity* self );
S_API bool SteamAPI_SteamNetworkingIdentity_IsLocalHost( SteamNetworkingIdentity* self );
S_API bool SteamAPI_SteamNetworkingIdentity_SetGenericString( SteamNetworkingIdentity* self, const char * pszString );
S_API const char * SteamAPI_SteamNetworkingIdentity_GetGenericString( SteamNetworkingIdentity* self );
S_API bool SteamAPI_SteamNetworkingIdentity_SetGenericBytes( SteamNetworkingIdentity* self, const void * data, uint32 cbLen );
S_API const uint8 * SteamAPI_SteamNetworkingIdentity_GetGenericBytes( SteamNetworkingIdentity* self, int & cbLen );
S_API bool SteamAPI_SteamNetworkingIdentity_IsEqualTo( SteamNetworkingIdentity* self, const SteamNetworkingIdentity & x );
S_API void SteamAPI_SteamNetworkingIdentity_ToString( SteamNetworkingIdentity* self, char * buf, uint32 cbBuf );
S_API bool SteamAPI_SteamNetworkingIdentity_ParseString( SteamNetworkingIdentity* self, const char * pszStr );

// SteamNetworkingMessage_t
S_API void SteamAPI_SteamNetworkingMessage_t_Release( SteamNetworkingMessage_t* self );

// SteamNetworkingConfigValue_t
S_API void SteamAPI_SteamNetworkingConfigValue_t_SetInt32( SteamNetworkingConfigValue_t* self, ESteamNetworkingConfigValue eVal, int32_t data );
S_API void SteamAPI_SteamNetworkingConfigValue_t_SetInt64( SteamNetworkingConfigValue_t* self, ESteamNetworkingConfigValue eVal, int64_t data );
S_API void SteamAPI_SteamNetworkingConfigValue_t_SetFloat( SteamNetworkingConfigValue_t* self, ESteamNetworkingConfigValue eVal, float data );
S_API void SteamAPI_SteamNetworkingConfigValue_t_SetPtr( SteamNetworkingConfigValue_t* self, ESteamNetworkingConfigValue eVal, void * data );
S_API void SteamAPI_SteamNetworkingConfigValue_t_SetString( SteamNetworkingConfigValue_t* self, ESteamNetworkingConfigValue eVal, const char * data );

// SteamDatagramHostedAddress
S_API void SteamAPI_SteamDatagramHostedAddress_Clear( SteamDatagramHostedAddress* self );
S_API SteamNetworkingPOPID SteamAPI_SteamDatagramHostedAddress_GetPopID( SteamDatagramHostedAddress* self );
S_API void SteamAPI_SteamDatagramHostedAddress_SetDevAddress( SteamDatagramHostedAddress* self, uint32 nIP, uint16 nPort, SteamNetworkingPOPID popid );
#endif // STEAMAPIFLAT_H
