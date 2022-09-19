//====== Copyright 1996-2013, Valve Corporation, All rights reserved. =======
//
// Purpose: interface to steam ugc
//
//=============================================================================

#ifndef ISTEAMUGC_H
#define ISTEAMUGC_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"
#include "isteamremotestorage.h"

// callbacks
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 


typedef uint64 UGCQueryHandle_t;
typedef uint64 UGCUpdateHandle_t;


const UGCQueryHandle_t k_UGCQueryHandleInvalid = 0xffffffffffffffffull;
const UGCUpdateHandle_t k_UGCUpdateHandleInvalid = 0xffffffffffffffffull;


// Matching UGC types for queries
enum EUGCMatchingUGCType
{
	k_EUGCMatchingUGCType_Items				 = 0,		// both mtx items and ready-to-use items
	k_EUGCMatchingUGCType_Items_Mtx			 = 1,
	k_EUGCMatchingUGCType_Items_ReadyToUse	 = 2,
	k_EUGCMatchingUGCType_Collections		 = 3,
	k_EUGCMatchingUGCType_Artwork			 = 4,
	k_EUGCMatchingUGCType_Videos			 = 5,
	k_EUGCMatchingUGCType_Screenshots		 = 6,
	k_EUGCMatchingUGCType_AllGuides			 = 7,		// both web guides and integrated guides
	k_EUGCMatchingUGCType_WebGuides			 = 8,
	k_EUGCMatchingUGCType_IntegratedGuides	 = 9,
	k_EUGCMatchingUGCType_UsableInGame		 = 10,		// ready-to-use items and integrated guides
	k_EUGCMatchingUGCType_ControllerBindings = 11,
	k_EUGCMatchingUGCType_GameManagedItems	 = 12,		// game managed items (not managed by users)
	k_EUGCMatchingUGCType_All				 = ~0,		// @note: will only be valid for CreateQueryUserUGCRequest requests
};

// Different lists of published UGC for a user.
// If the current logged in user is different than the specified user, then some options may not be allowed.
enum EUserUGCList
{
	k_EUserUGCList_Published,
	k_EUserUGCList_VotedOn,
	k_EUserUGCList_VotedUp,
	k_EUserUGCList_VotedDown,
	k_EUserUGCList_WillVoteLater,
	k_EUserUGCList_Favorited,
	k_EUserUGCList_Subscribed,
	k_EUserUGCList_UsedOrPlayed,
	k_EUserUGCList_Followed,
};

// Sort order for user published UGC lists (defaults to creation order descending)
enum EUserUGCListSortOrder
{
	k_EUserUGCListSortOrder_CreationOrderDesc,
	k_EUserUGCListSortOrder_CreationOrderAsc,
	k_EUserUGCListSortOrder_TitleAsc,
	k_EUserUGCListSortOrder_LastUpdatedDesc,
	k_EUserUGCListSortOrder_SubscriptionDateDesc,
	k_EUserUGCListSortOrder_VoteScoreDesc,
	k_EUserUGCListSortOrder_ForModeration,
};

// Combination of sorting and filtering for queries across all UGC
enum EUGCQuery
{
	k_EUGCQuery_RankedByVote								  = 0,
	k_EUGCQuery_RankedByPublicationDate						  = 1,
	k_EUGCQuery_AcceptedForGameRankedByAcceptanceDate		  = 2,
	k_EUGCQuery_RankedByTrend								  = 3,
	k_EUGCQuery_FavoritedByFriendsRankedByPublicationDate	  = 4,
	k_EUGCQuery_CreatedByFriendsRankedByPublicationDate		  = 5,
	k_EUGCQuery_RankedByNumTimesReported					  = 6,
	k_EUGCQuery_CreatedByFollowedUsersRankedByPublicationDate = 7,
	k_EUGCQuery_NotYetRated									  = 8,
	k_EUGCQuery_RankedByTotalVotesAsc						  = 9,
	k_EUGCQuery_RankedByVotesUp								  = 10,
	k_EUGCQuery_RankedByTextSearch							  = 11,
	k_EUGCQuery_RankedByTotalUniqueSubscriptions			  = 12,
	k_EUGCQuery_RankedByPlaytimeTrend						  = 13,
	k_EUGCQuery_RankedByTotalPlaytime						  = 14,
	k_EUGCQuery_RankedByAveragePlaytimeTrend				  = 15,
	k_EUGCQuery_RankedByLifetimeAveragePlaytime				  = 16,
	k_EUGCQuery_RankedByPlaytimeSessionsTrend				  = 17,
	k_EUGCQuery_RankedByLifetimePlaytimeSessions			  = 18,
	k_EUGCQuery_RankedByLastUpdatedDate						  = 19,
};

enum EItemUpdateStatus
{
	k_EItemUpdateStatusInvalid 				= 0, // The item update handle was invalid, job might be finished, listen too SubmitItemUpdateResult_t
	k_EItemUpdateStatusPreparingConfig 		= 1, // The item update is processing configuration data
	k_EItemUpdateStatusPreparingContent		= 2, // The item update is reading and processing content files
	k_EItemUpdateStatusUploadingContent		= 3, // The item update is uploading content changes to Steam
	k_EItemUpdateStatusUploadingPreviewFile	= 4, // The item update is uploading new preview file image
	k_EItemUpdateStatusCommittingChanges	= 5  // The item update is committing all changes
};

enum EItemState
{
	k_EItemStateNone			= 0,	// item not tracked on client
	k_EItemStateSubscribed		= 1,	// current user is subscribed to this item. Not just cached.
	k_EItemStateLegacyItem		= 2,	// item was created with ISteamRemoteStorage
	k_EItemStateInstalled		= 4,	// item is installed and usable (but maybe out of date)
	k_EItemStateNeedsUpdate		= 8,	// items needs an update. Either because it's not installed yet or creator updated content
	k_EItemStateDownloading		= 16,	// item update is currently downloading
	k_EItemStateDownloadPending	= 32,	// DownloadItem() was called for this item, content isn't available until DownloadItemResult_t is fired
};

enum EItemStatistic
{
	k_EItemStatistic_NumSubscriptions					 = 0,
	k_EItemStatistic_NumFavorites						 = 1,
	k_EItemStatistic_NumFollowers						 = 2,
	k_EItemStatistic_NumUniqueSubscriptions				 = 3,
	k_EItemStatistic_NumUniqueFavorites					 = 4,
	k_EItemStatistic_NumUniqueFollowers					 = 5,
	k_EItemStatistic_NumUniqueWebsiteViews				 = 6,
	k_EItemStatistic_ReportScore						 = 7,
	k_EItemStatistic_NumSecondsPlayed					 = 8,
	k_EItemStatistic_NumPlaytimeSessions				 = 9,
	k_EItemStatistic_NumComments						 = 10,
	k_EItemStatistic_NumSecondsPlayedDuringTimePeriod	 = 11,
	k_EItemStatistic_NumPlaytimeSessionsDuringTimePeriod = 12,
};

enum EItemPreviewType
{
	k_EItemPreviewType_Image							= 0,	// standard image file expected (e.g. jpg, png, gif, etc.)
	k_EItemPreviewType_YouTubeVideo						= 1,	// video id is stored
	k_EItemPreviewType_Sketchfab						= 2,	// model id is stored
	k_EItemPreviewType_EnvironmentMap_HorizontalCross	= 3,	// standard image file expected - cube map in the layout
																// +---+---+-------+
																// |   |Up |       |
																// +---+---+---+---+
																// | L | F | R | B |
																// +---+---+---+---+
																// |   |Dn |       |
																// +---+---+---+---+
	k_EItemPreviewType_EnvironmentMap_LatLong			= 4,	// standard image file expected
	k_EItemPreviewType_ReservedMax						= 255,	// you can specify your own types above this value
};

const uint32 kNumUGCResultsPerPage = 50;
const uint32 k_cchDeveloperMetadataMax = 5000;

// Details for a single published file/UGC
struct SteamUGCDetails_t
{
	PublishedFileId_t m_nPublishedFileId;
	EResult m_eResult;												// The result of the operation.	
	EWorkshopFileType m_eFileType;									// Type of the file
	AppId_t m_nCreatorAppID;										// ID of the app that created this file.
	AppId_t m_nConsumerAppID;										// ID of the app that will consume this file.
	char m_rgchTitle[k_cchPublishedDocumentTitleMax];				// title of document
	char m_rgchDescription[k_cchPublishedDocumentDescriptionMax];	// description of document
	uint64 m_ulSteamIDOwner;										// Steam ID of the user who created this content.
	uint32 m_rtimeCreated;											// time when the published file was created
	uint32 m_rtimeUpdated;											// time when the published file was last updated
	uint32 m_rtimeAddedToUserList;									// time when the user added the published file to their list (not always applicable)
	ERemoteStoragePublishedFileVisibility m_eVisibility;			// visibility
	bool m_bBanned;													// whether the file was banned
	bool m_bAcceptedForUse;											// developer has specifically flagged this item as accepted in the Workshop
	bool m_bTagsTruncated;											// whether the list of tags was too long to be returned in the provided buffer
	char m_rgchTags[k_cchTagListMax];								// comma separated list of all tags associated with this file	
	// file/url information
	UGCHandle_t m_hFile;											// The handle of the primary file
	UGCHandle_t m_hPreviewFile;										// The handle of the preview file
	char m_pchFileName[k_cchFilenameMax];							// The cloud filename of the primary file
	int32 m_nFileSize;												// Size of the primary file
	int32 m_nPreviewFileSize;										// Size of the preview file
	char m_rgchURL[k_cchPublishedFileURLMax];						// URL (for a video or a website)
	// voting information
	uint32 m_unVotesUp;												// number of votes up
	uint32 m_unVotesDown;											// number of votes down
	float m_flScore;												// calculated score
	// collection details
	uint32 m_unNumChildren;							
};

//-----------------------------------------------------------------------------
// Purpose: Steam UGC support API
//-----------------------------------------------------------------------------
class ISteamUGC
{
public:

	// Query UGC associated with a user. Creator app id or consumer app id must be valid and be set to the current running app. unPage should start at 1.
	virtual UGCQueryHandle_t CreateQueryUserUGCRequest( AccountID_t unAccountID, EUserUGCList eListType, EUGCMatchingUGCType eMatchingUGCType, EUserUGCListSortOrder eSortOrder, AppId_t nCreatorAppID, AppId_t nConsumerAppID, uint32 unPage ) = 0;

	// Query for all matching UGC. Creator app id or consumer app id must be valid and be set to the current running app. unPage should start at 1.
	STEAM_FLAT_NAME( CreateQueryAllUGCRequestPage )
	virtual UGCQueryHandle_t CreateQueryAllUGCRequest( EUGCQuery eQueryType, EUGCMatchingUGCType eMatchingeMatchingUGCTypeFileType, AppId_t nCreatorAppID, AppId_t nConsumerAppID, uint32 unPage ) = 0;

	// Query for all matching UGC using the new deep paging interface. Creator app id or consumer app id must be valid and be set to the current running app. pchCursor should be set to NULL or "*" to get the first result set.
	STEAM_FLAT_NAME( CreateQueryAllUGCRequestCursor )
	virtual UGCQueryHandle_t CreateQueryAllUGCRequest( EUGCQuery eQueryType, EUGCMatchingUGCType eMatchingeMatchingUGCTypeFileType, AppId_t nCreatorAppID, AppId_t nConsumerAppID, const char *pchCursor = NULL ) = 0;

	// Query for the details of the given published file ids (the RequestUGCDetails call is deprecated and replaced with this)
	virtual UGCQueryHandle_t CreateQueryUGCDetailsRequest( PublishedFileId_t *pvecPublishedFileID, uint32 unNumPublishedFileIDs ) = 0;

	// Send the query to Steam
	STEAM_CALL_RESULT( SteamUGCQueryCompleted_t )
	virtual SteamAPICall_t SendQueryUGCRequest( UGCQueryHandle_t handle ) = 0;

	// Retrieve an individual result after receiving the callback for querying UGC
	virtual bool GetQueryUGCResult( UGCQueryHandle_t handle, uint32 index, SteamUGCDetails_t *pDetails ) = 0;
	virtual uint32 GetQueryUGCNumTags( UGCQueryHandle_t handle, uint32 index ) = 0;
	virtual bool GetQueryUGCTag( UGCQueryHandle_t handle, uint32 index, uint32 indexTag, STEAM_OUT_STRING_COUNT( cchValueSize ) char* pchValue, uint32 cchValueSize ) = 0;
	virtual bool GetQueryUGCTagDisplayName( UGCQueryHandle_t handle, uint32 index, uint32 indexTag, STEAM_OUT_STRING_COUNT( cchValueSize ) char* pchValue, uint32 cchValueSize ) = 0;
	virtual bool GetQueryUGCPreviewURL( UGCQueryHandle_t handle, uint32 index, STEAM_OUT_STRING_COUNT(cchURLSize) char *pchURL, uint32 cchURLSize ) = 0;
	virtual bool GetQueryUGCMetadata( UGCQueryHandle_t handle, uint32 index, STEAM_OUT_STRING_COUNT(cchMetadatasize) char *pchMetadata, uint32 cchMetadatasize ) = 0;
	virtual bool GetQueryUGCChildren( UGCQueryHandle_t handle, uint32 index, PublishedFileId_t* pvecPublishedFileID, uint32 cMaxEntries ) = 0;
	virtual bool GetQueryUGCStatistic( UGCQueryHandle_t handle, uint32 index, EItemStatistic eStatType, uint64 *pStatValue ) = 0;
	virtual uint32 GetQueryUGCNumAdditionalPreviews( UGCQueryHandle_t handle, uint32 index ) = 0;
	virtual bool GetQueryUGCAdditionalPreview( UGCQueryHandle_t handle, uint32 index, uint32 previewIndex, STEAM_OUT_STRING_COUNT(cchURLSize) char *pchURLOrVideoID, uint32 cchURLSize, STEAM_OUT_STRING_COUNT(cchURLSize) char *pchOriginalFileName, uint32 cchOriginalFileNameSize, EItemPreviewType *pPreviewType ) = 0;
	virtual uint32 GetQueryUGCNumKeyValueTags( UGCQueryHandle_t handle, uint32 index ) = 0;
	virtual bool GetQueryUGCKeyValueTag( UGCQueryHandle_t handle, uint32 index, uint32 keyValueTagIndex, STEAM_OUT_STRING_COUNT(cchKeySize) char *pchKey, uint32 cchKeySize, STEAM_OUT_STRING_COUNT(cchValueSize) char *pchValue, uint32 cchValueSize ) = 0;

	// Return the first value matching the pchKey. Note that a key may map to multiple values.  Returns false if there was an error or no matching value was found.
	STEAM_FLAT_NAME( GetQueryFirstUGCKeyValueTag )
	virtual bool GetQueryUGCKeyValueTag( UGCQueryHandle_t handle, uint32 index, const char *pchKey, STEAM_OUT_STRING_COUNT(cchValueSize) char *pchValue, uint32 cchValueSize ) = 0;

	// Release the request to free up memory, after retrieving results
	virtual bool ReleaseQueryUGCRequest( UGCQueryHandle_t handle ) = 0;

	// Options to set for querying UGC
	virtual bool AddRequiredTag( UGCQueryHandle_t handle, const char *pTagName ) = 0;
	virtual bool AddRequiredTagGroup( UGCQueryHandle_t handle, const SteamParamStringArray_t *pTagGroups ) = 0; // match any of the tags in this group
	virtual bool AddExcludedTag( UGCQueryHandle_t handle, const char *pTagName ) = 0;
	virtual bool SetReturnOnlyIDs( UGCQueryHandle_t handle, bool bReturnOnlyIDs ) = 0;
	virtual bool SetReturnKeyValueTags( UGCQueryHandle_t handle, bool bReturnKeyValueTags ) = 0;
	virtual bool SetReturnLongDescription( UGCQueryHandle_t handle, bool bReturnLongDescription ) = 0;
	virtual bool SetReturnMetadata( UGCQueryHandle_t handle, bool bReturnMetadata ) = 0;
	virtual bool SetReturnChildren( UGCQueryHandle_t handle, bool bReturnChildren ) = 0;
	virtual bool SetReturnAdditionalPreviews( UGCQueryHandle_t handle, bool bReturnAdditionalPreviews ) = 0;
	virtual bool SetReturnTotalOnly( UGCQueryHandle_t handle, bool bReturnTotalOnly ) = 0;
	virtual bool SetReturnPlaytimeStats( UGCQueryHandle_t handle, uint32 unDays ) = 0;
	virtual bool SetLanguage( UGCQueryHandle_t handle, const char *pchLanguage ) = 0;
	virtual bool SetAllowCachedResponse( UGCQueryHandle_t handle, uint32 unMaxAgeSeconds ) = 0;

	// Options only for querying user UGC
	virtual bool SetCloudFileNameFilter( UGCQueryHandle_t handle, const char *pMatchCloudFileName ) = 0;

	// Options only for querying all UGC
	virtual bool SetMatchAnyTag( UGCQueryHandle_t handle, bool bMatchAnyTag ) = 0;
	virtual bool SetSearchText( UGCQueryHandle_t handle, const char *pSearchText ) = 0;
	virtual bool SetRankedByTrendDays( UGCQueryHandle_t handle, uint32 unDays ) = 0;
	virtual bool SetTimeCreatedDateRange( UGCQueryHandle_t handle, RTime32 rtStart, RTime32 rtEnd ) = 0;
	virtual bool SetTimeUpdatedDateRange( UGCQueryHandle_t handle, RTime32 rtStart, RTime32 rtEnd ) = 0;
	virtual bool AddRequiredKeyValueTag( UGCQueryHandle_t handle, const char *pKey, const char *pValue ) = 0;

	// DEPRECATED - Use CreateQueryUGCDetailsRequest call above instead!
	STEAM_CALL_RESULT( SteamUGCRequestUGCDetailsResult_t )
	virtual SteamAPICall_t RequestUGCDetails( PublishedFileId_t nPublishedFileID, uint32 unMaxAgeSeconds ) = 0;

	// Steam Workshop Creator API
	STEAM_CALL_RESULT( CreateItemResult_t )
	virtual SteamAPICall_t CreateItem( AppId_t nConsumerAppId, EWorkshopFileType eFileType ) = 0; // create new item for this app with no content attached yet

	virtual UGCUpdateHandle_t StartItemUpdate( AppId_t nConsumerAppId, PublishedFileId_t nPublishedFileID ) = 0; // start an UGC item update. Set changed properties before commiting update with CommitItemUpdate()

	virtual bool SetItemTitle( UGCUpdateHandle_t handle, const char *pchTitle ) = 0; // change the title of an UGC item
	virtual bool SetItemDescription( UGCUpdateHandle_t handle, const char *pchDescription ) = 0; // change the description of an UGC item
	virtual bool SetItemUpdateLanguage( UGCUpdateHandle_t handle, const char *pchLanguage ) = 0; // specify the language of the title or description that will be set
	virtual bool SetItemMetadata( UGCUpdateHandle_t handle, const char *pchMetaData ) = 0; // change the metadata of an UGC item (max = k_cchDeveloperMetadataMax)
	virtual bool SetItemVisibility( UGCUpdateHandle_t handle, ERemoteStoragePublishedFileVisibility eVisibility ) = 0; // change the visibility of an UGC item
	virtual bool SetItemTags( UGCUpdateHandle_t updateHandle, const SteamParamStringArray_t *pTags ) = 0; // change the tags of an UGC item
	virtual bool SetItemContent( UGCUpdateHandle_t handle, const char *pszContentFolder ) = 0; // update item content from this local folder
	virtual bool SetItemPreview( UGCUpdateHandle_t handle, const char *pszPreviewFile ) = 0; //  change preview image file for this item. pszPreviewFile points to local image file, which must be under 1MB in size
	virtual bool SetAllowLegacyUpload( UGCUpdateHandle_t handle, bool bAllowLegacyUpload ) = 0; //  use legacy upload for a single small file. The parameter to SetItemContent() should either be a directory with one file or the full path to the file.  The file must also be less than 10MB in size.
	virtual bool RemoveAllItemKeyValueTags( UGCUpdateHandle_t handle ) = 0; // remove all existing key-value tags (you can add new ones via the AddItemKeyValueTag function)
	virtual bool RemoveItemKeyValueTags( UGCUpdateHandle_t handle, const char *pchKey ) = 0; // remove any existing key-value tags with the specified key
	virtual bool AddItemKeyValueTag( UGCUpdateHandle_t handle, const char *pchKey, const char *pchValue ) = 0; // add new key-value tags for the item. Note that there can be multiple values for a tag.
	virtual bool AddItemPreviewFile( UGCUpdateHandle_t handle, const char *pszPreviewFile, EItemPreviewType type ) = 0; //  add preview file for this item. pszPreviewFile points to local file, which must be under 1MB in size
	virtual bool AddItemPreviewVideo( UGCUpdateHandle_t handle, const char *pszVideoID ) = 0; //  add preview video for this item
	virtual bool UpdateItemPreviewFile( UGCUpdateHandle_t handle, uint32 index, const char *pszPreviewFile ) = 0; //  updates an existing preview file for this item. pszPreviewFile points to local file, which must be under 1MB in size
	virtual bool UpdateItemPreviewVideo( UGCUpdateHandle_t handle, uint32 index, const char *pszVideoID ) = 0; //  updates an existing preview video for this item
	virtual bool RemoveItemPreview( UGCUpdateHandle_t handle, uint32 index ) = 0; // remove a preview by index starting at 0 (previews are sorted)

	STEAM_CALL_RESULT( SubmitItemUpdateResult_t )
	virtual SteamAPICall_t SubmitItemUpdate( UGCUpdateHandle_t handle, const char *pchChangeNote ) = 0; // commit update process started with StartItemUpdate()
	virtual EItemUpdateStatus GetItemUpdateProgress( UGCUpdateHandle_t handle, uint64 *punBytesProcessed, uint64* punBytesTotal ) = 0;

	// Steam Workshop Consumer API
	STEAM_CALL_RESULT( SetUserItemVoteResult_t )
	virtual SteamAPICall_t SetUserItemVote( PublishedFileId_t nPublishedFileID, bool bVoteUp ) = 0;
	STEAM_CALL_RESULT( GetUserItemVoteResult_t )
	virtual SteamAPICall_t GetUserItemVote( PublishedFileId_t nPublishedFileID ) = 0;
	STEAM_CALL_RESULT( UserFavoriteItemsListChanged_t )
	virtual SteamAPICall_t AddItemToFavorites( AppId_t nAppId, PublishedFileId_t nPublishedFileID ) = 0;
	STEAM_CALL_RESULT( UserFavoriteItemsListChanged_t )
	virtual SteamAPICall_t RemoveItemFromFavorites( AppId_t nAppId, PublishedFileId_t nPublishedFileID ) = 0;
	STEAM_CALL_RESULT( RemoteStorageSubscribePublishedFileResult_t )
	virtual SteamAPICall_t SubscribeItem( PublishedFileId_t nPublishedFileID ) = 0; // subscribe to this item, will be installed ASAP
	STEAM_CALL_RESULT( RemoteStorageUnsubscribePublishedFileResult_t )
	virtual SteamAPICall_t UnsubscribeItem( PublishedFileId_t nPublishedFileID ) = 0; // unsubscribe from this item, will be uninstalled after game quits
	virtual uint32 GetNumSubscribedItems() = 0; // number of subscribed items 
	virtual uint32 GetSubscribedItems( PublishedFileId_t* pvecPublishedFileID, uint32 cMaxEntries ) = 0; // all subscribed item PublishFileIDs

	// get EItemState flags about item on this client
	virtual uint32 GetItemState( PublishedFileId_t nPublishedFileID ) = 0;

	// get info about currently installed content on disc for items that have k_EItemStateInstalled set
	// if k_EItemStateLegacyItem is set, pchFolder contains the path to the legacy file itself (not a folder)
	virtual bool GetItemInstallInfo( PublishedFileId_t nPublishedFileID, uint64 *punSizeOnDisk, STEAM_OUT_STRING_COUNT( cchFolderSize ) char *pchFolder, uint32 cchFolderSize, uint32 *punTimeStamp ) = 0;

	// get info about pending update for items that have k_EItemStateNeedsUpdate set. punBytesTotal will be valid after download started once
	virtual bool GetItemDownloadInfo( PublishedFileId_t nPublishedFileID, uint64 *punBytesDownloaded, uint64 *punBytesTotal ) = 0;
		
	// download new or update already installed item. If function returns true, wait for DownloadItemResult_t. If the item is already installed,
	// then files on disk should not be used until callback received. If item is not subscribed to, it will be cached for some time.
	// If bHighPriority is set, any other item download will be suspended and this item downloaded ASAP.
	virtual bool DownloadItem( PublishedFileId_t nPublishedFileID, bool bHighPriority ) = 0;

	// game servers can set a specific workshop folder before issuing any UGC commands.
	// This is helpful if you want to support multiple game servers running out of the same install folder
	virtual bool BInitWorkshopForGameServer( DepotId_t unWorkshopDepotID, const char *pszFolder ) = 0;

	// SuspendDownloads( true ) will suspend all workshop downloads until SuspendDownloads( false ) is called or the game ends
	virtual void SuspendDownloads( bool bSuspend ) = 0;

	// usage tracking
	STEAM_CALL_RESULT( StartPlaytimeTrackingResult_t )
	virtual SteamAPICall_t StartPlaytimeTracking( PublishedFileId_t *pvecPublishedFileID, uint32 unNumPublishedFileIDs ) = 0;
	STEAM_CALL_RESULT( StopPlaytimeTrackingResult_t )
	virtual SteamAPICall_t StopPlaytimeTracking( PublishedFileId_t *pvecPublishedFileID, uint32 unNumPublishedFileIDs ) = 0;
	STEAM_CALL_RESULT( StopPlaytimeTrackingResult_t )
	virtual SteamAPICall_t StopPlaytimeTrackingForAllItems() = 0;

	// parent-child relationship or dependency management
	STEAM_CALL_RESULT( AddUGCDependencyResult_t )
	virtual SteamAPICall_t AddDependency( PublishedFileId_t nParentPublishedFileID, PublishedFileId_t nChildPublishedFileID ) = 0;
	STEAM_CALL_RESULT( RemoveUGCDependencyResult_t )
	virtual SteamAPICall_t RemoveDependency( PublishedFileId_t nParentPublishedFileID, PublishedFileId_t nChildPublishedFileID ) = 0;

	// add/remove app dependence/requirements (usually DLC)
	STEAM_CALL_RESULT( AddAppDependencyResult_t )
	virtual SteamAPICall_t AddAppDependency( PublishedFileId_t nPublishedFileID, AppId_t nAppID ) = 0;
	STEAM_CALL_RESULT( RemoveAppDependencyResult_t )
	virtual SteamAPICall_t RemoveAppDependency( PublishedFileId_t nPublishedFileID, AppId_t nAppID ) = 0;
	// request app dependencies. note that whatever callback you register for GetAppDependenciesResult_t may be called multiple times
	// until all app dependencies have been returned
	STEAM_CALL_RESULT( GetAppDependenciesResult_t )
	virtual SteamAPICall_t GetAppDependencies( PublishedFileId_t nPublishedFileID ) = 0;
	
	// delete the item without prompting the user
	STEAM_CALL_RESULT( DeleteItemResult_t )
	virtual SteamAPICall_t DeleteItem( PublishedFileId_t nPublishedFileID ) = 0;

	// Show the app's latest Workshop EULA to the user in an overlay window, where they can accept it or not
	virtual bool ShowWorkshopEULA() = 0;
	// Retrieve information related to the user's acceptance or not of the app's specific Workshop EULA
	STEAM_CALL_RESULT( WorkshopEULAStatus_t )
	virtual SteamAPICall_t GetWorkshopEULAStatus() = 0;
};

#define STEAMUGC_INTERFACE_VERSION "STEAMUGC_INTERFACE_VERSION016"

// Global interface accessor
inline ISteamUGC *SteamUGC();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamUGC *, SteamUGC, STEAMUGC_INTERFACE_VERSION );

// Global accessor for the gameserver client
inline ISteamUGC *SteamGameServerUGC();
STEAM_DEFINE_GAMESERVER_INTERFACE_ACCESSOR( ISteamUGC *, SteamGameServerUGC, STEAMUGC_INTERFACE_VERSION );

//-----------------------------------------------------------------------------
// Purpose: Callback for querying UGC
//-----------------------------------------------------------------------------
struct SteamUGCQueryCompleted_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 1 };
	UGCQueryHandle_t m_handle;
	EResult m_eResult;
	uint32 m_unNumResultsReturned;
	uint32 m_unTotalMatchingResults;
	bool m_bCachedData;	// indicates whether this data was retrieved from the local on-disk cache
	char m_rgchNextCursor[k_cchPublishedFileURLMax]; // If a paging cursor was used, then this will be the next cursor to get the next result set.
};


//-----------------------------------------------------------------------------
// Purpose: Callback for requesting details on one piece of UGC
//-----------------------------------------------------------------------------
struct SteamUGCRequestUGCDetailsResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 2 };
	SteamUGCDetails_t m_details;
	bool m_bCachedData; // indicates whether this data was retrieved from the local on-disk cache
};


//-----------------------------------------------------------------------------
// Purpose: result for ISteamUGC::CreateItem() 
//-----------------------------------------------------------------------------
struct CreateItemResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 3 };
	EResult m_eResult;
	PublishedFileId_t m_nPublishedFileId; // new item got this UGC PublishFileID
	bool m_bUserNeedsToAcceptWorkshopLegalAgreement;
};


//-----------------------------------------------------------------------------
// Purpose: result for ISteamUGC::SubmitItemUpdate() 
//-----------------------------------------------------------------------------
struct SubmitItemUpdateResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 4 };
	EResult m_eResult;
	bool m_bUserNeedsToAcceptWorkshopLegalAgreement;
	PublishedFileId_t m_nPublishedFileId;
};


//-----------------------------------------------------------------------------
// Purpose: a Workshop item has been installed or updated
//-----------------------------------------------------------------------------
struct ItemInstalled_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 5 };
	AppId_t m_unAppID;
	PublishedFileId_t m_nPublishedFileId;
};


//-----------------------------------------------------------------------------
// Purpose: result of DownloadItem(), existing item files can be accessed again
//-----------------------------------------------------------------------------
struct DownloadItemResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 6 };
	AppId_t m_unAppID;
	PublishedFileId_t m_nPublishedFileId;
	EResult m_eResult;
};

//-----------------------------------------------------------------------------
// Purpose: result of AddItemToFavorites() or RemoveItemFromFavorites()
//-----------------------------------------------------------------------------
struct UserFavoriteItemsListChanged_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 7 };
	PublishedFileId_t m_nPublishedFileId;
	EResult m_eResult;
	bool m_bWasAddRequest;
};

//-----------------------------------------------------------------------------
// Purpose: The result of a call to SetUserItemVote()
//-----------------------------------------------------------------------------
struct SetUserItemVoteResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 8 };
	PublishedFileId_t m_nPublishedFileId;
	EResult m_eResult;
	bool m_bVoteUp;
};

//-----------------------------------------------------------------------------
// Purpose: The result of a call to GetUserItemVote()
//-----------------------------------------------------------------------------
struct GetUserItemVoteResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 9 };
	PublishedFileId_t m_nPublishedFileId;
	EResult m_eResult;
	bool m_bVotedUp;
	bool m_bVotedDown;
	bool m_bVoteSkipped;
};

//-----------------------------------------------------------------------------
// Purpose: The result of a call to StartPlaytimeTracking()
//-----------------------------------------------------------------------------
struct StartPlaytimeTrackingResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 10 };
	EResult m_eResult;
};

//-----------------------------------------------------------------------------
// Purpose: The result of a call to StopPlaytimeTracking()
//-----------------------------------------------------------------------------
struct StopPlaytimeTrackingResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 11 };
	EResult m_eResult;
};

//-----------------------------------------------------------------------------
// Purpose: The result of a call to AddDependency
//-----------------------------------------------------------------------------
struct AddUGCDependencyResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 12 };
	EResult m_eResult;
	PublishedFileId_t m_nPublishedFileId;
	PublishedFileId_t m_nChildPublishedFileId;
};

//-----------------------------------------------------------------------------
// Purpose: The result of a call to RemoveDependency
//-----------------------------------------------------------------------------
struct RemoveUGCDependencyResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 13 };
	EResult m_eResult;
	PublishedFileId_t m_nPublishedFileId;
	PublishedFileId_t m_nChildPublishedFileId;
};


//-----------------------------------------------------------------------------
// Purpose: The result of a call to AddAppDependency
//-----------------------------------------------------------------------------
struct AddAppDependencyResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 14 };
	EResult m_eResult;
	PublishedFileId_t m_nPublishedFileId;
	AppId_t m_nAppID;
};

//-----------------------------------------------------------------------------
// Purpose: The result of a call to RemoveAppDependency
//-----------------------------------------------------------------------------
struct RemoveAppDependencyResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 15 };
	EResult m_eResult;
	PublishedFileId_t m_nPublishedFileId;
	AppId_t m_nAppID;
};

//-----------------------------------------------------------------------------
// Purpose: The result of a call to GetAppDependencies.  Callback may be called
//			multiple times until all app dependencies have been returned.
//-----------------------------------------------------------------------------
struct GetAppDependenciesResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 16 };
	EResult m_eResult;
	PublishedFileId_t m_nPublishedFileId;
	AppId_t m_rgAppIDs[32];
	uint32 m_nNumAppDependencies;		// number returned in this struct
	uint32 m_nTotalNumAppDependencies;	// total found
};

//-----------------------------------------------------------------------------
// Purpose: The result of a call to DeleteItem
//-----------------------------------------------------------------------------
struct DeleteItemResult_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 17 };
	EResult m_eResult;
	PublishedFileId_t m_nPublishedFileId;
};


//-----------------------------------------------------------------------------
// Purpose: signal that the list of subscribed items changed
//-----------------------------------------------------------------------------
struct UserSubscribedItemsListChanged_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 18 };
	AppId_t m_nAppID;
};


//-----------------------------------------------------------------------------
// Purpose: Status of the user's acceptable/rejection of the app's specific Workshop EULA
//-----------------------------------------------------------------------------
struct WorkshopEULAStatus_t
{
	enum { k_iCallback = k_iSteamUGCCallbacks + 20 };
	EResult m_eResult;
	AppId_t m_nAppID;
	uint32 m_unVersion;
	RTime32 m_rtAction;
	bool m_bAccepted;
	bool m_bNeedsAction;
};

#pragma pack( pop )

#endif // ISTEAMUGC_H
