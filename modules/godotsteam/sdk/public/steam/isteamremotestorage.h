//====== Copyright � 1996-2008, Valve Corporation, All rights reserved. =======
//
// Purpose: public interface to user remote file storage in Steam
//
//=============================================================================

#ifndef ISTEAMREMOTESTORAGE_H
#define ISTEAMREMOTESTORAGE_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"


//-----------------------------------------------------------------------------
// Purpose: Defines the largest allowed file size. Cloud files cannot be written
// in a single chunk over 100MB (and cannot be over 200MB total.)
//-----------------------------------------------------------------------------
const uint32 k_unMaxCloudFileChunkSize = 100 * 1024 * 1024;


//-----------------------------------------------------------------------------
// Purpose: Structure that contains an array of const char * strings and the number of those strings
//-----------------------------------------------------------------------------
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 
struct SteamParamStringArray_t
{
	const char ** m_ppStrings;
	int32 m_nNumStrings;
};
#pragma pack( pop )

// A handle to a piece of user generated content
typedef uint64 UGCHandle_t;
typedef uint64 PublishedFileUpdateHandle_t;
typedef uint64 PublishedFileId_t;
const PublishedFileId_t k_PublishedFileIdInvalid = 0;
const UGCHandle_t k_UGCHandleInvalid = 0xffffffffffffffffull;
const PublishedFileUpdateHandle_t k_PublishedFileUpdateHandleInvalid = 0xffffffffffffffffull;

// Handle for writing to Steam Cloud
typedef uint64 UGCFileWriteStreamHandle_t;
const UGCFileWriteStreamHandle_t k_UGCFileStreamHandleInvalid = 0xffffffffffffffffull;

const uint32 k_cchPublishedDocumentTitleMax = 128 + 1;
const uint32 k_cchPublishedDocumentDescriptionMax = 8000;
const uint32 k_cchPublishedDocumentChangeDescriptionMax = 8000;
const uint32 k_unEnumeratePublishedFilesMaxResults = 50;
const uint32 k_cchTagListMax = 1024 + 1;
const uint32 k_cchFilenameMax = 260;
const uint32 k_cchPublishedFileURLMax = 256;


enum ERemoteStoragePlatform
{
	k_ERemoteStoragePlatformNone		= 0,
	k_ERemoteStoragePlatformWindows		= (1 << 0),
	k_ERemoteStoragePlatformOSX			= (1 << 1),
	k_ERemoteStoragePlatformPS3			= (1 << 2),
	k_ERemoteStoragePlatformLinux		= (1 << 3),
	k_ERemoteStoragePlatformSwitch		= (1 << 4),
	k_ERemoteStoragePlatformAndroid		= (1 << 5),
	k_ERemoteStoragePlatformIOS			= (1 << 6),
	// NB we get one more before we need to widen some things

	k_ERemoteStoragePlatformAll = 0xffffffff
};

enum ERemoteStoragePublishedFileVisibility
{
	k_ERemoteStoragePublishedFileVisibilityPublic = 0,
	k_ERemoteStoragePublishedFileVisibilityFriendsOnly = 1,
	k_ERemoteStoragePublishedFileVisibilityPrivate = 2,
	k_ERemoteStoragePublishedFileVisibilityUnlisted = 3,
};


enum EWorkshopFileType
{
	k_EWorkshopFileTypeFirst = 0,

	k_EWorkshopFileTypeCommunity			  = 0,		// normal Workshop item that can be subscribed to
	k_EWorkshopFileTypeMicrotransaction		  = 1,		// Workshop item that is meant to be voted on for the purpose of selling in-game
	k_EWorkshopFileTypeCollection			  = 2,		// a collection of Workshop or Greenlight items
	k_EWorkshopFileTypeArt					  = 3,		// artwork
	k_EWorkshopFileTypeVideo				  = 4,		// external video
	k_EWorkshopFileTypeScreenshot			  = 5,		// screenshot
	k_EWorkshopFileTypeGame					  = 6,		// Greenlight game entry
	k_EWorkshopFileTypeSoftware				  = 7,		// Greenlight software entry
	k_EWorkshopFileTypeConcept				  = 8,		// Greenlight concept
	k_EWorkshopFileTypeWebGuide				  = 9,		// Steam web guide
	k_EWorkshopFileTypeIntegratedGuide		  = 10,		// application integrated guide
	k_EWorkshopFileTypeMerch				  = 11,		// Workshop merchandise meant to be voted on for the purpose of being sold
	k_EWorkshopFileTypeControllerBinding	  = 12,		// Steam Controller bindings
	k_EWorkshopFileTypeSteamworksAccessInvite = 13,		// internal
	k_EWorkshopFileTypeSteamVideo			  = 14,		// Steam video
	k_EWorkshopFileTypeGameManagedItem		  = 15,		// managed completely by the game, not the user, and not shown on the web

	// Update k_EWorkshopFileTypeMax if you add values.
	k_EWorkshopFileTypeMax = 16
	
};

enum EWorkshopVote
{
	k_EWorkshopVoteUnvoted = 0,
	k_EWorkshopVoteFor = 1,
	k_EWorkshopVoteAgainst = 2,
	k_EWorkshopVoteLater = 3,
};

enum EWorkshopFileAction
{
	k_EWorkshopFileActionPlayed = 0,
	k_EWorkshopFileActionCompleted = 1,
};

enum EWorkshopEnumerationType
{
	k_EWorkshopEnumerationTypeRankedByVote = 0,
	k_EWorkshopEnumerationTypeRecent = 1,
	k_EWorkshopEnumerationTypeTrending = 2,
	k_EWorkshopEnumerationTypeFavoritesOfFriends = 3,
	k_EWorkshopEnumerationTypeVotedByFriends = 4,
	k_EWorkshopEnumerationTypeContentByFriends = 5,
	k_EWorkshopEnumerationTypeRecentFromFollowedUsers = 6,
};

enum EWorkshopVideoProvider
{
	k_EWorkshopVideoProviderNone = 0,
	k_EWorkshopVideoProviderYoutube = 1
};


enum EUGCReadAction
{
	// Keeps the file handle open unless the last byte is read.  You can use this when reading large files (over 100MB) in sequential chunks.
	// If the last byte is read, this will behave the same as k_EUGCRead_Close.  Otherwise, it behaves the same as k_EUGCRead_ContinueReading.
	// This value maintains the same behavior as before the EUGCReadAction parameter was introduced.
	k_EUGCRead_ContinueReadingUntilFinished = 0,

	// Keeps the file handle open.  Use this when using UGCRead to seek to different parts of the file.
	// When you are done seeking around the file, make a final call with k_EUGCRead_Close to close it.
	k_EUGCRead_ContinueReading = 1,

	// Frees the file handle.  Use this when you're done reading the content.  
	// To read the file from Steam again you will need to call UGCDownload again. 
	k_EUGCRead_Close = 2,	
};

enum ERemoteStorageLocalFileChange
{
	k_ERemoteStorageLocalFileChange_Invalid = 0,

	// The file was updated from another device
	k_ERemoteStorageLocalFileChange_FileUpdated = 1,

	// The file was deleted by another device
	k_ERemoteStorageLocalFileChange_FileDeleted = 2,
};

enum ERemoteStorageFilePathType
{
	k_ERemoteStorageFilePathType_Invalid = 0,
	
	// The file is directly accessed by the game and this is the full path
	k_ERemoteStorageFilePathType_Absolute = 1,

	// The file is accessed via the ISteamRemoteStorage API and this is the filename
	k_ERemoteStorageFilePathType_APIFilename = 2,
};


//-----------------------------------------------------------------------------
// Purpose: Functions for accessing, reading and writing files stored remotely 
//			and cached locally
//-----------------------------------------------------------------------------
class ISteamRemoteStorage
{
	public:
		// NOTE
		//
		// Filenames are case-insensitive, and will be converted to lowercase automatically.
		// So "foo.bar" and "Foo.bar" are the same file, and if you write "Foo.bar" then
		// iterate the files, the filename returned will be "foo.bar".
		//

		// file operations
		virtual bool	FileWrite( const char *pchFile, const void *pvData, int32 cubData ) = 0;
		virtual int32	FileRead( const char *pchFile, void *pvData, int32 cubDataToRead ) = 0;
		
		STEAM_CALL_RESULT( RemoteStorageFileWriteAsyncComplete_t )
		virtual SteamAPICall_t FileWriteAsync( const char *pchFile, const void *pvData, uint32 cubData ) = 0;
		
		STEAM_CALL_RESULT( RemoteStorageFileReadAsyncComplete_t )
		virtual SteamAPICall_t FileReadAsync( const char *pchFile, uint32 nOffset, uint32 cubToRead ) = 0;
		virtual bool	FileReadAsyncComplete( SteamAPICall_t hReadCall, void *pvBuffer, uint32 cubToRead ) = 0;
		
		virtual bool	FileForget( const char *pchFile ) = 0;
		virtual bool	FileDelete( const char *pchFile ) = 0;
		STEAM_CALL_RESULT( RemoteStorageFileShareResult_t )
		virtual SteamAPICall_t FileShare( const char *pchFile ) = 0;
		virtual bool	SetSyncPlatforms( const char *pchFile, ERemoteStoragePlatform eRemoteStoragePlatform ) = 0;

		// file operations that cause network IO
		virtual UGCFileWriteStreamHandle_t FileWriteStreamOpen( const char *pchFile ) = 0;
		virtual bool FileWriteStreamWriteChunk( UGCFileWriteStreamHandle_t writeHandle, const void *pvData, int32 cubData ) = 0;
		virtual bool FileWriteStreamClose( UGCFileWriteStreamHandle_t writeHandle ) = 0;
		virtual bool FileWriteStreamCancel( UGCFileWriteStreamHandle_t writeHandle ) = 0;

		// file information
		virtual bool	FileExists( const char *pchFile ) = 0;
		virtual bool	FilePersisted( const char *pchFile ) = 0;
		virtual int32	GetFileSize( const char *pchFile ) = 0;
		virtual int64	GetFileTimestamp( const char *pchFile ) = 0;
		virtual ERemoteStoragePlatform GetSyncPlatforms( const char *pchFile ) = 0;

		// iteration
		virtual int32 GetFileCount() = 0;
		virtual const char *GetFileNameAndSize( int iFile, int32 *pnFileSizeInBytes ) = 0;

		// configuration management
		virtual bool GetQuota( uint64 *pnTotalBytes, uint64 *puAvailableBytes ) = 0;
		virtual bool IsCloudEnabledForAccount() = 0;
		virtual bool IsCloudEnabledForApp() = 0;
		virtual void SetCloudEnabledForApp( bool bEnabled ) = 0;

		// user generated content

		// Downloads a UGC file.  A priority value of 0 will download the file immediately,
		// otherwise it will wait to download the file until all downloads with a lower priority
		// value are completed.  Downloads with equal priority will occur simultaneously.
		STEAM_CALL_RESULT( RemoteStorageDownloadUGCResult_t )
		virtual SteamAPICall_t UGCDownload( UGCHandle_t hContent, uint32 unPriority ) = 0;
		
		// Gets the amount of data downloaded so far for a piece of content. pnBytesExpected can be 0 if function returns false
		// or if the transfer hasn't started yet, so be careful to check for that before dividing to get a percentage
		virtual bool	GetUGCDownloadProgress( UGCHandle_t hContent, int32 *pnBytesDownloaded, int32 *pnBytesExpected ) = 0;

		// Gets metadata for a file after it has been downloaded. This is the same metadata given in the RemoteStorageDownloadUGCResult_t call result
		virtual bool	GetUGCDetails( UGCHandle_t hContent, AppId_t *pnAppID, STEAM_OUT_STRING() char **ppchName, int32 *pnFileSizeInBytes, STEAM_OUT_STRUCT() CSteamID *pSteamIDOwner ) = 0;

		// After download, gets the content of the file.  
		// Small files can be read all at once by calling this function with an offset of 0 and cubDataToRead equal to the size of the file.
		// Larger files can be read in chunks to reduce memory usage (since both sides of the IPC client and the game itself must allocate
		// enough memory for each chunk).  Once the last byte is read, the file is implicitly closed and further calls to UGCRead will fail
		// unless UGCDownload is called again.
		// For especially large files (anything over 100MB) it is a requirement that the file is read in chunks.
		virtual int32	UGCRead( UGCHandle_t hContent, void *pvData, int32 cubDataToRead, uint32 cOffset, EUGCReadAction eAction ) = 0;

		// Functions to iterate through UGC that has finished downloading but has not yet been read via UGCRead()
		virtual int32	GetCachedUGCCount() = 0;
		virtual	UGCHandle_t GetCachedUGCHandle( int32 iCachedContent ) = 0;

		// publishing UGC
		STEAM_CALL_RESULT( RemoteStoragePublishFileProgress_t )
		virtual SteamAPICall_t	PublishWorkshopFile( const char *pchFile, const char *pchPreviewFile, AppId_t nConsumerAppId, const char *pchTitle, const char *pchDescription, ERemoteStoragePublishedFileVisibility eVisibility, SteamParamStringArray_t *pTags, EWorkshopFileType eWorkshopFileType ) = 0;
		virtual PublishedFileUpdateHandle_t CreatePublishedFileUpdateRequest( PublishedFileId_t unPublishedFileId ) = 0;
		virtual bool UpdatePublishedFileFile( PublishedFileUpdateHandle_t updateHandle, const char *pchFile ) = 0;
		virtual bool UpdatePublishedFilePreviewFile( PublishedFileUpdateHandle_t updateHandle, const char *pchPreviewFile ) = 0;
		virtual bool UpdatePublishedFileTitle( PublishedFileUpdateHandle_t updateHandle, const char *pchTitle ) = 0;
		virtual bool UpdatePublishedFileDescription( PublishedFileUpdateHandle_t updateHandle, const char *pchDescription ) = 0;
		virtual bool UpdatePublishedFileVisibility( PublishedFileUpdateHandle_t updateHandle, ERemoteStoragePublishedFileVisibility eVisibility ) = 0;
		virtual bool UpdatePublishedFileTags( PublishedFileUpdateHandle_t updateHandle, SteamParamStringArray_t *pTags ) = 0;
		STEAM_CALL_RESULT( RemoteStorageUpdatePublishedFileResult_t )
		virtual SteamAPICall_t	CommitPublishedFileUpdate( PublishedFileUpdateHandle_t updateHandle ) = 0;
		// Gets published file details for the given publishedfileid.  If unMaxSecondsOld is greater than 0,
		// cached data may be returned, depending on how long ago it was cached.  A value of 0 will force a refresh.
		// A value of k_WorkshopForceLoadPublishedFileDetailsFromCache will use cached data if it exists, no matter how old it is.
		STEAM_CALL_RESULT( RemoteStorageGetPublishedFileDetailsResult_t )
		virtual SteamAPICall_t	GetPublishedFileDetails( PublishedFileId_t unPublishedFileId, uint32 unMaxSecondsOld ) = 0;
		STEAM_CALL_RESULT( RemoteStorageDeletePublishedFileResult_t )
		virtual SteamAPICall_t	DeletePublishedFile( PublishedFileId_t unPublishedFileId ) = 0;
		// enumerate the files that the current user published with this app
		STEAM_CALL_RESULT( RemoteStorageEnumerateUserPublishedFilesResult_t )
		virtual SteamAPICall_t	EnumerateUserPublishedFiles( uint32 unStartIndex ) = 0;
		STEAM_CALL_RESULT( RemoteStorageSubscribePublishedFileResult_t )
		virtual SteamAPICall_t	SubscribePublishedFile( PublishedFileId_t unPublishedFileId ) = 0;
		STEAM_CALL_RESULT( RemoteStorageEnumerateUserSubscribedFilesResult_t )
		virtual SteamAPICall_t	EnumerateUserSubscribedFiles( uint32 unStartIndex ) = 0;
		STEAM_CALL_RESULT( RemoteStorageUnsubscribePublishedFileResult_t )
		virtual SteamAPICall_t	UnsubscribePublishedFile( PublishedFileId_t unPublishedFileId ) = 0;
		virtual bool UpdatePublishedFileSetChangeDescription( PublishedFileUpdateHandle_t updateHandle, const char *pchChangeDescription ) = 0;
		STEAM_CALL_RESULT( RemoteStorageGetPublishedItemVoteDetailsResult_t )
		virtual SteamAPICall_t	GetPublishedItemVoteDetails( PublishedFileId_t unPublishedFileId ) = 0;
		STEAM_CALL_RESULT( RemoteStorageUpdateUserPublishedItemVoteResult_t )
		virtual SteamAPICall_t	UpdateUserPublishedItemVote( PublishedFileId_t unPublishedFileId, bool bVoteUp ) = 0;
		STEAM_CALL_RESULT( RemoteStorageGetPublishedItemVoteDetailsResult_t )
		virtual SteamAPICall_t	GetUserPublishedItemVoteDetails( PublishedFileId_t unPublishedFileId ) = 0;
		STEAM_CALL_RESULT( RemoteStorageEnumerateUserPublishedFilesResult_t )
		virtual SteamAPICall_t	EnumerateUserSharedWorkshopFiles( CSteamID steamId, uint32 unStartIndex, SteamParamStringArray_t *pRequiredTags, SteamParamStringArray_t *pExcludedTags ) = 0;
		STEAM_CALL_RESULT( RemoteStoragePublishFileProgress_t )
		virtual SteamAPICall_t	PublishVideo( EWorkshopVideoProvider eVideoProvider, const char *pchVideoAccount, const char *pchVideoIdentifier, const char *pchPreviewFile, AppId_t nConsumerAppId, const char *pchTitle, const char *pchDescription, ERemoteStoragePublishedFileVisibility eVisibility, SteamParamStringArray_t *pTags ) = 0;
		STEAM_CALL_RESULT( RemoteStorageSetUserPublishedFileActionResult_t )
		virtual SteamAPICall_t	SetUserPublishedFileAction( PublishedFileId_t unPublishedFileId, EWorkshopFileAction eAction ) = 0;
		STEAM_CALL_RESULT( RemoteStorageEnumeratePublishedFilesByUserActionResult_t )
		virtual SteamAPICall_t	EnumeratePublishedFilesByUserAction( EWorkshopFileAction eAction, uint32 unStartIndex ) = 0;
		// this method enumerates the public view of workshop files
		STEAM_CALL_RESULT( RemoteStorageEnumerateWorkshopFilesResult_t )
		virtual SteamAPICall_t	EnumeratePublishedWorkshopFiles( EWorkshopEnumerationType eEnumerationType, uint32 unStartIndex, uint32 unCount, uint32 unDays, SteamParamStringArray_t *pTags, SteamParamStringArray_t *pUserTags ) = 0;

		STEAM_CALL_RESULT( RemoteStorageDownloadUGCResult_t )
		virtual SteamAPICall_t UGCDownloadToLocation( UGCHandle_t hContent, const char *pchLocation, uint32 unPriority ) = 0;

		// Cloud dynamic state change notification
		virtual int32 GetLocalFileChangeCount() = 0;
		virtual const char *GetLocalFileChange( int iFile, ERemoteStorageLocalFileChange *pEChangeType, ERemoteStorageFilePathType *pEFilePathType ) = 0;

		// Indicate to Steam the beginning / end of a set of local file
		// operations - for example, writing a game save that requires updating two files.
		virtual bool BeginFileWriteBatch() = 0;
		virtual bool EndFileWriteBatch() = 0;
};

#define STEAMREMOTESTORAGE_INTERFACE_VERSION "STEAMREMOTESTORAGE_INTERFACE_VERSION016"

// Global interface accessor
inline ISteamRemoteStorage *SteamRemoteStorage();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamRemoteStorage *, SteamRemoteStorage, STEAMREMOTESTORAGE_INTERFACE_VERSION );

// callbacks
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 




//-----------------------------------------------------------------------------
// Purpose: The result of a call to FileShare()
//-----------------------------------------------------------------------------
struct RemoteStorageFileShareResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 7 };
	EResult m_eResult;			// The result of the operation
	UGCHandle_t m_hFile;		// The handle that can be shared with users and features
	char m_rgchFilename[k_cchFilenameMax]; // The name of the file that was shared
};


// k_iSteamRemoteStorageCallbacks + 8 is deprecated! Do not reuse


//-----------------------------------------------------------------------------
// Purpose: The result of a call to PublishFile()
//-----------------------------------------------------------------------------
struct RemoteStoragePublishFileResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 9 };
	EResult m_eResult;				// The result of the operation.
	PublishedFileId_t m_nPublishedFileId;
	bool m_bUserNeedsToAcceptWorkshopLegalAgreement;
};

// k_iSteamRemoteStorageCallbacks + 10 is deprecated! Do not reuse



//-----------------------------------------------------------------------------
// Purpose: The result of a call to DeletePublishedFile()
//-----------------------------------------------------------------------------
struct RemoteStorageDeletePublishedFileResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 11 };
	EResult m_eResult;				// The result of the operation.
	PublishedFileId_t m_nPublishedFileId;
};


//-----------------------------------------------------------------------------
// Purpose: The result of a call to EnumerateUserPublishedFiles()
//-----------------------------------------------------------------------------
struct RemoteStorageEnumerateUserPublishedFilesResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 12 };
	EResult m_eResult;				// The result of the operation.
	int32 m_nResultsReturned;
	int32 m_nTotalResultCount;
	PublishedFileId_t m_rgPublishedFileId[ k_unEnumeratePublishedFilesMaxResults ];
};


//-----------------------------------------------------------------------------
// Purpose: The result of a call to SubscribePublishedFile()
//-----------------------------------------------------------------------------
struct RemoteStorageSubscribePublishedFileResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 13 };
	EResult m_eResult;				// The result of the operation.
	PublishedFileId_t m_nPublishedFileId;
};


//-----------------------------------------------------------------------------
// Purpose: The result of a call to EnumerateSubscribePublishedFiles()
//-----------------------------------------------------------------------------
struct RemoteStorageEnumerateUserSubscribedFilesResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 14 };
	EResult m_eResult;				// The result of the operation.
	int32 m_nResultsReturned;
	int32 m_nTotalResultCount;
	PublishedFileId_t m_rgPublishedFileId[ k_unEnumeratePublishedFilesMaxResults ];
	uint32 m_rgRTimeSubscribed[ k_unEnumeratePublishedFilesMaxResults ];
};

#if defined(VALVE_CALLBACK_PACK_SMALL)
	VALVE_COMPILE_TIME_ASSERT( sizeof( RemoteStorageEnumerateUserSubscribedFilesResult_t ) == (1 + 1 + 1 + 50 + 100) * 4 );
#elif defined(VALVE_CALLBACK_PACK_LARGE)
	VALVE_COMPILE_TIME_ASSERT( sizeof( RemoteStorageEnumerateUserSubscribedFilesResult_t ) == (1 + 1 + 1 + 50 + 100) * 4 + 4 );
#else
#warning You must first include steam_api_common.h
#endif

//-----------------------------------------------------------------------------
// Purpose: The result of a call to UnsubscribePublishedFile()
//-----------------------------------------------------------------------------
struct RemoteStorageUnsubscribePublishedFileResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 15 };
	EResult m_eResult;				// The result of the operation.
	PublishedFileId_t m_nPublishedFileId;
};


//-----------------------------------------------------------------------------
// Purpose: The result of a call to CommitPublishedFileUpdate()
//-----------------------------------------------------------------------------
struct RemoteStorageUpdatePublishedFileResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 16 };
	EResult m_eResult;				// The result of the operation.
	PublishedFileId_t m_nPublishedFileId;
	bool m_bUserNeedsToAcceptWorkshopLegalAgreement;
};


//-----------------------------------------------------------------------------
// Purpose: The result of a call to UGCDownload()
//-----------------------------------------------------------------------------
struct RemoteStorageDownloadUGCResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 17 };
	EResult m_eResult;				// The result of the operation.
	UGCHandle_t m_hFile;			// The handle to the file that was attempted to be downloaded.
	AppId_t m_nAppID;				// ID of the app that created this file.
	int32 m_nSizeInBytes;			// The size of the file that was downloaded, in bytes.
	char m_pchFileName[k_cchFilenameMax];		// The name of the file that was downloaded. 
	uint64 m_ulSteamIDOwner;		// Steam ID of the user who created this content.
};


//-----------------------------------------------------------------------------
// Purpose: The result of a call to GetPublishedFileDetails()
//-----------------------------------------------------------------------------
struct RemoteStorageGetPublishedFileDetailsResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 18 };
	EResult m_eResult;				// The result of the operation.
	PublishedFileId_t m_nPublishedFileId;
	AppId_t m_nCreatorAppID;		// ID of the app that created this file.
	AppId_t m_nConsumerAppID;		// ID of the app that will consume this file.
	char m_rgchTitle[k_cchPublishedDocumentTitleMax];		// title of document
	char m_rgchDescription[k_cchPublishedDocumentDescriptionMax];	// description of document
	UGCHandle_t m_hFile;			// The handle of the primary file
	UGCHandle_t m_hPreviewFile;		// The handle of the preview file
	uint64 m_ulSteamIDOwner;		// Steam ID of the user who created this content.
	uint32 m_rtimeCreated;			// time when the published file was created
	uint32 m_rtimeUpdated;			// time when the published file was last updated
	ERemoteStoragePublishedFileVisibility m_eVisibility;
	bool m_bBanned;
	char m_rgchTags[k_cchTagListMax];	// comma separated list of all tags associated with this file
	bool m_bTagsTruncated;			// whether the list of tags was too long to be returned in the provided buffer
	char m_pchFileName[k_cchFilenameMax];		// The name of the primary file
	int32 m_nFileSize;				// Size of the primary file
	int32 m_nPreviewFileSize;		// Size of the preview file
	char m_rgchURL[k_cchPublishedFileURLMax];	// URL (for a video or a website)
	EWorkshopFileType m_eFileType;	// Type of the file
	bool m_bAcceptedForUse;			// developer has specifically flagged this item as accepted in the Workshop
};


struct RemoteStorageEnumerateWorkshopFilesResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 19 };
	EResult m_eResult;
	int32 m_nResultsReturned;
	int32 m_nTotalResultCount;
	PublishedFileId_t m_rgPublishedFileId[ k_unEnumeratePublishedFilesMaxResults ];
	float m_rgScore[ k_unEnumeratePublishedFilesMaxResults ];
	AppId_t m_nAppId;
	uint32 m_unStartIndex;
};


//-----------------------------------------------------------------------------
// Purpose: The result of GetPublishedItemVoteDetails
//-----------------------------------------------------------------------------
struct RemoteStorageGetPublishedItemVoteDetailsResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 20 };
	EResult m_eResult;
	PublishedFileId_t m_unPublishedFileId;
	int32 m_nVotesFor;
	int32 m_nVotesAgainst;
	int32 m_nReports;
	float m_fScore;
};


//-----------------------------------------------------------------------------
// Purpose: User subscribed to a file for the app (from within the app or on the web)
//-----------------------------------------------------------------------------
struct RemoteStoragePublishedFileSubscribed_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 21 };
	PublishedFileId_t m_nPublishedFileId;	// The published file id
	AppId_t m_nAppID;						// ID of the app that will consume this file.
};

//-----------------------------------------------------------------------------
// Purpose: User unsubscribed from a file for the app (from within the app or on the web)
//-----------------------------------------------------------------------------
struct RemoteStoragePublishedFileUnsubscribed_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 22 };
	PublishedFileId_t m_nPublishedFileId;	// The published file id
	AppId_t m_nAppID;						// ID of the app that will consume this file.
};


//-----------------------------------------------------------------------------
// Purpose: Published file that a user owns was deleted (from within the app or the web)
//-----------------------------------------------------------------------------
struct RemoteStoragePublishedFileDeleted_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 23 };
	PublishedFileId_t m_nPublishedFileId;	// The published file id
	AppId_t m_nAppID;						// ID of the app that will consume this file.
};


//-----------------------------------------------------------------------------
// Purpose: The result of a call to UpdateUserPublishedItemVote()
//-----------------------------------------------------------------------------
struct RemoteStorageUpdateUserPublishedItemVoteResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 24 };
	EResult m_eResult;				// The result of the operation.
	PublishedFileId_t m_nPublishedFileId;	// The published file id
};


//-----------------------------------------------------------------------------
// Purpose: The result of a call to GetUserPublishedItemVoteDetails()
//-----------------------------------------------------------------------------
struct RemoteStorageUserVoteDetails_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 25 };
	EResult m_eResult;				// The result of the operation.
	PublishedFileId_t m_nPublishedFileId;	// The published file id
	EWorkshopVote m_eVote;			// what the user voted
};

struct RemoteStorageEnumerateUserSharedWorkshopFilesResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 26 };
	EResult m_eResult;				// The result of the operation.
	int32 m_nResultsReturned;
	int32 m_nTotalResultCount;
	PublishedFileId_t m_rgPublishedFileId[ k_unEnumeratePublishedFilesMaxResults ];
};

struct RemoteStorageSetUserPublishedFileActionResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 27 };
	EResult m_eResult;				// The result of the operation.
	PublishedFileId_t m_nPublishedFileId;	// The published file id
	EWorkshopFileAction m_eAction;	// the action that was attempted
};

struct RemoteStorageEnumeratePublishedFilesByUserActionResult_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 28 };
	EResult m_eResult;				// The result of the operation.
	EWorkshopFileAction m_eAction;	// the action that was filtered on
	int32 m_nResultsReturned;
	int32 m_nTotalResultCount;
	PublishedFileId_t m_rgPublishedFileId[ k_unEnumeratePublishedFilesMaxResults ];
	uint32 m_rgRTimeUpdated[ k_unEnumeratePublishedFilesMaxResults ];
};


//-----------------------------------------------------------------------------
// Purpose: Called periodically while a PublishWorkshopFile is in progress
//-----------------------------------------------------------------------------
struct RemoteStoragePublishFileProgress_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 29 };
	double m_dPercentFile;
	bool m_bPreview;
};


//-----------------------------------------------------------------------------
// Purpose: Called when the content for a published file is updated
//-----------------------------------------------------------------------------
struct RemoteStoragePublishedFileUpdated_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 30 };
	PublishedFileId_t m_nPublishedFileId;	// The published file id
	AppId_t m_nAppID;						// ID of the app that will consume this file.
	uint64 m_ulUnused;						// not used anymore
};

//-----------------------------------------------------------------------------
// Purpose: Called when a FileWriteAsync completes
//-----------------------------------------------------------------------------
struct RemoteStorageFileWriteAsyncComplete_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 31 };
	EResult	m_eResult;						// result
};

//-----------------------------------------------------------------------------
// Purpose: Called when a FileReadAsync completes
//-----------------------------------------------------------------------------
struct RemoteStorageFileReadAsyncComplete_t
{
	enum { k_iCallback = k_iSteamRemoteStorageCallbacks + 32 };
	SteamAPICall_t m_hFileReadAsync;		// call handle of the async read which was made
	EResult	m_eResult;						// result
	uint32 m_nOffset;						// offset in the file this read was at
	uint32 m_cubRead;						// amount read - will the <= the amount requested
};

//-----------------------------------------------------------------------------
// Purpose: one or more files for this app have changed locally after syncing
//			to remote session changes
//			Note: only posted if this happens DURING the local app session
//-----------------------------------------------------------------------------
STEAM_CALLBACK_BEGIN( RemoteStorageLocalFileChange_t, k_iSteamRemoteStorageCallbacks + 33 )
STEAM_CALLBACK_END( 0 )

#pragma pack( pop )


#endif // ISTEAMREMOTESTORAGE_H
