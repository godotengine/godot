//====== Copyright © 1996-2014 Valve Corporation, All rights reserved. =======
//
// Purpose: interface to Steam Inventory
//
//=============================================================================

#ifndef ISTEAMINVENTORY_H
#define ISTEAMINVENTORY_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"

// callbacks
#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif


// Every individual instance of an item has a globally-unique ItemInstanceID.
// This ID is unique to the combination of (player, specific item instance)
// and will not be transferred to another player or re-used for another item.
typedef uint64 SteamItemInstanceID_t;

static const SteamItemInstanceID_t k_SteamItemInstanceIDInvalid = (SteamItemInstanceID_t)~0;

// Types of items in your game are identified by a 32-bit "item definition number".
// Valid definition numbers are between 1 and 999999999; numbers less than or equal to
// zero are invalid, and numbers greater than or equal to one billion (1x10^9) are
// reserved for internal Steam use.
typedef int32 SteamItemDef_t;


enum ESteamItemFlags
{
	// Item status flags - these flags are permanently attached to specific item instances
	k_ESteamItemNoTrade = 1 << 0, // This item is account-locked and cannot be traded or given away.

	// Action confirmation flags - these flags are set one time only, as part of a result set
	k_ESteamItemRemoved = 1 << 8,	// The item has been destroyed, traded away, expired, or otherwise invalidated
	k_ESteamItemConsumed = 1 << 9,	// The item quantity has been decreased by 1 via ConsumeItem API.

	// All other flag bits are currently reserved for internal Steam use at this time.
	// Do not assume anything about the state of other flags which are not defined here.
};

struct SteamItemDetails_t
{
	SteamItemInstanceID_t m_itemId;
	SteamItemDef_t m_iDefinition;
	uint16 m_unQuantity;
	uint16 m_unFlags; // see ESteamItemFlags
};

typedef int32 SteamInventoryResult_t;

static const SteamInventoryResult_t k_SteamInventoryResultInvalid = -1;

typedef uint64 SteamInventoryUpdateHandle_t;
const SteamInventoryUpdateHandle_t k_SteamInventoryUpdateHandleInvalid = 0xffffffffffffffffull;

//-----------------------------------------------------------------------------
// Purpose: Steam Inventory query and manipulation API
//-----------------------------------------------------------------------------
class ISteamInventory
{
public:

	// INVENTORY ASYNC RESULT MANAGEMENT
	//
	// Asynchronous inventory queries always output a result handle which can be used with
	// GetResultStatus, GetResultItems, etc. A SteamInventoryResultReady_t callback will
	// be triggered when the asynchronous result becomes ready (or fails).
	//

	// Find out the status of an asynchronous inventory result handle. Possible values:
	//  k_EResultPending - still in progress
	//  k_EResultOK - done, result ready
	//  k_EResultExpired - done, result ready, maybe out of date (see DeserializeResult)
	//  k_EResultInvalidParam - ERROR: invalid API call parameters
	//  k_EResultServiceUnavailable - ERROR: service temporarily down, you may retry later
	//  k_EResultLimitExceeded - ERROR: operation would exceed per-user inventory limits
	//  k_EResultFail - ERROR: unknown / generic error
	virtual EResult GetResultStatus( SteamInventoryResult_t resultHandle ) = 0;

	// Copies the contents of a result set into a flat array. The specific
	// contents of the result set depend on which query which was used.
	virtual bool GetResultItems( SteamInventoryResult_t resultHandle,
								STEAM_OUT_ARRAY_COUNT( punOutItemsArraySize,Output array) SteamItemDetails_t *pOutItemsArray,
								uint32 *punOutItemsArraySize ) = 0;

	// In combination with GetResultItems, you can use GetResultItemProperty to retrieve
	// dynamic string properties for a given item returned in the result set.
	// 
	// Property names are always composed of ASCII letters, numbers, and/or underscores.
	//
	// Pass a NULL pointer for pchPropertyName to get a comma - separated list of available
	// property names.
	//
	// If pchValueBuffer is NULL, *punValueBufferSize will contain the 
	// suggested buffer size. Otherwise it will be the number of bytes actually copied
	// to pchValueBuffer. If the results do not fit in the given buffer, partial 
	// results may be copied.
	virtual bool GetResultItemProperty( SteamInventoryResult_t resultHandle, 
										uint32 unItemIndex, 
										const char *pchPropertyName,
										STEAM_OUT_STRING_COUNT( punValueBufferSizeOut ) char *pchValueBuffer, uint32 *punValueBufferSizeOut ) = 0;

	// Returns the server time at which the result was generated. Compare against
	// the value of IClientUtils::GetServerRealTime() to determine age.
	virtual uint32 GetResultTimestamp( SteamInventoryResult_t resultHandle ) = 0;

	// Returns true if the result belongs to the target steam ID, false if the
	// result does not. This is important when using DeserializeResult, to verify
	// that a remote player is not pretending to have a different user's inventory.
	virtual bool CheckResultSteamID( SteamInventoryResult_t resultHandle, CSteamID steamIDExpected ) = 0;

	// Destroys a result handle and frees all associated memory.
	virtual void DestroyResult( SteamInventoryResult_t resultHandle ) = 0;


	// INVENTORY ASYNC QUERY
	//

	// Captures the entire state of the current user's Steam inventory.
	// You must call DestroyResult on this handle when you are done with it.
	// Returns false and sets *pResultHandle to zero if inventory is unavailable.
	// Note: calls to this function are subject to rate limits and may return
	// cached results if called too frequently. It is suggested that you call
	// this function only when you are about to display the user's full inventory,
	// or if you expect that the inventory may have changed.
	virtual bool GetAllItems( SteamInventoryResult_t *pResultHandle ) = 0;


	// Captures the state of a subset of the current user's Steam inventory,
	// identified by an array of item instance IDs. The results from this call
	// can be serialized and passed to other players to "prove" that the current
	// user owns specific items, without exposing the user's entire inventory.
	// For example, you could call GetItemsByID with the IDs of the user's
	// currently equipped cosmetic items and serialize this to a buffer, and
	// then transmit this buffer to other players upon joining a game.
	virtual bool GetItemsByID( SteamInventoryResult_t *pResultHandle, STEAM_ARRAY_COUNT( unCountInstanceIDs ) const SteamItemInstanceID_t *pInstanceIDs, uint32 unCountInstanceIDs ) = 0;


	// RESULT SERIALIZATION AND AUTHENTICATION
	//
	// Serialized result sets contain a short signature which can't be forged
	// or replayed across different game sessions. A result set can be serialized
	// on the local client, transmitted to other players via your game networking,
	// and deserialized by the remote players. This is a secure way of preventing
	// hackers from lying about posessing rare/high-value items.

	// Serializes a result set with signature bytes to an output buffer. Pass
	// NULL as an output buffer to get the required size via punOutBufferSize.
	// The size of a serialized result depends on the number items which are being
	// serialized. When securely transmitting items to other players, it is
	// recommended to use "GetItemsByID" first to create a minimal result set.
	// Results have a built-in timestamp which will be considered "expired" after
	// an hour has elapsed. See DeserializeResult for expiration handling.
	virtual bool SerializeResult( SteamInventoryResult_t resultHandle, STEAM_OUT_BUFFER_COUNT(punOutBufferSize) void *pOutBuffer, uint32 *punOutBufferSize ) = 0;

	// Deserializes a result set and verifies the signature bytes. Returns false
	// if bRequireFullOnlineVerify is set but Steam is running in Offline mode.
	// Otherwise returns true and then delivers error codes via GetResultStatus.
	//
	// The bRESERVED_MUST_BE_FALSE flag is reserved for future use and should not
	// be set to true by your game at this time.
	//
	// DeserializeResult has a potential soft-failure mode where the handle status
	// is set to k_EResultExpired. GetResultItems() still succeeds in this mode.
	// The "expired" result could indicate that the data may be out of date - not
	// just due to timed expiration (one hour), but also because one of the items
	// in the result set may have been traded or consumed since the result set was
	// generated. You could compare the timestamp from GetResultTimestamp() to
	// ISteamUtils::GetServerRealTime() to determine how old the data is. You could
	// simply ignore the "expired" result code and continue as normal, or you
	// could challenge the player with expired data to send an updated result set.
	virtual bool DeserializeResult( SteamInventoryResult_t *pOutResultHandle, STEAM_BUFFER_COUNT(punOutBufferSize) const void *pBuffer, uint32 unBufferSize, bool bRESERVED_MUST_BE_FALSE = false ) = 0;

	
	// INVENTORY ASYNC MODIFICATION
	//
	
	// GenerateItems() creates one or more items and then generates a SteamInventoryCallback_t
	// notification with a matching nCallbackContext parameter. This API is only intended
	// for prototyping - it is only usable by Steam accounts that belong to the publisher group 
	// for your game.
	// If punArrayQuantity is not NULL, it should be the same length as pArrayItems and should
	// describe the quantity of each item to generate.
	virtual bool GenerateItems( SteamInventoryResult_t *pResultHandle, STEAM_ARRAY_COUNT(unArrayLength) const SteamItemDef_t *pArrayItemDefs, STEAM_ARRAY_COUNT(unArrayLength) const uint32 *punArrayQuantity, uint32 unArrayLength ) = 0;

	// GrantPromoItems() checks the list of promotional items for which the user may be eligible
	// and grants the items (one time only).  On success, the result set will include items which
	// were granted, if any. If no items were granted because the user isn't eligible for any
	// promotions, this is still considered a success.
	virtual bool GrantPromoItems( SteamInventoryResult_t *pResultHandle ) = 0;

	// AddPromoItem() / AddPromoItems() are restricted versions of GrantPromoItems(). Instead of
	// scanning for all eligible promotional items, the check is restricted to a single item
	// definition or set of item definitions. This can be useful if your game has custom UI for
	// showing a specific promo item to the user.
	virtual bool AddPromoItem( SteamInventoryResult_t *pResultHandle, SteamItemDef_t itemDef ) = 0;
	virtual bool AddPromoItems( SteamInventoryResult_t *pResultHandle, STEAM_ARRAY_COUNT(unArrayLength) const SteamItemDef_t *pArrayItemDefs, uint32 unArrayLength ) = 0;

	// ConsumeItem() removes items from the inventory, permanently. They cannot be recovered.
	// Not for the faint of heart - if your game implements item removal at all, a high-friction
	// UI confirmation process is highly recommended.
	virtual bool ConsumeItem( SteamInventoryResult_t *pResultHandle, SteamItemInstanceID_t itemConsume, uint32 unQuantity ) = 0;

	// ExchangeItems() is an atomic combination of item generation and consumption. 
	// It can be used to implement crafting recipes or transmutations, or items which unpack 
	// themselves into other items (e.g., a chest). 
	// Exchange recipes are defined in the ItemDef, and explicitly list the required item 
	// types and resulting generated type. 
	// Exchange recipes are evaluated atomically by the Inventory Service; if the supplied
	// components do not match the recipe, or do not contain sufficient quantity, the 
	// exchange will fail.
	virtual bool ExchangeItems( SteamInventoryResult_t *pResultHandle,
								STEAM_ARRAY_COUNT(unArrayGenerateLength) const SteamItemDef_t *pArrayGenerate, STEAM_ARRAY_COUNT(unArrayGenerateLength) const uint32 *punArrayGenerateQuantity, uint32 unArrayGenerateLength,
								STEAM_ARRAY_COUNT(unArrayDestroyLength) const SteamItemInstanceID_t *pArrayDestroy, STEAM_ARRAY_COUNT(unArrayDestroyLength) const uint32 *punArrayDestroyQuantity, uint32 unArrayDestroyLength ) = 0;
	

	// TransferItemQuantity() is intended for use with items which are "stackable" (can have
	// quantity greater than one). It can be used to split a stack into two, or to transfer
	// quantity from one stack into another stack of identical items. To split one stack into
	// two, pass k_SteamItemInstanceIDInvalid for itemIdDest and a new item will be generated.
	virtual bool TransferItemQuantity( SteamInventoryResult_t *pResultHandle, SteamItemInstanceID_t itemIdSource, uint32 unQuantity, SteamItemInstanceID_t itemIdDest ) = 0;


	// TIMED DROPS AND PLAYTIME CREDIT
	//

	// Deprecated. Calling this method is not required for proper playtime accounting.
	virtual void SendItemDropHeartbeat() = 0;

	// Playtime credit must be consumed and turned into item drops by your game. Only item
	// definitions which are marked as "playtime item generators" can be spawned. The call
	// will return an empty result set if there is not enough playtime credit for a drop.
	// Your game should call TriggerItemDrop at an appropriate time for the user to receive
	// new items, such as between rounds or while the player is dead. Note that players who
	// hack their clients could modify the value of "dropListDefinition", so do not use it
	// to directly control rarity.
	// See your Steamworks configuration to set playtime drop rates for individual itemdefs.
	// The client library will suppress too-frequent calls to this method.
	virtual bool TriggerItemDrop( SteamInventoryResult_t *pResultHandle, SteamItemDef_t dropListDefinition ) = 0;


	// Deprecated. This method is not supported.
	virtual bool TradeItems( SteamInventoryResult_t *pResultHandle, CSteamID steamIDTradePartner,
							 STEAM_ARRAY_COUNT(nArrayGiveLength) const SteamItemInstanceID_t *pArrayGive, STEAM_ARRAY_COUNT(nArrayGiveLength) const uint32 *pArrayGiveQuantity, uint32 nArrayGiveLength,
							 STEAM_ARRAY_COUNT(nArrayGetLength) const SteamItemInstanceID_t *pArrayGet, STEAM_ARRAY_COUNT(nArrayGetLength) const uint32 *pArrayGetQuantity, uint32 nArrayGetLength ) = 0;


	// ITEM DEFINITIONS
	//
	// Item definitions are a mapping of "definition IDs" (integers between 1 and 1000000)
	// to a set of string properties. Some of these properties are required to display items
	// on the Steam community web site. Other properties can be defined by applications.
	// Use of these functions is optional; there is no reason to call LoadItemDefinitions
	// if your game hardcodes the numeric definition IDs (eg, purple face mask = 20, blue
	// weapon mod = 55) and does not allow for adding new item types without a client patch.
	//

	// LoadItemDefinitions triggers the automatic load and refresh of item definitions.
	// Every time new item definitions are available (eg, from the dynamic addition of new
	// item types while players are still in-game), a SteamInventoryDefinitionUpdate_t
	// callback will be fired.
	virtual bool LoadItemDefinitions() = 0;

	// GetItemDefinitionIDs returns the set of all defined item definition IDs (which are
	// defined via Steamworks configuration, and not necessarily contiguous integers).
	// If pItemDefIDs is null, the call will return true and *punItemDefIDsArraySize will
	// contain the total size necessary for a subsequent call. Otherwise, the call will
	// return false if and only if there is not enough space in the output array.
	virtual bool GetItemDefinitionIDs(
				STEAM_OUT_ARRAY_COUNT(punItemDefIDsArraySize,List of item definition IDs) SteamItemDef_t *pItemDefIDs,
				STEAM_DESC(Size of array is passed in and actual size used is returned in this param) uint32 *punItemDefIDsArraySize ) = 0;

	// GetItemDefinitionProperty returns a string property from a given item definition.
	// Note that some properties (for example, "name") may be localized and will depend
	// on the current Steam language settings (see ISteamApps::GetCurrentGameLanguage).
	// Property names are always composed of ASCII letters, numbers, and/or underscores.
	// Pass a NULL pointer for pchPropertyName to get a comma - separated list of available
	// property names. If pchValueBuffer is NULL, *punValueBufferSize will contain the 
	// suggested buffer size. Otherwise it will be the number of bytes actually copied
	// to pchValueBuffer. If the results do not fit in the given buffer, partial 
	// results may be copied.
	virtual bool GetItemDefinitionProperty( SteamItemDef_t iDefinition, const char *pchPropertyName,
		STEAM_OUT_STRING_COUNT(punValueBufferSizeOut) char *pchValueBuffer, uint32 *punValueBufferSizeOut ) = 0;

	// Request the list of "eligible" promo items that can be manually granted to the given
	// user.  These are promo items of type "manual" that won't be granted automatically.
	// An example usage of this is an item that becomes available every week.
	STEAM_CALL_RESULT( SteamInventoryEligiblePromoItemDefIDs_t )
	virtual SteamAPICall_t RequestEligiblePromoItemDefinitionsIDs( CSteamID steamID ) = 0;

	// After handling a SteamInventoryEligiblePromoItemDefIDs_t call result, use this
	// function to pull out the list of item definition ids that the user can be
	// manually granted via the AddPromoItems() call.
	virtual bool GetEligiblePromoItemDefinitionIDs(
		CSteamID steamID,
		STEAM_OUT_ARRAY_COUNT(punItemDefIDsArraySize,List of item definition IDs) SteamItemDef_t *pItemDefIDs,
		STEAM_DESC(Size of array is passed in and actual size used is returned in this param) uint32 *punItemDefIDsArraySize ) = 0;

	// Starts the purchase process for the given item definitions.  The callback SteamInventoryStartPurchaseResult_t
	// will be posted if Steam was able to initialize the transaction.
	// 
	// Once the purchase has been authorized and completed by the user, the callback SteamInventoryResultReady_t 
	// will be posted.
	STEAM_CALL_RESULT( SteamInventoryStartPurchaseResult_t )
	virtual SteamAPICall_t StartPurchase( STEAM_ARRAY_COUNT(unArrayLength) const SteamItemDef_t *pArrayItemDefs, STEAM_ARRAY_COUNT(unArrayLength) const uint32 *punArrayQuantity, uint32 unArrayLength ) = 0;

	// Request current prices for all applicable item definitions
	STEAM_CALL_RESULT( SteamInventoryRequestPricesResult_t )
	virtual SteamAPICall_t RequestPrices() = 0;

	// Returns the number of items with prices.  Need to call RequestPrices() first.
	virtual uint32 GetNumItemsWithPrices() = 0;
	
	// Returns item definition ids and their prices in the user's local currency.
	// Need to call RequestPrices() first.
	virtual bool GetItemsWithPrices( STEAM_ARRAY_COUNT(unArrayLength) STEAM_OUT_ARRAY_COUNT(pArrayItemDefs, Items with prices) SteamItemDef_t *pArrayItemDefs,
									 STEAM_ARRAY_COUNT(unArrayLength) STEAM_OUT_ARRAY_COUNT(pPrices, List of prices for the given item defs) uint64 *pCurrentPrices,
									 STEAM_ARRAY_COUNT(unArrayLength) STEAM_OUT_ARRAY_COUNT(pPrices, List of prices for the given item defs) uint64 *pBasePrices,
									 uint32 unArrayLength ) = 0;

	// Retrieves the price for the item definition id
	// Returns false if there is no price stored for the item definition.
	virtual bool GetItemPrice( SteamItemDef_t iDefinition, uint64 *pCurrentPrice, uint64 *pBasePrice ) = 0;

	// Create a request to update properties on items
	virtual SteamInventoryUpdateHandle_t StartUpdateProperties() = 0;
	// Remove the property on the item
	virtual bool RemoveProperty( SteamInventoryUpdateHandle_t handle, SteamItemInstanceID_t nItemID, const char *pchPropertyName ) = 0;
	// Accessor methods to set properties on items

	STEAM_FLAT_NAME( SetPropertyString )
	virtual bool SetProperty( SteamInventoryUpdateHandle_t handle, SteamItemInstanceID_t nItemID, const char *pchPropertyName, const char *pchPropertyValue ) = 0;

	STEAM_FLAT_NAME( SetPropertyBool )
	virtual bool SetProperty( SteamInventoryUpdateHandle_t handle, SteamItemInstanceID_t nItemID, const char *pchPropertyName, bool bValue ) = 0;

	STEAM_FLAT_NAME( SetPropertyInt64 )
	virtual bool SetProperty( SteamInventoryUpdateHandle_t handle, SteamItemInstanceID_t nItemID, const char *pchPropertyName, int64 nValue ) = 0;

	STEAM_FLAT_NAME( SetPropertyFloat )
	virtual bool SetProperty( SteamInventoryUpdateHandle_t handle, SteamItemInstanceID_t nItemID, const char *pchPropertyName, float flValue ) = 0;

	// Submit the update request by handle
	virtual bool SubmitUpdateProperties( SteamInventoryUpdateHandle_t handle, SteamInventoryResult_t * pResultHandle ) = 0;
	
	virtual bool InspectItem( SteamInventoryResult_t *pResultHandle, const char *pchItemToken ) = 0;
};

#define STEAMINVENTORY_INTERFACE_VERSION "STEAMINVENTORY_INTERFACE_V003"

// Global interface accessor
inline ISteamInventory *SteamInventory();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamInventory *, SteamInventory, STEAMINVENTORY_INTERFACE_VERSION );

// Global accessor for the gameserver client
inline ISteamInventory *SteamGameServerInventory();
STEAM_DEFINE_GAMESERVER_INTERFACE_ACCESSOR( ISteamInventory *, SteamGameServerInventory, STEAMINVENTORY_INTERFACE_VERSION );

// SteamInventoryResultReady_t callbacks are fired whenever asynchronous
// results transition from "Pending" to "OK" or an error state. There will
// always be exactly one callback per handle.
struct SteamInventoryResultReady_t
{
	enum { k_iCallback = k_iSteamInventoryCallbacks + 0 };
	SteamInventoryResult_t m_handle;
	EResult m_result;
};


// SteamInventoryFullUpdate_t callbacks are triggered when GetAllItems
// successfully returns a result which is newer / fresher than the last
// known result. (It will not trigger if the inventory hasn't changed,
// or if results from two overlapping calls are reversed in flight and
// the earlier result is already known to be stale/out-of-date.)
// The normal ResultReady callback will still be triggered immediately
// afterwards; this is an additional notification for your convenience.
struct SteamInventoryFullUpdate_t
{
	enum { k_iCallback = k_iSteamInventoryCallbacks + 1 };
	SteamInventoryResult_t m_handle;
};


// A SteamInventoryDefinitionUpdate_t callback is triggered whenever
// item definitions have been updated, which could be in response to 
// LoadItemDefinitions() or any other async request which required
// a definition update in order to process results from the server.
struct SteamInventoryDefinitionUpdate_t
{
	enum { k_iCallback = k_iSteamInventoryCallbacks + 2 };
};

// Returned 
struct SteamInventoryEligiblePromoItemDefIDs_t
{
	enum { k_iCallback = k_iSteamInventoryCallbacks + 3 };
	EResult m_result;
	CSteamID m_steamID;
	int m_numEligiblePromoItemDefs;
	bool m_bCachedData;	// indicates that the data was retrieved from the cache and not the server
};

// Triggered from StartPurchase call
struct SteamInventoryStartPurchaseResult_t
{
	enum { k_iCallback = k_iSteamInventoryCallbacks + 4 };
	EResult m_result;
	uint64 m_ulOrderID;
	uint64 m_ulTransID;
};


// Triggered from RequestPrices
struct SteamInventoryRequestPricesResult_t
{
	enum { k_iCallback = k_iSteamInventoryCallbacks + 5 };
	EResult m_result;
	char m_rgchCurrency[4];
};

#pragma pack( pop )


#endif // ISTEAMCONTROLLER_H
