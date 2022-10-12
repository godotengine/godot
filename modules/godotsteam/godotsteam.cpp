/////////////////////////////////////////////////
///// SILENCE STEAMWORKS WARNINGS
/////////////////////////////////////////////////
//
// Turn off MSVC-only warning about strcpy
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS 1
#pragma warning(disable:4996)
#pragma warning(disable:4828)
#endif


/////////////////////////////////////////////////
///// HEADER INCLUDES
/////////////////////////////////////////////////
//
// Include GodotSteam header
#include "godotsteam.h"

// Include some Godot headers
#include "core/io/ip_address.h"
#include "core/io/ip.h"

// Include some system headers
#include "fstream"
#include "vector"


/////////////////////////////////////////////////
///// DEFINING CONSTANTS
/////////////////////////////////////////////////
//
// Define Steam API constants
#define INVALID_BREAKPAD_HANDLE 0
#define GAME_EXTRA_INFO_MAX 64
#define AUTH_TICKET_INVALID 0
#define API_CALL_INVALID 0x0
#define APP_ID_INVALID 0x0
#define DEPOT_ID_INVALID 0x0
#define STEAM_ACCOUNT_ID_MASK 0xFFFFFFFF
#define STEAM_ACCOUNT_INSTANCE_MASK 0x000FFFFF
#define STEAM_USER_CONSOLE_INSTANCE 2
#define STEAM_USER_DESKTOP_INSTANCE 1
#define STEAM_USER_WEB_INSTANCE 4
#define QUERY_PORT_ERROR 0xFFFE
#define QUERY_PORT_NOT_INITIALIZED 0xFFFF
#define STEAM_BUFFER_SIZE 255
#define STEAM_LARGE_BUFFER_SIZE 8160

// Define Steam Server API constants
#define FLAG_ACTIVE 0x01
#define FLAG_DEDICATED 0x04
#define FLAG_LINUX 0x08
#define FLAG_NONE 0x00
#define FLAG_PASSWORDED 0x10
#define FLAG_PRIVATE 0x20
#define FLAG_SECURE 0x02

// Define Friends constants
#define CHAT_METADATA_MAX 8192
#define ENUMERATED_FOLLOWERS_MAX 50
#define FRIENDS_GROUP_LIMIT 100
#define INVALID_FRIEND_GROUP_ID -1
#define MAX_FRIENDS_GROUP_NAME 64
#define MAX_RICH_PRESENCE_KEY_LENGTH 64
#define MAX_RICH_PRESENCE_KEYS 20
#define MAX_RICH_PRESENCE_VALUE_LENTH 256
#define PERSONA_NAME_MAX_UTF8 128
#define PERSONA_NAME_MAX_UTF16 32

// Define HTML Surface constants
#define INVALID_HTMLBROWSER 0

// Define HTTP constants
#define HTTPCOOKIE_INVALID_HANDLE 0
#define HTTPREQUEST_INVALID_HANDLE 0

// Define Input constants
#define INPUT_MAX_ANALOG_ACTIONS 16
#define INPUT_MAX_ANALOG_ACTION_DATA 1.0f
#define INPUT_MAX_COUNT 16
#define INPUT_MAX_DIGITAL_ACTIONS 128
#define INPUT_MAX_ORIGINS 8
#define INPUT_MIN_ANALOG_ACTION_DATA -1.0f

// Define Inventory constants
#define INVENTORY_RESULT_INVALID -1
#define ITEM_INSTANCE_ID_INVALID 0

// Define Matchmaking constants
#define SERVER_QUERY_INVALID 0xffffffff
#define MAX_LOBBY_KEY_LENGTH 255
#define FAVORITE_FLAG_FAVORITE 0x01
#define FAVORITE_FLAG_HISTORY 0x02
#define FAVORITE_FLAG_NONE 0x00

// Define Matchmaking Servers constants
#define MAX_GAME_SERVER_GAME_DATA 2048
#define MAX_GAME_SERVER_GAME_DESCRIPTION 64
#define MAX_GAME_SERVER_GAME_DIR 32
#define MAX_GAME_SERVER_MAP_NAME 32
#define MAX_GAME_SERVER_NAME 64
#define MAX_GAME_SERVER_TAGS 128

// Define Music Remote constants
#define MUSIC_NAME_MAX_LENGTH 255
#define MUSIC_PNG_MAX_LENGTH 65535

// Define Remote Play constants
#define DEVICE_FORM_FACTOR_UNKNOWN 0
#define DEVICE_FORM_FACTOR_PHONE 1
#define DEVICE_FORM_FACTOR_TABLET 2
#define DEVICE_FORM_FACTOR_COMPUTER 3
#define DEVICE_FORM_FACTOR_TV 4

// Define Remote Storage constants
#define FILE_NAME_MAX 260
#define PUBLISHED_DOCUMENT_CHANGE_DESCRIPTION_MAX 8000
#define PUBLISHED_DOCUMENT_DESCRIPTION_MAX 8000
#define PUBLISHED_DOCUMENT_TITLE_MAX 128 + 1
#define PUBLISHED_FILE_URL_MAX 256
#define TAG_LIST_MAX 1024 + 1
#define PUBLISHED_FILE_ID_INVALID 0
#define PUBLISHED_FILE_UPDATE_HANDLE_INVALID 0
#define UGC_FILE_STREAM_HANDLE_INVALID 0
#define UGC_HANDLE_INVALID 0
#define ENUMERATE_PUBLISHED_FILES_MAX_RESULTS 50
#define MAX_CLOUD_FILE_CHUNK_SIZE 100 * 1024 * 1024

// Define Screenshot constants
#define SCREENSHOT_INVALID_HANDLE 0
#define UFS_TAG_TYPE_MAX 255
#define UFS_TAG_VALUE_MAX 255
#define MAX_TAGGED_PUBLISHED_FILES 32
#define MAX_TAGGED_USERS 32
#define SCREENSHOT_THUMB_WIDTH 200

// Define UGC constants
#define NUM_UGC_RESULTS_PER_PAGE 50
#define DEVELOPER_METADATA_MAX 5000
#define UGC_QUERY_HANDLE_INVALID 0
#define UGC_UPDATE_HANDLE_INVALID 0

// Define User Stats constants
#define LEADERBOARD_DETAIL_MAX 64
#define LEADERBOARD_NAME_MAX 128
#define STAT_NAME_MAX 128


/////////////////////////////////////////////////
///// DECLARING STRUCTS
/////////////////////////////////////////////////
//
struct SteamNetworkingIdentity networkingIdentity;
struct SteamNetworkingConfigValue_t networkingConfigValue;
struct SteamNetworkingIPAddr networkingIPAddress;
struct SteamNetworkPingLocation_t networkPingLocation;


/////////////////////////////////////////////////
///// STEAM SINGLETON? STEAM SINGLETON
/////////////////////////////////////////////////
//
Steam* Steam::singleton = NULL;


/////////////////////////////////////////////////
///// STEAM OBJECT WITH CALLBACKS
/////////////////////////////////////////////////
//
Steam::Steam():
	// Apps callbacks ////////////////////////////
	callbackDLCInstalled(this, &Steam::dlc_installed),
	callbackFileDetailsResult(this, &Steam::file_details_result),
	callbackNewLaunchURLParameters(this, &Steam::new_launch_url_parameters),
	callbackTimedTrialStatus(this, &Steam::timed_trial_status),

	// Apps List callbacks ///////////////////////
	callbackAppInstalled(this, &Steam::app_installed),
	callbackAppUninstalled(this, &Steam::app_uninstalled),

	// Friends callbacks ////////////////////////
	callbackAvatarLoaded(this, &Steam::avatar_loaded),
	callbackClanActivityDownloaded(this, &Steam::clan_activity_downloaded),
	callbackFriendRichPresenceUpdate(this, &Steam::friend_rich_presence_update),
	callbackConnectedChatJoin(this, &Steam::connected_chat_join),
	callbackConnectedChatLeave(this, &Steam::connected_chat_leave),
	callbackConnectedClanChatMessage(this, &Steam::connected_clan_chat_message),
	callbackConnectedFriendChatMessage(this, &Steam::connected_friend_chat_message),
	callbackJoinRequested(this, &Steam::join_requested),
	callbackOverlayToggled(this, &Steam::overlay_toggled),
	callbackJoinGameRequested(this, &Steam::join_game_requested),
	callbackChangeServerRequested(this, &Steam::change_server_requested),
	callbackJoinClanChatComplete(this, &Steam::join_clan_chat_complete),
	callbackPersonaStateChange(this, &Steam::persona_state_change),
	callbackNameChanged(this, &Steam::name_changed),
	callbackOverlayBrowserProtocol(this, &Steam::overlay_browser_protocol),
	callbackUnreadChatMessagesChanged(this, &Steam::unread_chat_messages_changed),
	callbackEquippedProfileItemsChanged(this, &Steam::equipped_profile_items_changed),
	callbackEquippedProfileItems(this, &Steam::equipped_profile_items),

	// Game Search callbacks ////////////////////
	callbackSearchForGameProgress(this, &Steam::search_for_game_progress),
	callbackSearchForGameResult(this, &Steam::search_for_game_result),
	callbackRequestPlayersForGameProgress(this, &Steam::request_players_for_game_progress),
	callbackRequestPlayersForGameResult(this, &Steam::request_players_for_game_result),
	callbackRequestPlayersForGameFinalResult(this, &Steam::request_players_for_game_final_result),
	callbackSubmitPlayerResult(this, &Steam::submit_player_result),
	callbackEndGameResult(this, &Steam::end_game_result),

	// Game Server callbacks ////////////////////
	callbackServerConnectFailure(this, &Steam::server_connect_failure),
	callbackServerConnected(this, &Steam::server_connected),
	callbackServerDisconnected(this, &Steam::server_disconnected),
	callbackClientApproved(this, &Steam::client_approved),
	callbackClientDenied(this, &Steam::client_denied),
	callbackClientKicked(this, &Steam::client_kick),
	callbackPolicyResponse(this, &Steam::policy_response),
	callbackClientGroupStatus(this, &Steam::client_group_status),
	callbackAssociateClan(this, &Steam::associate_clan),
	callbackPlayerCompat(this, &Steam::player_compat),

	// Game Server Stat callbacks ///////////////
	callbackStatsStored(this, &Steam::stats_stored),
	callbackStatsUnloaded(this, &Steam::stats_unloaded),

	// HTML Surface callbacks ///////////////////
	callbackHTMLBrowserReady(this, &Steam::html_browser_ready),
	callbackHTMLCanGoBackandforward(this, &Steam::html_can_go_backandforward),
	callbackHTMLChangedTitle(this, &Steam::html_changed_title),
	callbackHTMLCloseBrowser(this, &Steam::html_close_browser),
	callbackHTMLFileOpenDialog(this, &Steam::html_file_open_dialog),
	callbackHTMLFinishedRequest(this, &Steam::html_finished_request),
	callbackHTMLHideTooltip(this, &Steam::html_hide_tooltip),
	callbackHTMLHorizontalScroll(this, &Steam::html_horizontal_scroll),
	callbackHTMLJSAlert(this, &Steam::html_js_alert),
	callbackHTMLJSConfirm(this, &Steam::html_js_confirm),
	callbackHTMLLinkAtPosition(this, &Steam::html_link_at_position),
	callbackHTMLNeedsPaint(this, &Steam::html_needs_paint),
	callbackHTMLNewWindow(this, &Steam::html_new_window),
	callbackHTMLOpenLinkInNewTab(this, &Steam::html_open_link_in_new_tab),
	callbackHTMLSearchResults(this, &Steam::html_search_results),
	callbackHTMLSetCursor(this, &Steam::html_set_cursor),
	callbackHTMLShowTooltip(this, &Steam::html_show_tooltip),
	callbackHTMLStartRequest(this, &Steam::html_start_request),
	callbackHTMLStatusText(this, &Steam::html_status_text),
	callbackHTMLUpdateTooltip(this, &Steam::html_update_tooltip),
	callbackHTMLURLChanged(this, &Steam::html_url_changed),
	callbackHTMLVerticalScroll(this, &Steam::html_vertical_scroll),

	// HTTP callbacks ///////////////////////////
	callbackHTTPRequestCompleted(this, &Steam::http_request_completed),
	callbackHTTPRequestDataReceived(this, &Steam::http_request_data_received),
	callbackHTTPRequestHeadersReceived(this, &Steam::http_request_headers_received),

	// Input callbacks //////////////////////////
	callbackInputDeviceConnected(this, &Steam::input_device_connected),
	callbackInputDeviceDisconnected(this, &Steam::input_device_disconnected),
	callbackInputConfigurationLoaded(this, &Steam::input_configuration_loaded),

	// Inventory callbacks //////////////////////
	callbackInventoryDefinitionUpdate(this, &Steam::inventory_definition_update),
	callbackInventoryFullUpdate(this, &Steam::inventory_full_update),
	callbackInventoryResultReady(this, &Steam::inventory_result_ready),

	// Matchmaking callbacks ////////////////////
	callbackFavoritesListAccountsUpdated(this, &Steam::favorites_list_accounts_updated),
	callbackFavoritesListChanged(this, &Steam::favorites_list_changed),
	callbackLobbyMessage(this, &Steam::lobby_message),
	callbackLobbyChatUpdate(this, &Steam::lobby_chat_update),
	callbackLobbyDataUpdate(this, &Steam::lobby_data_update),
	callbackLobbyJoined(this, &Steam::lobby_joined),
	callbackLobbyGameCreated(this, &Steam::lobby_game_created),
	callbackLobbyInvite(this, &Steam::lobby_invite),
	callbackLobbyKicked(this, &Steam::lobby_kicked),

	// Music Remote callbacks ///////////////////
	callbackMusicPlayerRemoteToFront(this, &Steam::music_player_remote_to_front),
	callbackMusicPlayerRemoteWillActivate(this, &Steam::music_player_remote_will_activate),
	callbackMusicPlayerRemoteWillDeactivate(this, &Steam::music_player_remote_will_deactivate),
	callbackMusicPlayerSelectsPlaylistEntry(this, &Steam::music_player_selects_playlist_entry),
	callbackMusicPlayerSelectsQueueEntry(this, &Steam::music_player_selects_queue_entry),
	callbackMusicPlayerWantsLooped(this, &Steam::music_player_wants_looped),
	callbackMusicPlayerWantsPause(this, &Steam::music_player_wants_pause),
	callbackMusicPlayerWantsPlayingRepeatStatus(this, &Steam::music_player_wants_playing_repeat_status),
	callbackMusicPlayerWantsPlayNext(this, &Steam::music_player_wants_play_next),
	callbackMusicPlayerWantsPlayPrevious(this, &Steam::music_player_wants_play_previous),
	callbackMusicPlayerWantsPlay(this, &Steam::music_player_wants_play),
	callbackMusicPlayerWantsShuffled(this, &Steam::music_player_wants_shuffled),
	callbackMusicPlayerWantsVolume(this, &Steam::music_player_wants_volume),
	callbackMusicPlayerWillQuit(this, &Steam::music_player_will_quit),

	// Networking callbacks /////////////////////
	callbackP2PSessionConnectFail(this, &Steam::p2p_session_connect_fail),
	callbackP2PSessionRequest(this, &Steam::p2p_session_request),

	// Networking Messages callbacks ////////////
	callbackNetworkMessagesSessionRequest(this, &Steam::network_messages_session_request),
	callbackNetworkMessagesSessionFailed(this, &Steam::network_messages_session_failed),

	// Networking Sockets callbacks /////////////
	callbackNetworkConnectionStatusChanged(this, &Steam::network_connection_status_changed),
	callbackNetworkAuthenticationStatus(this, &Steam::network_authentication_status),
	callbackNetworkingFakeIPResult(this, &Steam::fake_ip_result),

	// Networking Utils callbacks ///////////////
	callbackRelayNetworkStatus(this, &Steam::relay_network_status),

	// Parental Settings callbacks //////////////
	callbackParentlSettingChanged(this, &Steam::parental_setting_changed),

	// Parties callbacks ////////////////////////
	callbackReserveNotification(this, &Steam::reservation_notification),
	callbackAvailableBeaconLocationsUpdated(this, &Steam::available_beacon_locations_updated),
	callbackActiveBeaconsUpdated(this, &Steam::active_beacons_updated),

	// Remote Play callbacks ////////////////////
	callbackRemotePlaySessionConnected(this, &Steam::remote_play_session_connected),
	callbackRemotePlaySessionDisconnected(this, &Steam::remote_play_session_disconnected),

	// Remote Storage callbacks /////////////////
	callbackLocalFileChanged(this, &Steam::local_file_changed),

	// Screenshot callbacks /////////////////////
	callbackScreenshotReady(this, &Steam::screenshot_ready),
	callbackScreenshotRequested(this, &Steam::screenshot_requested),

	// UGC callbacks ////////////////////////////
	callbackItemDownloaded(this, &Steam::item_downloaded),
	callbackItemInstalled(this, &Steam::item_installed),
	callbackUserSubscribedItemsListChanged(this, &Steam::user_subscribed_items_list_changed),

	// User callbacks ///////////////////////////
	callbackClientGameServerDeny(this, &Steam::client_game_server_deny),
	callbackGameWebCallback(this, &Steam::game_web_callback),
	callbackGetAuthSessionTicketResponse(this, &Steam::get_auth_session_ticket_response),
	callbackIPCFailure(this, &Steam::ipc_failure),
	callbackLicensesUpdated(this, &Steam::licenses_updated),
	callbackMicrotransactionAuthResponse(this, &Steam::microstransaction_auth_response),
	callbackSteamServerConnected(this, &Steam::steam_server_connected),
	callbackSteamServerDisconnected(this, &Steam::steam_server_disconnected),
	callbackValidateAuthTicketResponse(this, &Steam::validate_auth_ticket_response),

	// User stat callbacks //////////////////////
	callbackUserAchievementStored(this, &Steam::user_achievement_stored),
	callbackCurrentStatsReceived(this, &Steam::current_stats_received),
	callbackUserStatsStored(this, &Steam::user_stats_stored),
	callbackUserStatsUnloaded(this, &Steam::user_stats_unloaded),

	// Utility callbacks ////////////////////////
	callbackGamepadTextInputDismissed(this, &Steam::gamepad_text_input_dismissed),
	callbackIPCountry(this, &Steam::ip_country),
	callbackLowPower(this, &Steam::low_power),
	callbackSteamAPICallCompleted(this, &Steam::steam_api_call_completed),
	callbackSteamShutdown(this, &Steam::steam_shutdown),
	callbackAppResumingFromSuspend(this, &Steam::app_resuming_from_suspend),
	callbackFloatingGamepadTextInputDismissed(this, &Steam::floating_gamepad_text_input_dismissed),

	// Video callbacks //////////////////////////
	callbackGetOPFSettingsResult(this, &Steam::get_opf_settings_result),
	callbackGetVideoResult(this, &Steam::get_video_result)
{
	is_init_success = false;
	singleton = this;
}


/////////////////////////////////////////////////
///// INTERNAL FUNCTIONS
/////////////////////////////////////////////////
//
// Get the Steam singleton, obviously
Steam* Steam::get_singleton(){
	return singleton;
}

// Creating a Steam ID for internal use
CSteamID Steam::createSteamID(uint64_t steam_id, int account_type){
	CSteamID converted_steam_id;
	if(account_type < 0 || account_type >= k_EAccountTypeMax){
		account_type = 1;
	}
	converted_steam_id.Set(steam_id, k_EUniversePublic, EAccountType(account_type));
	return converted_steam_id;
}


/////////////////////////////////////////////////
///// MAIN FUNCTIONS
/////////////////////////////////////////////////
//
//! Checks if your executable was launched through Steam and relaunches it through Steam if it wasn't.
bool Steam::restartAppIfNecessary(uint32 app_id){
	return SteamAPI_RestartAppIfNecessary((AppId_t)app_id);
}

//! Initialize Steamworks
Dictionary Steam::steamInit(bool retrieve_stats){
	// Create the response dictionary
	Dictionary initialize;
	// Attempt to initialize Steamworks
	is_init_success = SteamAPI_Init();
	// Set the default status response
	int status = RESULT_FAIL;
	String verbal = "Steamworks failed to initialize.";
	// Steamworks initialized with no problems
	if(is_init_success){
		status = RESULT_OK;
		verbal = "Steamworks active.";
	}
	// The Steam client is not running
	if(!isSteamRunning()){
		status = RESULT_SERVICE_UNAVAILABLE;
		verbal = "Steam not running.";
	}
	else if(SteamUser() == NULL){
		status = RESULT_UNEXPECTED_ERROR;
		verbal = "Invalid app ID or app not installed.";
	}
	// Steam is connected and active, so load the stats and achievements if requested
	if(status == RESULT_OK && SteamUserStats() != NULL && retrieve_stats){
		requestCurrentStats();
	}
	// Get this app ID
	current_app_id = getAppID();
	// Compile the response
	initialize["status"] = status;
	initialize["verbal"] = verbal;
	// Return the Steamworks status
	return initialize;
}

//! Returns true/false if Steam is running.
bool Steam::isSteamRunning(void){
	return SteamAPI_IsSteamRunning();
}

// Initialize the server to Steam.
bool Steam::serverInit(const String& ip, uint16 game_port, uint16 query_port, int server_mode, const String& version_string){
	// Convert the server mode back
	EServerMode mode;
	if(server_mode == 1){
		mode = eServerModeNoAuthentication;
	}
	else if(server_mode == 2){
		mode = eServerModeAuthentication;
	}
	else{
		mode = eServerModeAuthenticationAndSecure;
	}
	// Resolve address and convert it from IP_Address string to uint32_t
	IP_Address address;
	if(ip.is_valid_ip_address()){
		address = ip;
	}
	else{
		address = IP::get_singleton()->resolve_hostname(ip, IP::TYPE_IPV4);
	}
	// Resolution failed
	if(!address.is_valid()){
		return false;
	}
	uint32_t ip4 = *((uint32_t *)address.get_ipv4());
	if(!SteamGameServer_Init(ip4, game_port, query_port, mode, version_string.utf8().get_data())){
		return false;
	}
	return true;
}

// Frees all API-related memory associated with the calling thread. This memory is released automatically by RunCallbacks so single-threaded servers do not need to call this.
void Steam::serverReleaseCurrentThreadMemory(){
	SteamAPI_ReleaseCurrentThreadMemory();
}

// Shut down the server connection to Steam.
void Steam::serverShutdown(){
	SteamGameServer_Shutdown();
}


/////////////////////////////////////////////////
///// APPS
/////////////////////////////////////////////////
//
//! Returns metadata for a DLC by index.
Array Steam::getDLCDataByIndex(){
	if(SteamApps() == NULL){
		return Array();
	}
	int32 count = SteamApps()->GetDLCCount();
	Array dlcData;
	for(int i = 0; i < count; i++){
		AppId_t app_id = 0;
		bool available = false;
		char name[STEAM_BUFFER_SIZE];
		bool success = SteamApps()->BGetDLCDataByIndex(i, &app_id, &available, name, STEAM_BUFFER_SIZE);
		if(success){
			Dictionary dlc;
			dlc["id"] = app_id;
			dlc["available"] = available;
			dlc["name"] = name;
			dlcData.append(dlc);
		}
	}
	return dlcData;
}

//! Check if given application/game is installed, not necessarily owned.
bool Steam::isAppInstalled(uint32_t app_id){
	if(SteamApps() == NULL){
		return false;
	}
	return SteamApps()->BIsAppInstalled((AppId_t)app_id);
}

//! Checks whether the current App ID is for Cyber Cafes.
bool Steam::isCybercafe(){
	if(SteamApps() == NULL){
		return false;
	}
	return SteamApps()->BIsCybercafe();
}

//! Checks if the user owns a specific DLC and if the DLC is installed
bool Steam::isDLCInstalled(uint32_t dlc_id){
	if(SteamApps() == NULL){
		return false;
	}
	return SteamApps()->BIsDlcInstalled((AppId_t)dlc_id);
}

//! Checks if the license owned by the user provides low violence depots.
bool Steam::isLowViolence(){
	if(SteamApps() == NULL){
		return false;
	}
	return SteamApps()->BIsLowViolence();
}

//! Checks if the active user is subscribed to the current App ID.
bool Steam::isSubscribed(){
	if(SteamApps() == NULL){
		return false;
	}
	return SteamApps()->BIsSubscribed();
}

//! Checks if the active user is subscribed to a specified AppId.
bool Steam::isSubscribedApp(uint32_t app_id){
	if(SteamApps() == NULL){
		return false;
	}
	return SteamApps()->BIsSubscribedApp((AppId_t)app_id);
}

//! Checks if the active user is accessing the current app_id via a temporary Family Shared license owned by another user.
//! If you need to determine the steam_id of the permanent owner of the license, use getAppOwner.
bool Steam::isSubscribedFromFamilySharing(){
	if(SteamApps() == NULL){
		return false;
	}
	return SteamApps()->BIsSubscribedFromFamilySharing();
}

//! Checks if the user is subscribed to the current app through a free weekend.
//! This function will return false for users who have a retail or other type of license.
//! Suggested you contact Valve on how to package and secure your free weekend properly.
bool Steam::isSubscribedFromFreeWeekend(){
	if(SteamApps() == NULL){
		return false;
	}
	return SteamApps()->BIsSubscribedFromFreeWeekend();
}

//! Check if game is a timed trial with limited playtime.
Dictionary Steam::isTimedTrial(){
	Dictionary trial;
	if(SteamApps() != NULL){
		uint32 allowed = 0;
		uint32 played = 0;
		if(SteamApps()->BIsTimedTrial(&allowed, &played)){
			trial["seconds_allowed"] = allowed;
			trial["seconds_played"] = played;
		}
	}
	return trial;
}

//! Checks if the user has a VAC ban on their account.
bool Steam::isVACBanned(){
	if(SteamApps() == NULL){
		return false;
	}
	return SteamApps()->BIsVACBanned();
}

//! Return the build ID for this app; will change based on backend updates.
int Steam::getAppBuildId(){
	if(SteamApps() == NULL){
		return 0;
	}
	return SteamApps()->GetAppBuildId();
}

//! Gets the install folder for a specific AppID.
String Steam::getAppInstallDir(uint32_t app_id){
	if(SteamApps() == NULL){
		return "";
	}
	char buffer[STEAM_BUFFER_SIZE];
	SteamApps()->GetAppInstallDir((AppId_t)app_id, buffer, STEAM_BUFFER_SIZE);
	String app_dir = buffer;
	return app_dir;
}

//! Gets the Steam ID of the original owner of the current app. If it's different from the current user then it is borrowed.
uint64_t Steam::getAppOwner(){
	if(SteamApps() == NULL){
		return 0;
	}
	CSteamID converted_steam_id = SteamApps()->GetAppOwner();
	return converted_steam_id.ConvertToUint64();
}

//! Gets a comma separated list of the languages the current app supports.
String Steam::getAvailableGameLanguages(){
	if(SteamApps() == NULL){
		return "None";
	}
	return SteamApps()->GetAvailableGameLanguages();
}

//! Checks if the user is running from a beta branch, and gets the name of the branch if they are.
String Steam::getCurrentBetaName(){
	String beta_name = "";
	if(SteamApps() != NULL){
		char name_string[STEAM_LARGE_BUFFER_SIZE];
		if (SteamApps()->GetCurrentBetaName(name_string, STEAM_LARGE_BUFFER_SIZE)) {
			beta_name = String(name_string);
		}
	}
	return beta_name;
}

//! Gets the current language that the user has set.
String Steam::getCurrentGameLanguage(){
	if(SteamApps() == NULL){
		return "None";
	}
	return SteamApps()->GetCurrentGameLanguage();
}

//! Get the number of DLC the user owns for a parent application/game.
int Steam::getDLCCount(){
	if(SteamApps() == NULL){
		return false;
	}
	return SteamApps()->GetDLCCount();
}

//! Gets the download progress for optional DLC.
Dictionary Steam::getDLCDownloadProgress(uint32_t dlc_id){
	Dictionary progress;
	if(SteamApps() == NULL){
		progress["ret"] = false;
	}
	else{
		uint64 downloaded = 0;
		uint64 total = 0;
		// Get the progress
		progress["ret"] = SteamApps()->GetDlcDownloadProgress((AppId_t)dlc_id, &downloaded, &total);
		if(progress["ret"]){
			progress["downloaded"] = uint64_t(downloaded);
			progress["total"] = uint64_t(total);
		}
	}
	return progress;
}

//! Gets the time of purchase of the specified app in Unix epoch format (time since Jan 1st, 1970).
uint32_t Steam::getEarliestPurchaseUnixTime(uint32_t app_id){
	if(SteamApps() == NULL){
		return 0;
	}
	return SteamApps()->GetEarliestPurchaseUnixTime((AppId_t)app_id);
}

//! Asynchronously retrieves metadata details about a specific file in the depot manifest.
void Steam::getFileDetails(const String& filename){
	if(SteamApps() != NULL){
		SteamApps()->GetFileDetails(filename.utf8().get_data());
	}
}

//! Gets a list of all installed depots for a given App ID.
//! @param app_id App ID to check.
//! @return Array of the installed depots, returned in mount order.
Array Steam::getInstalledDepots(uint32_t app_id){
	if(SteamApps() == NULL){
		return Array();
	}
	Array installed_depots;
	DepotId_t *depots = new DepotId_t[32];
	uint32 installed = SteamApps()->GetInstalledDepots((AppId_t)app_id, depots, 32);
	for(uint32 i = 0; i < installed; i++){
		installed_depots.append(depots[i]);
	}
	delete[] depots;
	return installed_depots;
}

//! Gets the command line if the game was launched via Steam URL, e.g. steam://run/<appid>//<command line>/. This method is preferable to launching with a command line via the operating system, which can be a security risk. In order for rich presence joins to go through this and not be placed on the OS command line, you must enable "Use launch command line" from the Installation > General page on your app.
String Steam::getLaunchCommandLine(){
	if(SteamApps() == NULL){
		return "";
	}
	char commands[STEAM_BUFFER_SIZE];
	SteamApps()->GetLaunchCommandLine(commands, STEAM_BUFFER_SIZE);
	String command_line;
	command_line += commands;
	return command_line;
}

//! Gets the associated launch parameter if the game is run via steam://run/<appid>/?param1=value1;param2=value2;param3=value3 etc.
String Steam::getLaunchQueryParam(const String& key){
	if(SteamApps() == NULL){
		return "";
	}
	return SteamApps()->GetLaunchQueryParam(key.utf8().get_data());
}

//! Allows you to install an optional DLC.
void Steam::installDLC(uint32_t dlc_id){
	if(SteamApps() != NULL){
		SteamApps()->InstallDLC((AppId_t)dlc_id);
	}
}

//! Allows you to force verify game content on next launch.
bool Steam::markContentCorrupt(bool missing_files_only){
	if(SteamApps() == NULL){
		return false;
	}
	return SteamApps()->MarkContentCorrupt(missing_files_only);
}

//! Allows you to uninstall an optional DLC.
void Steam::uninstallDLC(uint32_t dlc_id){
	if(SteamApps() != NULL){
		SteamApps()->UninstallDLC((AppId_t)dlc_id);
	}
}


/////////////////////////////////////////////////
///// APP LISTS
/////////////////////////////////////////////////
//
//! This is a restricted interface that can only be used by previously approved apps, contact your Steam Account Manager if you believe you need access to this API.
//!
//! Get the number of installed apps for this player.
uint32 Steam::getNumInstalledApps(){
	if(SteamAppList() == NULL){
		return 0;
	}
	return SteamAppList()->GetNumInstalledApps();
}

//! Get a list of app IDs for installed apps for this player.
Array Steam::getInstalledApps(uint32 max_app_ids){
	Array installed_apps;
	if(SteamAppList() != NULL){
		uint32 *app_ids = nullptr;
		uint32 these_apps = SteamAppList()->GetInstalledApps(app_ids, max_app_ids);
		// Which is greater?
		if(these_apps < max_app_ids){
			max_app_ids = these_apps;
		}
		// Parse it
		for(uint32 i = 0; i < max_app_ids; i++){
			installed_apps.append(app_ids[i]);
		}
	}
	return installed_apps;
}

//! Get a given app ID's name.
String Steam::getAppName(uint32_t app_id, int name_max){
	String app_name;
	if(SteamAppList() != NULL){
		char* app_buffer = new char[name_max];
		int buffer_size = SteamAppList()->GetAppName((AppId_t)app_id, app_buffer, name_max);
		if(buffer_size != 0){
			app_name += app_buffer;
		}
		delete[] app_buffer;
	}
	return app_name;
}

//! Get a given app ID's install directory.
String Steam::getAppListInstallDir(uint32_t app_id, int name_max){
	String directory_name;
	if(SteamAppList() != NULL){
		char* directory_buffer = new char[name_max];
		int buffer_size = SteamAppList()->GetAppInstallDir((AppId_t)app_id, directory_buffer, name_max);
		if(buffer_size != 0){
			directory_name += directory_buffer;
		}
		delete[] directory_buffer;
	}
	return directory_name;
}

//! Get a given app ID's build.
int Steam::getAppListBuildId(uint32_t app_id){
	if(SteamAppList() == NULL){
		return 0;
	}
	return SteamAppList()->GetAppBuildId((AppId_t)app_id);
}


/////////////////////////////////////////////////
///// FRIENDS
/////////////////////////////////////////////////
//
//! Activates the overlay with optional dialog to open the following: "Friends", "Community", "Players", "Settings", "OfficialGameGroup", "Stats", "Achievements", "LobbyInvite".
void Steam::activateGameOverlay(const String& url){
	if(SteamFriends() != NULL){
		SteamFriends()->ActivateGameOverlay(url.utf8().get_data());
	}
}

//! Activates game overlay to open the invite dialog. Invitations will be sent for the provided lobby.
void Steam::activateGameOverlayInviteDialog(uint64_t steam_id){
	if(SteamFriends() != NULL){
		CSteamID user_id = (uint64)steam_id;
		SteamFriends()->ActivateGameOverlayInviteDialog(user_id);
	}
}

//! Activates the game overlay to open an invite dialog that will send the provided Rich Presence connect string to selected friends.
void Steam::activateGameOverlayInviteDialogConnectString(const String& connect_string){
	if(SteamFriends() != NULL){
		SteamFriends()->ActivateGameOverlayInviteDialogConnectString(connect_string.utf8().get_data());
	}
}

//! Activates the overlay with the application/game Steam store page.
void Steam::activateGameOverlayToStore(uint32_t app_id){
	if(SteamFriends() != NULL){
		SteamFriends()->ActivateGameOverlayToStore(AppId_t(app_id), EOverlayToStoreFlag(0));
	}
}

//! Activates the overlay to the following: "steamid", "chat", "jointrade", "stats", "achievements", "friendadd", "friendremove", "friendrequestaccept", "friendrequestignore".
void Steam::activateGameOverlayToUser(const String& url, uint64_t steam_id){
	if(SteamFriends() != NULL){
		CSteamID user_id = (uint64)steam_id;
		SteamFriends()->ActivateGameOverlayToUser(url.utf8().get_data(), user_id);
	}
}

//! Activates the overlay with specified web address.
void Steam::activateGameOverlayToWebPage(const String& url){
	if(SteamFriends() != NULL){
		SteamFriends()->ActivateGameOverlayToWebPage(url.utf8().get_data());
	}
}

//! Clear the game information in Steam; used in 'View Game Info'.
void Steam::clearRichPresence(){
	if(SteamFriends() != NULL){
		SteamFriends()->ClearRichPresence();
	}
}

//! Closes the specified Steam group chat room in the Steam UI.
bool Steam::closeClanChatWindowInSteam(uint64_t chat_id){
	if(SteamFriends() == NULL){
		return false;
	}
	CSteamID chat = (uint64)chat_id;
	return SteamFriends()->CloseClanChatWindowInSteam(chat);
}

//! For clans a user is a member of, they will have reasonably up-to-date information, but for others you'll have to download the info to have the latest.
void Steam::downloadClanActivityCounts(uint64_t clan_id, int clans_to_request){
	if(SteamFriends() != NULL){
		clan_activity = (uint64)clan_id;
		SteamFriends()->DownloadClanActivityCounts(&clan_activity, clans_to_request);
	}
}

//! Gets the list of users that the current user is following.
void Steam::enumerateFollowingList(uint32 start_index){
	if(SteamFriends() != NULL){
		SteamAPICall_t api_call = SteamFriends()->EnumerateFollowingList(start_index);
		callResultEnumerateFollowingList.Set(api_call, this, &Steam::enumerate_following_list);
	}
}

//! Gets the Steam ID at the given index in a Steam group chat.
uint64_t Steam::getChatMemberByIndex(uint64_t clan_id, int user){
	if(SteamFriends() == NULL){
		return 0;
	}
	clan_activity = (uint64)clan_id;
	CSteamID chat_id = SteamFriends()->GetChatMemberByIndex(clan_activity, user);
	return chat_id.ConvertToUint64();
}

//! Gets the most recent information we have about what the users in a Steam Group are doing.
Dictionary Steam::getClanActivityCounts(uint64_t clan_id){
	Dictionary activity;
	if(SteamFriends() == NULL){
		return activity;
	}
	clan_activity = (uint64)clan_id;
	int online = 0;
	int ingame = 0;
	int chatting = 0;
	bool success = SteamFriends()->GetClanActivityCounts(clan_activity, &online, &ingame, &chatting);
	// Add these to the dictionary if successful
	if(success){
		activity["clan"] = clan_id;
		activity["online"] = online;
		activity["ingame"] = ingame;
		activity["chatting"] = chatting;
	}
	return activity;
}

//! Gets the Steam group's Steam ID at the given index.
uint64_t Steam::getClanByIndex(int clan){
	if(SteamFriends() == NULL){
		return 0;
	}
	return SteamFriends()->GetClanByIndex(clan).ConvertToUint64();
}

//! Get the number of users in a Steam group chat.
int Steam::getClanChatMemberCount(uint64_t clan_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	clan_activity = (uint64)clan_id;
	return SteamFriends()->GetClanChatMemberCount(clan_activity);
}

//!  Gets the data from a Steam group chat room message.  This should only ever be called in response to a GameConnectedClanChatMsg_t callback.
Dictionary Steam::getClanChatMessage(uint64_t chat_id, int message){
	Dictionary chat_message;
	if(SteamFriends() == NULL){
		return chat_message;
	}
	CSteamID chat = (uint64)chat_id;
	char text[STEAM_LARGE_BUFFER_SIZE];
	EChatEntryType type = k_EChatEntryTypeInvalid;
	CSteamID user_id;
	chat_message["ret"] = SteamFriends()->GetClanChatMessage(chat, message, text, STEAM_LARGE_BUFFER_SIZE, &type, &user_id);
	chat_message["text"] = String(text);
	chat_message["type"] = type;
	uint64_t user = user_id.ConvertToUint64();
	chat_message["chatter"] = user;
	return chat_message;
}

//! Gets the number of Steam groups that the current user is a member of.  This is used for iteration, after calling this then GetClanByIndex can be used to get the Steam ID of each Steam group.
int Steam::getClanCount(){
	if(SteamFriends() == NULL){
		return 0;
	}
	return SteamFriends()->GetClanCount();
}

//! Gets the display name for the specified Steam group; if the local client knows about it.
String Steam::getClanName(uint64_t clan_id){
	if(SteamFriends() == NULL){
		return "";
	}
	clan_activity = (uint64)clan_id;
	return String::utf8(SteamFriends()->GetClanName(clan_activity));
}

//! Returns the steam_id of a clan officer, by index, of range [0,GetClanOfficerCount).
uint64_t Steam::getClanOfficerByIndex(uint64_t clan_id, int officer){
	if(SteamFriends() == NULL){
		return 0;
	}
	clan_activity = (uint64)clan_id;
	CSteamID officer_id = SteamFriends()->GetClanOfficerByIndex(clan_activity, officer);
	return officer_id.ConvertToUint64();
}

//! Returns the number of officers in a clan (including the owner).
int Steam::getClanOfficerCount(uint64_t clan_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	clan_activity = (uint64)clan_id;
	return SteamFriends()->GetClanOfficerCount(clan_activity);
}

//! Returns the steam_id of the clan owner.
uint64_t Steam::getClanOwner(uint64_t clan_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	clan_activity = (uint64)clan_id;
	CSteamID owner_id = SteamFriends()->GetClanOwner(clan_activity);
	return owner_id.ConvertToUint64();
}

//! Gets the unique tag (abbreviation) for the specified Steam group; If the local client knows about it.  The Steam group abbreviation is a unique way for people to identify the group and is limited to 12 characters. In some games this will appear next to the name of group members.
String Steam::getClanTag(uint64_t clan_id){
	if(SteamFriends() == NULL){
		return "";
	}
	clan_activity = (uint64)clan_id;
	return String::utf8(SteamFriends()->GetClanTag(clan_activity));
}

//! Gets the Steam ID of the recently played with user at the given index.
uint64_t Steam::getCoplayFriend(int friend_number){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID friend_id = SteamFriends()->GetCoplayFriend(friend_number);
	return friend_id.ConvertToUint64();
}

//! Gets the number of players that the current users has recently played with, across all games.  This is used for iteration, after calling this then GetCoplayFriend can be used to get the Steam ID of each player.  These players are have been set with previous calls to SetPlayedWith.
int Steam::getCoplayFriendCount(){
	if(SteamFriends() == NULL){
		return 0;
	}
	return SteamFriends()->GetCoplayFriendCount();
}

//! Gets the number of users following the specified user.
void Steam::getFollowerCount(uint64_t steam_id){
	if(SteamFriends() != NULL){
		CSteamID user_id = (uint64)steam_id;
		SteamAPICall_t api_call = SteamFriends()->GetFollowerCount(user_id);
		callResultFollowerCount.Set(api_call, this, &Steam::get_follower_count);
	}
}

//! Returns the Steam ID of a user.
uint64_t Steam::getFriendByIndex(int friend_number, int friend_flags){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID friend_id = SteamFriends()->GetFriendByIndex(friend_number, friend_flags);
	return friend_id.ConvertToUint64();
}

//! Gets the app ID of the game that user played with someone on their recently-played-with list.
uint32 Steam::getFriendCoplayGame(uint64_t friend_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID steam_id = (uint64)friend_id;
	return SteamFriends()->GetFriendCoplayGame(steam_id);
}

//! Gets the timestamp of when the user played with someone on their recently-played-with list.  The time is provided in Unix epoch format (seconds since Jan 1st 1970).
int Steam::getFriendCoplayTime(uint64_t friend_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID steam_id = (uint64)friend_id;
	return SteamFriends()->GetFriendCoplayTime(steam_id);
}

//! Get number of friends user has.
int Steam::getFriendCount(){
	if(SteamFriends() == NULL){
		return 0;
	}
	return SteamFriends()->GetFriendCount(0x04);
}

//! Iterators for getting users in a chat room, lobby, game server or clan.
int Steam::getFriendCountFromSource(uint64_t source_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID source = (uint64)source_id;
	return SteamFriends()->GetFriendCountFromSource(source);
}

//! Gets the Steam ID at the given index from a source (Steam group, chat room, lobby, or game server).
uint64_t Steam::getFriendFromSourceByIndex(uint64_t source_id, int friend_number){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID source = (uint64)source_id;
	CSteamID friend_id = SteamFriends()->GetFriendFromSourceByIndex(source, friend_number);
	return friend_id.ConvertToUint64();
}

//! Returns dictionary of friend game played if valid
Dictionary Steam::getFriendGamePlayed(uint64_t steam_id){
	Dictionary friend_game;
	if(SteamFriends() == NULL){
		return friend_game;
	}
	FriendGameInfo_t game_info;
	CSteamID user_id = (uint64)steam_id;
	bool success = SteamFriends()->GetFriendGamePlayed(user_id, &game_info);
	// If successful
	if(success){
		// Is there a valid lobby?
		if(game_info.m_steamIDLobby.IsValid()){
			friend_game["id"] = game_info.m_gameID.AppID();
			// Convert the IP address back to a string
			const int NBYTES = 4;
			uint8 octet[NBYTES];
			char gameIP[16];
			for(int j = 0; j < NBYTES; j++){
				octet[j] = game_info.m_unGameIP >> (j * 8);
			}
			sprintf(gameIP, "%d.%d.%d.%d", octet[0], octet[1], octet[2], octet[3]);
			friend_game["ip"] = gameIP;
			friend_game["game_port"] = game_info.m_usGamePort;
			friend_game["query_port"] = (char)game_info.m_usQueryPort;
			friend_game["lobby"] = uint64_t(game_info.m_steamIDLobby.ConvertToUint64());
		}
		else{
			friend_game["id"] = game_info.m_gameID.AppID();
			friend_game["ip"] = "0.0.0.0";
			friend_game["game_port"] = "0";
			friend_game["query_port"] = "0";
			friend_game["lobby"] = "No valid lobby";
		}
	}
	return friend_game;
}

//! Gets the data from a Steam friends message. This should only ever be called in response to a GameConnectedFriendChatMsg_t callback.
Dictionary Steam::getFriendMessage(uint64_t friend_id, int message){
	Dictionary chat;
	if(SteamFriends() == NULL){
		return chat;
	}
	char text[STEAM_LARGE_BUFFER_SIZE];
	EChatEntryType type = k_EChatEntryTypeInvalid;
	chat["ret"] = SteamFriends()->GetFriendMessage(createSteamID(friend_id), message, text, STEAM_LARGE_BUFFER_SIZE, &type);
	chat["text"] = String(text);
	return chat;
}

//! Get given friend's Steam username.
String Steam::getFriendPersonaName(uint64_t steam_id){
	if(SteamFriends() != NULL && steam_id > 0){
		CSteamID user_id = (uint64)steam_id;
		bool is_data_loading = SteamFriends()->RequestUserInformation(user_id, true);
		if(!is_data_loading){
			return String::utf8(SteamFriends()->GetFriendPersonaName(user_id));
		}
	}
	return "";
}

//! Accesses old friends names; returns an empty string when there are no more items in the history.
String Steam::getFriendPersonaNameHistory(uint64_t steam_id, int name_history){
	if(SteamFriends() == NULL){
		return "";
	}
	CSteamID user_id = (uint64)steam_id;
	return String::utf8(SteamFriends()->GetFriendPersonaNameHistory(user_id, name_history));
}

//! Returns the current status of the specified user.
int Steam::getFriendPersonaState(uint64_t steam_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->GetFriendPersonaState(user_id);
}

//! Returns a relationship to a user.
int Steam::getFriendRelationship(uint64_t steam_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->GetFriendRelationship(user_id);
}

//! Get a Rich Presence value from a specified friend (typically only used for debugging).
String Steam::getFriendRichPresence(uint64_t friend_id, const String& key){
	if(SteamFriends() == NULL){
		return "";
	}
	CSteamID user = (uint64)friend_id;
	return SteamFriends()->GetFriendRichPresence(user, key.utf8().get_data());
}

//! Gets the number of Rich Presence keys that are set on the specified user.
int Steam::getFriendRichPresenceKeyCount(uint64_t friend_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID user = (uint64)friend_id;
	return SteamFriends()->GetFriendRichPresenceKeyCount(user);
}

//! Returns an empty string ("") if the index is invalid or the specified user has no Rich Presence data available.
String Steam::getFriendRichPresenceKeyByIndex(uint64_t friend_id, int key){
	if(SteamFriends() == NULL){
		return "";
	}
	CSteamID user = (uint64)friend_id;
	return SteamFriends()->GetFriendRichPresenceKeyByIndex(user, key);
}

//! Gets the number of friends groups (tags) the user has created.  This is used for iteration, after calling this then GetFriendsGroupIDByIndex can be used to get the ID of each friend group.  This is not to be confused with Steam groups. Those can be obtained with GetClanCount.
int Steam::getFriendsGroupCount(){
	if(SteamFriends() == NULL){
		return 0;
	}
	return SteamFriends()->GetFriendsGroupCount();
}

//! Gets the friends group ID for the given index.
int16 Steam::getFriendsGroupIDByIndex(int16 friend_group){
	if(SteamFriends() == NULL){
		return 0;
	}
	return SteamFriends()->GetFriendsGroupIDByIndex(friend_group);
}

//! Gets the number of friends in a given friends group.  This should be called before getting the list of friends with GetFriendsGroupMembersList.
int Steam::getFriendsGroupMembersCount(int16 friend_group){
	if(SteamFriends() == NULL){
		return 0;
	}
	return SteamFriends()->GetFriendsGroupMembersCount(friend_group);
}

//! Gets the number of friends in the given friends group.  If fewer friends exist than requested those positions' Steam IDs will be invalid.  You must call GetFriendsGroupMembersCount before calling this to set up the pOutSteamIDMembers array with an appropriate size!
Array Steam::getFriendsGroupMembersList(int16 friend_group, int member_count){
	Array member_list;
	if(SteamFriends() == NULL){
		return member_list;
	}
	CSteamID friend_ids;
	SteamFriends()->GetFriendsGroupMembersList((FriendsGroupID_t)friend_group, &friend_ids, member_count);
	uint64_t friends = friend_ids.ConvertToUint64();
	member_list.append(friends);
	return member_list;
}

//! Gets the name for the given friends group.
String Steam::getFriendsGroupName(int16 friend_group){
	if(SteamFriends() == NULL){
		return "";
	}
	return String::utf8(SteamFriends()->GetFriendsGroupName(friend_group));
}

//! Get friend's steam level, obviously.
int Steam::getFriendSteamLevel(uint64_t steam_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->GetFriendSteamLevel(user_id);
}

//! Gets the large (184x184) avatar of the current user, which is a handle to be used in GetImageRGBA(), or 0 if none set.
int Steam::getLargeFriendAvatar(uint64_t steam_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->GetLargeFriendAvatar(user_id);
}

//! Gets the medium (64x64) avatar of the current user, which is a handle to be used in GetImageRGBA(), or 0 if none set.
int Steam::getMediumFriendAvatar(uint64_t steam_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->GetMediumFriendAvatar(user_id);
}

//! Get the user's Steam username.
String Steam::getPersonaName(){
	if(SteamFriends() == NULL){
		return "";
	}
	return String::utf8(SteamFriends()->GetPersonaName());
}

//! Gets the status of the current user.
int Steam::getPersonaState(){
	if(SteamFriends() == NULL){
		return 0;
	}
	return SteamFriends()->GetPersonaState();
}

//! Get player's avatar.
void Steam::getPlayerAvatar(int size, uint64_t steam_id){
	// If no Steam ID is given, use the current user's
	if(steam_id == 0){
		steam_id = getSteamID();
	}
	if(SteamFriends() != NULL && size > 0 && size < 4){
		int handle = -2;
		switch(size){
			case 1:{
				handle = getSmallFriendAvatar(steam_id);
				size = 32; break;
			}
			case 2:{
				handle = getMediumFriendAvatar(steam_id);
				size = 64; break;
			}
			case 3:{
				handle = getLargeFriendAvatar(steam_id);
				size = 184; break;
			}
			default:
				return;
		}
		if(handle <= 0){
			printf("[Steam] Error retrieving avatar handle.");
		}
		AvatarImageLoaded_t* avatar_data = new AvatarImageLoaded_t;
		CSteamID avatar_id = (uint64)steam_id;
		avatar_data->m_steamID = avatar_id;
		avatar_data->m_iImage = handle;
		avatar_data->m_iWide = size;
		avatar_data->m_iTall = size;
		avatar_loaded(avatar_data);
		delete avatar_data;
	}
}

//! Returns nickname the current user has set for the specified player. Returns NULL if the no nickname has been set for that player.
String Steam::getPlayerNickname(uint64_t steam_id){
	if(SteamFriends() == NULL){
		return "";
	}
	CSteamID user_id = (uint64)steam_id;
	return String::utf8(SteamFriends()->GetPlayerNickname(user_id));
}

// Returns a string property for a user's equipped profile item.
String Steam::getProfileItemPropertyString(uint64_t steam_id, int item_type, int item_property){
	if(SteamFriends() == NULL){
		return "";
	}
	CSteamID user_id = (uint64)steam_id;
	return String::utf8(SteamFriends()->GetProfileItemPropertyString(user_id, (ECommunityProfileItemType)item_type, (ECommunityProfileItemProperty)item_property));
}

// Returns an unsigned integer property for a user's equipped profile item.
uint32 Steam::getProfileItemPropertyInt(uint64_t steam_id, int item_type, int item_property){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->GetProfileItemPropertyUint(user_id, (ECommunityProfileItemType)item_type, (ECommunityProfileItemProperty)item_property);
}

//! Get list of players user has recently played game with.
Array Steam::getRecentPlayers(){
	if(SteamFriends() == NULL){
		return Array();
	}
	int count = SteamFriends()->GetCoplayFriendCount();
	Array recents;
	for(int i = 0; i < count; i++){
		CSteamID player_id = SteamFriends()->GetCoplayFriend(i);
		if(SteamFriends()->GetFriendCoplayGame(player_id) == SteamUtils()->GetAppID()){
			Dictionary player;
			int time = SteamFriends()->GetFriendCoplayTime(player_id);
			int status = SteamFriends()->GetFriendPersonaState(player_id);
			player["id"] = (uint64_t)player_id.ConvertToUint64();
			player["name"] = String::utf8(SteamFriends()->GetFriendPersonaName(player_id));
			player["time"] = time;
			player["status"] = status;
			recents.append(player);
		}
	}
	return recents;
}

//! Gets the small (32x32) avatar of the current user, which is a handle to be used in GetImageRGBA(), or 0 if none set.
int Steam::getSmallFriendAvatar(uint64_t steam_id){
	if(SteamFriends() == NULL){
		return 0;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->GetSmallFriendAvatar(user_id);
}

//! Get list of friends groups (tags) the user has created. This is not to be confused with Steam groups.
Array Steam::getUserFriendsGroups(){
	if(SteamFriends() == NULL){
		return Array();
	}
	int tag_count = SteamFriends()->GetFriendsGroupCount();
	Array friends_groups;
	for(int i = 0; i < tag_count; i++){
		Dictionary tags;
		int16 friends_group_id = SteamFriends()->GetFriendsGroupIDByIndex(i);
		String group_name = SteamFriends()->GetFriendsGroupName(friends_group_id);
		int group_members = SteamFriends()->GetFriendsGroupMembersCount(friends_group_id);
		tags["id"] = friends_group_id;
		tags["name"] = group_name;
		tags["members"] = group_members;
		friends_groups.append(tags);
	}
	return friends_groups;
}

//! If current user is chat restricted, he can't send or receive any text/voice chat messages. The user can't see custom avatars. But the user can be online and send/recv game invites.
uint32 Steam::getUserRestrictions(){
	if(SteamFriends() == NULL){
		return 0;
	}
	return SteamFriends()->GetUserRestrictions();
}

//! Get a list of user's Steam friends; a mix of different Steamworks API friend functions.
Array Steam::getUserSteamFriends(){
	if(SteamFriends() == NULL){
		return Array();
	}
	int count = SteamFriends()->GetFriendCount(0x04);
	Array steam_friends;
	for(int i = 0; i < count; i++){
		Dictionary friends;
		CSteamID friend_id = SteamFriends()->GetFriendByIndex(i, 0x04);
		int status = SteamFriends()->GetFriendPersonaState(friend_id);
		friends["id"] = (uint64_t)friend_id.ConvertToUint64();
		friends["name"] = String::utf8(SteamFriends()->GetFriendPersonaName(friend_id));
		friends["status"] = status;
		steam_friends.append(friends);
	}
	return steam_friends;
}

//! Get list of user's Steam groups; a mix of different Steamworks API group functions.
Array Steam::getUserSteamGroups(){
	if(SteamFriends() == NULL){
		return Array();
	}
	int group_count = SteamFriends()->GetClanCount();
	Array steam_groups;
	for(int i = 0; i < group_count; i++){
		Dictionary groups;
		CSteamID group_id = SteamFriends()->GetClanByIndex(i);
		String name = SteamFriends()->GetClanName(group_id);
		String tag = SteamFriends()->GetClanTag(group_id);
		groups["id"] = (uint64_t)group_id.ConvertToUint64();
		groups["name"] = name;
		groups["tag"] = tag;
		steam_groups.append(groups);
	}
	return steam_groups;
}

// After calling RequestEquippedProfileItems, you can use this function to check if the user has a type of profile item equipped or not.
bool Steam::hasEquippedProfileItem(uint64_t steam_id, int item_type){
	if(SteamFriends() == NULL){
		return false;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->BHasEquippedProfileItem(user_id, (ECommunityProfileItemType)item_type);
}

//! Returns true if the specified user meets any of the criteria specified in iFriendFlags.
bool Steam::hasFriend(uint64_t steam_id, int friend_flags){
	if(SteamFriends() == NULL){
		return false;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->HasFriend(user_id, friend_flags);
}

//! Invite friend to current game/lobby.
bool Steam::inviteUserToGame(uint64_t steam_id, const String& connect_string){
	if(SteamFriends() == NULL){
		return false;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->InviteUserToGame(user_id, connect_string.utf8().get_data());
}

//! Checks if a user in the Steam group chat room is an admin.
bool Steam::isClanChatAdmin(uint64_t chat_id, uint64_t steam_id){
	if(SteamFriends() == NULL){
		return false;
	}
	CSteamID chat = (uint64)chat_id;
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->IsClanChatAdmin(chat, user_id);
}

//! Checks if the Steam group is public.
bool Steam::isClanPublic(uint64_t clan_id){
	if(SteamFriends() == NULL){
		return false;
	}
	clan_activity = (uint64)clan_id;
	return SteamFriends()->IsClanPublic(clan_activity);
}

//! Checks if the Steam group is an official game group/community hub.
bool Steam::isClanOfficialGameGroup(uint64_t clan_id){
	if(SteamFriends() == NULL){
		return false;
	}
	clan_activity = (uint64)clan_id;
	return SteamFriends()->IsClanOfficialGameGroup(clan_activity);
}

//! Checks if the Steam Group chat room is open in the Steam UI.
bool Steam::isClanChatWindowOpenInSteam(uint64_t chat_id){
	if(SteamFriends() == NULL){
		return false;
	}
	CSteamID chat = (uint64)chat_id;
	return SteamFriends()->IsClanChatWindowOpenInSteam(chat);
}

//! Checks if the current user is following the specified user.
void Steam::isFollowing(uint64_t steam_id){
	if(SteamFriends() != NULL){
		CSteamID user_id = (uint64)steam_id;
		SteamAPICall_t api_call = SteamFriends()->IsFollowing(user_id);
		callResultIsFollowing.Set(api_call, this, &Steam::is_following);
	}
}

//! Returns true if the local user can see that steam_id_user is a member or in steamIDSource.
bool Steam::isUserInSource(uint64_t steam_id, uint64_t source_id){
	if(SteamFriends() == NULL){
		return false;
	}
	CSteamID user_id = (uint64)steam_id;
	CSteamID source = (uint64)source_id;
	return SteamFriends()->IsUserInSource(user_id, source);
}

//! Allows the user to join Steam group (clan) chats right within the game.
void Steam::joinClanChatRoom(uint64_t clan_id){
	if(SteamFriends() != NULL){
		clan_activity = (uint64)clan_id;
		SteamFriends()->JoinClanChatRoom(clan_activity);
	}
}

//! Leaves a Steam group chat that the user has previously entered with JoinClanChatRoom.
bool Steam::leaveClanChatRoom(uint64_t clan_id){
	if(SteamFriends() == NULL){
		return false;
	}
	clan_activity = (uint64)clan_id;
	return SteamFriends()->LeaveClanChatRoom(clan_activity);
}

//! Opens the specified Steam group chat room in the Steam UI.
bool Steam::openClanChatWindowInSteam(uint64_t chat_id){
	if(SteamFriends() == NULL){
		return false;
	}
	CSteamID chat = (uint64)chat_id;
	return SteamFriends()->OpenClanChatWindowInSteam(chat);
}

//! Call this before calling ActivateGameOverlayToWebPage() to have the Steam Overlay Browser block navigations to your specified protocol (scheme) uris and instead dispatch a OverlayBrowserProtocolNavigation_t callback to your game.
bool Steam::registerProtocolInOverlayBrowser(const String& protocol){
	if(SteamFriends() == NULL){
		return false;
	}
	return SteamFriends()->RegisterProtocolInOverlayBrowser(protocol.utf8().get_data());
}

//! Sends a message to a Steam friend.
bool Steam::replyToFriendMessage(uint64_t steam_id, const String& message){
	if(SteamFriends() == NULL){
		return false;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->ReplyToFriendMessage(user_id, message.utf8().get_data());
}

//! Requests information about a clan officer list; when complete, data is returned in ClanOfficerListResponse_t call result.
void Steam::requestClanOfficerList(uint64_t clan_id){
	if(SteamFriends() != NULL){
		clan_activity = (uint64)clan_id;
		SteamAPICall_t api_call = SteamFriends()->GetFollowerCount(clan_activity);
		callResultClanOfficerList.Set(api_call, this, &Steam::request_clan_officer_list);
	}
}

//! Requests rich presence for a specific user.
void Steam::requestFriendRichPresence(uint64_t friend_id){
	if(SteamFriends() != NULL){
		CSteamID user = (uint64)friend_id;
		return SteamFriends()->RequestFriendRichPresence(user);
	}
}

//! Requests information about a user - persona name & avatar; if bRequireNameOnly is set, then the avatar of a user isn't downloaded.
bool Steam::requestUserInformation(uint64_t steam_id, bool require_name_only){
	if(SteamFriends() == NULL){
		return false;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamFriends()->RequestUserInformation(user_id, require_name_only);
}

//! Sends a message to a Steam group chat room.
bool Steam::sendClanChatMessage(uint64_t chat_id, const String& text){
	if(SteamFriends() == NULL){
		return false;
	}
	CSteamID chat = (uint64)chat_id;
	return SteamFriends()->SendClanChatMessage(chat, text.utf8().get_data());
}

//! User is in a game pressing the talk button (will suppress the microphone for all voice comms from the Steam friends UI).
void Steam::setInGameVoiceSpeaking(uint64_t steam_id, bool speaking){
	if(SteamFriends() != NULL){
		CSteamID user_id = (uint64)steam_id;
		SteamFriends()->SetInGameVoiceSpeaking(user_id, speaking);
	}
}

//! Listens for Steam friends chat messages.
bool Steam::setListenForFriendsMessages(bool intercept){
	if(SteamFriends() == NULL){
		return false;
	}
	return SteamFriends()->SetListenForFriendsMessages(intercept);
}

//! Sets the player name, stores it on the server and publishes the changes to all friends who are online.
void Steam::setPersonaName(const String& name){
	if(SteamFriends() != NULL){
		SteamFriends()->SetPersonaName(name.utf8().get_data());
	}
}

//! Set player as 'Played With' for game.
void Steam::setPlayedWith(uint64_t steam_id){
	if(SteamFriends() != NULL){
		CSteamID user_id = (uint64)steam_id;
		SteamFriends()->SetPlayedWith(user_id);
	}
}

//! Set the game information in Steam; used in 'View Game Info'
bool Steam::setRichPresence(const String& key, const String& value){
	// Rich presence data is automatically shared between friends in the same game.
	// Each user has a set of key/value pairs, up to 20 can be set.
	// Two magic keys (status, connect).
	// setGameInfo() to an empty string deletes the key.
	if(SteamFriends() == NULL){
		return false;
	}
	return SteamFriends()->SetRichPresence(key.utf8().get_data(), value.utf8().get_data());
}


/////////////////////////////////////////////////
///// GAME SEARCH
/////////////////////////////////////////////////
//
//! A keyname and a list of comma separated values: one of which is must be found in order for the match to qualify; fails if a search is currently in progress.
int Steam::addGameSearchParams(const String& key, const String& values){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	return SteamGameSearch()->AddGameSearchParams(key.utf8().get_data(), values.utf8().get_data());
}

//! All players in lobby enter the queue and await a SearchForGameNotificationCallback_t callback. Fails if another search is currently in progress. If not the owner of the lobby or search already in progress this call fails. Periodic callbacks will be sent as queue time estimates change.
int Steam::searchForGameWithLobby(uint64_t lobby_id, int player_min, int player_max){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	CSteamID lobby = (uint64)lobby_id;
	return SteamGameSearch()->SearchForGameWithLobby(lobby, player_min, player_max);
}

//! User enter the queue and await a SearchForGameNotificationCallback_t callback. fails if another search is currently in progress. Periodic callbacks will be sent as queue time estimates change.
int Steam::searchForGameSolo(int player_min, int player_max){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	return SteamGameSearch()->SearchForGameSolo(player_min, player_max);
}

//! After receiving SearchForGameResultCallback_t, accept or decline the game. Multiple SearchForGameResultCallback_t will follow as players accept game until the host starts or cancels the game.
int Steam::acceptGame(){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	return SteamGameSearch()->AcceptGame();
}

//! After receiving SearchForGameResultCallback_t, accept or decline the game. Multiple SearchForGameResultCallback_t will follow as players accept game until the host starts or cancels the game.
int Steam::declineGame(){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	return SteamGameSearch()->DeclineGame();
}

//! After receiving GameStartedByHostCallback_t get connection details to server.
String Steam::retrieveConnectionDetails(uint64_t host_id){
	if(SteamGameSearch() == NULL){
		return "";
	}
	CSteamID host = (uint64)host_id;
	char connection_details;
	SteamGameSearch()->RetrieveConnectionDetails(host, &connection_details, 256);
	// Messy conversion from char to String
	String details;
	details += connection_details;
	return details;
}

//! Leaves queue if still waiting.
int Steam::endGameSearch(){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	return SteamGameSearch()->EndGameSearch();
}

//! A keyname and a list of comma separated values: all the values you allow.
int Steam::setGameHostParams(const String& key, const String& value){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	return SteamGameSearch()->SetGameHostParams(key.utf8().get_data(), value.utf8().get_data());
}

//! Set connection details for players once game is found so they can connect to this server.
int Steam::setConnectionDetails(const String& details, int connection_details){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	return SteamGameSearch()->SetConnectionDetails(details.utf8().get_data(), connection_details);
}

//! Mark server as available for more players with nPlayerMin,nPlayerMax desired. Accept no lobbies with playercount greater than nMaxTeamSize.
int Steam::requestPlayersForGame(int player_min, int player_max, int max_team_size){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	return SteamGameSearch()->RequestPlayersForGame(player_min, player_max, max_team_size);
}

//! Accept the player list and release connection details to players.
int Steam::hostConfirmGameStart(uint64_t game_id){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	return SteamGameSearch()->HostConfirmGameStart(game_id);
}

//! Cancel request and leave the pool of game hosts looking for players.
int Steam::cancelRequestPlayersForGame(){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	return SteamGameSearch()->CancelRequestPlayersForGame();
}

//! Submit a result for one player. does not end the game. ullUniqueGameID continues to describe this game.
int Steam::submitPlayerResult(uint64_t game_id, uint64_t player_id, int player_result){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	CSteamID player = (uint64)player_id;
	return SteamGameSearch()->SubmitPlayerResult(game_id, player, (EPlayerResult_t)player_result);
}

//! Ends the game. no further SubmitPlayerResults for ullUniqueGameID will be accepted.
int Steam::endGame(uint64_t game_id){
	if(SteamGameSearch() == NULL){
		return 9;
	}
	return SteamGameSearch()->EndGame(game_id);
}


/////////////////////////////////////////////////
///// GAME SERVER FUNCTIONS
/////////////////////////////////////////////////
//
// NOTE: The following, if set, must be set before calling LogOn; they may not be changed after.
//
// Game product identifier; currently used by the master server for version checking purposes.
void Steam::setProduct(const String& product){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetProduct(product.utf8().get_data());
}

// Description of the game; required field and is displayed in the Steam server browser.
void Steam::setGameDescription(const String& description){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetGameDescription(description.utf8().get_data());
}

// If your game is a mod, pass the string that identifies it. Default is empty meaning the app is the original game.
void Steam::setModDir(const String& mod_directory){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetModDir(mod_directory.utf8().get_data());
}

// Is this a dedicated server? Default is false.
void Steam::setDedicatedServer(bool dedicated){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetDedicatedServer(dedicated);
}

// NOTE: The following are login functions.
//
// Begin process to login to a persistent game server account. You need to register for callbacks to determine the result of this operation.
void Steam::logOn(const String& token){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->LogOn(token.utf8().get_data());
}

// Login to a generic, anonymous account.
void Steam::logOnAnonymous(){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->LogOnAnonymous();
}

// Begin process of logging game server out of Steam.
void Steam::logOff(){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->LogOff();
}

// Status functions.
bool Steam::serverLoggedOn(){
	if(SteamGameServer() == NULL){
		return false;
	}
	return SteamGameServer()->BLoggedOn();
}

bool Steam::secure(){
	if(SteamGameServer() == NULL){
		return false;
	}
	return SteamGameServer()->BSecure();
}

uint64_t Steam::getServerSteamID(){
	if(SteamGameServer() == NULL){
		return 0;
	}
	CSteamID serverID = SteamGameServer()->GetSteamID();
	return serverID.ConvertToUint64();
}

// Returns true if the master server has requested a restart. Only returns true once per request.
bool Steam::wasRestartRequested(){
	if(SteamGameServer() == NULL){
		return false;
	}
	return SteamGameServer()->WasRestartRequested();
}

// NOTE: These are server state functions and can be changed at any time.
//
// Max player count that will be reported to server browser and client queries.
void Steam::setMaxPlayerCount(int players_max){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetMaxPlayerCount(players_max);
}

// Number of bots. Default is zero.
void Steam::setBotPlayerCount(int bots){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetBotPlayerCount(bots);
}

// Set the naem of the server as it will appear in the server browser.
void Steam::setServerName(const String& name){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetServerName(name.utf8().get_data());
}

// Set name of map to report in server browser.
void Steam::setMapName(const String& map){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetMapName(map.utf8().get_data());
}

// Let people know if your server requires a password.
void Steam::setPasswordProtected(bool password_protected){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetPasswordProtected(password_protected);
}

// Spectator server. Default is zero, meaning it is now used.
void Steam::setSpectatorPort(uint16 port){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetSpectatorPort(port);
}

// Name of spectator server. Only used if spectator port is non-zero.
void Steam::setSpectatorServerName(const String& name){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetSpectatorServerName(name.utf8().get_data());
}

// Call this to clear the whole list of key/values that are sent in rule queries.
void Steam::clearAllKeyValues(){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->ClearAllKeyValues();
}

// Call this to add/update a key/value pair.
void Steam::setKeyValue(const String& key, const String& value){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetKeyValue(key.utf8().get_data(), value.utf8().get_data());
}

// Set a string defining game tags for this server; optional. Allows users to filter in matchmaking/server browser.
void Steam::setGameTags(const String& tags){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetGameTags(tags.utf8().get_data());
}

// Set a string defining game data for this server; optional. Allows users to filter in matchmaking/server browser.
void Steam::setGameData(const String& data){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetGameData(data.utf8().get_data());
}

// Region identifier; optional. Default is empty meaning 'world'.
void Steam::setRegion(const String& region){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetRegion(region.utf8().get_data());
}

// NOTE: These functions are player list management / authentication.
//
// Retrieve ticket to be sent to the entity who wishes to authenticate you (using BeginAuthSession API).
Dictionary Steam::getServerAuthSessionTicket(){
	// Create the dictionary to use
	Dictionary auth_ticket;
	if(SteamGameServer() != NULL){
		uint32_t ticket_size = 1024;
		PoolByteArray buffer;
		buffer.resize(ticket_size);
		uint32_t id = SteamGameServer()->GetAuthSessionTicket(buffer.write().ptr(), ticket_size, &ticket_size);
		// Add this data to the dictionary
		auth_ticket["id"] = id;
		auth_ticket["buffer"] = buffer;
		auth_ticket["size"] = ticket_size;
	}
	return auth_ticket;
}

// Authenticate the ticket from the entity Steam ID to be sure it is valid and isn't reused.
uint32 Steam::beginServerAuthSession(PoolByteArray ticket, int ticket_size, uint64_t steam_id){
	if(SteamGameServer() == NULL){
		return -1;
	}
	CSteamID authSteamID = createSteamID(steam_id);
	return SteamGameServer()->BeginAuthSession(ticket.read().ptr(), ticket_size, authSteamID);
}

// Stop tracking started by beginAuthSession; called when no longer playing game with this entity;
void Steam::endServerAuthSession(uint64_t steam_id){
	if(SteamGameServer() != NULL){
		CSteamID authSteamID = createSteamID(steam_id);
		SteamGameServer()->EndAuthSession(authSteamID);
	}
}

// Cancel auth ticket from getAuthSessionTicket; called when no longer playing game with the entity you gave the ticket to.
void Steam::cancelServerAuthTicket(uint32_t auth_ticket){
	if(SteamGameServer() != NULL){
		SteamGameServer()->CancelAuthTicket(auth_ticket);
	}
}

// After receiving a user's authentication data, and passing it to sendUserConnectAndAuthenticate, use to determine if user owns DLC
int Steam::userHasLicenceForApp(uint64_t steam_id, uint32 app_id){
	if(SteamGameServer() == NULL){
		return 0;
	}
	CSteamID userID = (uint64)steam_id;
	return SteamGameServer()->UserHasLicenseForApp(userID, (AppId_t)app_id);
}

// Ask if user is in specified group; results returned by GSUserGroupStatus_t.
bool Steam::requestUserGroupStatus(uint64_t steam_id, int group_id){
	if(SteamGameServer() == NULL){
		return false;
	}
	CSteamID userID = (uint64)steam_id;
	CSteamID clan_id = (uint64)group_id;
	return SteamGameServer()->RequestUserGroupStatus(userID, clan_id);
}

// NOTE: These are in GameSocketShare mode, where instead of ISteamGameServer creating sockets to talk to master server, it lets the game use its socket to forward messages back and forth.
//
// These are used when you've elected to multiplex the game server's UDP socket rather than having the master server updater use its own sockets.
Dictionary Steam::handleIncomingPacket(int packet, const String& ip, uint16 port){
	Dictionary result;
	if(SteamGameServer() == NULL){
		return result;
	}
	PoolByteArray data;
	data.resize(packet);
	// Resolve address and convert it from IP_Address string to uint32_t
	IP_Address address;
	if(ip.is_valid_ip_address()){
		address = ip;
	}
	else{
		address = IP::get_singleton()->resolve_hostname(ip, IP::TYPE_IPV4);
	}
	// Resolution failed
	if(!address.is_valid()){
		return result;
	}
	uint32_t ip4 = *((uint32_t *)address.get_ipv4());
	if(SteamGameServer()->HandleIncomingPacket(data.write().ptr(), packet, ip4, port)){
		result["data"] = data;
	}
	return result;
}

// AFTER calling HandleIncomingPacket for any packets that came in that frame, call this. This gets a packet that the master server updater needs to send out on UDP. Returns 0 if there are no more packets.
Dictionary Steam::getNextOutgoingPacket(){
	Dictionary packet;
	if(SteamGameServer() == NULL){
		return packet;
	}
	PoolByteArray out;
	int maxOut = 16 * 1024;
	uint32 address;
	uint16 port;
	// Retrieve the packet information
	int length = SteamGameServer()->GetNextOutgoingPacket(&out, maxOut, &address, &port);
	// Place packet information in dictionary and return it
	packet["length"] = length;
	packet["out"] = out;
	packet["address"] = address;
	packet["port"] = port;
	return packet;
}

// Gets the public IP of the server according to Steam.
//uint32 Steam::getPublicIP(){
//	if(SteamGameServer() == NULL){
//		return 0;
//	}
//	return SteamGameServer()->GetPublicIP();
//}

// NOTE: These are heartbeat/advertisement functions.
//
// Call this as often as you like to tell the master server updater whether or not you want it to be active (default: off).
void Steam::setAdvertiseServerActive(bool active){
	if(SteamGameServer() == NULL){
		return;
	}
	SteamGameServer()->SetAdvertiseServerActive(active);
}

// Associate this game server with this clan for the purposes of computing player compatibility.
void Steam::associateWithClan(uint64_t clan_id){
	if(SteamGameServer() == NULL){
		return;
	}
	CSteamID group_id = (uint64)clan_id;
	SteamGameServer()->AssociateWithClan(group_id);
}

// Ask if any of the current players dont want to play with this new player - or vice versa.
void Steam::computeNewPlayerCompatibility(uint64_t steam_id){
	if(SteamGameServer() == NULL){
		return;
	}
	CSteamID userID = (uint64)steam_id;
	SteamGameServer()->ComputeNewPlayerCompatibility(userID);
}


/////////////////////////////////////////////////
///// GAME SERVER STATS
/////////////////////////////////////////////////
//
// Resets the unlock status of an achievement for the specified user.
bool Steam::clearUserAchievement(uint64_t steam_id, const String& name){
	if(SteamGameServerStats() == NULL){
		return false;
	}
	CSteamID userID = (uint64)steam_id;
	return SteamGameServerStats()->ClearUserAchievement(userID, name.utf8().get_data());
}

// Gets the unlock status of the Achievement.
Dictionary Steam::serverGetUserAchievement(uint64_t steam_id, const String& name){
	// Set dictionary to fill in
	Dictionary achievement;
	if(SteamGameServerStats() == NULL){
		return achievement;
	}
	CSteamID user_id = (uint64)steam_id;
	bool unlocked = false;
	bool result = SteamGameServerStats()->GetUserAchievement(user_id, name.utf8().get_data(), &unlocked);
	// Populate the dictionary
	achievement["result"] = result;
	achievement["user"] = steam_id;
	achievement["name"] = name;
	achievement["unlocked"] = unlocked;
	return achievement;
}

// Gets the current value of the a stat for the specified user.
uint32_t Steam::serverGetUserStatInt(uint64_t steam_id, const String& name){
	if(SteamGameServerStats() != NULL){
		CSteamID userID = (uint64)steam_id;
		int32 value = 0;
		if(SteamGameServerStats()->GetUserStat(userID, name.utf8().get_data(), &value)){
			return value;
		}
	}
	return 0;
}

// Gets the current value of the a stat for the specified user.
float Steam::serverGetUserStatFloat(uint64_t steam_id, const String& name){
	if(SteamGameServerStats() != NULL){
		CSteamID userID = (uint64)steam_id;
		float value = 0.0;
		if(SteamGameServerStats()->GetUserStat(userID, name.utf8().get_data(), &value)){
			return value;
		}
	}
	return 0.0;
}

// Asynchronously downloads stats and achievements for the specified user from the server.
void Steam::serverRequestUserStats(uint64_t steam_id){
	if(SteamGameServerStats() != NULL){
		CSteamID userID = (uint64)steam_id;
		SteamAPICall_t api_call = SteamGameServerStats()->RequestUserStats(userID);
		callResultStatReceived.Set(api_call, this, &Steam::stats_received);
	}
}

// Unlocks an achievement for the specified user.
bool Steam::setUserAchievement(uint64_t steam_id, const String& name){
	if(SteamGameServerStats() == NULL){
		return false;
	}
	CSteamID userID = (uint64)steam_id;
	return SteamGameServerStats()->SetUserAchievement(userID, name.utf8().get_data());
}

// Sets / updates the value of a given stat for the specified user.
bool Steam::setUserStatInt(uint64_t steam_id, const String& name, int32 stat){
	if(SteamGameServerStats() == NULL){
		return false;
	}
	CSteamID userID = (uint64)steam_id;
	return SteamGameServerStats()->SetUserStat(userID, name.utf8().get_data(), stat);
}

// Sets / updates the value of a given stat for the specified user.
bool Steam::setUserStatFloat(uint64_t steam_id, const String& name, float stat){
	if(SteamGameServerStats() == NULL){
		return false;
	}
	CSteamID userID = (uint64)steam_id;
	return SteamGameServerStats()->SetUserStat(userID, name.utf8().get_data(), stat);
}

// Send the changed stats and achievements data to the server for permanent storage for the specified user.
void Steam::storeUserStats(uint64_t steam_id){
	if(SteamGameServerStats() != NULL){
		CSteamID userID = (uint64)steam_id;
		SteamGameServerStats()->StoreUserStats(userID);
	}
}

// Updates an AVGRATE stat with new values for the specified user.
bool Steam::updateUserAvgRateStat(uint64_t steam_id, const String& name, float this_session, double session_length){
	if(SteamGameServerStats() == NULL){
		return false;
	}
	CSteamID userID = (uint64)steam_id;
	return SteamGameServerStats()->UpdateUserAvgRateStat(userID, name.utf8().get_data(), this_session, session_length);
}


/////////////////////////////////////////////////
///// HTML SURFACE
/////////////////////////////////////////////////
//
//! Add a header to any HTTP requests from this browser.
void Steam::addHeader(const String& key, const String& value, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->AddHeader(this_handle, key.utf8().get_data(), value.utf8().get_data());
	}
}

//! Sets whether a pending load is allowed or if it should be canceled.  NOTE:You MUST call this in response to a HTML_StartRequest_t callback.
void Steam::allowStartRequest(bool allowed, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->AllowStartRequest(this_handle, allowed);
	}
}

//! Copy the currently selected text from the current page in an HTML surface into the local clipboard.
void Steam::copyToClipboard(uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->CopyToClipboard(this_handle);
	}
}

//! Create a browser object for displaying of an HTML page. NOTE: You MUST call RemoveBrowser when you are done using this browser to free up the resources associated with it. Failing to do so will result in a memory leak.
void Steam::createBrowser(const String& user_agent, const String& user_css){
	if(SteamHTMLSurface() != NULL){
		SteamHTMLSurface()->CreateBrowser(user_agent.utf8().get_data(), user_css.utf8().get_data());
	}
}

//! Run a javascript script in the currently loaded page.
void Steam::executeJavascript(const String& script, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->ExecuteJavascript(this_handle, script.utf8().get_data());
	}
}

//! Find a string in the current page of an HTML surface. This is the equivalent of "ctrl+f" in your browser of choice. It will highlight all of the matching strings. You should call StopFind when the input string has changed or you want to stop searching.
void Steam::find(const String& search, bool currently_in_find, bool reverse, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->Find(this_handle, search.utf8().get_data(), currently_in_find, reverse);
	}
}

//! Retrieves details about a link at a specific position on the current page in an HTML surface.
void Steam::getLinkAtPosition(int x, int y, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->GetLinkAtPosition(this_handle, x, y);
	}
}

//! Navigate back in the page history.
void Steam::goBack(uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->GoBack(this_handle);
	}
}

//! Navigate forward in the page history
void Steam::goForward(uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->GoForward(this_handle);
	}
}

//! Initializes the HTML Surface API. This must be called prior to using any other functions in this interface. You MUST call Shutdown when you are done using the interface to free up the resources associated with it. Failing to do so will result in a memory leak!
void Steam::htmlInit(){
	if(SteamHTMLSurface() != NULL){
		SteamHTMLSurface()->Init();
	}
}

//! Allows you to react to a page wanting to open a javascript modal dialog notification.
void Steam::jsDialogResponse(bool result, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->JSDialogResponse(this_handle, result);
	}
}

//! cUnicodeChar is the unicode character point for this keypress (and potentially multiple chars per press).
void Steam::keyChar(uint32 unicode_char, int key_modifiers, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->KeyChar(this_handle, unicode_char, (ISteamHTMLSurface::EHTMLKeyModifiers)key_modifiers);
	}
}

//! Keyboard interactions, native keycode is the virtual key code value from your OS.
void Steam::keyDown(uint32 native_key_code, int key_modifiers, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->KeyDown(this_handle, native_key_code, (ISteamHTMLSurface::EHTMLKeyModifiers)key_modifiers);
	}
}

//! Keyboard interactions, native keycode is the virtual key code value from your OS.
void Steam::keyUp(uint32 native_key_code, int key_modifiers, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->KeyUp(this_handle, native_key_code, (ISteamHTMLSurface::EHTMLKeyModifiers)key_modifiers);
	}
}

//! Navigate to a specified URL. If you send POST data with pchPostData then the data should be formatted as: name1=value1&name2=value2. You can load any URI scheme supported by Chromium Embedded Framework including but not limited to: http://, https://, ftp://, and file:///. If no scheme is specified then http:// is used.
void Steam::loadURL(const String& url, const String& post_data, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->LoadURL(this_handle, url.utf8().get_data(), post_data.utf8().get_data());
	}
}

//! Tells an HTML surface that a mouse button has been double clicked. The click will occur where the surface thinks the mouse is based on the last call to MouseMove.
void Steam::mouseDoubleClick(int mouse_button, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->MouseDoubleClick(this_handle, (ISteamHTMLSurface::EHTMLMouseButton)mouse_button);
	}
}

//! Tells an HTML surface that a mouse button has been pressed. The click will occur where the surface thinks the mouse is based on the last call to MouseMove.
void Steam::mouseDown(int mouse_button, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->MouseDown(this_handle, (ISteamHTMLSurface::EHTMLMouseButton)mouse_button);
	}
}

//! Tells an HTML surface where the mouse is.
void Steam::mouseMove(int x, int y, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->MouseMove(this_handle, x, y);
	}
}

//! Tells an HTML surface that a mouse button has been released. The click will occur where the surface thinks the mouse is based on the last call to MouseMove.
void Steam::mouseUp(int mouse_button, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->MouseUp(this_handle, (ISteamHTMLSurface::EHTMLMouseButton)mouse_button);
	}
}

//! Tells an HTML surface that the mouse wheel has moved.
void Steam::mouseWheel(int32 delta, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->MouseWheel(this_handle, delta);
	}
}

//! Paste from the local clipboard to the current page in an HTML surface.
void Steam::pasteFromClipboard(uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->PasteFromClipboard(this_handle);
	}
}

//! Refreshes the current page. The reload will most likely hit the local cache instead of going over the network. This is equivalent to F5 or Ctrl+R in your browser of choice.
void Steam::reload(uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->Reload(this_handle);
	}
}

//! You MUST call this when you are done with an HTML surface, freeing the resources associated with it. Failing to call this will result in a memory leak!
void Steam::removeBrowser(uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->RemoveBrowser(this_handle);
	}
}

//! Enable/disable low-resource background mode, where javascript and repaint timers are throttled, resources are more aggressively purged from memory, and audio/video elements are paused. When background mode is enabled, all HTML5 video and audio objects will execute ".pause()" and gain the property "._steam_background_paused = 1". When background mode is disabled, any video or audio objects with that property will resume with ".play()".
void Steam::setBackgroundMode(bool background_mode, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->SetBackgroundMode(this_handle, background_mode);
	}
}

//! Set a webcookie for a specific hostname. You can read more about the specifics of setting cookies here on wikipedia.
void Steam::setCookie(const String& hostname, const String& key, const String& value, const String& path, uint32 expires, bool secure, bool http_only){
	if(SteamHTMLSurface() != NULL){
		SteamHTMLSurface()->SetCookie(hostname.utf8().get_data(), key.utf8().get_data(), value.utf8().get_data(), path.utf8().get_data(), expires, secure, http_only);
	}
}

//! Scroll the current page horizontally.
void Steam::setHorizontalScroll(uint32 absolute_pixel_scroll, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->SetHorizontalScroll(this_handle, absolute_pixel_scroll);
	}
}

//! Tell a HTML surface if it has key focus currently, controls showing the I-beam cursor in text controls amongst other things.
void Steam::setKeyFocus(bool has_key_focus, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->SetKeyFocus(this_handle, has_key_focus);
	}
}

//! Zoom the current page in an HTML surface. The current scale factor is available from HTML_NeedsPaint_t.flPageScale, HTML_HorizontalScroll_t.flPageScale, and HTML_VerticalScroll_t.flPageScale.
void Steam::setPageScaleFactor(float zoom, int point_x, int point_y, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->SetPageScaleFactor(this_handle, zoom, point_x, point_y);
	}
}

//! Sets the display size of a surface in pixels.
void Steam::setSize(uint32 width, uint32 height, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->SetSize(this_handle, width, height);
	}
}

//! Scroll the current page vertically.
void Steam::setVerticalScroll(uint32 absolute_pixel_scroll, uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->SetVerticalScroll(this_handle, absolute_pixel_scroll);
	}
}

//! Shutdown the ISteamHTMLSurface interface, releasing the memory and handles. You MUST call this when you are done using this interface to prevent memory and handle leaks. After calling this then all of the functions provided in this interface will fail until you call Init to reinitialize again.
bool Steam::htmlShutdown(){
	if(SteamHTMLSurface() == NULL){
		return false;
	}
	return SteamHTMLSurface()->Shutdown();
}

//! Cancel a currently running find.
void Steam::stopFind(uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->StopFind(this_handle);
	}
}

//! Stop the load of the current HTML page.
void Steam::stopLoad(uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->StopLoad(this_handle);
	}
}

//! Open the current pages HTML source code in default local text editor, used for debugging.
void Steam::viewSource(uint32 this_handle){
	if(SteamHTMLSurface() != NULL){
		// If no handle is passed use internal one
		if(this_handle == 0){
			this_handle = browser_handle;
		}
		SteamHTMLSurface()->ViewSource(this_handle);
	}
}


/////////////////////////////////////////////////
///// HTTP
/////////////////////////////////////////////////
//
//! Creates a cookie container to store cookies during the lifetime of the process. This API is just for during process lifetime, after steam restarts no cookies are persisted and you have no way to access the cookie container across repeat executions of your process.
uint32_t Steam::createCookieContainer(bool allow_responses_to_modify){
	if(SteamHTTP() == NULL){
		return 0;
	}
	return SteamHTTP()->CreateCookieContainer(allow_responses_to_modify);
}

//! Initializes a new HTTP request.
uint32_t Steam::createHTTPRequest(int request_method, const String& absolute_url){
	if(SteamHTTP() != NULL){
		return SteamHTTP()->CreateHTTPRequest((EHTTPMethod)request_method, absolute_url.utf8().get_data());
	}
	return HTTPREQUEST_INVALID_HANDLE;
}

//! Defers a request which has already been sent by moving it at the back of the queue.
bool Steam::deferHTTPRequest(uint32 request_handle){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->DeferHTTPRequest(request_handle);
}

//! Gets progress on downloading the body for the request.
float Steam::getHTTPDownloadProgressPct(uint32 request_handle){
	float percent_one = 0.0;
	if(SteamHTTP() != NULL){
		SteamHTTP()->GetHTTPDownloadProgressPct(request_handle, &percent_one);
	}
	return percent_one;
}

//! Check if the reason the request failed was because we timed it out (rather than some harder failure).
bool Steam::getHTTPRequestWasTimedOut(uint32 request_handle){
	bool was_timed_out = false;
	if(SteamHTTP() != NULL){
		SteamHTTP()->GetHTTPRequestWasTimedOut(request_handle, &was_timed_out);
	}
	return was_timed_out;
}

//! Gets the body data from an HTTP response.
PoolByteArray Steam::getHTTPResponseBodyData(uint32 request_handle, uint32 buffer_size){
	PoolByteArray body_data;
	body_data.resize(buffer_size);
	if(SteamHTTP() != NULL){
		SteamHTTP()->GetHTTPResponseBodyData(request_handle, body_data.write().ptr(), buffer_size);
	}
	return body_data;
}

//! Gets the size of the body data from an HTTP response.
uint32 Steam::getHTTPResponseBodySize(uint32 request_handle){
	uint32 body_size = 0;
	if(SteamHTTP() != NULL){
		SteamHTTP()->GetHTTPResponseBodySize(request_handle, &body_size);
	}
	return body_size;
}

//! Checks if a header is present in an HTTP response and returns its size.
uint32 Steam::getHTTPResponseHeaderSize(uint32 request_handle, const String& header_name){
	uint32 response_header_size = 0;
	if(SteamHTTP() != NULL){
		SteamHTTP()->GetHTTPResponseHeaderSize(request_handle, header_name.utf8().get_data(), &response_header_size);
	}
	return response_header_size;
}

//! Gets a header value from an HTTP response.
uint8 Steam::getHTTPResponseHeaderValue(uint32 request_handle, const String& header_name, uint32 buffer_size){
	uint8 value_buffer = 0;
	if(SteamHTTP() != NULL){
		SteamHTTP()->GetHTTPResponseHeaderValue(request_handle, header_name.utf8().get_data(), &value_buffer, buffer_size);
	}
	return value_buffer;
}

//! Gets the body data from a streaming HTTP response.
uint8 Steam::getHTTPStreamingResponseBodyData(uint32 request_handle, uint32 offset, uint32 buffer_size){
	uint8 body_data_buffer = 0;
	if(SteamHTTP() != NULL){
		SteamHTTP()->GetHTTPStreamingResponseBodyData(request_handle, offset, &body_data_buffer, buffer_size);
	}
	return body_data_buffer;
}

//! Prioritizes a request which has already been sent by moving it at the front of the queue.
bool Steam::prioritizeHTTPRequest(uint32 request_handle){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->PrioritizeHTTPRequest(request_handle);
}

//! Releases a cookie container, freeing the memory allocated within Steam.
bool Steam::releaseCookieContainer(uint32 cookie_handle){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->ReleaseCookieContainer(cookie_handle);
}

//! Releases an HTTP request handle, freeing the memory allocated within Steam.
bool Steam::releaseHTTPRequest(uint32 request_handle){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->ReleaseHTTPRequest(request_handle);
}

//! Sends an HTTP request.
bool Steam::sendHTTPRequest(uint32 request_handle){
	if(SteamHTTP() == NULL){
		return false;
	}
	SteamAPICall_t call_handle;
	return SteamHTTP()->SendHTTPRequest(request_handle, &call_handle);
}

//! Sends an HTTP request and streams the response back in chunks.
bool Steam::sendHTTPRequestAndStreamResponse(uint32 request_handle){
	if(SteamHTTP() == NULL){
		return false;
	}
	SteamAPICall_t call_handle;
	return SteamHTTP()->SendHTTPRequestAndStreamResponse(request_handle, &call_handle);
}

//! Adds a cookie to the specified cookie container that will be used with future requests.
bool Steam::setHTTPCookie(uint32 cookie_handle, const String& host, const String& url, const String& cookie){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->SetCookie(cookie_handle, host.utf8().get_data(), url.utf8().get_data(), cookie.utf8().get_data());
}

//! Set an absolute timeout in milliseconds for the HTTP request. This is the total time timeout which is different than the network activity timeout which is set with SetHTTPRequestNetworkActivityTimeout which can bump everytime we get more data.
bool Steam::setHTTPRequestAbsoluteTimeoutMS(uint32 request_handle, uint32 milliseconds){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->SetHTTPRequestAbsoluteTimeoutMS(request_handle, milliseconds);
}

//! Set a context value for the request, which will be returned in the HTTPRequestCompleted_t callback after sending the request. This is just so the caller can easily keep track of which callbacks go with which request data. Must be called before sending the request.
bool Steam::setHTTPRequestContextValue(uint32 request_handle, uint64_t context_value){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->SetHTTPRequestContextValue(request_handle, context_value);
}

//! Associates a cookie container to use for an HTTP request.
bool Steam::setHTTPRequestCookieContainer(uint32 request_handle, uint32 cookie_handle){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->SetHTTPRequestCookieContainer(request_handle, cookie_handle);
}

//! Set a GET or POST parameter value on the HTTP request. Must be called prior to sending the request.
bool Steam::setHTTPRequestGetOrPostParameter(uint32 request_handle, const String& name, const String& value){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->SetHTTPRequestGetOrPostParameter(request_handle, name.utf8().get_data(), value.utf8().get_data());
}

//! Set a request header value for the HTTP request. Must be called before sending the request.
bool Steam::setHTTPRequestHeaderValue(uint32 request_handle, const String& header_name, const String& header_value){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->SetHTTPRequestHeaderValue(request_handle, header_name.utf8().get_data(), header_value.utf8().get_data());
}

//! Set the timeout in seconds for the HTTP request.
bool Steam::setHTTPRequestNetworkActivityTimeout(uint32 request_handle, uint32 timeout_seconds){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->SetHTTPRequestNetworkActivityTimeout(request_handle, timeout_seconds);
}

//! Sets the body for an HTTP Post request.
uint8 Steam::setHTTPRequestRawPostBody(uint32 request_handle, const String& content_type, uint32 body_length){
	uint8 body = 0;
	if(SteamHTTP()){
		SteamHTTP()->SetHTTPRequestRawPostBody(request_handle, content_type.utf8().get_data(), &body, body_length);
	}
	return body;
}

//! Sets that the HTTPS request should require verified SSL certificate via machines certificate trust store. This currently only works Windows and macOS.
bool Steam::setHTTPRequestRequiresVerifiedCertificate(uint32 request_handle, bool require_verified_certificate){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->SetHTTPRequestRequiresVerifiedCertificate(request_handle, require_verified_certificate);
}

//! Set additional user agent info for a request.
bool Steam::setHTTPRequestUserAgentInfo(uint32 request_handle, const String& user_agent_info){
	if(SteamHTTP() == NULL){
		return false;
	}
	return SteamHTTP()->SetHTTPRequestUserAgentInfo(request_handle, user_agent_info.utf8().get_data());
}


/////////////////////////////////////////////////
///// INPUT
/////////////////////////////////////////////////
//
//! Reconfigure the controller to use the specified action set.
void Steam::activateActionSet(uint64_t input_handle, uint64_t action_set_handle){
	if(SteamInput() != NULL){
		SteamInput()->ActivateActionSet((InputHandle_t)input_handle, (ControllerActionSetHandle_t)action_set_handle);
	}
}

//! Reconfigure the controller to use the specified action set layer.
void Steam::activateActionSetLayer(uint64_t input_handle, uint64_t action_set_layer_handle){
	if(SteamInput() != NULL){
		SteamInput()->ActivateActionSetLayer((InputHandle_t)input_handle, (ControllerActionSetHandle_t)action_set_layer_handle);
	}
}

//! Reconfigure the controller to stop using the specified action set.
void Steam::deactivateActionSetLayer(uint64_t input_handle, uint64_t action_set_handle){
	if(SteamInput() != NULL){
		SteamInput()->DeactivateActionSetLayer((InputHandle_t)input_handle, (ControllerActionSetHandle_t)action_set_handle);
	}
}

//! Reconfigure the controller to stop using all action set layers.
void Steam::deactivateAllActionSetLayers(uint64_t input_handle){
	if(SteamInput() != NULL){
		SteamInput()->DeactivateAllActionSetLayers((InputHandle_t)input_handle);
	}
}

//! Lookup the handle for an Action Set. Best to do this once on startup, and store the handles for all future API calls.
uint64_t Steam::getActionSetHandle(const String& action_set_name){
	if(SteamInput() != NULL){
		return (uint64_t)SteamInput()->GetActionSetHandle(action_set_name.utf8().get_data());
	}
	return 0;
}

//! Get an action origin that you can use in your glyph look up table or passed into GetGlyphForActionOrigin or GetStringForActionOrigin.
int Steam::getActionOriginFromXboxOrigin(uint64_t input_handle, int origin){
	if(SteamInput() == NULL){
		return 0;
	}
	return SteamInput()->GetActionOriginFromXboxOrigin((InputHandle_t)input_handle, (EXboxOrigin)origin);
}

//! Fill an array with all of the currently active action set layers for a specified controller handle.
Array Steam::getActiveActionSetLayers(uint64_t input_handle){
	Array handles;
	if(SteamInput() != NULL){
		InputActionSetHandle_t *out = new InputActionSetHandle_t[INPUT_MAX_COUNT];
		int ret = SteamInput()->GetActiveActionSetLayers(input_handle, out);
		for(int i = 0; i < ret; i++){
			handles.push_back((int)out[i]);
		}
		delete[] out;
	}
	return handles;
}

//! Returns the current state of the supplied analog game action.
Dictionary Steam::getAnalogActionData(uint64_t input_handle, uint64_t analog_action_handle){
	ControllerAnalogActionData_t data;
	Dictionary action_data;
	memset(&data, 0, sizeof(data));
	if(SteamInput() != NULL){
		data = SteamInput()->GetAnalogActionData((InputHandle_t)input_handle, (ControllerAnalogActionHandle_t)analog_action_handle);
	}
	action_data["mode"] = data.eMode;
	action_data["x"] = data.x;
	action_data["y"] = data.y;
	action_data["active"] = data.bActive;
	return action_data;
}

//! Get the handle of the specified Analog action.
uint64_t Steam::getAnalogActionHandle(const String& action_name){
	if(SteamInput() != NULL){
		return (uint64_t)SteamInput()->GetAnalogActionHandle(action_name.utf8().get_data());
	}
	return 0;
}

//! Get the origin(s) for an analog action within an action.
Array Steam::getAnalogActionOrigins(uint64_t input_handle, uint64_t action_set_handle, uint64_t analog_action_handle){
	Array list;
	if(SteamInput() != NULL){
		EInputActionOrigin *out = new EInputActionOrigin[STEAM_CONTROLLER_MAX_ORIGINS];
		int ret = SteamInput()->GetAnalogActionOrigins((InputHandle_t)input_handle, (ControllerActionSetHandle_t)action_set_handle, (ControllerAnalogActionHandle_t)analog_action_handle, out);
		for (int i = 0; i < ret; i++){
			list.push_back((int)out[i]);
		}
		delete[] out;
	}
	return list;
}

//! Get current controllers handles.
Array Steam::getConnectedControllers(){
	Array list;
	if(SteamInput() != NULL){
		InputHandle_t *handles = new InputHandle_t[INPUT_MAX_COUNT];
		int ret = SteamInput()->GetConnectedControllers(handles);
		printf("[Steam] Inputs found %d controllers.", ret);
		for (int i = 0; i < ret; i++){
			list.push_back((uint64_t)handles[i]);
		}
		delete[] handles;
	}
	return list;
}

//! Returns the associated controller handle for the specified emulated gamepad.
uint64_t Steam::getControllerForGamepadIndex(int index){
	if(SteamInput() != NULL){
		return (uint64_t)SteamInput()->GetControllerForGamepadIndex(index);
	}
	return 0;
}

//! Get the currently active action set for the specified controller.
uint64_t Steam::getCurrentActionSet(uint64_t input_handle){
	if(SteamInput() != NULL){
		return (uint64_t)SteamInput()->GetCurrentActionSet((InputHandle_t)input_handle);
	}
	return 0;
}

//! Get's the major and minor device binding revisions for Steam Input API configurations. Minor revisions are for small changes such as adding a new option action or updating localization in the configuration. When updating a Minor revision only one new configuration needs to be update with the "Use Action Block" flag set. Major revisions are to be used when changing the number of action sets or otherwise reworking configurations to the degree that older configurations are no longer usable. When a user's binding disagree's with the major revision of the current official configuration Steam will forcibly update the user to the new configuration. New configurations will need to be made for every controller when updating the Major revision.
Array Steam::getDeviceBindingRevision(uint64_t input_handle){
	Array revision;
	if(SteamInput() != NULL){
		int major = 0;
		int minor = 0;
		bool success = SteamInput()->GetDeviceBindingRevision((InputHandle_t)input_handle, &major, &minor);
		if(success){
			revision.append(major);
			revision.append(minor);
		}
	}
	return revision;
}

//! Returns the current state of the supplied digital game action.
Dictionary Steam::getDigitalActionData(uint64_t input_handle, uint64_t digital_action_handle){
	ControllerDigitalActionData_t data;
	Dictionary d;
	memset(&data, 0, sizeof(data));
	if(SteamInput() != NULL){
		data = SteamInput()->GetDigitalActionData((InputHandle_t)input_handle, (ControllerDigitalActionHandle_t)digital_action_handle);
	}
	d["bState"] = data.bState;
	d["bActive"] = data.bActive;
	return d;
}

//! Get the handle of the specified digital action.
uint64_t Steam::getDigitalActionHandle(const String& action_name){
	if(SteamInput() != NULL){
		return (uint64_t)SteamInput()->GetDigitalActionHandle(action_name.utf8().get_data());
	}
	return 0;
}

//! Get the origin(s) for an analog action within an action.
Array Steam::getDigitalActionOrigins(uint64_t input_handle, uint64_t action_set_handle, uint64_t digital_action_handle){
	Array list;
	if(SteamInput() != NULL){
		EInputActionOrigin *out = new EInputActionOrigin[STEAM_CONTROLLER_MAX_ORIGINS];
		int ret = SteamInput()->GetDigitalActionOrigins((InputHandle_t)input_handle, (ControllerActionSetHandle_t)action_set_handle, (ControllerDigitalActionHandle_t)digital_action_handle, out);
		for (int i=0; i<ret; i++){
			list.push_back((int)out[i]);
		}
		delete[] out;
	}
	return list;
}

//! Returns the associated gamepad index for the specified controller.
int Steam::getGamepadIndexForController(uint64_t input_handle){
	if(SteamInput() != NULL){
		return SteamInput()->GetGamepadIndexForController((InputHandle_t)input_handle);
	}
	return -1;
}

//! Get a local path to art for on-screen glyph for a particular origin.
String Steam::getGlyphForActionOrigin(int origin){
	if(SteamInput() == NULL || origin < 0 || origin > k_EInputActionOrigin_MaximumPossibleValue){
		return "";
	}
	return SteamInput()->GetGlyphForActionOrigin_Legacy((EInputActionOrigin)origin);
}

//! Get the input type (device model) for the specified controller. 
String Steam::getInputTypeForHandle(uint64_t input_handle){
	if(SteamInput() == NULL){
		return "";
	}
	ESteamInputType inputType = SteamInput()->GetInputTypeForHandle((InputHandle_t)input_handle);
	if(inputType == k_ESteamInputType_SteamController){
		return "Steam controller";
	}
	else if(inputType == k_ESteamInputType_XBox360Controller){
		return "XBox 360 controller";
	}
	else if(inputType == k_ESteamInputType_XBoxOneController){
		return "XBox One controller";
	}
	else if(inputType == k_ESteamInputType_GenericGamepad){
		return "Generic XInput";
	}
	else if(inputType == k_ESteamInputType_PS4Controller){
		return "PS4 controller";
	}
	else if(inputType == k_ESteamInputType_AppleMFiController){
		return "Apple iController";
	}
	else if(inputType == k_ESteamInputType_AndroidController){
		return "Android Controller";
	}
	else if(inputType == k_ESteamInputType_SwitchJoyConPair){
		return "Switch Jon Cons (Pair)";
	}
	else if(inputType == k_ESteamInputType_SwitchJoyConSingle){
		return "Switch Jon Con (Single)";
	}
	else if(inputType == k_ESteamInputType_SwitchProController){
		return "Switch Pro Controller";
	}
	else if(inputType == k_ESteamInputType_MobileTouch){
		return "Mobile Touch";
	}
	else if(inputType == k_ESteamInputType_PS3Controller){
		return "PS3 Controller";
	}
	else if(inputType == k_ESteamInputType_PS5Controller){
		return "PS5 Controller";
	}
	else{
		return "Unknown";
	}
}

//! Returns raw motion data for the specified controller.
Dictionary Steam::getMotionData(uint64_t input_handle){
	Dictionary motion_data;
	if(SteamInput() != NULL){
		ControllerMotionData_t data;
		memset(&data, 0, sizeof(data));
		data = SteamInput()->GetMotionData((InputHandle_t)input_handle);
		motion_data["rot_quat_x"] = data.rotQuatX;
		motion_data["rot_quat_y"] = data.rotQuatY;
		motion_data["rot_quat_z"] = data.rotQuatZ;
		motion_data["rot_quat_w"] = data.rotQuatW;
		motion_data["pos_accel_x"] = data.posAccelX;
		motion_data["pos_accel_y"] = data.posAccelY;
		motion_data["pos_accel_z"] = data.posAccelZ;
		motion_data["rot_vel_x"] = data.rotVelX;
		motion_data["rot_vel_y"] = data.rotVelY;
		motion_data["rot_vel_z"] = data.rotVelZ;
	}
	return motion_data;
}

//! Get the Steam Remote Play session ID associated with a device, or 0 if there is no session associated with it. See isteamremoteplay.h for more information on Steam Remote Play sessions.
int Steam::getRemotePlaySessionID(uint64_t input_handle){
	if(SteamInput() == NULL){
		return 0;
	}
	return SteamInput()->GetRemotePlaySessionID((InputHandle_t)input_handle);
}

//! Returns a localized string (from Steam's language setting) for the specified origin.
String Steam::getStringForActionOrigin(int origin){
	if(SteamInput() == NULL){
		return "";
	}
	return SteamInput()->GetStringForActionOrigin((EInputActionOrigin)origin);
}

//! Start SteamInputs interface.
bool Steam::inputInit(bool explicitly_call_runframe){
	if(SteamInput() != NULL){
		return SteamInput()->Init(explicitly_call_runframe);
	}
	return false;
}

//! Stop SteamInputs interface.
bool Steam::inputShutdown(){
	if(SteamInput() != NULL){
		return SteamInput()->Shutdown();
	}
	return false;
}

//! Set the controller LED color on supported controllers.
void Steam::setLEDColor(uint64_t input_handle, int color_r, int color_g, int color_b, int flags){
	if(SteamInput() != NULL){
		SteamInput()->SetLEDColor((InputHandle_t)input_handle, color_r, color_g, color_b, flags);
	}
}

//! Syncronize controllers.
void Steam::runFrame(bool reserved_value){
	if(SteamInput() != NULL){
		SteamInput()->RunFrame(reserved_value);
	}
}

//! Invokes the Steam overlay and brings up the binding screen.
bool Steam::showBindingPanel(uint64_t input_handle){
	if(SteamInput() != NULL){
		return SteamInput()->ShowBindingPanel((InputHandle_t)input_handle);
	}
	return false;
}

//! Stops the momentum of an analog action (where applicable, ie a touchpad w/ virtual trackball settings).
void Steam::stopAnalogActionMomentum(uint64_t input_handle, uint64_t action){
	if(SteamInput() != NULL){
		SteamInput()->StopAnalogActionMomentum((InputHandle_t)input_handle, (InputAnalogActionHandle_t)action);
	}
}

//! Get the equivalent origin for a given controller type or the closest controller type that existed in the SDK you built into your game if eDestinationInputType is k_ESteamInputType_Unknown. This action origin can be used in your glyph look up table or passed into GetGlyphForActionOrigin or GetStringForActionOrigin.
int Steam::translateActionOrigin(int destination_input, int source_origin){
	if(SteamInput() == NULL){
		return 0;
	}
	return SteamInput()->TranslateActionOrigin((ESteamInputType)destination_input, (EInputActionOrigin)source_origin);
}

//! Triggers a (low-level) haptic pulse on supported controllers.
void Steam::triggerHapticPulse(uint64_t input_handle, int target_pad, int duration){
	if(SteamInput() != NULL){
		SteamInput()->Legacy_TriggerHapticPulse((InputHandle_t)input_handle, (ESteamControllerPad)target_pad, duration);
	}
}

//! Triggers a repeated haptic pulse on supported controllers.
void Steam::triggerRepeatedHapticPulse(uint64_t input_handle, int target_pad, int duration, int offset, int repeat, int flags){
	if(SteamInput() != NULL){
		SteamInput()->Legacy_TriggerRepeatedHapticPulse((InputHandle_t)input_handle, (ESteamControllerPad)target_pad, duration, offset, repeat, flags);
	}
}

//! Trigger a vibration event on supported controllers.
void Steam::triggerVibration(uint64_t input_handle, uint16_t left_speed, uint16_t right_speed){
	if(SteamInput() != NULL){
		SteamInput()->TriggerVibration((InputHandle_t)input_handle, (unsigned short)left_speed, (unsigned short)right_speed);
	}
}

//! Set the absolute path to the Input Action Manifest file containing the in-game actions and file paths to the official configurations. Used in games that bundle Steam Input configurations inside of the game depot instead of using the Steam Workshop.
bool Steam::setInputActionManifestFilePath(const String& manifest_path){
	if(SteamInput() == NULL){
		return false;
	}
	return SteamInput()->SetInputActionManifestFilePath(manifest_path.utf8().get_data());
}

//! Waits on an IPC event from Steam sent when there is new data to be fetched from the data drop. Returns true when data was recievied before the timeout expires. Useful for games with a dedicated input thread.
bool Steam::waitForData(bool wait_forever, uint32 timeout){
	if(SteamInput() == NULL){
		return false;
	}
	return SteamInput()->BWaitForData(wait_forever, timeout);
}

//! Returns true if new data has been received since the last time action data was accessed via GetDigitalActionData or GetAnalogActionData. The game will still need to call SteamInput()->RunFrame() or SteamAPI_RunCallbacks() before this to update the data stream.
bool Steam::newDataAvailable(){
	if(SteamInput() == NULL){
		return false;
	}
	return SteamInput()->BNewDataAvailable();
}

//! Enable SteamInputDeviceConnected_t and SteamInputDeviceDisconnected_t callbacks. Each controller that is already connected will generate a device connected callback when you enable them.
void Steam::enableDeviceCallbacks(){
	if(SteamInput() != NULL){
		SteamInput()->EnableDeviceCallbacks();
	}
}

//! Get a local path to a PNG file for the provided origin's glyph. 
String Steam::getGlyphPNGForActionOrigin(int origin, int size, uint32 flags){
	if(SteamInput() == NULL){
		return "";
	}
	return SteamInput()->GetGlyphPNGForActionOrigin((EInputActionOrigin)origin, (ESteamInputGlyphSize)size, flags);
}

//! Get a local path to a SVG file for the provided origin's glyph. 
String Steam::getGlyphSVGForActionOrigin(int origin, uint32 flags){
	if(SteamInput() == NULL){
		return "";
	}
	return SteamInput()->GetGlyphSVGForActionOrigin((EInputActionOrigin)origin, flags);
}

//! Trigger a vibration event on supported controllers including Xbox trigger impulse rumble - Steam will translate these commands into haptic pulses for Steam Controllers.
void Steam::triggerVibrationExtended(uint64_t input_handle, uint16_t left_speed, uint16_t right_speed, uint16_t left_trigger_speed, uint16_t right_trigger_speed){
	if(SteamInput() != NULL){
		SteamInput()->TriggerVibrationExtended((InputHandle_t)input_handle, (unsigned short)left_speed, (unsigned short)right_speed, (unsigned short)left_trigger_speed, (unsigned short)right_trigger_speed);
	}
}

//! Send a haptic pulse, works on Steam Deck and Steam Controller devices.
void Steam::triggerSimpleHapticEvent(uint64_t input_handle, int haptic_location, uint8 intensity, const String& gain_db, uint8 other_intensity, const String& other_gain_db){
	if(SteamInput() != NULL){
		// Convert the strings over to char
		char *gain = new char[sizeof(gain_db)];
		strcpy(gain, gain_db.utf8().get_data());
		char *other_gain = new char[sizeof(other_gain_db)];
		strcpy(other_gain, other_gain_db.utf8().get_data());
		SteamInput()->TriggerSimpleHapticEvent((InputHandle_t)input_handle, (EControllerHapticLocation)haptic_location, intensity, *gain, other_intensity, *other_gain);
		delete[] gain;
		delete[] other_gain;
	}
}

//! Returns a localized string (from Steam's language setting) for the specified Xbox controller origin.
String Steam::getStringForXboxOrigin(int origin){
	if(SteamInput() == NULL){
		return "";
	}
	return SteamInput()->GetStringForXboxOrigin((EXboxOrigin)origin);
}

//! Get a local path to art for on-screen glyph for a particular Xbox controller origin
String Steam::getGlyphForXboxOrigin(int origin){
	if(SteamInput() == NULL){
		return "";
	}
	return SteamInput()->GetGlyphForXboxOrigin((EXboxOrigin)origin);
}

//! Get a bitmask of the Steam Input Configuration types opted in for the current session. Returns ESteamInputConfigurationEnableType values.	
//! Note: user can override the settings from the Steamworks Partner site so the returned values may not exactly match your default configuration.
uint16 Steam::getSessionInputConfigurationSettings(){
	if(SteamInput() == NULL){
		return 0;
	}
	return SteamInput()->GetSessionInputConfigurationSettings();
}

//! Returns a localized string (from Steam's language setting) for the user-facing action name corresponding to the specified handle.
String Steam::getStringForDigitalActionName(uint64_t action_handle){
	if(SteamInput() == NULL){
		return "";
	}
	return SteamInput()->GetStringForDigitalActionName((InputDigitalActionHandle_t)action_handle);
}

//! Returns a localized string (from Steam's language setting) for the user-facing action name corresponding to the specified handle.
String Steam::getStringForAnalogActionName(uint64_t action_handle){
	if(SteamInput() == NULL){
		return "";
	}
	return SteamInput()->GetStringForAnalogActionName((InputAnalogActionHandle_t)action_handle);
}


/////////////////////////////////////////////////
///// INVENTORY
/////////////////////////////////////////////////
//
//! When dealing with any inventory handles, you should call CheckResultSteamID on the result handle when it completes to verify that a remote player is not pretending to have a different user's inventory.
//! Also, you must call DestroyResult on the provided inventory result when you are done with it.
//!
//! Grant a specific one-time promotional item to the current user.
int32 Steam::addPromoItem(uint32 item){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
		if(SteamInventory()->AddPromoItem(&new_inventory_handle, item)){
			// Update the internally stored handle
			inventory_handle = new_inventory_handle;
		}
	}
	return new_inventory_handle;
}

//! Grant a specific one-time promotional items to the current user.
int32 Steam::addPromoItems(PoolIntArray items){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
		int count = items.size();
		SteamItemDef_t *new_items = new SteamItemDef_t[items.size()];
		for(int i = 0; i < count; i++){
			new_items[i] = items[i];
		}
		if(SteamInventory()->AddPromoItems(&new_inventory_handle, new_items, count)){
			// Update the internally stored handle
			inventory_handle = new_inventory_handle;
		}
		delete[] new_items;
	}
	return new_inventory_handle;
}

//! Checks whether an inventory result handle belongs to the specified Steam ID.
bool Steam::checkResultSteamID(uint64_t steam_id_expected, int32 this_inventory_handle){
	if(SteamInventory() == NULL){
		return false;
	}
	CSteamID steam_id = (uint64)steam_id_expected;
	// If no inventory handle is passed, use internal one
	if(this_inventory_handle == 0){
		this_inventory_handle = inventory_handle;
	}
	return SteamInventory()->CheckResultSteamID((SteamInventoryResult_t)this_inventory_handle, steam_id);
}

//! Consumes items from a user's inventory. If the quantity of the given item goes to zero, it is permanently removed.
int32 Steam::consumeItem(uint64_t item_consume, uint32 quantity){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
	 	if(SteamInventory()->ConsumeItem(&new_inventory_handle, (SteamItemInstanceID_t)item_consume, quantity)){
	 		// Update the internally stored handle
			inventory_handle = new_inventory_handle;
	 	}
	}
	return new_inventory_handle;
}

//! Deserializes a result set and verifies the signature bytes.
int32 Steam::deserializeResult(PoolByteArray buffer){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
		if(SteamInventory()->DeserializeResult(&new_inventory_handle, &buffer, buffer.size(), false)){
			// Update the internally stored handle
			inventory_handle = new_inventory_handle;
		}
	}
	return new_inventory_handle;
}

//! Destroys a result handle and frees all associated memory.
void Steam::destroyResult(int this_inventory_handle){
	if(SteamInventory() != NULL){
		// If no inventory handle is passed, use internal one
		if(this_inventory_handle == 0){
			this_inventory_handle = inventory_handle;
		}	
		SteamInventory()->DestroyResult((SteamInventoryResult_t)this_inventory_handle);
	}
}

//! Grant one item in exchange for a set of other items.
int32 Steam::exchangeItems(const PoolIntArray output_items, const uint32 output_quantity, const uint64_t input_items, const uint32 input_quantity){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
		if(SteamInventory()->ExchangeItems(&new_inventory_handle, output_items.read().ptr(), &output_quantity, 1, (const uint64 *)input_items, &input_quantity, 1)){
			// Update the internally stored handle
			inventory_handle = new_inventory_handle;
		}
	}
	return new_inventory_handle;
}

//! Grants specific items to the current user, for developers only.
int32 Steam::generateItems(const PoolIntArray items, const uint32 quantity){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
		if(SteamInventory()->GenerateItems(&new_inventory_handle, items.read().ptr(), &quantity, items.size())){
			// Update the internally stored handle
			inventory_handle = new_inventory_handle;
		}
	}
	return new_inventory_handle;
}

//! Start retrieving all items in the current users inventory.
int32 Steam::getAllItems(){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
		if(SteamInventory()->GetAllItems(&new_inventory_handle)){
			// Update the internally stored handle
			inventory_handle = new_inventory_handle;
		}
	}
	return new_inventory_handle;
}

//! Gets a string property from the specified item definition.  Gets a property value for a specific item definition.
String Steam::getItemDefinitionProperty(uint32 definition, const String& name){
	if(SteamInventory() == NULL){
		return "";
	}
	uint32 buffer_size = STEAM_BUFFER_SIZE;
	char *buffer = new char[buffer_size];
	SteamInventory()->GetItemDefinitionProperty(definition, name.utf8().get_data(), buffer, &buffer_size);
	String property = String::utf8(buffer, buffer_size);
	delete[] buffer;
	return property;
}

//! Gets the state of a subset of the current user's inventory.
int32 Steam::getItemsByID(const uint64_t id_array, uint32 count){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
		if(SteamInventory()->GetItemsByID(&new_inventory_handle, (const uint64 *)id_array, count)){
			// Update the internally stored handle
			inventory_handle = new_inventory_handle;
		}
	}
	return new_inventory_handle;
}

//! After a successful call to RequestPrices, you can call this method to get the pricing for a specific item definition.
uint64_t Steam::getItemPrice(uint32 definition){
	if(SteamInventory() == NULL){
		return 0;
	}
	uint64 price = 0;
	uint64 basePrice = 0;
	SteamInventory()->GetItemPrice(definition, &price, &basePrice);
	return price;
}

//! After a successful call to RequestPrices, you can call this method to get all the pricing for applicable item definitions. Use the result of GetNumItemsWithPrices as the the size of the arrays that you pass in.
Array Steam::getItemsWithPrices(uint32 length){
	if(SteamInventory() == NULL){
		return Array();
	}
	// Create the return array
	Array price_array;
	// Create a temporary array
	SteamItemDef_t *ids = new SteamItemDef_t[length];
	uint64 *prices = new uint64[length];
	uint64 *base_prices = new uint64[length];
	if(SteamInventory()->GetItemsWithPrices(ids, prices, base_prices, length)){
		for(uint32 i = 0; i < length; i++){
			Dictionary price_group;
			price_group["item"] = ids[i];
			price_group["price"] = (uint64_t)prices[i];
			price_group["base_prices"] = (uint64_t)base_prices[i];
			price_array.append(price_group);
		}
	}
	delete[] ids;
	delete[] prices;
	delete[] base_prices;
	return price_array;
}

//! After a successful call to RequestPrices, this will return the number of item definitions with valid pricing.
uint32 Steam::getNumItemsWithPrices(){
	if(SteamInventory() == NULL){
		return 0;
	}
	return SteamInventory()->GetNumItemsWithPrices();
}

//! Gets the dynamic properties from an item in an inventory result set.
String Steam::getResultItemProperty(uint32 index, const String& name, int32 this_inventory_handle){
	if(SteamInventory() != NULL){
		// Set up variables to fill
		uint32 buffer_size = 256;
		char *value = new char[buffer_size];
		// If no inventory handle is passed, use internal one
		if(this_inventory_handle == 0){
			this_inventory_handle = inventory_handle;
		}
		SteamInventory()->GetResultItemProperty((SteamInventoryResult_t)this_inventory_handle, index, name.utf8().get_data(), (char*)value, &buffer_size);
		String property = value;
		delete[] value;
		return property;
	}
	return "";
}

//! Get the items associated with an inventory result handle.
Array Steam::getResultItems(int32 this_inventory_handle){
	if(SteamInventory() == NULL){
		return Array();
	}
	// Set up return array
	Array items;
	uint32 size = 0;
	if(SteamInventory()->GetResultItems((SteamInventoryResult_t)this_inventory_handle, NULL, &size)){
		items.resize(size);
		SteamItemDetails_t *item_array = new SteamItemDetails_t[size];
		// If no inventory handle is passed, use internal one
		if(this_inventory_handle == 0){
			this_inventory_handle = inventory_handle;
		}
		if(SteamInventory()->GetResultItems((SteamInventoryResult_t)this_inventory_handle, item_array, &size)){
			for(uint32 i = 0; i < size; i++){
				items.push_back((uint64_t)item_array[i].m_itemId);
			}
		}
		delete[] item_array;
	}
	return items;
}

//! Find out the status of an asynchronous inventory result handle.
String Steam::getResultStatus(int32 this_inventory_handle){
	if(SteamInventory() == NULL){
		return "";
	}
	// If no inventory handle is passed, use internal one
	if(this_inventory_handle == 0){
		this_inventory_handle = inventory_handle;
	}
	int result = SteamInventory()->GetResultStatus((SteamInventoryResult_t)this_inventory_handle);
	// Parse result
	if(result == k_EResultPending){
		return "Still in progress.";
	}
	else if(result == k_EResultOK){
		return "Finished successfully.";
	}
	else if(result == k_EResultExpired){
		return "Finished but may be out-of-date.";
	}
	else if(result == k_EResultInvalidParam){
		return "ERROR: invalid API call parameters.";
	}
	else if(result == k_EResultServiceUnavailable){
		return "ERROR: server temporarily down; retry later.";
	}
	else if(result == k_EResultLimitExceeded){
		return "ERROR: operation would exceed per-user inventory limits.";
	}
	else{
		return "ERROR: generic / unknown.";
	}
}

//! Gets the server time at which the result was generated.
uint32 Steam::getResultTimestamp(int32 this_inventory_handle){
	if(SteamInventory() == NULL){
		return 0;
	}
	// If no inventory handle is passed, use internal one
	if(this_inventory_handle == 0){
		this_inventory_handle = inventory_handle;
	}
	return SteamInventory()->GetResultTimestamp((SteamInventoryResult_t)this_inventory_handle);
}

//! Grant all potential one-time promotional items to the current user.
int32 Steam::grantPromoItems(){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
		if(SteamInventory()->GrantPromoItems(&new_inventory_handle)){
			// Update the internally stored handle
			inventory_handle = new_inventory_handle;
		}
	}
	return new_inventory_handle;
}

//! Triggers an asynchronous load and refresh of item definitions.
bool Steam::loadItemDefinitions(){
	if(SteamInventory() == NULL){
		return false;
	}
	return SteamInventory()->LoadItemDefinitions();
}

//! Request the list of "eligible" promo items that can be manually granted to the given user.
void Steam::requestEligiblePromoItemDefinitionsIDs(uint64_t steam_id){
	if(SteamInventory() != NULL){
		CSteamID user_id = (uint64)steam_id;
		SteamAPICall_t api_call = SteamInventory()->RequestEligiblePromoItemDefinitionsIDs(user_id);
		callResultEligiblePromoItemDefIDs.Set(api_call, this, &Steam::inventory_eligible_promo_item);
	}
}

//! Request prices for all item definitions that can be purchased in the user's local currency. A SteamInventoryRequestPricesResult_t call result will be returned with the user's local currency code. After that, you can call GetNumItemsWithPrices and GetItemsWithPrices to get prices for all the known item definitions, or GetItemPrice for a specific item definition.
void Steam::requestPrices(){
	if(SteamInventory() != NULL){
		SteamAPICall_t api_call = SteamInventory()->RequestPrices();
		callResultRequestPrices.Set(api_call, this, &Steam::inventory_request_prices_result);
	}
}

//! Serialized result sets contain a short signature which can't be forged or replayed across different game sessions.
String Steam::serializeResult(int32 this_inventory_handle){
	String result_serialized;
	if(SteamInventory() != NULL){
		// If no inventory handle is passed, use internal one
		if(this_inventory_handle == 0){
			this_inventory_handle = inventory_handle;
		}
		// Set up return array
		uint32 buffer_size = STEAM_BUFFER_SIZE;
		char *buffer = new char[buffer_size];
		if(SteamInventory()->SerializeResult((SteamInventoryResult_t)this_inventory_handle, buffer, &buffer_size)){
			result_serialized = String::utf8(buffer, buffer_size);
		}
		delete[] buffer;
	}
	return result_serialized;
}

//! Starts the purchase process for the user, given a "shopping cart" of item definitions that the user would like to buy. The user will be prompted in the Steam Overlay to complete the purchase in their local currency, funding their Steam Wallet if necessary, etc.
void Steam::startPurchase(const PoolIntArray items, const uint32 quantity){
	if(SteamInventory() != NULL){
		SteamAPICall_t api_call = SteamInventory()->StartPurchase(items.read().ptr(), &quantity, items.size());
		callResultStartPurchase.Set(api_call, this, &Steam::inventory_start_purchase_result);
	}
}

//! Transfer items between stacks within a user's inventory.
int32 Steam::transferItemQuantity(uint64_t item_id, uint32 quantity, uint64_t item_destination, bool split){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
		if(split){
			if(SteamInventory()->TransferItemQuantity(&new_inventory_handle, (SteamItemInstanceID_t)item_id, quantity, k_SteamItemInstanceIDInvalid)){
				// Update the internally stored handle
				inventory_handle = new_inventory_handle;
			}
		}
		else{
			if(SteamInventory()->TransferItemQuantity(&new_inventory_handle, (SteamItemInstanceID_t)item_id, quantity, (SteamItemInstanceID_t)item_destination)){
				// Update the internally stored handle
				inventory_handle = new_inventory_handle;
			}
		}
	}
	return new_inventory_handle;
}

//! Trigger an item drop if the user has played a long enough period of time.
int32 Steam::triggerItemDrop(uint32 definition){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
		if(SteamInventory()->TriggerItemDrop(&new_inventory_handle, (SteamItemDef_t)definition)){
			// Update the internally stored handle
			inventory_handle = new_inventory_handle;
		}
	}
	return new_inventory_handle;
}

//! Starts a transaction request to update dynamic properties on items for the current user. This call is rate-limited by user, so property modifications should be batched as much as possible (e.g. at the end of a map or game session). After calling SetProperty or RemoveProperty for all the items that you want to modify, you will need to call SubmitUpdateProperties to send the request to the Steam servers. A SteamInventoryResultReady_t callback will be fired with the results of the operation.
void Steam::startUpdateProperties(){
	if(SteamInventory() != NULL){
		inventory_update_handle = SteamInventory()->StartUpdateProperties();
	}
}

//! Submits the transaction request to modify dynamic properties on items for the current user. See StartUpdateProperties.
int32 Steam::submitUpdateProperties(int32 this_inventory_update_handle){
	int32 new_inventory_handle = 0;
	if(SteamInventory() != NULL){
		// If no inventory update handle is passed, use internal one
		if(this_inventory_update_handle == 0){
			this_inventory_update_handle = inventory_update_handle;
		}
		if(SteamInventory()->SubmitUpdateProperties((SteamInventoryUpdateHandle_t)this_inventory_update_handle, &new_inventory_handle)){
			// Update the internally stored handle
			inventory_handle = new_inventory_handle;
		}
	}
	return new_inventory_handle;
}

//! Removes a dynamic property for the given item.
bool Steam::removeProperty(uint64_t item_id, const String& name, int32 this_inventory_update_handle){
	if(SteamInventory() == NULL){
		return false;
	}
	// If no inventory update handle is passed, use internal one
	if(this_inventory_update_handle == 0){
		this_inventory_update_handle = inventory_update_handle;
	}
	return SteamInventory()->RemoveProperty((SteamInventoryUpdateHandle_t)this_inventory_update_handle, (SteamItemInstanceID_t)item_id, name.utf8().get_data());
}

//! Sets a dynamic property for the given item. Supported value types are strings.
bool Steam::setPropertyString(uint64_t item_id, const String& name, const String& value, int32 this_inventory_update_handle){
	if(SteamInventory() == NULL){
		return false;
	}
	// If no inventory update handle is passed, use internal one
	if(this_inventory_update_handle == 0){
		this_inventory_update_handle = inventory_update_handle;
	}
	return SteamInventory()->SetProperty((SteamInventoryUpdateHandle_t)this_inventory_update_handle, (SteamItemInstanceID_t)item_id, name.utf8().get_data(), value.utf8().get_data());
}

//! Sets a dynamic property for the given item. Supported value types are boolean.
bool Steam::setPropertyBool(uint64_t item_id, const String& name, bool value, int32 this_inventory_update_handle){
	if(SteamInventory() == NULL){
		return false;
	}
	// If no inventory update handle is passed, use internal one
	if(this_inventory_update_handle == 0){
		this_inventory_update_handle = inventory_update_handle;
	}
	return SteamInventory()->SetProperty((SteamInventoryUpdateHandle_t)this_inventory_update_handle, (SteamItemInstanceID_t)item_id, name.utf8().get_data(), value);
}

//! Sets a dynamic property for the given item. Supported value types are 64 bit integers.
bool Steam::setPropertyInt(uint64_t item_id, const String& name, uint64_t value, int32 this_inventory_update_handle){
	if(SteamInventory() == NULL){
		return false;
	}
	// If no inventory update handle is passed, use internal one
	if(this_inventory_update_handle == 0){
		this_inventory_update_handle = inventory_update_handle;
	}
	return SteamInventory()->SetProperty((SteamInventoryUpdateHandle_t)this_inventory_update_handle, (SteamItemInstanceID_t)item_id, name.utf8().get_data(), (int64)value);
}

//! Sets a dynamic property for the given item. Supported value types are 32 bit floats.
bool Steam::setPropertyFloat(uint64_t item_id, const String& name, float value, int32 this_inventory_update_handle){
	if(SteamInventory() == NULL){
		return false;
	}
	// If no inventory update handle is passed, use internal one
	if(this_inventory_update_handle == 0){
		this_inventory_update_handle = inventory_update_handle;
	}
	return SteamInventory()->SetProperty((SteamInventoryUpdateHandle_t)this_inventory_update_handle, (SteamItemInstanceID_t)item_id, name.utf8().get_data(), value);
}


/////////////////////////////////////////////////
///// MATCHMAKING
/////////////////////////////////////////////////
//
//! Gets an array of the details of the favorite game servers.
Array Steam::getFavoriteGames(){
	if(SteamMatchmaking() == NULL){
		return Array();
	}
	int count = SteamMatchmaking()->GetFavoriteGameCount();
	Array favorites;
	for(int i = 0; i < count; i++){
		Dictionary favorite;
		AppId_t app_id = 0;
		uint32 ip = 0;
		uint16 port = 0;
		uint16 query_port = 0;
		uint32 flags = 0;
		uint32 last_played = 0;
		favorite["ret"] = SteamMatchmaking()->GetFavoriteGame(i, &app_id, &ip, &port, &query_port, &flags, &last_played);
		if(favorite["ret"]){
			favorite["app"] = app_id;
			// Convert the IP address back to a string
			const int NBYTES = 4;
			uint8 octet[NBYTES];
			char favoriteIP[16];
			for(int j = 0; j < NBYTES; j++){
				octet[j] = ip >> (j * 8);
			}
			sprintf(favoriteIP, "%d.%d.%d.%d", octet[0], octet[1], octet[2], octet[3]);
			favorite["ip"] = favoriteIP;
			favorite["game_port"] = port;
			favorite["query_port"] = query_port;
			favorite["flags"] = flags;
			favorite["played"] = last_played;
			favorites.append(favorite);
		}
	}
	return favorites;
}

//! Adds the game server to the local list; updates the time played of the server if it already exists in the list.
int Steam::addFavoriteGame(uint32 ip, uint16 port, uint16 query_port, uint32 flags, uint32 last_played){
	if(SteamMatchmaking() == NULL){
		return 0;
	}
	return SteamMatchmaking()->AddFavoriteGame(current_app_id, ip, port, query_port, flags, last_played);
}

//! Removes the game server from the local storage; returns true if one was removed.
bool Steam::removeFavoriteGame(uint32 app_id, uint32 ip, uint16 port, uint16 query_port, uint32 flags){
	if(SteamMatchmaking() == NULL){
		return false;
	}
	return SteamMatchmaking()->RemoveFavoriteGame((AppId_t)app_id, ip, port, query_port, flags);
}

//! Get a list of relevant lobbies.
void Steam::requestLobbyList(){
	if(SteamMatchmaking() != NULL){
		SteamAPICall_t api_call = SteamMatchmaking()->RequestLobbyList();
		callResultLobbyList.Set(api_call, this, &Steam::lobby_match_list);
	}
}

//! Adds a string comparison filter to the next RequestLobbyList call.
void Steam::addRequestLobbyListStringFilter(const String& key_to_match, const String& value_to_match, int comparison_type){
	if(SteamMatchmaking() != NULL){
		SteamMatchmaking()->AddRequestLobbyListStringFilter(key_to_match.utf8().get_data(), value_to_match.utf8().get_data(), (ELobbyComparison)comparison_type);
	}
}

//! Adds a numerical comparison filter to the next RequestLobbyList call.
void Steam::addRequestLobbyListNumericalFilter(const String& key_to_match, int value_to_match, int comparison_type){
	if(SteamMatchmaking() != NULL){
		SteamMatchmaking()->AddRequestLobbyListNumericalFilter(key_to_match.utf8().get_data(), value_to_match, (ELobbyComparison)comparison_type);
	}
}

//! Returns results closest to the specified value. Multiple near filters can be added, with early filters taking precedence.
void Steam::addRequestLobbyListNearValueFilter(const String& key_to_match, int value_to_be_close_to){
	if(SteamMatchmaking() != NULL){
		SteamMatchmaking()->AddRequestLobbyListNearValueFilter(key_to_match.utf8().get_data(), value_to_be_close_to);
	}
}

//! Returns only lobbies with the specified number of slots available.
void Steam::addRequestLobbyListFilterSlotsAvailable(int slots_available){
	if(SteamMatchmaking() != NULL){
		SteamMatchmaking()->AddRequestLobbyListFilterSlotsAvailable(slots_available);
	}
}

//! Sets the distance for which we should search for lobbies (based on users IP address to location map on the Steam backed).
void Steam::addRequestLobbyListDistanceFilter(int distance_filter){
	if(SteamMatchmaking() != NULL){
		SteamMatchmaking()->AddRequestLobbyListDistanceFilter((ELobbyDistanceFilter)distance_filter);
	}
}

//! Sets how many results to return, the lower the count the faster it is to download the lobby results & details to the client.
void Steam::addRequestLobbyListResultCountFilter(int max_results){
	if(SteamMatchmaking() != NULL){
		SteamMatchmaking()->AddRequestLobbyListResultCountFilter(max_results);
	}
}

//! Create a lobby on the Steam servers, if private the lobby will not be returned by any RequestLobbyList() call.
void Steam::createLobby(int lobby_type, int max_members){
	if(SteamMatchmaking() != NULL){
		SteamAPICall_t api_call = SteamMatchmaking()->CreateLobby((ELobbyType)lobby_type, max_members);
		callResultCreateLobby.Set(api_call, this, &Steam::lobby_created);
	}
}

//! Join an existing lobby.
void Steam::joinLobby(uint64_t steam_lobby_id){
	if(SteamMatchmaking() != NULL){
		CSteamID lobby_id = (uint64)steam_lobby_id;
		SteamMatchmaking()->JoinLobby(lobby_id);
	}
}

//! Leave a lobby, this will take effect immediately on the client side, other users will be notified by LobbyChatUpdate_t callback.
void Steam::leaveLobby(uint64_t steam_lobby_id){
	if(SteamMatchmaking() != NULL){
		CSteamID lobby_id = (uint64)steam_lobby_id;
		SteamMatchmaking()->LeaveLobby(lobby_id);
	}
}

//! Invite another user to the lobby, the target user will receive a LobbyInvite_t callback, will return true if the invite is successfully sent, whether or not the target responds.
bool Steam::inviteUserToLobby(uint64_t steam_lobby_id, uint64_t steam_id_invitee){
	if(SteamMatchmaking() == NULL){
		return false;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	CSteamID invitee_id = (uint64)steam_id_invitee;
	return SteamMatchmaking()->InviteUserToLobby(lobby_id, invitee_id);
}

//! Lobby iteration, for viewing details of users in a lobby.
int Steam::getNumLobbyMembers(uint64_t steam_lobby_id){
	if(SteamMatchmaking() == NULL){
		return 0;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	return SteamMatchmaking()->GetNumLobbyMembers(lobby_id);
}

//! Returns the CSteamID of a user in the lobby.
uint64_t Steam::getLobbyMemberByIndex(uint64_t steam_lobby_id, int member){
	if(SteamMatchmaking() == NULL){
		return 0;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	CSteamID lobbyMember = SteamMatchmaking()->GetLobbyMemberByIndex(lobby_id, member);
	return lobbyMember.ConvertToUint64();
}

//! Get data associated with this lobby.
String Steam::getLobbyData(uint64_t steam_lobby_id, const String& key){
	if(SteamMatchmaking() == NULL){
		return "";
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	return SteamMatchmaking()->GetLobbyData(lobby_id, key.utf8().get_data());
}

//! Sets a key/value pair in the lobby metadata.
bool Steam::setLobbyData(uint64_t steam_lobby_id, const String& key, const String& value){
	if(SteamMatchmaking() == NULL){
		return false;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	return SteamMatchmaking()->SetLobbyData(lobby_id, key.utf8().get_data(), value.utf8().get_data());
}

//! Get lobby data by the lobby's ID
Dictionary Steam::getAllLobbyData(uint64_t steam_lobby_id){
	Dictionary data;
	if(SteamMatchmaking() == NULL){
		return data;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	int data_count = SteamMatchmaking()->GetLobbyDataCount(lobby_id);
	char key[MAX_LOBBY_KEY_LENGTH];
	char value[CHAT_METADATA_MAX];
	for(int i = 0; i < data_count; i++){
		bool success = SteamMatchmaking()->GetLobbyDataByIndex(lobby_id, i, key, MAX_LOBBY_KEY_LENGTH, value, CHAT_METADATA_MAX);
		if(success){
			data["index"] = i;
			data["key"] = key;
			data["value"] = value;
		}
	}
	return data;
}

//! Removes a metadata key from the lobby.
bool Steam::deleteLobbyData(uint64_t steam_lobby_id, const String& key){
	if(SteamMatchmaking() == NULL){
		return false;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	return SteamMatchmaking()->DeleteLobbyData(lobby_id, key.utf8().get_data());
}

//! Gets per-user metadata for someone in this lobby.
String Steam::getLobbyMemberData(uint64_t steam_lobby_id, uint64_t steam_id_user, const String& key){
	if(SteamMatchmaking() == NULL){
		return "";
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	CSteamID user_id = (uint64)steam_id_user;
	return SteamMatchmaking()->GetLobbyMemberData(lobby_id, user_id, key.utf8().get_data());
}

//! Sets per-user metadata (for the local user implicitly).
void Steam::setLobbyMemberData(uint64_t steam_lobby_id, const String& key, const String& value){
	if(SteamMatchmaking() != NULL){
		CSteamID lobby_id = (uint64)steam_lobby_id;
		SteamMatchmaking()->SetLobbyMemberData(lobby_id, key.utf8().get_data(), value.utf8().get_data());
	}
}

//! Broadcasts a chat message to the all the users in the lobby.
bool Steam::sendLobbyChatMsg(uint64_t steam_lobby_id, const String& message_body){
	if(SteamMatchmaking() == NULL){
		return false;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	return SteamMatchmaking()->SendLobbyChatMsg(lobby_id, message_body.utf8().get_data(), message_body.size() + 1);
}

//! Refreshes metadata for a lobby you're not necessarily in right now.
bool Steam::requestLobbyData(uint64_t steam_lobby_id){
	if(SteamMatchmaking() == NULL){
		return false;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	return SteamMatchmaking()->RequestLobbyData(lobby_id);
}

//! Sets the game server associated with the lobby.
void Steam::setLobbyGameServer(uint64_t steam_lobby_id, const String& server_ip, uint16 server_port, uint64_t steam_id_game_server){
	if(SteamMatchmaking() != NULL){
		// Resolve address and convert it from IP_Address string to uint32_t
		IP_Address address;
		if(server_ip.is_valid_ip_address()){
			address = server_ip;
		}
		else{
			address = IP::get_singleton()->resolve_hostname(server_ip, IP::TYPE_IPV4);
		}
		// Resolution succeeded
		if(address.is_valid()){
			uint32_t ip4 = *((uint32_t *)address.get_ipv4());
			CSteamID lobby_id = (uint64)steam_lobby_id;
			// If setting a game server with no server (fake) Steam ID
			if(steam_id_game_server == 0){
				SteamMatchmaking()->SetLobbyGameServer(lobby_id, ip4, server_port, k_steamIDNil);
			}
			else{
				CSteamID game_id = (uint64)steam_id_game_server;
				SteamMatchmaking()->SetLobbyGameServer(lobby_id, ip4, server_port, game_id);
			}
		}
	}
}

//! Returns the details of a game server set in a lobby - returns false if there is no game server set, or that lobby doesn't exist.
Dictionary Steam::getLobbyGameServer(uint64_t steam_lobby_id){
	Dictionary game_server;
	if(SteamMatchmaking() != NULL){
		CSteamID lobby_id = (uint64)steam_lobby_id;
		uint32 server_ip = 0;
		uint16 server_port = 0;
		CSteamID server_id;
		game_server["ret"] = SteamMatchmaking()->GetLobbyGameServer(lobby_id, &server_ip, &server_port, &server_id);
		if(game_server["ret"]){
			// Convert the IP address back to a string
			const int NBYTES = 4;
			uint8 octet[NBYTES];
			char ip[16];
			for(int i = 0; i < NBYTES; i++){
				octet[i] = server_ip >> (i * 8);
			}
			sprintf(ip, "%d.%d.%d.%d", octet[0], octet[1], octet[2], octet[3]);
			game_server["ip"] = ip;
			game_server["port"] = server_port;
			// Convert the server ID
			int server = server_id.ConvertToUint64();
			game_server["id"] = server;
		}
	}
	return game_server;
}

//! Set the limit on the # of users who can join the lobby.
bool Steam::setLobbyMemberLimit(uint64_t steam_lobby_id, int max_members){
	if(SteamMatchmaking() == NULL){
		return false;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	return SteamMatchmaking()->SetLobbyMemberLimit(lobby_id, max_members);
}

//! Returns the current limit on the # of users who can join the lobby; returns 0 if no limit is defined.
int Steam::getLobbyMemberLimit(uint64_t steam_lobby_id){
	if(SteamMatchmaking() == NULL){
		return 0;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	return SteamMatchmaking()->GetLobbyMemberLimit(lobby_id);
}

//! Updates which type of lobby it is.
bool Steam::setLobbyType(uint64_t steam_lobby_id, int lobby_type){
	if(SteamMatchmaking() == NULL){
		return false;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	return SteamMatchmaking()->SetLobbyType(lobby_id, (ELobbyType)lobby_type);
}

//! Sets whether or not a lobby is joinable - defaults to true for a new lobby.
bool Steam::setLobbyJoinable(uint64_t steam_lobby_id, bool joinable){
	if(SteamMatchmaking() == NULL){
		return false;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	return SteamMatchmaking()->SetLobbyJoinable(lobby_id, joinable);
}

//! Returns the current lobby owner.
uint64_t Steam::getLobbyOwner(uint64_t steam_lobby_id){
	if(SteamMatchmaking() == NULL){
		return 0;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	CSteamID owner_id = SteamMatchmaking()->GetLobbyOwner(lobby_id);
	return owner_id.ConvertToUint64();
}

//! Changes who the lobby owner is.
bool Steam::setLobbyOwner(uint64_t steam_lobby_id, uint64_t steam_id_new_owner){
	if(SteamMatchmaking() == NULL){
		return false;
	}
	CSteamID lobby_id = (uint64)steam_lobby_id;
	CSteamID owner_id = (uint64)steam_id_new_owner;
	return SteamMatchmaking()->SetLobbyOwner(lobby_id, owner_id);
}


/////////////////////////////////////////////////
///// MATCHMAKING SERVERS
/////////////////////////////////////////////////
//
//! Cancel an outstanding server list request.
void Steam::cancelQuery(uint64_t server_list_request){
	if(SteamMatchmakingServers() != NULL){
		SteamMatchmakingServers()->CancelQuery((HServerListRequest)server_list_request);
	}
}

//! Cancel an outstanding individual server query.
void Steam::cancelServerQuery(int server_query){
	if(SteamMatchmakingServers() != NULL){
		SteamMatchmakingServers()->CancelServerQuery((HServerQuery)server_query);
	}
}

//! Gets the number of servers in the given list.
int Steam::getServerCount(uint64_t server_list_request){
	if(SteamMatchmakingServers() == NULL){
		return 0;
	}
	return SteamMatchmakingServers()->GetServerCount((HServerListRequest)server_list_request);
}

//! Get the details of a given server in the list.
Dictionary Steam::getServerDetails(uint64_t server_list_request, int server){
	// Create a dictionary to populate
	Dictionary game_server;
	if(SteamMatchmakingServers() != NULL){
		gameserveritem_t* server_item = new gameserveritem_t;
		SteamMatchmakingServers()->GetServerDetails((HServerListRequest)server_list_request, server);
		// Populate the dictionary
		game_server["ping"] = server_item->m_nPing;
		game_server["success_response"] = server_item->m_bHadSuccessfulResponse;
		game_server["no_refresh"] = server_item->m_bDoNotRefresh;
		game_server["game_dir"] = server_item->m_szGameDir;
		game_server["map"] = server_item->m_szMap;
		game_server["description"] = server_item->m_szGameDescription;
		game_server["app_id"] = server_item->m_nAppID;
		game_server["players"] = server_item->m_nPlayers;
		game_server["max_players"] = server_item->m_nMaxPlayers;
		game_server["bot_players"] = server_item->m_nBotPlayers;
		game_server["password"] = server_item->m_bPassword;
		game_server["secure"] = server_item->m_bSecure;
		game_server["last_played"] = server_item->m_ulTimeLastPlayed;
		game_server["server_version"] = server_item->m_nServerVersion;
		// Clean up
		delete server_item;
	}
	// Return the dictionary
	return game_server;
}

//! Returns true if the list is currently refreshing its server list.
bool Steam::isRefreshing(uint64_t server_list_request){
	if(SteamMatchmakingServers() == NULL){
		return false;
	}
	return SteamMatchmakingServers()->IsRefreshing((HServerListRequest)server_list_request);
}

//! Queries an individual game servers directly via IP/Port to request an updated ping time and other details from the server.
int Steam::pingServer(const String& ip, uint16 port){
	if(SteamMatchmakingServers() == NULL){
		return 0;
	}
	// Resolve address and convert it from IP_Address string to uint32_t
	IP_Address address;
	if(ip.is_valid_ip_address()){
		address = ip;
	}
	else{
		address = IP::get_singleton()->resolve_hostname(ip, IP::TYPE_IPV4);
	}
	// Resolution failed
	if(!address.is_valid()){
		return 0;
	}
	uint32_t ip4 = *((uint32_t *)address.get_ipv4());
	return SteamMatchmakingServers()->PingServer(ip4, port, ping_response);
}

//! Request the list of players currently playing on a server.
int Steam::playerDetails(const String& ip, uint16 port){
	if(SteamMatchmakingServers() == NULL){
		return 0;
	}
	// Resolve address and convert it from IP_Address string to uint32_t
	IP_Address address;
	if(ip.is_valid_ip_address()){
		address = ip;
	}
	else{
		address = IP::get_singleton()->resolve_hostname(ip, IP::TYPE_IPV4);
	}
	// Resolution failed
	if(!address.is_valid()){
		return 0;
	}
	uint32_t ip4 = *((uint32_t *)address.get_ipv4());
	return SteamMatchmakingServers()->PlayerDetails(ip4, port, player_response);
}

//! Ping every server in your list again but don't update the list of servers. Query callback installed when the server list was requested will be used again to post notifications and RefreshComplete, so the callback must remain valid until another RefreshComplete is called on it or the request is released with ReleaseRequest( hRequest ).
void Steam::refreshQuery(uint64_t server_list_request){
	if(SteamMatchmakingServers() != NULL){
		SteamMatchmakingServers()->RefreshQuery((HServerListRequest)server_list_request);
	}
}

//! Refresh a single server inside of a query (rather than all the servers).
void Steam::refreshServer(uint64_t server_list_request, int server){
	if(SteamMatchmakingServers() != NULL){
		SteamMatchmakingServers()->RefreshServer((HServerListRequest)server_list_request, server);
	}
}

//! Releases the asynchronous request object and cancels any pending query on it if there's a pending query in progress.
void Steam::releaseRequest(uint64_t server_list_request){
	if(SteamMatchmakingServers() != NULL){
		SteamMatchmakingServers()->ReleaseRequest((HServerListRequest)server_list_request);
	}
}

//! Request a new list of servers of a particular type.  These calls each correspond to one of the EMatchMakingType values.
void Steam::requestFavoritesServerList(uint32 app_id, Array filters){
	if(SteamMatchmakingServers() != NULL){
		uint32 filter_size = filters.size();
		MatchMakingKeyValuePair_t* filters_array = new MatchMakingKeyValuePair_t[filter_size];
		for (uint32 i=0; i < filter_size; i++){
			// Create a new filter pair to populate
			MatchMakingKeyValuePair_t *filter_array = new MatchMakingKeyValuePair_t();
			// Get the key/value pair
			Array pair = filters[i];
			// Get the key from the filter pair
			String key = (String)pair[0];
			char *this_key = new char[256];
			strcpy(this_key, key.utf8().get_data());
			filter_array->m_szKey[i] = *this_key;
			delete[] this_key;
			// Get the value from the filter pair
			String value = pair[1];
			char *this_value = new char[256];
			strcpy(this_value, value.utf8().get_data());
			filter_array->m_szValue[i] = *this_value;
			delete[] this_value;
			// Append this to the array
			filters_array[i] = *filter_array;
			// Free up the memory
			delete filter_array;
		}
		server_list_request = SteamMatchmakingServers()->RequestFavoritesServerList((AppId_t)app_id, &filters_array, filter_size, server_list_response);
		delete[] filters_array;
	}
}

//! Request a new list of servers of a particular type.  These calls each correspond to one of the EMatchMakingType values.
void Steam::requestFriendsServerList(uint32 app_id, Array filters){
	if(SteamMatchmakingServers() != NULL){
		uint32 filter_size = filters.size();
		MatchMakingKeyValuePair_t* filters_array = new MatchMakingKeyValuePair_t[filter_size];
		for (uint32 i=0; i < filter_size; i++){
			// Create a new filter pair to populate
			MatchMakingKeyValuePair_t *filter_array = new MatchMakingKeyValuePair_t();
			// Get the key/value pair
			Array pair = filters[i];
			// Get the key from the filter pair
			String key = (String)pair[0];
			char *this_key = new char[256];
			strcpy(this_key, key.utf8().get_data());
			filter_array->m_szKey[i] = *this_key;
			delete[] this_key;
			// Get the value from the filter pair
			String value = pair[1];
			char *this_value = new char[256];
			strcpy(this_value, value.utf8().get_data());
			filter_array->m_szValue[i] = *this_value;
			delete[] this_value;
			// Append this to the array
			filters_array[i] = *filter_array;
			// Free up the memory
			delete filter_array;
		}
		server_list_request = SteamMatchmakingServers()->RequestFriendsServerList((AppId_t)app_id, &filters_array, filter_size, server_list_response);
		delete[] filters_array;
	}
}

//! Request a new list of servers of a particular type.  These calls each correspond to one of the EMatchMakingType values.
void Steam::requestHistoryServerList(uint32 app_id, Array filters){
	if(SteamMatchmakingServers() != NULL){
		uint32 filter_size = filters.size();
		MatchMakingKeyValuePair_t* filters_array = new MatchMakingKeyValuePair_t[filter_size];
		for (uint32 i=0; i < filter_size; i++){
			// Create a new filter pair to populate
			MatchMakingKeyValuePair_t *filter_array = new MatchMakingKeyValuePair_t();
			// Get the key/value pair
			Array pair = filters[i];
			// Get the key from the filter pair
			String key = (String)pair[0];
			char *this_key = new char[256];
			strcpy(this_key, key.utf8().get_data());
			filter_array->m_szKey[i] = *this_key;
			delete[] this_key;
			// Get the value from the filter pair
			String value = pair[1];
			char *this_value = new char[256];
			strcpy(this_value, value.utf8().get_data());
			filter_array->m_szValue[i] = *this_value;
			delete[] this_value;
			// Append this to the array
			filters_array[i] = *filter_array;
			// Free up the memory
			delete filter_array;
		}
		server_list_request = SteamMatchmakingServers()->RequestHistoryServerList((AppId_t)app_id, &filters_array, filter_size, server_list_response);
		delete[] filters_array;
	}
}

//! Request a new list of servers of a particular type.  These calls each correspond to one of the EMatchMakingType values.
void Steam::requestInternetServerList(uint32 app_id, Array filters){
	if(SteamMatchmakingServers() != NULL){
		uint32 filter_size = filters.size();
		MatchMakingKeyValuePair_t* filters_array = new MatchMakingKeyValuePair_t[filter_size];
		for (uint32 i=0; i < filter_size; i++){
			// Create a new filter pair to populate
			MatchMakingKeyValuePair_t *filter_array = new MatchMakingKeyValuePair_t();
			// Get the key/value pair
			Array pair = filters[i];
			// Get the key from the filter pair
			String key = (String)pair[0];
			char *this_key = new char[256];
			strcpy(this_key, key.utf8().get_data());
			filter_array->m_szKey[i] = *this_key;
			delete[] this_key;
			// Get the value from the filter pair
			String value = pair[1];
			char *this_value = new char[256];
			strcpy(this_value, value.utf8().get_data());
			filter_array->m_szValue[i] = *this_value;
			delete[] this_value;
			// Append this to the array
			filters_array[i] = *filter_array;
			// Free up the memory
			delete filter_array;
		}
		server_list_request = SteamMatchmakingServers()->RequestInternetServerList((AppId_t)app_id, &filters_array, filter_size, server_list_response);
		delete[] filters_array;
	}
}

//! Request a new list of servers of a particular type.  These calls each correspond to one of the EMatchMakingType values.
void Steam::requestLANServerList(uint32 app_id){
	if(SteamMatchmakingServers() != NULL){
		server_list_request = SteamMatchmakingServers()->RequestLANServerList((AppId_t)app_id, server_list_response);
	}
}

//! Request a new list of servers of a particular type.  These calls each correspond to one of the EMatchMakingType values.
void Steam::requestSpectatorServerList(uint32 app_id, Array filters){
	if(SteamMatchmakingServers() != NULL){
		uint32 filter_size = filters.size();
		MatchMakingKeyValuePair_t* filters_array = new MatchMakingKeyValuePair_t[filter_size];
		for (uint32 i=0; i < filter_size; i++){
			// Create a new filter pair to populate
			MatchMakingKeyValuePair_t *filter_array = new MatchMakingKeyValuePair_t();
			// Get the key/value pair
			Array pair = filters[i];
			// Get the key from the filter pair
			String key = (String)pair[0];
			char *this_key = new char[256];
			strcpy(this_key, key.utf8().get_data());
			filter_array->m_szKey[i] = *this_key;
			delete[] this_key;
			// Get the value from the filter pair
			String value = pair[1];
			char *this_value = new char[256];
			strcpy(this_value, value.utf8().get_data());
			filter_array->m_szValue[i] = *this_value;
			delete[] this_value;
			// Append this to the array
			filters_array[i] = *filter_array;
			// Free up the memory
			delete filter_array;
		}
		server_list_request = SteamMatchmakingServers()->RequestSpectatorServerList((AppId_t)app_id, &filters_array, filter_size, server_list_response);
		delete[] filters_array;
	}
}

//! Request the list of rules that the server is running (See ISteamGameServer::SetKeyValue() to set the rules server side)
int Steam::serverRules(const String& ip, uint16 port){
	if(SteamMatchmakingServers() == NULL){
		return 0;
	}
	// Resolve address and convert it from IP_Address string to uint32_t
	IP_Address address;
	if(ip.is_valid_ip_address()){
		address = ip;
	}
	else{
		address = IP::get_singleton()->resolve_hostname(ip, IP::TYPE_IPV4);
	}
	// Resolution failed
	if(!address.is_valid()){
		return 0;
	}
	uint32_t ip4 = *((uint32_t *)address.get_ipv4());
	return SteamMatchmakingServers()->ServerRules(ip4, port, rules_response);
}


/////////////////////////////////////////////////
///// MUSIC
/////////////////////////////////////////////////
//
//! Is Steam music enabled.
bool Steam::musicIsEnabled(){
	if(SteamMusic() == NULL){
		return false;
	}
	return SteamMusic()->BIsEnabled();
}

//! Is Steam music playing something.
bool Steam::musicIsPlaying(){
	if(SteamMusic() == NULL){
		return false;
	}
	return SteamMusic()->BIsPlaying();
}

// Gets the current status of the Steam Music player
int Steam::getPlaybackStatus(){
	if(SteamMusic() == NULL){
		return 0;
	}
	return SteamMusic()->GetPlaybackStatus();
}

//! Get the volume level of the music.
float Steam::musicGetVolume(){
	if(SteamMusic() == NULL){
		return 0;
	}
	return SteamMusic()->GetVolume();
}

//! Pause whatever Steam music is playing.
void Steam::musicPause(){
	if(SteamMusic() != NULL){
		SteamMusic()->Pause();
	}
}

//! Play current track/album.
void Steam::musicPlay(){
	if(SteamMusic() != NULL){
		SteamMusic()->Play();
	}
}

//! Play next track/album.
void Steam::musicPlayNext(){
	if(SteamMusic() != NULL){
		SteamMusic()->PlayNext();
	}
}

//! Play previous track/album.
void Steam::musicPlayPrev(){
	if(SteamMusic() != NULL){
		SteamMusic()->PlayPrevious();
	}
}

//! Set the volume of Steam music.
void Steam::musicSetVolume(float volume){
	if(SteamMusic() != NULL){
		SteamMusic()->SetVolume(volume);
	}
}


/////////////////////////////////////////////////
///// MUSIC REMOTE
/////////////////////////////////////////////////
//
// These functions do not have any offical notes or comments.  All descriptions are assumed.
//
//! If remote access was successfully activated.
bool Steam::activationSuccess(bool activate){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->BActivationSuccess(activate);
}

//! Is a remote music client / host connected?
bool Steam::isCurrentMusicRemote(){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->BIsCurrentMusicRemote();
}

//! Did the currenty music entry just change?
bool Steam::currentEntryDidChange(){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->CurrentEntryDidChange();
}

//! Is the current music entry available?
bool Steam::currentEntryIsAvailable(bool available){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->CurrentEntryIsAvailable(available);
}

//! Will the current music entry change?
bool Steam::currentEntryWillChange(){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->CurrentEntryWillChange();
}

//! Disconnect from remote music client / host.
bool Steam::deregisterSteamMusicRemote(){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->DeregisterSteamMusicRemote();
}

//! Enable track loop on client.
bool Steam::enableLooped(bool loop){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->EnableLooped(loop);
}

//! Enable playlists on client.
bool Steam::enablePlaylists(bool playlists){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->EnablePlaylists(playlists);
}

//! Play the next track on client.
bool Steam::enablePlayNext(bool next){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->EnablePlayNext(next);
}

//! Play previous track on client.
bool Steam::enablePlayPrevious(bool previous){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->EnablePlayPrevious(previous);
}

//! Enable the music queue on the client.
bool Steam::enableQueue(bool queue){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->EnableQueue(queue);
}

//! Enable shuffle on the client.
bool Steam::enableShuffled(bool shuffle){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->EnableShuffled(shuffle);
}

//! Has the playlist changed?
bool Steam::playlistDidChange(){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->PlaylistDidChange();
}

//! Will the playlist change?
bool Steam::playlistWillChange(){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->PlaylistWillChange();
}

//! Did the song queue change?
bool Steam::queueDidChange(){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->QueueDidChange();
}

//! Will the song queue change?
bool Steam::queueWillChange(){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->QueueWillChange();
}

//! Connect to a music remote client / host?
bool Steam::registerSteamMusicRemote(const String& name){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->RegisterSteamMusicRemote(name.utf8().get_data());
}

//! Reset the playlist entries.
bool Steam::resetPlaylistEntries(){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->ResetPlaylistEntries();
}

//! Reset the song queue entries.
bool Steam::resetQueueEntries(){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->ResetQueueEntries();
}

//! Set a new current playlist.
bool Steam::setCurrentPlaylistEntry(int id){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->SetCurrentPlaylistEntry(id);
}

//! Set a new current song queue.
bool Steam::setCurrentQueueEntry(int id){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->SetCurrentQueueEntry(id);
}

//! Set a new display name.
bool Steam::setDisplayName(const String& name){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->SetDisplayName(name.utf8().get_data());
}

//! Set a new playlist entry.
bool Steam::setPlaylistEntry(int id, int position, const String& entry_text){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->SetPlaylistEntry(id, position, entry_text.utf8().get_data());
}

// Set a PNG icon for a song? A playlist?
bool Steam::setPNGIcon64x64(PoolByteArray icon){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	uint32 icon_size = icon.size();
	return SteamMusicRemote()->SetPNGIcon_64x64((void*)icon.read().ptr(), icon_size);
}

//! Set a new queue entry.
bool Steam::setQueueEntry(int id, int position, const String& entry_text){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->SetQueueEntry(id, position, entry_text.utf8().get_data());
}

// Update the current song entry's cover art.
bool Steam::updateCurrentEntryCoverArt(PoolByteArray art){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	uint32 art_size = art.size();
	return SteamMusicRemote()->UpdateCurrentEntryCoverArt((void*)art.read().ptr(), art_size);
}

//! Update the current seconds that have elapsed for an entry.
bool Steam::updateCurrentEntryElapsedSeconds(int seconds){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->UpdateCurrentEntryElapsedSeconds(seconds);
}

//! Update the current song entry's text?
bool Steam::updateCurrentEntryText(const String& text){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->UpdateCurrentEntryText(text.utf8().get_data());
}

//! Update looped or not.
bool Steam::updateLooped(bool looped){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->UpdateLooped(looped);
}

//! Update the current playback status; 0 - undefined, 1 - playing, 2 - paused, 3 - idle.
bool Steam::updatePlaybackStatus(int status){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->UpdatePlaybackStatus((AudioPlayback_Status)status);
}

//! Update whether to shuffle or not.
bool Steam::updateShuffled(bool shuffle){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->UpdateShuffled(shuffle);
}

//! Volume is between 0.0 and 1.0.
bool Steam::updateVolume(float volume){
	if(SteamMusicRemote() == NULL){
		return false;
	}
	return SteamMusicRemote()->UpdateVolume(volume);
}


/////////////////////////////////////////////////
///// NETWORKING
/////////////////////////////////////////////////
//
//! This allows the game to specify accept an incoming packet.
bool Steam::acceptP2PSessionWithUser(uint64_t steam_id_remote) {
	if (SteamNetworking() == NULL) {
		return false;
	}
	CSteamID steam_id = createSteamID(steam_id_remote);
	return SteamNetworking()->AcceptP2PSessionWithUser(steam_id);
}

//! Allow or disallow P2P connections to fall back to being relayed through the Steam servers if a direct connection or NAT-traversal cannot be established.
bool Steam::allowP2PPacketRelay(bool allow) {
	if (SteamNetworking() == NULL) {
		return false;
	}
	return SteamNetworking()->AllowP2PPacketRelay(allow);
}

//! Closes a P2P channel when you're done talking to a user on the specific channel.
bool Steam::closeP2PChannelWithUser(uint64_t steam_id_remote, int channel) {
	if (SteamNetworking() == NULL) {
		return false;
	}
	CSteamID steam_id = createSteamID(steam_id_remote);
	return SteamNetworking()->CloseP2PChannelWithUser(steam_id, channel);
}

//! This should be called when you're done communicating with a user, as this will free up all of the resources allocated for the connection under-the-hood.
bool Steam::closeP2PSessionWithUser(uint64_t steam_id_remote) {
	if (SteamNetworking() == NULL) {
		return false;
	}
	CSteamID steam_id = createSteamID(steam_id_remote);
	return SteamNetworking()->CloseP2PSessionWithUser(steam_id);
}

//! Fills out a P2PSessionState_t structure with details about the connection like whether or not there is an active connection.
Dictionary Steam::getP2PSessionState(uint64_t steam_id_remote) {
	Dictionary result;
	if (SteamNetworking() == NULL) {
		return result;
	}
	CSteamID steam_id = createSteamID(steam_id_remote);
	P2PSessionState_t p2pSessionState;
	bool success = SteamNetworking()->GetP2PSessionState(steam_id, &p2pSessionState);
	if (!success) {
		return result;
	}
	result["connection_active"] = p2pSessionState.m_bConnectionActive; // true if we've got an active open connection
	result["connecting"] = p2pSessionState.m_bConnecting; // true if we're currently trying to establish a connection
	result["session_error"] = p2pSessionState.m_eP2PSessionError; // last error recorded (see enum in isteamnetworking.h)
	result["using_relay"] = p2pSessionState.m_bUsingRelay; // true if it's going through a relay server (TURN)
	result["bytes_queued_for_send"] = p2pSessionState.m_nBytesQueuedForSend;
	result["packets_queued_for_send"] = p2pSessionState.m_nPacketsQueuedForSend;
	result["remote_ip"] = p2pSessionState.m_nRemoteIP; // potential IP:Port of remote host. Could be TURN server.
	result["remote_port"] = p2pSessionState.m_nRemotePort; // Only exists for compatibility with older authentication api's
	return result;
}

//! Calls IsP2PPacketAvailable() under the hood, returns the size of the available packet or zero if there is no such packet.
uint32_t Steam::getAvailableP2PPacketSize(int channel){
	if (SteamNetworking() == NULL) {
		return 0;
	}
	uint32_t messageSize = 0;
	return (SteamNetworking()->IsP2PPacketAvailable(&messageSize, channel)) ? messageSize : 0;
}

//! Reads in a packet that has been sent from another user via SendP2PPacket.
Dictionary Steam::readP2PPacket(uint32_t packet, int channel){
	Dictionary result;
	if (SteamNetworking() == NULL) {
		return result;
	}
	PoolByteArray data;
	data.resize(packet);
	CSteamID steam_id;
	uint32_t bytesRead = 0;
	if (SteamNetworking()->ReadP2PPacket(data.write().ptr(), packet, &bytesRead, &steam_id, channel)){
		data.resize(bytesRead);
		uint64_t steam_id_remote = steam_id.ConvertToUint64();
		result["data"] = data;
		result["steam_id_remote"] = steam_id_remote;
	}
	else {
		data.resize(0);
	}
	return result;
}

//! Sends a P2P packet to the specified user.
bool Steam::sendP2PPacket(uint64_t steam_id_remote, PoolByteArray data, int send_type, int channel){
	if (SteamNetworking() == NULL) {
		return false;
	}
	CSteamID steam_id = createSteamID(steam_id_remote);
	return SteamNetworking()->SendP2PPacket(steam_id, data.read().ptr(), data.size(), EP2PSend(send_type), channel);
}


/////////////////////////////////////////////////
///// NETWORKING MESSAGES
/////////////////////////////////////////////////
//
//! AcceptSessionWithUser() should only be called in response to a SteamP2PSessionRequest_t callback SteamP2PSessionRequest_t will be posted if another user tries to send you a message, and you haven't tried to talk to them.
bool Steam::acceptSessionWithUser(const String& identity_reference){
	if(SteamNetworkingMessages() == NULL){
		return false;
	}
	return SteamNetworkingMessages()->AcceptSessionWithUser(networking_identities[identity_reference.utf8().get_data()]);
}

//! Call this  when you're done talking to a user on a specific channel. Once all open channels to a user have been closed, the open session to the user will be closed, and any new data from this user will trigger a SteamP2PSessionRequest_t callback.
bool Steam::closeChannelWithUser(const String& identity_reference, int channel){
	if(SteamNetworkingMessages() == NULL){
		return false;
	}
	return SteamNetworkingMessages()->CloseChannelWithUser(networking_identities[identity_reference.utf8().get_data()], channel);
}

//! Call this when you're done talking to a user to immediately free up resources under-the-hood.
bool Steam::closeSessionWithUser(const String& identity_reference){
	if(SteamNetworkingMessages() == NULL){
		return false;
	}
	return SteamNetworkingMessages()->CloseSessionWithUser(networking_identities[identity_reference.utf8().get_data()]);
}

//! Returns information about the latest state of a connection, if any, with the given peer.
Dictionary Steam::getSessionConnectionInfo(const String& identity_reference, bool get_connection, bool get_status){
	Dictionary connection_info;
	if(SteamNetworkingMessages() != NULL){
		SteamNetConnectionInfo_t this_info;
		SteamNetConnectionRealTimeStatus_t this_status;
		int connection_state = SteamNetworkingMessages()->GetSessionConnectionInfo(networkingIdentity, &this_info, &this_status);
		// Parse the data to a dictionary
		connection_info["connection_state"] = connection_state;
		// If getting the connection information
		if(get_connection){
			char identity[STEAM_BUFFER_SIZE];
			this_info.m_identityRemote.ToString(identity, STEAM_BUFFER_SIZE);
			connection_info["identity"] = identity;
			connection_info["user_data"] = (uint64_t)this_info.m_nUserData;
			connection_info["listen_socket"] = this_info.m_hListenSocket;
			char ip_address[STEAM_BUFFER_SIZE];
			this_info.m_addrRemote.ToString(ip_address, STEAM_BUFFER_SIZE, true);
			connection_info["remote_address"] = ip_address;
			connection_info["remote_pop"] = this_info.m_idPOPRemote;
			connection_info["pop_relay"] = this_info.m_idPOPRelay;
			connection_info["connection_state"] = this_info.m_eState;
			connection_info["end_reason"] = this_info.m_eEndReason;
			connection_info["end_debug"] = this_info.m_szEndDebug;
			connection_info["debug_description"] = this_info.m_szConnectionDescription;
		}
		// If getting the quick status
		if(get_status){
			connection_info["state"] = this_status.m_eState;
			connection_info["ping"] = this_status.m_nPing;
			connection_info["local_quality"] = this_status.m_flConnectionQualityLocal;
			connection_info["remote_quality"] = this_status.m_flConnectionQualityRemote;
			connection_info["packets_out_per_second"] = this_status.m_flOutPacketsPerSec;
			connection_info["bytes_out_per_second"] = this_status.m_flOutBytesPerSec;
			connection_info["packets_in_per_second"] = this_status.m_flInPacketsPerSec;
			connection_info["bytes_in_per_second"] = this_status.m_flInBytesPerSec;
			connection_info["send_rate"] = this_status.m_nSendRateBytesPerSecond;
			connection_info["pending_unreliable"] = this_status.m_cbPendingUnreliable;
			connection_info["pending_reliable"] = this_status.m_cbPendingReliable;
			connection_info["sent_unacknowledged_reliable"] = this_status.m_cbSentUnackedReliable;
			connection_info["queue_time"] = (uint64_t)this_status.m_usecQueueTime;
		}
	}
	return connection_info;
}

//! Reads the next message that has been sent from another user via SendMessageToUser() on the given channel. Returns number of messages returned into your list.  (0 if no message are available on that channel.)
Array Steam::receiveMessagesOnChannel(int channel, int max_messages){
	Array messages;
	if(SteamNetworkingMessages() != NULL){
		// Allocate the space for the messages
		SteamNetworkingMessage_t** channel_messages = new SteamNetworkingMessage_t*[max_messages];
		// Get the messages
		int available_messages = SteamNetworkingMessages()->ReceiveMessagesOnChannel(channel, channel_messages, max_messages);
		// Loop through and create the messages as dictionaries then add to the messages array
		for(int i = 0; i < available_messages; i++){
			// Set up the mesage dictionary
			Dictionary message;
			// Get the data / message
			int message_size = channel_messages[i]->m_cbSize;
			PoolByteArray data;
			data.resize(message_size);
			uint8_t* source_data = (uint8_t*)channel_messages[i]->m_pData;
			uint8_t* output_data = data.write().ptr();
			for(int j = 0; j < message_size; j++){
				output_data[j] = source_data[j];
			}
			message["payload"] = data;
			message["size"] = message_size;
			message["connection"] = channel_messages[i]->m_conn;
			char identity[STEAM_BUFFER_SIZE];
			channel_messages[i]->m_identityPeer.ToString(identity, STEAM_BUFFER_SIZE);
			message["identity"] = identity;
			message["user_data"] = (uint64_t)channel_messages[i]->m_nConnUserData;
			message["time_received"] = (uint64_t)channel_messages[i]->m_usecTimeReceived;
			message["message_number"] = (uint64_t)channel_messages[i]->m_nMessageNumber;
			message["channel"] = channel_messages[i]->m_nChannel;
			message["flags"] = channel_messages[i]->m_nFlags;
			message["user_data"] = (uint64_t)channel_messages[i]->m_nUserData;
			messages.append(message);
			// Release the message
			channel_messages[i]->Release();
		}
		delete [] channel_messages;
	}
	return messages;
}

//! Sends a message to the specified host. If we don't already have a session with that user, a session is implicitly created. There might be some handshaking that needs to happen before we can actually begin sending message data.
int Steam::sendMessageToUser(const String& identity_reference, const PoolByteArray data, int flags, int channel){
	if(SteamNetworkingMessages() == NULL){
		return 0;
	}
	return SteamNetworkingMessages()->SendMessageToUser(networking_identities[identity_reference.utf8().get_data()], data.read().ptr(), data.size(), flags, channel);
}


/////////////////////////////////////////////////
///// NETWORKING SOCKETS
/////////////////////////////////////////////////
//
//! Creates a "server" socket that listens for clients to connect to by calling ConnectByIPAddress, over ordinary UDP (IPv4 or IPv6)
uint32 Steam::createListenSocketIP(const String& ip_reference, Array options){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	const SteamNetworkingConfigValue_t *these_options = new SteamNetworkingConfigValue_t[sizeof(options)];
	these_options = convertOptionsArray(options);
	uint32 listen_socket = SteamNetworkingSockets()->CreateListenSocketIP(ip_addresses[ip_reference.utf8().get_data()], sizeof(options), these_options);
	delete[] these_options;
	return listen_socket;
}

//! Like CreateListenSocketIP, but clients will connect using ConnectP2P. The connection will be relayed through the Valve network.
uint32 Steam::createListenSocketP2P(int virtual_port, Array options){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	uint32 listen_socket = SteamNetworkingSockets()->CreateListenSocketP2P(virtual_port, sizeof(options), convertOptionsArray(options));
	return listen_socket;
}

//! Begin connecting to a server that is identified using a platform-specific identifier. This uses the default rendezvous service, which depends on the platform and library configuration. (E.g. on Steam, it goes through the steam backend.) The traffic is relayed over the Steam Datagram Relay network.
uint32 Steam::connectP2P(const String& identity_reference, int virtual_port, Array options){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamNetworkingSockets()->ConnectP2P(networking_identities[identity_reference.utf8().get_data()], virtual_port, sizeof(options), convertOptionsArray(options));
}

//! Client call to connect to a server hosted in a Valve data center, on the specified virtual port. You must have placed a ticket for this server into the cache, or else this connect attempt will fail!
uint32 Steam::connectToHostedDedicatedServer(const String& identity_reference, int virtual_port, Array options){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamNetworkingSockets()->ConnectToHostedDedicatedServer(networking_identities[identity_reference.utf8().get_data()], virtual_port, sizeof(options), convertOptionsArray(options));
}

//! Accept an incoming connection that has been received on a listen socket.
int Steam::acceptConnection(uint32 connection_handle){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamNetworkingSockets()->AcceptConnection((HSteamNetConnection)connection_handle);
}

//! Disconnects from the remote host and invalidates the connection handle. Any unread data on the connection is discarded.
bool Steam::closeConnection(uint32 peer, int reason, const String& debug_message, bool linger){
	if(SteamNetworkingSockets() == NULL){
		return false;
	}
	return SteamNetworkingSockets()->CloseConnection((HSteamNetConnection)peer, reason, debug_message.utf8().get_data(), linger);
}

//! Destroy a listen socket. All the connections that were accepted on the listen socket are closed ungracefully.
bool Steam::closeListenSocket(uint32 socket){
	if(SteamNetworkingSockets() == NULL){
		return false;
	}
	return SteamNetworkingSockets()->CloseListenSocket((HSteamListenSocket)socket);
}

//! Create a pair of connections that are talking to each other, e.g. a loopback connection. This is very useful for testing, or so that your client/server code can work the same even when you are running a local "server".
Dictionary Steam::createSocketPair(bool loopback, const String& identity_reference1, const String& identity_reference2){
	// Create a dictionary to populate
	Dictionary connection_pair;
	if(SteamNetworkingSockets() != NULL){
		// Turn the strings back to structs - Should be a check for failure to parse from string
		const SteamNetworkingIdentity identity_struct1 = networking_identities[identity_reference1.utf8().get_data()];
		const SteamNetworkingIdentity identity_struct2 = networking_identities[identity_reference2.utf8().get_data()];
		// Get connections
		uint32 connection1 = 0;
		uint32 connection2 = 0;
		bool success = SteamNetworkingSockets()->CreateSocketPair(&connection1, &connection2, loopback, &identity_struct1, &identity_struct2);
		// Populate the dictionary
		connection_pair["success"] = success;
		connection_pair["connection1"] = connection1;
		connection_pair["connect2"] = connection2;
	}
	return connection_pair;
}

//! Send a message to the remote host on the specified connection.
Dictionary Steam::sendMessageToConnection(uint32 connection_handle, const PoolByteArray data, int flags){
	Dictionary message_response;
	if(SteamNetworkingSockets() != NULL){
		int64 number;
		int result = SteamNetworkingSockets()->SendMessageToConnection((HSteamNetConnection)connection_handle, data.read().ptr(), data.size(), flags, &number);
		// Populate the dictionary
		message_response["result"] = result;
		message_response["message_number"] = (uint64_t)number;
	}
	return message_response;
}

//! Send one or more messages without copying the message payload. This is the most efficient way to send messages. To use this function, you must first allocate a message object using ISteamNetworkingUtils::AllocateMessage. (Do not declare one on the stack or allocate your own.)
void Steam::sendMessages(int messages, const PoolByteArray data, uint32 connection_handle, int flags){
	if(SteamNetworkingSockets() != NULL){
		SteamNetworkingMessage_t *networkMessage;
		networkMessage = SteamNetworkingUtils()->AllocateMessage(0);
		networkMessage->m_pData = (void *)data.read().ptr();
		networkMessage->m_cbSize = data.size();
		networkMessage->m_conn = (HSteamNetConnection)connection_handle;
		networkMessage->m_nFlags = flags;
		int64 result;
		SteamNetworkingSockets()->SendMessages(messages, &networkMessage, &result);
		// Release the message
		networkMessage->Release();
	}
}

//! Flush any messages waiting on the Nagle timer and send them at the next transmission opportunity (often that means right now).
int Steam::flushMessagesOnConnection(uint32 connection_handle){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamNetworkingSockets()->FlushMessagesOnConnection((HSteamNetConnection)connection_handle);
}

//! Fetch the next available message(s) from the connection, if any. Returns the number of messages returned into your array, up to nMaxMessages. If the connection handle is invalid, -1 is returned. If no data is available, 0, is returned.
Array Steam::receiveMessagesOnConnection(uint32 connection_handle, int max_messages){
	Array messages;
	if(SteamNetworkingSockets() != NULL){
		// Allocate the space for the messages
		SteamNetworkingMessage_t** connection_messages = new SteamNetworkingMessage_t*[max_messages];
		// Get the messages
		int available_messages = SteamNetworkingSockets()->ReceiveMessagesOnConnection((HSteamNetConnection)connection_handle, connection_messages, max_messages);
		// Loop through and create the messages as dictionaries then add to the messages array
		for(int i = 0; i < available_messages; i++){
			// Create the message dictionary to send back
			Dictionary message;
			// Get the message data
			int message_size = connection_messages[i]->m_cbSize;
			PoolByteArray data;
			data.resize(message_size);
			uint8_t* source_data = (uint8_t*)connection_messages[i]->m_pData;
			uint8_t* output_data = data.write().ptr();
			for(int j = 0; j < message_size; j++){
				output_data[j] = source_data[j];
			}
			message["payload"] = data;
			message["size"] = message_size;
			message["connection"] = connection_messages[i]->m_conn;
			char identity[STEAM_BUFFER_SIZE];
			connection_messages[i]->m_identityPeer.ToString(identity, STEAM_BUFFER_SIZE);
			message["identity"] = identity;
			message["user_data"] = (uint64_t)connection_messages[i]->m_nConnUserData;
			message["time_received"] = (uint64_t)connection_messages[i]->m_usecTimeReceived;
			message["message_number"] = (uint64_t)connection_messages[i]->m_nMessageNumber;
			message["channel"] = connection_messages[i]->m_nChannel;
			message["flags"] = connection_messages[i]->m_nFlags;
			message["user_data"] = (uint64_t)connection_messages[i]->m_nUserData;
			messages.append(message);
			// Release the message
			connection_messages[i]->Release();
		}
		delete [] connection_messages;
	}
	return messages;
}

//! Create a new poll group.
uint32 Steam::createPollGroup(){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamNetworkingSockets()->CreatePollGroup();
}

//! Destroy a poll group created with CreatePollGroup.
bool Steam::destroyPollGroup(uint32 poll_group){
	if(SteamNetworkingSockets() == NULL){
		return false;
	}
	return SteamNetworkingSockets()->DestroyPollGroup((HSteamNetPollGroup)poll_group);
}

//! Assign a connection to a poll group. Note that a connection may only belong to a single poll group. Adding a connection to a poll group implicitly removes it from any other poll group it is in.
bool Steam::setConnectionPollGroup(uint32 connection_handle, uint32 poll_group){
	if(SteamNetworkingSockets() == NULL){
		return false;
	}
	return SteamNetworkingSockets()->SetConnectionPollGroup((HSteamNetConnection)connection_handle, (HSteamNetPollGroup)poll_group);
}

//! Same as ReceiveMessagesOnConnection, but will return the next messages available on any connection in the poll group. Examine SteamNetworkingMessage_t::m_conn to know which connection. (SteamNetworkingMessage_t::m_nConnUserData might also be useful.)
Array Steam::receiveMessagesOnPollGroup(uint32 poll_group, int max_messages){
	Array messages;
	if(SteamNetworkingSockets() != NULL){
		// Allocate the space for the messages
		SteamNetworkingMessage_t** poll_messages = new SteamNetworkingMessage_t*[max_messages];
		// Get the messages
		int available_messages = SteamNetworkingSockets()->ReceiveMessagesOnPollGroup((HSteamNetPollGroup)poll_group, poll_messages, max_messages);
		// Loop through and create the messages as dictionaries then add to the messages array
		for(int i = 0; i < available_messages; i++){
			// Create the message dictionary to send back
			Dictionary message;
			// Get the message data
			int message_size = poll_messages[i]->m_cbSize;
			PoolByteArray data;
			data.resize(message_size);
			uint8_t* source_data = (uint8_t*)poll_messages[i]->m_pData;
			uint8_t* output_data = data.write().ptr();
			for(int j = 0; j < message_size; j++){
				output_data[j] = source_data[j];
			}
			message["payload"] = data;
			message["size"] = message_size;
			message["connection"] = poll_messages[i]->m_conn;
			char identity[STEAM_BUFFER_SIZE];
			poll_messages[i]->m_identityPeer.ToString(identity, STEAM_BUFFER_SIZE);
			message["identity"] = identity;
			message["user_data"] = (uint64_t)poll_messages[i]->m_nConnUserData;
			message["time_received"] = (uint64_t)poll_messages[i]->m_usecTimeReceived;
			message["message_number"] = (uint64_t)poll_messages[i]->m_nMessageNumber;
			message["channel"] = poll_messages[i]->m_nChannel;
			message["flags"] = poll_messages[i]->m_nFlags;
			message["user_data"] = (uint64_t)poll_messages[i]->m_nUserData;
			messages.append(message);
			// Release the message
			poll_messages[i]->Release();
		}
		delete [] poll_messages;
	}
	return messages;
}

//! Returns basic information about the high-level state of the connection. Returns false if the connection handle is invalid.
Dictionary Steam::getConnectionInfo(uint32 connection_handle){
	Dictionary connection_info;
	if(SteamNetworkingSockets() != NULL){
		SteamNetConnectionInfo_t info;
		if(SteamNetworkingSockets()->GetConnectionInfo((HSteamNetConnection)connection_handle, &info)){
			char identity[STEAM_BUFFER_SIZE];
			info.m_identityRemote.ToString(identity, STEAM_BUFFER_SIZE);
			connection_info["identity"] = identity;
			connection_info["user_data"] = (uint64_t)info.m_nUserData;
			connection_info["listen_socket"] = info.m_hListenSocket;
			char ip_address[STEAM_BUFFER_SIZE];
			info.m_addrRemote.ToString(ip_address, STEAM_BUFFER_SIZE, true);
			connection_info["remote_address"] = ip_address;
			connection_info["remote_pop"] = info.m_idPOPRemote;
			connection_info["pop_relay"] = info.m_idPOPRelay;
			connection_info["connection_state"] = info.m_eState;
			connection_info["end_reason"] = info.m_eEndReason;
			connection_info["end_debug"] = info.m_szEndDebug;
			connection_info["debug_description"] = info.m_szConnectionDescription;
		}
	}
	return connection_info;
}

//! Returns very detailed connection stats in diagnostic text format. Useful for dumping to a log, etc. The format of this information is subject to change.
Dictionary Steam::getDetailedConnectionStatus(uint32 connection){
	Dictionary connectionStatus;
	if(SteamNetworkingSockets() != NULL){
		char buffer[STEAM_LARGE_BUFFER_SIZE];
		int success = SteamNetworkingSockets()->GetDetailedConnectionStatus((HSteamNetConnection)connection, buffer, STEAM_LARGE_BUFFER_SIZE);
		// Add data to dictionary
		connectionStatus["success"] = success;
		connectionStatus["status"] = buffer;
	}
	// Send the data back to the user
	return connectionStatus; 
}

//! Fetch connection user data. Returns -1 if handle is invalid or if you haven't set any userdata on the connection.
uint64_t Steam::getConnectionUserData(uint32 peer){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamNetworkingSockets()->GetConnectionUserData((HSteamNetConnection)peer);
}

//! Set a name for the connection, used mostly for debugging
void Steam::setConnectionName(uint32 peer, const String& name){
	if(SteamNetworkingSockets() != NULL){
		SteamNetworkingSockets()->SetConnectionName((HSteamNetConnection)peer, name.utf8().get_data());
	}
}

//! Fetch connection name into your buffer, which is at least nMaxLen bytes. Returns false if handle is invalid.
String Steam::getConnectionName(uint32 peer){
	// Set empty string variable for use
	String connection_name = "";
	if(SteamNetworkingSockets() != NULL){
		char name[STEAM_BUFFER_SIZE];
		if(SteamNetworkingSockets()->GetConnectionName((HSteamNetConnection)peer, name, STEAM_BUFFER_SIZE)){
			connection_name += name;	
		}
	}
	return connection_name;
}

//! Returns local IP and port that a listen socket created using CreateListenSocketIP is bound to.
bool Steam::getListenSocketAddress(uint32 socket){
	if(SteamNetworkingSockets() == NULL){
		return false;
	}
	SteamNetworkingIPAddr address;
	return SteamNetworkingSockets()->GetListenSocketAddress((HSteamListenSocket)socket, &address);
}

//! Get the identity assigned to this interface.
String Steam::getIdentity(){
	String identity_string = "";
	if(SteamNetworkingSockets() != NULL){
		SteamNetworkingIdentity this_identity;
		if(SteamNetworkingSockets()->GetIdentity(&this_identity)){
			char *this_buffer = new char[128];
			this_identity.ToString(this_buffer, 128);
			identity_string = String(this_buffer);
			delete[] this_buffer;
		}
	}
	return identity_string;
}

//! Indicate our desire to be ready participate in authenticated communications. If we are currently not ready, then steps will be taken to obtain the necessary certificates. (This includes a certificate for us, as well as any CA certificates needed to authenticate peers.)
int Steam::initAuthentication(){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamNetworkingSockets()->InitAuthentication();
}

//! Query our readiness to participate in authenticated communications. A SteamNetAuthenticationStatus_t callback is posted any time this status changes, but you can use this function to query it at any time.
int Steam::getAuthenticationStatus(){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamNetworkingSockets()->GetAuthenticationStatus(NULL);
}

// Call this when you receive a ticket from your backend / matchmaking system. Puts the ticket into a persistent cache, and optionally returns the parsed ticket.
//Dictionary Steam::receivedRelayAuthTicket(){
//	Dictionary ticket;
//	if(SteamNetworkingSockets() != NULL){
//		SteamDatagramRelayAuthTicket parsed_ticket;
//		PoolByteArray incoming_ticket;
//		incoming_ticket.resize(512);		
//		if(SteamNetworkingSockets()->ReceivedRelayAuthTicket(incoming_ticket.write().ptr(), 512, &parsed_ticket)){
//			char game_server;
//			parsed_ticket.m_identityGameserver.ToString(&game_server, 128);
//			ticket["game_server"] = game_server;
//			char authorized_client;
//			parsed_ticket.m_identityAuthorizedClient.ToString(&authorized_client, 128);
//			ticket["authorized_client"] = authorized_client;
//			ticket["public_ip"] = parsed_ticket.m_unPublicIP;		// uint32
//			ticket["expiry"] = parsed_ticket.m_rtimeTicketExpiry;	// RTime32
//			ticket["routing"] = parsed_ticket.m_routing.GetPopID();			// SteamDatagramHostAddress
//			ticket["app_id"] = parsed_ticket.m_nAppID;				// uint32
//			ticket["restrict_to_v_port"] = parsed_ticket.m_nRestrictToVirtualPort;	// int
//			ticket["number_of_extras"] = parsed_ticket.m_nExtraFields;		// int
//			ticket["extra_fields"] = parsed_ticket.m_vecExtraFields;		// ExtraField
//		}
//	}
//	return ticket;
//}

// Search cache for a ticket to talk to the server on the specified virtual port. If found, returns the number of seconds until the ticket expires, and optionally the complete cracked ticket. Returns 0 if we don't have a ticket.
//int Steam::findRelayAuthTicketForServer(int port){
//	int expires_in_seconds = 0;
//	if(SteamNetworkingSockets() != NULL){
//		expires_in_seconds = SteamNetworkingSockets()->FindRelayAuthTicketForServer(game_server, port, &relay_auth_ticket);
//	}
//	return expires_in_seconds;
//}



//! Returns the value of the SDR_LISTEN_PORT environment variable. This is the UDP server your server will be listening on. This will configured automatically for you in production environments.
uint16 Steam::getHostedDedicatedServerPort(){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamNetworkingSockets()->GetHostedDedicatedServerPort();
}

//! Returns 0 if SDR_LISTEN_PORT is not set. Otherwise, returns the data center the server is running in. This will be k_SteamDatagramPOPID_dev in non-production envirionment.
uint32 Steam::getHostedDedicatedServerPOPId(){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamNetworkingSockets()->GetHostedDedicatedServerPOPID();
}

// Return info about the hosted server. This contains the PoPID of the server, and opaque routing information that can be used by the relays to send traffic to your server.
//int Steam::getHostedDedicatedServerAddress(){
//	int result = 2;
//	if(SteamNetworkingSockets() != NULL){
//		result = SteamNetworkingSockets()->GetHostedDedicatedServerAddress(&hosted_address);
//	}
//	return result;
//}

//! Create a listen socket on the specified virtual port. The physical UDP port to use will be determined by the SDR_LISTEN_PORT environment variable. If a UDP port is not configured, this call will fail.
uint32 Steam::createHostedDedicatedServerListenSocket(int port, Array options){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamGameServerNetworkingSockets()->CreateHostedDedicatedServerListenSocket(port, sizeof(options), convertOptionsArray(options));
}

// Generate an authentication blob that can be used to securely login with your backend, using SteamDatagram_ParseHostedServerLogin. (See steamdatagram_gamecoordinator.h)
//int Steam::getGameCoordinatorServerLogin(const String& app_data){
//	int result = 2;
//	if(SteamNetworkingSockets() != NULL){	
//		SteamDatagramGameCoordinatorServerLogin *server_login = new SteamDatagramGameCoordinatorServerLogin;
//		server_login->m_cbAppData = app_data.size();
//		strcpy(server_login->m_appData, app_data.utf8().get_data());
//		int signed_blob = k_cbMaxSteamDatagramGameCoordinatorServerLoginSerialized;
//		routing_blob.resize(signed_blob);
//		result = SteamNetworkingSockets()->GetGameCoordinatorServerLogin(server_login, &signed_blob, routing_blob.write().ptr());
//		delete server_login;
//	}
//	return result;
//}

// Returns a small set of information about the real-time state of the connection and the queue status of each lane.
Dictionary Steam::getConnectionRealTimeStatus(uint32 connection, int lanes, bool get_status){
	// Create the dictionary for returning
	Dictionary real_time_status;
	if(SteamNetworkingSockets() != NULL){
		SteamNetConnectionRealTimeStatus_t this_status;
		SteamNetConnectionRealTimeLaneStatus_t *lanes_array = new SteamNetConnectionRealTimeLaneStatus_t[lanes];
		int result = SteamNetworkingSockets()->GetConnectionRealTimeStatus((HSteamNetConnection)connection, &this_status, lanes, lanes_array);
		// Append the status
		real_time_status["response"] = result;
		// If the result is good, get more data
		if(result == 0){
			// Get the connection status if requested
			Dictionary connection_status;
			if(get_status){
				connection_status["state"] = this_status.m_eState;
				connection_status["ping"] = this_status.m_nPing;
				connection_status["local_quality"] = this_status.m_flConnectionQualityLocal;
				connection_status["remote_quality"] = this_status.m_flConnectionQualityRemote;
				connection_status["packets_out_per_second"] = this_status.m_flOutPacketsPerSec;
				connection_status["bytes_out_per_second"] = this_status.m_flOutBytesPerSec;
				connection_status["packets_in_per_second"] = this_status.m_flInPacketsPerSec;
				connection_status["bytes_in_per_second"] = this_status.m_flInBytesPerSec;
				connection_status["send_rate"] = this_status.m_nSendRateBytesPerSecond;
				connection_status["pending_unreliable"] = this_status.m_cbPendingUnreliable;
				connection_status["pending_reliable"] = this_status.m_cbPendingReliable;
				connection_status["sent_unacknowledged_reliable"] = this_status.m_cbSentUnackedReliable;
				connection_status["queue_time"] = (uint64_t)this_status.m_usecQueueTime;
			}
			real_time_status["connection_status"] = connection_status;
			// Get the lane information
			Array lanes_status;
			for(int i = 0; i < lanes; i++){
				Dictionary lane_status;
				lane_status["pending_unreliable"] = lanes_array[i].m_cbPendingUnreliable;
				lane_status["pending_reliable"] = lanes_array[i].m_cbPendingReliable;
				lane_status["sent_unacknowledged_reliable"] = lanes_array[i].m_cbSentUnackedReliable;
				lane_status["queue_time"] = (uint64_t)lanes_array[i].m_usecQueueTime;
				lanes_status.append(lane_status);
			}
			delete[] lanes_array;
			real_time_status["lanes_status"] = lanes_status;
		}
	}
	return real_time_status;
}

// Configure multiple outbound messages streams ("lanes") on a connection, and control head-of-line blocking between them.
// Messages within a given lane are always sent in the order they are queued, but messages from different lanes may be sent out of order.
// Each lane has its own message number sequence.  The first message sent on each lane will be assigned the number 1.
int Steam::configureConnectionLanes(uint32 connection, int lanes, Array priorities, Array weights) {
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	// Convert the priorities array to an int
	int *lane_priorities = new int[lanes];
	for(int i = 0; i < lanes; i++){
		lane_priorities[i] = priorities[i];
	}
	// Convert the weights array to an int
	uint16 *lane_weights = new uint16[lanes];
	for(int i = 0; i < lanes; i++){
		lane_weights[i] = weights[i];
	}
	int result = SteamNetworkingSockets()->ConfigureConnectionLanes((HSteamNetConnection)connection, lanes, lane_priorities, lane_weights);
	delete[] lane_priorities;
	delete[] lane_weights;
	return result;
}

// Certificate provision by the application. On Steam, we normally handle all this automatically and you will not need to use these advanced functions.
Dictionary Steam::getCertificateRequest(){
	Dictionary cert_information;
	if(SteamNetworkingSockets() != NULL){
		int *certificate = new int[512];
		int cert_size = 0;
		SteamNetworkingErrMsg error_message;
		if(SteamNetworkingSockets()->GetCertificateRequest(&cert_size, &certificate, error_message)){
			cert_information["certificate"] = certificate;
			cert_information["cert_size"] = cert_size;
			cert_information["error_message"] = error_message;
		}
		delete[] certificate;
	}
	return cert_information;
}

// Set the certificate. The certificate blob should be the output of SteamDatagram_CreateCert.
Dictionary Steam::setCertificate(const PoolByteArray& certificate){
	Dictionary certificate_data;
	if(SteamNetworkingSockets() != NULL){
		bool success = false;
		SteamNetworkingErrMsg error_message;
		success = SteamNetworkingSockets()->SetCertificate((void*)certificate.read().ptr(), certificate.size(), error_message);
		if(success){
			certificate_data["response"] = success;
			certificate_data["error"] = error_message;
		}
	}
	return certificate_data;
}

// Reset the identity associated with this instance. Any open connections are closed.  Any previous certificates, etc are discarded.
// You can pass a specific identity that you want to use, or you can pass NULL, in which case the identity will be invalid until you set it using SetCertificate.
// NOTE: This function is not actually supported on Steam!  It is included for use on other platforms where the active user can sign out and a new user can sign in.
void Steam::resetIdentity(const String& identity_reference){
	if(SteamNetworkingSockets() != NULL){
		SteamNetworkingIdentity resetting_identity = networking_identities[identity_reference.utf8().get_data()];
		SteamNetworkingSockets()->ResetIdentity(&resetting_identity);
	}
}

// Invoke all callback functions queued for this interface. See k_ESteamNetworkingConfig_Callback_ConnectionStatusChanged, etc.
// You don't need to call this if you are using Steam's callback dispatch mechanism (SteamAPI_RunCallbacks and SteamGameserver_RunCallbacks).
void Steam::runNetworkingCallbacks(){
	if(SteamNetworkingSockets() != NULL){
		SteamNetworkingSockets()->RunCallbacks();		
	}
}

// Begin asynchronous process of allocating a fake IPv4 address that other peers can use to contact us via P2P.
// IP addresses returned by this function are globally unique for a given appid.
// Returns false if a request was already in progress, true if a new request was started.
// A SteamNetworkingFakeIPResult_t will be posted when the request completes.
bool Steam::beginAsyncRequestFakeIP(int num_ports){
	if(SteamNetworkingSockets() == NULL){
		return false;
	}
	return SteamNetworkingSockets()->BeginAsyncRequestFakeIP(num_ports);
}

// Return info about the FakeIP and port(s) that we have been assigned, if any.
// idxFirstPort is currently reserved and must be zero. Make sure and check SteamNetworkingFakeIPResult_t::m_eResult
Dictionary Steam::getFakeIP(int first_port){
	// Create the return dictionary
	Dictionary fake_ip;
	if(SteamNetworkingSockets() != NULL){
		SteamNetworkingFakeIPResult_t fake_ip_result;
		SteamNetworkingSockets()->GetFakeIP(first_port, &fake_ip_result);
		// Populate the dictionary
		fake_ip["result"] = fake_ip_result.m_eResult;
		fake_ip["identity_type"] = fake_ip_result.m_identity.m_eType;
		fake_ip["ip"] = fake_ip_result.m_unIP;
		char ports[8];
		for (size_t i = 0; i < sizeof(fake_ip_result.m_unPorts) / sizeof(fake_ip_result.m_unPorts[0]); i++){
			ports[i] = fake_ip_result.m_unPorts[i];
		}
		fake_ip["ports"] = ports;
	}
	return fake_ip;
}

// Create a listen socket that will listen for P2P connections sent to our FakeIP.
// A peer can initiate connections to this listen socket by calling ConnectByIPAddress.
uint32 Steam::createListenSocketP2PFakeIP(int fake_port, Array options){
	if(SteamNetworkingSockets() == NULL){
		return 0;
	}
	return SteamNetworkingSockets()->CreateListenSocketP2PFakeIP(fake_port, sizeof(options), convertOptionsArray(options));
}

// If the connection was initiated using the "FakeIP" system, then we we can get an IP address for the remote host.  If the remote host had a global FakeIP at the time the connection was established, this function will return that global IP.
// Otherwise, a FakeIP that is unique locally will be allocated from the local FakeIP address space, and that will be returned.
Dictionary Steam::getRemoteFakeIPForConnection(uint32 connection){
	// Create the return dictionary
	Dictionary this_fake_address;
	if(SteamNetworkingSockets() != NULL){
		SteamNetworkingIPAddr fake_address;
		int result = SteamNetworkingSockets()->GetRemoteFakeIPForConnection((HSteamNetConnection)connection, &fake_address);
		// Send back the data
		this_fake_address["result"] = result;
		this_fake_address["port"] = fake_address.m_port;
		this_fake_address["ip_type"] = fake_address.GetFakeIPType();
		ip_addresses["fake_ip_address"] = fake_address;
		}

	return this_fake_address;
}

// Get an interface that can be used like a UDP port to send/receive datagrams to a FakeIP address.
// This is intended to make it easy to port existing UDP-based code to take advantage of SDR.
// To create a "client" port (e.g. the equivalent of an ephemeral UDP port) pass -1.
void Steam::createFakeUDPPort(int fake_server_port_index){
	if(SteamNetworkingSockets() != NULL){
		SteamNetworkingSockets()->CreateFakeUDPPort(fake_server_port_index);
	}
}


/////////////////////////////////////////////////
///// NETWORKING TYPES
/////////////////////////////////////////////////
//
// Create a new network identity and store it for use
bool Steam::addIdentity(const String& reference_name){
	networking_identities[reference_name.utf8().get_data()] = SteamNetworkingIdentity();
	if(networking_identities.count(reference_name.utf8().get_data()) > 0){
		return true;
	}
	return false;
}

// Clear a network identity's data
void Steam::clearIdentity(const String& reference_name){
	networking_identities[reference_name.utf8().get_data()].Clear();
}


// Get a list of all known network identities
Array Steam::getIdentities(){
	Array these_identities;
	// Loop through the map
	for(auto& identity : networking_identities){
		Dictionary this_identity;
		this_identity["reference_name"] = identity.first;
		this_identity["steam_id"] = (uint64_t)getIdentitySteamID64(identity.first);
		this_identity["type"] = networking_identities[identity.first].m_eType;
		these_identities.append(this_identity);
	}
	return these_identities;
}


// Return true if we are the invalid type.  Does not make any other validity checks (e.g. is SteamID actually valid)
bool Steam::isIdentityInvalid(const String& reference_name){
	return networking_identities[reference_name.utf8().get_data()].IsInvalid();
}

// Set a 32-bit Steam ID
void Steam::setIdentitySteamID(const String& reference_name, uint32 steam_id){
	networking_identities[reference_name.utf8().get_data()].SetSteamID(createSteamID(steam_id));
}

// Return black CSteamID (!IsValid()) if identity is not a SteamID
uint32 Steam::getIdentitySteamID(const String& reference_name){
	CSteamID steam_id = networking_identities[reference_name.utf8().get_data()].GetSteamID();
	return steam_id.ConvertToUint64();
}

// Takes SteamID as raw 64-bit number
void Steam::setIdentitySteamID64(const String& reference_name, uint64_t steam_id){
	networking_identities[reference_name.utf8().get_data()].SetSteamID64(steam_id);
}

// Returns 0 if identity is not SteamID
uint64_t Steam::getIdentitySteamID64(const String& reference_name){
	return networking_identities[reference_name.utf8().get_data()].GetSteamID64();
}

// Set to specified IP:port.
bool Steam::setIdentityIPAddr(const String& reference_name, const String& ip_address_name){
	if(ip_addresses.count(ip_address_name.utf8().get_data()) > 0){
		const SteamNetworkingIPAddr this_address = ip_addresses[ip_address_name.utf8().get_data()];
		networking_identities[reference_name.utf8().get_data()].SetIPAddr(this_address);
		return true;
	}
	return false;
}

// Returns null if we are not an IP address.
uint32 Steam::getIdentityIPAddr(const String& reference_name){
	const SteamNetworkingIPAddr* this_address = networking_identities[reference_name.utf8().get_data()].GetIPAddr();
	if (this_address == NULL){
		return 0;
	}
	return this_address->GetIPv4();
}

// Retrieve this identity's Playstation Network ID.
uint64_t Steam::getPSNID(const String& reference_name){
	return networking_identities[reference_name.utf8().get_data()].GetPSNID();
}

// Retrieve this identity's Google Stadia ID.
uint64_t Steam::getStadiaID(const String& reference_name){
	return networking_identities[reference_name.utf8().get_data()].GetStadiaID();
}

// Retrieve this identity's XBox pair ID.
String Steam::getXboxPairwiseID(const String& reference_name){
	return networking_identities[reference_name.utf8().get_data()].GetXboxPairwiseID();
}

// Set to localhost. (We always use IPv6 ::1 for this, not 127.0.0.1).
void Steam::setIdentityLocalHost(const String& reference_name){
	networking_identities[reference_name.utf8().get_data()].SetLocalHost();
}

// Return true if this identity is localhost.
bool Steam::isIdentityLocalHost(const String& reference_name){
	return networking_identities[reference_name.utf8().get_data()].IsLocalHost();
}

// Returns false if invalid length.
bool Steam::setGenericString(const String& reference_name, const String& this_string){
	return networking_identities[reference_name.utf8().get_data()].SetGenericString(this_string.utf8().get_data());
}

// Returns nullptr if not generic string type
String Steam::getGenericString(const String& reference_name){
	return networking_identities[reference_name.utf8().get_data()].GetGenericString();
}

// Returns false if invalid size.
bool Steam::setGenericBytes(const String& reference_name, uint8 data){
	const void *this_data = &data;
	return networking_identities[reference_name.utf8().get_data()].SetGenericBytes(this_data, sizeof(data));
}

// Returns null if not generic bytes type.
uint8 Steam::getGenericBytes(const String& reference_name){
	uint8 these_bytes = 0;
	if(!reference_name.empty()){
		int length = 0;
		const uint8* generic_bytes = networking_identities[reference_name.utf8().get_data()].GetGenericBytes(length);
		these_bytes = *generic_bytes;
	}
	return these_bytes;
}

// Add a new IP address struct
bool Steam::addIPAddress(const String& reference_name){
	ip_addresses[reference_name.utf8().get_data()] = SteamNetworkingIPAddr();
	if(ip_addresses.count(reference_name.utf8().get_data()) > 0){
		return true;
	}
	return false;
}

// Get a list of all IP address structs and their names
Array Steam::getIPAddresses(){
	Array these_addresses;
	// Loop through the map
	for(auto& address : ip_addresses){
		Dictionary this_address;
		this_address["reference_name"] = address.first;
		this_address["localhost"] = isAddressLocalHost(address.first);
		this_address["ip_address"] = getIPv4(address.first);
		these_addresses.append(this_address);
	}
	return these_addresses;
}

// IP Address - Set everything to zero. E.g. [::]:0
void Steam::clearIPAddress(const String& reference_name){
	ip_addresses[reference_name.utf8().get_data()].Clear();
}

// Return true if the IP is ::0. (Doesn't check port.)
bool Steam::isIPv6AllZeros(const String& reference_name){
	return ip_addresses[reference_name.utf8().get_data()].IsIPv6AllZeros();
}

// Set IPv6 address. IP is interpreted as bytes, so there are no endian issues. (Same as inaddr_in6.) The IP can be a mapped IPv4 address.
void Steam::setIPv6(const String& reference_name, uint8 ipv6, uint16 port){
	const uint8 *this_ipv6 = &ipv6;
	ip_addresses[reference_name.utf8().get_data()].SetIPv6(this_ipv6, port);
}

// Sets to IPv4 mapped address. IP and port are in host byte order.
void Steam::setIPv4(const String& reference_name, uint32 ip, uint16 port){
	ip_addresses[reference_name.utf8().get_data()].SetIPv4(ip, port);
}

// Return true if IP is mapped IPv4.
bool Steam::isIPv4(const String& reference_name){
	return ip_addresses[reference_name.utf8().get_data()].IsIPv4();
}

// Returns IP in host byte order (e.g. aa.bb.cc.dd as 0xaabbccdd). Returns 0 if IP is not mapped IPv4.
uint32 Steam::getIPv4(const String& reference_name){
	return ip_addresses[reference_name.utf8().get_data()].GetIPv4();
}

// Set to the IPv6 localhost address ::1, and the specified port.
void Steam::setIPv6LocalHost(const String& reference_name, uint16 port){
	ip_addresses[reference_name.utf8().get_data()].SetIPv6LocalHost(port);
}

// Set the Playstation Network ID for this identity.
void Steam::setPSNID(const String& reference_name, uint64_t psn_id){
	networking_identities[reference_name.utf8().get_data()].SetPSNID(psn_id);
}

// Set the Google Stadia ID for this identity.
void Steam::setStadiaID(const String& reference_name, uint64_t stadia_id){
	networking_identities[reference_name.utf8().get_data()].SetStadiaID(stadia_id);
}

// Set the Xbox Pairwise ID for this identity.
bool Steam::setXboxPairwiseID(const String& reference_name, const String& xbox_id){
	return networking_identities[reference_name.utf8().get_data()].SetXboxPairwiseID(xbox_id.utf8().get_data());
}

// Return true if this identity is localhost. (Either IPv6 ::1, or IPv4 127.0.0.1).
bool Steam::isAddressLocalHost(const String& reference_name){
	return ip_addresses[reference_name.utf8().get_data()].IsLocalHost();
}

// Parse back a string that was generated using ToString. If we don't understand the string, but it looks "reasonable" (it matches the pattern type:<type-data> and doesn't have any funky characters, etc), then we will return true, and the type is set to k_ESteamNetworkingIdentityType_UnknownType.
// false will only be returned if the string looks invalid.
bool Steam::parseIdentityString(const String& reference_name, const String& string_to_parse){
	if(!reference_name.empty() && !string_to_parse.empty()){
		if(networking_identities[reference_name.utf8().get_data()].ParseString(string_to_parse.utf8().get_data())){
			return true;
		}
		return false;
	}
	return false;
}

// Parse an IP address and optional port.  If a port is not present, it is set to 0. (This means that you cannot tell if a zero port was explicitly specified.).
bool Steam::parseIPAddressString(const String& reference_name, const String& string_to_parse){
	if(!reference_name.empty() && !string_to_parse.empty()){
		if(ip_addresses[reference_name.utf8().get_data()].ParseString(string_to_parse.utf8().get_data())){
			return true;
		}
		return false;
	}
	return false;
}

// Print to a string, with or without the port. Mapped IPv4 addresses are printed as dotted decimal (12.34.56.78), otherwise this will print the canonical form according to RFC5952.
// If you include the port, IPv6 will be surrounded by brackets, e.g. [::1:2]:80. Your buffer should be at least k_cchMaxString bytes to avoid truncation.
String Steam::toIPAddressString(const String& reference_name, bool with_port){
	String ip_address_string = "";
	char *this_buffer = new char[128];
	ip_addresses[reference_name.utf8().get_data()].ToString(this_buffer, 128, with_port);
	ip_address_string = String(this_buffer);
	delete[] this_buffer;
	return ip_address_string;
}

// Print to a human-readable string.  This is suitable for debug messages or any other time you need to encode the identity as a string.
// It has a URL-like format (type:<type-data>). Your buffer should be at least k_cchMaxString bytes big to avoid truncation.
String Steam::toIdentityString(const String& reference_name){
	String identity_string = "";
	char *this_buffer = new char[128];
	networking_identities[reference_name.utf8().get_data()].ToString(this_buffer, 128);
	identity_string = String(this_buffer);
	delete[] this_buffer;
	return identity_string;
}

// Helper function to turn an array of options into an array of SteamNetworkingConfigValue_t structs
const SteamNetworkingConfigValue_t* Steam::convertOptionsArray(Array options){
	// Get the number of option arrays in the array.
	int options_size = sizeof(options);
	// Create the array for options.
	SteamNetworkingConfigValue_t *option_array = new SteamNetworkingConfigValue_t[options_size];
	// If there are options
	if(options_size > 0){
		// Populate the options
		for(int i = 0; i < options_size; i++){
			SteamNetworkingConfigValue_t this_option;
			Array sent_option = options[i];
			// Get the configuration value.
			// This is a convoluted way of doing it but can't seem to cast the value as an enum so here we are.
			ESteamNetworkingConfigValue this_value = ESteamNetworkingConfigValue((int)sent_option[0]);
			if((int)sent_option[1] == 1){
				this_option.SetInt32(this_value, sent_option[2]);
			}
			else if((int)sent_option[1] == 2){
				this_option.SetInt64(this_value, sent_option[2]);
			}
			else if((int)sent_option[1] == 3){
				this_option.SetFloat(this_value, sent_option[2]);
			}
			else if((int)sent_option[1] == 4){
				char *this_string = { 0 };
				String passed_string = sent_option[2];
				strcpy(this_string, passed_string.utf8().get_data());
				this_option.SetString(this_value, this_string);
			}
			else{
				Object *this_pointer;
				this_pointer = sent_option[2];
				this_option.SetPtr(this_value, this_pointer);
			}
			option_array[i] = this_option;
		}
	}
	return option_array;
}


/////////////////////////////////////////////////
///// NETWORKING UTILS
/////////////////////////////////////////////////
//
//! If you know that you are going to be using the relay network (for example, because you anticipate making P2P connections), call this to initialize the relay network. If you do not call this, the initialization will be delayed until the first time you use a feature that requires access to the relay network, which will delay that first access.
void Steam::initRelayNetworkAccess(){
	if(SteamNetworkingUtils() != NULL){
		SteamNetworkingUtils()->InitRelayNetworkAccess();
	}
}

//! Fetch current status of the relay network.  If you want more details, you can pass a non-NULL value.
int Steam::getRelayNetworkStatus(){
	if(SteamNetworkingUtils() == NULL){
		return 0;
	}
	return SteamNetworkingUtils()->GetRelayNetworkStatus(NULL);
}

//! Return location info for the current host. Returns the approximate age of the data, in seconds, or -1 if no data is available.
Dictionary Steam::getLocalPingLocation(){
	Dictionary ping_location;
	if(SteamNetworkingUtils() != NULL){
		SteamNetworkPingLocation_t location;
		float ping = SteamNetworkingUtils()->GetLocalPingLocation(location);
		char m_data[512];
		for(size_t i = 0; i < sizeof(location.m_data) / sizeof(location.m_data[0]); i++){
			m_data[i] = location.m_data[i];
		}
		// Populate the dictionary
		ping_location["location"] = m_data;
		ping_location["ping"] = ping;
	}
	return ping_location;
}

//! Estimate the round-trip latency between two arbitrary locations, in milliseconds. This is a conservative estimate, based on routing through the relay network. For most basic relayed connections, this ping time will be pretty accurate, since it will be based on the route likely to be actually used.
int Steam::estimatePingTimeBetweenTwoLocations(uint8 location1, uint8 location2){
	if(SteamNetworkingUtils() == NULL){
		return 0;
	}
	// Add these locations to ping structs
	SteamNetworkPingLocation_t ping_location1;
	SteamNetworkPingLocation_t ping_location2;
	ping_location1.m_data[0] = {location1};
	ping_location2.m_data[0] = {location2};
	return SteamNetworkingUtils()->EstimatePingTimeBetweenTwoLocations(ping_location1, ping_location2);
}

//! Same as EstimatePingTime, but assumes that one location is the local host. This is a bit faster, especially if you need to calculate a bunch of these in a loop to find the fastest one.
int Steam::estimatePingTimeFromLocalHost(uint8 location){
	if(SteamNetworkingUtils() == NULL){
		return 0;
	}
	// Add this location to ping struct
	SteamNetworkPingLocation_t ping_location;
	ping_location.m_data[0] = {location};
	return SteamNetworkingUtils()->EstimatePingTimeFromLocalHost(ping_location);
}

//! Convert a ping location into a text format suitable for sending over the wire. The format is a compact and human readable. However, it is subject to change so please do not parse it yourself. Your buffer must be at least k_cchMaxSteamNetworkingPingLocationString bytes.
String Steam::convertPingLocationToString(uint8 location){
	String location_string = "";
	if(SteamNetworkingUtils() != NULL){
		char buffer;
		// Add this location to ping struct
		SteamNetworkPingLocation_t ping_location;
		ping_location.m_data[0] = {location};
		SteamNetworkingUtils()->ConvertPingLocationToString(ping_location, &buffer, k_cchMaxSteamNetworkingPingLocationString);
		location_string += buffer;
	}
	return location_string;
}

//! Parse back SteamNetworkPingLocation_t string. Returns false if we couldn't understand the string.
Dictionary Steam::parsePingLocationString(const String& location_string){
	Dictionary parse_string;
	if(SteamNetworkingUtils() != NULL){
		SteamNetworkPingLocation_t result;
		bool success = SteamNetworkingUtils()->ParsePingLocationString(location_string.utf8().get_data(), result);
		// Populate the dictionary
		parse_string["success"] = success;
		char m_data[512];
		for(size_t i = 0; i < sizeof(result.m_data) / sizeof(result.m_data[0]); i++){
			m_data[i] = result.m_data[i];
		}
		parse_string["ping_location"] = m_data;
	}
	return parse_string;
}

//! Check if the ping data of sufficient recency is available, and if it's too old, start refreshing it.
bool Steam::checkPingDataUpToDate(float max_age_in_seconds){
	if(SteamNetworkingUtils() == NULL){
		return false;
	}
	return SteamNetworkingUtils()->CheckPingDataUpToDate(max_age_in_seconds);
}

//! Fetch ping time of best available relayed route from this host to the specified data center.
Dictionary Steam::getPingToDataCenter(uint32 pop_id){
	Dictionary data_center_ping;
	if(SteamNetworkingUtils() != NULL){
		SteamNetworkingPOPID via_relay_pop;
		int ping = SteamNetworkingUtils()->GetPingToDataCenter((SteamNetworkingPOPID)pop_id, &via_relay_pop);
		// Populate the dictionary
		data_center_ping["pop_relay"] = via_relay_pop;
		data_center_ping["ping"] = ping;
	}
	return data_center_ping;
}

//! Get *direct* ping time to the relays at the point of presence.
int Steam::getDirectPingToPOP(uint32 pop_id){
	if(SteamNetworkingUtils() == NULL){
		return 0;
	}
	return SteamNetworkingUtils()->GetDirectPingToPOP((SteamNetworkingPOPID)pop_id);
}

//! Get number of network points of presence in the config
int Steam::getPOPCount(){
	if(SteamNetworkingUtils() == NULL){
		return 0;
	}
	return SteamNetworkingUtils()->GetPOPCount();
}

//! Get list of all POP IDs. Returns the number of entries that were filled into your list.
Array Steam::getPOPList(){
	Array pop_list;
	if(SteamNetworkingUtils() != NULL){
		SteamNetworkingPOPID *list = new SteamNetworkingPOPID[256];
		int pops = SteamNetworkingUtils()->GetPOPList(list, 256);
		// Iterate and add
		for(int i = 0; i < pops; i++){
			int pop_id = list[i];
			pop_list.append(pop_id);
		}
		delete[] list;
	}
	return pop_list;
}

// Set a configuration value.
//bool Steam::setConfigValue(int setting, int scope_type, uint32_t connection_handle, int data_type, auto value){
//	if(SteamNetworkingUtils() == NULL){
//		return false;
//	}
//	return SteamNetworkingUtils()->SetConfigValue((ESteamNetworkingConfigValue)setting, (ESteamNetworkingConfigScope)scope_type, connection_handle, (ESteamNetworkingConfigDataType)data_type, value);
//}

// Get a configuration value.
Dictionary Steam::getConfigValue(int config_value, int scope_type, uint32_t connection_handle){
	Dictionary config_info;
	if(SteamNetworkingUtils() != NULL){
		ESteamNetworkingConfigDataType data_type;
		size_t buffer_size;
		PoolByteArray config_result;
		int result = SteamNetworkingUtils()->GetConfigValue((ESteamNetworkingConfigValue)config_value, (ESteamNetworkingConfigScope)scope_type, connection_handle, &data_type, &config_result, &buffer_size);
		// Populate the dictionary
		config_info["result"] = result;
		config_info["type"] = data_type;
		config_info["value"] = config_result;
		config_info["buffer"] = (uint64_t)buffer_size;
	}
	return config_info;
}

//! Returns info about a configuration value.
Dictionary Steam::getConfigValueInfo(int config_value){
	Dictionary config_info;
	if(SteamNetworkingUtils() != NULL){
		ESteamNetworkingConfigDataType data_type;
		ESteamNetworkingConfigScope scope;
		if(SteamNetworkingUtils()->GetConfigValueInfo((ESteamNetworkingConfigValue)config_value, &data_type, &scope)){
			// Populate the dictionary
			config_info["type"] = data_type;
			config_info["scope"] = scope;
		}
	}
	return config_info;
}

// The following functions are handy shortcuts for common use cases.
bool Steam::setGlobalConfigValueInt32(int config, int32 value){
	if(SteamNetworkingUtils() == NULL){
		return false;
	}
	return SteamNetworkingUtils()->SetGlobalConfigValueInt32((ESteamNetworkingConfigValue)config, value);
}
bool Steam::setGlobalConfigValueFloat(int config, float value){
	if(SteamNetworkingUtils() == NULL){
		return false;
	}
	return SteamNetworkingUtils()->SetGlobalConfigValueFloat((ESteamNetworkingConfigValue)config, value);
}
bool Steam::setGlobalConfigValueString(int config, const String& value){
	if(SteamNetworkingUtils() == NULL){
		return false;
	}
	return SteamNetworkingUtils()->SetGlobalConfigValueString((ESteamNetworkingConfigValue)config, value.utf8().get_data());
}
bool Steam::setConnectionConfigValueInt32(uint32 connection, int config, int32 value){
	if(SteamNetworkingUtils() == NULL){
		return false;
	}
	return SteamNetworkingUtils()->SetConnectionConfigValueInt32(connection, (ESteamNetworkingConfigValue)config, value);
}
bool Steam::setConnectionConfigValueFloat(uint32 connection, int config, float value){
	if(SteamNetworkingUtils() == NULL){
		return false;
	}
	return SteamNetworkingUtils()->SetConnectionConfigValueFloat(connection, (ESteamNetworkingConfigValue)config, value);
}

bool Steam::setConnectionConfigValueString(uint32 connection, int config, const String& value){
	if(SteamNetworkingUtils() == NULL){
		return false;
	}
	return SteamNetworkingUtils()->SetConnectionConfigValueString(connection, (ESteamNetworkingConfigValue)config, value.utf8().get_data());
}

//! A general purpose high resolution local timer with the following properties: Monotonicity is guaranteed. The initial value will be at least 24*3600*30*1e6, i.e. about 30 days worth of microseconds. In this way, the timestamp value of 0 will always be at least "30 days ago". Also, negative numbers will never be returned. Wraparound / overflow is not a practical concern.
uint64_t Steam::getLocalTimestamp(){
	if(SteamNetworkingUtils() == NULL){
		return 0;
	}
	return SteamNetworkingUtils()->GetLocalTimestamp();
}


/////////////////////////////////////////////////
///// PARENTAL SETTINGS
/////////////////////////////////////////////////
//
// There are no notes about these functions, names can assume functionality.
bool Steam::isParentalLockEnabled(){
	if(SteamParentalSettings() == NULL){
		return false;
	}
	return SteamParentalSettings()->BIsParentalLockEnabled();
}

bool Steam::isParentalLockLocked(){
	if(SteamParentalSettings() == NULL){
		return false;
	}
	return SteamParentalSettings()->BIsParentalLockLocked();
}

bool Steam::isAppBlocked(uint32 app_id){
	if(SteamParentalSettings() == NULL){
		return false;
	}
	return SteamParentalSettings()->BIsAppBlocked((AppId_t)app_id);
}

bool Steam::isAppInBlockList(uint32 app_id){
	if(SteamParentalSettings() == NULL){
		return false;
	}
	return SteamParentalSettings()->BIsAppInBlockList((AppId_t)app_id);
}

bool Steam::isFeatureBlocked(int feature){
	if(SteamParentalSettings() == NULL){
		return false;
	}
	return SteamParentalSettings()->BIsFeatureBlocked((EParentalFeature)feature);
}

bool Steam::isFeatureInBlockList(int feature){
	if(SteamParentalSettings() == NULL){
		return false;
	}
	return SteamParentalSettings()->BIsFeatureInBlockList((EParentalFeature)feature);
}


/////////////////////////////////////////////////
///// PARTIES
/////////////////////////////////////////////////
//
//! Get the list of locations in which you can post a party beacon.
Array Steam::getAvailableBeaconLocations(uint32 max){
	if(SteamParties() == NULL){
		return Array();
	}
	Array beaconLocations;
	uint32 locations = 0;
	SteamPartyBeaconLocation_t *beacons = new SteamPartyBeaconLocation_t[256];
	if(SteamParties()->GetNumAvailableBeaconLocations(&locations)){
		// If max is lower than locations, set it to locations
		if(max < locations){
			max = locations;
		}
		// Now get the beacon location list
		if(SteamParties()->GetAvailableBeaconLocations(beacons, max)){
			for(uint32 i = 0; i < max; i++){
				Dictionary beacon_data;
				beacon_data["type"] = beacons[i].m_eType;
				beacon_data["location_id"] = (uint64_t)beacons[i].m_ulLocationID;
				beaconLocations.append(beacon_data);
			}
		}
	}
	delete[] beacons;
	return beaconLocations;
}

//! Create a beacon. You can only create one beacon at a time. Steam will display the beacon in the specified location, and let up to unOpenSlots users "follow" the beacon to your party.
void Steam::createBeacon(uint32 open_slots, uint64_t location, int type, const String& connect_string, const String& metadata){
	if(SteamParties() != NULL){
		// Add data to the beacon location struct
		SteamPartyBeaconLocation_t *beacon_data = new SteamPartyBeaconLocation_t;
		beacon_data->m_eType = (ESteamPartyBeaconLocationType)type;
		beacon_data->m_ulLocationID = location;
		SteamAPICall_t api_call = SteamParties()->CreateBeacon(open_slots, beacon_data, connect_string.utf8().get_data(), metadata.utf8().get_data());
		callResultCreateBeacon.Set(api_call, this, &Steam::create_beacon);
		delete beacon_data;
	}
}

//! When a user follows your beacon, Steam will reserve one of the open party slots for them, and send your game a ReservationNotificationCallback_t callback. When that user joins your party, call OnReservationCompleted to notify Steam that the user has joined successfully.
void Steam::onReservationCompleted(uint64_t beacon_id, uint64_t steam_id){
	if(SteamParties() != NULL){
		CSteamID user_id = (uint64)steam_id;
		SteamParties()->OnReservationCompleted(beacon_id, user_id);
	}
}

//! To cancel a reservation (due to timeout or user input), call this. Steam will open a new reservation slot. Note: The user may already be in-flight to your game, so it's possible they will still connect and try to join your party.
void Steam::cancelReservation(uint64_t beacon_id, uint64_t steam_id){
	if(SteamParties() != NULL){
		CSteamID user_id = (uint64)steam_id;
		SteamParties()->CancelReservation(beacon_id, user_id);
	}
}

//! If a user joins your party through other matchmaking (perhaps a direct Steam friend, or your own matchmaking system), your game should reduce the number of open slots that Steam is managing through the party beacon. For example, if you created a beacon with five slots, and Steam sent you two ReservationNotificationCallback_t callbacks, and then a third user joined directly, you would want to call ChangeNumOpenSlots with a value of 2 for unOpenSlots. That value represents the total number of new users that you would like Steam to send to your party.
void Steam::changeNumOpenSlots(uint64_t beacon_id, uint32 open_slots){
	if(SteamParties() != NULL){
		SteamAPICall_t api_call = SteamParties()->ChangeNumOpenSlots(beacon_id, open_slots);
		callResultChangeNumOpenSlots.Set(api_call, this, &Steam::change_num_open_slots);
	}
}

//! Call this method to destroy the Steam party beacon. This will immediately cause Steam to stop showing the beacon in the target location. Note that any users currently in-flight may still arrive at your party expecting to join.
bool Steam::destroyBeacon(uint64_t beacon_id){
	if(SteamParties() == NULL){
		return false;
	}
	return SteamParties()->DestroyBeacon(beacon_id);
}

//! Get the number of active party beacons created by other users for your game, that are visible to the current user.
uint32 Steam::getNumActiveBeacons(){
	if(SteamParties() == NULL){
		return 0;
	}
	return SteamParties()->GetNumActiveBeacons();
}

//! Use with ISteamParties::GetNumActiveBeacons to iterate the active beacons visible to the current user. unIndex is a zero-based index, so iterate over the range [0, GetNumActiveBeacons() - 1]. The return is a PartyBeaconID_t that can be used with ISteamParties::GetBeaconDetails to get information about the beacons suitable for display to the user.
uint64_t Steam::getBeaconByIndex(uint32 index){
	if(SteamParties() == NULL){
		return 0;
	}
	return SteamParties()->GetBeaconByIndex(index);
}

//! Get details about the specified beacon. You can use the ISteamFriends API to get further details about pSteamIDBeaconOwner, and ISteamParties::GetBeaconLocationData to get further details about pLocation. The pchMetadata contents are specific to your game, and will be whatever was set (if anything) by the game process that created the beacon.
Dictionary Steam::getBeaconDetails(uint64_t beacon_id){
	Dictionary details;
	if(SteamParties() != NULL){
		CSteamID owner;
		SteamPartyBeaconLocation_t location;
		char metadata[STEAM_LARGE_BUFFER_SIZE];
		if(SteamParties()->GetBeaconDetails(beacon_id, &owner, &location, metadata, STEAM_LARGE_BUFFER_SIZE)){
			details["beacon_id"] = beacon_id;
			details["owner_id"] = (uint64_t)owner.ConvertToUint64();
			details["type"] = location.m_eType;
			details["location_id"] = (uint64_t)location.m_ulLocationID;
			details["metadata"] = metadata;
		}
	}
	return details;
}

//! When the user indicates they wish to join the party advertised by a given beacon, call this method. On success, Steam will reserve a slot for this user in the party and return the necessary "join game" string to use to complete the connection.
void Steam::joinParty(uint64_t beacon_id){
	if(SteamParties() != NULL){
		SteamAPICall_t api_call = SteamParties()->JoinParty(beacon_id);
		callResultJoinParty.Set(api_call, this, &Steam::join_party);
	}
}

//! Query general metadata for the given beacon location. For instance the Name, or the URL for an icon if the location type supports icons (for example, the icon for a Steam Chat Room Group).
String Steam::getBeaconLocationData(uint64_t location_id, int location_type, int location_data){
	String beacon_location_data = "";
	if(SteamParties() != NULL){
		char *beacon_data = new char[2048];
		// Add data to SteamPartyBeaconLocation struct
		SteamPartyBeaconLocation_t *beacon = new SteamPartyBeaconLocation_t;
		beacon->m_eType = (ESteamPartyBeaconLocationType)location_type;
		beacon->m_ulLocationID = location_id;
		if(SteamParties()->GetBeaconLocationData(*beacon, (ESteamPartyBeaconLocationData)location_data, (char*)beacon_data, 2048)){
			beacon_location_data = beacon_data;
		}
		delete[] beacon_data;
		delete beacon;
	}
	return beacon_location_data;
}


/////////////////////////////////////////////////
///// REMOTE PLAY 
/////////////////////////////////////////////////
//
//! Get the number of currently connected Steam Remote Play sessions.
uint32 Steam::getSessionCount(){
	if(SteamRemotePlay() == NULL){
		return 0;
	}
	return SteamRemotePlay()->GetSessionCount();
}

//! Get the currently connected Steam Remote Play session ID at the specified index.
uint32 Steam::getSessionID(uint32 index){
	if(SteamRemotePlay() == NULL){
		return 0;
	}
	return SteamRemotePlay()->GetSessionID(index);
}

//! Get the SteamID of the connected user.
uint64_t Steam::getSessionSteamID(uint32 session_id){
	if(SteamRemotePlay() == NULL){
		return 0;
	}
	CSteamID steam_id = SteamRemotePlay()->GetSessionSteamID(session_id);
	return steam_id.ConvertToUint64();
}

//! Get the name of the session client device.
String Steam::getSessionClientName(uint32 session_id){
	if(SteamRemotePlay() == NULL){
		return "";
	}
	return SteamRemotePlay()->GetSessionClientName(session_id);
}

//! Get the form factor of the session client device.
int Steam::getSessionClientFormFactor(uint32 session_id){
	if(SteamRemotePlay() == NULL){
		return 0;
	}
	return SteamRemotePlay()->GetSessionClientFormFactor(session_id);	
}

//! Get the resolution, in pixels, of the session client device. This is set to 0x0 if the resolution is not available.
Dictionary Steam::getSessionClientResolution(uint32 session_id){
	Dictionary resolution;
	if (SteamNetworking() == NULL) {
		return resolution;
	}
	int resolutionX = 0;
	int resolutionY = 0;
	bool success = SteamRemotePlay()->BGetSessionClientResolution(session_id, &resolutionX, &resolutionY);
	if(success){
		resolution["success"] = success;
		resolution["x"] = resolutionX;
		resolution["y"] = resolutionY;
	}	
	return resolution;
}

//! Invite a friend to join the game using Remote Play Together
bool Steam::sendRemotePlayTogetherInvite(uint64_t friend_id){
	if(SteamRemotePlay() == NULL){
		return false;
	}
	CSteamID steam_id = (uint64)friend_id;
	return SteamRemotePlay()->BSendRemotePlayTogetherInvite(steam_id);
}


/////////////////////////////////////////////////
///// REMOTE STORAGE
/////////////////////////////////////////////////
//
//! Delete a given file in Steam Cloud.
bool Steam::fileDelete(const String& file){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->FileDelete(file.utf8().get_data());
}

//! Check if a given file exists in Steam Cloud.
bool Steam::fileExists(const String& file){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->FileExists(file.utf8().get_data());
}

//! Delete file from remote storage but leave it on local disk to remain accessible.
bool Steam::fileForget(const String& file){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->FileForget(file.utf8().get_data());
}

//! Check if a given file is persisted in Steam Cloud.
bool Steam::filePersisted(const String& file){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->FilePersisted(file.utf8().get_data());
}

//! Read given file from Steam Cloud.
Dictionary Steam::fileRead(const String& file, int32_t data_to_read){
	Dictionary d;
	if(SteamRemoteStorage() == NULL){
		d["ret"] = false;
		return d;
	}
	PoolByteArray data;
	data.resize(data_to_read);
	d["ret"] = SteamRemoteStorage()->FileRead(file.utf8().get_data(), data.write().ptr(), data_to_read);
	d["buf"] = data;
	return d;
}

//! Starts an asynchronous read from a file. The offset and amount to read should be valid for the size of the file, as indicated by GetFileSize or GetFileTimestamp.
void Steam::fileReadAsync(const String& file, uint32 offset, uint32_t data_to_read){
	if(SteamRemoteStorage() != NULL){
		SteamAPICall_t api_call = SteamRemoteStorage()->FileReadAsync(file.utf8().get_data(), offset, data_to_read);
		callResultFileReadAsyncComplete.Set(api_call, this, &Steam::file_read_async_complete);
	}
}

//! Share a file.
void Steam::fileShare(const String& file){
	if(SteamRemoteStorage() != NULL){
		SteamAPICall_t api_call = SteamRemoteStorage()->FileShare(file.utf8().get_data());
		callResultFileShareResult.Set(api_call, this, &Steam::file_share_result);
	}
}

//! Write to given file from Steam Cloud.
bool Steam::fileWrite(const String& file, PoolByteArray data, int32 size){
	if(SteamRemoteStorage() != NULL){
		// Get the size from the poolbytearray, just in case
		int32 data_size = data.size();
		// If a size was passed, use that instead
		if(size > 0){
			data_size = size;
		}
		return SteamRemoteStorage()->FileWrite(file.utf8().get_data(), data.read().ptr(), data_size);
	}
	return false;
}

//! Creates a new file and asynchronously writes the raw byte data to the Steam Cloud, and then closes the file. If the target file already exists, it is overwritten.
void Steam::fileWriteAsync(const String& file, PoolByteArray data, int32 size){
	if(SteamRemoteStorage() != NULL){
		// Get the size from the PoolByteArray, just in case
		int32 data_size = data.size();
		// If a size was passed, use that instead
		if(size > 0){
			data_size = size;
		}
		SteamAPICall_t api_call = SteamRemoteStorage()->FileWriteAsync(file.utf8().get_data(), data.read().ptr(), data_size);
		callResultFileWriteAsyncComplete.Set(api_call, this, &Steam::file_write_async_complete);
	}
}

//! Cancels a file write stream that was started by FileWriteStreamOpen.  This trashes all of the data written and closes the write stream, but if there was an existing file with this name, it remains untouched.
bool Steam::fileWriteStreamCancel(uint64_t write_handle){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->FileWriteStreamCancel((UGCFileWriteStreamHandle_t)write_handle);
}

//! Closes a file write stream that was started by FileWriteStreamOpen. This flushes the stream to the disk, overwriting the existing file if there was one.
bool Steam::fileWriteStreamClose(uint64_t write_handle){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->FileWriteStreamClose((UGCFileWriteStreamHandle_t)write_handle);
}

//! Creates a new file output stream allowing you to stream out data to the Steam Cloud file in chunks. If the target file already exists, it is not overwritten until FileWriteStreamClose has been called. To write data out to this stream you can use FileWriteStreamWriteChunk, and then to close or cancel you use FileWriteStreamClose and FileWriteStreamCancel respectively.
uint64_t Steam::fileWriteStreamOpen(const String& file){
	if(SteamRemoteStorage() == NULL){
		return 0;
	}
	return SteamRemoteStorage()->FileWriteStreamOpen(file.utf8().get_data());
}

//! Writes a blob of data to the file write stream.
bool Steam::fileWriteStreamWriteChunk(uint64_t write_handle, PoolByteArray data){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->FileWriteStreamWriteChunk((UGCFileWriteStreamHandle_t)write_handle, data.read().ptr(), data.size());
}

//! Gets the number of cached UGC.
int32 Steam::getCachedUGCCount(){
	if(SteamRemoteStorage() == NULL){
		return 0;
	}
	return SteamRemoteStorage()->GetCachedUGCCount();
}

//! Gets the cached UGC's handle.
uint64_t Steam::getCachedUGCHandle(int32 content){
	if(SteamRemoteStorage() == NULL){
		return 0;
	}
	return SteamRemoteStorage()->GetCachedUGCHandle(content);
}

//! Gets the total number of local files synchronized by Steam Cloud.
int32_t Steam::getFileCount(){
	if(SteamRemoteStorage() == NULL){
		return 0;
	}
	return SteamRemoteStorage()->GetFileCount();
}

//! Gets the file name and size of a file from the index.
Dictionary Steam::getFileNameAndSize(int file){
	Dictionary d;
	String name = "";
	int32_t size = 0;
	if(SteamRemoteStorage() != NULL){
		name = String(SteamRemoteStorage()->GetFileNameAndSize(file, &size));
	}
	d["name"] = name;
	d["size"] = size;
	return d;
}

//! Get the size of a given file.
int32_t Steam::getFileSize(const String& file){
	if(SteamRemoteStorage() == NULL){
		return -1;
	}
	return SteamRemoteStorage()->GetFileSize(file.utf8().get_data());
}

//! Get the timestamp of when the file was uploaded/changed.
int64_t Steam::getFileTimestamp(const String& file){
	if(SteamRemoteStorage() == NULL){
		return -1;
	}
	return SteamRemoteStorage()->GetFileTimestamp(file.utf8().get_data());
}

//! Gets the number of bytes available, and used on the users Steam Cloud storage.
Dictionary Steam::getQuota(){
	Dictionary d;
	uint64_t total = 0;
	uint64_t available = 0;
	if(SteamRemoteStorage() != NULL){
		SteamRemoteStorage()->GetQuota((uint64*)&total, (uint64*)&available);
	}
	d["total_bytes"] = total;
	d["available_bytes"] = available;
	return d;
}

//! Obtains the platforms that the specified file will syncronize to.
Dictionary Steam::getSyncPlatforms(const String& file){
	// Set dictionary to populate
	Dictionary platforms;
	if(SteamRemoteStorage() == NULL){
		return platforms;
	}
	// Get the bitwise platform
	uint32_t platform = SteamRemoteStorage()->GetSyncPlatforms(file.utf8().get_data());
	// Assign this to bitwise in the dictionary
	platforms["bitwise"] = platform;
	// Now assign the verbose platform
	if(platform == 0){
		platforms["verbose"] = "none";
	}
	else if(platform == (1<<0)){
		platforms["verbose"] = "windows";
	}
	else if(platform == (1<<1)){
		platforms["verbose"] = "osx";
	}
	else if(platform == (1<<2)){
		platforms["verbose"] = "playstation 3";
	}
	else if(platform == (1<<3)){
		platforms["verbose"] = "linux / steamos";
	}
	else if(platform == (1<<4)){
		platforms["verbose"] = "reserved";
	}
	else{
		platforms["verbose"] = "all";
	}
	// Return the dictionary
	return platforms;
}

// Gets metadata for a file after it has been downloaded. This is the same metadata given in the RemoteStorageDownloadUGCResult_t call result.
Dictionary Steam::getUGCDetails(uint64_t content){
	// Create the return dictionary
	Dictionary ugc_details;
	if(SteamRemoteStorage() != NULL){
		AppId_t app_id;
		char *filename;
		int32 file_size;
		CSteamID steam_owner_id;
		if(SteamRemoteStorage()->GetUGCDetails((UGCHandle_t)content, &app_id, &filename, &file_size, &steam_owner_id)){
			ugc_details["handle"] = content;
			ugc_details["app_id"] = app_id;
			ugc_details["size"] = file_size;
			ugc_details["filename"] = filename;
			ugc_details["owner_id"] = (uint64_t)steam_owner_id.ConvertToUint64();
		}
	}
	return ugc_details;
}

// Gets the amount of data downloaded so far for a piece of content. pnBytesExpected can be 0 if function returns false or if the transfer hasn't started yet, so be careful to check for that before dividing to get a percentage.
Dictionary Steam::getUGCDownloadProgress(uint64_t content){
	// Create the return dictionary
	Dictionary ugc_download_progress;
	if(SteamRemoteStorage() != NULL){
		int32 bytes_downloaded;
		int32 bytes_expected;
		if(SteamRemoteStorage()->GetUGCDownloadProgress((UGCHandle_t)content, &bytes_downloaded, &bytes_expected)){
			ugc_download_progress["bytes_downloaded"] = bytes_downloaded;
			ugc_download_progress["bytes_expected"] = bytes_expected;
		}
	}
	return ugc_download_progress;
}

//! Is Steam Cloud enabled on the user's account?
bool Steam::isCloudEnabledForAccount(){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->IsCloudEnabledForAccount();
}

//! Is Steam Cloud enabled for this application?
bool Steam::isCloudEnabledForApp(){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->IsCloudEnabledForApp();
}

//! Set Steam Cloud enabled for this application.
void Steam::setCloudEnabledForApp(bool enabled){
	if(SteamRemoteStorage() != NULL){
		SteamRemoteStorage()->SetCloudEnabledForApp(enabled);
	}
}

//! Allows you to specify which operating systems a file will be synchronized to. Use this if you have a multiplatform game but have data which is incompatible between platforms.
bool Steam::setSyncPlatforms(const String& file, int platform){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->SetSyncPlatforms(file.utf8().get_data(), (ERemoteStoragePlatform)platform);
}

//! Downloads a UGC file.  A priority value of 0 will download the file immediately, otherwise it will wait to download the file until all downloads with a lower priority value are completed.  Downloads with equal priority will occur simultaneously.
void Steam::ugcDownload(uint64_t content, uint32 priority){
	if(SteamRemoteStorage() != NULL){
		SteamAPICall_t api_call = SteamRemoteStorage()->UGCDownload((UGCHandle_t)content, priority);
		callResultDownloadUGCResult.Set(api_call, this, &Steam::download_ugc_result);
	}
}

//! Downloads a UGC file to a specific location.
void Steam::ugcDownloadToLocation(uint64_t content, const String& location, uint32 priority){
	if(SteamRemoteStorage() != NULL){
		SteamAPICall_t api_call = SteamRemoteStorage()->UGCDownloadToLocation((UGCHandle_t)content, location.utf8().get_data(), priority);
		callResultDownloadUGCResult.Set(api_call, this, &Steam::download_ugc_result);
	}
}

//! After download, gets the content of the file. 
PoolByteArray Steam::ugcRead(uint64_t content, int32 data_size, uint32 offset, int action){
	PoolByteArray file_contents;
	file_contents.resize(data_size);
	if(SteamRemoteStorage() != NULL){
		SteamRemoteStorage()->UGCRead((UGCHandle_t)content, file_contents.write().ptr(), data_size, offset, (EUGCReadAction)action);
	}
	return file_contents;
}

//! Indicate to Steam the beginning / end of a set of local file operations - for example, writing a game save that requires updating two files.
bool Steam::beginFileWriteBatch(){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->BeginFileWriteBatch();
}

//! Indicate to Steam the beginning / end of a set of local file operations - for example, writing a game save that requires updating two files.
bool Steam::endFileWriteBatch(){
	if(SteamRemoteStorage() == NULL){
		return false;
	}
	return SteamRemoteStorage()->EndFileWriteBatch();	
}

//! Cloud dynamic state change notification, used to get the total number of files changed; paired with getLocalFileChange
uint32_t Steam::getLocalFileChangeCount(){
	if(SteamRemoteStorage() == NULL){
		return 0;
	}
	return SteamRemoteStorage()->GetLocalFileChangeCount();
}

//! Cloud dynamic state change notification, for iteration with getLocalFileChangeCount
Dictionary Steam::getLocalFileChange(int file){
	Dictionary file_change;
	if(SteamRemoteStorage() != NULL){
		ERemoteStorageLocalFileChange change_type;
		ERemoteStorageFilePathType file_path_type;
		String changed_file = SteamRemoteStorage()->GetLocalFileChange(file, &change_type, &file_path_type);
		// Add these to the dictionary
		file_change["file"] = changed_file;
		file_change["change_type"] = change_type;
		file_change["path_type"] = file_path_type;
	}
	return file_change;
}


/////////////////////////////////////////////////
///// SCREENSHOTS
/////////////////////////////////////////////////
//
//! Adds a screenshot to the user's Steam screenshot library from disk.
uint32_t Steam::addScreenshotToLibrary(const String& filename, const String& thumbnail_filename, int width, int height){
	if(SteamScreenshots() == NULL){
		return 0;
	}
	return SteamScreenshots()->AddScreenshotToLibrary(filename.utf8().get_data(), thumbnail_filename.utf8().get_data(), width, height);
}

//! Adds a VR screenshot to the user's Steam screenshot library from disk in the supported type.
uint32_t Steam::addVRScreenshotToLibrary(int type, const String& filename, const String& vr_filename){
	if(SteamScreenshots() == NULL){
		return 0;
	}
	return SteamScreenshots()->AddVRScreenshotToLibrary((EVRScreenshotType)type, filename.utf8().get_data(), vr_filename.utf8().get_data());
}

//! Toggles whether the overlay handles screenshots.
void Steam::hookScreenshots(bool hook){
	if(SteamScreenshots() != NULL){
		SteamScreenshots()->HookScreenshots(hook);
	}
}

//! Checks if the app is hooking screenshots.
bool Steam::isScreenshotsHooked(){
	if(SteamScreenshots() == NULL){
		return false;
	}
	return SteamScreenshots()->IsScreenshotsHooked();
}

//! Sets optional metadata about a screenshot's location.
bool Steam::setLocation(uint32_t screenshot, const String& location){
	if(SteamScreenshots() == NULL){
		return false;
	}
	ScreenshotHandle handle = (ScreenshotHandle)screenshot;
	return SteamScreenshots()->SetLocation(handle, location.utf8().get_data());
}

//! Tags a published file as being visible in the screenshot.
bool Steam::tagPublishedFile(uint32_t screenshot, uint64_t file_id){
	if(SteamScreenshots() == NULL){
		return false;
	}
	PublishedFileId_t file = (uint64)file_id;
	return SteamScreenshots()->TagPublishedFile((ScreenshotHandle)screenshot, file);
}

//! Tags a Steam user as being visible in the screenshot.  You can tag up to the value declared by k_nScreenshotMaxTaggedUsers in a single screenshot. Tagging more users than that will just be discarded.  This function has a built in delay before saving the tag which allows you to call it repeatedly for each item.  You can get the handle to tag the screenshot once it has been successfully saved from the ScreenshotReady_t callback or via the WriteScreenshot, AddScreenshotToLibrary, AddVRScreenshotToLibrary calls.
bool Steam::tagUser(uint32 screenshot, uint64_t steam_id){
	if(SteamScreenshots() == NULL){
		return false;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamScreenshots()->TagUser((ScreenshotHandle)screenshot, user_id);
}

//! Causes Steam overlay to take a screenshot.
void Steam::triggerScreenshot(){
	if(SteamScreenshots() != NULL){
		SteamScreenshots()->TriggerScreenshot();
	}
}

//! Writes a screenshot to the user's Steam screenshot library.
uint32_t Steam::writeScreenshot(const PoolByteArray& rgb, int width, int height){
	if(SteamScreenshots() == NULL){
		return 0;
	}
	return SteamScreenshots()->WriteScreenshot((void*)rgb.read().ptr(), rgb.size(), width, height);
}


/////////////////////////////////////////////////
///// UGC
/////////////////////////////////////////////////
//
//! Adds a dependency between the given item and the appid. This list of dependencies can be retrieved by calling GetAppDependencies.
//! This is a soft-dependency that is displayed on the web. It is up to the application to determine whether the item can actually be used or not.
void Steam::addAppDependency(uint64_t published_file_id, uint32_t app_id){
	if(SteamUGC() != NULL){
		PublishedFileId_t file_id = (uint64_t)published_file_id;
		AppId_t app = (uint32_t)app_id;
		SteamAPICall_t api_call = SteamUGC()->AddAppDependency(file_id, app);
		callResultAddAppDependency.Set(api_call, this, &Steam::add_app_dependency_result);
	}
}

//! Adds a workshop item as a dependency to the specified item. If the nParentPublishedFileID item is of type k_EWorkshopFileTypeCollection, than the nChildPublishedFileID is simply added to that collection.
//! Otherwise, the dependency is a soft one that is displayed on the web and can be retrieved via the ISteamUGC API using a combination of the m_unNumChildren member variable of the SteamUGCDetails_t struct and GetQueryUGCChildren.
void Steam::addDependency(uint64_t published_file_id, uint64_t child_published_file_id){
	if(SteamUGC() != NULL){
		PublishedFileId_t parent = (uint64_t)published_file_id;
		PublishedFileId_t child = (uint64_t)child_published_file_id;
		SteamAPICall_t api_call = SteamUGC()->AddDependency(parent, child);
		callResultAddUGCDependency.Set(api_call, this, &Steam::add_ugc_dependency_result);
	}
}

//! Adds a excluded tag to a pending UGC Query. This will only return UGC without the specified tag.
bool Steam::addExcludedTag(uint64_t query_handle, const String& tag_name){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->AddExcludedTag(handle, tag_name.utf8().get_data());
}

//! Adds a key-value tag pair to an item. Keys can map to multiple different values (1-to-many relationship).
bool Steam::addItemKeyValueTag(uint64_t update_handle, const String& key, const String& value){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = (uint64_t)update_handle;
	return SteamUGC()->AddItemKeyValueTag(handle, key.utf8().get_data(), value.utf8().get_data());
}

//! Adds an additional preview file for the item.
bool Steam::addItemPreviewFile(uint64_t query_handle, const String& preview_file, int type){
	if(SteamUGC() == NULL){
		return false;
	}
	EItemPreviewType previewType;
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	if(type == 0){
		previewType = k_EItemPreviewType_Image;
	}
	else if(type == 1){
		previewType = k_EItemPreviewType_YouTubeVideo;
	}
	else if(type == 2){
		previewType = k_EItemPreviewType_Sketchfab;
	}
	else if(type == 3){
		previewType = k_EItemPreviewType_EnvironmentMap_HorizontalCross;
	}
	else if(type == 4){
		previewType = k_EItemPreviewType_EnvironmentMap_LatLong;
	}
	else{
		previewType = k_EItemPreviewType_ReservedMax;
	}
	return SteamUGC()->AddItemPreviewFile(handle, preview_file.utf8().get_data(), previewType);
}

//! Adds an additional video preview from YouTube for the item.
bool Steam::addItemPreviewVideo(uint64_t query_handle, const String& video_id){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->AddItemPreviewVideo(handle, video_id.utf8().get_data());
}

//! Adds a workshop item to the users favorites list.
void Steam::addItemToFavorites(uint32_t app_id, uint64_t published_file_id){
	if(SteamUGC() != NULL){
		AppId_t app = (uint32_t)app_id;
		PublishedFileId_t file_id = (uint64_t)published_file_id;
		SteamAPICall_t api_call = SteamUGC()->AddItemToFavorites(app, file_id);
		callResultFavoriteItemListChanged.Set(api_call, this, &Steam::user_favorite_items_list_changed);
	}
}

//! Adds a required key-value tag to a pending UGC Query. This will only return workshop items that have a key = pKey and a value = pValue.
bool Steam::addRequiredKeyValueTag(uint64_t query_handle, const String& key, const String& value){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->AddRequiredKeyValueTag(handle, key.utf8().get_data(), value.utf8().get_data());
}

//! Adds a required tag to a pending UGC Query. This will only return UGC with the specified tag.
bool Steam::addRequiredTag(uint64_t query_handle, const String& tag_name){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->AddRequiredTag(handle, tag_name.utf8().get_data());
}

//! Adds the requirement that the returned items from the pending UGC Query have at least one of the tags in the given set (logical "or"). For each tag group that is added, at least one tag from each group is required to be on the matching items.
bool Steam::addRequiredTagGroup(uint64_t query_handle, Array tag_array){
	bool added_tag_group = false;
	if(SteamUGC() != NULL){
		UGCQueryHandle_t handle = uint64(query_handle);
		SteamParamStringArray_t *tags = new SteamParamStringArray_t();
		tags->m_ppStrings = new const char*[tag_array.size()];
		uint32 strCount = tag_array.size();
		for (uint32 i=0; i < strCount; i++) {
			String str = (String)tag_array[i];
			tags->m_ppStrings[i] = str.utf8().get_data();
		}
		tags->m_nNumStrings = tag_array.size();
		added_tag_group = SteamUGC()->AddRequiredTagGroup(handle, tags);
		delete tags;
	}
	return added_tag_group;
}

//! Lets game servers set a specific workshop folder before issuing any UGC commands.
bool Steam::initWorkshopForGameServer(uint32_t workshop_depot_id){
	bool initialized_workshop = false;
	if(SteamUGC() != NULL){
		DepotId_t workshop = (uint32_t)workshop_depot_id;
		const char *folder = new char[256];
		initialized_workshop = SteamUGC()->BInitWorkshopForGameServer(workshop, (char*)folder);
		delete[] folder;
	}
	return initialized_workshop;
}

//! Creates a new workshop item with no content attached yet.
void Steam::createItem(uint32 app_id, int file_type){
	if(SteamUGC() != NULL){
		SteamAPICall_t api_call = SteamUGC()->CreateItem((AppId_t)app_id, (EWorkshopFileType)file_type);
		callResultItemCreate.Set(api_call, this, &Steam::item_created);
	}
}

//! Query for all matching UGC. You can use this to list all of the available UGC for your app.
uint64_t Steam::createQueryAllUGCRequest(int query_type, int matching_type, uint32_t creator_id, uint32_t consumer_id, uint32 page){
	if(SteamUGC() == NULL){
		return 0;
	}
	EUGCQuery query;
	if(query_type == 0){
		query = k_EUGCQuery_RankedByVote;
	}
	else if(query_type == 1){
		query = k_EUGCQuery_RankedByPublicationDate;
	}
	else if(query_type == 2){
		query = k_EUGCQuery_AcceptedForGameRankedByAcceptanceDate;
	}
	else if(query_type == 3){
		query = k_EUGCQuery_RankedByTrend;
	}
	else if(query_type == 4){
		query = k_EUGCQuery_FavoritedByFriendsRankedByPublicationDate;
	}
	else if(query_type == 5){
		query = k_EUGCQuery_CreatedByFriendsRankedByPublicationDate;
	}
	else if(query_type == 6){
		query = k_EUGCQuery_RankedByNumTimesReported;
	}
	else if(query_type == 7){
		query = k_EUGCQuery_CreatedByFollowedUsersRankedByPublicationDate;
	}
	else if(query_type == 8){
		query = k_EUGCQuery_NotYetRated;
	}
	else if(query_type == 9){
		query = k_EUGCQuery_RankedByTotalVotesAsc;
	}
	else if(query_type == 10){
		query = k_EUGCQuery_RankedByVotesUp;
	}
	else if(query_type == 11){
		query = k_EUGCQuery_RankedByTextSearch;
	}
	else if(query_type == 12){
		query = k_EUGCQuery_RankedByTotalUniqueSubscriptions;
	}
	else if(query_type == 13){
		query = k_EUGCQuery_RankedByPlaytimeTrend;
	}
	else if(query_type == 14){
		query = k_EUGCQuery_RankedByTotalPlaytime;
	}
	else if(query_type == 15){
		query = k_EUGCQuery_RankedByAveragePlaytimeTrend;
	}
	else if(query_type == 16){
		query = k_EUGCQuery_RankedByLifetimeAveragePlaytime;
	}
	else if(query_type == 17){
		query = k_EUGCQuery_RankedByPlaytimeSessionsTrend;
	}
	else{
		query = k_EUGCQuery_RankedByLifetimePlaytimeSessions;
	}
	EUGCMatchingUGCType match;
	if(matching_type == 0){
		match = k_EUGCMatchingUGCType_All;
	}
	else if(matching_type == 1){
		match = k_EUGCMatchingUGCType_Items_Mtx;
	}
	else if(matching_type == 2){
		match = k_EUGCMatchingUGCType_Items_ReadyToUse;
	}
	else if(matching_type == 3){
		match = k_EUGCMatchingUGCType_Collections;
	}
	else if(matching_type == 4){
		match = k_EUGCMatchingUGCType_Artwork;
	}
	else if(matching_type == 5){
		match = k_EUGCMatchingUGCType_Videos;
	}
	else if(matching_type == 6){
		match = k_EUGCMatchingUGCType_Screenshots;
	}
	else if(matching_type == 7){
		match = k_EUGCMatchingUGCType_AllGuides;
	}
	else if(matching_type == 8){
		match = k_EUGCMatchingUGCType_WebGuides;
	}
	else if(matching_type == 9){
		match = k_EUGCMatchingUGCType_IntegratedGuides;
	}
	else if(matching_type == 10){
		match = k_EUGCMatchingUGCType_UsableInGame;
	}
	else if(matching_type == 11){
		match = k_EUGCMatchingUGCType_ControllerBindings;
	}
	else{
		match = k_EUGCMatchingUGCType_GameManagedItems;
	}
	AppId_t creator = (uint32_t)creator_id;
	AppId_t consumer = (uint32_t)consumer_id;
	UGCQueryHandle_t handle = SteamUGC()->CreateQueryAllUGCRequest(query, match, creator, consumer, page);
	return (uint64_t)handle;
}

//! Query for the details of specific workshop items.
uint64_t Steam::createQueryUGCDetailsRequest(Array published_file_ids){
	uint64_t this_handle = 0;
	if(SteamUGC() != NULL){
		uint32 fileCount = published_file_ids.size();
		if(fileCount != 0){
			PublishedFileId_t *file_ids = new PublishedFileId_t[fileCount];
			for(uint32 i = 0; i < fileCount; i++){
				file_ids[i] = (uint64_t)published_file_ids[i];
			}
			UGCQueryHandle_t handle = SteamUGC()->CreateQueryUGCDetailsRequest(file_ids, fileCount);
			delete[] file_ids;
			this_handle = (uint64_t)handle;
		}
	}
	return this_handle;
}

//! Query UGC associated with a user. You can use this to list the UGC the user is subscribed to amongst other things.
uint64_t Steam::createQueryUserUGCRequest(uint64_t steam_id, int list_type, int matching_ugc_type, int sort_order, uint32_t creator_id, uint32_t consumer_id, uint32 page){
	if(SteamUGC() == NULL){
		return 0;
	}
	// Get tue universe ID from the Steam ID
	CSteamID user_id = (uint64)steam_id;
	AccountID_t account = user_id.ConvertToUint64();
	EUserUGCList list;
	if(list_type == 0){
		list = k_EUserUGCList_Published;
	}
	else if(list_type == 1){
		list = k_EUserUGCList_VotedOn;
	}
	else if(list_type == 2){
		list = k_EUserUGCList_VotedUp;
	}
	else if(list_type == 3){
		list = k_EUserUGCList_VotedDown;
	}
	else if(list_type == 4){
		list = k_EUserUGCList_WillVoteLater;
	}
	else if(list_type == 5){
		list = k_EUserUGCList_Favorited;
	}
	else if(list_type == 6){
		list = k_EUserUGCList_Subscribed;
	}
	else if(list_type == 7){
		list = k_EUserUGCList_UsedOrPlayed;
	}
	else{
		list = k_EUserUGCList_Followed;
	}
	EUGCMatchingUGCType match;
	if(matching_ugc_type == 0){
		match = k_EUGCMatchingUGCType_All;
	}
	else if(matching_ugc_type == 1){
		match = k_EUGCMatchingUGCType_Items_Mtx;
	}
	else if(matching_ugc_type == 2){
		match = k_EUGCMatchingUGCType_Items_ReadyToUse;
	}
	else if(matching_ugc_type == 3){
		match = k_EUGCMatchingUGCType_Collections;
	}
	else if(matching_ugc_type == 4){
		match = k_EUGCMatchingUGCType_Artwork;
	}
	else if(matching_ugc_type == 5){
		match = k_EUGCMatchingUGCType_Videos;
	}
	else if(matching_ugc_type == 6){
		match = k_EUGCMatchingUGCType_Screenshots;
	}
	else if(matching_ugc_type == 7){
		match = k_EUGCMatchingUGCType_AllGuides;
	}
	else if(matching_ugc_type == 8){
		match = k_EUGCMatchingUGCType_WebGuides;
	}
	else if(matching_ugc_type == 9){
		match = k_EUGCMatchingUGCType_IntegratedGuides;
	}
	else if(matching_ugc_type == 10){
		match = k_EUGCMatchingUGCType_UsableInGame;
	}
	else if(matching_ugc_type == 11){
		match = k_EUGCMatchingUGCType_ControllerBindings;
	}
	else{
		match = k_EUGCMatchingUGCType_GameManagedItems;
	}
	EUserUGCListSortOrder sort;
	if(sort_order == 0){
		sort = k_EUserUGCListSortOrder_CreationOrderDesc;
	}
	else if(sort_order == 1){
		sort = k_EUserUGCListSortOrder_CreationOrderAsc;
	}
	else if(sort_order == 2){
		sort = k_EUserUGCListSortOrder_TitleAsc;
	}
	else if(sort_order == 3){
		sort = k_EUserUGCListSortOrder_LastUpdatedDesc;
	}
	else if(sort_order == 4){
		sort = k_EUserUGCListSortOrder_SubscriptionDateDesc;
	}
	else if(sort_order == 5){
		sort = k_EUserUGCListSortOrder_VoteScoreDesc;
	}
	else{
		sort = k_EUserUGCListSortOrder_ForModeration;
	}
	AppId_t creator = (int)creator_id;
	AppId_t consumer = (int)consumer_id;
	UGCQueryHandle_t handle = SteamUGC()->CreateQueryUserUGCRequest(account, list, match, sort, creator, consumer, page);
	return (uint64_t)handle;
}

//! Deletes the item without prompting the user.
void Steam::deleteItem(uint64_t published_file_id){
	if(SteamUGC() != NULL){
		PublishedFileId_t file_id = (uint64_t)published_file_id;
		SteamAPICall_t api_call = SteamUGC()->DeleteItem(file_id);
		callResultDeleteItem.Set(api_call, this, &Steam::item_deleted);
	}
}

//! Download new or update already installed item. If returns true, wait for DownloadItemResult_t. If item is already installed, then files on disk should not be used until callback received.
//! If item is not subscribed to, it will be cached for some time. If bHighPriority is set, any other item download will be suspended and this item downloaded ASAP.
bool Steam::downloadItem(uint64_t published_file_id, bool high_priority){
	if(SteamUGC() == NULL){
		return false;
	}
	PublishedFileId_t file_id = (uint64_t)published_file_id;
	return SteamUGC()->DownloadItem(file_id, high_priority);
}

//! Get info about a pending download of a workshop item that has k_EItemStateNeedsUpdate set.
Dictionary Steam::getItemDownloadInfo(uint64_t published_file_id){
	Dictionary info;
	if(SteamUGC() == NULL){
		return info;
	}
	uint64 downloaded = 0;
	uint64 total = 0;
	info["ret"] = SteamUGC()->GetItemDownloadInfo((PublishedFileId_t)published_file_id, &downloaded, &total);
	if(info["ret"]){
		info["downloaded"] = uint64_t(downloaded);
		info["total"] = uint64_t(total);
	}
	return info;
}

//! Gets info about currently installed content on the disc for workshop items that have k_EItemStateInstalled set.
Dictionary Steam::getItemInstallInfo(uint64_t published_file_id){
	Dictionary info;
	if(SteamUGC() == NULL){
		info["ret"] = false;
		return info;
	}
	PublishedFileId_t file_id = (uint64_t)published_file_id;
	uint64 sizeOnDisk;
	char folder[1024] = { 0 };
	uint32 timeStamp;
	info["ret"] = SteamUGC()->GetItemInstallInfo((PublishedFileId_t)file_id, &sizeOnDisk, folder, sizeof(folder), &timeStamp);
	if(info["ret"]){
		info["size"] = (int)sizeOnDisk;
		info["folder"] = folder;
		info["foldersize"] = (uint32)sizeof(folder);
		info["timestamp"] = timeStamp;
	}
	return info;
}

//! Gets the current state of a workshop item on this client.
uint32 Steam::getItemState(uint64_t published_file_id){
	if(SteamUGC() == NULL){
		return 0;
	}
	PublishedFileId_t file_id = (uint64_t)published_file_id;
	return SteamUGC()->GetItemState(file_id);
}

//! Gets the progress of an item update.
Dictionary Steam::getItemUpdateProgress(uint64_t update_handle){
	Dictionary updateProgress;
	if(SteamUGC() == NULL){
		return updateProgress;
	}
	UGCUpdateHandle_t handle = (uint64_t)update_handle;
	uint64 processed = 0;
	uint64 total = 0;
	EItemUpdateStatus status = SteamUGC()->GetItemUpdateProgress(handle, &processed, &total);
	updateProgress["status"] = status;
	updateProgress["processed"] = uint64_t(processed);
	updateProgress["total"] = uint64_t(total);
	return updateProgress;
}

//! Gets the total number of items the current user is subscribed to for the game or application.
uint32 Steam::getNumSubscribedItems(){
	if(SteamUser() == NULL){
		return 0;
	}
	return SteamUGC()->GetNumSubscribedItems();
}

//! Retrieve the details of an additional preview associated with an individual workshop item after receiving a querying UGC call result.
Dictionary Steam::getQueryUGCAdditionalPreview(uint64_t query_handle, uint32 index, uint32 preview_index){
	Dictionary preview;
	if(SteamUGC() == NULL){
		return preview;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	char *url_or_video_id = new char[256];
	char *original_filename = new char[256];
	EItemPreviewType previewType;
	bool success = SteamUGC()->GetQueryUGCAdditionalPreview(handle, index, preview_index, (char*)url_or_video_id, 256, (char*)original_filename, 256, &previewType);
	if(success){
		preview["success"] = success;
		preview["handle"] = (uint64_t)handle;
		preview["index"] = index;
		preview["preview"] = preview_index;
		preview["urlOrVideo"] = url_or_video_id;
		preview["filename"] = original_filename;
		preview["type"] = previewType;
	}
	delete[] url_or_video_id;
	delete[] original_filename;
	return preview;
}

//! Retrieve the ids of any child items of an individual workshop item after receiving a querying UGC call result. These items can either be a part of a collection or some other dependency (see AddDependency).
Dictionary Steam::getQueryUGCChildren(uint64_t query_handle, uint32 index, uint32_t child_count){
	Dictionary children;
	if(SteamUGC() == NULL){
		return children;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	PoolVector<uint64_t> vec;
	vec.resize(child_count);
	bool success = SteamUGC()->GetQueryUGCChildren(handle, index, (PublishedFileId_t*)vec.write().ptr(), child_count);
	if(success) {
		Array godot_arr;
		godot_arr.resize(child_count);
		for (uint32_t i = 0; i < child_count; i++) {
			godot_arr[i] = vec[i];
		}
		
		children["success"] = success;
		children["handle"] = (uint64_t)handle;
		children["index"] = index;
		children["children"] = godot_arr;
	}
	return children;
}

//! Retrieve the details of a key-value tag associated with an individual workshop item after receiving a querying UGC call result.
Dictionary Steam::getQueryUGCKeyValueTag(uint64_t query_handle, uint32 index, uint32 key_value_tag_index){
	Dictionary tag;
	if(SteamUGC() == NULL){
		return tag;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	char *key = new char[256];
	char *value = new char[256];
	bool success = SteamUGC()->GetQueryUGCKeyValueTag(handle, index, key_value_tag_index, (char*)key, 256, (char*)value, 256);
	if(success){
		tag["success"] = success;
		tag["handle"] = (uint64_t)handle;
		tag["index"] = index;
		tag["tag"] = key_value_tag_index;
		tag["key"] = key;
		tag["value"] = value;
	}
	delete[] key;
	delete[] value;
	return tag;
}

//! Retrieve the developer set metadata of an individual workshop item after receiving a querying UGC call result.
String Steam::getQueryUGCMetadata(uint64_t query_handle, uint32 index){
	String query_ugc_metadata = "";
	if(SteamUGC() != NULL){
		UGCQueryHandle_t handle = (uint64_t)query_handle;
		char *metadata = new char[5000];
		bool success = SteamUGC()->GetQueryUGCMetadata(handle, index, (char*)metadata, 5000);
		if(success){
			query_ugc_metadata = metadata;
		}
		delete[] metadata;
	}
	return query_ugc_metadata;
}

//! Retrieve the number of additional previews of an individual workshop item after receiving a querying UGC call result.
uint32 Steam::getQueryUGCNumAdditionalPreviews(uint64_t query_handle, uint32 index){
	if(SteamUser() == NULL){
		return 0;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->GetQueryUGCNumAdditionalPreviews(handle, index);
}

//! Retrieve the number of key-value tags of an individual workshop item after receiving a querying UGC call result.
uint32 Steam::getQueryUGCNumKeyValueTags(uint64_t query_handle, uint32 index){
	if(SteamUser() == NULL){
		return 0;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->GetQueryUGCNumKeyValueTags(handle, index);
}

//! Retrieve the number of tags for an individual workshop item after receiving a querying UGC call result. You should call this in a loop to get the details of all the workshop items returned.
uint32 Steam::getQueryUGCNumTags(uint64_t query_handle, uint32 index){
	if(SteamUGC() == NULL){
		return 0;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->GetQueryUGCNumTags(handle, index);
}

//! Retrieve the URL to the preview image of an individual workshop item after receiving a querying UGC call result.
String Steam::getQueryUGCPreviewURL(uint64_t query_handle, uint32 index){
	String query_ugc_preview_url = "";
	if(SteamUGC() != NULL){
		UGCQueryHandle_t handle = (uint64_t)query_handle;
		char *url = new char[256];
		bool success = SteamUGC()->GetQueryUGCPreviewURL(handle, index, (char*)url, 256);
		if(success){
			query_ugc_preview_url = url;
		}
		delete[] url;
	}
	return query_ugc_preview_url;
}

//! Retrieve the details of an individual workshop item after receiving a querying UGC call result.
Dictionary Steam::getQueryUGCResult(uint64_t query_handle, uint32 index){
	Dictionary ugcResult;
	if(SteamUGC() == NULL){
		return ugcResult;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	SteamUGCDetails_t pDetails;
	bool success = SteamUGC()->GetQueryUGCResult(handle, index, &pDetails);
	if(success){
		ugcResult["result"] = (uint64_t)pDetails.m_eResult;
		ugcResult["file_id"] = (uint64_t)pDetails.m_nPublishedFileId;
		ugcResult["file_type"] = (uint64_t)pDetails.m_eFileType;
		ugcResult["creator_app_id"] = (uint32_t)pDetails.m_nCreatorAppID;
		ugcResult["consumer_app_id"] = (uint32_t)pDetails.m_nConsumerAppID;
		ugcResult["title"] = String::utf8(pDetails.m_rgchTitle);
		ugcResult["description"] = String::utf8(pDetails.m_rgchDescription);
		ugcResult["steam_id_owner"] = (uint64_t)pDetails.m_ulSteamIDOwner;
		ugcResult["time_created"] = pDetails.m_rtimeCreated;
		ugcResult["time_updated"] = pDetails.m_rtimeUpdated;
		ugcResult["time_added_to_user_list"] = pDetails.m_rtimeAddedToUserList;
		ugcResult["visibility"] = (uint64_t)pDetails.m_eVisibility;
		ugcResult["banned"] = pDetails.m_bBanned;
		ugcResult["accepted_for_use"] = pDetails.m_bAcceptedForUse;
		ugcResult["tags_truncated"] = pDetails.m_bTagsTruncated;
		ugcResult["tags"] = pDetails.m_rgchTags;
		ugcResult["handle_file"] = (uint64_t)pDetails.m_hFile;
		ugcResult["handle_preview_file"] = (uint64_t)pDetails.m_hPreviewFile;
		ugcResult["file_name"] = pDetails.m_pchFileName;
		ugcResult["file_size"] = pDetails.m_nFileSize;
		ugcResult["preview_file_size"] = pDetails.m_nPreviewFileSize;
		ugcResult["url"] = pDetails.m_rgchURL;
		ugcResult["votes_up"] = pDetails.m_unVotesUp;
		ugcResult["votes_down"] = pDetails.m_unVotesDown;
		ugcResult["score"] = pDetails.m_flScore;
		ugcResult["num_children"] = pDetails.m_unNumChildren;
	}
	return ugcResult;
}

//! Retrieve various statistics of an individual workshop item after receiving a querying UGC call result.
Dictionary Steam::getQueryUGCStatistic(uint64_t query_handle, uint32 index, int stat_type){
	Dictionary ugcStat;
	if(SteamUGC() == NULL){
		return ugcStat;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	EItemStatistic type;
		if(stat_type == 0){
		type = k_EItemStatistic_NumSubscriptions;
	}
	else if(stat_type == 1){
		type = k_EItemStatistic_NumFavorites;
	}
	else if(stat_type == 2){
		type = k_EItemStatistic_NumFollowers;
	}
	else if(stat_type == 3){
		type = k_EItemStatistic_NumUniqueSubscriptions;
	}
	else if(stat_type == 4){
		type = k_EItemStatistic_NumUniqueFavorites;
	}
	else if(stat_type == 5){
		type = k_EItemStatistic_NumUniqueFollowers;
	}
	else if(stat_type == 6){
		type = k_EItemStatistic_NumUniqueWebsiteViews;
	}
	else if(stat_type == 7){
		type = k_EItemStatistic_ReportScore;
	}
	else if(stat_type == 8){
		type = k_EItemStatistic_NumSecondsPlayed;
	}
	else if(stat_type == 9){
		type = k_EItemStatistic_NumPlaytimeSessions;
	}
	else if(stat_type == 10){
		type = k_EItemStatistic_NumComments;
	}
	else if(stat_type == 11){
		type = k_EItemStatistic_NumSecondsPlayedDuringTimePeriod;
	}
	else{
		type = k_EItemStatistic_NumPlaytimeSessionsDuringTimePeriod;
	}
	uint64 value = 0;
	bool success = SteamUGC()->GetQueryUGCStatistic(handle, index, type, &value);
	if(success){
		ugcStat["success"] = success;
		ugcStat["handle"] = (uint64_t)handle;
		ugcStat["index"] = index;
		ugcStat["type"] = type;
		ugcStat["value"] = (uint64_t)value;
	}	
	return ugcStat;
}

//! Retrieve the "nth" tag associated with an individual workshop item after receiving a querying UGC call result.
//! You should call this in a loop to get the details of all the workshop items returned.
String Steam::getQueryUGCTag(uint64_t query_handle, uint32 index, uint32 tag_index){
	if(SteamUGC() == NULL){
		return "";
	}
	// Set a default tag to return
	char *tag = new char[64];
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	SteamUGC()->GetQueryUGCTag(handle, index, tag_index, tag, 64);
	tag[63] = '\0';
	String tag_name = tag;
	delete[] tag;
	return tag_name;
}

//! Retrieve the "nth" display string (usually localized) for a tag, which is associated with an individual workshop item after receiving a querying UGC call result.
//! You should call this in a loop to get the details of all the workshop items returned.
String Steam::getQueryUGCTagDisplayName(uint64_t query_handle, uint32 index, uint32 tag_index){
	if(SteamUGC() == NULL){
		return "";
	}
	// Set a default tag name to return
	char *tag = new char[256];
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	SteamUGC()->GetQueryUGCTagDisplayName(handle, index, tag_index, tag, 256);
	tag[255] = '\0';
	String tagDisplay = tag;
	delete[] tag;
	return tagDisplay;
}

//! Gets a list of all of the items the current user is subscribed to for the current game.
Array Steam::getSubscribedItems(){
	if(SteamUGC() == NULL){
		return Array();
	}
	Array subscribed;
	uint32 num_items = SteamUGC()->GetNumSubscribedItems();
	PublishedFileId_t *items = new PublishedFileId_t[num_items];
	uint32 item_list = SteamUGC()->GetSubscribedItems(items, num_items);
	for(uint32 i = 0; i < item_list; i++){
		subscribed.append((uint64_t)items[i]);
	}
	delete[] items;
	return subscribed;
}

//! Gets the users vote status on a workshop item.
void Steam::getUserItemVote(uint64_t published_file_id){
	if(SteamUGC() != NULL){
		PublishedFileId_t file_id = (uint64_t)published_file_id;
		SteamAPICall_t api_call = SteamUGC()->GetUserItemVote(file_id);
		callResultGetUserItemVote.Set(api_call, this, &Steam::get_item_vote_result);
	}
}

//! Releases a UGC query handle when you are done with it to free up memory.
bool Steam::releaseQueryUGCRequest(uint64_t query_handle){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->ReleaseQueryUGCRequest(handle);
}

//! Removes the dependency between the given item and the appid. This list of dependencies can be retrieved by calling GetAppDependencies.
void Steam::removeAppDependency(uint64_t published_file_id, uint32_t app_id){
	if(SteamUGC() != NULL){
		PublishedFileId_t file_id = (uint64_t)published_file_id;
		AppId_t app = (uint32_t)app_id;
		SteamAPICall_t api_call = SteamUGC()->RemoveAppDependency(file_id, app);
		callResultRemoveAppDependency.Set(api_call, this, &Steam::remove_app_dependency_result);
	}
}

//! Removes a workshop item as a dependency from the specified item.
void Steam::removeDependency(uint64_t published_file_id, uint64_t child_published_file_id){
	if(SteamUGC() != NULL){
		PublishedFileId_t file_id = (uint64_t)published_file_id;
		PublishedFileId_t childID = (uint64_t)child_published_file_id;
		SteamAPICall_t api_call = SteamUGC()->RemoveDependency(file_id, childID);
		callResultRemoveUGCDependency.Set(api_call, this, &Steam::remove_ugc_dependency_result);
	}
}

//! Removes a workshop item from the users favorites list.
void Steam::removeItemFromFavorites(uint32_t app_id, uint64_t published_file_id){
	if(SteamUGC() != NULL){
		PublishedFileId_t file_id = (uint64_t)published_file_id;
		AppId_t app = (uint32_t)app_id;
		SteamAPICall_t api_call = SteamUGC()->RemoveItemFromFavorites(app, file_id);
		callResultFavoriteItemListChanged.Set(api_call, this, &Steam::user_favorite_items_list_changed);
	}
}

//! Removes an existing key value tag from an item.
bool Steam::removeItemKeyValueTags(uint64_t update_handle, const String& key){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->RemoveItemKeyValueTags(handle, key.utf8().get_data());
}

//! Removes an existing preview from an item.
bool Steam::removeItemPreview(uint64_t update_handle, uint32 index){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->RemoveItemPreview(handle, index);
}

//! Send a UGC query to Steam.
void Steam::sendQueryUGCRequest(uint64_t update_handle){
	if(SteamUGC() != NULL){
		UGCUpdateHandle_t handle = uint64(update_handle);
		SteamAPICall_t api_call = SteamUGC()->SendQueryUGCRequest(handle);
		callResultUGCQueryCompleted.Set(api_call, this, &Steam::ugc_query_completed);
	}
}

//! Sets whether results will be returned from the cache for the specific period of time on a pending UGC Query.
bool Steam::setAllowCachedResponse(uint64_t update_handle, uint32 max_age_seconds){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->SetAllowCachedResponse(handle, max_age_seconds);
}

//! Sets to only return items that have a specific filename on a pending UGC Query.
bool Steam::setCloudFileNameFilter(uint64_t update_handle, const String& match_cloud_filename){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->SetCloudFileNameFilter(handle, match_cloud_filename.utf8().get_data());
}

//! Sets the folder that will be stored as the content for an item.
bool Steam::setItemContent(uint64_t update_handle, const String& content_folder){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->SetItemContent(handle, content_folder.utf8().get_data());
}

//! Sets a new description for an item.
bool Steam::setItemDescription(uint64_t update_handle, const String& description){
	if(SteamUGC() == NULL){
		return false;
	}
	if ((uint32_t)description.length() > (uint32_t)k_cchPublishedDocumentDescriptionMax){
		printf("Description cannot have more than %d ASCII characters. Description not set.", k_cchPublishedDocumentDescriptionMax);
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->SetItemDescription(handle, description.utf8().get_data());
}

//! Sets arbitrary metadata for an item. This metadata can be returned from queries without having to download and install the actual content.
bool Steam::setItemMetadata(uint64_t update_handle, const String& metadata){
	if(SteamUGC() == NULL){
		return false;
	}
	if (metadata.utf8().length() > 5000){
		printf("Metadata cannot be more than %d bytes. Metadata not set.", 5000);
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->SetItemMetadata(handle, metadata.utf8().get_data());
}

//! Sets the primary preview image for the item.
bool Steam::setItemPreview(uint64_t update_handle, const String& preview_file){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->SetItemPreview(handle, preview_file.utf8().get_data());
}

//! Sets arbitrary developer specified tags on an item.
bool Steam::setItemTags(uint64_t update_handle, Array tag_array){
	bool tags_set = false;
	if(SteamUGC() != NULL){
		UGCUpdateHandle_t handle = uint64(update_handle);
		SteamParamStringArray_t *tags = new SteamParamStringArray_t();
		tags->m_ppStrings = new const char*[tag_array.size()];
		uint32 strCount = tag_array.size();
		for (uint32 i=0; i < strCount; i++) {
			String str = (String)tag_array[i];
			tags->m_ppStrings[i] = str.utf8().get_data();
		}
		tags->m_nNumStrings = tag_array.size();
		tags_set = SteamUGC()->SetItemTags(handle, tags);
		delete tags;
	}
	return tags_set;
}

//! Sets a new title for an item.
bool Steam::setItemTitle(uint64_t update_handle, const String& title){
	if(SteamUGC() == NULL){
		return false;
	}
	if (title.length() > 255){
		printf("Title cannot have more than %d ASCII characters. Title not set.", 255);
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->SetItemTitle(handle, title.utf8().get_data());
}

//! Sets the language of the title and description that will be set in this item update.
bool Steam::setItemUpdateLanguage(uint64_t update_handle, const String& language){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->SetItemUpdateLanguage(handle, language.utf8().get_data());
}

//! Sets the visibility of an item.
bool Steam::setItemVisibility(uint64_t update_handle, int visibility){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->SetItemVisibility(handle, (ERemoteStoragePublishedFileVisibility)visibility);
}

//! Sets the language to return the title and description in for the items on a pending UGC Query.
bool Steam::setLanguage(uint64_t query_handle, const String& language){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetLanguage(handle, language.utf8().get_data());
}

//! Sets whether workshop items will be returned if they have one or more matching tag, or if all tags need to match on a pending UGC Query.
bool Steam::setMatchAnyTag(uint64_t query_handle, bool match_any_tag){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetMatchAnyTag(handle, match_any_tag);
}

//! Sets whether the order of the results will be updated based on the rank of items over a number of days on a pending UGC Query.
bool Steam::setRankedByTrendDays(uint64_t query_handle, uint32 days){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetRankedByTrendDays(handle, days);
}

//! Sets whether to return any additional images/videos attached to the items on a pending UGC Query.
bool Steam::setReturnAdditionalPreviews(uint64_t query_handle, bool return_additional_previews){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetReturnAdditionalPreviews(handle, return_additional_previews);
}

//! Sets whether to return the IDs of the child items of the items on a pending UGC Query.
bool Steam::setReturnChildren(uint64_t query_handle, bool return_children){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetReturnChildren(handle, return_children);
}

//! Sets whether to return any key-value tags for the items on a pending UGC Query.
bool Steam::setReturnKeyValueTags(uint64_t query_handle, bool return_key_value_tags){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetReturnKeyValueTags(handle, return_key_value_tags);
}

//! Sets whether to return the full description for the items on a pending UGC Query.
bool Steam::setReturnLongDescription(uint64_t query_handle, bool return_long_description){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetReturnLongDescription(handle, return_long_description);
}

//! Sets whether to return the developer specified metadata for the items on a pending UGC Query.
bool Steam::setReturnMetadata(uint64_t query_handle, bool return_metadata){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetReturnMetadata(handle, return_metadata);
}

//! Sets whether to only return IDs instead of all the details on a pending UGC Query.
bool Steam::setReturnOnlyIDs(uint64_t query_handle, bool return_only_ids){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetReturnOnlyIDs(handle, return_only_ids);
}

//! Sets whether to return the the playtime stats on a pending UGC Query.
bool Steam::setReturnPlaytimeStats(uint64_t query_handle, uint32 days){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetReturnPlaytimeStats(handle, days);
}

//! Sets whether to only return the the total number of matching items on a pending UGC Query.
bool Steam::setReturnTotalOnly(uint64_t query_handle, bool return_total_only){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetReturnTotalOnly(handle, return_total_only);
}

//! Sets a string to that items need to match in either the title or the description on a pending UGC Query.
bool Steam::setSearchText(uint64_t query_handle, const String& search_text){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCQueryHandle_t handle = (uint64_t)query_handle;
	return SteamUGC()->SetSearchText(handle, search_text.utf8().get_data());
}

//! Allows the user to rate a workshop item up or down.
void Steam::setUserItemVote(uint64_t published_file_id, bool vote_up){
	if(SteamUGC() != NULL){
		PublishedFileId_t file_id = (uint64_t)published_file_id;
		SteamAPICall_t api_call = SteamUGC()->SetUserItemVote(file_id, vote_up);
		callResultSetUserItemVote.Set(api_call, this, &Steam::set_user_item_vote);
	}
}

//! Starts the item update process.
uint64_t Steam::startItemUpdate(uint32_t app_id, uint64_t published_file_id){
	if(SteamUGC() == NULL){
		return 0;
	}
	AppId_t app = (uint32_t)app_id;
	PublishedFileId_t file_id = (uint64_t)published_file_id;
	return SteamUGC()->StartItemUpdate(app, file_id);
}

//! Start tracking playtime on a set of workshop items.
void Steam::startPlaytimeTracking(Array published_file_ids){
	if(SteamUGC() != NULL){
		uint32 fileCount = published_file_ids.size();
		if(fileCount > 0){
			PublishedFileId_t *file_ids = new PublishedFileId_t[fileCount];
			for(uint32 i = 0; i < fileCount; i++){
				file_ids[i] = (uint64_t)published_file_ids[i];
			}
			SteamAPICall_t api_call = SteamUGC()->StartPlaytimeTracking(file_ids, fileCount);
			callResultStartPlaytimeTracking.Set(api_call, this, &Steam::start_playtime_tracking);
			delete[] file_ids;
		}
	}
}

//! Stop tracking playtime on a set of workshop items.
void Steam::stopPlaytimeTracking(Array published_file_ids){
	if(SteamUGC() != NULL){
		uint32 fileCount = published_file_ids.size();
		if(fileCount > 0){
			PublishedFileId_t *file_ids = new PublishedFileId_t[fileCount];
			Array files;
			for(uint32 i = 0; i < fileCount; i++){
				file_ids[i] = (uint64_t)published_file_ids[i];
			}
			SteamAPICall_t api_call = SteamUGC()->StopPlaytimeTracking(file_ids, fileCount);
			callResultStopPlaytimeTracking.Set(api_call, this, &Steam::stop_playtime_tracking);
			delete[] file_ids;
		}
	}
}

//! Stop tracking playtime of all workshop items.
void Steam::stopPlaytimeTrackingForAllItems(){
	if(SteamUGC() != NULL){
		SteamAPICall_t api_call = SteamUGC()->StopPlaytimeTrackingForAllItems();
		callResultStopPlaytimeTracking.Set(api_call, this, &Steam::stop_playtime_tracking);
	}
}

//! Returns any app dependencies that are associated with the given item.
void Steam::getAppDependencies(uint64_t published_file_id){
	if(SteamUGC() != NULL){
		PublishedFileId_t file_id = (uint64_t)published_file_id;
		SteamAPICall_t api_call = SteamUGC()->GetAppDependencies(file_id);
		callResultGetAppDependencies.Set(api_call, this, &Steam::get_app_dependencies_result);
	}
}

//! Uploads the changes made to an item to the Steam Workshop; to be called after setting your changes.
void Steam::submitItemUpdate(uint64_t update_handle, const String& change_note){
	if(SteamUGC() != NULL){
		UGCUpdateHandle_t handle = uint64(update_handle);
		SteamAPICall_t api_call = SteamUGC()->SubmitItemUpdate(handle, change_note.utf8().get_data());
		callResultItemUpdate.Set(api_call, this, &Steam::item_updated);
	}
}

//! Subscribe to a workshop item. It will be downloaded and installed as soon as possible.
void Steam::subscribeItem(uint64_t published_file_id){
	if(SteamUGC() != NULL){
		PublishedFileId_t file_id = (uint64_t)published_file_id;
		SteamAPICall_t api_call = SteamUGC()->SubscribeItem(file_id);
		callResultSubscribeItem.Set(api_call, this, &Steam::subscribe_item);
	}
}

//! SuspendDownloads( true ) will suspend all workshop downloads until SuspendDownloads( false ) is called or the game ends.
void Steam::suspendDownloads(bool suspend){
	if(SteamUGC() != NULL){
		SteamUGC()->SuspendDownloads(suspend);
	}
}

//! Unsubscribe from a workshop item. This will result in the item being removed after the game quits.
void Steam::unsubscribeItem(uint64_t published_file_id){
	if(SteamUGC() != NULL){
		PublishedFileId_t file_id = (uint64_t)published_file_id;
		SteamAPICall_t api_call = SteamUGC()->UnsubscribeItem(file_id);
		callResultUnsubscribeItem.Set(api_call, this, &Steam::unsubscribe_item);
	}
}

//! Updates an existing additional preview file for the item.
bool Steam::updateItemPreviewFile(uint64_t update_handle, uint32 index, const String& preview_file){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->UpdateItemPreviewFile(handle, index, preview_file.utf8().get_data());
}

//! Updates an additional video preview from YouTube for the item.
bool Steam::updateItemPreviewVideo(uint64_t update_handle, uint32 index, const String& video_id){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->UpdateItemPreviewVideo(handle, index, video_id.utf8().get_data());
}

//! Show the app's latest Workshop EULA to the user in an overlay window, where they can accept it or not.
bool Steam::showWorkshopEULA(){
	if(SteamUGC() == NULL){
		return false;
	}
	return SteamUGC()->ShowWorkshopEULA();
}

//! Retrieve information related to the user's acceptance or not of the app's specific Workshop EULA.
void Steam::getWorkshopEULAStatus(){
	if(SteamUGC() != NULL){
		SteamAPICall_t api_call = SteamUGC()->GetWorkshopEULAStatus();
		callResultWorkshopEULAStatus.Set(api_call, this, &Steam::workshop_eula_status);
	}
}

// Set the time range this item was created.
bool Steam::setTimeCreatedDateRange(uint64_t update_handle, uint32 start, uint32 end){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->SetTimeCreatedDateRange(handle, start, end);
}

// Set the time range this item was updated.
bool Steam::setTimeUpdatedDateRange(uint64_t update_handle, uint32 start, uint32 end){
	if(SteamUGC() == NULL){
		return false;
	}
	UGCUpdateHandle_t handle = uint64(update_handle);
	return SteamUGC()->SetTimeUpdatedDateRange(handle, start, end);
}


/////////////////////////////////////////////////
///// USERS
/////////////////////////////////////////////////
//
//! Set the rich presence data for an unsecured game server that the user is playing on. This allows friends to be able to view the game info and join your game.
void Steam::advertiseGame(const String& server_ip, int port){
	if(SteamUser() != NULL){
		// Resolve address and convert it from IP_Address struct to uint32_t
		IP_Address address;
		if(server_ip.is_valid_ip_address()){
			address = server_ip;
		}
		else{
			address = IP::get_singleton()->resolve_hostname(server_ip, IP::TYPE_IPV4);
		}
		// Resolution failed
		if(address.is_valid()){
			uint32_t ip4 = *((uint32_t *)address.get_ipv4());
			CSteamID gameserverID = SteamUser()->GetSteamID();
			SteamUser()->AdvertiseGame(gameserverID, ip4, port);
		}
	}
}

//! Authenticate the ticket from the entity Steam ID to be sure it is valid and isn't reused.
int Steam::beginAuthSession(PoolByteArray ticket, int ticket_size, uint64_t steam_id){
	if(SteamUser() == NULL){
		return -1;
	}
	CSteamID authSteamID = createSteamID(steam_id);
	return SteamUser()->BeginAuthSession(ticket.read().ptr(), ticket_size, authSteamID);
}

//! Cancels an auth ticket.
void Steam::cancelAuthTicket(uint32_t auth_ticket){
	if(SteamUser() != NULL){
		SteamUser()->CancelAuthTicket(auth_ticket);
	}
}

//! Decodes the compressed voice data returned by GetVoice.
Dictionary Steam::decompressVoice(const PoolByteArray& voice, uint32 voice_size, uint32 sample_rate){
	Dictionary decompressed;
	if(SteamUser() != NULL){
		uint32 written = 0;
		PoolByteArray outputBuffer;
		int result = SteamUser()->DecompressVoice(voice.read().ptr(), voice_size, &outputBuffer, outputBuffer.size(), &written, sample_rate);
		if(result == 0){
			decompressed["uncompressed"] = outputBuffer;
			decompressed["size"] = written;
		}
	}
	return decompressed;
}

//! Ends an auth session.
void Steam::endAuthSession(uint64_t steam_id){
	if(SteamUser() != NULL){
		CSteamID authSteamID = createSteamID(steam_id);
		SteamUser()->EndAuthSession(authSteamID);
	}
}

//! Get the authentication ticket data.
Dictionary Steam::getAuthSessionTicket(){
	// Create the dictionary to use
	Dictionary auth_ticket;
	if(SteamUser() != NULL){
		uint32_t ticket_size = 1024;
		PoolByteArray buffer;
		buffer.resize(ticket_size);
		uint32_t id = SteamUser()->GetAuthSessionTicket(buffer.write().ptr(), ticket_size, &ticket_size);
		// Add this data to the dictionary
		auth_ticket["id"] = id;
		auth_ticket["buffer"] = buffer;
		auth_ticket["size"] = ticket_size;
	}
	return auth_ticket;
}

//! Checks to see if there is captured audio data available from GetVoice, and gets the size of the data.
int Steam::getAvailableVoice(){
	if(SteamUser() == NULL){
		return 0;
	}
	uint32 bytesAvailable = 0;
	return SteamUser()->GetAvailableVoice(&bytesAvailable, NULL, 0);
}

//! Retrieves anti indulgence / duration control for current user / game combination.
void Steam::getDurationControl(){
	if(SteamUser() != NULL){
		SteamAPICall_t api_call = SteamUser()->GetDurationControl();
		callResultDurationControl.Set(api_call, this, &Steam::duration_control);
	}
}

//! Retrieve an encrypted ticket. This should be called after requesting an encrypted app ticket with RequestEncryptedAppTicket and receiving the EncryptedAppTicketResponse_t call result.
Dictionary Steam::getEncryptedAppTicket(){
	Dictionary encrypted;
	if(SteamUser() != NULL){
		uint32_t ticket_size = 1024;
		PoolByteArray buffer;
		buffer.resize(ticket_size);
		if(SteamUser()->GetEncryptedAppTicket(buffer.write().ptr(), ticket_size, &ticket_size)){
			encrypted["buffer"] = buffer;
			encrypted["size"] = ticket_size;
		}
	}
	return encrypted;
}

//! Trading Card badges data access, if you only have one set of cards, the series will be 1.
//! The user has can have two different badges for a series; the regular (max level 5) and the foil (max level 1).
int Steam::getGameBadgeLevel(int series, bool foil){
	if(SteamUser()== NULL){
		return 0;
	}
	return SteamUser()->GetGameBadgeLevel(series, foil);
}

//! Get the user's Steam level.
int Steam::getPlayerSteamLevel(){
	if(SteamUser() == NULL){
		return 0;
	}
	return SteamUser()->GetPlayerSteamLevel(); 
}

//! Get user's Steam ID.
uint64_t Steam::getSteamID(){
	if(SteamUser() == NULL){
		return 0;
	}
	CSteamID steam_id = SteamUser()->GetSteamID();
	return steam_id.ConvertToUint64();
}

//! Read captured audio data from the microphone buffer.
uint32 Steam::getVoice(){
	if(SteamUser() == NULL){
		return 0;
	}
	uint32 written = 0;
	uint8 buffer[1024];
	int response = SteamUser()->GetVoice(true, &buffer, 1024, &written, false, NULL, 0, NULL, 0);
	if(response == 1){
		return written;
	}
	return 0;
}

//! Gets the native sample rate of the Steam voice decoder.
uint32 Steam::getVoiceOptimalSampleRate(){
	if(SteamUser() == NULL){
		return 0;
	}
	return SteamUser()->GetVoiceOptimalSampleRate();
}

//! This starts the state machine for authenticating the game client with the game server. It is the client portion of a three-way handshake between the client, the game server, and the steam servers.
Dictionary Steam::initiateGameConnection(uint64_t server_id, uint32 server_ip, uint16 server_port, bool secure){
	Dictionary connection;
	if(SteamUser() != NULL){
		PoolByteArray auth;
		int authSize = 2048;
		auth.resize(authSize);
		CSteamID server = (uint64)server_id;
		if(SteamUser()->InitiateGameConnection_DEPRECATED(&auth, authSize, server, server_ip, server_port, secure) > 0){
			connection["auth_blob"] = auth;
			connection["server_id"] = server_id;
			connection["server_ip"] = server_ip;
			connection["server_port"] = server_port;
		}
	}
	return connection;
}

//! Checks if the current users looks like they are behind a NAT device.
bool Steam::isBehindNAT(){
	if(SteamUser() == NULL){
		return false;
	}
	return SteamUser()->BIsBehindNAT();
}

//! Checks whether the user's phone number is used to uniquely identify them.
bool Steam::isPhoneIdentifying(){
	if(SteamUser() == NULL){
		return false;
	}
	return SteamUser()->BIsPhoneIdentifying();
}

//! Checks whether the current user's phone number is awaiting (re)verification.
bool Steam::isPhoneRequiringVerification(){
	if(SteamUser() == NULL){
		return false;
	}
	return SteamUser()->BIsPhoneRequiringVerification();
}

//! Checks whether the current user has verified their phone number.
bool Steam::isPhoneVerified(){
	if(SteamUser() == NULL){
		return false;
	}
	return SteamUser()->BIsPhoneVerified();
}

//! Checks whether the current user has Steam Guard two factor authentication enabled on their account.
bool Steam::isTwoFactorEnabled(){
	if(SteamUser() == NULL){
		return false;
	}
	return SteamUser()->BIsTwoFactorEnabled();
}

//! Check, true/false, if user is logged into Steam currently.
bool Steam::loggedOn(){
	if(SteamUser() == NULL){
		return false;
	}
	return SteamUser()->BLoggedOn();
}

//! Requests an application ticket encrypted with the secret "encrypted app ticket key".
void Steam::requestEncryptedAppTicket(const String& secret){
	if(SteamUser() != NULL){
		SteamAPICall_t api_call = SteamUser()->RequestEncryptedAppTicket((void*)secret.utf8().get_data(), sizeof(&secret));
		callResultEncryptedAppTicketResponse.Set(api_call, this, &Steam::encrypted_app_ticket_response);
	}
}

//! Requests a URL which authenticates an in-game browser for store check-out, and then redirects to the specified URL.
void Steam::requestStoreAuthURL(const String& redirect){
	if(SteamUser() != NULL){
		SteamAPICall_t api_call = SteamUser()->RequestStoreAuthURL(redirect.utf8().get_data());
		callResultStoreAuthURLResponse.Set(api_call, this, &Steam::store_auth_url_response);
	}
}

//! Starts voice recording.
void Steam::startVoiceRecording(){
	if(SteamUser() != NULL){
		SteamUser()->StartVoiceRecording();
	}
}

//
bool Steam::setDurationControlOnlineState(int new_state){
	if(SteamUser() == NULL){
		return false;
	}
	return SteamUser()->BSetDurationControlOnlineState((EDurationControlOnlineState)new_state);
}

//! Stops voice recording.
void Steam::stopVoiceRecording(){
	if(SteamUser() != NULL){
		SteamUser()->StopVoiceRecording();
	}
}

//! Notify the game server that we are disconnecting. NOTE: This is part of the old user authentication API and should not be mixed with the new API.
void Steam::terminateGameConnection(uint32 server_ip, uint16 server_port){
	if(SteamUser() != NULL){
		SteamUser()->TerminateGameConnection_DEPRECATED(server_ip, server_port);
	}
}

//! Checks if the user owns a specific piece of Downloadable Content (DLC). This can only be called after sending the users auth ticket to ISteamGameServer::BeginAuthSession.
int Steam::userHasLicenseForApp(uint64_t steam_id, uint32_t app_id){
	if(SteamUser() == NULL){
		return 2;
	}
	CSteamID user_id = (uint64)steam_id;
	return SteamUser()->UserHasLicenseForApp(user_id, (AppId_t)app_id);
}


/////////////////////////////////////////////////
///// USER STATS
/////////////////////////////////////////////////
//
//! Attaches a piece of user generated content the current user's entry on a leaderboard.
void Steam::attachLeaderboardUGC(uint64_t ugc_handle, uint64_t this_leaderboard){
	if(SteamUserStats() != NULL){
		// If no leaderboard is passed, use internal one
		if(this_leaderboard == 0){
			this_leaderboard = leaderboard_handle;
		}
		SteamAPICall_t api_call = SteamUserStats()->AttachLeaderboardUGC((SteamLeaderboard_t)this_leaderboard, (UGCHandle_t)ugc_handle);
		callResultLeaderboardUGCSet.Set(api_call, this, &Steam::leaderboard_ugc_set);
	}
}

//! Clears a given achievement.
bool Steam::clearAchievement(const String& name){
	if(SteamUserStats() == NULL){
		return false;
	}
	return SteamUserStats()->ClearAchievement(name.utf8().get_data());
}

//! Request all rows for friends of user.
void Steam::downloadLeaderboardEntries(int start, int end, int type, uint64_t this_leaderboard){
	if(SteamUserStats() != NULL){
		// If no leaderboard is passed, use internal one
		if(this_leaderboard == 0){
			this_leaderboard = leaderboard_handle;
		}
		SteamAPICall_t api_call = SteamUserStats()->DownloadLeaderboardEntries((SteamLeaderboard_t)this_leaderboard, ELeaderboardDataRequest(type), start, end);
		callResultEntries.Set(api_call, this, &Steam::leaderboard_scores_downloaded);
	}
}

//! Request a maximum of 100 users with only one outstanding call at a time.
void Steam::downloadLeaderboardEntriesForUsers(Array users_id, uint64_t this_leaderboard){
	if(SteamUserStats() != NULL){
		// If no leaderboard is passed, use internal one
		if(this_leaderboard == 0){
			this_leaderboard = leaderboard_handle;
		}
		int usersCount = users_id.size();
		if(usersCount > 0){
			CSteamID *users = new CSteamID[usersCount];
			for(int i = 0; i < usersCount; i++){
				CSteamID user = createSteamID(users_id[i]);
				users[i] = user;
			}
			SteamAPICall_t api_call = SteamUserStats()->DownloadLeaderboardEntriesForUsers((SteamLeaderboard_t)this_leaderboard, users, usersCount);
			callResultEntries.Set(api_call, this, &Steam::leaderboard_scores_downloaded);
			delete[] users;
		}
	}
}

//! Find a given leaderboard, by name.
void Steam::findLeaderboard(const String& name){
	if(SteamUserStats() != NULL){
		SteamAPICall_t api_call = SteamUserStats()->FindLeaderboard(name.utf8().get_data());
		callResultFindLeaderboard.Set(api_call, this, &Steam::leaderboard_find_result);
	}
}

//! Gets a leaderboard by name, it will create it if it's not yet created.
void Steam::findOrCreateLeaderboard(const String& name, int sort_method, int display_type){
	if(SteamUserStats() != NULL){
		SteamAPICall_t api_call = SteamUserStats()->FindOrCreateLeaderboard(name.utf8().get_data(), (ELeaderboardSortMethod)sort_method, (ELeaderboardDisplayType)display_type);
		callResultFindLeaderboard.Set(api_call, this, &Steam::leaderboard_find_result);
	}
}

//! Return true/false if user has given achievement and the bool status of it being achieved or not.
Dictionary Steam::getAchievement(const String& name){
	Dictionary achieve;
	bool achieved = false;
	if(SteamUserStats() == NULL){
		achieve["ret"] = false;
	}
	else{
		achieve["ret"] = SteamUserStats()->GetAchievement(name.utf8().get_data(), &achieved);
	}
	achieve["achieved"] = achieved;
	return achieve;
}

//! Returns the percentage of users who have unlocked the specified achievement.
Dictionary Steam::getAchievementAchievedPercent(const String& name){
	Dictionary achieve;
	float percent = 0.f;
	if(SteamUserStats() == NULL){
		achieve["ret"] = false;
	} else {
		achieve["ret"] = SteamUserStats()->GetAchievementAchievedPercent(name.utf8().get_data(), &percent);
	}
	achieve["percent"] = percent;
	return achieve;
}

//! Get the achievement status, and the time it was unlocked if unlocked (in seconds since January 1, 19).
Dictionary Steam::getAchievementAndUnlockTime(const String& name){
	Dictionary achieve;
	if(SteamUserStats() == NULL){
		return achieve;
	}
	bool achieved = false;
	uint32 unlockTime = 0;
	// Get the data from Steam
	bool retrieved = SteamUserStats()->GetAchievementAndUnlockTime(name.utf8().get_data(), &achieved, &unlockTime);
	if(retrieved){
		achieve["retrieve"] = retrieved;
		achieve["achieved"] = achieved;
		achieve["unlocked"] = unlockTime;
	}
	return achieve;
}

//! Get general attributes for an achievement
String Steam::getAchievementDisplayAttribute(const String& name, const String& key){
	if(SteamUserStats() == NULL){
		return "";
	}
	return SteamUserStats()->GetAchievementDisplayAttribute(name.utf8().get_data(), key.utf8().get_data());
}

//! Gets the icon for an achievement
int Steam::getAchievementIcon(const String& name){
	if(SteamUserStats() == NULL){
		return 0;
	}
	return SteamUserStats()->GetAchievementIcon(name.utf8().get_data());
}

//! Gets the 'API name' for an achievement index
String Steam::getAchievementName(uint32_t achievement){
	if(SteamUserStats() == NULL){
		return "";
	}
	return SteamUserStats()->GetAchievementName(achievement);
}

//! For achievements that have related Progress stats, use this to query what the bounds of that progress are. You may want this info to selectively call IndicateAchievementProgress when appropriate milestones of progress have been made, to show a progress notification to the user.
Dictionary Steam::getAchievementProgressLimitsInt(const String& name){
	Dictionary progress;
	if(SteamUserStats() != NULL){
		int32 min = 0;
		int32 max = 0;
		if(SteamUserStats()->GetAchievementProgressLimits(name.utf8().get_data(), &min, &max)){
			progress["name"] = name;
			progress["min"] = min;
			progress["max"] = max;
		}
	}
	return progress;
}

//! For achievements that have related Progress stats, use this to query what the bounds of that progress are. You may want this info to selectively call IndicateAchievementProgress when appropriate milestones of progress have been made, to show a progress notification to the user.
Dictionary Steam::getAchievementProgressLimitsFloat(const String& name){
	Dictionary progress;
	if(SteamUserStats() != NULL){
		float min = 0.0;
		float max = 0.0;
		if(SteamUserStats()->GetAchievementProgressLimits(name.utf8().get_data(), &min, &max)){
			progress["name"] = name;
			progress["min"] = min;
			progress["max"] = max;
		}
	}
	return progress;
}

//! Gets the lifetime totals for an aggregated stat; as an int
uint64_t Steam::getGlobalStatInt(const String& name){
	if(SteamUserStats() == NULL){
		return 0;
	}
	int64 stat = 0;
	SteamUserStats()->GetGlobalStat(name.utf8().get_data(), &stat);
	return (uint64_t)stat;
}

//! Gets the lifetime totals for an aggregated stat; as an int
double Steam::getGlobalStatFloat(const String& name){
	if(SteamUserStats() == NULL){
		return 0;
	}
	double stat = 0;
	SteamUserStats()->GetGlobalStat(name.utf8().get_data(), &stat);
	return stat;
}

//! Gets the daily history for an aggregated stat; int.
uint64_t Steam::getGlobalStatIntHistory(const String& name){
	if(SteamUserStats() == NULL){
		return 0;
	}
	int64 history = 0;
	SteamUserStats()->GetGlobalStatHistory(name.utf8().get_data(), &history, 60);
	return (uint64_t)history;
}

//! Gets the daily history for an aggregated stat; float / double.
double Steam::getGlobalStatFloatHistory(const String& name){
	if(SteamUserStats() == NULL){
		return 0;
	}
	double history = 0;
	SteamUserStats()->GetGlobalStatHistory(name.utf8().get_data(), &history, 60);
	return history;
}

//! Returns the display type of a leaderboard handle.
Dictionary Steam::getLeaderboardDisplayType(uint64_t this_leaderboard){
	// Create a diciontary to return
	Dictionary display;
	if(SteamUserStats() != NULL){
		// If no leaderboard is passed, use internal one
		if(this_leaderboard == 0){
			this_leaderboard = leaderboard_handle;
		}
		int display_type = SteamUserStats()->GetLeaderboardDisplayType((SteamLeaderboard_t)this_leaderboard);
		// Add a verbal response
		if(display_type == 3){
			display["result"] = 3;
			display["verbal"] = "Display is time in milliseconds";
		}
		else if(display_type == 2){
			display["result"] = 2;
			display["verbal"] = "Display is time in seconds";
		}
		else if(display_type == 1){
			display["result"] = 1;
			display["verbal"] = "Display is simple numerical value";
		}
		else{
			display["result"] = 0;
			display["verbal"] = "Display type or leaderboard handle is invalid";
		}
	}
	return display;
}

//! Get the total number of entries in a leaderboard, as of the last request.
int Steam::getLeaderboardEntryCount(uint64_t this_leaderboard){
	if(SteamUserStats() == NULL){
		return -1;
	}
	// If no leaderboard is passed, use internal one
	if(this_leaderboard == 0){
		this_leaderboard = leaderboard_handle;
	}
	return SteamUserStats()->GetLeaderboardEntryCount((SteamLeaderboard_t)this_leaderboard);
}

//! Get the name of a leaderboard.
String Steam::getLeaderboardName(uint64_t this_leaderboard){
	if(SteamUserStats() == NULL){
		return "";
	}
	// If no leaderboard is passed, use internal one
	if(this_leaderboard == 0){
		this_leaderboard = leaderboard_handle;
	}
	return SteamUserStats()->GetLeaderboardName((SteamLeaderboard_t)this_leaderboard);
}

//! Returns the sort order of a leaderboard handle.
Dictionary Steam::getLeaderboardSortMethod(uint64_t this_leaderboard){
	// Create a dictionary to return
	Dictionary sort;
	if(SteamUserStats() != NULL){
		// If no leaderboard is passed, use internal one
		if(this_leaderboard == 0){
			this_leaderboard = leaderboard_handle;
		}
		// Get the sort method int	
		int sort_method = SteamUserStats()->GetLeaderboardSortMethod((SteamLeaderboard_t)this_leaderboard);
		// Add a verbal response
		if(sort_method == 2){
			sort["result"] = 2;
			sort["verbal"] = "Top score is highest number";
		}
		else if(sort_method == 1){
			sort["result"] = 1;
			sort["verbal"] = "Top score is lowest number";
		}
		else{
			sort["result"] = 0;
			sort["verbal"] = "Sort method or leaderboard handle is invalid";
		}
	}	
	return sort;
}

//! Gets the info on the most achieved achievement for the game.
Dictionary Steam::getMostAchievedAchievementInfo(){
	Dictionary entry;
	if(SteamUserStats() == NULL){
		return entry;
	}
	char *name = new char[64];
	float percent = 0;
	bool achieved = false;
	// Get the data from Steam
	int result = SteamUserStats()->GetMostAchievedAchievementInfo(name, 64, &percent, &achieved);
	if(result > -1){
		entry["rank"] = result;
		entry["name"] = name;
		entry["percent"] = percent;
		entry["achieved"] = achieved;
	}
	delete[] name;
	return entry;
}

//! Gets the info on the next most achieved achievement for the game.
Dictionary Steam::getNextMostAchievedAchievementInfo(int iterator){
	Dictionary entry;
	if(SteamUserStats() == NULL){
		return entry;
	}
	char *name = new char[64];
	float percent = 0;
	bool achieved = false;
	// Get the data from Steam
	int result = SteamUserStats()->GetNextMostAchievedAchievementInfo(iterator, name, 64, &percent, &achieved);
	if(result > -1){
		entry["rank"] = result;
		entry["name"] = name;
		entry["percent"] = percent;
		entry["achieved"] = achieved;
	}
	delete[] name;
	return entry;
}

//! Get the number of achievements.
uint32_t Steam::getNumAchievements(){
	if(SteamUserStats() == NULL){
		return 0;
	}
	return SteamUserStats()->GetNumAchievements();
}

//!  Get the amount of players currently playing the current game (online + offline).
void Steam::getNumberOfCurrentPlayers(){
	if(SteamUserStats() != NULL){
		SteamAPICall_t api_call = SteamUserStats()->GetNumberOfCurrentPlayers();
		callResultNumberOfCurrentPlayers.Set(api_call, this, &Steam::number_of_current_players);
	}
}

//! Get the value of a float statistic.
float Steam::getStatFloat(const String& name){
	if(SteamUserStats() == NULL){
		return 0;
	}
	float statValue = 0;
	SteamUserStats()->GetStat(name.utf8().get_data(), &statValue);
	return statValue;
}

//! Get the value of an integer statistic.
int Steam::getStatInt(const String& name){
	if(SteamUserStats() == NULL){
		return 0;
	}
	int32_t statValue = 0;
	SteamUserStats()->GetStat(name.utf8().get_data(), &statValue);
	return statValue;
}

//! Gets the unlock status of the Achievement.
Dictionary Steam::getUserAchievement(uint64_t steam_id, const String& name){
	Dictionary achieve;
	if(SteamUserStats() == NULL){
		return achieve;
	}
	bool achieved = false;
	CSteamID user_id = (uint64)steam_id;
	// Get the data from Steam
	bool success = SteamUserStats()->GetUserAchievement(user_id, name.utf8().get_data(), &achieved);
	if(success){
		achieve["steam_id"] = steam_id;
		achieve["retrieved"] = success;
		achieve["name"] = name;
		achieve["achieved"] = achieved;
	}
	return achieve;
}

//! Gets the achievement status, and the time it was unlocked if unlocked.
Dictionary Steam::getUserAchievementAndUnlockTime(uint64_t steam_id, const String& name){
	Dictionary achieve;
	if(SteamUserStats() == NULL){
		return achieve;
	}
	bool achieved = false;
	uint32 unlocked = 0;
	CSteamID user_id = (uint64)steam_id;
	// Get the data from Steam
	bool success = SteamUserStats()->GetUserAchievementAndUnlockTime(user_id, name.utf8().get_data(), &achieved, &unlocked);
	if(success){
		achieve["retrieved"] = success;
		achieve["name"] = name;
		achieve["achieved"] = achieved;
		achieve["unlocked"] = unlocked;
	}
	return achieve;
}

//! Gets the current value of a float stat for the specified user.
float Steam::getUserStatFloat(uint64_t steam_id, const String& name){
	if(SteamUserStats() == NULL){
		return 0;
	}
	float statValue = 0;
	CSteamID user_id = (uint64)steam_id;
	SteamUserStats()->GetUserStat(user_id, name.utf8().get_data(), &statValue);
	return statValue;
}

//! Gets the current value of an integer stat for the specified user.
int Steam::getUserStatInt(uint64_t steam_id, const String& name){
	if(SteamUserStats() == NULL){
		return 0;
	}
	int32_t statValue = 0;
	CSteamID user_id = (uint64)steam_id;
	SteamUserStats()->GetUserStat(user_id, name.utf8().get_data(), &statValue);
	return statValue;
}

//! Achievement progress, triggers an AchievementProgress callback, that is all.
//! Calling this with X out of X progress will NOT set the achievement, the game must still do that.
bool Steam::indicateAchievementProgress(const String& name, int current_progress, int max_progress){
	if(SteamUserStats() == NULL){
		return 0;
	}
	return SteamUserStats()->IndicateAchievementProgress(name.utf8().get_data(), current_progress, max_progress);
}

//! Request all statistics and achievements from Steam servers.
bool Steam::requestCurrentStats(){
	if(SteamUserStats() == NULL){
		return false;
	}
	// If the user isn't logged in, you can't get stats
	if(!loggedOn()){
		return false;
	}
	return SteamUserStats()->RequestCurrentStats();
}

//! Asynchronously fetch the data for the percentages.
void Steam::requestGlobalAchievementPercentages(){
	if(SteamUserStats() != NULL){
		SteamAPICall_t api_call = SteamUserStats()->RequestGlobalAchievementPercentages();
		callResultGlobalAchievementPercentagesReady.Set(api_call, this, &Steam::global_achievement_percentages_ready);
	}
}

//! Asynchronously fetches global stats data, which is available for stats marked as "aggregated" in the App Admin panel of the Steamworks website.  The limit is 60.
void Steam::requestGlobalStats(int history_days){
	if(SteamUserStats() != NULL){
		SteamAPICall_t api_call = SteamUserStats()->RequestGlobalStats(history_days);
		callResultGetGlobalStatsReceived.Set(api_call, this, &Steam::global_stats_received);
	}
}

//! Asynchronously downloads stats and achievements for the specified user from the server.
void Steam::requestUserStats(uint64_t steam_id){
	if(SteamUserStats() != NULL){
		CSteamID user_id = (uint64)steam_id;
		SteamAPICall_t api_call = SteamUserStats()->RequestUserStats(user_id);
		callResultUserStatsReceived.Set(api_call, this, &Steam::user_stats_received);
	}
}

//! Reset all Steam statistics; optional to reset achievements.
bool Steam::resetAllStats(bool achievements_too){
	if(SteamUserStats() == NULL){
		return false;
	}
	return SteamUserStats()->ResetAllStats(achievements_too);
}

//! Set a given achievement.
bool Steam::setAchievement(const String& name){
	if(SteamUserStats() == NULL){
		return false;
	}
	return SteamUserStats()->SetAchievement(name.utf8().get_data());
}

//! Set the maximum number of details to return for leaderboard entries
int Steam::setLeaderboardDetailsMax(int max){
	// If the user submitted too high of a maximum, set to the real max
	if(max > k_cLeaderboardDetailsMax){
		max = k_cLeaderboardDetailsMax;
	}
	// If the user submitted too low of a maximum, set it to 0
	if(max < 0){
		max = 0;
	}
	// Now set the internal variable
	leaderboard_details_max = max;
	return leaderboard_details_max;
}

//! Set a float statistic.
bool Steam::setStatFloat(const String& name, float value){
	if(SteamUserStats() == NULL){
		return false;
	}
	return SteamUserStats()->SetStat(name.utf8().get_data(), value);
}

//! Set an integer statistic.
bool Steam::setStatInt(const String& name, int value){
	if(SteamUserStats() == NULL){
		return false;
	}
	return SteamUserStats()->SetStat(name.utf8().get_data(), value);
}

//! Store all statistics, and achievements, on Steam servers; must be called to "pop" achievements.
bool Steam::storeStats(){
	if(SteamUserStats() == NULL){
		return 0;
	}
	return SteamUserStats()->StoreStats();
}

//! Updates an AVGRATE stat with new values.
bool Steam::updateAvgRateStat(const String& name, float this_session, double session_length){
	if(SteamUserStats() == NULL){
		return false;
	}
	return SteamUserStats()->UpdateAvgRateStat(name.utf8().get_data(), this_session, session_length);
}

//! Upload a leaderboard score for the user.
void Steam::uploadLeaderboardScore(int score, bool keep_best, PoolIntArray details, uint64_t this_leaderboard){
	if(SteamUserStats() != NULL){
		// If no leaderboard is passed, use internal one
		if(this_leaderboard == 0){
			this_leaderboard = leaderboard_handle;
		}
		ELeaderboardUploadScoreMethod method = keep_best ? k_ELeaderboardUploadScoreMethodKeepBest : k_ELeaderboardUploadScoreMethodForceUpdate;
		int detailsSize = details.size();
		const int32 *details_pointer = NULL;
		if(detailsSize > 0){
			PoolIntArray::Read r = details.read();
			details_pointer = r.ptr();
		}
		SteamAPICall_t api_call = SteamUserStats()->UploadLeaderboardScore((SteamLeaderboard_t)this_leaderboard, method, (int32)score, details_pointer, detailsSize);
		callResultUploadScore.Set(api_call, this, &Steam::leaderboard_score_uploaded);
	}
}

//! Get the currently used leaderboard entries.
Array Steam::getLeaderboardEntries(){
	if(SteamUserStats() == NULL){
		return Array();
	}
	return leaderboard_entries_array;
}


/////////////////////////////////////////////////
///// UTILS
/////////////////////////////////////////////////
//
//! Filters the provided input message and places the filtered result into pchOutFilteredText.
String Steam::filterText(int context, uint64_t steam_id, const String& message){
	String new_message = "";
	if(SteamUtils() != NULL){
		char *filtered = new char[2048];
		CSteamID source_id = (uint64)steam_id;
		SteamUtils()->FilterText((ETextFilteringContext)context, source_id, message.utf8().get_data(), filtered, strlen(filtered)+1);
		new_message = filtered;
		delete[] filtered;
	}
	return new_message;
}

//! Used to get the failure reason of a call result. The primary usage for this function is debugging. The failure reasons are typically out of your control and tend to not be very important. Just keep retrying your API Call until it works.
String Steam::getAPICallFailureReason(){
	if(SteamUtils() == NULL){
		return "ERROR: Steam Utils not present.";
	}
	int failure = SteamUtils()->GetAPICallFailureReason(api_handle);
	// Parse the failure
	if(failure == k_ESteamAPICallFailureSteamGone){
		return "The local Steam process has stopped responding, it may have been forcefully closed or is frozen.";
	}
	else if(failure == k_ESteamAPICallFailureNetworkFailure){
		return "The network connection to the Steam servers has been lost, or was already broken.";
	}
	else if(failure == k_ESteamAPICallFailureInvalidHandle){
		return "The SteamAPICall_t handle passed in no longer exists.";
	}
	else if(failure == k_ESteamAPICallFailureMismatchedCallback){
		return "GetAPICallResult was called with the wrong callback type for this API call.";
	}
	else{
		return "No failure.";
	}
}

//! Get the Steam ID of the running application/game.
uint32_t Steam::getAppID(){
	if(SteamUtils() == NULL){
		return 0;
	}
	return SteamUtils()->GetAppID();
}

//! Get the amount of battery power, clearly for laptops.
int Steam::getCurrentBatteryPower(){
	if(SteamUtils() == NULL){
		return 0;
	}
	return SteamUtils()->GetCurrentBatteryPower();
}

//! Gets the image bytes from an image handle.
Dictionary Steam::getImageRGBA(int image){
	Dictionary d;
	bool success = false;
	if(SteamUtils() != NULL){
		uint32 width;
		uint32 height;
		success = SteamUtils()->GetImageSize(image, &width, &height);
		if(success){
			PoolByteArray data;
			data.resize(width * height * 4);
			success = SteamUtils()->GetImageRGBA(image, data.write().ptr(), data.size());
			if(success){
				d["buffer"] = data;
			}
		}
	}
	d["success"] = success;
	return d;
}

//! Gets the size of a Steam image handle.
Dictionary Steam::getImageSize(int image){
	Dictionary d;
	bool success = false;
	if(SteamUtils() != NULL){
		uint32 width;
		uint32 height;
		success = SteamUtils()->GetImageSize(image, &width, &height);
		d["width"] = width;
		d["height"] = height;
	}
	d["success"] = success;
	return d;
}

//! Returns the number of IPC calls made since the last time this function was called.
uint32 Steam::getIPCCallCount(){
	if(SteamUtils() == NULL){
		return 0;
	}
	return SteamUtils()->GetIPCCallCount();
}

//! Get the user's country by IP.
String Steam::getIPCountry(){
	if(SteamUtils() == NULL){
		return "";
	}
	return SteamUtils()->GetIPCountry();
}

//! Return amount of time, in seconds, user has spent in this session.
int Steam::getSecondsSinceAppActive(){
	if(SteamUtils() == NULL){
		return 0;
	}
	return SteamUtils()->GetSecondsSinceAppActive();
}

//! Returns the number of seconds since the user last moved the mouse.
int Steam::getSecondsSinceComputerActive(){
	if(SteamUtils() == NULL){
		return 0;
	}
	return SteamUtils()->GetSecondsSinceComputerActive();
}

//! Get the actual time.
int Steam::getServerRealTime(){
	if(SteamUtils() == NULL){
		return 0;
	}
	return SteamUtils()->GetServerRealTime();
}

//! Get the Steam user interface language.
String Steam::getSteamUILanguage(){
	if(SteamUtils() == NULL){
		return "";
	}
	return SteamUtils()->GetSteamUILanguage();
}

//! Initializes text filtering. Returns false if filtering is unavailable for the language the user is currently running in. If the language is unsupported, the FilterText API will act as a passthrough.
bool Steam::initFilterText(){
	if(SteamUtils() == NULL){
		return false;
	}
	return SteamUtils()->InitFilterText();
}

//! Checks if an API Call is completed. Provides the backend of the CallResult wrapper.
Dictionary Steam::isAPICallCompleted(){
	Dictionary completed;
	if(SteamUtils() != NULL){
		bool failed = false;
		bool valid = SteamUtils()->IsAPICallCompleted(api_handle, &failed);
		// Populate the dictionary
		completed["completed"] = valid;
		completed["failed"] = failed;
	}
	return completed;
}

//! Returns true/false if Steam overlay is enabled.
bool Steam::isOverlayEnabled(){
	if(SteamUtils() == NULL){
		return false;
	}
	return SteamUtils()->IsOverlayEnabled();
}

//! Returns whether the current launcher is a Steam China launcher. You can cause the client to behave as the Steam China launcher by adding -dev -steamchina to the command line when running Steam.
bool Steam::isSteamChinaLauncher(){
	if(SteamUtils() == NULL){
		return false;
	}
	return SteamUtils()->IsSteamChinaLauncher();
}

//! Returns true if currently running on the Steam Deck device
bool Steam::isSteamRunningOnSteamDeck(){
	if(SteamUtils() == NULL){
		return false;
	}
	return SteamUtils()->IsSteamRunningOnSteamDeck();
}

//! Returns true if Steam & the Steam Overlay are running in Big Picture mode.
bool Steam::isSteamInBigPictureMode(){
	if(SteamUtils() == NULL){
		return false;
	}
	return SteamUtils()->IsSteamInBigPictureMode();
}

//! Is Steam running in VR?
bool Steam::isSteamRunningInVR(){
	if(SteamUtils() == NULL){
		return 0;
	}
	return SteamUtils()->IsSteamRunningInVR();
}

//! Checks if the HMD view will be streamed via Steam In-Home Streaming.
bool Steam::isVRHeadsetStreamingEnabled(){
	if(SteamUtils() == NULL){
		return false;
	}
	return SteamUtils()->IsVRHeadsetStreamingEnabled();	
}

//! Checks if the Overlay needs a present. Only required if using event driven render updates.
bool Steam::overlayNeedsPresent(){
	if(SteamUtils() == NULL){
		return false;
	}
	return SteamUtils()->BOverlayNeedsPresent();
}

//! Sets the inset of the overlay notification from the corner specified by SetOverlayNotificationPosition.
void Steam::setOverlayNotificationInset(int horizontal, int vertical){
	if(SteamUtils() != NULL){
		SteamUtils()->SetOverlayNotificationInset(horizontal, vertical);
	}
}

//! Set the position where overlay shows notifications.
void Steam::setOverlayNotificationPosition(int pos){
	if((pos >= 0) && (pos < 4) && (SteamUtils() != NULL)){
		SteamUtils()->SetOverlayNotificationPosition(ENotificationPosition(pos));
	}
}

//! Set whether the HMD content will be streamed via Steam In-Home Streaming.
void Steam::setVRHeadsetStreamingEnabled(bool enabled){
	if(SteamUtils() != NULL){
		SteamUtils()->SetVRHeadsetStreamingEnabled(enabled);
	}
}

//! Activates the Big Picture text input dialog which only supports gamepad input.
bool Steam::showGamepadTextInput(int input_mode, int line_input_mode, const String& description, uint32 max_text, const String& preset_text){
	if(SteamUtils() == NULL){
		return false;
	}
	// Convert modes
	EGamepadTextInputMode mode;
	if(input_mode == 0){
		mode = k_EGamepadTextInputModeNormal;
	}
	else{
		mode = k_EGamepadTextInputModePassword;
	}
	EGamepadTextInputLineMode lineMode;
	if(line_input_mode == 0){
		lineMode = k_EGamepadTextInputLineModeSingleLine;
	}
	else{
		lineMode = k_EGamepadTextInputLineModeMultipleLines;
	}
	return SteamUtils()->ShowGamepadTextInput(mode, lineMode, description.utf8().get_data(), max_text, preset_text.utf8().get_data());
}

//! Opens a floating keyboard over the game content and sends OS keyboard keys directly to the game.
//! The text field position is specified in pixels relative the origin of the game window and is used to position the floating keyboard in a way that doesn't cover the text field
bool Steam::showFloatingGamepadTextInput(int input_mode, int text_field_x_position, int text_field_y_position, int text_field_width, int text_field_height) {
	if(SteamUtils() == NULL){
		return false;
	}
	return SteamUtils()->ShowFloatingGamepadTextInput((EFloatingGamepadTextInputMode)input_mode, text_field_x_position, text_field_y_position, text_field_width, text_field_height);
}

//! In game launchers that don't have controller support you can call this to have Steam Input translate the controller input into mouse/kb to navigate the launcher
void Steam::setGameLauncherMode(bool mode) {
	if(SteamUtils() == NULL){
		return;
	}
	SteamUtils()->SetGameLauncherMode(mode);
}

//! Ask SteamUI to create and render its OpenVR dashboard.
void Steam::startVRDashboard(){
	if(SteamUtils() != NULL){
		SteamUtils()->StartVRDashboard();
	}
}

// Dismisses the floating keyboard.
bool Steam::dismissFloatingGamepadTextInput(){
	if(SteamUtils() == NULL){
		return false;
	}
	return SteamUtils()->DismissFloatingGamepadTextInput();
}


/////////////////////////////////////////////////
///// VIDEO
/////////////////////////////////////////////////
//
//! Get the OPF details for 360 video playback.
void Steam::getOPFSettings(uint32_t app_id){
	if(SteamVideo() != NULL){
		SteamVideo()->GetOPFSettings((AppId_t)app_id);
	}
}

//! Gets the OPF string for the specified video App ID.
String Steam::getOPFStringForApp(uint32_t app_id){
	String opf_string = "";
	if(SteamVideo() != NULL){
		int32 size = 48000;
		char *buffer = new char[size];
		if(SteamVideo()->GetOPFStringForApp((AppId_t)app_id, buffer, &size)){
			opf_string = buffer;
		}
		delete[] buffer;
	}
	return opf_string;
}

//! Asynchronously gets the URL suitable for streaming the video associated with the specified video app ID.
void Steam::getVideoURL(uint32_t app_id){
	if(SteamVideo() != NULL){
		SteamVideo()->GetVideoURL((AppId_t)app_id);
	}
}

//! Checks if the user is currently live broadcasting and gets the number of users.
Dictionary Steam::isBroadcasting(){
	Dictionary broadcast;
	if(SteamVideo() != NULL){
		int viewers = 0;
		bool broadcasting = SteamVideo()->IsBroadcasting(&viewers);
		// Populate the dictionary
		broadcast["broadcasting"] = broadcasting;
		broadcast["viewers"] = viewers;
	}
	return broadcast;
}


/////////////////////////////////////////////////
///// SIGNALS / CALLBACKS
/////////////////////////////////////////////////
//
// APPS CALLBACKS ///////////////////////////////
//
//! Triggered after the current user gains ownership of DLC and that DLC is installed.
void Steam::dlc_installed(DlcInstalled_t* call_data){
	uint32_t app_id = (AppId_t)call_data->m_nAppID;
	emit_signal("dlc_installed", app_id);
}

//! Called after requesting the details of a specific file.
void Steam::file_details_result(FileDetailsResult_t* file_data){
	uint32_t result = file_data->m_eResult;
	uint64_t file_size = file_data->m_ulFileSize;
	uint32_t flags = file_data->m_unFlags;
	uint8 *file_hash = new uint8[20];
	file_hash = file_data->m_FileSHA;
	emit_signal("file_details_result", result, file_size, file_hash, flags);
	delete[] file_hash;
}

//! Posted after the user executes a steam url with command line or query parameters such as steam://run/<appid>//?param1=value1;param2=value2;param3=value3; while the game is already running. The new params can be queried with getLaunchCommandLine and getLaunchQueryParam.
void Steam::new_launch_url_parameters(NewUrlLaunchParameters_t* call_data){
	emit_signal("new_launch_url_parameters");
}

//! Purpose: called for games in Timed Trial mode
void Steam::timed_trial_status(TimedTrialStatus_t* call_data){
	uint32_t app_id = call_data->m_unAppID;
	bool is_offline = call_data->m_bIsOffline;
	uint32 seconds_allowed = call_data->m_unSecondsAllowed;
	uint32 seconds_played = call_data->m_unSecondsPlayed;
	emit_signal("timed_trial_status", app_id, is_offline, seconds_allowed, seconds_played);
}

// APPS LIST CALLBACKS ///////////////////////////
//
//! Sent when a new app is installed.
void Steam::app_installed(SteamAppInstalled_t* call_data){
	uint32_t app_id = (uint32_t)call_data->m_nAppID;
	uint32_t install_folder_index = call_data->m_iInstallFolderIndex;
	emit_signal("app_installed", app_id, install_folder_index);
}

//! Sent when an app is uninstalled.
void Steam::app_uninstalled(SteamAppUninstalled_t* call_data){
	uint32_t app_id = (uint32_t)call_data->m_nAppID;
	uint32_t install_folder_index = call_data->m_iInstallFolderIndex;
	emit_signal("app_uninstalled", app_id, install_folder_index);	
}

// FRIENDS CALLBACKS ////////////////////////////
//
//! Called when a large avatar is loaded if you have tried requesting it when it was unavailable.
void Steam::avatar_loaded(AvatarImageLoaded_t* avatarData){
	uint32 width, height;
	bool success = SteamUtils()->GetImageSize(avatarData->m_iImage, &width, &height);
	if(!success){
		printf("[Steam] Failed to get image size.\n");
		return;
	}
	PoolByteArray data;
	data.resize(width * height * 4);
	success = SteamUtils()->GetImageRGBA(avatarData->m_iImage, data.write().ptr(), data.size());
	if(!success){
		printf("[Steam] Failed to load image buffer from callback\n");
		return;
	}
	CSteamID steam_id = avatarData->m_steamID;
	uint64_t avatar_id = steam_id.ConvertToUint64();
	call_deferred("emit_signal", "avatar_loaded", avatar_id, width, data);
}

//! Called when a Steam group activity has received.
void Steam::clan_activity_downloaded(DownloadClanActivityCountsResult_t* call_data){
	bool success = call_data->m_bSuccess;
	// Set up the dictionary to populate
	Dictionary activity;
	if(success){
		int online = 0;
		int in_game = 0;
		int chatting = 0;
		activity["ret"] = SteamFriends()->GetClanActivityCounts(clan_activity, &online, &in_game, &chatting);
		if(activity["ret"]){
			activity["online"] = online;
			activity["ingame"] = in_game;
			activity["chatting"] = chatting;
		}
	}
	emit_signal("clan_activity_downloaded", activity);
}

//! Called when Rich Presence data has been updated for a user, this can happen automatically when friends in the same game update their rich presence, or after a call to requestFriendRichPresence.
void Steam::friend_rich_presence_update(FriendRichPresenceUpdate_t* call_data){
	uint64_t steam_id = call_data->m_steamIDFriend.ConvertToUint64();
	AppId_t app_id = call_data->m_nAppID;
	emit_signal("friend_rich_presence_updated", steam_id, app_id);
}

//! Called when a user has joined a Steam group chat that the we are in.
void Steam::connected_chat_join(GameConnectedChatJoin_t* call_data){
	uint64_t chat_id = call_data->m_steamIDClanChat.ConvertToUint64();
	uint64_t steam_id = call_data->m_steamIDUser.ConvertToUint64();
	emit_signal("chat_joined", chat_id, steam_id);
}

//! Called when a user has left a Steam group chat that the we are in.
void Steam::connected_chat_leave(GameConnectedChatLeave_t* call_data){
	uint64_t chat_id = call_data->m_steamIDClanChat.ConvertToUint64();
	uint64_t steam_id = call_data->m_steamIDUser.ConvertToUint64();
	bool kicked = call_data->m_bKicked;
	bool dropped = call_data->m_bDropped;
	emit_signal("chat_left", chat_id, steam_id, kicked, dropped);
}

//! Called when a chat message has been received in a Steam group chat that we are in.
void Steam::connected_clan_chat_message(GameConnectedClanChatMsg_t* call_data){
	Dictionary chat;
	char text[2048];
	EChatEntryType type = k_EChatEntryTypeInvalid;
	CSteamID user_id;
	chat["ret"] = SteamFriends()->GetClanChatMessage(call_data->m_steamIDClanChat, call_data->m_iMessageID, text, 2048, &type, &user_id);
	chat["text"] = String(text);
	chat["type"] = type;
	chat["chatter"] = uint64_t(user_id.ConvertToUint64());
	emit_signal("clan_chat_message", chat);
}

//! Called when chat message has been received from a friend
void Steam::connected_friend_chat_message(GameConnectedFriendChatMsg_t* call_data){
	uint64_t steam_id = call_data->m_steamIDUser.ConvertToUint64();
	int message = call_data->m_iMessageID;
	Dictionary chat;
	char text[2048];
	EChatEntryType type = k_EChatEntryTypeInvalid;
	chat["ret"] = SteamFriends()->GetFriendMessage(createSteamID(steam_id), message, text, 2048, &type);
	chat["text"] = String(text);
	emit_signal("friend_chat_message", chat);
}

//! Called when the user tries to join a lobby from their friends list or from an invite. The game client should attempt to connect to specified lobby when this is received. If the game isn't running yet then the game will be automatically launched with the command line parameter +connect_lobby <64-bit lobby Steam ID> instead.
void Steam::join_requested(GameLobbyJoinRequested_t* call_data){
	CSteamID lobby_id = call_data->m_steamIDLobby;
	uint64_t lobby = lobby_id.ConvertToUint64();
	CSteamID friend_id = call_data->m_steamIDFriend;
	uint64_t steam_id = friend_id.ConvertToUint64();
	emit_signal("join_requested", lobby, steam_id);
}

//! Posted when the Steam Overlay activates or deactivates. The game can use this to be pause or resume single player games.
void Steam::overlay_toggled(GameOverlayActivated_t* call_data){
	if(call_data->m_bActive){
		emit_signal("overlay_toggled", true);
	}
	else{
		emit_signal("overlay_toggled", false);
	}
}

//! Called when the user tries to join a game from their friends list or after a user accepts an invite by a friend with inviteUserToGame.
void Steam::join_game_requested(GameRichPresenceJoinRequested_t* call_data){
	CSteamID steam_id = call_data->m_steamIDFriend;
	uint64_t user = steam_id.ConvertToUint64();
	String connect = call_data->m_rgchConnect;
	emit_signal("join_game_requested", user, connect);
}

//! This callback is made when joining a game. If the user is attempting to join a lobby, then the callback GameLobbyJoinRequested_t will be made.
void Steam::change_server_requested(GameServerChangeRequested_t* call_data){
	String server = call_data->m_rgchServer;
	String password = call_data->m_rgchPassword;
	emit_signal("change_server_requested", server, password);
}

//
void Steam::join_clan_chat_complete(JoinClanChatRoomCompletionResult_t* call_data){
	uint64_t chat_id = call_data->m_steamIDClanChat.ConvertToUint64();
	EChatRoomEnterResponse response = call_data->m_eChatRoomEnterResponse;
	emit_signal("chat_join_complete", chat_id, response);
}

//! Signal for a user change
void Steam::persona_state_change(PersonaStateChange_t* call_data){
	uint64_t steam_id = call_data->m_ulSteamID;
	int flags = call_data->m_nChangeFlags;
	emit_signal("persona_state_change", steam_id, flags);
}

//! Reports the result of an attempt to change the user's persona name.
void Steam::name_changed(SetPersonaNameResponse_t* call_data){
	bool success = call_data->m_bSuccess;
	bool local_success = call_data->m_bLocalSuccess;
	EResult result = call_data->m_result;
	emit_signal("name_changed", success, local_success, result);
}

//! Dispatched when an overlay browser instance is navigated to a protocol/scheme registered by RegisterProtocolInOverlayBrowser().
void Steam::overlay_browser_protocol(OverlayBrowserProtocolNavigation_t* call_data){
	String uri = call_data->rgchURI;
	emit_signal("overlay_browser_protocol", uri);
}

//! Purpose: Invoked when the status of unread messages changes
void Steam::unread_chat_messages_changed(UnreadChatMessagesChanged_t* call_data){
	emit_signal("unread_chat_messages_changed");
}

// Callback for when a user's equipped Steam Commuity profile items have changed. This can be for the current user or their friends.
void Steam::equipped_profile_items_changed(EquippedProfileItemsChanged_t* call_data){
	CSteamID this_steam_id = call_data->m_steamID;
	uint64_t steam_id = this_steam_id.ConvertToUint64();
	emit_signal("equipped_profile_items_changed", steam_id);
}

// Call from RequestEquippedProfileItems... nice.
void Steam::equipped_profile_items(EquippedProfileItems_t* call_data){
	int result = call_data->m_eResult;
	CSteamID this_steam_id = call_data->m_steamID;
	uint64_t steam_id = this_steam_id.ConvertToUint64();
	bool has_animated_avatar = call_data->m_bHasAnimatedAvatar;
	bool has_avatar_frame = call_data->m_bHasAvatarFrame;
	bool has_profile_modifier = call_data->m_bHasProfileModifier;
	bool has_profile_background = call_data->m_bHasProfileBackground;
	bool has_mini_profile_background = call_data->m_bHasMiniProfileBackground;
	// Pass all profile data to a dictionary
	Dictionary profile_data;
	profile_data["avatar_animated"] = has_animated_avatar;
	profile_data["avatar_frame"] = has_avatar_frame;
	profile_data["profile_modifier"] = has_profile_modifier;
	profile_data["profile_background"] = has_profile_background;
	profile_data["profile_mini_background"] = has_mini_profile_background;
	emit_signal("equipped_profile_items", result, steam_id, profile_data);
}

// GAME SEARCH CALLBACKS ////////////////////////
//
//! There are no notes about this in Valve's header files or documentation.
void Steam::search_for_game_progress(SearchForGameProgressCallback_t* call_data){
	EResult result = call_data->m_eResult;
	uint64_t search_id = call_data->m_ullSearchID;
	uint64_t lobby_id = call_data->m_lobbyID.ConvertToUint64();
	uint64_t steamIDEndedSearch = call_data->m_steamIDEndedSearch.ConvertToUint64();
	// Create a dictionary for search progress
	Dictionary search_progress;
	search_progress["lobby_id"] = lobby_id;
	search_progress["ended_search_id"] = steamIDEndedSearch;
	search_progress["seconds_remaining_estimate"] = call_data->m_nSecondsRemainingEstimate;
	search_progress["players_searching"] = call_data->m_cPlayersSearching;
	emit_signal("search_for_game_progress", result, search_id, search_progress);
}

//! Notification to all players searching that a game has been found.
void Steam::search_for_game_result(SearchForGameResultCallback_t* call_data){
	EResult result = call_data->m_eResult;
	uint64_t search_id = call_data->m_ullSearchID;
	uint64_t host_id = call_data->m_steamIDHost.ConvertToUint64();
	// Create a dictionary for search results
	Dictionary search_result;
	search_result["count_players_ingame"] = call_data->m_nCountPlayersInGame;
	search_result["count_accepted_game"] = call_data->m_nCountAcceptedGame;
	search_result["host_id"] = host_id;
	search_result["final_callback"] = call_data->m_bFinalCallback;
	emit_signal("search_for_game_result", result, search_id, search_result);
}

//! Callback from RequestPlayersForGame when the matchmaking service has started or ended search; callback will also follow a call from CancelRequestPlayersForGame - m_bSearchInProgress will be false.
void Steam::request_players_for_game_progress(RequestPlayersForGameProgressCallback_t* call_data){
	EResult result = call_data->m_eResult;
	uint64_t search_id = call_data->m_ullSearchID;
	emit_signal("request_players_for_game_progress", result, search_id);
}

//! Callback from RequestPlayersForGame, one of these will be sent per player followed by additional callbacks when players accept or decline the game.
void Steam::request_players_for_game_result(RequestPlayersForGameResultCallback_t* call_data){
	EResult result = call_data->m_eResult;
	uint64_t search_id = call_data->m_ullSearchID;
	uint64_t player_id = call_data->m_SteamIDPlayerFound.ConvertToUint64();
	uint64_t lobby_id = call_data->m_SteamIDLobby.ConvertToUint64();
	uint64_t unique_game_id = call_data->m_ullUniqueGameID;
	// Create a dictionary for the player data
	Dictionary player_data;
	player_data["player_id"] = player_id;
	player_data["lobby_id"] = lobby_id;
	player_data["player_accept_state"] = call_data->m_ePlayerAcceptState;
	player_data["player_index"] = call_data->m_nPlayerIndex;
	player_data["total_players"] = call_data->m_nTotalPlayersFound;
	player_data["total_players_accepted_game"] = call_data->m_nTotalPlayersAcceptedGame;
	player_data["suggested_team_index"] = call_data->m_nSuggestedTeamIndex;
	player_data["unique_game_id"] = unique_game_id;
	// Send the data back via signal
	emit_signal("request_players_for_game_result", result, search_id, player_data);
}

//! There are no notes about this in Valve's header files or documentation.
void Steam::request_players_for_game_final_result(RequestPlayersForGameFinalResultCallback_t* call_data){
	EResult result = call_data->m_eResult;
	uint64_t search_id = call_data->m_ullSearchID;
	uint64_t game_id = call_data->m_ullUniqueGameID;
	emit_signal("request_players_for_game_final", result, search_id, game_id);
}

//! This callback confirms that results were received by the matchmaking service for this player.
void Steam::submit_player_result(SubmitPlayerResultResultCallback_t* call_data){
	EResult result = call_data->m_eResult;
	uint64_t game_id = call_data->ullUniqueGameID;
	uint64_t player_id = call_data->steamIDPlayer.ConvertToUint64();
	emit_signal("submit_player_result", result, game_id, player_id);
}

//! This callback confirms that the game is recorded as complete on the matchmaking service, the next call to RequestPlayersForGame will generate a new unique game ID.
void Steam::end_game_result(EndGameResultCallback_t* call_data){
	EResult result = call_data->m_eResult;
	uint64_t game_id = call_data->ullUniqueGameID;
	emit_signal("end_game_result", result, game_id);
}

// GAME SERVER CALLBACKS ////////////////////////
//
// Logging the game server onto Steam
void Steam::server_connect_failure(SteamServerConnectFailure_t* serverData){
	int result = serverData->m_eResult;
	bool retrying = serverData->m_bStillRetrying;
	emit_signal("server_connect_failure", result, retrying);
}

// Server has connected to the Steam back-end; serverData has no fields.
void Steam::server_connected(SteamServersConnected_t* serverData){
	emit_signal("server_connected");
}

// Called if the client has lost connection to the Steam servers. Real-time services will be disabled until a matching SteamServersConnected_t has been posted.
void Steam::server_disconnected(SteamServersDisconnected_t* serverData){
	int result = serverData->m_eResult;
	emit_signal("server_disconnected", result);
}

// Client has been approved to connect to this game server.
void Steam::client_approved(GSClientApprove_t* clientData){
	uint64_t steam_id = clientData->m_SteamID.ConvertToUint64();
	uint64_t owner_id = clientData->m_OwnerSteamID.ConvertToUint64();
	emit_signal("client_approved", steam_id, owner_id);
}

// Client has been denied to connection to this game server.
void Steam::client_denied(GSClientDeny_t* clientData){
	uint64_t steam_id = clientData->m_SteamID.ConvertToUint64();
	int reason;
	// Convert reason.
	if(clientData->m_eDenyReason == k_EDenyInvalid){
		reason = DENY_INVALID;
	}
	else if(clientData->m_eDenyReason == k_EDenyInvalidVersion){
		reason = DENY_INVALID_VERSION;
	}
	else if(clientData->m_eDenyReason == k_EDenyGeneric){
		reason = DENY_GENERIC;
	}
	else if(clientData->m_eDenyReason == k_EDenyNotLoggedOn){
		reason = DENY_NOT_LOGGED_ON;
	}
	else if(clientData->m_eDenyReason == k_EDenyNoLicense){
		reason = DENY_NO_LICENSE;
	}
	else if(clientData->m_eDenyReason == k_EDenyCheater){
		reason = DENY_CHEATER;
	}
	else if(clientData->m_eDenyReason == k_EDenyLoggedInElseWhere){
		reason = DENY_LOGGED_IN_ELSEWHERE;
	}
	else if(clientData->m_eDenyReason == k_EDenyUnknownText){
		reason = DENY_UNKNOWN_TEXT;
	}
	else if(clientData->m_eDenyReason == k_EDenyIncompatibleAnticheat){
		reason = DENY_INCOMPATIBLE_ANTI_CHEAT;
	}
	else if(clientData->m_eDenyReason == k_EDenyMemoryCorruption){
		reason = DENY_MEMORY_CORRUPTION;
	}
	else if(clientData->m_eDenyReason == k_EDenyIncompatibleSoftware){
		reason = DENY_INCOMPATIBLE_SOFTWARE;
	}
	else if(clientData->m_eDenyReason == k_EDenySteamConnectionLost){
		reason = DENY_STEAM_CONNECTION_LOST;
	}
	else if(clientData->m_eDenyReason == k_EDenySteamConnectionError){
		reason = DENY_STEAM_CONNECTION_ERROR;
	}
	else if(clientData->m_eDenyReason == k_EDenySteamResponseTimedOut){
		reason = DENY_STEAM_RESPONSE_TIMED_OUT;
	}
	else if(clientData->m_eDenyReason == k_EDenySteamValidationStalled){
		reason = DENY_STEAM_VALIDATION_STALLED;
	}
	else{
		reason = DENY_STEAM_OWNER_LEFT_GUEST_USER;
	}
	emit_signal("client_denied", steam_id, reason);
}

// Request the game server should kick the user.
void Steam::client_kick(GSClientKick_t* clientData){
	uint64_t steam_id = clientData->m_SteamID.ConvertToUint64();
	int reason;
	// Convert reason.
	if(clientData->m_eDenyReason == k_EDenyInvalid){
		reason = DENY_INVALID;
	}
	else if(clientData->m_eDenyReason == k_EDenyInvalidVersion){
		reason = DENY_INVALID_VERSION;
	}
	else if(clientData->m_eDenyReason == k_EDenyGeneric){
		reason = DENY_GENERIC;
	}
	else if(clientData->m_eDenyReason == k_EDenyNotLoggedOn){
		reason = DENY_NOT_LOGGED_ON;
	}
	else if(clientData->m_eDenyReason == k_EDenyNoLicense){
		reason = DENY_NO_LICENSE;
	}
	else if(clientData->m_eDenyReason == k_EDenyCheater){
		reason = DENY_CHEATER;
	}
	else if(clientData->m_eDenyReason == k_EDenyLoggedInElseWhere){
		reason = DENY_LOGGED_IN_ELSEWHERE;
	}
	else if(clientData->m_eDenyReason == k_EDenyUnknownText){
		reason = DENY_UNKNOWN_TEXT;
	}
	else if(clientData->m_eDenyReason == k_EDenyIncompatibleAnticheat){
		reason = DENY_INCOMPATIBLE_ANTI_CHEAT;
	}
	else if(clientData->m_eDenyReason == k_EDenyMemoryCorruption){
		reason = DENY_MEMORY_CORRUPTION;
	}
	else if(clientData->m_eDenyReason == k_EDenyIncompatibleSoftware){
		reason = DENY_INCOMPATIBLE_SOFTWARE;
	}
	else if(clientData->m_eDenyReason == k_EDenySteamConnectionLost){
		reason = DENY_STEAM_CONNECTION_LOST;
	}
	else if(clientData->m_eDenyReason == k_EDenySteamConnectionError){
		reason = DENY_STEAM_CONNECTION_ERROR;
	}
	else if(clientData->m_eDenyReason == k_EDenySteamResponseTimedOut){
		reason = DENY_STEAM_RESPONSE_TIMED_OUT;
	}
	else if(clientData->m_eDenyReason == k_EDenySteamValidationStalled){
		reason = DENY_STEAM_VALIDATION_STALLED;
	}
	else{
		reason = DENY_STEAM_OWNER_LEFT_GUEST_USER;
	}
	emit_signal("client_kick", steam_id, reason);
}

// Received when the game server requests to be displayed as secure (VAC protected).
// m_bSecure is true if the game server should display itself as secure to users, false otherwise.
void Steam::policy_response(GSPolicyResponse_t* policyData){
	uint8 secure = policyData->m_bSecure;
	emit_signal("policy_response", secure);
}

// Sent as a reply to RequestUserGroupStatus().
void Steam::client_group_status(GSClientGroupStatus_t* clientData){
	uint64_t steam_id = clientData->m_SteamIDUser.ConvertToUint64();
	uint64_t group_id = clientData->m_SteamIDGroup.ConvertToUint64();
	bool member = clientData->m_bMember;
	bool officer = clientData->m_bOfficer;
	emit_signal("client_group_status", steam_id, group_id, member, officer);
}

// Sent as a reply to AssociateWithClan().
void Steam::associate_clan(AssociateWithClanResult_t* clanData){
	int result;
	if(clanData->m_eResult == k_EResultOK){
		result = RESULT_OK;
	}
	else{
		result = RESULT_FAIL;
	}
	emit_signal("associate_clan", result);
}

// Sent as a reply to ComputeNewPlayerCompatibility().
void Steam::player_compat(ComputeNewPlayerCompatibilityResult_t* playerData){
	int result;
	if(playerData->m_eResult == k_EResultNoConnection){
		result = RESULT_NO_CONNECTION;
	}
	else if(playerData->m_eResult == k_EResultTimeout){
		result = RESULT_TIMEOUT;
	}
	else if(playerData->m_eResult == k_EResultFail){
		result = RESULT_FAIL;
	}
	else{
		result = RESULT_OK;
	}
	int players_dont_like_candidate = playerData->m_cPlayersThatDontLikeCandidate;
	int players_candidate_doesnt_like = playerData->m_cPlayersThatCandidateDoesntLike;
	int clan_players_dont_like_candidate = playerData->m_cClanPlayersThatDontLikeCandidate;
	uint64_t steam_id = playerData->m_SteamIDCandidate.ConvertToUint64();
	emit_signal("player_compat", result, players_dont_like_candidate, players_candidate_doesnt_like, clan_players_dont_like_candidate, steam_id);
}

// GAME SERVER STATS CALLBACKS //////////////////
//
// Result when getting the latests stats and achievements for a user from the server.
void Steam::stats_received(GSStatsReceived_t* callData, bool bioFailure){
	EResult result = callData->m_eResult;
	uint64_t steam_id = callData->m_steamIDUser.ConvertToUint64();
	emit_signal("stats_received", result, steam_id);
}

// Result of a request to store the user stats.
void Steam::stats_stored(GSStatsStored_t* callData){
	EResult result = callData->m_eResult;
	uint64_t steam_id = callData->m_steamIDUser.ConvertToUint64();
	emit_signal("stats_stored", result, steam_id);
}

// Callback indicating that a user's stats have been unloaded.
void Steam::stats_unloaded(GSStatsUnloaded_t* callData){
	uint64_t steam_id = callData->m_steamIDUser.ConvertToUint64();
	emit_signal("stats_unloaded", steam_id);
}

// HTML SURFACE CALLBACKS ///////////////////////
// 
//! A new browser was created and is ready for use.
void Steam::html_browser_ready(HTML_BrowserReady_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	emit_signal("html_browser_ready", browser_handle);
}

//! Called when page history status has changed the ability to go backwards and forward.
void Steam::html_can_go_backandforward(HTML_CanGoBackAndForward_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	bool go_back = call_data->bCanGoBack;
	bool go_forward = call_data->bCanGoForward;
	emit_signal("html_can_go_backandforward", browser_handle, go_back, go_forward);
}

//! Called when the current page in a browser gets a new title.
void Steam::html_changed_title(HTML_ChangedTitle_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	const String& title = call_data->pchTitle;
	emit_signal("html_changed_title", browser_handle, title);
}

//! Called when the browser has been requested to close due to user interaction; usually because of a javascript window.close() call.
void Steam::html_close_browser(HTML_CloseBrowser_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	emit_signal("html_close_browser", browser_handle);
}

//! Called when a browser surface has received a file open dialog from a <input type="file"> click or similar, you must call FileLoadDialogResponse with the file(s) the user selected.
void Steam::html_file_open_dialog(HTML_FileOpenDialog_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	const String& title = call_data->pchTitle;
	const String& initial_file = call_data->pchInitialFile;
	// Allows you to react to a page wanting to open a file load dialog. NOTE: You MUST call this in response to a HTML_FileOpenDialog_t callback.
	// So it is added here unless there is a case to use it separately.
	SteamHTMLSurface()->FileLoadDialogResponse(browser_handle, &call_data->pchInitialFile);
	// Send the signal back to the user
	emit_signal("html_file_open_dialog", browser_handle, title, initial_file);
}

//! Called when a browser has finished loading a page.
void Steam::html_finished_request(HTML_FinishedRequest_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	const String& url = call_data->pchURL;
	const String& title = call_data->pchPageTitle;
	emit_signal("html_finished_request", browser_handle, url, title);
}

//! Called when a a browser wants to hide a tooltip.
void Steam::html_hide_tooltip(HTML_HideToolTip_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	emit_signal("html_hide_tooltip", browser_handle);
}

//! Provides details on the visibility and size of the horizontal scrollbar.
void Steam::html_horizontal_scroll(HTML_HorizontalScroll_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	// Create dictionary to bypass argument limit in Godot
	Dictionary scroll_data;
	scroll_data["scroll_max"] = call_data->unScrollMax;
	scroll_data["scroll_current"] = call_data->unScrollCurrent;
	scroll_data["page_scale"] = call_data->flPageScale;
	scroll_data["visible"] = call_data->bVisible;
	scroll_data["page_size"] = call_data->unPageSize;
	emit_signal("html_horizontal_scroll", browser_handle, scroll_data);
}

//! Called when the browser wants to display a Javascript alert dialog, call JSDialogResponse when the user dismisses this dialog; or right away to ignore it.
void Steam::html_js_alert(HTML_JSAlert_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	const String& message = call_data->pchMessage;
	emit_signal("html_js_alert", browser_handle, message);
}

//! Called when the browser wants to display a Javascript confirmation dialog, call JSDialogResponse when the user dismisses this dialog; or right away to ignore it.
void Steam::html_js_confirm(HTML_JSConfirm_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	const String& message = call_data->pchMessage;
	emit_signal("html_js_confirm", browser_handle, message);
}

//! Result of a call to GetLinkAtPosition.
void Steam::html_link_at_position(HTML_LinkAtPosition_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	// Create dictionary to bypass Godot argument limit
	Dictionary link_data;
	link_data["x"] = call_data->x;
	link_data["y"] = call_data->y;
	link_data["url"] = call_data->pchURL;
	link_data["input"] = call_data->bInput;
	link_data["live_link"] = call_data->bLiveLink;
	emit_signal("html_link_at_position", browser_handle, link_data);
}

//! Called when a browser surface has a pending paint. This is where you get the actual image data to render to the screen.
void Steam::html_needs_paint(HTML_NeedsPaint_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	// Create dictionary to bypass Godot argument limit
	Dictionary page_data;
	page_data["bgra"] = call_data->pBGRA;
	page_data["wide"] = call_data->unWide;
	page_data["tall"] = call_data->unTall;
	page_data["update_x"] = call_data->unUpdateX;
	page_data["update_y"] = call_data->unUpdateY;
	page_data["update_wide"] = call_data->unUpdateWide;
	page_data["update_tall"] = call_data->unUpdateTall;
	page_data["scroll_x"] = call_data->unScrollX;
	page_data["scroll_y"] = call_data->unScrollY;
	page_data["page_scale"] = call_data->flPageScale;
	page_data["page_serial"] = call_data->unPageSerial;
	emit_signal("html_needs_paint", browser_handle, page_data);
}

//! A browser has created a new HTML window.
void Steam::html_new_window(HTML_NewWindow_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	// Create a dictionary to bypass Godot argument limit
	Dictionary window_data;
	window_data["url"] = call_data->pchURL;
	window_data["x"] = call_data->unX;
	window_data["y"] = call_data->unY;
	window_data["wide"] = call_data->unWide;
	window_data["tall"] = call_data->unTall;
	window_data["new_handle"] = call_data->unNewWindow_BrowserHandle_IGNORE;
	emit_signal("html_new_window", browser_handle, window_data);
}

//! The browser has requested to load a url in a new tab.
void Steam::html_open_link_in_new_tab(HTML_OpenLinkInNewTab_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	const String& url = call_data->pchURL;
	emit_signal("html_open_link_in_new_tab", browser_handle, url);
}

//! Results from a search.
void Steam::html_search_results(HTML_SearchResults_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	uint32 results = call_data->unResults;
	uint32 current_match = call_data->unCurrentMatch;
	emit_signal("html_search_results", browser_handle, results, current_match);
}

//! Called when a browser wants to change the mouse cursor.
void Steam::html_set_cursor(HTML_SetCursor_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	uint32 mouse_cursor = call_data->eMouseCursor;
	emit_signal("html_set_cursor", browser_handle, mouse_cursor);
}

//! Called when a browser wants to display a tooltip.
void Steam::html_show_tooltip(HTML_ShowToolTip_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	const String& message = call_data->pchMsg;
	emit_signal("html_show_tooltip", browser_handle, message);
}

//! Called when a browser wants to navigate to a new page.
void Steam::html_start_request(HTML_StartRequest_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	const String& url = call_data->pchURL;
	const String& target = call_data->pchTarget;
	const String& post_data = call_data->pchPostData;
	bool redirect = call_data->bIsRedirect;
	emit_signal("html_start_request", browser_handle, url, target, post_data, redirect);
}

//! Called when a browser wants you to display an informational message. This is most commonly used when you hover over links.
void Steam::html_status_text(HTML_StatusText_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	const String& message = call_data->pchMsg;
	emit_signal("html_status_text", browser_handle, message);
}

//! Called when the text of an existing tooltip has been updated.
void Steam::html_update_tooltip(HTML_UpdateToolTip_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	const String& message = call_data->pchMsg;
	emit_signal("html_update_tooltip", browser_handle, message);
}

//! Called when the browser is navigating to a new url.
void Steam::html_url_changed(HTML_URLChanged_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	// Create a dictionary to bypass Godot argument limit
	Dictionary url_data;
	url_data["url"] = call_data->pchURL;
	url_data["post_data"] = call_data->pchPostData;
	url_data["redirect"] = call_data->bIsRedirect;
	url_data["title"] = call_data->pchPageTitle;
	url_data["new_navigation"] = call_data->bNewNavigation;
	emit_signal("html_url_changed", browser_handle, url_data);
}

//! Provides details on the visibility and size of the vertical scrollbar.
void Steam::html_vertical_scroll(HTML_VerticalScroll_t* call_data){
	browser_handle = call_data->unBrowserHandle;
	// Create dictionary to bypass argument limit in Godot
	Dictionary scroll_data;
	scroll_data["scroll_max"] = call_data->unScrollMax;
	scroll_data["scroll_current"] = call_data->unScrollCurrent;
	scroll_data["page_scale"] = call_data->flPageScale;
	scroll_data["visible"] = call_data->bVisible;
	scroll_data["page_size"] = call_data->unPageSize;
	emit_signal("html_vertical_scroll", browser_handle, scroll_data);
}

// HTTP CALLBACKS ///////////////////////////////
//
//! Result when an HTTP request completes. If you're using GetHTTPStreamingResponseBodyData then you should be using the HTTPRequestHeadersReceived_t or HTTPRequestDataReceived_t.
void Steam::http_request_completed(HTTPRequestCompleted_t* call_data){
	uint32 cookie_handle = call_data->m_hRequest;
	uint64_t context_value = call_data->m_ulContextValue;
	bool request_success = call_data->m_bRequestSuccessful;
	int status_code = call_data->m_eStatusCode;
	uint32 body_size = call_data->m_unBodySize;
	emit_signal("http_request_completed", cookie_handle, context_value, request_success, status_code, body_size);
}

//! Triggered when a chunk of data is received from a streaming HTTP request.
void Steam::http_request_data_received(HTTPRequestDataReceived_t* call_data){
	uint32 cookie_handle = call_data->m_hRequest;
	uint64_t context_value = call_data->m_ulContextValue;
	uint32 offset = call_data->m_cOffset;
	uint32 bytes_received = call_data->m_cBytesReceived;
	emit_signal("http_request_data_received", cookie_handle, context_value, offset, bytes_received);
}

//! Triggered when HTTP headers are received from a streaming HTTP request.
void Steam::http_request_headers_received(HTTPRequestHeadersReceived_t* call_data){
	uint32 cookie_handle = call_data->m_hRequest;
	uint64_t context_value = call_data->m_ulContextValue;
	emit_signal("http_request_headers_received", cookie_handle, context_value);
}

// INPUT CALLBACKS //////////////////////////////
//
// Purpose: when callbacks are enabled this fires each time a controller action state changes
//void Steam::input_action_event(SteamInputActionEvent_t* call_data){
//	uint64_t input_handle = call_data->controllerHandle;
//	int event_type = call_data->eEventType;
//	uint64_t analog_handle = call_data->analogAction.actionHandle;
//	uint64_t digital_handle = call_data->digitalAction.actionHandle;
//	// Get the digital action data
//	Dictionary digital_action;
//	digital_action["state"] = call_data->digitalAction.digitalActionData.bState;
//	digital_action["active"] = call_data->digitalAction.digitalActionData.bActive;
//	// Get the analog action data
//	Dictionary analog_action;
//	analog_action["mode"] = call_data->analogAction.analogActionData.eMode;
//	analog_action["x"] = call_data->analogAction.analogActionData.x;
//	analog_action["y"] = call_data->analogAction.analogActionData.y;
//	analog_action["active"] = call_data->analogAction.analogActionData.bActive;
//	// Split into a dictionary since Godot won't allow more than 6 arguments sent back
//	Dictionary action_information;
//	action_information["analog_action_handle"] = analog_handle;
//	action_information["analog_action_data"] = analog_action;
//	action_information["digital_action_handle"] = digital_handle;
//	action_information["digital_action_data"] = digital_action;
//	emit_signal("input_action_event", input_handle, event_type, action_information);
//}

//! Purpose: called when a new controller has been connected, will fire once per controller if multiple new controllers connect in the same frame
void Steam::input_device_connected(SteamInputDeviceConnected_t* call_data){
	uint64_t input_handle = call_data->m_ulConnectedDeviceHandle;
	emit_signal("input_device_connected", input_handle);
}

//! Purpose: called when a new controller has been connected, will fire once per controller if multiple new controllers connect in the same frame
void Steam::input_device_disconnected(SteamInputDeviceDisconnected_t* call_data){
	uint64_t input_handle = call_data->m_ulDisconnectedDeviceHandle;
	emit_signal("input_device_disconnected", input_handle);
}

//! Purpose: called when a controller configuration has been loaded, will fire once per controller per focus change for Steam Input enabled controllers
void Steam::input_configuration_loaded(SteamInputConfigurationLoaded_t* call_data){
	uint32_t app_id = call_data->m_unAppID;
	uint64_t device_handle = call_data->m_ulDeviceHandle;
	// Split into a dictionary since Godot won't allow more than 6 arguments sent back
	Dictionary config_data;
	uint64_t mapping_creator = call_data->m_ulMappingCreator.ConvertToUint64();
	config_data["mapping_creator"] = mapping_creator;
	config_data["major_revision"] = call_data->m_unMajorRevision;						// uint32
	config_data["minor_revision"] = call_data->m_unMinorRevision;						// uint32
	config_data["uses_steam_input_api"] = call_data->m_bUsesSteamInputAPI;				// bool
	config_data["uses_gamepad_api"] = call_data->m_bUsesGamepadAPI;						// bool
	emit_signal("input_configuration_loaded", app_id, device_handle, config_data);
}

// INVENTORY CALLBACKS //////////////////////////
//
//! This callback is triggered whenever item definitions have been updated, which could be in response to LoadItemDefinitions or any time new item definitions are available (eg, from the dynamic addition of new item types while players are still in-game).
void Steam::inventory_definition_update(SteamInventoryDefinitionUpdate_t *call_data){
	// Create the return array
	Array definitions;
	// Set the array size variable
	uint32 size = 0;
	// Get the item defition IDs
	if(SteamInventory()->GetItemDefinitionIDs(NULL, &size)){
		SteamItemDef_t *id_array = new SteamItemDef_t[size];
		if(SteamInventory()->GetItemDefinitionIDs(id_array, &size)){
			// Loop through the temporary array and populate the return array
			for(uint32 i = 0; i < size; i++){
				definitions.append(id_array[i]);
			}
		}
		// Delete the temporary array
		delete[] id_array;
	}
	// Return the item array as a signal
	emit_signal("inventory_defintion_update", definitions);
}

//! Triggered when GetAllItems successfully returns a result which is newer / fresher than the last known result. (It will not trigger if the inventory hasn't changed, or if results from two overlapping calls are reversed in flight and the earlier result is already known to be stale/out-of-date.)
//! The regular SteamInventoryResultReady_t callback will still be triggered immediately afterwards; this is an additional notification for your convenience.
void Steam::inventory_full_update(SteamInventoryFullUpdate_t *call_data){
	// Set the handle
	inventory_handle = call_data->m_handle;
	// Send the handle back to the user
	emit_signal("inventory_full_update", call_data->m_handle);
}

//! This is fired whenever an inventory result transitions from k_EResultPending to any other completed state, see GetResultStatus for the complete list of states. There will always be exactly one callback per handle.
void Steam::inventory_result_ready(SteamInventoryResultReady_t *call_data){
	// Get the result
	int result = call_data->m_result;
	// Get the handle and pass it over
	inventory_handle = call_data->m_handle;
	emit_signal("inventory_result_ready", result, inventory_handle);
}

// MATCHMAKING CALLBACKS ////////////////////////
//
//! Called when an account on your favorites list is updated
void Steam::favorites_list_accounts_updated(FavoritesListAccountsUpdated_t* call_data){
	int result = call_data->m_eResult;
	emit_signal("favorites_list_accounts_updated", result);
}

//! A server was added/removed from the favorites list, you should refresh now.
void Steam::favorites_list_changed(FavoritesListChanged_t* call_data){
	Dictionary favorite;
	// Convert the IP address back to a string
	const int NBYTES = 4;
	uint8 octet[NBYTES];
	char favoriteIP[16];
	for(int j = 0; j < NBYTES; j++){
		octet[j] = call_data->m_nIP >> (j * 8);
	}
	sprintf(favoriteIP, "%d.%d.%d.%d", octet[0], octet[1], octet[2], octet[3]);
	favorite["ip"] = favoriteIP; 
	favorite["query_port"] = call_data->m_nQueryPort;
	favorite["connection_port"] = call_data->m_nConnPort;
	favorite["app_id"] = call_data->m_nAppID;
	favorite["flags"] = call_data->m_nFlags;
	favorite["add"] = call_data->m_bAdd;
	favorite["account_id"] = call_data->m_unAccountId;
	emit_signal("favorites_list_changed", favorite);
}

//! Signal when a lobby chat message is received
void Steam::lobby_message(LobbyChatMsg_t* call_data){
	CSteamID lobby_id = call_data->m_ulSteamIDLobby;
	CSteamID user_id = call_data->m_ulSteamIDUser;
	uint8 chat_type = call_data->m_eChatEntryType;
	// Convert the chat type over
	EChatEntryType type = (EChatEntryType)chat_type;
	// Get the chat message data
	char buffer[STEAM_LARGE_BUFFER_SIZE];
	int size = SteamMatchmaking()->GetLobbyChatEntry(lobby_id, call_data->m_iChatID, &user_id, &buffer, STEAM_LARGE_BUFFER_SIZE, &type);
	uint64_t lobby = lobby_id.ConvertToUint64();
	uint64_t user = user_id.ConvertToUint64();
	emit_signal("lobby_message", lobby, user, String::utf8(buffer, size), chat_type);
}

//! A lobby chat room state has changed, this is usually sent when a user has joined or left the lobby.
void Steam::lobby_chat_update(LobbyChatUpdate_t* call_data){
	uint64_t lobby_id = call_data->m_ulSteamIDLobby;
	uint64_t changed_id = call_data->m_ulSteamIDUserChanged;
	uint64_t making_change_id = call_data->m_ulSteamIDMakingChange;
	uint32 chat_state = call_data->m_rgfChatMemberStateChange;
	emit_signal("lobby_chat_update", lobby_id, changed_id, making_change_id, chat_state);
}

//! The lobby metadata has changed.
void Steam::lobby_data_update(LobbyDataUpdate_t* call_data){
	uint64_t member_id = call_data->m_ulSteamIDMember;
	uint64_t lobby_id = call_data->m_ulSteamIDLobby;
	uint8 success = call_data->m_bSuccess;
	emit_signal("lobby_data_update", success, lobby_id, member_id);
}

//! Posted if a user is forcefully removed from a lobby; can occur if a user loses connection to Steam.
void Steam::lobby_kicked(LobbyKicked_t* call_data){
	CSteamID steam_lobby_id = call_data->m_ulSteamIDLobby;
	uint64_t lobby_id = steam_lobby_id.ConvertToUint64();
	CSteamID steam_admin_id = call_data->m_ulSteamIDAdmin;
	uint64_t admin_id = steam_admin_id.ConvertToUint64();
	uint8 due_to_disconnect = call_data->m_bKickedDueToDisconnect;
	emit_signal("lobby_kicked", lobby_id, admin_id, due_to_disconnect);
}

//! Received upon attempting to enter a lobby. Lobby metadata is available to use immediately after receiving this.
void Steam::lobby_joined(LobbyEnter_t* lobbyData){
	CSteamID steam_lobby_id = lobbyData->m_ulSteamIDLobby;
	uint64_t lobby_id = steam_lobby_id.ConvertToUint64();
	uint32_t permissions = lobbyData->m_rgfChatPermissions;
	bool locked = lobbyData->m_bLocked;
	uint32_t response = lobbyData->m_EChatRoomEnterResponse;
	emit_signal("lobby_joined", lobby_id, permissions, locked, response);
}

//! A game server has been set via SetLobbyGameServer for all of the members of the lobby to join. It's up to the individual clients to take action on this; the typical game behavior is to leave the lobby and connect to the specified game server; but the lobby may stay open throughout the session if desired.
void Steam::lobby_game_created(LobbyGameCreated_t* call_data){
	uint64_t lobby_id = call_data->m_ulSteamIDLobby;
	uint64_t server_id = call_data->m_ulSteamIDGameServer;
	uint32 ip = call_data->m_unIP;
	uint16 port = call_data->m_usPort;
	// Convert the IP address back to a string
	const int NBYTES = 4;
	uint8 octet[NBYTES];
	char server_ip[16];
	for(int i = 0; i < NBYTES; i++){
		octet[i] = ip >> (i * 8);
	}
	sprintf(server_ip, "%d.%d.%d.%d", octet[0], octet[1], octet[2], octet[3]);
	emit_signal("lobby_game_created", lobby_id, server_id, server_ip, port);
}

//! Someone has invited you to join a Lobby. Normally you don't need to do anything with this, as the Steam UI will also display a '<user> has invited you to the lobby, join?' notification and message. If the user outside a game chooses to join, your game will be launched with the parameter +connect_lobby <64-bit lobby id>, or with the callback GameLobbyJoinRequested_t if they're already in-game.
void Steam::lobby_invite(LobbyInvite_t* lobbyData){
	CSteamID inviter_id = lobbyData->m_ulSteamIDUser;
	uint64_t inviter = inviter_id.ConvertToUint64();
	CSteamID lobby_id = lobbyData->m_ulSteamIDLobby;
	uint64_t lobby = lobby_id.ConvertToUint64();
	CSteamID game_id = lobbyData->m_ulGameID;
	uint64_t game = game_id.ConvertToUint64();
	emit_signal("lobby_invite", inviter, lobby, game);
}

// MUSIC REMOTE CALLBACKS ///////////////////////
//
// The majority of callback for Music Remote have no fields and no descriptions. They seem to be primarily fired as responses to functions.
void Steam::music_player_remote_to_front(MusicPlayerRemoteToFront_t* call_data){
	emit_signal("music_player_remote_to_front");
}
void Steam::music_player_remote_will_activate(MusicPlayerRemoteWillActivate_t* call_data){
	emit_signal("music_player_remote_will_activate");
}
void Steam::music_player_remote_will_deactivate(MusicPlayerRemoteWillDeactivate_t* call_data){
	emit_signal("music_player_remote_will_deactivate");
}
void Steam::music_player_selects_playlist_entry(MusicPlayerSelectsPlaylistEntry_t* call_data){
	int entry = call_data->nID;
	emit_signal("music_player_selects_playlist_entry", entry);
}
void Steam::music_player_selects_queue_entry(MusicPlayerSelectsQueueEntry_t* call_data){
	int entry = call_data->nID;
	emit_signal("music_player_selects_queue_entry", entry);
}
void Steam::music_player_wants_looped(MusicPlayerWantsLooped_t* call_data){
	bool looped = call_data->m_bLooped;
	emit_signal("music_player_wants_looped", looped);
}
void Steam::music_player_wants_pause(MusicPlayerWantsPause_t* call_data){
	emit_signal("music_player_wants_pause");
}
void Steam::music_player_wants_playing_repeat_status(MusicPlayerWantsPlayingRepeatStatus_t* call_data){
	int status = call_data->m_nPlayingRepeatStatus;
	emit_signal("music_player_wants_playing_repeat_status", status);
}
void Steam::music_player_wants_play_next(MusicPlayerWantsPlayNext_t* call_data){
	emit_signal("music_player_wants_play_next");
}
void Steam::music_player_wants_play_previous(MusicPlayerWantsPlayPrevious_t* call_data){
	emit_signal("music_player_wants_play_previous");
}
void Steam::music_player_wants_play(MusicPlayerWantsPlay_t* call_data){
	emit_signal("music_player_wants_play");
}
void Steam::music_player_wants_shuffled(MusicPlayerWantsShuffled_t* call_data){
	bool shuffled = call_data->m_bShuffled;
	emit_signal("music_player_wants_shuffled", shuffled);
}
void Steam::music_player_wants_volume(MusicPlayerWantsVolume_t* call_data){
	float volume = call_data->m_flNewVolume;
	emit_signal("music_player_wants_volume", volume);
}
void Steam::music_player_will_quit(MusicPlayerWillQuit_t* call_data){
	emit_signal("music_player_will_quit");
}

// NETWORKING CALLBACKS /////////////////////////
//
//! Called when packets can't get through to the specified user. All queued packets unsent at this point will be dropped, further attempts to send will retry making the connection (but will be dropped if we fail again).
void Steam::p2p_session_connect_fail(P2PSessionConnectFail_t* call_data) {
	uint64_t steam_id_remote = call_data->m_steamIDRemote.ConvertToUint64();
	uint8_t session_error = call_data->m_eP2PSessionError;
	emit_signal("p2p_session_connect_fail", steam_id_remote, session_error);
}

//! A user wants to communicate with us over the P2P channel via the sendP2PPacket. In response, a call to acceptP2PSessionWithUser needs to be made, if you want to open the network channel with them.
void Steam::p2p_session_request(P2PSessionRequest_t* call_data){
	uint64_t steam_id_remote = call_data->m_steamIDRemote.ConvertToUint64();
	emit_signal("p2p_session_request", steam_id_remote);
}

// NETWORKING MESSAGES CALLBACKS ////////////////
//
//! Posted when a remote host is sending us a message, and we do not already have a session with them.
void Steam::network_messages_session_request(SteamNetworkingMessagesSessionRequest_t* call_data){
	SteamNetworkingIdentity remote = call_data->m_identityRemote;
	char identity[STEAM_BUFFER_SIZE];
	remote.ToString(identity, STEAM_BUFFER_SIZE);
	emit_signal("network_messages_session_request", identity);
}

//! Posted when we fail to establish a connection, or we detect that communications have been disrupted it an unusual way.
void Steam::network_messages_session_failed(SteamNetworkingMessagesSessionFailed_t* call_data){
	SteamNetConnectionInfo_t info = call_data->m_info;
	// Parse out the reason for failure
	int reason = info.m_eEndReason;
	emit_signal("network_messages_session_failed", reason);
}

// NETWORKING SOCKETS CALLBACKS /////////////////
//
//! This callback is posted whenever a connection is created, destroyed, or changes state. The m_info field will contain a complete description of the connection at the time the change occurred and the callback was posted. In particular, m_info.m_eState will have the new connection state.
void Steam::network_connection_status_changed(SteamNetConnectionStatusChangedCallback_t* call_data){
	// Connection handle.
	uint64_t connect_handle = call_data->m_hConn;
	// Full connection info.
	SteamNetConnectionInfo_t connection_info = call_data->m_info;
	// Move connection info into a dictionary
	Dictionary connection;
	char identity[STEAM_BUFFER_SIZE];
	connection_info.m_identityRemote.ToString(identity, STEAM_BUFFER_SIZE);
	connection["identity"] = identity;
	connection["user_data"] = (uint64_t)connection_info.m_nUserData;
	connection["listen_socket"] = connection_info.m_hListenSocket;
	char ip_address[STEAM_BUFFER_SIZE];
	connection_info.m_addrRemote.ToString(ip_address, STEAM_BUFFER_SIZE, true);
	connection["remote_address"] = ip_address;
	connection["remote_pop"] = connection_info.m_idPOPRemote;
	connection["pop_relay"] = connection_info.m_idPOPRelay;
	connection["connection_state"] = connection_info.m_eState;
	connection["end_reason"] = connection_info.m_eEndReason;
	connection["end_debug"] = connection_info.m_szEndDebug;
	connection["debug_description"] = connection_info.m_szConnectionDescription;
	// Previous state (current state is in m_info.m_eState).
	int old_state = call_data->m_eOldState;
	// Send the data back via signal
	emit_signal("network_connection_status_changed", connect_handle, connection, old_state);
}

//! This callback is posted whenever the state of our readiness changes.
void Steam::network_authentication_status(SteamNetAuthenticationStatus_t* call_data){
	// Status.
	int available = call_data->m_eAvail;
	// Non-localized English language status. For diagnostic / debugging purposes only.
	char *debug_message = new char[256];
	sprintf(debug_message, "%s", call_data->m_debugMsg);
	// Send the data back via signal
	emit_signal("network_authentication_status", available, debug_message);
	delete[] debug_message;
}

// A struct used to describe a "fake IP" we have been assigned to use as an identifier.
// This callback is posted when ISteamNetworkingSoockets::BeginAsyncRequestFakeIP completes.
void Steam::fake_ip_result(SteamNetworkingFakeIPResult_t* call_data){
	int result = call_data->m_eResult;
	// Pass this new networking identity to the map
	networking_identities["fake_ip_identity"] = call_data->m_identity;
	uint32 ip = call_data->m_unIP;
	// Convert the IP address back to a string
	const int NBYTES = 4;
	uint8 octet[NBYTES];
	char fake_ip[16];
	for(int i = 0; i < NBYTES; i++){
		octet[i] = ip >> (i * 8);
	}
	sprintf(fake_ip, "%d.%d.%d.%d", octet[0], octet[1], octet[2], octet[3]);
	// Get the ports as an array
	Array port_list;
	uint16* ports = call_data->m_unPorts;
	for(uint16 i = 0; i < sizeof(ports); i++){
		port_list.append(ports[i]);
	}
	emit_signal("fake_ip_result", result, "fake_ip_identity", fake_ip, port_list);
}

// NETWORKING UTILS CALLBACKS ///////////////////
//
//! A struct used to describe our readiness to use the relay network.
void Steam::relay_network_status(SteamRelayNetworkStatus_t* call_data){
	int available = call_data->m_eAvail;
	int ping_measurement = call_data->m_bPingMeasurementInProgress;
	int available_config = call_data->m_eAvailNetworkConfig;
	int available_relay = call_data->m_eAvailAnyRelay;
	char *debug_message = new char[256];
	sprintf(debug_message, "%s", call_data->m_debugMsg);
//	debug_message = call_data->m_debugMsg;
	emit_signal("relay_network_status", available, ping_measurement, available_config, available_relay, debug_message);
	delete[] debug_message;
}

// PARENTAL SETTINGS CALLBACKS //////////////////
//
//! Purpose: Callback for querying UGC
void Steam::parental_setting_changed(SteamParentalSettingsChanged_t* call_data){
	emit_signal("parental_setting_changed");
}

// PARTIES CALLBACKS ////////////////////////////
//
//! After creating a beacon, when a user "follows" that beacon Steam will send you this callback to know that you should be prepared for the user to join your game. When they do join, be sure to call ISteamParties::OnReservationCompleted to let Steam know.
void Steam::reservation_notification(ReservationNotificationCallback_t* call_data){
	uint64_t beacon_id = call_data->m_ulBeaconID;
	uint64_t steam_id = call_data->m_steamIDJoiner.ConvertToUint64();
	emit_signal("reservation_notifications", beacon_id, steam_id);
}

//! Notification that the list of available locations for posting a beacon has been updated. 
void Steam::available_beacon_locations_updated(AvailableBeaconLocationsUpdated_t* call_data){
	emit_signal("available_beacon_locations_updated");
}

//! Notification that the list of active beacons visible to the current user has changed. 
void Steam::active_beacons_updated(ActiveBeaconsUpdated_t* call_data){
	emit_signal("active_beacons_updated");
}

// REMOTE PLAY CALLBACKS ////////////////////////
//
//! The session ID of the session that just connected.
void Steam::remote_play_session_connected(SteamRemotePlaySessionConnected_t* call_data){
	uint32 session_id = call_data->m_unSessionID;
	emit_signal("remote_play_session_connected", session_id);
}

//! The session ID of the session that just disconnected.
void Steam::remote_play_session_disconnected(SteamRemotePlaySessionDisconnected_t* call_data){
	uint32 session_id = call_data->m_unSessionID;
	emit_signal("remote_play_session_disconnected", session_id);
}

// REMOTE STORAGE CALLBACKS /////////////////////
//
//! Purpose: one or more files for this app have changed locally after syncing to remote session changes.
//! Note: only posted if this happens DURING the local app session.
void Steam::local_file_changed(RemoteStorageLocalFileChange_t* call_data){
	emit_signal("local_file_changed");
}

// SCREENSHOT CALLBACKS /////////////////////////
//
//! A screenshot successfully written or otherwise added to the library and can now be tagged.
void Steam::screenshot_ready(ScreenshotReady_t* call_data){
	uint32_t handle = call_data->m_hLocal;
	uint32_t result = call_data->m_eResult;
	emit_signal("screenshot_ready", handle, result);
}

//! A screenshot has been requested by the user from the Steam screenshot hotkey. This will only be called if hookScreenshots has been enabled, in which case Steam will not take the screenshot itself.
void Steam::screenshot_requested(ScreenshotRequested_t* call_data){
	emit_signal("screenshot_requested");
}

// UGC CALLBACKS ////////////////////////////////
//
//! Called when a workshop item has been downloaded.
void Steam::item_downloaded(DownloadItemResult_t* call_data){
	EResult result = call_data->m_eResult;
	PublishedFileId_t file_id = call_data->m_nPublishedFileId;
	AppId_t app_id = call_data->m_unAppID;
	emit_signal("item_downloaded", result, (uint64_t)file_id, (uint32_t)app_id);
}

//! Called when a workshop item has been installed or updated.
void Steam::item_installed(ItemInstalled_t* call_data){
	AppId_t app_id = call_data->m_unAppID;
	PublishedFileId_t file_id = call_data->m_nPublishedFileId;
	emit_signal("item_installed", app_id, (uint64_t)file_id);
}

//! Purpose: signal that the list of subscribed items changed.
void Steam::user_subscribed_items_list_changed(UserSubscribedItemsListChanged_t* call_data){
	uint32 app_id = call_data->m_nAppID;
	emit_signal("user_subscribed_items_list_changed", app_id);
}

// USER CALLBACKS ///////////////////////////////
//
//! Sent by the Steam server to the client telling it to disconnect from the specified game server, which it may be in the process of or already connected to. The game client should immediately disconnect upon receiving this message. This can usually occur if the user doesn't have rights to play on the game server.
void Steam::client_game_server_deny(ClientGameServerDeny_t* call_data){
	uint32 app_id = call_data->m_uAppID;
	uint32 server_ip = call_data->m_unGameServerIP;
	uint16 server_port = call_data->m_usGameServerPort;
	uint16 secure = call_data->m_bSecure;
	uint32 reason = call_data->m_uReason;
	// Convert the IP address back to a string
	const int NBYTES = 4;
	uint8 octet[NBYTES];
	char ip[16];
	for(int j = 0; j < NBYTES; j++){
		octet[j] = server_ip >> (j * 8);
	}
	sprintf(ip, "%d.%d.%d.%d", octet[0], octet[1], octet[2], octet[3]);
	emit_signal("client_game_server_deny", app_id, ip, server_port, secure, reason);
}

//! Sent to your game in response to a steam://gamewebcallback/ command from a user clicking a link in the Steam overlay browser. You can use this to add support for external site signups where you want to pop back into the browser after some web page signup sequence, and optionally get back some detail about that.
void Steam::game_web_callback(GameWebCallback_t* call_data){
	String url = call_data->m_szURL;
	emit_signal("game_web_callback", url);
}

//! Result when creating an auth session ticket.
void Steam::get_auth_session_ticket_response(GetAuthSessionTicketResponse_t* call_data){
	uint32 auth_ticket = call_data->m_hAuthTicket;
	int result = call_data->m_eResult;
	emit_signal("get_auth_session_ticket_response", auth_ticket, result);
}

//! Called when the callback system for this client is in an error state (and has flushed pending callbacks). When getting this message the client should disconnect from Steam, reset any stored Steam state and reconnect. This usually occurs in the rare event the Steam client has some kind of fatal error.
void Steam::ipc_failure(IPCFailure_t *call_data){
	uint8 type = call_data->m_eFailureType;
	emit_signal("ipc_failure", type);
}

//! Called whenever the users licenses (owned packages) changes.
void Steam::licenses_updated(LicensesUpdated_t* call_data){
	emit_signal("licenses_updated");
}

//! Called when a user has responded to a microtransaction authorization request.
void Steam::microstransaction_auth_response(MicroTxnAuthorizationResponse_t *call_data){
	uint32 app_id = call_data->m_unAppID;
	uint64_t order_id = call_data->m_ulOrderID;
	bool authorized;
	if(call_data->m_bAuthorized == 1){
		authorized = true;
	}
	else{
		authorized = false;
	}
	emit_signal("microstransaction_auth_response", app_id, order_id, authorized);
}

//! Called when a connections to the Steam back-end has been established. This means the Steam client now has a working connection to the Steam servers. Usually this will have occurred before the game has launched, and should only be seen if the user has dropped connection due to a networking issue or a Steam server update.
void Steam::steam_server_connected(SteamServersConnected_t* connectData){
	emit_signal("steam_server_connected");
}

//! Called if the client has lost connection to the Steam servers. Real-time services will be disabled until a matching SteamServersConnected_t has been posted.
void Steam::steam_server_disconnected(SteamServersDisconnected_t* connectData){
	emit_signal("steam_server_disconnected");
}

//! Called when an auth ticket has been validated.
void Steam::validate_auth_ticket_response(ValidateAuthTicketResponse_t* call_data){
	uint64_t auth_id = call_data->m_SteamID.ConvertToUint64();
	uint32_t response = call_data->m_eAuthSessionResponse;
	uint64_t owner_id = call_data->m_OwnerSteamID.ConvertToUint64();
	emit_signal("validate_auth_ticket_response", auth_id, response, owner_id);
}

// USER STATS CALLBACKS /////////////////////////
//
//! Result of a request to store the achievements on the server, or an "indicate progress" call. If both m_nCurProgress and m_nMaxProgress are zero, that means the achievement has been fully unlocked.
void Steam::user_achievement_stored(UserAchievementStored_t* call_data){
	CSteamID game_id = call_data->m_nGameID;
	uint64_t game = game_id.ConvertToUint64();
	bool group_achieve = call_data->m_bGroupAchievement;
	String name = call_data->m_rgchAchievementName;
	uint32_t current_progress = call_data->m_nCurProgress;
	uint32_t max_progress = call_data->m_nMaxProgress;
	emit_signal("user_achievement_stored", game, group_achieve, name, current_progress, max_progress);
}

//! Called when the latest stats and achievements for the local user have been received from the server.
void Steam::current_stats_received(UserStatsReceived_t* call_data){
	CSteamID game_id = call_data->m_nGameID;
	uint64_t game = game_id.ConvertToUint64();
	uint32_t result = call_data->m_eResult;
	CSteamID user_id = call_data->m_steamIDUser;
	uint64_t user = user_id.ConvertToUint64();
	emit_signal("current_stats_received", game, result, user);
}

//! Result of a request to store the user stats.
void Steam::user_stats_stored(UserStatsStored_t* call_data){
	CSteamID game_id = call_data->m_nGameID;
	uint64_t game = game_id.ConvertToUint64();
	uint32_t result = call_data->m_eResult;
	emit_signal("user_stats_stored", game, result);
}

//! Callback indicating that a user's stats have been unloaded. Call RequestUserStats again before accessing stats for this user.
void Steam::user_stats_unloaded(UserStatsUnloaded_t* call_data){
	CSteamID steam_id = call_data->m_steamIDUser;
	uint64_t user = steam_id.ConvertToUint64();
	emit_signal("user_stats_unloaded", user);
}

// UTILITY CALLBACKS ////////////////////////////
//
//! Called when the big picture gamepad text input has been closed.
void Steam::gamepad_text_input_dismissed(GamepadTextInputDismissed_t* call_data){
	const uint32 buffer_length = 1024+1;
	char *text = new char[buffer_length];
	uint32 length = buffer_length;
	if(call_data->m_bSubmitted){
		SteamUtils()->GetEnteredGamepadTextInput(text, buffer_length);
		length = SteamUtils()->GetEnteredGamepadTextLength();
	}
	emit_signal("gamepad_text_input_dismissed", call_data->m_bSubmitted, String::utf8(text, (int)length));
	delete[] text;
}

//! Called when the country of the user changed. The country should be updated with getIPCountry.
void Steam::ip_country(IPCountry_t* call_data){
	emit_signal("ip_country");
}

//! Called when running on a laptop and less than 10 minutes of battery is left, and then fires then every minute afterwards.
void Steam::low_power(LowBatteryPower_t* timeLeft){
	uint8 power = timeLeft->m_nMinutesBatteryLeft;
	emit_signal("low_power", power);
}

//! Called when a SteamAPICall_t has completed (or failed)
void Steam::steam_api_call_completed(SteamAPICallCompleted_t* call_data){
	uint64_t async_call = call_data->m_hAsyncCall;
	int callback = call_data->m_iCallback;
	uint32 parameter = call_data->m_cubParam;
	emit_signal("steam_api_call_completed", async_call, callback, parameter);
}

//! Called when Steam wants to shutdown.
void Steam::steam_shutdown(SteamShutdown_t* call_data){
	emit_signal("steam_shutdown");
}

//! Sent after the device returns from sleep/suspend mode.
void Steam::app_resuming_from_suspend(AppResumingFromSuspend_t *call_data) {
	emit_signal("app_resuming_from_suspend");
}

//! Sent after the device returns from sleep/suspend mode.
void Steam::floating_gamepad_text_input_dismissed(FloatingGamepadTextInputDismissed_t *call_data) {
	emit_signal("floating_gamepad_text_input_dismissed");
}

// VIDEO CALLBACKS //////////////////////////////
//
//! Triggered when the OPF Details for 360 video playback are retrieved. After receiving this you can use GetOPFStringForApp to access the OPF details.
void Steam::get_opf_settings_result(GetOPFSettingsResult_t* call_data){
	int result = call_data->m_eResult;
	uint32 app_id = call_data->m_unVideoAppID;
	emit_signal("broadcast_upload_stop", result, app_id);
}

//! Provides the result of a call to GetVideoURL.
void Steam::get_video_result(GetVideoURLResult_t* call_data){
	int result = call_data->m_eResult;
	uint32 app_id = call_data->m_unVideoAppID;
	String url = call_data->m_rgchURL;
	emit_signal("get_video_result", result, app_id, url);
}


/////////////////////////////////////////////////
///// SIGNALS / CALL RESULTS ////////////////////
/////////////////////////////////////////////////
//
// STEAMWORKS ERROR SIGNAL //////////////////////
//
//! Intended to serve as generic error messaging for failed call results
void Steam::steamworksError(const String& failed_signal){
	// Emit the signal to inform the user of the failure
	emit_signal("steamworks_error", failed_signal, "io failure");
}

// FRIENDS CALL RESULTS /////////////////////////
//
//! Marks the return of a request officer list call.
void Steam::request_clan_officer_list(ClanOfficerListResponse_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("request_clan_officer_list");
	}
	else{
		Array officers_list;
		String message;
		if(!call_data->m_bSuccess){
			message = "Clan officer list response failed.";
		}
		else{
			CSteamID ownerSteamID = SteamFriends()->GetClanOwner(call_data->m_steamIDClan);
			int officers = SteamFriends()->GetClanOfficerCount(call_data->m_steamIDClan);
			message = "The owner of the clan is: " + (String)String::utf8(SteamFriends()->GetFriendPersonaName(ownerSteamID)) + " (" + itos(ownerSteamID.ConvertToUint64()) + ") and there are " + itos(call_data->m_cOfficers) + " officers.";
			for(int i = 0; i < officers; i++){
				Dictionary officer;
				CSteamID officerSteamID = SteamFriends()->GetClanOfficerByIndex(call_data->m_steamIDClan, i);
				uint64_t id = officerSteamID.ConvertToUint64();
				officer["id"] = id;
				officer["name"] = String::utf8(SteamFriends()->GetFriendPersonaName(officerSteamID));
				officers_list.append(officer);
			}
		}
		emit_signal("request_clan_officer_list", message, officers_list);
	}
}

//! Returns the result of enumerateFollowingList.
void Steam::enumerate_following_list(FriendsEnumerateFollowingList_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("enumerate_following_list");
	}
	else{
		Array following;
		String message;
		if(call_data->m_eResult != k_EResultOK){
			message = "Failed to acquire list.";
		}
		else{
			int followers_parsed = 0;
			message = "Retrieved " + itos(call_data->m_nResultsReturned) + " of " + itos(call_data->m_nTotalResultCount) + " people followed.";
			int32 count = call_data->m_nTotalResultCount;
			for(int i = 0; i < count; i++){
				Dictionary follow;
				int num = i;
				uint64_t id = call_data->m_rgSteamID[i].ConvertToUint64();
				follow["num"] = num;
				follow["id"] = id;
				following.append(follow);
			}
			followers_parsed += call_data->m_nResultsReturned;
			// There are more followers so make another callback.
			if(followers_parsed < count){
				SteamAPICall_t api_call = SteamFriends()->EnumerateFollowingList(call_data->m_nResultsReturned);
				callResultEnumerateFollowingList.Set(api_call, this, &Steam::enumerate_following_list);
			}
		}
		emit_signal("enumerate_following_list", message, following);
	}
}

//! Returns the result of getFollowerCount.
void Steam::get_follower_count(FriendsGetFollowerCount_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("get_follower_count");
	}
	else{
		EResult result = call_data->m_eResult;
		uint64_t steam_id = call_data->m_steamID.ConvertToUint64();
		int count = call_data->m_nCount;
		emit_signal("follower_count", result, steam_id, count);
	}
}

//! Returns the result of isFollowing.
void Steam::is_following(FriendsIsFollowing_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("is_following");
	}
	else{
		EResult result = call_data->m_eResult;
		uint64_t steam_id = call_data->m_steamID.ConvertToUint64();
		bool following = call_data->m_bIsFollowing;
		emit_signal("is_following", result, steam_id, following);
	}
}

// INVENTORY CALL RESULTS ///////////////////////
//
//! Returned when you have requested the list of "eligible" promo items that can be manually granted to the given user. These are promo items of type "manual" that won't be granted automatically.
void Steam::inventory_eligible_promo_item(SteamInventoryEligiblePromoItemDefIDs_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("inventory_eligible_promo_item");
	}
	else{
		// Clean up call data
		CSteamID steam_id = call_data->m_steamID;
		int result = call_data->m_result;
		int eligible = call_data->m_numEligiblePromoItemDefs;
		bool cached = call_data->m_bCachedData;
		// Create the return array
		Array definitions;
		// Create the temporary ID array
		SteamItemDef_t *id_array = new SteamItemDef_t[eligible];
		// Convert eligible size
		uint32 array_size = (int)eligible;
		// Get the list
		if(SteamInventory()->GetEligiblePromoItemDefinitionIDs(steam_id, id_array, &array_size)){
			// Loop through the temporary array and populate the return array
			for(int i = 0; i < eligible; i++){
				definitions.append(id_array[i]);
			}
		}
		// Delete the temporary array
		delete[] id_array;
		// Return the item array as a signal
		emit_signal("inventory_eligible_promo_Item", result, cached, definitions);
	}
}

//! Returned after StartPurchase is called.
void Steam::inventory_start_purchase_result(SteamInventoryStartPurchaseResult_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("inventory_start_purchase_result");
	}
	else{
		if(call_data->m_result == k_EResultOK){
			uint64_t order_id = call_data->m_ulOrderID;
			uint64_t transaction_id = call_data->m_ulTransID;
			emit_signal("inventory_start_purchase_result", "success", order_id, transaction_id);
		}
		else{
			emit_signal("inventory_start_purchase_result", "failure", 0, 0);
		}
	}
}

//! Returned after RequestPrices is called.
void Steam::inventory_request_prices_result(SteamInventoryRequestPricesResult_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("inventory_request_prices_result");
	}
	else{
		int result = call_data->m_result;
		String currency = call_data->m_rgchCurrency;
		emit_signal("inventory_request_prices_result", result, currency);
	}
}

// MATCHMAKING CALL RESULTS /////////////////////
//
//! Signal the lobby has been created.
void Steam::lobby_created(LobbyCreated_t *lobbyData, bool io_failure){
	if(io_failure){
		steamworksError("lobby_created");
	}
	else{
		int connect = lobbyData->m_eResult;
		CSteamID lobby_id = lobbyData->m_ulSteamIDLobby;
		uint64_t lobby = lobby_id.ConvertToUint64();
		emit_signal("lobby_created", connect, lobby);
	}
}

//! Result when requesting the lobby list. You should iterate over the returned lobbies with getLobbyByIndex, from 0 to m_nLobbiesMatching-1.
void Steam::lobby_match_list(LobbyMatchList_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("lobby_match_list");
	}
	else{
		int lobby_count = call_data->m_nLobbiesMatching;
		Array lobbies;
		for(int i = 0; i < lobby_count; i++){
			CSteamID lobby_id = SteamMatchmaking()->GetLobbyByIndex(i);
			uint64_t lobby = lobby_id.ConvertToUint64();
			lobbies.append(lobby);
		}
		emit_signal("lobby_match_list", lobbies);
	}
}

// MATCHMAKING SERVER CALL RESULTS //////////////
//
void Steam::server_Responded(gameserveritem_t server){
	emit_signal("server_responded");
}

//
void Steam::server_Failed_To_Respond(){
	emit_signal("server_failed_to_respond");
}

// PARTIES CALL RESULTS /////////////////////////
//
//! This callback is used as a call response for ISteamParties::JoinParty. On success, you will have reserved a slot in the beacon-owner's party, and should use m_rgchConnectString to connect to their game and complete the process.
void Steam::join_party(JoinPartyCallback_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("join_party");
	}
	else{
		int result = call_data->m_eResult;
		uint64_t beacon_id = call_data->m_ulBeaconID;
		uint64_t steam_id = call_data->m_SteamIDBeaconOwner.ConvertToUint64();
		String connect_string = call_data->m_rgchConnectString;
		emit_signal("join_party", result, beacon_id, steam_id, connect_string);
	}
}

//! This callback is used as a call response for ISteamParties::CreateBeacon. If successful, your beacon has been posted in the desired location and you may start receiving ISteamParties::ReservationNotificationCallback_t callbacks for users following the beacon. 
void Steam::create_beacon(CreateBeaconCallback_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("create_beacon");
	}
	else{
		int result = call_data->m_eResult;
		uint64_t beacon_id = call_data->m_ulBeaconID;
		emit_signal("create_beacon", result, beacon_id);
	}
}

//! Call result for ISteamParties::ChangeNumOpenSlots. 
void Steam::change_num_open_slots(ChangeNumOpenSlotsCallback_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("change_num_open_slots");
	}
	else{
		int result = call_data->m_eResult;
		emit_signal("change_num_open_slots", result);
	}
}

// REMOTE STORAGE CALL RESULTS //////////////////
//
//! Response when reading a file asyncrounously with FileReadAsync.
void Steam::file_read_async_complete(RemoteStorageFileReadAsyncComplete_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("file_read_async_complete");
	}
	else{
		uint64_t handle = call_data->m_hFileReadAsync;
		int result = call_data->m_eResult;
		uint32 offset = call_data->m_nOffset;
		uint32 read = call_data->m_cubRead;
		// Was read complete?
		PoolByteArray buffer;
		buffer.resize(read);
		bool complete = SteamRemoteStorage()->FileReadAsyncComplete(handle, buffer.write().ptr(), read);
		// Create a dictionary and populate it with the results
		Dictionary file_read;
		file_read["result"] = result;
		file_read["handle"] = handle;
		file_read["buffer"] = buffer;
		file_read["offset"] = offset;
		file_read["read"] = read;
		file_read["complete"] = complete;
		emit_signal("file_read_async_complete", file_read);
	}
}

//! Response to a file being shared.
void Steam::file_share_result(RemoteStorageFileShareResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("file_share_result");
	}
	else{
		int result = call_data->m_eResult;
		uint64_t handle = call_data->m_hFile;
		char name[k_cchFilenameMax];
		strcpy(name, call_data->m_rgchFilename);
		emit_signal("file_share_result", result, handle, name);
	}
}

//! Response when writing a file asyncrounously with FileWriteAsync.
void Steam::file_write_async_complete(RemoteStorageFileWriteAsyncComplete_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("file_write_async_complete");
	}
	else{
		int result = call_data->m_eResult;
		emit_signal("file_write_async_complete", result);
	}
}

//! Response when downloading UGC
void Steam::download_ugc_result(RemoteStorageDownloadUGCResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("download_ugc_result");
	}
	else{
		int result = call_data->m_eResult;
		uint64_t handle = call_data->m_hFile;
		uint32_t app_id = call_data->m_nAppID;
		int32 size = call_data->m_nSizeInBytes;
		char filename[k_cchFilenameMax];
		strcpy(filename, call_data->m_pchFileName);
		uint64_t owner_id = call_data->m_ulSteamIDOwner;
		// Pass some variable to download dictionary to bypass argument limit
		Dictionary download_data;
		download_data["handle"] = handle;
		download_data["app_id"] = app_id;
		download_data["size"] = size;
		download_data["filename"] = filename;
		download_data["owner_id"] = owner_id;
		emit_signal("download_ugc_result", result, download_data);
	}
}

//! Called when the user has unsubscribed from a piece of UGC. Result from ISteamUGC::UnsubscribeItem.
void Steam::unsubscribe_item(RemoteStorageUnsubscribePublishedFileResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("unsubscribe_item");
	}
	else{
		int result = call_data->m_eResult;
		int file_id = call_data->m_nPublishedFileId;
		emit_signal("unsubscribe_item", result, file_id);
	}
}

//! Called when the user has subscribed to a piece of UGC. Result from ISteamUGC::SubscribeItem.
void Steam::subscribe_item(RemoteStorageSubscribePublishedFileResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("subscribe_item");
	}
	else{
		int result = call_data->m_eResult;
		int file_id = call_data->m_nPublishedFileId;
		emit_signal("subscribe_item", result, file_id);
	}
}

// UGC CALL RESULTS /////////////////////////////
//
//! The result of a call to AddAppDependency.
void Steam::add_app_dependency_result(AddAppDependencyResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("add_app_dependency_result");
	}
	else{
		EResult result = call_data->m_eResult;
		PublishedFileId_t file_id = call_data->m_nPublishedFileId;
		AppId_t app_id = call_data->m_nAppID;
		emit_signal("add_app_dependency_result", result, (uint64_t)file_id, (uint32_t)app_id);
	}
}

//! The result of a call to AddDependency.
void Steam::add_ugc_dependency_result(AddUGCDependencyResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("add_ugc_dependency_result");
	}
	else{
		EResult result = call_data->m_eResult;
		PublishedFileId_t file_id = call_data->m_nPublishedFileId;
		PublishedFileId_t child_id = call_data->m_nChildPublishedFileId;
		emit_signal("add_ugc_dependency_result", result, (uint64_t)file_id, (uint64_t)child_id);
	}
}

//! Result of a workshop item being created.
void Steam::item_created(CreateItemResult_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("item_created");
	}
	else{
		EResult result = call_data->m_eResult;
		PublishedFileId_t file_id = call_data->m_nPublishedFileId;
		bool accept_tos = call_data->m_bUserNeedsToAcceptWorkshopLegalAgreement;
		emit_signal("item_created", result, (uint64_t)file_id, accept_tos);
	}
}

//! Called when getting the app dependencies for an item.
void Steam::get_app_dependencies_result(GetAppDependenciesResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("get_app_dependencies_result");
	}
	else{
		EResult result = call_data->m_eResult;
		PublishedFileId_t file_id = call_data->m_nPublishedFileId;
//		AppId_t app_id = call_data->m_rgAppIDs;
		uint32 app_dependencies = call_data->m_nNumAppDependencies;
		uint32 total_app_dependencies = call_data->m_nTotalNumAppDependencies;
//		emit_signal("get_app_dependencies_result", result, (uint64_t)file_id, app_id, appDependencies, totalAppDependencies);
		emit_signal("get_app_dependencies_result", result, (uint64_t)file_id, app_dependencies, total_app_dependencies);
	}
}

//! Called when an attempt at deleting an item completes.
void Steam::item_deleted(DeleteItemResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("item_deleted");
	}
	else{
		EResult result = call_data->m_eResult;
		PublishedFileId_t file_id = call_data->m_nPublishedFileId;
		emit_signal("item_deleted", result, (uint64_t)file_id);
	}
}

//! Called when getting the users vote status on an item.
void Steam::get_item_vote_result(GetUserItemVoteResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("get_item_vote_result");
	}
	else{
		EResult result = call_data->m_eResult;
		PublishedFileId_t file_id = call_data->m_nPublishedFileId;
		bool vote_up = call_data->m_bVotedUp;
		bool vote_down = call_data->m_bVotedDown;
		bool vote_skipped = call_data->m_bVoteSkipped;
		emit_signal("get_item_vote_result", result, (uint64_t)file_id, vote_up, vote_down, vote_skipped);
	}
}

//! Purpose: The result of a call to RemoveAppDependency.
void Steam::remove_app_dependency_result(RemoveAppDependencyResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("remove_app_dependency_result");
	}
	else{
		EResult result = call_data->m_eResult;
		PublishedFileId_t file_id = call_data->m_nPublishedFileId;
		AppId_t app_id = call_data->m_nAppID;
		emit_signal("remove_app_dependency_result", result, (uint64_t)file_id, (uint32_t)app_id);
	}
}

//! Purpose: The result of a call to RemoveDependency.
void Steam::remove_ugc_dependency_result(RemoveUGCDependencyResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("remove_ugc_dependency_result");
	}
	else{
		EResult result = call_data->m_eResult;
		PublishedFileId_t file_id = call_data->m_nPublishedFileId;
		PublishedFileId_t child_id = call_data->m_nChildPublishedFileId;
		emit_signal("remove_ugc_dependency_result", result, (uint64_t)file_id, (uint64_t)child_id);
	}
}

//! Called when the user has voted on an item.
void Steam::set_user_item_vote(SetUserItemVoteResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("set_user_item_vote");
	}
	else{
		EResult result = call_data->m_eResult;
		PublishedFileId_t file_id = call_data->m_nPublishedFileId;
		bool vote_up = call_data->m_bVoteUp;
		emit_signal("set_user_item_vote", result, (uint64_t)file_id, vote_up);
	}
}

//! Called when workshop item playtime tracking has started.
void Steam::start_playtime_tracking(StartPlaytimeTrackingResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("start_playtime_tracking");
	}
	else{
		EResult result = call_data->m_eResult;
		emit_signal("start_playtime_tracking", result);
	}
}

//! Called when a UGC query request completes.
void Steam::ugc_query_completed(SteamUGCQueryCompleted_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("ugc_query_completed");
	}
	else{
		UGCQueryHandle_t handle = call_data->m_handle;
		EResult result = call_data->m_eResult;
		uint32 results_returned = call_data->m_unNumResultsReturned;
		uint32 total_matching = call_data->m_unTotalMatchingResults;
		bool cached = call_data->m_bCachedData;
		emit_signal("ugc_query_completed", (uint64_t)handle, result, results_returned, total_matching, cached);
	}
}

//! Called when workshop item playtime tracking has stopped.
void Steam::stop_playtime_tracking(StopPlaytimeTrackingResult_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("stop_playtime_tracking");
	}
	else{
		EResult result = call_data->m_eResult;
		emit_signal("stop_playtime_tracking", result);
	}
}

//! Result of a workshop item being updated.
void Steam::item_updated(SubmitItemUpdateResult_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("item_updated");
	}
	else{
		EResult result = call_data->m_eResult;
		bool accept_tos = call_data->m_bUserNeedsToAcceptWorkshopLegalAgreement;
		emit_signal("item_updated", result, accept_tos);
	}
}

//! Called when the user has added or removed an item to/from their favorites.
void Steam::user_favorite_items_list_changed(UserFavoriteItemsListChanged_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("user_favorite_items_list_changed");
	}
	else{
		EResult result = call_data->m_eResult;
		PublishedFileId_t file_id = call_data->m_nPublishedFileId;
		bool was_add_request = call_data->m_bWasAddRequest;
		emit_signal("user_favorite_items_list_changed", result, (uint64_t)file_id, was_add_request);
	}
}

//! Purpose: Status of the user's acceptable/rejection of the app's specific Workshop EULA.
void Steam::workshop_eula_status(WorkshopEULAStatus_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("workshop_eula_status");
	}
	else{
		int result = call_data->m_eResult;
		uint32 app_id = call_data->m_nAppID;
		// Slim down signal arguments since Godot seems to limit them to six max
		Dictionary eula_data;
		eula_data["version"] = call_data->m_unVersion;			// int
		eula_data["action"] = call_data->m_rtAction;			// int
		eula_data["accepted"] = call_data->m_bAccepted;			// bool
		eula_data["needs_action"] = call_data->m_bNeedsAction;	// bool
		emit_signal("workshop_eula_status", result, app_id, eula_data);
	}
}

// USERS CALL RESULTS ///////////////////////////
//
//! Sent for games with enabled anti indulgence / duration control, for enabled users. Lets the game know whether persistent rewards or XP should be granted at normal rate, half rate, or zero rate.
void Steam::duration_control(DurationControl_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("duration_control");
	}
	else{
		int result = call_data->m_eResult;
		uint32 app_id = call_data->m_appid;
		bool applicable = call_data->m_bApplicable;
		int32 seconds_last = call_data->m_csecsLast5h;
		int progress = call_data->m_progress;
		int notification = call_data->m_notification;
		String verbal = "";
		// Get a more verbal response
		if(notification == 1){
			verbal = "you've been playing for an hour";
		}
		else if(notification == 2){
			verbal = "you've been playing for 3 hours; take a break";
		}
		else if(notification == 3){
			verbal = "your xp / progress is half normal";
		}
		else if(notification == 4){
			verbal = "your xp / progress is zero";
		}
		else{
			verbal = "no notification";
		}
		// Create dictionary due to "too many arguments" issue
		Dictionary duration;
		duration["app_id"] = app_id;
		duration["applicable"] = applicable;
		duration["seconds_last_5hrs"] = seconds_last;
		duration["progress"] = progress;
		duration["notification"] = notification;
		duration["notification_verbal"] = verbal;
		emit_signal("duration_control", result, duration);
	}
}

//! Called when an encrypted application ticket has been received.
void Steam::encrypted_app_ticket_response(EncryptedAppTicketResponse_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("encrypted_app_ticket_response");
	}
	else{
		String result;
		if(call_data->m_eResult == k_EResultOK){
			result = "ok";
		}
		else if(call_data->m_eResult == k_EResultNoConnection){
			result = "no connection";
		}
		else if(call_data->m_eResult == k_EResultDuplicateRequest){
			result = "duplicate request";
		}
		else{
			result = "limit exceeded";
		}
		emit_signal("encrypted_app_ticket_response", result);
	}
}

//! Called when a connection attempt has failed. This will occur periodically if the Steam client is not connected, and has failed when retrying to establish a connection.
void Steam::steam_server_connect_failed(SteamServerConnectFailure_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("steam_server_connect_failed");
	}
	else{
		int result = call_data->m_eResult;
		bool retrying = call_data->m_bStillRetrying;
		emit_signal("steam_server_connected_failed", result, retrying);
	}
}

//! Response when we have recieved the authentication URL after a call to requestStoreAuthURL.
void Steam::store_auth_url_response(StoreAuthURLResponse_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("store_auth_url_response");
	}
	else{
		String url = call_data->m_szURL;
		emit_signal("store_auth_url_response", url);
	}
}

// USER STATS CALL RESULTS //////////////////////
//
//! Global achievements percentages are ready.
void Steam::global_achievement_percentages_ready(GlobalAchievementPercentagesReady_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("global_achievement_percentages_ready");
	}
	else{
		CSteamID game_id = call_data->m_nGameID;
		uint64_t game = game_id.ConvertToUint64();
		uint32_t result = call_data->m_eResult;
		emit_signal("global_achievement_percentages_ready", game, result);
	}
}

//! Called when the global stats have been received from the server.
void Steam::global_stats_received(GlobalStatsReceived_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("global_stats_received");
	}
	else{
		uint64_t game_id = call_data->m_nGameID;
		String result;
		if(call_data->m_eResult == k_EResultOK){
			result = "ok";
		}
		else if(call_data->m_eResult == k_EResultInvalidState){
			result = "invalid";
		}
		else{
			result = "fail";
		}
		emit_signal("global_stats_received", game_id, result);
	}
}

//! Result when finding a leaderboard.
void Steam::leaderboard_find_result(LeaderboardFindResult_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("leaderboard_find_result");
	}
	else{
		leaderboard_handle = call_data->m_hSteamLeaderboard;
		uint8_t found = call_data->m_bLeaderboardFound;
		emit_signal("leaderboard_find_result", (uint64_t)leaderboard_handle, found);
	}
}

//! Called when scores for a leaderboard have been downloaded and are ready to be retrieved. After calling you must use GetDownloadedLeaderboardEntry to retrieve the info for each downloaded entry.
void Steam::leaderboard_scores_downloaded(LeaderboardScoresDownloaded_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("leaderboard_scores_downloaded");
	}
	else{
		// Set up a message to fill in
		String message;
		// Incorrect leaderboard
		if(call_data->m_hSteamLeaderboard != leaderboard_handle){
			message = "Leaderboard handle was incorrect";
		}
		// Clear previous leaderboard entries
		leaderboard_entries_array.clear();
		// Create the entry pointer and details array
		LeaderboardEntry_t *entry = memnew(LeaderboardEntry_t);
		PoolIntArray details;
		int32 *details_pointer = NULL;
		// Resize array
		if(leaderboard_details_max > 0){
			details.resize(leaderboard_details_max);
			PoolIntArray::Write w = details.write();
			details_pointer = w.ptr();
			for(int i = 0; i < leaderboard_details_max; i++){
				details_pointer[i] = 0;
			}
		}
		// Loop through the entries and add them as dictionaries to the array
		for(int i = 0; i < call_data->m_cEntryCount; i++){
			if(SteamUserStats()->GetDownloadedLeaderboardEntry(call_data->m_hSteamLeaderboardEntries, i, entry, details_pointer, leaderboard_details_max)){
				Dictionary entry_dict;
				entry_dict["score"] = entry->m_nScore;
				entry_dict["steam_id"] = uint64_t(entry->m_steamIDUser.ConvertToUint64());
				entry_dict["global_rank"] = entry->m_nGlobalRank;
				entry_dict["ugc_handle"] = uint64_t(entry->m_hUGC);
				if(leaderboard_details_max > 0){
					PoolIntArray array;
					array.resize(leaderboard_details_max);
					PoolIntArray::Write w = array.write();
					int32_t *ptr = w.ptr();
					for(int j = 0; j < leaderboard_details_max; j++){
						ptr[j] = details_pointer[j];
					}
					entry_dict["details"] = array;
				}
				leaderboard_entries_array.append(entry_dict);
			}
			message = "Leaderboard entries successfully retrieved";
		}
		memdelete(entry);
		// Emit the signal, with array, back
		emit_signal("leaderboard_scores_downloaded", message, leaderboard_entries_array);
	}
}

//! Result indicating that a leaderboard score has been uploaded.
void Steam::leaderboard_score_uploaded(LeaderboardScoreUploaded_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("leaderboard_scores_uploaded");
	}
	else{
		// Incorrect leaderboard
		if(call_data->m_hSteamLeaderboard != leaderboard_handle){
			return;
		}
		uint8 success = call_data->m_bSuccess;
		int32 score = call_data->m_nScore;
		uint8 score_changed = call_data->m_bScoreChanged;
		int global_rank_new = call_data->m_nGlobalRankNew;
		int global_rank_prev = call_data->m_nGlobalRankPrevious;
		emit_signal("leaderboard_score_uploaded", success, score, score_changed, global_rank_new, global_rank_prev);
	}
}

//! Result indicating that user generated content has been attached to one of the current user's leaderboard entries.
void Steam::leaderboard_ugc_set(LeaderboardUGCSet_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("leaderboard_ugc_set");
	}
	else{
		leaderboard_handle = call_data->m_hSteamLeaderboard;
		String result;
		if(call_data->m_eResult == k_EResultOK){
			result = "ok";
		}
		else if(call_data->m_eResult == k_EResultTimeout){
			result = "timeout";
		}
		else{
			result = "invalid";
		}
		emit_signal("leaderboard_ugc_set", (uint64_t)leaderboard_handle, result);
	}
}

//! Gets the current number of players for the current AppId.
void Steam::number_of_current_players(NumberOfCurrentPlayers_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("number_of_current_players");
	}
	else{
		uint8 success = call_data->m_bSuccess;
		int32 players = call_data->m_cPlayers;
		emit_signal("number_of_current_players", success, players);
	}
}

//! Called when the latest stats and achievements for a specific user (including the local user) have been received from the server.
void Steam::user_stats_received(UserStatsReceived_t* call_data, bool io_failure){
	if(io_failure){
		steamworksError("user_stats_received");
	}
	else{
		CSteamID game_id = call_data->m_nGameID;
		uint64_t game = game_id.ConvertToUint64();
		uint32_t result = call_data->m_eResult;
		CSteamID user_id = call_data->m_steamIDUser;
		uint64_t user = user_id.ConvertToUint64();
		emit_signal("user_stats_received", game, result, user);
	}
}

// UTILITY CALL RESULTS /////////////////////////
//
//! CallResult for checkFileSignature.
void Steam::check_file_signature(CheckFileSignature_t *call_data, bool io_failure){
	if(io_failure){
		steamworksError("check_file_signature");
	}
	else{
		String signature;
		if(call_data->m_eCheckFileSignature == k_ECheckFileSignatureNoSignaturesFoundForThisApp){
			signature = "app not signed";
		}
		else if(call_data->m_eCheckFileSignature == k_ECheckFileSignatureNoSignaturesFoundForThisFile){
			signature = "file not signed";
		}
		else if(call_data->m_eCheckFileSignature == k_ECheckFileSignatureFileNotFound){
			signature = "file does not exist";		
		}
		else if(call_data->m_eCheckFileSignature == k_ECheckFileSignatureInvalidSignature){
			signature = "signature invalid";
		}
		else if(call_data->m_eCheckFileSignature == k_ECheckFileSignatureValidSignature){
			signature = "valid";		
		}
		else{
			signature = "invalid response";
		}
		emit_signal("check_file_signature", signature);
	}
}


/////////////////////////////////////////////////
///// BIND METHODS
/////////////////////////////////////////////////
//
void Steam::_bind_methods(){
	/////////////////////////////////////////////
	// FUNCTION BINDS
	/////////////////////////////////////////////
	//
	// STEAM MAIN BIND METHODS //////////////////
	ClassDB::bind_method("run_callbacks", &Steam::run_callbacks);
	ClassDB::bind_method(D_METHOD("restartAppIfNecessary", "app_id"), &Steam::restartAppIfNecessary);
	ClassDB::bind_method(D_METHOD("steamInit", "retrieve_stats"), &Steam::steamInit, DEFVAL(true));
	ClassDB::bind_method("isSteamRunning", &Steam::isSteamRunning);
	ClassDB::bind_method(D_METHOD("serverInit", "ip", "game_port", "query_port", "server_mode", "version_string"), &Steam::serverInit);
	ClassDB::bind_method("serverReleaseCurrentThreadMemory", &Steam::serverReleaseCurrentThreadMemory);
	ClassDB::bind_method("serverShutdown", &Steam::serverShutdown);

	// APPS BIND METHODS ////////////////////////
	ClassDB::bind_method("getDLCDataByIndex", &Steam::getDLCDataByIndex);
	ClassDB::bind_method(D_METHOD("isAppInstalled", "app_id"), &Steam::isAppInstalled);
	ClassDB::bind_method("isCybercafe", &Steam::isCybercafe);
	ClassDB::bind_method(D_METHOD("isDLCInstalled", "dlc_id"), &Steam::isDLCInstalled);
	ClassDB::bind_method("isLowViolence", &Steam::isLowViolence);
	ClassDB::bind_method("isSubscribed", &Steam::isSubscribed);
	ClassDB::bind_method(D_METHOD("isSubscribedApp", "app_id"), &Steam::isSubscribedApp);
	ClassDB::bind_method("isSubscribedFromFamilySharing", &Steam::isSubscribedFromFamilySharing);
	ClassDB::bind_method("isSubscribedFromFreeWeekend", &Steam::isSubscribedFromFreeWeekend);
	ClassDB::bind_method("isTimedTrial", &Steam::isTimedTrial);
	ClassDB::bind_method("isVACBanned", &Steam::isVACBanned);
	ClassDB::bind_method("getAppBuildId", &Steam::getAppBuildId);
	ClassDB::bind_method(D_METHOD("getAppInstallDir", "app_id"), &Steam::getAppInstallDir);
	ClassDB::bind_method("getAppOwner", &Steam::getAppOwner);
	ClassDB::bind_method("getAvailableGameLanguages", &Steam::getAvailableGameLanguages);
	ClassDB::bind_method("getCurrentBetaName", &Steam::getCurrentBetaName);
	ClassDB::bind_method("getCurrentGameLanguage", &Steam::getCurrentGameLanguage);
	ClassDB::bind_method("getDLCCount", &Steam::getDLCCount);
	ClassDB::bind_method(D_METHOD("getDLCDownloadProgress", "dlc_id"), &Steam::getDLCDownloadProgress);
	ClassDB::bind_method(D_METHOD("getEarliestPurchaseUnixTime", "app_id"), &Steam::getEarliestPurchaseUnixTime);
	ClassDB::bind_method(D_METHOD("getFileDetails", "filename"), &Steam::getFileDetails);
	ClassDB::bind_method(D_METHOD("getInstalledDepots", "app_id"), &Steam::getInstalledDepots);
	ClassDB::bind_method("getLaunchCommandLine", &Steam::getLaunchCommandLine);
	ClassDB::bind_method(D_METHOD("getLaunchQueryParam", "key"), &Steam::getLaunchQueryParam);
	ClassDB::bind_method(D_METHOD("installDLC", "dlc_id"), &Steam::installDLC);
	ClassDB::bind_method(D_METHOD("markContentCorrupt", "missing_files_only"), &Steam::markContentCorrupt);
	ClassDB::bind_method(D_METHOD("uninstallDLC", "dlc_id"), &Steam::uninstallDLC);

	// APP LIST BIND METHODS ////////////////////
	ClassDB::bind_method("getNumInstalledApps", &Steam::getNumInstalledApps);
	ClassDB::bind_method(D_METHOD("getInstalledApps", "max_app_ids"), &Steam::getInstalledApps);
	ClassDB::bind_method(D_METHOD("getAppName", "app_id", "name_max"), &Steam::getAppName);
	ClassDB::bind_method(D_METHOD("getAppListInstallDir", "app_id", "name_max"), &Steam::getAppListInstallDir);
	ClassDB::bind_method(D_METHOD("getAppListBuildId", "app_id"), &Steam::getAppListInstallDir);

	// FRIENDS BIND METHODS /////////////////////
	ClassDB::bind_method(D_METHOD("activateGameOverlay", "type"), &Steam::activateGameOverlay, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("activateGameOverlayInviteDialog", "steam_id"), &Steam::activateGameOverlayInviteDialog);
	ClassDB::bind_method(D_METHOD("activateGameOverlayInviteDialogConnectString", "connect_string"), &Steam::activateGameOverlayInviteDialogConnectString);
	ClassDB::bind_method(D_METHOD("activateGameOverlayToStore", "app_id"), &Steam::activateGameOverlayToStore, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("activateGameOverlayToUser", "type", "steam_id"), &Steam::activateGameOverlayToUser, DEFVAL(""), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("activateGameOverlayToWebPage", "url"), &Steam::activateGameOverlayToWebPage);
	ClassDB::bind_method("clearRichPresence", &Steam::clearRichPresence);
	ClassDB::bind_method(D_METHOD("closeClanChatWindowInSteam", "chat_id"), &Steam::closeClanChatWindowInSteam);
	ClassDB::bind_method(D_METHOD("downloadClanActivityCounts", "chat_id", "clans_to_request"), &Steam::downloadClanActivityCounts);
	ClassDB::bind_method(D_METHOD("enumerateFollowingList", "start_index"), &Steam::enumerateFollowingList);
	ClassDB::bind_method(D_METHOD("getChatMemberByIndex", "clan_id", "user"), &Steam::getChatMemberByIndex);
	ClassDB::bind_method(D_METHOD("getClanActivityCounts", "clan_id"), &Steam::getClanActivityCounts);
	ClassDB::bind_method(D_METHOD("getClanByIndex", "clan"), &Steam::getClanByIndex);
	ClassDB::bind_method(D_METHOD("getClanChatMemberCount", "clan_id"), &Steam::getClanChatMemberCount);
	ClassDB::bind_method(D_METHOD("getClanChatMessage", "chat_id", "message"), &Steam::getClanChatMessage);
	ClassDB::bind_method("getClanCount", &Steam::getClanCount);
	ClassDB::bind_method(D_METHOD("getClanName", "clan_id"), &Steam::getClanName);
	ClassDB::bind_method(D_METHOD("getClanOfficerByIndex", "clan_id", "officer"), &Steam::getClanOfficerByIndex);
	ClassDB::bind_method(D_METHOD("getClanOfficerCount", "clan_id"), &Steam::getClanOfficerCount);
	ClassDB::bind_method(D_METHOD("getClanOwner", "clan_id"), &Steam::getClanOwner);
	ClassDB::bind_method(D_METHOD("getClanTag", "clan_id"), &Steam::getClanTag);
	ClassDB::bind_method(D_METHOD("getCoplayFriend", "friend_number"), &Steam::getCoplayFriend);
	ClassDB::bind_method("getCoplayFriendCount", &Steam::getCoplayFriendCount);
	ClassDB::bind_method(D_METHOD("getFollowerCount", "steam_id"), &Steam::getFollowerCount);
	ClassDB::bind_method(D_METHOD("getFriendByIndex", "friend_number", "friend_flags"), &Steam::getFriendByIndex);
	ClassDB::bind_method(D_METHOD("getFriendCoplayGame", "friend_id"), &Steam::getFriendCoplayGame);
	ClassDB::bind_method(D_METHOD("getFriendCoplayTime", "friend_id"), &Steam::getFriendCoplayTime);
	ClassDB::bind_method("getFriendCount", &Steam::getFriendCount);
	ClassDB::bind_method(D_METHOD("getFriendCountFromSource", "source_id"), &Steam::getFriendCountFromSource);
	ClassDB::bind_method(D_METHOD("getFriendFromSourceByIndex", "source_id", "friend_number"), &Steam::getFriendFromSourceByIndex);
	ClassDB::bind_method(D_METHOD("getFriendGamePlayed", "steam_id"), &Steam::getFriendGamePlayed);
	ClassDB::bind_method(D_METHOD("getFriendMessage", "friend_id", "message"), &Steam::getFriendMessage);
	ClassDB::bind_method(D_METHOD("getFriendPersonaName", "steam_id"), &Steam::getFriendPersonaName);	
	ClassDB::bind_method(D_METHOD("getFriendPersonaNameHistory", "steam_id", "name_history"), &Steam::getFriendPersonaNameHistory);
	ClassDB::bind_method(D_METHOD("getFriendPersonaState", "steam_id"), &Steam::getFriendPersonaState);
	ClassDB::bind_method(D_METHOD("getFriendRelationship", "steam_id"), &Steam::getFriendRelationship);
	ClassDB::bind_method(D_METHOD("getFriendRichPresence", "friend_id", "key"), &Steam::getFriendRichPresence);
	ClassDB::bind_method(D_METHOD("getFriendRichPresenceKeyCount", "friend_id"), &Steam::getFriendRichPresenceKeyCount);
	ClassDB::bind_method(D_METHOD("getFriendRichPresenceKeyByIndex", "friend_id", "key"), &Steam::getFriendRichPresenceKeyByIndex);
	ClassDB::bind_method("getFriendsGroupCount", &Steam::getFriendsGroupCount);
	ClassDB::bind_method(D_METHOD("getFriendsGroupIDByIndex", "friend_group"), &Steam::getFriendsGroupIDByIndex);
	ClassDB::bind_method(D_METHOD("getFriendsGroupMembersCount", "friend_group"), &Steam::getFriendsGroupMembersCount);
	ClassDB::bind_method(D_METHOD("getFriendsGroupMembersList", "friend_group", "member_count"), &Steam::getFriendsGroupMembersList);
	ClassDB::bind_method(D_METHOD("getFriendsGroupName", "friend_group"), &Steam::getFriendsGroupName);
	ClassDB::bind_method(D_METHOD("getFriendSteamLevel", "steam_id"), &Steam::getFriendSteamLevel);
	ClassDB::bind_method(D_METHOD("getLargeFriendAvatar", "steam_id"), &Steam::getLargeFriendAvatar);
	ClassDB::bind_method(D_METHOD("getMediumFriendAvatar", "steam_id"), &Steam::getMediumFriendAvatar);
	ClassDB::bind_method("getPersonaName", &Steam::getPersonaName);
	ClassDB::bind_method("getPersonaState", &Steam::getPersonaState);
	ClassDB::bind_method(D_METHOD("getPlayerAvatar", "size", "steam_id"), &Steam::getPlayerAvatar, DEFVAL(2), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("getPlayerNickname", "steam_id"), &Steam::getPlayerNickname);
	ClassDB::bind_method(D_METHOD("getProfileItemPropertyString", "steam_id", "item_type", "item_property"), &Steam::getProfileItemPropertyString);
	ClassDB::bind_method(D_METHOD("getProfileItemPropertyInt", "steam_id", "item_type", "item_propery"), &Steam::getProfileItemPropertyInt);
	ClassDB::bind_method("getRecentPlayers", &Steam::getRecentPlayers);
	ClassDB::bind_method(D_METHOD("getSmallFriendAvatar", "steam_id"), &Steam::getSmallFriendAvatar);
	ClassDB::bind_method("getUserFriendsGroups", &Steam::getUserFriendsGroups);
	ClassDB::bind_method("getUserRestrictions", &Steam::getUserRestrictions);
	ClassDB::bind_method("getUserSteamFriends", &Steam::getUserSteamFriends);
	ClassDB::bind_method("getUserSteamGroups", &Steam::getUserSteamGroups);
	ClassDB::bind_method(D_METHOD("hasEquippedProfileItem", "steam_id", "friend_flags"), &Steam::hasEquippedProfileItem);
	ClassDB::bind_method(D_METHOD("hasFriend", "steam_id", "friend_flags"), &Steam::hasFriend);
	ClassDB::bind_method(D_METHOD("inviteUserToGame", "friend_id", "connect_string"), &Steam::inviteUserToGame);
	ClassDB::bind_method(D_METHOD("isClanChatAdmin", "chat_id", "steam_id"), &Steam::isClanChatAdmin);
	ClassDB::bind_method(D_METHOD("isClanPublic", "clan_id"), &Steam::isClanPublic);
	ClassDB::bind_method(D_METHOD("isClanOfficialGameGroup", "clan_id"), &Steam::isClanOfficialGameGroup);
	ClassDB::bind_method(D_METHOD("isClanChatWindowOpenInSteam", "chat_id"), &Steam::isClanChatWindowOpenInSteam);
	ClassDB::bind_method(D_METHOD("isFollowing", "steam_id"), &Steam::isFollowing);
	ClassDB::bind_method(D_METHOD("isUserInSource", "steam_id", "source_id"), &Steam::isUserInSource);
	ClassDB::bind_method(D_METHOD("joinClanChatRoom", "clan_id"), &Steam::joinClanChatRoom);
	ClassDB::bind_method(D_METHOD("leaveClanChatRoom", "clan_id"), &Steam::leaveClanChatRoom);
	ClassDB::bind_method(D_METHOD("openClanChatWindowInSteam", "chat_id"), &Steam::openClanChatWindowInSteam);
	ClassDB::bind_method(D_METHOD("registerProtocolInOverlayBrowser", "protocol"), &Steam::registerProtocolInOverlayBrowser);
	ClassDB::bind_method(D_METHOD("replyToFriendMessage", "steam_id", "message"), &Steam::replyToFriendMessage);
	ClassDB::bind_method(D_METHOD("requestClanOfficerList", "clan_id"), &Steam::requestClanOfficerList);
	ClassDB::bind_method(D_METHOD("requestFriendRichPresence", "friend_id"), &Steam::requestFriendRichPresence);
	ClassDB::bind_method(D_METHOD("requestUserInformation", "steam_id", "require_name_only"), &Steam::requestUserInformation);
	ClassDB::bind_method(D_METHOD("sendClanChatMessage", "chat_id", "text"), &Steam::sendClanChatMessage);
	ClassDB::bind_method(D_METHOD("setInGameVoiceSpeaking", "steam_id", "speaking"), &Steam::setInGameVoiceSpeaking);
	ClassDB::bind_method(D_METHOD("setListenForFriendsMessages", "intercept"), &Steam::setListenForFriendsMessages);
	ClassDB::bind_method(D_METHOD("setPersonaName", "name"), &Steam::setPersonaName);
	ClassDB::bind_method(D_METHOD("setPlayedWith", "steam_id"), &Steam::setPlayedWith);	
	ClassDB::bind_method(D_METHOD("setRichPresence", "key", "value"), &Steam::setRichPresence);
	
	// GAME SEARCH BIND METHODS /////////////////
	ClassDB::bind_method(D_METHOD("addGameSearchParams", "key", "values"), &Steam::addGameSearchParams);
	ClassDB::bind_method(D_METHOD("searchForGameWithLobby", "lobby_id", "player_min", "player_max"), &Steam::searchForGameWithLobby);
	ClassDB::bind_method(D_METHOD("searchForGameSolo", "player_min", "player_max"), &Steam::searchForGameSolo);
	ClassDB::bind_method("acceptGame", &Steam::acceptGame);
	ClassDB::bind_method("declineGame", &Steam::declineGame);
	ClassDB::bind_method(D_METHOD("retrieveConnectionDetails", "host_id"), &Steam::retrieveConnectionDetails);
	ClassDB::bind_method("endGameSearch", &Steam::endGameSearch);
	ClassDB::bind_method(D_METHOD("setGameHostParams", "key", "value"), &Steam::setGameHostParams);
	ClassDB::bind_method(D_METHOD("setConnectionDetails", "details", "connection_details"), &Steam::setConnectionDetails);
	ClassDB::bind_method(D_METHOD("requestPlayersForGame", "player_min", "player_max", "max_team_size"), &Steam::requestPlayersForGame);
	ClassDB::bind_method(D_METHOD("hostConfirmGameStart", "game_id"), &Steam::hostConfirmGameStart);
	ClassDB::bind_method("cancelRequestPlayersForGame", &Steam::cancelRequestPlayersForGame);
	ClassDB::bind_method(D_METHOD("submitPlayerResult", "game_id", "player_id", "player_result"), &Steam::submitPlayerResult);
	ClassDB::bind_method(D_METHOD("endGame", "game_id"), &Steam::endGame);

	// GAME SERVER BIND METHODS /////////////////
	ClassDB::bind_method(D_METHOD("associateWithClan", "clan_id"), &Steam::associateWithClan);
	ClassDB::bind_method(D_METHOD("beginServerAuthSession", "ticket", "ticket_size", "steam_id"), &Steam::beginServerAuthSession);
	ClassDB::bind_method("serverLoggedOn", &Steam::serverLoggedOn);
	ClassDB::bind_method("secure", &Steam::secure);
	ClassDB::bind_method(D_METHOD("cancelServerAuthTicket", "auth_ticket"), &Steam::cancelServerAuthTicket);
	ClassDB::bind_method("clearAllKeyValues", &Steam::clearAllKeyValues);
	ClassDB::bind_method(D_METHOD("computeNewPlayerCompatibility", "steam_id"), &Steam::computeNewPlayerCompatibility);
	ClassDB::bind_method(D_METHOD("setAdvertiseServerActive", "active"), &Steam::setAdvertiseServerActive);
	ClassDB::bind_method(D_METHOD("endServerAuthSession", "steam_id"), &Steam::endServerAuthSession);
	ClassDB::bind_method("getServerAuthSessionTicket", &Steam::getServerAuthSessionTicket);
	ClassDB::bind_method("getNextOutgoingPacket", &Steam::getNextOutgoingPacket);
//	ClassDB::bind_method("getPublicIP", &Steam::getPublicIP);
	ClassDB::bind_method("getServerSteamID", &Steam::getServerSteamID);
	ClassDB::bind_method(D_METHOD("handleIncomingPacket", "packet", "ip", "port"), &Steam::handleIncomingPacket);
	ClassDB::bind_method("logOff", &Steam::logOff);
	ClassDB::bind_method(D_METHOD("logOn", "token"), &Steam::logOn);
	ClassDB::bind_method("logOnAnonymous", &Steam::logOnAnonymous);
	ClassDB::bind_method(D_METHOD("requestUserGroupStatus", "steam_id", "group_id"), &Steam::requestUserGroupStatus);
	ClassDB::bind_method(D_METHOD("setBotPlayerCount", "bots"), &Steam::setBotPlayerCount);
	ClassDB::bind_method(D_METHOD("setDedicatedServer", "dedicated"), &Steam::setDedicatedServer);
	ClassDB::bind_method(D_METHOD("setGameData", "data"), &Steam::setGameData);
	ClassDB::bind_method(D_METHOD("setGameDescription", "description"), &Steam::setGameDescription);
	ClassDB::bind_method(D_METHOD("setGameTags", "tags"), &Steam::setGameTags);
	ClassDB::bind_method(D_METHOD("setKeyValue", "key", "value"), &Steam::setKeyValue);
	ClassDB::bind_method(D_METHOD("setMapName", "map"), &Steam::setMapName);
	ClassDB::bind_method(D_METHOD("setMaxPlayerCount", "players_max"), &Steam::setMaxPlayerCount);
	ClassDB::bind_method(D_METHOD("setModDir", "mod_directory"), &Steam::setModDir);
	ClassDB::bind_method(D_METHOD("setPasswordProtected", "password_protected"), &Steam::setPasswordProtected);
	ClassDB::bind_method(D_METHOD("setProduct", "product"), &Steam::setProduct);
	ClassDB::bind_method(D_METHOD("setRegion", "region"), &Steam::setRegion);
	ClassDB::bind_method(D_METHOD("setServerName", "name"), &Steam::setServerName);
	ClassDB::bind_method(D_METHOD("setSpectatorPort", "port"), &Steam::setSpectatorPort);
	ClassDB::bind_method(D_METHOD("setSpectatorServerName", "name"), &Steam::setSpectatorServerName);
	ClassDB::bind_method(D_METHOD("userHasLicenceForApp", "steam_id", "app_id"), &Steam::userHasLicenceForApp);
	ClassDB::bind_method("wasRestartRequested", &Steam::wasRestartRequested);	
	
	// GAME SERVER STATS BIND METHODS ///////////
	ClassDB::bind_method(D_METHOD("clearUserAchievement", "steam_id", "name"), &Steam::clearUserAchievement);
	ClassDB::bind_method(D_METHOD("serverGetUserAchievement", "steam_id", "name"), &Steam::serverGetUserAchievement);
	ClassDB::bind_method(D_METHOD("serverGetUserStatInt", "steam_id", "name"), &Steam::serverGetUserStatInt);
	ClassDB::bind_method(D_METHOD("serverGetUserStatFloat", "steam_id", "name"), &Steam::serverGetUserStatFloat);
	ClassDB::bind_method(D_METHOD("serverRequestUserStats", "steam_id"), &Steam::serverRequestUserStats);
	ClassDB::bind_method(D_METHOD("setUserAchievement", "steam_id", "name"), &Steam::setUserAchievement);
	ClassDB::bind_method(D_METHOD("setUserStatInt", "steam_id", "name", "stat"), &Steam::setUserStatInt);
	ClassDB::bind_method(D_METHOD("setUserStatFloat", "steam_id", "name", "stat"), &Steam::setUserStatFloat);
	ClassDB::bind_method(D_METHOD("storeUserStats", "steam_id"), &Steam::storeUserStats);
	ClassDB::bind_method(D_METHOD("updateUserAvgRateStat", "steam_id", "name", "this_session", "session_length"), &Steam::updateUserAvgRateStat);

	// HTML SURFACE BIND METHODS ////////////////
	ClassDB::bind_method(D_METHOD("addHeader", "key", "value", "this_handle"), &Steam::addHeader, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("allowStartRequest", "allowed", "this_handle"), &Steam::allowStartRequest, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("copyToClipboard", "this_handle"), &Steam::copyToClipboard, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("createBrowser", "user_agent", "user_css"), &Steam::createBrowser);
	ClassDB::bind_method(D_METHOD("executeJavascript", "script", "this_handle"), &Steam::executeJavascript, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("find", "search", "currently_in_find", "reverse", "this_handle"), &Steam::find, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("getLinkAtPosition", "x", "y", "this_handle"), &Steam::getLinkAtPosition, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("goBack", "this_handle"), &Steam::goBack, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("goForward", "this_handle"), &Steam::goForward, DEFVAL(0));
	ClassDB::bind_method("htmlInit", &Steam::htmlInit);
	ClassDB::bind_method(D_METHOD("jsDialogResponse", "result", "this_handle"), &Steam::jsDialogResponse, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("keyChar", "unicode_char", "key_modifiers", "this_handle"), &Steam::keyChar, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("keyDown", "native_key_code", "key_modifiers", "this_handle"), &Steam::keyDown, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("keyUp", "native_key_code", "key_modifiers", "this_handle"), &Steam::keyUp, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("loadURL", "url", "post_data", "this_handle"), &Steam::loadURL, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("mouseDoubleClick", "mouse_button", "this_handle"), &Steam::mouseDoubleClick, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("mouseDown", "mouse_button", "this_handle"), &Steam::mouseDown, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("mouseMove", "x", "y", "this_handle"), &Steam::mouseMove, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("mouseUp", "mouse_button", "this_handle"), &Steam::mouseUp, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("mouseWheel", "delta", "this_handle"), &Steam::mouseWheel, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("pasteFromClipboard", "this_handle"), &Steam::pasteFromClipboard, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("reload", "this_handle"), &Steam::reload, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("removeBrowser", "this_handle"), &Steam::removeBrowser, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("setBackgroundMode", "background_mode", "this_handle"), &Steam::setBackgroundMode, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("setCookie", "hostname", "key", "value", "path", "expires", "secure", "http_only"), &Steam::setCookie);
	ClassDB::bind_method(D_METHOD("setHorizontalScroll", "absolute_pixel_scroll", "this_handle"), &Steam::setHorizontalScroll, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("setKeyFocus", "has_key_focus", "this_handle"), &Steam::setKeyFocus, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("setPageScaleFactor", "zoom", "point_x", "point_y", "this_handle"), &Steam::setPageScaleFactor, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("setSize", "width", "height", "this_handle"), &Steam::setSize, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("setVerticalScroll", "absolute_pixel_scroll", "this_handle"), &Steam::setVerticalScroll, DEFVAL(0));
	ClassDB::bind_method("htmlShutdown", &Steam::htmlShutdown);
	ClassDB::bind_method(D_METHOD("stopFind", "this_handle"), &Steam::stopFind, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("stopLoad", "this_handle"), &Steam::stopLoad, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("viewSource", "this_handle"), &Steam::viewSource, DEFVAL(0));

	// HTTP BIND METHODS ////////////////////////
	ClassDB::bind_method(D_METHOD("createCookieContainer", "allow_response_to_modify"), &Steam::createCookieContainer);
	ClassDB::bind_method(D_METHOD("createHTTPRequest", "request_method", "absolute_url"), &Steam::createHTTPRequest);
	ClassDB::bind_method(D_METHOD("deferHTTPRequest", "request_handle"), &Steam::deferHTTPRequest);
	ClassDB::bind_method(D_METHOD("getHTTPDownloadProgressPct", "request_handle"), &Steam::getHTTPDownloadProgressPct);
	ClassDB::bind_method(D_METHOD("getHTTPRequestWasTimedOut", "request_handle"), &Steam::getHTTPRequestWasTimedOut);
	ClassDB::bind_method(D_METHOD("getHTTPResponseBodyData", "request_handle", "buffer_size"), &Steam::getHTTPResponseBodyData);
	ClassDB::bind_method(D_METHOD("getHTTPResponseBodySize", "request_handle"), &Steam::getHTTPResponseBodySize);
	ClassDB::bind_method(D_METHOD("getHTTPResponseHeaderSize", "request_handle", "header_name"), &Steam::getHTTPResponseHeaderSize);
	ClassDB::bind_method(D_METHOD("getHTTPResponseHeaderValue","request_handle", "header_name", "buffer_size"), &Steam::getHTTPResponseHeaderValue);
	ClassDB::bind_method(D_METHOD("getHTTPStreamingResponseBodyData", "request_handle", "offset", "buffer_size"), &Steam::getHTTPStreamingResponseBodyData);
	ClassDB::bind_method(D_METHOD("prioritizeHTTPRequest", "request_handle"), &Steam::prioritizeHTTPRequest);
	ClassDB::bind_method(D_METHOD("releaseCookieContainer", "cookie_handle"), &Steam::releaseCookieContainer);
	ClassDB::bind_method(D_METHOD("releaseHTTPRequest", "request_handle"), &Steam::releaseHTTPRequest);
	ClassDB::bind_method(D_METHOD("sendHTTPRequest", "request_handle"), &Steam::sendHTTPRequest);
	ClassDB::bind_method(D_METHOD("sendHTTPRequestAndStreamResponse", "request_handle"), &Steam::sendHTTPRequestAndStreamResponse);
	ClassDB::bind_method(D_METHOD("setHTTPCookie", "cookie_handle", "host", "url", "cookie"), &Steam::setHTTPCookie);
	ClassDB::bind_method(D_METHOD("setHTTPRequestAbsoluteTimeoutMS", "request_handle", "milliseconds"), &Steam::setHTTPRequestAbsoluteTimeoutMS);
	ClassDB::bind_method(D_METHOD("setHTTPRequestContextValue", "request_handle", "context_value"), &Steam::setHTTPRequestContextValue);
	ClassDB::bind_method(D_METHOD("setHTTPRequestCookieContainer", "request_handle", "cookie_handle"), &Steam::setHTTPRequestCookieContainer);
	ClassDB::bind_method(D_METHOD("setHTTPRequestGetOrPostParameter", "request_handle", "name", "value"), &Steam::setHTTPRequestGetOrPostParameter);
	ClassDB::bind_method(D_METHOD("setHTTPRequestHeaderValue", "request_handle", "header_name", "header_value"), &Steam::setHTTPRequestHeaderValue);
	ClassDB::bind_method(D_METHOD("setHTTPRequestNetworkActivityTimeout", "request_handle", "timeout_seconds"), &Steam::setHTTPRequestNetworkActivityTimeout);
	ClassDB::bind_method(D_METHOD("setHTTPRequestRawPostBody", "request_handle", "content_type", "body_length"), &Steam::setHTTPRequestRawPostBody);
	ClassDB::bind_method(D_METHOD("setHTTPRequestRequiresVerifiedCertificate", "request_handle", "require_verified_certificate"), &Steam::setHTTPRequestRequiresVerifiedCertificate);
	ClassDB::bind_method(D_METHOD("setHTTPRequestUserAgentInfo", "request_handle", "user_agent_info"), &Steam::setHTTPRequestUserAgentInfo);

	// INPUT BIND METHODS /////////////////
	ClassDB::bind_method(D_METHOD("activateActionSet", "input_handle", "action_set_handle"), &Steam::activateActionSet);
	ClassDB::bind_method(D_METHOD("activateActionSetLayer", "input_handle", "action_set_layer_handle"), &Steam::activateActionSetLayer);
	ClassDB::bind_method(D_METHOD("deactivateActionSetLayer", "input_handle", "action_set_handle"), &Steam::deactivateActionSetLayer);
	ClassDB::bind_method(D_METHOD("deactivateAllActionSetLayers", "input_handle"), &Steam::deactivateAllActionSetLayers);
	ClassDB::bind_method(D_METHOD("getActionSetHandle", "action_set_name"), &Steam::getActionSetHandle);
	ClassDB::bind_method(D_METHOD("getActionOriginFromXboxOrigin","input_handle", "origin"), &Steam::getActionOriginFromXboxOrigin);
	ClassDB::bind_method(D_METHOD("getActiveActionSetLayers", "input_handle"), &Steam::getActiveActionSetLayers);
	ClassDB::bind_method(D_METHOD("getAnalogActionData", "input_handle", "analog_action_handle"), &Steam::getAnalogActionData);
	ClassDB::bind_method(D_METHOD("getAnalogActionHandle", "action_name"), &Steam::getAnalogActionHandle);
	ClassDB::bind_method(D_METHOD("getAnalogActionOrigins", "input_handle", "action_set_handle", "analog_action_handle"), &Steam::getAnalogActionOrigins);
	ClassDB::bind_method("getConnectedControllers", &Steam::getConnectedControllers);
	ClassDB::bind_method(D_METHOD("getControllerForGamepadIndex", "index"), &Steam::getControllerForGamepadIndex);
	ClassDB::bind_method(D_METHOD("getCurrentActionSet", "input_handle"), &Steam::getCurrentActionSet);
	ClassDB::bind_method(D_METHOD("getDeviceBindingRevision", "input_handle"), &Steam::getDeviceBindingRevision);
	ClassDB::bind_method(D_METHOD("getDigitalActionData", "input_handle", "digital_action_handle"), &Steam::getDigitalActionData);
	ClassDB::bind_method(D_METHOD("getDigitalActionHandle", "action_name"), &Steam::getDigitalActionHandle);
	ClassDB::bind_method(D_METHOD("getDigitalActionOrigins", "input_handle", "action_set_handle", "digital_action_handle"), &Steam::getDigitalActionOrigins);
	ClassDB::bind_method(D_METHOD("getGamepadIndexForController", "input_handle"), &Steam::getGamepadIndexForController);
	ClassDB::bind_method(D_METHOD("getGlyphForActionOrigin", "origin"), &Steam::getGlyphForActionOrigin);
	ClassDB::bind_method(D_METHOD("getInputTypeForHandle", "input_handle"), &Steam::getInputTypeForHandle);
	ClassDB::bind_method(D_METHOD("getMotionData", "input_handle"), &Steam::getMotionData);
	ClassDB::bind_method(D_METHOD("getRemotePlaySessionID", "input_handle"), &Steam::getRemotePlaySessionID);
	ClassDB::bind_method(D_METHOD("getStringForActionOrigin", "origin"), &Steam::getStringForActionOrigin);
	ClassDB::bind_method(D_METHOD("inputInit", "explicitly_call_runframe"), &Steam::inputInit, DEFVAL(false));
	ClassDB::bind_method("inputShutdown", &Steam::inputShutdown);
	ClassDB::bind_method(D_METHOD("runFrame", "reserved_value"), &Steam::runFrame, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("setLEDColor", "input_handle", "color_r", "color_g", "color_b", "flags"), &Steam::setLEDColor);
	ClassDB::bind_method(D_METHOD("showBindingPanel", "input_handle"), &Steam::showBindingPanel);
	ClassDB::bind_method(D_METHOD("stopAnalogActionMomentum", "input_handle", "action"), &Steam::stopAnalogActionMomentum);
	ClassDB::bind_method(D_METHOD("translateActionOrigin", "destination_input", "source_origin"), &Steam::translateActionOrigin);
	ClassDB::bind_method(D_METHOD("triggerHapticPulse", "input_handle", "target_pad", "duration"), &Steam::triggerHapticPulse);
	ClassDB::bind_method(D_METHOD("triggerRepeatedHapticPulse", "input_handle", "target_pad", "duration", "offset", "repeat", "flags"), &Steam::triggerRepeatedHapticPulse);
	ClassDB::bind_method(D_METHOD("triggerVibration", "input_handle", "left_speed", "right_speed"), &Steam::triggerVibration);
	ClassDB::bind_method(D_METHOD("setInputActionManifestFilePath", "manifest_path"), &Steam::setInputActionManifestFilePath);
	ClassDB::bind_method(D_METHOD("waitForData", "wait_forever", "timeout"), &Steam::waitForData);
	ClassDB::bind_method("newDataAvailable", &Steam::newDataAvailable);
	ClassDB::bind_method("enableDeviceCallbacks", &Steam::enableDeviceCallbacks);
	ClassDB::bind_method(D_METHOD("getGlyphPNGForActionOrigin", "origin", "size", "flags"), &Steam::getGlyphPNGForActionOrigin);
	ClassDB::bind_method(D_METHOD("getGlyphSVGForActionOrigin", "origin", "flags"), &Steam::getGlyphSVGForActionOrigin);
	ClassDB::bind_method(D_METHOD("triggerVibrationExtended", "input_handle", "left_speed", "right_speed", "left_trigger_speed", "right_trigger_speed"), &Steam::triggerVibrationExtended);
	ClassDB::bind_method(D_METHOD("triggerSimpleHapticEvent", "input_handle", "haptic_location", "intensity", "gain_db", "other_intensity", "other_gain_db"), &Steam::triggerSimpleHapticEvent);
	ClassDB::bind_method(D_METHOD("getStringForXboxOrigin", "origin"), &Steam::getStringForXboxOrigin);
	ClassDB::bind_method(D_METHOD("getGlyphForXboxOrigin", "origin"), &Steam::getGlyphForXboxOrigin);
	ClassDB::bind_method("getSessionInputConfigurationSettings",  &Steam::getSessionInputConfigurationSettings);
	ClassDB::bind_method(D_METHOD("getStringForDigitalActionName", "action_handle"), &Steam::getStringForDigitalActionName);
	ClassDB::bind_method(D_METHOD("getStringForAnalogActionName", "action_handle"), &Steam::getStringForAnalogActionName);

	// INVENTORY BIND METHODS ///////////////////
	ClassDB::bind_method(D_METHOD("addPromoItem", "item"), &Steam::addPromoItem);
	ClassDB::bind_method(D_METHOD("addPromoItems", "items"), &Steam::addPromoItems);
	ClassDB::bind_method(D_METHOD("checkResultSteamID", "steam_id_expected", "this_inventory_handle"), &Steam::checkResultSteamID, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("consumeItem", "item_consume", "quantity"), &Steam::consumeItem);
	ClassDB::bind_method(D_METHOD("deserializeResult" "buffer"), &Steam::deserializeResult);
	ClassDB::bind_method(D_METHOD("destroyResult", "this_inventory_handle"), &Steam::destroyResult, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("exchangeItems", "output_items", "output_quantity", "input_items", "input_quantity"), &Steam::exchangeItems);
	ClassDB::bind_method(D_METHOD("generateItems", "items", "quantity"), &Steam::generateItems);
	ClassDB::bind_method("getAllItems", &Steam::getAllItems);
	ClassDB::bind_method(D_METHOD("getItemDefinitionProperty", "definition", "name"), &Steam::getItemDefinitionProperty);
	ClassDB::bind_method(D_METHOD("getItemsByID", "id_array", "count"), &Steam::getItemsByID);
	ClassDB::bind_method(D_METHOD("getItemPrice", "definition"), &Steam::getItemPrice);
	ClassDB::bind_method(D_METHOD("getItemsWithPrices", "length"), &Steam::getItemsWithPrices);
	ClassDB::bind_method("getNumItemsWithPrices", &Steam::getNumItemsWithPrices);
	ClassDB::bind_method(D_METHOD("getResultItemProperty", "index", "name", "this_inventory_handle"), &Steam::getResultItemProperty, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("getResultItems", "this_inventory_handle"), &Steam::getResultItems, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("getResultStatus", "this_inventory_handle"), &Steam::getResultStatus, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("getResultTimestamp", "this_inventory_handle"), &Steam::getResultTimestamp, DEFVAL(0));
	ClassDB::bind_method("grantPromoItems", &Steam::grantPromoItems);
	ClassDB::bind_method("loadItemDefinitions", &Steam::loadItemDefinitions);
	ClassDB::bind_method(D_METHOD("requestEligiblePromoItemDefinitionsIDs", "steam_id"), &Steam::requestEligiblePromoItemDefinitionsIDs);
	ClassDB::bind_method("requestPrices", &Steam::requestPrices);
	ClassDB::bind_method(D_METHOD("serializeResult", "this_inventory_handle"), &Steam::serializeResult, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("startPurchase", "items", "quantity"), &Steam::startPurchase);
	ClassDB::bind_method(D_METHOD("transferItemQuantity", "item_id", "quantity", "item_destination", "split"), &Steam::transferItemQuantity);
	ClassDB::bind_method(D_METHOD("triggerItemDrop", "definition"), &Steam::triggerItemDrop);
	ClassDB::bind_method("startUpdateProperties", &Steam::startUpdateProperties);
	ClassDB::bind_method(D_METHOD("submitUpdateProperties", "this_inventory_update_handle"), &Steam::submitUpdateProperties, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("removeProperty", "item_id", "name", "this_inventory_update_handle"), &Steam::removeProperty, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("setPropertyString", "item_id", "name", "value", "this_inventory_update_handle"), &Steam::setPropertyString, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("setPropertyBool", "item_id", "name", "value", "this_inventory_update_handle"), &Steam::setPropertyBool, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("setPropertyInt", "item_id", "name", "value", "this_inventory_update_handle"), &Steam::setPropertyInt, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("setPropertyFloat", "item_id", "name", "value", "this_inventory_update_handle"), &Steam::setPropertyFloat, DEFVAL(0));

	// MATCHMAKING BIND METHODS /////////////////
	ClassDB::bind_method("getFavoriteGames", &Steam::getFavoriteGames);
	ClassDB::bind_method(D_METHOD("addFavoriteGame", "ip", "port", "query_port", "flags", "last_played"), &Steam::addFavoriteGame);
	ClassDB::bind_method(D_METHOD("removeFavoriteGame", "app_id", "ip", "port", "query_port", "flags"), &Steam::removeFavoriteGame);
	ClassDB::bind_method("requestLobbyList", &Steam::requestLobbyList);
	ClassDB::bind_method(D_METHOD("addRequestLobbyListStringFilter", "key_to_match", "value_to_match", "comparison_type"), &Steam::addRequestLobbyListStringFilter);
	ClassDB::bind_method(D_METHOD("addRequestLobbyListNumericalFilter", "key_to_match", "value_to_match", "comparison_type"), &Steam::addRequestLobbyListNumericalFilter);
	ClassDB::bind_method(D_METHOD("addRequestLobbyListNearValueFilter", "key_to_match", "value_to_be_close_to"), &Steam::addRequestLobbyListNearValueFilter);
	ClassDB::bind_method(D_METHOD("addRequestLobbyListFilterSlotsAvailable", "slots_available"), &Steam::addRequestLobbyListFilterSlotsAvailable);
	ClassDB::bind_method(D_METHOD("addRequestLobbyListDistanceFilter", "distance_filter"), &Steam::addRequestLobbyListDistanceFilter);
	ClassDB::bind_method(D_METHOD("addRequestLobbyListResultCountFilter", "max_results"), &Steam::addRequestLobbyListResultCountFilter);
	ClassDB::bind_method(D_METHOD("createLobby", "lobby_type", "max_members"), &Steam::createLobby, DEFVAL(2));
	ClassDB::bind_method(D_METHOD("joinLobby", "steam_lobby_id"), &Steam::joinLobby);
	ClassDB::bind_method(D_METHOD("leaveLobby", "steam_lobby_id"), &Steam::leaveLobby);
	ClassDB::bind_method(D_METHOD("inviteUserToLobby", "steam_lobby_id", "steam_id_invitee"), &Steam::inviteUserToLobby);
	ClassDB::bind_method(D_METHOD("getNumLobbyMembers", "steam_lobby_id"), &Steam::getNumLobbyMembers);
	ClassDB::bind_method(D_METHOD("getLobbyMemberByIndex", "steam_lobby_id", "member"), &Steam::getLobbyMemberByIndex);
	ClassDB::bind_method(D_METHOD("getLobbyData", "steam_lobby_id", "key"), &Steam::getLobbyData);
	ClassDB::bind_method(D_METHOD("setLobbyData", "steam_lobby_id", "key", "value"), &Steam::setLobbyData);
	ClassDB::bind_method(D_METHOD("getAllLobbyData", "steam_lobby_id"), &Steam::getAllLobbyData);
	ClassDB::bind_method(D_METHOD("deleteLobbyData", "steam_lobby_id", "key"), &Steam::deleteLobbyData);
	ClassDB::bind_method(D_METHOD("getLobbyMemberData", "steam_lobby_id", "steam_id_user", "key"), &Steam::getLobbyMemberData);
	ClassDB::bind_method(D_METHOD("setLobbyMemberData", "steam_lobby_id", "key", "value"), &Steam::setLobbyMemberData);
	ClassDB::bind_method(D_METHOD("sendLobbyChatMsg", "steam_lobby_id", "message_body"), &Steam::sendLobbyChatMsg);
	ClassDB::bind_method(D_METHOD("requestLobbyData", "steam_lobby_id"), &Steam::requestLobbyData);
	ClassDB::bind_method(D_METHOD("setLobbyGameServer", "steam_lobby_id", "server_ip", "server_port", "steam_id_game_server"), &Steam::setLobbyGameServer);
	ClassDB::bind_method(D_METHOD("getLobbyGameServer", "steam_lobby_id"), &Steam::getLobbyGameServer);
	ClassDB::bind_method(D_METHOD("setLobbyMemberLimit", "steam_lobby_id", "max_members"), &Steam::setLobbyMemberLimit);
	ClassDB::bind_method(D_METHOD("getLobbyMemberLimit", "steam_lobby_id"), &Steam::getLobbyMemberLimit);
	ClassDB::bind_method(D_METHOD("setLobbyType", "steam_lobby_id", "lobby_type"), &Steam::setLobbyType);
	ClassDB::bind_method(D_METHOD("setLobbyJoinable", "steam_lobby_id", "joinable"), &Steam::setLobbyJoinable);
	ClassDB::bind_method(D_METHOD("getLobbyOwner", "steam_lobby_id"), &Steam::getLobbyOwner);
	ClassDB::bind_method(D_METHOD("setLobbyOwner", "steam_lobby_id", "steam_id_new_owner"), &Steam::setLobbyOwner);

	// MATCHMAKING SERVERS BIND METHODS /////////
	ClassDB::bind_method(D_METHOD("cancelQuery", "server_list_request"), &Steam::cancelQuery);
	ClassDB::bind_method(D_METHOD("cancelServerQuery", "server_query"), &Steam::cancelServerQuery);
	ClassDB::bind_method(D_METHOD("getServerCount", "server_list_request"), &Steam::getServerCount);
	ClassDB::bind_method(D_METHOD("getServerDetails", "server_list_request", "server"), &Steam::getServerDetails);
	ClassDB::bind_method(D_METHOD("isRefreshing", "server_list_request"), &Steam::isRefreshing);
	ClassDB::bind_method(D_METHOD("pingServer", "ip", "port"), &Steam::pingServer);
	ClassDB::bind_method(D_METHOD("playerDetails", "ip", "port"), &Steam::playerDetails);
	ClassDB::bind_method(D_METHOD("refreshQuery", "server_list_request"), &Steam::refreshQuery);
	ClassDB::bind_method(D_METHOD("refreshServer", "server_list_request", "server"), &Steam::refreshServer);
	ClassDB::bind_method(D_METHOD("releaseRequest", "server_list_request"), &Steam::releaseRequest);
	ClassDB::bind_method(D_METHOD("requestFavoritesServerList", "app_id", "filters"), &Steam::requestFavoritesServerList);
	ClassDB::bind_method(D_METHOD("requestFriendsServerList", "app_id", "filters"), &Steam::requestFriendsServerList);
	ClassDB::bind_method(D_METHOD("requestHistoryServerList", "app_id", "filters"), &Steam::requestHistoryServerList);
	ClassDB::bind_method(D_METHOD("requestInternetServerList", "app_id", "filters"), &Steam::requestInternetServerList);
	ClassDB::bind_method(D_METHOD("requestLANServerList", "app_id"), &Steam::requestLANServerList);
	ClassDB::bind_method(D_METHOD("requestSpectatorServerList", "app_id", "filters"), &Steam::requestSpectatorServerList);
	ClassDB::bind_method(D_METHOD("serverRules", "ip", "port"), &Steam::serverRules);
	
	// MUSIC BIND METHODS ///////////////////////
	ClassDB::bind_method("musicIsEnabled", &Steam::musicIsEnabled);
	ClassDB::bind_method("musicIsPlaying", &Steam::musicIsPlaying);
	ClassDB::bind_method("getPlaybackStatus", &Steam::getPlaybackStatus);
	ClassDB::bind_method("musicGetVolume", &Steam::musicGetVolume);
	ClassDB::bind_method("musicPause", &Steam::musicPause);
	ClassDB::bind_method("musicPlay", &Steam::musicPlay);
	ClassDB::bind_method("musicPlayNext", &Steam::musicPlayNext);
	ClassDB::bind_method("musicPlayPrev", &Steam::musicPlayPrev);
	ClassDB::bind_method(D_METHOD("musicSetVolume", "volume"), &Steam::musicSetVolume);
	
	// MUSIC REMOTE BIND METHODS ////////////////
	ClassDB::bind_method(D_METHOD("activationSuccess", "activate"), &Steam::activationSuccess);
	ClassDB::bind_method("isCurrentMusicRemote", &Steam::isCurrentMusicRemote);
	ClassDB::bind_method("currentEntryDidChange", &Steam::currentEntryDidChange);
	ClassDB::bind_method(D_METHOD("currentEntryIsAvailable", "available"), &Steam::currentEntryIsAvailable);
	ClassDB::bind_method("currentEntryWillChange", &Steam::currentEntryWillChange);
	ClassDB::bind_method("deregisterSteamMusicRemote", &Steam::deregisterSteamMusicRemote);
	ClassDB::bind_method(D_METHOD("enableLooped", "loop"), &Steam::enableLooped);
	ClassDB::bind_method(D_METHOD("enablePlaylists", "playlists"), &Steam::enablePlaylists);
	ClassDB::bind_method(D_METHOD("enablePlayNext", "next"), &Steam::enablePlayNext);
	ClassDB::bind_method(D_METHOD("enablePlayPrevious", "previous"), &Steam::enablePlayPrevious);
	ClassDB::bind_method(D_METHOD("enableQueue", "queue"), &Steam::enableQueue);
	ClassDB::bind_method(D_METHOD("enableShuffled", "shuffle"), &Steam::enableShuffled);
	ClassDB::bind_method("playlistDidChange", &Steam::playlistDidChange);
	ClassDB::bind_method("playlistWillChange", &Steam::playlistWillChange);
	ClassDB::bind_method("queueDidChange", &Steam::queueDidChange);
	ClassDB::bind_method("queueWillChange", &Steam::queueWillChange);
	ClassDB::bind_method(D_METHOD("registerSteamMusicRemote", "name"), &Steam::registerSteamMusicRemote);
	ClassDB::bind_method("resetPlaylistEntries", &Steam::resetPlaylistEntries);
	ClassDB::bind_method("resetQueueEntries", &Steam::resetQueueEntries);
	ClassDB::bind_method(D_METHOD("setCurrentPlaylistEntry", "id"), &Steam::setCurrentPlaylistEntry);
	ClassDB::bind_method(D_METHOD("setCurrentQueueEntry", "id"), &Steam::setCurrentQueueEntry);
	ClassDB::bind_method(D_METHOD("setDisplayName", "name"), &Steam::setDisplayName);
	ClassDB::bind_method(D_METHOD("setPlaylistEntry", "id", "position", "entry_text"), &Steam::setPlaylistEntry);
	ClassDB::bind_method(D_METHOD("setPNGIcon64x64", "icon"), &Steam::setPNGIcon64x64);
	ClassDB::bind_method(D_METHOD("setQueueEntry", "id", "position", "entry_text"), &Steam::setQueueEntry);
	ClassDB::bind_method(D_METHOD("updateCurrentEntryCoverArt", "art"), &Steam::updateCurrentEntryCoverArt);
	ClassDB::bind_method(D_METHOD("updateCurrentEntryElapsedSeconds", "seconds"), &Steam::updateCurrentEntryElapsedSeconds);
	ClassDB::bind_method(D_METHOD("updateCurrentEntryText", "text"), &Steam::updateCurrentEntryText);
	ClassDB::bind_method(D_METHOD("updateLooped", "looped"), &Steam::updateLooped);
	ClassDB::bind_method(D_METHOD("updatePlaybackStatus", "status"), &Steam::updatePlaybackStatus);
	ClassDB::bind_method(D_METHOD("updateShuffled", "shuffle"), &Steam::updateShuffled);
	ClassDB::bind_method(D_METHOD("updateVolume", "volume"), &Steam::updateVolume);

	// NETWORKING BIND METHODS //////////////////
	ClassDB::bind_method(D_METHOD("acceptP2PSessionWithUser", "steam_id_remote"), &Steam::acceptP2PSessionWithUser);
	ClassDB::bind_method(D_METHOD("allowP2PPacketRelay", "allow"), &Steam::allowP2PPacketRelay);
	ClassDB::bind_method(D_METHOD("closeP2PChannelWithUser", "steam_id_remote", "channel"), &Steam::closeP2PChannelWithUser);
	ClassDB::bind_method(D_METHOD("closeP2PSessionWithUser", "steam_id_remote"), &Steam::closeP2PSessionWithUser);
	ClassDB::bind_method(D_METHOD("getP2PSessionState", "steam_id_remote"), &Steam::getP2PSessionState);
	ClassDB::bind_method(D_METHOD("getAvailableP2PPacketSize", "channel"), &Steam::getAvailableP2PPacketSize, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("readP2PPacket", "packet", "channel"), &Steam::readP2PPacket, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("sendP2PPacket", "steam_id_remote", "data", "send_type", "channel"), &Steam::sendP2PPacket, DEFVAL(0));

	// NETWORKING MESSAGES BIND METHODS /////////
	ClassDB::bind_method(D_METHOD("acceptSessionWithUser", "identity_reference"), &Steam::acceptSessionWithUser);
	ClassDB::bind_method(D_METHOD("closeChannelWithUser", "identity_reference", "channel"), &Steam::closeChannelWithUser);
	ClassDB::bind_method(D_METHOD("closeSessionWithUser", "identity_reference"), &Steam::closeSessionWithUser);
	ClassDB::bind_method(D_METHOD("getSessionConnectionInfo", "identity_reference", "get_connection", "get_status"), &Steam::getSessionConnectionInfo);
	ClassDB::bind_method(D_METHOD("receiveMessagesOnChannel", "channel", "max_messages"), &Steam::receiveMessagesOnChannel);
	ClassDB::bind_method(D_METHOD("sendMessageToUser", "identity_reference", "data", "flags", "channel"), &Steam::sendMessageToUser);
	
	// NETWORKING SOCKETS BIND METHODS //////////
	ClassDB::bind_method(D_METHOD("acceptConnection", "connection"), &Steam::acceptConnection);
	ClassDB::bind_method(D_METHOD("beginAsyncRequestFakeIP", "num_ports"), &Steam::beginAsyncRequestFakeIP);
	ClassDB::bind_method(D_METHOD("closeConnection", "peer", "reason", "debug_message", "linger"), &Steam::closeConnection);
	ClassDB::bind_method(D_METHOD("closeListenSocket", "socket"), &Steam::closeListenSocket);
	ClassDB::bind_method(D_METHOD("configureConnectionLanes", "connection", "lanes", "priorities", "weights"), &Steam::configureConnectionLanes);
	ClassDB::bind_method(D_METHOD("connectP2P", "identity_reference", "virtual_port", "options"), &Steam::connectP2P);
	ClassDB::bind_method(D_METHOD("connectToHostedDedicatedServer", "identity_reference", "virtual_port", "options"), &Steam::connectToHostedDedicatedServer);
	ClassDB::bind_method(D_METHOD("createFakeUDPPort", "fake_server_port"), &Steam::createFakeUDPPort);
	ClassDB::bind_method(D_METHOD("createHostedDedicatedServerListenSocket", "virtual_port", "options"), &Steam::createHostedDedicatedServerListenSocket);
	ClassDB::bind_method(D_METHOD("createListenSocketIP", "ip_reference", "options"), &Steam::createListenSocketIP);
	ClassDB::bind_method(D_METHOD("createListenSocketP2P", "virtual_port", "options"), &Steam::createListenSocketP2P);
	ClassDB::bind_method(D_METHOD("createListenSocketP2PFakeIP", "fake_port", "options"), &Steam::createListenSocketP2PFakeIP);
	ClassDB::bind_method("createPollGroup", &Steam::createPollGroup);
	ClassDB::bind_method(D_METHOD("createSocketPair", "loopback", "identity_reference1", "identity_reference2"), &Steam::createSocketPair);
	ClassDB::bind_method(D_METHOD("destroyPollGroup", "poll_group"), &Steam::destroyPollGroup);
//	ClassDB::bind_method(D_METHOD("findRelayAuthTicketForServer", "port"), &Steam::findRelayAuthTicketForServer);	<------ Uses datagram relay structs which were removed from base SDK
	ClassDB::bind_method(D_METHOD("flushMessagesOnConnection", "connection_handle"), &Steam::flushMessagesOnConnection);
	ClassDB::bind_method("getAuthenticationStatus", &Steam::getAuthenticationStatus);		
	ClassDB::bind_method("getCertificateRequest", &Steam::getCertificateRequest);
	ClassDB::bind_method(D_METHOD("getConnectionInfo", "connection_handle"), &Steam::getConnectionInfo);
	ClassDB::bind_method(D_METHOD("getConnectionName", "peer"), &Steam::getConnectionName);
	ClassDB::bind_method(D_METHOD("getConnectionRealTimeStatus", "connection_handle", "lanes", "get_status"), &Steam::getConnectionRealTimeStatus, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("getConnectionUserData", "peer"), &Steam::getConnectionUserData);
	ClassDB::bind_method(D_METHOD("getDetailedConnectionStatus", "connection_handle"), &Steam::getDetailedConnectionStatus);
	ClassDB::bind_method(D_METHOD("getFakeIP", "first_port"), &Steam::getFakeIP, DEFVAL(0));
//	ClassDB::bind_method(D_METHOD("getGameCoordinatorServerLogin", "app_data"), &Steam::getGameCoordinatorServerLogin);	<------ Uses datagram relay structs which were removed from base SDK
//	ClassDB::bind_method("getHostedDedicatedServerAddress", &Steam::getHostedDedicatedServerAddress);	<------ Uses datagram relay structs which were removed from base SDK
	ClassDB::bind_method("getHostedDedicatedServerPOPId", &Steam::getHostedDedicatedServerPOPId);
	ClassDB::bind_method("getHostedDedicatedServerPort", &Steam::getHostedDedicatedServerPort);
	ClassDB::bind_method(D_METHOD("getListenSocketAddress", "socket"), &Steam::getListenSocketAddress);
	ClassDB::bind_method("getIdentity", &Steam::getIdentity);
	ClassDB::bind_method(D_METHOD("getRemoteFakeIPForConnection", "connection"), &Steam::getRemoteFakeIPForConnection);
	ClassDB::bind_method("initAuthentication", &Steam::initAuthentication);
	ClassDB::bind_method(D_METHOD("receiveMessagesOnConnection", "connection", "max_messages"), &Steam::receiveMessagesOnConnection);	
	ClassDB::bind_method(D_METHOD("receiveMessagesOnPollGroup", "poll_group", "max_messages"), &Steam::receiveMessagesOnPollGroup);
//	ClassDB::bind_method("receivedRelayAuthTicket", &Steam::receivedRelayAuthTicket);	<------ Uses datagram relay structs which were removed from base SDK
	ClassDB::bind_method(D_METHOD("resetIdentity", "this_identity"), &Steam::resetIdentity);
	ClassDB::bind_method("runNetworkingCallbacks", &Steam::runNetworkingCallbacks);
	ClassDB::bind_method(D_METHOD("sendMessages", "messages", "data", "connection_handle", "flags"), &Steam::sendMessages);
	ClassDB::bind_method(D_METHOD("sendMessageToConnection", "connection_handle", "data", "flags"), &Steam::sendMessageToConnection);
	ClassDB::bind_method(D_METHOD("setCertificate", "certificate"), &Steam::setCertificate);	
	ClassDB::bind_method(D_METHOD("setConnectionPollGroup", "connection_handle", "poll_group"), &Steam::setConnectionPollGroup);
	ClassDB::bind_method(D_METHOD("setConnectionName", "peer", "name"), &Steam::setConnectionName);
	
	// NETWORKING TYPES BIND METHODS ////////////
	ClassDB::bind_method(D_METHOD("addIdentity", "reference_name"), &Steam::addIdentity);
	ClassDB::bind_method(D_METHOD("addIPAddress", "reference_name"), &Steam::addIPAddress);
	ClassDB::bind_method(D_METHOD("clearIdentity", "reference_name"), &Steam::clearIdentity);
	ClassDB::bind_method(D_METHOD("clearIPAddress", "reference_name"), &Steam::clearIPAddress);
	ClassDB::bind_method(D_METHOD("getGenericBytes", "reference_name"), &Steam::getGenericBytes);
	ClassDB::bind_method(D_METHOD("getGenericString", "reference_name"), &Steam::getGenericString);
	ClassDB::bind_method("getIdentities", &Steam::getIdentities);
	ClassDB::bind_method(D_METHOD("getIdentityIPAddr", "reference_name"), &Steam::getIdentityIPAddr);
	ClassDB::bind_method(D_METHOD("getIdentitySteamID", "reference_name"), &Steam::getIdentitySteamID);
	ClassDB::bind_method(D_METHOD("getIdentitySteamID64", "reference_name"), &Steam::getIdentitySteamID64);
	ClassDB::bind_method("getIPAddresses", &Steam::getIPAddresses);
	ClassDB::bind_method(D_METHOD("getIPv4", "reference_name"), &Steam::getIPv4);
	ClassDB::bind_method(D_METHOD("getPSNID", "reference_name"), &Steam::getPSNID);
	ClassDB::bind_method(D_METHOD("getStadiaID", "reference_name"), &Steam::getStadiaID);
	ClassDB::bind_method(D_METHOD("getXboxPairwiseID", "reference_name"), &Steam::getXboxPairwiseID);
	ClassDB::bind_method(D_METHOD("isAddressLocalHost", "reference_name"), &Steam::isAddressLocalHost);
	ClassDB::bind_method(D_METHOD("isIdentityInvalid", "reference_name"), &Steam::isIdentityInvalid);
	ClassDB::bind_method(D_METHOD("isIdentityLocalHost", "reference_name"), &Steam::isIdentityLocalHost);
	ClassDB::bind_method(D_METHOD("isIPv4", "reference_name"), &Steam::isIPv4);
	ClassDB::bind_method(D_METHOD("isIPv6AllZeros", "reference_name"), &Steam::isIPv6AllZeros);
	ClassDB::bind_method(D_METHOD("parseIdentityString", "reference_name", "string_to_parse"), &Steam::parseIdentityString);
	ClassDB::bind_method(D_METHOD("parseIPAddressString", "reference_name", "string_to_parse"), &Steam::parseIPAddressString);
	ClassDB::bind_method(D_METHOD("setGenericBytes", "reference_name", "data"), &Steam::setGenericBytes);
	ClassDB::bind_method(D_METHOD("setGenericString", "reference_name", "this_string"), &Steam::setGenericString);
	ClassDB::bind_method(D_METHOD("setIdentityIPAddr", "reference_name", "ip_address_name"), &Steam::setIdentityIPAddr);
	ClassDB::bind_method(D_METHOD("setIdentityLocalHost", "reference_name"), &Steam::setIdentityLocalHost);
	ClassDB::bind_method(D_METHOD("setIdentitySteamID", "reference_name", "steam_id"), &Steam::setIdentitySteamID);
	ClassDB::bind_method(D_METHOD("setIdentitySteamID64", "reference_name", "steam_id"), &Steam::setIdentitySteamID64);
	ClassDB::bind_method(D_METHOD("setIPv4", "reference_name", "ip", "port"), &Steam::setIPv4);
	ClassDB::bind_method(D_METHOD("setIPv6", "reference_name", "ipv6", "port"), &Steam::setIPv6);
	ClassDB::bind_method(D_METHOD("setIPv6LocalHost", "reference_name", "port"), &Steam::setIPv6LocalHost, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("setPSNID", "reference_name", "psn_id"), &Steam::setPSNID);
	ClassDB::bind_method(D_METHOD("setStadiaID", "reference_name", "stadia_id"), &Steam::setStadiaID);
	ClassDB::bind_method(D_METHOD("setXboxPairwiseID", "reference_name", "xbox_id"), &Steam::setXboxPairwiseID); 
	ClassDB::bind_method(D_METHOD("toIdentityString", "reference_name"), &Steam::toIdentityString);
	ClassDB::bind_method(D_METHOD("toIPAddressString", "reference_name", "with_port"), &Steam::toIPAddressString);
	
	// NETWORKING UTILS BIND METHODS ////////////
	ClassDB::bind_method(D_METHOD("checkPingDataUpToDate", "max_age_in_seconds"), &Steam::checkPingDataUpToDate);
	ClassDB::bind_method(D_METHOD("convertPingLocationToString", "location"), &Steam::convertPingLocationToString);
	ClassDB::bind_method(D_METHOD("estimatePingTimeBetweenTwoLocations", "location1", "location2"), &Steam::estimatePingTimeBetweenTwoLocations);
	ClassDB::bind_method(D_METHOD("estimatePingTimeFromLocalHost", "location"), &Steam::estimatePingTimeFromLocalHost);
	ClassDB::bind_method(D_METHOD("getConfigValue", "config_value", "scope_type", "connection_handle"), &Steam::getConfigValue);
	ClassDB::bind_method(D_METHOD("getConfigValueInfo", "config_value"), &Steam::getConfigValueInfo);
	ClassDB::bind_method(D_METHOD("getDirectPingToPOP", "pop_id"), &Steam::getDirectPingToPOP);
	ClassDB::bind_method("getLocalPingLocation", &Steam::getLocalPingLocation);
	ClassDB::bind_method("getLocalTimestamp", &Steam::getLocalTimestamp);
	ClassDB::bind_method(D_METHOD("getPingToDataCenter", "pop_id"), &Steam::getPingToDataCenter);
	ClassDB::bind_method("getPOPCount", &Steam::getPOPCount);
	ClassDB::bind_method("getPOPList", &Steam::getPOPList);
	ClassDB::bind_method("getRelayNetworkStatus", &Steam::getRelayNetworkStatus);
	ClassDB::bind_method("initRelayNetworkAccess", &Steam::initRelayNetworkAccess);
	ClassDB::bind_method(D_METHOD("parsePingLocationString", "string"), &Steam::parsePingLocationString);
	ClassDB::bind_method(D_METHOD("setConnectionConfigValueFloat", "connection", "config", "value"), &Steam::setConnectionConfigValueFloat);
	ClassDB::bind_method(D_METHOD("setConnectionConfigValueInt32", "connection", "config", "value"), &Steam::setConnectionConfigValueInt32);
	ClassDB::bind_method(D_METHOD("setConnectionConfigValueString", "connection", "config", "value"), &Steam::setConnectionConfigValueString);
//	ClassDB::bind_method(D_METHOD("setConfigValue", "setting", "scope_type", "connection_handle", "data_type", "value"), &Steam::setConfigValue);
	ClassDB::bind_method(D_METHOD("setGlobalConfigValueFloat", "config", "value"), &Steam::setGlobalConfigValueFloat);
	ClassDB::bind_method(D_METHOD("setGlobalConfigValueInt32", "config", "value"), &Steam::setGlobalConfigValueInt32);
	ClassDB::bind_method(D_METHOD("setGlobalConfigValueString", "config", "value"), &Steam::setGlobalConfigValueString);
	
	// PARENTAL SETTINGS BIND METHODS ///////////
	ClassDB::bind_method("isParentalLockEnabled", &Steam::isParentalLockEnabled);
	ClassDB::bind_method("isParentalLockLocked", &Steam::isParentalLockLocked);
	ClassDB::bind_method(D_METHOD("isAppBlocked", "app_id"), &Steam::isAppBlocked);
	ClassDB::bind_method(D_METHOD("isAppInBlockList", "app_id"), &Steam::isAppInBlockList);
	ClassDB::bind_method(D_METHOD("isFeatureBlocked", "feature"), &Steam::isFeatureBlocked);
	ClassDB::bind_method(D_METHOD("isFeatureInBlockList", "feature"), &Steam::isFeatureInBlockList);

	// PARTIES BIND METHODS /////////////////////
	ClassDB::bind_method(D_METHOD("cancelReservation", "beacon_id", "steam_id"), &Steam::cancelReservation);
	ClassDB::bind_method(D_METHOD("changeNumOpenSlots", "beacon_id", "open_slots"), &Steam::changeNumOpenSlots);
	ClassDB::bind_method(D_METHOD("createBeacon", "open_slots", "location_id", "type", "connect_string", "metadata"), &Steam::createBeacon);
	ClassDB::bind_method(D_METHOD("destroyBeacon", "beacon_id"), &Steam::destroyBeacon);
	ClassDB::bind_method(D_METHOD("getAvailableBeaconLocations", "max"), &Steam::getAvailableBeaconLocations);
	ClassDB::bind_method(D_METHOD("getBeaconByIndex", "index"), &Steam::getBeaconByIndex);
	ClassDB::bind_method(D_METHOD("getBeaconDetails", "beacon_id"), &Steam::getBeaconDetails);
	ClassDB::bind_method(D_METHOD("getBeaconLocationData", "location_id", "location_type", "location_data"), &Steam::getBeaconLocationData);
	ClassDB::bind_method("getNumActiveBeacons", &Steam::getNumActiveBeacons);
	ClassDB::bind_method(D_METHOD("joinParty", "beacon_id"), &Steam::joinParty);
	ClassDB::bind_method(D_METHOD("onReservationCompleted", "beacon_id", "steam_id"), &Steam::onReservationCompleted);
	
	// REMOTE PLAY BIND METHODS /////////////////
	ClassDB::bind_method("getSessionCount", &Steam::getSessionCount);
	ClassDB::bind_method(D_METHOD("getSessionID", "index"), &Steam::getSessionID);
	ClassDB::bind_method(D_METHOD("getSessionSteamID", "session_id"), &Steam::getSessionSteamID);
	ClassDB::bind_method(D_METHOD("getSessionClientName", "session_id"), &Steam::getSessionClientName);
	ClassDB::bind_method(D_METHOD("getSessionClientFormFactor", "session_id"), &Steam::getSessionClientFormFactor);
	ClassDB::bind_method(D_METHOD("getSessionClientResolution", "session_id"), &Steam::getSessionClientResolution);
	ClassDB::bind_method(D_METHOD("sendRemotePlayTogetherInvite", "friend_id"), &Steam::sendRemotePlayTogetherInvite);

	// REMOTE STORAGE BIND METHODS //////////////
	ClassDB::bind_method("beginFileWriteBatch", &Steam::beginFileWriteBatch);
	ClassDB::bind_method("endFileWriteBatch", &Steam::endFileWriteBatch);
	ClassDB::bind_method(D_METHOD("fileDelete", "file"), &Steam::fileDelete);
	ClassDB::bind_method(D_METHOD("fileExists", "file"), &Steam::fileExists);
	ClassDB::bind_method(D_METHOD("fileForget", "file"), &Steam::fileForget);
	ClassDB::bind_method(D_METHOD("filePersisted", "file"), &Steam::filePersisted);
	ClassDB::bind_method(D_METHOD("fileRead", "file", "data_to_read"), &Steam::fileRead);
	ClassDB::bind_method(D_METHOD("fileReadAsync", "file", "offset", "data_to_read"), &Steam::fileReadAsync);
	ClassDB::bind_method(D_METHOD("fileShare", "file"), &Steam::fileShare);
	ClassDB::bind_method(D_METHOD("fileWrite", "file", "data", "size"), &Steam::fileWrite, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("fileWriteAsync", "file", "data", "size"), &Steam::fileWriteAsync, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("fileWriteStreamCancel", "write_handle"), &Steam::fileWriteStreamCancel);
	ClassDB::bind_method(D_METHOD("fileWriteStreamClose", "write_handle"), &Steam::fileWriteStreamClose);
	ClassDB::bind_method(D_METHOD("fileWriteStreamOpen", "file"), &Steam::fileWriteStreamOpen);
	ClassDB::bind_method(D_METHOD("fileWriteStreamWriteChunk", "write_handle", "data"), &Steam::fileWriteStreamWriteChunk);
	ClassDB::bind_method("getCachedUGCCount", &Steam::getCachedUGCCount);
	ClassDB::bind_method(D_METHOD("getCachedUGCHandle", "content"), &Steam::getCachedUGCHandle);
	ClassDB::bind_method("getFileCount", &Steam::getFileCount);
	ClassDB::bind_method(D_METHOD("getFileNameAndSize", "file"), &Steam::getFileNameAndSize);
	ClassDB::bind_method(D_METHOD("getFileSize", "file"), &Steam::getFileSize);
	ClassDB::bind_method(D_METHOD("getFileTimestamp", "file"), &Steam::getFileTimestamp);
	ClassDB::bind_method(D_METHOD("getLocalFileChange", "file"), &Steam::getLocalFileChange);
	ClassDB::bind_method("getLocalFileChangeCount", &Steam::getLocalFileChangeCount);
	ClassDB::bind_method("getQuota", &Steam::getQuota);
	ClassDB::bind_method(D_METHOD("getSyncPlatforms", "file"), &Steam::getSyncPlatforms);
	ClassDB::bind_method(D_METHOD("getUGCDetails", "content"), &Steam::getUGCDetails);
	ClassDB::bind_method(D_METHOD("getUGCDownloadProgress", "content"), &Steam::getUGCDownloadProgress);
	ClassDB::bind_method("isCloudEnabledForAccount", &Steam::isCloudEnabledForAccount);
	ClassDB::bind_method("isCloudEnabledForApp", &Steam::isCloudEnabledForApp);
	ClassDB::bind_method(D_METHOD("setCloudEnabledForApp", "enabled"), &Steam::setCloudEnabledForApp);
	ClassDB::bind_method(D_METHOD("setSyncPlatforms", "file", "platform"), &Steam::setSyncPlatforms);
	ClassDB::bind_method(D_METHOD("ugcDownload", "content", "priority"), &Steam::ugcDownload);
	ClassDB::bind_method(D_METHOD("ugcDownloadToLocation", "content", "location", "priority"), &Steam::ugcDownloadToLocation);
	ClassDB::bind_method(D_METHOD("ugcRead", "content", "data_size", "offset", "action"), &Steam::ugcRead);
	
	// SCREENSHOT BIND METHODS //////////////////
	ClassDB::bind_method(D_METHOD("addScreenshotToLibrary", "filename", "thumbnail_filename", "width", "height"), &Steam::addScreenshotToLibrary);
	ClassDB::bind_method(D_METHOD("addVRScreenshotToLibrary", "type", "filename", "vr_filename"), &Steam::addVRScreenshotToLibrary);
	ClassDB::bind_method(D_METHOD("hookScreenshots", "hook"), &Steam::hookScreenshots);
	ClassDB::bind_method("isScreenshotsHooked", &Steam::isScreenshotsHooked);
	ClassDB::bind_method(D_METHOD("setLocation", "screenshot", "location"), &Steam::setLocation);
	ClassDB::bind_method(D_METHOD("tagPublishedFile", "screenshot", "file_id"), &Steam::tagPublishedFile);
	ClassDB::bind_method(D_METHOD("taguser", "screenshot", "steam_id"), &Steam::tagUser);
	ClassDB::bind_method("triggerScreenshot", &Steam::triggerScreenshot);
	ClassDB::bind_method(D_METHOD("writeScreenshot", "rgb", "width", "height"), &Steam::writeScreenshot);

	// UGC BIND METHODS ////////////////////
	ClassDB::bind_method(D_METHOD("addAppDependency", "published_file_id", "app_id"), &Steam::addAppDependency);
	ClassDB::bind_method(D_METHOD("addDependency", "published_file_id", "child_published_file_id"), &Steam::addDependency);
	ClassDB::bind_method(D_METHOD("addExcludedTag", "query_handle", "tag_name"), &Steam::addExcludedTag);
	ClassDB::bind_method(D_METHOD("addItemKeyValueTag", "query_handle", "key", "value"), &Steam::addItemKeyValueTag);
	ClassDB::bind_method(D_METHOD("addItemPreviewFile", "query_handle", "preview_file", "type"), &Steam::addItemPreviewFile);
	ClassDB::bind_method(D_METHOD("addItemPreviewVideo", "query_handle", "video_id"), &Steam::addItemPreviewVideo);
	ClassDB::bind_method(D_METHOD("addItemToFavorites", "app_id", "published_file_id"), &Steam::addItemToFavorites);
	ClassDB::bind_method(D_METHOD("addRequiredKeyValueTag", "query_handle", "key", "value"), &Steam::addRequiredKeyValueTag);
	ClassDB::bind_method(D_METHOD("addRequiredTag", "query_handle", "tag_name"), &Steam::addRequiredTag);
	ClassDB::bind_method(D_METHOD("addRequiredTagGroup", "query_handle", "tag_array"), &Steam::addRequiredTagGroup);
	ClassDB::bind_method(D_METHOD("initWorkshopForGameServer", "workshop_depot_id"), &Steam::initWorkshopForGameServer);
	ClassDB::bind_method(D_METHOD("createItem", "app_id", "file_type"), &Steam::createItem);
	ClassDB::bind_method(D_METHOD("createQueryAllUGCRequest", "query_type", "matching_type", "creator_id", "consumer_id", "page"), &Steam::createQueryAllUGCRequest);
	ClassDB::bind_method(D_METHOD("createQueryUGCDetailsRequest", "published_file_id"), &Steam::createQueryUGCDetailsRequest);
	ClassDB::bind_method(D_METHOD("createQueryUserUGCRequest", "account_id", "list_type", "matching_ugc_type", "sort_order", "creator_id", "consumer_id", "page"), &Steam::createQueryUserUGCRequest);
	ClassDB::bind_method(D_METHOD("deleteItem", "published_file_id"), &Steam::deleteItem);
	ClassDB::bind_method(D_METHOD("downloadItem", "published_file_id", "high_priority"), &Steam::downloadItem);
	ClassDB::bind_method(D_METHOD("getItemDownloadInfo", "published_file_id"), &Steam::getItemDownloadInfo);
	ClassDB::bind_method(D_METHOD("getItemInstallInfo", "published_file_id"), &Steam::getItemInstallInfo);
	ClassDB::bind_method(D_METHOD("getItemState", "published_file_id"), &Steam::getItemState);
	ClassDB::bind_method(D_METHOD("getItemUpdateProgress", "update_handle"), &Steam::getItemUpdateProgress);
	ClassDB::bind_method("getNumSubscribedItems", &Steam::getNumSubscribedItems);
	ClassDB::bind_method(D_METHOD("getQueryUGCAdditionalPreview", "query_handle", "index", "preview_index"), &Steam::getQueryUGCAdditionalPreview);
	ClassDB::bind_method(D_METHOD("getQueryUGCChildren", "query_handle", "index", "child_count"), &Steam::getQueryUGCChildren);
	ClassDB::bind_method(D_METHOD("getQueryUGCKeyValueTag", "query_handle", "index", "key_value_tag_index"), &Steam::getQueryUGCKeyValueTag);
	ClassDB::bind_method(D_METHOD("getQueryUGCMetadata", "query_handle", "index"), &Steam::getQueryUGCMetadata);
	ClassDB::bind_method(D_METHOD("getQueryUGCNumAdditionalPreviews", "query_handle", "index"), &Steam::getQueryUGCNumAdditionalPreviews);
	ClassDB::bind_method(D_METHOD("getQueryUGCNumKeyValueTags", "query_handle", "index"), &Steam::getQueryUGCNumKeyValueTags);
	ClassDB::bind_method(D_METHOD("getQueryUGCNumTags", "query_handle", "index"), &Steam::getQueryUGCNumTags);
	ClassDB::bind_method(D_METHOD("getQueryUGCPreviewURL", "query_handle", "index"), &Steam::getQueryUGCPreviewURL);
	ClassDB::bind_method(D_METHOD("getQueryUGCResult", "query_handle", "index"), &Steam::getQueryUGCResult);
	ClassDB::bind_method(D_METHOD("getQueryUGCStatistic", "query_handle", "index", "stat_type"), &Steam::getQueryUGCStatistic);
	ClassDB::bind_method(D_METHOD("getQueryUGCTag", "query_handle", "index", "tag_index"), &Steam::getQueryUGCTag);
	ClassDB::bind_method(D_METHOD("getQueryUGCTagDisplayName", "query_handle", "index", "tag_index"), &Steam::getQueryUGCTagDisplayName);
	ClassDB::bind_method("getSubscribedItems", &Steam::getSubscribedItems);
	ClassDB::bind_method(D_METHOD("getUserItemVote", "published_file_id"), &Steam::getUserItemVote);
	ClassDB::bind_method(D_METHOD("releaseQueryUGCRequest", "query_handle"), &Steam::releaseQueryUGCRequest);
	ClassDB::bind_method(D_METHOD("removeAppDependency", "published_file_id", "app_id"), &Steam::removeAppDependency);
	ClassDB::bind_method(D_METHOD("removeDependency", "published_file_id", "child_published_file_id"), &Steam::removeDependency);
	ClassDB::bind_method(D_METHOD("removeItemFromFavorites", "app_id", "published_file_id"), &Steam::removeItemFromFavorites);
	ClassDB::bind_method(D_METHOD("removeItemKeyValueTags", "update_handle", "key"), &Steam::removeItemKeyValueTags);
	ClassDB::bind_method(D_METHOD("removeItemPreview", "update_handle", "index"), &Steam::removeItemPreview);
	ClassDB::bind_method(D_METHOD("sendQueryUGCRequest", "update_handle"), &Steam::sendQueryUGCRequest);
	ClassDB::bind_method(D_METHOD("setAllowCachedResponse", "update_handle", "max_age_seconds"), &Steam::setAllowCachedResponse);
	ClassDB::bind_method(D_METHOD("setCloudFileNameFilter", "update_handle", "match_cloud_filename"), &Steam::setCloudFileNameFilter);
	ClassDB::bind_method(D_METHOD("setItemContent", "update_handle", "content_folder"), &Steam::setItemContent);
	ClassDB::bind_method(D_METHOD("setItemDescription", "update_handle", "description"), &Steam::setItemDescription);
	ClassDB::bind_method(D_METHOD("setItemMetadata", "update_handle", "metadata"), &Steam::setItemMetadata);
	ClassDB::bind_method(D_METHOD("setItemPreview", "update_handle", "preview_file"), &Steam::setItemPreview);
	ClassDB::bind_method(D_METHOD("setItemTags", "update_handle", "tag_array"), &Steam::setItemTags);
	ClassDB::bind_method(D_METHOD("setItemTitle", "update_handle", "title"), &Steam::setItemTitle);
	ClassDB::bind_method(D_METHOD("setItemUpdateLanguage", "update_handle", "language"), &Steam::setItemUpdateLanguage);
	ClassDB::bind_method(D_METHOD("setItemVisibility", "update_handle", "visibility"), &Steam::setItemVisibility);
	ClassDB::bind_method(D_METHOD("setLanguage", "query_handle", "language"), &Steam::setLanguage);
	ClassDB::bind_method(D_METHOD("setMatchAnyTag", "query_handle", "match_any_tag"), &Steam::setMatchAnyTag);
	ClassDB::bind_method(D_METHOD("setRankedByTrendDays", "query_handle", "days"), &Steam::setRankedByTrendDays);
	ClassDB::bind_method(D_METHOD("setReturnAdditionalPreviews", "query_handle", "return_additional_previews"), &Steam::setReturnAdditionalPreviews);
	ClassDB::bind_method(D_METHOD("setReturnChildren", "query_handle", "return_children"), &Steam::setReturnChildren);
	ClassDB::bind_method(D_METHOD("setReturnKeyValueTags", "query_handle", "return_key_value_tags"), &Steam::setReturnKeyValueTags);
	ClassDB::bind_method(D_METHOD("setReturnLongDescription", "query_handle", "return_long_description"), &Steam::setReturnLongDescription);
	ClassDB::bind_method(D_METHOD("setReturnMetadata", "query_handle", "return_metadata"), &Steam::setReturnMetadata);
	ClassDB::bind_method(D_METHOD("setReturnOnlyIDs", "query_handle", "return_only_ids"), &Steam::setReturnOnlyIDs);
	ClassDB::bind_method(D_METHOD("setReturnPlaytimeStats", "query_handle", "days"), &Steam::setReturnPlaytimeStats);
	ClassDB::bind_method(D_METHOD("setReturnTotalOnly", "query_handle", "return_total_only"), &Steam::setReturnTotalOnly);
	ClassDB::bind_method(D_METHOD("setSearchText", "query_handle", "search_text"), &Steam::setSearchText);
	ClassDB::bind_method(D_METHOD("setUserItemVote", "published_file_id", "vote_up"), &Steam::setUserItemVote);
	ClassDB::bind_method(D_METHOD("startItemUpdate", "app_id", "file_id"), &Steam::startItemUpdate);
	ClassDB::bind_method(D_METHOD("startPlaytimeTracking", "published_file_ids"), &Steam::startPlaytimeTracking);
	ClassDB::bind_method(D_METHOD("stopPlaytimeTracking", "published_file_ids"), &Steam::stopPlaytimeTracking);
	ClassDB::bind_method("stopPlaytimeTrackingForAllItems", &Steam::stopPlaytimeTrackingForAllItems);
	ClassDB::bind_method(D_METHOD("getAppDependencies", "published_file_id"), &Steam::getAppDependencies);
	ClassDB::bind_method(D_METHOD("submitItemUpdate", "update_handle", "change_note"), &Steam::submitItemUpdate);
	ClassDB::bind_method(D_METHOD("subscribeItem", "published_file_id"), &Steam::subscribeItem);
	ClassDB::bind_method(D_METHOD("suspendDownloads", "suspend"), &Steam::suspendDownloads);
	ClassDB::bind_method(D_METHOD("unsubscribeItem", "published_file_id"), &Steam::unsubscribeItem);
	ClassDB::bind_method(D_METHOD("updateItemPreviewFile", "update_handle", "index", "preview_file"), &Steam::updateItemPreviewFile);
	ClassDB::bind_method(D_METHOD("updateItemPreviewVideo", "update_handle", "index", "video_id"), &Steam::updateItemPreviewVideo);
	ClassDB::bind_method("showWorkshopEULA", &Steam::showWorkshopEULA);	
	ClassDB::bind_method("getWorkshopEULAStatus", &Steam::getWorkshopEULAStatus);
	ClassDB::bind_method(D_METHOD("setTimeCreatedDateRange", "update_handle", "start", "end"), &Steam::setTimeCreatedDateRange);
	ClassDB::bind_method(D_METHOD("setTimeUpdatedDateRange", "update_handle", "start", "end"), &Steam::setTimeUpdatedDateRange);

	// USER BIND METHODS ////////////////////////
	ClassDB::bind_method(D_METHOD("advertiseGame", "server_ip", "port"), &Steam::advertiseGame);
	ClassDB::bind_method(D_METHOD("beginAuthSession", "ticket", "ticket_size", "steam_id"), &Steam::beginAuthSession);
	ClassDB::bind_method(D_METHOD("cancelAuthTicket", "auth_ticket"), &Steam::cancelAuthTicket);
	ClassDB::bind_method(D_METHOD("decompressVoice", "voice", "voice_size", "sample_rate"), &Steam::decompressVoice);
	ClassDB::bind_method(D_METHOD("endAuthSession", "steam_id"), &Steam::endAuthSession);
	ClassDB::bind_method("getAuthSessionTicket", &Steam::getAuthSessionTicket);
	ClassDB::bind_method("getAvailableVoice", &Steam::getAvailableVoice);
	ClassDB::bind_method("getDurationControl", &Steam::getDurationControl);
	ClassDB::bind_method("getEncryptedAppTicket", &Steam::getEncryptedAppTicket);
	ClassDB::bind_method(D_METHOD("getGameBadgeLevel", "series", "foil"), &Steam::getGameBadgeLevel);
	ClassDB::bind_method("getPlayerSteamLevel", &Steam::getPlayerSteamLevel);
	ClassDB::bind_method("getSteamID", &Steam::getSteamID);
	ClassDB::bind_method("getVoice", &Steam::getVoice);
	ClassDB::bind_method("getVoiceOptimalSampleRate", &Steam::getVoiceOptimalSampleRate);
	ClassDB::bind_method(D_METHOD("initiateGameConnection", "server_id", "server_ip", "server_port", "secure"), &Steam::initiateGameConnection);
	ClassDB::bind_method("isBehindNAT", &Steam::isBehindNAT);
	ClassDB::bind_method("isPhoneIdentifying", &Steam::isPhoneIdentifying);
	ClassDB::bind_method("isPhoneRequiringVerification", &Steam::isPhoneRequiringVerification);
	ClassDB::bind_method("isPhoneVerified", &Steam::isPhoneVerified);
	ClassDB::bind_method("isTwoFactorEnabled", &Steam::isTwoFactorEnabled);
	ClassDB::bind_method("loggedOn", &Steam::loggedOn);
	ClassDB::bind_method(D_METHOD("requestEncryptedAppTicket", "secret"), &Steam::requestEncryptedAppTicket);
	ClassDB::bind_method(D_METHOD("requestStoreAuthURL", "redirect"), &Steam::requestStoreAuthURL);
	ClassDB::bind_method("startVoiceRecording", &Steam::startVoiceRecording);
	ClassDB::bind_method(D_METHOD("setDurationControlOnlineState", "new_state"), &Steam::setDurationControlOnlineState);
	ClassDB::bind_method("stopVoiceRecording", &Steam::stopVoiceRecording);
	ClassDB::bind_method(D_METHOD("terminateGameConnection", "server_ip", "server_port"), &Steam::terminateGameConnection);
	ClassDB::bind_method(D_METHOD("userHasLicenseForApp", "steam_id", "app_id"), &Steam::userHasLicenseForApp);
	
	// USER STATS BIND METHODS //////////////////
	ClassDB::bind_method(D_METHOD("attachLeaderboardUGC", "ugc_handle", "this_leaderboard"), &Steam::attachLeaderboardUGC, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("clearAchievement", "achievement_name"), &Steam::clearAchievement);
	ClassDB::bind_method(D_METHOD("downloadLeaderboardEntries", "start", "end", "type", "this_leaderboard"), &Steam::downloadLeaderboardEntries, DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("downloadLeaderboardEntriesForUsers", "users_id", "this_leaderboard"), &Steam::downloadLeaderboardEntriesForUsers, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("findLeaderboard", "leaderboard_name"), &Steam::findLeaderboard);
	ClassDB::bind_method(D_METHOD("findOrCreateLeaderboard", "leaderboard_name", "sort_method", "display_type"), &Steam::findOrCreateLeaderboard);
	ClassDB::bind_method(D_METHOD("getAchievement", "achievement_name"), &Steam::getAchievement);
	ClassDB::bind_method(D_METHOD("getAchievementAchievedPercent", "achievement_name"), &Steam::getAchievementAchievedPercent);
	ClassDB::bind_method(D_METHOD("getAchievementAndUnlockTime", "achievement_name"), &Steam::getAchievementAndUnlockTime);
	ClassDB::bind_method(D_METHOD("getAchievementDisplayAttribute", "achievement_name", "key"), &Steam::getAchievementDisplayAttribute);
	ClassDB::bind_method(D_METHOD("getAchievementIcon", "achievement_name"), &Steam::getAchievementIcon);
	ClassDB::bind_method(D_METHOD("getAchievementName", "achievement"), &Steam::getAchievementName);
	ClassDB::bind_method(D_METHOD("getAchievementProgressLimitsInt", "achievement_name"), &Steam::getAchievementProgressLimitsInt);
	ClassDB::bind_method(D_METHOD("getAchievementProgressLimitsFloat", "achievement_name"), &Steam::getAchievementProgressLimitsFloat);
	ClassDB::bind_method(D_METHOD("getGlobalStatInt", "stat_name"), &Steam::getGlobalStatInt);
	ClassDB::bind_method(D_METHOD("getGlobalStatFloat", "stat_name"), &Steam::getGlobalStatFloat);
	ClassDB::bind_method(D_METHOD("getGlobalStatIntHistory", "stat_name"), &Steam::getGlobalStatIntHistory);
	ClassDB::bind_method(D_METHOD("getGlobalStatFloatHistory", "stat_name"), &Steam::getGlobalStatFloatHistory);
	ClassDB::bind_method(D_METHOD("getLeaderboardDisplayType", "this_leaderboard"), &Steam::getLeaderboardDisplayType, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("getLeaderboardEntryCount", "this_leaderboard"), &Steam::getLeaderboardEntryCount, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("getLeaderboardName", "this_leaderboard"), &Steam::getLeaderboardName, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("getLeaderboardSortMethod", "this_leaderboard"), &Steam::getLeaderboardSortMethod, DEFVAL(0));
	ClassDB::bind_method("getMostAchievedAchievementInfo", &Steam::getMostAchievedAchievementInfo);
	ClassDB::bind_method(D_METHOD("getNextMostAchievedAchievementInfo", "iterator"), &Steam::getNextMostAchievedAchievementInfo);
	ClassDB::bind_method("getNumAchievements", &Steam::getNumAchievements);
	ClassDB::bind_method("getNumberOfCurrentPlayers", &Steam::getNumberOfCurrentPlayers);
	ClassDB::bind_method(D_METHOD("getStatFloat", "stat_name"), &Steam::getStatFloat);
	ClassDB::bind_method(D_METHOD("getStatInt", "stat_name"), &Steam::getStatInt);
	ClassDB::bind_method(D_METHOD("getUserAchievement", "steam_id", "name"), &Steam::getUserAchievement);
	ClassDB::bind_method(D_METHOD("getUserAchievementAndUnlockTime", "steam_id", "name"), &Steam::getUserAchievementAndUnlockTime);
	ClassDB::bind_method(D_METHOD("getUserStatFloat", "steam_id", "name"), &Steam::getUserStatFloat);
	ClassDB::bind_method(D_METHOD("getUserStatInt", "steam_id", "name"), &Steam::getUserStatInt);
	ClassDB::bind_method(D_METHOD("indicateAchievementProgress", "name", "current_progress", "max_progress"), &Steam::indicateAchievementProgress);
	ClassDB::bind_method("requestCurrentStats", &Steam::requestCurrentStats);
	ClassDB::bind_method("requestGlobalAchievementPercentages", &Steam::requestGlobalAchievementPercentages);
	ClassDB::bind_method(D_METHOD("requestGlobalStats", "history_days"), &Steam::requestGlobalStats);
	ClassDB::bind_method(D_METHOD("requestUserStats", "steam_id"), &Steam::requestUserStats);
	ClassDB::bind_method(D_METHOD("resetAllStats", "achievements_too"), &Steam::resetAllStats);
	ClassDB::bind_method(D_METHOD("setAchievement", "name"), &Steam::setAchievement);
	ClassDB::bind_method(D_METHOD("setLeaderboardDetailsMax", "max"), &Steam::setLeaderboardDetailsMax);
	ClassDB::bind_method(D_METHOD("setStatFloat", "name", "value"), &Steam::setStatFloat);
	ClassDB::bind_method(D_METHOD("setStatInt", "name", "value"), &Steam::setStatInt);
	ClassDB::bind_method("storeStats", &Steam::storeStats);
	ClassDB::bind_method(D_METHOD("updateAvgRateStat", "name", "this_session", "session_length"), &Steam::updateAvgRateStat);
	ClassDB::bind_method(D_METHOD("uploadLeaderboardScore", "score", "keep_best", "details", "this_leaderboard"), &Steam::uploadLeaderboardScore, DEFVAL(true), DEFVAL(PoolIntArray()), DEFVAL(0));
	ClassDB::bind_method("getLeaderboardEntries", &Steam::getLeaderboardEntries);

	// UTILS BIND METHODS ///////////////////////
	ClassDB::bind_method(D_METHOD("filterText", "context", "steam_id", "message"), &Steam::filterText);
	ClassDB::bind_method("getAPICallFailureReason", &Steam::getAPICallFailureReason);
	ClassDB::bind_method("getAppID", &Steam::getAppID);
	ClassDB::bind_method("getCurrentBatteryPower", &Steam::getCurrentBatteryPower);
	ClassDB::bind_method(D_METHOD("getImageRGBA", "image"), &Steam::getImageRGBA);
	ClassDB::bind_method(D_METHOD("getImageSize", "image"), &Steam::getImageSize);
	ClassDB::bind_method("getIPCCallCount", &Steam::getIPCCallCount);
	ClassDB::bind_method("getIPCountry", &Steam::getIPCountry);
	ClassDB::bind_method("getSecondsSinceAppActive", &Steam::getSecondsSinceAppActive);
	ClassDB::bind_method("getSecondsSinceComputerActive", &Steam::getSecondsSinceComputerActive);
	ClassDB::bind_method("getServerRealTime", &Steam::getServerRealTime);
	ClassDB::bind_method("getSteamUILanguage", &Steam::getSteamUILanguage);
	ClassDB::bind_method("initFilterText", &Steam::initFilterText);
	ClassDB::bind_method("isAPICallCompleted", &Steam::isAPICallCompleted);
	ClassDB::bind_method("isOverlayEnabled", &Steam::isOverlayEnabled);
	ClassDB::bind_method("isSteamChinaLauncher", &Steam::isSteamChinaLauncher);
	ClassDB::bind_method("isSteamInBigPictureMode", &Steam::isSteamInBigPictureMode);
	ClassDB::bind_method("isSteamRunningInVR", &Steam::isSteamRunningInVR);
	ClassDB::bind_method("isVRHeadsetStreamingEnabled", &Steam::isVRHeadsetStreamingEnabled);
	ClassDB::bind_method("overlayNeedsPresent", &Steam::overlayNeedsPresent);
	ClassDB::bind_method(D_METHOD("setOverlayNotificationInset", "horizontal", "vertical"), &Steam::setOverlayNotificationInset);
	ClassDB::bind_method(D_METHOD("setOverlayNotificationPosition", "pos"), &Steam::setOverlayNotificationPosition);
	ClassDB::bind_method(D_METHOD("setVRHeadsetStreamingEnabled", "enabled"), &Steam::setVRHeadsetStreamingEnabled);
	ClassDB::bind_method(D_METHOD("showGamepadTextInput", "input_mode", "line_input_mode", "description", "max_text", "preset_text"), &Steam::showGamepadTextInput);
	ClassDB::bind_method(D_METHOD("showFloatingGamepadTextInput", "input_mode", "text_field_x_position", "text_field_y_position", "text_field_width", "text_field_height"), &Steam::showFloatingGamepadTextInput);
	ClassDB::bind_method(D_METHOD("setGameLauncherMode", "mode"), &Steam::setGameLauncherMode);
	ClassDB::bind_method("startVRDashboard", &Steam::startVRDashboard);	
	ClassDB::bind_method("isSteamRunningOnSteamDeck", &Steam::isSteamRunningOnSteamDeck);
	ClassDB::bind_method("dismissFloatingGamepadTextInput", &Steam::dismissFloatingGamepadTextInput);

	// VIDEO BIND METHODS ///////////////////////
	ClassDB::bind_method(D_METHOD("getOPFSettings", "app_id"), &Steam::getOPFSettings);
	ClassDB::bind_method(D_METHOD("getOPFStringForApp", "app_id"), &Steam::getOPFStringForApp);
	ClassDB::bind_method(D_METHOD("getVideoURL" "app_id"), &Steam::getVideoURL);
	ClassDB::bind_method("isBroadcasting", &Steam::isBroadcasting);

	/////////////////////////////////////////////
	// CALLBACK SIGNAL BINDS
	/////////////////////////////////////////////
	//
	// STEAMWORKS SIGNALS ///////////////////////
	ADD_SIGNAL(MethodInfo("steamworks_error", PropertyInfo(Variant::STRING, "failed_signal"), PropertyInfo(Variant::STRING, "io failure")));

	// APPS SIGNALS /////////////////////////////
	ADD_SIGNAL(MethodInfo("file_details_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "fileSize"), PropertyInfo(Variant::INT, "fileHash"), PropertyInfo(Variant::INT, "flags")));
	ADD_SIGNAL(MethodInfo("dlc_installed", PropertyInfo(Variant::INT, "app")));
	ADD_SIGNAL(MethodInfo("new_launch_url_parameters"));
	ADD_SIGNAL(MethodInfo("timed_trial_status", PropertyInfo(Variant::INT, "app_id"), PropertyInfo(Variant::BOOL, "is_offline"), PropertyInfo(Variant::INT, "seconds_allowed"), PropertyInfo(Variant::INT, "seconds_played")));

	// APP LIST SIGNALS /////////////////////////
	ADD_SIGNAL(MethodInfo("app_installed", PropertyInfo(Variant::INT, "app_id"), PropertyInfo(Variant::INT, "install_folder_index")));
	ADD_SIGNAL(MethodInfo("app_uninstalled", PropertyInfo(Variant::INT, "app_id"), PropertyInfo(Variant::INT, "install_folder_index")));

	// FRIENDS SIGNALS //////////////////////////
	ADD_SIGNAL(MethodInfo("avatar_loaded", PropertyInfo(Variant::INT, "steam_id"), PropertyInfo(Variant::INT, "size"), PropertyInfo(Variant::ARRAY, "data")));
	ADD_SIGNAL(MethodInfo("request_clan_officer_list", PropertyInfo(Variant::STRING, "message"), PropertyInfo(Variant::ARRAY, "officer_list")));
	ADD_SIGNAL(MethodInfo("clan_activity_downloaded", PropertyInfo(Variant::DICTIONARY, "activity")));
	ADD_SIGNAL(MethodInfo("friend_rich_presence_update", PropertyInfo(Variant::INT, "steam_id"), PropertyInfo(Variant::INT, "app_id")));
	ADD_SIGNAL(MethodInfo("enumerate_following_list", PropertyInfo(Variant::STRING, "message"), PropertyInfo(Variant::ARRAY, "following")));
	ADD_SIGNAL(MethodInfo("get_follower_count", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "steam_id"), PropertyInfo(Variant::INT, "count")));
	ADD_SIGNAL(MethodInfo("is_following", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "steam_id"), PropertyInfo(Variant::BOOL, "following")));
	ADD_SIGNAL(MethodInfo("connected_chat_join", PropertyInfo(Variant::INT, "chat_id"), PropertyInfo(Variant::INT, "steam_id")));
	ADD_SIGNAL(MethodInfo("connected_chat_leave", PropertyInfo(Variant::INT, "chat_id"), PropertyInfo(Variant::INT, "steam_id"), PropertyInfo(Variant::BOOL, "kicked"), PropertyInfo(Variant::BOOL, "dropped")));
	ADD_SIGNAL(MethodInfo("connected_clan_chat_message", PropertyInfo(Variant::DICTIONARY, "chat")));
	ADD_SIGNAL(MethodInfo("connected_friend_chat_message", PropertyInfo(Variant::DICTIONARY, "chat")));
	ADD_SIGNAL(MethodInfo("join_requested", PropertyInfo(Variant::INT, "lobby_id"), PropertyInfo(Variant::STRING, "steam_id")));
	ADD_SIGNAL(MethodInfo("overlay_toggled", PropertyInfo(Variant::BOOL, "active")));
	ADD_SIGNAL(MethodInfo("join_game_requested", PropertyInfo(Variant::INT, "user"), PropertyInfo(Variant::STRING, "connect")));
	ADD_SIGNAL(MethodInfo("change_server_requested", PropertyInfo(Variant::STRING, "server"), PropertyInfo(Variant::STRING, "password")));
	ADD_SIGNAL(MethodInfo("join_clan_chat_complete", PropertyInfo(Variant::INT, "chat_id"), PropertyInfo(Variant::INT, "response")));
	ADD_SIGNAL(MethodInfo("persona_state_change", PropertyInfo(Variant::INT, "steam_id"), PropertyInfo(Variant::INT, "flags")));
	ADD_SIGNAL(MethodInfo("name_changed", PropertyInfo(Variant::BOOL, "success"), PropertyInfo(Variant::BOOL, "local_success"), PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("overlay_browser_protocol", PropertyInfo(Variant::STRING, "uri")));
	ADD_SIGNAL(MethodInfo("unread_chat_messages_changed"));
	ADD_SIGNAL(MethodInfo("equipped_profile_items_changed"));
	ADD_SIGNAL(MethodInfo("equipped_profile_items"));

	// GAME SEARCH SIGNALS //////////////////////
	ADD_SIGNAL(MethodInfo("search_for_game_progress", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "search_id"), PropertyInfo(Variant::DICTIONARY, "search_progress")));
	ADD_SIGNAL(MethodInfo("search_for_game_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "search_id"), PropertyInfo(Variant::DICTIONARY, "search_result")));
	ADD_SIGNAL(MethodInfo("request_players_for_game_progress", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "search_id")));
	ADD_SIGNAL(MethodInfo("request_players_for_game_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "search_id"), PropertyInfo(Variant::DICTIONARY, "player_data")));
	ADD_SIGNAL(MethodInfo("request_players_for_game_final_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "search_id"), PropertyInfo(Variant::INT, "game_id")));
	ADD_SIGNAL(MethodInfo("submit_player_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "game_id"), PropertyInfo(Variant::INT, "player_id")));
	ADD_SIGNAL(MethodInfo("end_game_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "game_id")));

	// GAME SERVER SIGNALS //////////////////////
	ADD_SIGNAL(MethodInfo("server_connect_failure", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::BOOL, "retrying")));
	ADD_SIGNAL(MethodInfo("server_connected"));
	ADD_SIGNAL(MethodInfo("server_disconnected", PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("client_approved", PropertyInfo(Variant::INT, "steam_id"), PropertyInfo(Variant::INT, "owner_id")));
	ADD_SIGNAL(MethodInfo("client_denied", PropertyInfo(Variant::INT, "steam_id"), PropertyInfo(Variant::INT, "reason")));
	ADD_SIGNAL(MethodInfo("client_kick", PropertyInfo(Variant::INT, "steam_id"), PropertyInfo(Variant::INT, "reason")));
	ADD_SIGNAL(MethodInfo("policy_response", PropertyInfo(Variant::INT, "secure")));
	ADD_SIGNAL(MethodInfo("client_group_status", PropertyInfo(Variant::INT, "steam_id"), PropertyInfo(Variant::INT, "group_id"), PropertyInfo(Variant::BOOL, "member"), PropertyInfo(Variant::BOOL, "officer")));
	ADD_SIGNAL(MethodInfo("associate_clan", PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("player_compat", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "players_dont_like_candidate"), PropertyInfo(Variant::INT, "players_candidate_doesnt_like"), PropertyInfo(Variant::INT, "clan_players_dont_like_candidate"), PropertyInfo(Variant::INT, "steam_id")));

	// GAME SERVER STATS SIGNALS ////////////////
	ADD_SIGNAL(MethodInfo("stats_received", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "steam_id")));
	ADD_SIGNAL(MethodInfo("stats_stored", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "steam_id")));
	ADD_SIGNAL(MethodInfo("stats_unloaded", PropertyInfo(Variant::INT, "steam_id")));

	// HTML SURFACE SIGNALS /////////////////////
	ADD_SIGNAL(MethodInfo("html_browser_ready", PropertyInfo(Variant::INT, "browser_handle")));
	ADD_SIGNAL(MethodInfo("html_can_go_backandforward", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::BOOL, "go_back"), PropertyInfo(Variant::BOOL, "go_forward")));
	ADD_SIGNAL(MethodInfo("html_changed_title", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::STRING, "title")));
	ADD_SIGNAL(MethodInfo("html_close_browser", PropertyInfo(Variant::INT, "browser_handle")));
	ADD_SIGNAL(MethodInfo("html_file_open_dialog", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::STRING, "title"), PropertyInfo(Variant::STRING, "initial_file")));
	ADD_SIGNAL(MethodInfo("html_finished_request", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::STRING, "url"), PropertyInfo(Variant::STRING, "title")));
	ADD_SIGNAL(MethodInfo("html_hide_tooltip", PropertyInfo(Variant::INT, "browser_handle")));
	ADD_SIGNAL(MethodInfo("html_horizontal_scroll", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::DICTIONARY, "scroll_data")));
	ADD_SIGNAL(MethodInfo("html_js_alert", PropertyInfo(Variant::INT, "browser_handle"),  PropertyInfo(Variant::STRING, "message")));
	ADD_SIGNAL(MethodInfo("html_js_confirm", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::STRING, "message")));
	ADD_SIGNAL(MethodInfo("html_link_at_position", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::DICTIONARY, "link_data")));
	ADD_SIGNAL(MethodInfo("html_needs_paint", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::DICTIONARY, "page_data")));
	ADD_SIGNAL(MethodInfo("html_new_window", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::DICTIONARY, "window_data")));
	ADD_SIGNAL(MethodInfo("html_open_link_in_new_tab", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::STRING, "url")));
	ADD_SIGNAL(MethodInfo("html_search_results", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::INT, "results"), PropertyInfo(Variant::INT, "current_match")));
	ADD_SIGNAL(MethodInfo("html_set_cursor", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::INT, "mouse_cursor")));
	ADD_SIGNAL(MethodInfo("html_show_tooltip", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::STRING, "message")));
	ADD_SIGNAL(MethodInfo("html_start_request", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::STRING, "url"), PropertyInfo(Variant::STRING, "target"), PropertyInfo(Variant::STRING, "post_data"), PropertyInfo(Variant::BOOL, "redirect")));
	ADD_SIGNAL(MethodInfo("html_status_text", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::STRING, "message")));
	ADD_SIGNAL(MethodInfo("html_update_tooltip", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::STRING, "message")));
	ADD_SIGNAL(MethodInfo("html_url_changed", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::DICTIONARY, "url_data")));
	ADD_SIGNAL(MethodInfo("html_vertical_scroll", PropertyInfo(Variant::INT, "browser_handle"), PropertyInfo(Variant::DICTIONARY, "scroll_data")));

	// HTTP SIGNALS /////////////////////////////
	ADD_SIGNAL(MethodInfo("http_request_completed", PropertyInfo(Variant::INT, "cookie_handle"), PropertyInfo(Variant::INT, "context_value"), PropertyInfo(Variant::BOOL, "request_success"), PropertyInfo(Variant::INT, "status_code"), PropertyInfo(Variant::INT, "body_size")));
	ADD_SIGNAL(MethodInfo("http_request_data_received", PropertyInfo(Variant::INT, "cookie_handle"), PropertyInfo(Variant::INT, "context_value"), PropertyInfo(Variant::INT, "offset"), PropertyInfo(Variant::INT, "bytes_received")));
	ADD_SIGNAL(MethodInfo("http_request_headers_received", PropertyInfo(Variant::INT, "cookie_handle"), PropertyInfo(Variant::INT, "context_value")));

	// INPUT SIGNALS ////////////////////////////
	ADD_SIGNAL(MethodInfo("input_action_event"));
	ADD_SIGNAL(MethodInfo("input_device_connected", PropertyInfo(Variant::INT, "input_handle")));
	ADD_SIGNAL(MethodInfo("input_device_disconnected", PropertyInfo(Variant::INT, "input_handle")));
	ADD_SIGNAL(MethodInfo("input_configuration_loaded", PropertyInfo(Variant::INT, "app_id"), PropertyInfo(Variant::INT, "device_handle"), PropertyInfo(Variant::DICTIONARY, "config_data")));

	// INVENTORY SIGNALS ////////////////////////
	ADD_SIGNAL(MethodInfo("inventory_definition_update", PropertyInfo(Variant::ARRAY, "definitions")));
	ADD_SIGNAL(MethodInfo("inventory_eligible_promo_item", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::BOOL, "cached"), PropertyInfo(Variant::ARRAY, "definitions")));
	ADD_SIGNAL(MethodInfo("inventory_full_update", PropertyInfo(Variant::INT, "inventory_handle")));
	ADD_SIGNAL(MethodInfo("inventory_result_ready", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "inventory_handle")));
	ADD_SIGNAL(MethodInfo("inventory_start_purchase_result", PropertyInfo(Variant::STRING, "result"), PropertyInfo(Variant::INT, "order_id"), PropertyInfo(Variant::INT, "transaction_id")));
	ADD_SIGNAL(MethodInfo("inventory_request_prices_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::STRING, "currency")));

	// MATCHMAKING SIGNALS //////////////////////
	ADD_SIGNAL(MethodInfo("favorites_list_accounts_updated", PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("favorites_list_changed", PropertyInfo(Variant::DICTIONARY, "favorite")));
	ADD_SIGNAL(MethodInfo("lobby_message", PropertyInfo(Variant::INT, "lobby_id"), PropertyInfo(Variant::INT, "user"), PropertyInfo(Variant::STRING, "message"), PropertyInfo(Variant::INT, "chat_type")));
	ADD_SIGNAL(MethodInfo("lobby_chat_update", PropertyInfo(Variant::INT, "lobby_id"), PropertyInfo(Variant::INT, "changed_id"), PropertyInfo(Variant::INT, "making_change_id"), PropertyInfo(Variant::INT, "chat_state")));
	ADD_SIGNAL(MethodInfo("lobby_created", PropertyInfo(Variant::INT, "connect"), PropertyInfo(Variant::INT, "lobby_id")));
	ADD_SIGNAL(MethodInfo("lobby_data_update"));
	ADD_SIGNAL(MethodInfo("lobby_joined", PropertyInfo(Variant::INT, "lobby"), PropertyInfo(Variant::INT, "permissions"), PropertyInfo(Variant::BOOL, "locked"), PropertyInfo(Variant::INT, "response")));
	ADD_SIGNAL(MethodInfo("lobby_game_created", PropertyInfo(Variant::INT, "lobby_id"), PropertyInfo(Variant::INT, "server_id"), PropertyInfo(Variant::STRING, "server_ip"), PropertyInfo(Variant::INT, "port")));
	ADD_SIGNAL(MethodInfo("lobby_invite", PropertyInfo(Variant::INT, "inviter"), PropertyInfo(Variant::INT, "lobby"), PropertyInfo(Variant::INT, "game")));
	ADD_SIGNAL(MethodInfo("lobby_match_list", PropertyInfo(Variant::ARRAY, "lobbies")));
	ADD_SIGNAL(MethodInfo("lobby_kicked", PropertyInfo(Variant::INT, "lobby_id"), PropertyInfo(Variant::INT, "admin_id"), PropertyInfo(Variant::INT, "due_to_disconnect")));
	
	// MATCHMAKING SERVER SIGNALS ///////////////
	ADD_SIGNAL(MethodInfo("server_responded"));
	ADD_SIGNAL(MethodInfo("server_failed_to_respond"));

	// MUSIC REMOTE SIGNALS /////////////////////
	ADD_SIGNAL(MethodInfo("music_player_remote_to_front"));
	ADD_SIGNAL(MethodInfo("music_player_remote_will_activate"));
	ADD_SIGNAL(MethodInfo("music_player_remote_will_deactivate"));
	ADD_SIGNAL(MethodInfo("music_player_selects_playlist_entry", PropertyInfo(Variant::INT, "entry")));
	ADD_SIGNAL(MethodInfo("music_player_selects_queue_entry", PropertyInfo(Variant::INT, "entry")));
	ADD_SIGNAL(MethodInfo("music_player_wants_looped", PropertyInfo(Variant::BOOL, "looped")));
	ADD_SIGNAL(MethodInfo("music_player_wants_pause"));
	ADD_SIGNAL(MethodInfo("music_player_wants_playing_repeat_status", PropertyInfo(Variant::INT, "status")));
	ADD_SIGNAL(MethodInfo("music_player_wants_play_next"));
	ADD_SIGNAL(MethodInfo("music_player_wants_play_previous"));
	ADD_SIGNAL(MethodInfo("music_player_wants_play"));
	ADD_SIGNAL(MethodInfo("music_player_wants_shuffled", PropertyInfo(Variant::BOOL, "shuffled")));
	ADD_SIGNAL(MethodInfo("music_player_wants_volume", PropertyInfo(Variant::REAL, "volume")));
	ADD_SIGNAL(MethodInfo("music_player_will_quit"));

	// NETWORKING SIGNALS ///////////////////////
	ADD_SIGNAL(MethodInfo("p2p_session_request", PropertyInfo(Variant::INT, "steam_id_remote")));
	ADD_SIGNAL(MethodInfo("p2p_session_connect_fail", PropertyInfo(Variant::INT, "steam_id_remote"), PropertyInfo(Variant::INT, "session_error")));

	// NETWORKING MESSAGES //////////////////////
	ADD_SIGNAL(MethodInfo("network_messages_session_request", PropertyInfo(Variant::STRING, "identity")));
	ADD_SIGNAL(MethodInfo("network_messages_session_failed", PropertyInfo(Variant::INT, "reason")));

	// NETWORKING SOCKETS SIGNALS ///////////////
	ADD_SIGNAL(MethodInfo("network_connection_status_changed", PropertyInfo(Variant::INT, "connect_handle"), PropertyInfo(Variant::DICTIONARY, "connection"), PropertyInfo(Variant::INT, "old_state")));
	ADD_SIGNAL(MethodInfo("network_authentication_status", PropertyInfo(Variant::INT, "available"), PropertyInfo(Variant::STRING, "debug_message")));
	ADD_SIGNAL(MethodInfo("fake_ip_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::STRING, "identity"), PropertyInfo(Variant::STRING, "fake_ip"), PropertyInfo(Variant::ARRAY, "port_list")));

	// NETWORKING UTILS SIGNALS /////////////////
	ADD_SIGNAL(MethodInfo("relay_network_status", PropertyInfo(Variant::INT, "available"), PropertyInfo(Variant::INT, "ping_measurement"), PropertyInfo(Variant::INT, "available_config"), PropertyInfo(Variant::INT, "available_relay"), PropertyInfo(Variant::STRING, "debug_message")));

	// PARENTAL SETTINGS SIGNALS ////////////////
	ADD_SIGNAL(MethodInfo("parental_setting_changed"));

	// PARTIES SIGANLS //////////////////////////
	ADD_SIGNAL(MethodInfo("join_party", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "beacon_id"), PropertyInfo(Variant::INT, "steam_id"), PropertyInfo(Variant::STRING, "connect_string")));
	ADD_SIGNAL(MethodInfo("create_beacon", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "beacon_id")));
	ADD_SIGNAL(MethodInfo("reservation_notification", PropertyInfo(Variant::INT, "beacon_id"), PropertyInfo(Variant::INT, "steam_id")));
	ADD_SIGNAL(MethodInfo("change_num_open_slots", PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("available_beacon_locations_updated"));
	ADD_SIGNAL(MethodInfo("active_beacons_updated"));

	// REMOTE PLAY SIGNALS //////////////////////
	ADD_SIGNAL(MethodInfo("remote_play_session_connected", PropertyInfo(Variant::INT, "session_id")));
	ADD_SIGNAL(MethodInfo("remote_play_session_disconnected", PropertyInfo(Variant::INT, "session_id")));

	// REMOTE STORAGE SIGNALS ///////////////////
	ADD_SIGNAL(MethodInfo("file_read_async_complete", PropertyInfo(Variant::DICTIONARY, "file_read")));
	ADD_SIGNAL(MethodInfo("file_share_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "handle"), PropertyInfo(Variant::STRING, "name")));
	ADD_SIGNAL(MethodInfo("file_write_async_complete", PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("download_ugc_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::DICTIONARY, "download_data")));
	ADD_SIGNAL(MethodInfo("unsubscribe_item", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id")));
	ADD_SIGNAL(MethodInfo("subscribe_item", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id")));
	ADD_SIGNAL(MethodInfo("local_file_changed"));

	// SCREENSHOT SIGNALS ///////////////////////
	ADD_SIGNAL(MethodInfo("screenshot_ready", PropertyInfo(Variant::INT, "handle"), PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("screenshot_requested"));

	// UGC SIGNALS //////////////////////////////
	ADD_SIGNAL(MethodInfo("add_app_dependency_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id"), PropertyInfo(Variant::INT, "app_id")));
	ADD_SIGNAL(MethodInfo("add_ugc_dependency_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id"), PropertyInfo(Variant::INT, "child_id")));
	ADD_SIGNAL(MethodInfo("item_created", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id"), PropertyInfo(Variant::BOOL, "accept_tos")));
	ADD_SIGNAL(MethodInfo("item_downloaded", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id"), PropertyInfo(Variant::INT, "app_id")));
	ADD_SIGNAL(MethodInfo("get_app_dependencies_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id"), PropertyInfo(Variant::INT, "app_dependencies"), PropertyInfo(Variant::INT, "total_app_dependencies")));
	ADD_SIGNAL(MethodInfo("item_deleted", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id")));
	ADD_SIGNAL(MethodInfo("get_item_vote_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id"), PropertyInfo(Variant::BOOL, "vote_up"), PropertyInfo(Variant::BOOL, "vote_down"), PropertyInfo(Variant::BOOL, "vote_skipped")));
	ADD_SIGNAL(MethodInfo("item_installed", PropertyInfo(Variant::INT, "app_id"), PropertyInfo(Variant::INT, "file_id")));
	ADD_SIGNAL(MethodInfo("remove_app_dependency_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id"), PropertyInfo(Variant::INT, "app_id")));
	ADD_SIGNAL(MethodInfo("remove_ugc_dependency_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id"), PropertyInfo(Variant::INT, "child_id")));
	ADD_SIGNAL(MethodInfo("set_user_item_vote", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id"), PropertyInfo(Variant::BOOL, "vote_up")));
	ADD_SIGNAL(MethodInfo("start_playtime_tracking", PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("ugc_query_completed", PropertyInfo(Variant::INT, "handle"), PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "results_returned"), PropertyInfo(Variant::INT, "total_matching"), PropertyInfo(Variant::BOOL, "cached")));
	ADD_SIGNAL(MethodInfo("stop_playtime_tracking", PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("item_updated", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::BOOL, "accept_tos")));
	ADD_SIGNAL(MethodInfo("user_favorite_items_list_changed", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "file_id"), PropertyInfo(Variant::BOOL, "was_add_request")));
	ADD_SIGNAL(MethodInfo("workshop_eula_status", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "app_id"), PropertyInfo(Variant::DICTIONARY, "eula_data")));
	ADD_SIGNAL(MethodInfo("user_subscribed_items_list_changed", PropertyInfo(Variant::INT, "app_id")));

	// USER SIGNALS /////////////////////////////
	ADD_SIGNAL(MethodInfo("client_game_server_deny", PropertyInfo(Variant::INT, "app_id"), PropertyInfo(Variant::STRING, "ip"), PropertyInfo(Variant::INT, "server_port"), PropertyInfo(Variant::INT, "secure"), PropertyInfo(Variant::INT, "reason")));
	ADD_SIGNAL(MethodInfo("duration_control", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::DICTIONARY, "duration")));
	ADD_SIGNAL(MethodInfo("encrypted_app_ticket_response", PropertyInfo(Variant::STRING, "result")));
	ADD_SIGNAL(MethodInfo("game_web_callback", PropertyInfo(Variant::STRING, "url")));
	ADD_SIGNAL(MethodInfo("get_auth_session_ticket_response", PropertyInfo(Variant::INT, "auth_ticket"), PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("ipc_failure", PropertyInfo(Variant::INT, "type")));
	ADD_SIGNAL(MethodInfo("licenses_updated"));
	ADD_SIGNAL(MethodInfo("microstransaction_auth_response", PropertyInfo(Variant::INT, "app_id"), PropertyInfo(Variant::INT, "order_id"), PropertyInfo(Variant::BOOL, "authorized")));
	ADD_SIGNAL(MethodInfo("steam_server_connect_failed", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::BOOL, "retrying")));
	ADD_SIGNAL(MethodInfo("steam_server_connected"));
	ADD_SIGNAL(MethodInfo("steam_server_disconnected"));
	ADD_SIGNAL(MethodInfo("store_auth_url_response", PropertyInfo(Variant::STRING, "url")));
	ADD_SIGNAL(MethodInfo("validate_auth_ticket_response", PropertyInfo(Variant::INT, "auth_id"), PropertyInfo(Variant::INT, "reponse"), PropertyInfo(Variant::INT, "owner_id")));

	// USER STATS SIGNALS ///////////////////////
	ADD_SIGNAL(MethodInfo("global_achievement_percentages_ready", PropertyInfo(Variant::INT, "game_id"), PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("global_stats_received", PropertyInfo(Variant::INT, "game_id"), PropertyInfo(Variant::STRING, "result")));
	ADD_SIGNAL(MethodInfo("leaderboard_find_result", PropertyInfo(Variant::INT, "leaderboard_handle"), PropertyInfo(Variant::INT, "found")));
	ADD_SIGNAL(MethodInfo("leaderboard_scores_downloaded", PropertyInfo(Variant::STRING, "message"), PropertyInfo(Variant::ARRAY, "leaderboard_entries")));
	ADD_SIGNAL(MethodInfo("leaderboard_score_uploaded", PropertyInfo(Variant::BOOL, "success"), PropertyInfo(Variant::INT, "score"), PropertyInfo(Variant::BOOL, "score_changed"), PropertyInfo(Variant::INT, "global_rank_new"), PropertyInfo(Variant::INT, "global_rank_previous")));
	ADD_SIGNAL(MethodInfo("leaderboard_ugc_set", PropertyInfo(Variant::INT, "leaderboard_handle"), PropertyInfo(Variant::STRING, "result")));
	ADD_SIGNAL(MethodInfo("number_of_current_players", PropertyInfo(Variant::INT, "success"), PropertyInfo(Variant::INT, "players")));
	ADD_SIGNAL(MethodInfo("user_achievement_stored", PropertyInfo(Variant::INT, "game_id"), PropertyInfo(Variant::BOOL, "group_achieve"), PropertyInfo(Variant::STRING, "achievement_name"), PropertyInfo(Variant::INT, "current_progress"), PropertyInfo(Variant::INT, "max_progress")));
	ADD_SIGNAL(MethodInfo("current_stats_received", PropertyInfo(Variant::INT, "game_id"), PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "user_id")));
	ADD_SIGNAL(MethodInfo("user_stats_received", PropertyInfo(Variant::INT, "game_id"), PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "user_id")));
	ADD_SIGNAL(MethodInfo("user_stats_stored", PropertyInfo(Variant::INT, "game_id"), PropertyInfo(Variant::INT, "result")));
	ADD_SIGNAL(MethodInfo("user_stats_unloaded", PropertyInfo(Variant::INT, "user_id")));

	// UTILITY SIGNALS //////////////////////////
	ADD_SIGNAL(MethodInfo("check_file_signature", PropertyInfo(Variant::STRING, "signature")));
	ADD_SIGNAL(MethodInfo("gamepad_text_input_dismissed", PropertyInfo(Variant::BOOL, "submitted"), PropertyInfo(Variant::STRING, "entered_text")));
	ADD_SIGNAL(MethodInfo("ip_country"));
	ADD_SIGNAL(MethodInfo("low_power", PropertyInfo(Variant::INT, "power")));
	ADD_SIGNAL(MethodInfo("steam_api_call_completed", PropertyInfo(Variant::INT, "async_call"), PropertyInfo(Variant::INT, "callback"), PropertyInfo(Variant::INT, "parameter")));
	ADD_SIGNAL(MethodInfo("steam_shutdown"));
	ADD_SIGNAL(MethodInfo("app_resuming_from_suspend"));
	ADD_SIGNAL(MethodInfo("floating_gamepad_text_input_dismissed"));

	// VIDEO SIGNALS ////////////////////////////
	ADD_SIGNAL(MethodInfo("get_opf_settings_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "app_id")));
	ADD_SIGNAL(MethodInfo("get_video_result", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "app_id"), PropertyInfo(Variant::STRING, "url")));


	/////////////////////////////////////////////
	// CONSTANT BINDS
	/////////////////////////////////////////////
	//
	// STEAM API CONSTANTS //////////////////////
	BIND_CONSTANT(INVALID_BREAKPAD_HANDLE);												// (BREAKPAD_HANDLE)0
	BIND_CONSTANT(GAME_EXTRA_INFO_MAX); 												// 64
	BIND_CONSTANT(AUTH_TICKET_INVALID);													// 0
	BIND_CONSTANT(API_CALL_INVALID); 													// 0x0
	BIND_CONSTANT(APP_ID_INVALID); 														// 0x0
	BIND_CONSTANT(DEPOT_ID_INVALID); 													// 0x0
	BIND_CONSTANT(STEAM_ACCOUNT_ID_MASK); 												// 0xFFFFFFFF
	BIND_CONSTANT(STEAM_ACCOUNT_INSTANCE_MASK); 										// 0x000FFFFF
	BIND_CONSTANT(STEAM_USER_CONSOLE_INSTANCE); 										// 2
	BIND_CONSTANT(STEAM_USER_DESKTOP_INSTANCE); 										// 1
	BIND_CONSTANT(STEAM_USER_WEB_INSTANCE); 											// 4
	BIND_CONSTANT(QUERY_PORT_ERROR); 													// 0xFFFE
	BIND_CONSTANT(QUERY_PORT_NOT_INITIALIZED); 											// 0xFFFF

	// STEAM SERVER API CONSTANTS ///////////////
	BIND_CONSTANT(FLAG_ACTIVE);															// 0x01
	BIND_CONSTANT(FLAG_DEDICATED);														// 0x04
	BIND_CONSTANT(FLAG_LINUX);															// 0x08
	BIND_CONSTANT(FLAG_NONE);															// 0x00
	BIND_CONSTANT(FLAG_PASSWORDED);														// 0x10
	BIND_CONSTANT(FLAG_PRIVATE);														// 0x20
	BIND_CONSTANT(FLAG_SECURE);															// 0x02

	// FRIENDS CONSTANTS ////////////////////////
	BIND_CONSTANT(CHAT_METADATA_MAX);													// 8192
	BIND_CONSTANT(ENUMERATED_FOLLOWERS_MAX);											// 50
	BIND_CONSTANT(FRIENDS_GROUP_LIMIT);													// 100
	BIND_CONSTANT(INVALID_FRIEND_GROUP_ID);												// -1
	BIND_CONSTANT(MAX_FRIENDS_GROUP_NAME);												// 64
	BIND_CONSTANT(MAX_RICH_PRESENCE_KEY_LENGTH);										// 64
	BIND_CONSTANT(MAX_RICH_PRESENCE_KEYS);												// 20
	BIND_CONSTANT(MAX_RICH_PRESENCE_VALUE_LENTH);										// 256
	BIND_CONSTANT(PERSONA_NAME_MAX_UTF8);												// 128
	BIND_CONSTANT(PERSONA_NAME_MAX_UTF16);												// 32

	// HTML SURFACE CONSTANTS ///////////////////
	BIND_CONSTANT(INVALID_HTMLBROWSER);													// 0

	// HTTP CONSTANTS ///////////////////////////
	BIND_CONSTANT(INVALID_HTTPCOOKIE_HANDLE);											// 0
	BIND_CONSTANT(INVALID_HTTPREQUEST_HANDLE);											// 0

	// INPUT CONSTANTS //////////////////////////
	BIND_CONSTANT(INPUT_MAX_ANALOG_ACTIONS);											// 16
	BIND_CONSTANT(INPUT_MAX_ANALOG_ACTION_DATA);										// 1.0f
	BIND_CONSTANT(INPUT_MAX_COUNT);														// 16
	BIND_CONSTANT(INPUT_MAX_DIGITAL_ACTIONS);											// 128
	BIND_CONSTANT(INPUT_MAX_ORIGINS);													// 8
	BIND_CONSTANT(INPUT_MIN_ANALOG_ACTION_DATA);										// -1.0f

	// INVENTORY CONSTANTS //////////////////////
	BIND_CONSTANT(INVENTORY_RESULT_INVALID);											// -1
	BIND_CONSTANT(ITEM_INSTANCE_ID_INVALID);											// 0

	// MATCHMAKING CONSTANTS ////////////////////
	BIND_CONSTANT(SERVER_QUERY_INVALID);												// 0xffffffff
	BIND_CONSTANT(MAX_LOBBY_KEY_LENGTH);												// 255
	BIND_CONSTANT(FAVORITE_FLAG_FAVORITE);												// 0x01
	BIND_CONSTANT(FAVORITE_FLAG_HISTORY);												// 0x02
	BIND_CONSTANT(FAVORITE_FLAG_NONE);													// 0x00

	// MATCHMAKING SERVERS CONSTANTS ////////////
	BIND_CONSTANT(MAX_GAME_SERVER_GAME_DATA);											// 2048
	BIND_CONSTANT(MAX_GAME_SERVER_GAME_DESCRIPTION);									// 64
	BIND_CONSTANT(MAX_GAME_SERVER_GAME_DIR);											// 32
	BIND_CONSTANT(MAX_GAME_SERVER_MAP_NAME);											// 32
	BIND_CONSTANT(MAX_GAME_SERVER_NAME);												// 64
	BIND_CONSTANT(MAX_GAME_SERVER_TAGS);												// 128

	// MUSIC REMOTE CONSTANTS ///////////////////
	BIND_CONSTANT(MUSIC_NAME_MAX_LENGTH); 												// 255
	BIND_CONSTANT(MUSIC_PNG_MAX_LENGTH); 												// 65535

	// REMOTE PLAY CONSTANTS ////////////////////
	BIND_CONSTANT(DEVICE_FORM_FACTOR_UNKNOWN);											// 0
	BIND_CONSTANT(DEVICE_FORM_FACTOR_PHONE);											// 1
	BIND_CONSTANT(DEVICE_FORM_FACTOR_TABLET);											// 2
	BIND_CONSTANT(DEVICE_FORM_FACTOR_COMPUTER);											// 3
	BIND_CONSTANT(DEVICE_FORM_FACTOR_TV);												// 4

	// REMOTE STORAGE CONSTANTS /////////////////
	BIND_CONSTANT(FILENAME_MAX); 														// 260
	BIND_CONSTANT(PUBLISHED_DOCUMENT_CHANGE_DESCRIPTION_MAX); 							// 8000
	BIND_CONSTANT(PUBLISHED_DOCUMENT_DESCRIPTION_MAX);									// 8000
	BIND_CONSTANT(PUBLISHED_DOCUMENT_TITLE_MAX);										// 128 + 1
	BIND_CONSTANT(PUBLISHED_FILE_URL_MAX);												// 256
	BIND_CONSTANT(TAG_LIST_MAX);														// 1024 + 1
	BIND_CONSTANT(PUBLISHED_FILE_ID_INVALID);											// 0
	BIND_CONSTANT(PUBLISHED_FILE_UPDATE_HANDLE_INVALID);								// 0xffffffffffffffffull
	BIND_CONSTANT(UGC_FILE_STREAM_HANDLE_INVALID);										// 0xffffffffffffffffull
	BIND_CONSTANT(UGC_HANDLE_INVALID);													// 0xffffffffffffffffull
	BIND_CONSTANT(ENUMERATE_PUBLISHED_FILES_MAX_RESULTS);								// 50
	BIND_CONSTANT(MAX_CLOUD_FILE_CHUNK_SIZE);											// 100 * 1024 * 1024

	// SCREENSHOT CONSTANTS /////////////////////
	BIND_CONSTANT(INVALID_SCREENSHOT_HANDLE); 											// 0
	BIND_CONSTANT(UFS_TAG_TYPE_MAX); 													// 255
	BIND_CONSTANT(UFS_TAG_VALUE_MAX); 													// 255
	BIND_CONSTANT(MAX_TAGGED_PUBLISHED_FILES); 											// 32
	BIND_CONSTANT(MAX_TAGGED_USERS); 													// 32
	BIND_CONSTANT(SCREENSHOT_THUMB_WIDTH); 												// 200

	// UGC CONSTANTS ////////////////////////////
	BIND_CONSTANT(NUM_UGC_RESULTS_PER_PAGE); 											// 50
	BIND_CONSTANT(DEVELOPER_METADATA_MAX); 												// 5000
	BIND_CONSTANT(UGC_QUERY_HANDLE_INVALID); 											// 0xffffffffffffffffull
	BIND_CONSTANT(UGC_UPDATE_HANDLE_INVALID); 											// 0xffffffffffffffffull

	// USER STATS CONSTANTS /////////////////////
	BIND_CONSTANT(LEADERBOARD_DETAIL_MAX);												// 64
	BIND_CONSTANT(LEADERBOARD_NAME_MAX);												// 128
	BIND_CONSTANT(STAT_NAME_MAX);														// 128

	/////////////////////////////////////////////
	// ENUM CONSTANT BINDS //////////////////////
	/////////////////////////////////////////////
	//
	// ACCOUNT TYPE /////////////////////////////
	BIND_CONSTANT(ACCOUNT_TYPE_INVALID);												// 0
	BIND_CONSTANT(ACCOUNT_TYPE_INDIVIDUAL);												// 1
	BIND_CONSTANT(ACCOUNT_TYPE_MULTISEAT);												// 2
	BIND_CONSTANT(ACCOUNT_TYPE_GAME_SERVER);											// 3
	BIND_CONSTANT(ACCOUNT_TYPE_ANON_GAME_SERVER);										// 4
	BIND_CONSTANT(ACCOUNT_TYPE_PENDING);												// 5
	BIND_CONSTANT(ACCOUNT_TYPE_CONTENT_SERVER);											// 6
	BIND_CONSTANT(ACCOUNT_TYPE_CLAN);													// 7
	BIND_CONSTANT(ACCOUNT_TYPE_CHAT);													// 8
	BIND_CONSTANT(ACCOUNT_TYPE_CONSOLE_USER);											// 9
	BIND_CONSTANT(ACCOUNT_TYPE_ANON_USER);												// 10
	BIND_CONSTANT(ACCOUNT_TYPE_MAX);													// 11

	// APP OWNERSHIP FLAGS //////////////////////
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_NONE);											// 0x0000
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_OWNS_LICENSE);									// 0x0001
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_FREE_LICENSE);									// 0x0002
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_REGION_RESTRICTED);								// 0x0004
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_LOW_VIOLENCE);									// 0x0008
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_INVALID_PLATFORM);								// 0x0010
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_SHARED_LICENSE);									// 0x0020
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_FREE_WEEKEND);									// 0x0040
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_RETAIL_LICENSE);									// 0x0080
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_LICENSE_LOCKED);									// 0x0100
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_LICENSE_PENDING);									// 0x0200
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_LICENSE_EXPIRED);									// 0x0400
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_LICENSE_PERMANENT);								// 0x0800
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_LICENSE_RECURRING);								// 0x1000
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_LICENSE_CANCELED);								// 0x2000
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_AUTO_GRANT);										// 0x4000
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_PENDING_GIFT);									// 0x8000
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_RENTAL_NOT_ACTIVATED);							// 0x10000
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_RENTAL);											// 0x20000
	BIND_CONSTANT(APP_OWNERSHIP_FLAGS_SITE_LICENSE);									// 0x40000

	// APP RELEASE STATE ////////////////////////
	BIND_CONSTANT(APP_RELEASE_STATE_UNKNOWN);											// 0
	BIND_CONSTANT(APP_RELEASE_STATE_UNAVAILABLE);										// 1
	BIND_CONSTANT(APP_RELEASE_STATE_PRERELEASE);										// 2
	BIND_CONSTANT(APP_RELEASE_STATE_PRELOAD_ONLY);										// 3
	BIND_CONSTANT(APP_RELEASE_STATE_RELEASED);											// 4

	// APP TYPE /////////////////////////////////
	BIND_CONSTANT(APP_TYPE_INVALID);													// 0x000
	BIND_CONSTANT(APP_TYPE_GAME);														// 0x001
	BIND_CONSTANT(APP_TYPE_APPLICATION);												// 0x002
	BIND_CONSTANT(APP_TYPE_TOOL);														// 0x004
	BIND_CONSTANT(APP_TYPE_DEMO);														// 0x008
	BIND_CONSTANT(APP_TYPE_MEDIA_DEPRECATED);											// 0x010
	BIND_CONSTANT(APP_TYPE_DLC);														// 0x020
	BIND_CONSTANT(APP_TYPE_GUIDE);														// 0x040
	BIND_CONSTANT(APP_TYPE_DRIVER);														// 0x080
	BIND_CONSTANT(APP_TYPE_CONFIG);														// 0x100
	BIND_CONSTANT(APP_TYPE_HARDWARE);													// 0x200
	BIND_CONSTANT(APP_TYPE_FRANCHISE);													// 0x400
	BIND_CONSTANT(APP_TYPE_VIDEO);														// 0x800
	BIND_CONSTANT(APP_TYPE_PLUGIN);														// 0x1000
	BIND_CONSTANT(APP_TYPE_MUSIC);														// 0x2000
	BIND_CONSTANT(APP_TYPE_SERIES);														// 0x4000
	BIND_CONSTANT(APP_TYPE_SHORTCUT);													// 0x40000000
	BIND_CONSTANT(APP_TYPE_DEPOT_ONLY);													// 0X80000000

	// AUTH SESSION RESPONSE ////////////////////
	BIND_CONSTANT(AUTH_SESSION_RESPONSE_OK);											// 0
	BIND_CONSTANT(AUTH_SESSION_RESPONSE_USER_NOT_CONNECTED_TO_STEAM);					// 1
	BIND_CONSTANT(AUTH_SESSION_RESPONSE_NO_LICENSE_OR_EXPIRED);							// 2
	BIND_CONSTANT(AUTH_SESSION_RESPONSE_VAC_BANNED);									// 3
	BIND_CONSTANT(AUTH_SESSION_RESPONSE_LOGGED_IN_ELSEWHERE);							// 4
	BIND_CONSTANT(AUTH_SESSION_RESPONSE_VAC_CHECK_TIMEDOUT);							// 5
	BIND_CONSTANT(AUTH_SESSION_RESPONSE_AUTH_TICKET_CANCELED);							// 6
	BIND_CONSTANT(AUTH_SESSION_RESPONSE_AUTH_TICKET_INVALID_ALREADY_USED);				// 7
	BIND_CONSTANT(AUTH_SESSION_RESPONSE_AUTH_TICKET_INVALID);							// 8
	BIND_CONSTANT(AUTH_SESSION_RESPONSE_PUBLISHER_ISSUED_BAN);							// 9

	// BEGIN AUTH SESSION RESULT ////////////////
	BIND_CONSTANT(BEGIN_AUTH_SESSION_RESULT_OK);										// 0
	BIND_CONSTANT(BEGIN_AUTH_SESSION_RESULT_INVALID_TICKET);							// 1
	BIND_CONSTANT(BEGIN_AUTH_SESSION_RESULT_DUPLICATE_REQUEST);							// 2
	BIND_CONSTANT(BEGIN_AUTH_SESSION_RESULT_INVALID_VERSION);							// 3
	BIND_CONSTANT(BEGIN_AUTH_SESSION_RESULT_GAME_MISMATCH);								// 4
	BIND_CONSTANT(BEGIN_AUTH_SESSION_RESULT_EXPIRED_TICKET);							// 5

	// BROADCAST UPLOAD RESULT //////////////////
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_NONE);										// 0
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_OK);											// 1
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_INIT_FAILED);									// 2
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_FRAME_FAILED);								// 3
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_TIME_OUT);									// 4
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_BANDWIDTH_EXCEEDED);							// 5
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_LOW_FPS);										// 6
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_MISSING_KEYFRAMES);							// 7
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_NO_CONNECTION);								// 8
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_RELAY_FAILED);								// 9
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_SETTINGS_CHANGED);							// 10
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_MISSING_AUDIO);								// 11
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_TOO_FAR_BEHIND);								// 12
	BIND_CONSTANT(BROADCAST_UPLOAD_RESULT_TRANSCODE_BEHIND);							// 13

	// CHAT ENTRY TYPE //////////////////////////
	BIND_CONSTANT(CHAT_ENTRY_TYPE_INVALID);												// 0
	BIND_CONSTANT(CHAT_ENTRY_TYPE_CHAT_MSG);											// 1
	BIND_CONSTANT(CHAT_ENTRY_TYPE_TYPING);												// 2
	BIND_CONSTANT(CHAT_ENTRY_TYPE_INVITE_GAME);											// 3
	BIND_CONSTANT(CHAT_ENTRY_TYPE_EMOTE);												// 4
	BIND_CONSTANT(CHAT_ENTRY_TYPE_LEFT_CONVERSATION);									// 6
	BIND_CONSTANT(CHAT_ENTRY_TYPE_ENTERED);												// 7
	BIND_CONSTANT(CHAT_ENTRY_TYPE_WAS_KICKED);											// 8
	BIND_CONSTANT(CHAT_ENTRY_TYPE_WAS_BANNED);											// 9
	BIND_CONSTANT(CHAT_ENTRY_TYPE_DISCONNECTED);										// 10
	BIND_CONSTANT(CHAT_ENTRY_TYPE_HISTORICAL_CHAT);										// 11
	BIND_CONSTANT(CHAT_ENTRY_TYPE_LINK_BLOCKED);										// 14

	// CHAT ROOM ENTER RESPONSE /////////////////
	BIND_CONSTANT(CHAT_ROOM_ENTER_RESPONSE_SUCCESS);									// 1
	BIND_CONSTANT(CHAT_ROOM_ENTER_RESPONSE_DOESNT_EXIST);								// 2
	BIND_CONSTANT(CHAT_ROOM_ENTER_RESPONSE_NOT_ALLOWED);								// 3
	BIND_CONSTANT(CHAT_ROOM_ENTER_RESPONSE_FULL);										// 4
	BIND_CONSTANT(CHAT_ROOM_ENTER_RESPONSE_ERROR);										// 5
	BIND_CONSTANT(CHAT_ROOM_ENTER_RESPONSE_BANNED);										// 6
	BIND_CONSTANT(CHAT_ROOM_ENTER_RESPONSE_LIMITED);									// 7
	BIND_CONSTANT(CHAT_ROOM_ENTER_RESPONSE_CLAN_DISABLED);								// 8
	BIND_CONSTANT(CHAT_ROOM_ENTER_RESPONSE_COMMUNITY_BAN);								// 9
	BIND_CONSTANT(CHAT_ROOM_ENTER_RESPONSE_MEMBER_BLOCKED_YOU);							// 10
	BIND_CONSTANT(CHAT_ROOM_ENTER_RESPONSE_YOU_BLOCKED_MEMBER);							// 11

	// CHAT STEAM ID INSTANCE FLAGS /////////////
	BIND_CONSTANT(CHAT_ACCOUNT_INSTANCE_MASK);											// 0X00000FFF
	BIND_CONSTANT(CHAT_INSTANCE_FLAG_CLAN);												// ((k_unSteamAccountInstanceMask + 1) >> 1)
	BIND_CONSTANT(CHAT_INSTANCE_FLAG_LOBBY);											// ((k_unSteamAccountInstanceMask + 1) >> 2)
	BIND_CONSTANT(CHAT_INSTANCE_FLAG_MMS_LOBBY);										// ((k_unSteamAccountInstanceMask + 1) >> 3)

	// DENY REASON //////////////////////////////
	BIND_CONSTANT(DENY_INVALID);														// 0
	BIND_CONSTANT(DENY_INVALID_VERSION);												// 1
	BIND_CONSTANT(DENY_GENERIC);														// 2
	BIND_CONSTANT(DENY_NOT_LOGGED_ON);													// 3
	BIND_CONSTANT(DENY_NO_LICENSE);														// 4
	BIND_CONSTANT(DENY_CHEATER);														// 5
	BIND_CONSTANT(DENY_LOGGED_IN_ELSEWHERE);											// 6
	BIND_CONSTANT(DENY_UNKNOWN_TEXT);													// 7
	BIND_CONSTANT(DENY_INCOMPATIBLE_ANTI_CHEAT);										// 8
	BIND_CONSTANT(DENY_MEMORY_CORRUPTION);												// 9
	BIND_CONSTANT(DENY_INCOMPATIBLE_SOFTWARE);											// 10
	BIND_CONSTANT(DENY_STEAM_CONNECTION_LOST);											// 11
	BIND_CONSTANT(DENY_STEAM_CONNECTION_ERROR);											// 12
	BIND_CONSTANT(DENY_STEAM_RESPONSE_TIMED_OUT);										// 13
	BIND_CONSTANT(DENY_STEAM_VALIDATION_STALLED);										// 14
	BIND_CONSTANT(DENY_STEAM_OWNER_LEFT_GUEST_USER);									// 15

	// GAME ID TYPE /////////////////////////////
	BIND_CONSTANT(GAME_TYPE_APP);														// 0
	BIND_CONSTANT(GAME_TYPE_GAME_MOD);													// 1
	BIND_CONSTANT(GAME_TYPE_SHORTCUT);													// 2
	BIND_CONSTANT(GAME_TYPE_P2P);														// 3

	// LAUNCH OPTION TYPE ///////////////////////
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_NONE);												// 0
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_DEFAULT);											// 1
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_SAFE_MODE);										// 2
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_MULTIPLAYER);										// 3
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_CONFIG);											// 4
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_OPEN_VR);											// 5
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_SERVER);											// 6
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_EDITOR);											// 7
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_MANUAL);											// 8
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_BENCHMARK);										// 9
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_OPTION1);											// 10
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_OPTION2);											// 11
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_OPTION3);											// 12
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_OCULUS_VR);										// 13
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_OPEN_VR_OVERLAY);									// 14
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_OS_VR);											// 15
	BIND_CONSTANT(LAUNCH_OPTION_TYPE_DIALOG);											// 1000

	// MARKETING MESSAGE FLAGS //////////////////
	BIND_CONSTANT(MARKETING_MESSAGE_FLAGS_NONE);										// 0
	BIND_CONSTANT(MARKETING_MESSAGE_FLAGS_HIGH_PRIORITY);								// (1<<0)
	BIND_CONSTANT(MARKETING_MESSAGE_FLAGS_PLATFORM_WINDOWS);							// (1<<1)
	BIND_CONSTANT(MARKETING_MESSAGE_FLAGS_PLATFORM_MAC);								// (1<<2)
	BIND_CONSTANT(MARKETING_MESSAGE_FLAGS_PLATFORM_LINUX);								// (1<<3)
	BIND_CONSTANT(MARKETING_MESSAGE_FLAGS_PLATFORM_RESTRICTIONS);

	// NOTIFICATION POSITION ////////////////////
	BIND_CONSTANT(POSITION_TOP_LEFT);													// 0
	BIND_CONSTANT(POSITION_TOP_RIGHT);													// 1
	BIND_CONSTANT(POSITION_BOTTOM_LEFT);												// 2
	BIND_CONSTANT(POSITION_BOTTOM_RIGHT);												// 3

	// RESULT ///////////////////////////////////
	BIND_CONSTANT(RESULT_OK);															// 1
	BIND_CONSTANT(RESULT_FAIL);															// 2
	BIND_CONSTANT(RESULT_NO_CONNECTION);												// 3
	BIND_CONSTANT(RESULT_INVALID_PASSWORD);												// 5
	BIND_CONSTANT(RESULT_LOGGED_IN_ELSEWHERE);											// 6
	BIND_CONSTANT(RESULT_INVALID_PROTOCOL_VER);											// 7
	BIND_CONSTANT(RESULT_INVALID_PARAM);												// 8
	BIND_CONSTANT(RESULT_FILE_NOT_FOUND);												// 9
	BIND_CONSTANT(RESULT_BUSY);															// 10
	BIND_CONSTANT(RESULT_INVALID_STATE);												// 11
	BIND_CONSTANT(RESULT_INVALID_NAME);													// 12
	BIND_CONSTANT(RESULT_INVALID_EMAIL);												// 13
	BIND_CONSTANT(RESULT_DUPLICATE_NAME);												// 14
	BIND_CONSTANT(RESULT_ACCESS_DENIED);												// 15
	BIND_CONSTANT(RESULT_TIMEOUT);														// 16
	BIND_CONSTANT(RESULT_BANNED);														// 17
	BIND_CONSTANT(RESULT_ACCOUNT_NOT_FOUND);											// 18
	BIND_CONSTANT(RESULT_INVALID_STEAM_ID);												// 19
	BIND_CONSTANT(RESULT_SERVICE_UNAVAILABLE);											// 20
	BIND_CONSTANT(RESULT_NOT_LOGGED_ON);												// 21
	BIND_CONSTANT(RESULT_PENDING);														// 22
	BIND_CONSTANT(RESULT_ENCRYPTION_FAILURE);											// 23
	BIND_CONSTANT(RESULT_INSUFFICIENT_PRIVILEGE);										// 24
	BIND_CONSTANT(RESULT_LIMIT_EXCEEDED);												// 25
	BIND_CONSTANT(RESULT_REVOKED);														// 26
	BIND_CONSTANT(RESULT_EXPIRED);														// 27
	BIND_CONSTANT(RESULT_ALREADY_REDEEMED);												// 28
	BIND_CONSTANT(RESULT_DUPLICATE_REQUEST);											// 29
	BIND_CONSTANT(RESULT_ALREADY_OWNED);												// 30
	BIND_CONSTANT(RESULT_IP_NOT_FOUND);													// 31
	BIND_CONSTANT(RESULT_PERSIST_FAILED);												// 32
	BIND_CONSTANT(RESULT_LOCKING_FAILED);												// 33
	BIND_CONSTANT(RESULT_LOG_ON_SESSION_REPLACED);										// 34
	BIND_CONSTANT(RESULT_CONNECT_FAILED);												// 35
	BIND_CONSTANT(RESULT_HANDSHAKE_FAILED);												// 36
	BIND_CONSTANT(RESULT_IO_FAILURE);													// 37
	BIND_CONSTANT(RESULT_REMOTE_DISCONNECT);											// 38
	BIND_CONSTANT(RESULT_SHOPPING_CART_NOT_FOUND);										// 39
	BIND_CONSTANT(RESULT_BLOCKED);														// 40
	BIND_CONSTANT(RESULT_IGNORED);														// 41
	BIND_CONSTANT(RESULT_NO_MATCH);														// 42
	BIND_CONSTANT(RESULT_ACCOUNT_DISABLED);												// 43
	BIND_CONSTANT(RESULT_SERVICE_READ_ONLY);											// 44
	BIND_CONSTANT(RESULT_ACCOUNT_NOT_FEATURED);											// 45
	BIND_CONSTANT(RESULT_ADMINISTRATOR_OK);												// 46
	BIND_CONSTANT(RESULT_CONTENT_VERSION);												// 47
	BIND_CONSTANT(RESULT_TRY_ANOTHER_CM);												// 48
	BIND_CONSTANT(RESULT_PASSWORD_REQUIRED_TO_KICK_SESSION);							// 49
	BIND_CONSTANT(RESULT_ALREADY_LOGGED_IN_ELSEWHERE);									// 50
	BIND_CONSTANT(RESULT_SUSPENDED);													// 51
	BIND_CONSTANT(RESULT_CANCELLED);													// 52
	BIND_CONSTANT(RESULT_DATA_CORRUPTION);												// 53
	BIND_CONSTANT(RESULT_DISK_FULL);													// 54
	BIND_CONSTANT(RESULT_REMOTE_CALL_FAILED);											// 55
	BIND_CONSTANT(RESULT_PASSWORD_UNSET);												// 56
	BIND_CONSTANT(RESULT_EXTERNAL_ACCOUNT_UNLINKED);									// 57
	BIND_CONSTANT(RESULT_PSN_TICKET_INVALID);											// 58
	BIND_CONSTANT(RESULT_EXTERNAL_ACCOUNT_ALREADY_LINKED);								// 59
	BIND_CONSTANT(RESULT_REMOTE_FILE_CONFLICT);											// 60
	BIND_CONSTANT(RESULT_ILLEGAL_PASSWORD);												// 61
	BIND_CONSTANT(RESULT_SAME_AS_PREVIOUS_VALUE);										// 62
	BIND_CONSTANT(RESULT_ACCOUNT_LOG_ON_DENIED);										// 63
	BIND_CONSTANT(RESULT_CANNOT_USE_OLD_PASSWORD);										// 64
	BIND_CONSTANT(RESULT_INVALID_LOGIN_AUTH_CODE);										// 65
	BIND_CONSTANT(RESULT_ACCOUNT_LOG_ON_DENIED_NO_MAIL);								// 66
	BIND_CONSTANT(RESULT_HARDWARE_NOT_CAPABLE_OF_IPT);									// 67
	BIND_CONSTANT(RESULT_IPT_INIT_ERROR);												// 68
	BIND_CONSTANT(RESULT_PARENTAL_CONTROL_RESTRICTED);									// 69
	BIND_CONSTANT(RESULT_FACEBOOK_QUERY_ERROR);											// 70
	BIND_CONSTANT(RESULT_EXPIRED_LOGIN_AUTH_CODE);										// 71
	BIND_CONSTANT(RESULT_IP_LOGIN_RESTRICTION_FAILED);									// 72
	BIND_CONSTANT(RESULT_ACCOUNT_LOCKED_DOWN);											// 73
	BIND_CONSTANT(RESULT_ACCOUNT_LOG_ON_DENIED_VERIFIED_EMAIL_REQUIRED);				// 74
	BIND_CONSTANT(RESULT_NO_MATCHING_URL);												// 75
	BIND_CONSTANT(RESULT_BAD_RESPONSE);													// 76
	BIND_CONSTANT(RESULT_REQUIRE_PASSWORD_REENTRY);										// 77
	BIND_CONSTANT(RESULT_VALUE_OUT_OF_RANGE);											// 78
	BIND_CONSTANT(RESULT_UNEXPECTED_ERROR);												// 79
	BIND_CONSTANT(RESULT_DISABLED);														// 80
	BIND_CONSTANT(RESULT_INVALID_CEG_SUBMISSION);										// 81
	BIND_CONSTANT(RESULT_RESTRICTED_DEVICE);											// 82
	BIND_CONSTANT(RESULT_REGION_LOCKED);												// 83
	BIND_CONSTANT(RESULT_RATE_LIMIT_EXCEEDED);											// 84
	BIND_CONSTANT(RESULT_ACCOUNT_LOGIN_DENIED_NEED_TWO_FACTOR);							// 85
	BIND_CONSTANT(RESULT_ITEM_DELETED);													// 86
	BIND_CONSTANT(RESULT_ACCOUNT_LOGIN_DENIED_THROTTLE);								// 87
	BIND_CONSTANT(RESULT_TWO_FACTOR_CODE_MISMATCH);										// 88
	BIND_CONSTANT(RESULT_TWO_FACTOR_ACTIVATION_CODE_MISMATCH);							// 89
	BIND_CONSTANT(RESULT_ACCOUNT_ASSOCIATED_TO_MULTIPLE_PARTNERS);						// 90
	BIND_CONSTANT(RESULT_NOT_MODIFIED);													// 91
	BIND_CONSTANT(RESULT_NO_MOBILE_DEVICE);												// 92
	BIND_CONSTANT(RESULT_TIME_NOT_SYNCED);												// 93
	BIND_CONSTANT(RESULT_SMS_CODE_FAILED);												// 94
	BIND_CONSTANT(RESULT_ACCOUNT_LIMIT_EXCEEDED);										// 95
	BIND_CONSTANT(RESULT_ACCOUNT_ACTIVITY_LIMIT_EXCEEDED);								// 96
	BIND_CONSTANT(RESULT_PHONE_ACTIVITY_LIMIT_EXCEEDED);								// 97
	BIND_CONSTANT(RESULT_REFUND_TO_WALLET);												// 98
	BIND_CONSTANT(RESULT_EMAIL_SEND_FAILURE);											// 99
	BIND_CONSTANT(RESULT_NOT_SETTLED);													// 100
	BIND_CONSTANT(RESULT_NEED_CAPTCHA);													// 101
	BIND_CONSTANT(RESULT_GSLT_DENIED);													// 102
	BIND_CONSTANT(RESULT_GS_OWNER_DENIED);												// 103
	BIND_CONSTANT(RESULT_INVALID_ITEM_TYPE);											// 104
	BIND_CONSTANT(RESULT_IP_BANNED);													// 105
	BIND_CONSTANT(RESULT_GSLT_EXPIRED);													// 106
	BIND_CONSTANT(RESULT_INSUFFICIENT_FUNDS);											// 107
	BIND_CONSTANT(RESULT_TOO_MANY_PENDING);												// 108

	// UNIVERSE /////////////////////////////////
	BIND_CONSTANT(UNIVERSE_INVALID);													// 0
	BIND_CONSTANT(UNIVERSE_PUBLIC);														// 1
	BIND_CONSTANT(UNIVERSE_BETA);														// 2
	BIND_CONSTANT(UNIVERSE_INTERNAL);													// 3
	BIND_CONSTANT(UNIVERSE_DEV);														// 4
	BIND_CONSTANT(UNIVERSE_MAX);														// 5

	// USER HAS LICENSE FOR APP RESULT //////////
	BIND_CONSTANT(USER_HAS_LICENSE_RESULT_HAS_LICENSE);									// 0
	BIND_CONSTANT(USER_HAS_LICENSE_RESULT_DOES_NOT_HAVE_LICENSE);						// 1
	BIND_CONSTANT(USER_HAS_LICENSE_RESULT_NO_AUTH);										// 2

	// VOICE RESULT /////////////////////////////
	BIND_CONSTANT(VOICE_RESULT_OK);														// 0
	BIND_CONSTANT(VOICE_RESULT_NOT_INITIALIZED);										// 1
	BIND_CONSTANT(VOICE_RESULT_NOT_RECORDING);											// 2
	BIND_CONSTANT(VOICE_RESULT_NO_DATE);												// 3
	BIND_CONSTANT(VOICE_RESULT_BUFFER_TOO_SMALL);										// 4
	BIND_CONSTANT(VOICE_RESULT_DATA_CORRUPTED);											// 5
	BIND_CONSTANT(VOICE_RESULT_RESTRICTED);												// 6

	// VR HMD TYPE
	BIND_CONSTANT(VR_HMD_TYPE_NONE);													// -1
	BIND_CONSTANT(VR_HMD_TYPE_UNKNOWN);													// 0
	BIND_CONSTANT(VR_HMD_TYPE_HTC_DEV);													// 1
	BIND_CONSTANT(VR_HMD_TYPE_HTC_VIVEPRE);												// 2
	BIND_CONSTANT(VR_HMD_TYPE_HTC_VIVE);												// 3
	BIND_CONSTANT(VR_HMD_TYPE_HTC_UNKNOWN);												// 20
	BIND_CONSTANT(VR_HMD_TYPE_OCULUS_DK1);												// 21
	BIND_CONSTANT(VR_HMD_TYPE_OCULUS_DK2);												// 22
	BIND_CONSTANT(VR_HMD_TYPE_OCULUS_RIFT);												// 23
	BIND_CONSTANT(VR_HMD_TYPE_OCULUS_UNKNOWN);											// 40

	// REGISTER ACTIVATION CODE RESULTS//////////
	BIND_CONSTANT(ACTIVATION_CODE_RESULT_OK);											// 0
	BIND_CONSTANT(ACTIVATION_CODE_RESULT_FAIL);											// 1
	BIND_CONSTANT(ACTIVATION_CODE_RESULT_ALREADY_REGISTERED);							// 2
	BIND_CONSTANT(ACTIVATION_CODE_RESULT_TIMEOUT);										// 3
	BIND_CONSTANT(ACTIVATION_CODE_RESULT_ALREADY_OWNED);								// 4

	// AVATAR ///////////////////////////////////
	BIND_CONSTANT(AVATAR_SMALL);														// 1
	BIND_CONSTANT(AVATAR_MEDIUM);														// 2
	BIND_CONSTANT(AVATAR_LARGE);														// 3

	// FRIEND FLAGS ENUM ////////////////////////
	BIND_CONSTANT(FRIEND_FLAG_NONE);													// 0X00
	BIND_CONSTANT(FRIEND_FLAG_BLOCKED);													// 0X01
	BIND_CONSTANT(FRIEND_FLAG_FRIENDSHIP_REQUESTED);									// 0X02
	BIND_CONSTANT(FRIEND_FLAG_IMMEDIATE);												// 0X04
	BIND_CONSTANT(FRIEND_FLAG_CLAN_MEMBER);												// 0X08
	BIND_CONSTANT(FRIEND_FLAG_ON_GAME_SERVER);											// 0X10
	BIND_CONSTANT(FRIEND_FLAG_REQUESTING_FRIENDSHIP);									// 0X80
	BIND_CONSTANT(FRIEND_FLAG_REQUESTING_INFO);											// 0X100
	BIND_CONSTANT(FRIEND_FLAG_IGNORED);													// 0X200
	BIND_CONSTANT(FRIEND_FLAG_IGNORED_FRIEND);											// 0X400
	BIND_CONSTANT(FRIEND_FLAG_CHAT_MEMBER);												// 0X1000
	BIND_CONSTANT(FRIEND_FLAG_ALL);														// 0XFFFF

	// FRIEND RELATIONSHIP //////////////////////
	BIND_CONSTANT(FRIEND_RELATION_NONE);												// 0
	BIND_CONSTANT(FRIEND_RELATION_BLOCKED);												// 1
	BIND_CONSTANT(FRIEND_RELATION_REQUEST_RECIPIENT);									// 2
	BIND_CONSTANT(FRIEND_RELATION_FRIEND);												// 3
	BIND_CONSTANT(FRIEND_RELATION_REQUEST_INITIATOR);									// 4
	BIND_CONSTANT(FRIEND_RELATION_IGNORED);												// 5
	BIND_CONSTANT(FRIEND_RELATION_IGNORED_FRIEND);										// 6
	BIND_CONSTANT(FRIEND_RELATION_SUGGESTED);											// 7
	BIND_CONSTANT(FRIEND_RELATION_MAX);													// 8

	// OVERLAY TO STORE FLAG ////////////////////
	BIND_CONSTANT(OVERLAY_TO_STORE_FLAG_NONE);											// 0
	BIND_CONSTANT(OVERLAY_TO_STORE_FLAG_ADD_TO_CART);									// 1
	BIND_CONSTANT(OVERLAY_TO_STORE_FLAG_AND_TO_CART_AND_SHOW);							// 2

	// OVERLAY TO WEB PAGE MODE /////////////////
	BIND_CONSTANT(OVERLAY_TO_WEB_PAGE_MODE_DEFAULT);									// 0
	BIND_CONSTANT(OVERLAY_TO_WEB_PAGE_MODE_MODAL);										// 1

	// PERSONA CHANGE ///////////////////////////
	BIND_CONSTANT(PERSONA_CHANGE_NAME);													// 0X0001
	BIND_CONSTANT(PERSONA_CHANGE_STATUS);												// 0X0002
	BIND_CONSTANT(PERSONA_CHANGE_COME_ONLINE);											// 0X0004
	BIND_CONSTANT(PERSONA_CHANGE_GONE_OFFLINE);											// 0X0008
	BIND_CONSTANT(PERSONA_CHANGE_GAME_PLAYED);											// 0X0010
	BIND_CONSTANT(PERSONA_CHANGE_GAME_SERVER);											// 0X0020
	BIND_CONSTANT(PERSONA_CHANGE_AVATAR);												// 0X0040
	BIND_CONSTANT(PERSONA_CHANGE_JOINED_SOURCE);										// 0X0080
	BIND_CONSTANT(PERSONA_CHANGE_LEFT_SOURCE);											// 0X0100
	BIND_CONSTANT(PERSONA_CHANGE_RELATIONSHIP_CHANGED);									// 0X0200
	BIND_CONSTANT(PERSONA_CHANGE_NAME_FIRST_SET);										// 0X0400
	BIND_CONSTANT(PERSONA_CHANGE_FACEBOOK_INFO);										// 0X0800
	BIND_CONSTANT(PERSONA_CHANGE_NICKNAME);												// 0X1000
	BIND_CONSTANT(PERSONA_CHANGE_STEAM_LEVEL);											// 0X2000

	// PERSONA STATE ////////////////////////////
	BIND_CONSTANT(PERSONA_STATE_OFFLINE);												// 0
	BIND_CONSTANT(PERSONA_STATE_ONLINE);												// 1
	BIND_CONSTANT(PERSONA_STATE_BUSY);													// 2
	BIND_CONSTANT(PERSONA_STATE_AWAY);													// 3
	BIND_CONSTANT(PERSONA_STATE_SNOOZE);												// 4
	BIND_CONSTANT(PERSONA_STATE_LOOKING_TO_TRADE);										// 5
	BIND_CONSTANT(PERSONA_STATE_LOOKING_TO_PLAY);										// 6
	BIND_CONSTANT(PERSONA_STATE_MAX);													// 7

	// USER RESTRICTION /////////////////////////
	BIND_CONSTANT(USER_RESTRICTION_NONE);												// 0
	BIND_CONSTANT(USER_RESTRICTION_UNKNOWN);											// 1
	BIND_CONSTANT(USER_RESTRICTION_ANY_CHAT);											// 2
	BIND_CONSTANT(USER_RESTRICTION_VOICE_CHAT);											// 4
	BIND_CONSTANT(USER_RESTRICTION_GROUP_CHAT);											// 8
	BIND_CONSTANT(USER_RESTRICTION_RATING);												// 16
	BIND_CONSTANT(USER_RESTRICTION_GAME_INVITES);										// 32
	BIND_CONSTANT(USER_RESTRICTION_TRADING);											// 64

	// GAME SEARCH ERROR CODE ///////////////////
	BIND_CONSTANT(GAME_SEARCH_ERROR_CODE_OK);											// 1
	BIND_CONSTANT(GAME_SEARCH_ERROR_CODE_SEARCH_AREADY_IN_PROGRESS);					// 2
	BIND_CONSTANT(GAME_SEARCH_ERROR_CODE_NO_SEARCH_IN_PROGRESS);						// 3
	BIND_CONSTANT(GAME_SEARCH_ERROR_CODE_NOT_LOBBY_LEADER);								// 4
	BIND_CONSTANT(GAME_SEARCH_ERROR_CODE_NO_HOST_AVAILABLE);							// 5
	BIND_CONSTANT(GAME_SEARCH_ERROR_CODE_SEARCH_PARAMS_INVALID);						// 6
	BIND_CONSTANT(GAME_SEARCH_ERROR_CODE_OFFLINE);										// 7
	BIND_CONSTANT(GAME_SEARCH_ERROR_CODE_NOT_AUTHORIZED);								// 8
	BIND_CONSTANT(GAME_SEARCH_ERROR_CODE_UNKNOWN_ERROR);								// 9

	// GAME SEARCH PLAYER RESULT ////////////////
	BIND_CONSTANT(PLAYER_RESULT_FAILED_TO_CONNECT);										// 1
	BIND_CONSTANT(PLAYER_RESULT_ABANDONED);												// 2
	BIND_CONSTANT(PLAYER_RESULT_KICKED);												// 3
	BIND_CONSTANT(PLAYER_RESULT_INCOMPLETE);											// 4
	BIND_CONSTANT(PLAYER_RESULT_COMPLETED);	

	// HTML KEY MODIFIERS ///////////////////////
	BIND_CONSTANT(HTML_KEY_MODIFIER_NONE);												// 0
	BIND_CONSTANT(HTML_KEY_MODIFIER_ALT_DOWN);											// (1<<0)
	BIND_CONSTANT(HTML_KEY_MODIFIER_CTRL_DOWN);											// (1<<1)
	BIND_CONSTANT(HTML_KEY_MODIFIER_SHIFT_DOWN);										// (1<<2)

	// HTML MOUSE BUTTON ////////////////////////
	BIND_CONSTANT(HTML_MOUSE_BUTTON_LEFT);												// 0
	BIND_CONSTANT(HTML_MOUSE_BUTTON_RIGHT);												// 1
	BIND_CONSTANT(HTML_MOUSE_BUTTON_MIDDLE);											// 2

	// MOUSE CURSOR /////////////////////////////
	BIND_CONSTANT(DC_USER);																// 0
	BIND_CONSTANT(DC_NONE);																// 1
	BIND_CONSTANT(DC_ARROW);															// 2
	BIND_CONSTANT(DC_IBEAM);															// 3
	BIND_CONSTANT(DC_HOUR_GLASS);														// 4
	BIND_CONSTANT(DC_WAIT_ARROW);														// 5
	BIND_CONSTANT(DC_CROSSHAIR);														// 6
	BIND_CONSTANT(DC_UP);																// 7
	BIND_CONSTANT(DC_SIZE_NW);															// 8
	BIND_CONSTANT(DC_SIZE_SE);															// 9
	BIND_CONSTANT(DC_SIZE_NE);															// 10
	BIND_CONSTANT(DC_SIZE_SW);															// 11
	BIND_CONSTANT(DC_SIZE_W);															// 12
	BIND_CONSTANT(DC_SIZE_E);															// 13
	BIND_CONSTANT(DC_SIZE_N);															// 14
	BIND_CONSTANT(DC_SIZE_S);															// 15
	BIND_CONSTANT(DC_SIZE_WE);															// 16
	BIND_CONSTANT(DC_SIZE_NS);															// 17
	BIND_CONSTANT(DC_SIZE_ALL);															// 18
	BIND_CONSTANT(DC_NO);																// 19
	BIND_CONSTANT(DC_HAND);																// 20
	BIND_CONSTANT(DC_BLANK);															// 21
	BIND_CONSTANT(DC_MIDDLE_PAN);														// 22
	BIND_CONSTANT(DC_NORTH_PAN);														// 23
	BIND_CONSTANT(DC_NORTH_EAST_PAN);													// 24
	BIND_CONSTANT(DC_EAST_PAN);															// 25
	BIND_CONSTANT(DC_SOUTH_EAST_PAN);													// 26
	BIND_CONSTANT(DC_SOUTH_PAN);														// 27
	BIND_CONSTANT(DC_SOUTH_WEST_PAN);													// 28
	BIND_CONSTANT(DC_WEST_PAN);															// 29
	BIND_CONSTANT(DC_NORTH_WEST_PAN);													// 30
	BIND_CONSTANT(DC_ALIAS);															// 31
	BIND_CONSTANT(DC_CELL);																// 32
	BIND_CONSTANT(DC_COL_RESIZE);														// 33
	BIND_CONSTANT(DC_COPY_CUR);															// 34
	BIND_CONSTANT(DC_VERTICAL_TEXT);													// 35
	BIND_CONSTANT(DC_ROW_RESIZE);														// 36
	BIND_CONSTANT(DC_ZOOM_IN);															// 37
	BIND_CONSTANT(DC_ZOOM_OUT);															// 38
	BIND_CONSTANT(DC_HELP);																// 39
	BIND_CONSTANT(DC_CUSTOM);															// 40
	BIND_CONSTANT(DC_LAST);																// 41

	// HTTP METHOD //////////////////////////////
	BIND_CONSTANT(HTTP_METHOD_INVALID);													// 0
	BIND_CONSTANT(HTTP_METHOD_GET);														// 1
	BIND_CONSTANT(HTTP_METHOD_HEAD);													// 2
	BIND_CONSTANT(HTTP_METHOD_POST);													// 3
	BIND_CONSTANT(HTTP_METHOD_PUT);														// 4
	BIND_CONSTANT(HTTP_METHOD_DELETE);													// 5
	BIND_CONSTANT(HTTP_METHOD_OPTIONS);													// 6
	BIND_CONSTANT(HTTP_METHOD_PATCH);													// 7

	// HTTP STATUS CODE /////////////////////////
	BIND_CONSTANT(HTTP_STATUS_CODE_INVALID);											// 0
	BIND_CONSTANT(HTTP_STATUS_CODE_100_CONTINUE);										// 100
	BIND_CONSTANT(HTTP_STATUS_CODE_101_SWITCHING_PROTOCOLS);							// 101
	BIND_CONSTANT(HTTP_STATUS_CODE_200_OK);												// 200
	BIND_CONSTANT(HTTP_STATUS_CODE_201_CREATED);										// 201
	BIND_CONSTANT(HTTP_STATUS_CODE_202_ACCEPTED);										// 202
	BIND_CONSTANT(HTTP_STATUS_CODE_203_NON_AUTHORITATIVE);								// 203
	BIND_CONSTANT(HTTP_STATUS_CODE_204_NO_CONTENT);										// 204
	BIND_CONSTANT(HTTP_STATUS_CODE_205_RESET_CONTENT);									// 205
	BIND_CONSTANT(HTTP_STATUS_CODE_206_PARTIAL_CONTENT);								// 206
	BIND_CONSTANT(HTTP_STATUS_CODE_300_MULTIPLE_CHOICES);								// 300
	BIND_CONSTANT(HTTP_STATUS_CODE_301_MOVED_PERMANENTLY);								// 301
	BIND_CONSTANT(HTTP_STATUS_CODE_302_FOUND);											// 302
	BIND_CONSTANT(HTTP_STATUS_CODE_303_SEE_OTHER);										// 303
	BIND_CONSTANT(HTTP_STATUS_CODE_304_NOT_MODIFIED);									// 304
	BIND_CONSTANT(HTTP_STATUS_CODE_305_USE_PROXY);										// 305
	BIND_CONSTANT(HTTP_STATUS_CODE_307_TEMPORARY_REDIRECT);								// 307
	BIND_CONSTANT(HTTP_STATUS_CODE_400_BAD_REQUEST);									// 400
	BIND_CONSTANT(HTTP_STATUS_CODE_401_UNAUTHORIZED);									// 401
	BIND_CONSTANT(HTTP_STATUS_CODE_402_PAYMENT_REQUIRED);								// 402
	BIND_CONSTANT(HTTP_STATUS_CODE_403_FORBIDDEN);										// 403
	BIND_CONSTANT(HTTP_STATUS_CODE_404_NOT_FOUND);										// 404
	BIND_CONSTANT(HTTP_STATUS_CODE_405_METHOD_NOT_ALLOWED);								// 405
	BIND_CONSTANT(HTTP_STATUS_CODE_406_NOT_ACCEPTABLE);									// 406
	BIND_CONSTANT(HTTP_STATUS_CODE_407_PROXY_AUTH_REQUIRED);							// 407
	BIND_CONSTANT(HTTP_STATUS_CODE_408_REQUEST_TIMEOUT);								// 408
	BIND_CONSTANT(HTTP_STATUS_CODE_409_CONFLICT);										// 409
	BIND_CONSTANT(HTTP_STATUS_CODE_410_GONE);											// 410
	BIND_CONSTANT(HTTP_STATUS_CODE_411_LENGTH_REQUIRED);								// 411
	BIND_CONSTANT(HTTP_STATUS_CODE_412_PRECONDITION_FAILED);							// 412
	BIND_CONSTANT(HTTP_STATUS_CODE_413_REQUEST_ENTITY_TOO_LARGE);						// 413
	BIND_CONSTANT(HTTP_STATUS_CODE_414_REQUEST_URI_TOO_LONG);							// 414
	BIND_CONSTANT(HTTP_STATUS_CODE_415_UNSUPPORTED_MEDIA_TYPE);							// 415
	BIND_CONSTANT(HTTP_STATUS_CODE_416_REQUESTED_RANGE_NOT_SATISFIABLE);				// 416
	BIND_CONSTANT(HTTP_STATUS_CODE_417_EXPECTATION_FAILED);								// 417
	BIND_CONSTANT(HTTP_STATUS_CODE_4XX_UNKNOWN);										// 418
	BIND_CONSTANT(HTTP_STATUS_CODE_429_TOO_MANY_REQUESTS);								// 429
	BIND_CONSTANT(HTTP_STATUS_CODE_500_INTERNAL_SERVER_ERROR);							// 500
	BIND_CONSTANT(HTTP_STATUS_CODE_501_NOT_IMPLEMENTED);								// 501
	BIND_CONSTANT(HTTP_STATUS_CODE_502_BAD_GATEWAY);									// 502
	BIND_CONSTANT(HTTP_STATUS_CODE_503_SERVICE_UNAVAILABLE);							// 503
	BIND_CONSTANT(HTTP_STATUS_CODE_504_GATEWAY_TIMEOUT);								// 504
	BIND_CONSTANT(HTTP_STATUS_CODE_505_HTTP_VERSION_NOT_SUPPORTED);						// 505
	BIND_CONSTANT(HTTP_STATUS_CODE_5XX_UNKNOWN);										// 599

	// INPUT ACTION ORIGIN //////////////////////
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_NONE);											// 0
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_A);												// 1
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_B);												// 2
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_X);												// 3
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_Y);												// 4
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_BUMPER);										// 5
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_RIGHT_BUMPER);									// 6
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFTGRIP);										// 7
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_RIGHTGRIP);										// 8
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_START);											// 9
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_BACK);											// 10
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_PAD_TOUCH);									// 11
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_PAD_SWIPE);									// 12
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_PAD_CLICK);									// 13
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_PAD_DPAD_NORTH);								// 14
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_PAD_DPAD_SOUTH);								// 15
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_PAD_DPAD_WEST);								// 16
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_PAD_DPAD_EAST);								// 17
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_RIGHT_PAD_TOUCH);									// 18
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_RIGHT_PAD_SWIPE);									// 19
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_RIGHT_PAD_CLICK);									// 20
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_RIGHT_PAD_DPAD_NORTH);							// 21
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_RIGHT_PAD_DPAD_SOUTH);							// 22
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_RIGHT_PAD_DPAD_WEST);								// 23
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_RIGHT_PAD_DPAD_EAST);								// 24
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_TRIGGER_PULL);								// 25
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_TRIGGER_CLICK);								// 26
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_RIGHT_TRIGGER_PULL);								// 27
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_RIGHT_TRIGGER_CLICK);								// 28
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_STICK_MOVE);									// 29
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_STICK_CLICK);								// 30
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_STICK_DPAD_NORTH);							// 31
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_STICK_DPAD_SOUTH);							// 32
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_STICK_DPAD_WEST);							// 33
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_LEFT_STICK_DPAD_EAST);							// 34
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_GYRO_MOVE);										// 35
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_GYRO_PITCH);										// 36
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_GYRO_YAW);										// 37
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_GYRO_ROLL);										// 38
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAM_CONTROLLER_RESERVED0);						// 39
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAM_CONTROLLER_RESERVED1);						// 40
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAM_CONTROLLER_RESERVED2);						// 41
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAM_CONTROLLER_RESERVED3);						// 42
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAM_CONTROLLER_RESERVED4);						// 43
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAM_CONTROLLER_RESERVED5);						// 44
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAM_CONTROLLER_RESERVED6);						// 45
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAM_CONTROLLER_RESERVED7);						// 46
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAM_CONTROLLER_RESERVED8);						// 47
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAM_CONTROLLER_RESERVED9);						// 48
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAM_CONTROLLER_RESERVED10);						// 49
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_X);											// 50
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_CIRCLE);										// 51
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_TRIANGLE);									// 52
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_SQUARE);										// 53
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_BUMPER);									// 54
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_BUMPER);								// 55
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_OPTIONS);										// 56
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_SHARE);										// 57
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_PAD_TOUCH);								// 58
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_PAD_SWIPE);								// 59
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_PAD_CLICK);								// 60
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_PAD_DPAD_NORTH);							// 61
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_PAD_DPAD_SOUTH);							// 62
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_PAD_DPAD_WEST);							// 63
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_PAD_DPAD_EAST);							// 64
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_PAD_TOUCH);								// 65
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_PAD_SWIPE);								// 66
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_PAD_CLICK);								// 67
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_PAD_DPAD_NORTH);						// 68
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_PAD_DPAD_SOUTH);						// 69
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_PAD_DPAD_WEST);							// 70
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_PAD_DPAD_EAST);							// 71
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_CENTER_PAD_TOUCH);							// 72
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_CENTER_PAD_SWIPE);							// 73
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_CENTER_PAD_CLICK);							// 74
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_CENTER_PAD_DPAD_NORTH);						// 75
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_CENTER_PAD_DPAD_SOUTH);						// 76
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_CENTER_PAD_DPAD_WEST);						// 77
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_CENTER_PAD_DPAD_EAST);						// 78
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_TRIGGER_PULL);							// 79
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_TRIGGER_CLICK);							// 80
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_TRIGGER_PULL);							// 81
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_TRIGGER_CLICK);							// 82
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_STICK_MOVE);								// 83
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_STICK_CLICK);							// 84
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_STICK_DPAD_NORTH);						// 85
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_STICK_DPAD_SOUTH);						// 86
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_STICK_DPAD_WEST);						// 87
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_LEFT_STICK_DPAD_EAST);						// 88
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_STICK_MOVE);							// 89
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_STICK_CLICK);							// 90
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_STICK_DPAD_NORTH);						// 91
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_STICK_DPAD_SOUTH);						// 92
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_STICK_DPAD_WEST);						// 93
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RIGHT_STICK_DPAD_EAST);						// 94
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_DPAD_NORTH);									// 95
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_DPAD_SOUTH);									// 96
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_DPAD_WEST);									// 97
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_DPAD_EAST);									// 98
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_GYRO_MOVE);									// 99
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_GYRO_PITCH);									// 100
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_GYRO_YAW);									// 101
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_GYRO_ROLL);									// 102
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RESERVED0);									// 103
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RESERVED1);									// 104
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RESERVED2);									// 105
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RESERVED3);									// 106
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RESERVED4);									// 107
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RESERVED5);									// 108
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RESERVED6);									// 109
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RESERVED7);									// 110
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RESERVED8);									// 111
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RESERVED9);									// 112
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS4_RESERVED10);									// 113
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_A);										// 114
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_B);										// 115
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_X);										// 116
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_Y);										// 117
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_LEFT_BUMPER);							// 118
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RIGHT_BUMPER);							// 119
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_MENU);									// 120
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_VIEW);									// 121
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_LEFT_TRIGGER_PULL);						// 122
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_LEFT_TRIGGER_CLICK);						// 123
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RIGHT_TRIGGER_PULL);						// 124
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RIGHT_TRIGGER_CLICK);					// 125
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_LEFT_STICK_MOVE);						// 126
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_LEFT_STICK_CLICK);						// 127
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_LEFT_STICK_DPAD_NORTH);					// 128
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_LEFT_STICK_DPAD_SOUTH);					// 129
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_LEFT_STICK_DPAD_WEST);					// 130
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_LEFT_STICK_DPAD_EAST);					// 131
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RIGHT_STICK_MOVE);						// 132
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RIGHT_STICK_CLICK);						// 133
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RIGHT_STICK_DPAD_NORTH);					// 134
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RIGHT_STICK_DPAD_SOUTH);					// 135
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RIGHT_STICK_DPAD_WEST);					// 136
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RIGHT_STICK_DPAD_EAST);					// 137
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_DPAD_NORTH);								// 138
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_DPAD_SOUTH);								// 139
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_DPAD_WEST);								// 140
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_DPAD_EAST);								// 141
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RESERVED0);								// 142
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RESERVED1);								// 143
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RESERVED2);								// 144
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RESERVED3);								// 145
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RESERVED4);								// 146
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RESERVED5);								// 147
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RESERVED6);								// 148
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RESERVED7);								// 149
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RESERVED8);								// 150
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RESERVED9);								// 151
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_ONE_RESERVED10);								// 152
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_A);										// 153
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_B);										// 154
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_X);										// 155
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_Y);										// 156
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_LEFT_BUMPER);							// 157
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RIGHT_BUMPER);							// 158
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_START);									// 159
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_BACK);									// 160
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_LEFT_TRIGGER_PULL);						// 161
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_LEFT_TRIGGER_CLICK);						// 162
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RIGHT_TRIGGER_PULL);						// 163
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RIGHT_TRIGGER_CLICK);					// 164
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_LEFT_STICK_MOVE);						// 165
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_LEFT_STICK_CLICK);						// 166
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_LEFT_STICK_DPAD_NORTH);					// 167
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_LEFT_STICK_DPAD_SOUTH);					// 168
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_LEFT_STICK_DPAD_WEST);					// 169
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_LEFT_STICK_DPAD_EAST);					// 170
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RIGHT_STICK_MOVE);						// 171
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RIGHT_STICK_CLICK);						// 172
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RIGHT_STICK_DPAD_NORTH);					// 173
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RIGHT_STICK_DPAD_SOUTH);					// 174
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RIGHT_STICK_DPAD_WEST);					// 175
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RIGHT_STICK_DPAD_EAST);					// 176
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_DPAD_NORTH);								// 177
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_DPAD_SOUTH);								// 178
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_DPAD_WEST);								// 179
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_DPAD_EAST);								// 180
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RESERVED0);								// 181
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RESERVED1);								// 182
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RESERVED2);								// 183
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RESERVED3);								// 184
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RESERVED4);								// 185
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RESERVED5);								// 186
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RESERVED6);								// 187
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RESERVED7);								// 188
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RESERVED8);								// 189
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RESERVED9);								// 190
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_XBOX_360_RESERVED10);								// 191
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_A);										// 192
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_B);										// 193
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_X);										// 194
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_Y);										// 195
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_LEFT_BUMPER);								// 196
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RIGHT_BUMPER);								// 197
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_PLUS);										// 198
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_MINUS);									// 199
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_CAPTURE);									// 200
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_LEFT_TRIGGER_PULL);						// 201
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_LEFT_TRIGGER_CLICK);						// 202
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RIGHT_TRIGGER_PULL);						// 203
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RIGHT_TRIGGER_CLICK);						// 204
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_LEFT_STICK_MOVE);							// 205
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_LEFT_STICK_CLICK);							// 206
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_LEFT_STICK_DPAD_NORTH);					// 207
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_LEFT_STICK_DPAD_SOUTH);					// 208
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_LEFT_STICK_DPAD_WEST);						// 209
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_LEFT_STICK_DPAD_EAST);						// 210
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RIGHT_STICK_MOVE);							// 211
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RIGHT_STICK_CLICK);						// 212
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RIGHT_STICK_DPAD_NORTH);					// 213
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RIGHT_STICK_DPAD_SOUTH);					// 214
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RIGHT_STICK_DPAD_WEST);					// 215
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RIGHT_STICK_DPAD_EAST);					// 216
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_DPAD_NORTH);								// 217
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_DPAD_SOUTH);								// 218
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_DPAD_WEST);								// 219
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_DPAD_EAST);								// 220
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_PRO_GYRO_MOVE);							// 221
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_PRO_GYRO_PITCH);							// 222
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_PRO_GYRO_YAW);								// 223
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_PRO_GYRO_ROLL);							// 224
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RESERVED0);								// 225
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RESERVED1);								// 226
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RESERVED2);								// 227
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RESERVED3);								// 228
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RESERVED4);								// 229
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RESERVED5);								// 230
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RESERVED6);								// 231
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RESERVED7);								// 232
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RESERVED8);								// 233
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RESERVED9);								// 234
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_SWITCH_RESERVED10);								// 235
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_OPTION);										// 242
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_CREATE);										// 243
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_MUTE);										// 244
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTPAD_TOUCH);								// 245
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTPAD_SWIPE);								// 246
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTPAD_CLICK);								// 247
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTPAD_DPADNORTH);							// 248
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTPAD_DPADSOUTH);							// 249
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTPAD_DPADWEST);							// 250
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTPAD_DPADEAST);							// 251
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTPAD_TOUCH);								// 252
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTPAD_SWIPE);								// 253
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTPAD_CLICK);								// 254
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTPAD_DPADNORTH);							// 255
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTPAD_DPADSOUTH);							// 256
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTPAD_DPADWEST);							// 257
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTPAD_DPADEAST);							// 258
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_CENTERPAD_TOUCH);								// 259
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_CENTERPAD_SWIPE);								// 260
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_CENTERPAD_CLICK);								// 261
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_CENTERPAD_DPADNORTH);							// 262
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_CENTERPAD_DPADSOUTH);							// 263
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_CENTERPAD_DPADWEST);							// 264
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_CENTERPAD_DPADEAST);							// 265
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTTRIGGER_PULL);							// 266
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTTRIGGER_CLICK);							// 267
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTTRIGGER_PULL);							// 268
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTTRIGGER_CLICK);							// 269
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTSTICK_MOVE);								// 270
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTSTICK_CLICK);								// 271
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTSTICK_DPADNORTH);							// 272
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTSTICK_DPADSOUTH);							// 273
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTSTICK_DPADWEST);							// 274
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_LEFTSTICK_DPADEAST);							// 275
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTSTICK_MOVE);								// 276
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTSTICK_CLICK);							// 277
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTSTICK_DPADNORTH);						// 278
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTSTICK_DPADSOUTH);						// 279
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTSTICK_DPADWEST);							// 280
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RIGHTSTICK_DPADEAST);							// 281
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_DPAD_NORTH);									// 282
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_DPAD_SOUTH);									// 283
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_DPAD_WEST);									// 284
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_DPAD_EAST);									// 285
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_GYRO_MOVE);									// 286
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_GYRO_PITCH);									// 287
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_GYRO_YAW);									// 288
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_GYRO_ROLL);									// 289
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_DPAD_MOVE);									// 290
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED1);									// 291
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED2);									// 292
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED3);									// 293
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED4);									// 294
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED5);									// 295
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED6);									// 296
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED7);									// 297
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED8);									// 298
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED9);									// 299
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED10);									// 300
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED11);									// 301
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED12);									// 302
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED13);									// 303
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED14);									// 304
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED15);									// 305
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED16);									// 306
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED17);									// 307
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED18);									// 308
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED19);									// 309
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_PS5_RESERVED20);									// 310
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_A);										// 311
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_B);										// 312
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_X);										// 313
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_Y);										// 314
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_L1);									// 315
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_R1);									// 316
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_MENU);									// 317
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_VIEW);									// 318
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTPAD_TOUCH);							// 319
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTPAD_SWIPE);							// 320
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTPAD_CLICK);							// 321
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTPAD_DPADNORTH);						// 322
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTPAD_DPADSOUTH);						// 323
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTPAD_DPADWEST);						// 324
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTPAD_DPADEAST);						// 325
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTPAD_TOUCH);						// 326
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTPAD_SWIPE);						// 327
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTPAD_CLICK);						// 328
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTPAD_DPADNORTH);					// 329
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTPAD_DPADSOUTH);					// 330
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTPAD_DPADWEST);						// 331
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTPAD_DPADEAST);						// 332
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_L2_SOFTPULL);							// 333
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_L2);									// 334
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_R2_SOFTPULL);							// 335
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_R2);									// 336
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTSTICK_MOVE);						// 337
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_L3);									// 338
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTSTICK_DPADNORTH);					// 339
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTSTICK_DPADSOUTH);					// 340
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTSTICK_DPADWEST);					// 341
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTSTICK_DPADEAST);					// 342
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_LEFTSTICK_TOUCH);						// 343
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTSTICK_MOVE);						// 344
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_R3);									// 345
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTSTICK_DPADNORTH);					// 346
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTSTICK_DPADSOUTH);					// 347
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTSTICK_DPADWEST);					// 348
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTSTICK_DPADEAST);					// 349
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RIGHTSTICK_TOUCH);						// 350
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_L4);									// 351
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_R4);									// 352
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_L5);									// 353
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_R5);									// 354
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_DPAD_MOVE);								// 355
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_DPAD_NORTH);							// 356
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_DPAD_SOUTH);							// 357
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_DPAD_WEST);								// 358
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_DPAD_EAST);								// 359
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_GYRO_MOVE);								// 360
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_GYRO_PITCH);							// 361
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_GYRO_YAW);								// 362
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_GYRO_ROLL);								// 363
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED1);								// 364
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED2);								// 365
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED3);								// 366
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED4);								// 367
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED5);								// 368
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED6);								// 369
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED7);								// 370
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED8);								// 371
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED9);								// 372
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED10);							// 373
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED11);							// 374
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED12);							// 375
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED13);							// 376
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED14);							// 377
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED15);							// 378
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED16);							// 379
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED17);							// 380
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED18);							// 381
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED19);							// 382
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_STEAMDECK_RESERVED20);							// 383
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_COUNT);											// 384
	BIND_CONSTANT(INPUT_ACTION_ORIGIN_MAXIMUMPOSSIBLEVALUE);							// 32767

	// STEAM INPUT TYPE /////////////////////////
	BIND_CONSTANT(INPUT_TYPE_UNKNOWN);													// 0
	BIND_CONSTANT(INPUT_TYPE_STEAM_CONTROLLER);											// 1
	BIND_CONSTANT(INPUT_TYPE_XBOX360_CONTROLLER);										// 2
	BIND_CONSTANT(INPUT_TYPE_XBOXONE_CONTROLLER);										// 3
	BIND_CONSTANT(INPUT_TYPE_GENERIC_XINPUT);											// 4
	BIND_CONSTANT(INPUT_TYPE_PS4_CONTROLLER);											// 5
	BIND_CONSTANT(INPUT_TYPE_APPLE_MFI_CONTROLLER);										// 6
	BIND_CONSTANT(INPUT_TYPE_ANDROID_CONTROLLER);										// 7
	BIND_CONSTANT(INPUT_TYPE_SWITCH_JOYCON_PAIR);										// 8
	BIND_CONSTANT(INPUT_TYPE_SWITCH_JOYCON_SINGLE);										// 9
	BIND_CONSTANT(INPUT_TYPE_SWITCH_PRO_CONTROLLER);									// 10
	BIND_CONSTANT(INPUT_TYPE_MOBILE_TOUCH);												// 11
	BIND_CONSTANT(INPUT_TYPE_PS3_CONTROLLER);											// 12
	BIND_CONSTANT(INPUT_TYPE_PS5_CONTROLLER);											// 13
	BIND_CONSTANT(INPUT_TYPE_STEAM_DECK_CONTROLLER);									// 14
	BIND_CONSTANT(INPUT_TYPE_COUNT);													// 15
	BIND_CONSTANT(INPUT_TYPE_MAXIMUM_POSSIBLE_VALUE);									// 255

	// CONFIGURATION ENABLE TYPE ////////////////
	BIND_CONSTANT(INPUT_CONFIGURATION_ENABLE_TYPE_NONE);								// 0x0000
	BIND_CONSTANT(INPUT_CONFIGURATION_ENABLE_TYPE_PLAYSTATION);							// 0x0001
	BIND_CONSTANT(INPUT_CONFIGURATION_ENABLE_TYPE_XBOX);								// 0x0002
	BIND_CONSTANT(INPUT_CONFIGURATION_ENABLE_TYPE_GENERIC);								// 0x0004
	BIND_CONSTANT(INPUT_CONFIGURATION_ENABLE_TYPE_SWITCH);								// 0x0008
	
	// GLYPH SIZE ///////////////////////////////
	BIND_CONSTANT(INPUT_GLYPH_SIZE_SMALL);												// 0
	BIND_CONSTANT(INPUT_GLYPH_SIZE_MEDIUM);												// 1
	BIND_CONSTANT(INPUT_GLYPH_SIZE_LARGE);												// 2
	BIND_CONSTANT(INPUT_GLYPH_SIZE_COUNT);												// 3

	// GLYPH STYLE //////////////////////////////
	BIND_CONSTANT(INPUT_GLYPH_STYLE_KNOCKOUT);											// 0x0
	BIND_CONSTANT(INPUT_GLYPH_STYLE_LIGHT);												// 0x1
	BIND_CONSTANT(INPUT_GLYPH_STYLE_DARK);												// 0x2
	BIND_CONSTANT(INPUT_GLYPH_STYLE_NEUTRAL_COLOR_ABXY);								// 0x10
	BIND_CONSTANT(INPUT_GLYPH_STYLE_SOLID_ABXY);										// 0x20

	// STEAM ITEM FLAGS /////////////////////////
	BIND_CONSTANT(STEAM_ITEM_NO_TRADE);													// (1<<0)
	BIND_CONSTANT(STEAM_ITEM_REMOVED);													// (1<<8)
	BIND_CONSTANT(STEAM_ITEM_CONSUMED);													// (1<<9)

	// CHAT MEMBER STATE CHANGE /////////////////
	BIND_CONSTANT(CHAT_MEMBER_STATE_CHANGE_ENTERED);									// 0X0001
	BIND_CONSTANT(CHAT_MEMBER_STATE_CHANGE_LEFT);										// 0X0002
	BIND_CONSTANT(CHAT_MEMBER_STATE_CHANGE_DISCONNECTED);								// 0X0004
	BIND_CONSTANT(CHAT_MEMBER_STATE_CHANGE_KICKED);										// 0X0008
	BIND_CONSTANT(CHAT_MEMBER_STATE_CHANGE_BANNED);										// 0X0010

	// LOBBY COMPARISON /////////////////////////
	BIND_CONSTANT(LOBBY_COMPARISON_EQUAL_TO_OR_LESS_THAN);								// -2
	BIND_CONSTANT(LOBBY_COMPARISON_LESS_THAN);											// -1
	BIND_CONSTANT(LOBBY_COMPARISON_EQUAL);												// 0
	BIND_CONSTANT(LOBBY_COMPARISON_GREATER_THAN);										// 1
	BIND_CONSTANT(LOBBY_COMPARISON_EQUAL_TO_GREATER_THAN);								// 2
	BIND_CONSTANT(LOBBY_COMPARISON_NOT_EQUAL);											// 3

	// LOBBY DISTANCE FILTER ////////////////////
	BIND_CONSTANT(LOBBY_DISTANCE_FILTER_CLOSE);											// 0
	BIND_CONSTANT(LOBBY_DISTANCE_FILTER_DEFAULT);										// 1
	BIND_CONSTANT(LOBBY_DISTANCE_FILTER_FAR);											// 2
	BIND_CONSTANT(LOBBY_DISTANCE_FILTER_WORLDWIDE);										// 3

	// LOBBY TYPE ///////////////////////////////
	BIND_CONSTANT(LOBBY_TYPE_PRIVATE);													// 0
	BIND_CONSTANT(LOBBY_TYPE_FRIENDS_ONLY);												// 1
	BIND_CONSTANT(LOBBY_TYPE_PUBLIC);													// 2
	BIND_CONSTANT(LOBBY_TYPE_INVISIBLE);												// 3

	// MATCHMAKING SERVER RESPONSE //////////////
	BIND_CONSTANT(SERVER_RESPONDED);													// 0
	BIND_CONSTANT(SERVER_FAILED_TO_RESPOND);											// 1
	BIND_CONSTANT(NO_SERVERS_LISTED_ON_MASTER_SERVER);									// 2

	// AUDIO PLAYBACK STATUS ////////////////////
	BIND_CONSTANT(AUDIO_PLAYBACK_UNDEFINED);											// 0
	BIND_CONSTANT(AUDIO_PLAYBACK_PLAYING);												// 1
	BIND_CONSTANT(AUDIO_PLAYBACK_PAUSED);												// 2
	BIND_CONSTANT(AUDIO_PLAYBACK_IDLE);													// 3

	// P2P SEND /////////////////////////////////
	BIND_CONSTANT(P2P_SEND_UNRELIABLE);													// 0
	BIND_CONSTANT(P2P_SEND_UNRELIABLE_NO_DELAY);										// 1
	BIND_CONSTANT(P2P_SEND_RELIABLE);													// 2
	BIND_CONSTANT(P2P_SEND_RELIABLE_WITH_BUFFERING);									// 3

	// P2P SESSION ERROR ////////////////////////
	BIND_CONSTANT(P2P_SESSION_ERROR_NONE);												// 0
	BIND_CONSTANT(P2P_SESSION_ERROR_NOT_RUNNING_APP);									// 1
	BIND_CONSTANT(P2P_SESSION_ERROR_NO_RIGHTS_TO_APP);									// 2
	BIND_CONSTANT(P2P_SESSION_ERROR_DESTINATION_NOT_LOGGED_ON);							// 3
	BIND_CONSTANT(P2P_SESSION_ERROR_TIMEOUT);											// 4
	BIND_CONSTANT(P2P_SESSION_ERROR_MAX);												// 5

	// SNET SOCKET CONNECTION TYPE //////////////
	BIND_CONSTANT(NET_SOCKET_CONNECTION_TYPE_NOT_CONNECTED);							// 0
	BIND_CONSTANT(NET_SOCKET_CONNECTION_TYPE_UDP);										// 1
	BIND_CONSTANT(NET_SOCKET_CONNECTION_TYPE_UDP_RELAY);								// 2

	// SNET SOCKET STATE ////////////////////////
	BIND_CONSTANT(NET_SOCKET_STATE_INVALID);											// 0
	BIND_CONSTANT(NET_SOCKET_STATE_CONNECTED);											// 1
	BIND_CONSTANT(NET_SOCKET_STATE_INITIATED);											// 10
	BIND_CONSTANT(NET_SOCKET_STATE_LOCAL_CANDIDATE_FOUND);								// 11
	BIND_CONSTANT(NET_SOCKET_STATE_RECEIVED_REMOTE_CANDIDATES);							// 12
	BIND_CONSTANT(NET_SOCKET_STATE_CHALLENGE_HANDSHAKE);								// 15
	BIND_CONSTANT(NET_SOCKET_STATE_DISCONNECTING);										// 21
	BIND_CONSTANT(NET_SOCKET_STATE_LOCAL_DISCONNECT);									// 22
	BIND_CONSTANT(NET_SOCKET_STATE_TIMEOUT_DURING_CONNECT);								// 23
	BIND_CONSTANT(NET_SOCKET_STATE_REMOTE_END_DISCONNECTED);							// 24
	BIND_CONSTANT(NET_SOCKET_STATE_BROKEN);												// 25

	// NETWORKING CONFIGURATION VALUE ///////////
	BIND_CONSTANT(NETWORKING_CONFIG_INVALID);											// 0
	BIND_CONSTANT(NETWORKING_CONFIG_FAKE_PACKET_LOSS_SEND);								// 2
	BIND_CONSTANT(NETWORKING_CONFIG_FAKE_PACKET_LOSS_RECV);								// 3
	BIND_CONSTANT(NETWORKING_CONFIG_FAKE_PACKET_LAG_SEND);								// 4
	BIND_CONSTANT(NETWORKING_CONFIG_FAKE_PACKET_LAG_RECV);								// 5
	BIND_CONSTANT(NETWORKING_CONFIG_FAKE_PACKET_REORDER_SEND);							// 6
	BIND_CONSTANT(NETWORKING_CONFIG_FAKE_PACKET_REORDER_RECV);							// 7
	BIND_CONSTANT(NETWORKING_CONFIG_FAKE_PACKET_REORDER_TIME);							// 8
	BIND_CONSTANT(NETWORKING_CONFIG_FAKE_PACKET_DUP_SEND);								// 26
	BIND_CONSTANT(NETWORKING_CONFIG_FAKE_PACKET_DUP_REVC);								// 27
	BIND_CONSTANT(NETWORKING_CONFIG_FAKE_PACKET_DUP_TIME_MAX);							// 28
	BIND_CONSTANT(NETWORKING_CONFIG_TIMEOUT_INITIAL);									// 24
	BIND_CONSTANT(NETWORKING_CONFIG_TIMEOUT_CONNECTED);									// 25
	BIND_CONSTANT(NETWORKING_CONFIG_SEND_BUFFER_SIZE);									// 9
	BIND_CONSTANT(NETWORKING_CONFIG_SEND_RATE_MIN);										// 10
	BIND_CONSTANT(NETWORKING_CONFIG_SEND_RATE_MAX);										// 11
	BIND_CONSTANT(NETWORKING_CONFIG_NAGLE_TIME);										// 12
	BIND_CONSTANT(NETWORKING_CONFIG_IP_ALLOW_WITHOUT_AUTH);								// 23
	BIND_CONSTANT(NETWORKING_CONFIG_SDR_CLIENT_CONSEC_PING_TIMEOUT_FAIL_INITIAL);		// 19
	BIND_CONSTANT(NETWORKING_CONFIG_SDR_CLIENT_CONSEC_PING_TIMEOUT_FAIL);				// 20
	BIND_CONSTANT(NETWORKING_CONFIG_SDR_CLIENT_MIN_PINGS_BEFORE_PING_ACCURATE);			// 21
	BIND_CONSTANT(NETWORKING_CONFIG_SDR_CLIENT_SINGLE_SOCKET);							// 22
	BIND_CONSTANT(NETWORKING_CONFIG_SDR_CLIENT_FORCE_RELAY_CLUSTER);					// 29
	BIND_CONSTANT(NETWORKING_CONFIG_SDR_CLIENT_DEBUG_TICKET_ADDRESS);					// 30
	BIND_CONSTANT(NETWORKING_CONFIG_SDR_CLIENT_FORCE_PROXY_ADDR);						// 31
	BIND_CONSTANT(NETWORKING_CONFIG_LOG_LEVEL_ACK_RTT);									// 13
	BIND_CONSTANT(NETWORKING_CONFIG_LOG_LEVEL_PACKET_DECODE);							// 14
	BIND_CONSTANT(NETWORKING_CONFIG_LOG_LEVEL_MESSAGE);									// 15
	BIND_CONSTANT(NETWORKING_CONFIG_LOG_LEVEL_PACKET_GAPS);								// 16
	BIND_CONSTANT(NETWORKING_CONFIG_LOG_LEVEL_P2P_RENDEZVOUS);							// 17
	BIND_CONSTANT(NETWORKING_CONFIG_LOG_LEVEL_SRD_RELAY_PINGS);							// 18

	// NETWORKING GET CONFIGURATION VALUE RESULT
	BIND_CONSTANT(NETWORKING_GET_CONFIG_VALUE_BAD_VALUE);								// -1
	BIND_CONSTANT(NETWORKING_GET_CONFIG_VALUE_BAD_SCOPE_OBJ);							// -2
	BIND_CONSTANT(NETWORKING_GET_CONFIG_VALUE_BUFFER_TOO_SMALL);						// -3
	BIND_CONSTANT(NETWORKING_GET_CONFIG_VALUE_OK);										// 1
	BIND_CONSTANT(NETWORKING_GET_CONFIG_VALUE_OK_INHERITED);							// 2
	BIND_CONSTANT(NETWORKING_GET_CONFIG_VALUE_FORCE_32BIT);								// 0X7FFFFFFF
	
	// NETWORKING CONNECTION STATE //////////////
	BIND_CONSTANT(CONNECTION_STATE_NONE);												// 0
	BIND_CONSTANT(CONNECTION_STATE_CONNECTING);											// 1
	BIND_CONSTANT(CONNECTION_STATE_FINDING_ROUTE);										// 2
	BIND_CONSTANT(CONNECTION_STATE_CONNECTED);											// 3
	BIND_CONSTANT(CONNECTION_STATE_CLOSED_BY_PEER);										// 4
	BIND_CONSTANT(CONNECTION_STATE_PROBLEM_DETECTED_LOCALLY);							// 5
	BIND_CONSTANT(CONNECTION_STATE_FIN_WAIT);											// -1
	BIND_CONSTANT(CONNECTION_STATE_LINGER);												// -2
	BIND_CONSTANT(CONNECTION_STATE_DEAD);												// -3
	BIND_CONSTANT(CONNECTION_STATE_FORCE32BIT);											// 0X7FFFFFFF

	// NETWORKING CONNECTION END ////////////////
	BIND_CONSTANT(CONNECTION_END_INVALID);												// 0
	BIND_CONSTANT(CONNECTION_END_APP_MIN);												// 1000
	BIND_CONSTANT(CONNECTION_END_MAX);													// 1999
	BIND_CONSTANT(CONNECTION_END_APP_EXCEPTION_MIN);									// 2000
	BIND_CONSTANT(CONNECTION_END_APP_EXCEPTION_MAX);									// 2999
	BIND_CONSTANT(CONNECTION_END_LOCAL_MIN);											// 3000
	BIND_CONSTANT(CONNECTION_END_LOCAL_OFFLINE_MODE);									// 3001
	BIND_CONSTANT(CONNECTION_END_LOCAL_MANY_RELAY_CONNECTIVITY);						// 3002
	BIND_CONSTANT(CONNECTION_END_LOCAL_HOSTED_sERVER_PRIMARY_RELAY);					// 3003
	BIND_CONSTANT(CONNECTION_END_LOCAL_NETWORK_CONFIG);									// 3004
	BIND_CONSTANT(CONNECTION_END_LOCAL_RIGHTS);											// 3005
	BIND_CONSTANT(CONNECTION_END_LOCAL_MAX);											// 3999
	BIND_CONSTANT(CONNECTION_END_REMOVE_MIN);											// 4000
	BIND_CONSTANT(CONNECTION_END_REMOTE_TIMEOUT);										// 4001
	BIND_CONSTANT(CONNECTION_END_REMOTE_BAD_CRYPT);										// 4002
	BIND_CONSTANT(CONNECTION_END_REMOTE_BAD_CERT);										// 4003
	BIND_CONSTANT(CONNECTION_END_REMOTE_NOT_LOGGED_IN);									// 4004
	BIND_CONSTANT(CONNECTION_END_REMOTE_NOT_RUNNING_APP);								// 4005
	BIND_CONSTANT(CONNECTION_END_BAD_PROTOCOL_VERSION);									// 4006
	BIND_CONSTANT(CONNECTION_END_REMOTE_MAX);											// 4999
	BIND_CONSTANT(CONNECTION_END_MISC_MIN);												// 5000
	BIND_CONSTANT(CONNECTION_END_MISC_GENERIC);											// 5001
	BIND_CONSTANT(CONNECTION_END_MISC_INTERNAL_ERROR);									// 5002
	BIND_CONSTANT(CONNECTION_END_MISC_TIMEOUT);											// 5003
	BIND_CONSTANT(CONNECTION_END_MISC_RELAY_CONNECTIVITY);								// 5004
	BIND_CONSTANT(CONNECTION_END_MISC_STEAM_CONNECTIVITY);								// 5005
	BIND_CONSTANT(CONNECTION_END_MISC_NO_RELAY_SESSIONS_TO_CLIENT);						// 5006
	BIND_CONSTANT(CONNECTION_END_MISC_MAX);												// 5999

	// NETWORKING IDENTITY TYPE /////////////////
	BIND_CONSTANT(IDENTITY_TYPE_INVALID);												// 0
	BIND_CONSTANT(IDENTITY_TYPE_STEAMID);												// 16
	BIND_CONSTANT(IDENTITY_TYPE_IP_ADDRESS);											// 1
	BIND_CONSTANT(IDENTITY_TYPE_GENERIC_STRING);										// 2
	BIND_CONSTANT(IDENTITY_TYPE_GENERIC_BYTES);											// 3
	BIND_CONSTANT(IDENTITY_TYPE_UNKNOWN_TYPE);											// 4
	BIND_CONSTANT(IDENTITY_TYPE_XBOX_PAIRWISE);											// 17
	BIND_CONSTANT(IDENTITY_TYPE_SONY_PSN);												// 18
	BIND_CONSTANT(IDENTITY_TYPE_GOOGLE_STADIA);											// 19 
	BIND_CONSTANT(IDENTITY_TYPE_FORCE_32BIT);											// 0X7FFFFFFF

	// NETWORKING SOCKET DEBUG OUTPUT ///////////
	BIND_CONSTANT(NETWORKING_SOCKET_DEBUG_OUTPUT_TYPE_NONE);							// 0
	BIND_CONSTANT(NETWORKING_SOCKET_DEBUG_OUTPUT_TYPE_BUG);								// 1
	BIND_CONSTANT(NETWORKING_SOCKET_DEBUG_OUTPUT_TYPE_ERROR);							// 2
	BIND_CONSTANT(NETWORKING_SOCKET_DEBUG_OUTPUT_TYPE_IMPORTANT);						// 3
	BIND_CONSTANT(NETWORKING_SOCKET_DEBUG_OUTPUT_TYPE_WARNING);							// 4
	BIND_CONSTANT(NETWORKING_SOCKET_DEBUG_OUTPUT_TYPE_MSG);								// 5
	BIND_CONSTANT(NETWORKING_SOCKET_DEBUG_OUTPUT_TYPE_VERBOSE);							// 6,
	BIND_CONSTANT(NETWORKING_SOCKET_DEBUG_OUTPUT_TYPE_DEBUG);							// 7
	BIND_CONSTANT(NETWORKING_SOCKET_DEBUG_OUTPUT_TYPE_EVERYTHING);						// 8
	BIND_CONSTANT(NETWORKING_SOCKET_DEBUG_OUTPUT_TYPE_FORCE_32BIT);						// 0X7FFFFFFF
	
	// NETWORKING AVAILABILITY //////////////////
	BIND_CONSTANT(NETWORKING_AVAILABILITY_CANNOT_TRY);									// -102
	BIND_CONSTANT(NETWORKING_AVAILABILITY_FAILED);										// -101
	BIND_CONSTANT(NETWORKING_AVAILABILITY_PREVIOUSLY);									// -100
	BIND_CONSTANT(NETWORKING_AVAILABILITY_NEVER_TRIED);									// 1
	BIND_CONSTANT(NETWORKING_AVAILABILITY_WAITING);										// 2
	BIND_CONSTANT(NETWORKING_AVAILABILITY_ATTEMPTING);									// 3
	BIND_CONSTANT(NETWORKING_AVAILABILITY_CURRENT);										// 100
	BIND_CONSTANT(NETWORKING_AVAILABILITY_UNKNOWN);										// 0
	BIND_CONSTANT(NETWORKING_AVAILABILITY_FORCE_32BIT);									// 0X7FFFFFFF

	// NETWORKING CONFIGURATION SCOPE ///////////
	BIND_CONSTANT(NETWORKING_CONFIG_SCOPE_GLOBAL);										// 1
	BIND_CONSTANT(NETWORKING_CONFIG_SCOPE_SOCKETS_INTERFACE);							// 2
	BIND_CONSTANT(NETWORKING_CONFIG_SCOPE_LISTEN_SOCKET);								// 3
	BIND_CONSTANT(NETWORKING_CONFIG_SCOPE_CONNECTION);									// 4
	BIND_CONSTANT(NETWORKING_CONFIG_SCOPE_FORCE_32BIT);									// 0X7FFFFFFF

	// NETWORKING CONFIGURATION DATA TYPE ///////
	BIND_CONSTANT(NETWORKING_CONFIG_TYPE_INT32);										// 1
	BIND_CONSTANT(NETWORKING_CONFIG_TYPE_INT64);										// 2
	BIND_CONSTANT(NETWORKING_CONFIG_TYPE_FLOAT);										// 3
	BIND_CONSTANT(NETWORKING_CONFIG_TYPE_STRING);										// 4
	BIND_CONSTANT(NETWORKING_CONFIG_TYPE_FUNCTION_PTR);									// 5
	BIND_CONSTANT(NETWORKING_CONFIG_TYPE_FORCE_32BIT);									// 0x7fffffff

	// PARENTAL SETTINGS ////////////////////////
	BIND_CONSTANT(FEATURE_INVALID);														// 0
	BIND_CONSTANT(FEATURE_STORE);														// 1
	BIND_CONSTANT(FEATURE_COMMUNITY);													// 2
	BIND_CONSTANT(FEATURE_PROFILE);														// 3
	BIND_CONSTANT(FEATURE_FRIENDS);														// 4
	BIND_CONSTANT(FEATURE_NEWS);														// 5
	BIND_CONSTANT(FEATURE_TRADING);														// 6
	BIND_CONSTANT(FEATURE_SETTINGS);													// 7
	BIND_CONSTANT(FEATURE_CONSOLE);														// 8
	BIND_CONSTANT(FEATURE_BROWSER);														// 9
	BIND_CONSTANT(FEATURE_PARENTAL_SETUP);												// 10
	BIND_CONSTANT(FEATURE_LIBRARY);														// 11
	BIND_CONSTANT(FEATURE_TEST);														// 12
	BIND_CONSTANT(FEATURE_SITE_LICENSE);												// 13
	BIND_CONSTANT(FEATURE_MAX);

	// STEAM PARTY BEACON LOCATION TYPE /////////
	BIND_CONSTANT(STEAM_PARTY_BEACON_LOCATIONTYPE_INVALID);								// 0
	BIND_CONSTANT(STEAM_PARTY_BEACON_LOCATIONTYPE_CHAT_GROUP);							// 1
	BIND_CONSTANT(STEAM_PARTY_BEACON_LOCATION_TYPE_MAX);

	// STEAM PARTY BEACON LOCATION DATA /////////
	BIND_CONSTANT(STEAM_PARTY_BEACON_LOCATION_DATA);									// 0
	BIND_CONSTANT(STEAM_PARTY_BEACON_LOCATION_DATA_NAME);								// 1
	BIND_CONSTANT(STEAM_PARTY_BEACON_LOCATION_DATA_URL_SMALL);
	BIND_CONSTANT(STEAM_PARTY_BEACON_LOCATION_DATA_URL_MEDIUM);
	BIND_CONSTANT(STEAM_PARTY_BEACON_LOCATION_DATA_URL_LARGE);

	// REMOTE STORAGE PLATFORM //////////////////
	BIND_CONSTANT(REMOTE_STORAGE_PLATFORM_NONE);										// 0
	BIND_CONSTANT(REMOTE_STORAGE_PLATFORM_WINDOWS);										// (1<<0)
	BIND_CONSTANT(REMOTE_STORAGE_PLATFORM_OSX);											// (1<<1)
	BIND_CONSTANT(REMOTE_STORAGE_PLATFORM_PS3);											// (1<<2)
	BIND_CONSTANT(REMOTE_STORAGE_PLATFORM_LINUX);										// (1<<3)
	BIND_CONSTANT(REMOTE_STORAGE_PLATFORM_RESERVED2);									// (1<<4)
	BIND_CONSTANT(REMOTE_STORAGE_PLATFORM_ALL);											// 0XFFFFFFFF

	// REMOTE STORAGE PUBLISHED FILE VISIBILITY /
	BIND_CONSTANT(REMOTE_STORAGE_PUBLISHED_VISIBLITY_PUBLIC);							// 0
	BIND_CONSTANT(REMOTE_STORAGE_PUBLISHED_VISIBLITY_FRIENDS_ONLY);						// 1
	BIND_CONSTANT(REMOTE_STORAGE_PUBLISHED_VISIBLITY_PRIVATE);							// 2

	// UGC READ ACTION //////////////////////////
	BIND_CONSTANT(UGC_READ_CONTINUE_READING_UNTIL_FINISHED);							// 0
	BIND_CONSTANT(UGC_READ_CONTINUE_READING);											// 1
	BIND_CONSTANT(UGC_READ_CLOSE);														// 2

	// WORKSHOP ENUMERATION TYPE ////////////////
	BIND_CONSTANT(WORKSHOP_ENUMERATION_TYPE_RANKED_BY_VOTE);							// 0
	BIND_CONSTANT(WORKSHOP_ENUMERATION_TYPE_RECENT);									// 1
	BIND_CONSTANT(WORKSHOP_ENUMERATION_TYPE_TRENDING);									// 2
	BIND_CONSTANT(WORKSHOP_ENUMERATION_TYPE_FAVORITES_OF_FRIENDS);						// 3
	BIND_CONSTANT(WORKSHOP_ENUMERATION_TYPE_VOTED_BY_FRIENDS);							// 4
	BIND_CONSTANT(WORKSHOP_ENUMERATION_TYPE_CONTENT_BY_FRIENDS);						// 5
	BIND_CONSTANT(WORKSHOP_ENUMERATION_TYPE_RECENT_FROM_FOLLOWED_USERS);				// 6

	// WORKSHOP FILE ACTION /////////////////////
	BIND_CONSTANT(WORKSHOP_FILE_ACTION_PLAYED);											// 0
	BIND_CONSTANT(WORKSHOP_FILE_ACTION_COMPLETED);										// 1

	// WORKSHOP FILE TYPE ///////////////////////
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_FIRST);											// 0
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_COMMUNITY);										// 0
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_MICROTRANSACTION);									// 1
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_COLLECTION);										// 2
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_ART);												// 3
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_VIDEO);											// 4
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_SCREENSHOT);										// 5
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_GAME);												// 6
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_SOFTWARE);											// 7
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_CONCEPT);											// 8
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_WEB_GUIDE);										// 9
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_INTEGRATED_GUIDE);									// 10
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_MERCH);											// 11
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_CONTROLLER_BINDING);								// 12
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_STEAMWORKS_ACCESS_INVITE);							// 13
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_STEAM_VIDEO);										// 14
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_GAME_MANAGED_ITEM);								// 15
	BIND_CONSTANT(WORKSHOP_FILE_TYPE_MAX);												// 16

	// WORKSHOP VIDEO PROVIDER //////////////////
	BIND_CONSTANT(WORKSHOP_VIDEO_PROVIDER_NONE);										// 0
	BIND_CONSTANT(WORKSHOP_VIDEO_PROVIDER_YOUTUBE);										// 1

	// WORKSHOP VOTE ////////////////////////////
	BIND_CONSTANT(WORKSHOP_VOTE_UNVOTED);												// 0
	BIND_CONSTANT(WORKSHOP_VOTE_FOR);													// 1
	BIND_CONSTANT(WORKSHOP_VOTE_AGAINST);												// 2
	BIND_CONSTANT(WORKSHOP_VOTE_LATER);													// 3

	// LOCAL FILE CHANGES ///////////////////////
	BIND_CONSTANT(LOCAL_FILE_CHANGE_INVALID);											// 0
	BIND_CONSTANT(LOCAL_FILE_CHANGE_FILE_UPDATED);										// 1
	BIND_CONSTANT(LOCAL_FILE_CHANGE_FILE_DELETED);										// 2
	
	// FILE PATH TYPES //////////////////////////
	BIND_CONSTANT(FILE_PATH_TYPE_INVALID);												// 0
	BIND_CONSTANT(FILE_PATH_TYPE_ABSOLUTE);												// 1
	BIND_CONSTANT(FILE_PATH_TYPE_API_FILENAME);											// 2

	// VR SCREENSHOT TYPE ///////////////////////
	BIND_CONSTANT(VR_SCREENSHOT_TYPE_NONE);												// 0
	BIND_CONSTANT(VR_SCREENSHOT_TYPE_MONO);												// 1
	BIND_CONSTANT(VR_SCREENSHOT_TYPE_STEREO);											// 2
	BIND_CONSTANT(VR_SCREENSHOT_TYPE_MONO_CUBE_MAP);									// 3
	BIND_CONSTANT(VR_SCREENSHOT_TYPE_MONO_PANORAMA);									// 4
	BIND_CONSTANT(VR_SCREENSHOT_TYPE_STEREO_PANORAMA);									// 5

	// ITEM REVIEW TYPE /////////////////////////
	BIND_CONSTANT(ITEM_PREVIEW_TYPE_IMAGE);												// 0
	BIND_CONSTANT(ITEM_PREVIEW_TYPE_YOUTUBE_VIDEO);										// 1
	BIND_CONSTANT(ITEM_PREVIEW_TYPE_SKETCHFAB);											// 2
	BIND_CONSTANT(ITEM_PREVIEW_TYPE_ENVIRONMENTMAP_HORIZONTAL_CROSS);					// 3
	BIND_CONSTANT(ITEM_PREVIEW_TYPE_ENVIRONMENTMAP_LAT_LONG);							// 4
	BIND_CONSTANT(ITEM_PREVIEW_TYPE_RESERVED_MAX);										// 255

	// ITEM STATE ///////////////////////////////
	BIND_CONSTANT(ITEM_STATE_NONE);														// 0
	BIND_CONSTANT(ITEM_STATE_SUBSCRIBED);												// 1
	BIND_CONSTANT(ITEM_STATE_LEGACY_ITEM);												// 2
	BIND_CONSTANT(ITEM_STATE_INSTALLED);												// 4
	BIND_CONSTANT(ITEM_STATE_NEEDS_UPDATE);												// 8
	BIND_CONSTANT(ITEM_STATE_DOWNLOADING);												// 16
	BIND_CONSTANT(ITEM_STATE_DOWNLOAD_PENDING);											// 32

	// ITEM STATISTIC ///////////////////////////
	BIND_CONSTANT(ITEM_STATISTIC_NUM_SUBSCRIPTIONS);									// 0
	BIND_CONSTANT(ITEM_STATISTIC_NUM_FAVORITES);										// 1
	BIND_CONSTANT(ITEM_STATISTIC_NUM_FOLLOWERS);										// 2
	BIND_CONSTANT(ITEM_STATISTIC_NUM_UNIQUE_SUBSCRIPTIONS);								// 3
	BIND_CONSTANT(ITEM_STATISTIC_NUM_UNIQUE_FAVORITES);									// 4
	BIND_CONSTANT(ITEM_STATISTIC_NUM_UNIQUE_FOLLOWERS);									// 5
	BIND_CONSTANT(ITEM_STATISTIC_NUM_UNIQUE_WEBSITE_VIEWS);								// 6
	BIND_CONSTANT(ITEM_STATISTIC_REPORT_SCORE);											// 7
	BIND_CONSTANT(ITEM_STATISTIC_NUM_SECONDS_PLAYED);									// 8
	BIND_CONSTANT(ITEM_STATISTIC_NUM_PLAYTIME_SESSIONS);								// 9
	BIND_CONSTANT(ITEM_STATISTIC_NUM_COMMENTS);											// 10
	BIND_CONSTANT(ITEM_STATISTIC_NUM_SECONDS_PLAYED_DURING_TIME_PERIOD);				// 11
	BIND_CONSTANT(ITEM_STATISTIC_NUM_PLAYTIME_SESSIONS_DURING_TIME_PERIOD);				// 12

	// ITEM UPDATE STATUS ///////////////////////
	BIND_CONSTANT(ITEM_UPDATE_STATUS_INVALID);											// 0
	BIND_CONSTANT(ITEM_UPDATE_STATUS_PREPARING_CONFIG);									// 1
	BIND_CONSTANT(ITEM_UPDATE_STATUS_PREPARING_CONTENT);								// 2
	BIND_CONSTANT(ITEM_UPDATE_STATUS_UPLOADING_CONTENT);								// 3
	BIND_CONSTANT(ITEM_UPDATE_STATUS_UPLOADING_PREVIEW_FILE);							// 4
	BIND_CONSTANT(ITEM_UPDATE_STATUS_COMMITTING_CHANGES);								// 5

	// UGC MATCHING UGC TYPE ////////////////////
	BIND_CONSTANT(UGC_MATCHINGUGCTYPE_ITEMS);											// 0
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_ITEMS_MTX);										// 1
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_ITEMS_READY_TO_USE);							// 2
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_COLLECTIONS);									// 3
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_ARTWORK);										// 4
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_VIDEOS);										// 5
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_SCREENSHOTS);									// 6
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_ALL_GUIDES);									// 7
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_WEB_GUIDES);									// 8
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_INTEGRATED_GUIDES);								// 9
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_USABLE_IN_GAME);								// 10
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_CONTROLLER_BINDINGS);							// 11
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_GAME_MANAGED_ITEMS);							// 12
	BIND_CONSTANT(UGC_MATCHING_UGC_TYPE_ALL);											// ~0

	// UGC QUERY ////////////////////////////////
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_VOTE);												// 0
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_PUBLICATION_DATE);								// 1
	BIND_CONSTANT(UGC_QUERY_ACCEPTED_FOR_GAME_RANKED_BY_ACCEPTANCE_DATE);				// 2
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_TREND);											// 3
	BIND_CONSTANT(UGC_QUERY_FAVORITED_BY_FRIENDS_RANKED_BY_PUBLICATION_DATE);			// 4
	BIND_CONSTANT(UGC_QUERY_CREATED_BY_FRIENDS_RANKED_BY_PUBLICATION_DATE);				// 5
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_NUM_TIMES_REPORTED);								// 6
	BIND_CONSTANT(UGC_QUERY_CREATED_BY_FOLLOWED_USERS_RANKED_BY_PUBLICATION_DATE);		// 7
	BIND_CONSTANT(UGC_QUERY_NOT_YET_RATED);												// 8
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_TOTAL_VOTES_ASC);									// 9
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_VOTES_UP);										// 10
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_TEXT_SEARCH);										// 11
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_TOTAL_UNIQUE_SUBSCRIPTIONS);						// 12
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_PLAYTIME_TREND);									// 13
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_TOTAL_PLAYTIME);									// 14
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_AVERAGE_PLAYTIME_TREND);							// 15
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_LIFETIME_AVERAGE_PLAYTIME);						// 16
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_PLAYTIME_SESSIONS_TREND);							// 17
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_LIFETIME_PLAYTIME_SESSIONS);						// 18
	BIND_CONSTANT(UGC_QUERY_RANKED_BY_LAST_UPDATED_DATE);								// 19

	// USER UGC LIST ////////////////////////////
	BIND_CONSTANT(USER_UGC_LIST_PUBLISHED);												// 0
	BIND_CONSTANT(USER_UGC_LIST_VOTED_ON);												// 1
	BIND_CONSTANT(USER_UGC_LIST_VOTED_UP);												// 2
	BIND_CONSTANT(USER_UGC_LIST_VOTED_DOWN);											// 3
	BIND_CONSTANT(USER_UGC_LIST_FAVORITED);												// 5
	BIND_CONSTANT(USER_UGC_LIST_SUBSCRIBED);											// 6
	BIND_CONSTANT(USER_UGC_LIST_USED_OR_PLAYED);										// 7
	BIND_CONSTANT(USER_UGC_LIST_FOLLOWED);												// 8

	// USER UGC LIST ORT ORDER //////////////////
	BIND_CONSTANT(USERUGCLISTSORTORDER_CREATIONORDERDESC);								// 0
	BIND_CONSTANT(USERUGCLISTSORTORDER_CREATIONORDERASC);								// 1
	BIND_CONSTANT(USERUGCLISTSORTORDER_TITLEASC);										// 2
	BIND_CONSTANT(USERUGCLISTSORTORDER_LASTUPDATEDDESC);								// 3
	BIND_CONSTANT(USERUGCLISTSORTORDER_SUBSCRIPTIONDATEDESC);							// 4
	BIND_CONSTANT(USERUGCLISTSORTORDER_VOTESCOREDESC);									// 5
	BIND_CONSTANT(USERUGCLISTSORTORDER_FORMODERATION);									// 6

	// FAILURE TYPE /////////////////////////////
	BIND_CONSTANT(FAILURE_FLUSHED_CALLBACK_QUEUE);										// 0
	BIND_CONSTANT(FAILURE_PIPE_FAIL);													// 1

	// DURATION CONTROL PROGRESS ////////////////
	BIND_CONSTANT(DURATION_CONTROL_PROGRESS_FULL);										// 0
	BIND_CONSTANT(DURATION_CONTROL_PROGRESS_HALF);										// 1
	BIND_CONSTANT(DURATION_CONTROL_PROGRESS_NONE);										// 2

	// DURATION CONTROL NOTIFICATION ////////////
	BIND_CONSTANT(DURATION_CONTROL_NOTIFICATION_NONE);									// 0
	BIND_CONSTANT(DURATION_CONTROL_NOTIFICATION_1_HOUR);								// 1
	BIND_CONSTANT(DURATION_CONTROL_NOTIFICATION_3_HOURS);								// 3
	BIND_CONSTANT(DURATION_CONTROL_NOTIFICATION_HALF_PROGRESS);							// 3
	BIND_CONSTANT(DURATION_CONTROL_NOTIFICATION_NO_PROGRESS);							// 4

	// LEADERBOARD DATA REQUEST /////////////////
	BIND_CONSTANT(LEADERBOARD_DATA_REQUEST_GLOBAL);										// 0
	BIND_CONSTANT(LEADERBOARD_DATA_REQUEST_GLOBAL_AROUND_USER);							// 1
	BIND_CONSTANT(LEADERBOARD_DATA_REQUEST_FRIENDS);									// 2
	BIND_CONSTANT(LEADERBOARD_DATA_REQUEST_USERS);										// 3

	// LEADERBOARD DISPLAY TYPE /////////////////
	BIND_CONSTANT(LEADERBOARD_DISPLAY_TYPE_NONE);										// 0
	BIND_CONSTANT(LEADERBOARD_DISPLAY_TYPE_NUMERIC);									// 1
	BIND_CONSTANT(LEADERBOARD_DISPLAY_TYPE_TIME_SECONDS);								// 2
	BIND_CONSTANT(LEADERBOARD_DISPLAY_TYPE_TIME_MILLISECONDS);							// 3

	// LEADERBOARD SORT METHOD //////////////////
	BIND_CONSTANT(LEADERBOARD_SORT_METHOD_NONE);										// 0
	BIND_CONSTANT(LEADERBOARD_SORT_METHOD_ASCENDING);									// 1
	BIND_CONSTANT(LEADERBOARD_SORT_METHOD_DESCENDING);									// 2

	// LEADERBOARD UPLOAD SCORE METHOD //////////
	BIND_CONSTANT(LEADERBOARD_UPLOAD_SCORE_METHOD);										// 0
	BIND_CONSTANT(LEADERBOARD_UPLOAD_SCORE_METHOD_KEEP_BEST);							// 1
	BIND_CONSTANT(LEADERBOARD_UPLOAD_SCORE_METHOD_FORCE_UPDATE);						// 2

	// STEAM USER STAT TYPE /////////////////////
	BIND_CONSTANT(STEAM_USER_STAT_TYPE_INVALID);										// 0
	BIND_CONSTANT(STEAM_USER_STAT_TYPE_INT);											// 1
	BIND_CONSTANT(STEAM_USER_STAT_TYPE_FLOAT);											// 2
	BIND_CONSTANT(STEAM_USER_STAT_TYPE_AVGRATE);										// 3
	BIND_CONSTANT(STEAM_USER_STAT_TYPE_ACHIEVEMENTS);									// 4
	BIND_CONSTANT(STEAM_USER_STAT_TYPE_GROUPACHIEVEMENTS);								// 5
	BIND_CONSTANT(STEAM_USER_STAT_TYPE_MAX);

	// CHECK FILE SIGNATURE /////////////////////
	BIND_CONSTANT(CHECK_FILE_SIGNATURE_INVALID_SIGNATURE);								// 0
	BIND_CONSTANT(CHECK_FILE_SIGNATURE_VALID_SIGNATURE);								// 1
	BIND_CONSTANT(CHECK_FILE_SIGNATURE_FILE_NOT_FOUND);									// 2
	BIND_CONSTANT(CHECK_FILE_SIGNATURE_NO_SIGNATURES_FOUND_FOR_THIS_APP);				// 3
	BIND_CONSTANT(CHECK_FILE_SIGNATURE_NO_SIGNATURES_FOUND_FOR_THIS_FILE);				// 4

	// GAMEPAD TEXT INPUT LINE MODE /////////////
	BIND_CONSTANT(GAMEPAD_TEXT_INPUT_LINE_MODE_SINGLE_LINE);							// 0
	BIND_CONSTANT(GAMEPAD_TEXT_INPUT_LINE_MODE_MULTIPLE_LINES);							// 1

	// GAMEPAD TEXT INPUT MODE //////////////////
	BIND_CONSTANT(GAMEPAD_TEXT_INPUT_MODE_NORMAL);										// 0
	BIND_CONSTANT(GAMEPAD_TEXT_INPUT_MODE_PASSWORD);									// 1

	// FLOATING GAMEPAD TEXT INPUT MODE /////////
	BIND_CONSTANT(FLOATING_GAMEPAD_TEXT_INPUT_MODE_SINGLE_LINE);						// 0
	BIND_CONSTANT(FLOATING_GAMEPAD_TEXT_INPUT_MODE_MULTIPLE_LINES);						// 1
	BIND_CONSTANT(FLOATING_GAMEPAD_TEXT_INPUT_MODE_EMAIL);								// 2
	BIND_CONSTANT(FLOATING_GAMEPAD_TEXT_INPUT_MODE_NUMERIC);							// 3

	// STEAM API CALL FAILURE ///////////////////
	BIND_CONSTANT(STEAM_API_CALL_FAILURE_NONE);											// -1
	BIND_CONSTANT(STEAM_API_CALL_FAILURE_STEAM_GONE);									// 0
	BIND_CONSTANT(STEAM_API_CALL_FAILURE_NETWORK_FAILURE);								// 1
	BIND_CONSTANT(STEAM_API_CALL_FAILURE_INVALID_HANDLE);								// 2
	BIND_CONSTANT(STEAM_API_CALL_FAILURE_MISMATCHED_CALLBACK);							// 3

	// TEXT FILTERING CONTEXT ///////////////////
	BIND_CONSTANT(TEXT_FILTERING_CONTEXT_UNKNOWN);										// 0
	BIND_CONSTANT(TEXT_FILTERING_CONTEXT_GAME_CONTENT);									// 1
	BIND_CONSTANT(TEXT_FILTERING_CONTEXT_CHAT);											// 2
	BIND_CONSTANT(TEXT_FILTERING_CONTEXT_NAME);											// 3

	// PROFILE ITEM TYPES
	BIND_CONSTANT(PROFILE_ITEM_TYPE_ANIMATED_AVATAR);									// 0
	BIND_CONSTANT(PROFILE_ITEM_TYPE_AVATAR_FRAME);										// 1
	BIND_CONSTANT(PROFILE_ITEM_TYPE_PROFILE_MODIFIER);									// 2
	BIND_CONSTANT(PROFILE_ITEM_TYPE_PROFILE_BACKGROUND);								// 3
	BIND_CONSTANT(PROFILE_ITEM_TYPE_MINI_PROFILE_BACKGROUND);							// 4
	
	// PROFILE ITEM PROPERTIES
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_IMAGE_SMALL);									// 0
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_IMAGE_LARGE);									// 1
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_INTERNAL_NAME);									// 2
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_TITLE);											// 3
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_DESCRIPTION);									// 4
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_APP_ID);										// 5
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_TYPE_ID);										// 6
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_CLASS);											// 7
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_MOVIE_WEBM);									// 8
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_MOVIE_MP4);										// 9
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_MOVIE_WEBM_SMALL);								// 10
	BIND_CONSTANT(PROFILE_ITEM_PROPERTY_MOVIE_MP4_SMALL);								// 11
		
}

Steam::~Steam(){
	// Store stats then shut down ///////////////
	if(is_init_success){
		SteamUserStats()->StoreStats();
		SteamAPI_Shutdown();
		SteamGameServer_Shutdown();
	}

	// Clear app ID and singleton variables /////
	singleton = NULL;
	current_app_id = 0;
}
