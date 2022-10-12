# GodotSteam for Godot Engine
An open-source and fully functional Steamworks SDK / API module and plug-in for the Godot Game Engine (version 3.x). For the Windows, Linux, and Mac platforms. 

Additional flavors include:
- [Godot 2.x](https://github.com/Gramps/GodotSteam/tree/godot2)
- [Godot 4.x](https://github.com/Gramps/GodotSteam/tree/godot4)
- [GDNative](https://github.com/Gramps/GodotSteam/tree/gdnative)

Documentation
----------
[Documentation is available here](https://gramps.github.io/GodotSteam/) and [is mirrored on and exported from CoaguCo's site](https://coaguco.com/godotsteam).

You can also check out the Search Help section inside Godot Engine after compiling it with GodotSteam.

Feel free to chat with us about GodotSteam on the [CoaguCo Discord server](https://discord.gg/SJRSq6K).

Current Build
----------
You can [download pre-compiled versions _(currently v3.14)_ of this repo here](https://github.com/Gramps/GodotSteam/releases).

**Version 3.16.1 Changes**
- Fixed: issues with getPSNID and getStadiaID functions when compiling on Linux

**Version 3.16 Changes**
- Added: new enums for Community Profile item types and properties in Friends class
- Added: new functions hasEquippedProfileItem, getProfileItemPropertyString, and getProfileItemPropertyInt in Friends class
- Added: new callbacks/signals _equipped_profile_items_changed_ and _equipped_profile_items_ in Friends class
- Added: new networking identity types
- Added: new functions setXboxPairwiseID, getXboxPairwiseID, setPSNID, getPSNID, setStadiaID, and getStadiaID to Networking Types class
- Changed: minor correction to createListenSocketP2P in attempt to fix possible crash

**Version 3.15 Changes**
- Changed: sendMessageToConnection and sendMessages now take PoolByteArrays to send any data
- Fixed: issue with receiving messages, now allows more than one at a time; _thanks to Frostings_
- Fixed: getQueryUGCChildren not working correctly; _thanks to EIREXE_

**Version 3.14 Changes**
- Added: inventory handle argument to various Inventory class functions, defaults to 0 to use internally store argument
- Changed: various Inventory class functions to send back the new inventory handle as well as storing it internally
- Fixed: various string issues; _thanks to Green Fox_
- Fixed: _file_read_async_complete_ call result not sending back the file buffer
- Fixed: missing variant type for _avatar_loaded_ signal
- Fixed: _enumerate_following_list_ calling the wrong signal name
- Fixed: print of Steamworks error didn't contain signal name
- Fixed: some variable and argument names
- Fixed: deserializeResult to accept incoming PoolByteArray buffer
- Fixed: various message functions in new networking classes; _thanks to Avantir-Chaosfire_

**Version 3.13.3 Changes**
- Fixed: get correct size of lobby message in sendLobbyChatMsg; _thanks to Green Fox_

**Version 3.13.2 Changes**
- Fixed: various functions and callbacks that sent back scrambled IP addresses

**Version 3.13.1 Changes**
- Changed: all HTML Surface functions can now have the handle passed to them or not; will use internal handle if not passed
- Changed: all HTML Surface callbacks now send back their browser handles, if applicable
- Changed: fileWrite and fileWriteAsync now allow you to pass size or not; will determine if not passed
- Fixed: fileWrite and fileWriteAsync passing wrong byte array size

**Version 3.13 Changes**
- Added: missing function getPlaybackStatus to Music class
- Added: missing function setDurationControlOnlineState to Users class
- Added: missing signals for Matchmaking Servers
- Added: missing PropertyInfo data for signals
- Changed: serverInit now takes the individual arguments and no longer a dictionary of arguments
- Changed: getAppName, getAppListInstallDir, and getAppListBuildId in App Lists to use uint32_t instead of uint32
- Changed: initGameServer to use correct arguments
- Changed: all signal / callback names for Game Server class to lower-case to match the all others
- Changed: server_connect_failure, policy_response, client_group_status callback to match function names
- Changed: various variables in Game Server class callbacks to match the others
- Changed: setMaxPlayerCount argument to players_max from max to be more clear
- Changed: setPasswordProtected argument to password_protected from password to be more clear
- Changed: call result / signal stat_received to stats_received
- Changed: createCookieContainer now sends back the cookie_handle
- Changed: checkResultSteamID changed argument name to match
- Changed: getItemsWithPrices return dictionary name
- Changed: getAppID now returns uint32_t
- Changed: getFavoriteGames to have more distinct port names in return dictionary
- Changed: some returned types and argument types to better match their Steamworks counterparts
- Changed: names of some keys and some integer types in getQueryUGCResult return dictionary
- Changed: keys in getBeaconDetails return dictionary to be more clear
- Changed: removed data_size argument from various Remote Storage functions and get size internally
- Changed: playerDetails and serverRules IP argument to a string
- Changed: various Networking Messages, Networking Sockets, and Networking Utils functions to use internal struct system with Networking Type functions
- Changed: a variety of miscellaneous small changes and corrections
- Fixed: some missing function binds
- Fixed: lobby_message callback data, thanks to _kongo555_
- Fixed: missing default value for getAvailableP2PPacketSize, readP2PPacket, sendP2PPacket
- Fixed: getAnalogActionData so the return dictionary has the right keys
- Fixed: getUserSteamFriends, getUserSteamGroups to give the correct Steam ID back
- Fixed: getFriendGamePlayed using wrong key name in return dictionary
- Fixed: toIdentityString to provide the correct string data
- Fixed: parseIdentityString to properly parse back the string data
- Fixed: getSesssionConnectionInfo now passes back all data
- Fixed: getLocalPingLocation should return both the ping and location ID in a dictionary
- Fixed: getPingToDataCenter, getPOPList, parsePingLocationString, closeConnection, getAuthenticationStatus, getConnectionInfo, createSocketPair functions
- Removed: requestAllProofOfPurchaseKeys and requestAppProofOfPurchaseKey as they are depreciated
- Removed: gameplay_stats signal from Game Server class as it wasn't connected to anything
- Removed: getUserDataFolder as it is depreciated
- Removed: leading _ in front of callbacks and call results internally
- Removed: initGameServer as it is unnecessary
- Removed: connectByIPAddress, isPingMeasurementInProgress, setLinkedLobby as they are not in the SDK

**Version 3.12.1 Changes**
- Fixed: incorrect case on app_installed and app_uninstalled, thanks to _craftablescience_

**Version 3.12 Changes**
- Added: missing D_METHOD to all functions, should show the right argument names in-editor
- Added: Input origin enums for PS5 and Steam Deck
- Added: Input Types, Input Glyph Style, Input Glyph Size, and Input Configuration Enable Type enums
- Added: getConnectionRealTimeStatus, configureConnectionLanes, connectP2PCustomSignaling, receivedP2PCustomSignal, getCertificateRequest, setCertificate, resetIdentity, runNetworkingCallbacks, beginAsyncRequestFakeIP, getFakeIP, createListenScoketP2PFakeIP, getRemoveFakeIPForConnection, and createFakeUDPPort functions and callback to NetworkingSockets class
- Added: dismissFloatingGamepadTextInput function to Utils class
- Added: setTimeCreatedDateRange and setTimeUpdatedDateRange to UGC class
- Added: NetworkingeDebugOutputType enums for NetworkingUtils
- Added: missing constant binds for Server API, OverlayToWebPageMode
- Fixed: minor compiler warnings
- Fixed: empty file hash being returned by file_details_result callback
- Fixed: a variety of small bugs and possible crashes, _thanks to qarmin_
- Fixed: missing binds for getFriendsGroupName, getFriendsGroupMembersList, getFriendsGroupIDByIndex, getFriendsGroupCount, getFriendMessage, getFriendCoplayTime, getFriendCoplayGame, getCoplayFriendCount, getCoplayFriend, getClanTag, getClanName, getClanCount, getClanChatMessage, getClanByIndex, getClanActivityCounts, fileWriteAsync, fileWriteStreamCancel, fileWriteStreamClose, fileWriteStreamOpen, fileWriteStreamWriteChunk, getCachedUGCCount, getUGCDownloadProgress, getUGCDetails, fileReadAsync, getOPFSettings, getOPFStringForApp, getVideoURL, isBroadcasting functions
- Fixed: setPNGIcon and updateCurrentEntryCoverArt in Music Remote class
- Fixed: missing getUGCDetails and getUGCDownloadProgress functions
- Changed: updated doc_class file for in-editor documentation
- Changed: updated to Steamworks 1.53
- Changed: lobby_data_update, removed lobby data queries as they should be done manually
- Changed: minor tweaks under-the-hood
- Changed: various generic 'int' to their actual types
- Changed: renamed servers and server stats to game server and game server stats respectively, to match SDK
- Changed: SteamNetworkingQuickConnectionStatus to SteamNetConnectionRealTimeStatus_t per Steamworks SDK 1.53, causes a break in previous GodotSteam versions
- Changed: getConfigValueInfo, removed name and next value from return dictionary as they are no longer passed by function in SDK 1.53
- Changed: rearranged functions in godotsteam.cpp class binds to match godotsteam.h order
- Changed: enum constant binds to match godotsteam.h enum order
- Removed: unused callback new_launch_query_parameters, broadcast_upload_start, broadcast_upload_stop
- Removed: allocateMessage as it shouldn't be used solo
- Removed: getQuickConnectionStatus and getFirstConfigValue as they were removed from SDK 1.53
- Removed: setDebugOutputFunction from Networking Utils

**Version 3.11.1 Changes**
- Removed: unused structs

**Version 3.11 Changes**
- Added: server branch merged into master
- Changed: spacing in default arguments in godotsteam.h
- Changed: renamed STEAM_GAMESERVER_CALLBACK as STEAM_CALLBACK
- Removed: SteamGameServer_RunCallbacks function

Known Issues
----------
- **Using MinGW causes crashes.** I strongly recommend you **do not use MinGW** to compile at this time.
- As of Steamworks SDK 1.53, you cannot compile with previous version of GodotSteam (3.11.1 or earlier) due to a code change in the SDK.
  - Using Steamworks SDK 1.53 or newer, you must use GodotSteam 3.12 or newer.
  - Using Steamworks SDK 1.53 or earlier, you must use GodotSteam 3.11.1 or earlier.

Quick How-To
----------
- Download this repository and unpack it.
- Download and unpack the [Steamworks SDK 1.53](https://partner.steamgames.com); this requires a Steam developer account.
  - Please see "Known Issues" above about versions.
- Download and unpack the [Godot source 3.x](https://github.com/godotengine/godot).
- Move the following to godotsteam/sdk/ folder:
````
    sdk/public/
    sdk/redistributable_bin/
````
- The repo's directory contents should now look like this:
````
    godotsteam/sdk/public/*
    godotsteam/sdk/redistributable_bin/*
    godotsteam/SCsub
    godotsteam/config.py
    godotsteam/godotsteam.cpp
    godotsteam/godotsteam.h
    godotsteam/register_types.cpp
    godotsteam/register_types.h
````
- Now move the "godotsteam" directory into the "modules" directory of the unpacked Godot Engine source.
  - You can also just put the godotsteam directory where ever you like and just apply the ````custom_modules=.../path/to/dir/godotsteam```` flag in SCONS when compiling.  Make sure the ````custom_modules=```` flag points to where godotsteam is located.
- Recompile for your platform:
  - **NOTE:** use SCONS flags ````production=yes tools=yes target=release_debug```` for your editor and ````production=yes tools=no target=release```` for your templates.
  - Windows ( http://docs.godotengine.org/en/stable/reference/compiling_for_windows.html )
  - Linux ( http://docs.godotengine.org/en/stable/reference/compiling_for_x11.html )
    - If using Ubuntu 16.10 or higher and having issues with PIE security in GCC, use LINKFLAGS='-no-pie' to get an executable instead of a shared library.
  - MacOS ( http://docs.godotengine.org/en/stable/reference/compiling_for_osx.html )
    - When creating templates for this, please refer to this post for assistance as the documentation is a bit lacking ( http://steamcommunity.com/app/404790/discussions/0/364042703865087202/ ).
- When recompiling the engine is finished do the following before running it the first time:
  - Copy the shared library (steam_api), for your OS, from sdk/redistributable_bin/ folders to the Godot binary location (by default in the godot source /bin/ file but you can move them to a new folder).
  - Create a steam_appid.txt file with your game's app ID or 480 in this folder.  Necessary if the editor or game is run _outside_ of Steam.

- The finished hierarchy should look like this (if you downloaded the pre-compiles, the editor files go in place of these tools files, which are the same thing):
  - Linux 32/64-bit
  ```
  libsteam_api.so
  steam_appid.txt
  ./godot.linux.tools.32 or ./godot.linux.tools.64
  ```
  - MacOS
  ```
  libsteam_api.dylib
  steam_appid.txt
  ./godot.osx.tools.32 or ./godot.osx.tools.64
  ```
  - Windows 32-bit
  ```
  steam_api.dll
  steam_appid.txt
  ./godot.windows.tools.32.exe
  ```
  - Windows 64-bit
  ```
  steam_api64.dll
  steam_appid.txt
  ./godot.windows.tools.64.exe
  ```
- Lack of the Steam API DLL/SO/DyLib (for your respective OS) or the steam_appid.txt will cause it fail and crash when testing or running the game outside of the Steam client.
  - **NOTE:** For MacOS, the libsteam_api.dylib and steam_appid.txt must be in the Content/MacOS/ folder in your application zip or the game will crash.
  - **NOTE:** For Linux, you may have to load the overlay library for Steam overlay to work:
  ```
  export LD_PRELOAD=~/.local/share/Steam/ubuntu12_32/gameoverlayrenderer.so;~/.local/share/Steam/ubuntu12_64/gameoverlayrenderer.so
  
  or 
  
  export LD_PRELOAD=~/.local/share/Steam/ubuntu12_32/gameoverlayrenderer.so;
  export LD_PRELOAD=~/.local/share/Steam/ubuntu12_64/gameoverlayrenderer.so;
  ```
  This can be done in an .sh file that runs these before running your executable.  This issue may not arise for all users and can also just be done by the user in a terminal separately.

From here you should be able to call various functions of Steamworks. You should be able to look up the functions in Godot itself under the search section. In addition, you should be able to read the Steamworks API documentation to see what all is available and cross-reference with GodotSteam.

- When uploading your game to Steam, you *must* upload your game's executable and Steam API DLL/SO/DyLIB (steam_api.dll, steam_api64.dll, libsteam_api.dylib, and/or libsteam_api.so).  *Do not* include the steam_appid.txt or any .lib files as they are unnecessary.

Donate
-------------
Pull-requests are the best way to help the project out but you can also donate through [Patreon](https://patreon.com/coaguco) or [Paypal](https://www.paypal.me/sithlordkyle)!

License
-------------
MIT license
