/**************************************************************************/
/*  register_types.cpp                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                            BLAZIUM ENGINE                              */
/*                        https://blazium.app                             */
/**************************************************************************/
/* Copyright (c) 2024-present Blazium Engine contributors.                */
/* Copyright (c) 2024 Dragos Daian, Randolph William Aarseth II.          */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "register_types.h"
#include "blazium_client.h"
#include "lobby/lobby_client.h"

void initialize_blazium_sdk_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_ABSTRACT_CLASS(BlaziumClient);
		GDREGISTER_CLASS(LobbyClient);
		GDREGISTER_CLASS(LobbyClient::LobbyInfo);
		GDREGISTER_CLASS(LobbyClient::LobbyPeer);
		GDREGISTER_CLASS(LobbyClient::CreateLobbyResponse::CreateLobbyResult);
		GDREGISTER_CLASS(LobbyClient::CreateLobbyResponse);
		GDREGISTER_CLASS(LobbyClient::LobbyResponse::LobbyResult);
		GDREGISTER_CLASS(LobbyClient::LobbyResponse);
		GDREGISTER_CLASS(LobbyClient::ListLobbyResponse::ListLobbyResult);
		GDREGISTER_CLASS(LobbyClient::ListLobbyResponse);
		GDREGISTER_CLASS(LobbyClient::ViewLobbyResponse::ViewLobbyResult);
		GDREGISTER_CLASS(LobbyClient::ViewLobbyResponse);
	}
}

void uninitialize_blazium_sdk_module(ModuleInitializationLevel p_level) {
}
