/**************************************************************************/
/*  lobby_client.h                                                        */
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

#ifndef LOBBY_CLIENT_H
#define LOBBY_CLIENT_H

#include "../blazium_client.h"
#include "core/io/json.h"
#include "modules/websocket/websocket_peer.h"

class LobbyClient : public BlaziumClient {
	GDCLASS(LobbyClient, BlaziumClient);
	enum CommandType {
		CREATE_LOBBY = 0,
		SIMPLE_REQUEST,
		LOBBY_VIEW,
		LOBBY_LIST
	};
	String server_url = "wss://lobby.blazium.app/connect";

public:
	class CreateLobbyResponse : public RefCounted {
		GDCLASS(CreateLobbyResponse, RefCounted);

	protected:
		static void _bind_methods() {
			ADD_SIGNAL(MethodInfo("finished", PropertyInfo(Variant::OBJECT, "result", PROPERTY_HINT_RESOURCE_TYPE, "CreateLobbyResult")));
		}

	public:
		class CreateLobbyResult : public RefCounted {
			GDCLASS(CreateLobbyResult, RefCounted);
			String error;
			String lobby_name;

		protected:
			static void _bind_methods() {
				ClassDB::bind_method(D_METHOD("has_error"), &CreateLobbyResult::has_error);
				ClassDB::bind_method(D_METHOD("get_error"), &CreateLobbyResult::get_error);
				ClassDB::bind_method(D_METHOD("get_lobby_name"), &CreateLobbyResult::get_lobby_name);
				ADD_PROPERTY(PropertyInfo(Variant::STRING, "error"), "", "get_error");
				ADD_PROPERTY(PropertyInfo(Variant::STRING, "lobby_name"), "", "get_lobby_name");
			}

		public:
			void set_error(String p_error) { this->error = p_error; }
			void set_lobby_name(String p_lobby_name) { this->lobby_name = p_lobby_name; }

			bool has_error() const { return !error.is_empty(); }
			String get_error() const { return error; }
			String get_lobby_name() const { return lobby_name; }
			CreateLobbyResult(const CreateLobbyResult &other) :
					error(other.error), lobby_name(other.lobby_name) {}
			CreateLobbyResult() {}
		};
		CreateLobbyResponse(const CreateLobbyResponse &other) {}
		CreateLobbyResponse() {}
	};
	class LobbyResponse : public RefCounted {
		GDCLASS(LobbyResponse, RefCounted);

	protected:
		static void _bind_methods() {
			ADD_SIGNAL(MethodInfo("finished", PropertyInfo(Variant::OBJECT, "result", PROPERTY_HINT_RESOURCE_TYPE, "LobbyResult")));
		}

	public:
		class LobbyResult : public RefCounted {
			GDCLASS(LobbyResult, RefCounted);

			String error;

		protected:
			static void _bind_methods() {
				ClassDB::bind_method(D_METHOD("has_error"), &LobbyResult::has_error);
				ClassDB::bind_method(D_METHOD("get_error"), &LobbyResult::get_error);
				ADD_PROPERTY(PropertyInfo(Variant::STRING, "error"), "", "get_error");
			}

		public:
			void set_error(String p_error) { this->error = p_error; }

			bool has_error() const { return !error.is_empty(); }
			String get_error() const { return error; }
			LobbyResult(const LobbyResult &other) :
					error(other.error) {}
			LobbyResult() {}
		};
		LobbyResponse(const LobbyResponse &other) {}
		LobbyResponse() {}
	};
	class ListLobbyResponse : public RefCounted {
		GDCLASS(ListLobbyResponse, RefCounted);

	protected:
		static void _bind_methods() {
			ADD_SIGNAL(MethodInfo("finished", PropertyInfo(Variant::OBJECT, "result", PROPERTY_HINT_RESOURCE_TYPE, "ListLobbyResult")));
		}

	public:
		class ListLobbyResult : public RefCounted {
			GDCLASS(ListLobbyResult, RefCounted);

			String error;
			TypedArray<String> lobbies;

		protected:
			static void _bind_methods() {
				ClassDB::bind_method(D_METHOD("has_error"), &ListLobbyResult::has_error);
				ClassDB::bind_method(D_METHOD("get_error"), &ListLobbyResult::get_error);
				ClassDB::bind_method(D_METHOD("get_lobbies"), &ListLobbyResult::get_lobbies);
				ADD_PROPERTY(PropertyInfo(Variant::STRING, "error"), "", "get_error");
				ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "lobbies"), "", "get_lobbies");
			}

		public:
			void set_error(String p_error) { this->error = p_error; }
			void set_lobbies(TypedArray<String> p_lobbies) { this->lobbies = p_lobbies; }

			bool has_error() const { return !error.is_empty(); }
			String get_error() const { return error; }
			TypedArray<String> get_lobbies() const { return lobbies; }
			ListLobbyResult(const ListLobbyResult &other) :
					error(other.error), lobbies(other.lobbies) {}
			ListLobbyResult() {}
		};
		ListLobbyResponse(const ListLobbyResponse &other) {}
		ListLobbyResponse() {}
	};
	class LobbyInfo : public RefCounted {
		GDCLASS(LobbyInfo, RefCounted);
		String host;
		int max_players = 0;
		bool sealed = false;

	protected:
		static void _bind_methods() {
			ClassDB::bind_method(D_METHOD("get_host"), &LobbyInfo::get_host);
			ClassDB::bind_method(D_METHOD("get_max_players"), &LobbyInfo::get_max_players);
			ClassDB::bind_method(D_METHOD("is_sealed"), &LobbyInfo::is_sealed);
			ADD_PROPERTY(PropertyInfo(Variant::STRING, "host"), "", "get_host");
			ADD_PROPERTY(PropertyInfo(Variant::INT, "max_players"), "", "get_max_players");
			ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sealed"), "", "is_sealed");
		}

	public:
		void set_host(String p_host) { this->host = p_host; }
		void set_max_players(int p_max_players) { this->max_players = p_max_players; }
		void set_sealed(bool p_sealed) { this->sealed = p_sealed; }

		String get_host() const { return host; }
		int get_max_players() const { return max_players; }
		bool is_sealed() const { return sealed; }
		LobbyInfo(const LobbyInfo &other) :
				host(other.host), max_players(other.max_players), sealed(other.sealed) {}
		LobbyInfo() {}
	};
	class LobbyPeer : public RefCounted {
		GDCLASS(LobbyPeer, RefCounted);
		String id;
		String name;
		bool ready = false;

	protected:
		static void _bind_methods() {
			ClassDB::bind_method(D_METHOD("get_id"), &LobbyPeer::get_id);
			ClassDB::bind_method(D_METHOD("get_name"), &LobbyPeer::get_name);
			ClassDB::bind_method(D_METHOD("is_ready"), &LobbyPeer::is_ready);
			ADD_PROPERTY(PropertyInfo(Variant::STRING, "id"), "", "get_id");
			ADD_PROPERTY(PropertyInfo(Variant::STRING, "name"), "", "get_name");
			ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ready"), "", "is_ready");
		}

	public:
		void set_id(String p_id) { this->id = p_id; }
		void set_name(String p_name) { this->name = p_name; }
		void set_ready(bool p_ready) { this->ready = p_ready; }

		String get_id() const { return id; }
		String get_name() const { return name; }
		bool is_ready() const { return ready; }
		LobbyPeer(const LobbyPeer &other) :
				id(other.id), name(other.name), ready(other.ready) {}
		LobbyPeer() {}
	};
	class ViewLobbyResponse : public RefCounted {
		GDCLASS(ViewLobbyResponse, RefCounted);

	protected:
		static void _bind_methods() {
			ADD_SIGNAL(MethodInfo("finished", PropertyInfo(Variant::OBJECT, "result", PROPERTY_HINT_RESOURCE_TYPE, "ViewLobbyResult")));
		}

	public:
		class ViewLobbyResult : public RefCounted {
			GDCLASS(ViewLobbyResult, RefCounted);
			String error;
			TypedArray<LobbyPeer> peers;
			Ref<LobbyInfo> lobby_info;

		protected:
			static void _bind_methods() {
				ClassDB::bind_method(D_METHOD("has_error"), &ViewLobbyResult::has_error);
				ClassDB::bind_method(D_METHOD("get_error"), &ViewLobbyResult::get_error);
				ClassDB::bind_method(D_METHOD("get_peers"), &ViewLobbyResult::get_peers);
				ClassDB::bind_method(D_METHOD("get_lobby_info"), &ViewLobbyResult::get_lobby_info);
				ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "peers"), "", "get_peers");
				ADD_PROPERTY(PropertyInfo(Variant::STRING, "error"), "", "get_error");
			}

		public:
			void set_error(String p_error) { this->error = p_error; }
			void set_peers(TypedArray<LobbyPeer> p_peers) { this->peers = p_peers; }
			void set_lobby_info(Ref<LobbyInfo> p_lobby_info) { this->lobby_info = p_lobby_info; }

			bool has_error() const { return !error.is_empty(); }
			String get_error() const { return error; }
			TypedArray<LobbyPeer> get_peers() const { return peers; }
			Ref<LobbyInfo> get_lobby_info() const { return lobby_info; }
			ViewLobbyResult() {
				lobby_info.instantiate();
			}
			~ViewLobbyResult() {
			}
		};
		ViewLobbyResponse(const ViewLobbyResponse &other) {}
		ViewLobbyResponse() {}
	};

private:
	Ref<WebSocketPeer> _socket;
	int _counter = 0;
	Dictionary _commands;

	String _get_data_from_dict(const Dictionary &dict, const String &key);
	void _receive_data(const Dictionary &data);
	void _send_data(const Dictionary &data);
	void _wait_ready();
	String _increment_counter();

protected:
	void _notification(int p_notification);
	static void _bind_methods();

public:
	bool connect_to_lobby(const String &game_id);
	void set_server_url(String p_server_url) { this->server_url = p_server_url; }
	String get_server_url() { return server_url; }
	Ref<CreateLobbyResponse> create_lobby(int max_players, const String &password);
	Ref<LobbyResponse> join_lobby(const String &lobby_name, const String &password);
	Ref<LobbyResponse> leave_lobby();
	Ref<ListLobbyResponse> list_lobby(int start, int count);
	Ref<ViewLobbyResponse> view_lobby(const String &lobby_name, const String &password);
	Ref<LobbyResponse> kick_peer(const String &peer_id);
	Ref<LobbyResponse> lobby_ready();
	Ref<LobbyResponse> lobby_unready();
	Ref<LobbyResponse> set_peer_name(const String &peer_name);
	Ref<LobbyResponse> seal_lobby();
	Ref<LobbyResponse> unseal_lobby();
	Ref<LobbyResponse> lobby_data(const String &peer_data);
	Ref<LobbyResponse> lobby_data_to(const String &peer_data, const String &target_peer);

	LobbyClient();
	~LobbyClient();
};

#endif // LOBBY_CLIENT_H
