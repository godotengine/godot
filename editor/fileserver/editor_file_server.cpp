/*************************************************************************/
/*  editor_file_server.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "editor_file_server.h"

#include "../editor_settings.h"
#include "core/io/marshalls.h"

//#define DEBUG_PRINT(m_p) print_line(m_p)
//#define DEBUG_TIME(m_what) printf("MS: %s - %lu\n", m_what, OS::get_singleton()->get_ticks_usec());

#define DEBUG_PRINT(m_what)
#define DEBUG_TIME(m_what)

void EditorFileServer::_close_client(ClientData *cd) {
	cd->connection->disconnect_from_host();
	{
		MutexLock lock(cd->efs->wait_mutex);
		cd->efs->to_wait.insert(cd->thread);
	}
	while (cd->files.size()) {
		memdelete(cd->files.front()->get());
		cd->files.erase(cd->files.front());
	}
	memdelete(cd);
}

void EditorFileServer::_subthread_start(void *s) {
	ClientData *cd = (ClientData *)s;

	cd->connection->set_no_delay(true);
	uint8_t buf4[8];
	Error err = cd->connection->get_data(buf4, 4);
	if (err != OK) {
		_close_client(cd);
		ERR_FAIL_COND(err != OK);
	}

	int passlen = decode_uint32(buf4);

	if (passlen > 512) {
		_close_client(cd);
		ERR_FAIL_COND(passlen > 512);
	} else if (passlen > 0) {
		Vector<char> passutf8;
		passutf8.resize(passlen + 1);
		err = cd->connection->get_data((uint8_t *)passutf8.ptr(), passlen);
		if (err != OK) {
			_close_client(cd);
			ERR_FAIL_COND(err != OK);
		}
		passutf8.write[passlen] = 0;
		String s2;
		s2.parse_utf8(passutf8.ptr());
		if (s2 != cd->efs->password) {
			encode_uint32(ERR_INVALID_DATA, buf4);
			cd->connection->put_data(buf4, 4);
			OS::get_singleton()->delay_usec(1000000);
			_close_client(cd);
			ERR_PRINT("CLIENT PASSWORD MISMATCH");
			ERR_FAIL();
		}
	} else {
		if (cd->efs->password != "") {
			encode_uint32(ERR_INVALID_DATA, buf4);
			cd->connection->put_data(buf4, 4);
			OS::get_singleton()->delay_usec(1000000);
			_close_client(cd);
			ERR_PRINT("CLIENT PASSWORD MISMATCH (should be empty!)");
			ERR_FAIL();
		}
	}

	encode_uint32(OK, buf4);
	cd->connection->put_data(buf4, 4);

	while (!cd->quit) {
		//wait for ID
		err = cd->connection->get_data(buf4, 4);
		DEBUG_TIME("get_data")

		if (err != OK) {
			_close_client(cd);
			ERR_FAIL_COND(err != OK);
		}
		int id = decode_uint32(buf4);

		//wait for command
		err = cd->connection->get_data(buf4, 4);
		if (err != OK) {
			_close_client(cd);
			ERR_FAIL_COND(err != OK);
		}
		int cmd = decode_uint32(buf4);

		switch (cmd) {
			case FileAccessNetwork::COMMAND_FILE_EXISTS:
			case FileAccessNetwork::COMMAND_GET_MODTIME:
			case FileAccessNetwork::COMMAND_OPEN_FILE: {
				DEBUG_TIME("open_file")
				err = cd->connection->get_data(buf4, 4);
				if (err != OK) {
					_close_client(cd);
					ERR_FAIL_COND(err != OK);
				}

				int namelen = decode_uint32(buf4);
				Vector<char> fileutf8;
				fileutf8.resize(namelen + 1);
				err = cd->connection->get_data((uint8_t *)fileutf8.ptr(), namelen);
				if (err != OK) {
					_close_client(cd);
					ERR_FAIL_COND(err != OK);
				}
				fileutf8.write[namelen] = 0;
				String s2;
				s2.parse_utf8(fileutf8.ptr());

				if (cmd == FileAccessNetwork::COMMAND_FILE_EXISTS) {
					print_verbose("FILE EXISTS: " + s2);
				}
				if (cmd == FileAccessNetwork::COMMAND_GET_MODTIME) {
					print_verbose("MOD TIME: " + s2);
				}
				if (cmd == FileAccessNetwork::COMMAND_OPEN_FILE) {
					print_verbose("OPEN: " + s2);
				}

				if (!s2.begins_with("res://")) {
					_close_client(cd);
					ERR_FAIL_COND(!s2.begins_with("res://"));
				}
				ERR_CONTINUE(cd->files.has(id));

				if (cmd == FileAccessNetwork::COMMAND_FILE_EXISTS) {
					encode_uint32(id, buf4);
					cd->connection->put_data(buf4, 4);
					encode_uint32(FileAccessNetwork::RESPONSE_FILE_EXISTS, buf4);
					cd->connection->put_data(buf4, 4);
					encode_uint32(FileAccess::exists(s2), buf4);
					cd->connection->put_data(buf4, 4);
					DEBUG_TIME("open_file_end")
					break;
				}

				if (cmd == FileAccessNetwork::COMMAND_GET_MODTIME) {
					encode_uint32(id, buf4);
					cd->connection->put_data(buf4, 4);
					encode_uint32(FileAccessNetwork::RESPONSE_GET_MODTIME, buf4);
					cd->connection->put_data(buf4, 4);
					encode_uint64(FileAccess::get_modified_time(s2), buf4);
					cd->connection->put_data(buf4, 8);
					DEBUG_TIME("open_file_end")
					break;
				}

				FileAccess *fa = FileAccess::open(s2, FileAccess::READ);
				if (!fa) {
					//not found, continue
					encode_uint32(id, buf4);
					cd->connection->put_data(buf4, 4);
					encode_uint32(FileAccessNetwork::RESPONSE_OPEN, buf4);
					cd->connection->put_data(buf4, 4);
					encode_uint32(ERR_FILE_NOT_FOUND, buf4);
					cd->connection->put_data(buf4, 4);
					DEBUG_TIME("open_file_end")
					break;
				}

				encode_uint32(id, buf4);
				cd->connection->put_data(buf4, 4);
				encode_uint32(FileAccessNetwork::RESPONSE_OPEN, buf4);
				cd->connection->put_data(buf4, 4);
				encode_uint32(OK, buf4);
				cd->connection->put_data(buf4, 4);
				encode_uint64(fa->get_len(), buf4);
				cd->connection->put_data(buf4, 8);

				cd->files[id] = fa;
				DEBUG_TIME("open_file_end")

			} break;
			case FileAccessNetwork::COMMAND_READ_BLOCK: {
				err = cd->connection->get_data(buf4, 8);
				if (err != OK) {
					_close_client(cd);
					ERR_FAIL_COND(err != OK);
				}

				ERR_CONTINUE(!cd->files.has(id));

				uint64_t offset = decode_uint64(buf4);

				err = cd->connection->get_data(buf4, 4);
				if (err != OK) {
					_close_client(cd);
					ERR_FAIL_COND(err != OK);
				}

				int blocklen = decode_uint32(buf4);
				ERR_CONTINUE(blocklen > (16 * 1024 * 1024));

				cd->files[id]->seek(offset);
				Vector<uint8_t> buf;
				buf.resize(blocklen);
				int read = cd->files[id]->get_buffer(buf.ptrw(), blocklen);
				ERR_CONTINUE(read < 0);

				print_verbose("GET BLOCK - offset: " + itos(offset) + ", blocklen: " + itos(blocklen));

				//not found, continue
				encode_uint32(id, buf4);
				cd->connection->put_data(buf4, 4);
				encode_uint32(FileAccessNetwork::RESPONSE_DATA, buf4);
				cd->connection->put_data(buf4, 4);
				encode_uint64(offset, buf4);
				cd->connection->put_data(buf4, 8);
				encode_uint32(read, buf4);
				cd->connection->put_data(buf4, 4);
				cd->connection->put_data(buf.ptr(), read);

			} break;
			case FileAccessNetwork::COMMAND_CLOSE: {
				print_verbose("CLOSED");
				ERR_CONTINUE(!cd->files.has(id));
				memdelete(cd->files[id]);
				cd->files.erase(id);
			} break;
		}
	}

	_close_client(cd);
}

void EditorFileServer::_thread_start(void *s) {
	EditorFileServer *self = (EditorFileServer *)s;
	while (!self->quit) {
		if (self->cmd == CMD_ACTIVATE) {
			self->server->listen(self->port);
			self->active = true;
			self->cmd = CMD_NONE;
		} else if (self->cmd == CMD_STOP) {
			self->server->stop();
			self->active = false;
			self->cmd = CMD_NONE;
		}

		if (self->active) {
			if (self->server->is_connection_available()) {
				ClientData *cd = memnew(ClientData);
				cd->connection = self->server->take_connection();
				cd->efs = self;
				cd->quit = false;
				cd->thread = Thread::create(_subthread_start, cd);
			}
		}

		self->wait_mutex.lock();
		while (self->to_wait.size()) {
			Thread *w = self->to_wait.front()->get();
			self->to_wait.erase(w);
			self->wait_mutex.unlock();
			Thread::wait_to_finish(w);
			memdelete(w);
			self->wait_mutex.lock();
		}
		self->wait_mutex.unlock();

		OS::get_singleton()->delay_usec(100000);
	}
}

void EditorFileServer::start() {
	stop();
	port = EDITOR_DEF("filesystem/file_server/port", 6010);
	password = EDITOR_DEF("filesystem/file_server/password", "");
	cmd = CMD_ACTIVATE;
}

bool EditorFileServer::is_active() const {
	return active;
}

void EditorFileServer::stop() {
	cmd = CMD_STOP;
}

EditorFileServer::EditorFileServer() {
	server.instance();
	quit = false;
	active = false;
	cmd = CMD_NONE;
	thread = Thread::create(_thread_start, this);

	EDITOR_DEF("filesystem/file_server/port", 6010);
	EDITOR_DEF("filesystem/file_server/password", "");
}

EditorFileServer::~EditorFileServer() {
	quit = true;
	Thread::wait_to_finish(thread);
	memdelete(thread);
}
