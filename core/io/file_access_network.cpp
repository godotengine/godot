/*************************************************************************/
/*  file_access_network.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "file_access_network.h"

#include "core/config/project_settings.h"
#include "core/io/ip.h"
#include "core/io/marshalls.h"
#include "core/os/os.h"

//#define DEBUG_PRINT(m_p) print_line(m_p)
//#define DEBUG_TIME(m_what) printf("MS: %s - %lli\n",m_what,OS::get_singleton()->get_ticks_usec());
#define DEBUG_PRINT(m_p)
#define DEBUG_TIME(m_what)

void FileAccessNetworkClient::lock_mutex() {
	mutex.lock();
	lockcount++;
}

void FileAccessNetworkClient::unlock_mutex() {
	lockcount--;
	mutex.unlock();
}

void FileAccessNetworkClient::put_32(int p_32) {
	uint8_t buf[4];
	encode_uint32(p_32, buf);
	client->put_data(buf, 4);
	DEBUG_PRINT("put32: " + itos(p_32));
}

void FileAccessNetworkClient::put_64(int64_t p_64) {
	uint8_t buf[8];
	encode_uint64(p_64, buf);
	client->put_data(buf, 8);
	DEBUG_PRINT("put64: " + itos(p_64));
}

int FileAccessNetworkClient::get_32() {
	uint8_t buf[4];
	client->get_data(buf, 4);
	return decode_uint32(buf);
}

int64_t FileAccessNetworkClient::get_64() {
	uint8_t buf[8];
	client->get_data(buf, 8);
	return decode_uint64(buf);
}

void FileAccessNetworkClient::_thread_func() {
	client->set_no_delay(true);
	while (!quit) {
		DEBUG_PRINT("SEM WAIT - " + itos(sem->get()));
		sem.wait();
		DEBUG_TIME("sem_unlock");
		//DEBUG_PRINT("semwait returned "+itos(werr));
		DEBUG_PRINT("MUTEX LOCK " + itos(lockcount));
		lock_mutex();
		DEBUG_PRINT("MUTEX PASS");

		{
			MutexLock lock(blockrequest_mutex);
			while (block_requests.size()) {
				put_32(block_requests.front()->get().id);
				put_32(FileAccessNetwork::COMMAND_READ_BLOCK);
				put_64(block_requests.front()->get().offset);
				put_32(block_requests.front()->get().size);
				block_requests.pop_front();
			}
		}

		DEBUG_PRINT("THREAD ITER");

		DEBUG_TIME("sem_read");
		int id = get_32();

		int response = get_32();
		DEBUG_PRINT("GET RESPONSE: " + itos(response));

		FileAccessNetwork *fa = nullptr;

		if (response != FileAccessNetwork::RESPONSE_DATA) {
			if (!accesses.has(id)) {
				unlock_mutex();
				ERR_FAIL_COND(!accesses.has(id));
			}
		}

		if (accesses.has(id)) {
			fa = accesses[id];
		}

		switch (response) {
			case FileAccessNetwork::RESPONSE_OPEN: {
				DEBUG_TIME("sem_open");
				int status = get_32();
				if (status != OK) {
					fa->_respond(0, Error(status));
				} else {
					int64_t len = get_64();
					fa->_respond(len, Error(status));
				}

				fa->sem.post();

			} break;
			case FileAccessNetwork::RESPONSE_DATA: {
				int64_t offset = get_64();
				int32_t len = get_32();

				Vector<uint8_t> block;
				block.resize(len);
				client->get_data(block.ptrw(), len);

				if (fa) { //may have been queued
					fa->_set_block(offset, block);
				}

			} break;
			case FileAccessNetwork::RESPONSE_FILE_EXISTS: {
				int status = get_32();
				fa->exists_modtime = status != 0;
				fa->sem.post();

			} break;
			case FileAccessNetwork::RESPONSE_GET_MODTIME: {
				uint64_t status = get_64();
				fa->exists_modtime = status;
				fa->sem.post();

			} break;
		}

		unlock_mutex();
	}
}

void FileAccessNetworkClient::_thread_func(void *s) {
	FileAccessNetworkClient *self = (FileAccessNetworkClient *)s;

	self->_thread_func();
}

Error FileAccessNetworkClient::connect(const String &p_host, int p_port, const String &p_password) {
	IPAddress ip;

	if (p_host.is_valid_ip_address()) {
		ip = p_host;
	} else {
		ip = IP::get_singleton()->resolve_hostname(p_host);
	}

	DEBUG_PRINT("IP: " + String(ip) + " port " + itos(p_port));
	Error err = client->connect_to_host(ip, p_port);
	ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot connect to host with IP: " + String(ip) + " and port: " + itos(p_port));
	while (client->get_status() == StreamPeerTCP::STATUS_CONNECTING) {
		//DEBUG_PRINT("trying to connect....");
		OS::get_singleton()->delay_usec(1000);
	}

	if (client->get_status() != StreamPeerTCP::STATUS_CONNECTED) {
		return ERR_CANT_CONNECT;
	}

	CharString cs = p_password.utf8();
	put_32(cs.length());
	client->put_data((const uint8_t *)cs.ptr(), cs.length());

	int e = get_32();

	if (e != OK) {
		return ERR_INVALID_PARAMETER;
	}

	thread.start(_thread_func, this);

	return OK;
}

FileAccessNetworkClient *FileAccessNetworkClient::singleton = nullptr;

FileAccessNetworkClient::FileAccessNetworkClient() {
	singleton = this;
	client.instance();
}

FileAccessNetworkClient::~FileAccessNetworkClient() {
	quit = true;
	sem.post();
	thread.wait_to_finish();
}

void FileAccessNetwork::_set_block(uint64_t p_offset, const Vector<uint8_t> &p_block) {
	int32_t page = p_offset / page_size;
	ERR_FAIL_INDEX(page, pages.size());
	if (page < pages.size() - 1) {
		ERR_FAIL_COND(p_block.size() != page_size);
	} else {
		ERR_FAIL_COND((uint64_t)p_block.size() != total_size % page_size);
	}

	{
		MutexLock lock(buffer_mutex);
		pages.write[page].buffer = p_block;
		pages.write[page].queued = false;
	}

	if (waiting_on_page == page) {
		waiting_on_page = -1;
		page_sem.post();
	}
}

void FileAccessNetwork::_respond(uint64_t p_len, Error p_status) {
	DEBUG_PRINT("GOT RESPONSE - len: " + itos(p_len) + " status: " + itos(p_status));
	response = p_status;
	if (response != OK) {
		return;
	}
	opened = true;
	total_size = p_len;
	int32_t pc = ((total_size - 1) / page_size) + 1;
	pages.resize(pc);
}

Error FileAccessNetwork::_open(const String &p_path, int p_mode_flags) {
	ERR_FAIL_COND_V(p_mode_flags != READ, ERR_UNAVAILABLE);
	if (opened) {
		close();
	}
	FileAccessNetworkClient *nc = FileAccessNetworkClient::singleton;
	DEBUG_PRINT("open: " + p_path);

	DEBUG_TIME("open_begin");

	nc->lock_mutex();
	nc->put_32(id);
	nc->accesses[id] = this;
	nc->put_32(COMMAND_OPEN_FILE);
	CharString cs = p_path.utf8();
	nc->put_32(cs.length());
	nc->client->put_data((const uint8_t *)cs.ptr(), cs.length());
	pos = 0;
	eof_flag = false;
	last_page = -1;
	last_page_buff = nullptr;

	//buffers.clear();
	nc->unlock_mutex();
	DEBUG_PRINT("OPEN POST");
	DEBUG_TIME("open_post");
	nc->sem.post(); //awaiting answer
	DEBUG_PRINT("WAIT...");
	sem.wait();
	DEBUG_TIME("open_end");
	DEBUG_PRINT("WAIT ENDED...");

	return response;
}

void FileAccessNetwork::close() {
	if (!opened) {
		return;
	}

	FileAccessNetworkClient *nc = FileAccessNetworkClient::singleton;

	DEBUG_PRINT("CLOSE");
	nc->lock_mutex();
	nc->put_32(id);
	nc->put_32(COMMAND_CLOSE);
	pages.clear();
	opened = false;
	nc->unlock_mutex();
}

bool FileAccessNetwork::is_open() const {
	return opened;
}

void FileAccessNetwork::seek(uint64_t p_position) {
	ERR_FAIL_COND_MSG(!opened, "File must be opened before use.");

	eof_flag = p_position > total_size;

	if (p_position >= total_size) {
		p_position = total_size;
	}

	pos = p_position;
}

void FileAccessNetwork::seek_end(int64_t p_position) {
	seek(total_size + p_position);
}

uint64_t FileAccessNetwork::get_position() const {
	ERR_FAIL_COND_V_MSG(!opened, 0, "File must be opened before use.");
	return pos;
}

uint64_t FileAccessNetwork::get_length() const {
	ERR_FAIL_COND_V_MSG(!opened, 0, "File must be opened before use.");
	return total_size;
}

bool FileAccessNetwork::eof_reached() const {
	ERR_FAIL_COND_V_MSG(!opened, false, "File must be opened before use.");
	return eof_flag;
}

uint8_t FileAccessNetwork::get_8() const {
	uint8_t v;
	get_buffer(&v, 1);
	return v;
}

void FileAccessNetwork::_queue_page(int32_t p_page) const {
	if (p_page >= pages.size()) {
		return;
	}
	if (pages[p_page].buffer.is_empty() && !pages[p_page].queued) {
		FileAccessNetworkClient *nc = FileAccessNetworkClient::singleton;
		{
			MutexLock lock(nc->blockrequest_mutex);

			FileAccessNetworkClient::BlockRequest br;
			br.id = id;
			br.offset = (uint64_t)p_page * page_size;
			br.size = page_size;
			nc->block_requests.push_back(br);
			pages.write[p_page].queued = true;
		}
		DEBUG_PRINT("QUEUE PAGE POST");
		nc->sem.post();
		DEBUG_PRINT("queued " + itos(p_page));
	}
}

uint64_t FileAccessNetwork::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);

	if (pos + p_length > total_size) {
		eof_flag = true;
	}
	if (pos + p_length >= total_size) {
		p_length = total_size - pos;
	}

	uint8_t *buff = last_page_buff;

	for (uint64_t i = 0; i < p_length; i++) {
		int32_t page = pos / page_size;

		if (page != last_page) {
			buffer_mutex.lock();
			if (pages[page].buffer.is_empty()) {
				waiting_on_page = page;
				for (int32_t j = 0; j < read_ahead; j++) {
					_queue_page(page + j);
				}
				buffer_mutex.unlock();
				DEBUG_PRINT("wait");
				page_sem.wait();
				DEBUG_PRINT("done");
			} else {
				for (int32_t j = 0; j < read_ahead; j++) {
					_queue_page(page + j);
				}
				buffer_mutex.unlock();
			}

			buff = pages.write[page].buffer.ptrw();
			last_page_buff = buff;
			last_page = page;
		}

		p_dst[i] = buff[pos - uint64_t(page) * page_size];
		pos++;
	}

	return p_length;
}

Error FileAccessNetwork::get_error() const {
	return pos == total_size ? ERR_FILE_EOF : OK;
}

void FileAccessNetwork::flush() {
	ERR_FAIL();
}

void FileAccessNetwork::store_8(uint8_t p_dest) {
	ERR_FAIL();
}

bool FileAccessNetwork::file_exists(const String &p_path) {
	FileAccessNetworkClient *nc = FileAccessNetworkClient::singleton;
	nc->lock_mutex();
	nc->put_32(id);
	nc->put_32(COMMAND_FILE_EXISTS);
	CharString cs = p_path.utf8();
	nc->put_32(cs.length());
	nc->client->put_data((const uint8_t *)cs.ptr(), cs.length());
	nc->unlock_mutex();
	DEBUG_PRINT("FILE EXISTS POST");
	nc->sem.post();
	sem.wait();

	return exists_modtime != 0;
}

uint64_t FileAccessNetwork::_get_modified_time(const String &p_file) {
	FileAccessNetworkClient *nc = FileAccessNetworkClient::singleton;
	nc->lock_mutex();
	nc->put_32(id);
	nc->put_32(COMMAND_GET_MODTIME);
	CharString cs = p_file.utf8();
	nc->put_32(cs.length());
	nc->client->put_data((const uint8_t *)cs.ptr(), cs.length());
	nc->unlock_mutex();
	DEBUG_PRINT("MODTIME POST");
	nc->sem.post();
	sem.wait();

	return exists_modtime;
}

uint32_t FileAccessNetwork::_get_unix_permissions(const String &p_file) {
	ERR_PRINT("Getting UNIX permissions from network drives is not implemented yet");
	return 0;
}

Error FileAccessNetwork::_set_unix_permissions(const String &p_file, uint32_t p_permissions) {
	ERR_PRINT("Setting UNIX permissions on network drives is not implemented yet");
	return ERR_UNAVAILABLE;
}

void FileAccessNetwork::configure() {
	GLOBAL_DEF("network/remote_fs/page_size", 65536);
	ProjectSettings::get_singleton()->set_custom_property_info("network/remote_fs/page_size", PropertyInfo(Variant::INT, "network/remote_fs/page_size", PROPERTY_HINT_RANGE, "1,65536,1,or_greater")); //is used as denominator and can't be zero
	GLOBAL_DEF("network/remote_fs/page_read_ahead", 4);
	ProjectSettings::get_singleton()->set_custom_property_info("network/remote_fs/page_read_ahead", PropertyInfo(Variant::INT, "network/remote_fs/page_read_ahead", PROPERTY_HINT_RANGE, "0,8,1,or_greater"));
}

FileAccessNetwork::FileAccessNetwork() {
	FileAccessNetworkClient *nc = FileAccessNetworkClient::singleton;
	nc->lock_mutex();
	id = nc->last_id++;
	nc->accesses[id] = this;
	nc->unlock_mutex();
	page_size = GLOBAL_GET("network/remote_fs/page_size");
	read_ahead = GLOBAL_GET("network/remote_fs/page_read_ahead");
}

FileAccessNetwork::~FileAccessNetwork() {
	close();

	FileAccessNetworkClient *nc = FileAccessNetworkClient::singleton;
	nc->lock_mutex();
	id = nc->last_id++;
	nc->accesses.erase(id);
	nc->unlock_mutex();
}
