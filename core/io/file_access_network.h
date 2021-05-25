/*************************************************************************/
/*  file_access_network.h                                                */
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

#ifndef FILE_ACCESS_NETWORK_H
#define FILE_ACCESS_NETWORK_H

#include "core/io/stream_peer_tcp.h"
#include "core/os/file_access.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"

class FileAccessNetwork;

class FileAccessNetworkClient {
	struct BlockRequest {
		int32_t id;
		uint64_t offset;
		int32_t size;
	};

	List<BlockRequest> block_requests;

	Semaphore sem;
	Thread thread;
	bool quit = false;
	Mutex mutex;
	Mutex blockrequest_mutex;
	Map<int, FileAccessNetwork *> accesses;
	Ref<StreamPeerTCP> client;
	int32_t last_id = 0;
	int32_t lockcount = 0;

	Vector<uint8_t> block;

	void _thread_func();
	static void _thread_func(void *s);

	void put_32(int32_t p_32);
	void put_64(int64_t p_64);
	int32_t get_32();
	int64_t get_64();
	void lock_mutex();
	void unlock_mutex();

	friend class FileAccessNetwork;
	static FileAccessNetworkClient *singleton;

public:
	static FileAccessNetworkClient *get_singleton() { return singleton; }

	Error connect(const String &p_host, int p_port, const String &p_password = "");

	FileAccessNetworkClient();
	~FileAccessNetworkClient();
};

class FileAccessNetwork : public FileAccess {
	Semaphore sem;
	Semaphore page_sem;
	Mutex buffer_mutex;
	bool opened = false;
	uint64_t total_size;
	mutable uint64_t pos = 0;
	int32_t id;
	mutable bool eof_flag = false;
	mutable int32_t last_page = -1;
	mutable uint8_t *last_page_buff = nullptr;

	int32_t page_size;
	int32_t read_ahead;

	mutable int waiting_on_page = -1;

	struct Page {
		int activity = 0;
		bool queued = false;
		Vector<uint8_t> buffer;
	};

	mutable Vector<Page> pages;

	mutable Error response;

	uint64_t exists_modtime;
	friend class FileAccessNetworkClient;
	void _queue_page(int32_t p_page) const;
	void _respond(uint64_t p_len, Error p_status);
	void _set_block(uint64_t p_offset, const Vector<uint8_t> &p_block);

public:
	enum Command {
		COMMAND_OPEN_FILE,
		COMMAND_READ_BLOCK,
		COMMAND_CLOSE,
		COMMAND_FILE_EXISTS,
		COMMAND_GET_MODTIME,
	};

	enum Response {
		RESPONSE_OPEN,
		RESPONSE_DATA,
		RESPONSE_FILE_EXISTS,
		RESPONSE_GET_MODTIME,
	};

	virtual Error _open(const String &p_path, int p_mode_flags); ///< open a file
	virtual void close(); ///< close a file
	virtual bool is_open() const; ///< true when file is open

	virtual void seek(uint64_t p_position); ///< seek to a given position
	virtual void seek_end(int64_t p_position = 0); ///< seek from the end of file
	virtual uint64_t get_position() const; ///< get position in the file
	virtual uint64_t get_length() const; ///< get size of the file

	virtual bool eof_reached() const; ///< reading passed EOF

	virtual uint8_t get_8() const; ///< get a byte
	virtual uint64_t get_buffer(uint8_t *p_dst, uint64_t p_length) const;

	virtual Error get_error() const; ///< get last error

	virtual void flush();
	virtual void store_8(uint8_t p_dest); ///< store a byte

	virtual bool file_exists(const String &p_path); ///< return true if a file exists

	virtual uint64_t _get_modified_time(const String &p_file);
	virtual uint32_t _get_unix_permissions(const String &p_file);
	virtual Error _set_unix_permissions(const String &p_file, uint32_t p_permissions);

	static void configure();

	FileAccessNetwork();
	~FileAccessNetwork();
};

#endif // FILE_ACCESS_NETWORK_H
