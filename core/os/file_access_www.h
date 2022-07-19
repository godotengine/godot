/*************************************************************************/
/*  file_access_unix.h                                                   */
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

#ifndef FILE_ACCESS_WWW_H
#define FILE_ACCESS_WWW_H

#include "core/os/file_access.h"
#include "core/os/memory.h"
#include "core/io/http_client.h"
#include "core/os/thread.h"
#include "core/os/semaphore.h"
#include "core/ring_buffer.h"

typedef void (*CloseNotificationFunc)(const String &p_file, int p_flags);

class FileAccessWww;

class FileAccessWwwClient { // 内部缓存类

	Semaphore sem;
	mutable HTTPClient *hc; // Http
	mutable String url; // 域名 或ip地址
	mutable int port; // 端口消息
	mutable String request_string; //url 请求地址
	mutable bool p_ssl = false; // is or not https
	mutable String path_src;

	RingBuffer<uint8_t> fileAccessInfo;

	unsigned long posStart;
	
	Thread thread;
	bool quit;
	Mutex mutex;
	bool isfileAccessInfo;


	void _thread_func();
	static void _thread_func(void *s);

	int lockcount;
	void lock_mutex();
	void unlock_mutex();
	Error http_request();
	friend class FileAccessWww;
	static FileAccessWwwClient *singleton;

public:
	mutable int total_size = 0; // 网络文件大小
	mutable int p_lenght = 512 * 1024;
	mutable int pos = 0; //文件 range 开始位置

	static FileAccessWwwClient *get_singleton() { return singleton; }

	int poll_back(uint8_t *p_buf, int pos,int p_size);
	void seek(unsigned long p_position); ///< seek to a given position
	Error connect(const String &p_path);

	FileAccessWwwClient();
	~FileAccessWwwClient();
};

class FileAccessWww : public FileAccess {

	Semaphore sem;
	int flags;
	void check_errors() const;
	mutable Error last_error;

	mutable int total_size = 0;
	mutable int pos = 0;
	Mutex buffer_mutex;
	String path_src;

	static FileAccess *create_libc();

public:
	static CloseNotificationFunc close_notification_func;

	virtual Error _open(const String &p_path, int p_mode_flags); ///< open a file
	virtual void close(); ///< close a file
	virtual bool is_open() const; ///< true when file is open
	virtual String get_path() const; /// returns the path for the current open file
	virtual String get_path_absolute() const; /// returns the absolute path for the current open file

	virtual void seek(size_t p_position); ///< seek to a given position
	virtual void seek_end(int64_t p_position = 0); ///< seek from the end of file
	virtual size_t get_position() const; ///< get position in the file
	virtual size_t get_len() const; ///< get size of the file

	virtual bool eof_reached() const; ///< reading passed EOF

	virtual uint8_t get_8() const; ///< get a byte
	virtual int get_buffer(uint8_t *p_dst, int p_length) const ;

	virtual Error get_error() const; ///< get last error

	virtual void flush();
	virtual void store_8(uint8_t p_dest); ///< store a byte
	virtual void store_buffer(const uint8_t *p_src, int p_length); ///< store an array of bytes

	virtual bool file_exists(const String &p_path); ///< return true if a file exists


	virtual uint64_t _get_modified_time(const String &p_file);
	virtual uint32_t _get_unix_permissions(const String &p_file);
	virtual Error _set_unix_permissions(const String &p_file, uint32_t p_permissions);

	FileAccessWww();
	virtual ~FileAccessWww();
};

#endif