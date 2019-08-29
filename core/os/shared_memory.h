/*************************************************************************/
/*  shared_memory.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef SHARED_MEMORY
#define SHARED_MEMORY

#include "core/error_list.h"
#include "core/int_types.h"

class String;

/*
	This class allows processes to share a memory region.

	Implementations don't provide locking. Clients must be careful to avoid race
	conditions:
	- The order in which processes try to create/open the shared memory block must
	  be well defined.
	- Only one process should be running between begin_access() and end_access() at
	  a given time.
	- The underlying shared memory block must be assumed to be destroyed when any of
	  the clients calls close(); therefore it's asumed it's undefined when that will
	  happen exactly.
	  Measures must be taken to avoid clients using the shared memory object beyond the
	  point one of them have done so.
	  In other words, when a certain client has called close(), the others have no
	  option but doing the same.

	By design, get_size()/set_size() can only be used between begin_access() and
	end_access(), to accomodate the API to what's doable across different platforms.
	Both functions don't necessarily share the same constraints at the implementation
	level, but restricting anything related to size the same way makes the API cleaner.

	Also, bear in mind that resizing the shared memory block may be a heavy operation
	on some platforms, so it's better to use this class for a sparingly resized block
	or either overallocate and resize in big increments as needed.
*/

class SharedMemory {

protected:
	static constexpr const char *ERR_STR_ALREADY_OPEN = "Already open.";
	static constexpr const char *ERR_STR_CANNOT_CREATE_OR_OPEN = "Cannot create/open.";
	static constexpr const char *ERR_STR_NOT_OPEN = "Not open.";
	static constexpr const char *ERR_STR_SIZE_NOT_AVAILABLE = "Size can only be get/set between begin_access() and end_access().";
	static constexpr const char *WARN_STR_CLOSING_BEFORE_END_ACCESS = "Closing before end_access().";
	static constexpr const char *WARN_STR_BEGIN_ACCESS_WHILE_ALREADY = "begin_access() called while access already initiated.";
	static constexpr const char *WARN_STR_END_ACCESS_WITHOUT_BEGIN_ACCESS = "end_access() with no previous begin_access().";

	static SharedMemory *(*create_func)(const String &p_name);

public:
	// begin_access() will return this if the initial size hasn't yet been set.
	static constexpr void *UNSIZED = &create_func; // Using any well-known address

	static SharedMemory *create(const String &p_name);

	virtual Error open() = 0;
	virtual void close() = 0;
	virtual bool is_open() = 0;

	virtual uint8_t *begin_access() = 0;
	virtual void end_access() = 0;

	// Returns the possibly new address, invalidating the one returned by begin_access().
	// On error, returns NULL and the state is the same as if end_access() had been called.
	// The implementation is not guaranteed to have the size of the underlying object matching
	// the size set, but at least it will be as big a requested, so observed behavior will be
	// the expected.
	virtual uint8_t *set_size(int64_t p_size) = 0;
	virtual int64_t get_size() = 0;

	virtual ~SharedMemory(){};
};

#endif
