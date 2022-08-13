/*************************************************************************/
/*  script_language_internals.h                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SCRIPT_LANGUAGE_INTERNALS_H
#define SCRIPT_LANGUAGE_INTERNALS_H

#include "core/object/script_language.h"

class ScriptLanguageInternals {
public:
	static ScriptLanguageThreadContext::DebugThreadID create_thread_id(const ScriptLanguage &p_language, uint64_t p_thread_id) {
		PackedByteArray thread_id;
		const uint8_t *raw = reinterpret_cast<const uint8_t *>(&p_thread_id);
		thread_id.resize(sizeof(Thread::ID) + 1);
		memcpy(thread_id.ptrw(), raw, sizeof(Thread::ID));
		static_assert(ScriptServer::MAX_LANGUAGES <= 256);
		thread_id.set(sizeof(Thread::ID), static_cast<uint8_t>(p_language.get_language_index()));
		// return Variant(thread_id);
		Variant value;
		value = thread_id;
		return value;
	}
};

// Initialization behavior shared by language-specific contexts and extension contexts
class ThreadContextRefBase {
	bool _guard_context_creation = false;

protected:
	bool _detect_infinite_loop() {
		if (_guard_context_creation) {
			// Infinite loop during context creation, happens when creation errors
			// are printed while debugging, which calls this function again.  Must
			// exit without printing.
			GENERATE_TRAP();
			return true;
		}
		_guard_context_creation = true;
		return false;
	}

	void _end_detect_infinite_loop() {
		_guard_context_creation = false;
	}

public:
	// called from thread local storage initializer
	ThreadContextRefBase() = default;
	~ThreadContextRefBase() = default;

	ThreadContextRefBase(const ThreadContextRefBase &) = delete;
	ThreadContextRefBase(const ThreadContextRefBase &&) = delete;
	ThreadContextRefBase &operator=(const ThreadContextRefBase &) = delete;
	ThreadContextRefBase &operator=(const ThreadContextRefBase &&) = delete;
};

template <class FACTORY, class CONTEXT>
class ThreadContextRef : public ThreadContextRefBase {
	Ref<CONTEXT> _context;

public:
	CONTEXT &context() {
		if (_detect_infinite_loop()) {
			// Intentionally crash below, as we cannot continue.
			_context.unref();
		} else {
			if (!_context.is_valid()) {
				_context = Ref<CONTEXT>(FACTORY::create_thread_context());
			}
		}

		_end_detect_infinite_loop();

		return *_context.ptr();
	}

	// To be called by main thread on language exit; because thread local storage for the
	// main thread is not cleared until after Godot checks for memory leaks.
	void free_context() {
		_context.unref();
	}
};

#endif // SCRIPT_LANGUAGE_INTERNALS_H
