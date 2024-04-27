/**************************************************************************/
/*  logger.cpp                                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
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

#include "logger.h"

#include "core/config/project_settings.h"
#include "core/core_globals.h"
#include "core/io/dir_access.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "core/os/time.h"
#include "core/string/print_string.h"

#if defined(MINGW_ENABLED) || defined(_MSC_VER)
#define sprintf sprintf_s
#endif

bool Logger::should_log(bool p_err) {
	return (!p_err || CoreGlobals::print_error_enabled) && (p_err || CoreGlobals::print_line_enabled);
}

bool Logger::_flush_stdout_on_print = true;

void Logger::set_flush_stdout_on_print(bool value) {
	_flush_stdout_on_print = value;
}

void Logger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type) {
	if (!should_log(true)) {
		return;
	}

	const char *err_type = "ERROR";
	switch (p_type) {
		case ERR_ERROR:
			err_type = "ERROR";
			break;
		case ERR_WARNING:
			err_type = "WARNING";
			break;
		case ERR_SCRIPT:
			err_type = "SCRIPT ERROR";
			break;
		case ERR_SHADER:
			err_type = "SHADER ERROR";
			break;
		default:
			ERR_PRINT("Unknown error type");
			break;
	}

	const char *err_details;
	if (p_rationale && *p_rationale) {
		err_details = p_rationale;
	} else {
		err_details = p_code;
	}

	if (p_editor_notify) {
		logf_error("%s: %s\n", err_type, err_details);
	} else {
		logf_error("USER %s: %s\n", err_type, err_details);
	}
	logf_error("   at: %s (%s:%i)\n", p_function, p_file, p_line);
}

void Logger::logf(const char *p_format, ...) {
	if (!should_log(false)) {
		return;
	}

	va_list argp;
	va_start(argp, p_format);

	logv(p_format, argp, false);

	va_end(argp);
}

void Logger::logf_error(const char *p_format, ...) {
	if (!should_log(true)) {
		return;
	}

	va_list argp;
	va_start(argp, p_format);

	logv(p_format, argp, true);

	va_end(argp);
}

void RotatedFileLogger::clear_old_backups() {
	int max_backups = max_files - 1; // -1 for the current file

	String basename = base_path.get_file().get_basename();
	String extension = base_path.get_extension();

	Ref<DirAccess> da = DirAccess::open(base_path.get_base_dir());
	if (da.is_null()) {
		return;
	}

	da->list_dir_begin();
	String f = da->get_next();
	HashSet<String> backups;
	while (!f.is_empty()) {
		if (!da->current_is_dir() && f.begins_with(basename) && f.get_extension() == extension && f != base_path.get_file()) {
			backups.insert(f);
		}
		f = da->get_next();
	}
	da->list_dir_end();

	if (backups.size() > (uint32_t)max_backups) {
		// since backups are appended with timestamp and Set iterates them in sorted order,
		// first backups are the oldest
		int to_delete = backups.size() - max_backups;
		for (HashSet<String>::Iterator E = backups.begin(); E && to_delete > 0; ++E, --to_delete) {
			da->remove(*E);
		}
	}
}

void RotatedFileLogger::rotate_file() {
	file.unref();

	if (FileAccess::exists(base_path)) {
		if (max_files > 1) {
			String timestamp = Time::get_singleton()->get_datetime_string_from_system().replace(":", ".");
			String backup_name = base_path.get_basename() + timestamp;
			if (!base_path.get_extension().is_empty()) {
				backup_name += "." + base_path.get_extension();
			}

			Ref<DirAccess> da = DirAccess::open(base_path.get_base_dir());
			if (da.is_valid()) {
				da->copy(base_path, backup_name);
			}
			clear_old_backups();
		}
	} else {
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_USERDATA);
		if (da.is_valid()) {
			da->make_dir_recursive(base_path.get_base_dir());
		}
	}

	file = FileAccess::open(base_path, FileAccess::WRITE);
	file->detach_from_objectdb(); // Note: This FileAccess instance will exist longer than ObjectDB, therefore can't be registered in ObjectDB.
}

RotatedFileLogger::RotatedFileLogger(const String &p_base_path, int p_max_files) :
		base_path(p_base_path.simplify_path()),
		max_files(p_max_files > 0 ? p_max_files : 1) {
	rotate_file();
}

void RotatedFileLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!should_log(p_err)) {
		return;
	}

	if (file.is_valid()) {
		const int static_buf_size = 512;
		char static_buf[static_buf_size];
		char *buf = static_buf;
		va_list list_copy;
		va_copy(list_copy, p_list);
		int len = vsnprintf(buf, static_buf_size, p_format, p_list);
		if (len >= static_buf_size) {
			buf = (char *)Memory::alloc_static(len + 1);
			vsnprintf(buf, len + 1, p_format, list_copy);
		}
		va_end(list_copy);
		file->store_buffer((uint8_t *)buf, len);

		if (len >= static_buf_size) {
			Memory::free_static(buf);
		}

		if (p_err || _flush_stdout_on_print) {
			// Don't always flush when printing stdout to avoid performance
			// issues when `print()` is spammed in release builds.
			file->flush();
		}
	}
}

void StdLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!should_log(p_err)) {
		return;
	}

	if (p_err) {
		vfprintf(stderr, p_format, p_list);
	} else {
		vprintf(p_format, p_list);
		if (_flush_stdout_on_print) {
			// Don't always flush when printing stdout to avoid performance
			// issues when `print()` is spammed in release builds.
			fflush(stdout);
		}
	}
}

CompositeLogger::CompositeLogger(const Vector<Logger *> &p_loggers) :
		loggers(p_loggers) {
}

void CompositeLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!should_log(p_err)) {
		return;
	}

	for (int i = 0; i < loggers.size(); ++i) {
		va_list list_copy;
		va_copy(list_copy, p_list);
		loggers[i]->logv(p_format, list_copy, p_err);
		va_end(list_copy);
	}
}

void CompositeLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type) {
	if (!should_log(true)) {
		return;
	}

	for (int i = 0; i < loggers.size(); ++i) {
		loggers[i]->log_error(p_function, p_file, p_line, p_code, p_rationale, p_editor_notify, p_type);
	}
}

void CompositeLogger::add_logger(Logger *p_logger) {
	loggers.push_back(p_logger);
}

CompositeLogger::~CompositeLogger() {
	for (int i = 0; i < loggers.size(); ++i) {
		memdelete(loggers[i]);
	}
}

////// UserLogManagerLogger //////
// This is the internal user log hooking system, which does all the hard work.

UserLogManagerLogger *UserLogManagerLogger::singleton = nullptr;

UserLogManagerLogger::UserLogManagerLogger() {
	ERR_FAIL_COND_MSG(singleton != nullptr, "Somehow created two UserLogManagerLoggers.");

	singleton = this;

	pre_buffering.set();

	// this is about to get overwritten by `recalculate_state`
	state.set(STATE_OFF);

	// we don't technically have to lock the mutex here but I'd rather preserve the formal recalculate_state mandatory-mutex invariant
	MutexLock lock(mutex);
	recalculate_state();
}

UserLogManagerLogger::~UserLogManagerLogger() {
	ERR_FAIL_COND_MSG(singleton != this, "UserLogManagerLogger::singleton not correct on exit.");

	singleton = nullptr;

	// there's no messages still in-flight, right?
	// right?
}

void UserLogManagerLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (state.get() == STATE_OFF) {
		// don't jump through the formatting hoops, just drop the message
		// if there's a hook active (or we're in the buffering zone), `state` will never transition through STATE_OFF
		// if at any point there isn't a hook active, dropping messages is valid
		return;
	}

	va_list list_copy;
	va_copy(list_copy, p_list);

	int len = vsnprintf(nullptr, 0, p_format, p_list);

	char *buf = (char *)Memory::alloc_static(len + 1);
	vsnprintf(buf, len + 1, p_format, list_copy);
	va_end(list_copy);

	Dictionary message;
	message["text"] = buf;
	message["type"] = "info";

	process(message);

	Memory::free_static(buf);
}

void UserLogManagerLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type) {
	if (state.get() == STATE_OFF) {
		// don't jump through the formatting hoops, just drop the message
		// if there's a hook active (or we're in the buffering zone), `state` will never transition through STATE_OFF
		// if at any point there isn't a hook active, dropping messages is valid
		return;
	}

	Dictionary message;
	message["function"] = p_function;
	message["file"] = p_file;
	message["line"] = p_line;
	message["text"] = p_code;
	message["rationale"] = p_rationale;
	switch (p_type) {
		case ERR_ERROR:
			message["type"] = "error";
			break;
		case ERR_WARNING:
			message["type"] = "warning";
			break;
		case ERR_SCRIPT:
			message["type"] = "script";
			break;
		case ERR_SHADER:
			message["type"] = "shader";
			break;

		default:
			// this is an error but I don't want to start spamming error messages *in the error handler*
			// that way lies madness and infinite loops/stack overflows
			message["type"] = "unknown";
			break;
	}

	process(message);
}

void UserLogManagerLogger::register_log_capture_non_thread_safe(const Callable &p_callable) {
	// It gets *extremely* hard to guarantee the proper semantics if you're allowed to call this from other threads.
	ERR_FAIL_COND_MSG(!Thread::is_main_thread(), "This call is forbidden outside the main thread.");

	if (pre_buffering.is_set()) {
		// Time to dispatch our messages! This catches this particular hook up to "realtime", replaying all buffered messages in fast-forward.

		// We can be certain nobody is *removing* things from the buffer right now
		// Nothing is ever removed from the buffer except for frame transitions and state changes
		// Frame transitions are locked to the main thread, and so are we
		// and Frame 0 is guaranteed to be STATE_BUFFERED

		// Adding things to the buffer is fine, we'll just loop until they're done; this is why we're not using an iterator but rather an index
		int index_to_send = 0;

		while (true) {
			Dictionary to_send;
			{
				// We do still need to lock the mutex while grabbing the Dictionary, though, just in case the buffer is being reallocated at this moment
				MutexLock lock(mutex);

				if (index_to_send >= buffered_logs.size()) {
					// Never mind that whole "send a message" plan!

					// We've reached the end of the log, and because it's locked,
					// we can be sure that no new messages will be added at this exact moment.
					// Add ourselves to the captures so we'll intercept future messages.
					register_callable(captures_non_thread_safe, p_callable);

					// This does mean that "the buffer" and "the non_thread_safe callables" are, in some cases, updated atomically
					// This is important; see process()

					// conceptually recalculate_state() gets called here
					// but in practice we don't because we know we're on frame 0 and therefore no change of state will be happening

					// We're done!
					break;
				}

				// We haven't reached the end; grab another message
				to_send = buffered_logs[index_to_send++];

				// Unlock the mutex to avoid deadlocks in case our dispatch adds/removes callbacks or logs messages
			}

			// Off you go, while no locks are held!
			dispatch_message(to_send, p_callable);
		}
	} else {
		// it's not Frame 0, therefore we don't have to mess around with the buffer at all
		// just lock, register ourselves as a callable, and recalculate the state so we can start capturing messages
		MutexLock lock(mutex);
		register_callable(captures_non_thread_safe, p_callable);
		recalculate_state();
	}
}

void UserLogManagerLogger::unregister_log_capture_non_thread_safe(const Callable &p_callable) {
	// It gets *extremely* hard to guarantee the proper semantics if you're allowed to call this from other threads.
	ERR_FAIL_COND_MSG(!Thread::is_main_thread(), "This call is forbidden outside the main thread.");

	MutexLock lock(mutex);
	unregister_callable(captures_non_thread_safe, p_callable);
	recalculate_state();
}

void UserLogManagerLogger::register_log_capture_buffered(const Callable &p_callable) {
	// It gets *extremely* hard to guarantee the proper semantics if you're allowed to call this from other threads.
	ERR_FAIL_COND_MSG(!Thread::is_main_thread(), "This call is forbidden outside the main thread.");

	MutexLock lock(mutex);
	register_callable(captures_buffered, p_callable);
	recalculate_state();
}

void UserLogManagerLogger::unregister_log_capture_buffered(const Callable &p_callable) {
	// It gets *extremely* hard to guarantee the proper semantics if you're allowed to call this from other threads.
	ERR_FAIL_COND_MSG(!Thread::is_main_thread(), "This call is forbidden outside the main thread.");

	MutexLock lock(mutex);
	unregister_callable(captures_buffered, p_callable);
	recalculate_state();
}

void UserLogManagerLogger::flush() {
	// if you're not sure why this is important, go read the giant comment near the end of recalculate_state
	// it avoids a nearly-impossible race condition
	ERR_FAIL_COND_MSG(!Thread::is_main_thread(), "This call is forbidden outside the main thread.");

	// This flushes our buffer and recycles it for the next frame.
	// If we don't have a buffer, we have nothing to do.
	if (state.get() != STATE_BUFFERING) {
		return;
	}

	// "Send all our buffered messages to all our buffered log readers"
	// To avoid inconsistent message delivery of anything that's already been buffered, we swap the entire buffer first
	// Anything that gets delivered while we're writing these messages gets to wait for next frame.
	// (We could loop until the buffer is empty)
	Vector<Dictionary> buffered_logs_mirror;
	{
		MutexLock lock(mutex);
		SWAP(buffered_logs_mirror, buffered_logs);
	}

	// Dispatch to all the buffered callables, in a thread-safe manner
	// Any buffered callables that disable themselves stop getting messages ASAP
	// any buffered callables that get attached might start getting messages midway through
	// we're ok with that, it's still a chronologically coherent block
	for (const Dictionary &log : buffered_logs_mirror) {
		// This is the same index-as-iterator-to-avoid-lock-issues dance we do in register_log_capture_non_thread_safe()
		int index_to_send_to = 0;

		while (true) {
			Callable callable;
			{
				MutexLock lock(mutex);
				if (index_to_send_to >= captures_buffered.size()) {
					// we done!
					break;
				}

				callable = captures_buffered[index_to_send_to++];
			}

			dispatch_message(log, callable);
		}
	}

	if (pre_buffering.is_set()) {
		// Buffering is done!
		// Read the comment near the bottom of recalculate_state for a possible race condition and how it's dealt with.
		// IT IS REALLY IMPORTANT THAT THIS HAPPENS ONLY ON THE MAIN THREAD.
		MutexLock lock(mutex);
		pre_buffering.clear();
		recalculate_state();
	}
}

void UserLogManagerLogger::process(const Dictionary &p_message) {
	Vector<Callable> captures_non_thread_safe_mirror;
	{
		// Shove another item into the buffer, if we need to.

		// We also atomically grab `captures_non_thread_safe` to avoid race conditions, in the extraordinarily unlikely case where:
		// * this function buffers a message, releases the lock, and is immediately preempted
		// * in another thread, an already-running register_log_capture_non_thread_safe dispatches our new message from the buffer
		// * that same register_log_capture_non_thread_safe gets to the end of the buffer and adds itself to captures_non_thread_safe
		// * this function resumes, reaches the captures_non_thread_safe section, and sends the same message again

		// register_log_capture_non_thread_safe intentionally finishes the buffered log message replay atomically with adding a new handler to the vector
		// so we do the same thing here, adding a message to the log atomically with copying the handlers
		// Vector COW behavior prevents this from being expensive in the common case
		MutexLock lock(mutex);
		if (state.get() == STATE_BUFFERING) {
			buffered_logs.append(p_message);
		}
		captures_non_thread_safe_mirror = captures_non_thread_safe;
	}

	// Dispatch to all the non_thread_safe callables at the moment we added to the buffer
	// We actually don't have to care about cutesy thread-safety for once because we're working off a local copy of the captures list

	for (const Callable &callable : captures_non_thread_safe_mirror) {
		dispatch_message(p_message, callable);
	}
}

void UserLogManagerLogger::dispatch_message(const Dictionary &p_message, const Callable &p_callable) {
	// ideally we should verify that the mutex is *not* held by this thread, but the current API provides no way to do that
	// (it's possible it's already been scooped up by another thread, that's fine)

	if (!p_callable.is_valid()) {
		// in most cases, this is going to be a deleted callable, so ignore it;
		// is_null() would be enough for that, we intentionally invalidate it to null in register_callable
		//
		// it's apparently possible for callables to be yanked out from under us, though, and if we call them, we crash
		// so, do the full is_valid() check
		//
		// (easier to put the check here than everywhere that calls this)
		return;
	}

	Variant message_variant = p_message;
	const Variant *args[1] = { &message_variant };
	Variant retval;
	Callable::CallError err;
	p_callable.callp(args, 1, retval, err);
}

void UserLogManagerLogger::recalculate_state() {
	// we should verify that the mutex is held (ideally by this thread) but the current API provides no way to do that
	// (aside from switching to a non-recursive mutex and trying to lock it and hoping it fails, which, no)

	State new_state = STATE_OFF;

	if (pre_buffering.is_set()) {
		// We always buffer on the first frame, in case someone hooks to us and expects a replay
		new_state = STATE_BUFFERING;
	} else if (!captures_buffered.is_empty()) {
		// Anything buffering means we need to preserve the buffer
		new_state = STATE_BUFFERING;
	} else if (!captures_non_thread_safe.is_empty()) {
		// Anything non-buffering means we still need to process
		new_state = STATE_PASSTHROUGH;
	}
	// Otherwise, we're off

	// if we're transitioning from buffering to anything else, we need to clear the buffer
	if (state.get() == STATE_BUFFERING && new_state != STATE_BUFFERING) {
		// NOTE: this line is why the register/unregister functions are locked to the main thread!

		// imagine this set of events:
		// * we're on frame 0
		// * non-main thread: we register a non_thread_safe handler
		// * non-main thread: the non_thread_safe handler starts replaying the buffer
		// * main thread: on the main thread, we transition to frame 1
		// * main thread: flush() calls this function
		// * main thread: we decide to clear the buffer between locks as it's being replayed in the non-main thread
		// * non-main thread: this doesn't crash, because our index iteration is safe, but this does prematurely terminate the replay
		// * non-main thread: the non_thread_safe handler gets hooked properly and starts echoing live messages

		// end result, a bunch of "guaranteed to be replayed" messages vanish into the ether

		// I don't like cases that end up with a batch of log messages mysteriously vanishing!
		// non_thread_safe replays happen only on frame 0
		// so this can be fixed by just ensuring that the frame0-frame1 transition must happen on the same thread as all non_thread_safe buffer replays
		// thus ensuring they cannot happen in parallel
		// and because the frame0-frame1 transition is locked to the main thread,
		// the non_thread_safe buffer replays must also be locked to the main thread,
		// and because those are part of the non_thread_safe_register function,
		// the non_thread_safe_register function must (ironically) be locked to the main thread.
		// for symmetry's sake, so is everything else (and because frankly the threading in this area is knotty enough already)

		buffered_logs.clear();
	}

	state.set(new_state);
}

void UserLogManagerLogger::register_callable(Vector<Callable> &p_vector, const Callable &p_callable) {
	// we should verify that the mutex is held (ideally by this thread) but the current API provides no way to do that

	// right now we're just letting people double-register if they want
	// this should maybe be a warning but that gets hairy with register_log_capture_non_thread_safe
	// so instead I'm just saying "you registered it twice, what did you expect would happen"
	int index = p_vector.find(Callable());
	if (index != -1) {
		// reuse an empty Callable() slot if there is one
		p_vector.write[index] = p_callable;
	} else {
		p_vector.append(p_callable);
	}
}

void UserLogManagerLogger::unregister_callable(Vector<Callable> &p_vector, const Callable &p_callable) {
	// we should verify that the mutex is held (ideally by this thread) but the current API provides no way to do that

	// find the index and replace it with Callable() to invalidate it without moving indices around, which would break in-flight iterators in other threads
	// to parallel the register-twice semantics we're intentionally removing exactly one copy here
	int index = p_vector.find(p_callable);
	if (index != -1) {
		p_vector.write[index] = Callable();
	}
}
