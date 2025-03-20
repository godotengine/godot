
#ifndef DISABLE_DEPRECATED

namespace core_bind {

// Semaphore

void Semaphore::_post_bind_compat_93605() {
	post(1);
}

void Semaphore::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("post"), &Semaphore::_post_bind_compat_93605);
}

// OS

String OS::_read_string_from_stdin_bind_compat_91201() {
	return read_string_from_stdin(1024);
}

Dictionary OS::_execute_with_pipe_bind_compat_94434(const String &p_path, const Vector<String> &p_arguments) {
	return execute_with_pipe(p_path, p_arguments, true);
}

void OS::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("read_string_from_stdin", "buffer_size"), &OS::read_string_from_stdin);
	ClassDB::bind_compatibility_method(D_METHOD("read_buffer_from_stdin", "buffer_size"), &OS::read_buffer_from_stdin);
	ClassDB::bind_compatibility_method(D_METHOD("read_string_from_stdin"), &OS::_read_string_from_stdin_bind_compat_91201);
	ClassDB::bind_compatibility_method(D_METHOD("execute_with_pipe", "path", "arguments"), &OS::_execute_with_pipe_bind_compat_94434);
}

} // namespace core_bind

#endif // DISABLE_DEPRECATED
