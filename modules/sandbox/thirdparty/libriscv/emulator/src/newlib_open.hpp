const auto g_path = machine.sysarg(0);
const int flags  = machine.template sysarg<int>(1);
const int mode   = machine.template sysarg<int>(2);
// This is a custom syscall for the newlib mini guest
std::string path = machine.memory.memstring(g_path);
if (machine.has_file_descriptors() && machine.fds().permit_filesystem) {

	if (machine.fds().filter_open != nullptr) {
		// filter_open() can modify the path
		if (!machine.fds().filter_open(machine.template get_userdata<void>(), path)) {
			machine.set_result(-EPERM);
			return;
		}

#if !defined(_MSC_VER)
		int res = open(path.c_str(), flags, mode);
		if (res > 0)
			res = machine.fds().assign_file(res);
		machine.set_result_or_error(res);
		return;
#endif
	}
}
machine.set_result(-EPERM);
return;
