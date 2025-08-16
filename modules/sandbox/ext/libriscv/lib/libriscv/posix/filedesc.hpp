#pragma once
#include <functional>
#include <string>
#include <map>
#include "../types.hpp"

#if defined(__APPLE__) || defined(__LINUX__)
#include <errno.h>
#endif

namespace riscv {

struct FileDescriptors
{
#ifdef _WIN32
    typedef uint64_t real_fd_type; // SOCKET is uint64_t
#else
    typedef int real_fd_type;
#endif

	// Insert and manage real FDs, return virtual FD
	int assign_file(real_fd_type fd) { return assign(fd, false); }
	int assign_socket(real_fd_type fd) { return assign(fd, true); }
	int assign(real_fd_type fd, bool socket);
	// Get real FD from virtual FD
    real_fd_type get(int vfd);
    real_fd_type translate(int vfd);
	// Remove virtual FD and return real FD
    real_fd_type erase(int vfd);

	bool is_socket(int) const;
	bool permit_write(int vfd) {
		if (is_socket(vfd)) return true;
		else return proxy_mode;
	}

	~FileDescriptors();

    std::map<int, real_fd_type> translation;

	// Default working directory (fake root)
	std::string cwd = "/home";

	static constexpr int FILE_D_BASE = 0x1000;
	static constexpr int SOCKET_D_BASE = 0x40001000;
	int file_counter = FILE_D_BASE;
	int socket_counter = SOCKET_D_BASE;

	bool permit_filesystem = false;
	bool permit_sockets = false;
	bool proxy_mode = false;

	std::function<bool(void*, std::string&)> filter_open = nullptr; /* NOTE: Can modify path */
	std::function<bool(void*, std::string&)> filter_readlink = nullptr; /* NOTE: Can modify path */
	std::function<bool(void*, const std::string&)> filter_stat = nullptr;
	std::function<bool(void*, uint64_t)> filter_ioctl = nullptr;
};

inline int FileDescriptors::assign(FileDescriptors::real_fd_type real_fd, bool socket)
{
	int virtfd;
	if (!socket)
		virtfd = file_counter++;
	else
		virtfd = socket_counter++;

	translation.emplace(virtfd, real_fd);
	return virtfd;
}
inline FileDescriptors::real_fd_type FileDescriptors::get(int virtfd)
{
	auto it = translation.find(virtfd);
	if (it != translation.end()) return it->second;
	return -EBADF;
}
inline FileDescriptors::real_fd_type FileDescriptors::translate(int virtfd)
{
	auto it = translation.find(virtfd);
	if (it != translation.end()) return it->second;
	// Only allow direct access to standard pipes and errors
	return (virtfd <= 2) ? virtfd : -1;
}
inline FileDescriptors::real_fd_type FileDescriptors::erase(int virtfd)
{
	auto it = translation.find(virtfd);
	if (it != translation.end()) {
        FileDescriptors::real_fd_type real_fd = it->second;
		// Remove the virt FD
		translation.erase(it);
		return real_fd;
	}
	return -EBADF;
}

inline bool FileDescriptors::is_socket(int virtfd) const
{
	return virtfd >= SOCKET_D_BASE;
}

} // riscv
