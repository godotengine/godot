#pragma once

#include "types.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace riscv {

// Hardware Abstraction Layer for cross-platform support
// Eliminates code duplication between iOS, Android, macOS, Linux, Windows

class PlatformHAL {
public:
    virtual ~PlatformHAL() = default;
    
    // Event/Epoll abstraction
    virtual int create_event_queue(int flags) = 0;
    virtual int event_ctl(int efd, int op, int fd, uint32_t events, uint64_t data) = 0;
    virtual int event_wait(int efd, struct EventResult* events, int maxevents, int timeout_ms) = 0;
    virtual int create_eventfd(int initval, int flags) = 0;
    virtual void close_event_queue(int efd) = 0;
    
    // Time/Clock abstraction
    virtual int get_time(int clkid, struct timespec* ts) = 0;
    virtual int get_timeofday(struct timeval* tv) = 0;
    virtual int nanosleep(const struct timespec* req, struct timespec* rem) = 0;
    
    // File operations abstraction
    virtual bool supports_getdents64() = 0;
    virtual int getdents64(int fd, void* dirp, size_t count) = 0;
    virtual bool supports_dup3() = 0;
    virtual int dup3(int oldfd, int newfd, int flags) = 0;
    virtual bool supports_pipe2() = 0;
    virtual int pipe2(int pipefd[2], int flags) = 0;
    virtual bool supports_preadv() = 0;
    virtual ssize_t preadv(int fd, const struct iovec* iov, int iovcnt, off_t offset) = 0;
    
    // Random number generation
    virtual ssize_t get_random(void* buf, size_t buflen, unsigned int flags) = 0;
    
    // Memory management
    virtual void* mmap_impl(void* addr, size_t length, int prot, int flags, int fd, off_t offset) = 0;
    virtual int munmap_impl(void* addr, size_t length) = 0;
    virtual int mprotect_impl(void* addr, size_t len, int prot) = 0;
    
    // Signal handling
    virtual bool supports_signalfd() = 0;
    virtual int signalfd(int fd, const void* mask, int flags) = 0;
};

// Event result structure for cross-platform compatibility
struct EventResult {
    uint32_t events;
    uint64_t data;
    int fd;
};

// Epoll constants (common across platforms)
constexpr uint32_t HAL_EPOLLIN     = 0x001;
constexpr uint32_t HAL_EPOLLOUT    = 0x004;
constexpr uint32_t HAL_EPOLLERR    = 0x008;
constexpr uint32_t HAL_EPOLLHUP    = 0x010;
constexpr uint32_t HAL_EPOLLRDHUP  = 0x2000;

constexpr int HAL_EPOLL_CTL_ADD = 1;
constexpr int HAL_EPOLL_CTL_DEL = 2;
constexpr int HAL_EPOLL_CTL_MOD = 3;

// Factory function to create platform-specific HAL
PlatformHAL* create_platform_hal();

} // namespace riscv
