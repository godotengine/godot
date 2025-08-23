#pragma once
#include <winsock2.h>
#include <ws2tcpip.h>

namespace riscv {
namespace ws2 {

// Declared in socket_calls.cpp
extern WSADATA global_winsock_data;
extern bool winsock_initialized;

inline void init() {
    if (!winsock_initialized) {
        WSAStartup(MAKEWORD(2, 2), &global_winsock_data);
        winsock_initialized = true;
    }
}

}
}
