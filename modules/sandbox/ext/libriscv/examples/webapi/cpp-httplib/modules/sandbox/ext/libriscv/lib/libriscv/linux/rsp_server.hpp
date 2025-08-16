#pragma once
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

namespace riscv {

template <int W>
RSP<W>::RSP(riscv::Machine<W>& m, uint16_t port)
	: m_machine{m}
{
	this->server_fd = socket(AF_INET,
	                         SOCK_STREAM 
#ifdef SOCK_NONBLOCK
	                         	| SOCK_NONBLOCK
#endif
	                        ,0);
	int opt = 1;
	if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
		&opt, sizeof(opt))) {
		close(server_fd);
		throw MachineException(SYSTEM_CALL_FAILED, "Failed to enable REUSEADDR/PORT");
	}
	struct sockaddr_in address;
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_port = htons(port);
	if (bind(server_fd, (struct sockaddr*) &address,
			sizeof(address)) < 0) {
		close(server_fd);
		throw MachineException(SYSTEM_CALL_FAILED, "GDB listener failed to bind to port");
	}
	if (listen(server_fd, 2) < 0) {
		close(server_fd);
		throw MachineException(SYSTEM_CALL_FAILED, "GDB listener failed to listen on port");
	}
}
template <int W>
std::unique_ptr<RSPClient<W>> RSP<W>::accept(int timeout_secs)
{
	struct timeval tv {
		.tv_sec = timeout_secs,
		.tv_usec = 0
	};
	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(server_fd, &fds);

	const int ret = select(server_fd + 1, &fds, NULL, NULL, &tv);
	if (ret <= 0) {
		return nullptr;
	}

	struct sockaddr_in address;
	int addrlen = sizeof(address);
	int sockfd = ::accept(server_fd, (struct sockaddr*) &address,
			(socklen_t*) &addrlen);
	if (sockfd < 0) {
		return nullptr;
	}
	// Disable Nagle
	int opt = 1;
	if (setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt))) {
		close(sockfd);
		return nullptr;
	}
	// Enable receive and send timeouts
	tv = {
		.tv_sec = 60,
		.tv_usec = 0
	};
	if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO | SO_SNDTIMEO, &tv, sizeof(tv))) {
		close(sockfd);
		return nullptr;
	}
	return std::make_unique<RSPClient<W>>(m_machine, sockfd);
}
template <int W> inline
RSP<W>::~RSP() {
	close(server_fd);
}

template <int W> inline
RSPClient<W>::~RSPClient() {
    if (!is_closed())
        close(this->sockfd);
}

template <int W> inline
void RSPClient<W>::close_now() {
    this->m_closed = true;
    close(this->sockfd);
}

template <int W>
bool RSPClient<W>::sendf(const char* fmt, ...)
{
    char buffer[PACKET_SIZE];
    va_list args;
    va_start(args, fmt);
    int plen = forge_packet(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    if (UNLIKELY(m_verbose)) {
        printf("TX >>> %.*s\n", plen, buffer);
    }
    int len = ::write(sockfd, buffer, plen);
    if (len <= 0) {
        this->close_now();
        return false;
    }
    // Acknowledgement
    int rlen = ::read(sockfd, buffer, 1);
    if (rlen <= 0) {
        this->close_now();
        return false;
    }
    return (buffer[0] == '+');
}

template <int W>
bool RSPClient<W>::send(const char* str)
{
    char buffer[PACKET_SIZE];
    int plen = forge_packet(buffer, sizeof(buffer), str, strlen(str));
    if (UNLIKELY(m_verbose)) {
        printf("TX >>> %.*s\n", plen, buffer);
    }
    int len = ::write(sockfd, buffer, plen);
    if (len <= 0) {
        this->close_now();
        return false;
    }
    // Acknowledgement
    int rlen = ::read(sockfd, buffer, 1);
    if (rlen <= 0) {
        this->close_now();
        return false;
    }
    return (buffer[0] == '+');
}
template <int W>
bool RSPClient<W>::process_one()
{
    char tmp[1024];
    int len = ::read(this->sockfd, tmp, sizeof(tmp));
    if (len <= 0) {
        this->close_now();
        return false;
    }
    if (UNLIKELY(m_verbose)) {
        printf("RX <<< %.*s\n", len, tmp);
    }
    for (int i = 0; i < len; i++)
    {
        char c = tmp[i];
        if (buffer.empty() && c == '+') {
            /* Ignore acks? */
        }
        else if (c == '$') {
            this->buffer.clear();
        }
        else if (c == '#') {
            reply_ack();
            process_data();
            this->buffer.clear();
            i += 2;
        }
        else {
            this->buffer.append(&c, 1);
            if (buffer.size() >= PACKET_SIZE)
                break;
        }
    }
    return true;
}

template <int W> inline
void RSPClient<W>::reply_ack() {
    ssize_t len = write(sockfd, "+", 1);
    if (len < 0) throw MachineException(SYSTEM_CALL_FAILED, "RSPClient: Unable to ACK");
}

template <int W>
void RSPClient<W>::kill() {
    close(sockfd);
}
} // riscv
