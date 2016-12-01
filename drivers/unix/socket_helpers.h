#ifndef SOCKET_HELPERS_H
#define SOCKET_HELPERS_H

#include <string.h>

#ifdef WINDOWS_ENABLED
 // Workaround mingw missing flags!
 #ifndef IPV6_V6ONLY
  #define IPV6_V6ONLY 27
 #endif
#endif

#ifdef UWP_ENABLED
#define in6addr_any IN6ADDR_ANY_INIT
#endif

// helpers for sockaddr -> IP_Address and back, should work for posix and winsock. All implementations should use this

static size_t _set_sockaddr(struct sockaddr_storage* p_addr, const IP_Address& p_ip, int p_port, IP_Address::AddrType p_sock_type = IP_Address::TYPE_ANY) {

	memset(p_addr, 0, sizeof(struct sockaddr_storage));

	// Dual stack (ANY) or matching ip type is required
	ERR_FAIL_COND_V(p_sock_type != IP_Address::TYPE_ANY && p_sock_type != p_ip.type,0);

	// IPv6 socket
	if (p_sock_type == IP_Address::TYPE_IPV6 || p_sock_type == IP_Address::TYPE_ANY) {

		struct sockaddr_in6* addr6 = (struct sockaddr_in6*)p_addr;
		addr6->sin6_family = AF_INET6;
		addr6->sin6_port = htons(p_port);
		if(p_ip.type == IP_Address::TYPE_IPV4) {
			// Remapping needed
			uint16_t base[8] = {0x0, 0x0, 0x0, 0x0, 0x0, 0xffff, p_ip.field16[0], p_ip.field16[1]};
			copymem(&addr6->sin6_addr.s6_addr, base, 16);

		} else {
			copymem(&addr6->sin6_addr.s6_addr, p_ip.field8, 16);
		}
		return sizeof(sockaddr_in6);

	} else { // IPv4 socket

		struct sockaddr_in* addr4 = (struct sockaddr_in*)p_addr;
		addr4->sin_family = AF_INET;    // host byte order
		addr4->sin_port = htons(p_port);  // short, network byte order
		addr4->sin_addr = *((struct in_addr*)&p_ip.field32[0]);
		return sizeof(sockaddr_in);
	};
};

static size_t _set_listen_sockaddr(struct sockaddr_storage* p_addr, int p_port, IP_Address::AddrType p_address_type, const List<String> *p_accepted_hosts) {

	memset(p_addr, 0, sizeof(struct sockaddr_storage));
	if (p_address_type == IP_Address::TYPE_IPV4) {
		struct sockaddr_in* addr4 = (struct sockaddr_in*)p_addr;
		addr4->sin_family = AF_INET;
		addr4->sin_port = htons(p_port);
		addr4->sin_addr.s_addr = INADDR_ANY; // TODO: use accepted hosts list
		return sizeof(sockaddr_in);
	} else {
		struct sockaddr_in6* addr6 = (struct sockaddr_in6*)p_addr;

		addr6->sin6_family = AF_INET6;
		addr6->sin6_port = htons(p_port);
		addr6->sin6_addr = in6addr_any; // TODO: use accepted hosts list
		return sizeof(sockaddr_in6);
	};
};

static int _socket_create(IP_Address::AddrType p_type, int type, int protocol) {

	ERR_FAIL_COND_V(p_type > IP_Address::TYPE_ANY || p_type < IP_Address::TYPE_NONE, ERR_INVALID_PARAMETER);

	int family = p_type == IP_Address::TYPE_IPV4 ? AF_INET : AF_INET6;
	int sockfd = socket(family, type, protocol);

	ERR_FAIL_COND_V( sockfd == -1, -1 );

	if(family == AF_INET6) {
		// Ensure IPv4 over IPv6 is enabled
		int no = 0;
		if(setsockopt(sockfd, IPPROTO_IPV6, IPV6_V6ONLY, (const char*)&no, sizeof(no)) != 0) {
			WARN_PRINT("Unable to set IPv4 address mapping over IPv6");
		}
	}

	return sockfd;
}


static void _set_ip_addr_port(IP_Address& r_ip, int& r_port, struct sockaddr_storage* p_addr) {

	if (p_addr->ss_family == AF_INET) {
		r_ip.type = IP_Address::TYPE_IPV4;

		struct sockaddr_in* addr4 = (struct sockaddr_in*)p_addr;
		r_ip.field32[0] = (uint32_t)addr4->sin_addr.s_addr;

		r_port = ntohs(addr4->sin_port);

	} else if (p_addr->ss_family == AF_INET6) {

		r_ip.type = IP_Address::TYPE_IPV6;

		struct sockaddr_in6* addr6 = (struct sockaddr_in6*)p_addr;
		copymem(&r_ip.field8, addr6->sin6_addr.s6_addr, 16);

		r_port = ntohs(addr6->sin6_port);
	};
};


#endif
