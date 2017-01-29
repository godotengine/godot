#ifndef SOCKET_HELPERS_H
#define SOCKET_HELPERS_H

#include <string.h>

#if defined(__MINGW32__ ) && (!defined(__MINGW64_VERSION_MAJOR) || __MINGW64_VERSION_MAJOR < 4)
  // Workaround for mingw-w64 < 4.0
  #ifndef IPV6_V6ONLY
    #define IPV6_V6ONLY 27
  #endif
#endif

// helpers for sockaddr -> IP_Address and back, should work for posix and winsock. All implementations should use this

// small test to see if current kernel supports ipv6
static bool _ipv6_available(){
	int sockfd = socket( AF_INET6,  SOCK_STREAM,  0 );
	if ( -1 == sockfd ){
		WARN_PRINT( strerror( errno ) ); 
		return false; // not supported 
	} else 
		close( sockfd );
	return true;
}


static size_t _set_sockaddr(struct sockaddr_storage* p_addr, const IP_Address& p_ip, int p_port, IP::Type p_sock_type = IP::TYPE_ANY) {

	memset(p_addr, 0, sizeof(struct sockaddr_storage));

	ERR_FAIL_COND_V(p_ip==IP_Address(),0);

	bool new_socket = false;
	if ( IP::TYPE_IPV6 == p_sock_type || IP::TYPE_ANY == p_sock_type ) {
			if ( _ipv6_available() )
				new_socket = true;
			else
				WARN_PRINT("IPv6 is not supported by current system.");
	}

	if ( new_socket ) { // ipv6
		// IPv6 only socket with IPv4 address
		ERR_FAIL_COND_V(p_sock_type == IP::TYPE_IPV6 && p_ip.is_ipv4(),0);

		struct sockaddr_in6* addr6 = (struct sockaddr_in6*)p_addr;
		addr6->sin6_family = AF_INET6;
		addr6->sin6_port = htons(p_port);
		copymem(&addr6->sin6_addr.s6_addr, p_ip.get_ipv6(), 16);
		return sizeof(sockaddr_in6);

	} else { // ipv4
		// IPv4 socket with IPv6 address
		ERR_FAIL_COND_V(!p_ip.is_ipv4(),0);

		uint32_t ipv4 = *((uint32_t *)p_ip.get_ipv4());
		struct sockaddr_in* addr4 = (struct sockaddr_in*)p_addr;
		addr4->sin_family = AF_INET;
		addr4->sin_port = htons(p_port);  // short, network byte order
		copymem(&addr4->sin_addr.s_addr, p_ip.get_ipv4(), 16);
		return sizeof(sockaddr_in);
	};
};

static size_t _set_listen_sockaddr(struct sockaddr_storage* p_addr, int p_port, IP::Type p_sock_type, const List<String> *p_accepted_hosts) {

	memset(p_addr, 0, sizeof(struct sockaddr_storage));
	// determine the correct socket domain
	bool new_socket = false;
	if ( IP::TYPE_IPV6 == p_sock_type || IP::TYPE_ANY == p_sock_type ) {
			if ( _ipv6_available() )
				new_socket = true;
			else
				WARN_PRINT("IPv6 is not supported by current system.");
	}
	if ( new_socket ) { // ipv6
		struct sockaddr_in6* addr6 = (struct sockaddr_in6*)p_addr;
		addr6->sin6_family = AF_INET6;
		addr6->sin6_port = htons(p_port);
		addr6->sin6_addr = in6addr_any; // TODO: use accepted hosts list
		return sizeof(sockaddr_in6);
	} else { // ipv4
		struct sockaddr_in* addr4 = (struct sockaddr_in*)p_addr;
		addr4->sin_family = AF_INET;
		addr4->sin_port = htons(p_port);
		addr4->sin_addr.s_addr = INADDR_ANY; // TODO: use accepted hosts list
		return sizeof(sockaddr_in);
	};
};

static int _socket_create(IP::Type p_type, int type, int protocol) {

	ERR_FAIL_COND_V(p_type > IP::TYPE_ANY || p_type < IP::TYPE_NONE, ERR_INVALID_PARAMETER);

	// determine socket domain
	int family = AF_INET; // default to ipv4
	if ( IP::TYPE_IPV6 == p_type || IP::TYPE_ANY == p_type ){
		if ( _ipv6_available() )
			family = AF_INET6; // prefer ipv6, if supported
		else
			WARN_PRINT("IPv6 is not supported by current system.");
	}
	
	int sockfd = socket(family, type, protocol);
	if ( -1 == sockfd )
		ERR_PRINT( strerror( errno ) ); // be a little more verbose on errors
	ERR_FAIL_COND_V( sockfd == -1, -1 );

	if(family == AF_INET6) {
		// Select IPv4 over IPv6 mapping
		int opt = p_type != IP::TYPE_ANY;
		if(setsockopt(sockfd, IPPROTO_IPV6, IPV6_V6ONLY, (const char*)&opt, sizeof(opt)) != 0) {
			WARN_PRINT("Unable to set/unset IPv4 address mapping over IPv6");
		}
	}

	return sockfd;
}


static void _set_ip_addr_port(IP_Address& r_ip, int& r_port, struct sockaddr_storage* p_addr) {

	if (p_addr->ss_family == AF_INET) {

		struct sockaddr_in* addr4 = (struct sockaddr_in*)p_addr;
		r_ip.set_ipv4((uint8_t *)&(addr4->sin_addr.s_addr));

		r_port = ntohs(addr4->sin_port);

	} else if (p_addr->ss_family == AF_INET6) {

		struct sockaddr_in6* addr6 = (struct sockaddr_in6*)p_addr;
		r_ip.set_ipv6(addr6->sin6_addr.s6_addr);

		r_port = ntohs(addr6->sin6_port);
	};
};


#endif
