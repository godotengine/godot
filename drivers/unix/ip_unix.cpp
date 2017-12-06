/*************************************************************************/
/*  ip_unix.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "ip_unix.h"

#if defined(UNIX_ENABLED) || defined(WINDOWS_ENABLED)

#include <string.h>

#ifdef WINDOWS_ENABLED
#include <stdio.h>
#include <winsock2.h>
// Needs to be included after winsocks2.h
#include <windows.h>
#include <ws2tcpip.h>
#ifndef UWP_ENABLED
#if defined(__MINGW32__) && (!defined(__MINGW64_VERSION_MAJOR) || __MINGW64_VERSION_MAJOR < 4)
// MinGW-w64 on Ubuntu 12.04 (our Travis build env) has bugs in this code where
// some includes are missing in dependencies of iphlpapi.h for WINVER >= 0x0600 (Vista).
// We don't use this Vista code for now, so working it around by disabling it.
// MinGW-w64 >= 4.0 seems to be better judging by its headers.
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0501 // Windows XP, disable Vista API
#include <iphlpapi.h>
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0600 // Reenable Vista API
#else
#include <iphlpapi.h>
#endif // MINGW hack
#endif
#else
#include <netdb.h>
#ifdef ANDROID_ENABLED
#include "platform/android/ifaddrs_android.h"
#else
#ifdef __FreeBSD__
#include <sys/types.h>
#endif
#include <ifaddrs.h>
#endif
#include <arpa/inet.h>
#include <sys/socket.h>
#ifdef __FreeBSD__
#include <netinet/in.h>
#endif
#endif

static IP_Address _sockaddr2ip(struct sockaddr *p_addr) {

	IP_Address ip;
	if (p_addr->sa_family == AF_INET) {
		struct sockaddr_in *addr = (struct sockaddr_in *)p_addr;
		ip.set_ipv4((uint8_t *)&(addr->sin_addr));
	} else if (p_addr->sa_family == AF_INET6) {
		struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)p_addr;
		ip.set_ipv6(addr6->sin6_addr.s6_addr);
	};

	return ip;
};

IP_Address IP_Unix::_resolve_hostname(const String &p_hostname, Type p_type) {

	struct addrinfo hints;
	struct addrinfo *result;

	memset(&hints, 0, sizeof(struct addrinfo));
	if (p_type == TYPE_IPV4) {
		hints.ai_family = AF_INET;
	} else if (p_type == TYPE_IPV6) {
		hints.ai_family = AF_INET6;
		hints.ai_flags = 0;
	} else {
		hints.ai_family = AF_UNSPEC;
		hints.ai_flags = AI_ADDRCONFIG;
	};
	hints.ai_flags &= !AI_NUMERICHOST;

	int s = getaddrinfo(p_hostname.utf8().get_data(), NULL, &hints, &result);
	if (s != 0) {
		ERR_PRINT("getaddrinfo failed!");
		return IP_Address();
	};

	if (result == NULL || result->ai_addr == NULL) {
		ERR_PRINT("Invalid response from getaddrinfo");
		return IP_Address();
	};

	IP_Address ip = _sockaddr2ip(result->ai_addr);

	freeaddrinfo(result);

	return ip;
}

#if defined(WINDOWS_ENABLED)

#if defined(UWP_ENABLED)

void IP_Unix::get_local_addresses(List<IP_Address> *r_addresses) const {

	using namespace Windows::Networking;
	using namespace Windows::Networking::Connectivity;

	auto hostnames = NetworkInformation::GetHostNames();

	for (int i = 0; i < hostnames->Size; i++) {

		if (hostnames->GetAt(i)->Type == HostNameType::Ipv4 || hostnames->GetAt(i)->Type == HostNameType::Ipv6 && hostnames->GetAt(i)->IPInformation != nullptr) {

			r_addresses->push_back(IP_Address(String(hostnames->GetAt(i)->CanonicalName->Data())));
		}
	}
};
#else

void IP_Unix::get_local_addresses(List<IP_Address> *r_addresses) const {

	ULONG buf_size = 1024;
	IP_ADAPTER_ADDRESSES *addrs;

	while (true) {

		addrs = (IP_ADAPTER_ADDRESSES *)memalloc(buf_size);
		int err = GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER | GAA_FLAG_SKIP_FRIENDLY_NAME,
				NULL, addrs, &buf_size);
		if (err == NO_ERROR) {
			break;
		};
		memfree(addrs);
		if (err == ERROR_BUFFER_OVERFLOW) {
			continue; // will go back and alloc the right size
		};

		ERR_EXPLAIN("Call to GetAdaptersAddresses failed with error " + itos(err));
		ERR_FAIL();
		return;
	};

	IP_ADAPTER_ADDRESSES *adapter = addrs;

	while (adapter != NULL) {

		IP_ADAPTER_UNICAST_ADDRESS *address = adapter->FirstUnicastAddress;
		while (address != NULL) {

			IP_Address ip;

			if (address->Address.lpSockaddr->sa_family == AF_INET) {

				SOCKADDR_IN *ipv4 = reinterpret_cast<SOCKADDR_IN *>(address->Address.lpSockaddr);

				ip.set_ipv4((uint8_t *)&(ipv4->sin_addr));
				r_addresses->push_back(ip);

			} else if (address->Address.lpSockaddr->sa_family == AF_INET6) { // ipv6

				SOCKADDR_IN6 *ipv6 = reinterpret_cast<SOCKADDR_IN6 *>(address->Address.lpSockaddr);

				ip.set_ipv6(ipv6->sin6_addr.s6_addr);
				r_addresses->push_back(ip);
			};

			address = address->Next;
		};
		adapter = adapter->Next;
	};

	memfree(addrs);
};

#endif

#else

void IP_Unix::get_local_addresses(List<IP_Address> *r_addresses) const {

	struct ifaddrs *ifAddrStruct = NULL;
	struct ifaddrs *ifa = NULL;
	int family;

	getifaddrs(&ifAddrStruct);

	for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
		if (!ifa->ifa_addr)
			continue;

		family = ifa->ifa_addr->sa_family;

		if (family != AF_INET && family != AF_INET6)
			continue;

		IP_Address ip = _sockaddr2ip(ifa->ifa_addr);
		r_addresses->push_back(ip);
	}

	if (ifAddrStruct != NULL) freeifaddrs(ifAddrStruct);
}
#endif

void IP_Unix::make_default() {

	_create = _create_unix;
}

IP *IP_Unix::_create_unix() {

	return memnew(IP_Unix);
}

IP_Unix::IP_Unix() {
}

#endif
