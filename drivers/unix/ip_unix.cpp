/*************************************************************************/
/*  ip_unix.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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


#ifdef WINDOWS_ENABLED
 #ifdef WINRT_ENABLED
  #include <ws2tcpip.h>
  #include <winsock2.h>
  #include <windows.h>
  #include <stdio.h>
 #else
  #define WINVER 0x0600
  #include <ws2tcpip.h>
  #include <winsock2.h>
  #include <windows.h>
  #include <stdio.h>
  #include <iphlpapi.h>
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

IP_Address IP_Unix::_resolve_hostname(const String& p_hostname) {

	struct hostent *he;
	if ((he=gethostbyname(p_hostname.utf8().get_data())) == NULL) {  // get the host info
		ERR_PRINT("gethostbyname failed!");
		return IP_Address();
	}
	IP_Address ip;

	ip.host= *((unsigned long*)he->h_addr);

	return ip;

}

#if defined(WINDOWS_ENABLED)

#if defined(WINRT_ENABLED)

void IP_Unix::get_local_addresses(List<IP_Address> *r_addresses) const {

	using namespace Windows::Networking;
	using namespace Windows::Networking::Connectivity;

	auto hostnames = NetworkInformation::GetHostNames();

	for (int i = 0; i < hostnames->Size; i++) {

		if (hostnames->GetAt(i)->Type == HostNameType::Ipv4 && hostnames->GetAt(i)->IPInformation != nullptr) {

			r_addresses->push_back(IP_Address(String(hostnames->GetAt(i)->CanonicalName->Data())));

		}
	}

};
#else

void IP_Unix::get_local_addresses(List<IP_Address> *r_addresses) const {

	ULONG buf_size = 1024;
	IP_ADAPTER_ADDRESSES* addrs;

	while (true) {

		addrs = (IP_ADAPTER_ADDRESSES*)memalloc(buf_size);
		int err = GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST |
									   GAA_FLAG_SKIP_MULTICAST |
									   GAA_FLAG_SKIP_DNS_SERVER |
									   GAA_FLAG_SKIP_FRIENDLY_NAME,
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


	IP_ADAPTER_ADDRESSES* adapter = addrs;

	while (adapter != NULL) {

		IP_ADAPTER_UNICAST_ADDRESS* address = adapter->FirstUnicastAddress;
		while (address != NULL) {

			char addr_chr[INET_ADDRSTRLEN];
			SOCKADDR_IN* ipv4 = reinterpret_cast<SOCKADDR_IN*>(address->Address.lpSockaddr);

			IP_Address ip;
			ip.host= *((unsigned long*)&ipv4->sin_addr);


			//inet_ntop(AF_INET, &ipv4->sin_addr, addr_chr, INET_ADDRSTRLEN);

			r_addresses->push_back(ip);

			address = address->Next;
		};
		adapter = adapter->Next;
	};

	memfree(addrs);
};

#endif

#else

void IP_Unix::get_local_addresses(List<IP_Address> *r_addresses) const {

	struct ifaddrs * ifAddrStruct=NULL;
	struct ifaddrs * ifa=NULL;

	getifaddrs(&ifAddrStruct);

	for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
		if (!ifa->ifa_addr)
			continue;
		if (ifa ->ifa_addr->sa_family==AF_INET) { // check it is IP4
			// is a valid IP4 Address

			IP_Address ip;
			ip.host= *((unsigned long*)&((struct sockaddr_in *)ifa->ifa_addr)->sin_addr);

			r_addresses->push_back(ip);
		}/* else if (ifa->ifa_addr->sa_family==AF_INET6) { // check it is IP6
			// is a valid IP6 Address
			tmpAddrPtr=&((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr;
			char addressBuffer[INET6_ADDRSTRLEN];
			inet_ntop(AF_INET6, tmpAddrPtr, addressBuffer, INET6_ADDRSTRLEN);
			printf("%s IP Address %s\n", ifa->ifa_name, addressBuffer);
		} */
	}

	if (ifAddrStruct!=NULL) freeifaddrs(ifAddrStruct);

}
#endif

void IP_Unix::make_default() {

	_create=_create_unix;
}

IP* IP_Unix::_create_unix() {

	return memnew( IP_Unix );
}

IP_Unix::IP_Unix() {
}

#endif
