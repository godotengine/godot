/*
 * libjingle
 * Copyright 2012, Google Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *  3. The name of the author may not be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ifaddrs-android.h"
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/utsname.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <unistd.h>
#include <errno.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
struct netlinkrequest {
  nlmsghdr header;
  ifaddrmsg msg;
};
namespace {
const int kMaxReadSize = 4096;
};
int set_ifname(struct ifaddrs* ifaddr, int interface) {
  char buf[IFNAMSIZ] = {0};
  char* name = if_indextoname(interface, buf);
  if (name == NULL) {
    return -1;
  }
  ifaddr->ifa_name = new char[strlen(name) + 1];
  strncpy(ifaddr->ifa_name, name, strlen(name) + 1);
  return 0;
}
int set_flags(struct ifaddrs* ifaddr) {
  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd == -1) {
    return -1;
  }
  ifreq ifr;
  memset(&ifr, 0, sizeof(ifr));
  strncpy(ifr.ifr_name, ifaddr->ifa_name, IFNAMSIZ - 1);
  int rc = ioctl(fd, SIOCGIFFLAGS, &ifr);
  close(fd);
  if (rc == -1) {
    return -1;
  }
  ifaddr->ifa_flags = ifr.ifr_flags;
  return 0;
}
int set_addresses(struct ifaddrs* ifaddr, ifaddrmsg* msg, void* data,
                  size_t len) {
  if (msg->ifa_family == AF_INET) {
    sockaddr_in* sa = new sockaddr_in;
    sa->sin_family = AF_INET;
    memcpy(&sa->sin_addr, data, len);
    ifaddr->ifa_addr = reinterpret_cast<sockaddr*>(sa);
  } else if (msg->ifa_family == AF_INET6) {
    sockaddr_in6* sa = new sockaddr_in6;
    sa->sin6_family = AF_INET6;
    sa->sin6_scope_id = msg->ifa_index;
    memcpy(&sa->sin6_addr, data, len);
    ifaddr->ifa_addr = reinterpret_cast<sockaddr*>(sa);
  } else {
    return -1;
  }
  return 0;
}
int make_prefixes(struct ifaddrs* ifaddr, int family, int prefixlen) {
  char* prefix = NULL;
  if (family == AF_INET) {
    sockaddr_in* mask = new sockaddr_in;
    mask->sin_family = AF_INET;
    memset(&mask->sin_addr, 0, sizeof(in_addr));
    ifaddr->ifa_netmask = reinterpret_cast<sockaddr*>(mask);
    if (prefixlen > 32) {
      prefixlen = 32;
    }
    prefix = reinterpret_cast<char*>(&mask->sin_addr);
  } else if (family == AF_INET6) {
    sockaddr_in6* mask = new sockaddr_in6;
    mask->sin6_family = AF_INET6;
    memset(&mask->sin6_addr, 0, sizeof(in6_addr));
    ifaddr->ifa_netmask = reinterpret_cast<sockaddr*>(mask);
    if (prefixlen > 128) {
      prefixlen = 128;
    }
    prefix = reinterpret_cast<char*>(&mask->sin6_addr);
  } else {
    return -1;
  }
  for (int i = 0; i < (prefixlen / 8); i++) {
    *prefix++ = 0xFF;
  }
  char remainder = 0xff;
  remainder <<= (8 - prefixlen % 8);
  *prefix = remainder;
  return 0;
}
int populate_ifaddrs(struct ifaddrs* ifaddr, ifaddrmsg* msg, void* bytes,
                     size_t len) {
  if (set_ifname(ifaddr, msg->ifa_index) != 0) {
    return -1;
  }
  if (set_flags(ifaddr) != 0) {
    return -1;
  }
  if (set_addresses(ifaddr, msg, bytes, len) != 0) {
    return -1;
  }
  if (make_prefixes(ifaddr, msg->ifa_family, msg->ifa_prefixlen) != 0) {
    return -1;
  }
  return 0;
}
int getifaddrs(struct ifaddrs** result) {
  int fd = socket(PF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
  if (fd < 0) {
    return -1;
  }
  netlinkrequest ifaddr_request;
  memset(&ifaddr_request, 0, sizeof(ifaddr_request));
  ifaddr_request.header.nlmsg_flags = NLM_F_ROOT | NLM_F_REQUEST;
  ifaddr_request.header.nlmsg_type = RTM_GETADDR;
  ifaddr_request.header.nlmsg_len = NLMSG_LENGTH(sizeof(ifaddrmsg));
  ssize_t count = send(fd, &ifaddr_request, ifaddr_request.header.nlmsg_len, 0);
  if (static_cast<size_t>(count) != ifaddr_request.header.nlmsg_len) {
    close(fd);
    return -1;
  }
  struct ifaddrs* start = NULL;
  struct ifaddrs* current = NULL;
  char buf[kMaxReadSize];
  ssize_t amount_read = recv(fd, &buf, kMaxReadSize, 0);
  while (amount_read > 0) {
    nlmsghdr* header = reinterpret_cast<nlmsghdr*>(&buf[0]);
    size_t header_size = static_cast<size_t>(amount_read);
    for ( ; NLMSG_OK(header, header_size);
          header = NLMSG_NEXT(header, header_size)) {
      switch (header->nlmsg_type) {
        case NLMSG_DONE:
          // Success. Return.
          *result = start;
          close(fd);
          return 0;
        case NLMSG_ERROR:
          close(fd);
          freeifaddrs(start);
          return -1;
        case RTM_NEWADDR: {
          ifaddrmsg* address_msg =
              reinterpret_cast<ifaddrmsg*>(NLMSG_DATA(header));
          rtattr* rta = IFA_RTA(address_msg);
          ssize_t payload_len = IFA_PAYLOAD(header);
          while (RTA_OK(rta, payload_len)) {
            if (rta->rta_type == IFA_ADDRESS) {
              int family = address_msg->ifa_family;
              if (family == AF_INET || family == AF_INET6) {
                ifaddrs* newest = new ifaddrs;
                memset(newest, 0, sizeof(ifaddrs));
                if (current) {
                  current->ifa_next = newest;
                } else {
                  start = newest;
                }
                if (populate_ifaddrs(newest, address_msg, RTA_DATA(rta),
                                     RTA_PAYLOAD(rta)) != 0) {
                  freeifaddrs(start);
                  *result = NULL;
                  return -1;
                }
                current = newest;
              }
            }
            rta = RTA_NEXT(rta, payload_len);
          }
          break;
        }
      }
    }
    amount_read = recv(fd, &buf, kMaxReadSize, 0);
  }
  close(fd);
  freeifaddrs(start);
  return -1;
}
void freeifaddrs(struct ifaddrs* addrs) {
  struct ifaddrs* last = NULL;
  struct ifaddrs* cursor = addrs;
  while (cursor) {
    delete[] cursor->ifa_name;
    delete cursor->ifa_addr;
    delete cursor->ifa_netmask;
    last = cursor;
    cursor = cursor->ifa_next;
    delete last;
  }
}
