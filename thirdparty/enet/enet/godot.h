/**
 @file  godot.h
 @brief ENet Godot header
*/
#ifndef __ENET_GODOT_H__
#define __ENET_GODOT_H__

#ifdef WINDOWS_ENABLED
#include <stdint.h>
#include <winsock2.h>
#endif
#ifdef UNIX_ENABLED
#include <arpa/inet.h>
#endif

#ifdef MSG_MAXIOVLEN
#define ENET_BUFFER_MAXIMUM MSG_MAXIOVLEN
#endif

typedef void *ENetSocket;

#define ENET_SOCKET_NULL NULL

#define ENET_HOST_TO_NET_16(value) (htons(value)) /**< macro that converts host to net byte-order of a 16-bit value */
#define ENET_HOST_TO_NET_32(value) (htonl(value)) /**< macro that converts host to net byte-order of a 32-bit value */

#define ENET_NET_TO_HOST_16(value) (ntohs(value)) /**< macro that converts net to host byte-order of a 16-bit value */
#define ENET_NET_TO_HOST_32(value) (ntohl(value)) /**< macro that converts net to host byte-order of a 32-bit value */

typedef struct
{
	void *data;
	size_t dataLength;
} ENetBuffer;

#define ENET_CALLBACK

#define ENET_API extern

typedef void ENetSocketSet;

#endif /* __ENET_GODOT_H__ */
