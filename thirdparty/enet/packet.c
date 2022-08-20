/** 
 @file  packet.c
 @brief ENet packet management functions
*/
#include <string.h>
#define ENET_BUILDING_LIB 1
#include "enet/enet.h"

/** @defgroup Packet ENet packet functions 
    @{ 
*/

/** Creates a packet that may be sent to a peer.
    @param data         initial contents of the packet's data; the packet's data will remain uninitialized if data is NULL.
    @param dataLength   size of the data allocated for this packet
    @param flags        flags for this packet as described for the ENetPacket structure.
    @returns the packet on success, NULL on failure
*/
ENetPacket *
enet_packet_create (const void * data, size_t dataLength, enet_uint32 flags)
{
    ENetPacket * packet = (ENetPacket *) enet_malloc (sizeof (ENetPacket));
    if (packet == NULL)
      return NULL;

    if (flags & ENET_PACKET_FLAG_NO_ALLOCATE)
      packet -> data = (enet_uint8 *) data;
    else
    if (dataLength <= 0)
      packet -> data = NULL;
    else
    {
       packet -> data = (enet_uint8 *) enet_malloc (dataLength);
       if (packet -> data == NULL)
       {
          enet_free (packet);
          return NULL;
       }

       if (data != NULL)
         memcpy (packet -> data, data, dataLength);
    }

    packet -> referenceCount = 0;
    packet -> flags = flags;
    packet -> dataLength = dataLength;
    packet -> freeCallback = NULL;
    packet -> userData = NULL;

    return packet;
}

/** Destroys the packet and deallocates its data.
    @param packet packet to be destroyed
*/
void
enet_packet_destroy (ENetPacket * packet)
{
    if (packet == NULL)
      return;

    if (packet -> freeCallback != NULL)
      (* packet -> freeCallback) (packet);
    if (! (packet -> flags & ENET_PACKET_FLAG_NO_ALLOCATE) &&
        packet -> data != NULL)
      enet_free (packet -> data);
    enet_free (packet);
}

/** Attempts to resize the data in the packet to length specified in the 
    dataLength parameter 
    @param packet packet to resize
    @param dataLength new size for the packet data
    @returns 0 on success, < 0 on failure
*/
int
enet_packet_resize (ENetPacket * packet, size_t dataLength)
{
    enet_uint8 * newData;
   
    if (dataLength <= packet -> dataLength || (packet -> flags & ENET_PACKET_FLAG_NO_ALLOCATE))
    {
       packet -> dataLength = dataLength;

       return 0;
    }

    newData = (enet_uint8 *) enet_malloc (dataLength);
    if (newData == NULL)
      return -1;

    memcpy (newData, packet -> data, packet -> dataLength);
    enet_free (packet -> data);
    
    packet -> data = newData;
    packet -> dataLength = dataLength;

    return 0;
}

static int initializedCRC32 = 0;
static enet_uint32 crcTable [256];

static enet_uint32 
reflect_crc (int val, int bits)
{
    int result = 0, bit;

    for (bit = 0; bit < bits; bit ++)
    {
        if(val & 1) result |= 1 << (bits - 1 - bit); 
        val >>= 1;
    }

    return result;
}

static void 
initialize_crc32 (void)
{
    int byte;

    for (byte = 0; byte < 256; ++ byte)
    {
        enet_uint32 crc = reflect_crc (byte, 8) << 24;
        int offset;

        for(offset = 0; offset < 8; ++ offset)
        {
            if (crc & 0x80000000)
                crc = (crc << 1) ^ 0x04c11db7;
            else
                crc <<= 1;
        }

        crcTable [byte] = reflect_crc (crc, 32);
    }

    initializedCRC32 = 1;
}
    
enet_uint32
enet_crc32 (const ENetBuffer * buffers, size_t bufferCount)
{
    enet_uint32 crc = 0xFFFFFFFF;
    
    if (! initializedCRC32) initialize_crc32 ();

    while (bufferCount -- > 0)
    {
        const enet_uint8 * data = (const enet_uint8 *) buffers -> data,
                         * dataEnd = & data [buffers -> dataLength];

        while (data < dataEnd)
        {
            crc = (crc >> 8) ^ crcTable [(crc & 0xFF) ^ *data++];        
        }

        ++ buffers;
    }

    return ENET_HOST_TO_NET_32 (~ crc);
}

/** @} */
