#ifndef __TYPE_COMPAT_H
#define __TYPE_COMPAT_H

#ifndef DOC_HIDDEN
#include <stdint.h>
typedef uint8_t __u8;
typedef uint16_t __u16;
typedef uint32_t __u32;
typedef int8_t __s8;
typedef int16_t __s16;
typedef int32_t __s32;

#include <endian.h>
#include <byteswap.h>
#if __BYTE_ORDER == __LITTLE_ENDIAN
#define __cpu_to_le32(x) (x)
#define __cpu_to_be32(x) bswap_32(x)
#define __cpu_to_le16(x) (x)
#define __cpu_to_be16(x) bswap_16(x)
#else
#define __cpu_to_le32(x) bswap_32(x)
#define __cpu_to_be32(x) (x)
#define __cpu_to_le16(x) bswap_16(x)
#define __cpu_to_be16(x) (x)
#endif

#define __le32_to_cpu __cpu_to_le32
#define __be32_to_cpu __cpu_to_be32
#define __le16_to_cpu __cpu_to_le16
#define __be16_to_cpu __cpu_to_be16

#define __le64 __u64
#define __le32 __u32
#define __le16 __u16
#define __le8  __u8
#define __be64 __u64
#define __be32 __u32
#define __be16 __u16
#define __be8  __u8
#endif /* DOC_HIDDEN */

#endif /* __TYPE_COMPAT_H */
