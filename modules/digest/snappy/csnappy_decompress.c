/*
Copyright 2011, Google Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
  * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

File modified for the Linux Kernel by
Zeev Tarantov <zeev.tarantov@gmail.com>

File modified for Sereal by
Steffen Mueller <smueller@cpan.org>
*/

#include "csnappy_internal.h"
#ifdef __KERNEL__
#include <linux/kernel.h>
#include <linux/module.h>
#endif
#include "csnappy.h"

int
csnappy_get_uncompressed_length(
	const char *src,
	uint32_t src_len,
	uint32_t *result)
{
	const char *src_base = src;
	uint32_t shift = 0;
	uint8_t c;
	/* Length is encoded in 1..5 bytes */
	*result = 0;
	for (;;) {
		if (shift >= 32)
			goto err_out;
		if (src_len == 0)
			goto err_out;
		c = *(const uint8_t *)src++;
		src_len -= 1;
		*result |= (uint32_t)(c & 0x7f) << shift;
		if (c < 128)
			break;
		shift += 7;
	}
	return src - src_base;
err_out:
	return CSNAPPY_E_HEADER_BAD;
}
#if defined(__KERNEL__) && !defined(STATIC)
EXPORT_SYMBOL(csnappy_get_uncompressed_length);
#endif

#if defined(__arm__) && !(ARCH_ARM_HAVE_UNALIGNED)
int csnappy_decompress_noheader(
	const char	*src_,
	uint32_t	src_remaining,
	char		*dst,
	uint32_t	*dst_len)
{
	const uint8_t * src = (const uint8_t *)src_;
	const uint8_t * const src_end = src + src_remaining;
	char * const dst_base = dst;
	char * const dst_end = dst + *dst_len;
	while (src < src_end) {
		uint32_t opcode = *src++;
		uint32_t length = (opcode >> 2) + 1;
		const uint8_t *copy_src;
		if (likely((opcode & 3) == 0)) {
			if (unlikely(length > 60)) {
				uint32_t extra_bytes = length - 60;
				int shift, max_shift;
				if (unlikely(src + extra_bytes > src_end))
					return CSNAPPY_E_DATA_MALFORMED;
				length = 0;
				for (shift = 0, max_shift = extra_bytes*8;
					shift < max_shift;
					shift += 8)
					length |= *src++ << shift;
				++length;
			}
			if (unlikely(src + length > src_end))
				return CSNAPPY_E_DATA_MALFORMED;
			copy_src = src;
			src += length;
		} else {
			uint32_t offset;
			if (likely((opcode & 3) == 1)) {
				if (unlikely(src + 1 > src_end))
					return CSNAPPY_E_DATA_MALFORMED;
				length = ((length - 1) & 7) + 4;
				offset = ((opcode >> 5) << 8) + *src++;
			} else if (likely((opcode & 3) == 2)) {
				if (unlikely(src + 2 > src_end))
					return CSNAPPY_E_DATA_MALFORMED;
				offset = src[0] | (src[1] << 8);
				src += 2;
			} else {
				if (unlikely(src + 4 > src_end))
					return CSNAPPY_E_DATA_MALFORMED;
				offset = src[0] | (src[1] << 8) |
					 (src[2] << 16) | (src[3] << 24);
				src += 4;
			}
			if (unlikely(!offset || (offset > dst - dst_base)))
				return CSNAPPY_E_DATA_MALFORMED;
			copy_src = (const uint8_t *)dst - offset;
		}
		if (unlikely(dst + length > dst_end))
			return CSNAPPY_E_OUTPUT_OVERRUN;
		do *dst++ = *copy_src++; while (--length);
	}
	*dst_len = dst - dst_base;
	return CSNAPPY_E_OK;
}
#else /* !(arm with no unaligned access) */
/*
 * Data stored per entry in lookup table:
 *      Range   Bits-used       Description
 *      ------------------------------------
 *      1..64   0..7            Literal/copy length encoded in opcode byte
 *      0..7    8..10           Copy offset encoded in opcode byte / 256
 *      0..4    11..13          Extra bytes after opcode
 *
 * We use eight bits for the length even though 7 would have sufficed
 * because of efficiency reasons:
 *      (1) Extracting a byte is faster than a bit-field
 *      (2) It properly aligns copy offset so we do not need a <<8
 */
static const uint16_t char_table[256] = {
	0x0001, 0x0804, 0x1001, 0x2001, 0x0002, 0x0805, 0x1002, 0x2002,
	0x0003, 0x0806, 0x1003, 0x2003, 0x0004, 0x0807, 0x1004, 0x2004,
	0x0005, 0x0808, 0x1005, 0x2005, 0x0006, 0x0809, 0x1006, 0x2006,
	0x0007, 0x080a, 0x1007, 0x2007, 0x0008, 0x080b, 0x1008, 0x2008,
	0x0009, 0x0904, 0x1009, 0x2009, 0x000a, 0x0905, 0x100a, 0x200a,
	0x000b, 0x0906, 0x100b, 0x200b, 0x000c, 0x0907, 0x100c, 0x200c,
	0x000d, 0x0908, 0x100d, 0x200d, 0x000e, 0x0909, 0x100e, 0x200e,
	0x000f, 0x090a, 0x100f, 0x200f, 0x0010, 0x090b, 0x1010, 0x2010,
	0x0011, 0x0a04, 0x1011, 0x2011, 0x0012, 0x0a05, 0x1012, 0x2012,
	0x0013, 0x0a06, 0x1013, 0x2013, 0x0014, 0x0a07, 0x1014, 0x2014,
	0x0015, 0x0a08, 0x1015, 0x2015, 0x0016, 0x0a09, 0x1016, 0x2016,
	0x0017, 0x0a0a, 0x1017, 0x2017, 0x0018, 0x0a0b, 0x1018, 0x2018,
	0x0019, 0x0b04, 0x1019, 0x2019, 0x001a, 0x0b05, 0x101a, 0x201a,
	0x001b, 0x0b06, 0x101b, 0x201b, 0x001c, 0x0b07, 0x101c, 0x201c,
	0x001d, 0x0b08, 0x101d, 0x201d, 0x001e, 0x0b09, 0x101e, 0x201e,
	0x001f, 0x0b0a, 0x101f, 0x201f, 0x0020, 0x0b0b, 0x1020, 0x2020,
	0x0021, 0x0c04, 0x1021, 0x2021, 0x0022, 0x0c05, 0x1022, 0x2022,
	0x0023, 0x0c06, 0x1023, 0x2023, 0x0024, 0x0c07, 0x1024, 0x2024,
	0x0025, 0x0c08, 0x1025, 0x2025, 0x0026, 0x0c09, 0x1026, 0x2026,
	0x0027, 0x0c0a, 0x1027, 0x2027, 0x0028, 0x0c0b, 0x1028, 0x2028,
	0x0029, 0x0d04, 0x1029, 0x2029, 0x002a, 0x0d05, 0x102a, 0x202a,
	0x002b, 0x0d06, 0x102b, 0x202b, 0x002c, 0x0d07, 0x102c, 0x202c,
	0x002d, 0x0d08, 0x102d, 0x202d, 0x002e, 0x0d09, 0x102e, 0x202e,
	0x002f, 0x0d0a, 0x102f, 0x202f, 0x0030, 0x0d0b, 0x1030, 0x2030,
	0x0031, 0x0e04, 0x1031, 0x2031, 0x0032, 0x0e05, 0x1032, 0x2032,
	0x0033, 0x0e06, 0x1033, 0x2033, 0x0034, 0x0e07, 0x1034, 0x2034,
	0x0035, 0x0e08, 0x1035, 0x2035, 0x0036, 0x0e09, 0x1036, 0x2036,
	0x0037, 0x0e0a, 0x1037, 0x2037, 0x0038, 0x0e0b, 0x1038, 0x2038,
	0x0039, 0x0f04, 0x1039, 0x2039, 0x003a, 0x0f05, 0x103a, 0x203a,
	0x003b, 0x0f06, 0x103b, 0x203b, 0x003c, 0x0f07, 0x103c, 0x203c,
	0x0801, 0x0f08, 0x103d, 0x203d, 0x1001, 0x0f09, 0x103e, 0x203e,
	0x1801, 0x0f0a, 0x103f, 0x203f, 0x2001, 0x0f0b, 0x1040, 0x2040
};

/*
 * Copy "len" bytes from "src" to "op", one byte at a time.  Used for
 * handling COPY operations where the input and output regions may
 * overlap.  For example, suppose:
 *    src    == "ab"
 *    op     == src + 2
 *    len    == 20
 * After IncrementalCopy(src, op, len), the result will have
 * eleven copies of "ab"
 *    ababababababababababab
 * Note that this does not match the semantics of either memcpy()
 * or memmove().
 */
static INLINE void IncrementalCopy(const char *src, char *op, int len)
{
	DCHECK_GT(len, 0);
	do {
		*op++ = *src++;
	} while (--len > 0);
}

/*
 * Equivalent to IncrementalCopy except that it can write up to ten extra
 * bytes after the end of the copy, and that it is faster.
 *
 * The main part of this loop is a simple copy of eight bytes at a time until
 * we've copied (at least) the requested amount of bytes.  However, if op and
 * src are less than eight bytes apart (indicating a repeating pattern of
 * length < 8), we first need to expand the pattern in order to get the correct
 * results. For instance, if the buffer looks like this, with the eight-byte
 * <src> and <op> patterns marked as intervals:
 *
 *    abxxxxxxxxxxxx
 *    [------]           src
 *      [------]         op
 *
 * a single eight-byte copy from <src> to <op> will repeat the pattern once,
 * after which we can move <op> two bytes without moving <src>:
 *
 *    ababxxxxxxxxxx
 *    [------]           src
 *        [------]       op
 *
 * and repeat the exercise until the two no longer overlap.
 *
 * This allows us to do very well in the special case of one single byte
 * repeated many times, without taking a big hit for more general cases.
 *
 * The worst case of extra writing past the end of the match occurs when
 * op - src == 1 and len == 1; the last copy will read from byte positions
 * [0..7] and write to [4..11], whereas it was only supposed to write to
 * position 1. Thus, ten excess bytes.
 */
static const int kMaxIncrementCopyOverflow = 10;
static INLINE void IncrementalCopyFastPath(const char *src, char *op, int len)
{
	while (op - src < 8) {
		UnalignedCopy64(src, op);
		len -= op - src;
		op += op - src;
	}
	while (len > 0) {
		UnalignedCopy64(src, op);
		src += 8;
		op += 8;
		len -= 8;
	}
}


/* A type that writes to a flat array. */
struct SnappyArrayWriter {
	char *base;
	char *op;
	char *op_limit;
};

static INLINE int
SAW__AppendFastPath(struct SnappyArrayWriter *this,
		    const char *ip, uint32_t len)
{
	char *op = this->op;
	const int space_left = this->op_limit - op;
	if (likely(space_left >= 16)) {
		UnalignedCopy64(ip, op);
		UnalignedCopy64(ip + 8, op + 8);
	} else {
                if (unlikely(space_left < (int32_t)len))
			return CSNAPPY_E_OUTPUT_OVERRUN;
		memcpy(op, ip, len);
	}
	this->op = op + len;
	return CSNAPPY_E_OK;
}

static INLINE int
SAW__Append(struct SnappyArrayWriter *this,
	    const char *ip, uint32_t len)
{
	char *op = this->op;
	const int space_left = this->op_limit - op;
        if (unlikely(space_left < (int32_t)len))
		return CSNAPPY_E_OUTPUT_OVERRUN;
	memcpy(op, ip, len);
	this->op = op + len;
	return CSNAPPY_E_OK;
}

static INLINE int
SAW__AppendFromSelf(struct SnappyArrayWriter *this,
		    uint32_t offset, uint32_t len)
{
	char *op = this->op;
	const int space_left = this->op_limit - op;
	/* -1u catches offset==0 */
	if (op - this->base <= offset - 1u)
		return CSNAPPY_E_DATA_MALFORMED;
	/* Fast path, used for the majority (70-80%) of dynamic invocations. */
	if (len <= 16 && offset >= 8 && space_left >= 16) {
		UnalignedCopy64(op - offset, op);
		UnalignedCopy64(op - offset + 8, op + 8);
        } else if (space_left >= (int32_t)(len + kMaxIncrementCopyOverflow)) {
		IncrementalCopyFastPath(op - offset, op, len);
	} else {
                if (space_left < (int32_t)len)
			return CSNAPPY_E_OUTPUT_OVERRUN;
		IncrementalCopy(op - offset, op, len);
	}
	this->op = op + len;
	return CSNAPPY_E_OK;
}

int
csnappy_decompress_noheader(
	const char	*src,
	uint32_t	src_remaining,
	char		*dst,
	uint32_t	*dst_len)
{
	struct SnappyArrayWriter writer;
	const char *end_minus5 = src + src_remaining - 5;
	uint32_t length, trailer, opword, extra_bytes;
	int ret, available;
	uint8_t opcode;
	char scratch[5];
	writer.op = writer.base = dst;
	writer.op_limit = writer.op + *dst_len;
	#define LOOP_COND() \
	if (unlikely(src >= end_minus5)) {		\
		available = end_minus5 + 5 - src;	\
		if (unlikely(available <= 0))		\
			goto out;			\
		memmove(scratch, src, available);	\
		src = scratch;				\
		end_minus5 = scratch + available - 5;	\
	}
	
	LOOP_COND();
	for (;;) {
		opcode = *(const uint8_t *)src++;
		if (opcode & 0x3) {
			opword = char_table[opcode];
			extra_bytes = opword >> 11;
			trailer = get_unaligned_le(src, extra_bytes);
			length = opword & 0xff;
			src += extra_bytes;
			trailer += opword & 0x700;
			ret = SAW__AppendFromSelf(&writer, trailer, length);
			if (ret < 0)
				return ret;
			LOOP_COND();
		} else {
			length = (opcode >> 2) + 1;
			available = end_minus5 + 5 - src;
			if (length <= 16 && available >= 16) {
				if ((ret = SAW__AppendFastPath(&writer, src, length)) < 0)
					return ret;
				src += length;
				LOOP_COND();
				continue;
			}
			if (unlikely(length > 60)) {
				extra_bytes = length - 60;
				length = get_unaligned_le(src, extra_bytes) + 1;
				src += extra_bytes;
				available = end_minus5 + 5 - src;
			}
                        if (unlikely(available < (int32_t)length))
				return CSNAPPY_E_DATA_MALFORMED;
			ret = SAW__Append(&writer, src, length);
			if (ret < 0)
				return ret;
			src += length;
			LOOP_COND();
		}
	}
#undef LOOP_COND
out:
	*dst_len = writer.op - writer.base;
	return CSNAPPY_E_OK;
}
#endif /* optimized for unaligned arch */

#if defined(__KERNEL__) && !defined(STATIC)
EXPORT_SYMBOL(csnappy_decompress_noheader);
#endif

int
csnappy_decompress(
	const char *src,
	uint32_t src_len,
	char *dst,
	uint32_t dst_len)
{
	int n;
	uint32_t olen = 0;
	/* Read uncompressed length from the front of the compressed input */
	n = csnappy_get_uncompressed_length(src, src_len, &olen);
	if (unlikely(n < CSNAPPY_E_OK))
		return n;
	/* Protect against possible DoS attack */
	if (unlikely(olen > dst_len))
		return CSNAPPY_E_OUTPUT_INSUF;
	return csnappy_decompress_noheader(src + n, src_len - n, dst, &olen);
}
#if defined(__KERNEL__) && !defined(STATIC)
EXPORT_SYMBOL(csnappy_decompress);

MODULE_LICENSE("BSD");
MODULE_DESCRIPTION("Snappy Decompressor");
#endif
