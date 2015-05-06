/* Copyright (C) 2002 Jean-Marc Valin */
/**
   @file speex_bits.h
   @brief Handles bit packing/unpacking
*/
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef BITS_H
#define BITS_H
/** @defgroup SpeexBits SpeexBits: Bit-stream manipulations
 *  This is the structure that holds the bit-stream when encoding or decoding
 * with Speex. It allows some manipulations as well.
 *  @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/** Bit-packing data structure representing (part of) a bit-stream. */
typedef struct SpeexBits {
   char *chars;   /**< "raw" data */
   int   nbBits;  /**< Total number of bits stored in the stream*/
   int   charPtr; /**< Position of the byte "cursor" */
   int   bitPtr;  /**< Position of the bit "cursor" within the current char */
   int   owner;   /**< Does the struct "own" the "raw" buffer (member "chars") */
   int   overflow;/**< Set to one if we try to read past the valid data */
   int   buf_size;/**< Allocated size for buffer */
   int   reserved1; /**< Reserved for future use */
   void *reserved2; /**< Reserved for future use */
} SpeexBits;

/** Initializes and allocates resources for a SpeexBits struct */
void speex_bits_init(SpeexBits *bits);

/** Initializes SpeexBits struct using a pre-allocated buffer*/
void speex_bits_init_buffer(SpeexBits *bits, void *buff, int buf_size);

/** Sets the bits in a SpeexBits struct to use data from an existing buffer (for decoding without copying data) */
void speex_bits_set_bit_buffer(SpeexBits *bits, void *buff, int buf_size);

/** Frees all resources associated to a SpeexBits struct. Right now this does nothing since no resources are allocated, but this could change in the future.*/
void speex_bits_destroy(SpeexBits *bits);

/** Resets bits to initial value (just after initialization, erasing content)*/
void speex_bits_reset(SpeexBits *bits);

/** Rewind the bit-stream to the beginning (ready for read) without erasing the content */
void speex_bits_rewind(SpeexBits *bits);

/** Initializes the bit-stream from the data in an area of memory */
void speex_bits_read_from(SpeexBits *bits, char *bytes, int len);

/** Append bytes to the bit-stream
 * 
 * @param bits Bit-stream to operate on
 * @param bytes pointer to the bytes what will be appended
 * @param len Number of bytes of append
 */
void speex_bits_read_whole_bytes(SpeexBits *bits, char *bytes, int len);

/** Write the content of a bit-stream to an area of memory
 * 
 * @param bits Bit-stream to operate on
 * @param bytes Memory location where to write the bits
 * @param max_len Maximum number of bytes to write (i.e. size of the "bytes" buffer)
 * @return Number of bytes written to the "bytes" buffer
*/
int speex_bits_write(SpeexBits *bits, char *bytes, int max_len);

/** Like speex_bits_write, but writes only the complete bytes in the stream. Also removes the written bytes from the stream */
int speex_bits_write_whole_bytes(SpeexBits *bits, char *bytes, int max_len);

/** Append bits to the bit-stream
 * @param bits Bit-stream to operate on
 * @param data Value to append as integer
 * @param nbBits number of bits to consider in "data"
 */
void speex_bits_pack(SpeexBits *bits, int data, int nbBits);

/** Interpret the next bits in the bit-stream as a signed integer
 *
 * @param bits Bit-stream to operate on
 * @param nbBits Number of bits to interpret
 * @return A signed integer represented by the bits read
 */
int speex_bits_unpack_signed(SpeexBits *bits, int nbBits);

/** Interpret the next bits in the bit-stream as an unsigned integer
 *
 * @param bits Bit-stream to operate on
 * @param nbBits Number of bits to interpret
 * @return An unsigned integer represented by the bits read
 */
unsigned int speex_bits_unpack_unsigned(SpeexBits *bits, int nbBits);

/** Returns the number of bytes in the bit-stream, including the last one even if it is not "full"
 *
 * @param bits Bit-stream to operate on
 * @return Number of bytes in the stream
 */
int speex_bits_nbytes(SpeexBits *bits);

/** Same as speex_bits_unpack_unsigned, but without modifying the cursor position 
 * 
 * @param bits Bit-stream to operate on
 * @param nbBits Number of bits to look for
 * @return Value of the bits peeked, interpreted as unsigned
 */
unsigned int speex_bits_peek_unsigned(SpeexBits *bits, int nbBits);

/** Get the value of the next bit in the stream, without modifying the
 * "cursor" position 
 * 
 * @param bits Bit-stream to operate on
 * @return Value of the bit peeked (one bit only)
 */
int speex_bits_peek(SpeexBits *bits);

/** Advances the position of the "bit cursor" in the stream 
 *
 * @param bits Bit-stream to operate on
 * @param n Number of bits to advance
 */
void speex_bits_advance(SpeexBits *bits, int n);

/** Returns the number of bits remaining to be read in a stream
 *
 * @param bits Bit-stream to operate on
 * @return Number of bits that can still be read from the stream
 */
int speex_bits_remaining(SpeexBits *bits);

/** Insert a terminator so that the data can be sent as a packet while auto-detecting 
 * the number of frames in each packet 
 *
 * @param bits Bit-stream to operate on
 */
void speex_bits_insert_terminator(SpeexBits *bits);

#ifdef __cplusplus
}
#endif

/* @} */
#endif
