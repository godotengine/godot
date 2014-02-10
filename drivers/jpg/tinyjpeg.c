/*
 * Small jpeg decoder library
 *
 * Copyright (c) 2006, Luc Saillard <luc@saillard.org>
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 * - Neither the name of the author nor the names of its contributors may be
 *  used to endorse or promote products derived from this software without
 *  specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>

#include "tinyjpeg.h"
#include "tinyjpeg-internal.h"

enum std_markers {
   DQT  = 0xDB, /* Define Quantization Table */
   SOF  = 0xC0, /* Start of Frame (size information) */
   DHT  = 0xC4, /* Huffman Table */
   SOI  = 0xD8, /* Start of Image */
   SOS  = 0xDA, /* Start of Scan */
   RST  = 0xD0, /* Reset Marker d0 -> .. */
   RST7 = 0xD7, /* Reset Marker .. -> d7 */
   EOI  = 0xD9, /* End of Image */
   DRI  = 0xDD, /* Define Restart Interval */
   APP0 = 0xE0,
};

#define cY	0
#define cCb	1
#define cCr	2

#define BLACK_Y 0
#define BLACK_U 127
#define BLACK_V 127

#if DEBUG
#define trace(fmt, args...) do { \
   fprintf(stderr, fmt, ## args); \
   fflush(stderr); \
} while(0)
#else
#define trace(fmt, ...) do { } while (0)
#endif

#define error(fmt, ...) do { \
   sprintf(error_string, fmt, ## __VA_ARGS__); \
   return -1; \
} while(0)


#if 0
static char *print_bits(unsigned int value, char *bitstr)
{
  int i, j;
  i=31;
  while (i>0)
   {
     if (value & (1UL<<i))
       break;
     i--;
   }
  j=0;
  while (i>=0)
   {
     bitstr[j++] = (value & (1UL<<i))?'1':'0';
     i--;
   }
  bitstr[j] = 0;
  return bitstr;
}

static void print_next_16bytes(int offset, const unsigned char *stream)
{
  trace("%4.4x: %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x\n",
	offset,
	stream[0], stream[1], stream[2], stream[3], 
	stream[4], stream[5], stream[6], stream[7],
	stream[8], stream[9], stream[10], stream[11], 
	stream[12], stream[13], stream[14], stream[15]);
}

#endif

/* Global variable to return the last error found while deconding */
static char error_string[256];

static const unsigned char zigzag[64] = 
{
   0,  1,  5,  6, 14, 15, 27, 28,
   2,  4,  7, 13, 16, 26, 29, 42,
   3,  8, 12, 17, 25, 30, 41, 43,
   9, 11, 18, 24, 31, 40, 44, 53,
  10, 19, 23, 32, 39, 45, 52, 54,
  20, 22, 33, 38, 46, 51, 55, 60,
  21, 34, 37, 47, 50, 56, 59, 61,
  35, 36, 48, 49, 57, 58, 62, 63
};

/* Set up the standard Huffman tables (cf. JPEG standard section K.3) */
/* IMPORTANT: these are only valid for 8-bit data precision! */
static const unsigned char bits_dc_luminance[17] =
{ 
  0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 
};
static const unsigned char val_dc_luminance[] =
{
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 
};
  
static const unsigned char bits_dc_chrominance[17] =
{
  0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 
};
static const unsigned char val_dc_chrominance[] = 
{
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 
};
  
static const unsigned char bits_ac_luminance[17] =
{
  0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d 
};
static const unsigned char val_ac_luminance[] =
{
  0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
  0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
  0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
  0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
  0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
  0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
  0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
  0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
  0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
  0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
  0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
  0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
  0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
  0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
  0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
  0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
  0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
  0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
  0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
  0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
  0xf9, 0xfa
};

static const unsigned char bits_ac_chrominance[17] =
{ 
  0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 
};

static const unsigned char val_ac_chrominance[] =
{
  0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
  0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
  0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
  0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
  0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
  0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
  0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
  0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
  0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
  0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
  0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
  0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
  0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
  0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
  0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
  0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
  0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
  0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
  0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
  0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
  0xf9, 0xfa
};


/*
 * 4 functions to manage the stream
 *
 *  fill_nbits: put at least nbits in the reservoir of bits.
 *              But convert any 0xff,0x00 into 0xff
 *  get_nbits: read nbits from the stream, and put it in result,
 *             bits is removed from the stream and the reservoir is filled
 *             automaticaly. The result is signed according to the number of
 *             bits.
 *  look_nbits: read nbits from the stream without marking as read.
 *  skip_nbits: read nbits from the stream but do not return the result.
 * 
 * stream: current pointer in the jpeg data (read bytes per bytes)
 * nbits_in_reservoir: number of bits filled into the reservoir
 * reservoir: register that contains bits information. Only nbits_in_reservoir
 *            is valid.
 *                          nbits_in_reservoir
 *                        <--    17 bits    -->
 *            Ex: 0000 0000 1010 0000 1111 0000   <== reservoir
 *                        ^
 *                        bit 1
 *            To get two bits from this example
 *                 result = (reservoir >> 15) & 3
 *
 */
#define fill_nbits(reservoir,nbits_in_reservoir,stream,nbits_wanted) do { \
   while (nbits_in_reservoir<nbits_wanted) \
    { \
      unsigned char c; \
      if (stream >= priv->stream_end) \
        longjmp(priv->jump_state, -EIO); \
      c = *stream++; \
      reservoir <<= 8; \
      if (c == 0xff && *stream == 0x00) \
        stream++; \
      reservoir |= c; \
      nbits_in_reservoir+=8; \
    } \
}  while(0);

/* Signed version !!!! */
#define get_nbits(reservoir,nbits_in_reservoir,stream,nbits_wanted,result) do { \
   fill_nbits(reservoir,nbits_in_reservoir,stream,(nbits_wanted)); \
   result = ((reservoir)>>(nbits_in_reservoir-(nbits_wanted))); \
   nbits_in_reservoir -= (nbits_wanted);  \
   reservoir &= ((1U<<nbits_in_reservoir)-1); \
   if ((unsigned int)result < (1UL<<((nbits_wanted)-1))) \
       result += (0xFFFFFFFFUL<<(nbits_wanted))+1; \
}  while(0);

#define look_nbits(reservoir,nbits_in_reservoir,stream,nbits_wanted,result) do { \
   fill_nbits(reservoir,nbits_in_reservoir,stream,(nbits_wanted)); \
   result = ((reservoir)>>(nbits_in_reservoir-(nbits_wanted))); \
}  while(0);

/* To speed up the decoding, we assume that the reservoir have enough bit 
 * slow version:
 * #define skip_nbits(reservoir,nbits_in_reservoir,stream,nbits_wanted) do { \
 *   fill_nbits(reservoir,nbits_in_reservoir,stream,(nbits_wanted)); \
 *   nbits_in_reservoir -= (nbits_wanted); \
 *   reservoir &= ((1U<<nbits_in_reservoir)-1); \
 * }  while(0);
 */
#define skip_nbits(reservoir,nbits_in_reservoir,stream,nbits_wanted) do { \
   nbits_in_reservoir -= (nbits_wanted); \
   reservoir &= ((1U<<nbits_in_reservoir)-1); \
}  while(0);


#define be16_to_cpu(x) (((x)[0]<<8)|(x)[1])

static void resync(struct jdec_private *priv);

/**
 * Get the next (valid) huffman code in the stream.
 *
 * To speedup the procedure, we look HUFFMAN_HASH_NBITS bits and the code is
 * lower than HUFFMAN_HASH_NBITS we have automaticaly the length of the code
 * and the value by using two lookup table.
 * Else if the value is not found, just search (linear) into an array for each
 * bits is the code is present.
 *
 * If the code is not present for any reason, -1 is return.
 */
static int get_next_huffman_code(struct jdec_private *priv, struct huffman_table *huffman_table)
{
  int value, hcode;
  unsigned int extra_nbits, nbits;
  uint16_t *slowtable;

  look_nbits(priv->reservoir, priv->nbits_in_reservoir, priv->stream, HUFFMAN_HASH_NBITS, hcode);
  value = huffman_table->lookup[hcode];
  if (__likely(value >= 0))
  { 
     unsigned int code_size = huffman_table->code_size[value];
     skip_nbits(priv->reservoir, priv->nbits_in_reservoir, priv->stream, code_size);
     return value;
  }

  /* Decode more bits each time ... */
  for (extra_nbits=0; extra_nbits<16-HUFFMAN_HASH_NBITS; extra_nbits++)
   {
     nbits = HUFFMAN_HASH_NBITS + 1 + extra_nbits;

     look_nbits(priv->reservoir, priv->nbits_in_reservoir, priv->stream, nbits, hcode);
     slowtable = huffman_table->slowtable[extra_nbits];
     /* Search if the code is in this array */
     while (slowtable[0]) {
	if (slowtable[0] == hcode) {
	   skip_nbits(priv->reservoir, priv->nbits_in_reservoir, priv->stream, nbits);
	   return slowtable[1];
	}
	slowtable+=2;
     }
   }
  return 0;
}




/**
 *
 * Decode a single block that contains the DCT coefficients.
 * The table coefficients is already dezigzaged at the end of the operation.
 *
 */
static void process_Huffman_data_unit(struct jdec_private *priv, int component)
{
  unsigned char j;
  unsigned int huff_code;
  unsigned char size_val, count_0;

  struct component *c = &priv->component_infos[component];
  short int DCT[64];


  /* Initialize the DCT coef table */
  memset(DCT, 0, sizeof(DCT));

  /* DC coefficient decoding */
  huff_code = get_next_huffman_code(priv, c->DC_table);
  //trace("+ %x\n", huff_code);
  if (huff_code) {
     get_nbits(priv->reservoir, priv->nbits_in_reservoir, priv->stream, huff_code, DCT[0]);
     DCT[0] += c->previous_DC;
     c->previous_DC = DCT[0];
  } else {
     DCT[0] = c->previous_DC;
  }

  /* AC coefficient decoding */
  j = 1;
  while (j<64)
   {
     huff_code = get_next_huffman_code(priv, c->AC_table);
     //trace("- %x\n", huff_code);

     size_val = huff_code & 0xF;
     count_0 = huff_code >> 4;

     if (size_val == 0)
      { /* RLE */
	if (count_0 == 0)
	  break;	/* EOB found, go out */
	else if (count_0 == 0xF)
	  j += 16;	/* skip 16 zeros */
      }
     else
      {
	j += count_0;	/* skip count_0 zeroes */
	if (__unlikely(j >= 64))
	 {
	   sprintf(error_string, "Bad huffman data (buffer overflow)");
	   break;
	 }
	get_nbits(priv->reservoir, priv->nbits_in_reservoir, priv->stream, size_val, DCT[j]);
	j++;
      }
   }

  for (j = 0; j < 64; j++)
    c->DCT[j] = DCT[zigzag[j]];
}

/*
 * Takes two array of bits, and build the huffman table for size, and code
 * 
 * lookup will return the symbol if the code is less or equal than HUFFMAN_HASH_NBITS.
 * code_size will be used to known how many bits this symbol is encoded.
 * slowtable will be used when the first lookup didn't give the result.
 */
static void build_huffman_table(const unsigned char *bits, const unsigned char *vals, struct huffman_table *table)
{
  unsigned int i, j, code, code_size, val, nbits;
  unsigned char huffsize[HUFFMAN_BITS_SIZE+1], *hz;
  unsigned int huffcode[HUFFMAN_BITS_SIZE+1], *hc;
  int next_free_entry;

  /*
   * Build a temp array 
   *   huffsize[X] => numbers of bits to write vals[X]
   */
  hz = huffsize;
  for (i=1; i<=16; i++)
   {
     for (j=1; j<=bits[i]; j++)
       *hz++ = i;
   }
  *hz = 0;

  memset(table->lookup, 0xff, sizeof(table->lookup));
  for (i=0; i<(16-HUFFMAN_HASH_NBITS); i++)
    table->slowtable[i][0] = 0;

  /* Build a temp array
   *   huffcode[X] => code used to write vals[X]
   */
  code = 0;
  hc = huffcode;
  hz = huffsize;
  nbits = *hz;
  while (*hz)
   {
     while (*hz == nbits)
      {
	*hc++ = code++;
	hz++;
      }
     code <<= 1;
     nbits++;
   }

  /*
   * Build the lookup table, and the slowtable if needed.
   */
  next_free_entry = -1;
  for (i=0; huffsize[i]; i++)
   {
     val = vals[i];
     code = huffcode[i];
     code_size = huffsize[i];

     trace("val=%2.2x code=%8.8x codesize=%2.2d\n", val, code, code_size);

     table->code_size[val] = code_size;
     if (code_size <= HUFFMAN_HASH_NBITS)
      {
	/*
	 * Good: val can be put in the lookup table, so fill all value of this
	 * column with value val 
	 */
	int repeat = 1UL<<(HUFFMAN_HASH_NBITS - code_size);
	code <<= HUFFMAN_HASH_NBITS - code_size;
	while ( repeat-- )
	  table->lookup[code++] = val;

      }
     else
      {
	/* Perhaps sorting the array will be an optimization */
	uint16_t *slowtable = table->slowtable[code_size-HUFFMAN_HASH_NBITS-1];
	while(slowtable[0])
	  slowtable+=2;
	slowtable[0] = code;
	slowtable[1] = val;
	slowtable[2] = 0;
	/* TODO: NEED TO CHECK FOR AN OVERFLOW OF THE TABLE */
      }

   }
}

static void build_default_huffman_tables(struct jdec_private *priv)
{
  if (   (priv->flags & TINYJPEG_FLAGS_MJPEG_TABLE) 
      && priv->default_huffman_table_initialized)
    return;

  build_huffman_table(bits_dc_luminance, val_dc_luminance, &priv->HTDC[0]);
  build_huffman_table(bits_ac_luminance, val_ac_luminance, &priv->HTAC[0]);

  build_huffman_table(bits_dc_chrominance, val_dc_chrominance, &priv->HTDC[1]);
  build_huffman_table(bits_ac_chrominance, val_ac_chrominance, &priv->HTAC[1]);

  priv->default_huffman_table_initialized = 1;
}



/*******************************************************************************
 *
 * Colorspace conversion routine
 *
 *
 * Note:
 * YCbCr is defined per CCIR 601-1, except that Cb and Cr are
 * normalized to the range 0..MAXJSAMPLE rather than -0.5 .. 0.5.
 * The conversion equations to be implemented are therefore
 *      R = Y                + 1.40200 * Cr
 *      G = Y - 0.34414 * Cb - 0.71414 * Cr
 *      B = Y + 1.77200 * Cb
 * 
 ******************************************************************************/

static unsigned char clamp(int i)
{
  if (i<0)
    return 0;
  else if (i>255)
    return 255;
  else
    return i;
}   


/**
 *  YCrCb -> YUV420P (1x1)
 *  .---.
 *  | 1 |
 *  `---'
 */
static void YCrCB_to_YUV420P_1x1(struct jdec_private *priv)
{
  const unsigned char *s, *y;
  unsigned char *p;
  int i,j;

  p = priv->plane[0];
  y = priv->Y;
  for (i=0; i<8; i++)
   {
     memcpy(p, y, 8);
     p+=priv->width;
     y+=8;
   }

  p = priv->plane[1];
  s = priv->Cb;
  for (i=0; i<8; i+=2)
   {
     for (j=0; j<8; j+=2, s+=2)
       *p++ = *s;
     s += 8; /* Skip one line */
     p += priv->width/2 - 4;
   }

  p = priv->plane[2];
  s = priv->Cr;
  for (i=0; i<8; i+=2)
   {
     for (j=0; j<8; j+=2, s+=2)
       *p++ = *s;
     s += 8; /* Skip one line */
     p += priv->width/2 - 4;
   }
}

/**
 *  YCrCb -> YUV420P (2x1)
 *  .-------.
 *  | 1 | 2 |
 *  `-------'
 */
static void YCrCB_to_YUV420P_2x1(struct jdec_private *priv)
{
  unsigned char *p;
  const unsigned char *s, *y1;
  unsigned int i;

  p = priv->plane[0];
  y1 = priv->Y;
  for (i=0; i<8; i++)
   {
     memcpy(p, y1, 16);
     p += priv->width;
     y1 += 16;
   }

  p = priv->plane[1];
  s = priv->Cb;
  for (i=0; i<8; i+=2)
   {
     memcpy(p, s, 8);
     s += 16; /* Skip one line */
     p += priv->width/2;
   }

  p = priv->plane[2];
  s = priv->Cr;
  for (i=0; i<8; i+=2)
   {
     memcpy(p, s, 8);
     s += 16; /* Skip one line */
     p += priv->width/2;
   }
}


/**
 *  YCrCb -> YUV420P (1x2)
 *  .---.
 *  | 1 |
 *  |---|
 *  | 2 |
 *  `---'
 */
static void YCrCB_to_YUV420P_1x2(struct jdec_private *priv)
{
  const unsigned char *s, *y;
  unsigned char *p;
  int i,j;

  p = priv->plane[0];
  y = priv->Y;
  for (i=0; i<16; i++)
   {
     memcpy(p, y, 8);
     p+=priv->width;
     y+=8;
   }

  p = priv->plane[1];
  s = priv->Cb;
  for (i=0; i<8; i++)
   {
     for (j=0; j<8; j+=2, s+=2)
       *p++ = *s;
     p += priv->width/2 - 4;
   }

  p = priv->plane[2];
  s = priv->Cr;
  for (i=0; i<8; i++)
   {
     for (j=0; j<8; j+=2, s+=2)
       *p++ = *s;
     p += priv->width/2 - 4;
   }
}

/**
 *  YCrCb -> YUV420P (2x2)
 *  .-------.
 *  | 1 | 2 |
 *  |---+---|
 *  | 3 | 4 |
 *  `-------'
 */
static void YCrCB_to_YUV420P_2x2(struct jdec_private *priv)
{
  unsigned char *p;
  const unsigned char *s, *y1;
  unsigned int i;

  p = priv->plane[0];
  y1 = priv->Y;
  for (i=0; i<16; i++)
   {
     memcpy(p, y1, 16);
     p += priv->width;
     y1 += 16;
   }

  p = priv->plane[1];
  s = priv->Cb;
  for (i=0; i<8; i++)
   {
     memcpy(p, s, 8);
     s += 8;
     p += priv->width/2;
   }

  p = priv->plane[2];
  s = priv->Cr;
  for (i=0; i<8; i++)
   {
     memcpy(p, s, 8);
     s += 8;
     p += priv->width/2;
   }
}

/**
 *  YCrCb -> RGB24 (1x1)
 *  .---.
 *  | 1 |
 *  `---'
 */
static void YCrCB_to_RGB24_1x1(struct jdec_private *priv)
{
  const unsigned char *Y, *Cb, *Cr;
  unsigned char *p;
  int i,j;
  int offset_to_next_row;

#define SCALEBITS       10
#define ONE_HALF        (1UL << (SCALEBITS-1))
#define FIX(x)          ((int)((x) * (1UL<<SCALEBITS) + 0.5))

  p = priv->plane[0];
  Y = priv->Y;
  Cb = priv->Cb;
  Cr = priv->Cr;
  offset_to_next_row = priv->width*3 - 8*3;
  for (i=0; i<8; i++) {

    for (j=0; j<8; j++) {

       int y, cb, cr;
       int add_r, add_g, add_b;
       int r, g , b;

       y  = (*Y++) << SCALEBITS;
       cb = *Cb++ - 128;
       cr = *Cr++ - 128;
       add_r = FIX(1.40200) * cr + ONE_HALF;
       add_g = - FIX(0.34414) * cb - FIX(0.71414) * cr + ONE_HALF;
       add_b = FIX(1.77200) * cb + ONE_HALF;


       r = (y + add_r) >> SCALEBITS;
       g = (y + add_g) >> SCALEBITS;
       b = (y + add_b) >> SCALEBITS;
       priv->decomp_block[i][j*3+0]=clamp(r);
       priv->decomp_block[i][j*3+1]=clamp(g);
       priv->decomp_block[i][j*3+2]=clamp(b);

    }

//    p += offset_to_next_row;
  }

#undef SCALEBITS
#undef ONE_HALF
#undef FIX

}

/**
 *  YCrCb -> BGR24 (1x1)
 *  .---.
 *  | 1 |
 *  `---'
 */
static void YCrCB_to_BGR24_1x1(struct jdec_private *priv)
{
  const unsigned char *Y, *Cb, *Cr;
  unsigned char *p;
  int i,j;
  int offset_to_next_row;

#define SCALEBITS       10
#define ONE_HALF        (1UL << (SCALEBITS-1))
#define FIX(x)          ((int)((x) * (1UL<<SCALEBITS) + 0.5))

  p = priv->plane[0];
  Y = priv->Y;
  Cb = priv->Cb;
  Cr = priv->Cr;
  offset_to_next_row = priv->width*3 - 8*3;
  for (i=0; i<8; i++) {

    for (j=0; j<8; j++) {

       int y, cb, cr;
       int add_r, add_g, add_b;
       int r, g , b;

       y  = (*Y++) << SCALEBITS;
       cb = *Cb++ - 128;
       cr = *Cr++ - 128;
       add_r = FIX(1.40200) * cr + ONE_HALF;
       add_g = - FIX(0.34414) * cb - FIX(0.71414) * cr + ONE_HALF;
       add_b = FIX(1.77200) * cb + ONE_HALF;

       b = (y + add_b) >> SCALEBITS;
       *p++ = clamp(b);
       g = (y + add_g) >> SCALEBITS;
       *p++ = clamp(g);
       r = (y + add_r) >> SCALEBITS;
       *p++ = clamp(r);

    }

    p += offset_to_next_row;
  }

#undef SCALEBITS
#undef ONE_HALF
#undef FIX

}


/**
 *  YCrCb -> RGB24 (2x1)
 *  .-------.
 *  | 1 | 2 |
 *  `-------'
 */
static void YCrCB_to_RGB24_2x1(struct jdec_private *priv)
{
  const unsigned char *Y, *Cb, *Cr;
  unsigned char *p;
  int i,j;
  int offset_to_next_row;

#define SCALEBITS       10
#define ONE_HALF        (1UL << (SCALEBITS-1))
#define FIX(x)          ((int)((x) * (1UL<<SCALEBITS) + 0.5))

  p = priv->plane[0];
  Y = priv->Y;
  Cb = priv->Cb;
  Cr = priv->Cr;
  offset_to_next_row = priv->width*3 - 16*3;
  for (i=0; i<8; i++) {

    for (j=0; j<8; j++) {

       int y, cb, cr;
       int add_r, add_g, add_b;
       int r, g , b;

       y  = (*Y++) << SCALEBITS;
       cb = *Cb++ - 128;
       cr = *Cr++ - 128;
       add_r = FIX(1.40200) * cr + ONE_HALF;
       add_g = - FIX(0.34414) * cb - FIX(0.71414) * cr + ONE_HALF;
       add_b = FIX(1.77200) * cb + ONE_HALF;

       r = (y + add_r) >> SCALEBITS;
       g = (y + add_g) >> SCALEBITS;
       b = (y + add_b) >> SCALEBITS;

       priv->decomp_block[i][j*6+0]=clamp(r);
       priv->decomp_block[i][j*6+1]=clamp(g);
       priv->decomp_block[i][j*6+2]=clamp(b);

       y  = (*Y++) << SCALEBITS;

       r = (y + add_r) >> SCALEBITS;
       g = (y + add_g) >> SCALEBITS;
       b = (y + add_b) >> SCALEBITS;

       priv->decomp_block[i][j*6+3]=clamp(r);
       priv->decomp_block[i][j*6+4]=clamp(g);
       priv->decomp_block[i][j*6+5]=clamp(b);

    }

    p += offset_to_next_row;
  }

#undef SCALEBITS
#undef ONE_HALF
#undef FIX

}

/*
 *  YCrCb -> BGR24 (2x1)
 *  .-------.
 *  | 1 | 2 |
 *  `-------'
 */
static void YCrCB_to_BGR24_2x1(struct jdec_private *priv)
{
  const unsigned char *Y, *Cb, *Cr;
  unsigned char *p;
  int i,j;
  int offset_to_next_row;

#define SCALEBITS       10
#define ONE_HALF        (1UL << (SCALEBITS-1))
#define FIX(x)          ((int)((x) * (1UL<<SCALEBITS) + 0.5))

  p = priv->plane[0];
  Y = priv->Y;
  Cb = priv->Cb;
  Cr = priv->Cr;
  offset_to_next_row = priv->width*3 - 16*3;
  for (i=0; i<8; i++) {

    for (j=0; j<8; j++) {

       int y, cb, cr;
       int add_r, add_g, add_b;
       int r, g , b;

       cb = *Cb++ - 128;
       cr = *Cr++ - 128;
       add_r = FIX(1.40200) * cr + ONE_HALF;
       add_g = - FIX(0.34414) * cb - FIX(0.71414) * cr + ONE_HALF;
       add_b = FIX(1.77200) * cb + ONE_HALF;

       y  = (*Y++) << SCALEBITS;
       b = (y + add_b) >> SCALEBITS;
       *p++ = clamp(b);
       g = (y + add_g) >> SCALEBITS;
       *p++ = clamp(g);
       r = (y + add_r) >> SCALEBITS;
       *p++ = clamp(r);

       y  = (*Y++) << SCALEBITS;
       b = (y + add_b) >> SCALEBITS;
       *p++ = clamp(b);
       g = (y + add_g) >> SCALEBITS;
       *p++ = clamp(g);
       r = (y + add_r) >> SCALEBITS;
       *p++ = clamp(r);

    }

    p += offset_to_next_row;
  }

#undef SCALEBITS
#undef ONE_HALF
#undef FIX

}

/**
 *  YCrCb -> RGB24 (1x2)
 *  .---.
 *  | 1 |
 *  |---|
 *  | 2 |
 *  `---'
 */
static void YCrCB_to_RGB24_1x2(struct jdec_private *priv)
{
  const unsigned char *Y, *Cb, *Cr;
  unsigned char *p, *p2;
  int i,j;
  int offset_to_next_row;

#define SCALEBITS       10
#define ONE_HALF        (1UL << (SCALEBITS-1))
#define FIX(x)          ((int)((x) * (1UL<<SCALEBITS) + 0.5))

  p = priv->plane[0];
  p2 = priv->plane[0] + priv->width*3;
  Y = priv->Y;
  Cb = priv->Cb;
  Cr = priv->Cr;
  offset_to_next_row = 2*priv->width*3 - 8*3;
  for (i=0; i<8; i++) {

    for (j=0; j<8; j++) {

       int y, cb, cr;
       int add_r, add_g, add_b;
       int r, g , b;

       cb = *Cb++ - 128;
       cr = *Cr++ - 128;
       add_r = FIX(1.40200) * cr + ONE_HALF;
       add_g = - FIX(0.34414) * cb - FIX(0.71414) * cr + ONE_HALF;
       add_b = FIX(1.77200) * cb + ONE_HALF;

       y  = (*Y++) << SCALEBITS;
       r = (y + add_r) >> SCALEBITS;
       g = (y + add_g) >> SCALEBITS;
       b = (y + add_b) >> SCALEBITS;

       priv->decomp_block[i*2][j*3+0]=clamp(r);
       priv->decomp_block[i*2][j*3+1]=clamp(g);
       priv->decomp_block[i*2][j*3+2]=clamp(b);

       y  = (Y[8-1]) << SCALEBITS;
       r = (y + add_r) >> SCALEBITS;
       g = (y + add_g) >> SCALEBITS;
       b = (y + add_b) >> SCALEBITS;

       priv->decomp_block[i*2+1][j*3+0]=clamp(r);
       priv->decomp_block[i*2+1][j*3+1]=clamp(g);
       priv->decomp_block[i*2+1][j*3+2]=clamp(b);

    }
    Y += 8;
    p += offset_to_next_row;
    p2 += offset_to_next_row;
  }

#undef SCALEBITS
#undef ONE_HALF
#undef FIX

}

/*
 *  YCrCb -> BGR24 (1x2)
 *  .---.
 *  | 1 |
 *  |---|
 *  | 2 |
 *  `---'
 */
static void YCrCB_to_BGR24_1x2(struct jdec_private *priv)
{
  const unsigned char *Y, *Cb, *Cr;
  unsigned char *p, *p2;
  int i,j;
  int offset_to_next_row;

#define SCALEBITS       10
#define ONE_HALF        (1UL << (SCALEBITS-1))
#define FIX(x)          ((int)((x) * (1UL<<SCALEBITS) + 0.5))

  p = priv->plane[0];
  p2 = priv->plane[0] + priv->width*3;
  Y = priv->Y;
  Cb = priv->Cb;
  Cr = priv->Cr;
  offset_to_next_row = 2*priv->width*3 - 8*3;
  for (i=0; i<8; i++) {

    for (j=0; j<8; j++) {

       int y, cb, cr;
       int add_r, add_g, add_b;
       int r, g , b;

       cb = *Cb++ - 128;
       cr = *Cr++ - 128;
       add_r = FIX(1.40200) * cr + ONE_HALF;
       add_g = - FIX(0.34414) * cb - FIX(0.71414) * cr + ONE_HALF;
       add_b = FIX(1.77200) * cb + ONE_HALF;

       y  = (*Y++) << SCALEBITS;
       b = (y + add_b) >> SCALEBITS;
       *p++ = clamp(b);
       g = (y + add_g) >> SCALEBITS;
       *p++ = clamp(g);
       r = (y + add_r) >> SCALEBITS;
       *p++ = clamp(r);

       y  = (Y[8-1]) << SCALEBITS;
       b = (y + add_b) >> SCALEBITS;
       *p2++ = clamp(b);
       g = (y + add_g) >> SCALEBITS;
       *p2++ = clamp(g);
       r = (y + add_r) >> SCALEBITS;
       *p2++ = clamp(r);

    }
    Y += 8;
    p += offset_to_next_row;
    p2 += offset_to_next_row;
  }

#undef SCALEBITS
#undef ONE_HALF
#undef FIX

}


/**
 *  YCrCb -> RGB24 (2x2)
 *  .-------.
 *  | 1 | 2 |
 *  |---+---|
 *  | 3 | 4 |
 *  `-------'
 */
static void YCrCB_to_RGB24_2x2(struct jdec_private *priv)
{
  const unsigned char *Y, *Cb, *Cr;
  unsigned char *p, *p2;
  int i,j;
  int offset_to_next_row;

#define SCALEBITS       10
#define ONE_HALF        (1UL << (SCALEBITS-1))
#define FIX(x)          ((int)((x) * (1UL<<SCALEBITS) + 0.5))

  p = priv->plane[0];
  p2 = priv->plane[0] + priv->width*3;
  Y = priv->Y;
  Cb = priv->Cb;
  Cr = priv->Cr;
  offset_to_next_row = (priv->width*3*2) - 16*3;
  for (i=0; i<8; i++) {

    for (j=0; j<8; j++) {

       int y, cb, cr;
       int add_r, add_g, add_b;
       int r, g , b;

       cb = *Cb++ - 128;
       cr = *Cr++ - 128;
       add_r = FIX(1.40200) * cr + ONE_HALF;
       add_g = - FIX(0.34414) * cb - FIX(0.71414) * cr + ONE_HALF;
       add_b = FIX(1.77200) * cb + ONE_HALF;

       y  = (*Y++) << SCALEBITS;
       r = (y + add_r) >> SCALEBITS;
       g = (y + add_g) >> SCALEBITS;
       b = (y + add_b) >> SCALEBITS;

       priv->decomp_block[i*2][j*6+0]=clamp(r);
       priv->decomp_block[i*2][j*6+1]=clamp(g);
       priv->decomp_block[i*2][j*6+2]=clamp(b);

       y  = (*Y++) << SCALEBITS;
       r = (y + add_r) >> SCALEBITS;
       g = (y + add_g) >> SCALEBITS;
       b = (y + add_b) >> SCALEBITS;

       priv->decomp_block[i*2][j*6+3]=clamp(r);
       priv->decomp_block[i*2][j*6+4]=clamp(g);
       priv->decomp_block[i*2][j*6+5]=clamp(b);

       y  = (Y[16-2]) << SCALEBITS;
       r = (y + add_r) >> SCALEBITS;
       g = (y + add_g) >> SCALEBITS;
       b = (y + add_b) >> SCALEBITS;

       priv->decomp_block[i*2+1][j*6+0]=clamp(r);
       priv->decomp_block[i*2+1][j*6+1]=clamp(g);
       priv->decomp_block[i*2+1][j*6+2]=clamp(b);

       y  = (Y[16-1]) << SCALEBITS;
       r = (y + add_r) >> SCALEBITS;
       g = (y + add_g) >> SCALEBITS;
       b = (y + add_b) >> SCALEBITS;

       priv->decomp_block[i*2+1][j*6+3]=clamp(r);
       priv->decomp_block[i*2+1][j*6+4]=clamp(g);
       priv->decomp_block[i*2+1][j*6+5]=clamp(b);

    }
    Y  += 16;
    p  += offset_to_next_row;
    p2 += offset_to_next_row;
  }

#undef SCALEBITS
#undef ONE_HALF
#undef FIX

}


/*
 *  YCrCb -> BGR24 (2x2)
 *  .-------.
 *  | 1 | 2 |
 *  |---+---|
 *  | 3 | 4 |
 *  `-------'
 */
static void YCrCB_to_BGR24_2x2(struct jdec_private *priv)
{
  const unsigned char *Y, *Cb, *Cr;
  unsigned char *p, *p2;
  int i,j;
  int offset_to_next_row;

#define SCALEBITS       10
#define ONE_HALF        (1UL << (SCALEBITS-1))
#define FIX(x)          ((int)((x) * (1UL<<SCALEBITS) + 0.5))

  p = priv->plane[0];
  p2 = priv->plane[0] + priv->width*3;
  Y = priv->Y;
  Cb = priv->Cb;
  Cr = priv->Cr;
  offset_to_next_row = (priv->width*3*2) - 16*3;
  for (i=0; i<8; i++) {

    for (j=0; j<8; j++) {

       int y, cb, cr;
       int add_r, add_g, add_b;
       int r, g , b;

       cb = *Cb++ - 128;
       cr = *Cr++ - 128;
       add_r = FIX(1.40200) * cr + ONE_HALF;
       add_g = - FIX(0.34414) * cb - FIX(0.71414) * cr + ONE_HALF;
       add_b = FIX(1.77200) * cb + ONE_HALF;

       y  = (*Y++) << SCALEBITS;
       b = (y + add_b) >> SCALEBITS;
       *p++ = clamp(b);
       g = (y + add_g) >> SCALEBITS;
       *p++ = clamp(g);
       r = (y + add_r) >> SCALEBITS;
       *p++ = clamp(r);

       y  = (*Y++) << SCALEBITS;
       b = (y + add_b) >> SCALEBITS;
       *p++ = clamp(b);
       g = (y + add_g) >> SCALEBITS;
       *p++ = clamp(g);
       r = (y + add_r) >> SCALEBITS;
       *p++ = clamp(r);

       y  = (Y[16-2]) << SCALEBITS;
       b = (y + add_b) >> SCALEBITS;
       *p2++ = clamp(b);
       g = (y + add_g) >> SCALEBITS;
       *p2++ = clamp(g);
       r = (y + add_r) >> SCALEBITS;
       *p2++ = clamp(r);

       y  = (Y[16-1]) << SCALEBITS;
       b = (y + add_b) >> SCALEBITS;
       *p2++ = clamp(b);
       g = (y + add_g) >> SCALEBITS;
       *p2++ = clamp(g);
       r = (y + add_r) >> SCALEBITS;
       *p2++ = clamp(r);
    }
    Y  += 16;
    p  += offset_to_next_row;
    p2 += offset_to_next_row;
  }

#undef SCALEBITS
#undef ONE_HALF
#undef FIX

}



/**
 *  YCrCb -> Grey (1x1)
 *  .---.
 *  | 1 |
 *  `---'
 */
static void YCrCB_to_Grey_1x1(struct jdec_private *priv)
{
  const unsigned char *y;
  unsigned char *p;
  unsigned int i;
  int offset_to_next_row;

  p = priv->plane[0];
  y = priv->Y;
  offset_to_next_row = priv->width;

  for (i=0; i<8; i++) {
     memcpy(p, y, 8);
     y+=8;
     p += offset_to_next_row;
  }
}

/**
 *  YCrCb -> Grey (2x1)
 *  .-------.
 *  | 1 | 2 |
 *  `-------'
 */
static void YCrCB_to_Grey_2x1(struct jdec_private *priv)
{
  const unsigned char *y;
  unsigned char *p;
  unsigned int i;

  p = priv->plane[0];
  y = priv->Y;

  for (i=0; i<8; i++) {
     memcpy(p, y, 16);
     y += 16;
     p += priv->width;
  }
}


/**
 *  YCrCb -> Grey (1x2)
 *  .---.
 *  | 1 |
 *  |---|
 *  | 2 |
 *  `---'
 */
static void YCrCB_to_Grey_1x2(struct jdec_private *priv)
{
  const unsigned char *y;
  unsigned char *p;
  unsigned int i;

  p = priv->plane[0];
  y = priv->Y;

  for (i=0; i<16; i++) {
     memcpy(p, y, 8);
     y += 8;
     p += priv->width;
  }
}

/**
 *  YCrCb -> Grey (2x2)
 *  .-------.
 *  | 1 | 2 |
 *  |---+---|
 *  | 3 | 4 |
 *  `-------'
 */
static void YCrCB_to_Grey_2x2(struct jdec_private *priv)
{
  const unsigned char *y;
  unsigned char *p;
  unsigned int i;

  p = priv->plane[0];
  y = priv->Y;

  for (i=0; i<16; i++) {
     memcpy(p, y, 16);
     y += 16;
     p += priv->width;
  }
}


/*
 * Decode all the 3 components for 1x1 
 */
static void decode_MCU_1x1_3planes(struct jdec_private *priv)
{
  // Y
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y, 8);
  
  // Cb
  process_Huffman_data_unit(priv, cCb);
  IDCT(&priv->component_infos[cCb], priv->Cb, 8);

  // Cr
  process_Huffman_data_unit(priv, cCr);
  IDCT(&priv->component_infos[cCr], priv->Cr, 8);
}

/*
 * Decode a 1x1 directly in 1 color
 */
static void decode_MCU_1x1_1plane(struct jdec_private *priv)
{
  // Y
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y, 8);
  
  // Cb
  process_Huffman_data_unit(priv, cCb);
  IDCT(&priv->component_infos[cCb], priv->Cb, 8);

  // Cr
  process_Huffman_data_unit(priv, cCr);
  IDCT(&priv->component_infos[cCr], priv->Cr, 8);
}


/*
 * Decode a 2x1
 *  .-------.
 *  | 1 | 2 |
 *  `-------'
 */
static void decode_MCU_2x1_3planes(struct jdec_private *priv)
{
  // Y
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y, 16);
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y+8, 16);

  // Cb
  process_Huffman_data_unit(priv, cCb);
  IDCT(&priv->component_infos[cCb], priv->Cb, 8);

  // Cr
  process_Huffman_data_unit(priv, cCr);
  IDCT(&priv->component_infos[cCr], priv->Cr, 8);
}

/*
 * Decode a 2x1
 *  .-------.
 *  | 1 | 2 |
 *  `-------'
 */
static void decode_MCU_2x1_1plane(struct jdec_private *priv)
{
  // Y
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y, 16);
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y+8, 16);

  // Cb
  process_Huffman_data_unit(priv, cCb);

  // Cr
  process_Huffman_data_unit(priv, cCr);
}


/*
 * Decode a 2x2
 *  .-------.
 *  | 1 | 2 |
 *  |---+---|
 *  | 3 | 4 |
 *  `-------'
 */
static void decode_MCU_2x2_3planes(struct jdec_private *priv)
{
  // Y
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y, 16);
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y+8, 16);
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y+64*2, 16);
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y+64*2+8, 16);

  // Cb
  process_Huffman_data_unit(priv, cCb);
  IDCT(&priv->component_infos[cCb], priv->Cb, 8);

  // Cr
  process_Huffman_data_unit(priv, cCr);
  IDCT(&priv->component_infos[cCr], priv->Cr, 8);
}

/*
 * Decode a 2x2 directly in GREY format (8bits)
 *  .-------.
 *  | 1 | 2 |
 *  |---+---|
 *  | 3 | 4 |
 *  `-------'
 */
static void decode_MCU_2x2_1plane(struct jdec_private *priv)
{
  // Y
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y, 16);
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y+8, 16);
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y+64*2, 16);
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y+64*2+8, 16);

  // Cb
  process_Huffman_data_unit(priv, cCb);

  // Cr
  process_Huffman_data_unit(priv, cCr);
}

/*
 * Decode a 1x2 mcu
 *  .---.
 *  | 1 |
 *  |---|
 *  | 2 |
 *  `---'
 */
static void decode_MCU_1x2_3planes(struct jdec_private *priv)
{
  // Y
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y, 8);
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y+64, 8);

  // Cb
  process_Huffman_data_unit(priv, cCb);
  IDCT(&priv->component_infos[cCb], priv->Cb, 8);

  // Cr
  process_Huffman_data_unit(priv, cCr);
  IDCT(&priv->component_infos[cCr], priv->Cr, 8);
}

/*
 * Decode a 1x2 mcu
 *  .---.
 *  | 1 |
 *  |---|
 *  | 2 |
 *  `---'
 */
static void decode_MCU_1x2_1plane(struct jdec_private *priv)
{
  // Y
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y, 8);
  process_Huffman_data_unit(priv, cY);
  IDCT(&priv->component_infos[cY], priv->Y+64, 8);

  // Cb
  process_Huffman_data_unit(priv, cCb);

  // Cr
  process_Huffman_data_unit(priv, cCr);
}

static void print_SOF(const unsigned char *stream)
{
  int width, height, nr_components, precision;
#if DEBUG
  const char *nr_components_to_string[] = {
     "????",
     "Grayscale",
     "????",
     "YCbCr",
     "CYMK"
  };
#endif

  precision = stream[2];
  height = be16_to_cpu(stream+3);
  width  = be16_to_cpu(stream+5);
  nr_components = stream[7];

  trace("> SOF marker\n");
  trace("Size:%dx%d nr_components:%d (%s)  precision:%d\n", 
      width, height,
      nr_components, nr_components_to_string[nr_components],
      precision);
}

/*******************************************************************************
 *
 * JPEG/JFIF Parsing functions
 *
 * Note: only a small subset of the jpeg file format is supported. No markers,
 * nor progressive stream is supported.
 *
 ******************************************************************************/

static void build_quantization_table(float *qtable, const unsigned char *ref_table)
{
  /* Taken from libjpeg. Copyright Independent JPEG Group's LLM idct.
   * For float AA&N IDCT method, divisors are equal to quantization
   * coefficients scaled by scalefactor[row]*scalefactor[col], where
   *   scalefactor[0] = 1
   *   scalefactor[k] = cos(k*PI/16) * sqrt(2)    for k=1..7
   * We apply a further scale factor of 8.
   * What's actually stored is 1/divisor so that the inner loop can
   * use a multiplication rather than a division.
   */
  int i, j;
  static const double aanscalefactor[8] = {
     1.0, 1.387039845, 1.306562965, 1.175875602,
     1.0, 0.785694958, 0.541196100, 0.275899379
  };
  const unsigned char *zz = zigzag;

  for (i=0; i<8; i++) {
     for (j=0; j<8; j++) {
       *qtable++ = ref_table[*zz++] * aanscalefactor[i] * aanscalefactor[j];
     }
   }

}

static int parse_DQT(struct jdec_private *priv, const unsigned char *stream)
{
  int qi;
  float *table;
  const unsigned char *dqt_block_end;

  trace("> DQT marker\n");
  dqt_block_end = stream + be16_to_cpu(stream);
  stream += 2;	/* Skip length */

  while (stream < dqt_block_end)
   {
     qi = *stream++;
#if SANITY_CHECK
     if (qi>>4)
       error("16 bits quantization table is not supported\n");
     if (qi>4)
       error("No more 4 quantization table is supported (got %d)\n", qi);
#endif
     table = priv->Q_tables[qi];
     build_quantization_table(table, stream);
     stream += 64;
   }
  trace("< DQT marker\n");
  return 0;
}

static int parse_SOF(struct jdec_private *priv, const unsigned char *stream)
{
  int i, width, height, nr_components, cid, sampling_factor;
  int Q_table;
  struct component *c;

  trace("> SOF marker\n");
  print_SOF(stream);

  height = be16_to_cpu(stream+3);
  width  = be16_to_cpu(stream+5);
  nr_components = stream[7];
#if SANITY_CHECK
  if (stream[2] != 8)
    error("Precision other than 8 is not supported\n");
  if (width>JPEG_MAX_WIDTH || height>JPEG_MAX_HEIGHT)
    error("Width and Height (%dx%d) seems suspicious\n", width, height);
  if (nr_components != 3)
    error("We only support YUV images\n");
  //if (height%16)
//    error("Height need to be a multiple of 16 (current height is %d)\n", height);
//  if (width%16)
  //  error("Width need to be a multiple of 16 (current Width is %d)\n", width);
#endif
  stream += 8;
  for (i=0; i<nr_components; i++) {
     cid = *stream++;
     sampling_factor = *stream++;
     Q_table = *stream++;
     c = &priv->component_infos[i];
#if SANITY_CHECK
     c->cid = cid;
     if (Q_table >= COMPONENTS)
       error("Bad Quantization table index (got %d, max allowed %d)\n", Q_table, COMPONENTS-1);
#endif
     c->Vfactor = sampling_factor&0xf;
     c->Hfactor = sampling_factor>>4;
     c->Q_table = priv->Q_tables[Q_table];
     trace("Component:%d  factor:%dx%d  Quantization table:%d\n",
           cid, c->Hfactor, c->Hfactor, Q_table );

  }
  priv->width = width;
  priv->height = height;

  trace("< SOF marker\n");

  return 0;
}

static int parse_SOS(struct jdec_private *priv, const unsigned char *stream)
{
  unsigned int i, cid, table;
  unsigned int nr_components = stream[2];

  trace("> SOS marker\n");

#if SANITY_CHECK
  if (nr_components != 3)
    error("We only support YCbCr image\n");
#endif

  stream += 3;
  for (i=0;i<nr_components;i++) {
     cid = *stream++;
     table = *stream++;
#if SANITY_CHECK
     if ((table&0xf)>=4)
	error("We do not support more than 2 AC Huffman table\n");
     if ((table>>4)>=4)
	error("We do not support more than 2 DC Huffman table\n");
     if (cid != priv->component_infos[i].cid)
        error("SOS cid order (%d:%d) isn't compatible with the SOF marker (%d:%d)\n",
	      i, cid, i, priv->component_infos[i].cid);
     trace("ComponentId:%d  tableAC:%d tableDC:%d\n", cid, table&0xf, table>>4);
#endif
     priv->component_infos[i].AC_table = &priv->HTAC[table&0xf];
     priv->component_infos[i].DC_table = &priv->HTDC[table>>4];
  }
  priv->stream = stream+3;
  trace("< SOS marker\n");
  return 0;
}

static int parse_DHT(struct jdec_private *priv, const unsigned char *stream)
{
  unsigned int count, i;
  unsigned char huff_bits[17];
  int length, index;

  length = be16_to_cpu(stream) - 2;
  stream += 2;	/* Skip length */

  trace("> DHT marker (length=%d)\n", length);

  while (length>0) {
     index = *stream++;

     /* We need to calculate the number of bytes 'vals' will takes */
     huff_bits[0] = 0;
     count = 0;
     for (i=1; i<17; i++) {
	huff_bits[i] = *stream++;
	count += huff_bits[i];
     }
#if SANITY_CHECK
     if (count >= HUFFMAN_BITS_SIZE)
       error("No more than %d bytes is allowed to describe a huffman table", HUFFMAN_BITS_SIZE);
     if ( (index &0xf) >= HUFFMAN_TABLES)
       error("No more than %d Huffman tables is supported (got %d)\n", HUFFMAN_TABLES, index&0xf);
     trace("Huffman table %s[%d] length=%d\n", (index&0xf0)?"AC":"DC", index&0xf, count);
#endif

     if (index & 0xf0 )
       build_huffman_table(huff_bits, stream, &priv->HTAC[index&0xf]);
     else
       build_huffman_table(huff_bits, stream, &priv->HTDC[index&0xf]);

     length -= 1;
     length -= 16;
     length -= count;
     stream += count;
  }
  trace("< DHT marker\n");
  return 0;
}

static int parse_DRI(struct jdec_private *priv, const unsigned char *stream)
{
  unsigned int length;

  trace("> DRI marker\n");

  length = be16_to_cpu(stream);

#if SANITY_CHECK
  if (length != 4)
    error("Length of DRI marker need to be 4\n");
#endif

  priv->restart_interval = be16_to_cpu(stream+2);

#if DEBUG
  trace("Restart interval = %d\n", priv->restart_interval);
#endif

  trace("< DRI marker\n");

  return 0;
}



static void resync(struct jdec_private *priv)
{
  int i;

  /* Init DC coefficients */
  for (i=0; i<COMPONENTS; i++)
     priv->component_infos[i].previous_DC = 0;

  priv->reservoir = 0;
  priv->nbits_in_reservoir = 0;
  if (priv->restart_interval > 0)
    priv->restarts_to_go = priv->restart_interval;
  else
    priv->restarts_to_go = -1;
}

static int find_next_rst_marker(struct jdec_private *priv)
{
  int rst_marker_found = 0;
  int marker;
  const unsigned char *stream = priv->stream;

  /* Parse marker */
  while (!rst_marker_found)
   {
     while (*stream++ != 0xff)
      {
	if (stream >= priv->stream_end)
	  error("EOF while search for a RST marker.");
      }
     /* Skip any padding ff byte (this is normal) */
     while (*stream == 0xff)
       stream++;

     marker = *stream++;
     if ((RST+priv->last_rst_marker_seen) == marker)
       rst_marker_found = 1;
     else if (marker >= RST && marker <= RST7)
       error("Wrong Reset marker found, abording");
     else if (marker == EOI)
       return 0;
   }
  trace("RST Marker %d found at offset %d\n", priv->last_rst_marker_seen, stream - priv->stream_begin);

  priv->stream = stream;
  priv->last_rst_marker_seen++;
  priv->last_rst_marker_seen &= 7;

  return 0;
}

static int parse_JFIF(struct jdec_private *priv, const unsigned char *stream)
{
  int chuck_len;
  int marker;
  int sos_marker_found = 0;
  int dht_marker_found = 0;
  const unsigned char *next_chunck;

  /* Parse marker */
  while (!sos_marker_found)
   {
     if (*stream++ != 0xff)
       goto bogus_jpeg_format;
     /* Skip any padding ff byte (this is normal) */
     while (*stream == 0xff)
       stream++;

     marker = *stream++;
     chuck_len = be16_to_cpu(stream);
     next_chunck = stream + chuck_len;
     switch (marker)
      {
       case SOF:
	 if (parse_SOF(priv, stream) < 0)
	   return -1;
	 break;
       case DQT:
	 if (parse_DQT(priv, stream) < 0)
	   return -1;
	 break;
       case SOS:
	 if (parse_SOS(priv, stream) < 0)
	   return -1;
	 sos_marker_found = 1;
	 break;
       case DHT:
	 if (parse_DHT(priv, stream) < 0)
	   return -1;
	 dht_marker_found = 1;
	 break;
       case DRI:
	 if (parse_DRI(priv, stream) < 0)
	   return -1;
	 break;
       default:
	 trace("> Unknown marker %2.2x\n", marker);
	 break;
      }

     stream = next_chunck;
   }

  if (!dht_marker_found) {
    trace("No Huffman table loaded, using the default one\n");
    build_default_huffman_tables(priv);
  }

#ifdef SANITY_CHECK
  if (   (priv->component_infos[cY].Hfactor < priv->component_infos[cCb].Hfactor)
      || (priv->component_infos[cY].Hfactor < priv->component_infos[cCr].Hfactor))
    error("Horizontal sampling factor for Y should be greater than horitontal sampling factor for Cb or Cr\n");
  if (   (priv->component_infos[cY].Vfactor < priv->component_infos[cCb].Vfactor)
      || (priv->component_infos[cY].Vfactor < priv->component_infos[cCr].Vfactor))
    error("Vertical sampling factor for Y should be greater than vertical sampling factor for Cb or Cr\n");
  if (   (priv->component_infos[cCb].Hfactor!=1) 
      || (priv->component_infos[cCr].Hfactor!=1)
      || (priv->component_infos[cCb].Vfactor!=1)
      || (priv->component_infos[cCr].Vfactor!=1))
    error("Sampling other than 1x1 for Cr and Cb is not supported");
#endif

  return 0;
bogus_jpeg_format:
  trace("Bogus jpeg format\n");
  return -1;
}

/*******************************************************************************
 *
 * Functions exported of the library.
 *
 * Note: Some applications can access directly to internal pointer of the
 * structure. It's is not recommended, but if you have many images to
 * uncompress with the same parameters, some functions can be called to speedup
 * the decoding.
 *
 ******************************************************************************/

/**
 * Allocate a new tinyjpeg decoder object.
 *
 * Before calling any other functions, an object need to be called.
 */
struct jdec_private *tinyjpeg_init(void *(*allocate_mem)(unsigned int),void (*free_mem)(void *))
{
  struct jdec_private *priv;
  unsigned int i;
 
  priv = (struct jdec_private *)allocate_mem(sizeof(struct jdec_private));
  for(i=0;i<sizeof(struct jdec_private);i++) {
	  char *pzero = (char*)priv;
	  pzero[i]=0;
  }
  priv->allocate_mem=allocate_mem;
  priv->free_mem=free_mem;
  if (priv == NULL)
    return NULL;
  return priv;
}

/**
 * Free a tinyjpeg object.
 *
 * No others function can be called after this one.
 */
void tinyjpeg_free(struct jdec_private *priv)
{
  int i;
  for (i=0; i<COMPONENTS; i++) {
     if (priv->components[i]) {
     //  priv->free_mem(priv->components[i]);
	priv->components[i] = NULL;
    }
  }
  priv->free_mem(priv);
}

/**
 * Initialize the tinyjpeg object and prepare the decoding of the stream.
 *
 * Check if the jpeg can be decoded with this jpeg decoder.
 * Fill some table used for preprocessing.
 */
int tinyjpeg_parse_header(struct jdec_private *priv, const unsigned char *buf, unsigned int size)
{
  int ret;

  /* Identify the file */
  if ((buf[0] != 0xFF) || (buf[1] != SOI))
    error("Not a JPG file ?\n");

  priv->stream_begin = buf+2;
  priv->stream_length = size-2;
  priv->stream_end = priv->stream_begin + priv->stream_length;

  ret = parse_JFIF(priv, priv->stream_begin);

  return ret;
}

static const decode_MCU_fct decode_mcu_3comp_table[4] = {
   decode_MCU_1x1_3planes,
   decode_MCU_1x2_3planes,
   decode_MCU_2x1_3planes,
   decode_MCU_2x2_3planes,
};

static const decode_MCU_fct decode_mcu_1comp_table[4] = {
   decode_MCU_1x1_1plane,
   decode_MCU_1x2_1plane,
   decode_MCU_2x1_1plane,
   decode_MCU_2x2_1plane,
};

static const convert_colorspace_fct convert_colorspace_yuv420p[4] = {
   YCrCB_to_YUV420P_1x1,
   YCrCB_to_YUV420P_1x2,
   YCrCB_to_YUV420P_2x1,
   YCrCB_to_YUV420P_2x2,
};

static const convert_colorspace_fct convert_colorspace_rgb24[4] = {
   YCrCB_to_RGB24_1x1,
   YCrCB_to_RGB24_1x2,
   YCrCB_to_RGB24_2x1,
   YCrCB_to_RGB24_2x2,
};

static const convert_colorspace_fct convert_colorspace_bgr24[4] = {
   YCrCB_to_BGR24_1x1,
   YCrCB_to_BGR24_1x2,
   YCrCB_to_BGR24_2x1,
   YCrCB_to_BGR24_2x2,
};

static const convert_colorspace_fct convert_colorspace_grey[4] = {
   YCrCB_to_Grey_1x1,
   YCrCB_to_Grey_1x2,
   YCrCB_to_Grey_2x1,
   YCrCB_to_Grey_2x2,
};

/**
 * Decode and convert the jpeg image into @pixfmt@ image
 *
 * Note: components will be automaticaly allocated if no memory is attached.
 */
int tinyjpeg_decode(struct jdec_private *priv, int pixfmt)
{
  unsigned int x, y, xstride_by_mcu, ystride_by_mcu;
  unsigned int bytes_per_blocklines[3], bytes_per_mcu[3];
  decode_MCU_fct decode_MCU;
  const decode_MCU_fct *decode_mcu_table;
  const convert_colorspace_fct *colorspace_array_conv;
  convert_colorspace_fct convert_to_pixfmt;

  if (pixfmt!=TINYJPEG_FMT_RGB24)
	  error("Only TINYJPEG_FMT_RGB24 is supported in this version");

  if (setjmp(priv->jump_state))
    return -1;

  /* To keep gcc happy initialize some array */
  bytes_per_mcu[1] = 0;
  bytes_per_mcu[2] = 0;
  bytes_per_blocklines[1] = 0;
  bytes_per_blocklines[2] = 0;

  decode_mcu_table = decode_mcu_3comp_table;
  switch (pixfmt) {
     case TINYJPEG_FMT_YUV420P:
       colorspace_array_conv = convert_colorspace_yuv420p;
       if (priv->components[0] == NULL)
	 priv->components[0] = (uint8_t *)priv->allocate_mem(priv->width * priv->height);
       if (priv->components[1] == NULL)
	 priv->components[1] = (uint8_t *)priv->allocate_mem(priv->width * priv->height/4);
       if (priv->components[2] == NULL)
	 priv->components[2] = (uint8_t *)priv->allocate_mem(priv->width * priv->height/4);
       bytes_per_blocklines[0] = priv->width;
       bytes_per_blocklines[1] = priv->width/4;
       bytes_per_blocklines[2] = priv->width/4;
       bytes_per_mcu[0] = 8;
       bytes_per_mcu[1] = 4;
       bytes_per_mcu[2] = 4;
       break;

     case TINYJPEG_FMT_RGB24:
       colorspace_array_conv = convert_colorspace_rgb24;
       if (priv->components[0] == NULL)
	 priv->components[0] = (uint8_t *)priv->allocate_mem(priv->width * priv->height * 3);
       bytes_per_blocklines[0] = priv->width * 3;
       bytes_per_mcu[0] = 3*8;
       break;

     case TINYJPEG_FMT_BGR24:
       colorspace_array_conv = convert_colorspace_bgr24;
       if (priv->components[0] == NULL)
	 priv->components[0] = (uint8_t *)priv->allocate_mem(priv->width * priv->height * 3);
       bytes_per_blocklines[0] = priv->width * 3;
       bytes_per_mcu[0] = 3*8;
       break;

     case TINYJPEG_FMT_GREY:
       decode_mcu_table = decode_mcu_1comp_table;
       colorspace_array_conv = convert_colorspace_grey;
       if (priv->components[0] == NULL)
	 priv->components[0] = (uint8_t *)priv->allocate_mem(priv->width * priv->height);
       bytes_per_blocklines[0] = priv->width;
       bytes_per_mcu[0] = 8;
       break;

     default:
       trace("Bad pixel format\n");
       return -1;
  }

  xstride_by_mcu = ystride_by_mcu = 8;
  if ((priv->component_infos[cY].Hfactor | priv->component_infos[cY].Vfactor) == 1) {
     decode_MCU = decode_mcu_table[0];
     convert_to_pixfmt = colorspace_array_conv[0];
     trace("Use decode 1x1 sampling\n");
  } else if (priv->component_infos[cY].Hfactor == 1) {
     decode_MCU = decode_mcu_table[1];
     convert_to_pixfmt = colorspace_array_conv[1];
     ystride_by_mcu = 16;
     trace("Use decode 1x2 sampling (not supported)\n");
  } else if (priv->component_infos[cY].Vfactor == 2) {
     decode_MCU = decode_mcu_table[3];
     convert_to_pixfmt = colorspace_array_conv[3];
     xstride_by_mcu = 16;
     ystride_by_mcu = 16;
     trace("Use decode 2x2 sampling\n");
  } else {
     decode_MCU = decode_mcu_table[2];
     convert_to_pixfmt = colorspace_array_conv[2];
     xstride_by_mcu = 16;
     trace("Use decode 2x1 sampling\n");
  }

  resync(priv);

  /* Don't forget to that block can be either 8 or 16 lines */
  bytes_per_blocklines[0] *= ystride_by_mcu;
  bytes_per_blocklines[1] *= ystride_by_mcu;
  bytes_per_blocklines[2] *= ystride_by_mcu;

  bytes_per_mcu[0] *= xstride_by_mcu/8;
  bytes_per_mcu[1] *= xstride_by_mcu/8;
  bytes_per_mcu[2] *= xstride_by_mcu/8;

  /* Just the decode the image by macroblock (size is 8x8, 8x16, or 16x16) */



  for (y=0; y < priv->height; y+=ystride_by_mcu)
   {
     //trace("Decoding row %d\n", y);
   ///  priv->plane[0] = priv->components[0] + (y/ystride_by_mcu * bytes_per_blocklines[0]);
   ///  priv->plane[1] = priv->components[1] + (y/ystride_by_mcu * bytes_per_blocklines[1]);
   ///  priv->plane[2] = priv->components[2] + (y/ystride_by_mcu * bytes_per_blocklines[2]);
     for (x=0; x < priv->width; x+=xstride_by_mcu)
      {
	     int i,copy_x,copy_y;
	decode_MCU(priv);
	convert_to_pixfmt(priv);
	//priv->plane[0] += bytes_per_mcu[0];
	//priv->plane[1] += bytes_per_mcu[1];
	//priv->plane[2] += bytes_per_mcu[2];

	copy_x=priv->width-x;
	if (copy_x>xstride_by_mcu)
		copy_x=xstride_by_mcu;

	copy_y=priv->height-y;
	if (copy_y>ystride_by_mcu)
		copy_y=ystride_by_mcu;

	for(i=0;i<copy_y;i++) {
		unsigned char *dst = &priv->components[0][((y+i)*priv->width+x)*3];
		memcpy(dst,&priv->decomp_block[i][0],copy_x*3);
	}

	if (priv->restarts_to_go>0)
	 {
	   priv->restarts_to_go--;
	   if (priv->restarts_to_go == 0)
	    {
	      priv->stream -= (priv->nbits_in_reservoir/8);
	      resync(priv);
	      if (find_next_rst_marker(priv) < 0)
		return -1;
	    }
	 }
      }
   }

  trace("Input file size: %d\n", priv->stream_length+2);
  trace("Input bytes actually read: %d\n", priv->stream - priv->stream_begin + 2);

  return 0;
}

const char *tinyjpeg_get_errorstring(struct jdec_private *priv)
{
  /* FIXME: the error string must be store in the context */
  priv = priv;
  return error_string;
}

void tinyjpeg_get_size(struct jdec_private *priv, unsigned int *width, unsigned int *height)
{
  *width = priv->width;
  *height = priv->height;
}

int tinyjpeg_get_components(struct jdec_private *priv, unsigned char **components)
{
  int i;
  for (i=0; priv->components[i] && i<COMPONENTS; i++)
    components[i] = priv->components[i];
  return 0;
}

int tinyjpeg_set_components(struct jdec_private *priv, unsigned char **components, unsigned int ncomponents)
{
  unsigned int i;
  if (ncomponents > COMPONENTS)
    ncomponents = COMPONENTS;
  for (i=0; i<ncomponents; i++)
    priv->components[i] = components[i];
  return 0;
}

int tinyjpeg_set_flags(struct jdec_private *priv, int flags)
{
  int oldflags = priv->flags;
  priv->flags = flags;
  return oldflags;
}

