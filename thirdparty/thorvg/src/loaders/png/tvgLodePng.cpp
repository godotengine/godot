/*
 * Copyright (c) 2020 - 2024 the ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
  LodePNG version 20200306

  Copyright (c) 2005-2020 Lode Vandevenne

  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

    3. This notice may not be removed or altered from any sourcedistribution.
*/

#include <cstdlib>
#include "tvgLodePng.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

#if defined(_MSC_VER) && (_MSC_VER >= 1310) /*Visual Studio: A few warning types are not desired here.*/
    #pragma warning( disable : 4244 ) /*implicit conversions: not warned by gcc -Wall -Wextra and requires too much casts*/
    #pragma warning( disable : 4996 ) /*VS does not like fopen, but fopen_s is not standard C so unusable here*/
#endif /*_MSC_VER */


/* convince the compiler to inline a function, for use when this measurably improves performance */
/* inline is not available in C90, but use it when supported by the compiler */
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)) || (defined(__cplusplus) && (__cplusplus >= 199711L))
    #define LODEPNG_INLINE inline
#else
    #define LODEPNG_INLINE /* not available */
#endif

/* restrict is not available in C90, but use it when supported by the compiler */
#if (defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))) ||\
    (defined(_MSC_VER) && (_MSC_VER >= 1400)) || \
    (defined(__WATCOMC__) && (__WATCOMC__ >= 1250) && !defined(__cplusplus))
    #define LODEPNG_RESTRICT __restrict
#else
    #define LODEPNG_RESTRICT /* not available */
#endif

#define LODEPNG_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LODEPNG_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define LODEPNG_ABS(x) ((x) < 0 ? -(x) : (x))


/* Replacements for C library functions such as memcpy and strlen, to support platforms
where a full C library is not available. The compiler can recognize them and compile
to something as fast. */

static void lodepng_memcpy(void* LODEPNG_RESTRICT dst, const void* LODEPNG_RESTRICT src, size_t size)
{
    size_t i;
    for (i = 0; i < size; i++) ((char*)dst)[i] = ((const char*)src)[i];
}


static void lodepng_memset(void* LODEPNG_RESTRICT dst, int value, size_t num)
{
    size_t i;
    for (i = 0; i < num; i++) ((char*)dst)[i] = (char)value;
}


/* does not check memory out of bounds, do not use on untrusted data */
static size_t lodepng_strlen(const char* a)
{
    const char* orig = a;
    /* avoid warning about unused function in case of disabled COMPILE... macros */
    (void)(&lodepng_strlen);
    while (*a) a++;
    return (size_t)(a - orig);
}


/* Safely check if adding two integers will overflow (no undefined
behavior, compiler removing the code, etc...) and output result. */
static int lodepng_addofl(size_t a, size_t b, size_t* result)
{
    *result = a + b; /* Unsigned addition is well defined and safe in C90 */
    return *result < a;
}


/* Safely check if multiplying two integers will overflow (no undefined
behavior, compiler removing the code, etc...) and output result. */
static int lodepng_mulofl(size_t a, size_t b, size_t* result)
{
    *result = a * b; /* Unsigned multiplication is well defined and safe in C90 */
    return (a != 0 && *result / a != b);
}


/* Safely check if a + b > c, even if overflow could happen. */
static int lodepng_gtofl(size_t a, size_t b, size_t c)
{
    size_t d;
    if (lodepng_addofl(a, b, &d)) return 1;
    return d > c;
}


/*
    Often in case of an error a value is assigned to a variable and then it breaks
    out of a loop (to go to the cleanup phase of a function). This macro does that.
    It makes the error handling code shorter and more readable.

    Example: if(!uivector_resize(&lz77_encoded, datasize)) ERROR_BREAK(83);
*/
#define CERROR_BREAK(errorvar, code){\
  errorvar = code;\
  break;\
}

/* version of CERROR_BREAK that assumes the common case where the error variable is named "error" */
#define ERROR_BREAK(code) CERROR_BREAK(error, code)

/* Set error var to the error code, and return it.*/
#define CERROR_RETURN_ERROR(errorvar, code){\
  errorvar = code;\
  return code;\
}

/* Try the code, if it returns error, also return the error. */
#define CERROR_TRY_RETURN(call){\
  unsigned error = call;\
  if(error) return error;\
}

/* Set error var to the error code, and return from the void function. */
#define CERROR_RETURN(errorvar, code){\
  errorvar = code;\
  return;\
}


/* dynamic vector of unsigned chars */
struct ucvector
{
    unsigned char* data;
    size_t size; /*used size*/
    size_t allocsize; /*allocated size*/
};


/* returns 1 if success, 0 if failure ==> nothing done */
static unsigned ucvector_resize(ucvector* p, size_t size)
{
    if (size > p->allocsize) {
        size_t newsize = size + (p->allocsize >> 1u);
        void* data = realloc(p->data, newsize);
        if(data) {
            p->allocsize = newsize;
            p->data = (unsigned char*)data;
        }
        else return 0; /*error: not enough memory*/
    }
    p->size = size;
    return 1; /*success*/
}


static ucvector ucvector_init(unsigned char* buffer, size_t size)
{
    ucvector v;
    v.data = buffer;
    v.allocsize = v.size = size;
    return v;
}


static unsigned lodepng_read32bitInt(const unsigned char* buffer)
{
    return (((unsigned)buffer[0] << 24u) | ((unsigned)buffer[1] << 16u) | ((unsigned)buffer[2] << 8u) | (unsigned)buffer[3]);
}


/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */
/* // End of common code and tools. Begin of Zlib related code.            // */
/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */

struct LodePNGBitReader
{
    const unsigned char* data;
    size_t size; /*size of data in bytes*/
    size_t bitsize; /*size of data in bits, end of valid bp values, should be 8*size*/
    size_t bp;
    unsigned buffer; /*buffer for reading bits. NOTE: 'unsigned' must support at least 32 bits*/
};


/* data size argument is in bytes. Returns error if size too large causing overflow */
static unsigned LodePNGBitReader_init(LodePNGBitReader* reader, const unsigned char* data, size_t size)
{
    size_t temp;
    reader->data = data;
    reader->size = size;
    /* size in bits, return error if overflow (if size_t is 32 bit this supports up to 500MB)  */
    if (lodepng_mulofl(size, 8u, &reader->bitsize)) return 105;
    /*ensure incremented bp can be compared to bitsize without overflow even when it would be incremented 32 too much and
    trying to ensure 32 more bits*/
    if (lodepng_addofl(reader->bitsize, 64u, &temp)) return 105;
    reader->bp = 0;
    reader->buffer = 0;
    return 0; /*ok*/
  }

/*
  ensureBits functions:
  Ensures the reader can at least read nbits bits in one or more readBits calls,
  safely even if not enough bits are available.
  Returns 1 if there are enough bits available, 0 if not.
*/

/*See ensureBits documentation above. This one ensures exactly 1 bit */
/*static unsigned ensureBits1(LodePNGBitReader* reader) {
  if(reader->bp >= reader->bitsize) return 0;
  reader->buffer = (unsigned)reader->data[reader->bp >> 3u] >> (reader->bp & 7u);
  return 1;
}*/

/*See ensureBits documentation above. This one ensures up to 9 bits */
static unsigned ensureBits9(LodePNGBitReader* reader, size_t nbits)
{
    size_t start = reader->bp >> 3u;
    size_t size = reader->size;
    if (start + 1u < size) {
        reader->buffer = (unsigned)reader->data[start + 0] | ((unsigned)reader->data[start + 1] << 8u);
        reader->buffer >>= (reader->bp & 7u);
        return 1;
    } else {
        reader->buffer = 0;
        if (start + 0u < size) reader->buffer |= reader->data[start + 0];
        reader->buffer >>= (reader->bp & 7u);
        return reader->bp + nbits <= reader->bitsize;
    }
}


/*See ensureBits documentation above. This one ensures up to 17 bits */
static unsigned ensureBits17(LodePNGBitReader* reader, size_t nbits)
{
    size_t start = reader->bp >> 3u;
    size_t size = reader->size;
    if (start + 2u < size) {
        reader->buffer = (unsigned)reader->data[start + 0] | ((unsigned)reader->data[start + 1] << 8u) | ((unsigned)reader->data[start + 2] << 16u);
        reader->buffer >>= (reader->bp & 7u);
        return 1;
    } else {
        reader->buffer = 0;
        if (start + 0u < size) reader->buffer |= reader->data[start + 0];
        if (start + 1u < size) reader->buffer |= ((unsigned)reader->data[start + 1] << 8u);
        reader->buffer >>= (reader->bp & 7u);
        return reader->bp + nbits <= reader->bitsize;
    }
}


/*See ensureBits documentation above. This one ensures up to 25 bits */
static LODEPNG_INLINE unsigned ensureBits25(LodePNGBitReader* reader, size_t nbits)
{
    size_t start = reader->bp >> 3u;
    size_t size = reader->size;
    if (start + 3u < size) {
        reader->buffer = (unsigned)reader->data[start + 0] | ((unsigned)reader->data[start + 1] << 8u) |  ((unsigned)reader->data[start + 2] << 16u) | ((unsigned)reader->data[start + 3] << 24u);
        reader->buffer >>= (reader->bp & 7u);
        return 1;
    } else {
        reader->buffer = 0;
        if (start + 0u < size) reader->buffer |= reader->data[start + 0];
        if (start + 1u < size) reader->buffer |= ((unsigned)reader->data[start + 1] << 8u);
        if (start + 2u < size) reader->buffer |= ((unsigned)reader->data[start + 2] << 16u);
        reader->buffer >>= (reader->bp & 7u);
        return reader->bp + nbits <= reader->bitsize;
    }
}


/*See ensureBits documentation above. This one ensures up to 32 bits */
static LODEPNG_INLINE unsigned ensureBits32(LodePNGBitReader* reader, size_t nbits)
{
    size_t start = reader->bp >> 3u;
    size_t size = reader->size;
    if(start + 4u < size) {
        reader->buffer = (unsigned)reader->data[start + 0] | ((unsigned)reader->data[start + 1] << 8u) | ((unsigned)reader->data[start + 2] << 16u) | ((unsigned)reader->data[start + 3] << 24u);
        reader->buffer >>= (reader->bp & 7u);
        reader->buffer |= (((unsigned)reader->data[start + 4] << 24u) << (8u - (reader->bp & 7u)));
        return 1;
    } else {
        reader->buffer = 0;
        if (start + 0u < size) reader->buffer |= reader->data[start + 0];
        if (start + 1u < size) reader->buffer |= ((unsigned)reader->data[start + 1] << 8u);
        if (start + 2u < size) reader->buffer |= ((unsigned)reader->data[start + 2] << 16u);
        if (start + 3u < size) reader->buffer |= ((unsigned)reader->data[start + 3] << 24u);
        reader->buffer >>= (reader->bp & 7u);
        return reader->bp + nbits <= reader->bitsize;
    }
}


/* Get bits without advancing the bit pointer. Must have enough bits available with ensureBits. Max nbits is 31. */
static unsigned peekBits(LodePNGBitReader* reader, size_t nbits)
{
    /* The shift allows nbits to be only up to 31. */
    return reader->buffer & ((1u << nbits) - 1u);
}


/* Must have enough bits available with ensureBits */
static void advanceBits(LodePNGBitReader* reader, size_t nbits)
{
    reader->buffer >>= nbits;
    reader->bp += nbits;
}


/* Must have enough bits available with ensureBits */
static unsigned readBits(LodePNGBitReader* reader, size_t nbits)
{
    unsigned result = peekBits(reader, nbits);
    advanceBits(reader, nbits);
    return result;
}


/* Public for testing only. steps and result must have numsteps values. */
unsigned lode_png_test_bitreader(const unsigned char* data, size_t size, size_t numsteps, const size_t* steps, unsigned* result)
{
    size_t i;
    LodePNGBitReader reader;
    unsigned error = LodePNGBitReader_init(&reader, data, size);
    if (error) return 0;
    for (i = 0; i < numsteps; i++) {
        size_t step = steps[i];
        unsigned ok;
        if (step > 25) ok = ensureBits32(&reader, step);
        else if (step > 17) ok = ensureBits25(&reader, step);
        else if (step > 9) ok = ensureBits17(&reader, step);
        else ok = ensureBits9(&reader, step);
        if (!ok) return 0;
        result[i] = readBits(&reader, step);
    }
    return 1;
}


static unsigned reverseBits(unsigned bits, unsigned num)
{
    /*TODO: implement faster lookup table based version when needed*/
    unsigned i, result = 0;
    for (i = 0; i < num; i++) result |= ((bits >> (num - i - 1u)) & 1u) << i;
    return result;
}

/* ////////////////////////////////////////////////////////////////////////// */
/* / Deflate - Huffman                                                      / */
/* ////////////////////////////////////////////////////////////////////////// */

#define FIRST_LENGTH_CODE_INDEX 257
#define LAST_LENGTH_CODE_INDEX 285
/*256 literals, the end code, some length codes, and 2 unused codes*/
#define NUM_DEFLATE_CODE_SYMBOLS 288
/*the distance codes have their own symbols, 30 used, 2 unused*/
#define NUM_DISTANCE_SYMBOLS 32
/*the code length codes. 0-15: code lengths, 16: copy previous 3-6 times, 17: 3-10 zeros, 18: 11-138 zeros*/
#define NUM_CODE_LENGTH_CODES 19

/*the base lengths represented by codes 257-285*/
static const unsigned LENGTHBASE[29]
  = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
     67, 83, 99, 115, 131, 163, 195, 227, 258};

/*the extra bits used by codes 257-285 (added to base length)*/
static const unsigned LENGTHEXTRA[29]
  = {0, 0, 0, 0, 0, 0, 0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,
      4,  4,  4,   4,   5,   5,   5,   5,   0};

/*the base backwards distances (the bits of distance codes appear after length codes and use their own huffman tree)*/
static const unsigned DISTANCEBASE[30]
  = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
     769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};

/*the extra bits of backwards distances (added to base)*/
static const unsigned DISTANCEEXTRA[30]
  = {0, 0, 0, 0, 1, 1, 2,  2,  3,  3,  4,  4,  5,  5,   6,   6,   7,   7,   8,
       8,    9,    9,   10,   10,   11,   11,   12,    12,    13,    13};

/*the order in which "code length alphabet code lengths" are stored as specified by deflate, out of this the huffman
tree of the dynamic huffman tree lengths is generated*/
static const unsigned CLCL_ORDER[NUM_CODE_LENGTH_CODES]
  = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

/* ////////////////////////////////////////////////////////////////////////// */

/*
Huffman tree struct, containing multiple representations of the tree
*/
struct HuffmanTree
{
    unsigned* codes; /*the huffman codes (bit patterns representing the symbols)*/
    unsigned* lengths; /*the lengths of the huffman codes*/
    unsigned maxbitlen; /*maximum number of bits a single code can get*/
    unsigned numcodes; /*number of symbols in the alphabet = number of codes*/
    /* for reading only */
    unsigned char* table_len; /*length of symbol from lookup table, or max length if secondary lookup needed*/
    unsigned short* table_value; /*value of symbol from lookup table, or pointer to secondary table if needed*/
};


static void HuffmanTree_init(HuffmanTree* tree)
{
    tree->codes = 0;
    tree->lengths = 0;
    tree->table_len = 0;
    tree->table_value = 0;
}


static void HuffmanTree_cleanup(HuffmanTree* tree)
{
    free(tree->codes);
    free(tree->lengths);
    free(tree->table_len);
    free(tree->table_value);
}


/* amount of bits for first huffman table lookup (aka root bits), see HuffmanTree_makeTable and huffmanDecodeSymbol.*/
/* values 8u and 9u work the fastest */
#define FIRSTBITS 9u

/* a symbol value too big to represent any valid symbol, to indicate reading disallowed huffman bits combination,
which is possible in case of only 0 or 1 present symbols. */
#define INVALIDSYMBOL 65535u

/* make table for huffman decoding */
static unsigned HuffmanTree_makeTable(HuffmanTree* tree)
{
    static const unsigned headsize = 1u << FIRSTBITS; /*size of the first table*/
    static const unsigned mask = (1u << FIRSTBITS) /*headsize*/ - 1u;
    size_t i, numpresent, pointer, size; /*total table size*/
    unsigned* maxlens = (unsigned*)malloc(headsize * sizeof(unsigned));
    if (!maxlens) return 83; /*alloc fail*/

    /* compute maxlens: max total bit length of symbols sharing prefix in the first table*/
    lodepng_memset(maxlens, 0, headsize * sizeof(*maxlens));
    for (i = 0; i < tree->numcodes; i++) {
        unsigned symbol = tree->codes[i];
        unsigned l = tree->lengths[i];
        unsigned index;
        if(l <= FIRSTBITS) continue; /*symbols that fit in first table don't increase secondary table size*/
        /*get the FIRSTBITS MSBs, the MSBs of the symbol are encoded first. See later comment about the reversing*/
        index = reverseBits(symbol >> (l - FIRSTBITS), FIRSTBITS);
        maxlens[index] = LODEPNG_MAX(maxlens[index], l);
    }
    /* compute total table size: size of first table plus all secondary tables for symbols longer than FIRSTBITS */
    size = headsize;
    for (i = 0; i < headsize; ++i) {
        unsigned l = maxlens[i];
        if (l > FIRSTBITS) size += (1u << (l - FIRSTBITS));
    }
    tree->table_len = (unsigned char*)malloc(size * sizeof(*tree->table_len));
    tree->table_value = (unsigned short*)malloc(size * sizeof(*tree->table_value));
    if (!tree->table_len || !tree->table_value) {
        free(maxlens);
        /* freeing tree->table values is done at a higher scope */
        return 83; /*alloc fail*/
    }
    /*initialize with an invalid length to indicate unused entries*/
    for (i = 0; i < size; ++i) tree->table_len[i] = 16;

    /*fill in the first table for long symbols: max prefix size and pointer to secondary tables*/
    pointer = headsize;
    for (i = 0; i < headsize; ++i) {
        unsigned l = maxlens[i];
        if(l <= FIRSTBITS) continue;
        tree->table_len[i] = l;
        tree->table_value[i] = pointer;
        pointer += (1u << (l - FIRSTBITS));
    }
    free(maxlens);

    /*fill in the first table for short symbols, or secondary table for long symbols*/
    numpresent = 0;
    for (i = 0; i < tree->numcodes; ++i) {
        unsigned l = tree->lengths[i];
        unsigned symbol = tree->codes[i]; /*the huffman bit pattern. i itself is the value.*/
        /*reverse bits, because the huffman bits are given in MSB first order but the bit reader reads LSB first*/
        unsigned reverse = reverseBits(symbol, l);
        if (l == 0) continue;
        numpresent++;

        if (l <= FIRSTBITS) {
            /*short symbol, fully in first table, replicated num times if l < FIRSTBITS*/
            unsigned num = 1u << (FIRSTBITS - l);
            unsigned j;
            for (j = 0; j < num; ++j) {
                /*bit reader will read the l bits of symbol first, the remaining FIRSTBITS - l bits go to the MSB's*/
                unsigned index = reverse | (j << l);
                if(tree->table_len[index] != 16) return 55; /*invalid tree: long symbol shares prefix with short symbol*/
                tree->table_len[index] = l;
                tree->table_value[index] = i;
            }
        } else {
            /*long symbol, shares prefix with other long symbols in first lookup table, needs second lookup*/
            /*the FIRSTBITS MSBs of the symbol are the first table index*/
            unsigned index = reverse & mask;
            unsigned maxlen = tree->table_len[index];
            /*log2 of secondary table length, should be >= l - FIRSTBITS*/
            unsigned tablelen = maxlen - FIRSTBITS;
            unsigned start = tree->table_value[index]; /*starting index in secondary table*/
            unsigned num = 1u << (tablelen - (l - FIRSTBITS)); /*amount of entries of this symbol in secondary table*/
            unsigned j;
            if (maxlen < l) return 55; /*invalid tree: long symbol shares prefix with short symbol*/
            for (j = 0; j < num; ++j) {
                unsigned reverse2 = reverse >> FIRSTBITS; /* l - FIRSTBITS bits */
                unsigned index2 = start + (reverse2 | (j << (l - FIRSTBITS)));
                tree->table_len[index2] = l;
                tree->table_value[index2] = i;
            }
        }
    }

    if (numpresent < 2) {
        /* In case of exactly 1 symbol, in theory the huffman symbol needs 0 bits,
        but deflate uses 1 bit instead. In case of 0 symbols, no symbols can
        appear at all, but such huffman tree could still exist (e.g. if distance
        codes are never used). In both cases, not all symbols of the table will be
        filled in. Fill them in with an invalid symbol value so returning them from
        huffmanDecodeSymbol will cause error. */
        for (i = 0; i < size; ++i) {
            if (tree->table_len[i] == 16) {
                /* As length, use a value smaller than FIRSTBITS for the head table,
                and a value larger than FIRSTBITS for the secondary table, to ensure
                valid behavior for advanceBits when reading this symbol. */
                tree->table_len[i] = (i < headsize) ? 1 : (FIRSTBITS + 1);
                tree->table_value[i] = INVALIDSYMBOL;
            }
        }
    } else {
        /* A good huffman tree has N * 2 - 1 nodes, of which N - 1 are internal nodes.
        If that is not the case (due to too long length codes), the table will not
        have been fully used, and this is an error (not all bit combinations can be
        decoded): an oversubscribed huffman tree, indicated by error 55. */
        for (i = 0; i < size; ++i) {
            if (tree->table_len[i] == 16) return 55;
        }
    }
    return 0;
}


/*
  Second step for the ...makeFromLengths and ...makeFromFrequencies functions.
  numcodes, lengths and maxbitlen must already be filled in correctly. return
  value is error.
*/
static unsigned HuffmanTree_makeFromLengths2(HuffmanTree* tree)
{
    unsigned* blcount;
    unsigned* nextcode;
    unsigned error = 0;
    unsigned bits, n;

    tree->codes = (unsigned*)malloc(tree->numcodes * sizeof(unsigned));
    blcount = (unsigned*)malloc((tree->maxbitlen + 1) * sizeof(unsigned));
    nextcode = (unsigned*)malloc((tree->maxbitlen + 1) * sizeof(unsigned));
    if (!tree->codes || !blcount || !nextcode) error = 83; /*alloc fail*/

    if (!error) {
        for (n = 0; n != tree->maxbitlen + 1; n++) blcount[n] = nextcode[n] = 0;
        /*step 1: count number of instances of each code length*/
        for (bits = 0; bits != tree->numcodes; ++bits) ++blcount[tree->lengths[bits]];
        /*step 2: generate the nextcode values*/
        for(bits = 1; bits <= tree->maxbitlen; ++bits) {
            nextcode[bits] = (nextcode[bits - 1] + blcount[bits - 1]) << 1u;
        }
        /*step 3: generate all the codes*/
        for (n = 0; n != tree->numcodes; ++n) {
            if (tree->lengths[n] != 0) {
                tree->codes[n] = nextcode[tree->lengths[n]]++;
                /*remove superfluous bits from the code*/
                tree->codes[n] &= ((1u << tree->lengths[n]) - 1u);
            }
        }
    }

    free(blcount);
    free(nextcode);

    if (!error) error = HuffmanTree_makeTable(tree);
    return error;
}


/*
  given the code lengths (as stored in the PNG file), generate the tree as defined
  by Deflate. maxbitlen is the maximum bits that a code in the tree can have.
  return value is error.
*/
static unsigned HuffmanTree_makeFromLengths(HuffmanTree* tree, const unsigned* bitlen, size_t numcodes, unsigned maxbitlen)
{
    unsigned i;
    tree->lengths = (unsigned*)malloc(numcodes * sizeof(unsigned));
    if (!tree->lengths) return 83; /*alloc fail*/
    for (i = 0; i != numcodes; ++i) tree->lengths[i] = bitlen[i];
    tree->numcodes = (unsigned)numcodes; /*number of symbols*/
    tree->maxbitlen = maxbitlen;
    return HuffmanTree_makeFromLengths2(tree);
}


/*get the literal and length code tree of a deflated block with fixed tree, as per the deflate specification*/
static unsigned generateFixedLitLenTree(HuffmanTree* tree)
{
    unsigned i, error = 0;
    unsigned* bitlen = (unsigned*)malloc(NUM_DEFLATE_CODE_SYMBOLS * sizeof(unsigned));
    if (!bitlen) return 83; /*alloc fail*/

    /*288 possible codes: 0-255=literals, 256=endcode, 257-285=lengthcodes, 286-287=unused*/
    for (i =   0; i <= 143; ++i) bitlen[i] = 8;
    for (i = 144; i <= 255; ++i) bitlen[i] = 9;
    for (i = 256; i <= 279; ++i) bitlen[i] = 7;
    for (i = 280; i <= 287; ++i) bitlen[i] = 8;

    error = HuffmanTree_makeFromLengths(tree, bitlen, NUM_DEFLATE_CODE_SYMBOLS, 15);

    free(bitlen);
    return error;
}


/*get the distance code tree of a deflated block with fixed tree, as specified in the deflate specification*/
static unsigned generateFixedDistanceTree(HuffmanTree* tree)
{
    unsigned i, error = 0;
    unsigned* bitlen = (unsigned*)malloc(NUM_DISTANCE_SYMBOLS * sizeof(unsigned));
    if (!bitlen) return 83; /*alloc fail*/

    /*there are 32 distance codes, but 30-31 are unused*/
    for (i = 0; i != NUM_DISTANCE_SYMBOLS; ++i) bitlen[i] = 5;
    error = HuffmanTree_makeFromLengths(tree, bitlen, NUM_DISTANCE_SYMBOLS, 15);

    free(bitlen);
    return error;
}


/*
  returns the code. The bit reader must already have been ensured at least 15 bits
*/
static unsigned huffmanDecodeSymbol(LodePNGBitReader* reader, const HuffmanTree* codetree)
{
    unsigned short code = peekBits(reader, FIRSTBITS);
    unsigned short l = codetree->table_len[code];
    unsigned short value = codetree->table_value[code];
    if (l <= FIRSTBITS) {
        advanceBits(reader, l);
        return value;
    } else {
        unsigned index2;
        advanceBits(reader, FIRSTBITS);
        index2 = value + peekBits(reader, l - FIRSTBITS);
        advanceBits(reader, codetree->table_len[index2] - FIRSTBITS);
        return codetree->table_value[index2];
    }
}


/* ////////////////////////////////////////////////////////////////////////// */
/* / Inflator (Decompressor)                                                / */
/* ////////////////////////////////////////////////////////////////////////// */

/*get the tree of a deflated block with fixed tree, as specified in the deflate specification
Returns error code.*/
static unsigned getTreeInflateFixed(HuffmanTree* tree_ll, HuffmanTree* tree_d)
{
    unsigned error = generateFixedLitLenTree(tree_ll);
    if (error) return error;
    return generateFixedDistanceTree(tree_d);
}


/*get the tree of a deflated block with dynamic tree, the tree itself is also Huffman compressed with a known tree*/
static unsigned getTreeInflateDynamic(HuffmanTree* tree_ll, HuffmanTree* tree_d, LodePNGBitReader* reader)
{
    /*make sure that length values that aren't filled in will be 0, or a wrong tree will be generated*/
    unsigned error = 0;
    unsigned n, HLIT, HDIST, HCLEN, i;

    /*see comments in deflateDynamic for explanation of the context and these variables, it is analogous*/
    unsigned* bitlen_ll = 0; /*lit,len code lengths*/
    unsigned* bitlen_d = 0; /*dist code lengths*/
    /*code length code lengths ("clcl"), the bit lengths of the huffman tree used to compress bitlen_ll and bitlen_d*/
    unsigned* bitlen_cl = 0;
    HuffmanTree tree_cl; /*the code tree for code length codes (the huffman tree for compressed huffman trees)*/

    if (!ensureBits17(reader, 14)) return 49; /*error: the bit pointer is or will go past the memory*/

    /*number of literal/length codes + 257. Unlike the spec, the value 257 is added to it here already*/
    HLIT =  readBits(reader, 5) + 257;
    /*number of distance codes. Unlike the spec, the value 1 is added to it here already*/
    HDIST = readBits(reader, 5) + 1;
    /*number of code length codes. Unlike the spec, the value 4 is added to it here already*/
    HCLEN = readBits(reader, 4) + 4;

    bitlen_cl = (unsigned*)malloc(NUM_CODE_LENGTH_CODES * sizeof(unsigned));
    if(!bitlen_cl) return 83 /*alloc fail*/;

    HuffmanTree_init(&tree_cl);

    while (!error) {
        /*read the code length codes out of 3 * (amount of code length codes) bits*/
        if (lodepng_gtofl(reader->bp, HCLEN * 3, reader->bitsize)) {
            ERROR_BREAK(50); /*error: the bit pointer is or will go past the memory*/
        }
        for (i = 0; i != HCLEN; ++i) {
            ensureBits9(reader, 3); /*out of bounds already checked above */
            bitlen_cl[CLCL_ORDER[i]] = readBits(reader, 3);
        }
        for (i = HCLEN; i != NUM_CODE_LENGTH_CODES; ++i) {
            bitlen_cl[CLCL_ORDER[i]] = 0;
        }

        error = HuffmanTree_makeFromLengths(&tree_cl, bitlen_cl, NUM_CODE_LENGTH_CODES, 7);
        if(error) break;

        /*now we can use this tree to read the lengths for the tree that this function will return*/
        bitlen_ll = (unsigned*)malloc(NUM_DEFLATE_CODE_SYMBOLS * sizeof(unsigned));
        bitlen_d = (unsigned*)malloc(NUM_DISTANCE_SYMBOLS * sizeof(unsigned));
        if (!bitlen_ll || !bitlen_d) ERROR_BREAK(83 /*alloc fail*/);
        lodepng_memset(bitlen_ll, 0, NUM_DEFLATE_CODE_SYMBOLS * sizeof(*bitlen_ll));
        lodepng_memset(bitlen_d, 0, NUM_DISTANCE_SYMBOLS * sizeof(*bitlen_d));

        /*i is the current symbol we're reading in the part that contains the code lengths of lit/len and dist codes*/
        i = 0;
        while (i < HLIT + HDIST) {
            unsigned code;
            ensureBits25(reader, 22); /* up to 15 bits for huffman code, up to 7 extra bits below*/
            code = huffmanDecodeSymbol(reader, &tree_cl);
            if (code <= 15) /*a length code*/ {
                if (i < HLIT) bitlen_ll[i] = code;
                else bitlen_d[i - HLIT] = code;
                ++i;
            } else if (code == 16) /*repeat previous*/ {
                unsigned replength = 3; /*read in the 2 bits that indicate repeat length (3-6)*/
                unsigned value; /*set value to the previous code*/

                if (i == 0) ERROR_BREAK(54); /*can't repeat previous if i is 0*/

                replength += readBits(reader, 2);

                if (i < HLIT + 1) value = bitlen_ll[i - 1];
                else value = bitlen_d[i - HLIT - 1];
                /*repeat this value in the next lengths*/
                for (n = 0; n < replength; ++n) {
                    if (i >= HLIT + HDIST) ERROR_BREAK(13); /*error: i is larger than the amount of codes*/
                    if (i < HLIT) bitlen_ll[i] = value;
                    else bitlen_d[i - HLIT] = value;
                    ++i;
                }
            } else if(code == 17) /*repeat "0" 3-10 times*/ {
                unsigned replength = 3; /*read in the bits that indicate repeat length*/
                replength += readBits(reader, 3);

                /*repeat this value in the next lengths*/
                for (n = 0; n < replength; ++n) {
                    if (i >= HLIT + HDIST) ERROR_BREAK(14); /*error: i is larger than the amount of codes*/

                    if (i < HLIT) bitlen_ll[i] = 0;
                    else bitlen_d[i - HLIT] = 0;
                    ++i;
                }
            } else if(code == 18) /*repeat "0" 11-138 times*/ {
                unsigned replength = 11; /*read in the bits that indicate repeat length*/
                replength += readBits(reader, 7);

                /*repeat this value in the next lengths*/
                for (n = 0; n < replength; ++n) {
                    if(i >= HLIT + HDIST) ERROR_BREAK(15); /*error: i is larger than the amount of codes*/

                    if(i < HLIT) bitlen_ll[i] = 0;
                    else bitlen_d[i - HLIT] = 0;
                    ++i;
                }
            } else /*if(code == INVALIDSYMBOL)*/ {
                ERROR_BREAK(16); /*error: tried to read disallowed huffman symbol*/
            }
            /*check if any of the ensureBits above went out of bounds*/
            if (reader->bp > reader->bitsize) {
                /*return error code 10 or 11 depending on the situation that happened in huffmanDecodeSymbol
                (10=no endcode, 11=wrong jump outside of tree)*/
                /* TODO: revise error codes 10,11,50: the above comment is no longer valid */
                ERROR_BREAK(50); /*error, bit pointer jumps past memory*/
            }
        }
        if (error) break;

        if (bitlen_ll[256] == 0) ERROR_BREAK(64); /*the length of the end code 256 must be larger than 0*/

        /*now we've finally got HLIT and HDIST, so generate the code trees, and the function is done*/
        error = HuffmanTree_makeFromLengths(tree_ll, bitlen_ll, NUM_DEFLATE_CODE_SYMBOLS, 15);
        if (error) break;
        error = HuffmanTree_makeFromLengths(tree_d, bitlen_d, NUM_DISTANCE_SYMBOLS, 15);

        break; /*end of error-while*/
    }

    free(bitlen_cl);
    free(bitlen_ll);
    free(bitlen_d);
    HuffmanTree_cleanup(&tree_cl);

    return error;
}


/*inflate a block with dynamic of fixed Huffman tree. btype must be 1 or 2.*/
static unsigned inflateHuffmanBlock(ucvector* out, LodePNGBitReader* reader, unsigned btype)
{
    unsigned error = 0;
    HuffmanTree tree_ll; /*the huffman tree for literal and length codes*/
    HuffmanTree tree_d; /*the huffman tree for distance codes*/

    HuffmanTree_init(&tree_ll);
    HuffmanTree_init(&tree_d);

    if (btype == 1) error = getTreeInflateFixed(&tree_ll, &tree_d);
    else /*if(btype == 2)*/ error = getTreeInflateDynamic(&tree_ll, &tree_d, reader);

    while (!error) /*decode all symbols until end reached, breaks at end code*/ {
        /*code_ll is literal, length or end code*/
        unsigned code_ll;
        ensureBits25(reader, 20); /* up to 15 for the huffman symbol, up to 5 for the length extra bits */
        code_ll = huffmanDecodeSymbol(reader, &tree_ll);
        if (code_ll <= 255) /*literal symbol*/ {
            if (!ucvector_resize(out, out->size + 1)) ERROR_BREAK(83 /*alloc fail*/);
            out->data[out->size - 1] = (unsigned char)code_ll;
        } else if (code_ll >= FIRST_LENGTH_CODE_INDEX && code_ll <= LAST_LENGTH_CODE_INDEX) /*length code*/ {
            unsigned code_d, distance;
            unsigned numextrabits_l, numextrabits_d; /*extra bits for length and distance*/
            size_t start, backward, length;

            /*part 1: get length base*/
            length = LENGTHBASE[code_ll - FIRST_LENGTH_CODE_INDEX];

            /*part 2: get extra bits and add the value of that to length*/
            numextrabits_l = LENGTHEXTRA[code_ll - FIRST_LENGTH_CODE_INDEX];
            if (numextrabits_l != 0) {
                /* bits already ensured above */
                length += readBits(reader, numextrabits_l);
            }

            /*part 3: get distance code*/
            ensureBits32(reader, 28); /* up to 15 for the huffman symbol, up to 13 for the extra bits */
            code_d = huffmanDecodeSymbol(reader, &tree_d);
            if (code_d > 29) {
                if (code_d <= 31) {
                    ERROR_BREAK(18); /*error: invalid distance code (30-31 are never used)*/
                } else /* if(code_d == INVALIDSYMBOL) */{
                    ERROR_BREAK(16); /*error: tried to read disallowed huffman symbol*/
                }
            }
            distance = DISTANCEBASE[code_d];

            /*part 4: get extra bits from distance*/
            numextrabits_d = DISTANCEEXTRA[code_d];
            if (numextrabits_d != 0) {
                /* bits already ensured above */
                distance += readBits(reader, numextrabits_d);
            }

            /*part 5: fill in all the out[n] values based on the length and dist*/
            start = out->size;
            if (distance > start) ERROR_BREAK(52); /*too long backward distance*/
            backward = start - distance;

            if (!ucvector_resize(out, out->size + length)) ERROR_BREAK(83 /*alloc fail*/);
            if (distance < length) {
                size_t forward;
                lodepng_memcpy(out->data + start, out->data + backward, distance);
                start += distance;
                for (forward = distance; forward < length; ++forward) {
                  out->data[start++] = out->data[backward++];
                }
            } else {
                lodepng_memcpy(out->data + start, out->data + backward, length);
            }
        } else if (code_ll == 256) {
            break; /*end code, break the loop*/
        } else /*if(code_ll == INVALIDSYMBOL)*/ {
            ERROR_BREAK(16); /*error: tried to read disallowed huffman symbol*/
        }
        /*check if any of the ensureBits above went out of bounds*/
        if (reader->bp > reader->bitsize) {
          /*return error code 10 or 11 depending on the situation that happened in huffmanDecodeSymbol
          (10=no endcode, 11=wrong jump outside of tree)*/
          /* TODO: revise error codes 10,11,50: the above comment is no longer valid */
          ERROR_BREAK(51); /*error, bit pointer jumps past memory*/
        }
    }

    HuffmanTree_cleanup(&tree_ll);
    HuffmanTree_cleanup(&tree_d);

    return error;
}


static unsigned inflateNoCompression(ucvector* out, LodePNGBitReader* reader, const LodePNGDecompressSettings* settings)
{
    size_t bytepos;
    size_t size = reader->size;
    unsigned LEN, NLEN, error = 0;

    /*go to first boundary of byte*/
    bytepos = (reader->bp + 7u) >> 3u;

    /*read LEN (2 bytes) and NLEN (2 bytes)*/
    if (bytepos + 4 >= size) return 52; /*error, bit pointer will jump past memory*/
    LEN = (unsigned)reader->data[bytepos] + ((unsigned)reader->data[bytepos + 1] << 8u); bytepos += 2;
    NLEN = (unsigned)reader->data[bytepos] + ((unsigned)reader->data[bytepos + 1] << 8u); bytepos += 2;

    /*check if 16-bit NLEN is really the one's complement of LEN*/
    if (!settings->ignore_nlen && LEN + NLEN != 65535) {
        return 21; /*error: NLEN is not one's complement of LEN*/
    }

    if (!ucvector_resize(out, out->size + LEN)) return 83; /*alloc fail*/

    /*read the literal data: LEN bytes are now stored in the out buffer*/
    if (bytepos + LEN > size) return 23; /*error: reading outside of in buffer*/

    lodepng_memcpy(out->data + out->size - LEN, reader->data + bytepos, LEN);
    bytepos += LEN;

    reader->bp = bytepos << 3u;

    return error;
}


static unsigned lodepng_inflatev(ucvector* out, const unsigned char* in, size_t insize, const LodePNGDecompressSettings* settings)
{
    unsigned BFINAL = 0;
    LodePNGBitReader reader;
    unsigned error = LodePNGBitReader_init(&reader, in, insize);

    if (error) return error;

    while (!BFINAL) {
        unsigned BTYPE;
        if (!ensureBits9(&reader, 3)) return 52; /*error, bit pointer will jump past memory*/
        BFINAL = readBits(&reader, 1);
        BTYPE = readBits(&reader, 2);

        if (BTYPE == 3) return 20; /*error: invalid BTYPE*/
        else if (BTYPE == 0) error = inflateNoCompression(out, &reader, settings); /*no compression*/
        else error = inflateHuffmanBlock(out, &reader, BTYPE); /*compression, BTYPE 01 or 10*/

        if (error) return error;
    }

    return error;
}


static unsigned inflatev(ucvector* out, const unsigned char* in, size_t insize, const LodePNGDecompressSettings* settings)
{
    if (settings->custom_inflate) {
        unsigned error = settings->custom_inflate(&out->data, &out->size, in, insize, settings);
        out->allocsize = out->size;
        return error;
    } else {
        return lodepng_inflatev(out, in, insize, settings);
    }
}


/* ////////////////////////////////////////////////////////////////////////// */
/* / Adler32                                                                / */
/* ////////////////////////////////////////////////////////////////////////// */

static unsigned update_adler32(unsigned adler, const unsigned char* data, unsigned len)
{
    unsigned s1 = adler & 0xffffu;
    unsigned s2 = (adler >> 16u) & 0xffffu;

    while (len != 0u) {
        unsigned i;
        /*at least 5552 sums can be done before the sums overflow, saving a lot of module divisions*/
        unsigned amount = len > 5552u ? 5552u : len;
        len -= amount;
        for (i = 0; i != amount; ++i) {
            s1 += (*data++);
            s2 += s1;
        }
        s1 %= 65521u;
        s2 %= 65521u;
    }

    return (s2 << 16u) | s1;
}

/*Return the adler32 of the bytes data[0..len-1]*/
static unsigned adler32(const unsigned char* data, unsigned len)
{
    return update_adler32(1u, data, len);
}

/* ////////////////////////////////////////////////////////////////////////// */
/* / Zlib                                                                   / */
/* ////////////////////////////////////////////////////////////////////////// */

static unsigned lodepng_zlib_decompressv(ucvector* out, const unsigned char* in, size_t insize, const LodePNGDecompressSettings* settings)
{
    unsigned error = 0;
    unsigned CM, CINFO, FDICT;

    if (insize < 2) return 53; /*error, size of zlib data too small*/
    /*read information from zlib header*/
    if ((in[0] * 256 + in[1]) % 31 != 0) {
        /*error: 256 * in[0] + in[1] must be a multiple of 31, the FCHECK value is supposed to be made that way*/
        return 24;
    }

    CM = in[0] & 15;
    CINFO = (in[0] >> 4) & 15;
    /*FCHECK = in[1] & 31;*/ /*FCHECK is already tested above*/
    FDICT = (in[1] >> 5) & 1;
    /*FLEVEL = (in[1] >> 6) & 3;*/ /*FLEVEL is not used here*/

    if (CM != 8 || CINFO > 7) {
        /*error: only compression method 8: inflate with sliding window of 32k is supported by the PNG spec*/
        return 25;
    }
    if (FDICT != 0) {
        /*error: the specification of PNG says about the zlib stream:
          "The additional flags shall not specify a preset dictionary."*/
        return 26;
    }

    error = inflatev(out, in + 2, insize - 2, settings);
    if (error) return error;

    if (!settings->ignore_adler32) {
        unsigned ADLER32 = lodepng_read32bitInt(&in[insize - 4]);
        unsigned checksum = adler32(out->data, (unsigned)(out->size));
        if(checksum != ADLER32) return 58; /*error, adler checksum not correct, data must be corrupted*/
    }

    return 0; /*no error*/
}


/*expected_size is expected output size, to avoid intermediate allocations. Set to 0 if not known. */
static unsigned zlib_decompress(unsigned char** out, size_t* outsize, size_t expected_size, const unsigned char* in, size_t insize, const LodePNGDecompressSettings* settings)
{
    if(settings->custom_zlib) {
        return settings->custom_zlib(out, outsize, in, insize, settings);
    } else {
        unsigned error;
        ucvector v = ucvector_init(*out, *outsize);
        if (expected_size) {
            /*reserve the memory to avoid intermediate reallocations*/
            ucvector_resize(&v, *outsize + expected_size);
            v.size = *outsize;
        }
        error = lodepng_zlib_decompressv(&v, in, insize, settings);
        *out = v.data;
        *outsize = v.size;
        return error;
    }
}


static void lodepng_decompress_settings_init(LodePNGDecompressSettings* settings)
{
    settings->ignore_adler32 = 0;
    settings->ignore_nlen = 0;
    settings->custom_zlib = 0;
    settings->custom_inflate = 0;
    settings->custom_context = 0;
}


/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */
/* // End of Zlib related code. Begin of PNG related code.                 // */
/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */


#if 0 //thorvg don't use crc
/* CRC polynomial: 0xedb88320 */
static unsigned lodepng_crc32_table[256] = {
            0u, 1996959894u, 3993919788u, 2567524794u,  124634137u, 1886057615u, 3915621685u, 2657392035u,
    249268274u, 2044508324u, 3772115230u, 2547177864u,  162941995u, 2125561021u, 3887607047u, 2428444049u,
    498536548u, 1789927666u, 4089016648u, 2227061214u,  450548861u, 1843258603u, 4107580753u, 2211677639u,
    325883990u, 1684777152u, 4251122042u, 2321926636u,  335633487u, 1661365465u, 4195302755u, 2366115317u,
    997073096u, 1281953886u, 3579855332u, 2724688242u, 1006888145u, 1258607687u, 3524101629u, 2768942443u,
    901097722u, 1119000684u, 3686517206u, 2898065728u,  853044451u, 1172266101u, 3705015759u, 2882616665u,
    651767980u, 1373503546u, 3369554304u, 3218104598u,  565507253u, 1454621731u, 3485111705u, 3099436303u,
    671266974u, 1594198024u, 3322730930u, 2970347812u,  795835527u, 1483230225u, 3244367275u, 3060149565u,
    1994146192u,   31158534u, 2563907772u, 4023717930u, 1907459465u,  112637215u, 2680153253u, 3904427059u,
    2013776290u,  251722036u, 2517215374u, 3775830040u, 2137656763u,  141376813u, 2439277719u, 3865271297u,
    1802195444u,  476864866u, 2238001368u, 4066508878u, 1812370925u,  453092731u, 2181625025u, 4111451223u,
    1706088902u,  314042704u, 2344532202u, 4240017532u, 1658658271u,  366619977u, 2362670323u, 4224994405u,
    1303535960u,  984961486u, 2747007092u, 3569037538u, 1256170817u, 1037604311u, 2765210733u, 3554079995u,
    1131014506u,  879679996u, 2909243462u, 3663771856u, 1141124467u,  855842277u, 2852801631u, 3708648649u,
    1342533948u,  654459306u, 3188396048u, 3373015174u, 1466479909u,  544179635u, 3110523913u, 3462522015u,
    1591671054u,  702138776u, 2966460450u, 3352799412u, 1504918807u,  783551873u, 3082640443u, 3233442989u,
    3988292384u, 2596254646u,   62317068u, 1957810842u, 3939845945u, 2647816111u,   81470997u, 1943803523u,
    3814918930u, 2489596804u,  225274430u, 2053790376u, 3826175755u, 2466906013u,  167816743u, 2097651377u,
    4027552580u, 2265490386u,  503444072u, 1762050814u, 4150417245u, 2154129355u,  426522225u, 1852507879u,
    4275313526u, 2312317920u,  282753626u, 1742555852u, 4189708143u, 2394877945u,  397917763u, 1622183637u,
    3604390888u, 2714866558u,  953729732u, 1340076626u, 3518719985u, 2797360999u, 1068828381u, 1219638859u,
    3624741850u, 2936675148u,  906185462u, 1090812512u, 3747672003u, 2825379669u,  829329135u, 1181335161u,
    3412177804u, 3160834842u,  628085408u, 1382605366u, 3423369109u, 3138078467u,  570562233u, 1426400815u,
    3317316542u, 2998733608u,  733239954u, 1555261956u, 3268935591u, 3050360625u,  752459403u, 1541320221u,
    2607071920u, 3965973030u, 1969922972u,   40735498u, 2617837225u, 3943577151u, 1913087877u,   83908371u,
    2512341634u, 3803740692u, 2075208622u,  213261112u, 2463272603u, 3855990285u, 2094854071u,  198958881u,
    2262029012u, 4057260610u, 1759359992u,  534414190u, 2176718541u, 4139329115u, 1873836001u,  414664567u,
    2282248934u, 4279200368u, 1711684554u,  285281116u, 2405801727u, 4167216745u, 1634467795u,  376229701u,
    2685067896u, 3608007406u, 1308918612u,  956543938u, 2808555105u, 3495958263u, 1231636301u, 1047427035u,
    2932959818u, 3654703836u, 1088359270u,  936918000u, 2847714899u, 3736837829u, 1202900863u,  817233897u,
    3183342108u, 3401237130u, 1404277552u,  615818150u, 3134207493u, 3453421203u, 1423857449u,  601450431u,
    3009837614u, 3294710456u, 1567103746u,  711928724u, 3020668471u, 3272380065u, 1510334235u,  755167117u
};


/* Calculate CRC32 of buffer
   Return the CRC of the bytes buf[0..len-1]. */
static unsigned lodepng_crc32(const unsigned char* data, size_t length)
{
    unsigned r = 0xffffffffu;
    size_t i;
    for (i = 0; i < length; ++i) {
        r = lodepng_crc32_table[(r ^ data[i]) & 0xffu] ^ (r >> 8u);
    }
    return r ^ 0xffffffffu;
}
#endif

/* ////////////////////////////////////////////////////////////////////////// */
/* / Reading and writing PNG color channel bits                             / */
/* ////////////////////////////////////////////////////////////////////////// */

/* The color channel bits of less-than-8-bit pixels are read with the MSB of bytes first,
so LodePNGBitWriter and LodePNGBitReader can't be used for those. */

static unsigned char readBitFromReversedStream(size_t* bitpointer, const unsigned char* bitstream)
{
    unsigned char result = (unsigned char)((bitstream[(*bitpointer) >> 3] >> (7 - ((*bitpointer) & 0x7))) & 1);
    ++(*bitpointer);
    return result;
}


/* TODO: make this faster */
static unsigned readBitsFromReversedStream(size_t* bitpointer, const unsigned char* bitstream, size_t nbits)
{
    unsigned result = 0;
    size_t i;
    for (i = 0 ; i < nbits; ++i) {
        result <<= 1u;
        result |= (unsigned)readBitFromReversedStream(bitpointer, bitstream);
    }
    return result;
}


static void setBitOfReversedStream(size_t* bitpointer, unsigned char* bitstream, unsigned char bit)
{
    /*the current bit in bitstream may be 0 or 1 for this to work*/
    if (bit == 0) bitstream[(*bitpointer) >> 3u] &=  (unsigned char)(~(1u << (7u - ((*bitpointer) & 7u))));
    else bitstream[(*bitpointer) >> 3u] |=  (1u << (7u - ((*bitpointer) & 7u)));
    ++(*bitpointer);
}

/* ////////////////////////////////////////////////////////////////////////// */
/* / PNG chunks                                                             / */
/* ////////////////////////////////////////////////////////////////////////// */

/*
  The lodepng_chunk functions are normally not needed, except to traverse the
  unknown chunks stored in the LodePNGInfo struct, or add new ones to it.
  It also allows traversing the chunks of an encoded PNG file yourself.

  The chunk pointer always points to the beginning of the chunk itself, that is
  the first byte of the 4 length bytes.

  In the PNG file format, chunks have the following format:
  -4 bytes length: length of the data of the chunk in bytes (chunk itself is 12 bytes longer)
  -4 bytes chunk type (ASCII a-z,A-Z only, see below)
  -length bytes of data (may be 0 bytes if length was 0)
  -4 bytes of CRC, computed on chunk name + data

  The first chunk starts at the 8th byte of the PNG file, the entire rest of the file
  exists out of concatenated chunks with the above format.

  PNG standard chunk ASCII naming conventions:
  -First byte: uppercase = critical, lowercase = ancillary
  -Second byte: uppercase = public, lowercase = private
  -Third byte: must be uppercase
  -Fourth byte: uppercase = unsafe to copy, lowercase = safe to copy
*/


/*
  Gets the length of the data of the chunk. Total chunk length has 12 bytes more.
  There must be at least 4 bytes to read from. If the result value is too large,
  it may be corrupt data.
*/
static unsigned lodepng_chunk_length(const unsigned char* chunk)
{
    return lodepng_read32bitInt(&chunk[0]);
}


/* check if the type is the given type */
static unsigned char lodepng_chunk_type_equals(const unsigned char* chunk, const char* type)
{
    if (lodepng_strlen(type) != 4) return 0;
    return (chunk[4] == type[0] && chunk[5] == type[1] && chunk[6] == type[2] && chunk[7] == type[3]);
}


/* 0: it's one of the critical chunk types, 1: it's an ancillary chunk (see PNG standard) */
static unsigned char lodepng_chunk_ancillary(const unsigned char* chunk)
{
    return ((chunk[4] & 32) != 0);
}


static const unsigned char* lodepng_chunk_data_const(const unsigned char* chunk)
{
    return &chunk[8];
}

#if 0 //thorvg don't use crc
/* returns 0 if the crc is correct, 1 if it's incorrect (0 for OK as usual!) */
static unsigned lodepng_chunk_check_crc(const unsigned char* chunk)
{
    unsigned length = lodepng_chunk_length(chunk);
    unsigned CRC = lodepng_read32bitInt(&chunk[length + 8]);
    /*the CRC is taken of the data and the 4 chunk type letters, not the length*/
    unsigned checksum = lodepng_crc32(&chunk[4], length + 4);
    if (CRC != checksum) return 1;
    else return 0;
}
#endif

static const unsigned char* lodepng_chunk_next_const(const unsigned char* chunk, const unsigned char* end)
{
    if (chunk >= end || end - chunk < 12) return end; /*too small to contain a chunk*/
    if (chunk[0] == 0x89 && chunk[1] == 0x50 && chunk[2] == 0x4e && chunk[3] == 0x47
        && chunk[4] == 0x0d && chunk[5] == 0x0a && chunk[6] == 0x1a && chunk[7] == 0x0a) {
        /* Is PNG magic header at start of PNG file. Jump to first actual chunk. */
        return chunk + 8;
    } else {
        size_t total_chunk_length;
        const unsigned char* result;
        if (lodepng_addofl(lodepng_chunk_length(chunk), 12, &total_chunk_length)) return end;
        result = chunk + total_chunk_length;
        if (result < chunk) return end; /*pointer overflow*/
        return result;
    }
}


/* ////////////////////////////////////////////////////////////////////////// */
/* / Color types, channels, bits                                            / */
/* ////////////////////////////////////////////////////////////////////////// */

/*checks if the colortype is valid and the bitdepth bd is allowed for this colortype.
Return value is a LodePNG error code.*/
static unsigned checkColorValidity(LodePNGColorType colortype, unsigned bd)
{
    switch(colortype) {
        case LCT_GREY:       if(!(bd == 1 || bd == 2 || bd == 4 || bd == 8 || bd == 16)) return 37; break;
        case LCT_RGB:        if(!(                                 bd == 8 || bd == 16)) return 37; break;
        case LCT_PALETTE:    if(!(bd == 1 || bd == 2 || bd == 4 || bd == 8            )) return 37; break;
        case LCT_GREY_ALPHA: if(!(                                 bd == 8 || bd == 16)) return 37; break;
        case LCT_RGBA:       if(!(                                 bd == 8 || bd == 16)) return 37; break;
        case LCT_MAX_OCTET_VALUE: return 31; /* invalid color type */
        default: return 31; /* invalid color type */
    }
    return 0; /*allowed color type / bits combination*/
}


static unsigned getNumColorChannels(LodePNGColorType colortype)
{
    switch(colortype) {
        case LCT_GREY: return 1;
        case LCT_RGB: return 3;
        case LCT_PALETTE: return 1;
        case LCT_GREY_ALPHA: return 2;
        case LCT_RGBA: return 4;
        case LCT_MAX_OCTET_VALUE: return 0; /* invalid color type */
        default: return 0; /*invalid color type*/
    }
}


static unsigned lodepng_get_bpp_lct(LodePNGColorType colortype, unsigned bitdepth)
{
    /*bits per pixel is amount of channels * bits per channel*/
    return getNumColorChannels(colortype) * bitdepth;
}


static void lodepng_color_mode_init(LodePNGColorMode* info)
{
    info->key_defined = 0;
    info->key_r = info->key_g = info->key_b = 0;
    info->colortype = LCT_RGBA;
    info->bitdepth = 8;
    info->palette = 0;
    info->palettesize = 0;
}


/*allocates palette memory if needed, and initializes all colors to black*/
static void lodepng_color_mode_alloc_palette(LodePNGColorMode* info)
{
    size_t i;
    /*if the palette is already allocated, it will have size 1024 so no reallocation needed in that case*/
    /*the palette must have room for up to 256 colors with 4 bytes each.*/
    if (!info->palette) info->palette = (unsigned char*)malloc(1024);
    if (!info->palette) return; /*alloc fail*/
    for (i = 0; i != 256; ++i) {
        /*Initialize all unused colors with black, the value used for invalid palette indices.
        This is an error according to the PNG spec, but common PNG decoders make it black instead.
        That makes color conversion slightly faster due to no error handling needed.*/
        info->palette[i * 4 + 0] = 0;
        info->palette[i * 4 + 1] = 0;
        info->palette[i * 4 + 2] = 0;
        info->palette[i * 4 + 3] = 255;
    }
}

static void lodepng_palette_clear(LodePNGColorMode* info)
{
    if (info->palette) free(info->palette);
    info->palette = 0;
    info->palettesize = 0;
}


static void lodepng_color_mode_cleanup(LodePNGColorMode* info)
{
    lodepng_palette_clear(info);
}


/*return value is error code (0 means no error)*/
static unsigned lodepng_color_mode_copy(LodePNGColorMode* dest, const LodePNGColorMode* source)
{
    lodepng_color_mode_cleanup(dest);
    lodepng_memcpy(dest, source, sizeof(LodePNGColorMode));
    if (source->palette) {
        dest->palette = (unsigned char*)malloc(1024);
        if (!dest->palette && source->palettesize) return 83; /*alloc fail*/
        lodepng_memcpy(dest->palette, source->palette, source->palettesize * 4);
    }
    return 0;
}


static int lodepng_color_mode_equal(const LodePNGColorMode* a, const LodePNGColorMode* b)
{
    size_t i;
    if (a->colortype != b->colortype) return 0;
    if (a->bitdepth != b->bitdepth) return 0;
    if (a->key_defined != b->key_defined) return 0;
    if (a->key_defined) {
        if(a->key_r != b->key_r) return 0;
        if(a->key_g != b->key_g) return 0;
        if(a->key_b != b->key_b) return 0;
    }
    if (a->palettesize != b->palettesize) return 0;
    for (i = 0; i != a->palettesize * 4; ++i) {
        if (a->palette[i] != b->palette[i]) return 0;
    }
    return 1;
}


static size_t lodepng_get_raw_size_lct(unsigned w, unsigned h, LodePNGColorType colortype, unsigned bitdepth)
{
    size_t bpp = lodepng_get_bpp_lct(colortype, bitdepth);
    size_t n = (size_t)w * (size_t)h;
    return ((n / 8u) * bpp) + ((n & 7u) * bpp + 7u) / 8u;
}


/* Returns the byte size of a raw image buffer with given width, height and color mode */
static size_t lodepng_get_raw_size(unsigned w, unsigned h, const LodePNGColorMode* color)
{
    return lodepng_get_raw_size_lct(w, h, color->colortype, color->bitdepth);
}


/*in an idat chunk, each scanline is a multiple of 8 bits, unlike the lodepng output buffer,
and in addition has one extra byte per line: the filter byte. So this gives a larger
result than lodepng_get_raw_size. Set h to 1 to get the size of 1 row including filter byte. */
static size_t lodepng_get_raw_size_idat(unsigned w, unsigned h, unsigned bpp)
{
    /* + 1 for the filter byte, and possibly plus padding bits per line. */
    /* Ignoring casts, the expression is equal to (w * bpp + 7) / 8 + 1, but avoids overflow of w * bpp */
    size_t line = ((size_t)(w / 8u) * bpp) + 1u + ((w & 7u) * bpp + 7u) / 8u;
    return (size_t)h * line;
}


/* Safely checks whether size_t overflow can be caused due to amount of pixels.
   This check is overcautious rather than precise. If this check indicates no overflow,
   you can safely compute in a size_t (but not an unsigned):
   -(size_t)w * (size_t)h * 8
   -amount of bytes in IDAT (including filter, padding and Adam7 bytes)
   -amount of bytes in raw color model
   Returns 1 if overflow possible, 0 if not. */
static int lodepng_pixel_overflow(unsigned w, unsigned h, const LodePNGColorMode* pngcolor, const LodePNGColorMode* rawcolor)
{
    size_t bpp = LODEPNG_MAX(lodepng_get_bpp_lct(pngcolor->colortype, pngcolor->bitdepth), lodepng_get_bpp_lct(rawcolor->colortype, rawcolor->bitdepth));
    size_t numpixels, total;
    size_t line; /* bytes per line in worst case */

    if (lodepng_mulofl((size_t)w, (size_t)h, &numpixels)) return 1;
    if (lodepng_mulofl(numpixels, 8, &total)) return 1; /* bit pointer with 8-bit color, or 8 bytes per channel color */

    /* Bytes per scanline with the expression "(w / 8u) * bpp) + ((w & 7u) * bpp + 7u) / 8u" */
    if (lodepng_mulofl((size_t)(w / 8u), bpp, &line)) return 1;
    if (lodepng_addofl(line, ((w & 7u) * bpp + 7u) / 8u, &line)) return 1;

    if (lodepng_addofl(line, 5, &line)) return 1; /* 5 bytes overhead per line: 1 filterbyte, 4 for Adam7 worst case */
    if (lodepng_mulofl(line, h, &total)) return 1; /* Total bytes in worst case */

    return 0; /* no overflow */
}


static void lodepng_info_init(LodePNGInfo* info)
{
    lodepng_color_mode_init(&info->color);
    info->interlace_method = 0;
    info->compression_method = 0;
    info->filter_method = 0;
}


static void lodepng_info_cleanup(LodePNGInfo* info)
{
    lodepng_color_mode_cleanup(&info->color);
}


/* index: bitgroup index, bits: bitgroup size(1, 2 or 4), in: bitgroup value, out: octet array to add bits to */
static void addColorBits(unsigned char* out, size_t index, unsigned bits, unsigned in)
{
    unsigned m = bits == 1 ? 7 : bits == 2 ? 3 : 1; /*8 / bits - 1*/
    /*p = the partial index in the byte, e.g. with 4 palettebits it is 0 for first half or 1 for second half*/
    unsigned p = index & m;
    in &= (1u << bits) - 1u; /*filter out any other bits of the input value*/
    in = in << (bits * (m - p));
    if(p == 0) out[index * bits / 8u] = in;
    else out[index * bits / 8u] |= in;
}

/*
    One node of a color tree
    This is the data structure used to count the number of unique colors and to get a palette
    index for a color. It's like an octree, but because the alpha channel is used too, each
    node has 16 instead of 8 children.
*/
struct ColorTree
{
    ColorTree* children[16]; /* up to 16 pointers to ColorTree of next level */
    int index; /* the payload. Only has a meaningful value if this is in the last level */
};

static void color_tree_init(ColorTree* tree)
{
    lodepng_memset(tree->children, 0, 16 * sizeof(*tree->children));
    tree->index = -1;
}

static void color_tree_cleanup(ColorTree* tree)
{
    int i;
    for (i = 0; i != 16; ++i) {
        if(tree->children[i]) {
            color_tree_cleanup(tree->children[i]);
            free(tree->children[i]);
        }
    }
}


/* returns -1 if color not present, its index otherwise */
static int color_tree_get(ColorTree* tree, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
    int bit = 0;
    for (bit = 0; bit < 8; ++bit) {
        int i = 8 * ((r >> bit) & 1) + 4 * ((g >> bit) & 1) + 2 * ((b >> bit) & 1) + 1 * ((a >> bit) & 1);
        if (!tree->children[i]) return -1;
        else tree = tree->children[i];
    }
    return tree ? tree->index : -1;
}


/* color is not allowed to already exist.
   Index should be >= 0 (it's signed to be compatible with using -1 for "doesn't exist")
   Returns error code, or 0 if ok */
static unsigned color_tree_add(ColorTree* tree, unsigned char r, unsigned char g, unsigned char b, unsigned char a, unsigned index)
{
    int bit;
    for (bit = 0; bit < 8; ++bit) {
        int i = 8 * ((r >> bit) & 1) + 4 * ((g >> bit) & 1) + 2 * ((b >> bit) & 1) + 1 * ((a >> bit) & 1);
        if (!tree->children[i]) {
            tree->children[i] = (ColorTree*)malloc(sizeof(ColorTree));
            if (!tree->children[i]) return 83; /*alloc fail*/
            color_tree_init(tree->children[i]);
        }
        tree = tree->children[i];
    }
    tree->index = (int)index;
    return 0;
}

/* put a pixel, given its RGBA color, into image of any color type */
static unsigned rgba8ToPixel(unsigned char* out, size_t i, const LodePNGColorMode* mode, ColorTree* tree /*for palette*/, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
    if (mode->colortype == LCT_GREY) {
        unsigned char gray = r; /*((unsigned short)r + g + b) / 3u;*/
        if (mode->bitdepth == 8) out[i] = gray;
        else if (mode->bitdepth == 16) out[i * 2 + 0] = out[i * 2 + 1] = gray;
        else {
            /*take the most significant bits of gray*/
            gray = ((unsigned)gray >> (8u - mode->bitdepth)) & ((1u << mode->bitdepth) - 1u);
            addColorBits(out, i, mode->bitdepth, gray);
        }
    } else if (mode->colortype == LCT_RGB) {
        if (mode->bitdepth == 8) {
            out[i * 3 + 0] = r;
            out[i * 3 + 1] = g;
            out[i * 3 + 2] = b;
        } else {
            out[i * 6 + 0] = out[i * 6 + 1] = r;
            out[i * 6 + 2] = out[i * 6 + 3] = g;
            out[i * 6 + 4] = out[i * 6 + 5] = b;
        }
    } else if(mode->colortype == LCT_PALETTE) {
        int index = color_tree_get(tree, r, g, b, a);
        if (index < 0) return 82; /*color not in palette*/
        if (mode->bitdepth == 8) out[i] = index;
        else addColorBits(out, i, mode->bitdepth, (unsigned)index);
    } else if (mode->colortype == LCT_GREY_ALPHA) {
        unsigned char gray = r; /*((unsigned short)r + g + b) / 3u;*/
        if (mode->bitdepth == 8) {
            out[i * 2 + 0] = gray;
            out[i * 2 + 1] = a;
        } else if (mode->bitdepth == 16) {
            out[i * 4 + 0] = out[i * 4 + 1] = gray;
            out[i * 4 + 2] = out[i * 4 + 3] = a;
        }
    } else if (mode->colortype == LCT_RGBA) {
        if (mode->bitdepth == 8) {
            out[i * 4 + 0] = r;
            out[i * 4 + 1] = g;
            out[i * 4 + 2] = b;
            out[i * 4 + 3] = a;
        } else {
            out[i * 8 + 0] = out[i * 8 + 1] = r;
            out[i * 8 + 2] = out[i * 8 + 3] = g;
            out[i * 8 + 4] = out[i * 8 + 5] = b;
            out[i * 8 + 6] = out[i * 8 + 7] = a;
        }
    }
    return 0; /*no error*/
}


/* put a pixel, given its RGBA16 color, into image of any color 16-bitdepth type */
static void rgba16ToPixel(unsigned char* out, size_t i, const LodePNGColorMode* mode, unsigned short r, unsigned short g, unsigned short b, unsigned short a)
{
    if (mode->colortype == LCT_GREY) {
        unsigned short gray = r; /*((unsigned)r + g + b) / 3u;*/
        out[i * 2 + 0] = (gray >> 8) & 255;
        out[i * 2 + 1] = gray & 255;
    } else if (mode->colortype == LCT_RGB) {
        out[i * 6 + 0] = (r >> 8) & 255;
        out[i * 6 + 1] = r & 255;
        out[i * 6 + 2] = (g >> 8) & 255;
        out[i * 6 + 3] = g & 255;
        out[i * 6 + 4] = (b >> 8) & 255;
        out[i * 6 + 5] = b & 255;
    } else if (mode->colortype == LCT_GREY_ALPHA) {
        unsigned short gray = r; /*((unsigned)r + g + b) / 3u;*/
        out[i * 4 + 0] = (gray >> 8) & 255;
        out[i * 4 + 1] = gray & 255;
        out[i * 4 + 2] = (a >> 8) & 255;
        out[i * 4 + 3] = a & 255;
    } else if (mode->colortype == LCT_RGBA) {
        out[i * 8 + 0] = (r >> 8) & 255;
        out[i * 8 + 1] = r & 255;
        out[i * 8 + 2] = (g >> 8) & 255;
        out[i * 8 + 3] = g & 255;
        out[i * 8 + 4] = (b >> 8) & 255;
        out[i * 8 + 5] = b & 255;
        out[i * 8 + 6] = (a >> 8) & 255;
        out[i * 8 + 7] = a & 255;
    }
}


/* Get RGBA8 color of pixel with index i (y * width + x) from the raw image with given color type. */
static void getPixelColorRGBA8(unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* a, const unsigned char* in, size_t i, const LodePNGColorMode* mode)
{
    if (mode->colortype == LCT_GREY) {
        if (mode->bitdepth == 8) {
            *r = *g = *b = in[i];
            if (mode->key_defined && *r == mode->key_r) *a = 0;
            else *a = 255;
        } else if (mode->bitdepth == 16) {
            *r = *g = *b = in[i * 2 + 0];
            if (mode->key_defined && 256U * in[i * 2 + 0] + in[i * 2 + 1] == mode->key_r) *a = 0;
            else *a = 255;
        } else {
            unsigned highest = ((1U << mode->bitdepth) - 1U); /* highest possible value for this bit depth */
            size_t j = i * mode->bitdepth;
            unsigned value = readBitsFromReversedStream(&j, in, mode->bitdepth);
            *r = *g = *b = (value * 255) / highest;
            if (mode->key_defined && value == mode->key_r) *a = 0;
            else *a = 255;
        }
    } else if (mode->colortype == LCT_RGB) {
        if (mode->bitdepth == 8) {
            *r = in[i * 3 + 0]; *g = in[i * 3 + 1]; *b = in[i * 3 + 2];
            if (mode->key_defined && *r == mode->key_r && *g == mode->key_g && *b == mode->key_b) *a = 0;
            else *a = 255;
        } else {
            *r = in[i * 6 + 0];
            *g = in[i * 6 + 2];
            *b = in[i * 6 + 4];
            if (mode->key_defined && 256U * in[i * 6 + 0] + in[i * 6 + 1] == mode->key_r
              && 256U * in[i * 6 + 2] + in[i * 6 + 3] == mode->key_g
              && 256U * in[i * 6 + 4] + in[i * 6 + 5] == mode->key_b) *a = 0;
            else *a = 255;
        }
    } else if (mode->colortype == LCT_PALETTE) {
        unsigned index;
        if (mode->bitdepth == 8) index = in[i];
        else {
            size_t j = i * mode->bitdepth;
            index = readBitsFromReversedStream(&j, in, mode->bitdepth);
        }
        /* out of bounds of palette not checked: see lodepng_color_mode_alloc_palette. */
        *r = mode->palette[index * 4 + 0];
        *g = mode->palette[index * 4 + 1];
        *b = mode->palette[index * 4 + 2];
        *a = mode->palette[index * 4 + 3];
    } else if (mode->colortype == LCT_GREY_ALPHA) {
        if (mode->bitdepth == 8) {
            *r = *g = *b = in[i * 2 + 0];
            *a = in[i * 2 + 1];
        } else {
            *r = *g = *b = in[i * 4 + 0];
            *a = in[i * 4 + 2];
        }
    } else if (mode->colortype == LCT_RGBA) {
        if (mode->bitdepth == 8) {
            *r = in[i * 4 + 0];
            *g = in[i * 4 + 1];
            *b = in[i * 4 + 2];
            *a = in[i * 4 + 3];
        } else {
            *r = in[i * 8 + 0];
            *g = in[i * 8 + 2];
            *b = in[i * 8 + 4];
            *a = in[i * 8 + 6];
        }
    }
}


/* Similar to getPixelColorRGBA8, but with all the for loops inside of the color
   mode test cases, optimized to convert the colors much faster, when converting
   to the common case of RGBA with 8 bit per channel. buffer must be RGBA with
   enough memory.*/
static void getPixelColorsRGBA8(unsigned char* LODEPNG_RESTRICT buffer, size_t numpixels, const unsigned char* LODEPNG_RESTRICT in, const LodePNGColorMode* mode)
{
    unsigned num_channels = 4;
    size_t i;
    if (mode->colortype == LCT_GREY) {
        if (mode->bitdepth == 8) {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                buffer[0] = buffer[1] = buffer[2] = in[i];
                buffer[3] = 255;
            }
            if (mode->key_defined) {
                buffer -= numpixels * num_channels;
                for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                    if(buffer[0] == mode->key_r) buffer[3] = 0;
                }
            }
        } else if (mode->bitdepth == 16) {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                buffer[0] = buffer[1] = buffer[2] = in[i * 2];
                buffer[3] = mode->key_defined && 256U * in[i * 2 + 0] + in[i * 2 + 1] == mode->key_r ? 0 : 255;
            }
        } else {
            unsigned highest = ((1U << mode->bitdepth) - 1U); /* highest possible value for this bit depth */
            size_t j = 0;
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                unsigned value = readBitsFromReversedStream(&j, in, mode->bitdepth);
                buffer[0] = buffer[1] = buffer[2] = (value * 255) / highest;
                buffer[3] = mode->key_defined && value == mode->key_r ? 0 : 255;
            }
        }
    } else if (mode->colortype == LCT_RGB) {
        if (mode->bitdepth == 8) {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                lodepng_memcpy(buffer, &in[i * 3], 3);
                buffer[3] = 255;
            }
            if (mode->key_defined) {
                buffer -= numpixels * num_channels;
                for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                    if (buffer[0] == mode->key_r && buffer[1]== mode->key_g && buffer[2] == mode->key_b) buffer[3] = 0;
                }
            }
        } else {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                buffer[0] = in[i * 6 + 0];
                buffer[1] = in[i * 6 + 2];
                buffer[2] = in[i * 6 + 4];
                buffer[3] = mode->key_defined
                  && 256U * in[i * 6 + 0] + in[i * 6 + 1] == mode->key_r
                  && 256U * in[i * 6 + 2] + in[i * 6 + 3] == mode->key_g
                  && 256U * in[i * 6 + 4] + in[i * 6 + 5] == mode->key_b ? 0 : 255;
            }
        }
    } else if (mode->colortype == LCT_PALETTE) {
        if (mode->bitdepth == 8) {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
              unsigned index = in[i];
              /* out of bounds of palette not checked: see lodepng_color_mode_alloc_palette. */
              lodepng_memcpy(buffer, &mode->palette[index * 4], 4);
            }
        } else {
            size_t j = 0;
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
              unsigned index = readBitsFromReversedStream(&j, in, mode->bitdepth);
              /* out of bounds of palette not checked: see lodepng_color_mode_alloc_palette. */
              lodepng_memcpy(buffer, &mode->palette[index * 4], 4);
            }
        }
    } else if (mode->colortype == LCT_GREY_ALPHA) {
        if (mode->bitdepth == 8) {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
              buffer[0] = buffer[1] = buffer[2] = in[i * 2 + 0];
              buffer[3] = in[i * 2 + 1];
            }
        } else {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
              buffer[0] = buffer[1] = buffer[2] = in[i * 4 + 0];
              buffer[3] = in[i * 4 + 2];
            }
        }
    } else if (mode->colortype == LCT_RGBA) {
        if (mode->bitdepth == 8) {
            lodepng_memcpy(buffer, in, numpixels * 4);
        } else {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                buffer[0] = in[i * 8 + 0];
                buffer[1] = in[i * 8 + 2];
                buffer[2] = in[i * 8 + 4];
                buffer[3] = in[i * 8 + 6];
            }
        }
    }
}


/* Similar to getPixelColorsRGBA8, but with 3-channel RGB output. */
static void getPixelColorsRGB8(unsigned char* LODEPNG_RESTRICT buffer, size_t numpixels, const unsigned char* LODEPNG_RESTRICT in, const LodePNGColorMode* mode)
{
    const unsigned num_channels = 3;
    size_t i;
    if (mode->colortype == LCT_GREY) {
        if (mode->bitdepth == 8) {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
               buffer[0] = buffer[1] = buffer[2] = in[i];
            }
        } else if (mode->bitdepth == 16) {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                buffer[0] = buffer[1] = buffer[2] = in[i * 2];
            }
        } else {
            unsigned highest = ((1U << mode->bitdepth) - 1U); /* highest possible value for this bit depth */
            size_t j = 0;
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                unsigned value = readBitsFromReversedStream(&j, in, mode->bitdepth);
                buffer[0] = buffer[1] = buffer[2] = (value * 255) / highest;
            }
        }
    } else if (mode->colortype == LCT_RGB) {
        if (mode->bitdepth == 8) {
           lodepng_memcpy(buffer, in, numpixels * 3);
        } else {
            for(i = 0; i != numpixels; ++i, buffer += num_channels) {
                buffer[0] = in[i * 6 + 0];
                buffer[1] = in[i * 6 + 2];
                buffer[2] = in[i * 6 + 4];
            }
        }
    } else if (mode->colortype == LCT_PALETTE) {
        if (mode->bitdepth == 8) {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                unsigned index = in[i];
                /* out of bounds of palette not checked: see lodepng_color_mode_alloc_palette. */
                lodepng_memcpy(buffer, &mode->palette[index * 4], 3);
            }
        } else {
            size_t j = 0;
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                unsigned index = readBitsFromReversedStream(&j, in, mode->bitdepth);
                /* out of bounds of palette not checked: see lodepng_color_mode_alloc_palette. */
                lodepng_memcpy(buffer, &mode->palette[index * 4], 3);
            }
        }
    } else if (mode->colortype == LCT_GREY_ALPHA) {
        if (mode->bitdepth == 8) {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                buffer[0] = buffer[1] = buffer[2] = in[i * 2 + 0];
            }
        } else {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                buffer[0] = buffer[1] = buffer[2] = in[i * 4 + 0];
            }
        }
    } else if (mode->colortype == LCT_RGBA) {
        if (mode->bitdepth == 8) {
            for(i = 0; i != numpixels; ++i, buffer += num_channels) {
                lodepng_memcpy(buffer, &in[i * 4], 3);
            }
        } else {
            for (i = 0; i != numpixels; ++i, buffer += num_channels) {
                buffer[0] = in[i * 8 + 0];
                buffer[1] = in[i * 8 + 2];
                buffer[2] = in[i * 8 + 4];
            }
        }
    }
}


/* Get RGBA16 color of pixel with index i (y * width + x) from the raw image with
   given color type, but the given color type must be 16-bit itself. */
static void getPixelColorRGBA16(unsigned short* r, unsigned short* g, unsigned short* b, unsigned short* a, const unsigned char* in, size_t i, const LodePNGColorMode* mode)
{
    if (mode->colortype == LCT_GREY) {
        *r = *g = *b = 256 * in[i * 2 + 0] + in[i * 2 + 1];
        if (mode->key_defined && 256U * in[i * 2 + 0] + in[i * 2 + 1] == mode->key_r) *a = 0;
        else *a = 65535;
    } else if (mode->colortype == LCT_RGB) {
        *r = 256u * in[i * 6 + 0] + in[i * 6 + 1];
        *g = 256u * in[i * 6 + 2] + in[i * 6 + 3];
        *b = 256u * in[i * 6 + 4] + in[i * 6 + 5];
        if (mode->key_defined
          && 256u * in[i * 6 + 0] + in[i * 6 + 1] == mode->key_r
          && 256u * in[i * 6 + 2] + in[i * 6 + 3] == mode->key_g
          && 256u * in[i * 6 + 4] + in[i * 6 + 5] == mode->key_b) *a = 0;
        else *a = 65535;
    } else if (mode->colortype == LCT_GREY_ALPHA) {
        *r = *g = *b = 256u * in[i * 4 + 0] + in[i * 4 + 1];
        *a = 256u * in[i * 4 + 2] + in[i * 4 + 3];
    } else if (mode->colortype == LCT_RGBA) {
        *r = 256u * in[i * 8 + 0] + in[i * 8 + 1];
        *g = 256u * in[i * 8 + 2] + in[i * 8 + 3];
        *b = 256u * in[i * 8 + 4] + in[i * 8 + 5];
        *a = 256u * in[i * 8 + 6] + in[i * 8 + 7];
    }
}

/*
  Converts raw buffer from one color type to another color type, based on
  LodePNGColorMode structs to describe the input and output color type.
  See the reference manual at the end of this header file to see which color conversions are supported.
  return value = LodePNG error code (0 if all went ok, an error if the conversion isn't supported)
  The out buffer must have size (w * h * bpp + 7) / 8, where bpp is the bits per pixel
  of the output color type (lodepng_get_bpp).
  For < 8 bpp images, there should not be padding bits at the end of scanlines.
  For 16-bit per channel colors, uses big endian format like PNG does.
  Return value is LodePNG error code
*/
static unsigned lodepng_convert(unsigned char* out, const unsigned char* in, const LodePNGColorMode* mode_out, const LodePNGColorMode* mode_in, unsigned w, unsigned h)
{
    size_t i;
    ColorTree tree;
    size_t numpixels = (size_t)w * (size_t)h;
    unsigned error = 0;

    if (mode_in->colortype == LCT_PALETTE && !mode_in->palette) {
        return 107; /* error: must provide palette if input mode is palette */
    }

    if (lodepng_color_mode_equal(mode_out, mode_in)) {
        size_t numbytes = lodepng_get_raw_size(w, h, mode_in);
        lodepng_memcpy(out, in, numbytes);
        return 0;
    }

    if (mode_out->colortype == LCT_PALETTE) {
        size_t palettesize = mode_out->palettesize;
        const unsigned char* palette = mode_out->palette;
        size_t palsize = (size_t)1u << mode_out->bitdepth;
        /* if the user specified output palette but did not give the values, assume
           they want the values of the input color type (assuming that one is palette).
           Note that we never create a new palette ourselves.*/
        if (palettesize == 0) {
            palettesize = mode_in->palettesize;
            palette = mode_in->palette;
            /* if the input was also palette with same bitdepth, then the color types are also
               equal, so copy literally. This to preserve the exact indices that were in the PNG
               even in case there are duplicate colors in the palette.*/
            if (mode_in->colortype == LCT_PALETTE && mode_in->bitdepth == mode_out->bitdepth) {
                size_t numbytes = lodepng_get_raw_size(w, h, mode_in);
                lodepng_memcpy(out, in, numbytes);
                return 0;
            }
        }
        if (palettesize < palsize) palsize = palettesize;
        color_tree_init(&tree);
        for (i = 0; i != palsize; ++i) {
            const unsigned char* p = &palette[i * 4];
            error = color_tree_add(&tree, p[0], p[1], p[2], p[3], (unsigned)i);
            if (error) break;
        }
    }

    if (!error) {
        if (mode_in->bitdepth == 16 && mode_out->bitdepth == 16) {
            for (i = 0; i != numpixels; ++i) {
                unsigned short r = 0, g = 0, b = 0, a = 0;
                getPixelColorRGBA16(&r, &g, &b, &a, in, i, mode_in);
                rgba16ToPixel(out, i, mode_out, r, g, b, a);
            }
        } else if (mode_out->bitdepth == 8 && mode_out->colortype == LCT_RGBA) {
            getPixelColorsRGBA8(out, numpixels, in, mode_in);
        } else if(mode_out->bitdepth == 8 && mode_out->colortype == LCT_RGB) {
            getPixelColorsRGB8(out, numpixels, in, mode_in);
        } else {
            unsigned char r = 0, g = 0, b = 0, a = 0;
            for (i = 0; i != numpixels; ++i) {
                getPixelColorRGBA8(&r, &g, &b, &a, in, i, mode_in);
                error = rgba8ToPixel(out, i, mode_out, &tree, r, g, b, a);
                if (error) break;
            }
        }
    }

    if (mode_out->colortype == LCT_PALETTE) {
        color_tree_cleanup(&tree);
    }

    return error;
}


/* Paeth predictor, used by PNG filter type 4
   The parameters are of type short, but should come from unsigned chars, the shorts
   are only needed to make the paeth calculation correct.
*/
static unsigned char paethPredictor(short a, short b, short c)
{
    short pa = LODEPNG_ABS(b - c);
    short pb = LODEPNG_ABS(a - c);
    short pc = LODEPNG_ABS(a + b - c - c);
    /* return input value associated with smallest of pa, pb, pc (with certain priority if equal) */
    if (pb < pa) { a = b; pa = pb; }
    return (pc < pa) ? c : a;
}


/*shared values used by multiple Adam7 related functions*/
static const unsigned ADAM7_IX[7] = { 0, 4, 0, 2, 0, 1, 0 }; /*x start values*/
static const unsigned ADAM7_IY[7] = { 0, 0, 4, 0, 2, 0, 1 }; /*y start values*/
static const unsigned ADAM7_DX[7] = { 8, 8, 4, 4, 2, 2, 1 }; /*x delta values*/
static const unsigned ADAM7_DY[7] = { 8, 8, 8, 4, 4, 2, 2 }; /*y delta values*/

/* Outputs various dimensions and positions in the image related to the Adam7 reduced images.
   passw: output containing the width of the 7 passes
   passh: output containing the height of the 7 passes
   filter_passstart: output containing the index of the start and end of each
   reduced image with filter bytes
   padded_passstart output containing the index of the start and end of each
   reduced image when without filter bytes but with padded scanlines
   passstart: output containing the index of the start and end of each reduced
   image without padding between scanlines, but still padding between the images
   w, h: width and height of non-interlaced image
   bpp: bits per pixel
   "padded" is only relevant if bpp is less than 8 and a scanline or image does not
   end at a full byte */
static void Adam7_getpassvalues(unsigned passw[7], unsigned passh[7], size_t filter_passstart[8], size_t padded_passstart[8], size_t passstart[8], unsigned w, unsigned h, unsigned bpp)
{
    /* the passstart values have 8 values: the 8th one indicates the byte after the end of the 7th (= last) pass */
    unsigned i;

    /* calculate width and height in pixels of each pass */
    for (i = 0; i != 7; ++i) {
        passw[i] = (w + ADAM7_DX[i] - ADAM7_IX[i] - 1) / ADAM7_DX[i];
        passh[i] = (h + ADAM7_DY[i] - ADAM7_IY[i] - 1) / ADAM7_DY[i];
        if(passw[i] == 0) passh[i] = 0;
        if(passh[i] == 0) passw[i] = 0;
    }

    filter_passstart[0] = padded_passstart[0] = passstart[0] = 0;
    for (i = 0; i != 7; ++i) {
        /* if passw[i] is 0, it's 0 bytes, not 1 (no filtertype-byte) */
        filter_passstart[i + 1] = filter_passstart[i]
                                + ((passw[i] && passh[i]) ? passh[i] * (1u + (passw[i] * bpp + 7u) / 8u) : 0);
        /* bits padded if needed to fill full byte at end of each scanline */
        padded_passstart[i + 1] = padded_passstart[i] + passh[i] * ((passw[i] * bpp + 7u) / 8u);
        /* only padded at end of reduced image */
        passstart[i + 1] = passstart[i] + (passh[i] * passw[i] * bpp + 7u) / 8u;
    }
}


/* ////////////////////////////////////////////////////////////////////////// */
/* / PNG Decoder                                                            / */
/* ////////////////////////////////////////////////////////////////////////// */

static unsigned unfilterScanline(unsigned char* recon, const unsigned char* scanline, const unsigned char* precon, size_t bytewidth, unsigned char filterType, size_t length)
{
    /* For PNG filter method 0
       unfilter a PNG image scanline by scanline. when the pixels are smaller than 1 byte,
       the filter works byte per byte (bytewidth = 1)
       precon is the previous unfiltered scanline, recon the result, scanline the current one
       the incoming scanlines do NOT include the filtertype byte, that one is given in the parameter filterType instead
       recon and scanline MAY be the same memory address! precon must be disjoint. */

    size_t i;
    switch (filterType) {
        case 0:
            for (i = 0; i != length; ++i) recon[i] = scanline[i];
            break;
        case 1:
            for (i = 0; i != bytewidth; ++i) recon[i] = scanline[i];
            for (i = bytewidth; i < length; ++i) recon[i] = scanline[i] + recon[i - bytewidth];
            break;
        case 2:
            if (precon) {
                for(i = 0; i != length; ++i) recon[i] = scanline[i] + precon[i];
            } else {
                for(i = 0; i != length; ++i) recon[i] = scanline[i];
            }
            break;
        case 3:
          if (precon) {
              for (i = 0; i != bytewidth; ++i) recon[i] = scanline[i] + (precon[i] >> 1u);
              for (i = bytewidth; i < length; ++i) recon[i] = scanline[i] + ((recon[i - bytewidth] + precon[i]) >> 1u);
          } else {
              for (i = 0; i != bytewidth; ++i) recon[i] = scanline[i];
              for (i = bytewidth; i < length; ++i) recon[i] = scanline[i] + (recon[i - bytewidth] >> 1u);
          }
          break;
        case 4:
            if (precon) {
                for (i = 0; i != bytewidth; ++i) {
                    recon[i] = (scanline[i] + precon[i]); /*paethPredictor(0, precon[i], 0) is always precon[i]*/
                }

                /* Unroll independent paths of the paeth predictor. A 6x and 8x version would also be possible but that
                   adds too much code. Whether this actually speeds anything up at all depends on compiler and settings. */
                if (bytewidth >= 4) {
                    for (; i + 3 < length; i += 4) {
                        size_t j = i - bytewidth;
                        unsigned char s0 = scanline[i + 0], s1 = scanline[i + 1], s2 = scanline[i + 2], s3 = scanline[i + 3];
                        unsigned char r0 = recon[j + 0], r1 = recon[j + 1], r2 = recon[j + 2], r3 = recon[j + 3];
                        unsigned char p0 = precon[i + 0], p1 = precon[i + 1], p2 = precon[i + 2], p3 = precon[i + 3];
                        unsigned char q0 = precon[j + 0], q1 = precon[j + 1], q2 = precon[j + 2], q3 = precon[j + 3];
                        recon[i + 0] = s0 + paethPredictor(r0, p0, q0);
                        recon[i + 1] = s1 + paethPredictor(r1, p1, q1);
                        recon[i + 2] = s2 + paethPredictor(r2, p2, q2);
                        recon[i + 3] = s3 + paethPredictor(r3, p3, q3);
                    }
                } else if (bytewidth >= 3) {
                    for (; i + 2 < length; i += 3) {
                        size_t j = i - bytewidth;
                        unsigned char s0 = scanline[i + 0], s1 = scanline[i + 1], s2 = scanline[i + 2];
                        unsigned char r0 = recon[j + 0], r1 = recon[j + 1], r2 = recon[j + 2];
                        unsigned char p0 = precon[i + 0], p1 = precon[i + 1], p2 = precon[i + 2];
                        unsigned char q0 = precon[j + 0], q1 = precon[j + 1], q2 = precon[j + 2];
                        recon[i + 0] = s0 + paethPredictor(r0, p0, q0);
                        recon[i + 1] = s1 + paethPredictor(r1, p1, q1);
                        recon[i + 2] = s2 + paethPredictor(r2, p2, q2);
                    }
                } else if (bytewidth >= 2) {
                    for (; i + 1 < length; i += 2) {
                        size_t j = i - bytewidth;
                        unsigned char s0 = scanline[i + 0], s1 = scanline[i + 1];
                        unsigned char r0 = recon[j + 0], r1 = recon[j + 1];
                        unsigned char p0 = precon[i + 0], p1 = precon[i + 1];
                        unsigned char q0 = precon[j + 0], q1 = precon[j + 1];
                        recon[i + 0] = s0 + paethPredictor(r0, p0, q0);
                        recon[i + 1] = s1 + paethPredictor(r1, p1, q1);
                    }
                }

                for (; i != length; ++i) {
                    recon[i] = (scanline[i] + paethPredictor(recon[i - bytewidth], precon[i], precon[i - bytewidth]));
                }
            } else {
                for (i = 0; i != bytewidth; ++i) {
                    recon[i] = scanline[i];
                }
                for (i = bytewidth; i < length; ++i) {
                    /* paethPredictor(recon[i - bytewidth], 0, 0) is always recon[i - bytewidth] */
                    recon[i] = (scanline[i] + recon[i - bytewidth]);
                }
            }
            break;
        default: return 36; /* error: invalid filter type given */
    }
    return 0;
}


static unsigned unfilter(unsigned char* out, const unsigned char* in, unsigned w, unsigned h, unsigned bpp)
{
    /* For PNG filter method 0
       this function unfilters a single image (e.g. without interlacing this is called once, with Adam7 seven times)
       out must have enough bytes allocated already, in must have the scanlines + 1 filtertype byte per scanline
       w and h are image dimensions or dimensions of reduced image, bpp is bits per pixel
       in and out are allowed to be the same memory address (but aren't the same size since in has the extra filter bytes) */

    unsigned y;
    unsigned char* prevline = 0;

    /* bytewidth is used for filtering, is 1 when bpp < 8, number of bytes per pixel otherwise */
    size_t bytewidth = (bpp + 7u) / 8u;
    /* the width of a scanline in bytes, not including the filter type */
    size_t linebytes = lodepng_get_raw_size_idat(w, 1, bpp) - 1u;

    for (y = 0; y < h; ++y) {
        size_t outindex = linebytes * y;
        size_t inindex = (1 + linebytes) * y; /* the extra filterbyte added to each row */
        unsigned char filterType = in[inindex];
        CERROR_TRY_RETURN(unfilterScanline(&out[outindex], &in[inindex + 1], prevline, bytewidth, filterType, linebytes));
        prevline = &out[outindex];
    }

    return 0;
}

/* in: Adam7 interlaced image, with no padding bits between scanlines, but between
   reduced images so that each reduced image starts at a byte.
   out: the same pixels, but re-ordered so that they're now a non-interlaced image with size w*h
   bpp: bits per pixel
   out has the following size in bits: w * h * bpp.
   in is possibly bigger due to padding bits between reduced images.
   out must be big enough AND must be 0 everywhere if bpp < 8 in the current implementation
   (because that's likely a little bit faster)
   NOTE: comments about padding bits are only relevant if bpp < 8 */
static void Adam7_deinterlace(unsigned char* out, const unsigned char* in, unsigned w, unsigned h, unsigned bpp)
{
    unsigned passw[7], passh[7];
    size_t filter_passstart[8], padded_passstart[8], passstart[8];
    unsigned i;

    Adam7_getpassvalues(passw, passh, filter_passstart, padded_passstart, passstart, w, h, bpp);

    if (bpp >= 8) {
        for(i = 0; i != 7; ++i) {
            unsigned x, y, b;
            size_t bytewidth = bpp / 8u;
            for (y = 0; y < passh[i]; ++y)
            for (x = 0; x < passw[i]; ++x) {
                size_t pixelinstart = passstart[i] + (y * passw[i] + x) * bytewidth;
                size_t pixeloutstart = ((ADAM7_IY[i] + (size_t)y * ADAM7_DY[i]) * (size_t)w + ADAM7_IX[i] + (size_t)x * ADAM7_DX[i]) * bytewidth;
                for (b = 0; b < bytewidth; ++b) {
                    out[pixeloutstart + b] = in[pixelinstart + b];
                }
            }
        }
    } else /* bpp < 8: Adam7 with pixels < 8 bit is a bit trickier: with bit pointers */ {
        for (i = 0; i != 7; ++i) {
            unsigned x, y, b;
            unsigned ilinebits = bpp * passw[i];
            unsigned olinebits = bpp * w;
            size_t obp, ibp; /* bit pointers (for out and in buffer) */
            for (y = 0; y < passh[i]; ++y)
            for (x = 0; x < passw[i]; ++x) {
                ibp = (8 * passstart[i]) + (y * ilinebits + x * bpp);
                obp = (ADAM7_IY[i] + (size_t)y * ADAM7_DY[i]) * olinebits + (ADAM7_IX[i] + (size_t)x * ADAM7_DX[i]) * bpp;
                for (b = 0; b < bpp; ++b) {
                    unsigned char bit = readBitFromReversedStream(&ibp, in);
                    setBitOfReversedStream(&obp, out, bit);
                }
            }
        }
    }
}


static void removePaddingBits(unsigned char* out, const unsigned char* in, size_t olinebits, size_t ilinebits, unsigned h)
{
    /* After filtering there are still padding bits if scanlines have non multiple of 8 bit amounts. They need
       to be removed (except at last scanline of (Adam7-reduced) image) before working with pure image buffers
       for the Adam7 code, the color convert code and the output to the user.
       in and out are allowed to be the same buffer, in may also be higher but still overlapping; in must
       have >= ilinebits*h bits, out must have >= olinebits*h bits, olinebits must be <= ilinebits
       also used to move bits after earlier such operations happened, e.g. in a sequence of reduced images from Adam7
       only useful if (ilinebits - olinebits) is a value in the range 1..7 */
    unsigned y;
    size_t diff = ilinebits - olinebits;
    size_t ibp = 0, obp = 0; /*input and output bit pointers*/
    for (y = 0; y < h; ++y) {
        size_t x;
        for (x = 0; x < olinebits; ++x) {
            unsigned char bit = readBitFromReversedStream(&ibp, in);
            setBitOfReversedStream(&obp, out, bit);
        }
        ibp += diff;
    }
}


/* out must be buffer big enough to contain full image, and in must contain the full decompressed data from
   the IDAT chunks (with filter index bytes and possible padding bits)
   return value is error */
static unsigned postProcessScanlines(unsigned char* out, unsigned char* in, unsigned w, unsigned h, const LodePNGInfo* info_png)
{
    /* This function converts the filtered-padded-interlaced data into pure 2D image buffer with the PNG's colortype.
       Steps:
       *) if no Adam7: 1) unfilter 2) remove padding bits (= possible extra bits per scanline if bpp < 8)
       *) if adam7: 1) 7x unfilter 2) 7x remove padding bits 3) Adam7_deinterlace
       NOTE: the in buffer will be overwritten with intermediate data! */
    unsigned bpp = lodepng_get_bpp_lct(info_png->color.colortype, info_png->color.bitdepth);
    if (bpp == 0) return 31; /* error: invalid colortype */

    if (info_png->interlace_method == 0) {
        if (bpp < 8 && w * bpp != ((w * bpp + 7u) / 8u) * 8u) {
            CERROR_TRY_RETURN(unfilter(in, in, w, h, bpp));
            removePaddingBits(out, in, w * bpp, ((w * bpp + 7u) / 8u) * 8u, h);
        }
        /* we can immediately filter into the out buffer, no other steps needed */
        else CERROR_TRY_RETURN(unfilter(out, in, w, h, bpp));
    } else /* interlace_method is 1 (Adam7) */ {
        unsigned passw[7], passh[7]; size_t filter_passstart[8], padded_passstart[8], passstart[8];
        unsigned i;

        Adam7_getpassvalues(passw, passh, filter_passstart, padded_passstart, passstart, w, h, bpp);

        for (i = 0; i != 7; ++i) {
            CERROR_TRY_RETURN(unfilter(&in[padded_passstart[i]], &in[filter_passstart[i]], passw[i], passh[i], bpp));
            /* TODO: possible efficiency improvement: if in this reduced image the bits fit nicely in 1 scanline,
               move bytes instead of bits or move not at all */
            if (bpp < 8) {
              /* remove padding bits in scanlines; after this there still may be padding
                 bits between the different reduced images: each reduced image still starts nicely at a byte */
              removePaddingBits(&in[passstart[i]], &in[padded_passstart[i]], passw[i] * bpp, ((passw[i] * bpp + 7u) / 8u) * 8u, passh[i]);
            }
        }
        Adam7_deinterlace(out, in, w, h, bpp);
    }
    return 0;
}


static unsigned readChunk_PLTE(LodePNGColorMode* color, const unsigned char* data, size_t chunkLength)
{
    unsigned pos = 0, i;
    color->palettesize = chunkLength / 3u;
    if (color->palettesize == 0 || color->palettesize > 256) return 38; /* error: palette too small or big */
    lodepng_color_mode_alloc_palette(color);
    if (!color->palette && color->palettesize) {
        color->palettesize = 0;
        return 83; /* alloc fail */
    }

    for (i = 0; i != color->palettesize; ++i) {
        color->palette[4 * i + 0] = data[pos++]; /*R*/
        color->palette[4 * i + 1] = data[pos++]; /*G*/
        color->palette[4 * i + 2] = data[pos++]; /*B*/
        color->palette[4 * i + 3] = 255; /*alpha*/
    }

    return 0; /* OK */
}


static unsigned readChunk_tRNS(LodePNGColorMode* color, const unsigned char* data, size_t chunkLength)
{
    unsigned i;
    if (color->colortype == LCT_PALETTE) {
        /* error: more alpha values given than there are palette entries */
        if (chunkLength > color->palettesize) return 39;

        for (i = 0; i != chunkLength; ++i) color->palette[4 * i + 3] = data[i];
    } else if (color->colortype == LCT_GREY) {
        /* error: this chunk must be 2 bytes for grayscale image */
        if (chunkLength != 2) return 30;

        color->key_defined = 1;
        color->key_r = color->key_g = color->key_b = 256u * data[0] + data[1];
    } else if (color->colortype == LCT_RGB) {
        /* error: this chunk must be 6 bytes for RGB image */
        if (chunkLength != 6) return 41;

        color->key_defined = 1;
        color->key_r = 256u * data[0] + data[1];
        color->key_g = 256u * data[2] + data[3];
        color->key_b = 256u * data[4] + data[5];
    }
    else return 42; /* error: tRNS chunk not allowed for other color models */

    return 0; /* OK */
}


/* read a PNG, the result will be in the same color type as the PNG (hence "generic") */
static void decodeGeneric(unsigned char** out, unsigned* w, unsigned* h, LodePNGState* state, const unsigned char* in, size_t insize)
{
    unsigned char IEND = 0;
    const unsigned char* chunk;
    unsigned char* idat; /*the data from idat chunks, zlib compressed*/
    size_t idatsize = 0;
    unsigned char* scanlines = 0;
    size_t scanlines_size = 0, expected_size = 0;
    size_t outsize = 0;

    /* safe output values in case error happens */
    *out = 0;
    *w = *h = 0;

    state->error = lodepng_inspect(w, h, state, in, insize); /*reads header and resets other parameters in state->info_png*/
    if (state->error) return;

    if (lodepng_pixel_overflow(*w, *h, &state->info_png.color, &state->info_raw)) {
        CERROR_RETURN(state->error, 92); /*overflow possible due to amount of pixels*/
    }

    /*the input filesize is a safe upper bound for the sum of idat chunks size*/
    idat = (unsigned char*)malloc(insize);
    if (!idat) CERROR_RETURN(state->error, 83); /*alloc fail*/

    chunk = &in[33]; /*first byte of the first chunk after the header*/

    /*loop through the chunks, ignoring unknown chunks and stopping at IEND chunk.
    IDAT data is put at the start of the in buffer*/
    while (!IEND && !state->error) {
        unsigned chunkLength;
        const unsigned char* data; /*the data in the chunk*/

        /*error: size of the in buffer too small to contain next chunk*/
        if ((size_t)((chunk - in) + 12) > insize || chunk < in) {
            if (state->decoder.ignore_end) break; /*other errors may still happen though*/
            CERROR_BREAK(state->error, 30);
        }

        /*length of the data of the chunk, excluding the length bytes, chunk type and CRC bytes*/
        chunkLength = lodepng_chunk_length(chunk);
        /*error: chunk length larger than the max PNG chunk size*/
        if (chunkLength > 2147483647) {
            if (state->decoder.ignore_end) break; /*other errors may still happen though*/
            CERROR_BREAK(state->error, 63);
        }

        if ((size_t)((chunk - in) + chunkLength + 12) > insize || (chunk + chunkLength + 12) < in) {
            CERROR_BREAK(state->error, 64); /*error: size of the in buffer too small to contain next chunk*/
        }

        data = lodepng_chunk_data_const(chunk);

        /*for unknown chunk order*/
        //unsigned unknown = 0;

        /*IDAT chunk, containing compressed image data*/
        if (lodepng_chunk_type_equals(chunk, "IDAT")) {
            size_t newsize;
            if (lodepng_addofl(idatsize, chunkLength, &newsize)) CERROR_BREAK(state->error, 95);
            if (newsize > insize) CERROR_BREAK(state->error, 95);
            lodepng_memcpy(idat + idatsize, data, chunkLength);
            idatsize += chunkLength;
        } else if (lodepng_chunk_type_equals(chunk, "IEND")) {
            /*IEND chunk*/
            IEND = 1;
        } else if (lodepng_chunk_type_equals(chunk, "PLTE")) {
            /*palette chunk (PLTE)*/
            state->error = readChunk_PLTE(&state->info_png.color, data, chunkLength);
            if (state->error) break;
        } else if (lodepng_chunk_type_equals(chunk, "tRNS")) {
            /*palette transparency chunk (tRNS). Even though this one is an ancillary chunk , it is still compiled
            in without 'LODEPNG_COMPILE_ANCILLARY_CHUNKS' because it contains essential color information that
            affects the alpha channel of pixels. */
            state->error = readChunk_tRNS(&state->info_png.color, data, chunkLength);
            if (state->error) break;
        } else /*it's not an implemented chunk type, so ignore it: skip over the data*/ {
            /*error: unknown critical chunk (5th bit of first byte of chunk type is 0)*/
            if (!state->decoder.ignore_critical && !lodepng_chunk_ancillary(chunk)) {
                CERROR_BREAK(state->error, 69);
            }
            //unknown = 1;
        }

#if 0 //We don't use CRC
        if (!state->decoder.ignore_crc && !unknown) /*check CRC if wanted, only on known chunk types*/ {
            if (lodepng_chunk_check_crc(chunk)) CERROR_BREAK(state->error, 57); /*invalid CRC*/
        }
#endif
        if (!IEND) chunk = lodepng_chunk_next_const(chunk, in + insize);
    }

    if (state->info_png.color.colortype == LCT_PALETTE && !state->info_png.color.palette) {
        state->error = 106; /* error: PNG file must have PLTE chunk if color type is palette */
    }

    if (!state->error) {
        /*predict output size, to allocate exact size for output buffer to avoid more dynamic allocation.
        If the decompressed size does not match the prediction, the image must be corrupt.*/
        if (state->info_png.interlace_method == 0) {
            size_t bpp = lodepng_get_bpp_lct(state->info_png.color.colortype, state->info_png.color.bitdepth);
            expected_size = lodepng_get_raw_size_idat(*w, *h, bpp);
        } else {
            size_t bpp = lodepng_get_bpp_lct(state->info_png.color.colortype, state->info_png.color.bitdepth);
            /*Adam-7 interlaced: expected size is the sum of the 7 sub-images sizes*/
            expected_size = 0;
            expected_size += lodepng_get_raw_size_idat((*w + 7) >> 3, (*h + 7) >> 3, bpp);
            if (*w > 4) expected_size += lodepng_get_raw_size_idat((*w + 3) >> 3, (*h + 7) >> 3, bpp);
            expected_size += lodepng_get_raw_size_idat((*w + 3) >> 2, (*h + 3) >> 3, bpp);
            if (*w > 2) expected_size += lodepng_get_raw_size_idat((*w + 1) >> 2, (*h + 3) >> 2, bpp);
            expected_size += lodepng_get_raw_size_idat((*w + 1) >> 1, (*h + 1) >> 2, bpp);
            if (*w > 1) expected_size += lodepng_get_raw_size_idat((*w + 0) >> 1, (*h + 1) >> 1, bpp);
            expected_size += lodepng_get_raw_size_idat((*w + 0), (*h + 0) >> 1, bpp);
        }
        state->error = zlib_decompress(&scanlines, &scanlines_size, expected_size, idat, idatsize, &state->decoder.zlibsettings);
    }

    if (!state->error && scanlines_size != expected_size) state->error = 91; /*decompressed size doesn't match prediction*/
    free(idat);

    if (!state->error) {
        outsize = lodepng_get_raw_size(*w, *h, &state->info_png.color);
        *out = (unsigned char*)malloc(outsize);
        if (!*out) state->error = 83; /*alloc fail*/
    }
    if (!state->error) {
        lodepng_memset(*out, 0, outsize);
        state->error = postProcessScanlines(*out, scanlines, *w, *h, &state->info_png);
    }
    free(scanlines);
}


static void lodepng_decoder_settings_init(LodePNGDecoderSettings* settings)
{
    settings->color_convert = 1;
    settings->ignore_crc = 0;
    settings->ignore_critical = 0;
    settings->ignore_end = 0;
    lodepng_decompress_settings_init(&settings->zlibsettings);
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

/*read the information from the header and store it in the LodePNGInfo. return value is error*/
unsigned lodepng_inspect(unsigned* w, unsigned* h, LodePNGState* state, const unsigned char* in, size_t insize)
{
    unsigned width, height;
    LodePNGInfo* info = &state->info_png;
    if (insize == 0 || in == 0) {
        CERROR_RETURN_ERROR(state->error, 48); /*error: the given data is empty*/
    }
    if (insize < 33) {
        CERROR_RETURN_ERROR(state->error, 27); /*error: the data length is smaller than the length of a PNG header*/
    }

    /* when decoding a new PNG image, make sure all parameters created after previous decoding are reset */
    /* TODO: remove this. One should use a new LodePNGState for new sessions */
    lodepng_info_cleanup(info);
    lodepng_info_init(info);

    if (in[0] != 137 || in[1] != 80 || in[2] != 78 || in[3] != 71 || in[4] != 13 || in[5] != 10 || in[6] != 26 || in[7] != 10) {
        CERROR_RETURN_ERROR(state->error, 28); /*error: the first 8 bytes are not the correct PNG signature*/
    }
    if (lodepng_chunk_length(in + 8) != 13) {
        CERROR_RETURN_ERROR(state->error, 94); /*error: header size must be 13 bytes*/
    }
    if (!lodepng_chunk_type_equals(in + 8, "IHDR")) {
        CERROR_RETURN_ERROR(state->error, 29); /*error: it doesn't start with a IHDR chunk!*/
    }

    /*read the values given in the header*/
    width = lodepng_read32bitInt(&in[16]);
    height = lodepng_read32bitInt(&in[20]);
    /*TODO: remove the undocumented feature that allows to give null pointers to width or height*/
    if (w) *w = width;
    if (h) *h = height;
    info->color.bitdepth = in[24];
    info->color.colortype = (LodePNGColorType)in[25];
    info->compression_method = in[26];
    info->filter_method = in[27];
    info->interlace_method = in[28];

    /*errors returned only after the parsing so other values are still output*/

    /*error: invalid image size*/
    if (width == 0 || height == 0) CERROR_RETURN_ERROR(state->error, 93);
    /*error: invalid colortype or bitdepth combination*/
    state->error = checkColorValidity(info->color.colortype, info->color.bitdepth);
    if (state->error) return state->error;
    /*error: only compression method 0 is allowed in the specification*/
    if (info->compression_method != 0) CERROR_RETURN_ERROR(state->error, 32);
    /*error: only filter method 0 is allowed in the specification*/
    if (info->filter_method != 0) CERROR_RETURN_ERROR(state->error, 33);
    /*error: only interlace methods 0 and 1 exist in the specification*/
    if (info->interlace_method > 1) CERROR_RETURN_ERROR(state->error, 34);

#if 0 //thorvg don't use crc
    if (!state->decoder.ignore_crc) {
        unsigned CRC = lodepng_read32bitInt(&in[29]);
        unsigned checksum = lodepng_crc32(&in[12], 17);
        if (CRC != checksum) {
          CERROR_RETURN_ERROR(state->error, 57); /*invalid CRC*/
        }
    }
#endif
    return state->error;
}


unsigned lodepng_decode(unsigned char** out, unsigned* w, unsigned* h, LodePNGState* state, const unsigned char* in, size_t insize)
{
    *out = 0;
    decodeGeneric(out, w, h, state, in, insize);
    if (state->error) return state->error;
    if (!state->decoder.color_convert || lodepng_color_mode_equal(&state->info_raw, &state->info_png.color)) {
        /*same color type, no copying or converting of data needed*/
        /*store the info_png color settings on the info_raw so that the info_raw still reflects what colortype
        the raw image has to the end user*/
        if (!state->decoder.color_convert) {
            state->error = lodepng_color_mode_copy(&state->info_raw, &state->info_png.color);
            if (state->error) return state->error;
        }
    } else { /*color conversion needed*/
        unsigned char* data = *out;
        size_t outsize;

        /*TODO: check if this works according to the statement in the documentation: "The converter can convert
        from grayscale input color type, to 8-bit grayscale or grayscale with alpha"*/
        if (!(state->info_raw.colortype == LCT_RGB || state->info_raw.colortype == LCT_RGBA) && !(state->info_raw.bitdepth == 8)) {
            return 56; /*unsupported color mode conversion*/
        }

        outsize = lodepng_get_raw_size(*w, *h, &state->info_raw);
        *out = (unsigned char*)malloc(outsize);
        if (!(*out)) {
            state->error = 83; /*alloc fail*/
        }
        else state->error = lodepng_convert(*out, data, &state->info_raw, &state->info_png.color, *w, *h);
        free(data);
    }
    return state->error;
}


void lodepng_state_init(LodePNGState* state)
{
    lodepng_decoder_settings_init(&state->decoder);
    lodepng_color_mode_init(&state->info_raw);
    lodepng_info_init(&state->info_png);
    state->error = 1;
}


void lodepng_state_cleanup(LodePNGState* state)
{
    lodepng_color_mode_cleanup(&state->info_raw);
    lodepng_info_cleanup(&state->info_png);
}
