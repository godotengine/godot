//---------------------------------------------------------------------------------
//
//  Little Color Management System
//  Copyright (c) 1998-2017 Marti Maria Saguer
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//---------------------------------------------------------------------------------


#include "lcms2_internal.h"

#ifdef CMS_USE_BIG_ENDIAN

static
void byteReverse(cmsUInt8Number * buf, cmsUInt32Number longs)
{
    do {

        cmsUInt32Number t = _cmsAdjustEndianess32(*(cmsUInt32Number *) buf);
        *(cmsUInt32Number *) buf = t;
        buf += sizeof(cmsUInt32Number);

    } while (--longs);

}

#else
#define byteReverse(buf, len)
#endif


typedef struct {

    cmsUInt32Number buf[4];
    cmsUInt32Number bits[2];
    cmsUInt8Number in[64];
    cmsContext ContextID;

} _cmsMD5;

#define F1(x, y, z) (z ^ (x & (y ^ z)))
#define F2(x, y, z) F1(z, x, y)
#define F3(x, y, z) (x ^ y ^ z)
#define F4(x, y, z) (y ^ (x | ~z))

#define STEP(f, w, x, y, z, data, s) \
    ( w += f(x, y, z) + data,  w = w<<s | w>>(32-s),  w += x )


static
void MD5_Transform(cmsUInt32Number buf[4], cmsUInt32Number in[16])

{
    register cmsUInt32Number a, b, c, d;

    a = buf[0];
    b = buf[1];
    c = buf[2];
    d = buf[3];

    STEP(F1, a, b, c, d, in[0] + 0xd76aa478, 7);
    STEP(F1, d, a, b, c, in[1] + 0xe8c7b756, 12);
    STEP(F1, c, d, a, b, in[2] + 0x242070db, 17);
    STEP(F1, b, c, d, a, in[3] + 0xc1bdceee, 22);
    STEP(F1, a, b, c, d, in[4] + 0xf57c0faf, 7);
    STEP(F1, d, a, b, c, in[5] + 0x4787c62a, 12);
    STEP(F1, c, d, a, b, in[6] + 0xa8304613, 17);
    STEP(F1, b, c, d, a, in[7] + 0xfd469501, 22);
    STEP(F1, a, b, c, d, in[8] + 0x698098d8, 7);
    STEP(F1, d, a, b, c, in[9] + 0x8b44f7af, 12);
    STEP(F1, c, d, a, b, in[10] + 0xffff5bb1, 17);
    STEP(F1, b, c, d, a, in[11] + 0x895cd7be, 22);
    STEP(F1, a, b, c, d, in[12] + 0x6b901122, 7);
    STEP(F1, d, a, b, c, in[13] + 0xfd987193, 12);
    STEP(F1, c, d, a, b, in[14] + 0xa679438e, 17);
    STEP(F1, b, c, d, a, in[15] + 0x49b40821, 22);

    STEP(F2, a, b, c, d, in[1] + 0xf61e2562, 5);
    STEP(F2, d, a, b, c, in[6] + 0xc040b340, 9);
    STEP(F2, c, d, a, b, in[11] + 0x265e5a51, 14);
    STEP(F2, b, c, d, a, in[0] + 0xe9b6c7aa, 20);
    STEP(F2, a, b, c, d, in[5] + 0xd62f105d, 5);
    STEP(F2, d, a, b, c, in[10] + 0x02441453, 9);
    STEP(F2, c, d, a, b, in[15] + 0xd8a1e681, 14);
    STEP(F2, b, c, d, a, in[4] + 0xe7d3fbc8, 20);
    STEP(F2, a, b, c, d, in[9] + 0x21e1cde6, 5);
    STEP(F2, d, a, b, c, in[14] + 0xc33707d6, 9);
    STEP(F2, c, d, a, b, in[3] + 0xf4d50d87, 14);
    STEP(F2, b, c, d, a, in[8] + 0x455a14ed, 20);
    STEP(F2, a, b, c, d, in[13] + 0xa9e3e905, 5);
    STEP(F2, d, a, b, c, in[2] + 0xfcefa3f8, 9);
    STEP(F2, c, d, a, b, in[7] + 0x676f02d9, 14);
    STEP(F2, b, c, d, a, in[12] + 0x8d2a4c8a, 20);

    STEP(F3, a, b, c, d, in[5] + 0xfffa3942, 4);
    STEP(F3, d, a, b, c, in[8] + 0x8771f681, 11);
    STEP(F3, c, d, a, b, in[11] + 0x6d9d6122, 16);
    STEP(F3, b, c, d, a, in[14] + 0xfde5380c, 23);
    STEP(F3, a, b, c, d, in[1] + 0xa4beea44, 4);
    STEP(F3, d, a, b, c, in[4] + 0x4bdecfa9, 11);
    STEP(F3, c, d, a, b, in[7] + 0xf6bb4b60, 16);
    STEP(F3, b, c, d, a, in[10] + 0xbebfbc70, 23);
    STEP(F3, a, b, c, d, in[13] + 0x289b7ec6, 4);
    STEP(F3, d, a, b, c, in[0] + 0xeaa127fa, 11);
    STEP(F3, c, d, a, b, in[3] + 0xd4ef3085, 16);
    STEP(F3, b, c, d, a, in[6] + 0x04881d05, 23);
    STEP(F3, a, b, c, d, in[9] + 0xd9d4d039, 4);
    STEP(F3, d, a, b, c, in[12] + 0xe6db99e5, 11);
    STEP(F3, c, d, a, b, in[15] + 0x1fa27cf8, 16);
    STEP(F3, b, c, d, a, in[2] + 0xc4ac5665, 23);

    STEP(F4, a, b, c, d, in[0] + 0xf4292244, 6);
    STEP(F4, d, a, b, c, in[7] + 0x432aff97, 10);
    STEP(F4, c, d, a, b, in[14] + 0xab9423a7, 15);
    STEP(F4, b, c, d, a, in[5] + 0xfc93a039, 21);
    STEP(F4, a, b, c, d, in[12] + 0x655b59c3, 6);
    STEP(F4, d, a, b, c, in[3] + 0x8f0ccc92, 10);
    STEP(F4, c, d, a, b, in[10] + 0xffeff47d, 15);
    STEP(F4, b, c, d, a, in[1] + 0x85845dd1, 21);
    STEP(F4, a, b, c, d, in[8] + 0x6fa87e4f, 6);
    STEP(F4, d, a, b, c, in[15] + 0xfe2ce6e0, 10);
    STEP(F4, c, d, a, b, in[6] + 0xa3014314, 15);
    STEP(F4, b, c, d, a, in[13] + 0x4e0811a1, 21);
    STEP(F4, a, b, c, d, in[4] + 0xf7537e82, 6);
    STEP(F4, d, a, b, c, in[11] + 0xbd3af235, 10);
    STEP(F4, c, d, a, b, in[2] + 0x2ad7d2bb, 15);
    STEP(F4, b, c, d, a, in[9] + 0xeb86d391, 21);

    buf[0] += a;
    buf[1] += b;
    buf[2] += c;
    buf[3] += d;
}


// Create a MD5 object
static
cmsHANDLE  MD5alloc(cmsContext ContextID)
{
    _cmsMD5* ctx = (_cmsMD5*) _cmsMallocZero(ContextID, sizeof(_cmsMD5));
    if (ctx == NULL) return NULL;

    ctx ->ContextID = ContextID;

    ctx->buf[0] = 0x67452301;
    ctx->buf[1] = 0xefcdab89;
    ctx->buf[2] = 0x98badcfe;
    ctx->buf[3] = 0x10325476;

    ctx->bits[0] = 0;
    ctx->bits[1] = 0;

    return (cmsHANDLE) ctx;
}


static
void MD5add(cmsHANDLE Handle, cmsUInt8Number* buf, cmsUInt32Number len)
{
    _cmsMD5* ctx = (_cmsMD5*) Handle;
    cmsUInt32Number t;

    t = ctx->bits[0];
    if ((ctx->bits[0] = t + (len << 3)) < t)
        ctx->bits[1]++;

    ctx->bits[1] += len >> 29;

    t = (t >> 3) & 0x3f;

    if (t) {

        cmsUInt8Number *p = (cmsUInt8Number *) ctx->in + t;

        t = 64 - t;
        if (len < t) {
            memmove(p, buf, len);
            return;
        }

        memmove(p, buf, t);
        byteReverse(ctx->in, 16);

        MD5_Transform(ctx->buf, (cmsUInt32Number *) ctx->in);
        buf += t;
        len -= t;
    }

    while (len >= 64) {
        memmove(ctx->in, buf, 64);
        byteReverse(ctx->in, 16);
        MD5_Transform(ctx->buf, (cmsUInt32Number *) ctx->in);
        buf += 64;
        len -= 64;
    }

    memmove(ctx->in, buf, len);
}

// Destroy the object and return the checksum
static
void MD5finish(cmsProfileID* ProfileID,  cmsHANDLE Handle)
{
    _cmsMD5* ctx = (_cmsMD5*) Handle;
    cmsUInt32Number count;
    cmsUInt8Number *p;

    count = (ctx->bits[0] >> 3) & 0x3F;

    p = ctx->in + count;
    *p++ = 0x80;

    count = 64 - 1 - count;

    if (count < 8) {

        memset(p, 0, count);
        byteReverse(ctx->in, 16);
        MD5_Transform(ctx->buf, (cmsUInt32Number *) ctx->in);

        memset(ctx->in, 0, 56);
    } else {
        memset(p, 0, count - 8);
    }
    byteReverse(ctx->in, 14);

    ((cmsUInt32Number *) ctx->in)[14] = ctx->bits[0];
    ((cmsUInt32Number *) ctx->in)[15] = ctx->bits[1];

    MD5_Transform(ctx->buf, (cmsUInt32Number *) ctx->in);

    byteReverse((cmsUInt8Number *) ctx->buf, 4);
    memmove(ProfileID ->ID8, ctx->buf, 16);

    _cmsFree(ctx ->ContextID, ctx);
}



// Assuming io points to an ICC profile, compute and store MD5 checksum
// In the header, rendering intentent, attributes and ID should be set to zero
// before computing MD5 checksum (per 6.1.13 in ICC spec)

cmsBool CMSEXPORT cmsMD5computeID(cmsHPROFILE hProfile)
{
    cmsContext   ContextID;
    cmsUInt32Number BytesNeeded;
    cmsUInt8Number* Mem = NULL;
    cmsHANDLE  MD5 = NULL;
    _cmsICCPROFILE* Icc = (_cmsICCPROFILE*) hProfile;
    _cmsICCPROFILE Keep;

    _cmsAssert(hProfile != NULL);

    ContextID = cmsGetProfileContextID(hProfile);

    // Save a copy of the profile header
    memmove(&Keep, Icc, sizeof(_cmsICCPROFILE));

    // Set RI, attributes and ID
    memset(&Icc ->attributes, 0, sizeof(Icc ->attributes));
    Icc ->RenderingIntent = 0;
    memset(&Icc ->ProfileID, 0, sizeof(Icc ->ProfileID));

    // Compute needed storage
    if (!cmsSaveProfileToMem(hProfile, NULL, &BytesNeeded)) goto Error;

    // Allocate memory
    Mem = (cmsUInt8Number*) _cmsMalloc(ContextID, BytesNeeded);
    if (Mem == NULL) goto Error;

    // Save to temporary storage
    if (!cmsSaveProfileToMem(hProfile, Mem, &BytesNeeded)) goto Error;

    // Create MD5 object
    MD5 = MD5alloc(ContextID);
    if (MD5 == NULL) goto Error;

    // Add all bytes
    MD5add(MD5, Mem, BytesNeeded);

    // Temp storage is no longer needed
    _cmsFree(ContextID, Mem);

    // Restore header
    memmove(Icc, &Keep, sizeof(_cmsICCPROFILE));

    // And store the ID
    MD5finish(&Icc ->ProfileID,  MD5);
    return TRUE;

Error:

    // Free resources as something went wrong
    // "MD5" cannot be other than NULL here, so no need to free it
    if (Mem != NULL) _cmsFree(ContextID, Mem);
    memmove(Icc, &Keep, sizeof(_cmsICCPROFILE));
    return FALSE;
}

