// Licensed under the Mach engine license (Apache or MIT at your choosing.) For details
// and a copy of this license, see https://github.com/hexops/mach/blob/main/LICENSE
// This copyright header, and a copy of the above open source licenses, must be provided
// with any redistributions of this software.

#include <string.h>
#include "MachSiegbertVogtDXCSA.h"

// Note: If this algorithm looks familiar to you, it is - but not exactly. Fun!

static uint8_t PADDING[64] =
{
    0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))
#define ROTL(x, n) (((x) << (n)) | ((x) >> (32-(n))))

#define FF(a, b, c, d, x, s, ac) { (a) += F((b), (c), (d)) + (x) + (uint32_t)(ac); (a) = ROTL((a), (s)); (a) += (b); }
#define GG(a, b, c, d, x, s, ac) { (a) += G((b), (c), (d)) + (x) + (uint32_t)(ac); (a) = ROTL((a), (s)); (a) += (b); }
#define HH(a, b, c, d, x, s, ac) { (a) += H((b), (c), (d)) + (x) + (uint32_t)(ac); (a) = ROTL((a), (s)); (a) += (b); }
#define II(a, b, c, d, x, s, ac) { (a) += I((b), (c), (d)) + (x) + (uint32_t)(ac); (a) = ROTL((a), (s)); (a) += (b); }

// Round 1
#define S11 7
#define S12 12
#define S13 17
#define S14 22

// Round 2
#define S21 5
#define S22 9
#define S23 14
#define S24 20

// Round 3
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6

// Round 4
#define S42 10
#define S43 15
#define S44 21

// Message digest computation
typedef struct Context {
    uint32_t i[2];      // bits handled mod 2^64
    uint32_t buf[4];    // scratch buffer
    uint8_t in[64];     // input buffer
    uint8_t digest[16]; // digest after final()
} Context;

static void transform(uint32_t* buf, uint32_t* in) {
    uint32_t a = buf[0], b = buf[1], c = buf[2], d = buf[3];

    // round 1
    FF(a, b, c, d, in[ 0], S11, (uint32_t) 3614090360u);
    FF(d, a, b, c, in[ 1], S12, (uint32_t) 3905402710u);
    FF(c, d, a, b, in[ 2], S13, (uint32_t)  606105819u);
    FF(b, c, d, a, in[ 3], S14, (uint32_t) 3250441966u);
    FF(a, b, c, d, in[ 4], S11, (uint32_t) 4118548399u);
    FF(d, a, b, c, in[ 5], S12, (uint32_t) 1200080426u);
    FF(c, d, a, b, in[ 6], S13, (uint32_t) 2821735955u);
    FF(b, c, d, a, in[ 7], S14, (uint32_t) 4249261313u);
    FF(a, b, c, d, in[ 8], S11, (uint32_t) 1770035416u);
    FF(d, a, b, c, in[ 9], S12, (uint32_t) 2336552879u);
    FF(c, d, a, b, in[10], S13, (uint32_t) 4294925233u);
    FF(b, c, d, a, in[11], S14, (uint32_t) 2304563134u);
    FF(a, b, c, d, in[12], S11, (uint32_t) 1804603682u);
    FF(d, a, b, c, in[13], S12, (uint32_t) 4254626195u);
    FF(c, d, a, b, in[14], S13, (uint32_t) 2792965006u);
    FF(b, c, d, a, in[15], S14, (uint32_t) 1236535329u);

    // round 2
    GG(a, b, c, d, in[ 1], S21, (uint32_t) 4129170786u);
    GG(d, a, b, c, in[ 6], S22, (uint32_t) 3225465664u);
    GG(c, d, a, b, in[11], S23, (uint32_t)  643717713u);
    GG(b, c, d, a, in[ 0], S24, (uint32_t) 3921069994u);
    GG(a, b, c, d, in[ 5], S21, (uint32_t) 3593408605u);
    GG(d, a, b, c, in[10], S22, (uint32_t)   38016083u);
    GG(c, d, a, b, in[15], S23, (uint32_t) 3634488961u);
    GG(b, c, d, a, in[ 4], S24, (uint32_t) 3889429448u);
    GG(a, b, c, d, in[ 9], S21, (uint32_t)  568446438u);
    GG(d, a, b, c, in[14], S22, (uint32_t) 3275163606u);
    GG(c, d, a, b, in[ 3], S23, (uint32_t) 4107603335u);
    GG(b, c, d, a, in[ 8], S24, (uint32_t) 1163531501u);
    GG(a, b, c, d, in[13], S21, (uint32_t) 2850285829u);
    GG(d, a, b, c, in[ 2], S22, (uint32_t) 4243563512u);
    GG(c, d, a, b, in[ 7], S23, (uint32_t) 1735328473u);
    GG(b, c, d, a, in[12], S24, (uint32_t) 2368359562u);

    // round 3
    HH(a, b, c, d, in[ 5], S31, (uint32_t) 4294588738u);
    HH(d, a, b, c, in[ 8], S32, (uint32_t) 2272392833u);
    HH(c, d, a, b, in[11], S33, (uint32_t) 1839030562u);
    HH(b, c, d, a, in[14], S34, (uint32_t) 4259657740u);
    HH(a, b, c, d, in[ 1], S31, (uint32_t) 2763975236u);
    HH(d, a, b, c, in[ 4], S32, (uint32_t) 1272893353u);
    HH(c, d, a, b, in[ 7], S33, (uint32_t) 4139469664u);
    HH(b, c, d, a, in[10], S34, (uint32_t) 3200236656u);
    HH(a, b, c, d, in[13], S31, (uint32_t)  681279174u);
    HH(d, a, b, c, in[ 0], S32, (uint32_t) 3936430074u);
    HH(c, d, a, b, in[ 3], S33, (uint32_t) 3572445317u);
    HH(b, c, d, a, in[ 6], S34, (uint32_t)   76029189u);
    HH(a, b, c, d, in[ 9], S31, (uint32_t) 3654602809u);
    HH(d, a, b, c, in[12], S32, (uint32_t) 3873151461u);
    HH(c, d, a, b, in[15], S33, (uint32_t)  530742520u);
    HH(b, c, d, a, in[ 2], S34, (uint32_t) 3299628645u);

    // round 4
    II(a, b, c, d, in[ 0], S41, (uint32_t) 4096336452u);
    II(d, a, b, c, in[ 7], S42, (uint32_t) 1126891415u);
    II(c, d, a, b, in[14], S43, (uint32_t) 2878612391u);
    II(b, c, d, a, in[ 5], S44, (uint32_t) 4237533241u);
    II(a, b, c, d, in[12], S41, (uint32_t) 1700485571u);
    II(d, a, b, c, in[ 3], S42, (uint32_t) 2399980690u);
    II(c, d, a, b, in[10], S43, (uint32_t) 4293915773u);
    II(b, c, d, a, in[ 1], S44, (uint32_t) 2240044497u);
    II(a, b, c, d, in[ 8], S41, (uint32_t) 1873313359u);
    II(d, a, b, c, in[15], S42, (uint32_t) 4264355552u);
    II(c, d, a, b, in[ 6], S43, (uint32_t) 2734768916u);
    II(b, c, d, a, in[13], S44, (uint32_t) 1309151649u);
    II(a, b, c, d, in[ 4], S41, (uint32_t) 4149444226u);
    II(d, a, b, c, in[11], S42, (uint32_t) 3174756917u);
    II(c, d, a, b, in[ 2], S43, (uint32_t)  718787259u);
    II(b, c, d, a, in[ 9], S44, (uint32_t) 3951481745u);

    buf[0] += a;
    buf[1] += b;
    buf[2] += c;
    buf[3] += d;
}

void update(Context* ctx, uint8_t* in_buf, uint32_t in_len) {
    uint32_t in[16];
    int32_t mdi = 0;
    uint32_t i = 0;
    uint32_t ii = 0;

    mdi = (int32_t)((ctx->i[0] >> 3) & 0x3F); // number of bytes mod 64

    // update # of bits
    if ((ctx->i[0] + ((uint32_t)in_len << 3)) < ctx->i[0]) ctx->i[1]++;

    ctx->i[0] += ((uint32_t)in_len << 3);
    ctx->i[1] += ((uint32_t)in_len >> 29);

    while (in_len--) {
        ctx->in[mdi++] = *in_buf++; // add new char, increment mdi
        if (mdi == 0x40) {
            for (i = 0, ii = 0; i < 16; i++, ii += 4) {
                in[i] = (((uint32_t)ctx->in[ii + 3]) << 24);
                in[i] |= (((uint32_t)ctx->in[ii + 2]) << 16);
                in[i] |= (((uint32_t)ctx->in[ii + 1]) << 8);
                in[i] |= ((uint32_t)ctx->in[ii]);
            }
            transform(ctx->buf, in);
            mdi = 0;
        }
    }
}

static const uint32_t secret_hash_offset = 0x14;

void machSiegbertVogtDXCSA(uint8_t* data, uint32_t data_size, uint32_t secret_out[4])
{
    // Initialize ctx
    uint64_t rng = 0;
    Context ctx;
    ctx.i[0] = ctx.i[1] = (uint32_t)0;
    ctx.buf[0] = (uint32_t)0x67452301 + (rng * 11);
    ctx.buf[1] = (uint32_t)0xefcdab89 + (rng * 71);
    ctx.buf[2] = (uint32_t)0x98badcfe + (rng * 37);
    ctx.buf[3] = (uint32_t)0x10325476 + (rng * 97);

    // Note: first 4 bytes of bin are "DXBC" (IL) header/file-magic, then 16-byte signing hash,
    // then remainder of the file contents.
    data_size -= secret_hash_offset;
    data += secret_hash_offset;

    uint32_t num_bits = data_size * 8;

    uint32_t full_chunks_size = data_size & 0xffffffc0;
    update(&ctx, data, full_chunks_size);

    uint32_t last_chunk_size = data_size - full_chunks_size;
    uint32_t padding_size = 64  - last_chunk_size;
    uint8_t* last_chunk_data = data + full_chunks_size;

    if (last_chunk_size >= 56) {
        update(&ctx, last_chunk_data, last_chunk_size);

        // Pad to 56 mod 64
        update(&ctx, PADDING, padding_size);

        uint32_t in[16];
        memset(in, 0, sizeof(in));
        in[0] = num_bits;
        in[15] = (num_bits >> 2) | 1;

        transform(ctx.buf, in);
    } else {
        update(&ctx, (uint8_t*) &num_bits, 4);

        if (last_chunk_size) update(&ctx, last_chunk_data, last_chunk_size);

        // Adjust for the space used for num_bits
        last_chunk_size += sizeof(uint32_t);
        padding_size -= sizeof(uint32_t);

        // Pad to 56 mod 64
        memcpy(&ctx.in[last_chunk_size], PADDING, padding_size);

        ((uint32_t*)ctx.in)[15] = (num_bits >> 2) | 1;

        uint32_t in[16];
        memcpy(in, ctx.in, 64);
        transform(ctx.buf, in);
    }
    memcpy(secret_out, ctx.buf, 4 * sizeof(uint32_t));
}