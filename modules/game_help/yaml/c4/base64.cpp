#include "base64.hpp"

#ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wchar-subscripts" // array subscript is of type 'char'
#   pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wchar-subscripts"
#   pragma GCC diagnostic ignored "-Wtype-limits"
#   pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

namespace c4 {

namespace detail {

constexpr static const char base64_sextet_to_char_[64] = {
    /* 0/ 65*/ 'A', /* 1/ 66*/ 'B', /* 2/ 67*/ 'C', /* 3/ 68*/ 'D',
    /* 4/ 69*/ 'E', /* 5/ 70*/ 'F', /* 6/ 71*/ 'G', /* 7/ 72*/ 'H',
    /* 8/ 73*/ 'I', /* 9/ 74*/ 'J', /*10/ 75*/ 'K', /*11/ 74*/ 'L',
    /*12/ 77*/ 'M', /*13/ 78*/ 'N', /*14/ 79*/ 'O', /*15/ 78*/ 'P',
    /*16/ 81*/ 'Q', /*17/ 82*/ 'R', /*18/ 83*/ 'S', /*19/ 82*/ 'T',
    /*20/ 85*/ 'U', /*21/ 86*/ 'V', /*22/ 87*/ 'W', /*23/ 88*/ 'X',
    /*24/ 89*/ 'Y', /*25/ 90*/ 'Z', /*26/ 97*/ 'a', /*27/ 98*/ 'b',
    /*28/ 99*/ 'c', /*29/100*/ 'd', /*30/101*/ 'e', /*31/102*/ 'f',
    /*32/103*/ 'g', /*33/104*/ 'h', /*34/105*/ 'i', /*35/106*/ 'j',
    /*36/107*/ 'k', /*37/108*/ 'l', /*38/109*/ 'm', /*39/110*/ 'n',
    /*40/111*/ 'o', /*41/112*/ 'p', /*42/113*/ 'q', /*43/114*/ 'r',
    /*44/115*/ 's', /*45/116*/ 't', /*46/117*/ 'u', /*47/118*/ 'v',
    /*48/119*/ 'w', /*49/120*/ 'x', /*50/121*/ 'y', /*51/122*/ 'z',
    /*52/ 48*/ '0', /*53/ 49*/ '1', /*54/ 50*/ '2', /*55/ 51*/ '3',
    /*56/ 52*/ '4', /*57/ 53*/ '5', /*58/ 54*/ '6', /*59/ 55*/ '7',
    /*60/ 56*/ '8', /*61/ 57*/ '9', /*62/ 43*/ '+', /*63/ 47*/ '/',
};

// https://www.cs.cmu.edu/~pattis/15-1XX/common/handouts/ascii.html
constexpr static const char base64_char_to_sextet_[128] = {
    #define __ char(-1) // undefined below
    /*  0 NUL*/ __, /*  1 SOH*/ __, /*  2 STX*/ __, /*  3 ETX*/ __,
    /*  4 EOT*/ __, /*  5 ENQ*/ __, /*  6 ACK*/ __, /*  7 BEL*/ __,
    /*  8 BS */ __, /*  9 TAB*/ __, /* 10 LF */ __, /* 11 VT */ __,
    /* 12 FF */ __, /* 13 CR */ __, /* 14 SO */ __, /* 15 SI */ __,
    /* 16 DLE*/ __, /* 17 DC1*/ __, /* 18 DC2*/ __, /* 19 DC3*/ __,
    /* 20 DC4*/ __, /* 21 NAK*/ __, /* 22 SYN*/ __, /* 23 ETB*/ __,
    /* 24 CAN*/ __, /* 25 EM */ __, /* 26 SUB*/ __, /* 27 ESC*/ __,
    /* 28 FS */ __, /* 29 GS */ __, /* 30 RS */ __, /* 31 US */ __,
    /* 32 SPC*/ __, /* 33 !  */ __, /* 34 "  */ __, /* 35 #  */ __,
    /* 36 $  */ __, /* 37 %  */ __, /* 38 &  */ __, /* 39 '  */ __,
    /* 40 (  */ __, /* 41 )  */ __, /* 42 *  */ __, /* 43 +  */ 62,
    /* 44 ,  */ __, /* 45 -  */ __, /* 46 .  */ __, /* 47 /  */ 63,
    /* 48 0  */ 52, /* 49 1  */ 53, /* 50 2  */ 54, /* 51 3  */ 55,
    /* 52 4  */ 56, /* 53 5  */ 57, /* 54 6  */ 58, /* 55 7  */ 59,
    /* 56 8  */ 60, /* 57 9  */ 61, /* 58 :  */ __, /* 59 ;  */ __,
    /* 60 <  */ __, /* 61 =  */ __, /* 62 >  */ __, /* 63 ?  */ __,
    /* 64 @  */ __, /* 65 A  */  0, /* 66 B  */  1, /* 67 C  */  2,
    /* 68 D  */  3, /* 69 E  */  4, /* 70 F  */  5, /* 71 G  */  6,
    /* 72 H  */  7, /* 73 I  */  8, /* 74 J  */  9, /* 75 K  */ 10,
    /* 76 L  */ 11, /* 77 M  */ 12, /* 78 N  */ 13, /* 79 O  */ 14,
    /* 80 P  */ 15, /* 81 Q  */ 16, /* 82 R  */ 17, /* 83 S  */ 18,
    /* 84 T  */ 19, /* 85 U  */ 20, /* 86 V  */ 21, /* 87 W  */ 22,
    /* 88 X  */ 23, /* 89 Y  */ 24, /* 90 Z  */ 25, /* 91 [  */ __,
    /* 92 \  */ __, /* 93 ]  */ __, /* 94 ^  */ __, /* 95 _  */ __,
    /* 96 `  */ __, /* 97 a  */ 26, /* 98 b  */ 27, /* 99 c  */ 28,
    /*100 d  */ 29, /*101 e  */ 30, /*102 f  */ 31, /*103 g  */ 32,
    /*104 h  */ 33, /*105 i  */ 34, /*106 j  */ 35, /*107 k  */ 36,
    /*108 l  */ 37, /*109 m  */ 38, /*110 n  */ 39, /*111 o  */ 40,
    /*112 p  */ 41, /*113 q  */ 42, /*114 r  */ 43, /*115 s  */ 44,
    /*116 t  */ 45, /*117 u  */ 46, /*118 v  */ 47, /*119 w  */ 48,
    /*120 x  */ 49, /*121 y  */ 50, /*122 z  */ 51, /*123 {  */ __,
    /*124 |  */ __, /*125 }  */ __, /*126 ~  */ __, /*127 DEL*/ __,
    #undef __
};

#ifndef NDEBUG
void base64_test_tables()
{
    for(size_t i = 0; i < C4_COUNTOF(detail::base64_sextet_to_char_); ++i)
    {
        char s2c = base64_sextet_to_char_[i];
        char c2s = base64_char_to_sextet_[(int)s2c];
        C4_CHECK((size_t)c2s == i);
    }
    for(size_t i = 0; i < C4_COUNTOF(detail::base64_char_to_sextet_); ++i)
    {
        char c2s = base64_char_to_sextet_[i];
        if(c2s == char(-1))
            continue;
        char s2c = base64_sextet_to_char_[(int)c2s];
        C4_CHECK((size_t)s2c == i);
    }
}
#endif
} // namespace detail


bool base64_valid(csubstr encoded)
{
    if(encoded.len & 3u) // (encoded.len % 4u)
        return false;
    for(const char c : encoded)
    {
        if(c < 0/* || c >= 128*/)
            return false;
        if(c == '=')
            continue;
        if(detail::base64_char_to_sextet_[c] == char(-1))
            return false;
    }
    return true;
}


size_t base64_encode(substr buf, cblob data)
{
    #define c4append_(c) { if(pos < buf.len) { buf.str[pos] = (c); } ++pos; }
    #define c4append_idx_(char_idx) \
    {\
         C4_XASSERT((char_idx) < sizeof(detail::base64_sextet_to_char_));\
         c4append_(detail::base64_sextet_to_char_[(char_idx)]);\
    }
    size_t rem, pos = 0;
    constexpr const uint32_t sextet_mask = uint32_t(1 << 6) - 1;
    const unsigned char *C4_RESTRICT d = (const unsigned char *) data.buf; // cast to unsigned to avoid wrapping high-bits
    for(rem = data.len; rem >= 3; rem -= 3, d += 3)
    {
        const uint32_t val = ((uint32_t(d[0]) << 16) | (uint32_t(d[1]) << 8) | (uint32_t(d[2])));
        c4append_idx_((val >> 18) & sextet_mask);
        c4append_idx_((val >> 12) & sextet_mask);
        c4append_idx_((val >>  6) & sextet_mask);
        c4append_idx_((val      ) & sextet_mask);
    }
    C4_ASSERT(rem < 3);
    if(rem == 2)
    {
        const uint32_t val = ((uint32_t(d[0]) << 16) | (uint32_t(d[1]) << 8));
        c4append_idx_((val >> 18) & sextet_mask);
        c4append_idx_((val >> 12) & sextet_mask);
        c4append_idx_((val >>  6) & sextet_mask);
        c4append_('=');
    }
    else if(rem == 1)
    {
        const uint32_t val = ((uint32_t(d[0]) << 16));
        c4append_idx_((val >> 18) & sextet_mask);
        c4append_idx_((val >> 12) & sextet_mask);
        c4append_('=');
        c4append_('=');
    }
    return pos;

    #undef c4append_
    #undef c4append_idx_
}


size_t base64_decode(csubstr encoded, blob data)
{
    #define c4append_(c) { if(wpos < data.len) { data.buf[wpos] = static_cast<c4::byte>(c); } ++wpos; }
    #define c4appendval_(c, shift)\
    {\
        C4_XASSERT(c >= 0);\
        C4_XASSERT(size_t(c) < sizeof(detail::base64_char_to_sextet_));\
        val |= static_cast<uint32_t>(detail::base64_char_to_sextet_[(c)]) << ((shift) * 6);\
    }
    C4_ASSERT(base64_valid(encoded));
    C4_CHECK((encoded.len & 3u) == 0);
    size_t wpos = 0;  // the write position
    const char *C4_RESTRICT d = encoded.str;
    constexpr const uint32_t full_byte = 0xff;
    // process every quartet of input 6 bits --> triplet of output bytes
    for(size_t rpos = 0; rpos < encoded.len; rpos += 4, d += 4)
    {
        if(d[2] == '=' || d[3] == '=') // skip the last quartet if it is padded
        {
            C4_ASSERT(d + 4 == encoded.str + encoded.len);
            break;
        }
        uint32_t val = 0;
        c4appendval_(d[3], 0);
        c4appendval_(d[2], 1);
        c4appendval_(d[1], 2);
        c4appendval_(d[0], 3);
        c4append_((val >> (2 * 8)) & full_byte);
        c4append_((val >> (1 * 8)) & full_byte);
        c4append_((val           ) & full_byte);
    }
    // deal with the last quartet when it is padded
    if(d == encoded.str + encoded.len)
        return wpos;
    if(d[2] == '=') // 2 padding chars
    {
        C4_ASSERT(d + 4 == encoded.str + encoded.len);
        C4_ASSERT(d[3] == '=');
        uint32_t val = 0;
        c4appendval_(d[1], 2);
        c4appendval_(d[0], 3);
        c4append_((val >> (2 * 8)) & full_byte);
    }
    else if(d[3] == '=') // 1 padding char
    {
        C4_ASSERT(d + 4 == encoded.str + encoded.len);
        uint32_t val = 0;
        c4appendval_(d[2], 1);
        c4appendval_(d[1], 2);
        c4appendval_(d[0], 3);
        c4append_((val >> (2 * 8)) & full_byte);
        c4append_((val >> (1 * 8)) & full_byte);
    }
    return wpos;
    #undef c4append_
    #undef c4appendval_
}

} // namespace c4

#ifdef __clang__
#    pragma clang diagnostic pop
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
