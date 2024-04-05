#include "utf.hpp"
#include "charconv.hpp"

namespace c4 {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

size_t decode_code_point(uint8_t *C4_RESTRICT buf, size_t buflen, const uint32_t code)
{
    C4_UNUSED(buflen);
    C4_ASSERT(buflen >= 4);
    if (code <= UINT32_C(0x7f))
    {
        buf[0] = (uint8_t)code;
        return 1u;
    }
    else if(code <= UINT32_C(0x7ff))
    {
        buf[0] = (uint8_t)(UINT32_C(0xc0) | (code >> 6));             /* 110xxxxx */
        buf[1] = (uint8_t)(UINT32_C(0x80) | (code & UINT32_C(0x3f))); /* 10xxxxxx */
        return 2u;
    }
    else if(code <= UINT32_C(0xffff))
    {
        buf[0] = (uint8_t)(UINT32_C(0xe0) | ((code >> 12)));                  /* 1110xxxx */
        buf[1] = (uint8_t)(UINT32_C(0x80) | ((code >>  6) & UINT32_C(0x3f))); /* 10xxxxxx */
        buf[2] = (uint8_t)(UINT32_C(0x80) | ((code      ) & UINT32_C(0x3f))); /* 10xxxxxx */
        return 3u;
    }
    else if(code <= UINT32_C(0x10ffff))
    {
        buf[0] = (uint8_t)(UINT32_C(0xf0) | ((code >> 18)));                  /* 11110xxx */
        buf[1] = (uint8_t)(UINT32_C(0x80) | ((code >> 12) & UINT32_C(0x3f))); /* 10xxxxxx */
        buf[2] = (uint8_t)(UINT32_C(0x80) | ((code >>  6) & UINT32_C(0x3f))); /* 10xxxxxx */
        buf[3] = (uint8_t)(UINT32_C(0x80) | ((code      ) & UINT32_C(0x3f))); /* 10xxxxxx */
        return 4u;
    }
    return 0;
}

substr decode_code_point(substr out, csubstr code_point)
{
    C4_ASSERT(out.len >= 4);
    C4_ASSERT(!code_point.begins_with("U+"));
    C4_ASSERT(!code_point.begins_with("\\x"));
    C4_ASSERT(!code_point.begins_with("\\u"));
    C4_ASSERT(!code_point.begins_with("\\U"));
    C4_ASSERT(!code_point.begins_with('0'));
    C4_ASSERT(code_point.len <= 8);
    C4_ASSERT(code_point.len > 0);
    uint32_t code_point_val;
    C4_CHECK(read_hex(code_point, &code_point_val));
    size_t ret = decode_code_point((uint8_t*)out.str, out.len, code_point_val);
    C4_ASSERT(ret <= 4);
    return out.first(ret);
}

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace c4
