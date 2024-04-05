#ifndef _C4_BASE64_HPP_
#define _C4_BASE64_HPP_

/** @file base64.hpp encoding/decoding for base64.
 * @see https://en.wikipedia.org/wiki/Base64
 * @see https://www.base64encode.org/
 * */

#include "charconv.hpp"
#include "blob.hpp"

namespace c4 {


/** check that the given buffer is a valid base64 encoding
 * @see https://en.wikipedia.org/wiki/Base64 */
C4CORE_EXPORT bool base64_valid(csubstr encoded);


/** base64-encode binary data.
 * @param encoded [out] output buffer for encoded data
 * @param data [in] the input buffer with the binary data
 *
 * @return the number of bytes needed to return the output (ie the
 * required size for @p encoded). No writes occur beyond the end of
 * the output buffer, so it is safe to do a speculative call where the
 * encoded buffer is empty, or maybe too small. The caller should
 * ensure that the returned size is smaller than the size of the
 * encoded buffer.
 *
 * @note the result depends on endianness. If transfer between
 * little/big endian systems is desired, the caller should normalize
 * @p data before encoding.
 *
 * @see https://en.wikipedia.org/wiki/Base64 */
C4CORE_EXPORT size_t base64_encode(substr encoded, cblob data);


/** decode the base64 encoding in the given buffer
 * @param encoded [in] the encoded base64
 * @param data [out] the output buffer
 *
 * @return the number of bytes needed to return the output (ie the
 * required size for @p data). No writes occur beyond the end of the
 * output buffer, so it is safe to do a speculative call where the
 * data buffer is empty, or maybe too small. The caller should ensure
 * that the returned size is smaller than the size of the data buffer.
 *
 * @note the result depends on endianness. If transfer between
 * little/big endian systems is desired, the caller should normalize
 * @p data after decoding.
 *
 * @see https://en.wikipedia.org/wiki/Base64 */
C4CORE_EXPORT size_t base64_decode(csubstr encoded, blob data);


namespace fmt {

template<typename CharOrConstChar>
struct base64_wrapper_
{
    blob_<CharOrConstChar> data;
    base64_wrapper_() : data() {}
    base64_wrapper_(blob_<CharOrConstChar> blob) : data(blob) {}
};
/** a tag type to mark a payload as base64-encoded */
using const_base64_wrapper = base64_wrapper_<cbyte>;
/** a tag type to mark a payload to be encoded as base64 */
using base64_wrapper = base64_wrapper_<byte>;


/** mark a variable to be written in base64 format */
template<class ...Args>
C4_ALWAYS_INLINE const_base64_wrapper cbase64(Args const& C4_RESTRICT ...args)
{
    return const_base64_wrapper(cblob(args...));
}
/** mark a csubstr to be written in base64 format */
C4_ALWAYS_INLINE const_base64_wrapper cbase64(csubstr s)
{
    return const_base64_wrapper(cblob(s.str, s.len));
}
/** mark a variable to be written in base64 format */
template<class ...Args>
C4_ALWAYS_INLINE const_base64_wrapper base64(Args const& C4_RESTRICT ...args)
{
    return const_base64_wrapper(cblob(args...));
}
/** mark a csubstr to be written in base64 format */
C4_ALWAYS_INLINE const_base64_wrapper base64(csubstr s)
{
    return const_base64_wrapper(cblob(s.str, s.len));
}

/** mark a variable to be read in base64 format */
template<class ...Args>
C4_ALWAYS_INLINE base64_wrapper base64(Args &... args)
{
    return base64_wrapper(blob(args...));
}
/** mark a variable to be read in base64 format */
C4_ALWAYS_INLINE base64_wrapper base64(substr s)
{
    return base64_wrapper(blob(s.str, s.len));
}

} // namespace fmt


/** write a variable in base64 format */
inline size_t to_chars(substr buf, fmt::const_base64_wrapper b)
{
    return base64_encode(buf, b.data);
}

/** read a variable in base64 format */
inline size_t from_chars(csubstr buf, fmt::base64_wrapper *b)
{
    return base64_decode(buf, b->data);
}

} // namespace c4

#endif /* _C4_BASE64_HPP_ */
