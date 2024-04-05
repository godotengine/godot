#ifndef _C4_YML_PREPROCESS_HPP_
#define _C4_YML_PREPROCESS_HPP_

/** @file preprocess.hpp Functions for preprocessing YAML prior to parsing. */

/** @defgroup Preprocessors Preprocessor functions
 *
 * These are the existing preprocessors:
 *
 * @code{.cpp}
 * size_t preprocess_json(csubstr json, substr buf)
 * size_t preprocess_rxmap(csubstr json, substr buf)
 * @endcode
 */

#ifndef _C4_YML_COMMON_HPP_
#include "./common.hpp"
#endif
#include "c4/substr.hpp"


namespace c4 {
namespace yml {

namespace detail {
using Preprocessor = size_t(csubstr, substr);
template<Preprocessor PP, class CharContainer>
substr preprocess_into_container(csubstr input, CharContainer *out)
{
    // try to write once. the preprocessor will stop writing at the end of
    // the container, but will process all the input to determine the
    // required container size.
    size_t sz = PP(input, to_substr(*out));
    // if the container size is not enough, resize, and run again in the
    // resized container
    if(sz > out->size())
    {
        out->resize(sz);
        sz = PP(input, to_substr(*out));
    }
    return to_substr(*out).first(sz);
}
} // namespace detail


//-----------------------------------------------------------------------------

/** @name preprocess_rxmap
 * Convert flow-type relaxed maps (with implicit bools) into strict YAML
 * flow map.
 *
 * @code{.yaml}
 * {a, b, c, d: [e, f], g: {a, b}}
 * # is converted into this:
 * {a: 1, b: 1, c: 1, d: [e, f], g: {a, b}}
 * @endcode

 * @note this is NOT recursive - conversion happens only in the top-level map
 * @param rxmap A relaxed map
 * @param buf output buffer
 * @param out output container
 */

//@{

/** Write into a given output buffer. This function is safe to call with
 * empty or small buffers; it won't write beyond the end of the buffer.
 *
 * @return the number of characters required for output
 */
RYML_EXPORT size_t preprocess_rxmap(csubstr rxmap, substr buf);


/** Write into an existing container. It is resized to contained the output.
 * @return a substr of the container
 * @overload preprocess_rxmap */
template<class CharContainer>
substr preprocess_rxmap(csubstr rxmap, CharContainer *out)
{
    return detail::preprocess_into_container<preprocess_rxmap>(rxmap, out);
}


/** Create a container with the result.
 * @overload preprocess_rxmap */
template<class CharContainer>
CharContainer preprocess_rxmap(csubstr rxmap)
{
    CharContainer out;
    preprocess_rxmap(rxmap, &out);
    return out;
}

//@}

} // namespace yml
} // namespace c4

#endif /* _C4_YML_PREPROCESS_HPP_ */
