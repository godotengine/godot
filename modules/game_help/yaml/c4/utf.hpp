#ifndef C4_UTF_HPP_
#define C4_UTF_HPP_

#include "language.hpp"
#include "substr_fwd.hpp"
#include <stddef.h>
#include <stdint.h>

namespace c4 {

substr decode_code_point(substr out, csubstr code_point);
size_t decode_code_point(uint8_t *C4_RESTRICT buf, size_t buflen, const uint32_t code);

} // namespace c4

#endif // C4_UTF_HPP_
