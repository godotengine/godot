#include "language.hpp"

namespace c4 {
namespace detail {

#ifndef __GNUC__
void use_char_pointer(char const volatile* v)
{
    C4_UNUSED(v);
}
#else
void foo() {} // to avoid empty file warning from the linker
#endif

} // namespace detail
} // namespace c4
