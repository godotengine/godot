#ifndef _C4_SUBSTR_FWD_HPP_
#define _C4_SUBSTR_FWD_HPP_

#include "export.hpp"

namespace c4 {

#ifndef DOXYGEN
template<class C> struct basic_substring;
using csubstr = C4CORE_EXPORT basic_substring<const char>;
using substr = C4CORE_EXPORT basic_substring<char>;
#endif // !DOXYGEN

} // namespace c4

#endif /* _C4_SUBSTR_FWD_HPP_ */
