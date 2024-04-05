#include "memory_util.hpp"
#include "error.hpp"

namespace c4 {


/** Fills 'dest' with the first 'pattern_size' bytes at 'pattern', 'num_times'. */
void mem_repeat(void* dest, void const* pattern, size_t pattern_size, size_t num_times)
{
    if(C4_UNLIKELY(num_times == 0))
        return;
    C4_ASSERT( ! mem_overlaps(dest, pattern, num_times*pattern_size, pattern_size));
    char *begin = static_cast<char*>(dest);
    char *end   = begin + num_times * pattern_size;
    // copy the pattern once
    ::memcpy(begin, pattern, pattern_size);
    // now copy from dest to itself, doubling up every time
    size_t n = pattern_size;
    while(begin + 2*n < end)
    {
        ::memcpy(begin + n, begin, n);
        n <<= 1; // double n
    }
    // copy the missing part
    if(begin + n < end)
    {
        ::memcpy(begin + n, begin, static_cast<size_t>(end - (begin + n)));
    }
}


} // namespace c4
