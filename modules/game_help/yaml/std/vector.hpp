#ifndef _C4_YML_STD_VECTOR_HPP_
#define _C4_YML_STD_VECTOR_HPP_

#include "../node.hpp"
#include "../c4/std/vector.hpp"
#include <vector>

namespace c4 {
namespace yml {

// vector is a sequence-like type, and it requires child nodes
// in the data tree hierarchy (a SEQ node in ryml parlance).
// So it should be serialized via write()/read().


template<class V, class Alloc>
void write(c4::yml::NodeRef *n, std::vector<V, Alloc> const& vec)
{
    *n |= c4::yml::SEQ;
    for(auto const& v : vec)
        n->append_child() << v;
}

template<class V, class Alloc>
bool read(c4::yml::ConstNodeRef const& n, std::vector<V, Alloc> *vec)
{
    vec->resize(n.num_children());
    size_t pos = 0;
    for(auto const ch : n)
        ch >> (*vec)[pos++];
    return true;
}

/** specialization: std::vector<bool> uses std::vector<bool>::reference as
 * the return value of its operator[]. */
template<class Alloc>
bool read(c4::yml::ConstNodeRef const& n, std::vector<bool, Alloc> *vec)
{
    vec->resize(n.num_children());
    size_t pos = 0;
    bool tmp;
    for(auto const ch : n)
    {
        ch >> tmp;
        (*vec)[pos++] = tmp;
    }
    return true;
}

} // namespace yml
} // namespace c4

#endif // _C4_YML_STD_VECTOR_HPP_
