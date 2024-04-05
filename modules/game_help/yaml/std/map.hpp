#ifndef _C4_YML_STD_MAP_HPP_
#define _C4_YML_STD_MAP_HPP_

/** @file map.hpp write/read std::map to/from a YAML tree. */

#include "../node.hpp"
#include <map>

namespace c4 {
namespace yml {

// std::map requires child nodes in the data
// tree hierarchy (a MAP node in ryml parlance).
// So it should be serialized via write()/read().

template<class K, class V, class Less, class Alloc>
void write(c4::yml::NodeRef *n, std::map<K, V, Less, Alloc> const& m)
{
    *n |= c4::yml::MAP;
    for(auto const& C4_RESTRICT p : m)
    {
        auto ch = n->append_child();
        ch << c4::yml::key(p.first);
        ch << p.second;
    }
}

template<class K, class V, class Less, class Alloc>
bool read(c4::yml::ConstNodeRef const& n, std::map<K, V, Less, Alloc> * m)
{
    K k{};
    V v{};
    for(auto const& C4_RESTRICT ch : n)
    {
        ch >> c4::yml::key(k);
        ch >> v;
        m->emplace(std::make_pair(std::move(k), std::move(v)));
    }
    return true;
}

} // namespace yml
} // namespace c4

#endif // _C4_YML_STD_MAP_HPP_
