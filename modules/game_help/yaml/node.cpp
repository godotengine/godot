#include "node.hpp"

namespace c4 {
namespace yml {




//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

size_t NodeRef::set_key_serialized(c4::fmt::const_base64_wrapper w)
{
    _apply_seed();
    csubstr encoded = this->to_arena(w);
    this->set_key(encoded);
    return encoded.len;
}

size_t NodeRef::set_val_serialized(c4::fmt::const_base64_wrapper w)
{
    _apply_seed();
    csubstr encoded = this->to_arena(w);
    this->set_val(encoded);
    return encoded.len;
}

} // namespace yml
} // namespace c4
