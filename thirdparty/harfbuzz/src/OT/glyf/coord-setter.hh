#ifndef OT_GLYF_COORD_SETTER_HH
#define OT_GLYF_COORD_SETTER_HH


#include "../../hb.hh"


namespace OT {
namespace glyf_impl {


struct coord_setter_t
{
  coord_setter_t (hb_array_t<int> coords) :
    coords (coords) {}

  int& operator [] (unsigned idx)
  {
    if (unlikely (idx >= HB_GLYF_VAR_COMPOSITE_MAX_AXES))
      return Crap(int);
    if (coords.length < idx + 1)
      coords.resize (idx + 1);
    return coords[idx];
  }

  hb_array_t<int> get_coords ()
  { return coords.as_array (); }

  hb_vector_t<int> coords;
};


} /* namespace glyf_impl */
} /* namespace OT */

#endif /* OT_GLYF_COORD_SETTER_HH */
