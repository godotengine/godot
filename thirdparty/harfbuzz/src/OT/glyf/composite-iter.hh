#ifndef OT_GLYF_COMPOSITE_ITER_HH
#define OT_GLYF_COMPOSITE_ITER_HH


#include "../../hb.hh"


namespace OT {
namespace glyf_impl {


template <typename CompositeGlyphRecord>
struct composite_iter_tmpl : hb_iter_with_fallback_t<composite_iter_tmpl<CompositeGlyphRecord>,
						     const CompositeGlyphRecord &>
{
  typedef const CompositeGlyphRecord *__item_t__;
  composite_iter_tmpl (hb_bytes_t glyph_, __item_t__ current_) :
      glyph (glyph_), current (nullptr), current_size (0)
  {
    set_current (current_);
  }

  composite_iter_tmpl () : glyph (hb_bytes_t ()), current (nullptr), current_size (0) {}

  const CompositeGlyphRecord & __item__ () const { return *current; }
  bool __more__ () const { return current; }
  void __next__ ()
  {
    if (!current->has_more ()) { current = nullptr; return; }

    set_current (&StructAtOffset<CompositeGlyphRecord> (current, current_size));
  }
  composite_iter_tmpl __end__ () const { return composite_iter_tmpl (); }
  bool operator != (const composite_iter_tmpl& o) const
  { return current != o.current; }


  void set_current (__item_t__ current_)
  {
    if (!glyph.check_range (current_, CompositeGlyphRecord::min_size))
    {
      current = nullptr;
      current_size = 0;
      return;
    }
    unsigned size = current_->get_size ();
    if (!glyph.check_range (current_, size))
    {
      current = nullptr;
      current_size = 0;
      return;
    }

    current = current_;
    current_size = size;
  }

  private:
  hb_bytes_t glyph;
  __item_t__ current;
  unsigned current_size;
};


} /* namespace glyf_impl */
} /* namespace OT */

#endif /* OT_GLYF_COMPOSITE_ITER_HH */
