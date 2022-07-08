#ifndef OT_GLYF_GLYF_HELPERS_HH
#define OT_GLYF_GLYF_HELPERS_HH


#include "../../hb-open-type.hh"
#include "../../hb-subset-plan.hh"

#include "loca.hh"


namespace OT {
namespace glyf_impl {


template<typename IteratorIn, typename IteratorOut,
	 hb_requires (hb_is_source_of (IteratorIn, unsigned int)),
	 hb_requires (hb_is_sink_of (IteratorOut, unsigned))>
static void
_write_loca (IteratorIn it, bool short_offsets, IteratorOut dest)
{
  unsigned right_shift = short_offsets ? 1 : 0;
  unsigned int offset = 0;
  dest << 0;
  + it
  | hb_map ([=, &offset] (unsigned int padded_size)
	    {
	      offset += padded_size;
	      DEBUG_MSG (SUBSET, nullptr, "loca entry offset %d", offset);
	      return offset >> right_shift;
	    })
  | hb_sink (dest)
  ;
}

static bool
_add_head_and_set_loca_version (hb_subset_plan_t *plan, bool use_short_loca)
{
  hb_blob_t *head_blob = hb_sanitize_context_t ().reference_table<head> (plan->source);
  hb_blob_t *head_prime_blob = hb_blob_copy_writable_or_fail (head_blob);
  hb_blob_destroy (head_blob);

  if (unlikely (!head_prime_blob))
    return false;

  head *head_prime = (head *) hb_blob_get_data_writable (head_prime_blob, nullptr);
  head_prime->indexToLocFormat = use_short_loca ? 0 : 1;
  bool success = plan->add_table (HB_OT_TAG_head, head_prime_blob);

  hb_blob_destroy (head_prime_blob);
  return success;
}

template<typename Iterator,
	 hb_requires (hb_is_source_of (Iterator, unsigned int))>
static bool
_add_loca_and_head (hb_subset_plan_t * plan, Iterator padded_offsets, bool use_short_loca)
{
  unsigned num_offsets = padded_offsets.len () + 1;
  unsigned entry_size = use_short_loca ? 2 : 4;
  char *loca_prime_data = (char *) hb_calloc (entry_size, num_offsets);

  if (unlikely (!loca_prime_data)) return false;

  DEBUG_MSG (SUBSET, nullptr, "loca entry_size %d num_offsets %d size %d",
	     entry_size, num_offsets, entry_size * num_offsets);

  if (use_short_loca)
    _write_loca (padded_offsets, true, hb_array ((HBUINT16 *) loca_prime_data, num_offsets));
  else
    _write_loca (padded_offsets, false, hb_array ((HBUINT32 *) loca_prime_data, num_offsets));

  hb_blob_t *loca_blob = hb_blob_create (loca_prime_data,
					 entry_size * num_offsets,
					 HB_MEMORY_MODE_WRITABLE,
					 loca_prime_data,
					 hb_free);

  bool result = plan->add_table (HB_OT_TAG_loca, loca_blob)
	     && _add_head_and_set_loca_version (plan, use_short_loca);

  hb_blob_destroy (loca_blob);
  return result;
}


} /* namespace glyf_impl */
} /* namespace OT */


#endif /* OT_GLYF_GLYF_HELPERS_HH */
