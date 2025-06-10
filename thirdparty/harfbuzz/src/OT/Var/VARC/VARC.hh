#ifndef OT_VAR_VARC_VARC_HH
#define OT_VAR_VARC_VARC_HH

#include "../../../hb-decycler.hh"
#include "../../../hb-geometry.hh"
#include "../../../hb-ot-layout-common.hh"
#include "../../../hb-ot-glyf-table.hh"
#include "../../../hb-ot-cff2-table.hh"
#include "../../../hb-ot-cff1-table.hh"

#include "coord-setter.hh"

namespace OT {

//namespace Var {

/*
 * VARC -- Variable Composites
 * https://github.com/harfbuzz/boring-expansion-spec/blob/main/VARC.md
 */

#ifndef HB_NO_VAR_COMPOSITES

struct VarComponent
{
  enum class flags_t : uint32_t
  {
    RESET_UNSPECIFIED_AXES	= 1u << 0,
    HAVE_AXES			= 1u << 1,
    AXIS_VALUES_HAVE_VARIATION	= 1u << 2,
    TRANSFORM_HAS_VARIATION	= 1u << 3,
    HAVE_TRANSLATE_X		= 1u << 4,
    HAVE_TRANSLATE_Y		= 1u << 5,
    HAVE_ROTATION		= 1u << 6,
    HAVE_CONDITION		= 1u << 7,
    HAVE_SCALE_X		= 1u << 8,
    HAVE_SCALE_Y		= 1u << 9,
    HAVE_TCENTER_X		= 1u << 10,
    HAVE_TCENTER_Y		= 1u << 11,
    GID_IS_24BIT		= 1u << 12,
    HAVE_SKEW_X			= 1u << 13,
    HAVE_SKEW_Y			= 1u << 14,
    RESERVED_MASK		= ~((1u << 15) - 1),
  };

  HB_INTERNAL hb_ubytes_t
  get_path_at (hb_font_t *font,
	       hb_codepoint_t parent_gid,
	       hb_draw_session_t &draw_session,
	       hb_array_t<const int> coords,
	       hb_transform_t transform,
	       hb_ubytes_t record,
	       hb_decycler_t *decycler,
	       signed *edges_left,
	       signed depth_left,
	       hb_glyf_scratch_t &scratch,
	       VarRegionList::cache_t *cache = nullptr) const;
};

struct VarCompositeGlyph
{
  static void
  get_path_at (hb_font_t *font,
	       hb_codepoint_t glyph,
	       hb_draw_session_t &draw_session,
	       hb_array_t<const int> coords,
	       hb_transform_t transform,
	       hb_ubytes_t record,
	       hb_decycler_t *decycler,
	       signed *edges_left,
	       signed depth_left,
	       hb_glyf_scratch_t &scratch,
	       VarRegionList::cache_t *cache = nullptr)
  {
    while (record)
    {
      const VarComponent &comp = * (const VarComponent *) (record.arrayZ);
      record = comp.get_path_at (font, glyph,
				 draw_session, coords, transform,
				 record,
				 decycler, edges_left, depth_left, scratch, cache);
    }
  }
};

HB_MARK_AS_FLAG_T (VarComponent::flags_t);

struct VARC
{
  friend struct VarComponent;

  static constexpr hb_tag_t tableTag = HB_TAG ('V', 'A', 'R', 'C');

  HB_INTERNAL bool
  get_path_at (hb_font_t *font,
	       hb_codepoint_t glyph,
	       hb_draw_session_t &draw_session,
	       hb_array_t<const int> coords,
	       hb_transform_t transform,
	       hb_codepoint_t parent_glyph,
	       hb_decycler_t *decycler,
	       signed *edges_left,
	       signed depth_left,
	       hb_glyf_scratch_t &scratch) const;

  bool
  get_path (hb_font_t *font,
	    hb_codepoint_t gid,
	    hb_draw_session_t &draw_session,
	    hb_glyf_scratch_t &scratch) const
  {
    hb_decycler_t decycler;
    signed edges = HB_MAX_GRAPH_EDGE_COUNT;

    return get_path_at (font,
			gid,
			draw_session,
			hb_array (font->coords, font->num_coords),
			HB_TRANSFORM_IDENTITY,
			HB_CODEPOINT_INVALID,
			&decycler,
			&edges,
			HB_MAX_NESTING_LEVEL,
			scratch);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (version.sanitize (c) &&
		  hb_barrier () &&
		  version.major == 1 &&
		  coverage.sanitize (c, this) &&
		  varStore.sanitize (c, this) &&
		  conditionList.sanitize (c, this) &&
		  axisIndicesList.sanitize (c, this) &&
		  glyphRecords.sanitize (c, this));
  }

  struct accelerator_t
  {
    friend struct VarComponent;

    accelerator_t (hb_face_t *face)
    {
      table = hb_sanitize_context_t ().reference_table<VARC> (face);
    }
    ~accelerator_t ()
    {
      auto *scratch = cached_scratch.get_relaxed ();
      if (scratch)
      {
	scratch->~hb_glyf_scratch_t ();
	hb_free (scratch);
      }

      table.destroy ();
    }

    bool
    get_path (hb_font_t *font, hb_codepoint_t gid, hb_draw_session_t &draw_session) const
    {
      if (!table->has_data ()) return false;

      hb_glyf_scratch_t *scratch;

      // Borrow the cached strach buffer.
      {
	scratch = cached_scratch.get_acquire ();
	if (!scratch || unlikely (!cached_scratch.cmpexch (scratch, nullptr)))
	{
	  scratch = (hb_glyf_scratch_t *) hb_calloc (1, sizeof (hb_glyf_scratch_t));
	  if (unlikely (!scratch))
	    return true;
	}
      }

      bool ret = table->get_path (font, gid, draw_session, *scratch);

      // Put it back.
      if (!cached_scratch.cmpexch (nullptr, scratch))
      {
        scratch->~hb_glyf_scratch_t ();
	hb_free (scratch);
      }

      return ret;
    }

    private:
    hb_blob_ptr_t<VARC> table;
    hb_atomic_ptr_t<hb_glyf_scratch_t> cached_scratch;
  };

  bool has_data () const { return version.major != 0; }

  protected:
  FixedVersion<> version; /* Version identifier */
  Offset32To<Coverage> coverage;
  Offset32To<MultiItemVariationStore> varStore;
  Offset32To<ConditionList> conditionList;
  Offset32To<TupleList> axisIndicesList;
  Offset32To<CFF2Index/*Of<VarCompositeGlyph>*/> glyphRecords;
  public:
  DEFINE_SIZE_STATIC (24);
};

struct VARC_accelerator_t : VARC::accelerator_t {
  VARC_accelerator_t (hb_face_t *face) : VARC::accelerator_t (face) {}
};

#endif

//}

}

#endif  /* OT_VAR_VARC_VARC_HH */
