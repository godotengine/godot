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

struct hb_varc_scratch_t
{
  hb_vector_t<unsigned> axisIndices;
  hb_vector_t<float> axisValues;
  hb_glyf_scratch_t glyf_scratch;
};

struct hb_varc_context_t
{
  hb_font_t *font;
  hb_draw_session_t *draw_session;
  hb_extents_t *extents;
  mutable hb_decycler_t decycler;
  mutable signed edges_left;
  mutable signed depth_left;
  hb_varc_scratch_t &scratch;
};

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
  get_path_at (const hb_varc_context_t &c,
	       hb_codepoint_t parent_gid,
	       hb_array_t<const int> coords,
	       hb_transform_t transform,
	       hb_ubytes_t record,
	       VarRegionList::cache_t *cache = nullptr) const;
};

struct VarCompositeGlyph
{
  static void
  get_path_at (const hb_varc_context_t &c,
	       hb_codepoint_t gid,
	       hb_array_t<const int> coords,
	       hb_transform_t transform,
	       hb_ubytes_t record,
	       VarRegionList::cache_t *cache)
  {
    while (record)
    {
      const VarComponent &comp = * (const VarComponent *) (record.arrayZ);
      record = comp.get_path_at (c,
				 gid,
				 coords, transform,
				 record,
				 cache);
    }
  }
};

HB_MARK_AS_FLAG_T (VarComponent::flags_t);

struct VARC
{
  friend struct VarComponent;

  static constexpr hb_tag_t tableTag = HB_TAG ('V', 'A', 'R', 'C');

  HB_INTERNAL bool
  get_path_at (const hb_varc_context_t &c,
	       hb_codepoint_t gid,
	       hb_array_t<const int> coords,
	       hb_transform_t transform = HB_TRANSFORM_IDENTITY,
	       hb_codepoint_t parent_gid = HB_CODEPOINT_INVALID,
	       VarRegionList::cache_t *parent_cache = nullptr) const;

  bool
  get_path (hb_font_t *font,
	    hb_codepoint_t gid,
	    hb_draw_session_t &draw_session,
	    hb_varc_scratch_t &scratch) const
  {
    hb_varc_context_t c {font,
			 &draw_session,
			 nullptr,
			 hb_decycler_t {},
			 HB_MAX_GRAPH_EDGE_COUNT,
			 HB_MAX_NESTING_LEVEL,
			 scratch};

    return get_path_at (c, gid,
			hb_array (font->coords, font->num_coords));
  }

  bool
  get_extents (hb_font_t *font,
	       hb_codepoint_t gid,
	       hb_extents_t *extents,
	       hb_varc_scratch_t &scratch) const
  {
    hb_varc_context_t c {font,
			 nullptr,
			 extents,
			 hb_decycler_t {},
			 HB_MAX_GRAPH_EDGE_COUNT,
			 HB_MAX_NESTING_LEVEL,
			 scratch};

    return get_path_at (c, gid,
			hb_array (font->coords, font->num_coords));
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
	scratch->~hb_varc_scratch_t ();
	hb_free (scratch);
      }

      table.destroy ();
    }

    bool
    get_path (hb_font_t *font, hb_codepoint_t gid, hb_draw_session_t &draw_session) const
    {
      if (!table->has_data ()) return false;

      auto *scratch = acquire_scratch ();
      if (unlikely (!scratch)) return true;
      bool ret = table->get_path (font, gid, draw_session, *scratch);
      release_scratch (scratch);
      return ret;
    }

    bool
    get_extents (hb_font_t *font,
		 hb_codepoint_t gid,
		 hb_glyph_extents_t *extents) const
    {
      if (!table->has_data ()) return false;

      hb_extents_t f_extents;

      auto *scratch = acquire_scratch ();
      if (unlikely (!scratch)) return true;
      bool ret = table->get_extents (font, gid, &f_extents, *scratch);
      release_scratch (scratch);

      if (ret)
	*extents = f_extents.to_glyph_extents (font->x_scale < 0, font->y_scale < 0);

      return ret;
    }

    private:

    hb_varc_scratch_t *acquire_scratch () const
    {
      hb_varc_scratch_t *scratch = cached_scratch.get_acquire ();

      if (!scratch || unlikely (!cached_scratch.cmpexch (scratch, nullptr)))
      {
	scratch = (hb_varc_scratch_t *) hb_calloc (1, sizeof (hb_varc_scratch_t));
	if (unlikely (!scratch))
	  return nullptr;
      }

      return scratch;
    }
    void release_scratch (hb_varc_scratch_t *scratch) const
    {
      if (!cached_scratch.cmpexch (nullptr, scratch))
      {
	scratch->~hb_varc_scratch_t ();
	hb_free (scratch);
      }
    }

    private:
    hb_blob_ptr_t<VARC> table;
    hb_atomic_t<hb_varc_scratch_t *> cached_scratch;
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
