#include "VARC.hh"

#ifndef HB_NO_VAR_COMPOSITES

#include "../../../hb-draw.hh"
#include "../../../hb-ot-layout-common.hh"
#include "../../../hb-ot-layout-gdef-table.hh"

namespace OT {

//namespace Var {


struct hb_transforming_pen_context_t
{
  hb_transform_t transform;
  hb_draw_funcs_t *dfuncs;
  void *data;
  hb_draw_state_t *st;
};

static void
hb_transforming_pen_move_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
			     void *data,
			     hb_draw_state_t *st,
			     float to_x, float to_y,
			     void *user_data HB_UNUSED)
{
  hb_transforming_pen_context_t *c = (hb_transforming_pen_context_t *) data;

  c->transform.transform_point (to_x, to_y);

  c->dfuncs->move_to (c->data, *c->st, to_x, to_y);
}

static void
hb_transforming_pen_line_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
			     void *data,
			     hb_draw_state_t *st,
			     float to_x, float to_y,
			     void *user_data HB_UNUSED)
{
  hb_transforming_pen_context_t *c = (hb_transforming_pen_context_t *) data;

  c->transform.transform_point (to_x, to_y);

  c->dfuncs->line_to (c->data, *c->st, to_x, to_y);
}

static void
hb_transforming_pen_quadratic_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
				  void *data,
				  hb_draw_state_t *st,
				  float control_x, float control_y,
				  float to_x, float to_y,
				  void *user_data HB_UNUSED)
{
  hb_transforming_pen_context_t *c = (hb_transforming_pen_context_t *) data;

  c->transform.transform_point (control_x, control_y);
  c->transform.transform_point (to_x, to_y);

  c->dfuncs->quadratic_to (c->data, *c->st, control_x, control_y, to_x, to_y);
}

static void
hb_transforming_pen_cubic_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
			      void *data,
			      hb_draw_state_t *st,
			      float control1_x, float control1_y,
			      float control2_x, float control2_y,
			      float to_x, float to_y,
			      void *user_data HB_UNUSED)
{
  hb_transforming_pen_context_t *c = (hb_transforming_pen_context_t *) data;

  c->transform.transform_point (control1_x, control1_y);
  c->transform.transform_point (control2_x, control2_y);
  c->transform.transform_point (to_x, to_y);

  c->dfuncs->cubic_to (c->data, *c->st, control1_x, control1_y, control2_x, control2_y, to_x, to_y);
}

static void
hb_transforming_pen_close_path (hb_draw_funcs_t *dfuncs HB_UNUSED,
				void *data,
				hb_draw_state_t *st,
				void *user_data HB_UNUSED)
{
  hb_transforming_pen_context_t *c = (hb_transforming_pen_context_t *) data;

  c->dfuncs->close_path (c->data, *c->st);
}

static inline void free_static_transforming_pen_funcs ();

static struct hb_transforming_pen_funcs_lazy_loader_t : hb_draw_funcs_lazy_loader_t<hb_transforming_pen_funcs_lazy_loader_t>
{
  static hb_draw_funcs_t *create ()
  {
    hb_draw_funcs_t *funcs = hb_draw_funcs_create ();

    hb_draw_funcs_set_move_to_func (funcs, hb_transforming_pen_move_to, nullptr, nullptr);
    hb_draw_funcs_set_line_to_func (funcs, hb_transforming_pen_line_to, nullptr, nullptr);
    hb_draw_funcs_set_quadratic_to_func (funcs, hb_transforming_pen_quadratic_to, nullptr, nullptr);
    hb_draw_funcs_set_cubic_to_func (funcs, hb_transforming_pen_cubic_to, nullptr, nullptr);
    hb_draw_funcs_set_close_path_func (funcs, hb_transforming_pen_close_path, nullptr, nullptr);

    hb_draw_funcs_make_immutable (funcs);

    hb_atexit (free_static_transforming_pen_funcs);

    return funcs;
  }
} static_transforming_pen_funcs;

static inline
void free_static_transforming_pen_funcs ()
{
  static_transforming_pen_funcs.free_instance ();
}

static hb_draw_funcs_t *
hb_transforming_pen_get_funcs ()
{
  return static_transforming_pen_funcs.get_unconst ();
}


hb_ubytes_t
VarComponent::get_path_at (hb_font_t *font,
			   hb_codepoint_t parent_gid,
			   hb_draw_session_t &draw_session,
			   hb_array_t<const int> coords,
			   hb_transform_t total_transform,
			   hb_ubytes_t total_record,
			   hb_decycler_t *decycler,
			   signed *edges_left,
			   signed depth_left,
			   hb_glyf_scratch_t &scratch,
			   VarRegionList::cache_t *cache) const
{
  const unsigned char *end = total_record.arrayZ + total_record.length;
  const unsigned char *record = total_record.arrayZ;

  auto &VARC = *font->face->table.VARC->table;
  auto &varStore = &VARC+VARC.varStore;

#define READ_UINT32VAR(name) \
  HB_STMT_START { \
    if (unlikely (unsigned (end - record) < HBUINT32VAR::min_size)) return hb_ubytes_t (); \
    hb_barrier (); \
    auto &varint = * (const HBUINT32VAR *) record; \
    unsigned size = varint.get_size (); \
    if (unlikely (unsigned (end - record) < size)) return hb_ubytes_t (); \
    name = (uint32_t) varint; \
    record += size; \
  } HB_STMT_END

  uint32_t flags;
  READ_UINT32VAR (flags);

  // gid

  hb_codepoint_t gid = 0;
  if (flags & (unsigned) flags_t::GID_IS_24BIT)
  {
    if (unlikely (unsigned (end - record) < HBGlyphID24::static_size))
      return hb_ubytes_t ();
    hb_barrier ();
    gid = * (const HBGlyphID24 *) record;
    record += HBGlyphID24::static_size;
  }
  else
  {
    if (unlikely (unsigned (end - record) < HBGlyphID16::static_size))
      return hb_ubytes_t ();
    hb_barrier ();
    gid = * (const HBGlyphID16 *) record;
    record += HBGlyphID16::static_size;
  }

  // Condition
  bool show = true;
  if (flags & (unsigned) flags_t::HAVE_CONDITION)
  {
    unsigned conditionIndex;
    READ_UINT32VAR (conditionIndex);
    const auto &condition = (&VARC+VARC.conditionList)[conditionIndex];
    auto instancer = MultiItemVarStoreInstancer(&varStore, nullptr, coords, cache);
    show = condition.evaluate (coords.arrayZ, coords.length, &instancer);
  }

  // Axis values

  auto &axisIndices = scratch.axisIndices;
  axisIndices.clear ();
  auto &axisValues = scratch.axisValues;
  axisValues.clear ();
  if (flags & (unsigned) flags_t::HAVE_AXES)
  {
    unsigned axisIndicesIndex;
    READ_UINT32VAR (axisIndicesIndex);
    axisIndices.extend ((&VARC+VARC.axisIndicesList)[axisIndicesIndex]);
    axisValues.resize (axisIndices.length);
    const HBUINT8 *p = (const HBUINT8 *) record;
    TupleValues::decompile (p, axisValues, (const HBUINT8 *) end);
    record = (const unsigned char *) p;
  }

  // Apply variations if any
  if (flags & (unsigned) flags_t::AXIS_VALUES_HAVE_VARIATION)
  {
    uint32_t axisValuesVarIdx;
    READ_UINT32VAR (axisValuesVarIdx);
    if (show && coords && !axisValues.in_error ())
      varStore.get_delta (axisValuesVarIdx, coords, axisValues.as_array (), cache);
  }

  auto component_coords = coords;
  /* Copying coords is expensive; so we have put an arbitrary
   * limit on the max number of coords for now. */
  if ((flags & (unsigned) flags_t::RESET_UNSPECIFIED_AXES) ||
      coords.length > HB_VAR_COMPOSITE_MAX_AXES)
    component_coords = hb_array<int> (font->coords, font->num_coords);

  // Transform

  uint32_t transformVarIdx = VarIdx::NO_VARIATION;
  if (flags & (unsigned) flags_t::TRANSFORM_HAS_VARIATION)
    READ_UINT32VAR (transformVarIdx);

#define PROCESS_TRANSFORM_COMPONENTS \
	HB_STMT_START { \
	PROCESS_TRANSFORM_COMPONENT (FWORD, HAVE_TRANSLATE_X, translateX); \
	PROCESS_TRANSFORM_COMPONENT (FWORD, HAVE_TRANSLATE_Y, translateY); \
	PROCESS_TRANSFORM_COMPONENT (F4DOT12, HAVE_ROTATION, rotation); \
	PROCESS_TRANSFORM_COMPONENT (F6DOT10, HAVE_SCALE_X, scaleX); \
	PROCESS_TRANSFORM_COMPONENT (F6DOT10, HAVE_SCALE_Y, scaleY); \
	PROCESS_TRANSFORM_COMPONENT (F4DOT12, HAVE_SKEW_X, skewX); \
	PROCESS_TRANSFORM_COMPONENT (F4DOT12, HAVE_SKEW_Y, skewY); \
	PROCESS_TRANSFORM_COMPONENT (FWORD, HAVE_TCENTER_X, tCenterX); \
	PROCESS_TRANSFORM_COMPONENT (FWORD, HAVE_TCENTER_Y, tCenterY); \
	} HB_STMT_END

  hb_transform_decomposed_t transform;

  // Read transform components
#define PROCESS_TRANSFORM_COMPONENT(type, flag, name) \
	if (flags & (unsigned) flags_t::flag) \
	{ \
	  static_assert (type::static_size == HBINT16::static_size, ""); \
	  if (unlikely (unsigned (end - record) < HBINT16::static_size)) \
	    return hb_ubytes_t (); \
	  hb_barrier (); \
	  transform.name = * (const HBINT16 *) record; \
	  record += HBINT16::static_size; \
	}
  PROCESS_TRANSFORM_COMPONENTS;
#undef PROCESS_TRANSFORM_COMPONENT

  // Read reserved records
  unsigned i = flags & (unsigned) flags_t::RESERVED_MASK;
  while (i)
  {
    HB_UNUSED uint32_t discard;
    READ_UINT32VAR (discard);
    i &= i - 1;
  }

  /* Parsing is over now. */

  if (show)
  {
    // Only use coord_setter if there's actually any axis overrides.
    coord_setter_t coord_setter (axisIndices ? component_coords : hb_array<int> ());
    // Go backwards, to reduce coord_setter vector reallocations.
    for (unsigned i = axisIndices.length; i; i--)
      coord_setter[axisIndices[i - 1]] = axisValues[i - 1];
    if (axisIndices)
      component_coords = coord_setter.get_coords ();

    // Apply transform variations if any
    if (transformVarIdx != VarIdx::NO_VARIATION && coords)
    {
      float transformValues[9];
      unsigned numTransformValues = 0;
#define PROCESS_TRANSFORM_COMPONENT(type, flag, name) \
	  if (flags & (unsigned) flags_t::flag) \
	    transformValues[numTransformValues++] = transform.name;
      PROCESS_TRANSFORM_COMPONENTS;
#undef PROCESS_TRANSFORM_COMPONENT
      varStore.get_delta (transformVarIdx, coords, hb_array (transformValues, numTransformValues), cache);
      numTransformValues = 0;
#define PROCESS_TRANSFORM_COMPONENT(type, flag, name) \
	  if (flags & (unsigned) flags_t::flag) \
	    transform.name = transformValues[numTransformValues++];
      PROCESS_TRANSFORM_COMPONENTS;
#undef PROCESS_TRANSFORM_COMPONENT
    }

    // Divide them by their divisors
#define PROCESS_TRANSFORM_COMPONENT(type, flag, name) \
	  if (flags & (unsigned) flags_t::flag) \
	  { \
	    HBINT16 int_v; \
	    int_v = roundf (transform.name); \
	    type typed_v = * (const type *) &int_v; \
	    float float_v = (float) typed_v; \
	    transform.name = float_v; \
	  }
    PROCESS_TRANSFORM_COMPONENTS;
#undef PROCESS_TRANSFORM_COMPONENT

    if (!(flags & (unsigned) flags_t::HAVE_SCALE_Y))
      transform.scaleY = transform.scaleX;

    total_transform.transform (transform.to_transform ());
    total_transform.scale (font->x_mult ? 1.f / font->x_multf : 0.f,
			   font->y_mult ? 1.f / font->y_multf : 0.f);

    VARC.get_path_at (font, gid,
		      draw_session, component_coords, total_transform,
		      parent_gid,
		      decycler, edges_left, depth_left - 1,
		      scratch);
  }

#undef PROCESS_TRANSFORM_COMPONENTS
#undef READ_UINT32VAR

  return hb_ubytes_t (record, end - record);
}

bool
VARC::get_path_at (hb_font_t *font,
		   hb_codepoint_t glyph,
		   hb_draw_session_t &draw_session,
		   hb_array_t<const int> coords,
		   hb_transform_t transform,
		   hb_codepoint_t parent_glyph,
		   hb_decycler_t *decycler,
		   signed *edges_left,
		   signed depth_left,
		   hb_glyf_scratch_t &scratch) const
{
  // Don't recurse on the same glyph.
  unsigned idx = glyph == parent_glyph ?
		 NOT_COVERED :
		 (this+coverage).get_coverage (glyph);
  if (idx == NOT_COVERED)
  {
    // Build a transforming pen to apply the transform.
    hb_draw_funcs_t *transformer_funcs = hb_transforming_pen_get_funcs ();
    hb_transforming_pen_context_t context {transform,
					   draw_session.funcs,
					   draw_session.draw_data,
					   &draw_session.st};
    hb_draw_session_t transformer_session {transformer_funcs, &context};
    hb_draw_session_t &shape_draw_session = transform.is_identity () ? draw_session : transformer_session;

    if (!font->face->table.glyf->get_path_at (font, glyph, shape_draw_session, coords, scratch))
#ifndef HB_NO_CFF
    if (!font->face->table.cff2->get_path_at (font, glyph, shape_draw_session, coords))
    if (!font->face->table.cff1->get_path (font, glyph, shape_draw_session)) // Doesn't have variations
#endif
      return false;
    return true;
  }

  if (depth_left <= 0)
    return true;

  if (*edges_left <= 0)
    return true;
  (*edges_left)--;

  hb_decycler_node_t node (*decycler);
  if (unlikely (!node.visit (glyph)))
    return true;

  hb_ubytes_t record = (this+glyphRecords)[idx];

  float static_cache[sizeof (void *) * 16];
  VarRegionList::cache_t *cache = (this+varStore).create_cache (hb_array (static_cache));

  transform.scale (font->x_multf, font->y_multf);

  VarCompositeGlyph::get_path_at (font, glyph,
				  draw_session, coords, transform,
				  record,
				  decycler, edges_left, depth_left,
				  scratch,
				  cache);

  (this+varStore).destroy_cache (cache, hb_array (static_cache));

  return true;
}

//} // namespace Var
} // namespace OT

#endif
