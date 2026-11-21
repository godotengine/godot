#ifndef HB_OT_VAR_VARC_CC
#define HB_OT_VAR_VARC_CC
#ifdef HB_OT_VAR_VARC_CC /* Pacify -Wunused-macros. */

#include "VARC.hh"

#ifndef HB_NO_VAR_COMPOSITES

#include "../../../hb-draw.hh"
#include "../../../hb-depend-data.hh"
#include "../../../hb-ot-layout-common.hh"
#include "../../../hb-ot-layout-gdef-table.hh"

namespace OT {

//namespace Var {

#define VARC_PROCESS_TRANSFORM_COMPONENTS \
	HB_STMT_START { \
	PROCESS_TRANSFORM_COMPONENT ( 0, FWORD, HAVE_TRANSLATE_X, translateX); \
	PROCESS_TRANSFORM_COMPONENT ( 0, FWORD, HAVE_TRANSLATE_Y, translateY); \
	PROCESS_TRANSFORM_COMPONENT (12, F4DOT12, HAVE_ROTATION, rotation); \
	PROCESS_TRANSFORM_COMPONENT (10, F6DOT10, HAVE_SCALE_X, scaleX); \
	PROCESS_TRANSFORM_COMPONENT (10, F6DOT10, HAVE_SCALE_Y, scaleY); \
	PROCESS_TRANSFORM_COMPONENT (12, F4DOT12, HAVE_SKEW_X, skewX); \
	PROCESS_TRANSFORM_COMPONENT (12, F4DOT12, HAVE_SKEW_Y, skewY); \
	PROCESS_TRANSFORM_COMPONENT ( 0, FWORD, HAVE_TCENTER_X, tCenterX); \
	PROCESS_TRANSFORM_COMPONENT ( 0, FWORD, HAVE_TCENTER_Y, tCenterY); \
	} HB_STMT_END

bool
VarComponent::decompile_record (const VARC &varc,
				hb_ubytes_t total_record,
				hb_vector_t<unsigned> *axis_indices,
				hb_vector_t<double> *axis_values,
				record_t *decoded,
				int64_t *budget)
{
  const unsigned char *start = total_record.arrayZ;
  const unsigned char *record = start;
  const unsigned char *end = start + total_record.length;

  if (axis_indices) axis_indices->clear ();
  if (axis_values) axis_values->clear ();

#define READ_UINT32VAR(name) \
  HB_STMT_START { \
    if (unlikely (unsigned (end - record) < HBUINT32VAR::min_size)) return false; \
    hb_barrier (); \
    auto &varint = * (const HBUINT32VAR *) record; \
    unsigned size = varint.get_size (); \
    if (unlikely (unsigned (end - record) < size)) return false; \
    name = (uint32_t) varint; \
    record += size; \
  } HB_STMT_END
#define READ_UINT32VAR_FIELD(name, field) \
  HB_STMT_START { \
    decoded->field##_offset = record - start; \
    READ_UINT32VAR (name); \
    decoded->field##_size = record - start - decoded->field##_offset; \
  } HB_STMT_END

  READ_UINT32VAR (decoded->flags);

  decoded->gid_offset = record - start;
  if (decoded->flags & (unsigned) flags_t::GID_IS_24BIT)
  {
    decoded->gid_size = HBGlyphID24::static_size;
    if (unlikely (unsigned (end - record) < HBGlyphID24::static_size)) return false;
    hb_barrier ();
    decoded->gid = * (const HBGlyphID24 *) record;
    record += HBGlyphID24::static_size;
  }
  else
  {
    decoded->gid_size = HBGlyphID16::static_size;
    if (unlikely (unsigned (end - record) < HBGlyphID16::static_size)) return false;
    hb_barrier ();
    decoded->gid = * (const HBGlyphID16 *) record;
    record += HBGlyphID16::static_size;
  }

  if (decoded->flags & (unsigned) flags_t::HAVE_CONDITION)
    READ_UINT32VAR_FIELD (decoded->condition_index, condition);

  if (decoded->flags & (unsigned) flags_t::HAVE_AXES)
  {
    READ_UINT32VAR_FIELD (decoded->axis_indices_index, axis_indices);
    unsigned axis_count;
    if (axis_indices)
    {
      axis_indices->extend ((&varc+varc.axisIndicesList)[decoded->axis_indices_index]);
      if (unlikely (axis_indices->in_error ())) return false;
      axis_count = axis_indices->length;
    }
    else
      axis_count = hb_len ((&varc+varc.axisIndicesList)[decoded->axis_indices_index]);
    if (unlikely (budget && axis_count &&
		  !hb_budget_spend (*budget, HB_BUDGET_1, axis_count)))
      return false;
    if (axis_values)
    {
      if (unlikely (!axis_values->resize (axis_count))) return false;
      const HBUINT8 *p = (const HBUINT8 *) record;
      if (unlikely (!TupleValues::decompile (p, *axis_values,
					    (const HBUINT8 *) end)))
	return false;
      record = (const unsigned char *) p;
    }
    else if (unlikely (!TupleValues::skip (record, end, axis_count)))
      return false;
  }

  decoded->axis_values_var_idx = VarIdx::NO_VARIATION;
  if (decoded->flags & (unsigned) flags_t::AXIS_VALUES_HAVE_VARIATION)
    READ_UINT32VAR_FIELD (decoded->axis_values_var_idx, axis_values_var);

  decoded->transform_var_idx = VarIdx::NO_VARIATION;
  if (decoded->flags & (unsigned) flags_t::TRANSFORM_HAS_VARIATION)
    READ_UINT32VAR_FIELD (decoded->transform_var_idx, transform_var);

#define PROCESS_TRANSFORM_COMPONENT(shift, type, flag, name) \
  if (decoded->flags & (unsigned) flags_t::flag) \
  { \
    static_assert (type::static_size == HBINT16::static_size, ""); \
    if (unlikely (unsigned (end - record) < HBINT16::static_size)) return false; \
    hb_barrier (); \
    decoded->transform.name = * (const HBINT16 *) record; \
    record += HBINT16::static_size; \
  }
  VARC_PROCESS_TRANSFORM_COMPONENTS;
#undef PROCESS_TRANSFORM_COMPONENT

  unsigned reserved = decoded->flags & (unsigned) flags_t::RESERVED_MASK;
  while (reserved)
  {
    uint32_t discard HB_UNUSED;
    READ_UINT32VAR (discard);
    reserved &= reserved - 1;
  }

  decoded->size = record - start;
#undef READ_UINT32VAR_FIELD
#undef READ_UINT32VAR
  return true;
}

bool
VARC::closure_glyphs (hb_set_t *glyphset) const
{
  if (unlikely (glyphset->in_error ()))
    return false;

  hb_set_t pending = *glyphset;
  if (unlikely (pending.in_error ()))
    return false;

  while (pending)
  {
    hb_codepoint_t gid = pending.get_min ();
    pending.del (gid);

    unsigned index = (this+coverage).get_coverage (gid);
    if (index == NOT_COVERED) continue;

    hb_ubytes_t record = (this+glyphRecords)[index];
    while (record)
    {
      VarComponent::record_t component;
      if (unlikely (!VarComponent::decompile_record (*this, record,
						     nullptr, nullptr, &component)))
	return false;
      if (!glyphset->has (component.gid))
      {
	glyphset->add (component.gid);
	if (unlikely (glyphset->in_error ()))
	  return false;
	pending.add (component.gid);
	if (unlikely (pending.in_error ()))
	  return false;
      }
      record = record.sub_array (component.size);
    }
  }
  return true;
}

void
VARC::depend (hb_depend_data_builder_t *depend_data) const
{
  const CFF2Index &records = this+glyphRecords;
  for (auto _ : + hb_zip (this+coverage,
			  hb_range ((unsigned) records.count)))
  {
    hb_codepoint_t gid = _.first;
    hb_ubytes_t record = records[_.second];
    while (record)
    {
	VarComponent::record_t component;
	if (unlikely (!VarComponent::decompile_record (*this, record,
						       nullptr, nullptr, &component)))
	{
	  depend_data->fail ();
	  return;
	}
      depend_data->add_depend (gid, tableTag, component.gid);
      record = record.sub_array (component.size);
    }
  }
}


#ifndef HB_NO_DRAW

struct hb_transforming_pen_context_t
{
  hb_transform_t<> transform;
  hb_draw_funcs_t *dfuncs;
  void *data;
  hb_draw_state_t *st;
  int64_t *budget_remaining;
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

static int64_t *
hb_transforming_pen_get_budget_remaining (hb_draw_funcs_t *, void *draw_data, void *)
{
  return ((hb_transforming_pen_context_t *) draw_data)->budget_remaining;
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
    hb_draw_funcs_set_get_budget_remaining_func (funcs, hb_transforming_pen_get_budget_remaining, nullptr, nullptr);

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
VarComponent::get_path_at (const hb_varc_context_t &c,
			   hb_codepoint_t parent_gid,
			   hb_array_t<const int> coords,
			   hb_transform_t<> total_transform,
			   hb_ubytes_t total_record,
			   hb_scalar_cache_t *cache) const
{
  if (unlikely (!hb_budget_spend (c.budget, HB_BUDGET_4)))
    return hb_ubytes_t ();

  const unsigned char *end = total_record.arrayZ + total_record.length;
  auto &VARC = *c.font->face->table.VARC->table;
  auto &varStore = &VARC+VARC.varStore;

  auto &axisIndices = c.scratch.axisIndices;
  auto &axisValues = c.scratch.axisValues;
  record_t component;
  if (unlikely (!decompile_record (VARC, total_record,
				   &axisIndices, &axisValues, &component,
				   &c.budget)))
    return hb_ubytes_t ();

  uint32_t flags = component.flags;
  hb_codepoint_t gid = component.gid;
  const unsigned char *record = total_record.arrayZ + component.size;

  // Condition
  bool show = true;
  if (flags & (unsigned) flags_t::HAVE_CONDITION)
  {
    const auto &condition = (&VARC+VARC.conditionList)[component.condition_index];
    auto instancer = MultiItemVarStoreInstancer(&varStore, nullptr, coords, cache);
    show = condition.evaluate (coords.arrayZ, coords.length, &instancer);
  }

  // Axis values

  // Apply variations if any
  if ((flags & (unsigned) flags_t::AXIS_VALUES_HAVE_VARIATION) &&
      show && !axisValues.in_error ())
  {
    if (unlikely (!hb_budget_spend (c.budget, axisValues.length,
				    coords.length)))
      return hb_ubytes_t ();
    varStore.get_delta (component.axis_values_var_idx, coords,
			axisValues.as_array (), cache);
  }

  auto component_coords = coords;
  /* Copying coords is expensive; so we have put an arbitrary
   * limit on the max number of coords for now. */
  if ((flags & (unsigned) flags_t::RESET_UNSPECIFIED_AXES) ||
      coords.length > HB_VAR_COMPOSITE_MAX_AXES)
    component_coords = hb_array (c.font->coords, c.font->num_coords);

  uint32_t transformVarIdx = component.transform_var_idx;
  hb_transform_decomposed_t<> transform = component.transform;

  if (show)
  {
    // Only use coord_setter if there's actually any axis overrides.
    if (unlikely (axisIndices &&
		  !hb_budget_spend (c.budget, HB_BUDGET_1,
				    component_coords.length)))
      return hb_ubytes_t ();
    coord_setter_t coord_setter (axisIndices ? component_coords : hb_array<int> ());
    for (unsigned i = 0; i < axisIndices.length; i++)
      coord_setter[axisIndices[i]] = hb_clamp_to<int> (round (axisValues[i]));
    if (axisIndices)
      component_coords = coord_setter.get_coords ();

    // Apply transform variations if any
    if (transformVarIdx != VarIdx::NO_VARIATION)
    {
      double transformValues[9];
      unsigned numTransformValues = 0;
#define PROCESS_TRANSFORM_COMPONENT(shift, type, flag, name) \
	  if (flags & (unsigned) flags_t::flag) \
	    transformValues[numTransformValues++] = (double) transform.name;
      VARC_PROCESS_TRANSFORM_COMPONENTS;
#undef PROCESS_TRANSFORM_COMPONENT
      if (unlikely (!hb_budget_spend (c.budget, numTransformValues,
				      coords.length)))
	return hb_ubytes_t ();
      varStore.get_delta (transformVarIdx, coords, hb_array (transformValues, numTransformValues), cache);
      numTransformValues = 0;
#define PROCESS_TRANSFORM_COMPONENT(shift, type, flag, name) \
	  if (flags & (unsigned) flags_t::flag) \
	    transform.name = (float) transformValues[numTransformValues++];
      VARC_PROCESS_TRANSFORM_COMPONENTS;
#undef PROCESS_TRANSFORM_COMPONENT
    }

    // Divide them by their divisors
#define PROCESS_TRANSFORM_COMPONENT(shift, type, flag, name) \
	  if (shift && (flags & (unsigned) flags_t::flag)) \
	     transform.name *= 1.f / (1 << shift);
    VARC_PROCESS_TRANSFORM_COMPONENTS;
#undef PROCESS_TRANSFORM_COMPONENT

    if (!(flags & (unsigned) flags_t::HAVE_SCALE_Y))
      transform.scaleY = transform.scaleX;

    transform.rotation *= HB_PI;
    transform.skewX *= HB_PI;
    transform.skewY *= HB_PI;

    total_transform.transform (transform.to_transform ());

    bool same_coords = component_coords.length == coords.length &&
		       component_coords.arrayZ == coords.arrayZ;

    c.depth_left--;
    VARC.get_path_at (c, gid,
		      component_coords, total_transform,
		      parent_gid,
		      same_coords ? cache : nullptr);
    c.depth_left++;
  }

  return hb_ubytes_t (record, end - record);
}

bool
VARC::get_path_at (const hb_varc_context_t &c,
		   hb_codepoint_t glyph,
		   hb_array_t<const int> coords,
		   hb_transform_t<> transform,
		   hb_codepoint_t parent_glyph,
		   hb_scalar_cache_t *parent_cache) const
{
  // Don't recurse on the same glyph.
  unsigned idx = glyph == parent_glyph ?
		 NOT_COVERED :
		 (this+coverage).get_coverage (glyph);
  if (idx == NOT_COVERED)
  {
    if (c.draw_session)
    {
      /* Out of budget: draw nothing, but signal success so remaining
       * leaves are skipped instead of falling back per-glyph. */
      if (unlikely (!hb_budget_spend (c.budget, HB_BUDGET_1))) return true;

      hb_transform_t<> leaf_transform = transform;
      leaf_transform.x0 *= c.font->x_multf;
      leaf_transform.y0 *= c.font->y_multf;

      // Build a transforming pen to apply the transform.
      hb_draw_funcs_t *transformer_funcs = hb_transforming_pen_get_funcs ();
      hb_transforming_pen_context_t context {leaf_transform,
					     c.draw_session->funcs,
					     c.draw_session->draw_data,
					     &c.draw_session->st,
					     &c.budget};
      hb_draw_session_t transformer_session {transformer_funcs, &context};
      hb_draw_session_t &shape_draw_session = leaf_transform.is_identity () ? *c.draw_session : transformer_session;

      if (c.font->face->table.glyf->get_path_at (c.font, glyph, shape_draw_session, coords, c.scratch.glyf_scratch, nullptr, &shape_draw_session.get_budget ())) return true;
#ifndef HB_NO_CFF
      if (c.font->face->table.cff2->get_path_at (c.font, glyph, shape_draw_session, coords, &shape_draw_session.get_budget ())) return true;
      if (c.font->face->table.cff1->get_path (c.font, glyph, shape_draw_session, &shape_draw_session.get_budget ())) return true; // Doesn't have variations
#endif
      return false;
    }
    else if (c.extents)
    {
      if (unlikely (!hb_budget_spend (c.budget, HB_BUDGET_1))) return true;

      hb_glyph_extents_t glyph_extents;
      if (!c.font->face->table.glyf->get_extents_at (c.font, glyph, &glyph_extents, coords, &c.budget))
#ifndef HB_NO_CFF
      if (!c.font->face->table.cff2->get_extents_at (c.font, glyph, &glyph_extents, coords, &c.budget))
      if (!c.font->face->table.cff1->get_extents (c.font, glyph, &glyph_extents, &c.budget)) // Doesn't have variations
#endif
	return false;

      hb_extents_t<> comp_extents (glyph_extents);
      hb_transform_t<> leaf_transform = transform;
      leaf_transform.x0 *= c.font->x_multf;
      leaf_transform.y0 *= c.font->y_multf;
      leaf_transform.transform_extents (comp_extents);
      c.extents->union_ (comp_extents);
    }
    return true;
  }

  if (c.depth_left <= 0)
    return true;

  if (c.edges_left <= 0)
    return true;
  (c.edges_left)--;

  hb_decycler_node_t node (c.decycler);
  if (unlikely (!node.visit (glyph)))
    return true;

  hb_ubytes_t record = (this+glyphRecords)[idx];

  hb_scalar_cache_t static_cache;
  hb_scalar_cache_t *cache = parent_cache ?
				  parent_cache :
				  (this+varStore).create_cache (&static_cache);

  VarCompositeGlyph::get_path_at (c,
				  glyph,
				  coords, transform,
				  record,
				  cache);

  if (cache != parent_cache)
    (this+varStore).destroy_cache (cache, &static_cache);

  return true;
}

#endif

#undef VARC_PROCESS_TRANSFORM_COMPONENTS

//} // namespace Var
} // namespace OT

#endif

#endif /* HB_OT_VAR_VARC_CC (pacify clang) */
#endif /* HB_OT_VAR_VARC_CC */
