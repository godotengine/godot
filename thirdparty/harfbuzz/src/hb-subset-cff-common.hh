/*
 * Copyright © 2018 Adobe Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Adobe Author(s): Michiharu Ariza
 */

#ifndef HB_SUBSET_CFF_COMMON_HH
#define HB_SUBSET_CFF_COMMON_HH

#include "hb.hh"

#include "hb-subset-plan.hh"
#include "hb-cff-interp-cs-common.hh"

namespace CFF {

/* Used for writing a temporary charstring */
struct str_encoder_t
{
  str_encoder_t (str_buff_t &buff_)
    : buff (buff_) {}

  void reset () { buff.reset (); }

  void encode_byte (unsigned char b)
  {
    if (likely ((signed) buff.length < buff.allocated))
      buff.arrayZ[buff.length++] = b;
    else
      buff.push (b);
  }

  void encode_int (int v)
  {
    if ((-1131 <= v) && (v <= 1131))
    {
      if ((-107 <= v) && (v <= 107))
	encode_byte (v + 139);
      else if (v > 0)
      {
	v -= 108;
	encode_byte ((v >> 8) + OpCode_TwoBytePosInt0);
	encode_byte (v & 0xFF);
      }
      else
      {
	v = -v - 108;
	encode_byte ((v >> 8) + OpCode_TwoByteNegInt0);
	encode_byte (v & 0xFF);
      }
    }
    else
    {
      if (unlikely (v < -32768))
	v = -32768;
      else if (unlikely (v > 32767))
	v = 32767;
      encode_byte (OpCode_shortint);
      encode_byte ((v >> 8) & 0xFF);
      encode_byte (v & 0xFF);
    }
  }

  // Encode number for CharString
  void encode_num_cs (const number_t& n)
  {
    if (n.in_int_range ())
    {
      encode_int (n.to_int ());
    }
    else
    {
      int32_t v = n.to_fixed ();
      encode_byte (OpCode_fixedcs);
      encode_byte ((v >> 24) & 0xFF);
      encode_byte ((v >> 16) & 0xFF);
      encode_byte ((v >> 8) & 0xFF);
      encode_byte (v & 0xFF);
    }
  }

  // Encode number for TopDict / Private
  void encode_num_tp (const number_t& n)
  {
    if (n.in_int_range ())
    {
      // TODO longint
      encode_int (n.to_int ());
    }
    else
    {
      // Sigh. BCD
      // https://learn.microsoft.com/en-us/typography/opentype/spec/cff2#table-5-nibble-definitions
      double v = n.to_real ();
      encode_byte (OpCode_BCD);

      // Based on:
      // https://github.com/fonttools/fonttools/blob/97ed3a61cde03e17b8be36f866192fbd56f1d1a7/Lib/fontTools/misc/psCharStrings.py#L265-L294

      char buf[16];
      /* FontTools has the following comment:
       *
       * # Note: 14 decimal digits seems to be the limitation for CFF real numbers
       * # in macOS. However, we use 8 here to match the implementation of AFDKO.
       *
       * We use 8 here to match FontTools X-).
       */

      hb_locale_t clocale HB_UNUSED;
      hb_locale_t oldlocale HB_UNUSED;
      oldlocale = hb_uselocale (clocale = newlocale (LC_ALL_MASK, "C", NULL));
      snprintf (buf, sizeof (buf), "%.8G", v);
      (void) hb_uselocale (((void) freelocale (clocale), oldlocale));

      char *s = buf;
      if (s[0] == '0' && s[1] == '.')
	s++;
      else if (s[0] == '-' && s[1] == '0' && s[2] == '.')
      {
	s[1] = '-';
	s++;
      }
      hb_vector_t<char> nibbles;
      while (*s)
      {
	char c = s[0];
	s++;

	switch (c)
	{
	  case 'E':
	  {
	    char c2 = *s;
	    if (c2 == '-')
	    {
	      s++;
	      nibbles.push (0x0C); // E-
	      continue;
	    }
	    if (c2 == '+')
	      s++;
	    nibbles.push (0x0B); // E
	    continue;
	  }

	  case '.': case ',': // Comma for some European locales in case no uselocale available.
	    nibbles.push (0x0A); // .
	    continue;

	  case '-':
	    nibbles.push (0x0E); // .
	    continue;
	}

	nibbles.push (c - '0');
      }
      nibbles.push (0x0F);
      if (nibbles.length % 2)
	nibbles.push (0x0F);

      unsigned count = nibbles.length;
      for (unsigned i = 0; i < count; i += 2)
        encode_byte ((nibbles[i] << 4) | nibbles[i+1]);
    }
  }

  void encode_op (op_code_t op)
  {
    if (Is_OpCode_ESC (op))
    {
      encode_byte (OpCode_escape);
      encode_byte (Unmake_OpCode_ESC (op));
    }
    else
      encode_byte (op);
  }

  void copy_str (const unsigned char *str, unsigned length)
  {
    assert ((signed) (buff.length + length) <= buff.allocated);
    hb_memcpy (buff.arrayZ + buff.length, str, length);
    buff.length += length;
  }

  bool in_error () const { return buff.in_error (); }

  protected:

  str_buff_t &buff;
};

struct cff_sub_table_info_t {
  cff_sub_table_info_t ()
    : fd_array_link (0),
      char_strings_link (0)
  {
    fd_select.init ();
  }

  table_info_t     fd_select;
  objidx_t     	   fd_array_link;
  objidx_t     	   char_strings_link;
};

template <typename OPSTR=op_str_t>
struct cff_top_dict_op_serializer_t : op_serializer_t
{
  bool serialize (hb_serialize_context_t *c,
		  const OPSTR &opstr,
		  const cff_sub_table_info_t &info) const
  {
    TRACE_SERIALIZE (this);

    switch (opstr.op)
    {
      case OpCode_CharStrings:
	return_trace (FontDict::serialize_link4_op(c, opstr.op, info.char_strings_link, whence_t::Absolute));

      case OpCode_FDArray:
	return_trace (FontDict::serialize_link4_op(c, opstr.op, info.fd_array_link, whence_t::Absolute));

      case OpCode_FDSelect:
	return_trace (FontDict::serialize_link4_op(c, opstr.op, info.fd_select.link, whence_t::Absolute));

      default:
	return_trace (copy_opstr (c, opstr));
    }
    return_trace (true);
  }
};

struct cff_font_dict_op_serializer_t : op_serializer_t
{
  bool serialize (hb_serialize_context_t *c,
		  const op_str_t &opstr,
		  const table_info_t &privateDictInfo) const
  {
    TRACE_SERIALIZE (this);

    if (opstr.op == OpCode_Private)
    {
      /* serialize the private dict size & offset as 2-byte & 4-byte integers */
      return_trace (UnsizedByteStr::serialize_int2 (c, privateDictInfo.size) &&
		    Dict::serialize_link4_op (c, opstr.op, privateDictInfo.link, whence_t::Absolute));
    }
    else
    {
      unsigned char *d = c->allocate_size<unsigned char> (opstr.length);
      if (unlikely (!d)) return_trace (false);
      /* Faster than hb_memcpy for small strings. */
      for (unsigned i = 0; i < opstr.length; i++)
	d[i] = opstr.ptr[i];
      //hb_memcpy (d, opstr.ptr, opstr.length);
    }
    return_trace (true);
  }
};

struct flatten_param_t
{
  str_buff_t     &flatStr;
  bool	drop_hints;
  const hb_subset_plan_t *plan;
};

template <typename ACC, typename ENV, typename OPSET, op_code_t endchar_op=OpCode_Invalid>
struct subr_flattener_t
{
  subr_flattener_t (const ACC &acc_,
		    const hb_subset_plan_t *plan_)
		   : acc (acc_), plan (plan_) {}

  bool flatten (str_buff_vec_t &flat_charstrings)
  {
    unsigned count = plan->num_output_glyphs ();
    if (!flat_charstrings.resize_exact (count))
      return false;
    for (unsigned int i = 0; i < count; i++)
    {
      hb_codepoint_t  glyph;
      if (!plan->old_gid_for_new_gid (i, &glyph))
      {
	/* add an endchar only charstring for a missing glyph if CFF1 */
	if (endchar_op != OpCode_Invalid) flat_charstrings[i].push (endchar_op);
	continue;
      }
      const hb_ubytes_t str = (*acc.charStrings)[glyph];
      unsigned int fd = acc.fdSelect->get_fd (glyph);
      if (unlikely (fd >= acc.fdCount))
	return false;


      ENV env (str, acc, fd,
	       plan->normalized_coords.arrayZ, plan->normalized_coords.length);
      cs_interpreter_t<ENV, OPSET, flatten_param_t> interp (env);
      flatten_param_t  param = {
        flat_charstrings.arrayZ[i],
        (bool) (plan->flags & HB_SUBSET_FLAGS_NO_HINTING),
	plan
      };
      if (unlikely (!interp.interpret (param)))
	return false;
    }
    return true;
  }

  const ACC &acc;
  const hb_subset_plan_t *plan;
};

struct subr_closures_t
{
  subr_closures_t (unsigned int fd_count) : global_closure (), local_closures ()
  {
    local_closures.resize_exact (fd_count);
  }

  void reset ()
  {
    global_closure.clear();
    for (unsigned int i = 0; i < local_closures.length; i++)
      local_closures[i].clear();
  }

  bool in_error () const { return local_closures.in_error (); }
  hb_set_t  global_closure;
  hb_vector_t<hb_set_t> local_closures;
};

struct parsed_cs_op_t : op_str_t
{
  parsed_cs_op_t (unsigned int subr_num_ = 0) :
    subr_num (subr_num_) {}

  bool is_hinting () const { return hinting_flag; }
  void set_hinting ()       { hinting_flag = true; }

  /* The layout of this struct is designed to fit within the
   * padding of op_str_t! */

  protected:
  bool	  hinting_flag = false;

  public:
  uint16_t subr_num;
};

struct parsed_cs_str_t : parsed_values_t<parsed_cs_op_t>
{
  parsed_cs_str_t () :
    parsed (false),
    hint_dropped (false),
    has_prefix_ (false),
    has_calls_ (false)
  {
    SUPER::init ();
  }

  void add_op (op_code_t op, const byte_str_ref_t& str_ref)
  {
    if (!is_parsed ())
      SUPER::add_op (op, str_ref);
  }

  void add_call_op (op_code_t op, const byte_str_ref_t& str_ref, unsigned int subr_num)
  {
    if (!is_parsed ())
    {
      has_calls_ = true;

      /* Pop the subroutine number. */
      values.pop ();

      SUPER::add_op (op, str_ref, {subr_num});
    }
  }

  void set_prefix (const number_t &num, op_code_t op = OpCode_Invalid)
  {
    has_prefix_ = true;
    prefix_op_ = op;
    prefix_num_ = num;
  }

  bool at_end (unsigned int pos) const
  {
    return ((pos + 1 >= values.length) /* CFF2 */
	|| (values[pos + 1].op == OpCode_return));
  }

  bool is_parsed () const { return parsed; }
  void set_parsed ()      { parsed = true; }

  bool is_hint_dropped () const { return hint_dropped; }
  void set_hint_dropped ()      { hint_dropped = true; }

  bool is_vsindex_dropped () const { return vsindex_dropped; }
  void set_vsindex_dropped ()      { vsindex_dropped = true; }

  bool has_prefix () const          { return has_prefix_; }
  op_code_t prefix_op () const         { return prefix_op_; }
  const number_t &prefix_num () const { return prefix_num_; }

  bool has_calls () const          { return has_calls_; }

  void compact ()
  {
    unsigned count = values.length;
    if (!count) return;
    auto &opstr = values.arrayZ;
    unsigned j = 0;
    for (unsigned i = 1; i < count; i++)
    {
      /* See if we can combine op j and op i. */
      bool combine =
        (opstr[j].op != OpCode_callsubr && opstr[j].op != OpCode_callgsubr) &&
        (opstr[i].op != OpCode_callsubr && opstr[i].op != OpCode_callgsubr) &&
        (opstr[j].is_hinting () == opstr[i].is_hinting ()) &&
        (opstr[j].ptr + opstr[j].length == opstr[i].ptr) &&
        (opstr[j].length + opstr[i].length <= 255);

      if (combine)
      {
	opstr[j].length += opstr[i].length;
	opstr[j].op = OpCode_Invalid;
      }
      else
      {
	opstr[++j] = opstr[i];
      }
    }
    values.shrink (j + 1);
  }

  protected:
  bool    parsed : 1;
  bool    hint_dropped : 1;
  bool    vsindex_dropped : 1;
  bool    has_prefix_ : 1;
  bool    has_calls_ : 1;
  op_code_t	prefix_op_;
  number_t	prefix_num_;

  private:
  typedef parsed_values_t<parsed_cs_op_t> SUPER;
};

struct parsed_cs_str_vec_t : hb_vector_t<parsed_cs_str_t>
{
  private:
  typedef hb_vector_t<parsed_cs_str_t> SUPER;
};

struct cff_subset_accelerator_t
{
  static cff_subset_accelerator_t* create (
      hb_blob_t* original_blob,
      const parsed_cs_str_vec_t& parsed_charstrings,
      const parsed_cs_str_vec_t& parsed_global_subrs,
      const hb_vector_t<parsed_cs_str_vec_t>& parsed_local_subrs) {
    cff_subset_accelerator_t* accel =
        (cff_subset_accelerator_t*) hb_malloc (sizeof(cff_subset_accelerator_t));
    if (unlikely (!accel)) return nullptr;
    new (accel) cff_subset_accelerator_t (original_blob,
                                          parsed_charstrings,
                                          parsed_global_subrs,
                                          parsed_local_subrs);
    return accel;
  }

  static void destroy (void* value) {
    if (!value) return;

    cff_subset_accelerator_t* accel = (cff_subset_accelerator_t*) value;
    accel->~cff_subset_accelerator_t ();
    hb_free (accel);
  }

  cff_subset_accelerator_t(
      hb_blob_t* original_blob_,
      const parsed_cs_str_vec_t& parsed_charstrings_,
      const parsed_cs_str_vec_t& parsed_global_subrs_,
      const hb_vector_t<parsed_cs_str_vec_t>& parsed_local_subrs_)
  {
    parsed_charstrings = parsed_charstrings_;
    parsed_global_subrs = parsed_global_subrs_;
    parsed_local_subrs = parsed_local_subrs_;

    // the parsed charstrings point to memory in the original CFF table so we must hold a reference
    // to it to keep the memory valid.
    original_blob = hb_blob_reference (original_blob_);
  }

  ~cff_subset_accelerator_t()
  {
    hb_blob_destroy (original_blob);
    auto *mapping = glyph_to_sid_map.get_relaxed ();
    if (mapping)
    {
      mapping->~glyph_to_sid_map_t ();
      hb_free (mapping);
    }
  }

  parsed_cs_str_vec_t parsed_charstrings;
  parsed_cs_str_vec_t parsed_global_subrs;
  hb_vector_t<parsed_cs_str_vec_t> parsed_local_subrs;
  mutable hb_atomic_ptr_t<glyph_to_sid_map_t> glyph_to_sid_map;

 private:
  hb_blob_t* original_blob;
};

struct subr_subset_param_t
{
  subr_subset_param_t (parsed_cs_str_t *parsed_charstring_,
		       parsed_cs_str_vec_t *parsed_global_subrs_,
		       parsed_cs_str_vec_t *parsed_local_subrs_,
		       hb_set_t *global_closure_,
		       hb_set_t *local_closure_,
		       bool drop_hints_) :
      current_parsed_str (parsed_charstring_),
      parsed_charstring (parsed_charstring_),
      parsed_global_subrs (parsed_global_subrs_),
      parsed_local_subrs (parsed_local_subrs_),
      global_closure (global_closure_),
      local_closure (local_closure_),
      drop_hints (drop_hints_) {}

  parsed_cs_str_t *get_parsed_str_for_context (call_context_t &context)
  {
    switch (context.type)
    {
      case CSType_CharString:
	return parsed_charstring;

      case CSType_LocalSubr:
	if (likely (context.subr_num < parsed_local_subrs->length))
	  return &(*parsed_local_subrs)[context.subr_num];
	break;

      case CSType_GlobalSubr:
	if (likely (context.subr_num < parsed_global_subrs->length))
	  return &(*parsed_global_subrs)[context.subr_num];
	break;
    }
    return nullptr;
  }

  template <typename ENV>
  void set_current_str (ENV &env, bool calling)
  {
    parsed_cs_str_t *parsed_str = get_parsed_str_for_context (env.context);
    if (unlikely (!parsed_str))
    {
      env.set_error ();
      return;
    }
    /* If the called subroutine is parsed partially but not completely yet,
     * it must be because we are calling it recursively.
     * Handle it as an error. */
    if (unlikely (calling && !parsed_str->is_parsed () && (parsed_str->values.length > 0)))
      env.set_error ();
    else
    {
      if (!parsed_str->is_parsed ())
        parsed_str->alloc (env.str_ref.total_size ());
      current_parsed_str = parsed_str;
    }
  }

  parsed_cs_str_t	*current_parsed_str;

  parsed_cs_str_t	*parsed_charstring;
  parsed_cs_str_vec_t	*parsed_global_subrs;
  parsed_cs_str_vec_t	*parsed_local_subrs;
  hb_set_t      *global_closure;
  hb_set_t      *local_closure;
  bool	  drop_hints;
};

struct subr_remap_t : hb_inc_bimap_t
{
  void create (const hb_set_t *closure)
  {
    /* create a remapping of subroutine numbers from old to new.
     * no optimization based on usage counts. fonttools doesn't appear doing that either.
     */

    alloc (closure->get_population ());
    for (auto old_num : *closure)
      add (old_num);

    if (get_population () < 1240)
      bias = 107;
    else if (get_population () < 33900)
      bias = 1131;
    else
      bias = 32768;
  }

  int biased_num (unsigned int old_num) const
  {
    hb_codepoint_t new_num = get (old_num);
    return (int)new_num - bias;
  }

  protected:
  int bias;
};

struct subr_remaps_t
{
  subr_remaps_t (unsigned int fdCount)
  {
    local_remaps.resize (fdCount);
  }

  bool in_error()
  {
    return local_remaps.in_error ();
  }

  void create (subr_closures_t& closures)
  {
    global_remap.create (&closures.global_closure);
    for (unsigned int i = 0; i < local_remaps.length; i++)
      local_remaps.arrayZ[i].create (&closures.local_closures[i]);
  }

  subr_remap_t	       global_remap;
  hb_vector_t<subr_remap_t>  local_remaps;
};

template <typename SUBSETTER, typename SUBRS, typename ACC, typename ENV, typename OPSET, op_code_t endchar_op=OpCode_Invalid>
struct subr_subsetter_t
{
  subr_subsetter_t (ACC &acc_, const hb_subset_plan_t *plan_)
      : acc (acc_), plan (plan_), closures(acc_.fdCount),
        remaps(acc_.fdCount)
  {}

  /* Subroutine subsetting with --no-desubroutinize runs in phases:
   *
   * 1. execute charstrings/subroutines to determine subroutine closures
   * 2. parse out all operators and numbers
   * 3. mark hint operators and operands for removal if --no-hinting
   * 4. re-encode all charstrings and subroutines with new subroutine numbers
   *
   * Phases #1 and #2 are done at the same time in collect_subrs ().
   * Phase #3 walks charstrings/subroutines forward then backward (hence parsing required),
   * because we can't tell if a number belongs to a hint op until we see the first moveto.
   *
   * Assumption: a callsubr/callgsubr operator must immediately follow a (biased) subroutine number
   * within the same charstring/subroutine, e.g., not split across a charstring and a subroutine.
   */
  bool subset (void)
  {
    unsigned fd_count = acc.fdCount;
    const cff_subset_accelerator_t* cff_accelerator = nullptr;
    if (acc.cff_accelerator) {
      cff_accelerator = acc.cff_accelerator;
      fd_count = cff_accelerator->parsed_local_subrs.length;
    }

    if (cff_accelerator) {
      // If we are not dropping hinting then charstrings are not modified so we can
      // just use a reference to the cached copies.
      cached_charstrings.resize_exact (plan->num_output_glyphs ());
      parsed_global_subrs = &cff_accelerator->parsed_global_subrs;
      parsed_local_subrs = &cff_accelerator->parsed_local_subrs;
    } else {
      parsed_charstrings.resize_exact (plan->num_output_glyphs ());
      parsed_global_subrs_storage.resize_exact (acc.globalSubrs->count);

      if (unlikely (!parsed_local_subrs_storage.resize (fd_count))) return false;

      for (unsigned int i = 0; i < acc.fdCount; i++)
      {
        unsigned count = acc.privateDicts[i].localSubrs->count;
        parsed_local_subrs_storage[i].resize (count);
        if (unlikely (parsed_local_subrs_storage[i].in_error ())) return false;
      }

      parsed_global_subrs = &parsed_global_subrs_storage;
      parsed_local_subrs = &parsed_local_subrs_storage;
    }

    if (unlikely (remaps.in_error()
                  || cached_charstrings.in_error ()
                  || parsed_charstrings.in_error ()
                  || parsed_global_subrs->in_error ()
                  || closures.in_error ())) {
      return false;
    }

    /* phase 1 & 2 */
    for (auto _ : plan->new_to_old_gid_list)
    {
      hb_codepoint_t new_glyph = _.first;
      hb_codepoint_t old_glyph = _.second;

      const hb_ubytes_t str = (*acc.charStrings)[old_glyph];
      unsigned int fd = acc.fdSelect->get_fd (old_glyph);
      if (unlikely (fd >= acc.fdCount))
        return false;

      if (cff_accelerator)
      {
        // parsed string already exists in accelerator, copy it and move
        // on.
        if (cached_charstrings)
          cached_charstrings[new_glyph] = &cff_accelerator->parsed_charstrings[old_glyph];
        else
          parsed_charstrings[new_glyph] = cff_accelerator->parsed_charstrings[old_glyph];

        continue;
      }

      ENV env (str, acc, fd);
      cs_interpreter_t<ENV, OPSET, subr_subset_param_t> interp (env);

      parsed_charstrings[new_glyph].alloc (str.length);
      subr_subset_param_t  param (&parsed_charstrings[new_glyph],
                                  &parsed_global_subrs_storage,
                                  &parsed_local_subrs_storage[fd],
                                  &closures.global_closure,
                                  &closures.local_closures[fd],
                                  plan->flags & HB_SUBSET_FLAGS_NO_HINTING);

      if (unlikely (!interp.interpret (param)))
        return false;

      /* complete parsed string esp. copy CFF1 width or CFF2 vsindex to the parsed charstring for encoding */
      SUBSETTER::complete_parsed_str (interp.env, param, parsed_charstrings[new_glyph]);

      /* mark hint ops and arguments for drop */
      if ((plan->flags & HB_SUBSET_FLAGS_NO_HINTING) || plan->inprogress_accelerator)
      {
	subr_subset_param_t  param (&parsed_charstrings[new_glyph],
				    &parsed_global_subrs_storage,
				    &parsed_local_subrs_storage[fd],
				    &closures.global_closure,
				    &closures.local_closures[fd],
				    plan->flags & HB_SUBSET_FLAGS_NO_HINTING);

	drop_hints_param_t  drop;
	if (drop_hints_in_str (parsed_charstrings[new_glyph], param, drop))
	{
	  parsed_charstrings[new_glyph].set_hint_dropped ();
	  if (drop.vsindex_dropped)
	    parsed_charstrings[new_glyph].set_vsindex_dropped ();
	}
      }

      /* Doing this here one by one instead of compacting all at the end
       * has massive peak-memory saving.
       *
       * The compacting both saves memory and makes further operations
       * faster.
       */
      parsed_charstrings[new_glyph].compact ();
    }

    /* Since parsed strings were loaded from accelerator, we still need
     * to compute the subroutine closures which would have normally happened during
     * parsing.
     *
     * Or if we are dropping hinting, redo closure to get actually used subrs.
     */
    if ((cff_accelerator ||
	(!cff_accelerator && plan->flags & HB_SUBSET_FLAGS_NO_HINTING)) &&
        !closure_subroutines(*parsed_global_subrs,
                             *parsed_local_subrs))
      return false;

    remaps.create (closures);

    populate_subset_accelerator ();
    return true;
  }

  bool encode_charstrings (str_buff_vec_t &buffArray, bool encode_prefix = true) const
  {
    unsigned num_glyphs = plan->num_output_glyphs ();
    if (unlikely (!buffArray.resize_exact (num_glyphs)))
      return false;
    hb_codepoint_t last = 0;
    for (auto _ : plan->new_to_old_gid_list)
    {
      hb_codepoint_t gid = _.first;
      hb_codepoint_t old_glyph = _.second;

      if (endchar_op != OpCode_Invalid)
        for (; last < gid; last++)
	{
	  // Hack to point vector to static string.
	  auto &b = buffArray.arrayZ[last];
	  b.length = 1;
	  b.arrayZ = const_cast<unsigned char *>(endchar_str);
	}

      last++; // Skip over gid
      unsigned int  fd = acc.fdSelect->get_fd (old_glyph);
      if (unlikely (fd >= acc.fdCount))
	return false;
      if (unlikely (!encode_str (get_parsed_charstring (gid), fd, buffArray.arrayZ[gid], encode_prefix)))
	return false;
    }
    if (endchar_op != OpCode_Invalid)
      for (; last < num_glyphs; last++)
      {
	// Hack to point vector to static string.
	auto &b = buffArray.arrayZ[last];
	b.length = 1;
	b.arrayZ = const_cast<unsigned char *>(endchar_str);
      }

    return true;
  }

  bool encode_subrs (const parsed_cs_str_vec_t &subrs, const subr_remap_t& remap, unsigned int fd, str_buff_vec_t &buffArray) const
  {
    unsigned int  count = remap.get_population ();

    if (unlikely (!buffArray.resize_exact (count)))
      return false;
    for (unsigned int new_num = 0; new_num < count; new_num++)
    {
      hb_codepoint_t old_num = remap.backward (new_num);
      assert (old_num != CFF_UNDEF_CODE);

      if (unlikely (!encode_str (subrs[old_num], fd, buffArray[new_num])))
	return false;
    }
    return true;
  }

  bool encode_globalsubrs (str_buff_vec_t &buffArray)
  {
    return encode_subrs (*parsed_global_subrs, remaps.global_remap, 0, buffArray);
  }

  bool encode_localsubrs (unsigned int fd, str_buff_vec_t &buffArray) const
  {
    return encode_subrs ((*parsed_local_subrs)[fd], remaps.local_remaps[fd], fd, buffArray);
  }

  protected:
  struct drop_hints_param_t
  {
    drop_hints_param_t ()
      : seen_moveto (false),
	ends_in_hint (false),
	all_dropped (false),
	vsindex_dropped (false) {}

    bool  seen_moveto;
    bool  ends_in_hint;
    bool  all_dropped;
    bool  vsindex_dropped;
  };

  bool drop_hints_in_subr (parsed_cs_str_t &str, unsigned int pos,
			   parsed_cs_str_vec_t &subrs, unsigned int subr_num,
			   const subr_subset_param_t &param, drop_hints_param_t &drop)
  {
    drop.ends_in_hint = false;
    bool has_hint = drop_hints_in_str (subrs[subr_num], param, drop);

    /* if this subr ends with a stem hint (i.e., not a number; potential argument for moveto),
     * then this entire subroutine must be a hint. drop its call. */
    if (drop.ends_in_hint)
    {
      str.values[pos].set_hinting ();
      /* if this subr call is at the end of the parent subr, propagate the flag
       * otherwise reset the flag */
      if (!str.at_end (pos))
	drop.ends_in_hint = false;
    }
    else if (drop.all_dropped)
    {
      str.values[pos].set_hinting ();
    }

    return has_hint;
  }

  /* returns true if it sees a hint op before the first moveto */
  bool drop_hints_in_str (parsed_cs_str_t &str, const subr_subset_param_t &param, drop_hints_param_t &drop)
  {
    bool  seen_hint = false;

    unsigned count = str.values.length;
    auto *values = str.values.arrayZ;
    for (unsigned int pos = 0; pos < count; pos++)
    {
      bool  has_hint = false;
      switch (values[pos].op)
      {
	case OpCode_callsubr:
	  has_hint = drop_hints_in_subr (str, pos,
					*param.parsed_local_subrs, values[pos].subr_num,
					param, drop);
	  break;

	case OpCode_callgsubr:
	  has_hint = drop_hints_in_subr (str, pos,
					*param.parsed_global_subrs, values[pos].subr_num,
					param, drop);
	  break;

	case OpCode_rmoveto:
	case OpCode_hmoveto:
	case OpCode_vmoveto:
	  drop.seen_moveto = true;
	  break;

	case OpCode_hintmask:
	case OpCode_cntrmask:
	  if (drop.seen_moveto)
	  {
	    values[pos].set_hinting ();
	    break;
	  }
	  HB_FALLTHROUGH;

	case OpCode_hstemhm:
	case OpCode_vstemhm:
	case OpCode_hstem:
	case OpCode_vstem:
	  has_hint = true;
	  values[pos].set_hinting ();
	  if (str.at_end (pos))
	    drop.ends_in_hint = true;
	  break;

	case OpCode_dotsection:
	  values[pos].set_hinting ();
	  break;

	default:
	  /* NONE */
	  break;
      }
      if (has_hint)
      {
	for (int i = pos - 1; i >= 0; i--)
	{
	  parsed_cs_op_t  &csop = values[(unsigned)i];
	  if (csop.is_hinting ())
	    break;
	  csop.set_hinting ();
	  if (csop.op == OpCode_vsindexcs)
	    drop.vsindex_dropped = true;
	}
	seen_hint |= has_hint;
      }
    }

    /* Raise all_dropped flag if all operators except return are dropped from a subr.
     * It may happen even after seeing the first moveto if a subr contains
     * only (usually one) hintmask operator, then calls to this subr can be dropped.
     */
    drop.all_dropped = true;
    for (unsigned int pos = 0; pos < count; pos++)
    {
      parsed_cs_op_t  &csop = values[pos];
      if (csop.op == OpCode_return)
	break;
      if (!csop.is_hinting ())
      {
	drop.all_dropped = false;
	break;
      }
    }

    return seen_hint;
  }

  bool closure_subroutines (const parsed_cs_str_vec_t& global_subrs,
                            const hb_vector_t<parsed_cs_str_vec_t>& local_subrs)
  {
    closures.reset ();
    for (auto _ : plan->new_to_old_gid_list)
    {
      hb_codepoint_t new_glyph = _.first;
      hb_codepoint_t old_glyph = _.second;
      unsigned int fd = acc.fdSelect->get_fd (old_glyph);
      if (unlikely (fd >= acc.fdCount))
        return false;

      // Note: const cast is safe here because the collect_subr_refs_in_str only performs a
      //       closure and does not modify any of the charstrings.
      subr_subset_param_t  param (const_cast<parsed_cs_str_t*> (&get_parsed_charstring (new_glyph)),
                                  const_cast<parsed_cs_str_vec_t*> (&global_subrs),
                                  const_cast<parsed_cs_str_vec_t*> (&local_subrs[fd]),
                                  &closures.global_closure,
                                  &closures.local_closures[fd],
                                  plan->flags & HB_SUBSET_FLAGS_NO_HINTING);
      collect_subr_refs_in_str (get_parsed_charstring (new_glyph), param);
    }

    return true;
  }

  void collect_subr_refs_in_subr (unsigned int subr_num, parsed_cs_str_vec_t &subrs,
				  hb_set_t *closure,
				  const subr_subset_param_t &param)
  {
    if (closure->has (subr_num))
      return;
    closure->add (subr_num);
    collect_subr_refs_in_str (subrs[subr_num], param);
  }

  void collect_subr_refs_in_str (const parsed_cs_str_t &str,
                                 const subr_subset_param_t &param)
  {
    if (!str.has_calls ())
      return;

    for (auto &opstr : str.values)
    {
      if (!param.drop_hints || !opstr.is_hinting ())
      {
	switch (opstr.op)
	{
	  case OpCode_callsubr:
	    collect_subr_refs_in_subr (opstr.subr_num, *param.parsed_local_subrs,
				       param.local_closure, param);
	    break;

	  case OpCode_callgsubr:
	    collect_subr_refs_in_subr (opstr.subr_num, *param.parsed_global_subrs,
				       param.global_closure, param);
	    break;

	  default: break;
	}
      }
    }
  }

  bool encode_str (const parsed_cs_str_t &str, const unsigned int fd, str_buff_t &buff, bool encode_prefix = true) const
  {
    str_encoder_t  encoder (buff);
    encoder.reset ();
    bool hinting = !(plan->flags & HB_SUBSET_FLAGS_NO_HINTING);
    /* if a prefix (CFF1 width or CFF2 vsindex) has been removed along with hints,
     * re-insert it at the beginning of charstreing */
    if (encode_prefix && str.has_prefix () && !hinting && str.is_hint_dropped ())
    {
      encoder.encode_num_cs (str.prefix_num ());
      if (str.prefix_op () != OpCode_Invalid)
	encoder.encode_op (str.prefix_op ());
    }

    unsigned size = 0;
    for (auto &opstr : str.values)
    {
      size += opstr.length;
      if (opstr.op == OpCode_callsubr || opstr.op == OpCode_callgsubr)
        size += 3;
    }
    if (!buff.alloc (buff.length + size, true))
      return false;

    for (auto &opstr : str.values)
    {
      if (hinting || !opstr.is_hinting ())
      {
	switch (opstr.op)
	{
	  case OpCode_callsubr:
	    encoder.encode_int (remaps.local_remaps[fd].biased_num (opstr.subr_num));
	    encoder.copy_str (opstr.ptr, opstr.length);
	    break;

	  case OpCode_callgsubr:
	    encoder.encode_int (remaps.global_remap.biased_num (opstr.subr_num));
	    encoder.copy_str (opstr.ptr, opstr.length);
	    break;

	  default:
	    encoder.copy_str (opstr.ptr, opstr.length);
	    break;
	}
      }
    }
    return !encoder.in_error ();
  }

  void compact_parsed_subrs () const
  {
    for (auto &cs : parsed_global_subrs_storage)
      cs.compact ();
    for (auto &vec : parsed_local_subrs_storage)
      for (auto &cs : vec)
	cs.compact ();
  }

  void populate_subset_accelerator () const
  {
    if (!plan->inprogress_accelerator) return;

    compact_parsed_subrs ();

    acc.cff_accelerator =
        cff_subset_accelerator_t::create(acc.blob,
                                         parsed_charstrings,
                                         parsed_global_subrs_storage,
                                         parsed_local_subrs_storage);
  }

  const parsed_cs_str_t& get_parsed_charstring (unsigned i) const
  {
    if (cached_charstrings) return *(cached_charstrings[i]);
    return parsed_charstrings[i];
  }

  protected:
  const ACC			&acc;
  const hb_subset_plan_t	*plan;

  subr_closures_t		closures;

  hb_vector_t<const parsed_cs_str_t*>     cached_charstrings;
  const parsed_cs_str_vec_t*              parsed_global_subrs;
  const hb_vector_t<parsed_cs_str_vec_t>* parsed_local_subrs;

  subr_remaps_t			remaps;

  private:

  parsed_cs_str_vec_t		parsed_charstrings;
  parsed_cs_str_vec_t		parsed_global_subrs_storage;
  hb_vector_t<parsed_cs_str_vec_t>  parsed_local_subrs_storage;
  typedef typename SUBRS::count_type subr_count_type;
};

} /* namespace CFF */

HB_INTERNAL bool
hb_plan_subset_cff_fdselect (const hb_subset_plan_t *plan,
			    unsigned int fdCount,
			    const CFF::FDSelect &src, /* IN */
			    unsigned int &subset_fd_count /* OUT */,
			    unsigned int &subset_fdselect_size /* OUT */,
			    unsigned int &subset_fdselect_format /* OUT */,
			    hb_vector_t<CFF::code_pair_t> &fdselect_ranges /* OUT */,
			    hb_inc_bimap_t &fdmap /* OUT */);

HB_INTERNAL bool
hb_serialize_cff_fdselect (hb_serialize_context_t *c,
			  unsigned int num_glyphs,
			  const CFF::FDSelect &src,
			  unsigned int fd_count,
			  unsigned int fdselect_format,
			  unsigned int size,
			  const hb_vector_t<CFF::code_pair_t> &fdselect_ranges);

#endif /* HB_SUBSET_CFF_COMMON_HH */
