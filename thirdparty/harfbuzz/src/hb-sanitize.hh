/*
 * Copyright © 2007,2008,2009,2010  Red Hat, Inc.
 * Copyright © 2012,2018  Google, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_SANITIZE_HH
#define HB_SANITIZE_HH

#include "hb.hh"
#include "hb-blob.hh"
#include "hb-dispatch.hh"


/*
 * Sanitize
 *
 *
 * === Introduction ===
 *
 * The sanitize machinery is at the core of our zero-cost font loading.  We
 * mmap() font file into memory and create a blob out of it.  Font subtables
 * are returned as a readonly sub-blob of the main font blob.  These table
 * blobs are then sanitized before use, to ensure invalid memory access does
 * not happen.  The toplevel sanitize API use is like, eg. to load the 'head'
 * table:
 *
 *   hb_blob_t *head_blob = hb_sanitize_context_t ().reference_table<OT::head> (face);
 *
 * The blob then can be converted to a head table struct with:
 *
 *   const head *head_table = head_blob->as<head> ();
 *
 * What the reference_table does is, to call hb_face_reference_table() to load
 * the table blob, sanitize it and return either the sanitized blob, or empty
 * blob if sanitization failed.  The blob->as() function returns the null
 * object of its template type argument if the blob is empty.  Otherwise, it
 * just casts the blob contents to the desired type.
 *
 * Sanitizing a blob of data with a type T works as follows (with minor
 * simplification):
 *
 *   - Cast blob content to T*, call sanitize() method of it,
 *   - If sanitize succeeded, return blob.
 *   - Otherwise, if blob is not writable, try making it writable,
 *     or copy if cannot be made writable in-place,
 *   - Call sanitize() again.  Return blob if sanitize succeeded.
 *   - Return empty blob otherwise.
 *
 *
 * === The sanitize() contract ===
 *
 * The sanitize() method of each object type shall return `true` if it's safe to
 * call other methods of the object, and `false` otherwise.
 *
 * Note that what sanitize() checks for might align with what the specification
 * describes as valid table data, but does not have to be.  In particular, we
 * do NOT want to be pedantic and concern ourselves with validity checks that
 * are irrelevant to our use of the table.  On the contrary, we want to be
 * lenient with error handling and accept invalid data to the extent that it
 * does not impose extra burden on us.
 *
 * Based on the sanitize contract, one can see that what we check for depends
 * on how we use the data in other table methods.  Ie. if other table methods
 * assume that offsets do NOT point out of the table data block, then that's
 * something sanitize() must check for (GSUB/GPOS/GDEF/etc work this way).  On
 * the other hand, if other methods do such checks themselves, then sanitize()
 * does not have to bother with them (glyf/local work this way).  The choice
 * depends on the table structure and sanitize() performance.  For example, to
 * check glyf/loca offsets in sanitize() would cost O(num-glyphs).  We try hard
 * to avoid such costs during font loading.  By postponing such checks to the
 * actual glyph loading, we reduce the sanitize cost to O(1) and total runtime
 * cost to O(used-glyphs).  As such, this is preferred.
 *
 * The same argument can be made re GSUB/GPOS/GDEF, but there, the table
 * structure is so complicated that by checking all offsets at sanitize() time,
 * we make the code much simpler in other methods, as offsets and referenced
 * objects do not need to be validated at each use site.
 */

/* This limits sanitizing time on really broken fonts. */
#ifndef HB_SANITIZE_MAX_EDITS
#define HB_SANITIZE_MAX_EDITS 32
#endif
#ifndef HB_SANITIZE_MAX_OPS_FACTOR
#define HB_SANITIZE_MAX_OPS_FACTOR 64
#endif
#ifndef HB_SANITIZE_MAX_OPS_MIN
#define HB_SANITIZE_MAX_OPS_MIN 16384
#endif
#ifndef HB_SANITIZE_MAX_OPS_MAX
#define HB_SANITIZE_MAX_OPS_MAX 0x3FFFFFFF
#endif
#ifndef HB_SANITIZE_MAX_SUBTABLES
#define HB_SANITIZE_MAX_SUBTABLES 0x4000
#endif

struct hb_sanitize_context_t :
       hb_dispatch_context_t<hb_sanitize_context_t, bool, HB_DEBUG_SANITIZE>
{
  hb_sanitize_context_t () :
	start (nullptr), end (nullptr),
	length (0),
	max_ops (0), max_subtables (0),
        recursion_depth (0),
	writable (false), edit_count (0),
	blob (nullptr),
	num_glyphs (65536),
	num_glyphs_set (false),
	lazy_some_gpos (false) {}

  const char *get_name () { return "SANITIZE"; }
  template <typename T, typename F>
  bool may_dispatch (const T *obj HB_UNUSED, const F *format)
  {
    return format->sanitize (this) &&
	   hb_barrier ();
  }
  static return_t default_return_value () { return true; }
  static return_t no_dispatch_return_value () { return false; }
  bool stop_sublookup_iteration (const return_t r) const { return !r; }

  bool visit_subtables (unsigned count)
  {
    max_subtables += count;
    return max_subtables < HB_SANITIZE_MAX_SUBTABLES;
  }

  private:
  template <typename T, typename ...Ts> auto
  _dispatch (const T &obj, hb_priority<1>, Ts&&... ds) HB_AUTO_RETURN
  ( obj.sanitize (this, std::forward<Ts> (ds)...) )
  template <typename T, typename ...Ts> auto
  _dispatch (const T &obj, hb_priority<0>, Ts&&... ds) HB_AUTO_RETURN
  ( obj.dispatch (this, std::forward<Ts> (ds)...) )
  public:
  template <typename T, typename ...Ts> auto
  dispatch (const T &obj, Ts&&... ds) HB_AUTO_RETURN
  ( _dispatch (obj, hb_prioritize, std::forward<Ts> (ds)...) )

  hb_sanitize_context_t (hb_blob_t *b) : hb_sanitize_context_t ()
  {
    init (b);

    if (blob)
      start_processing ();
  }

  ~hb_sanitize_context_t ()
  {
    if (blob)
      end_processing ();
  }

  void init (hb_blob_t *b)
  {
    this->blob = hb_blob_reference (b);
    this->writable = false;
  }

  void set_num_glyphs (unsigned int num_glyphs_)
  {
    num_glyphs = num_glyphs_;
    num_glyphs_set = true;
  }
  unsigned int get_num_glyphs () { return num_glyphs; }

  void set_max_ops (int max_ops_) { max_ops = max_ops_; }

  template <typename T>
  void set_object (const T *obj)
  {
    reset_object ();

    if (!obj) return;

    const char *obj_start = (const char *) obj;
    if (unlikely (obj_start < this->start || this->end <= obj_start))
    {
      this->start = this->end = nullptr;
      this->length = 0;
    }
    else
    {
      this->start = obj_start;
      this->end   = obj_start + hb_min (size_t (this->end - obj_start), obj->get_size ());
      this->length = this->end - this->start;
    }
  }

  void reset_object ()
  {
    this->start = this->blob->data;
    this->end = this->start + this->blob->length;
    this->length = this->end - this->start;
    assert (this->start <= this->end); /* Must not overflow. */
  }

  void start_processing ()
  {
    reset_object ();
    unsigned m;
    if (unlikely (hb_unsigned_mul_overflows (this->end - this->start, HB_SANITIZE_MAX_OPS_FACTOR, &m)))
      this->max_ops = HB_SANITIZE_MAX_OPS_MAX;
    else
      this->max_ops = hb_clamp (m,
				(unsigned) HB_SANITIZE_MAX_OPS_MIN,
				(unsigned) HB_SANITIZE_MAX_OPS_MAX);
    this->edit_count = 0;
    this->debug_depth = 0;
    this->recursion_depth = 0;

    DEBUG_MSG_LEVEL (SANITIZE, start, 0, +1,
		     "start [%p..%p] (%lu bytes)",
		     this->start, this->end,
		     (unsigned long) (this->end - this->start));
  }

  void end_processing ()
  {
    DEBUG_MSG_LEVEL (SANITIZE, this->start, 0, -1,
		     "end [%p..%p] %u edit requests",
		     this->start, this->end, this->edit_count);

    hb_blob_destroy (this->blob);
    this->blob = nullptr;
    this->start = this->end = nullptr;
    this->length = 0;
  }

  unsigned get_edit_count () { return edit_count; }


  bool check_ops(unsigned count)
  {
    /* Avoid underflow */
    if (unlikely (this->max_ops < 0 || count >= (unsigned) this->max_ops))
    {
      this->max_ops = -1;
      return false;
    }
    this->max_ops -= (int) count;
    return true;
  }

#ifndef HB_OPTIMIZE_SIZE
  HB_ALWAYS_INLINE
#endif
  bool check_range (const void *base,
		    unsigned int len) const
  {
    const char *p = (const char *) base;
    bool ok = (uintptr_t) (p - this->start) <= this->length &&
	      (unsigned int) (this->end - p) >= len &&
	      ((this->max_ops -= len) > 0);

    DEBUG_MSG_LEVEL (SANITIZE, p, this->debug_depth+1, 0,
		     "check_range [%p..%p]"
		     " (%u bytes) in [%p..%p] -> %s",
		     p, p + len, len,
		     this->start, this->end,
		     ok ? "OK" : "OUT-OF-RANGE");

    return likely (ok);
  }
#ifndef HB_OPTIMIZE_SIZE
  HB_ALWAYS_INLINE
#endif
  bool check_range_fast (const void *base,
			 unsigned int len) const
  {
    const char *p = (const char *) base;
    bool ok = ((uintptr_t) (p - this->start) <= this->length &&
	       (unsigned int) (this->end - p) >= len);

    DEBUG_MSG_LEVEL (SANITIZE, p, this->debug_depth+1, 0,
		     "check_range_fast [%p..%p]"
		     " (%u bytes) in [%p..%p] -> %s",
		     p, p + len, len,
		     this->start, this->end,
		     ok ? "OK" : "OUT-OF-RANGE");

    return likely (ok);
  }

#ifndef HB_OPTIMIZE_SIZE
  HB_ALWAYS_INLINE
#endif
  bool check_point (const void *base) const
  {
    const char *p = (const char *) base;
    bool ok = (uintptr_t) (p - this->start) <= this->length;

    DEBUG_MSG_LEVEL (SANITIZE, p, this->debug_depth+1, 0,
		     "check_point [%p]"
		     " in [%p..%p] -> %s",
		     p,
		     this->start, this->end,
		     ok ? "OK" : "OUT-OF-RANGE");

    return likely (ok);
  }

  template <typename T>
  bool check_range (const T *base,
		    unsigned int a,
		    unsigned int b) const
  {
    unsigned m;
    return !hb_unsigned_mul_overflows (a, b, &m) &&
	   this->check_range (base, m);
  }

  template <typename T>
  bool check_range (const T *base,
		    unsigned int a,
		    unsigned int b,
		    unsigned int c) const
  {
    unsigned m;
    return !hb_unsigned_mul_overflows (a, b, &m) &&
	   this->check_range (base, m, c);
  }

  template <typename T>
  HB_ALWAYS_INLINE
  bool check_array_sized (const T *base, unsigned int len, unsigned len_size) const
  {
    if (len_size >= 4)
    {
      if (unlikely (hb_unsigned_mul_overflows (len, hb_static_size (T), &len)))
	return false;
    }
    else
      len = len * hb_static_size (T);
    return this->check_range (base, len);
  }

  template <typename T>
  bool check_array (const T *base, unsigned int len) const
  {
    return this->check_range (base, len, hb_static_size (T));
  }

  template <typename T>
  bool check_array (const T *base,
		    unsigned int a,
		    unsigned int b) const
  {
    return this->check_range (base, hb_static_size (T), a, b);
  }

  bool check_start_recursion (int max_depth)
  {
    if (unlikely (recursion_depth >= max_depth)) return false;
    return ++recursion_depth;
  }

  bool end_recursion (bool result)
  {
    recursion_depth--;
    return result;
  }

  template <typename Type>
#ifndef HB_OPTIMIZE_SIZE
  HB_ALWAYS_INLINE
#endif
  bool check_struct (const Type *obj) const
  {
    if (sizeof (uintptr_t) == sizeof (uint32_t))
      return likely (this->check_range_fast (obj, obj->min_size));
    else
      return likely (this->check_point ((const char *) obj + obj->min_size));
  }

  bool may_edit (const void *base, unsigned int len)
  {
    if (this->edit_count >= HB_SANITIZE_MAX_EDITS)
      return false;

    const char *p = (const char *) base;
    this->edit_count++;

    DEBUG_MSG_LEVEL (SANITIZE, p, this->debug_depth+1, 0,
       "may_edit(%u) [%p..%p] (%u bytes) in [%p..%p] -> %s",
       this->edit_count,
       p, p + len, len,
       this->start, this->end,
       this->writable ? "GRANTED" : "DENIED");

    return this->writable;
  }

  template <typename Type, typename ValueType>
  bool try_set (const Type *obj, const ValueType &v)
  {
    if (this->may_edit (obj, hb_static_size (Type)))
    {
      * const_cast<Type *> (obj) = v;
      return true;
    }
    return false;
  }

  template <typename Type>
  hb_blob_t *sanitize_blob (hb_blob_t *blob)
  {
    bool sane;

    init (blob);

  retry:
    DEBUG_MSG_FUNC (SANITIZE, start, "start");

    start_processing ();

    if (unlikely (!start))
    {
      end_processing ();
      return blob;
    }

    Type *t = reinterpret_cast<Type *> (const_cast<char *> (start));

    sane = t->sanitize (this);
    if (sane)
    {
      if (edit_count)
      {
	DEBUG_MSG_FUNC (SANITIZE, start, "passed first round with %u edits; going for second round", edit_count);

	/* sanitize again to ensure no toe-stepping */
	edit_count = 0;
	sane = t->sanitize (this);
	if (edit_count) {
	  DEBUG_MSG_FUNC (SANITIZE, start, "requested %u edits in second round; FAILING", edit_count);
	  sane = false;
	}
      }
    }
    else
    {
      if (edit_count && !writable) {
	start = hb_blob_get_data_writable (blob, nullptr);
	end = start + blob->length;

	if (start)
	{
	  writable = true;
	  /* ok, we made it writable by relocating.  try again */
	  DEBUG_MSG_FUNC (SANITIZE, start, "retry");
	  goto retry;
	}
      }
    }

    end_processing ();

    DEBUG_MSG_FUNC (SANITIZE, start, sane ? "PASSED" : "FAILED");
    if (sane)
    {
      hb_blob_make_immutable (blob);
      return blob;
    }
    else
    {
      hb_blob_destroy (blob);
      return hb_blob_get_empty ();
    }
  }

  template <typename Type>
  hb_blob_t *reference_table (const hb_face_t *face, hb_tag_t tableTag = Type::tableTag)
  {
    if (!num_glyphs_set)
      set_num_glyphs (hb_face_get_glyph_count (face));
    return sanitize_blob<Type> (hb_face_reference_table (face, tableTag));
  }

  const char *start, *end;
  unsigned length;
  mutable int max_ops, max_subtables;
  private:
  int recursion_depth;
  bool writable;
  unsigned int edit_count;
  hb_blob_t *blob;
  unsigned int num_glyphs;
  bool  num_glyphs_set;
  public:
  bool lazy_some_gpos;
};

struct hb_sanitize_with_object_t
{
  template <typename T>
  hb_sanitize_with_object_t (hb_sanitize_context_t *c, const T& obj) : c (c)
  { c->set_object (obj); }
  ~hb_sanitize_with_object_t ()
  { c->reset_object (); }

  private:
  hb_sanitize_context_t *c;
};


#endif /* HB_SANITIZE_HH */
