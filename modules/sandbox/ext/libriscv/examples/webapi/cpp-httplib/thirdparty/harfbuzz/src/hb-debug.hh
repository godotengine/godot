/*
 * Copyright © 2017  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_DEBUG_HH
#define HB_DEBUG_HH

#include "hb.hh"
#include "hb-atomic.hh"
#include "hb-algs.hh"


#ifndef HB_DEBUG
#define HB_DEBUG 0
#endif


/*
 * Global runtime options.
 */

struct hb_options_t
{
  bool unused : 1; /* In-case sign bit is here. */
  bool initialized : 1;
  bool uniscribe_bug_compatible : 1;
};

union hb_options_union_t {
  unsigned i;
  hb_options_t opts;
};
static_assert ((sizeof (hb_atomic_t<unsigned>) >= sizeof (hb_options_union_t)), "");

HB_INTERNAL void
_hb_options_init ();

extern HB_INTERNAL hb_atomic_t<unsigned> _hb_options;

static inline hb_options_t
hb_options ()
{
#ifdef HB_NO_GETENV
  return hb_options_t ();
#endif
  /* Make a local copy, so we can access bitfield threadsafely. */
  hb_options_union_t u;
  u.i = _hb_options;

  if (unlikely (!u.i))
  {
    _hb_options_init ();
    u.i = _hb_options;
  }

  return u.opts;
}


/*
 * Debug output (needs enabling at compile time.)
 */

static inline bool
_hb_debug (unsigned int level,
	   unsigned int max_level)
{
  return level < max_level;
}

#define DEBUG_LEVEL_ENABLED(WHAT, LEVEL) (_hb_debug ((LEVEL), HB_DEBUG_##WHAT))
#define DEBUG_ENABLED(WHAT) (DEBUG_LEVEL_ENABLED (WHAT, 0))

static inline void
_hb_print_func (const char *func)
{
  if (func)
  {
    unsigned int func_len = strlen (func);
    /* Skip "static" */
    if (0 == strncmp (func, "static ", 7))
      func += 7;
    /* Skip "typename" */
    if (0 == strncmp (func, "typename ", 9))
      func += 9;
    /* Skip return type */
    const char *space = strchr (func, ' ');
    if (space)
      func = space + 1;
    /* Skip parameter list */
    const char *paren = strchr (func, '(');
    if (paren)
      func_len = paren - func;
    fprintf (stderr, "%.*s", (int) func_len, func);
  }
}

template <int max_level> static inline void
_hb_debug_msg_va (const char *what,
		  const void *obj,
		  const char *func,
		  bool indented,
		  unsigned int level,
		  int level_dir,
		  const char *message,
		  va_list ap) HB_PRINTF_FUNC(7, 0);
template <int max_level> static inline void
_hb_debug_msg_va (const char *what,
		  const void *obj,
		  const char *func,
		  bool indented,
		  unsigned int level,
		  int level_dir,
		  const char *message,
		  va_list ap)
{
  if (!_hb_debug (level, max_level))
    return;

  fprintf (stderr, "%-10s", what ? what : "");

  if (obj)
    fprintf (stderr, "(%*p) ", (int) (2 * sizeof (void *)), obj);
  else
    fprintf (stderr, " %*s  ", (int) (2 * sizeof (void *)), "");

  if (indented) {
#define VBAR	"\342\224\202"	/* U+2502 BOX DRAWINGS LIGHT VERTICAL */
#define VRBAR	"\342\224\234"	/* U+251C BOX DRAWINGS LIGHT VERTICAL AND RIGHT */
#define DLBAR	"\342\225\256"	/* U+256E BOX DRAWINGS LIGHT ARC DOWN AND LEFT */
#define ULBAR	"\342\225\257"	/* U+256F BOX DRAWINGS LIGHT ARC UP AND LEFT */
#define LBAR	"\342\225\264"	/* U+2574 BOX DRAWINGS LIGHT LEFT */
    static const char bars[] =
      VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR
      VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR
      VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR
      VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR
      VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR VBAR;
    fprintf (stderr, "%2u %s" VRBAR "%s",
	     level,
	     bars + sizeof (bars) - 1 - hb_min ((unsigned int) sizeof (bars) - 1, (unsigned int) (sizeof (VBAR) - 1) * level),
	     level_dir ? (level_dir > 0 ? DLBAR : ULBAR) : LBAR);
  } else
    fprintf (stderr, "   " VRBAR LBAR);

  _hb_print_func (func);

  if (message)
  {
    fprintf (stderr, ": ");
    vfprintf (stderr, message, ap);
  }

  fprintf (stderr, "\n");
}
template <> inline void HB_PRINTF_FUNC(7, 0)
_hb_debug_msg_va<0> (const char *what HB_UNUSED,
		     const void *obj HB_UNUSED,
		     const char *func HB_UNUSED,
		     bool indented HB_UNUSED,
		     unsigned int level HB_UNUSED,
		     int level_dir HB_UNUSED,
		     const char *message HB_UNUSED,
		     va_list ap HB_UNUSED) {}

template <int max_level> static inline void
_hb_debug_msg (const char *what,
	       const void *obj,
	       const char *func,
	       bool indented,
	       unsigned int level,
	       int level_dir,
	       const char *message,
	       ...) HB_PRINTF_FUNC(7, 8);
template <int max_level> static inline void HB_PRINTF_FUNC(7, 8)
_hb_debug_msg (const char *what,
	       const void *obj,
	       const char *func,
	       bool indented,
	       unsigned int level,
	       int level_dir,
	       const char *message,
	       ...)
{
  va_list ap;
  va_start (ap, message);
  _hb_debug_msg_va<max_level> (what, obj, func, indented, level, level_dir, message, ap);
  va_end (ap);
}
template <> inline void
_hb_debug_msg<0> (const char *what HB_UNUSED,
		  const void *obj HB_UNUSED,
		  const char *func HB_UNUSED,
		  bool indented HB_UNUSED,
		  unsigned int level HB_UNUSED,
		  int level_dir HB_UNUSED,
		  const char *message HB_UNUSED,
		  ...) HB_PRINTF_FUNC(7, 8);
template <> inline void HB_PRINTF_FUNC(7, 8)
_hb_debug_msg<0> (const char *what HB_UNUSED,
		  const void *obj HB_UNUSED,
		  const char *func HB_UNUSED,
		  bool indented HB_UNUSED,
		  unsigned int level HB_UNUSED,
		  int level_dir HB_UNUSED,
		  const char *message HB_UNUSED,
		  ...) {}

#define DEBUG_MSG_LEVEL(WHAT, OBJ, LEVEL, LEVEL_DIR, ...)	_hb_debug_msg<HB_DEBUG_##WHAT> (#WHAT, (OBJ), nullptr,    true, (LEVEL), (LEVEL_DIR), __VA_ARGS__)
#define DEBUG_MSG(WHAT, OBJ, ...)				_hb_debug_msg<HB_DEBUG_##WHAT> (#WHAT, (OBJ), nullptr,    false, 0, 0, __VA_ARGS__)
#define DEBUG_MSG_FUNC(WHAT, OBJ, ...)				_hb_debug_msg<HB_DEBUG_##WHAT> (#WHAT, (OBJ), HB_FUNC, false, 0, 0, __VA_ARGS__)


/*
 * Printer
 */

template <typename T>
struct hb_printer_t {
  const char *print (const T&) { return "something"; }
};

template <>
struct hb_printer_t<bool> {
  const char *print (bool v) { return v ? "true" : "false"; }
};

template <>
struct hb_printer_t<hb_empty_t> {
  const char *print (hb_empty_t) { return ""; }
};


/*
 * Trace
 */

template <typename T>
static inline void _hb_warn_no_return (bool returned)
{
  if (unlikely (!returned)) {
    fprintf (stderr, "OUCH, returned with no call to return_trace().  This is a bug, please report.\n");
  }
}
template <>
/*static*/ inline void _hb_warn_no_return<hb_empty_t> (bool returned HB_UNUSED) {}
template <>
/*static*/ inline void _hb_warn_no_return<void> (bool returned HB_UNUSED) {}

template <int max_level, typename ret_t>
struct hb_auto_trace_t
{
  explicit inline hb_auto_trace_t (unsigned int *plevel_,
				   const char *what_,
				   const void *obj_,
				   const char *func,
				   const char *message,
				   ...) HB_PRINTF_FUNC(6, 7)
				   : plevel (plevel_), what (what_), obj (obj_), returned (false)
  {
    if (plevel) ++*plevel;

    va_list ap;
    va_start (ap, message);
    _hb_debug_msg_va<max_level> (what, obj, func, true, plevel ? *plevel : 0, +1, message, ap);
    va_end (ap);
  }
  ~hb_auto_trace_t ()
  {
    _hb_warn_no_return<ret_t> (returned);
    if (!returned) {
      _hb_debug_msg<max_level> (what, obj, nullptr, true, plevel ? *plevel : 1, -1, " ");
    }
    if (plevel) --*plevel;
  }

  template <typename T>
  T ret (T&& v,
	 const char *func = "",
	 unsigned int line = 0)
  {
    if (unlikely (returned)) {
      fprintf (stderr, "OUCH, double calls to return_trace().  This is a bug, please report.\n");
      return std::forward<T> (v);
    }

    _hb_debug_msg<max_level> (what, obj, func, true, plevel ? *plevel : 1, -1,
			      "return %s (line %u)",
			      hb_printer_t<hb_decay<decltype (v)>>().print (v), line);
    if (plevel) --*plevel;
    plevel = nullptr;
    returned = true;
    return std::forward<T> (v);
  }

  private:
  unsigned int *plevel;
  const char *what;
  const void *obj;
  bool returned;
};
template <typename ret_t> /* Make sure we don't use hb_auto_trace_t when not tracing. */
struct hb_auto_trace_t<0, ret_t>
{
  explicit inline hb_auto_trace_t (unsigned int *plevel_,
				   const char *what_,
				   const void *obj_,
				   const char *func,
				   const char *message,
				   ...) HB_PRINTF_FUNC(6, 7) {}

  template <typename T>
  T ret (T&& v,
	 const char *func HB_UNUSED = nullptr,
	 unsigned int line HB_UNUSED = 0) { return std::forward<T> (v); }
};

/* For disabled tracing; optimize out everything.
 * https://github.com/harfbuzz/harfbuzz/pull/605 */
template <typename ret_t>
struct hb_no_trace_t {
  template <typename T>
  T ret (T&& v,
	 const char *func HB_UNUSED = nullptr,
	 unsigned int line HB_UNUSED = 0) { return std::forward<T> (v); }
};

#define return_trace(RET) return trace.ret (RET, HB_FUNC, __LINE__)


/*
 * Instances.
 */

#ifndef HB_DEBUG_ARABIC
#define HB_DEBUG_ARABIC (HB_DEBUG+0)
#endif

#ifndef HB_DEBUG_BLOB
#define HB_DEBUG_BLOB (HB_DEBUG+0)
#endif

#ifndef HB_DEBUG_CORETEXT
#define HB_DEBUG_CORETEXT (HB_DEBUG+0)
#endif

#ifndef HB_DEBUG_DIRECTWRITE
#define HB_DEBUG_DIRECTWRITE (HB_DEBUG+0)
#endif

#ifndef HB_DEBUG_FT
#define HB_DEBUG_FT (HB_DEBUG+0)
#endif

#ifndef HB_DEBUG_JUSTIFY
#define HB_DEBUG_JUSTIFY (HB_DEBUG+0)
#endif

#ifndef HB_DEBUG_OBJECT
#define HB_DEBUG_OBJECT (HB_DEBUG+0)
#endif

#ifndef HB_DEBUG_SHAPE_PLAN
#define HB_DEBUG_SHAPE_PLAN (HB_DEBUG+0)
#endif

#ifndef HB_DEBUG_UNISCRIBE
#define HB_DEBUG_UNISCRIBE (HB_DEBUG+0)
#endif

#ifndef HB_DEBUG_WASM
#define HB_DEBUG_WASM (HB_DEBUG+0)
#endif

/*
 * With tracing.
 */

#ifndef HB_DEBUG_APPLY
#define HB_DEBUG_APPLY (HB_DEBUG+0)
#endif
#if HB_DEBUG_APPLY
#define TRACE_APPLY(this) \
	hb_auto_trace_t<HB_DEBUG_APPLY, bool> trace \
	(&c->debug_depth, c->get_name (), this, HB_FUNC, \
	 "idx %u gid %u lookup %d", \
	 c->buffer->idx, c->buffer->cur().codepoint, (int) c->lookup_index)
#else
#define TRACE_APPLY(this) hb_no_trace_t<bool> trace
#endif

#ifndef HB_DEBUG_SANITIZE
#define HB_DEBUG_SANITIZE (HB_DEBUG+0)
#endif
#if HB_DEBUG_SANITIZE
#define TRACE_SANITIZE(this) \
	hb_auto_trace_t<HB_DEBUG_SANITIZE, bool> trace \
	(&c->debug_depth, c->get_name (), this, HB_FUNC, \
	 " ")
#else
#define TRACE_SANITIZE(this) hb_no_trace_t<bool> trace
#endif

#ifndef HB_DEBUG_SERIALIZE
#define HB_DEBUG_SERIALIZE (HB_DEBUG+0)
#endif
#if HB_DEBUG_SERIALIZE
#define TRACE_SERIALIZE(this) \
	hb_auto_trace_t<HB_DEBUG_SERIALIZE, bool> trace \
	(&c->debug_depth, "SERIALIZE", c, HB_FUNC, \
	 " ")
#else
#define TRACE_SERIALIZE(this) hb_no_trace_t<bool> trace
#endif

#ifndef HB_DEBUG_SUBSET
#define HB_DEBUG_SUBSET (HB_DEBUG+0)
#endif
#if HB_DEBUG_SUBSET
#define TRACE_SUBSET(this) \
  hb_auto_trace_t<HB_DEBUG_SUBSET, bool> trace \
  (&c->debug_depth, c->get_name (), this, HB_FUNC, \
   " ")
#else
#define TRACE_SUBSET(this) hb_no_trace_t<bool> trace
#endif

#ifndef HB_DEBUG_SUBSET_REPACK
#define HB_DEBUG_SUBSET_REPACK (HB_DEBUG+0)
#endif

#ifndef HB_DEBUG_PAINT
#define HB_DEBUG_PAINT (HB_DEBUG+0)
#endif
#if HB_DEBUG_PAINT
#define TRACE_PAINT(this) \
  HB_UNUSED hb_auto_trace_t<HB_DEBUG_PAINT, void> trace \
  (&c->debug_depth, c->get_name (), this, HB_FUNC, \
   " ")
#else
#define TRACE_PAINT(this) HB_UNUSED hb_no_trace_t<void> trace
#endif


#ifndef HB_DEBUG_DISPATCH
#define HB_DEBUG_DISPATCH ( \
	HB_DEBUG_APPLY + \
	HB_DEBUG_SANITIZE + \
	HB_DEBUG_SERIALIZE + \
	HB_DEBUG_SUBSET + \
	HB_DEBUG_PAINT + \
	0)
#endif
#if HB_DEBUG_DISPATCH
#define TRACE_DISPATCH(this, format) \
	hb_auto_trace_t<context_t::max_debug_depth, typename context_t::return_t> trace \
	(&c->debug_depth, c->get_name (), this, HB_FUNC, \
	 "format %u", (unsigned) format)
#else
#define TRACE_DISPATCH(this, format) hb_no_trace_t<typename context_t::return_t> trace
#endif


#ifndef HB_BUFFER_MESSAGE_MORE
#define HB_BUFFER_MESSAGE_MORE (HB_DEBUG+1)
#endif


#endif /* HB_DEBUG_HH */
