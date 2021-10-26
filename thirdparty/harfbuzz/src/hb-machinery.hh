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

#ifndef HB_MACHINERY_HH
#define HB_MACHINERY_HH

#include "hb.hh"
#include "hb-blob.hh"

#include "hb-dispatch.hh"
#include "hb-sanitize.hh"
#include "hb-serialize.hh"


/*
 * Casts
 */

/* StructAtOffset<T>(P,Ofs) returns the struct T& that is placed at memory
 * location pointed to by P plus Ofs bytes. */
template<typename Type>
static inline const Type& StructAtOffset(const void *P, unsigned int offset)
{ return * reinterpret_cast<const Type*> ((const char *) P + offset); }
template<typename Type>
static inline Type& StructAtOffset(void *P, unsigned int offset)
{ return * reinterpret_cast<Type*> ((char *) P + offset); }
template<typename Type>
static inline const Type& StructAtOffsetUnaligned(const void *P, unsigned int offset)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
  return * reinterpret_cast<const Type*> ((const char *) P + offset);
#pragma GCC diagnostic pop
}
template<typename Type>
static inline Type& StructAtOffsetUnaligned(void *P, unsigned int offset)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
  return * reinterpret_cast<Type*> ((char *) P + offset);
#pragma GCC diagnostic pop
}

/* StructAfter<T>(X) returns the struct T& that is placed after X.
 * Works with X of variable size also.  X must implement get_size() */
template<typename Type, typename TObject>
static inline const Type& StructAfter(const TObject &X)
{ return StructAtOffset<Type>(&X, X.get_size()); }
template<typename Type, typename TObject>
static inline Type& StructAfter(TObject &X)
{ return StructAtOffset<Type>(&X, X.get_size()); }


/*
 * Size checking
 */

/* Size signifying variable-sized array */
#ifndef HB_VAR_ARRAY
#define HB_VAR_ARRAY 1
#endif

/* Check _assertion in a method environment */
#define _DEFINE_INSTANCE_ASSERTION1(_line, _assertion) \
  void _instance_assertion_on_line_##_line () const \
  { static_assert ((_assertion), ""); }
# define _DEFINE_INSTANCE_ASSERTION0(_line, _assertion) _DEFINE_INSTANCE_ASSERTION1 (_line, _assertion)
# define DEFINE_INSTANCE_ASSERTION(_assertion) _DEFINE_INSTANCE_ASSERTION0 (__LINE__, _assertion)

/* Check that _code compiles in a method environment */
#define _DEFINE_COMPILES_ASSERTION1(_line, _code) \
  void _compiles_assertion_on_line_##_line () const \
  { _code; }
# define _DEFINE_COMPILES_ASSERTION0(_line, _code) _DEFINE_COMPILES_ASSERTION1 (_line, _code)
# define DEFINE_COMPILES_ASSERTION(_code) _DEFINE_COMPILES_ASSERTION0 (__LINE__, _code)


#define DEFINE_SIZE_STATIC(size) \
  DEFINE_INSTANCE_ASSERTION (sizeof (*this) == (size)) \
  unsigned int get_size () const { return (size); } \
  static constexpr unsigned null_size = (size); \
  static constexpr unsigned min_size = (size); \
  static constexpr unsigned static_size = (size)

#define DEFINE_SIZE_UNION(size, _member) \
  DEFINE_COMPILES_ASSERTION ((void) this->u._member.static_size) \
  DEFINE_INSTANCE_ASSERTION (sizeof(this->u._member) == (size)) \
  static constexpr unsigned null_size = (size); \
  static constexpr unsigned min_size = (size)

#define DEFINE_SIZE_MIN(size) \
  DEFINE_INSTANCE_ASSERTION (sizeof (*this) >= (size)) \
  static constexpr unsigned null_size = (size); \
  static constexpr unsigned min_size = (size)

#define DEFINE_SIZE_UNBOUNDED(size) \
  DEFINE_INSTANCE_ASSERTION (sizeof (*this) >= (size)) \
  static constexpr unsigned min_size = (size)

#define DEFINE_SIZE_ARRAY(size, array) \
  DEFINE_COMPILES_ASSERTION ((void) (array)[0].static_size) \
  DEFINE_INSTANCE_ASSERTION (sizeof (*this) == (size) + (HB_VAR_ARRAY+0) * sizeof ((array)[0])) \
  static constexpr unsigned null_size = (size); \
  static constexpr unsigned min_size = (size)

#define DEFINE_SIZE_ARRAY_SIZED(size, array) \
  unsigned int get_size () const { return (size - (array).min_size + (array).get_size ()); } \
  DEFINE_SIZE_ARRAY(size, array)



/*
 * Lazy loaders.
 */

template <typename Data, unsigned int WheresData>
struct hb_data_wrapper_t
{
  static_assert (WheresData > 0, "");

  Data * get_data () const
  { return *(((Data **) (void *) this) - WheresData); }

  bool is_inert () const { return !get_data (); }

  template <typename Stored, typename Subclass>
  Stored * call_create () const { return Subclass::create (get_data ()); }
};
template <>
struct hb_data_wrapper_t<void, 0>
{
  bool is_inert () const { return false; }

  template <typename Stored, typename Funcs>
  Stored * call_create () const { return Funcs::create (); }
};

template <typename T1, typename T2> struct hb_non_void_t { typedef T1 value; };
template <typename T2> struct hb_non_void_t<void, T2> { typedef T2 value; };

template <typename Returned,
	  typename Subclass = void,
	  typename Data = void,
	  unsigned int WheresData = 0,
	  typename Stored = Returned>
struct hb_lazy_loader_t : hb_data_wrapper_t<Data, WheresData>
{
  typedef typename hb_non_void_t<Subclass,
				 hb_lazy_loader_t<Returned,Subclass,Data,WheresData,Stored>
				>::value Funcs;

  void init0 () {} /* Init, when memory is already set to 0. No-op for us. */
  void init ()  { instance.set_relaxed (nullptr); }
  void fini ()  { do_destroy (instance.get ()); }

  void free_instance ()
  {
  retry:
    Stored *p = instance.get ();
    if (unlikely (p && !cmpexch (p, nullptr)))
      goto retry;
    do_destroy (p);
  }

  static void do_destroy (Stored *p)
  {
    if (p && p != const_cast<Stored *> (Funcs::get_null ()))
      Funcs::destroy (p);
  }

  const Returned * operator -> () const { return get (); }
  const Returned & operator * () const  { return *get (); }
  explicit operator bool () const
  { return get_stored () != Funcs::get_null (); }
  template <typename C> operator const C * () const { return get (); }

  Stored * get_stored () const
  {
  retry:
    Stored *p = this->instance.get ();
    if (unlikely (!p))
    {
      if (unlikely (this->is_inert ()))
	return const_cast<Stored *> (Funcs::get_null ());

      p = this->template call_create<Stored, Funcs> ();
      if (unlikely (!p))
	p = const_cast<Stored *> (Funcs::get_null ());

      if (unlikely (!cmpexch (nullptr, p)))
      {
	do_destroy (p);
	goto retry;
      }
    }
    return p;
  }
  Stored * get_stored_relaxed () const
  {
    return this->instance.get_relaxed ();
  }

  bool cmpexch (Stored *current, Stored *value) const
  {
    /* This *must* be called when there are no other threads accessing. */
    return this->instance.cmpexch (current, value);
  }

  const Returned * get () const { return Funcs::convert (get_stored ()); }
  const Returned * get_relaxed () const { return Funcs::convert (get_stored_relaxed ()); }
  Returned * get_unconst () const { return const_cast<Returned *> (Funcs::convert (get_stored ())); }

  /* To be possibly overloaded by subclasses. */
  static Returned* convert (Stored *p) { return p; }

  /* By default null/init/fini the object. */
  static const Stored* get_null () { return &Null (Stored); }
  static Stored *create (Data *data)
  {
    Stored *p = (Stored *) hb_calloc (1, sizeof (Stored));
    if (likely (p))
      p->init (data);
    return p;
  }
  static Stored *create ()
  {
    Stored *p = (Stored *) hb_calloc (1, sizeof (Stored));
    if (likely (p))
      p->init ();
    return p;
  }
  static void destroy (Stored *p)
  {
    p->fini ();
    hb_free (p);
  }

//  private:
  /* Must only have one pointer. */
  hb_atomic_ptr_t<Stored *> instance;
};

/* Specializations. */

template <typename T, unsigned int WheresFace>
struct hb_face_lazy_loader_t : hb_lazy_loader_t<T,
						hb_face_lazy_loader_t<T, WheresFace>,
						hb_face_t, WheresFace> {};

template <typename T, unsigned int WheresFace>
struct hb_table_lazy_loader_t : hb_lazy_loader_t<T,
						 hb_table_lazy_loader_t<T, WheresFace>,
						 hb_face_t, WheresFace,
						 hb_blob_t>
{
  static hb_blob_t *create (hb_face_t *face)
  { return hb_sanitize_context_t ().reference_table<T> (face); }
  static void destroy (hb_blob_t *p) { hb_blob_destroy (p); }

  static const hb_blob_t *get_null ()
  { return hb_blob_get_empty (); }

  static const T* convert (const hb_blob_t *blob)
  { return blob->as<T> (); }

  hb_blob_t* get_blob () const { return this->get_stored (); }
};

template <typename Subclass>
struct hb_font_funcs_lazy_loader_t : hb_lazy_loader_t<hb_font_funcs_t, Subclass>
{
  static void destroy (hb_font_funcs_t *p)
  { hb_font_funcs_destroy (p); }
  static const hb_font_funcs_t *get_null ()
  { return hb_font_funcs_get_empty (); }
};
template <typename Subclass>
struct hb_unicode_funcs_lazy_loader_t : hb_lazy_loader_t<hb_unicode_funcs_t, Subclass>
{
  static void destroy (hb_unicode_funcs_t *p)
  { hb_unicode_funcs_destroy (p); }
  static const hb_unicode_funcs_t *get_null ()
  { return hb_unicode_funcs_get_empty (); }
};


#endif /* HB_MACHINERY_HH */
