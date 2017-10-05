/*
 * Copyright © 2012,2017  Google, Inc.
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

#ifndef HB_SET_PRIVATE_HH
#define HB_SET_PRIVATE_HH

#include "hb-private.hh"


/*
 * hb_set_t
 */

/* TODO Keep a free-list so we can free pages that are completely zeroed.  At that
 * point maybe also use a sentinel value for "all-1" pages? */

struct hb_set_t
{
  struct page_map_t
  {
    inline int cmp (const page_map_t *o) const { return (int) o->major - (int) major; }

    uint32_t major;
    uint32_t index;
  };

  struct page_t
  {
    inline void init0 (void) { memset (&v, 0, sizeof (v)); }
    inline void init1 (void) { memset (&v, 0xff, sizeof (v)); }

    inline unsigned int len (void) const
    { return ARRAY_LENGTH_CONST (v); }

    inline bool is_empty (void) const
    {
      for (unsigned int i = 0; i < len (); i++)
        if (v[i])
	  return false;
      return true;
    }

    inline void add (hb_codepoint_t g) { elt (g) |= mask (g); }
    inline void del (hb_codepoint_t g) { elt (g) &= ~mask (g); }
    inline bool has (hb_codepoint_t g) const { return !!(elt (g) & mask (g)); }

    inline void add_range (hb_codepoint_t a, hb_codepoint_t b)
    {
      elt_t *la = &elt (a);
      elt_t *lb = &elt (b);
      if (la == lb)
        *la |= (mask (b) << 1) - mask(a);
      else
      {
	*la |= ~(mask (a) - 1);
	la++;

	memset (la, 0xff, (char *) lb - (char *) la);

	*lb |= ((mask (b) << 1) - 1);
      }
    }

    inline bool is_equal (const page_t *other) const
    {
      return 0 == memcmp (&v, &other->v, sizeof (v));
    }

    inline unsigned int get_population (void) const
    {
      unsigned int pop = 0;
      for (unsigned int i = 0; i < len (); i++)
        pop += hb_popcount (v[i]);
      return pop;
    }

    inline bool next (hb_codepoint_t *codepoint) const
    {
      unsigned int m = (*codepoint + 1) & MASK;
      if (!m)
      {
	*codepoint = INVALID;
	return false;
      }
      unsigned int i = m / ELT_BITS;
      unsigned int j = m & ELT_MASK;

      const elt_t vv = v[i] & ~((elt_t (1) << j) - 1);
      for (const elt_t *p = &vv; i < len (); p = &v[++i])
	if (*p)
	{
	  *codepoint = i * ELT_BITS + elt_get_min (*p);
	  return true;
	}

      *codepoint = INVALID;
      return false;
    }
    inline bool previous (hb_codepoint_t *codepoint) const
    {
      unsigned int m = (*codepoint - 1) & MASK;
      if (m == MASK)
      {
	*codepoint = INVALID;
	return false;
      }
      unsigned int i = m / ELT_BITS;
      unsigned int j = m & ELT_MASK;

      const elt_t vv = v[i] & ((elt_t (1) << (j + 1)) - 1);
      for (const elt_t *p = &vv; (int) i >= 0; p = &v[--i])
	if (*p)
	{
	  *codepoint = i * ELT_BITS + elt_get_max (*p);
	  return true;
	}

      *codepoint = INVALID;
      return false;
    }
    inline hb_codepoint_t get_min (void) const
    {
      for (unsigned int i = 0; i < len (); i++)
        if (v[i])
	  return i * ELT_BITS + elt_get_min (v[i]);
      return INVALID;
    }
    inline hb_codepoint_t get_max (void) const
    {
      for (int i = len () - 1; i >= 0; i--)
        if (v[i])
	  return i * ELT_BITS + elt_get_max (v[i]);
      return 0;
    }

    typedef unsigned long long elt_t;
    static const unsigned int PAGE_BITS = 512;
    static_assert ((PAGE_BITS & ((PAGE_BITS) - 1)) == 0, "");

    static inline unsigned int elt_get_min (const elt_t &elt) { return hb_ctz (elt); }
    static inline unsigned int elt_get_max (const elt_t &elt) { return hb_bit_storage (elt) - 1; }

    typedef hb_vector_size_t<elt_t, PAGE_BITS / 8> vector_t;

    static const unsigned int ELT_BITS = sizeof (elt_t) * 8;
    static const unsigned int ELT_MASK = ELT_BITS - 1;
    static const unsigned int BITS = sizeof (vector_t) * 8;
    static const unsigned int MASK = BITS - 1;
    static_assert (PAGE_BITS == BITS, "");

    elt_t &elt (hb_codepoint_t g) { return v[(g & MASK) / ELT_BITS]; }
    elt_t const &elt (hb_codepoint_t g) const { return v[(g & MASK) / ELT_BITS]; }
    elt_t mask (hb_codepoint_t g) const { return elt_t (1) << (g & ELT_MASK); }

    vector_t v;
  };
  static_assert (page_t::PAGE_BITS == sizeof (page_t) * 8, "");

  hb_object_header_t header;
  bool successful; /* Allocations successful */
  mutable unsigned int population;
  hb_vector_t<page_map_t, 1> page_map;
  hb_vector_t<page_t, 1> pages;

  inline void init_shallow (void)
  {
    successful = true;
    population = 0;
    page_map.init ();
    pages.init ();
  }
  inline void init (void)
  {
    hb_object_init (this);
    init_shallow ();
  }
  inline void fini_shallow (void)
  {
    page_map.fini ();
    pages.fini ();
  }
  inline void fini (void)
  {
    hb_object_fini (this);
    fini_shallow ();
  }

  inline bool resize (unsigned int count)
  {
    if (unlikely (!successful)) return false;
    if (!pages.resize (count) || !page_map.resize (count))
    {
      pages.resize (page_map.len);
      successful = false;
      return false;
    }
    return true;
  }

  inline void clear (void) {
    if (unlikely (hb_object_is_inert (this)))
      return;
    successful = true;
    population = 0;
    page_map.resize (0);
    pages.resize (0);
  }
  inline bool is_empty (void) const {
    unsigned int count = pages.len;
    for (unsigned int i = 0; i < count; i++)
      if (!pages[i].is_empty ())
        return false;
    return true;
  }

  inline void dirty (void) { population = (unsigned int) -1; }

  inline void add (hb_codepoint_t g)
  {
    if (unlikely (!successful)) return;
    if (unlikely (g == INVALID)) return;
    dirty ();
    page_t *page = page_for_insert (g); if (unlikely (!page)) return;
    page->add (g);
  }
  inline bool add_range (hb_codepoint_t a, hb_codepoint_t b)
  {
    if (unlikely (!successful)) return true; /* https://github.com/harfbuzz/harfbuzz/issues/657 */
    if (unlikely (a > b || a == INVALID || b == INVALID)) return false;
    dirty ();
    unsigned int ma = get_major (a);
    unsigned int mb = get_major (b);
    if (ma == mb)
    {
      page_t *page = page_for_insert (a); if (unlikely (!page)) return false;
      page->add_range (a, b);
    }
    else
    {
      page_t *page = page_for_insert (a); if (unlikely (!page)) return false;
      page->add_range (a, major_start (ma + 1) - 1);

      for (unsigned int m = ma + 1; m < mb; m++)
      {
	page = page_for_insert (major_start (m)); if (unlikely (!page)) return false;
	page->init1 ();
      }

      page = page_for_insert (b); if (unlikely (!page)) return false;
      page->add_range (major_start (mb), b);
    }
    return true;
  }

  template <typename T>
  inline void add_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  {
    if (unlikely (!successful)) return;
    if (!count) return;
    dirty ();
    hb_codepoint_t g = *array;
    while (count)
    {
      unsigned int m = get_major (g);
      page_t *page = page_for_insert (g); if (unlikely (!page)) return;
      unsigned int start = major_start (m);
      unsigned int end = major_start (m + 1);
      do
      {
	page->add (g);

	array = (const T *) ((const char *) array + stride);
	count--;
      }
      while (count && (g = *array, start <= g && g < end));
    }
  }

  /* Might return false if array looks unsorted.
   * Used for faster rejection of corrupt data. */
  template <typename T>
  inline bool add_sorted_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  {
    if (unlikely (!successful)) return true; /* https://github.com/harfbuzz/harfbuzz/issues/657 */
    if (!count) return true;
    dirty ();
    hb_codepoint_t g = *array;
    hb_codepoint_t last_g = g;
    while (count)
    {
      unsigned int m = get_major (g);
      page_t *page = page_for_insert (g); if (unlikely (!page)) return false;
      unsigned int end = major_start (m + 1);
      do
      {
        /* If we try harder we can change the following comparison to <=;
	 * Not sure if it's worth it. */
        if (g < last_g) return false;
	last_g = g;
	page->add (g);

	array = (const T *) ((const char *) array + stride);
	count--;
      }
      while (count && (g = *array, g < end));
    }
    return true;
  }

  inline void del (hb_codepoint_t g)
  {
    /* TODO perform op even if !successful. */
    if (unlikely (!successful)) return;
    page_t *p = page_for (g);
    if (!p)
      return;
    dirty ();
    p->del (g);
  }
  inline void del_range (hb_codepoint_t a, hb_codepoint_t b)
  {
    /* TODO perform op even if !successful. */
    /* TODO Optimize, like add_range(). */
    if (unlikely (!successful)) return;
    for (unsigned int i = a; i < b + 1; i++)
      del (i);
  }
  inline bool has (hb_codepoint_t g) const
  {
    const page_t *p = page_for (g);
    if (!p)
      return false;
    return p->has (g);
  }
  inline bool intersects (hb_codepoint_t first,
			  hb_codepoint_t last) const
  {
    hb_codepoint_t c = first - 1;
    return next (&c) && c <= last;
  }
  inline void set (const hb_set_t *other)
  {
    if (unlikely (!successful)) return;
    unsigned int count = other->pages.len;
    if (!resize (count))
      return;
    population = other->population;
    memcpy (pages.arrayZ, other->pages.arrayZ, count * sizeof (pages.arrayZ[0]));
    memcpy (page_map.arrayZ, other->page_map.arrayZ, count * sizeof (page_map.arrayZ[0]));
  }

  inline bool is_equal (const hb_set_t *other) const
  {
    if (get_population () != other->get_population ())
      return false;

    unsigned int na = pages.len;
    unsigned int nb = other->pages.len;

    unsigned int a = 0, b = 0;
    for (; a < na && b < nb; )
    {
      if (page_at (a).is_empty ()) { a++; continue; }
      if (other->page_at (b).is_empty ()) { b++; continue; }
      if (page_map[a].major != other->page_map[b].major ||
	  !page_at (a).is_equal (&other->page_at (b)))
        return false;
      a++;
      b++;
    }
    for (; a < na; a++)
      if (!page_at (a).is_empty ()) { return false; }
    for (; b < nb; b++)
      if (!other->page_at (b).is_empty ()) { return false; }

    return true;
  }

  inline bool is_subset (const hb_set_t *larger_set) const
  {
    if (get_population () > larger_set->get_population ())
      return false;

    /* TODO Optimize to use pages. */
    hb_codepoint_t c = INVALID;
    while (next (&c))
      if (!larger_set->has (c))
        return false;

    return true;
  }

  template <class Op>
  inline void process (const hb_set_t *other)
  {
    if (unlikely (!successful)) return;

    dirty ();

    unsigned int na = pages.len;
    unsigned int nb = other->pages.len;
    unsigned int next_page = na;

    unsigned int count = 0, newCount = 0;
    unsigned int a = 0, b = 0;
    for (; a < na && b < nb; )
    {
      if (page_map[a].major == other->page_map[b].major)
      {
        count++;
	a++;
	b++;
      }
      else if (page_map[a].major < other->page_map[b].major)
      {
        if (Op::passthru_left)
	  count++;
        a++;
      }
      else
      {
        if (Op::passthru_right)
	  count++;
        b++;
      }
    }
    if (Op::passthru_left)
      count += na - a;
    if (Op::passthru_right)
      count += nb - b;

    if (count > pages.len)
      if (!resize (count))
        return;
    newCount = count;

    /* Process in-place backward. */
    a = na;
    b = nb;
    for (; a && b; )
    {
      if (page_map[a - 1].major == other->page_map[b - 1].major)
      {
	a--;
	b--;
	count--;
	page_map[count] = page_map[a];
	Op::process (page_at (count).v, page_at (a).v, other->page_at (b).v);
      }
      else if (page_map[a - 1].major > other->page_map[b - 1].major)
      {
	a--;
	if (Op::passthru_left)
	{
	  count--;
	  page_map[count] = page_map[a];
	}
      }
      else
      {
	b--;
	if (Op::passthru_right)
	{
	  count--;
	  page_map[count].major = other->page_map[b].major;
	  page_map[count].index = next_page++;
	  page_at (count).v = other->page_at (b).v;
	}
      }
    }
    if (Op::passthru_left)
      while (a)
      {
	a--;
	count--;
	page_map[count] = page_map [a];
      }
    if (Op::passthru_right)
      while (b)
      {
	b--;
	count--;
	page_map[count].major = other->page_map[b].major;
	page_map[count].index = next_page++;
	page_at (count).v = other->page_at (b).v;
      }
    assert (!count);
    if (pages.len > newCount)
      resize (newCount);
  }

  inline void union_ (const hb_set_t *other)
  {
    process<HbOpOr> (other);
  }
  inline void intersect (const hb_set_t *other)
  {
    process<HbOpAnd> (other);
  }
  inline void subtract (const hb_set_t *other)
  {
    process<HbOpMinus> (other);
  }
  inline void symmetric_difference (const hb_set_t *other)
  {
    process<HbOpXor> (other);
  }
  inline bool next (hb_codepoint_t *codepoint) const
  {
    if (unlikely (*codepoint == INVALID)) {
      *codepoint = get_min ();
      return *codepoint != INVALID;
    }

    page_map_t map = {get_major (*codepoint), 0};
    unsigned int i;
    page_map.bfind (map, &i);
    if (i < page_map.len && page_map[i].major == map.major)
    {
      if (pages[page_map[i].index].next (codepoint))
      {
	*codepoint += page_map[i].major * page_t::PAGE_BITS;
	return true;
      }
      i++;
    }
    for (; i < page_map.len; i++)
    {
      hb_codepoint_t m = pages[page_map[i].index].get_min ();
      if (m != INVALID)
      {
	*codepoint = page_map[i].major * page_t::PAGE_BITS + m;
	return true;
      }
    }
    *codepoint = INVALID;
    return false;
  }
  inline bool previous (hb_codepoint_t *codepoint) const
  {
    if (unlikely (*codepoint == INVALID)) {
      *codepoint = get_max ();
      return *codepoint != INVALID;
    }

    page_map_t map = {get_major (*codepoint), 0};
    unsigned int i;
    page_map.bfind (map, &i);
    if (i < page_map.len && page_map[i].major == map.major)
    {
      if (pages[page_map[i].index].previous (codepoint))
      {
	*codepoint += page_map[i].major * page_t::PAGE_BITS;
	return true;
      }
    }
    i--;
    for (; (int) i >= 0; i--)
    {
      hb_codepoint_t m = pages[page_map[i].index].get_max ();
      if (m != INVALID)
      {
	*codepoint = page_map[i].major * page_t::PAGE_BITS + m;
	return true;
      }
    }
    *codepoint = INVALID;
    return false;
  }
  inline bool next_range (hb_codepoint_t *first, hb_codepoint_t *last) const
  {
    hb_codepoint_t i;

    i = *last;
    if (!next (&i))
    {
      *last = *first = INVALID;
      return false;
    }

    /* TODO Speed up. */
    *last = *first = i;
    while (next (&i) && i == *last + 1)
      (*last)++;

    return true;
  }
  inline bool previous_range (hb_codepoint_t *first, hb_codepoint_t *last) const
  {
    hb_codepoint_t i;

    i = *first;
    if (!previous (&i))
    {
      *last = *first = INVALID;
      return false;
    }

    /* TODO Speed up. */
    *last = *first = i;
    while (previous (&i) && i == *first - 1)
      (*first)--;

    return true;
  }

  inline unsigned int get_population (void) const
  {
    if (population != (unsigned int) -1)
      return population;

    unsigned int pop = 0;
    unsigned int count = pages.len;
    for (unsigned int i = 0; i < count; i++)
      pop += pages[i].get_population ();

    population = pop;
    return pop;
  }
  inline hb_codepoint_t get_min (void) const
  {
    unsigned int count = pages.len;
    for (unsigned int i = 0; i < count; i++)
      if (!page_at (i).is_empty ())
        return page_map[i].major * page_t::PAGE_BITS + page_at (i).get_min ();
    return INVALID;
  }
  inline hb_codepoint_t get_max (void) const
  {
    unsigned int count = pages.len;
    for (int i = count - 1; i >= 0; i++)
      if (!page_at (i).is_empty ())
        return page_map[i].major * page_t::PAGE_BITS + page_at (i).get_max ();
    return INVALID;
  }

  static  const hb_codepoint_t INVALID = HB_SET_VALUE_INVALID;

  inline page_t *page_for_insert (hb_codepoint_t g)
  {
    page_map_t map = {get_major (g), pages.len};
    unsigned int i;
    if (!page_map.bfind (map, &i))
    {
      if (!resize (pages.len + 1))
	return nullptr;

      pages[map.index].init0 ();
      memmove (&page_map[i + 1], &page_map[i], (page_map.len - 1 - i) * sizeof (page_map[0]));
      page_map[i] = map;
    }
    return &pages[page_map[i].index];
  }
  inline page_t *page_for (hb_codepoint_t g)
  {
    page_map_t key = {get_major (g)};
    const page_map_t *found = page_map.bsearch (key);
    if (found)
      return &pages[found->index];
    return nullptr;
  }
  inline const page_t *page_for (hb_codepoint_t g) const
  {
    page_map_t key = {get_major (g)};
    const page_map_t *found = page_map.bsearch (key);
    if (found)
      return &pages[found->index];
    return nullptr;
  }
  inline page_t &page_at (unsigned int i) { return pages[page_map[i].index]; }
  inline const page_t &page_at (unsigned int i) const { return pages[page_map[i].index]; }
  inline unsigned int get_major (hb_codepoint_t g) const { return g / page_t::PAGE_BITS; }
  inline hb_codepoint_t major_start (unsigned int major) const { return major * page_t::PAGE_BITS; }
};


#endif /* HB_SET_PRIVATE_HH */
