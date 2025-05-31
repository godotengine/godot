/*
 * Copyright © 2012,2017  Google, Inc.
 * Copyright © 2021 Behdad Esfahbod
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

#ifndef HB_BIT_SET_HH
#define HB_BIT_SET_HH

#include "hb.hh"
#include "hb-bit-page.hh"


struct hb_bit_set_t
{
  hb_bit_set_t () = default;
  ~hb_bit_set_t () = default;

  hb_bit_set_t (const hb_bit_set_t& other) : hb_bit_set_t () { set (other, true); }
  hb_bit_set_t ( hb_bit_set_t&& other)  noexcept : hb_bit_set_t () { hb_swap (*this, other); }
  hb_bit_set_t& operator= (const hb_bit_set_t& other) { set (other); return *this; }
  hb_bit_set_t& operator= (hb_bit_set_t&& other)  noexcept { hb_swap (*this, other); return *this; }
  friend void swap (hb_bit_set_t &a, hb_bit_set_t &b) noexcept
  {
    if (likely (!a.successful || !b.successful))
      return;
    hb_swap (a.population, b.population);
    hb_swap (a.last_page_lookup, b.last_page_lookup);
    hb_swap (a.page_map, b.page_map);
    hb_swap (a.pages, b.pages);
  }

  void init ()
  {
    successful = true;
    population = 0;
    last_page_lookup = 0;
    page_map.init ();
    pages.init ();
  }
  void fini ()
  {
    page_map.fini ();
    pages.fini ();
  }

  using page_t = hb_bit_page_t;
  struct page_map_t
  {
    int cmp (const page_map_t &o) const { return cmp (o.major); }
    int cmp (uint32_t o_major) const { return (int) o_major - (int) major; }

    uint32_t major;
    uint32_t index;
  };

  bool successful = true; /* Allocations successful */
  mutable unsigned int population = 0;
  mutable hb_atomic_int_t last_page_lookup = 0;
  hb_sorted_vector_t<page_map_t> page_map;
  hb_vector_t<page_t> pages;

  void err () { if (successful) successful = false; } /* TODO Remove */
  bool in_error () const { return !successful; }

  bool resize (unsigned int count, bool clear = true, bool exact_size = false)
  {
    if (unlikely (!successful)) return false;

    if (pages.length < count && count <= 2)
      exact_size = true; // Most sets are small and local

    if (unlikely (!pages.resize (count, clear, exact_size) ||
	!page_map.resize (count, clear)))
    {
      pages.resize (page_map.length, clear, exact_size);
      successful = false;
      return false;
    }
    return true;
  }

  void alloc (unsigned sz)
  {
    sz >>= (page_t::PAGE_BITS_LOG_2 - 1);
    pages.alloc (sz);
    page_map.alloc (sz);
  }

  void reset ()
  {
    successful = true;
    clear ();
  }

  void clear ()
  {
    resize (0);
    if (likely (successful))
      population = 0;
  }
  bool is_empty () const
  {
    unsigned int count = pages.length;
    for (unsigned int i = 0; i < count; i++)
      if (!pages[i].is_empty ())
	return false;
    return true;
  }
  explicit operator bool () const { return !is_empty (); }

  uint32_t hash () const
  {
    uint32_t h = 0;
    for (auto &map : page_map)
    {
      auto &page = pages.arrayZ[map.index];
      if (unlikely (page.is_empty ())) continue;
      h = h * 31 + hb_hash (map.major) + hb_hash (page);
    }
    return h;
  }

  private:
  void dirty () { population = UINT_MAX; }
  public:

  void add (hb_codepoint_t g)
  {
    if (unlikely (!successful)) return;
    if (unlikely (g == INVALID)) return;
    dirty ();
    page_t *page = page_for (g, true); if (unlikely (!page)) return;
    page->add (g);
  }
  bool add_range (hb_codepoint_t a, hb_codepoint_t b)
  {
    if (unlikely (!successful)) return true; /* https://github.com/harfbuzz/harfbuzz/issues/657 */
    if (unlikely (a > b || a == INVALID || b == INVALID)) return false;
    dirty ();
    unsigned int ma = get_major (a);
    unsigned int mb = get_major (b);
    if (ma == mb)
    {
      page_t *page = page_for (a, true); if (unlikely (!page)) return false;
      page->add_range (a, b);
    }
    else
    {
      page_t *page = page_for (a, true); if (unlikely (!page)) return false;
      page->add_range (a, major_start (ma + 1) - 1);

      for (unsigned int m = ma + 1; m < mb; m++)
      {
	page = page_for (major_start (m), true); if (unlikely (!page)) return false;
	page->init1 ();
      }

      page = page_for (b, true); if (unlikely (!page)) return false;
      page->add_range (major_start (mb), b);
    }
    return true;
  }

  /* Duplicated here from hb-machinery.hh to avoid including it. */
  template<typename Type>
  static inline const Type& StructAtOffsetUnaligned(const void *P, unsigned int offset)
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
    return * reinterpret_cast<const Type*> ((const char *) P + offset);
#pragma GCC diagnostic pop
  }

  template <typename T>
  void set_array (bool v, const T *array, unsigned int count, unsigned int stride=sizeof(T))
  {
    if (unlikely (!successful)) return;
    if (!count) return;
    dirty ();
    hb_codepoint_t g = *array;
    while (count)
    {
      unsigned int m = get_major (g);
      page_t *page = page_for (g, v); if (unlikely (v && !page)) return;
      unsigned int start = major_start (m);
      unsigned int end = major_start (m + 1);
      do
      {
        if (g != INVALID && (v || page)) /* The v check is to optimize out the page check if v is true. */
	  page->set (g, v);

	array = &StructAtOffsetUnaligned<T> (array, stride);
	count--;
      }
      while (count && (g = *array, start <= g && g < end));
    }
  }

  template <typename T>
  void add_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  { set_array (true, array, count, stride); }
  template <typename T>
  void add_array (const hb_array_t<const T>& arr) { add_array (&arr, arr.len ()); }

  template <typename T>
  void del_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  { set_array (false, array, count, stride); }
  template <typename T>
  void del_array (const hb_array_t<const T>& arr) { del_array (&arr, arr.len ()); }

  /* Might return false if array looks unsorted.
   * Used for faster rejection of corrupt data. */
  template <typename T>
  bool set_sorted_array (bool v, const T *array, unsigned int count, unsigned int stride=sizeof(T))
  {
    if (unlikely (!successful)) return true; /* https://github.com/harfbuzz/harfbuzz/issues/657 */
    if (unlikely (!count)) return true;
    dirty ();
    hb_codepoint_t g = *array;
    hb_codepoint_t last_g = g;
    while (count)
    {
      unsigned int m = get_major (g);
      page_t *page = page_for (g, v); if (unlikely (v && !page)) return false;
      unsigned int end = major_start (m + 1);
      do
      {
	/* If we try harder we can change the following comparison to <=;
	 * Not sure if it's worth it. */
	if (g < last_g) return false;
	last_g = g;

        if (g != INVALID && (v || page)) /* The v check is to optimize out the page check if v is true. */
	  page->add (g);

	array = &StructAtOffsetUnaligned<T> (array, stride);
	count--;
      }
      while (count && (g = *array, g < end));
    }
    return true;
  }

  template <typename T>
  bool add_sorted_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  { return set_sorted_array (true, array, count, stride); }
  template <typename T>
  bool add_sorted_array (const hb_sorted_array_t<const T>& arr) { return add_sorted_array (&arr, arr.len ()); }

  template <typename T>
  bool del_sorted_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  { return set_sorted_array (false, array, count, stride); }
  template <typename T>
  bool del_sorted_array (const hb_sorted_array_t<const T>& arr) { return del_sorted_array (&arr, arr.len ()); }

  void del (hb_codepoint_t g)
  {
    if (unlikely (!successful)) return;
    page_t *page = page_for (g);
    if (!page)
      return;
    dirty ();
    page->del (g);
  }

  private:
  void del_pages (int ds, int de)
  {
    if (ds <= de)
    {
      // Pre-allocate the workspace that compact() will need so we can bail on allocation failure
      // before attempting to rewrite the page map.
      hb_vector_t<unsigned> compact_workspace;
      if (unlikely (!allocate_compact_workspace (compact_workspace))) return;

      unsigned int write_index = 0;
      for (unsigned int i = 0; i < page_map.length; i++)
      {
	int m = (int) page_map.arrayZ[i].major;
	if (m < ds || de < m)
	  page_map.arrayZ[write_index++] = page_map.arrayZ[i];
      }
      compact (compact_workspace, write_index);
      resize (write_index);
    }
  }


  public:
  void del_range (hb_codepoint_t a, hb_codepoint_t b)
  {
    if (unlikely (!successful)) return;
    if (unlikely (a > b || a == INVALID)) return;
    dirty ();
    unsigned int ma = get_major (a);
    unsigned int mb = get_major (b);
    /* Delete pages from ds through de if ds <= de. */
    int ds = (a == major_start (ma))? (int) ma: (int) (ma + 1);
    int de = (b + 1 == major_start (mb + 1))? (int) mb: ((int) mb - 1);
    if (ds > de || (int) ma < ds)
    {
      page_t *page = page_for (a);
      if (page)
      {
	if (ma == mb)
	  page->del_range (a, b);
	else
	  page->del_range (a, major_start (ma + 1) - 1);
      }
    }
    if (de < (int) mb && ma != mb)
    {
      page_t *page = page_for (b);
      if (page)
	page->del_range (major_start (mb), b);
    }
    del_pages (ds, de);
  }

  bool get (hb_codepoint_t g) const
  {
    const page_t *page = page_for (g);
    if (!page)
      return false;
    return page->get (g);
  }
  bool may_have (hb_codepoint_t g) const { return get (g); }

  /* Has interface. */
  bool operator [] (hb_codepoint_t k) const { return get (k); }
  bool has (hb_codepoint_t k) const { return (*this)[k]; }
  /* Predicate. */
  bool operator () (hb_codepoint_t k) const { return has (k); }

  /* Sink interface. */
  hb_bit_set_t& operator << (hb_codepoint_t v)
  { add (v); return *this; }
  hb_bit_set_t& operator << (const hb_codepoint_pair_t& range)
  { add_range (range.first, range.second); return *this; }

  bool intersects (const hb_bit_set_t &other) const
  {
    unsigned int na = pages.length;
    unsigned int nb = other.pages.length;

    unsigned int a = 0, b = 0;
    for (; a < na && b < nb; )
    {
      if (page_map.arrayZ[a].major == other.page_map.arrayZ[b].major)
      {
	if (page_at (a).intersects (other.page_at (b)))
	  return true;
	a++;
	b++;
      }
      else if (page_map.arrayZ[a].major < other.page_map.arrayZ[b].major)
	a++;
      else
	b++;
    }
    return false;
  }
  bool may_intersect (const hb_bit_set_t &other) const
  { return intersects (other); }

  bool intersects (hb_codepoint_t first, hb_codepoint_t last) const
  {
    hb_codepoint_t c = first - 1;
    return next (&c) && c <= last;
  }
  void set (const hb_bit_set_t &other, bool exact_size = false)
  {
    if (unlikely (!successful)) return;
    unsigned int count = other.pages.length;
    if (unlikely (!resize (count, false, exact_size)))
      return;
    population = other.population;

    page_map = other.page_map;
    pages = other.pages;
  }

  bool is_equal (const hb_bit_set_t &other) const
  {
    if (has_population () && other.has_population () &&
	population != other.population)
      return false;

    unsigned int na = pages.length;
    unsigned int nb = other.pages.length;

    unsigned int a = 0, b = 0;
    for (; a < na && b < nb; )
    {
      if (page_at (a).is_empty ()) { a++; continue; }
      if (other.page_at (b).is_empty ()) { b++; continue; }
      if (page_map.arrayZ[a].major != other.page_map.arrayZ[b].major ||
	  !page_at (a).is_equal (other.page_at (b)))
	return false;
      a++;
      b++;
    }
    for (; a < na; a++)
      if (!page_at (a).is_empty ()) { return false; }
    for (; b < nb; b++)
      if (!other.page_at (b).is_empty ()) { return false; }

    return true;
  }

  bool is_subset (const hb_bit_set_t &larger_set) const
  {
    if (has_population () && larger_set.has_population () &&
	population > larger_set.population)
      return false;

    uint32_t spi = 0;
    for (uint32_t lpi = 0; spi < page_map.length && lpi < larger_set.page_map.length; lpi++)
    {
      uint32_t spm = page_map.arrayZ[spi].major;
      uint32_t lpm = larger_set.page_map.arrayZ[lpi].major;
      auto sp = page_at (spi);

      if (spm < lpm && !sp.is_empty ())
        return false;

      if (lpm < spm)
        continue;

      auto lp = larger_set.page_at (lpi);
      if (!sp.is_subset (lp))
        return false;

      spi++;
    }

    while (spi < page_map.length)
      if (!page_at (spi++).is_empty ())
        return false;

    return true;
  }

  private:
  bool allocate_compact_workspace (hb_vector_t<unsigned>& workspace)
  {
    if (unlikely (!workspace.resize_exact (pages.length)))
    {
      successful = false;
      return false;
    }

    return true;
  }

  /*
   * workspace should be a pre-sized vector allocated to hold at exactly pages.length
   * elements.
   */
  void compact (hb_vector_t<unsigned>& workspace,
                unsigned int length)
  {
    assert(workspace.length == pages.length);
    hb_vector_t<unsigned>& old_index_to_page_map_index = workspace;

    hb_fill (old_index_to_page_map_index.writer(), 0xFFFFFFFF);
    for (unsigned i = 0; i < length; i++)
      old_index_to_page_map_index[page_map[i].index] =  i;

    compact_pages (old_index_to_page_map_index);
  }
  void compact_pages (const hb_vector_t<unsigned>& old_index_to_page_map_index)
  {
    unsigned int write_index = 0;
    for (unsigned int i = 0; i < pages.length; i++)
    {
      if (old_index_to_page_map_index[i] == 0xFFFFFFFF) continue;

      if (write_index < i)
	pages[write_index] = pages[i];

      page_map[old_index_to_page_map_index[i]].index = write_index;
      write_index++;
    }
  }
  public:

  void process_ (hb_bit_page_t::vector_t (*op) (const hb_bit_page_t::vector_t &, const hb_bit_page_t::vector_t &),
		 bool passthru_left, bool passthru_right,
		 const hb_bit_set_t &other)
  {
    if (unlikely (!successful)) return;

    dirty ();

    unsigned int na = pages.length;
    unsigned int nb = other.pages.length;
    unsigned int next_page = na;

    unsigned int count = 0, newCount = 0;
    unsigned int a = 0, b = 0;
    unsigned int write_index = 0;

    // Pre-allocate the workspace that compact() will need so we can bail on allocation failure
    // before attempting to rewrite the page map.
    hb_vector_t<unsigned> compact_workspace;
    if (!passthru_left && unlikely (!allocate_compact_workspace (compact_workspace))) return;

    for (; a < na && b < nb; )
    {
      if (page_map.arrayZ[a].major == other.page_map.arrayZ[b].major)
      {
	if (!passthru_left)
	{
	  // Move page_map entries that we're keeping from the left side set
	  // to the front of the page_map vector. This isn't necessary if
	  // passthru_left is set since no left side pages will be removed
	  // in that case.
	  if (write_index < a)
	    page_map.arrayZ[write_index] = page_map.arrayZ[a];
	  write_index++;
	}

	count++;
	a++;
	b++;
      }
      else if (page_map.arrayZ[a].major < other.page_map.arrayZ[b].major)
      {
	if (passthru_left)
	  count++;
	a++;
      }
      else
      {
	if (passthru_right)
	  count++;
	b++;
      }
    }
    if (passthru_left)
      count += na - a;
    if (passthru_right)
      count += nb - b;

    if (!passthru_left)
    {
      na  = write_index;
      next_page = write_index;
      compact (compact_workspace, write_index);
    }

    if (unlikely (!resize (count)))
      return;

    newCount = count;

    /* Process in-place backward. */
    a = na;
    b = nb;
    for (; a && b; )
    {
      if (page_map.arrayZ[a - 1].major == other.page_map.arrayZ[b - 1].major)
      {
	a--;
	b--;
	count--;
	page_map.arrayZ[count] = page_map.arrayZ[a];
	page_at (count).v = op (page_at (a).v, other.page_at (b).v);
	page_at (count).dirty ();
      }
      else if (page_map.arrayZ[a - 1].major > other.page_map.arrayZ[b - 1].major)
      {
	a--;
	if (passthru_left)
	{
	  count--;
	  page_map.arrayZ[count] = page_map.arrayZ[a];
	}
      }
      else
      {
	b--;
	if (passthru_right)
	{
	  count--;
	  page_map.arrayZ[count].major = other.page_map.arrayZ[b].major;
	  page_map.arrayZ[count].index = next_page++;
	  page_at (count) = other.page_at (b);
	}
      }
    }
    if (passthru_left)
      while (a)
      {
	a--;
	count--;
	page_map.arrayZ[count] = page_map.arrayZ[a];
      }
    if (passthru_right)
      while (b)
      {
	b--;
	count--;
	page_map.arrayZ[count].major = other.page_map.arrayZ[b].major;
	page_map.arrayZ[count].index = next_page++;
	page_at (count) = other.page_at (b);
      }
    assert (!count);
    resize (newCount);
  }
  template <typename Op>
  static hb_bit_page_t::vector_t
  op_ (const hb_bit_page_t::vector_t &a, const hb_bit_page_t::vector_t &b)
  { return Op{} (a, b); }
  template <typename Op>
  void process (const Op& op, const hb_bit_set_t &other)
  {
    process_ (op_<Op>, op (1, 0), op (0, 1), other);
  }

  void union_ (const hb_bit_set_t &other) { process (hb_bitwise_or, other); }
  void intersect (const hb_bit_set_t &other) { process (hb_bitwise_and, other); }
  void subtract (const hb_bit_set_t &other) { process (hb_bitwise_gt, other); }
  void symmetric_difference (const hb_bit_set_t &other) { process (hb_bitwise_xor, other); }

  bool next (hb_codepoint_t *codepoint) const
  {
    if (unlikely (*codepoint == INVALID)) {
      *codepoint = get_min ();
      return *codepoint != INVALID;
    }

    const auto* page_map_array = page_map.arrayZ;
    unsigned int major = get_major (*codepoint);
    unsigned int i = last_page_lookup;

    if (unlikely (i >= page_map.length || page_map_array[i].major != major))
    {
      page_map.bfind (major, &i, HB_NOT_FOUND_STORE_CLOSEST);
      if (i >= page_map.length) {
        *codepoint = INVALID;
        return false;
      }
      last_page_lookup = i;
    }

    const auto* pages_array = pages.arrayZ;
    const page_map_t &current = page_map_array[i];
    if (likely (current.major == major))
    {
      if (pages_array[current.index].next (codepoint))
      {
        *codepoint += current.major * page_t::PAGE_BITS;
        return true;
      }
      i++;
    }

    for (; i < page_map.length; i++)
    {
      const page_map_t &current = page_map_array[i];
      hb_codepoint_t m = pages_array[current.index].get_min ();
      if (m != INVALID)
      {
	*codepoint = current.major * page_t::PAGE_BITS + m;
        last_page_lookup = i;
	return true;
      }
    }
    *codepoint = INVALID;
    return false;
  }
  bool previous (hb_codepoint_t *codepoint) const
  {
    if (unlikely (*codepoint == INVALID)) {
      *codepoint = get_max ();
      return *codepoint != INVALID;
    }

    page_map_t map = {get_major (*codepoint), 0};
    unsigned int i;
    page_map.bfind (map, &i, HB_NOT_FOUND_STORE_CLOSEST);
    if (i < page_map.length && page_map.arrayZ[i].major == map.major)
    {
      if (pages[page_map.arrayZ[i].index].previous (codepoint))
      {
	*codepoint += page_map.arrayZ[i].major * page_t::PAGE_BITS;
	return true;
      }
    }
    i--;
    for (; (int) i >= 0; i--)
    {
      hb_codepoint_t m = pages.arrayZ[page_map.arrayZ[i].index].get_max ();
      if (m != INVALID)
      {
	*codepoint = page_map.arrayZ[i].major * page_t::PAGE_BITS + m;
	return true;
      }
    }
    *codepoint = INVALID;
    return false;
  }
  bool next_range (hb_codepoint_t *first, hb_codepoint_t *last) const
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
  bool previous_range (hb_codepoint_t *first, hb_codepoint_t *last) const
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

  unsigned int next_many (hb_codepoint_t  codepoint,
			  hb_codepoint_t *out,
			  unsigned int    size) const
  {
    // By default, start at the first bit of the first page of values.
    unsigned int start_page = 0;
    unsigned int start_page_value = 0;
    if (unlikely (codepoint != INVALID))
    {
      const auto* page_map_array = page_map.arrayZ;
      unsigned int major = get_major (codepoint);
      unsigned int i = last_page_lookup;
      if (unlikely (i >= page_map.length || page_map_array[i].major != major))
      {
	page_map.bfind (major, &i, HB_NOT_FOUND_STORE_CLOSEST);
	if (i >= page_map.length)
	  return 0;  // codepoint is greater than our max element.
      }
      start_page = i;
      start_page_value = page_remainder (codepoint + 1);
      if (unlikely (start_page_value == 0))
      {
        // The export-after value was last in the page. Start on next page.
        start_page++;
        start_page_value = 0;
      }
    }

    unsigned int initial_size = size;
    for (unsigned int i = start_page; i < page_map.length && size; i++)
    {
      uint32_t base = major_start (page_map.arrayZ[i].major);
      unsigned int n = pages[page_map.arrayZ[i].index].write (base, start_page_value, out, size);
      out += n;
      size -= n;
      start_page_value = 0;
    }
    return initial_size - size;
  }

  unsigned int next_many_inverted (hb_codepoint_t  codepoint,
				   hb_codepoint_t *out,
				   unsigned int    size) const
  {
    unsigned int initial_size = size;
    // By default, start at the first bit of the first page of values.
    unsigned int start_page = 0;
    unsigned int start_page_value = 0;
    if (unlikely (codepoint != INVALID))
    {
      const auto* page_map_array = page_map.arrayZ;
      unsigned int major = get_major (codepoint);
      unsigned int i = last_page_lookup;
      if (unlikely (i >= page_map.length || page_map_array[i].major != major))
      {
        page_map.bfind(major, &i, HB_NOT_FOUND_STORE_CLOSEST);
        if (unlikely (i >= page_map.length))
        {
          // codepoint is greater than our max element.
          while (++codepoint != INVALID && size)
          {
            *out++ = codepoint;
            size--;
          }
          return initial_size - size;
        }
      }
      start_page = i;
      start_page_value = page_remainder (codepoint + 1);
      if (unlikely (start_page_value == 0))
      {
        // The export-after value was last in the page. Start on next page.
        start_page++;
        start_page_value = 0;
      }
    }

    hb_codepoint_t next_value = codepoint + 1;
    for (unsigned int i=start_page; i<page_map.length && size; i++)
    {
      uint32_t base = major_start (page_map.arrayZ[i].major);
      unsigned int n = pages[page_map.arrayZ[i].index].write_inverted (base, start_page_value, out, size, &next_value);
      out += n;
      size -= n;
      start_page_value = 0;
    }
    while (next_value < HB_SET_VALUE_INVALID && size) {
      *out++ = next_value++;
      size--;
    }
    return initial_size - size;
  }

  bool has_population () const { return population != UINT_MAX; }
  unsigned int get_population () const
  {
    if (has_population ())
      return population;

    unsigned int pop = 0;
    unsigned int count = pages.length;
    for (unsigned int i = 0; i < count; i++)
      pop += pages[i].get_population ();

    population = pop;
    return pop;
  }
  hb_codepoint_t get_min () const
  {
    unsigned count = pages.length;
    for (unsigned i = 0; i < count; i++)
    {
      const auto& map = page_map.arrayZ[i];
      const auto& page = pages.arrayZ[map.index];

      if (!page.is_empty ())
	return map.major * page_t::PAGE_BITS + page.get_min ();
    }
    return INVALID;
  }
  hb_codepoint_t get_max () const
  {
    unsigned count = pages.length;
    for (signed i = count - 1; i >= 0; i--)
    {
      const auto& map = page_map.arrayZ[(unsigned) i];
      const auto& page = pages.arrayZ[map.index];

      if (!page.is_empty ())
	return map.major * page_t::PAGE_BITS + page.get_max ();
    }
    return INVALID;
  }

  static constexpr hb_codepoint_t INVALID = page_t::INVALID;

  /*
   * Iterator implementation.
   */
  struct iter_t : hb_iter_with_fallback_t<iter_t, hb_codepoint_t>
  {
    static constexpr bool is_sorted_iterator = true;
    static constexpr bool has_fast_len = true;
    iter_t (const hb_bit_set_t &s_ = Null (hb_bit_set_t),
	    bool init = true) : s (&s_), v (INVALID), l(0)
    {
      if (init)
      {
	l = s->get_population () + 1;
	__next__ ();
      }
    }

    typedef hb_codepoint_t __item_t__;
    hb_codepoint_t __item__ () const { return v; }
    bool __more__ () const { return v != INVALID; }
    void __next__ () { s->next (&v); if (l) l--; }
    void __prev__ () { s->previous (&v); }
    unsigned __len__ () const { return l; }
    iter_t end () const { return iter_t (*s, false); }
    bool operator != (const iter_t& o) const
    { return s != o.s || v != o.v; }

    protected:
    const hb_bit_set_t *s;
    hb_codepoint_t v;
    unsigned l;
  };
  iter_t iter () const { return iter_t (*this); }
  operator iter_t () const { return iter (); }

  protected:

  page_t *page_for (hb_codepoint_t g, bool insert = false)
  {
    unsigned major = get_major (g);

    /* The extra page_map length is necessary; can't just rely on vector here,
     * since the next check would be tricked because a null page also has
     * major==0, which we can't distinguish from an actually major==0 page... */
    unsigned i = last_page_lookup;
    if (likely (i < page_map.length))
    {
      auto &cached_page = page_map.arrayZ[i];
      if (cached_page.major == major)
	return &pages.arrayZ[cached_page.index];
    }

    page_map_t map = {major, pages.length};
    if (!page_map.bfind (map, &i, HB_NOT_FOUND_STORE_CLOSEST))
    {
      if (!insert)
        return nullptr;

      if (unlikely (!resize (pages.length + 1)))
	return nullptr;

      pages.arrayZ[map.index].init0 ();
      memmove (page_map.arrayZ + i + 1,
	       page_map.arrayZ + i,
	       (page_map.length - 1 - i) * page_map.item_size);
      page_map.arrayZ[i] = map;
    }

    last_page_lookup = i;
    return &pages.arrayZ[page_map.arrayZ[i].index];
  }
  const page_t *page_for (hb_codepoint_t g) const
  {
    unsigned major = get_major (g);

    /* The extra page_map length is necessary; can't just rely on vector here,
     * since the next check would be tricked because a null page also has
     * major==0, which we can't distinguish from an actually major==0 page... */
    unsigned i = last_page_lookup;
    if (likely (i < page_map.length))
    {
      auto &cached_page = page_map.arrayZ[i];
      if (cached_page.major == major)
	return &pages.arrayZ[cached_page.index];
    }

    page_map_t key = {major};
    if (!page_map.bfind (key, &i))
      return nullptr;

    last_page_lookup = i;
    return &pages.arrayZ[page_map.arrayZ[i].index];
  }
  page_t &page_at (unsigned int i)
  {
    assert (i < page_map.length);
    return pages.arrayZ[page_map.arrayZ[i].index];
  }
  const page_t &page_at (unsigned int i) const
  {
    assert (i < page_map.length);
    return pages.arrayZ[page_map.arrayZ[i].index];
  }
  unsigned int get_major (hb_codepoint_t g) const { return g >> page_t::PAGE_BITS_LOG_2; }
  unsigned int page_remainder (hb_codepoint_t g) const { return g & page_t::PAGE_BITMASK; }
  hb_codepoint_t major_start (unsigned int major) const { return major << page_t::PAGE_BITS_LOG_2; }
};


#endif /* HB_BIT_SET_HH */
