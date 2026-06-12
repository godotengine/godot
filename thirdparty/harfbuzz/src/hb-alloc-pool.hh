/*
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
 * Author(s): Behdad Esfahbod
 */

#ifndef HB_ALLOC_POOL_HH
#define HB_ALLOC_POOL_HH

#include "hb-vector.hh"

/* Memory pool for persistent small- to medium-sized allocations.
 *
 * Some AI musings on this, not necessarily true:
 *
 * This is a very simple implementation, but it's good enough for our
 * purposes.  It's not thread-safe.  It's not very fast.  It's not
 * very memory efficient.  It's not very cache efficient.  It's not
 * very anything efficient.  But it's simple and it works.  And it's
 * good enough for our purposes.  If you need something more
 * sophisticated, use a real allocator.  Or use a real language. */

struct hb_alloc_pool_t
{
  unsigned ChunkSize = 65536 - 2 * sizeof (void *);

  void *alloc (size_t size, unsigned alignment = 2 * sizeof (void *))
  {
    if (unlikely (chunks.in_error ())) return nullptr;

    assert (alignment > 0);
    assert (alignment <= 2 * sizeof (void *));
    assert ((alignment & (alignment - 1)) == 0); /* power of two */

    if (size > (ChunkSize) / 4)
    {
      /* Big chunk, allocate separately.  */
      hb_vector_t<char> chunk;
      if (unlikely (!chunk.resize (size))) return nullptr;
      void *ret = chunk.arrayZ;
      chunks.push (std::move (chunk));
      if (chunks.in_error ()) return nullptr;
      if (chunks.length > 1)
      {
        // Bring back the previous last chunk to the end, so that
	// we can continue to allocate from it.
	hb_swap (chunks.arrayZ[chunks.length - 1], chunks.arrayZ[chunks.length - 2]);
      }
      return ret;
    }

    unsigned pad = (unsigned) ((alignment - ((uintptr_t) current_chunk.arrayZ & (alignment - 1))) & (alignment - 1));

    // Small chunk, allocate from the last chunk.
    if (current_chunk.length < pad + size)
    {
      chunks.push ();
      if (unlikely (chunks.in_error ())) return nullptr;
      hb_vector_t<char> &chunk = chunks.arrayZ[chunks.length - 1];
      if (unlikely (!chunk.resize (ChunkSize))) return nullptr;
      current_chunk = chunk;
      pad = (unsigned) ((alignment - ((uintptr_t) current_chunk.arrayZ & (alignment - 1))) & (alignment - 1));
    }

    current_chunk += pad;

    assert (current_chunk.length >= size);
    void *ret = current_chunk.arrayZ;
    current_chunk += size;
    return ret;
  }

  void discard (void *p_, size_t size)
  {
    // Reclaim memory if we can.
    char *p = (char *) p_;
    if (current_chunk.arrayZ == p + size && current_chunk.backwards_length >= size)
      current_chunk -= size;
  }

  private:
  hb_vector_t<hb_vector_t<char>> chunks;
  hb_array_t<char> current_chunk;
};


#endif /* HB_ALLOC_POOL_HH */
