/*
 * Copyright © 2012  Google, Inc.
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

#include "hb.hh"
#include "hb-shaper.hh"
#include "hb-machinery.hh"


static const hb_shaper_entry_t _hb_all_shapers[] = {
#define HB_SHAPER_IMPLEMENT(name) {#name, _hb_##name##_shape},
#include "hb-shaper-list.hh"
#undef HB_SHAPER_IMPLEMENT
};
#ifndef HB_NO_SHAPER
static_assert (0 != ARRAY_LENGTH_CONST (_hb_all_shapers), "No shaper enabled.");
#endif

static inline void free_static_shapers ();

static struct hb_shapers_lazy_loader_t : hb_lazy_loader_t<hb_shaper_entry_t,
							  hb_shapers_lazy_loader_t>
{
  static hb_shaper_entry_t *create ()
  {
    char *env = getenv ("HB_SHAPER_LIST");
    if (!env || !*env)
      return nullptr;

    hb_shaper_entry_t *shapers = (hb_shaper_entry_t *) hb_calloc (1, sizeof (_hb_all_shapers));
    if (unlikely (!shapers))
      return nullptr;

    hb_memcpy (shapers, _hb_all_shapers, sizeof (_hb_all_shapers));

     /* Reorder shaper list to prefer requested shapers. */
    unsigned int i = 0;
    char *end, *p = env;
    for (;;)
    {
      end = strchr (p, ',');
      if (!end)
	end = p + strlen (p);

      for (unsigned int j = i; j < ARRAY_LENGTH_CONST (_hb_all_shapers); j++)
	if (end - p == (int) strlen (shapers[j].name) &&
	    0 == strncmp (shapers[j].name, p, end - p))
	{
	  /* Reorder this shaper to position i */
	 struct hb_shaper_entry_t t = shapers[j];
	 memmove (&shapers[i + 1], &shapers[i], sizeof (shapers[i]) * (j - i));
	 shapers[i] = t;
	 i++;
	}

      if (!*end)
	break;
      else
	p = end + 1;
    }

    hb_atexit (free_static_shapers);

    return shapers;
  }
  static void destroy (hb_shaper_entry_t *p) { hb_free (p); }
  static const hb_shaper_entry_t *get_null ()      { return _hb_all_shapers; }
} static_shapers;

static inline
void free_static_shapers ()
{
  static_shapers.free_instance ();
}

const hb_shaper_entry_t *
_hb_shapers_get ()
{
  return static_shapers.get_unconst ();
}
