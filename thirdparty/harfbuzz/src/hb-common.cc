/*
 * Copyright © 2009,2010  Red Hat, Inc.
 * Copyright © 2011,2012  Google, Inc.
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

#include "hb-private.hh"

#include "hb-machinery-private.hh"

#include <locale.h>
#ifdef HAVE_XLOCALE_H
#include <xlocale.h>
#endif


/* hb_options_t */

hb_atomic_int_t _hb_options;

void
_hb_options_init (void)
{
  hb_options_union_t u;
  u.i = 0;
  u.opts.initialized = 1;

  char *c = getenv ("HB_OPTIONS");
  u.opts.uniscribe_bug_compatible = c && strstr (c, "uniscribe-bug-compatible");

  /* This is idempotent and threadsafe. */
  _hb_options.set_relaxed (u.i);
}


/* hb_tag_t */

/**
 * hb_tag_from_string:
 * @str: (array length=len) (element-type uint8_t):
 * @len:
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_tag_t
hb_tag_from_string (const char *str, int len)
{
  char tag[4];
  unsigned int i;

  if (!str || !len || !*str)
    return HB_TAG_NONE;

  if (len < 0 || len > 4)
    len = 4;
  for (i = 0; i < (unsigned) len && str[i]; i++)
    tag[i] = str[i];
  for (; i < 4; i++)
    tag[i] = ' ';

  return HB_TAG (tag[0], tag[1], tag[2], tag[3]);
}

/**
 * hb_tag_to_string:
 * @tag:
 * @buf: (out caller-allocates) (array fixed-size=4) (element-type uint8_t):
 *
 *
 *
 * Since: 0.9.5
 **/
void
hb_tag_to_string (hb_tag_t tag, char *buf)
{
  buf[0] = (char) (uint8_t) (tag >> 24);
  buf[1] = (char) (uint8_t) (tag >> 16);
  buf[2] = (char) (uint8_t) (tag >>  8);
  buf[3] = (char) (uint8_t) (tag >>  0);
}


/* hb_direction_t */

const char direction_strings[][4] = {
  "ltr",
  "rtl",
  "ttb",
  "btt"
};

/**
 * hb_direction_from_string:
 * @str: (array length=len) (element-type uint8_t):
 * @len:
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_direction_t
hb_direction_from_string (const char *str, int len)
{
  if (unlikely (!str || !len || !*str))
    return HB_DIRECTION_INVALID;

  /* Lets match loosely: just match the first letter, such that
   * all of "ltr", "left-to-right", etc work!
   */
  char c = TOLOWER (str[0]);
  for (unsigned int i = 0; i < ARRAY_LENGTH (direction_strings); i++)
    if (c == direction_strings[i][0])
      return (hb_direction_t) (HB_DIRECTION_LTR + i);

  return HB_DIRECTION_INVALID;
}

/**
 * hb_direction_to_string:
 * @direction:
 *
 *
 *
 * Return value: (transfer none):
 *
 * Since: 0.9.2
 **/
const char *
hb_direction_to_string (hb_direction_t direction)
{
  if (likely ((unsigned int) (direction - HB_DIRECTION_LTR)
	      < ARRAY_LENGTH (direction_strings)))
    return direction_strings[direction - HB_DIRECTION_LTR];

  return "invalid";
}


/* hb_language_t */

struct hb_language_impl_t {
  const char s[1];
};

static const char canon_map[256] = {
   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,  '-',  0,   0,
  '0', '1', '2', '3', '4', '5', '6', '7',  '8', '9',  0,   0,   0,   0,   0,   0,
  '-', 'a', 'b', 'c', 'd', 'e', 'f', 'g',  'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
  'p', 'q', 'r', 's', 't', 'u', 'v', 'w',  'x', 'y', 'z',  0,   0,   0,   0,  '-',
   0,  'a', 'b', 'c', 'd', 'e', 'f', 'g',  'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
  'p', 'q', 'r', 's', 't', 'u', 'v', 'w',  'x', 'y', 'z',  0,   0,   0,   0,   0
};

static bool
lang_equal (hb_language_t  v1,
	    const void    *v2)
{
  const unsigned char *p1 = (const unsigned char *) v1;
  const unsigned char *p2 = (const unsigned char *) v2;

  while (*p1 && *p1 == canon_map[*p2]) {
    p1++;
    p2++;
  }

  return *p1 == canon_map[*p2];
}

#if 0
static unsigned int
lang_hash (const void *key)
{
  const unsigned char *p = key;
  unsigned int h = 0;
  while (canon_map[*p])
    {
      h = (h << 5) - h + canon_map[*p];
      p++;
    }

  return h;
}
#endif


struct hb_language_item_t {

  struct hb_language_item_t *next;
  hb_language_t lang;

  inline bool operator == (const char *s) const {
    return lang_equal (lang, s);
  }

  inline hb_language_item_t & operator = (const char *s) {
    /* If a custom allocated is used calling strdup() pairs
    badly with a call to the custom free() in fini() below.
    Therefore don't call strdup(), implement its behavior.
    */
    size_t len = strlen(s) + 1;
    lang = (hb_language_t) malloc(len);
    if (likely (lang))
    {
      memcpy((unsigned char *) lang, s, len);
      for (unsigned char *p = (unsigned char *) lang; *p; p++)
	*p = canon_map[*p];
    }

    return *this;
  }

  void fini (void) { free ((void *) lang); }
};


/* Thread-safe lock-free language list */

static hb_atomic_ptr_t <hb_language_item_t> langs;

#ifdef HB_USE_ATEXIT
static void
free_langs (void)
{
retry:
  hb_language_item_t *first_lang = langs.get ();
  if (unlikely (!langs.cmpexch (first_lang, nullptr)))
    goto retry;

  while (first_lang) {
    hb_language_item_t *next = first_lang->next;
    first_lang->fini ();
    free (first_lang);
    first_lang = next;
  }
}
#endif

static hb_language_item_t *
lang_find_or_insert (const char *key)
{
retry:
  hb_language_item_t *first_lang = langs.get ();

  for (hb_language_item_t *lang = first_lang; lang; lang = lang->next)
    if (*lang == key)
      return lang;

  /* Not found; allocate one. */
  hb_language_item_t *lang = (hb_language_item_t *) calloc (1, sizeof (hb_language_item_t));
  if (unlikely (!lang))
    return nullptr;
  lang->next = first_lang;
  *lang = key;
  if (unlikely (!lang->lang))
  {
    free (lang);
    return nullptr;
  }

  if (unlikely (!langs.cmpexch (first_lang, lang)))
  {
    lang->fini ();
    free (lang);
    goto retry;
  }

#ifdef HB_USE_ATEXIT
  if (!first_lang)
    atexit (free_langs); /* First person registers atexit() callback. */
#endif

  return lang;
}


/**
 * hb_language_from_string:
 * @str: (array length=len) (element-type uint8_t): a string representing
 *       ISO 639 language code
 * @len: length of the @str, or -1 if it is %NULL-terminated.
 *
 * Converts @str representing an ISO 639 language code to the corresponding
 * #hb_language_t.
 *
 * Return value: (transfer none):
 * The #hb_language_t corresponding to the ISO 639 language code.
 *
 * Since: 0.9.2
 **/
hb_language_t
hb_language_from_string (const char *str, int len)
{
  if (!str || !len || !*str)
    return HB_LANGUAGE_INVALID;

  hb_language_item_t *item = nullptr;
  if (len >= 0)
  {
    /* NUL-terminate it. */
    char strbuf[64];
    len = MIN (len, (int) sizeof (strbuf) - 1);
    memcpy (strbuf, str, len);
    strbuf[len] = '\0';
    item = lang_find_or_insert (strbuf);
  }
  else
    item = lang_find_or_insert (str);

  return likely (item) ? item->lang : HB_LANGUAGE_INVALID;
}

/**
 * hb_language_to_string:
 * @language: an #hb_language_t to convert.
 *
 * See hb_language_from_string().
 *
 * Return value: (transfer none):
 * A %NULL-terminated string representing the @language. Must not be freed by
 * the caller.
 *
 * Since: 0.9.2
 **/
const char *
hb_language_to_string (hb_language_t language)
{
  /* This is actually nullptr-safe! */
  return language->s;
}

/**
 * hb_language_get_default:
 *
 *
 *
 * Return value: (transfer none):
 *
 * Since: 0.9.2
 **/
hb_language_t
hb_language_get_default (void)
{
  static hb_atomic_ptr_t <hb_language_t> default_language;

  hb_language_t language = default_language.get ();
  if (unlikely (language == HB_LANGUAGE_INVALID))
  {
    language = hb_language_from_string (setlocale (LC_CTYPE, nullptr), -1);
    (void) default_language.cmpexch (HB_LANGUAGE_INVALID, language);
  }

  return language;
}


/* hb_script_t */

/**
 * hb_script_from_iso15924_tag:
 * @tag: an #hb_tag_t representing an ISO 15924 tag.
 *
 * Converts an ISO 15924 script tag to a corresponding #hb_script_t.
 *
 * Return value:
 * An #hb_script_t corresponding to the ISO 15924 tag.
 *
 * Since: 0.9.2
 **/
hb_script_t
hb_script_from_iso15924_tag (hb_tag_t tag)
{
  if (unlikely (tag == HB_TAG_NONE))
    return HB_SCRIPT_INVALID;

  /* Be lenient, adjust case (one capital letter followed by three small letters) */
  tag = (tag & 0xDFDFDFDFu) | 0x00202020u;

  switch (tag) {

    /* These graduated from the 'Q' private-area codes, but
     * the old code is still aliased by Unicode, and the Qaai
     * one in use by ICU. */
    case HB_TAG('Q','a','a','i'): return HB_SCRIPT_INHERITED;
    case HB_TAG('Q','a','a','c'): return HB_SCRIPT_COPTIC;

    /* Script variants from https://unicode.org/iso15924/ */
    case HB_TAG('C','y','r','s'): return HB_SCRIPT_CYRILLIC;
    case HB_TAG('L','a','t','f'): return HB_SCRIPT_LATIN;
    case HB_TAG('L','a','t','g'): return HB_SCRIPT_LATIN;
    case HB_TAG('S','y','r','e'): return HB_SCRIPT_SYRIAC;
    case HB_TAG('S','y','r','j'): return HB_SCRIPT_SYRIAC;
    case HB_TAG('S','y','r','n'): return HB_SCRIPT_SYRIAC;
  }

  /* If it looks right, just use the tag as a script */
  if (((uint32_t) tag & 0xE0E0E0E0u) == 0x40606060u)
    return (hb_script_t) tag;

  /* Otherwise, return unknown */
  return HB_SCRIPT_UNKNOWN;
}

/**
 * hb_script_from_string:
 * @str: (array length=len) (element-type uint8_t): a string representing an
 *       ISO 15924 tag.
 * @len: length of the @str, or -1 if it is %NULL-terminated.
 *
 * Converts a string @str representing an ISO 15924 script tag to a
 * corresponding #hb_script_t. Shorthand for hb_tag_from_string() then
 * hb_script_from_iso15924_tag().
 *
 * Return value:
 * An #hb_script_t corresponding to the ISO 15924 tag.
 *
 * Since: 0.9.2
 **/
hb_script_t
hb_script_from_string (const char *str, int len)
{
  return hb_script_from_iso15924_tag (hb_tag_from_string (str, len));
}

/**
 * hb_script_to_iso15924_tag:
 * @script: an #hb_script_ to convert.
 *
 * See hb_script_from_iso15924_tag().
 *
 * Return value:
 * An #hb_tag_t representing an ISO 15924 script tag.
 *
 * Since: 0.9.2
 **/
hb_tag_t
hb_script_to_iso15924_tag (hb_script_t script)
{
  return (hb_tag_t) script;
}

/**
 * hb_script_get_horizontal_direction:
 * @script:
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_direction_t
hb_script_get_horizontal_direction (hb_script_t script)
{
  /* https://docs.google.com/spreadsheets/d/1Y90M0Ie3MUJ6UVCRDOypOtijlMDLNNyyLk36T6iMu0o */
  switch ((hb_tag_t) script)
  {
    /* Unicode-1.1 additions */
    case HB_SCRIPT_ARABIC:
    case HB_SCRIPT_HEBREW:

    /* Unicode-3.0 additions */
    case HB_SCRIPT_SYRIAC:
    case HB_SCRIPT_THAANA:

    /* Unicode-4.0 additions */
    case HB_SCRIPT_CYPRIOT:

    /* Unicode-4.1 additions */
    case HB_SCRIPT_KHAROSHTHI:

    /* Unicode-5.0 additions */
    case HB_SCRIPT_PHOENICIAN:
    case HB_SCRIPT_NKO:

    /* Unicode-5.1 additions */
    case HB_SCRIPT_LYDIAN:

    /* Unicode-5.2 additions */
    case HB_SCRIPT_AVESTAN:
    case HB_SCRIPT_IMPERIAL_ARAMAIC:
    case HB_SCRIPT_INSCRIPTIONAL_PAHLAVI:
    case HB_SCRIPT_INSCRIPTIONAL_PARTHIAN:
    case HB_SCRIPT_OLD_SOUTH_ARABIAN:
    case HB_SCRIPT_OLD_TURKIC:
    case HB_SCRIPT_SAMARITAN:

    /* Unicode-6.0 additions */
    case HB_SCRIPT_MANDAIC:

    /* Unicode-6.1 additions */
    case HB_SCRIPT_MEROITIC_CURSIVE:
    case HB_SCRIPT_MEROITIC_HIEROGLYPHS:

    /* Unicode-7.0 additions */
    case HB_SCRIPT_MANICHAEAN:
    case HB_SCRIPT_MENDE_KIKAKUI:
    case HB_SCRIPT_NABATAEAN:
    case HB_SCRIPT_OLD_NORTH_ARABIAN:
    case HB_SCRIPT_PALMYRENE:
    case HB_SCRIPT_PSALTER_PAHLAVI:

    /* Unicode-8.0 additions */
    case HB_SCRIPT_HATRAN:
    case HB_SCRIPT_OLD_HUNGARIAN:

    /* Unicode-9.0 additions */
    case HB_SCRIPT_ADLAM:

    /* Unicode-11.0 additions */
    case HB_SCRIPT_HANIFI_ROHINGYA:
    case HB_SCRIPT_OLD_SOGDIAN:
    case HB_SCRIPT_SOGDIAN:

      return HB_DIRECTION_RTL;


    /* https://github.com/harfbuzz/harfbuzz/issues/1000 */
    case HB_SCRIPT_OLD_ITALIC:
    case HB_SCRIPT_RUNIC:

      return HB_DIRECTION_INVALID;
  }

  return HB_DIRECTION_LTR;
}


/* hb_user_data_array_t */

bool
hb_user_data_array_t::set (hb_user_data_key_t *key,
			   void *              data,
			   hb_destroy_func_t   destroy,
			   hb_bool_t           replace)
{
  if (!key)
    return false;

  if (replace) {
    if (!data && !destroy) {
      items.remove (key, lock);
      return true;
    }
  }
  hb_user_data_item_t item = {key, data, destroy};
  bool ret = !!items.replace_or_insert (item, lock, (bool) replace);

  return ret;
}

void *
hb_user_data_array_t::get (hb_user_data_key_t *key)
{
  hb_user_data_item_t item = {nullptr, nullptr, nullptr};

  return items.find (key, &item, lock) ? item.data : nullptr;
}


/* hb_version */

/**
 * hb_version:
 * @major: (out): Library major version component.
 * @minor: (out): Library minor version component.
 * @micro: (out): Library micro version component.
 *
 * Returns library version as three integer components.
 *
 * Since: 0.9.2
 **/
void
hb_version (unsigned int *major,
	    unsigned int *minor,
	    unsigned int *micro)
{
  *major = HB_VERSION_MAJOR;
  *minor = HB_VERSION_MINOR;
  *micro = HB_VERSION_MICRO;
}

/**
 * hb_version_string:
 *
 * Returns library version as a string with three components.
 *
 * Return value: library version string.
 *
 * Since: 0.9.2
 **/
const char *
hb_version_string (void)
{
  return HB_VERSION_STRING;
}

/**
 * hb_version_atleast:
 * @major:
 * @minor:
 * @micro:
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.30
 **/
hb_bool_t
hb_version_atleast (unsigned int major,
		    unsigned int minor,
		    unsigned int micro)
{
  return HB_VERSION_ATLEAST (major, minor, micro);
}



/* hb_feature_t and hb_variation_t */

static bool
parse_space (const char **pp, const char *end)
{
  while (*pp < end && ISSPACE (**pp))
    (*pp)++;
  return true;
}

static bool
parse_char (const char **pp, const char *end, char c)
{
  parse_space (pp, end);

  if (*pp == end || **pp != c)
    return false;

  (*pp)++;
  return true;
}

static bool
parse_uint (const char **pp, const char *end, unsigned int *pv)
{
  char buf[32];
  unsigned int len = MIN (ARRAY_LENGTH (buf) - 1, (unsigned int) (end - *pp));
  strncpy (buf, *pp, len);
  buf[len] = '\0';

  char *p = buf;
  char *pend = p;
  unsigned int v;

  /* Intentionally use strtol instead of strtoul, such that
   * -1 turns into "big number"... */
  errno = 0;
  v = strtol (p, &pend, 0);
  if (errno || p == pend)
    return false;

  *pv = v;
  *pp += pend - p;
  return true;
}

static bool
parse_uint32 (const char **pp, const char *end, uint32_t *pv)
{
  char buf[32];
  unsigned int len = MIN (ARRAY_LENGTH (buf) - 1, (unsigned int) (end - *pp));
  strncpy (buf, *pp, len);
  buf[len] = '\0';

  char *p = buf;
  char *pend = p;
  unsigned int v;

  /* Intentionally use strtol instead of strtoul, such that
   * -1 turns into "big number"... */
  errno = 0;
  v = strtol (p, &pend, 0);
  if (errno || p == pend)
    return false;

  *pv = v;
  *pp += pend - p;
  return true;
}

#if defined (HAVE_NEWLOCALE) && defined (HAVE_STRTOD_L)
#define USE_XLOCALE 1
#define HB_LOCALE_T locale_t
#define HB_CREATE_LOCALE(locName) newlocale (LC_ALL_MASK, locName, nullptr)
#define HB_FREE_LOCALE(loc) freelocale (loc)
#elif defined(_MSC_VER)
#define USE_XLOCALE 1
#define HB_LOCALE_T _locale_t
#define HB_CREATE_LOCALE(locName) _create_locale (LC_ALL, locName)
#define HB_FREE_LOCALE(loc) _free_locale (loc)
#define strtod_l(a, b, c) _strtod_l ((a), (b), (c))
#endif

#ifdef USE_XLOCALE


static void free_static_C_locale (void);

static struct hb_C_locale_lazy_loader_t : hb_lazy_loader_t<hb_remove_ptr_t<HB_LOCALE_T>::value,
							  hb_C_locale_lazy_loader_t>
{
  static inline HB_LOCALE_T create (void)
  {
    HB_LOCALE_T C_locale = HB_CREATE_LOCALE ("C");

#ifdef HB_USE_ATEXIT
    atexit (free_static_C_locale);
#endif

    return C_locale;
  }
  static inline void destroy (HB_LOCALE_T p)
  {
    HB_FREE_LOCALE (p);
  }
  static inline HB_LOCALE_T get_null (void)
  {
    return nullptr;
  }
} static_C_locale;

#ifdef HB_USE_ATEXIT
static
void free_static_C_locale (void)
{
  static_C_locale.free_instance ();
}
#endif

static HB_LOCALE_T
get_C_locale (void)
{
  return static_C_locale.get_unconst ();
}
#endif /* USE_XLOCALE */

static bool
parse_float (const char **pp, const char *end, float *pv)
{
  char buf[32];
  unsigned int len = MIN (ARRAY_LENGTH (buf) - 1, (unsigned int) (end - *pp));
  strncpy (buf, *pp, len);
  buf[len] = '\0';

  char *p = buf;
  char *pend = p;
  float v;

  errno = 0;
#ifdef USE_XLOCALE
  v = strtod_l (p, &pend, get_C_locale ());
#else
  v = strtod (p, &pend);
#endif
  if (errno || p == pend)
    return false;

  *pv = v;
  *pp += pend - p;
  return true;
}

static bool
parse_bool (const char **pp, const char *end, uint32_t *pv)
{
  parse_space (pp, end);

  const char *p = *pp;
  while (*pp < end && ISALPHA(**pp))
    (*pp)++;

  /* CSS allows on/off as aliases 1/0. */
  if (*pp - p == 2 && 0 == strncmp (p, "on", 2))
    *pv = 1;
  else if (*pp - p == 3 && 0 == strncmp (p, "off", 3))
    *pv = 0;
  else
    return false;

  return true;
}

/* hb_feature_t */

static bool
parse_feature_value_prefix (const char **pp, const char *end, hb_feature_t *feature)
{
  if (parse_char (pp, end, '-'))
    feature->value = 0;
  else {
    parse_char (pp, end, '+');
    feature->value = 1;
  }

  return true;
}

static bool
parse_tag (const char **pp, const char *end, hb_tag_t *tag)
{
  parse_space (pp, end);

  char quote = 0;

  if (*pp < end && (**pp == '\'' || **pp == '"'))
  {
    quote = **pp;
    (*pp)++;
  }

  const char *p = *pp;
  while (*pp < end && (ISALNUM(**pp) || **pp == '_'))
    (*pp)++;

  if (p == *pp || *pp - p > 4)
    return false;

  *tag = hb_tag_from_string (p, *pp - p);

  if (quote)
  {
    /* CSS expects exactly four bytes.  And we only allow quotations for
     * CSS compatibility.  So, enforce the length. */
     if (*pp - p != 4)
       return false;
    if (*pp == end || **pp != quote)
      return false;
    (*pp)++;
  }

  return true;
}

static bool
parse_feature_indices (const char **pp, const char *end, hb_feature_t *feature)
{
  parse_space (pp, end);

  bool has_start;

  feature->start = 0;
  feature->end = (unsigned int) -1;

  if (!parse_char (pp, end, '['))
    return true;

  has_start = parse_uint (pp, end, &feature->start);

  if (parse_char (pp, end, ':')) {
    parse_uint (pp, end, &feature->end);
  } else {
    if (has_start)
      feature->end = feature->start + 1;
  }

  return parse_char (pp, end, ']');
}

static bool
parse_feature_value_postfix (const char **pp, const char *end, hb_feature_t *feature)
{
  bool had_equal = parse_char (pp, end, '=');
  bool had_value = parse_uint32 (pp, end, &feature->value) ||
                   parse_bool (pp, end, &feature->value);
  /* CSS doesn't use equal-sign between tag and value.
   * If there was an equal-sign, then there *must* be a value.
   * A value without an equal-sign is ok, but not required. */
  return !had_equal || had_value;
}

static bool
parse_one_feature (const char **pp, const char *end, hb_feature_t *feature)
{
  return parse_feature_value_prefix (pp, end, feature) &&
	 parse_tag (pp, end, &feature->tag) &&
	 parse_feature_indices (pp, end, feature) &&
	 parse_feature_value_postfix (pp, end, feature) &&
	 parse_space (pp, end) &&
	 *pp == end;
}

/**
 * hb_feature_from_string:
 * @str: (array length=len) (element-type uint8_t): a string to parse
 * @len: length of @str, or -1 if string is %NULL terminated
 * @feature: (out): the #hb_feature_t to initialize with the parsed values
 *
 * Parses a string into a #hb_feature_t.
 *
 * TODO: document the syntax here.
 *
 * Return value:
 * %true if @str is successfully parsed, %false otherwise.
 *
 * Since: 0.9.5
 **/
hb_bool_t
hb_feature_from_string (const char *str, int len,
			hb_feature_t *feature)
{
  hb_feature_t feat;

  if (len < 0)
    len = strlen (str);

  if (likely (parse_one_feature (&str, str + len, &feat)))
  {
    if (feature)
      *feature = feat;
    return true;
  }

  if (feature)
    memset (feature, 0, sizeof (*feature));
  return false;
}

/**
 * hb_feature_to_string:
 * @feature: an #hb_feature_t to convert
 * @buf: (array length=size) (out): output string
 * @size: the allocated size of @buf
 *
 * Converts a #hb_feature_t into a %NULL-terminated string in the format
 * understood by hb_feature_from_string(). The client in responsible for
 * allocating big enough size for @buf, 128 bytes is more than enough.
 *
 * Since: 0.9.5
 **/
void
hb_feature_to_string (hb_feature_t *feature,
		      char *buf, unsigned int size)
{
  if (unlikely (!size)) return;

  char s[128];
  unsigned int len = 0;
  if (feature->value == 0)
    s[len++] = '-';
  hb_tag_to_string (feature->tag, s + len);
  len += 4;
  while (len && s[len - 1] == ' ')
    len--;
  if (feature->start != 0 || feature->end != (unsigned int) -1)
  {
    s[len++] = '[';
    if (feature->start)
      len += MAX (0, snprintf (s + len, ARRAY_LENGTH (s) - len, "%u", feature->start));
    if (feature->end != feature->start + 1) {
      s[len++] = ':';
      if (feature->end != (unsigned int) -1)
	len += MAX (0, snprintf (s + len, ARRAY_LENGTH (s) - len, "%u", feature->end));
    }
    s[len++] = ']';
  }
  if (feature->value > 1)
  {
    s[len++] = '=';
    len += MAX (0, snprintf (s + len, ARRAY_LENGTH (s) - len, "%u", feature->value));
  }
  assert (len < ARRAY_LENGTH (s));
  len = MIN (len, size - 1);
  memcpy (buf, s, len);
  buf[len] = '\0';
}

/* hb_variation_t */

static bool
parse_variation_value (const char **pp, const char *end, hb_variation_t *variation)
{
  parse_char (pp, end, '='); /* Optional. */
  return parse_float (pp, end, &variation->value);
}

static bool
parse_one_variation (const char **pp, const char *end, hb_variation_t *variation)
{
  return parse_tag (pp, end, &variation->tag) &&
	 parse_variation_value (pp, end, variation) &&
	 parse_space (pp, end) &&
	 *pp == end;
}

/**
 * hb_variation_from_string:
 *
 * Since: 1.4.2
 */
hb_bool_t
hb_variation_from_string (const char *str, int len,
			  hb_variation_t *variation)
{
  hb_variation_t var;

  if (len < 0)
    len = strlen (str);

  if (likely (parse_one_variation (&str, str + len, &var)))
  {
    if (variation)
      *variation = var;
    return true;
  }

  if (variation)
    memset (variation, 0, sizeof (*variation));
  return false;
}

/**
 * hb_variation_to_string:
 *
 * Since: 1.4.2
 */
void
hb_variation_to_string (hb_variation_t *variation,
			char *buf, unsigned int size)
{
  if (unlikely (!size)) return;

  char s[128];
  unsigned int len = 0;
  hb_tag_to_string (variation->tag, s + len);
  len += 4;
  while (len && s[len - 1] == ' ')
    len--;
  s[len++] = '=';
  len += MAX (0, snprintf (s + len, ARRAY_LENGTH (s) - len, "%g", variation->value));

  assert (len < ARRAY_LENGTH (s));
  len = MIN (len, size - 1);
  memcpy (buf, s, len);
  buf[len] = '\0';
}

/* If there is no visibility control, then hb-static.cc will NOT
 * define anything.  Instead, we get it to define one set in here
 * only, so only libharfbuzz.so defines them, not other libs. */
#ifdef HB_NO_VISIBILITY
#undef HB_NO_VISIBILITY
#include "hb-static.cc"
#define HB_NO_VISIBILITY 1
#endif
