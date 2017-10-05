/*
 * Copyright © 2018  Google, Inc.
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
 * Google Author(s): Garret Rieger
 */

#ifndef HB_OT_HDMX_TABLE_HH
#define HB_OT_HDMX_TABLE_HH

#include "hb-open-type-private.hh"
#include "hb-subset-plan.hh"

/*
 * hdmx -- Horizontal Device Metrics
 * https://docs.microsoft.com/en-us/typography/opentype/spec/hdmx
 */
#define HB_OT_TAG_hdmx HB_TAG('h','d','m','x')


namespace OT {


struct DeviceRecord
{
  struct SubsetView
  {
    const DeviceRecord *source_device_record;
    unsigned int size_device_record;
    hb_subset_plan_t *subset_plan;

    inline void init(const DeviceRecord *source_device_record,
		     unsigned int size_device_record,
		     hb_subset_plan_t   *subset_plan)
    {
      this->source_device_record = source_device_record;
      this->size_device_record = size_device_record;
      this->subset_plan = subset_plan;
    }

    inline unsigned int len () const
    {
      return this->subset_plan->glyphs.len;
    }

    inline const HBUINT8* operator [] (unsigned int i) const
    {
      if (unlikely (i >= len())) return nullptr;
      hb_codepoint_t gid = this->subset_plan->glyphs [i];

      const HBUINT8* width = &(this->source_device_record->widths[gid]);

      if (width < ((const HBUINT8 *) this->source_device_record) + size_device_record)
	return width;
      else
	return nullptr;
    }
  };

  static inline unsigned int get_size (unsigned int count)
  {
    unsigned int raw_size = min_size + count * HBUINT8::static_size;
    if (raw_size % 4)
      /* Align to 32 bits */
      return raw_size + (4 - (raw_size % 4));
    return raw_size;
  }

  inline bool serialize (hb_serialize_context_t *c, const SubsetView &subset_view)
  {
    TRACE_SERIALIZE (this);

    if (unlikely (!c->allocate_size<DeviceRecord> (get_size (subset_view.len()))))
      return_trace (false);

    this->pixel_size.set (subset_view.source_device_record->pixel_size);
    this->max_width.set (subset_view.source_device_record->max_width);

    for (unsigned int i = 0; i < subset_view.len(); i++)
    {
      const HBUINT8 *width = subset_view[i];
      if (!width)
      {
	DEBUG_MSG(SUBSET, nullptr, "HDMX width for new gid %d is missing.", i);
	return_trace (false);
      }
      widths[i].set (*width);
    }

    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c, unsigned int size_device_record) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  c->check_range (this, size_device_record)));
  }

  HBUINT8 pixel_size;   /* Pixel size for following widths (as ppem). */
  HBUINT8 max_width;    /* Maximum width. */
  HBUINT8 widths[VAR];  /* Array of widths (numGlyphs is from the 'maxp' table). */
  public:
  DEFINE_SIZE_ARRAY (2, widths);
};


struct hdmx
{
  static const hb_tag_t tableTag = HB_OT_TAG_hdmx;

  inline unsigned int get_size (void) const
  {
    return min_size + num_records * size_device_record;
  }

  inline const DeviceRecord& operator [] (unsigned int i) const
  {
    if (unlikely (i >= num_records)) return Null(DeviceRecord);
    return StructAtOffset<DeviceRecord> (this->data, i * size_device_record);
  }

  inline bool serialize (hb_serialize_context_t *c, const hdmx *source_hdmx, hb_subset_plan_t *plan)
  {
    TRACE_SERIALIZE (this);

    if (unlikely (!c->extend_min ((*this))))  return_trace (false);

    this->version.set (source_hdmx->version);
    this->num_records.set (source_hdmx->num_records);
    this->size_device_record.set (DeviceRecord::get_size (plan->glyphs.len));

    for (unsigned int i = 0; i < source_hdmx->num_records; i++)
    {
      DeviceRecord::SubsetView subset_view;
      subset_view.init (&(*source_hdmx)[i], source_hdmx->size_device_record, plan);

      if (!c->start_embed<DeviceRecord> ()->serialize (c, subset_view))
	return_trace (false);
    }

    return_trace (true);
  }

  static inline size_t get_subsetted_size (hb_subset_plan_t *plan)
  {
    return min_size + DeviceRecord::get_size (plan->glyphs.len);
  }

  inline bool subset (hb_subset_plan_t *plan) const
  {
    size_t dest_size = get_subsetted_size (plan);
    hdmx *dest = (hdmx *) malloc (dest_size);
    if (unlikely (!dest))
    {
      DEBUG_MSG(SUBSET, nullptr, "Unable to alloc %lu for hdmx subset output.", (unsigned long) dest_size);
      return false;
    }

    hb_serialize_context_t c (dest, dest_size);
    hdmx *hdmx_prime = c.start_serialize<hdmx> ();
    if (!hdmx_prime || !hdmx_prime->serialize (&c, this, plan)) {
      free (dest);
      return false;
    }
    c.end_serialize ();

    hb_blob_t *hdmx_prime_blob = hb_blob_create ((const char *) dest,
						 dest_size,
						 HB_MEMORY_MODE_READONLY,
						 dest,
						 free);
    bool result = plan->add_table (HB_OT_TAG_hdmx, hdmx_prime_blob);
    hb_blob_destroy (hdmx_prime_blob);

    return result;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && version == 0 &&
		  !hb_unsigned_mul_overflows (num_records, size_device_record) &&
		  size_device_record >= DeviceRecord::min_size &&
		  c->check_range (this, get_size()));
  }

  protected:
  HBUINT16	version;		/* Table version number (0) */
  HBUINT16	num_records;		/* Number of device records. */
  HBUINT32	size_device_record;	/* Size of a device record, 32-bit aligned. */
  HBUINT8	data[VAR];		/* Array of device records. */
  public:
  DEFINE_SIZE_ARRAY (8, data);
};

} /* namespace OT */


#endif /* HB_OT_HDMX_TABLE_HH */
