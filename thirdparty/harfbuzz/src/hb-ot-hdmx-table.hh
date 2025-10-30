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

#include "hb-open-type.hh"

/*
 * hdmx -- Horizontal Device Metrics
 * https://docs.microsoft.com/en-us/typography/opentype/spec/hdmx
 */
#define HB_OT_TAG_hdmx HB_TAG('h','d','m','x')


namespace OT {


struct DeviceRecord
{
  static unsigned int get_size (unsigned count)
  { return hb_ceil_to_4 (min_size + count * HBUINT8::static_size); }

  template<typename Iterator,
	   hb_requires (hb_is_iterator (Iterator))>
  bool serialize (hb_serialize_context_t *c,
		  unsigned pixelSize,
		  Iterator it,
		  const hb_vector_t<hb_codepoint_pair_t> new_to_old_gid_list,
		  unsigned num_glyphs)
  {
    TRACE_SERIALIZE (this);

    if (unlikely (!c->extend (this, num_glyphs)))  return_trace (false);

    this->pixelSize = pixelSize;
    this->maxWidth =
    + it
    | hb_reduce (hb_max, 0u);

    for (auto &_ : new_to_old_gid_list)
      widthsZ[_.first] = *it++;

    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c, unsigned sizeDeviceRecord) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  hb_barrier () &&
			  c->check_range (this, sizeDeviceRecord)));
  }

  HBUINT8			pixelSize;	/* Pixel size for following widths (as ppem). */
  HBUINT8			maxWidth;	/* Maximum width. */
  UnsizedArrayOf<HBUINT8>	widthsZ;	/* Array of widths (numGlyphs is from the 'maxp' table). */
  public:
  DEFINE_SIZE_UNBOUNDED (2);
};


struct hdmx
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_hdmx;

  unsigned int get_size () const
  { return min_size + numRecords * sizeDeviceRecord; }

  template<typename Iterator,
	   hb_requires (hb_is_iterator (Iterator))>
  bool serialize (hb_serialize_context_t *c,
		  unsigned version,
		  Iterator it,
		  hb_array_t<const hb_codepoint_pair_t> new_to_old_gid_list,
		  unsigned num_glyphs)
  {
    TRACE_SERIALIZE (this);

    if (unlikely (!c->extend_min ((*this))))  return_trace (false);

    this->version = version;
    this->numRecords = it.len ();
    this->sizeDeviceRecord = DeviceRecord::get_size (num_glyphs);

    for (const hb_item_type<Iterator>& _ : +it)
      c->start_embed<DeviceRecord> ()->serialize (c, _.first, _.second, new_to_old_gid_list, num_glyphs);

    return_trace (c->successful ());
  }


  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);

    auto *hdmx_prime = c->serializer->start_embed <hdmx> ();

    unsigned num_input_glyphs = get_num_glyphs ();
    auto it =
    + hb_range ((unsigned) numRecords)
    | hb_map ([c, num_input_glyphs, this] (unsigned _)
	{
	  const DeviceRecord *device_record =
	    &StructAtOffset<DeviceRecord> (&firstDeviceRecord,
					   _ * sizeDeviceRecord);
	  auto row =
	    + hb_iter (c->plan->new_to_old_gid_list)
	    | hb_map ([num_input_glyphs, device_record] (hb_codepoint_pair_t _)
		      {
			return device_record->widthsZ.as_array (num_input_glyphs) [_.second];
		      })
	    ;
	  return hb_pair ((unsigned) device_record->pixelSize, +row);
	})
    ;

    hdmx_prime->serialize (c->serializer, version, it,
			   c->plan->new_to_old_gid_list,
			   c->plan->num_output_glyphs ());
    return_trace (true);
  }

  unsigned get_num_glyphs () const
  {
    return sizeDeviceRecord - DeviceRecord::min_size;
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  hb_barrier () &&
		  !hb_unsigned_mul_overflows (numRecords, sizeDeviceRecord) &&
                  min_size + numRecords * sizeDeviceRecord > numRecords * sizeDeviceRecord &&
		  sizeDeviceRecord >= DeviceRecord::min_size &&
		  c->check_range (this, get_size ()));
  }

  protected:
  HBUINT16	version;	/* Table version number (0) */
  HBUINT16	numRecords;	/* Number of device records. */
  HBUINT32	sizeDeviceRecord;
				/* Size of a device record, 32-bit aligned. */
  DeviceRecord	firstDeviceRecord;
				/* Array of device records. */
  public:
  DEFINE_SIZE_MIN (8);
};

} /* namespace OT */


#endif /* HB_OT_HDMX_TABLE_HH */
