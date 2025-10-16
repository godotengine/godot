#ifndef _BSDIFF_PATCH_READER_INTERFACE_H_
#define _BSDIFF_PATCH_READER_INTERFACE_H_

#include <stddef.h>

#include "bsdiff/control_entry.h"

namespace bsdiff {

class PatchReaderInterface {
 public:
  virtual ~PatchReaderInterface() = default;

  // Initialize the control stream, diff stream and extra stream from the
  // corresponding offset of |patch_data|.
  virtual bool Init(const uint8_t* patch_data, size_t patch_size) = 0;

  // Read the control stream and parse the metadata of |diff_size_|,
  // |extra_size_| and |offset_incremental_|.
  virtual bool ParseControlEntry(ControlEntry* control_entry) = 0;

  // Read the data in |diff_stream_| and write |size| decompressed data to
  // |buf|.
  virtual bool ReadDiffStream(uint8_t* buf, size_t size) = 0;

  // Read the data in |extra_stream_| and write |size| decompressed data to
  // |buf|.
  virtual bool ReadExtraStream(uint8_t* buf, size_t size) = 0;

  // Returns the new file size as read from the header.
  virtual uint64_t new_file_size() const = 0;

  // Close the control/diff/extra stream. Return false if errors occur when
  // closing any of these streams.
  virtual bool Finish() = 0;

 protected:
  PatchReaderInterface() = default;
};

}  // namespace bsdiff

#endif  // _BSDIFF_PATCH_WRITER_INTERFACE_H_
