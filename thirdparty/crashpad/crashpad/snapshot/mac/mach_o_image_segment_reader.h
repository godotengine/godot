// Copyright 2014 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_SEGMENT_READER_H_
#define CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_SEGMENT_READER_H_

#include <mach/mach.h>
#include <stdint.h>
#include <sys/types.h>

#include <map>
#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/mac/process_types.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

//! \brief Determines whether a module appears to be a malformed OpenCL
//!     `cl_kernels` module based on its name and Mach-O file type.
//!
//! `cl_kernels` modules require special handling because they’re malformed on
//! OS X 10.10 and later. A `cl_kernels` module always has Mach-O type
//! `MH_BUNDLE` and is named `"cl_kernels"` until macOS 10.14, and
//! `"/private/var/db/CVMS/cvmsCodeSignObj"` plus 16 random characters on macOS
//! 10.14.
//!
//! Malformed `cl_kernels` modules have a single `__TEXT` segment, but one of
//! the sections within it claims to belong to the `__LD` segment. This mismatch
//! shouldn’t happen. This errant section also has the `S_ATTR_DEBUG` flag set,
//! which shouldn’t happen unless all of the other sections in the segment also
//! have this bit set (they don’t). These odd sections are reminiscent of unwind
//! information stored in `MH_OBJECT` images, although `cl_kernels` images claim
//! to be `MH_BUNDLE`.
//!
//! This function is exposed for testing purposes only.
//!
//! \param[in] mach_o_file_type The Mach-O type of the module being examined.
//! \param[in] module_name The pathname that `dyld` reported having loaded the
//!     module from.
//! \param[out] has_timestamp Optional, may be `nullptr`. If provided, and the
//!     module is a maformed `cl_kernels` module, this will be set to `true` if
//!     the module was loaded from the filesystem (as is the case when loaded
//!     from the CVMS directory) and is expected to have a timestamp, and
//!     `false` otherwise. Note that even when loaded from the filesystem, these
//!     modules are unlinked from the filesystem after loading.
//!
//! \return `true` if the module appears to be a malformed `cl_kernels` module
//!     based on the provided information, `false` otherwise.
bool IsMalformedCLKernelsModule(uint32_t mach_o_file_type,
                                const std::string& module_name,
                                bool* has_timestamp);

//! \brief A reader for `LC_SEGMENT` or `LC_SEGMENT_64` load commands in Mach-O
//!     images mapped into another process.
//!
//! This class is capable of reading both `LC_SEGMENT` and `LC_SEGMENT_64` based
//! on the bitness of the remote process.
//!
//! A MachOImageSegmentReader will normally be instantiated by a
//! MachOImageReader.
class MachOImageSegmentReader {
 public:
  MachOImageSegmentReader();
  ~MachOImageSegmentReader();

  //! \brief Reads the segment load command from another process.
  //!
  //! This method must only be called once on an object. This method must be
  //! called successfully before any other method in this class may be called.
  //!
  //! \param[in] process_reader The reader for the remote process.
  //! \param[in] load_command_address The address, in the remote process’
  //!     address space, where the `LC_SEGMENT` or `LC_SEGMENT_64` load command
  //!     to be read is located. This address is determined by a Mach-O image
  //!     reader, such as MachOImageReader, as it walks Mach-O load commands.
  //! \param[in] load_command_info A string to be used in logged messages. This
  //!     string is for diagnostic purposes only, and may be empty.
  //! \param[in] module_name The path used to load the module. This string is
  //!     used to relax otherwise strict parsing rules for common modules with
  //!     known defects.
  //! \param[in] file_type The module’s Mach-O file type. This is used to relax
  //!     otherwise strict parsing rules for common modules with known defects.
  //!
  //! \return `true` if the load command was read successfully. `false`
  //!     otherwise, with an appropriate message logged.
  bool Initialize(ProcessReaderMac* process_reader,
                  mach_vm_address_t load_command_address,
                  const std::string& load_command_info,
                  const std::string& module_name,
                  uint32_t file_type);

  //! \brief Sets the image’s slide value.
  //!
  //! This method must only be called once on an object, after Initialize() is
  //! called successfully. It must be called before Address(), Size(),
  //! GetSectionByName(), or GetSectionAtIndex() can be called.
  //!
  //! This method is provided because slide is a property of the image that
  //! cannot be determined until at least some segments have been read. As such,
  //! it is not necessarily known at the time that Initialize() is called.
  void SetSlide(mach_vm_size_t slide);

  //! \brief Returns the segment’s name.
  //!
  //! The segment’s name is taken from the load command’s `segname` field.
  //! Common segment names are `"__TEXT"`, `"__DATA"`, and `"__LINKEDIT"`.
  //! Symbolic constants for these common names are defined in
  //! `<mach-o/loader.h>`.
  std::string Name() const;

  //! \return The segment’s actual load address in memory, adjusted for any
  //!     “slide”.
  //!
  //! \note For the segment’s preferred load address, not adjusted for slide,
  //!     use vmaddr().
  mach_vm_address_t Address() const;

  //! \return The segment’s actual size address in memory, adjusted for any
  //!     growth in the case of a nonsliding segment.
  //!
  //! \note For the segment’s preferred size, not adjusted for growth, use
  //!     vmsize().
  mach_vm_address_t Size() const;

  //! \brief The segment’s preferred load address.
  //!
  //! \return The segment’s preferred load address as stored in the Mach-O file.
  //!
  //! \note This value is not adjusted for any “slide” that may have occurred
  //!     when the image was loaded. Use Address() for a value adjusted for
  //!     slide.
  //!
  //! \sa MachOImageReader::GetSegmentByName()
  mach_vm_address_t vmaddr() const { return segment_command_.vmaddr; }

  //! \brief Returns the segment’s size as mapped into memory.
  //!
  //! \note For non-sliding segments, this value is not adjusted for any growth
  //!     that may have occurred when the image was loaded. Use Size() for a
  //!     value adjusted for growth.
  mach_vm_size_t vmsize() const { return segment_command_.vmsize; }

  //! \brief Returns the file offset of the mapped segment in the file from
  //!     which it was mapped.
  //!
  //! The file offset is the difference between the beginning of the
  //! `mach_header` or `mach_header_64` and the beginning of the segment’s
  //! mapped region. For segments that are not mapped from a file (such as
  //! `__PAGEZERO` segments), this will be `0`.
  mach_vm_size_t fileoff() const { return segment_command_.fileoff; }

  //! \brief Returns the number of sections in the segment.
  //!
  //! This will return `0` for a segment without any sections, typical for
  //! `__PAGEZERO` and `__LINKEDIT` segments.
  //!
  //! Although the Mach-O file format uses a `uint32_t` for this field, there is
  //! an overall limit of 255 sections in an entire Mach-O image file (not just
  //! in a single segment) imposed by the symbol table format. Symbols will not
  //! be able to reference anything in a section beyond the first 255 in a
  //! Mach-O image file.
  uint32_t nsects() const { return segment_command_.nsects; }

  //! \brief Obtain section information by section name.
  //!
  //! \param[in] section_name The name of the section to search for, without the
  //!     leading segment name. For example, use `"__text"`, not
  //!     `"__TEXT,__text"` or `"__TEXT.__text"`.
  //! \param[out] address The actual address that the section was loaded at in
  //!     memory, taking any “slide” into account if the section did not load at
  //!     its preferred address as stored in the Mach-O image file. This
  //!     parameter can be `nullptr`.
  //!
  //! \return A pointer to the section information if it was found, or `nullptr`
  //!     if it was not found. The caller does not take ownership; the lifetime
  //!     of the returned object is scoped to the lifetime of this
  //!     MachOImageSegmentReader object.
  //!
  //! \note The process_types::section::addr field gives the section’s preferred
  //!     load address as stored in the Mach-O image file, and is not adjusted
  //!     for any “slide” that may have occurred when the image was loaded.
  //!
  //! \sa MachOImageReader::GetSectionByName()
  const process_types::section* GetSectionByName(
      const std::string& section_name,
      mach_vm_address_t* address) const;

  //! \brief Obtain section information by section index.
  //!
  //! \param[in] index The index of the section to return, in the order that it
  //!     appears in the segment load command. Unlike
  //!     MachOImageReader::GetSectionAtIndex(), this is a 0-based index. This
  //!     parameter must be in the range of valid indices aas reported by
  //!     nsects().
  //! \param[out] address The actual address that the section was loaded at in
  //!     memory, taking any “slide” into account if the section did not load at
  //!     its preferred address as stored in the Mach-O image file. This
  //!     parameter can be `nullptr`.
  //!
  //! \return A pointer to the section information. If \a index is out of range,
  //!     execution is aborted.  The caller does not take ownership; the
  //!     lifetime of the returned object is scoped to the lifetime of this
  //!     MachOImageSegmentReader object.
  //!
  //! \note The process_types::section::addr field gives the section’s preferred
  //!     load address as stored in the Mach-O image file, and is not adjusted
  //!     for any “slide” that may have occurred when the image was loaded.
  //! \note Unlike MachOImageReader::GetSectionAtIndex(), this method does not
  //!     accept out-of-range values for \a index, and aborts execution instead
  //!     of returning `nullptr` upon encountering an out-of-range value. This
  //!     is because this method is expected to be used in a loop that can be
  //!     limited to nsects() iterations, so an out-of-range error can be
  //!     treated more harshly as a logic error, as opposed to a data error.
  //!
  //! \sa MachOImageReader::GetSectionAtIndex()
  const process_types::section* GetSectionAtIndex(
      size_t index,
      mach_vm_address_t* address) const;

  //! Returns whether the segment slides.
  //!
  //! Most segments slide, but the `__PAGEZERO` segment does not, it grows
  //! instead. This method identifies non-sliding segments in the same way that
  //! the kernel does.
  bool SegmentSlides() const;

  //! \brief Returns a segment name string.
  //!
  //! Segment names may be 16 characters long, and are not necessarily
  //! `NUL`-terminated. This function will return a segment name based on up to
  //! the first 16 characters found at \a segment_name_c.
  static std::string SegmentNameString(const char* segment_name_c);

  //! \brief Returns a section name string.
  //!
  //! Section names may be 16 characters long, and are not necessarily
  //! `NUL`-terminated. This function will return a section name based on up to
  //! the first 16 characters found at \a section_name_c.
  static std::string SectionNameString(const char* section_name_c);

  //! \brief Returns a segment and section name string.
  //!
  //! A segment and section name string is composed of a segment name string
  //! (see SegmentNameString()) and a section name string (see
  //! SectionNameString()) separated by a comma. An example is
  //! `"__TEXT,__text"`.
  static std::string SegmentAndSectionNameString(const char* segment_name_c,
                                                 const char* section_name_c);

 private:
  //! \brief The internal implementation of Name().
  //!
  //! This is identical to Name() but does not perform the
  //! InitializationStateDcheck check. It may be called during initialization
  //! provided that the caller only does so after segment_command_ has been
  //! read successfully.
  std::string NameInternal() const;

  // The segment command data read from the remote process.
  process_types::segment_command segment_command_;

  // Section structures read from the remote process in the order that they are
  // given in the remote process.
  std::vector<process_types::section> sections_;

  // Maps section names to indices into the sections_ vector.
  std::map<std::string, size_t> section_map_;

  // The image’s slide. Note that the segment’s slide may be 0 and not the value
  // of the image’s slide if SegmentSlides() is false. In that case, the
  // segment is extended instead of slid, so its size as loaded will be
  // increased by this value.
  mach_vm_size_t slide_;

  InitializationStateDcheck initialized_;
  InitializationStateDcheck initialized_slide_;

  DISALLOW_COPY_AND_ASSIGN(MachOImageSegmentReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_SEGMENT_READER_H_
