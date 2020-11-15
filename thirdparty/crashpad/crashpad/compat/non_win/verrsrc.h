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

#ifndef CRASHPAD_COMPAT_NON_WIN_VERRSRC_H_
#define CRASHPAD_COMPAT_NON_WIN_VERRSRC_H_

#include <stdint.h>

//! \file

//! \brief The magic number for a VS_FIXEDFILEINFO structure, stored in
//!     VS_FIXEDFILEINFO::dwSignature.
#define VS_FFI_SIGNATURE 0xfeef04bd

//! \brief The version of a VS_FIXEDFILEINFO structure, stored in
//!     VS_FIXEDFILEINFO::dwStrucVersion.
#define VS_FFI_STRUCVERSION 0x00010000

//! \anchor VS_FF_x
//! \name VS_FF_*
//!
//! \brief File attribute values for VS_FIXEDFILEINFO::dwFileFlags and
//!     VS_FIXEDFILEINFO::dwFileFlagsMask.
//! \{
#define VS_FF_DEBUG 0x00000001
#define VS_FF_PRERELEASE 0x00000002
#define VS_FF_PATCHED 0x00000004
#define VS_FF_PRIVATEBUILD 0x00000008
#define VS_FF_INFOINFERRED 0x00000010
#define VS_FF_SPECIALBUILD 0x00000020
//! \}

//! \anchor VOS_x
//! \name VOS_*
//!
//! \brief Operating system values for VS_FIXEDFILEINFO::dwFileOS.
//! \{
#define VOS_UNKNOWN 0x00000000
#define VOS_DOS 0x00010000
#define VOS_OS216 0x00020000
#define VOS_OS232 0x00030000
#define VOS_NT 0x00040000
#define VOS_WINCE 0x00050000
#define VOS__BASE 0x00000000
#define VOS__WINDOWS16 0x00000001
#define VOS__PM16 0x00000002
#define VOS__PM32 0x00000003
#define VOS__WINDOWS32 0x00000004
#define VOS_DOS_WINDOWS16 0x00010001
#define VOS_DOS_WINDOWS32 0x00010004
#define VOS_OS216_PM16 0x00020002
#define VOS_OS232_PM32 0x00030003
#define VOS_NT_WINDOWS32 0x00040004
//! \}

//! \anchor VFT_x
//! \name VFT_*
//!
//! \brief File type values for VS_FIXEDFILEINFO::dwFileType.
//! \{
#define VFT_UNKNOWN 0x00000000
#define VFT_APP 0x00000001
#define VFT_DLL 0x00000002
#define VFT_DRV 0x00000003
#define VFT_FONT 0x00000004
#define VFT_VXD 0x00000005
#define VFT_STATIC_LIB 0x00000007
//! \}

//! \anchor VFT2_x
//! \name VFT2_*
//!
//! \brief File subtype values for VS_FIXEDFILEINFO::dwFileSubtype.
//! \{
#define VFT2_UNKNOWN 0x00000000
#define VFT2_DRV_PRINTER 0x00000001
#define VFT2_DRV_KEYBOARD 0x00000002
#define VFT2_DRV_LANGUAGE 0x00000003
#define VFT2_DRV_DISPLAY 0x00000004
#define VFT2_DRV_MOUSE 0x00000005
#define VFT2_DRV_NETWORK 0x00000006
#define VFT2_DRV_SYSTEM 0x00000007
#define VFT2_DRV_INSTALLABLE 0x00000008
#define VFT2_DRV_SOUND 0x00000009
#define VFT2_DRV_COMM 0x0000000A
#define VFT2_DRV_INPUTMETHOD 0x0000000B
#define VFT2_DRV_VERSIONED_PRINTER 0x0000000C
#define VFT2_FONT_RASTER 0x00000001
#define VFT2_FONT_VECTOR 0x00000002
#define VFT2_FONT_TRUETYPE 0x00000003
//! \}

//! \brief Version information for a file.
//!
//! On Windows, this information is derived from a file’s version information
//! resource, and is obtained by calling `VerQueryValue()` with an `lpSubBlock`
//! argument of `"\"` (a single backslash).
struct VS_FIXEDFILEINFO {
  //! \brief The structure’s magic number, ::VS_FFI_SIGNATURE.
  uint32_t dwSignature;

  //! \brief The structure’s version, ::VS_FFI_STRUCVERSION.
  uint32_t dwStrucVersion;

  //! \brief The more-significant portion of the file’s version number.
  //!
  //! This field contains the first two components of a four-component version
  //! number. For a file whose version is 1.2.3.4, this field would be
  //! `0x00010002`.
  //!
  //! \sa dwFileVersionLS
  uint32_t dwFileVersionMS;

  //! \brief The less-significant portion of the file’s version number.
  //!
  //! This field contains the last two components of a four-component version
  //! number. For a file whose version is 1.2.3.4, this field would be
  //! `0x00030004`.
  //!
  //! \sa dwFileVersionMS
  uint32_t dwFileVersionLS;

  //! \brief The more-significant portion of the product’s version number.
  //!
  //! This field contains the first two components of a four-component version
  //! number. For a product whose version is 1.2.3.4, this field would be
  //! `0x00010002`.
  //!
  //! \sa dwProductVersionLS
  uint32_t dwProductVersionMS;

  //! \brief The less-significant portion of the product’s version number.
  //!
  //! This field contains the last two components of a four-component version
  //! number. For a product whose version is 1.2.3.4, this field would be
  //! `0x00030004`.
  //!
  //! \sa dwProductVersionMS
  uint32_t dwProductVersionLS;

  //! \brief A bitmask of \ref VS_FF_x "VS_FF_*" values indicating which bits in
  //!     #dwFileFlags are valid.
  uint32_t dwFileFlagsMask;

  //! \brief A bitmask of \ref VS_FF_x "VS_FF_*" values identifying attributes
  //!     of the file. Only bits present in #dwFileFlagsMask are valid.
  uint32_t dwFileFlags;

  //! \brief The file’s intended operating system, a value of \ref VOS_x
  //!     "VOS_*".
  uint32_t dwFileOS;

  //! \brief The file’s type, a value of \ref VFT_x "VFT_*".
  uint32_t dwFileType;

  //! \brief The file’s subtype, a value of \ref VFT2_x "VFT2_*" corresponding
  //!     to its #dwFileType, if the file type has subtypes.
  uint32_t dwFileSubtype;

  //! \brief The more-significant portion of the file’s creation date.
  //!
  //! The intended encoding of this field is unknown. This field is unused and
  //! always has the value `0`.
  uint32_t dwFileDateMS;

  //! \brief The less-significant portion of the file’s creation date.
  //!
  //! The intended encoding of this field is unknown. This field is unused and
  //! always has the value `0`.
  uint32_t dwFileDateLS;
};

#endif  // CRASHPAD_COMPAT_NON_WIN_VERRSRC_H_
