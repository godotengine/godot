/*
 * FileReaderFwd.h
 * ---------------
 * Purpose: Forward declaration for class FileReader.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

#include "typedefs.h"

OPENMPT_NAMESPACE_BEGIN

class FileReaderTraitsMemory;

#if defined(MPT_FILEREADER_STD_ISTREAM)

class FileReaderTraitsStdStream;

typedef FileReaderTraitsStdStream FileReaderTraitsDefault;

#else // !MPT_FILEREADER_STD_ISTREAM

typedef FileReaderTraitsMemory FileReaderTraitsDefault;

#endif // MPT_FILEREADER_STD_ISTREAM

namespace detail {

template <typename Ttraits>
class FileReader;

} // namespace detail

typedef detail::FileReader<FileReaderTraitsDefault> FileReader;

typedef detail::FileReader<FileReaderTraitsMemory> MemoryFileReader;

OPENMPT_NAMESPACE_END

