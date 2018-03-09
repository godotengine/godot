// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "mkvparser/mkvparser.h"

#if defined(_MSC_VER) && _MSC_VER < 1800
#include <float.h>  // _isnan() / _finite()
#define MSC_COMPAT
#endif

#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstring>
#include <memory>
#include <new>

#include "common/webmids.h"

namespace mkvparser {
const long long kStringElementSizeLimit = 20 * 1000 * 1000;
const float MasteringMetadata::kValueNotPresent = FLT_MAX;
const long long Colour::kValueNotPresent = LLONG_MAX;
const float Projection::kValueNotPresent = FLT_MAX;

#ifdef MSC_COMPAT
inline bool isnan(double val) { return !!_isnan(val); }
inline bool isinf(double val) { return !_finite(val); }
#else
inline bool isnan(double val) { return std::isnan(val); }
inline bool isinf(double val) { return std::isinf(val); }
#endif  // MSC_COMPAT

IMkvReader::~IMkvReader() {}

template <typename Type>
Type* SafeArrayAlloc(unsigned long long num_elements,
                     unsigned long long element_size) {
  if (num_elements == 0 || element_size == 0)
    return NULL;

  const size_t kMaxAllocSize = 0x80000000;  // 2GiB
  const unsigned long long num_bytes = num_elements * element_size;
  if (element_size > (kMaxAllocSize / num_elements))
    return NULL;
  if (num_bytes != static_cast<size_t>(num_bytes))
    return NULL;

  return new (std::nothrow) Type[static_cast<size_t>(num_bytes)];
}

void GetVersion(int& major, int& minor, int& build, int& revision) {
  major = 1;
  minor = 0;
  build = 0;
  revision = 30;
}

long long ReadUInt(IMkvReader* pReader, long long pos, long& len) {
  if (!pReader || pos < 0)
    return E_FILE_FORMAT_INVALID;

  len = 1;
  unsigned char b;
  int status = pReader->Read(pos, 1, &b);

  if (status < 0)  // error or underflow
    return status;

  if (status > 0)  // interpreted as "underflow"
    return E_BUFFER_NOT_FULL;

  if (b == 0)  // we can't handle u-int values larger than 8 bytes
    return E_FILE_FORMAT_INVALID;

  unsigned char m = 0x80;

  while (!(b & m)) {
    m >>= 1;
    ++len;
  }

  long long result = b & (~m);
  ++pos;

  for (int i = 1; i < len; ++i) {
    status = pReader->Read(pos, 1, &b);

    if (status < 0) {
      len = 1;
      return status;
    }

    if (status > 0) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    result <<= 8;
    result |= b;

    ++pos;
  }

  return result;
}

// Reads an EBML ID and returns it.
// An ID must at least 1 byte long, cannot exceed 4, and its value must be
// greater than 0.
// See known EBML values and EBMLMaxIDLength:
// http://www.matroska.org/technical/specs/index.html
// Returns the ID, or a value less than 0 to report an error while reading the
// ID.
long long ReadID(IMkvReader* pReader, long long pos, long& len) {
  if (pReader == NULL || pos < 0)
    return E_FILE_FORMAT_INVALID;

  // Read the first byte. The length in bytes of the ID is determined by
  // finding the first set bit in the first byte of the ID.
  unsigned char temp_byte = 0;
  int read_status = pReader->Read(pos, 1, &temp_byte);

  if (read_status < 0)
    return E_FILE_FORMAT_INVALID;
  else if (read_status > 0)  // No data to read.
    return E_BUFFER_NOT_FULL;

  if (temp_byte == 0)  // ID length > 8 bytes; invalid file.
    return E_FILE_FORMAT_INVALID;

  int bit_pos = 0;
  const int kMaxIdLengthInBytes = 4;
  const int kCheckByte = 0x80;

  // Find the first bit that's set.
  bool found_bit = false;
  for (; bit_pos < kMaxIdLengthInBytes; ++bit_pos) {
    if ((kCheckByte >> bit_pos) & temp_byte) {
      found_bit = true;
      break;
    }
  }

  if (!found_bit) {
    // The value is too large to be a valid ID.
    return E_FILE_FORMAT_INVALID;
  }

  // Read the remaining bytes of the ID (if any).
  const int id_length = bit_pos + 1;
  long long ebml_id = temp_byte;
  for (int i = 1; i < id_length; ++i) {
    ebml_id <<= 8;
    read_status = pReader->Read(pos + i, 1, &temp_byte);

    if (read_status < 0)
      return E_FILE_FORMAT_INVALID;
    else if (read_status > 0)
      return E_BUFFER_NOT_FULL;

    ebml_id |= temp_byte;
  }

  len = id_length;
  return ebml_id;
}

long long GetUIntLength(IMkvReader* pReader, long long pos, long& len) {
  if (!pReader || pos < 0)
    return E_FILE_FORMAT_INVALID;

  long long total, available;

  int status = pReader->Length(&total, &available);
  if (status < 0 || (total >= 0 && available > total))
    return E_FILE_FORMAT_INVALID;

  len = 1;

  if (pos >= available)
    return pos;  // too few bytes available

  unsigned char b;

  status = pReader->Read(pos, 1, &b);

  if (status != 0)
    return status;

  if (b == 0)  // we can't handle u-int values larger than 8 bytes
    return E_FILE_FORMAT_INVALID;

  unsigned char m = 0x80;

  while (!(b & m)) {
    m >>= 1;
    ++len;
  }

  return 0;  // success
}

// TODO(vigneshv): This function assumes that unsigned values never have their
// high bit set.
long long UnserializeUInt(IMkvReader* pReader, long long pos, long long size) {
  if (!pReader || pos < 0 || (size <= 0) || (size > 8))
    return E_FILE_FORMAT_INVALID;

  long long result = 0;

  for (long long i = 0; i < size; ++i) {
    unsigned char b;

    const long status = pReader->Read(pos, 1, &b);

    if (status < 0)
      return status;

    result <<= 8;
    result |= b;

    ++pos;
  }

  return result;
}

long UnserializeFloat(IMkvReader* pReader, long long pos, long long size_,
                      double& result) {
  if (!pReader || pos < 0 || ((size_ != 4) && (size_ != 8)))
    return E_FILE_FORMAT_INVALID;

  const long size = static_cast<long>(size_);

  unsigned char buf[8];

  const int status = pReader->Read(pos, size, buf);

  if (status < 0)  // error
    return status;

  if (size == 4) {
    union {
      float f;
      unsigned long ff;
    };

    ff = 0;

    for (int i = 0;;) {
      ff |= buf[i];

      if (++i >= 4)
        break;

      ff <<= 8;
    }

    result = f;
  } else {
    union {
      double d;
      unsigned long long dd;
    };

    dd = 0;

    for (int i = 0;;) {
      dd |= buf[i];

      if (++i >= 8)
        break;

      dd <<= 8;
    }

    result = d;
  }

  if (mkvparser::isinf(result) || mkvparser::isnan(result))
    return E_FILE_FORMAT_INVALID;

  return 0;
}

long UnserializeInt(IMkvReader* pReader, long long pos, long long size,
                    long long& result_ref) {
  if (!pReader || pos < 0 || size < 1 || size > 8)
    return E_FILE_FORMAT_INVALID;

  signed char first_byte = 0;
  const long status = pReader->Read(pos, 1, (unsigned char*)&first_byte);

  if (status < 0)
    return status;

  unsigned long long result = first_byte;
  ++pos;

  for (long i = 1; i < size; ++i) {
    unsigned char b;

    const long status = pReader->Read(pos, 1, &b);

    if (status < 0)
      return status;

    result <<= 8;
    result |= b;

    ++pos;
  }

  result_ref = static_cast<long long>(result);
  return 0;
}

long UnserializeString(IMkvReader* pReader, long long pos, long long size,
                       char*& str) {
  delete[] str;
  str = NULL;

  if (size >= LONG_MAX || size < 0 || size > kStringElementSizeLimit)
    return E_FILE_FORMAT_INVALID;

  // +1 for '\0' terminator
  const long required_size = static_cast<long>(size) + 1;

  str = SafeArrayAlloc<char>(1, required_size);
  if (str == NULL)
    return E_FILE_FORMAT_INVALID;

  unsigned char* const buf = reinterpret_cast<unsigned char*>(str);

  const long status = pReader->Read(pos, static_cast<long>(size), buf);

  if (status) {
    delete[] str;
    str = NULL;

    return status;
  }

  str[required_size - 1] = '\0';
  return 0;
}

long ParseElementHeader(IMkvReader* pReader, long long& pos, long long stop,
                        long long& id, long long& size) {
  if (stop >= 0 && pos >= stop)
    return E_FILE_FORMAT_INVALID;

  long len;

  id = ReadID(pReader, pos, len);

  if (id < 0)
    return E_FILE_FORMAT_INVALID;

  pos += len;  // consume id

  if (stop >= 0 && pos >= stop)
    return E_FILE_FORMAT_INVALID;

  size = ReadUInt(pReader, pos, len);

  if (size < 0 || len < 1 || len > 8) {
    // Invalid: Negative payload size, negative or 0 length integer, or integer
    // larger than 64 bits (libwebm cannot handle them).
    return E_FILE_FORMAT_INVALID;
  }

  // Avoid rolling over pos when very close to LLONG_MAX.
  const unsigned long long rollover_check =
      static_cast<unsigned long long>(pos) + len;
  if (rollover_check > LLONG_MAX)
    return E_FILE_FORMAT_INVALID;

  pos += len;  // consume length of size

  // pos now designates payload

  if (stop >= 0 && pos > stop)
    return E_FILE_FORMAT_INVALID;

  return 0;  // success
}

bool Match(IMkvReader* pReader, long long& pos, unsigned long expected_id,
           long long& val) {
  if (!pReader || pos < 0)
    return false;

  long long total = 0;
  long long available = 0;

  const long status = pReader->Length(&total, &available);
  if (status < 0 || (total >= 0 && available > total))
    return false;

  long len = 0;

  const long long id = ReadID(pReader, pos, len);
  if (id < 0 || (available - pos) > len)
    return false;

  if (static_cast<unsigned long>(id) != expected_id)
    return false;

  pos += len;  // consume id

  const long long size = ReadUInt(pReader, pos, len);
  if (size < 0 || size > 8 || len < 1 || len > 8 || (available - pos) > len)
    return false;

  pos += len;  // consume length of size of payload

  val = UnserializeUInt(pReader, pos, size);
  if (val < 0)
    return false;

  pos += size;  // consume size of payload

  return true;
}

bool Match(IMkvReader* pReader, long long& pos, unsigned long expected_id,
           unsigned char*& buf, size_t& buflen) {
  if (!pReader || pos < 0)
    return false;

  long long total = 0;
  long long available = 0;

  long status = pReader->Length(&total, &available);
  if (status < 0 || (total >= 0 && available > total))
    return false;

  long len = 0;
  const long long id = ReadID(pReader, pos, len);
  if (id < 0 || (available - pos) > len)
    return false;

  if (static_cast<unsigned long>(id) != expected_id)
    return false;

  pos += len;  // consume id

  const long long size = ReadUInt(pReader, pos, len);
  if (size < 0 || len <= 0 || len > 8 || (available - pos) > len)
    return false;

  unsigned long long rollover_check =
      static_cast<unsigned long long>(pos) + len;
  if (rollover_check > LLONG_MAX)
    return false;

  pos += len;  // consume length of size of payload

  rollover_check = static_cast<unsigned long long>(pos) + size;
  if (rollover_check > LLONG_MAX)
    return false;

  if ((pos + size) > available)
    return false;

  if (size >= LONG_MAX)
    return false;

  const long buflen_ = static_cast<long>(size);

  buf = SafeArrayAlloc<unsigned char>(1, buflen_);
  if (!buf)
    return false;

  status = pReader->Read(pos, buflen_, buf);
  if (status != 0)
    return false;

  buflen = buflen_;

  pos += size;  // consume size of payload
  return true;
}

EBMLHeader::EBMLHeader() : m_docType(NULL) { Init(); }

EBMLHeader::~EBMLHeader() { delete[] m_docType; }

void EBMLHeader::Init() {
  m_version = 1;
  m_readVersion = 1;
  m_maxIdLength = 4;
  m_maxSizeLength = 8;

  if (m_docType) {
    delete[] m_docType;
    m_docType = NULL;
  }

  m_docTypeVersion = 1;
  m_docTypeReadVersion = 1;
}

long long EBMLHeader::Parse(IMkvReader* pReader, long long& pos) {
  if (!pReader)
    return E_FILE_FORMAT_INVALID;

  long long total, available;

  long status = pReader->Length(&total, &available);

  if (status < 0)  // error
    return status;

  pos = 0;

  // Scan until we find what looks like the first byte of the EBML header.
  const long long kMaxScanBytes = (available >= 1024) ? 1024 : available;
  const unsigned char kEbmlByte0 = 0x1A;
  unsigned char scan_byte = 0;

  while (pos < kMaxScanBytes) {
    status = pReader->Read(pos, 1, &scan_byte);

    if (status < 0)  // error
      return status;
    else if (status > 0)
      return E_BUFFER_NOT_FULL;

    if (scan_byte == kEbmlByte0)
      break;

    ++pos;
  }

  long len = 0;
  const long long ebml_id = ReadID(pReader, pos, len);

  if (ebml_id == E_BUFFER_NOT_FULL)
    return E_BUFFER_NOT_FULL;

  if (len != 4 || ebml_id != libwebm::kMkvEBML)
    return E_FILE_FORMAT_INVALID;

  // Move read pos forward to the EBML header size field.
  pos += 4;

  // Read length of size field.
  long long result = GetUIntLength(pReader, pos, len);

  if (result < 0)  // error
    return E_FILE_FORMAT_INVALID;
  else if (result > 0)  // need more data
    return E_BUFFER_NOT_FULL;

  if (len < 1 || len > 8)
    return E_FILE_FORMAT_INVALID;

  if ((total >= 0) && ((total - pos) < len))
    return E_FILE_FORMAT_INVALID;

  if ((available - pos) < len)
    return pos + len;  // try again later

  // Read the EBML header size.
  result = ReadUInt(pReader, pos, len);

  if (result < 0)  // error
    return result;

  pos += len;  // consume size field

  // pos now designates start of payload

  if ((total >= 0) && ((total - pos) < result))
    return E_FILE_FORMAT_INVALID;

  if ((available - pos) < result)
    return pos + result;

  const long long end = pos + result;

  Init();

  while (pos < end) {
    long long id, size;

    status = ParseElementHeader(pReader, pos, end, id, size);

    if (status < 0)  // error
      return status;

    if (size == 0)
      return E_FILE_FORMAT_INVALID;

    if (id == libwebm::kMkvEBMLVersion) {
      m_version = UnserializeUInt(pReader, pos, size);

      if (m_version <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvEBMLReadVersion) {
      m_readVersion = UnserializeUInt(pReader, pos, size);

      if (m_readVersion <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvEBMLMaxIDLength) {
      m_maxIdLength = UnserializeUInt(pReader, pos, size);

      if (m_maxIdLength <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvEBMLMaxSizeLength) {
      m_maxSizeLength = UnserializeUInt(pReader, pos, size);

      if (m_maxSizeLength <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvDocType) {
      if (m_docType)
        return E_FILE_FORMAT_INVALID;

      status = UnserializeString(pReader, pos, size, m_docType);

      if (status)  // error
        return status;
    } else if (id == libwebm::kMkvDocTypeVersion) {
      m_docTypeVersion = UnserializeUInt(pReader, pos, size);

      if (m_docTypeVersion <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvDocTypeReadVersion) {
      m_docTypeReadVersion = UnserializeUInt(pReader, pos, size);

      if (m_docTypeReadVersion <= 0)
        return E_FILE_FORMAT_INVALID;
    }

    pos += size;
  }

  if (pos != end)
    return E_FILE_FORMAT_INVALID;

  // Make sure DocType, DocTypeReadVersion, and DocTypeVersion are valid.
  if (m_docType == NULL || m_docTypeReadVersion <= 0 || m_docTypeVersion <= 0)
    return E_FILE_FORMAT_INVALID;

  // Make sure EBMLMaxIDLength and EBMLMaxSizeLength are valid.
  if (m_maxIdLength <= 0 || m_maxIdLength > 4 || m_maxSizeLength <= 0 ||
      m_maxSizeLength > 8)
    return E_FILE_FORMAT_INVALID;

  return 0;
}

Segment::Segment(IMkvReader* pReader, long long elem_start,
                 // long long elem_size,
                 long long start, long long size)
    : m_pReader(pReader),
      m_element_start(elem_start),
      // m_element_size(elem_size),
      m_start(start),
      m_size(size),
      m_pos(start),
      m_pUnknownSize(0),
      m_pSeekHead(NULL),
      m_pInfo(NULL),
      m_pTracks(NULL),
      m_pCues(NULL),
      m_pChapters(NULL),
      m_pTags(NULL),
      m_clusters(NULL),
      m_clusterCount(0),
      m_clusterPreloadCount(0),
      m_clusterSize(0) {}

Segment::~Segment() {
  const long count = m_clusterCount + m_clusterPreloadCount;

  Cluster** i = m_clusters;
  Cluster** j = m_clusters + count;

  while (i != j) {
    Cluster* const p = *i++;
    delete p;
  }

  delete[] m_clusters;

  delete m_pTracks;
  delete m_pInfo;
  delete m_pCues;
  delete m_pChapters;
  delete m_pTags;
  delete m_pSeekHead;
}

long long Segment::CreateInstance(IMkvReader* pReader, long long pos,
                                  Segment*& pSegment) {
  if (pReader == NULL || pos < 0)
    return E_PARSE_FAILED;

  pSegment = NULL;

  long long total, available;

  const long status = pReader->Length(&total, &available);

  if (status < 0)  // error
    return status;

  if (available < 0)
    return -1;

  if ((total >= 0) && (available > total))
    return -1;

  // I would assume that in practice this loop would execute
  // exactly once, but we allow for other elements (e.g. Void)
  // to immediately follow the EBML header.  This is fine for
  // the source filter case (since the entire file is available),
  // but in the splitter case over a network we should probably
  // just give up early.  We could for example decide only to
  // execute this loop a maximum of, say, 10 times.
  // TODO:
  // There is an implied "give up early" by only parsing up
  // to the available limit.  We do do that, but only if the
  // total file size is unknown.  We could decide to always
  // use what's available as our limit (irrespective of whether
  // we happen to know the total file length).  This would have
  // as its sense "parse this much of the file before giving up",
  // which a slightly different sense from "try to parse up to
  // 10 EMBL elements before giving up".

  for (;;) {
    if ((total >= 0) && (pos >= total))
      return E_FILE_FORMAT_INVALID;

    // Read ID
    long len;
    long long result = GetUIntLength(pReader, pos, len);

    if (result)  // error, or too few available bytes
      return result;

    if ((total >= 0) && ((pos + len) > total))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > available)
      return pos + len;

    const long long idpos = pos;
    const long long id = ReadID(pReader, pos, len);

    if (id < 0)
      return E_FILE_FORMAT_INVALID;

    pos += len;  // consume ID

    // Read Size

    result = GetUIntLength(pReader, pos, len);

    if (result)  // error, or too few available bytes
      return result;

    if ((total >= 0) && ((pos + len) > total))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > available)
      return pos + len;

    long long size = ReadUInt(pReader, pos, len);

    if (size < 0)  // error
      return size;

    pos += len;  // consume length of size of element

    // Pos now points to start of payload

    // Handle "unknown size" for live streaming of webm files.
    const long long unknown_size = (1LL << (7 * len)) - 1;

    if (id == libwebm::kMkvSegment) {
      if (size == unknown_size)
        size = -1;

      else if (total < 0)
        size = -1;

      else if ((pos + size) > total)
        size = -1;

      pSegment = new (std::nothrow) Segment(pReader, idpos, pos, size);
      if (pSegment == NULL)
        return E_PARSE_FAILED;

      return 0;  // success
    }

    if (size == unknown_size)
      return E_FILE_FORMAT_INVALID;

    if ((total >= 0) && ((pos + size) > total))
      return E_FILE_FORMAT_INVALID;

    if ((pos + size) > available)
      return pos + size;

    pos += size;  // consume payload
  }
}

long long Segment::ParseHeaders() {
  // Outermost (level 0) segment object has been constructed,
  // and pos designates start of payload.  We need to find the
  // inner (level 1) elements.
  long long total, available;

  const int status = m_pReader->Length(&total, &available);

  if (status < 0)  // error
    return status;

  if (total > 0 && available > total)
    return E_FILE_FORMAT_INVALID;

  const long long segment_stop = (m_size < 0) ? -1 : m_start + m_size;

  if ((segment_stop >= 0 && total >= 0 && segment_stop > total) ||
      (segment_stop >= 0 && m_pos > segment_stop)) {
    return E_FILE_FORMAT_INVALID;
  }

  for (;;) {
    if ((total >= 0) && (m_pos >= total))
      break;

    if ((segment_stop >= 0) && (m_pos >= segment_stop))
      break;

    long long pos = m_pos;
    const long long element_start = pos;

    // Avoid rolling over pos when very close to LLONG_MAX.
    unsigned long long rollover_check = pos + 1ULL;
    if (rollover_check > LLONG_MAX)
      return E_FILE_FORMAT_INVALID;

    if ((pos + 1) > available)
      return (pos + 1);

    long len;
    long long result = GetUIntLength(m_pReader, pos, len);

    if (result < 0)  // error
      return result;

    if (result > 0) {
      // MkvReader doesn't have enough data to satisfy this read attempt.
      return (pos + 1);
    }

    if ((segment_stop >= 0) && ((pos + len) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > available)
      return pos + len;

    const long long idpos = pos;
    const long long id = ReadID(m_pReader, idpos, len);

    if (id < 0)
      return E_FILE_FORMAT_INVALID;

    if (id == libwebm::kMkvCluster)
      break;

    pos += len;  // consume ID

    if ((pos + 1) > available)
      return (pos + 1);

    // Read Size
    result = GetUIntLength(m_pReader, pos, len);

    if (result < 0)  // error
      return result;

    if (result > 0) {
      // MkvReader doesn't have enough data to satisfy this read attempt.
      return (pos + 1);
    }

    if ((segment_stop >= 0) && ((pos + len) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > available)
      return pos + len;

    const long long size = ReadUInt(m_pReader, pos, len);

    if (size < 0 || len < 1 || len > 8) {
      // TODO(tomfinegan): ReadUInt should return an error when len is < 1 or
      // len > 8 is true instead of checking this _everywhere_.
      return size;
    }

    pos += len;  // consume length of size of element

    // Avoid rolling over pos when very close to LLONG_MAX.
    rollover_check = static_cast<unsigned long long>(pos) + size;
    if (rollover_check > LLONG_MAX)
      return E_FILE_FORMAT_INVALID;

    const long long element_size = size + pos - element_start;

    // Pos now points to start of payload

    if ((segment_stop >= 0) && ((pos + size) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    // We read EBML elements either in total or nothing at all.

    if ((pos + size) > available)
      return pos + size;

    if (id == libwebm::kMkvInfo) {
      if (m_pInfo)
        return E_FILE_FORMAT_INVALID;

      m_pInfo = new (std::nothrow)
          SegmentInfo(this, pos, size, element_start, element_size);

      if (m_pInfo == NULL)
        return -1;

      const long status = m_pInfo->Parse();

      if (status)
        return status;
    } else if (id == libwebm::kMkvTracks) {
      if (m_pTracks)
        return E_FILE_FORMAT_INVALID;

      m_pTracks = new (std::nothrow)
          Tracks(this, pos, size, element_start, element_size);

      if (m_pTracks == NULL)
        return -1;

      const long status = m_pTracks->Parse();

      if (status)
        return status;
    } else if (id == libwebm::kMkvCues) {
      if (m_pCues == NULL) {
        m_pCues = new (std::nothrow)
            Cues(this, pos, size, element_start, element_size);

        if (m_pCues == NULL)
          return -1;
      }
    } else if (id == libwebm::kMkvSeekHead) {
      if (m_pSeekHead == NULL) {
        m_pSeekHead = new (std::nothrow)
            SeekHead(this, pos, size, element_start, element_size);

        if (m_pSeekHead == NULL)
          return -1;

        const long status = m_pSeekHead->Parse();

        if (status)
          return status;
      }
    } else if (id == libwebm::kMkvChapters) {
      if (m_pChapters == NULL) {
        m_pChapters = new (std::nothrow)
            Chapters(this, pos, size, element_start, element_size);

        if (m_pChapters == NULL)
          return -1;

        const long status = m_pChapters->Parse();

        if (status)
          return status;
      }
    } else if (id == libwebm::kMkvTags) {
      if (m_pTags == NULL) {
        m_pTags = new (std::nothrow)
            Tags(this, pos, size, element_start, element_size);

        if (m_pTags == NULL)
          return -1;

        const long status = m_pTags->Parse();

        if (status)
          return status;
      }
    }

    m_pos = pos + size;  // consume payload
  }

  if (segment_stop >= 0 && m_pos > segment_stop)
    return E_FILE_FORMAT_INVALID;

  if (m_pInfo == NULL)  // TODO: liberalize this behavior
    return E_FILE_FORMAT_INVALID;

  if (m_pTracks == NULL)
    return E_FILE_FORMAT_INVALID;

  return 0;  // success
}

long Segment::LoadCluster(long long& pos, long& len) {
  for (;;) {
    const long result = DoLoadCluster(pos, len);

    if (result <= 1)
      return result;
  }
}

long Segment::DoLoadCluster(long long& pos, long& len) {
  if (m_pos < 0)
    return DoLoadClusterUnknownSize(pos, len);

  long long total, avail;

  long status = m_pReader->Length(&total, &avail);

  if (status < 0)  // error
    return status;

  if (total >= 0 && avail > total)
    return E_FILE_FORMAT_INVALID;

  const long long segment_stop = (m_size < 0) ? -1 : m_start + m_size;

  long long cluster_off = -1;  // offset relative to start of segment
  long long cluster_size = -1;  // size of cluster payload

  for (;;) {
    if ((total >= 0) && (m_pos >= total))
      return 1;  // no more clusters

    if ((segment_stop >= 0) && (m_pos >= segment_stop))
      return 1;  // no more clusters

    pos = m_pos;

    // Read ID

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    long long result = GetUIntLength(m_pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)
      return E_BUFFER_NOT_FULL;

    if ((segment_stop >= 0) && ((pos + len) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long idpos = pos;
    const long long id = ReadID(m_pReader, idpos, len);

    if (id < 0)
      return E_FILE_FORMAT_INVALID;

    pos += len;  // consume ID

    // Read Size

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    result = GetUIntLength(m_pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)
      return E_BUFFER_NOT_FULL;

    if ((segment_stop >= 0) && ((pos + len) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long size = ReadUInt(m_pReader, pos, len);

    if (size < 0)  // error
      return static_cast<long>(size);

    pos += len;  // consume length of size of element

    // pos now points to start of payload

    if (size == 0) {
      // Missing element payload: move on.
      m_pos = pos;
      continue;
    }

    const long long unknown_size = (1LL << (7 * len)) - 1;

    if ((segment_stop >= 0) && (size != unknown_size) &&
        ((pos + size) > segment_stop)) {
      return E_FILE_FORMAT_INVALID;
    }

    if (id == libwebm::kMkvCues) {
      if (size == unknown_size) {
        // Cues element of unknown size: Not supported.
        return E_FILE_FORMAT_INVALID;
      }

      if (m_pCues == NULL) {
        const long long element_size = (pos - idpos) + size;

        m_pCues = new (std::nothrow) Cues(this, pos, size, idpos, element_size);
        if (m_pCues == NULL)
          return -1;
      }

      m_pos = pos + size;  // consume payload
      continue;
    }

    if (id != libwebm::kMkvCluster) {
      // Besides the Segment, Libwebm allows only cluster elements of unknown
      // size. Fail the parse upon encountering a non-cluster element reporting
      // unknown size.
      if (size == unknown_size)
        return E_FILE_FORMAT_INVALID;

      m_pos = pos + size;  // consume payload
      continue;
    }

    // We have a cluster.

    cluster_off = idpos - m_start;  // relative pos

    if (size != unknown_size)
      cluster_size = size;

    break;
  }

  if (cluster_off < 0) {
    // No cluster, die.
    return E_FILE_FORMAT_INVALID;
  }

  long long pos_;
  long len_;

  status = Cluster::HasBlockEntries(this, cluster_off, pos_, len_);

  if (status < 0) {  // error, or underflow
    pos = pos_;
    len = len_;

    return status;
  }

  // status == 0 means "no block entries found"
  // status > 0 means "found at least one block entry"

  // TODO:
  // The issue here is that the segment increments its own
  // pos ptr past the most recent cluster parsed, and then
  // starts from there to parse the next cluster.  If we
  // don't know the size of the current cluster, then we
  // must either parse its payload (as we do below), looking
  // for the cluster (or cues) ID to terminate the parse.
  // This isn't really what we want: rather, we really need
  // a way to create the curr cluster object immediately.
  // The pity is that cluster::parse can determine its own
  // boundary, and we largely duplicate that same logic here.
  //
  // Maybe we need to get rid of our look-ahead preloading
  // in source::parse???
  //
  // As we're parsing the blocks in the curr cluster
  //(in cluster::parse), we should have some way to signal
  // to the segment that we have determined the boundary,
  // so it can adjust its own segment::m_pos member.
  //
  // The problem is that we're asserting in asyncreadinit,
  // because we adjust the pos down to the curr seek pos,
  // and the resulting adjusted len is > 2GB.  I'm suspicious
  // that this is even correct, but even if it is, we can't
  // be loading that much data in the cache anyway.

  const long idx = m_clusterCount;

  if (m_clusterPreloadCount > 0) {
    if (idx >= m_clusterSize)
      return E_FILE_FORMAT_INVALID;

    Cluster* const pCluster = m_clusters[idx];
    if (pCluster == NULL || pCluster->m_index >= 0)
      return E_FILE_FORMAT_INVALID;

    const long long off = pCluster->GetPosition();
    if (off < 0)
      return E_FILE_FORMAT_INVALID;

    if (off == cluster_off) {  // preloaded already
      if (status == 0)  // no entries found
        return E_FILE_FORMAT_INVALID;

      if (cluster_size >= 0)
        pos += cluster_size;
      else {
        const long long element_size = pCluster->GetElementSize();

        if (element_size <= 0)
          return E_FILE_FORMAT_INVALID;  // TODO: handle this case

        pos = pCluster->m_element_start + element_size;
      }

      pCluster->m_index = idx;  // move from preloaded to loaded
      ++m_clusterCount;
      --m_clusterPreloadCount;

      m_pos = pos;  // consume payload
      if (segment_stop >= 0 && m_pos > segment_stop)
        return E_FILE_FORMAT_INVALID;

      return 0;  // success
    }
  }

  if (status == 0) {  // no entries found
    if (cluster_size >= 0)
      pos += cluster_size;

    if ((total >= 0) && (pos >= total)) {
      m_pos = total;
      return 1;  // no more clusters
    }

    if ((segment_stop >= 0) && (pos >= segment_stop)) {
      m_pos = segment_stop;
      return 1;  // no more clusters
    }

    m_pos = pos;
    return 2;  // try again
  }

  // status > 0 means we have an entry

  Cluster* const pCluster = Cluster::Create(this, idx, cluster_off);
  if (pCluster == NULL)
    return -1;

  if (!AppendCluster(pCluster)) {
    delete pCluster;
    return -1;
  }

  if (cluster_size >= 0) {
    pos += cluster_size;

    m_pos = pos;

    if (segment_stop > 0 && m_pos > segment_stop)
      return E_FILE_FORMAT_INVALID;

    return 0;
  }

  m_pUnknownSize = pCluster;
  m_pos = -pos;

  return 0;  // partial success, since we have a new cluster

  // status == 0 means "no block entries found"
  // pos designates start of payload
  // m_pos has NOT been adjusted yet (in case we need to come back here)
}

long Segment::DoLoadClusterUnknownSize(long long& pos, long& len) {
  if (m_pos >= 0 || m_pUnknownSize == NULL)
    return E_PARSE_FAILED;

  const long status = m_pUnknownSize->Parse(pos, len);

  if (status < 0)  // error or underflow
    return status;

  if (status == 0)  // parsed a block
    return 2;  // continue parsing

  const long long start = m_pUnknownSize->m_element_start;
  const long long size = m_pUnknownSize->GetElementSize();

  if (size < 0)
    return E_FILE_FORMAT_INVALID;

  pos = start + size;
  m_pos = pos;

  m_pUnknownSize = 0;

  return 2;  // continue parsing
}

bool Segment::AppendCluster(Cluster* pCluster) {
  if (pCluster == NULL || pCluster->m_index < 0)
    return false;

  const long count = m_clusterCount + m_clusterPreloadCount;

  long& size = m_clusterSize;
  const long idx = pCluster->m_index;

  if (size < count || idx != m_clusterCount)
    return false;

  if (count >= size) {
    const long n = (size <= 0) ? 2048 : 2 * size;

    Cluster** const qq = new (std::nothrow) Cluster*[n];
    if (qq == NULL)
      return false;

    Cluster** q = qq;
    Cluster** p = m_clusters;
    Cluster** const pp = p + count;

    while (p != pp)
      *q++ = *p++;

    delete[] m_clusters;

    m_clusters = qq;
    size = n;
  }

  if (m_clusterPreloadCount > 0) {
    Cluster** const p = m_clusters + m_clusterCount;
    if (*p == NULL || (*p)->m_index >= 0)
      return false;

    Cluster** q = p + m_clusterPreloadCount;
    if (q >= (m_clusters + size))
      return false;

    for (;;) {
      Cluster** const qq = q - 1;
      if ((*qq)->m_index >= 0)
        return false;

      *q = *qq;
      q = qq;

      if (q == p)
        break;
    }
  }

  m_clusters[idx] = pCluster;
  ++m_clusterCount;
  return true;
}

bool Segment::PreloadCluster(Cluster* pCluster, ptrdiff_t idx) {
  if (pCluster == NULL || pCluster->m_index >= 0 || idx < m_clusterCount)
    return false;

  const long count = m_clusterCount + m_clusterPreloadCount;

  long& size = m_clusterSize;
  if (size < count)
    return false;

  if (count >= size) {
    const long n = (size <= 0) ? 2048 : 2 * size;

    Cluster** const qq = new (std::nothrow) Cluster*[n];
    if (qq == NULL)
      return false;
    Cluster** q = qq;

    Cluster** p = m_clusters;
    Cluster** const pp = p + count;

    while (p != pp)
      *q++ = *p++;

    delete[] m_clusters;

    m_clusters = qq;
    size = n;
  }

  if (m_clusters == NULL)
    return false;

  Cluster** const p = m_clusters + idx;

  Cluster** q = m_clusters + count;
  if (q < p || q >= (m_clusters + size))
    return false;

  while (q > p) {
    Cluster** const qq = q - 1;

    if ((*qq)->m_index >= 0)
      return false;

    *q = *qq;
    q = qq;
  }

  m_clusters[idx] = pCluster;
  ++m_clusterPreloadCount;
  return true;
}

long Segment::Load() {
  if (m_clusters != NULL || m_clusterSize != 0 || m_clusterCount != 0)
    return E_PARSE_FAILED;

  // Outermost (level 0) segment object has been constructed,
  // and pos designates start of payload.  We need to find the
  // inner (level 1) elements.

  const long long header_status = ParseHeaders();

  if (header_status < 0)  // error
    return static_cast<long>(header_status);

  if (header_status > 0)  // underflow
    return E_BUFFER_NOT_FULL;

  if (m_pInfo == NULL || m_pTracks == NULL)
    return E_FILE_FORMAT_INVALID;

  for (;;) {
    const long status = LoadCluster();

    if (status < 0)  // error
      return status;

    if (status >= 1)  // no more clusters
      return 0;
  }
}

SeekHead::Entry::Entry() : id(0), pos(0), element_start(0), element_size(0) {}

SeekHead::SeekHead(Segment* pSegment, long long start, long long size_,
                   long long element_start, long long element_size)
    : m_pSegment(pSegment),
      m_start(start),
      m_size(size_),
      m_element_start(element_start),
      m_element_size(element_size),
      m_entries(0),
      m_entry_count(0),
      m_void_elements(0),
      m_void_element_count(0) {}

SeekHead::~SeekHead() {
  delete[] m_entries;
  delete[] m_void_elements;
}

long SeekHead::Parse() {
  IMkvReader* const pReader = m_pSegment->m_pReader;

  long long pos = m_start;
  const long long stop = m_start + m_size;

  // first count the seek head entries

  int entry_count = 0;
  int void_element_count = 0;

  while (pos < stop) {
    long long id, size;

    const long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)  // error
      return status;

    if (id == libwebm::kMkvSeek)
      ++entry_count;
    else if (id == libwebm::kMkvVoid)
      ++void_element_count;

    pos += size;  // consume payload

    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;

  if (entry_count > 0) {
    m_entries = new (std::nothrow) Entry[entry_count];

    if (m_entries == NULL)
      return -1;
  }

  if (void_element_count > 0) {
    m_void_elements = new (std::nothrow) VoidElement[void_element_count];

    if (m_void_elements == NULL)
      return -1;
  }

  // now parse the entries and void elements

  Entry* pEntry = m_entries;
  VoidElement* pVoidElement = m_void_elements;

  pos = m_start;

  while (pos < stop) {
    const long long idpos = pos;

    long long id, size;

    const long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)  // error
      return status;

    if (id == libwebm::kMkvSeek && entry_count > 0) {
      if (ParseEntry(pReader, pos, size, pEntry)) {
        Entry& e = *pEntry++;

        e.element_start = idpos;
        e.element_size = (pos + size) - idpos;
      }
    } else if (id == libwebm::kMkvVoid && void_element_count > 0) {
      VoidElement& e = *pVoidElement++;

      e.element_start = idpos;
      e.element_size = (pos + size) - idpos;
    }

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;

  ptrdiff_t count_ = ptrdiff_t(pEntry - m_entries);
  assert(count_ >= 0);
  assert(count_ <= entry_count);

  m_entry_count = static_cast<int>(count_);

  count_ = ptrdiff_t(pVoidElement - m_void_elements);
  assert(count_ >= 0);
  assert(count_ <= void_element_count);

  m_void_element_count = static_cast<int>(count_);

  return 0;
}

int SeekHead::GetCount() const { return m_entry_count; }

const SeekHead::Entry* SeekHead::GetEntry(int idx) const {
  if (idx < 0)
    return 0;

  if (idx >= m_entry_count)
    return 0;

  return m_entries + idx;
}

int SeekHead::GetVoidElementCount() const { return m_void_element_count; }

const SeekHead::VoidElement* SeekHead::GetVoidElement(int idx) const {
  if (idx < 0)
    return 0;

  if (idx >= m_void_element_count)
    return 0;

  return m_void_elements + idx;
}

long Segment::ParseCues(long long off, long long& pos, long& len) {
  if (m_pCues)
    return 0;  // success

  if (off < 0)
    return -1;

  long long total, avail;

  const int status = m_pReader->Length(&total, &avail);

  if (status < 0)  // error
    return status;

  assert((total < 0) || (avail <= total));

  pos = m_start + off;

  if ((total < 0) || (pos >= total))
    return 1;  // don't bother parsing cues

  const long long element_start = pos;
  const long long segment_stop = (m_size < 0) ? -1 : m_start + m_size;

  if ((pos + 1) > avail) {
    len = 1;
    return E_BUFFER_NOT_FULL;
  }

  long long result = GetUIntLength(m_pReader, pos, len);

  if (result < 0)  // error
    return static_cast<long>(result);

  if (result > 0)  // underflow (weird)
  {
    len = 1;
    return E_BUFFER_NOT_FULL;
  }

  if ((segment_stop >= 0) && ((pos + len) > segment_stop))
    return E_FILE_FORMAT_INVALID;

  if ((pos + len) > avail)
    return E_BUFFER_NOT_FULL;

  const long long idpos = pos;

  const long long id = ReadID(m_pReader, idpos, len);

  if (id != libwebm::kMkvCues)
    return E_FILE_FORMAT_INVALID;

  pos += len;  // consume ID
  assert((segment_stop < 0) || (pos <= segment_stop));

  // Read Size

  if ((pos + 1) > avail) {
    len = 1;
    return E_BUFFER_NOT_FULL;
  }

  result = GetUIntLength(m_pReader, pos, len);

  if (result < 0)  // error
    return static_cast<long>(result);

  if (result > 0)  // underflow (weird)
  {
    len = 1;
    return E_BUFFER_NOT_FULL;
  }

  if ((segment_stop >= 0) && ((pos + len) > segment_stop))
    return E_FILE_FORMAT_INVALID;

  if ((pos + len) > avail)
    return E_BUFFER_NOT_FULL;

  const long long size = ReadUInt(m_pReader, pos, len);

  if (size < 0)  // error
    return static_cast<long>(size);

  if (size == 0)  // weird, although technically not illegal
    return 1;  // done

  pos += len;  // consume length of size of element
  assert((segment_stop < 0) || (pos <= segment_stop));

  // Pos now points to start of payload

  const long long element_stop = pos + size;

  if ((segment_stop >= 0) && (element_stop > segment_stop))
    return E_FILE_FORMAT_INVALID;

  if ((total >= 0) && (element_stop > total))
    return 1;  // don't bother parsing anymore

  len = static_cast<long>(size);

  if (element_stop > avail)
    return E_BUFFER_NOT_FULL;

  const long long element_size = element_stop - element_start;

  m_pCues =
      new (std::nothrow) Cues(this, pos, size, element_start, element_size);
  if (m_pCues == NULL)
    return -1;

  return 0;  // success
}

bool SeekHead::ParseEntry(IMkvReader* pReader, long long start, long long size_,
                          Entry* pEntry) {
  if (size_ <= 0)
    return false;

  long long pos = start;
  const long long stop = start + size_;

  long len;

  // parse the container for the level-1 element ID

  const long long seekIdId = ReadID(pReader, pos, len);
  if (seekIdId < 0)
    return false;

  if (seekIdId != libwebm::kMkvSeekID)
    return false;

  if ((pos + len) > stop)
    return false;

  pos += len;  // consume SeekID id

  const long long seekIdSize = ReadUInt(pReader, pos, len);

  if (seekIdSize <= 0)
    return false;

  if ((pos + len) > stop)
    return false;

  pos += len;  // consume size of field

  if ((pos + seekIdSize) > stop)
    return false;

  pEntry->id = ReadID(pReader, pos, len);  // payload

  if (pEntry->id <= 0)
    return false;

  if (len != seekIdSize)
    return false;

  pos += seekIdSize;  // consume SeekID payload

  const long long seekPosId = ReadID(pReader, pos, len);

  if (seekPosId != libwebm::kMkvSeekPosition)
    return false;

  if ((pos + len) > stop)
    return false;

  pos += len;  // consume id

  const long long seekPosSize = ReadUInt(pReader, pos, len);

  if (seekPosSize <= 0)
    return false;

  if ((pos + len) > stop)
    return false;

  pos += len;  // consume size

  if ((pos + seekPosSize) > stop)
    return false;

  pEntry->pos = UnserializeUInt(pReader, pos, seekPosSize);

  if (pEntry->pos < 0)
    return false;

  pos += seekPosSize;  // consume payload

  if (pos != stop)
    return false;

  return true;
}

Cues::Cues(Segment* pSegment, long long start_, long long size_,
           long long element_start, long long element_size)
    : m_pSegment(pSegment),
      m_start(start_),
      m_size(size_),
      m_element_start(element_start),
      m_element_size(element_size),
      m_cue_points(NULL),
      m_count(0),
      m_preload_count(0),
      m_pos(start_) {}

Cues::~Cues() {
  const long n = m_count + m_preload_count;

  CuePoint** p = m_cue_points;
  CuePoint** const q = p + n;

  while (p != q) {
    CuePoint* const pCP = *p++;
    assert(pCP);

    delete pCP;
  }

  delete[] m_cue_points;
}

long Cues::GetCount() const {
  if (m_cue_points == NULL)
    return -1;

  return m_count;  // TODO: really ignore preload count?
}

bool Cues::DoneParsing() const {
  const long long stop = m_start + m_size;
  return (m_pos >= stop);
}

bool Cues::Init() const {
  if (m_cue_points)
    return true;

  if (m_count != 0 || m_preload_count != 0)
    return false;

  IMkvReader* const pReader = m_pSegment->m_pReader;

  const long long stop = m_start + m_size;
  long long pos = m_start;

  long cue_points_size = 0;

  while (pos < stop) {
    const long long idpos = pos;

    long len;

    const long long id = ReadID(pReader, pos, len);
    if (id < 0 || (pos + len) > stop) {
      return false;
    }

    pos += len;  // consume ID

    const long long size = ReadUInt(pReader, pos, len);
    if (size < 0 || (pos + len > stop)) {
      return false;
    }

    pos += len;  // consume Size field
    if (pos + size > stop) {
      return false;
    }

    if (id == libwebm::kMkvCuePoint) {
      if (!PreloadCuePoint(cue_points_size, idpos))
        return false;
    }

    pos += size;  // skip payload
  }
  return true;
}

bool Cues::PreloadCuePoint(long& cue_points_size, long long pos) const {
  if (m_count != 0)
    return false;

  if (m_preload_count >= cue_points_size) {
    const long n = (cue_points_size <= 0) ? 2048 : 2 * cue_points_size;

    CuePoint** const qq = new (std::nothrow) CuePoint*[n];
    if (qq == NULL)
      return false;

    CuePoint** q = qq;  // beginning of target

    CuePoint** p = m_cue_points;  // beginning of source
    CuePoint** const pp = p + m_preload_count;  // end of source

    while (p != pp)
      *q++ = *p++;

    delete[] m_cue_points;

    m_cue_points = qq;
    cue_points_size = n;
  }

  CuePoint* const pCP = new (std::nothrow) CuePoint(m_preload_count, pos);
  if (pCP == NULL)
    return false;

  m_cue_points[m_preload_count++] = pCP;
  return true;
}

bool Cues::LoadCuePoint() const {
  const long long stop = m_start + m_size;

  if (m_pos >= stop)
    return false;  // nothing else to do

  if (!Init()) {
    m_pos = stop;
    return false;
  }

  IMkvReader* const pReader = m_pSegment->m_pReader;

  while (m_pos < stop) {
    const long long idpos = m_pos;

    long len;

    const long long id = ReadID(pReader, m_pos, len);
    if (id < 0 || (m_pos + len) > stop)
      return false;

    m_pos += len;  // consume ID

    const long long size = ReadUInt(pReader, m_pos, len);
    if (size < 0 || (m_pos + len) > stop)
      return false;

    m_pos += len;  // consume Size field
    if ((m_pos + size) > stop)
      return false;

    if (id != libwebm::kMkvCuePoint) {
      m_pos += size;  // consume payload
      if (m_pos > stop)
        return false;

      continue;
    }

    if (m_preload_count < 1)
      return false;

    CuePoint* const pCP = m_cue_points[m_count];
    if (!pCP || (pCP->GetTimeCode() < 0 && (-pCP->GetTimeCode() != idpos)))
      return false;

    if (!pCP->Load(pReader)) {
      m_pos = stop;
      return false;
    }
    ++m_count;
    --m_preload_count;

    m_pos += size;  // consume payload
    if (m_pos > stop)
      return false;

    return true;  // yes, we loaded a cue point
  }

  return false;  // no, we did not load a cue point
}

bool Cues::Find(long long time_ns, const Track* pTrack, const CuePoint*& pCP,
                const CuePoint::TrackPosition*& pTP) const {
  if (time_ns < 0 || pTrack == NULL || m_cue_points == NULL || m_count == 0)
    return false;

  CuePoint** const ii = m_cue_points;
  CuePoint** i = ii;

  CuePoint** const jj = ii + m_count;
  CuePoint** j = jj;

  pCP = *i;
  if (pCP == NULL)
    return false;

  if (time_ns <= pCP->GetTime(m_pSegment)) {
    pTP = pCP->Find(pTrack);
    return (pTP != NULL);
  }

  while (i < j) {
    // INVARIANT:
    //[ii, i) <= time_ns
    //[i, j)  ?
    //[j, jj) > time_ns

    CuePoint** const k = i + (j - i) / 2;
    if (k >= jj)
      return false;

    CuePoint* const pCP = *k;
    if (pCP == NULL)
      return false;

    const long long t = pCP->GetTime(m_pSegment);

    if (t <= time_ns)
      i = k + 1;
    else
      j = k;

    if (i > j)
      return false;
  }

  if (i != j || i > jj || i <= ii)
    return false;

  pCP = *--i;

  if (pCP == NULL || pCP->GetTime(m_pSegment) > time_ns)
    return false;

  // TODO: here and elsewhere, it's probably not correct to search
  // for the cue point with this time, and then search for a matching
  // track.  In principle, the matching track could be on some earlier
  // cue point, and with our current algorithm, we'd miss it.  To make
  // this bullet-proof, we'd need to create a secondary structure,
  // with a list of cue points that apply to a track, and then search
  // that track-based structure for a matching cue point.

  pTP = pCP->Find(pTrack);
  return (pTP != NULL);
}

const CuePoint* Cues::GetFirst() const {
  if (m_cue_points == NULL || m_count == 0)
    return NULL;

  CuePoint* const* const pp = m_cue_points;
  if (pp == NULL)
    return NULL;

  CuePoint* const pCP = pp[0];
  if (pCP == NULL || pCP->GetTimeCode() < 0)
    return NULL;

  return pCP;
}

const CuePoint* Cues::GetLast() const {
  if (m_cue_points == NULL || m_count <= 0)
    return NULL;

  const long index = m_count - 1;

  CuePoint* const* const pp = m_cue_points;
  if (pp == NULL)
    return NULL;

  CuePoint* const pCP = pp[index];
  if (pCP == NULL || pCP->GetTimeCode() < 0)
    return NULL;

  return pCP;
}

const CuePoint* Cues::GetNext(const CuePoint* pCurr) const {
  if (pCurr == NULL || pCurr->GetTimeCode() < 0 || m_cue_points == NULL ||
      m_count < 1) {
    return NULL;
  }

  long index = pCurr->m_index;
  if (index >= m_count)
    return NULL;

  CuePoint* const* const pp = m_cue_points;
  if (pp == NULL || pp[index] != pCurr)
    return NULL;

  ++index;

  if (index >= m_count)
    return NULL;

  CuePoint* const pNext = pp[index];

  if (pNext == NULL || pNext->GetTimeCode() < 0)
    return NULL;

  return pNext;
}

const BlockEntry* Cues::GetBlock(const CuePoint* pCP,
                                 const CuePoint::TrackPosition* pTP) const {
  if (pCP == NULL || pTP == NULL)
    return NULL;

  return m_pSegment->GetBlock(*pCP, *pTP);
}

const BlockEntry* Segment::GetBlock(const CuePoint& cp,
                                    const CuePoint::TrackPosition& tp) {
  Cluster** const ii = m_clusters;
  Cluster** i = ii;

  const long count = m_clusterCount + m_clusterPreloadCount;

  Cluster** const jj = ii + count;
  Cluster** j = jj;

  while (i < j) {
    // INVARIANT:
    //[ii, i) < pTP->m_pos
    //[i, j) ?
    //[j, jj)  > pTP->m_pos

    Cluster** const k = i + (j - i) / 2;
    assert(k < jj);

    Cluster* const pCluster = *k;
    assert(pCluster);

    // const long long pos_ = pCluster->m_pos;
    // assert(pos_);
    // const long long pos = pos_ * ((pos_ < 0) ? -1 : 1);

    const long long pos = pCluster->GetPosition();
    assert(pos >= 0);

    if (pos < tp.m_pos)
      i = k + 1;
    else if (pos > tp.m_pos)
      j = k;
    else
      return pCluster->GetEntry(cp, tp);
  }

  assert(i == j);
  // assert(Cluster::HasBlockEntries(this, tp.m_pos));

  Cluster* const pCluster = Cluster::Create(this, -1, tp.m_pos);  //, -1);
  if (pCluster == NULL)
    return NULL;

  const ptrdiff_t idx = i - m_clusters;

  if (!PreloadCluster(pCluster, idx)) {
    delete pCluster;
    return NULL;
  }
  assert(m_clusters);
  assert(m_clusterPreloadCount > 0);
  assert(m_clusters[idx] == pCluster);

  return pCluster->GetEntry(cp, tp);
}

const Cluster* Segment::FindOrPreloadCluster(long long requested_pos) {
  if (requested_pos < 0)
    return 0;

  Cluster** const ii = m_clusters;
  Cluster** i = ii;

  const long count = m_clusterCount + m_clusterPreloadCount;

  Cluster** const jj = ii + count;
  Cluster** j = jj;

  while (i < j) {
    // INVARIANT:
    //[ii, i) < pTP->m_pos
    //[i, j) ?
    //[j, jj)  > pTP->m_pos

    Cluster** const k = i + (j - i) / 2;
    assert(k < jj);

    Cluster* const pCluster = *k;
    assert(pCluster);

    // const long long pos_ = pCluster->m_pos;
    // assert(pos_);
    // const long long pos = pos_ * ((pos_ < 0) ? -1 : 1);

    const long long pos = pCluster->GetPosition();
    assert(pos >= 0);

    if (pos < requested_pos)
      i = k + 1;
    else if (pos > requested_pos)
      j = k;
    else
      return pCluster;
  }

  assert(i == j);
  // assert(Cluster::HasBlockEntries(this, tp.m_pos));

  Cluster* const pCluster = Cluster::Create(this, -1, requested_pos);
  if (pCluster == NULL)
    return NULL;

  const ptrdiff_t idx = i - m_clusters;

  if (!PreloadCluster(pCluster, idx)) {
    delete pCluster;
    return NULL;
  }
  assert(m_clusters);
  assert(m_clusterPreloadCount > 0);
  assert(m_clusters[idx] == pCluster);

  return pCluster;
}

CuePoint::CuePoint(long idx, long long pos)
    : m_element_start(0),
      m_element_size(0),
      m_index(idx),
      m_timecode(-1 * pos),
      m_track_positions(NULL),
      m_track_positions_count(0) {
  assert(pos > 0);
}

CuePoint::~CuePoint() { delete[] m_track_positions; }

bool CuePoint::Load(IMkvReader* pReader) {
  // odbgstream os;
  // os << "CuePoint::Load(begin): timecode=" << m_timecode << endl;

  if (m_timecode >= 0)  // already loaded
    return true;

  assert(m_track_positions == NULL);
  assert(m_track_positions_count == 0);

  long long pos_ = -m_timecode;
  const long long element_start = pos_;

  long long stop;

  {
    long len;

    const long long id = ReadID(pReader, pos_, len);
    if (id != libwebm::kMkvCuePoint)
      return false;

    pos_ += len;  // consume ID

    const long long size = ReadUInt(pReader, pos_, len);
    assert(size >= 0);

    pos_ += len;  // consume Size field
    // pos_ now points to start of payload

    stop = pos_ + size;
  }

  const long long element_size = stop - element_start;

  long long pos = pos_;

  // First count number of track positions

  while (pos < stop) {
    long len;

    const long long id = ReadID(pReader, pos, len);
    if ((id < 0) || (pos + len > stop)) {
      return false;
    }

    pos += len;  // consume ID

    const long long size = ReadUInt(pReader, pos, len);
    if ((size < 0) || (pos + len > stop)) {
      return false;
    }

    pos += len;  // consume Size field
    if ((pos + size) > stop) {
      return false;
    }

    if (id == libwebm::kMkvCueTime)
      m_timecode = UnserializeUInt(pReader, pos, size);

    else if (id == libwebm::kMkvCueTrackPositions)
      ++m_track_positions_count;

    pos += size;  // consume payload
  }

  if (m_timecode < 0 || m_track_positions_count <= 0) {
    return false;
  }

  // os << "CuePoint::Load(cont'd): idpos=" << idpos
  //   << " timecode=" << m_timecode
  //   << endl;

  m_track_positions = new (std::nothrow) TrackPosition[m_track_positions_count];
  if (m_track_positions == NULL)
    return false;

  // Now parse track positions

  TrackPosition* p = m_track_positions;
  pos = pos_;

  while (pos < stop) {
    long len;

    const long long id = ReadID(pReader, pos, len);
    if (id < 0 || (pos + len) > stop)
      return false;

    pos += len;  // consume ID

    const long long size = ReadUInt(pReader, pos, len);
    assert(size >= 0);
    assert((pos + len) <= stop);

    pos += len;  // consume Size field
    assert((pos + size) <= stop);

    if (id == libwebm::kMkvCueTrackPositions) {
      TrackPosition& tp = *p++;
      if (!tp.Parse(pReader, pos, size)) {
        return false;
      }
    }

    pos += size;  // consume payload
    if (pos > stop)
      return false;
  }

  assert(size_t(p - m_track_positions) == m_track_positions_count);

  m_element_start = element_start;
  m_element_size = element_size;

  return true;
}

bool CuePoint::TrackPosition::Parse(IMkvReader* pReader, long long start_,
                                    long long size_) {
  const long long stop = start_ + size_;
  long long pos = start_;

  m_track = -1;
  m_pos = -1;
  m_block = 1;  // default

  while (pos < stop) {
    long len;

    const long long id = ReadID(pReader, pos, len);
    if ((id < 0) || ((pos + len) > stop)) {
      return false;
    }

    pos += len;  // consume ID

    const long long size = ReadUInt(pReader, pos, len);
    if ((size < 0) || ((pos + len) > stop)) {
      return false;
    }

    pos += len;  // consume Size field
    if ((pos + size) > stop) {
      return false;
    }

    if (id == libwebm::kMkvCueTrack)
      m_track = UnserializeUInt(pReader, pos, size);
    else if (id == libwebm::kMkvCueClusterPosition)
      m_pos = UnserializeUInt(pReader, pos, size);
    else if (id == libwebm::kMkvCueBlockNumber)
      m_block = UnserializeUInt(pReader, pos, size);

    pos += size;  // consume payload
  }

  if ((m_pos < 0) || (m_track <= 0)) {
    return false;
  }

  return true;
}

const CuePoint::TrackPosition* CuePoint::Find(const Track* pTrack) const {
  if (pTrack == NULL) {
    return NULL;
  }

  const long long n = pTrack->GetNumber();

  const TrackPosition* i = m_track_positions;
  const TrackPosition* const j = i + m_track_positions_count;

  while (i != j) {
    const TrackPosition& p = *i++;

    if (p.m_track == n)
      return &p;
  }

  return NULL;  // no matching track number found
}

long long CuePoint::GetTimeCode() const { return m_timecode; }

long long CuePoint::GetTime(const Segment* pSegment) const {
  assert(pSegment);
  assert(m_timecode >= 0);

  const SegmentInfo* const pInfo = pSegment->GetInfo();
  assert(pInfo);

  const long long scale = pInfo->GetTimeCodeScale();
  assert(scale >= 1);

  const long long time = scale * m_timecode;

  return time;
}

bool Segment::DoneParsing() const {
  if (m_size < 0) {
    long long total, avail;

    const int status = m_pReader->Length(&total, &avail);

    if (status < 0)  // error
      return true;  // must assume done

    if (total < 0)
      return false;  // assume live stream

    return (m_pos >= total);
  }

  const long long stop = m_start + m_size;

  return (m_pos >= stop);
}

const Cluster* Segment::GetFirst() const {
  if ((m_clusters == NULL) || (m_clusterCount <= 0))
    return &m_eos;

  Cluster* const pCluster = m_clusters[0];
  assert(pCluster);

  return pCluster;
}

const Cluster* Segment::GetLast() const {
  if ((m_clusters == NULL) || (m_clusterCount <= 0))
    return &m_eos;

  const long idx = m_clusterCount - 1;

  Cluster* const pCluster = m_clusters[idx];
  assert(pCluster);

  return pCluster;
}

unsigned long Segment::GetCount() const { return m_clusterCount; }

const Cluster* Segment::GetNext(const Cluster* pCurr) {
  assert(pCurr);
  assert(pCurr != &m_eos);
  assert(m_clusters);

  long idx = pCurr->m_index;

  if (idx >= 0) {
    assert(m_clusterCount > 0);
    assert(idx < m_clusterCount);
    assert(pCurr == m_clusters[idx]);

    ++idx;

    if (idx >= m_clusterCount)
      return &m_eos;  // caller will LoadCluster as desired

    Cluster* const pNext = m_clusters[idx];
    assert(pNext);
    assert(pNext->m_index >= 0);
    assert(pNext->m_index == idx);

    return pNext;
  }

  assert(m_clusterPreloadCount > 0);

  long long pos = pCurr->m_element_start;

  assert(m_size >= 0);  // TODO
  const long long stop = m_start + m_size;  // end of segment

  {
    long len;

    long long result = GetUIntLength(m_pReader, pos, len);
    assert(result == 0);
    assert((pos + len) <= stop);  // TODO
    if (result != 0)
      return NULL;

    const long long id = ReadID(m_pReader, pos, len);
    if (id != libwebm::kMkvCluster)
      return NULL;

    pos += len;  // consume ID

    // Read Size
    result = GetUIntLength(m_pReader, pos, len);
    assert(result == 0);  // TODO
    assert((pos + len) <= stop);  // TODO

    const long long size = ReadUInt(m_pReader, pos, len);
    assert(size > 0);  // TODO
    // assert((pCurr->m_size <= 0) || (pCurr->m_size == size));

    pos += len;  // consume length of size of element
    assert((pos + size) <= stop);  // TODO

    // Pos now points to start of payload

    pos += size;  // consume payload
  }

  long long off_next = 0;

  while (pos < stop) {
    long len;

    long long result = GetUIntLength(m_pReader, pos, len);
    assert(result == 0);
    assert((pos + len) <= stop);  // TODO
    if (result != 0)
      return NULL;

    const long long idpos = pos;  // pos of next (potential) cluster

    const long long id = ReadID(m_pReader, idpos, len);
    if (id < 0)
      return NULL;

    pos += len;  // consume ID

    // Read Size
    result = GetUIntLength(m_pReader, pos, len);
    assert(result == 0);  // TODO
    assert((pos + len) <= stop);  // TODO

    const long long size = ReadUInt(m_pReader, pos, len);
    assert(size >= 0);  // TODO

    pos += len;  // consume length of size of element
    assert((pos + size) <= stop);  // TODO

    // Pos now points to start of payload

    if (size == 0)  // weird
      continue;

    if (id == libwebm::kMkvCluster) {
      const long long off_next_ = idpos - m_start;

      long long pos_;
      long len_;

      const long status = Cluster::HasBlockEntries(this, off_next_, pos_, len_);

      assert(status >= 0);

      if (status > 0) {
        off_next = off_next_;
        break;
      }
    }

    pos += size;  // consume payload
  }

  if (off_next <= 0)
    return 0;

  Cluster** const ii = m_clusters + m_clusterCount;
  Cluster** i = ii;

  Cluster** const jj = ii + m_clusterPreloadCount;
  Cluster** j = jj;

  while (i < j) {
    // INVARIANT:
    //[0, i) < pos_next
    //[i, j) ?
    //[j, jj)  > pos_next

    Cluster** const k = i + (j - i) / 2;
    assert(k < jj);

    Cluster* const pNext = *k;
    assert(pNext);
    assert(pNext->m_index < 0);

    // const long long pos_ = pNext->m_pos;
    // assert(pos_);
    // pos = pos_ * ((pos_ < 0) ? -1 : 1);

    pos = pNext->GetPosition();

    if (pos < off_next)
      i = k + 1;
    else if (pos > off_next)
      j = k;
    else
      return pNext;
  }

  assert(i == j);

  Cluster* const pNext = Cluster::Create(this, -1, off_next);
  if (pNext == NULL)
    return NULL;

  const ptrdiff_t idx_next = i - m_clusters;  // insertion position

  if (!PreloadCluster(pNext, idx_next)) {
    delete pNext;
    return NULL;
  }
  assert(m_clusters);
  assert(idx_next < m_clusterSize);
  assert(m_clusters[idx_next] == pNext);

  return pNext;
}

long Segment::ParseNext(const Cluster* pCurr, const Cluster*& pResult,
                        long long& pos, long& len) {
  assert(pCurr);
  assert(!pCurr->EOS());
  assert(m_clusters);

  pResult = 0;

  if (pCurr->m_index >= 0) {  // loaded (not merely preloaded)
    assert(m_clusters[pCurr->m_index] == pCurr);

    const long next_idx = pCurr->m_index + 1;

    if (next_idx < m_clusterCount) {
      pResult = m_clusters[next_idx];
      return 0;  // success
    }

    // curr cluster is last among loaded

    const long result = LoadCluster(pos, len);

    if (result < 0)  // error or underflow
      return result;

    if (result > 0)  // no more clusters
    {
      // pResult = &m_eos;
      return 1;
    }

    pResult = GetLast();
    return 0;  // success
  }

  assert(m_pos > 0);

  long long total, avail;

  long status = m_pReader->Length(&total, &avail);

  if (status < 0)  // error
    return status;

  assert((total < 0) || (avail <= total));

  const long long segment_stop = (m_size < 0) ? -1 : m_start + m_size;

  // interrogate curr cluster

  pos = pCurr->m_element_start;

  if (pCurr->m_element_size >= 0)
    pos += pCurr->m_element_size;
  else {
    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    long long result = GetUIntLength(m_pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)  // weird
      return E_BUFFER_NOT_FULL;

    if ((segment_stop >= 0) && ((pos + len) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long id = ReadUInt(m_pReader, pos, len);

    if (id != libwebm::kMkvCluster)
      return -1;

    pos += len;  // consume ID

    // Read Size

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    result = GetUIntLength(m_pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)  // weird
      return E_BUFFER_NOT_FULL;

    if ((segment_stop >= 0) && ((pos + len) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long size = ReadUInt(m_pReader, pos, len);

    if (size < 0)  // error
      return static_cast<long>(size);

    pos += len;  // consume size field

    const long long unknown_size = (1LL << (7 * len)) - 1;

    if (size == unknown_size)  // TODO: should never happen
      return E_FILE_FORMAT_INVALID;  // TODO: resolve this

    // assert((pCurr->m_size <= 0) || (pCurr->m_size == size));

    if ((segment_stop >= 0) && ((pos + size) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    // Pos now points to start of payload

    pos += size;  // consume payload (that is, the current cluster)
    if (segment_stop >= 0 && pos > segment_stop)
      return E_FILE_FORMAT_INVALID;

    // By consuming the payload, we are assuming that the curr
    // cluster isn't interesting.  That is, we don't bother checking
    // whether the payload of the curr cluster is less than what
    // happens to be available (obtained via IMkvReader::Length).
    // Presumably the caller has already dispensed with the current
    // cluster, and really does want the next cluster.
  }

  // pos now points to just beyond the last fully-loaded cluster

  for (;;) {
    const long status = DoParseNext(pResult, pos, len);

    if (status <= 1)
      return status;
  }
}

long Segment::DoParseNext(const Cluster*& pResult, long long& pos, long& len) {
  long long total, avail;

  long status = m_pReader->Length(&total, &avail);

  if (status < 0)  // error
    return status;

  assert((total < 0) || (avail <= total));

  const long long segment_stop = (m_size < 0) ? -1 : m_start + m_size;

  // Parse next cluster.  This is strictly a parsing activity.
  // Creation of a new cluster object happens later, after the
  // parsing is done.

  long long off_next = 0;
  long long cluster_size = -1;

  for (;;) {
    if ((total >= 0) && (pos >= total))
      return 1;  // EOF

    if ((segment_stop >= 0) && (pos >= segment_stop))
      return 1;  // EOF

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    long long result = GetUIntLength(m_pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)  // weird
      return E_BUFFER_NOT_FULL;

    if ((segment_stop >= 0) && ((pos + len) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long idpos = pos;  // absolute
    const long long idoff = pos - m_start;  // relative

    const long long id = ReadID(m_pReader, idpos, len);  // absolute

    if (id < 0)  // error
      return static_cast<long>(id);

    if (id == 0)  // weird
      return -1;  // generic error

    pos += len;  // consume ID

    // Read Size

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    result = GetUIntLength(m_pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)  // weird
      return E_BUFFER_NOT_FULL;

    if ((segment_stop >= 0) && ((pos + len) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long size = ReadUInt(m_pReader, pos, len);

    if (size < 0)  // error
      return static_cast<long>(size);

    pos += len;  // consume length of size of element

    // Pos now points to start of payload

    if (size == 0)  // weird
      continue;

    const long long unknown_size = (1LL << (7 * len)) - 1;

    if ((segment_stop >= 0) && (size != unknown_size) &&
        ((pos + size) > segment_stop)) {
      return E_FILE_FORMAT_INVALID;
    }

    if (id == libwebm::kMkvCues) {
      if (size == unknown_size)
        return E_FILE_FORMAT_INVALID;

      const long long element_stop = pos + size;

      if ((segment_stop >= 0) && (element_stop > segment_stop))
        return E_FILE_FORMAT_INVALID;

      const long long element_start = idpos;
      const long long element_size = element_stop - element_start;

      if (m_pCues == NULL) {
        m_pCues = new (std::nothrow)
            Cues(this, pos, size, element_start, element_size);
        if (m_pCues == NULL)
          return false;
      }

      pos += size;  // consume payload
      if (segment_stop >= 0 && pos > segment_stop)
        return E_FILE_FORMAT_INVALID;

      continue;
    }

    if (id != libwebm::kMkvCluster) {  // not a Cluster ID
      if (size == unknown_size)
        return E_FILE_FORMAT_INVALID;

      pos += size;  // consume payload
      if (segment_stop >= 0 && pos > segment_stop)
        return E_FILE_FORMAT_INVALID;

      continue;
    }

    // We have a cluster.
    off_next = idoff;

    if (size != unknown_size)
      cluster_size = size;

    break;
  }

  assert(off_next > 0);  // have cluster

  // We have parsed the next cluster.
  // We have not created a cluster object yet.  What we need
  // to do now is determine whether it has already be preloaded
  //(in which case, an object for this cluster has already been
  // created), and if not, create a new cluster object.

  Cluster** const ii = m_clusters + m_clusterCount;
  Cluster** i = ii;

  Cluster** const jj = ii + m_clusterPreloadCount;
  Cluster** j = jj;

  while (i < j) {
    // INVARIANT:
    //[0, i) < pos_next
    //[i, j) ?
    //[j, jj)  > pos_next

    Cluster** const k = i + (j - i) / 2;
    assert(k < jj);

    const Cluster* const pNext = *k;
    assert(pNext);
    assert(pNext->m_index < 0);

    pos = pNext->GetPosition();
    assert(pos >= 0);

    if (pos < off_next)
      i = k + 1;
    else if (pos > off_next)
      j = k;
    else {
      pResult = pNext;
      return 0;  // success
    }
  }

  assert(i == j);

  long long pos_;
  long len_;

  status = Cluster::HasBlockEntries(this, off_next, pos_, len_);

  if (status < 0) {  // error or underflow
    pos = pos_;
    len = len_;

    return status;
  }

  if (status > 0) {  // means "found at least one block entry"
    Cluster* const pNext = Cluster::Create(this,
                                           -1,  // preloaded
                                           off_next);
    if (pNext == NULL)
      return -1;

    const ptrdiff_t idx_next = i - m_clusters;  // insertion position

    if (!PreloadCluster(pNext, idx_next)) {
      delete pNext;
      return -1;
    }
    assert(m_clusters);
    assert(idx_next < m_clusterSize);
    assert(m_clusters[idx_next] == pNext);

    pResult = pNext;
    return 0;  // success
  }

  // status == 0 means "no block entries found"

  if (cluster_size < 0) {  // unknown size
    const long long payload_pos = pos;  // absolute pos of cluster payload

    for (;;) {  // determine cluster size
      if ((total >= 0) && (pos >= total))
        break;

      if ((segment_stop >= 0) && (pos >= segment_stop))
        break;  // no more clusters

      // Read ID

      if ((pos + 1) > avail) {
        len = 1;
        return E_BUFFER_NOT_FULL;
      }

      long long result = GetUIntLength(m_pReader, pos, len);

      if (result < 0)  // error
        return static_cast<long>(result);

      if (result > 0)  // weird
        return E_BUFFER_NOT_FULL;

      if ((segment_stop >= 0) && ((pos + len) > segment_stop))
        return E_FILE_FORMAT_INVALID;

      if ((pos + len) > avail)
        return E_BUFFER_NOT_FULL;

      const long long idpos = pos;
      const long long id = ReadID(m_pReader, idpos, len);

      if (id < 0)  // error (or underflow)
        return static_cast<long>(id);

      // This is the distinguished set of ID's we use to determine
      // that we have exhausted the sub-element's inside the cluster
      // whose ID we parsed earlier.

      if (id == libwebm::kMkvCluster || id == libwebm::kMkvCues)
        break;

      pos += len;  // consume ID (of sub-element)

      // Read Size

      if ((pos + 1) > avail) {
        len = 1;
        return E_BUFFER_NOT_FULL;
      }

      result = GetUIntLength(m_pReader, pos, len);

      if (result < 0)  // error
        return static_cast<long>(result);

      if (result > 0)  // weird
        return E_BUFFER_NOT_FULL;

      if ((segment_stop >= 0) && ((pos + len) > segment_stop))
        return E_FILE_FORMAT_INVALID;

      if ((pos + len) > avail)
        return E_BUFFER_NOT_FULL;

      const long long size = ReadUInt(m_pReader, pos, len);

      if (size < 0)  // error
        return static_cast<long>(size);

      pos += len;  // consume size field of element

      // pos now points to start of sub-element's payload

      if (size == 0)  // weird
        continue;

      const long long unknown_size = (1LL << (7 * len)) - 1;

      if (size == unknown_size)
        return E_FILE_FORMAT_INVALID;  // not allowed for sub-elements

      if ((segment_stop >= 0) && ((pos + size) > segment_stop))  // weird
        return E_FILE_FORMAT_INVALID;

      pos += size;  // consume payload of sub-element
      if (segment_stop >= 0 && pos > segment_stop)
        return E_FILE_FORMAT_INVALID;
    }  // determine cluster size

    cluster_size = pos - payload_pos;
    assert(cluster_size >= 0);  // TODO: handle cluster_size = 0

    pos = payload_pos;  // reset and re-parse original cluster
  }

  pos += cluster_size;  // consume payload
  if (segment_stop >= 0 && pos > segment_stop)
    return E_FILE_FORMAT_INVALID;

  return 2;  // try to find a cluster that follows next
}

const Cluster* Segment::FindCluster(long long time_ns) const {
  if ((m_clusters == NULL) || (m_clusterCount <= 0))
    return &m_eos;

  {
    Cluster* const pCluster = m_clusters[0];
    assert(pCluster);
    assert(pCluster->m_index == 0);

    if (time_ns <= pCluster->GetTime())
      return pCluster;
  }

  // Binary search of cluster array

  long i = 0;
  long j = m_clusterCount;

  while (i < j) {
    // INVARIANT:
    //[0, i) <= time_ns
    //[i, j) ?
    //[j, m_clusterCount)  > time_ns

    const long k = i + (j - i) / 2;
    assert(k < m_clusterCount);

    Cluster* const pCluster = m_clusters[k];
    assert(pCluster);
    assert(pCluster->m_index == k);

    const long long t = pCluster->GetTime();

    if (t <= time_ns)
      i = k + 1;
    else
      j = k;

    assert(i <= j);
  }

  assert(i == j);
  assert(i > 0);
  assert(i <= m_clusterCount);

  const long k = i - 1;

  Cluster* const pCluster = m_clusters[k];
  assert(pCluster);
  assert(pCluster->m_index == k);
  assert(pCluster->GetTime() <= time_ns);

  return pCluster;
}

const Tracks* Segment::GetTracks() const { return m_pTracks; }
const SegmentInfo* Segment::GetInfo() const { return m_pInfo; }
const Cues* Segment::GetCues() const { return m_pCues; }
const Chapters* Segment::GetChapters() const { return m_pChapters; }
const Tags* Segment::GetTags() const { return m_pTags; }
const SeekHead* Segment::GetSeekHead() const { return m_pSeekHead; }

long long Segment::GetDuration() const {
  assert(m_pInfo);
  return m_pInfo->GetDuration();
}

Chapters::Chapters(Segment* pSegment, long long payload_start,
                   long long payload_size, long long element_start,
                   long long element_size)
    : m_pSegment(pSegment),
      m_start(payload_start),
      m_size(payload_size),
      m_element_start(element_start),
      m_element_size(element_size),
      m_editions(NULL),
      m_editions_size(0),
      m_editions_count(0) {}

Chapters::~Chapters() {
  while (m_editions_count > 0) {
    Edition& e = m_editions[--m_editions_count];
    e.Clear();
  }
  delete[] m_editions;
}

long Chapters::Parse() {
  IMkvReader* const pReader = m_pSegment->m_pReader;

  long long pos = m_start;  // payload start
  const long long stop = pos + m_size;  // payload stop

  while (pos < stop) {
    long long id, size;

    long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)  // error
      return status;

    if (size == 0)  // weird
      continue;

    if (id == libwebm::kMkvEditionEntry) {
      status = ParseEdition(pos, size);

      if (status < 0)  // error
        return status;
    }

    pos += size;
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;
  return 0;
}

int Chapters::GetEditionCount() const { return m_editions_count; }

const Chapters::Edition* Chapters::GetEdition(int idx) const {
  if (idx < 0)
    return NULL;

  if (idx >= m_editions_count)
    return NULL;

  return m_editions + idx;
}

bool Chapters::ExpandEditionsArray() {
  if (m_editions_size > m_editions_count)
    return true;  // nothing else to do

  const int size = (m_editions_size == 0) ? 1 : 2 * m_editions_size;

  Edition* const editions = new (std::nothrow) Edition[size];

  if (editions == NULL)
    return false;

  for (int idx = 0; idx < m_editions_count; ++idx) {
    m_editions[idx].ShallowCopy(editions[idx]);
  }

  delete[] m_editions;
  m_editions = editions;

  m_editions_size = size;
  return true;
}

long Chapters::ParseEdition(long long pos, long long size) {
  if (!ExpandEditionsArray())
    return -1;

  Edition& e = m_editions[m_editions_count++];
  e.Init();

  return e.Parse(m_pSegment->m_pReader, pos, size);
}

Chapters::Edition::Edition() {}

Chapters::Edition::~Edition() {}

int Chapters::Edition::GetAtomCount() const { return m_atoms_count; }

const Chapters::Atom* Chapters::Edition::GetAtom(int index) const {
  if (index < 0)
    return NULL;

  if (index >= m_atoms_count)
    return NULL;

  return m_atoms + index;
}

void Chapters::Edition::Init() {
  m_atoms = NULL;
  m_atoms_size = 0;
  m_atoms_count = 0;
}

void Chapters::Edition::ShallowCopy(Edition& rhs) const {
  rhs.m_atoms = m_atoms;
  rhs.m_atoms_size = m_atoms_size;
  rhs.m_atoms_count = m_atoms_count;
}

void Chapters::Edition::Clear() {
  while (m_atoms_count > 0) {
    Atom& a = m_atoms[--m_atoms_count];
    a.Clear();
  }

  delete[] m_atoms;
  m_atoms = NULL;

  m_atoms_size = 0;
}

long Chapters::Edition::Parse(IMkvReader* pReader, long long pos,
                              long long size) {
  const long long stop = pos + size;

  while (pos < stop) {
    long long id, size;

    long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)  // error
      return status;

    if (size == 0)
      continue;

    if (id == libwebm::kMkvChapterAtom) {
      status = ParseAtom(pReader, pos, size);

      if (status < 0)  // error
        return status;
    }

    pos += size;
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;
  return 0;
}

long Chapters::Edition::ParseAtom(IMkvReader* pReader, long long pos,
                                  long long size) {
  if (!ExpandAtomsArray())
    return -1;

  Atom& a = m_atoms[m_atoms_count++];
  a.Init();

  return a.Parse(pReader, pos, size);
}

bool Chapters::Edition::ExpandAtomsArray() {
  if (m_atoms_size > m_atoms_count)
    return true;  // nothing else to do

  const int size = (m_atoms_size == 0) ? 1 : 2 * m_atoms_size;

  Atom* const atoms = new (std::nothrow) Atom[size];

  if (atoms == NULL)
    return false;

  for (int idx = 0; idx < m_atoms_count; ++idx) {
    m_atoms[idx].ShallowCopy(atoms[idx]);
  }

  delete[] m_atoms;
  m_atoms = atoms;

  m_atoms_size = size;
  return true;
}

Chapters::Atom::Atom() {}

Chapters::Atom::~Atom() {}

unsigned long long Chapters::Atom::GetUID() const { return m_uid; }

const char* Chapters::Atom::GetStringUID() const { return m_string_uid; }

long long Chapters::Atom::GetStartTimecode() const { return m_start_timecode; }

long long Chapters::Atom::GetStopTimecode() const { return m_stop_timecode; }

long long Chapters::Atom::GetStartTime(const Chapters* pChapters) const {
  return GetTime(pChapters, m_start_timecode);
}

long long Chapters::Atom::GetStopTime(const Chapters* pChapters) const {
  return GetTime(pChapters, m_stop_timecode);
}

int Chapters::Atom::GetDisplayCount() const { return m_displays_count; }

const Chapters::Display* Chapters::Atom::GetDisplay(int index) const {
  if (index < 0)
    return NULL;

  if (index >= m_displays_count)
    return NULL;

  return m_displays + index;
}

void Chapters::Atom::Init() {
  m_string_uid = NULL;
  m_uid = 0;
  m_start_timecode = -1;
  m_stop_timecode = -1;

  m_displays = NULL;
  m_displays_size = 0;
  m_displays_count = 0;
}

void Chapters::Atom::ShallowCopy(Atom& rhs) const {
  rhs.m_string_uid = m_string_uid;
  rhs.m_uid = m_uid;
  rhs.m_start_timecode = m_start_timecode;
  rhs.m_stop_timecode = m_stop_timecode;

  rhs.m_displays = m_displays;
  rhs.m_displays_size = m_displays_size;
  rhs.m_displays_count = m_displays_count;
}

void Chapters::Atom::Clear() {
  delete[] m_string_uid;
  m_string_uid = NULL;

  while (m_displays_count > 0) {
    Display& d = m_displays[--m_displays_count];
    d.Clear();
  }

  delete[] m_displays;
  m_displays = NULL;

  m_displays_size = 0;
}

long Chapters::Atom::Parse(IMkvReader* pReader, long long pos, long long size) {
  const long long stop = pos + size;

  while (pos < stop) {
    long long id, size;

    long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)  // error
      return status;

    if (size == 0)  // 0 length payload, skip.
      continue;

    if (id == libwebm::kMkvChapterDisplay) {
      status = ParseDisplay(pReader, pos, size);

      if (status < 0)  // error
        return status;
    } else if (id == libwebm::kMkvChapterStringUID) {
      status = UnserializeString(pReader, pos, size, m_string_uid);

      if (status < 0)  // error
        return status;
    } else if (id == libwebm::kMkvChapterUID) {
      long long val;
      status = UnserializeInt(pReader, pos, size, val);

      if (status < 0)  // error
        return status;

      m_uid = static_cast<unsigned long long>(val);
    } else if (id == libwebm::kMkvChapterTimeStart) {
      const long long val = UnserializeUInt(pReader, pos, size);

      if (val < 0)  // error
        return static_cast<long>(val);

      m_start_timecode = val;
    } else if (id == libwebm::kMkvChapterTimeEnd) {
      const long long val = UnserializeUInt(pReader, pos, size);

      if (val < 0)  // error
        return static_cast<long>(val);

      m_stop_timecode = val;
    }

    pos += size;
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;
  return 0;
}

long long Chapters::Atom::GetTime(const Chapters* pChapters,
                                  long long timecode) {
  if (pChapters == NULL)
    return -1;

  Segment* const pSegment = pChapters->m_pSegment;

  if (pSegment == NULL)  // weird
    return -1;

  const SegmentInfo* const pInfo = pSegment->GetInfo();

  if (pInfo == NULL)
    return -1;

  const long long timecode_scale = pInfo->GetTimeCodeScale();

  if (timecode_scale < 1)  // weird
    return -1;

  if (timecode < 0)
    return -1;

  const long long result = timecode_scale * timecode;

  return result;
}

long Chapters::Atom::ParseDisplay(IMkvReader* pReader, long long pos,
                                  long long size) {
  if (!ExpandDisplaysArray())
    return -1;

  Display& d = m_displays[m_displays_count++];
  d.Init();

  return d.Parse(pReader, pos, size);
}

bool Chapters::Atom::ExpandDisplaysArray() {
  if (m_displays_size > m_displays_count)
    return true;  // nothing else to do

  const int size = (m_displays_size == 0) ? 1 : 2 * m_displays_size;

  Display* const displays = new (std::nothrow) Display[size];

  if (displays == NULL)
    return false;

  for (int idx = 0; idx < m_displays_count; ++idx) {
    m_displays[idx].ShallowCopy(displays[idx]);
  }

  delete[] m_displays;
  m_displays = displays;

  m_displays_size = size;
  return true;
}

Chapters::Display::Display() {}

Chapters::Display::~Display() {}

const char* Chapters::Display::GetString() const { return m_string; }

const char* Chapters::Display::GetLanguage() const { return m_language; }

const char* Chapters::Display::GetCountry() const { return m_country; }

void Chapters::Display::Init() {
  m_string = NULL;
  m_language = NULL;
  m_country = NULL;
}

void Chapters::Display::ShallowCopy(Display& rhs) const {
  rhs.m_string = m_string;
  rhs.m_language = m_language;
  rhs.m_country = m_country;
}

void Chapters::Display::Clear() {
  delete[] m_string;
  m_string = NULL;

  delete[] m_language;
  m_language = NULL;

  delete[] m_country;
  m_country = NULL;
}

long Chapters::Display::Parse(IMkvReader* pReader, long long pos,
                              long long size) {
  const long long stop = pos + size;

  while (pos < stop) {
    long long id, size;

    long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)  // error
      return status;

    if (size == 0)  // No payload.
      continue;

    if (id == libwebm::kMkvChapString) {
      status = UnserializeString(pReader, pos, size, m_string);

      if (status)
        return status;
    } else if (id == libwebm::kMkvChapLanguage) {
      status = UnserializeString(pReader, pos, size, m_language);

      if (status)
        return status;
    } else if (id == libwebm::kMkvChapCountry) {
      status = UnserializeString(pReader, pos, size, m_country);

      if (status)
        return status;
    }

    pos += size;
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;
  return 0;
}

Tags::Tags(Segment* pSegment, long long payload_start, long long payload_size,
           long long element_start, long long element_size)
    : m_pSegment(pSegment),
      m_start(payload_start),
      m_size(payload_size),
      m_element_start(element_start),
      m_element_size(element_size),
      m_tags(NULL),
      m_tags_size(0),
      m_tags_count(0) {}

Tags::~Tags() {
  while (m_tags_count > 0) {
    Tag& t = m_tags[--m_tags_count];
    t.Clear();
  }
  delete[] m_tags;
}

long Tags::Parse() {
  IMkvReader* const pReader = m_pSegment->m_pReader;

  long long pos = m_start;  // payload start
  const long long stop = pos + m_size;  // payload stop

  while (pos < stop) {
    long long id, size;

    long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)
      return status;

    if (size == 0)  // 0 length tag, read another
      continue;

    if (id == libwebm::kMkvTag) {
      status = ParseTag(pos, size);

      if (status < 0)
        return status;
    }

    pos += size;
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;

  return 0;
}

int Tags::GetTagCount() const { return m_tags_count; }

const Tags::Tag* Tags::GetTag(int idx) const {
  if (idx < 0)
    return NULL;

  if (idx >= m_tags_count)
    return NULL;

  return m_tags + idx;
}

bool Tags::ExpandTagsArray() {
  if (m_tags_size > m_tags_count)
    return true;  // nothing else to do

  const int size = (m_tags_size == 0) ? 1 : 2 * m_tags_size;

  Tag* const tags = new (std::nothrow) Tag[size];

  if (tags == NULL)
    return false;

  for (int idx = 0; idx < m_tags_count; ++idx) {
    m_tags[idx].ShallowCopy(tags[idx]);
  }

  delete[] m_tags;
  m_tags = tags;

  m_tags_size = size;
  return true;
}

long Tags::ParseTag(long long pos, long long size) {
  if (!ExpandTagsArray())
    return -1;

  Tag& t = m_tags[m_tags_count++];
  t.Init();

  return t.Parse(m_pSegment->m_pReader, pos, size);
}

Tags::Tag::Tag() {}

Tags::Tag::~Tag() {}

int Tags::Tag::GetSimpleTagCount() const { return m_simple_tags_count; }

const Tags::SimpleTag* Tags::Tag::GetSimpleTag(int index) const {
  if (index < 0)
    return NULL;

  if (index >= m_simple_tags_count)
    return NULL;

  return m_simple_tags + index;
}

void Tags::Tag::Init() {
  m_simple_tags = NULL;
  m_simple_tags_size = 0;
  m_simple_tags_count = 0;
}

void Tags::Tag::ShallowCopy(Tag& rhs) const {
  rhs.m_simple_tags = m_simple_tags;
  rhs.m_simple_tags_size = m_simple_tags_size;
  rhs.m_simple_tags_count = m_simple_tags_count;
}

void Tags::Tag::Clear() {
  while (m_simple_tags_count > 0) {
    SimpleTag& d = m_simple_tags[--m_simple_tags_count];
    d.Clear();
  }

  delete[] m_simple_tags;
  m_simple_tags = NULL;

  m_simple_tags_size = 0;
}

long Tags::Tag::Parse(IMkvReader* pReader, long long pos, long long size) {
  const long long stop = pos + size;

  while (pos < stop) {
    long long id, size;

    long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)
      return status;

    if (size == 0)  // 0 length tag, read another
      continue;

    if (id == libwebm::kMkvSimpleTag) {
      status = ParseSimpleTag(pReader, pos, size);

      if (status < 0)
        return status;
    }

    pos += size;
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;
  return 0;
}

long Tags::Tag::ParseSimpleTag(IMkvReader* pReader, long long pos,
                               long long size) {
  if (!ExpandSimpleTagsArray())
    return -1;

  SimpleTag& st = m_simple_tags[m_simple_tags_count++];
  st.Init();

  return st.Parse(pReader, pos, size);
}

bool Tags::Tag::ExpandSimpleTagsArray() {
  if (m_simple_tags_size > m_simple_tags_count)
    return true;  // nothing else to do

  const int size = (m_simple_tags_size == 0) ? 1 : 2 * m_simple_tags_size;

  SimpleTag* const displays = new (std::nothrow) SimpleTag[size];

  if (displays == NULL)
    return false;

  for (int idx = 0; idx < m_simple_tags_count; ++idx) {
    m_simple_tags[idx].ShallowCopy(displays[idx]);
  }

  delete[] m_simple_tags;
  m_simple_tags = displays;

  m_simple_tags_size = size;
  return true;
}

Tags::SimpleTag::SimpleTag() {}

Tags::SimpleTag::~SimpleTag() {}

const char* Tags::SimpleTag::GetTagName() const { return m_tag_name; }

const char* Tags::SimpleTag::GetTagString() const { return m_tag_string; }

void Tags::SimpleTag::Init() {
  m_tag_name = NULL;
  m_tag_string = NULL;
}

void Tags::SimpleTag::ShallowCopy(SimpleTag& rhs) const {
  rhs.m_tag_name = m_tag_name;
  rhs.m_tag_string = m_tag_string;
}

void Tags::SimpleTag::Clear() {
  delete[] m_tag_name;
  m_tag_name = NULL;

  delete[] m_tag_string;
  m_tag_string = NULL;
}

long Tags::SimpleTag::Parse(IMkvReader* pReader, long long pos,
                            long long size) {
  const long long stop = pos + size;

  while (pos < stop) {
    long long id, size;

    long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)  // error
      return status;

    if (size == 0)  // weird
      continue;

    if (id == libwebm::kMkvTagName) {
      status = UnserializeString(pReader, pos, size, m_tag_name);

      if (status)
        return status;
    } else if (id == libwebm::kMkvTagString) {
      status = UnserializeString(pReader, pos, size, m_tag_string);

      if (status)
        return status;
    }

    pos += size;
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;
  return 0;
}

SegmentInfo::SegmentInfo(Segment* pSegment, long long start, long long size_,
                         long long element_start, long long element_size)
    : m_pSegment(pSegment),
      m_start(start),
      m_size(size_),
      m_element_start(element_start),
      m_element_size(element_size),
      m_pMuxingAppAsUTF8(NULL),
      m_pWritingAppAsUTF8(NULL),
      m_pTitleAsUTF8(NULL) {}

SegmentInfo::~SegmentInfo() {
  delete[] m_pMuxingAppAsUTF8;
  m_pMuxingAppAsUTF8 = NULL;

  delete[] m_pWritingAppAsUTF8;
  m_pWritingAppAsUTF8 = NULL;

  delete[] m_pTitleAsUTF8;
  m_pTitleAsUTF8 = NULL;
}

long SegmentInfo::Parse() {
  assert(m_pMuxingAppAsUTF8 == NULL);
  assert(m_pWritingAppAsUTF8 == NULL);
  assert(m_pTitleAsUTF8 == NULL);

  IMkvReader* const pReader = m_pSegment->m_pReader;

  long long pos = m_start;
  const long long stop = m_start + m_size;

  m_timecodeScale = 1000000;
  m_duration = -1;

  while (pos < stop) {
    long long id, size;

    const long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)  // error
      return status;

    if (id == libwebm::kMkvTimecodeScale) {
      m_timecodeScale = UnserializeUInt(pReader, pos, size);

      if (m_timecodeScale <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvDuration) {
      const long status = UnserializeFloat(pReader, pos, size, m_duration);

      if (status < 0)
        return status;

      if (m_duration < 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvMuxingApp) {
      const long status =
          UnserializeString(pReader, pos, size, m_pMuxingAppAsUTF8);

      if (status)
        return status;
    } else if (id == libwebm::kMkvWritingApp) {
      const long status =
          UnserializeString(pReader, pos, size, m_pWritingAppAsUTF8);

      if (status)
        return status;
    } else if (id == libwebm::kMkvTitle) {
      const long status = UnserializeString(pReader, pos, size, m_pTitleAsUTF8);

      if (status)
        return status;
    }

    pos += size;

    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  const double rollover_check = m_duration * m_timecodeScale;
  if (rollover_check > static_cast<double>(LLONG_MAX))
    return E_FILE_FORMAT_INVALID;

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;

  return 0;
}

long long SegmentInfo::GetTimeCodeScale() const { return m_timecodeScale; }

long long SegmentInfo::GetDuration() const {
  if (m_duration < 0)
    return -1;

  assert(m_timecodeScale >= 1);

  const double dd = double(m_duration) * double(m_timecodeScale);
  const long long d = static_cast<long long>(dd);

  return d;
}

const char* SegmentInfo::GetMuxingAppAsUTF8() const {
  return m_pMuxingAppAsUTF8;
}

const char* SegmentInfo::GetWritingAppAsUTF8() const {
  return m_pWritingAppAsUTF8;
}

const char* SegmentInfo::GetTitleAsUTF8() const { return m_pTitleAsUTF8; }

///////////////////////////////////////////////////////////////
// ContentEncoding element
ContentEncoding::ContentCompression::ContentCompression()
    : algo(0), settings(NULL), settings_len(0) {}

ContentEncoding::ContentCompression::~ContentCompression() {
  delete[] settings;
}

ContentEncoding::ContentEncryption::ContentEncryption()
    : algo(0),
      key_id(NULL),
      key_id_len(0),
      signature(NULL),
      signature_len(0),
      sig_key_id(NULL),
      sig_key_id_len(0),
      sig_algo(0),
      sig_hash_algo(0) {}

ContentEncoding::ContentEncryption::~ContentEncryption() {
  delete[] key_id;
  delete[] signature;
  delete[] sig_key_id;
}

ContentEncoding::ContentEncoding()
    : compression_entries_(NULL),
      compression_entries_end_(NULL),
      encryption_entries_(NULL),
      encryption_entries_end_(NULL),
      encoding_order_(0),
      encoding_scope_(1),
      encoding_type_(0) {}

ContentEncoding::~ContentEncoding() {
  ContentCompression** comp_i = compression_entries_;
  ContentCompression** const comp_j = compression_entries_end_;

  while (comp_i != comp_j) {
    ContentCompression* const comp = *comp_i++;
    delete comp;
  }

  delete[] compression_entries_;

  ContentEncryption** enc_i = encryption_entries_;
  ContentEncryption** const enc_j = encryption_entries_end_;

  while (enc_i != enc_j) {
    ContentEncryption* const enc = *enc_i++;
    delete enc;
  }

  delete[] encryption_entries_;
}

const ContentEncoding::ContentCompression*
ContentEncoding::GetCompressionByIndex(unsigned long idx) const {
  const ptrdiff_t count = compression_entries_end_ - compression_entries_;
  assert(count >= 0);

  if (idx >= static_cast<unsigned long>(count))
    return NULL;

  return compression_entries_[idx];
}

unsigned long ContentEncoding::GetCompressionCount() const {
  const ptrdiff_t count = compression_entries_end_ - compression_entries_;
  assert(count >= 0);

  return static_cast<unsigned long>(count);
}

const ContentEncoding::ContentEncryption* ContentEncoding::GetEncryptionByIndex(
    unsigned long idx) const {
  const ptrdiff_t count = encryption_entries_end_ - encryption_entries_;
  assert(count >= 0);

  if (idx >= static_cast<unsigned long>(count))
    return NULL;

  return encryption_entries_[idx];
}

unsigned long ContentEncoding::GetEncryptionCount() const {
  const ptrdiff_t count = encryption_entries_end_ - encryption_entries_;
  assert(count >= 0);

  return static_cast<unsigned long>(count);
}

long ContentEncoding::ParseContentEncAESSettingsEntry(
    long long start, long long size, IMkvReader* pReader,
    ContentEncAESSettings* aes) {
  assert(pReader);
  assert(aes);

  long long pos = start;
  const long long stop = start + size;

  while (pos < stop) {
    long long id, size;
    const long status = ParseElementHeader(pReader, pos, stop, id, size);
    if (status < 0)  // error
      return status;

    if (id == libwebm::kMkvAESSettingsCipherMode) {
      aes->cipher_mode = UnserializeUInt(pReader, pos, size);
      if (aes->cipher_mode != 1)
        return E_FILE_FORMAT_INVALID;
    }

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  return 0;
}

long ContentEncoding::ParseContentEncodingEntry(long long start, long long size,
                                                IMkvReader* pReader) {
  assert(pReader);

  long long pos = start;
  const long long stop = start + size;

  // Count ContentCompression and ContentEncryption elements.
  int compression_count = 0;
  int encryption_count = 0;

  while (pos < stop) {
    long long id, size;
    const long status = ParseElementHeader(pReader, pos, stop, id, size);
    if (status < 0)  // error
      return status;

    if (id == libwebm::kMkvContentCompression)
      ++compression_count;

    if (id == libwebm::kMkvContentEncryption)
      ++encryption_count;

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (compression_count <= 0 && encryption_count <= 0)
    return -1;

  if (compression_count > 0) {
    compression_entries_ =
        new (std::nothrow) ContentCompression*[compression_count];
    if (!compression_entries_)
      return -1;
    compression_entries_end_ = compression_entries_;
  }

  if (encryption_count > 0) {
    encryption_entries_ =
        new (std::nothrow) ContentEncryption*[encryption_count];
    if (!encryption_entries_) {
      delete[] compression_entries_;
      return -1;
    }
    encryption_entries_end_ = encryption_entries_;
  }

  pos = start;
  while (pos < stop) {
    long long id, size;
    long status = ParseElementHeader(pReader, pos, stop, id, size);
    if (status < 0)  // error
      return status;

    if (id == libwebm::kMkvContentEncodingOrder) {
      encoding_order_ = UnserializeUInt(pReader, pos, size);
    } else if (id == libwebm::kMkvContentEncodingScope) {
      encoding_scope_ = UnserializeUInt(pReader, pos, size);
      if (encoding_scope_ < 1)
        return -1;
    } else if (id == libwebm::kMkvContentEncodingType) {
      encoding_type_ = UnserializeUInt(pReader, pos, size);
    } else if (id == libwebm::kMkvContentCompression) {
      ContentCompression* const compression =
          new (std::nothrow) ContentCompression();
      if (!compression)
        return -1;

      status = ParseCompressionEntry(pos, size, pReader, compression);
      if (status) {
        delete compression;
        return status;
      }
      *compression_entries_end_++ = compression;
    } else if (id == libwebm::kMkvContentEncryption) {
      ContentEncryption* const encryption =
          new (std::nothrow) ContentEncryption();
      if (!encryption)
        return -1;

      status = ParseEncryptionEntry(pos, size, pReader, encryption);
      if (status) {
        delete encryption;
        return status;
      }
      *encryption_entries_end_++ = encryption;
    }

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;
  return 0;
}

long ContentEncoding::ParseCompressionEntry(long long start, long long size,
                                            IMkvReader* pReader,
                                            ContentCompression* compression) {
  assert(pReader);
  assert(compression);

  long long pos = start;
  const long long stop = start + size;

  bool valid = false;

  while (pos < stop) {
    long long id, size;
    const long status = ParseElementHeader(pReader, pos, stop, id, size);
    if (status < 0)  // error
      return status;

    if (id == libwebm::kMkvContentCompAlgo) {
      long long algo = UnserializeUInt(pReader, pos, size);
      if (algo < 0)
        return E_FILE_FORMAT_INVALID;
      compression->algo = algo;
      valid = true;
    } else if (id == libwebm::kMkvContentCompSettings) {
      if (size <= 0)
        return E_FILE_FORMAT_INVALID;

      const size_t buflen = static_cast<size_t>(size);
      unsigned char* buf = SafeArrayAlloc<unsigned char>(1, buflen);
      if (buf == NULL)
        return -1;

      const int read_status =
          pReader->Read(pos, static_cast<long>(buflen), buf);
      if (read_status) {
        delete[] buf;
        return status;
      }

      compression->settings = buf;
      compression->settings_len = buflen;
    }

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  // ContentCompAlgo is mandatory
  if (!valid)
    return E_FILE_FORMAT_INVALID;

  return 0;
}

long ContentEncoding::ParseEncryptionEntry(long long start, long long size,
                                           IMkvReader* pReader,
                                           ContentEncryption* encryption) {
  assert(pReader);
  assert(encryption);

  long long pos = start;
  const long long stop = start + size;

  while (pos < stop) {
    long long id, size;
    const long status = ParseElementHeader(pReader, pos, stop, id, size);
    if (status < 0)  // error
      return status;

    if (id == libwebm::kMkvContentEncAlgo) {
      encryption->algo = UnserializeUInt(pReader, pos, size);
      if (encryption->algo != 5)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvContentEncKeyID) {
      delete[] encryption->key_id;
      encryption->key_id = NULL;
      encryption->key_id_len = 0;

      if (size <= 0)
        return E_FILE_FORMAT_INVALID;

      const size_t buflen = static_cast<size_t>(size);
      unsigned char* buf = SafeArrayAlloc<unsigned char>(1, buflen);
      if (buf == NULL)
        return -1;

      const int read_status =
          pReader->Read(pos, static_cast<long>(buflen), buf);
      if (read_status) {
        delete[] buf;
        return status;
      }

      encryption->key_id = buf;
      encryption->key_id_len = buflen;
    } else if (id == libwebm::kMkvContentSignature) {
      delete[] encryption->signature;
      encryption->signature = NULL;
      encryption->signature_len = 0;

      if (size <= 0)
        return E_FILE_FORMAT_INVALID;

      const size_t buflen = static_cast<size_t>(size);
      unsigned char* buf = SafeArrayAlloc<unsigned char>(1, buflen);
      if (buf == NULL)
        return -1;

      const int read_status =
          pReader->Read(pos, static_cast<long>(buflen), buf);
      if (read_status) {
        delete[] buf;
        return status;
      }

      encryption->signature = buf;
      encryption->signature_len = buflen;
    } else if (id == libwebm::kMkvContentSigKeyID) {
      delete[] encryption->sig_key_id;
      encryption->sig_key_id = NULL;
      encryption->sig_key_id_len = 0;

      if (size <= 0)
        return E_FILE_FORMAT_INVALID;

      const size_t buflen = static_cast<size_t>(size);
      unsigned char* buf = SafeArrayAlloc<unsigned char>(1, buflen);
      if (buf == NULL)
        return -1;

      const int read_status =
          pReader->Read(pos, static_cast<long>(buflen), buf);
      if (read_status) {
        delete[] buf;
        return status;
      }

      encryption->sig_key_id = buf;
      encryption->sig_key_id_len = buflen;
    } else if (id == libwebm::kMkvContentSigAlgo) {
      encryption->sig_algo = UnserializeUInt(pReader, pos, size);
    } else if (id == libwebm::kMkvContentSigHashAlgo) {
      encryption->sig_hash_algo = UnserializeUInt(pReader, pos, size);
    } else if (id == libwebm::kMkvContentEncAESSettings) {
      const long status = ParseContentEncAESSettingsEntry(
          pos, size, pReader, &encryption->aes_settings);
      if (status)
        return status;
    }

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  return 0;
}

Track::Track(Segment* pSegment, long long element_start, long long element_size)
    : m_pSegment(pSegment),
      m_element_start(element_start),
      m_element_size(element_size),
      content_encoding_entries_(NULL),
      content_encoding_entries_end_(NULL) {}

Track::~Track() {
  Info& info = const_cast<Info&>(m_info);
  info.Clear();

  ContentEncoding** i = content_encoding_entries_;
  ContentEncoding** const j = content_encoding_entries_end_;

  while (i != j) {
    ContentEncoding* const encoding = *i++;
    delete encoding;
  }

  delete[] content_encoding_entries_;
}

long Track::Create(Segment* pSegment, const Info& info, long long element_start,
                   long long element_size, Track*& pResult) {
  if (pResult)
    return -1;

  Track* const pTrack =
      new (std::nothrow) Track(pSegment, element_start, element_size);

  if (pTrack == NULL)
    return -1;  // generic error

  const int status = info.Copy(pTrack->m_info);

  if (status) {  // error
    delete pTrack;
    return status;
  }

  pResult = pTrack;
  return 0;  // success
}

Track::Info::Info()
    : uid(0),
      defaultDuration(0),
      codecDelay(0),
      seekPreRoll(0),
      nameAsUTF8(NULL),
      language(NULL),
      codecId(NULL),
      codecNameAsUTF8(NULL),
      codecPrivate(NULL),
      codecPrivateSize(0),
      lacing(false) {}

Track::Info::~Info() { Clear(); }

void Track::Info::Clear() {
  delete[] nameAsUTF8;
  nameAsUTF8 = NULL;

  delete[] language;
  language = NULL;

  delete[] codecId;
  codecId = NULL;

  delete[] codecPrivate;
  codecPrivate = NULL;
  codecPrivateSize = 0;

  delete[] codecNameAsUTF8;
  codecNameAsUTF8 = NULL;
}

int Track::Info::CopyStr(char* Info::*str, Info& dst_) const {
  if (str == static_cast<char * Info::*>(NULL))
    return -1;

  char*& dst = dst_.*str;

  if (dst)  // should be NULL already
    return -1;

  const char* const src = this->*str;

  if (src == NULL)
    return 0;

  const size_t len = strlen(src);

  dst = SafeArrayAlloc<char>(1, len + 1);

  if (dst == NULL)
    return -1;

  strcpy(dst, src);

  return 0;
}

int Track::Info::Copy(Info& dst) const {
  if (&dst == this)
    return 0;

  dst.type = type;
  dst.number = number;
  dst.defaultDuration = defaultDuration;
  dst.codecDelay = codecDelay;
  dst.seekPreRoll = seekPreRoll;
  dst.uid = uid;
  dst.lacing = lacing;
  dst.settings = settings;

  // We now copy the string member variables from src to dst.
  // This involves memory allocation so in principle the operation
  // can fail (indeed, that's why we have Info::Copy), so we must
  // report this to the caller.  An error return from this function
  // therefore implies that the copy was only partially successful.

  if (int status = CopyStr(&Info::nameAsUTF8, dst))
    return status;

  if (int status = CopyStr(&Info::language, dst))
    return status;

  if (int status = CopyStr(&Info::codecId, dst))
    return status;

  if (int status = CopyStr(&Info::codecNameAsUTF8, dst))
    return status;

  if (codecPrivateSize > 0) {
    if (codecPrivate == NULL)
      return -1;

    if (dst.codecPrivate)
      return -1;

    if (dst.codecPrivateSize != 0)
      return -1;

    dst.codecPrivate = SafeArrayAlloc<unsigned char>(1, codecPrivateSize);

    if (dst.codecPrivate == NULL)
      return -1;

    memcpy(dst.codecPrivate, codecPrivate, codecPrivateSize);
    dst.codecPrivateSize = codecPrivateSize;
  }

  return 0;
}

const BlockEntry* Track::GetEOS() const { return &m_eos; }

long Track::GetType() const { return m_info.type; }

long Track::GetNumber() const { return m_info.number; }

unsigned long long Track::GetUid() const { return m_info.uid; }

const char* Track::GetNameAsUTF8() const { return m_info.nameAsUTF8; }

const char* Track::GetLanguage() const { return m_info.language; }

const char* Track::GetCodecNameAsUTF8() const { return m_info.codecNameAsUTF8; }

const char* Track::GetCodecId() const { return m_info.codecId; }

const unsigned char* Track::GetCodecPrivate(size_t& size) const {
  size = m_info.codecPrivateSize;
  return m_info.codecPrivate;
}

bool Track::GetLacing() const { return m_info.lacing; }

unsigned long long Track::GetDefaultDuration() const {
  return m_info.defaultDuration;
}

unsigned long long Track::GetCodecDelay() const { return m_info.codecDelay; }

unsigned long long Track::GetSeekPreRoll() const { return m_info.seekPreRoll; }

long Track::GetFirst(const BlockEntry*& pBlockEntry) const {
  const Cluster* pCluster = m_pSegment->GetFirst();

  for (int i = 0;;) {
    if (pCluster == NULL) {
      pBlockEntry = GetEOS();
      return 1;
    }

    if (pCluster->EOS()) {
      if (m_pSegment->DoneParsing()) {
        pBlockEntry = GetEOS();
        return 1;
      }

      pBlockEntry = 0;
      return E_BUFFER_NOT_FULL;
    }

    long status = pCluster->GetFirst(pBlockEntry);

    if (status < 0)  // error
      return status;

    if (pBlockEntry == 0) {  // empty cluster
      pCluster = m_pSegment->GetNext(pCluster);
      continue;
    }

    for (;;) {
      const Block* const pBlock = pBlockEntry->GetBlock();
      assert(pBlock);

      const long long tn = pBlock->GetTrackNumber();

      if ((tn == m_info.number) && VetEntry(pBlockEntry))
        return 0;

      const BlockEntry* pNextEntry;

      status = pCluster->GetNext(pBlockEntry, pNextEntry);

      if (status < 0)  // error
        return status;

      if (pNextEntry == 0)
        break;

      pBlockEntry = pNextEntry;
    }

    ++i;

    if (i >= 100)
      break;

    pCluster = m_pSegment->GetNext(pCluster);
  }

  // NOTE: if we get here, it means that we didn't find a block with
  // a matching track number.  We interpret that as an error (which
  // might be too conservative).

  pBlockEntry = GetEOS();  // so we can return a non-NULL value
  return 1;
}

long Track::GetNext(const BlockEntry* pCurrEntry,
                    const BlockEntry*& pNextEntry) const {
  assert(pCurrEntry);
  assert(!pCurrEntry->EOS());  //?

  const Block* const pCurrBlock = pCurrEntry->GetBlock();
  assert(pCurrBlock && pCurrBlock->GetTrackNumber() == m_info.number);
  if (!pCurrBlock || pCurrBlock->GetTrackNumber() != m_info.number)
    return -1;

  const Cluster* pCluster = pCurrEntry->GetCluster();
  assert(pCluster);
  assert(!pCluster->EOS());

  long status = pCluster->GetNext(pCurrEntry, pNextEntry);

  if (status < 0)  // error
    return status;

  for (int i = 0;;) {
    while (pNextEntry) {
      const Block* const pNextBlock = pNextEntry->GetBlock();
      assert(pNextBlock);

      if (pNextBlock->GetTrackNumber() == m_info.number)
        return 0;

      pCurrEntry = pNextEntry;

      status = pCluster->GetNext(pCurrEntry, pNextEntry);

      if (status < 0)  // error
        return status;
    }

    pCluster = m_pSegment->GetNext(pCluster);

    if (pCluster == NULL) {
      pNextEntry = GetEOS();
      return 1;
    }

    if (pCluster->EOS()) {
      if (m_pSegment->DoneParsing()) {
        pNextEntry = GetEOS();
        return 1;
      }

      // TODO: there is a potential O(n^2) problem here: we tell the
      // caller to (pre)load another cluster, which he does, but then he
      // calls GetNext again, which repeats the same search.  This is
      // a pathological case, since the only way it can happen is if
      // there exists a long sequence of clusters none of which contain a
      // block from this track.  One way around this problem is for the
      // caller to be smarter when he loads another cluster: don't call
      // us back until you have a cluster that contains a block from this
      // track. (Of course, that's not cheap either, since our caller
      // would have to scan the each cluster as it's loaded, so that
      // would just push back the problem.)

      pNextEntry = NULL;
      return E_BUFFER_NOT_FULL;
    }

    status = pCluster->GetFirst(pNextEntry);

    if (status < 0)  // error
      return status;

    if (pNextEntry == NULL)  // empty cluster
      continue;

    ++i;

    if (i >= 100)
      break;
  }

  // NOTE: if we get here, it means that we didn't find a block with
  // a matching track number after lots of searching, so we give
  // up trying.

  pNextEntry = GetEOS();  // so we can return a non-NULL value
  return 1;
}

bool Track::VetEntry(const BlockEntry* pBlockEntry) const {
  assert(pBlockEntry);
  const Block* const pBlock = pBlockEntry->GetBlock();
  assert(pBlock);
  assert(pBlock->GetTrackNumber() == m_info.number);
  if (!pBlock || pBlock->GetTrackNumber() != m_info.number)
    return false;

  // This function is used during a seek to determine whether the
  // frame is a valid seek target.  This default function simply
  // returns true, which means all frames are valid seek targets.
  // It gets overridden by the VideoTrack class, because only video
  // keyframes can be used as seek target.

  return true;
}

long Track::Seek(long long time_ns, const BlockEntry*& pResult) const {
  const long status = GetFirst(pResult);

  if (status < 0)  // buffer underflow, etc
    return status;

  assert(pResult);

  if (pResult->EOS())
    return 0;

  const Cluster* pCluster = pResult->GetCluster();
  assert(pCluster);
  assert(pCluster->GetIndex() >= 0);

  if (time_ns <= pResult->GetBlock()->GetTime(pCluster))
    return 0;

  Cluster** const clusters = m_pSegment->m_clusters;
  assert(clusters);

  const long count = m_pSegment->GetCount();  // loaded only, not preloaded
  assert(count > 0);

  Cluster** const i = clusters + pCluster->GetIndex();
  assert(i);
  assert(*i == pCluster);
  assert(pCluster->GetTime() <= time_ns);

  Cluster** const j = clusters + count;

  Cluster** lo = i;
  Cluster** hi = j;

  while (lo < hi) {
    // INVARIANT:
    //[i, lo) <= time_ns
    //[lo, hi) ?
    //[hi, j)  > time_ns

    Cluster** const mid = lo + (hi - lo) / 2;
    assert(mid < hi);

    pCluster = *mid;
    assert(pCluster);
    assert(pCluster->GetIndex() >= 0);
    assert(pCluster->GetIndex() == long(mid - m_pSegment->m_clusters));

    const long long t = pCluster->GetTime();

    if (t <= time_ns)
      lo = mid + 1;
    else
      hi = mid;

    assert(lo <= hi);
  }

  assert(lo == hi);
  assert(lo > i);
  assert(lo <= j);

  while (lo > i) {
    pCluster = *--lo;
    assert(pCluster);
    assert(pCluster->GetTime() <= time_ns);

    pResult = pCluster->GetEntry(this);

    if ((pResult != 0) && !pResult->EOS())
      return 0;

    // landed on empty cluster (no entries)
  }

  pResult = GetEOS();  // weird
  return 0;
}

const ContentEncoding* Track::GetContentEncodingByIndex(
    unsigned long idx) const {
  const ptrdiff_t count =
      content_encoding_entries_end_ - content_encoding_entries_;
  assert(count >= 0);

  if (idx >= static_cast<unsigned long>(count))
    return NULL;

  return content_encoding_entries_[idx];
}

unsigned long Track::GetContentEncodingCount() const {
  const ptrdiff_t count =
      content_encoding_entries_end_ - content_encoding_entries_;
  assert(count >= 0);

  return static_cast<unsigned long>(count);
}

long Track::ParseContentEncodingsEntry(long long start, long long size) {
  IMkvReader* const pReader = m_pSegment->m_pReader;
  assert(pReader);

  long long pos = start;
  const long long stop = start + size;

  // Count ContentEncoding elements.
  int count = 0;
  while (pos < stop) {
    long long id, size;
    const long status = ParseElementHeader(pReader, pos, stop, id, size);
    if (status < 0)  // error
      return status;

    // pos now designates start of element
    if (id == libwebm::kMkvContentEncoding)
      ++count;

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (count <= 0)
    return -1;

  content_encoding_entries_ = new (std::nothrow) ContentEncoding*[count];
  if (!content_encoding_entries_)
    return -1;

  content_encoding_entries_end_ = content_encoding_entries_;

  pos = start;
  while (pos < stop) {
    long long id, size;
    long status = ParseElementHeader(pReader, pos, stop, id, size);
    if (status < 0)  // error
      return status;

    // pos now designates start of element
    if (id == libwebm::kMkvContentEncoding) {
      ContentEncoding* const content_encoding =
          new (std::nothrow) ContentEncoding();
      if (!content_encoding)
        return -1;

      status = content_encoding->ParseContentEncodingEntry(pos, size, pReader);
      if (status) {
        delete content_encoding;
        return status;
      }

      *content_encoding_entries_end_++ = content_encoding;
    }

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;

  return 0;
}

Track::EOSBlock::EOSBlock() : BlockEntry(NULL, LONG_MIN) {}

BlockEntry::Kind Track::EOSBlock::GetKind() const { return kBlockEOS; }

const Block* Track::EOSBlock::GetBlock() const { return NULL; }

bool PrimaryChromaticity::Parse(IMkvReader* reader, long long read_pos,
                                long long value_size, bool is_x,
                                PrimaryChromaticity** chromaticity) {
  if (!reader)
    return false;

  if (!*chromaticity)
    *chromaticity = new PrimaryChromaticity();

  if (!*chromaticity)
    return false;

  PrimaryChromaticity* pc = *chromaticity;
  float* value = is_x ? &pc->x : &pc->y;

  double parser_value = 0;
  const long long parse_status =
      UnserializeFloat(reader, read_pos, value_size, parser_value);

  // Valid range is [0, 1]. Make sure the double is representable as a float
  // before casting.
  if (parse_status < 0 || parser_value < 0.0 || parser_value > 1.0 ||
      (parser_value > 0.0 && parser_value < FLT_MIN))
    return false;

  *value = static_cast<float>(parser_value);

  return true;
}

bool MasteringMetadata::Parse(IMkvReader* reader, long long mm_start,
                              long long mm_size, MasteringMetadata** mm) {
  if (!reader || *mm)
    return false;

  std::unique_ptr<MasteringMetadata> mm_ptr(new MasteringMetadata());
  if (!mm_ptr.get())
    return false;

  const long long mm_end = mm_start + mm_size;
  long long read_pos = mm_start;

  while (read_pos < mm_end) {
    long long child_id = 0;
    long long child_size = 0;

    const long long status =
        ParseElementHeader(reader, read_pos, mm_end, child_id, child_size);
    if (status < 0)
      return false;

    if (child_id == libwebm::kMkvLuminanceMax) {
      double value = 0;
      const long long value_parse_status =
          UnserializeFloat(reader, read_pos, child_size, value);
      if (value < -FLT_MAX || value > FLT_MAX ||
          (value > 0.0 && value < FLT_MIN)) {
        return false;
      }
      mm_ptr->luminance_max = static_cast<float>(value);
      if (value_parse_status < 0 || mm_ptr->luminance_max < 0.0 ||
          mm_ptr->luminance_max > 9999.99) {
        return false;
      }
    } else if (child_id == libwebm::kMkvLuminanceMin) {
      double value = 0;
      const long long value_parse_status =
          UnserializeFloat(reader, read_pos, child_size, value);
      if (value < -FLT_MAX || value > FLT_MAX ||
          (value > 0.0 && value < FLT_MIN)) {
        return false;
      }
      mm_ptr->luminance_min = static_cast<float>(value);
      if (value_parse_status < 0 || mm_ptr->luminance_min < 0.0 ||
          mm_ptr->luminance_min > 999.9999) {
        return false;
      }
    } else {
      bool is_x = false;
      PrimaryChromaticity** chromaticity;
      switch (child_id) {
        case libwebm::kMkvPrimaryRChromaticityX:
        case libwebm::kMkvPrimaryRChromaticityY:
          is_x = child_id == libwebm::kMkvPrimaryRChromaticityX;
          chromaticity = &mm_ptr->r;
          break;
        case libwebm::kMkvPrimaryGChromaticityX:
        case libwebm::kMkvPrimaryGChromaticityY:
          is_x = child_id == libwebm::kMkvPrimaryGChromaticityX;
          chromaticity = &mm_ptr->g;
          break;
        case libwebm::kMkvPrimaryBChromaticityX:
        case libwebm::kMkvPrimaryBChromaticityY:
          is_x = child_id == libwebm::kMkvPrimaryBChromaticityX;
          chromaticity = &mm_ptr->b;
          break;
        case libwebm::kMkvWhitePointChromaticityX:
        case libwebm::kMkvWhitePointChromaticityY:
          is_x = child_id == libwebm::kMkvWhitePointChromaticityX;
          chromaticity = &mm_ptr->white_point;
          break;
        default:
          return false;
      }
      const bool value_parse_status = PrimaryChromaticity::Parse(
          reader, read_pos, child_size, is_x, chromaticity);
      if (!value_parse_status)
        return false;
    }

    read_pos += child_size;
    if (read_pos > mm_end)
      return false;
  }

  *mm = mm_ptr.release();
  return true;
}

bool Colour::Parse(IMkvReader* reader, long long colour_start,
                   long long colour_size, Colour** colour) {
  if (!reader || *colour)
    return false;

  std::unique_ptr<Colour> colour_ptr(new Colour());
  if (!colour_ptr.get())
    return false;

  const long long colour_end = colour_start + colour_size;
  long long read_pos = colour_start;

  while (read_pos < colour_end) {
    long long child_id = 0;
    long long child_size = 0;

    const long status =
        ParseElementHeader(reader, read_pos, colour_end, child_id, child_size);
    if (status < 0)
      return false;

    if (child_id == libwebm::kMkvMatrixCoefficients) {
      colour_ptr->matrix_coefficients =
          UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->matrix_coefficients < 0)
        return false;
    } else if (child_id == libwebm::kMkvBitsPerChannel) {
      colour_ptr->bits_per_channel =
          UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->bits_per_channel < 0)
        return false;
    } else if (child_id == libwebm::kMkvChromaSubsamplingHorz) {
      colour_ptr->chroma_subsampling_horz =
          UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->chroma_subsampling_horz < 0)
        return false;
    } else if (child_id == libwebm::kMkvChromaSubsamplingVert) {
      colour_ptr->chroma_subsampling_vert =
          UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->chroma_subsampling_vert < 0)
        return false;
    } else if (child_id == libwebm::kMkvCbSubsamplingHorz) {
      colour_ptr->cb_subsampling_horz =
          UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->cb_subsampling_horz < 0)
        return false;
    } else if (child_id == libwebm::kMkvCbSubsamplingVert) {
      colour_ptr->cb_subsampling_vert =
          UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->cb_subsampling_vert < 0)
        return false;
    } else if (child_id == libwebm::kMkvChromaSitingHorz) {
      colour_ptr->chroma_siting_horz =
          UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->chroma_siting_horz < 0)
        return false;
    } else if (child_id == libwebm::kMkvChromaSitingVert) {
      colour_ptr->chroma_siting_vert =
          UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->chroma_siting_vert < 0)
        return false;
    } else if (child_id == libwebm::kMkvRange) {
      colour_ptr->range = UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->range < 0)
        return false;
    } else if (child_id == libwebm::kMkvTransferCharacteristics) {
      colour_ptr->transfer_characteristics =
          UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->transfer_characteristics < 0)
        return false;
    } else if (child_id == libwebm::kMkvPrimaries) {
      colour_ptr->primaries = UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->primaries < 0)
        return false;
    } else if (child_id == libwebm::kMkvMaxCLL) {
      colour_ptr->max_cll = UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->max_cll < 0)
        return false;
    } else if (child_id == libwebm::kMkvMaxFALL) {
      colour_ptr->max_fall = UnserializeUInt(reader, read_pos, child_size);
      if (colour_ptr->max_fall < 0)
        return false;
    } else if (child_id == libwebm::kMkvMasteringMetadata) {
      if (!MasteringMetadata::Parse(reader, read_pos, child_size,
                                    &colour_ptr->mastering_metadata))
        return false;
    } else {
      return false;
    }

    read_pos += child_size;
    if (read_pos > colour_end)
      return false;
  }
  *colour = colour_ptr.release();
  return true;
}

bool Projection::Parse(IMkvReader* reader, long long start, long long size,
                       Projection** projection) {
  if (!reader || *projection)
    return false;

  std::unique_ptr<Projection> projection_ptr(new Projection());
  if (!projection_ptr.get())
    return false;

  const long long end = start + size;
  long long read_pos = start;

  while (read_pos < end) {
    long long child_id = 0;
    long long child_size = 0;

    const long long status =
        ParseElementHeader(reader, read_pos, end, child_id, child_size);
    if (status < 0)
      return false;

    if (child_id == libwebm::kMkvProjectionType) {
      long long projection_type = kTypeNotPresent;
      projection_type = UnserializeUInt(reader, read_pos, child_size);
      if (projection_type < 0)
        return false;

      projection_ptr->type = static_cast<ProjectionType>(projection_type);
    } else if (child_id == libwebm::kMkvProjectionPrivate) {
      unsigned char* data = SafeArrayAlloc<unsigned char>(1, child_size);

      if (data == NULL)
        return false;

      const int status =
          reader->Read(read_pos, static_cast<long>(child_size), data);

      if (status) {
        delete[] data;
        return false;
      }

      projection_ptr->private_data = data;
      projection_ptr->private_data_length = static_cast<size_t>(child_size);
    } else {
      double value = 0;
      const long long value_parse_status =
          UnserializeFloat(reader, read_pos, child_size, value);
      // Make sure value is representable as a float before casting.
      if (value_parse_status < 0 || value < -FLT_MAX || value > FLT_MAX ||
          (value > 0.0 && value < FLT_MIN)) {
        return false;
      }

      switch (child_id) {
        case libwebm::kMkvProjectionPoseYaw:
          projection_ptr->pose_yaw = static_cast<float>(value);
          break;
        case libwebm::kMkvProjectionPosePitch:
          projection_ptr->pose_pitch = static_cast<float>(value);
          break;
        case libwebm::kMkvProjectionPoseRoll:
          projection_ptr->pose_roll = static_cast<float>(value);
          break;
        default:
          return false;
      }
    }

    read_pos += child_size;
    if (read_pos > end)
      return false;
  }

  *projection = projection_ptr.release();
  return true;
}

VideoTrack::VideoTrack(Segment* pSegment, long long element_start,
                       long long element_size)
    : Track(pSegment, element_start, element_size),
      m_colour(NULL),
      m_projection(NULL) {}

VideoTrack::~VideoTrack() {
  delete m_colour;
  delete m_projection;
}

long VideoTrack::Parse(Segment* pSegment, const Info& info,
                       long long element_start, long long element_size,
                       VideoTrack*& pResult) {
  if (pResult)
    return -1;

  if (info.type != Track::kVideo)
    return -1;

  long long width = 0;
  long long height = 0;
  long long display_width = 0;
  long long display_height = 0;
  long long display_unit = 0;
  long long stereo_mode = 0;

  double rate = 0.0;

  IMkvReader* const pReader = pSegment->m_pReader;

  const Settings& s = info.settings;
  assert(s.start >= 0);
  assert(s.size >= 0);

  long long pos = s.start;
  assert(pos >= 0);

  const long long stop = pos + s.size;

  Colour* colour = NULL;
  Projection* projection = NULL;

  while (pos < stop) {
    long long id, size;

    const long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)  // error
      return status;

    if (id == libwebm::kMkvPixelWidth) {
      width = UnserializeUInt(pReader, pos, size);

      if (width <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvPixelHeight) {
      height = UnserializeUInt(pReader, pos, size);

      if (height <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvDisplayWidth) {
      display_width = UnserializeUInt(pReader, pos, size);

      if (display_width <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvDisplayHeight) {
      display_height = UnserializeUInt(pReader, pos, size);

      if (display_height <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvDisplayUnit) {
      display_unit = UnserializeUInt(pReader, pos, size);

      if (display_unit < 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvStereoMode) {
      stereo_mode = UnserializeUInt(pReader, pos, size);

      if (stereo_mode < 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvFrameRate) {
      const long status = UnserializeFloat(pReader, pos, size, rate);

      if (status < 0)
        return status;

      if (rate <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvColour) {
      if (!Colour::Parse(pReader, pos, size, &colour))
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvProjection) {
      if (!Projection::Parse(pReader, pos, size, &projection))
        return E_FILE_FORMAT_INVALID;
    }

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;

  VideoTrack* const pTrack =
      new (std::nothrow) VideoTrack(pSegment, element_start, element_size);

  if (pTrack == NULL)
    return -1;  // generic error

  const int status = info.Copy(pTrack->m_info);

  if (status) {  // error
    delete pTrack;
    return status;
  }

  pTrack->m_width = width;
  pTrack->m_height = height;
  pTrack->m_display_width = display_width;
  pTrack->m_display_height = display_height;
  pTrack->m_display_unit = display_unit;
  pTrack->m_stereo_mode = stereo_mode;
  pTrack->m_rate = rate;
  pTrack->m_colour = colour;
  pTrack->m_projection = projection;

  pResult = pTrack;
  return 0;  // success
}

bool VideoTrack::VetEntry(const BlockEntry* pBlockEntry) const {
  return Track::VetEntry(pBlockEntry) && pBlockEntry->GetBlock()->IsKey();
}

long VideoTrack::Seek(long long time_ns, const BlockEntry*& pResult) const {
  const long status = GetFirst(pResult);

  if (status < 0)  // buffer underflow, etc
    return status;

  assert(pResult);

  if (pResult->EOS())
    return 0;

  const Cluster* pCluster = pResult->GetCluster();
  assert(pCluster);
  assert(pCluster->GetIndex() >= 0);

  if (time_ns <= pResult->GetBlock()->GetTime(pCluster))
    return 0;

  Cluster** const clusters = m_pSegment->m_clusters;
  assert(clusters);

  const long count = m_pSegment->GetCount();  // loaded only, not pre-loaded
  assert(count > 0);

  Cluster** const i = clusters + pCluster->GetIndex();
  assert(i);
  assert(*i == pCluster);
  assert(pCluster->GetTime() <= time_ns);

  Cluster** const j = clusters + count;

  Cluster** lo = i;
  Cluster** hi = j;

  while (lo < hi) {
    // INVARIANT:
    //[i, lo) <= time_ns
    //[lo, hi) ?
    //[hi, j)  > time_ns

    Cluster** const mid = lo + (hi - lo) / 2;
    assert(mid < hi);

    pCluster = *mid;
    assert(pCluster);
    assert(pCluster->GetIndex() >= 0);
    assert(pCluster->GetIndex() == long(mid - m_pSegment->m_clusters));

    const long long t = pCluster->GetTime();

    if (t <= time_ns)
      lo = mid + 1;
    else
      hi = mid;

    assert(lo <= hi);
  }

  assert(lo == hi);
  assert(lo > i);
  assert(lo <= j);

  pCluster = *--lo;
  assert(pCluster);
  assert(pCluster->GetTime() <= time_ns);

  pResult = pCluster->GetEntry(this, time_ns);

  if ((pResult != 0) && !pResult->EOS())  // found a keyframe
    return 0;

  while (lo != i) {
    pCluster = *--lo;
    assert(pCluster);
    assert(pCluster->GetTime() <= time_ns);

    pResult = pCluster->GetEntry(this, time_ns);

    if ((pResult != 0) && !pResult->EOS())
      return 0;
  }

  // weird: we're on the first cluster, but no keyframe found
  // should never happen but we must return something anyway

  pResult = GetEOS();
  return 0;
}

Colour* VideoTrack::GetColour() const { return m_colour; }

Projection* VideoTrack::GetProjection() const { return m_projection; }

long long VideoTrack::GetWidth() const { return m_width; }

long long VideoTrack::GetHeight() const { return m_height; }

long long VideoTrack::GetDisplayWidth() const {
  return m_display_width > 0 ? m_display_width : GetWidth();
}

long long VideoTrack::GetDisplayHeight() const {
  return m_display_height > 0 ? m_display_height : GetHeight();
}

long long VideoTrack::GetDisplayUnit() const { return m_display_unit; }

long long VideoTrack::GetStereoMode() const { return m_stereo_mode; }

double VideoTrack::GetFrameRate() const { return m_rate; }

AudioTrack::AudioTrack(Segment* pSegment, long long element_start,
                       long long element_size)
    : Track(pSegment, element_start, element_size) {}

long AudioTrack::Parse(Segment* pSegment, const Info& info,
                       long long element_start, long long element_size,
                       AudioTrack*& pResult) {
  if (pResult)
    return -1;

  if (info.type != Track::kAudio)
    return -1;

  IMkvReader* const pReader = pSegment->m_pReader;

  const Settings& s = info.settings;
  assert(s.start >= 0);
  assert(s.size >= 0);

  long long pos = s.start;
  assert(pos >= 0);

  const long long stop = pos + s.size;

  double rate = 8000.0;  // MKV default
  long long channels = 1;
  long long bit_depth = 0;

  while (pos < stop) {
    long long id, size;

    long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)  // error
      return status;

    if (id == libwebm::kMkvSamplingFrequency) {
      status = UnserializeFloat(pReader, pos, size, rate);

      if (status < 0)
        return status;

      if (rate <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvChannels) {
      channels = UnserializeUInt(pReader, pos, size);

      if (channels <= 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvBitDepth) {
      bit_depth = UnserializeUInt(pReader, pos, size);

      if (bit_depth <= 0)
        return E_FILE_FORMAT_INVALID;
    }

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;

  AudioTrack* const pTrack =
      new (std::nothrow) AudioTrack(pSegment, element_start, element_size);

  if (pTrack == NULL)
    return -1;  // generic error

  const int status = info.Copy(pTrack->m_info);

  if (status) {
    delete pTrack;
    return status;
  }

  pTrack->m_rate = rate;
  pTrack->m_channels = channels;
  pTrack->m_bitDepth = bit_depth;

  pResult = pTrack;
  return 0;  // success
}

double AudioTrack::GetSamplingRate() const { return m_rate; }

long long AudioTrack::GetChannels() const { return m_channels; }

long long AudioTrack::GetBitDepth() const { return m_bitDepth; }

Tracks::Tracks(Segment* pSegment, long long start, long long size_,
               long long element_start, long long element_size)
    : m_pSegment(pSegment),
      m_start(start),
      m_size(size_),
      m_element_start(element_start),
      m_element_size(element_size),
      m_trackEntries(NULL),
      m_trackEntriesEnd(NULL) {}

long Tracks::Parse() {
  assert(m_trackEntries == NULL);
  assert(m_trackEntriesEnd == NULL);

  const long long stop = m_start + m_size;
  IMkvReader* const pReader = m_pSegment->m_pReader;

  int count = 0;
  long long pos = m_start;

  while (pos < stop) {
    long long id, size;

    const long status = ParseElementHeader(pReader, pos, stop, id, size);

    if (status < 0)  // error
      return status;

    if (size == 0)  // weird
      continue;

    if (id == libwebm::kMkvTrackEntry)
      ++count;

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;

  if (count <= 0)
    return 0;  // success

  m_trackEntries = new (std::nothrow) Track*[count];

  if (m_trackEntries == NULL)
    return -1;

  m_trackEntriesEnd = m_trackEntries;

  pos = m_start;

  while (pos < stop) {
    const long long element_start = pos;

    long long id, payload_size;

    const long status =
        ParseElementHeader(pReader, pos, stop, id, payload_size);

    if (status < 0)  // error
      return status;

    if (payload_size == 0)  // weird
      continue;

    const long long payload_stop = pos + payload_size;
    assert(payload_stop <= stop);  // checked in ParseElement

    const long long element_size = payload_stop - element_start;

    if (id == libwebm::kMkvTrackEntry) {
      Track*& pTrack = *m_trackEntriesEnd;
      pTrack = NULL;

      const long status = ParseTrackEntry(pos, payload_size, element_start,
                                          element_size, pTrack);
      if (status)
        return status;

      if (pTrack)
        ++m_trackEntriesEnd;
    }

    pos = payload_stop;
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;

  return 0;  // success
}

unsigned long Tracks::GetTracksCount() const {
  const ptrdiff_t result = m_trackEntriesEnd - m_trackEntries;
  assert(result >= 0);

  return static_cast<unsigned long>(result);
}

long Tracks::ParseTrackEntry(long long track_start, long long track_size,
                             long long element_start, long long element_size,
                             Track*& pResult) const {
  if (pResult)
    return -1;

  IMkvReader* const pReader = m_pSegment->m_pReader;

  long long pos = track_start;
  const long long track_stop = track_start + track_size;

  Track::Info info;

  info.type = 0;
  info.number = 0;
  info.uid = 0;
  info.defaultDuration = 0;

  Track::Settings v;
  v.start = -1;
  v.size = -1;

  Track::Settings a;
  a.start = -1;
  a.size = -1;

  Track::Settings e;  // content_encodings_settings;
  e.start = -1;
  e.size = -1;

  long long lacing = 1;  // default is true

  while (pos < track_stop) {
    long long id, size;

    const long status = ParseElementHeader(pReader, pos, track_stop, id, size);

    if (status < 0)  // error
      return status;

    if (size < 0)
      return E_FILE_FORMAT_INVALID;

    const long long start = pos;

    if (id == libwebm::kMkvVideo) {
      v.start = start;
      v.size = size;
    } else if (id == libwebm::kMkvAudio) {
      a.start = start;
      a.size = size;
    } else if (id == libwebm::kMkvContentEncodings) {
      e.start = start;
      e.size = size;
    } else if (id == libwebm::kMkvTrackUID) {
      if (size > 8)
        return E_FILE_FORMAT_INVALID;

      info.uid = 0;

      long long pos_ = start;
      const long long pos_end = start + size;

      while (pos_ != pos_end) {
        unsigned char b;

        const int status = pReader->Read(pos_, 1, &b);

        if (status)
          return status;

        info.uid <<= 8;
        info.uid |= b;

        ++pos_;
      }
    } else if (id == libwebm::kMkvTrackNumber) {
      const long long num = UnserializeUInt(pReader, pos, size);

      if ((num <= 0) || (num > 127))
        return E_FILE_FORMAT_INVALID;

      info.number = static_cast<long>(num);
    } else if (id == libwebm::kMkvTrackType) {
      const long long type = UnserializeUInt(pReader, pos, size);

      if ((type <= 0) || (type > 254))
        return E_FILE_FORMAT_INVALID;

      info.type = static_cast<long>(type);
    } else if (id == libwebm::kMkvName) {
      const long status =
          UnserializeString(pReader, pos, size, info.nameAsUTF8);

      if (status)
        return status;
    } else if (id == libwebm::kMkvLanguage) {
      const long status = UnserializeString(pReader, pos, size, info.language);

      if (status)
        return status;
    } else if (id == libwebm::kMkvDefaultDuration) {
      const long long duration = UnserializeUInt(pReader, pos, size);

      if (duration < 0)
        return E_FILE_FORMAT_INVALID;

      info.defaultDuration = static_cast<unsigned long long>(duration);
    } else if (id == libwebm::kMkvCodecID) {
      const long status = UnserializeString(pReader, pos, size, info.codecId);

      if (status)
        return status;
    } else if (id == libwebm::kMkvFlagLacing) {
      lacing = UnserializeUInt(pReader, pos, size);

      if ((lacing < 0) || (lacing > 1))
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvCodecPrivate) {
      delete[] info.codecPrivate;
      info.codecPrivate = NULL;
      info.codecPrivateSize = 0;

      const size_t buflen = static_cast<size_t>(size);

      if (buflen) {
        unsigned char* buf = SafeArrayAlloc<unsigned char>(1, buflen);

        if (buf == NULL)
          return -1;

        const int status = pReader->Read(pos, static_cast<long>(buflen), buf);

        if (status) {
          delete[] buf;
          return status;
        }

        info.codecPrivate = buf;
        info.codecPrivateSize = buflen;
      }
    } else if (id == libwebm::kMkvCodecName) {
      const long status =
          UnserializeString(pReader, pos, size, info.codecNameAsUTF8);

      if (status)
        return status;
    } else if (id == libwebm::kMkvCodecDelay) {
      info.codecDelay = UnserializeUInt(pReader, pos, size);
    } else if (id == libwebm::kMkvSeekPreRoll) {
      info.seekPreRoll = UnserializeUInt(pReader, pos, size);
    }

    pos += size;  // consume payload
    if (pos > track_stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != track_stop)
    return E_FILE_FORMAT_INVALID;

  if (info.number <= 0)  // not specified
    return E_FILE_FORMAT_INVALID;

  if (GetTrackByNumber(info.number))
    return E_FILE_FORMAT_INVALID;

  if (info.type <= 0)  // not specified
    return E_FILE_FORMAT_INVALID;

  info.lacing = (lacing > 0) ? true : false;

  if (info.type == Track::kVideo) {
    if (v.start < 0)
      return E_FILE_FORMAT_INVALID;

    if (a.start >= 0)
      return E_FILE_FORMAT_INVALID;

    info.settings = v;

    VideoTrack* pTrack = NULL;

    const long status = VideoTrack::Parse(m_pSegment, info, element_start,
                                          element_size, pTrack);

    if (status)
      return status;

    pResult = pTrack;
    assert(pResult);

    if (e.start >= 0)
      pResult->ParseContentEncodingsEntry(e.start, e.size);
  } else if (info.type == Track::kAudio) {
    if (a.start < 0)
      return E_FILE_FORMAT_INVALID;

    if (v.start >= 0)
      return E_FILE_FORMAT_INVALID;

    info.settings = a;

    AudioTrack* pTrack = NULL;

    const long status = AudioTrack::Parse(m_pSegment, info, element_start,
                                          element_size, pTrack);

    if (status)
      return status;

    pResult = pTrack;
    assert(pResult);

    if (e.start >= 0)
      pResult->ParseContentEncodingsEntry(e.start, e.size);
  } else {
    // neither video nor audio - probably metadata or subtitles

    if (a.start >= 0)
      return E_FILE_FORMAT_INVALID;

    if (v.start >= 0)
      return E_FILE_FORMAT_INVALID;

    if (info.type == Track::kMetadata && e.start >= 0)
      return E_FILE_FORMAT_INVALID;

    info.settings.start = -1;
    info.settings.size = 0;

    Track* pTrack = NULL;

    const long status =
        Track::Create(m_pSegment, info, element_start, element_size, pTrack);

    if (status)
      return status;

    pResult = pTrack;
    assert(pResult);
  }

  return 0;  // success
}

Tracks::~Tracks() {
  Track** i = m_trackEntries;
  Track** const j = m_trackEntriesEnd;

  while (i != j) {
    Track* const pTrack = *i++;
    delete pTrack;
  }

  delete[] m_trackEntries;
}

const Track* Tracks::GetTrackByNumber(long tn) const {
  if (tn < 0)
    return NULL;

  Track** i = m_trackEntries;
  Track** const j = m_trackEntriesEnd;

  while (i != j) {
    Track* const pTrack = *i++;

    if (pTrack == NULL)
      continue;

    if (tn == pTrack->GetNumber())
      return pTrack;
  }

  return NULL;  // not found
}

const Track* Tracks::GetTrackByIndex(unsigned long idx) const {
  const ptrdiff_t count = m_trackEntriesEnd - m_trackEntries;

  if (idx >= static_cast<unsigned long>(count))
    return NULL;

  return m_trackEntries[idx];
}

long Cluster::Load(long long& pos, long& len) const {
  if (m_pSegment == NULL)
    return E_PARSE_FAILED;

  if (m_timecode >= 0)  // at least partially loaded
    return 0;

  if (m_pos != m_element_start || m_element_size >= 0)
    return E_PARSE_FAILED;

  IMkvReader* const pReader = m_pSegment->m_pReader;
  long long total, avail;
  const int status = pReader->Length(&total, &avail);

  if (status < 0)  // error
    return status;

  if (total >= 0 && (avail > total || m_pos > total))
    return E_FILE_FORMAT_INVALID;

  pos = m_pos;

  long long cluster_size = -1;

  if ((pos + 1) > avail) {
    len = 1;
    return E_BUFFER_NOT_FULL;
  }

  long long result = GetUIntLength(pReader, pos, len);

  if (result < 0)  // error or underflow
    return static_cast<long>(result);

  if (result > 0)
    return E_BUFFER_NOT_FULL;

  if ((pos + len) > avail)
    return E_BUFFER_NOT_FULL;

  const long long id_ = ReadID(pReader, pos, len);

  if (id_ < 0)  // error
    return static_cast<long>(id_);

  if (id_ != libwebm::kMkvCluster)
    return E_FILE_FORMAT_INVALID;

  pos += len;  // consume id

  // read cluster size

  if ((pos + 1) > avail) {
    len = 1;
    return E_BUFFER_NOT_FULL;
  }

  result = GetUIntLength(pReader, pos, len);

  if (result < 0)  // error
    return static_cast<long>(result);

  if (result > 0)
    return E_BUFFER_NOT_FULL;

  if ((pos + len) > avail)
    return E_BUFFER_NOT_FULL;

  const long long size = ReadUInt(pReader, pos, len);

  if (size < 0)  // error
    return static_cast<long>(cluster_size);

  if (size == 0)
    return E_FILE_FORMAT_INVALID;

  pos += len;  // consume length of size of element

  const long long unknown_size = (1LL << (7 * len)) - 1;

  if (size != unknown_size)
    cluster_size = size;

  // pos points to start of payload
  long long timecode = -1;
  long long new_pos = -1;
  bool bBlock = false;

  long long cluster_stop = (cluster_size < 0) ? -1 : pos + cluster_size;

  for (;;) {
    if ((cluster_stop >= 0) && (pos >= cluster_stop))
      break;

    // Parse ID

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    long long result = GetUIntLength(pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)
      return E_BUFFER_NOT_FULL;

    if ((cluster_stop >= 0) && ((pos + len) > cluster_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long id = ReadID(pReader, pos, len);

    if (id < 0)  // error
      return static_cast<long>(id);

    if (id == 0)
      return E_FILE_FORMAT_INVALID;

    // This is the distinguished set of ID's we use to determine
    // that we have exhausted the sub-element's inside the cluster
    // whose ID we parsed earlier.

    if (id == libwebm::kMkvCluster)
      break;

    if (id == libwebm::kMkvCues)
      break;

    pos += len;  // consume ID field

    // Parse Size

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    result = GetUIntLength(pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)
      return E_BUFFER_NOT_FULL;

    if ((cluster_stop >= 0) && ((pos + len) > cluster_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long size = ReadUInt(pReader, pos, len);

    if (size < 0)  // error
      return static_cast<long>(size);

    const long long unknown_size = (1LL << (7 * len)) - 1;

    if (size == unknown_size)
      return E_FILE_FORMAT_INVALID;

    pos += len;  // consume size field

    if ((cluster_stop >= 0) && (pos > cluster_stop))
      return E_FILE_FORMAT_INVALID;

    // pos now points to start of payload

    if (size == 0)
      continue;

    if ((cluster_stop >= 0) && ((pos + size) > cluster_stop))
      return E_FILE_FORMAT_INVALID;

    if (id == libwebm::kMkvTimecode) {
      len = static_cast<long>(size);

      if ((pos + size) > avail)
        return E_BUFFER_NOT_FULL;

      timecode = UnserializeUInt(pReader, pos, size);

      if (timecode < 0)  // error (or underflow)
        return static_cast<long>(timecode);

      new_pos = pos + size;

      if (bBlock)
        break;
    } else if (id == libwebm::kMkvBlockGroup) {
      bBlock = true;
      break;
    } else if (id == libwebm::kMkvSimpleBlock) {
      bBlock = true;
      break;
    }

    pos += size;  // consume payload
    if (cluster_stop >= 0 && pos > cluster_stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (cluster_stop >= 0 && pos > cluster_stop)
    return E_FILE_FORMAT_INVALID;

  if (timecode < 0)  // no timecode found
    return E_FILE_FORMAT_INVALID;

  if (!bBlock)
    return E_FILE_FORMAT_INVALID;

  m_pos = new_pos;  // designates position just beyond timecode payload
  m_timecode = timecode;  // m_timecode >= 0 means we're partially loaded

  if (cluster_size >= 0)
    m_element_size = cluster_stop - m_element_start;

  return 0;
}

long Cluster::Parse(long long& pos, long& len) const {
  long status = Load(pos, len);

  if (status < 0)
    return status;

  if (m_pos < m_element_start || m_timecode < 0)
    return E_PARSE_FAILED;

  const long long cluster_stop =
      (m_element_size < 0) ? -1 : m_element_start + m_element_size;

  if ((cluster_stop >= 0) && (m_pos >= cluster_stop))
    return 1;  // nothing else to do

  IMkvReader* const pReader = m_pSegment->m_pReader;

  long long total, avail;

  status = pReader->Length(&total, &avail);

  if (status < 0)  // error
    return status;

  if (total >= 0 && avail > total)
    return E_FILE_FORMAT_INVALID;

  pos = m_pos;

  for (;;) {
    if ((cluster_stop >= 0) && (pos >= cluster_stop))
      break;

    if ((total >= 0) && (pos >= total)) {
      if (m_element_size < 0)
        m_element_size = pos - m_element_start;

      break;
    }

    // Parse ID

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    long long result = GetUIntLength(pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)
      return E_BUFFER_NOT_FULL;

    if ((cluster_stop >= 0) && ((pos + len) > cluster_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long id = ReadID(pReader, pos, len);

    if (id < 0)
      return E_FILE_FORMAT_INVALID;

    // This is the distinguished set of ID's we use to determine
    // that we have exhausted the sub-element's inside the cluster
    // whose ID we parsed earlier.

    if ((id == libwebm::kMkvCluster) || (id == libwebm::kMkvCues)) {
      if (m_element_size < 0)
        m_element_size = pos - m_element_start;

      break;
    }

    pos += len;  // consume ID field

    // Parse Size

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    result = GetUIntLength(pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)
      return E_BUFFER_NOT_FULL;

    if ((cluster_stop >= 0) && ((pos + len) > cluster_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long size = ReadUInt(pReader, pos, len);

    if (size < 0)  // error
      return static_cast<long>(size);

    const long long unknown_size = (1LL << (7 * len)) - 1;

    if (size == unknown_size)
      return E_FILE_FORMAT_INVALID;

    pos += len;  // consume size field

    if ((cluster_stop >= 0) && (pos > cluster_stop))
      return E_FILE_FORMAT_INVALID;

    // pos now points to start of payload

    if (size == 0)
      continue;

    // const long long block_start = pos;
    const long long block_stop = pos + size;

    if (cluster_stop >= 0) {
      if (block_stop > cluster_stop) {
        if (id == libwebm::kMkvBlockGroup || id == libwebm::kMkvSimpleBlock) {
          return E_FILE_FORMAT_INVALID;
        }

        pos = cluster_stop;
        break;
      }
    } else if ((total >= 0) && (block_stop > total)) {
      m_element_size = total - m_element_start;
      pos = total;
      break;
    } else if (block_stop > avail) {
      len = static_cast<long>(size);
      return E_BUFFER_NOT_FULL;
    }

    Cluster* const this_ = const_cast<Cluster*>(this);

    if (id == libwebm::kMkvBlockGroup)
      return this_->ParseBlockGroup(size, pos, len);

    if (id == libwebm::kMkvSimpleBlock)
      return this_->ParseSimpleBlock(size, pos, len);

    pos += size;  // consume payload
    if (cluster_stop >= 0 && pos > cluster_stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (m_element_size < 1)
    return E_FILE_FORMAT_INVALID;

  m_pos = pos;
  if (cluster_stop >= 0 && m_pos > cluster_stop)
    return E_FILE_FORMAT_INVALID;

  if (m_entries_count > 0) {
    const long idx = m_entries_count - 1;

    const BlockEntry* const pLast = m_entries[idx];
    if (pLast == NULL)
      return E_PARSE_FAILED;

    const Block* const pBlock = pLast->GetBlock();
    if (pBlock == NULL)
      return E_PARSE_FAILED;

    const long long start = pBlock->m_start;

    if ((total >= 0) && (start > total))
      return E_PARSE_FAILED;  // defend against trucated stream

    const long long size = pBlock->m_size;

    const long long stop = start + size;
    if (cluster_stop >= 0 && stop > cluster_stop)
      return E_FILE_FORMAT_INVALID;

    if ((total >= 0) && (stop > total))
      return E_PARSE_FAILED;  // defend against trucated stream
  }

  return 1;  // no more entries
}

long Cluster::ParseSimpleBlock(long long block_size, long long& pos,
                               long& len) {
  const long long block_start = pos;
  const long long block_stop = pos + block_size;

  IMkvReader* const pReader = m_pSegment->m_pReader;

  long long total, avail;

  long status = pReader->Length(&total, &avail);

  if (status < 0)  // error
    return status;

  assert((total < 0) || (avail <= total));

  // parse track number

  if ((pos + 1) > avail) {
    len = 1;
    return E_BUFFER_NOT_FULL;
  }

  long long result = GetUIntLength(pReader, pos, len);

  if (result < 0)  // error
    return static_cast<long>(result);

  if (result > 0)  // weird
    return E_BUFFER_NOT_FULL;

  if ((pos + len) > block_stop)
    return E_FILE_FORMAT_INVALID;

  if ((pos + len) > avail)
    return E_BUFFER_NOT_FULL;

  const long long track = ReadUInt(pReader, pos, len);

  if (track < 0)  // error
    return static_cast<long>(track);

  if (track == 0)
    return E_FILE_FORMAT_INVALID;

  pos += len;  // consume track number

  if ((pos + 2) > block_stop)
    return E_FILE_FORMAT_INVALID;

  if ((pos + 2) > avail) {
    len = 2;
    return E_BUFFER_NOT_FULL;
  }

  pos += 2;  // consume timecode

  if ((pos + 1) > block_stop)
    return E_FILE_FORMAT_INVALID;

  if ((pos + 1) > avail) {
    len = 1;
    return E_BUFFER_NOT_FULL;
  }

  unsigned char flags;

  status = pReader->Read(pos, 1, &flags);

  if (status < 0) {  // error or underflow
    len = 1;
    return status;
  }

  ++pos;  // consume flags byte
  assert(pos <= avail);

  if (pos >= block_stop)
    return E_FILE_FORMAT_INVALID;

  const int lacing = int(flags & 0x06) >> 1;

  if ((lacing != 0) && (block_stop > avail)) {
    len = static_cast<long>(block_stop - pos);
    return E_BUFFER_NOT_FULL;
  }

  status = CreateBlock(libwebm::kMkvSimpleBlock, block_start, block_size,
                       0);  // DiscardPadding

  if (status != 0)
    return status;

  m_pos = block_stop;

  return 0;  // success
}

long Cluster::ParseBlockGroup(long long payload_size, long long& pos,
                              long& len) {
  const long long payload_start = pos;
  const long long payload_stop = pos + payload_size;

  IMkvReader* const pReader = m_pSegment->m_pReader;

  long long total, avail;

  long status = pReader->Length(&total, &avail);

  if (status < 0)  // error
    return status;

  assert((total < 0) || (avail <= total));

  if ((total >= 0) && (payload_stop > total))
    return E_FILE_FORMAT_INVALID;

  if (payload_stop > avail) {
    len = static_cast<long>(payload_size);
    return E_BUFFER_NOT_FULL;
  }

  long long discard_padding = 0;

  while (pos < payload_stop) {
    // parse sub-block element ID

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    long long result = GetUIntLength(pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)  // weird
      return E_BUFFER_NOT_FULL;

    if ((pos + len) > payload_stop)
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long id = ReadID(pReader, pos, len);

    if (id < 0)  // error
      return static_cast<long>(id);

    if (id == 0)  // not a valid ID
      return E_FILE_FORMAT_INVALID;

    pos += len;  // consume ID field

    // Parse Size

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    result = GetUIntLength(pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)  // weird
      return E_BUFFER_NOT_FULL;

    if ((pos + len) > payload_stop)
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long size = ReadUInt(pReader, pos, len);

    if (size < 0)  // error
      return static_cast<long>(size);

    pos += len;  // consume size field

    // pos now points to start of sub-block group payload

    if (pos > payload_stop)
      return E_FILE_FORMAT_INVALID;

    if (size == 0)  // weird
      continue;

    const long long unknown_size = (1LL << (7 * len)) - 1;

    if (size == unknown_size)
      return E_FILE_FORMAT_INVALID;

    if (id == libwebm::kMkvDiscardPadding) {
      status = UnserializeInt(pReader, pos, size, discard_padding);

      if (status < 0)  // error
        return status;
    }

    if (id != libwebm::kMkvBlock) {
      pos += size;  // consume sub-part of block group

      if (pos > payload_stop)
        return E_FILE_FORMAT_INVALID;

      continue;
    }

    const long long block_stop = pos + size;

    if (block_stop > payload_stop)
      return E_FILE_FORMAT_INVALID;

    // parse track number

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    result = GetUIntLength(pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)  // weird
      return E_BUFFER_NOT_FULL;

    if ((pos + len) > block_stop)
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long track = ReadUInt(pReader, pos, len);

    if (track < 0)  // error
      return static_cast<long>(track);

    if (track == 0)
      return E_FILE_FORMAT_INVALID;

    pos += len;  // consume track number

    if ((pos + 2) > block_stop)
      return E_FILE_FORMAT_INVALID;

    if ((pos + 2) > avail) {
      len = 2;
      return E_BUFFER_NOT_FULL;
    }

    pos += 2;  // consume timecode

    if ((pos + 1) > block_stop)
      return E_FILE_FORMAT_INVALID;

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    unsigned char flags;

    status = pReader->Read(pos, 1, &flags);

    if (status < 0) {  // error or underflow
      len = 1;
      return status;
    }

    ++pos;  // consume flags byte
    assert(pos <= avail);

    if (pos >= block_stop)
      return E_FILE_FORMAT_INVALID;

    const int lacing = int(flags & 0x06) >> 1;

    if ((lacing != 0) && (block_stop > avail)) {
      len = static_cast<long>(block_stop - pos);
      return E_BUFFER_NOT_FULL;
    }

    pos = block_stop;  // consume block-part of block group
    if (pos > payload_stop)
      return E_FILE_FORMAT_INVALID;
  }

  if (pos != payload_stop)
    return E_FILE_FORMAT_INVALID;

  status = CreateBlock(libwebm::kMkvBlockGroup, payload_start, payload_size,
                       discard_padding);
  if (status != 0)
    return status;

  m_pos = payload_stop;

  return 0;  // success
}

long Cluster::GetEntry(long index, const mkvparser::BlockEntry*& pEntry) const {
  assert(m_pos >= m_element_start);

  pEntry = NULL;

  if (index < 0)
    return -1;  // generic error

  if (m_entries_count < 0)
    return E_BUFFER_NOT_FULL;

  assert(m_entries);
  assert(m_entries_size > 0);
  assert(m_entries_count <= m_entries_size);

  if (index < m_entries_count) {
    pEntry = m_entries[index];
    assert(pEntry);

    return 1;  // found entry
  }

  if (m_element_size < 0)  // we don't know cluster end yet
    return E_BUFFER_NOT_FULL;  // underflow

  const long long element_stop = m_element_start + m_element_size;

  if (m_pos >= element_stop)
    return 0;  // nothing left to parse

  return E_BUFFER_NOT_FULL;  // underflow, since more remains to be parsed
}

Cluster* Cluster::Create(Segment* pSegment, long idx, long long off) {
  if (!pSegment || off < 0)
    return NULL;

  const long long element_start = pSegment->m_start + off;

  Cluster* const pCluster =
      new (std::nothrow) Cluster(pSegment, idx, element_start);

  return pCluster;
}

Cluster::Cluster()
    : m_pSegment(NULL),
      m_element_start(0),
      m_index(0),
      m_pos(0),
      m_element_size(0),
      m_timecode(0),
      m_entries(NULL),
      m_entries_size(0),
      m_entries_count(0)  // means "no entries"
{}

Cluster::Cluster(Segment* pSegment, long idx, long long element_start
                 /* long long element_size */)
    : m_pSegment(pSegment),
      m_element_start(element_start),
      m_index(idx),
      m_pos(element_start),
      m_element_size(-1 /* element_size */),
      m_timecode(-1),
      m_entries(NULL),
      m_entries_size(0),
      m_entries_count(-1)  // means "has not been parsed yet"
{}

Cluster::~Cluster() {
  if (m_entries_count <= 0) {
    delete[] m_entries;
    return;
  }

  BlockEntry** i = m_entries;
  BlockEntry** const j = m_entries + m_entries_count;

  while (i != j) {
    BlockEntry* p = *i++;
    assert(p);

    delete p;
  }

  delete[] m_entries;
}

bool Cluster::EOS() const { return (m_pSegment == NULL); }

long Cluster::GetIndex() const { return m_index; }

long long Cluster::GetPosition() const {
  const long long pos = m_element_start - m_pSegment->m_start;
  assert(pos >= 0);

  return pos;
}

long long Cluster::GetElementSize() const { return m_element_size; }

long Cluster::HasBlockEntries(
    const Segment* pSegment,
    long long off,  // relative to start of segment payload
    long long& pos, long& len) {
  assert(pSegment);
  assert(off >= 0);  // relative to segment

  IMkvReader* const pReader = pSegment->m_pReader;

  long long total, avail;

  long status = pReader->Length(&total, &avail);

  if (status < 0)  // error
    return status;

  assert((total < 0) || (avail <= total));

  pos = pSegment->m_start + off;  // absolute

  if ((total >= 0) && (pos >= total))
    return 0;  // we don't even have a complete cluster

  const long long segment_stop =
      (pSegment->m_size < 0) ? -1 : pSegment->m_start + pSegment->m_size;

  long long cluster_stop = -1;  // interpreted later to mean "unknown size"

  {
    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    long long result = GetUIntLength(pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)  // need more data
      return E_BUFFER_NOT_FULL;

    if ((segment_stop >= 0) && ((pos + len) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    if ((total >= 0) && ((pos + len) > total))
      return 0;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long id = ReadID(pReader, pos, len);

    if (id < 0)  // error
      return static_cast<long>(id);

    if (id != libwebm::kMkvCluster)
      return E_PARSE_FAILED;

    pos += len;  // consume Cluster ID field

    // read size field

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    result = GetUIntLength(pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)  // weird
      return E_BUFFER_NOT_FULL;

    if ((segment_stop >= 0) && ((pos + len) > segment_stop))
      return E_FILE_FORMAT_INVALID;

    if ((total >= 0) && ((pos + len) > total))
      return 0;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long size = ReadUInt(pReader, pos, len);

    if (size < 0)  // error
      return static_cast<long>(size);

    if (size == 0)
      return 0;  // cluster does not have entries

    pos += len;  // consume size field

    // pos now points to start of payload

    const long long unknown_size = (1LL << (7 * len)) - 1;

    if (size != unknown_size) {
      cluster_stop = pos + size;
      assert(cluster_stop >= 0);

      if ((segment_stop >= 0) && (cluster_stop > segment_stop))
        return E_FILE_FORMAT_INVALID;

      if ((total >= 0) && (cluster_stop > total))
        // return E_FILE_FORMAT_INVALID;  //too conservative
        return 0;  // cluster does not have any entries
    }
  }

  for (;;) {
    if ((cluster_stop >= 0) && (pos >= cluster_stop))
      return 0;  // no entries detected

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    long long result = GetUIntLength(pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)  // need more data
      return E_BUFFER_NOT_FULL;

    if ((cluster_stop >= 0) && ((pos + len) > cluster_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long id = ReadID(pReader, pos, len);

    if (id < 0)  // error
      return static_cast<long>(id);

    // This is the distinguished set of ID's we use to determine
    // that we have exhausted the sub-element's inside the cluster
    // whose ID we parsed earlier.

    if (id == libwebm::kMkvCluster)
      return 0;  // no entries found

    if (id == libwebm::kMkvCues)
      return 0;  // no entries found

    pos += len;  // consume id field

    if ((cluster_stop >= 0) && (pos >= cluster_stop))
      return E_FILE_FORMAT_INVALID;

    // read size field

    if ((pos + 1) > avail) {
      len = 1;
      return E_BUFFER_NOT_FULL;
    }

    result = GetUIntLength(pReader, pos, len);

    if (result < 0)  // error
      return static_cast<long>(result);

    if (result > 0)  // underflow
      return E_BUFFER_NOT_FULL;

    if ((cluster_stop >= 0) && ((pos + len) > cluster_stop))
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > avail)
      return E_BUFFER_NOT_FULL;

    const long long size = ReadUInt(pReader, pos, len);

    if (size < 0)  // error
      return static_cast<long>(size);

    pos += len;  // consume size field

    // pos now points to start of payload

    if ((cluster_stop >= 0) && (pos > cluster_stop))
      return E_FILE_FORMAT_INVALID;

    if (size == 0)  // weird
      continue;

    const long long unknown_size = (1LL << (7 * len)) - 1;

    if (size == unknown_size)
      return E_FILE_FORMAT_INVALID;  // not supported inside cluster

    if ((cluster_stop >= 0) && ((pos + size) > cluster_stop))
      return E_FILE_FORMAT_INVALID;

    if (id == libwebm::kMkvBlockGroup)
      return 1;  // have at least one entry

    if (id == libwebm::kMkvSimpleBlock)
      return 1;  // have at least one entry

    pos += size;  // consume payload
    if (cluster_stop >= 0 && pos > cluster_stop)
      return E_FILE_FORMAT_INVALID;
  }
}

long long Cluster::GetTimeCode() const {
  long long pos;
  long len;

  const long status = Load(pos, len);

  if (status < 0)  // error
    return status;

  return m_timecode;
}

long long Cluster::GetTime() const {
  const long long tc = GetTimeCode();

  if (tc < 0)
    return tc;

  const SegmentInfo* const pInfo = m_pSegment->GetInfo();
  assert(pInfo);

  const long long scale = pInfo->GetTimeCodeScale();
  assert(scale >= 1);

  const long long t = m_timecode * scale;

  return t;
}

long long Cluster::GetFirstTime() const {
  const BlockEntry* pEntry;

  const long status = GetFirst(pEntry);

  if (status < 0)  // error
    return status;

  if (pEntry == NULL)  // empty cluster
    return GetTime();

  const Block* const pBlock = pEntry->GetBlock();
  assert(pBlock);

  return pBlock->GetTime(this);
}

long long Cluster::GetLastTime() const {
  const BlockEntry* pEntry;

  const long status = GetLast(pEntry);

  if (status < 0)  // error
    return status;

  if (pEntry == NULL)  // empty cluster
    return GetTime();

  const Block* const pBlock = pEntry->GetBlock();
  assert(pBlock);

  return pBlock->GetTime(this);
}

long Cluster::CreateBlock(long long id,
                          long long pos,  // absolute pos of payload
                          long long size, long long discard_padding) {
  if (id != libwebm::kMkvBlockGroup && id != libwebm::kMkvSimpleBlock)
    return E_PARSE_FAILED;

  if (m_entries_count < 0) {  // haven't parsed anything yet
    assert(m_entries == NULL);
    assert(m_entries_size == 0);

    m_entries_size = 1024;
    m_entries = new (std::nothrow) BlockEntry*[m_entries_size];
    if (m_entries == NULL)
      return -1;

    m_entries_count = 0;
  } else {
    assert(m_entries);
    assert(m_entries_size > 0);
    assert(m_entries_count <= m_entries_size);

    if (m_entries_count >= m_entries_size) {
      const long entries_size = 2 * m_entries_size;

      BlockEntry** const entries = new (std::nothrow) BlockEntry*[entries_size];
      if (entries == NULL)
        return -1;

      BlockEntry** src = m_entries;
      BlockEntry** const src_end = src + m_entries_count;

      BlockEntry** dst = entries;

      while (src != src_end)
        *dst++ = *src++;

      delete[] m_entries;

      m_entries = entries;
      m_entries_size = entries_size;
    }
  }

  if (id == libwebm::kMkvBlockGroup)
    return CreateBlockGroup(pos, size, discard_padding);
  else
    return CreateSimpleBlock(pos, size);
}

long Cluster::CreateBlockGroup(long long start_offset, long long size,
                               long long discard_padding) {
  assert(m_entries);
  assert(m_entries_size > 0);
  assert(m_entries_count >= 0);
  assert(m_entries_count < m_entries_size);

  IMkvReader* const pReader = m_pSegment->m_pReader;

  long long pos = start_offset;
  const long long stop = start_offset + size;

  // For WebM files, there is a bias towards previous reference times
  //(in order to support alt-ref frames, which refer back to the previous
  // keyframe).  Normally a 0 value is not possible, but here we tenatively
  // allow 0 as the value of a reference frame, with the interpretation
  // that this is a "previous" reference time.

  long long prev = 1;  // nonce
  long long next = 0;  // nonce
  long long duration = -1;  // really, this is unsigned

  long long bpos = -1;
  long long bsize = -1;

  while (pos < stop) {
    long len;
    const long long id = ReadID(pReader, pos, len);
    if (id < 0 || (pos + len) > stop)
      return E_FILE_FORMAT_INVALID;

    pos += len;  // consume ID

    const long long size = ReadUInt(pReader, pos, len);
    assert(size >= 0);  // TODO
    assert((pos + len) <= stop);

    pos += len;  // consume size

    if (id == libwebm::kMkvBlock) {
      if (bpos < 0) {  // Block ID
        bpos = pos;
        bsize = size;
      }
    } else if (id == libwebm::kMkvBlockDuration) {
      if (size > 8)
        return E_FILE_FORMAT_INVALID;

      duration = UnserializeUInt(pReader, pos, size);

      if (duration < 0)
        return E_FILE_FORMAT_INVALID;
    } else if (id == libwebm::kMkvReferenceBlock) {
      if (size > 8 || size <= 0)
        return E_FILE_FORMAT_INVALID;
      const long size_ = static_cast<long>(size);

      long long time;

      long status = UnserializeInt(pReader, pos, size_, time);
      assert(status == 0);
      if (status != 0)
        return -1;

      if (time <= 0)  // see note above
        prev = time;
      else
        next = time;
    }

    pos += size;  // consume payload
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;
  }
  if (bpos < 0)
    return E_FILE_FORMAT_INVALID;

  if (pos != stop)
    return E_FILE_FORMAT_INVALID;
  assert(bsize >= 0);

  const long idx = m_entries_count;

  BlockEntry** const ppEntry = m_entries + idx;
  BlockEntry*& pEntry = *ppEntry;

  pEntry = new (std::nothrow)
      BlockGroup(this, idx, bpos, bsize, prev, next, duration, discard_padding);

  if (pEntry == NULL)
    return -1;  // generic error

  BlockGroup* const p = static_cast<BlockGroup*>(pEntry);

  const long status = p->Parse();

  if (status == 0) {  // success
    ++m_entries_count;
    return 0;
  }

  delete pEntry;
  pEntry = 0;

  return status;
}

long Cluster::CreateSimpleBlock(long long st, long long sz) {
  assert(m_entries);
  assert(m_entries_size > 0);
  assert(m_entries_count >= 0);
  assert(m_entries_count < m_entries_size);

  const long idx = m_entries_count;

  BlockEntry** const ppEntry = m_entries + idx;
  BlockEntry*& pEntry = *ppEntry;

  pEntry = new (std::nothrow) SimpleBlock(this, idx, st, sz);

  if (pEntry == NULL)
    return -1;  // generic error

  SimpleBlock* const p = static_cast<SimpleBlock*>(pEntry);

  const long status = p->Parse();

  if (status == 0) {
    ++m_entries_count;
    return 0;
  }

  delete pEntry;
  pEntry = 0;

  return status;
}

long Cluster::GetFirst(const BlockEntry*& pFirst) const {
  if (m_entries_count <= 0) {
    long long pos;
    long len;

    const long status = Parse(pos, len);

    if (status < 0) {  // error
      pFirst = NULL;
      return status;
    }

    if (m_entries_count <= 0) {  // empty cluster
      pFirst = NULL;
      return 0;
    }
  }

  assert(m_entries);

  pFirst = m_entries[0];
  assert(pFirst);

  return 0;  // success
}

long Cluster::GetLast(const BlockEntry*& pLast) const {
  for (;;) {
    long long pos;
    long len;

    const long status = Parse(pos, len);

    if (status < 0) {  // error
      pLast = NULL;
      return status;
    }

    if (status > 0)  // no new block
      break;
  }

  if (m_entries_count <= 0) {
    pLast = NULL;
    return 0;
  }

  assert(m_entries);

  const long idx = m_entries_count - 1;

  pLast = m_entries[idx];
  assert(pLast);

  return 0;
}

long Cluster::GetNext(const BlockEntry* pCurr, const BlockEntry*& pNext) const {
  assert(pCurr);
  assert(m_entries);
  assert(m_entries_count > 0);

  size_t idx = pCurr->GetIndex();
  assert(idx < size_t(m_entries_count));
  assert(m_entries[idx] == pCurr);

  ++idx;

  if (idx >= size_t(m_entries_count)) {
    long long pos;
    long len;

    const long status = Parse(pos, len);

    if (status < 0) {  // error
      pNext = NULL;
      return status;
    }

    if (status > 0) {
      pNext = NULL;
      return 0;
    }

    assert(m_entries);
    assert(m_entries_count > 0);
    assert(idx < size_t(m_entries_count));
  }

  pNext = m_entries[idx];
  assert(pNext);

  return 0;
}

long Cluster::GetEntryCount() const { return m_entries_count; }

const BlockEntry* Cluster::GetEntry(const Track* pTrack,
                                    long long time_ns) const {
  assert(pTrack);

  if (m_pSegment == NULL)  // this is the special EOS cluster
    return pTrack->GetEOS();

  const BlockEntry* pResult = pTrack->GetEOS();

  long index = 0;

  for (;;) {
    if (index >= m_entries_count) {
      long long pos;
      long len;

      const long status = Parse(pos, len);
      assert(status >= 0);

      if (status > 0)  // completely parsed, and no more entries
        return pResult;

      if (status < 0)  // should never happen
        return 0;

      assert(m_entries);
      assert(index < m_entries_count);
    }

    const BlockEntry* const pEntry = m_entries[index];
    assert(pEntry);
    assert(!pEntry->EOS());

    const Block* const pBlock = pEntry->GetBlock();
    assert(pBlock);

    if (pBlock->GetTrackNumber() != pTrack->GetNumber()) {
      ++index;
      continue;
    }

    if (pTrack->VetEntry(pEntry)) {
      if (time_ns < 0)  // just want first candidate block
        return pEntry;

      const long long ns = pBlock->GetTime(this);

      if (ns > time_ns)
        return pResult;

      pResult = pEntry;  // have a candidate
    } else if (time_ns >= 0) {
      const long long ns = pBlock->GetTime(this);

      if (ns > time_ns)
        return pResult;
    }

    ++index;
  }
}

const BlockEntry* Cluster::GetEntry(const CuePoint& cp,
                                    const CuePoint::TrackPosition& tp) const {
  assert(m_pSegment);
  const long long tc = cp.GetTimeCode();

  if (tp.m_block > 0) {
    const long block = static_cast<long>(tp.m_block);
    const long index = block - 1;

    while (index >= m_entries_count) {
      long long pos;
      long len;

      const long status = Parse(pos, len);

      if (status < 0)  // TODO: can this happen?
        return NULL;

      if (status > 0)  // nothing remains to be parsed
        return NULL;
    }

    const BlockEntry* const pEntry = m_entries[index];
    assert(pEntry);
    assert(!pEntry->EOS());

    const Block* const pBlock = pEntry->GetBlock();
    assert(pBlock);

    if ((pBlock->GetTrackNumber() == tp.m_track) &&
        (pBlock->GetTimeCode(this) == tc)) {
      return pEntry;
    }
  }

  long index = 0;

  for (;;) {
    if (index >= m_entries_count) {
      long long pos;
      long len;

      const long status = Parse(pos, len);

      if (status < 0)  // TODO: can this happen?
        return NULL;

      if (status > 0)  // nothing remains to be parsed
        return NULL;

      assert(m_entries);
      assert(index < m_entries_count);
    }

    const BlockEntry* const pEntry = m_entries[index];
    assert(pEntry);
    assert(!pEntry->EOS());

    const Block* const pBlock = pEntry->GetBlock();
    assert(pBlock);

    if (pBlock->GetTrackNumber() != tp.m_track) {
      ++index;
      continue;
    }

    const long long tc_ = pBlock->GetTimeCode(this);

    if (tc_ < tc) {
      ++index;
      continue;
    }

    if (tc_ > tc)
      return NULL;

    const Tracks* const pTracks = m_pSegment->GetTracks();
    assert(pTracks);

    const long tn = static_cast<long>(tp.m_track);
    const Track* const pTrack = pTracks->GetTrackByNumber(tn);

    if (pTrack == NULL)
      return NULL;

    const long long type = pTrack->GetType();

    if (type == 2)  // audio
      return pEntry;

    if (type != 1)  // not video
      return NULL;

    if (!pBlock->IsKey())
      return NULL;

    return pEntry;
  }
}

BlockEntry::BlockEntry(Cluster* p, long idx) : m_pCluster(p), m_index(idx) {}
BlockEntry::~BlockEntry() {}
const Cluster* BlockEntry::GetCluster() const { return m_pCluster; }
long BlockEntry::GetIndex() const { return m_index; }

SimpleBlock::SimpleBlock(Cluster* pCluster, long idx, long long start,
                         long long size)
    : BlockEntry(pCluster, idx), m_block(start, size, 0) {}

long SimpleBlock::Parse() { return m_block.Parse(m_pCluster); }
BlockEntry::Kind SimpleBlock::GetKind() const { return kBlockSimple; }
const Block* SimpleBlock::GetBlock() const { return &m_block; }

BlockGroup::BlockGroup(Cluster* pCluster, long idx, long long block_start,
                       long long block_size, long long prev, long long next,
                       long long duration, long long discard_padding)
    : BlockEntry(pCluster, idx),
      m_block(block_start, block_size, discard_padding),
      m_prev(prev),
      m_next(next),
      m_duration(duration) {}

long BlockGroup::Parse() {
  const long status = m_block.Parse(m_pCluster);

  if (status)
    return status;

  m_block.SetKey((m_prev > 0) && (m_next <= 0));

  return 0;
}

BlockEntry::Kind BlockGroup::GetKind() const { return kBlockGroup; }
const Block* BlockGroup::GetBlock() const { return &m_block; }
long long BlockGroup::GetPrevTimeCode() const { return m_prev; }
long long BlockGroup::GetNextTimeCode() const { return m_next; }
long long BlockGroup::GetDurationTimeCode() const { return m_duration; }

Block::Block(long long start, long long size_, long long discard_padding)
    : m_start(start),
      m_size(size_),
      m_track(0),
      m_timecode(-1),
      m_flags(0),
      m_frames(NULL),
      m_frame_count(-1),
      m_discard_padding(discard_padding) {}

Block::~Block() { delete[] m_frames; }

long Block::Parse(const Cluster* pCluster) {
  if (pCluster == NULL)
    return -1;

  if (pCluster->m_pSegment == NULL)
    return -1;

  assert(m_start >= 0);
  assert(m_size >= 0);
  assert(m_track <= 0);
  assert(m_frames == NULL);
  assert(m_frame_count <= 0);

  long long pos = m_start;
  const long long stop = m_start + m_size;

  long len;

  IMkvReader* const pReader = pCluster->m_pSegment->m_pReader;

  m_track = ReadUInt(pReader, pos, len);

  if (m_track <= 0)
    return E_FILE_FORMAT_INVALID;

  if ((pos + len) > stop)
    return E_FILE_FORMAT_INVALID;

  pos += len;  // consume track number

  if ((stop - pos) < 2)
    return E_FILE_FORMAT_INVALID;

  long status;
  long long value;

  status = UnserializeInt(pReader, pos, 2, value);

  if (status)
    return E_FILE_FORMAT_INVALID;

  if (value < SHRT_MIN)
    return E_FILE_FORMAT_INVALID;

  if (value > SHRT_MAX)
    return E_FILE_FORMAT_INVALID;

  m_timecode = static_cast<short>(value);

  pos += 2;

  if ((stop - pos) <= 0)
    return E_FILE_FORMAT_INVALID;

  status = pReader->Read(pos, 1, &m_flags);

  if (status)
    return E_FILE_FORMAT_INVALID;

  const int lacing = int(m_flags & 0x06) >> 1;

  ++pos;  // consume flags byte

  if (lacing == 0) {  // no lacing
    if (pos > stop)
      return E_FILE_FORMAT_INVALID;

    m_frame_count = 1;
    m_frames = new (std::nothrow) Frame[m_frame_count];
    if (m_frames == NULL)
      return -1;

    Frame& f = m_frames[0];
    f.pos = pos;

    const long long frame_size = stop - pos;

    if (frame_size > LONG_MAX || frame_size <= 0)
      return E_FILE_FORMAT_INVALID;

    f.len = static_cast<long>(frame_size);

    return 0;  // success
  }

  if (pos >= stop)
    return E_FILE_FORMAT_INVALID;

  unsigned char biased_count;

  status = pReader->Read(pos, 1, &biased_count);

  if (status)
    return E_FILE_FORMAT_INVALID;

  ++pos;  // consume frame count
  if (pos > stop)
    return E_FILE_FORMAT_INVALID;

  m_frame_count = int(biased_count) + 1;

  m_frames = new (std::nothrow) Frame[m_frame_count];
  if (m_frames == NULL)
    return -1;

  if (!m_frames)
    return E_FILE_FORMAT_INVALID;

  if (lacing == 1) {  // Xiph
    Frame* pf = m_frames;
    Frame* const pf_end = pf + m_frame_count;

    long long size = 0;
    int frame_count = m_frame_count;

    while (frame_count > 1) {
      long frame_size = 0;

      for (;;) {
        unsigned char val;

        if (pos >= stop)
          return E_FILE_FORMAT_INVALID;

        status = pReader->Read(pos, 1, &val);

        if (status)
          return E_FILE_FORMAT_INVALID;

        ++pos;  // consume xiph size byte

        frame_size += val;

        if (val < 255)
          break;
      }

      Frame& f = *pf++;
      assert(pf < pf_end);
      if (pf >= pf_end)
        return E_FILE_FORMAT_INVALID;

      f.pos = 0;  // patch later

      if (frame_size <= 0)
        return E_FILE_FORMAT_INVALID;

      f.len = frame_size;
      size += frame_size;  // contribution of this frame

      --frame_count;
    }

    if (pf >= pf_end || pos > stop)
      return E_FILE_FORMAT_INVALID;

    {
      Frame& f = *pf++;

      if (pf != pf_end)
        return E_FILE_FORMAT_INVALID;

      f.pos = 0;  // patch later

      const long long total_size = stop - pos;

      if (total_size < size)
        return E_FILE_FORMAT_INVALID;

      const long long frame_size = total_size - size;

      if (frame_size > LONG_MAX || frame_size <= 0)
        return E_FILE_FORMAT_INVALID;

      f.len = static_cast<long>(frame_size);
    }

    pf = m_frames;
    while (pf != pf_end) {
      Frame& f = *pf++;
      assert((pos + f.len) <= stop);

      if ((pos + f.len) > stop)
        return E_FILE_FORMAT_INVALID;

      f.pos = pos;
      pos += f.len;
    }

    assert(pos == stop);
    if (pos != stop)
      return E_FILE_FORMAT_INVALID;

  } else if (lacing == 2) {  // fixed-size lacing
    if (pos >= stop)
      return E_FILE_FORMAT_INVALID;

    const long long total_size = stop - pos;

    if ((total_size % m_frame_count) != 0)
      return E_FILE_FORMAT_INVALID;

    const long long frame_size = total_size / m_frame_count;

    if (frame_size > LONG_MAX || frame_size <= 0)
      return E_FILE_FORMAT_INVALID;

    Frame* pf = m_frames;
    Frame* const pf_end = pf + m_frame_count;

    while (pf != pf_end) {
      assert((pos + frame_size) <= stop);
      if ((pos + frame_size) > stop)
        return E_FILE_FORMAT_INVALID;

      Frame& f = *pf++;

      f.pos = pos;
      f.len = static_cast<long>(frame_size);

      pos += frame_size;
    }

    assert(pos == stop);
    if (pos != stop)
      return E_FILE_FORMAT_INVALID;

  } else {
    assert(lacing == 3);  // EBML lacing

    if (pos >= stop)
      return E_FILE_FORMAT_INVALID;

    long long size = 0;
    int frame_count = m_frame_count;

    long long frame_size = ReadUInt(pReader, pos, len);

    if (frame_size <= 0)
      return E_FILE_FORMAT_INVALID;

    if (frame_size > LONG_MAX)
      return E_FILE_FORMAT_INVALID;

    if ((pos + len) > stop)
      return E_FILE_FORMAT_INVALID;

    pos += len;  // consume length of size of first frame

    if ((pos + frame_size) > stop)
      return E_FILE_FORMAT_INVALID;

    Frame* pf = m_frames;
    Frame* const pf_end = pf + m_frame_count;

    {
      Frame& curr = *pf;

      curr.pos = 0;  // patch later

      curr.len = static_cast<long>(frame_size);
      size += curr.len;  // contribution of this frame
    }

    --frame_count;

    while (frame_count > 1) {
      if (pos >= stop)
        return E_FILE_FORMAT_INVALID;

      assert(pf < pf_end);
      if (pf >= pf_end)
        return E_FILE_FORMAT_INVALID;

      const Frame& prev = *pf++;
      assert(prev.len == frame_size);
      if (prev.len != frame_size)
        return E_FILE_FORMAT_INVALID;

      assert(pf < pf_end);
      if (pf >= pf_end)
        return E_FILE_FORMAT_INVALID;

      Frame& curr = *pf;

      curr.pos = 0;  // patch later

      const long long delta_size_ = ReadUInt(pReader, pos, len);

      if (delta_size_ < 0)
        return E_FILE_FORMAT_INVALID;

      if ((pos + len) > stop)
        return E_FILE_FORMAT_INVALID;

      pos += len;  // consume length of (delta) size
      if (pos > stop)
        return E_FILE_FORMAT_INVALID;

      const long exp = 7 * len - 1;
      const long long bias = (1LL << exp) - 1LL;
      const long long delta_size = delta_size_ - bias;

      frame_size += delta_size;

      if (frame_size <= 0)
        return E_FILE_FORMAT_INVALID;

      if (frame_size > LONG_MAX)
        return E_FILE_FORMAT_INVALID;

      curr.len = static_cast<long>(frame_size);
      // Check if size + curr.len could overflow.
      if (size > LLONG_MAX - curr.len) {
        return E_FILE_FORMAT_INVALID;
      }
      size += curr.len;  // contribution of this frame

      --frame_count;
    }

    // parse last frame
    if (frame_count > 0) {
      if (pos > stop || pf >= pf_end)
        return E_FILE_FORMAT_INVALID;

      const Frame& prev = *pf++;
      assert(prev.len == frame_size);
      if (prev.len != frame_size)
        return E_FILE_FORMAT_INVALID;

      if (pf >= pf_end)
        return E_FILE_FORMAT_INVALID;

      Frame& curr = *pf++;
      if (pf != pf_end)
        return E_FILE_FORMAT_INVALID;

      curr.pos = 0;  // patch later

      const long long total_size = stop - pos;

      if (total_size < size)
        return E_FILE_FORMAT_INVALID;

      frame_size = total_size - size;

      if (frame_size > LONG_MAX || frame_size <= 0)
        return E_FILE_FORMAT_INVALID;

      curr.len = static_cast<long>(frame_size);
    }

    pf = m_frames;
    while (pf != pf_end) {
      Frame& f = *pf++;
      if ((pos + f.len) > stop)
        return E_FILE_FORMAT_INVALID;

      f.pos = pos;
      pos += f.len;
    }

    if (pos != stop)
      return E_FILE_FORMAT_INVALID;
  }

  return 0;  // success
}

long long Block::GetTimeCode(const Cluster* pCluster) const {
  if (pCluster == 0)
    return m_timecode;

  const long long tc0 = pCluster->GetTimeCode();
  assert(tc0 >= 0);

  // Check if tc0 + m_timecode would overflow.
  if (tc0 < 0 || LLONG_MAX - tc0 < m_timecode) {
    return -1;
  }

  const long long tc = tc0 + m_timecode;

  return tc;  // unscaled timecode units
}

long long Block::GetTime(const Cluster* pCluster) const {
  assert(pCluster);

  const long long tc = GetTimeCode(pCluster);

  const Segment* const pSegment = pCluster->m_pSegment;
  const SegmentInfo* const pInfo = pSegment->GetInfo();
  assert(pInfo);

  const long long scale = pInfo->GetTimeCodeScale();
  assert(scale >= 1);

  // Check if tc * scale could overflow.
  if (tc != 0 && scale > LLONG_MAX / tc) {
    return -1;
  }
  const long long ns = tc * scale;

  return ns;
}

long long Block::GetTrackNumber() const { return m_track; }

bool Block::IsKey() const {
  return ((m_flags & static_cast<unsigned char>(1 << 7)) != 0);
}

void Block::SetKey(bool bKey) {
  if (bKey)
    m_flags |= static_cast<unsigned char>(1 << 7);
  else
    m_flags &= 0x7F;
}

bool Block::IsInvisible() const { return bool(int(m_flags & 0x08) != 0); }

Block::Lacing Block::GetLacing() const {
  const int value = int(m_flags & 0x06) >> 1;
  return static_cast<Lacing>(value);
}

int Block::GetFrameCount() const { return m_frame_count; }

const Block::Frame& Block::GetFrame(int idx) const {
  assert(idx >= 0);
  assert(idx < m_frame_count);

  const Frame& f = m_frames[idx];
  assert(f.pos > 0);
  assert(f.len > 0);

  return f;
}

long Block::Frame::Read(IMkvReader* pReader, unsigned char* buf) const {
  assert(pReader);
  assert(buf);

  const long status = pReader->Read(pos, len, buf);
  return status;
}

long long Block::GetDiscardPadding() const { return m_discard_padding; }

}  // namespace mkvparser
