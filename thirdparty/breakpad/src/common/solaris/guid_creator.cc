// Copyright (c) 2007, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: Alfred Peng

#include <cassert>
#include <ctime>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "common/solaris/guid_creator.h"

//
// GUIDGenerator
//
// This class is used to generate random GUID.
// Currently use random number to generate a GUID. This should be OK since
// we don't expect crash to happen very offen.
//
class GUIDGenerator {
 public:
  GUIDGenerator() {
    srandom(time(NULL));
  }

  bool CreateGUID(GUID *guid) const {
    guid->data1 = random();
    guid->data2 = (uint16_t)(random());
    guid->data3 = (uint16_t)(random());
    *reinterpret_cast<uint32_t*>(&guid->data4[0]) = random();
    *reinterpret_cast<uint32_t*>(&guid->data4[4]) = random();
    return true;
  }
};

// Guid generator.
const GUIDGenerator kGuidGenerator;

bool CreateGUID(GUID *guid) {
  return kGuidGenerator.CreateGUID(guid);
}

// Parse guid to string.
bool GUIDToString(const GUID *guid, char *buf, int buf_len) {
  // Should allow more space the the max length of GUID.
  assert(buf_len > kGUIDStringLength);
  int num = snprintf(buf, buf_len, kGUIDFormatString,
                     guid->data1, guid->data2, guid->data3,
                     *reinterpret_cast<const uint32_t*>(&(guid->data4[0])),
                     *reinterpret_cast<const uint32_t*>(&(guid->data4[4])));
  if (num != kGUIDStringLength)
    return false;

  buf[num] = '\0';
  return true;
}
