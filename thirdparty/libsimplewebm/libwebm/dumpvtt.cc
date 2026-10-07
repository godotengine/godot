// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#include <cstdio>
#include <cstdlib>

#include "webvtt/vttreader.h"
#include "webvtt/webvttparser.h"

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    fprintf(stdout, "usage: dumpvtt <vtt file>\n");
    return EXIT_SUCCESS;
  }

  libwebvtt::VttReader reader;
  const char* const filename = argv[1];

  if (int e = reader.Open(filename)) {
    (void)e;
    fprintf(stderr, "open failed\n");
    return EXIT_FAILURE;
  }

  libwebvtt::Parser parser(&reader);

  if (int e = parser.Init()) {
    (void)e;
    fprintf(stderr, "parser init failed\n");
    return EXIT_FAILURE;
  }

  for (libwebvtt::Cue cue;;) {
    const int e = parser.Parse(&cue);

    if (e < 0) {  // error
      fprintf(stderr, "error parsing cue\n");
      return EXIT_FAILURE;
    }

    if (e > 0)  // EOF
      return EXIT_SUCCESS;

    fprintf(stdout, "cue identifier: \"%s\"\n", cue.identifier.c_str());

    const libwebvtt::Time& st = cue.start_time;
    fprintf(stdout, "cue start time: \"HH=%i MM=%i SS=%i SSS=%i\"\n", st.hours,
            st.minutes, st.seconds, st.milliseconds);

    const libwebvtt::Time& sp = cue.stop_time;
    fprintf(stdout, "cue stop time: \"HH=%i MM=%i SS=%i SSS=%i\"\n", sp.hours,
            sp.minutes, sp.seconds, sp.milliseconds);

    {
      typedef libwebvtt::Cue::settings_t::const_iterator iter_t;
      iter_t i = cue.settings.begin();
      const iter_t j = cue.settings.end();

      if (i == j) {
        fprintf(stdout, "cue setting: <no settings present>\n");
      } else {
        while (i != j) {
          const libwebvtt::Setting& setting = *i++;
          fprintf(stdout, "cue setting: name=%s value=%s\n",
                  setting.name.c_str(), setting.value.c_str());
        }
      }
    }

    {
      typedef libwebvtt::Cue::payload_t::const_iterator iter_t;
      iter_t i = cue.payload.begin();
      const iter_t j = cue.payload.end();

      int idx = 1;
      while (i != j) {
        const std::string& payload = *i++;
        const char* const payload_str = payload.c_str();
        fprintf(stdout, "cue payload[%i]: \"%s\"\n", idx, payload_str);
        ++idx;
      }
    }

    fprintf(stdout, "\n");
    fflush(stdout);
  }
}
