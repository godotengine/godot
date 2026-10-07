// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#include "webvttparser.h"

#include <ctype.h>

#include <climits>
#include <cstddef>

namespace libwebvtt {

// NOLINT'ing this enum because clang-format puts it in a single line which
// makes it look really unreadable.
enum {
  kNUL = '\x00',
  kSPACE = ' ',
  kTAB = '\x09',
  kLF = '\x0A',
  kCR = '\x0D'
};  // NOLINT

Reader::~Reader() {}

LineReader::~LineReader() {}

int LineReader::GetLine(std::string* line_ptr) {
  if (line_ptr == NULL)
    return -1;

  std::string& ln = *line_ptr;
  ln.clear();

  // Consume characters from the stream, until we
  // reach end-of-line (or end-of-stream).

  // The WebVTT spec states that lines may be
  // terminated in any of these three ways:
  //  LF
  //  CR
  //  CR LF

  // We interrogate each character as we read it from the stream.
  // If we detect an end-of-line character, we consume the full
  // end-of-line indication, and we're done; otherwise, accumulate
  // the character and repeat.

  for (;;) {
    char c;
    const int e = GetChar(&c);

    if (e < 0)  // error
      return e;

    if (e > 0)  // EOF
      return (ln.empty()) ? 1 : 0;

    // We have a character, so we must first determine
    // whether we have reached end-of-line.

    if (c == kLF)
      return 0;  // handle the easy end-of-line case immediately

    if (c == kCR)
      break;  // handle the hard end-of-line case outside of loop

    if (c == '\xFE' || c == '\xFF')  // not UTF-8
      return -1;

    // To defend against pathological or malicious streams, we
    // cap the line length at some arbitrarily-large value:
    enum { kMaxLineLength = 10000 };  // arbitrary

    if (ln.length() >= kMaxLineLength)
      return -1;

    // We don't have an end-of-line character, so accumulate
    // the character in our line buffer.
    ln.push_back(c);
  }

  // We detected a CR.  We must interrogate the next character
  // in the stream, to determine whether we have a LF (which
  // would make it part of this same line).

  char c;
  const int e = GetChar(&c);

  if (e < 0)  // error
    return e;

  if (e > 0)  // EOF
    return 0;

  // If next character in the stream is not a LF, return it
  // to the stream (because it's part of the next line).
  if (c != kLF)
    UngetChar(c);

  return 0;
}

Parser::Parser(Reader* r) : reader_(r), unget_(-1) {}

Parser::~Parser() {}

int Parser::Init() {
  int e = ParseBOM();

  if (e < 0)  // error
    return e;

  if (e > 0)  // EOF
    return -1;

  // Parse "WEBVTT".  We read from the stream one character at-a-time, in
  // order to defend against non-WebVTT streams (e.g. binary files) that don't
  // happen to comprise lines of text demarcated with line terminators.

  const char kId[] = "WEBVTT";

  for (const char* p = kId; *p; ++p) {
    char c;
    e = GetChar(&c);

    if (e < 0)  // error
      return e;

    if (e > 0)  // EOF
      return -1;

    if (c != *p)
      return -1;
  }

  std::string line;

  e = GetLine(&line);

  if (e < 0)  // error
    return e;

  if (e > 0)  // EOF
    return 0;  // weird but valid

  if (!line.empty()) {
    // Parse optional characters that follow "WEBVTT"

    const char c = line[0];

    if (c != kSPACE && c != kTAB)
      return -1;
  }

  // The WebVTT spec requires that the "WEBVTT" line
  // be followed by an empty line (to separate it from
  // first cue).

  e = GetLine(&line);

  if (e < 0)  // error
    return e;

  if (e > 0)  // EOF
    return 0;  // weird but we allow it

  if (!line.empty())
    return -1;

  return 0;  // success
}

int Parser::Parse(Cue* cue) {
  if (cue == NULL)
    return -1;

  // Parse first non-blank line

  std::string line;
  int e;

  for (;;) {
    e = GetLine(&line);

    if (e)  // EOF is OK here
      return e;

    if (!line.empty())
      break;
  }

  // A WebVTT cue comprises an optional cue identifier line followed
  // by a (non-optional) timings line.  You determine whether you have
  // a timings line by scanning for the arrow token, the lexeme of which
  // may not appear in the cue identifier line.

  const char kArrow[] = "-->";
  std::string::size_type arrow_pos = line.find(kArrow);

  if (arrow_pos != std::string::npos) {
    // We found a timings line, which implies that we don't have a cue
    // identifier.

    cue->identifier.clear();
  } else {
    // We did not find a timings line, so we assume that we have a cue
    // identifier line, and then try again to find the cue timings on
    // the next line.

    cue->identifier.swap(line);

    e = GetLine(&line);

    if (e < 0)  // error
      return e;

    if (e > 0)  // EOF
      return -1;

    arrow_pos = line.find(kArrow);

    if (arrow_pos == std::string::npos)  // not a timings line
      return -1;
  }

  e = ParseTimingsLine(&line, arrow_pos, &cue->start_time, &cue->stop_time,
                       &cue->settings);

  if (e)  // error
    return e;

  // The cue payload comprises all the non-empty
  // lines that follow the timings line.

  Cue::payload_t& p = cue->payload;
  p.clear();

  for (;;) {
    e = GetLine(&line);

    if (e < 0)  // error
      return e;

    if (line.empty())
      break;

    p.push_back(line);
  }

  if (p.empty())
    return -1;

  return 0;  // success
}

int Parser::GetChar(char* c) {
  if (unget_ >= 0) {
    *c = static_cast<char>(unget_);
    unget_ = -1;
    return 0;
  }

  return reader_->GetChar(c);
}

void Parser::UngetChar(char c) { unget_ = static_cast<unsigned char>(c); }

int Parser::ParseBOM() {
  // Explanation of UTF-8 BOM:
  // http://en.wikipedia.org/wiki/Byte_order_mark

  static const char BOM[] = "\xEF\xBB\xBF";  // UTF-8 BOM

  for (int i = 0; i < 3; ++i) {
    char c;
    int e = GetChar(&c);

    if (e < 0)  // error
      return e;

    if (e > 0)  // EOF
      return 1;

    if (c != BOM[i]) {
      if (i == 0) {  // we don't have a BOM
        UngetChar(c);
        return 0;  // success
      }

      // We started a BOM, so we must finish the BOM.
      return -1;  // error
    }
  }

  return 0;  // success
}

int Parser::ParseTimingsLine(std::string* line_ptr,
                             std::string::size_type arrow_pos, Time* start_time,
                             Time* stop_time, Cue::settings_t* settings) {
  if (line_ptr == NULL)
    return -1;

  std::string& line = *line_ptr;

  if (arrow_pos == std::string::npos || arrow_pos >= line.length())
    return -1;

  // Place a NUL character at the start of the arrow token, in
  // order to demarcate the start time from remainder of line.
  line[arrow_pos] = kNUL;
  std::string::size_type idx = 0;

  int e = ParseTime(line, &idx, start_time);
  if (e)  // error
    return e;

  // Detect any junk that follows the start time,
  // but precedes the arrow symbol.

  while (char c = line[idx]) {
    if (c != kSPACE && c != kTAB)
      return -1;
    ++idx;
  }

  // Place a NUL character at the end of the line,
  // so the scanner has a place to stop, and begin
  // the scan just beyond the arrow token.

  line.push_back(kNUL);
  idx = arrow_pos + 3;

  e = ParseTime(line, &idx, stop_time);
  if (e)  // error
    return e;

  e = ParseSettings(line, idx, settings);
  if (e)  // error
    return e;

  return 0;  // success
}

int Parser::ParseTime(const std::string& line, std::string::size_type* idx_ptr,
                      Time* time) {
  if (idx_ptr == NULL)
    return -1;

  std::string::size_type& idx = *idx_ptr;

  if (idx == std::string::npos || idx >= line.length())
    return -1;

  if (time == NULL)
    return -1;

  // Consume any whitespace that precedes the timestamp.

  while (char c = line[idx]) {
    if (c != kSPACE && c != kTAB)
      break;
    ++idx;
  }

  // WebVTT timestamp syntax comes in three flavors:
  //  SS[.sss]
  //  MM:SS[.sss]
  //  HH:MM:SS[.sss]

  // Parse a generic number value.  We don't know which component
  // of the time we have yet, until we do more parsing.

  int val = ParseNumber(line, &idx);

  if (val < 0)  // error
    return val;

  Time& t = *time;

  // The presence of a colon character indicates that we have
  // an [HH:]MM:SS style syntax.

  if (line[idx] == ':') {
    // We have either HH:MM:SS or MM:SS

    // The value we just parsed is either the hours or minutes.
    // It must be followed by another number value (that is
    // either minutes or seconds).

    const int first_val = val;

    ++idx;  // consume colon

    // Parse second value

    val = ParseNumber(line, &idx);

    if (val < 0)
      return val;

    if (val >= 60)  // either MM or SS
      return -1;

    if (line[idx] == ':') {
      // We have HH:MM:SS

      t.hours = first_val;
      t.minutes = val;  // vetted above

      ++idx;  // consume MM:SS colon

      // We have parsed the hours and minutes.
      // We must now parse the seconds.

      val = ParseNumber(line, &idx);

      if (val < 0)
        return val;

      if (val >= 60)  // SS part of HH:MM:SS
        return -1;

      t.seconds = val;
    } else {
      // We have MM:SS

      // The implication here is that the hour value was omitted
      // from the timestamp (because it was 0).

      if (first_val >= 60)  // minutes
        return -1;

      t.hours = 0;
      t.minutes = first_val;
      t.seconds = val;  // vetted above
    }
  } else {
    // We have SS (only)

    // The time is expressed as total number of seconds,
    // so the seconds value has no upper bound.

    t.seconds = val;

    // Convert SS to HH:MM:SS

    t.minutes = t.seconds / 60;
    t.seconds -= t.minutes * 60;

    t.hours = t.minutes / 60;
    t.minutes -= t.hours * 60;
  }

  // We have parsed the hours, minutes, and seconds.
  // We must now parse the milliseconds.

  char c = line[idx];

  // TODO(matthewjheaney): one option here is to slightly relax the
  // syntax rules for WebVTT timestamps, to permit the comma character
  // to also be used as the seconds/milliseconds separator.  This
  // would handle streams that use localization conventions for
  // countries in Western Europe.  For now we obey the rules specified
  // in the WebVTT spec (allow "full stop" only).

  const bool have_milliseconds = (c == '.');

  if (!have_milliseconds) {
    t.milliseconds = 0;
  } else {
    ++idx;  // consume FULL STOP

    val = ParseNumber(line, &idx);

    if (val < 0)
      return val;

    if (val >= 1000)
      return -1;

    if (val < 10)
      t.milliseconds = val * 100;
    else if (val < 100)
      t.milliseconds = val * 10;
    else
      t.milliseconds = val;
  }

  // We have parsed the time proper.  We must check for any
  // junk that immediately follows the time specifier.

  c = line[idx];

  if (c != kNUL && c != kSPACE && c != kTAB)
    return -1;

  return 0;  // success
}

int Parser::ParseSettings(const std::string& line, std::string::size_type idx,
                          Cue::settings_t* settings) {
  settings->clear();

  if (idx == std::string::npos || idx >= line.length())
    return -1;

  for (;;) {
    // We must parse a line comprising a sequence of 0 or more
    // NAME:VALUE pairs, separated by whitespace.  The line iself is
    // terminated with a NUL char (indicating end-of-line).

    for (;;) {
      const char c = line[idx];

      if (c == kNUL)  // end-of-line
        return 0;  // success

      if (c != kSPACE && c != kTAB)
        break;

      ++idx;  // consume whitespace
    }

    // We have consumed the whitespace, and have not yet reached
    // end-of-line, so there is something on the line for us to parse.

    settings->push_back(Setting());
    Setting& s = settings->back();

    // Parse the NAME part of the settings pair.

    for (;;) {
      const char c = line[idx];

      if (c == ':')  // we have reached end of NAME part
        break;

      if (c == kNUL || c == kSPACE || c == kTAB)
        return -1;

      s.name.push_back(c);

      ++idx;
    }

    if (s.name.empty())
      return -1;

    ++idx;  // consume colon

    // Parse the VALUE part of the settings pair.

    for (;;) {
      const char c = line[idx];

      if (c == kNUL || c == kSPACE || c == kTAB)
        break;

      if (c == ':')  // suspicious when part of VALUE
        return -1;  // TODO(matthewjheaney): verify this behavior

      s.value.push_back(c);

      ++idx;
    }

    if (s.value.empty())
      return -1;
  }
}

int Parser::ParseNumber(const std::string& line,
                        std::string::size_type* idx_ptr) {
  if (idx_ptr == NULL)
    return -1;

  std::string::size_type& idx = *idx_ptr;

  if (idx == std::string::npos || idx >= line.length())
    return -1;

  if (!isdigit(line[idx]))
    return -1;

  int result = 0;

  while (isdigit(line[idx])) {
    const char c = line[idx];
    const int i = c - '0';

    if (result > INT_MAX / 10)
      return -1;

    result *= 10;

    if (result > INT_MAX - i)
      return -1;

    result += i;

    ++idx;
  }

  return result;
}

bool Time::operator==(const Time& rhs) const {
  if (hours != rhs.hours)
    return false;

  if (minutes != rhs.minutes)
    return false;

  if (seconds != rhs.seconds)
    return false;

  return (milliseconds == rhs.milliseconds);
}

bool Time::operator<(const Time& rhs) const {
  if (hours < rhs.hours)
    return true;

  if (hours > rhs.hours)
    return false;

  if (minutes < rhs.minutes)
    return true;

  if (minutes > rhs.minutes)
    return false;

  if (seconds < rhs.seconds)
    return true;

  if (seconds > rhs.seconds)
    return false;

  return (milliseconds < rhs.milliseconds);
}

bool Time::operator>(const Time& rhs) const { return rhs.operator<(*this); }

bool Time::operator<=(const Time& rhs) const { return !this->operator>(rhs); }

bool Time::operator>=(const Time& rhs) const { return !this->operator<(rhs); }

presentation_t Time::presentation() const {
  const presentation_t h = 1000LL * 3600LL * presentation_t(hours);
  const presentation_t m = 1000LL * 60LL * presentation_t(minutes);
  const presentation_t s = 1000LL * presentation_t(seconds);
  const presentation_t result = h + m + s + milliseconds;
  return result;
}

Time& Time::presentation(presentation_t d) {
  if (d < 0) {  // error
    hours = 0;
    minutes = 0;
    seconds = 0;
    milliseconds = 0;

    return *this;
  }

  seconds = static_cast<int>(d / 1000);
  milliseconds = static_cast<int>(d - 1000 * seconds);

  minutes = seconds / 60;
  seconds -= 60 * minutes;

  hours = minutes / 60;
  minutes -= 60 * hours;

  return *this;
}

Time& Time::operator+=(presentation_t rhs) {
  const presentation_t d = this->presentation();
  const presentation_t dd = d + rhs;
  this->presentation(dd);
  return *this;
}

Time Time::operator+(presentation_t d) const {
  Time t(*this);
  t += d;
  return t;
}

Time& Time::operator-=(presentation_t d) { return this->operator+=(-d); }

presentation_t Time::operator-(const Time& t) const {
  const presentation_t rhs = t.presentation();
  const presentation_t lhs = this->presentation();
  const presentation_t result = lhs - rhs;
  return result;
}

}  // namespace libwebvtt
