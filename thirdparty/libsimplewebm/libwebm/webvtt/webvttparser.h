// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#ifndef WEBVTT_WEBVTTPARSER_H_
#define WEBVTT_WEBVTTPARSER_H_

#include <list>
#include <string>

namespace libwebvtt {

class Reader {
 public:
  // Fetch a character from the stream. Return
  // negative if error, positive if end-of-stream,
  // and 0 if a character is available.
  virtual int GetChar(char* c) = 0;

 protected:
  virtual ~Reader();
};

class LineReader : protected Reader {
 public:
  // Consume a line of text from the stream, stripping off
  // the line terminator characters.  Returns negative if error,
  // 0 on success, and positive at end-of-stream.
  int GetLine(std::string* line);

 protected:
  virtual ~LineReader();

  // Puts a character back into the stream.
  virtual void UngetChar(char c) = 0;
};

// As measured in thousandths of a second,
// e.g. a duration of 1 equals 0.001 seconds,
// and a duration of 1000 equals 1 second.
typedef long long presentation_t;  // NOLINT

struct Time {
  int hours;
  int minutes;
  int seconds;
  int milliseconds;

  bool operator==(const Time& rhs) const;
  bool operator<(const Time& rhs) const;
  bool operator>(const Time& rhs) const;
  bool operator<=(const Time& rhs) const;
  bool operator>=(const Time& rhs) const;

  presentation_t presentation() const;
  Time& presentation(presentation_t);

  Time& operator+=(presentation_t);
  Time operator+(presentation_t) const;

  Time& operator-=(presentation_t);
  presentation_t operator-(const Time&) const;
};

struct Setting {
  std::string name;
  std::string value;
};

struct Cue {
  std::string identifier;

  Time start_time;
  Time stop_time;

  typedef std::list<Setting> settings_t;
  settings_t settings;

  typedef std::list<std::string> payload_t;
  payload_t payload;
};

class Parser : private LineReader {
 public:
  explicit Parser(Reader* r);
  virtual ~Parser();

  // Pre-parse enough of the stream to determine whether
  // this is really a WEBVTT file. Returns 0 on success,
  // negative if error.
  int Init();

  // Parse the next WebVTT cue from the stream. Returns 0 if
  // an entire cue was parsed, negative if error, and positive
  // at end-of-stream.
  int Parse(Cue* cue);

 private:
  // Returns the next character in the stream, using the look-back character
  // if present (as per Reader::GetChar).
  virtual int GetChar(char* c);

  // Puts a character back into the stream (as per LineReader::UngetChar).
  virtual void UngetChar(char c);

  // Check for presence of a UTF-8 BOM in the stream.  Returns
  // negative if error, 0 on success, and positive at end-of-stream.
  int ParseBOM();

  // Parse the distinguished "cue timings" line, which includes the start
  // and stop times and settings.  Argument |line| contains the complete
  // line of text (as returned by ParseLine()), which the function is free
  // to modify as it sees fit, to facilitate scanning.  Argument |arrow_pos|
  // is the offset of the arrow token ("-->"), which indicates that this is
  // the timings line.  Returns negative if error, 0 on success.
  //
  static int ParseTimingsLine(std::string* line,
                              std::string::size_type arrow_pos,
                              Time* start_time, Time* stop_time,
                              Cue::settings_t* settings);

  // Parse a single time specifier (from the timings line), starting
  // at the given offset; lexical scanning stops when a NUL character
  // is detected. The function modifies offset |off| by the number of
  // characters consumed.  Returns negative if error, 0 on success.
  //
  static int ParseTime(const std::string& line, std::string::size_type* off,
                       Time* time);

  // Parse the cue settings from the timings line, starting at the
  // given offset.  Returns negative if error, 0 on success.
  //
  static int ParseSettings(const std::string& line, std::string::size_type off,
                           Cue::settings_t* settings);

  // Parse a non-negative integer from the characters in |line| beginning
  // at offset |off|.  The function increments |off| by the number
  // of characters consumed.  Returns the value, or negative if error.
  //
  static int ParseNumber(const std::string& line, std::string::size_type* off);

  Reader* const reader_;

  // Provides one character's worth of look-back, to facilitate scanning.
  int unget_;

  // Disable copy ctor and copy assign for Parser.
  Parser(const Parser&);
  Parser& operator=(const Parser&);
};

}  // namespace libwebvtt

#endif  // WEBVTT_WEBVTTPARSER_H_
