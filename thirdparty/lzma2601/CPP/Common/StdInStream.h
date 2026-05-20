// Common/StdInStream.h

#ifndef ZIP7_INC_COMMON_STD_IN_STREAM_H
#define ZIP7_INC_COMMON_STD_IN_STREAM_H

#include <stdio.h>

#include "MyString.h"
#include "MyTypes.h"

class CStdInStream
{
  FILE *_stream;
  // bool _streamIsOpen;
public:
  int CodePage;

  CStdInStream(FILE *stream = NULL):
      _stream(stream),
      // _streamIsOpen(false),
      CodePage(-1)
      {}

  /*
  ~CStdInStream() { Close(); }

  bool Open(LPCTSTR fileName) throw();
  bool Close() throw();
  */

  // returns:
  //   false, if ZERO character in stream
  //   true, if EOF or '\n'
  bool ScanAStringUntilNewLine(AString &s);
  bool ScanUStringUntilNewLine(UString &s);
  // bool ReadToString(AString &resultString);

  bool Eof() const throw() { return (feof(_stream) != 0); }
  bool Error() const throw() { return (ferror(_stream) != 0); }

  int GetChar();
};

extern CStdInStream g_StdIn;

#endif
