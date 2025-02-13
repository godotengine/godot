// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../sys/platform.h"
#include "../sys/ref.h"
#include "../sys/filename.h"
#include "../sys/estring.h"

#include <vector>
#include <iostream>
#include <cstdio>
#include <string.h>

namespace embree
{
  /*! stores the location of a stream element in the source */
  class ParseLocation
  {
  public:
    ParseLocation () : lineNumber(-1), colNumber(-1) {}
    ParseLocation (std::shared_ptr<std::string> fileName, ssize_t lineNumber, ssize_t colNumber, ssize_t /*charNumber*/)
      : fileName(fileName), lineNumber(lineNumber), colNumber(colNumber) {}

    std::string str() const
    {
      std::string str = "unknown";
      if (fileName) str = *fileName;
      if (lineNumber >= 0) str += " line " + toString(lineNumber);
      if (lineNumber >= 0 && colNumber >= 0) str += " character " + toString(colNumber);
      return str;
    }

  private:
    std::shared_ptr<std::string> fileName;         /// name of the file (or stream) the token is from
    ssize_t lineNumber;           /// the line number the token is from
    ssize_t colNumber;            /// the character number in the current line
  };

  /*! a stream class templated over the stream elements */
  template<typename T> class Stream : public RefCount
  {
    enum { BUF_SIZE = 1024 };
    
  private:
    virtual T next() = 0;
    virtual ParseLocation location() = 0;
    __forceinline std::pair<T,ParseLocation> nextHelper() {
      ParseLocation l = location();
      T v = next();
      return std::pair<T,ParseLocation>(v,l);
    }
    __forceinline void push_back(const std::pair<T,ParseLocation>& v) {
      if (past+future == BUF_SIZE) pop_front();
      size_t end = (start+past+future++)%BUF_SIZE;
      buffer[end] = v;
    }
    __forceinline void pop_front() {
      if (past == 0) THROW_RUNTIME_ERROR("stream buffer empty");
      start = (start+1)%BUF_SIZE; past--;
    }
  public:
    Stream () : start(0), past(0), future(0), buffer(BUF_SIZE) {}
    virtual ~Stream() {}
    
  public:
    
    const ParseLocation& loc() {
      if (future == 0) push_back(nextHelper());
      return buffer[(start+past)%BUF_SIZE].second;
    }
    T get() {
      if (future == 0) push_back(nextHelper());
      T t = buffer[(start+past)%BUF_SIZE].first;
      past++; future--;
      return t;
    }
    const T& peek() {
      if (future == 0) push_back(nextHelper());
      return buffer[(start+past)%BUF_SIZE].first;
    }
    const T& unget(size_t n = 1) {
      if (past < n) THROW_RUNTIME_ERROR ("cannot unget that many items");
      past -= n; future += n;
      return peek();
    }
    void drop() {
      if (future == 0) push_back(nextHelper());
      past++; future--;
    }
  private:
    size_t start,past,future;
    std::vector<std::pair<T,ParseLocation> > buffer;
  };
  
  /*! warps an iostream stream */
  class StdStream : public Stream<int>
  {
  public:
    StdStream (std::istream& cin, const std::string& name = "std::stream")
      : cin(cin), lineNumber(1), colNumber(0), charNumber(0), name(std::shared_ptr<std::string>(new std::string(name))) {}
    ~StdStream() {}
    ParseLocation location() {
      return ParseLocation(name,lineNumber,colNumber,charNumber);
    }
    int next() {
      int c = cin.get();
      if (c == '\n') { lineNumber++; colNumber = 0; } else if (c != '\r') colNumber++;
      charNumber++;
      return c;
    }
  private:
    std::istream& cin;
    ssize_t lineNumber;           /// the line number the token is from
    ssize_t colNumber;            /// the character number in the current line
    ssize_t charNumber;           /// the character in the file
    std::shared_ptr<std::string> name;             /// name of buffer
  };

  /*! creates a stream from a file */
  class FileStream : public Stream<int>
  {
  public:
    FileStream (const FileName& fileName)
      : lineNumber(1), colNumber(0), charNumber(0), name(std::shared_ptr<std::string>(new std::string(fileName.str())))
    {
      if (ifs) ifs.close();
      ifs.open(fileName.str());
      if (!ifs.is_open()) THROW_RUNTIME_ERROR("cannot open file " + fileName.str());
    }
    ~FileStream() { 
      if (ifs) ifs.close();
    }

  public:
    ParseLocation location() {
      return ParseLocation(name,lineNumber,colNumber,charNumber);
    }

    int next() {
      int c = ifs.get();
      if (c == '\n') { lineNumber++; colNumber = 0; } else if (c != '\r') colNumber++;
      charNumber++;
      return c;
    }


  private:
    std::ifstream ifs;
    ssize_t lineNumber;           /// the line number the token is from
    ssize_t colNumber;            /// the character number in the current line
    ssize_t charNumber;           /// the character in the file
    std::shared_ptr<std::string> name;             /// name of buffer
  };

  /*! creates a stream from a string */
  class StrStream : public Stream<int>
  {
  public:

    StrStream (const char* str)
      : str(str), lineNumber(1), colNumber(0), charNumber(0) {}

  public:
    ParseLocation location() {
      return ParseLocation(std::shared_ptr<std::string>(),lineNumber,colNumber,charNumber);
    }

    int next() {
      int c = str[charNumber];
      if (c == 0) return EOF;
      if (c == '\n') { lineNumber++; colNumber = 0; } else if (c != '\r') colNumber++;
      charNumber++;
      return c;
    }

  private:
    const char* str;
    ssize_t lineNumber;           /// the line number the token is from
    ssize_t colNumber;            /// the character number in the current line
    ssize_t charNumber;           /// the character in the file
  };

  /*! creates a character stream from a command line */
  class CommandLineStream : public Stream<int>
  {
  public:
    CommandLineStream (int argc, char** argv, const std::string& name = "command line")
      : i(0), j(0), charNumber(0), name(std::shared_ptr<std::string>(new std::string(name)))
    {
      if (argc > 0) {
	for (size_t i=0; argv[0][i] && i<1024; i++) charNumber++;
	charNumber++;
      }
      for (ssize_t k=1; k<argc; k++) args.push_back(argv[k]);
    }
    ~CommandLineStream() {}
  public:
    ParseLocation location() {
      return ParseLocation(name,0,charNumber,charNumber);
    }
    int next() {
      if (i == args.size()) return EOF;
      if (j == args[i].size()) { i++; j=0; charNumber++; return ' '; }
      charNumber++;
      return args[i][j++];
    }
  private:
    size_t i,j;
    std::vector<std::string> args;
    ssize_t charNumber;           /// the character in the file
    std::shared_ptr<std::string> name;             /// name of buffer
  };
}
