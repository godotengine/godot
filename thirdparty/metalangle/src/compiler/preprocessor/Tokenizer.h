//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_PREPROCESSOR_TOKENIZER_H_
#define COMPILER_PREPROCESSOR_TOKENIZER_H_

#include "common/angleutils.h"
#include "compiler/preprocessor/Input.h"
#include "compiler/preprocessor/Lexer.h"

namespace angle
{

namespace pp
{

class Diagnostics;

class Tokenizer : public Lexer
{
  public:
    struct Context
    {
        Diagnostics *diagnostics;

        Input input;
        // The location where yytext points to. Token location should track
        // scanLoc instead of Input::mReadLoc because they may not be the same
        // if text is buffered up in the scanner input buffer.
        Input::Location scanLoc;

        bool leadingSpace;
        bool lineStart;
    };

    Tokenizer(Diagnostics *diagnostics);
    ~Tokenizer() override;

    bool init(size_t count, const char *const string[], const int length[]);

    void setFileNumber(int file);
    void setLineNumber(int line);
    void setMaxTokenSize(size_t maxTokenSize);

    void lex(Token *token) override;

  private:
    bool initScanner();
    void destroyScanner();

    void *mHandle;         // Scanner handle.
    Context mContext;      // Scanner extra.
    size_t mMaxTokenSize;  // Maximum token size
};

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_TOKENIZER_H_
