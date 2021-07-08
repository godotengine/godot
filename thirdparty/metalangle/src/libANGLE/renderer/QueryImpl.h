//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// QueryImpl.h: Defines the abstract rx::QueryImpl class.

#ifndef LIBANGLE_RENDERER_QUERYIMPL_H_
#define LIBANGLE_RENDERER_QUERYIMPL_H_

#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"

namespace gl
{
class Context;
}  // namespace gl

namespace rx
{
class QueryImpl : angle::NonCopyable
{
  public:
    explicit QueryImpl(gl::QueryType type) : mType(type) {}
    virtual ~QueryImpl() {}

    virtual void onDestroy(const gl::Context *context);

    virtual angle::Result begin(const gl::Context *context)                              = 0;
    virtual angle::Result end(const gl::Context *context)                                = 0;
    virtual angle::Result queryCounter(const gl::Context *context)                       = 0;
    virtual angle::Result getResult(const gl::Context *context, GLint *params)           = 0;
    virtual angle::Result getResult(const gl::Context *context, GLuint *params)          = 0;
    virtual angle::Result getResult(const gl::Context *context, GLint64 *params)         = 0;
    virtual angle::Result getResult(const gl::Context *context, GLuint64 *params)        = 0;
    virtual angle::Result isResultAvailable(const gl::Context *context, bool *available) = 0;

    gl::QueryType getType() const { return mType; }

  private:
    gl::QueryType mType;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_QUERYIMPL_H_
