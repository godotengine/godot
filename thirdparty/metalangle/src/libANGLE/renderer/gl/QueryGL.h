//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// QueryGL.h: Defines the class interface for QueryGL.

#ifndef LIBANGLE_RENDERER_GL_QUERYGL_H_
#define LIBANGLE_RENDERER_GL_QUERYGL_H_

#include <deque>

#include "libANGLE/renderer/QueryImpl.h"

namespace rx
{

class FunctionsGL;
class StateManagerGL;

class QueryGL : public QueryImpl
{
  public:
    QueryGL(gl::QueryType type);
    ~QueryGL() override;

    // OpenGL is only allowed to have one query of each type active at any given time. Since ANGLE
    // virtualizes contexts, queries need to be able to be paused and resumed.
    // A query is "paused" by ending it and pushing the ID into a list of queries awaiting readback.
    // When it is "resumed", a new query is generated and started.
    // When a result is required, the queries are "flushed" by iterating over the list of pending
    // queries and merging their results.
    virtual angle::Result pause(const gl::Context *context)  = 0;
    virtual angle::Result resume(const gl::Context *context) = 0;
};

class StandardQueryGL : public QueryGL
{
  public:
    StandardQueryGL(gl::QueryType type, const FunctionsGL *functions, StateManagerGL *stateManager);
    ~StandardQueryGL() override;

    angle::Result begin(const gl::Context *context) override;
    angle::Result end(const gl::Context *context) override;
    angle::Result queryCounter(const gl::Context *context) override;
    angle::Result getResult(const gl::Context *context, GLint *params) override;
    angle::Result getResult(const gl::Context *context, GLuint *params) override;
    angle::Result getResult(const gl::Context *context, GLint64 *params) override;
    angle::Result getResult(const gl::Context *context, GLuint64 *params) override;
    angle::Result isResultAvailable(const gl::Context *context, bool *available) override;

    angle::Result pause(const gl::Context *context) override;
    angle::Result resume(const gl::Context *context) override;

  private:
    angle::Result flush(const gl::Context *context, bool force);

    template <typename T>
    angle::Result getResultBase(const gl::Context *context, T *params);

    gl::QueryType mType;

    const FunctionsGL *mFunctions;
    StateManagerGL *mStateManager;

    GLuint mActiveQuery;
    std::deque<GLuint> mPendingQueries;
    GLuint64 mResultSum;
};

class SyncProviderGL;
class SyncQueryGL : public QueryGL
{
  public:
    SyncQueryGL(gl::QueryType type, const FunctionsGL *functions);
    ~SyncQueryGL() override;

    static bool IsSupported(const FunctionsGL *functions);

    angle::Result begin(const gl::Context *context) override;
    angle::Result end(const gl::Context *context) override;
    angle::Result queryCounter(const gl::Context *context) override;
    angle::Result getResult(const gl::Context *context, GLint *params) override;
    angle::Result getResult(const gl::Context *context, GLuint *params) override;
    angle::Result getResult(const gl::Context *context, GLint64 *params) override;
    angle::Result getResult(const gl::Context *context, GLuint64 *params) override;
    angle::Result isResultAvailable(const gl::Context *context, bool *available) override;

    angle::Result pause(const gl::Context *context) override;
    angle::Result resume(const gl::Context *context) override;

  private:
    angle::Result flush(const gl::Context *context, bool force);

    template <typename T>
    angle::Result getResultBase(const gl::Context *context, T *params);

    const FunctionsGL *mFunctions;

    std::unique_ptr<SyncProviderGL> mSyncProvider;
    bool mFinished;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_QUERYGL_H_
