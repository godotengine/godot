//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_EXCEPTION_H
#define MATERIALX_EXCEPTION_H

#include <MaterialXCore/Export.h>

#include <exception>

/// @file
/// Base exception classes

MATERIALX_NAMESPACE_BEGIN

/// @class Exception
/// The base class for exceptions that are propagated from the MaterialX library
/// to the client application.
class MX_CORE_API Exception : public std::exception
{
  public:
    explicit Exception(const string& msg) :
        _msg(msg)
    {
    }

    Exception(const Exception& e) :
        _msg(e._msg)
    {
    }

    Exception& operator=(const Exception& e)
    {
        _msg = e._msg;
        return *this;
    }

    virtual ~Exception() noexcept
    {
    }

    const char* what() const noexcept override
    {
        return _msg.c_str();
    }

  private:
    string _msg;
};

MATERIALX_NAMESPACE_END

#endif
