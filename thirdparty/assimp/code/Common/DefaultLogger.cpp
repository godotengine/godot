/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
---------------------------------------------------------------------------
*/

/** @file  DefaultLogger.cpp
 *  @brief Implementation of DefaultLogger (and Logger)
 */

// Default log streams
#include "Win32DebugLogStream.h"
#include "StdOStreamLogStream.h"
#include "FileLogStream.h"
#include <assimp/StringUtils.h>

#include <assimp/DefaultIOSystem.h>
#include <assimp/NullLogger.hpp>
#include <assimp/DefaultLogger.hpp>
#include <assimp/ai_assert.h>
#include <iostream>
#include <stdio.h>

#ifndef ASSIMP_BUILD_SINGLETHREADED
#   include <thread>
#   include <mutex>
    std::mutex loggerMutex;
#endif

namespace Assimp    {

// ----------------------------------------------------------------------------------
NullLogger DefaultLogger::s_pNullLogger;
Logger *DefaultLogger::m_pLogger = &DefaultLogger::s_pNullLogger;

static const unsigned int SeverityAll = Logger::Info | Logger::Err | Logger::Warn | Logger::Debugging;

// ----------------------------------------------------------------------------------
// Represents a log-stream + its error severity
struct LogStreamInfo {
    unsigned int  m_uiErrorSeverity;
    LogStream    *m_pStream;

    // Constructor
    LogStreamInfo( unsigned int uiErrorSev, LogStream *pStream ) :
        m_uiErrorSeverity( uiErrorSev ),
        m_pStream( pStream ) {
        // empty
    }

    // Destructor
    ~LogStreamInfo() {
        delete m_pStream;
    }
};

// ----------------------------------------------------------------------------------
// Construct a default log stream
LogStream* LogStream::createDefaultStream(aiDefaultLogStream    streams,
    const char* name /*= "AssimpLog.txt"*/,
    IOSystem* io            /*= NULL*/)
{
    switch (streams)
    {
        // This is a platform-specific feature
    case aiDefaultLogStream_DEBUGGER:
#ifdef WIN32
        return new Win32DebugLogStream();
#else
        return nullptr;
#endif

    // Platform-independent default streams
    case aiDefaultLogStream_STDERR:
        return new StdOStreamLogStream(std::cerr);
    case aiDefaultLogStream_STDOUT:
        return new StdOStreamLogStream(std::cout);
    case aiDefaultLogStream_FILE:
        return (name && *name ? new FileLogStream(name,io) : nullptr );
    default:
        // We don't know this default log stream, so raise an assertion
        ai_assert(false);

    };

    // For compilers without dead code path detection
    return nullptr;
}

// ----------------------------------------------------------------------------------
//  Creates the only singleton instance
Logger *DefaultLogger::create(const char* name /*= "AssimpLog.txt"*/,
    LogSeverity severity                       /*= NORMAL*/,
    unsigned int defStreams                    /*= aiDefaultLogStream_DEBUGGER | aiDefaultLogStream_FILE*/,
    IOSystem* io                               /*= NULL*/) {
    // enter the mutex here to avoid concurrency problems
#ifndef ASSIMP_BUILD_SINGLETHREADED
    std::lock_guard<std::mutex> lock(loggerMutex);
#endif

    if ( m_pLogger && !isNullLogger() ) {
        delete m_pLogger;
    }

    m_pLogger = new DefaultLogger( severity );

    // Attach default log streams
    // Stream the log to the MSVC debugger?
    if ( defStreams & aiDefaultLogStream_DEBUGGER ) {
        m_pLogger->attachStream( LogStream::createDefaultStream( aiDefaultLogStream_DEBUGGER ) );
    }

    // Stream the log to COUT?
    if ( defStreams & aiDefaultLogStream_STDOUT ) {
        m_pLogger->attachStream( LogStream::createDefaultStream( aiDefaultLogStream_STDOUT ) );
    }

    // Stream the log to CERR?
    if ( defStreams & aiDefaultLogStream_STDERR ) {
        m_pLogger->attachStream( LogStream::createDefaultStream( aiDefaultLogStream_STDERR ) );
    }

    // Stream the log to a file
    if ( defStreams & aiDefaultLogStream_FILE && name && *name ) {
        m_pLogger->attachStream( LogStream::createDefaultStream( aiDefaultLogStream_FILE, name, io ) );
    }

    return m_pLogger;
}

// ----------------------------------------------------------------------------------
void Logger::debug(const char* message) {

    // SECURITY FIX: otherwise it's easy to produce overruns since
    // sometimes importers will include data from the input file
    // (i.e. node names) in their messages.
    if (strlen(message)>MAX_LOG_MESSAGE_LENGTH) {
        return;
    }
    return OnDebug(message);
}

// ----------------------------------------------------------------------------------
void Logger::info(const char* message)  {

    // SECURITY FIX: see above
    if (strlen(message)>MAX_LOG_MESSAGE_LENGTH) {
        return;
    }
    return OnInfo(message);
}

// ----------------------------------------------------------------------------------
void Logger::warn(const char* message)  {

    // SECURITY FIX: see above
    if (strlen(message)>MAX_LOG_MESSAGE_LENGTH) {
        return;
    }
    return OnWarn(message);
}

// ----------------------------------------------------------------------------------
void Logger::error(const char* message) {
    // SECURITY FIX: see above
    if (strlen(message)>MAX_LOG_MESSAGE_LENGTH) {
        return;
    }
    return OnError(message);
}

// ----------------------------------------------------------------------------------
void DefaultLogger::set( Logger *logger ) {
    // enter the mutex here to avoid concurrency problems
#ifndef ASSIMP_BUILD_SINGLETHREADED
    std::lock_guard<std::mutex> lock(loggerMutex);
#endif

    if ( nullptr == logger ) {
        logger = &s_pNullLogger;
    }
    if ( nullptr != m_pLogger && !isNullLogger() ) {
        delete m_pLogger;
    }

    DefaultLogger::m_pLogger = logger;
}

// ----------------------------------------------------------------------------------
bool DefaultLogger::isNullLogger() {
    return m_pLogger == &s_pNullLogger;
}

// ----------------------------------------------------------------------------------
Logger *DefaultLogger::get() {
    return m_pLogger;
}

// ----------------------------------------------------------------------------------
//  Kills the only instance
void DefaultLogger::kill() {
    // enter the mutex here to avoid concurrency problems
#ifndef ASSIMP_BUILD_SINGLETHREADED
    std::lock_guard<std::mutex> lock(loggerMutex);
#endif

	if ( m_pLogger == &s_pNullLogger ) {
		return;
	}
    delete m_pLogger;
    m_pLogger = &s_pNullLogger;
}

// ----------------------------------------------------------------------------------
//  Debug message
void DefaultLogger::OnDebug( const char* message ) {
    if ( m_Severity == Logger::NORMAL ) {
        return;
    }

	static const size_t Size = MAX_LOG_MESSAGE_LENGTH + 16;
	char msg[Size];
	ai_snprintf(msg, Size, "Debug, T%u: %s", GetThreadID(), message);

    WriteToStreams( msg, Logger::Debugging );
}

// ----------------------------------------------------------------------------------
//  Logs an info
void DefaultLogger::OnInfo( const char* message ){
	static const size_t Size = MAX_LOG_MESSAGE_LENGTH + 16;
	char msg[Size];
    ai_snprintf(msg, Size, "Info,  T%u: %s", GetThreadID(), message );

    WriteToStreams( msg , Logger::Info );
}

// ----------------------------------------------------------------------------------
//  Logs a warning
void DefaultLogger::OnWarn( const char* message ) {
	static const size_t Size = MAX_LOG_MESSAGE_LENGTH + 16;
	char msg[Size];
	ai_snprintf(msg, Size, "Warn,  T%u: %s", GetThreadID(), message );

    WriteToStreams( msg, Logger::Warn );
}

// ----------------------------------------------------------------------------------
//  Logs an error
void DefaultLogger::OnError( const char* message ) {
	static const size_t Size = MAX_LOG_MESSAGE_LENGTH + 16;
	char msg[ Size ];
    ai_snprintf(msg, Size, "Error, T%u: %s", GetThreadID(), message );

    WriteToStreams( msg, Logger::Err );
}

// ----------------------------------------------------------------------------------
//  Will attach a new stream
bool DefaultLogger::attachStream( LogStream *pStream, unsigned int severity ) {
    if ( nullptr == pStream ) {
        return false;
    }

    if (0 == severity)  {
        severity = Logger::Info | Logger::Err | Logger::Warn | Logger::Debugging;
    }

    for ( StreamIt it = m_StreamArray.begin();
        it != m_StreamArray.end();
        ++it )
    {
        if ( (*it)->m_pStream == pStream ) {
            (*it)->m_uiErrorSeverity |= severity;
            return true;
        }
    }

    LogStreamInfo *pInfo = new LogStreamInfo( severity, pStream );
    m_StreamArray.push_back( pInfo );
    return true;
}

// ----------------------------------------------------------------------------------
//  Detach a stream
bool DefaultLogger::detatchStream( LogStream *pStream, unsigned int severity ) {
    if ( nullptr == pStream ) {
        return false;
    }

    if (0 == severity)  {
        severity = SeverityAll;
    }

    bool res( false );
    for ( StreamIt it = m_StreamArray.begin(); it != m_StreamArray.end(); ++it ) {
        if ( (*it)->m_pStream == pStream ) {
            (*it)->m_uiErrorSeverity &= ~severity;
            if ( (*it)->m_uiErrorSeverity == 0 ) {
                // don't delete the underlying stream 'cause the caller gains ownership again
                (**it).m_pStream = nullptr;
                delete *it;
                m_StreamArray.erase( it );
                res = true;
                break;
            }
            return true;
        }
    }
    return res;
}

// ----------------------------------------------------------------------------------
//  Constructor
DefaultLogger::DefaultLogger(LogSeverity severity)
    :   Logger  ( severity )
    ,   noRepeatMsg (false)
    ,   lastLen( 0 ) {
    lastMsg[0] = '\0';
}

// ----------------------------------------------------------------------------------
//  Destructor
DefaultLogger::~DefaultLogger() {
    for ( StreamIt it = m_StreamArray.begin(); it != m_StreamArray.end(); ++it ) {
        // also frees the underlying stream, we are its owner.
        delete *it;
    }
}

// ----------------------------------------------------------------------------------
//  Writes message to stream
void DefaultLogger::WriteToStreams(const char *message, ErrorSeverity ErrorSev ) {
    ai_assert(nullptr != message);

    // Check whether this is a repeated message
    if (! ::strncmp( message,lastMsg, lastLen-1))
    {
        if (!noRepeatMsg)
        {
            noRepeatMsg = true;
            message = "Skipping one or more lines with the same contents\n";
        }
        else return;
    }
    else
    {
        // append a new-line character to the message to be printed
        lastLen = ::strlen(message);
        ::memcpy(lastMsg,message,lastLen+1);
        ::strcat(lastMsg+lastLen,"\n");

        message = lastMsg;
        noRepeatMsg = false;
        ++lastLen;
    }
    for ( ConstStreamIt it = m_StreamArray.begin();
        it != m_StreamArray.end();
        ++it)
    {
        if ( ErrorSev & (*it)->m_uiErrorSeverity )
            (*it)->m_pStream->write( message);
    }
}

// ----------------------------------------------------------------------------------
//  Returns thread id, if not supported only a zero will be returned.
unsigned int DefaultLogger::GetThreadID()
{
    // fixme: we can get this value via std::threads
    // std::this_thread::get_id().hash() returns a (big) size_t, not sure if this is useful in this case.
#ifdef WIN32
    return (unsigned int)::GetCurrentThreadId();
#else
    return 0; // not supported
#endif
}

// ----------------------------------------------------------------------------------

} // !namespace Assimp
