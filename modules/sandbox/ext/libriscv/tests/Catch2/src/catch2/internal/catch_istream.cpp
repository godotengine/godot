
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_istream.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_debug_console.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_stdstreams.hpp>

#include <cstdio>
#include <fstream>

namespace Catch {

    Catch::IStream::~IStream() = default;

namespace Detail {
    namespace {
        template<typename WriterF, std::size_t bufferSize=256>
        class StreamBufImpl final : public std::streambuf {
            char data[bufferSize];
            WriterF m_writer;

        public:
            StreamBufImpl() {
                setp( data, data + sizeof(data) );
            }

            ~StreamBufImpl() noexcept override {
                StreamBufImpl::sync();
            }

        private:
            int overflow( int c ) override {
                sync();

                if( c != EOF ) {
                    if( pbase() == epptr() )
                        m_writer( std::string( 1, static_cast<char>( c ) ) );
                    else
                        sputc( static_cast<char>( c ) );
                }
                return 0;
            }

            int sync() override {
                if( pbase() != pptr() ) {
                    m_writer( std::string( pbase(), static_cast<std::string::size_type>( pptr() - pbase() ) ) );
                    setp( pbase(), epptr() );
                }
                return 0;
            }
        };

        ///////////////////////////////////////////////////////////////////////////

        struct OutputDebugWriter {

            void operator()( std::string const& str ) {
                if ( !str.empty() ) {
                    writeToDebugConsole( str );
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////////

        class FileStream final : public IStream {
            std::ofstream m_ofs;
        public:
            FileStream( std::string const& filename ) {
                m_ofs.open( filename.c_str() );
                CATCH_ENFORCE( !m_ofs.fail(), "Unable to open file: '" << filename << '\'' );
                m_ofs << std::unitbuf;
            }
        public: // IStream
            std::ostream& stream() override {
                return m_ofs;
            }
        };

        ///////////////////////////////////////////////////////////////////////////

        class CoutStream final : public IStream {
            std::ostream m_os;
        public:
            // Store the streambuf from cout up-front because
            // cout may get redirected when running tests
            CoutStream() : m_os( Catch::cout().rdbuf() ) {}

        public: // IStream
            std::ostream& stream() override { return m_os; }
            bool isConsole() const override { return true; }
        };

        class CerrStream : public IStream {
            std::ostream m_os;

        public:
            // Store the streambuf from cerr up-front because
            // cout may get redirected when running tests
            CerrStream(): m_os( Catch::cerr().rdbuf() ) {}

        public: // IStream
            std::ostream& stream() override { return m_os; }
            bool isConsole() const override { return true; }
        };

        ///////////////////////////////////////////////////////////////////////////

        class DebugOutStream final : public IStream {
            Detail::unique_ptr<StreamBufImpl<OutputDebugWriter>> m_streamBuf;
            std::ostream m_os;
        public:
            DebugOutStream()
            :   m_streamBuf( Detail::make_unique<StreamBufImpl<OutputDebugWriter>>() ),
                m_os( m_streamBuf.get() )
            {}

        public: // IStream
            std::ostream& stream() override { return m_os; }
        };

    } // unnamed namespace
} // namespace Detail

    ///////////////////////////////////////////////////////////////////////////

    auto makeStream( std::string const& filename ) -> Detail::unique_ptr<IStream> {
        if ( filename.empty() || filename == "-" ) {
            return Detail::make_unique<Detail::CoutStream>();
        }
        if( filename[0] == '%' ) {
            if ( filename == "%debug" ) {
                return Detail::make_unique<Detail::DebugOutStream>();
            } else if ( filename == "%stderr" ) {
                return Detail::make_unique<Detail::CerrStream>();
            } else if ( filename == "%stdout" ) {
                return Detail::make_unique<Detail::CoutStream>();
            } else {
                CATCH_ERROR( "Unrecognised stream: '" << filename << '\'' );
            }
        }
        return Detail::make_unique<Detail::FileStream>( filename );
    }

}
