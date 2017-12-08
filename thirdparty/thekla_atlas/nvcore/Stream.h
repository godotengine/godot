// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_CORE_STREAM_H
#define NV_CORE_STREAM_H

#include "nvcore.h"
#include "Debug.h"

namespace nv
{

    /// Base stream class.
    class NVCORE_CLASS Stream {
    public:

        enum ByteOrder {
            LittleEndian = false,
            BigEndian = true,
        };

        /// Get the byte order of the system.
        static ByteOrder getSystemByteOrder() { 
#if NV_LITTLE_ENDIAN
            return LittleEndian;
#else
            return BigEndian;
#endif
        }


        /// Ctor.
        Stream() : m_byteOrder(LittleEndian) { }

        /// Virtual destructor.
        virtual ~Stream() {}

        /// Set byte order.
        void setByteOrder(ByteOrder bo) { m_byteOrder = bo; }

        /// Get byte order.
        ByteOrder byteOrder() const { return m_byteOrder; }


        /// Serialize the given data.
        virtual uint serialize( void * data, uint len ) = 0;

        /// Move to the given position in the archive.
        virtual void seek( uint pos ) = 0;

        /// Return the current position in the archive.
        virtual uint tell() const = 0;

        /// Return the current size of the archive.
        virtual uint size() const = 0;

        /// Determine if there has been any error.
        virtual bool isError() const = 0;

        /// Clear errors.
        virtual void clearError() = 0;

        /// Return true if the stream is at the end.
        virtual bool isAtEnd() const = 0;

        /// Return true if the stream is seekable.
        virtual bool isSeekable() const = 0;

        /// Return true if this is an input stream.
        virtual bool isLoading() const = 0;

        /// Return true if this is an output stream.
        virtual bool isSaving() const = 0;


        void advance(uint offset) { seek(tell() + offset); }


        // friends
        friend Stream & operator<<( Stream & s, bool & c ) {
#if NV_OS_DARWIN && !NV_CC_CPP11
            nvStaticCheck(sizeof(bool) == 4);
            uint8 b = c ? 1 : 0;
            s.serialize( &b, 1 );
            c = (b != 0);
#else
            nvStaticCheck(sizeof(bool) == 1);
            s.serialize( &c, 1 );
#endif
            return s;
        }
        friend Stream & operator<<( Stream & s, char & c ) {
            nvStaticCheck(sizeof(char) == 1);
            s.serialize( &c, 1 );
            return s;
        }
        friend Stream & operator<<( Stream & s, uint8 & c ) {
            nvStaticCheck(sizeof(uint8) == 1);
            s.serialize( &c, 1 );
            return s;
        }
        friend Stream & operator<<( Stream & s, int8 & c ) {
            nvStaticCheck(sizeof(int8) == 1);
            s.serialize( &c, 1 );
            return s;
        }
        friend Stream & operator<<( Stream & s, uint16 & c ) {
            nvStaticCheck(sizeof(uint16) == 2);
            return s.byteOrderSerialize( &c, 2 );
        }
        friend Stream & operator<<( Stream & s, int16 & c ) {
            nvStaticCheck(sizeof(int16) == 2);
            return s.byteOrderSerialize( &c, 2 );
        }
        friend Stream & operator<<( Stream & s, uint32 & c ) {
            nvStaticCheck(sizeof(uint32) == 4);
            return s.byteOrderSerialize( &c, 4 );
        }
        friend Stream & operator<<( Stream & s, int32 & c ) {
            nvStaticCheck(sizeof(int32) == 4);
            return s.byteOrderSerialize( &c, 4 );
        }
        friend Stream & operator<<( Stream & s, uint64 & c ) {
            nvStaticCheck(sizeof(uint64) == 8);
            return s.byteOrderSerialize( &c, 8 );
        }
        friend Stream & operator<<( Stream & s, int64 & c ) {
            nvStaticCheck(sizeof(int64) == 8);
            return s.byteOrderSerialize( &c, 8 );
        }
        friend Stream & operator<<( Stream & s, float & c ) {
            nvStaticCheck(sizeof(float) == 4);
            return s.byteOrderSerialize( &c, 4 );
        }
        friend Stream & operator<<( Stream & s, double & c ) {
            nvStaticCheck(sizeof(double) == 8);
            return s.byteOrderSerialize( &c, 8 );
        }

    protected:

        /// Serialize in the stream byte order.
        Stream & byteOrderSerialize( void * v, uint len ) {
            if( m_byteOrder == getSystemByteOrder() ) {
                serialize( v, len );
            }
            else {
                for( uint i = len; i > 0; i-- ) {
                    serialize( (uint8 *)v + i - 1, 1 );
                }
            }
            return *this;
        }


    private:

        ByteOrder m_byteOrder;

    };

} // nv namespace

#endif // NV_CORE_STREAM_H
