/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team



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

/** @file metadata.h
 *  @brief Defines the data structures for holding node meta information.
 */
#pragma once
#ifndef AI_METADATA_H_INC
#define AI_METADATA_H_INC

#if defined(_MSC_VER) && (_MSC_VER <= 1500)
#  include "Compiler/pstdint.h"
#else
#  include <stdint.h>
#endif

// -------------------------------------------------------------------------------
/**
  * Enum used to distinguish data types
  */
 // -------------------------------------------------------------------------------
typedef enum aiMetadataType {
    AI_BOOL       = 0,
    AI_INT32      = 1,
    AI_UINT64     = 2,
    AI_FLOAT      = 3,
    AI_DOUBLE     = 4,
    AI_AISTRING   = 5,
    AI_AIVECTOR3D = 6,
    AI_META_MAX   = 7,

#ifndef SWIG
    FORCE_32BIT = INT_MAX
#endif
} aiMetadataType;

// -------------------------------------------------------------------------------
/**
  * Metadata entry
  *
  * The type field uniquely identifies the underlying type of the data field
  */
 // -------------------------------------------------------------------------------
struct aiMetadataEntry {
    aiMetadataType mType;
    void* mData;
};

#ifdef __cplusplus

#include <string>

// -------------------------------------------------------------------------------
/**
  * Helper functions to get the aiType enum entry for a type
  */
 // -------------------------------------------------------------------------------

inline aiMetadataType GetAiType( bool )       { return AI_BOOL; }
inline aiMetadataType GetAiType( int32_t )    { return AI_INT32; }
inline aiMetadataType GetAiType( uint64_t )   { return AI_UINT64; }
inline aiMetadataType GetAiType( float )      { return AI_FLOAT; }
inline aiMetadataType GetAiType( double )     { return AI_DOUBLE; }
inline aiMetadataType GetAiType( const aiString & )   { return AI_AISTRING; }
inline aiMetadataType GetAiType( const aiVector3D & ) { return AI_AIVECTOR3D; }

#endif // __cplusplus

// -------------------------------------------------------------------------------
/**
  * Container for holding metadata.
  *
  * Metadata is a key-value store using string keys and values.
  */
 // -------------------------------------------------------------------------------
struct aiMetadata {
    /** Length of the mKeys and mValues arrays, respectively */
    unsigned int mNumProperties;

    /** Arrays of keys, may not be NULL. Entries in this array may not be NULL as well. */
    C_STRUCT aiString* mKeys;

    /** Arrays of values, may not be NULL. Entries in this array may be NULL if the
      * corresponding property key has no assigned value. */
    C_STRUCT aiMetadataEntry* mValues;

#ifdef __cplusplus

    /** 
     *  @brief  The default constructor, set all members to zero by default.
     */
    aiMetadata() AI_NO_EXCEPT
    : mNumProperties(0)
    , mKeys(nullptr)
    , mValues(nullptr) {
        // empty
    }

    aiMetadata( const aiMetadata &rhs )
    : mNumProperties( rhs.mNumProperties )
    , mKeys( nullptr )
    , mValues( nullptr ) {
        mKeys = new aiString[ mNumProperties ];
        for ( size_t i = 0; i < static_cast<size_t>( mNumProperties ); ++i ) {
            mKeys[ i ] = rhs.mKeys[ i ];
        }
        mValues = new aiMetadataEntry[ mNumProperties ];
        for ( size_t i = 0; i < static_cast<size_t>(mNumProperties); ++i ) {
            mValues[ i ].mType = rhs.mValues[ i ].mType;
            switch ( rhs.mValues[ i ].mType ) {
            case AI_BOOL:
                mValues[ i ].mData = new bool;
                ::memcpy( mValues[ i ].mData, rhs.mValues[ i ].mData, sizeof(bool) );
                break;
            case AI_INT32: {
                int32_t v;
                ::memcpy( &v, rhs.mValues[ i ].mData, sizeof( int32_t ) );
                mValues[ i ].mData = new int32_t( v );
                }
                break;
            case AI_UINT64: {
                    uint64_t v;
                    ::memcpy( &v, rhs.mValues[ i ].mData, sizeof( uint64_t ) );
                    mValues[ i ].mData = new  uint64_t( v );
                }
                break;
            case AI_FLOAT: {
                    float v;
                    ::memcpy( &v, rhs.mValues[ i ].mData, sizeof( float ) );
                    mValues[ i ].mData = new float( v );
                }
                break;
            case AI_DOUBLE: {
                    double v;
                    ::memcpy( &v, rhs.mValues[ i ].mData, sizeof( double ) );
                    mValues[ i ].mData = new double( v );
                }
                break;
            case AI_AISTRING: {
                    aiString v;
                    rhs.Get<aiString>( mKeys[ i ], v );
                    mValues[ i ].mData = new aiString( v );
                }
                break;
            case AI_AIVECTOR3D: {
                    aiVector3D v;
                    rhs.Get<aiVector3D>( mKeys[ i ], v );
                    mValues[ i ].mData = new aiVector3D( v );
                }
                break;
#ifndef SWIG
            case FORCE_32BIT:
#endif
            default:
                break;
            }

        }
    }

    /** 
     *  @brief The destructor.
     */
    ~aiMetadata() {
        delete [] mKeys;
        mKeys = nullptr;
        if (mValues) {
            // Delete each metadata entry
            for (unsigned i=0; i<mNumProperties; ++i) {
                void* data = mValues[i].mData;
                switch (mValues[i].mType) {
                case AI_BOOL:
                    delete static_cast< bool* >( data );
                    break;
                case AI_INT32:
                    delete static_cast< int32_t* >( data );
                    break;
                case AI_UINT64:
                    delete static_cast< uint64_t* >( data );
                    break;
                case AI_FLOAT:
                    delete static_cast< float* >( data );
                    break;
                case AI_DOUBLE:
                    delete static_cast< double* >( data );
                    break;
                case AI_AISTRING:
                    delete static_cast< aiString* >( data );
                    break;
                case AI_AIVECTOR3D:
                    delete static_cast< aiVector3D* >( data );
                    break;
#ifndef SWIG
                case FORCE_32BIT:
#endif
                default:
                    break;
                }
            }

            // Delete the metadata array
            delete [] mValues;
            mValues = nullptr;
        }
    }

    /**
     *  @brief Allocates property fields + keys.
     *  @param  numProperties   Number of requested properties.
     */
    static inline
    aiMetadata *Alloc( unsigned int numProperties ) {
        if ( 0 == numProperties ) {
            return nullptr;
        }

        aiMetadata *data = new aiMetadata;
        data->mNumProperties = numProperties;
        data->mKeys = new aiString[ data->mNumProperties ]();
        data->mValues = new aiMetadataEntry[ data->mNumProperties ]();

        return data;
    }

    /**
     *  @brief Deallocates property fields + keys.
     */
    static inline
    void Dealloc( aiMetadata *metadata ) {
        delete metadata;
    }

	template<typename T>
	inline
    void Add(const std::string& key, const T& value) {
		aiString* new_keys = new aiString[mNumProperties + 1];
		aiMetadataEntry* new_values = new aiMetadataEntry[mNumProperties + 1];

		for(unsigned int i = 0; i < mNumProperties; ++i)
		{
			new_keys[i] = mKeys[i];
			new_values[i] = mValues[i];
		}

		delete mKeys;
		delete mValues;

		mKeys = new_keys;
		mValues = new_values;

		mNumProperties++;

		Set(mNumProperties - 1, key, value);
	}

    template<typename T>
    inline 
    bool Set( unsigned index, const std::string& key, const T& value ) {
        // In range assertion
        if ( index >= mNumProperties ) {
            return false;
        }

        // Ensure that we have a valid key.
        if ( key.empty() ) {
            return false;
        }

        // Set metadata key
        mKeys[index] = key;

        // Set metadata type
        mValues[index].mType = GetAiType(value);
        // Copy the given value to the dynamic storage
        mValues[index].mData = new T(value);

        return true;
    }

    template<typename T>
    inline 
    bool Get( unsigned index, T& value ) const {
        // In range assertion
        if ( index >= mNumProperties ) {
            return false;
        }

        // Return false if the output data type does
        // not match the found value's data type
        if ( GetAiType( value ) != mValues[ index ].mType ) {
            return false;
        }

        // Otherwise, output the found value and
        // return true
        value = *static_cast<T*>(mValues[index].mData);

        return true;
    }

    template<typename T>
    inline 
    bool Get( const aiString& key, T& value ) const {
        // Search for the given key
        for ( unsigned int i = 0; i < mNumProperties; ++i ) {
            if ( mKeys[ i ] == key ) {
                return Get( i, value );
            }
        }
        return false;
    }

    template<typename T>
    inline
    bool Get( const std::string& key, T& value ) const {
        return Get(aiString(key), value);
    }

	/// Return metadata entry for analyzing it by user.
	/// \param [in] pIndex - index of the entry.
	/// \param [out] pKey - pointer to the key value.
	/// \param [out] pEntry - pointer to the entry: type and value.
	/// \return false - if pIndex is out of range, else - true.
	inline
    bool Get(size_t index, const aiString*& key, const aiMetadataEntry*& entry) const {
        if ( index >= mNumProperties ) {
            return false;
        }

		key = &mKeys[index];
		entry = &mValues[index];

		return true;
	}

#endif // __cplusplus

};

#endif // AI_METADATA_H_INC
