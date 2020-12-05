/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

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

----------------------------------------------------------------------
*/

/** @file  MaterialSystem.cpp
 *  @brief Implementation of the material system of the library
 */

#include <assimp/Hash.h>
#include <assimp/fast_atof.h>
#include <assimp/ParsingUtils.h>
#include "MaterialSystem.h"
#include <assimp/types.h>
#include <assimp/material.h>
#include <assimp/DefaultLogger.hpp>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// Get a specific property from a material
aiReturn aiGetMaterialProperty(const aiMaterial* pMat,
    const char* pKey,
    unsigned int type,
    unsigned int index,
    const aiMaterialProperty** pPropOut)
{
    ai_assert( pMat != NULL );
    ai_assert( pKey != NULL );
    ai_assert( pPropOut != NULL );

    /*  Just search for a property with exactly this name ..
     *  could be improved by hashing, but it's possibly
     *  no worth the effort (we're bound to C structures,
     *  thus std::map or derivates are not applicable. */
    for ( unsigned int i = 0; i < pMat->mNumProperties; ++i ) {
        aiMaterialProperty* prop = pMat->mProperties[i];

        if (prop /* just for safety ... */
            && 0 == strcmp( prop->mKey.data, pKey )
            && (UINT_MAX == type  || prop->mSemantic == type) /* UINT_MAX is a wild-card, but this is undocumented :-) */
            && (UINT_MAX == index || prop->mIndex == index))
        {
            *pPropOut = pMat->mProperties[i];
            return AI_SUCCESS;
        }
    }
    *pPropOut = NULL;
    return AI_FAILURE;
}

// ------------------------------------------------------------------------------------------------
// Get an array of floating-point values from the material.
aiReturn aiGetMaterialFloatArray(const aiMaterial* pMat,
    const char* pKey,
    unsigned int type,
    unsigned int index,
    ai_real* pOut,
    unsigned int* pMax)
{
    ai_assert( pOut != nullptr );
    ai_assert( pMat != nullptr );

    const aiMaterialProperty* prop;
    aiGetMaterialProperty(pMat,pKey,type,index, (const aiMaterialProperty**) &prop);
    if ( nullptr == prop) {
        return AI_FAILURE;
    }

    // data is given in floats, convert to ai_real
    unsigned int iWrite = 0;
    if( aiPTI_Float == prop->mType || aiPTI_Buffer == prop->mType)  {
        iWrite = prop->mDataLength / sizeof(float);
        if (pMax) {
            iWrite = std::min(*pMax,iWrite); ;
        }

        for (unsigned int a = 0; a < iWrite; ++a) {
            pOut[ a ] = static_cast<ai_real> ( reinterpret_cast<float*>(prop->mData)[a] );
        }

        if (pMax) {
            *pMax = iWrite;
        }
    }
    // data is given in doubles, convert to float
    else if( aiPTI_Double == prop->mType)  {
        iWrite = prop->mDataLength / sizeof(double);
        if (pMax) {
            iWrite = std::min(*pMax,iWrite); ;
        }
        for (unsigned int a = 0; a < iWrite;++a)    {
            pOut[a] = static_cast<ai_real> ( reinterpret_cast<double*>(prop->mData)[a] );
        }
        if (pMax) {
            *pMax = iWrite;
        }
    }
    // data is given in ints, convert to float
    else if( aiPTI_Integer == prop->mType)  {
        iWrite = prop->mDataLength / sizeof(int32_t);
        if (pMax) {
            iWrite = std::min(*pMax,iWrite); ;
        }
        for (unsigned int a = 0; a < iWrite;++a)    {
            pOut[a] = static_cast<ai_real> ( reinterpret_cast<int32_t*>(prop->mData)[a] );
        }
        if (pMax) {
            *pMax = iWrite;
        }
    }
    // a string ... read floats separated by spaces
    else {
        if (pMax) {
            iWrite = *pMax;
        }
        // strings are zero-terminated with a 32 bit length prefix, so this is safe
        const char *cur = prop->mData + 4;
        ai_assert( prop->mDataLength >= 5 );
        ai_assert( !prop->mData[ prop->mDataLength - 1 ] );
        for ( unsigned int a = 0; ;++a) {
            cur = fast_atoreal_move<ai_real>(cur,pOut[a]);
            if ( a==iWrite-1 ) {
                break;
            }
            if ( !IsSpace(*cur) ) {
                ASSIMP_LOG_ERROR("Material property" + std::string(pKey) +
                    " is a string; failed to parse a float array out of it.");
                return AI_FAILURE;
            }
        }

        if (pMax) {
            *pMax = iWrite;
        }
    }
    return AI_SUCCESS;
}

// ------------------------------------------------------------------------------------------------
// Get an array if integers from the material
aiReturn aiGetMaterialIntegerArray(const aiMaterial* pMat,
    const char* pKey,
    unsigned int type,
    unsigned int index,
    int* pOut,
    unsigned int* pMax)
{
    ai_assert( pOut != NULL );
    ai_assert( pMat != NULL );

    const aiMaterialProperty* prop;
    aiGetMaterialProperty(pMat,pKey,type,index,(const aiMaterialProperty**) &prop);
    if (!prop) {
        return AI_FAILURE;
    }

    // data is given in ints, simply copy it
    unsigned int iWrite = 0;
    if( aiPTI_Integer == prop->mType || aiPTI_Buffer == prop->mType)    {
        iWrite = std::max(static_cast<unsigned int>(prop->mDataLength / sizeof(int32_t)), 1u);
        if (pMax) {
            iWrite = std::min(*pMax,iWrite);
        }
        if (1 == prop->mDataLength) {
            // bool type, 1 byte
            *pOut = static_cast<int>(*prop->mData);
        }
        else {
            for (unsigned int a = 0; a < iWrite;++a) {
                pOut[a] = static_cast<int>(reinterpret_cast<int32_t*>(prop->mData)[a]);
            }
        }
        if (pMax) {
            *pMax = iWrite;
        }
    }
    // data is given in floats convert to int
    else if( aiPTI_Float == prop->mType)    {
        iWrite = prop->mDataLength / sizeof(float);
        if (pMax) {
            iWrite = std::min(*pMax,iWrite); ;
        }
        for (unsigned int a = 0; a < iWrite;++a) {
            pOut[a] = static_cast<int>(reinterpret_cast<float*>(prop->mData)[a]);
        }
        if (pMax) {
            *pMax = iWrite;
        }
    }
    // it is a string ... no way to read something out of this
    else    {
        if (pMax) {
            iWrite = *pMax;
        }
        // strings are zero-terminated with a 32 bit length prefix, so this is safe
        const char *cur =  prop->mData+4;
        ai_assert( prop->mDataLength >= 5 );
        ai_assert( !prop->mData[ prop->mDataLength - 1 ] );
        for (unsigned int a = 0; ;++a) {
            pOut[a] = strtol10(cur,&cur);
            if(a==iWrite-1) {
                break;
            }
            if(!IsSpace(*cur)) {
                ASSIMP_LOG_ERROR("Material property" + std::string(pKey) +
                    " is a string; failed to parse an integer array out of it.");
                return AI_FAILURE;
            }
        }

        if (pMax) {
            *pMax = iWrite;
        }
    }
    return AI_SUCCESS;
}

// ------------------------------------------------------------------------------------------------
// Get a color (3 or 4 floats) from the material
aiReturn aiGetMaterialColor(const aiMaterial* pMat,
    const char* pKey,
    unsigned int type,
    unsigned int index,
    aiColor4D* pOut)
{
    unsigned int iMax = 4;
    const aiReturn eRet = aiGetMaterialFloatArray(pMat,pKey,type,index,(ai_real*)pOut,&iMax);

    // if no alpha channel is defined: set it to 1.0
    if (3 == iMax) {
        pOut->a = 1.0;
    }

    return eRet;
}

// ------------------------------------------------------------------------------------------------
// Get a aiUVTransform (4 floats) from the material
aiReturn aiGetMaterialUVTransform(const aiMaterial* pMat,
    const char* pKey,
    unsigned int type,
    unsigned int index,
    aiUVTransform* pOut)
{
    unsigned int iMax = 4;
    return aiGetMaterialFloatArray(pMat,pKey,type,index,(ai_real*)pOut,&iMax);
}

// ------------------------------------------------------------------------------------------------
// Get a string from the material
aiReturn aiGetMaterialString(const aiMaterial* pMat,
    const char* pKey,
    unsigned int type,
    unsigned int index,
    aiString* pOut)
{
    ai_assert (pOut != NULL);

    const aiMaterialProperty* prop;
    aiGetMaterialProperty(pMat,pKey,type,index,(const aiMaterialProperty**)&prop);
    if (!prop) {
        return AI_FAILURE;
    }

    if( aiPTI_String == prop->mType) {
        ai_assert(prop->mDataLength>=5);

        // The string is stored as 32 but length prefix followed by zero-terminated UTF8 data
        pOut->length = static_cast<unsigned int>(*reinterpret_cast<uint32_t*>(prop->mData));

        ai_assert( pOut->length+1+4==prop->mDataLength );
        ai_assert( !prop->mData[ prop->mDataLength - 1 ] );
        memcpy(pOut->data,prop->mData+4,pOut->length+1);
    }
    else {
        // TODO - implement lexical cast as well
        ASSIMP_LOG_ERROR("Material property" + std::string(pKey) +
            " was found, but is no string" );
        return AI_FAILURE;
    }
    return AI_SUCCESS;
}

// ------------------------------------------------------------------------------------------------
// Get the number of textures on a particular texture stack
unsigned int aiGetMaterialTextureCount(const C_STRUCT aiMaterial* pMat,
    C_ENUM aiTextureType type)
{
    ai_assert (pMat != NULL);

    // Textures are always stored with ascending indices (ValidateDS provides a check, so we don't need to do it again)
    unsigned int max = 0;
    for (unsigned int i = 0; i < pMat->mNumProperties;++i) {
        aiMaterialProperty* prop = pMat->mProperties[i];

        if ( prop /* just a sanity check ... */
            && 0 == strcmp( prop->mKey.data, _AI_MATKEY_TEXTURE_BASE )
            && prop->mSemantic == type) {

            max = std::max(max,prop->mIndex+1);
        }
    }
    return max;
}

// ------------------------------------------------------------------------------------------------
aiReturn aiGetMaterialTexture(const C_STRUCT aiMaterial* mat,
    aiTextureType type,
    unsigned int  index,
    C_STRUCT aiString* path,
    aiTextureMapping* _mapping  /*= NULL*/,
    unsigned int* uvindex       /*= NULL*/,
    ai_real* blend              /*= NULL*/,
    aiTextureOp* op             /*= NULL*/,
    aiTextureMapMode* mapmode   /*= NULL*/,
    unsigned int* flags         /*= NULL*/
    )
{
    ai_assert( NULL != mat );
    ai_assert( NULL != path );

    // Get the path to the texture
    if (AI_SUCCESS != aiGetMaterialString(mat,AI_MATKEY_TEXTURE(type,index),path))  {
        return AI_FAILURE;
    }

    // Determine mapping type
    int mapping_ = static_cast<int>(aiTextureMapping_UV);
    aiGetMaterialInteger(mat,AI_MATKEY_MAPPING(type,index), &mapping_);
    aiTextureMapping mapping = static_cast<aiTextureMapping>(mapping_);
    if (_mapping)
        *_mapping = mapping;

    // Get UV index
    if (aiTextureMapping_UV == mapping && uvindex)  {
        aiGetMaterialInteger(mat,AI_MATKEY_UVWSRC(type,index),(int*)uvindex);
    }
    // Get blend factor
    if (blend)  {
        aiGetMaterialFloat(mat,AI_MATKEY_TEXBLEND(type,index),blend);
    }
    // Get texture operation
    if (op){
        aiGetMaterialInteger(mat,AI_MATKEY_TEXOP(type,index),(int*)op);
    }
    // Get texture mapping modes
    if (mapmode)    {
        aiGetMaterialInteger(mat,AI_MATKEY_MAPPINGMODE_U(type,index),(int*)&mapmode[0]);
        aiGetMaterialInteger(mat,AI_MATKEY_MAPPINGMODE_V(type,index),(int*)&mapmode[1]);
    }
    // Get texture flags
    if (flags){
        aiGetMaterialInteger(mat,AI_MATKEY_TEXFLAGS(type,index),(int*)flags);
    }

    return AI_SUCCESS;
}


static const unsigned int DefaultNumAllocated = 5;

// ------------------------------------------------------------------------------------------------
// Construction. Actually the one and only way to get an aiMaterial instance
aiMaterial::aiMaterial() 
: mProperties( nullptr )
, mNumProperties( 0 )
, mNumAllocated( DefaultNumAllocated ) {
    // Allocate 5 entries by default
    mProperties = new aiMaterialProperty*[ DefaultNumAllocated ];
}

// ------------------------------------------------------------------------------------------------
aiMaterial::~aiMaterial()
{
    Clear();

    delete[] mProperties;
}

// ------------------------------------------------------------------------------------------------
aiString aiMaterial::GetName() {
    aiString name;
    Get(AI_MATKEY_NAME, name);

    return name;
}

// ------------------------------------------------------------------------------------------------
void aiMaterial::Clear()
{
    for ( unsigned int i = 0; i < mNumProperties; ++i )    {
        // delete this entry
        delete mProperties[ i ];
        AI_DEBUG_INVALIDATE_PTR(mProperties[i]);
    }
    mNumProperties = 0;

    // The array remains allocated, we just invalidated its contents
}

// ------------------------------------------------------------------------------------------------
aiReturn aiMaterial::RemoveProperty ( const char* pKey,unsigned int type, unsigned int index )
{
    ai_assert( nullptr != pKey );

    for (unsigned int i = 0; i < mNumProperties;++i) {
        aiMaterialProperty* prop = mProperties[i];

        if (prop && !strcmp( prop->mKey.data, pKey ) &&
            prop->mSemantic == type && prop->mIndex == index)
        {
            // Delete this entry
            delete mProperties[i];

            // collapse the array behind --.
            --mNumProperties;
            for (unsigned int a = i; a < mNumProperties;++a)    {
                mProperties[a] = mProperties[a+1];
            }
            return AI_SUCCESS;
        }
    }

    return AI_FAILURE;
}

// ------------------------------------------------------------------------------------------------
aiReturn aiMaterial::AddBinaryProperty (const void* pInput,
    unsigned int pSizeInBytes,
    const char* pKey,
    unsigned int type,
    unsigned int index,
    aiPropertyTypeInfo pType
    )
{
    ai_assert( pInput != NULL );
    ai_assert( pKey != NULL );
    ai_assert( 0 != pSizeInBytes );

    if ( 0 == pSizeInBytes ) {

    }

    // first search the list whether there is already an entry with this key
    unsigned int iOutIndex( UINT_MAX );
    for ( unsigned int i = 0; i < mNumProperties; ++i ) {
        aiMaterialProperty *prop( mProperties[ i ] );

        if (prop /* just for safety */ && !strcmp( prop->mKey.data, pKey ) &&
            prop->mSemantic == type && prop->mIndex == index){

            delete mProperties[i];
            iOutIndex = i;
        }
    }

    // Allocate a new material property
    aiMaterialProperty* pcNew = new aiMaterialProperty();

    // .. and fill it
    pcNew->mType = pType;
    pcNew->mSemantic = type;
    pcNew->mIndex = index;

    pcNew->mDataLength = pSizeInBytes;
    pcNew->mData = new char[pSizeInBytes];
    memcpy (pcNew->mData,pInput,pSizeInBytes);

    pcNew->mKey.length = ::strlen(pKey);
    ai_assert ( MAXLEN > pcNew->mKey.length);
    strcpy( pcNew->mKey.data, pKey );

    if (UINT_MAX != iOutIndex)  {
        mProperties[iOutIndex] = pcNew;
        return AI_SUCCESS;
    }

    // resize the array ... double the storage allocated
    if (mNumProperties == mNumAllocated)    {
        const unsigned int iOld = mNumAllocated;
        mNumAllocated *= 2;

        aiMaterialProperty** ppTemp;
        try {
            ppTemp = new aiMaterialProperty*[mNumAllocated];
        } catch (std::bad_alloc&) {
            delete pcNew;
            return AI_OUTOFMEMORY;
        }

        // just copy all items over; then replace the old array
        memcpy (ppTemp,mProperties,iOld * sizeof(void*));

        delete[] mProperties;
        mProperties = ppTemp;
    }
    // push back ...
    mProperties[mNumProperties++] = pcNew;

    return AI_SUCCESS;
}

// ------------------------------------------------------------------------------------------------
aiReturn aiMaterial::AddProperty (const aiString* pInput,
    const char* pKey,
    unsigned int type,
    unsigned int index)
{
    ai_assert(sizeof(ai_uint32)==4);
    return AddBinaryProperty(pInput,
        static_cast<unsigned int>(pInput->length+1+4),
        pKey,
        type,
        index,
        aiPTI_String);
}

// ------------------------------------------------------------------------------------------------
uint32_t Assimp::ComputeMaterialHash(const aiMaterial* mat, bool includeMatName /*= false*/)
{
    uint32_t hash = 1503; // magic start value, chosen to be my birthday :-)
    for ( unsigned int i = 0; i < mat->mNumProperties; ++i ) {
        aiMaterialProperty* prop;

        // Exclude all properties whose first character is '?' from the hash
        // See doc for aiMaterialProperty.
        if ((prop = mat->mProperties[i]) && (includeMatName || prop->mKey.data[0] != '?'))  {

            hash = SuperFastHash(prop->mKey.data,(unsigned int)prop->mKey.length,hash);
            hash = SuperFastHash(prop->mData,prop->mDataLength,hash);

            // Combine the semantic and the index with the hash
            hash = SuperFastHash((const char*)&prop->mSemantic,sizeof(unsigned int),hash);
            hash = SuperFastHash((const char*)&prop->mIndex,sizeof(unsigned int),hash);
        }
    }
    return hash;
}

// ------------------------------------------------------------------------------------------------
void aiMaterial::CopyPropertyList(aiMaterial* pcDest,
    const aiMaterial* pcSrc
    )
{
    ai_assert(NULL != pcDest);
    ai_assert(NULL != pcSrc);

    unsigned int iOldNum = pcDest->mNumProperties;
    pcDest->mNumAllocated += pcSrc->mNumAllocated;
    pcDest->mNumProperties += pcSrc->mNumProperties;

    aiMaterialProperty** pcOld = pcDest->mProperties;
    pcDest->mProperties = new aiMaterialProperty*[pcDest->mNumAllocated];

    if (iOldNum && pcOld)   {
        for (unsigned int i = 0; i < iOldNum;++i) {
            pcDest->mProperties[i] = pcOld[i];
        }
    }

    if ( pcOld ) {
        delete[] pcOld;
    }

    for (unsigned int i = iOldNum; i< pcDest->mNumProperties;++i)   {
        aiMaterialProperty* propSrc = pcSrc->mProperties[i];

        // search whether we have already a property with this name -> if yes, overwrite it
        aiMaterialProperty* prop;
        for ( unsigned int q = 0; q < iOldNum; ++q ) {
            prop = pcDest->mProperties[q];
            if (prop /* just for safety */ && prop->mKey == propSrc->mKey && prop->mSemantic == propSrc->mSemantic
                && prop->mIndex == propSrc->mIndex) {
                delete prop;

                // collapse the whole array ...
                memmove(&pcDest->mProperties[q],&pcDest->mProperties[q+1],i-q);
                i--;
                pcDest->mNumProperties--;
            }
        }

        // Allocate the output property and copy the source property
        prop = pcDest->mProperties[i] = new aiMaterialProperty();
        prop->mKey = propSrc->mKey;
        prop->mDataLength = propSrc->mDataLength;
        prop->mType = propSrc->mType;
        prop->mSemantic = propSrc->mSemantic;
        prop->mIndex = propSrc->mIndex;

        prop->mData = new char[propSrc->mDataLength];
        memcpy(prop->mData,propSrc->mData,prop->mDataLength);
    }
}
