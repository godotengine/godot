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

/** @file material.inl
 *  @brief Defines the C++ getters for the material system
 */

#pragma once
#ifndef AI_MATERIAL_INL_INC
#define AI_MATERIAL_INL_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiPropertyTypeInfo ai_real_to_property_type_info(float) {
	return aiPTI_Float;
}

AI_FORCE_INLINE
aiPropertyTypeInfo ai_real_to_property_type_info(double) {
	return aiPTI_Double;
}
// ---------------------------------------------------------------------------

//! @cond never

// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::GetTexture( aiTextureType type,
       unsigned int  index,
       C_STRUCT aiString* path,
       aiTextureMapping* mapping    /*= NULL*/,
       unsigned int* uvindex        /*= NULL*/,
       ai_real* blend               /*= NULL*/,
       aiTextureOp* op              /*= NULL*/,
       aiTextureMapMode* mapmode    /*= NULL*/) const {
    return ::aiGetMaterialTexture(this,type,index,path,mapping,uvindex,blend,op,mapmode);
}

// ---------------------------------------------------------------------------
AI_FORCE_INLINE
unsigned int aiMaterial::GetTextureCount(aiTextureType type) const {
    return ::aiGetMaterialTextureCount(this,type);
}

// ---------------------------------------------------------------------------
template <typename Type>
AI_FORCE_INLINE
aiReturn aiMaterial::Get(const char* pKey,unsigned int type,
        unsigned int idx, Type* pOut,
        unsigned int* pMax) const {
    unsigned int iNum = pMax ? *pMax : 1;

    const aiMaterialProperty* prop;
    const aiReturn ret = ::aiGetMaterialProperty(this,pKey,type,idx,
        (const aiMaterialProperty**)&prop);
    if ( AI_SUCCESS == ret )    {

        if (prop->mDataLength < sizeof(Type)*iNum) {
            return AI_FAILURE;
        }

        if (prop->mType != aiPTI_Buffer) {
            return AI_FAILURE;
        }

        iNum = std::min((size_t)iNum,prop->mDataLength / sizeof(Type));
        ::memcpy(pOut,prop->mData,iNum * sizeof(Type));
        if (pMax) {
            *pMax = iNum;
        }
    }
    return ret;
}

// ---------------------------------------------------------------------------
template <typename Type>
AI_FORCE_INLINE
aiReturn aiMaterial::Get(const char* pKey,unsigned int type,
        unsigned int idx,Type& pOut) const {
    const aiMaterialProperty* prop;
    const aiReturn ret = ::aiGetMaterialProperty(this,pKey,type,idx,
        (const aiMaterialProperty**)&prop);
    if ( AI_SUCCESS == ret ) {

        if (prop->mDataLength < sizeof(Type)) {
            return AI_FAILURE;
        }

        if (prop->mType != aiPTI_Buffer) {
            return AI_FAILURE;
        }

        ::memcpy( &pOut, prop->mData, sizeof( Type ) );
    }
    return ret;
}

// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::Get(const char* pKey,unsigned int type,
        unsigned int idx,ai_real* pOut,
        unsigned int* pMax) const {
    return ::aiGetMaterialFloatArray(this,pKey,type,idx,pOut,pMax);
}
// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::Get(const char* pKey,unsigned int type,
        unsigned int idx,int* pOut,
        unsigned int* pMax) const {
    return ::aiGetMaterialIntegerArray(this,pKey,type,idx,pOut,pMax);
}
// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::Get(const char* pKey,unsigned int type,
        unsigned int idx,ai_real& pOut) const {
    return aiGetMaterialFloat(this,pKey,type,idx,&pOut);
}
// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::Get(const char* pKey,unsigned int type,
        unsigned int idx,int& pOut) const {
    return aiGetMaterialInteger(this,pKey,type,idx,&pOut);
}
// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::Get(const char* pKey,unsigned int type,
        unsigned int idx,aiColor4D& pOut) const {
    return aiGetMaterialColor(this,pKey,type,idx,&pOut);
}
// ---------------------------------------------------------------------------
AI_FORCE_INLINE aiReturn aiMaterial::Get(const char* pKey,unsigned int type,
        unsigned int idx,aiColor3D& pOut) const {
    aiColor4D c;
    const aiReturn ret = aiGetMaterialColor(this,pKey,type,idx,&c);
    pOut = aiColor3D(c.r,c.g,c.b);
    return ret;
}
// ---------------------------------------------------------------------------
AI_FORCE_INLINE aiReturn aiMaterial::Get(const char* pKey,unsigned int type,
        unsigned int idx,aiString& pOut) const {
    return aiGetMaterialString(this,pKey,type,idx,&pOut);
}
// ---------------------------------------------------------------------------
AI_FORCE_INLINE aiReturn aiMaterial::Get(const char* pKey,unsigned int type,
        unsigned int idx,aiUVTransform& pOut) const {
    return aiGetMaterialUVTransform(this,pKey,type,idx,&pOut);
}

// ---------------------------------------------------------------------------
template<class TYPE>
aiReturn aiMaterial::AddProperty (const TYPE* pInput,
    const unsigned int pNumValues,
    const char* pKey,
    unsigned int type,
    unsigned int index)
{
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(TYPE),
        pKey,type,index,aiPTI_Buffer);
}

// ---------------------------------------------------------------------------
AI_FORCE_INLINE aiReturn aiMaterial::AddProperty(const float* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(float),
        pKey,type,index,aiPTI_Float);
}

// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty(const double* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(double),
        pKey,type,index,aiPTI_Double);
}

// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty(const aiUVTransform* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(aiUVTransform),
        pKey,type,index,ai_real_to_property_type_info(pInput->mRotation));
}

// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty(const aiColor4D* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(aiColor4D),
        pKey,type,index,ai_real_to_property_type_info(pInput->a));
}

// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty(const aiColor3D* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(aiColor3D),
        pKey,type,index,ai_real_to_property_type_info(pInput->b));
}

// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty(const aiVector3D* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(aiVector3D),
        pKey,type,index,ai_real_to_property_type_info(pInput->x));
}

// ---------------------------------------------------------------------------
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty(const int* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(int),
        pKey,type,index,aiPTI_Integer);
}


// ---------------------------------------------------------------------------
// The template specializations below are for backwards compatibility.
// The recommended way to add material properties is using the non-template
// overloads.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
template<>
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty<float>(const float* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(float),
        pKey,type,index,aiPTI_Float);
}

// ---------------------------------------------------------------------------
template<>
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty<double>(const double* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(double),
        pKey,type,index,aiPTI_Double);
}

// ---------------------------------------------------------------------------
template<>
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty<aiUVTransform>(const aiUVTransform* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(aiUVTransform),
        pKey,type,index,aiPTI_Float);
}

// ---------------------------------------------------------------------------
template<>
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty<aiColor4D>(const aiColor4D* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(aiColor4D),
        pKey,type,index,aiPTI_Float);
}

// ---------------------------------------------------------------------------
template<>
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty<aiColor3D>(const aiColor3D* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(aiColor3D),
        pKey,type,index,aiPTI_Float);
}

// ---------------------------------------------------------------------------
template<>
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty<aiVector3D>(const aiVector3D* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(aiVector3D),
        pKey,type,index,aiPTI_Float);
}

// ---------------------------------------------------------------------------
template<>
AI_FORCE_INLINE
aiReturn aiMaterial::AddProperty<int>(const int* pInput,
        const unsigned int pNumValues,
        const char* pKey,
        unsigned int type,
        unsigned int index) {
    return AddBinaryProperty((const void*)pInput,
        pNumValues * sizeof(int),
        pKey,type,index,aiPTI_Integer);
}

//! @endcond

#endif //! AI_MATERIAL_INL_INC
