//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2016 LunarG, Inc.
// Copyright (C) 2015-2016 Google, Inc.
// Copyright (C) 2017 ARM Limited.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef _TYPES_INCLUDED
#define _TYPES_INCLUDED

#include "../Include/Common.h"
#include "../Include/BaseTypes.h"
#include "../Public/ShaderLang.h"
#include "arrays.h"
#include "SpirvIntrinsics.h"

#include <algorithm>

namespace glslang {

class TIntermAggregate;

const int GlslangMaxTypeLength = 200;  // TODO: need to print block/struct one member per line, so this can stay bounded

const char* const AnonymousPrefix = "anon@"; // for something like a block whose members can be directly accessed
inline bool IsAnonymous(const TString& name)
{
    return name.compare(0, 5, AnonymousPrefix) == 0;
}

//
// Details within a sampler type
//
enum TSamplerDim {
    EsdNone,
    Esd1D,
    Esd2D,
    Esd3D,
    EsdCube,
    EsdRect,
    EsdBuffer,
    EsdSubpass,  // goes only with non-sampled image (image is true)
    EsdNumDims
};

struct TSampler {   // misnomer now; includes images, textures without sampler, and textures with sampler
    TBasicType type : 8;  // type returned by sampler
    TSamplerDim dim : 8;
    bool    arrayed : 1;
    bool     shadow : 1;
    bool         ms : 1;
    bool      image : 1;  // image, combined should be false
    bool   combined : 1;  // true means texture is combined with a sampler, false means texture with no sampler
    bool    sampler : 1;  // true means a pure sampler, other fields should be clear()

#ifdef GLSLANG_WEB
    bool is1D()          const { return false; }
    bool isBuffer()      const { return false; }
    bool isRect()        const { return false; }
    bool isSubpass()     const { return false; }
    bool isCombined()    const { return true; }
    bool isImage()       const { return false; }
    bool isImageClass()  const { return false; }
    bool isMultiSample() const { return false; }
    bool isExternal()    const { return false; }
    void setExternal(bool e) { }
    bool isYuv()         const { return false; }
#else
    unsigned int vectorSize : 3;  // vector return type size.
    // Some languages support structures as sample results.  Storing the whole structure in the
    // TSampler is too large, so there is an index to a separate table.
    static const unsigned structReturnIndexBits = 4;                        // number of index bits to use.
    static const unsigned structReturnSlots = (1<<structReturnIndexBits)-1; // number of valid values
    static const unsigned noReturnStruct = structReturnSlots;               // value if no return struct type.
    // Index into a language specific table of texture return structures.
    unsigned int structReturnIndex : structReturnIndexBits;

    bool   external : 1;  // GL_OES_EGL_image_external
    bool        yuv : 1;  // GL_EXT_YUV_target

#ifdef ENABLE_HLSL
    unsigned int getVectorSize() const { return vectorSize; }
    void clearReturnStruct() { structReturnIndex = noReturnStruct; }
    bool hasReturnStruct() const { return structReturnIndex != noReturnStruct; }
    unsigned getStructReturnIndex() const { return structReturnIndex; }
#endif

    bool is1D()          const { return dim == Esd1D; }
    bool is2D()          const { return dim == Esd2D; }
    bool isBuffer()      const { return dim == EsdBuffer; }
    bool isRect()        const { return dim == EsdRect; }
    bool isSubpass()     const { return dim == EsdSubpass; }
    bool isCombined()    const { return combined; }
    bool isImage()       const { return image && !isSubpass(); }
    bool isImageClass()  const { return image; }
    bool isMultiSample() const { return ms; }
    bool isExternal()    const { return external; }
    void setExternal(bool e) { external = e; }
    bool isYuv()         const { return yuv; }
#endif
    bool isTexture()     const { return !sampler && !image; }
    bool isPureSampler() const { return sampler; }

    void setCombined(bool c) { combined = c; }
    void setBasicType(TBasicType t) { type = t; }
    TBasicType getBasicType()  const { return type; }
    bool isShadow()      const { return shadow; }
    bool isArrayed()     const { return arrayed; }

    void clear()
    {
        type = EbtVoid;
        dim = EsdNone;
        arrayed = false;
        shadow = false;
        ms = false;
        image = false;
        combined = false;
        sampler = false;
#ifndef GLSLANG_WEB
        external = false;
        yuv = false;
#endif

#ifdef ENABLE_HLSL
        clearReturnStruct();
        // by default, returns a single vec4;
        vectorSize = 4;
#endif
    }

    // make a combined sampler and texture
    void set(TBasicType t, TSamplerDim d, bool a = false, bool s = false, bool m = false)
    {
        clear();
        type = t;
        dim = d;
        arrayed = a;
        shadow = s;
        ms = m;
        combined = true;
    }

    // make an image
    void setImage(TBasicType t, TSamplerDim d, bool a = false, bool s = false, bool m = false)
    {
        clear();
        type = t;
        dim = d;
        arrayed = a;
        shadow = s;
        ms = m;
        image = true;
    }

    // make a texture with no sampler
    void setTexture(TBasicType t, TSamplerDim d, bool a = false, bool s = false, bool m = false)
    {
        clear();
        type = t;
        dim = d;
        arrayed = a;
        shadow = s;
        ms = m;
    }

    // make a pure sampler, no texture, no image, nothing combined, the 'sampler' keyword
    void setPureSampler(bool s)
    {
        clear();
        sampler = true;
        shadow = s;
    }

#ifndef GLSLANG_WEB
    // make a subpass input attachment
    void setSubpass(TBasicType t, bool m = false)
    {
        clear();
        type = t;
        image = true;
        dim = EsdSubpass;
        ms = m;
    }
#endif

    bool operator==(const TSampler& right) const
    {
        return      type == right.type &&
                     dim == right.dim &&
                 arrayed == right.arrayed &&
                  shadow == right.shadow &&
         isMultiSample() == right.isMultiSample() &&
          isImageClass() == right.isImageClass() &&
            isCombined() == right.isCombined() &&
         isPureSampler() == right.isPureSampler() &&
            isExternal() == right.isExternal() &&
                 isYuv() == right.isYuv()
#ifdef ENABLE_HLSL
      && getVectorSize() == right.getVectorSize() &&
  getStructReturnIndex() == right.getStructReturnIndex()
#endif
        ;
    }

    bool operator!=(const TSampler& right) const
    {
        return ! operator==(right);
    }

    TString getString() const
    {
        TString s;

        if (isPureSampler()) {
            s.append("sampler");
            return s;
        }

        switch (type) {
        case EbtInt:    s.append("i");   break;
        case EbtUint:   s.append("u");   break;
#ifndef GLSLANG_WEB
        case EbtFloat16: s.append("f16"); break;
        case EbtInt8:   s.append("i8");  break;
        case EbtUint16: s.append("u8");  break;
        case EbtInt16:  s.append("i16"); break;
        case EbtUint8:  s.append("u16"); break;
        case EbtInt64:  s.append("i64"); break;
        case EbtUint64: s.append("u64"); break;
#endif
        default:  break;
        }
        if (isImageClass()) {
            if (isSubpass())
                s.append("subpass");
            else
                s.append("image");
        } else if (isCombined()) {
            s.append("sampler");
        } else {
            s.append("texture");
        }
        if (isExternal()) {
            s.append("ExternalOES");
            return s;
        }
        if (isYuv()) {
            return "__" + s + "External2DY2YEXT";
        }
        switch (dim) {
        case Esd2D:      s.append("2D");      break;
        case Esd3D:      s.append("3D");      break;
        case EsdCube:    s.append("Cube");    break;
#ifndef GLSLANG_WEB
        case Esd1D:      s.append("1D");      break;
        case EsdRect:    s.append("2DRect");  break;
        case EsdBuffer:  s.append("Buffer");  break;
        case EsdSubpass: s.append("Input"); break;
#endif
        default:  break;  // some compilers want this
        }
        if (isMultiSample())
            s.append("MS");
        if (arrayed)
            s.append("Array");
        if (shadow)
            s.append("Shadow");

        return s;
    }
};

//
// Need to have association of line numbers to types in a list for building structs.
//
class TType;
struct TTypeLoc {
    TType* type;
    TSourceLoc loc;
};
typedef TVector<TTypeLoc> TTypeList;

typedef TVector<TString*> TIdentifierList;

//
// Following are a series of helper enums for managing layouts and qualifiers,
// used for TPublicType, TType, others.
//

enum TLayoutPacking {
    ElpNone,
    ElpShared,      // default, but different than saying nothing
    ElpStd140,
    ElpStd430,
    ElpPacked,
    ElpScalar,
    ElpCount        // If expanding, see bitfield width below
};

enum TLayoutMatrix {
    ElmNone,
    ElmRowMajor,
    ElmColumnMajor, // default, but different than saying nothing
    ElmCount        // If expanding, see bitfield width below
};

// Union of geometry shader and tessellation shader geometry types.
// They don't go into TType, but rather have current state per shader or
// active parser type (TPublicType).
enum TLayoutGeometry {
    ElgNone,
    ElgPoints,
    ElgLines,
    ElgLinesAdjacency,
    ElgLineStrip,
    ElgTriangles,
    ElgTrianglesAdjacency,
    ElgTriangleStrip,
    ElgQuads,
    ElgIsolines,
};

enum TVertexSpacing {
    EvsNone,
    EvsEqual,
    EvsFractionalEven,
    EvsFractionalOdd
};

enum TVertexOrder {
    EvoNone,
    EvoCw,
    EvoCcw
};

// Note: order matters, as type of format is done by comparison.
enum TLayoutFormat {
    ElfNone,

    // Float image
    ElfRgba32f,
    ElfRgba16f,
    ElfR32f,
    ElfRgba8,
    ElfRgba8Snorm,

    ElfEsFloatGuard,    // to help with comparisons

    ElfRg32f,
    ElfRg16f,
    ElfR11fG11fB10f,
    ElfR16f,
    ElfRgba16,
    ElfRgb10A2,
    ElfRg16,
    ElfRg8,
    ElfR16,
    ElfR8,
    ElfRgba16Snorm,
    ElfRg16Snorm,
    ElfRg8Snorm,
    ElfR16Snorm,
    ElfR8Snorm,

    ElfFloatGuard,      // to help with comparisons

    // Int image
    ElfRgba32i,
    ElfRgba16i,
    ElfRgba8i,
    ElfR32i,

    ElfEsIntGuard,     // to help with comparisons

    ElfRg32i,
    ElfRg16i,
    ElfRg8i,
    ElfR16i,
    ElfR8i,
    ElfR64i,

    ElfIntGuard,       // to help with comparisons

    // Uint image
    ElfRgba32ui,
    ElfRgba16ui,
    ElfRgba8ui,
    ElfR32ui,

    ElfEsUintGuard,    // to help with comparisons

    ElfRg32ui,
    ElfRg16ui,
    ElfRgb10a2ui,
    ElfRg8ui,
    ElfR16ui,
    ElfR8ui,
    ElfR64ui,

    ElfCount
};

enum TLayoutDepth {
    EldNone,
    EldAny,
    EldGreater,
    EldLess,
    EldUnchanged,

    EldCount
};

enum TLayoutStencil {
    ElsNone,
    ElsRefUnchangedFrontAMD,
    ElsRefGreaterFrontAMD,
    ElsRefLessFrontAMD,
    ElsRefUnchangedBackAMD,
    ElsRefGreaterBackAMD,
    ElsRefLessBackAMD,

    ElsCount
};

enum TBlendEquationShift {
    // No 'EBlendNone':
    // These are used as bit-shift amounts.  A mask of such shifts will have type 'int',
    // and in that space, 0 means no bits set, or none.  In this enum, 0 means (1 << 0), a bit is set.
    EBlendMultiply,
    EBlendScreen,
    EBlendOverlay,
    EBlendDarken,
    EBlendLighten,
    EBlendColordodge,
    EBlendColorburn,
    EBlendHardlight,
    EBlendSoftlight,
    EBlendDifference,
    EBlendExclusion,
    EBlendHslHue,
    EBlendHslSaturation,
    EBlendHslColor,
    EBlendHslLuminosity,
    EBlendAllEquations,

    EBlendCount
};

enum TInterlockOrdering {
    EioNone,
    EioPixelInterlockOrdered,
    EioPixelInterlockUnordered,
    EioSampleInterlockOrdered,
    EioSampleInterlockUnordered,
    EioShadingRateInterlockOrdered,
    EioShadingRateInterlockUnordered,

    EioCount,
};

enum TShaderInterface
{
    // Includes both uniform blocks and buffer blocks
    EsiUniform = 0,
    EsiInput,
    EsiOutput,
    EsiNone,

    EsiCount
};

class TQualifier {
public:
    static const int layoutNotSet = -1;

    void clear()
    {
        precision = EpqNone;
        invariant = false;
        makeTemporary();
        declaredBuiltIn = EbvNone;
#ifndef GLSLANG_WEB
        noContraction = false;
        nullInit = false;
        spirvByReference = false;
        spirvLiteral = false;
#endif
        defaultBlock = false;
    }

    // drop qualifiers that don't belong in a temporary variable
    void makeTemporary()
    {
        semanticName = nullptr;
        storage = EvqTemporary;
        builtIn = EbvNone;
        clearInterstage();
        clearMemory();
        specConstant = false;
        nonUniform = false;
        nullInit = false;
        defaultBlock = false;
        clearLayout();
#ifndef GLSLANG_WEB
        spirvStorageClass = -1;
        spirvDecorate = nullptr;
        spirvByReference = false;
        spirvLiteral = false;
#endif
    }

    void clearInterstage()
    {
        clearInterpolation();
#ifndef GLSLANG_WEB
        patch = false;
        sample = false;
#endif
    }

    void clearInterpolation()
    {
        centroid     = false;
        smooth       = false;
        flat         = false;
#ifndef GLSLANG_WEB
        nopersp      = false;
        explicitInterp = false;
        pervertexNV = false;
        perPrimitiveNV = false;
        perViewNV = false;
        perTaskNV = false;
#endif
        pervertexEXT = false;
    }

    void clearMemory()
    {
#ifndef GLSLANG_WEB
        coherent     = false;
        devicecoherent = false;
        queuefamilycoherent = false;
        workgroupcoherent = false;
        subgroupcoherent  = false;
        shadercallcoherent = false;
        nonprivate = false;
        volatil      = false;
        restrict     = false;
        readonly     = false;
        writeonly    = false;
#endif
    }

    const char*         semanticName;
    TStorageQualifier   storage   : 6;
    TBuiltInVariable    builtIn   : 9;
    TBuiltInVariable    declaredBuiltIn : 9;
    static_assert(EbvLast < 256, "need to increase size of TBuiltInVariable bitfields!");
    TPrecisionQualifier precision : 3;
    bool invariant    : 1; // require canonical treatment for cross-shader invariance
    bool centroid     : 1;
    bool smooth       : 1;
    bool flat         : 1;
    // having a constant_id is not sufficient: expressions have no id, but are still specConstant
    bool specConstant : 1;
    bool nonUniform   : 1;
    bool explicitOffset   : 1;
    bool defaultBlock : 1; // default blocks with matching names have structures merged when linking

#ifdef GLSLANG_WEB
    bool isWriteOnly() const { return false; }
    bool isReadOnly() const { return false; }
    bool isRestrict() const { return false; }
    bool isCoherent() const { return false; }
    bool isVolatile() const { return false; }
    bool isSample() const { return false; }
    bool isMemory() const { return false; }
    bool isMemoryQualifierImageAndSSBOOnly() const { return false; }
    bool bufferReferenceNeedsVulkanMemoryModel() const { return false; }
    bool isInterpolation() const { return flat || smooth; }
    bool isExplicitInterpolation() const { return false; }
    bool isAuxiliary() const { return centroid; }
    bool isPatch() const { return false; }
    bool isNoContraction() const { return false; }
    void setNoContraction() { }
    bool isPervertexNV() const { return false; }
    bool isPervertexEXT() const { return pervertexEXT; }
    void setNullInit() {}
    bool isNullInit() const { return false; }
    void setSpirvByReference() { }
    bool isSpirvByReference() { return false; }
    void setSpirvLiteral() { }
    bool isSpirvLiteral() { return false; }
#else
    bool noContraction: 1; // prevent contraction and reassociation, e.g., for 'precise' keyword, and expressions it affects
    bool nopersp      : 1;
    bool explicitInterp : 1;
    bool pervertexNV  : 1;
    bool pervertexEXT : 1;
    bool perPrimitiveNV : 1;
    bool perViewNV : 1;
    bool perTaskNV : 1;
    bool patch        : 1;
    bool sample       : 1;
    bool restrict     : 1;
    bool readonly     : 1;
    bool writeonly    : 1;
    bool coherent     : 1;
    bool volatil      : 1;
    bool devicecoherent : 1;
    bool queuefamilycoherent : 1;
    bool workgroupcoherent : 1;
    bool subgroupcoherent  : 1;
    bool shadercallcoherent : 1;
    bool nonprivate   : 1;
    bool nullInit : 1;
    bool spirvByReference : 1;
    bool spirvLiteral : 1;
    bool isWriteOnly() const { return writeonly; }
    bool isReadOnly() const { return readonly; }
    bool isRestrict() const { return restrict; }
    bool isCoherent() const { return coherent; }
    bool isVolatile() const { return volatil; }
    bool isSample() const { return sample; }
    bool isMemory() const
    {
        return shadercallcoherent || subgroupcoherent || workgroupcoherent || queuefamilycoherent || devicecoherent || coherent || volatil || restrict || readonly || writeonly || nonprivate;
    }
    bool isMemoryQualifierImageAndSSBOOnly() const
    {
        return shadercallcoherent || subgroupcoherent || workgroupcoherent || queuefamilycoherent || devicecoherent || coherent || volatil || restrict || readonly || writeonly;
    }
    bool bufferReferenceNeedsVulkanMemoryModel() const
    {
        // include qualifiers that map to load/store availability/visibility/nonprivate memory access operands
        return subgroupcoherent || workgroupcoherent || queuefamilycoherent || devicecoherent || coherent || nonprivate;
    }
    bool isInterpolation() const
    {
        return flat || smooth || nopersp || explicitInterp;
    }
    bool isExplicitInterpolation() const
    {
        return explicitInterp;
    }
    bool isAuxiliary() const
    {
        return centroid || patch || sample || pervertexNV || pervertexEXT;
    }
    bool isPatch() const { return patch; }
    bool isNoContraction() const { return noContraction; }
    void setNoContraction() { noContraction = true; }
    bool isPervertexNV() const { return pervertexNV; }
    bool isPervertexEXT() const { return pervertexEXT; }
    void setNullInit() { nullInit = true; }
    bool isNullInit() const { return nullInit; }
    void setSpirvByReference() { spirvByReference = true; }
    bool isSpirvByReference() const { return spirvByReference; }
    void setSpirvLiteral() { spirvLiteral = true; }
    bool isSpirvLiteral() const { return spirvLiteral; }
#endif

    bool isPipeInput() const
    {
        switch (storage) {
        case EvqVaryingIn:
        case EvqFragCoord:
        case EvqPointCoord:
        case EvqFace:
        case EvqVertexId:
        case EvqInstanceId:
            return true;
        default:
            return false;
        }
    }

    bool isPipeOutput() const
    {
        switch (storage) {
        case EvqPosition:
        case EvqPointSize:
        case EvqClipVertex:
        case EvqVaryingOut:
        case EvqFragColor:
        case EvqFragDepth:
        case EvqFragStencil:
            return true;
        default:
            return false;
        }
    }

    bool isParamInput() const
    {
        switch (storage) {
        case EvqIn:
        case EvqInOut:
        case EvqConstReadOnly:
            return true;
        default:
            return false;
        }
    }

    bool isParamOutput() const
    {
        switch (storage) {
        case EvqOut:
        case EvqInOut:
            return true;
        default:
            return false;
        }
    }

    bool isUniformOrBuffer() const
    {
        switch (storage) {
        case EvqUniform:
        case EvqBuffer:
            return true;
        default:
            return false;
        }
    }

    bool isUniform() const
    {
        switch (storage) {
        case EvqUniform:
            return true;
        default:
            return false;
        }
    }

    bool isIo() const
    {
        switch (storage) {
        case EvqUniform:
        case EvqBuffer:
        case EvqVaryingIn:
        case EvqFragCoord:
        case EvqPointCoord:
        case EvqFace:
        case EvqVertexId:
        case EvqInstanceId:
        case EvqPosition:
        case EvqPointSize:
        case EvqClipVertex:
        case EvqVaryingOut:
        case EvqFragColor:
        case EvqFragDepth:
        case EvqFragStencil:
            return true;
        default:
            return false;
        }
    }

    // non-built-in symbols that might link between compilation units
    bool isLinkable() const
    {
        switch (storage) {
        case EvqGlobal:
        case EvqVaryingIn:
        case EvqVaryingOut:
        case EvqUniform:
        case EvqBuffer:
        case EvqShared:
            return true;
        default:
            return false;
        }
    }

    TBlockStorageClass getBlockStorage() const {
        if (storage == EvqUniform && !isPushConstant()) {
            return EbsUniform;
        }
        else if (storage == EvqUniform) {
            return EbsPushConstant;
        }
        else if (storage == EvqBuffer) {
            return EbsStorageBuffer;
        }
        return EbsNone;
    }

    void setBlockStorage(TBlockStorageClass newBacking) {
#ifndef GLSLANG_WEB
        layoutPushConstant = (newBacking == EbsPushConstant);
#endif
        switch (newBacking) {
        case EbsUniform :
            if (layoutPacking == ElpStd430) {
                // std430 would not be valid
                layoutPacking = ElpStd140;
            }
            storage = EvqUniform;
            break;
        case EbsStorageBuffer : 
            storage = EvqBuffer;
            break;
#ifndef GLSLANG_WEB
        case EbsPushConstant :
            storage = EvqUniform;
            layoutSet = TQualifier::layoutSetEnd;
            layoutBinding = TQualifier::layoutBindingEnd;
            break;
#endif
        default:
            break;
        }
    }

#ifdef GLSLANG_WEB
    bool isPerView() const { return false; }
    bool isTaskMemory() const { return false; }
    bool isArrayedIo(EShLanguage language) const { return false; }
#else
    bool isPerPrimitive() const { return perPrimitiveNV; }
    bool isPerView() const { return perViewNV; }
    bool isTaskMemory() const { return perTaskNV; }
    bool isAnyPayload() const {
        return storage == EvqPayload || storage == EvqPayloadIn;
    }
    bool isAnyCallable() const {
        return storage == EvqCallableData || storage == EvqCallableDataIn;
    }

    // True if this type of IO is supposed to be arrayed with extra level for per-vertex data
    bool isArrayedIo(EShLanguage language) const
    {
        switch (language) {
        case EShLangGeometry:
            return isPipeInput();
        case EShLangTessControl:
            return ! patch && (isPipeInput() || isPipeOutput());
        case EShLangTessEvaluation:
            return ! patch && isPipeInput();
        case EShLangFragment:
            return (pervertexNV || pervertexEXT) && isPipeInput();
        case EShLangMeshNV:
            return ! perTaskNV && isPipeOutput();

        default:
            return false;
        }
    }
#endif

    // Implementing an embedded layout-qualifier class here, since C++ can't have a real class bitfield
    void clearLayout()  // all layout
    {
        clearUniformLayout();

#ifndef GLSLANG_WEB
        layoutPushConstant = false;
        layoutBufferReference = false;
        layoutPassthrough = false;
        layoutViewportRelative = false;
        // -2048 as the default value indicating layoutSecondaryViewportRelative is not set
        layoutSecondaryViewportRelativeOffset = -2048;
        layoutShaderRecord = false;
        layoutBufferReferenceAlign = layoutBufferReferenceAlignEnd;
        layoutFormat = ElfNone;
#endif

        clearInterstageLayout();

        layoutSpecConstantId = layoutSpecConstantIdEnd;
    }
    void clearInterstageLayout()
    {
        layoutLocation = layoutLocationEnd;
        layoutComponent = layoutComponentEnd;
#ifndef GLSLANG_WEB
        layoutIndex = layoutIndexEnd;
        clearStreamLayout();
        clearXfbLayout();
#endif
    }

#ifndef GLSLANG_WEB
    void clearStreamLayout()
    {
        layoutStream = layoutStreamEnd;
    }
    void clearXfbLayout()
    {
        layoutXfbBuffer = layoutXfbBufferEnd;
        layoutXfbStride = layoutXfbStrideEnd;
        layoutXfbOffset = layoutXfbOffsetEnd;
    }
#endif

    bool hasNonXfbLayout() const
    {
        return hasUniformLayout() ||
               hasAnyLocation() ||
               hasStream() ||
               hasFormat() ||
               isShaderRecord() ||
               isPushConstant() ||
               hasBufferReference();
    }
    bool hasLayout() const
    {
        return hasNonXfbLayout() ||
               hasXfb();
    }

    TLayoutMatrix  layoutMatrix  : 3;
    TLayoutPacking layoutPacking : 4;
    int layoutOffset;
    int layoutAlign;

                 unsigned int layoutLocation             : 12;
    static const unsigned int layoutLocationEnd      =  0xFFF;

                 unsigned int layoutComponent            :  3;
    static const unsigned int layoutComponentEnd      =     4;

                 unsigned int layoutSet                  :  7;
    static const unsigned int layoutSetEnd           =   0x3F;

                 unsigned int layoutBinding              : 16;
    static const unsigned int layoutBindingEnd      =  0xFFFF;

                 unsigned int layoutIndex                :  8;
    static const unsigned int layoutIndexEnd      =      0xFF;

                 unsigned int layoutStream               :  8;
    static const unsigned int layoutStreamEnd      =     0xFF;

                 unsigned int layoutXfbBuffer            :  4;
    static const unsigned int layoutXfbBufferEnd      =   0xF;

                 unsigned int layoutXfbStride            : 14;
    static const unsigned int layoutXfbStrideEnd     = 0x3FFF;

                 unsigned int layoutXfbOffset            : 13;
    static const unsigned int layoutXfbOffsetEnd     = 0x1FFF;

                 unsigned int layoutAttachment           :  8;  // for input_attachment_index
    static const unsigned int layoutAttachmentEnd      = 0XFF;

                 unsigned int layoutSpecConstantId       : 11;
    static const unsigned int layoutSpecConstantIdEnd = 0x7FF;

#ifndef GLSLANG_WEB
    // stored as log2 of the actual alignment value
                 unsigned int layoutBufferReferenceAlign :  6;
    static const unsigned int layoutBufferReferenceAlignEnd = 0x3F;

    TLayoutFormat layoutFormat                           :  8;

    bool layoutPushConstant;
    bool layoutBufferReference;
    bool layoutPassthrough;
    bool layoutViewportRelative;
    int layoutSecondaryViewportRelativeOffset;
    bool layoutShaderRecord;

    // GL_EXT_spirv_intrinsics
    int spirvStorageClass;
    TSpirvDecorate* spirvDecorate;
#endif

    bool hasUniformLayout() const
    {
        return hasMatrix() ||
               hasPacking() ||
               hasOffset() ||
               hasBinding() ||
               hasSet() ||
               hasAlign();
    }
    void clearUniformLayout() // only uniform specific
    {
        layoutMatrix = ElmNone;
        layoutPacking = ElpNone;
        layoutOffset = layoutNotSet;
        layoutAlign = layoutNotSet;

        layoutSet = layoutSetEnd;
        layoutBinding = layoutBindingEnd;
#ifndef GLSLANG_WEB
        layoutAttachment = layoutAttachmentEnd;
#endif
    }

    bool hasMatrix() const
    {
        return layoutMatrix != ElmNone;
    }
    bool hasPacking() const
    {
        return layoutPacking != ElpNone;
    }
    bool hasAlign() const
    {
        return layoutAlign != layoutNotSet;
    }
    bool hasAnyLocation() const
    {
        return hasLocation() ||
               hasComponent() ||
               hasIndex();
    }
    bool hasLocation() const
    {
        return layoutLocation != layoutLocationEnd;
    }
    bool hasSet() const
    {
        return layoutSet != layoutSetEnd;
    }
    bool hasBinding() const
    {
        return layoutBinding != layoutBindingEnd;
    }
#ifdef GLSLANG_WEB
    bool hasOffset() const { return false; }
    bool isNonPerspective() const { return false; }
    bool hasIndex() const { return false; }
    unsigned getIndex() const { return 0; }
    bool hasComponent() const { return false; }
    bool hasStream() const { return false; }
    bool hasFormat() const { return false; }
    bool hasXfb() const { return false; }
    bool hasXfbBuffer() const { return false; }
    bool hasXfbStride() const { return false; }
    bool hasXfbOffset() const { return false; }
    bool hasAttachment() const { return false; }
    TLayoutFormat getFormat() const { return ElfNone; }
    bool isPushConstant() const { return false; }
    bool isShaderRecord() const { return false; }
    bool hasBufferReference() const { return false; }
    bool hasBufferReferenceAlign() const { return false; }
    bool isNonUniform() const { return false; }
#else
    bool hasOffset() const
    {
        return layoutOffset != layoutNotSet;
    }
    bool isNonPerspective() const { return nopersp; }
    bool hasIndex() const
    {
        return layoutIndex != layoutIndexEnd;
    }
    unsigned getIndex() const { return layoutIndex; }
    bool hasComponent() const
    {
        return layoutComponent != layoutComponentEnd;
    }
    bool hasStream() const
    {
        return layoutStream != layoutStreamEnd;
    }
    bool hasFormat() const
    {
        return layoutFormat != ElfNone;
    }
    bool hasXfb() const
    {
        return hasXfbBuffer() ||
               hasXfbStride() ||
               hasXfbOffset();
    }
    bool hasXfbBuffer() const
    {
        return layoutXfbBuffer != layoutXfbBufferEnd;
    }
    bool hasXfbStride() const
    {
        return layoutXfbStride != layoutXfbStrideEnd;
    }
    bool hasXfbOffset() const
    {
        return layoutXfbOffset != layoutXfbOffsetEnd;
    }
    bool hasAttachment() const
    {
        return layoutAttachment != layoutAttachmentEnd;
    }
    TLayoutFormat getFormat() const { return layoutFormat; }
    bool isPushConstant() const { return layoutPushConstant; }
    bool isShaderRecord() const { return layoutShaderRecord; }
    bool hasBufferReference() const { return layoutBufferReference; }
    bool hasBufferReferenceAlign() const
    {
        return layoutBufferReferenceAlign != layoutBufferReferenceAlignEnd;
    }
    bool isNonUniform() const
    {
        return nonUniform;
    }

    // GL_EXT_spirv_intrinsics
    bool hasSprivDecorate() const { return spirvDecorate != nullptr; }
    void setSpirvDecorate(int decoration, const TIntermAggregate* args = nullptr);
    void setSpirvDecorateId(int decoration, const TIntermAggregate* args);
    void setSpirvDecorateString(int decoration, const TIntermAggregate* args);
    const TSpirvDecorate& getSpirvDecorate() const { assert(spirvDecorate); return *spirvDecorate; }
    TSpirvDecorate& getSpirvDecorate() { assert(spirvDecorate); return *spirvDecorate; }
    TString getSpirvDecorateQualifierString() const;
#endif
    bool hasSpecConstantId() const
    {
        // Not the same thing as being a specialization constant, this
        // is just whether or not it was declared with an ID.
        return layoutSpecConstantId != layoutSpecConstantIdEnd;
    }
    bool isSpecConstant() const
    {
        // True if type is a specialization constant, whether or not it
        // had a specialization-constant ID, and false if it is not a
        // true front-end constant.
        return specConstant;
    }
    bool isFrontEndConstant() const
    {
        // True if the front-end knows the final constant value.
        // This allows front-end constant folding.
        return storage == EvqConst && ! specConstant;
    }
    bool isConstant() const
    {
        // True if is either kind of constant; specialization or regular.
        return isFrontEndConstant() || isSpecConstant();
    }
    void makeSpecConstant()
    {
        storage = EvqConst;
        specConstant = true;
    }
    static const char* getLayoutPackingString(TLayoutPacking packing)
    {
        switch (packing) {
        case ElpStd140:   return "std140";
#ifndef GLSLANG_WEB
        case ElpPacked:   return "packed";
        case ElpShared:   return "shared";
        case ElpStd430:   return "std430";
        case ElpScalar:   return "scalar";
#endif
        default:          return "none";
        }
    }
    static const char* getLayoutMatrixString(TLayoutMatrix m)
    {
        switch (m) {
        case ElmColumnMajor: return "column_major";
        case ElmRowMajor:    return "row_major";
        default:             return "none";
        }
    }
#ifdef GLSLANG_WEB
    static const char* getLayoutFormatString(TLayoutFormat f) { return "none"; }
#else
    static const char* getLayoutFormatString(TLayoutFormat f)
    {
        switch (f) {
        case ElfRgba32f:      return "rgba32f";
        case ElfRgba16f:      return "rgba16f";
        case ElfRg32f:        return "rg32f";
        case ElfRg16f:        return "rg16f";
        case ElfR11fG11fB10f: return "r11f_g11f_b10f";
        case ElfR32f:         return "r32f";
        case ElfR16f:         return "r16f";
        case ElfRgba16:       return "rgba16";
        case ElfRgb10A2:      return "rgb10_a2";
        case ElfRgba8:        return "rgba8";
        case ElfRg16:         return "rg16";
        case ElfRg8:          return "rg8";
        case ElfR16:          return "r16";
        case ElfR8:           return "r8";
        case ElfRgba16Snorm:  return "rgba16_snorm";
        case ElfRgba8Snorm:   return "rgba8_snorm";
        case ElfRg16Snorm:    return "rg16_snorm";
        case ElfRg8Snorm:     return "rg8_snorm";
        case ElfR16Snorm:     return "r16_snorm";
        case ElfR8Snorm:      return "r8_snorm";

        case ElfRgba32i:      return "rgba32i";
        case ElfRgba16i:      return "rgba16i";
        case ElfRgba8i:       return "rgba8i";
        case ElfRg32i:        return "rg32i";
        case ElfRg16i:        return "rg16i";
        case ElfRg8i:         return "rg8i";
        case ElfR32i:         return "r32i";
        case ElfR16i:         return "r16i";
        case ElfR8i:          return "r8i";

        case ElfRgba32ui:     return "rgba32ui";
        case ElfRgba16ui:     return "rgba16ui";
        case ElfRgba8ui:      return "rgba8ui";
        case ElfRg32ui:       return "rg32ui";
        case ElfRg16ui:       return "rg16ui";
        case ElfRgb10a2ui:    return "rgb10_a2ui";
        case ElfRg8ui:        return "rg8ui";
        case ElfR32ui:        return "r32ui";
        case ElfR16ui:        return "r16ui";
        case ElfR8ui:         return "r8ui";
        case ElfR64ui:        return "r64ui";
        case ElfR64i:         return "r64i";
        default:              return "none";
        }
    }
    static const char* getLayoutDepthString(TLayoutDepth d)
    {
        switch (d) {
        case EldAny:       return "depth_any";
        case EldGreater:   return "depth_greater";
        case EldLess:      return "depth_less";
        case EldUnchanged: return "depth_unchanged";
        default:           return "none";
        }
    }
    static const char* getLayoutStencilString(TLayoutStencil s)
    {
        switch (s) {
        case ElsRefUnchangedFrontAMD: return "stencil_ref_unchanged_front_amd";
        case ElsRefGreaterFrontAMD:   return "stencil_ref_greater_front_amd";
        case ElsRefLessFrontAMD:      return "stencil_ref_less_front_amd";
        case ElsRefUnchangedBackAMD:  return "stencil_ref_unchanged_back_amd";
        case ElsRefGreaterBackAMD:    return "stencil_ref_greater_back_amd";
        case ElsRefLessBackAMD:       return "stencil_ref_less_back_amd";
        default:                      return "none";
        }
    }
    static const char* getBlendEquationString(TBlendEquationShift e)
    {
        switch (e) {
        case EBlendMultiply:      return "blend_support_multiply";
        case EBlendScreen:        return "blend_support_screen";
        case EBlendOverlay:       return "blend_support_overlay";
        case EBlendDarken:        return "blend_support_darken";
        case EBlendLighten:       return "blend_support_lighten";
        case EBlendColordodge:    return "blend_support_colordodge";
        case EBlendColorburn:     return "blend_support_colorburn";
        case EBlendHardlight:     return "blend_support_hardlight";
        case EBlendSoftlight:     return "blend_support_softlight";
        case EBlendDifference:    return "blend_support_difference";
        case EBlendExclusion:     return "blend_support_exclusion";
        case EBlendHslHue:        return "blend_support_hsl_hue";
        case EBlendHslSaturation: return "blend_support_hsl_saturation";
        case EBlendHslColor:      return "blend_support_hsl_color";
        case EBlendHslLuminosity: return "blend_support_hsl_luminosity";
        case EBlendAllEquations:  return "blend_support_all_equations";
        default:                  return "unknown";
        }
    }
    static const char* getGeometryString(TLayoutGeometry geometry)
    {
        switch (geometry) {
        case ElgPoints:             return "points";
        case ElgLines:              return "lines";
        case ElgLinesAdjacency:     return "lines_adjacency";
        case ElgLineStrip:          return "line_strip";
        case ElgTriangles:          return "triangles";
        case ElgTrianglesAdjacency: return "triangles_adjacency";
        case ElgTriangleStrip:      return "triangle_strip";
        case ElgQuads:              return "quads";
        case ElgIsolines:           return "isolines";
        default:                    return "none";
        }
    }
    static const char* getVertexSpacingString(TVertexSpacing spacing)
    {
        switch (spacing) {
        case EvsEqual:              return "equal_spacing";
        case EvsFractionalEven:     return "fractional_even_spacing";
        case EvsFractionalOdd:      return "fractional_odd_spacing";
        default:                    return "none";
        }
    }
    static const char* getVertexOrderString(TVertexOrder order)
    {
        switch (order) {
        case EvoCw:                 return "cw";
        case EvoCcw:                return "ccw";
        default:                    return "none";
        }
    }
    static int mapGeometryToSize(TLayoutGeometry geometry)
    {
        switch (geometry) {
        case ElgPoints:             return 1;
        case ElgLines:              return 2;
        case ElgLinesAdjacency:     return 4;
        case ElgTriangles:          return 3;
        case ElgTrianglesAdjacency: return 6;
        default:                    return 0;
        }
    }
    static const char* getInterlockOrderingString(TInterlockOrdering order)
    {
        switch (order) {
        case EioPixelInterlockOrdered:          return "pixel_interlock_ordered";
        case EioPixelInterlockUnordered:        return "pixel_interlock_unordered";
        case EioSampleInterlockOrdered:         return "sample_interlock_ordered";
        case EioSampleInterlockUnordered:       return "sample_interlock_unordered";
        case EioShadingRateInterlockOrdered:    return "shading_rate_interlock_ordered";
        case EioShadingRateInterlockUnordered:  return "shading_rate_interlock_unordered";
        default:                                return "none";
        }
    }
#endif
};

// Qualifiers that don't need to be keep per object.  They have shader scope, not object scope.
// So, they will not be part of TType, TQualifier, etc.
struct TShaderQualifiers {
    TLayoutGeometry geometry; // geometry/tessellation shader in/out primitives
    bool pixelCenterInteger;  // fragment shader
    bool originUpperLeft;     // fragment shader
    int invocations;
    int vertices;             // for tessellation "vertices", geometry & mesh "max_vertices"
    TVertexSpacing spacing;
    TVertexOrder order;
    bool pointMode;
    int localSize[3];         // compute shader
    bool localSizeNotDefault[3];        // compute shader
    int localSizeSpecId[3];   // compute shader specialization id for gl_WorkGroupSize
#ifndef GLSLANG_WEB
    bool earlyFragmentTests;  // fragment input
    bool postDepthCoverage;   // fragment input
    bool earlyAndLateFragmentTestsAMD; //fragment input
    TLayoutDepth layoutDepth;
    TLayoutStencil layoutStencil;
    bool blendEquation;       // true if any blend equation was specified
    int numViews;             // multiview extenstions
    TInterlockOrdering interlockOrdering;
    bool layoutOverrideCoverage;        // true if layout override_coverage set
    bool layoutDerivativeGroupQuads;    // true if layout derivative_group_quadsNV set
    bool layoutDerivativeGroupLinear;   // true if layout derivative_group_linearNV set
    int primitives;                     // mesh shader "max_primitives"DerivativeGroupLinear;   // true if layout derivative_group_linearNV set
    bool layoutPrimitiveCulling;        // true if layout primitive_culling set
    TLayoutDepth getDepth() const { return layoutDepth; }
    TLayoutStencil getStencil() const { return layoutStencil; }
#else
    TLayoutDepth getDepth() const { return EldNone; }
#endif

    void init()
    {
        geometry = ElgNone;
        originUpperLeft = false;
        pixelCenterInteger = false;
        invocations = TQualifier::layoutNotSet;
        vertices = TQualifier::layoutNotSet;
        spacing = EvsNone;
        order = EvoNone;
        pointMode = false;
        localSize[0] = 1;
        localSize[1] = 1;
        localSize[2] = 1;
        localSizeNotDefault[0] = false;
        localSizeNotDefault[1] = false;
        localSizeNotDefault[2] = false;
        localSizeSpecId[0] = TQualifier::layoutNotSet;
        localSizeSpecId[1] = TQualifier::layoutNotSet;
        localSizeSpecId[2] = TQualifier::layoutNotSet;
#ifndef GLSLANG_WEB
        earlyFragmentTests = false;
        earlyAndLateFragmentTestsAMD = false;
        postDepthCoverage = false;
        layoutDepth = EldNone;
        layoutStencil = ElsNone;
        blendEquation = false;
        numViews = TQualifier::layoutNotSet;
        layoutOverrideCoverage      = false;
        layoutDerivativeGroupQuads  = false;
        layoutDerivativeGroupLinear = false;
        layoutPrimitiveCulling      = false;
        primitives                  = TQualifier::layoutNotSet;
        interlockOrdering = EioNone;
#endif
    }

#ifdef GLSLANG_WEB
    bool hasBlendEquation() const { return false; }
#else
    bool hasBlendEquation() const { return blendEquation; }
#endif

    // Merge in characteristics from the 'src' qualifier.  They can override when
    // set, but never erase when not set.
    void merge(const TShaderQualifiers& src)
    {
        if (src.geometry != ElgNone)
            geometry = src.geometry;
        if (src.pixelCenterInteger)
            pixelCenterInteger = src.pixelCenterInteger;
        if (src.originUpperLeft)
            originUpperLeft = src.originUpperLeft;
        if (src.invocations != TQualifier::layoutNotSet)
            invocations = src.invocations;
        if (src.vertices != TQualifier::layoutNotSet)
            vertices = src.vertices;
        if (src.spacing != EvsNone)
            spacing = src.spacing;
        if (src.order != EvoNone)
            order = src.order;
        if (src.pointMode)
            pointMode = true;
        for (int i = 0; i < 3; ++i) {
            if (src.localSize[i] > 1)
                localSize[i] = src.localSize[i];
        }
        for (int i = 0; i < 3; ++i) {
            localSizeNotDefault[i] = src.localSizeNotDefault[i] || localSizeNotDefault[i];
        }
        for (int i = 0; i < 3; ++i) {
            if (src.localSizeSpecId[i] != TQualifier::layoutNotSet)
                localSizeSpecId[i] = src.localSizeSpecId[i];
        }
#ifndef GLSLANG_WEB
        if (src.earlyFragmentTests)
            earlyFragmentTests = true;
        if (src.earlyAndLateFragmentTestsAMD)
            earlyAndLateFragmentTestsAMD = true;
        if (src.postDepthCoverage)
            postDepthCoverage = true;
        if (src.layoutDepth)
            layoutDepth = src.layoutDepth;
        if (src.layoutStencil)
            layoutStencil = src.layoutStencil;
        if (src.blendEquation)
            blendEquation = src.blendEquation;
        if (src.numViews != TQualifier::layoutNotSet)
            numViews = src.numViews;
        if (src.layoutOverrideCoverage)
            layoutOverrideCoverage = src.layoutOverrideCoverage;
        if (src.layoutDerivativeGroupQuads)
            layoutDerivativeGroupQuads = src.layoutDerivativeGroupQuads;
        if (src.layoutDerivativeGroupLinear)
            layoutDerivativeGroupLinear = src.layoutDerivativeGroupLinear;
        if (src.primitives != TQualifier::layoutNotSet)
            primitives = src.primitives;
        if (src.interlockOrdering != EioNone)
            interlockOrdering = src.interlockOrdering;
        if (src.layoutPrimitiveCulling)
            layoutPrimitiveCulling = src.layoutPrimitiveCulling;
#endif
    }
};

//
// TPublicType is just temporarily used while parsing and not quite the same
// information kept per node in TType.  Due to the bison stack, it can't have
// types that it thinks have non-trivial constructors.  It should
// just be used while recognizing the grammar, not anything else.
// Once enough is known about the situation, the proper information
// moved into a TType, or the parse context, etc.
//
class TPublicType {
public:
    TBasicType basicType;
    TSampler sampler;
    TQualifier qualifier;
    TShaderQualifiers shaderQualifiers;
    int vectorSize : 4;
    int matrixCols : 4;
    int matrixRows : 4;
    bool coopmat   : 1;
    TArraySizes* arraySizes;
    const TType* userDef;
    TSourceLoc loc;
    TArraySizes* typeParameters;
#ifndef GLSLANG_WEB
    // SPIR-V type defined by spirv_type directive
    TSpirvType* spirvType;
#endif

#ifdef GLSLANG_WEB
    bool isCoopmat() const { return false; }
#else
    bool isCoopmat() const { return coopmat; }
#endif

    void initType(const TSourceLoc& l)
    {
        basicType = EbtVoid;
        vectorSize = 1;
        matrixRows = 0;
        matrixCols = 0;
        arraySizes = nullptr;
        userDef = nullptr;
        loc = l;
        typeParameters = nullptr;
        coopmat = false;
#ifndef GLSLANG_WEB
        spirvType = nullptr;
#endif
    }

    void initQualifiers(bool global = false)
    {
        qualifier.clear();
        if (global)
            qualifier.storage = EvqGlobal;
    }

    void init(const TSourceLoc& l, bool global = false)
    {
        initType(l);
        sampler.clear();
        initQualifiers(global);
        shaderQualifiers.init();
    }

    void setVector(int s)
    {
        matrixRows = 0;
        matrixCols = 0;
        vectorSize = s;
    }

    void setMatrix(int c, int r)
    {
        matrixRows = r;
        matrixCols = c;
        vectorSize = 0;
    }

    bool isScalar() const
    {
        return matrixCols == 0 && vectorSize == 1 && arraySizes == nullptr && userDef == nullptr;
    }

#ifndef GLSLANG_WEB
    // GL_EXT_spirv_intrinsics
    void setSpirvType(const TSpirvInstruction& spirvInst, const TSpirvTypeParameters* typeParams = nullptr);
#endif

    // "Image" is a superset of "Subpass"
    bool isImage()   const { return basicType == EbtSampler && sampler.isImage(); }
    bool isSubpass() const { return basicType == EbtSampler && sampler.isSubpass(); }
};

//
// Base class for things that have a type.
//
class TType {
public:
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    // for "empty" type (no args) or simple scalar/vector/matrix
    explicit TType(TBasicType t = EbtVoid, TStorageQualifier q = EvqTemporary, int vs = 1, int mc = 0, int mr = 0,
                   bool isVector = false) :
                            basicType(t), vectorSize(vs), matrixCols(mc), matrixRows(mr), vector1(isVector && vs == 1), coopmat(false),
                            arraySizes(nullptr), structure(nullptr), fieldName(nullptr), typeName(nullptr), typeParameters(nullptr)
#ifndef GLSLANG_WEB
                            , spirvType(nullptr)
#endif
                            {
                                sampler.clear();
                                qualifier.clear();
                                qualifier.storage = q;
                                assert(!(isMatrix() && vectorSize != 0));  // prevent vectorSize != 0 on matrices
                            }
    // for explicit precision qualifier
    TType(TBasicType t, TStorageQualifier q, TPrecisionQualifier p, int vs = 1, int mc = 0, int mr = 0,
          bool isVector = false) :
                            basicType(t), vectorSize(vs), matrixCols(mc), matrixRows(mr), vector1(isVector && vs == 1), coopmat(false),
                            arraySizes(nullptr), structure(nullptr), fieldName(nullptr), typeName(nullptr), typeParameters(nullptr)
#ifndef GLSLANG_WEB
                            , spirvType(nullptr)
#endif
                            {
                                sampler.clear();
                                qualifier.clear();
                                qualifier.storage = q;
                                qualifier.precision = p;
                                assert(p >= EpqNone && p <= EpqHigh);
                                assert(!(isMatrix() && vectorSize != 0));  // prevent vectorSize != 0 on matrices
                            }
    // for turning a TPublicType into a TType, using a shallow copy
    explicit TType(const TPublicType& p) :
                            basicType(p.basicType),
                            vectorSize(p.vectorSize), matrixCols(p.matrixCols), matrixRows(p.matrixRows), vector1(false), coopmat(p.coopmat),
                            arraySizes(p.arraySizes), structure(nullptr), fieldName(nullptr), typeName(nullptr), typeParameters(p.typeParameters)
#ifndef GLSLANG_WEB
                            , spirvType(p.spirvType)
#endif
                            {
                                if (basicType == EbtSampler)
                                    sampler = p.sampler;
                                else
                                    sampler.clear();
                                qualifier = p.qualifier;
                                if (p.userDef) {
                                    if (p.userDef->basicType == EbtReference) {
                                        basicType = EbtReference;
                                        referentType = p.userDef->referentType;
                                    } else {
                                        structure = p.userDef->getWritableStruct();  // public type is short-lived; there are no sharing issues
                                    }
                                    typeName = NewPoolTString(p.userDef->getTypeName().c_str());
                                }
                                if (p.isCoopmat() && p.typeParameters && p.typeParameters->getNumDims() > 0) {
                                    int numBits = p.typeParameters->getDimSize(0);
                                    if (p.basicType == EbtFloat && numBits == 16) {
                                        basicType = EbtFloat16;
                                        qualifier.precision = EpqNone;
                                    } else if (p.basicType == EbtUint && numBits == 8) {
                                        basicType = EbtUint8;
                                        qualifier.precision = EpqNone;
                                    } else if (p.basicType == EbtInt && numBits == 8) {
                                        basicType = EbtInt8;
                                        qualifier.precision = EpqNone;
                                    }
                                }
                            }
    // for construction of sampler types
    TType(const TSampler& sampler, TStorageQualifier q = EvqUniform, TArraySizes* as = nullptr) :
        basicType(EbtSampler), vectorSize(1), matrixCols(0), matrixRows(0), vector1(false), coopmat(false),
        arraySizes(as), structure(nullptr), fieldName(nullptr), typeName(nullptr),
        sampler(sampler), typeParameters(nullptr)
#ifndef GLSLANG_WEB
        , spirvType(nullptr)
#endif
    {
        qualifier.clear();
        qualifier.storage = q;
    }
    // to efficiently make a dereferenced type
    // without ever duplicating the outer structure that will be thrown away
    // and using only shallow copy
    TType(const TType& type, int derefIndex, bool rowMajor = false)
                            {
                                if (type.isArray()) {
                                    shallowCopy(type);
                                    if (type.getArraySizes()->getNumDims() == 1) {
                                        arraySizes = nullptr;
                                    } else {
                                        // want our own copy of the array, so we can edit it
                                        arraySizes = new TArraySizes;
                                        arraySizes->copyDereferenced(*type.arraySizes);
                                    }
                                } else if (type.basicType == EbtStruct || type.basicType == EbtBlock) {
                                    // do a structure dereference
                                    const TTypeList& memberList = *type.getStruct();
                                    shallowCopy(*memberList[derefIndex].type);
                                    return;
                                } else {
                                    // do a vector/matrix dereference
                                    shallowCopy(type);
                                    if (matrixCols > 0) {
                                        // dereference from matrix to vector
                                        if (rowMajor)
                                            vectorSize = matrixCols;
                                        else
                                            vectorSize = matrixRows;
                                        matrixCols = 0;
                                        matrixRows = 0;
                                        if (vectorSize == 1)
                                            vector1 = true;
                                    } else if (isVector()) {
                                        // dereference from vector to scalar
                                        vectorSize = 1;
                                        vector1 = false;
                                    } else if (isCoopMat()) {
                                        coopmat = false;
                                        typeParameters = nullptr;
                                    }
                                }
                            }
    // for making structures, ...
    TType(TTypeList* userDef, const TString& n) :
                            basicType(EbtStruct), vectorSize(1), matrixCols(0), matrixRows(0), vector1(false), coopmat(false),
                            arraySizes(nullptr), structure(userDef), fieldName(nullptr), typeParameters(nullptr)
#ifndef GLSLANG_WEB
                            , spirvType(nullptr)
#endif
                            {
                                sampler.clear();
                                qualifier.clear();
                                typeName = NewPoolTString(n.c_str());
                            }
    // For interface blocks
    TType(TTypeList* userDef, const TString& n, const TQualifier& q) :
                            basicType(EbtBlock), vectorSize(1), matrixCols(0), matrixRows(0), vector1(false), coopmat(false),
                            qualifier(q), arraySizes(nullptr), structure(userDef), fieldName(nullptr), typeParameters(nullptr)
#ifndef GLSLANG_WEB
                            , spirvType(nullptr)
#endif
                            {
                                sampler.clear();
                                typeName = NewPoolTString(n.c_str());
                            }
    // for block reference (first parameter must be EbtReference)
    explicit TType(TBasicType t, const TType &p, const TString& n) :
                            basicType(t), vectorSize(1), matrixCols(0), matrixRows(0), vector1(false),
                            arraySizes(nullptr), structure(nullptr), fieldName(nullptr), typeName(nullptr)
#ifndef GLSLANG_WEB
                            , spirvType(nullptr)
#endif
                            {
                                assert(t == EbtReference);
                                typeName = NewPoolTString(n.c_str());
                                qualifier.clear();
                                qualifier.storage = p.qualifier.storage;
                                referentType = p.clone();
                            }
    virtual ~TType() {}

    // Not for use across pool pops; it will cause multiple instances of TType to point to the same information.
    // This only works if that information (like a structure's list of types) does not change and
    // the instances are sharing the same pool.
    void shallowCopy(const TType& copyOf)
    {
        basicType = copyOf.basicType;
        sampler = copyOf.sampler;
        qualifier = copyOf.qualifier;
        vectorSize = copyOf.vectorSize;
        matrixCols = copyOf.matrixCols;
        matrixRows = copyOf.matrixRows;
        vector1 = copyOf.vector1;
        arraySizes = copyOf.arraySizes;  // copying the pointer only, not the contents
        fieldName = copyOf.fieldName;
        typeName = copyOf.typeName;
        if (isStruct()) {
            structure = copyOf.structure;
        } else {
            referentType = copyOf.referentType;
        }
        typeParameters = copyOf.typeParameters;
#ifndef GLSLANG_WEB
        spirvType = copyOf.spirvType;
#endif
        coopmat = copyOf.isCoopMat();
    }

    // Make complete copy of the whole type graph rooted at 'copyOf'.
    void deepCopy(const TType& copyOf)
    {
        TMap<TTypeList*,TTypeList*> copied;  // to enable copying a type graph as a graph, not a tree
        deepCopy(copyOf, copied);
    }

    // Recursively make temporary
    void makeTemporary()
    {
        getQualifier().makeTemporary();

        if (isStruct())
            for (unsigned int i = 0; i < structure->size(); ++i)
                (*structure)[i].type->makeTemporary();
    }

    TType* clone() const
    {
        TType *newType = new TType();
        newType->deepCopy(*this);

        return newType;
    }

    void makeVector() { vector1 = true; }

    virtual void hideMember() { basicType = EbtVoid; vectorSize = 1; }
    virtual bool hiddenMember() const { return basicType == EbtVoid; }

    virtual void setFieldName(const TString& n) { fieldName = NewPoolTString(n.c_str()); }
    virtual const TString& getTypeName() const
    {
        assert(typeName);
        return *typeName;
    }

    virtual const TString& getFieldName() const
    {
        assert(fieldName);
        return *fieldName;
    }
    TShaderInterface getShaderInterface() const
    {
        if (basicType != EbtBlock)
            return EsiNone;

        switch (qualifier.storage) {
        default:
            return EsiNone;
        case EvqVaryingIn:
            return EsiInput;
        case EvqVaryingOut:
            return EsiOutput;
        case EvqUniform:
        case EvqBuffer:
            return EsiUniform;
        }
    }

    virtual TBasicType getBasicType() const { return basicType; }
    virtual const TSampler& getSampler() const { return sampler; }
    virtual TSampler& getSampler() { return sampler; }

    virtual       TQualifier& getQualifier()       { return qualifier; }
    virtual const TQualifier& getQualifier() const { return qualifier; }

    virtual int getVectorSize() const { return vectorSize; }  // returns 1 for either scalar or vector of size 1, valid for both
    virtual int getMatrixCols() const { return matrixCols; }
    virtual int getMatrixRows() const { return matrixRows; }
    virtual int getOuterArraySize()  const { return arraySizes->getOuterSize(); }
    virtual TIntermTyped*  getOuterArrayNode() const { return arraySizes->getOuterNode(); }
    virtual int getCumulativeArraySize()  const { return arraySizes->getCumulativeSize(); }
#ifdef GLSLANG_WEB
    bool isArrayOfArrays() const { return false; }
#else
    bool isArrayOfArrays() const { return arraySizes != nullptr && arraySizes->getNumDims() > 1; }
#endif
    virtual int getImplicitArraySize() const { return arraySizes->getImplicitSize(); }
    virtual const TArraySizes* getArraySizes() const { return arraySizes; }
    virtual       TArraySizes* getArraySizes()       { return arraySizes; }
    virtual TType* getReferentType() const { return referentType; }
    virtual const TArraySizes* getTypeParameters() const { return typeParameters; }
    virtual       TArraySizes* getTypeParameters()       { return typeParameters; }

    virtual bool isScalar() const { return ! isVector() && ! isMatrix() && ! isStruct() && ! isArray(); }
    virtual bool isScalarOrVec1() const { return isScalar() || vector1; }
    virtual bool isScalarOrVector() const { return !isMatrix() && !isStruct() && !isArray(); }
    virtual bool isVector() const { return vectorSize > 1 || vector1; }
    virtual bool isMatrix() const { return matrixCols ? true : false; }
    virtual bool isArray()  const { return arraySizes != nullptr; }
    virtual bool isSizedArray() const { return isArray() && arraySizes->isSized(); }
    virtual bool isUnsizedArray() const { return isArray() && !arraySizes->isSized(); }
    virtual bool isArrayVariablyIndexed() const { assert(isArray()); return arraySizes->isVariablyIndexed(); }
    virtual void setArrayVariablyIndexed() { assert(isArray()); arraySizes->setVariablyIndexed(); }
    virtual void updateImplicitArraySize(int size) { assert(isArray()); arraySizes->updateImplicitSize(size); }
    virtual bool isStruct() const { return basicType == EbtStruct || basicType == EbtBlock; }
    virtual bool isFloatingDomain() const { return basicType == EbtFloat || basicType == EbtDouble || basicType == EbtFloat16; }
    virtual bool isIntegerDomain() const
    {
        switch (basicType) {
        case EbtInt8:
        case EbtUint8:
        case EbtInt16:
        case EbtUint16:
        case EbtInt:
        case EbtUint:
        case EbtInt64:
        case EbtUint64:
        case EbtAtomicUint:
            return true;
        default:
            break;
        }
        return false;
    }
    virtual bool isOpaque() const { return basicType == EbtSampler
#ifndef GLSLANG_WEB
            || basicType == EbtAtomicUint || basicType == EbtAccStruct || basicType == EbtRayQuery
#endif
        ; }
    virtual bool isBuiltIn() const { return getQualifier().builtIn != EbvNone; }

    // "Image" is a superset of "Subpass"
    virtual bool isImage()   const { return basicType == EbtSampler && getSampler().isImage(); }
    virtual bool isSubpass() const { return basicType == EbtSampler && getSampler().isSubpass(); }
    virtual bool isTexture() const { return basicType == EbtSampler && getSampler().isTexture(); }
    // Check the block-name convention of creating a block without populating it's members:
    virtual bool isUnusableName() const { return isStruct() && structure == nullptr; }
    virtual bool isParameterized()  const { return typeParameters != nullptr; }
#ifdef GLSLANG_WEB
    bool isAtomic() const { return false; }
    bool isCoopMat() const { return false; }
    bool isReference() const { return false; }
    bool isSpirvType() const { return false; }
#else
    bool isAtomic() const { return basicType == EbtAtomicUint; }
    bool isCoopMat() const { return coopmat; }
    bool isReference() const { return getBasicType() == EbtReference; }
    bool isSpirvType() const { return getBasicType() == EbtSpirvType; }
#endif

    // return true if this type contains any subtype which satisfies the given predicate.
    template <typename P>
    bool contains(P predicate) const
    {
        if (predicate(this))
            return true;

        const auto hasa = [predicate](const TTypeLoc& tl) { return tl.type->contains(predicate); };

        return isStruct() && std::any_of(structure->begin(), structure->end(), hasa);
    }

    // Recursively checks if the type contains the given basic type
    virtual bool containsBasicType(TBasicType checkType) const
    {
        return contains([checkType](const TType* t) { return t->basicType == checkType; } );
    }

    // Recursively check the structure for any arrays, needed for some error checks
    virtual bool containsArray() const
    {
        return contains([](const TType* t) { return t->isArray(); } );
    }

    // Check the structure for any structures, needed for some error checks
    virtual bool containsStructure() const
    {
        return contains([this](const TType* t) { return t != this && t->isStruct(); } );
    }

    // Recursively check the structure for any unsized arrays, needed for triggering a copyUp().
    virtual bool containsUnsizedArray() const
    {
        return contains([](const TType* t) { return t->isUnsizedArray(); } );
    }

    virtual bool containsOpaque() const
    {
        return contains([](const TType* t) { return t->isOpaque(); } );
    }

    // Recursively checks if the type contains a built-in variable
    virtual bool containsBuiltIn() const
    {
        return contains([](const TType* t) { return t->isBuiltIn(); } );
    }

    virtual bool containsNonOpaque() const
    {
        const auto nonOpaque = [](const TType* t) {
            switch (t->basicType) {
            case EbtVoid:
            case EbtFloat:
            case EbtDouble:
            case EbtFloat16:
            case EbtInt8:
            case EbtUint8:
            case EbtInt16:
            case EbtUint16:
            case EbtInt:
            case EbtUint:
            case EbtInt64:
            case EbtUint64:
            case EbtBool:
            case EbtReference:
                return true;
            default:
                return false;
            }
        };

        return contains(nonOpaque);
    }

    virtual bool containsSpecializationSize() const
    {
        return contains([](const TType* t) { return t->isArray() && t->arraySizes->isOuterSpecialization(); } );
    }

#ifdef GLSLANG_WEB
    bool containsDouble() const { return false; }
    bool contains16BitFloat() const { return false; }
    bool contains64BitInt() const { return false; }
    bool contains16BitInt() const { return false; }
    bool contains8BitInt() const { return false; }
    bool containsCoopMat() const { return false; }
    bool containsReference() const { return false; }
#else
    bool containsDouble() const
    {
        return containsBasicType(EbtDouble);
    }
    bool contains16BitFloat() const
    {
        return containsBasicType(EbtFloat16);
    }
    bool contains64BitInt() const
    {
        return containsBasicType(EbtInt64) || containsBasicType(EbtUint64);
    }
    bool contains16BitInt() const
    {
        return containsBasicType(EbtInt16) || containsBasicType(EbtUint16);
    }
    bool contains8BitInt() const
    {
        return containsBasicType(EbtInt8) || containsBasicType(EbtUint8);
    }
    bool containsCoopMat() const
    {
        return contains([](const TType* t) { return t->coopmat; } );
    }
    bool containsReference() const
    {
        return containsBasicType(EbtReference);
    }
#endif

    // Array editing methods.  Array descriptors can be shared across
    // type instances.  This allows all uses of the same array
    // to be updated at once.  E.g., all nodes can be explicitly sized
    // by tracking and correcting one implicit size.  Or, all nodes
    // can get the explicit size on a redeclaration that gives size.
    //
    // N.B.:  Don't share with the shared symbol tables (symbols are
    // marked as isReadOnly().  Such symbols with arrays that will be
    // edited need to copyUp() on first use, so that
    // A) the edits don't effect the shared symbol table, and
    // B) the edits are shared across all users.
    void updateArraySizes(const TType& type)
    {
        // For when we may already be sharing existing array descriptors,
        // keeping the pointers the same, just updating the contents.
        assert(arraySizes != nullptr);
        assert(type.arraySizes != nullptr);
        *arraySizes = *type.arraySizes;
    }
    void copyArraySizes(const TArraySizes& s)
    {
        // For setting a fresh new set of array sizes, not yet worrying about sharing.
        arraySizes = new TArraySizes;
        *arraySizes = s;
    }
    void transferArraySizes(TArraySizes* s)
    {
        // For setting an already allocated set of sizes that this type can use
        // (no copy made).
        arraySizes = s;
    }
    void clearArraySizes()
    {
        arraySizes = nullptr;
    }

    // Add inner array sizes, to any existing sizes, via copy; the
    // sizes passed in can still be reused for other purposes.
    void copyArrayInnerSizes(const TArraySizes* s)
    {
        if (s != nullptr) {
            if (arraySizes == nullptr)
                copyArraySizes(*s);
            else
                arraySizes->addInnerSizes(*s);
        }
    }
    void changeOuterArraySize(int s) { arraySizes->changeOuterSize(s); }

    // Recursively make the implicit array size the explicit array size.
    // Expicit arrays are compile-time or link-time sized, never run-time sized.
    // Sometimes, policy calls for an array to be run-time sized even if it was
    // never variably indexed: Don't turn a 'skipNonvariablyIndexed' array into
    // an explicit array.
    void adoptImplicitArraySizes(bool skipNonvariablyIndexed)
    {
        if (isUnsizedArray() && !(skipNonvariablyIndexed || isArrayVariablyIndexed()))
            changeOuterArraySize(getImplicitArraySize());
        // For multi-dim per-view arrays, set unsized inner dimension size to 1
        if (qualifier.isPerView() && arraySizes && arraySizes->isInnerUnsized())
            arraySizes->clearInnerUnsized();
        if (isStruct() && structure->size() > 0) {
            int lastMember = (int)structure->size() - 1;
            for (int i = 0; i < lastMember; ++i)
                (*structure)[i].type->adoptImplicitArraySizes(false);
            // implement the "last member of an SSBO" policy
            (*structure)[lastMember].type->adoptImplicitArraySizes(getQualifier().storage == EvqBuffer);
        }
    }


    void updateTypeParameters(const TType& type)
    {
        // For when we may already be sharing existing array descriptors,
        // keeping the pointers the same, just updating the contents.
        assert(typeParameters != nullptr);
        assert(type.typeParameters != nullptr);
        *typeParameters = *type.typeParameters;
    }
    void copyTypeParameters(const TArraySizes& s)
    {
        // For setting a fresh new set of type parameters, not yet worrying about sharing.
        typeParameters = new TArraySizes;
        *typeParameters = s;
    }
    void transferTypeParameters(TArraySizes* s)
    {
        // For setting an already allocated set of sizes that this type can use
        // (no copy made).
        typeParameters = s;
    }
    void clearTypeParameters()
    {
        typeParameters = nullptr;
    }

    // Add inner array sizes, to any existing sizes, via copy; the
    // sizes passed in can still be reused for other purposes.
    void copyTypeParametersInnerSizes(const TArraySizes* s)
    {
        if (s != nullptr) {
            if (typeParameters == nullptr)
                copyTypeParameters(*s);
            else
                typeParameters->addInnerSizes(*s);
        }
    }

    const char* getBasicString() const
    {
        return TType::getBasicString(basicType);
    }

    static const char* getBasicString(TBasicType t)
    {
        switch (t) {
        case EbtFloat:             return "float";
        case EbtInt:               return "int";
        case EbtUint:              return "uint";
        case EbtSampler:           return "sampler/image";
#ifndef GLSLANG_WEB
        case EbtVoid:              return "void";
        case EbtDouble:            return "double";
        case EbtFloat16:           return "float16_t";
        case EbtInt8:              return "int8_t";
        case EbtUint8:             return "uint8_t";
        case EbtInt16:             return "int16_t";
        case EbtUint16:            return "uint16_t";
        case EbtInt64:             return "int64_t";
        case EbtUint64:            return "uint64_t";
        case EbtBool:              return "bool";
        case EbtAtomicUint:        return "atomic_uint";
        case EbtStruct:            return "structure";
        case EbtBlock:             return "block";
        case EbtAccStruct:         return "accelerationStructureNV";
        case EbtRayQuery:          return "rayQueryEXT";
        case EbtReference:         return "reference";
        case EbtString:            return "string";
        case EbtSpirvType:         return "spirv_type";
#endif
        default:                   return "unknown type";
        }
    }

#ifdef GLSLANG_WEB
    TString getCompleteString() const { return ""; }
    const char* getStorageQualifierString() const { return ""; }
    const char* getBuiltInVariableString() const { return ""; }
    const char* getPrecisionQualifierString() const { return ""; }
    TString getBasicTypeString() const { return ""; }
#else
    TString getCompleteString(bool syntactic = false, bool getQualifiers = true, bool getPrecision = true,
                              bool getType = true, TString name = "", TString structName = "") const
    {
        TString typeString;

        const auto appendStr  = [&](const char* s)  { typeString.append(s); };
        const auto appendUint = [&](unsigned int u) { typeString.append(std::to_string(u).c_str()); };
        const auto appendInt  = [&](int i)          { typeString.append(std::to_string(i).c_str()); };

        if (getQualifiers) {
          if (qualifier.hasSprivDecorate())
            appendStr(qualifier.getSpirvDecorateQualifierString().c_str());

          if (qualifier.hasLayout()) {
            // To reduce noise, skip this if the only layout is an xfb_buffer
            // with no triggering xfb_offset.
            TQualifier noXfbBuffer = qualifier;
            noXfbBuffer.layoutXfbBuffer = TQualifier::layoutXfbBufferEnd;
            if (noXfbBuffer.hasLayout()) {
              appendStr("layout(");
              if (qualifier.hasAnyLocation()) {
                appendStr(" location=");
                appendUint(qualifier.layoutLocation);
                if (qualifier.hasComponent()) {
                  appendStr(" component=");
                  appendUint(qualifier.layoutComponent);
                }
                if (qualifier.hasIndex()) {
                  appendStr(" index=");
                  appendUint(qualifier.layoutIndex);
                }
              }
              if (qualifier.hasSet()) {
                appendStr(" set=");
                appendUint(qualifier.layoutSet);
              }
              if (qualifier.hasBinding()) {
                appendStr(" binding=");
                appendUint(qualifier.layoutBinding);
              }
              if (qualifier.hasStream()) {
                appendStr(" stream=");
                appendUint(qualifier.layoutStream);
              }
              if (qualifier.hasMatrix()) {
                appendStr(" ");
                appendStr(TQualifier::getLayoutMatrixString(qualifier.layoutMatrix));
              }
              if (qualifier.hasPacking()) {
                appendStr(" ");
                appendStr(TQualifier::getLayoutPackingString(qualifier.layoutPacking));
              }
              if (qualifier.hasOffset()) {
                appendStr(" offset=");
                appendInt(qualifier.layoutOffset);
              }
              if (qualifier.hasAlign()) {
                appendStr(" align=");
                appendInt(qualifier.layoutAlign);
              }
              if (qualifier.hasFormat()) {
                appendStr(" ");
                appendStr(TQualifier::getLayoutFormatString(qualifier.layoutFormat));
              }
              if (qualifier.hasXfbBuffer() && qualifier.hasXfbOffset()) {
                appendStr(" xfb_buffer=");
                appendUint(qualifier.layoutXfbBuffer);
              }
              if (qualifier.hasXfbOffset()) {
                appendStr(" xfb_offset=");
                appendUint(qualifier.layoutXfbOffset);
              }
              if (qualifier.hasXfbStride()) {
                appendStr(" xfb_stride=");
                appendUint(qualifier.layoutXfbStride);
              }
              if (qualifier.hasAttachment()) {
                appendStr(" input_attachment_index=");
                appendUint(qualifier.layoutAttachment);
              }
              if (qualifier.hasSpecConstantId()) {
                appendStr(" constant_id=");
                appendUint(qualifier.layoutSpecConstantId);
              }
              if (qualifier.layoutPushConstant)
                appendStr(" push_constant");
              if (qualifier.layoutBufferReference)
                appendStr(" buffer_reference");
              if (qualifier.hasBufferReferenceAlign()) {
                appendStr(" buffer_reference_align=");
                appendUint(1u << qualifier.layoutBufferReferenceAlign);
              }

              if (qualifier.layoutPassthrough)
                appendStr(" passthrough");
              if (qualifier.layoutViewportRelative)
                appendStr(" layoutViewportRelative");
              if (qualifier.layoutSecondaryViewportRelativeOffset != -2048) {
                appendStr(" layoutSecondaryViewportRelativeOffset=");
                appendInt(qualifier.layoutSecondaryViewportRelativeOffset);
              }
              if (qualifier.layoutShaderRecord)
                appendStr(" shaderRecordNV");

              appendStr(")");
            }
          }

          if (qualifier.invariant)
            appendStr(" invariant");
          if (qualifier.noContraction)
            appendStr(" noContraction");
          if (qualifier.centroid)
            appendStr(" centroid");
          if (qualifier.smooth)
            appendStr(" smooth");
          if (qualifier.flat)
            appendStr(" flat");
          if (qualifier.nopersp)
            appendStr(" noperspective");
          if (qualifier.explicitInterp)
            appendStr(" __explicitInterpAMD");
          if (qualifier.pervertexNV)
            appendStr(" pervertexNV");
          if (qualifier.pervertexEXT)
              appendStr(" pervertexEXT");
          if (qualifier.perPrimitiveNV)
            appendStr(" perprimitiveNV");
          if (qualifier.perViewNV)
            appendStr(" perviewNV");
          if (qualifier.perTaskNV)
            appendStr(" taskNV");
          if (qualifier.patch)
            appendStr(" patch");
          if (qualifier.sample)
            appendStr(" sample");
          if (qualifier.coherent)
            appendStr(" coherent");
          if (qualifier.devicecoherent)
            appendStr(" devicecoherent");
          if (qualifier.queuefamilycoherent)
            appendStr(" queuefamilycoherent");
          if (qualifier.workgroupcoherent)
            appendStr(" workgroupcoherent");
          if (qualifier.subgroupcoherent)
            appendStr(" subgroupcoherent");
          if (qualifier.shadercallcoherent)
            appendStr(" shadercallcoherent");
          if (qualifier.nonprivate)
            appendStr(" nonprivate");
          if (qualifier.volatil)
            appendStr(" volatile");
          if (qualifier.restrict)
            appendStr(" restrict");
          if (qualifier.readonly)
            appendStr(" readonly");
          if (qualifier.writeonly)
            appendStr(" writeonly");
          if (qualifier.specConstant)
            appendStr(" specialization-constant");
          if (qualifier.nonUniform)
            appendStr(" nonuniform");
          if (qualifier.isNullInit())
            appendStr(" null-init");
          if (qualifier.isSpirvByReference())
            appendStr(" spirv_by_reference");
          if (qualifier.isSpirvLiteral())
            appendStr(" spirv_literal");
          appendStr(" ");
          appendStr(getStorageQualifierString());
        }
        if (getType) {
          if (syntactic) {
            if (getPrecision && qualifier.precision != EpqNone) {
              appendStr(" ");
              appendStr(getPrecisionQualifierString());
            }
            if (isVector() || isMatrix()) {
              appendStr(" ");
              switch (basicType) {
              case EbtDouble:
                appendStr("d");
                break;
              case EbtInt:
                appendStr("i");
                break;
              case EbtUint:
                appendStr("u");
                break;
              case EbtBool:
                appendStr("b");
                break;
              case EbtFloat:
              default:
                break;
              }
              if (isVector()) {
                appendStr("vec");
                appendInt(vectorSize);
              } else {
                appendStr("mat");
                appendInt(matrixCols);
                appendStr("x");
                appendInt(matrixRows);
              }
            } else if (isStruct() && structure) {
                appendStr(" ");
                appendStr(structName.c_str());
                appendStr("{");
                bool hasHiddenMember = true;
                for (size_t i = 0; i < structure->size(); ++i) {
                  if (!(*structure)[i].type->hiddenMember()) {
                    if (!hasHiddenMember)
                      appendStr(", ");
                    typeString.append((*structure)[i].type->getCompleteString(syntactic, getQualifiers, getPrecision, getType, (*structure)[i].type->getFieldName()));
                    hasHiddenMember = false;
                  }
                }
                appendStr("}");
            } else {
                appendStr(" ");
                switch (basicType) {
                case EbtDouble:
                  appendStr("double");
                  break;
                case EbtInt:
                  appendStr("int");
                  break;
                case EbtUint:
                  appendStr("uint");
                  break;
                case EbtBool:
                  appendStr("bool");
                  break;
                case EbtFloat:
                  appendStr("float");
                  break;
                default:
                  appendStr("unexpected");
                  break;
                }
            }
            if (name.length() > 0) {
              appendStr(" ");
              appendStr(name.c_str());
            }
            if (isArray()) {
              for (int i = 0; i < (int)arraySizes->getNumDims(); ++i) {
                int size = arraySizes->getDimSize(i);
                if (size == UnsizedArraySize && i == 0 && arraySizes->isVariablyIndexed())
                  appendStr("[]");
                else {
                  if (size == UnsizedArraySize) {
                    appendStr("[");
                    if (i == 0)
                      appendInt(arraySizes->getImplicitSize());
                    appendStr("]");
                  }
                  else {
                    appendStr("[");
                    appendInt(arraySizes->getDimSize(i));
                    appendStr("]");
                  }
                }
              }
            }
          }
          else {
            if (isArray()) {
              for (int i = 0; i < (int)arraySizes->getNumDims(); ++i) {
                int size = arraySizes->getDimSize(i);
                if (size == UnsizedArraySize && i == 0 && arraySizes->isVariablyIndexed())
                  appendStr(" runtime-sized array of");
                else {
                  if (size == UnsizedArraySize) {
                    appendStr(" unsized");
                    if (i == 0) {
                      appendStr(" ");
                      appendInt(arraySizes->getImplicitSize());
                    }
                  }
                  else {
                    appendStr(" ");
                    appendInt(arraySizes->getDimSize(i));
                  }
                  appendStr("-element array of");
                }
              }
            }
            if (isParameterized()) {
              appendStr("<");
              for (int i = 0; i < (int)typeParameters->getNumDims(); ++i) {
                appendInt(typeParameters->getDimSize(i));
                if (i != (int)typeParameters->getNumDims() - 1)
                  appendStr(", ");
              }
              appendStr(">");
            }
            if (getPrecision && qualifier.precision != EpqNone) {
              appendStr(" ");
              appendStr(getPrecisionQualifierString());
            }
            if (isMatrix()) {
              appendStr(" ");
              appendInt(matrixCols);
              appendStr("X");
              appendInt(matrixRows);
              appendStr(" matrix of");
            }
            else if (isVector()) {
              appendStr(" ");
              appendInt(vectorSize);
              appendStr("-component vector of");
            }

            appendStr(" ");
            typeString.append(getBasicTypeString());

            if (qualifier.builtIn != EbvNone) {
              appendStr(" ");
              appendStr(getBuiltInVariableString());
            }

            // Add struct/block members
            if (isStruct() && structure) {
              appendStr("{");
              bool hasHiddenMember = true;
              for (size_t i = 0; i < structure->size(); ++i) {
                if (!(*structure)[i].type->hiddenMember()) {
                  if (!hasHiddenMember)
                    appendStr(", ");
                  typeString.append((*structure)[i].type->getCompleteString());
                  typeString.append(" ");
                  typeString.append((*structure)[i].type->getFieldName());
                  hasHiddenMember = false;
                }
              }
              appendStr("}");
            }
          }
        }

        return typeString;
    }

    TString getBasicTypeString() const
    {
        if (basicType == EbtSampler)
            return sampler.getString();
        else
            return getBasicString();
    }

    const char* getStorageQualifierString() const { return GetStorageQualifierString(qualifier.storage); }
    const char* getBuiltInVariableString() const { return GetBuiltInVariableString(qualifier.builtIn); }
    const char* getPrecisionQualifierString() const { return GetPrecisionQualifierString(qualifier.precision); }
#endif

    const TTypeList* getStruct() const { assert(isStruct()); return structure; }
    void setStruct(TTypeList* s) { assert(isStruct()); structure = s; }
    TTypeList* getWritableStruct() const { assert(isStruct()); return structure; }  // This should only be used when known to not be sharing with other threads
    void setBasicType(const TBasicType& t) { basicType = t; }
    
    int computeNumComponents() const
    {
        int components = 0;

        if (getBasicType() == EbtStruct || getBasicType() == EbtBlock) {
            for (TTypeList::const_iterator tl = getStruct()->begin(); tl != getStruct()->end(); tl++)
                components += ((*tl).type)->computeNumComponents();
        } else if (matrixCols)
            components = matrixCols * matrixRows;
        else
            components = vectorSize;

        if (arraySizes != nullptr) {
            components *= arraySizes->getCumulativeSize();
        }

        return components;
    }

    // append this type's mangled name to the passed in 'name'
    void appendMangledName(TString& name) const
    {
        buildMangledName(name);
        name += ';' ;
    }

    // These variables are inconsistently declared inside and outside of gl_PerVertex in glslang right now.
    // They are declared inside of 'in gl_PerVertex', but sitting as standalone when they are 'out'puts.
    bool isInconsistentGLPerVertexMember(const TString& name) const
    {
        if (name == "gl_SecondaryPositionNV" ||
            name == "gl_PositionPerViewNV")
            return true;
        return false;
    }


    // Do two structure types match?  They could be declared independently,
    // in different places, but still might satisfy the definition of matching.
    // From the spec:
    //
    // "Structures must have the same name, sequence of type names, and
    //  type definitions, and member names to be considered the same type.
    //  This rule applies recursively for nested or embedded types."
    //
    // If type mismatch in structure, return member indices through lpidx and rpidx.
    // If matching members for either block are exhausted, return -1 for exhausted
    // block and the index of the unmatched member. Otherwise return {-1,-1}.
    //
    bool sameStructType(const TType& right, int* lpidx = nullptr, int* rpidx = nullptr) const
    {
        // Initialize error to general type mismatch.
        if (lpidx != nullptr) {
            *lpidx = -1;
            *rpidx = -1;
        }

        // Most commonly, they are both nullptr, or the same pointer to the same actual structure
        // TODO: Why return true when neither types are structures?
        if ((!isStruct() && !right.isStruct()) ||
            (isStruct() && right.isStruct() && structure == right.structure))
            return true;

        if (!isStruct() || !right.isStruct())
            return false;

        // Structure names have to match
        if (*typeName != *right.typeName)
            return false;

        // There are inconsistencies with how gl_PerVertex is setup. For now ignore those as errors if they
        // are known inconsistencies.
        bool isGLPerVertex = *typeName == "gl_PerVertex";

        // Both being nullptr was caught above, now they both have to be structures of the same number of elements
        if (lpidx == nullptr &&
            (structure->size() != right.structure->size() && !isGLPerVertex)) {
            return false;
        }

        // Compare the names and types of all the members, which have to match
        for (size_t li = 0, ri = 0; li < structure->size() || ri < right.structure->size(); ++li, ++ri) {
            if (lpidx != nullptr) {
                *lpidx = static_cast<int>(li);
                *rpidx = static_cast<int>(ri);
            }
            if (li < structure->size() && ri < right.structure->size()) {
                if ((*structure)[li].type->getFieldName() == (*right.structure)[ri].type->getFieldName()) {
                    if (*(*structure)[li].type != *(*right.structure)[ri].type)
                        return false;
                } else {
                    // Skip hidden members
                    if ((*structure)[li].type->hiddenMember()) {
                        ri--;
                        continue;
                    } else if ((*right.structure)[ri].type->hiddenMember()) {
                        li--;
                        continue;
                    }
                    // If one of the members is something that's inconsistently declared, skip over it
                    // for now.
                    if (isGLPerVertex) {
                        if (isInconsistentGLPerVertexMember((*structure)[li].type->getFieldName())) {
                            ri--;
                            continue;
                        } else if (isInconsistentGLPerVertexMember((*right.structure)[ri].type->getFieldName())) {
                            li--;
                            continue;
                        }
                    } else {
                        return false;
                    }
                }
            // If we get here, then there should only be inconsistently declared members left
            } else if (li < structure->size()) {
                if (!(*structure)[li].type->hiddenMember() && !isInconsistentGLPerVertexMember((*structure)[li].type->getFieldName())) {
                    if (lpidx != nullptr) {
                        *rpidx = -1;
                    }
                    return false;
                }
            } else {
                if (!(*right.structure)[ri].type->hiddenMember() && !isInconsistentGLPerVertexMember((*right.structure)[ri].type->getFieldName())) {
                    if (lpidx != nullptr) {
                        *lpidx = -1;
                    }
                    return false;
                }
            }
        }

        return true;
    }

     bool sameReferenceType(const TType& right) const
    {
        if (isReference() != right.isReference())
            return false;

        if (!isReference() && !right.isReference())
            return true;

        assert(referentType != nullptr);
        assert(right.referentType != nullptr);

        if (referentType == right.referentType)
            return true;

        return *referentType == *right.referentType;
    }

    // See if two types match, in all aspects except arrayness
    // If mismatch in structure members, return member indices in lpidx and rpidx.
    bool sameElementType(const TType& right, int* lpidx = nullptr, int* rpidx = nullptr) const
    {
        if (lpidx != nullptr) {
            *lpidx = -1;
            *rpidx = -1;
        }
        return basicType == right.basicType && sameElementShape(right, lpidx, rpidx);
    }

    // See if two type's arrayness match
    bool sameArrayness(const TType& right) const
    {
        return ((arraySizes == nullptr && right.arraySizes == nullptr) ||
                (arraySizes != nullptr && right.arraySizes != nullptr && *arraySizes == *right.arraySizes));
    }

    // See if two type's arrayness match in everything except their outer dimension
    bool sameInnerArrayness(const TType& right) const
    {
        assert(arraySizes != nullptr && right.arraySizes != nullptr);
        return arraySizes->sameInnerArrayness(*right.arraySizes);
    }

    // See if two type's parameters match
    bool sameTypeParameters(const TType& right) const
    {
        return ((typeParameters == nullptr && right.typeParameters == nullptr) ||
                (typeParameters != nullptr && right.typeParameters != nullptr && *typeParameters == *right.typeParameters));
    }

#ifndef GLSLANG_WEB
    // See if two type's SPIR-V type contents match
    bool sameSpirvType(const TType& right) const
    {
        return ((spirvType == nullptr && right.spirvType == nullptr) ||
                (spirvType != nullptr && right.spirvType != nullptr && *spirvType == *right.spirvType));
    }
#endif

    // See if two type's elements match in all ways except basic type
    // If mismatch in structure members, return member indices in lpidx and rpidx.
    bool sameElementShape(const TType& right, int* lpidx = nullptr, int* rpidx = nullptr) const
    {
        if (lpidx != nullptr) {
            *lpidx = -1;
            *rpidx = -1;
        }
        return ((basicType != EbtSampler && right.basicType != EbtSampler) || sampler == right.sampler) &&
               vectorSize == right.vectorSize &&
               matrixCols == right.matrixCols &&
               matrixRows == right.matrixRows &&
                  vector1 == right.vector1    &&
              isCoopMat() == right.isCoopMat() &&
               sameStructType(right, lpidx, rpidx) &&
               sameReferenceType(right);
    }

    // See if a cooperative matrix type parameter with unspecified parameters is
    // an OK function parameter
    bool coopMatParameterOK(const TType& right) const
    {
        return isCoopMat() && right.isCoopMat() && (getBasicType() == right.getBasicType()) &&
               typeParameters == nullptr && right.typeParameters != nullptr;
    }

    bool sameCoopMatBaseType(const TType &right) const {
        bool rv = coopmat && right.coopmat;
        if (getBasicType() == EbtFloat || getBasicType() == EbtFloat16)
            rv = right.getBasicType() == EbtFloat || right.getBasicType() == EbtFloat16;
        else if (getBasicType() == EbtUint || getBasicType() == EbtUint8)
            rv = right.getBasicType() == EbtUint || right.getBasicType() == EbtUint8;
        else if (getBasicType() == EbtInt || getBasicType() == EbtInt8)
            rv = right.getBasicType() == EbtInt || right.getBasicType() == EbtInt8;
        else
            rv = false;
        return rv;
    }


    // See if two types match in all ways (just the actual type, not qualification)
    bool operator==(const TType& right) const
    {
#ifndef GLSLANG_WEB
        return sameElementType(right) && sameArrayness(right) && sameTypeParameters(right) && sameSpirvType(right);
#else
        return sameElementType(right) && sameArrayness(right) && sameTypeParameters(right);
#endif
    }

    bool operator!=(const TType& right) const
    {
        return ! operator==(right);
    }

    unsigned int getBufferReferenceAlignment() const
    {
#ifndef GLSLANG_WEB
        if (getBasicType() == glslang::EbtReference) {
            return getReferentType()->getQualifier().hasBufferReferenceAlign() ?
                        (1u << getReferentType()->getQualifier().layoutBufferReferenceAlign) : 16u;
        }
#endif
        return 0;
    }

#ifndef GLSLANG_WEB
    const TSpirvType& getSpirvType() const { assert(spirvType); return *spirvType; }
#endif

protected:
    // Require consumer to pick between deep copy and shallow copy.
    TType(const TType& type);
    TType& operator=(const TType& type);

    // Recursively copy a type graph, while preserving the graph-like
    // quality. That is, don't make more than one copy of a structure that
    // gets reused multiple times in the type graph.
    void deepCopy(const TType& copyOf, TMap<TTypeList*,TTypeList*>& copiedMap)
    {
        shallowCopy(copyOf);

#ifndef GLSLANG_WEB
        // GL_EXT_spirv_intrinsics
        if (copyOf.qualifier.spirvDecorate) {
            qualifier.spirvDecorate = new TSpirvDecorate;
            *qualifier.spirvDecorate = *copyOf.qualifier.spirvDecorate;
        }

        if (copyOf.spirvType) {
            spirvType = new TSpirvType;
            *spirvType = *copyOf.spirvType;
        }
#endif

        if (copyOf.arraySizes) {
            arraySizes = new TArraySizes;
            *arraySizes = *copyOf.arraySizes;
        }

        if (copyOf.typeParameters) {
            typeParameters = new TArraySizes;
            *typeParameters = *copyOf.typeParameters;
        }

        if (copyOf.isStruct() && copyOf.structure) {
            auto prevCopy = copiedMap.find(copyOf.structure);
            if (prevCopy != copiedMap.end())
                structure = prevCopy->second;
            else {
                structure = new TTypeList;
                copiedMap[copyOf.structure] = structure;
                for (unsigned int i = 0; i < copyOf.structure->size(); ++i) {
                    TTypeLoc typeLoc;
                    typeLoc.loc = (*copyOf.structure)[i].loc;
                    typeLoc.type = new TType();
                    typeLoc.type->deepCopy(*(*copyOf.structure)[i].type, copiedMap);
                    structure->push_back(typeLoc);
                }
            }
        }

        if (copyOf.fieldName)
            fieldName = NewPoolTString(copyOf.fieldName->c_str());
        if (copyOf.typeName)
            typeName = NewPoolTString(copyOf.typeName->c_str());
    }


    void buildMangledName(TString&) const;

    TBasicType basicType : 8;
    int vectorSize       : 4;  // 1 means either scalar or 1-component vector; see vector1 to disambiguate.
    int matrixCols       : 4;
    int matrixRows       : 4;
    bool vector1         : 1;  // Backward-compatible tracking of a 1-component vector distinguished from a scalar.
                               // GLSL 4.5 never has a 1-component vector; so this will always be false until such
                               // functionality is added.
                               // HLSL does have a 1-component vectors, so this will be true to disambiguate
                               // from a scalar.
    bool coopmat         : 1;
    TQualifier qualifier;

    TArraySizes* arraySizes;    // nullptr unless an array; can be shared across types
    // A type can't be both a structure (EbtStruct/EbtBlock) and a reference (EbtReference), so
    // conserve space by making these a union
    union {
        TTypeList* structure;       // invalid unless this is a struct; can be shared across types
        TType *referentType;        // invalid unless this is an EbtReference
    };
    TString *fieldName;         // for structure field names
    TString *typeName;          // for structure type name
    TSampler sampler;
    TArraySizes* typeParameters;// nullptr unless a parameterized type; can be shared across types
#ifndef GLSLANG_WEB
    TSpirvType* spirvType;  // SPIR-V type defined by spirv_type directive
#endif
};

} // end namespace glslang

#endif // _TYPES_INCLUDED_
