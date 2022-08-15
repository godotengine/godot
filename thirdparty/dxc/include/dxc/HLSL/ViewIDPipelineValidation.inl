///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// ViewIDPipelineValidation.inl                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file implements inter-stage validation for ViewID.                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////


namespace hlsl {

namespace {

typedef std::vector<DxilSignatureAllocator::DummyElement> ElementVec;

struct ComponentMask : public PSVComponentMask {
  uint32_t Data[4];
  ComponentMask() : PSVComponentMask(Data, 0) {
    memset(Data, 0, sizeof(Data));
  }
  ComponentMask(const ComponentMask &other) : PSVComponentMask(Data, other.NumVectors) {
    *this = other;
  }
  ComponentMask(const PSVComponentMask &other) : PSVComponentMask(Data, other.NumVectors) {
    *this = other;
  }
  ComponentMask &operator=(const PSVComponentMask &other) {
    NumVectors = other.NumVectors;
    if (other.Mask && NumVectors) {
      memcpy(Data, other.Mask, sizeof(uint32_t) * PSVComputeMaskDwordsFromVectors(NumVectors));
    }
    else {
      memset(Data, 0, sizeof(Data));
    }
    return *this;
  }
  ComponentMask &operator=(const ComponentMask &other) {
    *this = (const PSVComponentMask &)other;
    return *this;
  }
  ComponentMask &operator|=(const PSVComponentMask &other) {
    NumVectors = std::max(NumVectors, other.NumVectors);
    PSVComponentMask::operator|=(other);
    return *this;
  }
};

static void InitElement(DxilSignatureAllocator::DummyElement &eOut,
                        const PSVSignatureElement &eIn,
                        DXIL::SigPointKind sigPoint) {
  eOut.rows = eIn.GetRows();
  eOut.cols = eIn.GetCols();
  eOut.row = eIn.GetStartRow();
  eOut.col = eIn.GetStartCol();
  eOut.kind = (DXIL::SemanticKind)eIn.GetSemanticKind();
  eOut.interpolation = (DXIL::InterpolationMode)eIn.GetInterpolationMode();
  eOut.interpretation = SigPoint::GetInterpretation(eOut.kind, sigPoint, 6, 1);
  eOut.indexFlags = eIn.GetDynamicIndexMask();

  // Tessfactors must be treated as dynamically indexed to prevent breaking them up
  if (eOut.interpretation == DXIL::SemanticInterpretationKind::TessFactor)
    eOut.indexFlags = (1 << eOut.cols) - 1;
}

static void CopyElements( ElementVec &outElements,
                          DXIL::SigPointKind sigPoint,
                          unsigned numElements,
                          unsigned streamIndex,
                          std::function<PSVSignatureElement(unsigned)> getElement) {
  outElements.clear();
  outElements.reserve(numElements);
  for (unsigned i = 0; i < numElements; i++) {
    auto inEl = getElement(i);
    if (!inEl.IsAllocated() || inEl.GetOutputStream() != streamIndex)
      continue;
    outElements.emplace_back(i);
    DxilSignatureAllocator::DummyElement &el = outElements.back();
    InitElement(el, inEl, sigPoint);
  }
}

static void AddViewIDElements(ElementVec &outElements,
                              ElementVec &inElements,
                              PSVComponentMask &mask,
                              unsigned viewIDCount) {
  // Compute needed elements
  for (unsigned adding = 0; adding < 2; adding++) {
    uint32_t numElements = 0;
    for (auto &E : inElements) {
      for (uint32_t row = 0; row < E.rows; row++) {
        for (uint32_t col = 0; col < E.cols; col++) {
          bool bDynIndex = E.indexFlags & (1 << col);
          if (row > 0 && bDynIndex)
            continue;
          if (adding) {
            uint32_t componentIndex = (E.GetStartRow() + row) * 4 + E.GetStartCol() + col;
            DxilSignatureAllocator::DummyElement NE(E.id);
            NE.kind = E.kind;
            NE.interpolation = E.interpolation;
            // All elements interpreted as Arbitrary for packing purposes
            NE.interpretation = DXIL::SemanticInterpretationKind::Arb;
            NE.rows = bDynIndex ? E.GetRows() : 1;
            for (uint32_t indexedRow = 0; indexedRow < NE.rows; indexedRow++) {
              if (mask.Get(componentIndex + (4 * indexedRow))) {
                NE.rows *= viewIDCount;
                break;
              }
            }
            outElements.push_back(NE);
          }
          numElements++;
        }
      }
    }
    if (!adding)
      outElements.reserve(numElements);
  }
}

static bool CheckFit(ElementVec &elements) {
  std::vector<DxilSignatureAllocator::PackElement*> packElements;
  packElements.reserve(elements.size());
  for (auto &E : elements)
    packElements.push_back(&E);
  // Since we are putting an upper limit of 4x32 registers regardless of actual element size,
  // we can just have allocator to use the default behavior.
  // This should be fixed if we enforce loose upper limit on total number of signature registers based on element size.
  DxilSignatureAllocator alloc(32, true);
  alloc.SetIgnoreIndexing(true);
  alloc.PackOptimized(packElements, 0, 32);
  for (auto &E : elements) {
    if (!E.IsAllocated())
      return false;
  }
  return true;
}

static bool CheckMaxVertexCount(const ElementVec &elements, unsigned maxVertexCount) {
  unsigned numComponents = 0;
  for (auto &E : elements)
    numComponents += E.GetRows() * E.GetCols();
  return numComponents <= 1024 && maxVertexCount <= 1024 && numComponents * maxVertexCount <= 1024;
}

static bool MergeElements(const ElementVec &priorElements,
                          ElementVec &inputElements,
                          uint32_t &numVectors,
                          unsigned &mismatchElementId) {
  inputElements.reserve(std::max(priorElements.size(), inputElements.size()));
  unsigned minElements = (unsigned)std::min(priorElements.size(), inputElements.size());
  for (unsigned i = 0; i < minElements; i++) {
    const DxilSignatureAllocator::DummyElement &priorEl = priorElements[i];
    DxilSignatureAllocator::DummyElement &inputEl = inputElements[i];
    // Verify elements match
    if (priorEl.rows != inputEl.rows ||
      priorEl.cols != inputEl.cols ||
      priorEl.row != inputEl.row ||
      priorEl.col != inputEl.col ||
      priorEl.kind != inputEl.kind ||
      // don't care about interpolation since normal signature matching ignores it: priorEl.interpolation != inputEl.interpolation ||
      priorEl.interpretation != inputEl.interpretation) {
      mismatchElementId = inputEl.id;
      return false;
    }
    // OR prior dynamic index flags into input element
    inputEl.indexFlags |= priorEl.indexFlags;
  }

  // Add extra incoming elements if there are more
  for (unsigned i = (unsigned)inputElements.size(); i < (unsigned)priorElements.size(); i++) {
    inputElements.push_back(priorElements[i]);
  }

  // Update numVectors to max
  for (unsigned i = 0; i < inputElements.size(); i++) {
    DxilSignatureAllocator::DummyElement &inputEl = inputElements[i];
    numVectors = std::max(numVectors, inputEl.row + inputEl.rows);
  }
  return true;
}

static void PropagateMask(const ComponentMask &priorMask,
                          ElementVec &inputElements,
                          ComponentMask &outMask,
                          std::function<PSVComponentMask(unsigned)> getMask) {
  // Iterate elements
  for (auto &E : inputElements) {
    for (unsigned row = 0; row < E.GetRows(); row++) {
      for (unsigned col = 0; col < E.GetCols(); col++) {
        uint32_t componentIndex = (E.GetStartRow() + row) * 4 + E.GetStartCol() + col;
        // If bit set in priorMask
        if (priorMask.Get(componentIndex)) {
          // get mask of outputs affected by inputs and OR into outMask
          outMask |= getMask(componentIndex);
        }
      }
    }
  }
}

bool DetectViewIDDependentTessFactor(const ElementVec &pcElements, ComponentMask &mask) {
  for (auto &E : pcElements) {
    if (E.GetKind() == DXIL::SemanticKind::TessFactor || E.GetKind() == DXIL::SemanticKind::InsideTessFactor) {
      for (unsigned row = 0; row < E.GetRows(); row++) {
        for (unsigned col = 0; col < E.GetCols(); col++) {
          uint32_t componentIndex = (E.GetStartRow() + row) * 4 + E.GetStartCol() + col;
          if (mask.Get(componentIndex))
            return true;
        }
      }
    }
  }
  return false;
}

class ViewIDValidator_impl : public hlsl::ViewIDValidator {
  ComponentMask m_PriorOutputMask;
  ComponentMask m_PriorPCMask;
  ElementVec m_PriorOutputSignature;
  ElementVec m_PriorPCSignature;
  unsigned m_ViewIDCount;
  unsigned m_GSRastStreamIndex;

  void ClearPriorState() {
    m_PriorOutputMask = ComponentMask();
    m_PriorPCMask = ComponentMask();
    m_PriorOutputSignature.clear();
    m_PriorPCSignature.clear();
  }

public:
  ViewIDValidator_impl(unsigned viewIDCount, unsigned gsRastStreamIndex)
    : m_PriorOutputMask(),
      m_ViewIDCount(viewIDCount),
      m_GSRastStreamIndex(gsRastStreamIndex)
  {}
  virtual ~ViewIDValidator_impl() {}
  Result ValidateStage(const DxilPipelineStateValidation &PSV,
                       bool bFinalStage,
                       bool bExpandInputOnly,
                       unsigned &mismatchElementId) override {
    if (!PSV.GetPSVRuntimeInfo0())
      return Result::InvalidPSV;
    if (!PSV.GetPSVRuntimeInfo1())
      return Result::InvalidPSVVersion;

    switch (PSV.GetShaderKind()) {
    case PSVShaderKind::Vertex: {
      if (bExpandInputOnly)
        return Result::InvalidUsage;

      // Initialize mask with direct ViewID dependent outputs
      ComponentMask mask(PSV.GetViewIDOutputMask(0));

      // capture output signature
      ElementVec outSig;
      CopyElements( outSig, DXIL::SigPointKind::VSOut, PSV.GetSigOutputElements(), 0,
                    [&](unsigned i) -> PSVSignatureElement {
                      return PSV.GetSignatureElement(PSV.GetOutputElement0(i));
                    });

      // Copy mask to prior mask
      m_PriorOutputMask = mask;

      // Capture output signature for next stage
      m_PriorOutputSignature = std::move(outSig);

      break;
    }
    case PSVShaderKind::Hull: {
      if (bFinalStage)
        return Result::InvalidUsage;

      // Initialize mask with direct ViewID dependent outputs
      ComponentMask outputMask(PSV.GetViewIDOutputMask(0));
      ComponentMask pcMask(PSV.GetViewIDPCOutputMask());

      // capture signatures
      ElementVec inSig, outSig, pcSig;
      CopyElements( inSig, DXIL::SigPointKind::HSCPIn, PSV.GetSigInputElements(), 0,
                    [&](unsigned i) -> PSVSignatureElement {
                      return PSV.GetSignatureElement(PSV.GetInputElement0(i));
                    });

      // Merge prior and input signatures, update prior mask size if necessary
      if (!MergeElements(m_PriorOutputSignature, inSig, m_PriorOutputMask.NumVectors, mismatchElementId))
        return Result::MismatchedSignatures;

      // Create new version with ViewID elements from merged signature
      ElementVec viewIDSig;
      AddViewIDElements(viewIDSig, inSig, m_PriorOutputMask, m_ViewIDCount);

      // Verify fit
      if (!CheckFit(viewIDSig))
        return Result::InsufficientInputSpace;

      if (bExpandInputOnly) {
        ClearPriorState();
        return Result::Success;
      }

      CopyElements(outSig, DXIL::SigPointKind::HSCPOut, PSV.GetSigOutputElements(), 0,
        [&](unsigned i) -> PSVSignatureElement {
        return PSV.GetSignatureElement(PSV.GetOutputElement0(i));
      });
      CopyElements(pcSig, DXIL::SigPointKind::PCOut, PSV.GetSigPatchConstOrPrimElements(), 0,
        [&](unsigned i) -> PSVSignatureElement {
        return PSV.GetSignatureElement(PSV.GetPatchConstOrPrimElement0(i));
      });

      // Propagate prior mask through input-output dependencies
      if (PSV.GetInputToOutputTable(0).IsValid()) {
        PropagateMask(m_PriorOutputMask, inSig, outputMask,
                      [&](unsigned i) -> PSVComponentMask { return PSV.GetInputToOutputTable(0).GetMaskForInput(i); });
      }
      if (PSV.GetInputToPCOutputTable().IsValid()) {
        PropagateMask(m_PriorOutputMask, inSig, pcMask,
                      [&](unsigned i) -> PSVComponentMask { return PSV.GetInputToPCOutputTable().GetMaskForInput(i); });
      }

      // Copy mask to prior mask
      m_PriorOutputMask = outputMask;
      m_PriorPCMask = pcMask;

      // Capture output signature for next stage
      m_PriorOutputSignature = std::move(outSig);
      m_PriorPCSignature = std::move(pcSig);

      if (DetectViewIDDependentTessFactor(pcSig, pcMask)) {
        return Result::SuccessWithViewIDDependentTessFactor;
      }

      break;
    }
    case PSVShaderKind::Domain: {
      // Initialize mask with direct ViewID dependent outputs
      ComponentMask mask(PSV.GetViewIDOutputMask(0));

      // capture signatures
      ElementVec inSig, pcSig, outSig;
      CopyElements( inSig, DXIL::SigPointKind::DSCPIn, PSV.GetSigInputElements(), 0,
                    [&](unsigned i) -> PSVSignatureElement {
                      return PSV.GetSignatureElement(PSV.GetInputElement0(i));
                    });
      CopyElements( pcSig, DXIL::SigPointKind::DSIn, PSV.GetSigPatchConstOrPrimElements(), 0,
                    [&](unsigned i) -> PSVSignatureElement {
                      return PSV.GetSignatureElement(PSV.GetPatchConstOrPrimElement0(i));
                    });

      // Merge prior and input signatures, update prior mask size if necessary
      if (!MergeElements(m_PriorOutputSignature, inSig, m_PriorOutputMask.NumVectors, mismatchElementId))
        return Result::MismatchedSignatures;
      if (!MergeElements(m_PriorPCSignature, pcSig, m_PriorPCMask.NumVectors, mismatchElementId))
        return Result::MismatchedPCSignatures;

      {
        // Create new version with ViewID elements from merged signature
        ElementVec viewIDSig;
        AddViewIDElements(viewIDSig, inSig, m_PriorOutputMask, m_ViewIDCount);

        // Verify fit
        if (!CheckFit(viewIDSig))
          return Result::InsufficientInputSpace;
      }

      {
        // Create new version with ViewID elements from merged signature
        ElementVec viewIDSig;
        AddViewIDElements(viewIDSig, pcSig, m_PriorPCMask, m_ViewIDCount);

        // Verify fit
        if (!CheckFit(viewIDSig))
          return Result::InsufficientPCSpace;
      }

      if (bExpandInputOnly) {
        ClearPriorState();
        return Result::Success;
      }

      CopyElements(outSig, DXIL::SigPointKind::DSOut, PSV.GetSigOutputElements(), 0,
        [&](unsigned i) -> PSVSignatureElement {
        return PSV.GetSignatureElement(PSV.GetOutputElement0(i));
      });

      // Propagate prior mask through input-output dependencies
      if (PSV.GetInputToOutputTable(0).IsValid()) {
        PropagateMask(m_PriorOutputMask, inSig, mask,
                      [&](unsigned i) -> PSVComponentMask { return PSV.GetInputToOutputTable(0).GetMaskForInput(i); });
      }
      if (PSV.GetPCInputToOutputTable().IsValid()) {
        PropagateMask(m_PriorPCMask, pcSig, mask,
                      [&](unsigned i) -> PSVComponentMask { return PSV.GetPCInputToOutputTable().GetMaskForInput(i); });
      }

      // Copy mask to prior mask
      m_PriorOutputMask = mask;
      m_PriorPCMask = ComponentMask();

      // Capture output signature for next stage
      m_PriorOutputSignature = std::move(outSig);
      m_PriorPCSignature.clear();

      break;
    }
    case PSVShaderKind::Geometry: {
      // capture signatures
      ElementVec inSig, outSig[4];
      CopyElements( inSig, DXIL::SigPointKind::GSVIn, PSV.GetSigInputElements(), 0,
                    [&](unsigned i) -> PSVSignatureElement {
                      return PSV.GetSignatureElement(PSV.GetInputElement0(i));
                    });

      // Merge prior and input signatures, update prior mask size if necessary
      if (!MergeElements(m_PriorOutputSignature, inSig, m_PriorOutputMask.NumVectors, mismatchElementId))
        return Result::MismatchedSignatures;

      // Create new version with ViewID elements from merged signature
      ElementVec viewIDSig;
      AddViewIDElements(viewIDSig, inSig, m_PriorOutputMask, m_ViewIDCount);

      // Verify fit
      if (!CheckFit(viewIDSig))
        return Result::InsufficientInputSpace;

      if (bExpandInputOnly) {
        ClearPriorState();
        return Result::Success;
      }

      for (unsigned streamIndex = 0; streamIndex < 4; streamIndex++) {
        // Initialize mask with direct ViewID dependent outputs
        ComponentMask mask(PSV.GetViewIDOutputMask(streamIndex));

        CopyElements( outSig[streamIndex], DXIL::SigPointKind::GSOut, PSV.GetSigOutputElements(), streamIndex,
                      [&](unsigned i) -> PSVSignatureElement {
                        return PSV.GetSignatureElement(PSV.GetOutputElement0(i));
                      });

        if (!outSig[streamIndex].empty()) {
          // Propagate prior mask through input-output dependencies
          if (PSV.GetInputToOutputTable(streamIndex).IsValid()) {
            PropagateMask(m_PriorOutputMask, inSig, mask,
              [&](unsigned i) -> PSVComponentMask { return PSV.GetInputToOutputTable(streamIndex).GetMaskForInput(i); });
          }

          // Create new version with ViewID elements from prior signature
          ElementVec viewIDSig;
          AddViewIDElements(viewIDSig, outSig[streamIndex], mask, m_ViewIDCount);

          // Verify fit
          if (!CheckMaxVertexCount(viewIDSig, PSV.GetPSVRuntimeInfo1()->MaxVertexCount))
            return Result::InsufficientOutputSpace;
          if (!CheckFit(viewIDSig))
            return Result::InsufficientOutputSpace;
        }

        // Capture this mask for the next stage
        if (m_GSRastStreamIndex == streamIndex)
          m_PriorOutputMask = mask;
      }

      if (m_GSRastStreamIndex < 4 && !bFinalStage) {
        m_PriorOutputSignature = std::move(outSig[m_GSRastStreamIndex]);
      } else {
        ClearPriorState();
        if (!bFinalStage)
          return Result::InvalidUsage;
      }

      return Result::Success;
    }
    case PSVShaderKind::Pixel: {
      // capture signatures
      ElementVec inSig;
      CopyElements( inSig, DXIL::SigPointKind::PSIn, PSV.GetSigInputElements(), 0,
                    [&](unsigned i) -> PSVSignatureElement {
                      return PSV.GetSignatureElement(PSV.GetInputElement0(i));
                    });

      // Merge prior and input signatures, update prior mask size if necessary
      if (!MergeElements(m_PriorOutputSignature, inSig, m_PriorOutputMask.NumVectors, mismatchElementId))
        return Result::MismatchedSignatures;

      // Create new version with ViewID elements from merged signature
      ElementVec viewIDSig;
      AddViewIDElements(viewIDSig, inSig, m_PriorOutputMask, m_ViewIDCount);

      // Verify fit
      if (!CheckFit(viewIDSig))
        return Result::InsufficientInputSpace;

      // Final stage, so clear output state.
      m_PriorOutputMask = ComponentMask();
      m_PriorOutputSignature.clear();

      // PS has to be the last stage, so return.
      return Result::Success;
    }
    case PSVShaderKind::Compute:
    default:
      return Result::InvalidUsage;
    }

    if (bFinalStage) {
      // Last stage was not pixel shader, so output has not yet been validated.
      // Create new version with ViewID elements from prior signature
      ElementVec viewIDSig;
      AddViewIDElements(viewIDSig, m_PriorOutputSignature, m_PriorOutputMask, m_ViewIDCount);

      // Verify fit
      if (!CheckFit(viewIDSig))
        return Result::InsufficientOutputSpace;
    }

    return Result::Success;
  }
};

} // namespace anonymous

ViewIDValidator* NewViewIDValidator(unsigned viewIDCount, unsigned gsRastStreamIndex) {
  return new ViewIDValidator_impl(viewIDCount, gsRastStreamIndex);
}

} // namespace hlsl
