//
// re-spirv
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace respv {
    struct SpecConstant {
        uint32_t specId = 0;
        std::vector<uint32_t> values;

        SpecConstant() {
            // Empty constructor.
        }

        SpecConstant(uint32_t pSpecId, const std::vector<uint32_t> &pValues) {
            specId = pSpecId;
            values = pValues;
        }
    };

    struct Instruction {
        uint32_t wordIndex = UINT32_MAX;
        uint32_t blockIndex = UINT32_MAX;

        Instruction(uint32_t pWordIndex, uint32_t pBlockIndex) {
            wordIndex = pWordIndex;
            blockIndex = pBlockIndex;
        }
    };

    struct Block {
        uint32_t labelInstructionIndex = UINT32_MAX;
        uint32_t terminatorInstructionIndex = UINT32_MAX;

        Block() {
            // Empty.
        }

        Block(uint32_t pLabelInstructionIndex, uint32_t pTerminatorInstructionIndex) {
            labelInstructionIndex = pLabelInstructionIndex;
            terminatorInstructionIndex = pTerminatorInstructionIndex;
        }
    };

    struct Function {
        uint32_t instructionIndex = UINT32_MAX;
        uint32_t labelInstructionIndex = UINT32_MAX;

        Function() {
            // Empty.
        }

        Function(uint32_t pInstructionIndex, uint32_t pLabelInstructionIndex) {
            instructionIndex = pInstructionIndex;
            labelInstructionIndex = pLabelInstructionIndex;
        }
    };

    struct Result {
        uint32_t instructionIndex = UINT32_MAX;

        Result() {
            // Empty.
        }

        Result(uint32_t pInstructionIndex) {
            instructionIndex = pInstructionIndex;
        }
    };

    struct Specialization {
        uint32_t constantInstructionIndex = UINT32_MAX;
        uint32_t decorationInstructionIndex = UINT32_MAX;

        Specialization() {
            // Empty.
        }

        Specialization(uint32_t pConstantInstructionIndex, uint32_t pDecorationInstructionIndex) {
            constantInstructionIndex = pConstantInstructionIndex;
            decorationInstructionIndex = pDecorationInstructionIndex;
        }
    };

    struct Decoration {
        uint32_t instructionIndex = UINT32_MAX;

        Decoration() {
            // Empty.
        }

        Decoration(uint32_t pInstructionIndex) {
            instructionIndex = pInstructionIndex;
        }
    };

    struct Variable {
        uint32_t instructionIndex = UINT32_MAX;

        Variable() {
            // Empty.
        }

        Variable(uint32_t pInstructionIndex) {
            instructionIndex = pInstructionIndex;
        }
    };

    struct AccessChain {
        uint32_t instructionIndex = UINT32_MAX;

        AccessChain() {
            // Empty.
        }

        AccessChain(uint32_t pInstructionIndex) {
            instructionIndex = pInstructionIndex;
        }
    };

    struct Phi {
        uint32_t instructionIndex = UINT32_MAX;

        Phi() {
            // Empty.
        }

        Phi(uint32_t pInstructionIndex) {
            instructionIndex = pInstructionIndex;
        }
    };

    struct LoopHeader {
        uint32_t instructionIndex = UINT32_MAX;
        uint32_t blockInstructionIndex = UINT32_MAX;

        LoopHeader() {
            // Empty.
        }

        LoopHeader(uint32_t pInstructionIndex, uint32_t pBlockInstructionIndex) {
            instructionIndex = pInstructionIndex;
            blockInstructionIndex = pBlockInstructionIndex;
        }
    };

    struct ListNode {
        uint32_t instructionIndex = UINT32_MAX;
        uint32_t nextListIndex = UINT32_MAX;

        ListNode() {
            // Empty.
        }

        ListNode(uint32_t pInstructionIndex, uint32_t pNextListIndex) {
            instructionIndex = pInstructionIndex;
            nextListIndex = pNextListIndex;
        }
    };

    struct Shader {
        const uint32_t *extSpirvWords = nullptr;
        size_t extSpirvWordCount = 0;
        std::vector<uint32_t> inlinedSpirvWords;
        std::vector<Instruction> instructions;
        std::vector<uint32_t> instructionAdjacentListIndices;
        std::vector<uint32_t> instructionInDegrees;
        std::vector<uint32_t> instructionOutDegrees;
        std::vector<uint32_t> instructionOrder;
        std::vector<Block> blocks;
        std::vector<uint32_t> blockPreOrderIndices;
        std::vector<uint32_t> blockPostOrderIndices;
        std::vector<Function> functions;
        std::vector<uint32_t> variableOrder;
        std::vector<Result> results;
        std::vector<Specialization> specializations;
        std::vector<Decoration> decorations;
        std::vector<Phi> phis;
        std::vector<LoopHeader> loopHeaders;
        std::vector<ListNode> listNodes;
        uint32_t defaultSwitchOpConstantInt = UINT32_MAX;

        Shader();
        
        // Data is only copied if pInlineFunctions is true. An extra processing pass is required if inlining is enabled.
        // This step is usually not required unless the shader compiler has disabled optimizations.
        Shader(const void *pData, size_t pSize, bool pInlineFunctions);
        void clear();
        bool checkData(const void *pData, size_t pSize);
        bool inlineData(const void *pData, size_t pSize);
        bool parseData(const void *pData, size_t pSize);
        bool parse(const void *pData, size_t pSize, bool pInlineFunctions);
        bool process(const void *pData, size_t pSize);
        bool sort(const void *pData, size_t pSize);
        bool empty() const;
    };

    struct Options {
        bool removeDeadCode = true;
    };

    struct Optimizer {
        static bool run(const Shader &pShader, const SpecConstant *pNewSpecConstants, uint32_t pNewSpecConstantCount, std::vector<uint8_t> &pOptimizedData, Options pOptions = Options());
    };
};
