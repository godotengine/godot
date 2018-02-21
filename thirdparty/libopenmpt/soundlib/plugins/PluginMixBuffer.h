/*
 * PluginMixBuffer.h
 * -----------------
 * Purpose: Helper class for managing plugin audio input and output buffers.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

OPENMPT_NAMESPACE_BEGIN

// At least this part of the code is ready for double-precision rendering... :>
// buffer_t: Sample buffer type (float, double, ...)
// bufferSize: Buffer size in samples
template<typename buffer_t, uint32 bufferSize>
class PluginMixBuffer
{
protected:

	std::vector<buffer_t>   mixBuffer;	// Actual buffer, contains all input and output buffers
	std::vector<buffer_t *> inputs;		// Pointers to input buffers
	std::vector<buffer_t *> outputs;	// Pointers to output buffers
	buffer_t *alignedBuffer;			// Aligned pointer to the buffer

	// Align buffer to 16 bytes
	static_assert(sizeof(buffer_t) < 16, "Check buffer alignment code");
	static const uintptr_t bufferAlignmentInBytes = (16 - 1);
	static const uintptr_t additionalBuffer = bufferAlignmentInBytes / sizeof(buffer_t);

	// Return pointer to an aligned buffer
	buffer_t *GetBuffer(size_t index) const
	{
		MPT_ASSERT(index < inputs.size() + outputs.size());
		return &alignedBuffer[bufferSize * index];
	}

public:

	// Allocate input and output buffers
	bool Initialize(uint32 numInputs, uint32 numOutputs)
	{
		// Short cut - we do not need to recreate the buffers.
		if(inputs.size() == numInputs && outputs.size() == numOutputs)
		{
			return true;
		}

		try
		{
			inputs.resize(numInputs);
			outputs.resize(numOutputs);

			// Create inputs + outputs buffers with additional alignment.
			const size_t totalBufferSize = bufferSize * (numInputs + numOutputs) + additionalBuffer;
			mixBuffer.assign(totalBufferSize, 0);

			// Align buffer start.
			alignedBuffer = reinterpret_cast<buffer_t *>((reinterpret_cast<uintptr_t>(mixBuffer.data()) + bufferAlignmentInBytes) & ~bufferAlignmentInBytes);
		} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
		{
			MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
			inputs.clear();
			outputs.clear();
			mixBuffer.clear();
			alignedBuffer = nullptr;
			return false;
		}

		for(uint32 i = 0; i < numInputs; i++)
		{
			inputs[i] = GetInputBuffer(i);
		}

		for(uint32 i = 0; i < numOutputs; i++)
		{
			outputs[i] = GetOutputBuffer(i);
		}

		return true;
	}

	// Silence all input buffers.
	void ClearInputBuffers(uint32 numSamples)
	{
		MPT_ASSERT(numSamples <= bufferSize);
		for(size_t i = 0; i < inputs.size(); i++)
		{
			memset(inputs[i], 0, numSamples * sizeof(buffer_t));
		}
	}

	// Silence all output buffers.
	void ClearOutputBuffers(uint32 numSamples)
	{
		MPT_ASSERT(numSamples <= bufferSize);
		for(size_t i = 0; i < outputs.size(); i++)
		{
			memset(outputs[i], 0, numSamples * sizeof(buffer_t));
		}
	}

	PluginMixBuffer()
	{
		Initialize(2, 0);
	}

	// Return pointer to a given input or output buffer
	buffer_t *GetInputBuffer(uint32 index) const { return GetBuffer(index); }
	buffer_t *GetOutputBuffer(uint32 index) const { return GetBuffer(inputs.size() + index); }

	// Return pointer array to all input or output buffers
	buffer_t **GetInputBufferArray() { return inputs.empty() ? nullptr : inputs.data(); }
	buffer_t **GetOutputBufferArray() { return outputs.empty() ? nullptr : outputs.data(); }

	bool Ok() const { return alignedBuffer != nullptr; }
};


OPENMPT_NAMESPACE_END
