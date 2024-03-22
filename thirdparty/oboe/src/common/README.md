# Notes on Implementation

## Latency from Resampling

There are two components of the latency. The resampler itself, and a buffer that
is used to adapt the block sizes.

1) The resampler is an FIR running at the target sample rate. So its latency is the number of taps.
From MultiChannelResampler.cpp, numTaps is

    Fastest: 2
    Low: 4
    Medium: 8
    High: 16
    Best: 32

For output, the device sampling rate is used, which is typically 48000.For input, the app sampling rate is used.

2) There is a block size adapter that collects odd sized blocks into larger blocks of the correct size.

The adapter contains one burst of frames, from getFramesPerBurst(). But if the app specifies a
particular size using setFramesPerCallback() then that size will be used.
Here is some pseudo-code to calculate the latency.

    latencyMillis = 0
    targetRate = isOutput ? deviceRate : applicationRate
    // Add latency from FIR
    latencyMillis += numTaps * 1000.0 / targetRate
    // Add latency from block size adaptation
    adapterSize = (callbackSize > 0) ? callbackSize : burstSize
    if (isOutput && isCallbackUsed) latencyMillis += adapterSize * 1000.0 / deviceRate
    else if (isInput && isCallbackUsed) latencyMillis += adapterSize * 1000.0 / applicationRate
    else if (isInput && !isCallbackUsed) latencyMillis += adapterSize * 1000.0 / deviceRate
