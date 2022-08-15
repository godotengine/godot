# DXR Fallback Compiler
The DXR Fallback Compiler is a specialized compiler that's a part of the [D3D12 Raytracing Fallback Layer](https://github.com/Microsoft/DirectX-Graphics-Samples/tree/master/Libraries/D3D12RaytracingFallback). The purpose of the DXR Fallback Compiler is to take input DXR shader libs and link them into a single compute shader that is runnable DX12 hardware (even without DXR driver support).

## Building the DXR Fallback Compiler
In order to build the DXR Fallback Compiler in Visual Studio, simply build the dxrfallbackcompiler project in the *Clang Libraries* folder.

## Using with the D3D12 Raytracing Fallback Layer
To use the DXR Fallback Compiler with the [DirectX Graphics Samples](https://github.com/Microsoft/DirectX-Graphics-Samples/blob/master/Samples/Desktop/D3D12Raytracing/readme.md), build a dxrfallbackcompiler.dll using the Build instructions and place the output dll in Samples/Desktop/D3D12Raytracing/tools/x64. 

If you're incorporating the Fallback Layer into your own personal project, you need to ensure that the dll is either alongside your executable or in the working directory.

## Overview
Note that the below overview and all proceeding documentation assumes familiarity with the DirectX Raytracing API.

The DXR Fallback Compiler addresses several challenges that native DX12 compute shaders are not normally capable of handling:
 * Combining multiple orthogonal shaders into a single large compute shader
 * Uses of all new DXR HLSL intrinsics
 * Invocation of another shader in the middle of shader code - *i.e. TraceRay and CallShader*
 * Recursive invocations of shader calls

These challenges are handled by abstractly viewing GPU execution of a DXR pipeline as State Machine traversal, where each shader is transformed into one or more state functions. further technical details are described in the header of [StateFunctionTransform.h](../DxrFallback/StateFunctionTransform.h).

## Building runtime.h
Download LLVM 3.7: http://releases.llvm.org/3.7.0/LLVM-3.7.0-win64.exe
You may need to adjust BINPATH in script.cmd to point to your llvm binaries
Run script.cmd and it should output a patched runtime.h
