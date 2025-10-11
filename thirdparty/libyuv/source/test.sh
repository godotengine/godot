#!/bin/bash
set -x

function runbenchmark1 {
  perf record /google/src/cloud/fbarchard/clean/google3/blaze-bin/third_party/libyuv/libyuv_test --gunit_filter=*$1 --libyuv_width=1280 --libyuv_height=720 --libyuv_repeat=1000 --libyuv_flags=-1 --libyuv_cpu_info=-1
  perf report | grep AVX
}

runbenchmark1 ABGRToI420
runbenchmark1 Android420ToI420
runbenchmark1 ARGBToI420
runbenchmark1 Convert16To8Plane
runbenchmark1 ConvertToARGB
runbenchmark1 ConvertToI420
runbenchmark1 CopyPlane
runbenchmark1 H010ToAB30
runbenchmark1 H010ToAR30
runbenchmark1 HalfFloatPlane
runbenchmark1 I010ToAB30
runbenchmark1 I010ToAR30
runbenchmark1 I420Copy
runbenchmark1 I420Psnr
runbenchmark1 I420Scale
runbenchmark1 I420Ssim
runbenchmark1 I420ToARGB
runbenchmark1 I420ToNV12
runbenchmark1 I420ToUYVY
runbenchmark1 I422ToI420
runbenchmark1 InitCpuFlags
runbenchmark1 J420ToARGB
runbenchmark1 NV12ToARGB
runbenchmark1 NV12ToI420
runbenchmark1 NV12ToI420Rotate
runbenchmark1 SetCpuFlags
runbenchmark1 YUY2ToI420
