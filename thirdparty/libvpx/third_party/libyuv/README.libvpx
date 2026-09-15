Name: libyuv
URL: https://chromium.googlesource.com/libyuv/libyuv
Version: a37e7bfece9e0676ae90a1700b0ec85b0f4f22a1
License: BSD
License File: LICENSE

Description:
libyuv is an open source project that includes YUV conversion and scaling
functionality.

The optimized scaler in libyuv is used in the multiple resolution encoder
example which down-samples the original input video (f.g. 1280x720) a number of
times in order to encode multiple resolution bit streams.

Local Modifications:
Disable ARGBToRGB24Row_AVX512VBMI due to build failure on Mac.
rm libyuv/include/libyuv.h libyuv/include/libyuv/compare_row.h
mv libyuv/include tmp/
mv libyuv/source tmp/
mv libyuv/LICENSE tmp/
rm -rf libyuv

mv tmp/* third_party/libyuv/
