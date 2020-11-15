rm -rf buildtools/
rm -rf .gclient
rm -rf .gclient_entries
rm -rf crashpad/build/
rm -rf crashpad/out/
rm -rf crashpad/third_party/gyp/
rm -rf crashpad/third_party/gtest/
rm -rf crashpad/third_party/libfuzzer/
(find . -type d -name ".git") | xargs rm -rf