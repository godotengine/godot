# Third-party libraries

This file list third-party libraries used in the Android source folder,
with their provenance and, when relevant, modifications made to those files.

## Google's licensing library

- Upstream: https://github.com/google/play-licensing/tree/master/lvl_library/
- Version: git (eb57657, 2018) with modifications
- License: Apache 2.0

Overwrite all files under:

- `aidl/com/android/vending/licensing`
- `src/com/google/android/vending/licensing`

Some files have been modified to silence linter errors or fix downstream issues.
See the `patches/com.google.android.vending.licensing.patch` file.
