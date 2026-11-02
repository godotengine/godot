# Emscripten 3.x compatibility

The pinned Dawn release is downloaded by `drivers/webgpu/dependencies.py`.
Preparation derives `library_webgpu_legacy.js` from the original JavaScript by
changing the `$stackSave` and `$stackRestore` dependency names to their 3.x
spellings. `0001-select-legacy-stack-dependencies.patch` selects this copy for
Emscripten 3.x. Newer Emscripten versions use the unchanged upstream JavaScript.
