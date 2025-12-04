# FBX Module Patches

This directory contains patches for third-party libraries used by the FBX module.

## ufbx_write_blob_support.patch

Adds blob property support to the ufbx_write library to enable embedding texture/image data directly in FBX files.

### Purpose
This patch exposes a public API for adding blob properties to FBX elements, which is needed to embed image data (like PNG textures) directly in the FBX file instead of saving them as external files.

### Changes
- Adds `ufbxw_add_blob()` function to the public API (in ufbx_write.h)
- Implements `ufbxw_add_blob()` function (in ufbx_write.c)
- Fixes blob serialization in `ufbxwi_save_props()` - replaces TODO with proper 'R' format serialization
- Adds base64 encoding function `ufbxwi_base64_encode_blob()` for ASCII format
- Adds 'R' case in `ufbxwi_ascii_dom_write()` for ASCII format support with base64 encoding

### Applying the Patch

To apply this patch to a clean ufbx_write source:

```bash
cd thirdparty/ufbx
git apply ../../modules/fbx/patches/ufbx_write_blob_support.patch
```

Or using the `patch` command:

```bash
cd thirdparty/ufbx
patch -p1 < ../../modules/fbx/patches/ufbx_write_blob_support.patch
```

### Usage
```c
ufbxw_blob content_blob;
content_blob.data = png_data;
content_blob.size = png_data_size;
ufbxw_add_blob(scene, texture_id, "Content", UFBXW_PROP_TYPE_BLOB, content_blob);
```

### Status
Currently applied directly to thirdparty/ufbx/ufbx_write.h and ufbx_write.c in this repository.
