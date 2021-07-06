#  ANGLE GLES 1.0 Headers

The GLES 1.0 headers ANGLE uses are generated using the Khronos tools but modified to include function pointer types and function prototype guards.

### Regenerating gl.h

1. Install **Python 3** (not 2) with the **lxml** addon. You can do this using `pip install lxml` from your Python's Scripts folder.
1. Clone [https://github.com/KhronosGroup/OpenGL-Registry.git](https://github.com/KhronosGroup/OpenGL-Registry.git).
1. Edit `OpenGL-Registry/xml/genheaders.py`:

   1. Look for the section titled `# GLES 1.x API + mandatory extensions - GLES/gl.h (no function pointers)`
   1. Change `prefixText        = prefixStrings + gles1PlatformStrings + genDateCommentString,` to `prefixText        = prefixStrings + gles1PlatformStrings + apiEntryPrefixStrings + genDateCommentString,`
   1. Change `genFuncPointers   = False,` to `genFuncPointers   = True,`
   1. Change `protectProto      = False,` to `protectProto      = 'nonzero',`
   1. Change `protectProtoStr   = 'GL_GLEXT_PROTOTYPES',` to `protectProtoStr   = 'GL_GLES_PROTOTYPES',`

1. Set your working directory to `OpenGL-Registry/xml/`.
1. Run `python genheaders.py ../api/GLES/gl.h`
1. The generated header will now be in `OpenGL-Registry/api/GLES/gl.h`. You can copy the header over to this folder.
1. Also update `scripts/gl.xml` with the latest version from `OpenGL-Registry/xml/`.
