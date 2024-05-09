#!/usr/bin/env python
'''
Generate a baked translated version of each material in the input document, using the ShaderTranslator class in the MaterialXShaderGen library
and the TextureBaker class in the MaterialXRenderGlsl library.
'''

import sys, os, argparse
from sys import platform

import MaterialX as mx
from MaterialX import PyMaterialXGenShader as mx_gen_shader
from MaterialX import PyMaterialXRender as mx_render
from MaterialX import PyMaterialXRenderGlsl as mx_render_glsl
if platform == "darwin":
    from MaterialX import PyMaterialXRenderMsl as mx_render_msl

def main():
    parser = argparse.ArgumentParser(description="Generate a translated baked version of each material in the input document.")
    parser.add_argument("--width", dest="width", type=int, default=0, help="Specify an optional width for baked textures (defaults to the maximum image height in the source document).")
    parser.add_argument("--height", dest="height", type=int, default=0, help="Specify an optional height for baked textures (defaults to the maximum image width in the source document).")
    parser.add_argument("--hdr", dest="hdr", action="store_true", help="Bake images with high dynamic range (e.g. in HDR or EXR format).")
    parser.add_argument("--path", dest="paths", action='append', nargs='+', help="An additional absolute search path location (e.g. '/projects/MaterialX')")
    parser.add_argument("--library", dest="libraries", action='append', nargs='+', help="An additional relative path to a custom data library folder (e.g. 'libraries/custom')")
    parser.add_argument('--writeDocumentPerMaterial', dest='writeDocumentPerMaterial', type=mx.stringToBoolean, default=True, help='Specify whether to write baked materials to seprate MaterialX documents. Default is True')
    if platform == "darwin":
        parser.add_argument("--glsl", dest="useGlslBackend", default=False, type=bool, help="Set to True to use GLSL backend (default = Metal).")

    parser.add_argument(dest="inputFilename", help="Filename of the input document.")
    parser.add_argument(dest="outputFilename", help="Filename of the output document.")
    parser.add_argument(dest="destShader", help="Destination shader for translation")
    opts = parser.parse_args()

    doc = mx.createDocument()
    try:
        mx.readFromXmlFile(doc, opts.inputFilename)
    except mx.ExceptionFileMissing as err:
        print(err)
        sys.exit(0)

    stdlib = mx.createDocument()
    searchPath = mx.getDefaultDataSearchPath()
    searchPath.append(os.path.dirname(opts.inputFilename))
    libraryFolders = []
    if opts.paths:
        for pathList in opts.paths:
            for path in pathList:
                searchPath.append(path)
    if opts.libraries:
        for libraryList in opts.libraries:
            for library in libraryList:
                libraryFolders.append(library)
    libraryFolders.extend(mx.getDefaultDataLibraryFolders())
    mx.loadLibraries(libraryFolders, searchPath, stdlib)
    doc.importLibrary(stdlib)

    valid, msg = doc.validate()
    if not valid:
        print("Validation warnings for input document:")
        print(msg)

    # Check the document for a UDIM set.
    udimSetValue = doc.getGeomPropValue(mx.UDIM_SET_PROPERTY)
    udimSet = udimSetValue.getData() if udimSetValue else []

    # Compute baking resolution from the source document.
    imageHandler = mx_render.ImageHandler.create(mx_render.StbImageLoader.create())
    imageHandler.setSearchPath(searchPath)
    if udimSet:
        resolver = doc.createStringResolver()
        resolver.setUdimString(udimSet[0])
        imageHandler.setFilenameResolver(resolver)
    imageVec = imageHandler.getReferencedImages(doc)
    bakeWidth, bakeHeight = mx_render.getMaxDimensions(imageVec)

    # Apply baking resolution settings.
    if opts.width > 0:
        bakeWidth = opts.width
    if opts.height > 0:
        bakeHeight = opts.height
    bakeWidth = max(bakeWidth, 4)
    bakeHeight = max(bakeHeight, 4)

    # Translate materials between shading models
    translator = mx_gen_shader.ShaderTranslator.create()
    try:
        translator.translateAllMaterials(doc, opts.destShader)
    except mx.Exception as err:
        print(err)
        sys.exit(0)
        
    # Bake translated materials to flat textures.
    baseType = mx_render.BaseType.FLOAT if opts.hdr else mx_render.BaseType.UINT8
    if platform == "darwin" and not opts.useGlslBackend:
        baker = mx_render_msl.TextureBaker.create(bakeWidth, bakeHeight, baseType)
    else:
        baker = mx_render_glsl.TextureBaker.create(bakeWidth, bakeHeight, baseType)
    baker.writeDocumentPerMaterial(opts.writeDocumentPerMaterial)
    baker.bakeAllMaterials(doc, searchPath, opts.outputFilename)

if __name__ == '__main__':
    main()
