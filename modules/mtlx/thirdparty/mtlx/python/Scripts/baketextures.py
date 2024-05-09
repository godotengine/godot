#!/usr/bin/env python
'''
Generate a baked version of each material in the input document, using the TextureBaker class in the MaterialXRenderGlsl library.
'''

import sys, os, argparse
from sys import platform

import MaterialX as mx
from MaterialX import PyMaterialXRender as mx_render
from MaterialX import PyMaterialXRenderGlsl as mx_render_glsl
if platform == "darwin":
    from MaterialX import PyMaterialXRenderMsl as mx_render_msl

def main():
    parser = argparse.ArgumentParser(description="Generate a baked version of each material in the input document.")
    parser.add_argument("--width", dest="width", type=int, default=1024, help="Specify the width of baked textures.")
    parser.add_argument("--height", dest="height", type=int, default=1024, help="Specify the height of baked textures.")
    parser.add_argument("--hdr", dest="hdr", action="store_true", help="Save images to hdr format.")
    parser.add_argument("--average", dest="average", action="store_true", help="Average baked images to generate constant values.")
    parser.add_argument("--path", dest="paths", action='append', nargs='+', help="An additional absolute search path location (e.g. '/projects/MaterialX')")
    parser.add_argument("--library", dest="libraries", action='append', nargs='+', help="An additional relative path to a custom data library folder (e.g. 'libraries/custom')")
    parser.add_argument('--writeDocumentPerMaterial', dest='writeDocumentPerMaterial', type=mx.stringToBoolean, default=True, help='Specify whether to write baked materials to seprate MaterialX documents. Default is True')
    if platform == "darwin":
        parser.add_argument("--glsl", dest="useGlslBackend", default=False, type=bool, help="Set to True to use GLSL backend (default = Metal).")
    parser.add_argument(dest="inputFilename", help="Filename of the input document.")
    parser.add_argument(dest="outputFilename", help="Filename of the output document.")
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

    baseType = mx_render.BaseType.FLOAT if opts.hdr else mx_render.BaseType.UINT8
    
    
    if platform == "darwin" and not opts.useGlslBackend:
        baker = mx_render_msl.TextureBaker.create(opts.width, opts.height, baseType)
    else:
        baker = mx_render_glsl.TextureBaker.create(opts.width, opts.height, baseType)
    
    if opts.average:
        baker.setAverageImages(True)
    baker.writeDocumentPerMaterial(opts.writeDocumentPerMaterial)
    baker.bakeAllMaterials(doc, searchPath, opts.outputFilename)

if __name__ == '__main__':
    main()
