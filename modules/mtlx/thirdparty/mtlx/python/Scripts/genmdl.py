#!/usr/bin/env python
'''
Generate MDL implementation directory based on MaterialX nodedefs
'''

import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'

import MaterialX as mx

def usage():
    print ('genmdl.py: Generate implementation directory for mdl based on existing MaterialX nodedefs in stdlib')
    print ('Usage:  genmdl.py <library search path> [<module name> <version>]')
    print ('- A new directory called "library/stdlib/genmdl/materialx" will be created with two files added:')
    print ('   - <module_name>.ref_mdl: Module with signature stubs for each MaterialX nodedef')
    print ('   - <module_name>_genmdl_impl.ref_mtlx: MaterialX nodedef implementation mapping file')
    print ('- By default <module_name>="mymodule" and <version>="1.6"')

def _getSubDirectories(libraryPath):
    return [name for name in os.listdir(libraryPath)
            if os.path.isdir(os.path.join(libraryPath, name))]

def _getMTLXFilesInDirectory(path):
    for file in os.listdir(path):
        if file.endswith('.mtlx'):
            yield file

def _loadLibrary(file, doc):
    libDoc = mx.createDocument()
    mx.readFromXmlFile(libDoc, file)
    libDoc.setSourceUri(file)
    doc.importLibrary(libDoc)

def _loadLibraries(doc, searchPath, libraryPath):
    librarySubPaths = _getSubDirectories(libraryPath)
    librarySubPaths.append(libraryPath)
    for path in librarySubPaths:
        filenames = _getMTLXFilesInDirectory(os.path.join(libraryPath, path))
        for filename in filenames:
            filePath = os.path.join(libraryPath, os.path.join(path, filename))
            _loadLibrary(filePath, doc)

def _writeHeader(file, version):
    file.write('mdl ' + version + ';\n')
    file.write('using core import *;\n')
    IMPORT_LIST = { '::anno::*', '::base::*', '.::swizzle::*', '.::cm::*', '::math::*', '::state::*', '::tex::*', '::state::*',  '.::vectormatrix::*', '.::hsv::*', '.::noise::*'}
    # To verify what are the minimal imports required
    for i in IMPORT_LIST:
        file.write('import' + i + ';\n')
    file.write('\n\n')
    file.write('// Helper function mapping texture node addressmodes to MDL wrap modes\n')
    file.write('::tex::wrap_mode map_addressmode( mx_addressmode_type value ) {\n')
    file.write('    switch (value) {\n')
    file.write('        case mx_addressmode_type_clamp:\n')
    file.write('        return ::tex::wrap_clamp;\n')
    file.write('    case mx_addressmode_type_mirror:\n')
    file.write('        return ::tex::wrap_mirrored_repeat;\n')
    file.write('    default:\n')
    file.write( '   return ::tex::wrap_repeat;\n')
    file.write('    }\n')
    file.write('}\n\n')

def _mapGeomProp(geomprop):
    outputValue = ''
    if len(geomprop):
        if geomprop.find('UV') >= 0:
            outputValue = 'swizzle::xy(::state::texture_coordinate(0))'
        elif geomprop.find('Pobject') >= 0:
            outputValue = '::state::transform_point(::state::coordinate_internal,::state::coordinate_object,::state::position())'
        elif geomprop.find('PWorld') >= 0:
            outputValue = '::state::transform_point(::state::coordinate_internal,::state::coordinate_world,::state::position())'
        elif geomprop.find('Nobject') >= 0:
            outputValue = '::state::transform_normal(::state::coordinate_internal,::state::coordinate_object,::state::normal())'
        elif geomprop.find('Nworld') >= 0:
            outputValue = '::state::transform_normal(::state::coordinate_internal,::state::coordinate_world,::state::normal())'
        elif geomprop.find('Tobject') >= 0:
            outputValue = '::state::transform_vector(::state::coordinate_internal,::state::coordinate_object,::state::texture_tangent_u(0))'
        elif geomprop.find('Tworld') >= 0:
            outputValue = 'state::transform_vector(::state::coordinate_internal,::state::coordinate_world,::state::texture_tangent_u(0))'
        elif geomprop.find('Bobject') >= 0:
            outputValue = 'state::transform_vector(::state::coordinate_internal,::state::coordinate_object,::state::texture_tangent_v(0))'
        elif geomprop.find('Bworld') >= 0:
            outputValue = '::state::transform_vector(::state::coordinate_internal,::state::coordinate_world,::state::texture_tangent_v(0))'
    return outputValue

def _writeValueAssignment(file, outputValue, outputType, writeEmptyValues):
    # Mapping of types to initializers
    assignMap = dict()
    assignMap['float2[<N>]'] = 'float2[]'

    if outputType == 'color4':
        outputType = 'mk_color4'

    elif outputType in assignMap:
        outputType = assignMap[outputType]
        writeEmptyValues = True

    if len(outputValue) or writeEmptyValues:
        file.write(' = ')
        if outputType:
            file.write(outputType + '(')
        if outputType == 'string':
            file.write('"')
        file.write(outputValue)
        if outputType == 'string':
            file.write('"')
        if outputType:
            file.write(')')

def _mapType(typeName, typeMap, functionName):
    if 'mx_constant_filename' == functionName:
        return 'string'
    elif ('transformpoint' in functionName) or ('transformvector' in functionName) or ('transformnormal' in functionName):
        if typeName == 'string':
            return 'mx_coordinatespace_type'
    if typeName in typeMap:
        return typeMap[typeName]
    return typeName

INDENT = '\t'
SPACE = ' '
QUOTE = '"'
FUNCTION_PREFIX = 'mx_'
FUNCTION_PARAMETER_PREFIX = 'mxp_'

# Basic template for writing out logic for "image" node definition
def _writeImageImplementation(file, outputType):
    file.write(INDENT + 'if ( mxp_uaddressmode == mx_addressmode_type_constant\n')
    file.write(INDENT + '     && ( mxp_texcoord.x < 0.0 || mxp_texcoord.x > 1.0))\n')
    file.write(INDENT + INDENT + 'return mxp_default;\n')
    file.write(INDENT + 'if ( mxp_vaddressmode == mx_addressmode_type_constant\n')
    file.write(INDENT + '     && ( mxp_texcoord.y < 0.0 || mxp_texcoord.y > 1.0))\n')
    file.write(INDENT + INDENT + 'return mxp_default;\n\n')

    file.write(INDENT + outputType + ' returnValue')
    isColor4 = (outputType == 'color4')
    if isColor4:
        outputType = 'float4'
        outputType = 'mk_color4( ::tex::lookup_' + outputType
    else:
        outputType = '::tex::lookup_' + outputType
    outputValue = 'tex: mxp_file, \n' \
        + INDENT*6 + 'coord: mxp_texcoord,\n' \
        + INDENT*6 + 'wrap_u: map_addressmode(mxp_uaddressmode),\n' \
        + INDENT*6 + 'wrap_v: map_addressmode(mxp_vaddressmode)'
    _writeValueAssignment(file, outputValue, outputType, True)
    if isColor4:
        file.write(')')
    file.write(';\n')
    file.write(INDENT + 'return returnValue;\n')

def _writeOneArgumentFunc(file, outputType, functionName):
        if outputType == 'color4':
            file.write(INDENT + 'return mk_color4(' + functionName + '(mk_float4(mxp_in)));\n')
        elif outputType == 'color':
            file.write(INDENT + 'return color(' + functionName + '(mk_float3(mxp_in)));\n')
        else:
            file.write(INDENT + 'return ' + functionName + '(mxp_in);\n')

def _writeOperatorFunc(file, outputType, arg1, functionName, arg2):
        if outputType == 'color4':
            file.write(INDENT + 'return mk_color4(mk_float4(' + arg1 +') ' + functionName + ' mk_float4(' + arg2 + '));\n')
        elif outputType == 'float3x3' or outputType == 'float4x4':
            file.write(INDENT + 'return ' + outputType + '(' + arg1 + ') ' + functionName + ' ' + outputType + '(' + arg2 + ');\n')
        else:
            file.write(INDENT + 'return ' + arg1 + ' ' + functionName + ' ' + arg2 + ';\n')

def _writeTwoArgumentFunc(file, outputType, functionName):
        if outputType == 'color4':
            file.write(INDENT + 'return mk_color4(' + functionName + '(mk_float4(mxp_in1), mk_float4(mxp_in2)));\n')
        elif outputType == 'color':
            file.write(INDENT + 'return color(' + functionName + '(float3(mxp_in1), float3(mxp_in2)));\n')
        else:
            file.write(INDENT + 'return ' + functionName + '(mxp_in1, mxp_in2);\n')

def _writeThreeArgumentFunc(file, outputType, functionName, arg1, arg2, arg3):
        if outputType == 'color4':
            file.write(INDENT + 'return mk_color4(' + functionName + '(mk_float4(' + arg1 + '), mk_float4(' + arg2 + '), mk_float4(' + arg3 + ')));\n')
        elif outputType == 'color':
            file.write(INDENT + 'return color(' + functionName + '(float3(' + arg1 + '), float3(' + arg2 + '), float3(' + arg3 + ')));\n')
        else:
            file.write(INDENT + 'return ' + functionName + '(' + outputType + '(' + arg1 + '),' + outputType + '(' + arg2 + '),' + outputType+ '(' + arg3 + '));\n')

def _writeTransformMatrix(file, nodeName):
    if nodeName.find('vector3M4') >= 0:
        file.write(INDENT + 'float4 returnValue = mxp_mat * float4(mxp_in.x, mxp_in.y,  mxp_in.z, 1.0);\n')
        file.write(INDENT + 'return float3(returnValue.x, returnValue.y, returnValue.z);\n')
    elif nodeName.find('vector2M3') >= 0:
        file.write(INDENT + 'float3 returnValue = mxp_mat * float3(mxp_in.x, mxp_in.y, 1.0);\n')
        file.write(INDENT + 'return float2(returnValue.x, returnValue.y);\n')
    else:
        file.write(INDENT + 'return mxp_mat * mxp_in;\n')

def _writeTwoArgumentCombine(file, outputType):
    if outputType == 'color':
        outputType = 'color3';
    file.write(INDENT + 'return mk_' + outputType + '(mxp_in1, mxp_in2);\n')

def _writeThreeArgumentCombine(file, outputType):
    if outputType == 'color':
        file.write(INDENT + 'return ' + outputType + '(mxp_in1, mxp_in2, mxp_in3);\n')
    else:
        file.write(INDENT + 'return mk_' + outputType + '(mxp_in1, mxp_in2, mxp_in3);\n')

def _writeFourArgumentCombine(file, outputType):
    if outputType == 'color':
        outputType = 'color3';
    file.write(INDENT + 'return mk_' + outputType + '(mxp_in1, mxp_in2, mxp_in3, mxp_in4);\n')

def _writeIfGreater(file, comparitor):
    file.write(INDENT + 'if (mxp_value1 ' + comparitor + ' mxp_value2) { return mxp_in1; } return mxp_in2;\n' )

def _writeTranformSpace(file, outputType, functionName, input, fromspace, tospace):
    file.write(INDENT + 'state::coordinate_space fromSpace = ::mx_map_space(' + fromspace + ');\n')
    file.write(INDENT + 'state::coordinate_space toSpace  = ::mx_map_space(' + tospace + ');\n')
    file.write(INDENT + 'return mk_' + outputType + '( state::' + functionName + '(fromSpace, toSpace, ' + input + '));\n')

def writeNormalMap(file):
    file.write(INDENT + 'if (mxp_space == "tangent")\n')
    file.write(INDENT + '{\n')
    file.write(INDENT + '    float3 v = mxp_in * 2.0 - 1.0;\n')
    file.write(INDENT + '    float3 B = ::math::normalize(::math::cross(mxp_normal, mxp_tangent));\n')
    file.write(INDENT + '    return ::math::normalize(mxp_tangent * v.x * mxp_scale + B * v.y * mxp_scale + mxp_normal * v.z);\n')
    file.write(INDENT + '}\n')
    file.write(INDENT + 'else\n')
    file.write(INDENT + '{\n')
    file.write(INDENT + '    float3 n = mxp_in * 2.0 - 1.0;\n')
    file.write(INDENT + '    return ::math::normalize(n);\n')
    file.write(INDENT + '}\n')

def _writeRemap(file, outputType):
    if outputType == 'color4':
        file.write(INDENT + 'color4 val = mk_color4(mxp_outlow);\n')
        file.write(INDENT + 'color4 val2 = mx_add(val, mx_subtract(mk_color4(mxp_in), mk_color4(mxp_inlow)));\n')
        file.write(INDENT + 'color4 val3 = mx_multiply(val2, mx_subtract(mk_color4(mxp_outhigh), mk_color4(mxp_outlow)));\n')
        file.write(INDENT + 'return mx_divide(val3, mx_subtract(mk_color4(mxp_inhigh), mk_color4(mxp_inlow)));\n')
    else:
        file.write(INDENT + 'return mxp_outlow + (mxp_in - mxp_inlow) * (mxp_outhigh - mxp_outlow) / (mxp_inhigh - mxp_inlow);\n')

def _writeSwitch(file, outputType):
    file.write(INDENT + outputType + ' returnValue;\n')
    file.write(INDENT + 'switch (int(mxp_which)) {\n')
    file.write(INDENT*2 + 'case 0: returnValue=mxp_in1; break;\n')
    file.write(INDENT*2 + 'case 1: returnValue=mxp_in2; break;\n')
    file.write(INDENT*2 + 'case 2: returnValue=mxp_in3; break;\n')
    file.write(INDENT*2 + 'case 3: returnValue=mxp_in4; break;\n')
    file.write(INDENT*2 + 'case 4: returnValue=mxp_in5; break;\n')
    file.write(INDENT*2 + 'default: returnValue=mxp_in1; break;\n')
    file.write(INDENT + '}\n')
    file.write(INDENT + 'return returnValue;\n')

def _writeOverlay(file, outputType):
    if outputType == 'color4':
    	file.write(INDENT + 'color4 upper, lower;\n')
    	file.write(INDENT + 'color4 fg_ = color4(mxp_fg);\n')
    	file.write(INDENT + 'color4 bg_ = color4(mxp_bg);\n')
    	file.write(INDENT + 'upper = mx_multiply(mx_multiply(mk_color4(2.0),bg_),fg_);\n')
    	file.write(INDENT + 'lower = mx_subtract(mx_add(bg_,fg_),mx_multiply(bg_,fg_));\n')
    	file.write(INDENT + 'color maskRGB = color(::math::step(float3(.5), float3(fg_.rgb)));\n')
    	file.write(INDENT + 'float maskA = ::math::step(.5, fg_.a);\n')
    	file.write(INDENT + 'color overlayvalRGB = ::math::lerp(lower.rgb, upper.rgb, maskRGB);\n')
    	file.write(INDENT + 'float overlayvalA = ::math::lerp(lower.a, upper.a, maskA);\n')
    	file.write(INDENT + 'color returnRGB = ::math::lerp(mxp_bg.rgb, overlayvalRGB, color(mxp_mix));\n')
    	file.write(INDENT + 'float returnA = ::math::lerp(mxp_bg.a, overlayvalA, mxp_mix);\n')
    	file.write(INDENT + 'return color4(returnRGB, returnA);\n')
    else:
        file.write(INDENT + outputType + ' upper, lower, mask, overlayval;\n')
        file.write(INDENT + outputType + ' fg_ = ' + outputType + '(mxp_fg);\n')
        file.write(INDENT + outputType + ' bg_ = ' + outputType + '(mxp_bg);\n')
        file.write(INDENT + 'upper = 2.0*bg_*fg_;\n')
        file.write(INDENT + 'lower = bg_+fg_-bg_*fg_;\n')
        if outputType == 'color':
            file.write(INDENT + 'mask = color(::math::step(float3(.5), float3(fg_)));\n')
        else:
            file.write(INDENT + 'mask = ::math::step(' + outputType + '(.5), fg_);\n')
        file.write(INDENT + 'overlayval = ::math::lerp(lower, upper, mask);\n')
        file.write(INDENT + 'return ' + outputType + '(::math::lerp(mxp_bg, overlayval, mxp_mix));\n')

def _writeDisjointOver(file, outputType):
    if outputType == 'float2':
        file.write(INDENT + 'float2 result;\n')
        file.write(INDENT + 'float summedAlpha = mxp_fg.y + mxp_bg.y;\n')
        file.write(INDENT + 'if (summedAlpha <= 1)\n')
        file.write(INDENT + '{\n')
        file.write(INDENT + '   result.x = mxp_fg.x + mxp_bg.x;\n')
        file.write(INDENT + '}\n')
        file.write(INDENT + 'else\n')
        file.write(INDENT + '{\n')
        file.write(INDENT + '    if (::math::abs(mxp_bg.y) < FLOAT_EPS)\n')
        file.write(INDENT + '    {\n')
        file.write(INDENT + '      result.x = 0.0;\n')
        file.write(INDENT + '    }\n')
        file.write(INDENT + '    else\n')
        file.write(INDENT + '    {\n')
        file.write(INDENT + '      result.x = mxp_fg.x + ((mxp_bg.x * (1-mxp_fg.y)) / mxp_bg.y);\n')
        file.write(INDENT + '    }\n')
        file.write(INDENT + '}\n')
        file.write(INDENT + 'result.y = ::math::min(summedAlpha, 1.0);\n')
        file.write(INDENT + 'result.x = result.x * mxp_mix + (1.0 - mxp_mix) * mxp_bg.x;\n')
        file.write(INDENT + 'result.y = result.y * mxp_mix + (1.0 - mxp_mix) * mxp_bg.y;\n')
        file.write(INDENT + 'return result;\n')
    else:
        file.write(INDENT + 'color4 result;\n')
        file.write(INDENT + 'float summedAlpha = mxp_fg.a + mxp_bg.a;\n')
        file.write(INDENT + 'if (summedAlpha <= 1)\n')
        file.write(INDENT + '{\n')
        file.write(INDENT + '   result.rgb = mxp_fg.rgb + mxp_bg.rgb;\n')
        file.write(INDENT + '}\n')
        file.write(INDENT + 'else\n')
        file.write(INDENT + '{\n')
        file.write(INDENT + '    if (::math::abs(mxp_bg.a) < FLOAT_EPS)\n')
        file.write(INDENT + '    {\n')
        file.write(INDENT + '      result.rgb = color(0.0,0.0,0.0);\n')
        file.write(INDENT + '    }\n')
        file.write(INDENT + '    else\n')
        file.write(INDENT + '    {\n')
        file.write(INDENT + '      result.rgb = mxp_fg.rgb + ((mxp_bg.rgb * (1-mxp_fg.a)) / mxp_bg.a);\n')
        file.write(INDENT + '    }\n')
        file.write(INDENT + '}\n')
        file.write(INDENT + 'result.a = ::math::min(summedAlpha, 1.0);\n')
        file.write(INDENT + 'result.rgb = result.rgb * mxp_mix + (1.0 - mxp_mix) * mxp_bg.rgb;\n')
        file.write(INDENT + 'result.a = result.a * mxp_mix + (1.0 - mxp_mix) * mxp_bg.a;\n')
        file.write(INDENT + 'return result;\n')

def main():

    if len(sys.argv) < 2:
        usage()
        sys.exit(0)

    _startPath = os.path.abspath(sys.argv[1])
    if os.path.exists(_startPath) == False:
        print('Start path does not exist: ' + _startPath + '. Using current directory.\n')
        _startPath = os.path.abspath(os.getcwd())

    moduleName = 'mymodule'
    if len(sys.argv) > 2:
        moduleName = sys.argv[2]

    version = '1.6'
    if len(sys.argv) > 3:
        version = sys.argv[3]

    LIBRARY = 'stdlib'

    doc = mx.createDocument()
    searchPath = os.path.join(_startPath, 'libraries')
    libraryPath = os.path.join(searchPath, LIBRARY)
    _loadLibraries(doc, searchPath, libraryPath)

    DEFINITION_PREFIX = 'ND_'
    IMPLEMENTATION_PREFIX = 'IM_'
    IMPLEMENTATION_STRING = 'impl'
    GENMDL = 'genmdl'
    DESINATION_FOLDER = 'genmdl/materialx'

    # Create target directory if don't exist
    impl_outputPath = os.path.join(libraryPath, GENMDL)
    if not os.path.exists(impl_outputPath):
        os.mkdir(impl_outputPath)
    outputPath = os.path.join(libraryPath, DESINATION_FOLDER)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    file = None

    # Write to single file if module name specified
    if len(moduleName):
        file = open(outputPath + '/' + moduleName + '_ref.mdl', 'w+')
        _writeHeader(file, version)

    # Dictionary to map from MaterialX type declarations
    # to MDL type declarations
    typeMap = dict()
    typeMap['boolean'] = 'bool'
    typeMap['integer'] = 'int'
    typeMap['color2'] = 'float2'
    typeMap['color3'] = 'color'
    typeMap['color4'] = 'color4'
    typeMap['vector2'] = 'float2'
    typeMap['vector3'] = 'float3'
    typeMap['vector4'] = 'float4'
    typeMap['matrix33'] = 'float3x3'
    typeMap['matrix44'] = 'float4x4'
    typeMap['filename'] = 'texture_2d' # Assume all file textures are 2d for now
    typeMap['geomname'] = 'string'
    typeMap['floatarray'] = 'float[<N>]'
    typeMap['integerarray'] = 'int[<N>]'
    typeMap['color2array'] = 'float2[<N>]'
    typeMap['color3array'] = 'color[<N>]'
    typeMap['color4array'] = 'float4[<N>]'
    typeMap['vector2array'] = 'float2[<N>]'
    typeMap['vector3array'] = 'float3[<N>]'
    typeMap['vector4array'] = 'float4[<N>]'
    typeMap['stringarray'] = 'string[<N>]'
    typeMap['geomnamearray'] = 'string[<N>]'
    typeMap['surfaceshader'] = 'material'
    typeMap['volumeshader'] = 'material'
    typeMap['displacementshader'] = 'material'
    typeMap['lightshader'] = 'material'

    functionTypeMap = dict()
    functionTypeMap['mx_separate2_color2'] = 'mx_separate2_color2_type'
    functionTypeMap['mx_separate3_color3'] = 'mx_separate3_color3_type'
    functionTypeMap['mx_separate4_color4'] = 'mx_separate4_color4_type'
    functionTypeMap['mx_separate2_vector2'] = 'mx_separate2_vector2_type'
    functionTypeMap['mx_separate3_vector3'] = 'mx_separate3_vector3_type'
    functionTypeMap['mx_separate4_vector4'] = 'mx_separate4_vector4_type'

    # Create an implementation per nodedef
    #
    implDoc = mx.createDocument()
    nodedefs = doc.getNodeDefs()
    nodeGraphs = doc.getNodeGraphs()
    implementedCont = 0;
    totalCount = 0;
    for nodedef in nodedefs:

        # Skip any node definitions which are implemented as node graphs
        nodeDefName = nodedef.getName()
        #print(nodeDef)
        implementationIsGraph = False
        for nodeGraph in nodeGraphs:
            graphName = nodeGraph.getName()
            #print('Scane nodegraph: ' + nodeGraph.getName() + '\n')
            if nodeGraph.getAttribute('nodedef') == nodeDefName:
                file.write('// Nodedef: ' + nodeDefName + ' is represented by a nodegraph: ' + graphName + '\n')
                implementationIsGraph = True
                break
        if implementationIsGraph:
            continue

        # These definitions are for organization only
        nodeGroup = nodedef.getAttribute('nodegroup')
        if nodeGroup == 'organization':
            continue

        # TODO: Skip array definitions for now
        nodeCategory = nodedef.getAttribute('node')
        if  nodeCategory == 'arrayappend':
            print('Skip ' + nodeDefName + ' implementation. Not supported yet')
            continue
        if  nodeCategory == 'curveadjust':
            print('Skip ' + nodeDefName + ' implementation. Not supported yet')
            #continue
        elif nodeCategory == 'geomcolor':
            print('Skip ' + nodeDefName + ' implementation. Not supported in MDL')
            #continue
        elif nodeCategory == 'geomattrvalue':
            print('Skip ' + nodeDefName + ' implementation. Not supported in MDL')
            #continue
        elif nodeCategory == 'geompropvalue':
            print('Skip ' + nodeDefName + ' implementation. Not supported in MDL')
            #continue

        if len(nodedef.getActiveOutputs()) == 0:
           continue

        totalCount += 1

        outputValue = ''
        outputType = ''

        # String out definition prefix
        nodeName = nodedef.getName()
        if len(nodeName) > 3:
            if (nodeName[0:3] == DEFINITION_PREFIX):
                nodeName = nodeName[3:]

        filename = nodeName + '.ref_mdl'

        implname = IMPLEMENTATION_PREFIX + nodeName + '_' + GENMDL
        impl = implDoc.addImplementation(implname)
        impl.setNodeDef(nodedef)
        if len(moduleName):
            impl.setFile('stdlib/' + DESINATION_FOLDER + '/' + moduleName + '.ref_mdl')
        else:
            impl.setFile('stdlib/' + DESINATION_FOLDER + '/' + filename)

        functionName = FUNCTION_PREFIX + nodeName
        functionCallName = functionName
        if len(moduleName):
            functionCallName = functionName
        impl.setFunction(functionCallName)
        impl.setLanguage(GENMDL)

        # If no module name, create a new mdl file per nodedef
        if len(moduleName) == 0:
            file = open(outputPath + '/materialx/' + filename, 'w+')
            _writeHeader(file, version)

        outType = nodedef.getType()
        routeInputToOutput = False

        # TODO: Skip multioutput nodes for now
        #if outType == 'multioutput':
        #    continue

        # Create a signature for the nodedef
        file.write('export ')
        # Add output argument
        if functionName in functionTypeMap:
            outType = functionTypeMap[functionName]
        else:
            outType = _mapType(outType, typeMap, functionName)

        file.write(outType + SPACE)
        file.write(functionName + '(\n')

        # Add input arguments
        #
        elems = nodedef.getActiveValueElements()
        lastComma = len(elems) - len(nodedef.getActiveOutputs())
        i = 0
        channelString = ''
        for elem in elems:

            dataType = ''
            defaultgeomprop = ''

            # Skip output elements
            if isinstance(elem, mx.Output):
                outputValue = elem.getAttribute('default')
                if outputValue == '[]':
                    outputValue = ''
                if not outputValue:
                    outputValue = elem.getAttribute('defaultinput')
                    if outputValue:
                        outputValue = FUNCTION_PARAMETER_PREFIX + outputValue
                    routeInputToOutput = True
                outputType = elem.getType()
                outputType = _mapType(outputType, typeMap, functionName)
                continue

            # Parameters map to uniforms
            elif isinstance(elem, mx.Parameter):
                dataType = 'uniform '
            # Inputs map to varyings
            elif isinstance(elem, mx.Input):
                dataType = ''
                defaultgeomprop = elem.getAttribute('defaultgeomprop')

            # Determine type
            typeString = elem.getType()
            isFileTexture = (typeString == 'filename')
            typeString  = _mapType(typeString , typeMap, functionName)
            isString = (typeString == 'string')

            # Determine value
            isGeometryInput = len(defaultgeomprop) > 0
            if isGeometryInput:
                valueString = _mapGeomProp(defaultgeomprop)
            else:
                valueString = elem.getValueString()

            parameterName = FUNCTION_PARAMETER_PREFIX + elem.getName();
            isEnumeration = len(elem.getAttribute('enum')) > 0
            # Remap enumerations.
            # Note: This is hard-coded since there are no type enumerations in MaterialX to map from
            if isEnumeration and not isGeometryInput:
                ADDRESS_MODE = { 'constant', 'clamp', 'periodic', 'mirror'}
                FILTER_LOOKUP = { 'closest', 'linear', 'cubic' }
                COORDINATE_SPACES = { 'model', 'object' , 'world' }
                FILTER_TYPE = { 'box', 'gaussian' }
                if valueString in ADDRESS_MODE:
                    typeString = 'mx_addressmode_type'
                    valueString = typeString + '_' + valueString
                elif valueString in FILTER_LOOKUP:
                    typeString = 'mx_filterlookup_type'
                    valueString = typeString + '_' + valueString
                elif valueString in COORDINATE_SPACES:
                    typeString = 'mx_coordinatespace_type'
                    valueString = typeString + '_' + valueString
                elif valueString in FILTER_TYPE:
                    typeString = 'mx_filter_type'
                    valueString = typeString + '_' + valueString

            if typeString == 'mx_coordinatespace_type' and valueString == '':
                valueString = 'mx_coordinatespace_type_model'
            file.write(INDENT + dataType + typeString + SPACE + parameterName)
            _writeValueAssignment(file, valueString, typeString, isFileTexture or isString)

            if nodeCategory == 'swizzle' and parameterName == 'mxp_channels':
                channelString = valueString

            # Add annotations if any
            description = elem.getAttribute('doc')
            if len(elem.getAttribute('enum')):
                description = description + 'Enumeration {' + elem.getAttribute('enum') + '}.'
            if len(elem.getAttribute('unittype')):
                description = description + 'Unit Type:' + elem.getAttribute('unittype') + '.'
            if len(elem.getAttribute('unit')):
                description = description + ' Unit:' + elem.getAttribute('unit') + "."
            uiname = elem.getAttribute('uiname')
            uigroup = elem.getAttribute('uifolder')
            if len(description) or len(uiname) or len(uigroup):
                file.write(INDENT + '\n' + INDENT + '[[')
                count = 0
                if len(description):
                    file.write("\n" + INDENT + INDENT + 'anno::description("' + description + '")')
                    count = count + 1
                if len(uiname):
                    if count > 0:
                        file.write(',')
                    file.write("\n" + INDENT + INDENT + 'anno::display_name("' + uiname + '")')
                    count = count + 1
                if len(uigroup):
                    if count > 0:
                        file.write(',')
                    file.write("\n" + INDENT + INDENT + 'anno::in_group("' + uigroup + '")')
                file.write('\n' + INDENT + ']]')

            i = i + 1
            if i < lastComma:
                file.write(',')
            file.write('\n')

        file.write(')\n')
        nodegroup = nodedef.getAttribute('nodegroup')
        if len(nodegroup):
            file.write(INDENT + '[[\n')
            file.write(INDENT + INDENT + 'anno::description("Node Group: ' + nodegroup + '")\n')
            file.write(INDENT + ']]\n')
        if outputType == 'material':
            if outputValue:
                file.write('= ' + outputValue + '; // TODO \n\n')
            else:
                file.write('= material(); // TODO \n\n')
        else:
            file.write('{\n')
            if functionName in functionTypeMap:
                file.write(INDENT + '// No-op. Return default value for now\n')
                file.write(INDENT + 'return ' + functionTypeMap[functionName] + '();\n')
            else:
                wroteImplementation = False
                if nodeCategory == 'constant':
                    file.write(INDENT + 'return mxp_value;\n')
                    wroteImplementation = True
                elif nodeCategory == 'absval':
                    _writeOneArgumentFunc(file, outputType, '::math::abs')
                    wroteImplementation = True
                elif nodeCategory == 'ceil':
                    _writeOneArgumentFunc(file, outputType, '::math::'+nodeCategory)
                    wroteImplementation = True
                elif nodeCategory == 'round':
                    _writeOneArgumentFunc(file, outputType, '::math::'+nodeCategory)
                    wroteImplementation = True
                elif nodeCategory == 'floor':
                    _writeOneArgumentFunc(file, outputType, '::math::'+nodeCategory)
                    wroteImplementation = True
                elif nodeCategory == 'sin':
                    _writeOneArgumentFunc(file, outputType, '::math::'+nodeCategory)
                    wroteImplementation = True
                elif nodeCategory == 'asin':
                    _writeOneArgumentFunc(file, outputType, '::math::'+nodeCategory)
                    wroteImplementation = True
                elif nodeCategory == 'cos':
                    _writeOneArgumentFunc(file, outputType, '::math::'+nodeCategory)
                    wroteImplementation = True
                elif nodeCategory == 'acos':
                    _writeOneArgumentFunc(file, outputType, '::math::'+nodeCategory)
                    wroteImplementation = True
                elif nodeCategory == 'tan':
                    _writeOneArgumentFunc(file, outputType, '::math::'+nodeCategory)
                    wroteImplementation = True
                elif nodeCategory == 'atan2':
                    _writeTwoArgumentFunc(file, outputType, '::math::'+nodeCategory)
                    wroteImplementation = True
                elif nodeCategory == 'sqrt':
                    _writeOneArgumentFunc(file, outputType, '::math::'+nodeCategory)
                    wroteImplementation = True
                elif nodeCategory == 'ln':
                    _writeOneArgumentFunc(file, outputType, '::math::log')
                    wroteImplementation = True
                elif nodeCategory == 'exp':
                    _writeOneArgumentFunc(file, outputType, '::math::exp')
                    wroteImplementation = True
                elif nodeCategory == 'sign':
                    _writeOneArgumentFunc(file, outputType, '::math::sign')
                    wroteImplementation = True
                elif nodeCategory == 'max':
                    _writeTwoArgumentFunc(file, outputType, '::math::max')
                    wroteImplementation = True
                elif nodeCategory == 'min':
                    _writeTwoArgumentFunc(file, outputType, '::math::min')
                    wroteImplementation = True
                elif nodeCategory == 'add':
                    _writeOperatorFunc(file, outputType, 'mxp_in1', '+', 'mxp_in2')
                    wroteImplementation = True
                elif nodeCategory == 'subtract':
                    _writeOperatorFunc(file, outputType, 'mxp_in1', '-', 'mxp_in2')
                    wroteImplementation = True
                elif nodeCategory == 'invert':
                    _writeOperatorFunc(file, outputType, 'mxp_amount', '-', 'mxp_in')
                    wroteImplementation = True
                elif nodeCategory == 'multiply':
                    _writeOperatorFunc(file, outputType, 'mxp_in1', '*', 'mxp_in2')
                    wroteImplementation = True
                elif nodeCategory == 'divide':
                    if outputType == 'color4':
                        file.write(INDENT + 'return mk_color4(mk_float4(mxp_in1) / mk_float4(mxp_in2));')
                        wroteImplementation = True
                    elif outputType == 'float3x3' or outputType == 'float4x4':
                        file.write(INDENT + 'return vectormatrix::mx_divide(mxp_in1, mxp_in2);\n')
                        wroteImplementation = True
                    else:
                        file.write(INDENT + 'return mxp_in1 / mxp_in2;\n')
                        wroteImplementation = True
                elif nodeCategory == 'modulo':
                    _writeTwoArgumentFunc(file, outputType, 'mx_mod')
                    wroteImplementation = True
                elif nodeCategory == 'power':
                    _writeTwoArgumentFunc(file, outputType, '::math::pow')
                    wroteImplementation = True
                elif nodeCategory == 'clamp':
                    _writeThreeArgumentFunc(file, outputType, '::math::clamp', 'mxp_in', 'mxp_low', 'mxp_high')
                    wroteImplementation = True
                elif nodeCategory == 'normalize':
                    _writeOneArgumentFunc(file, outputType, '::math::normalize')
                    wroteImplementation = True
                elif nodeCategory == 'magnitude':
                    _writeOneArgumentFunc(file, outputType, '::math::length')
                    wroteImplementation = True
                elif nodeCategory == 'dotproduct':
                    _writeTwoArgumentFunc(file, outputType, '::math::dot')
                    wroteImplementation = True
                elif nodeCategory == 'crossproduct':
                    _writeTwoArgumentFunc(file, outputType, '::math::cross')
                    wroteImplementation = True
                elif nodeCategory == 'image':
                    _writeImageImplementation(file, outputType)
                    wroteImplementation = True
                elif nodeCategory == 'transformmatrix':
                    _writeTransformMatrix(file, nodeName)
                    wroteImplementation = True
                elif nodeCategory == 'determinant':
                    _writeOneArgumentFunc(file, outputType, 'vectormatrix::mx_determinant')
                    wroteImplementation = True
                elif nodeCategory == 'smoothstep':
                    _writeThreeArgumentFunc(file, outputType, '::math::smoothstep', 'mxp_in', 'mxp_low', 'mxp_high')
                    wroteImplementation = True
                elif nodeCategory == 'luminance':
                    if nodeName.find('color4') > 0:
                    	file.write(INDENT + 'color rgb = color(mxp_in.rgb);\n')
                    	file.write(INDENT + 'color4 returnValue = mk_color4(::math::luminance(rgb));\n')
                    	file.write(INDENT + 'returnValue.a = mxp_in.a;\n')
                    	file.write(INDENT + 'return returnValue;\n')
                    else:
                        _writeOneArgumentFunc(file, outputType, '::math::luminance')
                    wroteImplementation = True
                elif nodeCategory == 'plus':
                    if outputType != 'color4':
                        _writeThreeArgumentFunc(file, outputType, '::math::lerp', 'mxp_fg', '(mxp_bg+mxp_fg)', 'mxp_mix')
                    else:
                        file.write(INDENT + 'color rgb = ::math::lerp(mxp_fg.rgb, mxp_bg.rgb+mxp_fg.rgb, color(mxp_mix));\n')
                        file.write(INDENT + 'float a  = ::math::lerp(mxp_fg.a, mxp_bg.a+mxp_fg.a, mxp_mix);\n')
                        file.write(INDENT + 'return color4(rgb,a);\n')
                    wroteImplementation = True
                elif nodeCategory == 'minus':
                    if outputType != 'color4':
                        _writeThreeArgumentFunc(file, outputType, '::math::lerp', 'mxp_bg', '(mxp_bg-mxp_fg)', 'mxp_mix')
                    else:
                        file.write(INDENT + 'color rgb = ::math::lerp(mxp_bg.rgb, mxp_bg.rgb-mxp_fg.rgb, color(mxp_mix));\n')
                        file.write(INDENT + 'float a  = ::math::lerp(mxp_bg.a, mxp_bg.a-mxp_fg.a, mxp_mix);\n')
                        file.write(INDENT + 'return color4(rgb,a);\n')
                    wroteImplementation = True
                elif nodeCategory == 'difference':
                    if outputType != 'color4':
                        _writeThreeArgumentFunc(file, outputType, '::math::lerp', 'mxp_bg', 'math::abs(mxp_bg-mxp_fg)', 'mxp_mix')
                    else:
                        file.write(INDENT + 'color rgb = ::math::lerp(mxp_bg.rgb, math::abs(mxp_bg.rgb-mxp_fg.rgb), color(mxp_mix));\n')
                        file.write(INDENT + 'float a  = ::math::lerp(mxp_bg.a, math::abs(mxp_bg.a-mxp_fg.a), mxp_mix);\n')
                        file.write(INDENT + 'return color4(rgb,a);\n')
                    wroteImplementation = True
                elif nodeCategory == 'burn':
                    if outputType != 'color4':
                        burnString = outputType + '(1.0)-(' + outputType + '(1.0)-mxp_bg)/mxp_fg'
                        _writeThreeArgumentFunc(file, outputType, '::math::lerp', 'mxp_bg', burnString, 'mxp_mix')
                    else:
                        dodgeStringRGB = 'color(1.0)-(color(1.0)-mxp_bg.rgb)/mxp_fg.rgb'
                        dodgeStringA = '1.0-(1.0-mxp_bg.a)/mxp_fg.a'
                        file.write(INDENT + 'color rgb = ::math::lerp(mxp_bg.rgb,'+ dodgeStringRGB + ', color(mxp_mix));\n')
                        file.write(INDENT + 'float a  = ::math::lerp(mxp_bg.a, '+ dodgeStringA + ', mxp_mix);\n')
                        file.write(INDENT + 'return color4(rgb,a);\n')
                    wroteImplementation = True
                elif nodeCategory == 'dodge':
                    if outputType != 'color4':
                        dodgeString = 'mxp_bg/(' + outputType + '(1.0)-mxp_fg)'
                        _writeThreeArgumentFunc(file, outputType, '::math::lerp', 'mxp_bg', dodgeString, 'mxp_mix')
                    else:
                        dodgeStringRGB = 'mxp_bg.rgb/(color(1.0)-mxp_fg.rgb)'
                        dodgeStringA = 'mxp_bg.a/(1.0-mxp_fg.a)'
                        file.write(INDENT + 'color rgb = ::math::lerp(mxp_bg.rgb,'+ dodgeStringRGB + ', color(mxp_mix));\n')
                        file.write(INDENT + 'float a  = ::math::lerp(mxp_bg.a, '+ dodgeStringA + ', mxp_mix);\n')
                        file.write(INDENT + 'return color4(rgb,a);\n')
                    wroteImplementation = True
                elif nodeCategory == 'screen':
                    if outputType != 'color4':
                        _writeThreeArgumentFunc(file, outputType, '::math::lerp', 'mxp_bg', 'mxp_bg+mxp_fg-mxp_bg*mxp_fg', 'mxp_mix')
                    else:
                        file.write(INDENT + 'color rgb = ::math::lerp(mxp_bg.rgb, mxp_bg.rgb+mxp_fg.rgb-mxp_bg.rgb*mxp_fg.rgb, color(mxp_mix));\n')
                        file.write(INDENT + 'float a  = ::math::lerp(mxp_bg.a, mxp_bg.a+mxp_fg.a-mxp_bg.a*mxp_fg.a, mxp_mix);\n')
                        file.write(INDENT + 'return color4(rgb,a);\n')
                    wroteImplementation = True
                elif nodeCategory == 'inside':
                    _writeOperatorFunc(file, outputType, 'mxp_in', '*', 'mxp_mask')
                    wroteImplementation = True
                elif nodeCategory == 'outside':
                    _writeOperatorFunc(file, outputType, 'mxp_in', '*', '(1.0 - mxp_mask)')
                    wroteImplementation = True
                elif nodeCategory == 'in':
                    if outputType == 'float2':
                        _writeOperatorFunc(file, outputType, 'mxp_fg', '*', 'mxp_bg*(1.0-mxp_fg.y)')
                    else:
                        _writeOperatorFunc(file, outputType, 'mxp_fg', '*', 'mx_multiply_color4FA(mxp_bg, 1.0-mxp_fg.a)')
                    wroteImplementation = True
                elif nodeCategory == 'mix':
                    _writeThreeArgumentFunc(file, outputType, '::math::lerp', 'mxp_bg', 'mxp_fg', 'mxp_mix')
                    wroteImplementation = True
                elif nodeCategory == 'swizzle':
                    _writeOneArgumentFunc(file, outputType, 'swizzle::' + channelString)
                    wroteImplementation = True
                elif nodeCategory == 'combine2':
                    _writeTwoArgumentCombine(file, outputType)
                    wroteImplementation = True
                elif nodeCategory == 'combine3':
                    _writeThreeArgumentCombine(file, outputType)
                    wroteImplementation = True
                elif nodeCategory == 'combine4':
                    _writeFourArgumentCombine(file, outputType)
                    wroteImplementation = True
                elif nodeCategory == 'ifgreater':
                    _writeIfGreater(file, '>')
                    wroteImplementation = True
                elif nodeCategory == 'ifgreatereq':
                    _writeIfGreater(file, '>=')
                    wroteImplementation = True
                elif nodeCategory == 'ifequal':
                    _writeIfGreater(file, '==')
                    wroteImplementation = True
                elif nodeCategory == 'convert':
                    if outputType == 'float':
                        file.write(INDENT + 'return ' + outputType + '(mxp_in);\n')
                    elif outputType == 'color':
                        file.write(INDENT + 'return mk_color3(mxp_in);\n')
                    else:
                        file.write(INDENT + 'return mk_' + outputType + '(mxp_in);\n')
                    wroteImplementation = True
                elif nodeCategory == 'ramplr':
                    if outputType == 'color4':
                        file.write(INDENT + 'color rgb = math::lerp(mxp_valuel.rgb, mxp_valuer.rgb, math::clamp(mxp_texcoord.x, 0.0, 1.0));\n')
                        file.write(INDENT + 'float a = math::lerp(mxp_valuel.a, mxp_valuer.a, math::clamp(mxp_texcoord.x, 0.0, 1.0));\n')
                        file.write(INDENT + 'return color4(rgb, a);')
                    else:
                        file.write(INDENT + 'return math::lerp(mxp_valuel, mxp_valuer, math::clamp(mxp_texcoord.x, 0.0, 1.0));\n')
                    wroteImplementation = True
                elif nodeCategory == 'ramptb':
                    if outputType == 'color4':
                        file.write(INDENT + 'color rgb = math::lerp(mxp_valuet.rgb, mxp_valueb.rgb, math::clamp(mxp_texcoord.y, 0.0, 1.0));\n')
                        file.write(INDENT + 'float a = math::lerp(mxp_valuet.a, mxp_valueb.a, math::clamp(mxp_texcoord.y, 0.0, 1.0));\n')
                        file.write(INDENT + 'return color4(rgb, a);')
                    else:
                        file.write(INDENT + 'return math::lerp(mxp_valuet, mxp_valueb, math::clamp(mxp_texcoord.y, 0.0, 1.0));\n')
                    wroteImplementation = True
                elif nodeCategory == 'splitlr':
                    if outputType == 'color4':
                        file.write(INDENT + 'color rgb = math::lerp(mxp_valuel.rgb, mxp_valuer.rgb, math::step(mxp_center, math::clamp(mxp_texcoord.x,0,1)));')
                        file.write(INDENT + 'float a = math::lerp(mxp_valuel.a, mxp_valuer.a, math::step(mxp_center, math::clamp(mxp_texcoord.x,0,1)));')
                        file.write(INDENT + 'return color4(rgb, a);')
                    else:
                        file.write(INDENT + 'return math::lerp(mxp_valuel, mxp_valuer, math::step(mxp_center, math::clamp(mxp_texcoord.x,0,1)));')
                    wroteImplementation = True
                elif nodeCategory == 'splittb':
                    if outputType == 'color4':
                        file.write(INDENT + 'color rgb = math::lerp(mxp_valuet.rgb, mxp_valueb.rgb, math::step(mxp_center, math::clamp(mxp_texcoord.x,0,1)));')
                        file.write(INDENT + 'float a = math::lerp(mxp_valuet.a, mxp_valueb.a, math::step(mxp_center, math::clamp(mxp_texcoord.x,0,1)));')
                        file.write(INDENT + 'return color4(rgb, a);')
                    else:
                        file.write(INDENT + 'return math::lerp(mxp_valuet, mxp_valueb, math::step(mxp_center, math::clamp(mxp_texcoord.x,0,1)));')
                    wroteImplementation = True
                elif nodeCategory == 'transformvector':
                    _writeTranformSpace(file, outputType, 'transform_vector', 'mxp_in', 'mxp_fromspace', 'mxp_tospace')
                    wroteImplementation = True
                elif nodeCategory == 'transformpoint':
                    _writeTranformSpace(file, outputType, 'transform_point', 'mxp_in', 'mxp_fromspace', 'mxp_tospace')
                    wroteImplementation = True
                elif nodeCategory == 'transformnormal':
                    _writeTranformSpace(file, outputType, 'transform_normal', 'mxp_in', 'mxp_fromspace', 'mxp_tospace')
                    wroteImplementation = True
                elif nodeCategory == 'position':
                    _writeTranformSpace(file, outputType, 'transform_point', 'state::position()', 'mx_coordinatespace_type_model', 'mxp_space')
                    wroteImplementation = True
                elif nodeCategory == 'normal':
                    _writeTranformSpace(file, outputType, 'transform_normal', 'state::normal()', 'mx_coordinatespace_type_model', 'mxp_space')
                    wroteImplementation = True
                elif nodeCategory == 'tangent':
                    _writeTranformSpace(file, outputType, 'transform_vector', 'state::texture_tangent_u(mxp_index)', 'mx_coordinatespace_type_model', 'mxp_space')
                    wroteImplementation = True
                elif nodeCategory == 'bitangent':
                    _writeTranformSpace(file, outputType, 'transform_vector', 'state::texture_tangent_v(mxp_index)', 'mx_coordinatespace_type_model', 'mxp_space')
                    wroteImplementation = True
                elif nodeCategory == 'texcoord':
                    file.write(INDENT + 'return mk_' + outputType + '(state::texture_coordinate(mxp_index));\n')
                    wroteImplementation = True
                elif nodeCategory == 'transpose':
                    file.write(INDENT + 'return ::math::transpose(mxp_in);\n')
                    wroteImplementation = True
                elif nodeCategory == 'determinant':
                    file.write(INDENT + 'return vectormatrix::mx_determinant(mxp_in);\n')
                    wroteImplementation = True
                elif nodeCategory == 'rotate2d':
                    file.write(INDENT + 'return vectormatrix::mx_rotate(mxp_in, mxp_amount);\n')
                    wroteImplementation = True
                elif nodeCategory == 'rotate3d':
                    file.write(INDENT + 'return vectormatrix::mx_rotate(mxp_in, mxp_amount, mxp_axis);\n')
                    wroteImplementation = True
                elif nodeCategory == 'remap':
                    _writeRemap(file, outputType)
                    wroteImplementation = True
                elif nodeCategory == 'time':
                    file.write(INDENT + 'return ::state::animation_time();\n')
                    wroteImplementation = True
                elif nodeCategory == 'hsvtorgb':
                    file.write(INDENT + 'return ::hsv::mx_hsvtorgb(mxp_in);\n')
                    wroteImplementation = True
                elif nodeCategory == 'rgbtohsv':
                    file.write(INDENT + 'return ::hsv::mx_rgbtohsv(mxp_in);\n')
                    wroteImplementation = True
                elif nodeCategory == 'switch':
                    _writeSwitch(file, outputType)
                    wroteImplementation = True
                elif nodeCategory == 'overlay':
                    _writeOverlay(file, outputType)
                    wroteImplementation = True
                elif nodeCategory == 'normalmap':
                    writeNormalMap(file)
                    wroteImplementation = True

                elif nodeCategory == 'premult':
                    if outputType == 'float2':
                        file.write(INDENT + 'return float2(mxp_in.x * mxp_in.y, mxp_in.y);\n')
                    elif outputType == 'color4':
                        file.write(INDENT + 'return mk_color4(mxp_in.rgb * mxp_in.a, mxp_in.a);\n')
                    wroteImplementation = True
                elif nodeCategory == 'unpremult':
                    if outputType == 'float2':
                        file.write(INDENT + 'return float2(mxp_in.x / mxp_in.y, mxp_in.y);\n')
                    elif outputType == 'color4':
                        file.write(INDENT + 'return mk_color4(mxp_in.rgb / mxp_in.a, mxp_in.a);\n')
                    wroteImplementation = True
                elif nodeCategory == 'disjointover':
                    _writeDisjointOver(file, outputType)
                    wroteImplementation = True
                elif nodeCategory == 'mask':
                    if outputType == 'float2':
                        file.write(INDENT + 'return (mxp_bg * mxp_fg.y * mxp_mix) + mxp_bg*(1.0-mxp_mix);\n')
                    elif outputType == 'color4':
                        file.write(INDENT + 'return mx_add(' +
                            'mx_multiply_color4FA(mxp_bg,mxp_fg.a*mxp_mix),' +
                            'mx_multiply_color4FA(mxp_bg, (1.0-mxp_mix)) );\n')
                    wroteImplementation = True
                elif nodeCategory == 'matte':
                    if outputType == 'float2':
                        file.write(INDENT + 'return ' +
                        'float2( mxp_fg.x*mxp_fg.y + mxp_bg.x*(1.0-mxp_fg.y), mxp_fg.y + (mxp_bg.y*(1.0-mxp_fg.y)) ) ' +
                        '* mxp_mix + (mxp_bg * (1.0-mxp_mix));\n')
                    elif outputType == 'color4':
                        file.write(INDENT + 'color4 ls = mk_color4(\n')
                        file.write(INDENT + '        mx_multiply_color3FA(mxp_fg.rgb,mxp_fg.a) +\n')
                        file.write(INDENT + '        mx_multiply_color3FA(mxp_bg.rgb,(1.0-mxp_fg.a)),\n')
                        file.write(INDENT + '        mxp_fg.a + (mxp_bg.a*(1.0-mxp_fg.a)) );\n')
                        file.write(INDENT + 'ls = mx_multiply(ls,mk_color4(mxp_mix));\n')
                        file.write(INDENT + 'color4 rs = mx_multiply_color4FA(mxp_bg,(1.0-mxp_mix));\n')
                        file.write(INDENT + 'color4 result = mx_add(ls, rs);\n')
                        file.write(INDENT + 'return result;\n')
                    wroteImplementation = True
                elif nodeCategory == 'out':
                    if outputType == 'float2':
                        file.write(INDENT + 'return (mxp_fg*(1.0-mxp_bg.y) * mxp_mix) + (mxp_bg * (1.0-mxp_mix));\n')
                    elif outputType == 'color4':
                        file.write(INDENT + 'float4 result =\n')
                        file.write(INDENT + '    (mk_float4(mxp_fg)*(1.0-mk_float4(mxp_bg).z)  * mxp_mix) +\n')
                        file.write(INDENT + '    (mk_float4(mxp_bg) * (1.0-mxp_mix));\n')
                        file.write(INDENT + 'return mk_color4(result);\n')
                    wroteImplementation = True
                elif nodeCategory == 'over':
                    if outputType == 'float2':
                        file.write(INDENT + 'return mxp_fg + (mxp_bg*(1.0-mxp_fg.y));\n')
                    elif outputType == 'color4':
                        file.write(INDENT + 'float4 val = mk_float4(mxp_fg) + (mk_float4(mxp_bg)*(1.0-mk_float4(mxp_fg).y));\n')
                        file.write(INDENT + 'return mk_color4(val);\n')
                    wroteImplementation = True

                if wroteImplementation:
                    implementedCont += 1
                else:
                    file.write(INDENT + '// Not implemented: ' + functionName + '\n')
                    file.write(INDENT + outputType + ' defaultValue')
                    if routeInputToOutput:
                        outputType = ''
                    _writeValueAssignment(file, outputValue, outputType, outputType == 'texture_2d')
                    file.write(';\n')
                    file.write(INDENT + 'return defaultValue;\n')
            file.write('}\n\n')

        if len(moduleName) == 0:
            file.close()

    if len(moduleName):
        file.close()

    # Save implementation reference file to disk
    implFileName = moduleName + '_gen_' + IMPLEMENTATION_STRING + '.ref_mtlx'
    implPath = os.path.join(impl_outputPath, implFileName)
    print('Wrote implementation file: ' + implPath + '. ' + str(implementedCont) + '/' + str(totalCount) + '\n')
    mx.writeToXmlFile(implDoc, implPath)

if __name__ == '__main__':
    main()
