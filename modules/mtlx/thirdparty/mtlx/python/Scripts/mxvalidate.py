#!/usr/bin/env python
'''
Verify that the given file is a valid MaterialX document.
'''

import argparse
import sys

import MaterialX as mx

def main():
    parser = argparse.ArgumentParser(description="Verify that the given file is a valid MaterialX document.")
    parser.add_argument("--resolve", dest="resolve", action="store_true", help="Resolve inheritance and string substitutions.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print summary of elements found in the document.")
    parser.add_argument("--stdlib", dest="stdlib", action="store_true", help="Import standard MaterialX libraries into the document.")
    parser.add_argument(dest="inputFilename", help="Filename of the input document.")
    opts = parser.parse_args()

    doc = mx.createDocument()
    try:
        mx.readFromXmlFile(doc, opts.inputFilename)
    except mx.ExceptionFileMissing as err:
        print(err)
        sys.exit(0)

    if opts.stdlib:
        stdlib = mx.createDocument()
        try:
            mx.loadLibraries(mx.getDefaultDataLibraryFolders(), mx.getDefaultDataSearchPath(), stdlib)            
        except Exception as err:
            print(err)
            sys.exit(0)
        doc.importLibrary(stdlib)

    (valid, message) = doc.validate()
    if (valid):
        print("%s is a valid MaterialX document in v%s" % (opts.inputFilename, mx.getVersionString()))
    else:
        print("%s is not a valid MaterialX document in v%s" % (opts.inputFilename, mx.getVersionString()))
        print(message)

    if opts.verbose:
        nodegraphs = doc.getNodeGraphs()
        materials = doc.getMaterialNodes()
        looks = doc.getLooks()
        lookgroups = doc.getLookGroups()
        collections = doc.getCollections()
        nodedefs = doc.getNodeDefs()
        implementations = doc.getImplementations()
        geominfos = doc.getGeomInfos()
        geompropdefs = doc.getGeomPropDefs()
        typedefs = doc.getTypeDefs()
        propsets = doc.getPropertySets()
        variantsets = doc.getVariantSets()
        backdrops = doc.getBackdrops()

        print("----------------------------------")
        print("Document Version: {}.{:02d}".format(*doc.getVersionIntegers()))
        print("%4d Custom Type%s%s" % (len(typedefs), pl(typedefs), listContents(typedefs, opts.resolve)))
        print("%4d Custom GeomProp%s%s" % (len(geompropdefs), pl(geompropdefs), listContents(geompropdefs, opts.resolve)))
        print("%4d NodeDef%s%s" % (len(nodedefs), pl(nodedefs), listContents(nodedefs, opts.resolve)))
        print("%4d Implementation%s%s" % (len(implementations), pl(implementations), listContents(implementations, opts.resolve)))
        print("%4d Nodegraph%s%s" % (len(nodegraphs), pl(nodegraphs), listContents(nodegraphs, opts.resolve)))
        print("%4d VariantSet%s%s" % (len(variantsets), pl(variantsets), listContents(variantsets, opts.resolve)))
        print("%4d Material%s%s" % (len(materials), pl(materials), listContents(materials, opts.resolve)))
        print("%4d Collection%s%s" % (len(collections), pl(collections), listContents(collections, opts.resolve)))
        print("%4d GeomInfo%s%s" % (len(geominfos), pl(geominfos), listContents(geominfos, opts.resolve)))
        print("%4d PropertySet%s%s" % (len(propsets), pl(propsets), listContents(propsets, opts.resolve)))
        print("%4d Look%s%s" % (len(looks), pl(looks), listContents(looks, opts.resolve)))
        print("%4d LookGroup%s%s" % (len(lookgroups), pl(lookgroups), listContents(lookgroups, opts.resolve)))
        print("%4d Top-level backdrop%s%s" % (len(backdrops), pl(backdrops), listContents(backdrops, opts.resolve)))
        print("----------------------------------")

def listContents(elemlist, resolve):
    if len(elemlist) == 0:
        return ''
    names = []
    for elem in elemlist:

        if elem.isA(mx.NodeDef):
            outtype = elem.getType()
            outs = ""
            if outtype == "multioutput":
                for ot in elem.getOutputs():
                    outs = outs + \
                        '\n\t    %s output "%s"' % (ot.getType(), ot.getName())
            names.append('%s %s "%s"%s' %
                         (outtype, elem.getNodeString(), elem.getName(), outs))
            names.append(listNodedefInterface(elem))

        elif elem.isA(mx.Implementation):
            impl = elem.getName()
            targs = []
            if elem.hasTarget():
                targs.append("target %s" % elem.getTarget())
            if targs:
                impl = "%s (%s)" % (impl, ", ".join(targs))
            if elem.hasFunction():
                if elem.hasFile():
                    impl = "%s [%s:%s()]" % (
                        impl, elem.getFile(), elem.getFunction())
                else:
                    impl = "%s [function %s()]" % (impl, elem.getFunction())
            elif elem.hasFile():
                impl = "%s [%s]" % (impl, elem.getFile())
            names.append(impl)

        elif elem.isA(mx.Backdrop):
            names.append('%s: contains "%s"' %
                         (elem.getName(), elem.getContainsString()))

        elif elem.isA(mx.NodeGraph):
            nchildnodes = len(elem.getChildren()) - elem.getOutputCount()
            backdrops = elem.getBackdrops()
            nbackdrops = len(backdrops)
            outs = ""
            if nbackdrops > 0:
                for bd in backdrops:
                    outs = outs + '\n\t    backdrop "%s"' % (bd.getName())
                    outs = outs + ' contains "%s"' % bd.getContainsString()
            if elem.getOutputCount() > 0:
                for ot in elem.getOutputs():
                    outs = outs + '\n\t    %s output "%s"' % (ot.getType(), ot.getName())
                    outs = outs + traverseInputs(ot, "", 0)
            nd = elem.getNodeDef()
            if nd:
                names.append('%s (implementation for nodedef "%s"): %d nodes%s' % (
                    elem.getName(), nd.getName(), nchildnodes, outs))
            else:
                names.append("%s: %d nodes, %d backdrop%s%s" % (
                    elem.getName(), nchildnodes, nbackdrops, pl(backdrops), outs))

        elif elem.isA(mx.Node, mx.SURFACE_MATERIAL_NODE_STRING):
            shaders = mx.getShaderNodes(elem)
            names.append("%s: %d connected shader node%s" % (elem.getName(), len(shaders), pl(shaders)))
            for shader in shaders:
                names.append('Shader node "%s" (%s), with bindings:%s' % (shader.getName(), shader.getCategory(), listShaderBindings(shader)))

        elif elem.isA(mx.GeomInfo):
            props = elem.getGeomProps()
            if props:
                propnames = " (Geomprops: " + ", ".join(map(
                            lambda x: "%s=%s" % (x.getName(), getConvertedValue(x)), props)) + ")"
            else:
                propnames = ""

            tokens = elem.getTokens()
            if tokens:
                tokennames = " (Tokens: " + ", ".join(map(
                             lambda x: "%s=%s" % (x.getName(), x.getValueString()), tokens)) + ")"
            else:
                tokennames = ""
            names.append("%s%s%s" % (elem.getName(), propnames, tokennames))

        elif elem.isA(mx.VariantSet):
            vars = elem.getVariants()
            if vars:
                varnames = " (variants " + ", ".join(map(
                           lambda x: '"' + x.getName()+'"', vars)) + ")"
            else:
                varnames = ""
            names.append("%s%s" % (elem.getName(), varnames))

        elif elem.isA(mx.PropertySet):
            props = elem.getProperties()
            if props:
                propnames = " (" + ", ".join(map(
                           lambda x: "%s %s%s" % (x.getType(), x.getName(), getTarget(x)), props)) + ")"
            else:
                propnames = ""
            names.append("%s%s" % (elem.getName(), propnames))

        elif elem.isA(mx.LookGroup):
            lks = elem.getLooks()
            if lks:
                names.append("%s (looks: %s)" % (elem.getName(), lks))
            else:
                names.append("%s (no looks)" % (elem.getName()))

        elif elem.isA(mx.Look):
            mas = ""
            if resolve:
                mtlassns = elem.getActiveMaterialAssigns()
            else:
                mtlassns = elem.getMaterialAssigns()
            for mtlassn in mtlassns:
                mas = mas + "\n\t    MaterialAssign %s to%s" % (
                    mtlassn.getMaterial(), getGeoms(mtlassn, resolve))
            pas = ""
            if resolve:
                propassns = elem.getActivePropertyAssigns()
            else:
                propassns = elem.getPropertyAssigns()
            for propassn in propassns:
                propertyname = propassn.getAttribute("property")
                pas = pas + "\n\t    PropertyAssign %s %s to%s" % (
                    propassn.getType(), propertyname, getGeoms(propassn, resolve))

            psas = ""
            if resolve:
                propsetassns = elem.getActivePropertySetAssigns()
            else:
                propsetassns = elem.getPropertySetAssigns()
            for propsetassn in propsetassns:
                propertysetname = propsetassn.getAttribute("propertyset")
                psas = psas + "\n\t    PropertySetAssign %s to%s" % (
                    propertysetname, getGeoms(propsetassn, resolve))

            varas = ""
            if resolve:
                variantassns = elem.getActiveVariantAssigns()
            else:
                variantassns = elem.getVariantAssigns()
            for varassn in variantassns:
                varas = varas + "\n\t    VariantAssign %s from variantset %s" % (
                    varassn.getVariantString(), varassn.getVariantSetString())

            visas = ""
            if resolve:
                visassns = elem.getActiveVisibilities()
            else:
                visassns = elem.getVisibilities()
            for vis in visassns:
                visstr = 'on' if vis.getVisible() else 'off'
                visas = visas + "\n\t    Set %s visibility%s %s to%s" % (
                    vis.getVisibilityType(), getViewerGeoms(vis), visstr, getGeoms(vis, resolve))

            names.append("%s%s%s%s%s%s" %
                         (elem.getName(), mas, pas, psas, varas, visas))

        else:
            names.append(elem.getName())
    return ":\n\t" + "\n\t".join(names)

def listShaderBindings(shader):
    s = ''
    for inp in shader.getInputs():
        bname = inp.getName()
        btype = inp.getType()
        if inp.hasOutputString():
            outname = inp.getOutputString()
            if inp.hasNodeGraphString():
                ngname = inp.getNodeGraphString()
                s = s + '\n\t    %s "%s" -> nodegraph "%s" output "%s"' % (btype, bname, ngname, outname)
            else:
                s = s + '\n\t    %s "%s" -> output "%s"' % (btype, bname, outname)
        else:
            bval = getConvertedValue(inp)
            s = s + '\n\t    %s "%s" = %s' % (btype, bname, bval)
    return s

def listNodedefInterface(nodedef):
    s = ''
    for inp in nodedef.getActiveInputs():
        iname = inp.getName()
        itype = inp.getType()
        if s:
            s = s + '\n\t'
        s = s + '    %s input "%s"' % (itype, iname)
    for tok in nodedef.getActiveTokens():
        tname = tok.getName()
        ttype = tok.getType()
        if s:
            s = s + '\n\t'
        s = s + '    %s token "%s"' % (ttype, tname)
    return s

def traverseInputs(node, port, depth):
    s = ''
    if node.isA(mx.Output):
        parent = node.getConnectedNode()
        s = s + traverseInputs(parent, "", depth+1)
    else:
        s = s + '%s%s -> %s %s "%s"' % (spc(depth), port,
                                        node.getType(), node.getCategory(), node.getName())
        ins = node.getActiveInputs()
        for i in ins:
            if i.hasInterfaceName():
                intname = i.getInterfaceName()
                s = s + \
                    '%s%s ^- %s interface "%s"' % (spc(depth+1),
                                                   i.getName(), i.getType(), intname)
            elif i.hasValueString():
                val = getConvertedValue(i)
                s = s + \
                    '%s%s = %s value %s' % (
                        spc(depth+1), i.getName(), i.getType(), val)
            else:
                parent = i.getConnectedNode()
                if parent:
                    s = s + traverseInputs(parent, i.getName(), depth+1)
        toks = node.getActiveTokens()
        for i in toks:
            if i.hasInterfaceName():
                intname = i.getInterfaceName()
                s = s + \
                    '%s[T]%s ^- %s interface "%s"' % (
                        spc(depth+1), i.getName(), i.getType(), intname)
            elif i.hasValueString():
                val = i.getValueString()
                s = s + \
                    '%s[T]%s = %s value "%s"' % (
                        spc(depth+1), i.getName(), i.getType(), val)
            else:
                s = s + \
                    '%s[T]%s error: no valueString' % (
                        spc(depth+1), i.getName())
    return s

def pl(elem):
    if len(elem) == 1:
        return ""
    else:
        return "s"

def spc(depth):
    return "\n\t    " + ": "*depth

# Return a value string for the element, converting units if appropriate
def getConvertedValue(elem):
    if elem.getType() in ["float", "vector2", "vector3", "vector4"]:
        if elem.hasUnit():
            u = elem.getUnit()
            print ("[Unit for %s is %s]" % (elem.getName(), u))
            if elem.hasUnitType():
                utype = elem.getUnitType()
                print ("[Unittype for %s is %s]" % (elem.getName(), utype))
            # NOTDONE...
    return elem.getValueString()

def getGeoms(elem, resolve):
    s = ""
    if elem.hasGeom():
        if resolve:
            s = s + ' geom "%s"' % elem.getActiveGeom()
        else:
            s = s + ' geom "%s"' % elem.getGeom()
    if elem.hasCollectionString():
        s = s + ' collection "%s"' % elem.getCollectionString()
    return s

def getViewerGeoms(elem):
    s = ""
    if elem.hasViewerGeom():
        s = s + ' viewergeom "%s"' % elem.getViewerGeom()
    if elem.hasViewerCollection():
        s = s + ' viewercollection "%s"' % elem.getViewerCollection()
    if s:
        s = " of" + s
    return s

def getTarget(elem):
    if elem.hasTarget():
        return ' [target "%s"]' % elem.getTarget()
    else:
        return ""

if __name__ == '__main__':
    main()
