#!/usr/bin/env python
'''
Generate the "NodeGraphs.mtlx" example file programmatically.
'''

import MaterialX as mx

def main():
    doc = mx.createDocument()

    #
    # Nodegraph example 1
    #
    ng1 = doc.addNodeGraph("NG_example1")
    img1 = ng1.addNode("image", "img1", "color3")
    # Because filenames look like string types, it is necessary to explicitly declare
    # this parameter value as type "filename".
    img1.setInputValue("file", "layer1.tif", "filename")

    img2 = ng1.addNode("image", "img2", "color3")
    img2.setInputValue("file", "layer2.tif", "filename")

    img3 = ng1.addNode("image", "img3", "float")
    img3.setInputValue("file", "mask1.tif", "filename")

    n0 = ng1.addNode("mix", "n0", "color3")
    # To connect an input to another node, you must first add the input with the expected
    # type, and then setConnectedNode() that input to the desired Node object.
    infg = n0.addInput("fg", "color3")
    infg.setConnectedNode(img1)
    inbg = n0.addInput("bg", "color3")
    inbg.setConnectedNode(img2)
    inmx = n0.addInput("mix", "float")
    inmx.setConnectedNode(img3)

    n1 = ng1.addNode("multiply", "n1", "color3")
    inp1 = n1.addInput("in1", "color3")
    inp1.setConnectedNode(n0)
    inp2 = n1.setInputValue("in2", 0.22)

    nout = ng1.addOutput("diffuse", "color3")
    nout.setConnectedNode(n1)

    #
    # Nodegraph example 3
    #
    ng3 = doc.addNodeGraph("NG_example3")

    img1 = ng3.addNode("image", "img1", "color3")
    img1.setInputValue("file", "<diff_albedo>", "filename")

    img2 = ng3.addNode("image", "img2", "color3")
    img2.setInputValue("file", "<dirt_albedo>", "filename")

    img3 = ng3.addNode("image", "img3", "float")
    img3.setInputValue("file", "<areamask>", "filename")

    img4 = ng3.addNode("image", "img4", "float")
    img4.setInputValue("file", "<noisemask>", "filename")

    n5 = ng3.addNode("constant", "n5", "color3")
    # For colorN, vectorN or matrix types, use the appropriate mx Type constructor.
    n5.setInputValue("value", mx.Color3(0.8,1.0,1.3))

    n6 = ng3.addNode("multiply", "n6", "color3")
    inp1 = n6.addInput("in1", "color3")
    inp1.setConnectedNode(n5)
    inp2 = n6.addInput("in2", "color3")
    inp2.setConnectedNode(img1)

    n7 = ng3.addNode("contrast", "n7", "color3")
    inp = n7.addInput("in", "color3")
    inp.setConnectedNode(img2)
    n7.setInputValue("amount", 0.2)
    n7.setInputValue("pivot", 0.5)

    n8 = ng3.addNode("mix", "n8", "color3")
    infg = n8.addInput("fg", "color3")
    infg.setConnectedNode(n7)
    inbg = n8.addInput("bg", "color3")
    inbg.setConnectedNode(n6)
    inmx = n8.addInput("mix", "float")
    inmx.setConnectedNode(img3)

    t1 = ng3.addNode("texcoord", "t1", "vector2")

    m1 = ng3.addNode("multiply", "m1", "vector2")
    inp1 = m1.addInput("in1", "vector2")
    inp1.setConnectedNode(t1)
    m1.setInputValue("in2", 0.003)
    # If limited floating-point precision results in output value strings like "0.00299999",
    # you could instead write this as a ValueString (must add the input to the node first):
    # inp2 = m1.addInput("in2", "float")
    # inp2.setValueString("0.003")

    n9 = ng3.addNode("noise2d", "n9", "color3")
    intx = n9.addInput("texcoord", "vector2")
    intx.setConnectedNode(m1)
    n9.setInputValue("amplitude", mx.Vector3(0.05,0.04,0.06))

    n10 = ng3.addNode("inside", "n10", "color3")
    inmask = n10.addInput("mask", "float")
    inmask.setConnectedNode(img4)
    inp = n10.addInput("in", "color3")
    inp.setConnectedNode(n9)

    n11 = ng3.addNode("add", "n11", "color3")
    inp1 = n11.addInput("in1", "color3")
    inp1.setConnectedNode(n10)
    inp2 = n11.addInput("in2", "color3")
    inp2.setConnectedNode(n8)

    nout1 = ng3.addOutput("albedo", "color3")
    nout1.setConnectedNode(n11)
    nout2 = ng3.addOutput("areamask", "float")
    nout2.setConnectedNode(img3)

    # It is not necessary to validate a document before writing but it's nice
    # to know for sure.  And you can validate any element (and its children)
    # independently, not just the whole document.
    rc = ng1.validate()
    if (len(rc) >= 1 and rc[0]):
        print("Nodegraph %s is valid." % ng1.getName())
    else:
        print("Nodegraph %s is NOT valid: %s" % (ng1.getName(), str(rc[1])))
    rc = ng3.validate()
    if (len(rc) >= 1 and rc[0]):
        print("Nodegraph %s is valid." % ng3.getName())
    else:
        print("Nodegraph %s is NOT valid: %s" % (ng3.getName(), str(rc[1])))

    outfile = "myNodeGraphs.mtlx"
    mx.writeToXmlFile(doc, outfile)
    print("Wrote nodegraphs to %s" % outfile)

if __name__ == '__main__':
    main()

