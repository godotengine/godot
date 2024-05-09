import { expect } from 'chai';
import Module from './_build/JsMaterialXCore.js';
import { getMtlxStrings } from './testHelpers';

describe('Custom Bindings', () =>
{
    const examplesPath = '../../resources/Materials/Examples/StandardSurface';

    let mx;
    before(async () =>
    {
        mx = await Module();
    });

    it('Optional parameters work as expected', () =>
    {
        const doc = mx.createDocument();
        // Call a method without optional argument
        const nodeGraph = doc.addNodeGraph();
        expect(nodeGraph).to.be.instanceof(mx.NodeGraph);
        expect(nodeGraph.getName()).to.equal('nodegraph1'); // Auto-constructed default value
        // Call a method with optional argument
        const nodeGraph2 = doc.addNodeGraph('myGraph');
        expect(nodeGraph2).to.be.instanceof(mx.NodeGraph);
        expect(nodeGraph2.getName()).to.equal('myGraph');

        // Call a method that requires at least one parameter
        const node = nodeGraph.addNode('node');
        expect(node).to.be.instanceof(mx.Node);

        // Omitting non-optional parameter should throw
        expect(() => { nodeGraph.addNode(); }).to.throw;
    });

    it('Vector <-> Array conversion', () =>
    {
        // Functions that return vectors in C++ should return an array in JS
        const doc = mx.createDocument();
        const nodeGraph = doc.addNodeGraph();
        doc.addNodeGraph();
        const nodeGraphs = doc.getNodeGraphs();
        expect(nodeGraphs).to.be.an.instanceof(Array);
        expect(nodeGraphs.length).to.equal(2);

        // Elements fetched through the vector -> array conversion should be editable and changes should be reflected
        // in the original objects.
        // Note: We cannot simply compare these objects for equality, since they're separately constructed pointers
        // to the same object.
        const backdrop = nodeGraph.addBackdrop();
        const backDrops = nodeGraphs[0].getBackdrops();
        expect(backDrops.length).to.equal(1);
        nodeGraphs[0].addBackdrop();
        expect(nodeGraph.getBackdrops().length).to.equal(2);

        // Functions that expect vectors as parameters in C++ should accept arrays in JS
        // Built-in types (strings)
        const pathSegments = ['path', 'to', 'something'];
        const namePath = mx.createNamePath(pathSegments);
        expect(namePath).to.equal(pathSegments.join(mx.NAME_PATH_SEPARATOR));

        // Complex (smart pointer) types
        const node1 = nodeGraph.addNode('node');
        const node2 = nodeGraph.addNode('node');
        const node3 = nodeGraph.addNode('node', 'anotherNode');
        backdrop.setContainsElements([node1, node2, node3]);
        const nodes = backdrop.getContainsElements();
        expect(nodes.length).to.equal(3);
        expect(nodes[0].getName()).to.equal('node1'); // Name auto-constructed from category
        expect(nodes[1].getName()).to.equal('node2'); // Name auto-constructed from category
        expect(nodes[2].getName()).to.equal('anotherNode'); // Name set explicitly at creation time
    });

    it('C++ exception handling', () =>
    {
        // Exceptions that are thrown and caught in C++ shouldn't bubble up to JS
        const doc = mx.createDocument();
        const nodeGraph1 = doc.addNodeGraph();
        const nodeGraph2 = doc.addNodeGraph();
        nodeGraph1.setInheritsFrom(nodeGraph2);
        nodeGraph2.setInheritsFrom(nodeGraph1);
        expect(nodeGraph1.hasInheritanceCycle()).to.not.throw;
        expect(nodeGraph1.hasInheritanceCycle()).to.be.true;

        // Exceptions that are not caught in C++ should throw with an exception pointer
        nodeGraph1.addNode('node', 'node1');
        expect(() => { nodeGraph1.addNode('node', 'node1'); }).to.throw;
        try
        {
            nodeGraph1.addNode('node', 'node1');
        } catch (errPtr)
        {
            expect(errPtr).to.be.a('number');
            expect(mx.getExceptionMessage(errPtr)).to.be.a('string');
        }
    });

    it('getReferencedSourceUris', async () =>
    {
        const doc = mx.createDocument();
        const filename = 'standard_surface_look_brass_tiled.mtlx';
        await mx.readFromXmlFile(doc, filename, examplesPath);
        const sourceUris = doc.getReferencedSourceUris();
        expect(sourceUris).to.be.instanceof(Array);
        expect(sourceUris.length).to.equal(3);
        expect(sourceUris[0]).to.be.a('string');
        expect(sourceUris.includes('standard_surface_brass_tiled.mtlx')).to.be.true;
    });

    it('Should invoke correct instance of \'validate\'', () =>
    {
        // We check whether the correct function is called by provoking an error message that is specific to the
        // function that we expect to be called.
        const message = {};

        // Should invoke Document::validate.
        const doc = mx.createDocument();
        expect(doc.validate()).to.be.true;
        doc.removeAttribute(mx.InterfaceElement.VERSION_ATTRIBUTE);
        expect(doc.validate()).to.be.true;

        // Should invoke Node::validate
        const node = doc.addNode('node');
        expect(node.validate()).to.be.true;
        node.setCategory('');
        expect(node.validate()).to.be.false;
        expect(node.validate(message)).to.be.false;
        expect(message.message).to.include('Node element is missing a category');

        // Should invoke inherited ValueElement::validate
        const token = new mx.Token(node, 'token');
        expect(token.validate()).to.be.true;
        token.setUnitType('bogus');
        expect(token.validate()).to.be.false;
        expect(token.validate(message)).to.be.false;
        expect(message.message).to.include('Unit type definition does not exist in document')
    });

    it('StringResolver name substitution getters', () =>
    {
        const fnTestData = {
            fnKey: 'fnValue',
            fnKey1: 'fnValue1'
        };
        const fnTestKeys = Object.keys(fnTestData);

        const gnTestData = {
            gnKey: 'gnValue',
            gnKey1: 'gnValue1'
        };
        const gnTestKeys = Object.keys(gnTestData);

        const resolver = mx.StringResolver.create();

        resolver.setFilenameSubstitution(fnTestKeys[0], fnTestData[fnTestKeys[0]]);
        resolver.setFilenameSubstitution(fnTestKeys[1], fnTestData[fnTestKeys[1]]);
        const fnSubs = resolver.getFilenameSubstitutions();
        expect(fnSubs).to.be.instanceof(Object);
        expect(Object.keys(fnSubs).length).to.equal(2);
        expect(fnSubs).to.deep.equal(fnTestData);

        resolver.setGeomNameSubstitution(gnTestKeys[0], gnTestData[gnTestKeys[0]]);
        resolver.setGeomNameSubstitution(gnTestKeys[1], gnTestData[gnTestKeys[1]]);
        const gnSubs = resolver.getGeomNameSubstitutions();
        expect(gnSubs).to.be.instanceof(Object);
        expect(Object.keys(gnSubs).length).to.equal(2);
        expect(gnSubs).to.deep.equal(gnTestData);
    });

    it('getShaderNodes', async () =>
    {
        const doc = mx.createDocument();
        const fileNames = ['standard_surface_marble_solid.mtlx'];
        const mtlxStrs = getMtlxStrings(fileNames, examplesPath);
        await mx.readFromXmlString(doc, mtlxStrs[0]);
        let matNodes = doc.getMaterialNodes();
        expect(matNodes.length).to.equal(1);
        const matNode = matNodes[0];

        // Should return a surface shader node but no displacement shader node
        let shaderNodes = mx.getShaderNodes(matNode);
        expect(shaderNodes).to.be.instanceof(Array);
        expect(shaderNodes.length).to.equal(1);
        expect(shaderNodes[0].getType()).to.equal(mx.SURFACE_SHADER_TYPE_STRING);
        shaderNodes = mx.getShaderNodes(matNode, mx.DISPLACEMENT_SHADER_TYPE_STRING);
        expect(shaderNodes).to.be.instanceof(Array);
        expect(shaderNodes.length).to.equal(0);
    });

    it('createValidName', () =>
    {
        const testString = '_Note_:Please,turn.this+-into*1#valid\nname for_me';
        const replaceRegex = /[^a-zA-Z0-9_:]/g
        expect(mx.createValidName(testString)).to.equal(testString.replace(replaceRegex, '_'));
        expect(mx.createValidName(testString, '-')).to.equal(testString.replace(replaceRegex, '-'));
    });

    it('getVersionIntegers', () =>
    {
        const versionStringArr = mx.getVersionString().split('.').map((value) => parseInt(value, 10));

        // Global getVersionIntegers
        const globalVersion = mx.getVersionIntegers();
        expect(globalVersion).to.be.instanceof(Array);
        expect(globalVersion.length).to.equal(3);
        expect(globalVersion).to.deep.equal(versionStringArr);

        // Document.getVersionIntegers
        versionStringArr.pop();
        const doc = mx.createDocument();
        const docVersion = doc.getVersionIntegers();
        expect(docVersion).to.be.instanceof(Array);
        expect(docVersion.length).to.equal(2);
        expect(docVersion).to.deep.equal(versionStringArr);

        // InterfaceElement.getVersionIntegers (via NodeDef)
        // TODO: This function can currently not be called, since we have a linker issue that messes up this function.
    });
});
