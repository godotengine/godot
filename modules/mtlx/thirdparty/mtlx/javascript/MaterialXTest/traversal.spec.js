import { expect } from 'chai';
import Module from './_build/JsMaterialXCore.js';

describe('Traversal', () =>
{
    let mx;
    before(async () =>
    {
        mx = await Module();
    });

    it('Traverse Graph', () =>
    {
        // Create a document.
        const doc = mx.createDocument();
        // Create a node graph with the following structure:
        //
        // [image1] [constant]     [image2]
        //        \ /                 |
        //    [multiply]          [contrast]         [noise3d]
        //             \____________  |  ____________/
        //                          [mix]
        //                            |
        //                         [output]
        //
        const nodeGraph = doc.addNodeGraph();
        const image1 = nodeGraph.addNode('image');
        const image2 = nodeGraph.addNode('image');
        const constant = nodeGraph.addNode('constant');
        const multiply = nodeGraph.addNode('multiply');
        const contrast = nodeGraph.addNode('contrast');
        const noise3d = nodeGraph.addNode('noise3d');
        const mix = nodeGraph.addNode('mix');
        const output = nodeGraph.addOutput();
        multiply.setConnectedNode('in1', image1);
        multiply.setConnectedNode('in2', constant);
        contrast.setConnectedNode('in', image2);
        mix.setConnectedNode('fg', multiply);
        mix.setConnectedNode('bg', contrast);
        mix.setConnectedNode('mask', noise3d);
        output.setConnectedNode(mix);

        expect(doc.validate()).to.be.true;

        // Traverse the document tree (implicit iterator).
        let nodeCount = 0;
        for (let elem of doc.traverseTree())
        {
            if (elem instanceof mx.Node)
            {
                nodeCount++;
            }
        }
        expect(nodeCount).to.equal(7);

        // Traverse the document tree (explicit iterator)
        let treeIter = doc.traverseTree();
        nodeCount = 0;
        let maxElementDepth = 0;
        for (let elem of treeIter)
        {
            if (elem instanceof mx.Node)
            {
                nodeCount++;
            }
            maxElementDepth = Math.max(maxElementDepth, treeIter.getElementDepth());
        }
        expect(nodeCount).to.equal(7);
        expect(maxElementDepth).to.equal(3);

        // Traverse the document tree (prune subtree).
        nodeCount = 0;
        treeIter = doc.traverseTree();
        for (let elem of treeIter)
        {
            if (elem instanceof mx.Node)
            {
                nodeCount++;
            }
            if (elem instanceof mx.NodeGraph)
            {
                treeIter.setPruneSubtree(true);
            }
        }
        expect(nodeCount).to.equal(0);

        // Traverse upstream from the graph output (implicit iterator)
        nodeCount = 0;
        for (let edge of output.traverseGraph())
        {
            const upstreamElem = edge.getUpstreamElement();
            const connectingElem = edge.getConnectingElement();
            const downstreamElem = edge.getDownstreamElement();
            if (upstreamElem instanceof mx.Node)
            {
                nodeCount++;
                if (downstreamElem instanceof mx.Node)
                {
                    expect(connectingElem instanceof mx.Input).to.be.true;
                }
            }
        }
        expect(nodeCount).to.equal(7);

        // Traverse upstream from the graph output (explicit iterator)
        nodeCount = 0;
        maxElementDepth = 0;
        let maxNodeDepth = 0;
        let graphIter = output.traverseGraph();
        for (let edge of graphIter)
        {
            const upstreamElem = edge.getUpstreamElement();
            if (upstreamElem instanceof mx.Node)
            {
                nodeCount++;
            }
            maxElementDepth = Math.max(maxElementDepth, graphIter.getElementDepth());
            maxNodeDepth = Math.max(maxNodeDepth, graphIter.getNodeDepth());
        }
        expect(nodeCount).to.equal(7);
        expect(maxElementDepth).to.equal(3);
        expect(maxNodeDepth).to.equal(3);

        // Traverse upstream from the graph output (prune subgraph)
        nodeCount = 0;
        graphIter = output.traverseGraph();
        for (let edge of graphIter)
        {
            const upstreamElem = edge.getUpstreamElement();
            expect(upstreamElem.getSelf()).to.be.an.instanceof(mx.Element);
            if (upstreamElem instanceof mx.Node)
            {
                nodeCount++;
            }
            if (upstreamElem.getCategory() === 'multiply')
            {
                graphIter.setPruneSubgraph(true);
            }
        }
        expect(nodeCount).to.equal(5);

        // Create and detect a cycle
        multiply.setConnectedNode('in2', mix);
        expect(output.hasUpstreamCycle()).to.be.true;
        expect(doc.validate()).to.be.false;
        multiply.setConnectedNode('in2', constant);
        expect(output.hasUpstreamCycle()).to.be.false;
        expect(doc.validate()).to.be.true;

        // Create and detect a loop
        contrast.setConnectedNode('in', contrast);
        expect(output.hasUpstreamCycle()).to.be.true;
        expect(doc.validate()).to.be.false;
        contrast.setConnectedNode('in', image2);
        expect(output.hasUpstreamCycle()).to.be.false;
        expect(doc.validate()).to.be.true;
    });

    describe("Traverse inheritance", () =>
    {
        let nodeDefInheritanceLevel2, nodeDefInheritanceLevel1, nodeDefParent;
        beforeEach(() =>
        {
            const doc = mx.createDocument();
            nodeDefParent = doc.addNodeDef();
            nodeDefParent.setName('BaseClass');
            nodeDefInheritanceLevel1 = doc.addNodeDef();
            nodeDefInheritanceLevel1.setName('InheritanceLevel1');
            nodeDefInheritanceLevel2 = doc.addNodeDef();
            nodeDefInheritanceLevel2.setName('InheritanceLevel2');
            nodeDefInheritanceLevel2.setInheritsFrom(nodeDefInheritanceLevel1);
            nodeDefInheritanceLevel1.setInheritsFrom(nodeDefParent);
        });

        it('for of loop', () =>
        {
            const inheritanceIterator = nodeDefInheritanceLevel2.traverseInheritance();
            let inheritanceChainLength = 0;
            for (const elem of inheritanceIterator)
            {
                if (elem instanceof mx.NodeDef)
                {
                    inheritanceChainLength++;
                }
            }
            expect(inheritanceChainLength).to.equal(2);;
        });

        it('while loop', () =>
        {
            const inheritanceIterator = nodeDefInheritanceLevel2.traverseInheritance();
            let inheritanceChainLength = 0;
            let elem = inheritanceIterator.next();
            while (!elem.done)
            {
                if (elem.value instanceof mx.NodeDef)
                {
                    inheritanceChainLength++;
                }
                elem = inheritanceIterator.next();
            }
            expect(inheritanceChainLength).to.equal(2);;
        });
    });
});
