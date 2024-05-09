// MaterialX is served through a script tag in the test setup.

function createStandardSurfaceMaterial(mx)
{
    const doc = mx.createDocument();
    const ssName = 'SR_default';
    const ssNode = doc.addChildOfCategory('standard_surface', ssName);
    ssNode.setType('surfaceshader');
    const smNode = doc.addChildOfCategory('surfacematerial', 'Default');
    smNode.setType('material');
    const shaderElement = smNode.addInput('surfaceshader');
    shaderElement.setType('surfaceshader');
    shaderElement.setNodeName(ssName);
    expect(doc.validate()).to.be.true;
    return doc;
}

describe('Generate ESSL Shaders', function ()
{
    let mx;
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');

    this.timeout(60000);

    before(async function ()
    {
        mx = await MaterialX();
    });

    it('Compile Shaders', () =>
    {
        const doc = createStandardSurfaceMaterial(mx);

        const gen = new mx.EsslShaderGenerator();
        const genContext = new mx.GenContext(gen);
        const stdlib = mx.loadStandardLibraries(genContext);

        doc.importLibrary(stdlib);

        const elem = mx.findRenderableElement(doc);
        try
        {
            const mxShader = gen.generate(elem.getNamePath(), elem, genContext);

            const fShader = mxShader.getSourceCode("pixel");
            const vShader = mxShader.getSourceCode("vertex");

            const glVertexShader = gl.createShader(gl.VERTEX_SHADER);
            gl.shaderSource(glVertexShader, vShader);
            gl.compileShader(glVertexShader);
            if (!gl.getShaderParameter(glVertexShader, gl.COMPILE_STATUS))
            {
                console.error("-------- VERTEX SHADER FAILED TO COMPILE: ----------------");
                console.error("--- VERTEX SHADER LOG ---");
                console.error(gl.getShaderInfoLog(glVertexShader));
                console.error("--- VERTEX SHADER START ---");
                console.error(fShader);
                console.error("--- VERTEX SHADER END ---");
            }
            expect(gl.getShaderParameter(glVertexShader, gl.COMPILE_STATUS)).to.equal(true);

            const glPixelShader = gl.createShader(gl.FRAGMENT_SHADER);
            gl.shaderSource(glPixelShader, fShader);
            gl.compileShader(glPixelShader);
            if (!gl.getShaderParameter(glPixelShader, gl.COMPILE_STATUS))
            {
                console.error("-------- PIXEL SHADER FAILED TO COMPILE: ----------------");
                console.error("--- PIXEL SHADER LOG ---");
                console.error(gl.getShaderInfoLog(glPixelShader));
                console.error("--- PIXEL SHADER START ---");
                console.error(fShader);
                console.error("--- PIXEL SHADER END ---");
            }
            expect(gl.getShaderParameter(glPixelShader, gl.COMPILE_STATUS)).to.equal(true);
        }
        catch (errPtr)
        {
            console.error("-------- Failed code generation: ----------------");
            console.error(mx.getExceptionMessage(errPtr));
        }
    });
});
