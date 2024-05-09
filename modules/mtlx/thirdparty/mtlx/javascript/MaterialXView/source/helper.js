//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

import * as THREE from 'three';

const IMAGE_PROPERTY_SEPARATOR = "_";
const UADDRESS_MODE_SUFFIX = IMAGE_PROPERTY_SEPARATOR + "uaddressmode";
const VADDRESS_MODE_SUFFIX = IMAGE_PROPERTY_SEPARATOR + "vaddressmode";
const FILTER_TYPE_SUFFIX = IMAGE_PROPERTY_SEPARATOR + "filtertype";
const IMAGE_PATH_SEPARATOR = "/";

/**
 * Initialized the environment texture as MaterialX expects it
 * @param {THREE.Texture} texture
 * @param {Object} capabilities
 * @returns {THREE.Texture}
 */
export function prepareEnvTexture(texture, capabilities)
{
    let newTexture = new THREE.DataTexture(texture.image.data, texture.image.width, texture.image.height, texture.format, texture.type);
    newTexture.wrapS = THREE.RepeatWrapping;
    newTexture.anisotropy = capabilities.getMaxAnisotropy();
    newTexture.minFilter = THREE.LinearMipmapLinearFilter;
    newTexture.magFilter = THREE.LinearFilter;
    newTexture.generateMipmaps = true;
    newTexture.needsUpdate = true;

    return newTexture;
}

/**
 * Get Three uniform from MaterialX vector
 * @param {any} value
 * @param {any} dimension
 * @returns {THREE.Uniform}
 */
function fromVector(value, dimension)
{
    let outValue;
    if (value)
    {
        outValue = value.data();
    }
    else
    {
        outValue = [];
        for (let i = 0; i < dimension; ++i)
            outValue.push(0.0);
    }

    return outValue;
}

/**
 * Get Three uniform from MaterialX matrix
 * @param {mx.matrix} matrix
 * @param {mx.matrix.size} dimension
 */
function fromMatrix(matrix, dimension)
{
    let vec = [];
    if (matrix)
    {
        for (let i = 0; i < matrix.numRows(); ++i)
        {
            for (let k = 0; k < matrix.numColumns(); ++k)
            {
                vec.push(matrix.getItem(i, k));
            }
        }
    } else
    {
        for (let i = 0; i < dimension; ++i)
            vec.push(0.0);
    }

    return vec;
}

/**
 * Get Three uniform from MaterialX value
 * @param {mx.Uniform.type} type
 * @param {mx.Uniform.value} value
 * @param {mx.Uniform.name} name
 * @param {mx.Uniforms} uniforms
 * @param {THREE.textureLoader} textureLoader
 */
function toThreeUniform(type, value, name, uniforms, textureLoader, searchPath, flipY)
{
    let outValue;
    switch (type)
    {
        case 'float':
        case 'integer':
        case 'boolean':
            outValue = value;
            break;
        case 'vector2':
            outValue = fromVector(value, 2);
            break;
        case 'vector3':
        case 'color3':
            outValue = fromVector(value, 3);
            break;
        case 'vector4':
        case 'color4':
            outValue = fromVector(value, 4);
            break;
        case 'matrix33':
            outValue = fromMatrix(value, 9);
            break;
        case 'matrix44':
            outValue = fromMatrix(value, 16);
            break;
        case 'filename':
            if (value)
            {
                let fullPath = searchPath + IMAGE_PATH_SEPARATOR + value;
                const texture = textureLoader.load(fullPath);
                // Set address & filtering mode
                if (texture)
                    setTextureParameters(texture, name, uniforms, flipY);
                outValue = texture;
            }
            break;
        case 'samplerCube':
        case 'string':
        default:
            break;
    }

    return outValue;
}

/**
 * Get Three wrapping mode
 * @param {mx.TextureFilter.wrap} mode
 * @returns {THREE.Wrapping}
 */
function getWrapping(mode)
{
    let wrap;
    switch (mode)
    {
        case 1:
            wrap = THREE.ClampToEdgeWrapping;
            break;
        case 2:
            wrap = THREE.RepeatWrapping;
            break;
        case 3:
            wrap = THREE.MirroredRepeatWrapping;
            break;
        default:
            wrap = THREE.RepeatWrapping;
            break;
    }
    return wrap;
}

/**
 * Get Three minification filter
 * @param {mx.TextureFilter.minFilter} type
 * @param {mx.TextureFilter.generateMipmaps} generateMipmaps
 */
function getMinFilter(type, generateMipmaps)
{
    const filterType = generateMipmaps ? THREE.LinearMipMapLinearFilter : THREE.LinearFilter;
    if (type === 0)
    {
        filterType = generateMipmaps ? THREE.NearestMipMapNearestFilter : THREE.NearestFilter;
    }
    return filterType;
}

/**
 * Set Three texture parameters
 * @param {THREE.Texture} texture
 * @param {mx.Uniform.name} name
 * @param {mx.Uniforms} uniforms
 * @param {mx.TextureFilter.generateMipmaps} generateMipmaps
 */
function setTextureParameters(texture, name, uniforms, flipY = true, generateMipmaps = true)
{
    const idx = name.lastIndexOf(IMAGE_PROPERTY_SEPARATOR);
    const base = name.substring(0, idx) || name;

    texture.generateMipmaps = generateMipmaps;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.magFilter = THREE.LinearFilter;
    texture.flipY = flipY;

    if (uniforms.find(base + UADDRESS_MODE_SUFFIX))
    {
        const uaddressmode = uniforms.find(base + UADDRESS_MODE_SUFFIX).getValue().getData();
        texture.wrapS = getWrapping(uaddressmode);
    }

    if (uniforms.find(base + VADDRESS_MODE_SUFFIX))
    {
        const vaddressmode = uniforms.find(base + VADDRESS_MODE_SUFFIX).getValue().getData();
        texture.wrapT = getWrapping(vaddressmode);
    }

    const filterType = uniforms.find(base + FILTER_TYPE_SUFFIX) ? uniforms.get(base + FILTER_TYPE_SUFFIX).value : -1;
    texture.minFilter = getMinFilter(filterType, generateMipmaps);
}

/**
 * Return the global light rotation matrix
 */
export function getLightRotation()
{
    return new THREE.Matrix4().makeRotationY(Math.PI / 2);
}

/**
 * Returns all lights nodes in a MaterialX document
 * @param {mx.Document} doc 
 * @returns {Array.<mx.Node>}
 */
export function findLights(doc)
{
    let lights = [];
    for (let node of doc.getNodes())
    {
        if (node.getType() === "lightshader")
            lights.push(node);
    }
    return lights;
}

/**
 * Register lights in shader generation context
 * @param {Object} mx MaterialX Module
 * @param {Array.<mx.Node>} lights Light nodes
 * @param {mx.GenContext} genContext Shader generation context
 * @returns {Array.<mx.Node>}
 */
export function registerLights(mx, lights, genContext)
{
    mx.HwShaderGenerator.unbindLightShaders(genContext);

    const lightTypesBound = {};
    const lightData = [];
    let lightId = 1;
    for (let light of lights)
    {
        let nodeDef = light.getNodeDef();
        let nodeName = nodeDef.getName();
        if (!lightTypesBound[nodeName])
        {
            lightTypesBound[nodeName] = lightId;
            mx.HwShaderGenerator.bindLightShader(nodeDef, lightId++, genContext);
        }

        const lightDirection = light.getValueElement("direction").getValue().getData().data();
        const lightColor = light.getValueElement("color").getValue().getData().data();
        const lightIntensity = light.getValueElement("intensity").getValue().getData();

        let rotatedLightDirection = new THREE.Vector3(...lightDirection)
        rotatedLightDirection.transformDirection(getLightRotation())

        lightData.push({
            type: lightTypesBound[nodeName],
            direction: rotatedLightDirection,
            color: new THREE.Vector3(...lightColor),
            intensity: lightIntensity
        });
    }

    // Make sure max light count is large enough
    genContext.getOptions().hwMaxActiveLightSources = Math.max(genContext.getOptions().hwMaxActiveLightSources, lights.length);

    return lightData;
}

/**
 * Get uniform values for a shader
 * @param {mx.shaderStage} shaderStage
 * @param {THREE.TextureLoader} textureLoader
 */
export function getUniformValues(shaderStage, textureLoader, searchPath, flipY)
{
    let threeUniforms = {};

    const uniformBlocks = Object.values(shaderStage.getUniformBlocks());
    uniformBlocks.forEach(uniforms =>
    {
        if (!uniforms.empty())
        {
            for (let i = 0; i < uniforms.size(); ++i)
            {
                const variable = uniforms.get(i);
                const value = variable.getValue()?.getData();
                const name = variable.getVariable();
                threeUniforms[name] = new THREE.Uniform(toThreeUniform(variable.getType().getName(), value, name, uniforms,
                    textureLoader, searchPath, flipY));
            }
        }
    });

    return threeUniforms;
}
