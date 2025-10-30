/*
 * WebXR Device API
 */

/**
 * @type {XR}
 */
Navigator.prototype.xr;

/**
 * @constructor
 */
function XRSessionInit() {};

/**
 * @type {Array<string>}
 */
XRSessionInit.prototype.requiredFeatures;

/**
 * @type {Array<string>}
 */
XRSessionInit.prototype.optionalFeatures;

/**
 * @constructor
 */
function XR() {}

/**
 * @type {?function (Event)}
 */
XR.prototype.ondevicechanged;

/**
 * @param {string} mode
 *
 * @return {!Promise<boolean>}
 */
XR.prototype.isSessionSupported = function(mode) {};

/**
 * @param {string} mode
 * @param {XRSessionInit} options
 *
 * @return {!Promise<XRSession>}
 */
XR.prototype.requestSession = function(mode, options) {};

/**
 * @constructor
 */
function XRSession() {}

/**
 * @type {XRRenderState}
 */
XRSession.prototype.renderState;

/**
 * @type {Array<XRInputSource>}
 */
XRSession.prototype.inputSources;

/**
 * @type {string}
 */
XRSession.prototype.visibilityState;

/**
 * @type {?number}
 */
XRSession.prototype.frameRate;

/**
 * @type {?Float32Array}
 */
XRSession.prototype.supportedFrameRates;

/**
 * @type {Array<string>}
 */
XRSession.prototype.enabledFeatures;

/**
 * @type {string}
 */
XRSession.prototype.environmentBlendMode;

/**
 * @type {?function (Event)}
 */
XRSession.prototype.onend;

/**
 * @type {?function (XRInputSourcesChangeEvent)}
 */
XRSession.prototype.oninputsourceschange;

/**
 * @type {?function (XRInputSourceEvent)}
 */
XRSession.prototype.onselectstart;

/**
 * @type {?function (XRInputSourceEvent)}
 */
XRSession.prototype.onselect;

/**
 * @type {?function (XRInputSourceEvent)}
 */
XRSession.prototype.onselectend;

/**
 * @type {?function (XRInputSourceEvent)}
 */
XRSession.prototype.onsqueezestart;

/**
 * @type {?function (XRInputSourceEvent)}
 */
XRSession.prototype.onsqueeze;

/**
 * @type {?function (XRInputSourceEvent)}
 */
XRSession.prototype.onsqueezeend;

/**
 * @type {?function (Event)}
 */
XRSession.prototype.onvisibilitychange;

/**
 * @param {XRRenderStateInit} state
 * @return {void}
 */
XRSession.prototype.updateRenderState = function (state) {};

/**
 * @param {XRFrameRequestCallback} callback
 * @return {number}
 */
XRSession.prototype.requestAnimationFrame = function (callback) {};

/**
 * @param {number} handle
 * @return {void}
 */
XRSession.prototype.cancelAnimationFrame = function (handle) {};

/**
 * @return {Promise<void>}
 */
XRSession.prototype.end = function () {};

/**
 * @param {string} referenceSpaceType
 * @return {Promise<XRReferenceSpace>}
 */
XRSession.prototype.requestReferenceSpace = function (referenceSpaceType) {};

/**
 * @param {number} rate
 * @return {Promise<undefined>}
 */
XRSession.prototype.updateTargetFrameRate = function (rate) {};

/**
 * @typedef {function(number, XRFrame): undefined}
 */
var XRFrameRequestCallback;

/**
 * @constructor
 */
function XRRenderStateInit() {}

/**
 * @type {number}
 */
XRRenderStateInit.prototype.depthNear;

/**
 * @type {number}
 */
XRRenderStateInit.prototype.depthFar;

/**
 * @type {number}
 */
XRRenderStateInit.prototype.inlineVerticalFieldOfView;

/**
 * @type {?XRWebGLLayer}
 */
XRRenderStateInit.prototype.baseLayer;

/**
 * @constructor
 */
function XRRenderState() {};

/**
 * @type {number}
 */
XRRenderState.prototype.depthNear;

/**
 * @type {number}
 */
XRRenderState.prototype.depthFar;

/**
 * @type {?number}
 */
XRRenderState.prototype.inlineVerticalFieldOfView;

/**
 * @type {?XRWebGLLayer}
 */
XRRenderState.prototype.baseLayer;

/**
 * @constructor
 */
function XRFrame() {}

/**
 * @type {XRSession}
 */
XRFrame.prototype.session;

/**
 * @param {XRReferenceSpace} referenceSpace
 * @return {?XRViewerPose}
 */
XRFrame.prototype.getViewerPose = function (referenceSpace) {};

/**
 * @param {XRSpace} space
 * @param {XRSpace} baseSpace
 * @return {XRPose}
 */
XRFrame.prototype.getPose = function (space, baseSpace) {};

/**
 * @param {Array<XRSpace>} spaces
 * @param {XRSpace} baseSpace
 * @param {Float32Array} transforms
 * @return {boolean}
 */
XRFrame.prototype.fillPoses = function (spaces, baseSpace, transforms) {};

/**
 * @param {Array<XRJointSpace>} jointSpaces
 * @param {Float32Array} radii
 * @return {boolean}
 */
XRFrame.prototype.fillJointRadii = function (jointSpaces, radii) {};

/**
 * @constructor
 */
function XRReferenceSpace() {};

/**
 * @type {Array<DOMPointReadOnly>}
 */
XRReferenceSpace.prototype.boundsGeometry;

/**
 * @param {XRRigidTransform} originOffset
 * @return {XRReferenceSpace}
 */
XRReferenceSpace.prototype.getOffsetReferenceSpace = function(originOffset) {};

/**
 * @type {?function (Event)}
 */
XRReferenceSpace.prototype.onreset;

/**
 * @constructor
 */
function XRRigidTransform() {};

/**
 * @type {DOMPointReadOnly}
 */
XRRigidTransform.prototype.position;

/**
 * @type {DOMPointReadOnly}
 */
XRRigidTransform.prototype.orientation;

/**
 * @type {Float32Array}
 */
XRRigidTransform.prototype.matrix;

/**
 * @type {XRRigidTransform}
 */
XRRigidTransform.prototype.inverse;

/**
 * @constructor
 */
function XRView() {}

/**
 * @type {string}
 */
XRView.prototype.eye;

/**
 * @type {Float32Array}
 */
XRView.prototype.projectionMatrix;

/**
 * @type {XRRigidTransform}
 */
XRView.prototype.transform;

/**
 * @constructor
 */
function XRViewerPose() {}

/**
 * @type {Array<XRView>}
 */
XRViewerPose.prototype.views;

/**
 * @constructor
 */
function XRViewport() {}

/**
 * @type {number}
 */
XRViewport.prototype.x;

/**
 * @type {number}
 */
XRViewport.prototype.y;

/**
 * @type {number}
 */
XRViewport.prototype.width;

/**
 * @type {number}
 */
XRViewport.prototype.height;

/**
 * @constructor
 */
function XRWebGLLayerInit() {};

/**
 * @type {boolean}
 */
XRWebGLLayerInit.prototype.antialias;

/**
 * @type {boolean}
 */
XRWebGLLayerInit.prototype.depth;

/**
 * @type {boolean}
 */
XRWebGLLayerInit.prototype.stencil;

/**
 * @type {boolean}
 */
XRWebGLLayerInit.prototype.alpha;

/**
 * @type {boolean}
 */
XRWebGLLayerInit.prototype.ignoreDepthValues;

/**
 * @type {boolean}
 */
XRWebGLLayerInit.prototype.ignoreDepthValues;

/**
 * @type {number}
 */
XRWebGLLayerInit.prototype.framebufferScaleFactor;

/**
 * @constructor
 *
 * @param {XRSession} session
 * @param {WebGLRenderContext|WebGL2RenderingContext} ctx
 * @param {?XRWebGLLayerInit} options
 */
function XRWebGLLayer(session, ctx, options) {}

/**
 * @type {boolean}
 */
XRWebGLLayer.prototype.antialias;

/**
 * @type {boolean}
 */
XRWebGLLayer.prototype.ignoreDepthValues;

/**
 * @type {number}
 */
XRWebGLLayer.prototype.framebufferWidth;

/**
 * @type {number}
 */
XRWebGLLayer.prototype.framebufferHeight;

/**
 * @type {WebGLFramebuffer}
 */
XRWebGLLayer.prototype.framebuffer;

/**
 * @param {XRView} view
 * @return {?XRViewport}
 */
XRWebGLLayer.prototype.getViewport = function(view) {};

/**
 * @param {XRSession} session
 * @return {number}
 */
XRWebGLLayer.prototype.getNativeFramebufferScaleFactor = function (session) {};

/**
 * @constructor
 */
function WebGLRenderingContextBase() {};

/**
 * @return {Promise<void>}
 */
WebGLRenderingContextBase.prototype.makeXRCompatible = function () {};

/**
 * @constructor
 */
function XRInputSourcesChangeEvent() {};

/**
 * @type {Array<XRInputSource>}
 */
XRInputSourcesChangeEvent.prototype.added;

/**
 * @type {Array<XRInputSource>}
 */
XRInputSourcesChangeEvent.prototype.removed;

/**
 * @constructor
 */
function XRInputSourceEvent() {};

/**
 * @type {XRFrame}
 */
XRInputSourceEvent.prototype.frame;

/**
 * @type {XRInputSource}
 */
XRInputSourceEvent.prototype.inputSource;

/**
 * @constructor
 */
function XRInputSource() {};

/**
 * @type {Gamepad}
 */
XRInputSource.prototype.gamepad;

/**
 * @type {XRSpace}
 */
XRInputSource.prototype.gripSpace;

/**
 * @type {string}
 */
XRInputSource.prototype.handedness;

/**
 * @type {string}
 */
XRInputSource.prototype.profiles;

/**
 * @type {string}
 */
XRInputSource.prototype.targetRayMode;

/**
 * @type {XRSpace}
 */
XRInputSource.prototype.targetRaySpace;

/**
 * @type {?XRHand}
 */
XRInputSource.prototype.hand;

/**
 * @constructor
 */
function XRHand() {};

/**
 * Note: In fact, XRHand acts like a Map<string, XRJointSpace>, but I don't know
 * how to represent that here. So, we're just giving the one method we call.
 *
 * @return {Array<XRJointSpace>}
 */
XRHand.prototype.values = function () {};

/**
 * @type {number}
 */
XRHand.prototype.size;

/**
 * @param {string} key
 * @return {XRJointSpace}
 */
XRHand.prototype.get = function (key) {};

/**
 * @constructor
 */
function XRSpace() {};

/**
 * @constructor
 * @extends {XRSpace}
 */
function XRJointSpace() {};

/**
 * @type {string}
 */
XRJointSpace.prototype.jointName;

/**
 * @constructor
 */
function XRPose() {};

/**
 * @type {XRRigidTransform}
 */
XRPose.prototype.transform;

/**
 * @type {boolean}
 */
XRPose.prototype.emulatedPosition;

/*
 * WebXR Layers API Level 1
 */

/**
 * @constructor XRLayer
 */
function XRLayer() {}

/**
 * @constructor XRLayerEventInit
 */
function XRLayerEventInit() {}

/**
 * @type {XRLayer}
 */
XRLayerEventInit.prototype.layer;

/**
 * @constructor XRLayerEvent
 *
 * @param {string} type
 * @param {XRLayerEventInit} init
 */
function XRLayerEvent(type, init) {};

/**
 * @type {XRLayer}
 */
XRLayerEvent.prototype.layer;

/**
 * @constructor XRCompositionLayer
 * @extends {XRLayer}
 */
function XRCompositionLayer() {};

/**
 * @type {string}
 */
XRCompositionLayer.prototype.layout;

/**
 * @type {boolean}
 */
XRCompositionLayer.prototype.blendTextureAberrationCorrection;

/**
 * @type {?boolean}
 */
XRCompositionLayer.prototype.chromaticAberrationCorrection;

/**
 * @type {boolean}
 */
XRCompositionLayer.prototype.forceMonoPresentation;

/**
 * @type {number}
 */
XRCompositionLayer.prototype.opacity;

/**
 * @type {number}
 */
XRCompositionLayer.prototype.mipLevels;

/**
 * @type {boolean}
 */
XRCompositionLayer.prototype.needsRedraw;

/**
 * @return {void}
 */
XRCompositionLayer.prototype.destroy = function () {};

/**
 * @constructor XRProjectionLayer
 * @extends {XRCompositionLayer}
 */
function XRProjectionLayer() {}

/**
 * @type {number}
 */
XRProjectionLayer.prototype.textureWidth;

/**
 * @type {number}
 */
XRProjectionLayer.prototype.textureHeight;

/**
 * @type {number}
 */
XRProjectionLayer.prototype.textureArrayLength;

/**
 * @type {boolean}
 */
XRProjectionLayer.prototype.ignoreDepthValues;

/**
 * @type {?number}
 */
XRProjectionLayer.prototype.fixedFoveation;

/**
 * @type {XRRigidTransform}
 */
XRProjectionLayer.prototype.deltaPose;

/**
 * @constructor XRQuadLayer
 * @extends {XRCompositionLayer}
 */
function XRQuadLayer() {}

/**
 * @type {XRSpace}
 */
XRQuadLayer.prototype.space;

/**
 * @type {XRRigidTransform}
 */
XRQuadLayer.prototype.transform;

/**
 * @type {number}
 */
XRQuadLayer.prototype.width;

/**
 * @type {number}
 */
XRQuadLayer.prototype.height;

/**
 * @type {?function (XRLayerEvent)}
 */
XRQuadLayer.prototype.onredraw;

/**
 * @constructor XRCylinderLayer
 * @extends {XRCompositionLayer}
 */
function XRCylinderLayer() {}

/**
 * @type {XRSpace}
 */
XRCylinderLayer.prototype.space;

/**
 * @type {XRRigidTransform}
 */
XRCylinderLayer.prototype.transform;

/**
 * @type {number}
 */
XRCylinderLayer.prototype.radius;

/**
 * @type {number}
 */
XRCylinderLayer.prototype.centralAngle;

/**
 * @type {number}
 */
XRCylinderLayer.prototype.aspectRatio;

/**
 * @type {?function (XRLayerEvent)}
 */
XRCylinderLayer.prototype.onredraw;

/**
 * @constructor XREquirectLayer
 * @extends {XRCompositionLayer}
 */
function XREquirectLayer() {}

/**
 * @type {XRSpace}
 */
XREquirectLayer.prototype.space;

/**
 * @type {XRRigidTransform}
 */
XREquirectLayer.prototype.transform;

/**
 * @type {number}
 */
XREquirectLayer.prototype.radius;

/**
 * @type {number}
 */
XREquirectLayer.prototype.centralHorizontalAngle;

/**
 * @type {number}
 */
XREquirectLayer.prototype.upperVerticalAngle;

/**
 * @type {number}
 */
XREquirectLayer.prototype.lowerVerticalAngle;

/**
 * @type {?function (XRLayerEvent)}
 */
XREquirectLayer.prototype.onredraw;

/**
 * @constructor XRCubeLayer
 * @extends {XRCompositionLayer}
 */
function XRCubeLayer() {}

/**
 * @type {XRSpace}
 */
XRCubeLayer.prototype.space;

/**
 * @type {DOMPointReadOnly}
 */
XRCubeLayer.prototype.orientation;

/**
 * @type {?function (XRLayerEvent)}
 */
XRCubeLayer.prototype.onredraw;

/**
 * @constructor XRSubImage
 */
function XRSubImage() {}

/**
 * @type {XRViewport}
 */
XRSubImage.prototype.viewport;

/**
 * @constructor XRWebGLSubImage
 * @extends {XRSubImage}
 */
function XRWebGLSubImage () {}

/**
 * @type {WebGLTexture}
 */
XRWebGLSubImage.prototype.colorTexture;

/**
 * @type {?WebGLTexture}
 */
XRWebGLSubImage.prototype.depthStencilTexture;

/**
 * @type {?WebGLTexture}
 */
XRWebGLSubImage.prototype.motionVectorTexture;

/**
 * @type {?number}
 */
XRWebGLSubImage.prototype.imageIndex;

/**
 * @type {number}
 */
XRWebGLSubImage.prototype.colorTextureWidth;

/**
 * @type {number}
 */
XRWebGLSubImage.prototype.colorTextureHeight;

/**
 * @type {?number}
 */
XRWebGLSubImage.prototype.depthStencilTextureWidth;

/**
 * @type {?number}
 */
XRWebGLSubImage.prototype.depthStencilTextureHeight;

/**
 * @type {?number}
 */

XRWebGLSubImage.prototype.motionVectorTextureWidth;

/**
 * @type {?number}
 */
XRWebGLSubImage.prototype.motionVectorTextureHeight;

/**
 * @constructor XRProjectionLayerInit
 */
function XRProjectionLayerInit() {}

/**
 * @type {string}
 */
XRProjectionLayerInit.prototype.textureType;

/**
 * @type {number}
 */
XRProjectionLayerInit.prototype.colorFormat;

/**
 * @type {number}
 */
XRProjectionLayerInit.prototype.depthFormat;

/**
 * @type {number}
 */
XRProjectionLayerInit.prototype.scaleFactor;

/**
 * @constructor XRLayerInit
 */
function XRLayerInit() {}

/**
 * @type {XRSpace}
 */
XRLayerInit.prototype.space;

/**
 * @type {number}
 */
XRLayerInit.prototype.colorFormat;

/**
 * @type {number}
 */
XRLayerInit.prototype.depthFormat;

/**
 * @type {number}
 */
XRLayerInit.prototype.mipLevels;

/**
 * @type {number}
 */
XRLayerInit.prototype.viewPixelWidth;

/**
 * @type {number}
 */
XRLayerInit.prototype.viewPixelHeight;

/**
 * @type {string}
 */
XRLayerInit.prototype.layout;

/**
 * @type {boolean}
 */
XRLayerInit.prototype.isStatic;

/**
 * @constructor XRQuadLayerInit
 * @extends {XRLayerInit}
 */
function XRQuadLayerInit() {}

/**
 * @type {string}
 */
XRQuadLayerInit.prototype.textureType;

/**
 * @type {?XRRigidTransform}
 */
XRQuadLayerInit.prototype.transform;

/**
 * @type {number}
 */
XRQuadLayerInit.prototype.width;

/**
 * @type {number}
 */
XRQuadLayerInit.prototype.height;

/**
 * @constructor XRCylinderLayerInit
 * @extends {XRLayerInit}
 */
function XRCylinderLayerInit() {}

/**
 * @type {string}
 */
XRCylinderLayerInit.prototype.textureType;

/**
 * @type {?XRRigidTransform}
 */
XRCylinderLayerInit.prototype.transform;

/**
 * @type {number}
 */
XRCylinderLayerInit.prototype.radius;

/**
 * @type {number}
 */
XRCylinderLayerInit.prototype.centralAngle;

/**
 * @type {number}
 */
XRCylinderLayerInit.prototype.aspectRatio;

/**
 * @constructor XREquirectLayerInit
 * @extends {XRLayerInit}
 */
function XREquirectLayerInit() {}

/**
 * @type {string}
 */
XREquirectLayerInit.prototype.textureType;

/**
 * @type {?XRRigidTransform}
 */
XREquirectLayerInit.prototype.transform;

/**
 * @type {number}
 */
XREquirectLayerInit.prototype.radius;

/**
 * @type {number}
 */
XREquirectLayerInit.prototype.centralHorizontalAngle;

/**
 * @type {number}
 */
XREquirectLayerInit.prototype.upperVerticalAngle;

/**
 * @type {number}
 */
XREquirectLayerInit.prototype.lowerVerticalAngle;

/**
 * @constructor XRCubeLayerInit
 * @extends {XRLayerInit}
 */
function XRCubeLayerInit() {}

/**
 * @type {DOMPointReadOnly}
 */
XRCubeLayerInit.prototype.orientation;

/**
 * @constructor XRWebGLBinding
 *
 * @param {XRSession} session
 * @param {WebGLRenderContext|WebGL2RenderingContext} context
 */
function XRWebGLBinding(session, context) {}

/**
 * @type {number}
 */
XRWebGLBinding.prototype.nativeProjectionScaleFactor;

/**
 * @type {number}
 */
XRWebGLBinding.prototype.usesDepthValues;

/**
 * @param {XRProjectionLayerInit} init
 * @return {XRProjectionLayer}
 */
XRWebGLBinding.prototype.createProjectionLayer = function (init) {};

/**
 * @param {XRQuadLayerInit} init
 * @return {XRQuadLayer}
 */
XRWebGLBinding.prototype.createQuadLayer = function (init) {};

/**
 * @param {XRCylinderLayerInit} init
 * @return {XRCylinderLayer}
 */
XRWebGLBinding.prototype.createCylinderLayer = function (init) {};

/**
 * @param {XREquirectLayerInit} init
 * @return {XREquirectLayer}
 */
XRWebGLBinding.prototype.createEquirectLayer = function (init) {};

/**
 * @param {XRCubeLayerInit} init
 * @return {XRCubeLayer}
 */
XRWebGLBinding.prototype.createCubeLayer = function (init) {};

/**
 * @param {XRCompositionLayer} layer
 * @param {XRFrame} frame
 * @param {string} eye
 * @return {XRWebGLSubImage}
 */
XRWebGLBinding.prototype.getSubImage = function (layer, frame, eye) {};

/**
 * @param {XRProjectionLayer} layer
 * @param {XRView} view
 * @return {XRWebGLSubImage}
 */
XRWebGLBinding.prototype.getViewSubImage = function (layer, view) {};

/**
 * @constructor XRMediaLayerInit
 */
function XRMediaLayerInit() {}

/**
 * @type {XRSpace}
 */
XRMediaLayerInit.prototype.space;

/**
 * @type {string}
 */
XRMediaLayerInit.prototype.layout;

/**
 * @type {boolean}
 */
XRMediaLayerInit.prototype.invertStereo;

/**
 * @constructor XRMediaQuadLayerInit
 * @extends {XRMediaLayerInit}
 */
function XRMediaQuadLayerInit() {}

/**
 * @type {XRRigidTransform}
 */
XRMediaQuadLayerInit.prototype.transform;

/**
 * @type {number}
 */
XRMediaQuadLayerInit.prototype.width;

/**
 * @type {number}
 */
XRMediaQuadLayerInit.prototype.height;

/**
 * @constructor XRMediaCylinderLayerInit
 * @extends {XRMediaLayerInit}
 */
function XRMediaCylinderLayerInit() {}

/**
 * @type {XRRigidTransform}
 */
XRMediaCylinderLayerInit.prototype.transform;

/**
 * @type {number}
 */
XRMediaCylinderLayerInit.prototype.radius;

/**
 * @type {number}
 */
XRMediaCylinderLayerInit.prototype.centralAngle;

/**
 * @type {?number}
 */
XRMediaCylinderLayerInit.prototype.aspectRatio;

/**
 * @constructor XRMediaEquirectLayerInit
 * @extends {XRMediaLayerInit}
 */
function XRMediaEquirectLayerInit() {}

/**
 * @type {XRRigidTransform}
 */
XRMediaEquirectLayerInit.prototype.transform;

/**
 * @type {number}
 */
XRMediaEquirectLayerInit.prototype.radius;

/**
 * @type {number}
 */
XRMediaEquirectLayerInit.prototype.centralHorizontalAngle;

/**
 * @type {number}
 */
XRMediaEquirectLayerInit.prototype.upperVerticalAngle;

/**
 * @type {number}
 */
XRMediaEquirectLayerInit.prototype.lowerVerticalAngle;

/**
 * @constructor XRMediaBinding
 *
 * @param {XRSession} session
 */
function XRMediaBinding(session) {}

/**
 * @param {HTMLVideoElement} video
 * @param {XRMediaQuadLayerInit} init
 * @return {XRQuadLayer}
 */
XRMediaBinding.prototype.createQuadLayer = function(video, init) {};

/**
 * @param {HTMLVideoElement} video
 * @param {XRMediaCylinderLayerInit} init
 * @return {XRCylinderLayer}
 */
XRMediaBinding.prototype.createCylinderLayer = function(video, init) {};

/**
 * @param {HTMLVideoElement} video
 * @param {XRMediaEquirectLayerInit} init
 * @return {XREquirectLayer}
 */
XRMediaBinding.prototype.createEquirectLayer = function(video, init) {};

/**
 * @type {Array<XRLayer>}
 */
XRRenderState.prototype.layers;
