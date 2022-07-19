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
 *
 * @param {XRSpace} space
 * @param {XRSpace} baseSpace
 * @return {XRPose}
 */
XRFrame.prototype.getPose = function (space, baseSpace) {};

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
 * @constructor
 */
function XRSpace() {};

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
