
/**
 * @constructor OVR_multiview2
 */
function OVR_multiview2() {}

/**
 * @type {number}
 */
OVR_multiview2.prototype.FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR;

/**
 * @type {number}
 */
OVR_multiview2.prototype.FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR;

/**
 * @type {number}
 */
OVR_multiview2.prototype.MAX_VIEWS_OVR;

/**
 * @type {number}
 */
OVR_multiview2.prototype.FRAMEBUFFER_INCOMPLETE_VIEW_TARGETS_OVR;

/**
 * @param {number} target
 * @param {number} attachment
 * @param {WebGLTexture} texture
 * @param {number} level
 * @param {number} baseViewIndex
 * @param {number} numViews
 * @return {void}
 */
OVR_multiview2.prototype.framebufferTextureMultiviewOVR = function(target, attachment, texture, level, baseViewIndex, numViews) {};

/**
 * @constructor OCULUS_multiview
 */
function OCULUS_multiview() {}

/**
 * @param {number} target
 * @param {number} attachment
 * @param {WebGLTexture} texture
 * @param {number} level
 * @param {number} baseViewIndex
 * @param {number} numViews
 * @return {void}
 */
OCULUS_multiview.prototype.framebufferTextureMultisampleMultiviewOVR = function(target, attachment, texture, level, samples, baseViewIndex, numViews) {};
