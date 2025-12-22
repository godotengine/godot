/**************************************************************************/
/*  register_types.cpp                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "register_types.h"

#include "visual_shader.h"
#include "vs_nodes/visual_shader_nodes.h"
#include "vs_nodes/visual_shader_particle_nodes.h"
#include "vs_nodes/visual_shader_sdf_nodes.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/visual_shader_editor_plugin.h"
#include "editor/visual_shader_language_plugin.h"

static void _editor_init() {
	Ref<EditorInspectorVisualShaderModePlugin> visual_shader_mode_plugin;
	visual_shader_mode_plugin.instantiate();
	EditorInspector::add_inspector_plugin(visual_shader_mode_plugin);

	Ref<VisualShaderConversionPlugin> visual_shader_convert;
	visual_shader_convert.instantiate();
	EditorNode::get_singleton()->add_resource_conversion_plugin(visual_shader_convert);

	Ref<VisualShaderLanguagePlugin> visual_shader_lang;
	visual_shader_lang.instantiate();
	EditorShaderLanguagePlugin::register_shader_language(visual_shader_lang);
}
#endif // TOOLS_ENABLED

void initialize_visual_shader_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(VisualShader);
		GDREGISTER_ABSTRACT_CLASS(VisualShaderNode);
		GDREGISTER_CLASS(VisualShaderNodeCustom);
		GDREGISTER_CLASS(VisualShaderNodeInput);
		GDREGISTER_ABSTRACT_CLASS(VisualShaderNodeOutput);
		GDREGISTER_ABSTRACT_CLASS(VisualShaderNodeResizableBase);
		GDREGISTER_ABSTRACT_CLASS(VisualShaderNodeGroupBase);
		GDREGISTER_ABSTRACT_CLASS(VisualShaderNodeConstant);
		GDREGISTER_ABSTRACT_CLASS(VisualShaderNodeVectorBase);
		GDREGISTER_CLASS(VisualShaderNodeFrame);
#ifndef DISABLE_DEPRECATED
		GDREGISTER_CLASS(VisualShaderNodeComment); // Deprecated, just for compatibility.
#endif
		GDREGISTER_CLASS(VisualShaderNodeFloatConstant);
		GDREGISTER_CLASS(VisualShaderNodeIntConstant);
		GDREGISTER_CLASS(VisualShaderNodeUIntConstant);
		GDREGISTER_CLASS(VisualShaderNodeBooleanConstant);
		GDREGISTER_CLASS(VisualShaderNodeColorConstant);
		GDREGISTER_CLASS(VisualShaderNodeVec2Constant);
		GDREGISTER_CLASS(VisualShaderNodeVec3Constant);
		GDREGISTER_CLASS(VisualShaderNodeVec4Constant);
		GDREGISTER_CLASS(VisualShaderNodeTransformConstant);
		GDREGISTER_CLASS(VisualShaderNodeFloatOp);
		GDREGISTER_CLASS(VisualShaderNodeIntOp);
		GDREGISTER_CLASS(VisualShaderNodeUIntOp);
		GDREGISTER_CLASS(VisualShaderNodeVectorOp);
		GDREGISTER_CLASS(VisualShaderNodeColorOp);
		GDREGISTER_CLASS(VisualShaderNodeTransformOp);
		GDREGISTER_CLASS(VisualShaderNodeTransformVecMult);
		GDREGISTER_CLASS(VisualShaderNodeFloatFunc);
		GDREGISTER_CLASS(VisualShaderNodeIntFunc);
		GDREGISTER_CLASS(VisualShaderNodeUIntFunc);
		GDREGISTER_CLASS(VisualShaderNodeVectorFunc);
		GDREGISTER_CLASS(VisualShaderNodeColorFunc);
		GDREGISTER_CLASS(VisualShaderNodeTransformFunc);
		GDREGISTER_CLASS(VisualShaderNodeUVFunc);
		GDREGISTER_CLASS(VisualShaderNodeUVPolarCoord);
		GDREGISTER_CLASS(VisualShaderNodeDotProduct);
		GDREGISTER_CLASS(VisualShaderNodeVectorLen);
		GDREGISTER_CLASS(VisualShaderNodeDeterminant);
		GDREGISTER_CLASS(VisualShaderNodeDerivativeFunc);
		GDREGISTER_CLASS(VisualShaderNodeClamp);
		GDREGISTER_CLASS(VisualShaderNodeFaceForward);
		GDREGISTER_CLASS(VisualShaderNodeOuterProduct);
		GDREGISTER_CLASS(VisualShaderNodeSmoothStep);
		GDREGISTER_CLASS(VisualShaderNodeStep);
		GDREGISTER_CLASS(VisualShaderNodeVectorDistance);
		GDREGISTER_CLASS(VisualShaderNodeVectorRefract);
		GDREGISTER_CLASS(VisualShaderNodeMix);
		GDREGISTER_CLASS(VisualShaderNodeVectorCompose);
		GDREGISTER_CLASS(VisualShaderNodeTransformCompose);
		GDREGISTER_CLASS(VisualShaderNodeVectorDecompose);
		GDREGISTER_CLASS(VisualShaderNodeTransformDecompose);
		GDREGISTER_CLASS(VisualShaderNodeTexture);
		GDREGISTER_CLASS(VisualShaderNodeCurveTexture);
		GDREGISTER_CLASS(VisualShaderNodeCurveXYZTexture);
		GDREGISTER_ABSTRACT_CLASS(VisualShaderNodeSample3D);
		GDREGISTER_CLASS(VisualShaderNodeTexture2DArray);
		GDREGISTER_CLASS(VisualShaderNodeTexture3D);
		GDREGISTER_CLASS(VisualShaderNodeCubemap);
		GDREGISTER_ABSTRACT_CLASS(VisualShaderNodeParameter);
		GDREGISTER_CLASS(VisualShaderNodeParameterRef);
		GDREGISTER_CLASS(VisualShaderNodeFloatParameter);
		GDREGISTER_CLASS(VisualShaderNodeIntParameter);
		GDREGISTER_CLASS(VisualShaderNodeUIntParameter);
		GDREGISTER_CLASS(VisualShaderNodeBooleanParameter);
		GDREGISTER_CLASS(VisualShaderNodeColorParameter);
		GDREGISTER_CLASS(VisualShaderNodeVec2Parameter);
		GDREGISTER_CLASS(VisualShaderNodeVec3Parameter);
		GDREGISTER_CLASS(VisualShaderNodeVec4Parameter);
		GDREGISTER_CLASS(VisualShaderNodeTransformParameter);
		GDREGISTER_ABSTRACT_CLASS(VisualShaderNodeTextureParameter);
		GDREGISTER_CLASS(VisualShaderNodeTexture2DParameter);
		GDREGISTER_CLASS(VisualShaderNodeTextureParameterTriplanar);
		GDREGISTER_CLASS(VisualShaderNodeTexture2DArrayParameter);
		GDREGISTER_CLASS(VisualShaderNodeTexture3DParameter);
		GDREGISTER_CLASS(VisualShaderNodeCubemapParameter);
		GDREGISTER_CLASS(VisualShaderNodeLinearSceneDepth);
		GDREGISTER_CLASS(VisualShaderNodeWorldPositionFromDepth);
		GDREGISTER_CLASS(VisualShaderNodeScreenNormalWorldSpace);
		GDREGISTER_CLASS(VisualShaderNodeIf);
		GDREGISTER_CLASS(VisualShaderNodeSwitch);
		GDREGISTER_CLASS(VisualShaderNodeFresnel);
		GDREGISTER_CLASS(VisualShaderNodeExpression);
		GDREGISTER_CLASS(VisualShaderNodeGlobalExpression);
		GDREGISTER_CLASS(VisualShaderNodeIs);
		GDREGISTER_CLASS(VisualShaderNodeCompare);
		GDREGISTER_CLASS(VisualShaderNodeMultiplyAdd);
		GDREGISTER_CLASS(VisualShaderNodeBillboard);
		GDREGISTER_CLASS(VisualShaderNodeDistanceFade);
		GDREGISTER_CLASS(VisualShaderNodeProximityFade);
		GDREGISTER_CLASS(VisualShaderNodeRandomRange);
		GDREGISTER_CLASS(VisualShaderNodeRemap);
		GDREGISTER_CLASS(VisualShaderNodeRotationByAxis);
		GDREGISTER_ABSTRACT_CLASS(VisualShaderNodeVarying);
		GDREGISTER_CLASS(VisualShaderNodeVaryingSetter);
		GDREGISTER_CLASS(VisualShaderNodeVaryingGetter);
		GDREGISTER_CLASS(VisualShaderNodeReroute);

		GDREGISTER_CLASS(VisualShaderNodeSDFToScreenUV);
		GDREGISTER_CLASS(VisualShaderNodeScreenUVToSDF);
		GDREGISTER_CLASS(VisualShaderNodeTextureSDF);
		GDREGISTER_CLASS(VisualShaderNodeTextureSDFNormal);
		GDREGISTER_CLASS(VisualShaderNodeSDFRaymarch);

		GDREGISTER_CLASS(VisualShaderNodeParticleOutput);
		GDREGISTER_ABSTRACT_CLASS(VisualShaderNodeParticleEmitter);
		GDREGISTER_CLASS(VisualShaderNodeParticleSphereEmitter);
		GDREGISTER_CLASS(VisualShaderNodeParticleBoxEmitter);
		GDREGISTER_CLASS(VisualShaderNodeParticleRingEmitter);
		GDREGISTER_CLASS(VisualShaderNodeParticleMeshEmitter);
		GDREGISTER_CLASS(VisualShaderNodeParticleMultiplyByAxisAngle);
		GDREGISTER_CLASS(VisualShaderNodeParticleConeVelocity);
		GDREGISTER_CLASS(VisualShaderNodeParticleRandomness);
		GDREGISTER_CLASS(VisualShaderNodeParticleAccelerator);
		GDREGISTER_CLASS(VisualShaderNodeParticleEmit);
#ifdef TOOLS_ENABLED
	} else if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		EditorNode::add_init_callback(_editor_init);
#endif // TOOLS_ENABLED
	}
}

void uninitialize_visual_shader_module(ModuleInitializationLevel p_level) {}
