def can_build(env, platform):
    if platform in ("linuxbsd", "windows", "android", "macos"):
        return not env["disable_xr"]
    else:
        # Not supported on these platforms.
        return False


def configure(env):
    pass


def get_doc_classes():
    return [
        "OpenXRInterface",
        "OpenXRAction",
        "OpenXRActionSet",
        "OpenXRActionMap",
        "OpenXRAPIExtension",
        "OpenXRExtensionWrapper",
        "OpenXRExtensionWrapperExtension",
        "OpenXRFrameSynthesisExtension",
        "OpenXRFutureResult",
        "OpenXRFutureExtension",
        "OpenXRInteractionProfile",
        "OpenXRInteractionProfileMetadata",
        "OpenXRIPBinding",
        "OpenXRHand",
        "OpenXRVisibilityMask",
        "OpenXRFoveatedInsetViewport",
        "OpenXRCompositionLayer",
        "OpenXRCompositionLayerQuad",
        "OpenXRCompositionLayerCylinder",
        "OpenXRCompositionLayerEquirect",
        "OpenXRBindingModifier",
        "OpenXRIPBindingModifier",
        "OpenXRActionBindingModifier",
        "OpenXRAnalogThresholdModifier",
        "OpenXRDpadBindingModifier",
        "OpenXRInteractionProfileEditorBase",
        "OpenXRInteractionProfileEditor",
        "OpenXRBindingModifierEditor",
        "OpenXRHapticBase",
        "OpenXRHapticVibration",
        "OpenXRRenderModelExtension",
        "OpenXRRenderModel",
        "OpenXRRenderModelManager",
        "OpenXRStructureBase",
        "OpenXRSpatialEntityExtension",
        "OpenXRSpatialEntityTracker",
        "OpenXRAnchorTracker",
        "OpenXRPlaneTracker",
        "OpenXRMarkerTracker",
        "OpenXRMeshTracker",
        "OpenXRSpatialCapabilityConfigurationBaseHeader",
        "OpenXRSpatialCapabilityConfigurationAnchor",
        "OpenXRSpatialCapabilityConfigurationQrCode",
        "OpenXRSpatialCapabilityConfigurationMicroQrCode",
        "OpenXRSpatialCapabilityConfigurationAruco",
        "OpenXRSpatialCapabilityConfigurationAprilTag",
        "OpenXRSpatialContextPersistenceConfig",
        "OpenXRSpatialCapabilityConfigurationPlaneTracking",
        "OpenXRSpatialContextWaterTightMeshConfig",
        "OpenXRSpatialCapabilityConfigurationMeshTracking",
        "OpenXRSpatialComponentData",
        "OpenXRSpatialComponentBounded2DList",
        "OpenXRSpatialComponentBounded3DList",
        "OpenXRSpatialComponentParentList",
        "OpenXRSpatialComponentMesh2DList",
        "OpenXRSpatialComponentMesh3DList",
        "OpenXRSpatialComponentPlaneAlignmentList",
        "OpenXRSpatialComponentPolygon2DList",
        "OpenXRSpatialComponentPlaneSemanticLabelList",
        "OpenXRSpatialComponentMarkerList",
        "OpenXRSpatialComponentMesh3DNormalsList",
        "OpenXRSpatialQueryResultData",
        "OpenXRSpatialComponentAnchorList",
        "OpenXRSpatialComponentPersistenceList",
        "OpenXRSpatialAnchorCapability",
        "OpenXRSpatialPlaneTrackingCapability",
        "OpenXRSpatialMarkerTrackingCapability",
        "OpenXRAndroidThreadSettingsExtension",
        "OpenXRUserPresenceExtension",
        "OpenXRSpatialContainerExtension",
        "OpenXRSpatialContainerSelfRenderingExtension",
        "OpenXRSpatialContainerState",
        "OpenXrSpatialFilterArchetype",
        "OpenXRSpatialCapabilityConfigurationImageTracking",
        "OpenXRSpatialComponentImage2DList",
        "OpenXRImageTracker",
        "OpenXRSpatialImageTrackingCapability",
        "OpenXRSpatialImageDatabase",
        "OpenXRSpatialMeshTrackingCapability",
    ]


def get_doc_path():
    return "doc_classes"
