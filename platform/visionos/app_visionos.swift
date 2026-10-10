/**************************************************************************/
/*  app_visionos.swift                                                    */
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

import SwiftUI
@preconcurrency import CompositorServices
import OSLog

// MARK: Helpers

extension os.Logger {
	static let godot = Logger(subsystem: "com.GodotFoundation.Godot", category: "SwiftUI")
}

// MARK: Renderer

final class RendererTaskExecutor: TaskExecutor {
	private let queue = DispatchQueue(label: "RenderThreadQueue", qos: .userInteractive)
	func enqueue(_ job: UnownedJob) {
		queue.async {
		    job.runSynchronously(on: self.asUnownedSerialExecutor())
		}
	}
	nonisolated func asUnownedSerialExecutor() -> UnownedTaskExecutor {
		return UnownedTaskExecutor(ordinary: self)
	}
	static let shared: RendererTaskExecutor = RendererTaskExecutor()
}

// MARK: Swift Bridge

/// Source of truth for SwiftUI scene state. ObjC/C++ mutates it through
/// `GDTSwiftBridge`; the scene reads its properties directly (Observation tracking).
@MainActor
@Observable
final class Model {
	static let shared = Model()

	var immersionStyle: any ImmersionStyle
	var upperLimbVisibility: Visibility = .automatic
	var persistentSystemOverlays: Visibility = .automatic

	private init() {
		immersionStyle = Self.readInitialImmersionStyleFromInfoPlist()
	}

	/// Seeds the project-setting-backed properties. Called at layer creation rather than
	/// from `init()`, because `ProjectSettings` is not loaded yet when the scene is declared.
	func seedFromProjectSettings() {
		upperLimbVisibility = GDTAppDelegateServiceVisionOS.initialUpperLimbVisibility.swiftUI
		persistentSystemOverlays = GDTAppDelegateServiceVisionOS.initialPersistentSystemOverlays.swiftUI
	}

	private static func readInitialImmersionStyleFromInfoPlist() -> any ImmersionStyle {
		guard let sceneManifest = Bundle.main.infoDictionary?["UIApplicationSceneManifest"] as? [String: Any],
		      let sceneConfigurations = sceneManifest["UISceneConfigurations"] as? [String: Any],
		      let cpSceneConfiguration = sceneConfigurations["UISceneSessionRoleImmersiveSpaceApplication"] as? [[String: Any]],
		      let immersionStyleString = cpSceneConfiguration.first?["UISceneInitialImmersionStyle"] as? String else {
			return .full
		}
		switch immersionStyleString {
		case "UIImmersionStyleFull": return .full
		case "UIImmersionStyleMixed": return .mixed
		case "UIImmersionStyleProgressive": return .progressive
		default: return .full
		}
	}
}

/// ObjC-accessible interface for `Model`.
@MainActor
@objc
public final class GDTSwiftBridge: NSObject {
	@objc public class var immersionStyle: GDTImmersionStyle {
		get { GDTImmersionStyle(fromSwiftUIType: Model.shared.immersionStyle) }
		set {
			guard let swiftUIStyle = newValue.swiftUI else { return }
			Model.shared.immersionStyle = swiftUIStyle
		}
	}

	@objc public class var upperLimbVisibility: GDTVisibility {
		get { GDTVisibility(fromSwiftUIType: Model.shared.upperLimbVisibility) }
		set { Model.shared.upperLimbVisibility = newValue.swiftUI }
	}

	@objc public class var persistentSystemOverlays: GDTVisibility {
		get { GDTVisibility(fromSwiftUIType: Model.shared.persistentSystemOverlays) }
		set { Model.shared.persistentSystemOverlays = newValue.swiftUI }
	}
}

/// ObjC-accessible version of SwiftUI's "SpatialEventCollection.Event".
/// https://developer.apple.com/documentation/swiftui/spatialeventcollection/event
@MainActor
@objc
public final class SpatialEventObjC: NSObject {

    // Ray
    @objc var hasRay: Bool = false
    @objc var rayOrigin: simd_double3 = .init()
    @objc var rayDirection: simd_double3  = .init()

    // Hand
    @objc enum Chirality: Int {
        case none, left, right
    }
    @objc var chirality: Chirality
    @objc var handPose: simd_double4x4

    // Phase
    @objc enum Phase: Int {
        case unknown, active, cancelled, ended
    }
    @objc var phase: Phase

    init(_ event: SpatialEventCollection.Event) {
        if let ray = event.selectionRay {
            hasRay = true
            rayOrigin = ray.origin.vector
            rayDirection = ray.direction.vector
        }

        chirality = event.chirality.map {
            switch $0 {
            case .left: return .left
            case .right: return .right
            }
        } ?? .none

        handPose = event.inputDevicePose.map {
            return $0.pose3D.matrix
        } ?? .init(diagonal: SIMD4<Double>(repeating: 1.0))

        phase = {
            switch event.phase {
            case .active: return .active
            case .cancelled: return .cancelled
            case .ended: return .ended
            default: return .unknown
            }
        }()
    }
}

// MARK: Compositor Services Scene

struct ContentStageConfiguration: CompositorLayerConfiguration {
	func makeConfiguration(capabilities: LayerRenderer.Capabilities, configuration: inout LayerRenderer.Configuration) {

		GDTAppDelegateServiceVisionOS.layerRendererCapabilities = capabilities as __CP_OBJECT_cp_layer_renderer_capabilities

		configuration.depthFormat = .depth32Float_stencil8
		configuration.colorFormat = .rgba16Float

		let foveationEnabled = capabilities.supportsFoveation
		configuration.isFoveationEnabled = foveationEnabled

		let options: LayerRenderer.Capabilities.SupportedLayoutsOptions = foveationEnabled ? [.foveationEnabled] : []
		let supportedLayouts = capabilities.supportedLayouts(options: options)
		if (!supportedLayouts.contains(.layered)) {
			fatalError("Only the .layered layout is supported by Godot's visionOS XR module.")
		}
		configuration.layout = .layered

		if GDTAppDelegateServiceVisionOS.isDynamicRenderQualityEnabled {
			let maxRenderQuality = GDTAppDelegateServiceVisionOS.maxRenderQuality
			Logger.godot.log("Enabled dynamic render quality (maxRenderQuality: \(maxRenderQuality))")
			configuration.maxRenderQuality = .init(maxRenderQuality)
		}
	}
}

extension GDTCompositorServicesRenderer: @unchecked Sendable {}

extension GDTImmersionStyle {

    var swiftUI: ImmersionStyle? {
        switch self {
        case .full: return .full
        case .mixed: return .mixed
        case .progressive: return .progressive
        @unknown default: return nil
        }
    }

    init(fromSwiftUIType swiftUIType: ImmersionStyle) {
    switch swiftUIType.self {
        case is FullImmersionStyle: self = .full
        case is MixedImmersionStyle: self = .mixed
        case is ProgressiveImmersionStyle: self = .progressive
        default: fatalError("Unsupported style")
        }
    }

}

extension GDTVisibility {
	var swiftUI: Visibility {
		switch self {
		case .automatic: return .automatic
		case .visible: return .visible
		case .hidden: return .hidden
		@unknown default: return .automatic
		}
	}

	init(fromSwiftUIType visibility: Visibility) {
		switch visibility {
		case .automatic: self = .automatic
		case .visible: self = .visible
		case .hidden: self = .hidden
		}
	}
}

struct CompositorServicesImmersiveSpace: Scene {

    let model: Model = .shared

    @State var renderer: GDTCompositorServicesRenderer!
    @State var didSetUpRenderer: Bool = false

	var body: some Scene {
		ImmersiveSpace(id: "ImmersiveSpace") {
			CompositorLayer(configuration: ContentStageConfiguration()) { @MainActor layerRenderer in

                Logger.godot.log("CompositorLayer init (initialImmersionStyle: \(String(describing: model.immersionStyle)))")

                model.seedFromProjectSettings()

				GDTAppDelegateServiceVisionOS.layerRenderer = layerRenderer
				renderer = GDTCompositorServicesRenderer(layerRenderer: layerRenderer,
                                                         capabilities: GDTAppDelegateServiceVisionOS.layerRendererCapabilities)

                layerRenderer.onSpatialEvent = { events in
                    for event in events {
                        renderer.onSpatialEvent(.init(event))
                    }
                }

                let signposter = OSSignposter(subsystem: "org.godotengine.godot.compositorservices", category: "loading")
                let signpostID = signposter.makeSignpostID()

                if !didSetUpRenderer {
                    let signpost = signposter.beginInterval("setup", id: signpostID)
                    renderer.setUp()
                    didSetUpRenderer = true
                    signposter.endInterval("setup", signpost)
                } else {
                    let signpost = signposter.beginInterval("updateXRInterface", id: signpostID)
                    renderer.updateXRInterface()
                    signposter.endInterval("updateXRInterface", signpost)
                }
				Task(executorPreference: RendererTaskExecutor.shared) {
                    let signpost = signposter.beginInterval("startRenderLoop", id: signpostID)
					await renderer.startRenderLoop()
                    signposter.endInterval("startRenderLoop", signpost)
				}
			}
			.onWorldRecenter {
				renderer.worldRecentered()
			}
		}
		.immersionStyle(
			selection: Binding(get: { model.immersionStyle }, set: { model.immersionStyle = $0 }),
			in: .mixed, .full, .progressive
		)
        .upperLimbVisibility(model.upperLimbVisibility)
        .persistentSystemOverlays(model.persistentSystemOverlays)
	}
}

// MARK: App

@main
struct SwiftUIApp: App {
	@UIApplicationDelegateAdaptor(GDTAppDelegateVisionOS.self) var appDelegate

	private var useCompositorServices: Bool = {
		guard let sceneManifest = Bundle.main.infoDictionary?["UIApplicationSceneManifest"] as? [String: Any],
			  let defaultSessionRole = sceneManifest["UIApplicationPreferredDefaultSceneSessionRole"] as? String else {
			return false
		}
		return defaultSessionRole == "CPSceneSessionRoleImmersiveSpaceApplication"
	}()

	init() {
		let useCompositorServices = self.useCompositorServices
		Logger.godot.log("visionOS app init (useCompositorServices: \(useCompositorServices))")
		GDTAppDelegateServiceVisionOS.renderMode = useCompositorServices ? .compositorServices : .windowed
	}

	var body: some Scene {
		GodotWindowScene()
		CompositorServicesImmersiveSpace()
	}
}
