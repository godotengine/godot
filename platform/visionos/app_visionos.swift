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

// MARK: Compositor Services Scene

struct ContentStageConfiguration: CompositorLayerConfiguration {
	func makeConfiguration(capabilities: LayerRenderer.Capabilities, configuration: inout LayerRenderer.Configuration) {

		GDTAppDelegateServiceVisionOS.layerRendererCapabilities = capabilities as __CP_OBJECT_cp_layer_renderer_capabilities

		configuration.depthFormat = .depth32Float
		configuration.colorFormat = .rgba16Float

		let foveationEnabled = capabilities.supportsFoveation
		configuration.isFoveationEnabled = foveationEnabled

		let options: LayerRenderer.Capabilities.SupportedLayoutsOptions = foveationEnabled ? [.foveationEnabled] : []
		let supportedLayouts = capabilities.supportedLayouts(options: options)
		if (!supportedLayouts.contains(.layered)) {
			fatalError("Only the .layered layout is supported by Godot's visionOS XR module.")
		}
		configuration.layout = .layered
	}
}

extension GDTCompositorServicesRenderer: @unchecked Sendable {}

struct CompositorServicesImmersiveSpace: Scene {

	fileprivate static var initialImmersionStyle: ImmersionStyle {
		guard let sceneManifest = Bundle.main.infoDictionary?["UIApplicationSceneManifest"] as? [String: Any],
			  let sceneConfigurations = sceneManifest["UISceneConfigurations"] as? [String: Any],
			  let cpSceneConfiguration = sceneConfigurations["UISceneSessionRoleImmersiveSpaceApplication"] as? [[String: Any]],
			  let immersionStyleString = cpSceneConfiguration.first?["UISceneInitialImmersionStyle"] as? String  else {
			return .full
		}
		switch immersionStyleString {
			case "UIImmersionStyleFull": return .full
			case "UIImmersionStyleMixed": return .mixed
			default: return .full
		}
	}

	@State var renderer: GDTCompositorServicesRenderer!

	var body: some Scene {
		ImmersiveSpace(id: "ImmersiveSpace") {
			CompositorLayer(configuration: ContentStageConfiguration()) { @MainActor layerRenderer in
				GDTAppDelegateServiceVisionOS.layerRenderer = layerRenderer
				renderer = GDTCompositorServicesRenderer(layerRenderer: layerRenderer)
				renderer.setUp()
				Task(executorPreference: RendererTaskExecutor.shared) {
					await renderer.startRenderLoop()
				}
			}
			.onWorldRecenter {
				renderer.worldRecentered()
			}
		}
		.immersionStyle(selection: .constant(Self.initialImmersionStyle), in: .mixed, .full)
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
		print("visionOS app init (useCompositorServices: \(useCompositorServices))")
		GDTAppDelegateServiceVisionOS.renderMode = useCompositorServices ? .compositorServices : .windowed
	}

	var body: some Scene {
		GodotWindowScene()
		CompositorServicesImmersiveSpace()
	}
}
