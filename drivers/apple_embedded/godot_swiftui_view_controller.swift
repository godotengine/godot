/**************************************************************************/
/*  godot_swiftui_view_controller.swift                                   */
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
import UIKit

struct GodotSwiftUIViewController: UIViewControllerRepresentable {

	func makeUIViewController(context: Context) -> GDTViewController {
		let viewController = GDTViewController()
		GDTAppDelegateService.viewController = viewController
		return viewController
	}

	func updateUIViewController(_ uiViewController: GDTViewController, context: Context) {
		// NOOP
	}

}

struct GodotWindowScene: Scene {
	@Environment(\.scenePhase) private var scenePhase

	// UIViewControllerRepresentable does not call viewWillDisappear() nor viewDidDisappear() when
	// backgrounding the app, or closing the app's main window, update the renderer here.
	private func updateRenderer(with newPhase: ScenePhase) -> Void {
		switch newPhase {
		case .active:
			print("GodotSwiftUIViewController scene active")
			GDTAppDelegateService.viewController?.godotView.startRendering()
		case .inactive:
			print("GodotSwiftUIViewController scene inactive")
			GDTAppDelegateService.viewController?.godotView.stopRendering()
		case .background:
			print("GodotSwiftUIViewController scene backgrounded")
			GDTAppDelegateService.viewController?.godotView.stopRendering()
		@unknown default:
			print("unknown default")
		}
	}

	var body: some Scene {
		WindowGroup {
			if #available(iOS 17, *) {
			GodotSwiftUIViewController()
				.ignoresSafeArea()
				.onChange(of: scenePhase) { _, newPhase in
					updateRenderer(with: newPhase)
				}
			} else {
				GodotSwiftUIViewController()
					.ignoresSafeArea()
				.onChange(of: scenePhase) { newPhase in
					updateRenderer(with: newPhase)
				}
			}
		}
	}
}
