/*************************************************************************/
/*  GodotGestureHandler.java                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

package org.godotengine.godot.input;

import android.view.GestureDetector;
import android.view.MotionEvent;
import org.godotengine.godot.GodotLib;
import org.godotengine.godot.GodotView;

/**
 * Handles gesture input related events for the {@link GodotView} view.
 * https://developer.android.com/reference/android/view/GestureDetector.SimpleOnGestureListener
 */
public class GodotGestureHandler extends GestureDetector.SimpleOnGestureListener {

	private final GodotView godotView;

	public GodotGestureHandler(GodotView godotView) {
		this.godotView = godotView;
	}

	private void queueEvent(Runnable task) {
		godotView.queueEvent(task);
	}

	@Override
	public boolean onDoubleTap(final MotionEvent event) {
		final int toolType = event.getToolType(0);
		final int buttonState = event.getButtonState();
		queueEvent(new Runnable() {
			@Override
			public void run() {
				GodotLib.doubleTap(toolType, buttonState, event.getX(), event.getY());
			}
		});
		return true;
	}

	@Override
	public boolean onScroll(MotionEvent e1, MotionEvent e2, final float distanceX, final float distanceY) {
		final int toolType = e1.getToolType(0);
		final float startX = e1.getX();
		final float startY = e1.getY();
		final float endX = e2.getX();
		final float endY = e2.getY();
		queueEvent(new Runnable() {
			@Override
			public void run() {
				GodotLib.scroll(toolType, startX, startY, endX, endY, distanceX, distanceY);
			}
		});
		return true;
	}
}
