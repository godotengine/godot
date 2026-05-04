package com.godot.game.test

import android.util.Log
import org.godotengine.godot.Dictionary
import org.godotengine.godot.Godot
import org.godotengine.godot.plugin.GodotPlugin
import org.godotengine.godot.plugin.SignalInfo
import org.godotengine.godot.plugin.UsedByGodot

class SignalTestPlugin(godot: Godot) : GodotPlugin(godot) {

	companion object {
		private val EMISSION_TEST_SIGNAL = SignalInfo("emission_test_signal")
		private val LAUNCH_TESTS_SIGNAL = SignalInfo("launch_tests", java.lang.Boolean::class.java, String::class.java)
	}

	override fun getPluginName() = "SignalTestPlugin"

	override fun getPluginSignals(): Set<SignalInfo?> {
		return setOf(
			EMISSION_TEST_SIGNAL,
			LAUNCH_TESTS_SIGNAL
		)
	}

	@UsedByGodot
	fun triggerTestSignal1() {
		emitSignal(EMISSION_TEST_SIGNAL)
	}

	@UsedByGodot
	fun triggerLaunchTestSignal() {
		emitSignal(LAUNCH_TESTS_SIGNAL, true, "second message")
	}
}
