package org.godotengine.godot

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.res.Configuration
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import androidx.annotation.CallSuper
import androidx.fragment.app.Fragment
import org.godotengine.godot.service.GodotService
import org.godotengine.godot.utils.beginBenchmarkMeasure
import org.godotengine.godot.utils.endBenchmarkMeasure

class LocalGodotFragment: Fragment(), GodotHost {
	companion object {
		private val TAG = LocalGodotFragment::class.java.simpleName
	}

	private var parentHost: GodotHost? = null
	private var localService: GodotService? = null

	private val serviceConnection = object: ServiceConnection {
		override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
			Log.d(TAG, "Connected to service $name")
			val godotServiceBinder = service as GodotService.ServiceBinder
			localService = godotServiceBinder.getLocalService()
			endBenchmarkMeasure("Startup", "GodotFragment::onCreate")
		}

		override fun onServiceDisconnected(name: ComponentName?) {
			Log.d(TAG, "Disconnected from service $name")
			localService = null
		}
	}

	override fun getGodot() = localService?.godot

	override fun onAttach(context: Context) {
		super.onAttach(context)
		if (parentFragment is GodotHost) {
			parentHost = parentFragment as GodotHost?
		} else if (activity is GodotHost) {
			parentHost = activity as GodotHost
		}
	}

	override fun onDetach() {
		super.onDetach()
		parentHost = null
	}

	override fun onCreate(savedInstanceState: Bundle?) {
		beginBenchmarkMeasure("Startup", "GodotFragment::onCreate")
		super.onCreate(savedInstanceState)

		startLocalService()
	}

	private fun startLocalService() {
		Log.d(TAG, "Binding to local GodotService")
		context?.bindService(
			Intent(context, GodotService::class.java),
			serviceConnection,
			Context.BIND_AUTO_CREATE
		)
	}

	private fun stopLocalService() {
		Log.d(TAG, "Unbinding from local GodotService")
		context?.unbindService(serviceConnection)
	}

	@CallSuper
	override fun onConfigurationChanged(newConfig: Configuration) {
		super.onConfigurationChanged(newConfig)
		godot?.onConfigurationChanged(newConfig)
	}

	@CallSuper
	override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
		super.onActivityResult(requestCode, resultCode, data)
		godot?.onActivityResult(requestCode, resultCode, data)
	}

	@CallSuper
	override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String?>, grantResults: IntArray) {
		super.onRequestPermissionsResult(requestCode, permissions, grantResults)
		godot?.onRequestPermissionsResult(requestCode, permissions, grantResults)
	}
}
