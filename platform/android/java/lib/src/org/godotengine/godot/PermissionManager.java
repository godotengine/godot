package org.godotengine.godot;

import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.support.v4.content.ContextCompat;
import android.util.SparseArray;
import java.util.ArrayList;
import java.util.List;

public final class PermissionManager {

	private static final int REQUEST_ALL_PERMISSION_REQ_CODE = 1001;
	private static final String SDK_VERSION_IS_LESS_THAN_23 = "Android SDK version is less than 23";
	private static final String PACKAGE_MANAGER_NAME_NOT_FOUND_EXCEPTION = "PackageManager.NameNotFoundException";
	private static final String ANY_PERMISSIONS_DOES_NOT_DEFINE_IN_ANDROID_MANIFEST_XML = "Any permissions does not define in AndroidManifest.xml";
	private static final String UNDEFINED_PERMISSION = "You want to request permission[%s] which does not exist in AndroidManifest.xml";

	private static PermissionManager sInstance;
	private Godot mGodot;
	private SparseArray<String> permissionList = new SparseArray<String>() {
		{
			append(0, "android.permission.ACCESS_CHECKIN_PROPERTIES");
			append(1, "android.permission.ACCESS_COARSE_LOCATION");
			append(2, "android.permission.ACCESS_FINE_LOCATION");
			append(3, "android.permission.ACCESS_LOCATION_EXTRA_COMMANDS");
			append(4, "android.permission.ACCESS_MOCK_LOCATION");
			append(5, "android.permission.ACCESS_NETWORK_STATE");
			append(6, "android.permission.ACCESS_SURFACE_FLINGER");
			append(7, "android.permission.ACCESS_WIFI_STATE");
			append(8, "android.permission.ACCOUNT_MANAGER");
			append(9, "android.permission.ADD_VOICEMAIL");
			append(10, "android.permission.AUTHENTICATE_ACCOUNTS");
			append(11, "android.permission.BATTERY_STATS");
			append(12, "android.permission.BIND_ACCESSIBILITY_SERVICE");
			append(13, "android.permission.BIND_APPWIDGET");
			append(14, "android.permission.BIND_DEVICE_ADMIN");
			append(15, "android.permission.BIND_INPUT_METHOD");
			append(16, "android.permission.BIND_NFC_SERVICE");
			append(17, "android.permission.BIND_NOTIFICATION_LISTENER_SERVICE");
			append(18, "android.permission.BIND_PRINT_SERVICE");
			append(19, "android.permission.BIND_REMOTEVIEWS");
			append(20, "android.permission.BIND_TEXT_SERVICE");
			append(21, "android.permission.BIND_VPN_SERVICE");
			append(22, "android.permission.BIND_WALLPAPER");
			append(23, "android.permission.BLUETOOTH");
			append(24, "android.permission.BLUETOOTH_ADMIN");
			append(25, "android.permission.BLUETOOTH_PRIVILEGED");
			append(26, "android.permission.BRICK");
			append(27, "android.permission.BROADCAST_PACKAGE_REMOVED");
			append(28, "android.permission.BROADCAST_SMS");
			append(29, "android.permission.BROADCAST_STICKY");
			append(30, "android.permission.BROADCAST_WAP_PUSH");
			append(31, "android.permission.CALL_PHONE");
			append(32, "android.permission.CALL_PRIVILEGED");
			append(33, "android.permission.CAMERA");
			append(34, "android.permission.CAPTURE_AUDIO_OUTPUT");
			append(35, "android.permission.CAPTURE_SECURE_VIDEO_OUTPUT");
			append(36, "android.permission.CAPTURE_VIDEO_OUTPUT");
			append(37, "android.permission.CHANGE_COMPONENT_ENABLED_STATE");
			append(38, "android.permission.CHANGE_CONFIGURATION");
			append(39, "android.permission.CHANGE_NETWORK_STATE");
			append(40, "android.permission.CHANGE_WIFI_MULTICAST_STATE");
			append(41, "android.permission.CHANGE_WIFI_STATE");
			append(42, "android.permission.CLEAR_APP_CACHE");
			append(43, "android.permission.CLEAR_APP_USER_DATA");
			append(44, "android.permission.CONTROL_LOCATION_UPDATES");
			append(45, "android.permission.DELETE_CACHE_FILES");
			append(46, "android.permission.DELETE_PACKAGES");
			append(47, "android.permission.DEVICE_POWER");
			append(48, "android.permission.DIAGNOSTIC");
			append(49, "android.permission.DISABLE_KEYGUARD");
			append(50, "android.permission.DUMP");
			append(51, "android.permission.EXPAND_STATUS_BAR");
			append(52, "android.permission.FACTORY_TEST");
			append(53, "android.permission.FLASHLIGHT");
			append(54, "android.permission.FORCE_BACK");
			append(55, "android.permission.GET_ACCOUNTS");
			append(56, "android.permission.GET_PACKAGE_SIZE");
			append(57, "android.permission.GET_TASKS");
			append(58, "android.permission.GET_TOP_ACTIVITY_INFO");
			append(59, "android.permission.GLOBAL_SEARCH");
			append(60, "android.permission.HARDWARE_TEST");
			append(61, "android.permission.INJECT_EVENTS");
			append(62, "android.permission.INSTALL_LOCATION_PROVIDER");
			append(63, "android.permission.INSTALL_PACKAGES");
			append(64, "android.permission.INSTALL_SHORTCUT");
			append(65, "android.permission.INTERNAL_SYSTEM_WINDOW");
			append(66, "android.permission.INTERNET");
			append(67, "android.permission.KILL_BACKGROUND_PROCESSES");
			append(68, "android.permission.LOCATION_HARDWARE");
			append(69, "android.permission.MANAGE_ACCOUNTS");
			append(70, "android.permission.MANAGE_APP_TOKENS");
			append(71, "android.permission.MANAGE_DOCUMENTS");
			append(72, "android.permission.MASTER_CLEAR");
			append(73, "android.permission.MEDIA_CONTENT_CONTROL");
			append(74, "android.permission.MODIFY_AUDIO_SETTINGS");
			append(75, "android.permission.MODIFY_PHONE_STATE");
			append(76, "android.permission.MOUNT_FORMAT_FILESYSTEMS");
			append(77, "android.permission.MOUNT_UNMOUNT_FILESYSTEMS");
			append(78, "android.permission.NFC");
			append(79, "android.permission.PERSISTENT_ACTIVITY");
			append(80, "android.permission.PROCESS_OUTGOING_CALLS");
			append(81, "android.permission.READ_CALENDAR");
			append(82, "android.permission.READ_CALL_LOG");
			append(83, "android.permission.READ_CONTACTS");
			append(84, "android.permission.READ_EXTERNAL_STORAGE");
			append(85, "android.permission.READ_FRAME_BUFFER");
			append(86, "android.permission.READ_HISTORY_BOOKMARKS");
			append(87, "android.permission.READ_INPUT_STATE");
			append(88, "android.permission.READ_LOGS");
			append(89, "android.permission.READ_PHONE_STATE");
			append(90, "android.permission.READ_PROFILE");
			append(91, "android.permission.READ_SMS");
			append(92, "android.permission.READ_SOCIAL_STREAM");
			append(93, "android.permission.READ_SYNC_SETTINGS");
			append(94, "android.permission.READ_SYNC_STATS");
			append(95, "android.permission.READ_USER_DICTIONARY");
			append(96, "android.permission.REBOOT");
			append(97, "android.permission.RECEIVE_BOOT_COMPLETED");
			append(98, "android.permission.RECEIVE_MMS");
			append(99, "android.permission.RECEIVE_SMS");
			append(100, "android.permission.RECEIVE_WAP_PUSH");
			append(101, "android.permission.RECORD_AUDIO");
			append(102, "android.permission.REORDER_TASKS");
			append(103, "android.permission.RESTART_PACKAGES");
			append(104, "android.permission.SEND_RESPOND_VIA_MESSAGE");
			append(105, "android.permission.SEND_SMS");
			append(106, "android.permission.SET_ACTIVITY_WATCHER");
			append(107, "android.permission.SET_ALARM");
			append(108, "android.permission.SET_ALWAYS_FINISH");
			append(109, "android.permission.SET_ANIMATION_SCALE");
			append(110, "android.permission.SET_DEBUG_APP");
			append(111, "android.permission.SET_ORIENTATION");
			append(112, "android.permission.SET_POINTER_SPEED");
			append(113, "android.permission.SET_PREFERRED_APPLICATIONS");
			append(114, "android.permission.SET_PROCESS_LIMIT");
			append(115, "android.permission.SET_TIME");
			append(116, "android.permission.SET_TIME_ZONE");
			append(117, "android.permission.SET_WALLPAPER");
			append(118, "android.permission.SET_WALLPAPER_HINTS");
			append(119, "android.permission.SIGNAL_PERSISTENT_PROCESSES");
			append(120, "android.permission.STATUS_BAR");
			append(121, "android.permission.SUBSCRIBED_FEEDS_READ");
			append(122, "android.permission.SUBSCRIBED_FEEDS_WRITE");
			append(123, "android.permission.SYSTEM_ALERT_WINDOW");
			append(124, "android.permission.TRANSMIT_IR");
			append(125, "android.permission.UNINSTALL_SHORTCUT");
			append(126, "android.permission.UPDATE_DEVICE_STATS");
			append(127, "android.permission.USE_CREDENTIALS");
			append(128, "android.permission.USE_SIP");
			append(129, "android.permission.VIBRATE");
			append(130, "android.permission.WAKE_LOCK");
			append(131, "android.permission.WRITE_APN_SETTINGS");
			append(132, "android.permission.WRITE_CALENDAR");
			append(133, "android.permission.WRITE_CALL_LOG");
			append(134, "android.permission.WRITE_CONTACTS");
			append(135, "android.permission.WRITE_EXTERNAL_STORAGE");
			append(136, "android.permission.WRITE_GSERVICES");
			append(137, "android.permission.WRITE_HISTORY_BOOKMARKS");
			append(138, "android.permission.WRITE_PROFILE");
			append(139, "android.permission.WRITE_SECURE_SETTINGS");
			append(140, "android.permission.WRITE_SETTINGS");
			append(141, "android.permission.WRITE_SMS");
			append(142, "android.permission.WRITE_SOCIAL_STREAM");
			append(143, "android.permission.WRITE_SYNC_SETTINGS");
			append(144, "android.permission.WRITE_USER_DICTIONARY");
		}
	};

	private PermissionManager(Godot godot) {
		this.mGodot = godot;
	}

	static PermissionManager getInstance(Godot godot) {
		if (sInstance == null)
			sInstance = new PermissionManager(godot);
		return sInstance;
	}

	RequestInfo requestPermission(int p_permission) {
		if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
			return new RequestInfo(true, SDK_VERSION_IS_LESS_THAN_23);
		}

		PackageInfo packageInfo;
		try {
			packageInfo = mGodot.getPackageManager().getPackageInfo(mGodot.getPackageName(),
					PackageManager.GET_PERMISSIONS);
		} catch (PackageManager.NameNotFoundException e) {
			e.printStackTrace();
			return new RequestInfo(false, PACKAGE_MANAGER_NAME_NOT_FOUND_EXCEPTION);
		}

		String[] manifestPermissions = packageInfo.requestedPermissions;
		if (manifestPermissions == null || manifestPermissions.length == 0)
			return new RequestInfo(false, ANY_PERMISSIONS_DOES_NOT_DEFINE_IN_ANDROID_MANIFEST_XML);

		String requested = permissionList.get(p_permission);
		int i = 0;
		boolean found = false;
		while (i < manifestPermissions.length) {
			if (manifestPermissions[i].equals(requested)) {
				found = true;
				i = manifestPermissions.length;
			}
			i++;
		}

		if (!found)
			return new RequestInfo(false, String.format(UNDEFINED_PERMISSION,
												  (requested != null ? requested : "unknown")));

		if (ContextCompat.checkSelfPermission(mGodot, requested) != PackageManager.PERMISSION_GRANTED) {
			mGodot.requestPermissions(new String[] { requested }, REQUEST_ALL_PERMISSION_REQ_CODE);
			return new RequestInfo(false, "Permission[" + requested + "] was requested");
		}

		return new RequestInfo(true, "Permission[" + requested + "] was granted");
	}

	RequestInfo requestPermissions(int[] p_permissions) {
		if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
			return new RequestInfo(true, SDK_VERSION_IS_LESS_THAN_23);
		}

		PackageInfo packageInfo;
		try {
			packageInfo = mGodot.getPackageManager().getPackageInfo(mGodot.getPackageName(),
					PackageManager.GET_PERMISSIONS);
		} catch (PackageManager.NameNotFoundException e) {
			e.printStackTrace();
			return new RequestInfo(false, PACKAGE_MANAGER_NAME_NOT_FOUND_EXCEPTION);
		}

		String[] manifestPermissions = packageInfo.requestedPermissions;
		if (manifestPermissions == null || manifestPermissions.length == 0)
			return new RequestInfo(false, ANY_PERMISSIONS_DOES_NOT_DEFINE_IN_ANDROID_MANIFEST_XML);

		List<String> notGrantedPermissionList = new ArrayList<>();
		for (String manifestPermission : manifestPermissions) {
			int i = 0;
			while (i < p_permissions.length) {
				String requested = permissionList.get(p_permissions[i]);
				if (manifestPermission.equals(requested)) {
					boolean granted = ContextCompat.checkSelfPermission(mGodot, requested) == PackageManager.PERMISSION_GRANTED;
					if (!granted)
						notGrantedPermissionList.add(requested);
					i = p_permissions.length;
				}
				i++;
			}
		}

		if (notGrantedPermissionList.size() == 0)
			return new RequestInfo(true, "All permissions were granted");

		String[] requestArray = new String[notGrantedPermissionList.size()];
		notGrantedPermissionList.toArray(requestArray);
		mGodot.requestPermissions(requestArray, REQUEST_ALL_PERMISSION_REQ_CODE);
		return new RequestInfo(false, "All permissions were requested");
	}

	RequestInfo checkPermissionStatus(int p_index) {
		if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
			return new RequestInfo(true, SDK_VERSION_IS_LESS_THAN_23);
		}
		String requested = permissionList.get(p_index);
		if (requested != null) {
			boolean granted = ContextCompat.checkSelfPermission(mGodot, requested) == PackageManager.PERMISSION_GRANTED;
			return new RequestInfo(granted, "Permission[" + requested + "] was" + (!granted ? " not " : " ") + "granted");
		}
		return new RequestInfo(false, String.format(UNDEFINED_PERMISSION, "unknown"));
	}

	SparseArray<String> getPermissionList() {
		return permissionList;
	}

	public static class RequestInfo {
		private boolean granted;
		private String resultText;

		public RequestInfo(boolean granted, String resultText) {
			this.granted = granted;
			this.resultText = resultText;
		}

		public boolean isGranted() {
			return granted;
		}

		public String getResultText() {
			return resultText;
		}
	}
}
